import logging
import sys

from pyspark import keyword_only
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import FloatType, ArrayType, DoubleType

from .estimators.model_evaluators import WildebeestDREstimationUtils
from .inputs.data_processing import (
    FeatureEngineering,
    DataPreprocessing,
)
from .model_utils import ModelBuilder
from .pipelines import ScoringPipeline
from .utilities.utils import (
    log_decorator,
    dataframe_reader,
    HasConfigParam,
    generate_paths,
    get_string_indexer_labels,
)
from pyspark.ml import PipelineModel
from pyspark.ml.feature import VectorAssembler

from pyspark.sql.functions import udf
from concurrent.futures import ThreadPoolExecutor

from pyspark.ml import Pipeline
import pyspark.sql.functions as f

# Set logger
logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s,%(msecs)d %(levelname)-8s \
            [%(filename)s:%(lineno)d] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger()


class BaseBuilder:
    def __init__(
        self,
        data_schema_location,
        raw_data_location,
        holidays_data_location,
        model_type="",
        do_bucketize=False,
        propensity_model_type="",
    ):
        self.data_preprocessor = DataPreprocessing()
        self.feature_engineer = None
        self.data_schema_location = data_schema_location
        self.raw_data_location = raw_data_location
        self.holidays_data_location = holidays_data_location
        self.model_type = model_type
        self.do_bucketize = do_bucketize
        self.propensity_model_type = propensity_model_type

    def build_data_schema(self):
        self.data_preprocessor.read_data_schema(self.data_schema_location)
        return self

    def build_data_preprocessor(self):
        self.data_preprocessor.read_holidays(self.holidays_data_location)
        self.data_preprocessor.read_raw_data(self.raw_data_location)
        return self

    def build_feature_engineer(self):
        self.feature_engineer = FeatureEngineering(
            self.data_preprocessor, self.model_type, self.do_bucketize, self.propensity_model_type
        )
        self.feature_engineer.create_featurization_pipeline()
        self.feature_engineer.create_featurized_data()
        return self

    def build(self):
        self.build_data_schema()
        self.build_data_preprocessor()
        self.build_feature_engineer()
        return self


class WildebeestTrainingExecutorBuilder(BaseBuilder):
    @keyword_only
    def __init__(
        self,
        data_schema_location,
        raw_data_location,
        holidays_data_location,
        s3_output_path,
        bucket,
        do_bucketize=False,
        model_type="regression",  # OLS or DRE
        propensity_model_type="LogisticRegression",  # RandomForest or LogisticRegression,
    ):
        BaseBuilder.__init__(
            self,
            data_schema_location,
            raw_data_location,
            holidays_data_location,
            model_type,
            do_bucketize,
        )
        self.s3_output_path = s3_output_path
        self.bucket = bucket
        self.model_trainer = None
        self.propensity_model_type = propensity_model_type

    def build_model_trainer(self):
        self.model_trainer: ModelBuilder = ModelBuilder()
        return self

    def build(self):
        super().build()
        self.build_model_trainer()
        return WildebeestTrainingExecutor(self)


class WildebeestTrainingExecutor:
    def __init__(self, builder: WildebeestTrainingExecutorBuilder):
        self.model_type: str = builder.model_type
        self.s3_output_path: str = builder.s3_output_path
        self.propensity_model_type: str = builder.propensity_model_type
        self.bucket = builder.bucket
        self.data_preprocessor: DataPreprocessing = builder.data_preprocessor
        self.feature_engineer: FeatureEngineering = builder.feature_engineer
        self.model_trainer = builder.model_trainer

    @log_decorator
    def run(self):
        featurization_pipeline = self.feature_engineer.featurization_pipeline
        processed_data = self.data_preprocessor.get_processed_data
        sampled_process_data = processed_data.sample(withReplacement=False, fraction=0.1, seed=1234)
        if self.model_type == "regression":
            (
                wildebeest_purchase_regression_model,
                wildebeest_purchase_regression_model_fitted,
            ) = self._train_regression_model()
            self._create_and_persist_scoring_pipeline(
                featurization_pipeline=featurization_pipeline,
                model=wildebeest_purchase_regression_model_fitted,
                sampled_process_data=sampled_process_data,
            )
        else:
            wildebeest_dre_model, wildebeest_dre_model_fitted = self._train_doubly_robust_model()
            self._create_and_persist_scoring_pipeline(
                featurization_pipeline=featurization_pipeline,
                model=wildebeest_dre_model,
                sampled_process_data=sampled_process_data,
            )

    def _train_regression_model(self):
        wildebeest_purchase_regression_model = self.model_trainer.build_linear_model(
            self.feature_engineer
        )
        regression_fitted_model = wildebeest_purchase_regression_model.fit(
            dataset=self.feature_engineer.featurized_data
        )
        return wildebeest_purchase_regression_model, regression_fitted_model

    def _train_doubly_robust_model(self):
        wildebeest_dre_model = self.model_trainer.build_doubly_robust_model(
            features_list=self.feature_engineer.get_features_list,
            labels=self.feature_engineer.get_labels_list,
            treatments_types=self.feature_engineer.get_string_indexer_treatments_names(),
            propensity_model_type=self.propensity_model_type,
        )
        wildebeest_dre_model_fitted = wildebeest_dre_model.fit(
            dataset=self.feature_engineer.featurized_data
        )
        wildebeest_dre_model.save_all_trained_models_to_s3(self.s3_output_path)
        wildebeest_dre_model.save_test_data(self.s3_output_path)
        wildebeest_dre_model_fitted.save_model_artifact_to_s3(self.s3_output_path)
        return wildebeest_dre_model, wildebeest_dre_model_fitted

    @log_decorator
    def _create_and_persist_scoring_pipeline(
        self, featurization_pipeline, model, sampled_process_data
    ):
        scoring_pipeline_creator = ScoringPipeline(self.s3_output_path)
        marketing_index = self.feature_engineer.get_marketing_index()
        if self.model_type == "dre":
            scoring_pipeline_creator.create_scoring_pipeline(
                featurization_pipeline_model=featurization_pipeline,
                sampled_process_data=sampled_process_data,
            )
        elif self.model_type == "regression":
            scoring_pipeline_creator.create_scoring_pipeline(
                featurization_pipeline_model=featurization_pipeline,
                sampled_process_data=sampled_process_data,
                marketing_index=marketing_index,
                model=model,
            )
        scoring_pipeline_creator.persist_scoring_pipeline()


class WildebeestDoublyRobustScoringExecutor(BaseBuilder, HasConfigParam):
    def __init__(
        self,
        scoring_data_location,
        holiday_data_location,
        pipeline_data_location,
        model_base_path,
        output_data_location,
        model_type="dre",
        do_bucketize=False,
    ):
        super().__init__("", scoring_data_location, holiday_data_location, model_type, do_bucketize)
        HasConfigParam.__init__(self)
        self.spark = SparkSession.builder.getOrCreate()
        self.sparse_to_dense_udf = udf(lambda v: v.toArray().tolist(), ArrayType(DoubleType()))
        self.second_element = f.udf(lambda v: float(v[1]), FloatType())
        # Assigning other attributes
        self.pipeline_data_location = pipeline_data_location
        self.model_base_path = model_base_path
        self.output_data_location = output_data_location

        self.labels = []
        self.trained_models = {
            "propensity": None,
            "counterfactual": None,
            "treatments": {"ada": None, "aap": None},
        }

    @log_decorator
    def _build_score_data(self):
        self.build_data_preprocessor()
        processed_data = self.data_preprocessor.get_processed_data
        return processed_data

    @log_decorator
    def _read_pipeline_data(self):
        """Load the fitted pipeline model."""
        scoring_pipeline_model = PipelineModel.load(self.pipeline_data_location)
        return scoring_pipeline_model

    def _get_feature_list(self):
        """Retrieve the feature list from the featurization stages of the fitted pipeline model."""

        # Load the fitted pipeline model
        scoring_pipeline_model = self._read_pipeline_data()

        # Get the featurization stages
        featurization_stages = scoring_pipeline_model.stages

        # Access the feature list from the VectorAssembler stage
        vector_assembler_stage = [
            stage for stage in featurization_stages if isinstance(stage, VectorAssembler)
        ][0]
        feature_list = vector_assembler_stage.getInputCols()

        return feature_list

    @log_decorator
    def _get_propensity_treatment_labels(self, scoring_pipeline_model):
        """Get all teh treatment classes/labels that propenity model was trained on."""
        return get_string_indexer_labels(scoring_pipeline_model)

    @log_decorator
    def _get_featurized_score_data(self, scoring_pipeline_model, processed_data):
        """Transform the processed data using the loaded pipeline model."""
        featurized_score_data = scoring_pipeline_model.transform(processed_data)
        return featurized_score_data

    @log_decorator
    def _load_models_from_s3(self):
        # Load propensity and counterfactual models
        for model_type in ["propensity", "counterfactual"]:
            model_path = generate_paths(self.model_base_path, model_type)
            try:
                self.trained_models[model_type] = RandomForestClassificationModel.load(model_path)
            except Exception as e:
                print(f"Error loading model from {model_path}. Error: {e}")

        # Load treatment models
        for treatment in self.trained_models["treatments"].keys():
            model_path = generate_paths(self.model_base_path, treatment)
            try:
                self.trained_models["treatments"][treatment] = RandomForestClassificationModel.load(
                    model_path
                )
            except Exception as e:
                print(f"Error loading model from {model_path}. Error: {e}")

    @staticmethod
    def _transform(model, data):
        return model.transform(data)

    def _get_propensity_scores(self, model, dataset):
        dataset = self._transform(model, dataset)
        dataset = dataset.withColumn(
            "propensity_prob_dense", self.sparse_to_dense_udf(dataset["propensity_prob"])
        )
        dataset = dataset.withColumn(
            "propensity_rawPrediction_dense",
            self.sparse_to_dense_udf(dataset["propensity_rawPrediction"]),
        )

        for i, label in enumerate(self.labels):
            dataset = dataset.withColumn(
                f"{label}_propensity_probability", dataset["propensity_prob_dense"][i]
            )

        for treatment in self.labels:
            dataset = dataset.withColumn(
                f"normalized_{treatment}_propensity_probability",
                dataset[f"{treatment}_propensity_probability"],
            )

        return dataset  # truncate_propensity_scores(dataset, labels=self.labels)

    def _get_counterfactual_probability(self, model, dataset):
        dataset = self._transform(model, dataset)
        dataset = dataset.withColumn(
            "probability_counterfactual",
            self.second_element(dataset["probability_counterfactual"]),
        )
        return dataset

    def _get_outcome_probability(self, model, dataset, treatment_type):
        dataset = self._transform(model, dataset)
        probability_col_name = f"{treatment_type}_probability"
        dataset = dataset.withColumn(
            probability_col_name,
            self.second_element(dataset[probability_col_name]),
        )
        return dataset

    @log_decorator
    def _get_score_data(self, dataset: DataFrame):
        dataset.cache()
        dataset = self._get_propensity_scores(self.trained_models["propensity"], dataset)
        dataset = self._get_counterfactual_probability(
            self.trained_models["counterfactual"], dataset
        )

        dre_utils = WildebeestDREstimationUtils()

        for treatment_type, model_type in self.trained_models["treatments"].items():
            dataset = self._get_outcome_probability(
                model_type, dataset, treatment_type=treatment_type
            )
            dataset, ate_att_dict = dre_utils.compute_lift_and_dre(
                dataset=dataset, treatment_type=treatment_type, perform_att_evaluations=False
            )

        dataset.unpersist()
        return dataset

    def _process_scored_data(self, dataset):
        # Common columns that always need to be selected
        common_columns = [
            "adUserId",
            "conversionCurrency",
            "conversionOps",
            "conversionIndicator",
            "treatment_type",
            "treatment_type_indexed",
        ]

        selected_columns = common_columns.copy()
        treatments = self.trained_models["treatments"].keys()

        for treatment in treatments:
            treatment_columns = [col.name for col in dataset.schema.fields if treatment in col.name]
            selected_columns.extend(treatment_columns)

        # Select the necessary columns from the dataset
        final_dataset = dataset.select(*selected_columns)

        # Add new columns dynamically based on treatments
        for treatment in treatments:
            final_dataset = final_dataset.withColumn(
                f"miops_{treatment.upper()}",
                f.col(f"{treatment}_lift") * f.col("conversionOps"),
            )

        return final_dataset

    @log_decorator
    def _persist_scored_data(
        self,
        final_dataset: DataFrame,
        output_path: str,
        file_format: str = "parquet",
    ) -> None:
        """
        Save the final processed dataset to the specified location.

        :param final_dataset: The final Spark DataFrame to be saved.
        :param output_path: The output path where the dataset should be saved.
        :param file_format: The format in which to save the dataset. Default is "parquet".
        """
        try:
            final_dataset.write.format(file_format).mode("overwrite").save(
                f"{output_path}/scored_data/"
            )
            logger.info(
                f"Final scored data saved successfully to {output_path} in {file_format} format."
            )
        except Exception as e:
            logger.error(f"Failed to save the final dataset. Error: {e}")
            raise e

    def run_steps(self):
        processed_data = self._build_score_data()

        # Repartitioning the data based on the default parallelism of the Spark context
        num_partitions = self.spark.sparkContext.defaultParallelism
        processed_data = processed_data.repartition(num_partitions)

        scoring_pipeline = self._read_pipeline_data()
        self.labels = self._get_propensity_treatment_labels(scoring_pipeline)
        self._load_models_from_s3()

        # Create a ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit jobs for parallel processing
            future_featurized_data = executor.submit(
                self._get_featurized_score_data, scoring_pipeline, processed_data
            )
            future_scored_data = executor.submit(
                self._get_score_data, future_featurized_data.result()
            )

            # Retrieve the results once the parallel processing is completed
            featurized_data = future_featurized_data.result()
            scored_dataset = future_scored_data.result()

        final_dataset = self._process_scored_data(dataset=scored_dataset)
        self._persist_scored_data(
            final_dataset=final_dataset,
            output_path=self.output_data_location,
        )
        return final_dataset


class WildebeestOLSScoringExecutor:
    # Entry point class for running purchase model scoring job
    # Chain of steps for model scoring execution until we persist the scored data output
    scored_data_cols = [
        "adUserId",
        "conversionCurrency",
        "conversionType",
        "liftWb",
        "liftWbStandardError",
        "liftMlaGnuA",
        "liftMlaGnuB",
        "miopsWb",
        "miopsMlaGnuA",
        "miopsMlaGnuB",
        "pDenomMlaGnuA",
        "pDenomMlaGnuB",
        "pNumMlaGnuA",
        "pNumMlaGnuB",
    ]

    def __init__(
        self,
        scoring_data_location,
        pipeline_data_location,
        purchase_attribution_data_location,
    ):
        self.spark = SparkSession.builder.getOrCreate()
        self.scoring_data_location = scoring_data_location
        self.pipeline_data_location = pipeline_data_location
        self.purchase_attribution_data_location = purchase_attribution_data_location

    @log_decorator
    def _read_scoring_data(self):
        self.scoring_data = dataframe_reader(self.scoring_data_location, self.spark)
        return self

    @log_decorator
    def _read_pipeline_data(self):
        self.scoring_pipeline = Pipeline.load(self.pipeline_data_location)
        return self

    @log_decorator
    def _score_model(self):
        scored_data = self.scoring_pipeline.fit(self.scoring_data).transform(self.scoring_data)
        self.scored_data = scored_data
        return self

    @log_decorator
    def _persist_scored_data(self):
        processed_scored_data = self.process_scored_data(self.scored_data)
        processed_scored_data.cache()
        self.processed_scored_data = processed_scored_data
        processed_scored_data.write.parquet(
            self.purchase_attribution_data_location, mode="overwrite", compression="zstd"
        )
        return self

    def run_steps(self):
        # Running chain of steps. Any subsequent step depends on running previous steps.
        self._read_scoring_data()
        self._read_pipeline_data()
        self._score_model()
        self._persist_scored_data()

    def process_scored_data(self, scored_data):
        predictionCol = self.scoring_pipeline.getStages()[-1].getPredictionCol()
        processed_scored_data = (
            scored_data.select(
                "adUserId", "conversionCurrency", f"{predictionCol}.*", "conversionOps"
            )
            .withColumn("conversionType", f.lit("orderItem"))
            .withColumn("miopsWb", f.col("liftWb") * f.col("conversionOps"))
            .withColumn("miopsMlaGnuA", f.col("liftMlaGnuA") * f.col("conversionOps"))
            .withColumn("miopsMlaGnuB", f.col("liftMlaGnuB") * f.col("conversionOps"))
            .select(*self.scored_data_cols)
        )
        return processed_scored_data

    def get_scored_data(self):
        return self.processed_scored_data

    def get_scoring_pipeline(self):
        return self.scoring_pipeline

    def get_scoring_data(self):
        return self.scoring_data

    def get_regression_model(self):
        return self.scoring_pipeline.getStages()[-1]
