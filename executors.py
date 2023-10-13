import logging
import sys

from pyspark import keyword_only
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import FloatType

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
    filter_by_range,
)
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
    ):
        self.data_preprocessor = DataPreprocessing()
        self.feature_engineer = None
        self.data_schema_location = data_schema_location
        self.raw_data_location = raw_data_location
        self.holidays_data_location = holidays_data_location
        self.model_type = model_type
        self.do_bucketize = do_bucketize

    def build_data_schema(self):
        self.data_preprocessor.read_data_schema(self.data_schema_location)
        return self

    def build_data_preprocessor(self):
        self.data_preprocessor.read_holidays(self.holidays_data_location)
        self.data_preprocessor.read_raw_data(self.raw_data_location)
        return self

    def build_feature_engineer(self):
        self.feature_engineer = FeatureEngineering(
            self.data_preprocessor, self.model_type, self.do_bucketize
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
        model_type="regression",
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
        self.bucket = builder.bucket
        self.data_preprocessor: DataPreprocessing = builder.data_preprocessor
        self.feature_engineer: FeatureEngineering = builder.feature_engineer
        self.model_trainer = builder.model_trainer

    @log_decorator
    def run(self):
        featurization_pipeline = self.feature_engineer.featurization_pipeline
        if self.model_type == "regression":
            (
                wildebeest_purchase_regression_model,
                wildebeest_purchase_regression_model_fitted,
            ) = self._train_regression_model()
            self._create_and_persist_scoring_pipeline(
                featurization_pipeline, wildebeest_purchase_regression_model_fitted
            )
        else:
            wildebeest_dre_model, wildebeest_dre_model_fitted = self._train_doubly_robust_model()
            self._create_and_persist_scoring_pipeline(featurization_pipeline, wildebeest_dre_model)

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
            marketing_features=self.data_preprocessor.marketing_features,
            labels=self.feature_engineer.get_string_indexer_labels(),
            treatments_types=self.feature_engineer.get_string_indexer_treatments_names(),
        )
        wildebeest_dre_model_fitted = wildebeest_dre_model.fit(
            dataset=self.feature_engineer.featurized_data
        )
        wildebeest_dre_model.save_models_to_s3(self.s3_output_path)
        wildebeest_dre_model.save_test_data(self.s3_output_path)
        wildebeest_dre_model_fitted.save_model_artifact_to_s3(self.s3_output_path)
        return wildebeest_dre_model, wildebeest_dre_model_fitted

    @log_decorator
    def _create_and_persist_scoring_pipeline(self, featurization_pipeline, model):
        scoring_pipeline_creator = ScoringPipeline(self.s3_output_path)
        marketing_index = self.feature_engineer.get_marketing_index()
        if self.model_type == "dre":
            scoring_pipeline_creator.create_scoring_pipeline(
                featurization_pipeline=featurization_pipeline
            )
        elif self.model_type == "regression":
            scoring_pipeline_creator.create_scoring_pipeline(
                featurization_pipeline=featurization_pipeline,
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
        self.second_element = f.udf(lambda v: float(v[1]), FloatType())
        # Assigning other attributes
        self.pipeline_data_location = pipeline_data_location
        self.model_base_path = model_base_path
        self.output_data_location = output_data_location

        self.treatment_ada, self.treatment_aap = self.getTreatments
        self.trained_models = {
            "propensity": None,
            "counterfactual": None,
            "treatments": {self.treatment_ada: None, self.treatment_aap: None},
        }

    @log_decorator
    def _build_score_data(self):
        self.build_data_preprocessor()
        processed_data = self.data_preprocessor.get_processed_data()
        return processed_data

    @log_decorator
    def _read_pipeline_data(self):
        scoring_pipeline = Pipeline.load(self.pipeline_data_location)
        return scoring_pipeline

    @log_decorator
    def _get_featurized_score_data(self, scoring_pipeline, processed_data):
        featurized_score_data = scoring_pipeline.fit(processed_data).transform(processed_data)
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

    def _get_propensity_scores(self, model, dataset, treatment):
        propensity_col_name = f"{treatment}_propensity_prob"
        dataset = self._transform(model, dataset)
        dataset = dataset.withColumn(
            propensity_col_name,
            self.second_element(dataset[f"{treatment}_propensity_prob"]),
        )
        return filter_by_range(dataset=dataset, col_to_filter=propensity_col_name)

    def _get_outcome_probability(self, model, dataset, treatment, outcome_type):
        dataset = self._transform(model, dataset)
        dataset = dataset.withColumn(
            f"{treatment}_probability_{outcome_type}",
            self.second_element(dataset[f"{treatment}_probability_outcome_{outcome_type}"]),
        )

        return dataset

    @log_decorator
    def _get_score_data(self, dataset: DataFrame):
        dataset.cache()
        results = {}  # Store results for each treatment
        for treatment, models in self.trained_models.items():
            dataset = self._get_propensity_scores(models["propensity"], dataset, treatment)
            dataset = self._get_outcome_probability(
                models["treated"], dataset, treatment, outcome_type="treated"
            )
            dataset = self._get_outcome_probability(
                models["untreated"], dataset, treatment, outcome_type="untreated"
            )

            # Do the evaluation, Compute lift, DRE, ATE, ATT
            dre_utils = WildebeestDREstimationUtils()
            dataset, ate_att_dict = dre_utils.compute_lift_and_dre(
                dataset=dataset, treatment_type=treatment
            )
        dataset.unpersist()
        return results, dataset

    def _process_Scored_data(self, dataset):
        # Common columns that always need to be selected
        common_columns = [
            "adCustomerOrderItemId",
            "adUserId",
            "conversionCurrency",
            "conversionOps",
            "conversionIndicator",
        ]

        selected_columns = common_columns.copy()

        for treatment in self.trained_models.keys():
            treatment_columns = [col.name for col in dataset.schema.fields if treatment in col.name]
            selected_columns.extend(treatment_columns)

        # Select the necessary columns from the dataset
        final_dataset = dataset.select(*selected_columns)

        # Add new columns dynamically based on treatments
        for treatment in self.trained_models.keys():
            final_dataset = final_dataset.withColumn(
                f"miopsFromATE_{treatment.upper()}",
                f.col(f"{treatment}_doubly_robust_estimate") * f.col("conversionOps"),
            ).withColumn(
                f"miopsFromLift_{treatment.upper()}",
                f.col(f"{treatment}_lift") * f.col("conversionOps"),
            )

        return final_dataset

    @log_decorator
    def _persist_scored_data(
        self,
        final_dataset: DataFrame,
        results_artifacts: {},
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
            spark = SparkSession.builder.appName("Save Scored Artifact").getOrCreate()
            results_artifact_df = spark.createDataFrame(
                [(k, v) for k, v in results_artifacts.items()], ["Key", "Value"]
            )
            # Write the DataFrame to the S3 location as a CSV file
            results_artifact_df.coalesce(1).write.parquet(
                f"{output_path}/model_artifact/", mode="overwrite"
            )
            logger.info(
                f"Final scored and results saved successfully to {output_path} in {file_format} format."
            )
        except Exception as e:
            logger.error(f"Failed to save the final dataset. Error: {e}")
            raise e

    def run_steps(self):
        processed_data = self._build_score_data()
        scoring_pipeline = self._read_pipeline_data()
        featurized_data = self._get_featurized_score_data(scoring_pipeline, processed_data)
        self._load_models_from_s3()
        result, scored_dataset = self._get_score_data(featurized_data)
        final_dataset = self._process_Scored_data(dataset=scored_dataset)
        self._persist_scored_data(
            final_dataset=final_dataset,
            results_artifacts=result,
            output_path=self.output_data_location,
        )


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
