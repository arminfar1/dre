import logging
from typing import Dict, Union, Any

from pyspark import keyword_only
from pyspark.ml.param import Param, Params
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import FloatType, StringType, StructType, StructField
import pyspark.sql.functions as f
from pyspark.ml.param.shared import (
    HasFeaturesCol,
    HasLabelCol,
    HasWeightCol,
)
from pyspark.ml import Estimator, Model
from pyspark.ml.classification import (
    RandomForestClassifier,
)
from pyspark.ml.util import DefaultParamsWritable, DefaultParamsReadable
from pyspark.ml.linalg import DenseVector

from .propensity import PropensityModel
from ..estimators.ml_models import (
    SparkBinaryClassificationMetrics,
)
from .model_evaluators import WildebeestDREstimationUtils
from ..utilities.utils import (
    split_train_test,
    log_decorator,
    generate_paths,
    flatten_dict,
    truncate_propensity_scores,
)


def sparse_to_dense(sparse_vec):
    # Define UDF to convert SparseVector to DenseVector
    return DenseVector(sparse_vec.toArray())


logger = logging.getLogger()


class WildebeestDoublyRobustEstimation(
    Estimator,
    HasFeaturesCol,
    HasLabelCol,
    HasWeightCol,
    DefaultParamsWritable,
    DefaultParamsReadable,
):
    @keyword_only
    def __init__(
        self,
        features_list,
        labels,
        treatments_types,
        propensity_model_type="RandomForest",
        **kwargs,
    ):
        super(WildebeestDoublyRobustEstimation, self).__init__()
        self.features_list = features_list
        self.labels = labels
        self.treatments_types = treatments_types
        self.propensity_model_type = propensity_model_type
        self.trained_models: Dict[str, Union[None, Any]] = {
            "propensity": None,
            "counterfactual": None,
            "treatments": {},
        }
        self.mode_artifact = {}
        self.second_element = f.udf(lambda v: float(v[1]), FloatType())  # Get the probability of 1
        self._setDefault(featuresCol="features")
        self._setDefault(labelCol="conversionIndicator")
        self._input_kwargs = kwargs
        self._set(**self._input_kwargs)

        # Initialize PropensityModel
        self.propensity_model = PropensityModel(
            propensity_model_type=self.propensity_model_type,
            featuresCol=self.getFeaturesCol(),
            labels=self.labels,
        )

    def setParams(self, featuresCol=None, labelCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setFeaturesCol(self, featuresCol):
        return self._set(featuresCol=featuresCol)

    def setLabelCol(self, labelCol):
        return self._set(labelCol=labelCol)

    def get_treatment_index(self, treatment_name):
        if treatment_name in self.labels:
            return float(self.labels.index(treatment_name))
        else:
            raise ValueError(f"Treatment name '{treatment_name}' not found in treatments.")

    @log_decorator
    def get_outcome_probability(
        self,
        train_dataset,
        test_dataset,
        treatment_type,
        treatment_index,
        exclude_treated_user=True,
    ):
        count = train_dataset.filter(f.col("treatment_type_indexed") == treatment_index).count()
        logger.info("Number of rows with treatment type %s is %s", treatment_type, count)
        treatment_col = f.col("treatment_type_indexed")

        # Filter train dataset based on the treatment type
        filtered_train_dataset = train_dataset.filter(treatment_col == treatment_index)

        prediction_col_name = (
            f"{treatment_type}_prediction"
            if not exclude_treated_user
            else "prediction_counterfactual"
        )
        probability_col_name = (
            f"{treatment_type}_probability"
            if not exclude_treated_user
            else "probability_counterfactual"
        )
        raw_prediction_col_name = (
            f"{treatment_type}_rawPrediction"
            if not exclude_treated_user
            else "rawPrediction_counterfactual"
        )

        logger.info("fitting started")
        # Train Random Forest model on filtered_dataset
        outcome_model = RandomForestClassifier(
            featuresCol=self.getFeaturesCol(),
            labelCol=self.getLabelCol(),
            numTrees=50,
            maxDepth=10,
            maxBins=32,
            featureSubsetStrategy="sqrt",
            minInstancesPerNode=20,
            subsamplingRate=0.8,
            impurity="gini",
            predictionCol=prediction_col_name,
            probabilityCol=probability_col_name,
            rawPredictionCol=raw_prediction_col_name,
        ).fit(filtered_train_dataset)
        logger.info("fitting ended")

        train_dataset = outcome_model.transform(train_dataset)
        test_dataset = outcome_model.transform(test_dataset)

        # Get evaluation metrics
        filtered_test_dataset = test_dataset.filter(treatment_col == treatment_index)
        metrics_dict = self.get_model_metrics(
            filtered_test_dataset,
            labelCol=self.getLabelCol(),
            predictionCol=prediction_col_name,
            probabilityCol=probability_col_name,
            rawPredictionCol=raw_prediction_col_name,
        )
        logger.info("Evaluation metrics for %s: %s", treatment_type, metrics_dict)

        train_dataset = train_dataset.withColumn(
            probability_col_name,
            self.second_element(train_dataset[probability_col_name]),
        )

        test_dataset = test_dataset.withColumn(
            probability_col_name,
            self.second_element(test_dataset[probability_col_name]),
        )

        return train_dataset, test_dataset, outcome_model

    @staticmethod
    def get_model_metrics(
        dataset,
        labelCol,
        predictionCol: str,
        probabilityCol: str,
        rawPredictionCol: str,
    ):
        # Print evaluations metrics for propensity
        metrics_obj = SparkBinaryClassificationMetrics(
            dataset,
            labelCol=labelCol,
            predictionCol=predictionCol,
            probabilityCol=probabilityCol,
            rawPredictionCol=rawPredictionCol,
        )
        metrics_dict = metrics_obj.get_metrics()

        return metrics_dict

    def estimate_propensity_scores(self, train, test):
        # Step 1: Estimate the propensity score for each treatment
        train, test, fitted_model = self.propensity_model.fit(train, test)
        self.trained_models["propensity"] = fitted_model

        # Step 2: Calibrate probabilities for Random Forest. Fore Logistic Regression, there is no need to calibrate
        # as it returns well-calibrated probabilities. See: https://scikit-learn.org/stable/modules/calibration.html
        train, test = self.propensity_model.calibrate_propensities(train, test)

        # Step 3: Truncate propensity scores and perform propensity overlap
        train = truncate_propensity_scores(train, labels=self.labels)
        test = truncate_propensity_scores(test, labels=self.labels)

        # Step 4: Perform propensity overlap
        # See this paper for more details: https://academic.oup.com/aje/article/188/1/250/5090958
        # https://arxiv.org/pdf/2206.15367.pdf
        train, test = self.propensity_model.perform_propensity_overlap(
            train,
            test,
        )

        return train, test

    def _fit_treatments(self, train_after_overlap, test_after_overlap):
        all_results = {treatment: {} for treatment in self.treatments_types}

        # Step 1: Fit outcome for untreated
        train_after_overlap, test_after_overlap, untreated_model = self.get_outcome_probability(
            train_after_overlap,
            test_after_overlap,
            treatment_type="no_impression",
            treatment_index=self.get_treatment_index("no_impression"),
            exclude_treated_user=True,
        )

        self.trained_models["counterfactual"] = untreated_model

        for treatment in self.treatments_types:
            logger.info("Fitting for treatment %s", treatment)

            treatment_index = self.get_treatment_index(treatment)

            # Step 2: Fit outcome for each treatment
            train_after_overlap, test_after_overlap, treated_model = self.get_outcome_probability(
                train_after_overlap,
                test_after_overlap,
                treatment_type=treatment,
                treatment_index=treatment_index,
                exclude_treated_user=False,
            )
            self.trained_models["treatments"][treatment] = treated_model

            # Step 3: Compute lift and the doubly robust estimates
            dre_utils = WildebeestDREstimationUtils()
            test_after_overlap, ate_att_dict = dre_utils.compute_lift_and_dre(
                dataset=test_after_overlap, treatment_type=treatment, label_col=self.getLabelCol()
            )
            all_results[treatment]["ATE_avg"] = ate_att_dict["ATE"]["average"]
            all_results[treatment]["ATE_CI"] = ate_att_dict["ATE"]["CI"]
            all_results[treatment]["ATT_avg"] = ate_att_dict["ATT"]["average"]
            all_results[treatment]["ATT_CI"] = ate_att_dict["ATT"]["CI"]

        logger.info("all results is : %s", all_results)
        return all_results, test_after_overlap

    def _fit(self, dataset: DataFrame, params=None):
        train, test = split_train_test(dataset)
        train_after_overlap, test_after_overlap = self.estimate_propensity_scores(train, test)
        self.mode_artifact, self.final_test_data = self._fit_treatments(
            train_after_overlap, test_after_overlap
        )

        return WildebeestDoublyRobustEstimationModel(modelArtifact=self.mode_artifact)

    def save_test_data(self, base_path):
        """
         Save the test dataframe that can be used later for scoring.
        Parameters:
             - base_path (str): path to write the file..
         Returns:
             None.
        """
        test_data_path = generate_paths(base_path, "final_test_data")
        try:
            self.final_test_data.write.parquet(test_data_path, mode="overwrite")
            logger.info(f"Test data saved to {test_data_path}.")
        except Exception as e:
            logger.error(f"Error saving test data to path {test_data_path}: {e}")

    def save_all_trained_models_to_s3(self, base_path):
        """
        Save trained models and their features importance to a specified S3 base path.
        Parameters:
            - base_path: S3 path to save the trained models.

        Returns:
            None.
        """

        def save_model_and_importance(
            model_to_save: Model, model_path: str, fi_path: str, spark: SparkSession
        ):
            """Helper function to save model and its feature importance."""
            if not model_to_save:
                logger.warning(f"No model found for path {model_path}. Skipping save operation.")
                return

            try:
                model_to_save.write().overwrite().save(model_path)
                logger.info(f"Model saved to {model_path}.")

                # Save features importance if available
                if hasattr(model_to_save, "featureImportances"):
                    feature_importance = model_to_save.featureImportances.toArray()
                    feature_names = self.features_list
                    df_fi = spark.createDataFrame(
                        [(t, float(i)) for t, i in zip(feature_names, feature_importance)],
                        ["feature", "importance"],
                    )
                    df_fi.coalesce(1).write.parquet(fi_path, mode="overwrite")
                    logger.info(f"Feature importance saved to {fi_path}.")
            except Exception as e:
                logger.error(f"Error saving model or feature importance for path {model_path}: {e}")

        # Save models and features importance
        spark_session = SparkSession.builder.getOrCreate()

        # Saving the propensity and counterfactual models
        for model_type, trained_model in {
            "propensity": self.trained_models["propensity"],
            "counterfactual": self.trained_models["counterfactual"],
        }.items():
            model_path = generate_paths(base_path, model_type)
            fi_path = generate_paths(base_path, model_type, "feature_importance")
            save_model_and_importance(trained_model, model_path, fi_path, spark=spark_session)

        # Saving the treatment models
        for treatment, trained_model in self.trained_models["treatments"].items():
            model_path = generate_paths(base_path, treatment)
            fi_path = generate_paths(base_path, treatment, "feature_importance")
            save_model_and_importance(trained_model, model_path, fi_path, spark=spark_session)


class WildebeestDoublyRobustEstimationModel(Model):
    modelArtifact = Param(
        Params._dummy(), "modelArtifact", "Model artifact parameter as a dictionary."
    )

    @keyword_only
    def __init__(self, modelArtifact=None):
        super(WildebeestDoublyRobustEstimationModel, self).__init__()
        self._setDefault(modelArtifact={})
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, modelArtifact=None):
        return self._set(modelArtifact=modelArtifact)

    def setModelArtifact(self, value):
        return self._set(modelArtifact=value)

    def getModelArtifact(self):
        return self.getOrDefault(self.modelArtifact)

    def _transform(self, dataset: DataFrame):
        raise NotImplementedError(
            "Transform not implemented for WildebeestDoublyRobustEstimationModel"
        )

    def unpack_model_artifact(self):
        # List of expected keys
        metrics = ["ATE_avg", "ATE_CI", "ATT_avg", "ATT_CI"]

        # Unpack the artifact
        modelArtifact = self.getModelArtifact()

        # Dynamically set attributes based on the treatments and keys
        for treatment, values in modelArtifact.items():
            for metric in metrics:
                # Construct an attribute name based on treatment and metric
                attr_name = f"{metric}_{treatment}"
                setattr(self, attr_name, values.get(metric, None))

    @property
    def get_model_summary(self):
        self.unpack_model_artifact()
        model_summary = {
            "ATE ada": self.ATE_avg_ada,
            "ATE ada CI": self.ATE_CI_ada,
            "ATT ada": self.ATT_avg_ada,
            "ATT ada CI": self.ATE_CI_ada,
            "ATE aap": self.ATE_avg_aap,
            "ATE aap CI": self.ATE_CI_aap,
            "ATT aap": self.ATT_avg_aap,
            "ATT aap CI": self.ATE_CI_aap,
        }
        return model_summary

    def save_model_artifact_to_s3(self, output_path):
        model_artifact = self.get_model_summary

        # Flatten the model_artifact dictionary
        flattened_artifact = flatten_dict(model_artifact)

        spark = SparkSession.builder.appName("Save Model Artifact").getOrCreate()

        # Define the schema for the DataFrame
        schema = StructType(
            [StructField("Key", StringType(), True), StructField("Value", StringType(), True)]
        )

        # Create DataFrame using the flattened_artifact and the defined schema
        model_artifact_df = spark.createDataFrame(
            [(k, v) for k, v in flattened_artifact.items()], schema=schema
        )

        # Write the DataFrame to the S3 location as a CSV file
        model_artifact_df.coalesce(1).write.parquet(
            f"{output_path}/model_artifact/", mode="overwrite"
        )
