import logging
import math
from typing import Dict, Union, Any, List

from pyspark import keyword_only
from pyspark.ml.param import Param, Params
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import FloatType, ArrayType, DoubleType, StringType
import pyspark.sql.functions as f
from pyspark.sql.functions import col
from pyspark.ml.param.shared import (
    HasFeaturesCol,
    HasLabelCol,
    HasWeightCol,
)
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Estimator, Model
from pyspark.ml.classification import (
    RandomForestClassifier,
    LogisticRegression,
)
from pyspark.ml.util import DefaultParamsWritable, DefaultParamsReadable
from pyspark.ml.linalg import DenseVector
from pyspark.sql.functions import udf

from ..estimators.ml_models import (
    SparkClassificationMetrics,
    CrossValidation,
)
from .model_evaluators import WildebeestDREstimationUtils
from ..utilities.utils import (
    split_train_test,
    log_decorator,
    generate_paths, flatten_dict,
)


def sparse_to_dense(sparse_vec):
    # Define UDF to convert SparseVector to DenseVector
    return DenseVector(sparse_vec.toArray())


logger = logging.getLogger()


class PropensityModel:
    """
    Initialize the PropensityModel class.

    Parameters:
    - featuresCol (str): Name of the column containing feature vectors.
    - labels (list): List of treatment labels.
    - model_used (str, optional): Model to be used for propensity score estimation. Defaults to "RandomForest".
    - do_balance_classes (bool, optional): Whether to balance classes or not. Defaults to True.
    - do_cross_validation (bool, optional): Whether to perform cross-validation. Defaults to False.
    - print_metrics (bool, optional): Whether to print model evaluation metrics. Defaults to True.
    """

    def __init__(
        self,
        featuresCol,
        labels,
        model_used="RandomForest",
        do_balance_classes=True,
        do_cross_validation=False,
        print_metrics=True,
    ):
        self.model_used = model_used
        self.features_col = featuresCol
        self.labels = labels
        self.do_balance_classes = do_balance_classes
        self.do_cross_validation = do_cross_validation
        self.print_metrics = print_metrics
        self.sparse_to_dense_udf = udf(lambda v: v.toArray().tolist(), ArrayType(DoubleType()))
        self.second_element = f.udf(lambda v: float(v[1]), FloatType())  # Get the probability of 1

    def getFeaturesCol(self):
        """
        Return the name of the column containing feature vectors.

        Returns:
        - str: Name of the features .
        """
        return self.features_col

    @staticmethod
    @log_decorator
    def handle_class_imbalance_and_weights(dataset, weighting_scheme="sqrt", max_class_ratio=8.0):
        """
        Handle class imbalance in the dataset and compute class weights. The class weights are being used
        in the model to help with prediction improvements.

        Parameters:
        - dataset (DataFrame): Input dataset with class imbalances.
        - weighting_scheme (str, optional): Scheme to compute class weights. Defaults to "sqrt".
        - max_class_ratio (float, optional): Maximum allowed ratio between classes. Defaults to 8.0.

        Returns:
        - DataFrame: Dataset with added 'weights' column.

        """

        # Initial number of rows in the data
        initial_row_count = dataset.count()
        logger.info(f"Initial number of rows: %s", initial_row_count)

        # 1. Determine Class Counts
        class_freqs = dataset.groupBy("treatment_type_indexed").count().collect()
        class_count_dict = {row["treatment_type_indexed"]: row["count"] for row in class_freqs}
        n_classes = len(class_freqs)

        # 2. Determine Sampling Fractions for balancing
        max_allowed_size = max_class_ratio * min(class_count_dict.values())
        sampling_fractions = {}
        for class_type, count in class_count_dict.items():
            if count > max_allowed_size:
                sampling_fractions[class_type] = max_allowed_size / count
            else:
                sampling_fractions[class_type] = 1.0

        # 3. Sample Data for balance
        dataset = dataset.sampleBy("treatment_type_indexed", sampling_fractions)

        # Number of rows after balancing
        balanced_row_count = dataset.count()
        logger.info(f"Number of rows after balancing: %s", balanced_row_count)

        # Recompute class frequencies after balancing
        balanced_class_freqs = dataset.groupBy("treatment_type_indexed").count().collect()
        balanced_class_count_dict = {
            row["treatment_type_indexed"]: row["count"] for row in balanced_class_freqs
        }

        # 4. Compute class weights based on chosen weighting_scheme
        if weighting_scheme == "sqrt":
            class_weights = {
                class_type: math.sqrt(balanced_row_count / (n_classes * count))
                for class_type, count in balanced_class_count_dict.items()
            }
        else:  # default to normalized inverse frequency
            class_weights = {
                class_type: balanced_row_count / (n_classes * count)
                for class_type, count in balanced_class_count_dict.items()
            }

        # Normalize class weights so they sum up to the number of classes
        sum_weights = sum(class_weights.values())
        class_weights = {k: (v * n_classes) / sum_weights for k, v in class_weights.items()}
        logger.info("class_weights: %s", class_weights)

        # Add a new column 'weights' to the DataFrame
        dataset = dataset.withColumn(
            "weights",
            f.when(
                dataset["treatment_type_indexed"].isin(list(class_weights.keys())),
                dataset["treatment_type_indexed"],
            ).otherwise(None),
        )
        dataset = dataset.replace(to_replace=class_weights, subset=["weights"])
        dataset = dataset.withColumn("weights", dataset["weights"].cast("double"))

        return dataset

    @log_decorator
    def fit_propensity(self, train, test):
        """
        Fit the propensity model to the training data and transform both train and test datasets.
        If self.do_balance_classes set to True, a balancing is being done. Then this method will
        return the balanced dataset.

        Parameters:
        - train (DataFrame): Training dataset.
        - test (DataFrame): Test dataset.

        Returns:
        - DataFrame, DataFrame, Model: Transformed train dataset, transformed test dataset,
        and the fitted propensity model.
        """
        logger.info("Adding class weights has set to %s", self.do_balance_classes)
        # Initialize the model based on the flag
        if self.do_balance_classes:
            train_dataset = self.handle_class_imbalance_and_weights(train)
            weightCol_param = "weights"
        else:
            weightCol_param = None
            train_dataset = train

        if self.model_used == "LogisticRegression":
            propensity = LogisticRegression(
                featuresCol=self.getFeaturesCol(),
                labelCol="treatment_type_indexed",
                weightCol=weightCol_param,
                predictionCol="propensity_pred",
                probabilityCol="propensity_prob",
                rawPredictionCol="propensity_rawPrediction",
                family="multinomial",
                maxIter=250,
                regParam=0.016,  # Regularization strength
                elasticNetParam=0.5,  # L1 regularization, between [0,1] both. I need to test this more
            )
        elif self.model_used == "RandomForest":
            propensity = RandomForestClassifier(
                featuresCol=self.getFeaturesCol(),
                labelCol="treatment_type_indexed",
                numTrees=150,
                maxDepth=15,
                minInstancesPerNode=10,
                maxBins=32,
                weightCol=weightCol_param,
                featureSubsetStrategy="sqrt",
                subsamplingRate=0.9,
                impurity="gini",
                predictionCol="propensity_pred",
                probabilityCol="propensity_prob",
                rawPredictionCol="propensity_rawPrediction",
            )

        if self.do_cross_validation:
            # Assuming you have a CrossValidation function defined elsewhere
            cv = CrossValidation(
                dataset=train_dataset,
                model=propensity,
                label_col="treatment_type_indexed",
                predictionCol="propensity_pred",
                probabilityCol="propensity_prob",
                rawPredictionCol="propensity_rawPrediction",
            )
            propensity_fitted, best_parms = cv.run()
            for param, value in best_parms.items():
                logger.info(f"{param.name}: {value}")
        else:
            propensity_fitted = propensity.fit(train_dataset)

        train_dataset = propensity_fitted.transform(train_dataset)
        test = propensity_fitted.transform(test)

        if self.print_metrics:
            metrics_dict, class_metrics, weighted_metrics = self.get_model_metrics(dataset=test)
            logger.info("Evaluation metrics for propensity: %s", metrics_dict)
            logger.info("Evaluation metrics for each class of propensity: %s", class_metrics)
            logger.info("Evaluation weighted metrics: %s", weighted_metrics)

        train_dataset = train_dataset.withColumn(
            "propensity_prob_dense",
            self.sparse_to_dense_udf(train_dataset["propensity_prob"]),
        )
        test = test.withColumn(
            "propensity_prob_dense", self.sparse_to_dense_udf(test["propensity_prob"])
        )

        for i, label in enumerate(self.labels):
            train_dataset = train_dataset.withColumn(
                f"{label}_propensity_probability", train_dataset["propensity_prob_dense"][i]
            )
            test = test.withColumn(
                f"{label}_propensity_probability", test["propensity_prob_dense"][i]
            )
        return train_dataset, test, propensity_fitted

    @log_decorator
    def platt_scaling(self, train_data, test_data, score_col, label_col):
        """
        Apply Platt scaling to calibrate the probabilities.
        https://en.wikipedia.org/wiki/Platt_scaling

        Parameters:
        - train_data (DataFrame): Training dataset.
        - test_data (DataFrame): Test dataset.
        - score_col (str): Name of the column containing scores to be calibrated.
        - label_col (str): Name of the label column.

        Returns:
        - DataFrame, DataFrame: Calibrated train and test datasets.
        """
        # Rename the existing 'probability' column to avoid conflict
        if "probability" in train_data.columns:
            train_data = train_data.withColumnRenamed("probability", "original_probability")
        if "probability" in test_data.columns:
            test_data = test_data.withColumnRenamed("probability", "original_probability")

        assembler = VectorAssembler(inputCols=[score_col], outputCol="score_col_feature")
        train_data = assembler.transform(train_data)
        test_data = assembler.transform(test_data)

        lr = LogisticRegression(
            featuresCol="score_col_feature",
            labelCol=label_col,
            predictionCol="prediction",  # Keep this as the default "prediction"
            regParam=0.015,  # Regularization strength
            elasticNetParam=1,  # L1 regularization
        )
        lr_model = lr.fit(train_data)

        calibrated_train = lr_model.transform(train_data)
        calibrated_test = lr_model.transform(test_data)

        # Extract the probability of the positive class (assuming binary classification)
        calibrated_train = calibrated_train.withColumn(
            f"calibrated_{score_col}", self.second_element(calibrated_train.probability)
        )
        calibrated_test = calibrated_test.withColumn(
            f"calibrated_{score_col}", self.second_element(calibrated_test.probability)
        )

        return calibrated_train.drop(
            "score_col_feature", "rawPrediction", "prediction", "probability"
        ), calibrated_test.drop("score_col_feature", "rawPrediction", "prediction", "probability")

    @log_decorator
    def calibrate_propensities(self, train, test):
        """
        Calibrate propensity scores using Platt scaling.

        Parameters:
        - train (DataFrame): Training dataset.
        - test (DataFrame): Test dataset.

        Returns:
        - DataFrame, DataFrame: Calibrated train and test datasets.
        """
        # Platt Scaling for the treatments in self.labels
        for treatment in self.labels:
            logger.info("Calibrating probabilities for treatment %s", treatment)
            score_col = f"{treatment}_propensity_probability"
            train, test = self.platt_scaling(train, test, score_col, f"is_treated_by_{treatment}")

        # Compute the sum of calibrated probabilities for each instance in train and test datasets
        sum_of_probs_train = sum(
            col(f"calibrated_{treatment}_propensity_probability") for treatment in self.labels
        )
        sum_of_probs_test = sum(
            col(f"calibrated_{treatment}_propensity_probability") for treatment in self.labels
        )

        # Normalize the calibrated probabilities or copy original propensities if logistic regression was used
        for treatment in self.labels:
            if self.model_used == "RandomForest":
                train = train.withColumn(
                    f"normalized_{treatment}_propensity_probability",
                    train[f"calibrated_{treatment}_propensity_probability"] / sum_of_probs_train,
                )
                test = test.withColumn(
                    f"normalized_{treatment}_propensity_probability",
                    test[f"calibrated_{treatment}_propensity_probability"] / sum_of_probs_test,
                )
            else:  # Logistic Regression
                train = train.withColumn(
                    f"normalized_{treatment}_propensity_probability", train[score_col]
                )
                test = test.withColumn(
                    f"normalized_{treatment}_propensity_probability", test[score_col]
                )
        return train, test

    def truncate_propensity_scores(self, train_dataset: DataFrame, test_dataset: DataFrame):
        """
        Truncate propensity scores for the labels based on provided bounds.

        Args:
            train_dataset: Training dataset DataFrame.
            test_dataset: Testing dataset DataFrame.

        Returns:
            Tuple containing filtered train and test DataFrames.
        """
        combined_filter = self._generate_combined_filter(self.labels)
        train_dataset = train_dataset.filter(combined_filter)
        test_dataset = test_dataset.filter(combined_filter)

        return train_dataset, test_dataset

    @staticmethod
    def _generate_combined_filter(labels: List[str], lower_bound=0.01, upper_bound=0.99) -> str:
        """
        Generate a combined filter expression for multiple labels based on provided bounds.

        Args:
            labels: List of labels.

        Returns:
            Combined filter expression string.
        """
        filters = [
            f"({treatment}_propensity_probability BETWEEN {lower_bound} AND {upper_bound})"
            for treatment in labels
        ]
        combined_filter = " AND ".join(filters)

        return combined_filter

    @log_decorator
    def perform_propensity_overlap(self, train, test, count_dropped_rows=False):
        """
        This function filters the train and test datasets to only include observations
        that fall within the Common Support Region of propensity scores for all classes.

        The Common Support Region is determined by finding the region where propensity
        scores for all classes overlap. This ensures that there's a common support for
        all treatments or classes, which is crucial for unbiased causal inference.
        """
        # Lists to store the minimum and maximum propensity scores for each treatment
        all_mins = []
        all_maxs = []

        if count_dropped_rows:
            original_train_count = train.count()
            original_test_count = test.count()

        for treatment in self.labels:
            truncated_propensity_col_name = f"{treatment}_propensity_probability"

            # Calculate the minimum and maximum propensity scores for the current treatment group
            min_max = train.agg(
                f.min(truncated_propensity_col_name), f.max(truncated_propensity_col_name)
            ).first()

            # If there are no results (None), skip this iteration
            if not min_max:
                continue

            all_mins.append(min_max[0])
            all_maxs.append(min_max[1])

        # Determine the common support region by finding the maximum of all minimum scores
        # and the minimum of all maximum scores. This ensures that the resulting region
        # has overlapping propensity scores for all classes.
        if None in all_mins or None in all_maxs:
            # Log a warning and skip the overlap calculation for this iteration
            logger.warning(
                "One or more treatments/classes have no rows in the train DataFrame. Skipping overlap calculation."
            )
            return train, test  # Return the original dataframes without filtering
        else:
            p_min = max(all_mins)
            p_max = min(all_maxs)

        if p_min > p_max:
            logger.warning(
                "No overlap in propensity scores detected. Returning original DataFrames."
            )
            return train, test

        logger.info(
            " all mins is %s, all max is %s, p-min is %s and p-max is %s",
            all_mins,
            all_maxs,
            p_min,
            p_max,
        )

        # Filter the datasets based on the global bounds for each treatment
        for treatment in self.labels:
            truncated_propensity_col_name = f"{treatment}_propensity_probability"

            global_filter_train = (train[truncated_propensity_col_name] >= p_min) & (
                train[truncated_propensity_col_name] <= p_max
            )
            global_filter_test = (test[truncated_propensity_col_name] >= p_min) & (
                test[truncated_propensity_col_name] <= p_max
            )

            train = train.filter(global_filter_train)
            test = test.filter(global_filter_test)

        # If required, count the number of rows dropped after filtering for overlap
        if count_dropped_rows:
            train_after_overlap_count = train.count()
            test_after_overlap_count = test.count()
            dropped_train_count = original_train_count - train_after_overlap_count
            dropped_test_count = original_test_count - test_after_overlap_count

            logger.info(f"Dropped {dropped_train_count} rows from train data after overlap.")
            logger.info(f"Dropped {dropped_test_count} rows from test data after overlap.")
        return train, test

    def get_model_metrics(self, dataset):
        # Print evaluations metrics for propensity
        metrics_obj = SparkClassificationMetrics(
            dataset,
            labelCol="treatment_type_indexed",
            labels_list=self.labels,
            predictionCol="propensity_pred",
            probabilityCol="propensity_prob",
            rawPredictionCol="propensity_rawPrediction",
        )

        metrics_dict = metrics_obj.get_multi_class_metrics()
        class_metrics, weighted_metrics = metrics_obj.get_class_metrics()
        return metrics_dict, class_metrics, weighted_metrics


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
        marketing_features,
        labels,
        treatments_types,
        **kwargs,
    ):
        super(WildebeestDoublyRobustEstimation, self).__init__()
        self.features_list = features_list
        self.marketing_features = marketing_features
        self.labels = labels
        self.treatments_types = treatments_types
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

        # Initialize PropensityModel within WildebeestDoublyRobustEstimation
        self.propensity_model = PropensityModel(
            featuresCol=self.getFeaturesCol(), labels=self.labels
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
        metrics_obj = SparkClassificationMetrics(
            dataset,
            labelCol=labelCol,
            predictionCol=predictionCol,
            probabilityCol=probabilityCol,
            rawPredictionCol=rawPredictionCol,
        )
        metrics_dict = metrics_obj.get_binary_metrics()

        return metrics_dict

    def estimate_propensity_scores(self, train, test):
        # Step 1: Estimate the propensity score for each treatment
        train, test, fitted_model = self.propensity_model.fit_propensity(train, test)
        self.trained_models["propensity"] = fitted_model

        # Step 2: Calibrate probabilities for Random Forest. Fore Logistic Regression, there is no need to calibrate
        # as it returns well-calibrated probabilities. See: https://scikit-learn.org/stable/modules/calibration.html
        train, test = self.propensity_model.calibrate_propensities(train, test)

        # Step 3: Truncate propensity scores and perform propensity overlap
        train, test = self.propensity_model.truncate_propensity_scores(train, test)
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
            all_results[treatment]["ATE_avg"] = ate_att_dict['ATE']["average"]
            all_results[treatment]["ATE_CI"] = ate_att_dict['ATE']["CI"]
            all_results[treatment]["ATT_avg"] = ate_att_dict['ATT']["average"]
            all_results[treatment]["ATT_CI"] = ate_att_dict['ATT']["CI"]

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
        test_data_path = generate_paths(base_path, "final_test_data")
        try:
            self.final_test_data.write.parquet(test_data_path, mode="overwrite")
            logger.info(f"Test data saved to {test_data_path}.")
        except Exception as e:
            logger.error(f"Error saving test data to path {test_data_path}: {e}")

    def save_models_to_s3(self, base_path):
        """Save trained models and their features importance to a specified S3 base path."""

        def save_model_and_importance(
            model: Model, model_path: str, fi_path: str, spark: SparkSession
        ):
            """Helper function to save model and its feature importance."""
            if not model:
                logger.warning(f"No model found for path {model_path}. Skipping save operation.")
                return

            try:
                model.write().overwrite().save(model_path)
                logger.info(f"Model saved to {model_path}.")

                # Save features importance if available
                if hasattr(model, "featureImportances"):
                    feature_importance = model.featureImportances.toArray()
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
        for model_type, model in {
            "propensity": self.trained_models["propensity"],
            "counterfactual": self.trained_models["counterfactual"],
        }.items():
            model_path = generate_paths(base_path, model_type)
            fi_path = generate_paths(base_path, model_type, "feature_importance")
            save_model_and_importance(model, model_path, fi_path, spark=spark_session)

        # Saving the treatment models
        for treatment, model in self.trained_models["treatments"].items():
            model_path = generate_paths(base_path, treatment)
            fi_path = generate_paths(base_path, treatment, "feature_importance")
            save_model_and_importance(model, model_path, fi_path, spark=spark_session)


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

        # Create DataFrame using the flattened_artifact
        model_artifact_df = spark.createDataFrame(
            [(k, v) for k, v in flattened_artifact.items()], ["Key", StringType()]
        )

        # Write the DataFrame to the S3 location as a CSV file
        model_artifact_df.coalesce(1).write.parquet(
            f"{output_path}/model_artifact/", mode="overwrite"
        )


