import logging
import math

from pyspark.sql.types import FloatType, ArrayType, DoubleType
import pyspark.sql.functions as f
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import (
    RandomForestClassifier,
    LogisticRegression,
)
from pyspark.sql.functions import udf

from ..estimators.ml_models import (
    SparkMultiClassClassificationMetrics,
    CrossValidation,
)
from ..utilities.utils import log_decorator

logger = logging.getLogger()


class PropensityModel:
    """
    Initialize the PropensityModel class.

    Parameters:
        - featuresCol (str): Name of the column containing feature vectors.
        - labels (list): List of treatment labels.
        - model_to_use (str, optional): Model to be used for propensity score estimation.
        Defaults to "RandomForest".
        - do_balance_classes (bool, optional): Whether to balance classes or not. Defaults to True.
        - do_cross_validation (bool, optional): Whether to perform cross-validation. Defaults to False.
        - print_metrics (bool, optional): Whether to print model evaluation metrics. Defaults to True.
    """

    def __init__(
        self,
        featuresCol,
        labels,
        propensity_model_type="",
        do_balance_classes=True,
        do_cross_validation=False,
        print_metrics=True,
    ):
        self.model_to_use = propensity_model_type
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
    def _handle_class_imbalance_and_weights(dataset, weighting_scheme="sqrt", max_class_ratio=5.0):
        """
        Handle class imbalance in the dataset and compute class weights. The class weights are being used
        in the model to help with prediction improvements.

        Parameters:
            - dataset (DataFrame): Input dataset with class imbalances.
            - weighting_scheme (str, optional): Scheme to compute class weights. Defaults to "sqrt".
            - max_class_ratio (float, optional): Maximum allowed ratio between classes. Defaults to 5.0.

        Returns:
            - DataFrame: Dataset with added 'weights' column.
        """

        # 1. Determine Class Counts
        class_freqs = dataset.groupBy("treatment_type_indexed").count().rdd.collectAsMap()
        n_classes = len(class_freqs)

        # 2. Determine Sampling Fractions for balancing
        max_allowed_size = max_class_ratio * min(class_freqs.values())
        sampling_fractions = {
            class_type: max_allowed_size / count if count > max_allowed_size else 1.0
            for class_type, count in class_freqs.items()
        }

        # 3. Sample Data for balance
        dataset = dataset.sampleBy("treatment_type_indexed", sampling_fractions)

        # Number of rows after balancing
        balanced_row_count = dataset.count()
        logger.info(f"Number of rows after balancing: %s", balanced_row_count)

        # Recompute class frequencies after balancing
        balanced_class_freqs = dataset.groupBy("treatment_type_indexed").count().rdd.collectAsMap()

        # 4. Compute class weights based on chosen weighting_scheme
        if weighting_scheme == "sqrt":
            class_weights = {
                class_type: math.sqrt(balanced_row_count / (n_classes * count))
                for class_type, count in balanced_class_freqs.items()
            }
        else:  # default to normalized inverse frequency
            class_weights = {
                class_type: balanced_row_count / (n_classes * count)
                for class_type, count in balanced_class_freqs.items()
            }

        # Normalize class weights, so they sum up to the number of classes
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
    def fit(self, train, test):
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
            train_dataset = self._handle_class_imbalance_and_weights(train)
            weightCol_param = "weights"
        else:
            weightCol_param = None
            train_dataset = train

        logger.info("A %s model will be used for training the propensity", self.model_to_use)

        if self.model_to_use in [("LogisticRegression",), "LogisticRegression"]:
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
        elif self.model_to_use in [("RandomForest",), "RandomForest"]:
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
        else:
            raise ValueError(f"Unsupported model for the propensity: {self.model_to_use}")

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
    def calibrate_propensities(self, train, test=None):
        """
        Calibrate and normalize propensity scores using Platt scaling. If a test dataset is provided,
        it will also be calibrated and normalized.

        Parameters:
            - train (DataFrame): Training dataset.
            - test (DataFrame, optional): Test dataset. If not provided, only the train dataset will
              be calibrated and normalized. Defaults to None.

        Returns:
            - tuple(DataFrame, DataFrame): Calibrated and normalized train and test datasets. If no test
              dataset is provided, the second element of the tuple will be None.
        """

        if self.model_to_use in [("LogisticRegression",), "LogisticRegression"]:
            logger.info("Since Logistic Regression is used no calibrations is required.")
            for treatment in self.labels:
                logger.info("Calibrating probabilities for treatment %s", treatment)
                train = train.withColumn(
                    f"normalized_{treatment}_propensity_probability",
                    train[f"{treatment}_propensity_probability"],
                )
                if test:
                    test = test.withColumn(
                        f"normalized_{treatment}_propensity_probability",
                        test[f"{treatment}_propensity_probability"],
                    )

        elif self.model_to_use in [("RandomForest",), "RandomForest"]:
            for treatment in self.labels:
                logger.info("Calibrating probabilities for treatment %s", treatment)
                score_col = f"{treatment}_propensity_probability"
                train, temp_test = self.platt_scaling(
                    train, test if test else None, score_col, f"is_treated_by_{treatment}"
                )
                if test:
                    test = temp_test

            # After calibrating all, compute the sum of calibrated probabilities
            sum_of_probs_train = sum(
                col(f"calibrated_{treatment}_propensity_probability") for treatment in self.labels
            )
            sum_of_probs_test = None
            if test:
                sum_of_probs_test = sum(
                    col(f"calibrated_{treatment}_propensity_probability")
                    for treatment in self.labels
                )

            # Normalize the calibrated probabilities
            for treatment in self.labels:
                train = train.withColumn(
                    f"normalized_{treatment}_propensity_probability",
                    train[f"calibrated_{treatment}_propensity_probability"] / sum_of_probs_train,
                )
                if test:
                    test = test.withColumn(
                        f"normalized_{treatment}_propensity_probability",
                        test[f"calibrated_{treatment}_propensity_probability"] / sum_of_probs_test,
                    )
        else:
            raise ValueError(f"Unsupported model for the propensity: {self.model_to_use}")

        return train, test if test else None

    @log_decorator
    def perform_propensity_overlap(self, train, test):
        """
        This function filters the train and test datasets to only include observations
        that fall within the Common Support Region of propensity scores for all classes.

        The Common Support Region is determined by finding the region where propensity
        scores for all classes overlap. This ensures that there's a common support for
        all treatments or classes, which is crucial for unbiased causal inference.

        Parameters:
            train: Training dataset DataFrame.
            test: Testing dataset DataFrame.
            Defaults to False.

        Returns:
            train and test DataFrames with overlapped propensity scores.

        """
        # Lists to store the minimum and maximum propensity scores for each treatment
        all_mins = []
        all_maxs = []

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

        return train, test

    def get_model_metrics(self, dataset):
        # Print evaluations metrics for propensity
        metrics_obj = SparkMultiClassClassificationMetrics(
            dataset,
            labelCol="treatment_type_indexed",
            labels_list=self.labels,
            predictionCol="propensity_pred",
            probabilityCol="propensity_prob",
            rawPredictionCol="propensity_rawPrediction",
        )

        metrics_dict = metrics_obj.get_metrics()
        class_metrics, weighted_metrics = metrics_obj.get_class_metrics()
        return metrics_dict, class_metrics, weighted_metrics
