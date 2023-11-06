import logging

import numpy as np
import pyspark.sql.functions as f
from pyspark.sql.functions import col, mean as _mean
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, Row

logger = logging.getLogger()


class WildebeestDREstimationUtils:
    """
    Utility class for computing DRE estimates, Average Treatment Effect (ATE),
    Average Treatment effect on the Treated (ATT), and lift for causal inference models.
    """

    @staticmethod
    def _compute_doubly_robust_estimates(
        dataset: DataFrame, treatment_type: str, label_col: str = ""
    ):
        """
        Compute the Doubly Robust Estimate (DRE) for causal inference.

        This function calculates the DRE based on the formula:
        The formula is derived from the concept of combining both the propensity score and regression adjustments
        to estimate causal effects, making it doubly robust. For a detailed explanation, refer to:
        https://matheusfacure.github.io/python-causality-handbook/12-Doubly-Robust-Estimation.html

        Parameters:
        - dataset (DataFrame): The input dataset containing treatment, outcome, etc.
        - treatment_type (str): The name of the treatment column.
        - label_col (str, optional): The name of the outcome column. If not provided, it defaults to 1.

        Returns:
        - DataFrame: The dataset with added columns for debiased factual, debiased counterfactual, and DRE.
        """

        treatment_indicator_col = f"is_treated_by_{treatment_type}"

        # Check for the existence of the propensity column and set accordingly
        if f"normalized_{treatment_type}_propensity_probability" in dataset.columns:
            propensity_column = f"normalized_{treatment_type}_propensity_probability"
        else:
            propensity_column = f"{treatment_type}_propensity_probability"

        mu_1_col = f"{treatment_type}_probability"
        mu_0_col = "probability_counterfactual"  # added treatment name

        # Check if label_col exists, if not set it to 1
        y_col = f.col(label_col) if label_col and label_col in dataset.columns else f.lit(1)

        # Debiased Factual
        debiased_factual_col_name = f"debiased_factual_{treatment_type}"  # added treatment name
        debiased_factual = col(treatment_indicator_col) * (y_col - col(mu_1_col)) / col(
            propensity_column
        ) + col(mu_1_col)

        # Debiased Counterfactual
        debiased_counterfactual_col_name = (
            f"debiased_counterfactual_{treatment_type}"  # added treatment name
        )
        debiased_counterfactual = (1 - col(treatment_indicator_col)) * (y_col - col(mu_0_col)) / (
            1 - col(propensity_column)
        ) + col(mu_0_col)

        dataset = dataset.withColumn(debiased_factual_col_name, debiased_factual)
        dataset = dataset.withColumn(debiased_counterfactual_col_name, debiased_counterfactual)

        # DRE column
        dataset = dataset.withColumn(
            f"{treatment_type}_doubly_robust_estimate",
            col(debiased_factual_col_name) - col(debiased_counterfactual_col_name),
        )

        return dataset

    @staticmethod
    def _calculate_lift(dataset: DataFrame, treatment_type: str):
        """
         Calculate the lift for causal inference based on the Doubly Robust Estimate (DRE).

        The lift is calculated as 1 - (debiased_counterfactual / debiased_factual).
        If debiased_factual is zero or the result is negative, the lift is set to 0.

        Parameters:
         - dataset (DataFrame): The input dataset containing debiased factual and debiased counterfactual columns.
         - treatment (str): The name of the treatment column.

         Returns:
         - DataFrame: The dataset with an added column for lift.
        """

        lift_col = f"{treatment_type}_lift"
        debiased_factual_col = f"debiased_factual_{treatment_type}"
        debiased_counterfactual_col = f"debiased_counterfactual_{treatment_type}"

        dataset = dataset.withColumn(
            lift_col,
            f.when(
                col(debiased_factual_col) != 0,
                f.when(
                    1 - col(debiased_counterfactual_col) / col(debiased_factual_col) > 0,
                    1 - col(debiased_counterfactual_col) / col(debiased_factual_col),
                ).otherwise(0),
            ).otherwise(0),
        )

        return dataset

    @staticmethod
    def _bootstrap_estimation(
        dataset: DataFrame, treatment_type: str, num_iterations: int = 1000, att_only=False
    ):
        dre_column = f"{treatment_type}_doubly_robust_estimate"

        if att_only:
            dataset = dataset.filter(col(f"is_treated_by_{treatment_type}") == 1)

        # Broadcast the dataset
        spark = SparkSession.builder.getOrCreate()
        dataset_bc = spark.sparkContext.broadcast(dataset.collect())

        #  bootstrap function that is used in parallelization
        def bootstrap_function(_):
            local_dataset = dataset_bc.value
            bootstrap_sample = [
                local_dataset[i] for i in np.random.choice(len(local_dataset), len(local_dataset))
            ]
            return float(sum(row[dre_column] for row in bootstrap_sample) / len(bootstrap_sample))

        # Parallelize the bootstrapping process
        rdd = spark.sparkContext.parallelize(range(num_iterations))
        bootstrapped_estimates_rdd = rdd.map(bootstrap_function)

        # Convert RDD to DataFrame
        bootstrapped_estimates_df = bootstrapped_estimates_rdd.map(
            lambda value: Row(value=value)
        ).toDF()

        # Calculate average using the aggregated sum and count
        sum_count = bootstrapped_estimates_rdd.aggregate(
            (0, 0),
            (lambda acc, value: (acc[0] + value, acc[1] + 1)),
            (lambda acc1, acc2: (acc1[0] + acc2[0], acc1[1] + acc2[1])),
        )
        avg_estimate = sum_count[0] / sum_count[1]

        # Calculate approximate percentiles using DataFrame's approxQuantile method
        lower_bound, upper_bound = bootstrapped_estimates_df.approxQuantile(
            "value", [0.025, 0.975], 0.01
        )

        return avg_estimate, (lower_bound, upper_bound)

    @staticmethod
    def _calculate_ATE_and_ATT(dataset: DataFrame, treatment_type: str, sample_size: int = 1500000):
        # Important: Ensure the smaller, selected dataset is persisted in memory to optimize reuse.
        # By persisting only the necessary columns, we reduce memory footprint  This can lead
        # to significant performance improvements, especially when the number of bootstrap iterations is large.
        dataset_count = dataset.count()
        if dataset_count > sample_size:
            # Sample the dataset to approximately 1 million rows
            fraction = sample_size / float(dataset.count())
            logger.warning(
                "Test dataset has %s rows.. Sampling %s of the dataset for bootstrapping.",
                dataset_count,
                fraction,
            )

            dataset = dataset.sample(withReplacement=False, fraction=fraction)

        calculated_dre_df = dataset.select(f"{treatment_type}_doubly_robust_estimate")

        # Ensure the dataset is persisted in memory
        calculated_dre_df.persist()

        avg_ATE, CI_ATE = WildebeestDREstimationUtils._bootstrap_estimation(
            calculated_dre_df, treatment_type
        )
        avg_ATT, CI_ATT = WildebeestDREstimationUtils._bootstrap_estimation(
            calculated_dre_df, treatment_type, att_only=True
        )

        # Unpersist the dataset
        calculated_dre_df.unpersist()

        return {
            "ATE": {"average": avg_ATE, "CI": CI_ATE},
            "ATT": {"average": avg_ATT, "CI": CI_ATT},
        }

    def compute_lift_and_dre(
        self,
        dataset: DataFrame,
        treatment_type: str,
        label_col: str = "",
        perform_att_evaluations=True,
    ):
        # calculate the dre estimates for each individual
        dataset = WildebeestDREstimationUtils._compute_doubly_robust_estimates(
            dataset, treatment_type, label_col
        )

        ate_att_dict = None
        if perform_att_evaluations:
            # Compute the doubly robust estimates
            ate_att_dict = self._calculate_ATE_and_ATT(
                dataset=dataset, treatment_type=treatment_type
            )

        # Compute lift
        dataset = self._calculate_lift(dataset=dataset, treatment_type=treatment_type)

        return dataset, ate_att_dict
