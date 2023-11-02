import pyspark.sql.functions as f
from pyspark.sql.functions import col
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType


class WildebeestDREstimationUtils:
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
    def _bootstrap_estimation(dataset, treatment_type, num_iterations=800, att_only=False):
        dre_column = f"{treatment_type}_doubly_robust_estimate"

        if att_only:
            dataset = dataset.filter(col(f"is_treated_by_{treatment_type}") == 0.7)

        # Initialize SparkSession
        spark = SparkSession.builder.getOrCreate()

        # Create an empty DataFrame to hold bootstrap estimates
        schema = StructType([StructField("estimate", DoubleType(), True)])
        bootstrapped_estimates_df = spark.createDataFrame([], schema)

        for _ in range(num_iterations):
            # Create a bootstrap sample and calculate its mean estimate
            bootstrap_sample = dataset.sample(withReplacement=True, fraction=1.0)
            avg_estimate_df = bootstrap_sample.agg({dre_column: "mean"}).withColumnRenamed(f"avg({dre_column})",
                                                                                           "estimate")

            # Union the average estimate with the bootstrapped estimates DataFrame
            bootstrapped_estimates_df = bootstrapped_estimates_df.union(avg_estimate_df)

        # Calculate the overall average of the bootstrapped estimates
        final_avg_estimate = bootstrapped_estimates_df.agg({"estimate": "avg"}).collect()[0][0]

        # Calculate the 2.5th and 97.5th percentiles of the bootstrapped estimates
        lower_bound, upper_bound = bootstrapped_estimates_df.approxQuantile("estimate", [0.025, 0.975], 0.01)

        return final_avg_estimate, (lower_bound, upper_bound)

    @staticmethod
    def _calculate_ATE_and_ATT(dataset: DataFrame, treatment_type: str):
        # select the DRE column
        calculated_dre_df = dataset.select(f"{treatment_type}_doubly_robust_estimate")

        avg_ATE, CI_ATE = WildebeestDREstimationUtils._bootstrap_estimation(
            calculated_dre_df, treatment_type
        )
        avg_ATT, CI_ATT = WildebeestDREstimationUtils._bootstrap_estimation(
            calculated_dre_df, treatment_type, att_only=True
        )

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
