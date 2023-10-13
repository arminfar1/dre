from pyspark import keyword_only
from pyspark.sql.types import IntegerType
import pyspark.sql.functions as f
from pyspark.ml.linalg import Vectors
from pyspark.ml import Transformer
from pyspark.ml.feature import ElementwiseProduct, Bucketizer
from pyspark.ml.param.shared import (
    HasInputCols,
    HasThreshold,
)
from functools import reduce
import operator
from pyspark.sql.functions import col
from pyspark.ml.util import DefaultParamsWritable, DefaultParamsReadable

from ..utilities.utils import HasConfigParam, hasColumn


class VariableTransformer(
    Transformer, HasConfigParam, DefaultParamsReadable, DefaultParamsWritable
):
    @keyword_only
    def __init__(self, configParam=None, do_bucketize=False):
        super(VariableTransformer, self).__init__()
        self.setConfigParam(configParam)
        self.do_bucketize = do_bucketize

    @keyword_only
    def setParams(self, configParam=None):
        return self.setConfigParam(configParam)

    def setConfigParam(self, configParam):
        return self._set(configParam=configParam)

    def prepare_bucket_columns(self, df, ada_num_days, aap_num_days, imp_threshold):
        df = df.withColumn(
            f"ada_in_imp_{ada_num_days}",
            reduce(operator.add, [col("adaLinkInD" + str(i)) for i in range(ada_num_days)]),
        )
        df = df.withColumn(
            f"aap_in_imp_{aap_num_days}",
            reduce(operator.add, [col("aapLinkInD" + str(i)) for i in range(aap_num_days)]),
        )

        df = df.withColumn(
            f"ada_in_imp_{ada_num_days}", col(f"ada_in_imp_{ada_num_days}").cast(IntegerType())
        )
        df = df.withColumn(
            f"aap_in_imp_{aap_num_days}", col(f"aap_in_imp_{aap_num_days}").cast(IntegerType())
        )
        return df

    @staticmethod
    def create_buckets(imp_threshold, input_column_name, output_column_name):
        return Bucketizer(
            splits=[float("-inf"), 0.00001, imp_threshold, 200, float("inf")],
            inputCol=input_column_name,
            outputCol=output_column_name,
        )

    def _transform(self, df):
        treatments_days = self.getConfigParam["treatments"]
        ada_num_days = treatments_days["ada_num_days"]
        aap_num_days = treatments_days["aap_num_days"]
        treatment_ada, treatment_aap = self.getTreatments
        imp_threshold = self.getConfigParam["imp_threshold"]

        if self.do_bucketize:
            df = self.prepare_bucket_columns(df, ada_num_days, aap_num_days, imp_threshold)
            ada_bucketizer = self.create_buckets(
                imp_threshold,
                input_column_name=f"ada_in_imp_{ada_num_days}",
                output_column_name=f"ada_in_imp_{ada_num_days}_bucket",
            )
            aap_bucketizer = self.create_buckets(
                imp_threshold,
                input_column_name=f"aap_in_imp_{aap_num_days}",
                output_column_name=f"aap_in_imp_{aap_num_days}_bucket",
            )
            transformed_df = ada_bucketizer.transform(df)
            transformed_df = aap_bucketizer.transform(transformed_df)

            transformed_df = (
                transformed_df.withColumn(
                    f"aap_out_imp_1", f.when(f.col("aapLinkOutD0") > 0, 1).otherwise(0)
                )
                .withColumn(
                    f"aap_out_imp_7",
                    reduce(operator.add, [f.col("aapLinkOutD" + str(i)) for i in range(0, 7)]),
                )
                .withColumn(
                    "temp_aap_out_imp_7", f.when(f.col(f"aap_out_imp_7") > 0, 1).otherwise(0)
                )
                .withColumn("customerIsPrime", f.col("customerIsPrime").cast(IntegerType()))
                .withColumn("intercept", f.lit(1))
            )

        else:
            transformed_df = (
                df.withColumn(
                    f"{treatment_ada}_sum",
                    reduce(
                        operator.add, [f.col("adaLinkInD" + str(i)) for i in range(0, ada_num_days)]
                    ),
                )
                .withColumn(
                    f"{treatment_aap}_sum",
                    reduce(
                        operator.add, [f.col("aapLinkInD" + str(i)) for i in range(0, aap_num_days)]
                    ),
                )
                .withColumn(
                    treatment_ada,
                    f.when(f.col(f"{treatment_ada}_sum") > imp_threshold, 1).otherwise(0),
                )
                .withColumn(
                    treatment_aap,
                    f.when(f.col(f"{treatment_aap}_sum") > imp_threshold, 1).otherwise(0),
                )
                .withColumn(f"aap_out_imp_1", f.when(f.col("aapLinkOutD0") > 0, 1).otherwise(0))
                .withColumn(
                    f"aap_out_imp_7",
                    reduce(operator.add, [f.col("aapLinkOutD" + str(i)) for i in range(0, 7)]),
                )
                .withColumn(f"aap_out_imp_7", f.when(f.col(f"aap_out_imp_7") > 0, 1).otherwise(0))
                .withColumn("customerIsPrime", f.col("customerIsPrime").cast(IntegerType()))
                .withColumn("intercept", f.lit(1))
            )

            # Identify what impression the user have seen
            transformed_df = transformed_df.withColumn(
                "treatment_type",
                f.when((f.col(treatment_ada) > 0) & (f.col(treatment_aap) == 0), "ada")
                .when((f.col(treatment_ada) == 0) & (f.col(treatment_aap) > 0), "aap")
                .when((f.col(treatment_ada) > 0) & (f.col(treatment_aap) > 0), "both_impressions")
                .otherwise("no_impression"),
            )

            # Exclude all the treatment_type that have both_impressions
            transformed_df = transformed_df.filter(
                transformed_df.treatment_type != "both_impressions"
            )

            transformed_df = transformed_df.withColumn(
                "is_treated_by_ada",
                f.when(f.col("treatment_type") == f.lit("ada"), 1.0).otherwise(0.0),
            )
            transformed_df = transformed_df.withColumn(
                "is_treated_by_aap",
                f.when(f.col("treatment_type") == f.lit("aap"), 1.0).otherwise(0.0),
            )
            transformed_df = transformed_df.withColumn(
                "is_treated_by_no_impression",
                f.when(f.col("treatment_type") == f.lit("no_impression"), 1.0).otherwise(0.0),
            )

            # Check for the presence of adCustomerOrderItemIdCount column
            if hasColumn(df, "adCustomerOrderItemIdCount"):
                transformed_df = transformed_df.withColumn(
                    "purchase_weight",
                    f.when(
                        f.col("adCustomerOrderItemIdCount") > 0,
                        1 / f.col("adCustomerOrderItemIdCount"),
                    ).otherwise(1),
                )
            else:
                transformed_df = transformed_df.withColumn("purchase_weight", f.lit(1))

        return transformed_df


class IndicatorTransformer(
    Transformer, HasInputCols, HasThreshold, DefaultParamsReadable, DefaultParamsWritable
):
    @keyword_only
    def __init__(self, inputCols=None, threshold=None):
        super(IndicatorTransformer, self).__init__()
        kwargs = self._input_kwargs
        self._set(**kwargs)

    @keyword_only
    def setParams(self, inputCols=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setInputCols(self, inputCols):
        return self._set(inputCols=inputCols)

    def setThreshold(self, threshold):
        return self._set(threshold=threshold)

    def _transform(self, df):
        col_list = self.getInputCols()
        threshold = self.getThreshold()
        if not hasattr(threshold, "__getitem__"):
            threshold = [threshold] * len(col_list)
        for c, th in zip(col_list, threshold):
            df = df.withColumn(c, f.when(f.col(c) > th, 1).otherwise(0))
        return df


class NoMarketingMaskTransformer(ElementwiseProduct):
    def __init__(self, nm_mask):
        super(NoMarketingMaskTransformer, self).__init__()
        self._setDefault()
        self.setParams(scalingVec=Vectors.dense(nm_mask))
        self.setInputCol("features")
        self.setOutputCol("nm_features")
