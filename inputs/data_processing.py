from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, StringIndexerModel
from pyspark.ml import Pipeline
import numpy as np
from pyspark.sql import functions as f
from pyspark.sql.functions import broadcast
from pyspark.sql.functions import col

from ..utilities.utils import (
    read_json_from_s3,
    dataframe_reader,
    log_decorator,
    HasConfigParam,
)
from ..inputs.transformers import VariableTransformer, IndicatorTransformer


class DataPreprocessing:
    def __init__(self):
        self.mla_training_data = None
        self.removed_holiday_feat_list = []
        self.holidays_df = None
        self.marketing_features = []
        self.traffic_features = []
        self.holiday_features = []
        self.cols_to_keep = []
        self.mla_data_schema = None
        self.spark = SparkSession.builder.getOrCreate()

    @log_decorator
    def read_data_schema(self, data_schema_location):
        mla_data_schema = read_json_from_s3(data_schema_location)
        self.mla_data_schema = mla_data_schema
        self.cols_to_keep = list(mla_data_schema[mla_data_schema.description != "other"].name)
        self.holiday_features = list(mla_data_schema[mla_data_schema.description == "holiday"].name)
        self.traffic_features = list(
            mla_data_schema[
                mla_data_schema.description.isin(["transit", "last_touch", "interaction"])
            ].name
        )
        self.marketing_features = list(
            mla_data_schema[mla_data_schema.marketing_indicator == 1].name
        )
        return self

    @log_decorator
    def read_holidays(self, holidays_data_location):
        self.holidays_df = dataframe_reader(holidays_data_location, self.spark)
        # Make sure the holiday columns are all integer
        for column_name in self.holiday_features:
            self.holidays_df = self.holidays_df.withColumn(
                column_name, col(column_name).cast("int")
            )

    @log_decorator
    def read_raw_data(self, raw_data_location):
        mla_training_data = dataframe_reader(raw_data_location, self.spark)
        mla_training_data = mla_training_data.withColumn(
            "date", (f.col("conversionTimestamp") / 1000).cast("timestamp")
        )
        mla_training_data = mla_training_data.withColumn("date", f.col("date").cast("date"))
        joined_df = mla_training_data.join(
            broadcast(self.holidays_df),
            mla_training_data.date == self.holidays_df.calendar_date,
            "left",
        )

        # Add the conversionIndicator column only if it doesn't already exist
        if "conversionIndicator" not in joined_df.columns:
            joined_df = joined_df.withColumn(
                "conversionIndicator",
                f.when(f.col("conversionType") == "orderItem", 1).otherwise(0),
            )

        if "adCustomerOrderItemIdCount" not in joined_df.columns:
            joined_df = joined_df.withColumn(
                "adCustomerOrderItemIdCount",
                f.when(f.col("conversionType") == "orderItem", 1).otherwise(0),
            )

        # Persist data
        joined_df.select(*self.cols_to_keep).cache()
        self.mla_training_data = joined_df

    def get_processed_data(self):
        return self.mla_training_data


class FeatureEngineering(HasConfigParam):
    def __init__(self, data_preprocessor: DataPreprocessing, model_type, do_bucketize):
        super().__init__()
        self.model_type = model_type
        self.do_bucketize = do_bucketize
        self.data_preprocessor = data_preprocessor

    @property
    def other_user_impression_features(self):
        return ["aap_out_imp_1", "aap_out_imp_7"]

    @property
    def customer_features(self):
        return ["customerIsPrime", "customerPropensity_dummy", "customerRfm_dummy"]

    @property
    def day_of_week_features(self):
        return ["day_of_week_dummy"]

    @property
    def imp_features(self):
        treatment_ada, treatment_aap = self.getTreatments
        return [
            f"{treatment}_dummy"
            for treatment in [treatment_ada, treatment_aap]
            if self.do_bucketize
        ] or [treatment_ada, treatment_aap]

    @property
    def get_features_list(self):
        features = (
            self.data_preprocessor.traffic_features
            + self.data_preprocessor.holiday_features
            + self.other_user_impression_features
            + self.customer_features
            + self.day_of_week_features
        )
        return (
            ["intercept"] + features + self.imp_features
            if self.model_type == "regression"
            else features
        )

    @staticmethod
    def create_day_of_week_stages():
        string_indexer = StringIndexer(inputCol="day_of_week", outputCol="day_of_week_indexed")
        one_hot_encoder = OneHotEncoder(
            inputCols=["day_of_week_indexed"], outputCols=["day_of_week_dummy"], dropLast=True
        )
        return [string_indexer, one_hot_encoder]

    @staticmethod
    def create_user_feature_encoder():
        return OneHotEncoder(
            inputCols=["customerPropensityBucket", "customerRfm_indexed"],
            outputCols=["customerPropensity_dummy", "customerRfm_dummy"],
            dropLast=True,
        )

    def create_encoder_stages(self):
        if not self.do_bucketize:
            return []
        treatment_ada, treatment_aap = self.getTreatments
        return [
            OneHotEncoder(
                inputCols=[f"{treatment}_bucket"], outputCols=[f"{treatment}_dummy"], dropLast=True
            )
            for treatment in [treatment_ada, treatment_aap]
        ]

    @log_decorator
    def create_featurization_pipeline(self):
        stages = (
            [
                VariableTransformer(
                    configParam=self.getConfigParam, do_bucketize=self.do_bucketize
                ),
                IndicatorTransformer(
                    inputCols=self.data_preprocessor.traffic_features, threshold=0
                ),
                IndicatorTransformer(
                    inputCols=self.data_preprocessor.holiday_features, threshold=0
                ),
                StringIndexer(inputCol="customerRfm", outputCol="customerRfm_indexed"),
                self.create_user_feature_encoder(),
            ]
            + self.create_day_of_week_stages()
            + self.create_encoder_stages()
            + [StringIndexer(inputCol="treatment_type", outputCol="treatment_type_indexed")]
            + [VectorAssembler(inputCols=self.get_features_list, outputCol="features")]
        )

        self.featurization_pipeline = Pipeline(stages=stages)
        return self

    @log_decorator
    def create_featurized_data(self):
        mla_training_data = self.data_preprocessor.mla_training_data
        self.pipeline_model = self.featurization_pipeline.fit(mla_training_data)
        self.featurized_data = self.pipeline_model.transform(mla_training_data)
        return self

    @property
    def get_impression_features_list(self):
        return self.imp_features

    def get_marketing_index(self):
        """
        Function to identify marketing features and create a mask for non-marketing features.

        Returns:
            nm_mask (numpy array): A mask for non-marketing features with 1 for non-marketing features
             and 0 for marketing features.
        """

        # Retrieve the feature metadata from the featurized data schema.
        feature_metadata = self.featurized_data.schema["features"].metadata["ml_attr"]

        # Filter for important features that include 'ada_in_imp' or 'aap_in_imp' in their names.
        imp_features_list = self.get_impression_features_list

        # Combine marketing features and important features
        marketing_features = self.data_preprocessor.marketing_features + imp_features_list

        # Get the index of marketing features
        marketing_index = [
            feature["idx"]
            for feature in feature_metadata["attrs"]["numeric"]
            if feature["name"] in marketing_features
        ]

        # Initialize a numpy array for non-marketing mask
        nm_mask = np.ones(feature_metadata["num_attrs"])

        # Set marketing features index to 0 in the non-marketing mask
        nm_mask[marketing_index] = 0

        return nm_mask

    def get_featurized_data(self):
        return self.featurized_data

    def get_featurization_pipeline(self):
        return self.featurization_pipeline

    def get_string_indexer_labels(self):
        stage = self.pipeline_model.stages[-2]

        if isinstance(stage, StringIndexerModel):
            return stage.labels
        else:
            raise ValueError(
                "Expected stage to be a StringIndexerModel, but found a different type."
            )

    def get_string_indexer_treatments_names(self):
        labels = self.get_string_indexer_labels()
        treatments_names = [label for label in labels if label != "no_impression"]
        if treatments_names:
            return treatments_names
        else:
            raise ValueError("No treatments found from the StringIndexerModel.")
