import pyspark.sql.functions as f
from pyspark.sql.types import FloatType
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.sql.functions import col, lit
from pyspark.sql import DataFrame
import functools
import itertools
import time
import datetime
import boto3
import re
import pandas as pd


def get_config_param():
    # Now just hard coded. Could be imported from a JSON config file.
    return {
        "treatments": {"aap_num_days": 14, "ada_num_days": 14},
        "imp_threshold": 10,
        "weight": True,
    }


def get_ols_default_model_artifact():
    return {
        "gnuA_coeff": None,
        "gnuB_coeff": None,
        "robust_cov": None,
        "V_mat": None,
        "features_idx_map": None,
    }


def get_second_element():
    return f.udf(lambda v: float(v[1]), FloatType())


def log_decorator(func):
    # A very simple logging decorator to track steps. Can be replaced by prod logger.
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f"[{datetime.datetime.now().ctime()}] Starting the step {func.__qualname__!r}...")
        result = func(*args, **kwargs)
        end_time = time.time()
        run_time = end_time - start_time
        print(
            f"[{datetime.datetime.now().ctime()}] Step {func.__qualname__!r} is finished in {run_time:.2f} seconds."
        )
        return result

    return wrapper


def set_columns_to_zero(df, zero_features):
    # Set the specified columns to zero in the DataFrame
    for feature, value in zero_features.items():
        df = df.withColumn(feature, f.lit(value))
    return df


def flatten_dict(d):
    """Flatten a nested dictionary."""
    def expand(key, value):
        if isinstance(value, dict):
            return [(key + ' ' + k, v) for k, v in flatten_dict(value).items()]
        else:
            return [(key, value)]
    items = [item for k, v in d.items() for item in expand(k, v)]
    return dict(items)


def read_json_from_s3(s3_path):
    # Read json file in S3 as pandas dataframe
    uri_header, bucket_name, prefix = parse_uri(s3_path)
    s3 = boto3.resource("s3")
    json_obj = s3.Bucket(bucket_name).Object(prefix).get()["Body"].read().decode("utf-8")
    json_df = pd.read_json(json_obj, lines=True)
    return json_df


def dataframe_reader(path, spark_session):
    # Flexible dataframe reader
    supported_format_list = ["csv", "parquet", "json"]
    uri_header, bucket_name, prefix = parse_uri(path)
    s3_file_iterator = get_s3_file_iterator(path, spark_session)
    # Figure out the data format and get the first single file name
    supported_format_found = False
    while not supported_format_found and s3_file_iterator.hasNext():
        file_rawpath = s3_file_iterator.next().getPath().toUri().getRawPath()
        file_format_check = [file_rawpath.endswith(x) for x in supported_format_list]
        supported_format_found = any(file_format_check)
    single_file_path = f"{uri_header}{bucket_name}/{file_rawpath}"
    file_format = list(itertools.compress(supported_format_list, file_format_check))[0]
    # Get number of executors
    n_executors = spark_session.sparkContext.defaultParallelism
    min_partition = n_executors * 2
    # Call spark dataframe reader for respective formats with single file schema inference.
    if file_format == "csv":
        small_data = spark_session.read.csv(single_file_path, header=True)
        data_schema = small_data.schema
        df = spark_session.read.csv(path, schema=data_schema, header=True)
    elif file_format == "json":
        small_data = spark_session.read.option("multiline", "true").json(single_file_path)
        data_schema = small_data.schema
        df = spark_session.read.json(path, schema=data_schema)
    elif file_format == "parquet":
        small_data = spark_session.read.parquet(single_file_path)
        data_schema = small_data.schema
        df = spark_session.read.parquet(path, schema=data_schema)
    else:
        raise "Unsupported file format!"
    # Ensure the number of partition is at least min_partition
    if df.rdd.getNumPartitions() < min_partition:
        df = df.repartition(min_partition)
    return df


def get_s3_file_iterator(path, spark_session):
    sc = spark_session.sparkContext
    stripped_path = re.sub("/\*.*", "", path)
    java_path = sc._jvm.java.net.URI.create(stripped_path)
    hadoop_path = sc._jvm.org.apache.hadoop.fs.Path(stripped_path)
    hadoop_fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(
        java_path, sc._jvm.org.apache.hadoop.conf.Configuration()
    )
    s3_file_iterator = hadoop_fs.listFiles(hadoop_path, True)
    return s3_file_iterator


def parse_uri(path):
    uri_header = re.findall(".*//", path)[0]
    bucket_name = re.sub(uri_header, "", path).split("/")[0]
    prefix = re.sub(f"{uri_header}{bucket_name}/", "", path)
    return uri_header, bucket_name, prefix


def split_train_test(dataset: DataFrame, rate=0.2):
    train, test = dataset.randomSplit([1 - rate, rate], seed=12345)
    return train, test


def balanced_split_train_test(
    dataset: DataFrame, label_col: str, rate=0.2
) -> (DataFrame, DataFrame):
    """
    Split the dataset into train and test sets with a balanced representation of classes.

    Parameters:
    - dataset: The input DataFrame.
    - label_col: The column name of the label.
    - rate: The proportion of the dataset to include in the test split.

    Returns:
    - train: The training set.
    - test: The test set.
    """

    # Get distinct classes
    classes = dataset.select(label_col).distinct().rdd.flatMap(lambda x: x).collect()

    train_dfs = []
    test_dfs = []

    for class_val in classes:
        # Filter dataset for each class
        class_dataset = dataset.filter(f.col(label_col) == class_val)

        # Split the class dataset into train and test
        class_train, class_test = class_dataset.randomSplit([1 - rate, rate])

        # Append to the list of train and test dataframes
        train_dfs.append(class_train)
        test_dfs.append(class_test)

    # Union all the train and test dataframes
    train = functools.reduce(DataFrame.unionByName, train_dfs)
    test = functools.reduce(DataFrame.unionByName, test_dfs)
    return train, test


def filter_by_range(dataset, col_to_filter, lower_bound=0.05, upper_bound=0.95):
    """
    Filter rows based on values in a column not being within a specified range.

    Args:
        dataset: The input DataFrame.
        col_to_filter: The column to check values.
        lower_bound: The lower bound of the range.
        upper_bound: The upper bound of the range.

    Returns:
        DataFrame with rows filtered.
    """
    return dataset.filter((col(col_to_filter) >= lower_bound) & (col(col_to_filter) <= upper_bound))


def winsorize_column(dataset, col_name, lower_quantile=0.01, upper_quantile=0.99):
    # Calculate the quantiles
    lower_bound, upper_bound = dataset.approxQuantile(
        col_name, [lower_quantile, upper_quantile], 0.01
    )
    print(
        "winsorize_column lower_bound: ",
        lower_bound,
        " winsorize_column upper_bound: ",
        upper_bound,
    )
    # Winsorize the column
    dataset = dataset.withColumn(
        col_name,
        f.when(f.col(col_name) < lower_bound, lower_bound)
        .when(f.col(col_name) > upper_bound, upper_bound)
        .otherwise(f.col(col_name)),
    )
    return dataset


def cap_values_below_zero(dataset, col_name, lower_cap=0):
    return dataset.withColumn(
        col_name, f.when(f.col(col_name) < lower_cap, lower_cap).otherwise(f.col(col_name))
    )


def set_columns_to_zeros(dataset, columns):
    select_expr = [
        lit(0).alias(col_name) if col_name in columns else col_name for col_name in dataset.columns
    ]
    return dataset.select(*select_expr)


def generate_paths(base, *path_components):
    """Generate model paths."""
    return f"{base}/{'/'.join(path_components)}/"


class HasConfigParam(Params):
    # This is the parameter name
    configParam = Param(Params._dummy(), "configParam", "model training configuration parameters")

    def __init__(self):
        super(HasConfigParam, self).__init__()
        self.config_data = get_config_param()
        self._setDefault(
            configParam="defaultConfig"
        )  # Here, "defaultConfig" is just a string key/name

    @property
    def getConfigParam(self):
        return self.config_data

    @property
    def getTreatments(self):
        treatments = self.getConfigParam["treatments"]
        ada = self._get_treatment_key(treatments, "ada")
        aap = self._get_treatment_key(treatments, "aap")

        if not ada or not aap:
            raise ValueError("Required treatments not found in the configuration.")

        # Construct the treatment keys
        treatment_ada = f"{ada}_in_imp_{treatments['ada_num_days']}"
        treatment_aap = f"{aap}_in_imp_{treatments['aap_num_days']}"

        return treatment_ada, treatment_aap

    @staticmethod
    def _get_treatment_key(treatments, substring):
        """
        Fetch the first key containing the specified substring.
        Returns None if no matching key is found.
        """
        for treatment in treatments.keys():
            if substring in treatment:
                return treatment
        return None


class HasThreshold(Params):
    threshold = Param(
        Params._dummy(),
        "threshold",
        "vector of threshold values(or a scalar) for constructing indicator variables",
    )

    def __init__(self):
        super(HasThreshold, self).__init__()
        self._setDefault(threshold=0)

    def getThreshold(self):
        return self.getOrDefault(self.threshold)


class HasNmFeaturesCol(Params):
    # Utility mixin class for model param
    nmFeaturesCol = Param(
        Params._dummy(),
        "nmFeaturesCol",
        "No marketing features column name.",
        typeConverter=TypeConverters.toString,
    )

    def __init__(self):
        super(HasNmFeaturesCol, self).__init__()
        self._setDefault(nmFeaturesCol="nm_features")

    def getNmFeaturesCol(self):
        return self.getOrDefault(self.nmFeaturesCol)


class HasModelArtifact(Params):
    # Utility mixin class for model param
    modelArtifact = Param(
        Params._dummy(),
        "modelArtifact",
        "model artifacts from training that are required for the scoring stage",
    )

    def __init__(self) -> None:
        super(HasModelArtifact, self).__init__()
        self._setDefault(modelArtifact=get_ols_default_model_artifact())

    def getModelArtifact(self):
        return self.getOrDefault(self.modelArtifact)


def hasColumn(df, colname):
    return colname in df.columns
