from pyspark import keyword_only
import pyspark.sql.functions as f
from pyspark.mllib.linalg import Vectors as MLLibVectors
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.ml import Model
from pyspark.ml.param.shared import (
    HasFeaturesCol,
    HasLabelCol,
    HasWeightCol,
    HasPredictionCol,
)
from wildebeest_purchase_model.utilities.utils import (
    HasModelArtifact,
    HasNmFeaturesCol,
    get_ols_default_model_artifact,
)
from wildebeest_purchase_model.score.scoring_udf import (
    get_ci_delta_method,
    scoring_output_type,
)
import pandas as pd
import numpy as np
from pyspark.ml import Estimator
from pyspark.ml.util import DefaultParamsWritable, DefaultParamsReadable

from wildebeest_purchase_model.inputs.data_processing import FeatureEngineering


class WildebeestPurchaseRegression(
    Estimator,
    HasFeaturesCol,
    HasLabelCol,
    HasWeightCol,
    DefaultParamsWritable,
    DefaultParamsReadable,
):
    @keyword_only
    def __init__(self, feature_engineer: FeatureEngineering = None):
        super(WildebeestPurchaseRegression, self).__init__()
        self.feature_engineer = feature_engineer
        self._setDefault(featuresCol="features")
        self._setDefault(labelCol="conversionIndicator")
        self._setDefault(weightCol="purchase_weight")

    @keyword_only
    def setParams(self, featuresCol=None, labelCol=None, weightCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setFeaturesCol(self, featuresCol):
        return self._set(featuresCol=featuresCol)

    def setWeightCol(self, weightCol):
        return self._set(weightCol=weightCol)

    def setLabelCol(self, labelCol):
        return self._set(labelCol=labelCol)

    def _fit(self, dataset):
        # The OLS model fitting - gnuA and gnuB
        # Prepare data: persist the minimal data we need
        imp_features_list = self.feature_engineer.get_impression_features_list
        features_col = self.getFeaturesCol()
        weight_col = self.getWeightCol()
        label_col = self.getLabelCol()
        lean_dataset = dataset.select(label_col, features_col, weight_col)
        lean_dataset.cache()
        # Perform the calculation using distributed linear algebra
        xtx_mat = (
            RowMatrix(
                lean_dataset.rdd.map(
                    lambda x: MLLibVectors.sparse(
                        x[features_col].size,
                        x[features_col].indices,
                        x[features_col].values * np.sqrt(x[weight_col]),
                    )
                )
            )
            .computeGramianMatrix()
            .toArray()
        )
        xty_vec = lean_dataset.rdd.map(
            lambda x: MLLibVectors.sparse(
                x[features_col].size,
                x[features_col].indices,
                x[features_col].values * x[label_col] * x[weight_col],
            ).toArray()
        ).sum()

        # Coefficient estimates - This is a numerically better way to get coefficient estimates than invxtx_mat.dot(xty_vec)
        b_vec = np.linalg.solve(xtx_mat, xty_vec)

        # inverse of XTX - This is a required for confidence interval of lift
        invxtx_mat = np.linalg.inv(xtx_mat)

        # Robust Covariance Matrix - The sandwich formula
        meat_matrix = (
            RowMatrix(
                lean_dataset.rdd.map(
                    lambda x: MLLibVectors.sparse(
                        x[features_col].size,
                        x[features_col].indices,
                        x[features_col].values
                        * (x[label_col] - np.dot(x[features_col].toArray(), b_vec))
                        * x[weight_col],
                    )
                )
            )
            .computeGramianMatrix()
            .toArray()
        )
        robust_cov = invxtx_mat.dot(meat_matrix).dot(invxtx_mat)

        # Get restricted regression matrix - Algebra can be further simplified.
        k = b_vec.size
        imp_features_list_len = len(imp_features_list)
        columns_to_zero_out = k - imp_features_list_len
        inv_sub_invxtx_mat = np.linalg.inv(
            invxtx_mat[(columns_to_zero_out):k, (columns_to_zero_out):k]
        )
        proj_mat = np.block(
            [
                [
                    np.zeros((columns_to_zero_out, columns_to_zero_out)),
                    np.zeros((columns_to_zero_out, imp_features_list_len)),
                ],
                [np.zeros((imp_features_list_len, columns_to_zero_out)), inv_sub_invxtx_mat],
            ]
        )
        V_mat = np.eye(k) - invxtx_mat.dot(proj_mat)

        # GnuA model coefficients
        b0_vec = V_mat.dot(b_vec)
        # Ensure the last two coefficients are zero. They can be off by machine precision from the calculation.
        b0_vec[columns_to_zero_out:k] = 0
        # Pack model artifacts: convert the numpy arrays to list for JSON serialization
        model_artifact = {
            "gnuA_coeff": b0_vec.tolist(),
            "gnuB_coeff": b_vec.tolist(),
            "robust_cov": robust_cov.tolist(),
            "V_mat": V_mat.tolist(),
            "features_idx_map": self.get_features_idx_map(lean_dataset),
        }
        return WildebeestPurchaseRegressionModel(
            modelArtifact=model_artifact, featuresCol=self.getFeaturesCol()
        )

    def get_features_idx_map(self, dataset):
        feature_attr = dataset.schema[self.getFeaturesCol()].metadata["ml_attr"]["attrs"]
        features_idx_map = feature_attr["numeric"] + feature_attr["binary"]
        features_idx_map = sorted(features_idx_map, key=lambda x: x["idx"])
        return features_idx_map


class WildebeestPurchaseRegressionModel(
    Model,
    HasModelArtifact,
    HasFeaturesCol,
    HasNmFeaturesCol,
    HasPredictionCol,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    @keyword_only
    def __init__(
        self,
        modelArtifact=get_ols_default_model_artifact(),
        featuresCol=None,
        nmFeaturesCol=None,
        predictionCol=None,
    ):
        super(WildebeestPurchaseRegressionModel, self).__init__()
        self._setDefault(featuresCol="features")
        kwargs = self._input_kwargs
        self._set(**kwargs)

    @keyword_only
    def setParams(
        self,
        modelArtifact=get_ols_default_model_artifact(),
        featuresCol=None,
        nmFeaturesCol=None,
        predictionCol=None,
    ):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setModelArtifact(self, modelArtifact):
        return self._set(modelArtifact=modelArtifact)

    def setFeaturesCol(self, featuresCol):
        return self._set(featuresCol=featuresCol)

    def setNmFeaturesCol(self, nmFeaturesCol):
        return self._set(nmFeaturesCol=nmFeaturesCol)

    def setPredictionCol(self, predictionCol):
        return self._set(predictionCol=predictionCol)

    def unpack_model_artifact(self):
        # Unpack the artifacts
        modelArtifact = self.getModelArtifact()
        self.features_idx_map = modelArtifact["features_idx_map"]
        self.gnuA_coeff = np.asarray(modelArtifact["gnuA_coeff"])
        self.gnuB_coeff = np.asarray(modelArtifact["gnuB_coeff"])
        self.robust_cov = np.asarray(modelArtifact["robust_cov"])
        self.V_mat = np.asarray(modelArtifact["V_mat"])

    def _transform(self, data_frame):
        # Model scoring and confidence interval calculation
        self.unpack_model_artifact()
        b = self.gnuB_coeff
        b0 = self.gnuA_coeff
        cov = self.robust_cov
        vcov_mat = self.V_mat.dot(self.robust_cov)
        vcovvt_mat = vcov_mat.dot(self.V_mat.T)
        get_ci_delta_method_udf = f.udf(
            lambda x, y: get_ci_delta_method(x, y, b, b0, cov, vcov_mat, vcovvt_mat),
            scoring_output_type,
        )
        scored_data = data_frame.withColumn(
            self.getPredictionCol(),
            get_ci_delta_method_udf(self.getFeaturesCol(), self.getNmFeaturesCol()),
        )
        return scored_data

    def get_model_summary(self):
        self.unpack_model_artifact()
        # Create model summary to be output for persistence
        vcov_mat = self.V_mat.dot(self.robust_cov)
        vcovvt_mat = vcov_mat.dot(self.V_mat.T)
        model_summary = pd.DataFrame(self.features_idx_map)
        model_summary["gnuA_coeff"] = self.gnuA_coeff
        model_summary["gnuA_robustse"] = np.sqrt(np.diag(vcovvt_mat))
        model_summary["gnuB_coeff"] = self.gnuB_coeff
        model_summary["gnuB_robustse"] = np.sqrt(np.diag(self.robust_cov))
        return model_summary
