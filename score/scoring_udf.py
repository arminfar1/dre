from pyspark.sql.types import StructField, StructType, DoubleType
import numpy as np

scoring_output_type = StructType(
    [
        StructField("liftWb", DoubleType(), True),
        StructField("liftWbStandardError", DoubleType(), True),
        StructField("liftMlaGnuA", DoubleType(), True),
        StructField("liftMlaGnuB", DoubleType(), True),
        StructField("pDenomMlaGnuA", DoubleType(), True),
        StructField("pDenomMlaGnuB", DoubleType(), True),
        StructField("pNumMlaGnuA", DoubleType(), True),
        StructField("pNumMlaGnuB", DoubleType(), True),
    ]
)


# Functions to be applied for each row of scoring data
def get_ci_delta_method(x, y, b, b0, cov, vcov, vcovvt):
    features = x.toArray()
    nm_features = y.toArray()
    cov_mat = _get_cov_matrix_delta_method(features, nm_features, cov, vcov, vcovvt)
    predictions = _get_predictions(features, nm_features, b, b0, cov_mat)
    return predictions


def _get_predictions(features_nparray, nm_features_nparray, b, b0, cov_mat):
    # Define the probabilities
    p_a = float(features_nparray.dot(b0))
    p_b = float(features_nparray.dot(b))
    p_a_nm = float(nm_features_nparray.dot(b0))
    p_b_nm = float(nm_features_nparray.dot(b))
    # Lifts: bounded by 0 and 1
    lift_a = min(max(1 - p_a_nm / p_a, 0), 1)
    lift_b = min(max(1 - p_b_nm / p_b, 0), 1)
    lift_wb = min(max(lift_b - lift_a, 0), 1)
    # Delta method to get standard error
    jacobian = np.array([1 / p_a, -p_a_nm / p_a**2, -1 / p_b, p_b_nm / p_b])
    lift_se = float(np.sqrt(jacobian.T.dot(cov_mat).dot(jacobian)))
    return [lift_wb, lift_se, lift_a, lift_b, p_a, p_b, p_a_nm, p_b_nm]


def _get_cov_matrix_delta_method(x, x0, cov, vcov, vcovvt):
    # Function to get the covariance matrix for the delta method
    v11 = x0.T.dot(vcovvt).dot(x0)
    v12 = x0.T.dot(vcovvt).dot(x)
    v13 = x0.T.dot(vcov).dot(x0)
    v14 = x0.T.dot(vcov).dot(x)
    v22 = x.T.dot(vcovvt).dot(x)
    v23 = x.T.dot(vcov).dot(x0)
    v24 = x.T.dot(vcov).dot(x)
    v33 = x0.T.dot(cov).dot(x0)
    v34 = x0.T.dot(cov).dot(x)
    v44 = x.T.dot(cov).dot(x)
    cov_mat = np.array(
        [[v11, v12, v13, v14], [v12, v22, v23, v24], [v13, v23, v33, v34], [v14, v24, v34, v44]]
    )
    return cov_mat
