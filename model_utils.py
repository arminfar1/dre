from .estimators.doubly_robust import WildebeestDoublyRobustEstimation
from .estimators.ols import WildebeestPurchaseRegression
from .inputs.data_processing import FeatureEngineering
from .utilities.utils import HasConfigParam


class ModelBuilder(HasConfigParam):
    """
    A class that builds a linear regression model or a doubly robust estimation model.
    """

    def __init__(self):
        """
        Initializes the ModelBuilder class by setting up configuration parameters and model.
        """
        HasConfigParam.__init__(self)
        self.dre_fitted_model = None
        self.dre = None
        self.wildebeest_purchase_regression_model = None
        self.configParam = self.getConfigParam

    def build_linear_model(self, feature_engineer: FeatureEngineering):
        """
        Builds a linear regression model using the specified feature engineering process.

        Parameters:
            feature_engineer (FeatureEngineering): An instance of FeatureEngineering that provides
            the necessary features for the regression model.

        Returns:
            WildebeestPurchaseRegression: An instance of the regression model.
        """
        self.wildebeest_purchase_regression_model = WildebeestPurchaseRegression(
            feature_engineer=feature_engineer
        )
        if not self.configParam["weight"]:
            self.wildebeest_purchase_regression_model.setWeightCol("intercept")
        return self.wildebeest_purchase_regression_model

    def build_doubly_robust_model(
        self, features_list, labels, treatments_types, propensity_model_type
    ):
        """
        Builds a doubly robust estimation model with the given configuration.

        Parameters:
            features_list (list): A list of strings representing the feature column names.
            labels (list): A list of strings representing the label column names for different treatments.
            treatments_types (list): A list of strings representing the types of treatments to evaluate.
            propensity_model_type (str): The type of model to use for propensity score estimation.

        Returns:
            WildebeestDoublyRobustEstimation: An instance of the doubly robust estimation model.
        """
        self.dre = WildebeestDoublyRobustEstimation(
            features_list=features_list,
            labels=labels,
            treatments_types=treatments_types,
            propensity_model_type=propensity_model_type,
        )
        return self.dre

    def get_regression_model(self):
        return self.wildebeest_purchase_regression_model

    def get_dre_model(self):
        return self.dre
