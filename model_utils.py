from .estimators.doubly_robust import WildebeestDoublyRobustEstimation
from .estimators.ols import WildebeestPurchaseRegression
from .inputs.data_processing import FeatureEngineering
from .utilities.utils import HasConfigParam


class ModelBuilder(HasConfigParam):
    def __init__(self):
        HasConfigParam.__init__(self)
        self.dre_fitted_model = None
        self.dre = None
        self.wildebeest_purchase_regression_model = None
        self.configParam = self.getConfigParam

    def build_linear_model(self, feature_engineer: FeatureEngineering):
        self.wildebeest_purchase_regression_model = WildebeestPurchaseRegression(
            feature_engineer=feature_engineer
        )
        if not self.configParam["weight"]:
            self.wildebeest_purchase_regression_model.setWeightCol("intercept")
        return self.wildebeest_purchase_regression_model

    def build_doubly_robust_model(
        self, features_list, labels, treatments_types, propensity_model_type
    ):
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
