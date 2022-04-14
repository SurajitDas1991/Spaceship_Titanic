from sklearn.impute import KNNImputer
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
# from sklearn.linear_model import Ridge
# from sklearn.linear_model import Lasso
import catboost as cb
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from datetime import datetime


class GetHyperParametersForModels:
    def __init__(self):
        pass

    def hyperparams_random_forest(self, X, y):

        # Define the hyperparameter configuration space
        rf_params = {
            "n_estimators": [10, 50,100],
            "max_features": ["sqrt", 0.5],
            "max_depth": [15, 20, 30,100],
            "min_samples_leaf": [2,8,16],
            "bootstrap": [True, False],
            "criterion": ["gini", "entropy"],
        }
        clf = RandomForestClassifier(random_state=0)
        cv = RepeatedKFold(n_splits=3, n_repeats=3, random_state=0)
        grid = GridSearchCV(
            clf,
            rf_params,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=2,
        )
        grid.fit(X, y)
        return grid.best_params_

    def hyperparams_SVR(self, X, y):
        # Define the hyperparameter configuration space
        rf_params = {
            "C": [1],
            "kernel": ["poly", "rbf", "sigmoid","linear"],
            "degree": [1,2,3],
        }
        clf = SVC()
        cv = RepeatedKFold(n_splits=3, n_repeats=3, random_state=0)
        grid = GridSearchCV(
            clf,
            rf_params,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=2,
        )
        grid.fit(X, y)
        return grid.best_params_

    def hyperparams_KNN(self, X, y):
        # Define the hyperparameter configuration space
        rf_params = {"n_neighbors": [2, 3, 5, 10,15]}
        clf = KNeighborsClassifier()
        cv = RepeatedKFold(n_splits=3, n_repeats=3, random_state=0)
        grid = GridSearchCV(
            clf,
            rf_params,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=2,
        )

        grid.fit(X, y)
        return grid.best_params_

    # def hyperparams_Lasso(self, X, y):
    #     # Define the hyperparameter configuration space
    #     rf_params = {"alpha": [0.001, 0.01, 0.1, 1, 100, 1000]}
    #     clf = Lasso()
    #     cv = RepeatedKFold(n_splits=3, n_repeats=3, random_state=0)
    #     grid = GridSearchCV(
    #         clf,
    #         rf_params,
    #         cv=cv,
    #         scoring="neg_root_mean_squared_error",
    #         n_jobs=-1,
    #         verbose=2,
    #     )
    #     grid.fit(X, y)
    #     return grid.best_params_

    # def hyperparams_ridge(self, X, y):
    #     # Define the hyperparameter configuration space
    #     rf_params = {"alpha": [0.001, 0.01, 0.1, 1, 100, 1000]}
    #     clf = Ridge()
    #     cv = RepeatedKFold(n_splits=3, n_repeats=3, random_state=0)
    #     grid = GridSearchCV(
    #         clf,
    #         rf_params,
    #         cv=cv,
    #         scoring="neg_root_mean_squared_error",
    #         n_jobs=-1,
    #         verbose=2,
    #     )
    #     #self.logger.info("Tuning Started for Ridge regression")
    #     grid.fit(X, y)
    #     return grid.best_params_

    def hyperparams_adaboost(self, X, y):
        # Define the hyperparameter configuration space
        rf_params = {
            "n_estimators": np.arange(10, 300, 10),
            "learning_rate": [0.01, 0.1, 0.5,1],
        }
        clf = AdaBoostClassifier()
        cv = RepeatedKFold(n_splits=3, n_repeats=3, random_state=0)
        grid = GridSearchCV(
            clf,
            rf_params,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=2,
        )
        #self.logger.info("Tuning Started for Adaboost")
        grid.fit(X, y)
        return grid.best_params_

    def hyperparams_gradientboost(self, X, y):
        rf_params = {
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.9, 0.4, 0.1],
            "n_estimators": [100, 500,1000],
            "max_depth": [4, 6, 8],
        }
        clf = GradientBoostingClassifier()
        cv = RepeatedKFold(n_splits=3, n_repeats=3, random_state=0)
        grid = GridSearchCV(
            clf,
            rf_params,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=2,
        )
        #self.logger.info("Tuning Started for GradientBoost")
        grid.fit(X, y)
        return grid.best_params_

    def hyperparams_extratrees(self, X, y):
        rf_params = {
            "n_estimators": [10, 50, 100],
            "min_samples_split": [2, 6, 10],
            "max_depth": [2,4,8],
        }
        clf = ExtraTreesClassifier()
        cv = RepeatedKFold(n_splits=3, n_repeats=3, random_state=0)
        grid = GridSearchCV(
            clf,
            rf_params,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=2,
        )
        #self.logger.info("Tuning Started for ExtraTreesRegressor")
        grid.fit(X, y)
        return grid.best_params_

    def hyperparams_xgboost(self, X, y):
        rf_params = {
            "n_estimators": [400, 500],
            "colsample_bytree": [0.7, 0.2,1],
            "max_depth": [5,20,40],
            "gamma": [0,1,3],
            # "eta": [0.1,0.03],
            "subsample": [0.7,0.1,1],
        }
        clf = xgb.XGBRFClassifier()
        cv = RepeatedKFold(n_splits=3, n_repeats=3, random_state=0)
        grid = GridSearchCV(
            clf,
            rf_params,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=2,
        )
        #self.logger.info("Tuning Started for XGBoost")
        grid.fit(X, y)
        return grid.best_params_

    def hyperparams_lightgbm(self, X, y):
        rf_params = {
            "learning_rate": [0.01, 0.01, 0.03],
            "boosting_type": ["gbdt", "dart", "goss"],
            "objective": ["regression"],
            "metric": ["auc"],
            "num_leaves": [20, 40],
            "reg_alpha": [0.01, 0.03],
            "max_depth": [10, 20],
        }
        clf = lgb.LGBMClassifier()
        cv = RepeatedKFold(n_splits=3, n_repeats=3, random_state=0)
        grid = GridSearchCV(
            clf,
            rf_params,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=2,
        )
        #self.logger.info("Tuning Started for LightGBM")
        grid.fit(X, y)
        return grid.best_params_

    def hyperparams_catboost(self, X, y):
        rf_params = {
            "iterations": [100, 150],
            "learning_rate": [0.03, 0.1],
            "depth": [2, 4, 6],
            "l2_leaf_reg": [0.2, 1, 3],
        }
        clf = cb.CatBoostClassifier()
        cv = RepeatedKFold(n_splits=3, n_repeats=3, random_state=0)
        grid = GridSearchCV(
            clf,
            rf_params,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=2,
        )
        #self.logger.info("Tuning Started for CatBoost")
        grid.fit(X, y)
        return grid.best_params_

    def hyperparams_stacking(self, X, y):
        rf_params = {"final_estimator__C": [1]}
        estimators = [
            ("rf", RandomForestClassifier(n_estimators=10, random_state=42)),
            ("adareg", AdaBoostClassifier()),
        ]
        clf = StackingClassifier(estimators, final_estimator=SVC())
        cv = RepeatedKFold(n_splits=3, n_repeats=3, random_state=0)
        grid = GridSearchCV(
            clf,
            rf_params,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=2,
        )
        #self.logger.info("Tuning Started for Stacking")
        grid.fit(X, y)
        return grid.best_params_
