import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories,get_appended_path,wrap_labels
from src.utils import hyperparameter_tuning,file_operations
import random
from sklearn.metrics import r2_score,roc_curve
from sklearn.ensemble import ( RandomForestClassifier,GradientBoostingClassifier,
ExtraTreesClassifier,
AdaBoostClassifier,
StackingClassifier)
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import Ridge
# from sklearn.linear_model import Lasso
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, confusion_matrix
from utils import *
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV

STAGE = "Training phase"

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )

class Model_Finder:
    def __init__(self,config):
        self.hyperparametertuning = hyperparameter_tuning.GetHyperParametersForModels()
        self.prediction_output_file=get_appended_path(config["artifacts"]["PREDICTION_OUTPUT_FILE"])
        self.visualization_path=get_appended_path(config["artifacts"]["VISUALIZATION_PATH"])
        self.models_path=get_appended_path(config['artifacts']['TRAINED_MODEL_DIR'])
        create_directories([self.prediction_output_file,self.visualization_path,self.models_path])
        # self.linearReg = LinearRegression()
        # self.RandomForestReg = RandomForestRegressor()
        self.models_objects = []

    def mean_square_error_for_models(self, x_test, y_test, dict_model_results):

        sns.set()

        rmse_num = []
        for name, m in dict_model_results.items():
            # Cross validation of the model
            model = m["model"]
            mse = mean_squared_error(y_test, model.predict(x_test),squared=False)
            lst = [name,mse]
            rmse_num.append(lst)
        df_results = pd.DataFrame(rmse_num, columns=["model", "Root Mean Squared Error"])
        df_results.sort_values(by="Root Mean Squared Error", ascending=True, inplace=True)
        df_results.reset_index(inplace=True, drop=True)
        df_results.to_csv(
            os.path.join(
                self.prediction_output_file,
                f"RootMeanSquaredErrorResult.csv",
            ),
            header=True,
        )

        plt.figure(figsize=(20, 10))
        plt.xlabel("Root mean_square_error", fontsize=12)
        plt.ylabel("Model Type", fontsize=12)
        # plt.xticks(rotation=90, fontsize=12)
        plt.title("The Root mean_square_error Comparison")
        chart = sns.barplot(x="Root Mean Squared Error", y="model", data=df_results)
        wrap_labels(chart, 10)
        for container in chart.containers:
            chart.bar_label(container)
        plt.savefig(
            os.path.join(self.visualization_path, f"RootMeanSquareError.png")
        )

        # Get the best model name as per RMSE
        best_model_name = df_results.iloc[0]["model"]
        # models[best_model_name[0]]['model']
        model_name=df_results['model']
        best_model=None
        for name, m in dict_model_results.items():
            if name==best_model_name:
                best_model=m['model']
                break
        return best_model_name,best_model
        #Get the actual model




    def check_r2_squared_results(self, x_test, y_test, dict_model_results):
        # Model = model_fit(x_train, x_test, y_train, y_test)

        sns.set()
        R_square_num = []
        for name, m in dict_model_results.items():
            # Cross validation of the model
            model = m["model"]
            R_square = r2_score(y_test, model.predict(x_test))
            lst = [name, R_square]
            R_square_num.append(lst)
        df_results = pd.DataFrame(R_square_num, columns=["model", "R-Squared"])
        df_results.sort_values(by="R-Squared", ascending=False, inplace=True)
        df_results.reset_index(inplace=True, drop=True)
        df_results.to_csv(
            os.path.join(
                self.prediction_output_file,
                f"R-SquaredResultComparison_cluster.csv",
            ),
            header=True,
        )

        plt.figure(figsize=(20, 10))
        plt.xlabel("R Square Score", fontsize=12)
        plt.ylabel("Model Type", fontsize=12)
        # plt.xticks(rotation=90, fontsize=12)
        plt.title("The R Square Score Comparsion")
        chart = sns.barplot(x="R-Squared", y="model", data=df_results)
        wrap_labels(chart, 10)
        for container in chart.containers:
            chart.bar_label(container)
        plt.savefig(
            os.path.join(self.visualization_path, f"R Square Score_cluster.png")
        )

    def base_model_checks(self, train_x, train_y, x_test, y_test):
        # Create a dictionary with the model which will be tested
        models = {
            "StackingClassifier": {"model": StackingClassifier(estimators = [
            ("rf", RandomForestClassifier()),
            ("catreg", cb.CatBoostClassifier()),
        ],final_estimator=SVC()).fit(train_x, train_y)},
            "LogisticRegression": {"model": LogisticRegression().fit(train_x, train_y)},
            "KNN": {
                "model": KNeighborsClassifier(
                    **self.hyperparametertuning.hyperparams_KNN(train_x, train_y)
                ).fit(train_x, train_y)
            },
            # "Ridge": {"model": Ridge(**self.hyperparametertuning.hyperparams_ridge(train_x,train_y)).fit(train_x, train_y)},
            # "Lasso": {"model": Lasso(**self.hyperparametertuning.hyperparams_Lasso(train_x,train_y)).fit(train_x, train_y)},
            "SVC": {"model": SVC(**self.hyperparametertuning.hyperparams_SVR(train_x,train_y)).fit(train_x, train_y)},

            "Catboost": {"model": cb.CatBoostClassifier(**self.hyperparametertuning.hyperparams_catboost(train_x,train_y)).fit(train_x, train_y)},
            "RandomForest": {"model": RandomForestClassifier(**self.hyperparametertuning.hyperparams_random_forest(train_x,train_y)).fit(train_x, train_y)},
            "GradientBoost": {"model": GradientBoostingClassifier(**self.hyperparametertuning.hyperparams_gradientboost(train_x,train_y)).fit(train_x, train_y)},
            "XGBoost": {"model": xgb.XGBClassifier(**self.hyperparametertuning.hyperparams_xgboost(train_x,train_y)).fit(train_x, train_y)},
            "LightGBM": {"model": lgb.LGBMClassifier(**self.hyperparametertuning.hyperparams_lightgbm(train_x,train_y)).fit(train_x, train_y)},
            "AdaBoost": {"model": AdaBoostClassifier(**self.hyperparametertuning.hyperparams_adaboost(train_x,train_y)).fit(train_x, train_y)},
             "ExtraTrees": {"model": ExtraTreesClassifier(**self.hyperparametertuning.hyperparams_extratrees(train_x,train_y)).fit(train_x, train_y)}
        }

        # Use the 10-fold cross validation for each model
        # to get the mean validation accuracy and the mean training time
        for name, m in models.items():
            # Cross validation of the model
            model = m["model"]
            result = cross_validate(model, train_x, train_y, cv=2)

            # Mean accuracy and mean training time
            mean_val_accuracy = round(
                sum(result["test_score"]) / len(result["test_score"]), 4
            )
            mean_fit_time = round(sum(result["fit_time"]) / len(result["fit_time"]), 4)

            # Add the result to the dictionary witht he models
            m["val_accuracy"] = mean_val_accuracy
            m["Training time (sec)"] = mean_fit_time

            # Display the result
            print(
                f"{name:27} Mean accuracy using 10-fold cross validation: {mean_val_accuracy*100:.2f}% - mean training time {mean_fit_time} sec"
            )

        # Create a DataFrame with the results
        models_result = []

        for name, v in models.items():
            lst = [name, v["val_accuracy"], v["Training time (sec)"]]
            models_result.append(lst)

        df_results = pd.DataFrame(
            models_result,
            columns=["model", "val_accuracy", "Training time (sec)"],
        )
        df_results.sort_values(
            by="val_accuracy", ascending=False, inplace=True
        )
        df_results.reset_index(inplace=True, drop=True)
        df_results.to_csv(
            os.path.join(
                self.prediction_output_file,
                f"CrossValidationByvalAccuracyResults.csv",
            ),
            header=True,
        )

        plt.figure(figsize=(30, 5))
        chart = sns.barplot(x="model", y="val_accuracy", data=df_results)
        wrap_labels(chart, 10)
        for container in chart.containers:
            chart.bar_label(container)
        plt.title(
            "Mean Validation Accuracy for each Model\ny-axis between 0.6 and 1.0",
            fontsize=12,
        )
        plt.ylim(0.6, 1)
        plt.xlabel("Model", fontsize=12)
        plt.ylabel("val_accuracy", fontsize=12)
        # plt.xticks(rotation=90, fontsize=12)
        plt.savefig(
            os.path.join(
                self.visualization_path, f"CV_validationaccuracy.png"
            )
        )
        # plt.show()
        return self.check_metrics(x_test, y_test, models)
        # self.check_r2_squared_results(x_test, y_test, models)
        # return self.mean_square_error_for_models(x_test, y_test, models)

    def check_metrics(self, x_test, y_test, dict_model_results):
        # Model = model_fit(x_train, x_test, y_train, y_test)

        sns.set()
        score_lst = []
        for name, m in dict_model_results.items():
            # Cross validation of the model
            model = m["model"]
            pred=model.predict(x_test)
            score = roc_auc_score(y_test, pred)
            lst = [name, score]
            score_lst.append(lst)
        df_results = pd.DataFrame(score_lst, columns=["model", "ROC_AUC"])
        df_results.sort_values(by="ROC_AUC", ascending=False, inplace=True)
        df_results.reset_index(inplace=True, drop=True)
        df_results.to_csv(
            os.path.join(
                self.prediction_output_file,
                f"ROC_AUCResultComparison.csv",
            ),
            header=True,
        )

        #Plot ROC Curves
        for name, m in dict_model_results.items():
            model = m["model"]
            pred=model.predict(x_test)
            fpr, tpr, thresholds = roc_curve(y_test, pred)
            auc = roc_auc_score(y_test, pred)
            plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (name, auc))
            plt.plot([0, 1], [0, 1],'r--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('1-Specificity(False Positive Rate)')
            plt.ylabel('Sensitivity(True Positive Rate)')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.savefig(
            os.path.join(self.visualization_path, f"ROC_curve.png")
            )


        plt.figure(figsize=(20, 10))
        plt.xlabel("ROC_AUC", fontsize=12)
        plt.ylabel("Model Type", fontsize=12)
        # plt.xticks(rotation=90, fontsize=12)
        plt.title("The ROC_AUC Comparsion")
        chart = sns.barplot(x="ROC_AUC", y="model", data=df_results)
        wrap_labels(chart, 10)
        for container in chart.containers:
            chart.bar_label(container)
        plt.savefig(
            os.path.join(self.visualization_path, f"ROC_AUC.png")
        )

        # Get the best model name as per RMSE
        best_model_name = df_results.iloc[0]["model"]
        # models[best_model_name[0]]['model']
        model_name=df_results['model']
        best_model=None
        for name, m in dict_model_results.items():
            if name==best_model_name:
                best_model=m['model']
                break
        return best_model_name,best_model

def main(config_path, params_path):
    ## read config files
    config = read_yaml(config_path)
    model_finder=Model_Finder(config)
    final_df = pd.read_pickle("./final_df.pkl")
    y=pd.read_pickle("./target.pkl")
    X_train, X_test, y_train, y_test = train_test_split(final_df, y, test_size=0.33, random_state=42)
    best_model_name, best_model=model_finder.base_model_checks(
                    X_train, y_train, X_test, y_test)
    file_op=file_operations.FileOperations(config['artifacts']['TRAINED_MODEL_DIR'])
    save_model = file_op.save_model(best_model, best_model_name)
    print(save_model)
    pass


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e
