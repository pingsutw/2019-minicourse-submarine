from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import numpy as np
import logging
import mlflow
import mlflow.sklearn
n_folds = 5

logger = logging.getLogger(__name__)


def rmse(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


def run_xgboost(**kwargs):
    mlflow.set_tracking_uri('http://mlflow:5000')
    mlflow.start_run(run_name="xgboost")
    ti = kwargs['ti']
    train, label, test, _ = ti.xcom_pull(task_ids='preprocessing')

    logging.info('variables successfully fetched from previous task')

    params = {
        "colsample_bytree": 0.4603,
        "gamma": 0.0468,
        "learning_rate": 0.05,
        "max_depth": 20,
        "min_child_weight": 2,
        "n_estimators": 2200,
        "reg_alpha": 0.4640,
        "reg_lambda": 0.8571,
        "subsample": 0.5213,
        "random_state": 7,
        "nthread": -1
    }

    model_xgb = xgb.XGBRegressor(params=params)

    mlflow.log_params(params)
    model_xgb.fit(train, label)
    xgb_train_pred = model_xgb.predict(train)
    score = rmse(label, xgb_train_pred)
    mlflow.log_metric("rmse", score)
    mlflow.sklearn.log_model(model_xgb, "models")
    logging.info("score : ", score)
    return [model_xgb, score]


def run_lightgbm(**kwargs):
    mlflow.set_tracking_uri('http://mlflow:5000')
    mlflow.start_run(run_name="lightgbm")
    ti = kwargs['ti']
    train, label, test, _ = ti.xcom_pull(task_ids='preprocessing')
    logging.info('variables successfully fetched from previous task')

    params = {
        "objective": 'regression',
        "num_leaves": 20,
        "learning_rate": 0.05,
        "n_estimators": 720,
        "max_bin": 55,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "feature_fraction": 0.2319,
        "feature_fraction_seed": 9,
        "bagging_seed": 9,
        "min_data_in_leaf": 6
    }

    model_lgb = lgb.LGBMRegressor(**params)
    mlflow.log_params(params)
    model_lgb.fit(train, label)
    lgb_train_pred = model_lgb.predict(train)
    score = rmse(label, lgb_train_pred)
    mlflow.log_metric("rmse", score)

    logging.info("score : ", score)
    return [model_lgb, score]
