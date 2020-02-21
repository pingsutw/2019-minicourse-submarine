import mlflow
import mlflow.xgboost
import mlflow.lightgbm
import logging
import os

logger = logging.getLogger(__name__)


def save_model_local(**kwargs):
    logging.info('Save model')

    ti = kwargs['ti']
    xgboost, lightgbm = ti.xcom_pull(task_ids='model_validation')
    # xgboost = ti.xcom_pull(task_ids='model_validation')

    path = os.path.join(kwargs['path_model_save_dir'], 'xgboost')
    mlflow.xgboost.save_model(xgboost, path)

    path = os.path.join(kwargs['path_model_save_dir'], 'lightgbm')
    mlflow.lightgbm.save_model(lightgbm.booster_, path)
