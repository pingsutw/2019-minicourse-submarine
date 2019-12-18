import keras
from keras.models import Sequential, load_model
import mlflow.keras
import numpy as np
import logging
import pandas as pd
import os

logger = logging.getLogger(__name__)


def output_csv(dirPath, predict, test_ID):
    sub = pd.DataFrame()
    sub['Id'] = test_ID
    sub['SalePrice'] = predict
    path = os.path.join(dirPath, 'submission.csv')
    sub.to_csv(path, index=False)


def model_validation(**kwargs):
    ti = kwargs['ti']
    xgboost, score1 = ti.xcom_pull(task_ids='xgboost')
    lightgbm, score2 = ti.xcom_pull(task_ids='lightgbm')
    _, _, test, test_ID = ti.xcom_pull(task_ids='preprocessing')

    # Find the best model and predict test result
    if score1 > score2:
        predict = np.expm1(xgboost.predict(test))
    else:
        predict = np.expm1(lightgbm.predict(test))
    output_csv(kwargs['path_model_predict_dir'], predict, test_ID)

    logging.info('Save model')
    path = os.path.join(kwargs['path_model_save_dir'], 'xgboost_model.pth')
    xgboost.save_model(path)

    path = os.path.join(kwargs['path_model_save_dir'], 'lightgbm_model.pth')
    lightgbm.booster_.save_model(path)
