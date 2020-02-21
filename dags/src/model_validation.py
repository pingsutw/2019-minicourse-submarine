import keras
from keras.models import Sequential, load_model
import mlflow
import mlflow.xgboost
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
    logging.info('xgboost score: ', score1)
    logging.info('lightgbm score: ', score2)

    # predict = np.expm1(xgboost.predict(test))
    if score1 > score2:
        logging.info("xgboost is better than lightgbm")
        predict = np.expm1(xgboost.predict(test))
    else:
        logging.info("lightgbm is better than xgboost")
        predict = np.expm1(lightgbm.predict(test))

    output_csv(kwargs['path_model_predict_dir'], predict, test_ID)
    return [xgboost, lightgbm]
