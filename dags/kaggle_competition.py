from datetime import timedelta

import airflow
from airflow.models import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
from airflow.operators.papermill_operator import PapermillOperator
from src.preprocessing import preprocessing
from src.training import run_xgboost, run_lightgbm
from src.model_validation import model_validation

PATH_SAVE_MODEL_DIR = "./models"
PATH_SAVE_PREDICT_DIR = "./data"
COMPETITION = "house-prices-advanced-regression-techniques"

args = {
    'owner': 'airflow',
    'start_date': airflow.utils.dates.days_ago(2),
    'provide_context': True  # this is set to True as we want to pass variables on from one task to another
}

# Kaggle competition : https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview
dag = DAG(
    dag_id='kaggle_competition',
    default_args=args,
    schedule_interval='0 0 * * *',
    dagrun_timeout=timedelta(minutes=60),
)

# download dataset from https://www.kaggle.com/c/house-prices-advanced-regression-techniques
data_ingestion = BashOperator(
    task_id='data_Ingestion',
    bash_command='./src/load_data.sh',
    params={'competition': COMPETITION},
    dag=dag
)

preprocessing = PythonOperator(
    task_id='preprocessing',
    python_callable=preprocessing,
    dag=dag
)

xgboost = PythonOperator(
    task_id='xgboost',
    python_callable=run_xgboost,
    dag=dag
)

lightgbm = PythonOperator(
    task_id='lightgbm',
    python_callable=run_lightgbm,
    dag=dag
)

model_validation = PythonOperator(
    task_id='model_validation',
    python_callable=model_validation,
    op_kwargs={'path_model_save_dir': PATH_SAVE_MODEL_DIR,
               'path_model_predict_dir': PATH_SAVE_PREDICT_DIR},
    dag=dag
)

keggle_summit = BashOperator(
    task_id='keggle_summit',
    bash_command='./src/submit.sh',
    dag=dag
)

# build DAG
data_ingestion >> preprocessing >> xgboost >> model_validation >> keggle_summit
preprocessing >> lightgbm >> model_validation
preprocessing >> model_validation

# run_this = PapermillOperator(
#     task_id="preprocessing",
#     input_nb="./data_visualization.ipynb",
#     output_nb="/tmp/out-{{ execution_date }}.ipynb",
#     parameters={"msgs": "Ran from Airflow at {{ execution_date }}!"},
#     dag=dag
# )

if __name__ == "__main__":
    dag.cli()
