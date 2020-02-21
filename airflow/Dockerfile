FROM puckel/docker-airflow:1.10.6

WORKDIR /usr/local/airflow/
COPY requirements.txt .
COPY kaggle.json /usr/local/airflow/.kaggle/kaggle.json
RUN pip install --user -r requirements.txt

USER root 
RUN chown -R airflow:airflow /usr/local/airflow/ \
    && chmod 777 /usr/local/airflow/
RUN apt-get update -y && apt-get install -y zip vim git

USER airflow