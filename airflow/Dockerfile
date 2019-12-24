FROM puckel/docker-airflow:latest

WORKDIR /usr/local/airflow/
COPY requirements.txt .
COPY kaggle.json /usr/local/airflow/.kaggle/kaggle.json
RUN pip install --user -r requirements.txt

USER root
RUN chmod 755 /usr/local/airflow/.kaggle/kaggle.json
RUN apt-get update -y && apt-get install -y zip vim git

USER airflow