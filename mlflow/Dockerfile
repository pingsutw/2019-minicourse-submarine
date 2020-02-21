FROM python:3.7.0

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -P /tmp && \
    bash /tmp/Miniconda3-latest-Linux-x86_64.sh -b

ENV PATH=$PATH:/root/miniconda3/bin
    
RUN pip install mlflow==1.5.0
RUN pip install psycopg2-binary==2.8.4

EXPOSE 5000
EXPOSE 5001
EXPOSE 5002

#mlflow models serve -m /mlflow/models/xgboost -p 5051 -h 0.0.0.0
