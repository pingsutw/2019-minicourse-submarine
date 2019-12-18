#!/bin/bash

echo "Submit kaggle testing data"
export PATH=$PATH:/usr/local/airflow/.local/bin
FILE=/usr/local/airflow/data/submission.csv

if [ -f "$FILE" ]; then
  kaggle competitions submit -c house-prices-advanced-regression-techniques -f $FILE -m "Submit from airflow"
  echo "Submit done!!!"
else
  echo "File not found !!"
  exit 1
fi