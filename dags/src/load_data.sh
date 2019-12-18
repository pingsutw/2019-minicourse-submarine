#!/bin/bash

export PATH=$PATH:/usr/local/airflow/.local/bin
COMPETITION="{{ params.competition }}"
DATA_DIR=/usr/local/airflow/data
if [ -f "${DATA_DIR}/${COMPETITION}.zip" ]; then
  echo "data already download~"
else
  echo "Dowmload kaggle dataset"
  kaggle competitions download -c "$COMPETITION" -p "$DATA_DIR"
  unzip "$DATA_DIR"/"$COMPETITION".zip -d "$DATA_DIR"
  echo "Dowmload done!!!"
fi