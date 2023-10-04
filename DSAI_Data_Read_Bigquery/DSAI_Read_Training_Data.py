import os
from google.cloud import bigquery

def ReadTrainingData():
    vAR_client = bigquery.Client()
    vAR_response_table_name = "DMV_DRIVER_RISK_TRAIN"
    vAR_sql =(
        "select * from `"+ os.environ["GCP_PROJECT_ID"]+"."+os.environ["GCP_BQ_SCHEMA_NAME"]+"."+vAR_response_table_name+"`"
    )

    vAR_df = vAR_client.query(vAR_sql).to_dataframe()


    return vAR_df