# combined_dag.py
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from soccerdata_py import soccerdata_WhoScored

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,  # Consider notifying on failure
    'email_on_retry': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=500),
}

dag = DAG(
    dag_id='gegenstats_tester_dag',  # Pythonic naming convention for variables
    default_args=default_args,
    schedule_interval='@monthly',
    catchup=False,
)

scrape_whoscored = PythonOperator(
    task_id='scrape_whoscored',
    python_callable=soccerdata_WhoScored,
    dag=dag,
)


# Set dependencies
scrape_whoscored

