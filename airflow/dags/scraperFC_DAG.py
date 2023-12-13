# scraperfc_dag.py
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from scraperFC_py import scraperFC_FBRef  # Import your scraping function

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),  # Adjust as needed
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'scraperFC_DAG',  # This is where you name your DAG
    default_args=default_args,
    description='DAG for scraping FBRef data using scraperFC',
    schedule_interval='@daily',  # Adjust as needed
    catchup=False
)

scrape_fbref_task = PythonOperator(
    task_id='scrape_fbref',
    python_callable=scraperFC_FBRef,
    dag=dag,
)

scrape_fbref_task