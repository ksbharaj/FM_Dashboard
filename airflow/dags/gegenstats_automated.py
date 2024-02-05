# combined_dag.py
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from scraperFC_py import scraperFC_Understat
from soccerdata_py import soccerdata_FBRef
from transformation_py import (transform_team_standard_stats, transform_team_attacking_stats, transform_matches,
                               transform_team_defending_stats, transform_team_misc_stats, transform_stadiums)
from snowflake_uploader_py import upsert_to_snowflake
from radar_chart_teams_py import defending_radar_chart, attacking_radar_chart, standard_radar_chart, prepare_XPTS_table

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
    dag_id='gegenstats_automated_dag',  # Pythonic naming convention for variables
    default_args=default_args,
    schedule_interval='@monthly',
    catchup=False,
)

scrape_understat = PythonOperator(
    task_id='scrape_understat',
    python_callable=scraperFC_Understat,
    dag=dag,
)

scrape_fbref = PythonOperator(
    task_id='scrape_fbref',
    python_callable=soccerdata_FBRef,
    dag=dag,
)

transform_team_standard_stats = PythonOperator(
    task_id='transform_team_standard_stats',
    python_callable=transform_team_standard_stats,
    provide_context=True,
    dag=dag,
)

transform_team_attacking_stats = PythonOperator(
    task_id='transform_team_attacking_stats',
    python_callable=transform_team_attacking_stats,
    provide_context=True,
    dag=dag,
)

transform_team_defending_stats = PythonOperator(
    task_id='transform_team_defending_stats',
    python_callable=transform_team_defending_stats,
    provide_context=True,
    dag=dag,
)

transform_team_misc_stats = PythonOperator(
    task_id='transform_team_misc_stats',
    python_callable=transform_team_misc_stats,
    provide_context=True,
    dag=dag,
)

transform_stadiums = PythonOperator(
    task_id='transform_stadiums',
    python_callable=transform_stadiums,
    provide_context=True,
    dag=dag,
)

transform_matches = PythonOperator(
    task_id='transform_matches',
    python_callable=transform_matches,
    provide_context=True,
    dag=dag,
)

upload_team_standard_stats = PythonOperator(
    task_id='upload_team_standard_stats',
    python_callable=upsert_to_snowflake,
    op_kwargs={
        'table_name': 'TEAM_STANDARD_STATS',
        'primary_keys': ['TEAM_FBREF_ID', 'COMPETITION', 'SEASON'],
        'source_task_id': 'transform_team_standard_stats',
        'db_table': 'TABLES'
    },
    provide_context=True,
    dag=dag,
)

upload_team_attacking_stats = PythonOperator(
    task_id='upload_team_attacking_stats',
    python_callable=upsert_to_snowflake,
    op_kwargs={
        'table_name': 'TEAM_ATTACKING_STATS',
        'primary_keys': ['TEAM_FBREF_ID', 'COMPETITION', 'SEASON'],
        'source_task_id': 'transform_team_attacking_stats',
        'db_table': 'TABLES'
    },
    provide_context=True,
    dag=dag,
)

upload_team_defending_stats = PythonOperator(
    task_id='upload_team_defending_stats',
    python_callable=upsert_to_snowflake,
    op_kwargs={
        'table_name': 'TEAM_DEFENDING_STATS',
        'primary_keys': ['TEAM_FBREF_ID', 'COMPETITION','SEASON'],
        'source_task_id': 'transform_team_defending_stats',
        'db_table': 'TABLES'
    },
    provide_context=True,
    dag=dag,
)

upload_team_misc_stats = PythonOperator(
    task_id='upload_team_misc_stats',
    python_callable=upsert_to_snowflake,
    op_kwargs={
        'table_name': 'TEAM_MISC_STATS',
        'primary_keys': ['TEAM_FBREF_ID', 'COMPETITION','SEASON'],
        'source_task_id': 'transform_team_misc_stats',
        'db_table': 'TABLES'
    },
    provide_context=True,
    dag=dag,
)

upload_stadiums = PythonOperator(
    task_id='upload_stadiums',
    python_callable=upsert_to_snowflake,
    op_kwargs={
        'table_name': 'STADIUMS',
        'primary_keys': ['TEAM_FBREF_ID','STADIUM', 'SEASON'],
        'source_task_id': 'transform_stadiums',
        'db_table': 'TABLES'
    },
    provide_context=True,
    dag=dag,
)

upload_matches = PythonOperator(
    task_id='upload_matches',
    python_callable=upsert_to_snowflake,
    op_kwargs={
        'table_name': 'MATCHES',
        'primary_keys': ['MATCH_ID'],
        'source_task_id': 'transform_matches',
        'db_table': 'TABLES'
    },
    provide_context=True,
    dag=dag,
)

team_standard_radar_chart = PythonOperator(
    task_id='team_standard_radar_chart',
    python_callable=standard_radar_chart,
    provide_context=True,
    dag=dag,
)

team_attacking_radar_chart = PythonOperator(
    task_id='team_attacking_radar_chart',
    python_callable=attacking_radar_chart,
    provide_context=True,
    dag=dag,
)

team_defending_radar_chart = PythonOperator(
    task_id='team_defending_radar_chart',
    python_callable=defending_radar_chart,
    provide_context=True,
    dag=dag,
)

team_prepare_XPTS_table = PythonOperator(
    task_id='team_prepare_XPTS_table',
    python_callable=prepare_XPTS_table,
    provide_context=True,
    dag=dag,
)

upload_team_standard_radar_chart = PythonOperator(
    task_id='upload_team_standard_radar_chart',
    python_callable=upsert_to_snowflake,
    op_kwargs={
        'table_name': 'STANDARD_RADAR',
        'primary_keys': ['SEASON','COMPETITION_ACRONYM','TEAM_NAME','VARIABLE'],
        'source_task_id': 'team_standard_radar_chart',
        'db_table': 'RADAR_CHARTS'
    },
    provide_context=True,
    dag=dag,
)

upload_team_attacking_radar_chart = PythonOperator(
    task_id='upload_team_attacking_radar_chart',
    python_callable=upsert_to_snowflake,
    op_kwargs={
        'table_name': 'ATTACKING_RADAR',
        'primary_keys': ['SEASON','COMPETITION_ACRONYM','TEAM_NAME','VARIABLE'],
        'source_task_id': 'team_attacking_radar_chart',
        'db_table': 'RADAR_CHARTS'
    },
    provide_context=True,
    dag=dag,
)

upload_team_defending_radar_chart = PythonOperator(
    task_id='upload_team_defending_radar_chart',
    python_callable=upsert_to_snowflake,
    op_kwargs={
        'table_name': 'DEFENDING_RADAR',
        'primary_keys': ['SEASON','COMPETITION_ACRONYM','TEAM_NAME','VARIABLE'],
        'source_task_id': 'team_defending_radar_chart',
        'db_table': 'RADAR_CHARTS'
    },
    provide_context=True,
    dag=dag,
)

upload_XPTS_table = PythonOperator(
    task_id='upload_XPTS_table',
    python_callable=upsert_to_snowflake,
    op_kwargs={
        'table_name': 'XPTS_TABLE',
        'primary_keys': ['SEASON','COMPETITION_ACRONYM'],
        'source_task_id': 'team_prepare_XPTS_table',
        'db_table': 'RADAR_CHARTS'
    },
    provide_context=True,
    dag=dag,
)

# Set dependencies
scrape_understat >> transform_team_standard_stats
scrape_fbref >> transform_team_standard_stats
scrape_understat >> transform_team_attacking_stats
scrape_fbref >> transform_team_attacking_stats
scrape_understat >> transform_team_defending_stats
scrape_fbref >> transform_team_defending_stats
scrape_understat >> transform_team_misc_stats
scrape_fbref >> transform_team_misc_stats
scrape_fbref >> transform_stadiums
scrape_fbref >> transform_matches

transform_team_standard_stats >> upload_team_standard_stats
transform_team_attacking_stats >> upload_team_attacking_stats
transform_team_defending_stats >> upload_team_defending_stats
transform_team_misc_stats >> upload_team_misc_stats
transform_stadiums >> upload_stadiums
transform_matches >> upload_matches

upload_team_standard_stats >> team_standard_radar_chart
upload_team_attacking_stats >> team_standard_radar_chart
upload_team_defending_stats >> team_standard_radar_chart

upload_team_standard_stats >> team_attacking_radar_chart
upload_team_attacking_stats >> team_attacking_radar_chart

upload_team_standard_stats >> team_defending_radar_chart
upload_team_defending_stats >> team_defending_radar_chart

upload_team_standard_stats >> team_prepare_XPTS_table
upload_team_attacking_stats >> team_prepare_XPTS_table
upload_team_defending_stats >> team_prepare_XPTS_table

team_prepare_XPTS_table >> upload_XPTS_table

team_standard_radar_chart >> upload_team_standard_radar_chart
team_attacking_radar_chart >> upload_team_attacking_radar_chart
team_defending_radar_chart >> upload_team_defending_radar_chart

