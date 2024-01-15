import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas

def upsert_to_snowflake(table_name, primary_keys, source_task_id, db_table, **kwargs):
    SNOWFLAKE_USER = 'kbharaj3'
    SNOWFLAKE_PASSWORD = 'Snowfl@key0014'
    SNOWFLAKE_ACCOUNT = 'qx25653.ca-central-1.aws'
    SNOWFLAKE_WAREHOUSE = 'FOOTY_STORE'
    SNOWFLAKE_DATABASE = 'GEGENSTATS'
    SNOWFLAKE_SCHEMA = db_table


    conn = snowflake.connector.connect(
        user=SNOWFLAKE_USER,
        password=SNOWFLAKE_PASSWORD,
        account=SNOWFLAKE_ACCOUNT,
        warehouse=SNOWFLAKE_WAREHOUSE,
        database=SNOWFLAKE_DATABASE,
        schema=SNOWFLAKE_SCHEMA
    )

    cur = conn.cursor()

    ti = kwargs['ti']
    dataframe = ti.xcom_pull(task_ids=source_task_id)

    if isinstance(primary_keys, str):
        primary_keys = [primary_keys]

    temp_table_name = f"{table_name}_TEMP"
    cur.execute(f"CREATE TEMPORARY TABLE {temp_table_name} LIKE {table_name}")
    success, nchunks, nrows, _ = write_pandas(conn, dataframe, temp_table_name, auto_create_table=True)

    on_condition = ' AND '.join([f"{table_name}.{pk} = {temp_table_name}.{pk}" for pk in primary_keys])

    update_columns = [col for col in dataframe.columns if col not in primary_keys]
    update_sql = ', '.join([f"{table_name}.{col} = {temp_table_name}.{col}" for col in update_columns])

    merge_sql = f"""
    MERGE INTO {table_name} USING {temp_table_name}
    ON {on_condition}
    """

    if update_columns:
        merge_sql += f"""
        WHEN MATCHED THEN
            UPDATE SET
                {update_sql}
        """

    merge_sql += f"""
    WHEN NOT MATCHED THEN
        INSERT ({', '.join(dataframe.columns)})
        VALUES ({', '.join([f"{temp_table_name}.{col}" for col in dataframe.columns])})
    """

    cur.execute(merge_sql)
    cur.execute(f"DROP TABLE IF EXISTS {temp_table_name}")

    print(f"Upsert completed. {nrows} rows processed.")

    conn.close()


