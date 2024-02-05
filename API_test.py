from fastapi import FastAPI
from pydantic import BaseModel
import snowflake.connector
from typing import List

# Snowflake credentials
SNOWFLAKE_USER = 'kbharaj3'
SNOWFLAKE_PASSWORD = 'Snowfl@key0014'
SNOWFLAKE_ACCOUNT = 'qx25653.ca-central-1.aws'
SNOWFLAKE_WAREHOUSE = 'FOOTY_STORE'
SNOWFLAKE_DATABASE = 'GEGENSTATS'
SNOWFLAKE_SCHEMA = 'TABLES'

app = FastAPI()

# Define a Pydantic model for your data (optional, for better data validation and documentation)
class Team(BaseModel):
    team_name: str
    team_id: str
    team_logo: str

@app.get("/teamNames", response_model=List[Team])
def get_team_names():
    with snowflake.connector.connect(
        user=SNOWFLAKE_USER,
        password=SNOWFLAKE_PASSWORD,
        account=SNOWFLAKE_ACCOUNT,
        warehouse=SNOWFLAKE_WAREHOUSE,
        database=SNOWFLAKE_DATABASE,
        schema=SNOWFLAKE_SCHEMA
    ) as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM TEAMS")
            rows = cursor.fetchall()
            # Transform data to match the Pydantic model
            data = [{"team_name": row[0], "team_id": row[1], "team_logo": row[2]} for row in rows]
            return data
