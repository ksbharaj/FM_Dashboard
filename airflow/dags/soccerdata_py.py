import soccerdata as sd
import pandas as pd
from tqdm import tqdm
import soccerdata as sd
from google.cloud import storage
import pickle
import datetime
import os

def upload_to_gcs(bucket_name, object_name, local_file):
    """Uploads a file to Google Cloud Storage."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    blob.upload_from_filename(local_file)

def make_season_integer(df):
    indexes = df.index.names
    df = df.reset_index()
    df['SEASON'] = df['SEASON'].astype(int)
    df = df.set_index(indexes)
    return df

def soccerdata_FBRef():
    fbref = sd.FBref(leagues="ENG-Premier League", seasons=["2324"], no_cache=True)
    fbref_team_standard = fbref.read_team_season_stats(stat_type='standard')
    # fbref_team_standard_oppo = fbref.read_team_season_stats(stat_type='standard', opponent_stats=True)
    fbref_team_shooting = fbref.read_team_season_stats(stat_type='shooting')
    fbref_team_passing = fbref.read_team_season_stats(stat_type='passing')
    fbref_team_defense = fbref.read_team_season_stats(stat_type='defense')
    fbref_team_possession = fbref.read_team_season_stats(stat_type='possession')
    fbref_team_misc = fbref.read_team_season_stats(stat_type='misc')
    fbref_schedule = fbref.read_schedule()

    fbref_team_standard = fbref_team_standard.rename_axis(
        index={'team': 'TEAM_NAME', 'season': 'SEASON', 'league': 'COMPETITION'})
    # fbref_team_standard_oppo = fbref_team_standard_oppo.rename_axis(
    #     index={'team': 'TEAM_NAME', 'season': 'SEASON', 'league': 'COMPETITION'})
    fbref_team_shooting = fbref_team_shooting.rename_axis(
        index={'team': 'TEAM_NAME', 'season': 'SEASON', 'league': 'COMPETITION'})
    fbref_team_passing = fbref_team_passing.rename_axis(
        index={'team': 'TEAM_NAME', 'season': 'SEASON', 'league': 'COMPETITION'})
    fbref_team_defense = fbref_team_defense.rename_axis(
        index={'team': 'TEAM_NAME', 'season': 'SEASON', 'league': 'COMPETITION'})
    fbref_team_possession = fbref_team_possession.rename_axis(
        index={'team': 'TEAM_NAME', 'season': 'SEASON', 'league': 'COMPETITION'})
    fbref_team_misc = fbref_team_misc.rename_axis(
        index={'team': 'TEAM_NAME', 'season': 'SEASON', 'league': 'COMPETITION'})
    fbref_schedule = fbref_schedule.rename_axis(index={'season': 'SEASON', 'league': 'COMPETITION'})

    fbref_schedule['date'] = pd.to_datetime(fbref_schedule['date'])
    fbref_schedule = fbref_schedule[fbref_schedule['date'] <= datetime.datetime.now().strftime('%Y-%m-%d')]

    fbref_team_standard = make_season_integer(fbref_team_standard)
    # fbref_team_standard_oppo = make_season_integer(fbref_team_standard_oppo)
    fbref_team_shooting = make_season_integer(fbref_team_shooting)
    fbref_team_passing = make_season_integer(fbref_team_passing)
    fbref_team_defense = make_season_integer(fbref_team_defense)
    fbref_team_possession = make_season_integer(fbref_team_possession)
    fbref_team_misc = make_season_integer(fbref_team_misc)
    fbref_schedule = make_season_integer(fbref_schedule)

    # file_path = '/tmp/league_table_soccerdata_FBRef.csv'
    # fbref_schedule.to_csv(file_path, index=True)
    #
    # upload_to_gcs('gegenstats', 'league_table_soccerdata_FBRef.csv', file_path)

    return (pickle.dumps(fbref_team_standard), pickle.dumps(fbref_team_shooting), pickle.dumps(fbref_team_passing),
            pickle.dumps(fbref_team_possession), pickle.dumps(fbref_team_misc), pickle.dumps(fbref_team_defense),
            pickle.dumps(fbref_schedule))
