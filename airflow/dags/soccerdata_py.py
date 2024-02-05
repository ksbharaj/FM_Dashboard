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
    ## Set up FBRef for scraping
    fbref = sd.FBref(leagues="ENG-Premier League", seasons=["2324"], no_cache=True)

    ## Various tables are scraped below
    fbref_team_standard = fbref.read_team_season_stats(stat_type='standard')
    # fbref_team_standard_oppo = fbref.read_team_season_stats(stat_type='standard', opponent_stats=True)
    fbref_team_shooting = fbref.read_team_season_stats(stat_type='shooting')
    fbref_team_shooting_oppo = fbref.read_team_season_stats(stat_type='shooting', opponent_stats=True)
    fbref_team_passing = fbref.read_team_season_stats(stat_type='passing')
    fbref_team_passing_oppo = fbref.read_team_season_stats(stat_type='passing', opponent_stats=True)
    fbref_team_defense = fbref.read_team_season_stats(stat_type='defense')
    fbref_team_possession = fbref.read_team_season_stats(stat_type='possession')
    fbref_team_misc = fbref.read_team_season_stats(stat_type='misc')
    fbref_keeper = fbref.read_team_season_stats(stat_type='keeper')
    fbref_schedule = fbref.read_schedule()
    fbref_lineups = fbref.read_lineup()

    ## Player names are corrected below
    fbref_lineups.replace({"Jarrell Quansah": "Jarell Quansah"}, inplace=True)
    fbref_lineups.replace({"Omotayo Adaramola": "Tayo Adaramola"}, inplace=True)

    ## Team names in FBref's lineups are corrected below
    fbref_lineups.loc[fbref_lineups['team'] == 'Brighton & Hove Albion', 'team'] = 'Brighton'
    fbref_lineups.loc[fbref_lineups['team'] == 'Manchester United', 'team'] = 'Manchester Utd'
    fbref_lineups.loc[fbref_lineups['team'] == 'Newcastle United', 'team'] = 'Newcastle Utd'
    fbref_lineups.loc[fbref_lineups['team'] == 'Nottingham Forest', 'team'] = "Nott'ham Forest"
    fbref_lineups.loc[fbref_lineups['team'] == 'Sheffield United', 'team'] = "Sheffield Utd"
    fbref_lineups.loc[fbref_lineups['team'] == 'Tottenham Hotspur', 'team'] = 'Tottenham'
    fbref_lineups.loc[fbref_lineups['team'] == 'West Ham United', 'team'] = 'West Ham'
    fbref_lineups.loc[fbref_lineups['team'] == 'Wolverhampton Wanderers', 'team'] = 'Wolves'

    ## Header names in each table are corrected below
    fbref_team_standard = fbref_team_standard.rename_axis(
        index={'team': 'TEAM_NAME', 'season': 'SEASON', 'league': 'COMPETITION'})
    # fbref_team_standard_oppo = fbref_team_standard_oppo.rename_axis(index={'team': 'TEAM_NAME', 'season': 'SEASON', 'league': 'COMPETITION'})
    fbref_team_shooting = fbref_team_shooting.rename_axis(
        index={'team': 'TEAM_NAME', 'season': 'SEASON', 'league': 'COMPETITION'})
    fbref_team_shooting_oppo = fbref_team_shooting_oppo.rename_axis(
        index={'team': 'TEAM_NAME', 'season': 'SEASON', 'league': 'COMPETITION'})
    fbref_team_passing = fbref_team_passing.rename_axis(
        index={'team': 'TEAM_NAME', 'season': 'SEASON', 'league': 'COMPETITION'})
    fbref_team_passing_oppo = fbref_team_passing_oppo.rename_axis(
        index={'team': 'TEAM_NAME', 'season': 'SEASON', 'league': 'COMPETITION'})
    fbref_team_defense = fbref_team_defense.rename_axis(
        index={'team': 'TEAM_NAME', 'season': 'SEASON', 'league': 'COMPETITION'})
    fbref_team_possession = fbref_team_possession.rename_axis(
        index={'team': 'TEAM_NAME', 'season': 'SEASON', 'league': 'COMPETITION'})
    fbref_team_misc = fbref_team_misc.rename_axis(
        index={'team': 'TEAM_NAME', 'season': 'SEASON', 'league': 'COMPETITION'})
    fbref_keeper = fbref_keeper.rename_axis(index={'team': 'TEAM_NAME', 'season': 'SEASON', 'league': 'COMPETITION'})
    fbref_schedule = fbref_schedule.rename_axis(index={'season': 'SEASON', 'league': 'COMPETITION'})
    fbref_lineups = fbref_lineups.rename_axis(index={'season': 'SEASON', 'league': 'COMPETITION'})
    fbref_lineups.rename({'team': 'TEAM_NAME'}, axis=1, inplace=True)

    ## Remove incomplete games from the schedule
    fbref_schedule = fbref_schedule.dropna(subset=['home_xg'])


    fbref_team_passing_oppo.reset_index(inplace=True)
    fbref_team_shooting_oppo.reset_index(inplace=True)

    fbref_team_passing_oppo['TEAM_NAME'] = fbref_team_passing_oppo['TEAM_NAME'].str.replace('vs ', '')
    fbref_team_shooting_oppo['TEAM_NAME'] = fbref_team_shooting_oppo['TEAM_NAME'].str.replace('vs ', '')

    fbref_team_passing_oppo.set_index(['COMPETITION', 'SEASON', 'TEAM_NAME'], inplace=True)
    fbref_team_shooting_oppo.set_index(['COMPETITION', 'SEASON', 'TEAM_NAME'], inplace=True)

    fbref_schedule['date'] = pd.to_datetime(fbref_schedule['date'])
    fbref_schedule = fbref_schedule[fbref_schedule['date'] <= datetime.datetime.now().strftime('%Y-%m-%d')]

    fbref_team_standard = make_season_integer(fbref_team_standard)
    # fbref_team_standard_oppo = make_season_integer(fbref_team_standard_oppo)
    fbref_team_shooting = make_season_integer(fbref_team_shooting)
    fbref_team_shooting_oppo = make_season_integer(fbref_team_shooting_oppo)
    fbref_team_passing = make_season_integer(fbref_team_passing)
    fbref_team_passing_oppo = make_season_integer(fbref_team_passing_oppo)
    fbref_team_defense = make_season_integer(fbref_team_defense)
    fbref_team_possession = make_season_integer(fbref_team_possession)
    fbref_team_misc = make_season_integer(fbref_team_misc)
    fbref_schedule = make_season_integer(fbref_schedule)
    fbref_keeper = make_season_integer(fbref_keeper)
    fbref_lineups = make_season_integer(fbref_lineups)

    # file_path = '/tmp/league_table_soccerdata_FBRef.csv'
    # fbref_schedule.to_csv(file_path, index=True)
    #
    # upload_to_gcs('gegenstats', 'league_table_soccerdata_FBRef.csv', file_path)

    return (pickle.dumps(fbref_team_standard), pickle.dumps(fbref_team_shooting), pickle.dumps(fbref_team_passing),
            pickle.dumps(fbref_team_possession), pickle.dumps(fbref_team_misc), pickle.dumps(fbref_team_defense),
            pickle.dumps(fbref_schedule), pickle.dumps(fbref_team_passing_oppo), pickle.dumps(fbref_keeper),
            pickle.dumps(fbref_team_shooting_oppo), pickle.dumps(fbref_lineups))

def soccerdata_WhoScored():
    ws = sd.WhoScored(leagues='ENG-Premier League', seasons="2324", headless=False, no_cache=True, no_store=True)

    ws_schedule = ws.read_schedule()
    ws_schedule = ws_schedule[ws_schedule['date'] <= datetime.datetime.now().strftime('%Y-%m-%d')]

    return pickle.dumps(ws_schedule)

