from google.cloud import storage
import pandas as pd
import pickle

def upload_to_gcs(bucket_name, object_name, local_file):
    """Uploads a file to Google Cloud Storage."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    blob.upload_from_filename(local_file)

def master_extractor(kwargs_input):
    ti = kwargs_input
    understat_table = ti.xcom_pull(task_ids='scrape_understat')
    (fbref_team_standard, fbref_team_shooting, fbref_team_passing, fbref_team_possession, fbref_team_misc,
     fbref_team_defense, fbref_schedule) = ti.xcom_pull(task_ids='scrape_fbref')
    return (understat_table, fbref_team_standard, fbref_team_shooting, fbref_team_passing,
            fbref_team_possession, fbref_team_misc, fbref_team_defense, fbref_schedule)

def teams_extractor(fbref_team_standard):
    team_names = fbref_team_standard.reset_index()[['TEAM_NAME', 'url']]
    team_names['TEAM_FBREF_ID'] = team_names['url'].apply(lambda x: x.split('/')[3])  ## Extracting team_id from URL
    team_names.drop(columns=['url'], inplace=True)
    team_names.columns = team_names.columns.droplevel(1)
    team_names['TEAM_FBREF_ID'] = team_names['TEAM_FBREF_ID'].astype('string')
    team_names.drop_duplicates(inplace=True)
    return team_names

def transform_team_standard_stats(**kwargs):
    understat_table, fbref_team_standard, _, _, _, _, _, _ = master_extractor(kwargs['ti'])

    understat_table = pickle.loads(understat_table)
    fbref_team_standard = pickle.loads(fbref_team_standard)

    team_names = teams_extractor(fbref_team_standard)

    team_matches_played = fbref_team_standard['Playing Time'].MP

    team_standard_stats = understat_table.copy()
    team_standard_stats['MATCHES_PLAYED'] = team_matches_played
    team_standard_stats.reset_index(inplace=True)

    team_standard_stats = team_standard_stats.merge(team_names, on='TEAM_NAME', how='left')
    team_standard_stats = team_standard_stats[['TEAM_FBREF_ID', 'SEASON', 'COMPETITION', 'MATCHES_PLAYED', 'TEAM_WINS',
                                               'TEAM_DRAWS', 'TEAM_LOSSES', 'TEAM_PTS', 'TEAM_XPTS']]

    return team_standard_stats

def transform_team_attacking_stats(**kwargs):
    (understat_table, fbref_team_standard, fbref_team_shooting, fbref_team_passing, fbref_team_possession,
     fbref_team_misc, _, _) = master_extractor(kwargs['ti'])

    understat_table = pickle.loads(understat_table)
    fbref_team_standard = pickle.loads(fbref_team_standard)
    fbref_team_shooting = pickle.loads(fbref_team_shooting)
    fbref_team_passing = pickle.loads(fbref_team_passing)
    fbref_team_possession = pickle.loads(fbref_team_possession)
    fbref_team_misc = pickle.loads(fbref_team_misc)

    team_names = teams_extractor(fbref_team_standard)

    team_attacking_stats = understat_table.copy()
    team_attacking_stats = team_attacking_stats.reset_index().merge(team_names, on='TEAM_NAME', how='left')
    team_attacking_stats = team_attacking_stats[['TEAM_NAME', 'TEAM_FBREF_ID', 'COMPETITION', 'SEASON', 'NPxG', 'xG']]

    team_attacking_stats.rename(columns={'NPxG': 'NPXG', 'xG': 'XG'}, inplace=True)
    team_attacking_stats.set_index(['COMPETITION', 'SEASON', 'TEAM_NAME'], inplace=True)

    team_goals_scored = understat_table.G
    team_shots = fbref_team_shooting['Standard'].Sh
    team_shotsOT = fbref_team_shooting['Standard'].SoT
    team_pass_completed = fbref_team_passing["Total"].Cmp
    team_pass_attempted = fbref_team_passing["Total"].Att
    team_takeons_attempted = fbref_team_possession['Take-Ons'].Att
    team_takeons_completed = fbref_team_possession['Take-Ons'].Succ
    team_crossesintoPA = fbref_team_passing['CrsPA']
    team_fouls_against = fbref_team_misc['Performance']['Fld']

    team_attacking_stats['GOALS_SCORED'] = team_goals_scored
    team_attacking_stats['SHOTS'] = team_shots
    team_attacking_stats['SHOTS_ON_TARGET'] = team_shotsOT
    team_attacking_stats['PASS_COMPLETED'] = team_pass_completed
    team_attacking_stats['PASS_ATTEMPTED'] = team_pass_attempted
    team_attacking_stats['TAKEONS_ATTEMPTED'] = team_takeons_attempted
    team_attacking_stats['TAKEONS_COMPLETED'] = team_takeons_completed
    team_attacking_stats['CROSSES_INTO_PA'] = team_crossesintoPA
    team_attacking_stats['FOULS_AGAINST'] = team_fouls_against

    team_attacking_stats.reset_index(inplace=True)
    team_attacking_stats.drop(columns=['TEAM_NAME'], inplace=True)

    return team_attacking_stats

def transform_team_defending_stats(**kwargs):
    (understat_table, fbref_team_standard, _, _, _,
     fbref_team_misc, fbref_team_defense, _) = master_extractor(kwargs['ti'])

    understat_table = pickle.loads(understat_table)
    fbref_team_standard = pickle.loads(fbref_team_standard)
    fbref_team_defense = pickle.loads(fbref_team_defense)
    fbref_team_misc = pickle.loads(fbref_team_misc)

    team_names = teams_extractor(fbref_team_standard)

    team_defending_stats = understat_table.copy()
    team_defending_stats = team_defending_stats.reset_index().merge(team_names, on='TEAM_NAME', how='left')
    team_defending_stats = team_defending_stats[['TEAM_NAME', 'TEAM_FBREF_ID', 'COMPETITION', 'SEASON']]

    team_defending_stats.set_index(['COMPETITION', 'SEASON', 'TEAM_NAME'], inplace=True)

    team_goals_conceded = understat_table.GA
    team_tackles = fbref_team_defense.Tackles.Tkl
    team_tackles_won = fbref_team_defense.Tackles.TklW
    team_fouls_made = fbref_team_misc['Performance']['Fls']
    team_interceptions = fbref_team_defense['Int']
    team_blocks_shots = fbref_team_defense['Blocks'].Sh
    team_blocks_pass = fbref_team_defense['Blocks'].Pass
    team_clearances = fbref_team_defense.Clr

    team_defending_stats['GOALS_CONCEDED'] = team_goals_conceded
    team_defending_stats['XG_AGAINST'] = understat_table['XG_AGAINST']
    team_defending_stats['NPXG_AGAINST'] = understat_table['NPXG_AGAINST']
    team_defending_stats['TACKLES'] = team_tackles
    team_defending_stats['TACKLES_WON'] = team_tackles_won
    team_defending_stats['FOULS_MADE'] = team_fouls_made
    team_defending_stats['INTERCEPTIONS'] = team_interceptions
    team_defending_stats['BLOCKED_SHOTS'] = team_blocks_shots
    team_defending_stats['BLOCKED_PASSES'] = team_blocks_pass
    team_defending_stats['CLEARANCES'] = team_clearances
    team_defending_stats['PPDA'] = understat_table['PPDA']

    team_defending_stats.reset_index(inplace=True)
    team_defending_stats.drop(columns=['TEAM_NAME'], inplace=True)

    return team_defending_stats

def transform_team_misc_stats(**kwargs):
    (understat_table, fbref_team_standard, _, _, _,
     fbref_team_misc, fbref_team_defense, _) = master_extractor(kwargs['ti'])

    understat_table = pickle.loads(understat_table)
    fbref_team_standard = pickle.loads(fbref_team_standard)
    fbref_team_misc = pickle.loads(fbref_team_misc)

    team_names = teams_extractor(fbref_team_standard)

    team_misc_stats = understat_table.copy()
    team_misc_stats = team_misc_stats.reset_index().merge(team_names, on='TEAM_NAME', how='left')
    team_misc_stats = team_misc_stats[['TEAM_NAME', 'TEAM_FBREF_ID', 'COMPETITION', 'SEASON']]
    team_misc_stats.set_index(['COMPETITION', 'SEASON', 'TEAM_NAME'], inplace=True)

    team_aerials_won = fbref_team_misc['Aerial Duels'].Won
    team_aerials_lost = fbref_team_misc['Aerial Duels'].Lost

    team_misc_stats['AERIALS_WON'] = team_aerials_won
    team_misc_stats['AERIALS_LOST'] = team_aerials_lost

    team_misc_stats.reset_index(inplace=True)
    team_misc_stats.drop(columns=['TEAM_NAME'], inplace=True)

    return team_misc_stats

def transform_stadiums(**kwargs):
    (_, fbref_team_standard, _, _, _, _, _, fbref_schedule) = master_extractor(kwargs['ti'])

    fbref_schedule = pickle.loads(fbref_schedule)
    fbref_team_standard = pickle.loads(fbref_team_standard)

    team_names = teams_extractor(fbref_team_standard)

    df_stadiums = fbref_schedule.reset_index()[['home_team', 'SEASON', 'venue']].drop_duplicates().rename(
        columns={'home_team': 'TEAM_NAME', 'venue': 'STADIUM'})

    df_stadiums = df_stadiums.merge(team_names, on='TEAM_NAME')[['STADIUM', 'SEASON', 'TEAM_FBREF_ID']]

    return df_stadiums

def transform_matches(**kwargs):
    (_, fbref_team_standard, _, _, _, _, _, fbref_schedule) = master_extractor(kwargs['ti'])

    fbref_schedule = pickle.loads(fbref_schedule)
    fbref_team_standard = pickle.loads(fbref_team_standard)

    team_names = teams_extractor(fbref_team_standard)

    df_matches = fbref_schedule.reset_index()[['game_id','date','time','home_team','away_team','COMPETITION','SEASON',
                                               'venue','week','day','score','home_xg','away_xg','attendance','referee']]

    df_matches = df_matches.merge(team_names.rename(columns={'TEAM_FBREF_ID': 'home_team_id',
                                                             'TEAM_NAME': 'home_team'}), on='home_team', how='left')
    df_matches = df_matches.merge(team_names.rename(columns={'TEAM_FBREF_ID': 'away_team_id',
                                                             'TEAM_NAME': 'away_team'}), on='away_team', how='left')

    ## KEY- getting rid of matches that do not have a game_id (Game_id is NaN)
    df_matches = df_matches[df_matches.game_id.notna()]

    # Combine df_matches date and time to a dt.datatime configuration
    df_matches['date'] = df_matches['date'].astype(str)
    df_matches['time'] = df_matches['time'].astype(str)
    df_matches['date_time'] = df_matches['date'] + ' ' + df_matches['time']
    df_matches['date_time'] = pd.to_datetime(df_matches['date_time'])

    df_matches[['home_team_score', 'away_team_score']] = df_matches['score'].str.split('–', expand=True)

    df_matches.rename(columns={'game_id': 'MATCH_ID', 'date_time': 'DATE_TIME', 'home_team_id': 'HOME_TEAM_ID',
                               'away_team_id': 'AWAY_TEAM_ID','venue': 'STADIUM', 'week': 'GAMEWEEK', 'day': 'DAY',
                               'home_team_score': 'HOME_TEAM_SCORE','away_team_score': 'AWAY_TEAM_SCORE', 'referee': 'REFEREE',
                               'home_xg': 'HOME_TEAM_XG','away_xg': 'AWAY_TEAM_XG','attendance': 'ATTENDANCE'}, inplace=True)

    df_matches.drop(columns=['date', 'time', 'home_team', 'away_team', 'score'], inplace=True)
    df_matches['DATE_TIME'] = df_matches['DATE_TIME'].astype('string')

    return df_matches




