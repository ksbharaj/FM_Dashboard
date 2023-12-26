import pandas as pd
from snowflake.connector.pandas_tools import write_pandas
import snowflake.connector
from sklearn.preprocessing import MinMaxScaler
from google.cloud import storage


def upload_to_gcs(bucket_name, object_name, local_file):
    """Uploads a file to Google Cloud Storage."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    blob.upload_from_filename(local_file)

def extract_from_db(tables_list, season):
    SNOWFLAKE_USER = 'kbharaj3'
    SNOWFLAKE_PASSWORD = 'Snowfl@key0014'
    SNOWFLAKE_ACCOUNT = 'qx25653.ca-central-1.aws'
    SNOWFLAKE_WAREHOUSE = 'FOOTY_STORE'
    SNOWFLAKE_DATABASE = 'GEGENSTATS'
    SNOWFLAKE_SCHEMA = 'TABLES'

    conn = snowflake.connector.connect(
        user=SNOWFLAKE_USER,
        password=SNOWFLAKE_PASSWORD,
        account=SNOWFLAKE_ACCOUNT,
        warehouse=SNOWFLAKE_WAREHOUSE,
        database=SNOWFLAKE_DATABASE,
        schema=SNOWFLAKE_SCHEMA
    )

    cursor = conn.cursor()

    requested_dataframes = []

    for table in tables_list:
        if "_" in table:
            query = f"SELECT * FROM {table} WHERE SEASON = {season}"
        else:
            query = f"SELECT * FROM {table}"
        cursor.execute(query)
        requested_rows = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        requested_df = pd.DataFrame(requested_rows, columns=column_names)
        requested_dataframes.append(requested_df)

    return requested_dataframes

def prepare_team_defending_stats():
    defending_tables = ["TEAMS", "TEAM_DEFENDING_STATS", "TEAM_STANDARD_STATS", "COMPETITIONS"]
    team_names, team_defending, team_standard, df_competitions = extract_from_db(defending_tables, season=2324)

    team_standard = team_standard.merge(team_names, on='TEAM_FBREF_ID', how='left')
    team_defending = team_defending.merge(team_names, on='TEAM_FBREF_ID', how='left')

    team_defending = team_defending.merge(team_standard, on=['TEAM_FBREF_ID', 'SEASON', 'TEAM_NAME', 'COMPETITION'], how='left')
    team_defending = team_defending.merge(df_competitions[['COMPETITION', 'COMPETITION_ACRONYM', 'SEASON']],
                                          on=['COMPETITION', 'SEASON'], how='left')

    team_defending['Clearances/Game'] = team_defending['CLEARANCES'] / team_defending['MATCHES_PLAYED']
    team_defending['Fouls Made/Game'] = team_defending['FOULS_MADE'] / team_defending['MATCHES_PLAYED']
    team_defending['Conceded/Game'] = team_defending['GOALS_CONCEDED'] / team_defending['MATCHES_PLAYED']
    team_defending['xG Against/Game'] = team_defending['XG_AGAINST'] / team_defending['MATCHES_PLAYED']
    team_defending['Tackles Attempted/Game'] = team_defending['TACKLES'] / team_defending['MATCHES_PLAYED']
    team_defending['Tackles Won (%)'] = team_defending['TACKLES_WON'] * 100 / team_defending['TACKLES']
    team_defending['Interceptions/Game'] = team_defending['INTERCEPTIONS'] / team_defending['MATCHES_PLAYED']
    team_defending['Blocked Shots/Game'] = team_defending['BLOCKED_SHOTS'] / team_defending['MATCHES_PLAYED']

    team_defending = team_defending[
        ['SEASON', 'TEAM_NAME', 'COMPETITION_ACRONYM', 'Clearances/Game', 'Fouls Made/Game', 'Conceded/Game',
         'xG Against/Game',
         'Tackles Attempted/Game', 'Tackles Won (%)', 'Interceptions/Game', 'Blocked Shots/Game']]

    team_defending_average = team_defending.drop(columns=['TEAM_NAME']).groupby(
        ['SEASON', 'COMPETITION_ACRONYM']).mean().reset_index()
    team_defending_average['TEAM_NAME'] = team_defending_average['COMPETITION_ACRONYM'] + '_' + team_defending_average[
        'SEASON'].astype(str) + '_Average'
    team_defending = pd.concat([team_defending, team_defending_average], ignore_index=True)

    scaler = MinMaxScaler()
    team_defending_scaled = (team_defending.drop(['TEAM_NAME', 'SEASON', 'COMPETITION_ACRONYM'], axis=1))
    team_defending_scaled_1 = team_defending_scaled.drop(['Fouls Made/Game', 'Conceded/Game', 'xG Against/Game'],axis=1)
    team_defending_scaled_2 = team_defending_scaled[['Fouls Made/Game', 'Conceded/Game', 'xG Against/Game']]
    team_defending_scaled_1 = pd.DataFrame(scaler.fit_transform(team_defending_scaled_1),columns=team_defending_scaled_1.columns)

    scaler = MinMaxScaler()
    team_defending_scaled_2 = pd.DataFrame(scaler.fit_transform(team_defending_scaled_2),columns=team_defending_scaled_2.columns)
    team_defending_scaled_2 = pd.DataFrame(scaler.fit_transform(1 - team_defending_scaled_2),columns=team_defending_scaled_2.columns)
    team_defending_scaled_2 = pd.DataFrame(scaler.inverse_transform(team_defending_scaled_2),columns=team_defending_scaled_2.columns)
    team_defending_scaled = pd.concat([team_defending_scaled_1, team_defending_scaled_2], axis=1)

    team_defending_scaled = pd.concat(
        [team_defending[['TEAM_NAME', 'SEASON', 'COMPETITION_ACRONYM']], team_defending_scaled], axis=1)

    return team_defending, team_defending_scaled

def prepare_team_attacking_stats():
    defending_tables = ["TEAMS", "TEAM_ATTACKING_STATS", "TEAM_STANDARD_STATS", "COMPETITIONS"]
    team_names, team_attacking, team_standard, df_competitions = extract_from_db(defending_tables, season=2324)

    team_standard = team_standard.merge(team_names, on='TEAM_FBREF_ID', how='left')
    team_attacking = team_attacking.merge(team_names, on='TEAM_FBREF_ID', how='left')

    team_attacking = team_attacking.merge(team_standard, on=['TEAM_FBREF_ID', 'SEASON', 'TEAM_NAME', 'COMPETITION'],
                                          how='left')
    team_attacking = team_attacking.merge(df_competitions[['COMPETITION', 'COMPETITION_ACRONYM', 'SEASON']],
                                          on=['COMPETITION', 'SEASON'], how='left')

    team_attacking['Goals/Game'] = team_attacking['GOALS_SCORED'] / team_attacking['MATCHES_PLAYED']
    team_attacking['Pass Completion (%)'] = team_attacking['PASS_COMPLETED'] * 100 / team_attacking['PASS_ATTEMPTED']
    team_attacking['Fouls Against/Game'] = team_attacking['FOULS_AGAINST'] / team_attacking['MATCHES_PLAYED']
    team_attacking['NPxG/Game'] = team_attacking['NPXG'] / team_attacking['MATCHES_PLAYED']
    team_attacking['Shots/Game'] = team_attacking['SHOTS'] / team_attacking['MATCHES_PLAYED']
    team_attacking['Shots On Target (%)'] = team_attacking['SHOTS_ON_TARGET'] * 100 / team_attacking['SHOTS']
    team_attacking['Take_ons Attempted/Game'] = team_attacking['TAKEONS_ATTEMPTED'] / team_attacking['MATCHES_PLAYED']
    team_attacking['Crosses into Pen Area'] = team_attacking['CROSSES_INTO_PA']

    team_attacking = team_attacking[
        ['SEASON', 'TEAM_NAME', 'COMPETITION_ACRONYM', 'Goals/Game', 'Pass Completion (%)', 'Fouls Against/Game', 'NPxG/Game',
         'Shots/Game', 'Shots On Target (%)', 'Take_ons Attempted/Game', 'Crosses into Pen Area']]

    team_attacking_average = team_attacking.drop(columns=['TEAM_NAME']).groupby(['SEASON', 'COMPETITION_ACRONYM']).mean().reset_index()
    team_attacking_average['TEAM_NAME'] = team_attacking_average['COMPETITION_ACRONYM'] + '_' + team_attacking_average[
        'SEASON'].astype(str) + '_Average'
    team_attacking = pd.concat([team_attacking, team_attacking_average], ignore_index=True)

    scaler = MinMaxScaler()
    team_attacking_scaled = (team_attacking.drop(['TEAM_NAME', 'SEASON', 'COMPETITION_ACRONYM'], axis=1))
    team_attacking_scaled = pd.DataFrame(scaler.fit_transform(team_attacking_scaled),
                                         columns=team_attacking_scaled.columns)
    team_attacking_scaled = pd.concat(
        [team_attacking[['TEAM_NAME', 'SEASON', 'COMPETITION_ACRONYM']], team_attacking_scaled], axis=1)


    return team_attacking, team_attacking_scaled

def defending_radar_chart():
    team_defending, team_defending_scaled = prepare_team_defending_stats()

    team_defending_radar_1 = team_defending_scaled.melt(id_vars=["SEASON", "TEAM_NAME", "COMPETITION_ACRONYM"]).sort_values(
        by=["COMPETITION_ACRONYM", "SEASON", "TEAM_NAME"]).rename(columns={'value': 'norm_value'})
    team_defending_radar_2 = team_defending.melt(id_vars=["SEASON", "TEAM_NAME", "COMPETITION_ACRONYM"]).sort_values(
        by=["COMPETITION_ACRONYM", "SEASON", "TEAM_NAME"])

    team_defending_radar = team_defending_radar_1.merge(team_defending_radar_2,
                                                        on=['SEASON', 'TEAM_NAME', 'COMPETITION_ACRONYM', 'variable'],
                                                        how='left')

    team_defending_radar.rename(columns={'variable': 'VARIABLE', 'norm_value': "NORM_VALUE",'value': 'VALUE'}, inplace=True)

    return team_defending_radar

    # file_path = '/tmp/defending_radar_chart.csv'
    # team_defending_radar.to_csv(file_path, index=True)
    #
    # upload_to_gcs('gegenstats', 'defending_radar_chart.csv', file_path)

def attacking_radar_chart():
    team_attacking, team_attacking_scaled = prepare_team_attacking_stats()

    team_attacking_radar_1 = team_attacking_scaled.melt(id_vars=["SEASON", "TEAM_NAME", "COMPETITION_ACRONYM"]).sort_values(
        by=["COMPETITION_ACRONYM", "SEASON", "TEAM_NAME"]).rename(columns={'value': 'norm_value'})
    team_attacking_radar_2 = team_attacking.melt(id_vars=["SEASON", "TEAM_NAME", "COMPETITION_ACRONYM"]).sort_values(
        by=["COMPETITION_ACRONYM", "SEASON", "TEAM_NAME"])

    team_attacking_radar = team_attacking_radar_1.merge(team_attacking_radar_2, on=['SEASON','TEAM_NAME',
                                                                        'COMPETITION_ACRONYM','variable'], how='left')

    team_attacking_radar.rename(columns={'variable': 'VARIABLE', 'norm_value': "NORM_VALUE",
                                         'value': 'VALUE'}, inplace=True)

    return team_attacking_radar

    # file_path = '/tmp/team_attacking_radar.csv'
    # team_attacking_radar.to_csv(file_path, index=True)
    #
    # upload_to_gcs('gegenstats', 'team_attacking_radar.csv', file_path)

def standard_radar_chart():
    team_defending, team_defending_scaled = prepare_team_defending_stats()
    team_attacking, team_attacking_scaled = prepare_team_attacking_stats()

    team_standard_radar_1 = team_attacking_scaled[['TEAM_NAME', 'SEASON', 'COMPETITION_ACRONYM', 'Pass Completion (%)',
                                                   'Goals/Game','NPxG/Game', 'Shots/Game', 'Shots On Target (%)']]
    team_standard_radar_2 = team_attacking[['TEAM_NAME', 'SEASON', 'COMPETITION_ACRONYM', 'Pass Completion (%)',
                                            'Goals/Game', 'NPxG/Game','Shots/Game', 'Shots On Target (%)']]
    team_standard_radar_3 = team_defending_scaled[['TEAM_NAME', 'SEASON', 'COMPETITION_ACRONYM', 'Tackles Won (%)',
                                                   'Conceded/Game','xG Against/Game']]
    team_standard_radar_4 = team_defending[['TEAM_NAME', 'SEASON', 'COMPETITION_ACRONYM', 'Tackles Won (%)',
                                            'Conceded/Game','xG Against/Game']]
    team_standard_radar_5 = team_standard_radar_1.merge(team_standard_radar_3,
                                                        on=['TEAM_NAME', 'SEASON', 'COMPETITION_ACRONYM'], how='left')
    team_standard_radar_6 = team_standard_radar_2.merge(team_standard_radar_4,
                                                        on=['TEAM_NAME', 'SEASON', 'COMPETITION_ACRONYM'], how='left')
    team_standard_radar_5 = team_standard_radar_5.melt(
        id_vars=["SEASON", "TEAM_NAME", "COMPETITION_ACRONYM"]).sort_values(
        by=["COMPETITION_ACRONYM", "SEASON", "TEAM_NAME"]).rename(columns={'value': 'norm_value'})
    team_standard_radar_6 = team_standard_radar_6.melt(
        id_vars=["SEASON", "TEAM_NAME", "COMPETITION_ACRONYM"]).sort_values(
        by=["COMPETITION_ACRONYM", "SEASON", "TEAM_NAME"])
    team_standard_radar = team_standard_radar_5.merge(team_standard_radar_6,on=['SEASON', 'TEAM_NAME', 'COMPETITION_ACRONYM',
                                                      'variable'], how='left')
    team_standard_radar.rename(columns={'variable': 'VARIABLE', 'norm_value': "NORM_VALUE",'value': 'VALUE'}, inplace=True)

    return team_standard_radar
    # file_path = '/tmp/team_standard_radar.csv'
    # team_standard_radar.to_csv(file_path, index=True)
    #
    # upload_to_gcs('gegenstats', 'team_standard_radar.csv', file_path)







