import pandas as pd
from snowflake.connector.pandas_tools import write_pandas
import snowflake.connector
from sklearn.preprocessing import MinMaxScaler
from google.cloud import storage

from plottable import ColDef, Table
from plottable.plots import image
from plottable.cmap import centered_cmap

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import io
import base64
import urllib


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

    # team_defending['Clearances/Game'] = team_defending['CLEARANCES'] / team_defending['MATCHES_PLAYED']
    team_defending['Fouls Made/Game'] = team_defending['FOULS_MADE'] / team_defending['MATCHES_PLAYED']
    team_defending['Conceded/Game'] = team_defending['GOALS_CONCEDED'] / team_defending['MATCHES_PLAYED']
    team_defending['xG Against/Game'] = team_defending['XG_AGAINST'] / team_defending['MATCHES_PLAYED']
    # team_defending['Tackles Attempted/Game'] = team_defending['TACKLES'] / team_defending['MATCHES_PLAYED']
    team_defending['Tackles Won (%)'] = team_defending['TACKLES_WON'] * 100 / team_defending['TACKLES']
    # team_defending['Interceptions/Game'] = team_defending['INTERCEPTIONS'] / team_defending['MATCHES_PLAYED']
    # team_defending['Blocked Shots/Game'] = team_defending['BLOCKED_SHOTS'] / team_defending['MATCHES_PLAYED']
    team_defending['Possession Won'] = team_defending['POSS_WON']
    team_defending['Opposition PPDA'] = team_defending['OPP_PPDA']
    team_defending['Final 3rd Passes Against/Game'] = team_defending['FINAL_3RD_PASSES_AGAINST'] / team_defending[
        'MATCHES_PLAYED']
    team_defending['Clean Sheets'] = team_defending['CLEAN_SHEETS']

    team_defending = team_defending[
        ['SEASON', 'TEAM_NAME', 'COMPETITION_ACRONYM', 'MATCHES_PLAYED', 'Fouls Made/Game', 'Conceded/Game',
         'xG Against/Game', 'Tackles Won (%)', 'Possession Won', 'Opposition PPDA', 'Final 3rd Passes Against/Game',
         'Clean Sheets']]

    team_defending_average = team_defending.drop(columns=['TEAM_NAME']).groupby(
        ['SEASON', 'COMPETITION_ACRONYM']).mean().reset_index()
    team_defending_average['TEAM_NAME'] = team_defending_average['COMPETITION_ACRONYM'] + '_' + team_defending_average[
        'SEASON'].astype(str) + '_Average'
    team_defending = pd.concat([team_defending, team_defending_average], ignore_index=True)

    scaler = MinMaxScaler()
    team_defending_scaled = (
        team_defending.drop(['TEAM_NAME', 'SEASON', 'COMPETITION_ACRONYM', 'MATCHES_PLAYED'], axis=1))
    team_defending_scaled_1 = team_defending_scaled.drop(['Fouls Made/Game', 'Conceded/Game', 'xG Against/Game',
                                                          'Final 3rd Passes Against/Game', 'Opposition PPDA'], axis=1)
    team_defending_scaled_2 = team_defending_scaled[['Fouls Made/Game', 'Conceded/Game', 'xG Against/Game',
                                                     'Final 3rd Passes Against/Game', 'Opposition PPDA']]
    team_defending_scaled_1['Possession Won'] = team_defending_scaled_1['Possession Won'] / team_defending[
        'MATCHES_PLAYED']
    team_defending_scaled_1['Clean Sheets'] = team_defending_scaled_1['Clean Sheets'] / team_defending['MATCHES_PLAYED']
    team_defending_scaled_1 = pd.DataFrame(scaler.fit_transform(team_defending_scaled_1),
                                           columns=team_defending_scaled_1.columns)

    scaler = MinMaxScaler()
    team_defending_scaled_2 = pd.DataFrame(scaler.fit_transform(team_defending_scaled_2), columns=team_defending_scaled_2.columns)
    team_defending_scaled_2 = pd.DataFrame(scaler.fit_transform(1 - team_defending_scaled_2), columns=team_defending_scaled_2.columns)
    team_defending_scaled_2 = pd.DataFrame(scaler.inverse_transform(team_defending_scaled_2), columns=team_defending_scaled_2.columns)
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

def prepare_XPTS_table():
    xpts_tables = ["TEAMS", "TEAM_ATTACKING_STATS", "TEAM_DEFENDING_STATS", "TEAM_STANDARD_STATS", "COMPETITIONS"]

    team_names, team_attacking, team_defending, team_standard, df_competitions = extract_from_db(xpts_tables, season=2324)

    team_standard = team_standard.merge(df_competitions[['COMPETITION', 'COMPETITION_ACRONYM', 'SEASON']],
                                        on=['COMPETITION', 'SEASON'], how='left')
    team_attacking = team_attacking.merge(df_competitions[['COMPETITION', 'COMPETITION_ACRONYM', 'SEASON']],
                                          on=['COMPETITION', 'SEASON'], how='left')
    team_defending = team_defending.merge(df_competitions[['COMPETITION', 'COMPETITION_ACRONYM', 'SEASON']],
                                          on=['COMPETITION', 'SEASON'], how='left')

    df_table = team_standard[['TEAM_FBREF_ID', 'SEASON', 'COMPETITION_ACRONYM', 'TEAM_PTS', 'TEAM_XPTS']]

    df_table = df_table.merge(team_names, how='left', on='TEAM_FBREF_ID')
    df_table = df_table.merge(team_attacking[['TEAM_FBREF_ID', 'SEASON', 'COMPETITION_ACRONYM', 'GOALS_SCORED', 'XG']],
                              on=['TEAM_FBREF_ID', 'COMPETITION_ACRONYM', 'SEASON'])
    df_table = df_table.merge(team_defending[['TEAM_FBREF_ID', 'SEASON', 'COMPETITION_ACRONYM', 'GOALS_CONCEDED',
                                              'XG_AGAINST']], on=['TEAM_FBREF_ID', 'COMPETITION_ACRONYM', 'SEASON'])
    df_table["GOAL_DIFFERENCE"] = df_table["GOALS_SCORED"] - df_table["GOALS_CONCEDED"]

    base64_table = pd.DataFrame(columns=['COMPETITION_ACRONYM', 'SEASON', 'TABLE_BASE64'])

    options = df_table[['COMPETITION_ACRONYM', 'SEASON']].drop_duplicates().reset_index(drop=True)

    colors_pos = [(1, 0, 0), (0, 0.6, 0)]  # Red to Green
    colors_neg = [(0, 0.6, 0), (1, 0, 0)]  # Green to Red
    n_bins = 2  # Discretizes the interpolation into bins
    cmap_name = 'custom_red_green'
    cm_pos = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors_pos, N=n_bins)
    cm_neg = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors_neg, N=n_bins)

    def plus_sign_formatter(value):
        return f"+{value:.0f}" if value > 0 else f"{value:.0f}"


    for index, row in options.iterrows():
        df_filtered = df_table[df_table['COMPETITION_ACRONYM'] == row['COMPETITION_ACRONYM']]
        df_filtered = df_filtered[df_filtered['SEASON'] == row['SEASON']]

        PTS_table = df_filtered[
            ['TEAM_NAME', 'TEAM_PTS', 'GOAL_DIFFERENCE', 'GOALS_SCORED', 'GOALS_CONCEDED']].sort_values(['TEAM_PTS',
                                                                                                         'GOAL_DIFFERENCE',
                                                                                                         'GOALS_SCORED',
                                                                                                         'GOALS_CONCEDED'],
                                                                                                        ascending=False).reset_index(
            drop=True).reset_index()

        PTS_table.rename(columns={'index': 'POS_NUM'}, inplace=True)
        PTS_table.POS_NUM += 1

        ## Change POS_NUM colun in PTS_table from 1,2,3 to 1st, 2nd,3rd etc/
        def ordinal(n):
            return "%d%s" % (n, "tsnrhtdd"[(n // 10 % 10 != 1) * (n % 10 < 4) * n % 10::4])

        PTS_table['POS'] = PTS_table['POS_NUM'].apply(ordinal)

        XPTS_table = df_filtered[['TEAM_NAME', 'TEAM_XPTS', 'GOAL_DIFFERENCE', 'GOALS_SCORED'
            , 'GOALS_CONCEDED']].sort_values(['TEAM_XPTS', 'GOAL_DIFFERENCE', 'GOALS_SCORED', 'GOALS_CONCEDED'],
                                             ascending=False).reset_index(drop=True).reset_index()

        XPTS_table.rename(columns={'index': 'XPTS POS_NUM'}, inplace=True)
        XPTS_table['XPTS POS_NUM'] += 1

        XPTS_table['XPTS POS'] = XPTS_table['XPTS POS_NUM'].apply(ordinal)

        PTS_XPTS_table = XPTS_table.merge(PTS_table, how='left',
                                          on=['TEAM_NAME', 'GOAL_DIFFERENCE', 'GOALS_SCORED', 'GOALS_CONCEDED'])

        PTS_XPTS_table = PTS_XPTS_table[['TEAM_NAME', 'XPTS POS', 'XPTS POS_NUM', 'POS', 'POS_NUM']]

        df_filtered = df_filtered.merge(PTS_XPTS_table, how='left', on='TEAM_NAME')

        df_filtered.rename(columns={'TEAM_LOGO_URL': 'TEAM_LOGO'}, inplace=True)

        import warnings
        warnings.filterwarnings("ignore")

        fin_table = df_filtered[
            ['POS', 'POS_NUM', 'TEAM_LOGO', 'XPTS POS', 'XPTS POS_NUM', 'TEAM_NAME', 'XG', 'GOALS_SCORED', 'XG_AGAINST',
             'GOALS_CONCEDED', 'TEAM_PTS', 'TEAM_XPTS']].sort_values('XPTS POS_NUM')
        fin_table['POS_NUM_DIFF'] = fin_table.apply(
            lambda row: 'https://i.imgur.com/AACUEGy.png' if row['POS_NUM'] == row['XPTS POS_NUM'] else \
                'https://i.imgur.com/5sTTYXm.png' if row['POS_NUM'] < row[
                    'XPTS POS_NUM'] else 'https://i.imgur.com/dGsmsnm.png', axis=1)
        fin_table['XG_DIFF'] = fin_table['GOALS_SCORED'] - fin_table['XG']
        fin_table['XGA_DIFF'] = fin_table['GOALS_CONCEDED'] - fin_table['XG_AGAINST']
        fin_table['XPTS_DIFF'] = fin_table['TEAM_PTS'] - fin_table['TEAM_XPTS']
        fin_table = fin_table[['POS', 'XPTS POS', 'POS_NUM_DIFF', 'TEAM_LOGO', 'TEAM_NAME', 'XG', 'XG_DIFF',
                               'XG_AGAINST', 'XGA_DIFF', 'TEAM_XPTS', 'XPTS_DIFF']]
        fin_table['XG_DIFF'] = fin_table['XG_DIFF'].astype(int)
        fin_table['XGA_DIFF'] = fin_table['XGA_DIFF'].astype(int)
        fin_table['XPTS_DIFF'] = fin_table['XPTS_DIFF'].astype(int)

        fin_table = fin_table.set_index(['POS'])

        # team_name_cols = ['TEAM_LOGO', 'TEAM_NAME']
        # xG_cols = ['XG', 'XG_DIFF']
        # xGA_cols = ['XG_AGAINST', 'XGA_DIFF']
        # xPTS_cols = ['TEAM_XPTS', 'XPTS_DIFF']

        fin_table['TEAM_LOGO'] = fin_table['TEAM_LOGO'].apply(lambda x: (io.BytesIO(urllib.request.urlopen(x).read())))
        fin_table['POS_NUM_DIFF'] = fin_table['POS_NUM_DIFF'].apply(
            lambda x: (io.BytesIO(urllib.request.urlopen(x).read())))

        plt.rcParams["font.family"] = ["Roboto"]
        plt.rcParams["savefig.bbox"] = "tight"
        plt.rcParams["text.color"] = "#e0e8df"

        fig, ax = plt.subplots(figsize=(10, 10))

        ax.set_title('XG TABLE', fontsize=15, fontweight='bold', color='gold',
                     loc='left', pad=15)

        subtitle_y_position = 1.005  # Adjust this value as needed
        ax.text(0.0, subtitle_y_position, f"Premier Division, {row['SEASON']} Season",
                fontsize=11, color='lightgray', transform=ax.transAxes)

        fig.set_facecolor("#464646")
        ax.set_facecolor("#464646")

        fig.subplots_adjust(top=0.85, bottom=0.1, left=0.1, right=0.9)

        col_defs = (
            [
                ColDef(name="POS", title="", group="POS", textprops={"ha": "center", "weight": "bold"}, width=0.3),
                ColDef(name="XPTS POS", title="", group="XPTS POS", textprops={"ha": "center", "weight": "bold"},
                       width=0.3),
                ColDef(name="POS_NUM_DIFF", title="", group="NAME", textprops={"ha": "center"}, width=0.06,
                       plot_fn=image),
                ColDef(name="TEAM_LOGO", title="", group="NAME", textprops={"ha": "center"}, width=0.2, plot_fn=image),
                ColDef(name="TEAM_NAME", title="", group="NAME", textprops={"ha": "left"}, width=0.8),
                ColDef(name="XG", title="", group="XG", textprops={"ha": "center"}, formatter="{:.1f}", width=0.3),
                ColDef(name="XG_DIFF", title="", group="XG",
                       textprops={"ha": "center", "fontsize": 14, "fontweight": "bold",
                                  "bbox": {"boxstyle": "circle,pad=0.1"}},
                       formatter=plus_sign_formatter, width=0.2,
                       cmap=centered_cmap(fin_table["XG_DIFF"], cmap=cm_pos, center=0)),
                ColDef(name="XG_AGAINST", title="", group="XGA", textprops={"ha": "center", }, formatter="{:.1f}",
                       width=0.3),
                ColDef(name="XGA_DIFF", title="", group="XGA",
                       textprops={"ha": "center", "fontsize": 14, "fontweight": "bold",
                                  "bbox": {"boxstyle": "circle,pad=0.1"}},
                       width=0.2, formatter=plus_sign_formatter,
                       cmap=centered_cmap(fin_table["XG_DIFF"], cmap=cm_neg, center=0)),
                ColDef(name="TEAM_XPTS", title="", group="XPTS", textprops={"ha": "center"}, formatter="{:.1f}",
                       width=0.3),
                ColDef(name="XPTS_DIFF", title="", group="XPTS",
                       textprops={"ha": "center", "fontsize": 14, "fontweight": "bold",
                                  "bbox": {"boxstyle": "circle,pad=0.1"}},
                       width=0.2, formatter=plus_sign_formatter,
                       cmap=centered_cmap(fin_table["XG_DIFF"], cmap=cm_pos, center=0)),
            ]
        )

        table = Table(
            fin_table,
            column_definitions=col_defs,
            row_dividers=True,
            footer_divider=True,
            ax=ax,
            textprops={"fontsize": 12},
            row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
            col_label_divider={"linewidth": 1, "linestyle": (0, (1, 5))},
        )

        table.autoset_fontcolors(colnames=["XG_DIFF", "XGA_DIFF", "XPTS_DIFF"])

        buf = io.BytesIO()

        # Save the figure to the buffer
        fig.savefig(buf, format='png')

        # Seek to the start of the buffer
        buf.seek(0)

        encoded_data = base64.b64encode(buf.getvalue()).decode()

        base64_table = pd.concat(
            [base64_table, pd.DataFrame([[row['COMPETITION_ACRONYM'], row['SEASON'], encoded_data]],
                                        columns=['COMPETITION_ACRONYM', 'SEASON', 'TABLE_BASE64'])])

    base64_table = base64_table.rename(columns={'TABLE_BASE64': 'TABLE_IMAGE'})

    return base64_table


def defending_radar_chart():
    team_defending, team_defending_scaled = prepare_team_defending_stats()

    team_defending_radar_1 = team_defending_scaled.melt(id_vars=["SEASON", "TEAM_NAME", "COMPETITION_ACRONYM"]).sort_values(
        by=["COMPETITION_ACRONYM", "SEASON", "TEAM_NAME"]).rename(columns={'value': 'norm_value'})
    team_defending_radar_2 = team_defending.melt(id_vars=["SEASON", "TEAM_NAME", "COMPETITION_ACRONYM"]).sort_values(
        by=["COMPETITION_ACRONYM", "SEASON", "TEAM_NAME"])

    team_defending_radar = team_defending_radar_1.merge(team_defending_radar_2,
                                                        on=['SEASON','TEAM_NAME','COMPETITION_ACRONYM','variable'],
                                                        how='left')

    team_defending_radar.rename(columns={'variable': 'VARIABLE', 'norm_value': "NORM_VALUE",
                                         'value': 'VALUE'}, inplace=True)

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
    team_standard_radar = team_standard_radar_5.merge(team_standard_radar_6,on=['SEASON', 'TEAM_NAME',
                                                    'COMPETITION_ACRONYM', 'variable'], how='left')
    team_standard_radar.rename(columns={'variable': 'VARIABLE', 'norm_value': "NORM_VALUE",'value': 'VALUE'},
                               inplace=True)

    return team_standard_radar
    # file_path = '/tmp/team_standard_radar.csv'
    # team_standard_radar.to_csv(file_path, index=True)
    #
    # upload_to_gcs('gegenstats', 'team_standard_radar.csv', file_path)







