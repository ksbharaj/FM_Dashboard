import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sqlalchemy import create_engine
import snowflake.connector

import io
import urllib

import snowflake.connector

from plottable import ColDef, Table
import matplotlib.colors as mcolors
from plottable.cmap import centered_cmap
from plottable.plots import image
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# Snowflake details
SNOWFLAKE_USER = 'kbharaj3'
SNOWFLAKE_PASSWORD = 'Snowfl@key0014'
SNOWFLAKE_ACCOUNT = 'qx25653.ca-central-1.aws'
SNOWFLAKE_WAREHOUSE = 'FOOTY_STORE'
SNOWFLAKE_DATABASE = 'GEGENSTATS'
SNOWFLAKE_SCHEMA = 'RADAR_CHARTS'

conn = snowflake.connector.connect(
    user=SNOWFLAKE_USER,
    password=SNOWFLAKE_PASSWORD,
    account=SNOWFLAKE_ACCOUNT,
    warehouse=SNOWFLAKE_WAREHOUSE,
    database=SNOWFLAKE_DATABASE,
    schema=SNOWFLAKE_SCHEMA
)

# Import data from Snowflake
cursor = conn.cursor()

cursor.execute('SELECT * FROM STANDARD_RADAR')
standad_chart_rows = cursor.fetchall()
column_names = [desc[0] for desc in cursor.description]
standard_chart_data = pd.DataFrame(standad_chart_rows, columns=column_names)

cursor.execute('SELECT * FROM ATTACKING_RADAR')
attacking_chart_rows = cursor.fetchall()
column_names = [desc[0] for desc in cursor.description]
attacking_chart_data = pd.DataFrame(attacking_chart_rows, columns=column_names)

cursor.execute('SELECT * FROM DEFENDING_RADAR')
defending_chart_rows = cursor.fetchall()
column_names = [desc[0] for desc in cursor.description]
defending_chart_data = pd.DataFrame(defending_chart_rows, columns=column_names)

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

cursor.execute('SELECT * FROM TEAMS')
team_rows = cursor.fetchall()
column_names = [desc[0] for desc in cursor.description]
team_names = pd.DataFrame(team_rows, columns=column_names)

cursor.execute('SELECT * FROM TEAM_MISC_STATS')
misc_rows = cursor.fetchall()
column_names = [desc[0] for desc in cursor.description]
team_misc = pd.DataFrame(misc_rows, columns=column_names)

cursor.execute('SELECT * FROM TEAM_STANDARD_STATS')
standard_rows = cursor.fetchall()
column_names = [desc[0] for desc in cursor.description]
team_standard = pd.DataFrame(standard_rows, columns=column_names)

cursor.execute('SELECT * FROM TEAM_ATTACKING_STATS')
attacking_rows = cursor.fetchall()
column_names = [desc[0] for desc in cursor.description]
team_attacking = pd.DataFrame(attacking_rows, columns=column_names)

cursor.execute('SELECT * FROM TEAM_DEFENDING_STATS')
defending_rows = cursor.fetchall()
column_names = [desc[0] for desc in cursor.description]
team_defending = pd.DataFrame(defending_rows, columns=column_names)

cursor.execute('SELECT * FROM COMPETITIONS')
competition_rows = cursor.fetchall()
column_names = [desc[0] for desc in cursor.description]
df_competitions = pd.DataFrame(competition_rows, columns=column_names)

# Create a Miscellaneous DataFrame
team_misc = team_misc.merge(df_competitions[['COMPETITION','COMPETITION_ACRONYM','SEASON']], on=['COMPETITION','SEASON'], how='left')
team_misc = team_misc.merge(team_names, on='TEAM_FBREF_ID', how='left')
team_misc = team_misc.merge(team_standard, on=['TEAM_FBREF_ID', 'SEASON', 'COMPETITION'], how='left')
team_misc['AERIAL DUELS WON RATIO (%)'] = team_misc['AERIALS_WON']*100/(team_misc['AERIALS_WON'] + team_misc['AERIALS_LOST'])
team_misc['AERIAL DUELS ATTEMPTED PER GAME'] = (team_misc['AERIALS_WON'] + team_misc['AERIALS_LOST'])/team_misc['MATCHES_PLAYED']


team_goal_output = team_standard.merge(team_defending[['TEAM_FBREF_ID', 'SEASON', 'COMPETITION', 'XG_AGAINST']], 
                                       on=['TEAM_FBREF_ID', 'SEASON', 'COMPETITION'], how='left')
team_goal_output = team_goal_output.merge(df_competitions[['COMPETITION','COMPETITION_ACRONYM','SEASON']], 
                                      on=['COMPETITION','SEASON'], how='left')
team_goal_output = team_goal_output.merge(team_attacking[['TEAM_FBREF_ID', 'SEASON', 'COMPETITION','NPXG']], 
                                          on=['TEAM_FBREF_ID', 'SEASON', 'COMPETITION'], how='left')
team_goal_output['EXPECTED GOALS AGAINST PER GAME'] = team_goal_output['XG_AGAINST']/team_goal_output['MATCHES_PLAYED']
team_goal_output['NON PENALTY EXPECTED GOALS PER GAME'] = team_goal_output['NPXG']/team_goal_output['MATCHES_PLAYED']
team_goal_output = team_goal_output.merge(team_names, on='TEAM_FBREF_ID', how='left')

df_table = team_goal_output[['TEAM_FBREF_ID', 'SEASON', 'TEAM_PTS', 'COMPETITION','COMPETITION_ACRONYM','TEAM_XPTS','TEAM_NAME','TEAM_LOGO_URL']].copy()
df_table = df_table.merge(team_attacking[['TEAM_FBREF_ID', 'SEASON', 'COMPETITION','GOALS_SCORED', 'XG']], 
                          on=['TEAM_FBREF_ID', 'SEASON','COMPETITION'])
df_table = df_table.merge(team_defending[['TEAM_FBREF_ID', 'SEASON', 'GOALS_CONCEDED', 'COMPETITION','XG_AGAINST']], 
                          on=['TEAM_FBREF_ID', 'SEASON','COMPETITION'])
df_table["GOAL_DIFFERENCE"] = df_table["GOALS_SCORED"] - df_table["GOALS_CONCEDED"]


def plus_sign_formatter(value):
    return f"+{value:.0f}" if value > 0 else f"{value:.0f}"


# Function to create the radar charts
def create_radar_chart(season, team_name, data, competition, chart_name):
    team_data = data[data['TEAM_NAME'] == team_name]
    team_data = team_data[team_data['COMPETITION_ACRONYM'] == competition]
    average_data = data[data['TEAM_NAME'] == competition+'_'+str(season)+"_Average"]

    team_data = team_data[team_data['SEASON'] == season]

    # Prepare data for plotting
    categories = team_data['VARIABLE']
    norm_values = team_data['NORM_VALUE']
    average_norm_values = average_data['NORM_VALUE']
    values = team_data['VALUE']

    difference = [golden - average for golden, average in zip(norm_values, average_norm_values)]

    difference_messages = []
    for diff in difference:
        if diff > 0:
            message = f"{diff:.2f} greater than the average"
        elif diff < 0:
            message = f"{abs(diff):.2f} lower than the average"
        else:
            message = "equal to the average"
        difference_messages.append(message)

    hover_text = (
        "<span style='font-size: 20px; color: #d3d3d3;'>%{theta}</span><br>"
        "<span style='font-size: 18px; color: white;'>Value: %{customdata[0]:.2f}</span><br>"
        "<span style='font-size: 12px; color: #d3d3d3;'>%{customdata[2]}</span><extra></extra>"
    )

    # Create radar chart
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=average_norm_values,
        theta=categories,
        name='AVERAGE',
        fillcolor='rgba(100, 100, 100, 0.65)',  # Different color for distinction
        line_color='rgba(100, 100, 100, 1)', # Line colour for Average plot
        fill='toself',
        customdata=np.stack((values, difference, difference_messages), axis=-1),
        hovertemplate=hover_text,
        marker=dict(
            size=1  # Hides the markers by setting their size to zero
        )
    ))

    fig.add_trace(go.Scatterpolar(
        r=norm_values,
        theta=categories,
        name=team_name,
        opacity=0.6,
        fillcolor='rgba(210, 210, 0, 0.6)',  # Adjusted for lighter opaque fill
        line_color='rgba(210, 210, 0, 1)',  # Adjusted for lighter line color
        fill='toself',
        customdata=np.stack((values, difference, difference_messages), axis=-1),
        hovertemplate=hover_text,
        marker=dict(
            size=1  # Hides the markers by setting their size to zero
        )
    ))

    fig.add_layout_image(
        dict(
            source='https://i.imgur.com/9yKFcv4.png',
            xref="paper", yref="paper",
            xanchor="center", yanchor="middle",
            x=0.5, y=0.484,
            sizex=1.06, sizey=1.06,
            opacity=0.7,  # Adjust opacity as needed
            layer="below",
            sizing="contain"
        )
    )

    for i, (value, category) in enumerate(zip(values, categories)):
        angle = (i / float(len(categories))) * 2 * np.pi 
        x = 0.5 + (1.1) * np.cos(angle) / 4
        y = 0.48 + (1.1) * np.sin(angle) / 2

        annotation_text = \
        f"<span style='font-size: 12px;'><b>{category}</b></span><br>" \
        f"<span style='font-size: 15px; color: rgba(210, 210, 0, 1);'>{value:.2f}</span>"

        fig.add_annotation(
            x=x,
            y=y,
            xref="x domain",
            yref="paper",
            text=annotation_text,  # Bold category name and value
            showarrow=False,
            font=dict(size=10, color='white'),
            align="center",
            xanchor='center',
            yanchor='middle',
            # sizing="contain",
            bordercolor="rgba(0, 0, 0, 0)",
        )



    # Update layout
    fig.update_layout(
        # autosize=False,
        # width=355*1,  # Set the width
        # height=400,  # Set the height
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(
                visible=False,
                range=[0, 1],
                linecolor='rgba(17, 17, 17, 1)',
                showline=False,
                gridcolor='white'
            ),
            angularaxis=dict(
                showline=False,  # Hide angular axis line
                gridcolor='rgba(0,0,0,0)',
                showticklabels=False  
            )
        ),
        paper_bgcolor='rgb(70, 70, 70)',      
        showlegend=False,
        title={
            'text': f'{chart_name} for {team_name}',
            'y':0.95,  # Sets the y position of the title (1 is the top of the figure)
            'x':0.5,  # Centers the title horizontally (0.5 is the center of the figure)
            'xanchor': 'center',  # Ensures the title is centered at the x position
            'yanchor': 'top'  # Ensures the title is at the top of the y position
        },
        hoverlabel=dict(
            bgcolor="rgba(20, 20, 20, 0.8)",
            font_family="Roboto, sans-serif",
            bordercolor="rgba(20, 20, 20, 0.8)",),
        font=dict(
            family="Roboto, sans-serif",  # Specify the font family
            size=15,                     # Specify the font size
            color="white"                # Specify the font color
        )
    )

    return fig




