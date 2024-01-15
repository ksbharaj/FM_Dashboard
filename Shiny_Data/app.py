import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import toml
import base64
from PIL import Image

from sqlalchemy import create_engine
import snowflake.connector

import io
import urllib

# import snowflake.connector

from plottable import ColDef, Table
import matplotlib.colors as mcolors
from plottable.cmap import centered_cmap
from plottable.plots import image
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# Snowflake details
SNOWFLAKE_USER = 'kbharaj3'
SNOWFLAKE_ACCOUNT = 'qx25653.ca-central-1.aws'
SNOWFLAKE_WAREHOUSE = 'FOOTY_STORE'
SNOWFLAKE_DATABASE = 'GEGENSTATS'
SNOWFLAKE_SCHEMA = 'RADAR_CHARTS'

if 'STREAMLIT_SHARING_MODE' in os.environ:
    SNOWFLAKE_PASSWORD = st.secrets["snowflake"]["password"]
else:
    local_secrets = toml.load('local_secrets.toml')
    SNOWFLAKE_PASSWORD = local_secrets['SNOWFLAKE_PASSWORD']


conn = snowflake.connector.connect(
    user=SNOWFLAKE_USER,
    password=SNOWFLAKE_PASSWORD,
    account=SNOWFLAKE_ACCOUNT,
    warehouse=SNOWFLAKE_WAREHOUSE,
    database=SNOWFLAKE_DATABASE,
    schema=SNOWFLAKE_SCHEMA
)

def fetch_data(cursor, query):
    cursor.execute(query)
    rows = cursor.fetchall()
    column_names = [desc[0] for desc in cursor.description]
    return pd.DataFrame(rows, columns=column_names)

## Import data from Snowflake
cursor = conn.cursor()

standard_chart_data = fetch_data(cursor, 'SELECT * FROM STANDARD_RADAR')
attacking_chart_data = fetch_data(cursor, 'SELECT * FROM ATTACKING_RADAR')
defending_chart_data = fetch_data(cursor, 'SELECT * FROM DEFENDING_RADAR')
xpts_table_images = fetch_data(cursor, 'SELECT * FROM XPTS_TABLE')

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

team_names = fetch_data(cursor, 'SELECT * FROM TEAMS')
team_misc = fetch_data(cursor, 'SELECT * FROM TEAM_MISC_STATS')
team_standard = fetch_data(cursor, 'SELECT * FROM TEAM_STANDARD_STATS')
team_attacking = fetch_data(cursor, 'SELECT * FROM TEAM_ATTACKING_STATS')
team_defending = fetch_data(cursor, 'SELECT * FROM TEAM_DEFENDING_STATS')
df_competitions = fetch_data(cursor, 'SELECT * FROM COMPETITIONS')
df_seasons = fetch_data(cursor, 'SELECT * FROM SEASONS')

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

# df_table = team_goal_output[['TEAM_FBREF_ID', 'SEASON', 'TEAM_PTS', 'COMPETITION','COMPETITION_ACRONYM','TEAM_XPTS','TEAM_NAME','TEAM_LOGO_URL']].copy()
# df_table = df_table.merge(team_attacking[['TEAM_FBREF_ID', 'SEASON', 'COMPETITION','GOALS_SCORED', 'XG']], 
#                           on=['TEAM_FBREF_ID', 'SEASON','COMPETITION'])
# df_table = df_table.merge(team_defending[['TEAM_FBREF_ID', 'SEASON', 'GOALS_CONCEDED', 'COMPETITION','XG_AGAINST']], 
#                           on=['TEAM_FBREF_ID', 'SEASON','COMPETITION'])
# df_table["GOAL_DIFFERENCE"] = df_table["GOALS_SCORED"] - df_table["GOALS_CONCEDED"]


def plus_sign_formatter(value):
    return f"+{value:.0f}" if value > 0 else f"{value:.0f}"


# Function to create the radar charts
def create_radar_chart(season, team_name, data, competition, chart_name, chart_width=800, chart_height=720, label_spread=2.75):
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
        x = 0.5 + (1.1) * np.cos(angle) / label_spread
        y = 0.48 + (1.1) * np.sin(angle) / 2

        annotation_text = \
        f"<span style='font-size: 13px;'><b>{category}</b></span><br>" \
        f"<span style='font-size: 16px; color: rgba(210, 210, 0, 1);'>{value:.2f}</span>"

        fig.add_annotation(
            x=x,
            y=y,
            xref="x domain",
            yref="paper",
            text=annotation_text,  # Bold category name and value
            showarrow=False,
            font=dict(size=12, color='white'),
            align="center",
            xanchor='center',
            yanchor='middle',
            # sizing="contain",
            bordercolor="rgba(0, 0, 0, 0)"
        )



    # Update layout
    fig.update_layout(
        autosize=False,
        width=chart_width,  # Set the width
        height=chart_height,  # Set the height
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
            'yanchor': 'top',  # Ensures the title is at the top of the y position
            'font': dict(
                family="Roboto, sans-serif",  # Specify the font family
                size=23,                     # Specify the font size
                color="white"                # Specify the font color
            )
        },
        hoverlabel=dict(
            bgcolor="rgba(20, 20, 20, 0.8)",
            font_family="Roboto, sans-serif",
            bordercolor="rgba(20, 20, 20, 0.8)",),
        font=dict(
            family="Roboto, sans-serif",  # Specify the font family
            size=50,                     # Specify the font size
            color="white"                # Specify the font color
        )
    )

    return fig


x_min_aerial = (team_misc['AERIAL DUELS WON RATIO (%)'].min()*0.95)
x_max_aerial = (team_misc['AERIAL DUELS WON RATIO (%)'].max()*1.05)
y_min_aerial = (team_misc['AERIAL DUELS ATTEMPTED PER GAME'].min()*0.95)
y_max_aerial = (team_misc['AERIAL DUELS ATTEMPTED PER GAME'].max()*1.05)

x_min_goal_output = 0.45
x_max_goal_output = 2.60
y_min_goal_output = 0.5
y_max_goal_output = 2.5

# Function to create the scatter charts
def create_FM_team_scatter_chart(df, chart_name, team_name, x_axis_label, y_axis_label, img_size, x_min, x_max, y_min, y_max, bottom_left_label, 
                                 bottom_right_label, top_left_label, top_right_label, bl_color, br_color, tl_color, tr_color):
    fig = go.Figure()
    x_axis_mean_val = df[x_axis_label].mean()
    y_axis_mean_val = df[y_axis_label].mean()

    # Add the scatter plot points
    for index, row in df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row[x_axis_label]],
            y=[row[y_axis_label]],
            mode='markers',
            text=row["TEAM_NAME"],
            marker=dict(
                opacity=0
            ),
            hoverinfo='text',
        ))

        # Add team logo as a layout_image
        fig.add_layout_image(
            dict(
                source=row["TEAM_LOGO_URL"],
                x=row[x_axis_label],
                y=row[y_axis_label],
                xref="x",
                yref="y",
                xanchor="center",
                yanchor="middle"
            )
        )


    # Update axes and layout as necessary
    # ...
        
    fig.update_xaxes(range=[x_min, x_max], title=x_axis_label)
    fig.update_yaxes(range=[y_min, y_max], title=y_axis_label)

    fig.add_shape(
        type='line',
        x0=x_axis_mean_val, y0=fig.layout.yaxis.range[0],  # start of the line
        x1=x_axis_mean_val, y1=fig.layout.yaxis.range[1],  # end of the line
        line=dict(color='White', width=3),
        layer='below'
    )

    # Add a horizontal line at the mean aerials attempted per game
    fig.add_shape(
        type='line',
        x0=fig.layout.xaxis.range[0], y0=y_axis_mean_val,  # start of the line
        x1=fig.layout.xaxis.range[1], y1=y_axis_mean_val,  # end of the line
        line=dict(color='White', width=3),
        layer='below'
    )

    fig.update_layout(
        width=625,
        height=625,
        showlegend=False,
        paper_bgcolor='rgb(70, 70, 70)',
        plot_bgcolor='rgb(70, 70, 70)',
        font=dict(
                family="Roboto, sans-serif",  # Specify the font family
                size=25,                     # Specify the font size
                color="white"                # Specify the font color
            ),
        hoverlabel=dict(
                bgcolor="rgba(20, 20, 20, 0.8)",
                font_family="Roboto, sans-serif"),
        title={
            'text': f'{chart_name}',
            'y':0.98,  # Sets the y position of the title (1 is the top of the figure)
            'x':0.5,  # Centers the title horizontally (0.5 is the center of the figure)
            'xanchor': 'center',  # Ensures the title is centered at the x position
            'yanchor': 'top',  # Ensures the title is at the top of the y position
            'font': dict(
                family="Roboto, sans-serif",  # Specify the font family
                size=23,                     # Specify the font size
                color="white"                # Specify the font color
            )
        },
        margin=dict(l=10, r=30, t=50, b=10),
        images= [dict(
            source= row["TEAM_LOGO_URL"],
            xref="x",
            yref="y",
            x= row[x_axis_label],
            y= row[y_axis_label],
            sizex=img_size*1.8 if row['TEAM_NAME'] == team_name else img_size,  # The size of the image in x axis units
            sizey=img_size*1.8 if row['TEAM_NAME'] == team_name else img_size,  # The size of the image in y axis units
            sizing="contain",
            opacity=1 if row['TEAM_NAME'] == team_name else 0.35,
            layer="above") for index, row in df.iterrows()]
    )

    fig.update_xaxes(
        title=dict(font=dict(size=25)),
        showline=True,  # Show the axis line
        linewidth=2,  # Width of the axis line
        linecolor='white',  # Color of the axis line
        gridcolor='rgba(0,0,0,0)',  # Set grid line color to transparent
        tickfont=dict(color='white', size=15),  # Set the color of the axis ticks (numbers)
    )

    fig.update_yaxes(
        title=dict(font=dict(size=25)),
        showline=True,
        linewidth=2,
        linecolor='white',
        gridcolor='rgba(0,0,0,0)',
        tickfont=dict(color='white', size=15),
    )

    fig.add_annotation(text=bottom_left_label,
                    xref="paper", yref="paper",
                    x=0, y=0,  # Bottom left corner
                    showarrow=False,
                    font=dict(size=15, color=bl_color, family="Roboto, sans-serif"),
                    align="left")

    fig.add_annotation(text=top_left_label,
                    xref="paper", yref="paper",
                    x=0, y=1,  # Top left corner
                    showarrow=False,
                    font=dict(size=15, color=tl_color, family="Roboto, sans-serif"),
                    align="left")

    fig.add_annotation(text=top_right_label,
                    xref="paper", yref="paper",
                    x=1, y=1,  # Top right corner
                    showarrow=False,
                    font=dict(size=15, color=tr_color, family="Roboto, sans-serif"),
                    align="right")

    fig.add_annotation(text=bottom_right_label,
                    xref="paper", yref="paper",
                    x=1, y=0,  # Bottom right corner
                    showarrow=False,
                    font=dict(size=15, color=br_color, family="Roboto, sans-serif"),
                    align="right")

    return fig


# Creation of the Streamit App
st.set_page_config(layout="centered")
css='''
    <style>
        section.main > div {max-width:80rem}
    </style>
    '''
st.markdown(css, unsafe_allow_html=True)
st.title('Team Analytics')


col1, col2, col3 = st.columns([1, 6, 1])

with col1:
    st.image(io.BytesIO(urllib.request.urlopen("https://i.imgur.com/qEwoaGU.png").read()), use_column_width=True)

if 'prev_selected_team_name' not in st.session_state:
    st.session_state.prev_selected_team_name = None

with col2:
    competition = st.selectbox('Select a Competition', standard_chart_data['COMPETITION_ACRONYM'].unique())
    filtered_comp = standard_chart_data[standard_chart_data['COMPETITION_ACRONYM'] == competition]

    ## Select season as desired
    available_seasons = filtered_comp['SEASON'].unique()
    avai_season_names = df_seasons[df_seasons['SEASON'].isin(available_seasons)][['SEASON', 'SEASON_NAME']]

    season_selected = st.selectbox('Select a Season', sorted(avai_season_names['SEASON_NAME'].to_list(), reverse=True))
    season = avai_season_names[avai_season_names['SEASON_NAME'] == season_selected]['SEASON'].iloc[0]

    filtered_standard = filtered_comp[filtered_comp['SEASON'] == season]

    # Get the list of valid team names for the selected competition and season
    valid_team_names = list(sorted([name for name in filtered_standard['TEAM_NAME'].unique() if "Average" not in name]))

    # Determine the default index for the team name selectbox
    default_index = 0
    if st.session_state.prev_selected_team_name in valid_team_names:
        default_index = valid_team_names.index(st.session_state.prev_selected_team_name)

    # Use the same variable 'team_name' for the team name selection
    team_name = st.selectbox('Select a Team', valid_team_names, index=default_index)

    # Update the session state with the currently selected team name
    st.session_state.prev_selected_team_name = team_name

with col3:
    st.image(io.BytesIO(urllib.request.urlopen(str(team_names[team_names['TEAM_NAME'] == team_name].TEAM_LOGO_URL.iloc[0])).read()), use_column_width=True)


tabs = st.tabs(["Radar Charts", "General Charts"])

with tabs[0]:
    # Create a two-column layout
    col1, col2 = st.columns([1.4, 0.75])  # Adjust the ratio if needed

    # Use the first column for the Standard Radar Chart
    with col1:
        fig_standard = create_radar_chart(season, team_name, standard_chart_data, competition, "Standard Radar Chart") 
        st.plotly_chart(fig_standard, use_container_width=False)  # Set to True to use the full width of the column

    # Use the second column for the Attacking and Defending Radar Charts
    with col2:
        fig_attacking = create_radar_chart(season, team_name, attacking_chart_data, competition, "Attacking Radar Chart", 420, 350,
                                           label_spread=3)
        st.plotly_chart(fig_attacking, use_container_width=False)  # Set to True to use the full width of the column

        fig_defending = create_radar_chart(season, team_name, defending_chart_data, competition, "Defending Radar Chart", 420, 350,
                                           label_spread=3)
        st.plotly_chart(fig_defending, use_container_width=False)  # Set to True to use the full width of the column

# Future Content Tab
with tabs[1]:
    col1, col2 = st.columns([1, 1])  # Adjust the ratio if needed

    with col1:
        filtered_misc = team_misc[team_misc['SEASON'] == season]
        filtered_misc = filtered_misc[filtered_misc['COMPETITION_ACRONYM'] == competition]
        fig_team_aerial_duels = create_FM_team_scatter_chart(filtered_misc, 'AERIAL', team_name, 'AERIAL DUELS WON RATIO (%)', 'AERIAL DUELS ATTEMPTED PER GAME', 
                                                            1.45, x_min_aerial, x_max_aerial, y_min_aerial, y_max_aerial, 
                                                            "Fewer Duels<br>Poor Dueling", "Fewer Duels<br>Strong Dueling",
                                                            "Lots of Duels<br>Poor Dueling", "Lots of Duels<br>Strong Dueling", "red", 
                                                            "orange", "orange", "green")
        st.plotly_chart(fig_team_aerial_duels, use_container_width=False)


    with col2:
        filtered_goal_output = team_goal_output[team_goal_output['SEASON'] == season]
        filtered_goal_output = filtered_goal_output[filtered_goal_output['COMPETITION_ACRONYM'] == competition]
        fig_team_aerial_duels = create_FM_team_scatter_chart(filtered_goal_output, 'GOAL OUTPUT', team_name, 'EXPECTED GOALS AGAINST PER GAME', 'NON PENALTY EXPECTED GOALS PER GAME', 
                                                            0.1, x_min_goal_output, x_max_goal_output, y_min_goal_output, y_max_goal_output, 
                                                            "Low non-penalty expected goals<br>Strong Defending", "Low non-penalty expected goals<br>Poor Defending",
                                                            "High non-penalty expected goals<br>Strong Defending", "High non-penalty expected goals<br>Poor Defending", 
                                                            "orange", "red", "green", "orange")
        st.plotly_chart(fig_team_aerial_duels, use_container_width=False)

    col3, col4, col5 = st.columns([1, 4, 1])  # Adjust the ratio if needed

    @st.cache_data
    def load_image(competition, season):
        binary_data = base64.b64decode(xpts_table_images[(xpts_table_images['COMPETITION_ACRONYM'] == competition) &
        (xpts_table_images['SEASON'] == season)]['TABLE_IMAGE'].iloc[0])
        image_buffer = io.BytesIO(binary_data)
        image = Image.open(image_buffer)
        return image
    
    with col4:
        image = load_image(competition, season)
        st.image(image, use_column_width=True)


# st.sidebar.caption("Note: Expand the plot for the best viewing experience.")


