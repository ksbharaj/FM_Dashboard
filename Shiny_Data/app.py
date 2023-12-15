import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sqlalchemy import create_engine
import snowflake.connector

import snowflake.connector

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

SNOWFLAKE_SCHEMA = 'FBREF_TEAMSTATS'

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

team_logos = \
[["18bb7c10", "https://i.imgur.com/SURo5sj.png"], ["8602292d", "https://i.imgur.com/M4mwH1X.png"],
 ["4ba7cbea", "https://i.imgur.com/0F6UIO4.png"], ["cd051869", "https://i.imgur.com/MOhGOQ4.png"],
 ["d07537b9", "https://i.imgur.com/EMCwD3X.png"], ["cff3d9bb", "https://i.imgur.com/6F7h3UR.png"],
 ["47c64c55", "https://i.imgur.com/LPhzr0K.png"], ["d3fd31cc", "https://i.imgur.com/Dgei0uj.png"],
 ["fd962109", "https://i.imgur.com/2zggSrF.png"], ["5bfb9659", "https://i.imgur.com/2tipH85.png"],
 ["a2d435b3", "https://i.imgur.com/ldOUO84.png"], ["822bd0ba", "https://i.imgur.com/fs7VK9G.png"],
 ["b8fd03ef", "https://i.imgur.com/FF8z7uZ.png"], ["19538871", "https://i.imgur.com/z47emBQ.png"],
 ["b2b47a98", "https://i.imgur.com/3kOLokZ.png"], ["e4a775cb", "https://i.imgur.com/z4Tl9Zu.png"],
 ["33c895d4", "https://i.imgur.com/35GrgJC.png"], ["361ca564", "https://i.imgur.com/fJ7rUWW.png"],
 ["7c21e445", "https://i.imgur.com/9BQHJRl.png"], ["8cec06e1", "https://i.imgur.com/6azSF88.png"]
 ]

team_logos = pd.DataFrame(team_logos, columns=['TEAM_FBREF_ID', 'LOGO_URL'])
team_misc = team_misc.merge(team_names, on='TEAM_FBREF_ID', how='left')
team_misc = team_misc.merge(team_standard, on=['TEAM_FBREF_ID', 'SEASON'], how='left')
team_misc = team_misc.merge(team_logos, on=['TEAM_FBREF_ID'], how='left')
team_misc['AERIAL DUELS WON RATIO (%)'] = team_misc['AERIALS_WON']*100/(team_misc['AERIALS_WON'] + 
                                                                team_misc['AERIALS_LOST'])
team_misc['AERIAL DUELS ATTEMPTED PER GAME'] = (team_misc['AERIALS_WON'] + 
                                           team_misc['AERIALS_LOST'])/team_misc['MATCHES_PLAYED']


team_goal_output = team_standard.merge(team_defending[['TEAM_FBREF_ID', 'SEASON', 'XG_AGAINST']], 
                                       on=['TEAM_FBREF_ID', 'SEASON'], how='left')
team_goal_output = team_goal_output.merge(team_attacking[['TEAM_FBREF_ID', 'SEASON', 'NPXG']], 
                                          on=['TEAM_FBREF_ID', 'SEASON'], how='left')
team_goal_output['EXPECTED GOALS AGAINST PER GAME'] = team_goal_output['XG_AGAINST']/team_goal_output['MATCHES_PLAYED']
team_goal_output['NON PENALTY EXPECTED GOALS PER GAME'] = team_goal_output['NPXG']/team_goal_output['MATCHES_PLAYED']
team_goal_output = team_goal_output.merge(team_names, on='TEAM_FBREF_ID', how='left')
team_goal_output = team_goal_output.merge(team_logos, on='TEAM_FBREF_ID', how='left')


def create_radar_chart(season, team_name, data, chart_name):
    team_data = data[data['TEAM_NAME'] == team_name]
    average_data = data[data['TEAM_NAME'] == 'Average_2223']

    team_data = team_data[team_data['SEASON'] == season]
    average_data = average_data[average_data['SEASON'] == season]

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


x_min_aerial = (team_misc['AERIAL DUELS WON RATIO (%)'].min()*0.95)
x_max_aerial = (team_misc['AERIAL DUELS WON RATIO (%)'].max()*1.05)
y_min_aerial = (team_misc['AERIAL DUELS ATTEMPTED PER GAME'].min()*0.95)
y_max_aerial = (team_misc['AERIAL DUELS ATTEMPTED PER GAME'].max()*1.05)

x_min_goal_output = (team_goal_output['EXPECTED GOALS AGAINST PER GAME'].min()*0.925)
x_max_goal_output = (team_goal_output['EXPECTED GOALS AGAINST PER GAME'].max()*1.075)
y_min_goal_output = (team_goal_output['NON PENALTY EXPECTED GOALS PER GAME'].min()*0.85)
y_max_goal_output = (team_goal_output['NON PENALTY EXPECTED GOALS PER GAME'].max()*1.07)

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
                size=1,  # Set a fixed size or normalize as you prefer
                opacity=0
            ),
            hoverinfo='text',
        ))

        # Add team logo as a layout_image
        fig.add_layout_image(
            dict(
                source=row["LOGO_URL"],
                x=row[x_axis_label],
                y=row[y_axis_label],
                xref="x",
                yref="y",
                sizex=5,  # Adjust the size of the image here
                sizey=5,  # Adjust the size of the image here
                xanchor="center",
                yanchor="middle"
            )
        )

        if row['TEAM_NAME'] == team_name:
            # Highlight the selected team
            fig.add_shape(
                type="circle",
                xref="x",
                yref="y",
                x0=row[x_axis_label] - img_size/1.75,  # Adjust size and position as needed
                y0=row[y_axis_label] - img_size/1.75,
                x1=row[x_axis_label] + img_size/1.75,
                y1=row[y_axis_label] + img_size/1.75,
                line_color="red",  # Choose a distinctive color
                line_width=3,
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
        width=700,
        height=700,
        showlegend=False,
        paper_bgcolor='rgb(70, 70, 70)',
        plot_bgcolor='rgb(70, 70, 70)',
        font=dict(
                family="Roboto, sans-serif",  # Specify the font family
                size=15,                     # Specify the font size
                color="white"                # Specify the font color
            ),
        hoverlabel=dict(
                bgcolor="rgba(20, 20, 20, 0.8)",
                font_family="Roboto, sans-serif"),
        title={
            'text': f'{chart_name}',
            'y':0.95,  # Sets the y position of the title (1 is the top of the figure)
            'x':0.5,  # Centers the title horizontally (0.5 is the center of the figure)
            'xanchor': 'center',  # Ensures the title is centered at the x position
            'yanchor': 'top'  # Ensures the title is at the top of the y position
        },
        margin=dict(l=10, r=30, t=10, b=10),
        images= [dict(
            source= row["LOGO_URL"],
            xref="x",
            yref="y",
            x= row[x_axis_label],
            y= row[y_axis_label],
            sizex=img_size,  # The size of the image in x axis units
            sizey=img_size,  # The size of the image in y axis units
            sizing="contain",
            layer="above") for index, row in df.iterrows()]
    )

    fig.update_xaxes(
        showline=True,  # Show the axis line
        linewidth=2,  # Width of the axis line
        linecolor='white',  # Color of the axis line
        gridcolor='rgba(0,0,0,0)',  # Set grid line color to transparent
        tickfont=dict(color='white')  # Set the color of the axis ticks (numbers)
    )

    fig.update_yaxes(
        showline=True,
        linewidth=2,
        linecolor='white',
        gridcolor='rgba(0,0,0,0)',
        tickfont=dict(color='white')
    )

    fig.add_annotation(text=bottom_left_label,
                    xref="paper", yref="paper",
                    x=0, y=0,  # Bottom left corner
                    showarrow=False,
                    font=dict(size=14, color=bl_color, family="Roboto, sans-serif"),
                    align="left")

    fig.add_annotation(text=top_left_label,
                    xref="paper", yref="paper",
                    x=0, y=1,  # Top left corner
                    showarrow=False,
                    font=dict(size=14, color=tl_color, family="Roboto, sans-serif"),
                    align="left")

    fig.add_annotation(text=top_right_label,
                    xref="paper", yref="paper",
                    x=1, y=1,  # Top right corner
                    showarrow=False,
                    font=dict(size=14, color=tr_color, family="Roboto, sans-serif"),
                    align="right")

    fig.add_annotation(text=bottom_right_label,
                    xref="paper", yref="paper",
                    x=1, y=0,  # Bottom right corner
                    showarrow=False,
                    font=dict(size=14, color=br_color, family="Roboto, sans-serif"),
                    align="right")

    return fig


st.title('Team Analytics')
team_name = st.selectbox('Select a Team', list(sorted(np.delete(standard_chart_data['TEAM_NAME'].unique(), 
                                                         np.where(standard_chart_data['TEAM_NAME'].unique() == 'Average_2223')))))
season = st.selectbox('Select a Season', (standard_chart_data['SEASON'].unique()))

# col1, col2 = st.columns(2)

tabs = st.tabs(["Radar Charts", "General"])

with tabs[0]:
    # Standard Radar Chart
    fig_standard = create_radar_chart(season, team_name, standard_chart_data, "Standard Radar Chart") 
    st.plotly_chart(fig_standard, use_container_width=True)

    # Attacking Radar Chart
    fig_attacking = create_radar_chart(season, team_name, attacking_chart_data, "Attacking Radar Chart")
    st.plotly_chart(fig_attacking, use_container_width=True)

    # Defending Radar Chart
    fig_defending = create_radar_chart(season, team_name, defending_chart_data, "Defending Radar Chart")
    st.plotly_chart(fig_defending, use_container_width=True)

# Future Content Tab
with tabs[1]:
    fig_team_aerial_duels = create_FM_team_scatter_chart(team_misc, 'AERIAL', team_name, 'AERIAL DUELS WON RATIO (%)', 'AERIAL DUELS ATTEMPTED PER GAME', 
                                                         0.75, x_min_aerial, x_max_aerial, y_min_aerial, y_max_aerial, 
                                                         "Fewer Duels<br>Poor Dueling", "Fewer Duels<br>Strong Dueling",
                                                         "Lots of Duels<br>Poor Dueling", "Lots of Duels<br>Strong Dueling", "red", 
                                                         "orange", "orange", "green")
    st.plotly_chart(fig_team_aerial_duels, use_container_width=False)

    fig_team_aerial_duels = create_FM_team_scatter_chart(team_goal_output, 'GOAL OUTPUT', team_name, 'EXPECTED GOALS AGAINST PER GAME', 'NON PENALTY EXPECTED GOALS PER GAME', 
                                                         0.065, x_min_goal_output, x_max_goal_output, y_min_goal_output, y_max_goal_output, 
                                                         "Low non-penalty expected goals<br>Strong Defending", "Low non-penalty expected goals<br>Poor Defending",
                                                         "High non-penalty expected goals<br>Strong Defending", "High non-penalty expected goals<br>Poor Defending", 
                                                         "orange", "red", "green", "orange")
    st.plotly_chart(fig_team_aerial_duels, use_container_width=False)

# st.sidebar.caption("Note: Expand the plot for the best viewing experience.")


