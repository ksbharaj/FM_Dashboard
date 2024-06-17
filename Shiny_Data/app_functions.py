import snowflake.connector
import snowflake.connector
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os

import matplotlib.pyplot as plt
from mplsoccer import Pitch, VerticalPitch
import matplotlib.patches as patches
import math
from datetime import datetime

import matplotlib as mpl
from matplotlib.colors import Normalize


def get_snowflake_cursor(schema):
    """
    Returns a Snowflake cursor object connected to the specified schema.
    
    Parameters:
    schema (str): The name of the Snowflake schema to connect to.
    
    Returns:
    snowflake.connector.cursor.SnowflakeCursor: A cursor object connected to the specified schema.
    """
    try:
        SNOWFLAKE_PASSWORD = st.secrets["snowflake"]["password"]
    except:
        SNOWFLAKE_PASSWORD = os.getenv('SNOWFLAKE_PASSWORD')
    conn = snowflake.connector.connect(
        user='karan14',
        password=SNOWFLAKE_PASSWORD,
        account='lv65293.ca-central-1.aws',
        warehouse='COMPUTE_WH',
        database='GEGENSTATS',
        schema=schema
    )

    return conn.cursor()

@st.cache_data
def fetch_data(_cursor, query, params=None):
    """
    Fetches data from Snowflake using the provided cursor and query.
    
    Parameters:
    cursor (snowflake.connector.cursor.SnowflakeCursor): The Snowflake cursor object.
    query (str): The SQL query to execute.
    
    Returns:
    pandas.DataFrame: The fetched data as a pandas DataFrame.
    """
    if params:
        _cursor.execute(query)
    else:
        _cursor.execute(query)
    rows = _cursor.fetchall()
    column_names = [desc[0] for desc in _cursor.description]
    return pd.DataFrame(rows, columns=column_names)

def create_radar_chart(season, team_name, data, competition, chart_name, chart_width=800, chart_height=720, label_spread=2.75):
    """
    Creates a radar chart using the given data.

    Parameters:
    - season (int): The season for which the chart is created.
    - team_name (str): The name of the team.
    - data (pandas.DataFrame): The data used to create the chart.
    - competition (str): The name of the competition.
    - chart_name (str): The name of the chart.
    - chart_width (int, optional): The width of the chart in pixels. Default is 800.
    - chart_height (int, optional): The height of the chart in pixels. Default is 720.
    - label_spread (float, optional): The spread of the labels on the chart. Default is 2.75.

    Returns:
    - fig (plotly.graph_objects.Figure): The radar chart figure.
    """

    team_data = data[data['TEAM_NAME'] == team_name]
    team_data = team_data[team_data['COMPETITION_ACRONYM'] == competition]
    average_data = data[data['TEAM_NAME'] == competition+'_'+str(season)+"_Average"]

    team_data = team_data[team_data['SEASON'] == season]

    categories = team_data['VARIABLE']
    norm_values = team_data['NORM_VALUE']
    average_values = average_data['VALUE']
    average_norm_values = average_data['NORM_VALUE']
    values = team_data['VALUE']

    difference = [golden - average for golden, average in zip(values, average_values)]

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

def create_FM_team_scatter_chart(df, chart_name, team_name, x_axis_label, y_axis_label, img_size, x_min, x_max, y_min, y_max, bottom_left_label, 
                                 bottom_right_label, top_left_label, top_right_label, bl_color, br_color, tl_color, tr_color):
    """
    Creates a scatter chart with team logos using the given data.

    Parameters:
    - df (pandas.DataFrame): The data used to create the chart.
    - chart_name (str): The name of the chart.
    - team_name (str): The name of the team.
    - x_axis_label (str): The label for the x-axis.
    - y_axis_label (str): The label for the y-axis.
    - img_size (int): The size of the team logos.
    - x_min (float): The minimum value for the x-axis.
    - x_max (float): The maximum value for the x-axis.
    - y_min (float): The minimum value for the y-axis.
    - y_max (float): The maximum value for the y-axis.
    - bottom_left_label (str): The label for the bottom left corner.
    - bottom_right_label (str): The label for the bottom right corner.
    - top_left_label (str): The label for the top left corner.
    - top_right_label (str): The label for the top right corner.
    - bl_color (str): The color for the bottom left corner.
    - br_color (str): The color for the bottom right corner.
    - tl_color (str): The color for the top left corner.
    - tr_color (str): The color for the top right corner.

    Returns:
    - fig (plotly.graph_objects.Figure): The scatter chart figure.
    """

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
                family="Roboto",  # Specify the font family
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
        title=dict(font=dict(size=18), standoff=2),
        showline=True,  # Show the axis line
        linewidth=2,  # Width of the axis line
        linecolor='white',  # Color of the axis line
        gridcolor='rgba(0,0,0,0)',  # Set grid line color to transparent
        tickfont=dict(color='white', size=15),  # Set the color of the axis ticks (numbers)
    )

    fig.update_yaxes(
        title=dict(font=dict(size=18), standoff=2),
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

def plot_defensive_actions(section_counts_percentage_filt):
    """
    Plot defensive actions on a soccer pitch.

    Args:
        section_counts_percentage_filt (pandas.DataFrame): A DataFrame containing the section counts and percentages.

    Returns:
        matplotlib.figure.Figure: The generated figure.

    Raises:
        None

    """

    sections = ["(0.0, 17.5]", "(17.5, 35.0]", "(35.0, 52.5]", "(52.5, 70.0]", "(70.0, 87.5]", "(87.5, 105.0]"]
    percentages = (section_counts_percentage_filt[['SECTION_1', 'SECTION_2', 'SECTION_3', 'SECTION_4', 'SECTION_5', 'SECTION_6']].iloc[0].values)

    max_val = math.ceil(max(percentages) / 5) * 5

    def get_green_color(percentage):
        green_intensity = int((percentage / max_val) * 255)
        return f'#00{green_intensity:02x}00'

    pitch = Pitch(pitch_color='#2B2B2B', line_color='white', goal_type='box', pitch_type='uefa', linewidth=1)
    fig, ax = pitch.draw(figsize=(12, 7))

    fig.patch.set_facecolor('#2B2B2B')

    gap_width = 0.3

    for i, percentage in enumerate(percentages):
        start_pos = float(sections[i].split(', ')[0][1:])
        end_pos = float(sections[i].split(', ')[1][:-1])

        if i != 0:  # not the first bar
            start_pos += gap_width

        section_width = end_pos - start_pos
        if i != len(percentages) - 1:  # not the last bar
            section_width -= gap_width

        color = get_green_color(percentage)

        rect = patches.Rectangle((start_pos, 0), section_width, 68,
                                 linewidth=1, edgecolor='black', facecolor=color, alpha=0.6, zorder=2)
        ax.add_patch(rect)

        ax.text(start_pos + section_width / 2, 8, str(int(percentage)) + '%', fontproperties='Roboto',
                va='center', ha='center', color='white', fontsize=20, zorder=3)

    scale_height = 3  # Reduced height of the scale rectangles by half
    scale_y_position = -10  # Position of the scale rectangles (negative to be below the pitch)
    scale_length = 105 / 2.5  # Half the pitch length for scale
    scale_start = (105 - scale_length) / 2.25  # Centering the scale

    for i in range(0, max_val + 1, 5):
        start_pos = scale_start + (i / max_val) * scale_length
        section_width = scale_length / (max_val / 5)

        color = get_green_color(i)

        rect = patches.Rectangle((start_pos, scale_y_position), section_width, scale_height,
                                 linewidth=1, edgecolor='black', facecolor=color, alpha=0.6, zorder=2)
        ax.add_patch(rect)

        if i == 0 or i == ((max_val)):
            ax.text(start_pos + (section_width / 2), scale_y_position - (0.5*scale_height),
                    f'{i}%',
                    color='white',
                    fontsize=14,
                    ha='center', va='top',
                    zorder=3)

    ax.set_ylim(bottom=-10)

    plt.rcParams['axes.facecolor'] = '#2B2B2B'
    plt.rcParams['figure.facecolor'] = '#2B2B2B'
    plt.rcParams['savefig.facecolor'] = '#2B2B2B'

    plt.title('DEFENSIVE ACTIONS', color='gold', fontsize=20, fontname='Roboto', loc='left')

    return fig

def create_set_piece_first_contacts_plot(def_set_piece_chart):
    """
    Create a set piece first contacts plot.

    Parameters:
    - def_set_piece_chart (DataFrame): A DataFrame containing the set piece data.

    Returns:
    - fig (Figure): The generated matplotlib figure.
    """

    pitch = VerticalPitch(pitch_color='#2B2B2B', line_color='white', goal_type='box', pitch_type='uefa', linewidth=1, half=True)
    fig, ax = pitch.draw(figsize=(8, 12))

    # Define the coordinates for the rectangles
    rect_coords = [
        [(13.84, 105), (13.84, 88.5), (30.09, 88.5), (30.09, 105)],  # Rect1
        [(30.59, 105), (30.59, 88.5), (37.41, 88.5), (37.41, 105)],  # Rect2
        [(37.91, 105), (37.91, 88.5), (54.16, 88.5), (54.16, 105)]  # Rect3
    ]

    # Example percentages for each rectangle (use your actual values here)
    percentages = [def_set_piece_chart.loc['Near post'].values[0], def_set_piece_chart.loc['Central'].values[0], 
                   def_set_piece_chart.loc['Far post'].values[0]]

    # Function to calculate green color based on percentage
    def get_green_color(percentage, max_percentage=100):  # Assuming 100 is the max percentage
        green_intensity = int((percentage / max_percentage) * 255)
        return f'#00{green_intensity:02x}00'

    # Create the rectangular patches
    for i, coords in enumerate(rect_coords):
        polygon = patches.Polygon(coords, closed=True, color="#00b200", zorder=2,  alpha=0.75)
        ax.add_patch(polygon)

        # Add text label in the center of each rectangle
        rect_center_x = (coords[0][0] + coords[2][0]) / 2 
        rect_center_y = (coords[0][1] + coords[2][1]) / 2
        ax.text(rect_center_x, rect_center_y, f'{int(percentages[i])}%',fontproperties='Roboto',
                va='center', ha='center', color='white', fontsize=18, zorder=3)

    # Set figure and axis background color
    fig.patch.set_facecolor('#2B2B2B')
    ax.patch.set_facecolor('#2B2B2B')

    arrow_start = (0, 107)  # Adjust these values as needed for your plot
    arrow_end = (18.84, 107)    # Adjust these values as needed for your plot

    # Draw the arrow
    ax.add_patch(patches.FancyArrow(
        arrow_start[0], arrow_start[1],  # x, y start point
        arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1],  # dx, dy length
        width=0.3,  # Width of the full arrow tail
        length_includes_head=False,  # The head is included in the calculation of the arrow's length
        head_width=1,  # Width of the arrow head
        head_length=1.5,  # Length of the arrow head
        color='lightgrey'  # Light grey color
    ))

    plt.rcParams['axes.facecolor'] = '#2B2B2B'
    plt.rcParams['figure.facecolor'] = '#2B2B2B'
    plt.rcParams['savefig.facecolor'] = '#2B2B2B'

    plt.title('SET PIECE FIRST CONTACTS - OWN BOX', color='gold', fontsize=20, fontname='Roboto', loc='left')

    return fig

def plot_shot_data(df_shots_last_5_matches, total_xg, Goals, Attempts, with_feet, with_head, direct_set_pieces):

    df_shots_last_5_matches['hover_text'] = (
        '<span style="font-size: 20px; line-height: 30px;"><b>' + df_shots_last_5_matches['PLAYER_FBREF_NAME'].astype(str) + '</b></span>' +
        '<span style="font-size: 12px; line-height: 22px;">  vs ' + df_shots_last_5_matches['OPPO_TEAM_NAME'].astype(str) + '</span>'
        '<br><span style="font-size: 15px; line-height: 18px;">Shot - ' + df_shots_last_5_matches['OUTCOME'].astype(str) + '</span>' +
        '<br><span style="font-size: 15px; line-height: 18px;">Shot xG - ' + df_shots_last_5_matches['XG'].astype(str) + '</span>' +
        '<br><span style="font-size: 15px; line-height: 18px;">Game time - ' + ((df_shots_last_5_matches['NEW_TIME_SECONDS']//60 + 1).astype(int)).astype(str) + "'"+ 
        '<span style="font-size: 12px; line-height: 22px;">  Period/Half: ' + df_shots_last_5_matches['PERIOD_ID'].astype(str) + '</span>'
    )

    outcome_colors = {'Goal': 'forestgreen', 'Saved': 'red', 'Blocked': 'orange', 'Off Target': 'red', 
                    'Woodwork': 'purple', 'Saved off Target': 'red'}
    outcome_markers = {'Goal': 'o', 'Saved': 'o', 'Blocked': 'X', 'Off Target': 'X',  'Woodwork': 'h',
                    'Saved off Target': 'X'}
    outcome_alpha = {'Goal': .8, 'Saved': 0.6, 'Blocked': 0.6, 'Off Target': 0.5,  'Woodwork': 0.7,
                    'Saved off Target': 0.5}

    # Create a Plotly figure
    fig = go.Figure()

    # Add the pitch as a background image
    fig.add_layout_image(
        dict(
            source="https://i.imgur.com/L0GXGh5.png",  # Path to your pitch background image
            xref="x",
            yref="y",
            x=0.,
            y=110,
            sizex=60,
            sizey=95,
            # sizing="stretch",
            opacity=1.0,
            layer="below")
    )

    # Define marker symbols in Plotly's format
    plotly_symbols = {
        'Goal': 'circle',
        'Saved': 'circle',
        'Blocked': 'x',
        'Off Target': 'x',
        'Woodwork': 'hexagon',
        'Saved off Target': 'x'
    }

    # header_values = []  # No header values, since we're incorporating the title directly in the cell
    header_values = [
        ['<span style="font-size: 10px; color: grey;">EXPECTED GOALS</span>', 
        '<span style="font-size: 10px; color: grey;">GOALS</span>',
        '<span style="font-size: 10px; color: grey;">ATTEMPTS</span>',
        '<span style="font-size: 10px; color: grey;">WITH FEET</span>', 
        '<span style="font-size: 10px; color: grey;">WITH HEAD</span>',
        '<span style="font-size: 10px; color: grey;">DIRECT SET PIECES</span>',
        # ... add other outcomes as needed
        '<span style="font-size: 22px; color: white;"><b>total_xg</b></span>',
        '<span style="font-size: 22px; color: white;"><b>Goals</b></span>',
        '<span style="font-size: 22px; color: white;"><b>Attempts</b></span>',
        '<span style="font-size: 22px; color: white;"><b>with_feet</b></span>',
        '<span style="font-size: 22px; color: white;"><b>with_head</b></span>',
        '<span style="font-size: 22px; color: white;"><b>direct_set_pieces</b></span>',
        ]
    ]

    # Add scatter plots for each outcome type
    for outcome, df_group in df_shots_last_5_matches.groupby('OUTCOME'):
        fig.add_trace(go.Scatter(
            x=(df_group['START_Y'] - 68)*-0.8825,  # Assuming START_Y is the horizontal axis in your pitch image
            y=df_group['START_X']+0.25,  # Assuming START_X is the vertical axis in your pitch image
            mode='markers',
            marker=dict(
                color=outcome_colors[outcome],
                symbol=plotly_symbols[outcome],
                size=df_group['xG_size']*5,  # Adjust size scaling factor as needed
                opacity=df_group['OUTCOME'].map(outcome_alpha)
            ),
            name=outcome,
            text=df_group['hover_text'],  # This will be displayed on hover
            hoverinfo='text',
            # hovertemplate='<b>%{text}</b>'  # Custom hover template for cleaner text display
        ))

    # Set axes to match the background image
    fig.update_xaxes(showgrid=False, zeroline=False, visible=False, range=[0, 68])
    fig.update_yaxes(showgrid=False, zeroline=False, visible=False, range=[52.5, 110])

    # Remove the plot background color
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(
                                family="Calibri, sans-serif",  # Set the font to Calibri, with sans-serif as a fallback
                                size=12,  # You can adjust the base size as needed
                                color="black"  # And also set a global font color if you wish
                            ), width=1150,height=700)

    fig.add_trace(go.Table(
        header=dict(
            values=[
                ["" ,"<b>TOTAL xG</b>", "", "<b>GOALS</b>", "", "<b>ATTEMPTS</b>", "", "<b>W/ FEET</b>", "", "<b>W/ HEAD</b>", "","<b>DIRECT FKs</b>"],
                ["", f"{round(total_xg, 1)}", "", f"{Goals}", "", f"{Attempts}", "", f"{with_feet}", "", f"{with_head}", "", f"{direct_set_pieces}"]
            ],
            line_color='#2B2B2B',
            fill_color='#2B2B2B',
            align=['right', 'left'],  # Align each column differently if needed
            font=dict(color=['grey', 'white'], size=[18, 28])  # Specify different styles for each column
        ),
        domain=dict(x=[0.835, 1], y=[0, 1])
    ))

    fig.update_layout(
        paper_bgcolor="#2B2B2B",
        plot_bgcolor="#2B2B2B",
        margin=dict(l=0, r=22, t=0, b=0),
        legend=dict(
            orientation="h",  # Set the legend orientation to horizontal
            yanchor="bottom",
            y=-0.03,  # Negative value to move the legend below the plot
            xanchor="center",
            x=0.45,
            font=dict(  # Update the font size
                size=25,  # Example size, adjust as needed
                color="white"  # Set the text color to white for better contrast on a dark background
            ),
            bgcolor="#2B2B2B",  # Set the background color of the legend
            itemsizing='constant'  # Use the same size for all legend markers
        ),
        title={
            'text': 'SHOT MAP',
            'y':0.97,  # Sets the y position of the title (1 is the top of the figure)
            'x':0.1,  # Centers the title horizontally (0.5 is the center of the figure)
            'xanchor': 'center',  # Ensures the title is centered at the x position
            'yanchor': 'top',  # Ensures the title is at the top of the y position
            'font': dict(
                family="Roboto",  # Specify the font family
                size=23,                     # Specify the font size
                color="gold"                # Specify the font color
            )
        }
    )
    

    return fig

def xT_generator(df, bins, pos_xT_only=False):
    xt = np.array(pd.read_csv("xTere.csv"))
    pitch = Pitch(line_zorder=2, pitch_type='uefa',axis=True, label=True)

    A = df.copy()
    A['DX'] = A.END_X - A.START_X
    A['DY'] = A.END_Y - A.START_Y
    A['move'] = True
    A.rename(columns={'START_X': 'x', 'START_Y': 'y'}, inplace=True)
    event = A.copy()
    move = event[event['move']].copy()
    bin_start_locations = pitch.bin_statistic(move['x'], move['y'], bins=bins)
    move = move[bin_start_locations['inside']].copy()
    bin_end_locations = pitch.bin_statistic(move['END_X'], move['END_Y'], bins=bins)
    move_success = move[(bin_end_locations['inside']) & (move['RESULT_ID'] == 1)].copy()

    grid_start = pitch.bin_statistic(move_success.x, move_success.y, bins=bins)
    grid_end = pitch.bin_statistic(move_success.END_X, move_success.END_Y, bins=bins)
    start_xt = xt[grid_start['binnumber'][1], grid_start['binnumber'][0]]
    end_xt = xt[grid_end['binnumber'][1], grid_end['binnumber'][0]]
    added_xt = end_xt - start_xt
    move_success['xt'] = added_xt

    if pos_xT_only is True:
        move_success = move_success[move_success['xt'] > 0]
    
    return move_success

def plot_match_momentum(df_xt, df_match_oi, df_goals):
    team1 = df_match_oi['HOME_TEAM_NAME'].iloc[0]
    team1_id = df_match_oi['HOME_TEAM_ID'].iloc[0]
    team2 = df_match_oi['AWAY_TEAM_NAME'].iloc[0]
    team2_id = df_match_oi['AWAY_TEAM_ID'].iloc[0]
    datetime_obj = datetime.strptime(df_match_oi['DATE_TIME'].iloc[0], '%Y-%m-%d %H:%M:%S')
    datetime_obj = datetime_obj.strftime('%Y-%m-%d')

    df_xt_grouped = df_xt.groupby(['PERIOD_ID', 'NEW_TIME_SECONDS', 'TEAM_NAME'])['xt'].sum().reset_index()
    pivot_df = df_xt_grouped.pivot(index=['NEW_TIME_SECONDS', 'PERIOD_ID'], columns='TEAM_NAME', values='xt').fillna(0)
    pivot_df['difference'] = pivot_df[df_match_oi['HOME_TEAM_NAME'].iloc[0]] - pivot_df[df_match_oi['AWAY_TEAM_NAME'].iloc[0]]
    pivot_df.reset_index(inplace=True)

    bins = np.linspace(pivot_df['NEW_TIME_SECONDS'].min(), pivot_df['NEW_TIME_SECONDS'].max(), 10)  # 9 intervals = 10 bin edges
    labels = [f'Bin {i+1}' for i in range(len(bins)-1)]
    pivot_df['bins'] = pd.cut(pivot_df['NEW_TIME_SECONDS'], bins=bins, labels=labels, include_lowest=True)

    bin_means = pivot_df.groupby(['PERIOD_ID', 'bins'])['difference'].sum().reset_index()
    bin_means['period_bin'] = bin_means['PERIOD_ID'].astype(str) + '-' + bin_means['bins'].astype(str)

    df_goals['bins'] = pd.cut(df_goals['NEW_TIME_SECONDS'], bins=bins, labels=labels, include_lowest=True)
    goals_binned = df_goals.groupby(['PERIOD_ID', 'bins', 'TEAM_FBREF_ID']).size().reset_index(name='goals')

    fig, ax = plt.subplots(figsize=(14, 7))

    # Change the background color
    fig.patch.set_facecolor('#2B2B2B')
    ax.set_facecolor('#2B2B2B')

    # Remove the border around the plot
    for spine in ax.spines.values():
        spine.set_edgecolor('#2B2B2B')
        # spine.set_visible(True)
        # spine.set_linewidth(0)

    # Plotting bars with custom colors for positive and negative values
    bars = ax.bar(bin_means['period_bin'], bin_means['difference'], color=bin_means['difference'].apply(lambda x: 'white' if x > 0 else 'yellow'))

    # Set y-axis limits
    y_lim = max((bin_means['difference'].abs().max())*1.75, 0.75)
    ax.set_ylim([-y_lim, y_lim])

    ax.grid(False)

    # Add horizontal line at y=0
    ax.axhline(0, color='grey', linewidth=1.25)

    # Customize the title, labels, and ticks
    ax.set_title('MATCH MOMENTUM', color='gold', fontsize=24, fontname='Roboto', loc='left')

    fig.text(0.12, 0.855, f'{team1} vs {team2}, {datetime_obj}', color='white', fontsize=16, fontname='Roboto')
    # ax.set_xlabel('Bins', color='white')
    ax.set_ylabel('DANGER OF POSSESSION', color='white', labelpad=-20, size=18)
    ax.tick_params(colors='#2B2B2B', which='both', labelcolor='#2B2B2B')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Add segregated x-axis labels for "1st Half" and "2nd Half"
    halfway_point = len(bin_means['period_bin']) // 2
    ax.text(halfway_point / 2.5, -y_lim, '1st Half', horizontalalignment='center', color='white', fontsize=20)
    ax.text(halfway_point + halfway_point / 2, -y_lim, '2nd Half', horizontalalignment='center', color='white', fontsize=20)
    ax.axvline(x=halfway_point - 0.5, color='gray', linestyle='-')

    # Add legend with circles
    colors = {team1: 'white', team2: 'yellow'}
    labels = list(colors.keys())
    handles = [plt.Line2D([0], [0], marker='o', color='none', markerfacecolor=colors[label], markersize=14) for label in labels]
    plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, facecolor='#2B2B2B', edgecolor='none', 
               framealpha=0, fontsize=20, labelcolor='white')

    # Plot goal markers
    team_colors = {team1_id: 'white', team2_id: 'yellow'}
    for i, row in goals_binned[goals_binned['goals'] > 0].iterrows():
        period_bin = str(row['PERIOD_ID']) + '-' + str(row['bins'])
        bar = bin_means[bin_means['period_bin'] == period_bin].index
        if len(bar) > 0:
            x_pos = bar[0]
            y_pos = bin_means.loc[x_pos, 'difference']
            for goal in range(row['goals']):
                if row['TEAM_FBREF_ID'] == team1_id:
                    if y_pos > 0:
                        ax.plot(x_pos, y_pos + 0.1 + goal * 0.1, marker='o', color=team_colors[row['TEAM_FBREF_ID']], markersize=8)
                    else:
                        ax.plot(x_pos, 0.075, marker='o', color=team_colors[row['TEAM_FBREF_ID']], markersize=8)
                else:
                    if y_pos > 0:
                        ax.plot(x_pos, -0.075, marker='o', color=team_colors[row['TEAM_FBREF_ID']], markersize=8)
                    else:
                        ax.plot(x_pos, y_pos - 0.1 - goal * 0.1, marker='o', color=team_colors[row['TEAM_FBREF_ID']], markersize=8)
    plt.tight_layout()
    return fig

def rest_dict_pass_map(teamIds, df_events, team_names, df_cards, df_subs):
    res_dict = {}
    var = 'PLAYER_FBREF_NAME'
    var2 = 'passRecipientName'
    for teamId in teamIds:
        mask = df_events['TEAM_FBREF_ID']== teamId
        team_ws_id = team_names[team_names['TEAM_FBREF_ID'] == teamId].TEAM_WS_ID.iloc[0]
        df_cards_oi = df_cards[df_cards['TEAM_WS_ID'] == team_ws_id]
        df_subs_oi = df_subs[df_subs['TEAM_WS_ID'] == team_ws_id]
        df_ = df_events[mask]

        teamName = df_['TEAM_NAME'].unique()[0]

        mask1 = df_cards_oi['CARD_TYPE'].apply(lambda x: x in ["SecondYellow", "Red"])

        if len(mask1) > 0:
            first_red_card_time = df_cards_oi[mask1].NEW_TIME_SECONDS.min()
        else:
            first_red_card_time = np.nan

        if len(df_subs_oi) > 0:
            first_sub_time = df_subs_oi.NEW_TIME_SECONDS.min()
        else:
            first_sub_time = np.nan

        max_minute = (df_.NEW_TIME_SECONDS.max())

        all_times = [first_red_card_time, first_sub_time, max_minute]
        filtered_times = [time for time in all_times if not np.isnan(time)]

        min_time = min(filtered_times)

        df_ = df_.sort_values(['PERIOD_ID','NEW_TIME_SECONDS', 'ORIGINAL_EVENT_ID'])

        passes_df = df_.reset_index().drop('index', axis=1)
        passes_df['playerId'] = passes_df['PLAYER_WS_ID'].astype('Int64')
        passes_df = passes_df[passes_df['playerId'].notnull()]
        passes_df['passRecipientName'] = passes_df['PLAYER_FBREF_NAME'].shift(-1)
        passes_df = passes_df[passes_df['passRecipientName'].notnull()]
        mask1 = passes_df['TYPE_NAME'].apply(lambda x: x in ['pass'])
        passes_df_all = passes_df[mask1]

        mask2 = passes_df_all['NEW_TIME_SECONDS'] < min_time
        players = passes_df_all[passes_df_all['NEW_TIME_SECONDS'] < min_time]['PLAYER_FBREF_NAME'].unique()
        mask3 = passes_df_all['PLAYER_FBREF_NAME'].apply(lambda x: x in players)
        passes_df_short = passes_df_all[mask2 & mask3]

        mask2 = passes_df_all['PLAYER_FBREF_NAME'] != passes_df_all['passRecipientName']
        mask3 = passes_df_all['RESULT_ID'] == 1
        passes_df_suc = passes_df_all[mask2&mask3]

        mask2 = passes_df_suc['NEW_TIME_SECONDS'] < min_time
        players = passes_df_suc[passes_df_suc['NEW_TIME_SECONDS'] < min_time]['PLAYER_FBREF_NAME'].unique()
        mask3 = passes_df_suc['PLAYER_FBREF_NAME'].apply(lambda x: x in players) & \
                passes_df_suc['passRecipientName'].apply(lambda x: x in players)
        passes_df_suc_short = passes_df_suc[mask2 & mask3] 

        res_dict[teamId] = {}
        res_dict[teamId]['passes_df_all'] = passes_df_all
        res_dict[teamId]['passes_df_short'] = passes_df_short
        res_dict[teamId]['passes_df_suc'] = passes_df_suc
        res_dict[teamId]['passes_df_suc_short'] = passes_df_suc_short
        res_dict[teamId]['min_time'] = min_time

        passes_df_all = res_dict[teamId]['passes_df_all']
        passes_df_suc = res_dict[teamId]['passes_df_suc']
        passes_df_short = res_dict[teamId]['passes_df_short']
        passes_df_suc_short = res_dict[teamId]['passes_df_suc_short']
        
        player_position = passes_df_short.groupby(var).agg({'START_X': ['median'], 'START_Y': ['median']})

        player_position.columns = ['START_X', 'START_Y']
        player_position.index.name = 'PLAYER_FBREF_NAME'
        player_position.index = player_position.index.astype(str)

        player_pass_count_all = passes_df_all.groupby(var).agg({'PLAYER_WS_ID':'count'}).rename(columns={'PLAYER_WS_ID':'num_passes_all'})
        player_pass_count_suc = passes_df_suc.groupby(var).agg({'PLAYER_WS_ID':'count'}).rename(columns={'PLAYER_WS_ID':'num_passes'})
        player_pass_count_suc_short = passes_df_suc_short.groupby(var).agg({'PLAYER_WS_ID':'count'}).rename(columns={'PLAYER_WS_ID':'num_passes2'})
        player_pass_count = player_pass_count_all.join(player_pass_count_suc).join(player_pass_count_suc_short)
        
            
        passes_df_all["pair_key"] = passes_df_all.apply(lambda x: "_".join([str(x[var]), str(x[var2])]), axis=1)
        passes_df_suc["pair_key"] = passes_df_suc.apply(lambda x: "_".join([str(x[var]), str(x[var2])]), axis=1)
        passes_df_suc_short["pair_key"] = passes_df_suc_short.apply(lambda x: "_".join([str(x[var]), str(x[var2])]), axis=1)

        
        pair_pass_count_all = passes_df_all.groupby('pair_key').agg({'PLAYER_WS_ID':'count'}).rename(columns={'PLAYER_WS_ID':'num_passes_all'})
        pair_pass_count_suc = passes_df_suc.groupby('pair_key').agg({'PLAYER_WS_ID':'count'}).rename(columns={'PLAYER_WS_ID':'num_passes'})
        pair_pass_count_suc_short = passes_df_suc_short.groupby('pair_key').agg({'PLAYER_WS_ID':'count'}).rename(columns={'PLAYER_WS_ID':'num_passes2'})
        pair_pass_count = pair_pass_count_all.join(pair_pass_count_suc).join(pair_pass_count_suc_short)

        player_position['z'] = player_position['START_X']
        player_position['START_X'] = player_position['START_Y']
        player_position['START_Y'] = player_position['z']
        
        res_dict[teamId]['player_position'] = player_position
        res_dict[teamId]['player_pass_count'] = player_pass_count
        res_dict[teamId]['pair_pass_count'] = pair_pass_count
    
    return res_dict
   

def plot_single_passmap(df_player_match, res_dict, df_match_oi, df_events, teamid_selected):
    datetime_obj = datetime.strptime(df_match_oi['DATE_TIME'].iloc[0], '%Y-%m-%d %H:%M:%S')
    datetime_obj = datetime_obj.strftime('%Y-%m-%d')
    team1, team2 = res_dict.keys()
    colors = [
        (60/255, 107/255, 137/255),   # RGB(60,85,106)
        (60/255, 143/255, 136/255),  # RGB(60,112,107)
        (130/255, 201/255, 133/255), # RGB(102,151,104)
        (80/255, 226/255, 80/255)    # RGB(80,226,80)
    ]

    cmap = mpl.colors.LinearSegmentedColormap.from_list("custom_cmap", colors)

    def change_range(value, old_range, new_range):
        new_value = ((value-old_range[0]) / (old_range[1]-old_range[0])) * (new_range[1]-new_range[0]) + new_range[0]
        
        if new_value >= new_range[1]:
            return new_range[1]
        elif new_value <= new_range[0]:
            return new_range[0]
        else:
            return new_value
    
    #nodes
    min_node_size = 8
    max_node_size = 20

    max_player_count = 88
    min_player_count = 1

    head_length = 0.2
    head_width = 0.15

    min_passes = 5

    player_numbers = df_player_match[['PLAYER_WS_ID', 'PLAYER_FBREF_NAME', 'TEAM_FBREF_ID', 'JERSEY_NUMBER']]

    max_overall = 0
    min_overall = 10000
    for val in res_dict.keys():
        team_max = res_dict[val]['pair_pass_count']['num_passes_all'].max()
        team_min = res_dict[val]['pair_pass_count']['num_passes_all'].min()
        max_overall = max(max_overall, team_max)
        min_overall = min(min_overall, team_min)
    
    plt.style.use('fivethirtyeight')

    fig, (ax, ax_legend) = plt.subplots(1, 2, figsize=(4, 3.5), dpi=350, gridspec_kw={'width_ratios': [2, 1]})

    fig.patch.set_facecolor('#2B2B2B')
    ax.set_facecolor('#2B2B2B')

    for spine in ax.spines.values():
        spine.set_edgecolor('#2B2B2B')

    ax.set_title('PASS MAP', color='gold', fontsize=8, fontname='Roboto', loc='left')
    fig.text(0.12, 0.88, f"{df_match_oi['HOME_TEAM_NAME'].iloc[0]} vs {df_match_oi['AWAY_TEAM_NAME'].iloc[0]}, {datetime_obj}", 
             color='white', fontsize=5, fontname='Roboto')

    #define dataframes
    position = res_dict[teamid_selected]['player_position']
    player_pass_count = res_dict[teamid_selected]['player_pass_count']
    pair_pass_count = res_dict[teamid_selected]['pair_pass_count']
    minutes_ = res_dict[teamid_selected]['min_time']

    pitch = VerticalPitch(pitch_type='uefa', pitch_color='#2B2B2B', line_color='white', 
                            goal_type='box', linewidth=1,
                        pad_bottom=10)

    #plot vertical pitches
    pitch.draw(ax=ax, constrained_layout=False, tight_layout=False)

    pair_stats = pair_pass_count.sort_values('num_passes',ascending=False)
    pair_stats2 = pair_stats[pair_stats['num_passes'] >= min_passes]

    mask = df_events['NEW_TIME_SECONDS'] < minutes_
    players_ = list(set(df_events[mask]['PLAYER_FBREF_NAME'].dropna()))

    mask_ = player_pass_count.index.map(lambda x: x in players_)
    player_pass_count = player_pass_count.loc[mask_]

    mask_ = pair_stats2.index.map(lambda x: (x.split('_')[0] in players_) &  (x.split('_')[1] in players_))
    pair_stats2 = pair_stats2[mask_]

    ind = position.index.map(lambda x: x in players_)
    position = position.loc[ind]

    team_numbers = player_numbers[player_numbers['TEAM_FBREF_ID'] == teamid_selected]
    team_numbers.set_index('PLAYER_FBREF_NAME', inplace=True)

    # Step 3: plotting nodes
    # print(player_pass_count)
    for var, row in player_pass_count.iterrows():
        player_x = position.loc[var]["START_X"]
        player_y = position.loc[var]["START_Y"]

        num_passes = row["num_passes"]

        marker_size = change_range(num_passes, (min_player_count, max_player_count), (min_node_size, max_node_size)) 

        ax.plot(player_x, player_y, '.', color="white", markersize=marker_size, zorder=5)
        ax.plot(player_x, player_y, '.', markersize=marker_size+3, zorder=4, color='black')

        player_pass_count.loc[var, 'marker_size'] = marker_size

        # Add jersey numbers
        jersey_number = team_numbers.loc[var, 'JERSEY_NUMBER']
        ax.text(player_x, player_y, str(jersey_number), color='black', ha='center', va='center', 
                fontsize=4, fontweight='bold', zorder=6)


    # Step 4: plotting edges  
    for pair_key, row in pair_stats2.iterrows():
        player1, player2 = pair_key.split("_")

        player1_x = position.loc[player1]["START_X"]
        player1_y = position.loc[player1]["START_Y"]

        player2_x = position.loc[player2]["START_X"]
        player2_y = position.loc[player2]["START_Y"]

        num_passes = row["num_passes"]
        # pass_value = row["pass_value"]

        line_width = 2.5
        alpha = change_range(num_passes, (min_overall, max_overall), (0.4, 1))

        norm_color = Normalize(vmin=min_overall, vmax=max_overall)
        norm_size = change_range(num_passes, (min_overall, max_overall), (0.1, 1))
        # print(norm(num_passes))
        # edge_cmap = cm.get_cmap(nodes_cmap)
        # edge_color = "#00FF00"

        x = player1_x
        y = player1_y
        dx = player2_x-player1_x
        dy = player2_y-player1_y
        rel = 80/105
        shift_x = 1.5
        shift_y = shift_x*rel

        slope = round(abs((player2_y - player1_y)*105/100 / (player2_x - player1_x)*68/100),1)

        color_ = cmap(norm_color(num_passes)) 
        # print(color_) 

        mutation_scale = 1
        if (slope > 0.5):
            if dy > 0:
                ax.annotate("", xy=(x+dx+shift_x, y+dy), xytext=(x+shift_x, y),zorder=2,
                        arrowprops=dict(arrowstyle=f'-|>, head_length = {head_length*alpha}, \
                                        head_width={head_width*alpha}',
                                        alpha = alpha,
                                        color=color_,
                                        # fc = 'blue',
                                        lw=line_width * norm_size,
                                        shrinkA=7,
                                        shrinkB=7))
                
                
            elif dy <= 0:
                ax.annotate("", xy=(x+dx-shift_x, y+dy), xytext=(x-shift_x, y),zorder=2,
                        arrowprops=dict(arrowstyle=f'-|>, head_length = {head_length*alpha}, \
                                        head_width={head_width*alpha}',
                                        alpha = alpha,
                                        color=color_,
                                        # fc = 'blue',
                                        lw=line_width * norm_size,
                                        shrinkA=7,
                                        shrinkB=7))
                
        elif (slope <= 0.5) & (slope >=0):
            if dx > 0:
                ax.annotate( "", xy=(x+dx, y+dy-shift_y), xytext=(x, y-shift_y),zorder=2,
                        arrowprops=dict(arrowstyle=f'-|>, head_length = {head_length*alpha}, \
                                        head_width={head_width*alpha}',
                                        alpha = alpha,
                                        color=color_,
                                        # fc = 'blue',
                                        lw=line_width * norm_size,
                                        shrinkA=7,
                                        shrinkB=7))

            elif dx <= 0:

                ax.annotate("", xy=(x+dx, y+dy+shift_y), xytext=(x, y+shift_y),zorder=2,
                        arrowprops=dict(arrowstyle=f'-|>, head_length = {head_length*alpha}, \
                                        head_width={head_width*alpha}',
                                        alpha = alpha,
                                        color=color_,
                                        # fc = 'blue',
                                        lw=line_width * norm_size,
                                        shrinkA=7,
                                        shrinkB=7))

        else:
            print(1)
    fig.set_facecolor('#2B2B2B')
    ax.patch.set_facecolor('#2B2B2B')
    ax_legend.set_facecolor('#2B2B2B')
    ax_legend.axis('off')
    for idx, player in enumerate(list(position.sort_values(by='START_Y', ascending=True).index)):
        jersey_number = team_numbers.loc[player, 'JERSEY_NUMBER']
        ax_legend.plot(0.07, 0.9 - idx * 0.054, '.', color="white", markersize=15, zorder=5, transform=ax_legend.transAxes)
        ax_legend.plot(0.07, 0.9 - idx * 0.054, '.', markersize=17.5, zorder=4, color='black', transform=ax_legend.transAxes)
        ax_legend.text(0.072, 0.892 - idx * 0.054, str(jersey_number), color='black', fontsize=5, zorder=6,
                    fontweight='bold', ha='center', transform=ax_legend.transAxes)
        ax_legend.text(0.17, 0.892 - idx * 0.054, player, color='white', fontsize=6, ha='left', transform=ax_legend.transAxes)

    # Add the legend for pass frequency
    legend_elements = [
        mpl.lines.Line2D([0], [0], color=(60/255, 107/255, 137/255), lw=3, label='Rare'),
        mpl.lines.Line2D([0], [0], color=(60/255, 143/255, 136/255), lw=3, label='Fairly Frequent'),
        mpl.lines.Line2D([0], [0], color=(130/255, 201/255, 133/255), lw=3, label='Frequent'),
        mpl.lines.Line2D([0], [0], color=(80/255, 226/255, 80/255), lw=3, label='Very Frequent')
    ]

    fig.legend(handles=legend_elements, loc='lower center', ncol=4, facecolor='#2B2B2B', edgecolor='none', 
            framealpha=0, fontsize=6, labelcolor='white')

    plt.rcParams['axes.facecolor'] = '#2B2B2B'
    plt.rcParams['figure.facecolor'] = '#2B2B2B'
    plt.rcParams['savefig.facecolor'] = '#2B2B2B'

    return fig


def plot_double_passmap(df_player_match, res_dict, df_match_oi, df_events, teamIds):
    datetime_obj = datetime.strptime(df_match_oi['DATE_TIME'].iloc[0], '%Y-%m-%d %H:%M:%S')
    datetime_obj = datetime_obj.strftime('%Y-%m-%d')
    team1, team2 = res_dict.keys()
    colors = [
        (60/255, 107/255, 137/255),   # RGB(60,85,106)
        (60/255, 143/255, 136/255),  # RGB(60,112,107)
        (130/255, 201/255, 133/255), # RGB(102,151,104)
        (80/255, 226/255, 80/255)    # RGB(80,226,80)
    ]

    cmap = mpl.colors.LinearSegmentedColormap.from_list("custom_cmap", colors)

    def change_range(value, old_range, new_range):
        new_value = ((value-old_range[0]) / (old_range[1]-old_range[0])) * (new_range[1]-new_range[0]) + new_range[0]
        
        if new_value >= new_range[1]:
            return new_range[1]
        elif new_value <= new_range[0]:
            return new_range[0]
        else:
            return new_value
    
    #nodes
    min_node_size = 8
    max_node_size = 20

    max_player_count = 88
    min_player_count = 1

    head_length = 0.2
    head_width = 0.15

    min_passes = 5

    player_numbers = df_player_match[['PLAYER_WS_ID', 'PLAYER_FBREF_NAME', 'TEAM_FBREF_ID', 'JERSEY_NUMBER']]

    max_overall = 0
    min_overall = 10000
    for val in res_dict.keys():
        team_max = res_dict[val]['pair_pass_count']['num_passes_all'].max()
        team_min = res_dict[val]['pair_pass_count']['num_passes_all'].min()
        max_overall = max(max_overall, team_max)
        min_overall = min(min_overall, team_min)

    plt.style.use('fivethirtyeight')

    fig,ax = plt.subplots(1,2,figsize=(5,4), dpi=400)

    fig.patch.set_facecolor('#2B2B2B')

    teamId_home = teamIds[0]
    teamId_away = teamIds[1]
    team_colors = ["white", "yellow"]

    ax[0].set_title('PASS MAP COMPARISON', color='gold', fontsize=8, fontname='Roboto', loc='left')
    fig.text(0.082, 0.865, f"{df_match_oi['HOME_TEAM_NAME'].iloc[0]} vs {df_match_oi['AWAY_TEAM_NAME'].iloc[0]}, {datetime_obj}", 
             color='grey', fontsize=5, fontname='Roboto')

    for i, teamid in enumerate([teamId_home, teamId_away]):    
        #define dataframes
        position = res_dict[teamid]['player_position']
        player_pass_count = res_dict[teamid]['player_pass_count']
        pair_pass_count = res_dict[teamid]['pair_pass_count']
        minutes_ = res_dict[teamid]['min_time']

        pitch = VerticalPitch(pitch_type='uefa', pitch_color='#2B2B2B', line_color='white', 
                            goal_type='box', linewidth=1,
                            pad_bottom=10)
        
        #plot vertical pitches
        pitch.draw(ax=ax[i], constrained_layout=False, tight_layout=False)
        
        pair_stats = pair_pass_count.sort_values('num_passes',ascending=False)
        pair_stats2 = pair_stats[pair_stats['num_passes'] >= min_passes]

        mask = df_events['NEW_TIME_SECONDS'] < minutes_
        players_ = list(set(df_events[mask]['PLAYER_FBREF_NAME'].dropna()))

        mask_ = player_pass_count.index.map(lambda x: x in players_)
        player_pass_count = player_pass_count.loc[mask_]

        mask_ = pair_stats2.index.map(lambda x: (x.split('_')[0] in players_) &  (x.split('_')[1] in players_))
        pair_stats2 = pair_stats2[mask_]
        
        ind = position.index.map(lambda x: x in players_)
        position = position.loc[ind]

        team_numbers = player_numbers[player_numbers['TEAM_FBREF_ID'] == teamid]
        team_numbers.set_index('PLAYER_FBREF_NAME', inplace=True)
        
        # Step 3: plotting nodes
        # print(player_pass_count)
        for var, row in player_pass_count.iterrows():
            player_x = position.loc[var]["START_X"]
            player_y = position.loc[var]["START_Y"]

            num_passes = row["num_passes"]

            marker_size = change_range(num_passes, (min_player_count, max_player_count), (min_node_size, max_node_size)) 

            ax[i].plot(player_x, player_y, '.', color=team_colors[i], markersize=marker_size, zorder=5)
            ax[i].plot(player_x, player_y, '.', markersize=marker_size+3, zorder=4, color='black')

            player_pass_count.loc[var, 'marker_size'] = marker_size

            # Add jersey numbers
            jersey_number = team_numbers.loc[var, 'JERSEY_NUMBER']
            ax[i].text(player_x, player_y, str(jersey_number), color='black', ha='center', va='center', 
                       fontsize=4, fontweight='bold', zorder=6)


        # Step 4: ploting edges  
        for pair_key, row in pair_stats2.iterrows():
            player1, player2 = pair_key.split("_")

            player1_x = position.loc[player1]["START_X"]
            player1_y = position.loc[player1]["START_Y"]

            player2_x = position.loc[player2]["START_X"]
            player2_y = position.loc[player2]["START_Y"]

            num_passes = row["num_passes"]
            # pass_value = row["pass_value"]

            line_width = 2.5
            alpha = change_range(num_passes, (min_overall, max_overall), (0.6, 1))

            norm_color = Normalize(vmin=min_overall, vmax=max_overall)
            norm_size = change_range(num_passes, (min_overall, max_overall), (0.1, 1))
            # print(norm(num_passes))
            # edge_cmap = cm.get_cmap(nodes_cmap)
            # edge_color = "#00FF00"

            x = player1_x
            y = player1_y
            dx = player2_x-player1_x
            dy = player2_y-player1_y
            rel = 80/105
            shift_x = 1.5
            shift_y = shift_x*rel

            slope = round(abs((player2_y - player1_y)*105/100 / (player2_x - player1_x)*68/100),1)

            color_ = cmap(norm_color(num_passes)) 
            # print(color_) 

            if (slope > 0.5):
                if dy > 0:
                    ax[i].annotate("", xy=(x+dx+shift_x, y+dy), xytext=(x+shift_x, y),zorder=2,
                            arrowprops=dict(arrowstyle=f'-|>, head_length = {head_length*alpha}, \
                                            head_width={head_width*alpha}',
                                            alpha = alpha,
                                            color=color_,
                                            # fc = 'blue',
                                            lw=line_width * norm_size,
                                            shrinkA=7,
                                            shrinkB=7))
                    
                    
                elif dy <= 0:
                    ax[i].annotate("", xy=(x+dx-shift_x, y+dy), xytext=(x-shift_x, y),zorder=2,
                            arrowprops=dict(arrowstyle=f'-|>, head_length = {head_length*alpha}, \
                                            head_width={head_width*alpha}',
                                            alpha = alpha,
                                            color=color_,
                                            # fc = 'blue',
                                            lw=line_width * norm_size,
                                            shrinkA=7,
                                            shrinkB=7))
                    
            elif (slope <= 0.5) & (slope >=0):
                if dx > 0:
    #                 print(2)

                    ax[i].annotate( "", xy=(x+dx, y+dy-shift_y), xytext=(x, y-shift_y),zorder=2,
                            arrowprops=dict(arrowstyle=f'-|>, head_length = {head_length*alpha}, \
                                            head_width={head_width*alpha}',
                                            alpha = alpha,
                                            color=color_,
                                            # fc = 'blue',
                                            lw=line_width * norm_size,
                                            shrinkA=7,
                                            shrinkB=7))

                elif dx <= 0:

                    ax[i].annotate("", xy=(x+dx, y+dy+shift_y), xytext=(x, y+shift_y),zorder=2,
                            arrowprops=dict(arrowstyle=f'-|>, head_length = {head_length*alpha}, \
                                            head_width={head_width*alpha}',
                                            alpha = alpha,
                                            color=color_,
                                            # fc = 'blue',
                                            lw=line_width * norm_size,
                                            shrinkA=7,
                                            shrinkB=7))
            else:
                print(1)

    legend_elements = [
        mpl.lines.Line2D([0], [0], color=(60/255, 107/255, 137/255), lw=3, label='Rare'),
        mpl.lines.Line2D([0], [0], color=(60/255, 143/255, 136/255), lw=3, label='Fairly Frequent'),
        mpl.lines.Line2D([0], [0], color=(130/255, 201/255, 133/255), lw=3, label='Frequent'),
        mpl.lines.Line2D([0], [0], color=(80/255, 226/255, 80/255), lw=3, label='Very Frequent')
    ]

    fig.legend(handles=legend_elements, loc='lower center', ncol=4, facecolor='#2B2B2B', edgecolor='none', 
            framealpha=0, fontsize=6, labelcolor='white')

    plt.rcParams['axes.facecolor'] = '#2B2B2B'
    plt.rcParams['figure.facecolor'] = '#2B2B2B'
    plt.rcParams['savefig.facecolor'] = '#2B2B2B'

    return fig

def plot_xg_match_story(df_shots, df_goals, df_match_oi, xg_vline, max_mins):
    team1 = df_match_oi['HOME_TEAM_NAME'].iloc[0]
    team1_id = df_match_oi['HOME_TEAM_ID'].iloc[0]
    team2 = df_match_oi['AWAY_TEAM_NAME'].iloc[0]
    team2_id = df_match_oi['AWAY_TEAM_ID'].iloc[0]
    datetime_obj = datetime.strptime(df_match_oi['DATE_TIME'].iloc[0], '%Y-%m-%d %H:%M:%S')
    datetime_obj = datetime_obj.strftime('%Y-%m-%d')
    df_shots["Minutes"] = (df_shots['NEW_TIME_SECONDS'] // 60)+1
    df_shots['cumulative_xG'] = df_shots.groupby('TEAM_NAME')['XG'].cumsum()
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.grid(False)

    teams = [team1, team2]
    colors = ['white', 'yellow'] 
    xg_max = []

    for team, color in zip(teams, colors):
        team_shots = df_shots[df_shots['TEAM_NAME'] == team]
        new_row_top = {
        "MATCH_ID": None, "ORIGINAL_EVENT_ID": None, "PERIOD_ID": None, "TIME_SECONDS": None, "NEW_TIME_SECONDS": None,
        "TEAM_FBREF_ID": None, "PLAYER_WS_ID": None, "START_X": None, "END_X": None, "START_Y": None, "END_Y": None,
        "RESULT_ID": None, "ACTION_ID": None, "TYPE_NAME": None, "BODYPART_NAME": None, "PLAYER_FBREF_NAME": None, "TEAM_NAME": None,
        "XG": None, "OUTCOME": None, "Minutes": 0.0, "cumulative_xG": 0.0
        }
        new_row_bottom = {
        "MATCH_ID": None, "ORIGINAL_EVENT_ID": None, "PERIOD_ID": None, "TIME_SECONDS": None, "NEW_TIME_SECONDS": None,
        "TEAM_FBREF_ID": None, "PLAYER_WS_ID": None, "START_X": None, "END_X": None, "START_Y": None, "END_Y": None,
        "RESULT_ID": None, "ACTION_ID": None, "TYPE_NAME": None, "BODYPART_NAME": None, "PLAYER_FBREF_NAME": None, "TEAM_NAME": None,
        "XG": None, "OUTCOME": None, "Minutes": float(max_mins), "cumulative_xG": team_shots["cumulative_xG"].max()
        }
        team_shots = pd.concat([pd.DataFrame([new_row_top]), team_shots, pd.DataFrame([new_row_bottom])], ignore_index=True)

        xg_max.append(team_shots['XG'].sum())
        ax.step(team_shots['Minutes'], team_shots['cumulative_xG'], where='post', label=team, color=color)
        

    for i, row in df_shots.iterrows():
        if row['OUTCOME'] == 'Goal':
            ax.plot(row['Minutes'], row['cumulative_xG'], 'o', color='black', markersize=8, zorder=3)
            ax.plot(row['Minutes'], row['cumulative_xG'], 'o', color='white', markersize=13, zorder=2)
    
    ax.axvline(xg_vline, color='grey', linewidth=1.25, zorder= 0)

    ax.axhline(max(xg_max)/2, color='grey', linewidth=1.25, zorder= 0)

    ax.set_facecolor('#2B2B2B')
    fig.patch.set_facecolor('#2B2B2B')
    plt.subplots_adjust(top=0.82)
    ax.spines['top'].set_color('#2B2B2B')
    ax.spines['right'].set_color('#2B2B2B')
    ax.spines['left'].set_color('#2B2B2B')
    ax.spines['bottom'].set_color('#2B2B2B')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    fig.suptitle('XG MATCH STORY', color='gold', fontsize=14, fontname='Roboto', x=0.08,ha='left')
    fig.text(0.08, 0.92, f'{team1} vs {team2}, {datetime_obj}', color='grey', fontsize=10, fontname='Roboto')
    # ax.set_title('PASS MAP', color='gold', fontsize=14, fontname='Roboto', loc='left')

    goals_team1 = df_goals[df_goals['TEAM_NAME'] == team1]['TEAM_NAME'].count()
    goals_team2 = df_goals[df_goals['TEAM_NAME'] == team2]['TEAM_NAME'].count()
    xg_team1 = df_shots[df_shots['TEAM_NAME'] == team1]['XG'].sum()
    xg_team2 = df_shots[df_shots['TEAM_NAME'] == team2]['XG'].sum()
    # Add total goals and xG stats
    fig.text(0.1, 0.88, 'GOALS', ha='left', color='grey', fontsize=10, fontname='Roboto', fontweight='bold')
    fig.text(0.102, 0.84, f'{goals_team1} - {goals_team2}', ha='left', color='white', fontsize=14, fontweight='bold', fontname='Roboto')

    fig.text(0.2, 0.88, 'XG TOTAL', ha='left', color='grey', fontsize=10, fontname='Roboto', fontweight='bold')
    fig.text(0.2, 0.84, f'{xg_team1:.2f} - {xg_team2:.2f}', ha='left', color='white', fontsize=14, fontweight='bold', fontname='Roboto')


    ax.set_xlabel('Minutes')
    ax.set_ylabel('xG', rotation=0, labelpad=12)
    ax.legend(loc='upper left', facecolor='#2B2B2B', edgecolor='none', framealpha=0, fontsize=12, labelcolor='white')

    plt.rcParams['axes.facecolor'] = '#2B2B2B'
    plt.rcParams['figure.facecolor'] = '#2B2B2B'
    plt.rcParams['savefig.facecolor'] = '#2B2B2B'

    return fig