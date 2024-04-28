import snowflake.connector
import snowflake.connector
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

import matplotlib.pyplot as plt
from mplsoccer import Pitch, VerticalPitch
import matplotlib.patches as patches
import math


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
        SNOWFLAKE_PASSWORD = 'Snowfl@key0014'
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
def fetch_data(_cursor, query):
    """
    Fetches data from Snowflake using the provided cursor and query.
    
    Parameters:
    cursor (snowflake.connector.cursor.SnowflakeCursor): The Snowflake cursor object.
    query (str): The SQL query to execute.
    
    Returns:
    pandas.DataFrame: The fetched data as a pandas DataFrame.
    """
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
