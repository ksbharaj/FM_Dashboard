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
def create_radar_chart(season, team_name, data, competition, chart_name, chart_width=1000, chart_height=820):
    team_data = data[data['TEAM_NAME'] == team_name]
    team_data = team_data[team_data['COMPETITION_ACRONYM'] == competition]
    average_data = data[data['TEAM_NAME'] == competition+'_'+str(season)+"_Average"]

    team_data = team_data[team_data['SEASON'] == season]
    # average_data = average_data[average_data['SEASON'] == season]

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
        x = 0.5 + (1.1) * np.cos(angle) / 3.5
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
            font=dict(size=12, color='white'),
            align="center",
            xanchor='center',
            yanchor='middle',
            # sizing="contain",
            bordercolor="rgba(0, 0, 0, 0)",
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
            'yanchor': 'top'  # Ensures the title is at the top of the y position
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

# x_min_goal_output = (team_goal_output['EXPECTED GOALS AGAINST PER GAME'].min()*0.925)
# x_max_goal_output = (team_goal_output['EXPECTED GOALS AGAINST PER GAME'].max()*1.075)
# y_min_goal_output = (team_goal_output['NON PENALTY EXPECTED GOALS PER GAME'].min()*0.85)
# y_max_goal_output = (team_goal_output['NON PENALTY EXPECTED GOALS PER GAME'].max()*1.07)

x_min_goal_output = 0.45
x_max_goal_output = 2.60
y_min_goal_output = 0.5
y_max_goal_output = 2.40

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
                size=1,  # Set a fixed size or normalize as you prefer
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
            'y':0.98,  # Sets the y position of the title (1 is the top of the figure)
            'x':0.5,  # Centers the title horizontally (0.5 is the center of the figure)
            'xanchor': 'center',  # Ensures the title is centered at the x position
            'yanchor': 'top'  # Ensures the title is at the top of the y position
        },
        margin=dict(l=10, r=30, t=50, b=10),
        images= [dict(
            source= row["TEAM_LOGO_URL"],
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


# Function to create the xG Table
def create_xG_table(df_sel, season, competition):

    df_filtered = df_sel[df_sel['COMPETITION_ACRONYM'] == competition]
    df_filtered = df_filtered[df_filtered['SEASON'] == season]

    PTS_table = df_filtered[['TEAM_NAME','TEAM_PTS', 'GOAL_DIFFERENCE','GOALS_SCORED','GOALS_CONCEDED']].sort_values(['TEAM_PTS', 
                    'GOAL_DIFFERENCE','GOALS_SCORED','GOALS_CONCEDED'], ascending=False).reset_index(drop=True).reset_index()
    PTS_table.rename(columns={'index':'POS_NUM'}, inplace=True)
    PTS_table.POS_NUM += 1

    def ordinal(n):
        return "%d%s" % (n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])

    PTS_table['POS'] = PTS_table['POS_NUM'].apply(ordinal)

    XPTS_table = df_filtered[['TEAM_NAME','TEAM_XPTS', 'GOAL_DIFFERENCE','GOALS_SCORED','GOALS_CONCEDED']].sort_values([
        'TEAM_XPTS', 'GOAL_DIFFERENCE','GOALS_SCORED','GOALS_CONCEDED'],
                                            ascending=False).reset_index(drop=True).reset_index()

    XPTS_table.rename(columns={'index':'XPTS POS_NUM'}, inplace=True)
    XPTS_table['XPTS POS_NUM'] += 1

    XPTS_table['XPTS POS'] = XPTS_table['XPTS POS_NUM'].apply(ordinal)

    PTS_XPTS_table = XPTS_table.merge(PTS_table, how='left', on=['TEAM_NAME','GOAL_DIFFERENCE','GOALS_SCORED','GOALS_CONCEDED'])

    PTS_XPTS_table = PTS_XPTS_table[['TEAM_NAME','XPTS POS','XPTS POS_NUM','POS','POS_NUM']]

    df_filtered = df_filtered.merge(PTS_XPTS_table, how='left', on='TEAM_NAME')

    df_filtered.rename(columns={'TEAM_LOGO_URL':'TEAM_LOGO'}, inplace=True)

    fin_table = df_filtered[['POS','POS_NUM','TEAM_LOGO','XPTS POS','XPTS POS_NUM','TEAM_NAME','XG','GOALS_SCORED','XG_AGAINST',
            'GOALS_CONCEDED','TEAM_PTS','TEAM_XPTS']].sort_values('XPTS POS_NUM')
    fin_table['POS_NUM_DIFF'] = fin_table.apply(lambda row: 'https://i.imgur.com/AACUEGy.png' if row['POS_NUM'] == row['XPTS POS_NUM'] else \
                                                'https://i.imgur.com/5sTTYXm.png' if row['POS_NUM'] < row['XPTS POS_NUM'] else 'https://i.imgur.com/dGsmsnm.png', axis=1)
    fin_table['XG_DIFF'] = fin_table['GOALS_SCORED']-fin_table['XG']
    fin_table['XGA_DIFF'] = fin_table['GOALS_CONCEDED']-fin_table['XG_AGAINST']
    fin_table['XPTS_DIFF'] = fin_table['TEAM_PTS']-fin_table['TEAM_XPTS']
    fin_table = fin_table[['POS','XPTS POS','POS_NUM_DIFF','TEAM_LOGO','TEAM_NAME','XG','XG_DIFF',
            'XG_AGAINST','XGA_DIFF','TEAM_XPTS','XPTS_DIFF']]
    fin_table['XG_DIFF'] = fin_table['XG_DIFF'].astype(int)
    fin_table['XGA_DIFF'] = fin_table['XGA_DIFF'].astype(int)
    fin_table['XPTS_DIFF'] = fin_table['XPTS_DIFF'].astype(int)

    fin_table = fin_table.set_index(['POS'])

    team_name_cols = ['TEAM_LOGO','TEAM_NAME']
    xG_cols = ['XG','XG_DIFF']
    xGA_cols = ['XG_AGAINST','XGA_DIFF']
    xPTS_cols = ['TEAM_XPTS','XPTS_DIFF']

    fin_table['TEAM_LOGO'] = fin_table['TEAM_LOGO'].apply(lambda x: (io.BytesIO(urllib.request.urlopen(x).read())))
    fin_table['POS_NUM_DIFF'] = fin_table['POS_NUM_DIFF'].apply(lambda x: (io.BytesIO(urllib.request.urlopen(x).read())))

    df = fin_table.copy()

    colors_pos = [(1, 0, 0), (0, 0.6, 0)]  # Red to Green
    colors_neg = [(0, 0.6, 0), (1, 0, 0)]  # Green to Red
    n_bins = 2 # Discretizes the interpolation into bins
    cmap_name = 'custom_red_green'
    cm_pos = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors_pos, N=n_bins)
    cm_neg = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors_neg, N=n_bins)

    col_defs = (
        [
        ColDef(name="POS", title="", group="POS",textprops={"ha": "center", "weight": "bold"}, width=0.3),
        ColDef(name="XPTS POS", title="", group="XPTS POS", textprops={"ha": "center", "weight": "bold"}, width=0.3),
        ColDef(name="POS_NUM_DIFF", title="", group="NAME",textprops={"ha": "center"}, width=0.06,plot_fn=image),
        ColDef(name="TEAM_LOGO", title="", group="NAME",textprops={"ha": "center"}, width=0.2, plot_fn=image),
        ColDef(name="TEAM_NAME", title="",group="NAME",textprops={"ha": "left"}, width=0.8),
        ColDef(name="XG", title="", group="XG",textprops={"ha": "center"}, formatter="{:.1f}", width=0.3),
        ColDef(name="XG_DIFF", title="", group="XG", textprops={"ha": "center","fontsize": 14,"fontweight": "bold","bbox": {"boxstyle": "circle,pad=0.1"}}, 
               formatter=plus_sign_formatter, width=0.2, cmap=centered_cmap(df["XG_DIFF"], cmap=cm_pos, center=0)),
        ColDef(name="XG_AGAINST", title="", group="XGA",textprops={"ha": "center",}, formatter="{:.1f}", width=0.3),
        ColDef(name="XGA_DIFF", title="", group="XGA",textprops={"ha": "center","fontsize": 14,"fontweight": "bold","bbox": {"boxstyle": "circle,pad=0.1"}}, 
               width=0.2, formatter=plus_sign_formatter,cmap=centered_cmap(df["XG_DIFF"], cmap=cm_neg, center=0)),
        ColDef(name="TEAM_XPTS", title="", group="XPTS",textprops={"ha": "center"}, formatter="{:.1f}", width=0.3),
        ColDef(name="XPTS_DIFF", title="", group="XPTS",textprops={"ha": "center","fontsize": 14,"fontweight": "bold","bbox": {"boxstyle": "circle,pad=0.1"}}, 
               width=0.2, formatter=plus_sign_formatter,cmap=centered_cmap(df["XG_DIFF"], cmap=cm_pos, center=0)),
        ]
    )
    
    plt.rcParams["font.family"] = ["Roboto"]
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["text.color"] = "#e0e8df"

    fig, ax = plt.subplots(figsize=(9, 9))

    ax.set_title('XG TABLE', fontsize=15, fontweight='bold', color='gold', loc='left',pad=15)

    subtitle_y_position = 1.005  # Adjust this value as needed
    ax.text(0.0, subtitle_y_position, 'Premier Division, 2223 Season', fontsize=11, color='lightgray', 
            transform=ax.transAxes)

    fig.set_facecolor("#464646")
    ax.set_facecolor("#464646")

    fig.subplots_adjust(top=0.85, bottom=0.1, left=0.1, right=0.9)

    table = Table(
        df,
        column_definitions=col_defs,
        row_dividers=True,
        footer_divider=True,
        ax=ax,
        textprops={"fontsize": 12},
        row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
        col_label_divider={"linewidth": 1, "linestyle": (0, (1, 5))},
    )

    table.autoset_fontcolors(colnames=["XG_DIFF", "XGA_DIFF", "XPTS_DIFF"])

    fig.subplots_adjust(top=0.85, bottom=0.1, left=0.1, right=0.9)
    return fig


# Creation of the Streamit App
st.set_page_config(layout="centered")
css='''
    <style>
        section.main > div {max-width:100rem}
    </style>
    '''
st.markdown(css, unsafe_allow_html=True)
st.title('Team Analytics')

col1, col2, col3 = st.columns([1, 6, 1])

with col1:
    st.image(io.BytesIO(urllib.request.urlopen("https://i.imgur.com/qEwoaGU.png").read()), use_column_width=True)

with col2:
    competition =  st.selectbox('Select a Competition', (standard_chart_data['COMPETITION_ACRONYM'].unique()))
    filtered_comp = standard_chart_data[standard_chart_data['COMPETITION_ACRONYM'] == competition]
    season = st.selectbox('Select a Season', (filtered_comp['SEASON'].unique()))
    filtered_standard = filtered_comp[filtered_comp['SEASON'] == season]
    team_name = st.selectbox('Select a Team', list(sorted([name for name in filtered_standard['TEAM_NAME'].unique() if "Average" not in name])))

with col3:
    st.image(io.BytesIO(urllib.request.urlopen(str(team_names[team_names['TEAM_NAME'] == team_name].TEAM_LOGO_URL.iloc[0])).read()), use_column_width=False)


tabs = st.tabs(["Radar Charts", "General Charts"])

with tabs[0]:
    # Create a two-column layout
    col1, col2 = st.columns([2, 1])  # Adjust the ratio if needed

    # Use the first column for the Standard Radar Chart
    with col1:
        fig_standard = create_radar_chart(season, team_name, standard_chart_data, competition, "Standard Radar Chart") 
        st.plotly_chart(fig_standard, use_container_width=False)  # Set to True to use the full width of the column

    # Use the second column for the Attacking and Defending Radar Charts
    with col2:
        fig_attacking = create_radar_chart(season, team_name, attacking_chart_data, competition, "Attacking Radar Chart", 500, 400)
        st.plotly_chart(fig_attacking, use_container_width=False)  # Set to True to use the full width of the column

        fig_defending = create_radar_chart(season, team_name, defending_chart_data, competition, "Defending Radar Chart", 500, 400)
        st.plotly_chart(fig_defending, use_container_width=False)  # Set to True to use the full width of the column

# Future Content Tab
with tabs[1]:
    filtered_misc = team_misc[team_misc['SEASON'] == season]
    filtered_misc = filtered_misc[filtered_misc['COMPETITION_ACRONYM'] == competition]
    fig_team_aerial_duels = create_FM_team_scatter_chart(filtered_misc, 'AERIAL', team_name, 'AERIAL DUELS WON RATIO (%)', 'AERIAL DUELS ATTEMPTED PER GAME', 
                                                         1.15, x_min_aerial, x_max_aerial, y_min_aerial, y_max_aerial, 
                                                         "Fewer Duels<br>Poor Dueling", "Fewer Duels<br>Strong Dueling",
                                                         "Lots of Duels<br>Poor Dueling", "Lots of Duels<br>Strong Dueling", "red", 
                                                         "orange", "orange", "green")
    st.plotly_chart(fig_team_aerial_duels, use_container_width=False)

    filtered_goal_output = team_goal_output[team_goal_output['SEASON'] == season]
    filtered_goal_output = filtered_goal_output[filtered_goal_output['COMPETITION_ACRONYM'] == competition]
    fig_team_aerial_duels = create_FM_team_scatter_chart(filtered_goal_output, 'GOAL OUTPUT', team_name, 'EXPECTED GOALS AGAINST PER GAME', 'NON PENALTY EXPECTED GOALS PER GAME', 
                                                         0.095, x_min_goal_output, x_max_goal_output, y_min_goal_output, y_max_goal_output, 
                                                         "Low non-penalty expected goals<br>Strong Defending", "Low non-penalty expected goals<br>Poor Defending",
                                                         "High non-penalty expected goals<br>Strong Defending", "High non-penalty expected goals<br>Poor Defending", 
                                                         "orange", "red", "green", "orange")
    st.plotly_chart(fig_team_aerial_duels, use_container_width=False)

    fig_xG_table = create_xG_table(df_table, season, competition)
    st.pyplot(fig_xG_table, use_container_width=True)


# st.sidebar.caption("Note: Expand the plot for the best viewing experience.")


