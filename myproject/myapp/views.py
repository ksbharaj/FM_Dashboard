from django.http import JsonResponse
import snowflake.connector
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
from .config import SNOWFLAKE_CONFIG  # Import the configuration
import json



def create_radar_chart(season, team_name, data, competition, chart_name):
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

def fetch_and_create_chart(competition, season, team):
    # Connect to the database and fetch data
    conn = snowflake.connector.connect(
        user=SNOWFLAKE_CONFIG['user'],
        password=SNOWFLAKE_CONFIG['password'],
        account=SNOWFLAKE_CONFIG['account'],
        warehouse=SNOWFLAKE_CONFIG['warehouse'],
        database=SNOWFLAKE_CONFIG['database'],
        schema='RADAR_CHARTS'
    )

    try:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM STANDARD_RADAR')
        rows = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        data = pd.DataFrame(rows, columns=column_names)

        # Filter data based on the request
        filtered_comp = data[data['COMPETITION_ACRONYM'] == competition] if competition else data
        filtered_season = filtered_comp[filtered_comp['SEASON'] == season] if season else filtered_comp
        # filtered_team = filtered_season[filtered_season['TEAM_NAME'] == team] if team else filtered_season

        # Create the radar chart
        fig = create_radar_chart(season, team, filtered_season, competition, 'Radar Chart Example')

        # Convert the figure to HTML
        chart_html = fig.to_html(full_html=False)
        

        return chart_html
    
    finally:
        # Always close the connection
        conn.close()
    

def radar_chart_view(request):
    # Get parameters from request
    competition_selected = request.GET.get('competition', 'EPL')
    season_selected = request.GET.get('season', 2324)
    team_selected = request.GET.get('team', 'Arsenal')

    # Use the shared function to get the chart HTML
    chart_html = fetch_and_create_chart(competition_selected, season_selected, team_selected)

    # Rest of your logic to render the page...
    return JsonResponse({'chart_html': chart_html})


def ajax_radar_chart_view(request):
    # Get parameters from request
    competition_selected = request.GET.get('competition')
    season_selected = request.GET.get('season')
    team_selected = request.GET.get('team')

    # Use the shared function to get the chart HTML
    chart_html = fetch_and_create_chart(competition_selected, season_selected, team_selected)

    return JsonResponse({'chart_html': chart_html})
    

