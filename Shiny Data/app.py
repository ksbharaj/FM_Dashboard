import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

standard_chart_data = pd.read_csv(r"C:\Users\ksbha\Documents\Python Scripts\Data Engineering Scraping project\Shiny Data\Standard_radar_chart_v2.csv")
attacking_chart_data = pd.read_csv(r"C:\Users\ksbha\Documents\Python Scripts\Data Engineering Scraping project\Shiny Data\Attacking_radar_chart_v2.csv")
defending_chart_data = pd.read_csv(r"C:\Users\ksbha\Documents\Python Scripts\Data Engineering Scraping project\Shiny Data\Defending_radar_chart_v2.csv")

def create_radar_chart(season, team_name, data, chart_name):
    team_data = data[data['TEAM_NAME'] == team_name]
    average_data = data[data['TEAM_NAME'] == 'AVERAGE']

    team_data = team_data[team_data['SEASON'] == season]
    average_data = average_data[average_data['SEASON'] == season]

    # Prepare data for plotting
    categories = team_data['Stats']
    norm_values = team_data['Norm_Values']
    average_norm_values = average_data['Norm_Values']
    values = team_data['Values']

    difference = [golden - average for golden, average in zip(norm_values, average_norm_values)]

    hover_text = (
    "<span style='font-size: 20px; color: #d3d3d3;'>%{theta}</span><br>"
    "<span style='font-size: 20px; color: white;'>Value: %{customdata[0]:.2f}</span><br>"
    "<span style='font-size: 15px; color: #d3d3d3;'>Difference from Average: %{customdata[1]:.2f}</span><extra></extra>"
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
        customdata=np.stack((values, difference), axis=-1),
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
        customdata=np.stack((values, difference), axis=-1),
        hovertemplate=hover_text,
        marker=dict(
            size=1  # Hides the markers by setting their size to zero
        )
    ))

    fig.add_layout_image(
        dict(
            source='https://i.imgur.com/9yKFcv4.png',
            xref="paper", yref="paper",
            x=0.296, y=1.01,
            sizex=0.405, sizey=1.05,
            opacity=0.7,  # Adjust opacity as needed
            layer="below",
            sizing="stretch"
        )
    )

    for i, (value, category) in enumerate(zip(values, categories)):
        angle = (i / float(len(categories))) * 2 * np.pi 
        x = 0.5 + (1.1) * np.cos(angle) / 4
        y = 0.48 + (1.1) * np.sin(angle) / 1.65

        annotation_text = \
        f"<span style='font-size: 10px;'><b>{category}</b></span><br>" \
        f"<span style='font-size: 13px; color: rgba(210, 210, 0, 1);'>{value:.2f}</span>"

        fig.add_annotation(
            x=x,
            y=y,
            xref="paper",
            yref="paper",
            text=annotation_text,  # Bold category name and value
            showarrow=False,
            font=dict(size=10, color='white'),
            align="center",
            xanchor='center',
            yanchor='middle',
        )

    # Update layout
    fig.update_layout(
        autosize=True,
        width=355,  # Set the width
        height=300,  # Set the height
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

st.title('Team Radar Charts')
team_name = st.selectbox('Select a Team', list(sorted(np.delete(standard_chart_data['TEAM_NAME'].unique(), 
                                                         np.where(standard_chart_data['TEAM_NAME'].unique() == 'AVERAGE')))))
season = st.selectbox('Select a Season', (standard_chart_data['SEASON'].unique()))

col1, col2 = st.columns(2)

with col1:
    # st.header('Standard Radar Chart')
    fig_standard = create_radar_chart(season, team_name, standard_chart_data, "Standard Radar Chart")
    st.plotly_chart(fig_standard, use_container_width=False)

with col2:
    # st.header('Attacking Radar Chart')
    fig_attacking = create_radar_chart(season, team_name, attacking_chart_data, "Attacking Radar Chart") # Assuming the same data is used
    st.plotly_chart(fig_attacking, use_container_width=False)

# with col3:
#     # st.header('Defending Radar Chart')
#     fig_defending = create_radar_chart(season, team_name, defending_chart_data, "Defending Radar Chart") # Assuming the same data is used
#     st.plotly_chart(fig_defending, use_container_width=False)

# # Call your function with the selected team






# 


