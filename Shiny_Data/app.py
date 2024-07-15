import streamlit as st
import base64
from PIL import Image

import io
import urllib
from plottable.plots import image

import pandas as pd

## Functions for calling snowflake data
from app_functions import get_snowflake_cursor, fetch_data

## Functions for creating charts
from app_functions import create_radar_chart, create_FM_team_scatter_chart, plot_defensive_actions, \
                          create_set_piece_first_contacts_plot, plot_shot_data, xT_generator, \
                          plot_match_momentum, rest_dict_pass_map, plot_single_passmap, plot_double_passmap, \
                          plot_xg_match_story

## Configure the streamlit page 
st.set_page_config(layout="centered")
css='''
    <style>
        section.main > div {max-width:85rem}
    </style>
    '''
st.markdown(css, unsafe_allow_html=True)

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) \
           Chrome/58.0.3029.110 Safari/537.3'}

# ==================================================================
# Section: Download Chart Data from Snowflake
# ==================================================================

cursor = get_snowflake_cursor('RADAR_CHARTS')

## Download Radar Charts

standard_radar_chart_data = fetch_data(cursor, 'SELECT * FROM STANDARD_RADAR')
attacking_radar_chart_data = fetch_data(cursor, 'SELECT * FROM ATTACKING_RADAR')
defending_radar_chart_data = fetch_data(cursor, 'SELECT * FROM DEFENDING_RADAR')

## Download XPts Table Images
xpts_table_images = fetch_data(cursor, 'SELECT * FROM XPTS_TABLE')

## Download Scatter Chart Plots
team_defending_chart = fetch_data(cursor, 'SELECT * FROM GEGENSTATS.RADAR_CHARTS.TEAM_DEFENDING_CHART')
team_goal_output = fetch_data(cursor, 'SELECT * FROM GEGENSTATS.RADAR_CHARTS.TEAM_GOAL_OUTPUT')
pressing_intensity_chart = fetch_data(cursor, 'SELECT * FROM GEGENSTATS.RADAR_CHARTS.PRESSING_INTENSITY_CHART')
set_piece_efficiency_chart = fetch_data(cursor, 'SELECT * FROM GEGENSTATS.RADAR_CHARTS.SET_PIECE_EFFICIENCY_CHART')
own_set_piece_efficiency_chart = fetch_data(cursor, 'SELECT * FROM GEGENSTATS.RADAR_CHARTS.OWN_SET_PIECE_EFFICIENCY_CHART')
crossing_chart = fetch_data(cursor, 'SELECT * FROM GEGENSTATS.RADAR_CHARTS.CROSSING_CHART')
attacking_efficiency_chart = fetch_data(cursor, 'SELECT * FROM GEGENSTATS.RADAR_CHARTS.TEAM_ATTACKING_EFFICIENCY_CHART')
scoring_chart = fetch_data(cursor, 'SELECT * FROM GEGENSTATS.RADAR_CHARTS.TEAM_SCORING_CHART')
shooting_chart = fetch_data(cursor, 'SELECT * FROM GEGENSTATS.RADAR_CHARTS.TEAM_SHOOTING_CHART')
df_goals_set_piece_chart = fetch_data(cursor, 'SELECT * FROM GEGENSTATS.RADAR_CHARTS.GOALS_FROM_SET_PIECES')
df_shots_last_5_matches = fetch_data(cursor, 'SELECT * FROM  GEGENSTATS.RADAR_CHARTS.SHOT_MAP')

## Download Pitch Map Plots
section_counts_percentage = fetch_data(cursor, 'SELECT * FROM  GEGENSTATS.RADAR_CHARTS.TEAM_DEFENSIVE_ACTIONS')
def_set_piece_final = fetch_data(cursor, 'SELECT * FROM  GEGENSTATS.RADAR_CHARTS.TEAM_DEF_SET_PIECE_FIRST_CONTACTS')
att_set_piece_final = fetch_data(cursor, 'SELECT * FROM  GEGENSTATS.RADAR_CHARTS.TEAM_ATT_SET_PIECE_FIRST_CONTACTS')

## Renaming Charts Charts for better readability
team_defending_chart.rename(columns={'BLOCKS_PER_GAME':'BLOCKS PER GAME', 'CLEARANCES_PER_GAME':'CLEARANCES PER GAME',
                                    'SHOTS_FACED_PER_GAME':'SHOTS FACED PER GAME', 'OPPOSITION_CONVERSION_RATE':'OPPOSITION CONVERSION RATE (%)',
                                    'CONCEDED_PER_GAME':'CONCEDED PER GAME', 'TACKLES_ATTEMPTED_PER_GAME':'TACKLES ATTEMPTED PER GAME',
                                    'TACKLES_WON_RATIO':'TACKLES WON RATIO (%)'}, inplace=True)

team_goal_output.rename(columns={'EXPECTED_GOALS_AGAINST_PER_GAME':'EXPECTED GOALS AGAINST PER GAME',
                                'NON_PENALTY_EXPECTED_GOALS_PER_GAME':'NON PENALTY EXPECTED GOALS PER GAME'}, inplace=True)

pressing_intensity_chart.rename(columns={'AVERAGE_DEFENSIVE_ACTION_FROM_DEFENDERS':'AVERAGE DEFENSIVE ACTION FROM DEFENDERS (YARDS)',
                                         'OPPOSITION_PASSES_PER_DEFENSIVE_ACTION':'OPPOSITION PASSES PER DEFENSIVE ACTION'}, inplace=True)

set_piece_efficiency_chart.rename(columns={'OPPOSITION_XG_FROM_SET_PIECE_CROSSES_PER_GAME':'OPPOSITION XG FROM SET PIECE CROSSES PER GAME',
                                            'OPPOSITION_CROSSES_FROM_SET_PIECE_PER_GAME':'OPPOSITION CROSSES FROM SET PIECE PER GAME'}, inplace=True)

own_set_piece_efficiency_chart.rename(columns={'XG_FROM_SET_PIECE_CROSSES_PER_GAME':'XG FROM SET PIECES CROSSES PER GAME',
                                            'CROSSES_FROM_SET_PIECES_PER_GAME':'CROSSES FROM SET PIECES PER GAME'}, inplace=True)

crossing_chart.rename(columns={'CROSS_COMPLETION':'CROSS COMPLETION (%)', 'CROSSES_ATTEMPTED_PER_GAME':'CROSSES ATTEMPTED PER GAME'}, inplace=True)

attacking_efficiency_chart.rename(columns={'CONVERSION_RATE':'CONVERSION RATE (%)', 'SHOTS_PER_GAME':'SHOTS PER GAME'}, inplace=True)

scoring_chart.rename(columns={'GOALS_PER_GAME':'GOALS PER GAME','NPXG_PER_GAME':'NPXG PER GAME'}, inplace=True)

shooting_chart.rename(columns={'SHOTS_ON_TARGET_PER_GAME':'SHOTS ON TARGET PER GAME','XG_PER_SHOT':'XG PER SHOT'}, inplace=True)

df_goals_set_piece_chart.rename(columns={'SET_PIECE_GOALS_SCORED':'SET PIECE GOALS SCORED',
                                'SET_PIECE_GOALS_CONCEDED':'SET PIECE GOALS CONCEDED'}, inplace=True)

# ==================================================================
# Section: Download Table Data from Snowflake
# ==================================================================

cursor_1 = get_snowflake_cursor('TABLES')

## Download Team, Competition and Season Data
team_names = fetch_data(cursor_1, 'SELECT * FROM TEAMS')
df_competitions = fetch_data(cursor_1, 'SELECT COMPETITION, COMPETITION_ACRONYM, SEASON FROM COMPETITIONS')
df_competition_logos = fetch_data(cursor_1, 'SELECT COMPETITION_ACRONYM, COMPETITION_LOGO  FROM COMPETITIONS')
df_seasons = fetch_data(cursor_1, 'SELECT * FROM SEASONS')

## Download Team statistics
team_misc = fetch_data(cursor_1, 'SELECT * FROM TEAM_MISC_STATS')
team_standard = fetch_data(cursor_1, 'SELECT * FROM TEAM_STANDARD_STATS')
team_attacking = fetch_data(cursor_1, 'SELECT * FROM TEAM_ATTACKING_STATS')
team_defending = fetch_data(cursor_1, 'SELECT * FROM TEAM_DEFENDING_STATS')

## Drop duplicates from competition logos dataframe
df_competition_logos = df_competition_logos.drop_duplicates()

# ==================================================================
# Section: Build DataFrames
# ==================================================================

## Create a Misc Stats for Aerial Duels Data
team_misc = team_misc.merge(df_competitions[['COMPETITION','COMPETITION_ACRONYM','SEASON']], on=['COMPETITION','SEASON'], how='left')
team_misc = team_misc.merge(team_names, on='TEAM_FBREF_ID', how='left')
team_misc = team_misc.merge(team_standard, on=['TEAM_FBREF_ID', 'SEASON', 'COMPETITION'], how='left')
team_misc['AERIAL DUELS WON RATIO (%)'] = team_misc['AERIALS_WON']*100/(team_misc['AERIALS_WON'] + team_misc['AERIALS_LOST'])
team_misc['AERIAL DUELS ATTEMPTED PER GAME'] = (team_misc['AERIALS_WON'] + team_misc['AERIALS_LOST'])/team_misc['MATCHES_PLAYED']

# ==================================================================
# Section: Set chart limits
# ==================================================================

## For aerial duels plot
x_min_aerial = round((team_misc['AERIAL DUELS WON RATIO (%)'].min() // 5)*5 , 2)
x_max_aerial = round((team_misc['AERIAL DUELS WON RATIO (%)'].max()//5)*5 + 5, 2)
y_min_aerial = round((team_misc['AERIAL DUELS ATTEMPTED PER GAME'].min()//5)*5, 2)
y_max_aerial = round((team_misc['AERIAL DUELS ATTEMPTED PER GAME'].max()//5)*5 + 5, 2)

## For goal output
x_min_goal_output = round((team_goal_output['EXPECTED GOALS AGAINST PER GAME'].min()//0.05)*0.05 - 0.1, 2)
x_max_goal_output = round((team_goal_output['EXPECTED GOALS AGAINST PER GAME'].max()//0.05)*0.05 + 0.15, 2)
y_min_goal_output = round((team_goal_output['NON PENALTY EXPECTED GOALS PER GAME'].min()//0.05)*0.05 - 0.1, 2)
y_max_goal_output = round((team_goal_output['NON PENALTY EXPECTED GOALS PER GAME'].max()//0.05)*0.05 + 0.15, 2)

# ====================================================================================================================================

# ====================================================================================================================================

# ==================================================================
# Section: Streamlit app creation
# ==================================================================

## Streamlit App's Title 
st.title('Team Analytics')
st.subheader("Hey! Analyze football data from the top 5 leagues in the last 3 seasons here! (Inspired by Football Manager 24)")
st.subheader("**Select the ***Competition***, ***Season*** and ***Team*** of from the sidebar. Then select from any one of the Radio Buttons:**")
st.markdown(" **Radar Charts 🌐**: Help summarize the team's statistics in relation to the league average for that season")
st.markdown(" **General Charts 🧿**: Show the team's Aerial Dominance and Goal Output against all other teams in the league. Also shows a personal favourite- the xPts table (or Justice table)")
st.markdown(" **Defending Charts 🤚**: Various interesting charts to show just how good the team is at defending")
st.markdown(" **Creating Charts 🪄**: Shows how effective the selected team is at crossing in-plan and from set-pieces")
st.markdown(" **Scoring Charts ⚽**: Various charts showing the goal threat the team possesses, including an Interactive Shot Map!")
st.markdown(" **Last Match ⏮️**: Analyze Shot Maps, Match Momentum and Pass Maps from the last game!")
st.markdown("""---""")


# ==================================================================
# Section: Streamlit's Sidebar
# ==================================================================
if 'competition' not in st.session_state:
    st.session_state.competition = "Serie A"
    st.session_state.season = 2324
    st.session_state.team_name = "Atalanta"
    st.session_state.proper_team_update = True

with st.sidebar:   
    st.write("If the Competition or Season selection changes, you may be asked to press a ***Load Teams*** button before progressing!")
    # with st.form(key='comp_season_form', border=False):
    # Display select box and update session state
    competitions = standard_radar_chart_data['COMPETITION_ACRONYM'].unique()
    competition_selected = st.selectbox(
        'Select a Competition',
        competitions,
        index=competitions.tolist().index(st.session_state.competition)
    )

    st.session_state.competition = competition_selected

    # Function to display logo
    def show_competition_logo(competition_acronym):
        logo_url = df_competition_logos[df_competition_logos['COMPETITION_ACRONYM'] == competition_acronym]['COMPETITION_LOGO'].iloc[0]
        response = urllib.request.urlopen(urllib.request.Request(logo_url, headers=headers))
        image = io.BytesIO(response.read())
        st.image(image, width=180)
    
    # Show logo if you want it to appear before submitting
    show_competition_logo(competition_selected)

    # Select season
    filtered_comp = standard_radar_chart_data[standard_radar_chart_data['COMPETITION_ACRONYM'] == competition_selected]
    available_seasons = filtered_comp['SEASON'].unique()
    avai_season_names = df_seasons[df_seasons['SEASON'].isin(available_seasons)][['SEASON', 'SEASON_NAME']]
    season_selected = st.selectbox('Select a Season', sorted(avai_season_names['SEASON_NAME'].to_list(), reverse=True))
    filtered_season = avai_season_names[avai_season_names['SEASON_NAME'] == season_selected]['SEASON'].iloc[0]
    filtered_comp_season = filtered_comp[filtered_comp['SEASON'] == filtered_season]

    st.session_state.season = filtered_season

    valid_team_names = list(sorted([name for name in filtered_comp_season['TEAM_NAME'].unique() if "Average" not in name]))

    if st.session_state.team_name not in valid_team_names:
        st.session_state.proper_team_update = False
        with st.form(key='comp_season_form'):
            submit_comp_season = st.form_submit_button(label='Load Teams')
        if submit_comp_season:
            st.session_state.competition = competition_selected
            st.session_state.season = filtered_season
            st.session_state.team_name = valid_team_names[1]
            st.session_state.proper_team_update = True

    
    if st.session_state.proper_team_update:
        team_name_selected = st.selectbox('Select a Team', valid_team_names)
        st.session_state.team_name = team_name_selected
        

        # Display team logo
        team_logo_url = team_names[team_names['TEAM_NAME'] == team_name_selected].TEAM_LOGO_URL.iloc[0]
        st.image(io.BytesIO(urllib.request.urlopen(urllib.request.Request(team_logo_url, headers=headers)).read()), width=180)

        # submit_team = st.form_submit_button(label='LOAD DATA HERE!')
        
        st.write(" ")
        st.write("Kindly maximize your browser window for the best viewing experience.")


# ==================================================================
# Section: Streamlit's Tabs
# ==================================================================

if st.session_state.proper_team_update:
    if 'active_tab' not in st.session_state:
        st.session_state['active_tab'] = 'Radar Charts'
    # tabs = st.tabs(["Radar Charts", "General Charts", "Defending Charts"])

    tab_options = ['Radar Charts 🌐', 'General Charts 🧿', 'Defending Charts 🤚', 'Creating Charts 🪄', 'Scoring Charts ⚽',
                   'Last Match ⏮️']
    st.session_state['active_tab'] = st.radio("Select a tab:", tab_options, horizontal=True)

    if st.session_state['active_tab'] == tab_options[0]:
        # Create a two-column layout
        col1, col2 = st.columns([1.4, 0.75])  # Adjust the ratio if needed

        # Use the first column for the Standard Radar Chart
        with col1:
            fig_standard = create_radar_chart(st.session_state.season, st.session_state.team_name, standard_radar_chart_data, 
                                            st.session_state.competition, "Standard Radar Chart") 
            st.caption("Comparison of Standard Stats between " + st.session_state.team_name + " (Golden) and " + st.session_state.competition + 
                       "'s league average in the " + season_selected + " season (Grey)")
            st.plotly_chart(fig_standard, use_container_width=False, config={'displayModeBar': False})  # Set to True to use the full width of the column

        # Use the second column for the Attacking and Defending Radar Charts
        with col2:
            fig_attacking = create_radar_chart(st.session_state.season, st.session_state.team_name, attacking_radar_chart_data, 
                                            st.session_state.competition, "Attacking Radar Chart", 420, 350, label_spread=3)
            st.caption("Here, we focus on the Attacking Stats (Top) and Defending Stats (Bottom)")
            st.plotly_chart(fig_attacking, use_container_width=False, config={'displayModeBar': False})  # Set to True to use the full width of the column

            fig_defending = create_radar_chart(st.session_state.season, st.session_state.team_name, defending_radar_chart_data, 
                                            st.session_state.competition, "Defending Radar Chart", 420, 350, label_spread=3)
            st.plotly_chart(fig_defending, use_container_width=False, config={'displayModeBar': False})  # Set to True to use the full width of the column

    elif st.session_state['active_tab'] == tab_options[1]:
        col1, col2 = st.columns([1, 1])  # Adjust the ratio if needed

        with col1:
            filtered_misc = team_misc[team_misc['SEASON'] == st.session_state.season]
            filtered_misc = filtered_misc[filtered_misc['COMPETITION_ACRONYM'] == st.session_state.competition]
            fig_team_aerial_duels = create_FM_team_scatter_chart(filtered_misc, 'AERIAL', st.session_state.team_name, 
                                                                'AERIAL DUELS WON RATIO (%)', 'AERIAL DUELS ATTEMPTED PER GAME', 
                                                                1.45, x_min_aerial, x_max_aerial, y_min_aerial, y_max_aerial, 
                                                                "Fewer Duels<br>Poor Dueling", "Fewer Duels<br>Strong Dueling",
                                                                "Lots of Duels<br>Poor Dueling", "Lots of Duels<br>Strong Dueling", "red", 
                                                                "orange", "orange", "green")
            st.caption("Comparison of Aerial Duels attempted and the win % by " + st.session_state.team_name + " in the " + 
                       st.session_state.competition + " in the " + season_selected + " season")
            st.plotly_chart(fig_team_aerial_duels, use_container_width=False)


        with col2:
            filtered_goal_output = team_goal_output[team_goal_output['SEASON'] == st.session_state.season]
            filtered_goal_output = filtered_goal_output[filtered_goal_output['COMPETITION_ACRONYM'] == st.session_state.competition]
            fig_team_aerial_duels = create_FM_team_scatter_chart(filtered_goal_output, 'GOAL OUTPUT', st.session_state.team_name, 
                                                                'EXPECTED GOALS AGAINST PER GAME', 'NON PENALTY EXPECTED GOALS PER GAME', 
                                                                0.1, x_min_goal_output, x_max_goal_output, y_min_goal_output, y_max_goal_output, 
                                                                "Low non-penalty expected goals<br>Strong Defending", "Low non-penalty expected goals<br>Poor Defending",
                                                                "High non-penalty expected goals<br>Strong Defending", "High non-penalty expected goals<br>Poor Defending", 
                                                                "orange", "red", "green", "orange")
            st.caption("Comparison of NPxG created and xG conceded by " + st.session_state.team_name + " in the " + 
                       st.session_state.competition + " in the " + season_selected + " season")
            st.plotly_chart(fig_team_aerial_duels, use_container_width=False)

        col3, col4, col5 = st.columns([1, 4, 1])  # Adjust the ratio if needed

        # @st.cache_data
        def load_image(competition, season):
            binary_data = base64.b64decode(xpts_table_images[(xpts_table_images['COMPETITION_ACRONYM'] == competition) &
            (xpts_table_images['SEASON'] == season)]['TABLE_IMAGE'].iloc[0])
            image_buffer = io.BytesIO(binary_data)
            image = Image.open(image_buffer)
            return image
        
        with col4:
            st.caption("xPts (Justice) Table for " + st.session_state.competition + " in the " + season_selected + 
                       " season. Positions sorted by xPts, calculated using Monte Carlo Simulation.")
            image = load_image(st.session_state.competition, st.session_state.season)
            st.image(image, use_column_width=True)

    elif st.session_state['active_tab'] == tab_options[2]:
        st.subheader("Expand the sections below to analyze various charts and graphs realted to Defending!")
        col1, col2 = st.columns([1, 1])  # Adjust the ratio if needed

        with col1:
            with st.expander("**TEAM DEFENDING**- shows "+ st.session_state.team_name + "'s blocks vs clearances per game"):
                filtered_defending = team_defending_chart[team_defending_chart['SEASON'] == st.session_state.season]
                filtered_defending = filtered_defending[filtered_defending['COMPETITION_ACRONYM'] == st.session_state.competition]
                fig_team_defending = create_FM_team_scatter_chart(filtered_defending, 'DEFENDING', st.session_state.team_name, 
                                                                'CLEARANCES PER GAME', 'BLOCKS PER GAME', 1., 9.5, 29.5, 6.5, 15.5, 
                                                                    "Fewer blocks<br>Fewer Clearances", "Fewer blocks<br>Lots of Clearances",
                                                                    "Lots of blocks<br>Fewer Clearances", "Fewer blocks<br>Lots of Clearances", 
                                                                    "red", "orange", "orange", "green")
                st.caption("Shows " + st.session_state.team_name + "'s position relative to the other teams in the " + 
                       st.session_state.competition + " in the " + season_selected + " season")
                st.plotly_chart(fig_team_defending, use_container_width=False)

        with col2:
            with st.expander("**DEFENSIVE ACTIONS**- shows the pitch zones where "+ st.session_state.team_name + "'s defenders defend"):
                section_counts_percentage_filt = section_counts_percentage[section_counts_percentage['SEASON'] == st.session_state.season]
                section_counts_percentage_filt = section_counts_percentage_filt[section_counts_percentage_filt['COMPETITION_ACRONYM'] == st.session_state.competition]
                section_counts_percentage_filt = section_counts_percentage_filt[section_counts_percentage_filt['TEAM_NAME'] == st.session_state.team_name]

                fig_team_defensive_actions = plot_defensive_actions(section_counts_percentage_filt)
                st.caption("Defensive actions here count as tackles, clearances, interceptiosn and blocks by Centre-Backs and Full-Backs ONLY")
                st.pyplot(fig_team_defensive_actions, use_container_width=True)


        col3, col4 = st.columns([1, 1])

        with col3:
            with st.expander("**DEFENSIVE EFFICIENCY**- shot rate conceded by "+ st.session_state.team_name + " and how well opponents convert against them"):
                fig_team_defensive_efficiency = create_FM_team_scatter_chart(filtered_defending, 'DEFENSIVE EFFICIENCY', st.session_state.team_name, 
                                                                'OPPOSITION CONVERSION RATE (%)', 'SHOTS FACED PER GAME',
                                                                    0.65, 7, 16, 5.5, 19, "Quiet defence<br>Impenetrable defence", 
                                                                    "Quiet defence<br>Leaky defence", "Busy defence<br>Impenetrable defence", 
                                                                    "Busy defence<br>Leaky defence", "green", "orange", "orange", "red")
                st.caption("Shows " + st.session_state.team_name + "'s position relative to the other teams in the " + 
                       st.session_state.competition + " in the " + season_selected + " season")
                st.plotly_chart(fig_team_defensive_efficiency, use_container_width=False)

        with col4:
            with st.expander("**GOALKEEPING**- goals conceded by "+ st.session_state.team_name + " vs the shots faced per game"):
                fig_team_goalkeeping = create_FM_team_scatter_chart(filtered_defending, 'GOALKEEPING', st.session_state.team_name, 'SHOTS FACED PER GAME', 
                                    'CONCEDED PER GAME', 0.45, 6, 18, 0.5, 2.8, "Impenetrable defence<br>Quiet defence", 
                                    "Impenetrable defence<br>Busy defence", "Leaky defence<br>Quiet defence", "Leaky defence<br>Busy defence", 
                                    "green", "orange", "orange", "red")
                st.caption("Shows " + st.session_state.team_name + "'s position relative to the other teams in the " + 
                       st.session_state.competition + " in the " + season_selected + " season")
                st.plotly_chart(fig_team_goalkeeping, use_container_width=False)

        col5, col6 = st.columns([1, 1])
        with col5:
            with st.expander("**TACKLING**- Tackles attempted per game by "+ st.session_state.team_name + " and their tackle win ratio"):
                fig_team_tackling = create_FM_team_scatter_chart(filtered_defending, 'TACKLING', st.session_state.team_name, 'TACKLES WON RATIO (%)', 
                                                                'TACKLES ATTEMPTED PER GAME', 0.6, 52, 67, 11, 24, 
                                                                    "Fewer Duels<br>Poor Dueling", "Fewer Duels<br>Strong Dueling",
                                                                    "Lots of Duels<br>Poor Dueling", "Lots of Duels<br>Strong Dueling", "red", 
                                                                    "orange", "orange", "green")
                st.caption("Shows " + st.session_state.team_name + "'s position relative to the other teams in the " + 
                       st.session_state.competition + " in the " + season_selected + " season")
                st.plotly_chart(fig_team_tackling, use_container_width=False)

        with col6:
            with st.expander("**PRESSING INTENSITY**- Opponent's PPDA vs Average Defensive Action from "+ st.session_state.team_name + "'s Defenders (CBs & FBs)"):
                filt_pressing_intensity_chart = pressing_intensity_chart[pressing_intensity_chart['SEASON'] == st.session_state.season]
                filt_pressing_intensity_chart = filt_pressing_intensity_chart[filt_pressing_intensity_chart['COMPETITION_ACRONYM'] == st.session_state.competition]
                fig_team_pressing_intensity = create_FM_team_scatter_chart(filt_pressing_intensity_chart, 'PRESSING INTENSITY', st.session_state.team_name, 
                                                                        'AVERAGE DEFENSIVE ACTION FROM DEFENDERS (YARDS)', 
                                                                'OPPOSITION PASSES PER DEFENSIVE ACTION', 0.75, 20, 35, 6, 24, 
                                                                "Low PPDA from Opposition<br>Defenders Defending Deeper", 
                                                                "Low PPDA from Opposition<br>Defenders Defending Higher",
                                                                "High PPDA from Opposition<br>Defenders Defending Deeper", 
                                                                "High PPDA from Opposition<br>Defenders Defending Higher", "red", "orange", "orange", "green")
                st.caption("Shows " + st.session_state.team_name + "'s position relative to the other teams in the " + 
                       st.session_state.competition + " in the " + season_selected + " season")
                st.plotly_chart(fig_team_pressing_intensity, use_container_width=False)

        col7, col8 = st.columns([1, 1])
        with col7:
            with st.expander("**SET PIECE DEFENSIVE EFFICIENCY**- Opponent's xG vs Total Crosses from Set Pieces from "+ st.session_state.team_name + "'s Opponents"):
                filt_set_piece_efficiency_chart = set_piece_efficiency_chart[set_piece_efficiency_chart['SEASON'] == st.session_state.season]
                filt_set_piece_efficiency_chart = filt_set_piece_efficiency_chart[filt_set_piece_efficiency_chart['COMPETITION_ACRONYM'] == st.session_state.competition]

                fig_set_piece_efficiency = create_FM_team_scatter_chart(filt_set_piece_efficiency_chart, 'SET PIECE DEFENSIVE EFFICIENCY', st.session_state.team_name, 
                                                                        'OPPOSITION CROSSES FROM SET PIECE PER GAME',
                                    'OPPOSITION XG FROM SET PIECE CROSSES PER GAME', 0.4, 2, 14, 0.05, 0.3, 
                                                                "Low xG conceded from crosses<br>Fewer crosses conceded", 
                                                                "Low xG conceded from crosses<br>Many crosses conceded",
                                                                "High xG conceded from crosses<br>Fewer crosses conceded", 
                                                                "High xG conceded from crosses<br>Many crosses conceded", 
                                                                "green", "orange", "orange", "red")
                st.caption("Shows " + st.session_state.team_name + "'s position relative to the other teams in the " + 
                       st.session_state.competition + " in the " + season_selected + " season")
                st.plotly_chart(fig_set_piece_efficiency, use_container_width=False)

        with col8:
            with st.expander("**SET PIECE FIRST CONTACTS IN OWN BOX**- Where in their box do "+ st.session_state.team_name + "get the first contact the most when defending set pieces?"):
                def_set_piece_final_filt = def_set_piece_final[def_set_piece_final['SEASON'] == st.session_state.season]
                def_set_piece_final_filt = def_set_piece_final_filt[def_set_piece_final_filt['COMPETITION_ACRONYM'] == st.session_state.competition]
                def_set_piece_final_filt = def_set_piece_final_filt[def_set_piece_final_filt['TEAM_NAME'] == st.session_state.team_name]

                def_set_piece_chart = def_set_piece_final_filt[['CROSS_END_LOCATION', 'PERC_1ST_CONTACT']].set_index('CROSS_END_LOCATION')

                fig_def_set_piece = create_set_piece_first_contacts_plot(def_set_piece_chart)
                st.caption("Zones are Far Post, Central and Near Post (Left to Right). The higher the percentage, the more the team gets the first contact in that zone.")
                st.pyplot(fig_def_set_piece, use_container_width=True)

    elif st.session_state['active_tab'] == tab_options[3]:
        col1, col2 = st.columns([1, 1])  # Adjust the ratio if needed
        with col1:
            filt_crossing_chart = crossing_chart[crossing_chart['SEASON'] == st.session_state.season]
            filt_crossing_chart = filt_crossing_chart[filt_crossing_chart['COMPETITION_ACRONYM'] == st.session_state.competition]
            fig_filt_crossing = create_FM_team_scatter_chart(filt_crossing_chart, 'CROSSING', st.session_state.team_name, 
                             'CROSS COMPLETION (%)', 'CROSSES ATTEMPTED PER GAME', 0.95, 17, 38, 11, 32, 
                                                        "Fewer crosses<br>Inaccurate crossing", 
                                                        "Fewer crosses<br>Accurate crossing",
                                                        "Lots of crosses<br>Inaccurate crossing", 
                                                        "Lots of crosses<br>Accurate crossing",
                                                        "red", "orange", "orange", "green")
            st.caption("Shows how frequently " + st.session_state.team_name + " cross the ball, and their success doing so in the " + 
                        st.session_state.competition + " in the " + season_selected + " season")
            st.plotly_chart(fig_filt_crossing, use_container_width=True)


        with col2:
            filt_own_set_piece_efficiency_chart = own_set_piece_efficiency_chart[own_set_piece_efficiency_chart['SEASON'] == st.session_state.season]
            filt_own_set_piece_efficiency_chart = filt_own_set_piece_efficiency_chart[filt_own_set_piece_efficiency_chart['COMPETITION_ACRONYM'] == st.session_state.competition]
            fig_own_set_piece_efficiency = create_FM_team_scatter_chart(filt_own_set_piece_efficiency_chart, 'SET PIECE ATTACKING EFFICIENCY', st.session_state.team_name, 
                                'CROSSES FROM SET PIECES PER GAME', 'XG FROM SET PIECES CROSSES PER GAME', 0.46, 3, 15, 0.05, 0.34, 
                                                            "Low xG per game<br>Fewer crosses per game", 
                                                            "Low xG conceded from crosses<br>Many crosses conceded",
                                                            "High xG per game<br>Fewer crosses per game", 
                                                            "Low xG per game<br>Many crosses conceded",
                                                            "red", "orange", "orange", "green")
            st.caption("Shows the danger " + st.session_state.team_name + " poses form their set pieces in the " + 
                        st.session_state.competition + " in the " + season_selected + " season")
            st.plotly_chart(fig_own_set_piece_efficiency, use_container_width=True)
    
    elif st.session_state['active_tab'] == tab_options[4]:
        col1, col2 = st.columns([1, 1])
        with col1:
            with st.expander("**ATTACKING EFFICIENCY**- shows "+ st.session_state.team_name + "'s shots per game vs their shot conversion rate"):
                filt_attacking_efficiency_chart = attacking_efficiency_chart[attacking_efficiency_chart['SEASON'] == st.session_state.season]
                filt_attacking_efficiency_chart = filt_attacking_efficiency_chart[filt_attacking_efficiency_chart['COMPETITION_ACRONYM'] == st.session_state.competition]
                
                fig_attacking_efficiency = create_FM_team_scatter_chart(filt_attacking_efficiency_chart, 'ATTACKING EFFICIENCY', st.session_state.team_name,
                                'CONVERSION RATE (%)', 'SHOTS PER GAME', 0.7, 5, 20, 7, 21, 
                                                            "Passive Shooting<br>Wasteful Shooting", 
                                                            "Passive Shooting<br>Clinical Shooting",
                                                            "Aggressive Shooting<br>Wasteful Shooting",
                                                            "Aggressive Shooting<br>Clinical Shooting",
                                                            "red", "orange", "orange", "green")
                st.caption("Shows " + st.session_state.team_name + "'s short rate and success at converting them to goals against all teams in the " + 
                            st.session_state.competition + " in the " + season_selected + " season")
                st.plotly_chart(fig_attacking_efficiency, use_container_width=True)
        
        with col2:
            with st.expander("**SCORING**- shows "+ st.session_state.team_name + "'s goals per game against Non-Penalty xG"):
                filt_scoring_chart = scoring_chart[scoring_chart['SEASON'] == st.session_state.season]
                filt_scoring_chart = filt_scoring_chart[filt_scoring_chart['COMPETITION_ACRONYM'] == st.session_state.competition]

                fig_scoring = create_FM_team_scatter_chart(filt_scoring_chart, 'SCORING', st.session_state.team_name, 
                             'NPXG PER GAME', 'GOALS PER GAME', 0.125, 0.5, 3.1, 0.5, 3.1, 
                                                        "Low Scoring<br>Low NPxG", 
                                                        "Low Scoring<br>High NPxG",
                                                        "High Scoring<br>Low NPxG",
                                                        "High Scoring<br>High NPxG",
                                                        "red", "orange", "orange", "green")
                st.caption("Gives a comparison of " + st.session_state.team_name + "'s goal-scoring ability relative to their non-penalty xG in the " + 
                            st.session_state.competition + " in the " + season_selected + " season")
                st.plotly_chart(fig_scoring, use_container_width=True)
        
        col3, col4 = st.columns([1, 1])
        with col3:
            with st.expander("**SHOOTING**- shows how aggressive "+ st.session_state.team_name + "'s shooting is vs the quality of their shots"):
                filt_shooting_chart = shooting_chart[shooting_chart['SEASON'] == st.session_state.season]
                filt_shooting_chart = filt_shooting_chart[filt_shooting_chart['COMPETITION_ACRONYM'] == st.session_state.competition]

                fig_shooting = create_FM_team_scatter_chart(filt_shooting_chart, 'SHOOTING', st.session_state.team_name, 
                             'XG PER SHOT', 'SHOTS ON TARGET PER GAME', 0.25, 0.05, 0.2, 2, 8, 
                                                        "Passive Shooting<br>Low-Quality Shooting", 
                                                        "Passive Shooting<br>High-Quality Shooting",
                                                        "Aggressive Shooting<br>Low-Quality Shooting",
                                                        "Aggressive Shooting<br>High-Quality Shooting",
                                                        "red", "orange", "orange", "green")
                st.caption("Gives a comparison of " + st.session_state.team_name + "'s shots on target per game vs xG per shot in the " + 
                            st.session_state.competition + " in the " + season_selected + " season")
                st.plotly_chart(fig_shooting, use_container_width=True)
        
        with col4:
            with st.expander("**SET PIECE FIRST CONTACTS IN OPPOSITION BOX**- Where in the opposition box do "+ st.session_state.team_name + "get the first contact the most when attacking set pieces?"):
                att_set_piece_final_filt = att_set_piece_final[att_set_piece_final['SEASON'] == st.session_state.season]
                att_set_piece_final_filt = att_set_piece_final_filt[att_set_piece_final_filt['COMPETITION_ACRONYM'] == st.session_state.competition]
                att_set_piece_final_filt = att_set_piece_final_filt[att_set_piece_final_filt['TEAM_NAME'] == st.session_state.team_name]

                att_set_piece_chart = att_set_piece_final_filt[['CROSS_END_LOCATION', 'PERC_1ST_CONTACT']].set_index('CROSS_END_LOCATION')

                fig_att_set_piece = create_set_piece_first_contacts_plot(att_set_piece_chart)
                st.caption("Zones are Far Post, Central and Near Post (Left to Right). The higher the percentage, the more the team gets the first contact in that zone.")
                st.pyplot(fig_att_set_piece, use_container_width=True)
        
        col5, col6 = st.columns([1, 1])
        with col5:
            with st.expander("**GOALS FROM SET PIECES**- Comparison of how many goals "+ st.session_state.team_name + " score against how many they concede from Set Pieces"):
                filt_df_goals_set_piece_chart = df_goals_set_piece_chart[df_goals_set_piece_chart['SEASON'] == st.session_state.season]
                filt_df_goals_set_piece_chart = filt_df_goals_set_piece_chart[filt_df_goals_set_piece_chart['COMPETITION_ACRONYM'] == st.session_state.competition]

                fig_goals_set_piece = create_FM_team_scatter_chart(filt_df_goals_set_piece_chart, 'GOALS FROM SET PIECES', st.session_state.team_name, 
                             'SET PIECE GOALS CONCEDED', 'SET PIECE GOALS SCORED', 0.9, 0, 16, 0, 17, 
                                                        "Low no. of set piece goals<br>Low no. of set piece goals conceded", 
                                                        "Low no. of set piece goals<br>High no. of set piece goals conceded",
                                                        "High no. of set piece goals<br>Low no. of set piece goals conceded",
                                                        "High no. of set piece goals<br>High no. of set piece goals conceded",
                                                        "orange", "red", "green", "orange")
                st.caption("Gives a comparison of " + st.session_state.team_name + "'s goals from set pieces vs goals conceded from set pieces in the " + 
                            st.session_state.competition + " in the " + season_selected + " season")
                st.plotly_chart(fig_goals_set_piece, use_container_width=True)
        
        # with col6:
        #     with st.expander("**GOALS FROM SET PIECES**- Comparison of how many goals "+ st.session_state.team_name + " score against how many they concede from Set Pieces"):
        col7, col8, col9 = st.columns([1, 18, 1]) 
        with col8:
            with st.expander("**SHOT MAP**- All shot locations by "+ st.session_state.team_name + " in the " + st.session_state.competition, expanded=True):
                df_shots_last_5_matches_filt = df_shots_last_5_matches[df_shots_last_5_matches['SEASON'] == st.session_state.season]
                df_shots_last_5_matches_filt = df_shots_last_5_matches_filt[df_shots_last_5_matches_filt['COMPETITION_ACRONYM'] == st.session_state.competition]
                df_shots_last_5_matches_filt = df_shots_last_5_matches_filt[df_shots_last_5_matches_filt['TEAM_NAME'] == st.session_state.team_name]
                last5_GWs = df_shots_last_5_matches_filt[['GAMEWEEK']].drop_duplicates().sort_values(by="GAMEWEEK")[-5:]
                # last5_GWs = df_shots_last_5_matches_filt[['GAMEWEEK']].drop_duplicates().sort_values(by="DATE_TIME")[-5:]
                df_shots_last_5_matches_filt = df_shots_last_5_matches_filt.merge(last5_GWs, on="GAMEWEEK")


                df_shots_last_5_matches_filt['norm_start_x'] = df_shots_last_5_matches_filt['START_X'] / 120
                df_shots_last_5_matches_filt['norm_start_y'] = df_shots_last_5_matches_filt['START_Y'] / 80

                max_size = 5
                min_size = 2

                xg_scaled = (df_shots_last_5_matches_filt['XG'] - df_shots_last_5_matches_filt['XG'].min()) / (df_shots_last_5_matches_filt['XG'].max() - df_shots_last_5_matches_filt['XG'].min())
                df_shots_last_5_matches_filt['xG_size'] = xg_scaled * (max_size - min_size) + min_size

                summary_data = df_shots_last_5_matches_filt['OUTCOME'].value_counts().reset_index()
                summary_data.columns = ['Outcome', 'Count']

                total_xg = df_shots_last_5_matches_filt['XG'].sum()
                Goals = df_shots_last_5_matches_filt['OUTCOME'].value_counts()['Goal']
                Attempts = df_shots_last_5_matches_filt.shape[0]
                with_feet = df_shots_last_5_matches_filt['BODY_PART'].str.contains('Foot', na=False).sum()
                with_head = df_shots_last_5_matches_filt['BODY_PART'].value_counts()['Head']
                direct_set_pieces = df_shots_last_5_matches_filt['NOTES'].str.contains('Free kick', na=False).sum()

                fig_shot_map = plot_shot_data(df_shots_last_5_matches_filt, total_xg, Goals, Attempts, with_feet, with_head, direct_set_pieces)
                st.caption("Shot locatios in the last 5 games.")
                st.plotly_chart(fig_shot_map, use_container_width=True, config={'displayModeBar': False, 'doubleClick': 'reset'})
                
    elif st.session_state['active_tab'] == tab_options[5]:
        col1, col2 = st.columns([1, 1])

        df_last_matches = fetch_data(cursor, "SELECT * FROM LAST_MATCHES")
        df_last_matches = df_last_matches[df_last_matches['SEASON'] == st.session_state.season]
        df_last_matches = df_last_matches[df_last_matches['COMPETITION_ACRONYM'] == st.session_state.competition]    
        df_match_oi = df_last_matches[(df_last_matches['HOME_TEAM_NAME'] == st.session_state.team_name) | 
                                      (df_last_matches['AWAY_TEAM_NAME'] == st.session_state.team_name)]  
        match_oi = df_match_oi['MATCH_ID'].iloc[0]

        df_events = fetch_data(cursor_1, f"SELECT * FROM EVENTS_SPADL WHERE MATCH_ID = '{match_oi}'")
        df_player_match = fetch_data(cursor_1, f"SELECT * FROM PLAYER_MATCH WHERE MATCH_ID = '{match_oi}'")

        df_events = df_events.merge(df_player_match[['PLAYER_WS_ID', 'PLAYER_FBREF_NAME', 'MATCH_ID']], on=['PLAYER_WS_ID', 'MATCH_ID'])
        df_events = df_events.merge(team_names[['TEAM_NAME', 'TEAM_FBREF_ID']], on="TEAM_FBREF_ID")

        df_successful_events = df_events[df_events['RESULT_ID'] == 1]

        df_successful_in_play_events = df_successful_events[df_successful_events['TYPE_NAME'].isin(['pass', 
                                                                'dribble', 'clearance', 'cross'])]
        df_successful_set_piece_events = df_successful_events[df_successful_events['TYPE_NAME'].isin(['throw_in', 'freekick_short',
                                                        'corner_short','goalkick', 'corner_crossed', 'freekick_crossed'])]
        
        in_play_xT = xT_generator(df_successful_in_play_events, (16, 12))
        set_piece_xT = xT_generator(df_successful_set_piece_events, (16, 12))
        all_xT = pd.concat([in_play_xT, set_piece_xT])

        df_goals = fetch_data(cursor_1, f"SELECT * FROM SHOT_EVENTS WHERE MATCH_ID = '{match_oi}' AND OUTCOME = 'Goal'")
        df_goals = df_events.merge(df_goals[['MATCH_ID', 'ACTION_ID']], on=['MATCH_ID', 'ACTION_ID'], how='inner')

        df_shots_fbref = fetch_data(cursor_1, f"SELECT * FROM SHOT_EVENTS WHERE MATCH_ID = '{match_oi}'")

        df_events = df_events.sort_values(['PERIOD_ID','NEW_TIME_SECONDS', 'ORIGINAL_EVENT_ID']).reset_index(drop=True)
        df_events.loc[df_events['PERIOD_ID'] == 2, 'NEW_TIME_SECONDS'] = df_events.loc[df_events['PERIOD_ID'] == 2, 'TIME_SECONDS'] + (45 * 60)

        # df_cards = fetch_data(cursor_1, "SELECT * FROM CARDS WHERE MATCH_ID = 1729243")
        # df_subs = fetch_data(cursor_1, "SELECT * FROM SUBS WHERE MATCH_ID = 1729243")

        teamIds = list(df_match_oi[['HOME_TEAM_ID', 'AWAY_TEAM_ID']].iloc[0])

        ws_match_oi = df_match_oi['WS_MATCH_ID'].iloc[0]

        df_cards = fetch_data(cursor_1, f"SELECT * FROM CARDS WHERE WS_MATCH_ID = '{ws_match_oi}'")
        df_subs = fetch_data(cursor_1, f"SELECT * FROM SUBS WHERE WS_MATCH_ID = '{ws_match_oi}'")

        teamid_selected = team_names[team_names['TEAM_NAME'] ==st.session_state.team_name]['TEAM_FBREF_ID'].iloc[0]
        pass_map_dict = rest_dict_pass_map(teamIds, df_events, team_names, df_cards, df_subs)

        max_mins = (df_events['NEW_TIME_SECONDS'].max()//60)+1

        col1, col2 = st.columns(2)

        with col1:
            with st.expander("**MATCH MOMENTUM**- Momentum plot of "+ st.session_state.team_name + "'s last match"):
                fig_match_momentum = plot_match_momentum(all_xT, df_match_oi, df_goals)
                st.pyplot(fig_match_momentum, use_container_width=True)
        
        with col2:
            with st.expander("**PASS MAP**- "+ st.session_state.team_name + "'s Pass Map from their last match"):
                fig_single_passmap = plot_single_passmap(df_player_match, pass_map_dict, df_match_oi, df_events, teamid_selected)
                st.pyplot(fig_single_passmap, use_container_width=True)
        
        col3, col4, col5 = st.columns([1,5,1])

        with col4:
            with st.expander("**PASS MAP COMPARISON**- "+ st.session_state.team_name + "'s and Opponent's Pass Maps compared from the last match"):
                fig_double_passmap = plot_double_passmap(df_player_match, pass_map_dict, df_match_oi, df_events, teamIds)
                st.pyplot(fig_double_passmap, use_container_width=True)

        col6, col7, col8 = st.columns([1, 18, 1]) 

        with col7:
            with st.expander("**SHOT MAP**- All shot locations by "+ st.session_state.team_name + " in their last match"):
                df_shots_last_match_filt = df_shots_last_5_matches[df_shots_last_5_matches['SEASON'] == st.session_state.season]
                df_shots_last_match_filt = df_shots_last_match_filt[df_shots_last_match_filt['COMPETITION_ACRONYM'] == st.session_state.competition]
                df_shots_last_match_filt = df_shots_last_match_filt[df_shots_last_match_filt['TEAM_NAME'] == st.session_state.team_name]
                # df_shots_last_match_filt['DATE_TIME'] = pd.datetime(df_shots_last_match_filt['DATE_TIME'])
                last_GW = df_shots_last_match_filt[['GAMEWEEK']].drop_duplicates().sort_values(by="GAMEWEEK")[-1:]
                # last_GW = df_shots_last_match_filt[['GAMEWEEK']].drop_duplicates().sort_values(by="DATE_TIME")[-1:]
                df_shots_last_match_filt = df_shots_last_match_filt.merge(last_GW, on="GAMEWEEK")

                df_shots_last_match_filt['norm_start_x'] = df_shots_last_match_filt['START_X'] / 120
                df_shots_last_match_filt['norm_start_y'] = df_shots_last_match_filt['START_Y'] / 80

                max_size = 5
                min_size = 2

                xg_scaled = (df_shots_last_match_filt['XG'] - df_shots_last_match_filt['XG'].min()) / (df_shots_last_match_filt['XG'].max() - df_shots_last_match_filt['XG'].min())
                df_shots_last_match_filt['xG_size'] = xg_scaled * (max_size - min_size) + min_size

                summary_data = df_shots_last_match_filt['OUTCOME'].value_counts().reset_index()
                summary_data.columns = ['Outcome', 'Count']

                total_xg = df_shots_last_match_filt['XG'].sum()
                try:
                    Goals = df_shots_last_match_filt['OUTCOME'].value_counts()['Goal']
                except:
                    Goals = 0
                Attempts = df_shots_last_match_filt.shape[0]
                try:
                    with_feet = df_shots_last_match_filt['BODY_PART'].str.contains('Foot', na=False).sum()
                except:
                    with_feet = 0
                try:
                    with_head = df_shots_last_match_filt['BODY_PART'].value_counts()['Head']
                except:
                    with_head = 0
                try:
                    direct_set_pieces = df_shots_last_match_filt['NOTES'].str.contains('Free kick', na=False).sum()
                except:
                    direct_set_pieces = 0

                fig_shot_map = plot_shot_data(df_shots_last_match_filt, total_xg, Goals, Attempts, with_feet, with_head, direct_set_pieces)
                st.caption("Shot locatios in the last 5 games.")
                st.plotly_chart(fig_shot_map, use_container_width=True, config={'displayModeBar': False, 'doubleClick': 'reset'})
            

        col9, col10, col11 = st.columns([1, 8, 1]) 

        with col10:
            with st.expander("**xG MATCH STORY**- "+ st.session_state.team_name + "'s and Opponent's cumulative xG in their last match"):
                xg_vline = (df_events[df_events['PERIOD_ID'] == 1]['NEW_TIME_SECONDS'].max() // 60)+1
                df_shots_ws = df_events[df_events['TYPE_NAME'].str.contains('shot', case=False, na=False)]
                df_shots = df_shots_ws.merge(df_shots_fbref[['MATCH_ID', 'ACTION_ID', 'XG', 'OUTCOME']],on=['MATCH_ID', 'ACTION_ID'],
                                             how='left')
                fig_xg_match_story = plot_xg_match_story(df_shots, df_goals, df_match_oi, xg_vline, max_mins)
                st.pyplot(fig_xg_match_story, use_container_width=True)

else:
    # st.warning('Please press "Load Teams" after changing the Competition or the Season.')
    st.error('Please press "Load Teams" after changing the Competition or the Season')


