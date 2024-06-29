import plotly.graph_objs as go
import pandas as pd
import streamlit as st
import hydralit_components as hc
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import chart_studio.plotly as py
from plotly import tools
from plotly.offline import init_notebook_mode, iplot
import plotly.figure_factory as ff
import plotly.io as pio
import warnings
import base64
warnings.filterwarnings("ignore")


st.set_page_config(layout='wide')
deliveries_path = os.path.join(os.getcwd(), 'pages', 'deliveries.csv')
matches_path = os.path.join(os.getcwd(), 'pages', 'matches.csv')
deliveries_df = pd.read_csv(deliveries_path)
matches_df = pd.read_csv(matches_path)

# 1. How many times has 200 or more been chased in a game ? 
high_scores = deliveries_df.groupby(['match_id', 'inning','batting_team','bowling_team'])['total_runs'].sum().reset_index()
high_scores1 = high_scores[high_scores['inning'] == 1]
high_scores2 = high_scores[high_scores['inning'] == 2]
high_scores1 = high_scores1.merge(high_scores2[['match_id','inning', 'total_runs']], on='match_id')
high_scores1.rename(columns={'inning_x':'inning_1','inning_y':'inning_2','total_runs_x':'inning1_runs','total_runs_y':'inning2_runs'}, inplace=True)
high_scores1 = high_scores1[high_scores1['inning1_runs'] >= 200]
high_scores1['is_score_chased'] = np.where(high_scores1['inning1_runs'] <= high_scores1['inning2_runs'], 'yes', 'no')
# Calculate the percentages
score_chased_counts = high_scores1['is_score_chased'].value_counts(normalize=True) * 100
score_chased_counts = score_chased_counts.reset_index()
score_chased_counts.columns = ['is_score_chased', 'percentage']
fig_200_chased = go.Figure(data=[go.Pie(labels=score_chased_counts['is_score_chased'], 
                             values=score_chased_counts['percentage'], 
                             textinfo='label+percent', 
                             insidetextorientation='radial')])

# Update layout
fig_200_chased.update_layout(title_text='Percentage wise instances of Team batting second who managed to chased 200 or more ?',
                  title_x=0.0001,
                  showlegend=True)
# Preprocess data 
# Define the list of teams and stadiums
teams = ['Chennai Super Kings', 'Punjab Kings', 'Delhi Capitals', 'Royal Challengers Bangalore',
         'Kolkata Knight Riders', 'Rajasthan Royals', 'Mumbai Indians', 'Sunrisers Hyderabad']
venues = {
    "M Chinnaswamy Stadium": "Bangalore",
    "Punjab Cricket Association Stadium": "Mohali",
    "Feroz Shah Kotla": "Delhi",
    "Wankhede Stadium": "Mumbai",
    "Eden Gardens": "Kolkata",
    "Sawai Mansingh Stadium": "Rajasthan",
    "Rajiv Gandhi International Stadium, Uppal": "Hyderabad",
    "MA Chidambaram Stadium, Chepauk": "Chennai",
    "Dr DY Patil Sports Academy": "Mumbai",
    "Brabourne Stadium": "Mumbai",
    "Sardar Patel Stadium, Motera": "Ahmedabad",
    "Brabourne Stadium, Mumbai": "Mumbai",
    "Himachal Pradesh Cricket Association Stadium": "Dharamsala",
    "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium": "Hyderabad",
    "Sharjah Cricket Stadium": "Sharjah",
    "Dubai International Cricket Stadium": "Sharjah",
    "Maharashtra Cricket Association Stadium": "Mumbai",
    "Punjab Cricket Association IS Bindra Stadium, Mohali": "Mohali",
    "M.Chinnaswamy Stadium": "Bangalore",
    "Rajiv Gandhi International Stadium": "Hyderabad",
    "MA Chidambaram Stadium": "Chennai",
    "Arun Jaitley Stadium": "Delhi",
    "MA Chidambaram Stadium, Chepauk, Chennai": "Chennai",
    "Wankhede Stadium, Mumbai": "Mumbai",
    "Narendra Modi Stadium, Ahmedabad": "Ahmedabad",
    "Arun Jaitley Stadium, Delhi": "Delhi",
    "Zayed Cricket Stadium, Abu Dhabi": "Abu Dhabi",
    "Dr DY Patil Sports Academy, Mumbai": "Mumbai",
    "Rajiv Gandhi International Stadium, Uppal, Hyderabad": "Hyderabad",
    "M Chinnaswamy Stadium, Bengaluru": "Bangalore",
    "Eden Gardens, Kolkata": "Kolkata",
    "Sawai Mansingh Stadium, Jaipur": "Rajasthan",
    "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam": "Hyderabad"
}

# Filter matches for the selected teams and venues
filtered_matches = matches_df[
    (matches_df['team1'].isin(teams)) &
    (matches_df['team2'].isin(teams)) &
    (matches_df['venue'].isin(venues.keys())) &
    (matches_df['season'].str[:4].astype(int).between(2008, 2023))
]

# Replace venue names
filtered_matches['venue'] = filtered_matches['venue'].replace(venues)

# Merge deliveries and filtered matches
filtered_deliveries = deliveries_df[deliveries_df['match_id'].isin(filtered_matches['id'])]
# Rename columns to match the desired format
deliveries_renamed = filtered_deliveries.rename(columns={
    'match_id': 'match_id',
    'inning': 'innings',
    'batting_team': 'batting_team',
    'bowling_team': 'bowling_team',
    'over': 'over',
    'ball': 'ball',
    'batter': 'striker',
    'bowler': 'bowler',
    'total_runs': 'runs_of_bat',
    'extra_runs': 'extras',
    'is_wicket': 'wicket_type',
    'player_dismissed': 'player_dismissed',
    'dismissal_kind': 'dismissal_kind',
    'fielder': 'fielder'
})

# Extract relevant columns and add match metadata
match_metadata = filtered_matches[['id', 'season', 'date', 'venue']]
combined_data = deliveries_renamed.merge(match_metadata, left_on='match_id', right_on='id', how='left')

# Select and reorder columns to match the required format
final_data_corrected = combined_data[[
    'match_id', 'season', 'date', 'venue', 'batting_team', 'bowling_team', 'innings', 'over', 'striker', 'bowler',
    'runs_of_bat', 'extras', 'wicket_type', 'player_dismissed', 'dismissal_kind', 'fielder'
]]

# 2. Plotting the runs distribution per over
runs_per_over = final_data_corrected.groupby('over')['runs_of_bat'].sum().reset_index()
fig_runs_per_over = px.line(runs_per_over, x='over', y='runs_of_bat', markers=True, title='Overwise Runs distribution - 3,4,15-18 seems to be most scoring overs')
fig_runs_per_over.update_traces(marker=dict(size=20))
# 3. Plotting the performance of teams at different venues
venue_performance = final_data_corrected.groupby(['venue', 'batting_team'])['runs_of_bat'].sum().reset_index()
fig_venue_performance = px.bar(venue_performance, x='runs_of_bat', y='venue', color='batting_team', barmode='group',
                               title='Team Performance at Different Venues')
# Function to convert image to base64 string
def img_to_base64(img_path):
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')
kkr_img_path = os.path.join(os.getcwd(), 'pages', 'teamlogo', 'kkr.png')
csk_img_path = os.path.join(os.getcwd(), 'pages', 'teamlogo', 'csk.png')
dc_img_path = os.path.join(os.getcwd(), 'pages', 'teamlogo', 'dc.png')
rcb_img_path = os.path.join(os.getcwd(), 'pages', 'teamlogo', 'rcb.png')
rr_img_path = os.path.join(os.getcwd(), 'pages', 'teamlogo', 'rr.png')
pbks_img_path = os.path.join(os.getcwd(), 'pages', 'teamlogo', 'pbks.png')
mi_img_path = os.path.join(os.getcwd(), 'pages', 'teamlogo', 'mi.png')
srh_img_path = os.path.join(os.getcwd(), 'pages', 'teamlogo', 'srh.png')
# Convert images to base64
kkr_img_base64 = img_to_base64(kkr_img_path)
csk_img_base64 = img_to_base64(csk_img_path)
dc_img_base64 = img_to_base64(dc_img_path)
rcb_img_base64 = img_to_base64(rcb_img_path)
rr_img_base64 = img_to_base64(rr_img_path)
pbks_img_base64 = img_to_base64(pbks_img_path)
mi_img_base64 = img_to_base64(mi_img_path)
srh_img_base64 = img_to_base64(srh_img_path)

# 4. Plotting the total runs scored by each team in powerplay 
# powerplay_data = final_data_corrected[final_data_corrected['over'] <= 6]
# powerplay_runs = powerplay_data.groupby('batting_team')['runs_of_bat'].sum().reset_index()
# fig_powerplay_runs = px.bar(powerplay_runs, x='batting_team', y='runs_of_bat', color='batting_team',
#                             title='Total Runs in Powerplay (Overs 1-6) by Each Team')
powerplay_data = final_data_corrected[final_data_corrected['over'] <= 6]
powerplay_runs = powerplay_data.groupby(['match_id', 'batting_team', 'over'])['runs_of_bat'].sum().reset_index()
average_powerplay_runs = powerplay_runs.groupby('batting_team')['runs_of_bat'].mean().reset_index()
fig_powerplay_runs = px.bar(average_powerplay_runs, x='batting_team', y='runs_of_bat', color='batting_team',
                            title='Average Runs per Over in Powerplay (Overs 1-6) by Each Team')
fig_powerplay_runs.add_layout_image(
    dict(
        source=f"data:image/png;base64,{kkr_img_base64}",
        xref="x", yref="y",
        x='Kolkata Knight Riders', y=average_powerplay_runs[average_powerplay_runs['batting_team'] == 'Kolkata Knight Riders']['runs_of_bat'].values[0],
        #x=0.1, y=0.85,  # Adjust x and y for precise positioning
        xanchor="center", yanchor="middle",
        sizex=2.1, sizey=2.1
    )
)
fig_powerplay_runs.add_layout_image(
    dict(
        source=f"data:image/png;base64,{csk_img_base64}",
        xref="x", yref="y",
        x='Chennai Super Kings', y=average_powerplay_runs[average_powerplay_runs['batting_team'] == 'Chennai Super Kings']['runs_of_bat'].values[0],
        #x=0.1, y=0.85,  # Adjust x and y for precise positioning
        xanchor="center", yanchor="middle",
        sizex=1.8, sizey=1.8
    )
)
fig_powerplay_runs.add_layout_image(
    dict(
        source=f"data:image/png;base64,{dc_img_base64}",
        xref="x", yref="y",
        x='Delhi Capitals', y=average_powerplay_runs[average_powerplay_runs['batting_team'] == 'Delhi Capitals']['runs_of_bat'].values[0],
        #x=0.1, y=0.85,  # Adjust x and y for precise positioning
        xanchor="center", yanchor="middle",
        sizex=1.8, sizey=1.8
    )
)
fig_powerplay_runs.add_layout_image(
    dict(
        source=f"data:image/png;base64,{mi_img_base64}",
        xref="x", yref="y",
        x='Mumbai Indians', y=average_powerplay_runs[average_powerplay_runs['batting_team'] == 'Mumbai Indians']['runs_of_bat'].values[0],
        #x=0.1, y=0.85,  # Adjust x and y for precise positioning
        xanchor="center", yanchor="middle",
        sizex=2.1, sizey=2.1
    )
)
fig_powerplay_runs.add_layout_image(
    dict(
        source=f"data:image/png;base64,{pbks_img_base64}",
        xref="x", yref="y",
        x='Punjab Kings', y=average_powerplay_runs[average_powerplay_runs['batting_team'] == 'Punjab Kings']['runs_of_bat'].values[0],
        #x=0.1, y=0.85,  # Adjust x and y for precise positioning
        xanchor="center", yanchor="middle",
        sizex=2.4, sizey=2.4
    )
)
fig_powerplay_runs.add_layout_image(
    dict(
        source=f"data:image/png;base64,{rr_img_base64}",
        xref="x", yref="y",
        x='Rajasthan Royals', y=average_powerplay_runs[average_powerplay_runs['batting_team'] == 'Rajasthan Royals']['runs_of_bat'].values[0],
        #x=0.1, y=0.85,  # Adjust x and y for precise positioning
        xanchor="center", yanchor="middle",
        sizex=2.4, sizey=2.4
    )
)
fig_powerplay_runs.add_layout_image(
    dict(
        source=f"data:image/png;base64,{rr_img_base64}",
        xref="x", yref="y",
        x='Rajasthan Royals', y=average_powerplay_runs[average_powerplay_runs['batting_team'] == 'Rajasthan Royals']['runs_of_bat'].values[0],
        #x=0.1, y=0.85,  # Adjust x and y for precise positioning
        xanchor="center", yanchor="middle",
        sizex=2.4, sizey=2.4
    )
)
fig_powerplay_runs.add_layout_image(
    dict(
        source=f"data:image/png;base64,{rcb_img_base64}",
        xref="x", yref="y",
        x='Royal Challengers Bangalore', y=average_powerplay_runs[average_powerplay_runs['batting_team'] == 'Royal Challengers Bangalore']['runs_of_bat'].values[0],
        #x=0.1, y=0.85,  # Adjust x and y for precise positioning
        xanchor="center", yanchor="middle",
        sizex=2.4, sizey=2.4
    )
)
fig_powerplay_runs.add_layout_image(
    dict(
        source=f"data:image/png;base64,{srh_img_base64}",
        xref="x", yref="y",
        x='Sunrisers Hyderabad', y=average_powerplay_runs[average_powerplay_runs['batting_team'] == 'Sunrisers Hyderabad']['runs_of_bat'].values[0],
        #x=0.1, y=0.85,  # Adjust x and y for precise positioning
        xanchor="center", yanchor="middle",
        sizex=2.4, sizey=2.4
    )
)
# 5. Plotting the total runs scored by each team in middle overs
# middle_overs_data = final_data_corrected[(final_data_corrected['over'] > 6) & (final_data_corrected['over'] <= 15)]
# middle_overs_runs = middle_overs_data.groupby('batting_team')['runs_of_bat'].sum().reset_index()
# fig_middle_overs_runs = px.bar(middle_overs_runs, x='batting_team', y='runs_of_bat', color='batting_team',
#                                title='Total Runs in Middle Overs (Overs 7-15) by Each Team')
middle_overs_data = final_data_corrected[(final_data_corrected['over'] > 6) & (final_data_corrected['over'] <= 15)]
middle_overs_runs = middle_overs_data.groupby(['match_id', 'batting_team', 'over'])['runs_of_bat'].sum().reset_index()
average_middle_overs_runs = middle_overs_runs.groupby('batting_team')['runs_of_bat'].mean().reset_index()
fig_middle_overs_runs = px.bar(average_middle_overs_runs, x='batting_team', y='runs_of_bat', color='batting_team',
                               title='Average Runs per Over in Middle Overs (Overs 7-15) by Each Team')
fig_middle_overs_runs.add_layout_image(
    dict(
        source=f"data:image/png;base64,{kkr_img_base64}",
        xref="x", yref="y",
        x='Kolkata Knight Riders', y=average_middle_overs_runs[average_middle_overs_runs['batting_team'] == 'Kolkata Knight Riders']['runs_of_bat'].values[0],
        #x=0.1, y=0.85,  # Adjust x and y for precise positioning
        xanchor="center", yanchor="middle",
        sizex=2.1, sizey=2.1
    )
)
fig_middle_overs_runs.add_layout_image(
    dict(
        source=f"data:image/png;base64,{csk_img_base64}",
        xref="x", yref="y",
        x='Chennai Super Kings', y=average_middle_overs_runs[average_middle_overs_runs['batting_team'] == 'Chennai Super Kings']['runs_of_bat'].values[0],
        #x=0.1, y=0.85,  # Adjust x and y for precise positioning
        xanchor="center", yanchor="middle",
        sizex=1.8, sizey=1.8
    )
)
fig_middle_overs_runs.add_layout_image(
    dict(
        source=f"data:image/png;base64,{dc_img_base64}",
        xref="x", yref="y",
        x='Delhi Capitals', y=average_middle_overs_runs[average_middle_overs_runs['batting_team'] == 'Delhi Capitals']['runs_of_bat'].values[0],
        #x=0.1, y=0.85,  # Adjust x and y for precise positioning
        xanchor="center", yanchor="middle",
        sizex=1.8, sizey=1.8
    )
)
fig_middle_overs_runs.add_layout_image(
    dict(
        source=f"data:image/png;base64,{mi_img_base64}",
        xref="x", yref="y",
        x='Mumbai Indians', y=average_middle_overs_runs[average_middle_overs_runs['batting_team'] == 'Mumbai Indians']['runs_of_bat'].values[0],
        #x=0.1, y=0.85,  # Adjust x and y for precise positioning
        xanchor="center", yanchor="middle",
        sizex=2.1, sizey=2.1
    )
)
fig_middle_overs_runs.add_layout_image(
    dict(
        source=f"data:image/png;base64,{pbks_img_base64}",
        xref="x", yref="y",
        x='Punjab Kings', y=average_middle_overs_runs[average_middle_overs_runs['batting_team'] == 'Punjab Kings']['runs_of_bat'].values[0],
        #x=0.1, y=0.85,  # Adjust x and y for precise positioning
        xanchor="center", yanchor="middle",
        sizex=2.4, sizey=2.4
    )
)
fig_middle_overs_runs.add_layout_image(
    dict(
        source=f"data:image/png;base64,{rr_img_base64}",
        xref="x", yref="y",
        x='Rajasthan Royals', y=average_middle_overs_runs[average_middle_overs_runs['batting_team'] == 'Rajasthan Royals']['runs_of_bat'].values[0],
        #x=0.1, y=0.85,  # Adjust x and y for precise positioning
        xanchor="center", yanchor="middle",
        sizex=2.4, sizey=2.4
    )
)
fig_middle_overs_runs.add_layout_image(
    dict(
        source=f"data:image/png;base64,{rr_img_base64}",
        xref="x", yref="y",
        x='Rajasthan Royals', y=average_middle_overs_runs[average_middle_overs_runs['batting_team'] == 'Rajasthan Royals']['runs_of_bat'].values[0],
        #x=0.1, y=0.85,  # Adjust x and y for precise positioning
        xanchor="center", yanchor="middle",
        sizex=2.4, sizey=2.4
    )
)
fig_middle_overs_runs.add_layout_image(
    dict(
        source=f"data:image/png;base64,{rcb_img_base64}",
        xref="x", yref="y",
        x='Royal Challengers Bangalore', y=average_middle_overs_runs[average_middle_overs_runs['batting_team'] == 'Royal Challengers Bangalore']['runs_of_bat'].values[0],
        #x=0.1, y=0.85,  # Adjust x and y for precise positioning
        xanchor="center", yanchor="middle",
        sizex=2.4, sizey=2.4
    )
)
fig_middle_overs_runs.add_layout_image(
    dict(
        source=f"data:image/png;base64,{srh_img_base64}",
        xref="x", yref="y",
        x='Sunrisers Hyderabad', y=average_middle_overs_runs[average_middle_overs_runs['batting_team'] == 'Sunrisers Hyderabad']['runs_of_bat'].values[0],
        #x=0.1, y=0.85,  # Adjust x and y for precise positioning
        xanchor="center", yanchor="middle",
        sizex=2.4, sizey=2.4
    )
)
# 6. Plotting the total runs scored by each team in death overs
# death_overs_data = final_data_corrected[final_data_corrected['over'] > 15]
# death_overs_runs = death_overs_data.groupby('batting_team')['runs_of_bat'].sum().reset_index()
# fig_death_overs_runs = px.bar(death_overs_runs, x='batting_team', y='runs_of_bat', color='batting_team',
#                               title='Total Runs in Death Overs (Overs 16-20) by Each Team')
death_overs_data = final_data_corrected[final_data_corrected['over'] > 15]
death_overs_runs = death_overs_data.groupby(['match_id', 'batting_team', 'over'])['runs_of_bat'].sum().reset_index()
average_death_overs_runs = death_overs_runs.groupby('batting_team')['runs_of_bat'].mean().reset_index()
fig_death_overs_runs = px.bar(average_death_overs_runs, x='batting_team', y='runs_of_bat', color='batting_team',
                              title='Average Runs per Over in Death Overs (Overs 16-20) by Each Team')
fig_death_overs_runs.add_layout_image(
    dict(
        source=f"data:image/png;base64,{kkr_img_base64}",
        xref="x", yref="y",
        x='Kolkata Knight Riders', y=average_death_overs_runs[average_death_overs_runs['batting_team'] == 'Kolkata Knight Riders']['runs_of_bat'].values[0],
        #x=0.1, y=0.85,  # Adjust x and y for precise positioning
        xanchor="center", yanchor="middle",
        sizex=2.1, sizey=2.1
    )
)
fig_death_overs_runs.add_layout_image(
    dict(
        source=f"data:image/png;base64,{csk_img_base64}",
        xref="x", yref="y",
        x='Chennai Super Kings', y=average_death_overs_runs[average_death_overs_runs['batting_team'] == 'Chennai Super Kings']['runs_of_bat'].values[0],
        #x=0.1, y=0.85,  # Adjust x and y for precise positioning
        xanchor="center", yanchor="middle",
        sizex=1.8, sizey=1.8
    )
)
fig_death_overs_runs.add_layout_image(
    dict(
        source=f"data:image/png;base64,{dc_img_base64}",
        xref="x", yref="y",
        x='Delhi Capitals', y=average_death_overs_runs[average_death_overs_runs['batting_team'] == 'Delhi Capitals']['runs_of_bat'].values[0],
        #x=0.1, y=0.85,  # Adjust x and y for precise positioning
        xanchor="center", yanchor="middle",
        sizex=1.8, sizey=1.8
    )
)
fig_death_overs_runs.add_layout_image(
    dict(
        source=f"data:image/png;base64,{mi_img_base64}",
        xref="x", yref="y",
        x='Mumbai Indians', y=average_death_overs_runs[average_death_overs_runs['batting_team'] == 'Mumbai Indians']['runs_of_bat'].values[0],
        #x=0.1, y=0.85,  # Adjust x and y for precise positioning
        xanchor="center", yanchor="middle",
        sizex=2.1, sizey=2.1
    )
)
fig_death_overs_runs.add_layout_image(
    dict(
        source=f"data:image/png;base64,{pbks_img_base64}",
        xref="x", yref="y",
        x='Punjab Kings', y=average_death_overs_runs[average_death_overs_runs['batting_team'] == 'Punjab Kings']['runs_of_bat'].values[0],
        #x=0.1, y=0.85,  # Adjust x and y for precise positioning
        xanchor="center", yanchor="middle",
        sizex=2.4, sizey=2.4
    )
)
fig_death_overs_runs.add_layout_image(
    dict(
        source=f"data:image/png;base64,{rr_img_base64}",
        xref="x", yref="y",
        x='Rajasthan Royals', y=average_death_overs_runs[average_death_overs_runs['batting_team'] == 'Rajasthan Royals']['runs_of_bat'].values[0],
        #x=0.1, y=0.85,  # Adjust x and y for precise positioning
        xanchor="center", yanchor="middle",
        sizex=2.4, sizey=2.4
    )
)
fig_death_overs_runs.add_layout_image(
    dict(
        source=f"data:image/png;base64,{rr_img_base64}",
        xref="x", yref="y",
        x='Rajasthan Royals', y=average_death_overs_runs[average_death_overs_runs['batting_team'] == 'Rajasthan Royals']['runs_of_bat'].values[0],
        #x=0.1, y=0.85,  # Adjust x and y for precise positioning
        xanchor="center", yanchor="middle",
        sizex=2.4, sizey=2.4
    )
)
fig_death_overs_runs.add_layout_image(
    dict(
        source=f"data:image/png;base64,{rcb_img_base64}",
        xref="x", yref="y",
        x='Royal Challengers Bangalore', y=average_death_overs_runs[average_death_overs_runs['batting_team'] == 'Royal Challengers Bangalore']['runs_of_bat'].values[0],
        #x=0.1, y=0.85,  # Adjust x and y for precise positioning
        xanchor="center", yanchor="middle",
        sizex=2.4, sizey=2.4
    )
)
fig_death_overs_runs.add_layout_image(
    dict(
        source=f"data:image/png;base64,{srh_img_base64}",
        xref="x", yref="y",
        x='Sunrisers Hyderabad', y=average_death_overs_runs[average_death_overs_runs['batting_team'] == 'Sunrisers Hyderabad']['runs_of_bat'].values[0],
        #x=0.1, y=0.85,  # Adjust x and y for precise positioning
        xanchor="center", yanchor="middle",
        sizex=2.4, sizey=2.4
    )
)
# 7. Venue wise team winning percentage 
file_path = os.path.join(os.getcwd(), 'pages', 'match_result.xlsx')
data = pd.read_excel(file_path)
teams = pd.unique(data[['Team 1', 'Team 2']].values.ravel('K'))
# Create a dataframe to store winning percentages
venue_win_percentage = pd.DataFrame(columns=['Team', 'Venue', 'Winning Percentage'])
#Calculate winning percentage for each team at each venue
for team in teams:
    for venue in data['Ground'].unique():
        total_matches = data[((data['Team 1'] == team) | (data['Team 2'] == team)) & (data['Ground'] == venue)].shape[0]
        won_matches = data[(data['Winner'] == team) & (data['Ground'] == venue)].shape[0]
        if total_matches > 0:
            win_percentage = (won_matches / total_matches) * 100
            venue_win_percentage = pd.concat([venue_win_percentage, pd.DataFrame([{'Team': team, 'Venue': venue, 'Winning Percentage': win_percentage}])], ignore_index=True)
fig_venue_team_win_percent = go.Figure()
for team in teams:
    team_data = venue_win_percentage[venue_win_percentage['Team'] == team]
    fig_venue_team_win_percent.add_trace(go.Bar(
        x=team_data['Venue'],
        y=team_data['Winning Percentage'],
        name=team,
        visible=True
    ))
dropdown_buttons = [
    dict(label='All Teams', method='update', args=[{'visible': [True] * len(teams)}, {'title': 'Winning Percentage of Each Team at Each Venue'}])
]

for team in teams:
    visibility = [team == t for t in teams]
    dropdown_buttons.append(
        dict(label=team, method='update', args=[{'visible': visibility}, {'title': f'Winning Percentage of {team} at Each Venue'}])
    )
fig_venue_team_win_percent.update_layout(
    updatemenus=[dict(active=0, buttons=dropdown_buttons, x=1.05, y=1.15)]
)
fig_venue_team_win_percent.update_layout(
    title='Winning Percentage of Each Team at Each Venue (You may display each team individually using the filter which is currently set to "All Teams")',
    xaxis_title='Venue',
    yaxis_title='Winning Percentage (%)',
    barmode='group'
)

# Define the primary menu definition
menu_data = [
    {'id': 'Performance breakdown by venue and wins', 'icon': "fas fa-chart-pie", 'label': "Performance breakdown by venue and wins"},
    {'id': 'Team wise scoring breakdown in game', 'icon': "fas fa-chart-area", 'label': "Team wise scoring breakdown in game"},
]
over_theme = {'txc_inactive': '#FFFFFF'}
menu_id = hc.nav_bar(
    menu_definition=menu_data,
    override_theme=over_theme,
    home_name=None,
    hide_streamlit_markers=True,
    sticky_nav=False,
    sticky_mode='pinned',
)

if st.sidebar.button('click me too'):
    st.info('You clicked at: {}'.format(datetime.datetime.now()))
st.info("PLEASE SCROLL DOWN TO SEE MORE ANALYTICAL INSIGHTS")

if menu_id == 'Performance breakdown by venue and wins':
    st.plotly_chart(fig_venue_team_win_percent, use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_200_chased, use_container_width=True)
    with col2:
        st.plotly_chart(fig_runs_per_over, use_container_width=True)
elif menu_id == 'Team wise scoring breakdown in game':   
    st.plotly_chart(fig_powerplay_runs, use_container_width=True)
    st.plotly_chart(fig_middle_overs_runs, use_container_width=True)
    st.plotly_chart(fig_death_overs_runs, use_container_width=True)
 