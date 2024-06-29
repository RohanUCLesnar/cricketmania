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
import warnings
import base64
warnings.filterwarnings("ignore")

st.set_page_config(layout='wide')
#Batter Analysis 
# Load data
file_path_deliveries = os.path.join(os.getcwd(), 'pages', 'deliveries.csv')
file_path_matches = os.path.join(os.getcwd(), 'pages', 'matches.csv')
matches = pd.read_csv(file_path_matches)
deliveries = pd.read_csv(file_path_deliveries)

# Define the team names for replacement
x = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore', 'Kolkata Knight Riders', 
     'Delhi Daredevils', 'Delhi Capitals', 'Kings XI Punjab', 'Chennai Super Kings', 'Rajasthan Royals', 'Deccan Chargers']
y = ['SRH', 'MI', 'RCB', 'KKR', 'DC', 'DC', 'KXIP', 'CSK', 'RR', 'SRH']

matches.replace(x, y, inplace=True)
deliveries.replace(x, y, inplace=True)

# Ensure only relevant teams are present
matches = matches[matches['team1'].isin(y) & matches['team2'].isin(y)]
deliveries = deliveries[deliveries['batting_team'].isin(y) & deliveries['bowling_team'].isin(y)]

# Add a 'season' column to the matches dataset
matches['season'] = pd.to_datetime(matches['date']).dt.year

# Calculate the number of matches played by each player
matches_played_by_player = deliveries.groupby('batter')['match_id'].nunique().reset_index()
matches_played_by_player.columns = ['batter', 'matches_played']

# Filter players with at least 50 matches played
min_matches = 50
players_with_min_matches = matches_played_by_player[matches_played_by_player['matches_played'] >= min_matches]

# Calculate total runs and dismissals
batting_stats = deliveries.groupby('batter').agg({'batsman_runs': 'sum', 'ball': 'count', 'is_wicket': 'sum'}).reset_index()

# Merge with the number of matches played
batting_stats = batting_stats.merge(players_with_min_matches, on='batter')

# Calculate batting average and strike rate
batting_stats['batting_average'] = batting_stats['batsman_runs'] / batting_stats['is_wicket']
batting_stats['strike_rate'] = (batting_stats['batsman_runs'] / batting_stats['ball']) * 100

# Calculate the number of sixes for each batter
sixes = deliveries[deliveries['batsman_runs'] == 6].groupby('batter').size().reset_index(name='sixes')

# Merge sixes with the batting stats DataFrame
batting_stats = batting_stats.merge(sixes, on='batter', how='left').fillna(0)

# Define the primary menu definition
menu_data = [
    {'id': 'Analytical overview of Batsman Performance', 'icon': "fas fa-chart-pie", 'label': "Analytical overview of Batsman Performance"},
    {'id': 'Analytical overview of Bowler Performance', 'icon': "fas fa-chart-area", 'label': "Analytical overview of Bowler Performance"},
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
# Add images for specific players
def load_image_base64(image_path):
    with open(image_path, 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return f'data:image/png;base64,{encoded_string}'

# Convert images to base64
abd_base64 = load_image_base64(os.path.join(os.getcwd(), 'pages', 'batter', 'ABD.png'))
jpd_base64 = load_image_base64(os.path.join(os.getcwd(), 'pages', 'batter', 'jpd.png'))
msd_base64 = load_image_base64(os.path.join(os.getcwd(), 'pages', 'batter', 'msd.png'))
adruss_base64 = load_image_base64(os.path.join(os.getcwd(), 'pages', 'batter', 'adruss.png'))
hpand_base64 = load_image_base64(os.path.join(os.getcwd(), 'pages', 'batter', 'hpand.png'))
maxi_base64 = load_image_base64(os.path.join(os.getcwd(), 'pages', 'batter', 'maxi.png'))
pollard_base64 = load_image_base64(os.path.join(os.getcwd(), 'pages', 'batter', 'pollard.png'))
pshaw_base64 = load_image_base64(os.path.join(os.getcwd(), 'pages', 'batter', 'pshaw.png'))
rpant_base64 = load_image_base64(os.path.join(os.getcwd(), 'pages', 'batter', 'rpant.png'))
snarine_base64 = load_image_base64(os.path.join(os.getcwd(), 'pages', 'batter', 'snarine.png'))
viru_base64 = load_image_base64(os.path.join(os.getcwd(), 'pages', 'batter', 'viru.png'))
warner_image_base64 = load_image_base64(os.path.join(os.getcwd(), 'pages', 'batter', 'dwarner.png'))
gayle_image_base64 = load_image_base64(os.path.join(os.getcwd(), 'pages', 'batter', 'gayle.png'))
se_marsh = load_image_base64(os.path.join(os.getcwd(), 'pages', 'batter', 'smarsh.png'))
ml_hayden = load_image_base64(os.path.join(os.getcwd(), 'pages', 'batter', 'hayden.png'))
sr_tendulkar = load_image_base64(os.path.join(os.getcwd(), 'pages', 'batter', 'tendulkar.png'))
mek_hussey  = load_image_base64(os.path.join(os.getcwd(), 'pages', 'batter', 'mhuss.png'))
rv_uthappa = load_image_base64(os.path.join(os.getcwd(), 'pages', 'batter', 'robin.png'))
ks_williamson = load_image_base64(os.path.join(os.getcwd(), 'pages', 'batter', 'kane.png'))
kl_rahul = load_image_base64(os.path.join(os.getcwd(), 'pages', 'batter', 'kl.png'))
rd_gaikwad = load_image_base64(os.path.join(os.getcwd(), 'pages', 'batter', 'rutug.png'))
jc_buttler = load_image_base64(os.path.join(os.getcwd(), 'pages', 'batter', 'jb.png'))
ybk_jaiswal = load_image_base64(os.path.join(os.getcwd(), 'pages', 'batter', 'jaiswal.png'))
ali = load_image_base64(os.path.join(os.getcwd(), 'pages', 'bowler', 'ali.png'))
ashwin = load_image_base64(os.path.join(os.getcwd(), 'pages', 'bowler', 'ash.png'))
bhaji = load_image_base64(os.path.join(os.getcwd(), 'pages', 'bowler', 'bhaji.png'))
dayle = load_image_base64(os.path.join(os.getcwd(), 'pages', 'bowler', 'dayle.png'))
malinga = load_image_base64(os.path.join(os.getcwd(), 'pages', 'bowler', 'malinga.png'))
mk = load_image_base64(os.path.join(os.getcwd(), 'pages', 'bowler', 'mk.png'))
murali =  load_image_base64(os.path.join(os.getcwd(), 'pages', 'bowler', 'murali.png'))
rashid = load_image_base64(os.path.join(os.getcwd(), 'pages', 'bowler', 'rashid.png'))
warne = load_image_base64(os.path.join(os.getcwd(), 'pages', 'bowler', 'warne.png'))
snarine = load_image_base64(os.path.join(os.getcwd(), 'pages', 'bowler', 'snarine.png'))
andreruss = load_image_base64(os.path.join(os.getcwd(), 'pages', 'bowler', 'adruss.png'))
bravo = load_image_base64(os.path.join(os.getcwd(), 'pages', 'bowler', 'bravo.png'))
hodge = load_image_base64(os.path.join(os.getcwd(), 'pages', 'bowler', 'hodge.png'))
hvpatel = load_image_base64(os.path.join(os.getcwd(), 'pages', 'bowler', 'hvpatel.png'))
kk = load_image_base64(os.path.join(os.getcwd(), 'pages', 'bowler', 'kk.png'))
mmsharma = load_image_base64(os.path.join(os.getcwd(), 'pages', 'bowler', 'mmsharma.png'))
rabada = load_image_base64(os.path.join(os.getcwd(), 'pages', 'bowler', 'rabada.png'))
tahir = load_image_base64(os.path.join(os.getcwd(), 'pages', 'bowler', 'tahir.png'))
yuzi = load_image_base64(os.path.join(os.getcwd(), 'pages', 'bowler', 'yuzi.png'))
Sohail_Tanvir = load_image_base64(os.path.join(os.getcwd(), 'pages', 'bowler', 'stanvir.png'))
RP_Singh = load_image_base64(os.path.join(os.getcwd(), 'pages', 'bowler', 'rpsingh.png'))
Pragyan_Ojha = load_image_base64(os.path.join(os.getcwd(), 'pages', 'bowler', 'ojha.png'))
Morne_Morkel = load_image_base64(os.path.join(os.getcwd(), 'pages', 'bowler', 'morkel.png'))
Bhuvneshwar_Kumar = load_image_base64(os.path.join(os.getcwd(), 'pages', 'bowler', 'bkumar.png'))
Andrew_Tye = load_image_base64(os.path.join(os.getcwd(), 'pages', 'bowler', 'atye.png'))
Mohammed_Shami = load_image_base64(os.path.join(os.getcwd(), 'pages', 'bowler', 'shami.png'))

if st.sidebar.button('click me too'):
    st.info('You clicked at: {}'.format(datetime.datetime.now()))
st.info("IMPORTANT: Scroll to see all charts, Click Full Screen on Bar charts for added visuals, & Hover over for more details")

# Get top 10 batters by batting average
top_10_batters_avg = batting_stats.sort_values(by='batting_average', ascending=False).head(10)
fig1 = go.Figure()
fig1.add_trace(go.Bar(
    x=top_10_batters_avg['batter'],
    y=top_10_batters_avg['batting_average'],
    text=[f"Average: {avg:.2f}<br>Runs: {runs}<br>Matches: {matches}" for avg, runs, matches in 
        zip(top_10_batters_avg['batting_average'], top_10_batters_avg['batsman_runs'], top_10_batters_avg['matches_played'])],
    hoverinfo='text',
    marker=dict(color=top_10_batters_avg['batting_average'], colorscale='Viridis')
))

for index, row in top_10_batters_avg.iterrows():
    if row['batter'] == 'AB de Villiers':
        fig1.add_layout_image(
            dict(
                source=abd_base64,
                x=row['batter'],
                xanchor="center",
                y=row['batting_average'] / 2,  # Position image in the middle of the bar
                sizex=0.8,                     # Adjust the size as needed
                sizey=15,                      # Adjust the size as needed
                xref="x",
                yref="y",
                opacity=1,
                layer="above"
            )
        )
    elif row['batter'] == 'JP Duminy':
        fig1.add_layout_image(
            dict(
                source=jpd_base64,
                x=row['batter'],
                xanchor="center",
                y=row['batting_average'] / 2,  # Position image in the middle of the bar
                sizex=0.8,                     # Adjust the size as needed
                sizey=20,                      # Adjust the size as needed
                xref="x",
                yref="y",
                opacity=1,
                layer="above"
            )
        )
    elif row['batter'] == 'KL Rahul':
        fig1.add_layout_image(
            dict(
                source=kl_rahul,
                x=row['batter'],
                xanchor="center",
                y=row['batting_average'] / 2,  # Position image in the middle of the bar
                sizex=0.8,                     # Adjust the size as needed
                sizey=20,                      # Adjust the size as needed
                xref="x",
                yref="y",
                opacity=1,
                layer="above"
            )
        )
    elif row['batter'] == 'KS Williamson':
        fig1.add_layout_image(
            dict(
                source=ks_williamson,
                x=row['batter'],
                xanchor="center",
                y=row['batting_average'] / 2,  # Position image in the middle of the bar
                sizex=0.8,                     # Adjust the size as needed
                sizey=20,                      # Adjust the size as needed
                xref="x",
                yref="y",
                opacity=1,
                layer="above"
            )
        )
    elif row['batter'] == 'JC Buttler':
        fig1.add_layout_image(
            dict(
                source=jc_buttler,
                x=row['batter'],
                xanchor="center",
                y=row['batting_average'] / 2,  # Position image in the middle of the bar
                sizex=0.8,                     # Adjust the size as needed
                sizey=20,                      # Adjust the size as needed
                xref="x",
                yref="y",
                opacity=1,
                layer="above"
            )
        )
    elif row['batter'] == 'DA Warner':
        fig1.add_layout_image(
            dict(
                source=warner_image_base64,
                x=row['batter'],
                xanchor="center",
                y=row['batting_average'] / 2,  # Position image in the middle of the bar
                sizex=0.8,                     # Adjust the size as needed
                sizey=20,                      # Adjust the size as needed
                xref="x",
                yref="y",
                opacity=1,
                layer="above"
            )
        )
    elif row['batter'] == 'MEK Hussey':
        fig1.add_layout_image(
            dict(
                source=mek_hussey,
                x=row['batter'],
                xanchor="center",
                y=row['batting_average'] / 2,  # Position image in the middle of the bar
                sizex=0.8,                     # Adjust the size as needed
                sizey=20,                      # Adjust the size as needed
                xref="x",
                yref="y",
                opacity=1,
                layer="above"
            )
        )
    elif row['batter'] == 'CH Gayle':
        fig1.add_layout_image(
            dict(
                source=gayle_image_base64,
                x=row['batter'],
                xanchor="center",
                y=row['batting_average'] / 2,  # Position image in the middle of the bar
                sizex=0.8,                     # Adjust the size as needed
                sizey=20,                      # Adjust the size as needed
                xref="x",
                yref="y",
                opacity=1,
                layer="above"
            )
        )
    elif row['batter'] == 'SE Marsh':
        fig1.add_layout_image(
            dict(
                source=se_marsh,
                x=row['batter'],
                xanchor="center",
                y=row['batting_average'] / 2,  # Position image in the middle of the bar
                sizex=0.8,                     # Adjust the size as needed
                sizey=20,                      # Adjust the size as needed
                xref="x",
                yref="y",
                opacity=1,
                layer="above"
            )
        )
    elif row['batter'] == 'MS Dhoni':
        fig1.add_layout_image(
            dict(
                source=msd_base64,
                x=row['batter'],
                xanchor="center",
                y=row['batting_average'] / 2,  # Position image in the middle of the bar
                sizex=0.8,                     # Adjust the size as needed
                sizey=20,                      # Adjust the size as needed
                xref="x",
                yref="y",
                opacity=1,
                layer="above"
            )
        )

fig1.update_layout(
    title="Top 10 Batters by Batting Average",
    xaxis_title="Batter",
    yaxis_title="Batting Average",
    showlegend=False
)

# Get top 10 batters by 30s, 50s, and 100s
unique_batters = deliveries['batter'].unique().tolist()
filtered_deliveries = deliveries[deliveries['batter'].isin(unique_batters)]
batter_match_scores = filtered_deliveries.groupby(['batter', 'match_id'])['batsman_runs'].sum().reset_index()
scores_30_above = batter_match_scores[batter_match_scores['batsman_runs'] >= 30].groupby('batter').size().reset_index(name='30s')
scores_50_above = batter_match_scores[batter_match_scores['batsman_runs'] >= 50].groupby('batter').size().reset_index(name='50s')
scores_100_above = batter_match_scores[batter_match_scores['batsman_runs'] >= 100].groupby('batter').size().reset_index(name='100s')
score_counts_df = scores_30_above.merge(scores_50_above, on='batter', how='left').merge(scores_100_above, on='batter', how='left').fillna(0)
top_5_batters = pd.concat([score_counts_df.sort_values(by=col, ascending=False).head(5) for col in ['100s', '50s', '30s']]).drop_duplicates().reset_index(drop=True)
melted_df = top_5_batters.melt(id_vars='batter', value_vars=['30s', '50s', '100s'], var_name='milestone', value_name='frequency')

fig2 = go.Figure()
sizeref = 2.*max(melted_df['frequency'])/(100.**2)
for milestone in ['30s', '50s', '100s']:
    filtered_df = melted_df[melted_df['milestone'] == milestone]
    fig2.add_trace(go.Scatter(
        x=filtered_df['batter'],
        y=filtered_df['milestone'].map({'30s': '30s', '50s': '50s', '100s': '100s'}),
        mode='markers',
        marker=dict(size=filtered_df['frequency'], 
                    sizemode='area', 
                    sizeref=sizeref, 
                    sizemin=5),
        name=milestone,
        text=[f"Batter: {row['batter']}<br>Frequency: {row['frequency']}" for index, row in filtered_df.iterrows()],
        hoverinfo='text'
    ))
fig2.update_layout(
    title="Top Batters by Innings Score Milestones",
    xaxis_title="Batter",
    yaxis_title="Innings Score Milestone",
    showlegend=True,
    template="plotly_dark"
)

# Get top 10 batters by strike rate
print(snarine_base64[:100])
top_10_batters_sr = batting_stats.sort_values(by='strike_rate', ascending=False).head(10)
fig3 = go.Figure()
fig3.add_trace(go.Bar(
    x=top_10_batters_sr['batter'],
    y=top_10_batters_sr['strike_rate'],
    text=[f"S/R: {avg:.2f}<br>Runs: {runs}<br>Matches: {matches}" for avg, runs, matches in 
        zip(top_10_batters_sr['strike_rate'], top_10_batters_sr['batsman_runs'], top_10_batters_sr['matches_played'])],
    hoverinfo='text',
    marker=dict(color=top_10_batters_sr['strike_rate'], colorscale='Viridis')
))
for index, row in top_10_batters_sr.iterrows():
    if row['batter'] == 'AD Russell':
        fig3.add_layout_image(
            dict(
                source=adruss_base64,
                x=row['batter'],
                xanchor="center",
                y=row['strike_rate'] / 2,  # Position image in the middle of the bar
                sizex=0.8,                     # Adjust the size as needed
                sizey=15,                      # Adjust the size as needed
                xref="x",
                yref="y",
                opacity=1,
                layer="above"
            )
        )
    elif row['batter'] == 'SP Narine':
        fig3.add_layout_image(
            dict(
                source=snarine_base64,
                x=row['batter'],
                xanchor="center",
                y=row['strike_rate'] / 2,  # Position image in the middle of the bar
                sizex=0.8,                     # Adjust the size as needed
                sizey=20,                      # Adjust the size as needed
                xref="x",
                yref="y",
                opacity=1,
                layer="above"
            )
        )
    elif row['batter'] == 'GJ Maxwell':
        fig3.add_layout_image(
            dict(
                source=maxi_base64,
                x=row['batter'],
                xanchor="center",
                y=row['strike_rate'] / 2,  # Position image in the middle of the bar
                sizex=0.8,                     # Adjust the size as needed
                sizey=20,                      # Adjust the size as needed
                xref="x",
                yref="y",
                opacity=1,
                layer="above"
            )
        )
    elif row['batter'] == 'V Sehwag':
        fig3.add_layout_image(
            dict(
                source=viru_base64,
                x=row['batter'],
                xanchor="center",
                y=row['strike_rate'] / 2,  # Position image in the middle of the bar
                sizex=0.8,                     # Adjust the size as needed
                sizey=20,                      # Adjust the size as needed
                xref="x",
                yref="y",
                opacity=1,
                layer="above"
            )
        )
    elif row['batter'] == 'AB de Villiers':
        fig3.add_layout_image(
            dict(
                source=abd_base64,
                x=row['batter'],
                xanchor="center",
                y=row['strike_rate'] / 2,  # Position image in the middle of the bar
                sizex=0.8,                     # Adjust the size as needed
                sizey=20,                      # Adjust the size as needed
                xref="x",
                yref="y",
                opacity=1,
                layer="above"
            )
        )
    elif row['batter'] == 'HH Pandya':
        fig3.add_layout_image(
            dict(
                source=hpand_base64,
                x=row['batter'],
                xanchor="center",
                y=row['strike_rate'] / 2,  # Position image in the middle of the bar
                sizex=0.8,                     # Adjust the size as needed
                sizey=20,                      # Adjust the size as needed
                xref="x",
                yref="y",
                opacity=1,
                layer="above"
            )
        )
    elif row['batter'] == 'JC Buttler':
        fig3.add_layout_image(
            dict(
                source=jc_buttler,
                x=row['batter'],
                xanchor="center",
                y=row['strike_rate'] / 2,  # Position image in the middle of the bar
                sizex=0.8,                     # Adjust the size as needed
                sizey=20,                      # Adjust the size as needed
                xref="x",
                yref="y",
                opacity=1,
                layer="above"
            )
        )
    elif row['batter'] == 'KA Pollard':
        fig3.add_layout_image(
            dict(
                source=pollard_base64,
                x=row['batter'],
                xanchor="center",
                y=row['strike_rate'] / 2,  # Position image in the middle of the bar
                sizex=0.8,                     # Adjust the size as needed
                sizey=20,                      # Adjust the size as needed
                xref="x",
                yref="y",
                opacity=1,
                layer="above"
            )
        )
    elif row['batter'] == 'RR Pant':
        fig3.add_layout_image(
            dict(
                source=rpant_base64,
                x=row['batter'],
                xanchor="center",
                y=row['strike_rate'] / 2,  # Position image in the middle of the bar
                sizex=0.8,                     # Adjust the size as needed
                sizey=20,                      # Adjust the size as needed
                xref="x",
                yref="y",
                opacity=1,
                layer="above"
            )
        )
    elif row['batter'] == 'PP Shaw':
        fig3.add_layout_image(
            dict(
                source=pshaw_base64,
                x=row['batter'],
                xanchor="center",
                y=row['strike_rate'] / 2,  # Position image in the middle of the bar
                sizex=0.8,                     # Adjust the size as needed
                sizey=20,                      # Adjust the size as needed
                xref="x",
                yref="y",
                opacity=1,
                layer="above"
            )
        )

fig3.update_layout(
    title="Top 10 Batters by Batting Strike Rate",
    xaxis_title="Batter",
    yaxis_title="Batting S/R",
    showlegend=False
)

# Top 20 batters bubble plot
top_20_batters = batting_stats.sort_values(by='batsman_runs', ascending=False).head(20)
fig4 = go.Figure(data=[go.Scatter(
    x=top_20_batters['batter'],
    y=top_20_batters['matches_played'],
    mode='markers',
    marker=dict(
        size=top_20_batters['sixes'],
        color=top_20_batters['batsman_runs'],
        showscale=True,
        sizeref=2.*max(top_20_batters['sixes'])/(110.**2),
        sizemode='area'
    ),
    text=top_20_batters.apply(lambda row: f"Batter: {row['batter']}<br>Runs: {row['batsman_runs']}<br>Sixes: {row['sixes']}<br>Matches: {row['matches_played']}", axis=1),
    hoverinfo='text'
)])
fig4.update_layout(
    title="Top 20 Batters Bubble Plot (2008-2023)",
    xaxis_title="Batter",
    yaxis_title="Matches Played"
)

# Top 10 scoring types by batter
scoring_types = deliveries.groupby(['batter', 'batsman_runs']).size().unstack(fill_value=0).reset_index()
top_10_scoring_types = scoring_types.sort_values(by=6, ascending=False).head(10)

top_10_scoring_types = pd.DataFrame({
'batter': ['CH Gayle', 'RG Sharma', 'AB de Villiers', 'DA Warner', 'KA Pollard', 
        'MS Dhoni', 'V Kohli', 'AD Russell', 'SK Raina', 'SR Watson'],
'0': [1437, 1739, 1008, 1592, 871, 1137, 1715, 545, 1224, 1169],
'1': [888, 1743, 1263, 1464, 775, 1280, 1969, 365, 1336, 865],
'2': [83, 237, 255, 314, 145, 287, 344, 49, 217, 127],
'3': [3, 6, 16, 23, 8, 12, 17, 0, 10, 8],
'4': [349, 512, 376, 561, 205, 302, 535, 148, 409, 349],
'6': [303, 240, 213, 209, 209, 201, 194, 176, 175, 174]})

# Melt the dataframe for easier plotting
melted_df = top_10_scoring_types.melt(id_vars='batter', value_vars=['1', '2', '3', '4', '6'], 
                                    var_name='run_type', value_name='frequency')
fig5 = go.Figure()
# Calculate the sizeref value
max_size = max(melted_df['frequency'])
sizeref = 2.*max_size/(100.**2)
run_name = ['Single - 1', 'Double - 2', 'Triple - 3', 'Boundary - 4', 'Sixer - 6']
k=0
for run_type in ['1', '2', '3', '4', '6']:
    filtered_df = melted_df[melted_df['run_type'] == run_type]
    fig5.add_trace(go.Scatter(
        x=filtered_df['batter'],
        y=filtered_df['run_type'],
        mode='markers',
        marker=dict(size=filtered_df['frequency'], 
                    sizemode='area', 
                    sizeref=sizeref, 
                    sizemin=4),
        name=f"{run_name[k]}",
        text=[f"Batter: {row['batter']}<br>Frequency: {row['frequency']}" for index, row in filtered_df.iterrows()],
        hoverinfo='text'
    ))
    k += 1
fig5.update_layout(
    title="Top 10 Scoring Types by Batter",
    xaxis_title="Batter",
    yaxis_title="Run Type",
    showlegend=True,
    template="plotly_dark"
)

# Orange Cap winners
deliveries_with_season = deliveries.merge(matches[['id', 'season']], left_on='match_id', right_on='id')
season_runs = deliveries_with_season.groupby(['season', 'batter'])['batsman_runs'].sum().reset_index()
orange_cap_winners = season_runs.loc[season_runs.groupby('season')['batsman_runs'].idxmax()].reset_index(drop=True)
orange_cap_winners.columns = ['season', 'batter', 'runs']
fig6 = go.Figure(data=[go.Bar(
    x=orange_cap_winners['season'],
    y=orange_cap_winners['runs'],
    text=orange_cap_winners.apply(lambda row: f"{row['batter']} - {row['runs']} runs", axis=1),
    hoverinfo='text',
    marker=dict(color='orange')
)])

for index, row in orange_cap_winners.iterrows():
    if row['batter'] == 'DA Warner':
        fig6.add_layout_image(
            dict(
                source=warner_image_base64,
                x=row['season'] - 0.5,  # Adjust the x position slightly to center the image
                y=row['runs'] + 80,     # Adjust the y position as needed
                sizex=1.2,              # Adjust the size as needed
                sizey=180,              # Adjust the size as needed
                xref="x",
                yref="y",
                opacity=1,
                layer="above"
            )
        )
    elif row['batter'] == 'CH Gayle':
        fig6.add_layout_image(
            dict(
                source=gayle_image_base64,
                x=row['season'] - 0.4,  # Adjust the x position slightly to center the image
                y=row['runs'] + 90,     # Adjust the y position as needed
                sizex=0.8,              # Adjust the size as needed
                sizey=120,              # Adjust the size as needed
                xref="x",
                yref="y",
                opacity=1,
                layer="above"
            )
        )
    elif row['batter'] == 'SE Marsh':
        fig6.add_layout_image(
            dict(
                source=se_marsh,
                x=row['season'] - 0.4,  # Adjust the x position slightly to center the image
                y=row['runs'] + 50,     # Adjust the y position as needed
                sizex=0.8,              # Adjust the size as needed
                sizey=100,              # Adjust the size as needed
                xref="x",
                yref="y",
                opacity=1,
                layer="above"
            )
        )
    elif row['batter'] == 'ML Hayden':
        fig6.add_layout_image(
            dict(
                source=ml_hayden,
                x=row['season'] - 0.4,  # Adjust the x position slightly to center the image
                y=row['runs'] + 70,     # Adjust the y position as needed
                sizex=0.9,              # Adjust the size as needed
                sizey=140,              # Adjust the size as needed
                xref="x",
                yref="y",
                opacity=1,
                layer="above"
            )
        )
    elif row['batter'] == 'SR Tendulkar':
        fig6.add_layout_image(
            dict(
                source=sr_tendulkar,
                x=row['season'] - 0.4,  # Adjust the x position slightly to center the image
                y=row['runs'] + 110,     # Adjust the y position as needed
                sizex=0.8,              # Adjust the size as needed
                sizey=100,              # Adjust the size as needed
                xref="x",
                yref="y",
                opacity=1,
                layer="above"
            )
        )
    elif row['batter'] == 'MEK Hussey':
        fig6.add_layout_image(
            dict(
                source=mek_hussey,
                x=row['season'] - 0.4,  # Adjust the x position slightly to center the image
                y=row['runs'] -130 ,     # Adjust the y position as needed
                sizex=0.99,              # Adjust the size as needed
                sizey=120,              # Adjust the size as needed
                xref="x",
                yref="y",
                opacity=1,
                layer="above"
            )
        )
    elif row['batter'] == 'RV Uthappa':
        fig6.add_layout_image(
            dict(
                source=rv_uthappa,
                x=row['season'] - 0.4,  # Adjust the x position slightly to center the image
                y=row['runs'] + 110,     # Adjust the y position as needed
                sizex=0.8,              # Adjust the size as needed
                sizey=100,              # Adjust the size as needed
                xref="x",
                yref="y",
                opacity=1,
                layer="above"
            )
        )
    elif row['batter'] == 'KS Williamson':
        fig6.add_layout_image(
            dict(
                source=ks_williamson,
                x=row['season'] - 0.4,  # Adjust the x position slightly to center the image
                y=row['runs'] - 140,     # Adjust the y position as needed
                sizex=1.59,              # Adjust the size as needed
                sizey=110,              # Adjust the size as needed
                xref="x",
                yref="y",
                opacity=1,
                layer="above"
            )
        )
    elif row['batter'] == 'KL Rahul':
        fig6.add_layout_image(
            dict(
                source=kl_rahul,
                x=row['season'] - 0.4,  # Adjust the x position slightly to center the image
                y=row['runs'] + 110,     # Adjust the y position as needed
                sizex=0.8,              # Adjust the size as needed
                sizey=100,              # Adjust the size as needed
                xref="x",
                yref="y",
                opacity=1,
                layer="above"
            )
        )
    elif row['batter'] == 'RD Gaikwad':
        fig6.add_layout_image(
            dict(
                source=rd_gaikwad,
                x=row['season'] - 0.4,  # Adjust the x position slightly to center the image
                y=row['runs'] + 110,     # Adjust the y position as needed
                sizex=0.8,              # Adjust the size as needed
                sizey=100,              # Adjust the size as needed
                xref="x",
                yref="y",
                opacity=1,
                layer="above"
            )
        )
    elif row['batter'] == 'JC Buttler':
        fig6.add_layout_image(
            dict(
                source=jc_buttler,
                x=row['season'] - 0.4,  # Adjust the x position slightly to center the image
                y=row['runs'] + 110,     # Adjust the y position as needed
                sizex=0.8,              # Adjust the size as needed
                sizey=100,              # Adjust the size as needed
                xref="x",
                yref="y",
                opacity=1,
                layer="above"
            )
        )
    elif row['batter'] == 'YBK Jaiswal':
        fig6.add_layout_image(
            dict(
                source=ybk_jaiswal,
                x=row['season'] - 0.4,  # Adjust the x position slightly to center the image
                y=row['runs'] + 110,     # Adjust the y position as needed
                sizex=0.8,              # Adjust the size as needed
                sizey=100,              # Adjust the size as needed
                xref="x",
                yref="y",
                opacity=1,
                layer="above"
            )
        )


fig6.update_layout(
    title="Orange Cap Winners (2008-2023)",
    xaxis_title="Year",
    yaxis_title="Runs",
    xaxis=dict(tickmode='linear')
)

#Bowler Analysis 
# Load the Excel file
file_path = os.path.join(os.getcwd(), 'pages', 'bowler.xlsx')
xls = pd.ExcelFile(file_path)
# Read each sheet into a separate DataFrame with the correct names
wicket_hauls = pd.read_excel(xls, sheet_name='4_5_haul')
worst_bowlers = pd.read_excel(xls, sheet_name='most_runs')
bowler_sr = pd.read_excel(xls, sheet_name='best_bowl_sr')
bowler_econ = pd.read_excel(xls, sheet_name='best_econ')
purple_cap = pd.read_excel(xls, sheet_name='purple')
most_wickets = pd.read_excel(xls, sheet_name='most_wkt')

# Select top 10 bowlers by economy rate

# Filter data based on conditions: over 50 matches and over 20 overs
filtered_bowler_econ = bowler_econ[(bowler_econ['Mat'] > 50) & (bowler_econ['Overs'] > 20)]
top_10_bowlers_econ = filtered_bowler_econ.nsmallest(10, 'Econ')
# Create the bar chart
fig7 = px.bar(top_10_bowlers_econ, x='Player', y='Econ', title='Top 10 Bowlers by Economy Rate',
              labels={'Econ': 'Economy Rate', 'Player': 'Bowler'}, 
              hover_data=['Mat', 'Overs', 'Span'],
              color='Econ',  # Use the 'Econ' column for coloring
              color_continuous_scale='Viridis')

for i, player in enumerate(top_10_bowlers_econ['Player']):
    if 'SP Narine' in player:
        fig7.add_layout_image(
            dict(
                source=snarine,
                xref="x", yref="y",
                x=player, y=top_10_bowlers_econ['Econ'].iloc[i] / 2,
                xanchor="center", yanchor="middle",
                sizex=1.2, sizey=1.2,
                opacity=1,
                layer="above"
            )
        )
    elif 'M Muralidaran' in player:
        fig7.add_layout_image(
            dict(
                source=murali,
                xref="x", yref="y",
                x=player, y=top_10_bowlers_econ['Econ'].iloc[i] / 2,
                xanchor="center", yanchor="middle",
                sizex=1.2, sizey=1.2,
                opacity=1,
                layer="above"
            )
        )
    elif 'DW Steyn' in player:
        fig7.add_layout_image(
            dict(
                source=dayle,
                xref="x", yref="y",
                x=player, y=top_10_bowlers_econ['Econ'].iloc[i] / 2,
                xanchor="center", yanchor="middle",
                sizex=1.2, sizey=1.2,
                opacity=1,
                layer="above"
            )
        )
    elif 'Rashid' in player:
        fig7.add_layout_image(
            dict(
                source=rashid,
                xref="x", yref="y",
                x=player, y=top_10_bowlers_econ['Econ'].iloc[i] / 2,
                xanchor="center", yanchor="middle",
                sizex=1.2, sizey=1.2,
                opacity=1,
                layer="above"
            )
        )
    elif 'MM Ali' in player:
        fig7.add_layout_image(
            dict(
                source=ali,
                xref="x", yref="y",
                x=player, y=top_10_bowlers_econ['Econ'].iloc[i] / 2,
                xanchor="center", yanchor="middle",
                sizex=1.2, sizey=1.2,
                opacity=1,
                layer="above"
            )
        )
    elif 'Harbhajan' in player:
        fig7.add_layout_image(
            dict(
                source=bhaji,
                xref="x", yref="y",
                x=player, y=top_10_bowlers_econ['Econ'].iloc[i] / 2,
                xanchor="center", yanchor="middle",
                sizex=1.2, sizey=1.2,
                opacity=1,
                layer="above"
            )
        )
    elif 'R Ashwin' in player:
        fig7.add_layout_image(
            dict(
                source=ashwin,
                xref="x", yref="y",
                x=player, y=top_10_bowlers_econ['Econ'].iloc[i] / 2,
                xanchor="center", yanchor="middle",
                sizex=1.2, sizey=1.2,
                opacity=1,
                layer="above"
            )
        )
    elif 'SL Malinga' in player:
        fig7.add_layout_image(
            dict(
                source=malinga,
                xref="x", yref="y",
                x=player, y=top_10_bowlers_econ['Econ'].iloc[i] / 2,
                xanchor="center", yanchor="middle",
                sizex=1.2, sizey=1.2,
                opacity=1,
                layer="above"
            )
        )
    elif 'M Kartik' in player:
        fig7.add_layout_image(
            dict(
                source=mk,
                xref="x", yref="y",
                x=player, y=top_10_bowlers_econ['Econ'].iloc[i] / 2,
                xanchor="center", yanchor="middle",
                sizex=1.2, sizey=1.2,
                opacity=1,
                layer="above"
            )
        )
    elif 'SK Warne' in player:
        fig7.add_layout_image(
            dict(
                source=warne,
                xref="x", yref="y",
                x=player, y=top_10_bowlers_econ['Econ'].iloc[i] / 2,
                xanchor="center", yanchor="middle",
                sizex=1.2, sizey=1.2,
                opacity=1,
                layer="above"
            )
        )

fig7.update_layout(
    xaxis={'categoryorder':'total ascending'},
    updatemenus=[{
        'direction': 'down',
        'showactive': True,
    }]
)
# fig7 = px.bar(top_10_bowlers_econ, x='Player', y='Econ', title='Top 10 Bowlers by Economy Rate',
#              labels={'Econ': 'Economy Rate', 'Player': 'Bowler'}, 
#              hover_data=['Mat', 'Overs', 'Span'],
#              color='Econ',  # Use the 'SR' column for coloring
#              color_continuous_scale='Viridis')

# # Add filter for selecting span of choice
# fig7.update_layout(
#     xaxis={'categoryorder':'total ascending'},
#     updatemenus=[{
#         'direction': 'down',
#         'showactive': True,
#     }]
# )

# Select top 10 bowlers by strike rate (SR)
# Parse the 'Span' column to extract start and end years
bowler_sr[['Start_Year', 'End_Year']] = bowler_sr['Span'].str.split('-', expand=True).astype(int)

# Filter data based on conditions: over 50 matches and over 20 overs
filtered_bowler_sr = bowler_sr[(bowler_sr['Mat'] > 50) & (bowler_sr['Overs'] > 20)]

top_10_bowlers_sr = filtered_bowler_sr.nsmallest(10, 'SR')
# Create the bar chart
fig8 = px.bar(top_10_bowlers_sr, x='Player', y='SR', title='Top 10 Bowlers by Bowling Strike-Rate',
             labels={'SR': 'Strike Rate', 'Player': 'Bowler'}, 
             hover_data=['Mat', 'Overs', 'Span'],
             color='SR',  # Use the 'SR' column for coloring
             color_continuous_scale='Viridis') #marker=dict(color=top_10_batters_sr['strike_rate'], colorscale='Viridis')
for i, player in enumerate(top_10_bowlers_sr['Player']):
    if 'BJ Hodge' in player:
        fig8.add_layout_image(
            dict(
                source=hodge,
                xref="x", yref="y",
                x=player, y=top_10_bowlers_econ['SR'].iloc[i] / 2,
                xanchor="center", yanchor="middle",
                sizex=1.25, sizey=1.25,
                opacity=1,
                layer="above"
            )
        )
    elif 'AD Russell' in player:
        fig8.add_layout_image(
            dict(
                source=andreruss,
                xref="x", yref="y",
                x=player, y=top_10_bowlers_econ['SR'].iloc[i] / 2,
                xanchor="center", yanchor="middle",
                sizex=1.25, sizey=1.25,
                opacity=1,
                layer="above"
            )
        )
    elif 'K Rabada' in player:
        fig8.add_layout_image(
            dict(
                source=rabada,
                xref="x", yref="y",
                x=player, y=top_10_bowlers_econ['SR'].iloc[i] / 2,
                xanchor="center", yanchor="middle",
                sizex=1.25, sizey=1.25,
                opacity=1,
                layer="above"
            )
        )
    elif 'HV Patel' in player:
        fig8.add_layout_image(
            dict(
                source=hvpatel,
                xref="x", yref="y",
                x=player, y=top_10_bowlers_econ['SR'].iloc[i] / 2,
                xanchor="center", yanchor="middle",
                sizex=1.5, sizey=1.5,
                opacity=1,
                layer="above"
            )
        )
    elif 'Imran Tahir' in player:
        fig8.add_layout_image(
            dict(
                source=tahir,
                xref="x", yref="y",
                x=player, y=top_10_bowlers_econ['SR'].iloc[i] / 2,
                xanchor="center", yanchor="middle",
                sizex=1.5, sizey=1.5,
                opacity=1,
                layer="above"
            )
        )
    elif 'SL Malinga' in player:
        fig8.add_layout_image(
            dict(
                source=malinga,
                xref="x", yref="y",
                x=player, y=top_10_bowlers_econ['SR'].iloc[i] / 2,
                xanchor="center", yanchor="middle",
                sizex=1.5, sizey=1.55,
                opacity=1,
                layer="above"
            )
        )
    elif 'DJ Bravo' in player:
        fig8.add_layout_image(
            dict(
                source=bravo,
                xref="x", yref="y",
                x=player, y=top_10_bowlers_econ['SR'].iloc[i] / 2,
                xanchor="center", yanchor="middle",
                sizex=1.5, sizey=1.5,
                opacity=1,
                layer="above"
            )
        )
    elif 'MM Sharma' in player:
        fig8.add_layout_image(
            dict(
                source=mmsharma,
                xref="x", yref="y",
                x=player, y=top_10_bowlers_econ['SR'].iloc[i] / 2,
                xanchor="center", yanchor="middle",
                sizex=1.5, sizey=1.5,
                opacity=1,
                layer="above"
            )
        )
    elif 'YS Chahal' in player:
        fig8.add_layout_image(
            dict(
                source=yuzi,
                xref="x", yref="y",
                x=player, y=top_10_bowlers_econ['SR'].iloc[i] / 3,
                xanchor="center", yanchor="middle",
                sizex=1.5, sizey=1.5,
                opacity=1,
                layer="above"
            )
        )
    elif 'KK Ahmed' in player:
        fig8.add_layout_image(
            dict(
                source=kk,
                xref="x", yref="y",
                x=player, y=top_10_bowlers_econ['SR'].iloc[i] / 2,
                xanchor="center", yanchor="middle",
                sizex=1.5, sizey=1.5,
                opacity=1,
                layer="above"
            )
        )
# Add filter for selecting span of choice 
fig8.update_layout(
    xaxis={'categoryorder':'total ascending'},
    updatemenus=[{
        'direction': 'down',
        'showactive': True,
    },
    ]
)

# Filter the top 10 bowlers based on their wickets
top_10_bowlers = wicket_hauls.nlargest(10, 'Wkts')

# Melt the dataframe to have a long format suitable for Plotly express
melted_df = top_10_bowlers.melt(id_vars=['Player'], value_vars=['Mdns', 4, 5], 
                                var_name='Run Type', value_name='Count')

# Rename 'Mdns' to 'Maidens' in the Run Type column
melted_df['Run Type'] = melted_df['Run Type'].replace({'Mdns': 'Maidens', 4: '4-wicket hauls', 5: '5-wicket hauls'})

# Create the bubble chart
fig9 = px.scatter(melted_df, x='Player', y='Run Type', size='Count', color='Run Type',
                 title='Top 10 Bowlers by 4-fers Wicket Hauls, 5-fers Wicket Hauls & Maiden Overs',
                 labels={'Count': 'Number of Hauls', 'Player': 'Bowler', 'Run Type': 'Run Type'},
                 hover_data=['Count'],
                 size_max=80)

# Update the layout for better visibility
fig9.update_layout(
    xaxis_title='Bowler',
    yaxis_title='Frequency of 4 or 5-fers or Maidens',
    xaxis={'categoryorder':'total descending'},
    legend_title_text='Filter by Run Type'
)

# Add filter for selecting run type
fig9.update_layout(
    updatemenus=[{
        'direction': 'down',
        'showactive': True,
    }]
)

# Purple Cap winner 

# Create the dataframe
data = {
    'YEAR': [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    'PLAYER': ['Sohail Tanvir', 'RP Singh', 'Pragyan Ojha', 'Lasith Malinga', 'Morne Morkel', 'Dwayne Bravo', 
               'Mohit Sharma', 'Dwayne Bravo', 'Bhuvneshwar Kumar', 'Bhuvneshwar Kumar', 'Andrew Tye', 
               'Imran Tahir', 'Kagiso Rabada', 'Harshal Patel', 'Yuzvendra Chahal', 'Mohammed Shami'],
    'TEAM': ['Rajasthan Royals', 'Deccan Chargers', 'Deccan Chargers', 'Mumbai Indians', 'Delhi Daredevils', 
             'Chennai Super Kings', 'Chennai Super Kings', 'Chennai Super Kings', 'Sunrisers Hyderabad', 
             'Sunrisers Hyderabad', 'Kings XI Punjab', 'Chennai Super Kings', 'Delhi Capitals', 
             'Royal Challengers Bangalore', 'Rajasthan Royals', 'Gujarat Titans'],
    'WICKETS': [22, 23, 21, 28, 25, 32, 23, 26, 23, 26, 24, 26, 30, 32, 27, 28]
}

df = pd.DataFrame(data)

# Create the line plot with dots
fig10 = px.line(df, x='YEAR', y='WICKETS', markers=True, text='WICKETS', title='Purple Cap Winners in IPL (2008-2023)',
              labels={'WICKETS': 'Wickets Taken', 'YEAR': 'Year'},
              hover_data={'PLAYER': True, 'TEAM': True, 'YEAR': True, 'WICKETS': True})
# Add custom images for specific players
custom_images = [
    dict(
        source=Sohail_Tanvir,
        x=2008,
        y=22,
        sizex=1.5,
        sizey=2,
        xref='x',
        yref='y',
        xanchor='center',
        yanchor='bottom'
    ),
    dict(
        source=RP_Singh,
        x=2009,
        y=23,
        sizex=1.5,
        sizey=2,
        xref='x',
        yref='y',
        xanchor='center',
        yanchor='bottom'
    ),
    dict(
        source=Pragyan_Ojha,
        x=2010,
        y=21,
        sizex=1.5,
        sizey=2,
        xref='x',
        yref='y',
        xanchor='center',
        yanchor='bottom'
    ),
    dict(
        source=malinga,
        x=2011,
        y=28,
        sizex=1.5,
        sizey=2,
        xref='x',
        yref='y',
        xanchor='center',
        yanchor='bottom'
    ),
    dict(
        source=Morne_Morkel,
        x=2012,
        y=25,
        sizex=1.5,
        sizey=2,
        xref='x',
        yref='y',
        xanchor='center',
        yanchor='bottom'
    ),
    dict(
        source=bravo,
        x=2013,
        y=30,
        sizex=1.5,
        sizey=2,
        xref='x',
        yref='y',
        xanchor='center',
        yanchor='bottom'
    ),
    dict(
        source=mmsharma,
        x=2014,
        y=23,
        sizex=1.5,
        sizey=2,
        xref='x',
        yref='y',
        xanchor='center',
        yanchor='bottom'
    ),
    dict(
        source=bravo,
        x=2015,
        y=26,
        sizex=1.5,
        sizey=2,
        xref='x',
        yref='y',
        xanchor='center',
        yanchor='bottom'
    ),
    dict(
        source=Bhuvneshwar_Kumar,
        x=2016,
        y=23,
        sizex=1.5,
        sizey=2,
        xref='x',
        yref='y',
        xanchor='center',
        yanchor='bottom'
    ),
    dict(
        source=Bhuvneshwar_Kumar,
        x=2017,
        y=26,
        sizex=1.5,
        sizey=2,
        xref='x',
        yref='y',
        xanchor='center',
        yanchor='bottom'
    ),
    dict(
        source=Andrew_Tye,
        x=2018,
        y=24,
        sizex=1.5,
        sizey=2,
        xref='x',
        yref='y',
        xanchor='center',
        yanchor='bottom'
    ),
    dict(
        source=tahir,
        x=2019,
        y=26,
        sizex=1.5,
        sizey=2,
        xref='x',
        yref='y',
        xanchor='center',
        yanchor='bottom'
    ),
    dict(
        source=rabada,
        x=2020,
        y=30,
        sizex=1.5,
        sizey=2,
        xref='x',
        yref='y',
        xanchor='center',
        yanchor='bottom'
    ),
    dict(
        source=hvpatel,
        x=2021,
        y=30,
        sizex=1.5,
        sizey=2,
        xref='x',
        yref='y',
        xanchor='center',
        yanchor='bottom'
    ),
    dict(
        source=yuzi,
        x=2022,
        y=27,
        sizex=1.5,
        sizey=2,
        xref='x',
        yref='y',
        xanchor='center',
        yanchor='bottom'
    ),
    dict(
        source=Mohammed_Shami,
        x=2023,
        y=28,
        sizex=1.5,
        sizey=2,
        xref='x',
        yref='y',
        xanchor='center',
        yanchor='bottom'
    )
]

for image in custom_images:
    fig10.add_layout_image(image)
# Adjust dot size (you can modify the 'size' parameter to adjust dot size)
fig10.update_traces(marker=dict(size=40, color='purple'), textposition='top center') #marker=dict(color='orange')

# Top 20 Bowlers Holistic 

top_20_bowlers = wicket_hauls.nlargest(20, 'Wkts')
# Create a new column with shortened names
top_20_bowlers['Short_Player'] = top_20_bowlers['Player'].apply(lambda x: x.split(' (')[0])

# Create the bubble plot with a similar aesthetic to the provided image
fig11 = px.scatter(top_20_bowlers, x='Short_Player',
                    y='Mat', size='Wkts', color='Wkts',
                 hover_data={'Player':True, 'Econ': True, 'SR': True, 'Mdns': True, 'Wkts': True},
                 title='Top 20 Bowlers Bubble Plot (2008-2023)',
                 labels={'Mat': 'Matches Played', 'Wkts': 'Wickets', 'Econ': 'Economy', 'SR': 'Bowling SR', 'Mdns': 'Maidens'},
                 size_max=50, #color_continuous_scale='Viridis'
                 )

# Update the layout for better visibility and aesthetics
fig11.update_layout(
    xaxis_title='Bowler',
    yaxis_title='Matches Played',
    #xaxis={'categoryorder':'total descending'},
    #legend_title_text='Wickets'
    xaxis={'categoryorder':'total descending', 'tickfont': {'size': 10}},  # Smaller font size for X-axis labels
    yaxis={'tickfont': {'size': 10}},  # Optional: adjust Y-axis font size
    legend_title_text='Wickets',
    height=500  # Increase the height of the figure for a longer Y-axis
)


if menu_id == 'Analytical overview of Batsman Performance':
    # Display the figures in the specified layout
    st.plotly_chart(fig4, use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.plotly_chart(fig3, use_container_width=True)
    st.plotly_chart(fig6, use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        st.plotly_chart(fig5, use_container_width=True)
    
elif menu_id == 'Analytical overview of Bowler Performance':
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig11, use_container_width=True)
    with col2:
        st.plotly_chart(fig7, use_container_width=True)
        
    st.plotly_chart(fig10, use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig9, use_container_width=True)
    with col2:
        st.plotly_chart(fig8, use_container_width=True)
    # elif menu_id == 'Predicted Valuation for 2024 Mega auction':
    #     #Edit-Bowl_trend