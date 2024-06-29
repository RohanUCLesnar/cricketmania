import random
import streamlit as st
import pickle
import pandas as pd
import plotly.express as px
import math
import numpy as np
import os 
st.markdown("<h1 style='text-align: center; color: white;'> IPL Score Predictor for 1st innings</h1>", unsafe_allow_html=True)
current_directory_score = os.getcwd()+'/pages/ml_model.pkl'
#SET PAGE WIDE

 
#Get the ML model 

filename_score =current_directory_score
model_score = pickle.load(open(filename_score,'rb'))
with st.expander("How to use me ?"):
    st.info("""A ML Model to predict IPL Scores between teams in an ongoing match. Then take that first innings score 
             to make prediction on the winning percent of both teams in second innigs depending on match conditions. 
            \n1. To ensure the model results accurate score and some reliability the minimum no. of current overs considered is greater than 5 overs for First Innings.
            \n2. Select Batting and Bowling Team and enter the Current over & runs >> Followed by the wickets that have fallen so far >> Enter the runs in last 5 overs & wickets fallen.
            \n3. The score prediction range obtained in Innings 1 can be used as a indicator for broadcasting teams during televising the match for enhanced viewership experience. 
 """)

# SELECT THE BATTING TEAM


batting_team_score= st.selectbox('Select the Batting Team ',('Chennai Super Kings', 'Delhi Daredevils', 'Kings XI Punjab','Kolkata Knight Riders','Mumbai Indians','Rajasthan Royals','Royal Challengers Bangalore','Sunrisers Hyderabad'))

prediction_array_score = []
  # Batting Team
if batting_team_score == 'Chennai Super Kings':
    prediction_array_score = prediction_array_score + [1,0,0,0,0,0,0,0]
elif batting_team_score == 'Delhi Daredevils':
    prediction_array_score = prediction_array_score + [0,1,0,0,0,0,0,0]
elif batting_team_score == 'Kings XI Punjab':
    prediction_array_score = prediction_array_score + [0,0,1,0,0,0,0,0]
elif batting_team_score == 'Kolkata Knight Riders':
    prediction_array_score = prediction_array_score + [0,0,0,1,0,0,0,0]
elif batting_team_score == 'Mumbai Indians':
    prediction_array_score = prediction_array_score + [0,0,0,0,1,0,0,0]
elif batting_team_score == 'Rajasthan Royals':
    prediction_array_score = prediction_array_score + [0,0,0,0,0,1,0,0]
elif batting_team_score == 'Royal Challengers Bangalore':
    prediction_array_score = prediction_array_score + [0,0,0,0,0,0,1,0]
elif batting_team_score == 'Sunrisers Hyderabad':
    prediction_array_score = prediction_array_score + [0,0,0,0,0,0,0,1]




#SELECT BOWLING TEAM

bowling_team_score = st.selectbox('Select the Bowling Team ',('Chennai Super Kings', 'Delhi Daredevils', 'Kings XI Punjab','Kolkata Knight Riders','Mumbai Indians','Rajasthan Royals','Royal Challengers Bangalore','Sunrisers Hyderabad'))
if bowling_team_score==batting_team_score:
    st.error('Bowling and Batting teams should be different')
# Bowling Team
if bowling_team_score == 'Chennai Super Kings':
    prediction_array_score = prediction_array_score + [1,0,0,0,0,0,0,0]
elif bowling_team_score == 'Delhi Daredevils':
    prediction_array_score = prediction_array_score  + [0,1,0,0,0,0,0,0]
elif bowling_team_score == 'Kings XI Punjab':
    prediction_array_score = prediction_array_score  + [0,0,1,0,0,0,0,0]
elif bowling_team_score == 'Kolkata Knight Riders':
    prediction_array_score = prediction_array_score  + [0,0,0,1,0,0,0,0]
elif bowling_team_score == 'Mumbai Indians':
    prediction_array_score = prediction_array_score  + [0,0,0,0,1,0,0,0]
elif bowling_team_score == 'Rajasthan Royals':
    prediction_array_score = prediction_array_score  + [0,0,0,0,0,1,0,0]
elif bowling_team_score == 'Royal Challengers Bangalore':
    prediction_array_score = prediction_array_score  + [0,0,0,0,0,0,1,0]
elif bowling_team_score == 'Sunrisers Hyderabad':
    prediction_array_score = prediction_array_score  + [0,0,0,0,0,0,0,1]
col11, col21 = st.columns(2)

#Enter the Current Ongoing Over
with col11:
    #overs = st.number_input('Enter the Current Over',min_value=5.1,max_value=19.5,value=5.1,step=0.1)
    over_list = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    overs = st.selectbox('Enter the Current Over',sorted(over_list))
    
    if overs-math.floor(overs)>0.5:
        st.error('Please enter valid over input as one over only contains 6 balls')
with col21:
#Enter Current Run
    runs = st.number_input('Enter Current runs',min_value=0,max_value=354,step=1,format='%i')


#Wickets Taken till now
wickets =st.slider('Enter Wickets fallen till now',0,9)
wickets=int(wickets)

col31, col41 = st.columns(2)

with col31:
#Runs in last 5 over
    runs_in_prev_5 = st.number_input('Runs scored in the last 5 overs',min_value=0,max_value=runs,step=1,format='%i')

with col41:
#Wickets in last 5 over
    wickets_in_prev_5 = st.number_input('Wickets taken in the last 5 overs',min_value=0,max_value=wickets,step=1,format='%i')

#Get all the data for predicting

prediction_array_score = prediction_array_score + [runs, wickets, overs, runs_in_prev_5,wickets_in_prev_5]
prediction_array_score = np.array([prediction_array_score])
predict_score = model_score.predict(prediction_array_score)


if st.button('Predict Score'):
    #Call the ML Model
    my_prediction_score = int(round(predict_score[0]))
    
    #Display the predicted Score Range
    x=f'PREDICTED MATCH SCORE : {my_prediction_score-5} to {my_prediction_score+5}' 
    st.success(x)


##################################################


current_directory = os.getcwd()+'/pages/pipe.pkl'

teams = ['Sunrisers Hyderabad',
 'Mumbai Indians',
 'Royal Challengers Bangalore',
 'Kolkata Knight Riders',
 'Kings XI Punjab',
 'Chennai Super Kings',
 'Rajasthan Royals',
 'Delhi Capitals']

overs = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
wickets = [0,1,2,3,4,5,6,7,8,9]

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
       'Jaipur', 'Chennai', 'Ahmedabad', 'Dharamsala', 'Abu Dhabi',
       'Sharjah', 'Mohali', 'Bengaluru']

pipe = pickle.load(open(current_directory,'rb'))
st.title("Predict Winner from 2nd innings of the current game")
with st.expander("How to use me?"):
    st.info("""Steps- Chasing Team in 1st innings is now batting team & Prior batting team is now bowling team> 
            enter host city > Target (as estimated in 1st innings) > current score > overs completed and wickets taken
            \n1. The second innigs as it progresses, and the game is on its 2nd half we're more intrigued about the winning percentage of each playing team and obtaining the score from first innings we can use that for predicting win percent.
            \n2. The second innigs takes match location/city into account as well to determine winner as pitch plays important role in cricket. 
            \n3. For enhanced accuracy we need some score and overs to have elapsed in 2nd innings ideally after first 5 overs for best results to get win percent. 
            \n4. The win probability histogram shows a ball by ball percent change of winning for the chasing team - each ball runs over a 1000 simulation to determine changing possibility of winning
            \n5. Note that the histogram is accurate to a percent, meaning if the win prediction percent for chasing team shows anything as 0. somthing it should be treated as 0%
 """)
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the Chasing team - Currently batting',sorted(teams))
with col2:
    # Remove the selected batting team from the list of options for bowling team
    available_teams = teams.copy()
    available_teams.remove(batting_team)
    bowling_team = st.selectbox('Second, select the bowling team - who has batted first',sorted(available_teams))

selected_city = st.selectbox('Select host city',sorted(cities))

target = st.number_input('Target', step=1, format='%d', value=0)

col3,col4,col5 = st.columns(3)

with col3:
    #score = st.number_input('Score')
    score = st.number_input('Score', step=1, format='%d', value=0)
with col4:
    #overs = st.number_input('Overs completed')
    overs = st.selectbox('Overs completed',sorted(overs))
with col5:
    #wickets = st.number_input('Wickets out')
    wickets = st.selectbox('Wickets out',sorted(wickets))
if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - (overs*6)
    wickets = 10 - wickets
    crr = score/overs
    rrr = (runs_left*6)/balls_left

    input_df = pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],'city':[selected_city],'runs_left':[runs_left],'balls_left':[balls_left],'wickets':[wickets],'total_runs_x':[target],'crr':[crr],'rrr':[rrr]})

    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]
    st.header(batting_team + "- " + str(round(win*100)) + "%")
    st.header(bowling_team + "- " + str(round(loss*100)) + "%")

    input_data = {
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [wickets],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    }
    input_df = pd.DataFrame(input_data)
    #print(input_df)

    input_df = pd.DataFrame(input_data)
    result = pipe.predict_proba(input_df)
    win_percentage = result[0][1] * 100

    # Simulation for each ball left
    win_probabilities = []
    for ball in range(balls_left):
        # Simulate all possible outcomes for the next ball
        runs = random.randint(0, 6)  # Simulate random runs between 0 and 6
        is_wicket = random.random() < 0.05  # Simulate 5% chance of getting out (adjust as per your need)
        new_runs_left = runs_left - runs
        new_balls_left = balls_left - 1
        new_wickets = wickets - (1 if runs == 0 else 0)
        new_crr = (score + runs) / (overs + (ball + 1) / 6)
        new_rrr = (new_runs_left * 6) / new_balls_left if new_balls_left > 0 else 0
            
        # Avoid negative values
        if new_runs_left < 0:
            continue
        if new_wickets < 0:
            new_wickets = 0
            
        new_input_data = {
                'batting_team': [batting_team],
                'bowling_team': [bowling_team],
                'city': [selected_city],
                'runs_left': [new_runs_left],
                'balls_left': [new_balls_left],
                'wickets': [new_wickets],
                'total_runs_x': [target],
                'crr': [new_crr],
                'rrr': [new_rrr]
            }
        new_input_df = pd.DataFrame(new_input_data)
        result = pipe.predict_proba(new_input_df)
        win_probabilities.append({'ball':ball,
                'win_probability': result[0][1] * 100
            })
        #'ball': 120 - new_balls_left,
    win_prob_df = pd.DataFrame(win_probabilities)
    print(win_prob_df)
    # Plotting the win probabilities using Plotly
    # fig = px.histogram(win_prob_df, x='ball', y='win_probability', nbins=balls_left,
    #                    title='Win Probability Over Remaining Balls only valid for over 30 remaining balls and any win percentage starting with 0 must be assumed as 0',
    #                    labels={'ball': 'Balls Left', 'win_probability': 'Win Probability (%)'})
    # st.plotly_chart(fig)
    # Create histogram figure
    fig = px.histogram(win_prob_df, x='ball', y='win_probability', nbins=balls_left,
                    labels={'ball': 'Balls Left', 'win_probability': 'Win Probability (%)'})

    # Update layout for better visualization
    fig.update_layout(
        title='Win Probability Over Remaining Balls (valid for over 30 remaining balls)<br>-consider anything below 1 starting with 0 as just 0.',
        title_font_size=17,  # Adjust title font size
        title_font_color='white',  # Title font color
        xaxis_title='Balls Left',
        yaxis_title='Win Probability (%)',
        bargap=0.02,  # Gap between bars
        font=dict(
            family='Arial',  # Font family
            size=14,  # Font size
            color='black'  # Font color
        ),
        autosize=False,  # Disable auto sizing
        width=1800,  # Set figure width
        height=500,  # Set figure height
    )

    # Update histogram bar color
    fig.update_traces(marker_color='salmon')

    # Plot the updated figure using Streamlit
    st.plotly_chart(fig)

