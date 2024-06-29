import streamlit as st
import hydralit_components as hc
import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import base64
st.set_page_config(layout='wide')
file_path = os.getcwd()+'/pages/auction_data.csv'
print(file_path)
data = pd.read_csv(file_path)
teams = [
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore', 
    'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings', 
    'Rajasthan Royals', 'Delhi Capitals'
]
years = list(range(2013, 2023))
roles = ['All-Rounder', 'Bowler', 'Batsman', 'Wicket Keeper']
player_origins = ['Indian', 'Overseas']
# Filter & Clean data
filtered_data = data[(data['Team'].isin(teams)) & (data['Year'].isin(years))]
filtered_data['Team'] = filtered_data['Team'].replace('Delhi Daredevils', 'Delhi Capitals')
filtered_data['Amount'] = pd.to_numeric(filtered_data['Amount'], errors='coerce')
total_expense_per_year = filtered_data.groupby('Year')['Amount'].sum().reset_index()
# Sidebar filters for year, teams, roles, and player origin
selected_years = st.sidebar.multiselect('Select Years', years, default=years)
selected_teams = st.sidebar.multiselect('Select Teams', teams, default=teams)
selected_roles = st.sidebar.multiselect('Select Role', roles, default=roles)
selected_origin = st.sidebar.multiselect('Select Player Origin', player_origins, default=player_origins)
# Filter the total_expense_per_year based on sidebar selection
filtered_expense_per_year = total_expense_per_year[total_expense_per_year['Year'].isin(selected_years)]
filtered_data = filtered_data[(filtered_data['Year'].isin(selected_years)) & 
                              (filtered_data['Team'].isin(selected_teams)) & 
                              (filtered_data['Role'].isin(selected_roles)) & 
                              (filtered_data['Player Origin'].isin(selected_origin))]
# Plot the total expense per year using histogram
hist_fig = px.histogram(filtered_expense_per_year, x='Year', y='Amount', nbins=len(filtered_expense_per_year), title='Year on Year Total Expense Including All Teams', color_discrete_sequence=['salmon'])
hist_fig.update_layout(bargap=0.1)
# Plot the bar chart of amount spent by teams
sort_vals = filtered_data.sort_values(by='Amount', ascending=False)
bar_fig = px.bar(sort_vals, x="Team", y="Amount", color="Role", barmode='group', title='Amount Spent by Teams on Different Roles')
# Group by Year and Role and count the number of players
players_count_by_role = filtered_data.groupby(['Year', 'Role']).size().reset_index(name='Count')
# Plot the bubble plot
bubble_fig = px.scatter(players_count_by_role, 
                 x='Year', 
                 y='Count', 
                 size='Count', 
                 color='Role', 
                 title='Number of Players Bought by Role Year on Year',
                 labels={'Year': 'Year', 'Count': 'Number of Players'},
                 hover_name='Role')
# Update layout for better spacing and interaction
bubble_fig.update_layout(showlegend=True)
# Create the rotated box plot
box_fig = px.box(filtered_data, x='Amount', y='Role', color='Role',
                 title='Box Plot for Amount Spent on Different Roles',
                 points='all',  # Show all points (including outliers)
                 labels={'Amount': 'Amount Spent', 'Role': 'Role'})
box_fig.update_layout(xaxis_title='Amount Spent', yaxis_title='Role')
# Specify the primary menu definition
menu_data = [{'id': 'Overview of Auction Data', 'icon': "fas fa-chart-pie", 'label': "Overview of Auction Data"},
    {'id': 'Teamwise spending trends', 'icon': "fas fa-chart-area", 'label': "Teamwise spending trends"},
    {'icon': "fas fa-dollar-sign", 'label': "Player Valuation Analysis"},
    {'icon': "fas fa-gavel", 'label': "Predicted Valuation for 2024 Mega auction"},
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
#st.info(f"{menu_id}")
if menu_id == 'Overview of Auction Data': 
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(hist_fig, use_container_width=True)
    with col2:
        st.plotly_chart(bar_fig, use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(bubble_fig, use_container_width=True)
    with col2:
        st.plotly_chart(box_fig, use_container_width=True)
elif menu_id == 'Player Valuation Analysis':
    # Top 25 Auction Earners
    top_25_buys = filtered_data.groupby(['Player', 'Player Origin'], as_index=False)['Amount'].sum().sort_values(by='Amount', ascending=False).head(25)
    top_25_fig = px.histogram(top_25_buys, x='Player', y='Amount', color='Player Origin', 
                              title='Top 25 Auction Earners 2013-2022',
                              labels={'Amount': 'Amount Spent', 'Player': 'Player', 'Player Origin': 'Origin'},
                              category_orders={'Player': top_25_buys['Player']})
    # Top Players by Role and their cumulative earnings
    filtered_data['Amount'] = pd.to_numeric(filtered_data['Amount'], errors='coerce')
    cumulative_amounts = filtered_data.groupby('Player').agg({'Amount': 'sum', 'Role': 'first'}).reset_index()
    top_players = cumulative_amounts.groupby('Role').apply(lambda x: x.nlargest(5, 'Amount')).reset_index(drop=True)
    top_players_fig = px.scatter(top_players, x='Amount', y='Role', size='Amount', color='Role',
                                 hover_data={'Player': True, 'Amount': True},
                                 title='Top Players by Role and their cumulative earnings 2013-2022', color_continuous_scale=px.colors.sequential.Viridis,
                                 labels={'Amount': 'Cumulative Amount', 'Role': 'Role'}) 
    # Top Player in Auction by Amount Year on Year
    yearly_top_players = filtered_data.loc[filtered_data.groupby('Year')['Amount'].idxmax()]
    top_player_yoy_fig = go.Figure(data=[go.Scatter(x=yearly_top_players['Year'], y=yearly_top_players['Amount'], 
                                                mode='lines+markers+text', name='Top Player', 
                                                text=yearly_top_players['Player'], textposition='top center')])
    top_player_yoy_fig.update_layout(title='Top Player in Auction by Amount Year on Year',
                                 xaxis_title='Year', yaxis_title='Amount',)
    
    st.plotly_chart(top_25_fig, use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(top_players_fig, use_container_width=True)
    with col2:
        st.plotly_chart(top_player_yoy_fig, use_container_width=True)

elif menu_id == 'Teamwise spending trends': 
    # Team wise spending trends
    filtered_data['Amount'] = pd.to_numeric(filtered_data['Amount'], errors='coerce')
    year_team_spend = filtered_data.groupby(['Year', 'Team'])['Amount'].sum().reset_index()
    trend_fig = px.line(year_team_spend, x='Year', y='Amount', color='Team', markers=True, 
                  title='Year on Year Spend Trend for All Teams (2013-2022)',
                  labels={'Amount': 'Total Amount Spent', 'Year': 'Year', 'Team': 'Team'})
    trend_fig.update_layout(
        xaxis=dict(
            tickmode='linear',
            tick0=2013,
            dtick=1
        ),
        yaxis=dict(
            title='Total Amount Spent'
        ),
        legend_title_text='Teams'
    )
    col1 = st.columns(1)
    st.plotly_chart(trend_fig, use_container_width=True)
    # Indian Players Year on Year Spending
    indian_players = filtered_data[filtered_data['Player Origin'] == 'Indian']
    indian_grouped_data = indian_players.groupby(['Team', 'Year'], as_index=False)['Amount'].sum()
    indian_spending_fig = px.bar(indian_grouped_data, x='Year', y='Amount', color='Team',
                                 title='Team-wise Year on Year Spending for Indian Players',
                                 labels={'Amount': 'Amount Spent', 'Year': 'Year', 'Team': 'Team'},
                                 barmode='stack')
    # Overseas Players Year on Year Spending
    overseas_players = filtered_data[filtered_data['Player Origin'] == 'Overseas']
    overseas_grouped_data = overseas_players.groupby(['Team', 'Year'], as_index=False)['Amount'].sum()
    overseas_spending_fig = px.bar(overseas_grouped_data, x='Year', y='Amount', color='Team',
                                   title='Team-wise Year on Year Spending for Overseas Players',
                                   labels={'Amount': 'Amount Spent', 'Year': 'Year', 'Team': 'Team'},
                                   barmode='stack')
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(indian_spending_fig, use_container_width=True)
    with col2:
        st.plotly_chart(overseas_spending_fig, use_container_width=True)
    
elif menu_id == 'Predicted Valuation for 2024 Mega auction':
    st.write("Predicted Valuation for 2024 Mega auction")
    # Preprocess the data
    data = data.dropna()
    categorical_features = ['Player', 'Role', 'Team', 'Player Origin']
    numerical_features = ['Year']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Split the data into training and test sets
    X = data[['Player', 'Role', 'Team', 'Year', 'Player Origin']]
    y = data['Amount']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    # Function to predict the auction price range for a given player
    def predict_price_range(player_name):
        player_data = data[data['Player'] == player_name].iloc[0:1]
        if player_data.empty:
            return f"No data available for player: {player_name}"

        predicted_price = model.predict(player_data[['Player', 'Role', 'Team', 'Year', 'Player Origin']])
        lower_bound = max(500000, predicted_price[0] - mae)
        upper_bound = predicted_price[0] + mae

        # Ensure the range does not exceed 5 million
        if (upper_bound - lower_bound) > 5000000:
            mid_point = (lower_bound + upper_bound) / 2
            lower_bound = mid_point - 2500000
            upper_bound = mid_point + 2500000

        price_range = (round(lower_bound / 1e6, 2), round(upper_bound / 1e6, 2))  # Convert to millions
        return price_range, player_data.iloc[0]['Role']

    # Player selection box
    player_names = data['Player'].unique()
    selected_player = st.selectbox('Select a Player', player_names)

    # Define the image path using os.path.join
    def get_base64_image(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()

    image_path = os.path.join(os.getcwd(), 'pages', 'player.png')
    base64_image = get_base64_image(image_path)

    # Show predicted price range for the selected player
    if selected_player:
        price_range, role = predict_price_range(selected_player)
        st.markdown(
            f"""
            <style>
            .player-info {{
                font-size: 24px;
                font-weight: bold;
            }}
            .price-range {{
                font-size: 24px;
                font-weight: bold;
                color: green;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            th, td {{
                border: 1px solid black;
                padding: 8px;
                text-align: center;
            }}
            th {{
                background-color: #000000;
            }}
            </style>
            <table>
                <tr>
                    <th>Player</th>
                    <th>Role</th>
                    <th>Image</th>
                    <th>Predicted Price Range (Million)</th>
                </tr>
                <tr>
                    <td class="player-info">{selected_player}</td>
                    <td class="player-info">{role}</td>
                    <td><img src="data:image/png;base64,{base64_image}" alt="Player Image" width="150"></td>
                    <td class="price-range">{price_range[0]} - {price_range[1]}</td>
                </tr>
            </table>
            """, unsafe_allow_html=True
        )
    # st.write("Predicted Valuation for 2024 Mega auction")
    # # Preprocess the data
    # data = data.dropna()
    # categorical_features = ['Player', 'Role', 'Team', 'Player Origin']
    # numerical_features = ['Year']

    # preprocessor = ColumnTransformer(
    #     transformers=[
    #         ('num', SimpleImputer(strategy='median'), numerical_features),
    #         ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    #     ])

    # # Split the data into training and test sets
    # X = data[['Player', 'Role', 'Team', 'Year', 'Player Origin']]
    # y = data['Amount']

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # # Create a pipeline
    # model = Pipeline(steps=[
    #     ('preprocessor', preprocessor),
    #     ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    # ])

    # # Train the model
    # model.fit(X_train, y_train)

    # # Evaluate the model
    # y_pred = model.predict(X_test)
    # mae = mean_absolute_error(y_test, y_pred)
    # #st.write(f'Mean Absolute Error: {mae:.2f} million')

    # # Function to predict the auction price range for a given player
    # def predict_price_range(player_name):
    #     player_data = data[data['Player'] == player_name].iloc[0:1]
    #     if player_data.empty:
    #         return f"No data available for player: {player_name}"

    #     predicted_price = model.predict(player_data[['Player', 'Role', 'Team', 'Year', 'Player Origin']])
    #     lower_bound = max(500000, predicted_price[0] - mae)
    #     upper_bound = predicted_price[0] + mae
    #     u = (round(lower_bound / 1e6, 2), round(upper_bound / 1e6, 2))  # Convert to millions
    #     return price_range, player_data.iloc[0]['Role']

    # # Player selection box
    # player_names = data['Player'].unique()
    # selected_player = st.selectbox('Select a Player', player_names)
    # # Define the image path using os.path.join
    # def get_base64_image(image_path):
    #     with open(image_path, "rb") as img_file:
    #         return base64.b64encode(img_file.read()).decode()
    
    # image_path = os.path.join(os.getcwd(), 'pages', 'player.png')
    # base64_image = get_base64_image(image_path)
    # # Show predicted price range for the selected player
    # if selected_player:
    #     price_range, role = predict_price_range(selected_player)
    #     st.markdown(
    #         f"""
    #         <style>
    #         .player-info {{
    #             font-size: 24px;
    #             font-weight: bold;
    #         }}
    #         .price-range {{
    #             font-size: 24px;
    #             font-weight: bold;
    #             color: green;
    #         }}
    #         table {{
    #             width: 100%;
    #             border-collapse: collapse;
    #         }}
    #         th, td {{
    #             border: 1px solid black;
    #             padding: 8px;
    #             text-align: center;
    #         }}
    #         th {{
    #             background-color: #000000;
    #         }}
    #         </style>
    #         <table>
    #             <tr>
    #                 <th>Player</th>
    #                 <th>Role</th>
    #                 <th>Image</th>
    #                 <th>Predicted Price Range (Million)</th>
    #             </tr>
    #             <tr>
    #                 <td class="player-info">{selected_player}</td>
    #                 <td class="player-info">{role}</td>
    #                 <td><img src="data:image/png;base64,{base64_image}" alt="Player Image" width="150"></td>
    #                 <td class="price-range">{price_range[0]} - {price_range[1]}</td>
    #             </tr>
    #         </table>
    #         """, unsafe_allow_html=True
    #     )