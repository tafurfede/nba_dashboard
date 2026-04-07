import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os

# Set page configuration
st.set_page_config(
    page_title="Basketball Prediction Dashboard",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1E88E5;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #F0F2F6;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .info-text {
        font-size: 1rem;
        color: #555;
    }
    .highlight {
        background-color: #FFEB3B;
        padding: 0 4px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def load_data():
    """Load all CSV data files"""
    try:
        games_df = pd.read_csv('data/games_dashboard.csv')
        accuracy_df = pd.read_csv('data/running_accuracy_metrics.csv')
        summary_df = pd.read_csv('data/season_summary_stats.csv')
        
        # Convert date columns
        games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'])
        accuracy_df['GAME_DATE'] = pd.to_datetime(accuracy_df['GAME_DATE'])
        
        return games_df, accuracy_df, summary_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

def get_todays_games(games_df):
    """Get today's games or latest upcoming games"""
    # First try with RECENT_FLAG
    todays_games = games_df[games_df['GAME_STATUS'] == 'Upcoming']
    latest_date = games_df['GAME_DATE'].max()
    # If none found, use latest date
    if len(todays_games) == 0:
        
        todays_games = games_df[games_df['GAME_DATE'] == latest_date]
    
    return todays_games, latest_date

def get_previous_games(games_df, n=25):
    """Get the n most recent completed games"""
    completed_games = games_df[games_df['GAME_STATUS'] == 'Completed']
    if len(completed_games) == 0:
        return pd.DataFrame()
    
    return completed_games.sort_values('GAME_DATE', ascending=False).head(n)

def get_team_rankings(accuracy_df):
    """Get team rankings by prediction accuracy"""
    team_accuracy = accuracy_df[accuracy_df['METRIC_TYPE'] == 'TEAM'].copy()
    # Get latest accuracy for each team
    latest_team_accuracy = team_accuracy.sort_values('GAME_DATE').groupby('TEAM_NAME').last().reset_index()
    return latest_team_accuracy.sort_values('METRIC_VALUE', ascending=False)

def get_accuracy_over_time(accuracy_df):
    """Get model accuracy over time for both overall and 7-day average"""
    # Get running accuracy
    overall_accuracy = accuracy_df[accuracy_df['METRIC_TYPE'] == 'OVERALL'].copy()
    overall_accuracy = overall_accuracy.sort_values('GAME_DATE')
    
    # Get 7-day average accuracy
    seven_day_avg = accuracy_df[accuracy_df['METRIC_TYPE'] == 'OVERALL_7_DAY_AVG'].copy()
    seven_day_avg = seven_day_avg.sort_values('GAME_DATE')
    
    return overall_accuracy, seven_day_avg

# Function to apply conditional formatting to the "Correct" column
def highlight_correct(val):
    if val == "✓":
        return "background-color: #CCFFCC;"  # Light green
    elif val == "✗":
        return "background-color: #FFCCCC;"  # Light red
    return ""

# Main function
def main():
    st.markdown('<div class="main-header">🏀 Basketball Prediction Model Dashboard</div>', unsafe_allow_html=True)
    
    # Load data
    games_df, accuracy_df, summary_df = load_data()
    
    if games_df is None or accuracy_df is None or summary_df is None:
        st.error("Failed to load data. Please check your data files.")
        return
        

    
    # Today's games with predictions

    todays_games, current_date = get_todays_games(games_df)
    st.markdown(f'<div class="sub-header">Game Predictions - {current_date.strftime("%B %d, %Y")}</div>', unsafe_allow_html=True)
    
    
    if len(todays_games) == 0:
        st.info("No upcoming games found.")
    else:
        # Create a formatted table for today's games
        todays_games_table = []
        for _, game in todays_games.iterrows():
            home_team = game['HOME_TEAM_NAME']
            away_team = game['VISITOR_TEAM_NAME']
            matchup = f"{away_team} @ {home_team}"
            
            # Determine prediction
            home_win_prob = game['HOME_WIN_PROB']
            pred_winner = home_team if home_win_prob > 0.5 else away_team
            pred_winner_prob = home_win_prob if home_win_prob > 0.5 else 1 - home_win_prob
            
            # Add to table
            todays_games_table.append({
                'Matchup': matchup,
                'Home_Win_Prob': home_win_prob,
                'Home_Team': home_team,
                'Away_Team': away_team,
                'Home_Accuracy': game['HOME_TEAM_RUNNING_ACCURACY'],
                'Away_Accuracy': game['VISITOR_TEAM_RUNNING_ACCURACY']
            })

        # Convert to DataFrame
        games_df_today = pd.DataFrame(todays_games_table)

        # Create a custom display for each matchup
        for i, row in games_df_today.iterrows():

            matchup = row['Matchup']
            home_team = row['Home_Team']
            away_team = row['Away_Team']
            home_win_prob = row['Home_Win_Prob']
            away_win_prob = 1 - home_win_prob
            home_accuracy = row['Home_Accuracy']
            away_accuracy = row['Away_Accuracy']
            pred_winner = home_team if home_win_prob > 0.5 else away_team
            pred_winner_prob = home_win_prob if home_win_prob > 0.5 else 1 - home_win_prob
            
            # Create column layout
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader(matchup)
                prediction = f"**{home_team}** wins" if home_win_prob > 0.5 else f"**{away_team}** wins"
                st.write(f"Prediction: {prediction}")
                st.write(f"Win Probability: {pred_winner_prob:.1%}")
            
            with col2:
                # Create figure with subplots
                # FIX 1: Add more margin_t to the second and third subplots
                fig = make_subplots(rows=3, cols=1, 
                                subplot_titles=[f"{pred_winner} Win Probability", 
                                                f"{home_team} Running Accuracy", 
                                                f"{away_team} Running Accuracy"],
                                vertical_spacing=0.25,  # Increased from 0.15 to 0.25
                                row_heights=[0.34, 0.33, 0.33])  # Adjusted to be more equal
                
                # Win probability bar
                fig.add_trace(
                    go.Bar(
                        x=[pred_winner_prob * 100], 
                        orientation='h',
                        marker=dict(color='#1E88E5' if home_win_prob > 0.5 else '#E53935'),
                        showlegend=False,
                        text=f"{pred_winner_prob:.1%}",
                        textposition='auto'
                    ),
                    row=1, col=1
                )
                
                # Home team accuracy bar
                fig.add_trace(
                    go.Bar(
                        x=[home_accuracy * 100], 
                        orientation='h',
                        marker=dict(color='#43A047'),
                        showlegend=False,
                        text=f"{home_accuracy:.1%}",
                        textposition='auto'
                    ),
                    row=2, col=1
                )
                
                # Away team accuracy bar
                fig.add_trace(
                    go.Bar(
                        x=[away_accuracy * 100], 
                        orientation='h',
                        marker=dict(color='#FB8C00'),
                        showlegend=False,
                        text=f"{away_accuracy:.1%}",
                        textposition='auto'
                    ),
                    row=3, col=1
                )
                
                # Update layout
                fig.update_layout(
                    height=300,  
                    margin=dict(l=10, r=10, t=30, b=10),  
                    xaxis=dict(range=[0, 100], ticksuffix='%'),
                    xaxis2=dict(range=[0, 100], ticksuffix='%'),
                    xaxis3=dict(range=[0, 100], ticksuffix='%')         
                )
                fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
                fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=2, col=1)
                fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=3, col=1)
                
                  
                st.plotly_chart(fig, use_container_width=True)
            
            # Add a divider between games
            st.divider()
        
    # Previous games with results
    st.markdown('<div class="sub-header">Recent Game Results</div>', unsafe_allow_html=True)
    previous_games = get_previous_games(games_df)
    
    if len(previous_games) == 0:
        st.info("No completed games found in the dataset.")
    else:
        # Create a formatted table for previous games
        prev_games_table = []
        
        for _, game in previous_games.iterrows():
            home_team = game['HOME_TEAM_NAME']
            away_team = game['VISITOR_TEAM_NAME']
            matchup = f"{away_team} @ {home_team}"
            
            # Game result
            home_score = game['PTS_home']
            away_score = game['PTS_away']
            score = f"{away_score} - {home_score}"
            actual_winner = home_team if home_score > away_score else away_team
            
            # Prediction results
            pred_winner = home_team if game['HOME_WIN_PROB'] > 0.5 else away_team
            correct = game['CORRECT'] 
            
            # Add to table
            prev_games_table.append({
                'Date': game['GAME_DATE'],
                'Matchup': matchup,
                'Score': score,
                'Winner': actual_winner,
                'Predicted': pred_winner,
                'Correct': correct,
                
            })
        
        prev_games_df = pd.DataFrame(prev_games_table)
        
        # Format the dataframe for display
        formatted_df = prev_games_df.copy()
        formatted_df['Date'] = formatted_df['Date'].dt.strftime('%Y-%m-%d')
        formatted_df['Correct'] = formatted_df['Correct'].map({True: '✓', False: '✗'})
        
        # Apply conditional formatting to the "Correct" column
        styled_df = formatted_df.style.map(highlight_correct, subset=["Correct"])
        
        # Display the styled DataFrame with a scrollbar
        st.dataframe(styled_df, use_container_width=True, height=400)
        

    # Accuracy over time chart
    st.markdown('<div class="sub-header">Model Accuracy Over Time</div>', unsafe_allow_html=True)
    overall_accuracy, seven_day_avg = get_accuracy_over_time(accuracy_df)
    
    # Create a combined figure with both trend lines
    fig = go.Figure()
    
    # Add running accuracy line
    fig.add_trace(go.Scatter(
        x=overall_accuracy['GAME_DATE'], 
        y=overall_accuracy['METRIC_VALUE'],
        mode='lines',
        name='Running Accuracy',
        line=dict(color='#FF4B4B', width=3)
    ))
    
    # Add 7-day average line
    fig.add_trace(go.Scatter(
        x=seven_day_avg['GAME_DATE'], 
        y=seven_day_avg['METRIC_VALUE'],
        mode='lines',
        name='7-Day Average',
        line=dict(color='#1E88E5', width=3, dash='dash')
    ))
    
    # Update layout
    fig.update_layout(
        title='Prediction Model Accuracy Trend',
        xaxis_title='Date',
        yaxis_title='Accuracy',
        yaxis=dict(tickformat=".1%"),
        hovermode="x unified",
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Add an explanation for the metrics
    with st.expander("About the Accuracy Metrics"):
        st.markdown("""
        - **Running Accuracy**: Cumulative accuracy of all predictions made since the start of the season
        - **7-Day Average**: Average accuracy of predictions made in the last 7 days only
        
        The 7-day average gives you a more recent picture of model performance, which can help identify if prediction quality is improving or declining in the short term.
        """)

    
    # Team Rankings by Prediction Accuracy
    st.markdown('<div class="sub-header">Team Rankings by Model Accuracy</div>', unsafe_allow_html=True)
    team_rankings = get_team_rankings(accuracy_df)
    
    if len(team_rankings) == 0:
        st.info("No team accuracy data found.")
    else:
        # Create a bar chart for team rankings
        fig = px.bar(
            team_rankings, 
            x='TEAM_NAME', 
            y='METRIC_VALUE',
            labels={'TEAM_NAME': 'Team', 'METRIC_VALUE': 'Prediction Accuracy'},
            title='Teams Ranked by Prediction Accuracy',
            text_auto='.1%',
            color='METRIC_VALUE',
            color_continuous_scale=px.colors.sequential.Blues
        )
        fig.update_layout(
            yaxis=dict(tickformat=".0%"),
            xaxis_title="",
            xaxis_tickangle=-45,
            height=600,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display team accuracy table
        st.markdown('<div class="sub-header">Team Accuracy Table</div>', unsafe_allow_html=True)
        
        # Format table for display
        team_table = team_rankings[['TEAM_NAME', 'METRIC_VALUE']].copy()
        team_table.columns = ['Team', 'Prediction Accuracy']
        team_table['Rank'] = range(1, len(team_table) + 1)
        team_table = team_table[['Rank', 'Team', 'Prediction Accuracy']]
        team_table['Prediction Accuracy'] = team_table['Prediction Accuracy'].map("{:.1%}".format)
        
        # Create a stylish DataTable
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(team_table.columns),
                fill_color='#1E88E5',
                align='center',
                font=dict(color='white', size=14)
            ),
            cells=dict(
                values=[team_table[col] for col in team_table.columns],
                fill_color='#F0F2F6',
                align='center',
                font=dict(size=13),
                height=30
            )
        )])
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        with st.sidebar:
            st.image("images/basketball_money2.jpg", width=300)
            

            min_date = games_df['GAME_DATE'].min().date()
            max_date = games_df['GAME_DATE'].max().date()
            season_summary = summary_df.iloc[0]

            st.markdown("### Model Information")
            st.markdown("""
            A machine learning model (XGBoost) was trained on historical NBA game data to predict the outcome of games.
                        
            The current model tends to be weak for playoff games.
            
            - **Model Performance**: The model has an overall accuracy of {:.1%}
            - **Home Court Advantage**: Home teams win {:.1%} of games
            - **Date Range**: These statistics are from {} to {}
            """.format(
                season_summary['ACCURACY'],
                season_summary['HOME_TEAM_WIN_PCT'],
                min_date.strftime('%Y-%m-%d'),
                max_date.strftime('%Y-%m-%d')
            ))
            st.markdown("### Predictions Dashboard")
            st.markdown("""
            The model predicts the probability of a team winning the game, and it shows how accurate the model has been for each team.
            """)


# Run the app
if __name__ == "__main__":
    main()