import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Machine Learning & Metrics
from sklearn.model_selection import TimeSeriesSplit, ShuffleSplit
from sklearn.metrics import log_loss, brier_score_loss, classification_report
from sklearn.calibration import calibration_curve
from scipy.stats import ttest_rel

# Hyperparameter Optimization
from skopt import BayesSearchCV
from skopt.space import Real, Integer

# Models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# =========================================
# 1. Configuration & Legacy Support
# =========================================

# Suppress warnings for cleaner output in production
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Numpy compatibility patches for older libraries (e.g., some versions of skopt)
# These ensure compatibility if the environment uses recently deprecated numpy types.
try:
    np.int = np.int32
    np.float = np.float64
    np.bool = np.bool_
except AttributeError:
    pass  # Newer numpy versions might not need/support this, safe to ignore if it fails.

# Global Constants
RANDOM_STATE = 8
N_ITER_BAYES = 25  # Reduced for demonstration; increase for production (e.g., 100+)
N_SPLITS_TIME_CV = 4

# Simulation Constants
SIM_N_SPLITS = 1000
SIM_PORTION_SIZE = 0.8
INITIAL_BANKROLL = 500.0
MIN_BET_CONFIDENCE = 0.60

# =========================================
# 2. Feature Engineering Functions
# =========================================

def calculate_simple_moving_averages(df, team_column, stat_columns, prefix, window=6):
    """
    Calculates simple rolling averages for a specific team perspective (Home or Away).

    Args:
        df (pd.DataFrame): Input dataframe.
        team_column (str): Column name to group by (e.g., 'HomeTeam').
        stat_columns (list): List of statistic columns to average.
        prefix (str): Prefix for the new columns.
        window (int): Rolling window size.

    Returns:
        pd.DataFrame: Dataframe with new moving average columns.
    """
    df_copy = df.copy()
    for stat in stat_columns:
        # Shift by 1 to exclude the current match from the average
        df_copy[f'{prefix}_{window}_{stat}'] = (
            df_copy.groupby(team_column)[stat]
                   .apply(lambda x: x.shift(1).rolling(window=window, min_periods=2).mean())
                   .reset_index(level=0, drop=True)
        )
    return df_copy

def create_total_stats(df, pairs):
    """
    Aggregates specific home and away statistic pairs into 'Total' columns.

    Args:
        df (pd.DataFrame): Input dataframe.
        pairs (list of tuples): Tuples containing (HomeCol, AwayCol, NewTotalName).

    Returns:
        tuple: (pd.DataFrame with new columns, list of new column names)
    """
    df_copy = df.copy()
    new_total_cols = []
    for home_col, away_col, total_col in pairs:
        if home_col in df_copy.columns and away_col in df_copy.columns:
            df_copy[total_col] = df_copy[home_col] + df_copy[away_col]
            new_total_cols.append(total_col)
    return df_copy, new_total_cols

def calculate_complex_rolling_stats(df, window=6):
    """
    Calculates comprehensive rolling averages for each team over their last N matches,
    regardless of whether they played home or away in those previous matches.

    Note: This function iterates through rows and may be slow on very large datasets.

    Args:
        df (pd.DataFrame): Dataframe containing match data, sorted by date.
        window (int): Number of matches to consider for the rolling average.

    Returns:
        pd.DataFrame: Dataframe with new comprehensive rolling average columns.
    """
    df_result = df.copy().sort_values('Date').reset_index(drop=True)

    # Mapping between Home stats and their Away counterparts for unified averaging
    stat_mappings = {
        'HomeGoals': 'AwayGoals', 'Home_xG': 'Away_xG', 'Home_Shots': 'Away_Shots',
        'Home_Shots_On_Target': 'Away_Shots_On_Target', 'Home_Shot_Distance': 'Away_Shot_Distance',
        'Home_Free_Kicks': 'Away_Free_Kicks', 'Home_Penalties': 'Away_Penalties',
        'Home_PSxG': 'Away_PSxG', 'Home_%Stopped_Crosses': 'Away_%Stopped_Crosses',
        'Home_SCA': 'Away_SCA', 'Home_GCA': 'Away_GCA',
        'Home_Passes_Attempted': 'Away_Passes_Attempted', 'Home_%Passes_Completed': 'Away_%Passes_Completed',
        'Home_xA': 'Away_xA', 'Home_Key_Passes': 'Away_Key_Passes',
        'Home_Final_Third_Passes': 'Away_Final_Third_Passes', 'Home_PPA': 'Away_PPA',
        'Home_CrsPA': 'Away_CrsPA', 'Home_Progressive_Passes': 'Away_Progressive_Passes',
        'Home_Points': 'Away_Points', 'Home_xPoints': 'Away_xPoints',
        'Home_PSxG_xPoints': 'Away_PSxG_xPoints', 'half_time_home_goals': 'half_time_away_goals',
        'Home_corners': 'Away_corners'
    }

    # Pre-initialize new columns with NaN to avoid fragmentation in the loop
    for home_stat in stat_mappings.keys():
        df_result[f'Home_MA{window}_{home_stat}'] = np.nan
        df_result[f'Away_MA{window}_{stat_mappings[home_stat]}'] = np.nan

    # Iterate through matches to calculate historical averages based on team history
    for idx, row in df_result.iterrows():
        current_date = row['Date']
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']

        # Identify previous matches for both teams involved in the current match
        # We look strictly at matches BEFORE the current match date
        mask_prev = df_result['Date'] < current_date
        home_history = df_result[mask_prev & ((df_result['HomeTeam'] == home_team) | (df_result['AwayTeam'] == home_team))].tail(window)
        away_history = df_result[mask_prev & ((df_result['HomeTeam'] == away_team) | (df_result['AwayTeam'] == away_team))].tail(window)

        # Calculate Home Team's recent performance average
        if not home_history.empty:
            for h_stat, a_stat in stat_mappings.items():
                # Extract the correct stat depending on if they were home or away in that historical match
                values = np.where(home_history['HomeTeam'] == home_team, home_history[h_stat], home_history[a_stat])
                df_result.at[idx, f'Home_MA{window}_{h_stat}'] = np.mean(values)

        # Calculate Away Team's recent performance average
        if not away_history.empty:
            for h_stat, a_stat in stat_mappings.items():
                values = np.where(away_history['HomeTeam'] == away_team, away_history[h_stat], away_history[a_stat])
                df_result.at[idx, f'Away_MA{window}_{a_stat}'] = np.mean(values)

    return df_result

# =========================================
# 3. Financial Simulation Functions
# =========================================

def adjust_stake(stake, min_stake=1.0, max_stake=50.0):
    """
    Applies staking constraints (min/max limits, rounding).
    """
    if stake < 0:
        return 0.0
    # Example constraint: Keep bets between 1€ and 50€ if they exceed a threshold
    if stake > 0 and stake < min_stake:
        return min_stake
    if stake > max_stake:
        return max_stake
    return round(stake, 2)

def run_profit_simulation(y_true, y_prob, odds_df, initial_bankroll=500.0, min_confidence=0.60):
    """
    Executes a Kelly Criterion-based betting simulation on a sequence of matches.

    Args:
        y_true (np.array): True binary outcomes (0 for Under, 1 for Over).
        y_prob (np.array): Predicted probabilities from the model [prob_under, prob_over].
        odds_df (pd.DataFrame): Dataframe containing 'odds_over' and 'odds_under' columns.
        initial_bankroll (float): Starting capital.
        min_confidence (float): Minimum model probability required to place a bet.

    Returns:
        tuple: (Final bankroll, List of individual bet profits)
    """
    bankroll = initial_bankroll
    profits_taken = 0.0
    seed_withdrawn = False
    bet_profits = []

    # Convert to numpy arrays for faster indexing in the loop
    odds_over_arr = odds_df['odds_over'].values
    odds_under_arr = odds_df['odds_under'].values

    for i in range(len(y_true)):
        if bankroll <= 1:
             break # Stop if busted

        odds_o = odds_over_arr[i]
        odds_u = odds_under_arr[i]
        prob_u, prob_o = y_prob[i] # Class 0 = Under, Class 1 = Over

        # Calculate implied probabilities
        implied_o = 1.0 / odds_o if odds_o > 1 else 1.0
        implied_u = 1.0 / odds_u if odds_u > 1 else 1.0

        stake = 0.0
        profit = 0.0

        # --- Betting Logic (Kelly Criterion) ---
        # Bet OVER if model sees value AND has high enough confidence
        if prob_o > implied_o and prob_o >= min_confidence:
            kelly_fraction = (prob_o * odds_o - 1) / (odds_o - 1)
            stake = adjust_stake(kelly_fraction * bankroll)
            if stake > 0:
                profit = stake * (odds_o - 1) if y_true[i] == 1 else -stake

        # Bet UNDER if model sees value AND has high enough confidence
        elif prob_u > implied_u and prob_u >= min_confidence:
            kelly_fraction = (prob_u * odds_u - 1) / (odds_u - 1)
            stake = adjust_stake(kelly_fraction * bankroll)
            if stake > 0:
                profit = stake * (odds_u - 1) if y_true[i] == 0 else -stake

        if stake > 0:
            bankroll += profit
            bet_profits.append(profit)

        # Optional: Withdraw initial seed capital once doubled (risk management)
        if not seed_withdrawn and (bankroll - initial_bankroll >= 125):
             profits_taken += initial_bankroll
             bankroll -= initial_bankroll
             seed_withdrawn = True

    total_wealth = bankroll + profits_taken
    return total_wealth, bet_profits

# =========================================
# 4. Main Pipeline
# =========================================

if __name__ == '__main__':
    print("🚀 Starting Predictive Modeling Pipeline...")

    # --- 4.1 Data Loading & Preprocessing ---
    print("\n[1/6] Loading and Preprocessing Data...")
    df = pd.read_csv('data/general_toy_dataset.csv', parse_dates=['Date'])
    df = df.sort_values(by='Date').reset_index(drop=True)
    test_df = pd.read_csv('data/df_6_test.csv')

    # Define base statistics to analyze
    base_stats = [
    'HomeGoals', 'Home_xG', 'AwayGoals','Away_xG', 'Home_Shots', 'Away_Shots', 'Home_Shots_On_Target', 'Away_Shots_On_Target',
    'Home_Shot_Distance', 'Away_Shot_Distance', 'Home_Free_Kicks', 'Away_Free_Kicks',
    'Home_Penalties', 'Away_Penalties', 'Home_PSxG', 'Away_PSxG', 'Home_%Stopped_Crosses',
    'Away_%Stopped_Crosses', 'Home_SCA', 'Away_SCA', 'Home_GCA', 'Away_GCA',
    'Home_Passes_Attempted', 'Away_Passes_Attempted', 'Home_%Passes_Completed',
    'Away_%Passes_Completed', 'Home_xA', 'Away_xA', 'Home_Key_Passes', 'Away_Key_Passes',
    'Home_Final_Third_Passes', 'Away_Final_Third_Passes', 'Home_PPA', 'Away_PPA',
    'Home_CrsPA', 'Away_CrsPA', 'Home_Progressive_Passes', 'Away_Progressive_Passes',
    'Home_Points', 'Away_Points', 'Home_xPoints','Away_xPoints', 'Home_PSxG_xPoints', 'Away_PSxG_xPoints', 'half_time_home_goals',
    'half_time_away_goals', 'Home_corners', 'Away_corners'
    ]

    # 1. Simple Moving Averages (per team, Home vs Away specific)
    df = calculate_simple_moving_averages(df, "HomeTeam", base_stats, "average_home", window=6)
    df = calculate_simple_moving_averages(df, "AwayTeam", base_stats, "average_away", window=6)

    # 2. Create Total (Match Aggregate) Stats
    stat_pairs = [
    ('HomeGoals', 'AwayGoals', 'TotalGoals'),
    ('Home_xG', 'Away_xG', 'Total_xG'),
    ('Home_Shots', 'Away_Shots', 'Total_Shots'),
    ('Home_Shots_On_Target', 'Away_Shots_On_Target', 'Total_Shots_On_Target'),
    ('Home_Shot_Distance', 'Away_Shot_Distance', 'Total_Shot_Distance'),
    ('Home_Free_Kicks', 'Away_Free_Kicks', 'Total_Free_Kicks'),
    ('Home_Penalties', 'Away_Penalties', 'Total_Penalties'),
    ('Home_PSxG', 'Away_PSxG', 'Total_PSxG'),
    ('Home_%Stopped_Crosses', 'Away_%Stopped_Crosses', 'Total_%Stopped_Crosses'),
    ('Home_SCA', 'Away_SCA', 'Total_SCA'),
    ('Home_GCA', 'Away_GCA', 'Total_GCA'),
    ('Home_Passes_Attempted', 'Away_Passes_Attempted', 'Total_Passes_Attempted'),
    ('Home_%Passes_Completed', 'Away_%Passes_Completed', 'Total_%Passes_Completed'),
    ('Home_xA', 'Away_xA', 'Total_xA'),
    ('Home_Key_Passes', 'Away_Key_Passes', 'Total_Key_Passes'),
    ('Home_Final_Third_Passes', 'Away_Final_Third_Passes', 'Total_Final_Third_Passes'),
    ('Home_PPA', 'Away_PPA', 'Total_PPA'),
    ('Home_CrsPA', 'Away_CrsPA', 'Total_CrsPA'),
    ('Home_Progressive_Passes', 'Away_Progressive_Passes', 'Total_Progressive_Passes'),
    ('Home_Points', 'Away_Points', 'Total_Points'),
    ('Home_xPoints', 'Away_xPoints', 'Total_xPoints'),
    ('Home_PSxG_xPoints', 'Away_PSxG_xPoints', 'Total_PSxG_xPoints'),
    ('half_time_home_goals', 'half_time_away_goals', 'Total_half_time_goals'),
    ('Home_corners', 'Away_corners', 'Total_corners')
    ]
    df, total_cols = create_total_stats(df, stat_pairs)

    # 3. Moving Averages of Total Stats
    df_ma_agg_home = calculate_simple_moving_averages(df, 'HomeTeam', total_cols, 'MA_Home_Agg', window=6)
    df_ma_agg_away = calculate_simple_moving_averages(df, 'AwayTeam', total_cols, 'MA_Away_Agg', window=6)

    # Merge these specific aggregated MA columns back to main df
    new_home_cols = [c for c in df_ma_agg_home.columns if c.startswith('MA_Home_Agg')]
    new_away_cols = [c for c in df_ma_agg_away.columns if c.startswith('MA_Away_Agg')]
    df = pd.concat([df, df_ma_agg_home[new_home_cols], df_ma_agg_away[new_away_cols]], axis=1)

    # 4. Complex Rolling Stats (Home/Away agnostic history)
    print("      > Calculating complex rolling stats (this may take a moment)...")
    df = calculate_complex_rolling_stats(df, window=6)

    # --- 4.2 Feature Selection & Splitting ---
    print("\n[2/6] Splitting Data into Train/Test...")

    # Define Feature List (Ensure these match columns created above)
    # This is a reduced subset for demonstration. Use your full list in production.
    FEATURES = [
    'odds_home_win', 'odds_draw', 'odds_away_win',
    'average_home_6_HomeGoals', 'average_home_6_Home_xG', 'average_home_6_AwayGoals', 'average_home_6_Away_xG', 
    'average_home_6_Home_Shots', 'average_home_6_Away_Shots', 'average_home_6_Home_Shots_On_Target', 
    'average_home_6_Away_Shots_On_Target', 'average_home_6_Home_Shot_Distance', 'average_home_6_Away_Shot_Distance', 
    'average_home_6_Home_Free_Kicks', 'average_home_6_Away_Free_Kicks', 'average_home_6_Home_Penalties', 
    'average_home_6_Away_Penalties', 'average_home_6_Home_PSxG', 'average_home_6_Away_PSxG', 
    'average_home_6_Home_%Stopped_Crosses', 'average_home_6_Away_%Stopped_Crosses', 'average_home_6_Home_SCA', 
    'average_home_6_Away_SCA', 'average_home_6_Home_GCA', 'average_home_6_Away_GCA', 
    'average_home_6_Home_Passes_Attempted', 'average_home_6_Away_Passes_Attempted', 
    'average_home_6_Home_%Passes_Completed', 'average_home_6_Away_%Passes_Completed', 
    'average_home_6_Home_xA', 'average_home_6_Away_xA', 'average_home_6_Home_Key_Passes', 
    'average_home_6_Away_Key_Passes', 'average_home_6_Home_Final_Third_Passes', 
    'average_home_6_Away_Final_Third_Passes', 'average_home_6_Home_PPA', 'average_home_6_Away_PPA', 
    'average_home_6_Home_CrsPA', 'average_home_6_Away_CrsPA', 'average_home_6_Home_Progressive_Passes', 
    'average_home_6_Away_Progressive_Passes', 'average_home_6_Home_Points', 'average_home_6_Away_Points', 
    'average_home_6_Home_xPoints', 'average_home_6_Away_xPoints', 'average_home_6_Home_PSxG_xPoints', 
    'average_home_6_Away_PSxG_xPoints', 'average_home_6_half_time_home_goals', 
    'average_home_6_half_time_away_goals', 'average_home_6_Home_corners', 'average_home_6_Away_corners',

    'average_away_6_HomeGoals', 'average_away_6_Home_xG', 'average_away_6_AwayGoals', 'average_away_6_Away_xG', 
    'average_away_6_Home_Shots', 'average_away_6_Away_Shots', 'average_away_6_Home_Shots_On_Target', 
    'average_away_6_Away_Shots_On_Target', 'average_away_6_Home_Shot_Distance', 'average_away_6_Away_Shot_Distance', 
    'average_away_6_Home_Free_Kicks', 'average_away_6_Away_Free_Kicks', 'average_away_6_Home_Penalties', 
    'average_away_6_Away_Penalties', 'average_away_6_Home_PSxG', 'average_away_6_Away_PSxG', 
    'average_away_6_Home_%Stopped_Crosses', 'average_away_6_Away_%Stopped_Crosses', 'average_away_6_Home_SCA', 
    'average_away_6_Away_SCA', 'average_away_6_Home_GCA', 'average_away_6_Away_GCA', 
    'average_away_6_Home_Passes_Attempted', 'average_away_6_Away_Passes_Attempted', 
    'average_away_6_Home_%Passes_Completed', 'average_away_6_Away_%Passes_Completed', 
    'average_away_6_Home_xA', 'average_away_6_Away_xA', 'average_away_6_Home_Key_Passes', 
    'average_away_6_Away_Key_Passes', 'average_away_6_Home_Final_Third_Passes', 
    'average_away_6_Away_Final_Third_Passes', 'average_away_6_Home_PPA', 'average_away_6_Away_PPA', 
    'average_away_6_Home_CrsPA', 'average_away_6_Away_CrsPA', 'average_away_6_Home_Progressive_Passes', 
    'average_away_6_Away_Progressive_Passes', 'average_away_6_Home_Points', 'average_away_6_Away_Points', 
    'average_away_6_Home_xPoints', 'average_away_6_Away_xPoints', 'average_away_6_Home_PSxG_xPoints', 
    'average_away_6_Away_PSxG_xPoints', 'average_away_6_half_time_home_goals', 
    'average_away_6_half_time_away_goals', 'average_away_6_Home_corners', 'average_away_6_Away_corners',

    'MA_Home_Agg_6_TotalGoals', 'MA_Home_Agg_6_Total_xG', 'MA_Home_Agg_6_Total_Shots', 
    'MA_Home_Agg_6_Total_Shots_On_Target', 'MA_Home_Agg_6_Total_Shot_Distance', 'MA_Home_Agg_6_Total_Free_Kicks', 
    'MA_Home_Agg_6_Total_Penalties', 'MA_Home_Agg_6_Total_PSxG', 'MA_Home_Agg_6_Total_%Stopped_Crosses', 
    'MA_Home_Agg_6_Total_SCA', 'MA_Home_Agg_6_Total_GCA', 'MA_Home_Agg_6_Total_Passes_Attempted', 
    'MA_Home_Agg_6_Total_%Passes_Completed', 'MA_Home_Agg_6_Total_xA', 'MA_Home_Agg_6_Total_Key_Passes', 
    'MA_Home_Agg_6_Total_Final_Third_Passes', 'MA_Home_Agg_6_Total_PPA', 'MA_Home_Agg_6_Total_CrsPA', 
    'MA_Home_Agg_6_Total_Progressive_Passes', 'MA_Home_Agg_6_Total_Points', 'MA_Home_Agg_6_Total_xPoints', 
    'MA_Home_Agg_6_Total_PSxG_xPoints', 'MA_Home_Agg_6_Total_half_time_goals', 'MA_Home_Agg_6_Total_corners',

    'MA_Away_Agg_6_TotalGoals', 'MA_Away_Agg_6_Total_xG', 'MA_Away_Agg_6_Total_Shots', 
    'MA_Away_Agg_6_Total_Shots_On_Target', 'MA_Away_Agg_6_Total_Shot_Distance', 'MA_Away_Agg_6_Total_Free_Kicks', 
    'MA_Away_Agg_6_Total_Penalties', 'MA_Away_Agg_6_Total_PSxG', 'MA_Away_Agg_6_Total_%Stopped_Crosses', 
    'MA_Away_Agg_6_Total_SCA', 'MA_Away_Agg_6_Total_GCA', 'MA_Away_Agg_6_Total_Passes_Attempted', 
    'MA_Away_Agg_6_Total_%Passes_Completed', 'MA_Away_Agg_6_Total_xA', 'MA_Away_Agg_6_Total_Key_Passes', 
    'MA_Away_Agg_6_Total_Final_Third_Passes', 'MA_Away_Agg_6_Total_PPA', 'MA_Away_Agg_6_Total_CrsPA', 
    'MA_Away_Agg_6_Total_Progressive_Passes', 'MA_Away_Agg_6_Total_Points', 'MA_Away_Agg_6_Total_xPoints', 
    'MA_Away_Agg_6_Total_PSxG_xPoints', 'MA_Away_Agg_6_Total_half_time_goals', 'MA_Away_Agg_6_Total_corners',

    'Home_MA6_HomeGoals', 'Away_MA6_AwayGoals', 'Home_MA6_Home_xG', 'Away_MA6_Away_xG', 
    'Home_MA6_Home_Shots', 'Away_MA6_Away_Shots', 'Home_MA6_Home_Shots_On_Target', 'Away_MA6_Away_Shots_On_Target', 
    'Home_MA6_Home_Shot_Distance', 'Away_MA6_Away_Shot_Distance', 'Home_MA6_Home_Free_Kicks', 
    'Away_MA6_Away_Free_Kicks', 'Home_MA6_Home_Penalties', 'Away_MA6_Away_Penalties', 
    'Home_MA6_Home_PSxG', 'Away_MA6_Away_PSxG', 'Home_MA6_Home_%Stopped_Crosses', 'Away_MA6_Away_%Stopped_Crosses', 
    'Home_MA6_Home_SCA', 'Away_MA6_Away_SCA', 'Home_MA6_Home_GCA', 'Away_MA6_Away_GCA', 
    'Home_MA6_Home_Passes_Attempted', 'Away_MA6_Away_Passes_Attempted', 
    'Home_MA6_Home_%Passes_Completed', 'Away_MA6_Away_%Passes_Completed', 'Home_MA6_Home_xA', 'Away_MA6_Away_xA', 
    'Home_MA6_Home_Key_Passes', 'Away_MA6_Away_Key_Passes', 'Home_MA6_Home_Final_Third_Passes', 
    'Away_MA6_Away_Final_Third_Passes', 'Home_MA6_Home_PPA', 'Away_MA6_Away_PPA', 'Home_MA6_Home_CrsPA', 
    'Away_MA6_Away_CrsPA', 'Home_MA6_Home_Progressive_Passes', 'Away_MA6_Away_Progressive_Passes', 
    'Home_MA6_Home_Points', 'Away_MA6_Away_Points', 'Home_MA6_Home_xPoints', 'Away_MA6_Away_xPoints', 
    'Home_MA6_Home_PSxG_xPoints', 'Away_MA6_Away_PSxG_xPoints', 'Home_MA6_half_time_home_goals', 
    'Away_MA6_half_time_away_goals', 'Home_MA6_Home_corners', 'Away_MA6_Away_corners',

    'odds_over', 'odds_under', 
    'Home_Standings_Points', 'Away_Standings_Points', 'Home_Standings_Rank', 'Away_Standings_Rank'
]



    # Drop rows with NaN values resulting from rolling windows
    df_clean = df.dropna(subset=FEATURES).reset_index(drop=True)


    X_train = df[FEATURES]
    y_train = df['Over']
    X_test = test_df[FEATURES]
    y_test = test_df['Over']

    print(f"      Training samples: {len(X_train)}")
    print(f"      Testing samples:  {len(X_test)}")

    # --- 4.3 Model Definition & Bayesian Optimization ---
    print(f"\n[3/6] Training Models with Bayesian Optimization (n_iter={N_ITER_BAYES})...")

    tscv = TimeSeriesSplit(n_splits=N_SPLITS_TIME_CV)

    search_spaces = {
        'XGB': {
            'n_estimators': Integer(100, 500),
            'learning_rate': Real(0.01, 0.1, 'log-uniform'),
            'max_depth': Integer(3, 6),
            'subsample': Real(0.7, 1.0),
            'colsample_bytree': Real(0.7, 1.0),
        },
        'LGBM': {
            'n_estimators': Integer(100, 500),
            'learning_rate': Real(0.01, 0.1, 'log-uniform'),
            'num_leaves': Integer(20, 40),
            'subsample': Real(0.7, 1.0),
            'colsample_bytree': Real(0.7, 1.0),
        },
        'CatBoost': {
            'n_estimators': Integer(100, 1000),
            'learning_rate': Real(0.01, 0.2, 'log-uniform'),
            'depth': Integer(3, 8),
            'reg_lambda': Real(0.1, 5.0, 'log-uniform'),
            'subsample': Real(0.6, 1.0),
    }
    }

    models = {
        'XGB': XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", use_label_encoder=False, n_jobs=-1, verbose=0),
        'LGBM': LGBMClassifier(random_state=RANDOM_STATE, metric='logloss', n_jobs=-1, verbose=0),
        'CatBoost': CatBoostClassifier(random_state=RANDOM_STATE, eval_metric='Logloss', thread_count=-1, verbose=0)

    }

    trained_models = {}
    model_log_losses = {}

    for name, model in models.items():
        print(f"      > Optimizing {name}...")
        start_t = time.time()
        search = BayesSearchCV(
            estimator=model,
            search_spaces=search_spaces[name],
            n_iter=N_ITER_BAYES,
            cv=tscv,
            scoring='neg_log_loss',
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=0,
            refit=True
        )
        search.fit(X_train, y_train)
        trained_models[name] = search.best_estimator_
        print(f"        {name} finished in {time.time()-start_t:.1f}s | Best CV LogLoss: {-search.best_score_:.4f}")

    # --- 4.4 Evaluation & Statistical Testing ---
    print("\n[4/6] Evaluating Models on Test Set...")
    results = {}
    for name, model in trained_models.items():
        y_prob = model.predict_proba(X_test)
        ll = log_loss(y_test, y_prob)
        results[name] = {'Log Loss': ll}

    results_df = pd.DataFrame(results).T.sort_values('Log Loss')
    print("\n--- Performance Summary ---")
    print(results_df)

    # Paired T-Test for Statistical Significance
    best_model_name = results_df.index[0]
    print(f"\n--- Statistical Significance (vs {best_model_name}) ---")
    for name in trained_models.keys():
        if name != best_model_name:
            t_stat, p_val = ttest_rel(model_log_losses[best_model_name], model_log_losses[name])
            sig = "Significant" if p_val < 0.05 else "Not Significant"
            print(f"  {best_model_name} vs {name}: p-value = {p_val:.4f} ({sig})")

    # --- 4.5 Profitability Simulation ---
    print(f"\n[5/6] Running Monte Carlo Profit Simulation ({SIM_N_SPLITS} runs)...")
    # Use the best model for simulation
    best_model = trained_models[best_model_name]
    y_prob_best = best_model.predict_proba(X_test)

    rs = ShuffleSplit(n_splits=SIM_N_SPLITS, test_size=SIM_PORTION_SIZE, random_state=RANDOM_STATE)
    wealth_results = []
    all_bets_profits = []

    test_indices = np.arange(len(test_df))
    y_test_values = y_test.values

    start_sim = time.time()
    for train_idx, portion_idx in rs.split(test_indices):
        w, bets = run_profit_simulation(
            y_test_values[portion_idx],
            y_prob_best[portion_idx],
            test_df.iloc[portion_idx],
            initial_bankroll=INITIAL_BANKROLL,
            min_confidence=MIN_BET_CONFIDENCE
        )
        wealth_results.append(w)
        all_bets_profits.extend(bets)

    # Simulation Metrics
    mean_wealth = np.mean(wealth_results)
    roi = (mean_wealth / INITIAL_BANKROLL) - 1
    ruin_prob = np.mean(np.array(wealth_results) <= 1.0)
    sharpe = np.mean(all_bets_profits) / np.std(all_bets_profits) if all_bets_profits else 0

    print(f"\n--- Financial Projection ({best_model_name}) ---")
    print(f"  Mean Final Wealth: {mean_wealth:.2f}€ (Starting: {INITIAL_BANKROLL}€)")
    print(f"  Expected ROI:      {roi:.2%}")
    print(f"  Risk of Ruin:      {ruin_prob:.2%}")
    print(f"  Sharpe Ratio/Bet:  {sharpe:.4f}")

    # --- 4.6 Final Analysis & Saving ---
    print("\n[6/6] Finalizing and Saving Best Model...")

    # Calibration Plot
    plt.figure(figsize=(8, 6))
    prob_true, prob_pred = calibration_curve(y_test, y_prob_best[:, 1], n_bins=10, strategy='uniform')
    plt.plot(prob_pred, prob_true, marker='o', label=f'{best_model_name}')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title(f'Calibration Plot - {best_model_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'model_calibration_{best_model_name.lower()}.png')
    print(f"      > Calibration plot saved to 'model_calibration_{best_model_name.lower()}.png'")

    # Save Model
    model_filename = f'model/best_model_{best_model_name.lower()}.joblib'
    # Ensure directory exists
    import os
    if not os.path.exists('model'):
        os.makedirs('model')
    joblib.dump(best_model, model_filename)
    print(f"      > Best model saved to {model_filename}")

    print("\n✅ Pipeline Complete.")