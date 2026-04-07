#!/usr/bin/env python3
"""
NBA Dashboard Data Updater
Fetches games from NBA API, trains XGBoost model on existing data,
predicts upcoming games, and updates all 3 dashboard CSVs.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

from nba_api.stats.endpoints import ScoreboardV2, LeagueStandings, LeagueGameLog
from nba_api.stats.static import teams as nba_teams
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = 'data'
SEASON = 2025  # 2025-26 season
SEASON_STR = '2025-26'

TEAM_ID_TO_NAME = {
    1610612737: 'Atlanta Hawks', 1610612738: 'Boston Celtics',
    1610612751: 'Brooklyn Nets', 1610612766: 'Charlotte Hornets',
    1610612741: 'Chicago Bulls', 1610612739: 'Cleveland Cavaliers',
    1610612742: 'Dallas Mavericks', 1610612743: 'Denver Nuggets',
    1610612765: 'Detroit Pistons', 1610612744: 'Golden State Warriors',
    1610612745: 'Houston Rockets', 1610612754: 'Indiana Pacers',
    1610612746: 'LA Clippers', 1610612747: 'Los Angeles Lakers',
    1610612763: 'Memphis Grizzlies', 1610612748: 'Miami Heat',
    1610612749: 'Milwaukee Bucks', 1610612750: 'Minnesota Timberwolves',
    1610612740: 'New Orleans Pelicans', 1610612752: 'New York Knicks',
    1610612760: 'Oklahoma City Thunder', 1610612753: 'Orlando Magic',
    1610612755: 'Philadelphia 76ers', 1610612756: 'Phoenix Suns',
    1610612757: 'Portland Trail Blazers', 1610612758: 'Sacramento Kings',
    1610612759: 'San Antonio Spurs', 1610612761: 'Toronto Raptors',
    1610612762: 'Utah Jazz', 1610612764: 'Washington Wizards',
}

TEAM_NAME_TO_ID = {v: k for k, v in TEAM_ID_TO_NAME.items()}


def fetch_games_range(start_date, end_date):
    """Fetch all NBA games in a date range using LeagueGameLog."""
    from_str = start_date.strftime('%m/%d/%Y')
    to_str = end_date.strftime('%m/%d/%Y')

    print(f"  Fetching LeagueGameLog from {start_date.date()} to {end_date.date()}...")
    log = LeagueGameLog(season='2025-26', date_from_nullable=from_str, date_to_nullable=to_str)
    time.sleep(1)
    df = log.get_data_frames()[0]

    if df.empty:
        return []

    # Each game has 2 rows (one per team). Pair them by GAME_ID.
    games = []
    for game_id, group in df.groupby('GAME_ID'):
        if len(group) != 2:
            continue

        # Determine home vs away from MATCHUP ("vs." = home, "@" = away)
        home_row = group[group['MATCHUP'].str.contains(' vs. ')].iloc[0] if len(group[group['MATCHUP'].str.contains(' vs. ')]) > 0 else None
        away_row = group[group['MATCHUP'].str.contains(' @ ')].iloc[0] if len(group[group['MATCHUP'].str.contains(' @ ')]) > 0 else None

        if home_row is None or away_row is None:
            continue

        home_id = int(home_row['TEAM_ID'])
        visitor_id = int(away_row['TEAM_ID'])
        pts_home = int(home_row['PTS']) if pd.notna(home_row['PTS']) else None
        pts_away = int(away_row['PTS']) if pd.notna(away_row['PTS']) else None

        home_name = TEAM_ID_TO_NAME.get(home_id, home_row['TEAM_NAME'])
        visitor_name = TEAM_ID_TO_NAME.get(visitor_id, away_row['TEAM_NAME'])

        is_final = pts_home is not None and pts_away is not None

        games.append({
            'GAME_ID': game_id,
            'GAME_DATE': home_row['GAME_DATE'],
            'HOME_TEAM_ID': home_id,
            'VISITOR_TEAM_ID': visitor_id,
            'HOME_TEAM_NAME': home_name,
            'VISITOR_TEAM_NAME': visitor_name,
            'PTS_home': pts_home,
            'PTS_away': pts_away,
            'IS_FINAL': is_final,
        })

    # Sort by date
    games.sort(key=lambda x: x['GAME_DATE'])
    return games


def fetch_todays_schedule(game_date):
    """Fetch today's schedule (upcoming games with no scores yet) via ScoreboardV2."""
    date_str = game_date.strftime('%Y-%m-%d')
    try:
        sb = ScoreboardV2(game_date=date_str)
        time.sleep(0.7)

        header = sb.game_header.get_data_frame()
        if header.empty:
            return []

        games = []
        for _, game in header.iterrows():
            home_id = game['HOME_TEAM_ID']
            visitor_id = game['VISITOR_TEAM_ID']
            home_name = TEAM_ID_TO_NAME.get(home_id, str(home_id))
            visitor_name = TEAM_ID_TO_NAME.get(visitor_id, str(visitor_id))

            games.append({
                'GAME_ID': game['GAME_ID'],
                'GAME_DATE': date_str,
                'HOME_TEAM_ID': home_id,
                'VISITOR_TEAM_ID': visitor_id,
                'HOME_TEAM_NAME': home_name,
                'VISITOR_TEAM_NAME': visitor_name,
                'PTS_home': None,
                'PTS_away': None,
                'IS_FINAL': False,
            })

        return games
    except Exception as e:
        print(f"  Error fetching schedule for {date_str}: {e}")
        return []


def build_team_stats(games_df):
    """Build running team stats from completed games for feature engineering."""
    completed = games_df[games_df['GAME_STATUS'] == 'Completed'].copy()
    completed = completed.sort_values('GAME_DATE')

    team_stats = {}
    for team_id in TEAM_ID_TO_NAME:
        team_stats[team_id] = {
            'wins': 0, 'losses': 0, 'home_wins': 0, 'home_losses': 0,
            'away_wins': 0, 'away_losses': 0, 'pts_scored': [],
            'pts_allowed': [], 'last10': [], 'streak': 0,
        }

    for _, game in completed.iterrows():
        hid = game['HOME_TEAM_ID']
        vid = game['VISITOR_TEAM_ID']
        home_won = game['HOME_WINS'] == 1
        pts_h = game['PTS_home'] if pd.notna(game['PTS_home']) else 100
        pts_a = game['PTS_away'] if pd.notna(game['PTS_away']) else 100

        if hid in team_stats:
            ts = team_stats[hid]
            ts['pts_scored'].append(pts_h)
            ts['pts_allowed'].append(pts_a)
            if home_won:
                ts['wins'] += 1; ts['home_wins'] += 1
                ts['last10'].append(1)
                ts['streak'] = ts['streak'] + 1 if ts['streak'] > 0 else 1
            else:
                ts['losses'] += 1; ts['home_losses'] += 1
                ts['last10'].append(0)
                ts['streak'] = ts['streak'] - 1 if ts['streak'] < 0 else -1
            ts['last10'] = ts['last10'][-10:]

        if vid in team_stats:
            ts = team_stats[vid]
            ts['pts_scored'].append(pts_a)
            ts['pts_allowed'].append(pts_h)
            if not home_won:
                ts['wins'] += 1; ts['away_wins'] += 1
                ts['last10'].append(1)
                ts['streak'] = ts['streak'] + 1 if ts['streak'] > 0 else 1
            else:
                ts['losses'] += 1; ts['away_losses'] += 1
                ts['last10'].append(0)
                ts['streak'] = ts['streak'] - 1 if ts['streak'] < 0 else -1
            ts['last10'] = ts['last10'][-10:]

    return team_stats


def get_features_for_game(team_stats, home_id, visitor_id):
    """Generate features for a single game matchup."""
    h = team_stats.get(home_id, {})
    v = team_stats.get(visitor_id, {})

    h_games = h.get('wins', 0) + h.get('losses', 0)
    v_games = v.get('wins', 0) + v.get('losses', 0)

    h_wp = h['wins'] / max(h_games, 1)
    v_wp = v['wins'] / max(v_games, 1)

    h_home_wp = h['home_wins'] / max(h['home_wins'] + h['home_losses'], 1)
    v_away_wp = v['away_wins'] / max(v['away_wins'] + v['away_losses'], 1)

    h_ppg = np.mean(h['pts_scored'][-20:]) if h['pts_scored'] else 110
    h_papg = np.mean(h['pts_allowed'][-20:]) if h['pts_allowed'] else 110
    v_ppg = np.mean(v['pts_scored'][-20:]) if v['pts_scored'] else 110
    v_papg = np.mean(v['pts_allowed'][-20:]) if v['pts_allowed'] else 110

    h_l10 = np.mean(h.get('last10', [0.5])) if h.get('last10') else 0.5
    v_l10 = np.mean(v.get('last10', [0.5])) if v.get('last10') else 0.5

    h_net = h_ppg - h_papg
    v_net = v_ppg - v_papg

    return {
        'home_wp': h_wp,
        'visitor_wp': v_wp,
        'home_home_wp': h_home_wp,
        'visitor_away_wp': v_away_wp,
        'home_ppg': h_ppg,
        'home_papg': h_papg,
        'visitor_ppg': v_ppg,
        'visitor_papg': v_papg,
        'home_net': h_net,
        'visitor_net': v_net,
        'net_diff': h_net - v_net,
        'wp_diff': h_wp - v_wp,
        'home_l10': h_l10,
        'visitor_l10': v_l10,
        'l10_diff': h_l10 - v_l10,
        'home_streak': h.get('streak', 0),
        'visitor_streak': v.get('streak', 0),
    }


def train_model(games_df):
    """Train XGBoost model on existing completed games."""
    completed = games_df[games_df['GAME_STATUS'] == 'Completed'].copy()
    completed = completed.sort_values('GAME_DATE').reset_index(drop=True)

    print("Training XGBoost model on existing data...")

    # Build features incrementally (using only data before each game)
    feature_rows = []
    labels = []

    # We need running stats, so process game by game
    running_stats = {}
    for team_id in TEAM_ID_TO_NAME:
        running_stats[team_id] = {
            'wins': 0, 'losses': 0, 'home_wins': 0, 'home_losses': 0,
            'away_wins': 0, 'away_losses': 0, 'pts_scored': [],
            'pts_allowed': [], 'last10': [], 'streak': 0,
        }

    for idx, game in completed.iterrows():
        hid = game['HOME_TEAM_ID']
        vid = game['VISITOR_TEAM_ID']
        home_won = game['HOME_WINS'] == 1

        # Only use games after we have enough data (skip first ~50 games)
        h_games = running_stats[hid]['wins'] + running_stats[hid]['losses']
        v_games = running_stats[vid]['wins'] + running_stats[vid]['losses']

        if h_games >= 5 and v_games >= 5:
            feats = get_features_for_game(running_stats, hid, vid)
            feature_rows.append(feats)
            labels.append(1 if home_won else 0)

        # Update running stats
        pts_h = game['PTS_home'] if pd.notna(game['PTS_home']) else 100
        pts_a = game['PTS_away'] if pd.notna(game['PTS_away']) else 100

        rs = running_stats[hid]
        rs['pts_scored'].append(pts_h); rs['pts_allowed'].append(pts_a)
        if home_won:
            rs['wins'] += 1; rs['home_wins'] += 1; rs['last10'].append(1)
            rs['streak'] = rs['streak'] + 1 if rs['streak'] > 0 else 1
        else:
            rs['losses'] += 1; rs['home_losses'] += 1; rs['last10'].append(0)
            rs['streak'] = rs['streak'] - 1 if rs['streak'] < 0 else -1
        rs['last10'] = rs['last10'][-10:]

        rs = running_stats[vid]
        rs['pts_scored'].append(pts_a); rs['pts_allowed'].append(pts_h)
        if not home_won:
            rs['wins'] += 1; rs['away_wins'] += 1; rs['last10'].append(1)
            rs['streak'] = rs['streak'] + 1 if rs['streak'] > 0 else 1
        else:
            rs['losses'] += 1; rs['away_losses'] += 1; rs['last10'].append(0)
            rs['streak'] = rs['streak'] - 1 if rs['streak'] < 0 else -1
        rs['last10'] = rs['last10'][-10:]

    X = pd.DataFrame(feature_rows)
    y = np.array(labels)

    print(f"  Training samples: {len(X)}")

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
    )
    model.fit(X, y)

    # Cross-validate
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"  CV Accuracy: {cv_scores.mean():.1%} (+/- {cv_scores.std():.1%})")

    return model, running_stats


def compute_running_metrics(games_df):
    """Compute running accuracy metrics for each team and overall."""
    completed = games_df[games_df['GAME_STATUS'] == 'Completed'].copy()
    completed = completed.sort_values('GAME_DATE')

    team_correct = {}
    team_total = {}
    overall_correct = 0
    overall_total = 0

    metrics_rows = []

    for _, game in completed.iterrows():
        date = game['GAME_DATE']
        hid = game['HOME_TEAM_ID']
        vid = game['VISITOR_TEAM_ID']
        h_name = game['HOME_TEAM_NAME']
        v_name = game['VISITOR_TEAM_NAME']
        correct = game['CORRECT']

        overall_total += 1
        overall_correct += 1 if correct else 0

        for tid, tname in [(hid, h_name), (vid, v_name)]:
            team_total[tid] = team_total.get(tid, 0) + 1
            team_correct[tid] = team_correct.get(tid, 0) + (1 if correct else 0)

            metrics_rows.append({
                'GAME_DATE': date,
                'METRIC_TYPE': 'TEAM',
                'METRIC_VALUE': team_correct[tid] / team_total[tid],
                'TEAM_ID': float(tid),
                'TEAM_NAME': tname,
            })

        metrics_rows.append({
            'GAME_DATE': date,
            'METRIC_TYPE': 'OVERALL',
            'METRIC_VALUE': overall_correct / overall_total,
            'TEAM_ID': np.nan,
            'TEAM_NAME': np.nan,
        })

        # 7-day rolling average
        recent = completed[
            (completed['GAME_DATE'] >= (pd.Timestamp(date) - timedelta(days=7))) &
            (completed['GAME_DATE'] <= pd.Timestamp(date))
        ]
        if len(recent) > 0:
            avg_7d = recent['CORRECT'].mean()
        else:
            avg_7d = overall_correct / overall_total

        metrics_rows.append({
            'GAME_DATE': date,
            'METRIC_TYPE': 'OVERALL_7_DAY_AVG',
            'METRIC_VALUE': avg_7d,
            'TEAM_ID': np.nan,
            'TEAM_NAME': np.nan,
        })

    return pd.DataFrame(metrics_rows)


def main():
    print("=" * 60)
    print("NBA Dashboard Data Updater")
    print("=" * 60)

    # Load existing data
    games_df = pd.read_csv(f'{DATA_DIR}/games_dashboard.csv')
    games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'])

    last_date = games_df['GAME_DATE'].max()
    today = pd.Timestamp(datetime.now().date())

    print(f"\nExisting data: {games_df['GAME_DATE'].min().date()} to {last_date.date()}")
    print(f"Today: {today.date()}")

    if last_date >= today:
        print("Data is already up to date!")
        # Still train model for today's upcoming games

    # Step 1: Fetch new games from NBA API
    new_games = []
    fetch_start = last_date + timedelta(days=1)

    print(f"\n--- Fetching games from {fetch_start.date()} to {today.date()} ---")

    # Fetch completed games (yesterday and before) via LeagueGameLog
    yesterday = today - timedelta(days=1)
    if fetch_start <= yesterday:
        completed_games = fetch_games_range(fetch_start, yesterday)
        print(f"  Completed games fetched: {len(completed_games)}")
        new_games.extend(completed_games)

    # Fetch today's schedule (may be upcoming or in-progress)
    print(f"  Fetching today's schedule ({today.date()})...")
    today_games = fetch_todays_schedule(today)
    # Check if any of today's games already have scores in LeagueGameLog
    today_completed = fetch_games_range(today, today)
    today_completed_ids = {g['GAME_ID'] for g in today_completed}

    for g in today_games:
        if g['GAME_ID'] in today_completed_ids:
            # Use the completed version with scores
            for cg in today_completed:
                if cg['GAME_ID'] == g['GAME_ID']:
                    new_games.append(cg)
                    break
        else:
            new_games.append(g)

    print(f"  Today's games: {len(today_games)}")
    print(f"\nTotal new games: {len(new_games)}")

    # Step 2: Train model on existing data
    model, team_stats = train_model(games_df)

    # Step 3: Process new games — add predictions and results
    if new_games:
        print(f"\n--- Processing {len(new_games)} new games ---")

        # Track running accuracy from existing data
        existing_completed = games_df[games_df['GAME_STATUS'] == 'Completed']
        total_correct = existing_completed['CORRECT'].sum()
        total_games = len(existing_completed)

        # Team running accuracy
        team_acc = {}
        for _, g in existing_completed.iterrows():
            for tid in [g['HOME_TEAM_ID'], g['VISITOR_TEAM_ID']]:
                if tid not in team_acc:
                    team_acc[tid] = {'correct': 0, 'total': 0}
                team_acc[tid]['total'] += 1
                team_acc[tid]['correct'] += 1 if g['CORRECT'] else 0

        new_rows = []
        for game in new_games:
            hid = game['HOME_TEAM_ID']
            vid = game['VISITOR_TEAM_ID']

            # Generate prediction
            feats = get_features_for_game(team_stats, hid, vid)
            feat_df = pd.DataFrame([feats])
            home_win_prob = float(model.predict_proba(feat_df)[0][1])

            pred_home_wins = 1 if home_win_prob > 0.5 else 0

            if game['IS_FINAL'] and game['PTS_home'] is not None:
                actual_home_wins = 1 if game['PTS_home'] > game['PTS_away'] else 0
                correct = (pred_home_wins == actual_home_wins)
                status = 'Completed'
                score_diff = (game['PTS_home'] or 0) - (game['PTS_away'] or 0)

                # Update running accuracy
                total_games += 1
                total_correct += 1 if correct else 0
                overall_acc = total_correct / total_games

                for tid in [hid, vid]:
                    if tid not in team_acc:
                        team_acc[tid] = {'correct': 0, 'total': 0}
                    team_acc[tid]['total'] += 1
                    team_acc[tid]['correct'] += 1 if correct else 0

                h_acc = team_acc[hid]['correct'] / team_acc[hid]['total']
                v_acc = team_acc[vid]['correct'] / team_acc[vid]['total']

                # Update team stats for future games
                pts_h = game['PTS_home']
                pts_a = game['PTS_away']
                home_won = actual_home_wins == 1

                ts = team_stats[hid]
                ts['pts_scored'].append(pts_h); ts['pts_allowed'].append(pts_a)
                if home_won:
                    ts['wins'] += 1; ts['home_wins'] += 1; ts['last10'].append(1)
                    ts['streak'] = ts['streak'] + 1 if ts['streak'] > 0 else 1
                else:
                    ts['losses'] += 1; ts['home_losses'] += 1; ts['last10'].append(0)
                    ts['streak'] = ts['streak'] - 1 if ts['streak'] < 0 else -1
                ts['last10'] = ts['last10'][-10:]

                ts = team_stats[vid]
                ts['pts_scored'].append(pts_a); ts['pts_allowed'].append(pts_h)
                if not home_won:
                    ts['wins'] += 1; ts['away_wins'] += 1; ts['last10'].append(1)
                    ts['streak'] = ts['streak'] + 1 if ts['streak'] > 0 else 1
                else:
                    ts['losses'] += 1; ts['away_losses'] += 1; ts['last10'].append(0)
                    ts['streak'] = ts['streak'] - 1 if ts['streak'] < 0 else -1
                ts['last10'] = ts['last10'][-10:]
            else:
                actual_home_wins = None
                correct = None
                status = 'Upcoming'
                score_diff = None
                overall_acc = total_correct / max(total_games, 1)
                h_acc = team_acc.get(hid, {}).get('correct', 0) / max(team_acc.get(hid, {}).get('total', 1), 1)
                v_acc = team_acc.get(vid, {}).get('correct', 0) / max(team_acc.get(vid, {}).get('total', 1), 1)

            matchup = f"{game['VISITOR_TEAM_NAME']} @ {game['HOME_TEAM_NAME']}"
            confidence = abs(home_win_prob - 0.5)

            # Check if today
            is_recent = game['GAME_DATE'] == today.strftime('%Y-%m-%d')

            new_rows.append({
                'GAME_ID': game['GAME_ID'],
                'GAME_DATE': game['GAME_DATE'],
                'SEASON': SEASON,
                'GAME_STATUS': status,
                'HOME_TEAM_ID': hid,
                'VISITOR_TEAM_ID': vid,
                'MATCHUP': matchup,
                'HOME_WINS': actual_home_wins if status == 'Completed' else pred_home_wins,
                'PTS_home': game['PTS_home'],
                'PTS_away': game['PTS_away'],
                'HOME_WIN_PROB': home_win_prob,
                'CORRECT': correct if correct is not None else '',
                'HOME_TEAM_RUNNING_ACCURACY': h_acc,
                'VISITOR_TEAM_RUNNING_ACCURACY': v_acc,
                'HOME_ROLE_RUNNING_ACCURACY': overall_acc,
                'AWAY_ROLE_RUNNING_ACCURACY': overall_acc,
                'OVERALL_RUNNING_ACCURACY': overall_acc,
                'RECENT_FLAG': is_recent,
                'HOME_TEAM_NAME': game['HOME_TEAM_NAME'],
                'VISITOR_TEAM_NAME': game['VISITOR_TEAM_NAME'],
                'SCORE_DIFF': score_diff,
                'PREDICTION_CONFIDENCE': confidence,
            })

            winner = game['HOME_TEAM_NAME'] if home_win_prob > 0.5 else game['VISITOR_TEAM_NAME']
            prob = max(home_win_prob, 1 - home_win_prob)
            result_str = ""
            if status == 'Completed':
                result_str = f" | {'✓' if correct else '✗'} ({game['PTS_home']}-{game['PTS_away']})"
            print(f"  {matchup}: {winner} ({prob:.1%}){result_str}")

        new_df = pd.DataFrame(new_rows)
        games_df = pd.concat([games_df, new_df], ignore_index=True)

    # Step 4: Update RECENT_FLAG — only today's games are recent
    games_df['RECENT_FLAG'] = games_df['GAME_DATE'] == today

    # Normalize GAME_DATE to consistent format
    games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'])

    # Step 5: Save updated games CSV
    games_df.to_csv(f'{DATA_DIR}/games_dashboard.csv', index=False)
    print(f"\n--- Saved games_dashboard.csv ({len(games_df)} total games) ---")

    # Step 6: Recompute running accuracy metrics
    print("\nComputing running accuracy metrics...")
    metrics_df = compute_running_metrics(games_df)
    metrics_df.to_csv(f'{DATA_DIR}/running_accuracy_metrics.csv', index=False)
    print(f"  Saved running_accuracy_metrics.csv ({len(metrics_df)} rows)")

    # Step 7: Update season summary
    completed = games_df[games_df['GAME_STATUS'] == 'Completed']
    total = len(completed)
    correct = completed['CORRECT'].sum()
    accuracy = correct / max(total, 1)
    home_wins = (completed['HOME_WINS'] == 1).sum()

    summary = pd.DataFrame([{
        'SEASON': SEASON,
        'TOTAL_GAMES': total,
        'CORRECT_PREDICTIONS': int(correct),
        'ACCURACY': accuracy,
        'HOME_TEAM_WINS': int(home_wins),
        'HOME_TEAM_WIN_PCT': home_wins / max(total, 1),
    }])
    summary.to_csv(f'{DATA_DIR}/season_summary_stats.csv', index=False)
    print(f"  Saved season_summary_stats.csv")

    print(f"\n{'=' * 60}")
    print(f"DONE — {total} completed games, {accuracy:.1%} accuracy")
    dates = pd.to_datetime(games_df['GAME_DATE'])
    print(f"Date range: {dates.min().date()} to {dates.max().date()}")

    upcoming = games_df[games_df['GAME_STATUS'] == 'Upcoming']
    if len(upcoming) > 0:
        print(f"\nToday's predictions ({len(upcoming)} games):")
        for _, g in upcoming.iterrows():
            winner = g['HOME_TEAM_NAME'] if g['HOME_WIN_PROB'] > 0.5 else g['VISITOR_TEAM_NAME']
            prob = max(g['HOME_WIN_PROB'], 1 - g['HOME_WIN_PROB'])
            print(f"  {g['MATCHUP']}: {winner} ({prob:.1%})")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
