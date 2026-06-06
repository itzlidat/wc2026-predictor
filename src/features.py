import sys
import pandas as pd
from pathlib import Path
from collections import defaultdict, deque

sys.path.insert(0, str(Path(__file__).parent))
from elo import expected_score, DATA_DIR, K, START_ELO, HOME_ADVANTAGE

FEATURE_COLS = [
    "elo_diff",
    "home_elo_rank",
    "away_elo_rank",
    "home_form",
    "away_form",
    "h2h_win_rate",
    "is_neutral",
]


def _win_rate(history: deque) -> float:
    return sum(history) / len(history) if history else 0.5


def _accumulate_state(df: pd.DataFrame) -> tuple:
    """
    Single chronological pass over all matches.
    Returns (ratings, team_history, h2h_history, records).
    records contains one dict per match with all feature columns + metadata.
    """
    ratings: dict[str, float] = {}
    team_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
    h2h_history: dict[frozenset, deque] = defaultdict(lambda: deque(maxlen=10))

    records = []
    for row in df.itertuples(index=False):
        home, away = row.home_team, row.away_team
        pair = frozenset([home, away])

        home_elo = ratings.get(home, START_ELO)
        away_elo = ratings.get(away, START_ELO)

        # Rank = 1 + number of rated teams strictly above this team's Elo
        rated_elos = list(ratings.values())
        home_elo_rank = 1 + sum(1 for e in rated_elos if e > home_elo)
        away_elo_rank = 1 + sum(1 for e in rated_elos if e > away_elo)

        home_form = _win_rate(team_history[home])
        away_form = _win_rate(team_history[away])

        h2h = h2h_history[pair]
        h2h_win_rate = sum(1 for w in h2h if w == home) / len(h2h) if h2h else 0.5

        if row.home_score > row.away_score:
            target, h2h_winner, actual_home = 2, home, 1.0
        elif row.home_score < row.away_score:
            target, h2h_winner, actual_home = 0, away, 0.0
        else:
            target, h2h_winner, actual_home = 1, None, 0.5

        records.append(
            {
                "date": row.date,
                "home_team": home,
                "away_team": away,
                "elo_diff": round(home_elo - away_elo, 4),
                "home_elo_rank": home_elo_rank,
                "away_elo_rank": away_elo_rank,
                "home_form": round(home_form, 4),
                "away_form": round(away_form, 4),
                "h2h_win_rate": round(h2h_win_rate, 4),
                "is_neutral": 1 if row.neutral == "TRUE" else 0,
                "target": target,
            }
        )

        # Update state after recording (no leakage)
        home_adj = home_elo + (0 if row.neutral == "TRUE" else HOME_ADVANTAGE)
        exp_home = expected_score(home_adj, away_elo)
        ratings[home] = home_elo + K * (actual_home - exp_home)
        ratings[away] = away_elo + K * ((1 - actual_home) - (1 - exp_home))

        team_history[home].append(1 if row.home_score > row.away_score else 0)
        team_history[away].append(1 if row.away_score > row.home_score else 0)
        h2h_history[pair].append(h2h_winner)

    return ratings, team_history, h2h_history, records


def get_current_state() -> tuple[dict, dict, dict]:
    """
    Return (ratings, team_history, h2h_history) after processing all
    historical matches — used by predict.py to build live features.
    """
    df = pd.read_csv(DATA_DIR / "results.csv", parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    ratings, team_history, h2h_history, _ = _accumulate_state(df)
    return ratings, team_history, h2h_history


def build_features() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "results.csv", parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    _, _, _, records = _accumulate_state(df)
    features = pd.DataFrame(records)
    # Filter to modern era; full history above ensures accurate state
    features = features[features["date"] >= "1990-01-01"].reset_index(drop=True)
    return features


if __name__ == "__main__":
    features = build_features()
    features.to_csv(DATA_DIR / "features.csv", index=False)

    print(f"Shape: {features.shape}")
    print(f"\nColumns: {list(features.columns)}")
    print(f"\nTarget distribution:")
    print(
        features["target"]
        .value_counts()
        .sort_index()
        .rename({0: "away_win", 1: "draw", 2: "home_win"})
    )
    print(f"\nSample (5 rows):")
    print(features.sample(5, random_state=42).to_string(index=False))
