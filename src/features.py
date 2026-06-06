import sys
import pandas as pd
from pathlib import Path
from collections import defaultdict, deque

sys.path.insert(0, str(Path(__file__).parent))
from elo import compute_elo, DATA_DIR


def _win_rate(history: deque) -> float:
    """Fraction of wins in history; 0.5 if no history (neutral prior)."""
    return sum(history) / len(history) if history else 0.5


def build_features() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "results.csv", parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Reuse the chronological Elo engine for before-match ratings
    match_elo, _ = compute_elo(df)

    base = df[["date", "home_team", "away_team", "home_score", "away_score", "neutral"]].copy()
    merged = base.merge(
        match_elo[["date", "home_team", "away_team", "home_elo_before", "away_elo_before"]],
        on=["date", "home_team", "away_team"],
        how="left",
    ).sort_values("date").reset_index(drop=True)

    # team -> deque[int]  (1=won that match, 0=did not win)
    team_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
    # frozenset({home, away}) -> deque of winning team name or None for draw
    h2h_history: dict[frozenset, deque] = defaultdict(lambda: deque(maxlen=10))

    records = []
    for row in merged.itertuples(index=False):
        home, away = row.home_team, row.away_team
        pair = frozenset([home, away])

        home_form = _win_rate(team_history[home])
        away_form = _win_rate(team_history[away])

        h2h = h2h_history[pair]
        h2h_win_rate = (
            sum(1 for w in h2h if w == home) / len(h2h) if h2h else 0.5
        )

        elo_diff = row.home_elo_before - row.away_elo_before

        if row.home_score > row.away_score:
            target = 2
            h2h_winner = home
        elif row.home_score < row.away_score:
            target = 0
            h2h_winner = away
        else:
            target = 1
            h2h_winner = None

        records.append(
            {
                "date": row.date,
                "home_team": home,
                "away_team": away,
                "elo_diff": round(elo_diff, 4),
                "home_form": round(home_form, 4),
                "away_form": round(away_form, 4),
                "h2h_win_rate": round(h2h_win_rate, 4),
                "is_neutral": 1 if row.neutral == "TRUE" else 0,
                "target": target,
            }
        )

        # Update rolling state AFTER recording (no data leakage)
        team_history[home].append(1 if row.home_score > row.away_score else 0)
        team_history[away].append(1 if row.away_score > row.home_score else 0)
        h2h_history[pair].append(h2h_winner)

    features = pd.DataFrame(records)

    # Filter to modern era; process full history above for accurate form/h2h
    features = features[features["date"] >= "1990-01-01"].reset_index(drop=True)
    return features


if __name__ == "__main__":
    features = build_features()
    features.to_csv(DATA_DIR / "features.csv", index=False)

    print(f"Shape: {features.shape}")
    print(f"\nTarget distribution:\n{features['target'].value_counts().sort_index().rename({0: 'away_win', 1: 'draw', 2: 'home_win'})}")
    print(f"\nSample (5 rows):")
    print(features.sample(5, random_state=42).to_string(index=False))
