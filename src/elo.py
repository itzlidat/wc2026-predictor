import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

K = 32
START_ELO = 1500
HOME_ADVANTAGE = 100


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def compute_elo(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process matches chronologically and return (match_df, final_ratings_df).

    match_df columns: date, home_team, away_team, home_elo_before,
        away_elo_before, home_elo_after, away_elo_after, result
    result: 1 = home win, 0 = draw, -1 = away win
    """
    df = df.sort_values("date").reset_index(drop=True)

    ratings: dict[str, float] = {}

    records = []
    for row in df.itertuples(index=False):
        home = row.home_team
        away = row.away_team

        home_before = ratings.get(home, START_ELO)
        away_before = ratings.get(away, START_ELO)

        # Apply home advantage only when the match is not at a neutral venue
        home_adj = home_before + (0 if row.neutral == "TRUE" else HOME_ADVANTAGE)

        exp_home = expected_score(home_adj, away_before)
        exp_away = 1 - exp_home

        if row.home_score > row.away_score:
            actual_home, actual_away = 1.0, 0.0
            result = 1
        elif row.home_score < row.away_score:
            actual_home, actual_away = 0.0, 1.0
            result = -1
        else:
            actual_home, actual_away = 0.5, 0.5
            result = 0

        home_after = home_before + K * (actual_home - exp_home)
        away_after = away_before + K * (actual_away - exp_away)

        ratings[home] = home_after
        ratings[away] = away_after

        records.append(
            {
                "date": row.date,
                "home_team": home,
                "away_team": away,
                "home_elo_before": round(home_before, 4),
                "away_elo_before": round(away_before, 4),
                "home_elo_after": round(home_after, 4),
                "away_elo_after": round(away_after, 4),
                "result": result,
            }
        )

    match_df = pd.DataFrame(records)

    final_ratings = (
        pd.DataFrame(
            [{"team": team, "elo": round(elo, 4)} for team, elo in ratings.items()]
        )
        .sort_values("elo", ascending=False)
        .reset_index(drop=True)
    )
    final_ratings.index += 1  # 1-based rank

    return match_df, final_ratings


def load_and_run() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(DATA_DIR / "results.csv", parse_dates=["date"])
    match_df, final_ratings = compute_elo(df)
    final_ratings.to_csv(DATA_DIR / "elo_ratings.csv", index_label="rank")
    return match_df, final_ratings


if __name__ == "__main__":
    match_df, final_ratings = load_and_run()
    print(f"Processed {len(match_df):,} matches")
    print(f"Tracked {len(final_ratings):,} teams\n")
    print("Top 20 teams by Elo rating:")
    print(final_ratings.head(20).to_string())
