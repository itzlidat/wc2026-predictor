import sys
import pandas as pd
from itertools import combinations
from pathlib import Path
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from elo import DATA_DIR, START_ELO
from features import get_current_state, FEATURE_COLS, _win_rate

PREDICTIONS_DIR = Path(__file__).parent.parent / "predictions"
MODEL_PATH = DATA_DIR / "model.json"

# 2026 FIFA World Cup group draw (Miami, December 5 2024)
GROUPS: dict[str, list[str]] = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}


def build_match_features(
    team1: str,
    team2: str,
    ratings: dict,
    rank_map: dict,
    team_history: dict,
    h2h_history: dict,
) -> dict:
    elo1 = ratings.get(team1, START_ELO)
    elo2 = ratings.get(team2, START_ELO)
    pair = frozenset([team1, team2])
    h2h = h2h_history.get(pair)
    if h2h:
        h2h_win_rate = sum(1 for w in h2h if w == team1) / len(h2h)
    else:
        h2h_win_rate = 0.5
    return {
        "elo_diff": round(elo1 - elo2, 4),
        "home_elo_rank": rank_map.get(team1, len(rank_map)),
        "away_elo_rank": rank_map.get(team2, len(rank_map)),
        "home_form": round(_win_rate(team_history[team1]), 4),
        "away_form": round(_win_rate(team_history[team2]), 4),
        "h2h_win_rate": round(h2h_win_rate, 4),
        "is_neutral": 1,
    }


def predict_group_stage() -> pd.DataFrame:
    model = XGBClassifier()
    model.load_model(MODEL_PATH)

    ratings, team_history, h2h_history = get_current_state()

    # Build rank lookup from ratings (1 = highest Elo)
    sorted_teams = sorted(ratings, key=ratings.get, reverse=True)
    rank_map = {team: rank + 1 for rank, team in enumerate(sorted_teams)}

    records = []
    for group, teams in GROUPS.items():
        for team1, team2 in combinations(teams, 2):
            feats = build_match_features(
                team1, team2, ratings, rank_map, team_history, h2h_history
            )
            X = pd.DataFrame([feats])[FEATURE_COLS]
            # probs: [away_win, draw, home_win] (class 0, 1, 2)
            probs = model.predict_proba(X)[0]
            away_p, draw_p, home_p = probs[0], probs[1], probs[2]

            if home_p >= away_p and home_p >= draw_p:
                winner = team1
            elif away_p >= home_p and away_p >= draw_p:
                winner = team2
            else:
                winner = "Draw"

            records.append(
                {
                    "group": group,
                    "home_team": team1,
                    "away_team": team2,
                    "home_win_prob": round(home_p, 4),
                    "draw_prob": round(draw_p, 4),
                    "away_win_prob": round(away_p, 4),
                    "predicted_winner": winner,
                }
            )

    return pd.DataFrame(records)


def print_results(df: pd.DataFrame) -> None:
    for group in sorted(df["group"].unique()):
        print(f"\n{'─'*60}")
        print(f"  GROUP {group}")
        print(f"{'─'*60}")
        grp = df[df["group"] == group]
        for _, row in grp.iterrows():
            bar_h = "█" * int(row["home_win_prob"] * 20)
            bar_a = "█" * int(row["away_win_prob"] * 20)
            print(
                f"  {row['home_team']:>24}  {row['home_win_prob']:.0%} {bar_h:<20}"
                f"  D:{row['draw_prob']:.0%}"
                f"  {bar_a:<20} {row['away_win_prob']:.0%}  {row['away_team']}"
            )
            print(f"  {'→ ' + row['predicted_winner']:>28}")


if __name__ == "__main__":
    print("Computing live state from historical matches...")
    results = predict_group_stage()
    results.to_csv(PREDICTIONS_DIR / "group_stage.csv", index=False)
    print(f"Saved {len(results)} match predictions to predictions/group_stage.csv")
    print_results(results)
