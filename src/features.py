import sys
import bisect
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, deque

sys.path.insert(0, str(Path(__file__).parent))
from elo import expected_score, DATA_DIR, K, START_ELO, HOME_ADVANTAGE

FIFA_CSV = DATA_DIR / "kaggle" / "fifa_ranking-2024-06-20.csv"

FEATURE_COLS = [
    "elo_diff",
    "home_elo_rank",
    "away_elo_rank",
    "home_fifa_rank",
    "away_fifa_rank",
    "home_form",
    "away_form",
    "h2h_win_rate",
    "home_goals_scored_avg",
    "home_goals_conceded_avg",
    "away_goals_scored_avg",
    "away_goals_conceded_avg",
    "is_neutral",
]

# Maps FIFA ranking CSV country_full → results.csv team name
FIFA_NAME_MAP = {
    "Korea Republic":              "South Korea",
    "Czechia":                     "Czech Republic",
    "USA":                         "United States",
    "Curacao":                     "Curaçao",
    "Côte d'Ivoire":               "Ivory Coast",
    "IR Iran":                     "Iran",
    "Cabo Verde":                  "Cape Verde",
    "Congo DR":                    "DR Congo",
    "China PR":                    "China",
    "Korea DPR":                   "North Korea",
    "St Kitts and Nevis":          "Saint Kitts and Nevis",
    "St Lucia":                    "Saint Lucia",
    "St Vincent and the Grenadines": "Saint Vincent and the Grenadines",
    "Sao Tome and Principe":       "São Tomé and Príncipe",
    "Guinea-Bissau":               "Guinea-Bissau",
}

# Default rank for teams absent from FIFA rankings (unranked / pre-1992 matches)
FIFA_DEFAULT_RANK = 200


def _win_rate(history: deque) -> float:
    return sum(history) / len(history) if history else 0.5


def _avg(history: deque, default: float) -> float:
    return sum(history) / len(history) if history else default


def _load_fifa_lookup() -> dict:
    """
    Returns dict: mapped_team_name -> (sorted int64 timestamps, rank array).
    Timestamps are nanoseconds since epoch (pandas .value) for fast bisect.
    """
    if not FIFA_CSV.exists():
        return {}
    fifa = pd.read_csv(FIFA_CSV, parse_dates=["rank_date"])
    lookup: dict[str, tuple] = {}
    for team, grp in fifa.groupby("country_full"):
        mapped = FIFA_NAME_MAP.get(team, team)
        grp = grp.sort_values("rank_date")
        lookup[mapped] = (
            grp["rank_date"].values.astype(np.int64),
            grp["rank"].fillna(FIFA_DEFAULT_RANK).values.astype(np.int32),
        )
    return lookup


def _get_fifa_rank(lookup: dict, team: str, ts_ns: int) -> int:
    if team not in lookup:
        return FIFA_DEFAULT_RANK
    dates, ranks = lookup[team]
    idx = bisect.bisect_right(dates, ts_ns) - 1
    return int(ranks[idx]) if idx >= 0 else FIFA_DEFAULT_RANK


def _accumulate_state(df: pd.DataFrame, fifa_lookup: dict) -> tuple:
    """
    Single chronological pass. Returns:
      (ratings, team_history, h2h_history, goals_scored, goals_conceded, records)

    NOTE: ginf.csv betting odds (odd_h/d/a) cover European club fixtures only —
    zero overlap with international results.csv — so odds features are omitted.
    """
    ratings: dict[str, float] = {}
    team_history:  dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
    h2h_history:   dict[frozenset, deque] = defaultdict(lambda: deque(maxlen=10))
    goals_scored:  dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
    goals_conceded: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))

    # Drop rows with no score — unplayed/future fixtures have NaN scores
    # and would corrupt rolling state and elo updates.
    df = df.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)

    records = []
    for row in df.itertuples(index=False):
        home, away = row.home_team, row.away_team
        pair  = frozenset([home, away])
        ts_ns = row.date.value  # int64 ns timestamp for bisect

        home_elo = ratings.get(home, START_ELO)
        away_elo = ratings.get(away, START_ELO)

        rated_elos    = list(ratings.values())
        home_elo_rank = 1 + sum(1 for e in rated_elos if e > home_elo)
        away_elo_rank = 1 + sum(1 for e in rated_elos if e > away_elo)

        home_fifa_rank = _get_fifa_rank(fifa_lookup, home, ts_ns)
        away_fifa_rank = _get_fifa_rank(fifa_lookup, away, ts_ns)

        home_form    = _win_rate(team_history[home])
        away_form    = _win_rate(team_history[away])

        h2h = h2h_history[pair]
        h2h_win_rate = sum(1 for w in h2h if w == home) / len(h2h) if h2h else 0.5

        home_goals_scored_avg   = _avg(goals_scored[home],   default=1.0)
        home_goals_conceded_avg = _avg(goals_conceded[home],  default=1.0)
        away_goals_scored_avg   = _avg(goals_scored[away],   default=1.0)
        away_goals_conceded_avg = _avg(goals_conceded[away],  default=1.0)

        if row.home_score > row.away_score:
            target, h2h_winner, actual_home = 2, home, 1.0
        elif row.home_score < row.away_score:
            target, h2h_winner, actual_home = 0, away, 0.0
        else:
            target, h2h_winner, actual_home = 1, None, 0.5

        records.append({
            "date":                   row.date,
            "home_team":              home,
            "away_team":              away,
            "elo_diff":               round(home_elo - away_elo, 4),
            "home_elo_rank":          home_elo_rank,
            "away_elo_rank":          away_elo_rank,
            "home_fifa_rank":         home_fifa_rank,
            "away_fifa_rank":         away_fifa_rank,
            "home_form":              round(home_form, 4),
            "away_form":              round(away_form, 4),
            "h2h_win_rate":           round(h2h_win_rate, 4),
            "home_goals_scored_avg":  round(home_goals_scored_avg, 4),
            "home_goals_conceded_avg":round(home_goals_conceded_avg, 4),
            "away_goals_scored_avg":  round(away_goals_scored_avg, 4),
            "away_goals_conceded_avg":round(away_goals_conceded_avg, 4),
            "is_neutral":             1 if row.neutral == "TRUE" else 0,
            "is_world_cup":           1 if row.tournament == "FIFA World Cup" else 0,
            "target":                 target,
        })

        # --- update state after recording (no leakage) ---
        home_adj  = home_elo + (0 if row.neutral == "TRUE" else HOME_ADVANTAGE)
        exp_home  = expected_score(home_adj, away_elo)
        ratings[home] = home_elo + K * (actual_home - exp_home)
        ratings[away] = away_elo + K * ((1 - actual_home) - (1 - exp_home))

        team_history[home].append(1 if row.home_score > row.away_score else 0)
        team_history[away].append(1 if row.away_score > row.home_score else 0)
        h2h_history[pair].append(h2h_winner)

        goals_scored[home].append(row.home_score)
        goals_conceded[home].append(row.away_score)
        goals_scored[away].append(row.away_score)
        goals_conceded[away].append(row.home_score)

    return ratings, team_history, h2h_history, goals_scored, goals_conceded, records


def get_current_state() -> dict:
    """Return all live state after the full match history — used by predict.py."""
    df = pd.read_csv(DATA_DIR / "results.csv", parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    fifa_lookup = _load_fifa_lookup()
    ratings, team_history, h2h_history, goals_scored, goals_conceded, _ = \
        _accumulate_state(df, fifa_lookup)
    return {
        "ratings":       ratings,
        "team_history":  team_history,
        "h2h_history":   h2h_history,
        "goals_scored":  goals_scored,
        "goals_conceded": goals_conceded,
        "fifa_lookup":   fifa_lookup,
    }


def build_features() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "results.csv", parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    fifa_lookup = _load_fifa_lookup()
    _, _, _, _, _, records = _accumulate_state(df, fifa_lookup)
    features = pd.DataFrame(records)
    features = features[features["date"] >= "1990-01-01"].reset_index(drop=True)
    return features


if __name__ == "__main__":
    features = build_features()
    features.to_csv(DATA_DIR / "features.csv", index=False)

    print(f"Shape: {features.shape}")
    print(f"Columns: {list(features.columns)}")

    covered = (features["home_fifa_rank"] != FIFA_DEFAULT_RANK).mean()
    print(f"\nFIFA rank coverage: {covered:.1%} of rows have a real ranking")

    print(f"\nTarget distribution:")
    print(features["target"].value_counts().sort_index()
          .rename({0: "away_win", 1: "draw", 2: "home_win"}))

    print(f"\nGoals feature sample stats:")
    print(features[["home_goals_scored_avg","home_goals_conceded_avg",
                     "away_goals_scored_avg","away_goals_conceded_avg"]].describe().round(3))
