import pandas as pd


FEATURE_COLUMNS = [
    "home_recent_points",
    "away_recent_points",
    "home_recent_goals_scored",
    "away_recent_goals_scored",
    "home_recent_goals_conceded",
    "away_recent_goals_conceded",
    "home_venue_points",
    "away_venue_points",
]


def get_team_history(matches, team, season, before_date):
    """Return a team's same-season matches before the prediction date."""
    history = matches[
        (matches["Season"] == season)
        & (matches["Date"] < before_date)
        & (
            (matches["HomeTeam"] == team)
            | (matches["AwayTeam"] == team)
        )
    ].copy()

    history = history.sort_values("Date")

    # Express each historical match from this team's perspective.
    at_home = history["HomeTeam"] == team

    history["goals_scored"] = history["FTHG"].where(
        at_home, history["FTAG"]
    )
    history["goals_conceded"] = history["FTAG"].where(
        at_home, history["FTHG"]
    )

    history["points"] = 0
    history.loc[
        history["goals_scored"] == history["goals_conceded"],
        "points",
    ] = 1
    history.loc[
        history["goals_scored"] > history["goals_conceded"],
        "points",
    ] = 3

    return history


def build_features(matches):
    """Create pre-match features for fixtures with enough history."""
    rows = []

    for _, match in matches.iterrows():
        home_history = get_team_history(
            matches, match["HomeTeam"], match["Season"], match["Date"]
        )
        away_history = get_team_history(
            matches, match["AwayTeam"], match["Season"], match["Date"]
        )

        home_last5 = home_history.tail(5)
        away_last5 = away_history.tail(5)

        home_last3_at_home = home_history[
            home_history["HomeTeam"] == match["HomeTeam"]
        ].tail(3)

        away_last3_away = away_history[
            away_history["AwayTeam"] == match["AwayTeam"]
        ].tail(3)

        # Skip fixtures without the complete history we agreed to use.
        if (
            len(home_last5) < 5
            or len(away_last5) < 5
            or len(home_last3_at_home) < 3
            or len(away_last3_away) < 3
        ):
            continue

        rows.append({
            "Date": match["Date"],
            "Season": match["Season"],
            "HomeTeam": match["HomeTeam"],
            "AwayTeam": match["AwayTeam"],
            "FTR": match["FTR"],
            "home_recent_points": home_last5["points"].mean(),
            "away_recent_points": away_last5["points"].mean(),
            "home_recent_goals_scored": home_last5["goals_scored"].mean(),
            "away_recent_goals_scored": away_last5["goals_scored"].mean(),
            "home_recent_goals_conceded": home_last5["goals_conceded"].mean(),
            "away_recent_goals_conceded": away_last5["goals_conceded"].mean(),
            "home_venue_points": home_last3_at_home["points"].mean(),
            "away_venue_points": away_last3_away["points"].mean(),
        })

    return pd.DataFrame(
        rows,
        columns=[
            "Date", "Season", "HomeTeam", "AwayTeam", "FTR",
            *FEATURE_COLUMNS,
        ],
    )