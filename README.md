# Premier League Match Prediction

## Research Question

Can recent team form, attacking and defensive performance, and
home/away results predict Premier League match outcomes better
than always predicting the most common result?

The possible outcomes are home win (H), draw (D), and away win (A).

## My Logic / Analytical Framework

My starting expectation was that recent results would matter most
because they reflect a team's current form. However, wins and losses
do not tell the whole story. I also included goals scored and conceded
to describe attacking and defensive performance.

I included home and away performance because a team's overall form
might differ from its performance at a particular venue. My approach
was to compare these measures for both opponents and test whether
they added useful information.

I used five previous league matches for recent form and three previous
home or away matches for venue performance. These were simple starting
choices that balanced recent information with more than one result.
I did not claim these were the optimal windows.

## Data

Source: https://www.football-data.co.uk/englandm.php

The project uses five Premier League seasons, from 2021–22 through
2025–26, containing 1,900 matches in the downloaded files.

Required columns:
- Date: match date
- HomeTeam and AwayTeam: team names
- FTHG and FTAG: full-time goals
- FTR: full-time result

Download the following files into data/ using these filenames:

| Season | Filename | Download |
|---|---|---|
| 2021–22 | epl_2021_22.csv | https://www.football-data.co.uk/mmz4281/2122/E0.csv |
| 2022–23 | epl_2022_23.csv | https://www.football-data.co.uk/mmz4281/2223/E0.csv |
| 2023–24 | epl_2023_24.csv | https://www.football-data.co.uk/mmz4281/2324/E0.csv |
| 2024–25 | epl_2024_25.csv | https://www.football-data.co.uk/mmz4281/2425/E0.csv |
| 2025–26 | epl_2025_26.csv | https://www.football-data.co.uk/mmz4281/2526/E0.csv |

Data checks found no missing required values, duplicate season/team
pairings, or results inconsistent with the scores.

## Features and Leakage Prevention

Eight features were created:

| Feature | Definition |
|---|---|
| home_recent_points | Home team's average points in its previous five league matches |
| away_recent_points | Away team's average points in its previous five league matches |
| home_recent_goals_scored | Home team's average goals scored in its previous five matches |
| away_recent_goals_scored | Away team's average goals scored in its previous five matches |
| home_recent_goals_conceded | Home team's average goals conceded in its previous five matches |
| away_recent_goals_conceded | Away team's average goals conceded in its previous five matches |
| home_venue_points | Home team's average points in its previous three home matches |
| away_venue_points | Away team's average points in its previous three away matches |

A win earns three points, a draw one, and a loss zero.

Every feature uses only matches dated before the predicted match.
The current match's goals and result are never prediction inputs.
Histories reset each season.

Matches without the full required history were excluded from modeling,
but still contributed history to later matches. This left 1,594 eligible
matches and excluded 306 early-season fixtures.

## Evaluation Design

The data was split chronologically:

- Training: 2021–22 through 2023–24, with 954 eligible matches.
- Validation: 2024–25, with 320 eligible matches.
- Final test: 2025–26, with 320 eligible matches.

Validation compared:
1. A baseline predicting the most common training outcome.
2. Logistic Regression using recent points only.
3. Logistic Regression using points and goal statistics.
4. Logistic Regression using all eight features.
5. Random Forest using all eight features.

All comparisons used the same eligible matches.

Logistic Regression used standardized features. Scaling was fitted
only on the data used to train the model.

Random Forest used 300 trees, maximum depth 5, and a minimum of
10 training observations per leaf.

All-feature Logistic Regression was selected as the primary probability
model because it had the lowest validation log loss. Random Forest was
retained as a comparison.

The final models were refitted on 1,274 matches from 2021–22 through
2024–25 and evaluated on the 2025–26 test season. No model settings were
changed based on final test performance.

Earlier test-season results could update the historical features for
later test matches. Model parameters remained fixed throughout testing.

## Exploratory Findings

Among eligible training matches:
- Home wins: 46.0%
- Draws: 22.2%
- Away wins: 31.8%

When the home team had stronger recent form, home wins occurred in
60.1% of matches. When the away team had stronger recent form, home
wins occurred in 28.1%.

For this graph, stronger form meant a difference greater than 0.4
recent points per match. This grouping was for visualization only;
the models used the original numeric features.

These patterns supported testing recent form as a predictor.
They do not establish causation.

## Validation Results

| Model | Accuracy | Macro F1 | Log loss |
|---|---:|---:|---:|
| Baseline | 0.4188 | 0.1968 | 1.0754 |
| Logistic Regression: form | 0.4875 | 0.3578 | 1.0483 |
| Logistic Regression: form + goals | 0.4719 | 0.3469 | 1.0395 |
| Logistic Regression: all features | 0.4875 | 0.3615 | 1.0348 |
| Random Forest | 0.5031 | 0.3691 | 1.0549 |

Higher accuracy and macro F1 are better. Lower log loss is better.

## Final Test Results

| Model | Accuracy | Macro F1 | Log loss |
|---|---:|---:|---:|
| Baseline | 0.4219 | 0.1978 | 1.0865 |
| Logistic Regression | 0.4719 | 0.3536 | 1.0632 |
| Random Forest | 0.4688 | 0.3421 | 1.0638 |

Logistic Regression correctly predicted 151 of 320 matches, compared
with 135 for the baseline. This was an improvement of five percentage
points, or 16 additional correct predictions.

It identified 102 of 135 home wins and 49 of 98 away wins, but none of
the 87 draws. Random Forest also identified no draws correctly.

Neither learned model selected draw as its predicted test outcome.
This does not mean it assigned draws zero probability; a win outcome
received the highest probability.

Logistic Regression exceeded Random Forest by only one correct
prediction. The difference is too small to claim a clear advantage.
Statistical significance was not assessed.

## What the Models Learned

Random Forest ranked home recent points highest among individual
features, followed by home goals scored and away recent points.

Logistic Regression associated stronger home attacking performance
and weaker away defense with higher home-versus-away win odds.
Stronger away attacking performance showed the opposite relationship.

Some coefficients were harder to interpret because points, goals,
and venue performance overlap. Feature importance and coefficients
describe model behavior, not causal effects.

My initial expectation about recent form was partly supported:
recent points alone improved validation performance over the baseline.
However, the analysis did not establish it as the strongest feature
group. Goal and venue features also improved probability performance.

## Historical Prediction Example

Match: Bournemouth vs Fulham
Date: October 3, 2025

- Bournemouth win: 57.4%
- Draw: 23.1%
- Fulham win: 19.5%

Predicted outcome: Bournemouth win
Actual outcome: Bournemouth win

This was the first eligible test match, selected chronologically
rather than because the prediction was correct. The model was trained
on earlier seasons, and the inputs contained only pre-match history.

This single example demonstrates the process. Overall performance
is assessed using the full test season.

## Limitations and Future Improvements

- No injuries, starting lineups, transfers, or managerial changes.
- No explicit adjustment for the strength of previous opponents.
- Five-match histories can be noisy.
- Season resets prevent predictions for early fixtures.
- Results apply to eligible matches in one league and one test season.
- Draw prediction was a major weakness.
- Match events such as red cards and unexpected mistakes add uncertainty.
- Probability calibration was not separately assessed.

Future work could add longer-term team strength, opponent-adjusted
form, rest days, and better draw-related features. Alternative history
windows could be evaluated using chronological validation within the
development data. Any changes motivated by these test results would
need a new untouched evaluation period.

## Project Files

- main.py: runs the complete workflow
- src/features.py: builds pre-match features
- src/eda.py: creates training-data graphs
- src/modeling.py: compares models on validation data
- src/final_evaluation.py: evaluates fixed models on the test season
- src/predict_match.py: demonstrates a historical prediction
- data/: raw CSVs and generated datasets
- results/: graphs, metrics, reports, and predictions
- requirements.txt: Python dependencies

## How to Run

From the project folder on macOS or Linux:

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt
    python main.py

Place the five source CSV files in data/ before running.

The script recreates the cleaned data, features, graphs, validation
reports, test reports, and example prediction.

## AI Use Disclosure

I used ChatGPT extensively to help develop the project framework,
generate Python code, interpret output, and draft documentation.
I provided my initial expectations, ran the code locally, and shared
the output for troubleshooting and discussion.

The model-selection recommendation and much of the implementation
were AI-assisted. The reported results came from running the code
on the downloaded data, rather than from invented examples.

AI conversation link or attached transcript:
https://chatgpt.com/share/6abaad40-fcdc-83e9-a23d-34e4d10832ea