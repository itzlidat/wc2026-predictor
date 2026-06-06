# 2026 FIFA World Cup Predictor

A machine learning model using XGBoost to predict match outcomes for the 2026 FIFA World Cup.

## Overview

This project trains on historical international football results to predict win/draw/loss probabilities for World Cup matches. Features include team ELO ratings, recent form, head-to-head records, and tournament context.

## Structure

- `data/` — raw and processed datasets
- `src/` — feature engineering, model training, and evaluation scripts
- `notebooks/` — exploratory data analysis and experiments
- `predictions/` — model output and tournament bracket predictions

## Setup

```bash
pip install -r requirements.txt
```

## Data

Historical international football results sourced from [martj42/international-football-results](https://github.com/martj42/international-football-results).
