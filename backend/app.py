import os
import datetime as _dt
from pathlib import Path
from typing import Tuple

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

try:
    import psycopg2
except ImportError as exc:
    raise ImportError(
        "psycopg2 is required. Install with: pip install psycopg2-binary"
    ) from exc

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DATABASE_URL: str = os.environ["DATABASE_URL"]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    conn = psycopg2.connect(DATABASE_URL)
    try:
        df1 = pd.read_sql("SELECT * FROM assessments;", conn)
        if not df1.empty:
            df1 = df1.iloc[:, 1:]
            df1.columns = [
                "candidate_name", "technology", "deadline",
                "end_client", "sender", "email_datetime_est"
            ]

        df2 = pd.read_sql("SELECT * FROM assessments_response;", conn)
        if not df2.empty:
            df2 = df2.iloc[:, 1:]
            df2.columns = [
                "candidate_name", "technology", "deadline", "end_client",
                "assigned_to", "task_status", "feedback", "sender", "email_datetime_est",
            ]
    finally:
        conn.close()

    for col in ["deadline", "email_datetime_est"]:
        if col in df1.columns:
            df1[col] = pd.to_datetime(df1[col], errors="coerce", utc=True)
        if col in df2.columns:
            df2[col] = pd.to_datetime(df2[col], errors="coerce", utc=True)

    return df1, df2

def _normalize_list(value):
    """Ensure we always work with a cleaned list of filter values."""
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    return [v for v in value if v not in ("", None)]


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    filtered_df = df.copy()
    for col, raw_value in filters.items():
        values = _normalize_list(raw_value)
        if not values:
            continue

        if col in ["deadline", "email_datetime_est"]:
            # Support both range filters ([start, end]) and single value equality.
            if len(values) >= 2:
                start_date = pd.to_datetime(values[0], utc=True)
                end_date = pd.to_datetime(values[1], utc=True)
                mask = (filtered_df[col] >= start_date) & (filtered_df[col] <= end_date)
                filtered_df = filtered_df.loc[mask]
            else:
                match_date = pd.to_datetime(values[0], utc=True)
                filtered_df = filtered_df[filtered_df[col] == match_date]
        else:
            filtered_df = filtered_df[filtered_df[col].isin(values)]
    return filtered_df


class FilterRequest(BaseModel):
    filters: dict


@app.get("/assessments")
def get_assessments():
    df1, _ = load_data()
    return df1.to_dict(orient="records")


@app.get("/responses")
def get_responses():
    _, df2 = load_data()
    return df2.to_dict(orient="records")


@app.post("/filter-assessments")
def filter_assessments(req: FilterRequest):
    df1, _ = load_data()
    result = apply_filters(df1, req.filters)
    return result.to_dict(orient="records")


@app.post("/filter-responses")
def filter_responses(req: FilterRequest):
    _, df2 = load_data()
    result = apply_filters(df2, req.filters)
    return result.to_dict(orient="records")


@app.post("/pending")
def pending(req: FilterRequest):
    _, df2 = load_data()

    # Pending = task_status null OR feedback null (and not completed)
    pending_df = df2[
        ((df2["task_status"].isna()) | (df2["feedback"].isna()))
        & (df2["task_status"] != "completed")
    ]

    # Apply user filters
    result = apply_filters(pending_df, req.filters)
    return result.to_dict(orient="records")
