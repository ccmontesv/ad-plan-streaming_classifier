import numpy as np
import pandas as pd
from src.config import HEURISTIC_THRESHOLD_RATIO, HEURISTIC_THRESHOLD_COUNT

def compute_heuristics(df: pd.DataFrame):
    """
    Applies heuristic rules to detect ad breaks:
    - Filters sessions eligible for ads
    - Flags sessions with inter-session gaps that match ad-like durations
    - Aggregates gap patterns per user
    """
    df = df.sort_values(by=["session_key", "start_time"])
    df["next_start"] = df.groupby("session_key")["start_time"].shift(-1)
    df["gap_to_next"] = (df["next_start"] - df["end_time"]).dt.total_seconds() / 60
    df = df[df["gap_to_next"].isna() | (df["gap_to_next"] >= 0)]
    df["ad_eligible"] = df["duration_min"] >= 15

    # Platform-specific ad gap logic
    conditions = [
        (df["service"] == "Netflix") & df["gap_to_next"].between(1.0, 1.5),
        (df["service"] == "Hulu") & df["gap_to_next"].between(1.5, 2.5)
    ]
    df["gap_flag"] = np.select(conditions, [True, True], default=False)
    df["has_next"] = df["next_start"].notna()
    df["gap_flag_eligible"] = df["ad_eligible"] & df["gap_flag"] & df["has_next"]

    # Aggregate stats per user
    grouped = df.groupby(["tv_id", "service"]).agg(
        total_sessions=("duration_min", "count"),
        eligible_sessions=("gap_flag_eligible", "count"),
        avg_duration_min=("duration_min", "mean"),
        gap_count=("gap_flag_eligible", "sum"),
        avg_gap_min=("gap_to_next", lambda x: x[df.loc[x.index, "gap_flag_eligible"]].mean())
    ).reset_index()

    # Heuristic rule: label as ad-supported if >15% of eligible sessions show ad-like gaps
    grouped["gap_ratio"] = grouped["gap_count"] / grouped["eligible_sessions"]
    grouped["customer_plan_gap_ratio"] = np.where(grouped["gap_ratio"] > HEURISTIC_THRESHOLD_RATIO, "ad-supported", "ad-free")
    grouped["customer_plan_gap_count"] = np.where(grouped["gap_count"] > HEURISTIC_THRESHOLD_COUNT, "ad-supported", "ad-free")

    return grouped, df