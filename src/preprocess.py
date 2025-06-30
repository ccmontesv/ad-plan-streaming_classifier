import pandas as pd
from src.config import RAW_DATA_PATH, MIN_DURATION_SECONDS

def load_and_clean_data():
    """
    Loads the raw session data and applies initial filtering:
    - Keep only OTT sessions on Netflix or Hulu
    - Drop excluded titles and rows with missing timestamps
    - Calculate session duration in minutes
    - Generate a unique session key
    """
    df = pd.read_csv(RAW_DATA_PATH, low_memory=False)

    # Select relevant columns
    columns = [
        "tv_id", "service", "start_time", "end_time", "duration",
        "program_content_offset_s", "content_type", "exclude_title",
        "title_id", "season_id", "episode"
    ]
    df = df[columns]

    # Filter criteria
    df = df[df['content_type'] == 'OTT']
    df = df[df['service'].isin(['Netflix', 'Hulu'])]
    df = df[df['exclude_title'] == False]
    df = df.dropna(subset=["start_time", "end_time", "season_id"])

    # Parse datetime and calculate duration
    df['start_time'] = pd.to_datetime(df['start_time'], utc=True, errors='coerce')
    df['end_time'] = pd.to_datetime(df['end_time'], utc=True, errors='coerce')
    df = df[df['duration'] > MIN_DURATION_SECONDS]
    df['duration_min'] = df['duration'] / 60

    # Create a session key to group repeated views of the same episode
    df["session_key"] = (
        df["tv_id"].astype(str) + "_" +
        df["service"].astype(str) + "_" +
        df["season_id"].astype(str) + "_" +
        df["episode"].astype(str)
    )

    return df