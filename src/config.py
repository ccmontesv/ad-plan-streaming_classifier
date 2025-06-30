from datetime import datetime

RAW_DATA_PATH = "data/raw/data.csv"
PROCESSED_DATA_PATH = "data/processed/processed.csv"
PDF_OUTPUT_DIR = "reports/"
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
PDF_OUTPUT_PATH = f"{PDF_OUTPUT_DIR}ad_plan_analysis_report_{TIMESTAMP}.pdf"

HEURISTIC_THRESHOLD_RATIO = 0.15
HEURISTIC_THRESHOLD_COUNT = 3
MIN_DURATION_SECONDS = 60