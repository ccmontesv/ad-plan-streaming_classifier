# Ad Plan Classifier – Antenna Technical Test

This project classifies streaming users as **ad-supported** or **ad-free** subscribers using **heuristics**, **unsupervised learning (clustering)**, and a **hybrid rule-based system**.

## Project Objective

The main goal is to infer the type of subscription (ad-supported vs. ad-free) for Netflix and Hulu users using anonymized session-level behavioral data. This includes:

- Detecting session gaps
- Measuring viewing patterns
- Clustering similar behaviors
- Applying heuristic and hybrid rules
- Producing a one-page PDF report per run

## Folder Structure

```
ad_plan_classifier/
│
├── main.py                  # Main pipeline script
├── requirements.txt         # Python dependencies
├── .gitignore               # Git ignored files
├── README.md                # Project documentation
│
├── data/
│   ├── raw/                 # Input session-level data (CSV)
│   └── processed/           # Cleaned & transformed data
│   └── results/             # Final results with the ad-type label for each customer
│
├── output/
│   └── report.pdf           # Final visualization/report
│
├── src/                     # Core source code
│   ├── __init__.py
│   ├── config.py            # File path configuration
│   ├── preprocess.py        # Data loading & cleaning
│   ├── heuristic.py         # Heuristic logic
│   ├── clustering.py        # Clustering with KMeans, PCA, t-SNE
│   ├── hybrid.py            # Hybrid rule logic
│   └── report.py            # PDF report generation
│
└── tests/                   # Unit tests for each module
    ├── test_preprocess.py
    ├── test_heuristic.py
    ├── test_clustering.py
    └── test_hybrid.py
```

## How to Run

You can run the project either using Docker (recommended for consistent environments) or directly in your local Python environment.

> Docker: Build the Docker image (run from project root):

```bash
docker build -t ad-plan-classifier .
```

devcontainer.json is also include in case you want to reopen the container in VS.

> Python venv: Make sure you are in the correct virtual environment.

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Run the project
python main.py
```

## Cloning the Repository with Git LFS (required because the data.csv file exceeds the 100MB)

Make sure you have Git LFS installed before cloning:

```bash
# Install Git LFS (only once per machine)
git lfs install

# Then clone the repository
git clone https://github.com/yourusername/your-repo.git
cd your-repo


## Testing

We use `pytest` for unit tests:

```bash
pytest tests/
```

## Output

A one-page PDF with:
- Cluster scatterplots (PCA + t-SNE)
- Heuristic, clustering, and hybrid summaries
- Feature importance rankings

## Authors

Developed as part of the Antenna technical test.

## Contact

For questions or suggestions, feel free to open an issue or contact the author.
