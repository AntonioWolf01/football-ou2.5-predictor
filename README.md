# ⚽ Football Over/Under 2.5 Goals Predictor

This project demonstrates a complete machine learning pipeline for predicting Over/Under 2.5 goals in football matches. It uses gradient boosting models (XGBoost, LightGBM, CatBoost) trained on match statistics and betting odds to identify _value bets_ against the Betfair Exchange market.

The primary goal is to demonstrate a functional, end-to-end pipeline, from data loading and model training to evaluation and profit simulation.

## 📊 Dataset Information

The project uses a toy dataset (_general_toy_dataset.csv_) extracted from approximately 10,000 football matches. This dataset is a placeholder and is used solely for demonstrating the pipeline's functionality.

* **Source:** Aggregated from public football data websites
* **Size:** ~1,900 fixtures
* **Structure:** Each row represents a single match and includes:
    * Match identifiers
    * Bookmakers & Betfair Exchange odds (1X2, Over 2.5, Under 2.5)
    * Final result (full-time goals)
    * Raw performance stats

### Train/Test Split
The dataset is split chronologically to simulate a real-world prediction scenario:

* **Training Set:** All matches from the 2023/2024 season.
* **Test Set:** 240 unseen matches from the 2024/2025 season (`df_6_test.csv`).

---

## 🚀 Reproducing Results

Follow these steps to set up the environment and run the full model training and evaluation pipeline.

### 1. Clone the Repository

```bash
git clone https://github.com/AntonioWolf01/football-ou2.5-predictor.git
cd football-ou2.5-predictor
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Main Pipeline
This script will load the data, create the predictive features, train all models (XGBoost, LightGBM, CatBoost) with Bayesian optimization, and save the results, models, and visualizations.

```bash
python "Create your Predictive Model.py"
```

### 5. Generate Results and Visualizations
The main script will automatically generate and save output files (e.g., calibration curves, profit simulation plots).

---

## 🛠️ Methodology Details

### Machine Learning Approach
The problem is framed as a **binary classification task**: will a match have Over 2.5 goals (Class 1) or Under 2.5 goals (Class 0)?

Because betting markets are probabilistic, the models are trained to output a **probability** rather than a simple binary prediction. Gradient boosting algorithms were chosen for their high performance on tabular data and their ability to handle complex, non-linear relationships between features.

### Evaluation Metrics
The primary evaluation metric used for hyperparameter optimization is **Log Loss** (Negative Log-Likelihood). This metric is standard for probabilistic classification as it heavily penalizes models that are confidently wrong, which is crucial for betting applications.

Other metrics used for final evaluation include:

* **Brier Score:** Measures the accuracy of probabilistic predictions.
* **Calibration Curves:** Visually assess whether the model's predicted probabilities are reliable (e.g., does a 30% prediction actually happen 30% of the time?).

### Monte Carlo Profit Simulation
To test the model's practical value, a Monte Carlo simulation is run on the test set. This simulation:

1.  Compares the model's predicted probability against the implied probability from the Betfair Exchange odds.
2.  Identifies _value bets_ where the model's confidence is higher than the market's, constrained to a probability of at least >60%.
3.  Simulates placing a level stake on these value bets (using _Kelly Criterion_).
4.  Runs this simulation thousands of times to generate a distribution of potential profit/loss, testing the strategy's robustness.

---

> **Important:** Given the educational nature and the toy dataset, the primary goal is to demonstrate a functional pipeline, not necessarily to achieve high profitability. Real-world application would require a much larger, more diverse, and rigorously validated dataset.
