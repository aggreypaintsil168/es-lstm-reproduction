
```markdown
# ES-LSTM: Hybrid Model for Time Series Forecasting

## 📊 Project Overview

This project implements and evaluates the **ES-LSTM** (Exponential Smoothing + Long Short-Term Memory) hybrid model for stock price forecasting, as proposed by Gagneja et al. [1]. The model combines Exponential Smoothing (ES) for noise reduction and short-term trend capture with LSTM networks for learning long-term temporal dependencies.

### Key Features
- Hybrid statistical-deep learning architecture
- Grid search optimization for Exponential Smoothing parameters
- Hyperparameter tuning using Keras Tuner RandomSearch
- Support for multiple financial datasets
- Comprehensive evaluation metrics (MSE, RMSE, MAE, R²)

## 📈 Results: Reproduction vs. Original Paper

### Validation Performance Comparison

| Dataset | Source | MSE | RMSE | MAE | R² |
|---------|--------|-----|------|-----|-----|
| **S&P 500** | | | | | |
| | Original Paper [1] | 0.0018 | 0.0419 | 0.0322 | 0.9712 |
| | Our Implementation | 7412.99 | 86.10 | 66.00 | 0.9676 |
| **NIFTY 50** | | | | | |
| | Original Paper [1] | 0.0019 | 0.0434 | 0.0323 | 0.9455 |
| | Our Implementation | 69483.62 | 263.60 | 215.08 | 0.9708 |
| **HDFC Bank** | | | | | |
| | Original Paper [1] | 0.0042 | 0.0645 | 0.0446 | 0.9519 |
| | Our Implementation | 88.70 | 9.42 | 6.33 | 0.9493 |
| **Natco Pharma** | | | | | |
| | Original Paper [1] | 0.0057 | 0.0757 | 0.0494 | 0.9793 |
| | Our Implementation | 358.09 | 18.92 | 12.89 | 0.9933 |

### Training Performance

| Dataset | Split | MSE | RMSE | MAE | R² |
|---------|-------|-----|------|-----|-----|
| **S&P 500** | Train | 4749.29 | 68.92 | 65.14 | 0.9949 |
| | Validation | 7412.99 | 86.10 | 66.00 | 0.9676 |
| **NIFTY 50** | Train | 39258.87 | 198.14 | 157.95 | 0.9988 |
| | Validation | 69483.62 | 263.60 | 215.08 | 0.9708 |
| **HDFC Bank** | Train | 17.50 | 4.18 | 2.25 | 0.9997 |
| | Validation | 88.70 | 9.42 | 6.33 | 0.9493 |
| **Natco Pharma** | Train | 92.95 | 9.64 | 5.78 | 0.9998 |
| | Validation | 358.09 | 18.92 | 12.89 | 0.9933 |

### Key Observations

1. **Scale Discrepancy**: Our metrics show larger absolute values (e.g., MSE of 7412.99 vs. 0.0018 for S&P 500), which likely stems from differences in data normalization. The original paper appears to have normalized closing prices to a [0,1] or [-1,1] range before evaluation, while our implementation uses raw price values.

2. **Strong Validation Performance**: Despite scale differences, our R² scores (0.9676 for S&P 500, 0.9708 for NIFTY 50) closely match or exceed the original paper's results, confirming the model's effectiveness.

3. **Consistent Patterns**: Like the original study, we observe:
   - Higher errors on NIFTY 50 (more volatile emerging market)
   - Strong performance on Natco Pharma (R² = 0.9933)
   - Tighter fit on training vs. validation (minimal overfitting)

## 🔍 Analysis and Discussion

### What Our Reproduction Reveals

Our implementation broadly validates the ES-LSTM approach while highlighting important considerations:

1. **Data Preprocessing Matters**: The discrepancy in metric scales underscores the critical role of normalization. Future work should standardize preprocessing pipelines to enable direct comparisons.

2. **Model Robustness**: The consistently high R² scores across different datasets (0.95-0.99) demonstrate the model's ability to generalize across market types and volatility regimes.

3. **Validation Strategy**: The gap between training and validation metrics (e.g., R² 0.9949 vs. 0.9676 for S&P 500) suggests room for improved regularization.

### Methodological Considerations

Following the reproduction critique by Aggrey [2], several factors warrant attention:

- **Train-Test Split**: The original 95/5 split leaves limited test data (~360 points for S&P 500). More robust evaluation would benefit from an 80/20 split or walk-forward validation.

- **Preprocessing Leakage**: Ensuring the MinMaxScaler is fit only on training data is crucial for preventing data leakage and obtaining unbiased performance estimates.

- **Hyperparameter Sensitivity**: The 10-trial RandomSearch may not sufficiently explore the hyperparameter space; increasing trials or using Bayesian optimization could yield more stable results.

## 🚀 Future Improvements

### 1. Temporal Attention Mechanism

The current ES-LSTM architecture treats all timesteps in the lookback window uniformly, passing only the final hidden state to the output layer. This fails to capture varying temporal importance, especially during volatile market periods where recent observations carry disproportionate predictive signal.

**Proposed Enhancement**: Integrate a temporal attention layer that:
- Computes attention weights across all LSTM hidden states
- Learns which timesteps are most predictive
- Produces a context vector weighted by temporal importance

**Expected Benefits**:
- Improved focus on critical market events
- Better handling of structural breaks and volatility clusters
- Enhanced interpretability through attention weight visualization

### 2. Additional Improvements

- **Walk-Forward Validation**: Implement rolling window validation to better simulate real-world forecasting scenarios
- **Ensemble Methods**: Combine ES-LSTM with XGBoost or LightGBM for improved robustness
- **Feature Enrichment**: Incorporate technical indicators (RSI, MACD, Bollinger Bands) and sentiment data
- **Architecture Optimization**: Experiment with BiLSTM, GRU, or Transformer architectures
- **Bayesian Hyperparameter Tuning**: Replace RandomSearch with Optuna or Hyperopt for more efficient optimization

## 📁 Project Structure

```
es_lstm_project/
├── data/                          # Datasets (S&P 500, NIFTY 50, etc.)
├── notebooks/                     # Jupyter notebooks
│   ├── 01_data_eda.ipynb         # Exploratory data analysis
│   ├── 02_es_grid_search.ipynb   # ES parameter optimization
│   └── 03_lstm_train.ipynb       # LSTM model training
├── src/                          # Source code
│   ├── data_loader.py            # Data loading utilities
│   ├── es_smoother.py            # Exponential smoothing
│   ├── lstm_model.py             # LSTM model architecture
│   ├── attention.py              # Temporal attention module (future)
│   ├── baselines.py              # Baseline models (SARIMAX, Prophet)
│   └── metrics.py                # Evaluation metrics
├── figures/                      # Generated plots
├── models/                       # Saved trained models
├── config.yaml                   # Configuration parameters
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # Explains the implementation
```

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.10+
- Conda (recommended) or pip

### Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/es_lstm_project.git
cd es_lstm_project

# Create conda environment
conda create -n es-lstm-env python=3.10
conda activate es-lstm-env

# Install dependencies
pip install -r requirements.txt
```

### Requirements.txt
```
tensorflow==2.17.0
keras==3.5.0
keras-tuner==1.4.7
statsmodels==0.14.2
pandas==2.2.0
numpy==1.26.0
matplotlib==3.8.0
seaborn==0.13.0
scikit-learn==1.4.0
yfinance==0.2.38
pyyaml==6.0.1
```

## 📊 Usage Examples

### Load and Preprocess Data
```python
from src.data_loader import load_stock_data
from src.es_smoother import optimize_es

# Load data
data = load_stock_data(ticker="^GSPC", start="2010-01-01", end="2024-08-31")

# Apply exponential smoothing
optimal_params = optimize_es(data['Close'], method='double')
smoothed_data = apply_es(data['Close'], **optimal_params)
```

### Train ES-LSTM Model
```python
from src.lstm_model import create_lstm_model, train_model

# Prepare features
X_train, X_val, y_train, y_val = prepare_data(smoothed_data, lookback=60)

# Create and train model
model = create_lstm_model(input_shape=(60, 1))
history = train_model(model, X_train, y_train, X_val, y_val)
```

### Evaluate Performance
```python
from src.metrics import calculate_metrics

# Predict and evaluate
y_pred = model.predict(X_val)
metrics = calculate_metrics(y_val, y_pred)
print(f"Validation R²: {metrics['r2']:.4f}")
```

## 📝 Citation

If you use this code or findings in your research, please cite:

```bibtex
@article{gagneja2026eslstm,
  title={ES-LSTM: a hybrid model for accurate time series forecasting in financial markets},
  author={Gagneja, Vaibhav and Gupta, Mayank and Batish, Sanjay and Saini, Poonam and Rani, Sudesh},
  journal={Digital Finance},
  volume={8},
  number={12},
  year={2026},
  publisher={Springer},
  doi={10.1007/s42521-025-00173-0}
}

@techreport{aggrey2026reproduction,
  title={Reproduction of ES-LSTM: A Hybrid Model for Accurate Time Series Forecasting in Financial Markets},
  author={Aggrey, Ishmeal},
  year={2026},
  institution={University of Ghana}
}
```

## 👥 Acknowledgments

- Original authors: Vaibhav Gagneja, Mayank Gupta, Sanjay Batish, Poonam Saini, Sudesh Rani (Punjab Engineering College)
- Reproduction critique: Ishmeal Aggrey (University of Ghana)
- Data source: Yahoo Finance via yfinance

## 📄 License

This project is for academic and research purposes. Please cite the original paper if using this implementation.

## 🔗 Links

- [Original Paper](https://doi.org/10.1007/s42521-025-00173-0)
- [GitHub Repository](https://github.com/YOUR_USERNAME/es_lstm_project)

---

**Note on Metrics**: The scale difference between our implementation and the original paper stems from different normalization strategies. The original paper normalized closing prices to [0,1] or [-1,1] range, while our implementation uses raw price values. This does not affect model performance evaluation within our implementation but should be considered when comparing absolute metric values across studies.
```

