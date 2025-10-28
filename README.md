# Accident Risk Predictor 🚗⚠️

An interactive Streamlit application for predicting road accident risk using machine learning. Built with XGBoost and trained on the Kaggle Playground Series S5E10 dataset.

## Features

- **Interactive Prediction**: Input road conditions and get instant accident risk predictions
- **Variable Explorer**: See how different factors affect accident risk
- **Scenario Comparison**: Compare risk between two different road conditions
- **Batch Processing**: Upload CSV files for bulk predictions
- **Model Insights**: View feature importance and model statistics

## Quick Start

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd accident-predict
   ```

2. **Install dependencies**
   ```bash
   pip install streamlit xgboost scikit-learn pandas numpy
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open in browser**
   - The app will automatically open at `http://localhost:8501`
   - If you have `train.csv` in the project folder, the model will auto-train on first run
   - Navigate to the "Predict" tab (default) to start making predictions

## Dataset

This project uses the [Kaggle Playground Series S5E10](https://www.kaggle.com/competitions/playground-series-s5e10) dataset for road accident risk prediction. The dataset includes features like:

- Road type (urban, rural, highway)
- Number of lanes
- Road curvature
- Speed limit
- Lighting conditions
- Weather conditions
- Time of day
- Holiday/school season indicators
- Historical accident data

## Model Details

- **Algorithm**: XGBoost Regressor
- **Hyperparameters**: Optimized for the competition dataset
  - 500 estimators
  - Learning rate: 0.05
  - Max depth: 6
  - Subsample: 0.8
- **Validation**: 3-fold cross-validation RMSE reported
- **Target**: Continuous accident risk score (0-1 scale)

## Usage

### Single Prediction
1. Go to the "Predict" tab
2. Adjust the road condition parameters using the form
3. Click "Predict risk" to get the accident risk score

### Explore Variables
1. Navigate to the "Explore" tab
2. Set baseline conditions
3. Select a variable to vary (lanes, curvature, speed limit, etc.)
4. View the interactive chart showing risk vs. the selected variable

### Compare Scenarios
1. Use the "Compare" tab
2. Set up two different scenarios (A and B)
3. Click "Compare" to see risk predictions side-by-side

### Batch Processing
1. Go to the "Batch" tab
2. Upload a CSV file with the same columns as the training data (minus the target)
3. Download the results with predicted risk scores

## File Structure

```
accident-predict/
├── app.py                 # Main Streamlit application
├── models/               # Saved model artifacts
│   ├── accident_xgb_model.pkl
│   └── preprocessor_v2.pkl
├── train.csv            # Training dataset (if available)
├── test.csv             # Test dataset (if available)
└── README.md            # This file
```

## Development

### Training a New Model
1. Place your `train.csv` file in the project root
2. Use the sidebar "Train from CSV" section
3. Click "Train model" to retrain with your data
4. The new model will be saved automatically

### Model Persistence
- Models are saved as pickle files in the `models/` directory
- The preprocessor handles categorical encoding and feature engineering
- Auto-loading attempts to load saved artifacts on app startup

## Requirements

- Python 3.7+
- streamlit
- xgboost
- scikit-learn
- pandas
- numpy

## Contributing

Feel free to open issues or submit pull requests! This project is designed to be educational and easily extensible.

## License

This project is licensed under the "Buy Me a Beer" License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Kaggle Playground Series S5E10 for the dataset
- XGBoost team for the excellent gradient boosting library
- Streamlit for making interactive ML apps so easy to build