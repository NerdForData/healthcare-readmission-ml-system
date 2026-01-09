# Healthcare Readmission ML System

A machine learning system for predicting 30-day hospital readmission risk in diabetic patients. This project uses the UCI Diabetes 130-US hospitals dataset to identify high-risk patients who may benefit from early intervention and follow-up care.

## 🎯 Project Overview

Hospital readmissions are a critical healthcare quality metric and a significant cost driver. This system predicts which patients are at high risk of being readmitted within 30 days of discharge, enabling healthcare providers to:

- Allocate resources more efficiently
- Implement targeted intervention programs
- Reduce preventable readmissions
- Improve patient outcomes

### Key Results

- **ROC-AUC**: 0.6771 (Gradient Boosting model)
- **PR-AUC**: 0.2306
- **Recall**: 80% at clinical threshold
- **Clinical Threshold**: 0.0817 (probability cutoff for flagging high-risk patients)

## 📁 Project Structure

```
healthcare-readmission-ml-system/
├── README.md
├── requirements.txt
├── data/
│   └── raw/
│       └── diabetic_data.csv
├── models/
│   ├── readmission_pipeline_logreg.joblib
│   ├── readmission_pipeline_gb.joblib
│   ├── final_readmission_pipeline.joblib
│   └── clinical_threshold.json
├── notebooks/
│   └── 01_healthcare_eda.ipynb
├── src/
│   ├── __init__.py
│   ├── data_prep.py          # Data loading and preprocessing
│   ├── train.py              # Model training script
│   ├── evaluate.py           # Clinical threshold analysis
│   ├── finalize_model.py     # Final model creation
└── artifacts/
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

1. **Clone the repository** (or navigate to the project directory):
   ```bash
   cd healthcare-readmission-ml-system
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Data

The project uses the **Diabetes 130-US hospitals dataset** containing clinical data from diabetic patients. The dataset includes:

- **101,766 encounters** (patient visits)
- **47 features** including demographics, diagnoses, medications, and hospital stay details
- **Target**: 30-day readmission (binary classification)

Place the `diabetic_data.csv` file in the `data/raw/` directory.

## 🔧 Usage

### 1. Exploratory Data Analysis

Run the Jupyter notebook to explore the data and key insights:

```bash
jupyter notebook notebooks/01_healthcare_eda.ipynb
```

**Key Insights from EDA:**
- Approximately 11.16% of encounters result in 30-day readmission
- Readmission risk increases with patient age
- Longer hospital stays correlate with higher readmission rates
- Higher number of diagnoses (comorbidities) increases risk
- Emergency admissions have higher readmission rates than elective admissions

### 2. Train Models

Train a logistic regression model:
```bash
python src/train.py --data-path data/raw/diabetic_data.csv --model-type logreg
```

Train a gradient boosting model (recommended):
```bash
python src/train.py --data-path data/raw/diabetic_data.csv --model-type gb
```

### 3. Evaluate and Set Clinical Threshold

The clinical threshold analysis helps determine the optimal probability cutoff for flagging high-risk patients:

```python
from src.evaluate import clinical_threshold_analysis

results = clinical_threshold_analysis(
    data_path="data/raw/diabetic_data.csv",
    model_path="models/readmission_pipeline_gb.joblib",
    desired_recall=0.80
)
```

### 4. Create Final Production Model

Generate the final model and threshold configuration:
```bash
python src/finalize_model.py
```

This creates:
- `models/final_readmission_pipeline.joblib` - Production-ready model
- `models/clinical_threshold.json` - Clinical threshold configuration

## 🧪 Model Performance

### Gradient Boosting Classifier (Recommended)

- **ROC-AUC**: 0.6771
- **PR-AUC**: 0.2306
- **Hyperparameters**:
  - n_estimators: 200
  - learning_rate: 0.05
  - max_depth: 3

### Logistic Regression

- **ROC-AUC**: 0.6463
- **PR-AUC**: 0.2047
- **Configuration**: Class-balanced, max_iter=2000

### Clinical Threshold Analysis

At a threshold of **0.0817** (designed to achieve 80% recall):
- **Precision**: 14.62%
- **Recall**: 80.00%
- **Patients flagged**: ~61% of validation set

This high-recall approach ensures that most patients at risk of readmission are identified, though it results in some false positives. The trade-off is appropriate for a clinical setting where missing high-risk patients is more costly than providing extra follow-up care.

## 📦 Project Components

### `src/data_prep.py`
- Loads and preprocesses the diabetic dataset
- Handles missing values (marked as "?")
- Creates binary target variable (readmitted within 30 days)
- Builds preprocessing pipeline (imputation + one-hot encoding)

### `src/train.py`
- Command-line interface for model training
- Supports logistic regression and gradient boosting
- Handles class imbalance with balanced weights
- Saves trained models to `models/` directory

### `src/evaluate.py`
- Clinical threshold analysis for operational decision-making
- Precision-recall curve computation
- Finds optimal threshold for desired recall level

### `src/finalize_model.py`
- Creates final production-ready model
- Saves clinical threshold configuration
- Generates deployment artifacts

## 🏥 Clinical Use Case

The system is designed to support clinical decision-making by:

1. **Identifying High-Risk Patients**: Patients with predicted probability > 0.0817 are flagged
2. **Resource Allocation**: Healthcare teams can prioritize follow-up care for flagged patients
3. **Intervention Planning**: Early identification enables proactive interventions such as:
   - Post-discharge phone calls
   - Home health visits
   - Medication reconciliation
   - Patient education programs

## 🛠️ Dependencies

Key packages (see `requirements.txt` for full list):
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## 📝 License

This project is for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📧 Contact

For questions or feedback about this project, please open an issue in the repository.

---

**Note**: This system is intended to support clinical decision-making, not replace it. All predictions should be reviewed by qualified healthcare professionals before taking action.
