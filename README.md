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
│   │__ app.py                # App with API interaction 
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

### 5. Run the API Server

Start the FastAPI server for real-time predictions:

```bash
cd /Users/aishwarya/Desktop/healthcare-readmission-ml-system
PYTHONPATH=$(pwd) uvicorn src.app:app --reload --port 3000
```

Or using the venv python directly:
```bash
PYTHONPATH=$(pwd) /path/to/venv/bin/uvicorn src.app:app --reload --port 3000
```

Once running, access:
- **API Health Check**: http://127.0.0.1:3000/
- **Interactive API Docs**: http://127.0.0.1:3000/docs
- **Alternative Docs**: http://127.0.0.1:3000/redoc

#### API Endpoints

**1. Health Check - `GET /`**
```bash
curl http://127.0.0.1:3000/
```

Returns server status, model type, threshold, and recall target.

**2. Single Patient Prediction - `POST /predict`**
```bash
curl -X POST "http://127.0.0.1:3000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "race": "Caucasian",
    "gender": "Female",
    "age": "[70-80)",
    "admission_type_id": 1,
    "discharge_disposition_id": 1,
    "admission_source_id": 7,
    "time_in_hospital": 3,
    "payer_code": "MC",
    "medical_specialty": "Cardiology",
    "num_lab_procedures": 45,
    "num_procedures": 0,
    "num_medications": 15,
    "number_outpatient": 0,
    "number_emergency": 0,
    "number_inpatient": 0,
    "diag_1": "428",
    "diag_2": "250",
    "diag_3": "401",
    "number_diagnoses": 9
  }'
```

Returns:
- `readmission_risk_score`: Probability of 30-day readmission (0-1)
- `high_risk_flag`: 1 if risk exceeds threshold, 0 otherwise
- `clinical_recommendation`: Suggested action based on risk level

**3. Batch Prediction - `POST /predict-batch`**

Submit multiple patients for scoring in a single request. See `/docs` for the request schema.

#### Interactive API Documentation

The FastAPI server provides automatic interactive documentation at `/docs`:

1. Navigate to http://127.0.0.1:3000/docs
2. Expand any endpoint to see its details
3. Click "Try it out" to test the endpoint
4. Fill in the request body with patient data
5. Click "Execute" to get predictions
6. View the response with risk scores and recommendations

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

### `src/app.py`
- FastAPI REST API for real-time predictions
- Provides health check, single prediction, and batch prediction endpoints
- Accepts patient data via JSON and returns readmission risk scores
- Includes automatic interactive documentation (Swagger UI)
- Designed for integration with hospital information systems
- Returns clinical recommendations based on threshold
- Supports both individual and batch patient scoring

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
