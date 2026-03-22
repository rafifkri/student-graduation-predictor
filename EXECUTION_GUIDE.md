# EXECUTION GUIDE
## Complete Step-by-Step Project Implementation

---

## Table of Contents

1. Prerequisites and Requirements
2. Environment Setup
3. Project Structure and Organization
4. Notebook Execution Sequence
5. Detailed Notebook Walkthroughs
6. System Deployment
7. Troubleshooting Guide
8. Next Steps and Recommendations

---

## 1. PREREQUISITES AND REQUIREMENTS

### System Requirements

- **Operating System**: Windows, macOS, or Linux
- **Python Version**: 3.8 or higher
- **Package Manager**: pip (included with Python) or conda
- **IDE/Editor**: Jupyter Notebook or VS Code with Jupyter extension
- **Disk Space**: Minimum 500 MB free space
- **RAM**: Minimum 2 GB (4 GB recommended)

### Software Prerequisites

Before starting, ensure the following are installed:

From terminal/command prompt:
```bash
python --version         # Verify Python 3.8+
pip --version           # Verify pip is installed
```

### Optional: Git (for version control)
```bash
git --version           # Optional, for project management
```

---

## 2. ENVIRONMENT SETUP

### Step 2.1: Create Project Directory

```bash
# Navigate to desired location
cd "path/to/your/projects"

# Create project directory
mkdir prediksi_lulus
cd prediksi_lulus
```

### Step 2.2: Create Virtual Environment

**Option A: Using pip (Recommended)**

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate
```

**Option B: Using conda**

```bash
# Create conda environment
conda create -n prediksi_lulus python=3.9

# Activate environment
conda activate prediksi_lulus
```

### Step 2.3: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Verify installation
python -c "import pandas, numpy, sklearn, streamlit; print('All dependencies installed successfully')"
```

### Step 2.4: Data Verification

```bash
# Verify dataset exists
python -c "
import pandas as pd
import os

if os.path.exists('data/dataset.csv'):
    df = pd.read_csv('data/dataset.csv')
    print(f'Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns')
else:
    print('ERROR: dataset.csv not found in data/ directory')
"
```

---

## 3. PROJECT STRUCTURE

### Directory Organization

```
prediksi_lulus/
├── notebook/                           # Analysis notebooks
│   ├── 1_EDA.ipynb                    # Exploratory Data Analysis
│   ├── 2_FeatureEngineering.ipynb     # Feature Engineering
│   ├── 3_ModelEvaluation.ipynb        # Model Training
│   └── 4_Visualization.ipynb          # Advanced Visualization
│
├── data/
│   └── dataset.csv                    # Training dataset (50 records)
│
├── dashboard/
│   └── app.py                         # Streamlit web application
│
├── src/
│   └── train_model.py                 # Model training script
│
├── savemodel/                          # Generated during training
│   ├── model_random_forest.pkl
│   └── label_encoder.pkl
│
├── requirements.txt                    # Python dependencies
├── .gitignore                         # Git exclusion rules
├── README.md                          # Quick start guide
├── PROJECT_OVERVIEW.md                # Project description
├── METHODOLOGY.md                     # Technical methodology
└── EXECUTION_GUIDE.md                 # This file
```

### File Descriptions

**Notebooks**:
- **1_EDA.ipynb**: Initial data exploration, quality assessment, distribution analysis
- **2_FeatureEngineering.ipynb**: Feature creation, selection ranking, data scaling
- **3_ModelEvaluation.ipynb**: Model training, evaluation, hyperparameter optimization
- **4_Visualization.ipynb**: Advanced plots, ROC curves, model interpretation

**Code**:
- **train_model.py**: Standalone training script for model retraining
- **app.py**: Streamlit dashboard for predictions and analysis

**Data**:
- **dataset.csv**: Input data with student records (columns: ipk, absen, kegiatan, lulus_tepat_waktu)

---

## 4. NOTEBOOK EXECUTION SEQUENCE

### Recommended Workflow

Execute notebooks in the following order:

```
1_EDA.ipynb
    ↓
2_FeatureEngineering.ipynb
    ↓
3_ModelEvaluation.ipynb
    ↓
4_Visualization.ipynb
    ↓
Complete
```

### Execution Timeline

| Notebook | Duration | Purpose |
|----------|----------|---------|
| 1_EDA.ipynb | 3-5 min | Data understanding |
| 2_FeatureEngineering.ipynb | 5-10 min | Feature preparation |
| 3_ModelEvaluation.ipynb | 15-30 min | Model training and optimization |
| 4_Visualization.ipynb | 5-10 min | Interpretation and insights |
| **TOTAL** | **30-55 min** | Complete analysis |

### Steps to Execute Notebooks

**Using Jupyter**:
```bash
# Launch Jupyter
jupyter notebook

# Select 1_EDA.ipynb from browser interface
# Click "Run" or press Shift+Enter for each cell
```

**Using VS Code**:
```bash
# Open VS Code
code .

# Open first notebook (1_EDA.ipynb)
# Select Python kernel when prompted
# Execute cells sequentially
```

---

## 5. DETAILED NOTEBOOK WALKTHROUGHS

### NOTEBOOK 1: 1_EDA.ipynb
**Phase**: Exploratory Data Analysis  
**Expected Duration**: 3-5 minutes

**Objectives**:
- Load dataset and inspect structure
- Assess data quality
- Analyze feature distributions
- Evaluate correlations
- Perform statistical tests
- Create visualizations

**Key Sections**:
1. **Data Loading**: Read CSV and display shape/metadata
2. **Quality Assessment**: Check missing values, duplicates, outliers
3. **Descriptive Statistics**: Calculate mean, std dev, skewness, kurtosis
4. **Correlation Analysis**: Compute Pearson correlation matrix
5. **Visualization**: Distribution plots, box plots, scatter plots, heatmaps
6. **Statistical Testing**: T-tests comparing graduation groups

**Expected Console Output**:
```
DATASET OVERVIEW
Dataset Shape: (50, 4)
Missing Values: 0
Data Types: ipk (float64), absen (int64), kegiatan (int64), lulus_tepat_waktu (object)

Correlation (IPK vs Absensi): -0.93 (Very Strong)
T-test p-value (IPK): < 0.001 (Highly Significant)
T-test p-value (Absensi): < 0.001 (Highly Significant)
```

**Success Criteria**:
- No errors during execution
- Data quality report generated
- Visualizations display correctly
- Statistical tests complete successfully

**Next Step**: Proceed to Notebook 2

---

### NOTEBOOK 2: 2_FeatureEngineering.ipynb
**Phase**: Feature Engineering and Selection  
**Expected Duration**: 5-10 minutes

**Objectives**:
- Create engineered features
- Rank features by importance
- Scale numerical features
- Prepare datasets for modeling

**Key Sections**:
1. **Feature Creation**: Generate ratio, polynomial, and categorical features
2. **Feature Selection Analysis**: Apply 4 selection methods (F-statistic, mutual information, permutation, correlation)
3. **Consensus Ranking**: Aggregate rankings across methods
4. **Feature Scaling**: Standardize features using StandardScaler
5. **Output Generation**: Save prepared datasets

**Features Created**:
- Ratio: ipk_per_absence, activity_efficiency
- Polynomial: ipk², absen², kegiatan² and interactions
- Categorical: attendance_risk, ipk_category

**Output Files**:
- `features_engineered_scaled.csv`: All 12 engineered features
- `features_top7_scaled.csv`: Top-7 consensus ranking
- `feature_engineering_summary.txt`: Summary statistics

**Expected Console Output**:
```
FEATURE ENGINEERING SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Original Features: 3
Engineered Features: 12 (9 new + 3 original)

TOP 7 CONSENSUS RANKING:
1. ipk                     (Score: 1.0)
2. absen                   (Score: 0.95)
3. kegiatan                (Score: 0.92)
4. ipk_per_absence        (Score: 0.88)
5. ipk_squared            (Score: 0.82)
6. absen_squared          (Score: 0.78)
7. activity_efficiency    (Score: 0.75)
```

**Success Criteria**:
- Feature engineering completes without errors
- Output CSV files created successfully
- Features ranked consistently across methods
- Scaling applied properly (mean ≈ 0, std ≈ 1)

**Next Step**: Proceed to Notebook 3

---

### NOTEBOOK 3: 3_ModelEvaluation.ipynb
**Phase**: Model Training and Evaluation  
**Expected Duration**: 15-30 minutes

**Important Note**: Grid search hyperparameter tuning is computationally intensive. First execution may take 20-30 minutes; subsequent runs will be faster.

**Objectives**:
- Train six machine learning models
- Evaluate each model comprehensively
- Optimize hyperparameters via grid search
- Compare model performance
- Select best model
- Generate trained model artifacts

**Models Trained**:
1. Logistic Regression (baseline / interpretable)
2. Decision Tree (simple / explainable)
3. Random Forest (primary candidate)
4. Gradient Boosting (strong learner)
5. Support Vector Machine (non-linear boundaries)
6. Neural Network (complex patterns)

**Evaluation Process**:
1. **Train-Test Split**: 80-20 stratified split
2. **Cross-Validation**: 5-fold stratified CV
3. **Metrics Calculation**: Accuracy, precision, recall, F1, ROC-AUC
4. **Hyperparameter Optimization**: Grid search over parameter space
5. **Final Evaluation**: Report best model performance

**Expected Outputs**:
```
MODEL PERFORMANCE COMPARISON
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model                 | Accuracy | F1-Score | ROC-AUC | CV Score
──────────────────────┼──────────┼──────────┼─────────┼──────────
Random Forest         | 1.0000   | 1.0000   | 1.0000  | 0.9800
Gradient Boosting     | 0.9000   | 0.8900   | 0.9500  | 0.9200
Support Vector Machine| 0.8500   | 0.8300   | 0.9000  | 0.8800
...

BEST MODEL: Random Forest
Optimized Hyperparameters:
  n_estimators: 100
  max_depth: 10
  min_samples_split: 5
  min_samples_leaf: 1

Model files saved:
  model_random_forest.pkl
  label_encoder.pkl
```

**Generated Models**:
- `model_random_forest.pkl`: Best trained model
- `label_encoder.pkl`: Encoder for target variable

**Success Criteria**:
- All 6 models train successfully
- Cross-validation completes
- Hyperparameter optimization finishes
- Model artifacts saved
- Performance comparison table generated

**Duration Note**: Grid search optimization typically takes 15-30 minutes. Monitor progress in notebook output.

**Next Step**: Proceed to Notebook 4

---

### NOTEBOOK 4: 4_Visualization.ipynb
**Phase**: Advanced Visualization and Interpretation  
**Expected Duration**: 5-10 minutes

**Objectives**:
- Create publication-quality visualizations
- Analyze model performance in depth
- Generate actionable insights
- Create risk stratification categories

**Visualizations Generated**:
1. **ROC Curve**: Shows model discrimination ability across thresholds
2. **Confusion Matrix**: Classification results breakdown
3. **Feature Importance**: Ranking of features by predictive power
4. **Probability Distribution**: Score distribution by class
5. **Risk Stratification**: Categorization of students into risk levels

**Risk Categories**:
- **High Risk** (probability 0.0-0.3): Priority intervention
- **Medium Risk** (probability 0.3-0.7): Standard support
- **Low Risk** (probability 0.7-1.0): Maintain current level

**Expected Console Output**:
```
MODEL PERFORMANCE VISUALIZATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ROC-AUC Score: 1.0000 (Perfect Classification)

CONFUSION MATRIX (Test Set):
                 Predicted Delay | Predicted On-Time
Actual Delay           5        |         0
Actual On-Time         0        |         5

RISK STRATIFICATION (Full Dataset):
High Risk (0.0-0.3):     15 students (30%)
Medium Risk (0.3-0.7):   16 students (32%)
Low Risk (0.7-1.0):      19 students (38%)
```

**Deliverables**:
- ROC curve plot
- Confusion matrix heatmap
- Feature importance barplot
- Probability distribution histograms
- Risk stratification summary

**Success Criteria**:
- All visualizations generate without errors
- Plots are clear and informative
- Risk categories are well-distributed
- Output ready for stakeholder presentation

**Project Completion**: Notebook 4 marks completion of analysis phase

---

## 6. SYSTEM DEPLOYMENT

### Option A: Streamlit Web Application (Recommended)

**Why Use Streamlit**:
- Interactive user interface
- Real-time predictions
- No coding required for end users
- Built-in data visualization
- Easy deployment

**Launch Application**:

```bash
# Activate virtual environment (if not already active)
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Start Streamlit application
streamlit run dashboard/app.py
```

**Access Application**:
- Browser automatically opens to `http://localhost:8501`
- If not, manually navigate to above URL

**Application Features**:

**Tab 1: Individual Prediction**
- Input single student metrics (IPK, Absensi, Kegiatan)
- Receive prediction (Lulus/Tidak Lulus) with confidence score
- View personalized recommendations
- Compare to dataset statistics

**Tab 2: Batch Prediction**
- Upload CSV file with multiple students
- Process all records simultaneously
- Download results with predictions
- View summary statistics

**Tab 3: Model Analysis**
- Feature importance visualization
- Data distribution plots
- Correlation analysis
- Model performance metrics

**Stopping Application**:
```bash
# Press Ctrl+C in terminal to stop
```

### Option B: Standalone Python Script

For programmatic access without web interface:

```bash
python src/train_model.py
```

This script:
- Loads training data
- Trains Random Forest model
- Evaluates on test set
- Saves trained model
- Prints performance metrics

### Option C: Integration into Existing Systems

To integrate predictions into other applications:

```python
import joblib
import numpy as np

# Load trained model and encoder
model = joblib.load('savemodel/model_random_forest.pkl')
encoder = joblib.load('savemodel/label_encoder.pkl')

# Prepare student data
student_data = np.array([[3.75, 2, 5]])  # [IPK, Absensi, Kegiatan]

# Make prediction
prediction_encoded = model.predict(student_data)[0]
prediction_text = encoder.inverse_transform([prediction_encoded])[0]
prediction_probability = model.predict_proba(student_data)[0]

print(f"Prediction: {prediction_text}")
print(f"Confidence: {max(prediction_probability):.2%}")
```

---

## 7. TROUBLESHOOTING GUIDE

### Issue 1: Module Import Errors

**Error**:
```
ModuleNotFoundError: No module named 'sklearn'
```

**Solutions**:
```bash
# Ensure virtual environment is activated
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Verify installation
python -c "import sklearn; print(sklearn.__version__)"
```

### Issue 2: Data File Not Found

**Error**:
```
FileNotFoundError: data/dataset.csv
```

**Solutions**:
```bash
# Check file exists
ls data/dataset.csv  # macOS/Linux
dir data\dataset.csv  # Windows

# Verify file path in notebook matches actual location
# Check for typos in filename (case-sensitive on Linux/macOS)
```

### Issue 3: Slow Notebook 3 Execution

**Expected**: 15-30 minutes for hyperparameter grid search

**If Significantly Slower**:
```python
# Reduce parameter search space in notebook cell

# Current (full search):
parameters = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Reduced (faster testing):
parameters = {
    'n_estimators': [100],
    'max_depth': [10],
    'min_samples_split': [5],
    'min_samples_leaf': [1]
}
```

### Issue 4: Jupyter Kernel Crashes

**Solution**:
```bash
# Restart kernel in Jupyter UI
# Kernel → Restart Kernel

# Or from command line:
jupyter kernelspec list
jupyter kernelspec remove unwanted_kernel
```

### Issue 5: Memory Issues

**Symptoms**: Slow execution, system freezing

**Solutions**:
```bash
# Close other applications

# Reduce cross-validation folds (in code):
cv_score = cross_val_score(model, X, y, cv=3)  # was cv=5

# Limit to primary model only (skip comparisons)
```

### Issue 6: Streamlit Port Already in Use

**Error**:
```
Error: Port 8501 already in use
```

**Solution**:
```bash
# Use different port
streamlit run dashboard/app.py --server.port 8502
```

### Issue 7: Missing Requirements Error

**Solution**:
```bash
# Ensure requirements.txt is in project root
# Reinstall all packages
pip install --force-reinstall -r requirements.txt
```

---

## 8. NEXT STEPS AND RECOMMENDATIONS

### Immediate Actions (Upon Completion)

1. **Review Results**
   - Examine all notebook outputs
   - Study model performance metrics
   - Review visualizations and insights

2. **Model Validation**
   - Verify accuracy on test set
   - Check cross-validation stability
   - Review predictions for business sense

3. **Documentation Review**
   - Read PROJECT_OVERVIEW.md for context
   - Study METHODOLOGY.md for technical details
   - Reference this guide for execution steps

### Short-Term Recommendations

1. **Collect Additional Data**
   - Expand beyond sample dataset (50 records)
   - Include more institutions for validation
   - Add temporal data (longitudinal trends)

2. **Fairness and Bias Analysis**
   - Test for demographic disparities
   - Check prediction consistency across subgroups
   - Document any biases discovered

3. **Pilot Intervention Program**
   - Select high-risk student cohort
   - Implement targeted support
   - Measure graduation rate improvement

### Medium-Term Enhancements

1. **Feature Expansion**
   - Incorporate additional academic factors
   - Add socioeconomic indicators
   - Include prior academic history

2. **Model Improvements**
   - Explore advanced architectures (LSTM for temporal data)
   - Implement ensemble models
   - Develop confidence intervals for predictions

3. **System Integration**
   - Connect to university information system
   - Build automated alerts for high-risk students
   - Create dashboard for academic advisors

### Production Deployment

1. **Model Monitoring**
   - Track prediction accuracy over time
   - Detect model drift
   - Schedule retraining cycles

2. **Performance Tracking**
   - Monitor graduation rate changes
   - Measure intervention effectiveness
   - Report metrics to leadership

3. **Continuous Improvement**
   - Gather feedback from users
   - Refine risk thresholds based on results
   - Update model with new data quarterly

---

## GETTING HELP

### Documentation Resources

- **README.md**: Quick start and feature overview
- **PROJECT_OVERVIEW.md**: Project objectives and novelty
- **METHODOLOGY.md**: Technical methodology and algorithms
- **EXECUTION_GUIDE.md**: This file - execution instructions

### Common Commands Reference

```bash
# Activate environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Run Jupyter
jupyter notebook

# Start Streamlit app
streamlit run dashboard/app.py

# Retrain model
python src/train_model.py

# Check dependencies
pip list

# Update dependencies
pip install --upgrade -r requirements.txt
```

---

## PROJECT COMPLETION CHECKLIST

- Notebooks 1-4 executed successfully
- All output files generated without errors
- Model artifacts created (pkl files)
- Web application launches successfully
- Predictions generated and validated
- Documentation reviewed
- Next steps identified and planned

**Upon completion of all items, the project is ready for institutional deployment.**

---

**Version**: 1.0  
**Last Updated**: March 2026  
**Status**: Ready for Execution ✅
