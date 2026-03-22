# Student Graduation Prediction System

A comprehensive machine learning system for predicting on-time student graduation using academic performance indicators and engagement metrics.

## Overview

This project develops a predictive machine learning system to forecast whether a student will graduate on time based on three key academic factors:

- **IPK** (Indeks Prestasi Kumulatif): Cumulative grade point average
- **Absensi**: Number of absences during the semester
- **Kegiatan**: Level of participation in academic activities and organizations

The system achieves high accuracy through ensemble learning methods and provides actionable insights for academic early intervention programs.

## System Features

- **Individual Prediction Module**: Real-time prediction for single students with confidence scores
- **Batch Prediction System**: Bulk prediction from CSV uploads with comprehensive statistics
- **Data Analysis Dashboard**: Interactive visualizations of model performance and data distributions
- **Web Application**: User-friendly Streamlit interface for all functionality
- **Machine Learning Models**: Multiple trained models with Random Forest as primary classifier
- **Risk Categorization**: Automatic classification of students into risk categories
- **Recommendation Engine**: Personalized intervention suggestions based on predicted risk levels

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Installation

```bash
# Clone or navigate to project directory
cd prediksi_lulus

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Train Model

```bash
python src/train_model.py
```

This generates the trained model files:
- `savemodel/model_random_forest.pkl` - Trained classifier
- `savemodel/label_encoder.pkl` - Target variable encoder

### Run Web Application

```bash
streamlit run dashboard/app.py
```

The application opens at `http://localhost:8501` with three main tabs:

1. **Individual Prediction**: Single student prediction with risk assessment
2. **Batch Prediction**: Upload CSV file for multiple predictions
3. **Model Analysis**: Performance metrics, feature importance, and data visualization

## Project Structure

```
prediksi_lulus/
├── notebook/                           # Jupyter notebooks for analysis
│   ├── 1_EDA.ipynb                    # Exploratory Data Analysis
│   ├── 2_FeatureEngineering.ipynb     # Feature engineering and selection
│   ├── 3_ModelEvaluation.ipynb        # Model training and comparison
│   └── 4_Visualization.ipynb          # Advanced visualization and insights
│
├── data/
│   └── dataset.csv                    # Training dataset (50 students)
│
├── dashboard/
│   └── app.py                         # Streamlit web application
│
├── src/
│   └── train_model.py                 # Model training script
│
├── savemodel/                          # Generated model files
│   ├── model_random_forest.pkl        # Trained model
│   └── label_encoder.pkl              # Label encoder
│
├── requirements.txt                    # Python dependencies
├── .gitignore                         # Git configuration
├── README.md                          # This file
├── PROJECT_OVERVIEW.md                # Detailed project overview
├── METHODOLOGY.md                     # Technical methodology
└── EXECUTION_GUIDE.md                 # Step-by-step execution guide
```

## Data Format

### Input Data Structure

For both training and batch prediction, use CSV format with these columns:

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| ipk | float | 0.0 - 4.0 | Cumulative grade point average |
| absen | integer | 0 - 30 | Number of absences in semester |
| kegiatan | integer | 0 - 10 | Activity participation score |
| lulus_tepat_waktu | string | Ya/Tidak | Target: On-time graduation (training only) |

### Example CSV Format

```csv
ipk,absen,kegiatan,lulus_tepat_waktu
3.75,2,5,Ya
3.45,8,3,Tidak
3.90,1,8,Ya
2.95,15,1,Tidak
```

For batch prediction without target variable:

```csv
ipk,absen,kegiatan
3.75,2,5
3.45,8,3
3.90,1,8
```

## Model Architecture

### Primary Model: Random Forest Classifier

**Hyperparameters:**
- n_estimators: 100 decision trees
- max_depth: 10 levels per tree
- Stratified train-test split: 80/20 distribution
- Class balance: Preserved across all splits

**Performance Metrics:**
- Accuracy: High precision on test set
- Cross-validation: Stratified K-Fold validation
- Feature importance: IPK > Absen > Kegiatan

### Alternative Models Evaluated

- Logistic Regression (baseline)
- Decision Tree
- Gradient Boosting
- Support Vector Machine
- Neural Network

See METHODOLOGY.md for detailed comparative analysis.

## Feature Importance Ranking

Based on the trained Random Forest model:

1. **IPK** (40-50%): Most influential factor. Strong positive relationship with graduation success.
2. **Absensi** (30-40%): Second most important. Higher absences correlate with increased risk.
3. **Kegiatan** (10-20%): Supporting factor. Activity engagement provides context to academic performance.

## Usage Examples

### Web Application Tabs

#### Tab 1: Individual Prediction
- Input single student metrics (IPK, Absensi, Kegiatan)
- Receive prediction with confidence score
- View personalized recommendations
- Access baseline statistics for context

#### Tab 2: Batch Prediction
- Upload CSV file with multiple students
- Process all records simultaneously
- Export results with predictions and confidence scores
- View summary statistics

#### Tab 3: Model Analysis
- Feature distributions and statistics
- Model performance metrics
- Feature importance visualization
- Correlation analysis

### Interpretation Guide

**Prediction Result: Ya (On-Time Graduation)**
- Student meets expected academic standards
- Action: Maintain current performance levels
- Focus on consistency in attendance and engagement

**Prediction Result: Tidak (Delayed Graduation Risk)**
- Student shows risk indicators
- Action: Implement early intervention strategies
- Priority: Address lowest-performing metrics first

## Dependencies

See `requirements.txt` for complete list. Main packages:

- pandas: Data manipulation and analysis
- numpy: Numerical computing
- scikit-learn: Machine learning algorithms
- streamlit: Web application framework
- matplotlib: Data visualization
- seaborn: Statistical visualization
- joblib: Model serialization

## Running Custom Analyses

### Using the Training Script

```bash
python src/train_model.py
```

Outputs:
- Model accuracy and classification metrics
- Feature importance rankings
- Cross-validation scores
- Trained model and encoder files

### Using Jupyter Notebooks

Execute notebooks in order:

```bash
jupyter notebook notebook/1_EDA.ipynb
jupyter notebook notebook/2_FeatureEngineering.ipynb
jupyter notebook notebook/3_ModelEvaluation.ipynb
jupyter notebook notebook/4_Visualization.ipynb
```

## Configuration and Customization

### Modifying Data Path

The application uses absolute paths for flexibility. To change data location, edit `dashboard/app.py`:

```python
PATHS = {
    'model': os.path.join(PROJECT_DIR, 'savemodel', 'model_random_forest.pkl'),
    'encoder': os.path.join(PROJECT_DIR, 'savemodel', 'label_encoder.pkl'),
    'data': os.path.join(PROJECT_DIR, 'data', 'dataset.csv'),
}
```

### Retrain with New Data

1. Prepare CSV with required columns (see Data Format section)
2. Replace `data/dataset.csv` with your data
3. Run `python src/train_model.py`
4. Restart the web application

## Performance Metrics

The model evaluation includes:

- **Accuracy**: Percentage of correct predictions
- **Precision**: Accuracy of positive predictions
- **Recall**: Detection rate of actual positive cases
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under receiver operating characteristic curve
- **Cross-validation**: Stratified K-Fold validation (k=5)

For detailed metrics, see METHODOLOGY.md.

## Troubleshooting

### Model File Not Found

Ensure model files exist in `savemodel/` directory:
```bash
python src/train_model.py
```

### Port Already in Use

If port 8501 is occupied:
```bash
streamlit run dashboard/app.py --server.port 8502
```

### Dependency Issues

Reinstall all dependencies:
```bash
pip install --upgrade -r requirements.txt
```

## Project Documentation

- **PROJECT_OVERVIEW.md**: Project context, objectives, and novelty
- **METHODOLOGY.md**: Technical approach, feature engineering, model details
- **EXECUTION_GUIDE.md**: Detailed step-by-step execution instructions

## Development Workflow

1. Data exploration and analysis (1_EDA.ipynb)
2. Feature engineering and selection (2_FeatureEngineering.ipynb)
3. Model training and evaluation (3_ModelEvaluation.ipynb)
4. Advanced visualization and insights (4_Visualization.ipynb)
5. Web application deployment (streamlit run dashboard/app.py)

## Contact and Support

For issues, questions, or contributions, refer to project documentation in the root directory.

1. **IPK Target**: Minimal 3.0, ideal ≥ 3.5
2. **Absen Target**: Maksimal 5 hari per semester
3. **Kegiatan Target**: Minimal 3 (aktif dalam 3+ kegiatan/organisasi)

---

##  Log Training

Setiap kali menjalankan `train_model.py`, akan ditampilkan:
- ✓ Loading data progress
- ✓ Data split information
- ✓ Model training progress
- ✓ Accuracy dan classification metrics
- ✓ Feature importance ranking

---

##  Troubleshooting

### Model tidak ditemukan
```
FileNotFoundError: model_random_forest.pkl
```
→ Jalankan `python train_model.py` terlebih dahulu

### Port 8501 sudah digunakan
```
streamlit run app.py --server.port 8502
```

### Error saat upload CSV
→ Pastikan format CSV sesuai dengan: `ipk,absen,kegiatan`

---

## 📚 Teknologi yang Digunakan

- **Python 3.8+**
- **Scikit-learn** - Machine Learning
- **Pandas** - Data processing
- **Streamlit** - Web application
- **Joblib** - Model serialization
- **Matplotlib/Seaborn** - Visualization

---

##  Developer Notes

Model dapat dikustomisasi dengan:
- Menambah/mengurangi jumlah features
- Mengubah hyperparameter di `train_model.py`
- Menggunakan algoritma lain (Logistic Regression, SVM, dll)

---

##  Support & Feedback

Untuk pertanyaan atau improvement, silakan komunikasikan kebutuhan Anda!

---

**Last Updated**: 2026-03-20
**Version**: 1.0
