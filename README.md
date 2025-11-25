# Credit Card Fraud Detection - Classification Model Comparison

A comprehensive machine learning project comparing multiple classification algorithms for detecting fraudulent credit card transactions using a highly imbalanced dataset.

## Project Overview

This project implements and compares five classification models to identify fraudulent credit card transactions. The dataset contains 284,807 transactions with only 0.17% fraud cases, making it an excellent case study for handling class imbalance in machine learning.

### Models Evaluated

1. **Logistic Regression** (Baseline)
2. **Gradient Boosting Classifier**
3. **Decision Tree Classifier**
4. **Gaussian Naive Bayes**
5. **Support Vector Machine (SVM)**

## Dataset

**Source**: [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle (MLG-ULB)

**Size**: 284,807 transactions, 31 features

**Features**:
- V1-V28: PCA-transformed features (anonymized for privacy)
- Time: Seconds elapsed between transactions
- Amount: Transaction amount
- Class: 0 (Normal) or 1 (Fraud)

**Class Distribution**:
- Normal transactions: 284,315 (99.83%)
- Fraudulent transactions: 492 (0.17%)
- Imbalance ratio: 1:577.88

## Project Structure

```
Lab3/
├── Classification.ipynb          # Main Jupyter notebook with all analysis
├── README.md                      # Project documentation (this file)
├── requirements.txt               # Python dependencies
├── CLAUDE.md                      # Project context for Claude Code
├── visualizations/                # Generated visualization outputs
│   ├── class_distribution.png
│   ├── correlation_heatmap.png
│   ├── all_confusion_matrices.png
│   ├── roc_curves_comparison.png
│   ├── pr_curves_comparison.png
│   ├── learning_curves_top2.png
│   ├── model_comparison_metrics.png
│   ├── training_time_complexity.png
│   ├── prediction_time_complexity.png
│   └── [additional visualizations]
├── model_comparison_results.csv   # Exported model performance metrics
└── big_o_complexity_results.csv   # Time complexity analysis results
```

## Notebook Sections

### Section 1: Exploratory Data Analysis and Data Preprocessing

**1.1 Exploratory Data Analysis**
- Dataset structure and statistical summary
- Missing value analysis (none found)
- Class distribution analysis (highly imbalanced)
- Feature distribution analysis
- Outlier detection using IQR method
- Correlation analysis with fraud detection
- Pairwise feature visualizations

**1.2 Data Preprocessing Pipeline**
- Feature selection: V17 and V14 (highest correlation with fraud)
- Train/test split (80/20) with stratification
- Feature scaling using StandardScaler
- Extreme outlier removal (0.21% of data)
- Class balancing using SMOTE (Synthetic Minority Over-sampling)
- Final balanced training set: 453,942 samples (1:1 ratio)

### Section 2: Baseline Logistic Regression Model

- Trained Logistic Regression as baseline classifier
- Comprehensive metrics: Accuracy, Precision, Recall, F1-Score, AUC-ROC, AUC-PR
- Training and prediction time measurements
- Confusion matrix visualization
- ROC and Precision-Recall curves
- Classification reports for train and test sets

### Section 3: Training Additional Classification Models

- Trained 4 additional models with default hyperparameters
- Recorded training and prediction times for each model
- Generated predictions and probability scores
- Created comprehensive metrics comparison table
- Visualized performance across all metrics
- Identified best performing model by each metric

### Section 4: Detailed Evaluation Metrics and Visualizations

- Comprehensive metrics comparison table with color gradients
- Confusion matrices for all 5 models (single visualization)
- ROC curves comparison (all models on one graph)
- Precision-Recall curves comparison
- Learning curves for top 2 performing models
- Detailed performance analysis and insights

### Section 5: Hyperparameter Tuning

- Selected best model based on F1-Score
- Defined comprehensive hyperparameter grids
- GridSearchCV with 5-fold cross-validation
- Optimal vs default hyperparameters comparison
- Performance improvement analysis
- Cross-validation scores visualization
- Training time comparison before/after tuning

### Section 6: Big O Notation Testing

- Empirical time complexity testing framework
- Training complexity analysis across dataset sizes (1k-100k samples)
- Prediction complexity analysis
- Big O estimation using curve fitting (O(1), O(n), O(n log n), O(n²))
- Individual complexity graphs for each model
- Combined comparison visualizations
- Scalability analysis and implications

## Key Features

### Machine Learning Techniques

- **Class Imbalance Handling**: SMOTE (Synthetic Minority Over-sampling Technique)
- **Feature Selection**: Correlation-based selection (V17, V14)
- **Outlier Removal**: IQR method with aggressive threshold (4.5 × IQR)
- **Feature Scaling**: StandardScaler for zero mean and unit variance
- **Model Validation**: Stratified train/test split
- **Hyperparameter Optimization**: GridSearchCV with cross-validation

### Evaluation Metrics

- **Classification Metrics**:
  - Accuracy
  - Precision (per class and macro/weighted average)
  - Recall/Sensitivity (per class and macro/weighted average)
  - F1-Score (per class and macro/weighted average)
  - AUC-ROC (Area Under ROC Curve)
  - AUC-PR (Area Under Precision-Recall Curve)

- **Performance Metrics**:
  - Training time
  - Average prediction time per sample
  - Time complexity (Big O notation)

- **Visual Analysis**:
  - Confusion matrices
  - ROC curves
  - Precision-Recall curves
  - Learning curves
  - Time complexity graphs

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Lab3
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Kaggle API** (for dataset download):
   - Sign up at [Kaggle](https://www.kaggle.com/)
   - Go to Account Settings → API → Create New API Token
   - Place `kaggle.json` in `~/.kaggle/` (Linux/Mac) or `C:\Users\<username>\.kaggle\` (Windows)

## Usage

### Running the Notebook

1. **Start Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

2. **Open `Classification.ipynb`**

3. **Run all cells** (Cell → Run All) or execute cells sequentially

### Dataset Download

The notebook automatically downloads the dataset using `kagglehub`:
```python
import kagglehub
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
```

**Note**: First run requires Kaggle API authentication.

### Expected Runtime

- **Full notebook execution**: 15-30 minutes (depending on hardware)
- **Section 5 (Hyperparameter Tuning)**: 5-15 minutes
- **Section 6 (Big O Testing)**: 5-10 minutes

## Results

### Model Performance Summary

The notebook generates comprehensive performance comparisons across all models. Key metrics include:

- **Accuracy**: Overall prediction correctness
- **Precision**: Fraud detection accuracy (minimize false positives)
- **Recall**: Fraud case capture rate (minimize false negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Overall discrimination ability
- **AUC-PR**: Performance on imbalanced data

### Output Files

- `model_comparison_results.csv`: Complete metrics table
- `big_o_complexity_results.csv`: Time complexity analysis
- `visualizations/`: All generated plots and charts

## Technologies Used

### Core Libraries

- **NumPy**: Numerical computations
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms and tools
- **Imbalanced-learn**: SMOTE implementation for class balancing

### Visualization

- **Matplotlib**: Base plotting library
- **Seaborn**: Statistical data visualization

### Development

- **Jupyter**: Interactive notebook environment
- **Kagglehub**: Dataset download utility

## Performance Targets

- **Minimum Acceptable**: 75% accuracy
- **Target**: 85% accuracy
- **Stretch Goal**: 90% accuracy

**Note**: Due to extreme class imbalance, metrics beyond accuracy (especially Precision-Recall) are more informative.

## Project Methodology

### Data Preprocessing Strategy

1. Feature selection based on correlation analysis
2. Remove extreme outliers from normal transactions only (preserve fraud samples)
3. Scale features to zero mean and unit variance
4. Balance classes using SMOTE on training set only
5. Evaluate on original imbalanced test set (real-world simulation)

### Model Training Strategy

1. Train all models with default hyperparameters first
2. Compare performance across multiple metrics
3. Select best model for hyperparameter tuning
4. Use GridSearchCV with cross-validation for optimization
5. Analyze time complexity for scalability assessment

### Validation Strategy

- **Critical**: Never touch test set during development
- Use balanced training set for model fitting
- Evaluate on imbalanced test set for realistic performance
- Cross-validation during hyperparameter tuning
- Final evaluation only on test set

## Key Insights

### Dataset Characteristics

- Extreme class imbalance (577:1 ratio) requires special handling
- PCA-transformed features provide privacy but limit interpretability
- V17 and V14 show strongest correlation with fraud (-0.33 and -0.30)
- Clear separability in V17 vs V14 feature space

### Model Performance

- All models significantly outperform random baseline
- SMOTE balancing improves fraud detection recall
- Tree-based models generally perform well on this task
- Hyperparameter tuning provides measurable improvements

### Computational Complexity

- Training complexity varies from O(n) to O(n²) across models
- Prediction complexity typically O(n) for most models
- SVM has highest training time for large datasets
- Decision trees offer fastest prediction times

## Limitations

1. **Feature Selection**: Only used 2 features (V17, V14) for simplicity
2. **Class Balancing**: SMOTE may create synthetic samples that don't represent real fraud patterns
3. **Outlier Removal**: Aggressive threshold may remove valid rare cases
4. **Real-world Deployment**: Test set still doesn't reflect true production conditions
5. **Cost-Sensitive Learning**: Didn't implement different costs for false positives vs false negatives

## Future Improvements

1. **Feature Engineering**:
   - Use all PCA features (V1-V28)
   - Create interaction features
   - Temporal feature engineering from Time variable

2. **Advanced Techniques**:
   - Ensemble methods (stacking, blending)
   - Deep learning approaches (neural networks)
   - Anomaly detection algorithms (Isolation Forest, One-Class SVM)
   - Cost-sensitive learning

3. **Model Optimization**:
   - RandomizedSearchCV for faster hyperparameter search
   - Bayesian optimization
   - AutoML frameworks

4. **Evaluation**:
   - Time-based validation split (temporal validation)
   - Cross-validation on imbalanced data
   - Cost-benefit analysis incorporating business metrics

5. **Deployment**:
   - Model serving infrastructure
   - Real-time prediction API
   - Model monitoring and drift detection
   - A/B testing framework

## Contributing

This is an educational project. For suggestions or improvements:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is created for educational purposes. The dataset is provided by MLG-ULB under their terms of use.

## Acknowledgments

- **Dataset**: Machine Learning Group - Université Libre de Bruxelles
- **Original Paper**: Andrea Dal Pozzolo, et al. "Calibrating Probability with Undersampling for Unbalanced Classification." IEEE Symposium Series on Computational Intelligence, 2015.
- **Framework**: Scikit-learn, Imbalanced-learn

## Contact

For questions or feedback about this project, please open an issue on the GitHub repository.

---

**Last Updated**: 2025-01-24

**Project Status**: Complete ✓

**Total Cells**: 131

**Total Visualizations**: 20+

**Code Generated With**: [Claude Code](https://claude.com/claude-code) by Anthropic
