# Cluade Code Project Context
This file provides context for Claude Code when working on the project

## Project Overview

**Project Type**: Machine Learning Research - Classification models

**Project Goal**: Compare multiple types of Classification models

## Project Context

This is a project where given a data set will use four Classification models and compare the models using metrics.

## Project Directory Setup
Lab3/
├──CLAUDE.md
├──Classification.ipynb
├──README.md
├──requirements.txt
└──visualizations/

## ML Preferences

### ML Framework and Tools
- **Primary Framework**: scikit-learn
- **Visualization**: matplotlib and seaborn
- **Hyperparameter Tuning**: GridSearchCV or RandomizedSearchCV

### Model Selection Philosophy
- Logistic Regression for a baseline model
- Gradient Boosting
- Decision Tree
- Naive Bayes
- Support Vector Machine

### Validation Strategy
- **CRITICAL**: Never touch the test set during development
- Use validation set for:
  - Hyperparameter tuning
- Only evaluate on test set once for final results

### Evaluation Metrics
- Compare models with 
    - Accuracy (with discussion of when it's misleading)
    - Precision (per class and macro/weighted average)
    - Recall/Sensitivity (per class and macro/weighted average)
    - F1-Score (per class and macro/weighted average)
    - AUC-ROC (Area Under ROC Curve)
    - AUC-PR (Area Under Precision-Recall Curve - important for imbalanced data)
    - Confusion Matrix
    - training time
    - average prediction time
- Should be orginized in a grid with the models as the rows and the metrics as the columns
- Example Grid

| Model             | Accuracy | Precision | Recall | F1    | AUC-ROC | Train Time | Avg. Prediction Time |
|-------------------|----------|-----------|--------|-------|---------|------------|----------------------|
| Logistic Reg      | 0.92     | 0.85      | 0.78   | 0.81  | 0.94    | 0.12s      | .09s                 |
| Gradient Boosting | 0.94     | 0.88      | 0.82   | 0.85  | 0.96    | 2.45s      | .32s                 |
| ...               | ...      | ...       | ...    | ...   | ...     | ...        | ...                  |

## File Setup
- The project will be written in one jupyter notebook
- The notebook will be written into 6 sections
- The notebook will also have Markdown notes through out the file
- Vizualizations should be stored in the Vizualizations folder

### Section One
- **Will be used for Exploratory data analysis and data-preprocessing**
- Will be split into two subsections

#### Section 1.1
- **Used for Exploratory Data Analysis**
- Statistical summary of all features
- Correlation matrix/heatmap
- Identification of outliers
- Missing value analysis
- Class distribution analysis
- Distribution plots for key numerical features
- Categorical feature analysis (unique values, frequency)

#### Section 1.2
- **Used for the Data-preprocessing pipeline**
- Used to handle missing values
- Encode categorical variables (one-hot, label encoding, etc.)
- Feature scaling/normalization
- Address class imbalance (if applicable):
    - SMOTE (Synthetic Minority Over-sampling)
    - Undersampling
    - Class weights
- Two Train/test split with stratification (one with preprocessing and one without preprocessing)

### Section Two
- **Will be used for the baseline Logistic Regression**
- Training and test performance metrics
- Visualizations
    - Residual plot
    - Confusion Matrix
    - Classification Report (precision, recall, F1 per class)
    - ROC Curve and AUC (for binary classification)

### Section Three
- **Will be used for the other four classification models**
- Train all four other classification models with default hyerparameters
- Record Training time
- Calculate Evaluation Metrics

### Section Four
- **Will be used for displaying the Evaluation metrics**
- Visualizations
    - Metrics Comparison table
    - Bar Charts: Compare metrics across models
    - Confusion Matrices: For ALL models (use subplots)
    - ROC Curves: Plot all models on same graph for comparison
    - Precision-Recall Curves: Especially important for imbalanced datasets
    - Learning Curves: For your top 2 models
        - Plot training vs validation score as function of training size

### Section Five
- **Will be used for hyperparameter tuning**
- Use the best performing model
- Use GridSearchCV or RandomizedSearchCV to tune hyperparameters
- document:
    - Initial Hyperparameters vs optimal tuned hyperparameters
    - Performence imporovment(before/after tuning)
    - Cross-Validation Scores
    - Training time comparison

### Section Six
- **Will be used for Big O notation testing**
- Test for the approxiate Big O notation for training and predication of all models
- Print out the results into a table
- Visualizations
    - Graph of all models training Big O notation
    - Graph of all models prediction Big O notation

## Data Sources

Credit Card Fraud Detection from Kaggle

### Getting Data from Data Sources
Use the following code to get the data
```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
```
## Testing Strategy

### Unit Tests
- **Always unit test all functions to ensure compatability**
- Test data loading functions
- Test preprocessing functions
- Test model initialization
- Mock heavy operations (model training)

### Integration Tests
- Test full pipeline end-to-end
- Use small sample dataset
- Verify outputs are generated correctly

### Model Tests
- Sanity checks (model can overfit small dataset)
- Invariance tests (predictions stable for same input)
- Shape checks (correct output dimensions)

## Performance Targets

### Accuracy Goals
- **Minimum acceptable**: 75% accuracy
- **Target**: 85% accuracy
- **Stretch goal**: 90% accuracy

## Git
- **The project will have a git repository**
- Eveytime a update is made to the project you will make a commit and explain the updates
- **Never commit raw data to git**

## Miscellaneous

### README
- The Readme should be updated everytime a change is made to the project
- The Redme will contain all pertinent information

### Requirements.txt
- The requriements.txt will be updated everytime a library is added or removed from the project

**Last Updated**: 2025-11-4

**Note**: This file should be updated whenever project goals, standards, or preferences change.
