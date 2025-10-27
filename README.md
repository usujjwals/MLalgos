# MLalgos
I will be posting different ML prediction models in this repo.

# Model: Multiple linear regression 
## Multiple Linear Regression for Toyota Corolla Price Prediction

This project is a comprehensive analysis of multiple linear regression techniques, likely originating from a chapter in the *Data Mining for Business Analytics* textbook. The goal is to predict the price of a used Toyota Corolla based on various car specifications.

The Jupyter Notebook explores data preparation, model fitting, model validation, and advanced techniques for variable selection and regularization.

### 📊 Dataset

The analysis is performed on the `ToyotaCorolla.csv` dataset. The notebook uses the first 1000 rows for the analysis.

* **Outcome (Target Variable):** `Price`
* **Predictors:** `Age_08_04`, `KM`, `Fuel_Type`, `HP`, `Met_Color`, `Automatic`, `CC`, `Doors`, `Quarterly_Tax`, `Weight`

### 📈 Analysis & Methodology

The notebook follows a structured workflow to build, evaluate, and refine a linear regression model.

#### 1. Baseline Model (Tables 6.3 & 6.4)

* **Data Preparation:** The categorical variable `Fuel_Type` is converted into dummy variables.
* **Data Partitioning:** The dataset is split into a 60% training set and a 40% validation set (`random_state=1`).
* **Model Fitting:** A standard `sklearn.linear_model.LinearRegression` model is fit to the training data.
* **Performance Evaluation:** The model's performance is evaluated on both the **training data** (Table 6.3) and the **validation data** (Table 6.4) using key metrics:
    * Root Mean Squared Error (RMSE)
    * Mean Absolute Error (MAE)
    * Mean Absolute Percentage Error (MAPE)
    * Adjusted R-squared
    * AIC (Akaike Information Criterion)
    * BIC (Bayesian Information Criterion)

#### 2. Residual Analysis (Figure 6.1)

* The residuals (the difference between actual and predicted prices) on the validation set are calculated and plotted in a histogram to assess the model's error distribution.

#### 3. Variable Selection Strategies (Tables 6.5 - 6.7)

To find the optimal combination of predictors, four different variable selection methods are applied using the `dmba` library:

* **Exhaustive Search:** Tests all possible combinations of predictors and identifies the best model based on adjusted R-squared (Table 6.5).
* **Backward Elimination:** Starts with all predictors and removes them one by one, keeping the model that performs best based on AIC (Table 6.6).
* **Forward Selection:** Starts with no predictors and adds them one by one, keeping the model that performs best based on AIC (Table 6.7).
* **Stepwise Selection:** A combination of forward and backward selection, also based on AIC.

All three AIC-based methods (Backward, Forward, Stepwise) converge on the same optimal 8-predictor model.

#### 4. Regularized Regression

To handle potential multicollinearity and prevent overfitting, several regularized models are fit using an `sklearn.pipeline` to first apply `StandardScaler`:

* **Lasso (`Lasso`):** Performs L1 regularization.
* **Lasso CV (`LassoCV`):** Automatically finds the best `alpha` (regularization strength) using cross-validation. The results show `LassoCV` shrinking several coefficients to zero, effectively performing feature selection.
* **Ridge (`Ridge`):** Performs L2 regularization.
* **Bayesian Ridge (`BayesianRidge`):** A probabilistic approach to linear regression.

A final data frame compares the coefficients from the standard `LinearRegression`, `LassoCV`, and `BayesianRidge` models, clearly showing the effect of regularization.

#### 5. Statistical Summary (Table 6.10)

* Finally, the full 11-predictor model is fit using `statsmodels.formula.api.ols` to generate a detailed statistical summary.
* This summary provides crucial insights not available in `sklearn`, such as the **p-value (P>|t|)** for each coefficient, allowing for an analysis of which predictors are statistically significant. The high condition number ($2.20e+06$) also suggests strong multicollinearity, reinforcing the need for variable selection or regularization.

## Model: TreeModel_rf_dt_bagging

## eBay Auction Analysis with Bagging and Random Forests

This project is a data science assignment focused on analyzing eBay auction data. The primary objective is to build and evaluate predictive models to classify auction outcomes using ensemble methods, specifically Bagging and Random Forests.

### 📝 Project Overview

The core of this project is the `AssignmentBaggingRF.ipynb` Jupyter Notebook. It details the entire machine learning workflow, including:
* **Data Loading:** Importing the `eBayAuctions.csv` dataset.
* **Data Preprocessing:** encoding categorical features for modeling.
* **Model Building:** Implementing a Bagging Classifier and a Random Forest Classifier.
* **Model Evaluation:** Assessing the performance of the models using metrics like accuracy, precision, recall, and F1-score.
* **Hyperparameter Tuning:** (Potentially) Optimizing the models to achieve the best performance.

### 🗂️ File Structure

* `BaggingRF.ipynb`: The main Jupyter Notebook containing all Python code and analysis.
* `eBayAuctions.csv`: The dataset used for this project, containing information about various eBay auctions.

### 📊 Dataset

The `eBayAuctions.csv` file provides the data for this analysis. While the notebook contains the full data dictionary, the dataset likely includes features such as:
* Item category
* Currency
* Seller rating
* Auction duration
* Starting price
* Ending price
* ...and a target variable (e.g., whether the auction was successful or not).

### ⚙️ Methodology

This project focuses on **ensemble learning** to improve predictive performance over a single model.

1.  **Bagging (Bootstrap Aggregating):** A `BaggingClassifier` (likely using Decision Trees as the base estimator) is trained. Bagging works by training multiple models on different random subsets of the data (with replacement) and averaging their predictions.

2.  **Random Forest:** A `RandomForestClassifier` is also implemented. Random Forest is an extension of bagging that also introduces randomness at the feature selection level for each split, further reducing variance and improving model robustness.

The performance of both models is compared to determine the most effective approach for this dataset.

## Model: KNN 
### k-NN for Personal Loan Acceptance (Universal Bank)

This project is an implementation of Problem 7.2: Personal Loan Acceptance from Chapter 7 (k-Nearest Neighbors) of the book *Data Mining for Business Analytics: Concepts, Techniques, and Applications in Python*.

The objective is to use the k-Nearest Neighbors (k-NN) algorithm to predict whether a customer at Universal Bank will accept a personal loan offer, based on their demographic and banking information. The analysis is performed on the `UniversalBank.csv` dataset, which includes data on 5,000 customers.

### 📊 Dataset

The `UniversalBank.csv` file contains 5,000 customer records. Key attributes used in the analysis include:

* **Predictors:** `Age`, `Experience`, `Income`, `Family`, `CCAvg`, `Mortgage`, `Securities_Account`, `CD_Account`, `Online`, `CreditCard`, and dummy variables for `Education`.
* **Target Variable:** `Personal_Loan` (1 for acceptance, 0 for rejection).

The `ID` and `ZIP Code` columns were excluded from the analysis.

### 📈 Analysis Workflow

The Jupyter Notebook is structured to answer a series of questions from the textbook, following a complete machine learning workflow:

1.  **Data Pre-processing:**
    * Loads the `UniversalBank.csv` dataset.
    * Drops the `ID` and `ZIP Code` columns.
    * Creates dummy variables for the categorical `Education` feature.

2.  **Initial Partition (60/40 Split):**
    * Splits the data into a 60% training set and a 40% validation set (`random_state=1`).
    * Applies `StandardScaler` to normalize the predictor variables (scaler is fit on the training set only).

3.  **Prediction with k=1 (Problem 7.2.a):**
    * Trains a `KNeighborsClassifier` with `k=1`.
    * Classifies a new customer profile (Age=40, Income=84, etc.), which is predicted to **not accept** the loan (Class 0).

4.  **Finding the Optimal k (Problem 7.2.b):**
    * Loops `k` from 1 to 11, training a new model for each `k` and recording its accuracy on the validation set.
    * Plots the validation accuracy vs. `k` to find the optimal value.
    * **Result:** The optimal `k` that balances accuracy and overfitting is found to be **k=3**, with a validation accuracy of ~95.55%.

5.  **Model Evaluation with Best k (Problem 7.2.c & 7.2.d):**
    * Trains a new model using the best `k=3`.
    * Generates a confusion matrix for the validation data.
    * Re-classifies the same new customer profile using the `k=3` model, which is again predicted to **not accept** the loan (Class 0).

6.  **Final Repartition & Comparison (Problem 7.2.e):**
    * Re-partitions the data into **training (50%)**, **validation (30%)**, and **test (20%)** sets.
    * Normalizes all three sets (fitting the `StandardScaler` *only* on the 50% training data).
    * Trains the final `k=3` model on the new training set.
    * Generates and compares the confusion matrices and accuracy scores for all three datasets (train, validation, and test) to check for model generalization.

### Final Model Performance (k=3)

| Dataset | Accuracy |
| :--- | :--- |
| **Training** (50%) | ~97.56% |
| **Validation** (30%) | ~95.47% |
| **Test** (20%) | ~95.80% |
