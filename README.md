# MLalgos
I will be posting different ML prediction models in this repo.

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
