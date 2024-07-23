# Data Science Midterm Project

## Objectives
This project aims to predict house prices in U.S. based on available features using supervised machine learning techniques. The goal is to gain insights from the data and use data visualization tools to present findings effectively.

## Process
### Step 1: Data Preparation
- **Loading, synthesizing and extracting** : Information from available JSON data files for different cities and states in U.S.
- **Exploratory data analysis** : Uncovering data patterns, spot outliers and learn correlation. 
- **Data cleaning/wrangling** : Handling missing values, removing duplicates, correcting data types, and handling outliers.
- **Feature Engineering** : Creating new columns that may improve model performance.

### Step 2: Model selection and Feature selection
- Training multiple baseline models on the preprocessed data: Linear Regression, Support Vector Machines SVM, Random Forest and XGBoost.
- Gathering evaluation metrics and compare results to pick the best-performing model for hyper tuning.
- Performing feature selection to get a reduced subset of the original features for better model scoring.
- Refitting model with smaller feature subset.
- Evaluating the model performance of dimensionality reduction to decide if it be should include feature selection in your final pipeline.

### Step 3: Hyperparameter tuning and pipeline creation
- Performing hyperparameter tuning on the best performing models from Part 2 and saving the tuned model.
- Building a final prediction pipeline and saving it for future usage.

## Key Insights
> - Single Family homes are popular among 90% of the dataset which points towards its commonality among different U.S. residents.
> -  **Boston, Honolulu, Nashville and Springfield** cities seems to have high priced housing with an average $1 Million house sale price.
> - Most of the houses in this dataset belongs to cluster with year build from 1900s to 2020s.
> 

## Model Results
- **Baseline Models :** With no hypertuning and preprocessed data, **XGBoost** is the best performing model based on the following metrics,
    - ***Mean Absolute Error***
    - ***Mean Squared Error***
    - ***Root Mean Squared Error***
    - ***R2 or Coefficient of Determination***
    - ***Adjusted R2***
- **XGBoost** contributed the ***lowest MAE and RMSE alongwith highest R2 and Adjusted R2 values*** despite the existence of overfitting. These metrics provided enough evidence to consider XGBoost hypertuning at this point to explore the opportunities to better train our model.
- This model was followed by **Random Forest** which lesser overfitting to training model and displayed similar results to XGBoost.
- **Linear Regression** is worth considering model as it achieves ***positive results with no overfitting*** to the training data despite the performance is lower than XGBoost and Random Forest comparatively.
<p align="center" style="margin-top: 20px; margin-bottom: 20px;">
<img width="60%" src="https://github.com/ThuyTran102/DS-Midterm-Project/blob/main/images/baseline_models_comparison.png" alt="Image1"></img>
</p>

- Features reduction with low correlation with the target variable not granting any higher stake in model performance and could prone the model to overfit the training data.
- This project has three baselines performing well and hypertuned : XGBoost, Random Forest and Linear Regression. The performance of each model by a lower volumne yet significant increase in each evaulation metric.
- ***Overall XGBoost model was the winner in exceeding significant metrics and predicting the best out of all models.***

<p align="center" style="margin-top: 20px; margin-bottom: 20px;">
<img width="60%" src="https://github.com/ThuyTran102/DS-Midterm-Project/blob/main/images/tuned_XGBoost.png" alt="Image2"></img>
</p>
<p align="center" style="margin-bottom: 20px;">
<img src="https://github.com/ThuyTran102/DS-Midterm-Project/blob/main/images/tuned_XGBoost_graph.png" alt="Image3"></img>
</p>

## Possible Feature Engineering 
- ***Age of the Property and Location interaction*** using current columns of ***year built, latitude & longitude*** was a prospective move but did not result in hypertuned and baseline model performance.

## Challenges 
- Insufficient and poor diverse data may have led to learning limitations : the model may have fail to learn more general and important patterns. 
- The model is too complex compared to the amount of data, resulting in overfitting to the training data and performing poorly on new data.
- The baseline models did not explained variance in the target variable, Sale Price, very well (70%-80%).
- Hyperparameter tuning long time with a lower yet significant increase.
- The manual Grid Search took a longer amount of time in avoidance to data leakage for **Median prices** based on city and state had to redone during each cross validation. 
- Manual Grid Search limited the scope of Randomized Search in this dataset as during feature engineering test data impacted the training data while fitting.

## Future Goals
- Implementing some research about other features and data that might be useful to predict housing prices.
- Enhancing the model by incorporating additional data sources or more advanced algorithms.
- Exploratory data analysis from this project has potential to provide visuals to prospective Builders and Home buyers to gather data about current and past house sale prices scenario.

