###### 1.
def def_Make_Prediction_Housing_Price(data):
    """Load data, process it with the pipeline, and make predictions of Housing Price.

    Args:
        data (pd.DataFrame): The input data for predictions.

    Returns:
        pd.Series: The predictions of Housing Price
    """
    import joblib
    prediction_pipeline = joblib.load('../models/final_prediction_pipeline.pkl')    # Load the saved prediction pipeline
    predictions = prediction_pipeline.predict(data)   # Make predictions
    
    return predictions





###### 2.
# Custom transformer to replace 'state' and 'city' with the median of 'sale_price'
from sklearn.base import BaseEstimator, TransformerMixin
class MedianTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Custom transformer to encode categorical features 'state' and 'city' with the median of 'sale_price' by category.
    
    This transformer performs the following steps:
    1. Computes the median 'sale_price' for each unique value in the specified columns.
    2. Replaces the values in the specified columns with the corresponding median 'sale_price'.
    3. Handles missing values by replacing them with the global median 'sale_price' of the column.

    Parameters:
    columns (list of str): List of column names to encode.

    Attributes:
    medians (dict): Dictionary storing the median 'sale_price' for each unique value in the specified columns.
    global_medians (dict): Dictionary storing the global median 'sale_price' for each column in case of missing values.
    """
    def __init__(self, columns):
        self.columns = columns
        self.medians = {}
        self.global_medians = {}

    def fit(self, X, y):
        for col in self.columns:
            self.medians[col] = y.groupby(X[col]).median()
            self.global_medians[col] = self.medians[col].median()
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = X[col].map(self.medians[col])
            # Check and handle NaN values
            X[col].fillna(self.global_medians[col], inplace=True)
        return X



###### 3.
from sklearn.model_selection import KFold
from itertools import product
from sklearn.metrics import mean_absolute_error

def def_Custom_Hyperparameter_Search(training_data, param_grid, model_class, n_splits=5):
    """
    Searches for the best hyperparameter values by training the model on multiple 
    training/validation folds and averaging the MAE scores.

    Args:
        training_data (pd.DataFrame): DataFrame containing the features and target data.
        param_grid (dict): Dictionary of possible hyperparameter values for the chosen model.
        model_class (class): The model class to be instantiated (e.g., RandomForestRegressor, XGBRegressor).
        n_splits (int): Number of splits for cross-validation. Default is 5.

    Returns:
        dict: A dict of the best hyperparameter settings based on the chosen metric.
    """
    ###############
    def custom_cross_validation(training_data, n_splits=5):
        """
        Divides the training data into multiple train/validation splits and computes city means.
        This function performs K-fold cross-validation on the provided training data. For each fold,
        it splits the data into training and validation sets, computes the mean sale price for each
        city in the training set, and applies these mean values to the validation set.
    
        Args:
            training_data (pd.DataFrame): DataFrame containing the features and target data.
            n_splits (int): Number of splits for cross-validation. Default is 5.
    
        Returns:
            tuple: A tuple containing two lists:
                - training_folds (list): List of training fold DataFrames.
                - validation_folds (list): List of validation fold DataFrames.
        """
        
        kf = KFold(n_splits=n_splits, shuffle=True)
    
        training_folds = []
        validation_folds = []
    
        for train_index, val_index in kf.split(training_data):
            # Split data into training and validation sets
            X_train, X_val = training_data.iloc[train_index], training_data.iloc[val_index]
    
            # Compute city means on training folds
            city_means = X_train.groupby('numerical__city')['sale_price'].mean()
    
            # Map city means to validation folds
            X_val['numerical__city'] = X_val['numerical__city'].map(city_means)
    
            # Append training and validation folds to the respective lists
            training_folds.append((X_train.drop(columns=['sale_price']), X_train['sale_price']))
            validation_folds.append((X_val.drop(columns=['sale_price']), X_val['sale_price']))
    
        return training_folds, validation_folds
    ###############
    
    # Apply above function
    training_folds, validation_folds = custom_cross_validation(training_data)
    
    param_combinations = list(product(*param_grid.values()))
    best_score = float('inf')
    best_params = None

    for params in param_combinations:
        param_dict = dict(zip(param_grid.keys(), params))
        fold_scores = []

        for (X_train, y_train), (X_val, y_val) in zip(training_folds, validation_folds):
            model = model_class(**param_dict)
            model.fit(X_train.drop(columns=['numerical__city', 'numerical__state']), y_train)
            preds = model.predict(X_val.drop(columns=['numerical__city', 'numerical__state']))
            score = mean_absolute_error(y_val, preds)
            fold_scores.append(score)
        
        mean_score = sum(fold_scores) / len(fold_scores)
        
        if mean_score < best_score:
            best_score = mean_score
            best_params = param_dict

    return best_params








    

###############
# from sklearn.model_selection import KFold
# from itertools import product
# from xgboost import XGBRegressor
# from sklearn.metrics import mean_absolute_error
# def def_Custom_Hyperparameter_Search_XGBRegressor(training_data, param_grid, n_splits=5):
#     """
#     Searches for the best hyperparameter values by training the model on multiple 
#     training/validation folds and averaging the scores.

#     Args:
#         training_data (pd.DataFrame): DataFrame containing the features and target data.
#         param_grid (dict): Dictionary of possible hyperparameter values for the chosen model.
#         n_splits (int): Number of splits for cross-validation. Default is 5.

#     Returns:
#         dict: A dict of the best hyperparameter settings based on the chosen metric.
#     """
#     ###############
#     def def_Custom_Cross_Validation(training_data, n_splits=5):
#         """
#         Divides the training data into multiple train/validation splits and computes city means.
#         This function performs K-fold cross-validation on the provided training data. For each fold,
#         it splits the data into training and validation sets, computes the mean sale price for each
#         city in the training set, and applies these mean values to the validation set.
    
#         Args:
#             training_data (pd.DataFrame): DataFrame containing the features and target data.
#             n_splits (int): Number of splits for cross-validation. Default is 5.
    
#         Returns:
#             tuple: A tuple containing two lists:
#                 - training_folds (list): List of training fold DataFrames.
#                 - validation_folds (list): List of validation fold DataFrames.
#         """
        
#         kf = KFold(n_splits=n_splits)
    
#         training_folds = []
#         validation_folds = []
    
#         for train_index, val_index in kf.split(training_data):
#             # Split data into training and validation sets
#             X_train, X_val = training_data.iloc[train_index], training_data.iloc[val_index]
    
#             # Compute city means on training folds
#             city_means = X_train.groupby('numerical__city')['sale_price'].mean()
    
#             # Map city means to validation folds
#             X_val['numerical__city'] = X_val['numerical__city'].map(city_means)
    
#             # Append training and validation folds to the respective lists
#             training_folds.append((X_train.drop(columns=['sale_price']), X_train['sale_price']))
#             validation_folds.append((X_val.drop(columns=['sale_price']), X_val['sale_price']))
    
#         return training_folds, validation_folds
#     ###############
#     # Apply above function
#     training_folds, validation_folds = def_Custom_Cross_Validation(training_data)
    
#     param_combinations = list(product(*param_grid.values()))
#     best_score = float('inf')
#     best_params = None

#     for params in param_combinations:
#         param_dict = dict(zip(param_grid.keys(), params))
#         fold_scores = []

#         for (X_train, y_train), (X_val, y_val) in zip(training_folds, validation_folds):
#             model = XGBRegressor(**param_dict)
#             model.fit(X_train.drop(columns=['numerical__city', 'numerical__state']), y_train)
#             preds = model.predict(X_val.drop(columns=['numerical__city', 'numerical__state']))
#             score = mean_absolute_error(y_val, preds)
#             fold_scores.append(score)
        
#         mean_score = sum(fold_scores) / len(fold_scores)
        
#         if mean_score < best_score:
#             best_score = mean_score
#             best_params = param_dict

#     return best_params



# #############################
# def def_Custom_Cross_Validation(training_data, n_splits=5):
#     """Creates n_splits sets of training and validation folds
#     Divides the training data into multiple train/validation splits and computes city means.
    
#     Args:
#       training_data: DataFrame containing features and target data.
#       n_splits: Number of splits for cross-validation.

#     Returns:
#       A tuple of lists, where the first index is a list of the training folds, 
#       and the second index is a list of the corresponding validation folds.
#     """
#     from sklearn.model_selection import KFold

#     kf = KFold(n_splits=n_splits)

#     training_folds = []
#     validation_folds = []

#     for train_index, val_index in kf.split(training_data):
#         # Split data into training and validation sets
#         X_train, X_val = training_data.iloc[train_index], training_data.iloc[val_index]

#         # Compute city means on training folds
#         city_means = X_train.groupby('numerical__city')['sale_price'].mean()

#         # Map city means to validation folds
#         X_val['numerical__city'] = X_val['numerical__city'].map(city_means)

#         # Append training and validation folds to the respective lists
#         training_folds.append((X_train.drop(columns=['sale_price']), X_train['sale_price']))
#         validation_folds.append((X_val.drop(columns=['sale_price']), X_val['sale_price']))

#     return training_folds, validation_folds
# #############################



# from itertools import product
# from xgboost import XGBRegressor
# from sklearn.metrics import mean_squared_error

# def def_Hyperparameter_Search(training_folds, validation_folds, param_grid):
#     """Outputs the best combination of hyperparameter settings in the param grid, 
#     given the training and validation folds

#     Args:
#       training_folds: List of training fold dataframes
#       validation_folds: List of validation fold dataframes
#       param_grid: Dictionary of possible hyperparameter values for the chosen model

#     Returns:
#       A list of the best hyperparameter settings based on the chosen metric
#     """
#     param_combinations = list(product(*param_grid.values()))
#     best_score = float('inf')
#     best_params = None

#     for params in param_combinations:
#         param_dict = dict(zip(param_grid.keys(), params))
#         fold_scores = []

#         for (X_train, y_train), (X_val, y_val) in zip(training_folds, validation_folds):
#             model = XGBRegressor(**param_dict)
#             model.fit(X_train.drop(columns=['numerical__city', 'numerical__state']), y_train)
#             preds = model.predict(X_val.drop(columns=['numerical__city', 'numerical__state']))
#             score = mean_squared_error(y_val, preds, squared=False)
#             fold_scores.append(score)
        
#         mean_score = sum(fold_scores) / len(fold_scores)
        
#         if mean_score < best_score:
#             best_score = mean_score
#             best_params = param_dict

#     return best_params