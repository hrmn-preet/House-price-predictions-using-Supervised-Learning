import pandas as pd
import numpy as np



########## 1.
def def_Extract_Info(result):
    """
    Extracts information from a data item.
    """
    return {
        'sale_price': result['description']['sold_price'],
        'list_price': result['list_price'],
        'price_reduced_amount' :result['price_reduced_amount'],
        'status': result['status'],
        'address': result['location']['address']['line'],
        'city': result['location']['address']['city'],
        'postal_code': result['location']['address']['postal_code'],
        'state': result['location']['address']['state'],
        'state_code': result['location']['address']['state_code'],
        'latitude': result['location']['address']['coordinate']['lat'] if result['location']['address']['coordinate'] else None,
        'longitude': result['location']['address']['coordinate']['lon'] if result['location']['address']['coordinate'] else None,
        'year_built' : result['description']['year_built'],
        'sub_type' : result['description']['sub_type'],
        'type' : result['description']['type'],
        'lot_sqft' : result['description']['lot_sqft'],
        'living_sqft' : result['description']['sqft'],
        'number_of_stories' : result['description']['stories'],
        'number_of_baths' : result['description']['baths'],
        'number_of_beds' : result['description']['beds'],
        'number_of_garages' : result['description']['garage'],
        'tags' : result['tags']
    }                      



########## 2.
def def_Encode_Tags(df, threshold=10):
    """
    Encodes the 'tags' column into dummy variables and removes tags with low frequency.

    Parameters:
    df (pd.DataFrame): The input dataframe containing a 'tags' column.
    threshold (float): The minimum percentage threshold for keeping a tag column. Default is 10.

    Returns:
    pd.DataFrame: The dataframe with encoded tags and original 'tags' column removed.
    """
    # Create new features from "tags" feature
    df_tags_dummies = df['tags'].str.join('|').str.get_dummies()
    
    # Calculate percentages of each tag
    column_sums = df_tags_dummies.sum()
    percentages = round((column_sums / len(df_tags_dummies)) * 100, 2)
    
    # Get columns' names that have percentage_totals greater than threshold
    columns_above_threshold = percentages[percentages > threshold].index
    
    # Keep only the columns with percentage_totals greater than threshold
    df_tags_dummies = df_tags_dummies[columns_above_threshold]
    
    # Concatenate the encoded tags DataFrame with the original DataFrame
    df = pd.concat([df, df_tags_dummies], axis=1)
    
    # Delete "tags" column because it becomes redundant
    df = df.drop('tags', axis=1)
    
    return df



########## 3. 
def def_Impute_Number_Of_Garages(df):
    """
    This function fills in the missing values for the 'number_of_garages' column in the DataFrame
    based on the following rules:
    - If its property type is apartment/mobile --> set 'number_of_garages' to 0.
    - If its property type is not apartment/mobile  AND _garage_2_or_more_ is 1 (Yes) --> set 'number_of_garages' to 2.
    - If its property type is not apartment/mobile  AND _garage_1_or_more_ is 1 (Yes) --> set 'number_of_garages' to 1.
    - Otherwise, its property type is **not** apartment/mobile  AND BOTH _garage_1_or_more_ and _garage_2_or_more_ are 0 (No)
        --> set 'number_of_garages' to 0.
    """
    def determine_garages(row):
        """
        Determine the number of garages based on the type of property and garage indicators.
        Parameters: row (pd.Series): A row of the DataFrame.
        Returns:    int: The number of garages.
        """
        # Check if the 'number_of_garages' value is missing (null)
        if pd.isnull(row['number_of_garages']):
            if row['type'] in ['apartment', 'mobile']:
                return 0
            else:
                if row['garage_2_or_more'] == 1:
                    return 2
                elif row['garage_1_or_more'] == 1:
                    return 1
                else:
                    return 0
        # If 'number_of_garages' is not missing, return the existing value
        return row['number_of_garages']
    
    # Apply the 'determine_garages' function to each row in the DataFrame
    df['number_of_garages'] = df.apply(determine_garages, axis=1).astype('int').astype('object')
    return df



########## 4.
def def_Impute_Number_Of_Stories(df):
    """
    This function fills in the missing values for the 'number_of_stories' column in the DataFrame
    based on the following rules:
    - If the property type is 'mobile' --> set 'number_of_stories' to 0.
    - If the property type is not 'mobile'  AND 'two_or_more_stories' is 1 (yes) --> set 'number_of_stories' to 2.
    - If the property type is not 'mobile'  AND 'single_story' is 1 (yes)         --> set 'number_of_stories' to 1.
    - Otherwise, the property type is not 'mobile'  AND both 'two_or_more_stories' and 'single_story is 0 (no) --> set 'number_of_stories' to 1.
    """
    def determine_stories(row):
        """
        Determine the number of stories based on the type of property and story indicators.
        Parameters: row (pd.Series): A row of the DataFrame.
        Returns:    int: The number of stories.
        """
        # Check if the 'number_of_stories' value is missing (null)
        if pd.isnull(row['number_of_stories']):
            if row['type'] == 'mobile':
                return 0
            else:
                if row['two_or_more_stories'] == 1:
                    return 2
                elif row['single_story'] == 1:
                    return 1
                else:
                    return 1
        # If 'number_of_stories' is not missing, return the existing value
        return row['number_of_stories']
    
    # Apply the 'determine_garages' function to each row in the DataFrame
    df['number_of_stories'] = df.apply(determine_stories, axis=1).astype('int').astype('object')
    return df



########## 5.
def def_Impute_Year_Built(df):
    """
    This function imputes missing values in the 'year_built' column based on property type.
    Fill missing values using the median year built of each property type.

    Parameters:
    df (DataFrame): DataFrame containing the 'type' and 'year_built' columns.

    Returns:
    DataFrame: DataFrame with missing 'year_built' values imputed.
    """

    property_types = df['type'].unique()
    
    for property_type in property_types:
        # Calculate the median year built for the current property type
        median_year_built = df.loc[df['type'] == property_type, 'year_built'].median()
        
        # Fill missing values with the calculated median
        df.loc[df['type'] == property_type, 'year_built'] = df.loc[df['type'] == property_type, 'year_built'].fillna(median_year_built)
        
        # if np.isnan(median_year_built): 
        #     # Fill missing values with 0 when median_year_built is null (because all NaN is NaN)
        #     df.loc[df['type'] == property_type, 'year_built'] = df.loc[df['type'] == property_type, 'year_built'].fillna(0)
        # else:
        #     # Fill missing values with the calculated median
        #     df.loc[df['type'] == property_type, 'year_built'] = df.loc[df['type'] == property_type, 'year_built'].fillna(median_year_built)
        
    # Change data type
    df['year_built'] = df['year_built'].astype('int')
    return df



########## 6.
def def_Regression_Impute(df, output, features):     
    """
    Imputes missing values for a specified feature that is continuous variable 
    using RandomForestRegressor based on other features.

    Parameters:
    df (DataFrame): DataFrame containing the features.
    output (str): The name of the feature to impute. (missing value)
    features (list): The names of features for prediction of missing values.

    Returns:
    Series: The imputed feature column.
    """
    from sklearn.ensemble import RandomForestRegressor
    
    # Encoding property types
    df['type_encoded'] = df['type'].astype('category').cat.codes

    # Select the features for prediction
    features.remove('type')
    features.append('type_encoded')
           
    # Create DataFrame with no missing values for training set
    df_train = df.dropna(subset=[output])
    
    # inputs and output variable for training
    X_train = df_train[features]
    y_train = df_train[output]
    
    # Model training
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    
    # Predict missing values
    df_missing = df[df[output].isnull()]
    X_missing = df_missing[features]
    df.loc[df[output].isnull(), output] = model.predict(X_missing)
    
    # Drop the temporary encoded column
    df.drop(columns=['type_encoded'], inplace=True)

    # Round up and change data type
    df[output] = df[output].round(0).astype('int')
    
    return df[output].round(0)



########## 7.
def def_Classification_Impute(df, output, features):
    """
    Imputes missing values for a specified feature that is categorical variable 
    using RandomForestClassifier based on other features.

    Parameters:
    df (DataFrame): DataFrame containing the features.
    output (str): The name of the feature to impute (with missing values).
    features (list): The names of features to be used for predicting missing values.

    Returns:
    Series: The imputed feature column.
    """
    from sklearn.ensemble import RandomForestClassifier
    
    # Encoding property types and target variable
    df['type_encoded'] = df['type'].astype('category').cat.codes
    df[output] = df[output].astype('category')
    
    # Select the features for prediction
    features.remove('type')
    features.append('type_encoded')
           
    # Split the dataset into training and testing datasets
    train_df = df[df[output].notna()]
    test_df = df[df[output].isna()]

    # Train the model
    X_train = train_df[features]
    y_train = train_df[output]
    X_test = test_df[features]
      
    # Model training
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Predict the missing values
    y_pred = model.predict(X_test)

    # Impute the missing values in the original dataframe
    df.loc[df[output].isna(), output] = y_pred
        
    # Drop the temporary encoded column
    df.drop(columns=['type_encoded'], inplace=True)

    # Change data type back
    df[output] = df[output].astype('int').astype('object')
    
    return df[output]



########## 8.
def def_Remove_Outliers(df, column_name, lower_percentile=0.25, upper_percentile=0.75, threshold=2):
    Q1 = df[column_name].quantile(lower_percentile)
    Q3 = df[column_name].quantile(upper_percentile)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    return df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]

