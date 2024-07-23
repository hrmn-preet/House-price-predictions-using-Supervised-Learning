import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


########## 1.
def def_Draw_Histograms_Univariate(dataframe, features, nrows, ncols):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 7))
    for i, feature in enumerate(features):
        row_index = i // ncols
        col_index = i % ncols
        ax = axes[row_index, col_index]
        sns.histplot(dataframe[feature], bins=20, kde=True, ax=ax, color='midnightblue')
        ax.set_title(f"Distribution of {feature.replace('_',' ').title()}", color='DarkBlue')
        ax.set_xlabel(feature.replace('_',' ').title())
        # ax.set_yscale('log') 
    plt.suptitle("Histogram and KDE chart for each numerical variable")
    plt.tight_layout()
    plt.show()
              


########## 2.
def def_Draw_Boxplot_Univariate(dataframe, features, nrows, ncols):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16,7))
    for i, feature in enumerate(features):
        row = i // ncols
        col = i % ncols
        ax = axes[row, col]
        sns.boxplot(data=dataframe[feature], orient='h', ax=ax)
    plt.suptitle("Boxplot chart for each numerical variable")
    plt.show()



########## 3. 
def def_Draw_Countplot_Univariate(dataframe, features, nrows, ncols):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 8))
    for i, feature in enumerate(features):
        row = i // ncols
        col = i % ncols
        ax = axes[row, col]
        sns.countplot(data=dataframe, x=feature, ax=ax)
        ax.set_xlabel(feature.replace('_',' ').title())
        ax.set_ylabel("No.of Houses")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.suptitle("Bar chart for each categorical variable")
    plt.tight_layout()
    plt.show()



########## 4.
def def_Draw_Barplot_Bivariate(df, features, nrows, ncols, target='sale_price'):
    """
    Draws barplots of the target variable against each feature in the features list.
    
    Parameters:
    df (DataFrame): The dataframe containing the data.
    target (str): The target variable to plot on the y-axis.
    features (list): A list of features to plot on the x-axis.
    nrows (int): The number of rows in the subplot grid.
    ncols (int): The number of columns in the subplot grid.
    
    Returns:
    None
    """
    fig, ax = plt.subplots(figsize=(13, 8), nrows=nrows, ncols=ncols)
    
    for i, feature in enumerate(features):
        row_index = i // ncols
        col_index = i % ncols
        sns.barplot(y=df[target], x=df[feature], ax=ax[row_index, col_index])
    plt.suptitle("Barplots of the target variable against each categorical predictors")
    plt.tight_layout()
    plt.show()



########## 5.
def def_Draw_Boxplot_Bivariate(df, features, nrows, ncols, target='sale_price'):
    """
    Draws barplots of the target variable against each feature in the features list.
    
    Parameters:
    df (DataFrame): The dataframe containing the data.
    target (str): The target variable to plot on the y-axis.
    features (list): A list of features to plot on the x-axis.
    nrows (int): The number of rows in the subplot grid.
    ncols (int): The number of columns in the subplot grid.
    
    Returns:
    None
    """
    fig, ax = plt.subplots(figsize=(13, 8), nrows=nrows, ncols=ncols)
    
    for i, feature in enumerate(features):
        row_index = i // ncols
        col_index = i % ncols
        sns.boxplot(y=df[target], x=df[feature], ax=ax[row_index, col_index])
    plt.suptitle("Barplots of the target variable against each categorical predictors")
    plt.tight_layout()
    plt.show()



########## 6.
def def_Draw_Barplot_Boxplot_Bivariate(dataframe, target, feature, ylim=None):
    """
    Draws a barplot and a boxplot of the target variable against a specified feature
    on a single row with two graphs.
    
    Parameters:
    dataframe (DataFrame): The DataFrame containing the data.
    target (str): The target variable for the plots.
    feature (str): The feature variable for the plots.
    ylim (tuple): The y-axis limits for the boxplot.
    """
    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    
    # Barplot
    sns.barplot(y=dataframe[target], x=dataframe[feature], ax=axs[0])
    axs[0].set_title(f"Barplot of {target.replace('_',' ').title()} against {feature.replace('_',' ').title()}")
    axs[0].set_xlabel(f"{feature.replace('_',' ').title()}")
    axs[0].set_ylabel(f"{target.replace('_',' ').title()}")


    # Boxplot
    data = dataframe[[target, feature]]
    sns.boxplot(x=feature, y=target, data=data, ax=axs[1])
    if ylim:
        axs[1].set_ylim(ylim)
    axs[1].set_title(f"Boxplot of {target.replace('_',' ').title()} against {feature.replace('_',' ').title()}")
    axs[1].set_xlabel(f"{feature.replace('_',' ').title()}")
    axs[1].set_ylabel(f"{target.replace('_',' ').title()}")
    
    # Adjust layout
    plt.suptitle(f"{target.replace('_',' ').title()} against {feature.replace('_',' ').title()}")
    plt.tight_layout()
    plt.show()



########## 7.
def def_Plot_Actual_vs_Predicted(model, y_train, y_test, X_train, X_test):
    """
    Plot the distribution of actual vs predicted values for both train and test sets.

    Parameters:
    - model: Trained model used for predictions.
    - y_train: Actual target values for the training set.
    - y_test: Actual target values for the test set.
    - X_train: Features of the training set.
    - X_test: Features of the test set.
    
    In each plot:
    - The x-axis represents the values of the target variable.
    - The y-axis represents the density of the data points.
    """
    plt.figure(figsize=(16, 5))

    # Plot for training set
    plt.subplot(1, 2, 1)
    ax1 = sns.distplot(y_train, hist=False, color="r", label="Actual Train Values")
    sns.distplot(model.predict(X_train), hist=False, color="b", label="Predicted Train Values", ax=ax1)
    plt.legend()

    # Plot for test set
    plt.subplot(1, 2, 2)
    ax2 = sns.distplot(y_test, hist=False, color="r", label="Actual Test Values")
    sns.distplot(model.predict(X_test), hist=False, color="b", label="Predicted Test Values", ax=ax2)
    plt.legend()

    plt.show()





########## 8.
def plot_Evaluation_and_Residuals(y_pred, y_true):
    """
    Plots the actual vs predicted values and the residuals for model evaluation.

    Args:
        y_pred (array-like): Predicted values from the model.
        y_true (array-like): True values.

    Returns:
        None
    """
    """
    Plots the actual vs predicted values and the residuals for model evaluation.

    Args:
        y_pred (array-like): Predicted values from the model.
        y_true (array-like): True values.

    Returns:
        None
    """
    # Ensure y_pred and y_true are numpy arrays and flattened
    y_pred = np.array(y_pred).flatten()
    y_true = np.array(y_true).flatten()

    # Check if the lengths are the same
    if len(y_pred) != len(y_true):
        raise ValueError("y_pred and y_true must have the same length")

    # Calculate residuals
    residuals = y_true - y_pred

    # Set up the figure with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # First plot - Actual vs. Predicted values
    axes[0].scatter(y_pred, y_true)
    axes[0].set_xlabel('Value from Model Predictions')
    axes[0].set_ylabel('True Value')
    axes[0].plot([0, 700000], [0, 700000], '-', color='r')
    axes[0].set_title('Plot Actual vs. Predicted values on testing data')

    # Second plot - Residual Plot
    axes[1].scatter(y_pred, residuals)
    axes[1].set_xlabel('Predicted House Price')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Residual Plot')
    axes[1].axhline(y=0, color='r', linestyle='--')

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()