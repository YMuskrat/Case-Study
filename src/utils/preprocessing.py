import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
import statsmodels.api as sm
import numpy as np
import scipy.stats as stats

def get_data_info(df):
    # Get data types and null values as a DataFrame
    data_info = pd.concat([df.dtypes, df.isnull().sum()], axis=1)
    data_info.columns = ['Data Types', 'Null Values']
     
    return data_info

def get_distribution_properties(df):
    mean_value={}
    standard_dev={}
    for col in df.columns:
        mean_value[col]=df[col].mean()
        standard_dev[col]= df[col].std()
    lower_treshold = {k: mean_value[k] - 3 * standard_dev[k] for k in mean_value.keys()}
    upper_treshold = {k: mean_value[k] + 3 * standard_dev[k] for k in mean_value.keys()}

    lower_subsets, upper_subsets = [], []
    for key, value in zip(lower_treshold.keys(), lower_treshold.values()):
        low_subset = df[df[key] < lower_treshold[key]]
        up_subset = df[df[key] > upper_treshold[key]]
        lower_subsets.append(low_subset)
        upper_subsets.append(up_subset)

    lower_outliers = pd.concat(lower_subsets)
    upper_outliers = pd.concat(upper_subsets)


    outliers = pd.concat({'lower_bound': lower_outliers.reset_index(drop=True), 'upper_bound': upper_outliers.reset_index(drop=True)}, axis=1)
    
    return mean_value, standard_dev,outliers

def get_bounding_box_properties(df):
    # get summary statistics of each predictor
    summary = df.describe()

    # calculate the IQR for each predictor
    q1 = summary.loc['25%']
    q3 = summary.loc['75%']
    iqr = q3 - q1

    # calculate the upper and lower bounds for outliers
    upper_bound = q3 + 1.8*iqr
    lower_bound = q1 - 1.8*iqr

    # create a boolean dataframe indicating outliers for each predictor
    outliers = (df < lower_bound) | (df > upper_bound)

    dfs = []
    for predictor in df.columns:
        predictor_outliers = df[outliers[predictor]][predictor]
        dfs.append(predictor_outliers)

    merged_df = pd.concat(dfs, axis=1)
    return upper_bound,lower_bound,merged_df

def get_df_intersection(df1,df2):
    # Create a new dataframe to store the intersection results
    intersection_df = pd.DataFrame(columns=df1.columns)

    # Iterate over the columns in df1
    for column in df1.columns:
        # Find the common values between the two columns
        common_values = df1[column].isin(df2[column])
        
        # Filter df1 based on the common values
        intersection_df[column] = df1.loc[common_values, column]

    # Add the 'ID' column to the intersection dataframe
    intersection_df['ID'] = df1['ID']

    # Print the intersection dataframe
    return intersection_df


def get_normal_distribution(df):
    # Create subplots
    fig, axs = plt.subplots(ncols=len(df.columns[1:]), figsize=(20, 5))

    # Loop over each column in the DataFrame and create a KDE plot
    for i, col in enumerate(df.columns[1:]):
        sns.histplot(df[col], kde=True, ax=axs[i])
        axs[i].set_xlabel('Values')
        axs[i].set_ylabel('Probability Density')
        axs[i].set_title(f'PDF Plot of {col} ' "\n" f'skew: {round(skew(df[col]), 6)}' "\n" f'kurtosis: {round(kurtosis(df[col]), 6)}')

        # Plot mean, mode, and median as vertical lines
        mean_val = np.mean(df[col])
        mode_val = df[col].mode().values[0]
        median_val = np.median(df[col])
        axs[i].axvline(mean_val, color='r', linestyle='--', label='Mean')
        axs[i].axvline(mode_val, color='g', linestyle='--', label='Mode')
        axs[i].axvline(median_val, color='b', linestyle='--', label='Median')
        
        # Calculate standard deviation and plot percentage data within each std
        std_val = np.std(df[col])
        lower1 = mean_val - std_val
        upper1 = mean_val + std_val
        lower2 = mean_val - (2 * std_val)
        upper2 = mean_val + (2 * std_val)
        lower3 = mean_val - (3 * std_val)
        upper3 = mean_val + (3 * std_val)
        
        percentage_within_std1 = (np.sum((df[col] >= lower1) & (df[col] <= upper1)) / len(df[col])) * 100
        percentage_within_std2 = (np.sum((df[col] >= lower2) & (df[col] <= upper2)) / len(df[col])) * 100
        percentage_within_std3 = (np.sum((df[col] >= lower3) & (df[col] <= upper3)) / len(df[col])) * 100
        
        axs[i].text(0.5, 0.9, f'Within 1 std: {percentage_within_std1:.2f}%', transform=axs[i].transAxes, ha='center')
        axs[i].text(0.5, 0.8, f'Within 2 std: {percentage_within_std2:.2f}%', transform=axs[i].transAxes, ha='center')
        axs[i].text(0.5, 0.7, f'Within 3 std: {percentage_within_std3:.2f}%', transform=axs[i].transAxes, ha='center')

        axs[i].legend()

    plt.tight_layout()
    plt.show()




def get_high_laverage_points(df_X,df_Y):
    # Merge input and output dataframes
    merged_df = pd.merge(df_X, df_Y, on='ID')

    # Fit linear regression model
    X = merged_df.iloc[:, :-1]  # all columns except last (output)
    y = merged_df.iloc[:, -1]   # last column (output)
    model = sm.OLS(y, sm.add_constant(X)).fit()

    # Calculate leverage values
    leverage = model.get_influence().hat_matrix_diag

    # Identify high leverage points
    threshold = 2 * len(X.columns) / len(X)
    high_leverage_points = merged_df[leverage > threshold]
    return high_leverage_points

def get_one_way_anova(df_X,df_Y):
        # Assuming you have a DataFrame called df with your predictors (X) and target variable (y)
    X = df_X[['in_0', 'in_1', 'in_2', 'in_3', 'in_4', 'in_5', 'in_6',  'in_7']]  # Select the columns for your predictors
    y = df_Y['out_0']  # Your target variable

    # Perform one-way ANOVA for each predictor
    f_scores, p_values = stats.f_oneway(X['in_0'], X['in_1'], X['in_2'], X['in_3'], X['in_4'], X['in_5'], X['in_6'],X['in_7'])

    # Create a DataFrame to store the results
    results = pd.DataFrame({'Predictor': X.columns, 'F-Score': f_scores, 'p-value': p_values})

    # Print the results
    return results
