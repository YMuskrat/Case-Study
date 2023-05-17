from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
import numpy as np


class VariableRelationship:

    def __init__(self, data):
        self.df=data[['in_0', 'in_1', 'in_2', 'in_3', 'in_4', 'in_5', 'in_6', 'in_7']]

    # Define the mathematical functions
    def linear_func(self,x, a, b):
        return a * x + b

    def quadratic_func(x, a, b, c):
        return a * x**2 + b * x + c

    def exponential_func(self,x, a, b):
        return a * np.exp(b * x)

    # Define the function to fit the data and calculate MSE
    def fit_and_calculate_mse(self,x, y, func):
        try:
            popt, _ = curve_fit(func, x, y)
            y_pred = func(x, *popt)
            mse = mean_squared_error(y, y_pred)
            return mse
        except:
            return np.inf
    def get_best_fit_relationship(self,):
        features = self.df.columns[:-1]  # Exclude the output variable

        best_fits = {}

        # Iterate through all pairs of features
        for i, feature1 in enumerate(features):
            for j, feature2 in enumerate(features):
                if i != j:
                    x = self.df[feature1]
                    y = self.df[feature2]

                    # Fit the data using different functions and calculate MSE
                    mse_linear = self.fit_and_calculate_mse(x, y, self.linear_func)
                    mse_quadratic = self.fit_and_calculate_mse(x, y, self.quadratic_func)
                    mse_exponential = self.fit_and_calculate_mse(x, y, self.exponential_func)

                    # Find the best fit based on the lowest MSE
                    best_fit = min(mse_linear, mse_quadratic, mse_exponential)
                    best_fit_name = {
                        mse_linear: 'Linear',
                        mse_quadratic: 'Quadratic',
                        mse_exponential: 'Exponential'
                    }[best_fit]

                    # Store the best fit and MSE if MSE is less than or equal to 0.5
                    if best_fit <= 0.6:
                        best_fits[(feature1, feature2)] = {
                            'Best Fit': best_fit_name,
                            'MSE': best_fit
                        }

        # Print the best fit and MSE for each pair of features
        for pair, fit_mse in best_fits.items():
            feature1, feature2 = pair
            best_fit = fit_mse['Best Fit']
            mse = fit_mse['MSE']
            print(f"Best fit between {feature1} and {feature2}: {best_fit}, MSE: {mse}") 
