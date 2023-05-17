import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import Lasso,Ridge



def MLR(data, sub=0):
    # Prepare the data
    df = data.copy()  # Create a copy of the original dataframe

    # Check if the columns exist in the dataframe before dropping
    columns_to_drop = ['ID', 'Cluster', 'out_0']
    df = df.drop(columns=[col for col in columns_to_drop if col in df])

    X = df
    y = data['out_0']

    # Create interaction terms and polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)

    # Should generalized to all cases (still needs refactoring but does the job for now)
    if sub==0:
        X_interact = np.hstack((X_poly, X_poly[:, 0:1] * X_poly[:, 1:2], X_poly[:, 0:1] * X_poly[:, 5:6],
                                X_poly[:, 1:2] * X_poly[:, 5:6], X_poly[:, 2:3] * X_poly[:, 0:1],
                                X_poly[:, 2:3] * X_poly[:, 1:2], X_poly[:, 2:3] * X_poly[:, 5:6],
                                X_poly[:, 3:4] * X_poly[:, 0:1], X_poly[:, 3:4] * X_poly[:, 1:2],
                                X_poly[:, 3:4] * X_poly[:, 4:5], X_poly[:, 3:4] * X_poly[:, 5:6],
                                X_poly[:, 4:5] * X_poly[:, 0:1], X_poly[:, 4:5] * X_poly[:, 1:2],
                                X_poly[:, 4:5] * X_poly[:, 5:6], X_poly[:, 5:6] * X_poly[:, 0:1],
                                X_poly[:, 5:6] * X_poly[:, 1:2], X_poly[:, 5:6] * X_poly[:, 6:7],
                                np.exp(X_poly[:, 6:7]), np.exp(X_poly[:, 5:6])))
    elif sub==1:
        X_interact = np.hstack((X, X_poly, X_poly[:, 0:1] * X_poly[:, 3:4], X_poly[:, 0:1] * X_poly[:, 6:7],
                            X_poly[:, 1:2] * X_poly[:, 0:1], X_poly[:, 1:2] * X_poly[:, 3:4],
                            X_poly[:, 2:3] * X_poly[:, 0:1], X_poly[:, 2:3] * X_poly[:, 3:4],
                            X_poly[:, 3:4] * X_poly[:, 0:1], X_poly[:, 3:4] * X_poly[:, 2:3],
                            X_poly[:, 3:4] * X_poly[:, 6:7], X_poly[:, 4:5] * X_poly[:, 0:1],
                            X_poly[:, 4:5] * X_poly[:, 3:4], X_poly[:, 5:6] * X_poly[:, 0:1],
                            X_poly[:, 5:6] * X_poly[:, 3:4], X_poly[:, 6:7] * X_poly[:, 0:1],
                            X_poly[:, 6:7] * X_poly[:, 3:4]))
    elif sub==2:
        X_interact = np.hstack((X, X_poly, X_poly[:, 4:5] * X_poly[:, 1:2]))
    elif sub==3:
        X_interact = np.hstack((X, X_poly, X_poly[:, 0:1] * X_poly[:, 1:2], X_poly[:, 0:1] * X_poly[:, 3:4],
                            X_poly[:, 1:2] * X_poly[:, 3:4], X_poly[:, 2:3] * X_poly[:, 3:4],
                            X_poly[:, 4:5] * X_poly[:, 1:2], X_poly[:, 4:5] * X_poly[:, 3:4],
                            X_poly[:, 5:6] * X_poly[:, 3:4], X_poly[:, 6:7] * X_poly[:, 1:2],
                            X_poly[:, 6:7] * X_poly[:, 3:4]))


    # Fit the Ridge regression model
    alpha = 0.01  # regularization parameter
    model = Ridge(alpha=alpha)
    model.fit(X_interact, y)

    # Perform cross-validation
    cv_scores = cross_val_score(model, X_interact, y, scoring='neg_mean_squared_error', cv=5)
    mse_scores = -cv_scores

    # Get the coefficients and intercept
    coefficients = model.coef_
    intercept = model.intercept_

    # Compute mean squared error
    y_pred = model.predict(X_interact)


    # Compute mean absolute error
    mae = mean_absolute_error(y, y_pred)
    # Create the equation string
    equation = "y = "
    for i, coef in enumerate(coefficients):
        if i == 0:
            equation += f"{coef:.4f}"
        else:
            equation += f" + {coef:.4f} * x{i-1}"

    equation += f" + {intercept:.4f}"

    print('Average MAE over the cross validation:', mae)

    return equation
    #print('CV scores:', cv_scores)
    #print('Coefficients:', coefficients)
    #print('Intercept:', intercept)
