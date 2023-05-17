from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_validate
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils import resample
import pandas as pd
import numpy as np
from keras.wrappers.scikit_learn import KerasRegressor
from src.utils.bootstraping import bootstrap_resampling
from sklearn.linear_model import Lasso

def MLP(data):
    X = data[['in_0', 'in_1', 'in_2', 'in_3', 'in_4', 'in_5', 'in_6', 'in_7']].values
    y = data['out_0'].values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Build the MLP model
    def create_model():
        model = Sequential()
        model.add(Dense(32, activation='relu', input_shape=(8,)))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    # Create the KerasRegressor
    model = KerasRegressor(build_fn=create_model, epochs=100, batch_size=32, verbose=0)

    #bootstrap_score = bootstrap_resampling(data[['in_0', 'in_1', 'in_2', 'in_3', 'in_4', 'in_5', 'in_6', 'in_7']], data['out_0'], model)

    # Train the model on the full training set
    model.fit(X_train, y_train)

    # Get the learned parameters
    weights = model.model.get_weights()

    # Output the equation of the learned parameters
    equation = "y = "
    for i in range(len(weights) // 2):
        for j, weight in enumerate(weights[2*i].flatten()):
            feature_name = f'in_{j}'
            equation += f"({weight:.4f} * {feature_name}) + "
    equation += f"{weights[-1].flatten()[0]:.4f}"
    
       # Incorporate Lasso regression
    lasso = Lasso(alpha=0.01)
    lasso.fit(X_train, y_train)
    equation_lasso = "y = "
    for i, weight in enumerate(lasso.coef_):
        feature_name = f'in_{i}'
        equation_lasso += f"({weight:.4f} * {feature_name}) + "
    equation_lasso += f"{lasso.intercept_:.4f}"

    # Define the scoring metrics
    scoring = {
        'mse': make_scorer(mean_squared_error, greater_is_better=False),
        'mae': make_scorer(mean_absolute_error, greater_is_better=False),
        'r2': make_scorer(r2_score)
    }

    # Evaluate the model using cross-validation
    results = cross_validate(lasso, X_train, y_train, cv=5, scoring=scoring)

    # Extract the evaluation metrics from the results
    metrics = {
        'mse': -np.mean(results['test_mse']),
        'mae cross_validation': -np.mean(results['test_mae']),
        #'mae_bootstrap': bootstrap_score
    }

    # Evaluate the model on the testing set
    y_pred = model.predict(X_test)
    metrics_test = {
        'mse': mean_squared_error(y_test, y_pred),
        'mae cross_validation': mean_absolute_error(y_test, y_pred),
        #'mae_bootstrap': bootstrap_score
    }

    # Combine the metrics into a DataFrame
    metrics_df = pd.DataFrame({'Train': metrics, 'Test': metrics_test, 'Equation': equation})

    return metrics_df,equation, equation_lasso
