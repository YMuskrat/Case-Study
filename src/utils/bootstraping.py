from sklearn.utils import resample
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split

def bootstrap_resampling(df, y, model, n_iterations=100, test_size=0.2, random_state=42):
    """
    Perform bootstrapping resampling and evaluate model performance.

    Args:
        df (DataFrame): The input DataFrame containing features and target.
        target_column (str): The name of the target column.
        model: The machine learning model to train and test.
        n_iterations (int): The number of bootstrap iterations.
        test_size (float): The proportion of the dataset to use as the test set.
        random_state (int): The random state for reproducibility.

    Returns:
        float: The average performance metric across bootstrap iterations.
    """

    metric_scores = []

    for _ in range(n_iterations):
        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(df,
                                                            y,
                                                            test_size=test_size,
                                                            random_state=random_state)

        # Create a bootstrap sample
        X_resampled, y_resampled = resample(X_train, y_train, random_state=random_state)

        # Fit the model on the bootstrap sample
        model.fit(X_resampled, y_resampled)

        # Evaluate the model on the test set
        y_pred = model.predict(X_test)
        score = mean_absolute_error(y_test, y_pred)
        metric_scores.append(score)

    # Calculate the average performance metric
    average_score = sum(metric_scores) / len(metric_scores)

    return average_score

