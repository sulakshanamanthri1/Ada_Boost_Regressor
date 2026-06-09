import seaborn as sns
import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def load_data():
    return sns.load_dataset("diamonds")


def preprocess_data(df):

    X = df.drop("price", axis=1)
    y = df["price"]

    categorical_cols = X.select_dtypes(
        include=["object", "category"]
    ).columns

    numerical_cols = X.select_dtypes(
        exclude=["object", "category"]
    ).columns

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_cols
            )
        ],
        remainder="passthrough"
    )

    return X, y, preprocessor


def train_and_save_model(X, y, preprocessor):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "regressor",
            AdaBoostRegressor(
                estimator=DecisionTreeRegressor(max_depth=4),
                n_estimators=100,
                random_state=42
            )
        )
    ])

    model.fit(X_train, y_train)

    joblib.dump(
        model,
        "models/adaboost_regressor.pkl"
    )


def load_model():

    return joblib.load(
        "models/adaboost_regressor.pkl"
    )


def evaluate_model(model, X, y):

    predictions = model.predict(X)

    mae = mean_absolute_error(y, predictions)

    mse = mean_squared_error(y, predictions)

    rmse = mse ** 0.5

    r2 = r2_score(y, predictions)

    return mae, mse, rmse, r2, predictions


def plot_predictions(y_actual, y_pred):

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(
        y_actual,
        y_pred
    )

    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    ax.set_title("Actual vs Predicted")

    return fig