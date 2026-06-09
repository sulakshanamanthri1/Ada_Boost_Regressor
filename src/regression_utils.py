import seaborn as sns
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor


def load_data():
    return sns.load_dataset("diamonds")


def preprocess_data(df):

    X = df.drop("price", axis=1)

    y = df["price"]

    categorical_cols = X.select_dtypes(
        include=["object", "category"]
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


def train_model():

    df = load_data()

    X, y, preprocessor = preprocess_data(df)

    model = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "regressor",
            AdaBoostRegressor(
                estimator=DecisionTreeRegressor(
                    max_depth=4
                ),
                n_estimators=100,
                random_state=42
            )
        )
    ])

    model.fit(X, y)

    return model