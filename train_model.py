from src.regression_utils import (
    load_data,
    preprocess_data,
    train_and_save_model
)

df = load_data()

X, y, preprocessor = preprocess_data(df)

train_and_save_model(
    X,
    y,
    preprocessor
)

print("AdaBoost Regressor model saved successfully!")