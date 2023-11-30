"""Build placeholder model."""
import os
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

from cse6242_project import PROJECT_ROOT


df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "training_data_2.csv"))

# Get features and labels from source dataset
X = df.iloc[:, 5:]
y = df["score"]

# Build pipeline
pipeline = Pipeline(
    [
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        # ("pca", PCA(n_components=20)),
        ("rf", RandomForestRegressor(max_depth=5, n_estimators=500)),
    ]
)

# Split data for train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10, random_state=1
)

# Fit pipeline against data
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

joblib.dump(pipeline, os.path.join(PROJECT_ROOT, "models", "rf_regressor_pipeline3.pkl"), compress=9)
