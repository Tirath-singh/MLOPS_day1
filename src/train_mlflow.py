import pandas as pd
import mlflow
from mlflow import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, root_mean_squared_error

# 1. Setup Tracking
mlflow.set_tracking_uri("sqlite:///mlflow.db")

experiment_name = "Advertising_Sales_Regression"
registered_model_name = "Sales_Prediction_Model"

mlflow.set_experiment(experiment_name)

# 2. Data Preparation
df = pd.read_csv(
    r"D:\\MLOPS_day1\\data\\data.csv"
)

X, y = df[["TV", "radio", "newspaper"]], df["sales"]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=56
)

# 3. Train Candidate Models (LOG ONLY, DO NOT REGISTER YET)
models = {
    "Linear_Regression": LinearRegression(),
    "Ridge_Regression": Ridge(alpha=1.0),
    "Random_Forest": RandomForestRegressor(max_depth=5, random_state=56)
}

batch_runs = []

for name, model in models.items():
    with mlflow.start_run(run_name=name) as run:
        model.fit(Xtrain, ytrain)
        rmse = root_mean_squared_error(ytest, model.predict(Xtest))

        mlflow.log_param("model_type", name)
        mlflow.log_metric("test_rmse", rmse)

        mlflow.sklearn.log_model(model, artifact_path="model")
        batch_runs.append((run.info.run_id, rmse))

# 4. Find the Single Best Model from this Batch
batch_runs.sort(key=lambda x: x[1])  # Sort by lowest RMSE
best_run_id, best_rmse = batch_runs[0]

# 5. Register ONLY the Winning Run as the Challenger
client = MlflowClient()
challenger_model = mlflow.register_model(
    source=f"runs:/{best_run_id}/model",
    name=registered_model_name
)
challenger_version = challenger_model.version

# Assign Challenger Alias
client.set_registered_model_alias(registered_model_name, "challenger", challenger_version)
print(f"Best batch run {best_run_id} registered as Challenger (v{challenger_version}, RMSE: {best_rmse:.4f})")

# 6. Challenger vs. Champion Evaluation Gate
try:
    champion_info = client.get_model_version_by_alias(registered_model_name, "champion")
    champion_run = client.get_run(champion_info.run_id)
    champion_rmse = champion_run.data.metrics["test_rmse"]
    champion_version = champion_info.version

    print(f"Current Champion: Version {champion_version} (RMSE: {champion_rmse:.4f})")

    if best_rmse < champion_rmse:
        client.set_registered_model_alias(registered_model_name, "champion", challenger_version)
        print(f"Title Change! Challenger (v{challenger_version}) defeated Champion (v{champion_version})")
    else:
        print(f"Defended! Champion (v{champion_version}) retains its title.")

except Exception:
    client.set_registered_model_alias(registered_model_name, "champion", challenger_version)
    print(f"No existing champion found. Version {challenger_version} crowned as first Champion!")