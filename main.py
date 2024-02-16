import warnings
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import sklearn
import joblib
import cloudpickle
import os
from mlflow.models import make_metric
import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor
from mlflow.models import MetricThreshold

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

#get arguments from command
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, required=False, default=0.7)
parser.add_argument("--l1_ratio", type=float, required=False, default=0.2)
args = parser.parse_args()

#evaluation function
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # Read the wine-quality csv file from local
    data = pd.read_csv("data/red-wine-quality.csv")
    data.to_csv("data/red-wine-quality.csv", index=False)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)
    
    data_dir = 'data/red-wine-quality'
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    data.to_csv(data_dir + '/data.csv')
    train.to_csv(data_dir + '/train.csv')
    test.to_csv(data_dir + '/test.csv')

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = args.alpha
    l1_ratio = args.l1_ratio
    
    # Set tracking URI
    mlflow.set_tracking_uri(uri="") # <----- Set where the runs should be stored locally or remote
    #mlflow.get_tracking_uri()
    
    # Create experiment
    #exp_id = mlflow.create_experiment(
    #    name="exp_created_artifact",
    #    tags={"version": "v1", "priority": "p1"},
    #    artifact_location=Path.cwd().joinpath("myartifacts").as_uri())
    
    exp = mlflow.set_experiment(experiment_name="experiment_model_evaluation")
    #get_exp = mlflow.get_experiment(exp_id)
    
    # Log experiment metadata
    print(f"Name: {exp.name}")
    print(f"Experiment_id: {exp.experiment_id}")
    print(f"Artifact location: {exp.artifact_location}")
    print(f"Tags: {exp.tags}")
    print(f"Lifecycle_stage: {exp.lifecycle_stage}")
    print(f"Creation timestamp: {exp.creation_time}")
    
    # Model training
    mlflow.start_run(run_name="run1.1")
    
    # Set tags
    tags = {
        "engineering": "ML platform",
        "release.candidate": "RC1",
        "release.version": "2.0"
    }
    mlflow.set_tags(tags)
    mlflow.autolog(
        log_input_examples=False,
        log_model_signatures=False,
        log_models=False
    )
    
    current_run = mlflow.active_run()
    print(f"Active run id: {current_run.info.run_id}")
    print(f"Active run name: {current_run.info.run_name}")
        
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)
    predicted_qualities = lr.predict(test_x)
    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)
    print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)
    
    # Log the data
    mlflow.log_artifact("data/red-wine-quality.csv")
    
    # Get information from the active run
    active_run = mlflow.active_run()
    print(f"Active run id is {active_run.info.run_id}")
    print(f"Active run name is {active_run.info.run_name}")

    # Set tags
    mlflow.set_tag("release.version", 0.2)
    mlflow.set_tags({"environment": "dev", "priority": "p1"})
    
    
    baseline_model = DummyRegressor()
    baseline_model.fit(train_x, train_y)
    baseline_predicted_qualities = baseline_model.predict(test_x)
    bl_rmse, bl_mae, bl_r2 = eval_metrics(test_y, baseline_predicted_qualities)
    
    print("Baseline Dummy model:")
    print(f" Baseline RMSE: {bl_rmse}")
    print(f" Baseline MAE: {bl_mae}")
    print(f" Baseline R2: {bl_r2}")
    
    sklearn_model_path = "sklearn_model.pkl"
    joblib.dump(lr, sklearn_model_path)
    artifacts = {
        "sklearn_model": sklearn_model_path,
        "data": data_dir
    }
    
    baseline_sklearn_model_path = "baseline_sklearn_model.pkl"
    joblib.dump(lr, baseline_sklearn_model_path)
    baseline_artifacts = {"baseline_sklearn_model": baseline_sklearn_model_path}
    
    class SklearnWrapper(mlflow.pyfunc.PythonModel):
        
        def __init__(self, artifacts_name):
            self.artifacts_name = artifacts_name
        
        def load_context(self, context):
            self.sklearn_model = joblib.load(context.artifacts[self.artifacts_name])
            
        def predict(self, context, model_input):
            return self.sklearn_model.predict(model_input.values)
    
    # Create a Conda environment for the new MLflow Model that contains all necessary dependencies.
    conda_env = {
        "channels": ["defaults"],
        "dependencies": [
            "python={}".format(3.12),
            "pip",
            {
                "pip": [
                    "mlflow=={}".format(mlflow.__version__),
                    "scikit-learn=={}".format(sklearn.__version__),
                    "cloudpickle=={}".format(cloudpickle.__version__),
                ],
            },
        ],
        "name": "sklearn_env",
    }
    
    mlflow.pyfunc.log_model(
        artifact_path="sklearn_mlflow_pyfunc",
        python_model=SklearnWrapper("sklearn_model"),
        artifacts=artifacts,
        code_path=["main.py"],
        conda_env=conda_env
    )
    
    mlflow.pyfunc.log_model(
        artifact_path="baseline_sklearn_mlflow_pyfunc",
        python_model=SklearnWrapper("baseline_sklearn_model"),
        artifacts=baseline_artifacts,
        code_path=["main.py"],
        conda_env=conda_env
    )
        
    # Custom metrics
    def squared_diff_plus_one(eval_df, _builtin_metrics):
        return np.sum(np.abs(eval_df["prediction"] - eval_df["target"] + 1)) ** 2
    
    def sum_on_target_divided_by_two(_eval_df, builtin_metrics):
        return builtin_metrics["sum_on_target"] / 2
    
    squared_diff_plus_one_metric = make_metric(
        eval_fn=squared_diff_plus_one,
        greater_is_better=False,
        name="squared diff plus one",
    )
    
    sum_on_target_divided_by_two_metric = make_metric(
        eval_fn=sum_on_target_divided_by_two,
        greater_is_better=True,
        name="sum on target divided by two",
    )
    
    def prediction_target_scatter(eval_df, _builtin_metrics, artifacts_dir):
        plt.scatter(eval_df["prediction"], eval_df["target"])
        plt.xlabel("Targets")
        plt.ylabel("Predictions")
        plt.title("Targets vs Predictions")
        plot_path = os.path.join(artifacts_dir, "example_scatter_plot.png")
        plt.savefig(plot_path)
        return {"example_scatter_plot_artifact": plot_path}
    
    artifacts_uri = mlflow.get_artifact_uri("sklearn_mlflow_pyfunc")
    
    thresholds = {
        "mean_squared_error": MetricThreshold(
            threshold=0.6, # Maximum MSE threshold
            min_absolute_change=0.1, # Minimum absolute improvement compared to baseline model
            min_relative_change=0.05, # Minimum relative improvement compared to baseline model
            greater_is_better=False
        )
    }
    baseline_model_uri = mlflow.get_artifact_uri("baseline_sklearn_mlflow_pyfunc")
    mlflow.evaluate(
        artifacts_uri,
        test,
        targets="quality",
        model_type="regressor",
        evaluators=["default"],
        custom_metrics=[squared_diff_plus_one_metric, sum_on_target_divided_by_two_metric],
        custom_artifacts=[prediction_target_scatter],
        validation_thresholds=thresholds,
        baseline_model=baseline_model_uri
    )

    # Get artifact uri
    artifacts_uri = mlflow.get_artifact_uri()
    print(f"The artifact path is: {artifacts_uri}")
    
    # End current mlflow run
    mlflow.end_run()
    
    # Get information from the last active run
    last_run = mlflow.last_active_run()
    print(f"Last active run id is {last_run.info.run_id}")
    print(f"Last active run name is {last_run.info.run_name}")