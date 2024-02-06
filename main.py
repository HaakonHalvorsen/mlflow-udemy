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

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

#get arguments from command
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, required=False, default=0.7)
parser.add_argument("--l1_ratio", type=float, required=False, default=0.8)
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
    train.to_csv("data/train.csv")
    test.to_csv("data/test.csv")

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
    
    exp = mlflow.set_experiment(experiment_name="experiment_signature")
    #get_exp = mlflow.get_experiment(exp_id)
    
    # Log experiment metadata
    print(f"Name: {exp.name}")
    print(f"Experiment_id: {exp.experiment_id}")
    #print(f"Artifact location: {exp.artifact_location}")
    #print(f"Tags: {exp.tags}")
    #print(f"Lifecycle_stage: {exp.lifecycle_stage}")
    #print(f"Creation timestamp: {exp.creation_time}")
    
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
    
    signature = infer_signature(test_x, predicted_qualities)
    input_example = {
        "columns": np.array(test_x.columns),
        "data": np.array(test_x.values)
    }
    
    # Log the data
    mlflow.log_artifact("data/red-wine-quality.csv")
    
    # Log model
    mlflow.sklearn.save_model(lr, "model", signature=signature, input_example=input_example)
    
    # Get information from the active run
    active_run = mlflow.active_run()
    print(f"Active run id is {active_run.info.run_id}")
    print(f"Active run name is {active_run.info.run_name}")

    # Set tags
    mlflow.set_tag("release.version", 0.2)
    mlflow.set_tags({"environment": "dev", "priority": "p1"})

    # Get artifact uri
    artifacts_uri = mlflow.get_artifact_uri()
    print(f"The artifact path is: {artifacts_uri}")
    
    # End current mlflow run
    mlflow.end_run()
    
    # Get information from the last active run
    last_run = mlflow.last_active_run()
    print(f"Last active run id is {last_run.info.run_id}")
    print(f"Last active run name is {last_run.info.run_name}")