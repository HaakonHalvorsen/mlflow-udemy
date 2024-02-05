import os
import mlflow
import warnings
import mlflow.sklearn
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from PIL import Image
from mlflow import MlflowException
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    
    # Set tracking uri to default mlruns folder locally
    mlflow.set_tracking_uri(uri="")
    
    # Set a description that will show in mlflow ui
    experiment_description = (
        "This experiment is running SVM models against "
        "the iris dataset to predict variety class."
    )
    
    # Set tags for the experiment
    experiment_tags = {
        "environment": "dev",
        "dataset": "iris",
        "version": "v1",
        "mlflow.note.content": experiment_description
    }
    
    # Experiment setup
    experiment_name = "iris_experiment"
    try:
        # Set a description that will show in mlflow ui
        experiment_description = (
            "This experiment is running SVM models against "
            "the iris dataset to predict variety class."
        )
    
        # Set tags for the experiment
        experiment_tags = {
            "environment": "dev",
            "dataset": "iris",
            "version": "v1",
            "mlflow.note.content": experiment_description
        }
        experiment_id = mlflow.create_experiment(name=experiment_name,
                                                 tags=experiment_tags)
        experiment = mlflow.get_experiment(experiment_id=experiment_id)
        
    except MlflowException:
        print(f"Experiment already exists: {experiment_name}")
        experiment = mlflow.set_experiment(experiment_name=experiment_name)
    
    # Log information about experiment
    print(f"Name: {experiment.name}")
    print(f"Experiment_id: {experiment.experiment_id}")
    print(f"Artifact Location: {experiment.artifact_location}")
    print(f"Tags: {experiment.tags}")
    print(f"Lifecycle_stage: {experiment.lifecycle_stage}")
    print(f"Creation timestamp: {experiment.creation_time}")
    
    # Read Iris dataset
    df = pd.read_csv("data/iris.csv")
    
    # Split dataset into features and labels
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1:]
    
    # Split dataset into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    # Save train and test dataset to csv
    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)
    df_train.to_csv("data/iris_train.csv")
    df_test.to_csv("data/iris_test.csv")
    print(f"df_train shape: {df_train.shape}")
    print(f"df_test shape: {df_test.shape}")
    
    # Check if there is an active run going on
    if mlflow.active_run() is not None:
        mlflow.end_run()
    
    run_description = "This is a run to train a SVM model on the iris dataset."
    run_tags = {
        "environment": "dev",
        "priority": 3,
        "version": "v2"
    }
    
    # Logging basics
    #   all parameters of the model used
    #   training metrics: accuracy, f1 score, log loss, precision, recall, roc, auc
    #   tags: estimator_name and estimator_class
    #   artifacts: training confusion matrix, model, estimator.html
    mlflow.autolog(silent=True)
    
    with mlflow.start_run(experiment_id=experiment.experiment_id,
                          description=run_description,
                          tags=run_tags) as run: 

        # Log datasets to mlflow
        mlflow.log_artifact("./data")
        
        params = {
            "C": 0.5,
            "kernel": "rbf",
            "degree": 3,
            "gamma": "scale",
            "coef0": 0.0,
            "shrinking": True,
            "probability": True,
            "tol": 1e-3,
            "cache_size": 200,
            "max_iter": -1,
            "decision_function_shape": "ovo"
        }
        
        # Log parameters to MLflow
        mlflow.log_params(params)
        
        # Define Support vector machine (svm) model for multi-class classification
        clf = svm.SVC(**params)
            
        # Convert labels from shape (120, 1) to (120,)
        y_train = y_train.to_numpy().flatten() 
        print(f"new y_train shape: {y_train.shape}")
    
        # Train and log model
        svm_model = clf.fit(X_train, y_train)
        # mlflow.sklearn.log_model(svm_model, artifact_path="sk_models") --> autolog do this
        
        # Predict
        y_test_predicted = svm_model.predict(X_test)
        y_test_probs = svm_model.predict_proba(X_test)
        print(f"y_test_predicted shape: {y_test_predicted.shape}")
        print(f"y_test_probs shape: {y_test_probs.shape}")
        
        # Evaluation
        clf_report = classification_report(y_true=y_test, y_pred=y_test_predicted)
        accuracy = accuracy_score(y_true=y_test, y_pred=y_test_predicted)
        cfm = confusion_matrix(y_true=y_test, y_pred=y_test_predicted)
        
        # Add confusion matrix image
        os.makedirs("./images", exist_ok=True)
        cfm_image_path = "images/cfm.png"
        classes = ["Setosa", "Versicolor", "Virginica"]
        df_cfm = pd.DataFrame(cfm, index = classes, columns = classes)
        plt.figure(figsize = (8,8))
        cfm_plot = sn.heatmap(cfm, annot=True)
        cfm_plot.figure.savefig(cfm_image_path)
        cfm_pil_image = Image.open(cfm_image_path)
        
        # Write classification report to file
        with open(os.path.join("./evaluation", "classification_report.txt"), "w+") as f:
            f.write(clf_report)
        
        # Log all evaluation metrics and artifacts
        mlflow.log_artifact("./evaluation")
        mlflow.log_artifact(__file__)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_image(cfm_pil_image, artifact_file=cfm_image_path)
        
    # Last active run
    last_active_run = mlflow.last_active_run()
    print(f"Last active run id: {last_active_run.info.run_id}")