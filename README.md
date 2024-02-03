# udemy-mlflow

## Get started
1. Make virtual environment and download all packages in requirements.txt.
2. Run main.py.
3. Run `mlflow ui` to access the user interface of MLflow.

## Useful mlflow functions
### Tracking
`mlflow.set_tracking_uri(uri)`: Should be set before experiment is created or set. Can be used to set the folder where ml runs will be stored. Default folder name is "mlruns". Can also store runs remotely somewhere, for instance in Databricks or Azure. <br>

`mlflow.get_tracking_uri()`: Get the current tracking URI. This may not correspond to the tracking URI of the currently active run, since the tracking URI can be updated via ``set_tracking_uri``. <br>

### Experiment
`mlflow.create_experiment(name, artifact_location, tags)`: Create an experiment with a given name and optional artifact location and tags. The machine learning runs will be stored in this experiment. <br>

`mlflow.set_experiment(experiment_name, experiment_id)`: Set the given experiment as the active experiment. The experiment must either be specified by name via experiment_name or by ID via experiment_id. The experiment name and ID cannot both be specified. <br>

`mlflow.get_experiment(experiment_id)`: Retrieve an experiment by experiment_id from the backend store. <br>

### Run
`mlflow.start_run(experiment_id, ++)`: Start a new MLflow run, setting it as the active run under which metrics and parameters will be logged (mostly during training of ML models). The return value can be used as a context manager within a ``with`` block; otherwise, you must call ``end_run()`` to terminate the current run. <br>

`mlflow.end_run(status)`: End an active MLflow run (if there is one). <br>

`mlflow.active_run()`: Get current active run infomation, like run id and run name from the Run object the function returns. Should be used in a start_run `with` clause or between start_run and end_run. <br>

`mlflow.last_active_run()`: Get the last active run object. Can be used after end_run to get the run information from the already ended run. <br>

### Logging

`mlflow.log_param(key, value)`: Log a parameter (e.g. model hyperparameter) under the current run. If no run is active, this method will create a new active run. <br>

`mlflow.log_params(params)`: Log a batch of params for the current run. If no run is active, this method will create a new active run. <br>

`mlflow.log_metric(key, value)`: Log a metric under the current run. If no run is active, this method will create a new active run. <br>

`mlflow.log_metrics(metrics)`: Log multiple metrics for the current run. If no run is active, this method will create a new active run. <br>

`mlflow.sklearn.log_model(sk_model, artifact_path)`: Log a scikit-learn model as an MLflow artifact for the current run. Produces an MLflow Model. Does not register a model, it's logging it! <br>

`mlflow.log_image(image)`: Log an image as an artifact. The image objects that are supported are `numpy.ndarray` and `PIL.Image.Image`.

`mlflow.log_artifact(local_path)`: Log a local file or directory as an artifact of the currently active run. If no run is active, this method will create a new active run. <br>

`mlflow.log_artifacts(local_dir)`: Log all the contents of a local directory as artifacts of the run. If no run is active, this method will create a new active run. <br>

`mlflow.get_artifact_uri()`: Get the absolute URI of the specified artifact in the currently active run. If `path` is not specified, the artifact root URI of the currently active run will be returned; calls to ``log_artifact`` and ``log_artifacts`` write artifact(s) to subdirectories of the artifact root URI. If no run is active, this method will create a new active run. <br>

`mlflow.set_tag(key, value)`: Set a tag under the current run. If no run is active, this method will create a new active run. <br>

`mlflow.set_tags(tags)`: Log a batch of tags for the current run. If no run is active, this method will create a new active run. <br>
