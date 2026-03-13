import mlflow


def start_experiment(experiment_name="document_pipeline"):
    mlflow.set_experiment(experiment_name)


def log_params(params: dict):
    for key, value in params.items():
        mlflow.log_param(key, value)


def log_metrics(metrics: dict):
    for key, value in metrics.items():
        mlflow.log_metric(key, value)