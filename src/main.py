import pandas as pd
import os
import yaml
from dotenv import load_dotenv
from clearml import Task, PipelineDecorator

load_dotenv()

with open("requirements.txt", "r") as f:
    PIPELINE_PACKAGES = list(filter(None, f.read().splitlines()))

def load_yaml_config(path: str = "config.yaml") -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found at: {path}")
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    print(f"Loaded configuration from {path}")
    return config

try:
    LOCAL_CONFIG = load_yaml_config("config.yaml")
except Exception as e:
    print(f"Warning: Could not load config.yaml ({e}). Using empty dict.")
    LOCAL_CONFIG = {}


@PipelineDecorator.component(
    return_values=["X_train", "X_test", "y_train", "y_test"],
    cache=True,
    task_type=Task.TaskTypes.data_processing,
    execution_queue="default",
    packages=PIPELINE_PACKAGES,
)
def step_process_data(dataset_config: dict):
    import pandas as pd
    import os
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from clearml import Dataset

    print(f"Retrieving dataset: {dataset_config['project']}/{dataset_config['name']}...")

    try:
        dataset_obj = Dataset.get(
            dataset_project=dataset_config['project'], 
            dataset_name=dataset_config['name'],
        )

        local_folder = dataset_obj.get_local_copy()
        print(f"Dataset downloaded to: {local_folder}")
        
    except ValueError:
        raise ValueError("Dataset not found! Please upload data first.")

    full_path = os.path.join(local_folder, dataset_config['filename'])
    df = pd.read_csv(full_path)

    # Предобработка
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    le = LabelEncoder()
    df["Churn"] = le.fit_transform(df["Churn"])
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=dataset_config['test_size'], 
        random_state=dataset_config['random_state']
    )
    print(f"Data processed. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test


@PipelineDecorator.component(
    return_values=["model"],
    cache=True,
    task_type=Task.TaskTypes.training,
    retry_on_failure=5,
    execution_queue="default",
    packages=PIPELINE_PACKAGES,
)
def step_train_model(X_train, y_train, model_config: dict):
    from sklearn.ensemble import RandomForestClassifier

    print(f"Training model with config: {model_config}")
    
    model = RandomForestClassifier(
        n_estimators=model_config['n_estimators'],
        random_state=model_config['random_state']
    )
    model.fit(X_train, y_train)

    return model


@PipelineDecorator.component(
    return_values=["accuracy"],
    cache=False,
    task_type=Task.TaskTypes.qc,
    execution_queue="default",
    packages=PIPELINE_PACKAGES,
)
def step_evaluate_model(model, X_test, y_test):
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    from clearml import Task

    print("Evaluating model...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy calculated: {acc:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    # print(f"Classification Report:\n{report}")
    
    # Логирование в ClearML
    task = Task.current_task()
    logger = task.get_logger()
    logger.report_scalar(title="Metrics", series="Accuracy", value=acc, iteration=1)
    logger.report_single_value(name="Accuracy", value=acc)
    logger.report_confusion_matrix(
        title="Model Performance",
        series="Confusion Matrix",
        matrix=cm,
        iteration=1,
        xaxis="Predicted",
        yaxis="Actual",
    )
    logger.report_text(f"Classification Report:\n{report}")

    return acc


@PipelineDecorator.component(
    return_values=["deploy_status"],
    task_type=Task.TaskTypes.custom,
    cache=False,
    execution_queue="default",
    packages=PIPELINE_PACKAGES,
)
def step_deploy_model(model, accuracy, deploy_config: dict):
    import os
    import joblib

    min_threshold = deploy_config["min_accuracy_threshold"]
    version = deploy_config["version"]

    if isinstance(accuracy, (list, tuple)):
        accuracy = accuracy[0]

    print(
        f"Checking deployment condition: Accuracy {accuracy:.4f} >= Threshold {min_threshold}"
    )

    if accuracy < min_threshold:
        msg = f"DEPLOYMENT SKIPPED: Accuracy ({accuracy:.4f}) is below threshold ({min_threshold})"
        print(msg)
        return msg

    print(f"Condition met. Starting deployment for version: {version}...")

    prod_path = os.path.join("prod_models", version)
    os.makedirs(prod_path, exist_ok=True)

    model_file = os.path.join(prod_path, "model.pkl")

    joblib.dump(model, model_file)
    print(f"Model serialized to production path: {os.path.abspath(model_file)}")

    from clearml import Task

    Task.current_task().upload_artifact(name="production_model", artifact_object=model)

    return "Deployed Success"


@PipelineDecorator.pipeline(
    name="Churn Automation Pipeline",
    project="Telco_Churn",
    version="3.0",
)
def run_pipeline(pipeline_settings: dict = LOCAL_CONFIG):
    X_train, X_test, y_train, y_test = step_process_data(
        dataset_config=pipeline_settings["dataset"]
    )

    model = step_train_model(
        X_train=X_train, 
        y_train=y_train, 
        model_config=pipeline_settings["model"]
    )

    accuracy = step_evaluate_model(
        model=model, 
        X_test=X_test, 
        y_test=y_test
    )

    step_deploy_model(
        model=model,
        accuracy=accuracy,
        deploy_config=pipeline_settings["deploy"]
    )


if __name__ == "__main__":
    run_pipeline()
    print("Pipeline submitted to queue 'default'. Check ClearML Dashboard!")