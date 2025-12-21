from dotenv import load_dotenv
from clearml import Task, PipelineDecorator

from utils.config_loader import PIPELINE_PACKAGES, LOCAL_CONFIG

load_dotenv()


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
        random_state=model_config['random_state'],
        class_weight='balanced' 
    )
    model.fit(X_train, y_train)

    return model


@PipelineDecorator.component(
    return_values=["f1_score"],
    cache=False,
    task_type=Task.TaskTypes.qc,
    execution_queue="default",
    packages=PIPELINE_PACKAGES,
)
def step_evaluate_model(model, X_test, y_test):
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score, 
        roc_auc_score, confusion_matrix, classification_report, 
        roc_curve, precision_recall_curve
    )
    from clearml import Task

    print("Evaluating model...")
    
    # 1. Получаем предсказания классов и вероятностей
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Вероятность класса 1 (Churn)

    # 2. Считаем скалярные метрики
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    print(f"Metrics -> Accuracy: {acc:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")

    # 3. Логируем скаляры в ClearML
    task = Task.current_task()
    logger = task.get_logger()
    
    # Сводная таблица метрик (Scalars)
    logger.report_scalar("Metrics", "F1_Score", f1, iteration=1)
    logger.report_scalar("Metrics", "Accuracy", acc, iteration=1)
    logger.report_scalar("Metrics", "Precision", precision, iteration=1)
    logger.report_scalar("Metrics", "Recall", recall, iteration=1)
    logger.report_scalar("Metrics", "ROC_AUC", roc_auc, iteration=1)

    # Single Value для сравнения экспериментов в таблице
    logger.report_single_value(name="F1_Score", value=f1)
    logger.report_single_value(name="ROC_AUC", value=roc_auc)

    # 4. Confusion Matrix (Интерактивная в ClearML)
    class_names = ["No Churn", "Churn"]

    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}")
    logger.report_confusion_matrix(
        title="Model Performance",
        series="Confusion Matrix",
        matrix=cm,
        iteration=1,
        xaxis="Predicted class",
        yaxis="True class",
        xlabels=class_names,
        ylabels=class_names
    )
    
    # 5. Classification Report (Текст)
    report = classification_report(y_test, y_pred)
    logger.report_text(f"Classification Report:\n{report}")

    # 6. ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})', color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True)
    # Логируем фигуру напрямую
    logger.report_matplotlib_figure(title="Performance Plots", series="ROC Curve", figure=plt)
    plt.close()

    # 7. Precision-Recall Curve Plot
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    plt.figure(figsize=(10, 6))
    plt.plot(rec, prec, label=f'F1 Score = {f1:.2f}', color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="best")
    plt.grid(True)
    logger.report_matplotlib_figure(title="Performance Plots", series="PR Curve", figure=plt)
    plt.close()

    # 8. Feature Importance Plot
    # Проверяем, есть ли у модели feature_importances_ (для деревьев)
    if hasattr(model, "feature_importances_"):
        plt.figure(figsize=(10, 8))
        features = X_test.columns.tolist()
        importances = model.feature_importances_
        indices = np.argsort(importances)
        plt.title("Feature Importances")
        plt.barh(range(len(indices)), importances[indices], color="b", align="center")
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel("Relative Importance")

        if logger:
            logger.report_matplotlib_figure(
                title="Feature Importance", series="Top Features", figure=plt
            )
        plt.close()
        

    return f1


@PipelineDecorator.component(
    return_values=["deploy_status"],
    task_type=Task.TaskTypes.custom,
    cache=False,
    execution_queue="default",
    packages=PIPELINE_PACKAGES,
)
def step_deploy_model(model, metric_value, deploy_config: dict):
    import os
    import joblib

    min_threshold = deploy_config["min_f1_threshold"]
    version = deploy_config["version"]

    if isinstance(metric_value, (list, tuple)):
        metric_value = metric_value[0]

    print(
        f"Checking deployment condition: F1-Score {metric_value:.4f} >= Threshold {min_threshold}"
    )

    if metric_value < min_threshold:
        msg = f"DEPLOYMENT SKIPPED: F1-Score ({metric_value:.4f}) is below threshold ({min_threshold})"
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
    version="3.1",
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

    f1_score_val = step_evaluate_model(
        model=model, 
        X_test=X_test, 
        y_test=y_test
    )

    step_deploy_model(
        model=model,
        metric_value=f1_score_val,
        deploy_config=pipeline_settings["deploy"]
    )


if __name__ == "__main__":
    # PipelineDecorator.run_locally() # Running locally for testing without agents
    run_pipeline()
    print("Pipeline submitted to queue 'default'. Check ClearML Dashboard!")