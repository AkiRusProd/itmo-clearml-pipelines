import pandas as pd
from dotenv import load_dotenv
from clearml import Task, PipelineDecorator, Dataset, OutputModel

# Загрузка переменных окружения
load_dotenv()

try:
    with open("requirements.txt", "r") as f:
        # filter(None, ...) удаляет пустые строки
        PIPELINE_PACKAGES = list(filter(None, f.read().splitlines()))
except FileNotFoundError:
    # Фоллбек, если файла нет, но лучше чтобы он был
    PIPELINE_PACKAGES = ['numpy<2.0', 'pandas', 'scikit-learn', 'clearml', 'python-dotenv']

# --- ШАГ 1: ПОДГОТОВКА ДАННЫХ ---
# Добавили кэширование. Если данные те же, шаг пропустится (экономия времени)
@PipelineDecorator.component(
    return_values=['X_train', 'X_test', 'y_train', 'y_test'], 
    cache=True,
    task_type=Task.TaskTypes.data_processing,
    execution_queue='default',
    packages=PIPELINE_PACKAGES
)
def step_process_data(dataset_project: str, dataset_name: str, local_path: str = None, test_size: float = 0.2, random_state: int = 42):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from clearml import Dataset
    import os

    # ЛОГИКА РАБОТЫ С ДАННЫМИ:
    # Пытаемся получить датасет из ClearML. Если нет - создаем из локального файла.
    try:
        # Пытаемся найти последнюю версию датасета
        dataset = Dataset.get(dataset_project=dataset_project, dataset_name=dataset_name)
        dataset_path = dataset.get_local_copy()
        print(f"Using cached dataset from: {dataset_path}")
        # Предполагаем, что имя файла стандартное, либо ищем csv
        csv_file = [f for f in os.listdir(dataset_path) if f.endswith('.csv')][0]
        full_path = os.path.join(dataset_path, csv_file)
    except ValueError:
        if local_path and os.path.exists(local_path):
            print(f"Dataset not found in ClearML. Creating new version from {local_path}...")
            dataset = Dataset.create(dataset_project=dataset_project, dataset_name=dataset_name)
            dataset.add_files(local_path)
            dataset.upload()
            dataset.finalize()
            full_path = local_path
        else:
            raise FileNotFoundError("Dataset not found in ClearML and no local path provided!")

    df = pd.read_csv(full_path)
    
    # Предобработка
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    le = LabelEncoder()
    df['Churn'] = le.fit_transform(df['Churn'])
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    df = pd.get_dummies(df, drop_first=True)
    
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test

# --- ШАГ 2: ОБУЧЕНИЕ ---
# Добавили retry_on_failure=True и execution_timeout (ДЛЯ ЗАДАНИЯ)
@PipelineDecorator.component(
    return_values=['model'], 
    cache=True,
    task_type=Task.TaskTypes.training,
    retry_on_failure=True,     # Автоматический перезапуск при ошибке
    execution_queue='default',
    packages=PIPELINE_PACKAGES,
)
def step_train_model(X_train: pd.DataFrame, y_train: pd.Series, n_estimators: int = 100):
    from sklearn.ensemble import RandomForestClassifier
    
    print(f"Training Random Forest with {n_estimators} estimators...")
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    
    return model

# --- ШАГ 3: ВАЛИДАЦИЯ ---
@PipelineDecorator.component(return_values=['accuracy'], cache=False, task_type=Task.TaskTypes.qc, execution_queue='default', packages=PIPELINE_PACKAGES)
def step_evaluate_model(model: object, X_test: pd.DataFrame, y_test: pd.Series):
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    from clearml import Task
    # --- 1. Бизнес-логика (Чистый Python) ---
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Accuracy calculated: {acc:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    # print(f"Classification Report:\n{report}")
    

    # --- 2. Логирование и Визуализация (ClearML) ---
    # Получаем логгер один раз. Если запускаем без ClearML, это вернет None (безопасно)
    task = Task.current_task()
    logger = task.get_logger() if task else None
    
    if logger:
        # А) Скаляр для графика (динамика)
        logger.report_scalar(
            title="Metrics", 
            series="Accuracy", 
            value=acc, 
            iteration=1
        )
        
        # Б) Single Value для таблицы сравнения экспериментов
        logger.report_single_value(name="Accuracy", value=acc)

        # В) Confusion Matrix (Красивая картинка вместо текста!)
        # ClearML умеет строить её сам по сырым данным
        logger.report_confusion_matrix(
            title="Model Performance",
            series="Confusion Matrix",
            matrix=cm,
            iteration=1,
            xaxis="Predicted",
            yaxis="Actual"
        )

        logger.report_text(f"Classification Report:\n{report}")
    
    return acc

# --- ШАГ 4: "ДЕПЛОЙ" (УСЛОВНЫЙ ШАГ) ---
# Этот шаг запустится только если модель хорошая
# --- ИСПРАВЛЕННЫЙ ШАГ ДЕПЛОЯ ---

# --- ИСПРАВЛЕННЫЙ ШАГ ДЕПЛОЯ (с защитой от типов) ---
@PipelineDecorator.component(
    return_values=['deploy_status'], 
    task_type=Task.TaskTypes.custom, 
    cache=False,
    execution_queue='default', 
    packages=PIPELINE_PACKAGES
)
def step_deploy_model(model, accuracy, min_threshold: float, version: str = "latest"):
    import os
    import joblib
    from sklearn.base import BaseEstimator
    
    # === ФИКС ПРОБЛЕМЫ ===
    # Если accuracy пришел как кортеж (например, (0.75,)), берем первый элемент
    if isinstance(accuracy, (list, tuple)):
        accuracy = accuracy[0]
    # =====================

    print(f"Checking deployment condition: Accuracy {accuracy:.4f} >= Threshold {min_threshold}")
    
    if accuracy < min_threshold:
        msg = f"DEPLOYMENT SKIPPED: Accuracy ({accuracy:.4f}) is below threshold ({min_threshold})"
        print(msg)
        return msg

    print(f"Condition met. Starting deployment for version: {version}...")
    
    prod_path = os.path.join("prod_models", version)
    os.makedirs(prod_path, exist_ok=True)
    
    model_file = os.path.join(prod_path, "model.pkl")
    
    # Сохраняем (эмуляция продакшена)
    joblib.dump(model, model_file)
    print(f"Model serialized to production path: {os.path.abspath(model_file)}")
    
    # Загружаем как артефакт (чтобы видеть в UI)
    from clearml import Task
    Task.current_task().upload_artifact(
        name="production_model", 
        artifact_object=model
    )
    
    return "Deployed Success"

# --- ИСПРАВЛЕННЫЙ КОНТРОЛЛЕР ---
@PipelineDecorator.pipeline(
    name='Churn Automation Pipeline', 
    project='Telco_Churn', 
    version='2.0.1',
    # pipeline_execution_queue='default',
    # packages=PIPELINE_PACKAGES
)
def run_pipeline(
    dataset_local_path,
    min_accuracy_threshold=0.75,
    rf_n_estimators=100
):
    # 1. Данные
    X_train, X_test, y_train, y_test = step_process_data(
        dataset_project="Telco_Churn",
        dataset_name="Customer_Churn_Raw",
        local_path=dataset_local_path
    )
    
    # 2. Обучение
    model = step_train_model(
        X_train=X_train, 
        y_train=y_train, 
        n_estimators=rf_n_estimators
    )
    
    # 3. Оценка
    accuracy = step_evaluate_model(
        model=model, 
        X_test=X_test, 
        y_test=y_test
    )
    
    # 4. Деплой (с логикой внутри)
    # Мы передаем accuracy (это Future объект) и порог (число).
    # ClearML сам "развернет" Future в число, когда запустит этот шаг.
    status = step_deploy_model(
        model=model, 
        accuracy=accuracy,          # <--- Передаем результат предыдущего шага
        min_threshold=min_accuracy_threshold, # <--- Передаем параметр пайплайна
        version="v2.0_prod"
    )

if __name__ == '__main__':
    # 1. Сначала отладка локально (можно закомментировать после проверки)
    # PipelineDecorator.run_locally()
    
    # 2. Абсолютный путь для локального запуска может не сработать на агенте, 
    # если агент на другой машине. Но если агент локальный - ок.
    # Лучше использовать относительный путь или Dataset API (который у тебя уже есть внутри шага).
    
    # Мы передаем путь None, чтобы логика внутри шага step_process_data
    # попыталась стянуть датасет из ClearML, если он там есть.
    import os
    abs_data_path = os.path.abspath('data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')

    # ЗАПУСК ПАЙПЛАЙНА
    # Вместо прямого вызова функции, мы создаем объект Pipeline
    # Это зарегистрирует пайплайн в ClearML, но не запустит выполнение кода локально,
    # а отправит его в очередь 'default'.
    
    pipeline_obj = run_pipeline(
        dataset_local_path=abs_data_path, 
        rf_n_estimators=150,
        min_accuracy_threshold=0.78
    )
    
    # ВАЖНО: Раскомментируй эту строку, чтобы отправить задачу Агенту!
    # pipeline_obj.start(queue='default') 
    pipeline_obj.start() 
    
    print("Pipeline submitted to queue 'default'. Check ClearML Dashboard!")