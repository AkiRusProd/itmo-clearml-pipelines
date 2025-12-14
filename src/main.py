import os
import pandas as pd
from dotenv import load_dotenv
from clearml import Task, PipelineDecorator

# 1. Загрузка переменных (чтобы подключиться к серверу)
load_dotenv()

# --- ШАГИ (STEPS) ---
# Кэширование включено: если данные не менялись, шаг не будет перезапускаться
@PipelineDecorator.component(return_values=['X_train', 'X_test', 'y_train', 'y_test'], cache=True)
def step_process_data(data_path: str, test_size: float = 0.2, random_state: int = 42):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    # Предобработка
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    le = LabelEncoder()
    df['Churn'] = le.fit_transform(df['Churn'])
    df = df.drop('customerID', axis=1)
    df = pd.get_dummies(df, drop_first=True)
    
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Data processed. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

@PipelineDecorator.component(return_values=['model'], cache=True)
def step_train_model(X_train: pd.DataFrame, y_train: pd.Series, n_estimators: int = 100):
    from sklearn.ensemble import RandomForestClassifier
    
    print(f"Training Random Forest with {n_estimators} estimators...")
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    
    return model

@PipelineDecorator.component(return_values=['accuracy', 'report'], cache=False)
def step_evaluate_model(model: object, X_test: pd.DataFrame, y_test: pd.Series):
    from sklearn.metrics import accuracy_score, classification_report
    from clearml import Task
    
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Accuracy: {acc}")
    print("Classification Report:\n", report)
    
    # Логируем метрику в задачу текущего шага
    current_task = Task.current_task()
    if current_task:
        current_task.get_logger().report_scalar(
            title="Performance", series="Accuracy", value=acc, iteration=1
        )
    
    return acc, report

# --- КОНТРОЛЛЕР ---

# Здесь мы задаем имя Проекта и имя Пайплайна.
# Именно этот декоратор создаст задачу, видимую во вкладке Pipelines.
@PipelineDecorator.pipeline(
    name='Churn Pipeline Controller', 
    project='Telco_Churn',  # Плоское имя проекта (без слешей)
    version='1.0.0'
)
def run_pipeline(
    data_path,
    train_test_ratio=0.2,
    rf_n_estimators=100
):
    # Логика графа
    X_train, X_test, y_train, y_test = step_process_data(
        data_path=data_path, 
        test_size=train_test_ratio
    )
    
    model = step_train_model(
        X_train=X_train, 
        y_train=y_train, 
        n_estimators=rf_n_estimators
    )
    
    step_evaluate_model(
        model=model, 
        X_test=X_test, 
        y_test=y_test
    )

if __name__ == '__main__':
    # Включаем локальный запуск КОНТРОЛЛЕРА. 
    # Это значит: "Мозг" пайплайна работает здесь (в python churn_pipeline.py), 
    # но он будет создавать задачи-шаги (Step Tasks) в ClearML.
    # Если у вас запущен clearml-agent, он подхватит шаги.
    PipelineDecorator.run_locally()
    
    # Абсолютный путь, чтобы локальный агент точно нашел файл
    abs_data_path = os.path.abspath('data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    print(f"Starting pipeline in project 'Telco_Churn' with data: {abs_data_path}")
    
    run_pipeline(
        data_path=abs_data_path,
        train_test_ratio=0.25,
        rf_n_estimators=150
    )