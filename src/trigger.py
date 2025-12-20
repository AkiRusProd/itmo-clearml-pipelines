import argparse
from clearml.automation import TriggerScheduler
from clearml import Task
from dotenv import load_dotenv

load_dotenv()

# 1. Парсим аргументы командной строки
parser = argparse.ArgumentParser(description="Run ClearML Trigger for a specific Task ID")
parser.add_argument("--task_id", type=str, required=True, help="ID of the pipeline task (controller) to trigger")
args = parser.parse_args()

# 2. Получаем задачу строго по ID
print(f"Loading base task by ID: {args.task_id}...")
base_task = Task.get_task(task_id=args.task_id)

print(f"Base task found: '{base_task.name}' [Project: {base_task.get_project_name()}]")

# 3. Создаем планировщик триггеров
trigger = TriggerScheduler(
    pooling_frequency_minutes=0.5  # Проверять события раз 0.5 минуты
)

# 4. Добавляем правило: запускать при появлении тега "new_data" на датасете
trigger.add_dataset_trigger(
    name="Trigger on new data tag",
    schedule_task_id=base_task.id,
    schedule_queue="services",     
    trigger_project="Telco_Churn",
    trigger_name="Customer_Churn_Raw",
    trigger_on_tags=["new_data"],  
)

print("Trigger Service started... Waiting for new datasets.")
# 5. Запускаем "вечный" цикл прослушки
trigger.start()