from clearml.automation import TriggerScheduler
from clearml import Task
from dotenv import load_dotenv

load_dotenv()


# 1. Находим ID задачи-пайплайна, которую будем запускать.
# Желательно брать успешную задачу из "архива" или последнюю успешную.
# Можно искать по имени:
base_task = Task.get_task(
    project_name="Telco_Churn", 
    task_name="Churn Automation Pipeline",
    # task_filter={'status': ['completed']} # Берем только успешно завершенную
)

if not base_task:
    print("Base task not found! Run the pipeline manually at least once.")
    exit(1)

print(f"Base task ID found: {base_task.id}")

# 2. Создаем планировщик триггеров
trigger = TriggerScheduler(
    pooling_frequency_minutes=0.5  # Проверять события раз 0.5 минуты
)

# 3. Добавляем правило: запускать, если создан/опубликован новый датасет
# trigger.add_dataset_trigger(
#     name='Trigger on New Data',           # Имя триггера (появится в консоли)
#     schedule_task_id=base_task.id,        # Какую задачу клонировать и запускать
#     schedule_queue='default',             # В какой очереди запускать

#     # УСЛОВИЯ СРАБАТЫВАНИЯ:
#     trigger_project="Telco_Churn",        # Следить за датасетами в этом проекте
#     trigger_name="Customer_Churn_Raw",  # (Опционально) Фильтр по имени датасета
#     trigger_on_publish=True,              # Срабатывать при публикации (Dataset.publish())
#     # trigger_on_tags=['ready_for_train'] # Или срабатывать при добавлении тега
# )

trigger.add_dataset_trigger(
    name="Trigger on new data tag",
    schedule_task_id=base_task.id,
    schedule_queue="default",

    trigger_project="Telco_Churn",
    trigger_name="Customer_Churn_Raw",
    trigger_on_tags=["new_data"]   # ✅ РАБОТАЕТ С ВЕРСИЯМИ
)


print("Trigger Service started... Waiting for new datasets.")
# 4. Запускаем "вечный" цикл прослушки
trigger.start()