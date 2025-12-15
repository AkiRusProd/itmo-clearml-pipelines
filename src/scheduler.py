from clearml.automation import TaskScheduler
from clearml import Task
from dotenv import load_dotenv

load_dotenv()
# https://clear.ml/docs/latest/docs/references/sdk/scheduler/

# ID задачи-ланчера, которую хотим клонировать по расписанию
# Предполагаем, что launcher уже был запущен вручную
task = Task.get_task(project_name="Telco_Churn", task_name="Churn Automation Pipeline")

# Scheduler — сам «cron-движок»
sched = TaskScheduler(
    sync_frequency_minutes=1.0  # как часто синхронизировать расписание
)

# Добавляем задачу в расписание
# sched.add_task(
#     schedule_task_id=task.id,  # id базовой задачи
#     queue="default",
#     name="Daily Churn Pipeline",
#     hour=2,       # в 02:00 UTC
#     minute=0,
#     recurring=True,
#     execute_immediately=False
# )

sched.add_task(schedule_task_id=task.id, queue="default", minute=5)

scheduler = TaskScheduler(
    sync_frequency_minutes=0.5  # как часто проверять расписание
)


# Запускаем сам планировщик (он будет блокировать выполнение)
sched.start()
