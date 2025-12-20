import argparse
from clearml.automation import TaskScheduler
from clearml import Task
from dotenv import load_dotenv

load_dotenv()

# https://clear.ml/docs/latest/docs/references/sdk/scheduler/

parser = argparse.ArgumentParser(description="Run ClearML Scheduler for a specific Task ID")
parser.add_argument("--task_id", type=str, required=True, help="ID of the pipeline task (controller) to schedule")
args = parser.parse_args()

print(f"Loading task by ID: {args.task_id}...")
task = Task.get_task(task_id=args.task_id)

print(f"Task found: '{task.name}' [Project: {task.get_project_name()}]")

sched = TaskScheduler(
    sync_frequency_minutes=0.5
)

# sched.add_task(
#     schedule_task_id=task.id,  # id базовой задачи
#     queue="default",
#     name="Daily Churn Pipeline",
#     hour=2,       # в 02:00 UTC
#     minute=0,
#     recurring=True,
#     execute_immediately=False
# )

# Добавляем задачу в расписание
# Важно: queue="services", чтобы избежать Deadlock (если у вас 1 агент на default)
sched.add_task(
    schedule_task_id=task.id, 
    queue="services", 
    minute=1, 
    name="Scheduler (every 1 min)"
)

print("Scheduler started. Press Ctrl+C to stop.")
sched.start()