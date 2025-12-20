from clearml import Task
from dotenv import load_dotenv
load_dotenv()

#---------------------------------------
PROJECT_NAME = "Telco_Churn" 

print(f"--- Searching all tasks in project: '{PROJECT_NAME}' ---")

tasks = Task.get_tasks(
    project_name=PROJECT_NAME,
)

if not tasks:
    print(f"No tasks found in project '{PROJECT_NAME}'. Check project name!")
else:
    print(f"{'ID':<34} | {'TYPE':<12} | {'STATUS':<10} | NAME")
    print("-" * 80)
    for t in tasks:
        print(f"{t.id:<34} | {str(t.task_type):<12} | {t.status:<10} | '{t.name}'")

print("-" * 80)


#---------------------------------------

# Вставьте ID той задачи, которую видите в UI, но не видит скрипт
# MY_UI_ID = "6039aedd9cf84ed4b5810fe5105e42c7" 
MY_UI_ID = "2eef602034af4f989d1819360dd7fcf3"

try:
    t = Task.get_task(task_id=MY_UI_ID)
    print(f"Found Task: {t.name}")
    print(f"Real Project: '{t.get_project_name()}'")
    print(f"Real Status:  {t.status}")
except Exception as e:
    print(f"Вы подключены к тому же серверу? Ошибка: {e}")