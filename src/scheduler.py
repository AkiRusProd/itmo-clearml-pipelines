from clearml.automation import TaskScheduler

def schedule_pipeline_execution():
    # Создаем планировщик
    scheduler = TaskScheduler(
        name="Nightly Churn Retraining",
        schedule_class=TaskScheduler.ScheduleMinute, # Для демо ставим запускать каждую минуту (или ScheduleDaily)
        schedule_minute=5,  # Каждые 5 минут
        # Указываем проект и имя пайплайна, который мы определили в декораторе
        base_task_project="Telco_Churn",
        base_task_name="Churn Automation Pipeline", 
        queue="default"
    )
    
    # Запускаем планировщик
    scheduler.start()
    print("Scheduler started. Pipeline will run every 5 minutes.")

# Вызвать эту функцию отдельно один раз
schedule_pipeline_execution()