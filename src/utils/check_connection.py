import os
from dotenv import load_dotenv

load_dotenv()

print(f"Testing connection to: {os.getenv('CLEARML_WEB_HOST')}")

from clearml import Task


def check_setup():
    try:
        print("Initializing ClearML Task...")

        task = Task.init(
            project_name="System Check",
            task_name="Connection Test",
            output_uri=True,  # Проверка, что files_host тоже работает
        )

        print("\nSUCCESS! Connection established.")
        print(f"View this task here: {task.get_output_log_web_page()}")

        task.get_logger().report_text("System is ready for Pipeline demo!")
        task.close()

    except Exception as e:
        print("\nERROR: Connection failed.")
        print(f"Details: {e}")


if __name__ == "__main__":
    check_setup()
