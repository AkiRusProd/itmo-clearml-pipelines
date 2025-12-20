from clearml import Dataset
from dotenv import load_dotenv
import os

load_dotenv()

PROJECT_NAME = "Telco_Churn"
DATASET_NAME = "Customer_Churn_Raw"
LOCAL_CSV_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"

def init_dataset():
    print(f"Checking local file: {LOCAL_CSV_PATH}")
    if not os.path.exists(LOCAL_CSV_PATH):
        raise FileNotFoundError(f"File not found: {LOCAL_CSV_PATH}")

    print("Creating Dataset in ClearML...")
    ds = Dataset.create(
        dataset_project=PROJECT_NAME, 
        dataset_name=DATASET_NAME
    )
    
    print(f"Uploading file: {LOCAL_CSV_PATH}")
    ds.add_files(LOCAL_CSV_PATH)
    ds.upload()
    ds.add_tags(["new_data"])
    ds.finalize()
    
    print(f"SUCCESS! Dataset ID: {ds.id}")
    print("Now you can run the Pipeline.")

if __name__ == "__main__":
    init_dataset()