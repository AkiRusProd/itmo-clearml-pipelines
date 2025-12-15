from clearml import Dataset
from dotenv import load_dotenv

load_dotenv()

ds = Dataset.create(dataset_project="Telco_Churn", dataset_name="Customer_Churn_Raw")
ds.add_files("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
ds.upload()
ds.add_tags(["new_data"])
ds.finalize()
