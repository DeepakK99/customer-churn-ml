import shutil
from pathlib import Path
import kagglehub

LOCAL_DATA_DIR = Path("data/raw")
LOCAL_DATA_DIR.mkdir(exist_ok=True)

print("⏳ Fetching dataset from Kaggle...")

cache_path = kagglehub.dataset_download("gauravtopre/bank-customer-churn-dataset")

print(f"📦 Files downloaded to cache. Moving to local project folder...")
for file_path in Path(cache_path).iterdir():
    if file_path.is_file():
        shutil.copy(file_path, LOCAL_DATA_DIR / file_path.name)

print(f"✅ Success! Dataset files are saved inside: {LOCAL_DATA_DIR.resolve()}")
