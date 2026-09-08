import shutil
from pathlib import Path
import kagglehub


def load_data():
    LOCAL_DATA_DIR = Path("data/raw")
    LOCAL_DATA_DIR.mkdir(exist_ok=True)

    if LOCAL_DATA_DIR.exists() and any(LOCAL_DATA_DIR.iterdir()):
        print(f"🔄 Data already downloaded! Loading local files from: {LOCAL_DATA_DIR.resolve()}")
        return [file for file in LOCAL_DATA_DIR.iterdir() if file.is_file()]

    print("⏳ Fetching dataset from Kaggle...")

    cache_path = kagglehub.dataset_download("gauravtopre/bank-customer-churn-dataset")

    saved_files = []

    print(f"📦 Files downloaded to cache. Moving to local project folder...")
    for file_path in Path(cache_path).iterdir():
        if file_path.is_file():
            shutil.copy(file_path, LOCAL_DATA_DIR / file_path.name)
            saved_files.append(LOCAL_DATA_DIR / file_path.name)

    print(f"✅ Success! Dataset files are saved inside: {LOCAL_DATA_DIR.resolve()}")

    return saved_files
