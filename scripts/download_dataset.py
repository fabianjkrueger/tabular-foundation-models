"""
Download the dataset from Kaggle and save it to the data/raw directory.

The dataset is available at:
`https://www.kaggle.com/datasets/dhoogla/bccc-cpacket-cloud-ddos-2024`

The reason I chose this particular dataset is that the description states:
"The cleaned up version of the dataset in parquet format, immediately usable
after loading."

This particular project is about using tabular foundational models using
in-context learning (ICL).
I want to see how well they work and compare them to alternative algorithms.
It is explicitly **not** about data cleaning, feature engineering,
visualizations, or any other data science or machine learning task.
Those tasks are super valuable and important.
This is why I have separate portfolio projects for those tasks.
But here, I dont want to spend time on data preparation or exploration, but
instead go for the modeling right away.

Let's see if the dataset is actually as usable as it promises.
I hope they didn't mean it was immediately usable for cleaning lol.

How to run this script:
`uv run scripts/download_dataset.py`
"""

# dependencies
import kagglehub
import shutil

# paths (from project package so they work regardless of cwd)
import tabular_foundation_models.paths as paths

# destination parent must exist
# otherwise shutil.move renames src to dst and fails if parent is missing
paths.PATH_DATA.mkdir(parents=True, exist_ok=True)
# replace existing data/raw so each run leaves the dataset at data/raw
# (not data/raw/<version>)
if paths.PATH_DATA_RAW.exists():
    shutil.rmtree(paths.PATH_DATA_RAW)
    
# download latest version to cache
path = kagglehub.dataset_download("dhoogla/bccc-cpacket-cloud-ddos-2024")
print("Downloaded dataset to:", path)

# move so dataset ends up at data/raw (rename cache dir to PATH_DATA_RAW)
shutil.move(path, paths.PATH_DATA_RAW)
print("Moved dataset to:", paths.PATH_DATA_RAW)
