"""
I never ever used TabICL before, so this script is a sandbox to try it out.

At the time of writing, I cannot predict the difficulty of the task, so maybe
this will stay a sandbox if it is super difficult or perhaps if it turns out to
be easy, then this might as well become my final script. We'll see.

Target columns are:
- `label`: binary classification of whether the packet is malicious or not
- `activity`: multi-class classification of the activity of the packet
    - my actual data is multic-class, too, so I will use this one

TODO:
- Implement some simple baselines such as majority class, random, etc.
- Implement some slightly more complex baselines such as KNN, SVM, etc.
- Compare to actually competetive models such as XGBoost
- If feasible, compare to TabPFN
- Maybe even compare to Flexynesis. It would be interesting to see how
it actually performs outside of biomolecular data.
- Run this outside of a notebook env and on powerful hardware
- Evaluate F1 and other useful metrics instead of accuracy
"""

# dependencies
import pandas as pd
import fastparquet as fp
import torch
import tabicl

from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import classification_report, accuracy_score

# constants
DEV_MODE_USE_SUBSET = True

# paths (from project package so they work regardless of cwd)
import tabular_foundation_models.paths as paths
PATH_PARQUET_FILE = str(
    paths.PATH_DATA_RAW / "bccc-cpacket-cloud-ddos-2024-merged.parquet"
)


if DEV_MODE_USE_SUBSET:
    device = torch.device("cpu")
else:
    # check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

# load dataset from Parquet file to a Pandas df
df = fp.ParquetFile(PATH_PARQUET_FILE).to_pandas()

# if still developing, use a small subset of data for rapid prototyping
# StratifiedShuffleSplit preserves class distribution so train_test_split later won't fail
if DEV_MODE_USE_SUBSET:
    sss = StratifiedShuffleSplit(n_splits=1, train_size=2500, random_state=42)
    idx, _ = next(sss.split(df, df["activity"]))
    df = df.iloc[idx].copy()


# drop the binary label column since we're using activity
df = df.drop('label', axis=1)

# separate features and target
X = df.drop('activity', axis=1)
y = df['activity']

# train-test split (80-20 is standard)
# this is just a dummy, so don't overcomplicate with cross val or val set
# goal is to get this running as fast as possible
# I got my actual data on my server, and it is split properly
# here, I just want to practice, so a quick and dirty split is good enough
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# initialize and train TabICL classifier
# use GPU if available, else default to CPU
# using defaults is fine for sandbox - TabICL handles everything
clf = tabicl.TabICLClassifier(device=device)

# just for future reference:
# this step took about 5 min for 500 samples when run in parallel on CPU
# I didn't catch the exact number of cores, but I think it was sth
# around 20 to 30
# but actually I think the real limiting factor for running this on a
# laptop is memory instead of compute
# ideally use a GPU if you have access
clf.fit(X_train, y_train)

# predict on test set
y_pred = clf.predict(X_test)

# evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# save predictions if needed
results_df = pd.DataFrame({
    'true_activity': y_test.values,
    'predicted_activity': y_pred
})
results_df.to_csv('tabicl_predictions.csv', index=False)
