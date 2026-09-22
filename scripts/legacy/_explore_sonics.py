"""Explore SONICS HF repo structure and download a small sample."""

import os
import shutil

import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_tree

REPO = "awsaf49/sonics"
OUT = "data/raw/sonics"

# 1. Explore fake_songs directory
print("=== Exploring fake_songs/ ===")
entries = list(list_repo_tree(REPO, repo_type="dataset", path_in_repo="fake_songs"))
folders = [e for e in entries if type(e).__name__ == "RepoFolder"]
files = [e for e in entries if type(e).__name__ == "RepoFile"]
print(f"Folders: {len(folders)}, Files: {len(files)}")

for e in entries[:10]:
    etype = type(e).__name__
    sz = getattr(e, "size", "?")
    print(f"  {e.path}  ({etype}, size={sz})")

if folders:
    print(f"\nExploring first subfolder: {folders[0].path}")
    sub = list(list_repo_tree(REPO, repo_type="dataset", path_in_repo=folders[0].path))
    for e in sub[:5]:
        print(f"  {e.path}  ({type(e).__name__})")

# 2. Check fake_songs.csv for filenames
print("\n=== Checking fake_songs.csv ===")
df_fake = pd.read_csv(f"{OUT}/metadata/fake_songs.csv")
print(f"Columns: {list(df_fake.columns)}")
print(f"Total rows: {len(df_fake)}")
print(f"Sample filenames:")
print(df_fake["filename"].head(5).tolist())

# 3. Try downloading using filenames from the CSV
print("\n=== Downloading 5 fake songs ===")
os.makedirs(f"{OUT}/fake_songs", exist_ok=True)
downloaded = 0
for _, row in df_fake.head(5).iterrows():
    fname = row["filename"]
    # Try different path patterns
    for pattern in [f"fake_songs/{fname}", fname]:
        try:
            path = hf_hub_download(
                repo_id=REPO,
                repo_type="dataset",
                filename=pattern,
                local_dir=f"{OUT}/hf_download",
            )
            shutil.copy2(path, f"{OUT}/fake_songs/{os.path.basename(fname)}")
            print(f"  OK: {fname}")
            downloaded += 1
            break
        except Exception as e:
            continue
    else:
        print(f"  FAIL: {fname}")

print(f"\nDownloaded {downloaded}/5 fake songs")

# 4. Check real_songs.csv
print("\n=== Checking real_songs.csv ===")
df_real = pd.read_csv(f"{OUT}/metadata/real_songs.csv")
print(f"Columns: {list(df_real.columns)}")
print(f"Total rows: {len(df_real)}")
print(f"Sample youtube_ids:")
print(df_real["youtube_id"].head(5).tolist())
