import glob
from typing import List

import pandas as pd


def merge_csv(csv_files: List[str]):
    df = pd.concat(
        map(pd.read_csv, csv_files),
    )
    sorted_df = df.sort_values(by=["issue_number"], ascending=True)
    sorted_df.to_csv("merged.csv")


csv_files = glob.glob("data/openj9_data*")
merge_csv(csv_files)
