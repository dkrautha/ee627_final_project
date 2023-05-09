from pathlib import Path

import numpy as np
import pandas as pd

SUBMISSIONS_SCORE_MAP = {
    1: 0.83199,
    2: 0.79637,
    4: 0.79666,
    5: 0.81262,
    7: 0.79208,
    9: 0.79728,
    10: 0.81471,
    11: 0.82949,
    12: 0.83112,
    13: 0.8327,
    14: 0.79449,
    15: 0.79354,
    16: 0.82945,
    19: 0.65616,
    20: 0.74816,
    21: 0.83375,
    22: 0.79758,
    23: 0.79612,
    24: 0.81421,
    25: 0.84675,
    26: 0.5659,
    28: 0.84296,
    29: 0.82146,
    30: 0.84287,
    31: 0.843,
    32: 0.85712,
    33: 0.71837,
    34: 0.85712,
}


# Place submissions in this folder
# filenames should be 1.csv, 2.csv, ...
# 1.csv should correspond to 1 in the SUBMISSIONS_SCORE_MAP, etc
SUBMISSIONS_DIR = Path("./kaggle_submissions/")
SUBMISSIONS_PATHS = [Path(f"{SUBMISSIONS_DIR}/{i}.csv") for i in SUBMISSIONS_SCORE_MAP]

# opening each file as a dataframe individually
INDIVIDUAL_DATAFRAMES = [
    pd.read_csv(path).rename(columns={"Predictor": f"Predictor_{path.stem}"})
    for path in SUBMISSIONS_PATHS
]
# perform the transformation to have values of -1 and 1 instead of
# 0 and 1
for df in INDIVIDUAL_DATAFRAMES:
    cols = df.columns
    predictor = cols[1]
    df[predictor] = 2 * df[predictor] - 1


# helper function to horizontally combine multiple dataframes into one
def df_reduce(list_of_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    out = list_of_dfs[0]
    for frame in list_of_dfs[1:]:
        out = pd.merge(out, frame, on="TrackID")
    return out


s_original = df_reduce(INDIVIDUAL_DATAFRAMES)
print(s_original)
print(s_original.shape)

# drop TrackID column for doing matrix math
s = s_original.drop("TrackID", axis=1)

s_transposed = s.transpose()

st_times_s = s_transposed.dot(s)

inv_st_times_s = np.linalg.inv(st_times_s)

s_transposed_times_x = pd.DataFrame(
    {
        "submission_score": [
            120000 * (2 * score - 1) for score in SUBMISSIONS_SCORE_MAP.values()
        ],
    },
)

s_ensemble = s.dot(inv_st_times_s.dot(s_transposed_times_x)).rename(
    columns={0: "submission_score"}
)
s_ensemble["user_id"] = s_original["TrackID"].apply(lambda x: x.split("_")[0])
s_ensemble["track_id"] = s_original["TrackID"].apply(lambda x: x.split("_")[1])
s_ensemble["TrackID"] = s_original["TrackID"]

s_ensemble_sorted = (
    s_ensemble.sort_values(["user_id", "submission_score"])
    .reset_index()
    .drop("index", axis=1)
)
print(s_ensemble_sorted.head(12))

# choosing values of zero or one for each entry
write_zero = True
predictors = []
for index, row in s_ensemble_sorted.iterrows():
    if index % 3 == 0:
        write_zero = not write_zero

    predictors.append(int(write_zero))

s_ensemble_sorted["Predictor"] = predictors
print(s_ensemble_sorted.head(12))

# writing desired columns to csv
s_ensemble_sorted.to_csv(
    "ensemble-05.csv", index=False, columns=["TrackID", "Predictor"]
)
