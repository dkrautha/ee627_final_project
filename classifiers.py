# %%

import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    DecisionTreeClassifier,
    GBTClassifier,
    LogisticRegression,
    RandomForestClassifier,
)
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from parsing import Path, load_data

spark = SparkSession.builder.appName("recommend-ML").getOrCreate()
sc = spark.sparkContext
sc.pythonExec = "python"
output = pd.read_csv(
    "data/output1.txt",
    header=None,
    sep="|",
    names=["userID", "trackID", "albumRating", "artistRating"],
)

# %%
# read test2_new.txt with schema ["artistID", "trackID", "label"]
test2 = pd.read_csv(
    "data/test2_new.txt",
    header=None,
    sep="|",
    names=["userID", "trackID", "recommendation"],
)
train_data_pandas = pd.merge(output, test2, on=["userID", "trackID"])

train_data = spark.createDataFrame(train_data_pandas)
cols = train_data.columns

# %%

numericCols = [
    "albumRating",
    "artistRating",
]
stages = []
assemblerInputs = numericCols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]

# %%

pipeline = Pipeline(stages=stages)
pipelineModel = pipeline.fit(train_data)
train_data = pipelineModel.transform(train_data)
selectedCols = ["features"] + cols
train_data = train_data.select(selectedCols)

# %%

_, _, _, _, _, test_map = load_data()
output_for_df_training = Path("./data/testItem2.df")
with open(output_for_df_training, "w") as f:
    f.write("userID,trackID\n")
    for user_id, tracks in test_map.items():
        for track in tracks:
            f.write(f"{user_id},{track}\n")

# %%

test_dataframe = pd.read_csv(
    output_for_df_training, header=0, sep=",", names=["userID", "trackID"]
)
test_dataframe = pd.merge(test_dataframe, output, on=["userID", "trackID"], how="left")
# replace all NaN with 0
test_dataframe = test_dataframe.fillna(0)
test_dataframe = spark.createDataFrame(test_dataframe)
pipeline = Pipeline(stages=stages)
pipelineModel = pipeline.fit(test_dataframe)
test_dataframe = pipelineModel.transform(test_dataframe)
test_dataframe = test_dataframe.select(selectedCols)

# %%
print(test_dataframe.head())
lr = LogisticRegression(featuresCol="features", labelCol="recommendation", maxIter=10)
lrModel = lr.fit(train_data)
predictions = lrModel.transform(test_dataframe)
sort_predictions = predictions.select(
    "userID", "trackID", "probability", "rawPrediction", "prediction"
).sort(col("userID").asc(), col("probability").desc())

# %%
import os


def write_to_submission(predictions_to_write):
    fileversion = 1
    while True:
        results_name = f"submissions/results-{fileversion:02}.csv"
        if not os.path.exists(results_name):
            break
        fileversion += 1
    i = 0
    curUserId = None
    with open(results_name, "w") as f:
        f.write("TrackID,Predictor\n")
        for row in predictions_to_write.collect():
            if curUserId != row["userID"]:
                i = 0
                curUserId = row["userID"]
            f.write(f"{row['userID']}_{row['trackID']},{0 if i < 3 else 1}\n")
            i += 1


# %%
write_to_submission(sort_predictions)

# %%

dt = DecisionTreeClassifier(
    featuresCol="features", labelCol="recommendation", maxDepth=3
)
dtModel = dt.fit(train_data)
predictions = dtModel.transform(test_dataframe)
sort_predictions = predictions.select(
    "userID", "trackID", "probability", "rawPrediction", "prediction"
).sort(col("userID").asc(), col("probability").desc())

# %%
sort_predictions.show(6)
write_to_submission(sort_predictions)

# %%
# random forest classifier in pySpark

rf = RandomForestClassifier(featuresCol="features", labelCol="recommendation")
rfModel = rf.fit(train_data)
predictions = rfModel.transform(test_dataframe)
sort_predictions = predictions.select(
    "userID", "trackID", "probability", "rawPrediction", "prediction"
).sort(col("userID").asc(), col("probability").desc())

# %%
sort_predictions.show(6)
write_to_submission(sort_predictions)

# %%
# gbt classifier in pySpark

gbt = GBTClassifier(maxIter=10, featuresCol="features", labelCol="recommendation")
gbtModel = gbt.fit(train_data)
predictions = gbtModel.transform(test_dataframe)
sort_predictions = predictions.select(
    "userID", "trackID", "probability", "rawPrediction", "prediction"
).sort(col("userID").asc(), col("probability").desc())

# %%
sort_predictions.show(6)
write_to_submission(sort_predictions)
