# %%
import pandas as pd
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession

# set python executable pyspark

# please download 'ratings.csv' from week 12 on canvas
spark = SparkSession.builder.appName("recommend-ML").getOrCreate()
sc = spark.sparkContext
sc.pythonExec = "python"
output = pd.read_csv(
    "data/output1.txt",
    header=None,
    sep="|",
    names=["userID", "trackID", "albumRating", "artistRating"],
)
# df.printSchema()
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
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler

numericCols = [
    "albumRating",
    "artistRating",
]  # TODO: Add Genres Later, 'num_genre_ratings', 'max', 'min', 'mean', 'variance', 'median']
stages = []
assemblerInputs = numericCols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]
# %%
# recommendation is 1 or 0, no need to index string
# label_stringIdx = StringIndexer(inputCol='recommendation', outputCol='label')
# stages += [label_stringIdx]
# %%
from pyspark.ml import Pipeline

pipeline = Pipeline(stages=stages)
pipelineModel = pipeline.fit(train_data)
train_data = pipelineModel.transform(train_data)
selectedCols = ["features"] + cols
train_data = train_data.select(selectedCols)

# # we split the last 76 users X 6 tracks = 468 records as the training
# # the first 50 users X 6 tracks = 300 records as the testing
# # make items where the userID is between 200596 and 201720 as training
# # make items where the userID is between 200031 and 200563 as testing
# from pyspark.sql.functions import col
#
# train = train_data.where(col("userID").between(200596, 201720))
# test = train_data.where(col("userID").between(200031, 200563))
# print("Training Dataset Count: " + str(train.count()))
# print("Test Dataset Count: " + str(test.count()))
# #
# # train_data.printSchema()
# # pd.DataFrame(train_data.take(5), columns=train_data.columns).transpose()
#
# # %%
# # logistic regression in pySpark
# from pyspark.ml.classification import LogisticRegression
#
# lr = LogisticRegression(featuresCol='features', labelCol='recommendation', maxIter=10)
# lrModel = lr.fit(train)


# %%
# predictions = lrModel.transform(test)
#
# sort_predictions = predictions.select('userID', 'trackID',
#                                       'recommendation', 'probability',
#                                       'rawPrediction', 'prediction'
#                                       ).sort(col("userID").asc(), col("probability").desc())
# sort_predictions.show(6)
# # %%
# # get boolean array of where recommendations column equals prediction column
# # bool_pred = predictions.select('recommendation', 'prediction').rdd.map(lambda row: row[0] == row[1]).collect()
# # %%
# # evaluate the model
# from pyspark.ml.evaluation import BinaryClassificationEvaluator
# evaluation = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="recommendation")
# print("The area under ROC for train set after 30 iterations is {}".format(evaluation.evaluate(predictions)))
# %%
from parsing import Path, TrackEntryMap, UserRatingHistoryMap, load_data

_, _, _, _, _, test_map = load_data()
output_for_df_training = Path("./data/testItem2.df")
with open(output_for_df_training, "w") as f:
    f.write("userID,trackID\n")
    for user_id, tracks in test_map.items():
        for track in tracks:
            f.write(f"{user_id},{track}\n")
# %%
from pyspark.sql.functions import col

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
from pyspark.ml.classification import DecisionTreeClassifier

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
from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(featuresCol="features", labelCol="recommendation")
rfModel = rf.fit(train_data)
predictions = rfModel.transform(test_dataframe)
sort_predictions = predictions.select(
    "userID", "trackID", "probability", "rawPrediction", "prediction"
).sort(col("userID").asc(), col("probability").desc())
# %%
# sort_predictions.show(6)
write_to_submission(sort_predictions)
# %%
# gbt classifier in pySpark
from pyspark.ml.classification import GBTClassifier

gbt = GBTClassifier(maxIter=10, featuresCol="features", labelCol="recommendation")
gbtModel = gbt.fit(train_data)
predictions = gbtModel.transform(test_dataframe)
sort_predictions = predictions.select(
    "userID", "trackID", "probability", "rawPrediction", "prediction"
).sort(col("userID").asc(), col("probability").desc())
# %%
# sort_predictions.show(6)
write_to_submission(sort_predictions)
# %%
