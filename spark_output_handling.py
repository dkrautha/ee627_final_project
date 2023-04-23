import pandas as pd


def main() -> None:
    spark_data = pd.read_csv("./myprediction.csv").drop("Unnamed: 0", axis=1)

    user_predictions: dict[int, list[tuple[int, float]]] = {}

    for _, row in spark_data.iterrows():
        user_id = int(row["userID"])
        item_id = int(row["itemID"])
        prediction = float(row["prediction"])
        lst = user_predictions.get(user_id, [])
        lst.append((item_id, prediction))
        user_predictions[user_id] = lst

    for prediction_list in user_predictions.values():
        prediction_list.sort(key=lambda x: x[1], reverse=True)

    with open("results.csv", "w") as results:
        results.write("TrackID,Predictor\n")
        for user_id, predictions in user_predictions.items():
            missing_predictions = len(predictions) != 6
            if missing_predictions:
                print(f"{user_id} is missing predictions")
            for i, track_rating in enumerate(predictions):
                if missing_predictions:
                    print(f"track_id: {track_rating[0]}")
                if i < 3:
                    results.write(f"{user_id}_{track_rating[0]},1")
                else:
                    results.write(f"{user_id}_{track_rating[0]},0")
                results.write("\n")


if __name__ == "__main__":
    main()
