import parsing

# import numpy as np


def search(
    track: str,
    user: str,
    train_data_dict: dict[str, parsing.UserRatingHistory],
    track_data_dict: dict[str, parsing.TrackEntry],
    album_set: set[str],
    genre_set: set[str],
):
    # Get all data about the track, will need artist, album, and genre ratings
    track_data = track_data_dict[track]
    # Get all ratings that the user has, will be compared to track_data
    user_data = train_data_dict[user]
    # helper var to increase readability
    track_genres = track_data.genres
    # Used to calculate genre_rating_mean
    genres_rated_count = genres_rated_sum = 0
    # helper var to increase readability
    items_rated = user_data.items_to_ratings
    # for each genre, if it was rated, add it to the mean calculator
    if track_genres is not None:
        for track_genre in track_genres:
            if track_genre not in items_rated:
                continue
            genres_rated_count += 1
            genres_rated_sum += items_rated[track_genre]
    track_artist = track_data.artist_id
    # For each possible metric, if it was rated by this user, add them to the mean with the given weight
    weighted_rating_sum = 0
    weighted_rating_count = 0
    if genres_rated_count > 0:
        GENRE_WEIGHT = 2
        weighted_rating_sum += genres_rated_sum * GENRE_WEIGHT
        weighted_rating_count += genres_rated_count * GENRE_WEIGHT
    if track_artist in items_rated:
        ARTIST_WEIGHT = 1
        weighted_rating_sum += items_rated[track_artist] * ARTIST_WEIGHT
        weighted_rating_count += ARTIST_WEIGHT
    if track in items_rated:
        TRACK_WEIGHT = 20
        weighted_rating_sum += items_rated[track] * TRACK_WEIGHT
        weighted_rating_count += TRACK_WEIGHT
    if track_data.album_id in items_rated:
        ALBUM_WEIGHT = 5
        weighted_rating_sum += items_rated[track_data.album_id] * ALBUM_WEIGHT
        weighted_rating_sum += ALBUM_WEIGHT
    # return this weighted sum
    return (
        weighted_rating_sum / weighted_rating_count if weighted_rating_count > 0 else -1
    )


def main():
    (
        track_list,
        album_list,
        artist_list,
        genre_list,
        train_list,
        test_list,
    ) = parsing.load_lists()
    # testing what training data is TODO: Remove
    album_count = track_count = genre_count = artist_count = 0
    if False:
        for user_rating in train_list:
            for item_rated in user_rating.items_to_ratings:
                if item_rated in track_list:
                    track_count += 1
                    # print(f"{item_rated}: Track")
                elif item_rated in genre_list:
                    genre_count += 1
                    # print(f"{item_rated}: Genre")
                elif item_rated in album_list:
                    album_count += 1
                    # print(f"{item_rated}: Album")
                elif item_rated in artist_list:
                    artist_count += 1
                    # print(f"{item_rated}: Artist")
        print(
            f"{track_count} Tracks\n{genre_count} Genres\n{album_count} Albums\n{artist_count} Artists"
        )

    ratings = {}

    with open("results.csv", "w") as results_file:
        results_file.write("TrackID,Predictor\n")
        for test_datum in test_list:
            test_ratings = {track: 0 for track in test_datum.tracks}
            for track in test_datum.tracks:
                test_ratings[track] = search(
                    track,
                    test_datum.user_id,
                    train_list,
                    track_list,
                    album_list,
                    genre_list,
                )
            for i, track_rating in enumerate(
                sorted(test_ratings.items(), key=lambda x: x[1])
            ):
                if i < 3:
                    # ratings[test_datum.user_id + track_rating[0]] = 0
                    results_file.write(f"{test_datum.user_id}_{track_rating[0]},0")
                else:
                    # ratings[test_datum.user_id + track_rating[0]] = 1
                    results_file.write(f"{test_datum.user_id}_{track_rating[0]},1")
                results_file.write("\n")


if __name__ == "__main__":
    main()
