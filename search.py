import math
import os.path

from parsing import TrackEntryMap, UserRatingHistoryMap, load_data


def search_weighted_avg_and_num_matches(
    track_id: str,
    user_id: str,
    training_map: UserRatingHistoryMap,
    track_map: TrackEntryMap,
) -> float:
    album_id, artist_id, genres = track_map[track_id]
    user_rating_history = training_map[user_id]

    track_weight = 10
    album_weight = 2
    artist_weight = 3
    genre_weight = 1

    weighted_rating_sum = 0
    weighted_rating_count = 0
    total_nubmer_of_ratings = 0
    if track_id in user_rating_history:
        weighted_rating_sum += user_rating_history[track_id] * track_weight
        weighted_rating_count += track_weight
        total_nubmer_of_ratings += 1
    if album_id in user_rating_history:
        weighted_rating_sum += user_rating_history[album_id] * album_weight
        weighted_rating_count += album_weight
        total_nubmer_of_ratings += 1
    if artist_id in user_rating_history:
        weighted_rating_sum += user_rating_history[artist_id] * artist_weight
        weighted_rating_count += artist_weight
        total_nubmer_of_ratings += 1

    if genres is not None:
        for genre_id in genres:
            if genre_id not in user_rating_history:
                continue
            weighted_rating_sum += user_rating_history[genre_id] * genre_weight
            weighted_rating_count += genre_weight
            total_nubmer_of_ratings += 1

    num_of_ratings_multiplier = math.sqrt(total_nubmer_of_ratings)

    return (
        num_of_ratings_multiplier * weighted_rating_sum / weighted_rating_count
        if weighted_rating_count > 0
        else -1
    )


def search_weighted_avg(
    track_id: str,
    user_id: str,
    training_map: UserRatingHistoryMap,
    track_map: TrackEntryMap,
    average_genres: bool = True,
) -> float:
    album_id, artist_id, genres = track_map[track_id]
    user_rating_history = training_map[user_id]

    track_weight = 1000
    album_weight = 20
    artist_weight = 60
    genre_weight = 0.5

    weighted_rating_sum = 0
    weighted_rating_count = 0
    if track_id in user_rating_history:
        weighted_rating_sum += user_rating_history[track_id] * track_weight
        weighted_rating_count += track_weight
    if album_id in user_rating_history:
        weighted_rating_sum += user_rating_history[album_id] ** 2 * album_weight
        weighted_rating_count += album_weight
    if artist_id in user_rating_history:
        weighted_rating_sum += user_rating_history[artist_id] * artist_weight
        weighted_rating_count += artist_weight

    genres_rated_count = 0
    genres_rated_sum = 0
    if genres is not None:
        for genre_id in genres:
            if genre_id not in user_rating_history:
                continue
            genres_rated_sum += user_rating_history[genre_id]
            genres_rated_count += 1
        if genres_rated_count > 0:
            if average_genres:
                weighted_rating_sum += (
                    genres_rated_sum / genres_rated_count
                ) * genre_weight
                weighted_rating_count += genre_weight
            else:
                weighted_rating_sum += genres_rated_sum * genre_weight
                weighted_rating_count += genres_rated_count * genre_weight

    return (
        weighted_rating_sum / weighted_rating_count if weighted_rating_count > 0 else -1
    )


def search_total_sum(
    track_id: str,
    user_id: str,
    training_map: UserRatingHistoryMap,
    track_map: TrackEntryMap,
) -> float:
    album_id, artist_id, genres = track_map[track_id]
    user_rating_history = training_map[user_id]

    rating_sum = 0
    if track_id in user_rating_history:
        rating_sum += user_rating_history[track_id]
    if album_id in user_rating_history:
        rating_sum += user_rating_history[album_id]
    if artist_id in user_rating_history:
        rating_sum += user_rating_history[artist_id]

    if genres is not None:
        for genre_id in genres:
            if genre_id not in user_rating_history:
                continue
            rating_sum += user_rating_history[genre_id]

    return rating_sum


def search_average(
    track_id: str,
    user_id: str,
    training_map: UserRatingHistoryMap,
    track_map: TrackEntryMap,
) -> float:
    album_id, artist_id, genres = track_map[track_id]
    user_rating_history = training_map[user_id]

    rating_sum = 0
    num_items = 0
    if track_id in user_rating_history:
        rating_sum += user_rating_history[track_id]
        num_items += 1
    if album_id in user_rating_history:
        rating_sum += user_rating_history[album_id]
        num_items += 1
    if artist_id in user_rating_history:
        rating_sum += user_rating_history[artist_id]
        num_items += 1

    if genres is not None:
        for genre_id in genres:
            if genre_id not in user_rating_history:
                continue
            rating_sum += user_rating_history[genre_id]
            num_items += 1

    return rating_sum / num_items if num_items > 0 else -1


def main():
    track_map, _, _, _, train_map, test_map = load_data()

    fileversion = 1
    while True:
        results_name = f"submissions/results-{fileversion:02}.csv"
        if not os.path.exists(results_name):
            break
        fileversion += 1
    with open(results_name, "w", encoding="utf-8") as results_file:
        results_file.write("TrackID,Predictor\n")
        for user_id, tracks in test_map.items():
            test_ratings = {track: 0.0 for track in tracks}
            for track_id in tracks:
                test_ratings[track_id] = search_weighted_avg(
                    track_id, user_id, train_map, track_map, True
                )

            sorted_test_ratings = sorted(test_ratings.items(), key=lambda x: x[1])
            for i, track_rating in enumerate(sorted_test_ratings):
                if i < 3:
                    results_file.write(f"{user_id}_{track_rating[0]},0")
                else:
                    results_file.write(f"{user_id}_{track_rating[0]},1")
                results_file.write("\n")


if __name__ == "__main__":
    main()
