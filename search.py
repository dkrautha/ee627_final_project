from parsing import (
    Path,
    TrackEntryMap,
    UserRatingHistoryMap,
    load_data,
)


def search_weighted_avg_and_num_matches(
    track_id: str,
    user_id: str,
    training_map: UserRatingHistoryMap,
    track_map: TrackEntryMap,
) -> float:
    album_id, artist_id, genres = track_map[track_id]
    user_rating_history = training_map[user_id]

    weighted_rating_sum = 0
    weighted_rating_count = 0
    total_numer_of_ratings = 0
    if track_id in user_rating_history:
        TRACK_WEIGHT = 10
        weighted_rating_sum += user_rating_history[track_id] * TRACK_WEIGHT
        weighted_rating_count += TRACK_WEIGHT
        total_numer_of_ratings += 1
    elif album_id in user_rating_history:
        ALBUM_WEIGHT = 7
        weighted_rating_sum += user_rating_history[album_id] * ALBUM_WEIGHT
        weighted_rating_count += ALBUM_WEIGHT
        total_numer_of_ratings += 1
    elif artist_id in user_rating_history:
        ARTIST_WEIGHT = 3
        weighted_rating_sum += user_rating_history[artist_id] * ARTIST_WEIGHT
        weighted_rating_count += ARTIST_WEIGHT
        total_numer_of_ratings += 1

    if genres is not None:
        for genre_id in genres:
            if genre_id not in user_rating_history:
                continue
            GENRE_WEIGHT = 1
            weighted_rating_sum += user_rating_history[genre_id] * GENRE_WEIGHT
            weighted_rating_count += GENRE_WEIGHT
            total_numer_of_ratings += 1

    num_of_ratings_multiplier = 0.6 * total_numer_of_ratings

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
) -> float:
    album_id, artist_id, genres = track_map[track_id]
    user_rating_history = training_map[user_id]

    weighted_rating_sum = 0
    weighted_rating_count = 0
    if track_id in user_rating_history:
        TRACK_WEIGHT = 10
        weighted_rating_sum += user_rating_history[track_id] * TRACK_WEIGHT
        weighted_rating_count += TRACK_WEIGHT
    elif album_id in user_rating_history:
        ALBUM_WEIGHT = 7
        weighted_rating_sum += user_rating_history[album_id] * ALBUM_WEIGHT
        weighted_rating_count += ALBUM_WEIGHT
    elif artist_id in user_rating_history:
        ARTIST_WEIGHT = 3
        weighted_rating_sum += user_rating_history[artist_id] * ARTIST_WEIGHT
        weighted_rating_count += ARTIST_WEIGHT

    if genres is not None:
        for genre_id in genres:
            if genre_id not in user_rating_history:
                continue
            GENRE_WEIGHT = 1
            weighted_rating_sum += user_rating_history[genre_id] * GENRE_WEIGHT
            weighted_rating_count += GENRE_WEIGHT

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

    sum = 0
    if track_id in user_rating_history:
        sum += user_rating_history[track_id]
    elif album_id in user_rating_history:
        sum += user_rating_history[album_id]
    elif artist_id in user_rating_history:
        sum += user_rating_history[artist_id]

    if genres is not None:
        for genre_id in genres:
            if genre_id not in user_rating_history:
                continue
            sum += user_rating_history[genre_id]

    return sum


def search_average(
    track_id: str,
    user_id: str,
    training_map: UserRatingHistoryMap,
    track_map: TrackEntryMap,
) -> float:
    album_id, artist_id, genres = track_map[track_id]
    user_rating_history = training_map[user_id]

    sum = 0
    num_items = 0
    if track_id in user_rating_history:
        sum += user_rating_history[track_id]
        num_items += 1
    elif album_id in user_rating_history:
        sum += user_rating_history[album_id]
        num_items += 1
    elif artist_id in user_rating_history:
        sum += user_rating_history[artist_id]
        num_items += 1

    if genres is not None:
        for genre_id in genres:
            if genre_id not in user_rating_history:
                continue
            sum += user_rating_history[genre_id]
            num_items += 1

    return sum / num_items if num_items > 0 else -1


def fun_data_statistics(track_map, album_map, artist_set, genre_set, train_map):
    print("Data Available")
    print(f"Tracks: {len(track_map)}")
    print(f"Albums: {len(album_map)}")
    print(f"Artists: {len(artist_set)}")
    print(f"Genres: {len(genre_set)}\n")

    print("Training Data Statistics")
    times_tracks_were_rated = 0
    times_albums_were_rated = 0
    times_artists_were_rated = 0
    times_genres_were_rated = 0
    for _, ratings in train_map.items():
        for item_id, _ in ratings.items():
            if item_id in track_map:
                times_tracks_were_rated += 1
            elif item_id in album_map:
                times_albums_were_rated += 1
            elif item_id in artist_set:
                times_artists_were_rated += 1
            elif item_id in genre_set:
                times_genres_were_rated += 1
            else:
                print(f"Unable to find {item_id} in tracks, albums, artists, or genres")
                return

    print(f"Times Tracks Were Rated: {times_tracks_were_rated}")
    print(f"Times Albums Were Rated: {times_albums_were_rated}")
    print(f"Times Artists Were Rated: {times_artists_were_rated}")
    print(f"Times Genres Were Rated: {times_genres_were_rated}")
    total_ratings = (
        times_tracks_were_rated
        + times_albums_were_rated
        + times_artists_were_rated
        + times_genres_were_rated
    )
    print(f"Total Ratings In System: {total_ratings} ")


def main():
    RESULTS_PATH = Path("results.csv")

    track_map, _, _, _, train_map, test_map = load_data()

    with open(RESULTS_PATH, "w") as results_file:
        results_file.write("TrackID,Predictor\n")
        for user_id, tracks in test_map.items():
            test_ratings = {track: 0.0 for track in tracks}
            for track_id in tracks:
                test_ratings[track_id] = search_weighted_avg_and_num_matches(
                    track_id, user_id, train_map, track_map
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
