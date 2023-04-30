import os.path
import math

from parsing import (
    Path,
    TrackEntryMap,
    UserRatingHistoryMap,
    load_data,
)

genres_used = tracks_used = artists_used = albums_used = 0


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
        ALBUM_WEIGHT = 2
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
            GENRE_WEIGHT = 1 * 0
            weighted_rating_sum += user_rating_history[genre_id] * GENRE_WEIGHT
            weighted_rating_count += GENRE_WEIGHT
            total_numer_of_ratings += 1 * 0

    num_of_ratings_multiplier = math.sqrt(total_numer_of_ratings)

    return (
        num_of_ratings_multiplier * weighted_rating_sum / weighted_rating_count
        if weighted_rating_count > 0
        else -1
    )


def search_track_datad(
        track_id: str,
        user_id: str,
        training_map: UserRatingHistoryMap,
        track_map: TrackEntryMap) -> dict:
    album_id, artist_id, genres = track_map[track_id]
    user_rating_history = training_map[user_id]
    label = None

    return {
        "label": label,
        "features": features,
        "userID": user_id,
        "trackID": track_id,
        "recmmendation": "yes" if label == 1 else "no",
        "album": album_id,
        "artist": artist_id,
        "num_genre_ratings": len(genre_ratings) if genre_ratings is not None else 0,
        "max": max(genre_ratings) if genre_ratings is not None else 0,
        "min": min(genre_ratings) if genre_ratings is not None else 0,
        "mean": sum(genre_ratings) / len(genre_ratings) if genre_ratings is not None else 0,
        "variance": sum((x - mean) ** 2 for x in genre_ratings) / len(
            genre_ratings) if genre_ratings is not None else 0,
        "median": median(genre_ratings) if genre_ratings is not None else 0,
    }


def search_weighted_avg(
        track_id: str,
        user_id: str,
        training_map: UserRatingHistoryMap,
        track_map: TrackEntryMap,
        average_genres: bool = True
) -> float:
    global tracks_used, albums_used, artists_used, genres_used
    album_id, artist_id, genres = track_map[track_id]
    user_rating_history = training_map[user_id]

    weighted_rating_sum = 0
    weighted_rating_count = 0
    if track_id in user_rating_history:
        # Though we don't expect tracks, they should overpower
        TRACK_WEIGHT = 1000
        weighted_rating_sum += user_rating_history[track_id] * TRACK_WEIGHT
        weighted_rating_count += TRACK_WEIGHT
        tracks_used += 1
    elif album_id in user_rating_history:
        ALBUM_WEIGHT = 20
        weighted_rating_sum += user_rating_history[album_id] ** 2 * ALBUM_WEIGHT
        weighted_rating_count += ALBUM_WEIGHT
        albums_used += 1
    elif artist_id in user_rating_history:
        ARTIST_WEIGHT = 60
        weighted_rating_sum += user_rating_history[artist_id] * ARTIST_WEIGHT
        weighted_rating_count += ARTIST_WEIGHT
        artists_used += 1

    genres_rated_count = genres_rated_sum = 0
    GENRE_WEIGHT = 0.5 * 0
    if genres is not None:
        for genre_id in genres:
            if genre_id not in user_rating_history:
                continue
            genres_rated_sum += user_rating_history[genre_id]
            genres_rated_count += 1
            genres_used += 1
        if genres_rated_count > 0:
            if average_genres:
                weighted_rating_sum += (genres_rated_sum / genres_rated_count) * GENRE_WEIGHT
                weighted_rating_count += GENRE_WEIGHT
            else:
                weighted_rating_sum += genres_rated_sum * GENRE_WEIGHT
                weighted_rating_count += genres_rated_count * GENRE_WEIGHT

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
    global tracks_used, albums_used, artists_used, genres_used
    # (
    #     track_list,
    #     album_list,
    #     artist_list,
    #     genre_list,
    #     train_list,
    #     test_list,
    # ) = load_data()
    # testing what training data is TODO: Remove
    album_count = track_count = genre_count = artist_count = 0
    # if True:
    #     for user_rating in train_list.values():
    #         for item_rated in user_rating.items_to_ratings:
    #             if item_rated in track_list:
    #                 track_count += 1
    #                 # print(f"{item_rated}: Track")
    #             elif item_rated in genre_list:
    #                 genre_count += 1
    #                 # print(f"{item_rated}: Genre")
    #             elif item_rated in album_list:
    #                 album_count += 1
    #                 # print(f"{item_rated}: Album")
    #             elif item_rated in artist_list:
    #                 artist_count += 1
    #                 # print(f"{item_rated}: Artist")
    #     print(f"{track_count} Tracks\n{genre_count} Genres\n{album_count} Albums\n{artist_count} Artists")

    track_map, _, _, _, train_map, test_map = load_data()

    fileversion = 1
    while True:
        results_name = f"submissions/results-{fileversion:02}.csv"
        if not os.path.exists(results_name):
            break
        fileversion += 1
    with open(results_name, "w") as results_file:
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
    print(f"Tracks:{tracks_used}\nAlbums:{albums_used}\nArtists:{artists_used}\nGenres:{genres_used}")


if __name__ == "__main__":
    main()
