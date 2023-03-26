#!/usr/bin/env python

import pickle
from pathlib import Path
from typing import NamedTuple, Iterator

TRAIN_FILENAME = Path("./trainItem2.txt")
TRAIN_CACHE_FILENAME = TRAIN_FILENAME.with_suffix(".pickle")

TEST_FILENAME = Path("./testItem2.txt")
TEST_CACHE_FILENAME = TEST_FILENAME.with_suffix(".pickle")

TRACK_FILENAME = Path("./trackData2.txt")
TRACK_CACHE_FILENAME = TRACK_FILENAME.with_suffix(".pickle")

ALBUM_FILENAME = Path("./albumData2.txt")
ALBUM_CACHE_FILENAME = ALBUM_FILENAME.with_suffix(".pickle")

ARTIST_FILENAME = Path("./artistData2.txt")
ARTIST_CACHE_FILENAME = ARTIST_FILENAME.with_suffix(".pickle")

GENRE_FILENAME = Path("./genreData2.txt")
GENRE_CACHE_FILENAME = GENRE_FILENAME.with_suffix(".pickle")

OUTPUT_FILENAME = Path("./output.txt")


class TrackEntry(NamedTuple):
    track_id: str
    album_id: str | None
    artist_id: str | None
    genres: tuple[str] | None


def parse_track_entry(s: str) -> TrackEntry:
    split = s.strip().split("|")
    id = split[0]
    album = split[1]
    artist = split[2]

    return TrackEntry(
        id,
        album if album != "None" else None,
        artist if artist != "None" else None,
        tuple(split[3:]) if len(split) > 3 else None,
    )


class AlbumEntry(NamedTuple):
    album_id: str
    artist_id: str | None
    genres: tuple[str] | None


def parse_album_entry(s: str) -> AlbumEntry:
    split = s.strip().split("|")
    id = split[0]
    artist = split[1]

    return AlbumEntry(
        id,
        artist if artist != "None" else None,
        tuple(split[2:]) if len(split) > 2 else None,
    )


class UserRatingHistory(NamedTuple):
    user_id: str
    tracks_to_ratings: dict[str, int]

    def __hash__(self) -> int:
        return hash(self.user_id)

    def __eq__(self, __value: object) -> bool:
        if type(__value) != type(self):
            raise NotImplementedError
        return self.user_id == __value.user_id


def parse_user_rating_history(
    user_id_and_num_tracks_str: str, iter: Iterator[str]
) -> UserRatingHistory:
    user_id, num_tracks = user_id_and_num_tracks_str.split("|")
    num_tracks = int(num_tracks)

    unparsed_entries = [next(iter).split("\t") for _ in range(num_tracks)]
    tracks_to_ratings = {entry[0]: int(entry[1]) for entry in unparsed_entries}

    return UserRatingHistory(user_id, tracks_to_ratings)


class TestCase(NamedTuple):
    user_id: str
    tracks: tuple[str]


def parse_test_case(user_id_and_num_tracks_str: str, iter: Iterator[str]):
    user_id, num_tracks = user_id_and_num_tracks_str.split("|")
    num_tracks = int(num_tracks)
    tracks = tuple(next(iter) for _ in range(num_tracks))
    return TestCase(user_id, tracks)


def cache_exists() -> bool:
    return (
        TRACK_CACHE_FILENAME.exists()
        and ALBUM_CACHE_FILENAME.exists()
        and GENRE_CACHE_FILENAME.exists()
        and ARTIST_CACHE_FILENAME.exists()
        and TRAIN_CACHE_FILENAME.exists()
        and TEST_CACHE_FILENAME.exists()
    )


def load_lists() -> (
    tuple[
        set[TrackEntry],
        set[AlbumEntry],
        set[str],
        set[str],
        set[UserRatingHistory],
        set[TestCase],
    ]
):
    if not cache_exists():
        print("generating cache")
        track_data = TRACK_FILENAME.read_text()
        track_list = set(parse_track_entry(line) for line in track_data.splitlines())
        track_pickle = pickle.dumps(track_list)
        TRACK_CACHE_FILENAME.write_bytes(track_pickle)

        album_data = ALBUM_FILENAME.read_text()
        album_list = set(parse_album_entry(line) for line in album_data.splitlines())
        album_pickle = pickle.dumps(album_list)
        ALBUM_CACHE_FILENAME.write_bytes(album_pickle)

        artist_data = ARTIST_FILENAME.read_text()
        artist_list = set(artist_data.splitlines())
        artist_pickle = pickle.dumps(artist_list)
        ARTIST_CACHE_FILENAME.write_bytes(artist_pickle)

        genre_data = GENRE_FILENAME.read_text()
        genre_list = set(genre_data.splitlines())
        genre_pickle = pickle.dumps(genre_list)
        GENRE_CACHE_FILENAME.write_bytes(genre_pickle)

        train_data_iter = iter(TRAIN_FILENAME.read_text().splitlines())
        train_list = set(
            parse_user_rating_history(line, train_data_iter) for line in train_data_iter
        )
        train_pickle = pickle.dumps(train_list)
        TRAIN_CACHE_FILENAME.write_bytes(train_pickle)

        test_data_iter = iter(TEST_FILENAME.read_text().splitlines())
        test_list = set(
            parse_test_case(line, test_data_iter) for line in test_data_iter
        )
        test_pickle = pickle.dumps(test_list)
        TEST_CACHE_FILENAME.write_bytes(test_pickle)

    else:
        print("loading from cache")
        track_list: set[TrackEntry] = pickle.loads(TRACK_CACHE_FILENAME.read_bytes())
        album_list: set[AlbumEntry] = pickle.loads(ALBUM_CACHE_FILENAME.read_bytes())
        artist_list: set[str] = pickle.loads(ARTIST_CACHE_FILENAME.read_bytes())
        genre_list: set[str] = pickle.loads(GENRE_CACHE_FILENAME.read_bytes())
        train_list: set[UserRatingHistory] = pickle.loads(
            TRAIN_CACHE_FILENAME.read_bytes()
        )
        test_list: set[TestCase] = pickle.loads(TEST_CACHE_FILENAME.read_bytes())

    return track_list, album_list, artist_list, genre_list, train_list, test_list


def main():
    (
        track_list,
        album_list,
        artist_list,
        genre_list,
        train_list,
        test_list,
    ) = load_lists()

    print(next(iter(track_list)))
    print(next(iter(album_list)))
    print(next(iter(artist_list)))
    print(next(iter(genre_list)))
    print(next(iter(train_list)))
    print(next(iter(test_list)))

    assert UserRatingHistory("240545", {}) in train_list


if __name__ == "__main__":
    main()
