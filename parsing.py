#!/usr/bin/env python

import pickle
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

DATA_PATH = "data/"

TRAIN_FILENAME = Path(DATA_PATH, "trainItem2.txt")
TRAIN_CACHE_FILENAME = TRAIN_FILENAME.with_suffix(".pickle")

TEST_FILENAME = Path(DATA_PATH, "testItem2.txt")
TEST_CACHE_FILENAME = TEST_FILENAME.with_suffix(".pickle")

TRACK_FILENAME = Path(DATA_PATH, "trackData2.txt")
TRACK_CACHE_FILENAME = TRACK_FILENAME.with_suffix(".pickle")

ALBUM_FILENAME = Path(DATA_PATH, "albumData2.txt")
ALBUM_CACHE_FILENAME = ALBUM_FILENAME.with_suffix(".pickle")

ARTIST_FILENAME = Path(DATA_PATH, "artistData2.txt")
ARTIST_CACHE_FILENAME = ARTIST_FILENAME.with_suffix(".pickle")

GENRE_FILENAME = Path(DATA_PATH, "genreData2.txt")
GENRE_CACHE_FILENAME = GENRE_FILENAME.with_suffix(".pickle")

OUTPUT_FILENAME = Path(DATA_PATH, "output.txt")


class TrackEntry(NamedTuple):
    album_id: str | None
    artist_id: str | None
    genres: tuple[str, ...] | None


def parse_track_entry(s: str) -> tuple[str, TrackEntry]:
    split = s.strip().split("|")
    id = split[0]
    album = split[1]
    artist = split[2]

    return id, TrackEntry(
        # id,
        album if album != "None" else None,
        artist if artist != "None" else None,
        tuple(split[3:]) if len(split) > 3 else None,
    )


class AlbumEntry(NamedTuple):
    artist_id: str | None
    genres: tuple[str, ...] | None


def parse_album_entry(s: str) -> tuple[str, AlbumEntry]:
    split = s.strip().split("|")
    id = split[0]
    artist = split[1]

    return id, AlbumEntry(
        # id,
        artist if artist != "None" else None,
        tuple(split[2:]) if len(split) > 2 else None,
    )


def parse_user_rating_history(
    user_id_and_num_tracks_str: str,
    iter: Iterator[str],
) -> tuple[str, dict[str, int]]:
    user_id, num_tracks = user_id_and_num_tracks_str.split("|")
    num_tracks = int(num_tracks)

    unparsed_entries = [next(iter).split("\t") for _ in range(num_tracks)]
    tracks_to_ratings = {entry[0]: int(entry[1]) for entry in unparsed_entries}

    return user_id, tracks_to_ratings


def parse_test_case(
    user_id_and_num_tracks_str: str,
    iter: Iterator[str],
) -> tuple[str, tuple[str, ...]]:
    user_id, num_tracks = user_id_and_num_tracks_str.split("|")
    num_tracks = int(num_tracks)
    tracks = tuple(next(iter) for _ in range(num_tracks))
    return user_id, tracks


def cache_exists() -> bool:
    return (
        TRACK_CACHE_FILENAME.exists()
        and ALBUM_CACHE_FILENAME.exists()
        and GENRE_CACHE_FILENAME.exists()
        and ARTIST_CACHE_FILENAME.exists()
        and TRAIN_CACHE_FILENAME.exists()
        and TEST_CACHE_FILENAME.exists()
    )


TrackEntryMap = dict[str, TrackEntry]
AlbumEntryMap = dict[str, AlbumEntry]
ArtistSet = set[str]
GenreSet = set[str]
UserRatingHistoryMap = dict[str, dict[str, int]]
TestCaseMap = dict[str, tuple[str, ...]]


def load_data() -> (
    tuple[
        TrackEntryMap,
        AlbumEntryMap,
        ArtistSet,
        GenreSet,
        UserRatingHistoryMap,
        TestCaseMap,
    ]
):
    if not cache_exists():
        print("generating cache\n")
        track_data = TRACK_FILENAME.read_text()
        track_map = dict(parse_track_entry(line) for line in track_data.splitlines())
        track_pickle = pickle.dumps(track_map)
        TRACK_CACHE_FILENAME.write_bytes(track_pickle)

        album_data = ALBUM_FILENAME.read_text()
        album_map = dict(parse_album_entry(line) for line in album_data.splitlines())
        album_pickle = pickle.dumps(album_map)
        ALBUM_CACHE_FILENAME.write_bytes(album_pickle)

        artist_data = ARTIST_FILENAME.read_text()
        artist_set = set(artist_data.splitlines())
        artist_pickle = pickle.dumps(artist_set)
        ARTIST_CACHE_FILENAME.write_bytes(artist_pickle)

        genre_data = GENRE_FILENAME.read_text()
        genre_set = set(genre_data.splitlines())
        genre_pickle = pickle.dumps(genre_set)
        GENRE_CACHE_FILENAME.write_bytes(genre_pickle)

        train_data_iter = iter(TRAIN_FILENAME.read_text().splitlines())
        train_map = dict(
            parse_user_rating_history(line, train_data_iter) for line in train_data_iter
        )
        train_pickle = pickle.dumps(train_map)
        TRAIN_CACHE_FILENAME.write_bytes(train_pickle)

        test_data_iter = iter(TEST_FILENAME.read_text().splitlines())
        test_map = dict(
            parse_test_case(line, test_data_iter) for line in test_data_iter
        )
        test_pickle = pickle.dumps(test_map)
        TEST_CACHE_FILENAME.write_bytes(test_pickle)

    else:
        print("loading from cache\n")
        track_map = pickle.loads(TRACK_CACHE_FILENAME.read_bytes())
        album_map = pickle.loads(ALBUM_CACHE_FILENAME.read_bytes())
        artist_set = pickle.loads(ARTIST_CACHE_FILENAME.read_bytes())
        genre_set = pickle.loads(GENRE_CACHE_FILENAME.read_bytes())
        train_map = pickle.loads(TRAIN_CACHE_FILENAME.read_bytes())
        test_map = pickle.loads(TEST_CACHE_FILENAME.read_bytes())

    return track_map, album_map, artist_set, genre_set, train_map, test_map
