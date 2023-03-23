#!/usr/bin/env python

from pathlib import Path
from pprint import pprint
from typing import NamedTuple

import numpy

train_filename = Path("./trainItem2.txt")
test_filename = Path("./testItem2.txt")

track_filename = Path("./trackData2.txt")
album_filename = Path("./albumData2.txt")
artist_filename = Path("./artistData2.txt")
genre_filename = Path("./genreData2.txt")

output_filename = Path("./output.txt")


class TrackEntry(NamedTuple):
    track_id: str
    album_id: str | None
    artist_id: str | None
    genres: list[str] | None


def parse_track_entry(s: str) -> TrackEntry:
    split = s.strip().split("|")
    id = split[0]
    album = split[1]
    artist = split[2]

    return TrackEntry(
        id,
        album if album != "None" else None,
        artist if artist != "None" else None,
        split[3:] if len(split) > 3 else None,
    )


class AlbumEntry(NamedTuple):
    album_id: str
    artist_id: str | None
    genres: list[str] | None


def parse_album_entry(s: str) -> AlbumEntry:
    split = s.strip().split("|")
    id = split[0]
    artist = split[1]

    return AlbumEntry(
        id,
        artist if artist != "None" else None,
        split[2:] if len(split) > 2 else None,
    )


class UserRatingInfo(NamedTuple):
    user_id: str
    ratings: dict[str, str]


def parse_user_rating_info(s: str) -> UserRatingInfo:
    ...


def main():
    with open(track_filename) as track_data:
        track_list = [parse_track_entry(line) for line in track_data]

    with open(album_filename) as album_data:
        album_list = [parse_album_entry(line) for line in album_data]

    with open(artist_filename) as artist_data:
        artist_list = [line.strip() for line in artist_data]

    with open(genre_filename) as genre_data:
        _genre_list = [line.strip() for line in genre_data]

    # this is not complete, but gets most of the way to parsing a UserRatingInfo
    with open(train_filename) as train_data:
        for line in train_data:
            split = line.strip().split("|")
            user_id = split[0]
            num_entries = int(split[1])

            entries = [
                train_data.readline().strip().split("\t") for _ in range(num_entries)
            ]
            pairs = {id: int(rating) for (id, rating) in entries}
            pprint(pairs)


if __name__ == "__main__":
    main()
