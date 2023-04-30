from pathlib import Path

import pandas as pd

TRACK_IDS = [
    line.split("|")[0]
    for line in Path("./data/trackData2.txt").read_text().splitlines()
]

ALBUM_IDS = [
    line.split("|")[0]
    for line in Path("./data/albumData2.txt").read_text().splitlines()
]

ARTIST_IDS = [line for line in Path("./data/artistData2.txt").read_text().splitlines()]

GENRE_IDS = [line for line in Path("./data/genreData2.txt").read_text().splitlines()]

user_item_matrix = pd.DataFrame(
    columns=["user_id"] + TRACK_IDS + ALBUM_IDS + ARTIST_IDS + GENRE_IDS
)

it = iter(Path("./data/trainItem2.txt").read_text().splitlines())
for user_id_and_num_tracks in it:
    user_id, num_tracks = user_id_and_num_tracks.split("|")
    num_tracks = int(num_tracks)

    unparsed_entries = [next(it).split("\t") for _ in range(num_tracks)]
    entries_to_ratings = {entry[0]: [int(entry[1])] for entry in unparsed_entries}
    entries_to_ratings["user_id"] = user_id

    df = pd.DataFrame.from_dict(entries_to_ratings)
    user_item_matrix = pd.concat([user_item_matrix, df])


user_item_matrix = user_item_matrix.replace("NaN", 0)
print(user_item_matrix)
