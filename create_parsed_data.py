import parsing


def main():
    (
        track_list,
        album_list,
        artist_list,
        genre_list,
        train_list,
        test_list,
    ) = parsing.load_lists()

    print(next(iter(track_list)))
    print(next(iter(album_list)))
    print(next(iter(artist_list)))
    print(next(iter(genre_list)))
    print(next(iter(train_list)))
    print(next(iter(test_list)))

    assert parsing.UserRatingHistory("240545", {}) in train_list


if __name__ == "__main__":
    main()
