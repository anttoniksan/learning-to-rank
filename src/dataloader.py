from pathlib import Path
from typing import Any

import polars as pl
from polars import read_csv
from torch.utils.data import Dataset

user_info_columns = ["user_id", "age", "gender", "occupation", "imdb_id"]
user_info_categoricals = ["gender", "occupation"]

movie_info_columns = [
    "movie_id",
    "title",
    "release_date",
    "video_release_date",
    "imdb_url",
    "unknown",
    "action",
    "adventure",
    "animation",
    "childrens",
    "comedy",
    "crime",
    "documentary",
    "drama",
    "fantasy",
    "film-noir",
    "horror",
    "musical",
    "mystery",
    "romance",
    "scifi",
    "thriller",
    "war",
    "western",
]


class MovieLensDataset(Dataset):
    def _get_column_mapping(self, columns: list[str]) -> set[str]:
        return {f"column_{x + 1}": columns[x] for x in range(len(columns))}

    def _load_movie_info(self, movie_data: Path) -> dict:
        movie_info = (
            read_csv(movie_data, separator="|", has_header=False, encoding="utf8-lossy")
            .rename(self._get_column_mapping(movie_info_columns))
            .drop("imdb_url", "title", "video_release_date", "unknown")
        )

        movie_info = movie_info.with_columns(
            pl.col("release_date")
            .str.strptime(pl.Date, "%d-%b-%Y")
            .dt.year()
            .fill_null(0)
            .cast(pl.Int32)
        )

        return movie_info

    def _load_user_info(self, user_data: Path) -> dict:
        user_info = (
            read_csv(user_data, separator="|", has_header=False, encoding="utf8-lossy")
            .rename(self._get_column_mapping(user_info_columns))
            .drop("imdb_id")
        )

        for key in user_info_categoricals:
            user_info = user_info.with_columns(
                pl.col(key).cast(pl.Categorical).to_physical()
            )

        return user_info

    def _prepare_data(
        self, data_file: Path, include_metadata: bool = False
    ) -> tuple[list[Any], list[int]]:
        ratings_info = read_csv(data_file, separator="\t", has_header=False).rename(
            self._get_column_mapping(["user_id", "movie_id", "rating", "timestamp"])
        )

        ratings = ratings_info.select(pl.col("user_id"), pl.col("movie_id"))
        labels = ratings_info.select(pl.col("rating"))

        if include_metadata:
            ratings = ratings.join(
                self.user_info, left_on="user_id", right_on="user_id", how="left"
            )

            ratings = ratings.join(
                self.movie_info, left_on="movie_id", right_on="movie_id", how="left"
            )

        labels = labels.to_numpy() / 5.0  # Normalize ratings to [0, 1]
        return ratings.to_numpy().tolist(), labels.tolist()

    def __init__(
        self,
        data_file: Path,
        user_data: Path,
        movie_data: Path,
        include_metadata: bool = False,
    ):
        self.user_info = self._load_user_info(user_data)
        self.movie_info = self._load_movie_info(movie_data)

        data, labels = self._prepare_data(data_file, include_metadata=include_metadata)
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
