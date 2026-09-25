"""Download, carga e preparação do dataset MovieLens (ml-latest-small)."""

import io
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
DATASET_DIR = RAW_DIR / "ml-latest-small"
USER_RATINGS_FILE = ROOT / "data" / "user_ratings.csv"
URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

MIN_RATINGS_PER_MOVIE = 5


def download_movielens() -> Path:
    """Baixa e extrai o dataset para data/raw/ caso ainda não exista."""
    if (DATASET_DIR / "ratings.csv").exists():
        return DATASET_DIR
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Baixando {URL} ...")
    resp = requests.get(URL, timeout=120)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(RAW_DIR)
    return DATASET_DIR


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retorna (ratings, movies) do MovieLens.

    As avaliações feitas pela interface ficam em data/user_ratings.csv e NÃO entram no
    treino: são aplicadas por fold-in no momento da recomendação (ver models.py).
    """
    download_movielens()
    ratings = pd.read_csv(DATASET_DIR / "ratings.csv")
    movies = pd.read_csv(DATASET_DIR / "movies.csv")
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)\s*$").astype(float)
    return ratings, movies


def filter_ratings(ratings: pd.DataFrame, min_per_movie: int = MIN_RATINGS_PER_MOVIE) -> pd.DataFrame:
    """Remove filmes com poucas avaliações (ruído e esparsidade extrema)."""
    counts = ratings["movieId"].value_counts()
    keep = counts[counts >= min_per_movie].index
    return ratings[ratings["movieId"].isin(keep)].reset_index(drop=True)


@dataclass
class Dataset:
    """Matriz usuário×item e os mapeamentos entre IDs originais e índices."""

    R: csr_matrix
    user_ids: np.ndarray  # índice -> userId
    item_ids: np.ndarray  # índice -> movieId
    user_index: dict
    item_index: dict

    @classmethod
    def from_ratings(cls, ratings: pd.DataFrame, item_ids: np.ndarray | None = None) -> "Dataset":
        user_ids = np.sort(ratings["userId"].unique())
        if item_ids is None:
            item_ids = np.sort(ratings["movieId"].unique())
        user_index = {u: i for i, u in enumerate(user_ids)}
        item_index = {m: i for i, m in enumerate(item_ids)}
        r = ratings[ratings["movieId"].isin(item_index)]
        rows = r["userId"].map(user_index).to_numpy()
        cols = r["movieId"].map(item_index).to_numpy()
        R = csr_matrix(
            (r["rating"].to_numpy(dtype=np.float32), (rows, cols)),
            shape=(len(user_ids), len(item_ids)),
        )
        return cls(R, user_ids, item_ids, user_index, item_index)

    def user_vector(self, ratings_by_movie: dict) -> csr_matrix:
        """Monta o vetor esparso (1×n_itens) de um usuário a partir de {movieId: nota}."""
        cols = [self.item_index[m] for m in ratings_by_movie if m in self.item_index]
        vals = [ratings_by_movie[m] for m in ratings_by_movie if m in self.item_index]
        return csr_matrix(
            (np.array(vals, dtype=np.float32), (np.zeros(len(cols), dtype=int), cols)),
            shape=(1, len(self.item_ids)),
        )


def dataset_stats(ratings: pd.DataFrame) -> dict:
    n_users = ratings["userId"].nunique()
    n_items = ratings["movieId"].nunique()
    n = len(ratings)
    per_user = ratings.groupby("userId").size()
    return {
        "usuarios": n_users,
        "filmes": n_items,
        "avaliacoes": n,
        "esparsidade": 1 - n / (n_users * n_items),
        "nota_media": ratings["rating"].mean(),
        "min_aval_por_usuario": int(per_user.min()),
        "mediana_aval_por_usuario": float(per_user.median()),
    }


def save_user_rating(user_id: int, movie_id: int, rating: float) -> None:
    """Persiste uma avaliação feita pela interface em data/user_ratings.csv."""
    USER_RATINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame(
        [{"userId": user_id, "movieId": movie_id, "rating": rating, "timestamp": int(time.time())}]
    )
    row.to_csv(USER_RATINGS_FILE, mode="a", header=not USER_RATINGS_FILE.exists(), index=False)


def load_user_ratings(user_id: int) -> dict:
    """Avaliações feitas pela interface para um usuário: {movieId: nota} (a mais recente vale)."""
    if not USER_RATINGS_FILE.exists():
        return {}
    df = pd.read_csv(USER_RATINGS_FILE)
    df = df[df["userId"] == user_id].drop_duplicates("movieId", keep="last")
    return dict(zip(df["movieId"].astype(int), df["rating"].astype(float)))


def clear_user_ratings(user_id: int) -> None:
    if not USER_RATINGS_FILE.exists():
        return
    df = pd.read_csv(USER_RATINGS_FILE)
    df[df["userId"] != user_id].to_csv(USER_RATINGS_FILE, index=False)
