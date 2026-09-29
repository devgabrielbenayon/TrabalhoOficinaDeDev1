"""Download, carga e preparação dos dados: FBref (métricas) + Transfermarkt (interações).

* FBref Big-5 2024/25 (Kaggle)      -> vetor de métricas de cada jogador (conteúdo)
* Transfermarkt (Kaggle)            -> clube × jogador (minutos jogados) = interações da
                                       filtragem colaborativa; posição detalhada; valor
Os resultados intermediários ficam em cache em data/processed/.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src import config as C
from src.matching import match_players

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
SHORTLIST_FILE = ROOT / "data" / "shortlists.csv"

# colunas de contagem do FBref que serão usadas (somadas quando o jogador tem 2 clubes)
FBREF_COUNTS = [
    "MP", "Starts", "Min", "Gls", "Ast", "xG", "npxG", "xAG", "PrgC", "PrgP", "PrgR",
    "Sh", "SoT", "Cmp", "Att", "PrgDist", "KP", "1/3", "PPA", "CrsPA", "SCA", "Tkl",
    "TklW", "Int", "Clr", "Att Pen", "Succ", "Recov", "Won", "AerLost", "Blocks",
]


# ---------------------------------------------------------------- download
def dataset_path(slug: str) -> Path:
    """Pasta do dataset: data/raw/<nome> se baixado manualmente, senão via kagglehub."""
    manual = RAW_DIR / slug.split("/")[1]
    if manual.exists():
        return manual
    import kagglehub

    return Path(kagglehub.dataset_download(slug))


# ---------------------------------------------------------------- FBref
def load_fbref() -> pd.DataFrame:
    """Uma linha por jogador (quem trocou de clube na temporada tem as linhas somadas)."""
    df = pd.read_csv(dataset_path(C.FBREF_DATASET) / C.FBREF_FILE)
    # no CSV, "Blocks" é da tabela de tipos de passe (passes bloqueados); os bloqueios
    # defensivos estão em "Blocks_stats_defense"
    df = df.drop(columns=["Blocks"]).rename(
        columns={"Blocks_stats_defense": "Blocks", "Lost_stats_misc": "AerLost"}
    )
    df = df[df["Born"].notna() & (df["Min"] > 0)]
    df["Born"] = df["Born"].astype(int)
    main = df.sort_values("Min", ascending=False).drop_duplicates(["Player", "Born"])
    sums = df.groupby(["Player", "Born"])[FBREF_COUNTS].sum().reset_index()
    squads = df.groupby(["Player", "Born"])["Squad"].agg(lambda s: " / ".join(s)).rename("Squads")
    out = (
        main[["Player", "Born", "Nation", "Pos", "Squad", "Comp", "Age"]]
        .merge(sums, on=["Player", "Born"])
        .merge(squads, on=["Player", "Born"])
    )
    out["AerAtt"] = out["Won"] + out["AerLost"]
    out["Nation"] = out["Nation"].str.split().str[-1]
    return out.reset_index(drop=True)


# ---------------------------------------------------------------- Transfermarkt
def load_transfermarkt() -> dict[str, pd.DataFrame]:
    p = dataset_path(C.TM_DATASET)
    tm = {
        "players": pd.read_csv(p / "players.csv", parse_dates=["date_of_birth"]),
        "clubs": pd.read_csv(p / "clubs.csv", usecols=["club_id", "name", "domestic_competition_id"]),
        "transfers": pd.read_csv(p / "transfers.csv", parse_dates=["transfer_date"]),
        "valuations": pd.read_csv(p / "player_valuations.csv", parse_dates=["date"]),
    }
    cache = PROCESSED_DIR / "appearances.pkl"
    if cache.exists():
        tm["appearances"] = pd.read_pickle(cache)
    else:
        app = pd.read_csv(
            p / "appearances.csv",
            usecols=["player_id", "player_club_id", "date", "competition_id", "minutes_played"],
            parse_dates=["date"],
        )
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        app.to_pickle(cache)
        tm["appearances"] = app
    return tm


def _tm_candidates(tm: dict) -> pd.DataFrame:
    """Jogadores do Transfermarkt que atuaram nas 5 grandes ligas em 2024/25."""
    league_of = {v: k for k, v in C.LEAGUES.items()}
    app = tm["appearances"]
    season = app[(app.date >= "2024-08-01") & (app.date < "2025-07-01")
                 & app.competition_id.isin(league_of)]
    main = (season.groupby(["player_id", "player_club_id", "competition_id"])["minutes_played"]
            .sum().reset_index().sort_values("minutes_played", ascending=False)
            .drop_duplicates("player_id"))
    cand = main.merge(tm["players"][["player_id", "name", "date_of_birth"]], on="player_id")
    cand["birth_year"] = cand["date_of_birth"].dt.year
    cand["league"] = cand["competition_id"].map(league_of)
    return cand.rename(columns={"player_club_id": "club_id"})


def value_at(valuations: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
    """Último valor de mercado de cada jogador ANTES de `date` (evita vazamento)."""
    v = valuations[valuations.date < date].sort_values("date")
    return v.drop_duplicates("player_id", keep="last").set_index("player_id")["market_value_in_eur"]


def build_players(force: bool = False) -> pd.DataFrame:
    """Tabela de jogadores: métricas FBref + id/posição/valor do Transfermarkt."""
    cache = PROCESSED_DIR / "players.pkl"
    if cache.exists() and not force:
        return pd.read_pickle(cache)
    fb = load_fbref()
    tm = load_transfermarkt()
    cand = _tm_candidates(tm)
    fb = match_players(fb, cand)
    info = tm["players"].set_index("player_id")
    fb["player_id"] = fb["player_id"].astype("Int64")
    ok = fb.player_id.notna()
    fb.loc[ok, "sub_position"] = fb.loc[ok, "player_id"].map(info["sub_position"]).values
    # clube (Transfermarkt) em que o jogador mais atuou em 2024/25
    fb["club_id"] = fb["player_id"].map(cand.set_index("player_id")["club_id"]).astype("Int64")
    fb["group"] = fb["sub_position"].map(C.POSITION_GROUPS)
    fb.loc[ok, "value_eur"] = fb.loc[ok, "player_id"].map(value_at(tm["valuations"], C.SPLIT_DATE)).values
    fb["league"] = fb["Comp"].map(C.LEAGUE_LABEL)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    fb.to_pickle(cache)
    return fb


# ---------------------------------------------------------------- interações
def build_interactions(appearances: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Clube × jogador com peso implícito, usando só aparições ANTES de `cutoff`.

    peso = log(1 + jogos completos) · 0,8^(anos desde a última aparição)
    """
    app = appearances[(appearances.date < cutoff) & (appearances.minutes_played > 0)]
    inter = (app.groupby(["player_club_id", "player_id"])
             .agg(minutes=("minutes_played", "sum"), last=("date", "max")).reset_index())
    years = (cutoff - inter["last"]).dt.days / 365.25
    inter["weight"] = np.log1p(inter["minutes"] / 90) * C.DECAY_PER_YEAR ** years
    return inter.rename(columns={"player_club_id": "club_id"})


def signings(transfers: pd.DataFrame, start, end) -> pd.DataFrame:
    """Transferências (chegadas) entre start e end."""
    t = transfers[(transfers.transfer_date >= start) & (transfers.transfer_date <= end)]
    return t[["player_id", "to_club_id", "from_club_id", "transfer_date", "transfer_fee"]]


@dataclass
class Dataset:
    """Matriz esparsa usuário×item (clube×jogador) e os mapeamentos de IDs."""

    R: csr_matrix
    user_ids: np.ndarray  # índice -> club_id
    item_ids: np.ndarray  # índice -> player_id
    user_index: dict
    item_index: dict

    @classmethod
    def from_interactions(cls, df: pd.DataFrame, user_col="club_id", item_col="player_id",
                          value_col="weight") -> "Dataset":
        user_ids = np.sort(df[user_col].unique())
        item_ids = np.sort(df[item_col].unique())
        user_index = {u: i for i, u in enumerate(user_ids)}
        item_index = {m: i for i, m in enumerate(item_ids)}
        R = csr_matrix(
            (df[value_col].to_numpy(dtype=np.float32),
             (df[user_col].map(user_index).to_numpy(), df[item_col].map(item_index).to_numpy())),
            shape=(len(user_ids), len(item_ids)),
        )
        return cls(R, user_ids, item_ids, user_index, item_index)

    def user_row(self, club_id, extra: dict | None = None) -> csr_matrix:
        """Linha do clube na matriz, somada a interações extras {player_id: peso} (fold-in)."""
        row = self.R[self.user_index[club_id]] if club_id in self.user_index else \
            csr_matrix((1, len(self.item_ids)), dtype=np.float32)
        if extra:
            cols = [self.item_index[p] for p in extra if p in self.item_index]
            vals = [extra[p] for p in extra if p in self.item_index]
            add = csr_matrix((np.array(vals, dtype=np.float32), ([0] * len(cols), cols)),
                             shape=row.shape)
            row = row.maximum(add)
        return row.tocsr()


# ---------------------------------------------------------------- shortlist (interface)
def save_shortlist(club_id: int, player_id: int, interest: int) -> None:
    """Olheiro adiciona um jogador à shortlist do clube com interesse de 1 a 5."""
    SHORTLIST_FILE.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{"club_id": club_id, "player_id": player_id, "interest": interest,
                         "timestamp": int(time.time())}])
    row.to_csv(SHORTLIST_FILE, mode="a", header=not SHORTLIST_FILE.exists(), index=False)


def load_shortlist(club_id: int) -> dict:
    if not SHORTLIST_FILE.exists():
        return {}
    df = pd.read_csv(SHORTLIST_FILE)
    df = df[df.club_id == club_id].drop_duplicates("player_id", keep="last")
    return dict(zip(df.player_id.astype(int), df.interest.astype(int)))


def clear_shortlist(club_id: int) -> None:
    if SHORTLIST_FILE.exists():
        df = pd.read_csv(SHORTLIST_FILE)
        df[df.club_id != club_id].to_csv(SHORTLIST_FILE, index=False)
