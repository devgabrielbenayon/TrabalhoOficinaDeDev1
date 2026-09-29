"""Scouting por similaridade: camada de decisão (filtros/regras) + KNN por métricas.

Exemplo:
    players = build_players()
    filt = ScoutFilter(leagues=["La Liga", "Serie A"], age=(18, 25), top_pct={"Int": 30})
    ranking, X, Zw = find_similar(players, ref_idx, n=10, filt=filt)
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from src import config as C
from src.features import build_feature_matrix, percentiles


@dataclass
class ScoutFilter:
    """Regras de decisão aplicadas ANTES do KNN (só candidatos que passam são vizinhos)."""

    leagues: list[str] | None = None        # nomes amigáveis: "Premier League", ...
    clubs: list[str] | None = None          # clube FBref (Squad)
    exclude_clubs: list[str] | None = None
    age: tuple[int, int] | None = None
    max_value_eur: float | None = None
    min_metrics: dict[str, float] = field(default_factory=dict)  # {"PrgP": 5.0} por 90
    top_pct: dict[str, float] = field(default_factory=dict)      # {"Int": 20} -> top 20% do grupo

    def mask(self, pool: pd.DataFrame, X: pd.DataFrame, base=None) -> pd.Series:
        m = pd.Series(True, index=pool.index)
        if self.leagues:
            m &= pool["league"].isin(self.leagues)
        if self.clubs:
            m &= pool["Squad"].isin(self.clubs)
        if self.exclude_clubs:
            m &= ~pool["Squad"].isin(self.exclude_clubs)
        if self.age:
            m &= pool["Age"].between(*self.age)
        if self.max_value_eur is not None:
            m &= pool["value_eur"].fillna(np.inf) <= self.max_value_eur
        for col, v in self.min_metrics.items():
            m &= X[col] >= v
        if self.top_pct:
            pct = percentiles(X[list(self.top_pct)], base)  # percentil no grupo todo
            for col, p in self.top_pct.items():
                m &= pct[col] >= 100 - p
        return m


def find_similar(players: pd.DataFrame, ref: int, n: int = 10, filt: ScoutFilter | None = None,
                 weights: dict | None = None, group: str | None = None, metric: str = "euclidean",
                 **feature_kw):
    """Os `n` jogadores mais parecidos com `players.loc[ref]` que passam nos filtros.

    Retorna (ranking, X, Zw). `similaridade_%` = % do grupo que está MAIS LONGE da
    referência do que o candidato (99% = mais parecido que 99% do grupo).
    """
    filt = filt or ScoutFilter()
    group = group or players.at[ref, "group"]
    pool, X, Zw = build_feature_matrix(players, group, weights, include=[ref], **feature_kw)
    base = pool["Min"] >= feature_kw.get("min_minutes", C.MIN_MINUTES)
    cand = Zw[filt.mask(pool, X, base) & (Zw.index != ref) & base]
    cols = ["Player", "Squad", "league", "Age", "Min", "value_eur", "sub_position"]
    if cand.empty:
        return pd.DataFrame(columns=cols + ["distancia", "similaridade_%"]), X, Zw
    q = Zw.loc[[ref]].to_numpy()
    nn = NearestNeighbors(n_neighbors=min(n, len(cand)), metric=metric).fit(cand.to_numpy())
    dist, idx = nn.kneighbors(q)
    all_d = pairwise_distances(q, Zw[base & (Zw.index != ref)].to_numpy(), metric=metric)[0]
    out = pool.loc[cand.index[idx[0]], cols].copy()
    out["distancia"] = dist[0]
    out["similaridade_%"] = [100 * (all_d > d).mean() for d in dist[0]]
    return out, X, Zw


def explain(Zw: pd.DataFrame, ref: int, other: int, top: int = 3) -> dict:
    """Métricas que mais aproximam e mais afastam dois jogadores (|Δz|·√peso)."""
    diff = (Zw.loc[ref] - Zw.loc[other]).abs().sort_values()
    return {"aproxima": diff.index[:top].tolist(), "afasta": diff.index[::-1][:top].tolist(),
            "diferenca": diff}
