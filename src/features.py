"""Engenharia de atributos para o KNN de jogadores.

Pipeline (por grupo de posição):
  1. pool elegível: jogadores do grupo com ≥ MIN_MINUTES (o jogador de referência é
     sempre incluído, mesmo abaixo do corte);
  2. contagens -> por 90 minutos;
  3. percentuais recalculados e encolhidos para a média do grupo pelo nº de tentativas
     (um zagueiro com 2/2 duelos aéreos não vira "100%");
  4. (opcional) métricas de produção multiplicadas pelo fator de nível da liga;
  5. z-score dentro do grupo (ou dentro da liga), cortado em ±3;
  6. multiplicação por √peso -> distância euclidiana = euclidiana ponderada.
"""

import numpy as np
import pandas as pd

from src import config as C


def per90_matrix(pool: pd.DataFrame, metrics) -> pd.DataFrame:
    """Valores "reais" (por 90 / percentuais encolhidos) — usados em filtros, radar e tabela."""
    X = pd.DataFrame(index=pool.index)
    for col in metrics:
        if col in C.RATE_COLS:
            num, den = C.RATE_COLS[col]
            prior = pool[num].sum() / max(pool[den].sum(), 1)
            X[col] = 100 * (pool[num] + prior * C.SHRINK_K) / (pool[den] + C.SHRINK_K)
        else:
            X[col] = pool[col] / (pool["Min"] / 90)
    return X


def build_feature_matrix(players: pd.DataFrame, group: str, weights: dict | None = None,
                         min_minutes: int = C.MIN_MINUTES, include=(),
                         league_adjust: bool = False, within_league: bool = False):
    """Retorna (pool, X, Zw).

    pool: linhas de `players` elegíveis; X: métricas por 90; Zw: z-score ponderado
    (é o espaço onde o KNN mede distância).
    """
    weights = weights or C.FEATURES[group]
    in_group = players["group"].notna() if group == "GERAL" else players["group"] == group
    ok = in_group & ((players["Min"] >= min_minutes) | players.index.isin(list(include)))
    pool = players[ok]
    X = per90_matrix(pool, weights)

    Xs = X.copy()
    if league_adjust:
        cols = [c for c in Xs if c in C.OUTPUT_COLS]
        Xs[cols] = Xs[cols].mul(pool["Comp"].map(C.LEAGUE_FACTOR).fillna(1.0), axis=0)

    # média/desvio calculados só com quem passou no corte de minutos
    base = pool["Min"] >= min_minutes
    if within_league:
        g = Xs[base].groupby(pool.loc[base, "Comp"])
        mu = g.mean().reindex(pool["Comp"]).set_axis(pool.index)
        sd = g.std(ddof=0).reindex(pool["Comp"]).set_axis(pool.index)
    else:
        mu, sd = Xs[base].mean(), Xs[base].std(ddof=0)
    Z = ((Xs - mu) / sd.replace(0, 1)).clip(-C.Z_CLIP, C.Z_CLIP).fillna(0.0)
    w = pd.Series(weights, dtype=float)[X.columns]
    return pool, X, Z * np.sqrt(w)


def percentiles(X: pd.DataFrame, base_mask=None) -> pd.DataFrame:
    """Percentil (0–100) de cada jogador em cada métrica, dentro do pool."""
    ref = X if base_mask is None else X[base_mask]
    out = {}
    for c in X:
        s = np.sort(ref[c].to_numpy())
        out[c] = 100 * np.searchsorted(s, X[c].to_numpy(), side="left") / len(s)
    return pd.DataFrame(out, index=X.index)
