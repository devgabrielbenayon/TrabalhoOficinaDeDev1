"""Modelos de recomendação clube -> jogador.

Todos seguem a mesma interface:
    fit(R)                        -> treina com a matriz esparsa clube×jogador (peso implícito)
    scores(club_row, club_id)     -> pontuação de cada jogador (item) para o clube
    recommend(club_row, k, ...)   -> k melhores jogadores do pool, EXCLUINDO quem já passou
                                     pelo clube

`club_row` é a linha 1×n_jogadores do clube (histórico + shortlist feita na interface),
então o mesmo código atende clubes do treino e clubes novos (fold-in), sem retreinar.

Filtragem colaborativa: Popularidade (baseline), Item-KNN, SVD.
Conteúdo: perfil estatístico (FBref) + faixa de valor do elenco.  Híbrido: combinação.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from src import config as C
from src.features import build_feature_matrix


def _minmax(s: np.ndarray, mask: np.ndarray) -> np.ndarray:
    v = s[mask]
    lo, hi = (v.min(), v.max()) if len(v) else (0.0, 1.0)
    out = np.zeros_like(s, dtype=np.float64)
    if hi > lo:
        out[mask] = (v - lo) / (hi - lo)
    return out


class BaseRecommender:
    name = "base"

    def fit(self, R: csr_matrix, pool_mask: np.ndarray):
        """pool_mask: itens (jogadores) que podem ser recomendados (universo scoutável)."""
        self.n_items = R.shape[1]
        self.pool_mask = pool_mask
        return self

    def scores(self, club_row: csr_matrix, club_id=None) -> np.ndarray:
        raise NotImplementedError

    def recommend(self, club_row: csr_matrix, k: int = C.K, club_id=None,
                  allowed: np.ndarray | None = None) -> np.ndarray:
        s = self.scores(club_row, club_id).astype(np.float64).copy()
        ok = self.pool_mask.copy() if allowed is None else self.pool_mask & allowed
        ok[club_row.indices] = False  # nunca recomendar quem já passou pelo clube
        s[~ok] = -np.inf
        k = min(k, int(ok.sum()))
        if k <= 0:
            return np.array([], dtype=int)
        top = np.argpartition(-s, k - 1)[:k]
        return top[np.argsort(-s[top], kind="stable")]


class PopularityRecommender(BaseRecommender):
    """Baseline não personalizado: jogadores mais valorizados (valor de mercado antes do
    corte). É também a resposta de cold start para um clube sem histórico."""

    name = "Popularidade"

    def __init__(self, item_value: np.ndarray):
        self.item_value = item_value

    def fit(self, R, pool_mask):
        super().fit(R, pool_mask)
        self._s = np.log1p(np.nan_to_num(self.item_value, nan=0.0))
        return self

    def scores(self, club_row, club_id=None):
        return self._s


class ItemKNNRecommender(BaseRecommender):
    """Jogadores parecidos = jogadores que passaram pelos MESMOS clubes (cosseno entre as
    colunas de R). Pontuação para o clube c:  Σ_i sim(j,i) · w_ci  sobre o histórico de c.
    Captura "rotas de transferência" (clubes que compram/vendem entre si) e ex-companheiros."""

    name = "Item-KNN"

    def __init__(self, k_neighbors: int = 100):
        self.k = k_neighbors

    def fit(self, R, pool_mask):
        super().fit(R, pool_mask)
        Rn = normalize(R.tocsc().astype(np.float64), axis=0)
        S = (Rn.T @ Rn).tocsc()          # esparsa: só pares que dividiram clube
        S.setdiag(0)
        S.eliminate_zeros()
        # poda: mantém os k maiores por coluna
        S = S.tocsc()
        for j in range(S.shape[1]):
            a, b = S.indptr[j], S.indptr[j + 1]
            if b - a > self.k:
                col = S.data[a:b]
                cut = np.partition(col, -self.k)[-self.k]
                col[col < cut] = 0
        S.eliminate_zeros()
        self.S = S.tocsr()
        return self

    def scores(self, club_row, club_id=None):
        return np.asarray((club_row @ self.S).todense()).ravel()

    def explain(self, club_row, item: int, top: int = 2) -> list[int]:
        """Jogadores do histórico do clube que mais contribuíram para recomendar `item`."""
        sims = np.asarray(self.S[club_row.indices, item].todense()).ravel() * club_row.data
        order = np.argsort(-sims)[:top]
        return [int(club_row.indices[o]) for o in order if sims[o] > 0]


class SVDRecommender(BaseRecommender):
    """Fatoração R ≈ U·Σ·Vᵀ. Fatores latentes ~ "mercados" (liga, país, patamar do clube).
    Fold-in para qualquer clube: z = x·V, pontuação = z·Vᵀ."""

    name = "SVD"

    def __init__(self, n_factors: int = 64, random_state: int = 42):
        self.n_factors = n_factors
        self.random_state = random_state

    def fit(self, R, pool_mask):
        super().fit(R, pool_mask)
        self.V = TruncatedSVD(self.n_factors, random_state=self.random_state).fit(R).components_
        return self

    def scores(self, club_row, club_id=None):
        z = np.asarray(club_row @ self.V.T).ravel()
        return z @ self.V


class ContentRecommender(BaseRecommender):
    """Baseado em conteúdo (FBref + valor de mercado), sem usar outros clubes.

    Para o clube c e o candidato j (grupo de posição g):
      estilo(j)  = −distância entre o vetor de métricas de j e o perfil médio dos
                   jogadores do histórico de c no grupo g (ponderado pelo peso na matriz);
      preço(j)   = −|log(valor_j) − média ponderada do log(valor) do histórico de c|;
      score      = estilo normalizado + price_weight · preço normalizado.
    Um clube cujo histórico não tem jogadores com métricas FBref fica só com o termo de
    preço (se houver) ou sem pontuação (cold start).
    """

    name = "Conteúdo"

    def __init__(self, players: pd.DataFrame, item_ids: np.ndarray, item_value: np.ndarray,
                 price_weight: float = 1.0):
        self.players = players
        self.item_ids = item_ids
        self.item_value = item_value
        self.price_weight = price_weight

    def fit(self, R, pool_mask):
        super().fit(R, pool_mask)
        pos = {pid: i for i, pid in enumerate(self.item_ids)}
        self.item_group = np.full(self.n_items, None, dtype=object)
        self.Zw = {}
        for g in C.GROUP_LABEL:
            pool, _, Zw = build_feature_matrix(self.players, g)
            Zw = Zw[pool.player_id.notna().to_numpy()]
            ids = pool.loc[Zw.index, "player_id"].astype(int).to_numpy()
            keep = np.array([p in pos for p in ids])
            Zw, ids = Zw[keep], ids[keep]
            cols = np.array([pos[p] for p in ids])
            self.item_group[cols] = g
            self.Zw[g] = (cols, Zw.to_numpy())
        self.log_value = np.log1p(np.nan_to_num(self.item_value, nan=0.0))
        return self

    def club_profile(self, club_row) -> dict:
        """Centróide, por grupo, dos jogadores do histórico do clube que têm métricas FBref.

        Usa só a linha do clube na matriz (dados ANTES do corte, peso maior para quem jogou
        mais e mais recentemente), então não há vazamento; a shortlist da interface entra
        aqui também e desloca o perfil.
        """
        w_all = np.zeros(self.n_items)
        w_all[club_row.indices] = club_row.data
        prof = {}
        for g, (cols, Z) in self.Zw.items():
            w = w_all[cols]
            if w.sum() > 0:
                prof[g] = np.average(Z, axis=0, weights=w)
        return prof

    def scores(self, club_row, club_id=None):
        style = np.zeros(self.n_items)
        has_style = np.zeros(self.n_items, dtype=bool)
        for g, centroid in self.club_profile(club_row).items():
            cols, Z = self.Zw[g]
            style[cols] = -np.linalg.norm(Z - centroid, axis=1)
            has_style[cols] = True
        price = np.zeros(self.n_items)
        known = club_row.indices[self.log_value[club_row.indices] > 0]
        if len(known):
            w = np.asarray(club_row[:, known].todense()).ravel()
            price = -np.abs(self.log_value - np.average(self.log_value[known], weights=w))
        mask = self.pool_mask
        return _minmax(style, mask & has_style) + self.price_weight * _minmax(price, mask)


class HybridRecommender(BaseRecommender):
    """α·CF + (1−α)·Conteúdo, com as pontuações normalizadas (min-max) no pool."""

    name = "Híbrido"

    def __init__(self, cf: BaseRecommender, content: ContentRecommender, alpha: float = 0.5):
        self.cf, self.content, self.alpha = cf, content, alpha

    def fit(self, R, pool_mask):
        super().fit(R, pool_mask)
        return self  # componentes já treinados

    def scores(self, club_row, club_id=None):
        m = self.pool_mask
        cf = _minmax(self.cf.scores(club_row, club_id), m)
        ct = _minmax(self.content.scores(club_row, club_id), m)
        return self.alpha * cf + (1 - self.alpha) * ct
