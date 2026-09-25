"""Modelos de recomendação: baseline de popularidade, Item-KNN e SVD.

Todos seguem a mesma interface:
    fit(R)                      -> treina com a matriz esparsa usuário×item R
    scores(user_row)            -> vetor de pontuações (1 por item) para um usuário
    recommend(user_row, k)      -> índices dos k melhores itens, EXCLUINDO os já avaliados
    predict(user_row, items)    -> nota prevista para itens específicos (usado no RMSE)

`user_row` é um vetor esparso 1×n_itens com as notas do usuário. Assim o mesmo código
serve para usuários do treino e para usuários novos criados na interface (fold-in),
sem precisar retreinar o modelo.
"""

import numpy as np
from scipy.sparse import csr_matrix, diags
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


def _user_mean(user_row: csr_matrix, default: float) -> float:
    return float(user_row.data.mean()) if user_row.nnz else default


def _center_rows(R: csr_matrix) -> tuple[csr_matrix, np.ndarray]:
    """Subtrai a média de cada usuário apenas das notas observadas."""
    R = R.tocsr().astype(np.float64)
    counts = np.diff(R.indptr)
    sums = np.asarray(R.sum(axis=1)).ravel()
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    C = R.copy()
    C.data -= np.repeat(means, counts)
    return C, means


class BaseRecommender:
    name = "base"

    def fit(self, R: csr_matrix):
        self.n_items = R.shape[1]
        self.global_mean = float(R.data.mean())
        return self

    def scores(self, user_row: csr_matrix) -> np.ndarray:
        raise NotImplementedError

    def recommend(self, user_row: csr_matrix, k: int = 10, exclude_seen: bool = True) -> np.ndarray:
        s = self.scores(user_row).astype(np.float64).copy()
        if exclude_seen and user_row.nnz:
            s[user_row.indices] = -np.inf  # nunca recomendar itens já conhecidos
        k = min(k, int(np.isfinite(s).sum()))
        top = np.argpartition(-s, k - 1)[:k] if k > 0 else np.array([], dtype=int)
        return top[np.argsort(-s[top])]

    def predict(self, user_row: csr_matrix, items: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class PopularityRecommender(BaseRecommender):
    """Ranking por média bayesiana: (C·m + Σnotas) / (C + n).

    Puxa para a média global filmes com poucas avaliações, evitando que um filme com
    uma única nota 5 fique no topo. Não é personalizado: é o baseline e a solução de
    cold start (usuário sem histórico).
    """

    name = "Popularidade"

    def __init__(self, C: float | None = None):
        self.C = C

    def fit(self, R: csr_matrix):
        super().fit(R)
        Rc = R.tocsc()
        n = np.diff(Rc.indptr).astype(np.float64)
        s = np.asarray(Rc.sum(axis=0)).ravel()
        C = self.C if self.C is not None else float(np.median(n[n > 0]))
        self.item_count = n
        self.item_mean = np.divide(s, n, out=np.full_like(s, self.global_mean), where=n > 0)
        self.bayes = (C * self.global_mean + s) / (C + n)
        # ranking = média bayesiana ponderada pelo log do nº de avaliações:
        # favorece filmes bem avaliados E conhecidos por muita gente
        self._rank = self.bayes * np.log1p(n)
        return self

    def scores(self, user_row: csr_matrix) -> np.ndarray:
        return self._rank

    def predict(self, user_row: csr_matrix, items: np.ndarray) -> np.ndarray:
        return self.bayes[items]


def _cosine_topk(M: csr_matrix, k: int) -> csr_matrix:
    """Similaridade de cosseno entre colunas de M, mantendo os k maiores vizinhos por item."""
    Mn = normalize(M.tocsc().astype(np.float64), axis=0)  # colunas norma 1 -> produto = cosseno
    S = (Mn.T @ Mn).toarray()
    np.fill_diagonal(S, 0.0)
    S[S < 0] = 0.0
    if k < S.shape[0]:
        idx = np.argpartition(-S, k, axis=0)[k:, :]
        np.put_along_axis(S, idx, 0.0, axis=0)
    return csr_matrix(S.astype(np.float32))  # S[i, j] = similaridade de i como vizinho de j


def _center_row(user_row: csr_matrix, global_mean: float) -> tuple[csr_matrix, float]:
    mu = _user_mean(user_row, global_mean)
    c = user_row.astype(np.float64).copy()
    c.data -= mu
    return c, mu


class ItemKNNRecommender(BaseRecommender):
    """Filtragem colaborativa baseada em itens ("quem viu X também viu Y").

    Duas matrizes de similaridade de cosseno item×item, cada uma com os k vizinhos
    mais próximos de cada filme:

    * Ranking (top-N): similaridade sobre as notas brutas. Pontuação do item j para
      o usuário u:  Σ_i sim(j,i) · r_ui  sobre os filmes i do histórico de u.
      Capta tanto "o que ele assistiu" quanto "o quanto gostou".
    * Nota prevista (RMSE): similaridade sobre notas centralizadas pela média do
      usuário, e  r̂_uj = r̄_u + Σ sim·(r_ui − r̄_u) / Σ sim.

    Na avaliação offline, ranquear pela nota prevista foi muito pior (favorece filmes
    obscuros com 1–2 vizinhos), por isso o ranking usa as notas brutas (ver README).
    """

    name = "Item-KNN"

    def __init__(self, k_neighbors: int = 50):
        self.k = k_neighbors

    def fit(self, R: csr_matrix):
        super().fit(R)
        self.S_rank = _cosine_topk(R, self.k)
        self.S_pred = _cosine_topk(_center_rows(R)[0], self.k)
        return self

    def scores(self, user_row: csr_matrix) -> np.ndarray:
        return np.asarray((user_row @ self.S_rank).todense()).ravel()

    def predict(self, user_row: csr_matrix, items: np.ndarray) -> np.ndarray:
        c, mu = _center_row(user_row, self.global_mean)
        S = self.S_pred[:, items]
        num = np.asarray((c @ S).todense()).ravel()
        mask = user_row.copy()
        mask.data[:] = 1.0
        den = np.asarray((mask @ S).todense()).ravel()
        pred = mu + np.divide(num, den, out=np.zeros_like(num), where=den > 1e-9)
        return np.clip(pred, 0.5, 5.0)

    def explain(self, user_row: csr_matrix, item: int, top: int = 3) -> list[int]:
        """Filmes do histórico que mais contribuíram para recomendar `item`."""
        sims = np.asarray(self.S_rank[user_row.indices, item].todense()).ravel()
        contrib = sims * user_row.data
        order = np.argsort(-contrib)[:top]
        return [int(user_row.indices[o]) for o in order if contrib[o] > 0]


class SVDRecommender(BaseRecommender):
    """Fatoração de matrizes com SVD truncado: R ≈ U·Σ·Vᵀ com poucos fatores latentes.

    Cada fator latente captura um "gosto" (ex.: ação, cult, animação). Para qualquer
    usuário (inclusive um novo, criado na interface) usa-se fold-in, sem retreinar:
        z = x·V   (perfil do usuário no espaço de fatores)
        x̂ = z·Vᵀ  (reconstrução = pontuação para todos os filmes)

    * Ranking (top-N): SVD sobre as notas brutas, com 0 onde não há nota ("PureSVD",
      Cremonesi et al., 2010).
    * Nota prevista (RMSE): SVD sobre notas centralizadas pela média do usuário,
      mais um viés de item:  r̂_uj = r̄_u + b_j + x̂_j.
    """

    name = "SVD"

    def __init__(self, n_factors: int = 50, random_state: int = 42):
        self.n_factors = n_factors
        self.random_state = random_state

    def fit(self, R: csr_matrix):
        super().fit(R)
        svd = lambda: TruncatedSVD(n_components=self.n_factors, random_state=self.random_state)
        self.V_rank = svd().fit(R).components_  # (fatores × itens)
        C, _ = _center_rows(R)
        self.V_pred = svd().fit(C).components_
        # viés de item: quanto cada filme fica acima/abaixo da média dos seus avaliadores
        n = np.diff(C.tocsc().indptr)
        self.item_bias = np.asarray(C.sum(axis=0)).ravel() / (n + 10.0)
        return self

    def scores(self, user_row: csr_matrix) -> np.ndarray:
        z = np.asarray(user_row @ self.V_rank.T).ravel()
        return z @ self.V_rank

    def predict(self, user_row: csr_matrix, items: np.ndarray) -> np.ndarray:
        c, mu = _center_row(user_row, self.global_mean)
        z = np.asarray(c @ self.V_pred.T).ravel()
        pred = mu + self.item_bias[items] + z @ self.V_pred[:, items]
        return np.clip(pred, 0.5, 5.0)


MODELS = {
    "Popularidade": PopularityRecommender,
    "Item-KNN": ItemKNNRecommender,
    "SVD": SVDRecommender,
}
