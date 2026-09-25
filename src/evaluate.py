"""Avaliação offline dos recomendadores com usuários de teste.

Protocolo:
  * Para cada usuário, 20% das avaliações (sorteio com seed fixa) vão para TESTE e o
    restante para TREINO. Os modelos só enxergam o treino.
  * Item relevante = filme do teste com nota >= 4.
  * Para cada usuário, gera top-10 excluindo os filmes do treino e compara com o teste.

Métricas:
  * Precision@10, Recall@10, NDCG@10 (qualidade do ranking)
  * RMSE (erro da nota prevista nos pares do teste)
  * Cobertura do catálogo (% dos filmes que aparecem em alguma recomendação)

Além dos 3 modelos, avalia uma variante de Item-KNN e SVD que ranqueia pela nota
prevista, para mostrar por que o ranking usa as notas brutas.

Execução:  python -m src.evaluate
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data import ROOT, Dataset, filter_ratings, load_raw
from src.models import MODELS

RESULTS_DIR = ROOT / "results"
K = 10
TEST_FRAC = 0.2
RELEVANT = 4.0
SEED = 42
EXAMPLE_USERS = [1, 15, 68, 414, 599]


def train_test_split(ratings: pd.DataFrame, test_frac=TEST_FRAC, seed=SEED):
    rng = np.random.default_rng(seed)
    test_mask = np.zeros(len(ratings), dtype=bool)
    for _, idx in ratings.groupby("userId").indices.items():
        if len(idx) < 10:
            continue
        n_test = int(round(len(idx) * test_frac))
        test_mask[rng.choice(idx, n_test, replace=False)] = True
    return ratings[~test_mask].reset_index(drop=True), ratings[test_mask].reset_index(drop=True)


def ndcg_at_k(rec: np.ndarray, relevant: set, k=K) -> float:
    gains = np.array([1.0 if i in relevant else 0.0 for i in rec[:k]])
    dcg = (gains / np.log2(np.arange(2, len(gains) + 2))).sum()
    ideal = min(len(relevant), k)
    idcg = (1.0 / np.log2(np.arange(2, ideal + 2))).sum()
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_model(model, ds: Dataset, test: pd.DataFrame, k=K) -> dict:
    test = test[test["movieId"].isin(ds.item_index) & test["userId"].isin(ds.user_index)]
    precs, recs, ndcgs, sq_err = [], [], [], []
    recommended = set()
    for uid, grp in test.groupby("userId"):
        u = ds.user_index[uid]
        row = ds.R[u]
        items = grp["movieId"].map(ds.item_index).to_numpy()
        preds = model.predict(row, items)
        sq_err.extend((preds - grp["rating"].to_numpy()) ** 2)

        rec = model.recommend(row, k)
        assert not set(rec) & set(row.indices), "recomendou item já avaliado!"
        recommended.update(rec.tolist())
        relevant = set(items[grp["rating"].to_numpy() >= RELEVANT])
        if not relevant:
            continue
        hits = len(set(rec) & relevant)
        precs.append(hits / k)
        recs.append(hits / len(relevant))
        ndcgs.append(ndcg_at_k(rec, relevant, k))
    return {
        "modelo": model.name,
        f"Precision@{k}": np.mean(precs),
        f"Recall@{k}": np.mean(recs),
        f"NDCG@{k}": np.mean(ndcgs),
        "RMSE": float(np.sqrt(np.mean(sq_err))),
        "Cobertura": len(recommended) / ds.R.shape[1],
        "usuarios_avaliados": len(precs),
    }


class RankByPrediction:
    """Variante para análise: ranqueia pela NOTA PREVISTA em vez da pontuação de ranking."""

    def __init__(self, model):
        self.model = model
        self.name = f"{model.name} (rank por nota prevista)"
        self.all_items = np.arange(model.n_items)

    def predict(self, user_row, items):
        return self.model.predict(user_row, items)

    def recommend(self, user_row, k=K):
        s = self.model.predict(user_row, self.all_items).astype(np.float64)
        s[user_row.indices] = -np.inf
        return np.argsort(-s)[:k]


def build_examples(models: dict, ds: Dataset, movies: pd.DataFrame, test: pd.DataFrame, k=K) -> str:
    titles = movies.set_index("movieId")["title"]
    lines = ["# Exemplos para 5 usuários de teste", ""]
    for uid in EXAMPLE_USERS:
        u = ds.user_index[uid]
        row = ds.R[u]
        hist = sorted(zip(row.indices, row.data), key=lambda t: -t[1])
        t = test[(test["userId"] == uid) & (test["rating"] >= RELEVANT)]
        relevant = set(t["movieId"])
        lines += [
            f"## Usuário {uid}",
            "",
            f"{row.nnz} avaliações no treino, {len(relevant)} filmes relevantes (nota ≥ 4) no teste.",
            "",
            "**Filmes favoritos no histórico:** "
            + "; ".join(f"{titles[ds.item_ids[i]]} ({r:.1f})" for i, r in hist[:5]),
            "",
            "| # | " + " | ".join(models) + " |",
            "|---|" + "---|" * len(models),
        ]
        recs = {n: [ds.item_ids[i] for i in m.recommend(row, k)] for n, m in models.items()}
        for pos in range(k):
            cells = []
            for n in models:
                mid = recs[n][pos]
                mark = " ✅" if mid in relevant else ""
                cells.append(f"{titles[mid]}{mark}")
            lines.append(f"| {pos + 1} | " + " | ".join(cells) + " |")
        hits = {n: sum(m in relevant for m in r) for n, r in recs.items()}
        lines += ["", "Acertos no teste: " + ", ".join(f"{n}: {h}" for n, h in hits.items()), ""]
    lines.append("✅ = filme que o usuário avaliou com nota ≥ 4 no conjunto de teste (oculto no treino).")
    return "\n".join(lines)


def run(verbose: bool = True) -> pd.DataFrame:
    ratings, movies = load_raw()
    train, test = train_test_split(ratings)
    train = filter_ratings(train)
    ds = Dataset.from_ratings(train)
    if verbose:
        print(f"Treino: {len(train)} avaliações | Teste: {len(test)} avaliações | "
              f"{ds.R.shape[0]} usuários x {ds.R.shape[1]} filmes")

    models, rows = {}, []
    for name, cls in MODELS.items():
        m = cls().fit(ds.R)
        models[name] = m
        rows.append(evaluate_model(m, ds, test))
        if verbose:
            print(f"  {name}: ok")
    for name in ("Item-KNN", "SVD"):
        rows.append(evaluate_model(RankByPrediction(models[name]), ds, test))

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / "metrics.csv", index=False)
    (RESULTS_DIR / "examples.md").write_text(build_examples(models, ds, movies, test), encoding="utf-8")
    if verbose:
        print()
        print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"\nResultados salvos em {RESULTS_DIR}")
    return df


if __name__ == "__main__":
    run()
