"""Avaliação offline com clubes de teste (divisão temporal).

Protocolo:
  * Treino: aparições (clube × jogador × minutos) ANTES de 01/07/2025.
  * Teste: contratações reais dos clubes das 5 grandes ligas entre 01/07/2025 e
    02/02/2026 (janelas de verão e de janeiro), de jogadores do universo scoutável
    (FBref 2024/25, ≥ 900 min) que NUNCA tinham passado pelo clube (exclui retorno de
    empréstimo). Cada clube que contratou ≥ 1 desses jogadores é um "usuário de teste".
  * Cada modelo recomenda K = 20 jogadores por clube; acerto = jogador contratado.

Métricas: Precision@20, Recall@20, NDCG@20, HitRate@20 (clubes com ≥ 1 acerto) e
cobertura. Os hiperparâmetros (α do híbrido etc.) foram escolhidos na janela de
validação 2024/25 (ver `tune`), sem olhar o teste.

Também valida o KNN de similaridade: no espaço "GERAL" (todas as métricas, sem saber a
posição), qual % dos 10 vizinhos de cada jogador tem a mesma posição?

Execução:  python -m src.evaluate            (avaliação no teste)
           python -m src.evaluate --tune     (busca de hiperparâmetros na validação)
"""

import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src import config as C
from src.data import ROOT, Dataset, build_interactions, build_players, load_transfermarkt, signings, value_at
from src.features import build_feature_matrix
from src.models import (BaseRecommender, ContentRecommender, HybridRecommender,
                        ItemKNNRecommender, PopularityRecommender, SVDRecommender)

RESULTS_DIR = ROOT / "results"
EXAMPLE_MISS = "Arsenal FC"  # clube grande sem acertos, mostrado como contraexemplo

# hiperparâmetros escolhidos na validação (python -m src.evaluate --tune)
#   (melhor NDCG@20 de cada modelo; o termo de preço piorou o Conteúdo em todas as combinações)
PARAMS = {"k_neighbors": 1000, "n_factors": 256, "price_weight": 0.0, "alpha": 0.75}


@dataclass
class Split:
    ds: Dataset
    pool_mask: np.ndarray
    item_value: np.ndarray
    test: dict            # club_id -> set(índices de itens contratados)
    cutoff: pd.Timestamp
    inter: pd.DataFrame   # interações usadas no treino


def build_split(tm: dict, players: pd.DataFrame, cutoff, end) -> Split:
    inter = build_interactions(tm["appearances"], cutoff)
    ds = Dataset.from_interactions(inter)
    scout = players[(players.Min >= C.MIN_MINUTES) & players.group.notna() & players.player_id.notna()]
    pool_ids = set(scout.player_id.astype(int))
    pool_mask = np.array([p in pool_ids for p in ds.item_ids])
    item_value = pd.Series(ds.item_ids).map(value_at(tm["valuations"], cutoff)).to_numpy(dtype=float)

    big5 = set(tm["clubs"].loc[tm["clubs"].domestic_competition_id.isin(C.LEAGUES.values()), "club_id"])
    s = signings(tm["transfers"], cutoff, end)
    s = s[s.to_club_id.isin(big5) & s.player_id.isin(pool_ids) & s.player_id.isin(ds.item_index)]
    hist = set(zip(inter.club_id, inter.player_id))
    before = tm["transfers"][tm["transfers"].transfer_date < cutoff]
    left_club = set(zip(before.from_club_id, before.player_id))  # saiu antes = volta de empréstimo
    new = np.array([(c, p) not in hist and (c, p) not in left_club
                    for c, p in zip(s.to_club_id, s.player_id)], dtype=bool)
    s = s[new]
    test = {c: {ds.item_index[p] for p in g.player_id} for c, g in s.groupby("to_club_id")}
    return Split(ds, pool_mask, item_value, test, pd.Timestamp(cutoff), inter)


class RandomRecommender(BaseRecommender):
    """Referência para a análise: sorteia jogadores do pool (média de 20 sorteios)."""

    name = "Aleatório"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def scores(self, club_row, club_id=None):
        return self.rng.random(self.n_items)


def make_models(split: Split, players: pd.DataFrame, params: dict = PARAMS) -> dict:
    R, m = split.ds.R, split.pool_mask
    pop = PopularityRecommender(split.item_value).fit(R, m)
    knn = ItemKNNRecommender(params["k_neighbors"]).fit(R, m)
    svd = SVDRecommender(params["n_factors"]).fit(R, m)
    content = ContentRecommender(players, split.ds.item_ids, split.item_value,
                                 params["price_weight"]).fit(R, m)
    hybrid = HybridRecommender(knn, content, params["alpha"]).fit(R, m)
    return {"Popularidade": pop, "Item-KNN": knn, "SVD": svd, "Conteúdo": content, "Híbrido": hybrid}


def recommend_for(model, pop, ds: Dataset, club_id, k=C.K, extra=None, allowed=None):
    """Recomendação com fallback de cold start: clube sem histórico -> popularidade."""
    row = ds.user_row(club_id, extra)
    m = model if row.nnz else pop
    return m.recommend(row, k, club_id=club_id, allowed=allowed), m


def ndcg(rec, relevant, k):
    gains = np.array([1.0 if i in relevant else 0.0 for i in rec[:k]])
    dcg = (gains / np.log2(np.arange(2, len(gains) + 2))).sum()
    idcg = (1.0 / np.log2(np.arange(2, min(len(relevant), k) + 2))).sum()
    return dcg / idcg if idcg else 0.0


def evaluate_model(model, pop, split: Split, k=C.K) -> dict:
    P, Rc, N, H = [], [], [], []
    recommended = set()
    for club, relevant in split.test.items():
        rec, _ = recommend_for(model, pop, split.ds, club, k)
        row = split.ds.user_row(club)
        assert not set(rec) & set(row.indices), "recomendou jogador que já passou pelo clube!"
        recommended.update(rec.tolist())
        hits = len(set(rec) & relevant)
        P.append(hits / k)
        Rc.append(hits / len(relevant))
        N.append(ndcg(rec, relevant, k))
        H.append(hits > 0)
    return {"modelo": model.name, f"Precision@{k}": np.mean(P), f"Recall@{k}": np.mean(Rc),
            f"NDCG@{k}": np.mean(N), f"HitRate@{k}": np.mean(H),
            "Cobertura": len(recommended) / split.pool_mask.sum(), "clubes": len(P)}


def position_validation(players: pd.DataFrame, k: int = 10) -> dict:
    """% dos k vizinhos (espaço GERAL, sem posição) com a mesma posição do jogador."""
    pool, _, Zw = build_feature_matrix(players, "GERAL")
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Zw.to_numpy())
    _, idx = nn.kneighbors(Zw.to_numpy())
    groups = pool["group"].to_numpy()
    subs = pool["sub_position"].to_numpy()
    same_g = (groups[idx[:, 1:]] == groups[:, None]).mean()
    same_s = (subs[idx[:, 1:]] == subs[:, None]).mean()
    pg = pool["group"].value_counts(normalize=True)
    ps = pool["sub_position"].value_counts(normalize=True)
    per_group = pd.Series((groups[idx[:, 1:]] == groups[:, None]).mean(axis=1), index=pool.index)
    return {"mesmo_grupo": same_g, "mesmo_grupo_acaso": float((pg ** 2).sum()),
            "mesma_sub_posicao": same_s, "mesma_sub_posicao_acaso": float((ps ** 2).sum()),
            "por_grupo": per_group.groupby(pool["group"]).mean().to_dict(), "jogadores": len(pool)}


def build_examples(models: dict, split: Split, tm: dict, players: pd.DataFrame, k=C.K) -> str:
    ds = split.ds
    names = tm["clubs"].set_index("club_id")["name"]
    pname = tm["players"].set_index("player_id")["name"]
    pinfo = players.dropna(subset=["player_id"]).set_index("player_id")
    label = lambda i: f"{pname.get(ds.item_ids[i], ds.item_ids[i])} ({pinfo['Squad'].get(ds.item_ids[i], '?')})"
    show = ["Popularidade", "Item-KNN", "Híbrido"]
    recs_all = {c: {n: recommend_for(models[n], models["Popularidade"], ds, c, k)[0] for n in show}
                for c in split.test}
    # 4 clubes com mais acertos do Híbrido + 1 clube grande sem acertos (mostra a limitação)
    by_hits = sorted(split.test, key=lambda c: (-len(set(recs_all[c]["Híbrido"]) & split.test[c]),
                                                -len(split.test[c])))
    clubs = by_hits[:4]
    miss = [c for c in split.test if names.get(c) == EXAMPLE_MISS
            and not set(recs_all[c]["Híbrido"]) & split.test[c]]
    clubs += miss or by_hits[-1:]
    lines = ["# Exemplos para 5 clubes de teste", "",
             f"Recomendações feitas com dados até {split.cutoff.date()}; ✅ = jogador que o clube "
             "realmente contratou entre 01/07/2025 e 02/02/2026. Os 4 primeiros clubes são os "
             "com mais acertos do Híbrido; o último é um clube grande sem acertos.", ""]
    for club in clubs:
        row = ds.user_row(club)
        recent = pd.Series(row.toarray().ravel(), index=range(ds.R.shape[1]))
        recent = recent[recent > 0].sort_values(ascending=False).index[:8]
        relevant = split.test[club]
        lines += [f"## {names[club]}", "",
                  "**Contratações reais (teste):** " + "; ".join(label(i) for i in relevant), "",
                  "**Histórico (jogadores com mais peso recente):** " + "; ".join(
                      str(pname.get(ds.item_ids[i], "?")) for i in recent), "",
                  "| # | " + " | ".join(show) + " |", "|---|" + "---|" * len(show)]
        recs = recs_all[club]
        for pos in range(k):
            cells = [label(recs[n][pos]) + (" ✅" if recs[n][pos] in relevant else "") for n in show]
            lines.append(f"| {pos + 1} | " + " | ".join(cells) + " |")
        hits = {n: len(set(r) & relevant) for n, r in recs.items()}
        lines += ["", f"Acertos no top-{k}: " + ", ".join(f"{n}: {h}" for n, h in hits.items()), ""]
    # cold start
    rec, used = recommend_for(models["Híbrido"], models["Popularidade"], ds, club_id=-1, k=10)
    lines += ["## 🆕 Clube novo (sem histórico)", "",
              f"Sem histórico a filtragem colaborativa não se aplica; o sistema usa **{used.name}**:", "",
              "; ".join(label(i) for i in rec), ""]
    return "\n".join(lines)


def tune(tm, players):
    """Busca simples de hiperparâmetros na janela de validação (2024/25)."""
    split = build_split(tm, players, C.VALID_SPLIT, C.VALID_END)
    print(f"Validação: {len(split.test)} clubes, {sum(map(len, split.test.values()))} contratações")
    rows = []
    for kn in (100, 300, 1000):
        for nf in (16, 64, 256):
            p = dict(PARAMS, k_neighbors=kn, n_factors=nf)
            ms = make_models(split, players, p)
            for name in ("Item-KNN", "SVD"):
                rows.append({"params": f"k={kn} f={nf}", **evaluate_model(ms[name], ms["Popularidade"], split)})
    base = make_models(split, players)
    for pw in (0.0, 0.5, 1.0, 2.0):
        for a in (0.0, 0.25, 0.5, 0.75, 1.0):
            base["Conteúdo"].price_weight = pw
            h = HybridRecommender(base["Item-KNN"], base["Conteúdo"], a).fit(split.ds.R, split.pool_mask)
            rows.append({"params": f"pw={pw} a={a}", **evaluate_model(h, base["Popularidade"], split)})
    df = pd.DataFrame(rows).drop_duplicates(["modelo", "params"])
    print(df.sort_values("NDCG@20", ascending=False).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def run(verbose: bool = True) -> pd.DataFrame:
    players = build_players()
    tm = load_transfermarkt()
    split = build_split(tm, players, C.SPLIT_DATE, C.TEST_END)
    if verbose:
        n_rel = sum(map(len, split.test.values()))
        print(f"Treino: {split.ds.R.nnz} interações | {split.ds.R.shape[0]} clubes × "
              f"{split.ds.R.shape[1]} jogadores | pool scoutável: {split.pool_mask.sum()}")
        print(f"Teste: {len(split.test)} clubes, {n_rel} contratações")
    models = make_models(split, players)
    rows = [evaluate_model(m, models["Popularidade"], split) for m in models.values()]
    rnd = RandomRecommender().fit(split.ds.R, split.pool_mask)
    rnd_runs = pd.DataFrame([evaluate_model(rnd, rnd, split) for _ in range(20)])
    rows.insert(0, {"modelo": rnd.name, **rnd_runs.drop(columns="modelo").mean().to_dict()})
    df = pd.DataFrame(rows)
    val = position_validation(players)

    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / "metrics.csv", index=False)
    pd.DataFrame([{k: v for k, v in val.items() if k != "por_grupo"}]).to_csv(
        RESULTS_DIR / "position_validation.csv", index=False)
    (RESULTS_DIR / "examples.md").write_text(build_examples(models, split, tm, players), encoding="utf-8")
    if verbose:
        print()
        print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("\nValidação do KNN de similaridade (10 vizinhos no espaço GERAL):")
        print(f"  mesmo grupo de posição: {val['mesmo_grupo']:.1%} (acaso: {val['mesmo_grupo_acaso']:.1%})")
        print(f"  mesma sub-posição:      {val['mesma_sub_posicao']:.1%} (acaso: {val['mesma_sub_posicao_acaso']:.1%})")
        print("  por grupo:", {g: f"{v:.0%}" for g, v in val["por_grupo"].items()})
        print(f"\nResultados salvos em {RESULTS_DIR}")
    return df


if __name__ == "__main__":
    if "--tune" in sys.argv:
        tune(load_transfermarkt(), build_players())
    else:
        run()
