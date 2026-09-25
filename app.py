"""Interface Streamlit do sistema de recomendação de filmes.

Execução:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st

from src.data import (
    Dataset,
    clear_user_ratings,
    dataset_stats,
    filter_ratings,
    load_raw,
    load_user_ratings,
    save_user_rating,
)
from src.evaluate import RESULTS_DIR
from src.models import MODELS, PopularityRecommender

NEW_USER_ID = 0  # IDs do MovieLens começam em 1

st.set_page_config(page_title="Recomendador de Filmes", page_icon="🎬", layout="wide")


# ---------------------------------------------------------------- carga e treino
@st.cache_resource(show_spinner="Carregando dados e treinando modelos...")
def load_everything():
    ratings, movies = load_raw()
    train = filter_ratings(ratings)
    ds = Dataset.from_ratings(train)
    models = {name: cls().fit(ds.R) for name, cls in MODELS.items()}
    return ratings, movies, ds, models


ratings, movies, ds, models = load_everything()
movie_info = movies.set_index("movieId")
title_of = movie_info["title"].to_dict()


def user_profile(user_id: int) -> dict:
    """Histórico completo do usuário: MovieLens + avaliações feitas na interface."""
    base = ratings[ratings["userId"] == user_id]
    profile = dict(zip(base["movieId"].astype(int), base["rating"].astype(float)))
    profile.update(load_user_ratings(user_id))
    return profile


def movies_table(movie_ids, extra: dict | None = None) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "Filme": [title_of[m] for m in movie_ids],
            "Gêneros": [movie_info.at[m, "genres"].replace("|", ", ") for m in movie_ids],
        }
    )
    for col, values in (extra or {}).items():
        df[col] = values
    df.index = np.arange(1, len(df) + 1)
    return df


# ---------------------------------------------------------------- sidebar
st.sidebar.title("🎬 Recomendador")
model_name = st.sidebar.radio("Modelo", list(MODELS), index=1)
n_recs = st.sidebar.slider("Nº de recomendações", 5, 30, 10)

user_options = [NEW_USER_ID] + ds.user_ids.tolist()
user_id = st.sidebar.selectbox(
    "Usuário",
    user_options,
    format_func=lambda u: "🆕 Novo usuário (sem histórico)" if u == NEW_USER_ID else f"Usuário {u}",
    index=1,
)
st.sidebar.caption(
    "Dica: usuários 1, 15, 68, 414 e 599 são os exemplos usados no relatório."
)

tab_rec, tab_data, tab_eval = st.tabs(["🍿 Histórico e recomendações", "📊 Dataset", "🧪 Avaliação"])

# ---------------------------------------------------------------- aba 1
with tab_rec:
    profile = user_profile(user_id)
    added = load_user_ratings(user_id)
    user_row = ds.user_vector(profile)

    label = "Novo usuário" if user_id == NEW_USER_ID else f"Usuário {user_id}"
    st.header(label)
    c1, c2, c3 = st.columns(3)
    c1.metric("Filmes avaliados", len(profile))
    c2.metric("Nota média", f"{np.mean(list(profile.values())):.2f}" if profile else "—")
    c3.metric("Avaliados nesta interface", len(added))

    col_hist, col_rec = st.columns([1, 1.3], gap="large")

    with col_hist:
        st.subheader("Histórico")
        if not profile:
            st.info("Este usuário ainda não avaliou nenhum filme.")
        else:
            hist = sorted(profile.items(), key=lambda t: -t[1])
            ids = [m for m, _ in hist]
            st.dataframe(
                movies_table(ids, {"Nota": [r for _, r in hist],
                                   "Origem": ["interface" if m in added else "MovieLens" for m in ids]}),
                height=420,
                column_config={"Nota": st.column_config.NumberColumn(format="%.1f ⭐")},
            )

    with col_rec:
        st.subheader("Recomendações")
        cold_start = user_row.nnz == 0
        model = models["Popularidade"] if cold_start else models[model_name]
        if cold_start:
            st.warning(
                "**Cold start:** sem histórico não há como aplicar filtragem colaborativa. "
                "Mostrando os filmes mais populares e bem avaliados. Avalie alguns filmes "
                "abaixo para receber recomendações personalizadas."
            )
        rec = model.recommend(user_row, n_recs)
        rec_ids = ds.item_ids[rec]
        extra = {"Nota prevista": models[model_name if not cold_start else "Popularidade"].predict(user_row, rec)}
        if model_name == "Item-KNN" and not cold_start:
            extra["Porque você avaliou"] = [
                "; ".join(title_of[ds.item_ids[i]] for i in model.explain(user_row, j, 2))
                for j in rec
            ]
        st.dataframe(
            movies_table(rec_ids, extra),
            height=420,
            column_config={"Nota prevista": st.column_config.NumberColumn(format="%.2f")},
        )
        assert not set(rec_ids) & set(profile), "recomendação contém filme já avaliado"
        st.caption(
            f"Modelo: **{model.name}**. Filmes já avaliados pelo usuário são sempre excluídos."
        )

    st.divider()
    st.subheader("⭐ Avaliar um filme")
    st.caption(
        "A avaliação é salva em `data/user_ratings.csv` e entra no perfil do usuário na "
        "hora (fold-in), sem retreinar o modelo. Digite para buscar pelo título."
    )
    catalog = [int(m) for m in ds.item_ids]
    # sugere primeiro os filmes recomendados, depois o resto do catálogo
    options = list(dict.fromkeys([int(m) for m in rec_ids] + sorted(catalog, key=lambda m: title_of[m])))
    with st.form("rate", clear_on_submit=True):
        fc1, fc2, fc3 = st.columns([3, 1.2, 0.8])
        movie = fc1.selectbox("Filme", options, format_func=lambda m: title_of[m])
        score = fc2.select_slider("Nota", options=[x / 2 for x in range(1, 11)], value=4.0)
        fc3.write("")
        submitted = fc3.form_submit_button("Salvar", use_container_width=True)
    if submitted:
        save_user_rating(user_id, movie, score)
        st.toast(f"Avaliação salva: {title_of[movie]} → {score} ⭐")
        st.rerun()

    if added and st.button("🗑️ Apagar avaliações feitas na interface para este usuário"):
        clear_user_ratings(user_id)
        st.rerun()

# ---------------------------------------------------------------- aba 2
with tab_data:
    st.header("MovieLens — ml-latest-small")
    st.markdown(
        "Dataset do [GroupLens](https://grouplens.org/datasets/movielens/latest/): "
        "avaliações de 0,5 a 5 estrelas feitas por usuários do site MovieLens (1996–2018)."
    )
    raw_stats = dataset_stats(ratings)
    used_stats = dataset_stats(filter_ratings(ratings))
    stats_df = pd.DataFrame({"Bruto": raw_stats, "Após filtro (≥5 aval./filme)": used_stats})
    st.dataframe(stats_df.style.format("{:,.4f}", subset=pd.IndexSlice[["esparsidade", "nota_media"], :]))

    d1, d2 = st.columns(2)
    with d1:
        st.subheader("Distribuição das notas")
        st.bar_chart(ratings["rating"].value_counts().sort_index(), x_label="nota", y_label="avaliações")
        st.subheader("Avaliações por filme (cauda longa)")
        per_movie = ratings.groupby("movieId").size()
        bins = pd.cut(per_movie, [0, 1, 4, 10, 50, 100, 400], labels=["1", "2–4", "5–10", "11–50", "51–100", "100+"])
        st.bar_chart(bins.value_counts().sort_index(), x_label="nº de avaliações", y_label="filmes")
    with d2:
        st.subheader("Gêneros mais avaliados")
        g = ratings.merge(movies, on="movieId")["genres"].str.split("|").explode()
        st.bar_chart(g.value_counts().head(15), horizontal=True)
        st.subheader("Avaliações por usuário")
        per_user = ratings.groupby("userId").size()
        ub = pd.cut(per_user, [0, 30, 50, 100, 200, 500, 3000], labels=["20–30", "31–50", "51–100", "101–200", "201–500", "500+"])
        st.bar_chart(ub.value_counts().sort_index(), x_label="nº de avaliações", y_label="usuários")

    st.subheader("Filmes mais populares (média bayesiana)")
    pop: PopularityRecommender = models["Popularidade"]
    top = np.argsort(-pop.scores(None))[:15]
    st.dataframe(
        movies_table(ds.item_ids[top], {"Avaliações": pop.item_count[top].astype(int),
                                        "Média": pop.item_mean[top].round(2)})
    )

# ---------------------------------------------------------------- aba 3
with tab_eval:
    st.header("Avaliação offline com usuários de teste")
    st.markdown(
        "Para cada usuário, **20% das avaliações foram escondidas** (teste). Os modelos "
        "treinam com o resto e recomendam 10 filmes; um acerto é um filme do teste com "
        "nota ≥ 4. Detalhes em `src/evaluate.py` e no README."
    )
    metrics_file = RESULTS_DIR / "metrics.csv"
    if st.button("▶️ Rodar avaliação novamente (~1 min)"):
        from src.evaluate import run

        with st.spinner("Avaliando..."):
            run(verbose=False)
    if metrics_file.exists():
        m = pd.read_csv(metrics_file).set_index("modelo")
        st.dataframe(m.style.format("{:.4f}", subset=m.columns[:-1]).highlight_max(
            subset=["Precision@10", "Recall@10", "NDCG@10", "Cobertura"], color="#2e7d3240"
        ).highlight_min(subset=["RMSE"], color="#2e7d3240"))
        main = m.loc[list(MODELS), ["Precision@10", "Recall@10", "NDCG@10"]]
        st.bar_chart(main.T, stack=False)
        examples = RESULTS_DIR / "examples.md"
        if examples.exists():
            with st.expander("Exemplos para 5 usuários de teste"):
                st.markdown(examples.read_text(encoding="utf-8"))
    else:
        st.info("Ainda não há resultados. Clique no botão acima ou rode `python -m src.evaluate`.")
