"""Interface Streamlit do sistema de scouting e recomendação de jogadores.

Execução:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA

from src import config as C
from src.data import build_players, clear_shortlist, load_shortlist, load_transfermarkt, save_shortlist
from src.evaluate import RESULTS_DIR, build_split, make_models, recommend_for
from src.features import percentiles
from src.scouting import ScoutFilter, explain, find_similar

NEW_CLUB = -1
MODEL_NAMES = ["Híbrido", "Item-KNN", "SVD", "Conteúdo", "Popularidade"]

st.set_page_config(page_title="Scouting de Jogadores", page_icon="⚽", layout="wide")


# ---------------------------------------------------------------- carga e treino
@st.cache_resource(show_spinner="Carregando dados e treinando modelos (1ª vez ~1 min)...")
def load_everything():
    players = build_players()
    tm = load_transfermarkt()
    # na interface usamos TODOS os dados disponíveis (a avaliação usa o corte de 01/07/2025)
    cutoff = tm["appearances"]["date"].max() + pd.Timedelta(days=1)
    split = build_split(tm, players, cutoff, cutoff)
    models = make_models(split, players)
    clubs = tm["clubs"].set_index("club_id")
    tm_names = tm["players"].set_index("player_id")["name"]
    return players, split, models, clubs, tm_names, cutoff


players, split, models, clubs, tm_names, cutoff = load_everything()
ds = split.ds
by_pid = players.dropna(subset=["player_id"]).set_index(players["player_id"].dropna().astype(int))

# ---------------------------------------------------------------- cores (paleta de referência validada)
dark = getattr(st.context.theme, "type", None) == "dark"
COL = {
    "s1": "#3987e5" if dark else "#2a78d6",   # referência
    "s2": "#d95926" if dark else "#eb6834",   # vizinho 1
    "s3": "#199e70" if dark else "#1baf7a",   # vizinho 2
    "muted": "#898781",
    "grid": "#2c2c2a" if dark else "#e1e0d9",
    "ink": "#ffffff" if dark else "#0b0b0b",
    "ink2": "#c3c2b7" if dark else "#52514e",
}


def fmt_value(v) -> str:
    if pd.isna(v):
        return "—"
    return f"€{v / 1e6:.1f}M" if v >= 1e6 else f"€{v / 1e3:.0f}k"


def player_label(i: int) -> str:
    r = players.loc[i]
    return f"{r.Player} — {r.Squad} ({C.GROUP_LABEL.get(r.group, '?')}, {int(r.Min)} min)"


def radar(ref: int, others: list[int], X: pd.DataFrame, pool: pd.DataFrame) -> go.Figure:
    base = pool["Min"] >= C.MIN_MINUTES
    pct = percentiles(X, base)
    cats = [C.METRIC_LABEL.get(c, c) for c in X.columns]
    fig = go.Figure()
    for i, color in zip([ref] + others, [COL["s1"], COL["s2"], COL["s3"]]):
        vals = pct.loc[i].tolist()
        raw = X.loc[i].tolist()
        h = color.lstrip("#")
        fig.add_trace(go.Scatterpolar(
            r=vals + vals[:1], theta=cats + cats[:1], name=players.at[i, "Player"],
            line=dict(color=color, width=2), fill="toself",
            fillcolor=f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},0.12)",
            customdata=np.array(raw + raw[:1]),
            hovertemplate="%{theta}<br>percentil %{r:.0f}<br>valor %{customdata:.2f}<extra>%{fullData.name}</extra>",
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 100], tickvals=[25, 50, 75, 100], gridcolor=COL["grid"],
                                   tickfont=dict(color=COL["muted"], size=10)),
                   angularaxis=dict(gridcolor=COL["grid"], tickfont=dict(color=COL["ink2"], size=11)),
                   bgcolor="rgba(0,0,0,0)"),
        legend=dict(orientation="h", y=-0.12, font=dict(color=COL["ink2"])),
        margin=dict(l=60, r=60, t=30, b=40), height=460, paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def pca_map(ref: int, neighbors: list[int], Zw: pd.DataFrame, pool: pd.DataFrame) -> go.Figure:
    base = (pool["Min"] >= C.MIN_MINUTES) | (pool.index == ref)
    Z = Zw[base]
    xy = pd.DataFrame(PCA(2, random_state=0).fit_transform(Z.to_numpy()), index=Z.index, columns=["x", "y"])
    names = players.loc[xy.index, "Player"] + " — " + players.loc[xy.index, "Squad"]
    others = xy.index.difference([ref] + neighbors)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xy.loc[others, "x"], y=xy.loc[others, "y"], mode="markers",
                             name="Demais do grupo", text=names[others],
                             marker=dict(size=8, color=COL["muted"], opacity=0.35),
                             hovertemplate="%{text}<extra></extra>"))
    fig.add_trace(go.Scatter(x=xy.loc[neighbors, "x"], y=xy.loc[neighbors, "y"], mode="markers",
                             name="Vizinhos (KNN)", text=names[neighbors],
                             marker=dict(size=10, color=COL["s2"], line=dict(width=2, color="rgba(0,0,0,0)")),
                             hovertemplate="%{text}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[xy.at[ref, "x"]], y=[xy.at[ref, "y"]], mode="markers+text",
                             name="Referência", text=[players.at[ref, "Player"]], textposition="top center",
                             textfont=dict(color=COL["ink"]),
                             marker=dict(size=14, color=COL["s1"], symbol="diamond"),
                             hovertemplate="%{text}<extra></extra>"))
    ax = dict(showticklabels=False, zeroline=False, gridcolor=COL["grid"], title=None)
    fig.update_layout(xaxis=ax, yaxis=ax, height=460, margin=dict(l=10, r=10, t=30, b=40),
                      legend=dict(orientation="h", y=-0.08, font=dict(color=COL["ink2"])),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


tab_sim, tab_club, tab_data, tab_eval = st.tabs(
    ["🔎 Jogadores similares", "🏟️ Recomendação para clube", "📊 Dataset", "🧪 Avaliação"])

# ================================================================= aba 1: KNN por métricas
with tab_sim:
    st.header("Jogadores estatisticamente similares")
    st.caption("KNN sobre métricas por 90 minutos (FBref 2024/25), padronizadas por z-score dentro "
               "do grupo de posição e ponderadas por posição. Os filtros são aplicados antes do KNN.")

    eligible = players[players.group.notna()].sort_values("Min", ascending=False)
    default_ref = eligible.index[eligible.Player.eq("Martín Zubimendi")]
    c1, c2, c3 = st.columns([3, 1.2, 1])
    ref = c1.selectbox("Jogador de referência (digite para buscar)", eligible.index,
                       index=int(eligible.index.get_loc(default_ref[0])) if len(default_ref) else 0,
                       format_func=player_label)
    groups = list(C.GROUP_LABEL)
    group = c2.selectbox("Comparar como", groups, index=groups.index(players.at[ref, "group"]),
                         format_func=lambda g: C.GROUP_LABEL[g], key=f"grp_{ref}")
    n = c3.slider("Nº de similares", 5, 25, 10)
    if players.at[ref, "Min"] < C.MIN_MINUTES:
        st.warning(f"⚠️ {players.at[ref, 'Player']} jogou só {int(players.at[ref, 'Min'])} min: "
                   "amostra pequena, métricas por 90 instáveis.")
    real_group = players.at[ref, "group"]
    if real_group != group:
        st.info(f"{players.at[ref, 'Player']} é {C.GROUP_LABEL.get(real_group, real_group)}. "
                f"A comparação usa as métricas de {C.GROUP_LABEL[group]} e os vizinhos são desse grupo.")

    feats = C.FEATURES[group]
    with st.expander("🎛️ Filtros (camada de decisão)", expanded=True):
        f1, f2, f3, f4 = st.columns(4)
        leagues = f1.multiselect("Ligas", list(C.LEAGUE_LABEL.values()))
        age = f2.slider("Idade", 16, 40, (16, 40))
        max_val = f3.select_slider("Valor de mercado máximo", [5e6, 10e6, 20e6, 35e6, 50e6, 80e6, None],
                                   value=None, format_func=lambda v: "sem limite" if v is None else fmt_value(v))
        min_min = f4.slider("Minutos mínimos", 450, 2500, C.MIN_MINUTES, step=90)
        g1, g2, g3 = st.columns([2, 1, 1])
        top_metrics = g1.multiselect("Exigir estar no top X% do grupo em", list(feats),
                                     format_func=lambda m: C.METRIC_LABEL.get(m, m))
        top_x = g2.slider("X (%)", 5, 50, 25, step=5, disabled=not top_metrics)
        excl_same = g3.checkbox("Excluir o clube do jogador", value=True)
    with st.expander("⚖️ Pesos das métricas e opções do KNN"):
        st.caption(f"Pesos iniciais do grupo **{C.GROUP_LABEL[group]}**. Peso 0 remove a métrica.")
        wcols = st.columns(3)
        weights = {m: wcols[k % 3].slider(C.METRIC_LABEL.get(m, m), 0.0, 3.0, float(w), 0.25,
                                          key=f"w_{group}_{m}") for k, (m, w) in enumerate(feats.items())}
        weights = {m: w for m, w in weights.items() if w > 0}
        o1, o2, o3 = st.columns(3)
        metric = o1.radio("Distância", ["euclidean", "cosine"],
                          format_func=lambda m: {"euclidean": "Euclidiana (estilo + intensidade)",
                                                 "cosine": "Cosseno (só estilo/perfil)"}[m])
        league_adjust = o2.checkbox("Ajustar pelo nível da liga (coef. UEFA)",
                                    help="Multiplica métricas de produção ofensiva pelo fator da liga.")
        within_league = o3.checkbox("Z-score dentro de cada liga",
                                    help="Compara cada jogador com a média da própria liga.")

    if not weights:
        st.error("Defina peso > 0 para pelo menos uma métrica.")
        st.stop()
    filt = ScoutFilter(leagues=leagues or None, age=age, max_value_eur=max_val,
                       exclude_clubs=[players.at[ref, "Squad"]] if excl_same else None,
                       top_pct={m: top_x for m in top_metrics})
    ranking, X, Zw = find_similar(players, ref, n, filt, weights, group=group, metric=metric,
                                  min_minutes=min_min, league_adjust=league_adjust,
                                  within_league=within_league)
    if ranking.empty:
        st.info("Nenhum jogador passa nos filtros. Afrouxe algum critério.")
    else:
        show = ranking.assign(
            Liga=ranking.league, Valor=ranking.value_eur.map(fmt_value),
            Posição=ranking.sub_position, Idade=ranking.Age.astype("Int64"),
        )[["Player", "Squad", "Liga", "Posição", "Idade", "Min", "Valor", "similaridade_%"]]
        show.columns = ["Jogador", "Clube", "Liga", "Posição", "Idade", "Minutos", "Valor", "Similaridade"]
        show.index = np.arange(1, len(show) + 1)
        st.dataframe(show, column_config={"Similaridade": st.column_config.ProgressColumn(
            format="%.0f%%", min_value=0, max_value=100,
            help="% do grupo que está mais longe da referência do que este jogador")})

        pool = players.loc[X.index]
        r1, r2 = st.columns(2)
        with r1:
            st.subheader("Radar (percentil no grupo)")
            comp = st.multiselect("Comparar com (até 2)", ranking.index.tolist(), ranking.index[:2].tolist(),
                                  max_selections=2, format_func=lambda i: players.at[i, "Player"])
            st.plotly_chart(radar(ref, comp, X[list(weights)], pool), width="stretch")
            if comp:
                e = explain(Zw, ref, comp[0])
                lab = lambda ms: ", ".join(C.METRIC_LABEL.get(m, m) for m in ms)
                st.markdown(f"**{players.at[comp[0], 'Player']}** × **{players.at[ref, 'Player']}**  \n"
                            f"🟰 Mais parecidos em: {lab(e['aproxima'])}  \n"
                            f"↔️ Mais diferentes em: {lab(e['afasta'])}")
        with r2:
            st.subheader("Mapa do grupo (PCA 2D)")
            st.plotly_chart(pca_map(ref, ranking.index.tolist(), Zw, pool), width="stretch")
            st.caption("Cada ponto é um jogador do grupo; a distância no mapa aproxima a distância "
                       "usada pelo KNN (2 componentes principais do espaço ponderado).")
        with st.expander("Tabela de métricas por 90 (referência e similares)"):
            tbl = X.loc[[ref] + ranking.index.tolist(), list(weights)].round(2)
            tbl.index = players.loc[tbl.index, "Player"]
            tbl.columns = [C.METRIC_LABEL.get(c, c) for c in tbl.columns]
            st.dataframe(tbl)

# ================================================================= aba 2: filtragem colaborativa
with tab_club:
    st.header("Recomendação de jogadores para um clube")
    st.caption("Filtragem colaborativa sobre a matriz clube × jogador (Transfermarkt): clubes com "
               "histórico de elenco parecido com o seu tiveram estes jogadores. Jogadores que já "
               "passaram pelo clube nunca são recomendados.")

    big5 = clubs[clubs.domestic_competition_id.isin(C.LEAGUES.values())]
    club_opts = [NEW_CLUB] + sorted(big5.index, key=lambda c: big5.at[c, "name"])
    default_club = club_opts.index(big5.index[big5.name.eq("Stade Rennais FC")][0]) \
        if big5.name.eq("Stade Rennais FC").any() else 1
    k1, k2, k3 = st.columns([2, 1.3, 1])
    club = k1.selectbox("Clube", club_opts, index=default_club,
                        format_func=lambda c: "🆕 Novo clube / olheiro (sem histórico)" if c == NEW_CLUB
                        else clubs.at[c, "name"])
    model_name = k2.selectbox("Modelo", MODEL_NAMES)
    n_recs = k3.slider("Nº de recomendações", 5, 30, 15)

    with st.expander("🎛️ Filtros"):
        h1, h2, h3, h4 = st.columns(4)
        f_groups = h1.multiselect("Posição", list(C.GROUP_LABEL), format_func=lambda g: C.GROUP_LABEL[g])
        f_leagues = h2.multiselect("Liga", list(C.LEAGUE_LABEL.values()), key="club_leagues")
        f_age = h3.slider("Idade", 16, 40, (16, 40), key="club_age")
        f_val = h4.select_slider("Valor máximo", [5e6, 10e6, 20e6, 35e6, 50e6, 80e6, None], value=None,
                                 format_func=lambda v: "sem limite" if v is None else fmt_value(v),
                                 key="club_val")
    items = pd.DataFrame({"player_id": ds.item_ids})
    info = by_pid.reindex(ds.item_ids)
    items = items.assign(group=info.group.to_numpy(), league=info.league.to_numpy(), Squad=info.Squad.to_numpy(),
                         Age=info.Age.to_numpy(), value_eur=split.item_value)
    allowed = ScoutFilter(leagues=f_leagues or None, age=f_age, max_value_eur=f_val).mask(items, None)
    if f_groups:
        allowed = allowed & items.group.isin(f_groups)
    allowed = allowed.to_numpy(dtype=bool, copy=True)

    shortlist = load_shortlist(club)
    row = ds.user_row(club, shortlist)
    col_h, col_r = st.columns([1, 1.6], gap="large")
    with col_h:
        st.subheader("Histórico")
        if row.nnz == 0:
            st.info("Clube sem histórico.")
        else:
            w = pd.Series(row.data, index=ds.item_ids[row.indices]).sort_values(ascending=False)
            hist = split.inter[split.inter.club_id == club].set_index("player_id")
            htab = pd.DataFrame({
                "Jogador": [tm_names.get(p, p) for p in w.index],
                "Última partida": [hist["last"].get(p, pd.NaT) for p in w.index],
                "Minutos": [hist["minutes"].get(p, np.nan) for p in w.index],
                "Origem": ["shortlist ⭐" if p in shortlist else "atuou no clube" for p in w.index],
                "Peso": w.round(2).to_numpy(),
            })
            htab.index = np.arange(1, len(htab) + 1)
            st.dataframe(htab, height=480, column_config={
                "Última partida": st.column_config.DateColumn(format="MM/YYYY"),
                "Minutos": st.column_config.NumberColumn(format="%d")})
            st.caption("Peso = log(1 + jogos completos) × 0,8^anos desde a última partida.")

    with col_r:
        st.subheader("Recomendações")
        rec, used = recommend_for(models[model_name], models["Popularidade"], ds, club, n_recs,
                                  extra=shortlist, allowed=allowed)
        if row.nnz == 0:
            st.warning("**Cold start:** sem histórico não há filtragem colaborativa. Mostrando os jogadores "
                       "mais valorizados (popularidade). Adicione jogadores à shortlist abaixo ou use a aba "
                       "🔎 para buscar similares a um jogador de referência.")
        rec_ids = ds.item_ids[rec]
        rinfo = by_pid.reindex(rec_ids)
        scores = used.scores(row, club)[rec]
        rtab = pd.DataFrame({
            "Jogador": rinfo.Player.to_numpy(), "Clube": rinfo.Squad.to_numpy(), "Liga": rinfo.league.to_numpy(),
            "Posição": rinfo.sub_position.to_numpy(), "Idade": rinfo.Age.astype("Int64").to_numpy(),
            "Valor": [fmt_value(v) for v in split.item_value[rec]], "Score": scores,
        })
        knn = models["Item-KNN"]
        if used.name in ("Item-KNN", "Híbrido") and row.nnz:
            rtab["Por quê (já passaram pelo clube)"] = [
                ", ".join(str(tm_names.get(ds.item_ids[i], "?")) for i in knn.explain(row, j, 2)) or "—"
                for j in rec]
        rtab.index = np.arange(1, len(rtab) + 1)
        st.dataframe(rtab, height=480, column_config={"Score": st.column_config.NumberColumn(format="%.3f")})
        assert not set(rec) & set(row.indices), "recomendou jogador já conhecido"
        st.caption(f"Modelo: **{used.name}**.")

    st.divider()
    st.subheader("⭐ Shortlist do olheiro (avaliar jogadores)")
    st.caption("O interesse (1–5) entra como interação do clube na hora (fold-in), sem retreinar, e é "
               "salvo em `data/shortlists.csv`. Jogadores na shortlist deixam de ser recomendados.")
    pool_idx = np.where(split.pool_mask)[0]
    opts = list(dict.fromkeys([int(ds.item_ids[i]) for i in rec] +
                              sorted((int(ds.item_ids[i]) for i in pool_idx), key=lambda p: str(by_pid.at[p, "Player"]))))
    with st.form("shortlist", clear_on_submit=True):
        s1, s2, s3 = st.columns([3, 1.2, 0.8])
        pid = s1.selectbox("Jogador", opts, format_func=lambda p: f"{by_pid.at[p, 'Player']} — {by_pid.at[p, 'Squad']}")
        interest = s2.select_slider("Interesse", [1, 2, 3, 4, 5], value=4)
        s3.write("")
        if s3.form_submit_button("Adicionar", width="stretch"):
            save_shortlist(club, pid, interest)
            st.toast(f"{by_pid.at[pid, 'Player']} adicionado com interesse {interest} ⭐")
            st.rerun()
    if shortlist and st.button("🗑️ Limpar shortlist deste clube"):
        clear_shortlist(club)
        st.rerun()

# ================================================================= aba 3: dataset
with tab_data:
    st.header("Dados")
    st.markdown(
        "- **FBref, 5 grandes ligas 2024/25** ([Kaggle](https://www.kaggle.com/datasets/hubertsidorowicz/"
        "football-players-stats-2024-2025)): métricas avançadas por jogador (Opta).\n"
        "- **Transfermarkt** ([Kaggle](https://www.kaggle.com/datasets/davidcariboo/player-scores)): "
        "aparições (clube × jogador × minutos), transferências, posições e valores de mercado.")
    matched = players.player_id.notna().mean()
    pool_n = int(split.pool_mask.sum())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Jogadores FBref", f"{len(players):,}".replace(",", "."))
    m2.metric("Cruzados com o Transfermarkt", f"{matched:.1%}")
    m3.metric("Universo scoutável (≥ 900 min)", pool_n)
    m4.metric("Matriz clube × jogador", f"{ds.R.shape[0]:,} × {ds.R.shape[1]:,}".replace(",", "."),
              f"esparsidade {1 - ds.R.nnz / (ds.R.shape[0] * ds.R.shape[1]):.3%}", delta_color="off")
    d1, d2 = st.columns(2)
    scout = players[(players.Min >= C.MIN_MINUTES) & players.group.notna()]
    with d1:
        st.subheader("Universo scoutável por posição")
        st.bar_chart(scout.group.map(C.GROUP_LABEL).value_counts(), horizontal=True, color=COL["s1"])
        st.subheader("Minutos jogados (todos os jogadores)")
        bins = pd.cut(players.Min, [0, 450, 900, 1800, 2700, 3500],
                      labels=["< 450", "450–900", "900–1800", "1800–2700", "2700+"])
        st.bar_chart(bins.value_counts().sort_index(), x_label="minutos", y_label="jogadores", color=COL["s1"])
    with d2:
        st.subheader("Universo scoutável por liga")
        st.bar_chart(scout.league.value_counts(), horizontal=True, color=COL["s1"])
        st.subheader("Idade (universo scoutável)")
        st.bar_chart(scout.Age.value_counts().sort_index(), x_label="idade", y_label="jogadores", color=COL["s1"])

# ================================================================= aba 4: avaliação
with tab_eval:
    st.header("Avaliação com clubes de teste")
    st.markdown("Treino com dados **até 30/06/2025**; teste = contratações reais dos clubes das 5 grandes "
                "ligas entre **01/07/2025 e 02/02/2026** (jogadores do universo scoutável que nunca tinham "
                "passado pelo clube). Cada modelo recomenda 20 jogadores por clube.")
    if st.button("▶️ Rodar avaliação novamente (~1 min)"):
        from src.evaluate import run

        with st.spinner("Avaliando..."):
            run(verbose=False)
    mfile = RESULTS_DIR / "metrics.csv"
    if mfile.exists():
        m = pd.read_csv(mfile).set_index("modelo")
        metric_cols = [c for c in m.columns if c != "clubes"]
        st.dataframe(m[metric_cols].style.format("{:.4f}").highlight_max(color="#2e7d3240"))
        e1, e2 = st.columns(2)
        e1.subheader("NDCG@20")
        e1.bar_chart(m["NDCG@20"].sort_values(), horizontal=True, color=COL["s1"])
        e2.subheader("HitRate@20 (clubes com ≥ 1 acerto)")
        e2.bar_chart(m["HitRate@20"].sort_values(), horizontal=True, color=COL["s1"])
        vfile = RESULTS_DIR / "position_validation.csv"
        if vfile.exists():
            v = pd.read_csv(vfile).iloc[0]
            st.subheader("Validação do KNN por métricas")
            st.markdown(f"Sem informar a posição (todas as métricas, peso 1), **{v.mesmo_grupo:.0%}** dos 10 "
                        f"vizinhos de cada jogador são do mesmo grupo de posição (acaso: {v.mesmo_grupo_acaso:.0%}) "
                        f"e **{v.mesma_sub_posicao:.0%}** da mesma sub-posição (acaso: {v.mesma_sub_posicao_acaso:.0%}).")
        ex = RESULTS_DIR / "examples.md"
        if ex.exists():
            with st.expander("Exemplos para 5 clubes de teste"):
                st.markdown(ex.read_text(encoding="utf-8"))
    else:
        st.info("Ainda não há resultados. Rode `python -m src.evaluate`.")
