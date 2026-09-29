"""Cruzamento de jogadores entre FBref (métricas) e Transfermarkt (clubes, posição, valor).

Os dois sites não compartilham IDs, então o cruzamento é feito por nome + ano de
nascimento, em três etapas cada vez mais tolerantes:
  1. nome normalizado idêntico + mesmo ano de nascimento;
  2. nome parecido (rapidfuzz ≥ 88) + mesmo ano de nascimento + mesma liga;
  3. nome parcialmente parecido (≥ 60) + mesmo ano + MESMO CLUBE (o mapa clube FBref →
     clube Transfermarkt é aprendido com os pares das etapas anteriores).
Cada jogador do Transfermarkt é usado no máximo uma vez.
"""

import re
import unicodedata

import pandas as pd
from rapidfuzz import fuzz, process


def normalize_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z ]", " ", s.lower().replace("-", " "))
    return re.sub(r"\s+", " ", s).strip()


def match_players(fb: pd.DataFrame, tm: pd.DataFrame) -> pd.DataFrame:
    """Retorna fb com a coluna `player_id` (Transfermarkt) e `match_step` (1, 2, 3 ou NaN).

    fb: colunas Player, Born, Comp, Squad.
    tm: candidatos do Transfermarkt com player_id, name, birth_year, league (id FBref) e
        club (clube FBref mais provável é inferido na etapa 3 via club_id).
    """
    fb = fb.copy()
    tm = tm.copy()
    fb["_n"] = fb["Player"].map(normalize_name)
    tm["_n"] = tm["name"].map(normalize_name)
    fb["player_id"] = pd.NA
    fb["match_step"] = pd.NA
    used = set()

    def assign(i, pid, step):
        fb.at[i, "player_id"] = pid
        fb.at[i, "match_step"] = step
        used.add(pid)

    # etapa 1: nome exato + ano
    key = tm.groupby(["_n", "birth_year"])["player_id"].agg(list).to_dict()
    for i, r in fb.iterrows():
        cands = [p for p in key.get((r._n, r.Born), []) if p not in used]
        if len(cands) == 1:
            assign(i, cands[0], 1)

    # etapa 2: nome parecido + ano + liga
    for i, r in fb[fb.player_id.isna()].iterrows():
        pool = tm[(tm.birth_year == r.Born) & (tm.league == r.Comp) & ~tm.player_id.isin(used)]
        if pool.empty:
            continue
        best = process.extract(r._n, pool["_n"].tolist(), scorer=fuzz.token_set_ratio, limit=2)
        if best and best[0][1] >= 88 and (len(best) == 1 or best[1][1] < best[0][1]):
            assign(i, pool.iloc[best[0][2]].player_id, 2)

    # etapa 3: mesmo clube + ano, nome parcialmente parecido (apelidos, nomes compostos)
    matched = fb.dropna(subset=["player_id"]).merge(tm[["player_id", "club_id"]], on="player_id")
    club_map = matched.groupby("Squad")["club_id"].agg(lambda s: s.mode().iloc[0]).to_dict()
    for i, r in fb[fb.player_id.isna()].iterrows():
        club = club_map.get(r.Squad)
        pool = tm[(tm.birth_year == r.Born) & (tm.club_id == club) & ~tm.player_id.isin(used)]
        if pool.empty:
            continue
        best = process.extractOne(r._n, pool["_n"].tolist(), scorer=fuzz.partial_ratio)
        if best and best[1] >= 60:
            assign(i, pool.iloc[best[2]].player_id, 3)

    return fb.drop(columns="_n")
