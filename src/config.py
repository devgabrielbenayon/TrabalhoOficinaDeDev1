"""Configurações do domínio: posições, métricas, pesos, ligas e datas da avaliação."""

import pandas as pd

# ---------------------------------------------------------------- datasets (Kaggle)
FBREF_DATASET = "hubertsidorowicz/football-players-stats-2024-2025"
FBREF_FILE = "players_data-2024_2025.csv"
TM_DATASET = "davidcariboo/player-scores"

# ---------------------------------------------------------------- ligas
# nome da liga no FBref -> id da competição no Transfermarkt
LEAGUES = {
    "eng Premier League": "GB1",
    "es La Liga": "ES1",
    "it Serie A": "IT1",
    "de Bundesliga": "L1",
    "fr Ligue 1": "FR1",
}
LEAGUE_LABEL = {
    "eng Premier League": "Premier League",
    "es La Liga": "La Liga",
    "it Serie A": "Serie A",
    "de Bundesliga": "Bundesliga",
    "fr Ligue 1": "Ligue 1",
}
# Fator de nível da liga (opcional, aplicado às métricas de produção ofensiva).
# Coeficiente UEFA de associações (5 temporadas, 2022/23–2026/27, consultado em
# 17/09/2026 em https://en.wikipedia.org/wiki/UEFA_coefficient) normalizado pela
# Inglaterra: ENG 103,796 | ITA 88,589 | ESP 83,243 | GER 81,545 | FRA 69,153.
# Serve de "gancho" para comparar ligas de níveis diferentes (ex.: Brasileirão),
# bastando incluir o fator da nova liga aqui.
LEAGUE_FACTOR = {
    "eng Premier League": 1.00,
    "it Serie A": 0.85,
    "es La Liga": 0.80,
    "de Bundesliga": 0.79,
    "fr Ligue 1": 0.67,
}

# ---------------------------------------------------------------- posições
# sub_position do Transfermarkt -> grupo de posição usado no KNN
POSITION_GROUPS = {
    "Centre-Back": "ZAG",
    "Left-Back": "LAT",
    "Right-Back": "LAT",
    "Defensive Midfield": "VOL",
    "Central Midfield": "MEI",
    "Attacking Midfield": "MEI",
    "Left Winger": "PON",
    "Right Winger": "PON",
    "Left Midfield": "PON",
    "Right Midfield": "PON",
    "Centre-Forward": "ATA",
    "Second Striker": "ATA",
}
GROUP_LABEL = {
    "ZAG": "Zagueiro",
    "LAT": "Lateral",
    "VOL": "Volante",
    "MEI": "Meia",
    "PON": "Ponta",
    "ATA": "Centroavante",
}

# ---------------------------------------------------------------- métricas
# nome amigável de cada coluna do FBref (usado na interface e no radar)
METRIC_LABEL = {
    "PrgP": "Passes progressivos",
    "PrgC": "Conduções progressivas",
    "PrgR": "Recepções progressivas",
    "PrgDist": "Distância progressiva de passe",
    "Int": "Interceptações",
    "TklW": "Desarmes certos",
    "Recov": "Recuperações de bola",
    "Blocks": "Bloqueios",
    "Clr": "Cortes",
    "Won": "Duelos aéreos ganhos",
    "Won%": "% duelos aéreos",
    "Cmp%": "% passes certos",
    "1/3": "Passes para o terço final",
    "KP": "Passes-chave",
    "PPA": "Passes para a área",
    "CrsPA": "Cruzamentos para a área",
    "xAG": "xAG (assistência esperada)",
    "npxG": "npxG (gol esperado sem pênalti)",
    "SCA": "Ações que criam finalização",
    "Succ": "Dribles certos",
    "Sh": "Finalizações",
    "SoT%": "% finalizações no alvo",
    "Att Pen": "Toques na área adversária",
}

# grupo -> {métrica: peso}. Pesos maiores = mais importantes para a similaridade.
FEATURES = {
    "ZAG": {"Int": 1.5, "TklW": 1.5, "Won": 1.5, "Blocks": 1.0, "Clr": 1.0, "Won%": 1.0,
            "PrgP": 1.0, "Cmp%": 1.0, "PrgDist": 0.75},
    "LAT": {"PrgC": 1.5, "PrgP": 1.25, "TklW": 1.25, "CrsPA": 1.0, "xAG": 1.0, "Int": 1.0,
            "Succ": 1.0, "KP": 0.75, "Recov": 0.75},
    "VOL": {"PrgP": 2.0, "Int": 2.0, "TklW": 1.5, "Recov": 1.5, "Cmp%": 1.0, "1/3": 1.0,
            "Blocks": 0.75, "Won": 0.75, "PrgC": 0.5},
    "MEI": {"PrgP": 1.5, "KP": 1.5, "xAG": 1.5, "SCA": 1.25, "PPA": 1.0, "PrgR": 1.0,
            "Succ": 1.0, "npxG": 0.75, "TklW": 0.5},
    "PON": {"Succ": 1.5, "PrgC": 1.5, "PrgR": 1.25, "xAG": 1.25, "npxG": 1.25, "Sh": 1.0,
            "SCA": 1.0, "CrsPA": 0.75, "Att Pen": 0.75},
    "ATA": {"npxG": 2.0, "Sh": 1.5, "Att Pen": 1.5, "SoT%": 1.0, "xAG": 1.0, "Won": 1.0,
            "PrgR": 1.0, "SCA": 0.75},
}
# espaço "GERAL": todas as métricas com peso 1 (validação: as métricas descobrem a posição?)
FEATURES["GERAL"] = {m: 1.0 for g in list(FEATURES) for m in FEATURES[g]}

# métricas percentuais -> (numerador, denominador) para recalcular e encolher
RATE_COLS = {
    "Cmp%": ("Cmp", "Att"),
    "Won%": ("Won", "AerAtt"),
    "SoT%": ("SoT", "Sh"),
}
# métricas de produção ofensiva: as únicas ajustadas pelo nível da liga
OUTPUT_COLS = {"npxG", "xAG", "SCA", "KP", "PPA", "PrgP", "Sh"}

MIN_MINUTES = 900  # ≈ 10 jogos completos
SHRINK_K = 30      # tentativas "fictícias" no encolhimento bayesiano dos percentuais
Z_CLIP = 3.0

# ---------------------------------------------------------------- filtragem colaborativa
SPLIT_DATE = pd.Timestamp("2025-07-01")   # treino: tudo antes; teste: contratações depois
TEST_END = pd.Timestamp("2026-02-02")     # fim da janela de janeiro/2026
VALID_SPLIT = pd.Timestamp("2024-07-01")  # validação (escolha de hiperparâmetros)
VALID_END = pd.Timestamp("2025-02-03")
DECAY_PER_YEAR = 0.8                      # peso de interações antigas cai 20% por ano
K = 20
