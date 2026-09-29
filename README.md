# ⚽ Scouting e Recomendação de Jogadores de Futebol

Sistema de recomendação para **scouting de futebol masculino**, com duas camadas:

1. **Filtragem colaborativa** (Item-KNN, SVD) sobre uma base real de interações **clube × jogador** (Transfermarkt): "clubes com histórico de elenco parecido com o seu tiveram estes jogadores".
2. **KNN por métricas de desempenho** (FBref 2024/25): "encontre os volantes mais parecidos com o Zubimendi que jogam na Serie A, têm até 24 anos e estão no top 30% em interceptações".

As duas se combinam num modelo **híbrido**. A interface é em **Streamlit**.

> 🎥 **Vídeo de demonstração:** _adicionar link aqui_

---

## 1. Objetivo

Ajudar o departamento de scouting de um clube a encontrar jogadores que ele **ainda não teve** e que se encaixam no seu perfil. O sistema deve:

- usar uma **base de interações usuário × item**: aqui, **clube (usuário) × jogador (item)**;
- aplicar **filtragem colaborativa** e gerar recomendações **sem jogadores que já passaram pelo clube**;
- permitir buscar **jogadores estatisticamente similares** a uma referência, com filtros por posição, liga, clube, idade, valor e limites de métricas;
- oferecer uma interface para consultar históricos e recomendações e para **avaliar jogadores** (shortlist do olheiro);
- ser **avaliado com clubes de teste**, usando contratações reais.

## 2. Fundamentação

| Conceito | Neste projeto |
|---|---|
| **Filtragem colaborativa com feedback implícito** | Não existem "notas" de clubes para jogadores. A interação é *ter atuado pelo clube*, com peso `log(1 + jogos completos) · 0,8^anos desde a última partida`: quem jogou mais e mais recentemente pesa mais. |
| **Item-KNN** | Dois jogadores são parecidos se passaram pelos **mesmos clubes** (cosseno entre as colunas da matriz). Pontuação para o clube *c*: `Σᵢ sim(j,i)·w_ci`. Captura **rotas de transferência** (clubes que compram e vendem entre si) e ex-companheiros. |
| **SVD (fatoração de matrizes)** | `R ≈ U·Σ·Vᵀ` com 256 fatores latentes, que funcionam como "mercados" (liga, país, patamar do clube). Usa *fold-in* para clubes novos: `z = x·V`, `score = z·Vᵀ`. |
| **Conteúdo (KNN por métricas)** | Cada jogador vira um vetor de métricas por 90 minutos, padronizado (z-score) dentro do seu grupo de posição e ponderado por posição. Similaridade = distância euclidiana ponderada `Σ wₖ (zₖ − z'ₖ)²`. |
| **Híbrido** | `α·CF + (1−α)·Conteúdo`, com as pontuações normalizadas (min-max) e α = 0,75. |
| **Cold start** | Um clube sem histórico não tem vizinhos. O sistema recomenda por **popularidade** (valor de mercado) e sugere a busca por jogador de referência, que não precisa de histórico. |
| **Camada de decisão** | Regras aplicadas **antes** do KNN e da recomendação (`ScoutFilter`): posição, liga, clube, idade, valor máximo, mínimo por 90 e "top X% do grupo" em uma métrica. |

## 3. Dados

### 3.1 Fontes usadas

| Dataset | Conteúdo | Uso |
|---|---|---|
| [Football Players Stats 2024-2025](https://www.kaggle.com/datasets/hubertsidorowicz/football-players-stats-2024-2025) (Kaggle, dados do FBref/Opta) | 2.854 linhas × 267 colunas: métricas avançadas das 5 grandes ligas (PrgP, PrgC, Int, TklW, xG, xAG, SCA, duelos aéreos...) | vetores de conteúdo do KNN |
| [Football Data from Transfermarkt](https://www.kaggle.com/datasets/davidcariboo/player-scores) (Kaggle) | 1,9 milhão de aparições (jogador × clube × jogo × minutos), transferências, posição detalhada (`sub_position`), valores de mercado | matriz clube × jogador, conjunto de teste, posição, valor |

Os dois são baixados automaticamente com `kagglehub`, sem precisar de login.

**Por que a temporada 2024/25?** Em **20/01/2026** a Opta encerrou o fornecimento de dados avançados ao FBref. Com isso, xG, passes progressivos e SCA deixaram de ser atualizados. A 2024/25 é a última temporada completa com esses dados, e o script confirma que as colunas estão 100% preenchidas.

### 3.2 Preparação ([src/data.py](src/data.py), [src/matching.py](src/matching.py))

1. **FBref:** 152 jogadores trocaram de clube durante a temporada e têm uma linha por clube. As contagens são somadas, e o resultado são 2.699 jogadores. A coluna `Blocks` do CSV é de *passes bloqueados*; os bloqueios defensivos estão em `Blocks_stats_defense`.
2. **Cruzamento FBref ↔ Transfermarkt:** os dois sites não compartilham IDs, então o cruzamento é feito em 3 etapas:
   1. nome normalizado + ano de nascimento;
   2. `rapidfuzz` ≥ 88 + mesmo ano + mesma liga;
   3. nome parcial + mesmo ano + mesmo clube.

   Resultado: **99,7% cruzados**, incluindo apelidos como *Sávio ↔ Savinho*, *Obite N'Dicka ↔ Evan Ndicka* e *Valentín ↔ Taty Castellanos*.
3. **Grupos de posição** pelo `sub_position` do Transfermarkt, que é mais fino que o `Pos` do FBref: ZAG (zagueiro), LAT (lateral), **VOL (volante = Defensive Midfield)**, MEI (meia), PON (ponta), ATA (centroavante). Goleiros ficam fora, porque suas métricas são outras.
4. **Universo scoutável:** jogadores de linha com **≥ 900 minutos** (≈ 10 jogos completos), totalizando **1.451 jogadores**.
5. **Matriz clube × jogador** (treino, antes de 01/07/2025): **1.098 clubes × 26.784 jogadores, 53.498 interações** (esparsidade de 99,82%). Inclui todas as ligas do Transfermarkt (Brasileirão, Portugal, Holanda...), o que dá mais co-ocorrências. As recomendações se restringem ao universo scoutável.

## 4. Método

### 4.1 Pipeline do KNN por métricas ([src/features.py](src/features.py))

1. **Corte de minutos:** o pool tem ≥ 900 min. O jogador de referência entra mesmo abaixo do corte, com um aviso de "amostra pequena". Sem esse corte, alguém com 90 min e 1 gol teria 1 gol/90 e distorceria os vizinhos.
2. **Por 90 minutos:** `contagem / (minutos / 90)`, para que minutos jogados não definam a similaridade.
3. **Percentuais** (% passes certos, % duelos aéreos, % finalizações no alvo) são recalculados e **encolhidos** para a média do grupo: `(acertos + p̄·30) / (tentativas + 30)`. Um zagueiro com 2 de 2 duelos não vira "100%".
4. **Z-score dentro do grupo de posição**, com média e desvio calculados só no pool elegível e valores cortados em ±3. Um volante é comparado com volantes.
5. **Pesos por posição:** `Z · √w`, de modo que a distância euclidiana vira euclidiana ponderada.

| Grupo | Métricas (peso) |
|---|---|
| ZAG | Interceptações (1,5), Desarmes certos (1,5), Duelos aéreos ganhos (1,5), Bloqueios, Cortes, % aéreos, Passes progressivos, % passes (1), Distância progressiva (0,75) |
| LAT | Conduções progressivas (1,5), Passes progressivos (1,25), Desarmes (1,25), Cruzamentos para a área, xAG, Interceptações, Dribles (1), Passes-chave, Recuperações (0,75) |
| **VOL** | **Passes progressivos (2), Interceptações (2), Desarmes (1,5), Recuperações (1,5)**, % passes, Passes para o terço final (1), Bloqueios, Aéreos (0,75), Conduções (0,5) |
| MEI | Passes progressivos, Passes-chave, xAG (1,5), SCA (1,25), Passes para a área, Recepções progressivas, Dribles (1), npxG (0,75), Desarmes (0,5) |
| PON | Dribles, Conduções progressivas (1,5), Recepções progressivas, xAG, npxG (1,25), Finalizações, SCA (1), Cruzamentos para a área, Toques na área (0,75) |
| ATA | npxG (2), Finalizações, Toques na área (1,5), % no alvo, xAG, Aéreos, Recepções progressivas (1), SCA (0,75) |

Os pesos podem ser ajustados com sliders na interface.

**Diferença de nível entre ligas.** Comparar um jogador do Brasileirão com um da Premier League exige cuidado. O sistema tem três modos:

- **Padrão:** z-score com todas as ligas juntas. Compara **estilo e intensidade**.
- **Z-score dentro da liga:** cada jogador é comparado com a média do seu campeonato (percentil relativo).
- **Ajuste de nível:** multiplica as **métricas de produção ofensiva** (npxG, xAG, SCA, passes-chave, passes progressivos, finalizações) por um fator de liga derivado do **coeficiente UEFA** (Inglaterra 1,00; Itália 0,85; Espanha 0,80; Alemanha 0,79; França 0,67; [fonte](https://en.wikipedia.org/wiki/UEFA_coefficient), 5 temporadas 2022/23–2026/27). As métricas defensivas não são ajustadas, porque dependem mais do estilo do time que do nível.

Para incluir o Brasileirão, basta acrescentar os dados e o fator da liga em [src/config.py](src/config.py).

**Distância:** euclidiana (padrão; considera perfil **e** intensidade) ou cosseno (só perfil: um jogador "igual, mas em menor volume" fica próximo).

### 4.2 Camada de decisão + KNN ([src/scouting.py](src/scouting.py))

```python
from src.data import build_players
from src.scouting import ScoutFilter, find_similar

players = build_players()
ref = players.index[players.Player == "Martín Zubimendi"][0]
filt = ScoutFilter(leagues=["Serie A", "Ligue 1"], age=(18, 24),
                   max_value_eur=30e6, top_pct={"Int": 30}, min_metrics={"PrgP": 4.0})
ranking, X, Zw = find_similar(players, ref, n=10, filt=filt)
```

A `ScoutFilter` elimina candidatos **antes** do `NearestNeighbors`. O top X% é calculado sobre o grupo inteiro, não só sobre quem sobrou. `similaridade_%` indica de quantos % do grupo o candidato está mais perto do que o resto ("mais parecido que 99% dos volantes").

### 4.3 Filtragem colaborativa e híbrido ([src/models.py](src/models.py))

Todos os modelos compartilham `recommend(linha_do_clube, k)`. Os jogadores que já passaram pelo clube e os que estão fora do universo scoutável ou dos filtros recebem −∞, e a avaliação confere isso com um `assert`.

- **Popularidade:** jogadores mais valorizados. É o baseline e a resposta de cold start.
- **Item-KNN:** cosseno entre as colunas da matriz, mantendo os 1.000 vizinhos de cada jogador.
- **SVD:** `TruncatedSVD` com 256 fatores.
- **Conteúdo:** proximidade entre o vetor de métricas do candidato e o **perfil médio dos jogadores do histórico do clube** na mesma posição, ponderado pelo peso na matriz. Por usar só dados anteriores ao corte, não há vazamento, e a shortlist desloca esse perfil.
- **Híbrido:** 0,75·Item-KNN + 0,25·Conteúdo.

### 4.4 Protocolo de avaliação ([src/evaluate.py](src/evaluate.py))

- **Divisão temporal**, que é a situação real do scouting: o modelo vê o passado e tenta prever o futuro.
  - **Treino:** aparições antes de **01/07/2025**.
  - **Teste:** contratações reais dos clubes das 5 grandes ligas entre **01/07/2025 e 02/02/2026** (janelas de verão e de janeiro), de jogadores do universo scoutável que **nunca** tinham passado pelo clube. Voltas de empréstimo são excluídas.
  - Ficam **109 clubes de teste e 323 contratações**.
- **Hiperparâmetros** escolhidos numa janela de **validação** separada (treino até 01/07/2024, contratações até 03/02/2025), sem olhar o teste: `python -m src.evaluate --tune`.
- **Métricas @20:**
  - **Precision@20:** fração das 20 recomendações que o clube contratou.
  - **Recall@20:** fração das contratações do clube que estavam no top-20.
  - **NDCG@20:** como o Recall, mas premia acertos no topo.
  - **HitRate@20:** % de clubes com pelo menos um acerto.
  - **Cobertura:** % do universo scoutável que aparece em alguma recomendação.
- **Validação do KNN por métricas:** sem informar a posição (todas as métricas, peso 1), mede quantos dos 10 vizinhos de cada jogador têm a mesma posição. Se as métricas descrevem bem o papel em campo, os vizinhos devem ser da mesma posição.

## 5. Resultados

Reproduza com `python -m src.evaluate`. A saída fica em [results/](results/).

### 5.1 Recomendação para clubes (teste: 2025/26)

| Modelo | Precision@20 | Recall@20 | NDCG@20 | HitRate@20 | Cobertura |
|---|---:|---:|---:|---:|---:|
| Aleatório (média de 20 sorteios) | 0,0018 | 0,011 | 0,0056 | 3,5% | 78% |
| Popularidade (baseline) | 0,0009 | 0,003 | 0,0016 | 0,9% | 1,7% |
| Item-KNN | 0,0064 | **0,064** | 0,0211 | 11,9% | **63%** |
| SVD | 0,0055 | 0,047 | 0,0218 | 10,1% | 35% |
| Conteúdo | 0,0046 | 0,029 | 0,0126 | 9,2% | 39% |
| **Híbrido** | **0,0073** | 0,056 | **0,0245** | **12,8%** | 51% |

- **Os números absolutos são baixos, e isso é esperado.** Cada clube contratou em média 3 jogadores de um universo de 1.451, e contratações dependem de preço, agente e vontade do jogador, que não estão nos dados. O que importa é a comparação com as referências.
- **A filtragem colaborativa acerta 3,5 a 4,5 vezes mais que o acaso:** o Híbrido tem NDCG 4,4× o aleatório e HitRate 3,6×. As "rotas de transferência" existem e são aprendíveis: o Rennes compra da Ligue 1, a Fiorentina e a Cremonese do mercado italiano.
- **Popularidade é pior que o acaso.** Recomendar Yamal, Mbappé e Haaland para todos os clubes quase nunca acerta. Esse baseline ruim é, por si só, um achado: no scouting, "o melhor jogador" não é a recomendação certa para cada clube.
- **O Híbrido foi o melhor em Precision, NDCG e HitRate, e o Item-KNN no Recall.** A diferença entre os dois é pequena (14 × 13 clubes com acerto) e, com 109 clubes, não é estatisticamente conclusiva.
- **O conteúdo sozinho tem HitRate 2,6× o do acaso.** Clubes tendem a contratar jogadores com perfil estatístico parecido com o dos que já tiveram, mas esse sinal é mais fraco que o das redes de transferência.

**Honestidade sobre a validação:** na janela de validação, o Híbrido com α = 0,75 (NDCG 0,029) ficou **abaixo** do Item-KNN puro (0,038). O α = 0,75 foi o melhor entre as misturas de verdade (α < 1). Além disso, a validação favorece o conteúdo: as métricas FBref de 2024/25 dos jogadores contratados em 2024 já refletem o estilo do clube novo, um vazamento que não existe no teste. Isso pesou contra a mistura e mesmo assim não se refletiu no teste. Com os dados de 2023/24, a validação seria limpa.

### 5.2 Validação do KNN por métricas

Sem informar a posição, **70% dos 10 vizinhos de cada jogador são do mesmo grupo de posição** (o acaso daria 18%) e **55% são da mesma sub-posição** (acaso: 13%). As métricas "descobrem" a posição sozinhas.

Por grupo: zagueiros 90%, centroavantes 79%, laterais 69%, pontas 66%, meias 58% e **volantes 45%**. Volantes se confundem com meias e zagueiros, o que é esperado e justifica os **pesos por posição**.

Exemplos (grupo e pesos padrão):

| Referência | 3 mais similares |
|---|---|
| Martín Zubimendi (VOL) | Enzo Barrenechea, Marten de Roon, Milan Badelj |
| Erling Haaland (ATA) | Patrik Schick, Ermedin Demirović, Moise Kean |
| Lamine Yamal (PON) | Michael Olise, Désiré Doué, Bukayo Saka |
| Declan Rice (MEI) | Joan Jordán, Nicolò Fagioli, Răzvan Marin |

### 5.3 Exemplos para cinco clubes

Tabelas completas (histórico, top-20 de 3 modelos e ✅ nos acertos) em [results/examples.md](results/examples.md). Os 4 primeiros são os clubes com mais acertos do Híbrido; o Arsenal entra como contraexemplo.

| Clube | Contratações reais (amostra) | Acertos top-20 (Pop / KNN / Híbrido) | Observação |
|---|---|---|---|
| Stade Rennais | Rongier, Mahdi Camara, Embolo, Merlin | 0 / 1 / **2** | Rongier (Marseille) e Camara (Brest): mercado francês |
| Fiorentina | Piccoli, Nicolussi Caviglia, Fazzini | 0 / 1 / **2** | Piccoli (Cagliari) no 11º lugar do Híbrido |
| Cremonese | Thorsby, Pezzella, Vardy, Sanabria | 0 / 1 / 1 | promovida: vive do mercado italiano de meio de tabela |
| Nottingham Forest | Bakwa, Ndoye, Kalimuendo, Savona | 0 / 0 / 1 | Bakwa (Strasbourg) no 15º lugar do Híbrido |
| Arsenal | Zubimendi, Eze, Madueke, Hincapié | 0 / 0 / 0 | clubes de elite contratam "o melhor disponível", não seguem rotas |
| 🆕 Clube novo | — | — | **Cold start:** mostra Yamal, Mbappé, Haaland (popularidade). Ao pôr Zubimendi na shortlist, passa a recomendar jogadores dos clubes por onde ele passou |

## 6. Interface (Streamlit)

```
streamlit run app.py
```

1. **🔎 Jogadores similares:**
   - escolha do jogador de referência e do grupo de comparação;
   - filtros (liga, idade, valor máximo, minutos mínimos, top X% em métricas, excluir o próprio clube);
   - sliders de peso, distância euclidiana ou cosseno, ajuste por liga e z-score por liga;
   - ranking com similaridade %;
   - **radar Plotly** em percentis do grupo (referência × até 2 similares), com os pontos em que os jogadores mais se parecem e mais diferem;
   - **mapa PCA 2D** do grupo.
2. **🏟️ Recomendação para clube:**
   - escolha do clube ou de "🆕 Novo clube" e do modelo, com filtros;
   - **histórico** (jogadores que atuaram pelo clube, com peso);
   - recomendações com a explicação "por quê" (jogadores do histórico que levaram à recomendação);
   - **shortlist** com interesse de 1 a 5, que entra na hora (*fold-in*), é salva em `data/shortlists.csv` e remove o jogador das recomendações.
3. **📊 Dataset:** cobertura, cruzamento, distribuições.
4. **🧪 Avaliação:** métricas, gráficos, validação de posição e exemplos.

A interface usa todos os dados até jul/2026. A avaliação usa o corte de 01/07/2025.

## 7. Limitações

- **Só as 5 grandes ligas e só 2024/25** no conteúdo (limite do dataset e do fim dos dados Opta no FBref). Contratações vindas de outras ligas (Portugal, Holanda, Brasil) não podem ser recomendadas: só cerca de 24% das chegadas aos clubes das 5 grandes ligas vieram do universo scoutável.
- **Métricas de uma única temporada**, sem tendência nem histórico de lesões.
- **Sem ajuste por posse de bola.** Times com pouca posse inflam métricas defensivas.
- **Matriz muito esparsa** (99,8%), porque cada jogador passa por poucos clubes. Jogadores de clubes com "exércitos de empréstimo" (ex.: Chelsea) têm muitas co-ocorrências e aparecem mais.
- **O fator de liga pelo coeficiente UEFA** mede o desempenho dos clubes na Europa, não o nível médio de cada jogador. É uma aproximação.
- **Contratações dependem de fatores fora dos dados** (preço, salário, agente, vontade do jogador), e 109 clubes de teste dão pouca confiança estatística nas diferenças entre modelos.
- **Validação com vazamento parcial** no modelo de conteúdo (ver 5.1).

Trabalhos futuros:
- incluir mais temporadas (validação sem vazamento, tendência de evolução);
- incluir mais ligas (Brasileirão, Portugal) e ajuste por posse;

---

## Como executar

Requisitos: **Python 3.10+** e internet na primeira execução (download de cerca de 240 MB via `kagglehub`, sem login).

```bash
git clone <url-do-repositorio>
cd TrabalhoOficinaDeDev1

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt

python -m src.evaluate          # baixa os dados, cruza as bases, avalia e gera results/
streamlit run app.py            # abre a interface em http://localhost:8501
python -m src.evaluate --tune   # (opcional) busca de hiperparâmetros na validação
```

Se o `kagglehub` não funcionar, baixe os dois datasets manualmente no Kaggle e extraia em `data/raw/football-players-stats-2024-2025/` e `data/raw/player-scores/`.

## Estrutura

```
├── app.py               # interface Streamlit
├── src/
│   ├── config.py        # posições, métricas, pesos, fatores de liga, datas
│   ├── data.py          # download, FBref, Transfermarkt, matriz clube × jogador, shortlist
│   ├── matching.py      # cruzamento FBref ↔ Transfermarkt
│   ├── features.py      # por 90, encolhimento, z-score por posição, pesos
│   ├── scouting.py      # ScoutFilter (decisão) + KNN de jogadores similares
│   ├── models.py        # Popularidade, Item-KNN, SVD, Conteúdo, Híbrido
│   └── evaluate.py      # divisão temporal, métricas, exemplos, validação
├── results/             # métricas e exemplos gerados pela avaliação
└── requirements.txt
```

## Referências

- Sarwar, B. et al. *Item-based Collaborative Filtering Recommendation Algorithms*. WWW, 2001.
- Hu, Y.; Koren, Y.; Volinsky, C. *Collaborative Filtering for Implicit Feedback Datasets*. ICDM, 2008.
- Cremonesi, P.; Koren, Y.; Turrin, R. *Performance of Recommender Algorithms on Top-N Recommendation Tasks*. RecSys, 2010.
- Burke, R. *Hybrid Recommender Systems: Survey and Experiments*. UMUAI, 2002.
- Sports Reference. *FBref & Stathead Data Update*, jan/2026.
