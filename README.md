# 🎬 Sistema de Recomendação de Filmes com Filtragem Colaborativa

Sistema de recomendação de filmes construído sobre o dataset **MovieLens (ml-latest-small)**. Ele usa **filtragem colaborativa** (Item-KNN e SVD) e é comparado com um baseline de popularidade. A interface é feita em **Streamlit**.

> 🎥 **Vídeo de demonstração:** _adicionar link aqui_

---

## 1. Objetivo

Recomendar a cada usuário filmes que ele **ainda não viu** e que provavelmente vai gostar, usando só o histórico de avaliações de todos os usuários (quem gostou do mesmo que você também gostou de...). O sistema deve:

- usar uma base de interações usuário × item;
- preparar os dados e implementar filtragem colaborativa;
- gerar recomendações personalizadas **sem itens já conhecidos** pelo usuário;
- oferecer uma interface para consultar históricos e recomendações e para avaliar filmes;
- ser avaliado com usuários de teste e métricas adequadas.

## 2. Fundamentação

**Filtragem colaborativa (FC)** recomenda com base em padrões de comportamento coletivo, sem olhar o conteúdo dos itens (gênero, sinopse etc.). Ela parte de uma matriz **R** (usuários × itens), em que `r_ui` é a nota do usuário *u* para o filme *i*. Essa matriz é muito esparsa: aqui, 98,3% das células estão vazias.

| Abordagem | Ideia | Implementação neste projeto |
|---|---|---|
| **Baseada em memória (Item-KNN)** | Dois filmes são parecidos se foram avaliados de forma parecida pelos mesmos usuários. Recomenda filmes parecidos com os que o usuário já avaliou bem. | Similaridade de **cosseno** entre as colunas de R, mantendo os **k = 50** vizinhos de cada filme. Pontuação: `score(u,j) = Σᵢ sim(j,i)·r_ui` |
| **Baseada em modelo (SVD)** | Aproxima R por poucos **fatores latentes** (`R ≈ U·Σ·Vᵀ`). Cada fator representa um "gosto" escondido (ex.: ação/aventura, filmes cult). | `TruncatedSVD` com **50 fatores**. Para qualquer usuário: `z = x·V`, `score = z·Vᵀ` (*fold-in*) |
| **Baseline: Popularidade** | Recomenda os filmes mais populares e bem avaliados, sem personalização. | Média bayesiana `(C·m + Σnotas)/(C + n)` × `log(1+n)` |

Conceitos importantes:

- **Centralização pela média do usuário** (`r_ui − r̄_u`) remove o viés de usuários "generosos" ou "exigentes". É usada para **prever notas**.
- **Cold start:** um usuário sem histórico não tem vizinhos nem fatores, então a FC não se aplica. Nesse caso o sistema usa o baseline de popularidade até o usuário avaliar alguns filmes.
- **Fold-in:** o perfil de um usuário novo (ou de alguém que acabou de avaliar um filme) é projetado no modelo já treinado sem retreinar. É isso que deixa a interface responder na hora.

## 3. Dados

**Fonte:** [MovieLens Latest Small](https://grouplens.org/datasets/movielens/latest/), do GroupLens Research (Universidade de Minnesota). O script baixa o dataset automaticamente na primeira execução. Ele pode ser usado para pesquisa e ensino ([licença](https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html)).

| | Bruto | Após filtro (≥ 5 avaliações por filme) |
|---|---:|---:|
| Usuários | 610 | 610 |
| Filmes | 9.724 | 3.650 |
| Avaliações | 100.836 | 90.274 |
| Esparsidade | 98,30% | 95,95% |
| Nota média | 3,50 | 3,54 |
| Avaliações por usuário (mín. / mediana) | 20 / 70,5 | 12 / 68 |

Arquivos usados: `ratings.csv` (userId, movieId, rating 0,5–5, timestamp) e `movies.csv` (movieId, title, genres).

**Exploração** (aba *Dataset* da interface):

- As notas se concentram em 3–4 (4,0 = 26,6%, 3,0 = 19,9%), e 48% das avaliações são ≥ 4.
- **Cauda longa:** a maioria dos filmes tem pouquíssimas avaliações (3.446 filmes têm só uma), enquanto só 18 filmes têm mais de 200.
- Drama, Comédia, Ação e Thriller dominam as avaliações.

**Preparação** ([src/data.py](src/data.py)):

1. Download e leitura dos CSVs.
2. Remoção de filmes com menos de 5 avaliações. Com tão poucos dados a similaridade não é confiável, e o filtro reduz a esparsidade de 98,3% para 96,0% mantendo 90% das avaliações.
3. Mapeamento de `userId`/`movieId` para índices e montagem da matriz esparsa `scipy.sparse.csr_matrix` (610 × 3.650).

## 4. Método

### 4.1 Modelos ([src/models.py](src/models.py))

Os três modelos têm a mesma interface: `fit(R)`, `recommend(usuario, k)` e `predict(usuario, itens)`.
**Em `recommend`, os filmes já avaliados recebem pontuação −∞ antes do top-N**, então nunca são recomendados. A avaliação também verifica isso com um `assert`.

Item-KNN e SVD têm **duas "cabeças"**:

| | Ranking (top-N) | Previsão de nota (RMSE) |
|---|---|---|
| Item-KNN | cosseno sobre notas **brutas**; `Σ sim·r_ui` | cosseno sobre notas **centralizadas**; `r̄_u + Σ sim·(r_ui − r̄_u) / Σ sim` |
| SVD | SVD das notas **brutas** (PureSVD, Cremonesi et al. 2010) | SVD das notas **centralizadas** + viés do filme: `r̄_u + b_j + x̂_j` |

**Por que duas cabeças?** Nos testes, ranquear pela nota prevista foi muito pior (tabela 5.2). A nota prevista não tem noção de *confiança*: um filme obscuro com um único vizinho pode ter previsão 5,0 e vai para o topo. A matriz bruta, com 0 onde não há nota, carrega também a informação de **o que o usuário escolheu assistir**, que é um sinal forte para recomendar.

### 4.2 Protocolo de avaliação ([src/evaluate.py](src/evaluate.py))

- **Divisão treino/teste por usuário:** para cada um dos 610 usuários, 20% das avaliações (sorteio, seed 42) ficam escondidas como teste. Os modelos treinam só com os outros 80%.
- **Usuários de teste:** os 600 usuários com pelo menos um filme relevante no teste.
- **Relevante:** filme do teste com nota ≥ 4.
- Para cada usuário, o modelo gera um top-10, excluindo os filmes do treino, que é comparado com os relevantes do teste.

**Métricas:**

| Métrica | O que mede |
|---|---|
| **Precision@10** | fração das 10 recomendações que são relevantes |
| **Recall@10** | fração dos relevantes do usuário que apareceram no top-10 |
| **NDCG@10** | como Recall, mas premia acertos nas primeiras posições |
| **RMSE** | erro médio da nota prevista nos pares (usuário, filme) do teste |
| **Cobertura** | % do catálogo que aparece em pelo menos uma recomendação (diversidade) |

## 5. Resultados

Reproduza com `python -m src.evaluate`. A saída fica em [results/metrics.csv](results/metrics.csv).

### 5.1 Comparação dos modelos

| Modelo | Precision@10 | Recall@10 | NDCG@10 | RMSE | Cobertura |
|---|---:|---:|---:|---:|---:|
| Popularidade (baseline) | 0,124 | 0,103 | 0,168 | 0,946 | 1,8% |
| **Item-KNN** | **0,205** | 0,193 | **0,269** | 0,908 | **22,4%** |
| **SVD** | 0,196 | **0,208** | 0,262 | **0,859** | 18,3% |

- Os dois modelos de filtragem colaborativa superam o baseline em todas as métricas: **+65% de Precision** e **+60% de NDCG** (Item-KNN), e **+100% de Recall** (SVD).
- **Item-KNN** teve o melhor ranking no topo (Precision e NDCG) e a maior cobertura: recomenda 12× mais filmes diferentes que o baseline.
- **SVD** teve o melhor Recall e a melhor previsão de notas (RMSE 0,859).
- O baseline de popularidade é forte (Precision 0,12). Filmes populares são populares porque muita gente gosta deles. Mesmo assim ele recomenda os mesmos 10 filmes para todos (cobertura de 1,8%).

### 5.2 Análise: ranquear pela nota prevista

| Modelo | Precision@10 | Recall@10 | NDCG@10 | Cobertura |
|---|---:|---:|---:|---:|
| Item-KNN (rank por nota prevista) | 0,006 | 0,004 | 0,008 | 61,4% |
| SVD (rank por nota prevista) | 0,103 | 0,091 | 0,145 | 7,5% |

Ranquear pela nota prevista (o que minimiza o RMSE) derruba a qualidade do top-N. No Item-KNN o ranking fica praticamente inútil: ele recomenda filmes obscuros com 1–2 vizinhos e previsão alta. Isso mostra que **um RMSE baixo não garante boas recomendações** e justifica usar métricas de ranking.

### 5.3 Exemplos para cinco usuários

Tabelas completas (histórico + top-10 dos 3 modelos, com ✅ nos acertos) em [results/examples.md](results/examples.md).

| Usuário | Perfil | Acertos no top-10 (Pop / KNN / SVD) | Observação |
|---|---|---|---|
| 1 | 179 aval.; gosta de *Seven*, *Usual Suspects* | 3 / **6** / 3 | KNN acerta *Star Wars*, *Terminator*, *Back to the Future*, *Roger Rabbit* |
| 15 | 107 aval.; *Star Wars*, *Aliens*, *T2* | 3 / 3 / **4** | SVD e KNN recomendam as sequências de *Star Wars* e *LOTR* |
| 68 | 929 aval.; *Star Wars*, *Princess Bride* | 4 / **5** / 4 | SVD traz títulos menos óbvios (*Clueless*, *8 Mile*) e acerta |
| 414 | 1.648 aval.; usuário muito ativo | **10 / 10** / 6 | O SVD fica mais "de nicho" (*Pianist*, *City of God*), mas continua relevante |
| 599 | 1.316 aval.; *Ghost in the Shell*, *Dr. Strangelove* | 1 / **2** / 1 | Gosto pouco convencional; é o caso mais difícil para todos |
| 🆕 novo | sem histórico | — | **Cold start:** mostra os populares. Depois de avaliar *Toy Story* com 5, o 1º recomendado vira *Toy Story 2* |

## 6. Interface (Streamlit)

```
streamlit run app.py
```

- **Barra lateral:** escolha do modelo (Popularidade / Item-KNN / SVD), do nº de recomendações e do usuário (inclui **"Novo usuário (sem histórico)"**).
- **Histórico e recomendações:** histórico do usuário com as notas, recomendações com nota prevista e, no Item-KNN, a **explicação** ("porque você avaliou X"). Há também um formulário para **avaliar filmes**. A avaliação é salva em `data/user_ratings.csv` e as recomendações se atualizam na hora (fold-in).
- **Dataset:** estatísticas e gráficos da exploração.
- **Avaliação:** tabela e gráfico das métricas, os exemplos dos 5 usuários e um botão para rodar a avaliação de novo.

## 7. Limitações

- **Cold start:** usuários sem histórico recebem só recomendações populares, e **filmes novos** (menos de 5 avaliações) nunca são recomendados.
- **Viés de popularidade:** mesmo os modelos de FC tendem a recomendar filmes muito avaliados. A cobertura máxima foi de 22% do catálogo.
- **Dataset pequeno e antigo:** 610 usuários, avaliações de 1996–2018, e usuários do MovieLens não representam o público em geral.
- **Avaliação offline:** um filme recomendado que não está no teste conta como erro, mas o usuário talvez gostasse dele sem ainda tê-lo visto. As métricas são, portanto, estimativas pessimistas. O ideal seria um teste A/B com usuários reais.
- **Divisão aleatória** (e não temporal): o modelo pode "ver o futuro" do usuário.
- **Hiperparâmetros** (k = 50 vizinhos, 50 fatores, limiar ≥ 4) escolhidos com poucos testes e sem conjunto de validação separado.
- **Sem informação de conteúdo:** gêneros, sinopses e elenco não são usados. Um modelo híbrido ajudaria no cold start de itens.

## 8. Conclusão

A filtragem colaborativa funciona bem mesmo com uma matriz 96% vazia: Item-KNN e SVD ficaram **~60% a ~100% acima do baseline** em todas as métricas de ranking e recomendaram um catálogo muito mais diverso. O Item-KNN é simples, rápido e **explicável** ("porque você viu X"), o que ajuda na interface. O SVD prevê melhor as notas e encontra títulos menos óbvios. O principal aprendizado da análise é que **otimizar a previsão de nota (RMSE) não é o mesmo que recomendar bem**: métricas de ranking como Precision, Recall e NDCG são mais adequadas ao problema. Como trabalhos futuros, ficam a combinação dos dois modelos (híbrido), o uso de gêneros para o cold start e uma divisão temporal na avaliação.

---

## Como executar

Requisitos: **Python 3.10+** e internet na primeira execução (download de cerca de 1 MB).

```bash
git clone <url-do-repositorio>
cd TrabalhoOficinaDeDev1

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt

python -m src.evaluate      # baixa os dados, avalia os modelos e gera results/
streamlit run app.py        # abre a interface em http://localhost:8501
```

## Estrutura

```
├── app.py               # interface Streamlit
├── src/
│   ├── data.py          # download, preparação, matriz usuário×item
│   ├── models.py        # Popularidade, Item-KNN, SVD
│   └── evaluate.py      # divisão treino/teste, métricas, exemplos
├── results/
│   ├── metrics.csv      # métricas geradas pela avaliação
│   └── examples.md      # recomendações para 5 usuários de teste
├── data/raw/            # dataset (baixado automaticamente, fora do git)
└── requirements.txt
```

## Referências

- Harper, F. M.; Konstan, J. A. *The MovieLens Datasets: History and Context*. ACM TiiS, 2015.
- Sarwar, B. et al. *Item-based Collaborative Filtering Recommendation Algorithms*. WWW, 2001.
- Cremonesi, P.; Koren, Y.; Turrin, R. *Performance of Recommender Algorithms on Top-N Recommendation Tasks*. RecSys, 2010.
- Koren, Y.; Bell, R.; Volinsky, C. *Matrix Factorization Techniques for Recommender Systems*. IEEE Computer, 2009.
