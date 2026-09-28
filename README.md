# Sistema de Recomendação de Filmes por Filtragem Colaborativa

Trabalho da disciplina Oficina I. Sistema que recomenda itens a partir do histórico de avaliações de vários usuários (filtragem colaborativa), com interface web para consultar históricos, ver recomendações e avaliar itens.

## 1. Objetivo

Construir e avaliar um sistema de recomendação que:

- usa uma base de interações usuário–item;
- prepara os dados e implementa filtragem colaborativa;
- gera recomendações personalizadas **sem incluir itens que o usuário já conhece**;
- trata o usuário **sem histórico** (cold-start);
- oferece uma interface para consultar histórico, recomendações e avaliar itens;
- é avaliado com pelo menos duas métricas adequadas, com exemplos para cinco usuários.

## 2. Fundamentação

**Filtragem colaborativa** parte da ideia de que usuários que avaliaram itens de forma parecida no passado tendem a gostar dos mesmos itens no futuro. Não usa o conteúdo dos itens, só a matriz de interações usuário × item.

- **Item-based:** dois itens são parecidos se os mesmos usuários os avaliam de modo parecido. Um item é pontuado pelas notas que o usuário deu aos itens mais similares a ele. Como a similaridade entre itens muda pouco, o modelo é estável e permite **explicar** a recomendação ("porque você avaliou X").
- **User-based:** busca os usuários mais parecidos com o alvo e pontua cada item pela média ponderada das notas desses vizinhos.
- **Similaridade:** cosseno sobre notas **centradas na média do usuário**, o que corrige o fato de alguns usuários darem notas sistematicamente altas ou baixas. Com **shrinkage** `n/(n+λ)`, pares com poucos co-avaliadores pesam menos. Só os `k` vizinhos mais similares são mantidos.
- **Cold-start:** sem histórico não há similaridade a calcular. O sistema recomenda os itens mais populares, ordenados pela **média bayesiana** `(v·R + m·C)/(v + m)`. Ela evita que um item com 2 notas 5,0 supere um clássico com 300 notas de média 4,4.
- **Baselines:** popularidade e aleatório, para mostrar que a filtragem colaborativa agrega valor.

## 3. Dados

Base padrão: **MovieLens `ml-latest-small`** (GroupLens), com avaliações de 0,5 a 5,0.

| | Bruto | Após tratamento |
|---|---|---|
| Interações | 100.836 | 94.794 |
| Usuários | 610 | 610 |
| Itens (filmes) | 9.724 | 4.980 |
| Esparsidade | — | 96,88 % |
| Nota média | — | 3,52 |

**Tratamento** (`src/recsys/data.py`):

1. remoção de nulos e conversão de tipos;
2. descarte de notas fora da escala configurada;
3. remoção de duplicatas usuário–item (fica a mais recente);
4. filtro iterativo de esparsidade: usuários com < 5 avaliações e filmes com < 3 avaliações saem;
5. mapeamento dos ids para índices contíguos e construção da matriz esparsa (CSR).

O relatório do que sobra após cada etapa aparece na aba **Resultados** da interface.

### Usar outra base de dados

Nenhum código precisa mudar. Edite `config.yaml`:

```yaml
dataset:
  interactions_path: data/minha_base.csv
  sep: ";"
  columns: {user: cliente, item: produto, rating: nota, timestamp: data}   # rating/timestamp podem ser null
  items_path: data/produtos.csv        # opcional (títulos); null se não houver
  items_columns: {item: id, title: nome, extra: categoria}
  rating_scale: [1, 5]
```

Se a base não tiver nota (`rating: null`), toda interação vale 1,0 (**feedback implícito**) e o modelo passa a trabalhar sem centralização. Isso foi testado com uma base sintética de compras.

## 4. Método

- **Divisão treino/teste:** hold-out **temporal por usuário**. Os 20 % de avaliações mais recentes de cada usuário (com ≥ 5 avaliações) vão para teste. O modelo é treinado no passado e testado no "futuro", como no uso real.
- **Item relevante:** nota ≥ 4,0 no teste.
- **Recomendação:** o modelo pontua todos os itens, **os já conhecidos pelo usuário são mascarados** e sai o top-N.
- **Hiperparâmetros** (`config.yaml`): 30 vizinhos, shrinkage 10, mínimo de 20 votos para a popularidade.
- **Detalhe de implementação:** os 30 vizinhos são os 30 itens (ou usuários) mais similares em toda a base, e não apenas entre os que o usuário avaliou. É uma simplificação comum que mantém o cálculo vetorizado e rápido.

### Métricas

| Métrica | O que mede | Sentido |
|---|---|---|
| **Precision@10** | fração dos 10 recomendados que o usuário de fato gostou | ↑ |
| **Recall@10** | fração dos itens que ele gostou que apareceram no top-10 | ↑ |
| **MAP@10** | precisão média nas posições dos acertos (premia acertos no topo) | ↑ |
| **NDCG@10** | qualidade do ranking, com desconto logarítmico pela posição | ↑ |
| **RMSE / MAE** | erro da nota prevista nos pares de teste | ↓ |
| **Cobertura** | fração do catálogo que aparece em alguma recomendação | ↑ |
| **Diversidade** | 1 − similaridade média entre os itens da lista | ↑ |

## 5. Resultados

Gerados por `python scripts/evaluate.py` (arquivo `results/metrics.csv`), sobre 18.736 interações de teste. As métricas de ranking são calculadas nos usuários que têm ao menos um item relevante no teste.

| Modelo | Precision@10 | Recall@10 | MAP@10 | NDCG@10 | Cobertura | Diversidade | RMSE | MAE |
|---|---|---|---|---|---|---|---|---|
| Aleatório | 0,0030 | 0,0012 | 0,0009 | 0,0030 | 0,696 | 0,931 | — | — |
| Popularidade | 0,0520 | 0,0403 | 0,0309 | 0,0657 | 0,012 | 0,535 | 0,994 | 0,770 |
| User-based CF | 0,0669 | 0,0632 | 0,0441 | 0,0911 | 0,144 | 0,637 | **0,936** | 0,714 |
| **Item-based CF** | **0,0840** | **0,0745** | **0,0579** | **0,1107** | **0,153** | 0,623 | 0,951 | **0,713** |

**Leitura:**

- O item-based CF supera a popularidade em NDCG@10 em cerca de **68 %** (0,111 contra 0,066) e o aleatório por uma ordem de grandeza. Recomendar pelo gosto de cada usuário compensa.
- A popularidade recomenda quase sempre os mesmos filmes (cobertura de 1,2 % do catálogo). O CF chega a 15 %, então sugere itens mais variados.
- O aleatório tem cobertura e diversidade altas, mas acerta quase nada. Essas duas métricas só fazem sentido lidas junto com as de acurácia.
- Nas notas previstas, o user-based tem o menor RMSE (0,936) e o item-based o menor MAE. Os dois batem a popularidade (0,994).
- Os valores absolutos de precisão são baixos, o que é normal: a base é muito esparsa (96,9 %) e o teste conta como acerto apenas o que o usuário **de fato avaliou** depois. Um filme bom que ele nunca avaliou conta como erro.

### Exemplos para cinco usuários

Em `results/exemplos_5_usuarios.md` há, para os usuários 380, 415, 574, 353 e 547, os filmes mais bem avaliados, o top-10 recomendado, a justificativa ("por semelhança com…"), os acertos no teste e o caso do usuário sem histórico. Trecho:

> **Usuário 380** — 896 filmes avaliados (entre os 5,0: *Halloween*, *King Kong*, *Guardians of the Galaxy*)
> 1. *Princess Bride, The (1987)* — por semelhança com *Star Wars: Episode V* e *Star Wars: Episode IV*
> 2. *American History X (1998)* — por semelhança com *Fight Club* e *Reservoir Dogs*
> …
> 8. *2001: A Space Odyssey (1968)* — por semelhança com *Blade Runner* e *Brazil* ✅ acerto no teste

## 6. Limitações

- **Cold-start:** usuário sem histórico recebe apenas os populares, sem personalização. Ela começa a partir da primeira avaliação feita na interface.
- **Esparsidade e viés de popularidade:** com poucos dados por item, as similaridades são ruidosas. O shrinkage ameniza, mas não elimina.
- **Avaliação offline:** o teste só conhece o que o usuário avaliou. Recomendações boas mas nunca avaliadas contam como erro, então as métricas subestimam a qualidade real.
- **Base pequena:** `ml-latest-small` tem 610 usuários. Os resultados não se generalizam para escalas maiores. A similaridade é calculada com matrizes densas item × item, o que só cabe em memória para catálogos de alguns milhares de itens.
- **Sem conteúdo:** o modelo ignora gênero, elenco e sinopse. Um híbrido (CF + conteúdo) ajudaria em itens novos.
- **Sem validação de hiperparâmetros:** vizinhos e shrinkage seguem valores usuais, não foram otimizados.
- **Uso acadêmico:** não há autenticação, concorrência nem persistência robusta. As avaliações da interface vão para um CSV local.

## 7. Conclusão

A filtragem colaborativa superou os baselines em todas as métricas de ranking. O item-based teve o melhor desempenho geral e ainda oferece explicações às recomendações. A abordagem é simples, interpretável e funciona sobre qualquer base de interações. Como próximos passos, ficam a otimização dos hiperparâmetros, a fatoração de matrizes (ex.: SVD/ALS) e um modelo híbrido para reduzir o problema do cold-start.

## 8. Como executar

Requer Python 3.11+ (testado com 3.13).

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

python scripts/download_data.py   # baixa o MovieLens para data/
python scripts/evaluate.py        # (opcional) regenera métricas e exemplos em results/
streamlit run app.py              # abre a interface em http://localhost:8501
pytest                            # testes das métricas e das regras de recomendação
```

### Interface

| Aba | Função |
|---|---|
| 📚 Histórico | itens avaliados pelo usuário, nota média e distribuição das notas |
| ✨ Recomendações | top-N sem itens já vistos, com justificativa; troca de modelo na barra lateral |
| ⭐ Avaliar itens | busca por título e nota; grava em `data/user_ratings.csv` e atualiza as recomendações |
| 📊 Resultados | tabela e gráfico de métricas e relatório do tratamento dos dados |

### Roteiro sugerido para a demonstração

1. Apresentar o problema e a base (barra lateral → "Sobre a base").
2. Escolher um usuário, mostrar o **Histórico** e depois as **Recomendações** (destacar que nada do histórico se repete e a justificativa).
3. Trocar o modelo (item-based → user-based → popularidade) e comparar.
4. Selecionar **Novo usuário (sem histórico)**: aparecem os populares.
5. Na aba **Avaliar itens**, avaliar 3 ou 4 filmes e ver as recomendações passarem a ser personalizadas.
6. Fechar na aba **Resultados** com as métricas.

## Estrutura

```
app.py                 interface Streamlit
config.yaml            base de dados, mapeamento de colunas e hiperparâmetros
scripts/               download_data.py, evaluate.py
src/recsys/            data.py, models.py, metrics.py, evaluation.py
tests/                 testes das métricas e dos recomendadores
results/               metrics.csv e exemplos_5_usuarios.md (gerados)
```
