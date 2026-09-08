# CFLOW Pixel Head — Correções, Impactos e Guia da Fase 3

Documento de acompanhamento das Fases 1 e 2 aplicadas em `anomaly-detection-pesquisa/differnet/`.

---

## 1. Resumo executivo

A revisão do pipeline CFLOW encontrou **quatro divergências silenciosas** entre o script de
treino e o de avaliação. A mais grave fazia o modelo em produção calcular uma **função
matemática diferente** daquela para a qual os pesos foram ajustados.

Corrigindo apenas essa divergência, a classe `vari-grip` sai de **Pixel AUROC 0.5583 → 0.8770**
e **AUPRO 0.2409 → 0.6451**, sem retreinar nada.

| Divergência | Treino | Avaliação | Gravidade |
|---|---|---|---|
| Clamp do bloco de acoplamento | `tanh(s) * 0.5` | `tanh(s) * 2.0` | 🔴 Crítica — transformação diferente |
| Blocos de atenção lidos | `cbam*` (mesmo em modelos SE) | `simsa*` | 🔴 Crítica — features diferentes |
| `out_size` (grade do positional encoding) | 96 (default) | 56 (fixo) | 🟠 Alta — condicionamento diferente |
| Agregação dos níveis | `/C` + min-max por nível | soma crua | 🟠 Alta — seleção vs. uso |

Nenhuma delas gerava erro: as três primeiras não afetam o formato do `state_dict`, e a quarta
é uma escolha de pós-processamento.

---

## 2. Os bugs em detalhe

### 2.1 🔴 Clamp do acoplamento afim (a causa dominante)

Um bloco de acoplamento afim calcula `y = x · exp(s) + t`, onde `s` é limitado por
`tanh(s) · clamp`. O valor de `clamp` **faz parte da definição da transformação**.

```python
# pixel_train_from_pretrained.py  (pesos ajustados com isto)
s = torch.tanh(s) * 0.5

# evaluate_full_pipeline.py       (pesos usados com isto)
s = torch.tanh(s) * 2.0
```

Com `clamp = 2.0`, a escala aplicada varia em `exp([-2, 2])` — uma faixa de **~55×** —
enquanto o flow foi treinado para operar em `exp([-0.5, 0.5])`, faixa de ~2.7×.
O `log_det` também fica 4× maior, distorcendo diretamente a log-verossimilhança que vira
o score de anomalia.

Analogia: é como treinar uma rede com `BatchNorm` e avaliá-la com as estatísticas trocadas.
Os pesos carregam sem erro, mas a função computada não é a mesma.

**Impacto medido** (`vari-grip`, conjunto de teste completo, sem smoothing):

| Configuração | Pixel AUROC |
|---|---|
| produção atual (`clamp = 2.0`) | 0.5583 |
| `clamp = 0.5` (como treinado) | **0.8770** |

### 2.2 🔴 Caminho de atenção errado no treino

`SEDifferNet.__init__` **declara** `cbam1..4` e `simsa1..4`, mas seu `forward()` usa
apenas `simsa1`, `simsa2` e `simsa4`. Os blocos CBAM existem só para compatibilidade de
checkpoint e **nunca recebem gradiente**.

O extrator de features do treino decidia por `hasattr`:

```python
self._use_cbam = hasattr(model, 'cbam1')   # True até para SEDifferNet
```

Ou seja: para modelos SE, o CFLOW foi treinado sobre features passadas por blocos de
atenção **aleatórios e nunca treinados**. O avaliador usava `simsa*`, o caminho correto.

A correção resolve pela **classe do modelo**, que é o que determina o `forward()`:

```python
def resolve_attention(model, attention='auto'):
    if attention == 'legacy_train':
        return 'cbam' if hasattr(model, 'cbam1') else 'se'   # reproduz o bug
    return 'cbam' if type(model).__name__ == 'CBAMDifferNet' else 'se'
```

**Resultado inesperado:** avaliar com `simsa*` (correto) dá **melhor** resultado que com
`cbam*` (como treinado) — 0.8770 contra 0.8466. Não tenho explicação mecanicista; a
hipótese é que ambos são atenção de canal e produzem features similares o bastante para o
flow generalizar. Vale investigar antes de tirar conclusões para o artigo.

### 2.3 🟠 `out_size` divergente

`out_size` define a grade `H×W` das features e, portanto, a grade do **positional encoding**
que condiciona o flow. Como o flow opera por posição (cada pixel é uma amostra), o
`state_dict` não depende de `out_size` — a divergência é indetectável a partir dos pesos.

- Treino: `--out_size` default **96**
- Avaliação: `out_size = c.img_size[0] // 8` = **56**, fixo

Um pixel na posição `(10, 10)` de uma grade 56×56 recebe um PE completamente diferente do
mesmo índice numa grade 96×96.

### 2.4 🟠 Função de agregação divergente

```python
# treino:                  nll = -log_p / C  → min-max por nível → soma
# evaluate_full_pipeline:  nll = -log_p      → soma crua
```

Sem normalização, o nível de maior magnitude domina. Em `vari-grip`, L1 tem magnitude ~764
contra −441 (L2) e −527 (L3) — e L1, sozinho, tem AUROC 0.5227 (quase aleatório). O score
de produção era dominado pelo pior nível.

Consequência prática: `best_pixel_auroc.pt` foi **selecionado** com um critério e
**aplicado** com outro.

### 2.5 🟡 Menores

- **Score de imagem = `max` dos pixels.** Um único falso positivo de borda define a imagem
  inteira. É por isso que os checkpoints registram `image_auroc = 0.448` — pior que aleatório.
- **Checkpoint sem hiperparâmetros.** Guardava só `epoch`, `pixel_auroc`, `image_auroc`.
  Impossível saber com que `out_size`/`clamp` o modelo foi treinado.
- **`nll_per_level` do avaliador sem chunking** (o do treino usava 4096) — risco de OOM.

---

## 3. O artefato de borda

### 3.1 Diagnóstico

Medições em `vari-grip` e `lightning-rod-suspension`:

| Nível | Canais | Viés borda−centro (vari-grip) | Correlação perfil treino↔teste |
|---|---|---|---|
| **L1** | 64 | **+282.8** | **0.695** |
| L2 | 192 | +11.9 | 0.986 |
| L3 | 256 | −32.5 | 0.985 |

**Causa:** L1 sai imediatamente após `Conv2d(3, 64, kernel=11, stride=4, padding=2)`.
Um kernel 11×11 com padding 2 faz as ativações da borda serem calculadas majoritariamente
sobre **zeros** — conteúdo que nunca aparece no interior da imagem. O flow, treinado só em
imagens normais, corretamente atribui alta NLL a esse padrão raro.

E a distribuição real das anomalias confirma o problema: apenas **12.9%** (vari-grip) e
**22.1%** (lightning-rod) dos pixels anômalos reais estão no anel de borda, que ocupa
**43.8%** da área.

### 3.2 Por que a calibração por posição falhou

A solução "óbvia" — estimar a NLL média por posição no conjunto de treino e subtrair — foi
testada e **piorou** os resultados (−0.062 e −0.069 de AUROC).

A tabela acima explica: o perfil posicional de L1 correlaciona apenas **0.695** entre treino
e teste, contra 0.98–0.99 para L2/L3. O viés de borda de L1 **depende do conteúdo da imagem**,
não é um deslocamento geométrico fixo. Subtrair uma média de treino injeta ruído em vez de
remover viés.

Por isso `compute_level_stats()` usa **um escalar (média, desvio) por nível**, não um mapa
por posição. Estatística global transfere; estatística por posição não.

### 3.3 Técnicas implementadas

| Técnica | Flag | vari-grip | lightning-rod | média |
|---|---|---|---|---|
| soma crua (comportamento antigo) | `--score_norm raw` | 0.5714 | 0.7930 | 0.6822 |
| padronização global por nível | `--score_norm per_level_std` | 0.6997 | 0.7561 | 0.7279 |
| reflect-pad + padronização | `--reflect_pad 56` + acima | 0.7151 | 0.7805 | **0.7478** |
| descartar borda de 1–2 células | `--border_margin` | +0.011 | +0.018 | consistente |

> Medições feitas em resolução nativa com `clamp = 2.0` (produção antiga). Servem para
> comparar técnicas entre si, não como valores absolutos.

**Reflect padding** ataca a causa raiz: preenche a região do padding com o espelhamento do
conteúdo real, de modo que a convolução da borda veja algo plausível em vez de zeros. A
implementação preenche a entrada, calcula as features numa grade proporcionalmente maior e
recorta de volta — preservando `out_size` e, portanto, a grade do PE.

⚠️ **Ressalva honesta:** o reflect padding **não reduziu** o viés de borda de L1 (em
`vari-grip` até aumentou: +283 → +389), mas melhorou o AUROC quando combinado com
padronização. O mecanismo não está explicado. Trate como hipótese, não como conclusão.

---

## 4. O que mudou no código

### 4.1 Novo: `core/cflow.py`

Fonte única de verdade, substituindo três cópias divergentes:

| Componente | Papel |
|---|---|
| `SEBackboneFeatureExtractor` | Features L1/L2/L3, com `reflect_pad` e `attention` |
| `pos_encoding`, `CondCouplingBlock`, `CondFlow` | Primitivas do flow, com `clamp_scale` |
| `CFlowPixelHead` | `nll_per_level` (com chunking) e `score_map(mode=...)` |
| `compute_level_stats` | Estatísticas escalares por nível |
| `border_mask`, `image_score_from_map` | Mitigação de borda e score de imagem |
| `save_cflow_checkpoint`, `read_cflow_hparams`, `resolve_hparam` | Metadados do checkpoint |

### 4.2 Resolução de hiperparâmetros

Precedência: **CLI > checkpoint > fallback com aviso**.

```
out_size=96 (checkpoint) | clamp_scale=0.5 (checkpoint) | n_blocks=6 | attention=se
```

Checkpoints novos carregam tudo. Checkpoints antigos emitem aviso explícito:

```
WARNING: 'out_size' not stored in this checkpoint; falling back to 56.
         Pass --out_size explicitly if training used another value.
```

### 4.3 Defaults — alinhados ao treino

A avaliação passou a **espelhar a configuração de treino**. Os valores de referência ficam
centralizados em `core/cflow.py` (`TRAIN_CLAMP_SCALE`, `TRAIN_OUT_SIZE`, `TRAIN_SCORE_MODE`,
`DEFAULT_ATTENTION`), consumidos pelos dois scripts.

| Parâmetro | Default (treino e avaliação) | Justificativa |
|---|---|---|
| `--clamp_scale` | **0.5** | Valor com que os pesos foram ajustados. |
| `--attention` | **`se`** | O pipeline CFLOW é construído sobre SEDifferNet; os blocos CBAM que um modelo SE declara nunca recebem gradiente. |
| `--out_size` | **96** | Default do treino. |
| `--score_norm` | **`per_level_minmax`** | Mesmo critério usado para selecionar os checkpoints. |
| `--reflect_pad`, `--border_margin` | `0` (desligado) | Técnicas novas, precisam de validação. |
| `--image_score` | `max` | Preservado. |

Forçar um caminho de atenção que o modelo não treina agora emite aviso:

```
WARNING: reading 'cbam' attention blocks from SEDifferNet, whose forward() uses 'se'.
         Those blocks receive no gradient during backbone training.
```

**Resultado dos defaults alinhados** (conjunto de teste completo, `--sigma 4`):

| Classe | Pixel AUROC (antes → depois) | Pixel AP | AUPRO |
|---|---|---|---|
| `vari-grip` | 0.5583 → **0.8987** | 0.0123 → **0.1022** | 0.2409 → **0.6484** |
| `lightning-rod-suspension` | 0.7930 → **0.8496** | → 0.0284 | → 0.4362 |

As duas classes melhoram, sem retreinar. O Pixel AP de `vari-grip` sobe ~8×, saindo da faixa
do baseline trivial.

**Isto altera os números já publicados.** Para reproduzir exatamente os resultados antigos:

```powershell
python scripts/eval/evaluate_full_pipeline.py --clamp_scale 2.0 --out_size 56 --attention se --score_norm raw
```

Verificado: essa linha reproduz `Pixel AUROC 0.5583` / `Image AUROC 0.9186`, idênticos ao
pré-refatoração.

---

## 5. Verificação realizada

1. **Equivalência numérica bit a bit**, executada *antes* da religação, comparando
   `core/cflow.py` com as duas implementações originais:

   | Comparação | max\|diff\| |
   |---|---|
   | backbone `attention='se'` == `evaluate_full_pipeline` | 0.000e+00 |
   | backbone `attention='legacy_train'` == treino | 0.000e+00 |
   | `score_map(raw, clamp=2.0)` == `evaluate_full_pipeline` | 0.000e+00 |
   | `score_map(per_level_minmax, clamp=0.5)` == treino | 0.000e+00 |
   | chunking vs. sem chunking | ≤ 3.1e-05 (ruído de GEMM) |

2. **Reprodução ponta a ponta:** flags legadas devolvem exatamente os números antigos.
3. **Roundtrip de checkpoint:** salvar → ler → resolver, com precedência CLI/checkpoint.
4. 40 arquivos compilam; `--help` funciona nos três entry points.

---

## 6. Fase 3 — Como validar

O que está acima **não é resultado final**. As técnicas foram escolhidas olhando AUROC de
teste em 2 classes — isso é **viés de seleção**. Serve para gerar hipóteses.

### Passo 1 — Reproduzir a linha de base antiga

```powershell
python scripts/eval/evaluate_full_pipeline.py --clamp_scale 2.0 --out_size 56 --attention se --sigma 4
```

Guarde o CSV. É a referência do que já está no artigo.

### Passo 2 — Quantificar cada correção isoladamente

Rode as **5 classes**, acumulando uma correção por vez:

```powershell
# A: só o clamp
python scripts/eval/evaluate_full_pipeline.py --clamp_scale 0.5 --out_size 56 --attention se --score_norm raw --sigma 4
# B: + out_size de treino
python scripts/eval/evaluate_full_pipeline.py --clamp_scale 0.5 --out_size 96 --attention se --score_norm raw --sigma 4
# C: + score de treino (= defaults atuais)
python scripts/eval/evaluate_full_pipeline.py --sigma 4
```

Isso produz a tabela de ablação dos bugs — material forte para a seção de metodologia.

### Passo 3 — Criar um split de validação

**Este é o passo que falta e não pode ser pulado.** Hoje só existem `train/` e `test/`.

1. Separe ~20% das imagens de teste (estratificado por normal/anômalo) como `val`.
2. Escolha `--score_norm`, `--reflect_pad` e `--border_margin` **só no `val`**.
3. Reporte no `test` apenas a configuração vencedora, uma única vez.

Sem isso, os ganhos da Fase 2 não são publicáveis.

### Passo 4 — Retreinar com a configuração correta

Os checkpoints atuais foram treinados com o caminho CBAM errado e selecionados com um score
diferente do usado em produção. Retreinar resolve tudo de uma vez e gera checkpoints com
metadados:

```powershell
python scripts/train/pixel_train_from_pretrained.py `
    --class_name vari-grip --image_score topk
```

Os defaults de treino (`--attention se`, `--clamp_scale 0.5`, `--out_size 96`,
`--score_norm per_level_minmax`) já são os mesmos da avaliação, e agora ficam gravados no
checkpoint. Confirme que o `pixel_auroc` reportado no treino **bate** com `evaluate_full_pipeline.py`.
Esse é o teste de regressão definitivo — hoje eles divergem (0.7982 vs 0.5583).

### Passo 5 — Contextualizar contra o baseline trivial

```powershell
python scripts/analysis/smoke_test_baseline.py
```

Pixel AP na faixa de 0.012–0.018 está próximo do baseline trivial (pixels anômalos são raros).
AUROC alto com AP baixo é um sinal clássico de desbalanceamento extremo — reporte os dois.

### Passo 6 — Ablação do smoothing

```powershell
python scripts/eval/evaluate_cflow_no_smoothing.py --sigmas 0,2,4,6
```

Refaça com o `clamp` corrigido: a conclusão anterior (smoothing quase irrelevante) foi obtida
sob a transformação errada e precisa ser revalidada.

---

## 7. Riscos e pontos em aberto

| Item | Situação |
|---|---|
| `out_size` real dos checkpoints publicados | **Desconhecido.** Sem metadados; usamos o default de treino (96) com aviso. |
| Atenção SE superar a CBAM usada no treino | Sem explicação mecanicista, mas SE é o caminho correto e é o default. |
| Reflect padding aumentar o viés de L1 e ainda assim ajudar | Sem explicação mecanicista. |
| Ganhos da Fase 2 (`reflect_pad`, `border_margin`) | Medidos em 2 de 5 classes, com viés de seleção. Continuam desligados por padrão. |
| Resultados já publicados | Mudam com os novos defaults. |
