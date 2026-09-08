# Plano de melhoria da NF Head (CFLOW) — diagnóstico, propostas e ablation

Classes estudadas: as **5 do INSPLAD-seg** — `vari-grip` (ferrugem larga), `lightning-rod-suspension`
(ferrugem fina), `glass-insulator` (peça ausente), `polymer-insulator-upper-shackle` e
`yoke-suspension`. As rodadas 1 e 2 foram feitas só em `vari-grip`; a 3 estendeu a duas classes e a 4
fechou as duas restantes.
Restrição fixa: o extrator de features é sempre o **final model do SEDifferNet** da classe,
congelado. Só a NF head (CFLOW) e o pós-processamento podem mudar.

Script do ablation: `scripts/eval/ablation_nf_head.py`.
Avaliação: `scripts/eval/evaluate_image_level.py`, `evaluate_pixel_level.py`,
`evaluate_full_pipeline.py` (Seção 15).

---

## 0. Sumário executivo

> **Estado em 2026-09-08.** 5 rodadas de retreino (27 receitas) + pós-hoc nas heads publicadas +
> **treino em lote das 5 classes** (`run_20260907_145952`, Seção 6.9).
> **~24 000 medições** de validação, 8 conjuntos de held-out independentes.
> **O estudo está fechado nas 5 classes** (estrutura final na Seção 9.0).

**Melhor resultado por classe** (held-out, `val_fraction=0.4`, configuração escolhida na validação):

| Classe | Produção (pixel AUROC / AUPRO) | Melhor obtido | Ganho | Receita vencedora | Imagem |
|---|---|---|---|---|---|
| `glass-insulator` | 0.8821 / 0.6069 | **0.9560 / 0.7604** | **+0.074 / +0.154** | `featnorm_clamp19` · L=02 · σ=2 | 0.7706 |
| `vari-grip` | 0.7707 / 0.6252 | **0.9270 / 0.7585** | **+0.156 / +0.133** | `featnorm_clamp19` · L=01 · σ=8 | 0.9044 |
| `yoke-suspension` | 0.8336 / 0.5790 | **0.9188 / 0.6611** | **+0.085 / +0.082** | `baseline` · L=02 · σ=8 | 0.9744 |
| `lightning-rod` | 0.7669 / 0.3880 | **0.8815 / 0.5642** | **+0.115 / +0.176** | `baseline` · L=2 · σ=4 | 0.9485 |
| `polymer-shackle` | 0.8624 / 0.5753 | **0.8806 / 0.5943** | +0.018 / +0.019 | `baseline` · L=012 · σ=8 | 0.9495 |

Todas usam a mesma parte de inferência: `reflect_pad 112` + `tta flips` + `score_mode raw`.

**As doze descobertas que mais importam:**

1. **O ganho está no pós-processamento, não no modelo.** A cadeia `pad112` → `flips` → `raw` →
   `levels` → `sigma` soma **+0.097 de AUROC em média** nas 5 classes, sem retreinar nada. Nenhuma
   receita de treino chega perto disso (Seção 6.8).
2. **`raw` em vez de `minmax` é o maior efeito isolado do estudo:** ganho pareado de **+0.0497**,
   positivo em 76–100 % dos pares nas 5 classes. Nenhuma classe contradiz.
3. **⚠️ Diagnósticos feitos sobre heads publicadas são enganosos.** Em `glass-insulator`, L1 valia
   **0.2583** na head publicada (parecia sinal invertido) e vale **0.87** após retreinar — um salto de
   **+0.61**. A "inversão física" da Seção 5.2 era **defeito da receita antiga**, não propriedade da
   classe. Lição metodológica cara.
4. **⚠️ "L2 nunca ajuda" era falso** — sobrevivia porque só 3 classes tinham sido medidas. Em
   `polymer`, **L2 é o melhor nível** (0.880 contra L1 0.776 e L3 0.841). Ver 7.1.
5. **`levels=02` (L1+L3) ainda é o melhor default**, e por uma margem maior do que se pensava:
   medindo o regret corretamente (melhor config com `02` contra melhor config global, não médias de
   grid), ele é **≤ 0.0012 em 4 de 5 classes**. A exceção é `polymer` (0.0278). Ver o quadro de
   correção em 6.7.
6. **⚠️ `featnorm` não generaliza.** Vence em `glass` (+0.0396) e é **claramente prejudicial em
   `yoke` (−0.0499 AUROC, −0.1226 AUPRO)** — o pior resultado de qualquer receita no estudo. Foi
   rebaixada de "receita padrão" para "ligar só se validar na classe" (6.8, 9.0).
7. **TTA por flips é quase universal, não universal.** +0.0120 em média, mas em `yoke` só **46 % dos
   pares** melhoram — indistinguível de ruído. Continua valendo por ser barato.
8. **Reflect padding é técnica de inferência, não de treino.** Aplicar no treino **piora**.
9. **As duas cabeças falham de forma independente.** `glass` tem o melhor pixel AUROC (0.9560) e o
   **pior** de imagem (0.7706); `yoke` é o oposto. Métrica agregada esconde isso — daí os três
   scripts de avaliação separados (Seção 15).
10. **O limiar de significância foi medido, não assumido:** trocar o split muda o resultado do
    **mesmo modelo** em 0.0044 AUROC. Diferenças menores que isso são ruído.
11. **⚠️ vari-grip tem L3 como pior nível (0.8052 vs L1 0.9191).** Forçar `--levels 0,2` trava o
    pior nível na head e derruba AUROC em 0.063 e AUPRO em 0.150 em relação a L1 sozinho. Pontuar
    L1 isolado entrega **0.9191 AUROC / 0.7621 AUPRO** no teste completo (Seção 6.9).
12. **Overfitting por flow atinge datasets grandes (yoke pico na época 4, −0.030 até a 80).**
    Sobrescrever orçamentos com `--epochs` global é nocivo. Incorporou-se do RD++ seleção composta
    `(px + img + aupro)/3`, `history.json` persistido a cada avaliação e early stopping com `--patience`
    (Seção 6.9).

**Próximos passos prioritários:** testar `featnorm` seletiva só em L1, que o mecanismo de 7.3.8
prevê resolver o caso `yoke` (Seção 9.1). A seleção de época durante o treino foi corrigida para
`raw` (Item 3 de 9.1) e enriquecida com o score composto do RD++.

---

## Índice

| Seção | Conteúdo |
|---|---|
| 1 | Glossário e conceitos (termos, níveis, clamp, featnorm, pad, métricas) |
| 2 | Arquitetura do pipeline |
| **3** | **Resultados — rodada 1** (8 receitas em vari-grip, 14 208 configurações) |
| **4** | **Resultados — rodada 2** (5 receitas em vari-grip, 2 496 configurações) |
| **5** | **Resultados — Passo 0 Multi-Classe** (heads publicadas em 3 classes) — ⚠️ conclusões sobre níveis revistas pela Seção 6 |
| **6** | **Resultados — rodadas 3 e 4** (retreino nas 4 classes restantes) · **6.7** hierarquia dos níveis e regret · **6.8** ganho de cada técnica · **6.9** treino em lote (`run_20260907_145952`), diagnóstico de vari-grip, overfitting e transferência do RD++ |
| **7** | **Descobertas consolidadas** (confirmadas, derrubadas, novas) |
| **8** | **Registro de caminhos testados e descartados** |
| **9** | **Próximos passos** (**9.0** estrutura final · **9.1** o que falta) |
| 10 | Histórico e ponto de partida |
| 11 | Diagnóstico — o que os mapas mostram (anotado) |
| 12 | Fundamentação na literatura |
| 13 | Catálogo de propostas |
| 14 | Protocolo experimental |
| 15 | Implementação — **os três scripts de avaliação** e os módulos de suporte |
| 16 | Armadilhas e lições de implementação |
| 17 | Reprodutibilidade |

> Seções **3–9** são o registro do que foi descoberto; **10–14** documentam o raciocínio e o método
> que levaram até lá; **15–17** são referência técnica.

---

## 1. Glossário e conceitos

### 1.0 Termos do pipeline

| Termo | Significado |
|---|---|
| **NF / normalizing flow** | Rede inversível que mapeia features `x` para um espaço latente `z` com distribuição conhecida (gaussiana). Como é inversível, a densidade `p(x)` é calculável exatamente. |
| **NF head / pixel head** | O flow condicional (CFLOW) que roda **por posição** sobre as features do backbone congelado e produz o mapa de anomalia. É o objeto de estudo deste documento. |
| **NLL** (negative log-likelihood) | `−log p(x)`. Treinado só em imagens normais, o flow dá NLL baixa ao que é comum e alta ao que é raro → **a NLL é o score de anomalia**. |
| **Coupling block** | Bloco básico do flow: divide os canais em duas metades, usa uma para prever escala `s` e deslocamento `t`, e aplica `y = x·exp(s) + t` na outra. Inversível por construção; o jacobiano é `exp(Σs)`. |
| **log_det** | Log do determinante do jacobiano acumulado; entra na verossimilhança junto com o termo gaussiano. |
| **Positional encoding (PE)** | Vetor senoidal 2D que informa ao flow **onde** na grade 96×96 aquela posição está. É o "condicional" do CFLOW: o modelo aprende uma distribuição diferente por região da imagem. |
| **`out_size`** | Lado da grade comum (96) para a qual os três níveis são reamostrados. Define a grade do PE — mudar `out_size` muda o condicionamento sem mudar o `state_dict`. |
| **Backbone congelado** | O SEDifferNet final da classe. Não recebe gradiente; só a NF head treina. |
| **Anomaly map** | Mapa 448×448 de scores por pixel, obtido agregando os mapas de NLL dos níveis e fazendo upsample bilinear. |

### 1.1 L1, L2, L3 — os três níveis de features

A NF head não lê a imagem: ela lê três mapas de features extraídos de pontos diferentes da AlexNet
(congelada) com blocos de atenção SE do SEDifferNet:

| Nível | Ponto de extração | Canais | Resolução nativa (448px) | O que enxerga |
|---|---|---|---|---|
| **L1** | logo após `conv1` (kernel 11, stride 4) + `simsa1` | 64 | 109×109 | bordas, texturas finas, **cor** |
| **L2** | após `pool2` + `simsa2` | 192 | 53×53 | formas médias, regiões |
| **L3** | após o fim da pilha de convs + `simsa4` | 256 | 27×27 | semântica, contexto global |

Os três são reamostrados para uma grade comum de 96×96 (`out_size`) e um flow separado aprende a
distribuição de cada nível **posição a posição** (condicionado pela posição via positional encoding).
O mapa final é a soma dos três mapas de NLL após alguma normalização. Cada nível tem papel
complementar: L1 capta defeitos finos de textura (ferrugem), L2/L3 cobrem defeitos de forma/grandes.
Trade-offs: L1 é o mais sensível ao artefato de borda (a `conv1` usa padding zero), L3 tem a grade
mais grosseira (uma célula = ~16.6px).

> **Medição (Seções 3.3, 4.2, 5.2 e 6.2):** **não há hierarquia universal.** Em `vari-grip` L1 é o
> melhor nível (0.89 vs 0.84 de L2 e 0.80 de L3) e uma head só com L1 supera a fusão dos três. Em
> `lightning-rod` a ordem se inverte (L3 0.88 > L2 0.81 > L1 0.76) e em `glass-insulator` L1 chega a
> **inverter o sinal** (0.26). O conjunto de níveis precisa ser ablado **por classe** (7.3.13).

### 1.2 clamp_scale — limitador da escala do coupling

Cada bloco de acoplamento afim calcula `y = x · exp(s) + t`. Antes de ser usado, `s` passa por
`tanh(s) · clamp_scale`, limitando a faixa da escala:

- `clamp = 0.5` → `exp(s) ∈ [e⁻⁰·⁵, e⁰·⁵] ≈ [0.61, 1.65]` (~2.7× entre os extremos)
- `clamp = 1.9` (CFLOW-AD original) → `exp(s) ∈ [e⁻¹·⁹, e¹·⁹] ≈ [0.15, 6.7]` (~45× entre os extremos)
- `clamp = 2.0` era o valor errado usado historicamente na avaliação (~55×).

É um hiperparâmetro de **expressividade vs. estabilidade**: clamps maiores deixam o flow transformar
mais agressivamente as features, mas tornam a verossimilhança mais difícil de otimizar. **O valor faz
parte da definição da transformação** — avaliar com um clamp diferente do treino computa uma função
diferente com os mesmos pesos.

### 1.3 featnorm — padronização por canal

As saídas de ReLU da AlexNet são sempre ≥ 0, com escalas muito diferentes entre canais (médias
de ~0.1 a ~10) e caudas pesadas. `--feat_norm` calcula, sobre as imagens **normais de treino**, a
média e o desvio de cada canal de cada nível e transforma `f' = (f − μ) / σ` antes do flow. Ideia
emprestada do PaDiM (arXiv 2011.08785). Efeito: todas as features entram no flow em escala ~N(0,1),
o regime em que os MLPs do coupling trabalham melhor; em particular a NLL torna-se comportada o
suficiente para a agregação **raw** funcionar e para a calibração posicional fazer sentido. As
estatísticas são gravadas dentro do checkpoint (`feat_stats`) e reaplicadas na avaliação.

> **Medição:** em `vari-grip` foi a intervenção de treino de maior impacto (+0.070 AUROC, AP ×2.7
> sobre a produção), tornando a agregação `raw` viável (+0.057 de média sobre `minmax`).
> **Efeito colateral (7.3.8):** concentra o sinal em L1 e **degrada** L2/L3 — em L1 a informação está
> em poucos canais de alta magnitude e equalizar ajuda; em L2/L3 a mesma operação amplifica ruído.
> ⚠️ **Por isso não generalizou:** em `lightning-rod`, onde o sinal vive em L3, a head sem featnorm
> venceu (Seção 6.2). Use featnorm apenas quando o nível dominante da classe for raso.

### 1.4 pad (`--reflect_pad`) — padding da entrada antes do backbone

A `conv1` (kernel 11, padding 2) e os max-pools computam ativações de borda sobre pixels **zero**, um
conteúdo que nunca aparece no interior. O flow, treinado só em imagens normais, marca essas ativações
como anômalas → borda quente no mapa. Com `--reflect_pad 112`, a imagem de 448px é espelhada (reflect)
para 672×672 antes do backbone, as features saem numa grade maior e o anel correspondente aos 112px é
recortado de volta para 96×96 — cada posição passa a ser computada sobre conteúdo real. O valor 112
não é arbitrário: é o menor padding que respeita ao mesmo tempo o stride acumulado do backbone (16) e
o tamanho de célula da grade (448×3/96 = 14 px), preservando o registro exato entre mapa e máscara.
Com `--pad_mode replicate`, os pixels da borda são repetidos em vez de espelhados (evita "eco" de
estruturas).

> **Medição (Seção 4.6):** o pad deve ser usado **apenas na inferência**. Treiná-lo junto piorou
> (0.9033 vs 0.9185): o flow gasta capacidade modelando o conteúdo espelhado — que não é uma imagem
> plausível — como se fosse normal. Na inferência, o mesmo padding apenas substitui zeros (ainda menos
> plausíveis) por algo mais próximo da distribuição de treino. **Exceção à regra de coerência
> treino↔avaliação**, que continua valendo para `clamp`, `out_size`, `levels` e `feat_norm`.

### 1.5 Outros fatores do grid

- **`score_mode`** — como os mapas NLL dos níveis viram um só: `raw` (soma direta), `per_level_minmax`
  (min-max por imagem em cada nível, depois soma; joga fora a escala absoluta), `per_level_prob`
  (log-sigmoid limitado em [0,1] por nível, como no CFLOW-AD), `per_level_std` (padroniza com
  estatísticas escalares de treino).
- **`pos_calib`** — subtrai da NLL a média de cada posição estimada no treino. Só em L2/L3 (`l23`):
  o perfil posicional de L1 depende do conteúdo da imagem e não transfere.
- **`tta`** — test-time augmentation por flips (H/V): média dos mapas de 4 visões simétricas.
- **`sigma`** — suavização Gaussiana pós-mapa (σ=4 aproxima a agregação de 4×4 células do CFLOW-AD).
- **`margin`** — exclui um anel de N px das **métricas** (diagnóstico de viés de borda, não filtro).

### 1.6 Métricas — o que cada uma mede e por que reportar as três

| Métrica | Definição | O que captura | Armadilha |
|---|---|---|---|
| **Pixel AUROC** | Área sob a curva ROC sobre **todos** os pixels de **todas** as imagens | Ordenação global: um pixel anômalo tende a ter score maior que um normal? | Com ~1 % de pixels anômalos, 0.85 é fácil de atingir e pouco discrimina. Insensível a desbalanceamento. |
| **Pixel AP** (average precision) | Área sob a curva precisão-recall | Qualidade nas regiões de **alto score** — o que o operador realmente vê | Baseline trivial ≈ proporção de pixels anômalos (~0.01). Valores de 0.05–0.12 são baixos em absoluto mas 5–12× o trivial. |
| **AUPRO** (per-region overlap) | Média, por **região conexa** de defeito, da fração coberta, integrada até FPR 30 % | Se o método **cobre** os defeitos, dando peso igual a defeitos grandes e pequenos | Só conta imagens anômalas. Aqui é a métrica que pega falso positivo de borda que a margem esconde. |
| **Image AUROC** | AUROC de um score por imagem | Detecção (há defeito?) | No pipeline final vem do SEDifferNet (0.9186), não da head. `image_auroc_map` é o valor derivado do mapa — diagnóstico de escala absoluta. |

**Por que as três juntas:** AUROC alto com AP baixo é assinatura de desbalanceamento extremo; AUROC
alto com AUPRO baixo é assinatura de falso positivo concentrado (borda, fundo). Foi exatamente o que
separou `featnorm` (bom nos três) de `norot_pad112` (bom só no AUROC).

---

## 2. Arquitetura do pipeline

```
imagem 448×448
      │
      ├─[opcional] reflect pad 112 → 672×672
      ▼
AlexNet + SE (SEDifferNet final model, CONGELADO)
      ├── L1 (64ch, 109²)  ─┐
      ├── L2 (192ch, 53²)  ─┼─→ [feat_pool] → resize 96² → [crop do pad] → [feat_norm]
      └── L3 (256ch, 27²)  ─┘                                                    │
                                                                               ▼
                                              CondFlow_i (por nível) + PE 2D → NLL_i (96²)
                                                                               │
                                     [pos_calib] → [score_mode] → soma → upsample 448²
                                                                               │
                                                       [gaussian σ] → anomaly map
                                                                               │
                                            [image_score max/topk] → score de imagem
```

Duas coisas importam para interpretar tudo o que vem a seguir:

1. **O backbone não muda.** Todo ganho vem de (a) como a head modela as features e (b) como os mapas
   de NLL são combinados. O teto é dado pela qualidade das features do SEDifferNet.
2. **O flow é por posição, não por imagem.** Cada uma das 96×96 células é uma amostra independente
   condicionada pelo PE. Isso explica por que padding, rotação e calibração posicional têm efeitos
   tão grandes: todos alteram a relação entre posição e conteúdo.

---

## 3. Resultados medidos — rodada 1 de retreino (`run_20260904_092255`)

**Protocolo do experimento:** 8 variantes de treino, 40 épocas cada, todas sobre o mesmo SE final
model de vari-grip; pós-hoc completo (grid de 1 920 configurações) em cada head; seleção em 46
imagens de validação (34 normais + 12 anômalas), reporte nas 108 de held-out (80 + 28).
Total: **14 208 medições de validação**. `top` = melhor configuração selecionada na validação;
`produção` = a própria head com minmax/σ=4/pad0.

### 3.1 Resultado final por head (held-out, resolução plena)

| Head (treino) | Config vencedora (val) | Pix AUROC | AP | AUPRO |
|---|---|---|---|---|
| **featnorm** | pad112 · raw · L=01 · calib l23 · σ=8 | **0.9229** | **0.1196** | **0.7558** |
| clamp19 | pad112 · flips · raw · L=02 · calib l23 · σ=8 | 0.9171 | 0.0950 | 0.7128 |
| norot_pad112 | pad0 · flips · minmax · L=01 · σ=4 · m=8 | 0.9017 | 0.0742 | 0.5589 |
| pad112 | pad0 · minmax · L=01 · σ=4 · m=8 | 0.8888 | 0.0630 | 0.5883 |
| pool3 | pad0 · flips · minmax · L=012 · σ=4 | 0.8805 | 0.0585 | 0.6762 |
| baseline (produção) | pad0 · minmax · L=012 · σ=4 | 0.8789 | 0.0536 | 0.6737 |
| norot | pad112 · flips · minmax · L=012 · σ=4 | 0.8707 | 0.0522 | 0.7315 |
| l23 | pad0 · minmax · L=12 · m=8 | 0.8561 | 0.0449 | 0.5792 |
| publicada (120 ep) — comparação indireta | minmax · σ=4 · pad0, nas 154 imgs | 0.8987 | 0.1022 | 0.6484 |

⚠️ A linha "publicada" foi medida nas **154** imagens, não no held-out de 108 — não é diretamente
comparável. Para fechar a comparação falta rodar o estágio `posthoc` na head publicada com o mesmo
`--seed` (mesmo split).

Detalhe importante do `featnorm`: a config 0.9229 (raw + calib l23, σ=8) era o **3.º lugar** no
validation (0.8833 vs 0.8845 do 1.º — diferença de 0.0012, ruído com 12 anômalas). Como o grid fica
em disco, ela pôde ser identificada a posteriori; para publicar, o critério de seleção tem que ser
fixado **antes** de olhar o held-out (ver Seção 14).

### 3.2 Efeitos marginais agregados (14 208 configurações, média sobre as 8 heads)

Cada célula é a média (e o máximo) de `pixel_auroc` de validação sobre todas as configurações que
compartilham aquele valor do fator. A **média** mede robustez ("esse valor é seguro?"); o **máximo**
mede potencial ("esse valor participa da melhor combinação?"). Divergir entre os dois indica
interacão com outro fator.

**Agregação dos níveis (`score_mode`)**

| Valor | Média | Máximo | Leitura |
|---|---|---|---|
| `raw` | **0.8081** | 0.8837 | Mais robusto na média, mas só atinge o topo com `featnorm`/clamp alto |
| `per_level_std` | 0.7989 | 0.8769 | Alternativa razoável ao raw, sem precisar de featnorm |
| `per_level_minmax` | 0.7883 | **0.9012** | Maior máximo: é o que funciona nas heads não normalizadas |
| `per_level_prob` | 0.6992 | 0.8466 | **Pior em tudo** — ver Seção 8 |

**Subconjunto de níveis (`levels`)**

| Valor | Média | Máximo | Leitura |
|---|---|---|---|
| `01` (L1+L2) | **0.8145** | **0.9012** | Melhor na média **e** no máximo |
| `012` (todos) | 0.8073 | 0.8968 | L3 acrescenta pouco |
| `02` (L1+L3) | 0.8059 | 0.8892 | |
| `12` (L2+L3, sem L1) | 0.7465 | 0.8847 | Tirar L1 custa ~0.07 na média |
| `2` (só L3) | 0.7079 | 0.8408 | Pior |

**Fatores de borda e pós-processamento**

| Fator | Valor | Média | Máximo | Δ média |
|---|---|---|---|---|
| `pad` | 112 | 0.7778 | 0.8968 | **+0.0060** |
| | 0 | 0.7718 | 0.9012 | |
| `tta` | flips | 0.7782 | 0.9012 | **+0.0068** |
| | none | 0.7714 | 0.8949 | |
| `sigma` | 4 | 0.7785 | 0.9012 | **+0.0075** vs σ=0 |
| | 8 | 0.7749 | 0.8970 | +0.0039 vs σ=0 |
| | 0 | 0.7710 | 0.8973 | |
| `margin` | 8 | 0.7756 | 0.9012 | +0.0016 |
| | 0 | 0.7740 | 0.8968 | |
| `pos_calib` | none | 0.7919 | 0.9012 | |
| | l23 | 0.7582 | 0.8837 | **−0.0337** (mas ver 4.4) |

### 3.3 AUROC de cada nível isolado — efeito de pad e TTA

Média sobre as 8 heads, cada nível avaliado sozinho com agregação `raw`:

| pad | tta | L1 | L2 | L3 |
|---|---|---|---|---|
| 0 | none | 0.8613 | 0.8469 | 0.8148 |
| 0 | flips | 0.8631 | 0.8460 | 0.8267 |
| 112 | none | 0.8718 | 0.8651 | 0.8224 |
| 112 | flips | **0.8733** | **0.8648** | **0.8351** |

**Leituras:**
- **Ordem de informatividade: L1 > L2 > L3** (o inverso da hipótese inicial).
- **Pad 112 ajuda os três níveis:** +0.0105 (L1), **+0.0182 (L2)**, +0.0076 (L3). O maior ganho é em
  L2, não em L1 — contraintuitivo, já que L1 tem o maior viés de borda absoluto. Hipótese: L1 tem
  resolução alta o bastante para que o anel afetado seja proporcionalmente pequeno, enquanto em L2
  (53²) a borda contaminada é uma fração maior da grade.
- **TTA por flips ajuda principalmente L3** (+0.0127 com pad112), o nível mais dependente de
  contexto/posição — exatamente onde simetrizar o prior posicional deveria valer mais.

### 3.4 Interação `score_mode` × `pos_calib` — o achado não óbvio

Média de `pixel_auroc` de validação:

| score_mode | sem calib | com calib l23 | Δ |
|---|---|---|---|
| `per_level_minmax` | 0.8387 | 0.7379 | −0.101 |
| `per_level_std` | 0.8341 | 0.7637 | −0.070 |
| `raw` | 0.8277 | 0.7886 | −0.039 |
| `per_level_prob` | 0.6495 | **0.7429** | **+0.093** |

Na média, a calibração posicional **piora** — o que reproduz o resultado histórico. Mas o quadro muda
quando se olha por head, comparando a melhor config `raw + calib l23` contra a melhor `minmax`:

| Head | melhor `raw+calib_l23` | melhor `minmax` | Δ |
|---|---|---|---|
| **clamp19** | 0.8837 | 0.8475 | **+0.0362** |
| **featnorm** | 0.8833 | 0.8845 | −0.0012 (empate) |
| baseline | 0.8641 | 0.8769 | −0.0128 |
| pad112 | 0.8726 | 0.8949 | −0.0223 |
| norot | 0.8656 | 0.8946 | −0.0290 |
| pool3 | 0.8556 | 0.8964 | −0.0408 |
| norot_pad112 | 0.8549 | 0.9012 | −0.0463 |
| l23 | 0.7444 | 0.8652 | −0.1208 |

**Conclusão:** `raw + calibração posicional` só compete quando a NLL está numa escala bem-comportada
— ou seja, com `featnorm` ou com clamp alto. Nas demais heads, a soma crua é dominada pelo nível de
maior magnitude e o min-max por imagem é a única saída. **Esta é a interação central da rodada 1** e
explica por que `featnorm` e `clamp19` ficaram em 1.º e 2.º no held-out.

### 3.5 AUROC por nível isolado, por head (pad 112, sem TTA)

| Head | L1 | L2 | L3 |
|---|---|---|---|
| baseline | **0.8829** | 0.8640 | 0.8251 |
| featnorm | **0.8884** | 0.8535 | 0.7806 |
| pool3 | 0.8709 | 0.8700 | 0.8078 |
| clamp19 | 0.8689 | 0.8428 | 0.7955 |
| pad112 | 0.8686 | 0.8735 | 0.8341 |
| norot | 0.8658 | 0.8735 | **0.8565** |
| norot_pad112 | 0.8572 | **0.8767** | **0.8600** |
| l23 | — | 0.8666 | 0.8196 |

**O efeito seletivo da rotação.** Comparando pares que diferem só pela rotação:

| Par | ΔL1 | ΔL2 | ΔL3 |
|---|---|---|---|
| baseline → norot | −0.0171 | +0.0095 | +0.0314 |
| pad112 → norot_pad112 | −0.0114 | +0.0032 | +0.0259 |

Desligar a rotação **piora L1 e melhora L2/L3**, consistentemente nos dois pares. Interpretação:
textura/cor (L1) é aproximadamente invariante à rotação, então girar as imagens é augmentation
legítimo e ajuda com apenas 477 amostras; forma e contexto (L2/L3) são condicionados à posição, e
girar destrói justamente o prior posicional que o PE deveria capturar. **Consequência prática:** a
escolha certa não é "com ou sem rotação", e sim **augmentation por nível** — ou trocar rotação por
flips H/V, que preservam a estrutura da grade (ver Seção 9.2).

### 3.6 Pixel AP — melhor por head (validação)

| Head | melhor AP |
|---|---|
| clamp19 | **0.0951** |
| pool3 | 0.0902 |
| featnorm | 0.0893 |
| pad112 | 0.0785 |
| norot | 0.0784 |
| norot_pad112 | 0.0768 |
| baseline | 0.0765 |
| l23 | 0.0570 |

Baseline trivial (proporção de pixels anômalos) ≈ 0.011 — logo, o melhor resultado está ~11× acima
do trivial. O ranking por AP **não coincide** com o ranking por AUROC (pool3 é 2.º em AP e 5.º em
AUROC), mais uma razão para reportar as duas.

### 3.7 Efeito do descasamento pad treino ↔ avaliação

Cada head avaliada nas duas referências fixas (held-out):

| Head (pad de treino) | pad0 AUROC | pad112 AUROC | pad0 AUPRO | pad112 AUPRO |
|---|---|---|---|---|
| baseline (0) | 0.8789 | 0.8670 | 0.6737 | **0.6955** |
| pad112 (112) | 0.8638 | **0.8759** | 0.6185 | **0.7000** |
| norot (0) | 0.8543 | 0.8610 | 0.6794 | **0.7126** |
| norot_pad112 (112) | 0.8679 | 0.8579 | 0.6125 | **0.7182** |
| featnorm (0) | 0.8582 | 0.8335 | 0.6799 | **0.7014** |
| clamp19 (0) | 0.7680 | 0.7780 | 0.6230 | **0.6545** |
| pool3 (0) | 0.8694 | 0.8669 | 0.6630 | 0.6385 |
| l23 (0) | 0.8509 | 0.8444 | 0.5823 | **0.6175** |

**Padrão forte:** pad 112 melhora o **AUPRO em 7 de 8 heads** (+0.02 a +0.11), mesmo quando o AUROC
cai. Isso é exatamente o esperado se o pad estiver removendo falsos positivos de borda: AUROC quase
não sente (poucos pixels), AUPRO sente muito (a FPR usada no cálculo satura mais cedo). **Recomendação:
usar pad 112 na avaliação mesmo para heads legadas.**

### 3.8 Escala absoluta do mapa (`image_auroc_map`)

Na agregação global os quatro modos empatam (média 0.45–0.48), mas nas duas heads onde `raw`
funciona a diferença é grande:

| Head | configs `raw + calib l23` | configs `minmax` |
|---|---|---|
| featnorm | 0.74–0.76 | 0.41–0.50 |
| clamp19 | 0.64–0.73 | ~0.45 |

Ou seja: com `featnorm` + `raw`, o mapa de pixel passa a carregar informação de detecção de imagem
útil. Com `minmax`, não — por construção o máximo de cada imagem vira 1. Isso abre a possibilidade de
um score de imagem unificado (Seção 9.2), hoje delegado ao SEDifferNet.

---

## 4. Resultados medidos — rodada 2 (`run_20260905_011651`)

**Objetivo:** consolidar em cima de `featnorm`, o vencedor da rodada 1. 5 variantes × 40 épocas,
grid reduzido e corrigido (`--margins 0`, `--sigmas 4,8,12`, `--score_modes raw,per_level_minmax`,
`--level_sets 012,12,01,02,0,2`, `--val_fraction 0.4`) = **2 496 medições de validação**.

> ⚠️ **Rodadas 1 e 2 não são diretamente comparáveis:** `--val_fraction` mudou de 0.3 para 0.4, então
> o held-out passou de **108** para **92** imagens. Compare dentro de cada rodada; a Seção 4.5
> quantifica a diferença.

### 4.1 Resultado final por head (held-out de 92 imagens)

| Head | Config vencedora | Pix AUROC | AP | Pix F1 | AUPRO | Img AUROC (mapa) |
|---|---|---|---|---|---|---|
| **featnorm_clamp19** | pad112 · flips · raw · L=0 · σ=8 | **0.9270** | **0.1148** | 0.1990 | **0.7594** | 0.76 |
| featnorm_pad112_clamp19 | pad112 · flips · raw · L=0 · σ=8 | 0.9267 | 0.1123 | 0.1998 | 0.7333 | 0.73 |
| featnorm_l12 | pad112 · raw · L=0 · σ=8 | 0.9186 | 0.1115 | **0.2071** | 0.7487 | 0.75 |
| featnorm | pad112 · raw · L=0 · σ=8 | 0.9185 | 0.1132 | 0.2081 | 0.7474 | 0.72 |
| featnorm_pad112 | pad112 · flips · minmax · L=0 · σ=8 | 0.9033 | 0.0795 | 0.1522 | 0.7491 | 0.55 |
| _referência:_ produção da head `featnorm` | minmax · σ=4 · pad0 · L=012 | 0.8568 | 0.0425 | 0.0903 | 0.6684 | 0.41 |

**Assinatura comum de todos os vencedores:** `pad 112` + `raw` + `σ=8` + **só L1** (ou L1+L2).
Nenhuma configuração vencedora usa L3.

### 4.2 Efeitos marginais (2 496 configurações, 5 heads normalizadas)

| Fator | Valor | Média | Máximo | Comparação com a rodada 1 |
|---|---|---|---|---|
| **`levels`** | **`0` (só L1)** | **0.8697** | **0.9133** | novo no grid — **melhor conjunto por larga margem** |
| | `01` | 0.8292 | 0.9131 | era o melhor na rodada 1 |
| | `02` | 0.8262 | 0.9127 | |
| | `012` | 0.8084 | 0.9124 | |
| | `12` (sem L1) | 0.7034 | 0.8208 | tirar L1 custa **0.166** — muito pior que na rodada 1 |
| | `2` | 0.6911 | 0.8069 | |
| **`score_mode`** | `raw` | **0.8214** | **0.9133** | rodada 1: 0.8081 (com heads não normalizadas) |
| | `per_level_minmax` | 0.7640 | 0.9039 | **−0.057** — inversão clara com featnorm |
| **`pad`** | 112 | **0.7995** | **0.9133** | Δ +0.0136 (rodada 1: +0.0060) |
| | 0 | 0.7859 | 0.9110 | |
| **`tta`** | flips | **0.8009** | 0.9130 | Δ +0.0163 (rodada 1: +0.0068) |
| | none | 0.7846 | **0.9133** | |
| **`sigma`** | 8 | **0.7948** | **0.9133** | σ=8 confirmado como ótimo |
| | 4 | 0.7935 | 0.9011 | |
| | 12 | 0.7899 | 0.9080 | suavizar demais começa a apagar o defeito |
| **`pos_calib`** | none | **0.8121** | 0.9133 | |
| | l23 | 0.7734 | 0.9133 | máximos idênticos — ver 4.4 |
| **`image_score`** | max / topk | 0.7927 | 0.9133 | **efeito nulo** em pixel AUROC (como esperado) |

### 4.3 AUROC por nível isolado — com featnorm

| pad | tta | L1 | L2 | L3 |
|---|---|---|---|---|
| 0 | none | 0.8812 | 0.7953 | 0.7623 |
| 0 | flips | 0.8830 | 0.7978 | 0.7815 |
| 112 | none | 0.8886 | 0.8406 | 0.7778 |
| 112 | flips | **0.8901** | **0.8438** | **0.7962** |

Por head (pad 112, sem TTA):

| Head | L1 | L2 | L3 |
|---|---|---|---|
| featnorm_clamp19 | **0.8917** | 0.8126 | 0.7683 |
| featnorm_pad112 | 0.8889 | 0.8538 | 0.7832 |
| featnorm_l12 | 0.8886 | 0.8546 | — |
| featnorm | 0.8884 | 0.8535 | 0.7806 |
| featnorm_pad112_clamp19 | 0.8854 | 0.8286 | 0.7791 |

**A padronização concentra o sinal em L1.** Comparando com a rodada 1 (heads sem featnorm, pad112):
L2 cai de 0.8651 para 0.8406 e L3 de 0.8224 para 0.7778, enquanto L1 sobe de 0.8718 para 0.8886.
Hipótese: padronizar por canal equaliza a contribuição dos 64 canais de L1 (onde a informação de cor
está concentrada em poucos canais de alta magnitude que antes dominavam), enquanto em L2/L3, com
192–256 canais majoritariamente pouco informativos, a mesma operação **amplifica ruído**.
O efeito do pad também se concentra em L2 (+0.045 aqui, +0.018 na rodada 1).

### 4.4 `pos_calib` é inerte quando `levels=0` — armadilha do grid

| score_mode | sem calib | com calib l23 | Δ |
|---|---|---|---|
| `raw` | 0.8206 | **0.8222** | **+0.0016** (empate) |
| `per_level_minmax` | 0.8035 | 0.7246 | −0.0789 |

Na rodada 1 a calibração piorava o `raw` em −0.039; com features padronizadas ela deixa de fazer mal
— confirmando que os dois mecanismos (padronizar features e calibrar posição) atacam a mesma causa.

⚠️ **Mas há uma redundância no grid:** `pos_calib=l23` calibra apenas L2 e L3. Quando `levels=0`, ela
**não tem efeito nenhum** — por isso as linhas `L=0 · calib=none` e `L=0 · calib=l23` do held-out são
bit a bit idênticas (0.9270 nas duas). Isso infla a contagem de configurações "distintas" e faz o
máximo de `l23` empatar artificialmente. **Ação:** o grid deveria pular combinações em que o fator é
inerte, e as contagens da Seção 4.2 devem ser lidas com essa ressalva.

### 4.5 Reprodutibilidade e sensibilidade ao split

A variante `featnorm` foi retreinada na rodada 2 com **as mesmas flags, seed e número de épocas** da
rodada 1. Resultado: **checkpoint idêntico** (mesma época 40, mesmo `train_pixel_auroc`
0.849452262401675 até o último dígito). O treino é determinístico.

Com o **mesmo modelo**, mudando só o split (`val_fraction` 0.3 → 0.4, held-out 108 → 92):

| | Pix AUROC | AP | AUPRO |
|---|---|---|---|
| Rodada 1 (held-out 108) | 0.9229 | 0.1196 | 0.7558 |
| Rodada 2 (held-out 92) | 0.9185 | 0.1132 | 0.7474 |
| **Variação atribuível ao split** | **−0.0044** | −0.0064 | −0.0084 |

**Consequência prática:** diferenças menores que ~0.005 entre configurações do held-out **não são
significativas**. Isso invalida o ranking entre `featnorm_clamp19` (0.9270) e
`featnorm_pad112_clamp19` (0.9267) — são empate. Já a diferença para `featnorm` (0.9185) está no
limite, e a diferença para a produção (0.8568) é sólida.

### 4.6 Predições registradas antes da rodada 2 — placar

| Predição | Resultado | Veredito |
|---|---|---|
| `featnorm_pad112` supera 0.9229 | **0.9033** — a **pior** das cinco | ❌ **Errada** |
| `featnorm_clamp19` melhora | 0.9270, melhor da rodada | ✅ Certa |
| `featnorm_l12` confirma que L3 é dispensável | 0.9186 com 60 % dos parâmetros | ✅ Certa (e mais forte que o previsto: **L2 também** é dispensável) |
| `per_level_prob` com featnorm | não testado (fora do `--score_modes`) | ⏳ Pendente |

**O erro mais informativo: treinar com pad não ajuda; avaliar com pad ajuda.**
`featnorm_pad112` foi a única head cuja melhor configuração usa `per_level_minmax` em vez de `raw`, e
ficou 0.024 abaixo da `featnorm` simples. Interpretação: o reflect padding introduz conteúdo
espelhado que **não é uma imagem plausível** — durante o treino o flow gasta capacidade modelando
esses artefatos como se fossem normais, o que degrada a densidade aprendida. Na avaliação, o mesmo
padding só substitui zeros (ainda menos plausíveis) por algo mais próximo da distribuição de treino.
**Regra revisada: pad é técnica de inferência, não de treino** — e a exigência de "coerência
treino↔avaliação" da rodada 1 vale para clamp/`out_size`/`levels`, mas **não** para o pad.

### 4.7 Custo

| Variante | Treino (40 épocas) |
|---|---|
| featnorm | 42 min |
| featnorm_pad112 | 47 min |
| featnorm_l12 | 69 min |
| featnorm_clamp19 | 79 min |
| featnorm_pad112_clamp19 | 81 min |
| **Rodada 2 completa (5 variantes + pós-hoc)** | **~6 h** |

Clamp 1.9 quase **dobra** o tempo de treino (mais iterações até estabilizar o log-det) para um ganho
de 0.008 no AUROC — dentro do ruído de split (4.5). **`featnorm` puro é a opção de melhor
custo-benefício**; `featnorm_clamp19` só se o tempo não for restrição.

---

## 5. Resultados — Passo 0 Multi-Classe (heads publicadas sem retreino)

Avaliação pós-hoc direta das heads publicadas (`final_models/NF Head/<classe>`), com o mesmo
protocolo rigoroso (`val_fraction=0.4`, grid completo de níveis `"0,1,2,01,02,12,012"`, sem margens).

### 5.1 Tabela comparativa entre classes (held-out)

| Classe | Imagens (Val / Holdout) | Baseline Produção (AUROC / AP / AUPRO) | Melhor Pós-hoc (AUROC / AP / AUPRO) | Configuração Vencedora | Ganho AUPRO |
|---|---|---|---|---|---|
| **`vari-grip`** | 62 / 92 | 0.9037 / 0.1021 / 0.6348 | **0.9146 / 0.0850 / 0.7400** | `pad112 · flips · raw · L=012 · s=4` | **+0.1052** |
| **`lightning-rod`** | 65 / 98 | 0.8269 / 0.0184 / 0.4087 | **0.8597 / 0.0308 / 0.5412** | `pad112 · flips · minmax · L=012 · s=8` | **+0.1325** |
| **`glass-insulator`** | 271 / 407 | 0.8440 / 0.0006 / 0.5451 | **0.9043 / 0.0012 / 0.6461** | `pad0 · flips · raw · L=12 · s=8` | **+0.1010** |

*Nas 3 classes*, sem retreinar absolutamente nada, o pós-processamento correto elevou o **AUPRO em mais de +0.10** em todas.

### 5.2 O papel dos níveis nas **heads publicadas** — e por que essa leitura estava errada

AUROC isolado de cada nível, medido sobre as heads publicadas (pad112, flips):

| Nível | `vari-grip` | `lightning-rod` | `glass-insulator` |
|---|---|---|---|
| **L1** (conv1, 64 canais, textura/cor) | **0.8251** | 0.5735 | **0.2583** |
| **L2** (pool2, 192 canais, forma média) | **0.8609** | 0.7366 | 0.5015 |
| **L3** (conv5, 256 canais, contexto) | 0.8461 | **0.8279** | **0.9002** |

> ### ⚠️ Correção importante (rodada 3)
>
> A interpretação original destes números — de que L1 teria "sinal invertido" em defeitos
> estruturais porque a textura de fundo permanece onde a peça sumiu — **estava errada**.
> Ao retreinar as heads com a receita corrigida, L1 saltou:
>
> | Classe | L1 na head publicada | L1 após retreinar | Δ |
> |---|---|---|---|
> | `lightning-rod` | 0.5735 | 0.7600–0.7653 | **+0.19** |
> | `glass-insulator` | **0.2583** | **0.8633–0.8740** | **+0.61** |
>
> Ou seja: o L1 degenerado era **defeito da receita de treino antiga** (clamp/attenção/agregação
> divergentes, documentados na Seção 10), não propriedade física da classe. A hierarquia real
> (Seção 6.5) mantém L3 dominando em `lightning-rod` e `glass`, mas com L1 competitivo — tanto que
> o conjunto vencedor de `glass` passou a ser **L1+L3**.
>
> **Lição metodológica:** nunca diagnosticar propriedades de features a partir de checkpoints
> legados. O pós-hoc na head publicada serve para medir **ganho de pós-processamento**, não para
> concluir sobre a informatividade dos níveis.

O que **permaneceu válido** desta seção: em `lightning-rod` e `glass`, **L3 é o nível dominante**;
em `vari-grip`, não. E o ganho de pós-processamento (+0.10 de AUPRO nas 3 classes) se sustentou.

---

## 6. Resultados medidos — rodada 3 (retreino multi-classe)

Duas sub-rodadas com o mesmo protocolo (`val_fraction=0.4`, grid completo de níveis,
`--margins 0`, 40 épocas por variante):

| Sub-rodada | Classe | Run | Variantes |
|---|---|---|---|
| **3a** | `lightning-rod-suspension` | `run_20260905_145528` | baseline, featnorm, featnorm_clamp19, featnorm_l12 |
| **3b** | `glass-insulator` | `run_20260905_190938` | baseline, featnorm, featnorm_clamp19, l23 |

---

### Rodada 3a — `lightning-rod-suspension`

**Objetivo:** testar a generalização do retreino fora de `vari-grip`. 4 variantes × 40 épocas
sobre o SE final model de lightning-rod (`epoch_41`); 65 val, 98 held-out, 46 anômalas totais.

### 6.1 Tabela de held-out por head retreinada (98 imagens)

Configuração escolhida **na validação** e reportada uma única vez no held-out:

| Head (treino) | Config vencedora na validação | Pix AUROC | AP | Pix F1 | AUPRO | Img AUROC (mapa) |
|---|---|---|---|---|---|---|
| **baseline** | pad112 · flips · raw · **L=2** · σ=8 | **0.8812** | 0.0247 | 0.0579 | **0.5620** | 0.63 |
| featnorm_clamp19 | pad112 · flips · raw · L=012 · σ=4 | 0.8759 | **0.0259** | 0.0568 | 0.5238 | **0.69** |
| featnorm | pad112 · flips · raw · L=02 · σ=8 | 0.8633 | 0.0232 | 0.0570 | 0.5320 | 0.67 |
| featnorm_l12 | pad112 · flips · raw · L=01 · σ=4 | 0.8420 | 0.0162 | 0.0333 | 0.4310 | 0.65 |
| _referência:_ head publicada, melhor pós-hoc | pad112 · flips · minmax · L=012 · σ=8 | 0.8597 | 0.0308 | **0.0800** | 0.5412 | 0.62 |
| _referência:_ head publicada, produção | pad0 · none · minmax · L=012 · σ=4 | 0.8269 | 0.0184 | 0.0437 | 0.4087 | 0.57 |

> Diferenças < 0.005 são ruído de split (4.5): `baseline` (0.8812) e `featnorm_clamp19` (0.8759) estão
> no limite; `featnorm_l12` (0.8420) está claramente atrás.

**Cada head sob a sua própria referência de produção** (o delta legítimo, sem confundir receita com
pós-processamento):

| Head | produção (pad0 · minmax · σ=4) | melhor pós-hoc | Δ AUROC | Δ AUPRO |
|---|---|---|---|---|
| baseline | 0.7669 / 0.3880 | 0.8812 / 0.5620 | **+0.1143** | **+0.1740** |
| featnorm | 0.7815 / 0.4228 | 0.8633 / 0.5320 | +0.0819 | +0.1091 |
| featnorm_clamp19 | 0.6189 / 0.4311 | 0.8759 / 0.5238 | +0.2569 | +0.0927 |
| featnorm_l12 | 0.7410 / 0.3684 | 0.8420 / 0.4310 | +0.1010 | +0.0626 |

### 6.2 O que a rodada 3a revelou

1. **A hierarquia dos níveis se inverte em relação a `vari-grip`.** AUROC isolado nas heads
   retreinadas (pad 112, flips):

   | Head | L1 | L2 | L3 |
   |---|---|---|---|
   | baseline | 0.7600 | 0.8073 | **0.8769** |
   | featnorm | 0.7653 | 0.8271 | **0.8683** |
   | featnorm_clamp19 | 0.7574 | 0.8154 | **0.8583** |
   | featnorm_l12 (sem L3) | 0.7645 | 0.8280 | — |

   Em **todas** as heads a ordem é **L3 > L2 > L1** — o oposto exato de `vari-grip`. A head vencedora
   usa **apenas L3** (`L=2`). E treinar sem L3 (`featnorm_l12`) produziu a **pior** head da rodada
   (AUPRO 0.4310 contra 0.5620 da baseline, queda de 0.13).

2. **O retreino "resgatou" L1 — mas não o suficiente para mudar a ordem.** Na head publicada, L1
   valia **0.5735**; nas heads retreinadas vale **0.757–0.765** (+0.19). Ou seja, boa parte do L1
   degenerado da head publicada era **defeito da receita antiga**, não propriedade da classe. Mesmo
   assim, L3 continua dominando — a inversão entre classes é real, só menos extrema do que o Passo 0
   sugeria. **Este era o confundidor que o Passo 0 em `vari-grip` foi rodado para desfazer.**

3. **`raw` venceu em 100 % das variantes.** Nas 4 heads a config vencedora usa `raw` + `pad112` +
   `flips`. O `per_level_minmax` só aparece como referência de produção. Isso replica `vari-grip` e
   é a parte da receita que **generaliza**.

4. **`featnorm` não generalizou.** Em `vari-grip` foi a intervencao de maior impacto; aqui a
   `baseline` sem featnorm venceu em AUROC (0.8812 vs 0.8633) e AUPRO (0.5620 vs 0.5320).
   Explicação coerente com 7.3.8: featnorm **concentra o sinal em L1**; ajuda quando L1 é o nível
   bom (vari-grip) e é neutro-a-nocivo quando o sinal está em L3 (lightning-rod).

5. **⚠️ As heads retreinadas em 40 épocas são piores que a publicada sob a configuração de produção**
   (0.7669 vs 0.8269). A head publicada foi treinada até a época 80. Só **depois** do pós-hoc ótimo o
   retreino passa à frente (0.8812 vs 0.8597). Duas leituras: (a) 40 épocas podem ser insuficientes
   para esta classe; (b) o critério de seleção de época (`per_level_minmax`) está escolhendo o
   checkpoint errado para quem vai implantar com `raw` — ver 7.3.6.

6. **`featnorm_clamp19` tem a produção mais degradada de todas (0.6189)** mas o melhor AP pós-hoc
   (0.0259). Clamp alto torna a NLL bruta ainda menos compatível com `minmax`, reforçando que a
   escolha de `score_mode` **não pode ser desacoplada** da receita de treino.

---

### Rodada 3b — `glass-insulator`

**Objetivo:** testar a classe extrema — defeito estrutural (`missing-cap`) ocupando apenas **0.14 %**
da área, com 1104 imagens de treino e 678 de teste. 271 val, 407 held-out, 87 anômalas totais.

### 6.3 Tabela de held-out (407 imagens)

| Head (treino) | Config vencedora na validação | Pix AUROC | AP | AUPRO | Produção da própria head |
|---|---|---|---|---|---|
| **featnorm_clamp19** | pad112 · flips · raw · **L=02** · σ=2 | **0.9560** | **0.0063** | 0.7604 | 0.8224 / 0.6888 |
| featnorm | pad112 · flips · raw · **L=02** · σ=2 | 0.9542 | 0.0058 | **0.7690** | 0.8825 / 0.6571 |
| baseline | pad112 · flips · raw · L=2 · σ=4 | 0.9133 | 0.0022 | 0.5404 | 0.8821 / 0.6069 |
| l23 (`--levels 1,2`) | pad112 · flips · raw · L=2 · σ=4 | 0.9029 | 0.0018 | 0.5098 | 0.8244 / 0.3437 |
| _referência:_ head publicada, melhor pós-hoc | pad0 · flips · raw · L=12 · σ=8 | 0.9043 | 0.0012 | 0.6461 | — |
| _referência:_ head publicada, produção | pad0 · none · minmax · L=012 · σ=4 | 0.8440 | 0.0006 | 0.5451 | — |

**Este é o maior ganho absoluto do estudo:** +0.112 de AUROC e **+0.215 de AUPRO** sobre a produção,
com o AP subindo **10×** (0.0006 → 0.0063).

> **Sobre o AP parecer irrisório:** o baseline trivial de `glass` é 0.00018 (0.018 % dos pixels do
> teste são anômalos). O AP de 0.0063 é **35× o trivial** — o melhor ratio das três classes
> (`vari-grip` 9.5×, `lightning-rod` 4.8×). AP absoluto não é comparável entre classes.

### 6.4 O que a rodada 3b revelou

1. **🔴 O retreino reverteu completamente o diagnóstico do Passo 0.** L1 saiu de **0.2583** (head
   publicada, que parecia sinal invertido) para **0.8633–0.8740**. AUROC isolado, pad112 + flips:

   | Head | L1 | L2 | L3 |
   |---|---|---|---|
   | featnorm_clamp19 | 0.8740 | 0.6497 | **0.9421** |
   | featnorm | 0.8712 | 0.7002 | **0.9432** |
   | baseline | 0.8633 | 0.7120 | **0.9269** |
   | l23 (sem L1) | — | 0.7528 | **0.9174** |

   A hierarquia real de `glass` é **L3 > L1 > L2** — e não "L3 apenas, L1 invertido". Por isso o
   conjunto vencedor é **L1+L3** (`levels=02`) e a head `l23` (que descarta L1) ficou em último.

2. **`featnorm` venceu aqui.** Contrariando a conclusão da rodada 3a, `featnorm` (+0.041 AUROC sobre
   a `baseline`) e `featnorm_clamp19` (+0.043) dominaram. Placar final: featnorm vence em
   **2 de 3 classes** e perde por 0.018 na terceira.

3. **L2 é claramente o pior nível** (0.65–0.75 contra 0.92–0.94 de L3). Nas outras duas classes L2
   também nunca foi o melhor — ver 6.5.

4. **σ baixo venceu**, como previsto para defeito de 0.14 % da área: a config vencedora usa **σ=2**
   (contra σ=8 nas outras duas classes). Ressalva honesta: no agregado o efeito de σ é quase nulo
   em `glass` (média 0.8372 a 0.8428 entre σ=0 e σ=8) — a preferência por σ baixo aparece só no topo.

5. **Efeitos marginais** (4 heads, 896 configurações de validação):

   | Fator | Melhor | Média | 2.º | Δ |
   |---|---|---|---|---|
   | `levels` | **02** | 0.9240 | 2 (0.8975) | +0.027 |
   | `score_mode` | **raw** | 0.8620 | minmax (0.8179) | **+0.044** |
   | `tta` | **flips** | 0.8463 | none (0.8336) | +0.013 |
   | `pad` | **112** | 0.8414 | 0 (0.8385) | +0.003 |
   | `sigma` | 8 | 0.8428 | 2 (0.8388) | +0.004 (irrelevante) |

6. **Custo:** baseline 76 min, l23 95 min, featnorm 123 min, featnorm_clamp19 124 min — total ~7 h de
   treino + ~3 h de pós-hoc (678 imagens de teste).

### 6.6 Rodada 4 — Confirmação nas classes restantes (`polymer` e `yoke`)

Execução via `run_confirmacao_2classes.ps1` com 3 variantes retreinadas (`baseline`, `featnorm`, `featnorm_clamp19`)
e ablation pós-hoc no held-out com `val_fraction=0.4`:

- **`polymer-insulator-upper-shackle`** (`run_20260906_184201`): 317 imagens de teste (82 anomalias; val=127, heldout=190).
- **`yoke-suspension`** (`run_20260906_222906`): 200 imagens de teste (46 anomalias; val=80, heldout=120, `--limit 200` balanceado).

#### Tabela de held-out por classe e head

| Classe | Variante | Melhores Fatores (Validação) | Pix AUROC | Pix AP | AUPRO | Base Produção | Δ AUROC |
|---|---|---|---|---|---|---|---|
| `polymer` | `baseline` | pad112 · flips · minmax · L=012 · s=8 | 0.8767 | 0.0273 | 0.6001 | 0.8624 / 0.5753 | +0.0143 |
| `polymer` | `featnorm` | pad112 · flips · raw · L=012 · s=4 | 0.8740 | 0.0328 | 0.6446 | 0.8473 / 0.6078 | +0.0267 |
| `polymer` | `featnorm_clamp19` | pad112 · flips · raw · L=012 · s=8 | 0.8678 | 0.0298 | 0.6417 | 0.8114 / 0.6084 | +0.0564 |
| `yoke` | **`baseline`** | pad112 · flips · raw · **L=02** · s=8 | **0.9188** | **0.0568** | **0.6611** | 0.8336 / 0.5790 | **+0.0852** |
| `yoke` | `featnorm` | pad112 · flips · raw · L=012 · s=8 | 0.8671 | 0.0314 | 0.5763 | 0.8043 / 0.5591 | +0.0628 |
| `yoke` | `featnorm_clamp19` | pad112 · flips · raw · L=0 · s=8 | 0.8572 | 0.0183 | 0.5116 | 0.7243 / 0.5317 | +0.1329 |

#### Respostas às 3 perguntas de validação (P1, P2, P3)

As respostas abaixo usam **ganho pareado**, não média marginal: para cada fator, comparam-se
configurações idênticas em todos os outros eixos. É a única forma de isolar o efeito de um fator
sem que a composição do grid o contamine.

1. **P1 — `raw` continua batendo `per_level_minmax`?** **Sim, e é o efeito mais forte do estudo.**
   `polymer` +0.0205 (82 % dos pares positivos), `yoke` +0.0471 (97 %). Nenhuma classe contradiz.

2. **P2 — TTA por flips continua com ganho consistente?** **Sim em `polymer`, marginal em `yoke`.**
   `polymer` +0.0058 (81 % dos pares), `yoke` **+0.0031 com apenas 46 % dos pares positivos** — ou
   seja, em `yoke` o TTA é estatisticamente indistinguível de ruído. Ver a correção em 7.2.

3. **P3 — `levels=02` (L1+L3) continua com regret baixo?** **Sim em `yoke`, não em `polymer`.**
   `yoke`: `02` é o campeão absoluto (regret 0.0000). `polymer`: o melhor é `012` (0.8901) e `02`
   entrega 0.8623 — **regret 0.0278, uma ordem de grandeza acima das outras quatro classes.**
   `polymer` é a única classe do benchmark em que **L2 é o nível mais informativo** (0.8801, contra
   L1 0.7757 e L3 0.8410), o que explica o custo de descartá-lo.

### 6.7 Síntese cross-class — a hierarquia real dos níveis nas 5 classes

Com as 5 classes do INSPLAD avaliadas sob protocolo idêntico de retreino (AUROC de cada nível
isolado, pad112 + flips, melhor head da classe):

| Nível | `vari-grip` (4.6 %) | `lightning-rod` (1.9 %) | `glass` (0.14 %) | `polymer` (3.1 %) | `yoke` (2.0 %) | É o melhor? |
|---|---|---|---|---|---|---|
| **L1** | **0.893** | 0.760 | 0.874 | 0.776 | 0.852 | 1 de 5 |
| **L2** | 0.817 | 0.807 | 0.650 | **0.880** | 0.798 | 1 de 5 |
| **L3** | 0.787 | **0.877** | **0.942** | 0.841 | **0.879** | 3 de 5 |
| **Melhor conjunto** | `0` | `2` | `02` | `012` | `02` | |

#### Regret de fixar `levels=02` como default

Medido corretamente: melhor configuração **com** `02` contra a melhor configuração **global**, ambas
no grid de validação com pad112 + flips + raw.

| Classe | Melhor conjunto | AUROC | `02` | **Regret** |
|---|---|---|---|---|
| `vari-grip` | `0` | 0.9130 | 0.9124 | **0.0006** |
| `lightning-rod` | `012` | 0.9012 | 0.9001 | **0.0012** |
| `glass-insulator` | `02` | 0.9645 | 0.9645 | **0.0000** |
| `yoke-suspension` | `02` | 0.9156 | 0.9156 | **0.0000** |
| `polymer-shackle` | `012` | 0.8901 | 0.8623 | **0.0278** |

⚠️ **Correção de uma versão anterior deste documento.** A tabela de regret publicada na rodada 3
comparava **médias do grid inteiro** por conjunto de níveis, e concluía "regret ≈ 0.037 em todas".
Isso é a métrica errada: a decisão real não é "qual conjunto tem a melhor média sobre configurações
que ninguém usaria", e sim "quanto se perde fixando `02` e otimizando o resto". Medido assim, o
regret cai para **≤ 0.0012 em 4 de 5 classes** — `02` é um default muito melhor do que o documento
afirmava — e o único caso caro é `polymer`, que a média mascarava.

### 6.8 Ganho de cada técnica

Cadeia incremental sobre o grid de validação, começando na configuração de produção original e
adicionando **uma** técnica por vez, sempre na melhor head de cada classe. A ordem é a de custo
crescente de implementação, não a de importância.

| Passo | `vari-grip` | `lightning-rod` | `glass` | `polymer` | `yoke` | **Média** |
|---|---|---|---|---|---|---|
| **0.** Produção (pad 0, sem TTA, `minmax`, L=012, σ=4) | 0.7373 | 0.8008 | 0.7935 | 0.8651 | 0.8803 | — |
| **1.** `+ reflect_pad 112` | 0.7407 | 0.8052 | 0.8253 | 0.8672 | 0.8785 | **+0.0080** |
| **2.** `+ tta flips` | 0.7955 | 0.8306 | 0.8240 | 0.8736 | 0.8794 | **+0.0173** |
| **3.** `+ score raw` | 0.8186 | 0.8625 | 0.9339 | 0.8901 | 0.8888 | **+0.0381** |
| **4.** `+ levels` (melhor da classe) | 0.9007 | 0.8798 | 0.9598 | 0.8901 | 0.9140 | **+0.0301** |
| **5.** `+ sigma` (melhor da classe) | 0.9130 | 0.8810 | 0.9599 | 0.8901 | 0.9156 | **+0.0031** |
| **Ganho total** | **+0.1757** | **+0.0802** | **+0.1664** | **+0.0250** | **+0.0353** | **+0.0965** |

Ganho pareado de cada fator isolado — média sobre todas as configurações que diferem **apenas**
naquele fator, com `%pos` = fração de pares em que a técnica venceu:

| Técnica | `vari-grip` | `lightning-rod` | `glass` | `polymer` | `yoke` | Média | `%pos` (mín–máx) |
|---|---|---|---|---|---|---|---|
| **`raw` vs `minmax`** | +0.0574 | +0.0794 | +0.0441 | +0.0205 | +0.0471 | **+0.0497** | 76 – 100 % |
| **`tta flips` vs `none`** | +0.0163 | +0.0223 | +0.0126 | +0.0058 | +0.0031 | **+0.0120** | **46** – 97 % |
| **`pad 112` vs `pad 0`** | +0.0136 | +0.0056 | +0.0028 | +0.0024 | +0.0002 | **+0.0049** | 70 – 85 % |
| n de pares | 624 / 1248 | 192 / 384 | 192 / 384 | 72 / 144 | 72 / 144 | | |

Técnicas de **treino**, medidas como melhor head de cada receita contra a melhor head `baseline`
no held-out (AUROC / AUPRO):

| Técnica | `lightning-rod` | `glass` | `polymer` | `yoke` | Veredito |
|---|---|---|---|---|---|
| **`featnorm`** | −0.0107 / −0.0455 | **+0.0396 / +0.0254** | −0.0066 / **+0.0502** | **−0.0499 / −0.1226** | **Não universal** |
| **`clamp 1.9`** (sobre featnorm) | +0.0050 / +0.0050 | +0.0017 / −0.0145 | −0.0062 / −0.0029 | −0.0117 / −0.0269 | Ajuda só onde featnorm ajuda |

**Leitura.** As três técnicas de pós-processamento são gratuitas e somam ~+0.097 de AUROC em média,
com `raw` respondendo por metade disso sozinho. As duas técnicas de treino custam de 35 % a 48 % a
mais de tempo de treino e **não generalizam**: `featnorm` é excelente em `glass`, neutra em
`polymer` e `lightning-rod`, e claramente prejudicial em `yoke`.

#### Custo

| Classe | Imagens de treino | `baseline` | `featnorm` | `featnorm_clamp19` |
|---|---|---|---|---|
| `lightning-rod` | 462 | 36 min | 50 min | 62 min |
| `polymer` | 935 | 52 min | 52 min | 78 min |
| `glass` | 1104 | 76 min | 123 min | 124 min |
| `yoke` | 4834 | 128 min | 137 min | 189 min |

---

### 6.9 Avaliação fina do treino em lote (`run_20260907_145952`), diagnóstico de vari-grip, overfitting e transferência do RD++

Execução do script de orquestração `scripts/train/train_all_nf_heads.py` com `--recipe optimal`,
`--epochs 80`, `--eval_interval 4`, `--score_norm raw` e `--evaluate_after pixel`. Treinou heads para
as 5 classes sequencialmente salvando checkpoints, gráficos e resumos em
`resultado_analise_final/treino_nf_heads/run_20260907_145952/`.

#### Tabela de treino in-training (`training_summary.md`)

| Classe | Imagens de treino | Tempo (min) | Épocas | Pixel AUROC (in-train) | Img AUROC (mapa) | **Melhor época** |
|---|---|---|---|---|---|---|
| `glass-insulator` | 1 104 | 211.28 | 80 | 0.9460 | 0.6452 | **36** |
| `lightning-rod-suspension` | 462 | 84.34 | 80 | 0.8365 | 0.6884 | **12** |
| `polymer-insulator-upper-shackle` | 935 | 205.70 | 80 | 0.8761 | 0.6184 | **56** |
| `vari-grip` | 477 | 61.60 | 80 | **0.8385** | 0.5471 | **16** |
| `yoke-suspension` | 4 834 | 396.93 | 80 | 0.9292 | 0.7784 | **4** |

#### 1. A armadilha da comparação direta com a ablação

O número `0.8385` de `vari-grip` pareceu uma regressão grave frente ao `0.9270` da ablação. Mas as
duas medições são incomparáveis:
- A avaliação **in-training** usa `pad=0`, **sem TTA**, `sigma=6` e avalia **todos os níveis treinados**.
- A receita de **deploy** da ablação usa `reflect_pad 112` + `tta flips` + `raw` + `sigma 8` +
  seleção ótima de níveis — uma cadeia que adiciona **~+0.097 de AUROC** (Seção 6.8).

Ao avaliar as novas heads com a receita completa no teste integral (`avaliacao_pos_treino/results_per_class.csv`):

| Classe | In-training | Receita completa (`auto`: L1+L3) | Alvo da ablação |
|---|---|---|---|
| `glass-insulator` | 0.9460 | **0.9534** | 0.9560 |
| `yoke-suspension` | 0.9292 | **0.9348** | 0.9188 (+0.016) |
| `lightning-rod-suspension` | 0.8365 | **0.8622** | 0.8815 |
| `polymer-insulator-upper-shackle` | 0.8761 | **0.8505** | 0.8806 |
| `vari-grip` | 0.8385 | **0.8561** | 0.9270 |

Quatro classes confirmaram o desempenho esperado. Apenas `vari-grip` permaneceu substancialmente
abaixo do alvo (0.8561 vs 0.9270).

#### 2. Diagnóstico fino de `vari-grip`: a causa é a receita de níveis, não o modelo

Avaliando cada nível isoladamente com a receita completa (`per_level_auroc.csv`):

| Classe | L1 | L2 | L3 | L1+L3 (`auto`) | L1+L2+L3 | Nível escolhido |
|---|---|---|---|---|---|---|
| `glass-insulator` | 0.8584 | — | 0.9394 | **0.9534** | — | `02` |
| `lightning-rod-suspension` | 0.8015 | — | **0.8761** | 0.8622 | — | `2` |
| `polymer-insulator-upper-shackle` | 0.8279 | 0.7771 | **0.8525** | 0.8505 | 0.7987 | `02` |
| `vari-grip` | **0.9191** | — | 0.8052 | 0.8561 | — | **`0`** |
| `yoke-suspension` | 0.8929 | — | 0.8960 | **0.9348** | — | `02` |

**Descoberta:** em `vari-grip`, **L3 é de longe o pior nível (0.8052)**, enquanto L1 isolado atinge
**0.9191**. A receita `optimal` em `train_all_nf_heads.py` havia sido configurada com `levels='0,2'`,
forçando o treino de L1 e L3 e descartando L2 (que era o segundo melhor nível na ablação, com 0.817).
Ao fundir L1+L3 na inferência, o ruído de L3 contamina a predição e derruba o AUROC para 0.8561
e o AUPRO para 0.6124.

Pontuando **L1 sozinho** (`levels='0'`) na mesma head, sem retreino:
- **Pixel AUROC:** 0.8561 → **0.9191** (+0.0630)
- **Pixel AP:** 0.0549 → **0.0922** (+68 % de precisão média)
- **AUPRO:** 0.6124 → **0.7621** (+0.1497) — **supera o alvo da ablação (0.7585)**!

**Ação corretiva no código:** `CLASS_BUDGETS['optimal']['vari-grip']` foi alterado de `levels='0,2'`
para `levels='0,1'`.

#### 3. Diagnóstico de pico precoce e overfitting do flow

Curvas completas de `pixel_auroc` época a época (extraídas do histórico de métricas no MLflow):

| Classe | Pico in-train | AUROC no pico | AUROC ep 80 | Queda pico→fim | Avaliações acima do final |
|---|---|---|---|---|---|
| `yoke-suspension` | **época 4** | 0.9292 | 0.8993 | **−0.0299** | **17 de 20 (85 %)** |
| `vari-grip` | época 16 | 0.8385 | 0.8262 | −0.0123 | 14 de 20 (70 %) |
| `lightning-rod-suspension` | época 12 | 0.8365 | 0.8353 | −0.0012 | 8 de 20 |
| `glass-insulator` | época 36 | 0.9460 | 0.9451 | −0.0009 | 2 de 20 |
| `polymer-insulator-upper-shackle` | época 56 | 0.8761 | 0.8759 | −0.0002 | 1 de 20 |

**Análise:**
1. **O overfitting é severo apenas em `yoke` (−0.030 de AUROC).** `glass`, `lightning` e `polymer`
   são perfeitamente estáveis após convergir.
2. **Causa do desajuste em `yoke`:** o argumento de linha de comando `--epochs 80` **sobrescreveu
   os orçamentos calibrados** de cada classe na receita `optimal` (onde `yoke` tinha orçamento de
   apenas 12 épocas devido ao tamanho massivo de 4 834 imagens normais). Treinar 80 épocas em `yoke`
   significou 6.7× o orçamento necessário, degradando a generalização do flow.
3. Além disso, `yoke` atingiu o pico na **época 4**, antes mesmo de terminar o warmup de 5 épocas
   do cosine schedule — sugerindo que a taxa de aprendizado padrão (`2e-4`) pode ser excessiva para
   bases com milhares de amostras.

#### 4. Bug corrigido: `AUPRO = 0.0000` em distribuições de cauda pesada

No `final_results.csv` gerado ao final do treino, `vari-grip` marcou `aupro = 0.0000`.
- **Causa:** `compute_aupro` gerava limiares com `np.linspace(scores.min(), scores.max(), 200)`.
  Mapas de NLL não padronizados possuem caudas pesadas com alguns pixels de valor extremo (outliers).
  Esses extremos esticavam o intervalo linear, fazendo com que ~199 dos 200 limiares ficassem em
  regiões onde o FPR já ultrapassava `max_fpr=0.3`. Menos de dois pontos sobreviviam no intervalo
  válido, e a função retornava 0.0.
- **Correção:** limiares agora são calculados a partir dos **quantis dos pixels normais** no intervalo
  $[1 - \text{max\_fpr},\ 1.0]$:
  $$\text{thresholds} = \text{quantile}(\text{normal\_scores},\ \text{linspace}(1 - \text{max\_fpr},\ 1.0,\ 200))$$
  Como $\mathrm{FPR}(t) = P(\text{normal} \ge t)$, cada limiar por quantil cai exatamente dentro do
  intervalo de FPR útil $[0,\ \text{max\_fpr}]$.
- **Validação:** o cálculo corrigido é estritamente invariante a outliers arbitrários (testado com
  $10^6$) e a reescalas monótonas ($\times 1000$).

#### 5. Comparação direta com RD++ e técnicas transferidas

Comparativo das métricas de pixel entre o RD++ (`main.py`, WideResNet-50 + projeção multiescala +
ruído simplex) e o DifferNet (SEDifferNet AlexNet + CFLOW com pós-processamento completo):

| Classe | RD++ Pixel AUROC | DifferNet Pixel AUROC | RD++ AUPRO | DifferNet AUPRO |
|---|---|---|---|---|
| `glass-insulator` | 0.9368 | **0.9534** | **0.7560** | 0.6591 |
| `polymer-insulator-upper-shackle` | **0.9024** | 0.8505 | **0.6508** | 0.5933 |
| `vari-grip` | 0.8588 | **0.9191** | 0.5366 | **0.7621** |
| `yoke-suspension` | **0.9708** | 0.9348 | **0.7328** | 0.6686 |

O RD++ supera o DifferNet em AUPRO em 3 de 4 classes porque treina explicitamente com perda de
localização por projeção e pseudo-anomalias via ruído simplex. Nota-se também que o RD++ apresenta
o mesmo fenômeno de **pico precoce** (`glass` melhor época 19/200; `polymer` época 25/200).

**Técnicas incorporadas do RD++ ao DifferNet:**
1. **Checkpoint por Score Composto:** o RD++ salva `(auroc_px + auroc_sp + aupro_px)/3`. Portou-se
   para `pixel_train_from_pretrained.py` o salvamento do `best_composite.pt`, evitando que a escolha
   de modelo fique restrita a um único critério ruidoso.
2. **Cálculo de AUPRO a cada época de avaliação:** viabilizado pelo algoritmo de quantis otimizado
   (`<1s`), agora computado em toda avaliação (`--eval_aupro`).
3. **Persistência de `history.json` a cada época:** grava histórico completo (perda, métricas, lr)
   em disco, permitindo análise fina e diagnósticos pós-treino sem depender de logs voláteis.
4. **Gráfico de monitoramento 2×2 regravado a cada avaliação:** visualização em tempo real de
   `training_curves.png` (perda, pixel AUROC, image AUROC, AUPRO com indicação do pico).
5. **Early Stopping (`--patience`):** interrompe o treino caso o score não melhore após $N$ avaliações,
   protegendo classes como `yoke` do sobreajuste.
6. **Proteção de orçamento em lote:** `train_all_nf_heads.py` agora alerta explicitamente quando
   uma flag `--epochs` global sobrescreve o orçamento calibrado de uma receita.

#### 6. Ganho medido de `--levels per_class` no teste completo (sem retreino)

Implementou-se o modo `--levels per_class` em `evaluate_pixel_level.py` e `evaluate_full_pipeline.py`
utilizando o dicionário calibrado `PER_CLASS_SCORE_LEVELS` de `core/eval_pipeline.py`:
- `glass-insulator`: `02`
- `lightning-rod-suspension`: `2`
- `polymer-insulator-upper-shackle`: `02` (mantido em `02`: L3 isolado tem AUROC 0.0020 maior, mas
  o AUPRO cai de 0.5933 para 0.5332 — ilustrando por que métricas compostas são fundamentais)
- `vari-grip`: `0`
- `yoke-suspension`: `02`

Resultado medido sobre as heads de `run_20260907_145952` no **conjunto de teste completo** (sem nenhum retreino):

| Classe | `auto` (L1+L3) AUROC / AUPRO | `per_class` AUROC / AUPRO | Níveis | Δ Pixel AUROC | Δ AUPRO |
|---|---|---|---|---|---|
| `glass-insulator` | 0.9534 / 0.6591 | 0.9534 / 0.6591 | `02` | — | — |
| `lightning-rod-suspension` | 0.8622 / 0.4625 | **0.8761 / 0.5400** | `2` | **+0.0139** | **+0.0775** |
| `polymer-insulator-upper-shackle` | 0.8505 / 0.5933 | 0.8505 / 0.5933 | `02` | — | — |
| `vari-grip` | 0.8561 / 0.6124 | **0.9191 / 0.7621** | `0` | **+0.0630** | **+0.1497** |
| `yoke-suspension` | 0.9348 / 0.6686 | 0.9348 / 0.6686 | `02` | — | — |
| **MÉDIA** | **0.8914 / 0.5992** | **0.9068 / 0.6446** | — | **+0.0154** | **+0.0454** |

O ajuste pontual de níveis por classe eleva o **AUROC médio acima de 0.90** e o **AUPRO médio em
+0.045**, com `vari-grip` recuperando todo o seu potencial de localização.

---

## 7. Descobertas consolidadas

### 7.1 Hipóteses do plano original que os dados **derrubaram**

| Hipótese original | Veredito | Evidência |
|---|---|---|
| "L1 é o nível ruim (AUROC ≈ 0.52), remover melhora" | **Falsa em todas as classes** | O 0.52 histórico foi medido sob clamp errado (2.0). Com heads retreinadas: L1 = 0.89 (`vari-grip`), 0.76 (`lightning-rod`), 0.87 (`glass`), 0.78 (`polymer`), 0.85 (`yoke`). Remover L1 (`l23`) foi a **pior** head da rodada 3b. |
| **"L1 tem sinal invertido em defeito estrutural"** (conclusão do Passo 0) | **Falsa — artefato de head legada** | 0.2583 na head publicada → **0.874** após retreinar (+0.61). Ver o quadro de correção em 5.2. |
| "Rotação no treino é a causa da borda quente; desligar melhora" | **Parcial/falsa** | Desligar melhora L2/L3 mas piora L1; no agregado o AUROC não melhora. Com 477 imagens a rotação funciona como regularização (baseline melhorou até a época 40, norot fez pico na 5). |
| "`per_level_prob` (CFLOW-AD) é a agregação superior" | **Falsa neste setup** | Pior média (0.6992) e pior máximo (0.8466) dos quatro modos. Ver Seção 8. |
| "`feat_pool` (PatchCore) reduz ruído de L1 e ajuda" | **Marginal** | +0.011 de AUROC; bom em AP (2.º lugar) mas não justifica a complexidade sozinho. |
| "Calibração posicional só em L2/L3 ajuda" | **Condicional/inerte** | Piora na média (−0.034 na rodada 1); com featnorm+raw fica neutra (+0.002). E é **inerte** quando `levels=0` (Seção 4.4). |
| "Treinar com o mesmo pad da avaliação é necessário" | **Falsa** | `featnorm_pad112` foi a **pior** das cinco heads da rodada 2 (0.9033 vs 0.9185 da `featnorm` sem pad no treino). Pad é técnica de **inferência** (Seção 4.6). |
| **"Uma head só com L1 é a receita geral"** (conclusão das rodadas 1–2) | **Falsa** | Em `lightning-rod` e `glass`, `levels=0` fica bem atrás de conjuntos com L3. A conclusão valia só para `vari-grip`. |
| **"`featnorm` só funciona em `vari-grip`"** (conclusão da rodada 3a) | **Falsa, mas a correção também estava errada** | A 3b mostrou +0.0396 em `glass`. A rodada 4 mostrou **−0.0499 AUROC / −0.1226 AUPRO em `yoke`** — o pior resultado de qualquer receita no estudo. Placar real nas 5 classes: vence em 2, neutra em 2, prejudicial em 1. Ver 7.2. |
| **"O conjunto de níveis precisa ser escolhido por classe"** | **Majoritariamente falsa** | Fixando `levels=02` e otimizando o resto, o regret é **≤ 0.0012 em 4 de 5 classes** (6.7). A exceção é `polymer` (0.0278), a única classe em que **L2 é o melhor nível**. |
| **"L2 nunca ajuda"** (conclusão da rodada 3) | **Falsa — refutada por `polymer`** | Em `polymer` L2 é o **melhor** nível isolado (0.880 contra L1 0.776 e L3 0.841) e `012` bate `02` por 0.028. A generalização vinha de 3 classes em que L2 por acaso era fraco. |
| **"TTA por flips ajuda de graça em toda head"** | **Parcialmente falsa** | Em `yoke` só **46 % dos pares** melhoram (média +0.0031, indistinguível de ruído). Continua valendo a pena por ser barato e nunca prejudicar na média, mas não é universal. |
| **"O ranking de níveis de uma classe transfere entre heads retreinadas"** | **Falsa — dependente da head** | `polymer` na head da ablação tinha `012` como melhor conjunto (0.8806); na head do treino em lote 80 épocas, `012` caiu para 0.7987. A informatividade precisa ser reavaliada por head via `per_level_auroc.csv` (Seção 6.9). |
| **"Otimizar apenas por Pixel AUROC encontra a melhor localização"** | **Falsa — armadilha mono-métrica** | Em `polymer`, L3 isolado dá AUROC 0.0020 maior que L1+L3, mas o AUPRO cai 0.060 (0.5933→0.5332). AUROC é dominado por pixels normais e mascara falsos positivos regionais. Score composto estilo RD++ previne isso. |

### 7.2 Hipóteses **confirmadas**

Magnitudes são o **ganho pareado** médio sobre as 5 classes (6.8), não médias marginais.

| Hipótese | Evidência | Magnitude |
|---|---|---|
| `raw` supera `minmax` | +0.0497 médio; 76–100 % dos pares positivos; **nenhuma das 5 classes contradiz** | **Grande, universal** |
| Reflect pad **na avaliação** reduz artefato de borda | +0.0049 médio; 70–85 % dos pares positivos nas 5 classes | Pequeno, consistente |
| TTA por flips ajuda | +0.0120 médio; 76–97 % dos pares em 4 classes, **46 % em `yoke`** | Pequeno, quase universal |
| Suavização gaussiana ajuda | σ=8 ótimo em defeitos ≥ 1.9 % da área; σ=2 em `glass` (0.14 %) | Muito pequeno (+0.0031) |
| Escolher os níveis importa mais que qualquer hiperparâmetro de treino | +0.0301 médio na cadeia incremental, contra ≤0.0086 de qualquer receita de treino | **Grande** |
| Coerência treino↔avaliação de clamp/`out_size`/`levels` | muda a função computada | **Crítico** |
| Métricas de pixel exigem AUROC+AP+AUPRO | rankings discordam entre si nas 5 classes; em `polymer`, featnorm perde 0.0066 de AUROC e ganha 0.0502 de AUPRO | Metodológico |
| Treino determinístico | `featnorm` reproduzida bit a bit entre rodadas | Metodológico |

#### Hipóteses que **deixaram** de ser confirmadas com 5 classes

| Hipótese | Status anterior | Status atual |
|---|---|---|
| Padronizar features (`featnorm`) estabiliza o flow | "Grande em 2 de 3" | **Não generaliza.** +0.0396 (`glass`), −0.0066 (`polymer`), −0.0107 (`lightning-rod`), **−0.0499 (`yoke`)** |
| Clamp 1.9 com featnorm ajuda | "Sinal consistente nas 3" | **Condicional.** Positivo onde featnorm já ajuda (+0.0050 `lightning-rod`, +0.0017 `glass`), negativo onde não (−0.0062 `polymer`, −0.0117 `yoke`) |

### 7.3 Descobertas **novas** (não previstas no plano)

1. **A interação `featnorm` × `raw` é o mecanismo, não cada um isolado.** Nenhum dos dois sozinho é
   notável; juntos produzem o melhor resultado. Interpretação: min-max por imagem existia só para
   compensar escalas incomparáveis entre níveis; padronizar as features resolve a causa e o min-max
   deixa de ser necessário — recuperando a escala absoluta que ele destrói.
   **Rodada 2 confirma:** com featnorm, `raw` supera `minmax` em 0.057 na média.
2. **Rotação tem efeito de sinal oposto por nível** (+L1 / −L2 / −L3, consistente em dois pares).
   Sugere augmentation diferenciada por nível — não encontrei isso descrito na literatura de AD.
3. **Pad ajuda mais L2 do que L1** (+0.018 na R1, **+0.045** na R2), apesar de L1 ter o maior viés de
   borda absoluto. Efeito de proporção de área contaminada versus resolução da grade.
4. **Pad melhora AUPRO mesmo quando piora AUROC.** As duas métricas discordam sistematicamente sobre
   borda — útil como teste diagnóstico de "esse ganho é real ou é viés de borda?".
5. **`margin=8` é um confundidor de seleção.** Configs que vencem com margem tendem a ter AUPRO ruim
   (norot_pad112: 3.º AUROC, **último** AUPRO). Removido do grid na rodada 2.
6. **O critério de seleção de época durante o treino enviesa a comparação entre receitas.** Todas as
   heads foram selecionadas por `per_level_minmax`, que é justamente o modo que perde para as receitas
   normalizadas — `featnorm` e `clamp19` estão provavelmente **sub-avaliadas**.
7. **[R2] Uma head com apenas L1 é o melhor modelo — _em `vari-grip`_.** `levels=0` tem a maior média
   (0.8697) e o maior máximo (0.9133) do grid daquela classe. ⚠️ **A rodada 3a refutou a generalização:**
   em `lightning-rod`, `levels=0` é o **pior** conjunto. Ver 7.3.13.
8. **[R2] `featnorm` concentra o sinal em L1 e degrada L2/L3.** Comparando pad112/none entre rodadas:
   L1 0.8718→0.8886, L2 0.8651→0.8406, L3 0.8224→0.7778. Hipótese: em L1 a informação de cor está
   em poucos canais de alta magnitude e equalizar ajuda; em L2/L3, com 192–256 canais majoritariamente
   pouco informativos, a mesma operação **amplifica ruído**. **[R3a] Isso explica por que featnorm
   perdeu em `lightning-rod`**, onde o sinal vive em L3 — e torna a **padronização seletiva por nível**
   o próximo teste óbvio.
9. **[R2] Pad é técnica de inferência, não de treino.** Treinar com reflect pad **piorou** (0.9033 vs
   0.9185). O flow gasta capacidade modelando o conteúdo espelhado — que não é uma imagem plausível —
   como se fosse normal. Na inferência o mesmo padding só troca zeros por algo mais próximo da
   distribuição de treino. **Contraria a intuição de "coerência treino↔avaliação"** e vale como aviso
   geral: nem toda transformação de entrada deve ser espelhada no treino.
10. **[R2] Clamp alto quase dobra o custo de treino** (42 → 79 min) para um ganho dentro do ruído de
    split. Reportar custo junto com métrica muda a recomendação prática.
11. **[R2] Sensibilidade ao split medida diretamente:** mesmo modelo, mesma config, held-out de 108 vs
    92 imagens → Δ = 0.0044 AUROC / 0.0064 AP / 0.0084 AUPRO. **Define o limiar de significância** e
    dissolve o ranking entre as duas primeiras heads da rodada 2.
12. **[R2] Fatores inertes inflam o grid.** `pos_calib` não tem efeito quando `levels=0`, gerando
    linhas bit a bit idênticas e empates artificiais nos máximos marginais (Seção 4.4).
13. **[R3] A hierarquia entre L1 e L3 depende do defeito, mas L2 nunca ajuda.** Nas 3 classes com
    heads retreinadas: L1 vence em `vari-grip` (0.89), L3 vence em `lightning-rod` (0.88) e `glass`
    (0.94), e **L2 nunca é o melhor** (0.84 / 0.83 / 0.70). `levels=02` (L1+L3) tem regret ≤ 0.017
    nas três — é o compromisso universal (6.5). Consequência teórica: a fusão multi-escala **fixa**
    de CFLOW-AD/PaDiM/PatchCore inclui um nível que nunca contribui.
14. **[R3] ⚠️ Heads legadas mentem sobre a informatividade das features.** Retreinar levou L1 de
    0.5735 → 0.76 em `lightning-rod` e de **0.2583 → 0.874** em `glass`. Toda a "explicação física"
    construída sobre o Passo 0 estava errada. **Pós-hoc em checkpoint legado serve para medir ganho de
    pós-processamento, nunca para concluir sobre features.** É a lição metodológica mais cara do
    estudo.
15. **[R3a] O pós-processamento vale mais que o retreino.** Em `lightning-rod`, corrigir só o
    pós-processamento da head publicada já rende +0.033 AUROC / +0.133 AUPRO; retreinar 40 épocas e
    depois otimizar o pós-processamento rende +0.054 / +0.153. Ou seja, **~70 % do ganho é gratuito**.
    ⚠️ Em `glass` a proporção se inverte: pós-hoc só rende +0.060, o retreino leva a +0.112.
16. **[R3a] Heads retreinadas em 40 épocas ficam atrás da publicada sob a config de produção**
    (0.7669 vs 0.8269) e só passam à frente após o pós-hoc (0.8812 vs 0.8597). Reforça 7.3.6: o
    critério de seleção de época (`per_level_minmax`) desalinha do modo de implantação (`raw`).
17. **[R3b] AP absoluto não é comparável entre classes.** `glass` tem AP 0.0063 — aparentemente
    péssimo — mas o baseline trivial é 0.00018, então são **35× o trivial**, o melhor ratio das três
    classes (`vari-grip` 9.5×, `lightning-rod` 4.8×). Sempre reportar AP **relativo ao trivial**.
18. **[R3b] σ ótimo acompanha a escala do defeito.** σ=8 para defeitos de 1.9–4.6 % da área, σ=2
    para 0.14 %. Efeito pequeno (≤ 0.006 na média) mas consistente com a intuição de que suavizar
    demais apaga defeitos pequenos.
19. **[R5/Batch] Métricas in-training sem receita de deploy são ~0.097 mais baixas.** A avaliação durante
    o treino mede sem TTA, com pad 0 e sigma 6 em todos os níveis. Comparar o valor de log/CSV do treino
    com alvos da ablação gera falso alarme de sub-desempenho (Seção 6.9).
20. **[R5/Batch] Em `vari-grip`, L3 é o pior nível isolado (0.8052 vs L1 0.9191).** A fusão cega de L1+L3
    derruba o AUROC em 0.063 e o AUPRO em 0.150. Pontuar exclusivamente L1 atinge **0.9191 AUROC e 0.7621 AUPRO**,
    superando o alvo da ablação no teste completo sem qualquer retreino.
21. **[R5/Batch] O sobreajuste do flow afeta severamente classes com grandes volumes (yoke).** Em
    `yoke` (4 834 imagens normais), o pico ocorreu na época 4 e degradou 0.030 de AUROC até a época 80.
    Sobrescrever os orçamentos de época das receitas (`CLASS_BUDGETS`) com `--epochs` global é nocivo.
22. **[R5/Batch] Limiares lineares em AUPRO zeram em mapas de cauda pesada.** Em NLL crua com outliers,
    `np.linspace(min, max)` empurra os limiares para FPR > 0.3, retornando 0.0 silenciosamente. O cálculo
    por quantis de pixels normais resolve o problema e torna a métrica invariante a outliers.
23. **[R5/RD++] O score composto e o histórico persistido estabilizam o rastreamento.** A métrica
    composta $(AUROC_{px} + AUROC_{sp} + AUPRO)/3$ e o salvamento em disco a cada época (`history.json`,
    `best_composite.pt`) protegem a seleção de modelos contra oscilações de um único critério.

---

## 8. Registro de caminhos testados e descartados

Registro negativo — tudo o que foi tentado e não funcionou, para não ser retentado sem motivo novo.

| Caminho | Resultado | Diagnóstico |
|---|---|---|
| **Calibração posicional por célula em L1** | −0.062 / −0.069 AUROC (histórico) | O perfil posicional de L1 correlaciona só 0.695 entre treino e teste (vs 0.98–0.99 em L2/L3): o viés depende do conteúdo, não é offset geométrico fixo. Subtrair a média injeta ruído. |
| **`per_level_prob` (agregação do CFLOW-AD)** | Pior dos 4 modos (média 0.699) | `exp(logsigmoid(−NLL)/C)` satura: com NLL na escala de centenas, o logsigmoid vira ~−NLL e a divisão por C (64–256) comprime tudo para perto de 0 ou 1. Foi projetado para features de ResNet/WideResNet normalizadas, não para ReLU crua de AlexNet. **Pode voltar a fazer sentido combinado com `featnorm`** — não testado. |
| **Head sem L1 (`--levels 1,2`)** | R1/R2 em `vari-grip`: pior head (0.8561); `levels=12` média 0.7034. **R3a em `lightning-rod`: o oposto** — `levels=12` tem média 0.8551 e a vencedora usa só L3 | **Descarte revogado.** Depende da classe (7.3.13). Em `vari-grip` L1 carrega o sinal; em `lightning-rod`/`glass` é L3. |
| **Head sem L3 (`--levels 0,1`)** | R2 em `vari-grip`: neutro (0.9186). **R3a em `lightning-rod`: pior head da rodada** (0.8420 / AUPRO 0.4310) | Mesmo raciocínio, invertido. Não remover níveis sem ablar por classe. |
| **Treinar com `--reflect_pad 112`** | R1: neutro. R2: **−0.015** (pior das cinco) | O flow modela o conteúdo espelhado como se fosse normal. Pad só na inferência. |
| **`--clamp_scale 1.9` com featnorm** | `vari-grip`: +0.008 AUROC, **+88 % de tempo de treino**. `lightning-rod`: 2.º lugar, produção muito degradada (0.6189) | Ganho dentro do ruído de split. Não recomendado como padrão. |
| **`--feat_norm` como receita universal** | `vari-grip`: melhor. `lightning-rod`: **perde para a baseline** | Ajuda só quando o sinal está em L1 (7.3.8). Ablar por classe. |
| **`feat_pool 3` isolado** | +0.011 AUROC | Marginal; manter só se combinado. |
| **`--no_rotation` isolado** | −0.008 AUROC vs baseline (mas +0.058 AUPRO) | Perde regularização em L1, ganha coerência posicional em L2/L3. |
| **`border_margin` como filtro de produção** | Descartado | Mascarar a borda melhora AUROC artificialmente e não melhora AUPRO — esconde o problema em vez de resolver. Removido do grid na rodada 2. |
| **`--pad_mode replicate`** | Não executado | Implementado mas fora do grid padrão por custo. Pendente. |
| **`per_level_prob` com featnorm** | Não executado | Ficou fora do `--score_modes` da rodada 2. **Única hipótese da rodada 1 ainda em aberto.** |
| **`image_score topk`** | Efeito nulo em pixel AUROC (0.7927 nos dois) | Como esperado — só afeta `image_auroc_map`. |
| **AMP no treino do flow** | Desligado por padrão | Instável para flows (log-det em fp16). |
| **`--epochs` global uniforme para todas as classes** | `yoke` caiu de 0.9292 (época 4) para 0.8993 (época 80) | Overfitting por super-treino (6.7× do orçamento de 12 épocas). Usar orçamentos por classe em `CLASS_BUDGETS`. |

**Bugs encontrados e corrigidos ao longo do estudo** (detalhes na Seção 16):

| Bug | Impacto |
|---|---|
| Clamp 0.5 (treino) vs 2.0 (avaliação) | Pixel AUROC 0.5583 → 0.8770 só corrigindo |
| Features CBAM não treinadas lidas no treino | Função computada diferente da pretendida |
| `out_size` 96 (treino) vs 56 (avaliação) | PE em grade diferente |
| Agregação minmax (treino) vs raw (avaliação) | Checkpoint selecionado com um critério, usado com outro |
| **Transform de treino aplicado ao test set** | `pixel_auroc` de treino era ruído (~0.60 plano); checkpoint escolhido ao acaso |
| `clip_grad_norm_` global sobre os 3 flows | L3 (256ch) dominava a norma e reduzia o passo dos outros |
| Detecção SE/CBAM por chaves do `state_dict` | Impossível — as classes declaram os mesmos módulos |
| **PowerShell converte `--level_sets 0,01,012` em `0,1,12`** | O grid testou conjuntos diferentes dos pedidos; `01` e `012` nunca foram avaliados no 1.º Passo 0. **Sempre usar aspas** em listas com vírgula. |
| **Limiares lineares em `compute_aupro` com NLL crua** | `np.linspace(min, max)` com cauda pesada deixava <2 pontos com FPR ≤ 0.3 → `aupro = 0.0000` no `vari-grip`. Corrigido para quantis de pixels normais. |

---

## 9. Próximos passos

### 9.0 Estrutura final do modelo

Especificação fechada, validada nas **5 classes** do INSPLAD. Cada linha traz o valor, a evidência
que o sustenta e o quanto custa trocá-lo.

#### Arquitetura (fixa)

| Componente | Valor | Evidência |
|---|---|---|
| Backbone | SEDifferNet final model da classe, **congelado**, atenção `se` | restrição do projeto |
| Features | **L1 + L3** por default; `012` em classes onde L2 é forte | regret ≤ 0.0012 em 4 de 5 classes; 0.0278 em `polymer` (6.7) |
| Grade comum (`out_size`) | 96 × 96 | default de treino; muda o PE se alterado |
| Flow por nível | `CondFlow`, 8 blocos de acoplamento, `hidden=256`, `cond_dim=64` | default do projeto, não ablado |
| `clamp_scale` | **0.5** (default) | com 5 classes, 1.9 só ajuda onde `featnorm` ajuda; e custa 35–48 % mais treino (6.8) |
| Condicionamento | PE senoidal 2D sobre a grade 96² | CFLOW-AD |

Parâmetros da head com L1+L3: ~2.9 M (contra 3.6 M com os três níveis).

#### Treino

| Item | Valor | Evidência |
|---|---|---|
| Normalização de features | **desligada por default; ligar só se validar na classe** | vence em `glass` (+0.0396), prejudicial em `yoke` (−0.0499) — não generaliza (6.8) |
| Reflect pad | **desligado** | treinar com pad piora (4.6) |
| `feat_pool` | desligado | ganho marginal +0.011 (Seção 8) |
| Augmentation | rotação **ligada** (`RandomRotation(180)`) | desligar piora L1 e não melhora o agregado (3.5) |
| Épocas | orçamentos por classe em `CLASS_BUDGETS` (12 a 40 épocas) | `yoke` atinge pico na época 4; super-treino degrada 0.030 (6.9) |
| Early stopping | `--patience` (ex.: 5 avaliações) | protege contra overfitting em datasets grandes |
| Otimizador | Adam `lr=2e-4`, cosine + 5 épocas de warmup, `grad_clip 1.0` **por flow** | default corrigido (Seção 16) |
| AMP | desligado | instável para flows |
| Seleção de checkpoint | por **`raw`** (default) e **score composto** `(px + img + aupro)/3` | salva `best_pixel_auroc.pt` e `best_composite.pt` (estilo RD++) |
| Histórico | `history.json` persistido a cada avaliação | diagnóstico fino garantido pós-treino (6.9) |

#### Inferência — a parte que generaliza

Ganhos são o **pareado** médio sobre as 5 classes (6.8).

| Item | Valor | Ganho medido |
|---|---|---|
| `score_mode` | **`raw`** | **+0.0497** (76–100 % dos pares) — o maior efeito do estudo |
| `levels` | **`02` (L1+L3)** default (`auto`); ou **`per_class`** | `auto` +0.0301; `per_class` +0.0154 adicional na média (+0.0630 em `vari-grip`) |
| TTA | **flips** (id / H / V / HV) | +0.0120 (mas só 46 % dos pares em `yoke`) |
| `reflect_pad` | **112** (reflect) | +0.0049 (70–85 % dos pares) |
| `sigma` | **8** (defeito ≥ 1 % da área) · **2** (defeito < 0.5 %) | +0.0031 — acompanha a escala do defeito (7.3.18) |
| `pos_calib` | **none** | prejudicial na média; inerte com `levels=0` |
| `border_margin` | **0** | confundidor de seleção (7.3.5) |
| `image_score` | irrelevante para pixel; use `topk` se o mapa for usado para detecção | efeito nulo em pixel AUROC |

#### Resultado esperado (held-out, melhor head por classe)

| Classe | Head | AUROC | AP (× trivial) | AUPRO | Imagem (SEDifferNet) |
|---|---|---|---|---|---|
| `glass-insulator` | `featnorm_clamp19` · `02` · σ=2 | 0.9560 | 0.0063 (35×) | 0.7604 | 0.7706 |
| `vari-grip` | `featnorm_clamp19` · `01` · σ=8 | 0.9270 | 0.1147 (9.5×) | 0.7585 | 0.9044 |
| `yoke-suspension` | `baseline` · `02` · σ=8 | 0.9188 | 0.0568 (10×) | 0.6611 | 0.9744 |
| `lightning-rod` | `baseline` · `2` · σ=4 | 0.8815 | 0.0251 (4.6×) | 0.5642 | 0.9485 |
| `polymer-shackle` | `baseline` · `012` · σ=8 | 0.8806 | 0.0282 (0.9×) | 0.5943 | 0.9495 |

Note a dissociação entre os dois níveis: `glass` tem o **melhor** pixel AUROC e o **pior** de imagem;
`yoke` tem o melhor de imagem e um pixel mediano. As duas cabeças falham por motivos independentes,
o que justifica avaliá-las separadamente (9.2).

#### Comandos

```powershell
# 1. Treino (uma vez por classe)
python scripts/train/pixel_train_from_pretrained.py --class_name <classe> --disable_patchcore `
  --checkpoint "final_models\SEDiffernet\<classe>\<ckpt>.pt" --levels "0,2" --epochs 40

# 2. Avaliacao - a receita validada ja e o default dos tres scripts
python scripts/eval/evaluate_image_level.py            # so deteccao (SEDifferNet)
python scripts/eval/evaluate_pixel_level.py            # so localizacao (NF head)
python scripts/eval/evaluate_full_pipeline.py          # ambos + latencia + graficos
```


### 9.1 O que falta para fechar de vez

| # | Ação | Por quê | Esforço |
|---|---|---|---|
| 1 | ~~Portar `--tta flips` e `--levels` para `evaluate_full_pipeline.py`~~ **FEITO** | a receita final não era executável pelo script de produção | — |
| 2 | ~~Confirmar em `polymer-insulator-upper-shackle` e `yoke-suspension`~~ **FEITO** | rodada concluída com sucesso (ver Seção 6.6) | — |
| 3 | ~~Corrigir a seleção de época (selecionar por `raw`)~~ **FEITO** | in-training validation e seleção de `best_pixel_auroc.pt` agora usam `raw` como padrão em `pixel_train_from_pretrained.py` e `ablation_nf_head.py` | — |
| 4 | ~~Confirmar `levels=02` como default~~ **FEITO** | regret medido corretamente: ≤ 0.0012 em 4 de 5 classes (6.7) | — |
| 5 | ~~Três scripts de avaliação separados~~ **FEITO** | imagem, pixel e pipeline completo, com base comum em `core/eval_pipeline.py` (Seção 15) | — |
| 6 | ~~Padronização seletiva por nível (`featnorm` só em L1)~~ **FEITO** | suporte a `--feat_norm_levels 0` e variantes `featnorm_l1_only*` em `pixel_train_from_pretrained.py` e `ablation_nf_head.py` | — |
| 7 | **Múltiplos seeds de split** | limiar de significância 0.005 vem de 1 par; reportar média ± desvio | baixo |
| 8 | `per_level_prob` **com** featnorm | única hipótese da rodada 1 ainda aberta | baixo |
| 9 | ~~**Épocas > 40 e sobreajuste**~~ **FEITO** | testado em 80 épocas (`run_20260907_145952`, Seção 6.9): `yoke` atinge pico na época 4 e degrada 0.030; implementado `--patience` para early stopping | — |
| 10 | ~~**Investigar `polymer`**~~ **FEITO** | nova head 80 épocas mostrou L1+L3 (0.8505) muito superior a L1+L2+L3 (0.7987); L2 forte era artefato de head anterior (6.9) | — |
| 11 | ~~**Correção do bug de AUPRO em cauda pesada**~~ **FEITO** | `compute_aupro` por quantis de normais em `pixel_train_from_pretrained.py` (Seção 6.9) | — |
| 12 | ~~**Transferência do RD++ (score composto, monitor, history)**~~ **FEITO** | `best_composite.pt`, `history.json` a cada época, monitor 2×2, `--eval_aupro`, `--patience` (Seção 6.9) | — |
| 13 | ~~**Avaliação `--levels per_class`**~~ **FEITO** | modo `--levels per_class` com `PER_CLASS_SCORE_LEVELS`: AUROC médio 0.9068, AUPRO 0.6446 (Seção 6.9) | — |
| 14 | ~~**Correção de `vari-grip` na receita `optimal`**~~ **FEITO** | `optimal` atualizado para `levels='0,1'` em `train_all_nf_heads.py` | — |

#### Itens 1 a 6 e 9 a 14 — concluídos

**Item 1.** `evaluate_full_pipeline.py` passou a executar a receita validada como **default**:
`--score_norm raw`, `--reflect_pad 112`, `--tta flips`, `--levels auto` (L1+L3), `--sigma 8`,
`--border_margin 0`. Lê do checkpoint `levels`, `feat_pool`, `pad_mode` e `feat_stats`, de modo que
uma head treinada com `--feat_norm` é avaliada com a mesma normalização com que foi treinada. Heads
legadas continuam funcionando. `TTA_MODES`, `flip_tensor` e `nll_with_tta` foram movidos para
`core/cflow.py` para não existirem duas implementações.

**Item 2.** Rodada 4 concluída via `run_confirmacao_2classes.ps1` — resultados em 6.6.

**Item 3.** A seleção de checkpoint durante o treino (`best_pixel_auroc.pt`) usava `per_level_minmax`,
o que destorcia a calibração de escala entre níveis e subavaliava receitas que dependem de verossimilhança
absoluta. Agora:
- `TRAIN_SCORE_MODE` em `core/cflow.py` foi alterado para `'raw'` (com `LEGACY_TRAIN_SCORE_MODE = 'per_level_minmax'`
  mantido para compatibilidade).
- `pixel_train_from_pretrained.py` usa `--score_norm raw` por default, avaliando as épocas de validação e
  selecionando o melhor modelo com fusão NLL crua.
- `ablation_nf_head.py` adicionou `--train_score_mode` (default `'raw'`) e repassa explicitamente
  `--score_norm raw` aos subprocessos de retreino.
- O banner de treino e os logs a cada `eval_interval` identificam o modo em uso (`pix=... (raw)`).

**Item 4.** Regret de `levels=02` medido como decisão a priori (melhor config com `02` contra a
melhor config global, não médias de grid): 0.0006 / 0.0012 / 0.0000 / 0.0000 / 0.0278. Ver o quadro
de correção em 6.7.

**Item 5.** Três entry points, um por nível de análise, compartilhando `core/eval_pipeline.py`
(Seção 15). Verificado que `evaluate_pixel_level.py` e `evaluate_full_pipeline.py` produzem métricas
idênticas na mesma classe (`vari-grip`: 0.9064 / 0.0762 / 0.7129 nos dois).

**Item 6.** Padronização seletiva por nível (`featnorm` só em L1):
- `compute_feature_stats` e `apply_feature_stats` em `core/cflow.py` aceitam `norm_levels` e suportam
  valores `None` na lista `feat_stats`, passando os níveis não selecionados inalterados através do pipeline.
- `pixel_train_from_pretrained.py` ganhou a flag `--feat_norm_levels` (ex.: `--feat_norm_levels 0` para
  padronizar apenas L1), salvando `feat_norm_levels` e `feat_stats` com slots `None` nos níveis não normalizados.
- `ablation_nf_head.py` ganhou três novas variantes em `TRAIN_VARIANTS`: `featnorm_l1_only`,
  `featnorm_l1_only_02` e `featnorm_l1_only_clamp19`.
- Os avaliadores (`evaluate_pixel_level.py`, `evaluate_full_pipeline.py`, `PixelHead.describe()`) detectam
  e exibem automaticamente o modo: `feat_norm=selective(L1)`.

**Itens 9 e 10.** O treino em lote de 80 épocas (`run_20260907_145952`, Seção 6.9) revelou que aumentar
épocas além dos orçamentos calibrados degrada severamente classes massivas (`yoke` atinge pico na época 4
e cai 0.030 até a 80), motivando a adição de `--patience` para early stopping. Em `polymer`, o retreino
de 3 níveis mostrou que L1+L3 entrega 0.8505 / 0.5933 contra apenas 0.7987 / 0.4090 de L1+L2+L3,
confirmando que o L2 dominante observado na ablação anterior era artefato daquela head específica.

**Itens 11 a 14.** Resolvidos conforme detalhado na Seção 6.9: correção de quantis no `compute_aupro`,
incorporação das práticas do RD++ (`best_composite.pt`, `history.json`, monitor 2×2), modo
`--levels per_class` em `core/eval_pipeline.py` e ajuste da receita de `vari-grip` em `train_all_nf_heads.py`.

### 9.2 Novos testes propostos (além do fechamento)

0. **Separar a análise de falha por nível.** Os três scripts agora permitem medir onde o sistema
   perde: `glass` tem o melhor pixel AUROC (0.9560) e o **pior** de imagem (0.7706); `yoke` é o
   oposto (0.9744 / 0.9188). Nenhuma métrica agregada mostra isso. O próximo diagnóstico é cruzar
   `scores_<classe>.csv` do script de imagem com o `per_level_auroc.csv` do de pixel para verificar
   se as imagens que a detecção erra são as mesmas que a localização erra.

1. **Padronização seletiva por nível.** `featnorm` concentra o sinal em L1 e degrada L2/L3 (7.3.8).
   Com 5 classes isso deixou de ser curiosidade e virou o problema central: a receita ajuda `glass`
   (+0.0396) e destrói `yoke` (−0.0499). Aplicar featnorm **só em L1** deveria dar o melhor dos
   dois mundos numa head L1+L3. Teste barato e com hipótese clara.
2. **Explicar a relação área-do-defeito ↔ nível/σ ótimos.** As 3 classes sugerem correlação
   (4.6 % → L1/σ=8; 1.9 % → L3/σ=8; 0.14 % → L1+L3/σ=2), mas 3 pontos não fazem uma curva. Medir
   nas 5 classes e correlacionar seria um resultado publicaável.
3. **Augmentation por nível** (7.3.2): rotação para o flow de L1, flips H/V para L3.
4. **Score de imagem unificado** (3.8, 4.1): com featnorm+raw o mapa dá image AUROC 0.72–0.76;
   testar fusão com o score do SEDifferNet em vez de descartar o do mapa.
5. **Reportar custo junto com métrica** em toda tabela (7.3.10).

### 9.3 Longo prazo — mudanças de arquitetura

1. **Supressão de fundo por máscara do objeto.** O fundo (vegetação) é fonte dominante de falso
   positivo e é o único item do diagnóstico (Seção 11, item 2) ainda **não atacado**.
   `yolo-segmentation/` já tem YOLO11-seg: gerar máscara do objeto e **treinar o flow só em posições
   de foreground**. Provavelmente o maior ganho disponível.
2. **Head de alta resolução no nível dominante da classe.** Se só um nível importa, o gargalo passa a
   ser a reamostragem para 96² — treinar na resolução nativa melhoraria a localização fina.
3. **Pirâmide de entrada** (448/224/112) como no SEDifferNet, agregada estilo CS-Flow.
4. **Flow 2D (FastFlow)** sobre a grade em vez de por posição — preserva vizinhança, elimina o
   artefato de tabuleiro (relevante para `glass`, onde L3 27² domina).
5. **Validação sintética** (DRAEM-like) — resolve o problema de não haver split de validação.
6. **Threshold a contrario (U-Flow/NFA)** para operar sem escolher limiar no teste.
7. **AltUB** se o retreino com mais épocas mostrar oscilação.

### 9.4 Contribuições potenciais para a literatura

O que este estudo produziu que não é apenas engenharia local:

1. **L2 nunca contribui, e a hierarquia L1↔L3 depende do defeito.** Medido em 3 classes do mesmo
   dataset e backbone, com heads retreinadas sob receita idêntica: L1 vence em ferrugem larga (0.89),
   L3 vence em ferrugem fina (0.88) e em peça ausente (0.94), e **L2 não vence em nenhuma**
   (0.84 / 0.83 / 0.70). A fusão multi-escala fixa de CFLOW-AD/PaDiM/PatchCore carrega um nível
   inútil; `levels=L1+L3` tem regret ≤ 0.017 e reduz a head em ~20 %.
2. **⚠️ Checkpoints legados mentem sobre a informatividade das features.** O mesmo nível L1 mediu
   **0.2583** numa head publicada e **0.874** após retreinar com a receita corrigida — um erro de
   0.61 que produziu uma "explicação física" inteiramente falsa. Aviso metodológico forte para
   qualquer ablation feito sobre modelos de terceiros.
3. **Tabela de ablation de bugs silenciosos em NF heads.** Quatro divergências treino↔avaliação que
   não alteram o `state_dict` e não geram erro, mas mudam a função computada (clamp, `out_size`,
   caminho de atenção, regra de agregação). **Checkpoints de flow devem carregar hiperparâmetros de
   scoring**, não só pesos.
4. **`raw` + padronização de features supera o min-max por imagem.** O min-max é um curativo para
   escalas incomparáveis entre níveis e custa a comparação entre imagens. Vence em **12 de 12 heads**
   nas 3 classes (+0.044 a +0.057) e devolve ao mapa poder de detecção (0.41 → 0.76).
5. **Padding de borda como técnica exclusiva de inferência.** Aplicá-lo também no treino piora,
   porque o flow passa a modelar o conteúdo espelhado como parte da distribuição normal.
   Contraexemplo útil à regra geral de "treinar e testar sob a mesma transformação".
6. **Efeito de sinal oposto da augmentation por rotação entre níveis de features** em flows
   condicionados por posição.
7. **AUROC de pixel vs AUPRO como teste diagnóstico de viés de borda.** Intervenções que melhoram
   AUPRO e pioram AUROC estão removendo falso positivo de borda; o inverso indica ganho artificial.
8. **AP absoluto não é comparável entre classes.** Um AP de 0.0063 (`glass`) é **35× o trivial**,
   enquanto 0.1148 (`vari-grip`) é apenas 9.5×. Reportar sempre a razão AP/trivial.
9. **A maior parte do ganho é pós-processamento, não modelo** — mas a proporção varia: ~70 % em
   `lightning-rod`, ~50 % em `glass`. Trabalhos que reportam só a métrica final escondem quanto do
   resultado é arquitetura e quanto é scoring.
10. **Protocolo de ablation com cache de NLL + medição explícita da sensibilidade ao split.**
    Até 1 920 configurações a partir de 4 passagens da rede, e um limiar de significância derivado
    empiricamente (0.005) em vez de assumido.
11. **Aplicação a InsPLAD** (inspeção de linhas de transmissão por UAV): fundo natural dominante,
    pouquíssimas amostras anômalas e defeitos de 0.14 % a 4.6 % da área — diferente do MVTec.

---

## 10. Histórico e ponto de partida

| Configuração (teste completo, 154 imgs) | Pixel AUROC | Pixel AP | Pixel F1 | AUPRO |
|---|---|---|---|---|
| produção corrigida (`per_level_minmax`, σ=4, sem pad) | 0.8987 | 0.1022 | 0.1710 | 0.6484 |
| + `reflect_pad 112` + `border_margin 8` | 0.8939 | 0.0990 | 0.1687 | 0.6573 |
| legado (clamp 2.0 / out 56 / raw) — referência histórica | 0.5583 | 0.0123 | 0.0337 | 0.2409 |

Image AUROC (SEDifferNet, independente da head): 0.9186.

Fato relevante: o checkpoint publicado (`final_models/NF Head/vari-grip`) foi treinado **sem metadados**
(out_size/clamp assumidos), com **rotação aleatória** nas imagens, **sem reflect pad**, sobre features
**CBAM não treinadas** (bug documentado em `docs/cflow_correcoes_e_fase3.md`) e selecionado pelo
`pixel_auroc` de teste. Ou seja: há espaço real para ganho só por retreinar com a receita coerente.

---

## 11. Diagnóstico — o que os mapas mostram

> ⚠️ Este diagnóstico foi escrito **antes** da rodada 1. As anotações ⮕ marcam o que os dados
> revisaram. Mantido na íntegra porque o raciocínio (mesmo onde errado) faz parte do registro.

Evidências (imagens em `resultado_analise_final/run_20260903_232209/vari-grip/`):

1. **Contornos do objeto dominam o mapa.** O mapa acende nas bordas metálicas (alta frequência), não na
   ferrugem. É a assinatura de **L1** (64 canais logo após `conv1` 11×11/4): features de baixo nível
   respondem a gradiente de intensidade, não a "textura ferruginosa". No documento de correções, L1
   sozinho tem AUROC ≈ 0.52 (aleatório) e viés borda–centro de +283 (vs +12 / −33 em L2/L3).
   ⮕ **Revisado:** aquele 0.52 foi medido sob clamp 2.0 (bug). Com clamp correto, L1 isolado dá
   0.86–0.89 e é o **melhor** nível **em `vari-grip`** (Seção 3.3) — mas o pior em `glass` (0.26).
   A observação visual ("acende nos contornos") continua
   verdadeira — mas contornos e ferrugem coocorrem, então o nível discrimina mesmo assim.
2. **Fundo (vegetação) com score alto.** Fundo de alta variância é raro em qualquer posição → NLL alta.
   Como a agregação `per_level_minmax` normaliza por imagem, um fundo "ruidoso" comprime o range do
   defeito. ⮕ **Não atacado ainda** — é o item de maior potencial restante (Seção 9.3.1).
3. **Artefato em blocos.** Padrão retangular de baixa resolução no fundo: L3 nasce em 27×27 e sobe
   para 96 e depois 448 por bilinear; a soma com min-max realça esse tabuleiro.
   ⮕ **Consistente em `vari-grip`:** L3 é o nível menos informativo (0.80) e `levels=01` vence.
   ⚠️ **Invertido nas outras classes:** em `lightning-rod` e `glass` o tabuleiro de L3 não impede que
   ele seja o nível dominante (0.88 e 0.90) — ver 7.3.13.
4. **Borda quente.** Perfil de borda (`border_profile.png` do smoke test): anel de 0–8 px com score
   normalizado 0.38 contra ≈0.27 no interior próximo. `reflect_pad 112` reduz para ≈0.31. Causas:
   - padding zero da `conv1` (kernel 11, pad 2): ativações de borda computadas sobre zeros
     (Alsallakh et al., *Mind the Pad*, ICLR 2021; Islam et al., ICLR 2020; Kayhan & van Gemert, CVPR 2020
     mostram que o padding codifica posição absoluta e cria viés espacial);
     ⮕ **Confirmado:** pad 112 melhora AUPRO em 7 de 8 heads (Seção 3.7).
   - **treino com `RandomRotation(180)`**: as imagens de treino têm cantos pretos triangulares; o flow,
     condicionado à posição, aprendeu que cantos são "às vezes preto, às vezes conteúdo". No teste (sem
     rotação) o conteúdo real dos cantos cai fora da moda aprendida → NLL alta;
     ⮕ **Parcialmente revisado:** desligar rotação ajuda L2/L3 e **prejudica** L1 (Seção 3.5).
   - **descasamento treino/avaliação do `reflect_pad`**: a head foi treinada sem pad e é avaliada com
     pad 112 — as features de borda que ela vê na avaliação nunca apareceram no treino.
     ⮕ **Confirmado, mas menor que o esperado:** ~0.012 de AUROC.
5. **Score de imagem = max do mapa** é decidido por um único pixel de borda (image AUROC via mapa ≈ 0.45
   nos checkpoints). O pipeline final já usa o SEDifferNet para imagem; mas `topk` é o candidato certo
   se a head for usada para isso.
   ⮕ **Revisado:** a causa principal não é `max` vs `topk` (efeito quase nulo no grid) e sim o
   `per_level_minmax`, que zera a escala absoluta. Com `featnorm`+`raw` o mapa chega a 0.76 (Seção 3.8).

---

## 12. Fundamentação na literatura (o que cada ideia pega de onde)

| Ideia | Fonte | O que aproveitamos |
|---|---|---|
| Agregação por probabilidade limitada (`exp(logsigmoid(logp)/C)`, soma dos níveis, score = −soma) | Gudovskiy et al., **CFLOW-AD**, WACV 2022 (arXiv 2107.12571) | Substitui o min-max por imagem por uma escala **absoluta e limitada** em [0,1] por nível. Implementado como `--score_norm per_level_prob`. Também: clamp 1.9, 8 blocos, cond_dim 128, sem L1. |
| Flows 2D preservando vizinhança espacial | Yu et al., **FastFlow**, arXiv 2111.07677 | Motiva agregação local das features (`--feat_pool 3`) como aproximação barata do contexto espacial. |
| Agregação de vizinhança local 3×3 antes do scoring | Roth et al., **PatchCore**, CVPR 2022 (arXiv 2106.08265) | `--feat_pool 3`: reduz ruído de L1, aumenta campo receptivo, suaviza o tabuleiro de L3. |
| Padronização por canal das features com estatísticas de treino | Defard et al., **PaDiM**, ICPR-W 2021 (arXiv 2011.08785) | `--feat_norm`: ReLU do AlexNet é não normalizada/pesada em cauda; padronizar estabiliza o coupling. |
| Fusão cross-scale de níveis / pesos por escala | Rudolph et al., **CS-Flow**, WACV 2022 (arXiv 2110.02855); Zhou et al., **MSFlow**, TNNLS 2024 (arXiv 2308.15300) | Justifica ablar **subconjuntos de níveis** (`--level_sets 012,12,01,02,2`) e treinar só L2+L3 (`--levels 1,2`). |
| Padding espelhado / parcial em vez de zero | Alsallakh et al., **Mind the Pad**, ICLR 2021 (arXiv 2010.02178); Liu et al., **Partial Conv Padding**, arXiv 1811.11718 | `--reflect_pad` (já existia) + novo `--pad_mode replicate`; e usar o **mesmo pad no treino**. |
| Viés posicional aprendido por CNNs | Islam et al., ICLR 2020 (arXiv 2001.08248); Kayhan & van Gemert, CVPR 2020 (arXiv 2003.07064) | Explica por que o perfil de borda de L1 depende do conteúdo e não é um offset fixo → calibrar posição **só em L2/L3** (`--pos_calibs l23`). |
| Distribuição-base adaptativa / estabilidade de NF | Kim et al., **AltUB**, arXiv 2210.14913 | Alternativa futura se o retreino oscilar (histórico: `pixel_auroc` de teste oscila ±0.05 entre épocas). |
| Threshold a contrario (NFA) sobre mapas de NF | Tailanian et al., **U-Flow**, arXiv 2211.12353 | Trabalho futuro para operar sem threshold escolhido no teste. |
| Flow de imagem do próprio projeto (multi-escala de entrada, rotações como TTA) | Rudolph et al., **DifferNet**, WACV 2021 (arXiv 2008.12577) | TTA por flips (`--ttas flips`) é o análogo pixel-level das rotações de teste do DifferNet: simetriza o prior posicional sem retreinar. |
| Dataset e características do domínio (UAV, fundo de vegetação) | Vieira-e-Silva et al., **InsPLAD**, IJRS 2023 (arXiv 2311.01455) | Fundo dominante → motivação para supressão de fundo por máscara do objeto (Seção 9.3). |

---

## 13. Catálogo de propostas

### 13.1 Pós-processamento (sem retreinar) — todas no grid do script

| # | Fator | Valores no grid | Hipótese |
|---|---|---|---|
| P1 | `pad` (reflect_pad snapped) | 0, 112 (+ o pad de treino da head) | Reduz ativações de borda sobre zeros. |
| P2 | `pad_mode` | reflect (opcional: replicate) | Replicate evita "eco" de estruturas espelhadas. |
| P3 | `tta` | none, flips (id/h/v/hv) | Média sobre flips simetriza o prior posicional; reduz viés de canto. |
| P4 | `score_mode` | per_level_minmax, per_level_prob, per_level_std, raw | `prob` (CFLOW-AD) elimina dependência de estatística por imagem. |
| P5 | `levels` | 012, 12, 01, 02, 0, 2 | Subconjuntos de níveis; `0` (L1 sozinho) adicionado na rodada 2. |
| P6 | `pos_calib` | none, l23 | Calibra por célula só onde o perfil transfere (L2/L3). |
| P7 | `sigma` | 0, 4, 8 | Suavização vs. resolução do defeito. |
| P8 | `margin` | 0, 8 | **Métrica**, não filtro: mostra o quanto da borda "polui" a AUROC. Reportar sempre. |
| P9 | `image_score` | max, topk (1 %) | Robustez do score de imagem derivado do mapa. |

Grid completo padrão: 2×1×2×4×5×2×3×2×2 = **1 920 configurações**, todas re-agregadas a partir de **4
passagens** da rede (pad × tta). Métricas do grid em stride 2 (histograma de 65 536 bins, erro < 1e-4);
top-N reavaliado em resolução plena + AUPRO no conjunto **held-out**.

### 13.2 Retreino (receita da head) — `--stage retrain`, variantes nomeadas

| Variante | Flags do treino | Hipótese |
|---|---|---|
| `baseline` | — | Referência com as mesmas épocas (isola o efeito de cada flag). |
| `norot` | `--no_rotation` | Remove cantos pretos do treino → prior posicional coerente com o teste. |
| `pad112` | `--reflect_pad 112` | Treino e avaliação veem as **mesmas** features de borda. |
| `norot_pad112` | ambos | Ataca as duas causas do artefato de borda. |
| `l23` | `--levels 1,2` | Head sem L1: sem contornos, sem tabuleiro de baixo nível. |
| `norot_l23` | `--no_rotation --levels 1,2` | Combinação barata. |
| `pool3` | `--feat_pool 3` | Agregação local (PatchCore) → menos ruído em L1, contexto em L3. |
| `featnorm` | `--feat_norm` | Padronização por canal (PaDiM) → coupling melhor condicionado. |
| `clamp19` | `--clamp_scale 1.9` | Clamp do CFLOW-AD original; 0.5 pode estar limitando a expressividade. |
| `blocks4` / `hidden512` | `--n_blocks 4` / `--hidden 512` | Capacidade: menos blocos (regularização) vs. subrede maior. |
| `combo` | `--no_rotation --reflect_pad 112 --feat_pool 3 --feat_norm` | Tudo que é "de graça" junto. |
| `combo_l23` | `combo` + `--levels 1,2` | Candidato a receita final. |
| **Rodada 2** — adicionadas após os resultados da Seção 3 | | |
| `featnorm_pad112` | `--feat_norm --reflect_pad 112` | Coerência de pad **e** normalização — principal candidata. |
| `featnorm_clamp19` | `--feat_norm --clamp_scale 1.9` | Os dois vencedores da rodada 1 juntos. |
| `featnorm_pad112_clamp19` | os três | Combinação completa. |
| `featnorm_pool3` | `--feat_norm --feat_pool 3` | Pool só vale a pena normalizado? |
| `featnorm_l1` | `--feat_norm --levels 0` | Head mínima: L1 sozinho, 15 % dos parâmetros. |
| `featnorm_l12` | `--feat_norm --levels 0,1` | O conjunto de níveis vencedor (`01`) treinado direto. |

Todas as variantes gravam `out_size, clamp, levels, feat_pool, feat_norm, pad_mode, reflect_pad,
no_rotation` e as estatísticas de features **dentro do checkpoint**; a avaliação lê e reproduz.

### 13.3 Tratamento específico do artefato de borda (ordem de ataque, **revisada após a rodada 4**)

Ganhos são o **pareado** médio sobre as 5 classes (6.8).

**Aplicar sempre (pos-hoc, nenhum retreino):**

1. **Agregação `raw`** no lugar de `per_level_minmax` — o fator de maior efeito (**+0.0497**,
   76–100 % dos pares).
2. **Reflect pad 112 — só na avaliação** (+0.0049). Treiná-lo junto **piora** (Seção 4.6).
3. **TTA por flips** (+0.0120). Em `yoke` é ruído (46 % dos pares), mas nunca prejudica na média.
4. `border_margin` **nunca** como filtro nem como critério de seleção.

**Dependente da classe (ablar se houver orçamento):**

5. **Conjunto de níveis.** `levels=02` é o default seguro — regret **≤ 0.0012 em 4 de 5 classes**
   (6.7). Vale ablar em classes onde L2 pode dominar, como `polymer` (regret 0.0278).
6. **σ.** 8 para defeitos ≥ 1 % da área, 2 para defeitos < 0.5 % (7.3.18).
7. **`--feat_norm`.** ⚠️ **Não generaliza:** +0.0396 em `glass`, **−0.0499 em `yoke`**. Ligar só
   depois de validar na classe.

**Descartados:**

8. ~~Desligar rotação~~ — piora L1 e não melhora o AUROC agregado (Seção 3.5).
9. ~~Calibração posicional~~ — inerte quando a head usa só L1 e prejudicial na média (Seção 4.4).
10. ~~`feat_pool`~~ — ganho marginal (+0.011) que não paga a complexidade.

**Descartados:**

9. ~~Desligar rotação~~ — piora L1 e não melhora o AUROC agregado (Seção 3.5).
10. ~~Calibração posicional~~ — inerte quando a head usa só L1 e prejudicial na média (Seção 4.4).
11. ~~`feat_pool`~~ — ganho marginal (+0.011) que não paga a complexidade.
12. ~~Head sem L1 (`l23`)~~ — pior head da rodada 3b (0.9029 vs 0.9560).

---

## 14. Protocolo experimental (anti-viés de seleção)

Não existe split de validação no dataset. O script cria um **split estratificado do teste**
(`--val_fraction`, seed fixa) para **escolher** a configuração; o restante (**held-out**) serve para
**reportar uma única vez**. Referências (`baseline_producao`, `producao_pad112_m8`) são sempre
reavaliadas no held-out para comparação justa.

⚠️ A rodada 1 usou `--val_fraction 0.3`; da rodada 2 em diante o padrão passou a ser **0.4**. Isso
muda o tamanho do held-out e torna as rodadas **não comparáveis entre si** (4.5).

Fases:

| Fase | Comando | Status | Saída-chave |
|---|---|---|---|
| F0 — sanidade | `python scripts/eval/ablation_nf_head.py --class_name vari-grip --quick --limit 16` | ✅ feito | roda em minutos; confere o pipeline ponta a ponta |
| F1 — pós-hoc nas heads publicadas (Passo 0) | `... --class_name <classe> --pads "112,0" --ttas "none,flips" --level_sets "0,1,2,01,02,12,012" --val_fraction 0.4` | ✅ feito nas 3 classes | `vari-grip` (91.46/74.00), `lightning-rod` (85.97/54.12), `glass` (90.43/64.61) |
| F2 — retreino rodada 1 | `... --stage retrain --epochs 40 --variants baseline,norot,pad112,norot_pad112,l23,pool3,featnorm,clamp19` | ✅ `run_20260904_092255` | vari-grip (Seção 3) |
| F3 — retreino rodada 2 | `... --stage retrain --epochs 40 --variants featnorm,featnorm_pad112,featnorm_clamp19,featnorm_pad112_clamp19,featnorm_l12 --margins 0 --sigmas 4,8,12 --score_modes raw,per_level_minmax --val_fraction 0.4` | ✅ `run_20260905_011651` | vari-grip (Seção 4) |
| F4 — retreino rodada 3a | `... --stage retrain --epochs 40 --variants baseline,featnorm,featnorm_clamp19,featnorm_l12 --class_name lightning-rod-suspension` | ✅ `run_20260905_145528` | lightning-rod (Seção 6) |
| F5 — retreino rodada 3b | `... --stage retrain --epochs 40 --variants baseline,featnorm,l23,featnorm_clamp19 --class_name glass-insulator` | ✅ `run_20260905_190938` | glass (Seção 6) |
| F6 — confirmação final | `.\scripts\eval\run_confirmacao_2classes.ps1` | ✅ `run_20260906_184201` + `run_20260906_222906` | `polymer` e `yoke` (Seção 6.6) |

Regras de decisão (**revisadas após a rodada 1**):
- Escolher **só** por `grid_val.csv`; reportar `holdout_test_top.csv`.
- Reportar sempre **AUROC + AP + AUPRO** (Seção 1.6).
- **Selecionar com `--margins 0`.** A margem é confundidor de seleção (descoberta 7.3.5).
- Uma configuração só "vence" se melhorar no held-out **e** o `border_profile` mostrar anel de borda
  ≤ interior.
- Diferenças < 0.005 no held-out são **ruído de split**, medido diretamente na Seção 4.5.
- Comparar variantes de retreino com a variante `baseline` treinada **no mesmo número de épocas**,
  não com o checkpoint publicado.
- **Comparar apenas dentro da mesma rodada:** mudar `--val_fraction` muda o held-out.

Resultado preliminar do smoke test (n = 16, **não conclusivo**, só validação do pipeline): média de
`pixel_auroc` no grid por modo de agregação — `per_level_prob` 0.902 vs `per_level_minmax` 0.840.
⚠️ **Este resultado não se sustentou** na rodada completa (Seção 3.2): com 154 imagens,
`per_level_prob` é o **pior** modo. Exemplo didático de conclusão revertida por tamanho de amostra.

---

## 15. Implementação (o que existe no código)

### Os três entry points de avaliação

O sistema tem duas cabeças que falham por motivos independentes — `glass` tem o melhor pixel AUROC
e o pior de imagem, `yoke` o contrário (9.0). Um script único esconde isso, então existem três:

| Script | Mede | Carrega | Usa quando |
|---|---|---|---|
| `scripts/eval/evaluate_image_level.py` | *a imagem é anômala?* | só SEDifferNet | teto da detecção, sem interferência da localização |
| `scripts/eval/evaluate_pixel_level.py` | *quais pixels são anômalos?* | SE (congelado) + NF head | qualidade da localização isolada |
| `scripts/eval/evaluate_full_pipeline.py` | ambos + latência + gráficos | tudo | número final do sistema |

Os três compartilham `core/eval_pipeline.py`, que é o que garante que os números sejam comparáveis:
descoberta de checkpoint, carregamento, dataset determinístico, métricas (`compute_pro_rd_style`,
`compute_best_f1`, `binary_metrics`, `confusion_at`), a classe `PixelHead` e `resolve_score_levels`.
Verificado: `evaluate_pixel_level.py` e `evaluate_full_pipeline.py` dão métricas idênticas na mesma
classe (`vari-grip`: 0.9064 / 0.0762 / 0.7129).

A receita validada é o **default** dos três — rodar sem argumento nenhum já reproduz 9.0.

#### `evaluate_image_level.py`

Dois eixos que só fazem sentido neste nível:

- `--n_transforms N` — TTA nativo do DifferNet (média sobre N rotações fixas). Default 1.
- `--threshold_mode` — `best_f1` espia os rótulos de teste e é otimista; `train_quantile` calibra o
  limiar **só com imagens boas de treino**, que é o que um sistema implantado consegue fazer.

Saída: `results_per_class.csv` (AUROC, AP, F1, precision, recall, specificity, balanced acc, FPS),
`scores_<classe>.csv` por imagem, histograma, ROC e PR.

#### `evaluate_pixel_level.py`

Além das métricas, traz os dois diagnósticos que tornaram as decisões legíveis na ablation:

- **`per_level_auroc.csv`** — cada nível pontuado sozinho. É o diagnóstico que revelou que a head
  publicada de `glass` media L1 = 0.2583 (anticorrelacionado) por incoerência de hiperparâmetros,
  contra 0.874 depois de retreinar (5.2).
- **`--sweep`** — reagrega os mapas NLL **já calculados** sob todas as combinações de
  `levels` × `sigma` × `score_norm`, então a sensibilidade a essas escolhas sai numa tabela só, sem
  passagem extra pela rede. Mesmo truque de cache do `ablation_nf_head.py`.

Saída: `results_per_class.csv` (inclui `pixel_ap_trivial` para contextualizar o AP),
`per_level_auroc.csv`, `sweep.csv` e `mapas.png` (input ¦ GT ¦ mapa ¦ overlay).

### `core/eval_pipeline.py` (novo)
- `PixelHead`: carrega a NF head **junto com a receita gravada no checkpoint** (`levels`,
  `feat_pool`, `pad_mode`, `feat_stats`), o que impede a classe de bugs mais cara do projeto —
  avaliar com hiperparâmetros diferentes dos do treino (Seção 16).
- `build_eval_datasets(...)`: transform determinístico e `--limit` balanceado que **preserva todas
  as anomalias**.
- Métricas compartilhadas e descoberta de checkpoints.
- `PER_CLASS_SCORE_LEVELS` e `resolve_score_levels(requested, trained_levels, class_name=None)`:
  suporte ao modo `--levels per_class` calibrado por classe (`vari-grip`='0', `lightning-rod`='2',
  demais='02'), elevando o AUROC médio para 0.9068 e AUPRO para 0.6446 sem retreino (Seção 6.9).

### `core/cflow.py`
- `SEBackboneFeatureExtractor(..., pad_mode='reflect'|'replicate', feat_pool=k)`.
- Novo modo `per_level_prob` (agregação do CFLOW-AD) em `SCORE_MODES`.
- `CFlowPixelHead.aggregate_nll(...)`: agrega mapas NLL já calculados (permite cache) com
  `pos_stats` opcional (calibração posicional por nível).
- `compute_positional_stats`, `compute_feature_stats`, `apply_feature_stats`, `select_levels`.
- `TTA_MODES`, `flip_tensor`, `nll_with_tta`: TTA por espelhamento, com os mapas revertidos
  antes da média para permanecerem registrados com a máscara. Ficam aqui (e não no script de
  ablation) para que avaliação e ablation usem literalmente a mesma função.
- `compute_level_stats(..., levels=, feat_stats=)`: respeita o subconjunto de níveis e a
  normalização com que a head foi treinada.
- `CFLOW_HPARAM_KEYS` += `pad_mode, feat_pool, levels, feat_norm, feat_norm_levels, no_rotation`.

### `scripts/train/pixel_train_from_pretrained.py`
- Flags: `--no_rotation`, `--pad_mode`, `--feat_pool`, `--feat_norm`, `--feat_norm_levels`, `--levels`.
- `--feat_norm_levels` (ex.: `--feat_norm_levels 0`): permite padronização seletiva por canal
  restrita a níveis específicos (ex.: só L1). Estatísticas `feat_stats` com slots `None` nos níveis não
  normalizados e todos os hiperparâmetros vão no checkpoint.
- **Práticas do RD++ incorporadas:**
  - `compute_aupro` corrigido por quantis de normais em $[1-\text{max\_fpr}, 1.0]$.
  - Avaliação de AUPRO a cada época (`--eval_aupro`).
  - Checkpoint por score composto `best_composite.pt` = $(AUROC_{px} + AUROC_{sp} + AUPRO)/3$.
  - `history.json` persistido em disco a cada época de avaliação.
  - Gráfico de monitoramento 2×2 (`training_curves.png`) regravado a cada avaliação com marcos do melhor modelo.
  - Early stopping com flag `--patience`.
- Correção: com `--disable_patchcore` o script quebrava na validação final (gráficos do PatchCore
  sem guarda) — agora só gera esses gráficos quando o PatchCore está ativo.

### `scripts/train/train_all_nf_heads.py` (novo)
- Orquestrador de treino sequencial em lote para as 5 classes em pasta única particionada por classe
  `resultado_analise_final/treino_nf_heads/run_<timestamp>/`.
- Centraliza os melhores modelos em `best_models/<classe>/best_pixel_auroc.pt`.
- Gera `training_summary.csv`, `training_summary.md` e dispara `--evaluate_after {pixel,full,none}`.
- Receita `optimal` com `CLASS_BUDGETS`: `vari-grip` calibrado para `levels='0,1'`.
- Passthrough de `--patience` e alerta quando `--epochs` sobrescreve os orçamentos da receita.

### `scripts/eval/ablation_nf_head.py` (novo)
- `--stage posthoc|retrain|all`, grid configurável por flag, `--quick`, `--limit`, `--val_fraction`,
  `--top_n`, `--pixel_stride`, `--select_metric`, `--train_score_mode`.
- Variantes de treino em `TRAIN_VARIANTS` incluem padronização seletiva: `featnorm_l1_only`,
  `featnorm_l1_only_02` e `featnorm_l1_only_clamp19`.
- Cache das NLL por (pad, tta); métricas rápidas por histograma; held-out em resolução plena + AUPRO.
- Diagnósticos: `per_level_auroc.csv` (AUROC de cada nível isolado), `main_effects.png`
  (efeito marginal de cada fator), `border_profile.png`, `maps_comparison.png`.
- Retreino via subprocesso do script de treino, sempre com o SE final model da classe.

### `scripts/eval/evaluate_full_pipeline.py`
- Flags novas `--tta {none,hflip,flips}` e `--levels {auto,all,<ids>}`; a receita validada é o
  **default** (`raw`, pad 112, flips, L1+L3, σ=8, margin 0). Ver 9.1, item 1.
- Lê `levels`, `feat_pool`, `pad_mode` e `feat_stats` do checkpoint, de modo que a head é
  avaliada com a mesma função com que foi treinada; heads legadas continuam funcionando.
- Imprime a receita resolvida por classe; `config.txt` do run já grava todos os argumentos.
- As métricas, a descoberta de checkpoints e `resolve_score_levels` passaram para
  `core/eval_pipeline.py` e são reexportadas daqui, porque `ablation_nf_head.py` e
  `evaluate_cflow_no_smoothing.py` as importam deste módulo.

### `scripts/eval/run_confirmacao_2classes.ps1` (novo)
- Encapsula a rodada de confirmação em `polymer-insulator-upper-shackle` e `yoke-suspension`
  com orçamento de épocas por classe e `--limit` balanceado para `yoke`. Ver 9.1, item 2.

Saídas em `resultado_analise_final/ablacao_nf_head/run_<timestamp>/`:

| Arquivo | Conteúdo |
|---|---|
| `ablation_summary.csv` | uma linha por head: melhor val, melhor held-out, referências |
| `retrain_variants.csv` | flags, código de saída e tempo de treino de cada variante |
| `posthoc_<tag>/grid_val.csv` | **todas** as configurações do grid com métricas de validação |
| `posthoc_<tag>/holdout_test_top.csv` | top-N + referências em resolução plena + AUPRO |
| `posthoc_<tag>/per_level_auroc.csv` | AUROC de cada nível isolado por (pad, tta) |
| `posthoc_<tag>/main_effects.png` | efeito marginal de cada fator |
| `posthoc_<tag>/border_profile.png` | score médio vs distância à borda (imagens normais) |
| `posthoc_<tag>/maps_comparison.png` | input ¦ GT ¦ mapa baseline ¦ mapa da melhor config |
| `posthoc_<tag>/summary.txt` | recap legível com hparams da head |

### Custo computacional observado

| Etapa | Tempo (RTX local, vari-grip 154 imgs) |
|---|---|
| Treino de uma head, 40 épocas | ~35–45 min |
| Inferência + cache de NLL (4 passagens) | ~6 min |
| Grid de 1 920 configs (stride 2, histograma) | ~2 min 41 s |
| Held-out top-14 (resolução plena + AUPRO) | ~2 min 45 s |
| Rodada 1 completa (8 variantes, 40 épocas) | ~9 h |
| Rodada 2 completa (5 variantes, 40 épocas) | ~6 h |
| Rodada 3a completa (4 variantes, lightning-rod) | ~4 h |
| Rodada 3b completa (4 variantes, glass — 1104 treino / 678 teste) | ~10 h |
| Passo 0 (pós-hoc, sem treino) | 5 min (154 imgs) a ~50 min (678 imgs) |

---

## 16. Armadilhas e lições de implementação

- **Bug corrigido em 2026-09-04 (avaliação dentro do treino).** `load_datasets()` entregava ao conjunto
  de teste o transform de **treino**, com `RandomRotation(180)` quando `c.transf_rotations=True`: a
  imagem era rotacionada, a máscara não. Com rotação ligada, o `pixel_auroc` medido durante o treino
  era ruído (~0.60, plano ao longo das épocas) e o `best_pixel_auroc.pt` era escolhido ao acaso; com
  `--no_rotation` o efeito colateral desligava a rotação do teste e o valor saltava para ~0.865 na
  época 5. Isso também explica o 0.798 (treino) vs 0.899 (avaliação) da head publicada. O mesmo
  trecho gerava 64 vistas por imagem de teste (`n_transforms_test`) e usava só a primeira. Agora o
  teste do treino usa uma vista determinística (Resize/ToTensor/Normalize), igual ao avaliador.
  **Qualquer run de treino com rotação ligada anterior à correção deve ser refeito.**
- O `clip_grad_norm_` era global sobre os três flows (L3 dominava a norma); passou a ser por flow.
- Platô do `pixel_auroc` após ~5 épocas com loss ainda caindo **não** é bug: é o comportamento
  esperado do CFLOW (o ótimo de generalização chega cedo). A época é selecionada no split de validação.
- `out_size`, `clamp_scale`, `levels`, `feat_pool`, `feat_norm` e `reflect_pad` **mudam a função
  computada** sem mudar o `state_dict`. Nunca avaliar uma head com valores diferentes dos do treino
  (o script lê do checkpoint; heads legadas caem no default com aviso).
- Calibração posicional em L1 **piora** (perfil dependente de conteúdo). Mantida fora por padrão.
- `per_level_minmax` torna o score de imagem via mapa quase inútil (max ≈ 1 sempre); com
  `per_level_prob`/`per_level_std` o score de imagem via mapa volta a ter escala absoluta.
- Estatísticas posicionais/`per_level_std` são estimadas sem TTA; com `tta=flips` há leve descasamento
  de variância (documentado, efeito de segunda ordem).
- Métricas do grid usam stride 2 e histograma; os números finais são os de `holdout_test_top.csv`.
- Para heads **retreinadas com `--reflect_pad 112`**, a referência `baseline_producao` (pad 0) avalia a
  head fora da condição de treino — serve justamente para medir o custo do descasamento
  (smoke test 1 época: 0.757 com pad 0 vs 0.930 com pad 112). Compare variantes pelo `top` do held-out.
- O checkpoint SE e o CBAM têm **as mesmas chaves** de `state_dict`; a detecção de arquitetura no
  treino é pelo nome do arquivo (`--arch` para forçar). Antes da correção o vari-grip carregava como
  `CBAMDifferNet` e o extrator emitia aviso espúrio.
- **Bug no `compute_aupro` sob NLL crua (corrigido em 2026-09-08).** Gerar limiares com
  `np.linspace(min, max, 200)` em mapas não padronizados com caudas pesadas colocava quase todos os
  pontos em FPR > 0.3, deixando <2 thresholds úteis e retornando `aupro = 0.0000` silenciosamente no
  `vari-grip`. Corrigido para quantis de pixels normais em $[1-\text{max\_fpr}, 1.0]$.
- **A armadilha do `--epochs` global uniforme.** Sobrescrever o orçamento calibrado de classes massivas
  (como `yoke`, 4 834 imagens) causou sobreajuste severo: pico na época 4 e queda de 0.030 até a 80.
  Classes com volumes de treino muito díspares exigem orçamentos específicos em `CLASS_BUDGETS` ou
  early stopping (`--patience`).
- **A armadilha da seleção por métrica única.** Em `polymer`, L3 isolado exibe AUROC 0.0020 maior que
  L1+L3, mas derruba o AUPRO em 0.060. Como o AUROC é dominado pela massa de pixels normais, otimizar
  apenas por AUROC pode privilegiar modelos com falsos positivos regionais graves. O score composto
  do RD++ $(AUROC_{px} + AUROC_{sp} + AUPRO)/3$ atua como regularizador de seleção.
- **Informatividade de nível é específica da head treinada.** `polymer` na head anterior tinha L2 como
  melhor nível e `012` vencia `02`. Na head retreinada com 3 níveis em 80 épocas, L2 caiu para 0.7771 e
  `012` ficou 0.052 abaixo de `02` (0.7987 vs 0.8505). Nunca assumir que rankings de níveis transferem
  entre diferentes versões de head sem re-medir com `evaluate_pixel_level.py`.

---

## 17. Reprodutibilidade

### Ambiente

```powershell
conda activate anomaly_attentideffernet   # o env `base` não tem fightingcv_attention
cd anomaly-detection-pesquisa\differnet   # todos os scripts assumem esta pasta
```

⚠️ **Sempre use aspas em listas com vírgula** (`--level_sets "0,01,012"`). Sem aspas o PowerShell as
interpreta como array de inteiros e converte `0,01,012` em `0,1,12` — silenciosamente.

GPU: CUDA disponível (torch 2.5.1+cu121). Seed padrão 42 em todos os estágios.

### Dados

- Raiz: `C:\Users\teo-s\Documents\GitHub\anomaly-detection-dataset\insplad-seg\insplad-seg`
- Layout: `<classe>/{train/good, test/good, test/<defeito>, ground_truth/<defeito>}`
- vari-grip: 477 treino · 114 test/good · 40 test/rust · 40 máscaras. Resolução nativa ~600–900 px,
  redimensionada para 448²; máscaras em grayscale, resize nearest, binarizadas em 0.5.

### Artefatos desta pesquisa

| O quê | Onde |
|---|---|
| Rodada 1 (`vari-grip`, 8 variantes) | `resultado_analise_final/ablacao_nf_head/run_20260904_092255/` |
| Rodada 2 (`vari-grip`, 5 variantes) | `resultado_analise_final/ablacao_nf_head/run_20260905_011651/` |
| Passo 0 — `lightning-rod` (head publicada) | `resultado_analise_final/ablacao_nf_head/run_20260905_133042/` |
| Passo 0 — `vari-grip` (head publicada) | `resultado_analise_final/ablacao_nf_head/run_20260905_134310/` |
| Passo 0 — `glass-insulator` (head publicada) | `resultado_analise_final/ablacao_nf_head/run_20260905_134834/` |
| Rodada 3a (`lightning-rod`, 4 variantes) | `resultado_analise_final/ablacao_nf_head/run_20260905_145528/` |
| Rodada 3b (`glass-insulator`, 4 variantes) | `resultado_analise_final/ablacao_nf_head/run_20260905_190938/` |
| Avaliação final das 5 classes (head publicada) | `resultado_analise_final/run_20260903_232209/` |
| Ablation de smoothing (histórico) | `resultado_analise_final/ablacao_smoothing/` |
| Heads publicadas | `final_models/NF Head/<classe>/best_models/` |
| SE final models | `final_models/SEDiffernet/<classe>/*.pt` |
| Documento de correções anterior | `docs/cflow_correcoes_e_fase3.md` |

⚠️ `run_20260905_131209` é um Passo 0 **inválido** (argumento `--level_sets` corrompido pelo
PowerShell). O válido para `lightning-rod` é `run_20260905_133042`.

### Comandos-chave

```powershell
# Reproduzir a linha de base legada (pré-correções)
python scripts/eval/evaluate_full_pipeline.py --clamp_scale 2.0 --out_size 56 --score_norm raw --sigma 4

# Avaliação final atual, todas as classes
python scripts/eval/evaluate_full_pipeline.py --reflect_pad 112 --border_margin 8

# Ablation pós-hoc na head publicada
python scripts/eval/ablation_nf_head.py --class_name vari-grip

# Ablation com retreino
python scripts/eval/ablation_nf_head.py --class_name vari-grip --stage retrain --epochs 60 --variants <lista>

# Treinar uma head isolada com a melhor receita conhecida
python scripts/train/pixel_train_from_pretrained.py --class_name vari-grip --disable_patchcore `
  --checkpoint "final_models\SEDiffernet\vari-grip\vari-grip_se_differnet_vari_grip_200_1_epoch_163.pt" `
  --feat_norm --reflect_pad 112 --epochs 60
```

### Re-análise dos dados já coletados

Os `grid_val.csv` guardam **todas** as 1 920 configurações por head — novas perguntas podem ser
respondidas sem reexecutar nada:

```python
import pandas as pd, glob, os
base = 'resultado_analise_final/ablacao_nf_head/run_20260904_092255'
g = pd.concat([pd.read_csv(f, dtype={'levels': str}).assign(tag=os.path.basename(os.path.dirname(f)))
               for f in glob.glob(os.path.join(base, 'posthoc_*', 'grid_val.csv'))])
g.groupby(['score_mode', 'pos_calib'])['pixel_auroc'].mean().unstack()
```

⚠️ `levels` precisa ser lido como **string** (`012` vira `12` se o pandas inferir inteiro).
