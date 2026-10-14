# Relatório de achados — vari-grip e yoke-suspension (AUROC-only) e consolidação das 5 classes

**Rodada:** `run_full_20261010_195638` (vari-grip, yoke-suspension).  
**Rodada anterior consolidada:** `run_full_20261007_123558` (glass, lightning, polymer) — ver [relatório anterior](relatorio_achados_run_full_20261007_123558.md).  
**Data de corte:** 2026-10-13, 19:00.  
**Protocolo desta rodada:** D-13 (uma seed de treino 42, split seed 42) e D-14 (seleção por Pixel AUROC; AUPRO, AP e F1 não calculados).  
**Natureza:** análise exploratória de artefatos salvos. Nenhum treino ou inferência foi executado para este relatório.

> **Atualização 2026-10-13:** a [análise científica consolidada](analise_cientifica_ablacao_nf_head_5classes.md) amplia este relatório com efeitos pareados nas 25 heads, IC hierárquico por classe, testes de Friedman e a recomendação de receita geral. Ela recomenda `per_level_std` (indutivo) no lugar da receita única com `per_level_std` e L1+L2+L3 citada na seção 3.5 abaixo, com níveis escolhidos por classe.

## 1. Cobertura e integridade

| Classe | Treinos (código 0) | Grids | Holdouts | Teste | Validação (anômalas) | Holdout (anômalas) |
|---|---:|---:|---:|---:|---:|---:|
| vari-grip | 5/5 | 5 | 5 | 154 | 62 (16) | 92 (24) |
| yoke-suspension | 5/5 | 5 | 5 | 1253 | 501 (18) | 752 (28) |

- As 10 heads têm `holdout_clean=True`, `select_metric=pixel_auroc` e `auroc_only=True`. As colunas de AUPRO, AP e F1 estão vazias (NaN) por desenho do protocolo, não por erro.
- O launcher terminou com código 0. No momento do corte, o único processo Python ativo não pertencia à ablação.
- **Yoke tem só 18 imagens anômalas na validação e 28 no holdout**, entre centenas de imagens normais. As diferenças de seleção entre variantes desta classe são, portanto, especialmente frágeis.

## 2. Resultado por head (seleção na validação, medida no holdout)

`Referência` = score `cflow_original` puro sobre a mesma head (pad 0, sem TTA, L1+L2+L3, sigma 0). `Produção legada` = minmax, todos os níveis, sigma 4. Os ICs de 95% vêm de bootstrap pareado por imagem e comparam o top-1 com a referência da mesma head.

### vari-grip

| Variante | Época | AUROC val | AUROC holdout | Referência | Produção legada | Δ vs referência [IC 95%] | Receita escolhida |
|---|---:|---:|---:|---:|---:|---|---|
| baseline | 10 | 0.8953 | 0.9220 | 0.9051 | 0.8594 | +0.0169 [+0.0104, +0.0235] | pad112 reflect, hflip, cflow_original, **L1**, up=False, σ8, max |
| featnorm | 5 | 0.9111 | 0.9277 | 0.8966 | 0.8276 | +0.0311 [+0.0213, +0.0426] | pad112 reflect, hflip, cflow_original, **L1**, up=True, σ8, topk |
| featnorm_l1_only | 20 | 0.9107 | 0.9268 | 0.9094 | 0.8733 | +0.0174 [+0.0092, +0.0262] | pad112 replicate, hflip, cflow_original, **L1**, up=False, σ8, max |
| clamp19 | 5 | 0.9114 | 0.9274 | 0.8823 | 0.7742 | +0.0451 [+0.0339, +0.0593] | pad112 replicate, none, cflow_original, **L1+L2**, up=True, σ8, topk |
| **featnorm_clamp19** | 5 | **0.9140** | **0.9295** | 0.8912 | 0.7456 | +0.0383 [+0.0231, +0.0551] | pad112 reflect, hflip, cflow_original, **L1**, up=True, σ8, topk |

### yoke-suspension

| Variante | Época | AUROC val | AUROC holdout | Referência | Produção legada | Δ vs referência [IC 95%] | Receita escolhida |
|---|---:|---:|---:|---:|---:|---|---|
| baseline | 2 | 0.9428 | 0.9449 | 0.9115 | 0.8373 | +0.0334 [+0.0236, +0.0424] | pad112 reflect, flips, per_level_std, **L1+L3**, up=True, σ12, topk |
| featnorm | 2 | 0.9179 | 0.9115 | 0.8995 | 0.7770 | +0.0120 [+0.0013, +0.0216] | pad112 reflect, none, per_level_std, **L1+L3**, up=False, σ12, topk |
| **featnorm_l1_only** | 2 | **0.9436** | **0.9466** | 0.9141 | 0.8366 | +0.0324 [+0.0230, +0.0414] | pad112 reflect, flips, per_level_std, **L1+L3**, up=True, σ12, topk |
| clamp19 | 2 | 0.9220 | 0.9060 | 0.8657 | 0.6687 | +0.0403 [+0.0296, +0.0512] | pad112 replicate, hflip, cflow_original, **L1+L2**, up=True, σ12, topk |
| featnorm_clamp19 | 2 | 0.9190 | 0.8994 | 0.8640 | 0.6780 | +0.0354 [+0.0235, +0.0469] | pad112 replicate, hflip, cflow_original, **L1+L2**, up=True, σ12, topk |

Em yoke, `featnorm_l1_only` vence `baseline` por **0.0008 na validação** e 0.0016 no holdout. Na prática, é um empate: com 18 anomalias na validação, essa diferença não distingue as duas variantes.

## 3. O que se pode inferir

### 3.1 O pós-processamento pesa mais que a variante de treino

Ganho no holdout em vari/yoke, do sistema legado até a melhor head:

| Classe | Produção legada → referência `cflow_original` | Referência → melhor receita na baseline | Baseline com receita → melhor variante | Total vs produção legada |
|---|---:|---:|---:|---:|
| vari-grip | +0.0457 | +0.0169 | +0.0075 | **+0.0701** |
| yoke-suspension | +0.0742 | +0.0334 | +0.0016 | **+0.1092** |

A decomposição depende da ordem dos passos, mas a conclusão é robusta: **trocar o score e a receita de inferência rende de 6 a 10 pontos de AUROC; trocar a variante de treino rende menos de 1 ponto** nessas duas classes.

### 3.2 Normalização e clamp não ajudam por si só, e às vezes atrapalham

Comparando variantes com a mesma referência fixa, contra a baseline da classe:

| Variante | vari-grip ΔAUROC | yoke ΔAUROC |
|---|---:|---:|
| featnorm | −0.0085 | −0.0121 |
| featnorm_l1_only | +0.0043 | +0.0026 |
| clamp19 | −0.0228 | −0.0458 |
| featnorm_clamp19 | −0.0139 | −0.0475 |

O clamp 1.9 piora a head sob receita fixa nas duas classes. Em vari-grip, porém, a busca pós-hoc recupera e supera essa perda (0.9295 contra 0.9220 da baseline). Repete-se o padrão de glass: **treino e receita interagem**, e o vencedor depende da receita. Normalizar só L1 é a única variante sem perda sob receita fixa nas duas classes, mas com ganho pequeno.

### 3.3 O nível de feature é a escolha mais importante — e é específico da classe

- **vari-grip: L1 sozinho** vence em 4 de 5 heads (regret médio na validação de 0.0005). Isso confirma o achado histórico de que L3 é o pior nível nessa classe.
- **yoke: L1+L3** (sem L2) para baseline/normalização e **L1+L2** com clamp. L1 sozinho custa 0.055 de AUROC na validação de yoke.
- Somando as 5 classes: glass prefere L1+L3, lightning e polymer usam combinações diferentes, vari usa L1 e yoke usa L1+L3. **Nenhum conjunto de níveis serve para todas.**

### 3.4 Fatores consistentes nas 10 heads (OFAT sobre a referência)

| Fator | ΔAUROC médio | Mín / máx | ICs > 0 / < 0 |
|---|---:|---|---|
| sigma 8 | +0.0109 | +0.0056 / +0.0182 | 10 / 0 |
| sigma 12 | +0.0085 | +0.0003 / +0.0181 | 5 / 0 |
| sigma 4 | +0.0074 | +0.0055 / +0.0106 | 10 / 0 |
| pad 112 | +0.0070 | +0.0045 / +0.0137 | 10 / 0 |
| TTA flips | +0.0062 | +0.0009 / +0.0111 | 9 / 0 |
| TTA hflip | +0.0038 | +0.0013 / +0.0066 | 8 / 0 |
| align_corners=False | −0.0011 | −0.0019 / −0.0003 | 0 / 7 |
| score raw | −0.0189 | −0.0412 / −0.0019 | 0 / 8 |
| score per_level_prob | −0.0808 | −0.1945 / +0.0131 | 0 / 9 |
| score per_level_minmax | −0.1045 | −0.1940 / −0.0364 | 0 / 10 |

Na validação, **pad 112 e nenhuma calibração posicional** aparecem em todos os vencedores. **Sigma 8** tem o menor regret (0.0006 médio), e `align_corners` é irrelevante (regret < 0.001). Ao contrário das três primeiras classes, aqui `raw` perde para `cflow_original`; as escolhas de score também dependem da classe.

### 3.5 Receita única versus receita por classe (validação, 25 heads)

A melhor receita fixa para as 25 heads é `pad112 reflect, flips, per_level_std, L1+L2+L3, up=True, σ8`. Ela perde em média **0.026 de AUROC** (máximo 0.049) em relação à melhor receita de cada head. Já uma receita fixa **por classe** perde pouco:

| Classe | Regret médio / máximo | Melhor receita fixa da classe |
|---|---|---|
| glass | 0.0020 / 0.0052 | pad112 reflect, flips, per_level_std, L1+L3, up=False, σ4 |
| lightning | 0.0069 / 0.0171 | pad112 reflect, flips, raw, L1+L3, up=True, σ12 |
| polymer | 0.0172 / 0.0448 | pad112 replicate, flips, raw, L1+L2, up=True, σ8 |
| vari-grip | 0.0006 / 0.0024 | pad112 reflect, hflip, cflow_original, L1, up=True, σ8 |
| yoke | 0.0125 / 0.0319 | pad112 reflect, hflip, cflow_original, L1+L2+L3, up=True, σ12 |

**Conclusão: o sistema deve usar receita por classe.** A receita única custa cerca de 2,6 pontos de AUROC.

### 3.6 As curvas de treino atingem o pico cedo

- **yoke: as 5 variantes atingem o melhor resultado na primeira avaliação (época 2)** e depois caem (baseline 0.9054 → 0.8926; clamp19 0.8831 → 0.8587). O ótimo real pode estar antes da época 2, que não foi avaliada.
- **vari-grip:** featnorm, clamp19 e featnorm_clamp19 atingem o pico na primeira avaliação (época 5); baseline na época 10; somente `featnorm_l1_only` na época 20.
- Isso indica orçamento de épocas excessivo para essas classes e resolução de avaliação insuficiente no início do treino. Proposta: avaliar a cada época nas primeiras épocas e usar early stopping pela validação, em vez de manter 40/12 épocas fixas.

### 3.7 O mapa não substitui o SEDifferNet na decisão por imagem

A AUROC por imagem do mapa top-1 (0.7077–0.7855 em vari; 0.7871–0.8681 em yoke) fica abaixo do SEDifferNet no mesmo holdout (0.9044 e 0.9577). Isso repete o achado das três primeiras classes.

### 3.8 Estabilidade do top-10

No holdout, as 10 melhores receitas de validação de cada head ficam em faixas estreitas: amplitude máxima por head de 0.0033 em vari e 0.011 em yoke. Assim, a escolha exata do top-1 importa pouco dentro de cada head; o que importa é a família de receita.

## 4. Melhor receita por classe — consolidado das 5 classes

| Classe | Variante | Receita | Critério de seleção | AUROC holdout | Produção legada (head baseline) | Ganho |
|---|---|---|---|---:|---:|---:|
| glass | featnorm | pad112 reflect, flips, per_level_std, L1+L3, up=False, σ2 | AUROC+AUPRO | 0.9467 | 0.8864 | +0.0603 |
| lightning | clamp19 | pad112 replicate, flips, cflow_original, L1+L2+L3, up=True, σ8 | AUROC+AUPRO | 0.8965 | 0.7658 | +0.1307 |
| polymer | featnorm_clamp19 | pad112 replicate, hflip, per_level_std, L1+L2+L3, up=False, σ4 | AUROC+AUPRO | 0.8848 ⚠ | 0.8534 | +0.0314 |
| vari-grip | featnorm_clamp19 | pad112 reflect, hflip, cflow_original, L1, up=True, σ8 | AUROC | 0.9295 | 0.8594 | +0.0701 |
| yoke | featnorm_l1_only ≈ baseline | pad112 reflect, flips, per_level_std, L1+L3, up=True, σ12 | AUROC | 0.9466 | 0.8373 | +0.1092 |
| **Média** | | | | **0.9208** | **0.8405** | **+0.0804** |

Médias do holdout em 5 classes, ancoradas no sistema legado da head baseline:

- Produção legada: **0.8405**.
- Referência `cflow_original` na head baseline: 0.8666 (+0.0261).
- Melhor receita pós-hoc na head baseline: 0.9030 (+0.0364).
- Melhor variante de treino com sua receita: **0.9208** (+0.0178).

**Ganho total: +0.080 de Pixel AUROC médio (de 0.841 para 0.921)**, sendo cerca de 78% atribuível ao score e à receita de inferência e cerca de 22% à variante de treino. O rateio depende da ordem dos passos.

⚠ Polymer: a receita selecionada por AUROC+AUPRO registrou AUPRO zero (investigação pendente). A receita que seria escolhida só por AUROC (`featnorm_clamp19`, pad112 replicate, flips, raw, L1+L2, σ4) não foi medida no holdout. Glass também teria outra receita sob AUROC-only.

## 5. Limites da interpretação

- **Protocolos misturados:** glass/lightning/polymer foram selecionados por AUROC+AUPRO; vari/yoke, só por AUROC. A tabela consolidada é útil para decisão prática, mas não forma uma comparação homogênea.
- **Uma seed (D-13):** não há estimativa da variação entre inicializações. Os vencedores de variante, sobretudo em yoke, podem trocar com outra seed.
- **Sem AUPRO (D-14):** nada aqui garante qualidade de localização por região em vari/yoke. A discrepância de AUPRO em polymer mostra que receitas boas em AUROC podem falhar em AUPRO.
- **Diferenças entre variantes não têm IC.** Os ICs disponíveis comparam o top-1 com a referência da mesma head.
- **Holdout já observado e muitas receitas testadas:** os números do top-1 são estimativas honestas para cada head selecionada, mas decisões tomadas depois de ver este holdout precisam de confirmação independente.
- **Yoke tem poucas anomalias**; vari tem poucas imagens no total (154).

## 6. Custo observado

| Classe | Treino (5 heads) | Setup → grid por head | Grid → holdout por head | Total aproximado |
|---|---:|---|---|---:|
| vari-grip | 3.5 h | 71–72 min | 4.7–4.9 min | ≈ 9.8 h |
| yoke-suspension | 11.0 h | 405–593 min | 45–79 min | ≈ 58.4 h |

Durante o grid de yoke foram observados: Python usando cerca de **1 de 20 threads**, **0.9 GB de RAM livre**, 12.5 GB comprometidos e **6.9 GB de swap em uso**. O gargalo é CPU de uma thread mais paginação de memória, não GPU. O modo AUROC-only removeu o custo de AUPRO, mas não esse gargalo. Os tempos são intervalos entre arquivos, não perfil de funções.

## 7. Próximos passos sugeridos (propostas, não decisões)

1. **Fixar receita por classe** com base na validação. Antes de usar, medir no holdout as receitas AUROC-only de glass e polymer que ainda não foram avaliadas.
2. **Treino curto com avaliação densa:** avaliar a cada época no início e aplicar early stopping. Yoke e a maior parte de vari atingem o pico na primeira avaliação.
3. **Desempate pré-registrado:** antes da próxima rodada, definir que diferenças de validação menores que uma tolerância (ex.: 0.002) favorecem a variante mais simples (baseline). Com essa regra, yoke ficaria com baseline.
4. **Reduzir o grid** usando a validação das 25 heads: pad 112, sem calibração posicional, sem `per_level_minmax`/`per_level_prob`, sigma {4, 8, 12}, align fixo. Manter os níveis e os scores `cflow_original`/`raw`/`std` abertos, pois dependem da classe.
5. **Grid na GPU e máscaras deduplicadas**, validados contra uma head já avaliada, para eliminar o gargalo de CPU e swap.
6. Retomar AUPRO para as receitas finalistas quando a localização por região voltar a ser avaliada.

## 8. Resumo

1. A rodada vari/yoke terminou íntegra: 10 heads treinadas, avaliadas e com holdout limpo, sob uma seed e seleção por AUROC.
2. **Melhor receita:** vari-grip — `featnorm_clamp19` + pad112, hflip, `cflow_original`, **somente L1**, σ8 (AUROC holdout **0.9295**). Yoke — `featnorm_l1_only` (empate prático com baseline) + pad112, flips, `per_level_std`, **L1+L3**, σ12 (**0.9466**).
3. **Quanto ganhamos:** +0.070 (vari) e +0.109 (yoke) sobre a produção legada. Nas 5 classes, a média sobe de **0.841 para 0.921 (+0.080)**, com cerca de 78% do ganho vindo da receita de inferência e 22% da variante de treino.
4. **O que inferir:** a receita de inferência importa muito mais que a variante de treino; os níveis de feature são a escolha decisiva e dependem da classe; receita por classe supera receita única em cerca de 2,6 pontos; sigma 8, pad 112 e TTA são ganhos consistentes; minmax e prob devem sair do grid.
5. As heads atingem o pico muito cedo, o que pede treino mais curto com avaliação densa.
6. Sem AUPRO e com uma seed, esses resultados valem para ranking pixel-level; não garantem localização por região nem robustez entre inicializações.
