# Ablação da cabeça de *normalizing flow* para localização de anomalias em inspeção aérea de linhas de transmissão: análise científica consolidada das cinco classes do INSPLAD-seg

**Versão:** 1.0 — 2026-10-13  
**Escopo dos dados:** rodadas `run_full_20261007_123558` (glass-insulator, lightning-rod-suspension, polymer-insulator-upper-shackle) e `run_full_20261010_195638` (vari-grip, yoke-suspension).  
**Unidade experimental:** 25 heads CFLOW locais (5 variantes de treino × 5 classes), uma seed de treino (42), split seed 42.  
**Tabelas-fonte reprodutíveis:** [`scripts/eval/consolidate_nf_head_ablation.py`](../scripts/eval/consolidate_nf_head_ablation.py) → `resultado_analise_final/ablacao_nf_head/analise_consolidada_20261013/`.  
**Estado de evidência:** exploratório-forte. As escolhas foram feitas na validação; o holdout foi usado para estimar o sistema escolhido, mas já foi inspecionado durante o projeto. A Seção 9 lista o que falta para uma submissão confirmatória.

---

## Resumo

**Contexto.** A localização de defeitos em imagens aéreas de linhas de transmissão combina fundo natural complexo, defeitos raros e pequenos (0,14–4,6 % da área) e poucas imagens normais por componente. *Normalizing flows* condicionais (CFLOW) estimam a densidade de features espaciais e usam a log-verossimilhança negativa (NLL) como score por pixel. A literatura concentra-se em arquitetura e treino; o peso das decisões de inferência é raramente quantificado com controle de seleção.

**Objetivo.** Quantificar, com o extrator visual congelado, a contribuição relativa das variantes de treino da NF head e das decisões de inferência (agregação de score, seleção de níveis, padding, TTA, suavização, calibração posicional, interpolação), e determinar se existe uma receita universal ou se a receita deve ser específica da classe.

**Métodos.** Para cada uma das 5 classes do INSPLAD-seg foram treinadas 5 heads (baseline, `featnorm`, `featnorm_l1_only`, `clamp19`, `featnorm_clamp19`) sobre o SEDifferNet congelado da classe. O conjunto de teste rotulado foi dividido em validação (40 %) e holdout (60 %), estratificados e auditados por manifesto. A época foi escolhida na validação; 5.310 receitas de inferência por head (132.750 no total) foram avaliadas na validação; a receita escolhida e um conjunto pré-definido de referências, OFAT e escadas foram medidos uma vez no holdout, com IC bootstrap pareado por imagem. Efeitos de fatores foram estimados por pares que diferem em um único fator, agregados por classe, com IC bootstrap hierárquico (classes → heads).

**Resultados.** O sistema escolhido em validação elevou a Pixel AUROC média no holdout de **0,8405** (receita legada de produção, head baseline) para **0,9208** (+0,080), com ganho positivo nas 5 classes (+0,031 a +0,131). Cerca de **78 %** do ganho vem do score e da receita de inferência e **22 %** da variante de treino. Os maiores efeitos isolados são: evitar a normalização *min-max* por imagem (+0,085 AUROC na validação; 25/25 heads; IC95 [+0,058; +0,116]), escolher corretamente os níveis de features (perda de 0,04–0,10 ao fixar um único nível) e suavização gaussiana (σ=8: +0,016; 25/25 heads). Os scores `raw`, `per_level_std` e o agregador original do CFLOW-AD são **equivalentes em AUROC** (|Δ| ≤ 0,002, IC contém zero), mas não em AUPRO. **As variantes de treino não diferem de forma detectável quando se faz a média entre classes** (Friedman p = 0,72), embora dentro de cada classe a escolha mude o resultado em até 0,047. Uma receita única para as 5 classes custa 0,026 de AUROC em relação à melhor receita por head; liberar apenas o conjunto de níveis por classe reduz essa perda para 0,014–0,015.

**Conclusões.** Com extrator congelado, o desempenho de localização por pixel é dominado pela inferência, e a decisão mais sensível à classe é o conjunto de níveis de features. Recomendamos um *template* fixo de inferência com níveis por classe. Os resultados ainda dependem de uma seed, de um único split e, para duas classes, apenas de AUROC; a Seção 9 define o protocolo confirmatório necessário para publicação.

**Palavras-chave:** detecção de anomalias; localização por pixel; *normalizing flows*; CFLOW-AD; ablação; seleção de modelos; inspeção por VANT; INSPLAD.

---

## 1. Introdução e questões de pesquisa

Modelos de densidade por posição, como o CFLOW-AD [2], produzem mapas de NLL em várias escalas que precisam ser agregados, interpolados e suavizados para virar um mapa de anomalia. Essas decisões costumam ser herdadas do código de referência e não ablacionadas. Em dados de inspeção aérea — fundo natural, defeitos minúsculos, poucas amostras — o risco de que convenções herdadas sejam subótimas, ou de que a escolha "ótima" seja apenas sobreajuste ao conjunto de teste, é alto.

Este estudo responde, com o extrator congelado, às seguintes questões:

- **QP1 — Inferência vs. treino.** Qual a contribuição relativa das decisões de inferência e das variantes de treino da NF head?
- **QP2 — Agregação.** O modo de agregação entre níveis importa, e o agregador original do CFLOW-AD é superior às alternativas locais?
- **QP3 — Níveis.** Existe um conjunto de níveis de features (L1, L2, L3) universalmente ótimo?
- **QP4 — Universalidade.** Existe uma receita única de inferência para todas as classes, ou a receita deve ser específica da classe? Quanto custa cada grau de liberdade?
- **QP5 — Treino.** Normalização de features e clamp do acoplamento afim melhoram a head de forma generalizável?

Contribuições: (i) uma ablação controlada de 132.750 receitas sobre 25 heads com seleção estritamente na validação e holdout auditado; (ii) estimativa de efeitos por pares de um único fator com inferência no nível da classe; (iii) decomposição hierárquica do custo de padronizar a receita (receita única → níveis por classe → receita por classe → receita por head); (iv) evidência de que uma métrica de região amplamente usada (AUPRO com limiares lineares) pode colapsar em mapas de cauda pesada; (v) um *template* de inferência indutivo (sem dependência do lote avaliado) recomendado para implantação.

---

## 2. Material e métodos

### 2.1 Dados

INSPLAD-seg [8], cinco componentes de linha de transmissão, imagens 448×448. Treino apenas com imagens normais.

| Classe | Treino normal | Teste | Validação (anômalas) | Holdout (anômalas) | Defeito típico |
|---|---:|---:|---:|---:|---|
| glass-insulator | 1.104 | 678 | 271 (35) | 407 (52) | peça ausente, ~0,14 % da área |
| lightning-rod-suspension | 462 | 163 | 65 (18) | 98 (28) | ferrugem fina, ~1,9 % |
| polymer-insulator-upper-shackle | 935 | 317 | 127 (33) | 190 (49) | desgaste/deformação, ~3,1 % |
| vari-grip | 477 | 154 | 62 (16) | 92 (24) | corrosão extensa, ~4,6 % |
| yoke-suspension | 4.834 | 1.253 | 501 (18) | 752 (28) | dano mecânico, ~2,0 % |

As partições vêm de `split_manifest.json`; todos os manifestos de treino e pós-hoc de cada classe têm a mesma assinatura de índices e interseção vazia entre validação e holdout. As frações de área são as estimadas em estudos anteriores do projeto e servem apenas como descrição.

### 2.2 Modelo

- **Extrator:** SEDifferNet da classe (AlexNet + blocos Squeeze-and-Excitation), pré-treinado e **congelado**.
- **Features:** L1 (64 canais), L2 (192), L3 (256), reamostradas para grade 96×96.
- **Head:** um flow condicional por nível, 8 blocos de acoplamento afim, *hidden* 256, codificação posicional senoidal de 64 dimensões, clamp 0,5 (baseline). Loss: NLL média por célula.
- **Variantes de treino** (todas treinam os três níveis):

| Variante | Alteração |
|---|---|
| `baseline` | — |
| `featnorm` | padronização por canal (μ, σ do treino normal) em L1, L2 e L3 |
| `featnorm_l1_only` | padronização apenas em L1 |
| `clamp19` | clamp do acoplamento 1,9 (valor do CFLOW-AD) |
| `featnorm_clamp19` | `featnorm` + `clamp19` |

Esta é uma head local, **não uma reprodução end-to-end do CFLOW-AD**: encoder, dimensão do PE, blocos de acoplamento, loss e amostragem diferem do código oficial.

### 2.3 Espaço de receitas de inferência

| Fator | Valores |
|---|---|
| Padding de entrada | 0; 112 px `reflect`; 112 px `replicate` |
| TTA | nenhum; flip horizontal; 4 flips |
| Agregação (`score_mode`) | `cflow_original` (agregador oficial: log-prob por canal, máximo por nível na coorte, exponenciação, soma, inversão); `raw` (soma de NLL); `per_level_std` (z-score escalar por nível com estatísticas do treino normal); `per_level_minmax` (min-max por imagem e nível); `per_level_prob` (aproximação local por logsigmoid) |
| Níveis | 0, 1, 2, 01, 02, 12, 012 (0=L1, 1=L2, 2=L3) |
| Calibração posicional | nenhuma; L2+L3 |
| `align_corners` | True; False |
| σ gaussiano | 0, 2, 4, 8, 12 |
| Score por imagem | max; top-k (não altera métricas de pixel) |

Combinações inertes ou inválidas foram removidas, restando 5.310 receitas de pixel distintas por head.

### 2.4 Protocolo de seleção e avaliação

1. **Treino:** somente imagens normais; épocas 40/40/30/40/12 (glass/lightning/polymer/vari/yoke), avaliação a cada 5/5/3/5/2 épocas.
2. **Seleção de época:** maior Pixel AUROC na validação com score `raw` e σ=4.
3. **Seleção de receita:** maior critério na validação (stride 2). glass/lightning/polymer: (Pixel AUROC + AUPRO)/2. vari/yoke: apenas Pixel AUROC, por decisão de custo (D-14).
4. **Seleção de variante por classe:** maior critério de validação entre as 5 heads.
5. **Holdout (stride 1, cálculo exato):** receita escolhida, top-10, três referências fixas, OFAT (um fator alterado sobre a referência) e escada (referência → vencedora). IC 95 % por bootstrap estratificado de imagens (1.000 reamostragens), pareado contra a referência da mesma head.

Referências fixas: **`cflow_original`** (pad 0, sem TTA, níveis 012, sem calibração, `align_corners=True`, σ=0, score max) e **produção legada** (min-max por imagem, níveis 012, `align_corners=False`, σ=4).

### 2.5 Métricas

Pixel AUROC; AUPRO no protocolo global do CFLOW-AD (limiares comuns, falsos positivos sobre pixels normais de todas as imagens, média por região, FPR ≤ 0,3); Pixel AP e melhor F1 por histograma; AUROC por imagem. Em vari/yoke apenas AUROC foi calculado.

### 2.6 Análise estatística

- **Efeitos pareados (validação).** Para cada contraste, pares de receitas idênticas exceto no fator. Δ médio por head, depois por classe. **A classe é a unidade de generalização** (n=5). IC 95 % por bootstrap hierárquico (5.000 reamostragens: classes com reposição, depois heads dentro da classe). Wilcoxon sobre as 5 médias de classe — com n=5, o menor p bilateral possível é 0,0625; por isso o teste é reportado como descritivo, e a força da evidência é lida pela concordância de sinal (classes e heads) e pelo IC.
- **Regret.** Para uma restrição (por exemplo, "níveis = 02"), regret = AUROC da melhor receita da head menos AUROC da melhor receita que respeita a restrição. Mede quanto se perde ao impor a escolha a priori.
- **Variantes de treino.** Teste de Friedman sobre a matriz classe × variante (holdout).
- **Multiplicidade.** Não há correção formal; as conclusões são sustentadas por efeitos grandes, sinais consistentes e replicação em validação e holdout, não por p-valores isolados.

---

## 3. Resultados

### 3.1 Ganho do sistema (QP1)

Médias no holdout, uma head por classe, com receitas sobre a **head baseline**, exceto na última linha:

| Etapa | glass | lightning | polymer | vari | yoke | **Média** | Δ acumulado |
|---|---:|---:|---:|---:|---:|---:|---:|
| Produção legada (min-max, σ4) | 0,8864 | 0,7658 | 0,8534 | 0,8594 | 0,8373 | **0,8405** | — |
| Referência `cflow_original` | 0,8761 | 0,7940 | 0,8461 | 0,9051 | 0,9115 | **0,8666** | +0,0261 |
| Melhor receita (validação) na baseline | 0,9156 | 0,8674 | 0,8653 | 0,9220 | 0,9449 | **0,9030** | +0,0625 |
| Melhor variante + sua receita (validação) | 0,9467 | 0,8965 | 0,8848 | 0,9295 | 0,9466 | **0,9208** | **+0,0804** |
| Oráculo: melhor variante no próprio holdout | 0,9510 | 0,8965 | 0,8848 | 0,9295 | 0,9466 | 0,9217 | — |

- A seleção por validação ficou a 0,0009 do oráculo: o protocolo de seleção não desperdiçou desempenho.
- **≈78 % do ganho** (0,0625 de 0,0804) é de inferência; **≈22 %**, de variante de treino. O rateio depende da ordem das etapas.
- Sobre as 25 heads, a receita escolhida supera a referência `cflow_original` da própria head em 24/25, com IC pareado inteiramente positivo em 24/25. A exceção é lightning/`featnorm_clamp19` (Δ −0,0022, IC [−0,020; +0,016]).
- A receita legada é especialmente ruim para heads não baseline: em média, 0,7935 contra 0,8679 da referência `cflow_original` nas 25 heads (a referência vence em 21/25).

### 3.2 Tabela principal de ablação (QP1–QP3)

**Tabela A — efeitos pareados na validação.** Cada linha compara receitas idênticas exceto no fator indicado. Δ em Pixel AUROC absoluto. "Classes +" e "Heads +" contam médias positivas. AUPRO: média nos 15 pares com AUPRO (3 classes).

| Fator | Contraste | Pares | Δ médio (classe) | IC 95 % (bootstrap hierárquico) | Classes + | Heads + | % pares + | Δ AUPRO |
|---|---|---:|---:|---|---:|---:|---:|---:|
| **Agregação** | `raw` vs `per_level_minmax` | 29.250 | **+0,0854** | [+0,0580; +0,1163] | 5/5 | 25/25 | 91 % | +0,101 |
| | `per_level_minmax` vs `cflow_original` | 15.750 | **−0,0750** | [−0,1084; −0,0462] | 0/5 | 0/25 | 9 % | −0,118 |
| | `per_level_prob` vs `cflow_original` | 15.750 | **−0,1131** | [−0,1465; −0,0796] | 0/5 | 2/25 | 13 % | −0,182 |
| | `raw` vs `cflow_original` | 15.750 | +0,0009 | [−0,0085; +0,0083] | 3/5 | 15/25 | 44 % | −0,036 |
| | `per_level_std` vs `cflow_original` | 15.750 | +0,0017 | [−0,0046; +0,0081] | 3/5 | 13/25 | 47 % | −0,012 |
| **Níveis** (vs 012) | L1+L3 (02) | 20.250 | +0,0003 | [−0,0142; +0,0177] | 2/5 | 10/25 | 45 % | +0,051 |
| | L1+L2 (01) | 20.250 | −0,0151 | [−0,0318; +0,0002] | 1/5 | 6/25 | 31 % | −0,022 |
| | L2+L3 (12) | 20.250 | **−0,0505** | [−0,0720; −0,0302] | 0/5 | 0/25 | 13 % | −0,176 |
| | só L1 (0) | 11.250 | **−0,0399** | [−0,0622; −0,0135] | 1/5 | 4/25 | 23 % | +0,014 |
| | só L2 (1) | 20.250 | **−0,0991** | [−0,1662; −0,0493] | 0/5 | 0/25 | 8 % | −0,272 |
| | só L3 (2) | 20.250 | **−0,0611** | [−0,0940; −0,0303] | 0/5 | 1/25 | 19 % | −0,115 |
| **Suavização** (vs σ0) | σ = 2 | 26.550 | +0,0060 | [+0,0046; +0,0074] | 5/5 | 25/25 | 92 % | +0,010 |
| | σ = 4 | 26.550 | +0,0115 | [+0,0088; +0,0141] | 5/5 | 25/25 | 87 % | +0,020 |
| | **σ = 8** | 26.550 | **+0,0164** | [+0,0113; +0,0213] | 5/5 | 25/25 | 73 % | +0,027 |
| | σ = 12 | 26.550 | +0,0155 | [+0,0078; +0,0225] | 5/5 | 22/25 | 64 % | +0,025 |
| **TTA** | flips vs nenhum | 44.250 | +0,0048 | [+0,0015; +0,0083] | 5/5 | 21/25 | 74 % | +0,003 |
| | hflip vs nenhum | 44.250 | +0,0024 | [+0,0006; +0,0045] | 5/5 | 20/25 | 72 % | +0,002 |
| | flips vs hflip | 44.250 | +0,0023 | [−0,0011; +0,0053] | 4/5 | 18/25 | 65 % | +0,001 |
| **Padding** | 112 `reflect` vs 0 | 44.250 | +0,0024 | [+0,0005; +0,0041] | 4/5 | 21/25 | 66 % | +0,002 |
| | 112 `replicate` vs 0 | 44.250 | +0,0003 | [−0,0021; +0,0028] | 3/5 | 14/25 | 58 % | +0,001 |
| | 112 `replicate` vs 112 `reflect` | 44.250 | −0,0020 | [−0,0044; +0,0005] | 1/5 | 7/25 | 48 % | −0,002 |
| **Calibração posicional** | L2+L3 vs nenhuma | 54.000 | **−0,0248** | [−0,0407; −0,0093] | 0/5 | 6/25 | 21 % | −0,018 |
| **Interpolação** | `align_corners` False vs True | 66.375 | +0,0002 | [−0,0001; +0,0005] | 4/5 | 18/25 | 65 % | +0,000 |

**Tabela B — confirmação no holdout (OFAT sobre a referência `cflow_original`, 25 heads).** "IC +/−" conta heads cujo IC pareado está inteiramente acima/abaixo de zero.

| Fator alterado | Δ AUROC médio | Mín / máx | IC + / IC − | Classes + | Δ AUPRO (15 heads) |
|---|---:|---|---|---:|---:|
| σ = 8 | **+0,0116** | −0,004 / +0,025 | 22 / 0 | 5/5 | +0,023 |
| σ = 12 | +0,0103 | −0,015 / +0,029 | 17 / 2 | 5/5 | +0,020 |
| `per_level_std` | +0,0075 | −0,025 / +0,041 | 12 / 6 | 4/5 | **−0,070** |
| σ = 4 | +0,0074 | +0,002 / +0,012 | 24 / 0 | 5/5 | +0,014 |
| TTA flips | +0,0068 | +0,001 / +0,015 | 21 / 0 | 5/5 | +0,001 |
| pad 112 `reflect` | +0,0061 | −0,005 / +0,014 | 21 / 2 | 5/5 | +0,011 |
| σ = 2 | +0,0030 | +0,002 / +0,004 | 25 / 0 | 5/5 | +0,006 |
| TTA hflip | +0,0029 | +0,001 / +0,007 | 15 / 0 | 5/5 | +0,003 |
| score max → top-k | 0,0000 | 0 / 0 | 0 / 0 | — | 0,000 |
| `align_corners=False` | −0,0010 | −0,003 / +0,001 | 2 / 15 | 1/5 | −0,003 |
| `raw` | −0,0019 | −0,041 / +0,054 | 7 / 12 | 3/5 | **−0,085** |
| níveis 02 | −0,0053 | −0,051 / +0,089 | 5 / 15 | 2/5 | +0,055 |
| níveis 01 | −0,0209 | −0,076 / +0,011 | 2 / 17 | 1/5 | −0,059 |
| níveis 2 | −0,0288 | −0,107 / +0,068 | 5 / 13 | 1/5 | −0,036 |
| níveis 12 | −0,0310 | −0,076 / +0,030 | 4 / 17 | 2/5 | −0,126 |
| níveis 0 | −0,0460 | −0,128 / +0,007 | 0 / 18 | 0/5 | −0,017 |
| `per_level_minmax` | −0,0741 | −0,245 / +0,019 | 0 / 15 | 0/5 | −0,089 |
| níveis 1 | −0,0768 | −0,246 / +0,015 | 0 / 20 | 0/5 | −0,227 |
| `per_level_prob` | −0,1081 | −0,336 / +0,055 | 1 / 20 | 0/5 | −0,149 |

Validação e holdout concordam em sinal e ordem de grandeza para todos os fatores com efeito claro (suavização, TTA, padding `reflect`, min-max, prob, calibração, níveis isolados). Os únicos contrastes cujo sinal difere entre classes — `raw`/`std` vs `cflow_original` e L1+L3 vs L1+L2+L3 — são exatamente os que têm IC contendo zero na Tabela A.

### 3.3 Níveis de features (QP3)

**Tabela C — regret médio na validação (5 heads por classe) ao restringir o conjunto de níveis.** Em negrito, o melhor conjunto por classe.

| Níveis | glass | lightning | polymer | vari | yoke | Média | Máximo |
|---|---:|---:|---:|---:|---:|---:|---:|
| L1 (0) | 0,0704 | 0,0578 | 0,0729 | **0,0005** | 0,0554 | 0,0514 | 0,0729 |
| L2 (1) | 0,1770 | 0,0811 | 0,0143 | 0,0572 | 0,0587 | 0,0777 | 0,1770 |
| L3 (2) | 0,0150 | 0,0147 | 0,0454 | 0,0958 | 0,0555 | 0,0453 | 0,0958 |
| L1+L2 (01) | 0,0794 | 0,0338 | **0,0000** | 0,0035 | 0,0282 | 0,0290 | 0,0794 |
| L1+L3 (02) | **0,0000** | **0,0025** | 0,0439 | 0,0079 | 0,0096 | 0,0128 | 0,0439 |
| L2+L3 (12) | 0,0390 | 0,0139 | 0,0121 | 0,0641 | 0,0247 | 0,0308 | 0,0641 |
| L1+L2+L3 (012) | 0,0212 | 0,0045 | 0,0134 | 0,0094 | **0,0050** | **0,0107** | **0,0212** |

Achados:

1. **Não há conjunto universal.** Cada classe tem um conjunto ótimo diferente: glass L1+L3, lightning L1+L3, polymer L1+L2, vari L1, yoke L1+L2+L3.
2. **Nenhum nível isolado é seguro** (regret máximo ≥ 0,073), e remover L1 (`12`) é sempre ruim (0/25 heads melhoram).
3. **Como default sem conhecimento da classe, L1+L2+L3 é minimax-ótimo** (regret médio 0,0107, máximo 0,0212). L1+L3 tem o menor regret em 2 classes, mas perde 0,044 em polymer.
4. A hipótese mecanística consistente com o padrão é a de escala: defeitos texturais extensos (vari, corrosão) são capturados por L1 (campo receptivo pequeno, alta frequência), e L3 adiciona ruído; defeitos estruturais (glass, peça ausente) dependem de L3 (contexto amplo). Classes de escala intermediária (polymer) usam L2. Essa leitura é compatível com o viés de textura das camadas rasas de CNNs [11], mas, com 5 classes, é uma hipótese, não uma lei.

### 3.4 Agregação de score (QP2)

1. **A normalização min-max por imagem é o maior defeito isolado da receita legada.** Ela custa 0,075 de AUROC em relação ao agregador original e é superada por `raw` em 25/25 heads (91 % dos pares), em validação e holdout. Mecanismo: ao forçar cada mapa a [0, 1], toda imagem — inclusive as normais — ganha um "máximo anômalo", destruindo a comparabilidade entre imagens, da qual AUROC e AUPRO globais dependem.
2. **A aproximação local `per_level_prob` é a pior opção** (−0,113; 0/5 classes) e não representa o agregador CFLOW-AD: omite a normalização pelo máximo da coorte e a inversão pelo máximo do mapa.
3. **`raw`, `per_level_std` e `cflow_original` são equivalentes em AUROC** (Δ ≤ 0,002; IC contém zero; sinais mistos entre classes). O ganho de +0,0138 do agregador original sobre `raw`, relatado em um *sweep* anterior, não se reproduz no grid pareado, que controla interpolação, níveis e σ.
4. **Não são equivalentes em AUPRO.** Sobre a referência, trocar `cflow_original` por `raw` ou `per_level_std` reduz o AUPRO em 0,085 e 0,070 no holdout (OFAT, 15 heads) e em 0,036 e 0,012 na média dos pares da validação. `cflow_original` é, portanto, **Pareto-dominante sobre `raw`** quando AUROC e AUPRO são considerados juntos. Ressalva: `cflow_original` é **transdutivo** (normaliza pelos máximos do lote avaliado), enquanto `per_level_std` usa só estatísticas do treino normal e funciona imagem a imagem. Essa propriedade motiva a recomendação de implantação da Seção 3.6.

### 3.5 Variantes de treino (QP5)

**Tabela D — Pixel AUROC no holdout do sistema escolhido (receita escolhida em validação) por classe e variante.** Em negrito, a variante escolhida pela validação.

| Classe | baseline | featnorm | featnorm_l1_only | clamp19 | featnorm_clamp19 | Amplitude |
|---|---:|---:|---:|---:|---:|---:|
| glass | 0,9156 | **0,9467** | 0,9094 | 0,9435 | 0,9510 | 0,042 |
| lightning | 0,8674 | 0,8605 | 0,8686 | **0,8965** | 0,8607 | 0,036 |
| polymer | 0,8653 | 0,8764 | 0,8682 | 0,8672 | **0,8848** | 0,020 |
| vari | 0,9220 | 0,9277 | 0,9268 | 0,9274 | **0,9295** | 0,008 |
| yoke | 0,9449 | 0,9115 | **0,9466** | 0,9060 | 0,8994 | 0,047 |
| **Média** | 0,9030 | 0,9046 | 0,9039 | 0,9081 | 0,9051 | 0,005 |
| Posto médio | 3,8 | 2,8 | 3,0 | 3,0 | 2,4 | — |

**Tabela E — efeito do treino sob receita fixa (referência `cflow_original`), Δ AUROC no holdout contra a baseline da classe.**

| Classe | featnorm | featnorm_l1_only | clamp19 | featnorm_clamp19 |
|---|---:|---:|---:|---:|
| glass | +0,0331 | +0,0015 | −0,0113 | −0,0211 |
| lightning | +0,0284 | +0,0030 | +0,0626 | +0,0689 |
| polymer | +0,0062 | −0,0150 | +0,0092 | +0,0107 |
| vari | −0,0085 | +0,0043 | −0,0228 | −0,0139 |
| yoke | −0,0121 | +0,0026 | −0,0458 | −0,0475 |
| **Média** | +0,0094 | −0,0007 | −0,0016 | −0,0006 |

Achados:

1. **Não há variante de treino universalmente superior.** As médias entre classes diferem em ≤ 0,005 (Tabela D) e o teste de Friedman não detecta diferença (p = 0,72 no sistema escolhido; p = 0,89 sob receita fixa).
2. **Dentro de cada classe, a variante importa** (amplitude de até 0,047 em yoke) e o sinal muda: clamp 1,9 ajuda lightning (+0,063 sob receita fixa) e prejudica yoke (−0,046) e vari (−0,023).
3. **Treino e inferência interagem.** Em glass, `featnorm_clamp19` piora a head sob a receita fixa (−0,021), mas, com a receita escolhida, chega a 0,951, o maior valor da classe. Comparar variantes sob uma única receita, ou só pelo melhor pós-hoc, leva a conclusões diferentes.
4. `featnorm_l1_only` é a única variante sem perda relevante sob receita fixa em nenhuma classe (|Δ| ≤ 0,015), mas também sem ganho relevante: é uma variante "neutra".

### 3.6 Melhor receita por classe e melhor receita geral (QP4)

**Tabela F — custo hierárquico de padronizar a receita (regret médio de AUROC na validação, 25 heads).**

| Graus de liberdade por classe | Regret médio | Leitura |
|---|---:|---|
| Nenhum: receita única para tudo | 0,0261 (máx. 0,049) | receita única é cara |
| Só níveis (template `per_level_std`, σ8) | 0,0152 | metade da perda se resolve com os níveis |
| Só níveis (template `cflow_original`, σ8) | 0,0137 | idem, versão transdutiva |
| Níveis + score | 0,0101 | |
| Níveis + score + σ | 0,0086 | |
| Níveis + score + σ + padding + TTA | 0,0079 | |
| Receita completa por classe (fixa entre variantes) | 0,0078 | |
| Receita por head | 0 | (por definição) |

O grau de liberdade que mais vale é o **conjunto de níveis por classe**; os demais somam menos de 0,006.

**Melhor receita geral (recomendação de implantação).**

| Componente | Valor | Justificativa |
|---|---|---|
| Treino | `baseline` (clamp 0,5, sem featnorm) | nenhuma variante vence na média (Friedman p = 0,72); parcimônia; variantes alternativas só com validação na classe |
| Padding | 112 px `reflect` | +0,0024 (val.), +0,0061 (holdout); `replicate` não ajuda |
| TTA | 4 flips | +0,0048 (val.), +0,0068 (holdout); 4× custo; dispensável em tempo real |
| Score | `per_level_std` | AUROC equivalente ao original; **indutivo** (estatísticas do treino normal), opera imagem a imagem |
| Níveis | **por classe**: glass 02, lightning 012, polymer 12, vari 0, yoke 012; default sem informação: 012 | Tabelas C e F |
| σ | 8 | maior efeito consistente (25/25 heads) |
| Calibração posicional | nenhuma | −0,025 (0/5 classes) |
| `align_corners` | True | irrelevante (|Δ| < 0,001) |

Os níveis do template foram otimizados dentro dele (`per_level_std`, σ8). Por isso, diferem em lightning (012) e polymer (L2+L3) do melhor conjunto com todos os fatores livres da Tabela C (L1+L3 e L1+L2), porque níveis e score interagem. Ao trocar o score do template, os níveis devem ser escolhidos novamente na validação.

Receita única totalmente fixa (se níveis por classe forem inviáveis): `pad112-reflect | flips | per_level_std | L1+L2+L3 | σ8`, com regret médio de 0,026.

**Tabela G — melhor receita por classe.** "Sistema escolhido" é o resultado do protocolo (Seção 2.4) medido no holdout. "Receita fixa da classe" é a receita única que minimiza o regret médio nas 5 heads da classe (robusta à variante); ela foi escolhida na validação, mas ainda **não foi medida no holdout**.

| Classe | Sistema escolhido (variante · receita) | AUROC holdout | AUPRO holdout | Receita fixa da classe (validação) | Regret médio / máx. |
|---|---|---:|---:|---|---|
| glass | featnorm · pad112 reflect, flips, std, L1+L3, up=F, σ2 | 0,9467 | 0,8422 | pad112 reflect, flips, std, L1+L3, up=F, σ4 | 0,0020 / 0,0052 |
| lightning | clamp19 · pad112 replicate, flips, cflow_original, L1+L2+L3, up=T, σ8 | 0,8965 | 0,6442 | pad112 reflect, flips, raw, L1+L3, up=T, σ12 | 0,0069 / 0,0171 |
| polymer | featnorm_clamp19 · pad112 replicate, hflip, std, L1+L2+L3, up=F, σ4 | 0,8848 | 0,0000* | pad112 replicate, flips, raw, L1+L2, up=T, σ8 | 0,0172 / 0,0448 |
| vari | featnorm_clamp19 · pad112 reflect, hflip, cflow_original, L1, up=T, σ8 | 0,9295 | n.c. | pad112 reflect, hflip, cflow_original, L1, up=T, σ8 | 0,0006 / 0,0024 |
| yoke | featnorm_l1_only · pad112 reflect, flips, std, L1+L3, up=T, σ12 | 0,9466 | n.c. | pad112 reflect, hflip, cflow_original, L1+L2+L3, up=T, σ12 | 0,0125 / 0,0319 |

n.c.: não calculado (D-14). *Valor salvo pela métrica; ver Seção 3.9. Para polymer, a alternativa com AUPRO positivo é `featnorm` (AUROC 0,8764, AUPRO 0,7131). Em yoke, `baseline` (0,9449) e `featnorm_l1_only` (0,9466) são um empate prático: diferença de validação de 0,0008, com 18 anomalias.

### 3.7 Dinâmica de treino

Época do pico de AUROC de validação (score `raw`, σ4) / épocas treinadas:

| Classe | baseline | featnorm | featnorm_l1_only | clamp19 | featnorm_clamp19 |
|---|---|---|---|---|---|
| glass | 15/40 | 40/40 | 10/40 | 5/40 | 10/40 |
| lightning | 15/40 | 35/40 | 40/40 | 30/40 | 35/40 |
| polymer | 30/30 | 15/30 | 30/30 | 12/30 | 15/30 |
| vari | 10/40 | 5/40 | 20/40 | 5/40 | 5/40 |
| yoke | 2/12 | 2/12 | 2/12 | 2/12 | 2/12 |

Em yoke (4.834 imagens normais), todas as heads atingem o pico na **primeira** avaliação e depois perdem até 0,024 (clamp19: 0,8831 → 0,8587). Em vari, 3/5 heads atingem o pico na primeira avaliação. Em lightning e polymer, há heads com pico na última avaliação. O orçamento ideal de épocas depende da classe e da variante; um orçamento fixo é ou curto demais ou longo demais. A resolução das avaliações no início do treino (a cada 2–5 épocas) é insuficiente para localizar o pico em yoke.

### 3.8 Localização por pixel versus decisão por imagem

Em **25/25 heads**, a AUROC por imagem derivada do mapa (max/top-k) é menor que a do SEDifferNet no mesmo holdout:

| Classe | Mapa (faixa nas 5 heads) | SEDifferNet |
|---|---|---:|
| glass | 0,709 – 0,743 | 0,771 |
| lightning | 0,665 – 0,753 | 0,948 |
| polymer | 0,630 – 0,692 | 0,950 |
| vari | 0,708 – 0,786 | 0,904 |
| yoke | 0,787 – 0,868 | 0,958 |

A receita foi otimizada para pixel, não para imagem; ainda assim, a separação indica que detector e localizador devem permanecer como estágios distintos. A AP de pixel é baixa em termos absolutos (glass 0,0025–0,0047; lightning 0,020–0,028; polymer 0,030–0,033), coerente com prevalências de pixels anômalos abaixo de 1 %. AUROC alta não implica segmentação operacional em limiar fixo.

### 3.9 Colapso da AUPRO global com limiares lineares

Quatorze linhas do holdout registram AUPRO = 0,0000. Todas estão em heads com clamp 1,9 (11 em polymer/`featnorm_clamp19`, 3 em glass) e usam `per_level_std` ou `raw`. Para essas mesmas receitas:

- a AUROC é alta (0,876–0,909);
- a AUPRO por imagem (estilo RD++) é 0,52–0,69;
- a mesma receita tem AUPRO de 0,74 na validação;
- a mesma receita sem TTA tem AUPRO de 0,706 no holdout.

A implementação oficial usa 1.000 limiares **igualmente espaçados** entre o mínimo e o máximo globais do mapa. Scores não limitados (z-score por nível, NLL bruta), com caudas mais pesadas em heads de clamp alto, podem concentrar todos os limiares com FPR ≤ 0,3 acima de todos os pixels das regiões anômalas: a curva selecionada fica com PRO = 0 e a área, zero. Esse é o mesmo mecanismo do *bug* do AUPRO de treino corrigido anteriormente por limiares em quantis. A explicação é **provável, mas ainda não verificada** nos mapas. Até lá, o valor 0 é tratado como artefato da métrica, e não como falha de localização. Para publicação, é necessário: (i) recalcular com limiares em quantis dos pixels normais; (ii) reportar as duas versões; (iii) verificar se o ranking muda.

### 3.10 Custo computacional

Tempo de parede total das duas rodadas: ≈ 145 h (treino ≈ 35 h; avaliação pós-hoc ≈ 110 h) em i7-13650HX e RTX 4060 Laptop de 8 GB. O gargalo é a avaliação do grid em CPU com uma única thread e paginação de memória (yoke: 405–593 min por head), não a GPU.

---

## 4. Discussão

**Por que a inferência domina.** Com o extrator congelado, a head só pode reordenar posições dentro do espaço de features já existente; decisões de inferência mudam diretamente a função de score avaliada. Duas delas são estruturais: a agregação (min-max por imagem destrói a escala absoluta da verossimilhança, da qual dependem as métricas globais) e o conjunto de níveis (um nível sem sinal soma ruído à NLL). As demais (σ, TTA, padding) são correções de segunda ordem, mas consistentes: atenuam ruído de alta frequência, assimetrias do prior posicional e artefatos de borda do *zero padding* [9, 10].

**Por que não há variante de treino universal.** Normalização por canal e clamp alteram a geometria da NLL de cada nível de modos diferentes. Seu efeito líquido depende de qual nível carrega o sinal na classe e de qual receita lê a NLL depois. Isso explica a troca de sinal entre classes e a interação treino × receita. Para quem implanta, a consequência é parcimônia: `baseline` como padrão e variantes adotadas apenas mediante validação na classe.

**Relação com a literatura.** O resultado de que a agregação min-max por imagem prejudica as métricas globais é coerente com a formulação original do CFLOW-AD, que normaliza pela coorte e não por imagem [2]. O agregador original, porém, é transdutivo. Mostramos que um z-score escalar com estatísticas do treino normal (`per_level_std`) iguala sua AUROC sem depender do lote, ao custo de AUPRO menor, um *trade-off* raramente explicitado. A importância de escolher níveis por classe dialoga com PaDiM/PatchCore [4, 5], que fixam níveis intermediários por convenção.

**Implicação metodológica.** Comparar variantes de treino apenas sob uma receita fixa, ou apenas pelo melhor pós-hoc, leva a conclusões diferentes (Seção 3.5). Estudos de ablação de heads de densidade deveriam reportar os dois.

---

## 5. Ameaças à validade

| Ameaça | Natureza | Mitigação atual | Mitigação necessária |
|---|---|---|---|
| Uma seed de treino | interna | bootstrap por imagem (não cobre seed) | ≥ 3 seeds nas finalistas |
| Um split; holdout já inspecionado | interna | seleção só em validação; manifestos auditados | holdout novo ou validação cruzada aninhada (Seção 9) |
| 5 classes como unidade de generalização | externa/estatística | concordância de sinal, IC hierárquico | mais classes ou datasets (MVTec-AD, VisA) |
| Critério de seleção misto (AUROC+AUPRO vs. AUROC) | interna | declarado; efeitos de fator calculados sobre o grid inteiro, independentes do critério | reexecutar a seleção com um único critério |
| AUPRO ausente em 2 classes; colapso em 14 linhas | construto | declarado; AUPRO legado como sanidade | recalcular AUPRO com limiares por quantis em todas as classes |
| `cflow_original` transdutivo | construto | val e holdout normalizados separadamente | recomendar `per_level_std` para implantação; testar calibração fixa |
| Multiplicidade (≈ 132 mil receitas) | conclusão | efeitos grandes e replicados validação → holdout | correção de multiplicidade nas comparações confirmatórias pré-registradas |
| Backbone selecionado antes do split local | interna | não auditado | auditar a origem dos checkpoints SEDifferNet |
| Head local ≠ CFLOW-AD | construto | terminologia explícita | reprodução end-to-end, se a comparação com CFLOW-AD for alegada |
| Grid em stride 2 | medição | holdout em stride 1 | — (efeito pequeno observado) |

---

## 6. Reconciliação com documentos anteriores

Os documentos de setembro ([plano](plano_ablation_nf_head.md), [relatório acadêmico](relatorio_academico_ablation_nf_head.md), [artigo](artigo_ablation_cflow_differnet.md)) usavam um protocolo em que, em parte das rodadas, a época era escolhida no teste completo, a AUPRO era a versão por imagem e o grid era menor. **Os números abaixo substituem os anteriores para fins de publicação.**

| Afirmação anterior | Valor anterior | Resultado sob o protocolo de outubro | Status |
|---|---|---|---|
| `raw` vs min-max é o maior efeito | +0,0497 | +0,0854, 25/25 heads | **Confirmado e ampliado** |
| L1+L3 é o default universal, regret ≤ 0,0012 em 4/5 classes | ≤ 0,0012 | regret 0 / 0,0025 / 0,0439 / 0,0079 / 0,0096; L1+L2+L3 é minimax | **Refutado** |
| `featnorm` é destrutivo em yoke | −0,0499 AUROC / −0,1226 AUPRO | −0,0121 (receita fixa); −0,0334 (sistema) | Direção confirmada, magnitude menor |
| vari: L1 isolado é o melhor | 0,9191 | regret de L1 = 0,0005; 4/5 vencedoras usam L1 | **Confirmado** |
| TTA flips | +0,0120 | +0,0048 (val.), +0,0068 (holdout) | Confirmado, menor |
| Reflect pad 112 | +0,0049 | +0,0024 (val.), +0,0061 (holdout); `replicate` ≈ 0 | Confirmado, pequeno |
| σ ótimo = 8 (σ=2 em glass) | — | σ=8 é o maior efeito; glass prefere σ ≤ 4 (regret de σ8 = 0,0052) | Confirmado |
| `cflow_original` > `raw` (+0,0138) | sweep sem controle de interpolação/níveis | +0,0009, IC [−0,0085; +0,0083] | **Não reproduzido** em AUROC; superior em AUPRO |
| Calibração posicional prejudica | −0,034 | −0,0248, 0/5 classes | Confirmado |
| Melhores AUROC por classe (setembro) | 0,9560 / 0,8815 / 0,8806 / 0,9270 / 0,9188 | 0,9467 / 0,8965 / 0,8848 / 0,9295 / 0,9466 | Substituídos (seleção limpa) |
| Ganho do pós-processamento | +0,0965 (validação, cadeia) | +0,0625 (holdout, baseline); +0,0804 com variante | Substituído |

---

## 7. Recomendações práticas

1. Nunca usar min-max por imagem nem a aproximação `per_level_prob` em mapas NLL.
2. Usar o *template* da Seção 3.6 com níveis por classe; sem informação da classe, usar L1+L2+L3.
3. Treinar `baseline`; adotar outra variante apenas se vencer na validação da classe por uma margem pré-definida (sugestão: ≥ 0,005 de AUROC).
4. Selecionar épocas com avaliação densa no início e *early stopping* pela validação.
5. Manter o detector de imagem (SEDifferNet) separado do localizador.
6. Reportar AUROC e AUPRO (com limiares robustos) juntos; nunca selecionar por AUROC isoladamente quando a cobertura de região importa.

---

## 8. Reprodutibilidade

- Treino e pós-hoc: [`run_nf_head_ablation.ps1`](../scripts/eval/run_nf_head_ablation.ps1) → [`ablation_nf_head.py`](../scripts/eval/ablation_nf_head.py) → [`pixel_train_from_pretrained.py`](../scripts/train/pixel_train_from_pretrained.py).
- Consolidação das tabelas deste documento (somente leitura de artefatos):
  `python scripts/eval/consolidate_nf_head_ablation.py`
- Artefatos por head: `config.txt`, `split_manifest.json`, `history.json`, `grid_val.csv`, `holdout_test_top.csv`, `summary.txt`.
- Ambiente: Python (Conda `anomaly_attentideffernet`), PyTorch 2.5.1, RTX 4060 Laptop 8 GB.
- Pendência de rastreabilidade: o código não foi arquivado com *hash* de commit junto às rodadas; registrar o commit antes da rodada confirmatória.

---

## 9. O que falta para uma submissão Qualis A

Os achados atuais são suficientes para um **estudo exploratório bem controlado**. Para uma revista Qualis A, revisores esperarão uma etapa confirmatória com hipóteses e critérios fixados antes de ver os dados. Proposta, em ordem de prioridade:

1. **Pré-registro interno** (data e commit) das hipóteses: H1 — o template da Seção 3.6 supera a receita legada em ≥ 0,03 de AUROC em ≥ 4/5 classes; H2 — níveis por classe superam níveis fixos; H3 — nenhuma variante de treino supera a baseline na média entre classes.
2. **Avaliação independente:** validação cruzada estratificada em 5 *folds* do conjunto de teste rotulado, com seleção aninhada, ou novo split nunca inspecionado. Essa é a principal fragilidade atual.
3. **Seeds:** 3 seeds de treino nas variantes finalistas (`baseline` e a vencedora de cada classe), com IC entre seeds.
4. **AUPRO robusta em todas as classes:** limiares por quantis de pixels normais, somados à versão oficial, nas receitas finalistas (inclusive vari e yoke).
5. **Receitas fixas no holdout:** medir o template geral e as receitas fixas da Tabela G (ainda medidas só na validação).
6. **Baselines externos no mesmo split:** PatchCore, PaDiM, CFLOW-AD oficial (WRN50) e RD++, para posicionar o sistema; os números do RD++ existentes vêm de outro protocolo.
7. **Custo:** latência por imagem e memória (com e sem TTA), essenciais para o argumento de embarque em VANT.
8. **Grid na GPU:** reduzir o custo de cerca de 110 h de avaliação e viabilizar os itens 2–5.

Sem os itens 2–4, a contribuição publicável mais segura é a **metodológica**: a decomposição do ganho, a dominância da inferência com extrator congelado, a não universalidade dos níveis e das variantes de treino e o colapso da AUPRO linear.

---

## 10. Resumo dos achados

1. **Ganho:** a Pixel AUROC média no holdout sobe de 0,841 para 0,921 (+0,080) nas 5 classes; ≈ 78 % do ganho vem da inferência.
2. **Inferência > treino:** nenhuma variante de treino vence na média entre classes (Friedman p = 0,72); dentro da classe, a variante muda o resultado em até 0,047 e interage com a receita.
3. **Agregação:** min-max por imagem (−0,075) e `per_level_prob` (−0,113) devem ser abandonados; `raw`, `per_level_std` e `cflow_original` empatam em AUROC, e `cflow_original` vence em AUPRO.
4. **Níveis:** a escolha mais sensível à classe (vari L1; glass/lightning L1+L3; polymer L1+L2; yoke L1+L2+L3). Sem informação da classe, L1+L2+L3 é minimax. A afirmação anterior de L1+L3 universal foi refutada.
5. **Fatores consistentes:** σ = 8 (+0,016; 25/25 heads), TTA flips (+0,005), pad 112 `reflect` (+0,002); calibração posicional prejudica (−0,025); `align_corners` é irrelevante.
6. **Melhor receita geral:** `baseline` + pad 112 `reflect` + flips + `per_level_std` + σ8 + níveis por classe (regret 0,015). Receita única totalmente fixa: regret 0,026.
7. **Melhores sistemas por classe (holdout):** glass 0,9467, lightning 0,8965, polymer 0,8848, vari 0,9295, yoke 0,9466.
8. **Riscos para publicação:** uma seed, holdout já visto, AUPRO ausente ou colapsada e falta de baselines externos no mesmo split. A Seção 9 define a etapa confirmatória.

---

## Referências

[1] M. Rudolph, B. Wandt, B. Rosenhahn. Same Same But DifferNet: Semi-Supervised Defect Detection with Normalizing Flows. WACV, 2021.  
[2] D. Gudovskiy, S. Ishizaka, K. Kozuka. CFLOW-AD: Real-Time Unsupervised Anomaly Detection with Localization via Conditional Normalizing Flows. WACV, 2022.  
[3] M. Rudolph, T. Wehrbein, B. Rosenhahn, B. Wandt. Fully Convolutional Cross-Scale-Flows for Image-based Defect Detection. WACV, 2022.  
[4] K. Roth et al. Towards Total Recall in Industrial Anomaly Detection. CVPR, 2022.  
[5] T. Defard, A. Setkov, A. Loesch, R. Audigier. PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization. ICPR Workshops, 2021.  
[6] H. Deng, X. Li. Anomaly Detection via Reverse Distillation from One-Class Embedding. CVPR, 2022.  
[8] A. L. B. Vieira-e-Silva et al. InsPLAD: A Dataset and Benchmark for Power Line Asset Inspection in UAV Images. Int. J. Remote Sensing, 44(23), 2023.  
[9] B. Alsallakh et al. Mind the Pad — CNNs can Develop Blind Spots. ICLR, 2021.  
[10] M. A. Islam, S. Jia, N. D. B. Bruce. How Much Position Information Do Convolutional Neural Networks Encode? ICLR, 2020.  
[11] R. Geirhos et al. ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness. ICLR, 2019.

A numeração segue a do [artigo anterior](artigo_ablation_cflow_differnet.md) para facilitar a fusão; [7] e [12] não são citados aqui.
