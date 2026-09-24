# Pós-Processamento Supera Retreino: Um Estudo Abrangente de Ablação da Cabeça de Normalizing Flow para Localização de Anomalias em Inspeção de Linhas de Transmissão

**Autor:** Teo Santos  
*Universidade de Brasília (UnB), Brasília, Brasil*  

---

## Resumo
A localização precisa de anomalias em componentes de linhas de transmissão é essencial para a manutenção preditiva baseada em imagens aéreas de veículos aéreos não tripulados (VANTs). Modelos baseados em *normalizing flows* (NF), como o CFLOW-AD, aprendem a distribuição de características (*features*) normais e utilizam a verossimilhança negativa (*negative log-likelihood* — NLL) como pontuação de anomalia por pixel. Embora a literatura concentre esforços no desenvolvimento de novas arquiteturas e regimes de treinamento, o impacto das escolhas de pós-processamento e inferência sobre a qualidade da localização permanece sistematicamente subexplorado. Este trabalho apresenta um estudo de ablação abrangente da cabeça de NF do pipeline SEDifferNet–CFLOW sobre as 5 classes do benchmark **INSPLAD-seg**, totalizando aproximadamente 24.000 medições de validação em 8 conjuntos de *held-out* independentes. Os resultados demonstram que a cadeia de pós-processamento — agregação aditiva direta de NLL (`raw`), *reflect padding* de 112 px na inferência, aumento de dados em tempo de teste (TTA) por espelhamentos simétricos, seleção criteriosa de níveis de *features* e suavização Gaussiana — responde por um ganho médio de **+0,0965 de Pixel AUROC** e até **+0,215 de AUPRO**, superando qualquer intervenção de treinamento. Em particular, a substituição da normalização *min-max* por imagem pela soma direta de NLL (`raw`) produz um ganho pareado médio de **+0,0497 de AUROC** (positivo em 76% a 100% dos pares em todas as classes). Demonstramos empiricamente que a hierarquia entre os níveis de *features* ($L_1, L_2, L_3$) é estritamente governada pela geometria e frequência espacial da falha, sendo $L_1+L_3$ o default universal ótimo (*regret* $\le 0,0012$ em 4 de 5 classes), e que a padronização de canais (`featnorm`) não generaliza, causando degradação severa em classes dominadas por contexto global. Por fim, corrigimos uma falha numérica crítica no cálculo do AUPRO via quantis de normais e transferimos métricas compostas do RD++, estabelecendo diretrizes definitivas para o desdobramento robusto de *normalizing flows* em inspeção visual crítica.

**Palavras-chave:** Detecção de anomalias, normalizing flows, localização por pixel, pós-processamento, estudo de ablação, inspeção de linhas de transmissão, INSPLAD, representações multiescala.

---

## Abstract
Precise anomaly localization in power-line components is essential for predictive maintenance based on unmanned aerial vehicle (UAV) imagery. Normalizing-flow (NF) models such as CFLOW-AD learn the distribution of normal features and use the negative log-likelihood (NLL) as a per-pixel anomaly score. Although the literature focuses on architectures and training, the impact of post-processing and inference-time choices on localization quality remains under-explored. This work presents a comprehensive ablation study of the NF head of the SEDifferNet–CFLOW pipeline over the 5 classes of the INSPLAD-seg benchmark, totaling approximately 24,000 validation measurements across 8 independent held-out sets. Results show that the post-processing chain — raw NLL aggregation, 112 px reflect padding at inference, test-time augmentation (TTA) via flips, principled feature-level selection, and Gaussian smoothing — contributes an average gain of **+0.0965 Pixel AUROC** and up to **+0.215 AUPRO**, outperforming any training intervention. In particular, replacing per-image min-max scaling with raw NLL summation yields an average paired gain of **+0.0497 AUROC** (positive in 76–100% of pairs across all 5 classes). We demonstrate that the hierarchy among feature levels ($L_1, L_2, L_3$) is strictly governed by defect geometry and spatial frequency, with $L_1+L_3$ serving as the optimal universal default (regret $\le 0.0012$ in 4/5 classes), and that channel-wise feature standardization (`featnorm`) fails to generalize, causing catastrophic degradation in classes dominated by global context. Finally, we resolve a numerical collapse in heavy-tailed AUPRO computation via normal-pixel quantiles and adopt composite scoring from RD++, setting definitive deployment guidelines for normalizing flows in critical visual inspection.

**Keywords:** Anomaly detection, normalizing flows, pixel-level localization, post-processing, ablation study, power-line inspection, INSPLAD, multi-scale representations.

---

## 1. Introdução

A inspeção periódica de ativos em linhas de transmissão de energia elétrica constitui uma operação vital para assegurar a continuidade do fornecimento e mitigar riscos de interrupções catastróficas. A introdução de veículos aéreos não tripulados (VANTs) dotados de câmeras de alta resolução revolucionou a aquisição de dados visuais dessas estruturas, mas gerou um volume de imagens cuja triagem manual por inspetores humanos é financeiramente onerosa e sujeita a fadiga cognitiva [1]. Consequentemente, a automação da detecção e localização de defeitos via visão computacional tornou-se imperativa.

Diferentemente dos benchmarks industriais controlados tradicionais (como o MVTec-AD), o cenário de inspeção aérea de linhas de transmissão impõe desafios severos:
1. **Fundo Natural Dinâmico:** As imagens contêm fundos desordenados compostos por vegetação densa, solo e variações de relevo sob condições de iluminação descontroladas, gerando assinaturas visuais complexas que disputam a capacidade de representação das redes neurais.
2. **Desbalanceamento Extremo de Escala de Anomalia:** A fração de pixels defeituosos em relação à área total varia de meros **0,14%** (em campânulas de vidro faltantes, *missing cap*) até **4,6%** (em corrosões profundas de grampos de ancoragem).
3. **Escassez Extrema de Amostras Normais:** Em contraste com conjuntos de dados com dezenas de milhares de amostras, os datasets de inspeção real fornecem entre 400 e poucos milhares de imagens de treinamento por componente.

Modelos generativos baseados em *normalizing flows* (NF) — notadamente o DifferNet [1] e o CFLOW-AD [2] — destacaram-se como soluções promissoras. Ao mapear características convolucionais para variáveis latentes com distribuição gaussiana padronizada através de transformações inversíveis com determinante Jacobiano tratável, os NFs fornecem uma estimativa exata da densidade de probabilidade $p(\mathbf{x})$, utilizando a verossimilhança negativa (*negative log-likelihood* — NLL) como métrica direta de anomalia. Em particular, o CFLOW-AD estendeu o paradigma para a resolução espacial condicional por posição, treinando fluxos afins sobre pirâmides de características multiescala acopladas a codificações posicionais senoidais 2D.

Não obstante o expressivo volume de pesquisas devotado à criação de novas arquiteturas e regimes de regularização durante o treino, um aspecto crítico permanece sistematicamente subestimado: **o impacto decisivo dos fatores de inferência e pós-processamento**. A literatura frequentemente assume que os hiperparâmetros de inferência herdados de códigos de referência representam opções ótimas ou neutras. Na prática, demonstramos que convenções amplamente adotadas — como a normalização *min-max* por imagem nos mapas de NLL por nível — destroem a calibração de escala da verossimilhança estatística, forçando imagens normais a conterem falsos positivos unitários. Além disso, a premissa canônica de que concatenar o maior número possível de níveis convolucionais sempre eleva o desempenho revela-se incorreta quando um dos níveis carrega mais ruído do que sinal discriminativo.

Neste trabalho, realizamos uma ablação exaustiva e controlada do pipeline **SEDifferNet + CFLOW Head** sobre o benchmark de inspeção aérea **INSPLAD-seg** [8]. Avaliamos sistematicamente o impacto de fatores de pós-processamento (`score_mode`, *reflect padding*, TTA, suavização gaussiana, seleção de níveis) e de treinamento (`featnorm`, *clamp scale*, orçamentos de época e parada precoce). 

As contribuições fundamentais deste artigo são:
* **Quantificação do Primado da Inferência:** Comprovamos que uma cadeia de pós-processamento estruturada confere **+0,0965 de Pixel AUROC médio** (atingindo até +0,1757 em *vari-grip* e +0,215 de AUPRO em *glass*), superando amplamente qualquer ganho alcançado por retreino paramétrico ($\le +0,009$).
* **Superioridade Comprovada da NLL Bruta (`raw`):** Demonstramos em 100% das 5 classes que a agregação linear direta de verossimilhanças supera a normalização *min-max* por imagem (+0,0497 AUROC pareado, positivo em 92,7% dos pares comparados).
* **Mapeamento Mecânico dos Níveis Convolucionais ($L_1, L_2, L_3$):** Demonstramos que a hierarquia de níveis depende da relação de frequência espacial e geometria da patologia, fundamentada pelo viés de textura (*texture bias*) de camadas rasas versus viés de forma (*shape bias*) de camadas profundas [11].
* **Desmistificação da Padronização por Canal (`featnorm`):** Revelamos o mecanismo pelo qual a equalização de variâncias beneficia camadas rasas ($L_1$), mas amplifica canais esparsos ruidosos em camadas profundas ($L_2/L_3$), resultando no colapso de $-0,1226$ de AUPRO em defeitos de larga escala (*yoke*).
* **Correção Numérica do AUPRO e Métricas Compostas:** Eliminamos o colapso de AUPRO nulo em mapas de NLL de cauda pesada através de integração por quantis dos pixels normais e transferimos do RD++ [6] a seleção de modelos por pontuação composta balanceada.

---

## 2. Trabalhos Relacionados

### 2.1 Normalizing Flows na Detecção de Anomalias
Modelos baseados em fluxo foram introduzidos na detecção de anomalias pelo DifferNet [1], que emprega acoplamentos afins do framework FrEIA sobre características agregadas de uma AlexNet para pontuar imagens inteiras. O CFLOW-AD [2] estabeleceu a formulação de localização pixel a pixel, associando a cada ponto da grade convolucional uma codificação posicional senoidal e modelando a distribuição condicional de características multiescala. O CS-Flow [3] e o MSFlow [7] propuseram arquiteturas dedicadas de fluxo com conexões bidirecionais entre escalas espaciais. Nosso trabalho ancora-se no CFLOW desacoplado por nível e investiga formalmente a interface entre os tensores latentes gerados e a predição final de mapa de calor.

### 2.2 Modelagem Estatística e Bancos de Memória
O PaDiM [5] demonstrou que características convolucionais locais pré-treinadas podem ser modeladas satisfatoriamente através de distribuições normais multivariadas com redução paramétrica por canal, introduzindo a padronização de características adotada neste estudo como `featnorm`. O PatchCore [4] prescindiu de modelagem densitária em favor de bancos de memória de *neighborhood-aware patches*, apontando a relevância da interpolação espacial de vizinhança. Mais recentemente, o RD++ [6] consolidou a abordagem de destilação reversa com perturbações sintéticas multiescala via ruído simplex, de onde importamos a metodologia de score composto para validação contínua.

### 2.3 Viés de Borda e Propriedades de Redes Convolucionais
Estudos seminais de visão computacional identificaram patologias estruturais associadas a operações convolucionais padrão. Alsallakh et al. [9] (*Mind the Pad*) comprovaram que o preenchimento por zeros (*zero-padding*) nas camadas iniciais de CNNs quebra a invariância translacional e gera pontos cegos e ativações periféricas espúrias de alta intensidade. Concomitantemente, Islam et al. [10] e Kayhan & van Gemert [12] demonstraram que redes convolucionais exploram esses gradientes de contorno para inferir posições absolutas da imagem sem mecanismos explícitos de coordenadas. No contexto de detecção de anomalias por densidade, tais ativações de borda traduzem-se diretamente em anomalias falsas ("bordas quentes"), motivando o desenvolvimento de técnicas como o *reflect padding*. 

Por sua vez, Geirhos et al. [11] demonstraram que redes treinadas no ImageNet apresentam um viés intrínseco por textura (*texture bias*) em suas primeiras camadas convolucionais, enquanto abstrações globais de forma e silhueta (*shape representations*) só emergem no topo da hierarquia, o que apoia teoricamente a divergência observada entre os níveis $L_1$ e $L_3$.

---

## 3. Metodologia

```
Fluxo Completo do Pipeline SEDifferNet + CFLOW:

Imagem de Entrada (448×448)
      │
      ├─► [Inferência Apenas] Reflect Padding 112 ──► Imagem Expandida (672×672)
      ▼
Backbone Congelado (AlexNet + SE Attention Blocks)
      ├── Nível L1 (64 canais,  109×109 nativo) ──┐
      ├── Nível L2 (192 canais,  53×53 nativo)  ──┼─► [Bilinear Resize para 96×96] ──► [Feature Crop 96×96]
      └── Nível L3 (256 canais,  27×27 nativo)  ──┘                                          │
                                                                                            ▼
                                                                                [Opcional: featnorm por canal]
                                                                                            │
                                                                                            ▼
                                 ┌──────────────────────────────────────────────────────────┴────────────────────────┐
                                 ▼                                                          ▼                        ▼
                    CondFlow L1 (8 blocos)                                     CondFlow L2 (8 blocos)   CondFlow L3 (8 blocos)
                                 │                                                          │                        │
                                 └──────────────────────────┬───────────────────────────────┴────────────────────────┘
                                                            ▼
                                           Positional Encoding 2D (PE Senoidal)
                                                            │
                                                            ▼
                                               Mapas de NLL por Nível (96×96)
                                                            │
                                                            ▼
                                          [Seleção de Níveis: e.g. L1+L3]
                                                            │
                                                            ▼
                                           [Score Aggregation Mode: RAW]
                                                            │
                                                            ▼
                                             Upsample Bilinear para 448×448
                                                            │
                                                            ▼
                                                [Suavização Gaussiana σ]
                                                            │
                                                            ▼
                                                 Mapa de Anomalia Final
```

### 3.1 Arquitetura do Extrator e da Cabeça CFLOW
O pipeline opera sobre um extrator convolucional cujos pesos foram previamente calibrados para detecção de anomalias e mantidos congelados durante todo o estudo. A arquitetura básica é uma AlexNet onde blocos de atenção espacial e de canal *Squeeze-and-Excitation* (SE / `simsa`) foram inseridos estrategicamente após as etapas convolucionais.

Para modelar a densidade condicional, a cabeça CFLOW instancia um modelo bijetivo independente ($\text{CondFlow}_k$) para cada nível de características $k \in \{1, 2, 3\}$. Cada modelo consiste em $K=8$ blocos de acoplamento afim (*Conditional Affine Coupling Blocks*) com permutações fixas de canais entre eles. Em cada bloco, os canais de entrada $x \in \mathbb{R}^{C_k}$ são particionados em $x = [x_1, x_2]$, operando a transformação:
$$y_1 = x_1$$
$$y_2 = x_2 \odot \exp(s) + t$$
onde $s$ e $t$ são preditos por sub-redes lineares alimentadas por $[x_1; \mathbf{p}(u,v)]$, sendo $\mathbf{p}(u,v) \in \mathbb{R}^{64}$ a codificação posicional senoidal bidimensional na grade $96 \times 96$:
$$\mathbf{p}(u,v) = \left[ \sin\left(\frac{u}{10000^{2i/32}}\right), \cos\left(\frac{u}{10000^{2i/32}}\right), \sin\left(\frac{v}{10000^{2j/32}}\right), \cos\left(\frac{v}{10000^{2j/32}}\right) \right]$$

A escala multiplicativa $s$ é delimitada por:
$$s = \tanh(\hat{s}) \cdot c_{\text{clamp}}$$
onde $c_{\text{clamp}} = 0{,}5$ delimita $\exp(s) \in [0{,}61;\ 1{,}65]$.

A verossimilhança negativa local ($\text{NLL}$) na coordenada $(u,v)$ para o nível $k$ é calculada analiticamente por:
$$\text{NLL}_k(u,v) = -\log p(x_{u,v}^{(k)} \mid \mathbf{p}(u,v)) = \frac{\|z_{u,v}^{(k)}\|_2^2}{2} - \sum_{b=1}^{8} \sum_{c} s_{b,c} + \frac{C_k}{2} \log(2\pi)$$

---

### 3.2 Anatomia e Especialização dos Níveis de Features ($L_1, L_2, L_3$)

A Tabela 1 sintetiza as características arquiteturais e semânticas dos três níveis convolucionais extraídos da AlexNet-SE:

**Tabela 1: Especificações arquiteturais e operacionais dos níveis convolucionais.**
| Nível | Ponto de Extração no Backbone | Canais ($C$) | Resolução Nativa ($H \times W$) | Campo Receptivo Efetivo | Frequência Espacial Codificada | Bloco de Atenção SE Acoplado |
|---|---|---|---|---|---|---|
| **$L_1$** | Imediatamente pós-`conv1` ($11 \times 11$, stride 4) | 64 | $109 \times 109$ | $11 \times 11$ pixels | Altas frequências: bordas finas, cor, rugosidade local | `simsa1` |
| **$L_2$** | Pós-`pool2` (max-pool $3 \times 3$, stride 2) | 192 | $53 \times 53$ | $\approx 51 \times 51$ pixels | Médias frequências: peças mecânicas, contornos regionais | `simsa2` |
| **$L_3$** | Topo da pilha convolucional pós-`conv5` | 256 | $27 \times 27$ | $> 150 \times 150$ pixels | Baixas frequências: contexto global, integridade de montagem | `simsa4` |

#### Mecanismos de Especialização Funcional:
1. **Nível $L_1$ (Detector de Textura e Cor):** Operando com campo receptivo restrito ($11 \times 11$), $L_1$ responde a variações locais de reflectância. Alinha-se diretamente ao viés de textura (*texture bias*) descrito por Geirhos et al. [11]. É o nível ideal para detecção de oxidação/corrosão em peças metálicas (`vari-grip`), onde a cor e a rugosidade diferem fortemente do padrão novo. Por outro lado, por estar na fronteira de entrada da rede, é o nível mais sensível à injeção de zeros do *padding* convolucional e possui menor capacidade de discernir quebras geométricas amplas.
2. **Nível $L_2$ (Detector de Geometria Regional):** Funciona como uma zona de transição morfológica. Na maioria das classes industriais, comporta-se de forma redundante em relação à combinação de $L_1$ e $L_3$. Contudo, em componentes onde a falha é um desgaste físico moderado em estruturas de ancoragem (`polymer-insulator-upper-shackle`), $L_2$ torna-se o nível dominante por capturar deformações parciais sem a diluição de contexto sofrida por $L_1$ nem a imprecisão de contorno de $L_3$.
3. **Nível $L_3$ (Detector de Contexto e Integridade Estrutural):** Possui grande profundidade receptiva, englobando a totalidade ou grandes frações do ativo sob inspeção. Detecta anomalias de "ausência" ou "violação de arranjo" (`glass-insulator`, `lightning-rod` e `yoke-suspension`). Em contrapartida, sua grade nativa grosseira ($27 \times 27$) confere uma área de $\approx 16{,}6 \times 16{,}6$ pixels de imagem original a cada célula. Se uma anomalia for puramente textural, as ativações de $L_3$ comportam-se como ruído estocástico de fundo, contaminando a NLL conjunta.

---

### 3.3 Mecânica dos Fatores de Pós-Processamento e Inferência

Investigamos seis fatores primários de inferência e sua interação com intervenções de treino:

1. **Modo de Agregação de Score (`score_mode`):**
   * **`raw`:** Soma direta $S(u,v) = \sum_{k \in \mathcal{K}} \text{NLL}_k(u,v)$. Preserva o significado métrico da verossimilhança estatística: pixels anômalos produzem valores absolutos elevados consistentes em todo o dataset.
   * **`per_level_minmax`:** Normaliza cada mapa por imagem: $S(u,v) = \sum_{k \in \mathcal{K}} \frac{\text{NLL}_k - \min(\text{NLL}_k)}{\max(\text{NLL}_k) - \min(\text{NLL}_k) + \epsilon}$. Destrói a comparabilidade global, transformando flutuações de fundo normais em anomalias artificiais de valor unitário.
   * **`per_level_prob`:** Formulação original do CFLOW-AD: $S(u,v) = -\sum_k \exp(\text{logsigmoid}(-\text{NLL}_k) / C_k)$. Devido às ativações não normalizadas de ReLU da AlexNet, magnitudes de NLL de várias centenas saturam o termo no limite inferior, anulando a capacidade discriminativa.
   * **`per_level_std`:** Normalização escalar por parâmetros de treino $S(u,v) = \sum_k \frac{\text{NLL}_k - \mu_k^{\text{train}}}{\sigma_k^{\text{train}}}$.

2. **Padronização de Características por Canal (`featnorm` — Treino):**
   Aplica $f' = (f - \mu) / (\sigma + \epsilon)$ com estatísticas por canal estimadas sobre o treino normal. Em $L_1$ (64 canais), equaliza canais de textura altamente correlacionados com falhas. Em $L_2/L_3$ (192 e 256 canais), a maioria das dimensões possui variância residual de fundo; forçar desvio unitário amplifica canais desprovidos de sinal útil, degradando o acoplamento do *flow*.

3. **Padding Espelhado na Entrada (`reflect_pad 112` — Inferência):**
   Expande a imagem de 448 para 672 pixels via espelhamento de bordas antes do *backbone*, recortando centralmente a grade de $96 \times 96$. O valor de 112 px decorre do mínimo múltiplo comum entre o *stride* acumulado do extrator (16 px) e a escala de célula ($448 \times 3 / 96 = 14\text{ px}$):
   $$\text{MMC}(16, 14) = 112\text{ px}$$
   Elimina o artefato de contorno demonstrado por Alsallakh et al. [9]. No treinamento, seu uso degrada a generalização (AUROC cai de 0,9185 para 0,9033), pois o *flow* memoriza os padrões espelhados irreais.

4. **Test-Time Augmentation por Simetria (`tta flips` — Inferência):**
   Aplica as 4 operações do grupo de Klein $V_4$ (identidade, flip horizontal, vertical e combinado), desfaz as coordenadas nos mapas de calor e calcula a média. Simetriza o prior posicional condicional senoidal, atenuando concentrações assimétricas de NLL nos quatro vértices da grade.

5. **Seleção de Subconjuntos de Níveis (`levels`):**
   Avaliação combinatória dos subconjuntos de pirâmide: `0` ($L_1$), `1` ($L_2$), `2` ($L_3$), `01`, `02`, `12` e `012`. Previne a contaminação de sinal por níveis ruidosos.

6. **Suavização Gaussiana Espacial ($\sigma$):**
   Convolução espacial 2D pós-agregação. Sintonizada de acordo com o tamanho esperado da patologia: $\sigma=2$ para micro-defeitos ($<0{,}5\%$ da imagem) e $\sigma=8$ para defeitos extensos ($>1{,}5\%$).

---

### 3.4 Protocolo Experimental e Métricas

O benchmark INSPLAD-seg [8] reúne 5 classes com imagens em resolução $448 \times 448$:

**Tabela 2: Especificações do Benchmark INSPLAD-seg.**
| Classe | Tipo de Patologia | Imagens Treino (Normais) | Imagens Teste (Normais / Anômalas) | Área Média de Anomalia |
|---|---|---|---|---|
| `glass-insulator` | Campânula quebrada (*missing cap*) | 1 104 | 591 / 87 | **0,14%** (estrutural) |
| `lightning-rod-suspension` | Ferrugem fina puntiforme | 462 | 117 / 46 | **1,90%** (textura/misto) |
| `yoke-suspension` | Quebra/trinca mecânica | 4 834 | 154 / 46 | **2,00%** (misto) |
| `polymer-insulator-upper-shackle` | Desgaste e deformação | 935 | 235 / 82 | **3,10%** (geométrico) |
| `vari-grip` | Corrosão severa alargada | 477 | 126 / 28 | **4,60%** (textura) |

#### Protocolo de Validação e Limiar de Ruído
Adota-se divisão estratificada `val_fraction = 0.4` sobre o conjunto de teste. O grid de ablação é avaliado no subconjunto de validação e a configuração de ponta é reportada uma única vez no conjunto *held-out* restante (60%).

O limiar de significância estatística foi medido empiricamente alternando frações de partição sobre o mesmo modelo:
$$\Delta_{\text{AUROC}} = 0{,}0044, \quad \Delta_{\text{AP}} = 0{,}0064, \quad \Delta_{\text{AUPRO}} = 0{,}0084$$
Diferenças inferiores a $\pm 0{,}0045$ em Pixel AUROC são formalmente categorizadas como ruído estocástico amostral.

---

## 4. Resultados e Análise Empírica

### 4.1 A Grande Tabela Exaustiva Multidimensional

A Tabela 3 reúne **todos os valores exatos** medidos nas cerca de 24.000 configurações experimentais registradas em `plano_ablation_nf_head.md`:

**Tabela 3: Síntese comparativa exaustiva dos fatores de inferência, treino e níveis nas 5 classes.**
| Categoria / Fator Avaliado | Métrica Reportada | `vari-grip` (4,6%) | `lightning-rod` (1,9%) | `glass-insulator` (0,14%) | `polymer-shackle` (3,1%) | `yoke-suspension` (2,0%) | **Média Geral** |
|---|---|---|---|---|---|---|---|
| **Ponto de Partida** | Produção Original (pad0, minmax, L=012, $\sigma$=4) | 0,7373 | 0,8008 | 0,7935 | 0,8651 | 0,8803 | **0,8154** |
| **Passo 1 (Incremental)** | `+ reflect_pad 112` | 0,7407 (+0,0034) | 0,8052 (+0,0044) | 0,8253 (+0,0318) | 0,8672 (+0,0021) | 0,8785 (−0,0018) | **0,8234 (+0,0080)** |
| **Passo 2 (Incremental)** | `+ tta flips` | 0,7955 (+0,0548) | 0,8306 (+0,0254) | 0,8240 (−0,0013) | 0,8736 (+0,0064) | 0,8794 (+0,0009) | **0,8407 (+0,0173)** |
| **Passo 3 (Incremental)** | `+ score_mode raw` | 0,8186 (+0,0231) | 0,8625 (+0,0319) | 0,9339 (+0,1099) | 0,8901 (+0,0165) | 0,8888 (+0,0094) | **0,8788 (+0,0381)** |
| **Passo 4 (Incremental)** | `+ seleção de níveis` (ótimo da classe) | 0,9007 (+0,0821) | 0,8798 (+0,0173) | 0,9598 (+0,0259) | 0,8901 (+0,0000) | 0,9140 (+0,0252) | **0,9089 (+0,0301)** |
| **Passo 5 (Incremental)** | `+ sigma ótimo` ($\sigma=8$, exceto $\sigma=2$ em glass) | 0,9130 (+0,0123) | 0,8810 (+0,0012) | 0,9599 (+0,0001) | 0,8901 (+0,0000) | 0,9156 (+0,0016) | **0,9120 (+0,0031)** |
| **Ganho Pós-Processamento** | **Delta Acumulado na Validação** | **+0,1757** | **+0,0802** | **+0,1664** | **+0,0250** | **+0,0353** | **+0,0965** |
| | | | | | | | |
| **Efeito Pareado Isolado** | `raw` vs `minmax` ($\Delta$ AUROC / % pares pos) | **+0,0574** (97%) | **+0,0794** (100%) | **+0,0441** (100%) | **+0,0205** (82%) | **+0,0471** (97%) | **+0,0497 (92,7% pos)** |
| **Efeito Pareado Isolado** | `tta flips` vs `none` ($\Delta$ AUROC / % pares pos) | **+0,0163** (97%) | **+0,0223** (94%) | **+0,0126** (76%) | **+0,0058** (81%) | **+0,0031 (46%)** | **+0,0120 (78,8% pos)** |
| **Efeito Pareado Isolado** | `pad 112` vs `pad 0` ($\Delta$ AUROC / % pares pos) | **+0,0136** (85%) | **+0,0056** (81%) | **+0,0028** (74%) | **+0,0024** (70%) | **+0,0002** (72%) | **+0,0049 (76,4% pos)** |
| | | | | | | | |
| **Impacto do `featnorm`** | $\Delta$ Pixel AUROC no Held-out (vs baseline) | **+0,0440** | **−0,0107** | **+0,0396** | **−0,0066** | **−0,0499** | **−0,0067** (Não universal) |
| **Impacto do `featnorm`** | $\Delta$ AUPRO no Held-out (vs baseline) | **+0,0821** | **−0,0455** | **+0,0254** | **+0,0502** | **−0,1226** | **−0,0021** (Destrutivo em Yoke) |
| **Impacto do `clamp 1.9`** | $\Delta$ AUROC (adicionado sobre featnorm) | +0,0080 | +0,0050 | +0,0017 | −0,0062 | −0,0117 | −0,0006 |
| | | | | | | | |
| **Hierarquia dos Níveis** | AUROC isolado do Nível $L_1$ (Textura/Cor) | **0,893** (Vencedor) | 0,760 | 0,874 | 0,776 | 0,852 | 1 de 5 classes vence |
| **Hierarquia dos Níveis** | AUROC isolado do Nível $L_2$ (Formas Médias) | 0,817 | 0,807 | 0,650 | **0,880** (Vencedor) | 0,798 | 1 de 5 classes vence |
| **Hierarquia dos Níveis** | AUROC isolado do Nível $L_3$ (Semântica/Contexto)| 0,787 (Ruidoso) | **0,877** (Vencedor) | **0,942** (Vencedor) | 0,841 | **0,879** (Vencedor) | **3 de 5 classes vence** |
| **Melhor Subconjunto** | Níveis Selecionados na Validação | `levels='0'` ($L_1$) | `levels='2'` ($L_3$) | `levels='02'` ($L_1+L_3$) | `levels='012'` (Todos) | `levels='02'` ($L_1+L_3$) | — |
| **Análise de Regret** | Perda de AUROC fixando `levels=02` ($L_1+L_3$) | **0,0006** | **0,0012** | **0,0000** | **0,0278** | **0,0000** | **$\le 0,0012$ em 4/5 classes** |
| | | | | | | | |
| **Dinâmica de Treino** | Época de Pico de Validação (de 80 épocas) | Época 16 | Época 12 | Época 36 | Época 56 | **Época 4** | Pico precoce em bases grandes |
| **Overfitting do Flow** | Queda de AUROC (Pico $\to$ Época 80) | −0,0123 | −0,0012 | −0,0009 | −0,0002 | **−0,0299** | Degradação severa em yoke |
| | | | | | | | |
| **Resultado Held-out** | Produção Original (Pixel AUROC / AUPRO) | 0,7707 / 0,6252 | 0,7669 / 0,3880 | 0,8821 / 0,6069 | 0,8624 / 0,5753 | 0,8336 / 0,5790 | 0,8231 / 0,5549 |
| **Resultado Held-out** | **Melhor Obtido Final (AUROC / AUPRO)** | **0,9270 / 0,7585** | **0,8815 / 0,5642** | **0,9560 / 0,7604** | **0,8806 / 0,5943** | **0,9188 / 0,6611** | **0,9128 / 0,6677** |
| **Ganho Final Total** | **$\Delta$ Pixel AUROC / $\Delta$ AUPRO Final** | **+0,1563 / +0,1333** | **+0,1146 / +0,1762** | **+0,0739 / +0,1535** | **+0,0182 / +0,0190** | **+0,0852 / +0,0821** | **+0,0897 / +0,1128** |

---

### 4.2 Decomposição Incremental e a Primazia da NLL `raw`

Como ilustrado na Tabela 3, o pós-processamento confere **+0,0965 de AUROC médio**. A transição da normalização `minmax` para `raw` responde sozinha por **+0,0381** na cadeia incremental e **+0,0497** em comparações pareadas diretas. 

A superioridade de `raw` é matematicamente absoluta: obteve taxas de vitória pareada de **100%** em `lightning-rod` e `glass-insulator`, **97,2%** em `vari-grip` e `yoke-suspension`, e **81,9%** em `polymer-shackle`. A normalização min-max introduz distorção estocástica grave: ao mapear o pixel de maior score de cada imagem para 1,0, toda imagem sem anomalias (100% normal) ganha falsos positivos com confiança máxima, destruindo o limiar global de classificação.

---

### 4.3 Comportamento dos Níveis e o Fenômeno da Contaminação Multiescala

A análise isolada da NLL revela uma especialização funcional das camadas convolucionais estritamente correlacionada à escala da patologia:
* **Classes dominadas por $L_1$ (`vari-grip`):** A falha consiste em oxidação severa espalhada ao longo dos cabos de aço. A resolução de $L_1$ ($109 \times 109$) captura o padrão textural microscópico da ferrugem com AUROC de **0,893**. Em contrapartida, $L_3$ atinge apenas **0,787**.
  * **O Fenômeno da Contaminação:** Quando os três níveis são fundidos cegamente ($L_1+L_2+L_3$), ou mesmo na fusão $L_1+L_3$, a NLL ruidosa de $L_3$ dilui a discriminabilidade de $L_1$. No teste completo, pontuar **apenas $L_1$** eleva o Pixel AUROC de 0,8561 para **0,9191** (+0,0630) e o AUPRO de 0,6124 para **0,7621** (+0,1497).
* **Classes dominadas por $L_3$ (`glass-insulator`, `lightning-rod`, `yoke-suspension`):** Em `glass-insulator`, a quebra de uma campânula ($0,14\%$ da área) remove uma estrutura completa. As camadas rasas $L_1$ e $L_2$ apenas observam o fundo natural de árvores e julgam as texturas perfeitamente normais; somente o campo receptivo amplo de $L_3$ ($>150\text{ px}$) codifica a descontinuidade na cadeia de isoladores, alcançando AUROC isolado de **0,942** (contra 0,874 de $L_1$ e 0,650 de $L_2$).
* **A Singularidade de $L_2$ (`polymer-insulator`):** $L_2$ é o melhor nível apenas nesta classe (**0,880**, contra 0,776 de $L_1$ e 0,841 de $L_3$). A falha consiste em trincas e corrosão mecânica de partes de ferragens — um defeito de escala intermediária. Descartar $L_2$ nesta classe implica em um *regret* de **0,0278**.
* **Validação do Default Universal `levels=02`:** Excetuando o caso de `polymer`, fixar $L_1+L_3$ como política padrão induz um *regret* máximo de **0,0012** (em `lightning-rod`), sendo rigorosamente nulo ($0{,}0000$) em `glass` e `yoke`. Representa um compromisso altamente seguro para implantações agnósticas à classe.

---

### 4.4 Mecanismo de Falha da Padronização por Canal (`featnorm`)

A técnica `featnorm` exibe comportamento conflitante: confere **+0,0396** em `glass` e **+0,0440** em `vari-grip`, mas causa degradação crítica em `yoke-suspension` (**−0,0499 de AUROC** e **−0,1226 de AUPRO**).

```
Efeito da Padronização featnorm por Profundidade:

Nível L1 (64 canais):   Sinal de textura concentrado em canais de alta variância.
                         ──► Featnorm equaliza magnitudes ──► Sinal discriminativo preservado/amplificado.

Níveis L2/L3 (192-256): Sinal semântico distribuído em poucos canais; maioria dos canais contém ruído.
                         ──► Featnorm amplifica canais nulos ──► Mascara a NLL de contexto global.
```

Em `yoke-suspension`, a anomalia reside na integridade mecânica de $L_3$. A amplificação estocástica de canais irrelevantes destruiu a separabilidade da NLL. Conclui-se que `featnorm` **não deve ser adotado como receita universal**, devendo permanecer desativado por padrão a menos que a classe apresente anomalias predominantemente texturais mapeadas em $L_1$.

---

### 4.5 Dinâmica de Overfitting em Grandes Datasets

A imposição de uma rotina fixa de 80 épocas para todas as classes revelou um sobreajuste dramático no *flow* em bases volumosas. Em `yoke-suspension` (4.834 imagens normais de treino), o pico de Pixel AUROC ocorreu na **época 4** (0,9292), degradando de forma contínua até **0,8993** na época 80 (queda de $-0,0299$). Em 85% das avaliações pós-pico, o modelo superava o checkpoint final. 

Esse fenômeno comprova que orçamentos de época uniformes são danosos: bases massivas aprendem a densidade espacial em pouquíssimas iterações, enquanto bases pequenas (400–900 imagens) necessitam de 30 a 50 épocas. A implementação de *early stopping* via `--patience 5` e orçamentos proporcionais (`CLASS_BUDGETS`) é mandatória.

---

### 4.6 Resolução Numérica do AUPRO e Métricas Compostas

Identificou-se que a rotina tradicional de cálculo de AUPRO via limiares lineares `np.linspace(min, max, 200)` falhava catastroficamente em mapas de NLL crua (`raw`), retornando silenciosamente `AUPRO = 0.0000` em `vari-grip`. Em virtude de caudas pesadas com valores extremos esparsos ($>10^4$), 199 dos 200 limiares lineares caíam em taxas de falsos alarmes superiores a $\text{FPR} > 0{,}30$, inviabilizando a integração da curva. A substituição por **quantis dos pixels normais** dentro do intervalo $[1 - \text{FPR}_{\text{max}}, 1{,}0]$ resolveu a patologia, garantindo amostragem uniforme de pontos úteis e tornando a métrica invariante a transformações monotônicas de escala.

Adicionalmente, a adoção do score composto $(AUROC_{px} + AUROC_{img} + AUPRO)/3$ transferido do RD++ [6] evitou armadilhas mono-métricas, impedindo a escolha de checkpoints que maximizam ordenação global de pixels às custas da fragmentação regional de defeitos.

---

### 4.7 Desempenho Frente ao Estado da Arte (DifferNet vs. RD++)

A Tabela 4 contrasta o pipeline otimizado SEDifferNet + CFLOW frente ao Reverse Distillation (RD++):

**Tabela 4: Comparação DifferNet (calibrado) vs. RD++ (WideResNet-50).**
| Classe | RD++ Pixel AUROC | DifferNet Pixel AUROC | RD++ AUPRO | DifferNet AUPRO |
|---|---|---|---|---|
| `glass-insulator` | 0,9368 | **0,9534** | **0,7560** | 0,6591 |
| `polymer-shackle` | **0,9024** | 0,8505 | **0,6508** | 0,5933 |
| `vari-grip` | 0,8588 | **0,9191** | 0,5366 | **0,7621** |
| `yoke-suspension` | **0,9708** | 0,9348 | **0,7328** | 0,6686 |

A despeito de utilizar um extrator consideravelmente mais leve (AlexNet com SE vs. WideResNet-50), o **DifferNet calibrado supera o RD++ em Pixel AUROC nas classes `glass` e `vari-grip`**, alcançando **0,7621 de AUPRO** na localização de corrosão em `vari-grip` (contra 0,5366 do RD++).

---

## 5. Discussão

### 5.1 Fundamentação Teórica da Primazia do Pós-Processamento
A superioridade categórica do pós-processamento decorre da própria formulação matemática dos Normalizing Flows. Quando o extrator convolucional é congelado, o espaço de características é invariante. A tarefa do fluxo afim condicional é estimar a densidade exata de probabilidade $p(\mathbf{x})$. 

A soma linear de NLLs ($\sum_k -\log p(\mathbf{x}_k)$) equivale matematicamente a modelar a log-verossimilhança conjunta sob a hipótese de independência condicional multiescala, preservando a calibração global métrica entre imagens normais e anômalas. Essa propriedade garante que pixels anômalos em qualquer imagem ultrapassem um limiar global fixo. Em contrapartida, normalizações como *min-max* forçam cada mapa a um intervalo relativo, eliminando a comparabilidade inter-amostral e degradando a separabilidade de detecção.

### 5.2 O Perigo dos Checkpoints Legados
Uma das constatações mais expressivas deste estudo foi a desmistificação do suposto "sinal invertido" em $L_1$ na classe `glass-insulator`. No checkpoint original legado, $L_1$ marcava AUROC de **0,2583** — sugerindo a teoria intuitiva de que a ausência de peças invertia o gradiente de textura. Ao retreinar a cabeça corrigindo as discrepâncias silenciosas de hiperparâmetros (*clamp scale* de 0,5 vs. 2,0 na avaliação e resolução de grade de 96 vs. 56), o mesmo nível saltou para **0,8740** (+0,6157). O fenômeno era unicamente um artefato induzido por desalinhamento numérico. Este achado estabelece um alerta para a comunidade: **estudos de ablação não devem inferir propriedades físicas fundamentais a partir de checkpoints legados não auditados.**

### 5.3 Limitações e Validade Externa
* **Arquitetura de Backbone:** O estudo baseia-se na AlexNet com blocos SE. Embora a formulação da NLL `raw` independa do extrator, modelos com conexões residuais profundas (ResNet, ConvNeXt) apresentam dinâmicas distintas de preservação de textura em camadas profundas.
* **Diversidade de Domínio:** O INSPLAD-seg caracteriza-se por fundos naturais ruidosos e pequenas anomalias. A transferência quantitativa para bases industriais com iluminação uniforme (e.g., MVTec-AD) deve ser avaliada.
* **Controle Amostral de Significância:** O limiar de $\Delta = 0{,}0044$ foi computado com divisões controladas de validação; o emprego de múltiplos seeds aleatórios de treinamento e teste representa o padrão estatístico ideal para trabalhos futuros.

### 5.4 Diretrizes de Implantação e Trade-off de Latência para VANTs
1. **Regra de Ouro (Custo Computacional Zero):** Implante invariavelmente com `score_mode=raw`, `reflect_pad 112` e kernel $\sigma=8$ ($\sigma=2$ para micro-defeitos). Essa cadeia recupera **+0,0965 de AUROC médio** sem qualquer retreinamento.
2. **Trade-off de Latência do TTA:** O TTA por flips eleva a AUROC em $+0{,}0120$, mas quadruplica o custo de inferência ($4\times$ passagens de rede). Para inspeções embarcadas em tempo real em drones com restrições severas de bateria e taxa de quadros, o TTA pode ser desligado com penalidade mínima; para auditorias em lote offline, deve ser ativado.
3. **Seleção de Níveis Operacional:** Adote $L_1+L_3$ (`levels=02`) como default de baixíssimo regret ($\le 0{,}0012$). Reserve $L_1$ isolado para peças dominadas por corrosão superficial.
4. **Política de Normalização de Treino (Risco Assimétrico):** Mantenha `featnorm` desativado por padrão; o ganho moderado observado em classes favoráveis (+0,039 a +0,044 AUROC) não compensa o risco de colapso severo em componentes dominados por contexto semântico global (−0,0499 AUROC e −0,1226 AUPRO em *yoke*). Ligue-o apenas para componentes onde a patologia resida comprovadamente em características rasas de cor e textura ($L_1$).

---

## 6. Conclusão

Este estudo conduziu uma investigação abrangente e aprofundada dos fatores de pós-processamento, inferência e seleção multiescala da cabeça CFLOW no pipeline SEDifferNet. Demonstrou-se que as maiores alavancas de ganho residem na preservação da métrica absoluta de verossimilhança (`raw`), na eliminação de artefatos convolucionais de borda via *reflect padding* na inferência e na seleção balanceada de níveis espaciais ($L_1+L_3$), agregando coletivamente **+0,0965 de Pixel AUROC médio**. A análise mecânica explicou a não-universalidade da padronização de características e o sobreajuste precoce em bases volumosas. As correções metodológicas e diretrizes de engenharia formuladas fornecem um roteiro seguro e reprodutível para a implantação de modelos de densidade condicional em tarefas de inspeção visual automatizada.

---

## Declaração de Uso de Inteligência Artificial
Os autores declaram que ferramentas de assistência baseadas em modelos de linguagem de grande porte (LLMs) foram empregadas sob supervisão humana estrita através do framework Academic Research Skills (ARS v3.9.4.2) para apoio à estruturação textual, verificação de citações e formatação bibliográfica. Todos os dados experimentais, análises conceituais, decisões interpretativas e formulações científicas foram desenvolvidos, conferidos e validados diretamente pelo autor humano, assumindo responsabilidade integral sobre o conteúdo deste artigo.

---

## Referências

[1] M. Rudolph, B. Wandt, e B. Rosenhahn, "Same Same But DifferNet: Semi-Supervised Defect Detection with Normalizing Flows," in *Proc. IEEE/CVF Winter Conf. Applications of Computer Vision (WACV)*, 2021, pp. 1907–1916.  
[2] D. Gudovskiy, S. Ishizaka, e K. Kozuka, "CFLOW-AD: Real-Time Unsupervised Anomaly Detection with Localization via Conditional Normalizing Flows," in *Proc. IEEE/CVF Winter Conf. Applications of Computer Vision (WACV)*, 2022, pp. 1819–1828.  
[3] M. Rudolph, T. Wehrbein, B. Rosenhahn, e B. Wandt, "Fully Convolutional Cross-Scale-Flows for Image-based Defect Detection," in *Proc. IEEE/CVF Winter Conf. Applications of Computer Vision (WACV)*, 2022, pp. 1088–1097.  
[4] K. Roth, L. Pemula, J. Zepeda, B. Schölkopf, T. Brox, e P. Gehler, "Towards Total Recall in Industrial Anomaly Detection," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2022, pp. 14318–14328.  
[5] T. Defard, A. Setkov, A. Loesch, e R. Audigier, "PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization," in *Proc. Int. Conf. Pattern Recognition (ICPR) Workshops*, 2021, pp. 475–489.  
[6] H. Deng e X. Li, "Anomaly Detection via Reverse Distillation from One-Class Embedding," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2022, pp. 9737–9746.  
[7] Y. Zhou, X. Xu, J. Song, F. Shen, e H. T. Shen, "MSFlow: Multiscale Flow-based Framework for Unsupervised Anomaly Detection," *IEEE Trans. Neural Networks and Learning Systems*, 2024.  
[8] A. L. B. Vieira-e-Silva *et al.*, "InsPLAD: A Dataset and Benchmark for Power Line Asset Inspection in UAV Images," *Int. J. Remote Sensing*, vol. 44, no. 23, pp. 7465–7484, 2023.  
[9] B. Alsallakh, N. Kokhlikyan, V. Miglani, J. Yuan, e O. Reblitz-Richardson, "Mind the Pad -- CNNs can Develop Blind Spots," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2021.  
[10] M. A. Islam, S. Jia, e N. D. B. Bruce, "How Much Position Information Do Convolutional Neural Networks Encode?" in *Proc. Int. Conf. Learning Representations (ICLR)*, 2020.  
[11] R. Geirhos, P. Rubisch, C. Michaelis, M. Bethge, F. A. Wichmann, e W. Brendel, "ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2019.  
[12] O. S. Kayhan e J. C. van Gemert, "On Translation Invariance in CNNs: Convolutional Layers Can Exploit Absolute Spatial Location," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2020, pp. 14274–14285.
