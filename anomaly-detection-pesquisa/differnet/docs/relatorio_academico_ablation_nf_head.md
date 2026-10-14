# Ablação da Cabeça CFLOW sobre SEDifferNet para Localização de Anomalias em Inspeção Aérea de Linhas de Transmissão

**Relatório acadêmico de métodos, resultados e comparação com DifferNet, AttentDifferNet e CFLOW-AD**  
**Versão:** 1.0 · 7 de outubro de 2026  
**Dados principais:** INSPLAD-seg · cinco classes · heads CFLOW sobre extratores AlexNet-SE congelados

> **⚠️ Atualização 2026-10-13.** Os números deste relatório (versão 1.0) vêm do protocolo de setembro. Para resultados com seleção limpa nas 5 classes, tabela de ablação, receitas por classe e receita geral, use a [análise científica consolidada](analise_cientifica_ablacao_nf_head_5classes.md). A seção 6 daquele documento lista, afirmação por afirmação, o que foi confirmado, reduzido ou refutado.

## Resumo

Este relatório documenta o estudo de ablação da cabeça de *normalizing flow* condicional (CFLOW) usado para localizar defeitos em imagens aéreas de componentes de linhas de transmissão. O estudo mantém congelado o extrator por classe, denominado `SEDifferNet` no código deste projeto, e varia decisões de treinamento e inferência da cabeça pixel-level. O protocolo compreendeu cinco classes do INSPLAD-seg, 27 receitas de treinamento e aproximadamente 24 mil avaliações/configurações registradas nos experimentos de ablação. A contribuição mais consistente veio da inferência: a cadeia incremental de *reflect padding* (112 px), TTA por flips, agregação NLL `raw`, seleção de níveis e suavização elevou o Pixel AUROC médio em 0,0965 na validação do grid; o efeito pareado de `raw` frente a min-max foi +0,0497. A hierarquia de níveis variou por classe e por checkpoint. Em uma head recém-treinada, L1 isolado elevou `vari-grip` de 0,8561/0,6124 para 0,9191/0,7621 (Pixel AUROC/AUPRO), enquanto em outras classes L3 ou a fusão L1+L3 foi preferível. O trabalho também identificou uma falha numérica no AUPRO com mapas de cauda pesada e incorporou histórico por época, score composto e parada antecipada ao fluxo de treinamento.

O relatório distingue os artigos originais de três objetos que não devem ser confundidos: DifferNet, detector image-level; AttentDifferNet (SENet/CBAM), extensão image-level publicada em WACV 2024; e a classe de código `SEDifferNet`, variante local que fornece as features ao novo head CFLOW pixel-level. A comparação é técnica e contextual; os números de benchmarks e splits diferentes não são tratados como uma disputa direta de estado da arte. Resultados obtidos após selecionar níveis no conjunto de teste completo são identificados como exploratórios, não como estimativas independentes de generalização.

**Palavras-chave:** detecção não supervisionada de anomalias; localização por pixel; normalizing flows; CFLOW-AD; AttentDifferNet; INSPLAD-seg; ablação; validação de modelos.

Uma avaliação pareada adicional do caminho oficial de score/upsampling do CFLOW-AD sobre as cinco heads locais obteve média macro de 0,8268 Pixel AUROC e 0,0337 AP, contra 0,8130 e 0,0224 para `raw` (+0,0138 AUROC e +0,0113 AP). O teste usa a coorte completa por classe, é transdutivo e reutiliza as mesmas NLLs; como `cflow_original` usa `align_corners=True` e os modos locais usam `False`, não isola apenas a fórmula de agregação.

## 1. Escopo e pergunta de pesquisa

O problema experimental é localizar regiões anômalas em imagens de inspeção UAV quando há imagens normais disponíveis para treinamento, mas os defeitos são raros. A pergunta central é: **mantendo o extrator visual fixo, quais decisões da cabeça pixel-level e de sua inferência melhoram a localização, e em que condições tais efeitos dependem da classe ou do checkpoint?**

O estudo não compara uma nova arquitetura ponta a ponta contra os artigos originais em condições idênticas. Ele audita e melhora uma implementação específica, com AlexNet e atenção SE, seguida por uma cabeça CFLOW treinada sobre features espaciais. A maior parte das hipóteses testadas é local à implementação e ao INSPLAD-seg. Conclusões sobre outros backbones, datasets ou sistemas embarcados permanecem hipóteses até serem replicadas.

### 1.1 Nomenclatura e atribuição bibliográfica

Há três nomes próximos, mas distintos:

| Nome neste relatório | Significado preciso | Nível de saída |
|---|---|---|
| **DifferNet** | Artigo original de Rudolph, Wandt e Rosenhahn (WACV 2021): flow sobre descritores globais multiescala de uma CNN pré-treinada. | Image-level; mapas interpretativos por gradiente da NLL. |
| **AttentDifferNet** | Nome usado pelo artigo *Attention Modules Improve Image-Level Anomaly Detection for Industrial Inspection: A DifferNet Case Study* (WACV 2024). Integra blocos SENet ou CBAM ao extrator AlexNet do DifferNet. | Image-level; rótulos de treino do artigo são image-level. |
| **`SEDifferNet`** | Nome da classe no código deste repositório (`core/model.py`): AlexNet e blocos SE (`simsa1`, `simsa2`, `simsa4`) no caminho de *forward*, com flow image-level. Neste estudo, os pesos desse extrator ficam congelados. | O modelo-base é image-level; seus mapas intermediários alimentam um novo head pixel-level. |
| **CFLOW pixel head deste projeto** | Três flows condicionais que recebem mapas espaciais intermediários do `SEDifferNet`, antes do *global average pooling*. | Mapa de anomalia por pixel. |

O artigo de 2024 fornecido pelo autor é sobre **AttentDifferNet**, não apresenta uma publicação com o nome literal `SEDifferNet`. Neste relatório, `SEDifferNet` é a denominação do artefato de código empregado. O artigo de AttentDifferNet é citado como antecedente científico para a inserção de atenção no DifferNet, não como prova de que a implementação local reproduz todos os detalhes daquela publicação. O próprio repositório declara que a classe usa SE no `forward` e mantém submódulos CBAM declarados por compatibilidade; portanto a presença de pesos/chaves CBAM não significa que eles contribuíram para o caminho treinado.

## 2. Relação com os trabalhos originais

### 2.1 DifferNet: densidade de descritores e score por imagem

O DifferNet de Rudolph et al. [1] extrai representações de uma AlexNet ImageNet pré-treinada e congelada em três escalas de entrada (448, 224 e 112 px). O *global average pooling* converte cada mapa em um vetor; os três vetores são concatenados (768 dimensões na configuração do artigo) e processados por um flow Real-NVP de oito blocos de acoplamento [6]. A função de score combina NLLs sobre transformações da imagem. Para interpretação espacial, o DifferNet propaga o gradiente da NLL até a imagem e suaviza os gradientes; isso não equivale a estimar uma densidade condicional em cada posição.

O head desta pesquisa muda precisamente essa unidade de modelagem: em vez de aplicar apenas um flow ao vetor global obtido por pooling, mantém a grade espacial das features, seleciona níveis L1/L2/L3 e estima NLL condicionada à posição para cada célula. O backbone image-level permanece congelado, mas o head CFLOW é um componente novo, treinado para densidade local. Assim, a localização por pixel reportada aqui não é a saída nativa do artigo DifferNet.

### 2.2 AttentDifferNet: atenção no extrator, ainda image-level

AttentDifferNet de Vieira-e-Silva et al. [2] adiciona três blocos de atenção em pontos sucessivos do AlexNet. O artigo estuda versões com SENet [4], que recalibra canais via *squeeze*, MLP de *excitation* e escala sigmoide, e com CBAM [5], que combina atenção de canal e espacial. O artigo reporta que os blocos são treinados a partir do zero enquanto os pesos convolucionais são inicializados com AlexNet pré-treinada. Avalia o AUROC de imagem e usa Grad-CAM para visualização; informa explicitamente que as máscaras pixel-level não são usadas no treinamento. No InsPLAD-fault, reporta 94,34% de AUROC médio para AttentDifferNet-SENet e 92,46% para DifferNet; esses são resultados image-level daquela versão do dataset e não são comparáveis aos Pixel AUROC/AUPRO deste relatório.

A implementação local chamada `SEDifferNet` tem parentesco conceitual com a variante SENet: AlexNet, atenção SE inserida em estágios do extrator e flow image-level. O head CFLOW deste estudo, entretanto, lê as ativações espaciais antes do pooling global e aprende uma distribuição por posição. Logo, atenção e CFLOW cumprem papéis diferentes: SE repondera canais do backbone; CFLOW aprende se um vetor de features local é provável para sua posição. Um bloco SE, isoladamente, não é segmentador nem fornece supervisão de máscara.

### 2.3 CFLOW-AD: localização por flows condicionais multiescala

CFLOW-AD de Gudovskiy et al. [3] foi proposto para localização não supervisionada. Usa encoder discriminativo pré-treinado, pirâmide de features multiescala e decoders de flow que estimam likelihoods por vetor de feature, condicionando as transformações a uma codificação posicional senoidal 2D. O código oficial usa WideResNet-50-2 por default, oito coupling blocks e dimensão condicional 128; o paper também avalia ResNet-18 e MobileNetV3. Para cada escala $k$, divide o log-likelihood pelo número de canais, subtrai o máximo desse valor ao longo de toda a coorte e das posições, e exponencia: $P_{ik}(u,v)=\exp(\ell_{ik}(u,v)-\max_{j,a,b}\ell_{jk}(a,b))$. Depois do upsample, soma $M_i(u,v)=\sum_kP_{ik}(u,v)$ e calcula $S_i(u,v)=\max_{j,a,b}M_j(a,b)-M_i(u,v)$; os dois máximos usam a coorte inteira [3]. O artigo compara ainda dois/três níveis de pooling e fluxos condicionais/incondicionais.

O projeto reaproveita a ideia de densidade condicional por posição e por escala, mas altera decisões concretas: usa features AlexNet-SE (64/192/256 canais), redimensiona os níveis para grade comum 96×96, usa PE de dimensão 64, clamp local padrão 0,5 e permite escolher níveis. `per_level_prob` é uma aproximação local limitada: o código calcula $-\sum_k\exp(\operatorname{logsigmoid}(-\operatorname{NLL}_k)/C_k)$, não normaliza cada escala pelo máximo cohort-wide e não inverte pelo máximo global de $M$. `raw` é outra alternativa, escolhida empiricamente nesta ablação; `per_level_minmax` é uma convenção histórica local, não o score definido pelo artigo CFLOW-AD.

### 2.4 Paridade com CFLOW-AD: o que ainda não foi reproduzido

Aplicar o caminho oficial de score/upsampling às mesmas NLLs não converte o experimento em uma reprodução integral do CFLOW-AD. O head/backbone e as NLLs ficam controlados, mas `align_corners` difere entre `cflow_original` e os modos locais; as demais diferenças continuam abertas:

| Componente CFLOW-AD | Configuração de referência | Configuração deste projeto | Status de comparação |
|---|---|---|---|
| **Agregação de inferência** | $\ell_{ik}=\log p_{ik}/C_k$; $P_{ik}=\exp(\ell_{ik}-\max_{j,u,v}\ell_{jk}(u,v))$; $M_i=\sum_kP_{ik}$; $S_i=\max_{j,u,v}M_j(u,v)-M_i$ [3] | `cflow_original`; máximos por escala e do mapa somado calculados sobre a coorte inteira | **Caminho score/upsampling testado** nas mesmas heads/NLLs; transdutivo e com `align_corners` distinto dos modos locais. |
| **Encoder e features** | WideResNet-50-2 default; pirâmide de pooling do encoder; ResNet-18 e MobileNetV3 também avaliados [3] | AlexNet-SE congelada; ativações L1/L2/L3 em blocos `simsa`, redimensionadas a 96² | Não testado como troca de backbone; a comparação atual isola apenas o agregador. |
| **Níveis e dimensão condicional** | Três escalas na referência; `condition_vec=128` [3] | Níveis locais selecionáveis; `cond_dim=64` | PE=128 não foi comparada; a escolha de níveis testa níveis locais, não os mesmos pooling outputs da referência. |
| **Blocos e rede de acoplamento** | Oito `FrEIA AllInOneBlock`, `affine_clamping=1.9`, affine global Softplus, permutação soft e sub-rede dependente de $C_k$ [3, repositório oficial] | Oito blocos customizados, hidden=256, clamp $\tanh(s)c$ e rotação cíclica fixa; `1.9` foi testado em algumas receitas | Paridade estrutural não testada; compartilhar oito blocos/clamp nominal não reproduz o decoder original. |
| **Loss e amostragem de features** | Log-likelihood normalizada por canais, loss `-LogSigmoid(log_prob)`, vetores espaciais amostrados em fibras/minibatches [3, repositório oficial] | Média de NLL por célula sobre os mapas de cada nível, sem loss `LogSigmoid` nem amostragem de fibras | Não testado como objetivo de treinamento; `cflow_original` só reagrupa NLLs já aprendidas. |
| **Upsampling** | Interpolação bilinear com `align_corners=True` no repositório oficial | `cflow_original` usa `align_corners=True`; os demais modos usam `_up` com `False` | Alinhado para o score original; os mapas de outros modos não usam exatamente a convenção oficial. |
| **Pré-processamento/TTA** | Escala/pooling e rotações de treino em MVTec/STC conforme configuração publicada [3] | Imagens 448² do INSPLAD-seg, reflect pad 112 e flips horizontais/verticais na inferência | Não reproduzido; pad/TTA local são variáveis adicionais, mantidas iguais entre os modos de score comparados. |
| **Treino e dados** | WRN50-2 default, input 256 (variável por classe nos resultados), código default 25 meta-epochs × 8 subepochs, warmup 2 [3] | AlexNet-SE, INSPLAD-seg a 448, Adam $2\times10^{-4}$, rotina local de NLL e agendas por classe | Não reproduzido; dataset, resolução, features, amostragem e loss diferem. |

**Interpretação e resultado do caminho de score:** o experimento responde como o caminho oficial de score/upsampling se comporta sobre NLLs das heads AlexNet-SE já treinadas, não qual seria o resultado do CFLOW-AD completo treinado conforme o paper. No sweep de 2026-10-07, `cflow_original` obteve média macro superior a `raw` (+0,0138 AUROC; +0,0113 AP), com AUROC maior em quatro de cinco classes e AP maior nas cinco. A avaliação é transdutiva, pois os máximos são estimados sobre toda a coorte de teste de cada classe. Como o modo oficial usa `align_corners=True` e os modos locais usam `False`, o teste não isola a fórmula da interpolação. Uma reprodução end-to-end ainda exige encoder, pooling, PE, decoder, loss/amostragem, resolução e protocolo de treino alinhados à referência.

### 2.5 Resumo comparativo

| Propriedade | DifferNet (2021) [1] | AttentDifferNet (2024) [2] | CFLOW-AD (2022) [3] | Experimento deste relatório |
|---|---|---|---|---|
| Objetivo central do artigo | Detecção de defeitos por imagem | Melhorar detecção por imagem com atenção | Detecção e localização por pixel | Ablação de um head pixel CFLOW sobre encoder local congelado |
| Representação pontuada | Vetor global concat. em três escalas de entrada | Vetor global DifferNet com atenção | Features espaciais multiescala | Features espaciais L1/L2/L3 do AlexNet-SE |
| Extrator | AlexNet ImageNet congelada | AlexNet modificada com três blocos de atenção | ResNet-18/WRN50/MobileNetV3 nos experimentos | AlexNet-SE local, congelada ao treinar CFLOW |
| Flow | Um flow sobre descritor global | Flow image-level herdado de DifferNet | Flows condicionais por escala/posição | Flow separado por nível, com PE 2D e mapa NLL |
| Localização nativa | Gradiente da NLL no input | Visualização Grad-CAM; não treinado com máscara pixel-level | NLL condicional gera mapa | NLL por célula, seleção de níveis, upsample e suavização |
| Agregação de escalas | Média de scores de transforms e descritores globais | Herdada de DifferNet | $\max M- M$ sobre mapas de probabilidade limitados | `raw`, min-max local, `std`, `per_level_prob` local ou `cflow_original` |
| Dados/métricas principais | MVTec AD e Magnetic Tile; AUROC imagem | InsPLAD-fault, MVTec AD, Semiconductor Wafer; AUROC imagem | MVTec AD e ShanghaiTech; métricas imagem e pixel | INSPLAD-seg; Pixel AUROC/AP/AUPRO e diagnóstico image-map |

As pontuações publicadas nesses trabalhos não são comparáveis diretamente às desta ablação: diferem em tarefa (imagem versus pixel), versão do dataset, divisão, backbone, resolução, limiar e pós-processamento.

## 3. Formulação do pipeline estudado

### 3.1 Extração de mapas e fluxo condicional

Uma imagem RGB $x \in \mathbb{R}^{3\times448\times448}$ é enviada ao backbone AlexNet-SE congelado. Pelas operações e pontos de extração implementados localmente, os mapas sem padding têm dimensões L1 (64 canais, 111×111), L2 (192 canais, 55×55) e L3 (256 canais, 27×27). Cada mapa é interpolado para a grade comum $96\times96$ (parâmetro `out_size=96`). O padding de inferência, quando ativo, é aplicado antes do backbone e a região central é recortada mantendo o alinhamento espacial.

Para cada nível $k$, um flow condicional estima a densidade da feature $f^k_{uv}$ condicionada pela posição $p_{uv}$:

$$z^k_{uv}=g_{\theta_k}(f^k_{uv};p_{uv}), \qquad
\operatorname{NLL}_k(u,v)=-\log p(f^k_{uv}\mid p_{uv}).$$

O flow usa coupling blocks afins. Na implementação local, $x_2$ e a PE condicionam a sub-rede; $x_1$ é transformado, $x_2$ permanece e uma permutação fixa é aplicada entre blocos:

$$y_1=x_1\odot\exp(s)+t,\qquad y_2=x_2,\qquad
s=\tanh(\hat{s})\,c_{\mathrm{clamp}}.$$

Cada NLL map de 96×96 é upsampleado para 448×448. O score final seleciona um subconjunto de níveis, agrega seus mapas, e aplica filtro Gaussiano opcional. As máscaras não entram no cálculo da loss de treinamento do flow; servem para avaliação pixel-level.

### 3.2 O que a NLL `raw` significa (e o que não significa)

A agregação local `raw` é:

$$S_{\mathrm{raw}}(u,v)=\sum_{k\in\mathcal{K}}\operatorname{NLL}_k(u,v).$$

Ela preserva a escala produzida pelos flows, ao contrário do min-max por imagem. Se as variáveis de nível fossem condicionalmente independentes dado o ponto, a soma poderia ser interpretada como NLL conjunta. Porém, os níveis provêm do mesmo backbone e não se demonstrou independência estatística entre eles. Neste estudo, `raw` deve ser entendido como **regra de agregação empiricamente superior no benchmark**, não como prova de que os mapas somados formam uma densidade conjunta calibrada.

As alternativas implementadas foram:

- **`per_level_minmax`:** para cada imagem e nível, subtrai o mínimo e divide pela amplitude do próprio mapa; em seguida soma. Preserva ordenação intramapa, mas remove a escala comparável entre imagens e níveis.
- **`per_level_std`:** padroniza cada nível por média e desvio escalar estimados em dados normais de treino; evita uma normalização dependente da imagem avaliada.
- **`per_level_prob`:** aproximação limitada implementada localmente, $S(u,v)=-\sum_k\exp(\operatorname{logsigmoid}(-\operatorname{NLL}_k)/C_k)$. Como o score é negativo, valores mais próximos de zero representam menor likelihood e maior anomalia. Ela não é idêntica ao score original CFLOW-AD: omite a normalização por máximo de log-likelihood em cada escala sobre a coorte e a inversão pelo máximo global do mapa somado [3]. A variante local foi testada; não foi vencedora nas condições avaliadas.

#### Score CFLOW-AD original versus modos deste projeto

A resposta direta é: **o CFLOW-AD original não define `raw` nem `per_level_minmax` como score final**. Ele normaliza o log-likelihood por canal, subtrai por escala o máximo observado na coorte de avaliação, exponencia e faz upsample dos mapas $P_{ik}$. Depois soma $M_i=\sum_kP_{ik}$ e calcula $S_i=\max_{j,u,v}M_j(u,v)-M_i$; esta é a regra descrita no artigo e executada pelo repositório oficial [3].

| Nome | Regra de score | Proveniência |
|---|---|---|
| **CFLOW-AD original** | $P_{ik}=\exp(\ell_{ik}-\max_{j,u,v}\ell_{jk})$, $M_i=\sum_kP_{ik}$, $S_i=\max_{j,u,v}M_j-M_i$ | Normalização por escala e inversão do mapa somado sobre a coorte, conforme artigo/repositório de Gudovskiy et al. [3]. |
| **`raw`** | $S=\sum_k\operatorname{NLL}_k$ | Alternativa deste estudo; foi a melhor entre as opções locais avaliadas, mas não é o score original CFLOW-AD. |
| **`per_level_minmax`** | Min-max independente por imagem e nível, seguido de soma | Convenção legada da implementação local; não foi proposta pelo artigo CFLOW-AD. |
| **`per_level_prob`** | $S=-\sum_k\exp(\operatorname{logsigmoid}(-\operatorname{NLL}_k)/C_k)$ | Aproximação local: usa logsigmoid, mas omite o máximo de log-likelihood por escala na coorte e a inversão pelo máximo global de $M$. |
| **`per_level_std`** | NLL padronizada por estatísticas escalares de treino | Outra alternativa local, não a fórmula publicada pelo CFLOW-AD. |

Consequentemente, o ganho pareado de +0,0497 reportado neste estudo significa que `raw` superou **o min-max legado local**; não demonstra superioridade de `raw` sobre o agregador canônico do CFLOW-AD, pois essa comparação não foi implementada exatamente.

## 4. Por que cada técnica e parâmetro foi incluído

A seleção distingue **fundamento do método**, **motivação experimental** e **resultado medido**. Parâmetros mantidos fixos por custo ou compatibilidade não são apresentados como ótimos universais.

| Técnica/parâmetro | Valores no estudo e papel | Motivação científica/operacional | Evidência ou decisão |
|---|---|---|---|
| **Extrator congelado** | AlexNet-SE final model por classe | Isola o efeito do head e reduz custo; mantém comparação com o detector image-level já disponível | Restrição experimental. Nenhuma claim de que AlexNet seja o melhor backbone pixel-level. |
| **Resolução de entrada** | 448×448 | Resolução do pipeline INSPLAD-seg e dos checkpoints de classe | Mantida fixa; resolução não foi objeto do grid principal. |
| **Níveis L1/L2/L3** | IDs `0/1/2`; subconjuntos `0`, `1`, `2`, `01`, `02`, `12`, `012` | Testa se a fusão multiescala ajuda ou se a NLL de um nível ruidoso dilui o sinal do nível informativo | Os vencedores variam por classe e checkpoint; evidência contra concatenar níveis cegamente. |
| **Grade CFLOW `out_size`** | 96×96 | Define a grade espacial e o suporte da PE; precisa ser idêntica em treino e avaliação | Default de código, não ablado. Descasamento histórico com grade 56 alterava a função sem erro de shape. |
| **Dimensão PE** | 64 | Codifica coordenadas $(u,v)$ para que a distribuição possa depender da posição | Fixa do projeto; não ablada. CFLOW-AD usou dimensão condicional 128 nas experiências de referência. |
| **Flow: blocos/hidden** | 8 coupling blocks; hidden=256 | 8 blocos seguem configuração consolidada de flow; dimensão interna controla capacidade e custo | Mantidos fixos, não comparar como hiperparâmetros vencedores. DifferNet original usa oito blocos também, mas sobre vetor global e com outras sub-redes. |
| **Clamp** | `0.5` baseline; `1.9` variante CFLOW-like | Limita magnitude da escala afim e estabiliza/condiciona as transformações invertíveis [1,3] | `0.5` é o valor gravado nos checkpoints locais. `1.9` foi avaliado sobretudo combinado com featnorm; não generalizou como ganho. Não é numericamente equivalente ao soft-clamp arctan com $\alpha=3$ do DifferNet. |
| **Agregação de score** | `raw`, `per_level_minmax`, `per_level_std`, `per_level_prob` | Separa a qualidade da densidade local da convenção que transforma likelihoods em mapa | `raw` ganha +0,0497 AUROC pareado sobre min-max. `per_level_prob` aproxima o score CFLOW-AD; min-max era convenção histórica local, não a fórmula do artigo CFLOW. |
| **Reflect padding** | 0 ou 112 px; modo reflect | Substitui zeros artificiais junto à borda por conteúdo espelhado [11–13] e busca manter o registro mapa-máscara | +0,0049 AUROC pareado em média; pad usado na inferência. Treinar com pad foi inferior em uma comparação específica. Valor 112 satisfaz alinhamento de stride 16 e célula de grade 14 px: $\operatorname{lcm}(16,14)=112$. |
| **TTA por flips** | `none`, `hflip`, `flips` (identidade, H, V, HV) | Médias de mapas simetrizados podem reduzir dependência de posições/cantos sem exigir retreino | `flips`: +0,0120 pareado em média; 46% de pares positivos em yoke. Custo de aproximadamente quatro passagens do ramo pixel. |
| **Suavização Gaussiana** | Grid incluiu $\sigma=0,2,4,8,12$ conforme rodada | Compromisso entre ruído espacial e preservação de defeitos pequenos | $\sigma=2$ foi escolhido para `glass` (defeito médio 0,14%); $\sigma=8$ apareceu nos defeitos maiores. Efeito médio da escolha de sigma foi pequeno (+0,0031 na cadeia). |
| **Feature normalization (`featnorm`)** | Desligada, ligada ou seletiva em L1 (`feat_norm_levels=0`) | Z-score por canal pode controlar escalas heterogêneas de ativações ReLU; motivação relacionada à modelagem estatística de features, mas não reproduz PaDiM [7] | Não generaliza: ajuda vari-grip e glass, prejudica yoke. A variante seletiva é suportada pelo código, mas não se deve alegar benefício universal sem uma rodada correspondente de teste. |
| **Feature pooling** | Kernel 0/1 desligado; kernel 3 variante | Agregação de vizinhança local pode reduzir ruído e ampliar contexto, inspirada na motivação de patches do PatchCore [8] | Efeito isolado marginal no estudo inicial (+0,011 AUROC); desligado como default. |
| **Rotação no treino** | `RandomRotation(180)` por configuração base; `--no_rotation` em variantes | DifferNet usa transforms para robustez image-level; em CFLOW espacial, rotação altera o prior de posição e pode criar cantos pretos | Efeito dependente do nível: desligar melhora L2/L3, mas pode piorar L1. Avaliação in-training antiga tinha bug imagem-máscara; dados anteriores à correção não devem ser usados. |
| **Padding no treino** | 0 ou 112 em variante experimental | Testa se coerência treino-inferência ajuda a densidade | A evidência disponível favorece padding de inferência, não treinamento; conteúdo refletido não é amostra natural de treino. |
| **Margem de borda** | 0 e 8 px em fases iniciais; seleção final usa 0 | Diagnóstico de quanto a borda afeta a métrica, não máscara de produção | Margem removida da seleção: excluir a região problemática pode inflar AUROC e ocultar falha. |
| **Otimizador e agenda** | Adam, LR $2\times10^{-4}$, cosine, warmup 5 épocas | Compatível com configuração do flow local e CFLOW; LR estável para estimar densidade | Usados no batch run; não houve grid de LR. Pico yoke na época 4 sugere testar LR/warmup, mas não prova causalmente que LR seja a causa. |
| **Clipping/AMP** | Grad clip 1.0 por flow; AMP desligado por padrão | Clipping separado evita que a norma do flow com mais canais domine as atualizações; precisão baixa pode afetar log-determinantes | Correções de engenharia/estabilidade; não são ganhos comparativos principais. |
| **Épocas e avaliação** | Ablação focal em 30–40; batch run sobrescreveu para 80; avaliação a cada 4 épocas | A seleção deve acompanhar a curva de generalização, não somente NLL de treino | Yoke atingiu pico na época 4 e perdeu 0,0299 até 80. O orçamento global 80 ultrapassou o orçamento de 12 épocas da receita yoke. |
| **AUPRO por quantis** | Thresholds dos pixels normais em $[1-\mathrm{FPR}_{max},1]$, até 2 milhões de scores amostrados | NLL raw tem cauda pesada; thresholds uniformes no intervalo de score podem não amostrar a faixa FPR útil | Corrige retorno silencioso `AUPRO=0` em vari-grip; invariância a outlier extremo e reescala foi testada localmente. |
| **Score composto e histórico** | `best_composite.pt`, `history.json`, plot regravado; `--patience` | AUROC pixel, AUROC de imagem derivada do mapa e AUPRO podem divergir; histórico torna curvas auditáveis | Portado como prática de monitoramento do RD++. O AUROC image-level composto aqui é **map-derived**, não necessariamente o score image-level original do RD++. Early stopping ainda precisa de avaliação em runs novas. |

### 4.1 Por que o `reflect_pad=112` é um parâmetro geométrico

A AlexNet tem stride acumulado 16 no caminho de features considerado. Com uma entrada 448 px e grade final 96, o passo espacial equivalente é 14 px. O padding precisa deslocar o campo de entrada por um múltiplo do stride e corresponder a número inteiro de células da grade. O menor valor positivo que satisfaz as duas condições é 112 px. Portanto, 112 não foi escolhido só por busca empírica: ele é o passo de padding que preserva o registro da grade para esta configuração. Se `img_size`, `out_size` ou stride mudar, o valor válido deve ser recalculado.

### 4.2 Por que TTA por flips não é o mesmo que as rotações do DifferNet

DifferNet e AttentDifferNet usam múltiplas transformações (inclusive rotações e ajustes de brilho/contraste no método original) para tornar o score de imagem mais robusto. Esta cabeça espacial usa apenas flips horizontais/verticais no conjunto de inferência, desfazendo cada transformação no mapa antes da média. A escolha é operacional: flips preservam uma correspondência espacial simples e não interpolam arbitrariamente as máscaras como rotações poderiam. Isso é uma adaptação para mapa pixel-level, não uma reprodução integral do conjunto de transforms do DifferNet.

## 5. Dados e protocolo experimental

### 5.1 INSPLAD-seg utilizado [10]

| Classe | Normais de treino | Teste normal/anômalo | Tipo/área média anotada |
|---|---:|---:|---|
| `vari-grip` | 477 | 114 / 40 | Ferrugem extensa; 4,6% |
| `lightning-rod-suspension` | 462 | 117 / 46 | Ferrugem fina; 1,9% |
| `glass-insulator` | 1.104 | 591 / 87 | *Missing cap*; 0,14% |
| `polymer-insulator-upper-shackle` | 935 | 235 / 82 | Oxidação/desgaste; 3,09% |
| `yoke-suspension` | 4.834 | 1.207 / 46 | Ferrugem/trinca; 1,96% |

As contagens são do teste completo usado no batch run. Na confirmação anterior, yoke foi limitado a 200 imagens (154 normais + 46 anômalas); números daquela rodada não devem ser misturados com este total. Também é necessário diferenciar estes dados do InsPLAD-fault do artigo AttentDifferNet: versão/tarefa e rótulos são diferentes.

### 5.2 Protocolo de ablação e métricas

A análise teve dois regimes que devem permanecer separados:

1. **Ablation pós-hoc:** mapas são calculados e reagrupados sob configurações diferentes. O split de teste disponível é particionado em validação (40%) e holdout (60%); a configuração é escolhida na porção de validação, e o holdout da rodada é reportado para comparação.
2. **Treino em lote `run_20260907_145952`:** a avaliação in-training usou o loader de teste determinístico, sem TTA, pad 0, sigma 6 e todos os níveis; o mesmo loader foi usado para selecionar o checkpoint. Portanto, suas curvas são métricas no teste reutilizado, não validação independente. O relatório pós-treino calculou receita de deploy em teste completo; o modo `per_class` também foi escolhido depois de inspecionar os resultados por classe.

Métricas: Pixel AUROC (ordenação global de pixels), Pixel AP (precision-recall sob forte desbalanceamento), AUPRO [15] e AUROC de imagem derivada do mapa quando indicada. No avaliador local, AUPRO é calculado por imagem anômala com thresholds lineares, considerando pontos com FPR<0,3 e normalizando o eixo FPR pelo máximo observado antes da integração; isso deve ser levado em conta ao comparar com implementações que integram explicitamente até FPR=0,3. AP é reportado com baseline trivial quando possível; o AUROC image-level nativo do `SEDifferNet` é uma avaliação separada e não deve ser confundido com o score de imagem reduzido do mapa CFLOW.

A unidade de comparação mais informativa foi o par de configurações que difere em um fator e mantém os demais constantes. A cadeia incremental é útil para comunicar uma receita acumulada, mas seus deltas dependem da ordem e não são estimativas causais isoladas. Diferenças abaixo de 0,005 AUROC foram tratadas com cautela porque uma medição controlada do mesmo modelo sob dois splits variou 0,0044. Esse valor é uma resolução empírica do estudo, não intervalo de confiança formal nem limiar estatístico universal.

## 6. Resultados principais

### 6.1 Ganho de inferência e score

Na cadeia incremental do grid de validação, a média das cinco classes passou de 0,8154 para 0,9120 Pixel AUROC: ganho total +0,0965. Os incrementos médios reportados foram +0,0080 para pad112, +0,0173 para TTA flips, +0,0381 para `raw`, +0,0301 para seleção dos níveis e +0,0031 para sigma. Em comparações pareadas, `raw` versus min-max obteve +0,0497 médio e foi positivo em 76–100% dos pares segundo a classe. TTA foi +0,0120; pad112 +0,0049. Estes valores pertencem ao grid de validação, não são uma única comparação de teste independente.

**Limites:** os máximos por escala e do mapa agregado foram estimados com toda a coorte de teste da respectiva classe, sem usar rótulos para calcular o score. Assim, os números são exploratórios/transdutivos e não equivalem a teste independente nem a modo streaming. A implementação não calculou AUPRO (`--skip_aupro`). Como `align_corners` também difere entre o modo oficial e os modos locais, falta um teste controlado que fixe a interpolação para isolar o efeito da fórmula. Artefatos: `resultado_analise_final/pixel/cflow_original_paired_run145952/`.

#### 6.1.1 Resultado por classe: agregação original versus `raw`

| Classe (n imagens) | `raw` AUROC / AP | `cflow_original` AUROC / AP | Δ AUROC / Δ AP |
|---|---:|---:|---:|
| glass (649) | 0,7589 / 0,0033 | **0,7700 / 0,0081** | +0,0111 / +0,0048 |
| lightning-rod (135) | 0,9070 / 0,0310 | **0,9172 / 0,0379** | +0,0103 / +0,0070 |
| polymer (283) | **0,6055** / 0,0087 | 0,6005 / **0,0161** | −0,0050 / +0,0074 |
| vari-grip (154) | 0,8648 / 0,0550 | **0,9051 / 0,0836** | +0,0403 / +0,0285 |
| yoke (1.199) | 0,9287 / 0,0140 | **0,9410 / 0,0228** | +0,0123 / +0,0088 |
| **Média macro por classe** | 0,8130 / 0,0224 | **0,8268 / 0,0337** | **+0,0138 / +0,0113** |

O total foi de 2.420 imagens e 252 anômalas. Os máximos por escala e do mapa agregado foram estimados com toda a coorte de teste da respectiva classe, sem usar rótulos para calcular o score. Assim, os números são exploratórios/transdutivos e não equivalem a teste independente nem a modo streaming. A implementação não calculou AUPRO (`--skip_aupro`). Artefatos: `resultado_analise_final/pixel/cflow_original_paired_run145952/`.

### 6.2 Por que um nível pior pode contaminar a soma

A hierarquia observada em heads retreinadas não foi universal. A tabela seguinte resume AUROC por nível isolado no protocolo principal de ablação; os números de `run_20260907_145952` são mostrados à parte na Seção 6.4 porque o ranking depende da head.

| Nível | vari-grip | lightning-rod | glass | polymer | yoke |
|---|---:|---:|---:|---:|---:|
| L1 | 0,893 | 0,760 | 0,874 | 0,776 | 0,852 |
| L2 | 0,817 | 0,807 | 0,650 | **0,880** | 0,798 |
| L3 | 0,787 | **0,877** | **0,942** | 0,841 | **0,879** |
| Melhor conjunto reportado na validação | `0` | `2` | `02` | `012` | `02` |

`levels=02` é um compromisso de baixo *regret* no grid original em quatro classes (≤0,0012); polymer é a exceção desse experimento. A afirmação de que L2 nunca contribui é falsa. Também é incorreto transformar este ranking em lei anatômica: o modelo, a receita de treino e o checkpoint influenciam a distribuição e o score dos níveis.

### 6.3 Treino: featnorm, clamp e dinâmica de épocas

- **Featnorm não generaliza.** No conjunto de receitas analisado, foi favorável em vari-grip (+0,044 AUROC) e glass (+0,0396), mas desfavorável em lightning-rod (−0,0107), polymer (−0,0066 em AUROC, apesar de +0,0502 AUPRO na comparação indicada) e yoke (−0,0499 AUROC / −0,1226 AUPRO). A interpretação de que normalizar ajuda L1 e pode amplificar canais fracos de níveis profundos é uma hipótese compatível com as observações, não um mecanismo isolado por uma ablação de canais.
- **Clamp 1,9 não foi ganho universal.** Os deltas medidos ao adicioná-lo sobre featnorm variaram de +0,0080 em vari-grip a −0,0117 em yoke. O custo de treino aumentou em várias receitas; por isso 0,5 permaneceu baseline.
- **Overfitting por classe:** no batch de 80 épocas, yoke atingiu 0,9292 na época 4 e terminou em 0,8993 (−0,0299); vari-grip caiu 0,0123; glass, lightning e polymer ficaram quase estáveis no AUROC depois do pico. `--epochs 80` global sobrescreveu o budget específico de yoke (12). A taxa de aprendizado excessiva foi aventada como hipótese, mas não foi testada em grid; o resultado sustenta reduzir/selecionar épocas para aquela configuração, não atribuir causalidade exclusiva ao LR.

### 6.4 Achados finos do batch run e ganho retrospectivo de `per_class`

As heads treinadas em `run_20260907_145952` foram medidas com a receita de deploy (raw, pad112, flips, sigma 8). L1/L2/L3 isolados apresentaram:

| Classe | L1 | L2 | L3 | L1+L3 (`auto`) | L1+L2+L3 |
|---|---:|---:|---:|---:|---:|
| glass | 0,8584 | — | 0,9394 | 0,9534 | — |
| lightning-rod | 0,8015 | — | 0,8761 | 0,8622 | — |
| polymer | 0,8279 | 0,7771 | 0,8525 | 0,8505 | 0,7987 |
| vari-grip | **0,9191** | — | 0,8052 | 0,8561 | — |
| yoke | 0,8929 | — | 0,8960 | 0,9348 | — |

Em vari-grip, o recipe optimal havia treinado L1+L3; L3 era o nível mais fraco. Selecionar L1 sozinho, sem retreino, subiu de 0,8561/0,6124 para 0,9191/0,7621 (AUROC/AUPRO). Isso supera numericamente o alvo de AUPRO 0,7585 de uma rodada anterior, mas foi medido após consultar o teste completo; portanto é evidência diagnóstica forte para hipótese, não uma estimativa confirmatória independente. A receita de treino em lote foi então alterada para manter L1 e L2 disponíveis (`levels=0,1`), permitindo selecionar L1 na inferência sem eliminar L2.

No mesmo batch, `per_class` versus `auto` resultou em médias de 0,8914→0,9068 AUROC e 0,5992→0,6446 AUPRO; os ganhos vieram de lightning-rod (nível 2) e, sobretudo, vari-grip (nível 0). Polymer foi mantido em `02`: L3 sozinho melhorou AUROC por apenas 0,0020, abaixo da variação empírica de split, enquanto perdeu 0,0601 AUPRO frente a `02`. `012` nesta head de polymer caiu para 0,7987/0,4090, contra resultado anterior 0,8806/0,5943 em outra head/rodada. Isso documenta dependência do ranking em relação ao checkpoint.

**Limite de validade:** o dicionário `PER_CLASS_SCORE_LEVELS` foi escolhido depois de olhar as métricas do teste integral de `run_20260907_145952`, e os números 0,9068/0,6446 são calculados nesse mesmo conjunto. Eles são resultados exploratórios retrospectivos, sujeitos a viés de seleção. Para alegar ganho de implantação, o mapa deve ser congelado e avaliado em um teste novo/intocado ou por um protocolo externo pré-registrado.

### 6.5 Correção de AUPRO em mapas de cauda pesada

**Escopo da correção:** os quantis foram implementados em `compute_aupro` no script de treinamento. Os avaliadores `evaluate_pixel_level.py` e `evaluate_full_pipeline.py` chamam `compute_pro_rd_style`, que continua usando thresholds lineares por imagem, média sobre imagens anômalas, filtro FPR<0,3 e reescala pelo maior FPR observado. Assim, os AUPRO das tabelas de avaliação abaixo não foram recalculados pela rotina por quantis. O AUPRO do monitor/score composto de treino usa `compute_aupro`; os dois valores não devem ser misturados sem recalcular ambos por um método unificado.

A implementação antiga distribuía 200 thresholds uniformemente entre o score mínimo e máximo. Em mapas NLL raw, alguns outliers estendem esse intervalo e deixam poucos thresholds com FPR ≤0,3; o código retornou AUPRO=0 em vari-grip. A versão corrigida calcula thresholds como quantis da distribuição de scores de pixels normais na região relevante, com amostragem limitada a até dois milhões de pixels normais para custo. Testes locais mostraram estabilidade a um outlier de $10^6$ e reescala ×1000. Essa correção é específica da implementação e deve ser validada contra uma implementação de referência PRO/AUPRO antes de tratar os números recalculados como intercambiáveis com benchmarks publicados.

### 6.6 Técnicas inspiradas no RD++

O README do repositório local identifica o método como *Revisiting Reverse Distillation for Anomaly Detection* (CVPR 2023, Tien et al. [9]). Isso o distingue do artigo anterior de reverse distillation (Deng e Li, 2022), que não é a referência do script `rd++/main.py` analisado aqui.

O RD++ treina um método diferente: encoder congelado, projeções/decoder reverso e sinais de pseudoanomalia com noise simplex para otimizar localização. A inspiração transferida aqui foi principalmente operacional: persistir curva de treinamento, observar AUPRO por avaliação, salvar checkpoints por mais de um critério e interromper runs longas. O score composto implementado é $(\mathrm{PixelAUROC}+\mathrm{ImageAUROC}_{map}+\mathrm{AUPRO})/3$; a parcela de imagem é derivada do mapa CFLOW, e não o AUROC image-level nativo do SEDifferNet. Portanto, o componente tem motivação análoga à seleção composta do RD++, mas não é uma reprodução semântica garantida da métrica usada no paper. As pseudoanomalias e a loss do RD++ não foram portadas.

| Classe | RD++ Pixel AUROC / AUPRO | CFLOW local, `auto` / receita | Leitura com cautela |
|---|---|---|---|
| glass | 0,9368 / 0,7560 | 0,9534 / 0,6591 | CFLOW maior AUROC; RD++ maior AUPRO |
| polymer | 0,9024 / 0,6508 | 0,8505 / 0,5933 | RD++ maior nas duas métricas |
| vari-grip | 0,8588 / 0,5366 | 0,8561 / 0,6124 | AUROC próximo; CFLOW maior AUPRO antes de `per_class` |
| yoke | 0,9708 / 0,7328 | 0,9348 / 0,6686 | RD++ maior nas duas métricas |

Este quadro é comparativo descritivo, não uma comparação controlada: são pipelines, backbones, checkpoints e condições de avaliação diferentes; não há linha para lightning-rod. Não se atribui a diferença apenas à arquitetura ou à perda.

## 7. Limitações, riscos de validade e próximos experimentos

1. **Teste também usado para seleção.** O dataset não fornece validação independente. A ablação cria validação/holdout a partir do teste; o treino pixel-level também avalia e seleciona checkpoints no loader de teste. Essa prática permite estudar decisões com os dados disponíveis, mas torna os resultados otimistas em relação a um teste totalmente intocado. `per_class` é especialmente exploratório porque foi configurado após métricas do teste completo.
2. **Poucos splits/seeds.** A variação de 0,0044 foi medida em uma comparação controlada de dois splits, não é erro padrão, intervalo de confiança ou teste de significância. É um alerta de resolução, não um limiar universal.
3. **Cinco classes de um dataset e um backbone local.** Hierarquia de níveis, sigma, padding e featnorm não são garantidos em MVTec, VisA, outros backbones ou novas câmeras.
4. **Comparabilidade bibliográfica limitada.** AttentDifferNet reporta image AUROC em InsPLAD-fault; CFLOW-AD reporta seus próprios benchmarks e protocolo; este estudo reporta pixel metrics em INSPLAD-seg. Não usar os respectivos valores como comparação direta de ranking.
5. **Explicações de textura/semântica são interpretação.** O viés de textura descrito para CNNs ImageNet [14] é uma motivação geral, não prova que os níveis desta AlexNet correspondam exclusivamente a “textura” ou “semântica”. As métricas de L1/L2/L3 são compatíveis com escalas de receptivo distintas, mas esta ablação não estabelece causalidade representacional.
6. **Parâmetros fixos.** `out_size=96`, PE=64, 8 blocos, hidden=256 e LR $2\times10^{-4}$ não foram ablatados sistematicamente. `pad_mode=replicate` e `per_level_prob` com featnorm permanecem questões abertas.
6. **Parâmetros fixos e calibração cohort-wide.** `out_size=96`, PE=64, 8 blocos, hidden=256 e LR $2\times10^{-4}$ não foram ablatados sistematicamente. `pad_mode=replicate` e `per_level_prob` com featnorm permanecem questões abertas. O score oficial foi testado, mas sua normalização usa máximos da coorte completa; falta validar um conjunto de calibração fixo para inferência independente.
7. **AUPRO.** Há duas implementações locais: quantis no treino e `compute_pro_rd_style` no avaliador. Unificá-las e validar contra a implementação de referência do benchmark é necessário antes de comparar AUPRO numericamente entre essas rotinas ou com outros artigos.
8. **Parada precoce.** A curva de yoke motiva early stopping e budgets por classe, mas a implementação de `--patience` precisa de novas runs independentes, com validação separada, antes de ser declarada solução geral.

Próximo protocolo recomendado: separar normais de treino em treino/validação antes de ajustar qualquer parâmetro; manter o conjunto de teste totalmente intocado; fixar previamente métrica de seleção composta e `levels`; avaliar múltiplos seeds; depois testar `per_level_prob` com normalização seletiva e repetir por classe. O teste `per_class` deve ser refeito em dados independentes.

## 8. Síntese das decisões de implantação

- `raw` é a escolha default **empírica para as heads testadas**, não o score definido pelo CFLOW-AD original.
- `raw` permanece uma alternativa local por imagem e superou o min-max legado nas ablações originais. Para avaliação offline em lote, o sweep pareado favoreceu `cflow_original` em média; seu uso exige a coorte completa e calibração/protocolo transdutivo explícitos.
- `reflect_pad=112`, TTA flips e sigma foram mantidos por balanço medido de localização/latência; flips são opcionais em tempo real por custo próximo a 4×.
- `levels=02` é um default de baixo regret no estudo de ablação original; a preferência `per_class` precisa validação externa, principalmente para vari-grip.
- `featnorm` e clamp 1,9 exigem validação por classe/head; não são defaults universais.
- A cabeça CFLOW pode produzir localização que o score global de DifferNet não foi projetado para fornecer. As duas tarefas devem ser reportadas separadamente.

## Referências

[1] M. Rudolph, B. Wandt, and B. Rosenhahn, “Same Same But DifferNet: Semi-Supervised Defect Detection with Normalizing Flows,” *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)*, 2021, pp. 1907–1916. https://doi.org/10.1109/WACV48630.2021.00195. Preprint: https://arxiv.org/abs/2008.12577.

[2] A. L. Vieira e Silva, F. Simões, D. Kowerko, T. Schlosser, F. Battisti, and V. Teichrieb, “Attention Modules Improve Image-Level Anomaly Detection for Industrial Inspection: A DifferNet Case Study,” *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)*, 2024, pp. 8246–8255. Official paper: https://openaccess.thecvf.com/content/WACV2024/html/Vieira_e_Silva_Attention_Modules_Improve_Image-Level_Anomaly_Detection_for_Industrial_Inspection_A_WACV_2024_paper.html.

[3] D. Gudovskiy, S. Ishizaka, and K. Kozuka, “CFLOW-AD: Real-Time Unsupervised Anomaly Detection with Localization via Conditional Normalizing Flows,” *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)*, 2022, pp. 1819–1828. https://doi.org/10.1109/WACV51458.2022.00188. Preprint: https://arxiv.org/abs/2107.12571. Official code: https://github.com/gudovskiy/cflow-ad.

[4] J. Hu, L. Shen, and G. Sun, “Squeeze-and-Excitation Networks,” *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2018, pp. 7132–7141. https://doi.org/10.1109/CVPR.2018.00745.

[5] S. Woo, J. Park, J.-Y. Lee, and I. S. Kweon, “CBAM: Convolutional Block Attention Module,” *Proceedings of the European Conference on Computer Vision (ECCV)*, 2018, pp. 3–19. https://doi.org/10.1007/978-3-030-01234-2_1.

[6] L. Dinh, J. Sohl-Dickstein, and S. Bengio, “Density Estimation Using Real NVP,” *International Conference on Learning Representations (ICLR)*, 2017. https://arxiv.org/abs/1605.08803.

[7] T. Defard, A. Setkov, A. Loesch, and R. Audigier, “PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization,” *Proceedings of the International Conference on Pattern Recognition Workshops*, 2021, pp. 475–489. https://arxiv.org/abs/2011.08785.

[8] K. Roth, L. Pemula, J. Zepeda, B. Schölkopf, T. Brox, and P. Gehler, “Towards Total Recall in Industrial Anomaly Detection,” *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2022, pp. 14318–14328. https://doi.org/10.1109/CVPR52688.2022.01392.

[9] T. D. Tien, A. T. Nguyen, N. H. Tran, T. D. Huy, S. T. M. Duong, C. D. T. Nguyen, and S. Q. H. Truong, “Revisiting Reverse Distillation for Anomaly Detection,” *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2023, pp. 24511–24520. Official implementation and citation: https://github.com/tientrandinh/Revisiting-Reverse-Distillation.

[10] A. L. B. Vieira-e-Silva et al., “InsPLAD: A Dataset and Benchmark for Power Line Asset Inspection in UAV Images,” *International Journal of Remote Sensing*, vol. 44, no. 23, pp. 7465–7484, 2023. https://doi.org/10.1080/01431161.2023.2282276.

[11] B. Alsallakh, N. Kokhlikyan, V. Miglani, J. Yuan, and O. Reblitz-Richardson, “Mind the Pad—CNNs Can Develop Blind Spots,” *International Conference on Learning Representations (ICLR)*, 2021. https://arxiv.org/abs/2010.02178.

[12] M. A. Islam, S. Jia, and N. D. B. Bruce, “How Much Position Information Do Convolutional Neural Networks Encode?” *International Conference on Learning Representations (ICLR)*, 2020. https://arxiv.org/abs/2001.08248.

[13] O. S. Kayhan and J. C. van Gemert, “On Translation Invariance in CNNs: Convolutional Layers Can Exploit Absolute Spatial Location,” *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2020, pp. 14274–14285. https://doi.org/10.1109/CVPR42600.2020.01429.

[14] R. Geirhos et al., “ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness,” *International Conference on Learning Representations (ICLR)*, 2019. https://arxiv.org/abs/1811.12231.

[15] P. Bergmann, M. Fauser, D. Sattlegger, and C. Steger, “MVTec AD—A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection,” *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2019, pp. 9592–9600. https://doi.org/10.1109/CVPR.2019.00982.


### Artefatos locais para reprodução

- Plano e resultados detalhados da ablação: `docs/plano_ablation_nf_head.md`.
- Manuscrito revisado em Markdown: `docs/artigo_ablation_cflow_differnet.md`.
- Código do backbone image-level: `core/model.py`.
- Implementação da cabeça pixel-level e dos modos de score: `core/cflow.py`.
- Treinamento da cabeça: `scripts/train/pixel_train_from_pretrained.py`.
- Avaliação pixel-level: `scripts/eval/evaluate_pixel_level.py`.
- Implementação do score por classe: `core/eval_pipeline.py`.
- Batch run analisado: `resultado_analise_final/treino_nf_heads/run_20260907_145952/`.

**Declaração de assistência:** este documento foi estruturado e redigido com assistência de IA a partir dos artefatos do repositório e das referências citadas. Interpretações e decisões científicas finais devem ser conferidas e aprovadas pelo responsável humano pelo estudo.
