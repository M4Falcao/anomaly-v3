# Plano — ablação de máscara de objeto (opção 2: corrigir a máscara antes de retreinar)

Estado: **implementado** (D-19), ainda não executado com o YOLO26. O código está em
[`ablation_bg_removal/`](../ablation_bg_removal/README.md); a seção 3.7 lista onde a
implementação difere desta proposta. Nenhum treino de modelo foi executado até agora.
Data: 2026-10-30.

## 1. Pergunta

Com a remoção de fundo, glass e polymer perderam muito Pixel AUROC (0.924→0.680 e 0.867→0.667),
enquanto lightning ganhou (0.866→0.921) e vari/yoke ficaram estáveis
(`resultado_analise_final/pixel_notebook/run_with_background` × `run_no_background`).
A máscara atual apaga parte do defeito: 41.7% dos pixels de GT de glass e 69.5% de polymer caem no
fundo preto, contra 8–17% nas demais classes.

Pergunta: **existe uma máscara de objeto, ampliada de forma controlada, que mantenha o defeito
visível e ainda retire fundo suficiente para melhorar AUROC/AUPRO em relação à rodada com fundo?**

## 2. Dilatação grande × envoltória convexa — evidência preliminar

Diagnóstico exploratório, 2026-10-30. Máscara reconstruída das imagens de
`insplad-seg-no-back-treino` (pixel com máx. RGB > 10 + fechamento 5×5), em 448×448, train+test.
"GT perdido" = fração dos pixels de GT fora da máscara. Usou o teste inteiro, então **serve só para
desenhar o grid, não para escolher a estratégia**.

| Classe | raw | dil16 | dil32 | dil64 | hull | hull+dil16 | bbox+10% |
|---|---|---|---|---|---|---|---|
| glass — área mantida / GT perdido | .27/.43 | .40/.35 | .50/.29 | .65/.26 | .41/.40 | .48/.34 | .63/.27 |
| lightning | .15/.09 | .25/.01 | .34/.00 | .51/.00 | .27/.02 | .34/.00 | .54/.00 |
| polymer | .09/.69 | .18/.56 | .27/.45 | .42/.23 | .20/.51 | .27/.41 | .45/.13 |
| vari-grip | .21/.07 | .33/.04 | .42/.04 | .58/.04 | .31/.05 | .39/.04 | .74/.04 |
| yoke | .20/.18 | .29/.06 | .38/.04 | .55/.03 | .30/.06 | .39/.04 | .59/.03 |

Leitura:

- **A dilatação é a opção mais segura.** Com a mesma área, ela perde o mesmo GT que a envoltória
  convexa ou menos, exceto em polymer, onde a envoltória fica um pouco à frente. A dilatação também
  tem um único parâmetro monotônico (raio) e uma garantia local: tudo a até r px da peça fica
  visível. A envoltória não tem parâmetro, não cobre o que está fora do contorno externo (ex.: a capa
  ausente na ponta de uma cadeia) e, em peças em "Y"/"V", inclui fundo arbitrário entre os braços.
  Isso gera um contorno artificial que depende da pose.
- **Nenhuma das duas corrige glass e polymer.** Mesmo dil64 ou bbox+10% ainda perdem 23–27% do GT.
  O problema está na máscara de origem: `yolo-segmentation/create_dataset.py` usa só
  `masks.data[0]`, a primeira instância, e **descarta** imagens sem detecção. Por isso o plano começa
  regenerando as máscaras.
- vari-grip tem um resíduo de ~4% que não muda com nenhuma estratégia. Provavelmente é GT fora do
  quadro ou desalinhado; vale auditar, mas não deve guiar a escolha.

## 3. Script proposto

`scripts/eval/ablation_object_mask.py` + launcher `scripts/eval/run_object_mask_ablation.ps1`.
Saída: `resultado_analise_final/ablacao_mascara_objeto/run_<timestamp>/` (nunca sobrescrever).

### 3.1 Estágios (`--stage`)

| Estágio | O que faz | Custo | Usa GPU |
|---|---|---|---|
| `masks` | Roda o YOLO-seg sobre o dataset **original** `insplad-seg\insplad-seg` e salva a máscara da **união de todas as instâncias**, com `conf` baixo, em cache. Imagem sem detecção ganha máscara cheia (= com fundo) e fica registrada no relatório; nenhuma imagem é descartada. **Implementado em [`ablation_bg_removal/`](../ablation_bg_removal/README.md)** com YOLO26-seg treinado por classe na conversão do dataset sem fundo (`train/good`). Saída: `masks/<classe>_<ts>/<split>/<defeito>/<stem>.png`. Máscara binária (o Ultralytics binariza em 0.5); o limiar efetivo é `conf`. | baixo | sim (YOLO) |
| `audit` | Para cada estratégia: área mantida, GT perdido (pixels e imagens >50%/>95%), comprimento do contorno, separado em **train / val / holdout** usando o manifesto de split. Gera figuras de exemplos. | minutos | não |
| `frozen` | **Braço A:** SEDifferNet com fundo congelado (D-01). Retreina só a NF head (preset `general`) com a entrada mascarada e avalia com a mesma receita de inferência. | médio | sim |
| `retrain` | **Braço B:** retreina o SEDifferNet e depois a NF head só para as finalistas do braço A (≤2 por classe). | alto | sim |
| `report` | Consolida as tabelas, faz o bootstrap pareado contra a rodada com fundo e escreve `summary.md`. | baixo | não |
| `all` | `masks → audit → frozen → report`. O braço B exige chamada explícita. | — | — |

### 3.2 Fatores do grid

- `--strategies` (default `raw,dil16,dil32,dil64,hull+dil16,bbox10`): `raw`, `dil{r}` com r em px na
  resolução 448, `hull`, `hull+dil{r}`, `bbox{m}` (margem em % do lado).
- `--fills` (default `black,imagenet_mean`): `black`, como no dataset atual; `imagenet_mean`, que
  vira zero depois da normalização e reduz o desajuste da hipótese 3; `blur` (Gaussiano forte do
  fundo original) opcional.
- `--feather` (default `0,8`): borda suave da máscara em px, para atenuar o contorno artificial
  (hipótese 2).
- `--mask_threshold` (default `0.25`): limiar sobre a máscara soft, antes da estratégia.
- Os demais fatores ficam fixos: receita `general` por classe, train seed 42, split seed 42 (D-13).

O grid completo tem 6×2×2 = 24 células por classe, caro demais para o braço A. Regra de poda,
decidida **antes** de ver os resultados:

1. No `audit`, descartar estratégias com **GT perdido na validação > 5%**.
2. Das que sobram, manter no braço A as 2 de **menor área**: mais fundo removido, defeito
   preservado.
3. Se nenhuma estratégia passar em uma classe (glass e polymer, se a máscara regenerada não
   resolver), usar `bbox20` e `dil64` nessa classe e anotar a classe como "máscara não confiável".

Com isso o braço A fica com no máximo 2 estratégias × 2 fills × 1 feather (o feather só entra se a
análise de contorno indicar falsos positivos na borda) ≈ 4 heads por classe.

### 3.3 Implementação

- `core/object_mask.py` (novo): `load_soft_mask`, `build_mask(strategy, soft, threshold)` e
  `apply_fill(img, mask, fill, feather)`. São funções puras, com testes sintéticos.
- `MaskedImageFolder` (wrapper de `ImageFolderMultiTransform` / `AlignedTestDataset`): aplica a
  máscara na imagem PIL **antes** das transformações. Rotação e resize atuam sobre a imagem já
  mascarada, então o alinhamento com o GT não muda.
- Avaliação de pixel: forçar `testset.transform` determinístico e `n_transforms=1` (armadilha do §6).
- Gravar no checkpoint e em `recipe.json` o bloco `object_mask = {strategy, fill, feather, threshold,
  mask_cache_hash}`. Avaliadores externos e o notebook D-18 precisam reaplicar a mesma máscara.
- `--dry_run` / `-PlanOnly` imprime classes, estratégias e contagens de máscaras sem treinar.

### 3.4 Comparabilidade (crítico)

- **Inventário fixo:** todas as rodadas usam o teste do dataset **original** (sem descarte) e o mesmo
  `split_manifest.json` da rodada com fundo (`val_fraction=0.4`, `split_seed=42`). Isso elimina a
  hipótese 1 (conjuntos diferentes).
- **Referências no mesmo protocolo:** (i) rodada com fundo, (ii) máscara `raw` + `black`, que
  reproduz o experimento sem fundo, mas no inventário completo.
- **Seleção:** época pelo Pixel AUROC de validação em `raw` (D-04); estratégia/fill finalista pelo
  Pixel AUROC de validação, com GT perdido ≤ 5% como restrição. Holdout só no fim.

### 3.5 Métricas reportadas

Por classe, separadas em val e holdout:

- Image AUROC (SEDifferNet; só no braço B) e image AUROC do mapa.
- Pixel AUROC, AP e AUPRO global (`compute_aupro_cflow`).
- **Pixel AUROC dentro da máscara dilatada**: separa o ganho de "o fundo deixou de competir" do
  ganho real de localização na peça.
- **AUROC no subconjunto de GT perdido**: pixels de defeito que ficaram fora da máscara, para mostrar
  o custo residual.
- **FP de contorno**: fração de pixels de imagens `good` acima do limiar de validação dentro de uma
  faixa de ±8 px da borda da máscara.
- Bootstrap pareado estratificado por imagem contra a rodada com fundo (mesmas imagens).

### 3.6 Critério de sucesso (pré-registrado)

Uma estratégia é **recomendada** para uma classe se, na validação:

1. o GT perdido for ≤ 5%; e
2. o ΔPixel AUROC contra "com fundo" for ≥ 0, com IC95 bootstrap não totalmente negativo; e
3. o AUPRO não piorar mais que 0.02.

O holdout só confirma ou desmente. Se a melhor estratégia empatar com "com fundo", a conclusão é
"a remoção de fundo não ajuda nessa classe". Isso é um resultado válido.

### 3.7 Implementação real × proposta (2026-10-30)

- **Local:** scripts numerados `01`–`08` em `ablation_bg_removal/` (pedido do usuário), não
  `scripts/eval/ablation_object_mask.py`. Launcher: `ablation_bg_removal/run_pipeline.ps1`, com
  `-Stage convert|train|masks|audit|select|frozen|retrain|report|all|ablation`.
- **Máscara aplicada por materialização em disco**, não por um `MaskedImageFolder`: cada
  variante gera `data/masked/<máscaras>/<variante>/<classe>/` com os mesmos nomes e a mesma
  árvore do original (JPEG q95, GT por hardlink). O treinador, o split, o notebook D-18 e os
  avaliadores existentes rodam **sem mudança em `core/`**; basta passar `--dataset`. O treino
  grava `--dataset` em `recipe.json`; o manifesto da materialização (estratégia, preenchimento,
  feather, pasta de máscaras) fica junto do dataset.
- **Estratégia definida no grid 448×448** (mesmo grid do SEDifferNet) e reamostrada para a
  resolução nativa; dilatação por transformada de distância.
- **Máscara binária**, sem `--mask_threshold`: o Ultralytics binariza em 0.5; o controle é o
  `conf` (0.10).
- **Split:** o audit e o relatório importam `stratified_test_split` do próprio treinador e usam a
  ordem de `AlignedTestDataset`. A etapa 6 recusa heads cujo `split_manifest.json` difira do
  audit.
- **Referência com fundo:** a head já treinada em `treino_nf_heads/run_20261025_181523` (mesmo
  encoder, receita `general`, seeds 42), não retreinada. O relatório confere
  `selection_validation_paths` contra a validação do audit.
- **Métricas de pixel por histograma** de 65 536 bins (bordas = quantis), testadas contra o
  sklearn; bootstrap pareado com 1 024 bins por imagem.
- **Braço B:** `main.py` escolhe o encoder por Image AUROC no teste inteiro (protocolo dos
  encoders de referência). Isso fica registrado como `encoder_selection=test_full`;
  `--encoder_pick latest` é a alternativa limpa.
- **Fonte de máscara de controle:** `--source nobg` / `-MaskSource nobg` deriva a máscara antiga
  do dataset sem fundo, sem GPU.

Verificação feita (sem treino): 22 testes unitários; smoke test em vari-grip com máscaras
`nobg`. Audit, poda, materialização, `--print_recipe` do treinador e igualdade de
inventário/rótulos/split entre o dataset original e o mascarado passaram. A etapa 8 rodou em
16 imagens estratificadas com a head de referência e com uma head fictícia em dataset
mascarado, gerando métricas, IC e figura. Artefatos do smoke test removidos.

## 4. Bloqueios e riscos

- **Pesos do YOLO:** os pesos antigos (`C:\Users\Pichau\...`) não existem nesta máquina. Por isso
  vamos treinar YOLO26-seg novos a partir da conversão do dataset sem fundo (`ablation_bg_removal/`).
  Os rótulos herdam os erros do segmentador antigo (primeira instância, cortes), atenuados por
  `conf` baixo e união de instâncias.
- **glass/missing-cap:** o defeito é a *ausência* da peça. Uma máscara do objeto presente nunca
  garante cobrir onde a capa deveria estar; dilatação grande ou bbox são as únicas defesas. Uma
  conclusão negativa nessa classe é plausível.
- Uma única seed (D-13); o teste completo já foi inspecionado (o holdout não é independente).
- A máscara reconstruída usada na seção 2 depende do limiar de preto; a ferrugem escura pode virar
  "fundo". O estágio `audit` deve usar as máscaras regeneradas.

## 5. Ordem de execução sugerida

1. Treinar o YOLO26-seg por classe (`-Stage train`) e gerar as máscaras (`-Stage masks`).
2. ~~Implementar o código + testes sintéticos + `--dry_run`~~ (feito em 2026-10-30, ver 3.7).
3. `-Stage audit` e `-Stage select` (sem treino); depois, revisar com o usuário a tabela de GT
   perdido e as finalistas.
4. `-Stage frozen` nas finalistas, **com autorização explícita**.
5. `-Stage report`; o braço B (`-Stage retrain`) só se o braço A mostrar ganho na validação.
