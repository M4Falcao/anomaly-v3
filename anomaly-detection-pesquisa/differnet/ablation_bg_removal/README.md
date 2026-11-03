# ablation_bg_removal — máscaras de objeto para a ablação de remoção de fundo (D-19)

Plano completo: [docs/plano_ablacao_mascara_objeto.md](../docs/plano_ablacao_mascara_objeto.md).
Registro da decisão: D-19 em [docs/contexto_e_decisoes_ia.md](../docs/contexto_e_decisoes_ia.md).

Esta pasta contém a ablação inteira: criação das máscaras (YOLO26-seg treinado com a
conversão do dataset sem fundo), audit das estratégias de ampliação, poda pré-registrada,
retreino da NF head com entrada mascarada (braço A), retreino opcional do SEDifferNet
(braço B) e relatório pareado contra a referência com fundo.

## Plano da ablação

### Motivação

Na avaliação descritiva D-18 (`resultado_analise_final/pixel_notebook/run_with_background`
× `run_no_background`), remover o fundo derrubou o Pixel AUROC de glass (0.924 → 0.680) e de
polymer (0.867 → 0.667). lightning subiu (0.866 → 0.921) e vari/yoke ficaram estáveis. A
queda acompanha a fração do GT de defeito que o segmentador antigo pintou de preto: glass
41.7%, polymer 69.5%, contra 8–17% nas demais classes. O teste sem fundo também perdeu mais
imagens anômalas que normais. A remoção de fundo ajuda quando o defeito continua visível;
o problema é a máscara.

### Pergunta

Existe uma máscara de objeto, ampliada de forma controlada, que preserve o defeito e ainda
retire fundo suficiente para melhorar Pixel AUROC/AUPRO em relação à rodada com fundo?

### Etapas

| # | Etapa | O que faz | Saída | Estado |
|---|---|---|---|---|
| 1 | **convert** (`01_build_yolo_dataset.py`) | Converte o dataset sem fundo em rótulos YOLO-seg: imagem original + polígonos dos pixels não-pretos de `train/good`; split interno 85/15, seed 42. | `data/yolo_seg/<classe>/` | ✅ executada em 2026-10-30 |
| 2 | **train** (`02_train_yolo_seg.py`) | Treina um YOLO26-seg por classe. Critérios de aceite: mask mAP de validação do YOLO e inspeção visual dos `previews`. | `runs/yolo_seg/<classe>_<ts>/` | ⏳ aguardando execução (GPU) |
| 3 | **masks** (`03_generate_masks.py`) | Máscara binária (união de instâncias, `conf` baixo, sem descarte) para **todo** o dataset original, train e test. `--source nobg` deriva a máscara antiga do dataset sem fundo (controle, sem GPU). | `masks/<classe>_<fonte>_<ts>/` | ⏳ depende da etapa 2 |
| 4 | **audit** (`04_audit_masks.py`) | Sem treino. Para cada estratégia (`raw`, `dil8/16/32/64`, `hull`, `hull+dil16`, `bbox10/20`), no grid 448: área mantida, **GT perdido** (pixels e imagens com perda > 50% e > 95%) e imagens sem detecção, por partição train_good/val/holdout. Usa a **mesma** função de split do treinador (`val_fraction=0.4`, `split_seed=42`). Cria a rodada. | `<run>/<classe>/audit/` | ✅ implementada |
| 5 | **poda** (`05_select_strategies.py`) | Regra fixada antes de ver resultados: GT perdido na **validação** ≤ 5%, 2 de menor área em train/good; se nenhuma passar, `dil64` + `bbox20` e `mask_reliable=false`. Variantes = finalistas × `--fills` (default `imagenet_mean` desde D-21; antes `black`,`imagenet_mean`) (+ `--feathers`) + referência `raw__black` (desligável com `--no_reference`). | `<run>/<classe>/finalists.json` | ✅ implementada |
| 6 | **frozen** — braço A (`06_train_nf_head_masked.py`) | SEDifferNet com fundo **congelado** (D-01). Materializa o dataset mascarado (mesmos nomes e árvore → mesmo split; GT por hardlink) e chama `pixel_train_from_pretrained.py --recipe general --dataset <mascarado>`. Confere o `split_manifest` contra o audit e indexa a head. | `<run>/<classe>/arm_A/<variante>/nf_head/`, `heads_index.json` | ✅ implementada |
| 7 | **retrain** — braço B (`07_retrain_encoder_masked.py`) | Retreina o SEDifferNet (`main.py` com `config` sobrescrito) e depois a NF head sobre o encoder novo. Rodado direto, usa só variantes do braço A com ΔAUROC de validação ≥ 0 (lidas do `summary.csv`) ou passadas em `--variants`; o `run_all.py` passa todas as candidatas (D-21). | `<run>/<classe>/arm_B/<variante>/{encoder,nf_head}/` | ✅ implementada |
| 8 | **report** (`08_report.py`) | Reproduz a receita de cada checkpoint no teste completo. Pixel AUROC/AP/AUPRO, Image AUROC, AUROC no objeto, GT perdido e sua detectabilidade, e FP de contorno; Δ contra a head **com fundo** (`treino_nf_heads/run_20261025_181523`) com bootstrap pareado; aplica o critério só na validação. Nível de imagem do SEDifferNet (64 rotações): no braço A mede o encoder **com fundo** recebendo a imagem mascarada só na inferência (diagnóstico sem treino, `--no_encoder_probe` desliga); no braço B mede o encoder retreinado. Ambos com Δ e IC95 contra o encoder com fundo na imagem original; não entram no critério. | `<run>/summary.{csv,md}`, `<run>/<classe>/eval/`, `report_<classe>.png` | ✅ implementada |
| 9 | **image** (`09_image_level_compare.py`) | Nível de imagem, sem treino: SEDifferNet **com fundo** (`final_models/SEDiffernet`) × **sem fundo** (`final_models/SEDiffernet_noBg`), cada um no seu domínio (`bg@orig`, `nobg@nobg`), pareado nas **mesmas imagens** (nomes normalizados), 64 rotações, IC95 pareado, mesmo split val/holdout. `--cross` acrescenta cada encoder no domínio do outro; `--extra NOME=RAIZ` avalia o encoder sem fundo em outro dataset (ex.: mascarado pela YOLO26). Ressalva: os dois encoders escolheram a época no próprio teste. | `resultado_analise_final/imagem_bg_vs_nobg/run_<ts>/` | ✅ vari-grip executado (2026-10-31) |

`<run>` = `resultado_analise_final/ablacao_mascara_objeto/run_<ts>/` (criada pelo audit; as
etapas seguintes usam `--run_dir latest` por padrão).

### Regras de comparabilidade

- **Inventário fixo:** todas as rodadas usam o teste do dataset **original**, sem imagens
  descartadas, e o mesmo split val/holdout da rodada com fundo.
- **Referências no mesmo protocolo:** (i) com fundo; (ii) máscara `raw` + `black`, que
  reproduz o experimento sem fundo no inventário completo.
- **Seleção:** época pelo Pixel AUROC de validação em `raw` (D-04); estratégia e
  preenchimento pela validação; o holdout só confirma no fim. Uma seed (D-13).
- **Métricas:** Pixel AUROC, AP, AUPRO global (`compute_aupro_cflow`), Image AUROC (do mapa;
  do SEDifferNet só no braço B), **AUROC dentro da máscara**, **AUROC nos pixels de GT
  perdidos** e **FP de contorno** (faixa de ±8 px da borda em imagens `good`).
- **Sucesso pré-registrado:** na validação, GT perdido ≤ 5%, ΔPixel AUROC ≥ 0 contra a
  rodada com fundo (IC95 bootstrap não totalmente negativo) e AUPRO sem piorar mais que
  0.02. Empate significa que a remoção de fundo não ajuda naquela classe, o que também é
  um resultado válido.

### Riscos conhecidos

- glass/missing-cap: o defeito é a **ausência** da peça, então a máscara do objeto presente
  pode nunca cobrir o lugar da capa. Uma conclusão negativa é plausível.
- Os rótulos do YOLO26 herdam erros do segmentador antigo (ver "Cuidados").
- Uma seed; o teste completo já foi inspecionado, então o holdout não é independente.

## Roteiro completo para uma classe (exemplo: vari-grip)

> **Atalho recomendado:** `python ablation_bg_removal\run_all.py` roda **todas** as etapas
> (YOLO → máscaras → audit → seleção → braço A → **braço B** → relatório de pixel → nível de
> imagem) classe por classe e gera `<rodada>\relatorio_pipeline.html`, um HTML autocontido com
> status, tabelas, gráficos e imagens. Os caminhos ficam no bloco `CONFIG` no topo do arquivo (ou em um JSON
> passado com `--config`). Etapas já concluídas são reaproveitadas, e `--run_dir <rodada>` retoma
> uma rodada. Exemplos: `--dry_run`, `--classes vari-grip,yoke-suspension`,
> `--stages audit,select,report,image`, `--force yolo`, `--html_only --run_dir <rodada>`.
> Uma classe que falha não interrompe as outras, e o nível de imagem roda mesmo se o pixel
> falhar.
>
> Defaults do `run_all.py` (D-21, 2026-10-31):
> - **Preenchimento.** Só `imagenet_mean`; `black` saiu, e `raw__black` só entra com
>   `include_reference_nobg=True`.
> - **Braço B.** Roda sempre sobre as candidatas de `finalists.json` e retreina o SEDifferNet na
>   imagem mascarada (`07`). **Não treina NF head** (D-24): o retreino só muda o flow de imagem;
>   AlexNet/SE ficam iguais, então a head seria idêntica à do braço A. Fica no índice como
>   `ENCODER_ONLY`. `07 --train_head` restaura o comportamento antigo.
> - **Avaliação depois do braço B.**
>   - O `report` (`08`) avalia o pixel das heads do braço A contra a referência com fundo.
>   - A etapa `image` (`09 --encoder`) inclui a condição `retrained_<variante>@masked_<variante>`.
> - **Falha no braço B.** Não bloqueia o relatório do braço A.
> - **Outros modos.** `arm_b_variants='auto'` restaura a regra antiga ("só se o braço A empatar
>   ou ganhar"); `arm_b_pre_epochs`/`arm_b_meta_epochs` servem para smoke test.

Executar a partir de `differnet/`, com `conda activate anomaly_attentideffernet`.
Todo comando aceita `-DryRun`/`--dry_run` para mostrar o plano sem executar.

```powershell
# 0) Plano (não treina nada)
.\ablation_bg_removal\run_pipeline.ps1 -Stage all -Classes vari-grip -DryRun

# 1) Conversão do dataset sem fundo -> YOLO-seg (já executada; só refazer se mudar parâmetros)
.\ablation_bg_removal\run_pipeline.ps1 -Stage convert -Classes vari-grip -Overwrite

# 2) Treino do YOLO26-seg (GPU). Conferir runs\yolo_seg\vari-grip_<ts>\ (mask mAP, curvas)
.\ablation_bg_removal\run_pipeline.ps1 -Stage train -Classes vari-grip

# 3) Máscaras do dataset original inteiro (train+test) com o best.pt mais recente.
#    Conferir masks\vari-grip_yolo_<ts>\previews\ e masks_report.csv (sem_detecção, multi_instância)
.\ablation_bg_removal\run_pipeline.ps1 -Stage masks -Classes vari-grip

# 4) Audit (CPU, sem treino): cria resultado_analise_final\ablacao_mascara_objeto\run_<ts>\
.\ablation_bg_removal\run_pipeline.ps1 -Stage audit -Classes vari-grip

# 5) Poda pré-registrada -> <run>\vari-grip\finalists.json
.\ablation_bg_removal\run_pipeline.ps1 -Stage select -Classes vari-grip

# 6) Braço A (GPU, treino real): até 5 heads (2 finalistas × 2 preenchimentos + raw__black)
.\ablation_bg_removal\run_pipeline.ps1 -Stage frozen -Classes vari-grip -DryRun   # conferir comandos
.\ablation_bg_removal\run_pipeline.ps1 -Stage frozen -Classes vari-grip

# 8) Relatório do braço A (GPU, inferência no teste completo): <run>\summary.md
.\ablation_bg_removal\run_pipeline.ps1 -Stage report -Classes vari-grip

# 7) Braço B (opcional, treino longo ~200 épocas do SEDifferNet): só se o braço A ganhou na validação
.\ablation_bg_removal\run_pipeline.ps1 -Stage retrain -Classes vari-grip -DryRun
.\ablation_bg_removal\run_pipeline.ps1 -Stage retrain -Classes vari-grip

# 8') Relatório final com os braços A e B (reaproveita as heads já avaliadas)
.\ablation_bg_removal\run_pipeline.ps1 -Stage report -Classes vari-grip -SkipExisting
```

Nível de imagem: `.\ablation_bg_removal\run_pipeline.ps1 -Stage image [-Cross] [-Classes ...]`.

Atalhos: `-Stage ablation` roda 4 → 5 → 6 → 8 em sequência. `-RunDir <pasta>` fixa uma
rodada específica (default `latest`). `-Variants dil32__black,raw__black` sobrescreve as
finalistas nas etapas 6/7. `-Limit N` no `report` avalia um subconjunto estratificado de N
imagens (smoke test; saída `summary_limitN.*`). Opções finas: `python
ablation_bg_removal\0X_*.py --help`.

Controle sem GPU: `-MaskSource nobg` nas etapas `masks` e `audit` usa a máscara antiga
(pixels não-pretos do dataset sem fundo) em vez do YOLO26.

Pontos de parada para revisar antes de seguir:

- **Depois da etapa 3:** inspecionar os `previews`. Se a máscara cortar a peça, ajustar
  `--conf` ou treinar de novo antes do audit.
- **Depois da etapa 5:** confirmar as finalistas antes de gastar GPU na etapa 6.
- **Holdout:** só olhar na etapa 8, depois que as finalistas e a receita estiverem congeladas.

Custo esperado para vari-grip (estimativa): etapa 2 com 402/71 imagens leva dezenas de
minutos; a etapa 3 processa 631 imagens em poucos minutos; a etapa 4 leva ~1 min; a etapa 6
tem até 5 heads com o orçamento `general` (40 épocas cada); a etapa 8 leva ~1 min por head
(154 imagens com TTA flips). No braço B, cada SEDifferNet leva ~200 épocas.

## Estrutura

| Caminho | Conteúdo |
|---|---|
| `common.py` | Caminhos (datasets, rodadas, head de referência), classes e operações puras (máscara a partir da imagem sem fundo, polígonos, união de instâncias). |
| `masking.py` | Estratégias (`raw`, `dil<r>`, `hull`, `hull+dil<r>`, `bbox<m>`) no grid 448, preenchimentos (`black`, `imagenet_mean`, `blur`), feather, faixa de contorno e materialização do dataset mascarado. |
| `protocol.py` | Inventário de teste na ordem de `AlignedTestDataset` e split com a **mesma** `stratified_test_split` do treinador. |
| `heads.py` | Execução logada do treinador, checagem de split e `heads_index.json`. |
| `evaluate.py` | Mapas com a receita do checkpoint (memmap), métricas por histograma (AUROC/AP/FPR) e bootstrap pareado. |
| `01_build_yolo_dataset.py` | Converte `insplad-seg-no-back-treino/<classe>/train/good` em `data/yolo_seg/<classe>/`. A imagem é a **original com fundo**; o rótulo vem dos pixels não-pretos da versão sem fundo. Cada componente conexo vira uma instância. |
| `02_train_yolo_seg.py` | Treina `yolo26s-seg.pt` (default) por classe -> `runs/yolo_seg/<classe>_<ts>/`. |
| `03_generate_masks.py` | Gera as máscaras no dataset **original** (train+test) -> `masks/<classe>_<fonte>_<ts>/<split>/<defeito>/<stem>.png`. |
| `04_audit_masks.py` … `08_report.py` | Etapas 4–8 descritas acima. |
| `_train_encoder_runner.py` | Uso interno do braço B: roda `main.py` com `config` sobrescrito (dataset, classe, pasta de checkpoints). |
| `run_pipeline.ps1` | Launcher de todas as etapas. |
| `run_all.py` | Orquestrador Python de ponta a ponta com bloco `CONFIG` de caminhos; repassa os caminhos às etapas por variáveis `ABR_*` (lidas em `common.py`). |
| `report_html.py` | Gera `relatorio_pipeline.html` autocontido (também isolado: `python ablation_bg_removal\report_html.py <rodada>`). |
| `tests/` | `test_common.py` (conversão) e `test_ablation.py` (estratégias, preenchimento, métricas por histograma × sklearn, regra de poda, materialização + split). Rodar `python -m unittest discover -s ablation_bg_removal\tests`. |
| `data/`, `runs/`, `masks/` | Saídas pesadas (dataset YOLO, datasets mascarados, pesos, máscaras); ignoradas pelo git. |

### Saídas de uma rodada (`<run>/`)

```
run_config.json                    split, raiz do dataset, fonte de máscara
<classe>/audit/                    audit_summary.csv, audit_per_image.csv, split_manifest.json,
                                   audit_config.json (máscaras usadas), audit_tradeoff.png, audit_examples.png
<classe>/finalists.json            regra, tabela de elegibilidade, finalistas, variantes
<classe>/arm_A/<variante>/nf_head/ run_<ts>/ do treinador (best_models/, history.json, split_manifest.json, ...) + train_*.txt
<classe>/arm_B/<variante>/         encoder/run_<ts>_<classe>/ + train_encoder_*.txt + nf_head/
<classe>/heads_index.json          heads treinadas, status, split_check
<classe>/eval/<braço>_<variante>/  metrics.json, per_image.npz (maps.npy só com --keep_maps)
<classe>/report_<classe>.png
summary.csv, summary.md            uma linha por classe × head, colunas val_* e holdout_*
pipeline_log.txt                   (run_all.py) log completo da rodada; anexa ao retomar com --run_dir
logs/<classe>__<etapa>.txt         (run_all.py) log de cada etapa
relatorio_pipeline.html            (run_all.py) relatório HTML
```

Nos logs `.txt`, as barras de progresso (tqdm) aparecem só no estado final, uma linha por
barra. No console, cada barra se atualiza na mesma linha.

## Correções em relação a `yolo-segmentation/create_dataset.py`

- União de **todas** as instâncias (antes: só `masks.data[0]`).
- `retina_masks=True`: máscara na resolução original, sem desalinhamento de letterbox.
- Imagem sem detecção não é descartada: recebe máscara cheia (`--no_detection_policy full`) e aparece em `masks_report.csv`.
- `conf=0.10` por padrão: perder parte da peça custa mais que incluir fundo.
- Só a máscara é salva; a imagem original não é alterada.

## Cuidados

- **Os rótulos herdam os erros do segmentador antigo.** Como vêm do dataset sem fundo,
  imagens em que ele pegou só uma instância ou cortou parte da peça ensinam o mesmo
  erro ao YOLO26. União de instâncias, `conf` baixo e a dilatação posterior atenuam, mas
  não eliminam o problema. O audit da ablação (GT perdido na validação) é o teste que
  decide.
- Só `train/good` alimenta o segmentador. Imagens de teste nunca entram no treino
  (`--source_splits` avisa se isso for alterado).
- Imagens do dataset original sem versão sem fundo (`missing_in_nobg`: glass 12,
  lightning 74, polymer 56, vari 3, yoke 264) ficam fora do treino do YOLO, mas
  recebem máscara na etapa 3.
- **Referência com fundo** = head já treinada em `treino_nf_heads/run_20261025_181523`
  (mesmo encoder, receita `general`, seed 42, split 42). O relatório confere se ela registrou
  a mesma validação (`reference_split_ok`). Para outra referência, use `08_report.py
  --reference_run`.
- O avaliador não suporta `cflow_original` (transdutivo; o preset `general` não usa).
  Métricas de pixel são por histograma (65 536 bins, validado contra o sklearn nos testes).
- Braço B: o `main.py` escolhe o SEDifferNet por Image AUROC no **teste inteiro**, o mesmo
  protocolo dos encoders de referência. Métricas de imagem do braço B no holdout não são
  independentes (marcado como `encoder_selection=test_full`); `--encoder_pick latest` evita isso.
- Os datasets mascarados ficam em `data/masked/<máscaras>/<variante>/` (JPEG q95,
  mesmo nome). Apague-os quando não forem mais necessários; o GT é hardlink.
