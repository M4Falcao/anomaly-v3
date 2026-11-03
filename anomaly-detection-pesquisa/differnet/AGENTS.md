# Instruções para agentes neste projeto

## 0. Regra obrigatória

Antes de propor ou executar decisões de método, treinamento, avaliação, interpretação de resultados ou mudanças no protocolo, consulte [`docs/contexto_e_decisoes_ia.md`](docs/contexto_e_decisoes_ia.md). Use a hierarquia de evidências e o checklist definidos lá. Toda decisão nova que afete método, dados, métricas, critérios de seleção ou interpretação deve ser acrescentada ao Registro de Decisões do documento (seção 11.1, próximo ID após o último `D-xx`), com justificativa, evidência, alternativas e verificação planejada. Não trate resultados exploratórios ou de holdout como confirmação independente.

Outras regras:
- Não iniciar treino real, grid longo ou inferência completa sem pedido explícito do usuário. Dry-run, `--help`, `--print_recipe`, `-PlanOnly` e testes sintéticos são livres.
- Não sobrescrever rodadas históricas em `resultado_analise_final/`; toda rodada nova recebe pasta com timestamp.
- Não apagar checkpoints, `mlruns/`, `mlartifacts/` nem resultados antigos.
- Responder ao usuário em português.

## 1. O que é o projeto

Pesquisa de detecção e **localização por pixel** de anomalias em imagens aéreas de componentes de linhas de transmissão (dataset **INSPLAD-seg**).

Pipeline em dois estágios:
1. **Nível de imagem — SEDifferNet:** DifferNet (Rudolph et al., WACV 2021) com backbone AlexNet + blocos Squeeze-and-Excitation e normalizing flow (FrEIA) sobre features multiescala. Um checkpoint por classe. Treino via `main.py` + `config.py`.
2. **Nível de pixel — NF head / "CFLOW pixel head":** normalizing flow local condicionado por positional encoding (64-d), treinado sobre ativações espaciais **congeladas** do SEDifferNet nos níveis L1/L2/L3 (64/192/256 canais), redimensionadas para grade 96×96. Produz mapas de NLL por posição, agregados em heatmap.

Pergunta central: com o encoder SEDifferNet de cada classe fixo, como escolhas de treino da NF head e de inferência/pós-processamento afetam Pixel AUROC/AUPRO, e o quanto isso depende da classe.

Fora de escopo (decisões D-01/D-02): descongelar o SEDifferNet nas ablações; reproduzir o CFLOW-AD end-to-end (WRN50-2, decoder oficial, loss/amostragem oficiais). Dizer "score CFLOW-AD aplicado à head local", nunca "reprodução do CFLOW-AD".

## 2. Ambiente

- Windows, **PowerShell 5.1**; GPU RTX 4060 Laptop 8 GB; 16 GB RAM (grids grandes em yoke paginam).
- Python da pesquisa: `C:\Users\teo-s\.conda\envs\anomaly_attentideffernet\python.exe` (Python 3.10, PyTorch 2.5.1+cu121). Ativar com `conda activate anomaly_attentideffernet`; o env base não tem `fightingcv_attention`.
- Rodar sempre a partir de `anomaly-detection-pesquisa/differnet/`.
- Instalação de pacotes: a ferramenta integrada de instalação mira o Python errado (`c:\python314`). Usar `<python do env> -m pip install ...`.
- Snippets Pylance exigem `workspaceRoot` explícito (`c:\Users\teo-s\Documents\GitHub\anomaly-v3`).
- `get_errors` pode dar falso negativo; confirmar com `python -m py_compile`.

### Armadilhas de PowerShell
- Nunca editar `.py` com `Get-Content | Set-Content`: o PS 5.1 decodifica como CP1252 e corrompe caracteres (`═`, acentos). Usar ferramentas de edição ou Python.
- Ler Markdown em português com `-Encoding UTF8`.
- Não encadear `&&`; usar `;`. Não fazer pipe direto de `foreach`; coletar em variável.
- Para achar o pico em `history.json`, comparar por índice numericamente; `Array.IndexOf` com o máximo de `Measure-Object` pode retornar -1.

## 3. Dados

Raiz: `C:\Users\teo-s\Documents\GitHub\anomaly-detection-dataset\`
- `insplad-seg\insplad-seg\` — **dataset original, usado em todas as ablações e números publicados.**
- `insplad-seg-no-back-treino\` — versão com background removido (pixels de fundo pretos), gerada com o pipeline de `../yolo-segmentation/`.
- Também existem `cardd`, `cars`, `mvtec` (fora do escopo atual).

Estrutura MVTec: `<classe>/train/good`, `<classe>/test/<good|defeito>`, `<classe>/ground_truth/<defeito>`.

Classes (`ALL_CLASSES` em `core/eval_pipeline.py`):

| Classe | Original train/test | Sem fundo train/test/GT | Fundo preto (sem fundo) |
|---|---|---|---|
| glass-insulator | 1104 / 678 | 1092 / 649 / 88 | ~74% |
| lightning-rod-suspension | 462 / 163 | 388 / 135 / 46 | ~85% |
| polymer-insulator-upper-shackle | 935 / 317 | 879 / 283 / 82 | ~91% |
| vari-grip | 477 / 154 | 474 / 154 / 40 | ~79% |
| yoke-suspension | 4834 / 1253 | 4570 / 1199 / 46 | ~81% |

**ATENÇÃO:** `config.py` aponta `dataset_path` para `insplad-seg-no-back-treino` e `modelname = "se_differnet_no_bg"`. Para reproduzir ou comparar com as ablações, passar **explicitamente** `--dataset` / `-DatasetRoot` com o `insplad-seg\insplad-seg`. Os dois datasets têm inventários diferentes (ex.: lightning perdeu 28 imagens `good` no teste), portanto **o split val/holdout muda e as métricas não são diretamente comparáveis**.

### Experimento sem background (2026-10, em diagnóstico)
SEDifferNet e NF head treinados no dataset sem fundo tiveram AUROC de imagem e de pixel **menores**. Hipóteses abertas, nenhuma verificada:
1. Conjuntos de teste diferentes (imagens descartadas na segmentação).
2. Contorno artificial do recorte (borda objeto/preto) gera falsos positivos e domina a variação aprendida.
3. 74–91% da imagem é preto constante: dilui o sinal global do SEDifferNet, deixa canais quase constantes e desajusta a normalização ImageNet.
4. Erros do YOLO podem apagar parte do defeito ou da peça (crítico em defeitos de "peça ausente").
5. Receitas, níveis, σ e orçamentos de época foram calibrados com fundo.
6. Possível avaliação treino-sem-fundo × teste-com-fundo (distribution shift).

Diagnósticos sugeridos: avaliar ambos os modelos na interseção de imagens; Pixel AUROC só dentro da máscara do objeto; fração do GT em área preta; heatmaps no contorno. Alternativas: crop por bounding box, fundo borrado/cor média, mascarar só o score ou a loss. Se virar mudança de método, registrar como novo `D-xx`.

## 4. Arquitetura do código (`differnet/`)

- `config.py` — configuração global do SEDifferNet (sem argparse; scripts sobrescrevem `c.class_name` etc.). img 448×448, `n_transforms=4`, `n_transforms_test=64`, rotações on, normalização ImageNet, seed 42.
- `main.py` — treino do SEDifferNet de uma classe; `train_insplad.py` + `run_treinamento_insplad.ps1` treinam todas as classes. `scripts/train/train_insplad_pipeline.py` está **vazio**.
- `core/`
  - `model.py`, `freia_funcs.py`, `train.py`, `utils.py` (`load_datasets`, `make_dataloaders`), `multi_transform_loader.py`, `localization.py`, `xai.py`, `mlflow_utils.py`, `paths.py`.
  - `cflow.py` — NF head: `pad_mode`, `feat_pool`, modos de score (`raw`, `per_level_std`, `per_level_minmax`, `per_level_prob`, `cflow_original`), `aggregate_nll`, `compute_positional_stats`, `compute_feature_stats`/`apply_feature_stats` (slots `None` para níveis não normalizados), `select_levels`, `TTA_MODES`, `flip_tensor`, `nll_with_tta`, `TRAIN_SCORE_MODE='raw'`.
  - `eval_pipeline.py` — base comum de avaliação: `DEVICE`, `ALL_CLASSES`, `find_se_checkpoint`, `find_nf_head_checkpoint`, `compute_aupro_cflow` (oficial, global), `aupro_rd_legacy`, `compute_best_f1`, `binary_metrics`, `confusion_at`, `build_eval_datasets`, `PixelHead`, `resolve_score_levels`, `PER_CLASS_SCORE_LEVELS`, `denormalize`.
  - `eval_common.py` — utilidades compartilhadas antigas.
- `scripts/train/`
  - `pixel_train_from_pretrained.py` — treina a NF head sobre um SEDifferNet congelado. `--recipe {custom,best_per_class,general,baseline}` (default `custom`), `--print_recipe`, `--auroc_only`, `--patience`, `--eval_aupro/--no_eval_aupro`, `--split_seed`, `--levels`, `--feat_norm`, `--feat_norm_levels`, `--no_rotation`, `--pad_mode`, `--arch`. Salva `best_pixel_auroc.pt`, `best_composite.pt`, `history.json`, `recipe.json`, `final_results.csv`, monitor 2×2; grava `selection_validation_paths`, `training_recipe`, `inference_recipe`, `inference_level_stats`.
  - `train_all_nf_heads.py` + `run_train_all_nf_heads.ps1` (também atalho na raiz) — lote de todas as classes. Default `general`; `--checkpoint_dir`/`-CheckpointDir` busca recursivamente **exatamente um** encoder SEDifferNet por classe (ausência/ambiguidade aborta no preflight); `--dry_run`, `--skip_existing`, `--evaluate_after {pixel,full,none}` (default `none`). Saída `resultado_analise_final/treino_nf_heads/run_<ts>/` com `best_models/` e `training_summary.csv/.md` (colunas `val_*` e `holdout_*` separadas, linha `MACRO_AVERAGE` só com classes `SUCCESS`).
- `scripts/eval/`
  - `ablation_nf_head.py` — grid pós-hoc (pad/tta/score/levels/pos_calib/sigma/margin/align_corners/image_score) com cache de NLL, split val/holdout, estudos reference/selection/OFAT/ladder, bootstrap pareado estratificado, `--stage {posthoc,retrain,all}`, `TRAIN_VARIANTS`, `--auroc_only`, `--select_metric`.
  - `run_nf_head_ablation.ps1` — launcher (`-PlanOnly`, `-Seeds` com uma seed só, `-AurocOnlyClasses`).
  - `consolidate_nf_head_ablation.py` — tabelas da análise consolidada.
  - `evaluate_image_level.py` (só SEDifferNet), `evaluate_pixel_level.py` (só NF head; `--sweep` é muito lento, usar `--limit`), `evaluate_full_pipeline.py` (imagem + pixel; reexporta funções de `core.eval_pipeline` — `ablation_nf_head` e `evaluate_cflow_no_smoothing` importam dele, não quebrar).
  - XAI/CAM: `evaluate_cam_xai_metrics.py`, `evaluate_cam_sanity_check.py`, `visualize_cam_metric_steps.py`; outros: `evaluate_model_roc.py`, `test_model_grad_maps.py`, `test_single_checkpoint.py`, `sweep_checkpoints.py`, `score_image_folder.py`, `ablation_image_level.py`, `ablation_full_pipeline.py`.
- `scripts/tools/` — `export_mlflow.py`, `read_checkpoint_auroc.py`, `batch_evaluate_cam_metrics.py`, `print_output_log.py`, `generate_dummy_gt.py`.
- `scripts/analysis/` — `analyze_class.py`, `smoke_test_baseline.py`.
- `avaliacao_completa_pixel_heads.ipynb` — avaliação descritiva do teste completo (D-18): lê receita do checkpoint, stride 1, margem 0, mapas em memmap, backend `histogram` (default, aproximado) ou `exact` (sklearn, muita RAM). Heatmaps, ROC/PR, F1/IoU/Dice, rankings FP/FN, latência, exportação CSV/Excel/JSON/HTML. Default: `treino_nf_heads/run_20261025_181523`. Reler antes de editar (o usuário edita).
- `legacy/` — código antigo; não usar para resultados novos.

### Artefatos e pastas de saída
- `checkpoints/run_<ts>/` — checkpoints do SEDifferNet por treino.
- `final_models/` — `SEDiffernet/`, `SEDiffernet_noBg/`, `CBAMDiffernet/`, `Differnet/`, `NF Head/`.
- `resultado_analise_final/` — `ablacao_nf_head/` (rodadas `run_full_20261007_123558`, `run_full_20261010_195638`, `analise_consolidada_20261013/`), `treino_nf_heads/`, `imagem/`, `pixel_notebook/`, `ablacao_image_level/`, `ablacao_full_pipeline/`, `treinamento_insplad/`, `attdiffernet/`.
- `mlruns/`, `mlflow.db`, `mlartifacts/` — MLflow.
- `results*/`, `pixel_pipeline*/`, `pixel_head_runs/` e o arquivo `notas` na raiz do workspace são históricos; podem conter comandos com nomes antigos de scripts (de propósito).

## 5. Nomenclatura

- **SEDifferNet / encoder / backbone:** modelo de imagem por classe; congelado nas ablações. CBAM e SE têm as mesmas chaves de `state_dict`; o tipo é inferido pelo nome do arquivo.
- **NF head / pixel head / CFLOW head:** flow local do nível de pixel.
- **Níveis:** L1/L2/L3 = índices `0/1/2`; strings como `02` = L1+L3, `012` = todos.
- **Variantes de treino** (`TRAIN_VARIANTS`): `baseline`, `featnorm`, `featnorm_l1_only`, `clamp19`, `featnorm_clamp19` (+ `*_02`, `*_clamp19` derivados).
- **Receita de inferência:** pad (`reflect`, 112), TTA (`none`/`hflip`/`flips`), score (`raw`, `per_level_std`, `cflow_original`, ...), níveis, σ do gaussiano, `align_corners`/up, `pos_calib`, margem, image score (`max`/`topk`).
- **Presets** (`--recipe`): `general` (treino baseline; inferência `per_level_std` + reflect112 + flips + σ8 + up=True; níveis glass 02, lightning 012, polymer 12, vari 0, yoke 012; épocas 40/40/30/40/12, avaliação a cada 5/5/3/5/2), `best_per_class` (variante vencedora histórica por classe), `baseline`, `custom`.
- **val / holdout:** partições estratificadas do `test` (`val_fraction=0.4`, `split_seed=42`). **train seed** é separada do **split seed**.
- **`holdout_clean`:** True só quando o checkpoint prova (por `selection_validation_paths`) que a época não foi escolhida no holdout.
- **`aupro`** = AUPRO global CFLOW-AD; **`aupro_rd_legacy`** = antigo por imagem (não comparável).
- **AUROC-only:** modo D-14; métricas omitidas ficam NaN, nunca zero.

## 6. Protocolo vigente (resumo; detalhes no documento de contexto)

- Treino só com `train/good`; máscaras nunca entram no gradiente.
- Uma train seed 42 e split seed 42 (D-13). Bootstrap por imagem não mede variância entre seeds.
- Época escolhida por Pixel AUROC em `raw` na validação (D-04); receita pós-hoc escolhida na validação; holdout nunca escolhe nada.
- `cflow_original` é transdutivo (máximos da coorte): calcular na coorte inteira, val e holdout separados.
- Margem 0 por padrão (D-08).
- AUROC in-training (pad 0, sem TTA, σ menor, todos os níveis) **não** é comparável à receita de deploy (diferença ~+0.1).
- Qualquer avaliação de pixel deve forçar `testset.transform` determinístico e `testset.n_transforms=1`: `load_datasets()` dá ao teste o transform de treino (RandomRotation), o que desalinha máscaras.

## 7. Achados consolidados (fonte: `docs/analise_cientifica_ablacao_nf_head_5classes.md`, D-15/D-16)

- Média holdout de Pixel AUROC nas 5 classes: 0.8405 (legado) → 0.9208 (+0.080); ~78% do ganho vem da receita de inferência, ~22% da variante de treino.
- Min-max por imagem prejudica (−0.075); `prob` −0.113; calibração posicional −0.025; σ8 +0.016 (25/25 heads); flips +0.005; pad reflect +0.002.
- `raw` ≈ `per_level_std` ≈ `cflow_original` em AUROC; `cflow_original` melhor em AUPRO, mas transdutivo.
- Níveis ótimos dependem da classe **e da head**: L1+L3 universal foi refutado (regret 0.044 em polymer); `012` é minimax. vari-grip: L3 é o pior nível (L1 sozinho 0.9191 vs fundido 0.8561).
- Nenhuma variante de treino vence em média (Friedman p = 0.72). featnorm ajuda glass e piora yoke.
- Receita única custa ~0.026 de AUROC versus receita por head.
- Overfitting relevante só em yoke (pico cedo, −0.030 até a época final).
- 14 linhas com AUPRO = 0 (heads clamp19, incl. polymer featnorm_clamp19): provável colapso da grade de limiares, **não verificado**.
- Limitações: uma seed, holdout já inspecionado (não é independente), AUPRO ausente em vari/yoke, n = 5 classes.
- Comparação externa: RD++ (`../rd++/`) ganha em AUPRO em 3 de 4 classes.

## 8. Métricas

- Pixel AUROC: ranking global de pixels; dominado por pixels normais, pode ficar alto com segmentação ruim.
- Pixel AP: área sob precisão×recall; sensível à classe rara (defeitos ≈ 2–3% da área); mais informativa que AUROC para localização.
- Pixel accuracy: dominada por pixels normais (≈97% só prevendo "normal"); não usar como métrica principal.
- AUPRO: sobreposição por região até FPR 0.3; usar `compute_aupro_cflow`.
- F1/IoU/Dice dependem de limiar; limiar escolhido no teste é oráculo, declarar.

## 9. Armadilhas conhecidas

- pandas: `g.pad` é o método `DataFrame.pad`; usar `g['pad']`. Documentos usam "−" unicode; normalizar antes de converter números.
- `evaluate_pixel_level.py --sweep` leva ~43 min por classe.
- TTA flips custa 4× (≈12.7 FPS vs ≈45 sem TTA); para latência, usar `--tta none`.
- `evaluate_full_pipeline.py` não tem `--disable_patchcore` (é flag do treinador). Presets desligam PatchCore.
- Avaliadores externos **não** leem automaticamente `inference_recipe` do checkpoint; passar a receita explicitamente (o notebook D-18 lê).
- `--epochs` global no lote sobrescreve os orçamentos por classe da receita (o script avisa).
- Em testes que mockam `subprocess.run`, pré-carregar trainer/MLflow antes do mock.
- Notebook tools não gravam `metadata.language`; normalizar com `nbformat` ao final.

## 10. Repositório externo ao `differnet/`

`anomaly-v3/anomaly-detection-pesquisa/` também contém `rd++/` (Reverse Distillation++, baseline de comparação), `simplenet/` e `yolo-segmentation/` (extração de máscaras e remoção de fundo). `mlruns/` e `metrics_history.txt` na raiz do workspace são de execuções antigas.

## 11. Comandos úteis

```powershell
# consultar preset sem treinar
python scripts/train/pixel_train_from_pretrained.py --class_name vari-grip --recipe general `
	--dataset 'C:\Users\teo-s\Documents\GitHub\anomaly-detection-dataset\insplad-seg\insplad-seg' --print_recipe

# lote de NF heads (plano apenas)
.\scripts\train\run_train_all_nf_heads.ps1 -CheckpointDir <pasta_dos_SEDifferNet> `
	-DatasetRoot 'C:\Users\teo-s\Documents\GitHub\anomaly-detection-dataset\insplad-seg\insplad-seg' -Recipe general -DryRun

# plano da ablação
.\scripts\eval\run_nf_head_ablation.ps1 -DatasetRoot <dataset> `
	-Python 'C:\Users\teo-s\.conda\envs\anomaly_attentideffernet\python.exe' -Seeds 42 -SplitSeed 42 -PlanOnly
```
