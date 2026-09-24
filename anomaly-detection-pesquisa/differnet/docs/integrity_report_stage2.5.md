# Relatório de Verificação de Integridade Acadêmica (Stage 2.5 — Rodada 2)

**Data:** 2026-09-24  
**Documentos Auditados:** `paper_ablation_nf_head.tex` e `artigo_ablation_cflow_differnet.md` (Versão Expandida)  
**Agente Responsável:** `integrity_verification_agent` (Protocolo Zero-Tolerance)  
**Veredito Geral:** **PASS (100% de conformidade)**

---

## 1. Verificação de Referências Bibliográficas (Fase A)
Auditoria exaustiva das 12 referências constantes no manuscrito:

| # | Chave de Citação | Autores e Ano | Título e Veículo | Status Oficial | Veredito |
|---|---|---|---|---|---|
| [1] | `rudolph2021differnet` | Rudolph, Wandt, Rosenhahn (2021) | *Same Same But DifferNet: Semi-Supervised Defect Detection with Normalizing Flows*, WACV 2021 | DOI: 10.1109/WACV48630.2021.00195 | **VERIFIED** |
| [2] | `gudovskiy2022cflow` | Gudovskiy, Ishizaka, Kozuka (2022) | *CFLOW-AD: Real-Time Unsupervised Anomaly Detection with Localization via Conditional Normalizing Flows*, WACV 2022 | DOI: 10.1109/WACV51458.2022.00188 | **VERIFIED** |
| [3] | `rudolph2022csflow` | Rudolph, Wehrbein, Rosenhahn, Wandt (2022) | *Fully Convolutional Cross-Scale-Flows for Image-based Defect Detection*, WACV 2022 | DOI: 10.1109/WACV51458.2022.00116 | **VERIFIED** |
| [4] | `defard2021padim` | Defard, Setkov, Loesch, Audigier (2021) | *PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization*, ICPR Workshops 2021 | Springer LNCS 12664, pp. 475–489 | **VERIFIED** |
| [5] | `roth2022patchcore` | Roth et al. (2022) | *Towards Total Recall in Industrial Anomaly Detection*, CVPR 2022 | DOI: 10.1109/CVPR52688.2022.01392 | **VERIFIED** |
| [6] | `deng2022rd` | Deng & Li (2022) | *Anomaly Detection via Reverse Distillation from One-Class Embedding*, CVPR 2022 | DOI: 10.1109/CVPR52688.2022.00951 | **VERIFIED** |
| [7] | `vieira2023insplad` | Vieira-e-Silva et al. (2023) | *InsPLAD: A Dataset and Benchmark for Power Line Asset Inspection in UAV Images*, IJRS 2023 | DOI: 10.1080/01431161.2023.2282276 | **VERIFIED** |
| [8] | `zhou2024msflow` | Zhou et al. (2024) | *MSFlow: Multiscale Flow-based Framework for Unsupervised Anomaly Detection*, IEEE TNNLS 2024 | DOI: 10.1109/TNNLS.2024.3368297 | **VERIFIED** |
| [9] | `alsallakh2021mind` | Alsallakh et al. (2021) | *Mind the Pad -- CNNs can Develop Blind Spots*, ICLR 2021 | OpenReview: ICLR 2021 | **VERIFIED** |
| [10] | `islam2020much` | Islam, Jia, Bruce (2020) | *How Much Position Information Do Convolutional Neural Networks Encode?*, ICLR 2020 | OpenReview: ICLR 2020 | **VERIFIED** |
| [11] | `geirhos2019imagenet` | Geirhos et al. (2019) | *ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness*, ICLR 2019 | OpenReview: ICLR 2019 | **VERIFIED** |
| [12] | `kayhan2020translation` | Kayhan & van Gemert (2020) | *On Translation Invariance in CNNs: Convolutional Layers Can Exploit Absolute Spatial Location*, CVPR 2020 | DOI: 10.1109/CVPR42600.2020.01429 | **VERIFIED** |

* **Total de Referências:** 12 / 12 verificadas com sucesso.
* **Citações Fantasma / Órfãs:** Zero. Todas as 12 são devidamente citadas no texto.

---

## 2. Auditoria de Dados e Métricas Empíricas
Confronto exaustivo da Grande Tabela Multidimensional (Tabela 3) contra `plano_ablation_nf_head.md`:
* **Cadeia Incremental:** 100% exata com Seção 6.8.
* **Efeitos Pareados:** 100% exatos com Seção 6.8.
* **Desempenho por Nível e Regret:** 100% exato com Seção 6.7 e 6.9.
* **Overfitting de Épocas:** 100% exato com Seção 6.9.3.
* **Resultados Finais no Held-out:** 100% exatos com Seção 0.

**Veredito:** **PASS (Avançar para a Fase 2 — Revisão por Pares Stage 3).**
