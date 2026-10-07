# Relatório Completo de Treinamento e Análise - DifferNet no Dataset InSPLAD

**Data de Execução:** 06/10/2026 23:01:30  
**Identificador do Run:** `run_20261006_230110`  
**Modo de Treinamento:** `eval_only`  
**Dispositivo Utilizado:** `cuda`  

## 1. Resumo Executivo

Este documento apresenta a análise integral de desempenho do pipeline de detecção não supervisionada e localização de anomalias baseado em **DifferNet** e **SEDifferNet** acoplado à cabeça espacial **CFLOW-AD** (Conditional Normalizing Flows) sobre todas as classes do dataset de inspeção de linhas elétricas **InSPLAD**.

- **Média Global Image AUROC:** **88.00%**
- **Média Global Pixel AUROC:** **93.13%**
- **Média Global AUPRO:** **68.13%**
- **F1-Score Médio de Detecção:** **91.66%**

## 2. Tabela Geral de Desempenho por Classe

| Classe | Amostras Teste | Img AUROC | Img AP | Img F1 | Pix AUROC | Pix AP | AUPRO | Precisão | Recall | Especificidade |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| glass-insulator | 10 | 0.7600 | 0.7962 | 0.8333 | 0.9695 | 0.0363 | 0.6722 | 0.7143 | 1.0000 | 0.6000 |
| vari-grip | 10 | 1.0000 | 1.0000 | 1.0000 | 0.8931 | 0.1266 | 0.6904 | 1.0000 | 1.0000 | 1.0000 |
| **MÉDIA GLOBAL** | 20 | **0.8800** | 0.8981 | 0.9166 | **0.9313** | 0.0814 | 0.6813 | 0.8572 | 1.0000 | 0.8000 |

## 3. Matrizes de Confusão e Contagem de Erros

| Classe | Verdadeiros Positivos (TP) | Falsos Positivos (FP) | Verdadeiros Negativos (TN) | Falsos Negativos (FN) |
| :--- | :---: | :---: | :---: | :---: |
| glass-insulator | 5 | 2 | 3 | 0 |
| vari-grip | 5 | 0 | 5 | 0 |

## 4. Gráfico Comparativo Multiclasse

![Comparativo Multiclasse](cross_class_summary/cross_class_metrics_bar.png)

## 5. Análise Qualitativa e Particularidades das Classes

### 5.1 glass-insulator

- **Detecção (Imagem):** AUROC de `0.7600`, com AP de `0.7962` e F1 de `0.8333`.
- **Localização (Pixel):** AUROC de `0.9695`, AUPRO de `0.6722`.
- **Diagnóstico Operacional:** Classe desafiadora com sobreposição moderada de features de fundo (vegetação/estruturas metálicas complexas) e ruído espacial.

- **Curva ROC e Distribuição:** [Ver Gráfico ROC](glass-insulator/analysis/plots/roc_curves.png) | [Ver Histograma de Scores](glass-insulator/analysis/plots/score_distribution.png)
- **Galeria de Inspeção (4 Painéis):** [Visualizar Amostras com Heatmap](glass-insulator/analysis/plots/inspection_gallery.png)

### 5.2 vari-grip

- **Detecção (Imagem):** AUROC de `1.0000`, com AP de `1.0000` e F1 de `1.0000`.
- **Localização (Pixel):** AUROC de `0.8931`, AUPRO de `0.6904`.
- **Diagnóstico Operacional:** Excelente separação da densidade probabilística de amostras anômalas contra sadias. A arquitetura apresenta alta sensibilidade a defeitos com baixíssima taxa de falsos alarmes.

- **Curva ROC e Distribuição:** [Ver Gráfico ROC](vari-grip/analysis/plots/roc_curves.png) | [Ver Histograma de Scores](vari-grip/analysis/plots/score_distribution.png)
- **Galeria de Inspeção (4 Painéis):** [Visualizar Amostras com Heatmap](vari-grip/analysis/plots/inspection_gallery.png)

## 6. Diretrizes e Recomendações para Implantação em Campo

1. **Filtragem Hierárquica em Dois Estágios:** Utilizar primeiro o score de imagem do SEDifferNet como gatilho de triagem rápida (rejeitando imagens sadias sem custo computacional adicional), e disparar o mapa CFLOW apenas nas imagens classificadas como anômalas.
2. **Suavização Gaussiana (Sigma 8):** O pós-processamento por filtro Gaussiano com sigma=8 elimina ruídos pontuais em alta frequência e estabiliza o AUPRO em componentes isoladores e para-raios.
3. **Padding Reflexivo (112 px):** Fundamental para mitigar artefatos espúrios nas bordas gerados por convoluções convolucionais em imagens de alta resolução.

---
*Relatório gerado automaticamente pelo pipeline unificado DifferNet InSPLAD.*  
