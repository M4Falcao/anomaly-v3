# Relatorio de achados parciais da ablacao NF head

**Rodada:** `run_full_20261007_123558`.  
**Data de corte:** 2026-10-10, 19:38, horario local da maquina.  
**Origem:** execucao do launcher `run_nf_head_ablation.ps1` no terminal indicado pelo usuario como `/6`.  
**Natureza:** analise exploratoria de artefatos salvos; nao e uma nova execucao experimental nem uma comparacao com o CFLOW-AD end-to-end.

> **Atualizacao 2026-10-13:** vari-grip e yoke foram concluidas em `run_full_20261010_195638` ([relatorio](relatorio_achados_run_full_20261010_195638.md)). A analise das 25 heads, com tabela de ablacao, testes estatisticos, receitas por classe e receita geral, esta na [analise cientifica consolidada](analise_cientifica_ablacao_nf_head_5classes.md). O AUPRO zero de polymer foi rastreado ate 14 linhas de holdout em heads com clamp 1,9 e e tratado la como provavel colapso da grade linear de limiares da metrica (secao 3.9).

## 1. Escopo e rastreabilidade

O sistema estudado e uma NF head pixel-level local sobre features AlexNet-SE de um SEDifferNet congelado por classe. O experimento altera a receita de treinamento da head e sua inferencia, sem retreinar o SEDifferNet. `cflow_original` designa a agregacao de score inspirada no caminho oficial sobre as NLLs locais, nao uma reproducao da arquitetura e treinamento completos do CFLOW-AD.

Fontes primarias desta analise, relativas a `../resultado_analise_final/ablacao_nf_head/run_full_20261007_123558/`:

- [Resumo de glass](../resultado_analise_final/ablacao_nf_head/run_full_20261007_123558/glass-insulator/ablation_summary.csv).
- [Resumo de lightning](../resultado_analise_final/ablacao_nf_head/run_full_20261007_123558/lightning-rod-suspension/ablation_summary.csv).
- [Resumo de polymer](../resultado_analise_final/ablacao_nf_head/run_full_20261007_123558/polymer-insulator-upper-shackle/ablation_summary.csv).
- `config.txt`, `retrain_variants.csv`, `split_manifest.json`, `history.json`, `final_results.csv`, `grid_val.csv`, `holdout_test_top.csv` e `summary.txt` nos subdiretorios de cada classe/head.
- [Contexto e decisoes](contexto_e_decisoes_ia.md), [script de ablacao](../scripts/eval/ablation_nf_head.py) e [metricas compartilhadas](../core/eval_pipeline.py), usados para interpretar os artefatos, nao para substituir resultados ausentes.

O terminal encerrou com codigo 1. Na consulta, nao havia processo Python ativo. Nao foi recuperado um traceback ou log que permitisse atribuir a causa: interrupcao manual, erro e falha externa nao podem ser distinguidos apenas pelo exit code. Nenhum treino foi iniciado, retomado ou interrompido para elaborar este documento.

Comando registrado no terminal, executado a partir de `differnet/`:

```powershell
.\scripts\eval\run_nf_head_ablation.ps1 `
	-DatasetRoot 'C:\Users\teo-s\Documents\GitHub\anomaly-detection-dataset\insplad-seg\insplad-seg' `
	-Python 'C:\Users\teo-s\.conda\envs\anomaly_attentideffernet\python.exe' `
	-Seeds 42 `
	-SplitSeed 42
```

O dataset efetivo e o caminho desse comando, nao necessariamente o default de `config.py`. Esse comando documenta a origem, mas nao deve ser repetido apenas para reproduzir as tabelas: ele inicia retreinos longos. Para reproduzir a analise numerica, usar os artefatos existentes.

## 2. Cobertura efetivamente disponivel

| Classe | Historicos de treino | Relatorios finais de treino | Grids completos | Holdouts completos | Situacao |
|---|---:|---:|---:|---:|---|
| glass-insulator | 5 | 5 | 5 | 5 | Cinco variantes com avaliacao completa |
| lightning-rod-suspension | 5 | 5 | 5 | 5 | Cinco variantes com avaliacao completa |
| polymer-insulator-upper-shackle | 5 | 5 | 5 | 5 | Cinco variantes com avaliacao completa |
| vari-grip | 2 | 1 | 0 | 0 | Baseline terminou; featnorm tem historico ate avaliacao 10/40 |
| yoke-suspension | 0 | 0 | 0 | 0 | Sem artefatos nesta rodada |

Ha 15 comparacoes pos-hoc completas de tres classes, 16 relatorios finais de treinamento e um treinamento parcial. Nao confundir diretorio/checkpoint existente com treinamento ou pos-hoc concluido. A rodada planejada de cinco classes esta incompleta.

| Classe | Teste total | Validacao | Anomalas na validacao | Holdout | Anomalas no holdout |
|---|---:|---:|---:|---:|---:|
| glass-insulator | 678 | 271 | 35 | 407 | 52 |
| lightning-rod-suspension | 163 | 65 | 18 | 98 | 28 |
| polymer-insulator-upper-shackle | 317 | 127 | 33 | 190 | 49 |
| vari-grip | 154 | 62 | 16 | 92 | 24 |

Os tamanhos foram lidos dos manifestos, nao de estimativas da conversa. Todas as comparacoes concluidas usam uma unica seed de treino, 42, e split seed 42.

Os dez manifestos encontrados em cada classe concluida (treino e pos-hoc) concordam nos indices de validacao/holdout: uma assinatura por classe e nenhuma intersecao de indices. Em vari-grip, os dois manifestos tambem concordam. Isso e uma checagem de consistencia do split, nao uma verificacao de duplicatas visuais ou de imagens correlacionadas da mesma inspecao. Os registros de retreino das tres classes concluidas mostram retorno 0 para as 15 variantes.

## 3. Protocolo e como ler os numeros

1. A head aprende com treino normal; o checkpoint `best_pixel_auroc.pt` e selecionado pela Pixel AUROC de validacao com score `raw` e sigma 4. A melhor epoca nao e selecionada com `cflow_original`.
2. A receita pos-hoc e escolhida em validacao pela media `(Pixel AUROC + AUPRO)/2`. O grid usa stride 2; o holdout usa stride 1. O image AUROC atua como desempate no codigo.
3. A referencia `baseline_cflow_original` usa pad 0, sem TTA, niveis 012, sem calibracao posicional, `align_corners=True`, sigma 0, margem 0 e image score max, sobre a head em questao.
4. Comparar variantes sob essa mesma referencia aproxima o efeito do treinamento/selecionador de epoca. Comparar o top-1 de cada variante mede o sistema inteiro apos selecao pos-hoc; nao isola o efeito do treino.
5. O delta top-1 menos referencia da mesma head mede ganho de inferencia, nao ganho de trocar a head. Os ICs salvos se referem a esse par, nao a diferencas entre variantes.
6. `holdout_clean=True` nas 15 linhas confirma a verificacao local de caminhos da selecao de epoca da NF head. Nao certifica a origem/selecionador do backbone, a independencia entre imagens da mesma inspecao nem a ausencia de adaptacao humana apos observar resultados.
7. O holdout ja foi inspecionado nesta conversa. Novas decisoes motivadas por ele sao exploratorias; nao podem ser confirmadas independentemente repetindo o mesmo holdout.

`aupro` segue a implementacao global CFLOW-AD local: thresholds comuns, FPR sobre pixels normais de todas as imagens e sobreposicao media por regiao. `aupro_rd_legacy` e outra metrica. O AUPRO durante o treino tambem usa outra rotina, com thresholds por quantis de pixels normais. Nao transportar valores entre essas rotinas como se fossem identicos.

As heads salvas usam grid espacial 96, oito blocos, condicao PE64, hidden 256 e niveis 012. A baseline local tem clamp 0.5; `clamp19` altera-o para 1.9. `featnorm` ativa normalizacao de features em todos os niveis; `featnorm_l1_only` normaliza somente o indice 0; `featnorm_clamp19` combina os dois fatores. As imagens configuradas sao 448 x 448. Nada disso troca o encoder por WRN50-2 ou a implementacao local do flow pela arquitetura oficial completa.

## 4. Resultados pos-hoc selecionados em validacao

Valores de holdout do top-1 de validacao de cada head, sem escolher o top-N pelo holdout. AP esta na escala 0 a 1 e e reportada separadamente da AUROC.

| Classe | Variante | Epoca selecionada | Criterio na val | Pixel AUROC holdout | Pixel AP holdout | AUPRO holdout |
|---|---|---:|---:|---:|---:|---:|
| glass | baseline | 15 | 0.8960 | 0.9156 | 0.002817 | 0.7829 |
| glass | featnorm | 40 | 0.9053 | 0.9467 | 0.004570 | 0.8422 |
| glass | featnorm_l1_only | 10 | 0.8922 | 0.9094 | 0.002530 | 0.7895 |
| glass | clamp19 | 5 | 0.8877 | 0.9435 | 0.003202 | 0.8127 |
| glass | featnorm_clamp19 | 10 | 0.8900 | 0.9510 | 0.004688 | 0.8212 |
| lightning | baseline | 15 | 0.7851 | 0.8674 | 0.020822 | 0.5825 |
| lightning | featnorm | 35 | 0.8009 | 0.8605 | 0.022204 | 0.5630 |
| lightning | featnorm_l1_only | 40 | 0.7898 | 0.8686 | 0.021680 | 0.5825 |
| lightning | clamp19 | 30 | 0.8122 | 0.8965 | 0.028191 | 0.6442 |
| lightning | featnorm_clamp19 | 35 | 0.8022 | 0.8607 | 0.019896 | 0.5930 |
| polymer | baseline | 30 | 0.8042 | 0.8653 | 0.029652 | 0.6907 |
| polymer | featnorm | 15 | 0.8192 | 0.8764 | 0.031606 | 0.7131 |
| polymer | featnorm_l1_only | 30 | 0.8072 | 0.8682 | 0.030505 | 0.6996 |
| polymer | clamp19 | 12 | 0.8062 | 0.8672 | 0.030384 | 0.6979 |
| polymer | featnorm_clamp19 | 15 | 0.8206 | 0.8848 | 0.032730 | **0.0000: investigar** |

**Leitura por classe:**

- Glass: `featnorm` maximiza o criterio de validacao e o AUPRO observado; `featnorm_clamp19` tem a maior AUROC observada no holdout. A diferenca de AUROC entre elas e aproximadamente 0.0043, sem IC entre variantes salvo. Nao escolher uma delas por esse pequeno delta de holdout.
- Lightning: `clamp19` tem o maior criterio de validacao e os maiores AUROC/AUPRO de holdout entre as variantes. E a candidata mais forte desta classe nesta seed.
- Polymer: `featnorm_clamp19` tem o maior criterio de validacao e maior AUROC de holdout, mas o AUPRO registrado e zero. `featnorm` tem o maior AUPRO positivo observado. O zero impede concluir que a combinacao e a melhor para localizacao; a discrepancia precisa ser auditada.
- Em glass, a AUROC alta acompanha AP muito baixa. AUROC sozinha nao demonstra segmentacao utilizavel em um limiar operacional, especialmente com anomalias raras.

### 4.1 Receitas selecionadas

Todos os top-1 usam margem 0 e nenhuma calibracao posicional. `up` significa `align_corners`; `std` significa `per_level_std`. Indices 0/1/2 correspondem a L1/L2/L3; portanto, `02` remove L2, enquanto `012` conserva os tres niveis. Todas as heads desta tabela foram treinadas com 012: a escolha abaixo e de inferencia, nao retreino de uma head de dois niveis.

| Classe | Variante | Pad / modo | TTA | Score | Niveis | up | Sigma |
|---|---|---|---|---|---|---|---:|
| glass | baseline | 112 / reflect | flips | std | 02 | False | 0 |
| glass | featnorm | 112 / reflect | flips | std | 02 | False | 2 |
| glass | featnorm_l1_only | 112 / reflect | flips | raw | 02 | False | 2 |
| glass | clamp19 | 112 / reflect | hflip | std | 02 | False | 4 |
| glass | featnorm_clamp19 | 112 / reflect | flips | std | 02 | False | 4 |
| lightning | baseline | 112 / replicate | flips | std | 02 | True | 12 |
| lightning | featnorm | 112 / replicate | flips | raw | 02 | True | 8 |
| lightning | featnorm_l1_only | 112 / replicate | flips | std | 02 | True | 12 |
| lightning | clamp19 | 112 / replicate | flips | cflow_original | 012 | True | 8 |
| lightning | featnorm_clamp19 | 112 / replicate | flips | cflow_original | 02 | False | 12 |
| polymer | baseline | 0 / inerte | flips | std | 012 | True | 2 |
| polymer | featnorm | 112 / reflect | flips | std | 012 | False | 2 |
| polymer | featnorm_l1_only | 0 / inerte | flips | std | 012 | True | 2 |
| polymer | clamp19 | 112 / replicate | flips | std | 012 | False | 4 |
| polymer | featnorm_clamp19 | 112 / replicate | hflip | std | 012 | False | 4 |

O image score selecionado foi `topk` em 14 heads e `max` em polymer/`featnorm_clamp19`. Essa escolha muda a agregacao por imagem, nao o mapa nem suas metricas de pixel. Padding nao e universal: polymer tem dois vencedores com pad 0. Reflect domina as selecoes de glass; replicate domina lightning. O sigma escolhido tambem depende da classe.

### 4.2 Ganho de pos-hoc com a head fixa

Referencia: `baseline_cflow_original` sobre a MESMA head de cada linha. Deltas sao absolutos, nao percentuais relativos; +0.01 equivale a um ponto percentual. IC 95% de delta AUROC: bootstrap pareado de imagens, 1.000 reamostragens, aproximacao por histogramas. Nao ha IC de delta AUPRO salvo.

| Classe | Variante | AUROC referencia | AUPRO referencia | Delta AUROC top-1 | IC 95% delta AUROC | Delta AUPRO top-1 |
|---|---|---:|---:|---:|---|---:|
| glass | baseline | 0.8761 | 0.6710 | +0.0395 | [+0.0292, +0.0500] | +0.1118 |
| glass | featnorm | 0.9093 | 0.6582 | +0.0374 | [+0.0291, +0.0477] | +0.1839 |
| glass | featnorm_l1_only | 0.8777 | 0.6011 | +0.0317 | [+0.0208, +0.0458] | +0.1884 |
| glass | clamp19 | 0.8648 | 0.4465 | +0.0786 | [+0.0573, +0.1021] | +0.3662 |
| glass | featnorm_clamp19 | 0.8550 | 0.4157 | +0.0960 | [+0.0748, +0.1200] | +0.4054 |
| lightning | baseline | 0.7940 | 0.4237 | +0.0734 | [+0.0591, +0.0873] | +0.1588 |
| lightning | featnorm | 0.8224 | 0.5042 | +0.0381 | [+0.0249, +0.0513] | +0.0589 |
| lightning | featnorm_l1_only | 0.7970 | 0.4537 | +0.0716 | [+0.0565, +0.0867] | +0.1288 |
| lightning | clamp19 | 0.8566 | 0.5468 | +0.0400 | [+0.0341, +0.0466] | +0.0975 |
| lightning | featnorm_clamp19 | 0.8629 | 0.5612 | -0.0022 | [-0.0200, +0.0158] | +0.0318 |
| polymer | baseline | 0.8461 | 0.6561 | +0.0192 | [+0.0129, +0.0252] | +0.0346 |
| polymer | featnorm | 0.8523 | 0.6608 | +0.0242 | [+0.0181, +0.0313] | +0.0523 |
| polymer | featnorm_l1_only | 0.8311 | 0.6446 | +0.0371 | [+0.0265, +0.0479] | +0.0550 |
| polymer | clamp19 | 0.8553 | 0.6138 | +0.0120 | [+0.0008, +0.0224] | +0.0841 |
| polymer | featnorm_clamp19 | 0.8568 | 0.6224 | +0.0281 | [+0.0177, +0.0385] | -0.6224 |

Em 14/15 heads, o delta AUROC e positivo e seu IC marginal fica acima de zero. AUPRO melhora em 14/15, mas falha fortemente na linha sinalizada de polymer. Esses ganhos demonstram potencial da receita de inferencia sobre as heads locais; nao demonstram superioridade universal de pos-hoc sobre qualquer retreino possivel. Os ICs nao sao corrigidos pela busca de configuracoes ou por multiplas comparacoes.

### 4.3 Efeito de treinamento sob a mesma referencia

Aqui a subtracao e variante menos `baseline` DA CLASSE, mantendo a receita original de referencia fixa. Nao reutilizar os ICs da secao anterior: eles comparam outros pares.

| Classe | Variante | Delta AUROC vs baseline da classe | Delta AUPRO vs baseline da classe |
|---|---|---:|---:|
| glass | featnorm | +0.0331 | -0.0128 |
| glass | featnorm_l1_only | +0.0015 | -0.0699 |
| glass | clamp19 | -0.0113 | -0.2246 |
| glass | featnorm_clamp19 | -0.0211 | -0.2553 |
| lightning | featnorm | +0.0284 | +0.0805 |
| lightning | featnorm_l1_only | +0.0030 | +0.0300 |
| lightning | clamp19 | +0.0626 | +0.1231 |
| lightning | featnorm_clamp19 | +0.0689 | +0.1375 |
| polymer | featnorm | +0.0062 | +0.0047 |
| polymer | featnorm_l1_only | -0.0150 | -0.0115 |
| polymer | clamp19 | +0.0092 | -0.0423 |
| polymer | featnorm_clamp19 | +0.0107 | -0.0337 |

Descoberta: clamp 1.9 tem sinal fortemente dependente da classe e da metrica. Em lightning, melhora ambas; em glass, piora ambas sob a referencia; em polymer, melhora AUROC mas piora AUPRO. A normalizacao global de features melhora AUROC nas tres classes, mas AUPRO de glass nao acompanha. Como as epocas foram escolhidas com `raw`, isso avalia a receita de treino mais seu selecionador, nao a melhor head possivel para `cflow_original`.

Comparando as duas referencias salvas sobre as mesmas heads, `baseline_cflow_original` supera `baseline_producao` em AUROC em 11/15 e em AUPRO em 13/15. A referencia de producao usa minmax, alinhamento False e sigma 4; portanto, essa contagem e uma comparacao de RECEITAS, nao uma ablacao causal pura do score.

### 4.4 Localizacao e decisao por imagem nao sao equivalentes

| Classe | AP dos top-1: minimo a maximo | F1 dos top-1: minimo a maximo | AUROC por mapa/imagem: minimo a maximo | SEDifferNet image AUROC no mesmo holdout |
|---|---|---|---|---:|
| glass | 0.00253 a 0.00469 | 0.0153 a 0.0272 | 0.7086 a 0.7434 | 0.7706 |
| lightning | 0.01990 a 0.02819 | 0.0432 a 0.0610 | 0.6653 a 0.7531 | 0.9485 |
| polymer | 0.02965 a 0.03273 | 0.0604 a 0.0695 | 0.6300 a 0.6923 | 0.9495 |

O F1 e o melhor F1 obtido pela rotina de histograma no holdout, nao F1 de um limiar definido antecipadamente na validacao. Mesmo esse valor otimista e baixo. Ha hipotese de desequilibrio pixel-level e falsos positivos, mas a prevalencia exata de pixels anomalos nao foi medida aqui; contagem de IMAGENS anomalas nao fornece essa prevalencia.

Todas as 15 receitas top-1 escolhidas por criterio pixel/regiao tem image AUROC inferior ao SEDifferNet no mesmo holdout. Isso nao testa a melhor receita selecionada especificamente para classificacao por imagem. Sugere manter conceitualmente separados o detector por imagem e o localizador, e validar eventual combinacao antes de propor substituir o SEDifferNet pelo max/topk do mapa.

## 5. Descobertas principais e hipoteses

### 5.1 Niveis dependem da classe

Os cinco top-1 de glass usam niveis 02; lightning usa 02 em quatro heads e 012 em `clamp19`; os cinco top-1 de polymer usam 012. Portanto, excluir L2 universalmente ja e contradito pelo piloto ampliado. Sao escolhas do criterio composto de validacao, nao prova de que L2 isoladamente e bom em toda classe.

### 5.2 O score e o treinamento interagem

Glass/`clamp19` tem AUROC/AUPRO 0.8648/0.4465 sob a referencia pura, mas 0.9435/0.8127 com a receita pos-hoc escolhida. A troca de head e a troca de agregador/niveis nao podem receber o mesmo credito causal. Hipotese: mudancas na escala e nas caudas das NLLs entre niveis alteram o efeito da agregacao. Para testar: medir distribuicoes de NLL e contribuicoes dos niveis em treino/validacao, mantendo a mesma coorte e receita; isso ainda nao foi feito nesta analise.

### 5.3 Seletor de validacao nao garante ganho no holdout

Lightning/`featnorm_clamp19` registra delta AUROC top-1 menos referencia da propria head de -0.0022, IC aproximado [-0.0200, +0.0158]. A selecao pos-hoc nao melhorou essa metrica nesse caso. Grid grande, poucas anomalias e diferencas entre coortes tornam instabilidade plausivel, mas esse caso sozinho nao identifica sua causa.

### 5.4 Discrepancia de AUPRO em polymer

O AUPRO zero de polymer/`featnorm_clamp19` e um dado salvo, nao um valor corrigido ou omitido. Ha pelo menos tres explicacoes concorrentes: deteccao de regioes ruim na faixa de FPR avaliada; sensibilidade da grade global de thresholds a caudas/outliers; inconsistencia numerica/de implementacao. AUROC 0.8848 nao distingue essas explicacoes. E necessario inspecionar mapas, thresholds, pontos PRO/FPR e testar sensibilidade; nao foi regenerada inferencia neste relatorio.

A escada desta head localiza a discrepancia: com pad 112/replicate e hflip, antes de trocar o score, AUROC/AUPRO sao 0.8634/0.6546. A troca para `per_level_std` aumenta AUROC para 0.8766 e registra AUPRO zero; alinhamento False e sigma 4 mantem zero. E uma localizacao no caminho cumulativo, nao prova de que `std` sempre falha: no OFAT sem essa combinacao de pad/TTA, `std` produz AUPRO 0.6357. Interacoes ou propriedades especificas desses mapas importam.

Na implementacao consultada, ausencia de regioes, mapas constantes ou faixa FPR degenerada retornam NaN, nao zero; o caminho de avaliacao chama a metrica diretamente. Nao foi encontrada conversao explicita de NaN para zero nesse caminho. Por isso, nao rotular automaticamente o valor salvo como campo ausente. O codigo atual nao esta arquivado junto da rodada, o que limita a auditoria retrospectiva.

### 5.5 A rodada nao estabelece vencedor universal

`featnorm`, `clamp19` e sua combinacao mudam de comportamento por classe e receita. Apos selecao pos-hoc de cada head, `featnorm_l1_only` muda pouco a AUROC contra a baseline da classe em lightning (+0.0012) e polymer (+0.0029); em glass reduz AUROC (-0.0062) e aumenta AUPRO (+0.0066). Isso e diferente dos ganhos contra a referencia da propria head em 4.2. Sem IC entre variantes e com classes faltantes, a evidencia atual nao autoriza elimina-la de todas as classes nem congelar uma receita unica para vari-grip e yoke.

### 5.6 OFAT: efeitos locais sobre a referencia

Cada linha altera somente um fator da referencia da MESMA head. Medias descritivas entre 15 heads correlacionadas, cinco por classe; nao sao 15 seeds nem estimativas de variancia entre treinos. Colunas IC+ e IC- contam intervalos marginais de delta AUROC inteiramente positivos/negativos, sem correcao de multiplicidade. Intervalos restantes cruzam zero.

| Fator alterado | Delta AUROC medio | Minimo | Maximo | Delta AUPRO medio | IC+ / IC- |
|---|---:|---:|---:|---:|---|
| pad 112 / reflect | +0.0055 | -0.0045 | +0.0131 | +0.0114 | 11 / 2 |
| hflip | +0.0023 | +0.0006 | +0.0045 | +0.0034 | 7 / 0 |
| flips | +0.0071 | +0.0005 | +0.0149 | +0.0006 | 12 / 0 |
| raw | +0.0095 | -0.0145 | +0.0542 | -0.0851 | 7 / 4 |
| per_level_minmax | -0.0538 | -0.2449 | +0.0190 | -0.0891 | 0 / 5 |
| per_level_prob | -0.1263 | -0.3363 | +0.0547 | -0.1485 | 1 / 11 |
| per_level_std | +0.0121 | -0.0253 | +0.0411 | -0.0701 | 10 / 3 |
| niveis 0 | -0.0668 | -0.1280 | +0.0035 | -0.0169 | 0 / 12 |
| niveis 1 | -0.0794 | -0.2456 | +0.0152 | -0.2274 | 0 / 10 |
| niveis 2 | +0.0020 | -0.0613 | +0.0684 | -0.0355 | 5 / 5 |
| niveis 01 | -0.0310 | -0.0761 | -0.0078 | -0.0587 | 0 / 15 |
| niveis 02 | -0.0056 | -0.0511 | +0.0891 | +0.0548 | 5 / 10 |
| niveis 12 | -0.0192 | -0.0761 | +0.0303 | -0.1257 | 4 / 7 |
| align_corners False | -0.0010 | -0.0031 | +0.0005 | -0.0034 | 2 / 8 |
| sigma 2 | +0.0029 | +0.0015 | +0.0044 | +0.0060 | 15 / 0 |
| sigma 4 | +0.0074 | +0.0019 | +0.0124 | +0.0144 | 14 / 0 |
| sigma 8 | +0.0121 | -0.0041 | +0.0248 | +0.0230 | 12 / 0 |
| sigma 12 | +0.0115 | -0.0150 | +0.0287 | +0.0200 | 12 / 2 |
| image score topk | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 / 0 |

Achados: smoothing leve tem o sinal local mais consistente em AUROC; sigma 12 nao e universalmente benifico. `minmax` e `per_level_prob` sao frequentemente prejudiciais nessa referencia. `std` melhora AUROC media mas piora AUPRO media, embora apareca em muitos vencedores de validacao quando combinado a outros fatores. Essa discordancia pede estudar interacoes e metricas, nao abandonar AUPRO.

A media de delta AUROC de niveis 02 passou de positiva no piloto de duas classes para -0.0056 com polymer incluido. Nos pares locais, apenas 5/15 ICs sao positivos e 10/15 negativos. Essa mudanca e uma refutacao pratica da extrapolacao prematura de glass para todas as classes, nao evidencia de que 012 vence toda receita otimizada.

### 5.7 Escada: interacoes e dependencia da ordem

A escada soma mudancas ate a receita escolhida em validacao, na ordem fixa `pad -> pad_mode -> tta -> score -> levels -> align -> pos_calib -> sigma -> margin -> image_score`, omitindo fatores inalterados. Cada delta compara com o passo anterior, nao com a referencia inicial. Nao interpretar como ranking global de importancia.

- Glass/baseline: mudar para `std` apos padding/TTA adiciona aproximadamente +0.0146 AUROC; remover L2 em seguida adiciona +0.0207.
- Glass/clamp19: o passo de score melhora AUROC em +0.0331, mas muda AUPRO em -0.4555; o passo seguinte, niveis 02, adiciona +0.0310 AUROC e +0.8027 AUPRO. Pode haver interacao de fusao e sensibilidade dos thresholds; nao e uma decomposicao causal universal.
- Lightning/featnorm: remover L2 nesse contexto reduz AUROC em -0.0219, IC [-0.0304, -0.0137], e AUPRO em -0.0203.
- Lightning/featnorm_clamp19: remover L2 reduz AUROC em -0.0688, IC [-0.0839, -0.0522], e AUPRO em -0.1142; smoothing posterior recupera parte do desempenho, sem assegurar ganho final de AUROC sobre a referencia.
- Lightning/clamp19: nao muda score ou niveis; padding/TTA/smoothing ja levam de AUROC 0.8566 para 0.8965, AUPRO 0.5468 para 0.6442.
- Polymer/baseline: flips adiciona +0.0052 AUROC, `std` mais +0.0130, sigma 2 mais +0.0010. AUPRO progride de 0.6561 para 0.6907.
- Polymer/featnorm_clamp19: a falha de AUPRO ocorre no passo de score, como detalhado em 5.4; AUROC continua aumentando. Maximizar ranking pixel-level e preservar deteccao por regiao nao sao objetivos automaticamente equivalentes.

## 6. Limites da interpretacao

- Tres classes com pos-hoc, uma seed; as 15 heads nao sao 15 replicacoes independentes: compartilham imagens e splits por classe.
- Os ICs de bootstrap pareado de AUROC reamostram imagens e usam histogramas aproximados; nao cobrem variancia do treino, busca de milhares de configuracoes ou backbone.
- AUPRO nao tem IC salvo. Ausencia de zero no IC de uma linha nao equivale a prova confirmatoria, principalmente apos multiplas comparacoes.
- `cflow_original` normaliza pela coorte e os mapas sao transdutivos. Validacao e holdout sao normalizados separadamente; o ranking pode mudar por composicao da coorte, alem da variacao de amostras.
- Historicos de treino e resultados pos-hoc diferem em receita, particao e protocolo de AUPRO. Seus numeros nao sao diretamente intercambiaveis.
- Nao ha benchmark que separe o tempo de AUPRO do restante do grid; economia de retirar AUPRO nao pode ser declarada em horas exatas.
- Dados antigos de setembro, defaults atuais e resultados desta rodada nao sao o mesmo experimento. Todos os numeros principais deste relatorio vem da rodada identificada.

## 7. Curvas de treinamento e orcamento de epocas

A tabela abaixo e VALIDACAO durante o treino, score raw, sigma 4, com o AUPRO da rotina do treinador. Nao e holdout nem o AUPRO global das secoes anteriores. A epoca de pico foi recalculada pelos indices do historico, considerando a primeira ocorrencia do maior AUROC.

| Classe | Variante | Ultima avaliacao | Pico AUROC: epoca | Pico AUROC | AUROC final | AUPRO de treino no pico AUROC |
|---|---|---:|---:|---:|---:|---:|
| glass | baseline | 40 | 15 | 0.8807 | 0.8776 | 0.6443 |
| glass | featnorm | 40 | 40 | 0.9217 | 0.9217 | 0.6528 |
| glass | featnorm_l1_only | 40 | 10 | 0.8950 | 0.8903 | 0.7000 |
| glass | clamp19 | 40 | 5 | 0.9152 | 0.8994 | 0.6199 |
| glass | featnorm_clamp19 | 40 | 10 | 0.9165 | 0.9046 | 0.5865 |
| lightning | baseline | 40 | 15 | 0.8492 | 0.8491 | 0.5552 |
| lightning | featnorm | 40 | 35 | 0.8757 | 0.8756 | 0.6053 |
| lightning | featnorm_l1_only | 40 | 40 | 0.8515 | 0.8515 | 0.5854 |
| lightning | clamp19 | 40 | 30 | 0.8805 | 0.8802 | 0.5875 |
| lightning | featnorm_clamp19 | 40 | 35 | 0.8842 | 0.8839 | 0.5947 |
| polymer | baseline | 30 | 30 | 0.8837 | 0.8837 | 0.6343 |
| polymer | featnorm | 30 | 15 | 0.8744 | 0.8737 | 0.6284 |
| polymer | featnorm_l1_only | 30 | 30 | 0.8713 | 0.8713 | 0.6782 |
| polymer | clamp19 | 30 | 12 | 0.8755 | 0.8742 | 0.6015 |
| polymer | featnorm_clamp19 | 30 | 15 | 0.8698 | 0.8671 | 0.5804 |
| vari-grip | baseline | 40 | 10 | 0.8712 | 0.8631 | 0.5460 |
| vari-grip | featnorm, parcial | 10 | 5 | 0.8328 | 0.8252 | 0.4644 |

Glass/clamp19 atinge pico cedo e cai cerca de 0.0158 AUROC ate epoca 40; isso e compativel com sobreajuste ou deriva da receita avaliada, nao prova do mecanismo. Glass/featnorm ainda melhora ate epoca 40; cortar todas as variantes em 10 epocas descartaria esse pico. Lightning precisa de 30-40 em varias variantes; polymer/baseline e normalizacao apenas L1 atingem pico na ultima avaliacao.

Early stopping por validacao e uma proposta testavel; usar os picos desta seed como orcamento definitivo nao estima o comportamento em seeds futuras. Nao se pode inferir que uma curva cuja ultima avaliacao e o pico ja convergiu. Vari-grip/featnorm esta parcial e seu pico observado na epoca 5 nao autoriza comparacao final com baseline.

## 8. Custo observado e gargalos

Cada pos-hoc completo declara 10.620 linhas de grid e 5.310 receitas distintas de pixel: duas agregacoes por imagem para cada receita de pixel. Sao 159.300 linhas e 79.650 receitas de pixel entre as 15 heads. Por head, ha 1.062 combinacoes antes de sigma e cinco sigmas. A busca inclui pads 0/112, modos reflect/replicate, TTA none/hflip/flips, cinco scores, dois alinhamentos, sete combinacoes de niveis e calibracao none/l23, descontadas combinacoes inertes/invalidas. A margem desta rodada e sempre 0.

| Classe | Treino das cinco variantes | Setup ate escrita do grid: por head | Grid ate escrita do holdout: por head | Soma aproximada dos intervalos das cinco heads |
|---|---|---|---|---|
| glass | 6.41 h | 251.9 a 359.5 min | 46.6 a 52.5 min | 38.33 h, incluindo treino |
| lightning | 5.60 h | 83.2 a 92.8 min | 12.2 a 13.0 min | 14.05 h, incluindo treino |
| polymer | 8.58 h | 165.6 a 172.0 min | 23.3 a 25.1 min | 24.64 h, incluindo treino |

Duracoes de treino: `retrain_variants.csv`. Intervalos pos-hoc: `LastWriteTime(grid_val.csv) - LastWriteTime(split_manifest.json)` e `LastWriteTime(holdout_test_top.csv) - LastWriteTime(grid_val.csv)`. Sao proxies de tempo de parede; incluem outros trabalhos entre esses arquivos, podem ser alterados por copia/reescrita e nao sao perfis de funcao. A soma das tres classes completas e aproximadamente 77 h; nao inclui a parte incompleta de vari-grip nem estabelece GPU-hours uteis.

A maior parte do custo observado esta entre setup e grid. Esse trecho inclui inferencia/cache de NLL, estatisticas de treino, diagnosticos e avaliacao das receitas; nao foi provado que AUPRO sozinho domina. A metrica global atual usa contagem otimizada, nao 1.000 varreduras completas do mapa por threshold. O trecho de holdout inclui varios candidatos, bootstrap e a metrica RD legacy, ainda calculada; sua contribuicao separada nao foi medida.

### 8.1 Propostas de aceleracao, nao implementadas

- Reutilizar checkpoints existentes para post-hoc com provenance explicita, em vez de retreinar por conveniencia.
- Medir separadamente inferencia, estatisticas, agregacao, AUROC/AP, AUPRO global, RD legacy e bootstrap antes de atribuir percentuais de tempo.
- Considerar remocao de RD legacy de novas analises que nao precisam dela, mantendo artefatos historicos e comparabilidade.
- Comparar selecao completa e reduzida APENAS na validacao, por classe; nao podar niveis com base em glass. Polymer ja mostra porque conservar 012 importa.
- Um grid candidato com pads 0/112, reflect, TTA none/flips, tres scores original/raw/std, sete conjuntos de niveis, alinhamento True, sem calibracao, sigmas 0/4/8/12 e image max tem 336 receitas de pixel. Contra 5.310, sao cerca de 15.8 vezes menos avaliacoes de receitas, nao 15.8 vezes menos tempo total medido. E uma proposta, nao protocolo aprovado nem resultado de benchmark.
- Retirar max/topk duplica menos linhas, mas nao divide por dois o trabalho de metricas pixel-level atual, calculado fora desse loop; tambem pode alterar desempate por metrica de imagem.
- Mudar `select_metric` para AUROC nao desliga o calculo de AUPRO no codigo atual. Isso seria outro protocolo, nao uma opcao equivalente de aceleracao.

Nenhuma dessas propostas foi aplicada neste pedido. Nao existe teste concluido mostrando ausencia de regret entre o grid proposto e o completo nas tres classes; contagens de favoritos em grids anteriores nao substituem esse teste.

## 9. Proximos testes discriminantes

| Questao | Evidencia atual | Teste que pode contradizer a interpretacao | Prioridade |
|---|---|---|---|
| AUPRO zero e falha de localizacao ou discretizacao? | Polymer/combinacao: AUROC 0.8848, AUPRO zero; aparece no passo de score | Reavaliar os mesmos mapas, auditar min/max, finitude, regioes e curva FPR/PRO; comparar thresholds mais densos/quantis como diagnostico, mantendo a metrica oficial principal | Alta, antes de proclamar vencedor |
| Qual foi a causa do exit 1? | Vari-grip parcial; nenhum traceback recuperado | Recuperar scrollback/log original e identificar ultima excecao; nao inferir OOM apenas pelo codigo | Alta, antes de retomar |
| Ganhos de normalizacao/clamp sao replicaveis? | Sinais diferentes por classe; uma seed | Repetir finalistas com seeds de treino adicionais, mesmo split; intervalos entre variantes pareados por imagem e seed | Alta |
| Ha dependencia indevida da coorte? | Score original normalizado separadamente por particao | Avaliar sensibilidade a tamanho/composicao da coorte; testar calibracao fixa ajustada apenas no treino como variante separada | Alta para deploy |
| Grid reduzido preserva selecao? | 5.310 receitas/head e custo alto | Comparar regret e ranking na validacao completa/reduzida de cada classe; predefinir tolerancia antes da comparacao | Media |
| Localizacao e operacional? | AP e melhor F1 baixos | Definir limiar na validacao e medir precisao/recall/F1, falsos positivos por imagem e exemplos de erro sem retunar no holdout | Alta para uso pratico |
| Backbone introduz dependencia anterior? | Checkpoint SEDifferNet ja selecionado antes do split local da head | Auditar treino/selecionador/particoes upstream e origem de cada imagem | Alta para alegacao confirmatoria |

Antes de qualquer nova rodada, preservar CSVs, manifestos, configs e checkpoints; registrar versao do codigo, ambiente, dataset e criterios. Novas escolhas motivadas pelo holdout atual precisam de validacao independente, split externo ou protocolo nested adequado. Seeds extras com o MESMO holdout estimam variacao do treino, mas nao restauram sua independencia apos selecao humana.

## 10. Resumo final

1. Foram gerados 15 pos-hoc completos em glass, lightning e polymer; vari-grip esta incompleta e yoke nao iniciou. O terminal terminou com codigo 1 e a causa nao foi recuperada.
2. Nao existe melhora universal demonstrada. Glass favorece featnorm em AUPRO; lightning favorece clamp19; polymer exige cautela com AUPRO zero da combinacao featnorm_clamp19.
3. A selecao de niveis e especifica da classe: 02 em glass, mas 012 em todas as receitas selecionadas de polymer. Nao eliminar L2 globalmente.
4. Ganhos de pos-hoc nao sao ganhos puros de treino. Separar comparacoes sob receita fixa dos resultados do sistema otimizado.
5. A receita pos-hoc aumenta AUROC sobre a referencia da mesma head em 14/15 pares, mas a combinacao em polymer falha em AUPRO e lightning/combinacao nao demonstra ganho AUROC. Nao escolher apenas pelo maior AUROC.
6. AP/F1 baixos e image AUROC dos mapas inferior ao SEDifferNet mostram que ranking pixel-level, localizacao de regioes e decisao por imagem devem ser avaliados separadamente.
7. As tres classes completas consumiram aproximadamente 77 h pelos intervalos registrados, com maior custo observado antes da escrita do grid. Nao ha perfil que atribua esse custo apenas ao AUPRO.
8. Nenhuma receita final ou corte de protocolo foi implementado por esta analise. E preciso verificar a discrepancia de AUPRO, recuperar a causa da parada, concluir classes faltantes e repetir finalistas com seeds adicionais antes de alegar robustez.