# Contexto do Projeto e Registro de Decisões para IAs

**Escopo:** pesquisa de localização de anomalias por pixel usando uma NF head local sobre features do SEDifferNet, no INSPLAD-seg.  
**Atualizado:** 2026-10-26.
**Instrução:** leia este documento antes de propor ou executar qualquer decisão experimental ou metodológica. Registre neste mesmo arquivo as decisões novas que alterem o protocolo ou a interpretação.

## 1. Para que existe este documento

Este é o ponto de partida operacional para assistentes de IA que trabalham neste repositório. Ele resume o objeto da pesquisa, as restrições vigentes, o protocolo, decisões já tomadas, evidência disponível e como preservar rigor científico. Não substitui os documentos técnicos e relatórios citados; indica como reconciliá-los.

Não trate uma recomendação anterior de IA como fato. Verifique o código, a configuração realmente executada e os artefatos da rodada relevante. Toda conclusão deve indicar a qual classe, checkpoint, split, métrica e configuração se refere.

## 2. Projeto em uma frase

O projeto estuda localização de defeitos/anomalias em imagens aéreas de componentes de linhas de transmissão do INSPLAD-seg, usando uma cabeça de *normalizing flow* (NF/CFLOW pixel head) treinada sobre ativações espaciais de um modelo SEDifferNet por classe, cujo checkpoint é mantido congelado.

A pergunta experimental central é: **mantendo fixo o extrator visual de uma classe, como as escolhas de treinamento da NF head e de inferência/pós-processamento afetam AUROC e localização por pixel, e até que ponto os efeitos dependem da classe e do checkpoint?**

## 3. Escopo e limites

### Dentro do escopo atual

- Cinco classes INSPLAD-seg: `glass-insulator`, `lightning-rod-suspension`, `polymer-insulator-upper-shackle`, `vari-grip` e `yoke-suspension`.
- Comparar variantes de treinamento da NF head e decisões pós-hoc, mantendo congelado o checkpoint SEDifferNet da classe.
- Estudar mapas pixel-level, agregação entre níveis, padding, TTA, interpolação e suavização.
- Usar `cflow_original` como referência de agregação/score sobre as NLLs locais já produzidas.

### Fora do escopo, salvo decisão explícita futura

- Não é uma reprodução end-to-end do CFLOW-AD publicado. O usuário explicitamente excluiu o ponto 5 da lista anterior de correções: não trocar o encoder/head pelo WRN50-2 e decoder oficial, nem reproduzir integralmente loss, amostragem e treinamento CFLOW-AD.
- Não descongelar nem retreinar o SEDifferNet nesta ablação.
- Não alegar comparação direta de estado da arte com resultados publicados em datasets, splits ou tarefas diferentes.

**Consequência terminológica:** dizer “score/agregação CFLOW-AD aplicado à head local” ou “CFLOW-AD score sobre NF head local”. Não chamar o sistema de reprodução original CFLOW-AD.

## 4. Arquitetura local e referência CFLOW-AD

### Pipeline local

1. Um checkpoint SEDifferNet por classe fornece features e fica congelado durante o treino da pixel head.
2. A cabeça local recebe ativações espaciais AlexNet-SE em níveis L1/L2/L3 (64/192/256 canais), redimensionadas para uma grade comum de 96×96.
3. A NF head é treinada para produzir mapas NLL por posição. A receita baseline usa as opções locais definidas em `scripts/train/pixel_train_from_pretrained.py`.
4. A inferência agrega mapas NLL, eventualmente aplica TTA/padding, seleciona níveis, suaviza e calcula métricas.

### Diferenças materiais em relação ao CFLOW-AD oficial

O encoder é AlexNet-SE local, não WRN50-2; os níveis são redimensionados para grade comum; o PE local é 64-dimensional; os coupling blocks, clamp e permutação são locais; a loss é média NLL por célula, sem a loss LogSigmoid e amostragem em fibras oficiais. Portanto, a fórmula de score `cflow_original` não torna arquitetura ou treino equivalentes.

`cflow_original` segue a agregação de inferência de referência: log-probabilidade normalizada por canais, máximo por nível sobre a coorte avaliada, exponenciação, upsample bilinear (`align_corners=True` na referência), soma dos níveis e inversão pelo máximo do mapa somado; sem suavização e score de imagem por máximo quando se usa a referência pura. O cálculo local mantém float64 para evitar underflow/ties da exponenciação.

A normalização `cflow_original` usa máximos da coorte. Por isso é **transdutiva** e dependente da composição de cada partição. No protocolo atual, val e holdout são normalizados separadamente, sem deixar os máximos do holdout influenciar seleção na validação. Isso não equivale à normalização em uma coorte única, e deve ser declarado ao comparar resultados.

## 5. Protocolo experimental vigente

### Dados e partições

- Treino: somente dados normais da pasta `train/good`; as máscaras de teste não devem entrar no gradiente.
- A validação e o holdout são partições estratificadas do split rotulado `test` existente, com `val_fraction=0.4` por padrão.
- `split_seed` define essa partição. Mantenha-a fixa para comparar variantes/seeds de treino.
- `train_seed` controla a inicialização/aleatoriedade do treino e deve ser relatada separadamente do `split_seed`.
- Nesta fase, usar somente uma train seed, 42, e manter split seed 42 (D-13). A estabilidade entre seeds foi relatada pelo usuário a partir de sua experiência; não foi medida nesta rodada. Repetições entre seeds ficam adiadas, não demonstradas como desnecessárias universalmente.
- O mesmo split deve ser usado pelo treinador e pela ablação. `--limit`, mudança de dataset ou mudança no inventário/ordem de arquivos altera o split e invalida a alegação de holdout limpo, a menos que o alinhamento seja demonstrado por caminhos/manifesto.
- O checkpoint de treino grava `selection_validation_paths`. A ablação só marca `holdout_clean=True` se puder verificar que a seleção da época não usou caminhos do holdout. Checkpoints antigos sem essa prova são diagnósticos, não resultados limpos.
- `val_fraction=0` reutiliza imagens para seleção e reporte; não é permitido para conclusões confirmatórias.

### Seleção de época e de configuração são decisões diferentes

- O launcher atual passa `--train_score_mode raw`. O treinador salva `best_pixel_auroc.pt` pela Pixel AUROC do score `raw` na validação de época. Isso não quer dizer que `raw` vence o grid pós-hoc.
- Na ablação, o grid contém vários scores, incluindo `cflow_original`. O default direto do script continua sendo `auroc_aupro`; nas próximas classes, `vari-grip` e `yoke-suspension`, o launcher passa `--auroc_only` e `--select_metric pixel_auroc` (D-14). Nelas, AUPRO global/legacy não é calculado nem participa da seleção. Nas três classes anteriores, preservar os resultados compostos já produzidos.
- Não misture essas duas escolhas ao explicar o resultado: a head é selecionada por AUROC de época `raw`; a receita de mapa é selecionada por AUROC+AUPRO nas avaliações históricas e por Pixel AUROC nesta nova fase das classes restantes.
- Qualquer alteração desses critérios exige decisão registrada, justificativa e análise de comparabilidade com rodadas anteriores.

### Métricas

- Pixel AUROC mede ordenação global de pixels positivos/negativos e pode ser dominado por grande número de pixels normais.
- AUPRO mede sobreposição por região para uma faixa de FPR. `compute_aupro_cflow` implementa o protocolo global do CFLOW-AD: thresholds comuns, falsos positivos em pixels normais de todas as imagens (incluindo imagens sem defeito) e média por região. O AUPRO RD++ antigo, per-image, permanece apenas como `aupro_rd_legacy`; não compare esse valor com `aupro` oficial.
- A seleção composta histórica `auroc_aupro=(pixel_auroc+aupro)/2` dá peso igual às duas métricas. É uma escolha operacional, não uma lei universal. D-14 a substitui por Pixel AUROC nas classes restantes; comparação entre receitas selecionadas por critérios diferentes deve declarar a mudança.
- Em `--auroc_only`, calcular Pixel AUROC e AUROC por imagem; adiar AP/F1 e todas as variantes de AUPRO, inclusive no treino por época e no relatório final. Campos não calculados ficam NaN/ausentes, nunca zero; os resumos registram a política. Ausência de AUPRO não permite alegar qualidade por região, e a discrepância de polymer continua aberta.
- O grid usa `pixel_stride=2` por padrão para acelerar triagem; holdout usa stride 1 e cálculo mais exato. Pequenas diferenças entre os números de val e holdout podem decorrer também dessa resolução de avaliação.
- Intervalos atuais por bootstrap reamostram imagens dentro de cada classe de rótulo. Não medem variância entre seeds de treinamento. Uma seed não estabelece robustez entre inicializações.
- Margem de borda padrão é zero. Margens diferentes mudam o conjunto de pixels avaliado e não devem ser escolhidas depois de observar o holdout.

## 6. Fatores e variantes existentes

Consulte os defaults reais de `scripts/eval/ablation_nf_head.py` e do launcher antes de executar: eles podem mudar. No launcher registrado em 2026-10-07/09, as variantes eram `baseline`, `featnorm`, `featnorm_l1_only`, `clamp19` e `featnorm_clamp19`; fatores pós-hoc incluíam padding/modo, TTA, score, níveis, calibração posicional, `align_corners`, sigma, margem e image score.

Interpretação mínima:

- **Variante de treino:** muda a head aprendida.
- **Configuração pós-hoc:** reinterpreta a mesma saída da head ou muda transformações da inferência.
- **OFAT:** baseline mudando um fator de cada vez; estima efeito isolado local à baseline.
- **Escada:** atualizações cumulativas em uma ordem fixa; os efeitos são condicionais à ordem e às interações.
- **Grid fatorial:** explora interações, mas tem custo alto e winner's curse quando há muitas combinações e poucas imagens positivas.

Não compare o melhor valor pós-hoc de cada variante como se fosse efeito puro do treino: isso mistura treinamento e seleção pós-hoc. Para isolar treino, fixe a receita de inferência antes da comparação.

### Execução da fase AUROC-only

O launcher agora seleciona por padrão apenas `vari-grip` e `yoke-suspension`, com uma seed 42. Não altera o grid, as cinco variantes, os orçamentos de épocas, o split, o backbone ou o bootstrap. Nas duas classes, propaga `--auroc_only` à ablação e ao treinador. O checkpoint `best_pixel_auroc.pt` continua sendo o escolhido; o monitor auxiliar `composite` do treinador passa a coincidir com Pixel AUROC nesse modo.

Para conferir os comandos sem treinar:

```powershell
.\scripts\eval\run_nf_head_ablation.ps1 `
	-DatasetRoot 'C:\Users\teo-s\Documents\GitHub\anomaly-detection-dataset\insplad-seg\insplad-seg' `
	-Python 'C:\Users\teo-s\.conda\envs\anomaly_attentideffernet\python.exe' `
	-Seeds 42 -SplitSeed 42 -PlanOnly
```

Sem `-PlanOnly`, o launcher inicia novos treinos do zero; não retoma automaticamente o treino parcial de vari-grip nem reutiliza sua baseline. A pasta de saída padrão recebe novo timestamp: não sobrescrever a rodada histórica com outro protocolo. `-Classes` continua permitindo seleção explícita; somente as classes de `-AurocOnlyClasses` usam o modo reduzido. O script Python aceita `--auroc_only` também em `--stage posthoc`, para avaliar checkpoints existentes explicitamente. Em futuras avaliações completas, omitir essa flag e definir a política do launcher/critério desejado; registrar a retomada antes de executar.

## 7. Hierarquia de evidências

Ao resolver discrepâncias ou tomar decisões, use esta ordem:

1. **Artefato e configuração da rodada específica:** `config.txt`, `split_manifest.json`, checkpoint metadata, `history.json`, CSVs de validação/holdout e timestamps/processos. Eles dizem o que realmente rodou.
2. **Código executado nessa revisão:** script/commit efetivo, argumentos do processo e implementação de métricas.
3. **Protocolo vigente deste documento e decisões registradas**, desde que não contradigam artefatos de uma rodada histórica.
4. **Plano e relatório acadêmico:** úteis para contexto e histórico, mas possuem rodadas/protocolos antigos. Não transportar um número sem indicar sua rodada e protocolo.
5. **Memória de IA, conversa e inferência:** pistas para localizar evidência, nunca fonte primária para uma conclusão numérica.

Se duas fontes divergem, não escolha silenciosamente uma delas: identifique a divergência, sua provável causa e o teste/artefato que a resolve.

## 8. Snapshot experimental — rodada de outubro de 2026

**Snapshot consultado em 2026-10-10.** Raiz observada: `resultado_analise_final/ablacao_nf_head/run_full_20261007_123558/`. Foi iniciada com `Seeds=42`, `SplitSeed=42`, cinco variantes por classe, `--stage retrain` (retreina NF head; SEDifferNet permanece congelado). Não atualizar este snapshot sem reler os artefatos.

- `glass-insulator`: cinco treinamentos e cinco pós-hoc completos.
- `lightning-rod-suspension`: cinco treinamentos e cinco pós-hoc completos.
- `polymer-insulator-upper-shackle`: os cinco diretórios de treino/checkpoint estavam presentes; o último treino ainda estava ativo na última checagem. Pós-hoc ainda não havia começado; sem conclusão held-out da rodada.
- `vari-grip` e `yoke-suspension`: ainda não iniciadas nesta rodada.
- Portanto, há 10 heads com grid e holdout completos, mas todos de apenas duas classes e uma seed.

### Sinais provisórios observados nas 10 heads completas

Comparação de variantes de treino sob o **mesmo score fixo `cflow_original`** no holdout, delta médio pareado à baseline das respectivas heads/classe (5 heads por classe):

| Variante | Glass ΔPixel AUROC / ΔAUPRO | Lightning ΔPixel AUROC / ΔAUPRO |
|---|---:|---:|
| `featnorm` | +0.033 / −0.013 | +0.028 / +0.081 |
| `featnorm_l1_only` | +0.002 / −0.070 | +0.003 / +0.030 |
| `clamp19` | −0.011 / −0.225 | +0.063 / +0.123 |
| `featnorm_clamp19` | −0.021 / −0.255 | +0.069 / +0.138 |

Leitura: o efeito do treino depende da classe e do score. `featnorm` melhora AUROC nas duas classes sob esta comparação, mas AUPRO não melhora em glass. `clamp19` e `featnorm_clamp19` ajudam lightning e prejudicam AUPRO de glass. `featnorm_l1_only` é fraca nestas duas classes; não a descarte para todas as classes antes de analisar polymer/vari-grip/yoke.

As melhores receitas pós-hoc escolhidas em validação foram diferentes: glass teve top-1 holdout 0.951 AUROC/0.821 AUPRO (`featnorm_clamp19`), enquanto lightning teve 0.897/0.644 (`clamp19`). Esses valores incluem efeito do pós-processamento selecionado; **não** isolam ganho de treinamento.

OFAT agregado de glass + lightning (10 heads): `pad=112` teve Δ médio +0.0065 AUROC/+0.014 AUPRO; `tta=flips` +0.0079/−0.0088; `sigma=4` +0.0080/+0.0138; `align_corners=False` −0.0015/−0.0051. `image_score` max/topk não altera nenhuma métrica de pixel. Os modos `per_level_minmax` e `per_level_prob` foram em média inferiores à baseline `cflow_original` nessas 10 heads. São efeitos de dois tipos de classe, uma seed e a baseline específica; não são veredito universal nem substituem a validação nas três classes faltantes.

Na validação das mesmas 10 heads, níveis `02` estiveram em 81% das linhas top-50 por head; `pos_calib=l23` não apareceu no top-50. Isso sustenta uma hipótese de poda para próxima busca, não uma confirmação fora dessas heads.

**Regra para comunicar snapshot:** chamar de parcial/preliminar; separar (a) comparação de heads com score fixo, (b) melhor receita pós-hoc escolhida em val e medida no holdout, e (c) ablação OFAT. Não agregar como se fossem a mesma pergunta.

## 9. Decisões metodológicas vigentes

| ID | Decisão vigente | Por quê / evidência | Limite / como revisar |
|---|---|---|---|
| D-01 | Manter o SEDifferNet de cada classe congelado; retreinar NF head e pós-processamento | Isola o objeto da ablação e conserva a restrição experimental original | Qualquer descongelamento muda a pergunta e exige novo protocolo |
| D-02 | Não implementar reprodução end-to-end CFLOW-AD nesta fase | Exclusão explícita do usuário; objetivo atual é estudar score/head local | Só reabrir mediante solicitação explícita e novo escopo |
| D-03 | Split estratificado `val_fraction=0.4`; `split_seed` fixo e separado de train seeds | Comparação justa e prevenção de seleção no holdout; manifesto de caminhos torna a separação auditável | Novo inventário/dataset/limit pode mudar o split; comparar manifestos |
| D-04 | Checkpoint da head escolhido pelo Pixel AUROC de validação no modo `raw` no launcher atual | O launcher passa `--train_score_mode raw`; não confundir com a seleção da receita pós-hoc | Mudança deve ser aplicada a todas as variantes e documentada |
| D-05 | `superseded` por D-14 somente nas classes restantes; preservar seleção histórica por `auroc_aupro` | Registra o critério realmente usado em glass/lightning/polymer | Não reinterpretar resultados compostos como se fossem selecionados por AUROC |
| D-06 | Usar AUPRO global CFLOW-AD como `aupro`; rotular métrica RD++ antiga como legacy | A métrica por imagem não conta FPs normais da mesma forma e não é comparável ao protocolo global | Versão da métrica e thresholds precisam acompanhar publicação |
| D-07 | `cflow_original` é referência de score sobre head local, não reprodução do modelo CFLOW-AD | Encoder, PE, decoder, loss e treino locais diferem substancialmente | Relatar componentes divergentes em qualquer comparação |
| D-08 | `margin=0` como default; holdout deve ser consultado só após congelar decisões | Evita escolher região avaliada segundo o resultado e mantém pixels de borda | Estudo explícito de borda pode ser diagnóstico separado |
| D-09 | Uma seed é evidência exploratória; bootstrap por imagem não substitui repetição de treino | Bootstrap mede incerteza de amostragem de imagens, não inicialização da head | Repetições ficam adiadas por D-13; não alegar robustez entre seeds sem verificá-la |
| D-10 | Reduzir grids é aceitável se a poda for decidida pela validação/piloto e registrada antes de avaliar holdout da próxima rodada | Grid completo observado demora horas por head e aumenta winner's curse; fatores claramente redundantes podem ser removidos | Não use o holdout atual para escolher a poda. Preservar scripts/config antigos para reprodutibilidade |
| D-13 | Uma train seed 42 nesta fase; split seed 42 | Decisão explícita do usuário, custo e experiência relatada de estabilidade | Não confundir experiência com teste de invariância; reabrir se o objetivo exigir robustez entre seeds |
| D-14 | Próximas classes vari-grip/yoke: selecionar e avaliar AUROC, adiar AUPRO/AP/F1 | Decisão explícita do usuário para reduzir custo e avançar a rodada | Não validar localização por região nem comparar critérios compostos como equivalentes; AUPRO fica para avaliação futura |

## 10. Checklist obrigatório para cada decisão de IA

Antes de recomendar ou executar qualquer alteração experimental:

1. **Pergunta:** qual hipótese científica a alteração testa? Separe efeito de treino, efeito de inferência e engenharia de runtime.
2. **Fonte:** consulte o código e os artefatos da rodada; anote caminho, classe, checkpoint, seed, split e métrica.
3. **Contaminação:** confirme quais amostras influenciaram gradiente, escolha de época, escolha de hiperparâmetro e reporte final. Holdout não escolhe nada.
4. **Comparabilidade:** mantenha fixos backbone/checkpoint, split, seed apropriada, resolução, alinhamento, máscara, métrica e preprocessing, salvo o fator estudado.
5. **Efeito e incerteza:** reporte direção e magnitude das métricas habilitadas, pares, IC e número de seeds. Em AUROC-only, declare AUPRO/AP/F1 como não calculados, não como zero. Não declare vencedor por microganho inferior à incerteza observável ou robustez entre seeds não testadas.
6. **Custo:** estime operações/configurações e faça benchmark limitado antes de lançar grids longos. Não execute treinamento ou altere processo ativo sem pedido/autorização.
7. **Decisão:** escolha a menor mudança que discrimina a hipótese; diga o que poderia refutá-la.
8. **Registro:** antes de encerrar, acrescente uma entrada ao Registro de Decisões se algo metodológico foi decidido. Se não há evidência suficiente, registre como hipótese, não como fato.

## 11. Onde registrar novas decisões

Acrescente cada decisão metodológica nova em **11.1 Registro cronológico** abaixo. Não apague decisões antigas: altere seu estado para `superseded` e crie nova entrada referenciando o ID anterior. Para correção puramente técnica sem impacto no protocolo, use o changelog do código/commit; se muda métricas, amostras, seleção ou inferência, entra aqui.

Use este formato para cada item:

```text
### D-XX — título curto
- Data / responsável:
- Estado: proposta | provisória | aceita | rejeitada | superseded
- Decisão:
- Por quê (racional científico/prático):
- Evidência: arquivos, métricas, run, classe, split, seeds e código
- Alternativas consideradas e motivo para não escolhê-las:
- Riscos/limites:
- Verificação que poderia refutar a decisão:
- Resultado da verificação / próxima revisão:
```

### 11.1 Registro cronológico

#### D-10 — Aceitar poda de grid somente com evidência de validação
- Data / responsável: 2026-10-10 / decisão de protocolo da pesquisa.
- Estado: aceita como regra; valores concretos de poda ainda são proposta até serem definidos para uma nova rodada.
- Decisão: remover combinações comprovadamente redundantes ou dominadas usando apenas pilotos de validação; manter métricas e configurações usadas em resultados históricos.
- Por quê: grid de 5.310 receitas de pixel por head levou horas e múltiplas decisões pós-hoc favorecem winner's curse; métricas de pixel não mudam com `image_score`.
- Evidência: `resultado_analise_final/ablacao_nf_head/run_full_20261007_123558/`; 10 heads completas de duas classes, uma seed; 10.620 linhas por grid devido duplicação max/topk. OFAT e grid val descritos na seção 8.
- Alternativas: conservar grid completo (mais abrangente, custo inviável); remover AUPRO inteiramente (mais rápido, mas muda seleção e ignora qualidade por região).
- Riscos/limites: poda inferida de duas classes não generaliza necessariamente a polymer, vari-grip ou yoke. Não escolher valores usando holdout.
- Verificação que poderia refutar: repetir o grid reduzido e completo lado a lado em um piloto apenas na validação de cada classe; comparar ranking e regret dos finalistas.
- Resultado da verificação / próxima revisão: pendente nas três classes restantes e em seeds adicionais.

#### D-11 — Não tratar `featnorm_l1_only` como descartada ainda
- Data / responsável: 2026-10-10 / síntese provisória desta rodada.
- Estado: provisória.
- Decisão: manter a variante até o pós-hoc de polymer e as avaliações de vari-grip/yoke; pode ser removida de rodadas futuras se não trouxer ganho replicável.
- Por quê: resultados fixos `cflow_original` são quase nulos em glass/lightning, mas no treino de polymer o histórico mostrou AUPRO de validação superior à baseline. Eliminar agora confundiria evidência parcial com universal.
- Evidência: tabela de D-10; históricos em `.../glass-insulator/train/`, `.../lightning-rod-suspension/train/`, `.../polymer-insulator-upper-shackle/train/`.
- Alternativas: descartar imediatamente (economiza um treino por classe); manter em toda grade futura (custo excessivo).
- Riscos/limites: AUPRO observado durante treino usa o protocolo local de treinamento, não é estimativa holdout nem substitui a comparação fixa de score.
- Verificação que poderia refutar: comparar no mesmo score fixo e split os checkpoints dessa variante contra baseline em todas as classes, com múltiplas seeds nas finalistas.
- Resultado da verificação / próxima revisão: aguardar pós-hoc polymer e novas classes.

#### D-12 — Atualizar evidência parcial sem promover vencedor universal
- Data / responsável: 2026-10-10, corte 19:38 local / GitHub Copilot, relatório solicitado pelo usuário.
- Estado: provisória; registro de interpretação, não aprovação de novo protocolo.
- Decisão: substituir, para consultas do estado atual, o snapshot parcial da seção 8 pelo [relatório atualizado da rodada](relatorio_achados_run_full_20261007_123558.md). Manter separadas comparação entre treinos sob receita fixa, seleção pós-hoc por validação e observações exploratórias de holdout. Nenhuma poda ou mudança de treino foi autorizada/aplicada nesta análise.
- Por quê: polymer agora tem cinco pós-hoc completos; suas cinco receitas selecionadas preservam 012, contradizendo a extrapolação de remover L2 universalmente. A combinação featnorm_clamp19 registra AUROC 0.8848 com AUPRO zero e não pode ser declarada melhor localizadora apenas pela AUROC.
- Evidência: 15 heads completas nas três classes, train seed 42, split seed 42; resumos CSV, holdouts, históricos e manifestos da rodada run_full_20261007_123558. Vari-grip tem baseline concluída e featnorm parcial; yoke não iniciou. Terminal saiu com código 1, sem causa recuperada; nenhum Python ativo no corte.
- Alternativas consideradas e motivo para não escolhê-las: repetir o snapshot antigo (omite polymer); escolher variante pela maior AUROC holdout (ignora seleção e AUPRO); corrigir/omitir o zero (não há causa técnica demonstrada).
- Riscos/limites: uma seed, heads correlacionadas, holdout já observado, dependência de coorte no score original, verificação holdout_clean apenas local à head. Os deltas fixos são individuais contra baseline da classe, não médias independentes entre cinco heads. Menções anteriores a poda não equivalem à autorização de valores concretos.
- Verificação que poderia refutar a decisão: auditar os mapas/curva PRO-FPR de polymer, completar vari-grip/yoke e repetir finalistas com outras seeds e avaliação independente; comparar grid reduzido/completo somente em validação antes de qualquer poda.
- Resultado da verificação / próxima revisão: artefatos e consistência dos índices foram conferidos; causa da parada e discrepância de AUPRO permanecem abertas. D-11 continua provisória: polymer/L1-only ganha sobre sua própria referência após pós-hoc, mas perde AUROC/AUPRO contra baseline sob score fixo; ainda faltam outras classes e seeds.

#### D-13 — Usar uma única seed nesta fase
- Data / responsável: 2026-10-10 / decisão explícita do usuário, implementada por GitHub Copilot.
- Estado: aceita para a fase atual.
- Decisão: executar as próximas classes com train seed 42 e split seed 42. O launcher aceita exatamente uma seed por chamada nesta fase; não iniciar repetições adicionais automaticamente.
- Por quê: o usuário relata que, pela sua experiência, variar seeds não muda o resultado do modelo e prefere priorizar continuidade/custo. Conservar 42 mantém a comparação com as classes já concluídas.
- Evidência: solicitação explícita nesta conversa; rodada run_full_20261007_123558 já usa 42/42. Não há comparação multiseed desta rodada que comprove invariância.
- Alternativas consideradas e motivo para não escolhê-las: múltiplas seeds agora (mais custo e contrário à decisão); trocar a seed única (reduz comparabilidade sem vantagem demonstrada).
- Riscos/limites: não estimar variância de inicialização nem garantir ranking entre treinos. IC bootstrap por imagem continua válido para seu objeto, mas não cobre seeds. A ressalva de D-09 permanece; sua exigência operacional de replicação é adiada.
- Verificação que poderia refutar a premissa: se futuramente necessário, repetir finalistas em seeds independentes, com mesmo split e receita, e verificar se diferenças excedem uma tolerância predefinida ou alteram ranking.
- Resultado da verificação / próxima revisão: invariância não testada nesta fase; reavaliar quando o objetivo exigir robustez entre inicializações ou houver sinais de instabilidade. Nenhum treino foi iniciado por este registro.

#### D-14 — Adiar AUPRO e selecionar somente por AUROC nas classes restantes
- Data / responsável: 2026-10-10 / decisão explícita do usuário, implementada por GitHub Copilot.
- Estado: aceita para vari-grip e yoke-suspension; supersede D-05 apenas nesse escopo.
- Decisão: usar Pixel AUROC de validação para selecionar receitas; calcular AUROC pixel e imagem, sem AP/F1, AUPRO global ou AUPRO RD legacy no grid/holdout. O treinador também deixa de calcular AUPRO por época/final e AP/F1 finais. Retomar essas métricas somente em avaliações futuras explicitamente planejadas.
- Por quê: o usuário considera o custo de AUPRO alto e decidiu adiar a avaliação por região para avançar as próximas classes. Alterar apenas select_metric não remove seus cálculos, por isso foi criada a opção --auroc_only e sua propagação ao treino.
- Evidência: solicitação explícita; chamadas antes incondicionais em scripts/eval/ablation_nf_head.py e na avaliação final de scripts/train/pixel_train_from_pretrained.py; tempos de parede no relatório D-12. Não há perfil que prove quanto do tempo total pertence somente ao AUPRO, nem economia medida nesta nova fase.
- Alternativas consideradas e motivo para não escolhê-las: manter seleção composta (contrário à decisão/custo); calcular AUPRO só nos finalistas agora (ainda contrário ao adiamento); retirar apenas AUPRO do grid (deixaria custo no holdout, legacy e treino final).
- Riscos/limites: a nova receita pode priorizar ranking pixel-level e perder desempenho por região, como já sugerido pela discrepância de polymer. Não preencher métricas omitidas com zero nem comparar a seleção AUROC-only à composta como se fossem o mesmo protocolo. Não reinterpretar artefatos antigos. AUPRO de polymer zero continua uma questão pendente, não resolvida por desativar a métrica nas outras classes.
- Verificação que poderia refutar a adequação: em avaliação futura, congelar finalistas antes do holdout e calcular AUPRO/AP/F1 com protocolo declarado; verificar se o ranking/qualidade de região diverge materialmente da AUROC. Escolhas motivadas pelo holdout atual exigem avaliação independente para confirmação.
- Resultado da verificação / próxima revisão: testes sintéticos passaram para AUROC exata/grid e avaliação final do treinador, sem chamadas às métricas desativadas e sem registro de NaN no MLflow. As duas interfaces --help reconheceram a flag no Conda indicado; retreino simulado confirmou sua propagação. PlanOnly confirmou apenas as duas classes restantes, uma seed, rejeição de multiseeds e modo composto preservado para classe anterior explícita. Nenhum novo treino ou benchmark de duração foi executado; grid, variantes, split, bootstrap e arquitetura não foram podados. A adequação regional será reavaliada quando AUPRO for retomado.

#### D-15 — Receita por classe e interpretação das 5 classes
- Data / responsável: 2026-10-13 / GitHub Copilot, análise solicitada pelo usuário.
- Estado: provisória; interpretação e proposta, sem alteração de código ou protocolo.
- Decisão: tratar a receita de inferência como específica de cada classe (níveis, score e sigma) e não adotar receita única. Usar como candidatas: vari-grip `featnorm_clamp19` + pad112 reflect/hflip/`cflow_original`/L1/up=True/σ8; yoke `featnorm_l1_only` (empate prático com baseline) + pad112 reflect/flips/`per_level_std`/L1+L3/up=True/σ12. Glass, lightning e polymer mantêm os vencedores compostos de D-12, com a ressalva de AUPRO de polymer.
- Por quê: na validação das 25 heads, a melhor receita única perde 0.026 de AUROC médio (máx. 0.049) contra a receita por head; receitas fixas por classe perdem 0.0006–0.017. Níveis ótimos diferem entre classes (vari L1; yoke e glass L1+L3; lightning/polymer outros).
- Evidência: [relatório vari/yoke](relatorio_achados_run_full_20261010_195638.md); 10 heads AUROC-only com holdout limpo, seed 42. Média holdout das 5 classes: produção legada 0.8405 → 0.9208 (+0.080), cerca de 78% vindo de score/receita e 22% da variante de treino.
- Alternativas consideradas e motivo para não escolhê-las: receita única (mais simples, mas custa ~2,6 pontos); escolher variante pelo maior AUROC do holdout (viola a seleção por validação).
- Riscos/limites: seleção mista (AUROC+AUPRO em 3 classes, AUROC em 2); uma seed; sem AUPRO em vari/yoke; yoke tem 18 anomalias na validação e o vencedor ganha por 0.0008; receitas AUROC-only de glass/polymer não foram medidas no holdout.
- Verificação que poderia refutar: repetir as candidatas com outra seed ou em avaliação independente; medir AUPRO das finalistas; avaliar no holdout as receitas AUROC-only de glass/polymer antes de fixá-las.
- Resultado da verificação / próxima revisão: pendente. Propostas abertas: treino curto com avaliação densa (picos na primeira avaliação em yoke e na maior parte de vari), desempate pré-registrado em favor da variante mais simples, grid reduzido e grid na GPU.

#### D-16 — Análise consolidada das 25 heads como fonte de verdade para publicação
- Data / responsável: 2026-10-13 / GitHub Copilot, análise solicitada pelo usuário.
- Estado: provisória (exploratória-forte); não altera código de treino nem de avaliação.
- Decisão: usar a [análise científica consolidada](analise_cientifica_ablacao_nf_head_5classes.md) e as tabelas de `resultado_analise_final/ablacao_nf_head/analise_consolidada_20261013/` (geradas por `scripts/eval/consolidate_nf_head_ablation.py`) como referência numérica para escrita científica. Os números de setembro (plano, relatório acadêmico, artigo) ficam marcados como históricos. Receita geral proposta: `baseline` + pad112 reflect + flips + `per_level_std` + σ8 + sem calibração + níveis por classe (dentro do template: glass 02, lightning 012, polymer 12, vari 0, yoke 012; sem informação: 012).
- Por quê: efeitos estimados por pares de um único fator (132.750 receitas), com a classe como unidade de generalização e IC bootstrap hierárquico, replicados no OFAT do holdout. `per_level_std` empata em AUROC com `cflow_original` (Δ +0,0017, IC [−0,0046; +0,0081]) e é indutivo, ao contrário do agregador original transdutivo. A receita única custa 0,026 de regret; liberar os níveis por classe reduz para 0,015.
- Evidência: validação — min-max −0,075 (0/5 classes), `prob` −0,113, calibração posicional −0,025, σ8 +0,016 (25/25 heads), flips +0,005, pad reflect +0,002; níveis: regret de 02 = 0,044 em polymer, 012 minimax (máx. 0,021); variantes: Friedman p = 0,72 (sistema) e 0,89 (receita fixa); holdout médio 0,8405 → 0,9208.
- Alternativas: `cflow_original` como default (melhor AUPRO, mas dependente do lote); L1+L3 universal (refutado); receita única (custa 2,6 pontos).
- Riscos/limites: uma seed; holdout já inspecionado; AUPRO ausente em vari/yoke; 14 AUPRO = 0 interpretados como provável colapso da grade linear de limiares, sem verificação; receitas fixas por classe e template medidos só na validação; com n = 5 classes, Wilcoxon não alcança p < 0,05.
- Verificação que poderia refutar: protocolo confirmatório da seção 9 do documento (validação cruzada aninhada ou split novo, 3 seeds, AUPRO por quantis, receitas fixas no holdout, baselines externos no mesmo split).
- Resultado da verificação / próxima revisão: pendente; nenhuma rodada nova foi iniciada.

#### D-17 — Presets executáveis no treinador da pixel head
- Data / responsável: 2026-10-25 / solicitação explícita do usuário, implementação por GitHub Copilot.
- Estado: aceita como opção de execução; superioridade de desempenho dos presets continua provisória.
- Decisão: adicionar `--recipe best_per_class`, `--recipe general` e `--recipe baseline` em `scripts/train/pixel_train_from_pretrained.py`. Sem opção, `custom` preserva os controles manuais. `--print_recipe` consulta as configurações sem carregar modelo ou iniciar treino. Checkpoint SEDifferNet omitido é descoberto pela classe, não fixado em lightning.
- Por quê: permitir executar as receitas do relatório sem misturar parâmetros de treino com inferência. Todos os presets treinam 012, sem padding, com seleção de época por raw/σ4; níveis, pad, TTA, score e sigma da receita são aplicados somente na avaliação final do melhor checkpoint. Remover flows do treino seria uma nova ablação, não reprodução do resultado de inferência por subconjunto.
- Evidência: Tabela G e template da seção 3.6 da análise científica. `best_per_class` usa featnorm em glass, clamp19 em lightning, featnorm_clamp19 em polymer/vari e normalização só L1 em yoke; herda os critérios históricos mistos, não um novo ranking de holdout. Em polymer mantém o vencedor histórico com alerta de AUPRO zero não resolvido.
- Alternativas consideradas e motivo para não escolhê-las: aplicar pad/TTA no treino (não foi o regime vencedor); treinar apenas os níveis de score (muda o modelo e sua aleatoriedade); calcular cflow_original por minibatch (muda a coorte e o score). O preset baseline usa head sem normalização, clamp 0.5, score raw, todos os níveis, pad 0, sem TTA e σ4; não é a referência legada min-max nem baseline_cflow_original.
- Riscos/limites: general é template candidato, não desempenho confirmado no holdout; nova inicialização/dataset não garante os números históricos. Campos fixos de presets rejeitam flags conflitantes; épocas e intervalo de avaliação podem ser explicitamente ajustados e são registrados. Presets desligam PatchCore. `--auroc_only` continua explícito, inclusive para executar a política D-14; seed/split permanecem 42 por padrão. O dataset padrão da configuração difere do usado nas ablações, por isso fornecê-lo explicitamente para comparações.
- Verificação que poderia refutar: testar as 15 combinações classe/preset, a CLI e conflitos; comparar numericamente agregação/smoothing com as funções compartilhadas; verificar coorte completa para cflow_original, estatísticas só do treino normal e round-trip dos metadados. Depois, avaliar checkpoints efetivamente treinados no mesmo protocolo antes de alegar reprodução de métricas.
- Resultado da verificação / próxima revisão: resolver/parser das 15 combinações e agregação numérica passaram; caminho final sintético passou nos três modos; CLI real de consulta passou. A receita é salva em recipe.json, checkpoints e identificada em final_results.csv; o melhor checkpoint recebe também inference_recipe/inference_level_stats. A política de escolha de época e os arquivos históricos não foram alterados; não foi iniciado treino real. Avaliadores externos ainda precisam receber a receita explicitamente: não assumir que seus defaults leem automaticamente esses novos metadados.

Uso a partir de `differnet/`, com o ambiente Conda da pesquisa ativado:

```powershell
python scripts/train/pixel_train_from_pretrained.py `
	--class_name vari-grip --recipe general `
	--dataset 'C:\Users\teo-s\Documents\GitHub\anomaly-detection-dataset\insplad-seg\insplad-seg' `
	--auroc_only --print_recipe
```

Remover `--print_recipe` inicia o treino. Trocar `general` por `best_per_class` ou `baseline` seleciona os outros modos. O preset general usa baseline no treino e inferência pad112 reflect, flips, per_level_std, up=True, σ8, max, sem calibração e margem 0; só os níveis de inferência mudam por classe: glass 02, lightning 012, polymer 12, vari 0, yoke 012. Orçamentos padrão: 40/40/30/40/12 épocas, com avaliações a cada 5/5/3/5/2, respectivamente.

Execução em lote dos presets (atualização operacional de D-17, 2026-10-25): `scripts/train/train_all_nf_heads.py` e wrapper `scripts/train/run_train_all_nf_heads.ps1` usam general por padrão e também aceitam best_per_class/baseline. Fornecer `--checkpoint_dir` / `-CheckpointDir` com a pasta dos encoders SEDifferNet, não das NF heads. A busca é recursiva, admite .pt/.pth/.ckpt e exige uma correspondência por classe no nome ou caminho; ausência/ambiguidade aborta antes do primeiro treino. As cinco classes são executadas em sequência. `--dry_run` / `-DryRun` apenas imprime o plano. `--auroc_only` / `-AurocOnly` propaga D-14 explicitamente. Validação escolhe a época; métricas finais vêm da linha CFLOW do final_results.csv, não do pixel_auroc de validação do checkpoint. Ao terminar, training_summary.csv e training_summary.md reúnem status, receita, época, duração, validação, holdout e média macro das classes concluídas, com contagem do denominador; falhas/métricas ausentes não viram zero. Checkpoints são centralizados em best_models/<classe>/best_models/. Nenhum treino real foi executado para validar essa automação: passaram dry-runs reais das cinco classes, compatibilidade dos 15 comandos e simulações de lote completo/parcial com conferência de CSV/Markdown/média. A avaliação externa adicional fica desligada nos presets para não aplicar defaults diferentes de receita ou encoder.

#### D-18 — Notebook descritivo para avaliação do teste completo
- Data / responsável: 2026-10-26 / solicitação explícita do usuário, implementação por GitHub Copilot.
- Estado: aceita como ferramenta de avaliação; não converte o teste reutilizado em holdout independente.
- Decisão: usar [avaliacao_completa_pixel_heads.ipynb](../avaliacao_completa_pixel_heads.ipynb) para ler a receita e as estatísticas do checkpoint e avaliar todas as imagens/pixels do teste, stride 1 e margem 0. AUPRO global, AP, F1, IoU/Dice e métricas por imagem voltam a ser calculados nesta avaliação solicitada; isso não muda a política de treino D-14.
- Por quê: os avaliadores antigos têm defaults anteriores aos presets e um AUPRO por imagem não equivalente ao global. A análise completa precisa reproduzir padding/TTA/alinhamento/níveis e distinguir ranking, segmentação em limiar e classificação por imagem.
- Evidência: checkpoint inference_recipe/inference_level_stats e training_recipe; funções compartilhadas de NLL/agregação e AUPRO global. O preflight confirmou as cinco heads de run_20261025_181523 sem executar inferência real.
- Alternativas consideradas e motivo para não escolhê-las: aplicar defaults antigos silenciosamente (muda a função); manter todos os mapas em RAM (risco de paginação); escolher limiar no teste e chamá-lo operacional (otimismo). Mapas grandes usam memmap por classe. O modo padrão histogram percorre todos os pixels, mas aproxima AUROC/AP e limiares com bins; exact usa sklearn para AUROC/AP e pode exigir muita RAM. F1/mIoU/Dice de limiar são contados diretamente nos pixels.
- Riscos/limites: teste completo inclui imagens de seleção. Limiares vêm da validação registrada quando disponível; na ausência, são declarados oráculos de teste. Scores não são probabilidades calibradas. ICs bootstrap reamostram imagens com bins reduzidos, condicionados à receita/coorte; não cobrem treino/seeds nem têm IC AUPRO. O possível colapso de AUPRO global não é corrigido automaticamente; legacy fica separado e opcional. Latência é observada, sem benchmark padronizado.
- Verificação que poderia refutar: comparar métricas sintéticas ao sklearn, validar coorte inteira em cflow_original e exportações em pipeline simulado, depois executar no kernel da pesquisa sobre as heads reais e auditar imagens/máscaras/contagens antes de interpretar resultados.
- Resultado da verificação / próxima revisão: todas as células compilaram; testes sintéticos de AUROC/AP, F1 macro/micro/weighted e mIoU passaram; pipeline simulado raw/std/original gerou figuras, CSV/Excel/JSON/HTML. Importação e descoberta das cinco heads reais passaram; a inferência completa não foi iniciada durante a criação do notebook. Dependências IPython/ipykernel/nbformat foram instaladas no Conda da pesquisa.

#### D-19 — Corrigir a máscara de objeto antes de retreinar sem fundo (ablação de máscara)
- Data / responsável: 2026-10-30 / escolha do usuário (opção 2), plano por GitHub Copilot.
- Estado: proposta; nenhum código, treino ou inferência executado.
- Decisão: investigar a remoção de fundo com uma máscara de objeto ampliada de forma controlada, conforme o [plano de ablação de máscara](plano_ablacao_mascara_objeto.md). Estratégia primária: dilatação morfológica, preferida à envoltória convexa. Máscaras regeneradas como união de instâncias YOLO, sem descartar imagens. Inventário de teste e split da rodada com fundo. Braço A com SEDifferNet congelado (D-01); braço B (retreino do encoder) só para finalistas.
- Por quê: nas rodadas D-18 `pixel_notebook/run_with_background` × `run_no_background`, a queda de Pixel AUROC acompanha o GT apagado: glass 41.7% do GT no preto (0.924→0.680), polymer 69.5% (0.867→0.667), contra lightning 9.3% (0.866→0.921), vari 8.4% e yoke 17.4% (estáveis). O teste sem fundo também perdeu mais anômalas que normais (glass 87→66, polymer 82→63). `create_dataset.py` usa só a primeira instância e descarta imagens sem detecção.
- Evidência: diagnóstico exploratório de 2026-10-30 sobre máscaras reconstruídas (seção 2 do plano). Com área comparável, a dilatação perde GT igual ou menor que a envoltória, exceto polymer (leve vantagem da envoltória). Nenhuma estratégia sobre a máscara atual baixa o GT perdido de glass/polymer abaixo de ~23%. Usou o teste inteiro; serve só para desenhar o grid.
- Alternativas consideradas e motivo para não escolhê-las: envoltória convexa (sem parâmetro, não cobre fora do contorno externo, inclui fundo arbitrário em peças em Y); mascarar só o score (opção 1, adiada por escolha do usuário); mascarar a loss (opção 3, adiada).
- Riscos/limites: pesos YOLO não encontrados localmente (fallback com máscara atual limita glass/polymer); missing-cap é ausência de peça, logo pode não haver máscara adequada; uma seed; teste já inspecionado.
- Verificação que poderia refutar a decisão: estágio `audit` com máscaras regeneradas mostrando GT perdido de validação > 5% para todas as estratégias; ou ΔPixel AUROC de validação < 0 contra a rodada com fundo no mesmo inventário/split.
- Resultado da verificação / próxima revisão: pendente; revisar após `masks` + `audit`.
- Atualização operacional 2026-10-30 (pedido do usuário): estágio `masks` implementado em [`ablation_bg_removal/`](../ablation_bg_removal/README.md) com **YOLO26-seg** (Ultralytics 8.4.194, instalado no env da pesquisa sem alterar torch/numpy), treinado por classe em uma **conversão do dataset sem fundo**: imagem original + polígonos dos pixels não-pretos de `train/good`; split interno 85/15, seed 42; teste nunca usado. Conversão executada: glass 928/164, lightning 330/58, polymer 746/132, vari 402/71, yoke 3882/686 (train/val YOLO), IoU polígono×máscara médio 0.988–0.994, 4 imagens rejeitadas por área; Ultralytics leu todos os rótulos. Geração de máscara: união de instâncias, `retina_masks`, `conf=0.10`, sem descarte (máscara cheia + log). Risco aceito: os rótulos herdam a falha de primeira instância/corte do segmentador antigo; o audit de GT perdido decide. Treino YOLO e geração de máscaras ainda não executados; testes sintéticos (6) e dry-runs passaram.
- Atualização de implementação 2026-10-30 (pedido do usuário: "implemente tudo"): etapas 4–8 implementadas em `ablation_bg_removal/` (`04_audit_masks.py`, `05_select_strategies.py`, `06_train_nf_head_masked.py`, `07_retrain_encoder_masked.py`, `08_report.py`, launcher `run_pipeline.ps1 -Stage audit|select|frozen|retrain|report|ablation`). Decisões de implementação, sem mudar o protocolo: (a) máscara aplicada por **dataset materializado** com os mesmos nomes e árvore, então o treinador, o split e o notebook D-18 rodam sem mudança em `core/`; (b) estratégias definidas no grid 448; (c) audit e relatório importam `stratified_test_split` do treinador, e heads com `split_manifest` diferente do audit são recusadas; (d) a referência com fundo é a head existente de `treino_nf_heads/run_20261025_181523`, com conferência de `selection_validation_paths`; (e) métricas de pixel por histograma de 65 536 bins (validadas contra o sklearn) e bootstrap pareado estratificado; (f) o critério da seção 3.6 do plano é aplicado automaticamente só na validação; (g) braço B herda a seleção de encoder por Image AUROC no teste inteiro (`encoder_selection=test_full`, ressalva no relatório; `--encoder_pick latest` como alternativa); (h) `--source nobg` como máscara de controle sem GPU. Verificação: 22 testes unitários e smoke test em vari-grip com máscaras `nobg` (audit, poda, materialização, `--print_recipe`, igualdade de split, etapa 8 em 16 imagens com métricas/IC/figura). Artefatos do smoke removidos. Nenhum treino de modelo nem inferência completa executados.

#### D-20 — vari-grip: usar `dil16` (override da regra de poda de D-19)
- Data / responsável: 2026-10-30 / decisão do usuário, registrada por GitHub Copilot.
- Estado: aceita para vari-grip; proposta de emenda da regra para as demais classes (ver abaixo).
- Decisão: no braço A de vari-grip (rodada `ablacao_mascara_objeto/run_20261030_215319`), treinar `dil16__black`, `dil16__imagenet_mean` e a referência `raw__black`, no lugar das finalistas que a regra de D-19 escolheria (`dil8`, `hull`). Registrado em `finalists.json` com `override=["dil16"]`.
- Por quê: na validação, `dil8` e `dil16` perdem praticamente o mesmo GT (2.12% × 2.05%) e o resíduo é constante de `dil16` a `dil64`. Nos exemplos, o resíduo é defeito em um **segundo vari-grip** vizinho ou cortado na borda, que o segmentador não marca, e nenhuma dilatação local resolve isso. `dil16` custa +5 pp de área, mas afasta o contorno artificial do objeto em ~3 células da grade 96 da NF head, o que reduz a contaminação das features da peça pela borda preta e dá margem para erros do YOLO26 (recall de instância da máscara 0.83). `hull` foi descartada pela análise de D-19 (sem parâmetro, inclui fundo arbitrário em peças em Y).
- Evidência: audit `run_20261030_215319/vari-grip/audit/` (máscaras `vari-grip_yolo_20261030_215259`, YOLO26s-seg com mask mAP50-95 0.819 na validação interna do YOLO, nenhuma imagem sem detecção). Somente geometria de máscara; **nenhuma métrica de modelo foi vista** antes da decisão.
- Alternativas consideradas e motivo para não escolhê-las: seguir a regra (`dil8`, `hull`) faria a regra de menor área favorecer o contorno mais próximo do objeto, justamente o artefato da hipótese 2; incluir `dil8__black` como controle de raio custaria +1 head (opcional).
- Riscos/limites: o audit também imprimiu o holdout. A geometria do holdout (GT perdido de 6.1% com `dil16`, concentrado em uma imagem de objeto cortado na borda) foi vista, mas não usada na escolha; isso vai aparecer nas métricas de holdout. O resíduo de vizinhos só sai com `bbox20` (≈ sem remoção de fundo). Uma seed.
- Emenda proposta para as próximas classes (decidir **antes** dos audits delas): menor `dil<r>` com `r ≥ 16` e GT perdido de validação ≤ 5%, mais a segunda menor área elegível como contraste; se nenhuma passar, manter o fallback de D-19.
- Verificação que poderia refutar: ΔPixel AUROC de validação de `dil16__*` < 0 contra a referência com fundo, ou `contour_fpr` ≫ `offcontour_fpr` no relatório (o contorno continua dominando os FPs).
- Resultado da verificação / próxima revisão: após `-Stage frozen` + `-Stage report` de vari-grip.
- Resultado (2026-10-31, `run_20261030_215319/summary.md`; validação = 62 imagens, 16 anômalas; holdout = 92): **refutada como ganho no braço A.** Pixel AUROC de validação (Δ contra a head com fundo 0.8837, IC95 pareado): `dil16__black` 0.8478 (−0.036 [−0.067, −0.010]; ΔAUPRO −0.108; FPR na faixa de contorno 0.041 × 0.007 fora dela, ou seja, a borda preta gera FP); `dil16__imagenet_mean` 0.8820 (−0.002 [−0.038, +0.027]; ΔAUPRO −0.003; FPR de contorno 0.013 × 0.010, igual à referência), que é **empate**; `raw__black` 0.9010 (+0.017 [−0.015, +0.046]) com GT perdido de 6.3%, inverte de sinal no holdout (−0.014). Nenhuma variante passa o critério pré-registrado. AUROC dentro do objeto (região `dil32`, sem IC): referência 0.790, `imagenet_mean` 0.738, `raw__black` 0.780, `black` 0.649. A máscara não melhorou a localização na peça com o encoder congelado. Diagnóstico de imagem: o SEDifferNet com fundo recebendo a imagem mascarada **colapsa** (Image AUROC de validação 0.961 → 0.451 / 0.552 / 0.454 para `dil16__black` / `dil16__imagenet_mean` / `raw__black`; IC95 todos < −0.25). A entrada mascarada é fora da distribuição para o encoder com fundo, o que também explica a perda dentro do objeto no braço A.

#### D-21 — Consequências da ablação de máscara em vari-grip (itens 1–3 aceitos em D-23; item 4 substituído)
- Data / responsável: 2026-10-31 / análise de GitHub Copilot sobre o relatório de D-20; decidida pelo usuário em D-23.
- Estado: itens (1)–(3) aceitos; item (4) substituído por D-23 (braço B sempre roda).
- Decisão proposta: (1) registrar vari-grip como "remoção de fundo não melhora a localização" no protocolo de encoder congelado; a head com fundo continua sendo a de vari-grip; (2) **remover o preenchimento `black` das variantes candidatas** nas próximas classes, mantendo só `imagenet_mean` (e `raw__black` apenas como referência do experimento sem fundo); (3) tratar o braço A como teste da *head* e não da remoção de fundo em si. Qualquer conclusão sobre "remover o fundo melhora o modelo" exige o braço B (encoder retreinado com a mesma máscara), porque o encoder com fundo não consome imagens mascaradas (colapso do Image AUROC); (4) não rodar o braço B em vari-grip. Pela regra ele nem seria disparado (nenhum Δ de validação ≥ 0), e o experimento antigo sem fundo já aproximava esse braço (0.900 → 0.897). Priorizar classes em que a remoção tem mais a ganhar: lightning (+0.055 na rodada D-18 sem fundo) e glass/polymer (onde a máscara antiga apagava o defeito).
- Por quê: preto gera FPs no contorno (4× a taxa fora dele) e piora AUROC/AUPRO com IC excluindo zero; `imagenet_mean` elimina o excesso de contorno e empata. O colapso do encoder mostra que o braço A mede uma head treinada sobre features deslocadas, não o benefício potencial de um modelo sem fundo.
- Evidência: `resultado_analise_final/ablacao_mascara_objeto/run_20261030_215319/{summary.md,summary.csv,vari-grip/eval/}`; uma seed; holdout não independente (D-18).
- Alternativas: rodar o braço B em vari-grip mesmo assim (≈ 200 épocas de SEDifferNet + head por variante; ganho esperado baixo dado o histórico); testar feather em `black` (desnecessário, já que `imagenet_mean` resolve o contorno).
- Riscos/limites: n de validação pequeno (16 anômalas). O ruído do Δ é ~±0.03 a 0.04 e diferenças menores que isso não são detectáveis em vari-grip. O AUROC dentro do objeto não tem IC. Uma classe só para a regra do preenchimento. Para as demais classes, o encoder congelado vai dar o mesmo problema de distribuição, então o braço B deve ser planejado desde o início se o objetivo for a conclusão sobre o modelo.
- Verificação que poderia refutar: em outra classe, `black` sem excesso de FP de contorno ou com Δ de validação ≥ `imagenet_mean`; ou braço B em lightning/glass/polymer sem ganho de validação contra a referência com fundo.
- Resultado da verificação / próxima revisão: pendente da escolha do usuário sobre a próxima classe e sobre o braço B.

#### D-22 — Nível de imagem: comparação pareada SEDifferNet com fundo × sem fundo
- Data / responsável: 2026-10-31 / pedido do usuário (foco em nível de imagem), implementação por GitHub Copilot.
- Estado: aceita como ferramenta; resultado parcial (só vari-grip).
- Decisão: comparar `final_models/SEDiffernet` (com fundo) e `final_models/SEDiffernet_noBg` (fundo preto) com `ablation_bg_removal/09_image_level_compare.py` (`run_pipeline.ps1 -Stage image`): cada encoder no seu domínio (`bg@orig`, `nobg@nobg`), **nas mesmas imagens** (interseção por nome normalizado "Fotos " = "Fotos-"), score DifferNet com 64 rotações fixas, Δ pareado com IC95 (bootstrap estratificado, 2000), split val/holdout do protocolo. Diagnósticos opcionais: `--cross` (cada encoder no domínio do outro) e `--extra` (encoder sem fundo em dataset mascarado pela YOLO26).
- Por quê: os números gravados nos checkpoints (com × sem fundo: glass 0.835/0.817, lightning 0.991/0.966, polymer 0.968/0.962, vari 0.977/0.958, yoke 0.984/0.963) vêm de testes diferentes (o sem fundo perdeu imagens, mais anômalas) e de seleção de época no próprio teste, sem IC.
- Evidência (vari-grip, `resultado_analise_final/imagem_bg_vs_nobg/run_20261031_132820/`, 154 imagens, 40 anômalas): `bg@orig` 0.9770 (reproduz o checkpoint; validação 0.9606 = relatório D-20); `nobg@nobg` 0.9544, Δ −0.023 [−0.051, +0.003], com validação −0.026 e holdout −0.019. A queda é consistente em sinal, mas não significativa em 95%. Diagnósticos: `bg@nobg` 0.598 e `nobg@orig` 0.764, ou seja, cada encoder colapsa fora do próprio domínio; `nobg@yolo_raw` 0.933 (−0.044 [−0.086, −0.008]), em que a máscara YOLO26 justa, diferente da usada no treino, já degrada o encoder sem fundo; `nobg@yolo_dil16_mean` 0.686, de modo que o encoder sem fundo não generaliza para fundo cinza/dilatado. Portanto, o encoder é sensível ao *estilo exato* da máscara do treino.
- Alternativas: comparar os valores dos checkpoints (testes diferentes, sem IC); retreinar já (caro antes de saber se a queda é real).
- Riscos/limites: os dois encoders escolheram a época pelo AUROC no próprio teste. Os AUROCs absolutos são otimistas e nenhuma partição é limpa; o viés de seleção pode ter magnitude comparável ao Δ (~0.02). Uma seed. A interseção exclui as imagens que a segmentação antiga descartou (sem fundo não há como pontuá-las).
- Verificação que poderia refutar: nas 5 classes, Δ `nobg@nobg` − `bg@orig` com IC95 cobrindo 0 em todas (sem efeito) ou acima de 0 (ganho); depois, um retreino com seleção de época na validação (opção 3 da análise de 2026-10-31) para eliminar o viés de seleção.
- Resultado da verificação / próxima revisão: rodar `-Stage image` nas 5 classes (~35 min sem `-Cross`, ~70 min com).

#### D-23 — Remover `black` e tornar o braço B parte fixa do pipeline
- Data / responsável: 2026-10-31 / decisão do usuário ("remova o black, e adicione o retreino do SEDifferNet de imagem; após o retreino a avaliação de pixel deve ser feita novamente"), implementação por GitHub Copilot.
- Estado: aceita. Aprova os itens (1)–(3) de D-21 e **substitui o item (4)**: o braço B passa a rodar sempre, não só quando o braço A empata ou ganha.
- Decisão:
  - **Preenchimento.** Nas próximas rodadas do `ablation_bg_removal/run_all.py`, as variantes candidatas usam só `imagenet_mean` (`fills='imagenet_mean'`; default do `05` idem). A referência `raw__black` sai do default (`include_reference_nobg=False`).
  - **Braço B.** A etapa `retrain` (`07`) entra no pipeline padrão, antes do `report`, sobre todas as candidatas de `finalists.json`. Ela retreina o SEDifferNet na imagem mascarada e treina a NF head (receita `general`) sobre esse encoder novo.
  - **Reavaliação de pixel.** O `report` (`08`) roda depois e avalia as heads dos braços A e B contra a head com fundo, com a mesma métrica, o mesmo split e o mesmo critério da seção 3.6.
  - **Avaliação de imagem.** A etapa `image` (`09 --encoder`) acrescenta `retrained_<variante>@masked_<variante>`, pareada contra `bg@orig`.
  - **Tolerância a falhas.** Uma falha no braço B não bloqueia o relatório do braço A.
- Por quê:
  - D-20 mostrou que `black` gera FP no contorno.
  - D-20/D-22 mostraram que nenhum encoder transfere entre estilos de máscara. O braço A só mede a head sobre features deslocadas, então a pergunta "remover o fundo melhora o modelo" exige o encoder retreinado com a mesma máscara.
  - A regra antiga de disparo nunca rodaria o braço B: em vari-grip, nenhum Δ de validação do braço A ficou ≥ 0.
- Evidência: D-20 (FP de contorno e colapso do encoder com fundo nas imagens mascaradas) e D-22 (`nobg@yolo_dil16_mean` 0.686). Smoke test em vari-grip (cópia temporária da rodada `run_20261030_215319`, depois removida):
  - `05` gerou só `dil16__imagenet_mean`;
  - o `07`, com 1 época de pré-treino + 1 de meta-treino + 1 época de head, terminou com `SUCCESS` e `split=ok`;
  - o `08 --limit 16` avaliou A/B;
  - a etapa `image` incluiu a condição retreinada (0.803 com 1 época; só valida a infraestrutura, não é resultado).
- Alternativas: manter a regra condicional de D-19, mais barata, mas que não responde à pergunta do usuário; retreinar também `raw__black` como controle do experimento antigo sem fundo, já aproximado por `final_models/SEDiffernet_noBg`.
- Riscos/limites:
  - O SEDifferNet do braço B escolhe a época pelo Image AUROC no **teste inteiro** (`encoder_pick=best_test`, igual aos `final_models`). As métricas de imagem dele são otimistas e o holdout não é limpo para imagem; a head de pixel continua selecionada só na validação. `arm_b_encoder_pick='latest'` evita a seleção no teste.
  - O custo é de ~200 épocas de SEDifferNet por classe (de horas a mais de um dia em yoke).
  - Uma seed.
- Verificação que poderia refutar: ΔPixel AUROC de validação do braço B `dil16__imagenet_mean` < 0 com IC excluindo 0 contra a head com fundo, e Image AUROC `retrained@masked` abaixo de `bg@orig` nas classes rodadas.
- Resultado da verificação / próxima revisão: após `run_all.py` com as etapas `select,frozen,retrain,report,image` nas classes escolhidas.
- Resultado parcial (2026-11-01, vari-grip, rodada `ablacao_mascara_objeto/run_20261101_011735`): ver D-24. No pixel, o braço B é **idêntico** ao A por construção. No nível de imagem, o encoder retreinado em `dil16__imagenet_mean` empata com o encoder com fundo.

#### D-24 — O retreino do SEDifferNet não muda as features da NF head (braço B de pixel = braço A)
- Data / responsável: 2026-11-01 / achado de GitHub Copilot ao analisar a rodada `run_20261101_011735` a pedido do usuário.
- Estado: achado verificado. Consequência aceita pelo usuário em 2026-11-01: opção "braço B só no nível de imagem".
- Implementação: `07_retrain_encoder_masked.py` não treina mais a head B por padrão (`status=ENCODER_ONLY` no índice; `--train_head` restaura o comportamento antigo); `run_all.py` passa o encoder ENCODER_ONLY para a etapa `image`. A rodada `run_20261101_011735` já tinha iniciado o `07` de lightning com o código antigo, então essa classe ainda treina a head B redundante.
- Achado:
  - Em vari-grip, as linhas de pixel dos braços A e B saíram iguais em todas as casas decimais. Os `maps.npy` de `eval/A_*` e `eval/B_*` têm o mesmo MD5, e os pesos das duas NF heads (`best_pixel_auroc.pt`) também são iguais.
  - O treinador recebeu de fato o encoder retreinado: o log da head B carrega `arm_B/.../best_img_auroc.pt`.
  - A causa é que `core/train.py` só otimiza `model.nf.parameters()`. Comparando o `state_dict` do encoder com fundo (`final_models/SEDiffernet/vari-grip/..._epoch_163.pt`) com o do retreinado, `alexnet.features`, `alexnet.classifier`, `cbam*` e `simsa*` são **bit a bit iguais**; só o `nf` (flow de imagem) difere.
  - A NF head de pixel usa apenas as ativações L1/L2/L3 do backbone (AlexNet ImageNet + blocos SE com a inicialização da seed 42, nunca treinados). Então, com a mesma seed, o mesmo dataset mascarado e a mesma receita, a head B reproduz a head A.
- Consequências:
  1. **Pixel.** "Retreinar o SEDifferNet" não altera o nível de pixel neste projeto. O braço B de pixel é redundante, gastou ~95 min em vari-grip e vai gastar o mesmo por classe. A única alavanca de pixel da remoção de fundo é a entrada mascarada (braço A).
  2. **Imagem.** O braço B continua sendo a medida certa: o retreino adapta o flow de imagem ao domínio mascarado.
  3. **Interpretação de D-01.** "Encoder congelado" na NF head equivale, na prática, a "AlexNet ImageNet + SE não treinados (seed 42)", e não a um encoder especializado por classe. Vale para todas as ablações de pixel anteriores e convém declarar isso no texto.
  4. **D-20 se reproduz.** A head A `dil16__imagenet_mean` desta rodada repete exatamente a de `run_20261030_215319` (0.8820; Δ −0.0017 [−0.0376, +0.0266]).
- Resultado vari-grip (154 imagens, 40 anômalas):
  - **Pixel** (A = B), Δ contra a head com fundo: validação −0.002 [−0.038, +0.027]; holdout −0.021 [−0.067, +0.011]. Não passa o critério.
  - **Imagem** (`09`), `retrained_dil16__imagenet_mean@masked`: 0.9605 contra 0.9770 de `bg@orig`, Δ −0.016 [−0.039, +0.002] (validação +0.001, holdout −0.026). Época 108, escolhida no teste inteiro.
  - Ficou acima de `nobg@nobg` (0.9544, Δ −0.023), mas não supera o encoder com fundo.
  - O encoder com fundo recebendo a imagem mascarada colapsa (validação 0.55), como já se sabia.
- Proposta original (decidida acima: opção de pular a head B):
  - **(a)** No `07`, retreinar só o encoder e reaproveitar a head do braço A no braço B, ou pular a head B e reportar o braço B só no nível de imagem. Isso economiza 1–2,5 h por classe.
  - **(b)** Se o objetivo for que a remoção de fundo mude o pixel por meio do encoder, seria preciso treinar o backbone ou os SE (fine-tuning). Isso está fora do escopo de D-01 e exigiria uma nova decisão.
- Verificação que poderia refutar: em outra classe, pesos de backbone diferentes entre o encoder retreinado e o de referência, ou head B ≠ head A com a mesma seed (por exemplo, por não determinismo de GPU). Confirmar com o mesmo teste de MD5 em lightning.

#### D-25 — Treino do SEDifferNet: avaliação por época com 64 rotações fixas; FP32 mantido
- Data / responsável: 2026-11-01 / pergunta do usuário ("o treino dessa ablation está diferente do treino normal? deveria estar igual"); decisão do usuário, implementação por GitHub Copilot.
- Estado: aceita e implementada em `core/train.py`.
- Contexto:
  - O braço B (`ablation_bg_removal/_train_encoder_runner.py`) já usava o treino normal atual (`main.py` + `core/train.py` + `config.py`), com os mesmos hiperparâmetros do `config.py`: lr 2e-4, `n_transforms` 4, 64 de teste, 30 pré-épocas + 170 meta, 8 coupling blocks, `fc_internal` 2048, `clamp_alpha` 3, rotações on. É o mesmo código que treinou `final_models/SEDiffernet_noBg` (14–17/10/2026).
  - Já `final_models/SEDiffernet` (com fundo; abr–mai/2026; código na raiz até `da7abe2`) foi treinado com duas diferenças:
    - **(1) AMP.** `autocast` fp16 + `GradScaler`, removido no commit `de2a9e4` "ajuste treinamento" de 14/10/2026.
    - **(2) Avaliação por época com rotações fixas.** O código antigo usava `test_loader.dataset.get_fixed = True`, com 64 rotações fixas. Desde a refatoração para `core/` (`1f86bbe`, 08/09/2026), o teste recebe o transform de treino (`RandomRotation`), e o Image AUROC por época, que escolhe `best_img_auroc.pt`, fica estocástico.
  - Lightning com fundo (`..._100_1_epoch_41.pt`, 09/04/2026) veio de uma versão anterior do `config.py`, que não dá para reconstruir com precisão.
- Decisão: restaurar (2) no treino normal (`get_fixed=True` só durante a avaliação de cada época) e manter FP32 (não restaurar AMP). Não retreinar a referência com fundo.
- Por quê: com rotações aleatórias, a seleção de época mede ruído. As 64 rotações fixas também são o score usado em `09`, no `08` e em `core.eval_pipeline.image_score`, então a época passa a ser escolhida pela mesma métrica que é reportada. O FP32 foi escolha explícita do usuário em `de2a9e4`.
- Evidência (encoder B de vari-grip, `run_20261101_011735`, imagens `dil16__imagenet_mean`):
  - O laço de avaliação do `train.py` com `get_fixed=True` deu Image AUROC 0.9605, idêntico ao `09`.
  - Com rotações aleatórias, duas passadas deram 0.9645 e 0.9548 (±0.005 de ruído) no mesmo checkpoint.
- Consequências:
  - Os encoders treinados de 08/09 a 01/11/2026 escolheram a época com AUROC ruidoso: `SEDiffernet_noBg`, braço B de vari-grip, e o de lightning, que estava em execução com o código antigo. Os scores reportados pelo `08`/`09` não mudam, porque sempre usaram rotações fixas; só a época escolhida pode não ser a melhor.
  - Continua existindo a diferença de AMP contra `final_models/SEDiffernet`. A comparação com fundo × sem fundo/mascarado no nível de imagem mistura o efeito da máscara com fp16 × fp32 e com o código de abr–mai, e isso deve ser declarado.
  - A seleção de época continua sendo no teste inteiro (o otimismo de D-22 permanece).
- Alternativas: restaurar também o AMP (igual à referência, mas reverte a decisão do usuário de `de2a9e4`); aplicar o protocolo antigo só no braço B (deixaria noBg e o treino normal divergentes); retreinar a referência com fundo com o código atual (~8 h/classe; recusado por ora).
- Verificação que poderia refutar: retreinar um encoder com o código novo e checar que o AUROC gravado em `best_img_auroc.pt` coincide com o do `09` para o mesmo checkpoint.
- Resultado da verificação (2026-11-02): o encoder B do polymer (treinado depois de D-25) gravou `best_img_auroc` 0.9328, e o `09` mediu 0.9328 (`própria`, n=317) no mesmo checkpoint. **Confirmada.**

#### D-26 — Resultado da ablação de máscara em 3 classes (vari-grip, lightning, polymer)
- Data / responsável: 2026-11-02 / análise de GitHub Copilot a pedido do usuário ("dada as classes rodadas, o que podemos inferir?").
- Estado: achados registrados. As consequências para o protocolo estão propostas e aguardam o usuário.
- Rodada: `resultado_analise_final/ablacao_mascara_objeto/run_20261101_011735` (YOLO26s-seg por classe; `dil16__imagenet_mean` forçado em todas as classes; referência de pixel `treino_nf_heads/run_20261025_181523`; uma seed; holdout não independente).
- Correções operacionais nesta análise:
  - (i) A etapa `image` do polymer tinha sido executada sem a condição retreinada, porque o processo `run_all.py` iniciado em 01/11 01:17 ainda tinha o filtro antigo (`status == 'SUCCESS'`) em memória. Reexecutada com `--stages image`, usando o cache para as demais condições.
  - (ii) O `08` passou a escrever `summary.md` com todas as classes do `summary.csv`; antes, só com a última processada.
- **Pixel** (braço A = braço B; D-24 confirmado em mais uma classe, pois lightning tem A e B idênticos em todas as métricas):

  | classe | GT perdido val (`dil16`) | Δ Pixel AUROC val [IC95] | Δ holdout [IC95] | Δ AUPRO val | AUROC no objeto (ref → A) | FPR contorno/fora (ref → A) | critério |
  |---|---|---|---|---|---|---|---|
  | vari-grip | 2.1% | −0.002 [−0.038, +0.027] | −0.021 [−0.067, +0.011] | −0.003 | 0.790 → 0.738 | 0.013/0.010 → 0.013/0.010 | falhou (delta) |
  | lightning | 0.0% | **+0.068 [+0.053, +0.087]** | **+0.075 [+0.058, +0.093]** | **+0.212** | 0.832 → 0.849 | 0.014/0.010 → 0.021/0.009 | **★ recomendada** |
  | polymer | **52.8%** | −0.225 [−0.313, −0.143] | −0.186 [−0.267, −0.109] | −0.131 | 0.793 → 0.749 | 0.024/0.009 → 0.007/0.010 | falhou (gt_lost, delta, aupro) |

- **Imagem** (Image AUROC na interseção; Δ contra `bg@orig`, IC95 pareado):

  | classe | `bg@orig` | `nobg@nobg` (preto antigo) | `retrained@masked` (braço B) | época B | encoder com fundo + imagem mascarada |
  |---|---|---|---|---|---|
  | vari-grip (154/40) | 0.977 | −0.023 [−0.051, +0.002] | −0.016 [−0.039, +0.002] | 108 (rotações aleatórias) | 0.686 |
  | lightning (135/39) | 0.990 | −0.030 [−0.058, −0.007] | −0.030 [−0.065, −0.002] | 12 (rotações aleatórias) | 0.351 |
  | polymer (283/63) | 0.964 | −0.003 [−0.020, +0.017] | −0.032 [−0.061, −0.007] | 18 (rotações fixas) | 0.750 |

- Inferências:
  1. **No pixel, o efeito da remoção de fundo depende da classe, e o principal preditor é o GT perdido pela máscara.**
     - Lightning (peça fina, com ~75% de fundo removido e 0% do defeito perdido) teve ganho grande e consistente em validação, holdout e AUPRO. É o mesmo sinal do experimento antigo sem fundo (+0.055; D-21).
     - Vari-grip (2% perdido) empata.
     - Polymer perde porque o defeito anotado fica no **suporte de fixação superior**, fora da peça segmentada (visto em `polymer.../audit/audit_examples.png`). Mesmo `dil64` perde 34% do GT, e só `bbox20` preserva (10% de perda na validação). O rótulo de "objeto" herdado do dataset sem fundo não contém a região do defeito, e nenhuma dilatação corrige isso.
  2. **O override `dil16` de D-20 não deveria ter valido para o polymer.** A regra de D-19 marcaria `mask_reliable=false` e cairia no fallback (`dil64`, `bbox20`). O `force_strategies='dil16'` global do `run_all.py` ignorou isso. O resultado do polymer é válido como "máscara errada degrada", não como teste justo da remoção de fundo.
  3. **No nível de imagem, a remoção de fundo não ajudou em nenhuma das 3 classes.**
     - O encoder retreinado na imagem mascarada fica entre 0.016 e 0.032 abaixo do encoder com fundo, com IC excluindo 0 em lightning e polymer. É o mesmo padrão do `nobg@nobg`.
     - O SEDifferNet de imagem se beneficia do contexto ou fundo. A explicação não é só falta de treino: o pico de AUROC do encoder mascarado ocorre cedo (épocas 12–18 em lightning/polymer) e depois cai (polymer: 0.933 → 0.921 na época 170).
     - Ressalvas: (a) fp16 × fp32 e o código de abr–mai na referência (D-25); (b) épocas escolhidas no teste para ambos (otimista para ambos); (c) no polymer, parte da queda pode vir do defeito apagado pela máscara.
  4. **Todo encoder colapsa fora do próprio domínio** (3/3 classes): com fundo + imagem mascarada fica em 0.35–0.75. Uma única imagem mascarada não serve aos dois níveis com o encoder de referência.
  5. **Arquitetura sugerida pelos dados:** usar o encoder com fundo no nível de imagem e, no pixel, a head com entrada mascarada **só onde a máscara cobre o defeito** (lightning). Isso é possível porque a head de pixel não depende do flow de imagem (D-24); só mudam a imagem de entrada e a head.
  6. **O contorno ainda gera FP com `imagenet_mean` em lightning** (0.021 × 0.009 fora), mas o ganho líquido é grande. `dil32` pode reduzir o efeito (não testado).
- Proposta (aguarda o usuário):
  - **(a)** Adotar `dil16__imagenet_mean` no pixel para lightning, como receita exploratória (holdout não independente; uma seed).
  - **(b)** Polymer: não mascarar (manter com fundo) ou testar `bbox20`. Tornar o override condicional: `force_strategies` só onde o GT perdido de validação for ≤ 5%; nas demais classes, seguir a regra/fallback de D-19.
  - **(c)** Glass e yoke: rodar primeiro só `convert,yolo,masks,audit` e decidir pela perda de GT antes de treinar heads.
  - **(d)** Nível de imagem: manter o encoder com fundo; não investir mais no braço B, salvo se surgir uma hipótese nova (por exemplo, fundo borrado em vez de removido).
- Verificação que poderia refutar: em glass/yoke, ganho de pixel com GT perdido alto, ou perda com GT perdido ≈ 0; Image AUROC de `retrained@masked` ≥ `bg@orig` com encoder treinado no mesmo código (FP32, rotações fixas) e época escolhida na validação.

#### D-27 — Por que a remoção de fundo piora o nível de imagem mesmo sem perda de defeito (hipóteses e literatura)
- Data / responsável: 2026-11-02 / análise de GitHub Copilot a pedido do usuário.
- Estado: hipóteses, ordenadas pela evidência local. **Nenhuma confirmada por experimento dedicado.**
- Fenômeno:
  - Em vari-grip e lightning (GT perdido ≤ 2%), o Image AUROC do encoder retreinado na imagem mascarada fica 0.016–0.030 abaixo do encoder com fundo (D-26).
  - No pixel, vari-grip empata; dentro do objeto, cai de 0.790 para 0.738.
  - Lightning melhora no pixel (+0.068), então o efeito não é uniforme.
- H1 — **As features do backbone não se adaptam à imagem mascarada** (evidência forte).
  - O retreino só ajusta o flow (D-24). AlexNet/SE seguem com pesos do ImageNet, treinados em cenas naturais, e regiões de cor constante com bordas artificiais são entrada fora da distribuição para CNNs ("missingness bias"; Jain et al., ICLR 2022, arXiv:2204.08945).
  - Hooker et al. (ROAR, NeurIPS 2019) mostram que remover pixels sem retreinar confunde a perda de informação com a mudança de distribuição. Aqui o retreino é parcial (só o flow), então a mudança de distribuição persiste nas features.
  - Evidência local: o encoder com fundo nas imagens mascaradas colapsa (0.35–0.75). Retreinar o flow recupera a maior parte, mas não tudo.
- H2 — **O contorno e o preenchimento viram a principal fonte de variação modelada pelo flow** (evidência moderada).
  - O DifferNet faz média de features globais multiescala (Rudolph et al., WACV 2021). Com fundo constante, o vetor médio mistura "objeto" e "preenchimento" em proporção que depende da área e do recorte de cada imagem, e as primeiras camadas respondem fortemente à borda artificial.
  - Flows tendem a modelar estatísticas locais e de baixo nível em vez de semântica (Kirichenko et al., NeurIPS 2020; Schirrmeister et al., NeurIPS 2020). O NLL passa a refletir a silhueta e o ruído da máscara do YOLO, e não a textura do defeito.
  - Evidência local:
    - Excesso de FP no contorno no pixel: 4× com `black` (D-20) e 2.4× com `imagenet_mean` em lightning (D-26).
    - O encoder sem fundo é sensível ao estilo exato da máscara (D-22: `nobg@yolo_raw` −0.044).
    - A área do objeto **não** prediz o rótulo (abaixo). A variação de área/contorno é ruído, não sinal.
- H3 — **O fundo é contexto informativo, não só ruído** (plausível; sem teste local).
  - Modelos de reconhecimento usam o fundo, e removê-lo reduz a acurácia mesmo com o objeto intacto (Xiao et al., "Noise or Signal", ICLR 2021). O contexto regula a percepção de objetos (Oliva & Torralba, Trends Cogn. Sci. 2007).
  - Aqui, por exemplo, a iluminação e o balanço de cor da cena servem de referência para tons de corrosão (4 dos 6 defeitos do INSPLAD são corrosão; Vieira-e-Silva et al., arXiv:2311.01619).
  - O campo receptivo efetivo é aproximadamente gaussiano e centrado (Luo et al., NIPS 2016). Perto da borda, as features misturam peça e entorno, o que explica a queda de AUROC dentro do objeto em vari-grip mesmo sem perder GT.
- H4 — **Atalho no fundo** (Geirhos et al., Nat. Mach. Intell. 2020; Beery et al., ECCV 2018): imagens anômalas viriam de voos/locais com fundos diferentes, e o encoder com fundo pontuaria em parte o local.
  - Teste local sem inferência (`masks_report.csv`, teste completo): AUROC da área do objeto como score = 0.468 [0.358, 0.584] em vari-grip, 0.551 [0.453, 0.652] em lightning e 0.528 [0.453, 0.607] em polymer. AUROC do desvio de área contra o treino = 0.570 / 0.550 / 0.539 (ICs cobrem 0.5).
  - **O atalho de enquadramento/escala foi refutado.** O conteúdo do fundo continua sem teste.
  - Sinal fraco: a proporção de imagens com várias instâncias difere entre normais e anômalas (lightning 0.26 × 0.48; vari 0.18 × 0.03), o que muda quanto fundo sobra entre peças.
- H5 — **Parte da diferença é artefato de protocolo** (evidência moderada):
  - (a) A referência escolheu a época com o mesmo score que é reportado (64 rotações fixas). Os encoders B de vari e lightning escolheram com rotações aleatórias (antes de D-25), cujo ruído é de ±0.005. Isso cria uma "maldição do vencedor" assimétrica a favor da referência (Cawley & Talbot, JMLR 2010). O polymer B, escolhido com rotações fixas, ainda cai −0.032, mas lá há defeito apagado.
  - (b) fp16/AMP e código de abr–mai na referência × FP32 no braço B (D-25).
  - (c) Época escolhida no teste para ambos.
  - O tamanho do efeito (0.02–0.03) é da mesma ordem desses vieses, e em vari-grip o IC encosta em 0.
- Contraponto da literatura: em modelos generativos de pixel, as estatísticas de fundo **confundem** a likelihood (Ren et al., NeurIPS 2019, razão de likelihood), o que favoreceria remover o fundo. Isso é coerente com o **ganho no pixel** de lightning: a head local pontua cada posição, e a vegetação ou os cabos geram NLL alto. No nível de imagem, o DifferNet já opera sobre embeddings semânticos médios, onde flows funcionam melhor (Kirichenko et al. 2020), e o fundo pesa menos como ruído.
- Diagnósticos propostos (aguardam o usuário):
  - **(1) Atalho de conteúdo (H4).** Pontuar com o encoder com fundo as imagens **só com fundo** (objeto preenchido). AUROC ≫ 0.5 indicaria atalho. Só inferência, poucos minutos por classe; a máscara invertida exige implementação.
  - **(2) H2/H3.** Fundo **borrado** (`fill=blur`) ou máscara com `feather`, que preserva cor e iluminação e suaviza o contorno. Exige retreinar o flow (~8 h/classe).
  - **(3) H5.** Comparação justa: retreinar o encoder com fundo no código atual, com época escolhida na validação.
  - **(4) H1.** Ajuste fino do backbone/SE na imagem mascarada (fora de D-01).

## 12. Fontes técnicas principais

- [Análise científica consolidada das 5 classes](analise_cientifica_ablacao_nf_head_5classes.md): **fonte principal para publicação** a partir de 2026-10-13; tabela de ablação, receitas, estatística e reconciliação com documentos anteriores.
- [Achados vari-grip/yoke e consolidação das 5 classes](relatorio_achados_run_full_20261010_195638.md): corte em 2026-10-13, seleção AUROC-only, receita por classe e ganho consolidado.
- [Achados atualizados da rodada de outubro](relatorio_achados_run_full_20261007_123558.md): corte em 2026-10-10, 15 pós-hoc completos; consultar esta atualização antes de usar o snapshot de duas classes da seção 8.
- [Plano histórico da ablação](plano_ablation_nf_head.md): sequência de hipóteses e experimentos anteriores; contém conclusões de protocolo anterior, conferir seção e rodada antes de reutilizar números.
- [Relatório acadêmico](relatorio_academico_ablation_nf_head.md): escopo, nomenclatura, métricas e distinção explícita entre score CFLOW-AD e reprodução end-to-end.
- [Correções da CFLOW Pixel Head](cflow_correcoes_e_fase3.md): falhas históricas de pipeline e correções.
- `scripts/eval/ablation_nf_head.py`: split, score grid, escolha, OFAT, escada, holdout, bootstrap e saídas atuais.
- `scripts/train/pixel_train_from_pretrained.py`: split de época, validação, escolha de checkpoint e manifesto.
- `core/cflow.py`: implementação de agregações locais e `cflow_original`.
- `core/eval_pipeline.py`: métricas compartilhadas e implementação `compute_aupro_cflow`.
- Artefatos por rodada em `resultado_analise_final/ablacao_nf_head/`: fonte primária de resultados daquela rodada.

## 13. Regra final

Se uma decisão não tiver um motivo explícito, evidência rastreável e um teste que possa contradizê-la, ela ainda não está pronta para ser tratada como conclusão científica. Registre incerteza e limite de generalização junto com qualquer resultado.
