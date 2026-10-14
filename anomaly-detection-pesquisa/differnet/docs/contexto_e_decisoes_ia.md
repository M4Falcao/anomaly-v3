# Contexto do Projeto e Registro de Decisões para IAs

**Escopo:** pesquisa de localização de anomalias por pixel usando uma NF head local sobre features do SEDifferNet, no INSPLAD-seg.  
**Atualizado:** 2026-10-10.  
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
