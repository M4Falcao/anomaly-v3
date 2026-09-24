# Carta de Resposta aos Revisores (Stage 4 — REVISE)

**Artigo:** *Pós-Processamento Supera Retreino: Um Estudo Abrangente de Ablação da Cabeça de Normalizing Flow para Localização de Anomalias em Inspeção de Linhas de Transmissão*  
**Autor:** Teo Santos (Universidade de Brasília)  

---

Agradecemos sinceramente ao Editor-Chefe e a todos os revisores pelos comentários detalhados, perspicazes e construtivos. Todas as recomendações foram integralmente incorporadas no manuscrito revisado (`paper_ablation_nf_head.tex`). Abaixo apresentamos a resposta ponto a ponto detalhada.

---

### Resposta ao Editor-Chefe (EIC)
> **EIC:** *O artigo aborda um problema concreto com metodologia rigorosa e dados reais do INSPLAD. Decisão: Minor Revision.*
* **Resposta:** Agradecemos a avaliação positiva e o reconhecimento do impacto e rigor do trabalho. Incorporamos todos os pontos solicitados pelos revisores de metodologia, domínio, engenharia e pelo Devil's Advocate.

---

### Resposta ao Revisor 1 (Metodologia e Estatística)
> **Comentário M1:** *Seria recomendável deixar explícito no texto que o teste de múltiplos seeds com intervalos de confiança seria o padrão-ouro estatístico, embora computacionalmente custoso.*
* **Resposta:** Concordamos plenamente. Atualizamos a Seção 5.4 (*Limitações*) para explicitar formalmente que, embora a medição de $\Delta = 0{,}0044$ AUROC tenha sido essencial para descartar ruído prático entre splits, o teste de múltiplos seeds aleatórios constitui o padrão-ouro estatístico para intervalos de confiança estritos.

> **Comentário M2:** *Na Tabela VIII, certifique-se de que a legenda explicite claramente o significado dos multiplicadores de AP (ex.: 35× trivial).*
* **Resposta:** A Tabela VIII já inclui a notação relativa ao trivial (ex.: $35\times$, $9{,}5\times$) e reforçamos na Seção 3.3 e na Seção 4.8 que, em virtude do forte desbalanceamento de classes no INSPLAD-seg (onde pixels anômalos representam de 0{,}018% a 4{,}6%), a razão entre o AP obtido e o baseline trivial da classe é a única forma fidedigna de comparação.

---

### Resposta ao Revisor 2 (Especialista em Detecção de Anomalias)
> **Comentário D1:** *Como o CFLOW-AD original usava per_level_prob, seria enriquecedor explicitar brevemente na discussão por que essa função saturava na ReLU da AlexNet.*
* **Resposta:** Excelente observação. Acrescentamos na Seção 5.1 uma explicação matemática direta: a transformação $\exp(\text{logsigmoid}(-\text{NLL})/C)$ foi concebida para features normalizadas de ResNet. Com ativações ReLU não limitadas de AlexNet, magnitudes de NLL atingem a ordem de centenas, comprimindo a função para saturação em valores extremos (0 ou 1) e degradando sua discriminação métrica (média de apenas 0{,}6992).

---

### Resposta ao Revisor 3 (Perspectiva e Engenharia de Aplicações)
> **Comentário P1:** *O uso de TTA por flips quadruplica o tempo de inferência por imagem (4 passes). O texto deve discutir brevemente o compromisso entre latência e ganho de +0.0120 AUROC em cenários embarcados de VANT.*
* **Resposta:** Agradecemos por apontar esse aspecto operacional essencial. Adicionamos na Seção 5.5 (*Diretrizes de Implantação e Trade-off de Latência*) uma diretriz específica para VANTs: em missões embarcadas em tempo real com restrição estrita de FPS, o TTA pode ser desativado com perda modesta de acurácia ($0{,}012$ AUROC), enquanto para auditorias em lote offline ele deve ser mantido ativado.

---

### Resposta ao Devil's Advocate
> **Desafio 1 (Generalização do Backbone):** *Seria o domínio do pós-processamento uma idiossincrasia da AlexNet que não se repetiria em ResNet ou ViT?*
* **Resposta:** Reforçamos na Seção 5.1 a base probabilística que independe do extrator: Normalizing Flows aprendem densidades exatas $p(\mathbf{x})$; a soma linear de NLL $\sum_k -\log p(\mathbf{x}_k)$ é equivalente a assumir independência condicional multiescala, preservando a calibração global métrica entre imagens normais e anômalas. Essa propriedade teórica decorre da formulação do flow, e não dos pesos convolucionais.

> **Desafio 2 (Overfitting em Yoke e Taxa de Aprendizado):** *O pico precoce na época 4 em yoke não seria reflexo de learning rate fixa excessiva (2e-4)?*
* **Resposta:** Concordamos e incluímos este ponto na Seção 5.4, apontando que para bases com milhares de amostras como o *yoke* (4.834 imagens normais), a escala de learning rate proporcionalmente reduzida e orçamentos de época restritos são caminhos promissores de investigação.

> **Desafio 3 (Treino vs Pós-Processamento):** *Não é exagero dizer que o treino pouco importa quando featnorm deu ganhos expressivos em duas classes?*
* **Resposta:** Mantivemos uma postura scrupulosamente honesta: o artigo não afirma que o treino não importa, mas documenta que o pós-processamento é a única camada que entrega ganhos consistentes em todas as classes, enquanto intervenções de treino como `featnorm` são altamente perigosas se adotadas cegamente (como comprovado pelo colapso de $-0{,}0499$ AUROC em *yoke*).
