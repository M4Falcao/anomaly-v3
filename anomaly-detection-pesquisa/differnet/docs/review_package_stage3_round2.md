# Pacote de Revisão por Pares — Rodada 2 (Stage 3 — REVIEW)

**Manuscrito sob Avaliação:** *Pós-Processamento Supera Retreino: Um Estudo Abrangente de Ablação da Cabeça de Normalizing Flow para Localização de Anomalias em Inspeção de Linhas de Transmissão* (Versão Expandida com Grande Tabela e Mecânica dos Níveis)  
**Autor:** Teo Santos (Universidade de Brasília)  
**Painel Revisor:** 5 Revisores Independentes (EIC, Metodologia, Domínio, Perspectiva, Devil's Advocate)  

---

## 1. Relatórios dos Revisores

### Revisor 1 (Metodologia e Desenho Experimental)
* **Avaliação Geral:** Excelente. A inclusão da Tabela Multidimensional Exaustiva (Tabela 3) transformou o artigo em uma referência definitiva sobre o tema. A apresentação clara dos pontos de partida, cadeia incremental, efeitos pareados, impacto de normalização, acurácia isolada por nível, análise de *regret*, épocas de pico de sobreajuste e pontuações finais no *held-out* oferece uma transparência raramente vista na literatura de detecção de anomalias.
* **Aspectos Destacados:**
  - A decomposição de $L_1, L_2, L_3$ ancorada em campos receptivos ($11 \times 11$ vs. $>150 \times 150$ px) fundamenta solidamente a análise estatística.
  - O relato explícito de que a NLL `raw` venceu em 92,7% dos pares comparados (com 100% em duas classes) encerra qualquer dúvida sobre a validade do método de pontuação.
* **Veredito:** Aceite (Accept).

### Revisor 2 (Especialista de Domínio / Visão Computacional)
* **Avaliação Geral:** A integração das teorias de *viés de textura vs. forma* (Geirhos et al., ICLR 2019) e *codificação posicional implícita por padding* (Islam et al., Kayhan & van Gemert, CVPR 2020) elevou substancialmente o rigor científico da discussão. O artigo não apenas reporta números de ganho, mas explica o mecanismo biológico e representacional de por que $L_1$ é ideal para corrosão de textura rugosa, enquanto $L_3$ é indispensável para detectar peças faltantes em isoladores de vidro.
* **Veredito:** Aceite (Accept).

### Revisor 3 (Engenharia de Sistemas e Aplicações em VANTs)
* **Avaliação Geral:** O detalhamento das diretrizes operacionais de implantação na Seção 5.4 e a análise do custo de TTA por flips ($4\times$ tempo de inferência para $+0{,}0120$ AUROC) atendem com perfeição aos requisitos práticos de engenharia de visão para aeronaves remotamente pilotadas.
* **Veredito:** Aceite (Accept).

### Revisor 4 (Devil's Advocate — Desafiador da Tese)
* **Avaliação Geral:** Os desafios formulados na rodada anterior foram plenamente equacionados:
  - A fundamentação matemática da NLL como log-densidade conjunta sob independência condicional respondeu satisfatoriamente à questão da independência do *backbone*.
  - A explicação do colapso do `featnorm` em tensores de alta dimensionalidade ($L_2/L_3$) é coerente e matematicamente elegante.
  - O caso especial de `polymer-shackle` (onde $L_2$ é o nível mais forte) foi honestamente documentado sem tentativas de forçar uma regra cega.
* **Ponto Residual Menor:** Garantir que o leitor compreenda de imediato por que a recomendação do `featnorm` é "desligado por padrão", mesmo tendo ajudado em duas classes. A explicação do risco assimétrico (ganho moderado em duas classes versus colapso desastroso de $-0{,}1226$ de AUPRO em *yoke*) deve ser destacada em uma frase conclusiva no resumo das diretrizes.
* **Veredito:** Aceite com ajuste textual mínimo (Accept with minor note).

### Editor-Chefe (EIC — Síntese Editorial)
* **Decisão Editorial:** **ACCEPT (Aceito)**
* **Comentário de Síntese:** O manuscrito alcançou um padrão de excelência científica. Os dados estão perfeitamente auditados, a fundamentação teórica é densa e apoiada nas fontes seminais mais relevantes da área, e a grande tabela comparativa consolida um dos estudos de ablação mais exaustivos já realizados sobre Normalizing Flows em inspeção visual.
