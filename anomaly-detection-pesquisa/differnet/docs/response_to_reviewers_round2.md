# Carta de Resposta aos Revisores — Rodada 2 (Stage 4 — REVISE)

**Artigo:** *Pós-Processamento Supera Retreino: Um Estudo Abrangente de Ablação da Cabeça de Normalizing Flow para Localização de Anomalias em Inspeção de Linhas de Transmissão*  
**Autor:** Teo Santos (Universidade de Brasília)  

---

Agradecemos imensamente ao Editor-Chefe e ao corpo de revisores pela recepção calorosa da versão expandida do manuscrito e pelo reconhecimento das melhorias substantivas trazidas pela grande tabela exaustiva, pela fundamentação mecanicista dos níveis convolucionais ($L_1, L_2, L_3$) e pela ancoragem nas teorias seminais de viés de textura versus forma e viés posicional de padding.

### Resposta ao Apontamento Residual do Devil's Advocate:
> **Devil's Advocate:** *Garantir que o leitor compreenda de imediato por que a recomendação do featnorm é "desligado por padrão", mesmo tendo ajudado em duas classes. A explicação do risco assimétrico (ganho moderado em duas classes versus colapso desastroso de −0,1226 de AUPRO em yoke) deve ser destacada em uma frase conclusiva no resumo das diretrizes.*

* **Resposta do Autor:** Incorporamos essa ponderação com precisão cirúrgica na Seção 5.4 (*Diretrizes de Implantação e Trade-off de Latência para VANTs*), Item 4, explicitando o conceito de **risco assimétrico**: enquanto o ganho do `featnorm` em classes favoráveis varia entre $+0{,}039$ e $+0{,}044$ AUROC, seu custo em classes desfavoráveis atinge uma perda ruinosa de $-0{,}0499$ de AUROC e $-0{,}1226$ de AUPRO. Diante desse perfil de risco assimétrico, o padrão prudente de engenharia de software e visão computacional é mantê-lo desativado por padrão, restringindo sua ativação a cenários onde a dominância de $L_1$ seja comprovada previamente.

O manuscrito final em Markdown (`artigo_ablation_cflow_differnet.md`) e em LaTeX (`paper_ablation_nf_head.tex`) foram devidamente atualizados.
