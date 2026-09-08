# Matemática do Modelo SEDifferNet: Uma Abordagem Detalhada

O **SEDifferNet** é um modelo focado na detecção de anomalias visuais usando aprendizado semi-supervisionado (treinado exclusivamente com amostras "normais"). Ele funde duas grandes áreas do Deep Learning:
1. **Redes Convolucionais com Mecanismos de Atenção** (AlexNet + Squeeze-and-Excitation).
2. **Modelos Generativos Invertíveis** (Normalizing Flows).

Este documento destrincha a matemática por trás de cada componente da arquitetura.

---

## 1. Extração de Características Multi-Escala e SE Attention

A primeira fase do modelo consiste em extrair um vetor de características descritivas $\mathbf{y}$ a partir da imagem original $\mathbf{x}$.

### 1.1. Processamento Multi-Escala
Em vez de analisar a imagem em uma única resolução, o modelo aplica a rede extratora em múltiplas escalas de redimensionamento da imagem de entrada $s \in \{0, 1, 2, \dots, S-1\}$ (onde $S$ é o número de escalas, geradas usando *Average Pooling* ou *Bilinear Interpolation* na entrada). Isso garante que anomalias minúsculas (arranhões) e estruturais (peças faltando) sejam detectadas.

### 1.2. Matemática do Squeeze-and-Excitation (SE)
Dentro das camadas da CNN (especificamente sobre as saídas da AlexNet), o modelo aplica blocos de atenção SE. Seja um tensor de características $\mathbf{U} \in \mathbb{R}^{C \times H \times W}$ (onde $C$ é o número de canais, $H$ altura, $W$ largura), o bloco SE opera em 3 passos:

**Passo A: Squeeze (Compressão Espacial)**
Comprime-se o tensor espacial em um vetor descritor de canais $\mathbf{z}_{se} \in \mathbb{R}^{C}$ através de um *Global Average Pooling* (GAP):
$$ z_{se, c} = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} U_{c}(i, j) $$

**Passo B: Excitation (Excitação e Descoberta de Padrões)**
O vetor $\mathbf{z}_{se}$ passa por um perceptron multicamadas (MLP) com gargalo de dimensionalidade (fator de redução $r$). A saída é o vetor de pesos de atenção $\mathbf{s} \in \mathbb{R}^{C}$:
$$ \mathbf{s} = \sigma( \mathbf{W}_2 \cdot \delta( \mathbf{W}_1 \cdot \mathbf{z}_{se} ) ) $$
Onde:
- $\delta$ é a função de ativação ReLU.
- $\sigma$ é a função Sigmoid (garante que os pesos fiquem entre 0 e 1).
- $\mathbf{W}_1 \in \mathbb{R}^{\frac{C}{r} \times C}$ reduz a dimensionalidade.
- $\mathbf{W}_2 \in \mathbb{R}^{C \times \frac{C}{r}}$ restaura a dimensionalidade.

**Passo C: Scale (Re-escalonamento)**
O tensor original é multiplicado pelos canais de atenção, dando mais peso às características relevantes para a anomalia:
$$ \tilde{\mathbf{U}}_{c} = s_c \cdot \mathbf{U}_c $$

A saída final da fase de extração é o vetor unidimensional $\mathbf{y} \in \mathbb{R}^{D}$, obtido através do *Global Average Pooling* final dos tensores $\tilde{\mathbf{U}}$ concatenados de todas as escalas.

---

## 2. O Coração Estatístico: Normalizing Flows

O vetor de características extraído, $\mathbf{y}$, possui uma distribuição de probabilidade do mundo real $p_Y(\mathbf{y})$ que é matematicamente intratável (não sabemos a fórmula dela).

O objetivo do **Normalizing Flow** é aplicar uma função invertível (bijetora) $f_\theta$ parametrizada por uma rede neural (cujos pesos são $\theta$) que mapeia o vetor complexo $\mathbf{y}$ para uma variável latente simples $\mathbf{z}$:
$$ \mathbf{z} = f_\theta(\mathbf{y}) $$
Neste espaço latente, impomos que a distribuição seja Normal Padrão: $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$.

### 2.1. O Teorema da Mudança de Variáveis
A regra fundamental do cálculo probabilístico para transformações de variáveis garante que a densidade de $\mathbf{y}$ pode ser recuperada por:
$$ p_Y(\mathbf{y}) = p_Z(f_\theta(\mathbf{y})) \cdot \left| \det \left( \frac{\partial f_\theta(\mathbf{y})}{\partial \mathbf{y}} \right) \right| $$
Onde:
- $p_Z(\mathbf{z}) = \frac{1}{(2\pi)^{D/2}} \exp \left( -\frac{1}{2} \|\mathbf{z}\|^2 \right)$
- $\mathbf{J} = \frac{\partial f_\theta(\mathbf{y})}{\partial \mathbf{y}}$ é a Matriz Jacobiana da transformação.

### 2.2. O Segredo das "Affine Coupling Layers" (GLOW / Real-NVP)
Para que a rede seja treinável, calcular o determinante da Jacobiana precisa ser rápido (O($D$) e não O($D^3$)). Para isso, o SEDifferNet usa camadas de acoplamento afim.

O vetor $\mathbf{y}$ é dividido em duas metades: $\mathbf{y} = [\mathbf{y}_1, \mathbf{y}_2]$.
A transformação ocorre apenas na segunda metade, condicionada à primeira:
$$ \mathbf{z}_1 = \mathbf{y}_1 $$
$$ \mathbf{z}_2 = \mathbf{y}_2 \odot \exp(s(\mathbf{y}_1)) + t(\mathbf{y}_1) $$
*(onde $\odot$ é a multiplicação elemento por elemento, e $s, t$ são sub-redes neurais comuns)*.

Essa jogada genial transforma a Matriz Jacobiana $\mathbf{J}$ em uma matriz **triangular inferior**:
$$
\mathbf{J} = 
\begin{bmatrix}
\mathbf{I} & \mathbf{0} \\
\frac{\partial \mathbf{z}_2}{\partial \mathbf{y}_1} & \text{diag}(\exp(s(\mathbf{y}_1)))
\end{bmatrix}
$$
Como o determinante de uma matriz triangular é apenas o produto de sua diagonal principal, temos uma fórmula extremamente eficiente e analítica:
$$ \det(\mathbf{J}) = \prod \exp(s(\mathbf{y}_1)) = \exp \left( \sum s(\mathbf{y}_1) \right) $$

---

## 3. A Função de Perda (Treinamento)

Como dispomos apenas de imagens sem defeitos no treinamento, treinamos a rede por estimativa de Máxima Verossimilhança (*Maximum Likelihood Estimation - MLE*). Queremos maximizar $p_Y(\mathbf{y})$ para dados normais.

Para facilitar as derivadas, maximizamos o **logaritmo da probabilidade**:
$$ \log p_Y(\mathbf{y}) = \log p_Z(\mathbf{z}) + \log \left| \det(\mathbf{J}) \right| $$

Substituindo a fórmula do log da Gaussiana $\log p_Z(\mathbf{z})$:
$$ \log p_Y(\mathbf{y}) = -\frac{1}{2} \|\mathbf{z}\|^2 - \frac{D}{2} \log(2\pi) + \log \left| \det(\mathbf{J}) \right| $$

Ignorando as constantes aditivas multiplicativas, a função de perda (*Negative Log-Likelihood - NLL*) a ser minimizada via Gradiente Descendente é:
> [!IMPORTANT] 
> **Função de Perda Final do SEDifferNet:**
> $$ \mathcal{L}(\theta) = \frac{1}{2} \|\mathbf{z}\|^2 - \sum_{i} \log |J_{ii}| $$

Durante as épocas do treinamento, a rede ajusta seus parâmetros $\theta$ (tanto na CNN via *backpropagation* guiada pelo Flow, quanto nas camadas $s$ e $t$ do Flow) de forma que vetores normais $\mathbf{y}$ sejam levados para o marco zero do espaço latente ($\mathbf{z} \approx \mathbf{0}$).

---

## 4. Fase de Inferência (Score de Anomalia e Detecção)

Em produção, apresentamos uma imagem totalmente nova $\mathbf{x}_{new}$ para a rede.

1. A CNN + SE Attention processa a imagem gerando $\mathbf{y}_{new}$.
2. O Flow mapeia para o espaço de distribuição latente: $\mathbf{z}_{new} = f_\theta(\mathbf{y}_{new})$.
3. O Score de Anomalia $A(\mathbf{x})$ baseia-se unicamente na probabilidade negativa em relação à distribuição normal. Como a densidade de uma normal é controlada pela sua distância à origem (A norma euclidiana / L2), obtemos:

> [!TIP]
> **Cálculo do Anomaly Score:**
> $$ A(\mathbf{x}_{new}) = \|\mathbf{z}_{new}\|^2 = \sum_{d=1}^{D} z_{d}^2 $$

- Se a imagem for **Normal**, os padrões extraídos foram vistos durante o treinamento e mapeados com sucesso. O vetor cai no "alvo", resultando em valores próximos de $0$ (Ex: $\|\mathbf{z}\|^2 = 0.5$).
- Se a imagem for **Anômala**, a extração de características gera padrões alienígenas para a rede $f_\theta$, que reage atirando o vetor para os rincões remotos do espaço multidimensional (Ex: $\|\mathbf{z}\|^2 = 1800.0$). Um simples threshold separa as duas categorias com maestria.
