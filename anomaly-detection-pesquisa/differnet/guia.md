# Guia de Scripts - DifferNet

Este guia documenta todos os scripts presentes na pasta `differnet`, suas funções, como utilizá-los e como eles se interligam no pipeline de pesquisa de detecção de anomalias (baseado na arquitetura Same Same But DifferNet, expandida para nível de pixel com CFlow/PatchCore e testes de interpretabilidade).

## 1. Configuração e Ponto de Entrada

### `config.py`
- **Para que serve:** É o arquivo central de configurações de todo o pipeline. Quase todos os outros scripts importam os parâmetros daqui. Não é um script executável via linha de comando.
- **O que configurar aqui:** Caminhos de datasets, nome da classe (ex: `lightning-rod-suspension`), hyperparâmetros de treinamento (epochs, batch_size, learning_rate), hyperparâmetros de arquitetura (n_scales, dimensão das features), e se os transformadores (rotação, cor) devem ser aplicados.

### `main.py`
- **Para que serve:** Script simples que atua como ponto de entrada para rodar o fluxo principal (geralmente treinar o modelo e em seguida testar as anomalias da classe configurada no config.py).
- **Como usar:** `python main.py`

## 2. Treinamento

### `train.py`
- **Para que serve:** Script principal para treinar o modelo DifferNet (ou variações como SEDifferNet e CBAMDifferNet) do zero. Este script foca na detecção de anomalias a **nível de imagem** usando Normalizing Flows. Extrai features usando uma CNN (AlexNet/ResNet) com ou sem módulos de atenção (SE/CBAM), e treina o fluxo normalizador para prever a densidade probabilística de amostras "boas". Salva os pesos e registra métricas no MLflow.
- **Como usar:** `python train.py` (Certifique-se de ajustar a classe e dataset no `config.py`).

### `train_backup.py`
- **Para que serve:** Uma versão estendida ou backup do `train.py`, contendo lógicas adicionais de treino, extração de heatmap SE por background subtraction, logs extras no MLflow, e fallback de gravação de arquivos LITE para evitar problemas de memória/espaço ao salvar.

### `pixel_train_from_pretrained.py`
- **Para que serve:** Script avançado projetado para treinar "cabeças" de avaliação em **nível de pixel** (spatial map). Utiliza extratores baseados em **CFlow** ou **PatchCore** anexados ao modelo DifferNet já pré-treinado. Utiliza os pesos salvos anteriormente para extrair features espacialmente e treinar estas cabeças para fazer a *segmentação* e *localização* precisa das anomalias na imagem.
- **Como usar:** `python pixel_train_from_pretrained.py --model_path <caminho_do_modelo.pt>`
- **Parâmetros Principais:**
  - `--model_path`: (Obrigatório) Caminho do checkpoint de pesos `.pt` gerado previamente pelo `train.py`.

## 3. Avaliação e Testes

### `evaluate.py`
- **Para que serve:** Realiza uma avaliação básica de inferência do modelo para os dados de teste e calcula a métrica de AUROC em nível de imagem.
- **Como usar:** `python evaluate.py`

### `evaluate_final.py`
- **Para que serve:** Pipeline de avaliação mais completo e avançado. Combina a inferência em nível de imagem gerada pelo fluxo normalizador original e as inferências de localização espacial de modelos como CFlow/PatchCore. Permite a fusão e calibração de scores, métricas avançadas (AUPRO, F1-Score) e salva matrizes de calor (heatmaps) no diretório de saída.

### `evaluate_metrics.py`
- **Para que serve:** Uma biblioteca central com as funções matemáticas de validação das métricas. Contém a implementação do cálculo espacial do AUPRO (Area Under Per-Region Overlap), ROAD (RemOve And Debias), Sanity Checks, e funções que gerenciam a extração visual da anomalia.

### `evaluate_sanity_check.py`
- **Para que serve:** Executa o Cascading Randomization (Sanity Check). Compara o mapa de calor de anomalia do modelo treinado contra um modelo recém-inicializado aleatoriamente medindo a correlação de Rank de Spearman. Útil para atestar que as marcações do modelo vêm de aprendizado real e não de ruídos ou bias da CNN.
- **Como usar:** `python evaluate_sanity_check.py --model_path <caminho.pt> --class_name <classe>`

### `test_model.py`
- **Para que serve:** Permite rodar especificamente um teste sobre um checkpoint salvo, retornando suas métricas de AUROC. Muito útil para validações manuais diretas via CLI.
- **Como usar:** `python test_model.py --model_path <caminho.pt> --dataset_path <caminho> --class_name <classe>`

### `test_single_model.py`
- **Para que serve:** Script de inferência otimizado focado em extrair o máximo de métricas e performance sistêmica. Diferente dos avaliadores básicos, ele gera métricas de Performance (Latency ms/img, uso de RAM, VRAM), otimiza o uso do cuDNN com TF32, e reporta várias métricas estatísticas detalhadas como PR-AUC, F1, Accuracy, Precision, Recall e PG2 (Presorted Good at 2%).
- **Como usar:** `python test_single_model.py --model_group cbamdiffernet --class_name transistor`

### `test_all_models.py`
- **Para que serve:** Varre recursivamente um diretório para rodar testes em múltiplos arquivos `.pt` sequencialmente. Ao final da bateria, salva um log `*.txt` sumarizando todos os scores AUROC obtidos. Ótimo para comparar epochs diferentes.
- **Como usar:** `python test_all_models.py --models_dir ./checkpoints/ --output_file resultados.txt`

## 4. Análise e Visualização

### `analyze_model.py`
- **Para que serve:** Executa uma inferência no conjunto de testes e gera gráficos KDE e histogramas comparando as distribuições de scores das imagens "Normais" e das imagens com "Anomalias". Ajuda a entender a precisão do modelo e determinar visualmente o limiar (threshold) de corte ótimo.
- **Como usar:** `python analyze_model.py --model_path <caminho.pt>`

### `visualize_metric_steps.py`
- **Para que serve:** Focado na explicabilidade (XAI). Aplica as técnicas de interpretabilidade visual Deletion, Insertion e ROAD (LeRF) utilizando a base de GradCAM. Salva imagens mostrando o passo a passo da modificação/descaracterização nas imagens sob a ótica do modelo, além de traçar o gráfico de decaimento do score (curva AUC).
- **Como usar:** `python visualize_metric_steps.py --model_path <caminho.pt> --index 0 --steps 5`

### `read_checkpoint_auroc.py`
- **Para que serve:** Lê o dicionário gravado internamente em um checkpoint `.pt` gerado nos treinos e cospe os valores históricos de evolução do AUROC que o script salvou, sem precisar carregar nenhum modelo na GPU. Útil quando não se usa o MLflow ou ele se perdeu.
- **Como usar:** `python read_checkpoint_auroc.py <caminho_do_checkpoint.pt>`

### `read_log.py`
- **Para que serve:** Um snippet simples que abre e exibe na tela o arquivo `output2.txt` tratando as codificações UTF-16 e UTF-8 para recuperar logs do terminal.

## 5. Utilitários e Infraestrutura

### `model.py`
- **Para que serve:** Script fundacional onde as redes neurais em PyTorch são declaradas. Contém o construtor `nf_head` (o fluxo normalizador) e as classes empacotadoras como `DifferNet` e `SEDifferNet` (Squeeze-and-Excitation).
- **Adicional:** Guarda a lógica de robustez para localização relativa dos pesos carregados via `load_weights`.

### `freia_funcs.py`
- **Para que serve:** Implementação das camadas reversíveis de Normalizing Flow (`ReversibleGraphNet`, `glow_coupling_layer`, `permute_layer`), extraídas nativamente do framework FrEIA.

### `localization.py`
- **Para que serve:** Contém os geradores e manipuladores de Attention Maps via GradCAM e lógicas para transformar os extratores dimensionais da CNN em máscaras de calor bidimensionais.

### `utils.py`
- **Para que serve:** Armazena utilitários globais: processadores de imagem, tensores, transformações rotacionais (color jitters/transforms) e a classe de Dataloader personalizada (`AlignedTestDataset`) que retorna de maneira correta as máscaras com suas imagens.

### `multi_transform_loader.py`
- **Para que serve:** Sobrescreve as lógicas do dataloader nativo para garantir que a rede receba a mesma imagem várias vezes com transformações randômicas no mesmo batch para generalizar melhor o Normalizing Flow.

### `export_mlflow.py`
- **Para que serve:** Simples utilitário que compacta a pasta local `mlruns` inteira para um arquivo ZIP nomeado com timestamp, facilitando arquivar logs do MLflow.
- **Como usar:** `python export_mlflow.py`

### `generate_dummy_gt.py`
- **Para que serve:** Script que gera máscaras em branco ou totalmente pretas artificialmente caso o dataset que você tente rodar não possua pastas `ground_truth`. Ajuda a bypassar os erros do dataloader em datasets que só servem para avaliar o nível de imagem.
