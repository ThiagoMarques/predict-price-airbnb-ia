# Seleção e Otimização de Modelos

Este diretório contém scripts para seleção, avaliação e otimização de modelos de Machine Learning seguindo o fluxo do livro.

## Fluxo de Trabalho

### 1. Avaliação Inicial (`01_avaliacao_inicial_modelos.py`)
- Começa com **LinearRegression** do sklearn
- Avalia no conjunto de treinamento primeiro
- Verifica se o modelo consegue aprender padrões básicos
- Avalia no conjunto de teste para ver generalização

**Executar:**
```bash
python models/01_avaliacao_inicial_modelos.py
```

### 2. Validação Cruzada (`02_validacao_cruzada.py`)
- Usa **cross-validation** para melhor estimativa de performance
- Compara LinearRegression vs RandomForest
- Usa KFold com 5 folds

**Executar:**
```bash
python models/02_validacao_cruzada.py
```

### 3. Lista de Modelos Promissores (`03_lista_modelos_promissores.py`)
- Testa vários modelos:
  - LinearRegression
  - Ridge, Lasso, ElasticNet
  - DecisionTree
  - RandomForest
  - GradientBoosting
- Ranking dos modelos por performance
- Identifica top 3 modelos promissores

**Executar:**
```bash
python models/03_lista_modelos_promissores.py
```

### 4. Otimização de Hiperparâmetros (`04_gridsearch_randomized.py`)
- **GridSearchCV**: Busca exaustiva em grade de hiperparâmetros
- **RandomizedSearchCV**: Busca aleatória (mais eficiente)
- Otimiza os melhores modelos identificados
- Salva modelo final otimizado

**Executar:**
```bash
python models/04_gridsearch_randomized.py
```

### 5. Preparação para Hugging Face (`05_preparar_huggingface.py`)
- Prepara modelo para envio ao Hugging Face Model Hub
- Cria README.md e MODEL_CARD.md
- Organiza arquivos necessários
- Gera instruções para upload

**Executar:**
```bash
python models/05_preparar_huggingface.py
```

### 6. Login no Hugging Face (`06_login_huggingface.py`)
- Faz login no Hugging Face Hub
- Necessário antes de enviar modelos

**Executar:**
```bash
python models/06_login_huggingface.py
```

### 7. Enviar para Hugging Face (`07_enviar_huggingface.py`)
- Envia modelo preparado para o Hugging Face Model Hub

**Executar:**
```bash
python models/07_enviar_huggingface.py
```

## Estrutura de Arquivos

```
models/
├── 01_avaliacao_inicial_modelos.py
├── 02_validacao_cruzada.py
├── 03_lista_modelos_promissores.py
├── 04_gridsearch_randomized.py
├── 05_preparar_huggingface.py
├── 06_login_huggingface.py
├── 07_enviar_huggingface.py
├── utils_models.py           # Funções utilitárias
├── saved/                    # Modelos salvos
│   ├── preprocessor.pkl
│   ├── linear_regression.pkl
│   ├── randomforest_best.pkl
│   └── gradientboosting_best.pkl
└── huggingface/              # Arquivos para Hugging Face
    ├── preprocessor.pkl
    ├── modelo_best.pkl
    ├── README.md
    ├── MODEL_CARD.md
    └── requirements.txt
```

## Ordem de Execução Recomendada

1. Execute os scripts na ordem numérica (01 → 05)
2. Cada script prepara dados para o próximo
3. O script 04 salva os modelos otimizados
4. O script 05 prepara tudo para Hugging Face
5. O script 06 faz login no Hugging Face
6. O script 07 envia o modelo para o Hugging Face

## Otimização: Dados Transformados

Os scripts de modelos (`01` a `04`) verificam automaticamente se existem dados transformados salvos em:
- `data/dados_treino_transformados.csv`
- `data/dados_teste_transformados.csv`

**Se existirem**, os scripts usam esses dados diretamente, evitando reprocessamento e acelerando significativamente a execução.

**Para gerar os dados transformados:**
```bash
python aplicar_transformadores.py
# ou
python src/06_aplicar_transformadores.py
```

**Vantagens:**
- ⚡ Execução muito mais rápida
- ✅ Garante que todos os modelos usam exatamente os mesmos dados
- ✅ Evita reprocessamento desnecessário
- ✅ Facilita comparação justa entre modelos

## Login no Hugging Face

Antes de enviar modelos, você precisa fazer login:

```bash
# Opção 1: Script interativo
python3 models/06_login_huggingface.py

# Opção 2: Com token direto
python3 models/06_login_huggingface.py --token SEU_TOKEN_AQUI

# Opção 3: Via Python
python3 -c "from huggingface_hub import login; login()"
```

**Obter token:**
1. Acesse: https://huggingface.co/settings/tokens
2. Crie um token com permissões de "write"
3. Use o token no login

## Enviar para Hugging Face

Após executar o script 05 e fazer login:

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path='models/huggingface',
    repo_id='ThiagoMarques/predict-price-airbnb-ia',
    repo_type='model'
)
```

Ou use o script:

```bash
python3 models/07_enviar_huggingface.py
```

## Dependências Adicionais

Para os scripts de otimização, você pode precisar:

```bash
pip install scipy  # Para RandomizedSearchCV com distribuições
pip install huggingface_hub  # Para upload ao Hugging Face
```

