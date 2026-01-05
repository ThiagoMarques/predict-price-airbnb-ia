# Previsão de Preços de Hospedagens Airbnb no Rio de Janeiro

## Projeto

Este projeto tem como objetivo projetar preços de hospedagens para estratégias futuras, seja pessoal para planejar férias ou negocial para calcular rentabilidade nos próximos meses.

Atualmente, a busca por preços pode ser uma tarefa repetitiva e cansativa, visto que feriados e outras datas comemorativas alteram significativamente os preços de hospedagem.

A abordagem utilizada para treinamento futuro do modelo será a de **aprendizado supervisionado**, visto que temos o preço médio dos aluguéis no Rio de Janeiro. Mas antes vamos preparar os dados.

## Dados

O conjunto de dados utilizado é proveniente do projeto [Inside Airbnb](https://insideairbnb.com/rio-de-janeiro/), uma iniciativa de dados abertos que coleta e disponibiliza informações públicas sobre anúncios da plataforma Airbnb.

O arquivo analisado corresponde ao `listings.csv`, contendo dados detalhados de anúncios ativos em uma determinada cidade ou região no momento da coleta.

O dataset reúne informações:
- **Geográficas**: latitude e longitude
- **Econômicas**: preço do aluguel, receita e ocupação estimadas
- **Características do imóvel**: tipo de propriedade, tipo de acomodação, capacidade, número de quartos e banheiros
- **Disponibilidade**: dias disponíveis no ano
- **Indicadores de desempenho**: avaliações, número de reviews e métricas de reputação do anfitrião

## Instalação

### Pré-requisitos

- Python 3.8 ou superior
- pip

### Instalação das dependências

```bash
pip install -r requirements.txt
```

## Uso

### 0. Visualização de Mapas

```bash
python src/00_visualizacao_mapas.py
```

Gera mapas geográficos:
- Distribuição geográfica dos imóveis
- Distribuição de preços por localização

### 1. Visualização dos Dados

```bash
python src/01_visualizacao_dados.py
```

Gera histogramas de:
- Preços
- Longitude
- Latitude
- Capacidade de acomodação
- Média de preços por bairro

### 2. Amostragem Estratificada

```bash
python src/02_amostragem_estratificada.py
```

Cria uma amostra estratificada de 20% dos dados, mantendo a proporção de estratos de preço.

### 3. Análise de Correlação

```bash
python src/03_analise_correlacao.py
```

Calcula o coeficiente de correlação de Pearson entre características e preço.

### 4. Processamento de Texto

```bash
python src/04_processamento_texto.py
```

Analisa descrições dos imóveis usando TF-IDF para identificar palavras-chave relacionadas ao preço.

### 5. Análise de Escalonamento

```bash
python src/05_analise_escalonamento.py
```

Identifica quais características necessitam de escalonamento.

### 6. Aplicar Transformadores

```bash
python src/06_aplicar_transformadores.py
```

Aplica os transformadores obrigatórios:
- SimpleImputer (valores faltantes)
- StandardScaler (escalonamento)
- OneHotEncoder (dados categóricos)

### 7. Treinar Modelo de ML

```bash
python src/07_modelo_ml.py
```

Treina modelos de Machine Learning (Regressão Linear e Random Forest) usando a pipeline completa.

## Transformadores Aplicados

### Obrigatórios

1. **SimpleImputer**
   - Numéricos: estratégia `'median'`
   - Categóricos: estratégia `'most_frequent'`

2. **StandardScaler**
   - Normaliza dados numéricos para média=0, desvio=1

3. **OneHotEncoder**
   - Converte categorias em colunas binárias

### Pipeline Completa

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), caracteristicas_numericas),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
        ]), caracteristicas_categoricas)
    ]
)
```

## Resultados Principais

### Correlações com Preço

As correlações encontradas foram todas fracas (r < 0.15), indicando que:
- O preço não é explicado de forma linear por um único atributo
- O mercado de Airbnb é multifatorial
- Relações não lineares e efeitos espaciais desempenham papel relevante

**Top 3 correlações:**
1. `bathrooms`: r = 0.1096
2. `bedrooms`: r = 0.1056
3. `accommodates`: r = 0.0875

### Palavras-Chave que Influenciam Preço

**Aumentam preço:**
- `pool`, `bedrooms`, `large`, `area`, `sea`, `house`, `best`

**Diminuem preço:**
- `subway`, `close`, `restaurants`, `conditioning`, `bed`, `station`, `air`, `studio`


**Nota**: Este projeto faz parte de uma série de análises sobre o mercado de hospedagens. O próximo artigo abordará a escolha e treino do modelo de aprendizado de máquina.
