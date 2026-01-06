# Classificação de Imagens - Tipos de Cômodos

Projeto de classificação de imagens de interiores usando técnicas de Machine Learning.

## Dataset

- **Fonte**: MIT LabelMe Indoor Scene Dataset
- **URL**: http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar
- **Classes**: bedroom, kitchen, bathroom, livingroom

## Estrutura do Projeto

```
image_classification/
├── data/              # Dados brutos e processados
│   ├── raw/           # Dataset original baixado
│   └── processed/     # Dados pré-processados (.npz)
├── scripts/            # Scripts de processamento
│   ├── 01_download.py      # Download do dataset
│   ├── 02_preprocess.py    # Pré-processamento e feature extraction
│   └── 03_train.py         # Treinamento dos modelos
├── models/             # Modelos treinados
├── results/            # Resultados e métricas
└── datasets.py         # Função fetch_airbnb_mnist() estilo sklearn
```

## Pré-processamento

1. **Resize**: 64×64 ou 32×32 pixels
2. **Grayscale**: Conversão para escala de cinza
3. **Feature Extraction**:
   - **Baseline**: Flatten (vetorização simples)
   - **HOG**: Histogram of Oriented Gradients (recomendado)

## Modelos

- Logistic Regression
- Linear SVM
- Random Forest
- Gradient Boosting

## Uso

```python
from datasets import fetch_airbnb_mnist

# Carregar dados com features HOG
data = fetch_airbnb_mnist(version="hog")
X, y = data["data"], data["target"]
target_names = data["target_names"]

# Ou com features flatten (baseline)
data = fetch_airbnb_mnist(version="flatten")
```

## Formato dos Dados

Arquivo `.npz` contém:
- `X`: Features (n_samples, n_features)
- `y`: Labels (n_samples,)
- `target_names`: Nomes das classes

