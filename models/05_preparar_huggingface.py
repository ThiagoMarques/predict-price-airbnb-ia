"""
Preparação para Hugging Face
Prepara o modelo para ser enviado ao Hugging Face Model Hub
"""
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))
from utils import clean_price
from utils_models import carregar_dados_transformados

def criar_model_card(modelo_info):
    """Cria model card para Hugging Face"""
    model_card = f"""---
license: mit
tags:
- machine-learning
- regression
- price-prediction
- airbnb
- brazil
- rio-de-janeiro
datasets:
- inside-airbnb
metrics:
- rmse
- mae
- r2
---

# {modelo_info['nome']} - Previsão de Preços Airbnb Rio de Janeiro

## Descrição

Modelo de Machine Learning para previsão de preços de hospedagens Airbnb no Rio de Janeiro.

## Métricas de Performance

- **RMSE**: {modelo_info['rmse']:.2f}
- **MAE**: R$ {modelo_info['mae']:.2f}
- **R²**: {modelo_info['r2']:.4f}

## Características Utilizadas

### Numéricas:
{chr(10).join(f"- {feat}" for feat in modelo_info['features_numericas'])}

### Categóricas:
{chr(10).join(f"- {feat}" for feat in modelo_info['features_categoricas'])}

## Como Usar

```python
import joblib
import pandas as pd

# Carregar modelo e preprocessor
preprocessor = joblib.load('preprocessor.pkl')
modelo = joblib.load('{modelo_info["nome"].lower()}_best.pkl')

# Preparar dados
X = pd.DataFrame({{
    # ... seus dados aqui
}})

# Prever
X_transformado = preprocessor.transform(X)
preco_previsto = modelo.predict(X_transformado)
```

## Dataset

Dados do [Inside Airbnb - Rio de Janeiro](https://insideairbnb.com/rio-de-janeiro/)

## Treinamento

- **Algoritmo**: {modelo_info['algoritmo']}
- **Hiperparâmetros**: {json.dumps(modelo_info['hiperparametros'], indent=2)}
- **Validação**: Cross-validation com 3 folds

## Limitações

- Modelo treinado apenas com dados do Rio de Janeiro
- Performance pode variar para outras cidades
- Preços em R$ (Reais brasileiros)

## Autor

Thiago Marques

## Licença

MIT
"""
    return model_card

def criar_readme_huggingface(modelo_info):
    """Cria README específico para Hugging Face"""
    readme = f"""# {modelo_info['nome']}

Modelo de Machine Learning para previsão de preços de hospedagens Airbnb no Rio de Janeiro.

## Performance

- RMSE: {modelo_info['rmse']:.2f}
- MAE: R$ {modelo_info['mae']:.2f}
- R²: {modelo_info['r2']:.4f}

## Uso

```python
import joblib

preprocessor = joblib.load('preprocessor.pkl')
modelo = joblib.load('{modelo_info["nome"].lower()}_best.pkl')

# Prever preço
X_transformado = preprocessor.transform(X)
preco = modelo.predict(X_transformado)
```

## Dataset

Inside Airbnb - Rio de Janeiro
"""
    return readme

def preparar_para_huggingface():
    """Prepara todos os arquivos necessários para Hugging Face"""
    print("=" * 70)
    print("PREPARAÇÃO PARA HUGGING FACE")
    print("=" * 70)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    saved_dir = os.path.join(project_root, 'models', 'saved')
    
    # Carregar preprocessor (se existir)
    preprocessor_path = os.path.join(saved_dir, 'preprocessor.pkl')
    preprocessor = None
    if os.path.exists(preprocessor_path):
        preprocessor = joblib.load(preprocessor_path)
        print("✓ Preprocessor carregado")
    else:
        print("⚠️  Preprocessor não encontrado (pode não ser necessário se dados já estão transformados)")
    
    # Tentar carregar modelos
    modelos_disponiveis = []
    modelos_para_tentar = ['elasticnet_best.pkl', 'randomforest_best.pkl', 'gradientboosting_best.pkl']
    for modelo_file in modelos_para_tentar:
        path = os.path.join(saved_dir, modelo_file)
        if os.path.exists(path):
            modelos_disponiveis.append((modelo_file, joblib.load(path)))
    
    if not modelos_disponiveis:
        print("❌ Nenhum modelo encontrado. Execute primeiro o script 04_gridsearch_randomized.py")
        return
    
    # Carregar dados transformados para obter métricas reais
    dados_transformados = carregar_dados_transformados(project_root)
    metricas_reais = {}
    if dados_transformados is not None:
        X_treino, y_treino, X_teste, y_teste = dados_transformados
        print("\n✓ Calculando métricas reais usando dados transformados...")
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        for modelo_file, modelo in modelos_disponiveis:
            y_pred = modelo.predict(X_teste)
            metricas_reais[modelo_file] = {
                'rmse': np.sqrt(mean_squared_error(y_teste, y_pred)),
                'mae': mean_absolute_error(y_teste, y_pred),
                'r2': r2_score(y_teste, y_pred)
            }
    
    # Criar diretório para Hugging Face Space
    hf_dir = Path(os.path.join(project_root, 'models', 'huggingface'))
    hf_dir.mkdir(exist_ok=True)
    
    # Criar estrutura de diretórios para Space
    models_saved_dir = hf_dir / 'models' / 'saved'
    models_saved_dir.mkdir(parents=True, exist_ok=True)
    
    # Copiar arquivos
    import shutil
    if preprocessor is not None:
        shutil.copy(os.path.join(saved_dir, 'preprocessor.pkl'), models_saved_dir / 'preprocessor.pkl')
        print("✓ Preprocessor copiado")
    
    print(f"\nModelos disponíveis: {len(modelos_disponiveis)}")
    
    # Encontrar melhor modelo (prioridade: gradientboosting > randomforest > elasticnet)
    melhor_modelo_file = None
    melhor_modelo = None
    melhor_metricas = {'rmse': 0.0, 'mae': 0.0, 'r2': -999}
    
    for modelo_file, modelo in modelos_disponiveis:
        # Copiar modelo
        shutil.copy(os.path.join(saved_dir, modelo_file), models_saved_dir / modelo_file)
        
        # Obter métricas reais se disponíveis
        metricas = metricas_reais.get(modelo_file, {'rmse': 0.0, 'mae': 0.0, 'r2': 0.0})
        
        # Priorizar melhor modelo por R²
        if metricas['r2'] > melhor_metricas['r2']:
            melhor_metricas = metricas
            melhor_modelo_file = modelo_file
            melhor_modelo = modelo
        
        nome_modelo = modelo_file.replace('_best.pkl', '').title()
        print(f"✓ {nome_modelo} copiado (R²: {metricas['r2']:.4f})")
    
    if melhor_modelo is None:
        melhor_modelo_file = modelos_disponiveis[0][0]
        melhor_modelo = modelos_disponiveis[0][1]
        melhor_metricas = metricas_reais.get(melhor_modelo_file, {'rmse': 0.0, 'mae': 0.0, 'r2': 0.0})
    
    nome_modelo = melhor_modelo_file.replace('_best.pkl', '').title()
    
    # Criar informações do modelo
    modelo_info = {
        'nome': nome_modelo,
        'algoritmo': type(melhor_modelo).__name__,
        'rmse': melhor_metricas['rmse'],
        'mae': melhor_metricas['mae'],
        'r2': melhor_metricas['r2'],
        'features_numericas': [
            'latitude', 'longitude', 'accommodates', 'bedrooms',
            'bathrooms', 'number_of_reviews', 'review_scores_rating',
            'availability_365', 'minimum_nights', 'maximum_nights'
        ],
        'features_categoricas': [
            'property_type', 'room_type', 'neighbourhood_cleansed'
        ],
        'hiperparametros': melhor_modelo.get_params()
    }
    
    # Criar README para Space
    readme_space = f"""---
title: Previsão de Preços Airbnb - Rio de Janeiro
emoji: 🏠
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---

# 🏠 Estimativa de Preço de Imóveis no Airbnb - Rio de Janeiro

Este Space demonstra um pipeline completo de Machine Learning, desde o pré-processamento dos dados até a predição de preços usando {modelo_info['algoritmo']}.

## 📊 Performance do Modelo

- **RMSE**: {modelo_info['rmse']:.2f}
- **MAE**: R$ {modelo_info['mae']:.2f}
- **R²**: {modelo_info['r2']:.4f}

## ⚠️ Aviso Importante

A estimativa é apenas educacional e possui limitações devido à ausência de fatores como:
- Sazonalidade e eventos locais
- Dinâmica de mercado em tempo real
- Fatores não capturados pelo modelo (localização exata, vista, etc.)
- Oferta e demanda momentâneas

**Não substitui análise real de mercado.**

## 🔗 Links

- [Repositório GitHub](https://github.com/ThiagoMarques/predict-price-airbnb-ia)
- [Inside Airbnb - Rio de Janeiro](https://insideairbnb.com/rio-de-janeiro/)

## 📝 Dataset

Dados do [Inside Airbnb - Rio de Janeiro](https://insideairbnb.com/rio-de-janeiro/)
"""
    
    # Criar requirements.txt para Space (com Gradio)
    requirements = """scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
joblib>=1.3.0
gradio>=4.0.0
"""
    
    # Salvar arquivos
    (hf_dir / 'README.md').write_text(readme_space)
    (hf_dir / 'requirements.txt').write_text(requirements)
    
    # app.py já deve existir (criado separadamente)
    if not (hf_dir / 'app.py').exists():
        print("⚠️  app.py não encontrado. Criando template básico...")
        # O app.py será criado separadamente
    
    print(f"\n✓ Melhor modelo selecionado: {nome_modelo} (R²: {melhor_metricas['r2']:.4f})")
    
    print("\n" + "=" * 70)
    print("ARQUIVOS PRONTOS PARA HUGGING FACE SPACE")
    print("=" * 70)
    print(f"Diretório: {hf_dir.absolute()}")
    print("\nEstrutura criada:")
    print("  📁 models/huggingface/")
    print("    ├── app.py (interface Gradio)")
    print("    ├── README.md (configuração do Space)")
    print("    ├── requirements.txt")
    print("    └── 📁 models/saved/")
    print("        ├── preprocessor.pkl")
    for modelo_file, _ in modelos_disponiveis:
        print(f"        └── {modelo_file}")
    
    print("\n" + "=" * 70)
    print("PRÓXIMOS PASSOS PARA CRIAR O SPACE")
    print("=" * 70)
    print("1. Fazer login: python3 models/06_login_huggingface.py")
    print("2. Criar Space no Hugging Face:")
    print("   - Acesse: https://huggingface.co/new-space")
    print("   - SDK: Gradio ✅")
    print("   - Hardware: Free ✅")
    print("   - Visibilidade: Public ✅")
    print("3. Enviar arquivos:")
    print("   python3 models/07_enviar_huggingface.py --repo-id seu-usuario/nome-space")
    print("\nOu manualmente via Git:")
    print("   cd models/huggingface")
    print("   git init")
    print("   git add .")
    print("   git commit -m 'Initial commit'")
    print("   git push https://huggingface.co/spaces/seu-usuario/nome-space")

if __name__ == "__main__":
    preparar_para_huggingface()

