"""
Lista de Modelos Promissores
Testa vários modelos para identificar os melhores candidatos
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer, mean_squared_error
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))
from utils import clean_price
from utils_models import carregar_dados_transformados

def criar_pipeline_preprocessamento(caracteristicas_numericas, caracteristicas_categoricas):
    """Cria pipeline de pré-processamento"""
    return ColumnTransformer(
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

def avaliar_modelo_cv(modelo, X, y, cv=3, nome_modelo=""):
    """Avalia modelo usando validação cruzada rápida (3 folds)"""
    kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
    rmse_scorer = make_scorer(
        lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
        greater_is_better=False
    )
    
    scores_rmse = -cross_val_score(modelo, X, y, cv=kfold, scoring=rmse_scorer, n_jobs=-1)
    scores_r2 = cross_val_score(modelo, X, y, cv=kfold, scoring='r2', n_jobs=-1)
    
    return {
        'nome': nome_modelo,
        'rmse_mean': scores_rmse.mean(),
        'rmse_std': scores_rmse.std(),
        'r2_mean': scores_r2.mean(),
        'r2_std': scores_r2.std()
    }

def main():
    print("=" * 70)
    print("AVALIAÇÃO DE MODELOS PROMISSORES")
    print("=" * 70)
    print("Testando vários modelos para identificar os melhores candidatos")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Tentar carregar dados transformados
    dados_transformados = carregar_dados_transformados(project_root)
    
    if dados_transformados is not None:
        X_treino_transformado, y_treino, _, _ = dados_transformados
        print("\n✓ Usando dados transformados salvos (sem reprocessamento)")
    else:
        print("⚠️  Dados transformados não encontrados. Processando dados originais...")
        data_path = os.path.join(project_root, 'data', 'listings.csv')
        df = pd.read_csv(data_path, low_memory=False)
        df['price'] = df['price'].apply(clean_price)
        
        caracteristicas_numericas = [
            'latitude', 'longitude', 'accommodates', 'bedrooms',
            'bathrooms', 'number_of_reviews', 'review_scores_rating',
            'availability_365', 'minimum_nights', 'maximum_nights'
        ]
        
        caracteristicas_categoricas = [
            'property_type', 'room_type', 'neighbourhood_cleansed'
        ]
        
        target = 'price'
        
        df_selecionado = df[caracteristicas_numericas + caracteristicas_categoricas + [target]].copy()
        df_completo = df_selecionado.dropna(subset=[target])
        
        X = df_completo[caracteristicas_numericas + caracteristicas_categoricas]
        y = df_completo[target]
        
        X_treino, X_teste, y_treino, y_teste = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        preprocessor = criar_pipeline_preprocessamento(caracteristicas_numericas, caracteristicas_categoricas)
        X_treino_transformado = preprocessor.fit_transform(X_treino)
    
    print(f"\nDados: {X_treino_transformado.shape[0]} amostras, {X_treino_transformado.shape[1]} features")
    print("Usando validação cruzada com 3 folds para avaliação rápida\n")
    
    # Lista de modelos promissores
    modelos = [
        LinearRegression(),
        Ridge(alpha=1.0),
        Lasso(alpha=0.1),
        ElasticNet(alpha=0.1, l1_ratio=0.5),
        DecisionTreeRegressor(max_depth=10, random_state=42),
        RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    ]
    
    nomes = [
        "LinearRegression",
        "Ridge",
        "Lasso",
        "ElasticNet",
        "DecisionTree",
        "RandomForest",
        "GradientBoosting"
    ]
    
    resultados = []
    
    for modelo, nome in zip(modelos, nomes):
        print(f"Avaliando {nome}...")
        metricas = avaliar_modelo_cv(modelo, X_treino_transformado, y_treino, cv=3, nome_modelo=nome)
        resultados.append(metricas)
    
    # Ordenar por R²
    resultados_ordenados = sorted(resultados, key=lambda x: x['r2_mean'], reverse=True)
    
    print("\n" + "=" * 70)
    print("RANKING DE MODELOS (ordenado por R²)")
    print("=" * 70)
    print(f"{'Pos':<5} {'Modelo':<25} {'RMSE':<15} {'R²':<15}")
    print("-" * 70)
    
    for i, res in enumerate(resultados_ordenados, 1):
        print(f"{i:<5} {res['nome']:<25} {res['rmse_mean']:<15.2f} {res['r2_mean']:<15.4f}")
    
    print("\n" + "=" * 70)
    print("TOP 3 MODELOS PROMISSORES")
    print("=" * 70)
    for i, res in enumerate(resultados_ordenados[:3], 1):
        print(f"{i}. {res['nome']}")
        print(f"   RMSE: {res['rmse_mean']:.2f} (+/- {res['rmse_std']:.2f})")
        print(f"   R²: {res['r2_mean']:.4f} (+/- {res['r2_std']:.4f})")
    
    print("\n" + "=" * 70)
    print("PRÓXIMOS PASSOS")
    print("=" * 70)
    print("1. Focar nos top 3 modelos para otimização")
    print("2. Usar GridSearchCV para busca exaustiva de hiperparâmetros")
    print("3. Usar RandomizedSearchCV para busca mais eficiente em espaços grandes")

if __name__ == "__main__":
    main()

