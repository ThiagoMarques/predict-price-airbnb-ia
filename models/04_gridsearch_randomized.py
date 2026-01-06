"""
GridSearchCV e RandomizedSearchCV
Otimização de hiperparâmetros dos melhores modelos
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
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

def grid_search_elastic_net(X, y):
    """GridSearchCV para ElasticNet (melhor modelo inicial)"""
    print("\n" + "=" * 70)
    print("GRIDSEARCHCV - ElasticNet")
    print("=" * 70)
    print("Busca exaustiva em grade de hiperparâmetros")
    
    param_grid = {
        'alpha': [0.01, 0.1, 0.5, 1.0, 2.0],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
    }
    
    modelo = ElasticNet(random_state=42, max_iter=2000)
    grid_search = GridSearchCV(
        modelo, param_grid, cv=3, scoring='neg_mean_squared_error',
        n_jobs=-1, verbose=1
    )
    
    print("Executando GridSearchCV...")
    grid_search.fit(X, y)
    
    print(f"\nMelhores parâmetros: {grid_search.best_params_}")
    print(f"Melhor score (RMSE): {np.sqrt(-grid_search.best_score_):.2f}")
    
    return grid_search.best_estimator_, grid_search.best_params_

def grid_search_random_forest(X, y):
    """GridSearchCV para RandomForest"""
    print("\n" + "=" * 70)
    print("GRIDSEARCHCV - RandomForestRegressor")
    print("=" * 70)
    print("Busca exaustiva em grade de hiperparâmetros")
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2, 4]
    }
    
    modelo = RandomForestRegressor(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        modelo, param_grid, cv=3, scoring='neg_mean_squared_error',
        n_jobs=-1, verbose=1
    )
    
    print("Executando GridSearchCV (pode demorar alguns minutos)...")
    grid_search.fit(X, y)
    
    print(f"\nMelhores parâmetros: {grid_search.best_params_}")
    print(f"Melhor score (RMSE): {np.sqrt(-grid_search.best_score_):.2f}")
    
    return grid_search.best_estimator_, grid_search.best_params_

def randomized_search_gradient_boosting(X, y):
    """RandomizedSearchCV para GradientBoosting"""
    print("\n" + "=" * 70)
    print("RANDOMIZEDSEARCHCV - GradientBoostingRegressor")
    print("=" * 70)
    print("Busca aleatória em espaço de hiperparâmetros (mais eficiente)")
    
    from scipy.stats import randint, uniform
    
    param_distributions = {
        'n_estimators': randint(50, 300),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.2),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10)
    }
    
    modelo = GradientBoostingRegressor(random_state=42)
    random_search = RandomizedSearchCV(
        modelo, param_distributions, n_iter=20, cv=3,
        scoring='neg_mean_squared_error', n_jobs=-1, verbose=1, random_state=42
    )
    
    print("Executando RandomizedSearchCV (20 iterações)...")
    random_search.fit(X, y)
    
    print(f"\nMelhores parâmetros: {random_search.best_params_}")
    print(f"Melhor score (RMSE): {np.sqrt(-random_search.best_score_):.2f}")
    
    return random_search.best_estimator_, random_search.best_params_

def avaliar_modelo_final(modelo, X_treino, y_treino, X_teste, y_teste, nome_modelo):
    """Avalia modelo final no conjunto de teste"""
    print(f"\n" + "=" * 70)
    print(f"AVALIAÇÃO FINAL - {nome_modelo}")
    print("=" * 70)
    
    y_pred_treino = modelo.predict(X_treino)
    y_pred_teste = modelo.predict(X_teste)
    
    print("\nConjunto de TREINO:")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_treino, y_pred_treino)):.2f}")
    print(f"  MAE: R$ {mean_absolute_error(y_treino, y_pred_treino):.2f}")
    print(f"  R²: {r2_score(y_treino, y_pred_treino):.4f}")
    
    print("\nConjunto de TESTE:")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_teste, y_pred_teste)):.2f}")
    print(f"  MAE: R$ {mean_absolute_error(y_teste, y_pred_teste):.2f}")
    print(f"  R²: {r2_score(y_teste, y_pred_teste):.4f}")
    
    return {
        'rmse_teste': np.sqrt(mean_squared_error(y_teste, y_pred_teste)),
        'r2_teste': r2_score(y_teste, y_pred_teste)
    }

def main():
    print("=" * 70)
    print("OTIMIZAÇÃO DE HIPERPARÂMETROS")
    print("=" * 70)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Tentar carregar dados transformados
    dados_transformados = carregar_dados_transformados(project_root)
    
    if dados_transformados is not None:
        X_treino_transformado, y_treino, X_teste_transformado, y_teste = dados_transformados
        preprocessor = None  # Não precisamos do preprocessor se dados já estão transformados
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
        X_teste_transformado = preprocessor.transform(X_teste)
    
    # GridSearchCV para ElasticNet (melhor modelo inicial)
    modelo_en, params_en = grid_search_elastic_net(X_treino_transformado, y_treino)
    metricas_en = avaliar_modelo_final(modelo_en, X_treino_transformado, y_treino,
                                       X_teste_transformado, y_teste, "ElasticNet (Otimizado)")
    
    # GridSearchCV para RandomForest
    modelo_rf, params_rf = grid_search_random_forest(X_treino_transformado, y_treino)
    metricas_rf = avaliar_modelo_final(modelo_rf, X_treino_transformado, y_treino,
                                       X_teste_transformado, y_teste, "RandomForest (Otimizado)")
    
    # RandomizedSearchCV para GradientBoosting
    modelo_gb, params_gb = randomized_search_gradient_boosting(X_treino_transformado, y_treino)
    metricas_gb = avaliar_modelo_final(modelo_gb, X_treino_transformado, y_treino,
                                       X_teste_transformado, y_teste, "GradientBoosting (Otimizado)")
    
    # Comparação final
    print("\n" + "=" * 70)
    print("COMPARAÇÃO FINAL DOS MODELOS OTIMIZADOS")
    print("=" * 70)
    print(f"{'Modelo':<30} {'RMSE (Teste)':<15} {'R² (Teste)':<15}")
    print("-" * 70)
    print(f"{'ElasticNet (Otimizado)':<30} {metricas_en['rmse_teste']:<15.2f} {metricas_en['r2_teste']:<15.4f}")
    print(f"{'RandomForest (Otimizado)':<30} {metricas_rf['rmse_teste']:<15.2f} {metricas_rf['r2_teste']:<15.4f}")
    print(f"{'GradientBoosting (Otimizado)':<30} {metricas_gb['rmse_teste']:<15.2f} {metricas_gb['r2_teste']:<15.4f}")
    
    # Encontrar melhor modelo
    modelos_comparacao = [
        (modelo_en, "ElasticNet", metricas_en),
        (modelo_rf, "RandomForest", metricas_rf),
        (modelo_gb, "GradientBoosting", metricas_gb)
    ]
    melhor_modelo, melhor_nome, melhor_metricas = max(modelos_comparacao, key=lambda x: x[2]['r2_teste'])
    
    print(f"\n✓ Melhor modelo: {melhor_nome} (R²: {melhor_metricas['r2_teste']:.4f})")
    
    # Salvar modelo
    saved_dir = os.path.join(project_root, 'models', 'saved')
    os.makedirs(saved_dir, exist_ok=True)
    joblib.dump(melhor_modelo, os.path.join(saved_dir, f'{melhor_nome.lower()}_best.pkl'))
    if preprocessor is not None:
        joblib.dump(preprocessor, os.path.join(saved_dir, 'preprocessor.pkl'))
    print(f"\n✓ Modelo salvo em: models/saved/{melhor_nome.lower()}_best.pkl")

if __name__ == "__main__":
    main()

