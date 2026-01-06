"""
Validação Cruzada
Avalia modelos usando cross-validation para melhor estimativa de performance
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
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

def avaliar_com_cv(modelo, X, y, cv=5, nome_modelo=""):
    """Avalia modelo usando validação cruzada"""
    kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    # RMSE scorer (negativo porque sklearn maximiza)
    rmse_scorer = make_scorer(
        lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
        greater_is_better=False
    )
    
    scores_rmse = -cross_val_score(modelo, X, y, cv=kfold, scoring=rmse_scorer)
    scores_r2 = cross_val_score(modelo, X, y, cv=kfold, scoring='r2')
    
    print(f"\n{nome_modelo} - Validação Cruzada ({cv} folds):")
    print(f"  RMSE: {scores_rmse.mean():.2f} (+/- {scores_rmse.std() * 2:.2f})")
    print(f"  R²: {scores_r2.mean():.4f} (+/- {scores_r2.std() * 2:.4f})")
    
    return {
        'rmse_mean': scores_rmse.mean(),
        'rmse_std': scores_rmse.std(),
        'r2_mean': scores_r2.mean(),
        'r2_std': scores_r2.std()
    }

def main():
    print("=" * 70)
    print("VALIDAÇÃO CRUZADA - Comparando Modelos")
    print("=" * 70)
    
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
    
    print("\n" + "=" * 70)
    print("PASSO 1: Validação Cruzada - LinearRegression")
    print("=" * 70)
    
    modelo_lr = LinearRegression()
    metricas_lr = avaliar_com_cv(modelo_lr, X_treino_transformado, y_treino, cv=5, 
                                 nome_modelo="LinearRegression")
    
    print("\n" + "=" * 70)
    print("PASSO 2: Validação Cruzada - RandomForestRegressor")
    print("=" * 70)
    print("RandomForest é mais complexo e pode capturar padrões não-lineares")
    
    modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    metricas_rf = avaliar_com_cv(modelo_rf, X_treino_transformado, y_treino, cv=5,
                                  nome_modelo="RandomForestRegressor")
    
    print("\n" + "=" * 70)
    print("COMPARAÇÃO DE MODELOS")
    print("=" * 70)
    print(f"{'Modelo':<25} {'RMSE Médio':<15} {'R² Médio':<15}")
    print("-" * 70)
    print(f"{'LinearRegression':<25} {metricas_lr['rmse_mean']:<15.2f} {metricas_lr['r2_mean']:<15.4f}")
    print(f"{'RandomForestRegressor':<25} {metricas_rf['rmse_mean']:<15.2f} {metricas_rf['r2_mean']:<15.4f}")
    
    melhor = "RandomForestRegressor" if metricas_rf['r2_mean'] > metricas_lr['r2_mean'] else "LinearRegression"
    print(f"\n✓ Melhor modelo (validação cruzada): {melhor}")
    
    print("\n" + "=" * 70)
    print("PRÓXIMOS PASSOS")
    print("=" * 70)
    print("1. Testar outros modelos promissores (XGBoost, GradientBoosting, etc.)")
    print("2. Usar GridSearchCV para otimizar hiperparâmetros")
    print("3. Usar RandomizedSearchCV para busca mais eficiente")

if __name__ == "__main__":
    main()

