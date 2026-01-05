"""
Modelo de Machine Learning
Treina modelos de regressão para prever preços de hospedagens
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from utils import clean_price

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

def avaliar_modelo(y_real, y_pred, nome_modelo):
    """Avalia modelo e retorna métricas"""
    mse = mean_squared_error(y_real, y_pred)
    mae = mean_absolute_error(y_real, y_pred)
    r2 = r2_score(y_real, y_pred)
    
    print(f"\n{nome_modelo}:")
    print(f"  MSE: {mse:.2f}")
    print(f"  MAE: R$ {mae:.2f}")
    print(f"  R²: {r2:.4f}")
    
    return {'mse': mse, 'mae': mae, 'r2': r2}

def plot_resultados(y_teste, y_pred, nome_modelo):
    """Plota gráficos de predições e resíduos"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].scatter(y_teste, y_pred, alpha=0.3, s=10)
    axes[0].plot([y_teste.min(), y_teste.max()], [y_teste.min(), y_teste.max()], 'r--', lw=2)
    axes[0].set_xlabel('Preço Real (R$)')
    axes[0].set_ylabel('Preço Previsto (R$)')
    axes[0].set_title(f'Predições vs Real - {nome_modelo}')
    axes[0].grid(True, alpha=0.3)
    
    residuos = y_teste - y_pred
    axes[1].scatter(y_pred, residuos, alpha=0.3, s=10)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Preço Previsto (R$)')
    axes[1].set_ylabel('Resíduos (Real - Previsto)')
    axes[1].set_title(f'Análise de Resíduos - {nome_modelo}')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/resultados_modelo_ml.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    df = pd.read_csv('data/listings.csv', low_memory=False)
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
    
    # Regressão Linear
    modelo_lr = LinearRegression()
    modelo_lr.fit(X_treino_transformado, y_treino)
    y_pred_lr = modelo_lr.predict(X_teste_transformado)
    metricas_lr = avaliar_modelo(y_teste, y_pred_lr, "Regressão Linear")
    
    # Random Forest
    modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    modelo_rf.fit(X_treino_transformado, y_treino)
    y_pred_rf = modelo_rf.predict(X_teste_transformado)
    metricas_rf = avaliar_modelo(y_teste, y_pred_rf, "Random Forest")
    
    melhor = "Random Forest" if metricas_rf['r2'] > metricas_lr['r2'] else "Regressão Linear"
    print(f"\nMelhor modelo: {melhor}")
    
    plot_resultados(y_teste, y_pred_rf, "Random Forest")
    print(f"\nGráficos salvos em: results/resultados_modelo_ml.png")

if __name__ == "__main__":
    main()
