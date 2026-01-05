"""
Aplicação de Transformadores Obrigatórios
Aplica SimpleImputer, StandardScaler e OneHotEncoder aos dados
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from utils import clean_price

def criar_pipeline(caracteristicas_numericas, caracteristicas_categoricas):
    """Cria pipeline completa de pré-processamento"""
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

def main():
    df = pd.read_csv('../data/listings.csv', low_memory=False)
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
    
    preprocessor = criar_pipeline(caracteristicas_numericas, caracteristicas_categoricas)
    
    X_treino_transformado = preprocessor.fit_transform(X_treino)
    X_teste_transformado = preprocessor.transform(X_teste)
    
    print(f"Shape após transformação:")
    print(f"  Treino: {X_treino_transformado.shape}")
    print(f"  Teste: {X_teste_transformado.shape}")
    
    # Salvar dados transformados
    df_treino_final = pd.DataFrame(X_treino_transformado)
    df_treino_final['price'] = y_treino.values
    df_teste_final = pd.DataFrame(X_teste_transformado)
    df_teste_final['price'] = y_teste.values
    
    df_treino_final.to_csv('../data/dados_treino_transformados.csv', index=False)
    df_teste_final.to_csv('../data/dados_teste_transformados.csv', index=False)
    
    print(f"\nDados transformados salvos:")
    print(f"  - ../data/dados_treino_transformados.csv")
    print(f"  - ../data/dados_teste_transformados.csv")

if __name__ == "__main__":
    main()
