"""
Amostragem Estratificada
Cria conjunto de teste estratificado mantendo proporção de estratos de preço
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from utils import clean_price

def criar_estratos_preco(df):
    """Cria estratos de preço usando quantis"""
    q1 = df['price'].quantile(0.2)
    q2 = df['price'].quantile(0.4)
    q3 = df['price'].quantile(0.6)
    q4 = df['price'].quantile(0.8)
    
    def categorizar(preco):
        if preco <= q1:
            return 'Muito Baixo'
        elif preco <= q2:
            return 'Baixo'
        elif preco <= q3:
            return 'Médio'
        elif preco <= q4:
            return 'Alto'
        else:
            return 'Muito Alto'
    
    return df['price'].apply(categorizar)

def main():
    df = pd.read_csv('../data/listings.csv', low_memory=False)
    df['price'] = df['price'].apply(clean_price)
    df_completo = df.dropna(subset=['price']).copy()
    
    df_completo['estrato_preco'] = criar_estratos_preco(df_completo)
    
    X = df_completo.drop('estrato_preco', axis=1)
    y = df_completo['estrato_preco']
    
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    X_teste.to_csv('../data/amostra_20_porcento.csv', index=False)
    print(f"Amostra estratificada salva: {len(X_teste)} registros (20%)")
    print(f"Conjunto de treino: {len(X_treino)} registros (80%)")

if __name__ == "__main__":
    main()
