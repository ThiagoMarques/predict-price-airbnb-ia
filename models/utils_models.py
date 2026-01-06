"""
Funções utilitárias para scripts de modelos
Carrega dados transformados quando disponíveis
"""
import pandas as pd
import numpy as np
import os

def carregar_dados_transformados(project_root):
    """
    Carrega dados transformados se existirem, senão retorna None
    Retorna: (X_treino, y_treino, X_teste, y_teste) ou None
    """
    treino_path = os.path.join(project_root, 'data', 'dados_treino_transformados.csv')
    teste_path = os.path.join(project_root, 'data', 'dados_teste_transformados.csv')
    
    if os.path.exists(treino_path) and os.path.exists(teste_path):
        print("✓ Carregando dados transformados salvos...")
        df_treino = pd.read_csv(treino_path)
        df_teste = pd.read_csv(teste_path)
        
        X_treino = df_treino.drop('price', axis=1).values
        y_treino = df_treino['price'].values
        X_teste = df_teste.drop('price', axis=1).values
        y_teste = df_teste['price'].values
        
        print(f"  Treino: {X_treino.shape[0]} amostras, {X_treino.shape[1]} features")
        print(f"  Teste: {X_teste.shape[0]} amostras, {X_teste.shape[1]} features")
        
        return X_treino, y_treino, X_teste, y_teste
    
    return None


