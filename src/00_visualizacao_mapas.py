"""
Visualização de Mapas - Localização e Preços
Gera mapas geográficos dos imóveis do Airbnb no Rio de Janeiro
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from utils import clean_price

def plot_mapa_localizacao(df):
    """Plota mapa de distribuição geográfica"""
    df_coordenadas = df.dropna(subset=['latitude', 'longitude']).copy()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(df_coordenadas['longitude'], df_coordenadas['latitude'], 
               alpha=0.2, s=1)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('Distribuição Geográfica dos Imóveis do Airbnb', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig('../results/mapa_localizacao_airbnb.png', dpi=300, bbox_inches='tight')
    plt.close()
    return len(df_coordenadas)

def plot_mapa_preco(df):
    """Plota mapa de distribuição de preços"""
    df_completo = df.dropna(subset=['latitude', 'longitude', 'price']).copy()
    q99 = df_completo['price'].quantile(0.99)
    df_filtrado = df_completo[df_completo['price'] <= q99].copy()
    
    fig, ax = plt.subplots(figsize=(14, 10))
    scatter = ax.scatter(df_filtrado['longitude'], df_filtrado['latitude'], 
                        c=df_filtrado['price'], cmap='viridis', s=1)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('Distribuição Geográfica dos Preços dos Imóveis do Airbnb', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Preço (R$)', fontsize=12, rotation=270, labelpad=20)
    plt.tight_layout()
    plt.savefig('../results/mapa_preco_localizacao_airbnb.png', dpi=300, bbox_inches='tight')
    plt.close()
    return len(df_filtrado)

def main():
    df = pd.read_csv('../data/listings.csv', low_memory=False)
    df['price'] = df['price'].apply(clean_price)
    
    n_localizacao = plot_mapa_localizacao(df)
    n_preco = plot_mapa_preco(df)
    
    print(f"Mapa de localização: {n_localizacao} imóveis")
    print(f"Mapa de preços: {n_preco} imóveis")
    print("Gráficos salvos em: ../results/")

if __name__ == "__main__":
    main()
