"""
Visualização de dados do Airbnb - Histogramas
Gera histogramas para análise exploratória dos dados
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from utils import clean_price

plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (12, 8)

def plot_histograms(df):
    """Plota histogramas para análise exploratória dos dados"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Histogramas - Dados do Airbnb', fontsize=16, fontweight='bold')
    
    prices = df['price'].dropna()
    if len(prices) > 0:
        q99 = prices.quantile(0.99)
        prices_filtered = prices[prices <= q99]
        axes[0, 0].hist(prices_filtered, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        axes[0, 0].set_xlabel('Preço (R$)')
        axes[0, 0].set_ylabel('Frequência')
        axes[0, 0].set_title('Distribuição de Preços')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axvline(prices.mean(), color='red', linestyle='--', linewidth=2, 
                          label=f'Média: R$ {prices.mean():.2f}')
        axes[0, 0].legend()
    
    longitudes = df['longitude'].dropna()
    axes[0, 1].hist(longitudes, bins=50, edgecolor='black', alpha=0.7, color='lightgreen')
    axes[0, 1].set_xlabel('Longitude')
    axes[0, 1].set_ylabel('Frequência')
    axes[0, 1].set_title('Distribuição de Longitude')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axvline(longitudes.mean(), color='red', linestyle='--', linewidth=2, 
                      label=f'Média: {longitudes.mean():.4f}')
    axes[0, 1].legend()
    
    latitudes = df['latitude'].dropna()
    axes[0, 2].hist(latitudes, bins=50, edgecolor='black', alpha=0.7, color='salmon')
    axes[0, 2].set_xlabel('Latitude')
    axes[0, 2].set_ylabel('Frequência')
    axes[0, 2].set_title('Distribuição de Latitude')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].axvline(latitudes.mean(), color='red', linestyle='--', linewidth=2, 
                      label=f'Média: {latitudes.mean():.4f}')
    axes[0, 2].legend()
    
    accommodates = df['accommodates'].dropna()
    axes[1, 0].hist(accommodates, bins=30, edgecolor='black', alpha=0.7, color='gold')
    axes[1, 0].set_xlabel('Capacidade de Pessoas')
    axes[1, 0].set_ylabel('Frequência')
    axes[1, 0].set_title('Distribuição de Capacidade de Acomodação')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axvline(accommodates.mean(), color='red', linestyle='--', linewidth=2, 
                      label=f'Média: {accommodates.mean():.2f}')
    axes[1, 0].legend()
    
    if 'neighbourhood_cleansed' in df.columns:
        price_by_neighborhood = df.groupby('neighbourhood_cleansed')['price'].mean().dropna()
        axes[1, 1].hist(price_by_neighborhood, bins=30, edgecolor='black', alpha=0.7, color='plum')
        axes[1, 1].set_xlabel('Média de Preço por Bairro (R$)')
        axes[1, 1].set_ylabel('Número de Bairros')
        axes[1, 1].set_title('Distribuição de Médias de Preços por Bairro')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axvline(price_by_neighborhood.mean(), color='red', linestyle='--', linewidth=2, 
                         label=f'Média Geral: R$ {price_by_neighborhood.mean():.2f}')
        axes[1, 1].legend()
    
    axes[1, 2].axis('off')
    stats_text = "ESTATÍSTICAS RESUMIDAS\n" + "="*40 + "\n\n"
    if len(prices) > 0:
        stats_text += f"PREÇOS:\n  Média: R$ {prices.mean():.2f}\n  Mediana: R$ {prices.median():.2f}\n"
        stats_text += f"  Desvio Padrão: R$ {prices.std():.2f}\n  Mínimo: R$ {prices.min():.2f}\n"
        stats_text += f"  Máximo: R$ {prices.max():.2f}\n\n"
    if len(longitudes) > 0:
        stats_text += f"LONGITUDE:\n  Média: {longitudes.mean():.4f}\n"
        stats_text += f"  Mínimo: {longitudes.min():.4f}\n  Máximo: {longitudes.max():.4f}\n\n"
    if len(latitudes) > 0:
        stats_text += f"LATITUDE:\n  Média: {latitudes.mean():.4f}\n"
        stats_text += f"  Mínimo: {latitudes.min():.4f}\n  Máximo: {latitudes.max():.4f}\n\n"
    if len(accommodates) > 0:
        stats_text += f"CAPACIDADE:\n  Média: {accommodates.mean():.2f} pessoas\n"
        stats_text += f"  Mínimo: {int(accommodates.min())}\n  Máximo: {int(accommodates.max())}\n"
    
    axes[1, 2].text(0.1, 0.9, stats_text, transform=axes[1, 2].transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig

def main():
    df = pd.read_csv('data/listings.csv', low_memory=False)
    df['price'] = df['price'].apply(clean_price)
    fig = plot_histograms(df)
    fig.savefig('results/histogramas_airbnb.png', dpi=300, bbox_inches='tight')
    print(f"Histogramas salvos em: results/histogramas_airbnb.png")

if __name__ == "__main__":
    main()
