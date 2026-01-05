"""
Análise de Correlação de Pearson
Calcula correlações entre características numéricas e preço
"""
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from utils import clean_price

def calcular_correlacoes(df, atributos, target='price'):
    """Calcula correlações de Pearson entre atributos e target"""
    correlacoes = []
    valores_target = df[target]
    
    for attr in atributos:
        if attr != target:
            valores = df[attr]
            r, p_value = pearsonr(valores_target, valores)
            correlacoes.append({
                'Atributo': attr,
                'Correlação (r)': r,
                'P-valor': p_value,
                'Interpretação': 'Forte' if abs(r) > 0.7 else ('Moderada' if abs(r) > 0.4 else 'Fraca')
            })
    
    df_corr = pd.DataFrame(correlacoes)
    return df_corr.sort_values('Correlação (r)', key=abs, ascending=False)

def main():
    df = pd.read_csv('../data/listings.csv', low_memory=False)
    df['price'] = df['price'].apply(clean_price)
    
    atributos = [
        'price', 'latitude', 'longitude', 'accommodates', 'bedrooms',
        'bathrooms', 'number_of_reviews', 'review_scores_rating',
        'availability_365', 'minimum_nights'
    ]
    
    df_limpo = df[atributos].dropna()
    df_correlacoes = calcular_correlacoes(df_limpo, atributos)
    
    print("Correlações com PREÇO (ordenadas por força):")
    print("=" * 70)
    print(df_correlacoes.to_string(index=False))
    
    print("\nTop 5 atributos mais correlacionados:")
    for idx, row in df_correlacoes.head(5).iterrows():
        print(f"  {row['Atributo']}: r = {row['Correlação (r)']:.4f} ({row['Interpretação']})")

if __name__ == "__main__":
    main()
