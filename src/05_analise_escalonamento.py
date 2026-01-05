"""
Análise de Escalonamento de Características
Identifica quais características necessitam de escalonamento
"""
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from utils import clean_price

def identificar_necessidade_escalonamento(df, caracteristicas):
    """Identifica características que necessitam de escalonamento"""
    df_analise = df[caracteristicas].dropna()
    analise = []
    
    for col in caracteristicas:
        if col in df_analise.columns:
            valores = df_analise[col].dropna()
            if len(valores) > 0:
                amplitude = valores.max() - valores.min()
                ordem_grandeza = np.log10(amplitude) if amplitude > 0 else 0
                
                analise.append({
                    'Característica': col,
                    'Amplitude': amplitude,
                    'Mínimo': valores.min(),
                    'Máximo': valores.max(),
                    'Ordem de Grandeza': ordem_grandeza
                })
    
    df_analise = pd.DataFrame(analise)
    amplitude_q75 = df_analise['Amplitude'].quantile(0.75)
    
    necessitam = []
    for idx, row in df_analise.iterrows():
        precisa = False
        razoes = []
        
        if row['Amplitude'] > amplitude_q75:
            precisa = True
            razoes.append("Amplitude muito grande")
        if row['Ordem de Grandeza'] > 3:
            precisa = True
            razoes.append("Ordem de grandeza alta")
        if row['Máximo'] > 1000:
            precisa = True
            razoes.append("Valores muito altos")
        
        if precisa:
            necessitam.append({
                'Característica': row['Característica'],
                'Amplitude': row['Amplitude'],
                'Mínimo': row['Mínimo'],
                'Máximo': row['Máximo'],
                'Razões': ', '.join(razoes)
            })
    
    return pd.DataFrame(necessitam)

def main():
    df = pd.read_csv('../data/listings.csv', low_memory=False)
    df['price'] = df['price'].apply(clean_price)
    
    caracteristicas_numericas = [
        'price', 'latitude', 'longitude', 'accommodates', 'bedrooms',
        'bathrooms', 'number_of_reviews', 'review_scores_rating',
        'availability_365', 'minimum_nights', 'maximum_nights'
    ]
    
    df_necessitam = identificar_necessidade_escalonamento(df, caracteristicas_numericas)
    
    print("Características que necessitam de escalonamento:")
    print("=" * 70)
    if len(df_necessitam) > 0:
        print(df_necessitam.to_string(index=False))
    else:
        print("Nenhuma característica identificada.")
    
    print("\nRecomendação: Aplicar StandardScaler ou MinMaxScaler em todas as")
    print("características numéricas antes de treinar modelos de ML.")

if __name__ == "__main__":
    main()
