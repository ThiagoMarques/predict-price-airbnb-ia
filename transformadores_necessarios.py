"""
Análise de Transformadores Necessários
Identifica quais transformadores aplicar aos dados
"""
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from utils import clean_price

def analisar_caracteristicas_numericas(df, caracteristicas):
    """Analisa características numéricas e recomenda scalers"""
    df_numerico = df[caracteristicas].dropna()
    recomendacoes = []
    
    for col in caracteristicas:
        if col in df_numerico.columns:
            valores = df_numerico[col].dropna()
            if len(valores) > 0:
                q1 = valores.quantile(0.25)
                q3 = valores.quantile(0.75)
                iqr = q3 - q1
                outliers = len(valores[(valores < q1 - 1.5*iqr) | (valores > q3 + 1.5*iqr)])
                pct_outliers = (outliers / len(valores)) * 100
                amplitude = valores.max() - valores.min()
                
                if pct_outliers > 10:
                    recomendacao = "RobustScaler (muitos outliers)"
                elif amplitude > 1000:
                    recomendacao = "StandardScaler ou MinMaxScaler"
                else:
                    recomendacao = "StandardScaler"
                
                recomendacoes.append({
                    'Característica': col,
                    'Amplitude': amplitude,
                    'Outliers (%)': pct_outliers,
                    'Recomendação': recomendacao
                })
    
    return pd.DataFrame(recomendacoes)

def analisar_caracteristicas_categoricas(df, caracteristicas):
    """Analisa características categóricas e recomenda encoders"""
    recomendacoes = []
    
    for col in caracteristicas:
        if col in df.columns:
            valores_unicos = df[col].nunique()
            
            if valores_unicos <= 5:
                recomendacao = "OneHotEncoder (poucas categorias)"
            elif valores_unicos <= 20:
                recomendacao = "OneHotEncoder (cuidado com dimensionalidade)"
            else:
                recomendacao = "Considerar agrupar categorias ou usar embedding"
            
            recomendacoes.append({
                'Característica': col,
                'Valores únicos': valores_unicos,
                'Recomendação': recomendacao
            })
    
    return pd.DataFrame(recomendacoes)

def main():
    df = pd.read_csv('data/listings.csv', low_memory=False)
    df['price'] = df['price'].apply(clean_price)
    
    caracteristicas_numericas = [
        'price', 'latitude', 'longitude', 'accommodates', 'bedrooms',
        'bathrooms', 'number_of_reviews', 'review_scores_rating',
        'availability_365', 'minimum_nights', 'maximum_nights'
    ]
    
    caracteristicas_categoricas = [
        'property_type', 'room_type', 'neighbourhood_cleansed',
        'host_is_superhost', 'instant_bookable'
    ]
    
    print("Transformadores para DADOS NUMÉRICOS:")
    print("=" * 70)
    df_num = analisar_caracteristicas_numericas(df, caracteristicas_numericas)
    print(df_num.to_string(index=False))
    
    print("\n\nTransformadores para DADOS CATEGÓRICOS:")
    print("=" * 70)
    df_cat = analisar_caracteristicas_categoricas(df, caracteristicas_categoricas)
    print(df_cat.to_string(index=False))
    
    print("\n\nResumo:")
    print("=" * 70)
    print("1. Numéricos: SimpleImputer + StandardScaler (ou RobustScaler se muitos outliers)")
    print("2. Categóricos: SimpleImputer + OneHotEncoder")
    print("3. Texto: TfidfVectorizer (já implementado)")

if __name__ == "__main__":
    main()
