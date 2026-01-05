"""
Funções utilitárias compartilhadas entre os scripts
"""
import pandas as pd
import numpy as np

def clean_price(price_str):
    """
    Limpa e converte o preço de string para float.
    Remove símbolos $, vírgulas e espaços.
    
    Parameters:
    -----------
    price_str : str
        String com o preço no formato "$XXX.XX"
    
    Returns:
    --------
    float
        Preço convertido para float, ou np.nan se inválido
    """
    if pd.isna(price_str) or price_str == '':
        return np.nan
    
    price_clean = str(price_str).replace('$', '').replace(',', '').strip()
    
    try:
        return float(price_clean)
    except:
        return np.nan

