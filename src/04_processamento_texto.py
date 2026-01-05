"""
Processamento de Texto com TF-IDF
Analisa descrições dos imóveis para identificar palavras-chave relacionadas ao preço
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from utils import clean_price

# Stop words em português e inglês
stop_words_portugues = [
    'o', 'a', 'os', 'as', 'um', 'uma', 'uns', 'umas',
    'de', 'do', 'da', 'dos', 'das', 'em', 'no', 'na', 'nos', 'nas',
    'para', 'por', 'com', 'sem', 'sob', 'sobre', 'entre', 'até',
    'que', 'qual', 'quais', 'quando', 'onde', 'como', 'quanto',
    'é', 'são', 'foi', 'foram', 'ser', 'estar', 'ter',
    'se', 'seu', 'sua', 'seus', 'suas', 'meu', 'minha', 'meus', 'minhas',
    'teu', 'tua', 'teus', 'tuas', 'nosso', 'nossa', 'nossos', 'nossas',
    'este', 'esta', 'estes', 'estas', 'esse', 'essa', 'esses', 'essas',
    'aquele', 'aquela', 'aqueles', 'aquelas', 'isto', 'isso', 'aquilo',
    'ele', 'ela', 'eles', 'elas', 'nós', 'você', 'vocês',
    'me', 'te', 'nos', 'vos', 'lhe', 'lhes', 'si', 'consigo',
    'mais', 'menos', 'muito', 'pouco', 'muita', 'pouca', 'muitos', 'poucos',
    'também', 'ainda', 'já', 'só', 'somente', 'apenas',
    'não', 'nem', 'ou', 'mas', 'porém', 'contudo', 'todavia', 'entretanto',
    'e', 'quando', 'enquanto', 'porque', 'pois',
    'ao', 'aos', 'às', 'pelo', 'pela', 'pelos', 'pelas', 'dum', 'duma',
    'duns', 'dumas', 'num', 'numa', 'nuns', 'numas',
    'era', 'eram', 'seja', 'sejam',
    'está', 'estão', 'estava', 'estavam',
    'tem', 'têm', 'tinha', 'tinham',
    'há', 'houve', 'havia',
    'vai', 'vão', 'iria', 'iriam',
    'pode', 'podem', 'poder',
    'deve', 'devem', 'dever'
]

stop_words_ingles = [
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
    'including', 'against', 'among', 'throughout', 'despite', 'towards',
    'upon', 'concerning', 'before', 'after', 'above', 'below', 'down',
    'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
    'here', 'there', 'when', 'where', 'why', 'how', 'if', 'because', 'as',
    'until', 'while', 'all', 'each', 'few', 'more', 'most', 'other',
    'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
    'too', 'very', 'just', 'now',
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
    'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'hers',
    'ours', 'theirs', 'myself', 'yourself', 'himself', 'herself', 'itself',
    'ourselves', 'yourselves', 'themselves',
    'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
    'will', 'would', 'shall', 'should', 'may', 'might', 'must', 'can', 'could',
    'am', 'ought', 'need', 'dare', 'used'
]

stop_words_html_tags = ['br', 'html', 'body', 'div', 'span', 'p', 'strong', 'em', 'b', 'i', 'u']

stop_words_numeros = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
    '20', '21', '22', '23', '24', '25', '26', '27', '28', '29',
    '30', '31', '32', '33', '34', '35', '36', '37', '38', '39',
    '40', '41', '42', '43', '44', '45', '46', '47', '48', '49',
    '50', '60', '70', '80', '90', '100', '200', '300', '365'
]

stop_words_combinadas = list(set(stop_words_portugues + stop_words_ingles + stop_words_html_tags + stop_words_numeros))

def analisar_palavras_preco(df_texto, vectorizer, top_n=15):
    """Analisa correlação entre palavras e preço"""
    X = vectorizer.fit_transform(df_texto['description'].fillna(''))
    feature_names = vectorizer.get_feature_names_out()
    precos = df_texto['price'].values
    X_dense = X.toarray()
    
    correlacoes = []
    for i, palavra in enumerate(feature_names):
        valores_palavra = X_dense[:, i]
        if np.std(valores_palavra) > 0:
            correlacao = np.corrcoef(valores_palavra, precos)[0, 1]
            if not np.isnan(correlacao):
                correlacoes.append({'Palavra': palavra, 'Correlação': correlacao})
    
    df_corr = pd.DataFrame(correlacoes)
    return df_corr.sort_values('Correlação', key=abs, ascending=False).head(top_n)

def comparar_caros_baratos(df_texto, preco_mediano):
    """Compara palavras entre imóveis caros e baratos"""
    caros = df_texto[df_texto['price'] > preco_mediano]
    baratos = df_texto[df_texto['price'] <= preco_mediano]
    
    vectorizer = TfidfVectorizer(max_features=30, stop_words=stop_words_combinadas, min_df=3)
    X_caros = vectorizer.fit_transform(caros['description'].fillna(''))
    X_baratos = vectorizer.fit_transform(baratos['description'].fillna(''))
    palavras = vectorizer.get_feature_names_out()
    
    tfidf_caros = X_caros.mean(axis=0).A1
    tfidf_baratos = X_baratos.mean(axis=0).A1
    
    df_comparacao = pd.DataFrame({
        'Palavra': palavras,
        'TF-IDF Caros': tfidf_caros,
        'TF-IDF Baratos': tfidf_baratos
    })
    df_comparacao['Diferença'] = df_comparacao['TF-IDF Caros'] - df_comparacao['TF-IDF Baratos']
    return df_comparacao.sort_values('Diferença', key=abs, ascending=False)

def main():
    df = pd.read_csv('../data/listings.csv', low_memory=False)
    df['price'] = df['price'].apply(clean_price)
    df_texto = df[['description', 'price']].dropna(subset=['description', 'price']).copy()
    
    # TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=100,
        stop_words=stop_words_combinadas,
        min_df=5,
        max_df=0.8,
        lowercase=True
    )
    
    X = vectorizer.fit_transform(df_texto['description'].fillna(''))
    feature_names = vectorizer.get_feature_names_out()
    
    print(f"Top 20 palavras mais relevantes:")
    for i, palavra in enumerate(feature_names[:20], 1):
        print(f"  {i:2d}. {palavra}")
    
    # Correlação com preço
    df_corr = analisar_palavras_preco(df_texto, vectorizer)
    print(f"\nTop 15 palavras mais correlacionadas com PREÇO:")
    for idx, row in df_corr.iterrows():
        sinal = "↑" if row['Correlação'] > 0 else "↓"
        print(f"  {row['Palavra']:20s}: r = {row['Correlação']:+.4f} ({sinal})")
    
    # Comparação caros vs baratos
    preco_mediano = df_texto['price'].median()
    df_comparacao = comparar_caros_baratos(df_texto, preco_mediano)
    
    print(f"\nPalavras que mais diferenciam imóveis CAROS:")
    for idx, row in df_comparacao.head(10).iterrows():
        if row['Diferença'] > 0:
            print(f"  {row['Palavra']:20s}: +{row['Diferença']:.4f}")
    
    print(f"\nPalavras que mais diferenciam imóveis BARATOS:")
    for idx, row in df_comparacao.tail(10).iterrows():
        if row['Diferença'] < 0:
            print(f"  {row['Palavra']:20s}: {row['Diferença']:.4f}")

if __name__ == "__main__":
    main()
