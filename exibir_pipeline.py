"""
Exibição da Pipeline de Transformação
Mostra estrutura e detalhes da pipeline de pré-processamento
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def criar_pipeline():
    """Cria e retorna a pipeline de pré-processamento"""
    caracteristicas_numericas = [
        'latitude', 'longitude', 'accommodates', 'bedrooms',
        'bathrooms', 'number_of_reviews', 'review_scores_rating',
        'availability_365', 'minimum_nights', 'maximum_nights'
    ]
    
    caracteristicas_categoricas = [
        'property_type', 'room_type', 'neighbourhood_cleansed'
    ]
    
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

def exibir_estrutura():
    """Exibe estrutura visual da pipeline"""
    print("Estrutura da Pipeline:")
    print("=" * 70)
    print("\nColumnTransformer")
    print("  ├── Pipeline 'num' (dados numéricos)")
    print("  │   ├── SimpleImputer(strategy='median')")
    print("  │   └── StandardScaler()")
    print("  │")
    print("  └── Pipeline 'cat' (dados categóricos)")
    print("      ├── SimpleImputer(strategy='most_frequent')")
    print("      └── OneHotEncoder(sparse_output=False, handle_unknown='ignore')")

def exibir_ordem_execucao():
    """Exibe ordem de execução dos transformadores"""
    print("\nOrdem de execução quando fit_transform() é chamado:")
    print("=" * 70)
    print("\n1. DADOS NUMÉRICOS:")
    print("   a) SimpleImputer.fit() → calcula medianas")
    print("   b) SimpleImputer.transform() → preenche valores faltantes")
    print("   c) StandardScaler.fit() → calcula médias e desvios")
    print("   d) StandardScaler.transform() → normaliza dados")
    print("\n2. DADOS CATEGÓRICOS:")
    print("   a) SimpleImputer.fit() → identifica valores mais frequentes")
    print("   b) SimpleImputer.transform() → preenche valores faltantes")
    print("   c) OneHotEncoder.fit() → identifica todas as categorias")
    print("   d) OneHotEncoder.transform() → cria colunas binárias")
    print("\n3. COMBINAÇÃO:")
    print("   → Junta dados numéricos normalizados + categóricos encoded")

def main():
    preprocessor = criar_pipeline()
    
    exibir_estrutura()
    exibir_ordem_execucao()
    
    print("\n\nCódigo da Pipeline:")
    print("=" * 70)
    print("""
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

preprocessor = ColumnTransformer(
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
    """)
    
    print("\nUso:")
    print("=" * 70)
    print("X_treino_transformado = preprocessor.fit_transform(X_treino)")
    print("X_teste_transformado = preprocessor.transform(X_teste)")

if __name__ == "__main__":
    main()
