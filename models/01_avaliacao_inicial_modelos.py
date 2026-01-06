"""
Avaliação Inicial de Modelos
Começa com LinearRegression e avalia no conjunto de treinamento
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))
from utils import clean_price
from utils_models import carregar_dados_transformados

def criar_pipeline_preprocessamento(caracteristicas_numericas, caracteristicas_categoricas):
    """Cria pipeline de pré-processamento"""
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

def avaliar_modelo(y_real, y_pred, nome_modelo):
    """Avalia modelo e retorna métricas"""
    mse = mean_squared_error(y_real, y_pred)
    mae = mean_absolute_error(y_real, y_pred)
    r2 = r2_score(y_real, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"\n{nome_modelo}:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MSE: {mse:.2f}")
    print(f"  MAE: R$ {mae:.2f}")
    print(f"  R²: {r2:.4f}")
    
    return {'mse': mse, 'mae': mae, 'r2': r2, 'rmse': rmse}

def main():
    print("=" * 70)
    print("AVALIAÇÃO INICIAL: LinearRegression no Conjunto de Treinamento")
    print("=" * 70)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Tentar carregar dados transformados
    dados_transformados = carregar_dados_transformados(project_root)
    
    if dados_transformados is not None:
        X_treino_transformado, y_treino, X_teste_transformado, y_teste = dados_transformados
        preprocessor = None  # Não precisamos do preprocessor se dados já estão transformados
        print("\n✓ Usando dados transformados salvos (sem reprocessamento)")
    else:
        print("⚠️  Dados transformados não encontrados. Processando dados originais...")
        data_path = os.path.join(project_root, 'data', 'listings.csv')
        df = pd.read_csv(data_path, low_memory=False)
        df['price'] = df['price'].apply(clean_price)
        
        caracteristicas_numericas = [
            'latitude', 'longitude', 'accommodates', 'bedrooms',
            'bathrooms', 'number_of_reviews', 'review_scores_rating',
            'availability_365', 'minimum_nights', 'maximum_nights'
        ]
        
        caracteristicas_categoricas = [
            'property_type', 'room_type', 'neighbourhood_cleansed'
        ]
        
        target = 'price'
        
        df_selecionado = df[caracteristicas_numericas + caracteristicas_categoricas + [target]].copy()
        df_completo = df_selecionado.dropna(subset=[target])
        
        X = df_completo[caracteristicas_numericas + caracteristicas_categoricas]
        y = df_completo[target]
        
        X_treino, X_teste, y_treino, y_teste = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"\nDados preparados:")
        print(f"  Treino: {len(X_treino)} registros")
        print(f"  Teste: {len(X_teste)} registros")
        
        preprocessor = criar_pipeline_preprocessamento(caracteristicas_numericas, caracteristicas_categoricas)
        X_treino_transformado = preprocessor.fit_transform(X_treino)
        X_teste_transformado = preprocessor.transform(X_teste)
        
        print(f"\nShape após transformação:")
        print(f"  Treino: {X_treino_transformado.shape}")
        print(f"  Teste: {X_teste_transformado.shape}")
    
    print("\n" + "=" * 70)
    print("PASSO 1: Treinar LinearRegression e avaliar no TREINAMENTO")
    print("=" * 70)
    print("Nota: Avaliar no conjunto de treinamento primeiro para ver se o modelo")
    print("consegue aprender os padrões básicos (mesmo que haja overfitting)")
    
    modelo_lr = LinearRegression()
    modelo_lr.fit(X_treino_transformado, y_treino)
    
    y_pred_treino = modelo_lr.predict(X_treino_transformado)
    metricas_treino = avaliar_modelo(y_treino, y_pred_treino, "LinearRegression (Treino)")
    
    print("\n" + "=" * 70)
    print("PASSO 2: Avaliar LinearRegression no TESTE")
    print("=" * 70)
    print("Avaliar no conjunto de teste para ver generalização")
    
    y_pred_teste = modelo_lr.predict(X_teste_transformado)
    metricas_teste = avaliar_modelo(y_teste, y_pred_teste, "LinearRegression (Teste)")
    
    print("\n" + "=" * 70)
    print("ANÁLISE INICIAL")
    print("=" * 70)
    print(f"Diferença R² (Treino - Teste): {metricas_treino['r2'] - metricas_teste['r2']:.4f}")
    
    if metricas_treino['r2'] - metricas_teste['r2'] > 0.1:
        print("⚠️  Grande diferença entre treino e teste indica possível overfitting")
    else:
        print("✓ Diferença pequena - modelo está generalizando bem")
    
    if metricas_teste['r2'] < 0.3:
        print("⚠️  R² baixo - modelo não está se adaptando bem aos dados")
        print("   Próximo passo: testar modelos mais complexos (RandomForest, etc.)")
    else:
        print("✓ R² aceitável - modelo está capturando padrões")
    
    # Salvar modelo
    import joblib
    saved_dir = os.path.join(project_root, 'models', 'saved')
    os.makedirs(saved_dir, exist_ok=True)
    joblib.dump(modelo_lr, os.path.join(saved_dir, 'linear_regression.pkl'))
    if preprocessor is not None:
        joblib.dump(preprocessor, os.path.join(saved_dir, 'preprocessor.pkl'))
    print("\n✓ Modelo salvo em: models/saved/")

if __name__ == "__main__":
    main()

