"""
Exemplo de uso do dataset de classificação de imagens
"""
import sys
from pathlib import Path

# Adicionar path
sys.path.insert(0, str(Path(__file__).parent))

from datasets import fetch_airbnb_mnist
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def exemplo_basico():
    """Exemplo básico de uso"""
    print("=" * 70)
    print("EXEMPLO DE USO - Classificação de Imagens")
    print("=" * 70)
    
    # Carregar dados com HOG
    print("\n📥 Carregando dataset com features HOG...")
    data = fetch_airbnb_mnist(version="hog")
    
    X, y = data["data"], data["target"]
    target_names = data["target_names"]
    
    print(f"  Shape: {X.shape}")
    print(f"  Classes: {target_names}")
    print(f"  Amostras: {len(y)}")
    
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Treinar modelo simples
    print("\n🔧 Treinando Logistic Regression...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # Predições
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n📊 Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Comparar com flatten
    print("\n" + "=" * 70)
    print("COMPARAÇÃO: HOG vs FLATTEN")
    print("=" * 70)
    
    try:
        data_flat = fetch_airbnb_mnist(version="flatten")
        X_flat, y_flat = data_flat["data"], data_flat["target"]
        
        X_train_flat, X_test_flat, y_train_flat, y_test_flat = train_test_split(
            X_flat, y_flat, test_size=0.2, random_state=42, stratify=y_flat
        )
        
        model_flat = LogisticRegression(max_iter=1000, random_state=42)
        model_flat.fit(X_train_flat, y_train_flat)
        y_pred_flat = model_flat.predict(X_test_flat)
        accuracy_flat = accuracy_score(y_test_flat, y_pred_flat)
        
        print(f"\nHOG Features:      {accuracy:.4f}")
        print(f"Flatten Features:  {accuracy_flat:.4f}")
        print(f"\nMelhoria: {((accuracy - accuracy_flat) / accuracy_flat * 100):.2f}%")
    except FileNotFoundError:
        print("\n⚠️ Dataset flatten não encontrado.")
        print("Execute: python scripts/02_preprocess.py --version flatten")

if __name__ == "__main__":
    exemplo_basico()

