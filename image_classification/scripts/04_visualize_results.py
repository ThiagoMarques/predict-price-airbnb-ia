"""
Visualização dos resultados dos modelos treinados
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import joblib

# Adicionar path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import fetch_airbnb_mnist
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Configurações
MODELS_DIR = Path(__file__).parent.parent / "models"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_model_and_predict(model_name, version="hog"):
    """Carrega modelo e faz predições"""
    model_path = MODELS_DIR / f"{model_name.lower()}_{version}.pkl"
    
    if not model_path.exists():
        return None, None, None
    
    # Carregar modelo
    model = joblib.load(model_path)
    
    # Carregar dados
    data = fetch_airbnb_mnist(version=version)
    X, y = data["data"], data["target"]
    target_names = data["target_names"]
    
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Predições
    y_pred = model.predict(X_test)
    
    return y_test, y_pred, target_names

def plot_confusion_matrix(y_test, y_pred, target_names, model_name, save_path):
    """Plota matriz de confusão"""
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=target_names,
        yticklabels=target_names
    )
    plt.title(f'Matriz de Confusão - {model_name}', fontsize=16, fontweight='bold')
    plt.ylabel('Classe Real', fontsize=12)
    plt.xlabel('Classe Prevista', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Salvo: {save_path}")

def compare_models(version="hog"):
    """Compara todos os modelos"""
    print("=" * 70)
    print("VISUALIZAÇÃO DE RESULTADOS")
    print("=" * 70)
    
    models = ["LogisticRegression", "LinearSVM", "RandomForest", "GradientBoosting"]
    results = {}
    
    # Coletar resultados
    for model_name in models:
        y_test, y_pred, target_names = load_model_and_predict(model_name, version)
        if y_test is not None:
            from sklearn.metrics import accuracy_score
            accuracy = accuracy_score(y_test, y_pred)
            results[model_name] = {
                'accuracy': accuracy,
                'y_test': y_test,
                'y_pred': y_pred,
                'target_names': target_names
            }
    
    if not results:
        print("❌ Nenhum modelo encontrado!")
        return
    
    # Plotar matrizes de confusão
    print("\n📊 Gerando matrizes de confusão...")
    for model_name, result in results.items():
        save_path = RESULTS_DIR / f"confusion_matrix_{model_name.lower()}_{version}.png"
        plot_confusion_matrix(
            result['y_test'],
            result['y_pred'],
            result['target_names'],
            model_name,
            save_path
        )
    
    # Comparação de accuracy
    print("\n📈 Gerando gráfico de comparação...")
    model_names = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in model_names]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, accuracies, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
    plt.title('Comparação de Accuracy dos Modelos', fontsize=16, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12)
    plt.xlabel('Modelo', fontsize=12)
    plt.ylim([0, 1])
    
    # Adicionar valores nas barras
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    save_path = RESULTS_DIR / f"model_comparison_{version}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Salvo: {save_path}")
    
    # Resumo
    print("\n" + "=" * 70)
    print("RESUMO DOS RESULTADOS")
    print("=" * 70)
    print(f"{'Modelo':<25} {'Accuracy':<10}")
    print("-" * 70)
    for model_name in sorted(results.keys(), key=lambda x: results[x]['accuracy'], reverse=True):
        print(f"{model_name:<25} {results[model_name]['accuracy']:.4f}")
    
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n🏆 Melhor modelo: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.4f})")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualizar resultados dos modelos")
    parser.add_argument(
        "--version",
        type=str,
        default="hog",
        choices=["hog", "flatten"],
        help="Versão de features"
    )
    
    args = parser.parse_args()
    compare_models(version=args.version)

