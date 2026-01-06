"""
Treinamento de modelos para classificação de imagens
"""
import numpy as np
from pathlib import Path
import sys

# Adicionar path para importar datasets
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import fetch_airbnb_mnist
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Configurações
MODELS_DIR = Path(__file__).parent.parent / "models"
RESULTS_DIR = Path(__file__).parent.parent / "results"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def train_models(version="hog", test_size=0.2, random_state=42):
    """
    Treina múltiplos modelos e compara performance
    
    Args:
        version: "hog" ou "flatten"
        test_size: proporção do conjunto de teste
        random_state: seed para reprodutibilidade
    """
    print("=" * 70)
    print(f"TREINAMENTO DE MODELOS - Versão: {version.upper()}")
    print("=" * 70)
    
    # Carregar dados
    print("\n📥 Carregando dataset...")
    data = fetch_airbnb_mnist(version=version)
    X, y = data["data"], data["target"]
    target_names = data["target_names"]
    
    print(f"  Shape: {X.shape}")
    print(f"  Classes: {target_names}")
    print(f"  Amostras: {len(y)}")
    
    # Dividir em treino e teste
    print(f"\n🔄 Dividindo dados (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"  Treino: {X_train.shape[0]} amostras")
    print(f"  Teste: {X_test.shape[0]} amostras")
    
    # Modelos a treinar
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            n_jobs=-1
        ),
        "LinearSVM": LinearSVC(
            max_iter=10000,
            random_state=random_state
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100,
            random_state=random_state
        )
    }
    
    results = {}
    
    # Treinar cada modelo
    for name, model in models.items():
        print(f"\n{'='*70}")
        print(f"🔧 Treinando: {name}")
        print(f"{'='*70}")
        
        # Treinar
        print("⏳ Treinando modelo...")
        model.fit(X_train, y_train)
        
        # Predições
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        print("⏳ Calculando cross-validation...")
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, n_jobs=-1)
        
        # Salvar modelo
        model_file = MODELS_DIR / f"{name.lower()}_{version}.pkl"
        joblib.dump(model, model_file)
        print(f"💾 Modelo salvo em: {model_file}")
        
        # Relatório
        report = classification_report(
            y_test, y_pred,
            target_names=target_names,
            output_dict=True
        )
        
        # Matriz de confusão
        cm = confusion_matrix(y_test, y_pred)
        
        results[name] = {
            "model": model,
            "accuracy": accuracy,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "classification_report": report,
            "confusion_matrix": cm,
            "y_test": y_test,
            "y_pred": y_pred
        }
        
        print(f"\n📊 Resultados:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  CV Score (média): {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # Comparação final
    print("\n" + "=" * 70)
    print("COMPARAÇÃO DE MODELOS")
    print("=" * 70)
    print(f"{'Modelo':<25} {'Accuracy':<12} {'CV Score':<15}")
    print("-" * 70)
    for name, result in results.items():
        print(f"{name:<25} {result['accuracy']:<12.4f} {result['cv_mean']:.4f} (+/- {result['cv_std']*2:.4f})")
    
    # Melhor modelo
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n🏆 Melhor modelo: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.4f})")
    
    # Salvar relatório detalhado
    report_file = RESULTS_DIR / f"results_{version}.txt"
    with open(report_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write(f"RELATÓRIO DE TREINAMENTO - {version.upper()}\n")
        f.write("=" * 70 + "\n\n")
        
        for name, result in results.items():
            f.write(f"\n{'='*70}\n")
            f.write(f"MODELO: {name}\n")
            f.write(f"{'='*70}\n")
            f.write(f"Accuracy: {result['accuracy']:.4f}\n")
            f.write(f"CV Score: {result['cv_mean']:.4f} (+/- {result['cv_std']*2:.4f})\n\n")
            f.write("Classification Report:\n")
            f.write(classification_report(
                result['y_test'],
                result['y_pred'],
                target_names=target_names
            ))
            f.write("\nConfusion Matrix:\n")
            f.write(str(result['confusion_matrix']))
            f.write("\n\n")
    
    print(f"\n💾 Relatório salvo em: {report_file}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Treinar modelos de classificação")
    parser.add_argument(
        "--version",
        type=str,
        default="hog",
        choices=["hog", "flatten"],
        help="Versão de features: 'hog' ou 'flatten'"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proporção do conjunto de teste"
    )
    
    args = parser.parse_args()
    
    train_models(version=args.version, test_size=args.test_size)

