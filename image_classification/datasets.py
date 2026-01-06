"""
Função estilo sklearn para carregar o dataset de classificação de imagens
"""
import numpy as np
from pathlib import Path

def fetch_airbnb_mnist(version="hog", data_home=None):
    """
    Carrega o dataset de classificação de imagens de cômodos
    
    Parâmetros:
    -----------
    version : str, default='hog'
        Versão das features: 'hog' ou 'flatten'
    
    data_home : str ou Path, default=None
        Diretório onde os dados estão armazenados.
        Se None, usa o diretório padrão: image_classification/data/processed/
    
    Retorna:
    --------
    dict : Dicionário contendo:
        - 'data': array numpy com as features (n_samples, n_features)
        - 'target': array numpy com os labels numéricos (n_samples,)
        - 'target_names': array com os nomes das classes
        - 'DESCR': descrição do dataset
    """
    if data_home is None:
        # Diretório padrão relativo a este arquivo
        data_home = Path(__file__).parent / "data" / "processed"
    else:
        data_home = Path(data_home)
    
    # Nome do arquivo
    filename = f"mini_airbnb_mnist_{version}.npz"
    filepath = data_home / filename
    
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado: {filepath}\n"
            f"Execute primeiro: python scripts/02_preprocess.py --version {version}"
        )
    
    # Carregar dados
    data = np.load(filepath, allow_pickle=True)
    
    # Descrição
    DESCR = f"""
    Dataset de Classificação de Imagens - Tipos de Cômodos
    
    Dataset baseado no MIT LabelMe Indoor Scene Dataset.
    
    Características:
    - Features: {version.upper()}
    - Classes: bedroom, kitchen, bathroom, livingroom
    - Total de amostras: {len(data['X'])}
    - Shape das features: {data['X'].shape}
    
    Classes:
    {', '.join(data['target_names'])}
    """
    
    return {
        "data": data["X"],
        "target": data["y"],
        "target_names": data["target_names"],
        "DESCR": DESCR
    }

# Exemplo de uso
if __name__ == "__main__":
    print("Testando carregamento do dataset...")
    
    # Carregar com HOG
    print("\n📊 Carregando dataset com features HOG:")
    data_hog = fetch_airbnb_mnist(version="hog")
    print(f"  Shape: {data_hog['data'].shape}")
    print(f"  Classes: {data_hog['target_names']}")
    print(f"  Amostras: {len(data_hog['target'])}")
    
    # Carregar com flatten
    try:
        print("\n📊 Carregando dataset com features Flatten:")
        data_flat = fetch_airbnb_mnist(version="flatten")
        print(f"  Shape: {data_flat['data'].shape}")
        print(f"  Classes: {data_flat['target_names']}")
        print(f"  Amostras: {len(data_flat['target'])}")
    except FileNotFoundError:
        print("  ⚠️ Dataset flatten não encontrado. Execute: python scripts/02_preprocess.py --version flatten")

