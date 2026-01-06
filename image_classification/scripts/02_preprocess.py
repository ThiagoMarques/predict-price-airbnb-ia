"""
Pré-processamento e extração de features das imagens
"""
import numpy as np
from pathlib import Path
from PIL import Image
from skimage import feature
from skimage.color import rgb2gray
import os
from tqdm import tqdm

# Configurações
DATA_DIR = Path(__file__).parent.parent / "data"
# O dataset extraído tem as imagens em Images/
RAW_DIR = DATA_DIR / "raw" / "Images"
# Fallback para estrutura alternativa
if not RAW_DIR.exists():
    RAW_DIR = DATA_DIR / "raw" / "indoorCVPR_09"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Classes de interesse
CLASSES = ["bedroom", "kitchen", "bathroom", "livingroom"]
IMAGE_SIZE = 64  # 64x64 ou 32x32
GRAYSCALE = True

def load_images_from_folder(folder_path, class_name):
    """Carrega imagens de uma pasta específica"""
    images = []
    labels = []
    
    folder = Path(folder_path) / class_name
    if not folder.exists():
        print(f"⚠️ Pasta não encontrada: {folder}")
        return images, labels
    
    image_files = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
    
    for img_path in tqdm(image_files, desc=f"Carregando {class_name}"):
        try:
            img = Image.open(img_path)
            images.append(img)
            labels.append(class_name)
        except Exception as e:
            print(f"⚠️ Erro ao carregar {img_path}: {e}")
    
    return images, labels

def preprocess_image(img, size=IMAGE_SIZE, grayscale=GRAYSCALE):
    """Pré-processa uma imagem"""
    # Resize
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    
    # Converter para array numpy
    img_array = np.array(img)
    
    # Grayscale se necessário
    if grayscale:
        if len(img_array.shape) == 3:
            img_array = rgb2gray(img_array)
    
    return img_array

def extract_features_flatten(img_array):
    """Extrai features usando flatten (baseline)"""
    return img_array.flatten()

def extract_features_hog(img_array, orientations=9, pixels_per_cell=(8, 8), 
                         cells_per_block=(2, 2)):
    """Extrai features usando HOG (Histogram of Oriented Gradients)"""
    features = feature.hog(
        img_array,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        visualize=False,
        channel_axis=None
    )
    return features

def process_dataset(version="hog"):
    """
    Processa o dataset completo
    
    Args:
        version: "hog" ou "flatten"
    """
    print("=" * 70)
    print(f"PRÉ-PROCESSAMENTO - Versão: {version.upper()}")
    print("=" * 70)
    
    if not RAW_DIR.exists():
        print(f"❌ Diretório não encontrado: {RAW_DIR}")
        print("Execute primeiro: python scripts/01_download.py")
        return
    
    all_images = []
    all_labels = []
    
    # Carregar imagens de cada classe
    for class_name in CLASSES:
        print(f"\n📂 Processando classe: {class_name}")
        images, labels = load_images_from_folder(RAW_DIR, class_name)
        
        if len(images) == 0:
            print(f"⚠️ Nenhuma imagem encontrada para {class_name}")
            continue
        
        # Pré-processar e extrair features
        for img in tqdm(images, desc=f"Processando {class_name}"):
            try:
                # Pré-processar
                img_processed = preprocess_image(img, size=IMAGE_SIZE, grayscale=GRAYSCALE)
                
                # Extrair features
                if version == "hog":
                    features = extract_features_hog(img_processed)
                else:  # flatten
                    features = extract_features_flatten(img_processed)
                
                all_images.append(features)
                all_labels.append(class_name)
            except Exception as e:
                print(f"⚠️ Erro ao processar imagem: {e}")
    
    # Converter para arrays numpy
    X = np.array(all_images)
    y = np.array(all_labels)
    
    # Mapear labels para números
    label_to_num = {label: idx for idx, label in enumerate(CLASSES)}
    y_numeric = np.array([label_to_num[label] for label in y])
    
    print("\n" + "=" * 70)
    print("ESTATÍSTICAS DO DATASET")
    print("=" * 70)
    print(f"Total de amostras: {len(X)}")
    print(f"Shape das features: {X.shape}")
    print(f"Classes: {CLASSES}")
    print(f"\nDistribuição por classe:")
    for class_name in CLASSES:
        count = np.sum(y == class_name)
        print(f"  {class_name}: {count} amostras")
    
    # Salvar em formato .npz
    output_file = PROCESSED_DIR / f"mini_airbnb_mnist_{version}.npz"
    print(f"\n💾 Salvando em: {output_file}")
    
    np.savez(
        output_file,
        X=X,
        y=y_numeric,
        target_names=np.array(CLASSES)
    )
    
    print(f"✅ Dataset salvo com sucesso!")
    print(f"📊 Tamanho do arquivo: {output_file.stat().st_size / (1024**2):.2f} MB")
    
    return X, y_numeric, CLASSES

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Pré-processar dataset de imagens")
    parser.add_argument(
        "--version",
        type=str,
        default="hog",
        choices=["hog", "flatten"],
        help="Versão de features: 'hog' ou 'flatten'"
    )
    
    args = parser.parse_args()
    
    process_dataset(version=args.version)

