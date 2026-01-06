"""
Funções utilitárias para pré-processamento de imagens
(Reutilizáveis para predição em tempo real)
"""
import numpy as np
from PIL import Image
from skimage import feature
from skimage.color import rgb2gray

def preprocess_image(img, size=64, grayscale=True):
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

def extract_features_flatten(img_array):
    """Extrai features usando flatten (baseline)"""
    return img_array.flatten()

