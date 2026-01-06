"""
Interface Gradio para Classificação de Imagens de Cômodos
"""
import gradio as gr
import numpy as np
from PIL import Image
from pathlib import Path
import joblib
import sys

# Adicionar path
sys.path.insert(0, str(Path(__file__).parent))

from datasets import fetch_airbnb_mnist
# Importar funções de pré-processamento
from skimage import feature
from skimage.color import rgb2gray

def preprocess_image(img, size=64, grayscale=True):
    """Pré-processa uma imagem"""
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    img_array = np.array(img)
    if grayscale and len(img_array.shape) == 3:
        img_array = rgb2gray(img_array)
    return img_array

def extract_features_hog(img_array, orientations=9, pixels_per_cell=(8, 8), 
                         cells_per_block=(2, 2)):
    """Extrai features usando HOG"""
    return feature.hog(
        img_array, orientations=orientations,
        pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block,
        visualize=False, channel_axis=None
    )

# Configurações
MODELS_DIR = Path(__file__).parent / "models"
VERSION = "hog"

# Classes
CLASSES = ["bedroom", "kitchen", "bathroom", "livingroom"]

# Carregar modelo (usar o melhor modelo)
def load_best_model():
    """Carrega o melhor modelo disponível"""
    # Tentar carregar GradientBoosting primeiro (geralmente melhor)
    model_paths = [
        MODELS_DIR / f"gradientboosting_{VERSION}.pkl",
        MODELS_DIR / f"randomforest_{VERSION}.pkl",
        MODELS_DIR / f"linearsvm_{VERSION}.pkl",
        MODELS_DIR / f"logisticregression_{VERSION}.pkl"
    ]
    
    for model_path in model_paths:
        if model_path.exists():
            print(f"📥 Carregando modelo: {model_path.name}")
            return joblib.load(model_path)
    
    raise FileNotFoundError("Nenhum modelo encontrado! Execute primeiro: python scripts/03_train.py")

# Carregar modelo
print("🔍 Carregando modelo...")
try:
    model = load_best_model()
    print("✅ Modelo carregado com sucesso!")
except Exception as e:
    print(f"❌ Erro ao carregar modelo: {e}")
    model = None

def classify_image(image):
    """Classifica uma imagem"""
    if model is None:
        return "❌ Erro: Modelo não carregado. Execute primeiro: python scripts/03_train.py"
    
    if image is None:
        return "⚠️ Por favor, faça upload de uma imagem."
    
    try:
        # Converter para PIL Image se necessário
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Pré-processar
        img_array = preprocess_image(image, size=64, grayscale=True)
        
        # Extrair features HOG
        features = extract_features_hog(img_array)
        
        # Fazer predição
        prediction = model.predict([features])[0]
        probabilities = model.predict_proba([features])[0]
        
        # Resultado
        predicted_class = CLASSES[prediction]
        confidence = probabilities[prediction] * 100
        
        # Formatar resultado
        result = f"""
## Resultado da Classificação

**Classe Prevista:** {predicted_class.upper()}
**Confiança:** {confidence:.2f}%

---

### Probabilidades por Classe:

"""
        for i, (class_name, prob) in enumerate(zip(CLASSES, probabilities)):
            bar_length = int(prob * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            result += f"**{class_name.capitalize()}:** {prob*100:.2f}% {bar}\n"
        
        result += """
---
**Nota:** Este modelo foi treinado com imagens de interiores do MIT LabelMe Dataset.
A precisão pode variar dependendo da qualidade e características da imagem enviada.
"""
        
        return result
        
    except Exception as e:
        return f"❌ Erro ao classificar imagem: {str(e)}"

# Criar interface Gradio
with gr.Blocks(title="Classificação de Imagens - Tipos de Cômodos") as demo:
    gr.Markdown("""
    # Classificação de Imagens - Tipos de Cômodos
    
    Esta aplicação classifica imagens de interiores em 4 categorias:
    - **Bedroom** (Quarto)
    - **Kitchen** (Cozinha)
    - **Bathroom** (Banheiro)
    - **Livingroom** (Sala de Estar)
    
    **Como usar:**
    1. Faça upload de uma imagem de um cômodo
    2. Clique em "Classificar Imagem"
    3. Veja a predição e as probabilidades para cada classe
    
    **Modelo:** Gradient Boosting com features HOG (Histogram of Oriented Gradients)
    """)
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                type="pil",
                label="Imagem do Cômodo",
                height=400
            )
            btn_classify = gr.Button("Classificar Imagem", variant="primary", size="lg")
        
        with gr.Column():
            output = gr.Markdown(label="Resultado")
    
    btn_classify.click(
        fn=classify_image,
        inputs=image_input,
        outputs=output
    )
    
    gr.Markdown("""
    ---
    ## Sobre o Modelo
    
    - **Dataset**: MIT LabelMe Indoor Scene Dataset
    - **Features**: HOG (Histogram of Oriented Gradients)
    - **Algoritmo**: Gradient Boosting Classifier
    - **Classes**: bedroom, kitchen, bathroom, livingroom
    
    ## Exemplos
    
    Tente fazer upload de imagens de:
    - Quartos com cama
    - Cozinhas com fogão/geladeira
    - Banheiros com pia/chuveiro
    - Salas com sofá/TV
    """)

if __name__ == "__main__":
    demo.launch()

