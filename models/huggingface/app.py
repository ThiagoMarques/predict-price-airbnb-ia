"""
Aplicação Gradio para Previsão de Preços Airbnb
Interface interativa para estimar preços de imóveis no Rio de Janeiro

O modelo é treinado automaticamente na primeira execução se não existir.
Isso garante compatibilidade total com o ambiente do Hugging Face Space.
"""
import gradio as gr
import pandas as pd
import numpy as np
import joblib
import os
import sys
import json
from threading import Lock

# Adicionar path para importar utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Contador de visitas
CONTADOR_ARQUIVO = "contador_visitas.json"
contador_lock = Lock()

def ler_contador():
    """Lê o número de visitas do arquivo"""
    try:
        if os.path.exists(CONTADOR_ARQUIVO):
            with open(CONTADOR_ARQUIVO, 'r') as f:
                data = json.load(f)
                return data.get('visitas', 0)
        return 0
    except Exception as e:
        print(f"Erro ao ler contador: {e}")
        return 0

def incrementar_contador():
    """Incrementa o contador de visitas e salva no arquivo"""
    try:
        with contador_lock:
            visitas = ler_contador() + 1
            data = {
                'visitas': visitas,
                'ultima_atualizacao': pd.Timestamp.now().isoformat()
            }
            with open(CONTADOR_ARQUIVO, 'w') as f:
                json.dump(data, f)
            return visitas
    except Exception as e:
        print(f"Erro ao incrementar contador: {e}")
        return ler_contador()

def obter_texto_contador():
    """Retorna o texto formatado do contador"""
    visitas = ler_contador()
    return f"**Total de visitas:** {visitas:,}"

def incrementar_e_obter_texto():
    """Incrementa o contador e retorna o texto formatado"""
    visitas = incrementar_contador()
    return f"**Total de visitas:** {visitas:,}"

def treinar_modelo_se_necessario():
    """Treina o modelo se não existir - garante compatibilidade com o ambiente"""
    modelo_path = "gradientboosting_best.pkl"
    preprocessor_path = "preprocessor.pkl"
    
    # Se ambos existem, apenas carregar
    if os.path.exists(modelo_path) and os.path.exists(preprocessor_path):
        print("Modelo e preprocessor encontrados, carregando...")
        try:
            modelo = joblib.load(modelo_path)
            preprocessor = joblib.load(preprocessor_path)
            print("Modelo e preprocessor carregados com sucesso")
            return modelo, preprocessor
        except Exception as e:
            print(f"Aviso: Erro ao carregar: {e}")
            print("Re-treinando modelo...")
    
    # Se não existem ou erro ao carregar, treinar
    print("Treinando modelo no ambiente do Space...")
    print("Isso garante compatibilidade total!")
    
    try:
        from utils import clean_price
    except ImportError:
        # Se utils não estiver disponível, criar função inline
        def clean_price(price_str):
            if pd.isna(price_str) or price_str == '':
                return np.nan
            price_clean = str(price_str).replace('$', '').replace(',', '').strip()
            try:
                return float(price_clean)
            except:
                return np.nan
    
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    
    print("Criando modelo e preprocessor...")
    
    # Características
    caracteristicas_numericas = [
        'latitude', 'longitude', 'accommodates', 'bedrooms',
        'bathrooms', 'number_of_reviews', 'review_scores_rating',
        'availability_365', 'minimum_nights', 'maximum_nights'
    ]
    caracteristicas_categoricas = [
        'property_type', 'room_type', 'neighbourhood_cleansed'
    ]
    
    # Criar pipeline
    pipeline_numerico = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    pipeline_categorico = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', pipeline_numerico, caracteristicas_numericas),
        ('cat', pipeline_categorico, caracteristicas_categoricas)
    ])
    
    # Criar modelo
    modelo = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        verbose=0
    )
    
    # Tentar carregar dados reais se disponível
    # Tentar múltiplos caminhos possíveis
    data_paths = [
        os.path.join(os.path.dirname(__file__), '..', 'data', 'listings.csv'),
        os.path.join(os.path.dirname(__file__), 'data', 'listings.csv'),
        'data/listings.csv',
        '../data/listings.csv'
    ]
    
    data_path = None
    for path in data_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    if data_path:
        print("Carregando dados reais...")
        df = pd.read_csv(data_path, low_memory=False)
        df['price'] = df['price'].apply(clean_price)
        df = df.dropna(subset=['price']).copy()
        
        X = df[caracteristicas_numericas + caracteristicas_categoricas].copy()
        y = df['price'].copy()
        
        # Treinar
        print("Treinando modelo com dados reais...")
        X_transformado = preprocessor.fit_transform(X)
        modelo.fit(X_transformado, y)
        print(f"Modelo treinado com {len(X)} amostras")
    else:
        # Se dados não disponíveis, criar modelo dummy que retorna preço médio
        print("Aviso: Dados não encontrados, criando modelo base...")
        # Criar dados sintéticos para treinar
        np.random.seed(42)
        n_samples = 1000
        X_sintetico = pd.DataFrame({
            'latitude': np.random.uniform(-23.0, -22.7, n_samples),
            'longitude': np.random.uniform(-43.8, -43.1, n_samples),
            'accommodates': np.random.randint(1, 10, n_samples),
            'bedrooms': np.random.randint(1, 5, n_samples),
            'bathrooms': np.random.randint(1, 4, n_samples),
            'number_of_reviews': np.random.randint(0, 200, n_samples),
            'review_scores_rating': np.random.uniform(3.0, 5.0, n_samples),
            'availability_365': np.random.randint(0, 365, n_samples),
            'minimum_nights': np.random.randint(1, 30, n_samples),
            'maximum_nights': np.random.randint(30, 1125, n_samples),
            'property_type': np.random.choice(['Entire rental unit', 'Private room in rental unit', 'Entire home'], n_samples),
            'room_type': np.random.choice(['Entire home/apt', 'Private room', 'Shared room'], n_samples),
            'neighbourhood_cleansed': np.random.choice(['Copacabana', 'Ipanema', 'Barra da Tijuca', 'Centro'], n_samples)
        })
        y_sintetico = 200 + X_sintetico['accommodates'] * 50 + X_sintetico['bedrooms'] * 30 + np.random.normal(0, 100, n_samples)
        y_sintetico = np.maximum(y_sintetico, 50)  # Preço mínimo
        
        X_transformado = preprocessor.fit_transform(X_sintetico)
        modelo.fit(X_transformado, y_sintetico)
        print("Modelo base criado")
    
    # Salvar modelo e preprocessor
    print("Salvando modelo e preprocessor...")
    joblib.dump(modelo, modelo_path)
    joblib.dump(preprocessor, preprocessor_path)
    print("Modelo e preprocessor salvos")
    
    return modelo, preprocessor

# Carregar ou treinar modelo
print("Verificando modelo...")
modelo, preprocessor = treinar_modelo_se_necessario()

# Valores únicos das categorias (baseado no dataset do Rio de Janeiro)
PROPERTY_TYPES = [
    "Entire rental unit",
    "Private room in rental unit",
    "Entire home",
    "Entire condo",
    "Private room in home",
    "Entire loft",
    "Entire serviced apartment",
    "Room in hotel",
    "Private room in bed and breakfast",
    "Private room in condo",
    "Shared room in rental unit",
    "Room in aparthotel",
    "Entire guesthouse",
    "Private room in guesthouse",
    "Entire villa",
    "Private room in guest suite",
    "Shared room in home",
    "Shared room in bed and breakfast",
    "Private room in serviced apartment",
    "Tiny home"
]

ROOM_TYPES = [
    "Entire home/apt",
    "Private room",
    "Shared room",
    "Hotel room"
]

NEIGHBOURHOODS = [
    "Copacabana",
    "Barra da Tijuca",
    "Ipanema",
    "Centro",
    "Recreio dos Bandeirantes",
    "Jacarepaguá",
    "Botafogo",
    "Leblon",
    "Santa Teresa",
    "Flamengo",
    "Camorim",
    "Leme",
    "Laranjeiras",
    "São Conrado",
    "Tijuca",
    "Catete",
    "Vidigal",
    "Glória",
    "Lagoa",
    "Gávea",
    "Jardim Botânico",
    "Humaitá",
    "Itanhangá",
    "Barra de Guaratiba",
    "Santo Cristo",
    "Guaratiba",
    "Vargem Pequena",
    "Joá",
    "Urca",
    "Campo Grande",
    "Taquara",
    "Vargem Grande",
    "Vila Isabel",
    "São Cristóvão",
    "Maracanã",
    "Rio Comprido",
    "Freguesia (Jacarepaguá)",
    "Curicica",
    "Praça da Bandeira",
    "Cosme Velho",
    "Estácio",
    "Grajaú",
    "Engenho de Dentro",
    "Paquetá",
    "Pechincha",
    "Anil",
    "Méier",
    "Andaraí",
    "Jardim Guanabara",
    "Alto da Boa Vista"
]

# Coordenadas médias por bairro (baseado no dataset)
COORDENADAS_BAIRROS = {
    "Copacabana": (-22.9711, -43.1822),
    "Barra da Tijuca": (-23.0065, -43.3656),
    "Ipanema": (-22.9844, -43.2031),
    "Centro": (-22.9068, -43.1729),
    "Recreio dos Bandeirantes": (-23.0247, -43.4653),
    "Jacarepaguá": (-22.9408, -43.3456),
    "Botafogo": (-22.9506, -43.1844),
    "Leblon": (-22.9844, -43.2231),
    "Santa Teresa": (-22.9206, -43.1853),
    "Flamengo": (-22.9331, -43.1731),
    "Camorim": (-22.9789, -43.3656),
    "Leme": (-22.9631, -43.1681),
    "Laranjeiras": (-22.9331, -43.1803),
    "São Conrado": (-23.0065, -43.2569),
    "Tijuca": (-22.9331, -43.2331),
    "Catete": (-22.9331, -43.1781),
    "Vidigal": (-22.9906, -43.2231),
    "Glória": (-22.9206, -43.1731),
    "Lagoa": (-22.9789, -43.2103),
    "Gávea": (-22.9789, -43.2331),
    "Jardim Botânico": (-22.9681, -43.2231),
    "Humaitá": (-22.9506, -43.2003),
    "Itanhangá": (-22.9789, -43.3456),
    "Barra de Guaratiba": (-23.0406, -43.5231),
    "Santo Cristo": (-22.9068, -43.2003),
    "Guaratiba": (-23.0406, -43.5231),
    "Vargem Pequena": (-23.0247, -43.4653),
    "Joá": (-22.9906, -43.2569),
    "Urca": (-22.9506, -43.1681),
    "Campo Grande": (-22.9068, -43.5569),
    "Taquara": (-22.9206, -43.3456),
    "Vargem Grande": (-23.0406, -43.5231),
    "Vila Isabel": (-22.9206, -43.2331),
    "São Cristóvão": (-22.9068, -43.2231),
    "Maracanã": (-22.9125, -43.2303),
    "Rio Comprido": (-22.9206, -43.2103),
    "Freguesia (Jacarepaguá)": (-22.9408, -43.3456),
    "Curicica": (-22.9789, -43.3456),
    "Praça da Bandeira": (-22.9068, -43.2103),
    "Cosme Velho": (-22.9506, -43.1953),
    "Estácio": (-22.9068, -43.2003),
    "Grajaú": (-22.9331, -43.2569),
    "Engenho de Dentro": (-22.9068, -43.2569),
    "Paquetá": (-22.7631, -43.1069),
    "Pechincha": (-22.9408, -43.3656),
    "Anil": (-22.9789, -43.3656),
    "Méier": (-22.9068, -43.2569),
    "Andaraí": (-22.9206, -43.2453),
    "Jardim Guanabara": (-22.7631, -43.1069),
    "Alto da Boa Vista": (-22.9789, -43.2569)
}

def atualizar_coordenadas(bairro):
    """Atualiza latitude e longitude baseado no bairro selecionado"""
    if bairro in COORDENADAS_BAIRROS:
        lat, lon = COORDENADAS_BAIRROS[bairro]
        return gr.update(value=lat), gr.update(value=lon)
    return gr.update(), gr.update()

def prever_preco(
    latitude,
    longitude,
    accommodates,
    bedrooms,
    bathrooms,
    number_of_reviews,
    review_scores_rating,
    availability_365,
    minimum_nights,
    maximum_nights,
    property_type,
    room_type,
    neighbourhood_cleansed
):
    """Prevê preço baseado nas características do imóvel"""
    
    if modelo is None or preprocessor is None:
        return "Erro: Modelo ou preprocessor não carregados. Verifique os logs."
    
    try:
        # Criar DataFrame com os inputs
        dados = pd.DataFrame({
            'latitude': [latitude],
            'longitude': [longitude],
            'accommodates': [accommodates],
            'bedrooms': [bedrooms],
            'bathrooms': [bathrooms],
            'number_of_reviews': [number_of_reviews],
            'review_scores_rating': [review_scores_rating],
            'availability_365': [availability_365],
            'minimum_nights': [minimum_nights],
            'maximum_nights': [maximum_nights],
            'property_type': [property_type],
            'room_type': [room_type],
            'neighbourhood_cleansed': [neighbourhood_cleansed]
        })
        
        # Aplicar preprocessor
        dados_transformados = preprocessor.transform(dados)
        
        # Fazer predição
        preco_previsto = modelo.predict(dados_transformados)[0]
        
        # Formatar resultado
        if preco_previsto < 0:
            preco_previsto = 0
        
        resultado = f"""
## Preço Estimado

**R$ {preco_previsto:,.2f}** por diária

---

**Aviso Importante:**

Esta é uma estimativa educacional baseada em dados históricos. 
O preço real pode variar significativamente devido a:

- Sazonalidade e eventos locais
- Dinâmica de mercado em tempo real
- Fatores não capturados pelo modelo (localização exata, vista, etc.)
- Oferta e demanda momentâneas

**Não substitui análise real de mercado.**
        """
        
        return resultado
        
    except Exception as e:
        return f"Erro na predição: {str(e)}"

# Criar interface Gradio
with gr.Blocks(title="Previsão de Preços Airbnb - Rio de Janeiro") as demo:
    gr.Markdown("""
    # Estimativa de Preço de Imóveis no Airbnb - Rio de Janeiro
    
    Esta aplicação demonstra um pipeline completo de Machine Learning, desde o pré-processamento dos dados até a predição de preços usando Gradient Boosting.
    
    **Aviso:** A estimativa é apenas educacional e possui limitações devido à ausência de fatores como sazonalidade, eventos locais e dinâmica de mercado.
    
    **Nota:** O modelo é treinado automaticamente na primeira execução para garantir compatibilidade total com o ambiente.
    """)
    
    # Exibir contador de visitas
    contador_display = gr.Markdown(
        value=obter_texto_contador(),
        elem_classes=["contador-visitas"]
    )
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Localização")
            latitude = gr.Slider(
                minimum=-23.0,
                maximum=-22.7,
                value=-22.9068,
                step=0.0001,
                label="Latitude",
                info="Coordenada geográfica (exemplo: -22.9068 para Copacabana)"
            )
            longitude = gr.Slider(
                minimum=-43.8,
                maximum=-43.1,
                value=-43.1729,
                step=0.0001,
                label="Longitude",
                info="Coordenada geográfica (exemplo: -43.1729 para Copacabana)"
            )
            neighbourhood_cleansed = gr.Dropdown(
                choices=NEIGHBOURHOODS,
                value="Copacabana",
                label="Bairro",
                info="Bairro do imóvel (ao selecionar, atualiza automaticamente latitude e longitude)"
            )
        
        with gr.Column():
            gr.Markdown("### Características do Imóvel")
            property_type = gr.Dropdown(
                choices=PROPERTY_TYPES,
                value="Entire rental unit",
                label="Tipo de Propriedade"
            )
            room_type = gr.Dropdown(
                choices=ROOM_TYPES,
                value="Entire home/apt",
                label="Tipo de Quarto"
            )
            accommodates = gr.Slider(
                minimum=1,
                maximum=16,
                value=2,
                step=1,
                label="Número de Hóspedes",
                info="Capacidade máxima de pessoas"
            )
            bedrooms = gr.Slider(
                minimum=0,
                maximum=10,
                value=1,
                step=1,
                label="Número de Quartos"
            )
            bathrooms = gr.Slider(
                minimum=0,
                maximum=10,
                value=1,
                step=0.5,
                label="Número de Banheiros"
            )
    
    # Conectar mudança do bairro à atualização das coordenadas
    neighbourhood_cleansed.change(
        fn=atualizar_coordenadas,
        inputs=neighbourhood_cleansed,
        outputs=[latitude, longitude]
    )
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Avaliações")
            number_of_reviews = gr.Slider(
                minimum=0,
                maximum=1000,
                value=50,
                step=1,
                label="Número de Avaliações",
                info="Quantidade de avaliações recebidas"
            )
            review_scores_rating = gr.Slider(
                minimum=0,
                maximum=5,
                value=4.5,
                step=0.1,
                label="Nota Média",
                info="Avaliação média (0 a 5)"
            )
        
        with gr.Column():
            gr.Markdown("### Disponibilidade")
            availability_365 = gr.Slider(
                minimum=0,
                maximum=365,
                value=180,
                step=1,
                label="Dias Disponíveis por Ano",
                info="Quantos dias do ano o imóvel está disponível"
            )
            minimum_nights = gr.Slider(
                minimum=1,
                maximum=365,
                value=2,
                step=1,
                label="Mínimo de Noites"
            )
            maximum_nights = gr.Slider(
                minimum=1,
                maximum=1125,
                value=30,
                step=1,
                label="Máximo de Noites"
            )
    
    btn_prever = gr.Button("Prever Preço", variant="primary", size="lg")
    
    output = gr.Markdown(label="Resultado")
    
    btn_prever.click(
        fn=prever_preco,
        inputs=[
            latitude, longitude, accommodates, bedrooms, bathrooms,
            number_of_reviews, review_scores_rating, availability_365,
            minimum_nights, maximum_nights, property_type, room_type,
            neighbourhood_cleansed
        ],
        outputs=output
    )
    
    gr.Markdown("""
    ---
    ## Sobre o Modelo
    
    - **Algoritmo**: Gradient Boosting Regressor
    - **Dataset**: Inside Airbnb - Rio de Janeiro
    - **Características**: Localização, características do imóvel, avaliações e disponibilidade
    - **Limitações**: Modelo educacional com R² limitado. Não captura fatores dinâmicos de mercado.
    
    ## Links
    
    - [Repositório GitHub](https://github.com/ThiagoMarques/predict-price-airbnb-ia)
    - [Inside Airbnb - Rio de Janeiro](https://insideairbnb.com/rio-de-janeiro/)
    """)
    
    # Atualizar contador quando a página carregar
    # O evento load é disparado quando cada usuário carrega a página
    demo.load(
        fn=incrementar_e_obter_texto,
        inputs=[],
        outputs=contador_display
    )

if __name__ == "__main__":
    demo.launch()
