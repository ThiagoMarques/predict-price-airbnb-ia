"""
Script para download do dataset MIT LabelMe Indoor Scene
"""
import os
import urllib.request
import ssl
import tarfile
from pathlib import Path

# Tentar usar requests como alternativa (mais robusto)
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# Configurações
DATASET_URL = "http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar"
DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
TAR_FILE = RAW_DIR / "indoorCVPR_09.tar"

def download_dataset():
    """Baixa o dataset do MIT LabelMe"""
    
    # Criar diretórios
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    # Verificar se já foi baixado
    if TAR_FILE.exists():
        print(f"✅ Dataset já existe em: {TAR_FILE}")
        return TAR_FILE
    
    print(f"📥 Baixando dataset de: {DATASET_URL}")
    print(f"📁 Salvando em: {TAR_FILE}")
    print("⏳ Isso pode demorar alguns minutos...")
    
    try:
        # Tentar usar requests primeiro (mais robusto com SSL)
        if HAS_REQUESTS:
            print("🔄 Usando requests para download...")
            response = requests.get(DATASET_URL, stream=True, verify=False, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(TAR_FILE, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\r📊 Progresso: {percent:.1f}% ({downloaded / (1024**2):.1f} MB)", end='', flush=True)
            
            print()  # Nova linha após progresso
            print(f"✅ Download concluído: {TAR_FILE}")
            print(f"📊 Tamanho: {TAR_FILE.stat().st_size / (1024**2):.2f} MB")
            return TAR_FILE
        else:
            # Fallback para urllib com tratamento SSL
            print("🔄 Usando urllib para download...")
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            opener = urllib.request.build_opener(
                urllib.request.HTTPSHandler(context=ssl_context)
            )
            urllib.request.install_opener(opener)
            
            urllib.request.urlretrieve(DATASET_URL, TAR_FILE)
            print(f"✅ Download concluído: {TAR_FILE}")
            print(f"📊 Tamanho: {TAR_FILE.stat().st_size / (1024**2):.2f} MB")
            return TAR_FILE
            
    except Exception as e:
        print(f"\n❌ Erro ao baixar: {e}")
        print("\n💡 Dicas para resolver:")
        print("   1. Verificar conexão com internet")
        print("   2. Instalar requests: pip install requests")
        print("   3. Baixar manualmente:")
        print(f"      URL: {DATASET_URL}")
        print(f"      Salvar em: {TAR_FILE}")
        print("   4. Ou usar wget/curl:")
        print(f"      wget {DATASET_URL} -O {TAR_FILE}")
        raise

def extract_dataset(tar_file=None):
    """Extrai o arquivo tar"""
    if tar_file is None:
        tar_file = TAR_FILE
    
    if not tar_file.exists():
        print(f"❌ Arquivo não encontrado: {tar_file}")
        return None
    
    extract_dir = RAW_DIR / "indoorCVPR_09"
    
    # Verificar se já foi extraído
    if extract_dir.exists():
        print(f"✅ Dataset já extraído em: {extract_dir}")
        return extract_dir
    
    print(f"📦 Extraindo arquivo: {tar_file}")
    print(f"📁 Destino: {extract_dir}")
    
    try:
        with tarfile.open(tar_file, 'r') as tar:
            tar.extractall(path=RAW_DIR)
        print(f"✅ Extração concluída: {extract_dir}")
        return extract_dir
    except Exception as e:
        print(f"❌ Erro ao extrair: {e}")
        raise

if __name__ == "__main__":
    print("=" * 70)
    print("DOWNLOAD DO DATASET MIT LABELME INDOOR SCENE")
    print("=" * 70)
    
    # Baixar
    tar_file = download_dataset()
    
    # Extrair
    extract_dir = extract_dataset(tar_file)
    
    print("\n" + "=" * 70)
    print("✅ PROCESSO CONCLUÍDO")
    print("=" * 70)
    print(f"📁 Dados brutos em: {extract_dir}")

