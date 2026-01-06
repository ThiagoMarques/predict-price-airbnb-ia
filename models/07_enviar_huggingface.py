"""
Enviar Modelo para Hugging Face
Faz upload do modelo preparado para o Hugging Face Model Hub
"""
from huggingface_hub import HfApi
import os
from pathlib import Path

def enviar_modelo(repo_id=None, repo_type='space'):
    """
    Envia modelo para o Hugging Face Hub
    
    Args:
        repo_id: ID do repositório (ex: 'ThiagoMarques/predict-price-airbnb-ia')
        repo_type: Tipo de repositório ('space' ou 'model')
    """
    print("=" * 70)
    print("ENVIAR PARA HUGGING FACE")
    print("=" * 70)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    hf_dir = Path(os.path.join(project_root, 'models', 'huggingface'))
    
    if not hf_dir.exists():
        print("❌ Diretório models/huggingface não encontrado.")
        print("   Execute primeiro: python3 models/05_preparar_huggingface.py")
        return
    
    # Verificar arquivos essenciais
    arquivos_essenciais = ['app.py', 'README.md', 'requirements.txt']
    arquivos_faltando = [f for f in arquivos_essenciais if not (hf_dir / f).exists()]
    
    if arquivos_faltando:
        print(f"❌ Arquivos faltando: {', '.join(arquivos_faltando)}")
        print("   Execute primeiro: python3 models/05_preparar_huggingface.py")
        return
    
    print(f"\nArquivos encontrados em models/huggingface/:")
    for arquivo in sorted(hf_dir.glob('*')):
        if arquivo.is_file() and arquivo.name != '.gitkeep':
            print(f"  ✓ {arquivo.name}")
    
    # Verificar modelos
    models_dir = hf_dir / 'models' / 'saved'
    if models_dir.exists():
        modelos = list(models_dir.glob('*.pkl'))
        if modelos:
            print(f"\nModelos encontrados:")
            for modelo in modelos:
                print(f"  ✓ {modelo.name}")
    
    if repo_id is None:
        repo_id = 'ThiagoMarques/predict-price-airbnb-ia'
        print(f"\n⚠️  Usando repo_id padrão: {repo_id}")
        print("   Para usar outro, forneça: --repo-id seu-usuario/nome-repositorio")
    
    print(f"\n📤 Enviando para: {repo_id} (tipo: {repo_type})")
    print("   Isso pode demorar alguns minutos...")
    
    try:
        api = HfApi()
        api.upload_folder(
            folder_path=str(hf_dir),
            repo_id=repo_id,
            repo_type=repo_type
        )
        print(f"\n✓ Arquivos enviados com sucesso!")
        if repo_type == 'space':
            print(f"   Acesse em: https://huggingface.co/spaces/{repo_id}")
        else:
            print(f"   Acesse em: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"\n❌ Erro ao enviar: {e}")
        print("\nVerifique:")
        print("1. Se você fez login (python3 models/06_login_huggingface.py)")
        print("2. Se o repo_id está correto")
        print("3. Se você tem permissão para criar repositórios no Hugging Face")
        print("4. Se o Space foi criado em: https://huggingface.co/new-space")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Enviar modelo para Hugging Face')
    parser.add_argument('--repo-id', type=str, 
                       default='ThiagoMarques/predict-price-airbnb-ia',
                       help='ID do repositório no Hugging Face')
    parser.add_argument('--repo-type', type=str, 
                       default='space',
                       choices=['space', 'model'],
                       help='Tipo de repositório (space ou model)')
    args = parser.parse_args()
    
    enviar_modelo(repo_id=args.repo_id, repo_type=args.repo_type)

