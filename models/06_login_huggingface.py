"""
Login no Hugging Face
Faz login no Hugging Face Hub para permitir upload de modelos
"""
from huggingface_hub import login
import os
import sys

def fazer_login(token=None):
    """
    Faz login no Hugging Face
    
    Args:
        token: Token do Hugging Face (opcional). Se não fornecido, será solicitado interativamente.
    """
    print("=" * 70)
    print("LOGIN NO HUGGING FACE")
    print("=" * 70)
    
    if token:
        print("\nUsando token fornecido...")
        login(token=token)
        print("✓ Login realizado com sucesso!")
    else:
        print("\n📝 INSTRUÇÕES:")
        print("1. Obtenha um token de acesso em: https://huggingface.co/settings/tokens")
        print("2. Crie um token com permissões de 'write'")
        print("3. Cole o token quando solicitado abaixo")
        print("\n" + "=" * 70)
        print("Iniciando login interativo...")
        print("=" * 70)
        
        try:
            login()
            print("\n✓ Login realizado com sucesso!")
        except KeyboardInterrupt:
            print("\n\n⚠️  Login cancelado pelo usuário")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ Erro no login: {e}")
            print("\nAlternativa: forneça o token como argumento:")
            print("  python3 models/06_login_huggingface.py --token SEU_TOKEN_AQUI")
            return False
    
    token_path = os.path.expanduser('~/.huggingface/token')
    if os.path.exists(token_path):
        print(f"\n✓ Token salvo em: {token_path}")
    
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Login no Hugging Face')
    parser.add_argument('--token', type=str, help='Token do Hugging Face (opcional)')
    args = parser.parse_args()
    
    fazer_login(token=args.token)

