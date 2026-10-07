"""Script para treinar o SEDifferNet em todas as classes do INSPLAD.

Utiliza o script padrão main.py e atualiza as configurações programaticamente.
"""

import sys
import os

# Adiciona o diretório atual ao path para garantir que importações do differnet funcionem
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import config as c
from core.eval_pipeline import ALL_CLASSES
from main import main

def train_all_classes():
    print("=" * 60)
    print("  TREINAMENTO EM LOTE - SEDIFFERNET (BACKBONE)")
    print("=" * 60)
    print(f"  Classes ({len(ALL_CLASSES)}): {ALL_CLASSES}")
    print("=" * 60)
    
    for idx, cls in enumerate(ALL_CLASSES, 1):
        print(f"\n[{idx}/{len(ALL_CLASSES)}] Iniciando treinamento para a classe: {cls}")
        
        # Atualiza a configuração para a classe atual antes de chamar o main
        c.class_name = cls
        c.mlflow_run_name = f"{cls}_{c.modelname}"
        
        try:
            main()
            print(f"[{idx}/{len(ALL_CLASSES)}] Treinamento concluído para a classe: {cls}")
        except KeyboardInterrupt:
            print(f"\nTreinamento interrompido pelo usuário na classe {cls}.")
            break
        except Exception as e:
            print(f"\nErro ao treinar a classe {cls}: {e}")
            continue

if __name__ == '__main__':
    train_all_classes()
