import torch
import sys
import os

def main(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print(f"Erro: O arquivo não foi encontrado: {checkpoint_path}")
        return

    print(f"Lendo o checkpoint: {checkpoint_path}")
    # Carregar o checkpoint (no modo CPU para evitar problemas de memória na GPU)
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if isinstance(ckpt, dict):
        epoch_saved = ckpt.get('epoch', 'Desconhecido')
        print(f"Epoch final do checkpoint: {epoch_saved}")
        
        image_aurocs = ckpt.get('image_aurocs', [])
        pixel_aurocs = ckpt.get('pixel_aurocs', [])
        
        print("\n--- Histórico de AUROC por Epoch ---")
        max_len = max(len(image_aurocs), len(pixel_aurocs))
        if max_len == 0:
            print("Nenhuma métrica de AUROC armazenada nas listas do checkpoint.")
            return

        print(f"{'Epoch':<10} | {'Image AUROC':<15} | {'Pixel AUROC':<15}")
        print("-" * 45)
        for i in range(max_len):
            img_auc = image_aurocs[i] if i < len(image_aurocs) else 'N/A'
            pix_auc = pixel_aurocs[i] if i < len(pixel_aurocs) else 'N/A'
            
            img_str = f"{img_auc:.4f}" if isinstance(img_auc, (float, int)) else str(img_auc)
            pix_str = f"{pix_auc:.4f}" if isinstance(pix_auc, (float, int)) else str(pix_auc)
            
            print(f"{i+1:<10} | {img_str:<15} | {pix_str:<15}")
    else:
        print("O arquivo não está no formato de dicionário esperado.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = r"c:\Users\teo-s\Documents\GitHub\anomaly-v3\anomaly-detection-pesquisa\differnet\checkpoints\lightning-rod-suspension_se_differnet_lightning_rod_suspension_100_1_epoch_150.pt"
    main(path)
