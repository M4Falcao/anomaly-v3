
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import numpy as np

def t2np(tensor):
    return tensor.cpu().data.numpy() if tensor is not None else None

def simulate():
    # Simulate parameters
    B = 1
    T = 64
    N = 16
    n_feat = 1536
    
    test_labels = []
    test_z = []
    
    print(f"Simulating N={N}, B={B}, T={T}")
    
    for i in range(N):
        # Simulate batch
        labels = torch.zeros(B) # shape (B,)
        z = torch.randn(B * T, n_feat) # shape (B*T, n_feat)
        
        test_labels.append(t2np(labels))
        test_z.append(z)
        
    # Reconstruct is_anomaly
    is_anomaly = np.array([0 if l == 0 else 1 for l in np.concatenate(test_labels)])
    print(f"is_anomaly shape: {is_anomaly.shape}")
    
    # Reconstruct anomaly_score
    z_grouped = torch.cat(test_z, dim=0).view(-1, T, n_feat)
    print(f"z_grouped shape: {z_grouped.shape}")
    
    anomaly_score = t2np(torch.mean(z_grouped ** 2, dim=(-2, -1)))
    print(f"anomaly_score shape: {anomaly_score.shape}")
    
    if len(is_anomaly) != len(anomaly_score):
        print("MISMATCH DETECTED!")
    else:
        print("Shapes match.")

if __name__ == "__main__":
    simulate()
