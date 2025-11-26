import torch

# Check if CUDA (GPU support) is available
cuda_available = torch.cuda.is_available()

if cuda_available:
    print("CUDA is available. GPU support is enabled.")
else:
    print("CUDA is not available. Using CPU.")

print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
