import cupy as cp

# Leggi il file contenente il kernel CUDA
with open('kernel.cu', 'r') as f:
    kernel_code = f.read()

# Compilare il kernel
from cupy import RawKernel
add_kernel = RawKernel(kernel_code, 'add_arrays')

# Creare gli array su GPU
n = 100
x = cp.random.rand(n, dtype=cp.float32)
y = cp.random.rand(n, dtype=cp.float32)
z = cp.zeros_like(x)

# Definire le dimensioni del blocco e della griglia
block_size = 32
grid_size = (n + block_size - 1) // block_size

# Eseguire il kernel
add_kernel((grid_size,), (block_size,), (x, y, z, n))

# Visualizzare il risultato
print(z.get())  # Spostiamo il risultato sulla CPU per visualizzarlo
