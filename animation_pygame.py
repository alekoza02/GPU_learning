import cupy as cp
import pygame
import numpy as np
import time

W, H = 800, 600

# === Compila il kernel CUDA ===
with open(f'lezione5.cu', 'r') as f:
    kernel_code = f.read()

mod = cp.RawModule(code=kernel_code)
shader = mod.get_function("shader")

# === Prepara la memoria ===
out_gpu = cp.zeros((H * W * 3,), dtype=cp.float32)
block = (16, 16)
grid = ((W + block[0] - 1) // block[0], (H + block[1] - 1) // block[1])

# === Inizializza Pygame ===
pygame.init()
screen = pygame.display.set_mode((W, H))
running = True
clock = pygame.time.Clock()

start_time = time.time()

while running:
    # Eventi
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    t = time.time() - start_time

    # Lancia il kernel
    mouse_x, mouse_y = pygame.mouse.get_pos()
    shader(grid, block, (out_gpu, np.int32(W), np.int32(H), np.float32(t), np.int32(mouse_x), np.int32(mouse_y)))

    # Converte a uint8 RGB per pygame
    frame = (cp.clip(out_gpu, 0, 1) * 255).astype(cp.uint8).reshape((H, W, 3))
    frame_cpu: np.ndarray = cp.asnumpy(frame)

    # Mostra con Pygame
    pygame.surfarray.blit_array(screen, frame_cpu.swapaxes(0, 1))
    pygame.display.flip()
    
    clock.tick()
    fps = clock.get_fps()  # Get current FPS as float
    pygame.display.set_caption(f"FPS: {fps:.2f}")  # Update window caption


pygame.quit()
