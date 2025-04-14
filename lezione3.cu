#define MAX_KERNEL_SIZE 64

extern "C" __global__ void memory(
    const float* __restrict__ img,
    const float* __restrict__ ker,
    float* __restrict__ out,
    float* __restrict__ times,
    unsigned int* __restrict__ debug,
    int H, int W, int KH, int KW
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    // Early return flag
    bool update = (i < H - KH && j < W - KW);

    float min_val = 1.0f / 0.0f;

    constexpr int TILE_KH = MAX_KERNEL_SIZE;
    constexpr int TILE_KW = MAX_KERNEL_SIZE;

    int local_thread_idx = threadIdx.y * blockDim.x + threadIdx.x;
    int threads_per_block = blockDim.x * blockDim.y;
    int total_tile_size = TILE_KH * TILE_KW;

    // Shared memory with +1 padding to avoid bank conflicts
    __shared__ float ker_tile[TILE_KH * (TILE_KW + 1)];

    int outW = W - KW;
    unsigned long long startClock = clock64();

    for (int tile_i = 0; tile_i < KH; tile_i += TILE_KH) {
        for (int tile_j = 0; tile_j < KW; tile_j += TILE_KW) {

            int tile_offset_i = tile_i;
            int tile_offset_j = tile_j;

            // Cooperative load of tile
            for (int idx = local_thread_idx; idx < total_tile_size; idx += threads_per_block) {
                int ki = idx / TILE_KW;
                int kj = idx % TILE_KW;

                int global_ki = tile_offset_i + ki;
                int global_kj = tile_offset_j + kj;

                float val = (global_ki < KH && global_kj < KW)
                    ? ker[global_ki * KW + global_kj] : 0.0f;

                ker_tile[ki * (TILE_KW + 1) + kj] = val;
            }

            __syncthreads();

            // Convolution using tile
            #pragma unroll
            for (int ki = 0; ki < TILE_KH; ++ki) {
                #pragma unroll
                for (int kj = 0; kj < TILE_KW; ++kj) {
                    int global_ki = tile_offset_i + ki;
                    int global_kj = tile_offset_j + kj;

                    if (global_ki < KH && global_kj < KW && i + global_ki < H && j + global_kj < W){
                        float diff = img[(i + global_ki) * W + (j + global_kj)] - ker_tile[ki * (TILE_KW + 1) + kj];
                        if (diff < min_val) min_val = diff;
                    }
                }
            }

            __syncthreads(); // Synchronize before loading next tile
        }
    }

    unsigned long long endClock = clock64();
    
    if (update) {
        int idx = i * outW + j;
        
        out[idx] = min_val;
        debug[idx] = local_thread_idx;
        
        float timeTaken = static_cast<float>(endClock - startClock);
        times[idx] = timeTaken;
    }

}
