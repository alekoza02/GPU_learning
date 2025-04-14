extern "C" __global__
void morph(const float* __restrict__ img, const float* __restrict__ ker, float* __restrict__ out, int H, int W, int KH, int KW) {
 
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int j_valid = min(j, H - KH - 1); // Ensure within valid bounds
    int i_valid = min(i, W - KW - 1); // Ensure within valid bounds

    float min_val = 1.0f / 0.0f; // Result is +infty

    #pragma unroll
    for (int kj = 0; kj < KH; ++kj) {
        for (int ki = 0; ki < KW; ++ki) {
            float val = img[(j_valid + kj) * W + (i_valid + ki)] - ker[kj * KW + ki];
            min_val = fminf(min_val, val);
        }
    }

    out[j_valid * (W - KW) + i_valid] = min_val;
}