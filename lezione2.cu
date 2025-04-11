extern "C" __global__
void morph(const float* img, const float* ker, float* out, int H, int W, int KH, int KW) {
 
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    int i_valid = min(i, H - KH - 1); // Ensure within valid bounds
    int j_valid = min(j, W - KW - 1); // Ensure within valid bounds

    float min_val = 1.0f / 0.0f; // Result is +infty

    #pragma unroll
    for (int ki = 0; ki < KH; ++ki) {
        for (int kj = 0; kj < KW; ++kj) {
            float val = img[(i_valid + ki) * W + (j_valid + kj)] - ker[ki * KW + kj];
            min_val = fminf(min_val, val);
        }
    }

    out[i_valid * (W - KW) + j_valid] = min_val;
}