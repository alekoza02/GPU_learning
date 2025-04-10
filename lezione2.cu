extern "C" __global__
void morph(const float* img, const float* ker, float* out, int H, int W, int KH, int KW) {
 
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    // does nothing if index exceeds the target position.
    if (i >= H - KH || j >= W - KW) return;

    float min_val = 1.0f / 0.0f; // Result is +infty

    for (int ki = 0; ki < KH; ++ki) {
        for (int kj = 0; kj < KW; ++kj) {
            float val = img[(i + ki) * W + (j + kj)] - ker[ki * KW + kj];
            if (val < min_val) min_val = val;
        }
    }

    out[i * (W - KW) + j] = min_val;
}