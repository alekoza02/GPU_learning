extern "C" __global__ 
void shader(float* __restrict__ out, int W, int H) {
    
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < W && i < H) {

        float aspect = float(W) / float(H);

        float ux = 2.0f * float(j) / float(W) - 1.0f;
        float uy = 2.0f * float(i) / float(H) - 1.0f;

        ux *= aspect;  // correct aspect ratio

        float r = sqrtf(ux * ux + uy * uy);
        float frequency = 20.0f;

        float value = sinf(frequency * r);         // range: [-1, 1]
        float grayscale = 0.5f * (value + 1.0f);   // range: [0, 1]

        int idx = (j * H + i) * 3;

        out[idx + 0] = (grayscale) * float(j) / float(W) + (1 - grayscale) * float(i) / float(H);  // R
        out[idx + 1] = (grayscale) * float(i) / float(H) + (1 - grayscale) * float(j) / float(W);  // G
        out[idx + 2] = grayscale;  // B
    }
}
