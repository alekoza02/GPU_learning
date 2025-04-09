extern "C" __global__
void add_arrays(const float *x, const float *y, float *z, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        z[idx] = x[idx] + y[idx];
    }
}
