extern "C" __global__
void shader(float* out, int W, int H, float t, int mouse_pos_x, int mouse_pos_y) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < W && j < H) {

        float aspect = float(W) / float(H);
        float ux = 2.0f * float(i) / float(W) - 1.0f;
        float uy = 2.0f * float(j) / float(H) - 1.0f;

        float m_p_x_n = 2.0f * float(mouse_pos_x) / float(W) - 1.0f;
        float m_p_y_n = 2.0f * float(mouse_pos_y) / float(H) - 1.0f;

        m_p_x_n *= aspect;
        ux *= aspect;

        float coord_x = ux - m_p_x_n;
        float coord_y = uy - m_p_y_n;

        float r = sqrtf(coord_x*coord_x + coord_y*coord_y);
        float val = sinf(20.0f * r - t);
        float g = 0.5f * (val + 1.0f);

        int idx = (j * W + i) * 3;
        out[idx+0] = (1.0f - g) * float(j)/float(H) + g * float(i)/float(W);  // R
        out[idx+1] = (1.0f - g) * float(i)/float(W) + g * float(j)/float(H);  // G
        out[idx+2] = g;  // B
    }
}