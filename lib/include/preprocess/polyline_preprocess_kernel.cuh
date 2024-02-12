#ifndef PREPROCESS__POLYLINE_PREPROCESS_KERNEL_HPP_
#define PREPROCESS__POLYLINE_PREPROCESS_KERNEL_HPP_

__device__ void transform_polyline(
  const int B, const int K, const int P, const int D, const float * polylines,
  const float * center_xyz, const float * center_yaw, float * output)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < K * P) {
    const float x = polylines[idx * D];
    const float y = polylines[idx * D + 1];
    const float z = polylines[idx * D + 2];
    const float dx = polylines[idx * D + 3];
    const float dy = polylines[idx * D + 4];
    const float dz = polylines[idx * D + 5];
    const float type_id = polylines[idx * D + 6];

    for (int b = 0; b < B; ++b) {
      const float cos_val = std::cos(center_yaw[b]);
      const float sin_val = std::sin(center_yaw[b]);

      // transform
      const float trans_x = cos_val * x - sin_val * y - center_xyz[b * 3];
      const float trans_y = sin_val * x + cos_val * y - center_xyz[b * 3 + 1];
      const float trans_z = z;
      const float trans_dx = cos_val * dx - sin_val * dy;
      const float trans_dy = sin_val * dx + cos_val * dy;
      const float trans_dz = dz;

      const int trans_idx = (b * K * P + idx) * D;
      output[trans_idx] = trans_x;
      output[trans_idx + 1] = trans_y;
      output[trans_idx + 2] = trans_z;
      output[trans_idx + 3] = trans_dx;
      output[trans_idx + 4] = trans_dy;
      output[trans_idx + 5] = trans_dz;
      output[trans_idx + 6] = type_id;
    }
  }
}

#endif  // PREPROCESS__POLYLINE_PREPROCESS_KERNEL_HPP_