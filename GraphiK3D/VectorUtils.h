#pragma once
#include "cuda_runtime.h"

__host__ __device__ float3 operator*(float3 v, float a);
__host__ __device__ float3 operator/(float3 v, float a);
__host__ __device__ float3 operator-(float3 a, float3 b);
__host__ __device__ float3 operator+(float3 a, float3 b);
__host__ __device__ float dot(float3 a, float3 b);
__host__ __device__ float3 cross(float3 a, float3 b);
__host__ __device__ float length(float3 v);
__host__ __device__ float3 norm(float3 v);