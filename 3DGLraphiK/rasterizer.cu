#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "rasterizer.h"

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>

#define VERTEX_SHADER_BLOCK_SIZE 256
#define RASTERIZER_BLOCK_SIZE 8
#define FRAMEBUFFER_BLOCK_SIZE 256

#define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)
inline void __getLastCudaError(const char *errorMessage, const char *file, const int line)
{
	cudaError_t err = cudaGetLastError();

	if (cudaSuccess != err)
	{
		FILE* log = fopen("log.txt", "a+");

		fprintf(log, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
			file, line, errorMessage, (int)err, cudaGetErrorString(err));

		fclose(log);
	}
}

static float4* transformation;
static float3 camera;
static int width;
static int height;

static int* depth = NULL;
static Fragment* fragmentBuffer = NULL;

static bool backCullingEnabled = true;
static bool renderWireframe = false;

void Init(IDirect3DDevice9Ex* device)
{
	//cudaD3D9SetDirect3DDevice(device);
	//getLastCudaError("cudaD3D9SetDirect3DDevice failed");

	cudaSetDevice(0);
	getLastCudaError("cudaSetDevice failed");

	cudaMalloc((void**)&transformation, 16 * sizeof(float));
}

Model* CreateModel(float3* vertices, float3* normals, float3* colors, int* indices, int numOfVertices, int numOfFaces)
{
	Model* model = new Model;

	model->numOfVertices = numOfVertices;
	model->numOfFaces = numOfFaces;

	cudaMalloc((void**)&model->vertexBufferIn, numOfVertices * sizeof(VertexShaderIn));
	cudaMalloc((void**)&model->vertexBufferOut, numOfVertices * sizeof(VertexShaderOut));
	cudaMalloc((void**)&model->primitivesBuffer, numOfFaces * sizeof(Triangle));
	cudaMalloc((void**)&model->indexBuffer, numOfFaces * 3 * sizeof(int));

	for (int i = 0; i < numOfVertices; i++)
	{
		VertexShaderIn vertex = { vertices[i], normals[i], colors[i] };
		cudaMemcpy(model->vertexBufferIn + i, &vertex, sizeof(VertexShaderIn), cudaMemcpyHostToDevice);
	}

	cudaMemcpy(model->indexBuffer, indices, numOfFaces * 3 * sizeof(int), cudaMemcpyHostToDevice);

	return model;
}

void SetTransformation(float4* transf, float3 cam)
{
	cudaMemcpy(transformation, transf, 4 * sizeof(float4), cudaMemcpyHostToDevice);
	camera = cam;
}

static cudaGraphicsResource* backBufferResource;
static void* backBufferLinear;
static IDirect3DSurface9* backBufferSurface;
static size_t backBufferPitch;

void Resize(unsigned int w, unsigned int h, IDirect3DSurface9* backBufSurface)
{
	width = w;
	height = h;

	if (depth != NULL) cudaFree(depth);
	if (fragmentBuffer != NULL) cudaFree(fragmentBuffer);

	cudaMalloc((void**)&depth, width * height * sizeof(int));
	getLastCudaError("cudaMalloc depth failed");

	cudaMalloc((void**)&fragmentBuffer, width * height * sizeof(Fragment));
	getLastCudaError("cudaMalloc fragmentBuffer failed");

	cudaGraphicsD3D9RegisterResource(&backBufferResource, backBufSurface, cudaGraphicsRegisterFlagsNone);
	getLastCudaError("cudaGraphicsD3D9RegisterResource failed");

	cudaMallocPitch(&backBufferLinear, &backBufferPitch, w * sizeof(int), h * sizeof(int));
	getLastCudaError("cudaMallocPitch failed");

	cudaMemset(backBufferLinear, 0xFF, backBufferPitch * h);
}

void FreeRasterizer()
{
	cudaFree(depth);
	cudaFree(fragmentBuffer);
	cudaFree(transformation);
}

void FreeModel(Model* model)
{

}

__forceinline__ __device__ float3 transform(float4* transformation, float3 v)
{
	float w = transformation[3].x * v.x + transformation[3].y * v.y + transformation[3].z * v.z + transformation[3].w;

	return make_float3(
		(transformation[0].x * v.x + transformation[0].y * v.y + transformation[0].z * v.z + transformation[0].w) / w,
		(transformation[1].x * v.x + transformation[1].y * v.y + transformation[1].z * v.z + transformation[1].w) / w,
		(transformation[2].x * v.x + transformation[2].y * v.y + transformation[2].z * v.z + transformation[2].w) / w
	);
}

__device__ float3 operator*(float3 v, float a)
{
	return make_float3(v.x * a, v.y * a, v.z * a);
}

__device__ float3 operator-(float3 a, float3 b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 operator+(float3 a, float3 b)
{
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__forceinline__ __device__ float dot(float3 a, float3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__forceinline__ __device__ float3 cross(float3 a, float3 b)
{
	return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

__forceinline__ __device__ float length(float3 v)
{
	return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

__forceinline__ __device__ float3 norm(float3 v)
{
	float l = length(v);

	return make_float3(v.x / l, v.y / l, v.z / l);
}

__device__ __forceinline__ float calculateSignedArea(float3* p1, float3* p2, float3* p3)
{
	return 0.5 * ((p3->x - p1->x) * (p2->y - p1->y) - (p2->x - p1->x) * (p3->y - p1->y));
}

__device__ __forceinline__ float calculateBarycentricCoordinateValue(float3* a, float3* b, float3* c, double area)
{
	return calculateSignedArea(a, b, c) / area;
}

__device__ __forceinline__ float3 calculateBarycentricCoordinate(float3* point, float3* p1, float3* p2, float3* p3)
{
	float area = calculateSignedArea(p1, p2, p3);

	float beta = calculateBarycentricCoordinateValue(p1, point, p3, area);
	float gamma = calculateBarycentricCoordinateValue(p1, p2, point, area);
	float alpha = 1.0 - beta - gamma;

	return make_float3(alpha, beta, gamma);
}

__device__ __forceinline__ float getZAtCoordinate(float3 barycentricCoord, float3 p1, float3 p2, float3 p3)
{
	return
		-(barycentricCoord.x * p1.z
			+ barycentricCoord.y * p2.z
			+ barycentricCoord.z * p3.z);
}

__device__ bool isBarycentricCoordInBounds(float3 barycentricCoord) {
	return
		barycentricCoord.x >= 0.0 && barycentricCoord.x <= 1.0 &&
		barycentricCoord.y >= 0.0 && barycentricCoord.y <= 1.0 &&
		barycentricCoord.z >= 0.0 && barycentricCoord.z <= 1.0;
}

__global__ void VertexShader(const VertexShaderIn* vertexIn, VertexShaderOut* vertexOut, float4* transformation, int vertCount)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= vertCount) return;

	vertexOut[index].Color = vertexIn[index].Color;
	vertexOut[index].ModelPos = vertexIn[index].Pos;
	vertexOut[index].Normal = vertexIn[index].Normal;

	vertexOut[index].Pos = transform(transformation, vertexIn[index].Pos);
}

__global__ void Assembler(VertexShaderOut* vertexOut, Triangle* primitivesBuffer, int* indices, int facesCount, float3 camera, bool cullBackface)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= facesCount) return;
	Triangle* triangle = &primitivesBuffer[index];

	triangle->v1 = vertexOut[indices[index * 3]];
	triangle->v2 = vertexOut[indices[index * 3 + 1]];
	triangle->v3 = vertexOut[indices[index * 3 + 2]];

	float3 v1 = triangle->v2.ModelPos - triangle->v1.ModelPos;
	float3 v2 = triangle->v3.ModelPos - triangle->v1.ModelPos;

	triangle->Visible = !cullBackface ||
		dot(
			triangle->v1.ModelPos - camera,
			norm(cross(v1, v2))
		) > 0;

	triangle->minX = triangle->v1.Pos.x;
	atomicMin(&triangle->minX, triangle->v2.Pos.x);
	atomicMin(&triangle->minX, triangle->v3.Pos.x);

	triangle->maxX = triangle->v1.Pos.x;
	atomicMax(&triangle->maxX, triangle->v2.Pos.x);
	atomicMax(&triangle->maxX, triangle->v3.Pos.x);

	triangle->minY = triangle->v1.Pos.y;
	atomicMin(&triangle->minY, triangle->v2.Pos.y);
	atomicMin(&triangle->minY, triangle->v3.Pos.y);

	triangle->maxY = triangle->v1.Pos.y;
	atomicMax(&triangle->maxY, triangle->v2.Pos.y);
	atomicMax(&triangle->maxY, triangle->v3.Pos.y);
}

__device__ void line(float3 start, float3 end, Fragment* depthBuffer, int width, int height)
{
	float3 color = make_float3(1, 1, 1);

	int x1 = start.x;
	int y1 = start.y;

	int x2 = end.x;
	int y2 = end.y;

	int dx = abs(x2 - x1);
	int dy = abs(y2 - y1);
	int sx = (x1 < x2) ? 1 : -1;
	int sy = (y1 < y2) ? 1 : -1;
	int err = dx - dy;

	if (x1 > 0 && x1 < width && y1 > 0 && y1 < height)
	{
		depthBuffer[x1 + y1 * width].Color = color;
	}

	while (!((x1 == x2) && (y1 == y2)))
	{
		int e2 = err << 1;
		if (e2 > -dy)
		{
			err -= dy;
			x1 += sx;
		}
		if (e2 < dx)
		{
			err += dx;
			y1 += sy;
		}

		if (x1 > 0 && x1 < width && y1 > 0 && y1 < height)
		{
			depthBuffer[x1 + y1 * width].Color = color;
		}
	}
}

__global__ void RasterizeWireframe(Triangle* primitivesBuffer, int* depth, Fragment* fragmentBuffer, int width, int height, int primitivesCount)
{
	Triangle* triangle = &primitivesBuffer[blockIdx.x];
	if (!triangle->Visible) return;

	float3 start = triangle->v1.Pos;
	float3 end = triangle->v2.Pos;

	if (threadIdx.x == 1)
	{
		end = triangle->v3.Pos;
	}
	else if (threadIdx.x == 2)
	{
		start = triangle->v3.Pos;
	}

	line(start, end, fragmentBuffer, width, height);
}

__global__ void RasterizeTriangle(Triangle* primitivesBuffer, int* depth, Fragment* fragmentBuffer, int width, int height, int primitivesCount)
{
	int index = blockIdx.x;
	if (index >= primitivesCount) return;

	Triangle* triangle = &primitivesBuffer[index];
	if (!triangle->Visible) return;

	int w = (triangle->maxX - triangle->minX) / blockDim.x + 1;
	int h = (triangle->maxY - triangle->minY) / blockDim.y + 1;

	int startX = triangle->minX + w * threadIdx.x;
	int startY = triangle->minY + h * threadIdx.y;

	if (startX + w < 0 || startX >= width || startY + h < 0 || startY >= height) return;

	for (int x = startX; x < startX + w; x++)
	{
		if (x < 0) continue;
		if (x >= width) break;

		for (int y = startY; y < startY + h; y++)
		{
			if (y < 0) continue;
			if (y >= height) break;

			float3 p = make_float3(x, y, 0);
			float3 coords = calculateBarycentricCoordinate(&p, &triangle->v1.Pos, &triangle->v2.Pos, &triangle->v3.Pos);

			if (!isBarycentricCoordInBounds(coords)) continue;

			int z = getZAtCoordinate(coords, triangle->v1.Pos, triangle->v2.Pos, triangle->v3.Pos) * 10000;
			int i = y * width + x;

			atomicMin(&depth[i], z);

			if (depth[i] == z)
			{
				fragmentBuffer[i].Color = triangle->v1.Color * coords.x + triangle->v2.Color * coords.y + triangle->v3.Color * coords.z;
				fragmentBuffer[i].Position = triangle->v1.Pos * coords.x + triangle->v2.Pos * coords.y + triangle->v3.Pos * coords.z;
				fragmentBuffer[i].Normal = triangle->v1.Normal * coords.x + triangle->v2.Normal * coords.y + triangle->v3.Normal * coords.z;
			}
		}
	}
}

__device__ __forceinline__ int Clamp(float v)
{
	if (v < 0) v = 0;
	else if (v > 1) v = 1;
	return v * 255;
}

__global__ void CopyToFrameBuffer(Fragment* fragmentBuffer, int* backBuffer, int width, int height, int pitch)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int* pixel;

	if (x >= width || y >= height) return;

	//frameBuffer[i] = (Clamp(c.x) << 16) | (Clamp(c.y) << 8) | Clamp(c.z);
	pixel = backBuffer + y * pitch + x;
	pixel[0] = 0xFF00FFFF;
}

__host__ void ClearBuffers()
{
	cudaMemset(depth, 5000000, width * height * sizeof(int));
	cudaMemset(fragmentBuffer, 0, width * height * sizeof(Fragment));
}
cudaStream_t stream = 0;
cudaGraphicsResource *ppResources[1] =
{
	backBufferResource
};

void Begin()
{
	ClearBuffers();

	cudaGraphicsMapResources(1, ppResources, stream);
}

void End()
{
	cudaArray *cuArray;

	cudaGraphicsSubResourceGetMappedArray(&cuArray, backBufferResource, 0, 0);
	getLastCudaError("cudaGraphicsSubResourceGetMappedArray failed");

	//CopyToFrameBuffer << <dim3((width + 16 - 1) / 16, (height + 16 - 1) / 16), dim3(16, 16) >> > (fragmentBuffer, (int*)backBufferLinear, width, height, backBufferPitch);

	cudaMemcpy2DToArray(cuArray, 0, 0, backBufferLinear, backBufferPitch, 5 * sizeof(int), 5, cudaMemcpyDeviceToDevice);
	getLastCudaError("cudaMemcpy2DToArray failed");

	cudaGraphicsUnmapResources(1, ppResources, stream);
	//cudaMemcpy(hostFrameBuffer, frameBuffer, width * height * sizeof(int), cudaMemcpyDeviceToHost);
}

void DrawModel(Model* model)
{
	/*int vertexShaderGridSize = (numOfVertices - 1) / VERTEX_SHADER_BLOCK_SIZE + 1;
	int assemblerGridSize = (numOfFaces - 1) / VERTEX_SHADER_BLOCK_SIZE + 1;

	VertexShader << <vertexShaderGridSize, VERTEX_SHADER_BLOCK_SIZE >> > (vertexBufferIn, vertexBufferOut, transformation, numOfVertices);
	Assembler << <assemblerGridSize, VERTEX_SHADER_BLOCK_SIZE >> > (vertexBufferOut, primitivesBuffer, indexBuffer, numOfFaces, camera, true);
	RasterizeTriangle << <numOfFaces, dim3(RASTERIZER_BLOCK_SIZE, RASTERIZER_BLOCK_SIZE) >> > (primitivesBuffer, depth, fragmentBuffer, width, height, numOfFaces);*/

	//RasterizeWireframe<< <numOfFaces, 3 >> > (primitivesBuffer, depth, fragmentBuffer, width, height, numOfFaces);
}
