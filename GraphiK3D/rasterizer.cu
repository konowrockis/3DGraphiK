#include "GL/glew.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "rasterizer.h"
#include "VectorUtils.h"

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cuda_gl_interop.h>

#include "GLM/vec3.hpp"
#include "GLM/vec4.hpp"
#include "GLM/mat4x4.hpp"

#include "stream_compaction.h"

#define VERTEX_SHADER_BLOCK_SIZE 256
#define RASTERIZER_BLOCK_SIZE 16
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

static cudaArray_t framebuf_device;
static int* framebuf;
static Triangle* compactionOutput;

void Init()
{
	cudaSetDevice(0);
	getLastCudaError("cudaSetDevice failed");

	cudaMalloc((void**)&transformation, 16 * sizeof(float));
}

RasterizerModel* CreateModel(Model* model)
{
	RasterizerModel* rasterizerModel = new RasterizerModel;

	rasterizerModel->numOfVertices = model->numOfVertices;
	rasterizerModel->numOfFaces = model->numOfFaces;

	cudaMalloc((void**)&rasterizerModel->vertexBufferIn, model->numOfVertices * sizeof(VertexShaderIn));
	cudaMalloc((void**)&rasterizerModel->vertexBufferOut, model->numOfVertices * sizeof(VertexShaderOut));
	cudaMalloc((void**)&rasterizerModel->primitivesBuffer, model->numOfFaces * sizeof(Triangle));
	cudaMalloc((void**)&rasterizerModel->indexBuffer, model->numOfFaces * 3 * sizeof(int));
	cudaMalloc((void**)&compactionOutput, model->numOfFaces * sizeof(Triangle));

	for (int i = 0; i < model->numOfVertices; i++)
	{
		VertexShaderIn vertex = { model->vertices[i], model->normals[i], model->colors[i] };
		cudaMemcpy(rasterizerModel->vertexBufferIn + i, &vertex, sizeof(VertexShaderIn), cudaMemcpyHostToDevice);
	}

	cudaMemcpy(rasterizerModel->indexBuffer, model->indices, model->numOfFaces * 3 * sizeof(int), cudaMemcpyHostToDevice);

	return rasterizerModel;
}

void SetTransformation(glm::mat4x4 transf, glm::vec3 cam)
{
	cudaMemcpy(transformation, &transf, sizeof(glm::mat4x4), cudaMemcpyHostToDevice);

	camera = make_float3(cam.x, cam.y, cam.z);
}

void Resize(unsigned int w, unsigned int h, GLuint texture)
{
	width = w;
	height = h;

	if (depth != NULL) cudaFree(depth);
	if (fragmentBuffer != NULL) cudaFree(fragmentBuffer);

	cudaMalloc((void**)&depth, width * height * sizeof(int));
	getLastCudaError("cudaMalloc depth failed");

	cudaMalloc((void**)&fragmentBuffer, width * height * sizeof(Fragment));
	getLastCudaError("cudaMalloc fragmentBuffer failed");

	cudaMalloc((void**)&framebuf, width * height * sizeof(int));
	getLastCudaError("cudaMalloc framebuf failed");

	cudaGraphicsResource* resource;

	cudaGraphicsGLRegisterImage(&resource, texture, GL_TEXTURE_3D, CU_GRAPHICS_REGISTER_FLAGS_NONE);
	getLastCudaError("cuGraphicsGLRegisterImage failed");

	cudaGraphicsMapResources(1, &resource, 0);
	getLastCudaError("cuGraphicsMapResources failed");

	cudaGraphicsSubResourceGetMappedArray(&framebuf_device, resource, 0, 0);
	getLastCudaError("cuGraphicsSubResourceGetMappedArray failed");

	cudaGraphicsUnmapResources(1, &resource, 0);
	getLastCudaError("cuGraphicsUnmapResources failed");
}

void FreeRasterizer()
{
	cudaFree(depth);
	cudaFree(fragmentBuffer);
	cudaFree(transformation);
}

void FreeModel(RasterizerModel* model)
{

}

__forceinline__ __device__ float3 transform(float4* transformation, float3 v)
{
	float w = transformation[0].w * v.x + transformation[1].w * v.y + transformation[2].w * v.z + transformation[3].w;

	return make_float3(
		(transformation[0].x * v.x + transformation[1].x * v.y + transformation[2].x * v.z + transformation[3].x) / w,
		(transformation[0].y * v.x + transformation[1].y * v.y + transformation[2].y * v.z + transformation[3].y) / w,
		(transformation[0].z * v.x + transformation[1].z * v.y + transformation[2].z * v.z + transformation[3].z) / w
	);
}

__device__ float calculateSignedArea(float3 p1, float3 p2, float3 p3)
{
	return 0.5 * ((p3.x - p1.x) * (p2.y - p1.y) - (p2.x - p1.x) * (p3.y - p1.y));
}

__device__ float calculateBarycentricCoordinateValue(float3 a, float3 b, float3 c, double area)
{
	return calculateSignedArea(a, b, c) / area;
}

__device__ float3 calculateBarycentricCoordinate(float3 point, float3 p1, float3 p2, float3 p3)
{
	float area = calculateSignedArea(p1, p2, p3);

	float beta = calculateBarycentricCoordinateValue(p1, point, p3, area);
	float gamma = calculateBarycentricCoordinateValue(p1, p2, point, area);
	float alpha = 1.0 - beta - gamma;

	return make_float3(alpha, beta, gamma);
}

__device__ float getZAtCoordinate(float3 barycentricCoord, float3 p1, float3 p2, float3 p3)
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

	vertexOut[index].Pos = transform(transformation, vertexIn[index].Pos);

	vertexOut[index].Color = vertexIn[index].Color;
	vertexOut[index].ModelPos = vertexIn[index].Pos;
	vertexOut[index].Normal = vertexIn[index].Normal;
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

	triangle->x = glm::min(glm::min(triangle->v1.Pos.x, triangle->v2.Pos.x), triangle->v3.Pos.x);
	triangle->y = glm::min(glm::min(triangle->v1.Pos.y, triangle->v2.Pos.y), triangle->v3.Pos.y);
	triangle->w = glm::max(glm::max(triangle->v1.Pos.x, triangle->v2.Pos.x), triangle->v3.Pos.x) - triangle->x;
	triangle->h = glm::max(glm::max(triangle->v1.Pos.y, triangle->v2.Pos.y), triangle->v3.Pos.y) - triangle->y;
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
	Triangle* triangle = primitivesBuffer + blockIdx.x;

	int minX = triangle->x + (triangle->w / blockDim.x + 1) * threadIdx.x;
	int minY = triangle->y + (triangle->h / blockDim.y + 1) * threadIdx.y;
	int maxY = glm::min(minY + triangle->h / (int)blockDim.y + 1, height);
	int maxX = glm::min(minX + triangle->w / (int)blockDim.y + 1, width);

	for (int y = glm::max(minY, 0); y < maxY; y++)
	{
		for (int x = glm::max(minX, 0); x < maxX; x++)
		{
			float3 coords = calculateBarycentricCoordinate(make_float3(x, y, 0), triangle->v1.Pos, triangle->v2.Pos, triangle->v3.Pos);

			if (isBarycentricCoordInBounds(coords)) 
			{
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
}

//__global__ void RasterizeTriangle(Triangle* primitivesBuffer, int* depth, Fragment* fragmentBuffer, int width, int height, int primitivesCount)
//{
//	Triangle* triangle = primitivesBuffer + blockIdx.z;
//
//	int minX = triangle->x + (triangle->w / blockDim.x + 1) * threadIdx.x;
//	int minY = triangle->y + (triangle->h / blockDim.y + 1) * threadIdx.y;
//	int maxY = glm::min(minY + triangle->h / (int)blockDim.y + 1, height);
//	int maxX = glm::min(minX + triangle->w / (int)blockDim.y + 1, width);
//
//	int y0 = GRID_SIZE * (blockIdx.y * blockDim.y + threadIdx.y);
//	int x0 = GRID_SIZE * (blockIdx.x * blockDim.x + threadIdx.x);
//
//	if (triangle->w < x0 || triangle->x > x0 + GRID_SIZE || triangle->h < y0 || triangle->y > y0 + GRID_SIZE) return;
//
//	for (int y1 = 0; y1 < GRID_SIZE; y1++)
//	{
//		for (int x1 = 0; x1 < GRID_SIZE; x1++)
//		{
//			int y = y0 + y1;
//			int x = x0 + x1;
//
//			float3 coords = calculateBarycentricCoordinate(make_float3(x, y, 0), triangle->v1.Pos, triangle->v2.Pos, triangle->v3.Pos);
//
//			if (isBarycentricCoordInBounds(coords))
//			{
//				int z = getZAtCoordinate(coords, triangle->v1.Pos, triangle->v2.Pos, triangle->v3.Pos) * 10000;
//				int i = y * width + x;
//
//				atomicMin(&depth[i], z);
//
//				if (depth[i] == z)
//				{
//					fragmentBuffer[i].Color = triangle->v1.Color * coords.x + triangle->v2.Color * coords.y + triangle->v3.Color * coords.z;
//					fragmentBuffer[i].Position = triangle->v1.Pos * coords.x + triangle->v2.Pos * coords.y + triangle->v3.Pos * coords.z;
//					fragmentBuffer[i].Normal = triangle->v1.Normal * coords.x + triangle->v2.Normal * coords.y + triangle->v3.Normal * coords.z;
//				}
//			}
//		}
//	}
//}


__device__ __forceinline__ int Clamp(float v)
{
	if (v < 0) v = 0;
	else if (v > 1) v = 1;
	return v * 255;
}

__global__ void CopyToFrameBuffer(Fragment* fragmentBuffer, int* backBuffer, int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;

	//frameBuffer[i] = (Clamp(c.x) << 16) | (Clamp(c.y) << 8) | Clamp(c.z);
	int* pixel = backBuffer + y * width + x;
	Fragment* fragment = fragmentBuffer + y * width + x;

	pixel[0] = (Clamp(fragment->Color.x) << 16) | (Clamp(fragment->Color.y) << 8) | Clamp(fragment->Color.z);
	//pixel[0] = (Clamp((fragment->Normal.x + 1) / 2) << 16) | (Clamp((fragment->Normal.y + 1) / 2) << 8) | Clamp((fragment->Normal.z + 1) / 2);
}

__host__ void ClearBuffers()
{
	cudaMemset(depth, 5000000, width * height * sizeof(int));
	cudaMemset(fragmentBuffer, 0, width * height * sizeof(Fragment));
}

void Begin()
{
	ClearBuffers();
}

void End()
{
	CopyToFrameBuffer << <dim3((width + 16 - 1) / 16, (height + 16 - 1) / 16), dim3(16, 16) >> > (fragmentBuffer, framebuf, width, height);

	cudaMemcpyToArray(framebuf_device, 0, 0, framebuf, width * height * sizeof(int), cudaMemcpyDeviceToDevice);
}

void DrawModel(RasterizerModel* model)
{
	int vertexShaderGridSize = (model->numOfVertices - 1) / VERTEX_SHADER_BLOCK_SIZE + 1;
	int assemblerGridSize = (model->numOfFaces - 1) / VERTEX_SHADER_BLOCK_SIZE + 1;
	int primitiveCount = model->numOfFaces;

	VertexShader << <vertexShaderGridSize, VERTEX_SHADER_BLOCK_SIZE >> > (model->vertexBufferIn, model->vertexBufferOut, transformation, model->numOfVertices);
	Assembler << <assemblerGridSize, VERTEX_SHADER_BLOCK_SIZE >> > (model->vertexBufferOut, model->primitivesBuffer, model->indexBuffer, model->numOfFaces, camera, true);

	//primitiveCount = Compact(model->numOfFaces, compactionOutput, model->primitivesBuffer);
	//cudaMemcpy(triangle->primitivesBuffer, compactionOutput, primitiveCount * sizeof(Triangle), cudaMemcpyDeviceToDevice);

	RasterizeTriangle << <primitiveCount, dim3(RASTERIZER_BLOCK_SIZE, RASTERIZER_BLOCK_SIZE) >> > (model->primitivesBuffer, depth, fragmentBuffer, width, height, model->numOfFaces);
	
	//RasterizeTriangle << <dim3(width / (RASTERIZER_BLOCK_SIZE * GRID_SIZE), height / (RASTERIZER_BLOCK_SIZE * GRID_SIZE), primitiveCount), dim3(RASTERIZER_BLOCK_SIZE, RASTERIZER_BLOCK_SIZE) >> > (compactionOutput, depth, fragmentBuffer, width, height, model->numOfFaces);

	//RasterizeWireframe<< <numOfFaces, 3 >> > (primitivesBuffer, depth, fragmentBuffer, width, height, numOfFaces);
}


float3 operator*(float3 v, float a)
{
	return make_float3(v.x * a, v.y * a, v.z * a);
}

float3 operator/(float3 v, float a)
{
	return make_float3(v.x / a, v.y / a, v.z / a);
}

float3 operator-(float3 a, float3 b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

float3 operator+(float3 a, float3 b)
{
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

float dot(float3 a, float3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

float3 cross(float3 a, float3 b)
{
	return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

float length(float3 v)
{
	return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

float3 norm(float3 v)
{
	float l = length(v);

	float3 a = make_float3(v.x / l, v.y / l, v.z / l);
	return a;
}