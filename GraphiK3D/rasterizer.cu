#include "GL/glew.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "rasterizer.h"
#include "VectorUtils.h"
#include "font.h"
#include <cstring>

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
#define RASTERIZER_BLOCK_SIZE 9
#define FRAMEBUFFER_BLOCK_SIZE 256
#define MAXIMUM_LINES 20

#define getLastCudaError(msg) __getLastCudaError (msg, __FILE__, __LINE__)
inline void __getLastCudaError(const char *errorMessage, const char *file, const int line)
{
	cudaError_t err = cudaGetLastError();

	if (cudaSuccess != err)
	{
		FILE* log = fopen("log.txt", "a+");

		fprintf(log, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
			file, line, errorMessage, (int)err, cudaGetErrorString(err));

		printf("%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
			file, line, errorMessage, (int)err, cudaGetErrorString(err));

		fclose(log);
	}
}

SceneParams_t SceneParams;

struct Rasterizer_t
{
	int Width;
	int Height;

	int* DepthBuffer;
	Fragment* FragmentBuffer;

	cudaArray_t DeviceFrameBuffer;
	int* FrameBuffer;

	Triangle* CompactionOutput;
	int CompactionOutputSize;
} Rasterizer;

TextLine* TextLines;

void Init()
{
	cudaSetDevice(0);
	getLastCudaError("cudaSetDevice failed");

	cudaMalloc((void**)&SceneParams.Transformation, 16 * sizeof(float));
	cudaMalloc((void**)&TextLines, sizeof(TextLine) * MAXIMUM_LINES);
}

void SetLine(int i, TextLine line)
{
	cudaMemcpy(TextLines + i, &line, sizeof(TextLine), cudaMemcpyHostToDevice);
}

Texture* LoadTexture(int textureWidth, int textureHeight, void* srcTexture)
{
	Texture* tex = new Texture;

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
	getLastCudaError("cudaCreateChannelDesc failed");
	cudaMallocArray(&tex->cuArray, &channelDesc, textureWidth, textureHeight);
	getLastCudaError("texture array mallocArray failed");

	cudaMemcpyToArray(tex->cuArray, 0, 0, srcTexture, textureWidth * textureHeight * 4 * sizeof(float), cudaMemcpyHostToDevice);

	// Specify texture
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = tex->cuArray;

	// Specify texture object parameters
	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeWrap;
	texDesc.addressMode[1] = cudaAddressModeWrap;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 1;

	// Create texture object
	cudaCreateTextureObject(&tex->Tex, &resDesc, &texDesc, NULL);
	getLastCudaError("createTexture object failed");

	return tex;
}

void FreeTexture(Texture* texture)
{
	cudaFreeArray(texture->cuArray);
	cudaDestroyTextureObject(texture->Tex);

	delete texture;
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

	if (Rasterizer.CompactionOutput == NULL)
	{
		cudaMalloc((void**)&Rasterizer.CompactionOutput, model->numOfFaces * sizeof(Triangle));
		Rasterizer.CompactionOutputSize = model->numOfFaces;
	}
	else if (Rasterizer.CompactionOutputSize < model->numOfFaces)
	{
		cudaFree(Rasterizer.CompactionOutput);
		cudaMalloc((void**)&Rasterizer.CompactionOutput, model->numOfFaces * sizeof(Triangle));
		Rasterizer.CompactionOutputSize = model->numOfFaces;
	}

	for (int i = 0; i < model->numOfVertices; i++)
	{
		VertexShaderIn vertex = { model->vertices[i], model->normals[i], model->colors[i] };
		cudaMemcpy(rasterizerModel->vertexBufferIn + i, &vertex, sizeof(VertexShaderIn), cudaMemcpyHostToDevice);
	}

	cudaMemcpy(rasterizerModel->indexBuffer, model->indices, model->numOfFaces * 3 * sizeof(int), cudaMemcpyHostToDevice);

	return rasterizerModel;
}

void SetLightParams(glm::vec3 diffuseColor, glm::vec3 specularColor, glm::vec3 ambientColor, float diffuseConstant, float specularConstant, float ambientConstant, int shininess)
{

}

void SetTransformation(glm::mat4x4 transf)
{
	cudaMemcpy(SceneParams.Transformation, &transf, sizeof(glm::mat4x4), cudaMemcpyHostToDevice);
}

void SetCameraPosition(glm::vec3 camera)
{
	SceneParams.CameraPosition = make_float3(camera.x, camera.y, camera.z);
}

void SetLightPosition(glm::vec3 light)
{
	SceneParams.LightPosition = make_float3(light.x, light.y, light.z);
}

void Resize(unsigned int width, unsigned int height, GLuint texture)
{
	Rasterizer.Width = width;
	Rasterizer.Height = height;

	if (Rasterizer.DepthBuffer != NULL) cudaFree(Rasterizer.DepthBuffer);
	if (Rasterizer.DepthBuffer != NULL) cudaFree(Rasterizer.FragmentBuffer);

	cudaMalloc((void**)&Rasterizer.DepthBuffer, width * height * sizeof(int));
	getLastCudaError("cudaMalloc depthBuffer failed");

	cudaMalloc((void**)&Rasterizer.FragmentBuffer, width * height * sizeof(Fragment));
	getLastCudaError("cudaMalloc fragmentBuffer failed");

	cudaMalloc((void**)&Rasterizer.FrameBuffer, width * height * sizeof(int));
	getLastCudaError("cudaMalloc frameBuffer failed");

	cudaGraphicsResource* resource;

	cudaGraphicsGLRegisterImage(&resource, texture, GL_TEXTURE_3D, CU_GRAPHICS_REGISTER_FLAGS_NONE);
	getLastCudaError("cuGraphicsGLRegisterImage failed");

	cudaGraphicsMapResources(1, &resource, 0);
	getLastCudaError("cuGraphicsMapResources failed");

	cudaGraphicsSubResourceGetMappedArray(&Rasterizer.DeviceFrameBuffer, resource, 0, 0);
	getLastCudaError("cuGraphicsSubResourceGetMappedArray failed");

	cudaGraphicsUnmapResources(1, &resource, 0);
	getLastCudaError("cuGraphicsUnmapResources failed");
}

void FreeRasterizer()
{
	cudaFree(Rasterizer.CompactionOutput);
	cudaFree(Rasterizer.DepthBuffer);
	cudaFree(Rasterizer.FragmentBuffer);
	cudaFree(Rasterizer.FrameBuffer);
	cudaFree(TextLines);
}

void FreeModel(RasterizerModel* model)
{
	cudaFree(model->indexBuffer);
	cudaFree(model->primitivesBuffer);
	cudaFree(model->vertexBufferIn);
	cudaFree(model->vertexBufferOut);

	delete model;
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

__global__ void VertexShader(const VertexShaderIn* vertexIn, VertexShaderOut* vertexOut, float4* transformation, int vertCount)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= vertCount) return;

	vertexOut[index].Pos = transform(transformation, vertexIn[index].Pos);

	vertexOut[index].Color = vertexIn[index].Color;
	vertexOut[index].ModelPos = vertexIn[index].Pos;
	vertexOut[index].Normal = vertexIn[index].Normal;
}

__global__ void Assembler(VertexShaderOut* vertexOut, Triangle* primitivesBuffer, int* indices, int facesCount, float3 camera, BackFaceCullingType backfaceCulling, int width, int height)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= facesCount) return;
	Triangle* triangle = &primitivesBuffer[index];

	triangle->v1 = vertexOut[indices[index * 3]];
	triangle->v2 = vertexOut[indices[index * 3 + 1]];
	triangle->v3 = vertexOut[indices[index * 3 + 2]];

	float3 v1 = triangle->v2.ModelPos - triangle->v1.ModelPos;
	float3 v2 = triangle->v3.ModelPos - triangle->v1.ModelPos;

	if (backfaceCulling == BackCullingNone)
	{
		triangle->Visible = true;
	}
	else if (backfaceCulling == BackCullingCW)
	{
		triangle->Visible = dot(
			triangle->v1.ModelPos - camera,
			norm(cross(v1, v2))
		) > 0;
	}
	else
	{
		triangle->Visible = dot(
			triangle->v1.ModelPos - camera,
			norm(cross(v1, v2))
		) < 0;
	}
	
	triangle->minx = glm::floor(glm::max(glm::min(glm::min(triangle->v1.Pos.x, triangle->v2.Pos.x), triangle->v3.Pos.x), 0.f));
	triangle->miny = glm::floor(glm::max(glm::min(glm::min(triangle->v1.Pos.y, triangle->v2.Pos.y), triangle->v3.Pos.y), 0.f));
	triangle->maxx = glm::ceil(glm::min(glm::max(glm::max(triangle->v1.Pos.x, triangle->v2.Pos.x), triangle->v3.Pos.x), (float)width));
	triangle->maxy = glm::ceil(glm::min(glm::max(glm::max(triangle->v1.Pos.y, triangle->v2.Pos.y), triangle->v3.Pos.y), (float)height));

	if (triangle->minx >= triangle->maxx || triangle->miny >= triangle->maxy)
	{
		triangle->Visible = false;
	}
}

__device__ void line(float3 start, float3 end, Fragment* frameBuffer, int width, int height)
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
		frameBuffer[x1 + y1 * width].Color = color;
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
			frameBuffer[x1 + y1 * width].Color = color;
		}
	}
}

__global__ void RasterizeWireframe(Triangle* primitivesBuffer, int* depth, Fragment* fragmentBuffer, int width, int height)
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

__global__ void RasterizeTriangle(Triangle* primitivesBuffer, int* depth, Fragment* fragmentBuffer, int width, cudaTextureObject_t tex)
{
	Triangle* triangle;

	triangle = primitivesBuffer + blockIdx.x;

	__shared__ float4 vals[12];
	float3 tmp = ((float3*)triangle)[threadIdx.x * 4 % 11];
	vals[threadIdx.x] = make_float4(tmp.x, tmp.y, tmp.z, 0);

	__shared__ float w[RASTERIZER_BLOCK_SIZE + 1];
	__shared__ float h[RASTERIZER_BLOCK_SIZE + 1];

	w[threadIdx.x + 1] = triangle->minx + (triangle->maxx - triangle->minx) / RASTERIZER_BLOCK_SIZE * (threadIdx.x + 1);
	h[threadIdx.y + 1] = triangle->miny + (triangle->maxy - triangle->miny) / RASTERIZER_BLOCK_SIZE * (threadIdx.y + 1);

	if (threadIdx.x == 0)
	{
		w[0] = triangle->minx;

		vals[9] = make_float4(triangle->v1.ModelPos.x, triangle->v1.ModelPos.y, triangle->v1.ModelPos.z, 0);
		vals[10] = make_float4(triangle->v2.ModelPos.x, triangle->v2.ModelPos.y, triangle->v2.ModelPos.z, 0);
		vals[11] = make_float4(triangle->v3.ModelPos.x, triangle->v3.ModelPos.y, triangle->v3.ModelPos.z, 0);
	}
	if (threadIdx.y == 0)
	{
		h[0] = triangle->miny;
	}

	__syncthreads();
	
	for (int y = h[threadIdx.y]; y < h[threadIdx.y + 1]; y++)
	{
		for (int x = w[threadIdx.x]; x < w[threadIdx.x + 1]; x++)
		{
			float area = 0.5f * ((vals[2].x - vals[0].x) * (vals[1].y - vals[0].y) - (vals[1].x - vals[0].x) * (vals[2].y - vals[0].y));
			
			float beta = 0.5f * ((vals[2].x - vals[0].x) * (y - vals[0].y) - (x - vals[0].x) * (vals[2].y - vals[0].y)) / area;
			float gamma = 0.5f * ((x - vals[0].x) * (vals[1].y - vals[0].y) - (vals[1].x - vals[0].x) * (y - vals[0].y)) / area;
			float alpha = 1.0f - beta - gamma;

			if (alpha >= 0.0 && alpha <= 1.0 && beta >= 0.0 && beta <= 1.0 && gamma >= 0.0 && gamma <= 1.0)
			{
				int z = (alpha * vals[0].z + beta * vals[1].z + gamma * vals[2].z) * -10000;
				int i = y * width + x;
				
				atomicMin(&depth[i], z);

				if (depth[i] == z)
				{
					Fragment* fragment = fragmentBuffer + i;

					fragment->Position.x = vals[9].x * alpha + vals[10].x * beta + vals[11].x * gamma;
					fragment->Position.y = vals[9].y * alpha + vals[10].y * beta + vals[11].y * gamma;
					fragment->Position.z = vals[9].z * alpha + vals[10].z * beta + vals[11].z * gamma;

					fragment->Normal.x = vals[3].x * alpha + vals[4].x * beta + vals[5].x * gamma;
					fragment->Normal.y = vals[3].y * alpha + vals[4].y * beta + vals[5].y * gamma;
					fragment->Normal.z = vals[3].z * alpha + vals[4].z * beta + vals[5].z * gamma;

					if (tex == NULL)
					{
						fragment->Color.x = vals[6].x * alpha + vals[7].x * beta + vals[8].x * gamma;
						fragment->Color.y = vals[6].y * alpha + vals[7].y * beta + vals[8].y * gamma;
						fragment->Color.z = vals[6].z * alpha + vals[7].z * beta + vals[8].z * gamma;
					}
					else
					{
						float4 color = tex2D<float4>(tex, 
							vals[6].x * alpha + vals[7].x * beta + vals[8].x * gamma, 
							vals[6].y * alpha + vals[7].y * beta + vals[8].y * gamma
						);

						fragment->Color.x = color.x;
						fragment->Color.y = color.y;
						fragment->Color.z = color.z;
					}
				}
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

__global__ void CopyToFrameBuffer(Fragment* fragmentBuffer, int* backBuffer, const int* depthBuffer, const int width, const int height, const SceneParams_t scene)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;

	int* pixel = backBuffer + y * width + x;
	Fragment* fragment = fragmentBuffer + y * width + x;

	if (scene.RenderOutput == RenderOutputNormal)
	{
		pixel[0] = (Clamp((fragment->Normal.x + 1) / 2) << 16) | (Clamp((fragment->Normal.y + 1) / 2) << 8) | Clamp((fragment->Normal.z + 1) / 2);
		return;
	}
	else if (scene.RenderOutput == RenderOutputZBuffer)
	{
		int z = Clamp(((float)*(depthBuffer + y * width + x)) / -17000.f);
		pixel[0] = (z << 16) | (z << 8) | z;
		return;
	}

	if (!scene.LightEnabled || (fragment->Position.x == 0 && fragment->Position.y == 0 && fragment->Position.z == 0))
	{
		pixel[0] = (Clamp(fragment->Color.x) << 16) | (Clamp(fragment->Color.y) << 8) | Clamp(fragment->Color.z);
		return;
	}

	float3 L, V, R;

	L.x = scene.LightPosition.x - fragment->Position.x;
	L.y = scene.LightPosition.y - fragment->Position.y;
	L.z = scene.LightPosition.z - fragment->Position.z;

	float len = length(L);
	L.x /= len; L.y /= len; L.z /= len;

	float kd = dot(fragment->Normal, L);

	R.x = 2 * kd * fragment->Normal.x - L.x;
	R.y = 2 * kd * fragment->Normal.y - L.y;
	R.z = 2 * kd * fragment->Normal.z - L.z;

	len = length(R);
	R.x /= len; R.y /= len; R.z /= len;

	V.x = scene.CameraPosition.x - fragment->Position.x;
	V.y = scene.CameraPosition.y - fragment->Position.y;
	V.z = scene.CameraPosition.z - fragment->Position.z;

	len = length(V);
	V.x /= len; V.y /= len; V.z /= len;

	kd *= scene.LightDiffuseConstant;
	float ks = scene.LightSpecularConstant * glm::pow(dot(R, V), scene.LightShininess);
	float ka = scene.LightAmbientColor.x * scene.LightAmbientConstant;

	pixel[0] =
		(Clamp(fragment->Color.x * (ka * scene.LightAmbientColor.x + kd * scene.LightDiffuseColor.x + ks * scene.LightSpecularColor.x)) << 16) |
		(Clamp(fragment->Color.y * (ka * scene.LightAmbientColor.y + kd * scene.LightDiffuseColor.y + ks * scene.LightSpecularColor.y)) << 8) |
		Clamp(fragment->Color.z * (ka * scene.LightAmbientColor.z + kd * scene.LightDiffuseColor.z + ks * scene.LightSpecularColor.z));
}

__host__ void ClearBuffers()
{
	cudaMemset(Rasterizer.DepthBuffer, 5000000, Rasterizer.Width * Rasterizer.Height * sizeof(int));
	cudaMemset(Rasterizer.FragmentBuffer, 0, Rasterizer.Width * Rasterizer.Height * sizeof(Fragment));
}

__global__ void DrawText(TextLine* lines, int* backBuffer, int width, int color) {
	TextLine* line = lines + blockIdx.x;
	int c = threadIdx.x;

	if (line->length < c) return;

	int x = threadIdx.x * CHAR_WIDTH;
	int y = blockIdx.x * CHAR_HEIGHT;

	int ch = line->text[c];

	for (int i = 0; i < CHAR_HEIGHT; i++) 
	{
		for (int j = 0; j < CHAR_WIDTH; j++) 
		{
			if (font[ch][i][j]) {
				backBuffer[(y + i) * width + (x + j)] = color;
			}
		}
	}
}

void Begin()
{
	ClearBuffers();
}

void End()
{
	CopyToFrameBuffer << <dim3((Rasterizer.Width + 16 - 1) / 16, (Rasterizer.Height + 16 - 1) / 16), dim3(16, 16) >> > (Rasterizer.FragmentBuffer, Rasterizer.FrameBuffer, Rasterizer.DepthBuffer, Rasterizer.Width, Rasterizer.Height, SceneParams);

	DrawText << <MAXIMUM_LINES, 128 >> > (TextLines, Rasterizer.FrameBuffer, Rasterizer.Width, 0xFFFFFF00);
	cudaMemcpyToArray(Rasterizer.DeviceFrameBuffer, 0, 0, Rasterizer.FrameBuffer, Rasterizer.Width * Rasterizer.Height * sizeof(int), cudaMemcpyDeviceToDevice);
}

void DrawModel(RasterizerModel* model, Texture* tex)
{
	int vertexShaderGridSize = (model->numOfVertices - 1) / VERTEX_SHADER_BLOCK_SIZE + 1;
	int assemblerGridSize = (model->numOfFaces - 1) / VERTEX_SHADER_BLOCK_SIZE + 1;
	int primitiveCount = model->numOfFaces;

	VertexShader << <vertexShaderGridSize, VERTEX_SHADER_BLOCK_SIZE >> > (model->vertexBufferIn, model->vertexBufferOut, SceneParams.Transformation, model->numOfVertices);
	Assembler << <assemblerGridSize, VERTEX_SHADER_BLOCK_SIZE >> > (model->vertexBufferOut, model->primitivesBuffer, model->indexBuffer, model->numOfFaces, SceneParams.CameraPosition, SceneParams.BackCulling, Rasterizer.Width, Rasterizer.Height);

	if (SceneParams.BackCulling != BackCullingNone)
	{
		primitiveCount = Compact(model->numOfFaces, Rasterizer.CompactionOutput, model->primitivesBuffer);
		cudaMemcpy(model->primitivesBuffer, Rasterizer.CompactionOutput, primitiveCount * sizeof(Triangle), cudaMemcpyDeviceToDevice);
	}

	if (SceneParams.RenderMode == RenderModeWireframe)
	{
		RasterizeWireframe << <primitiveCount, 3 >> > (model->primitivesBuffer, Rasterizer.DepthBuffer, Rasterizer.FragmentBuffer, Rasterizer.Width, Rasterizer.Height);
	}
	else if (SceneParams.RenderMode == RenderModeTriangles)
	{
		RasterizeTriangle << <primitiveCount, dim3(RASTERIZER_BLOCK_SIZE, RASTERIZER_BLOCK_SIZE) >> > (model->primitivesBuffer, Rasterizer.DepthBuffer, Rasterizer.FragmentBuffer, Rasterizer.Width, NULL);
		//RasterizeTriangle << <primitiveCount, dim3(RASTERIZER_BLOCK_SIZE, RASTERIZER_BLOCK_SIZE) >> > (Rasterizer.CompactionOutput, Rasterizer.DepthBuffer, Rasterizer.FragmentBuffer, Rasterizer.Width, tex->Tex);
	}
	else 
	{
		RasterizeTriangle << <primitiveCount, dim3(RASTERIZER_BLOCK_SIZE, RASTERIZER_BLOCK_SIZE) >> > (model->primitivesBuffer, Rasterizer.DepthBuffer, Rasterizer.FragmentBuffer, Rasterizer.Width, tex->Tex);
	}
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
