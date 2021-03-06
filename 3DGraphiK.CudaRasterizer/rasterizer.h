#pragma once
#include <stdio.h>
#include "device_launch_parameters.h"
#include "cuda_d3d9_interop.h"

#define DLLEXPORT __declspec(dllexport)

struct VertexShaderIn
{
	float3 Pos;
	float3 Normal;
	float3 Color;
};

struct VertexShaderOut
{
	float3 Pos;
	float3 Normal;
	float3 Color;
	float3 ModelPos;
};

struct Triangle
{
	VertexShaderOut v1;
	VertexShaderOut v2;
	VertexShaderOut v3;

	bool Visible;

	int minX;
	int maxX;
	int minY;
	int maxY;
};

struct Fragment
{
	float3 Position;
	float3 Normal;
	float3 Color;
};

struct Model {
	int numOfVertices;
	int numOfFaces;
	VertexShaderIn* vertexBufferIn;
	VertexShaderOut* vertexBufferOut;
	Triangle* primitivesBuffer;
	int* indexBuffer;
};

extern "C"
{
	DLLEXPORT void Init(IDirect3DDevice9Ex* device);
	DLLEXPORT Model* CreateModel(float3* vertices, float3* normals, float3* colors, int* indices, int numOfVertices, int numOfFaces);
	DLLEXPORT void SetTransformation(float4* transformation, float3 camera);
	DLLEXPORT void FreeRasterizer();
	DLLEXPORT void FreeModel(Model* Model);
	DLLEXPORT void Resize(unsigned int w, unsigned int h, IDirect3DSurface9* backBufSurface);

	DLLEXPORT void Begin();
	DLLEXPORT void End();
	DLLEXPORT void DrawModel(Model* model);
}