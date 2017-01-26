#pragma once
#include "device_launch_parameters.h"

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

	float minx, miny, maxx, maxy;
};

struct Fragment
{
	float3 Position;
	float3 Normal;
	float3 Color;
};

struct Model 
{
	int numOfVertices;
	int numOfFaces;

	int* indices;
	float3* vertices;
	float3* normals;
	float3* colors;
};

struct RasterizerModel
{
	int numOfVertices;
	int numOfFaces;

	VertexShaderIn* vertexBufferIn;
	int* indexBuffer;

	VertexShaderOut* vertexBufferOut;
	Triangle* primitivesBuffer;
};

struct Texture {
	cudaArray* cuArray;
	cudaTextureObject_t Tex;
};

