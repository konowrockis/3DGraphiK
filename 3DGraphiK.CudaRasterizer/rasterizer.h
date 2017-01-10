#pragma once
#include <stdio.h>
#include "device_launch_parameters.h"

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
extern "C"
{
	DLLEXPORT void Init(float3* vertices, float3* normals, float3* colors, int* indices, int numberOfVertices, int numberOfFaces);
	DLLEXPORT void SetTransformation(float4* transformation, float3 camera);
	DLLEXPORT void Rasterize();
	DLLEXPORT void Resize(int w, int h, int* buf);
}