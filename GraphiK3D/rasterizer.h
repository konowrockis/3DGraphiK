#pragma once
#include <stdio.h>
#include "device_launch_parameters.h"
#include "structs.h"
#include "GLM/vec3.hpp"
#include "GLM/vec4.hpp"
#include "GLM/mat4x4.hpp"

struct TextLine {
	int length;
	char text[128];
};

void Init();
RasterizerModel* CreateModel(Model* model);

void SetTransformation(glm::mat4x4 transformation);
void SetLightParams(glm::vec3 diffuseColor, glm::vec3 specularColor, glm::vec3 ambientColor, float diffuseConstant, float specularConstant, float ambientConstant, int shininess);
void SetCameraPosition(glm::vec3 camera);
void SetLightPosition(glm::vec3 lightPosition);

void FreeRasterizer();
void FreeModel(RasterizerModel* Model);
void Resize(unsigned int w, unsigned int h, GLuint framebufferTexture);

void Begin();
void End();
void DrawModel(RasterizerModel* model, Texture* texture);

Texture* LoadTexture(int textureWidth, int textureHeight, void* srcTexture);
void FreeTexture(Texture* texture);

void SetLine(int i, TextLine line);

enum RenderModeType { RenderModeWireframe, RenderModeTriangles, RenderModeTexture };
enum BackFaceCullingType { BackCullingCCW, BackCullingCW, BackCullingNone };
enum RenderOutputType { RenderOutputColor, RenderOutputNormal, RenderOutputZBuffer };

struct SceneParams_t
{
	float3 CameraPosition;
	float3 LightPosition;
	float4* Transformation;

	BackFaceCullingType BackCulling;
	RenderModeType RenderMode;
	RenderOutputType RenderOutput;

	bool LightEnabled;
	float3 LightDiffuseColor;
	float3 LightSpecularColor;
	float3 LightAmbientColor;
	float LightDiffuseConstant;
	float LightSpecularConstant;
	float LightAmbientConstant;
	int LightShininess;

	bool IsHelpEnabled;
};

extern SceneParams_t SceneParams;