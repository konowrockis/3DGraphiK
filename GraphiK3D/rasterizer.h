#pragma once
#include <stdio.h>
#include "device_launch_parameters.h"
#include "structs.h"
#include "GLM/vec3.hpp"
#include "GLM/vec4.hpp"
#include "GLM/mat4x4.hpp"

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
void DrawModel(RasterizerModel* model);
