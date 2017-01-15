#pragma once
#include "structs.h"

class ModelLoader
{
private:
	static void CreateColors(Model* model);
	static void LoadFromFile(Model* model, const char* fileName);
	static void ComputeNormals(Model* model);

public:
	static Model* LoadModel(const char* fileName);
};