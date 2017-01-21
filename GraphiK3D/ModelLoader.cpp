#include "ModelLoader.h"
#include "VectorUtils.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

Model* ModelLoader::LoadModel(const char* fileName)
{
	Model* model = new Model;

	LoadFromFile(model, fileName);
	CreateColors(model);
	ComputeNormals(model);
	
	return model;
}

void ModelLoader::CreateColors(Model* model)
{
	srand(time(NULL));

	model->colors = new float3[model->numOfVertices];

	for (int i = 0; i < model->numOfVertices; i++)
	{
		//model->colors[i] = make_float3((float)rand() / RAND_MAX, (float)rand() / RAND_MAX, (float)rand() / RAND_MAX);
		model->colors[i] = make_float3(0.85, 0.2, 0.2);

	}
}

void ModelLoader::LoadFromFile(Model* model, const char* fileName)
{
	FILE* modelFile = fopen(fileName, "r");

	char buf[255];

	int mode = 0;
	int iv = 0, ii = 0;

	while (fscanf(modelFile, "%[^\n]\n", buf) != EOF && mode <= 2)
	{
		if (buf[0] == '\r' || buf[0] == '\n' || buf[0] == '\0' || buf[0] == '\#')
		{
			continue;
		}

		switch (mode)
		{
		case 0:
			if (buf[0] == 'O') continue;

			int edgesCount;

			sscanf(buf, "%i %i %i", &model->numOfVertices, &model->numOfFaces, &edgesCount);

			model->vertices = new float3[model->numOfVertices];
			model->indices = new int[model->numOfFaces * 3];

			mode = 1;
			break;

		case 1:
			float x, y, z;

			sscanf(buf, "%f %f %f", &x, &y, &z);
			model->vertices[iv++] = make_float3(x, y, z);

			if (iv >= model->numOfVertices)
			{
				mode = 2;
			}
			break;

		case 2:
			int i1, i2, i3;
			int numOfInd;

			sscanf(buf, "%i %i %i %i", &numOfInd, &i1, &i2, &i3);
			model->indices[ii++] = i1;
			model->indices[ii++] = i2;
			model->indices[ii++] = i3;

			if (ii >= model->numOfFaces * 3)
			{
				mode = 3;
			}
			break;
		}
	}

	fclose(modelFile);
}

void ModelLoader::ComputeNormals(Model* model)
{
	float3* normals = new float3[model->numOfFaces];
	model->normals = new float3[model->numOfVertices];

	for (int i = 0; i < model->numOfFaces; i++)
	{
		float3 v1 = model->vertices[model->indices[i * 3 + 1]] - model->vertices[model->indices[i * 3]];
		float3 v2 = model->vertices[model->indices[i * 3 + 2]] - model->vertices[model->indices[i * 3]];

		normals[i] = norm(cross(v2, v1));
	}

	for (int i = 0; i < model->numOfVertices; i++)
	{
		float3 v = make_float3(0, 0, 0);
		int count = 0;

		for (int j = 0; j < model->numOfFaces * 3; j++)
		{
			if (model->indices[j] == i)
			{
				v = v + normals[j / 3];
				count++;
			}
		}

		if (count >= 1)
		{
			float3 a = norm(v / count);
			model->normals[i] = norm(v / count);
		}
		else
		{

		}
	}

	delete normals;
}