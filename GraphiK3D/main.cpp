#define GLEW_STATIC
#include "GL/glew.h"
#include "GLFW/glfw3.h"
#include "GLM/glm.hpp"
#include "GLM/vec3.hpp"
#include "GLM/vec4.hpp"
#include "GLM/mat4x4.hpp"
#include "GLM/gtc/matrix_transform.hpp"
#include "GLM/gtc/quaternion.hpp"
#include "GLM/gtx/quaternion.hpp"

#include <stdlib.h>
#include <stdio.h>
#include "rasterizer.h"
#include <time.h>

#include "ModelLoader.h"
#include <Windows.h>

int width = 640;
int height = 480;
bool isLeftMousePressed = false;
bool isRightMousePressed = false;
bool isMiddleMousePressed = false;
double lastMouseX = 0;
double lastMouseY = 0;

glm::quat rotation;
glm::vec4 translation = glm::vec4(0, -1, 6, 1);
glm::mat4x4 ViewMatrix;

glm::vec4 lightPosition = glm::vec4(0, 0, -30, 1);
glm::quat lightRotation;

static void error_callback(int error, const char* description)
{
	fprintf(stderr, "Error: %s\n", description);
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (action == GLFW_PRESS)
	{
		if (key == GLFW_KEY_ESCAPE)
		{
			glfwSetWindowShouldClose(window, GLFW_TRUE);
		}
		else if (key == GLFW_KEY_SPACE)
		{
			CHOOSECOLOR cc;                 // common dialog box structure 
			static COLORREF acrCustClr[16]; // array of custom colors 
			HBRUSH hbrush;                  // brush handle
			static DWORD rgbCurrent;        // initial color selection

			ZeroMemory(&cc, sizeof(cc));
			cc.lStructSize = sizeof(cc);
			cc.hwndOwner = NULL;
			cc.lpCustColors = (LPDWORD)acrCustClr;
			cc.rgbResult = rgbCurrent;
			cc.Flags = CC_FULLOPEN | CC_RGBINIT;

			if (ChooseColor(&cc) == TRUE)
			{
				hbrush = CreateSolidBrush(cc.rgbResult);
				rgbCurrent = cc.rgbResult;
			}
		}
	}
}

static void mouseMoveCallback(GLFWwindow *window, double x, double y)
{ 
	if (isLeftMousePressed) 
	{
		float easingFactor = 30;
		rotation = glm::quat(glm::vec3((lastMouseY - y) / easingFactor, (lastMouseX - x) / easingFactor, 0)) * rotation;
	}
	else if (isRightMousePressed) 
	{
		if (isMiddleMousePressed)
		{
			float newVal = lightPosition.z + (y - lastMouseY) / 30;

			if (newVal > -40 && newVal < 40)
			{
				lightPosition.z = newVal;
			}
		}
		else
		{
			float newVal = translation.z + (y - lastMouseY) / 50;

			if (newVal > 1 && newVal < 30)
			{
				translation.z = newVal;
			}
		}
	}
	else if (isMiddleMousePressed) 
	{
		float easingFactor = 50;
		lightRotation = glm::quat(glm::vec3((y - lastMouseY) / easingFactor, (x - lastMouseX) / easingFactor, 0)) * lightRotation;
	}

	lastMouseX = x;
	lastMouseY = y;
}

static void mousePressCallback(GLFWwindow *window, int button, int action, int mods)
{
	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) 
	{
		isLeftMousePressed = true;
	}
	else if (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS) 
	{
		isMiddleMousePressed = true;
	}
	else if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) 
	{
		isRightMousePressed = true;
	}
	else if (action == GLFW_RELEASE) 
	{
		isLeftMousePressed = isRightMousePressed = isMiddleMousePressed = false;
	}
}

GLuint texture;

glm::mat4 getViewMatrix()
{
	glm::mat4 viewScale = glm::scale(glm::mat4(1.f), glm::vec3(width / 2, height / 2, 1));
	glm::mat4 viewMove = glm::translate(viewScale, glm::vec3(1, 1, 0));
	glm::mat4 projectionMatrix = glm::perspective(glm::radians(60.f), (float)width / (float)height, 1.f, 100.f);

	return viewMove * projectionMatrix;
}

void initCuda()
{
	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);

	// set basic parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	// Create texture data (4-component unsigned byte)
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

	// Unbind the texture
	glBindTexture(GL_TEXTURE_2D, 0);

	Init();
	Resize(width, height, texture);
}

time_t seconds;

int main()
{
	glfwSetErrorCallback(error_callback);

	if (!glfwInit())
		exit(EXIT_FAILURE);

	GLFWwindow* window = glfwCreateWindow(width, height, "3DGraphiK", /*glfwGetPrimaryMonitor()*/NULL, NULL);

	if (!window)
	{
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	glfwMakeContextCurrent(window);
	glfwSetKeyCallback(window, key_callback);
	glfwSetCursorPosCallback(window, mouseMoveCallback);
	glfwSetMouseButtonCallback(window, mousePressCallback);
	//glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK) {
		return 0;
	}

	glfwSwapInterval(0);

	initCuda();
	ViewMatrix = getViewMatrix();

	seconds = time(NULL);
	int fpstracker = 0;
	float fps = 0;

	glDisable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);

	glBindTexture(GL_TEXTURE_2D, texture);
	glEnable(GL_TEXTURE_2D);

	RasterizerModel* g = CreateModel(ModelLoader::LoadModel("Assets\\mushroom.off"));
	RasterizerModel* m = CreateModel(ModelLoader::LoadModel("Assets\\teapot.off"));

	while (!glfwWindowShouldClose(window)) 
	{
		time_t seconds2 = time(NULL);

		fpstracker++;
		if (seconds2 - seconds >= 1) {

			fps = fpstracker / (seconds2 - seconds);
			fpstracker = 0;
			seconds = seconds2;

			printf("%f\n", fps);
		}

		SetTransformation(glm::translate(ViewMatrix, glm::vec3(translation)) * glm::toMat4(rotation));
		SetCameraPosition(-translation * glm::toMat4(rotation));
		SetLightPosition(lightPosition * glm::toMat4(lightRotation));

		Begin();
		DrawModel(m);

		End();

		glBegin(GL_QUADS);
		glTexCoord2f(1.0, 1.0); glVertex3f(-1.0, -1.0, 0.5);
		glTexCoord2f(0.0, 1.0); glVertex3f(1.0, -1.0, 0.5);
		glTexCoord2f(0.0, 0.0); glVertex3f(1.0, 1.0, 0.5);
		glTexCoord2f(1.0, 0.0); glVertex3f(-1.0, 1.0, 0.5);
		glEnd();

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	FreeModel(m);
	FreeModel(g);
	FreeRasterizer();

	glfwDestroyWindow(window);
	glfwTerminate();
}
