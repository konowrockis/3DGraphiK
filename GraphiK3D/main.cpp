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

int width = 640;
int height = 480;
bool isLeftMousePressed = false;
bool isRightMousePressed = false;
bool isMiddleMousePressed = false;
double lastMouseX = 0;
double lastMouseY = 0;

glm::quat rotation;
glm::vec3 translation(0, -1, 6);
glm::mat4x4 ViewMatrix;

static void error_callback(int error, const char* description)
{
	fprintf(stderr, "Error: %s\n", description);
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GLFW_TRUE);
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
		float newVal = translation.z + (y - lastMouseY) / 50;

		if (newVal > 1 && newVal < 30)
		{
			translation.z = newVal;

			/*glm::mat4 ViewTranslate = glm::translate(glm::mat4(1.f), translation);
			glm::mat4 rot = glm::toMat4(rotation);
			glm::mat4 res = ViewScale * ViewMv * Proj * ViewTranslate * rot;

			SetTransformation(res, glm::vec4(-translation, 1) * rot);*/
		}
	}
	else if (isMiddleMousePressed) 
	{

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
	glm::mat4 projectionMatrix = glm::perspective(glm::radians(60.f), (float)width / (float)height, 0.1f, 100.f);

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

	GLFWwindow* window = glfwCreateWindow(width, height, "3DGraphiK", NULL, NULL);

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

		glm::mat4 translate = glm::translate(ViewMatrix, translation);
		glm::mat4 rotate = glm::toMat4(rotation);
		SetTransformation(translate * rotate, glm::vec4(-translation, 1) * rotate);

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

	glfwDestroyWindow(window);
	glfwTerminate();
}
