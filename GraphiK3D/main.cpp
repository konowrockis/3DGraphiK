#define GLEW_STATIC
#include "GL/glew.h"
#include "GLFW/glfw3.h"
#include "GLM/glm.hpp"
#include "GLM/vec3.hpp"
#include "GLM/vec4.hpp"
#include "GLM/mat4x4.hpp"
#include "GLM/gtc/matrix_transform.hpp"

#include <stdlib.h>
#include <stdio.h>
#include "rasterizer.h"
#include <time.h>

#include "ModelLoader.h"

int width = 1900;
int height = 1060;

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

}

static  void mousePressCallback(GLFWwindow *window, int button, int action, int mods)
{
	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {

	}
	else if (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS) {

	}
	else if (action == GLFW_RELEASE) {

	}
}

GLuint texture;

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

	glfwSetKeyCallback(window, key_callback);
	glfwSetCursorPosCallback(window, mouseMoveCallback);
	glfwSetMouseButtonCallback(window, mousePressCallback);
	glfwMakeContextCurrent(window);

	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK) {
		return 0;
	}

	glfwSwapInterval(0);

	initCuda();

	seconds = time(NULL);
	int fpstracker = 0;
	float fps = 0;

	glDisable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);
	
	RasterizerModel* m = CreateModel(ModelLoader::LoadModel("Assets\\teapot.off"));

	glm::vec3 translation(0, -1, 12);
	glm::vec3 sc(width / 2, height / 2, 1);
	glm::vec3 mv(1, 1, 0);

	glm::mat4 Proj = glm::perspective(glm::radians(60.f), 16.f / 9.f, 0.1f, 100.f);
	glm::mat4 ViewTranslate = glm::translate(glm::mat4(1.f), translation);
	
	glm::mat4 ViewMv = glm::translate(glm::mat4(1.f), mv);
	glm::mat4 ViewScale = glm::scale(glm::mat4(1.f), sc);
	
	glm::mat4 res = ViewScale * ViewMv * Proj * ViewTranslate;

	SetTransformation(res, -translation);

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

		Begin();
		DrawModel(m);
		End();

		glBindTexture(GL_TEXTURE_2D, texture);
		glEnable(GL_TEXTURE_2D);

		glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

		glMatrixMode(GL_PROJECTION);
		glPushMatrix();
		glLoadIdentity();
		glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		glPushAttrib(GL_VIEWPORT_BIT);
		glViewport(0, 0, width, height);

		glBegin(GL_QUADS);
		glTexCoord2f(0.0, 0.0); glVertex3f(-1.0, -1.0, 0.5);
		glTexCoord2f(1.0, 0.0); glVertex3f(1.0, -1.0, 0.5);
		glTexCoord2f(1.0, 1.0); glVertex3f(1.0, 1.0, 0.5);
		glTexCoord2f(0.0, 1.0); glVertex3f(-1.0, 1.0, 0.5);
		glEnd();

		glPopAttrib();

		glMatrixMode(GL_PROJECTION);
		glPopMatrix();

		glDisable(GL_TEXTURE_2D);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glfwDestroyWindow(window);
	glfwTerminate();
}
