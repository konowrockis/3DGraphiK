#define GLEW_STATIC
#include "GL/glew.h"
#include "GLFW/glfw3.h"
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>

#include <stdlib.h>
#include <stdio.h>

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

void DisplayImage(GLuint texture, unsigned int x, unsigned int y, unsigned int width, unsigned int height)
{
	
}

static cudaArray_t framebuf;
GLuint texture;

static int data[480 * 640];

void initCuda()
{
	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);

	for (int i = 0; i < 480 * 640; i++)
	{
		data[i] = 0xFFFF0000;
	}

	// set basic parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	// Create texture data (4-component unsigned byte)
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 640, 480, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);

	// Unbind the texture
	glBindTexture(GL_TEXTURE_2D, 0);

	CUgraphicsResource* resource;

	//cuGraphicsGLRegisterImage(&resource, texture, GL_TEXTURE_3D, CU_GRAPHICS_REGISTER_FLAGS_NONE);
	/*cuGraphicsMapResources(1, resource, 0);
	cuGraphicsSubResourceGetMappedArray(&framebuf, resource, 0, 0);
	cuGraphicsUnmapResources(1, resource, 0);*/
}

int main()
{
	glfwSetErrorCallback(error_callback);

	if (!glfwInit())
		exit(EXIT_FAILURE);

	GLFWwindow* window = glfwCreateWindow(640, 480, "3DGraphiK", NULL, NULL);

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

	glfwSwapInterval(1);

	initCuda();

	glDisable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);

	while (!glfwWindowShouldClose(window)) {
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
		glViewport(0, 0, 640, 480);

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
