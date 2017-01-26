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

int width = 1900;
int height = 1000;
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

#define MAX_MODELS 3
#define MAX_TEXTURES 3
RasterizerModel* Models[MAX_MODELS];
Texture* Textures[MAX_TEXTURES];
int currentModel, currentTexture;

static void GetColor(float3* color)
{
	CHOOSECOLOR cc;                 // common dialog box structure 
	static COLORREF acrCustClr[16]; // array of custom colors 
	static DWORD rgbCurrent;        // initial color selection

	rgbCurrent =
		((int)(color->x * 255) << 16) |
		((int)(color->y * 255) << 8) |
		(int)(color->z * 255);

	ZeroMemory(&cc, sizeof(cc));
	cc.lStructSize = sizeof(cc);
	cc.hwndOwner = NULL;
	cc.lpCustColors = (LPDWORD)acrCustClr;
	cc.rgbResult = rgbCurrent;
	cc.Flags = CC_FULLOPEN | CC_RGBINIT;

	if (ChooseColor(&cc) == TRUE)
	{
		color->x = (float)((cc.rgbResult >> 16) & 0xFF) / 255.f;
		color->y = (float)((cc.rgbResult >> 8) & 0xFF) / 255.f;
		color->z = (float)(cc.rgbResult & 0xFF) / 255.f;
	}
}

static void updateTexts()
{
	TextLine line;
	if (SceneParams.IsHelpEnabled)
	{
		if (SceneParams.BackCulling == BackCullingNone)
			line.length = sprintf_s(line.text, 128, "c  Backface Culling    Disabled");
		else if (SceneParams.BackCulling == BackCullingCW)
			line.length = sprintf_s(line.text, 128, "c  Backface Culling    Clockwise");
		else
			line.length = sprintf_s(line.text, 128, "c  Backface Culling    Counter Clockwise");

		SetLine(1, line);

		if (SceneParams.RenderMode == RenderModeWireframe)
			line.length = sprintf_s(line.text, 128, "r  Rendering Mode      Wireframe");
		else if (SceneParams.RenderMode == RenderModeTriangles)
			line.length = sprintf_s(line.text, 128, "r  Rendering Mode      Color");
		else
			line.length = sprintf_s(line.text, 128, "r  Rendering Mode      Texture");

		SetLine(2, line);

		if (SceneParams.RenderOutput == RenderOutputColor)
			line.length = sprintf_s(line.text, 128, "o  Rendering Output    Color");
		else if (SceneParams.RenderOutput == RenderOutputNormal)
			line.length = sprintf_s(line.text, 128, "o  Rendering Output    Normal Buffer");
		else
			line.length = sprintf_s(line.text, 128, "o  Rendering Output    Z Buffer");

		SetLine(3, line);

		if (SceneParams.LightEnabled)
			line.length = sprintf_s(line.text, 128, "l  Toggle Lightning    Enabled");
		else
			line.length = sprintf_s(line.text, 128, "l  Toggle Lightning    Disabled");

		SetLine(4, line);

		line.length = sprintf_s(line.text, 128, "a  Change Ambient Color");
		SetLine(5, line);
		line.length = sprintf_s(line.text, 128, "s  Change Specular Color");
		SetLine(6, line);
		line.length = sprintf_s(line.text, 128, "d  Change Diffuse Color");
		SetLine(7, line);

		line.length = sprintf_s(line.text, 128, "q+arrows  Change Ambient Constant    %f", SceneParams.LightAmbientConstant);
		SetLine(8, line);
		line.length = sprintf_s(line.text, 128, "w+arrows  Change Specular Constant   %f", SceneParams.LightSpecularConstant);
		SetLine(9, line);
		line.length = sprintf_s(line.text, 128, "e+arrows  Change Diffuse Constant    %f", SceneParams.LightDiffuseConstant);
		SetLine(10, line);
		line.length = sprintf_s(line.text, 128, "z+arrows  Change Light Shininess     %i", SceneParams.LightShininess);
		SetLine(11, line);

		line.length = sprintf_s(line.text, 128, "m  Change Model");
		SetLine(12, line);
		line.length = sprintf_s(line.text, 128, "m  Change Texture");
		SetLine(13, line);
		line.length = sprintf_s(line.text, 128, "F1 Toggle Help");
		SetLine(14, line);
		line.length = sprintf_s(line.text, 128, "LMB + Move  Rotate Model");
		SetLine(15, line);
		line.length = sprintf_s(line.text, 128, "MMB + Move  Move Light");
		SetLine(16, line);
		line.length = sprintf_s(line.text, 128, "RMB + Move  Move Model");
		SetLine(17, line);
	}
	else
	{
		line.length = 0;

		for (int i = 1; i < 16; i++)
		{
			SetLine(i, line);
		}
	}
}

static void error_callback(int error, const char* description)
{
	fprintf(stderr, "Error: %s\n", description);
}

bool isQDown, isWDown, isEDown, isZDown;

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
			
		}
		else if (key == GLFW_KEY_F1)
		{
			SceneParams.IsHelpEnabled = !SceneParams.IsHelpEnabled;
			updateTexts();
		}
		else if (key == GLFW_KEY_C)
		{
			if (SceneParams.BackCulling == BackCullingNone)
				SceneParams.BackCulling = BackCullingCW;
			else if (SceneParams.BackCulling == BackCullingCW)
				SceneParams.BackCulling = BackCullingCCW;
			else
				SceneParams.BackCulling = BackCullingNone;

			updateTexts();
		}
		else if (key == GLFW_KEY_R)
		{
			if (SceneParams.RenderMode == RenderModeWireframe)
				SceneParams.RenderMode = RenderModeTriangles;
			else if (SceneParams.RenderMode == RenderModeTriangles)
				SceneParams.RenderMode = RenderModeTexture;
			else
				SceneParams.RenderMode = RenderModeWireframe;

			updateTexts();
		}
		else if (key == GLFW_KEY_O)
		{
			if (SceneParams.RenderOutput == RenderOutputColor)
				SceneParams.RenderOutput = RenderOutputNormal;
			else if (SceneParams.RenderOutput == RenderOutputNormal)
				SceneParams.RenderOutput = RenderOutputZBuffer;
			else
				SceneParams.RenderOutput = RenderOutputColor;

			updateTexts();
		}
		else if (key == GLFW_KEY_L)
		{
			SceneParams.LightEnabled = !SceneParams.LightEnabled;

			updateTexts();
		}
		else if (key == GLFW_KEY_A)
		{
			GetColor(&SceneParams.LightAmbientColor);
		}
		else if (key == GLFW_KEY_D)
		{
			GetColor(&SceneParams.LightDiffuseColor);
		}
		else if (key == GLFW_KEY_S)
		{
			GetColor(&SceneParams.LightSpecularColor);
		}
		else if (key == GLFW_KEY_Q)
		{
			isQDown = true;
		}
		else if (key == GLFW_KEY_W)
		{
			isWDown = true;
		}
		else if (key == GLFW_KEY_E)
		{
			isEDown = true;
		}
		else if (key == GLFW_KEY_Z)
		{
			isZDown = true;
		}
		else if (key == GLFW_KEY_UP)
		{
			if (isQDown)
			{
				if (SceneParams.LightAmbientConstant < 1.f)
					SceneParams.LightAmbientConstant += 0.1f;
			}
			else if (isWDown)
			{
				if (SceneParams.LightSpecularConstant < 1.f)
					SceneParams.LightSpecularConstant += 0.1f;
			}
			else if (isEDown)
			{
				if (SceneParams.LightDiffuseConstant < 1.f)
					SceneParams.LightDiffuseConstant += 0.1f;
			}
			else if (isZDown)
			{
				SceneParams.LightShininess++;
			}

			updateTexts();
		}
		else if (key == GLFW_KEY_DOWN)
		{
			if (isQDown)
			{
				if (SceneParams.LightAmbientConstant > 0.f)
					SceneParams.LightAmbientConstant -= 0.1f;
			}
			else if (isWDown)
			{
				if (SceneParams.LightSpecularConstant > 0.f)
					SceneParams.LightSpecularConstant -= 0.1f;
			}
			else if (isEDown)
			{
				if (SceneParams.LightDiffuseConstant > 0.f)
					SceneParams.LightDiffuseConstant -= 0.1f;
			}
			else if (isZDown)
			{
				if (SceneParams.LightShininess > 0)
					SceneParams.LightShininess--;
			}

			updateTexts();
		}
		else if (key == GLFW_KEY_M)
		{
			currentModel = (currentModel + 1) % MAX_MODELS;

			if (currentModel = 0)
			{
				translation.z = -1;
			}
			else
			{
				translation.z = 1;
			}
		}
		else if (key == GLFW_KEY_T)
		{
			currentTexture = (currentTexture + 1) % MAX_TEXTURES;
		}
	}
	else if (action == GLFW_RELEASE)
	{
		if (key == GLFW_KEY_Q)
		{
			isQDown = false;
		}
		else if (key == GLFW_KEY_W)
		{
			isWDown = false;
		}
		else if (key == GLFW_KEY_E)
		{
			isEDown = false;
		}
		else if (key == GLFW_KEY_Z)
		{
			isZDown = false;
		}
	}
}

static void mouseMoveCallback(GLFWwindow *window, double x, double y)
{ 
	if (isLeftMousePressed) 
	{
		float easingFactor = 30;
		rotation = glm::quat(glm::vec3((lastMouseY - y) / easingFactor, (x - lastMouseX) / easingFactor, 0)) * rotation;
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
		lightRotation = glm::quat(glm::vec3((y - lastMouseY) / easingFactor, (lastMouseX - x) / easingFactor, 0)) * lightRotation;
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

BYTE* ConvertBMPToRGBBuffer(BYTE* Buffer, int width, int height)
{
	// first make sure the parameters are valid
	if ((NULL == Buffer) || (width == 0) || (height == 0))
		return NULL;

	// find the number of padding bytes

	int padding = 0;
	int scanlinebytes = width * 3;
	while ((scanlinebytes + padding) % 4 != 0)     // DWORD = 4 bytes
		padding++;
	// get the padded scanline width
	int psw = scanlinebytes + padding;

	// create new buffer
	BYTE* newbuf = new BYTE[width*height * 3];

	// now we loop trough all bytes of the original buffer, 
	// swap the R and B bytes and the scanlines
	long bufpos = 0;
	long newpos = 0;
	for (int y = 0; y < height; y++)
		for (int x = 0; x < 3 * width; x += 3)
		{
			newpos = y * 3 * width + x;
			bufpos = (height - y - 1) * psw + x;

			newbuf[newpos] = Buffer[bufpos + 2];
			newbuf[newpos + 1] = Buffer[bufpos + 1];
			newbuf[newpos + 2] = Buffer[bufpos];
		}

	return newbuf;
}

BYTE* LoadBMP(int* width, int* height, long* size, LPCTSTR bmpfile)
{
	// declare bitmap structures
	BITMAPFILEHEADER bmpheader;
	BITMAPINFOHEADER bmpinfo;
	// value to be used in ReadFile funcs
	DWORD bytesread;
	// open file to read from
	HANDLE file = CreateFile(bmpfile, GENERIC_READ, FILE_SHARE_READ,
		NULL, OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, NULL);
	if (NULL == file)
		return NULL; // coudn't open file


					 // read file header
	if (ReadFile(file, &bmpheader, sizeof(BITMAPFILEHEADER), &bytesread, NULL) == false)
	{
		CloseHandle(file);
		return NULL;
	}

	//read bitmap info

	if (ReadFile(file, &bmpinfo, sizeof(BITMAPINFOHEADER), &bytesread, NULL) == false)
	{
		CloseHandle(file);
		return NULL;
	}

	// check if file is actually a bmp
	if (bmpheader.bfType != 'MB')
	{
		CloseHandle(file);
		return NULL;
	}

	// get image measurements
	*width = bmpinfo.biWidth;
	*height = abs(bmpinfo.biHeight);

	// check if bmp is uncompressed
	if (bmpinfo.biCompression != BI_RGB)
	{
		CloseHandle(file);
		return NULL;
	}

	// check if we have 24 bit bmp
	if (bmpinfo.biBitCount != 24)
	{
		CloseHandle(file);
		return NULL;
	}


	// create buffer to hold the data
	*size = bmpheader.bfSize - bmpheader.bfOffBits;
	BYTE* Buffer = new BYTE[*size];
	// move file pointer to start of bitmap data
	SetFilePointer(file, bmpheader.bfOffBits, NULL, FILE_BEGIN);
	// read bmp data
	if (ReadFile(file, Buffer, *size, &bytesread, NULL) == false)
	{
		delete[] Buffer;
		CloseHandle(file);
		return NULL;
	}

	// everything successful here: close file and return buffer

	CloseHandle(file);

	return Buffer;
}

Texture* LoadTextureFromFile(LPCTSTR bmpfile)
{
	TCHAR  buffer[256] = TEXT("");
	TCHAR** lppPart = { NULL };
	GetFullPathName(bmpfile, 256, buffer, lppPart);

	int width, height;
	long size;
	BYTE* buf = LoadBMP(&width, &height, &size, buffer);
	buf = ConvertBMPToRGBBuffer(buf, width, height);

	float* tex = new float[height * width * 4];
	srand(time(NULL));

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			float r = (float)(buf[(y * width + x) * 3]) / 255;
			float g = (float)(buf[(y * width + x) * 3 + 1]) / 255;
			float b = (float)(buf[(y * width + x) * 3 + 2]) / 255;

			tex[(y * width + x) * 4] = b;
			tex[(y * width + x) * 4 + 1] = g;
			tex[(y * width + x) * 4 + 2] = r;
			tex[(y * width + x) * 4 + 3] = 0;
		}
	}

	return LoadTexture(width, height, tex);
}

time_t seconds;
static int fpstracker = 0;
static float fps = 0;

void updateFps()
{
	time_t seconds2 = time(NULL);

	fpstracker++;
	if (seconds2 - seconds >= 1) 
	{
		fps = (float)fpstracker / (seconds2 - seconds);
		fpstracker = 0;
		seconds = seconds2;

		TextLine line;
		line.length = sprintf_s(line.text, 128, "Current fps: %f", fps);
		SetLine(0, line);
	}
}

void initDefaults()
{
	SceneParams.BackCulling = BackCullingNone;
	SceneParams.RenderMode = RenderModeTriangles;
	SceneParams.RenderOutput = RenderOutputColor;

	SceneParams.LightEnabled = true;
	SceneParams.LightDiffuseColor.x = 1.f;
	SceneParams.LightDiffuseColor.y = 1.f;
	SceneParams.LightDiffuseColor.z = 1.f;

	SceneParams.LightSpecularColor.x = 1.f;
	SceneParams.LightSpecularColor.y = 1.f;
	SceneParams.LightSpecularColor.z = 1.f;

	SceneParams.LightAmbientColor.x = 1.f;
	SceneParams.LightAmbientColor.y = 1.f;
	SceneParams.LightAmbientColor.z = 1.f;

	SceneParams.LightAmbientConstant = 0.0;
	SceneParams.LightSpecularConstant = 0.5;
	SceneParams.LightDiffuseConstant = 0.5;

	SceneParams.LightShininess = 127;

	SceneParams.IsHelpEnabled = true;

	Models[0] = CreateModel(ModelLoader::LoadModel("Assets\\teapot.off"));
	Models[1] = CreateModel(ModelLoader::LoadModel("Assets\\mushroom.off"));
	Models[2] = CreateModel(ModelLoader::LoadModel("Assets\\test.off"));

	Textures[0] = LoadTextureFromFile("..\\Debug\\Assets\\texture.bmp");
	Textures[1] = LoadTextureFromFile("..\\Debug\\Assets\\texture2.bmp");
	Textures[2] = LoadTextureFromFile("..\\Debug\\Assets\\texture3.bmp");
}

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
	
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);

	glBindTexture(GL_TEXTURE_2D, texture);
	glEnable(GL_TEXTURE_2D);

	initDefaults();
	updateTexts();

	while (!glfwWindowShouldClose(window)) 
	{
		updateFps();

		SetTransformation(glm::translate(ViewMatrix, glm::vec3(translation)) * glm::toMat4(rotation));
		SetCameraPosition(-translation * glm::toMat4(rotation));
		SetLightPosition(lightPosition * glm::toMat4(lightRotation));

		Begin();
		DrawModel(Models[currentModel], Textures[currentTexture]);

		End();

		glBegin(GL_QUADS);
		glTexCoord2f(0.0, 1.0); glVertex3f(-1.0, -1.0, 0.5);
		glTexCoord2f(1.0, 1.0); glVertex3f(1.0, -1.0, 0.5);
		glTexCoord2f(1.0, 0.0); glVertex3f(1.0, 1.0, 0.5);
		glTexCoord2f(0.0, 0.0); glVertex3f(-1.0, 1.0, 0.5);
		glEnd();

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	for (int i = 0; i < MAX_MODELS; i++) 
	{
		FreeModel(Models[i]);
	}

	for (int i = 0; i < MAX_TEXTURES; i++)
	{
		FreeTexture(Textures[i]);
	}
	
	FreeRasterizer();

	glfwDestroyWindow(window);
	glfwTerminate();
}
