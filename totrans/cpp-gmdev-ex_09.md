# Getting Started with OpenGL

In the previous three chapters, we rendered 2D objects called sprites in our tiny Bazooka game using the **Simple and Fast Media Library** (**SFML**). At the core of SFML is OpenGL; this is used to render anything on screen, including 2D objects.

SFML does a great job of putting everything in a nice little package, and this allows us to get a 3D game going very quickly. However, in order to understand how a graphics library actually works, we need to learn how OpenGL works by delving deeper into how to use it so that we can render anything on the screen.

In this chapter, we will discover how to use a graphics library, such as OpenGL, in order to render 3D objects in any scene. We will cover the following topics:

*   What is OpenGL?
*   Creating our first OpenGL project
*   Creating a window and ClearScreen
*   Creating a `Mesh` class
*   Creating a Camera class
*   The Shaderloader class
*   The Light Renderer class
*   Drawing the object

# What is OpenGL?

So, what is this OpenGL that we speak of? Well, OpenGL is a collection of graphics APIs; essentially, this is a collection of code that allows you to gain access to the features of your graphics hardware. The current version of OpenGL is 4.6, but any graphics hardware that is capable of running OpenGL 4.5 can run 4.6 as well.

OpenGL is entirely hardware and operating system independent, so it doesn't matter if you have a NVIDIA or AMD GPU; it will work the same on both hardware. The way in which OpenGL's features work is defined by a specification that is used by graphics hardware manufacturers while they're developing the drivers for their hardware. This is why we sometimes have to update the graphics hardware drivers if something doesn't look right or if the game is not performing well.

Furthermore, OpenGL runs the same, regardless of whether you are running a Windows or a Linux machine. It is, however, deprecated on macOS Mojave, but if you are running a macOS version earlier than Mojave, then it is still compatible.

OpenGL is only responsible for rendering objects in the scene. Unlike SFML, which allows you to create a window and then gain access to the keyboard and mouse input, we will need to add a separate library that will handle all of this for us.

So, let's start preparing our project by rendering a 3D OpenGL object in the scene.

# Creating our first OpenGL project

Now that we have gained an understanding of what OpenGL is, let's examine how to create our first OpenGL project, as follows:

1.  Create a new empty C++ project in Visual Studio and call it `OpenGLProject`.
2.  Then, download GLEW; this is a C/C++ extension loader library. OpenGL supports extensions that various GPU vendors can use to write and extend the functionality of OpenGL. This library will determine what extensions are supported on the platform.
3.  Go to [http://glew.sourceforge.net/](http://glew.sourceforge.net/) and download the Windows 32-bit and 64-bit Binaries:

![](img/0ce25f6f-52f8-4912-b6c2-83fcc5d6d88b.png)

4.  Next, we need to download GLFW; this is a platform-independent API that is used for creating a window, reading inputs, and handling events. Go to [https://www.glfw.org/download.html](https://www.glfw.org/download.html) and download the 64-bit Windows binary. In this book, we will be primarily looking at implementing it on the Windows platform:

![](img/f22aa0a8-f6d0-4268-8cc5-ff9c92012def.png)

5.  Next, we need to download `glm`, which is used to do all the math for our graphics calculations. Go to [https://glm.g-truc.net/0.9.9/index.html](https://glm.g-truc.net/0.9.9/index.html) and download GLM from the site.
6.  Now that we have downloaded all the required libraries and headers, we can start adding them to our project.
7.  In the root directory (where the Visual Studio project file is stored) of the project, create a new directory called `Dependencies`.
8.  From this directory, extract `glew`, `glfw`, and `glm`; the `Dependencies` directory should now look as follows:

![](img/ebf9e9ff-bd57-480d-befd-112fa32dbe8d.png)

9.  Open the Visual Studio project. We need to set the location of the headers and library files. To do this, open the project properties of `OpenGLProject` and set Configuration to Release and Platform to x64.

10.  Under C/C++ | General, select Additional Include Directories and select the following directories for GLEW and GLFW:

![](img/1cc6ccbb-4723-4054-a044-cbbc56c15d22.png)

11.  Next, under Linker |General, select Additional Library Directories, and then select the location of the `.lib` files in the `glew` and `glfw` directories, as follows:

![](img/9d5322e4-665a-459d-91c9-b4b681fed335.png)

12.  Next, we have to go to Linker | Input and specify which `.lib` files we are using.
13.  Under Linker | Input, select Additional Dependencies and then add opengl32.lib, glfw3.lib, and glew32.lib, as follows:

![](img/e1bdafc5-3f98-4608-a86b-de19c87cccdc.png)

14.  Although we didn't specifically download `opengl32.lib`, it is included when you update the driver of the graphics hardware. Therefore, make sure that you are running the most recent drivers for your GPU; if not, download them from the manufacturer's website.
15.  Finally, we have to add the `glew32.dll` and `glfw3.dll` files to the root directory of the project. `glew32.dll` is inside `glew-2.1.0/ bin/Release/64`, whereas `glfw3.dll` is inside `glfw-3.2.1/lib-vc2015`.
16.  The root directory of the project file should now look as follows:

![](img/479c3905-64d5-430c-a95b-7fd2f332dd29.png)

17.  With this out of the way, we can finally start working on the project.

# Creating a window and ClearScreen

Now, let's explore how we can work with the OpenGL project that we created:

1.  The first thing we have to do is create a window so that we can start rendering the game objects to the screen.
2.  Create a new `.cpp` file; Visual Studio will automatically call this `source.cpp`, so keep it as it is.
3.  At the top of the file, include the `glew` and `glfw` headers. Make sure that you include `glew.h` first since it contains the correct OpenGL header files to be included:

```cpp
#include <GL/glew.h> 
#include <GLFW/glfw3.h> 

Then create a main function and add the following to it. 

int main(int argc, char **argv) 
{ 

   glfwInit(); 

   GLFWwindow* window = glfwCreateWindow(800, 600,
    " Hello OpenGL ", NULL, NULL); 

   return 0; 
} 
```

4.  The first thing we need to do here is initialize `glfw` by calling `glfwInit()`.
5.  Once initialized, we can create the window that our game scene will be rendered to. To create a window, we need to create a new instance of `GLFWWindow` called window and call `glfwCreateWindow`. This takes five parameters, including the width and height of the window, along with the name of the window. The final two parameters—`monitor` and `share`—are set to `NULL`. The `monitor` parameter takes a specific monitor on which the window will be created. If it is set to `null`, then the default monitor is chosen. The `share` parameter will let us share the window resource with our users. Here, we set it to `NULL` as we don't want to share the window resources.
6.  Now, run the project; you will see that a window briefly appears before the application closes.
7.  Well, that's not very fun. Let's add the rest of the code so that we can see something being rendered on the viewport.
8.  The first thing we need to do is initialize OpenGL Context. OpenGL Context is a collection of all the current states of OpenGL. We will discuss the different states in the upcoming sections.
9.  To do this, call `glfwMakeCurrentContext` and pass in the window that we just created:

```cpp
glfwMakeContextCurrent(window);     
```

10.  We can now initialize GLEW by calling `glewInit()`.

11.  Next, we will add the following code between `glewInit()` and `return 0` in the `main` function:

```cpp
   while (!glfwWindowShouldClose(window)){ 

               // render our scene 

         glfwSwapBuffers(window); 
         glfwPollEvents(); 
   } 

         glfwTerminate();
```

12.  Here, we are creating a `while` loop, calling `glfwWindowShouldClose`, and then passing it in the current window. While the window is open, the `glfwSwapBuffers(window);` and `glfwPollEvents();` commands will be executed.
13.  In the `while` loop, we will render our scene. Then, we will swap display buffers. The display buffer is where the current frame is rendered and stored. While the current frame is being shown, the next frame is actually being rendered in the background, which we don't get to see. When the next frame is ready, the current frame is swapped with the new frame. This swapping of frames is done by `glfwSwapBuffer` and is managed by OpenGL.
14.  After we swap the display buffer, we need to check for any events that were triggered, such as the window being closed in `glfwPollEvents()`. Once the window is closed, `glfw` is terminated.
15.  If you run the project now, you will see a black window; while it doesn't vanish, it is still not very impressive. We can use OpenGL to clear the viewport with a color of our choice, so let's do that next.
16.  Create a new function called `void renderScene()`. Whatever we render to the scene from now on will also be added to this function. Add a new prototype for `void renderScene()` to the top of the `source.cpp` file.
17.  In the `renderScene` function, add the following lines of code:

```cpp
void renderScene(){ 

   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 
   glClearColor(1.0, 0.0, 0.0, 1.0);//clear yellow 

   // Draw game objects here 
}
```

In the first function, we call `glClear()`. All OpenGL functions start with the `gl` prefix; the `glClear` function clears the buffer. In this case, we are asking OpenGL to clear the color buffer and the depth buffer. The color buffer is where all the color information is stored for the scene. The depth buffer stores whichever pixel is in front; this means that if a pixel is behind another pixel, then that pixel will not be stored. This is especially important for 3D scenes, where some objects can be behind other objects and get occluded by the objects that are in front of it. We only require the pixel information regarding the objects that are in front as we will only get to see those objects and not the objects that are behind them.

Next, we call the `glClearColor` function and pass in an RGBA value; in this case, red. The `glCearColor` function clears the color buffer with the specific color in every frame. The buffers need to be cleared in every frame; otherwise, the previous frame will be overwritten with the image in the current frame. Imagine this to be like clearing the blackboard before drawing anything on it in every frame.

The depth buffer is also cleared after every frame using a default white color. This means that we don't have to clear it manually as this will be done by default.

18.  Now, call `renderScene` before swapping the buffer and run the project again. You should see a nice yellow viewport, as follows:

![](img/f1fccce8-99bb-4500-94d0-a0b3602ac427.png)

Before drawing the objects, we have to create some additional classes that will help us define the shape that we want to draw. We also need to create a camera class in order to set up a virtual camera through which we can view the scene. Furthermore, we need to write a basic vertex, a `shader` fragment, and a `Shaderloader` class, which will create a `shader` program that we can use to render our shape.

First, let's create the `Mesh` class, which is where we will define the different shapes that we want to draw.

# Creating a Mesh class

The following steps explain how to create a `Mesh` class:

1.  Create new `.h` and `.cpp` files called `Mesh.h` and `Mesh.cpp`, respectively. These will be used to create a new `Mesh` class. In the `Mesh.h` file, add the following code:

```cpp
#include <vector> 
#include "Dependencies/glm/glm/glm.hpp" 

enum MeshType { 

   kTriangle = 0, 
   kQuad = 1, 
   kCube = 2, 
   kSphere = 3 

}; 

struct Vertex { 

   glm::vec3 pos; 
   glm::vec3 normal; 
   glm::vec3 color; 
   glm::vec2 texCoords; 

}; 

class Mesh { 

public: 
   static void setTriData(std::vector<Vertex>& vertices, 
     std::vector<uint32_t>&indices); 
   static void setQuadData(std::vector<Vertex>& vertices,    
     std::vector<uint32_t>&indices); 
   static void setCubeData(std::vector<Vertex>& vertices, 
     std::vector<uint32_t>&indices); 
   static void setSphereData(std::vector<Vertex>& vertices, 
     std::vector<uint32_t>&indices); 

};
```

2.  At the top of the `Mesh.h` file, we include a vector so that we can store points in a vector and include `glm.hpp`. This will help us define points in the space using the `vec3` variable.

3.  Then, we create a new `enum` type called `MeshType` and create four types: `Mesh Triangle`, `Quad`, `Cube`, and `Sphere`. We've done this so that we can specify the kind of mesh we are using and so the data will be populated accordingly.
4.  Next, we create a new `struct` type called `Vertex`, which has `vec3` properties called `pos`, `Color`, and `Normal`, and a `vec2` property called `textCoords`.

Each vertex has certain properties, such as `Position`, `Color`, `Normal`, and `Texture Coordinate`. `Position` and `Color` store the position and color information for each vertex, respectively. `Normal` specifies which direction the normal attribute is pointing to while `Texture Coordinate` specifies how a texture needs to be laid out. We will cover the normal and texture coordinate attributes when we cover lighting and how to apply textures to our objects.

5.  Then, the `Mesh` class is created. This has four functions, which are for setting the vertex and the index data per vertex.
6.  In the `Mesh.cpp` file, we include the `Mesh.h` file and then set the data for the four shapes. Here is an example of how `setTriData` sets the values for the vertices and indices:

```cpp
#include "Mesh.h" 

void Mesh::setTriData(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices) { 

   std::vector<Vertex> _vertices = { 

{ { 0.0f, -1.0f, 0.0f },          // Position 
{ 0.0f, 0.0f, 1.0 },              // Normal 
{ 1.0f, 0.0f, 0.0 },              // Color 
{ 0.0, 1.0 }                      // Texture Coordinate 
},                                // 0 

         { { 1.0f, 1.0f, 0.0f },{ 0.0f, 0.0f, 1.0 },{ 0.0f, 1.0f, 
          0.0 },{ 0.0, 0.0 } }, // 1 

         { { -1.0f, 1.0f, 0.0f },{ 0.0f, 0.0f, 1.0 },{ 0.0f, 0.0f, 
          1.0 },{ 1.0, 0.0 } }, // 2 
   }; 

   std::vector<uint32_t> _indices = { 
         0, 1, 2, 
   }; 

   vertices.clear(); indices.clear(); 

   vertices = _vertices; 
   indices = _indices; 
} 
```

7.  For each of the three vertices of the triangle, we set the position, normal, color, and texture coordinate information in the `vertices` vector.

Next, we set the indices in the `indices` vector. For definitions of the other functions, you can refer to the project that comes with this book. Then, we set the `_vertices` and `_indices` vectors to the reference vertices and indices, respectively.

# Creating a Camera class

The following steps will help you create a `Camera` class:

1.  Create two files: `Camera.h` and `Camera.cpp`. In the `Camera.h` file, include the following code:

```cpp
#include <GL/glew.h> 

#include "Dependencies/glm/glm/glm.hpp" 
#include "Dependencies/glm/glm/gtc/matrix_transform.hpp" 

```

2.  Then, create the `Camera` class itself, as follows:

```cpp
class Camera 
{ 
public: 

   Camera(GLfloat FOV, GLfloat width, GLfloat height, GLfloat 
     nearPlane, GLfloat farPlane, glm::vec3 camPos); 
   ~Camera(); 

   glm::mat4 getViewMatrix(); 
   glm::mat4 getProjectionMatrix(); 
   glm::vec3 getCameraPosition(); 

private: 

   glm::mat4 viewMatrix; 
   glm::mat4 projectionMatrix; 
   glm::vec3 cameraPos; 

};
```

3.  In the constructor and the public region of the `camera` class, we get the **field of view** (**FOV**), the width and height of the viewport, the distance to `nearPlane`, the distance to `farPlane`, and the position that we want to set the camera at.
4.  We also add three getters to get the view matrix, projection matrix, and the camera position.
5.  In the private section, we create three variables: two 4 x 4 matrices for setting the view and projection matrices and a `vec3` property to specify the camera position.
6.  In the `Camera.cpp` file, we include the `Camera.h` file at the top and create the `camera` constructor, as follows:

```cpp
#include "Camera.h" 

Camera::Camera(GLfloat FOV, GLfloat width, GLfloat height, GLfloat nearPlane, GLfloat farPlane, glm::vec3 camPos){ 

   cameraPos = camPos; 
   glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, 0.0f); 
   glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f); 

   viewMatrix = glm::lookAt(cameraPos, cameraFront, cameraUp); 
   projectionMatrix = glm::perspective(FOV, width /height, 
                      nearPlane, farPlane); 
} 
```

7.  In the constructor, we set the camera position to the local variable and set up two `vec3` properties called `cameraFront` and `cameraUp`. Our camera is going to be a stationary camera that will always be looking toward the center of the world coordinates; the `up` vector will always be pointing toward the positive y-axis.
8.  To create `viewMatrix`, we call the `glm::lookAt` function and pass in the `cameraPos`, `cameraFront`, and `cameraUp` vectors.
9.  We create the projection matrix by setting the `FOV` value of the `FOV`; this is an aspect ratio that is given by the `width` value over the `height`, `nearPlane`, and `farPlane` values.

10.  With the view and projection matrices set, we can now create the getter functions, as follows:

```cpp
glm::mat4 Camera::getViewMatrix() { 

   return viewMatrix; 
} 
glm::mat4 Camera::getProjectionMatrix() { 

   return projectionMatrix; 
} 

glm::vec3 Camera::getCameraPosition() { 

   return cameraPos; 
} 
```

Next, we'll create the `shaderLoader` class, which will let us create the `shader` program.

# The ShaderLoader class

The following steps will show you how to implement the `ShaderLoader` class in an OpenGL project:

1.  In the `ShaderLoader` class, create a public function called `createProgram` that takes a vertex and fragment `shader` file.
2.  We'll also create two private functions: `readShader`, which returns a string, and `createShader`, which returns an unsigned GL `int`:

```cpp
#include <GL/glew.h> 

class ShaderLoader { 

   public: 

         GLuint CreateProgram(const char* vertexShaderFilename, 
           const char* fragmentShaderFilename); 

   private: 

         std::string readShader(const char *filename); 
         GLuint createShader(GLenum shaderType, std::string source, 
           const char* shaderName); 
};
```

3.  In the `ShaderLoader.cpp` file, we include our `ShaderLoader.h` header file, the `iostream` system header file, and the `fstream` vector, as follows:

```cpp
#include "ShaderLoader.h"  

#include<iostream> 
#include<fstream> 
#include<vector>
```

`iostream` is used when you want to print something to the console; `fstream` is used for reading a file. We'll need this as we will be passing in vertex and shader files for the `fstream` to read, as well as vectors for storing character strings.

4.  First, we create the `readerShader` function; this will be used to read the `shader` file that we passed in:

```cpp
std::string ShaderLoader::readShader(const char *filename) 
{ 
   std::string shaderCode; 
   std::ifstream file(filename, std::ios::in); 

   if (!file.good()){ 
         std::cout << "Can't read file " << filename << std::endl; 
         std::terminate(); 
   } 

   file.seekg(0, std::ios::end); 
   shaderCode.resize((unsigned int)file.tellg()); 
   file.seekg(0, std::ios::beg); 
   file.read(&shaderCode[0], shaderCode.size()); 
   file.close(); 
   return shaderCode; 
} 

```

The contents of the `shader` file are then stored in a string and returned.

5.  Next, we create the `createShader` function, which will actually compile the shader, as follows:

```cpp
GLuint ShaderLoader::createShader(GLenum shaderType, std::string source, const char* shaderName) 
{ 

   int compile_result = 0; 

   GLuint shader = glCreateShader(shaderType); 
   const char *shader_code_ptr = source.c_str(); 
   const int shader_code_size = source.size(); 

   glShaderSource(shader, 1, &shader_code_ptr, 
     &shader_code_size); 
   glCompileShader(shader); 
   glGetShaderiv(shader, GL_COMPILE_STATUS, 
     &compile_result); 

   //check for errors 

   if (compile_result == GL_FALSE) 
   { 

         int info_log_length = 0; 
         glGetShaderiv(shader, GL_INFO_LOG_LENGTH, 
           &info_log_length); 

         std::vector<char> shader_log(info_log_length); 

         glGetShaderInfoLog(shader, info_log_length, NULL, 
           &shader_log[0]); 
         std::cout << "ERROR compiling shader: " << 
          shaderName << std::endl <<&shader_log[0] <<
          std::endl; 
         return 0; 
   } 
   return shader; 
}  
```

6.  The `CreateShader` function takes the following three parameters:

*   The first parameter is the `enum` parameter, called `shaderType`, which specifies the type of `shader` being sent to be compiled. In this case, it could be a vertex shader or a fragment shader.
*   The second parameter is the string that contains the shader code.
*   The final parameter is a string with the `shader` type, which will be used to specify whether there is a problem compiling the `shader` type.

7.  In the `CreateShader` function, we call `glCreateShader` in order to specify the type of shader that is being created; then, `glCompileShader` is called to compile the shader. Afterward, we get the compiled result of the shader.
8.  If there is a problem with compiling the shader, then we send out a message stating that there is an error compiling the shader alongside `shaderLog`, which will detail the compilation error. If there are no errors during compilation, then the shader is returned.

9.  The final function is the `createProgram` function, which takes the `vertex` and `fragment` shaders:

```cpp
GLuint ShaderLoader::createProgram (const char* vertexShaderFilename, const char* fragmentShaderFilename){

  std::string vertex_shader_code = readShader 
                                   (vertexShaderFilename);

  std::string fragment_shader_code = readShader 
                                     (fragmentShaderFilename);

  GLuint vertex_shader = createShader (GL_VERTEX_SHADER, 
                         vertex_shader_code,
                         “vertex shader” );

  GLuint fragment_shader = createShader (GL_FRAGMENT_SHADER, 
                           fragment_shader_code,
                           “fragment shader”);

  int link_result = 0;
  //create the program handle, attach the shaders and link it
  GLuint program = glCreateProgram();
  glAttachShader(program, vertex_shader);
  glAttachShader(program, fragment_shader);

  glLinkProgram(program);
  glGetProgramiv(program, GL_LINK_STATUS, &link_result);
  //check for link errors
  if (link_result == GL_FALSE) {

    int info_log_length = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &info_log_length);

    std::vector<char> program_log(info_log_length);

    glGetProgramInfoLog(program, info_log_length, NULL, 
      &program_log[0]);
    std::cout << “Shader Loader : LINK ERROR” << std::endl 
      <<&program_log[0] << std::endl;

    return 0;
  }
  return program;
}
```

10.  This function takes the vertex and fragment shader files, reads them, and then compiles both files.
11.  Then, we create a new `shaderProgram` function by calling `glCreateProgram()` and assigned to program.

12.  Now, we have to attach both shaders to the program by calling `glAttachShader` and passing the program and the shader.
13.  Finally, we link the program by calling `glLinkProgram`. After, we pass in the program and check for any linking errors.
14.  If there are any linking errors, we send out an error message to the console, along with a program log that will detail the linking error. If not, then the program is returned.

# The Light Renderer class

Now, it's time to draw our first object; to do so, perform the following steps:

1.  We will draw a basic light source that will appear above the current scene so that we can visualize the location of the light source in the scene. We will use this location of the light source later to calculate the lighting on our object. Note that a flat-shaded object doesn't need to have lighting calculations made on it.
2.  First, create a `LightRenderer.h` file and a `.cpp` file, and then create the `LightRenderer` class.
3.  At the top of the `LightRenderer.h` file, include the following headers:

```cpp
#include <GL/glew.h> 

#include "Dependencies/glm/glm/glm.hpp" 
#include "Dependencies/glm/glm/gtc/type_ptr.hpp" 

#include "Mesh.h" 
#include "ShaderLoader.h"; 
#include "Camera.h"  
```

4.  We will need `glew.h` to call the OpenGL commands, while we'll need the `glm` headers to define `vec3` and the matrices.
5.  We will also need `Mesh.h`, which allows us to define the shape of the light in the light source. You can use the `ShaderLoader` class to load in the shaders in order to render the object and `Camera.h` to get the camera's location, view, and projection matrices onto the scene.

6.  We will create the `LightRenderer` class next:

```cpp
class LightRenderer 
{ 

}; 
```

We will add the following `public` section to this class:

```cpp
public: 
   LightRenderer(MeshType meshType, Camera* camera); 
   ~LightRenderer(); 

   void draw(); 

   void setPosition(glm::vec3 _position); 
   void setColor(glm::vec3 _color); 
   void setProgram(GLuint program); 

   glm::vec3 getPosition(); 
   glm::vec3 getColor(); 

```

7.  In the public section, we create the constructor that we pass `MeshType` to; this will be used to set the shape of the object that we want to render. Then, we have the destructor. Here, we have a function called `draw`, which will be used to draw the mesh. Then, we have a couple of setters for setting the position, color, and shader program for the object.
8.  After defining the public section, we set the `private` section, as follows:

```cpp
private: 

   Camera* camera; 

   std::vector<Vertex>vertices; 
   std::vector<GLuint>indices; 

glm::vec3 position, color; 

GLuint vbo, ebo, vao, program;  
```

9.  In the private section, we have a `private` variable so that we can store the camera locally. We create vectors to store the vertex and index data; we also create local variables to store the position and color information. Then, we have `GLuint`, which will store `vbo`, `ebo`, `vao`, and the program variable.

The program variable will have the shader program that we want to use to draw the object. Then, we have `vbo`, which stands for vertex buffer object; `ebo`, which stands for Element Buffer Object; and `vao`, which stands for Vertex Array Object. Let's examine each of these buffer objects and find out what they do:

*   **Vertex Buffer Object** (**VBO**): This is the geometrical information; it includes attributes such as position, color, normal, and texture coordinates. These are stored on a per vertex basis on the GPU.
*   **Element Buffer Object** (**EBO**): This is used to store the index of each vertex and will be used while drawing the mesh.
*   **Vertex Array Object** (**VAO**): This is a helper container object that stores all the VBOs and attributes. This is used as you may have more than one VBO per object, and it would be tedious to bind the VBOs all over again when you render each frame.

Buffers are used to store information in the GPU memory for fast and efficient access to the data. Modern GPUs have a memory bandwidth of approximately 600 GB/s, which is enormous compared to the current high-end CPUs that only have approximately 12 GB/s.

Buffer objects are used to store, retrieve, and move data. It is very easy to generate a buffer object in OpenGL. You can easily generate one by calling `glGenBuffers()`.

That is all for `LightRender.h`; now, let's move on to `LightRenderer.cpp`, as follows:

1.  At the top of `LightRenderer.cpp`, include `LightRenderer.h`. Then, add the constructor, as follows:

```cpp
LightRenderer::LightRenderer(MeshType meshType, Camera* camera) { 

} 
```

2.  In the `LightRenderer` constructor, we start adding the code. First, we initialize the local camera, as follows:

```cpp
this->camera = camera; 
```

3.  Next, we set the shape of the object that we want to draw, depending on the `MeshType` type. For this, we will create a `switch` statement and call the appropriate `setData` function, depending on the type, as follows:

```cpp
   switch (modelType) { 

         case kTriangle: Mesh::setTriData(vertices, indices); 
           break; 
         case kQuad: Mesh::setQuadData(vertices, indices); break; 
         case kCube: Mesh::setCubeData(vertices, indices); break; 
         case kSphere: Mesh::setSphereData(vertices, indices); 
           break; 
   }  
```

4.  Next, we will generate and bind the `vao` buffer object, as follows:

```cpp
glGenVertexArrays(1, &vao); 
glBindVertexArray(vao); 

```

The `glGenVertexArrays` function takes two parameters; the first parameter is the number of vertex array object names that we want to generate. In this case, we just want to create one, so it is specified as such. The second parameter takes in an array where the vertex array names are stored, so we pass in the `vao` buffer object.

5.  The `glBindVertexArray` function is called `next`, and `vao` is passed into it in order to bind the `vao` buffer object. The `vao` buffer object will be bound for the duration of the application. A buffer is an object that's managing a certain piece of memory; buffers can be of different types and, therefore, they need to be bound to a specific buffer target so that they can give meaning to the buffer.
6.  Once the `vao` buffer object has been bound, we can generate the vertex buffer object and store the vertex attributes.
7.  To generate the vertex buffer object, we call `glGenBuffers()`; this also takes two parameters. The first parameter is the number of buffers that we want to generate, while the second is the array of VBOs. In this case, since we have one `vbo` buffer object, we will just pass in `1` for the first parameter and pass in the `vbo` as the second parameter:

```cpp
glGenBuffers(1, &vbo);  
```

8.  Next, we have to specify the buffer type. This is done by using the `glBindBuffer()` function; this takes two parameters again. The first is the buffer type and, in this case, it is of the `GL_ARRAY_BUFFER` type, while the second parameter is the name of the buffer object, which is `vbo`. Now, add the following line of code:

```cpp
glBindBuffer(GL_ARRAY_BUFFER, vbo);
```

9.  In the next step, we actually pass in the data that we want to store in the buffer. This is done by calling `glBufferData`; the `glBufferData` function takes four parameters:

*   The first parameter is the buffer type, which, in this case, is `GL_ARRAY_BUFFER`.
*   The second parameter is the size in bytes of the buffer data to store.
*   The third parameter is the pointer to the data, which will be copied.
*   The fourth parameter is the expected usage of the data being stored.

In our case, we will just modify the data once and use it many times, so it will be called `GL_STATIC_DRAW`.

10.  Now, add the `glBufferData` function for storing the data, as follows:

```cpp
glBufferData(GL_ARRAY_BUFFER,  
sizeof(Vertex) * vertices.size(),  
&vertices[0],  
GL_STATIC_DRAW); 
```

Now, we have to set the vertex attributes that we are going to use. While creating the `struct` vertex, we have attributes such as position, color, normal, and texture coordinates; however, we may not need all of these attributes all of the time. Therefore, we only need to specify the ones that we need. In our case, since we are not using any lighting calculation or applying any textures to the object, we don't need to specify this – we will just need the position and color attributes for now. However, these attributes need to be enabled first.

11.  To enable these attributes, we'll call `glEnableVertexAttribArray` and pass in the index that we want to enable. The position will be in the 0th index, so we will set the value as follows:

```cpp
glEnableVertexAttribArray(0);   
```

12.  Next, we call `glVertexAttribPointer` so that we can set the attribute that we want to use. The first attribute will be positioned at the 0th index. This takes six parameters, as follows:

*   The first parameter is the index of the vertex attribute, which, in this case, is 0.
*   The second parameter is the size of the attribute. Essentially, this is the number of components that the vertex attribute has. In this case, it is the position of the *x*, *y*, and *z* components, so it is specified as `3`.
*   The third parameter is for the variable types of the components; since they are specified in `GLfloat`, we specify `GL_FLOAT`.
*   The fourth parameter is a Boolean that specifies whether the values should be normalized or whether they should be converted into fixed-point values. Since we don't want the values to be normalized, we specify `GL_FALSE`.
*   The fifth parameter is called the stride, which is the offset of consecutive vertex attributes. Imagine the vertices being laid out in the memory as follows. The stride refers to the blocks of memory that you will have to go through to get to the next set of vertex attributes; this is the size of the `struct` vertex:

![](img/351fcc48-0f0d-46fe-871f-6123bdb95baf.png)

*   The sixth parameter is the offset of the first component of the vertex attribute within the `struct` vertex. The attribute that we are looking at here is the position attribute, which is at the start of the `struct` vertex, so we will pass `0`:

![](img/19796af6-4a67-44f2-827d-5abf7621efe9.png)

13.  Set the `glVertexAttribute` pointer, as follows:

```cpp
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (GLvoid*)0);
```

14.  Let's create one more attribute pointer so that we can color the object. Like we did previously, we need to enable the attribute and set the `attrib` pointer, as follows:

```cpp
glEnableVertexAttribArray(1); 
glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(offsetof(Vertex, Vertex::color))); 

```

15.  Since the next attribute index is `1`, we enable the attribute array using `1`. While setting the attribute pointer, the first parameter is `1`, since this is the first index. `color` has three components – *r*, *g*, and *b –* so the next parameter is `3`. Colors are defined as floats, and so we specify `GL_FLOAT` for this parameter.

16.  Since we don't want the fourth parameter to be normalized, we set the parameter to `GL_FALSE`. The fifth parameter is the stride and it is still equal to the size of the `struct` vertex. Finally, for the offset, we use the `offsetof` function to set the offset of `vertex::color` in the `struct` vertex.

Next, we have to set the element buffer object. This is done in the same way as setting the vertex buffer object: we need to generate the element, set the binding, and then bind the data to the buffer, as follows:

1.  First, we generate the buffer by calling `glGenBuffers`. This is done by passing in the number of buffers that we want to create, which is `1`, and then passing in the name of the buffer object to generate it:

```cpp
glGenBuffers(1, &ebo); 
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo); 

glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * indices.size(), &indices[0], GL_STATIC_DRAW); 
```

2.  Next, we bind the buffer type to the buffer object, which, in this case, is `GL_ELEMENT_ARRAY_BUFFER`. It will store the element or index data.
3.  Then, we set the index data itself by calling `glBufferData`. We pass in the buffer type first, set the size of the element data, and then pass in the data and the usage with `GL_STATIC_DRAW`, like we did previously.
4.  At the end of the constructor, we unbind the buffer and the vertex array as a precaution:

```cpp
glBindBuffer(GL_ARRAY_BUFFER, 0); 
glBindVertexArray(0);
```

5.  Next, we will create the `draw` function; this will be used to draw the object itself. To do this, add the `draw` function, as follows:

```cpp
void LightRenderer::draw() { 

}
```

6.  We will use this function to add the code for drawing the object. The first thing we will do is create a `glm::mat4` function called `model` and initialize it; then, we will use the `glm::translate` function to translate the object to the required position:

```cpp
glm::mat4 model = glm::mat4(1.0f); 

   model = glm::translate(glm::mat4(1.0),position); 
```

Next, we will set the model, view, and projection matrices to transform the object from its local space. This was covered in [Chapter 2](ee788533-687d-4231-91a4-cb1de9ca01dd.xhtml), *Mathematics and Graphics Concepts,* so now is a good time for you to go and refresh your memory of graphics concepts.

The model, view, and projection matrices are set in the vertex shader. Information is sent to the shader by calling `glUseProgram`, which takes in a shader program:

```cpp
glUseProgram(this->program); 
```

Then, we can send the required information through the uniform variables. We will create a uniform data type in the shader using a name. In the `draw` function, we need to get the location of this uniform variable by calling `glGetUniformLocation`, and then passing in the program and the variable string in the shader that we set, as follows:

```cpp
   GLint modelLoc = glGetUniformLocation(program, "model"); 
```

This will return a `GLuint` value with the location of the variable, which is the model matrix here.

Now, we can set the value of the model matrix using the `glUniform` function. Since we are setting a matrix uniform, we use the `glUniformMatrix3fv` function; this takes four parameters. The first parameter is the location that we obtained in the previous step, whereas the second parameter is the amount of data that we are passing in; in this case, we are just passing in one matrix, so we specify this as `1`. The third parameter is a Boolean value, which specifies whether the data needs to be transposed. We don't want the matrix to be transposed, so we specify it as `GL_FALSE`. The final parameter is the pointer to the data, `gl::ptr_value`; we pass this into the model matrix.

Now, add the function to set the model matrix, as follows:

```cpp
glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
```

Similar to the model matrix, we have to pass in the view and projection matrices to the shader as well. To do this, we get the view and projection matrices from the `camera` class. Then, we get the location of the uniform variable that we defined in the shader and set the value of the view and projection matrices using the `glUniformMatrix4fv` function:

```cpp
   glm::mat4 view = camera->getViewMatrix(); 
   GLint vLoc = glGetUniformLocation(program, "view"); 
   glUniformMatrix4fv(vLoc, 1, GL_FALSE, glm::value_ptr(view)); 

   glm::mat4 proj = camera->getprojectionMatrix(); 
   GLint pLoc = glGetUniformLocation(program, "projection"); 
   glUniformMatrix4fv(pLoc, 1, GL_FALSE, glm::value_ptr(proj)); 
```

Once we have all the required data to draw the object, we can finally draw the object. At this point, we call `glBindVertexArray`, bind the `vao` buffer object, and then call the `glDrawElements` function to draw the object.

The `glDrawElements` function takes four parameters. The first parameter is the mode that we can use to draw the lines by calling `GL_LINES`. Alternatively, we can draw triangles by using `GL_TRIANGLES`. There are, in fact, many more types of modes that can be specified, but in our case, we will only specify `GL_TRIANGLES`.

The second parameter is the number of elements or the number of indices that need to be drawn. This is specified when we created the object. The third parameter is the type of index data that we will be passing, which is of the `GL_UNSIGNED_INT` type. The final parameter is the location where the indices are stored – this is set to 0.

Add the following lines of code:

```cpp
glBindVertexArray(vao); 
glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0); 
```

For safety purposes, we will unbind the vertex array and the program variable by setting their values to `0`:

```cpp
glBindVertexArray(0); 
glUseProgram(0); 

```

This marks the end of the `draw` function.

Add the destructor and the rest of the setters and getters to finish the class, as follows:

```cpp
LightRenderer::~LightRenderer() { 

} 

void LightRenderer::setPosition(glm::vec3 _position) { 

   position = _position; 
} 

void LightRenderer::setColor(glm::vec3 _color) { 

   this->color = _color; 
} 

void LightRenderer::setProgram(GLuint _program) { 

   this->program = _program; 
} 

//getters 
glm::vec3 LightRenderer::getPosition() { 

   return position; 
} 

glm::vec3 LightRenderer::getColor() { 

   return color; 
} 

```

# Drawing the object

Let's go back to the `source.cpp` file and render `LightRenderer`, as follows:

1.  At the top of the file, include `ShaderLoader.h`, `Camera.h`, and `LightRenderer.h`, and then create an instance of the `Camera` and `LightRenderer` classes called `camera` and `light`, as follows:

```cpp
#include "ShaderLoader.h" 
#include "Camera.h" 
#include "LightRenderer.h" 
Camera* camera; 
LightRenderer* light;
```

2.  Create a new function called `initGame` and add the prototype for it to the top of the file. In the `gameInit` function, load the shader and initialize the camera and light.
3.  Add the new function, as follows:

```cpp
 void initGame(){ 
... 

}  
```

4.  The first thing we do is enable depth testing so that only the pixels in the front are drawn. This is done by calling the `glEnable()` function and passing in the `GL_DEPTH_TEST` variable; this will enable the following depth test:

```cpp
   glEnable(GL_DEPTH_TEST); 
```

5.  Next, we will create a new instance of `ShaderLoader` called `shader` in the `init` function. Then, we need to call the `createProgram` function and pass in the vertex and fragment shader files to shade in the light source. The program will return a `GLuint` value, which we store in a variable called `flatShaderProgram`, as follows:

```cpp
ShaderLoader shader; 

GLuint flatShaderProgram = shader.createProgram("Assets/Shaders/FlatModel.vs", "Assets/Shaders/FlatModel.fs"); 
```

6.  The vertex and shader files are located in the `Assets` folder under `Shaders`; the `FlatModel.vs` file will look as follows:

```cpp
#version 450 core 

layout (location = 0) in vec3 Position; 
layout (location = 1) in vec3 Color; 

uniform mat4 projection; 
uniform mat4 view; 
uniform mat4 model; 

out vec3 outColor; 

void main(){ 

   gl_Position = projection * view * model * vec4(Position, 1.0); 

   outColor = Color; 
}
```

`#version` specifies the version of GLSL that we are using, which is `450`. This stands for OpenGL version 4.50\. Next, `layout (location = 0)` and `layout (location = 1)` specify the location of the vertex attributes that are passed in; in this case, this is the position and color. The `0` and `1` indices correspond to the index numbers while setting `vertexAttribPointer`. In the variables that are specified, this data is placed in the shader and stored in shader-specific `vec3` data types called `Position` and `Color`.

The three uniforms that we sent from the `draw` call for storing the model, view, and projection matrices are stored in a variable type called `uniform` and a `mat4` store data type, both of which are matrices. After this, we create another variable of the `out` type, which specifies that this will be sent out of the vertex shader; this is of the `vec3` type and is called `outColor`. Next, all the actual work is done inside the `main` function. For this, we transform the local coordinate system by multiplying the position by the model, view, and projection matrices. The result is stored in a GLSL intrinsic variable called `gl_Position`—this is the final position of the object. Then, we store the `Color` attribute in the `out vec3` variable that we created called `outColor`—that's it for the vertex shader!

7.  Next, let's take a look at the fragment shader's `FlatModel.fs` file:

```cpp
#version 450 core 

in vec3 outColor; 

out vec4 color; 

void main(){ 

   color = vec4(outColor, 1.0f); 

} 

```

In the fragment shader file, we also specify the version of GLSL that we are using.

Next, we specify an `in vec3` variable called `outColor`, which will be the color that was sent out of the vertex shader. This can be used in the fragment shader. We also create an `out vec4` variable called `color`, which will be sent out of the fragment shader and will be used to color the object. The color that's being sent out of the fragment shader is expected to be a `vec4` variable. Then, in the main function, we convert `outColor` from a `vec3` variable into a `vec4` variable, and then set it to the `color` variable.

In the shaders, we can convert a `vec3` variable into a `vec4` variable by simply performing the following operation. This may look a bit strange, but for the sake of convenience, this unique feature is available in shader programming to make our lives a little easier.

8.  Going back to the `source.cpp` file, when we pass in the vertex and fragment shader files, they will create `flatShaderProgram`. Next, in the `initGame` function, we create and initialize the camera, as follows:

```cpp
camera = new Camera(45.0f, 800, 600, 0.1f, 100.0f, glm::vec3(0.0f, 
         4.0f, 6.0f)); 
```

Here, we create a new camera with an FOV of `45`, a width and height of `800` x `600`, a near and far plane of `0.1f` and `100.0f`, as well as a position of `0` along the *X*-axis, `4.0` along the Y-axis, and `6.0` along the Z-axis.

9.  Next, we create `light`, as follows:

```cpp
light = new LightRenderer(MeshType::kTriangle, camera); 
light->setProgram(flatShaderProgram); 
light->setPosition(glm::vec3(0.0f, 0.0f, 0.0f)); 
```

10.  This is done with the shape of a triangle, which is then passed to the camera. Then, we set the shader to `flatShaderProgram` and set the position to the center of the world.

11.  Now, we call the `draw` function of the light in the `renderScene()` function, as follows:

```cpp
void renderScene(){ 

   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 
   glClearColor(1.0, 1.0, 0.0, 1.0);//clear yellow  
   light->draw(); 

}
```

12.  I changed the clear screen color to yellow so that the triangle can be seen clearly. Next, call the `initGame` function in the `main` function, as follows:

```cpp
int main(int argc, char **argv) 
{ 

   glfwInit(); 

   GLFWwindow* window = glfwCreateWindow(800, 600, 
                        " Hello OpenGL ", NULL, NULL); 

   glfwMakeContextCurrent(window); 

   glewInit(); 

   initGame(); 

   while (!glfwWindowShouldClose(window)){ 

         renderScene(); 

         glfwSwapBuffers(window); 
         glfwPollEvents(); 
   } 

   glfwTerminate(); 

   delete camera; 
   delete light; 

   return 0; 
}
```

13.  Delete the camera and light at the end so that the system resource is released.
14.  Now, run the project to see the glorious triangle that we set as the shape of the light source:

![](img/7deb66c1-2c3c-4ca6-b00c-931c519ebb61.png)

15.  Change the `MeshType` type to cube to see a cube being drawn instead:

![](img/5dae76b6-4eac-495b-a865-a2d452a444b9.png)

If you get an error instead of the colored object as output, then this could either mean that you have done something incorrectly or that your drivers haven't been updated and that your GPU doesn't support OpenGL 4.5.

16.  To make sure GLFW supports the version of your driver, add the following code, which checks for any GLFW errors. Then, run the project and look at the console output for any errors:

```cpp
static void glfwError(int id, const char* description)
{
  std::cout << description << std::endl;
}

int main(int argc, char **argv)
{

  glfwSetErrorCallback(&glfwError);

  glfwInit();

  GLFWwindow* window = glfwCreateWindow(800, 600, " Hello OpenGL ", 
                       NULL, NULL);

  glfwMakeContextCurrent(window);

  glewInit();

  initGame();

  while (!glfwWindowShouldClose(window)){

    renderScene();

    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  glfwTerminate();

  delete camera;
  delete light;

  return 0;
}
```

If you get the following output, then this could mean that the OpenGL version you're using is not supported:

![](img/1a6a8556-ab34-460b-b184-429841ce741b.png)

This will be accompanied by the following error, suggesting that the GLSL version you're using is not supported:

![](img/845bccb8-4844-4fee-a27e-fb487739520e.png)

In this case, change the version of the shader code at the top of the shader to `330` instead of `450`, and then try running the project again.

This should give you the desired output.

# Summary

In this chapter, we created a new OpenGL project and added the necessary libraries to get the project working. Then, we created a new window to work with using GLFW. After using a couple more lines of code, we were able to clear the viewport with the color of our choice.

Next, we started preparing some classes that could help us draw objects such as the `Mesh` class, which defined the shape of the object, and the `Camera` class, which we use in order to view the object. Then, we created a `ShaderLoader` class, which helped us create the shader program that is used to draw the object.

With the necessary preparation done, we created a `LightRenderer` class. This is used to draw an object that represents a light position that's defined by a shape. We used this class to draw our first object.

In the next chapter, we will explore how to draw other objects by adding textures and physics to the rendering engine.