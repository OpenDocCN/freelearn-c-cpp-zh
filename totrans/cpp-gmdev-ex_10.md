# Building on the Game Objects

n the last chapter, we looked at how to draw basic shapes using OpenGL. Now that we have covered the basics, let's improve our objects by adding some textures to them so that the objects don't just look like a plain cube and sphere.

We can write our physics as we did last time, but when dealing with 3D objects, writing our own physics can become difficult and time consuming. To simplify the process, we will use the help of an external physics library to handle the physics and collision detection.

We will cover the following topics in this chapter:

*   Creating the `MeshRenderer` class
*   Creating the `TextureLoader` class
*   Adding Bullet Physics
*   Adding rigid bodies

# Creating the MeshRenderer class

For drawing regular game objects, we will create a separate class from the `LightRenderer` class by adding texture, and we will also add motion to the object by adding physical properties. We will draw a textured object and then add physics to this object in the next section of this chapter. To do this, we will create a new `.h` and `.cpp` file called `MeshRenderer`.

In the `MeshRenderer.h` file, we will do the following:

1.  First, we will add the includes as follows:

```cpp
#include <vector> 

#include "Camera.h" 
#include "LightRenderer.h" 

#include <GL/glew.h> 

#include "Dependencies/glm/glm/glm.hpp" 
#include "Dependencies/glm/glm/gtc/matrix_transform.hpp" 
#include "Dependencies/glm/glm/gtc/type_ptr.hpp" 
```

2.  Next, we will create the class itself as follows:

```cpp
Class MeshRenderer{  

}; 
```

3.  We will create the `public` section first as follows:

```cpp
   public: 
         MeshRenderer(MeshType modelType, Camera* _camera); 
         ~MeshRenderer(); 

          void draw(); 

         void setPosition(glm::vec3 _position); 
         void setScale(glm::vec3 _scale); 
         void setProgram(GLuint _program); 
         void setTexture(GLuint _textureID); 
```

In this section, we create the constructor, which takes a `ModelType` and the `_camera`. We add the destructor afterward. We have a separate function for drawing the object.

4.  We then use some `setter` functions to set the position, scale, the shader program, and the `textureID` function , which we will be using to set the texture on the object.
5.  Next, we will add the `private` section as follows:

```cpp
   private: 

         std::vector<Vertex>vertices; 
         std::vector<GLuint>indices; 
         glm::mat4 modelMatrix; 

         Camera* camera; 

         glm::vec3 position, scale; 

               GLuint vao, vbo, ebo, texture, program;  
```

In the `private` section, we have vectors to store the vertices and the indices. Then, we have a `glm::mat4` variable called `modelMatrix` to store the model matrix value in.

6.  We create a local variable for the camera and `vec3s` for storing the position and scale value.
7.  Finally, we have `Gluint` to store `vao`, `vbo`, `ebo`, `textureID`, and the shader program.

We will now move on to setting up the `MeshRenderer.cpp` file by going through the following steps:

1.  First, we will include the `MeshRenderer.h` file at the top of `MeshRenderer.cpp`.
2.  Next, we will create the constructor for `MeshRenderer` as follows:

```cpp
MeshRenderer::MeshRenderer(MeshType modelType, Camera* _camera) { 

} 
```

3.  For this, we first initialize the `camera`, `position`, and `scale` local values as follows:

```cpp
   camera = _camera; 

   scale = glm::vec3(1.0f, 1.0f, 1.0f); 
   position = glm::vec3(0.0, 0.0, 0.0);
```

4.  Then we create a `switch` statement, as we did in `LightRenderer`, to get the mesh data, as follows:

```cpp
   switch (modelType){ 

         case kTriangle: Mesh::setTriData(vertices, indices);  
               break; 
         case kQuad: Mesh::setQuadData(vertices, indices);  
               break; 
         case kCube: Mesh::setCubeData(vertices, indices); 
               break; 
         case kSphere: Mesh::setSphereData(vertices, indices);  
               break; 
   } 
```

5.  Then, we generate and bind `vao`, `vbo`, and `ebo`. In addition to this, we set the data for `vbo` and `ebo` as follows:

```cpp
   glGenVertexArrays(1, &vao); 
   glBindVertexArray(vao); 

   glGenBuffers(1, &vbo); 
   glBindBuffer(GL_ARRAY_BUFFER, vbo); 
   glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex) * vertices.size(),
   &vertices[0], GL_STATIC_DRAW); 

   glGenBuffers(1, &ebo); 
   glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo); 
   glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * 
      indices.size(), &indices[0], GL_STATIC_DRAW); 
```

6.  The next step is to set the attributes. In this case, we will be setting the `position` attribute, but instead of color, we will set the texture coordinate attribute, as it will be required to set the texture on top of the object.
7.  The attribute at the 0th index will still be a vertex position, but the attribute of the first index will be a texture coordinate this time, as shown in the following code:

```cpp
glEnableVertexAttribArray(0);

glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), 
   (GLvoid*)0);

glEnableVertexAttribArray(1);

glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex),   
   (void*)(offsetof(Vertex, Vertex::texCoords)));
```

Here, the attribute for the vertex position remains the same, but for the texture coordinate, the first index is enabled as before. The change occurs in the number of components. The texture coordinate is defined in the *x*- and *y*-axes, as this is a 2D texture, so for the second parameter, we specify `2` instead of `3`. The stride still remains the same, but the offset is changed to `texCoords`.

8.  To close the constructor, we unbind the buffers and `vertexArray` as follows:

```cpp
glBindBuffer(GL_ARRAY_BUFFER, 0); 
glBindVertexArray(0); 
```

9.  We now add the `draw` function as follows:

```cpp
void MeshRenderer::draw() { 

} 

```

10.  In this `draw` function, we will first set the model matrix as follows:

```cpp

   glm::mat4 TranslationMatrix = glm::translate(glm::mat4(1.0f),  
      position); 

   glm::mat4 scaleMatrix = glm::scale(glm::mat4(1.0f), scale); 

   modelMatrix = glm::mat4(1.0f); 

   modelMatrix = TranslationMatrix *scaleMatrix; 
```

11.  We will create two matrices for storing `translationMatrix` and `scaleMatrix` and then we set the values.
12.  We will then initialize the `modelMatrix` variable and the multiply scale and translation matrix and assign them to the `modelMatrix` variable.
13.  Next, instead of creating a separate view and projection matrix, we can create a single matrix called `vp` and assign the multiplied view and projection matrices to it as follows:

```cpp
glm::mat4 vp = camera->getprojectionMatrix() * camera->
               getViewMatrix(); 

```

Obviously, the order in which the view and projection matrices are multiplied matters and cannot be reversed.

14.  We can now send the values to the GPU.
15.  Before we send the values to the shader, the first thing we have to do is call `glUseProgram` and set the shader program so that the data is sent to the correct program. Once this is complete, we can set the values for `vp` and `modelMatrix` as follows:

```cpp
glUseProgram(this->program); 

GLint vpLoc = glGetUniformLocation(program, "vp"); 
glUniformMatrix4fv(vpLoc, 1, GL_FALSE, glm::value_ptr(vp)); 

GLint modelLoc = glGetUniformLocation(program, "model"); 
glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(modelMatrix));  
```

16.  Next, we will bind the `texture` object. We use the `glBindTexture` function to bind the texture. The function takes two parameters, with the first being the texture target. We have a 2D texture, so we pass in `GL_TEXTURE_2D` as the first parameter and the second parameter as a texture ID. To do this, we add the following line to bind the texture:

```cpp
glBindTexture(GL_TEXTURE_2D, texture);  
```

You might be wondering why we aren't using `glUniformMatrix4fv` or something similar while setting the texture location, as we did for the matrices. Well, since we have just the one texture, the program sets the uniform location as the 0th index by default so we don't have to worry about it. This all that we require to bind the texture.

17.  Next, we can bind the `vao` and draw the object as follows:

```cpp
glBindVertexArray(vao);           
glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);   
```

18.  Unbind the `VertexArray` at the end as follows:

```cpp
glBindVertexArray(0); 

```

19.  Next, we will add the definition for the destructor and `setters` as follows:

```cpp
MeshRenderer::~MeshRenderer() { 

} 

// setters  

void MeshRenderer::setTexture(GLuint textureID) { 

   texture = textureID; 

} 

void MeshRenderer::setScale(glm::vec3 _scale) { 

   this->scale = _scale; 
} 

void MeshRenderer::setPosition(glm::vec3 _position) { 

   this->position = _position; 
} 

void MeshRenderer::setProgram(GLuint _program) { 

   this->program = _program; 
} 
```

# Creating the TextureLoader class

We created the `MeshRenderer` class, but we still need to load the texture and set the texture ID, which can be passed to the `MeshRendered` object. For this, we will create a `TextureLoader` class that will be responsible for loading the textures. Let's see how to do this.

We first need to create the new `.h` and `.cpp` file called `TextureLoader`.

To load the JPEG or PNG image, we will use a header-only library called STB. This can be downloaded from [https://github.com/nothings/stb](https://github.com/nothings/stb). Clone or download the source from the link and place the `stb-master` folder in the `Dependencies` folder.

In the `TextureLoader` class, add the following:

```cpp
#include <string> 
#include <GL/glew.h> 

class TextureLoader 
{ 
public: 
   TextureLoader(); 

   GLuint getTextureID(std::string  texFileName); 
   ~TextureLoader(); 
}; 
```

We will then use the `string` and `glew.h` libraries, as we will be passing the location of the file where the JPEG is located and `STB` will load the file from there. We will add a constructor and a destructor as they are required; otherwise, the compiler will give an error. We will then create a function called `getTextureID`, which takes a string as an input and returns `GLuint`, which will be the texture ID.

In the `TextureLoader.cpp` file, we include `TextureLoader.h`. We then add the following to include `STB`:

```cpp
#define STB_IMAGE_IMPLEMENTATION 
#include "Dependencies/stb-master/stb_image.h" 
```

We add `#define` as it is required in a `TextureLoader.cpp` file, navigate to `stb_image.h`, and include it in the project. We then add the constructor and destructor as follows:

```cpp

TextureLoader::TextureLoader(){ 

} 

TextureLoader::~TextureLoader(){ 

} 
```

Next, we create the `getTextureID` function as follows:

```cpp
GLuint TextureLoader::getTextureID(std::string texFileName){ 

}  
```

In the `getTextureID` function, we will first create three `int` variables to store the width, height, and number of channels. An image usually only has three channels: red, green, and blue. However, it could have a fourth channel, the alpha channel, which is used for transparency. JPEG pictures have only three channels, but the PNG file could have three or four channels.

In our game, we will only be using a JPEG file, so the `channels` parameter will always be three, as shown in the following code:

```cpp
   int width, height, channels;  
```

We will use the `stbi_load` function to load the image data to an unsigned char pointer, as follows:

```cpp
stbi_uc* image = stbi_load(texFileName.c_str(), &width, &height,   
                 &channels, STBI_rgb); 
```

The function takes five parameters. The first is the string of the location of the file/filename. Then, it returns the width, height, and number of channels as the second, third, and fourth parameters, and in the fifth parameter, you set the required components. In this case, we want just the `r`, `g`, and `b` channels, so we specify `STBI_rgb`.

We then have to generate and bind the texture as follows:

```cpp
GLuint mtexture; 
glGenTextures(1, &mtexture); 
glBindTexture(GL_TEXTURE_2D, mtexture);    
```

First, a texture ID called `mtexture` of the `GLuint` type is created. Then, we call the `glGenTextures` function, pass in the number of objects we want to create, and pass in the array names, which is `mtexture`. We also have to bind the texture type by calling `glBindTexture` and pass in the texture type, which is `GL_TEXTURE_2D`, specifying that it is a 2D texture and stating the texture ID.

Next, we have to set the texture wrapping. Texture wrapping dictates what happens when the texture coordinate is greater or less than `1` in *x* and *y*.

Textures can be wrapped in one of four ways: `GL_REPEAT`, `GL_MIRRORED_REPEAT`, `GL_CLAMP_TO_EDGE`, or `GL_CLAMP_TO_BORDER`.

If we imagine a texture applied to a quad, then the positive *s**-*axis runs horizontally and the *t*-axis runs vertically, starting at the origin (the bottom-left corner), as shown in the following screenshot:

![](img/4d8df6ec-bef3-40c7-95b5-795dd859db2b.png)

Let's look at the different ways that the textures can be wrapped, as shown in the following list:

*   `GL_REPEAT` just repeats the texture when applied to a quad.
*   `GL_MIRRORER_REPEAT` repeats the texture, but also mirrors the texture the next time.
*   `GL_CLAMP_TO_EDGE` takes the `rgb` value at the edge of the texture and repeats the value for the entire object. In the following screenshot, the red border pixels are repeated.
*   `GL_CLAMP_TO_BORDER` takes a user-specific value and applies it to the end of the object instead of applying the edge color, as shown in the following screenshot:

![](img/4d87c32b-008b-4a1c-8d45-e3446ca4d721.png)

For our purposes, we need `GL_REPEAT`, which is set as the default anyway, but if you had to set it, you will need to add the following:

```cpp
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT); 
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);  
```

You use the `glTexParameteri` function, which takes three parameters. The first is the texture type, which is `GL_TEXTURE_2D`. The next parameter is the direction in which you want the wrapping to apply, which is `S` or `T`. The `S` direction is the same as *x* and `T` is the same as *y*. The last parameter is the wrapping parameter itself.

Next, we can set the texture filtering. Sometimes, when you apply a low-quality texture to a big quad, if you zoom in closer, the texture will be pixelated, as shown in the left-hand picture in the following screenshot:

![](img/9d41fb43-47e4-4d47-9ae8-495d188392d7.png)

The picture on the left is the output of setting the texture filtering to `GL_NEAREST`, and the picture on the right is the result of applying texture filtering to `GL_LINEAR`. The `GL_LINEAR` wrapping linearly interpolates with the texel value of the surrounding values to give a much smoother result when compared to `GL_NEAREST`.

When the texture is magnified, it is better to set the value to `GL_LINEAR` to get a smoother picture, and when the picture is minimized, it can then be set to `GL_NEAREST`, as the texels (which are texture elements) will be so small that we won't be able to see them anyway.

To set the texture filtering, we use the same `glTexParameteri` function, but instead of passing in the wrapping direction as the second parameter we specify `GL_TEXTURE_MIN_FILTER` and `GL_TEXTURE_MAG_FILTER` as the second parameter and pass in `GL_NEAREST` or `GL_LINEAR` as the third parameter, as follows:

```cpp
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); 
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); 
```

It doesn't make sense to load a huge image with the object so far away that you can't even see it, so for optimization purposes, you can create mipmaps. Mipmaps basically take the texture and converts it to a lower resolution. This will automatically change the image to a lower resolution image when the texture is too far away from the camera. It will also change to a higher resolution image when the camera is closer.

Here is the mipmap chain for the texture we are using:

![](img/e794d3ff-8581-4313-b14c-52eafa2f18ef.png)

The mipmap quality can be set using the `glTexParameteri` function again. This basically replaces `GL_NEAREST` with either `GL_NEAREST_MIPMAP_NEAREST`, `GL_LINEAR_MIPMAP_NEAREST`, `GL_NEAREST_MIPMAP_LINEAR`, or `GL_LINEAR_ MIPMAP_LINEAR`.

The best option is `GL_LINEAR_MIPMAP_LINEAR` because it linearly interpolates the value of the texel between two mipmaps, as well as samples, by linearly interpolating between the surrounding texels (a texel is the lowest unit of an image in the same way that a pixel is the smallest unit of a screen to represent a color at a location on the screen. If a 1080p picture is shown on a 1080p screen, then 1 texel is mapped to 1 pixel).

So, we will use the following as our new filtering/mipmap values:

```cpp
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, 
   GL_LINEAR_MIPMAP_LINEAR); 
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); 
```

Once this has been set, we can finally create the texture using the `glTexImage2D` function, as follows:

```cpp
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0,GL_RGB, 
    GL_UNSIGNED_BYTE, image); 
```

The `glTexImage2D` function takes nine parameters. These are described as follows:

*   The first is the texture type, which is `GL_TEXTURE_2D`.
*   The second is the mipmap level. If we want to use a lower quality picture, we can set this value to `1`, `2`, or `3`. For our purposes, we will leave this value as `0`, which is the base level.
*   For the third parameter, we will specify which all-color channels we want to store from the image. Since we want to store all three channels, we specify `GL_RGB`.
*   The fourth and fifth parameters that we specify are the width and height of the picture.
*   The next parameter has to be set to `0`, as specified in the documentation (which can be found at [https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glTexImage2D.xhtml](https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glTexImage2D.xhtml)).
*   The next parameter that we specify is the data format of the image source.
*   The next parameter is the type of data that is passed in, which is `GL_UNSIGNED_BYTE`.
*   Finally, we set the image data.

Now that the texture is created, we call `glGenerateMipmap` and pass in the `GL_TEXTURE_2D` texture type, as follows:

```cpp
glGenerateMipmap(GL_TEXTURE_2D); 
```

We then unbind the texture, free the picture, and finally return the `textureID` function like so:

```cpp
glBindTexture(GL_TEXTURE_2D, 0); 
stbi_image_free(image); 

   return mtexture; 
```

With all that done, we call finally add our texture to the game object.

In the `source.cpp`, include `MeshRenderer.h` and `TextureLoader.h` by going through  the following steps:

1.  At the top, create a `MeshRenderer` pointer object called a sphere as follows:

```cpp
Camera* camera; 
LightRenderer* light; 
MeshRenderer* sphere;
```

2.  In the `init` function, create a new shader program called `texturedShaderProgram` of the `GLuint` type as follows:

```cpp
GLuint flatShaderProgram = shader.CreateProgram(
                           "Assets/Shaders/FlatModel.vs", 
                           "Assets/Shaders/FlatModel.fs"); 
GLuint texturedShaderProgram = shader.CreateProgram(
                               "Assets/Shaders/TexturedModel.vs",   
                               "Assets/Shaders/TexturedModel.fs");
```

3.  We will now load the two shaders called `TexturedModel.vs` and `TexturedModel.fs` as follows:

*   Here is the `TexturedModel.vs` shader:

```cpp
#version 450 core 
layout (location = 0) in vec3 position; 
layout (location = 1) in vec2 texCoord; 

out vec2 TexCoord; 

uniform mat4 vp; 
uniform mat4 model; 

void main(){ 

   gl_Position = vp * model *vec4(position, 1.0); 

   TexCoord = texCoord; 
} 

```

The only difference between this and `FlatModel.vs` is that here, the second location is a `vec2` called `texCoord`. We create an out `vec2` called `TexCoord`, into which we will store this value in the `main` function.

*   Here is the `TexturedModel.fs` shader:

```cpp
 #version 450 core 

in vec2 TexCoord; 

out vec4 color; 

// texture 
uniform sampler2D Texture; 

void main(){ 

         color = texture(Texture, TexCoord);  
} 
```

We create a new `vec2` called `TexCoord` to receive the value from the vertex shader.

We then create a new uniform type called `sampler2D` and call it `Texture`. The texture is received through a sampler that will be used to sample the texture depending upon the wrap and filtering parameters we set while creating the texture.

Then, the color is set depending upon the sampler and texture coordinates using the `texture` function. This function takes sampler and texture coordinates as parameters. The texel at a texture coordinate is sampled based on the sampler, and that color value is returned and assigned to the object at that texture coordinate.

Let's continue creating the `MeshRenderer` object. Load the `globe.jpg` texture file using the `getTextureID` function of the `TextureLoader` class and set it to a `GLuint` called `sphereTexture` as follows:

```cpp
 TextureLoader tLoader; 
GLuint sphereTexture = tLoader.getTextureID("Assets/Textures/globe.jpg");  
```

Create the sphere `MeshRederer` object, set the mesh type, and pass the camera. Set the program, texture, position, and scale as follows:

```cpp
   sphere = new MeshRenderer(MeshType::kSphere, camera); 
   sphere->setProgram(texturedShaderProgram); 
   sphere->setTexture(sphereTexture); 
   sphere->setPosition(glm::vec3(0.0f, 0.0f, 0.0f)); 
   sphere->setScale(glm::vec3(1.0f)); 
```

In the `renderScene` function, draw the `sphere` object as follows:

```cpp
void renderScene(){ 

   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 
   glClearColor(1.0, 1.0, 0.0, 1.0); 

   sphere->draw(); 

}  
```

You should now see the textured globe when you run the project, as shown in the following screenshot:

![](img/9ef6e337-5eec-4eec-8aa6-9d83485789fe.png)

The camera is created as follows, and is set at the *z* position of four units:

```cpp

camera = new Camera(45.0f, 800, 600, 0.1f, 100.0f, glm::vec3(0.0f, 
         0.0f, 4.0f)); 
```

# Adding Bullet Physics

To add physics to our game, we will be using the Bullet Physics engine. This is an open source project that is widely used in AAA games and movies. It is used for collision detection as well as soft- and rigid-body dynamics. The library is free for commercial use.

Download the source from [https://github.com/bulletphysics/bullet3](https://github.com/bulletphysics/bullet3) , and using CMake you will need to build the project for the release version of x64\. For your convenience, the header and `lib` files are included in the project for the chapter. You can take the folder and paste it into the `dependencies` folder.

Now that we have the folder, let's take a look at how to add Bullet Physics by following these steps:

1.  Add the `include` folder in C/C++ | General | Additional Include Directories as shown in the following screenshot:

![](img/76e2dde2-5097-4e5e-bf7f-6eed3826eaaa.png)

2.  Add the `lib/win64/Rls` folder in Linker | General | Additional Library Directories:

![](img/51590433-e7ac-4e36-b063-09cac2f2fbb6.png)

3.  Add `BulletCollision.lib`, `BulletDynamics.lib`, and `LinearMath.lib` to Linker | Input | Additional Dependencies, as shown in the following screenshot:

![](img/1369c25b-a19e-4dd4-b3f7-b0f325fbf94f.png)

These libraries are responsible for the calculation of the movement of the game objects based on conditions such as gravity and external force, collision detection, and memory allocation.

4.  With the prep work out of the way, we can start adding physics to the game. In the `source.cpp` file, include `btBulletDynamicsCommon.h` at the top of the file, as follows:

```cpp
#include "Camera.h" 
#include "LightRenderer.h" 
#include "MeshRenderer.h" 
#include "TextureLoader.h" 

#include <btBulletDynamicsCommon.h> 
```

5.  After this, create a new pointer object to `btDiscreteDynamicsWorld` as follows:

```cpp
btDiscreteDynamicsWorld* dynamicsWorld; 
```

6.  This object keeps track of all the physics settings and objects in the current scene.

However, before we create `dynamicWorld`, the Bullet Physics library requires some objects to be initialized first.

These required objects are listed as follows:

*   `btBroadPhaseInerface`: Collision detection is actually done in two phases: `broadphase` and `narrowphase`. In the `broadphase`, the physics engine eliminates all the objects that are unlikely to collide. This check is done using the objects' bounding boxes. Then, in the `narrowphase`, the actual shape of the object is used to check the likelihood of a collision. Pairs of objects are created with a strong likelihood of collision. In the following screenshot, the red box around the sphere is used for `broadphase` collision and the white wiremesh of the sphere is used for `narrowphase` collision:

![](img/f0bc4f36-35eb-4c02-b8bc-eef4a2ce1530.png)

*   `btDefaultColliusion` **configuration**: This is used for setting up default memory.
*   `btCollisionDispatcher`**:** A pair of objects that have a strong likelihood of colliding are tested for collision using actual shapes. This is used for getting details of the collision detection, such as which object collided with which other object.

*   `btSequentialImpulseConstraintSolver`: You can create constraints, such as a hinge constraint or slider constraint, which can restrict the motion or rotation of one object about another object. For example, if there is a hinge joint between the wall and the door, then the door can only rotate around the joint and cannot be moved about, as it is fixed at the hinge joint. The constraint solver is responsible for calculating this correctly. The calculation is repeated a number of times to get close to the optimal solution.

In the `init` function, before we create the `sphere` object, we will initialize these objects as follows:

```cpp
//init physics 
btBroadphaseInterface* broadphase = new btDbvtBroadphase(); 
btDefaultCollisionConfiguration* collisionConfiguration = 
   new btDefaultCollisionConfiguration(); 
btCollisionDispatcher* dispatcher = 
   new btCollisionDispatcher(collisionConfiguration); 
btSequentialImpulseConstraintSolver* solver = 
   new btSequentialImpulseConstraintSolver(); 
```

7.  Then, we will create a new `dynamicWorld` by passing the `dispatcher`, `broadphase`, `solver`, and `collisionConfiguration` as parameters to the `btDiscreteDynamicsWorld` function, as follows:

```cpp
dynamicsWorld = new btDiscreteDynamicsWorld(dispatcher, broadphase, solver, collisionConfiguration); 
```

8.  Now that our physics world is created, we can set the parameters for our physics. The basic parameter is gravity. We set its value to real-world conditions, as follows:

```cpp
dynamicsWorld->setGravity(btVector3(0, -9.8f, 0)); 
```

# Adding rigid bodies

Now we can create rigid bodies or soft bodies and watch them interact with other rigid or soft bodies. A rigid body is an animate or inanimate object that doesn't change its shape or physical properties. Soft bodies, on the other hand, can be squishy and made to change shape.

In the following example, we will focus on the creation of a rigid body.

To create a rigid body, we have to specify the shape of the object and the motion state, and then set the mass and inertia of the objects. Shapes are defined using `btCollisionShape`. An object can have different shapes, or sometimes even a combination of shapes, called a compound shape. We use `btBoxShape` to create cubes and cuboids and `btSphereShape` to create spheres. We can also create other shapes, such as `btCapsuleShape`, `btCylinderShape`, and `btConeShape`, which will be used for `narrowphase` collision by the library.

In our case, we are going to create a sphere shape and see our Earth sphere bounce around. So, let's begin:

1.  Using the following code, create a `btSphere` for creating a sphere shape and set the radius as `1.0`, which is the radius of our rendered sphere as well:

```cpp
   btCollisionShape* sphereShape = new btSphereShape(1.0f);   
```

2.  Next, set the `btDefaultMotionState`, where we specify the rotation and position of the sphere, as follows:

```cpp
btDefaultMotionState* sphereMotionState = new btDefaultMotionState(btTransform(btQuaternion(0, 0, 0, 1), btVector3(0, 10.0f, 0))); 
```

We set the rotation to `0` and set the position of the rigid body to a distance of `10.0f` along the *y*-axis. We should also set the mass and inertia and calculate the inertia of the `sphereShape` as follows:

```cpp
btScalar mass = 10.0; 
btVector3 sphereInertia(0, 0, 0); 
sphereShape->calculateLocalInertia(mass, sphereInertia); 

```

3.  To create the rigid body, we first have to create `btRiigidBodyConstructionInfo` and pass the variables to it as follows:

```cpp
btScalar mass = 10.0; 
btVector3 sphereInertia(0, 0, 0); 
sphereShape->calculateLocalInertia(mass, sphereInertia); 

btRigidBody::btRigidBodyConstructionInfo sphereRigidBodyCI(mass, 
sphereMotionState, sphereShape, sphereInertia); 

```

4.  Now, create the rigid body object by passing `btRiigidBodyConstructionInfo` into it using the following code:

```cpp
btRigidBody* sphereRigidBody = new btRigidBody(sphereRigidBodyCI); 
```

5.  Now, set the physical properties of the rigid body, including friction and restitution, using the following code:

```cpp
sphereRigidBody->setRestitution(1.0f); 
sphereRigidBody->setFriction(1.0f);  
```

These values are between `0.0f` and `1.0.0.0`, meaning that the object is really smooth and has no friction, and has no restitution or bounciness. The `1.0` figure, on the other hand, means that the object is rough on the outside and extremely bouncy, like a bouncy ball.

6.  After these necessary parameters are set, we need to add the rigid body to the `dynamicWorld` we created as follows, using the `addRigidBody` function of the `dynamicsWorld`:

```cpp
dynamicsWorld->addRigidBody(sphereRigidBody); 
```

Now, for our sphere mesh to actually behave like the sphere body, we have to pass the rigid body to the sphere mesh class and make some minor changes. Open the `MeshRenderer.h` and `.cpp` files. In the `MeshRenderer.h` file, include the `btBulletDynamicsCommon.h` header and add a local `btRigidBody` called `rigidBody` to the `private` section. You should also change the constructor to take a rigid body, as follows:

```cpp
#include <btBulletDynamicsCommon.h> 

   class MeshRenderer{ 

public: 
MeshRenderer(MeshType modelType, Camera* _camera, btRigidBody* _rigidBody); 
         . 
         . 
   private: 
         . 
         . 
         btRigidBody* rigidBody; 
};
```

7.  In the `MeshRenderer.cpp` file, change the constructor to take a `rigidBody` variable and set the local `rigidBody` variable to it as follows:

```cpp
MeshRenderer::MeshRenderer(MeshType modelType, Camera* _camera, btRigidBody* _rigidBody) { 

   rigidBody = _rigidBody; 
   camera = _camera; 
   . 
   . 
}
```

8.  Then, in the `draw` function, we have to replace the code where we set the `modelMatrix` variable with the code where we get the sphere rigid body value, as follows:

```cpp
   btTransform t; 

   rigidBody->getMotionState()->getWorldTransform(t); 
```

9.  We use the `btTransform` variable to get the transformation from the rigid body's `getMotionState` function and then get the `WorldTransform` variable and set it to our `brTransform` variable `t`, as follows:

```cpp
   btQuaternion rotation = t.getRotation(); 
   btVector3 translate = t.getOrigin(); 
```

10.  We create two new variables of the `btQuaternion` type to store rotation and `btVector3` to store the translation values using the `getRotation` and `getOrigin` functions of the `btTransform` class, as follows:

```cpp
glm::mat4 RotationMatrix = glm::rotate(glm::mat4(1.0f), rotation.getAngle(),glm::vec3(rotation.getAxis().getX(),rotation.getAxis().getY(), rotation.getAxis().getZ())); 

glm::mat4 TranslationMatrix = glm::translate(glm::mat4(1.0f), 
                              glm::vec3(translate.getX(),  
                              translate.getY(), translate.getZ())); 

glm::mat4 scaleMatrix = glm::scale(glm::mat4(1.0f), scale); 
```

11.  Next, we create three `glm::mat4` variables, called `RotationMatrix`, `TranslationMatrix`, and `ScaleMatrix`, and set the values of rotation and translation using the `glm::rotate` and `glm:: translation` functions. We then pass in the rotation and translation values we stored earlier, as shown in the following code. We will keep the `ScaleMatrix` variable as is:

```cpp
   modelMatrix = TranslationMatrix * RotationMatrix * scaleMatrix;  
```

The new `modelMatrix` variable will be the multiplication of the scale, rotation, and translation matrices in that order. The rest of the code will remain the same in the `draw` function.

12.  In the `init` function, change the code to reflect the modified `MeshRenderer` constructor:

```cpp
   // Sphere Mesh 

   sphere = new MeshRenderer(MeshType::kSphere, camera, 
            sphereRigidBody); 
   sphere->setProgram(texturedShaderProgram); 
   sphere->setTexture(sphereTexture); 
   sphere->setScale(glm::vec3(1.0f)); 
```

13.  We don't have to set the position, as that will be set by the rigid body. Set the camera as shown in the following code so that we can see the sphere:

```cpp
camera = new Camera(45.0f, 800, 600, 0.1f, 100.0f, glm::vec3(0.0f, 
         4.0f, 20.0f)); 
```

14.  Now, run the project. We can see the sphere being drawn, but it is not moving. That's because we have to update the physics bodies.
15.  We have to use the `dynamicsWorld` and `stepSimulation` functions to update the simulation every frame. To do this, we have to calculate the delta time between the previous and current frames.
16.  At the top of the `source.cpp`, include `<chrono>` so that we can calculate the tick update. Now, we have to make changes to the `main` function and the `while` loop as follows:

```cpp
auto previousTime = std::chrono::high_resolution_clock::now(); 

while (!glfwWindowShouldClose(window)){ 

         auto currentTime = std::chrono::
                            high_resolution_clock::now(); 
         float dt = std::chrono::duration<float, std::
                    chrono::seconds::period>(currentTime - 
                    previousTime).count(); 

         dynamicsWorld->stepSimulation(dt); 

         renderScene(); 

         glfwSwapBuffers(window); 
         glfwPollEvents(); 

         previousTime = currentTime; 
   } 

```

Just before the `while` loop, we create a variable called `previousTime` and initialize it with the current time. In the `while` loop, we get the current time and store it in the variable. Then, we calculate the delta time between the previous time and the current time by subtracting the two. We have the delta time now, so we call the `stepSimulation` and pass in the delta time. Then we render the scene and swap the buffer and poll for events as usual. Finally, we set the current time as the previous time.

Now, when we run the project, we can see the sphere falling down, which is pretty cool. However, the sphere doesn't interact with anything.

Let's add a box rigid body at the bottom and watch the sphere bounce off it. After the sphere `MeshRenderer` object, add the following code to create a box rigid body:

```cpp
   btCollisionShape* groundShape = new btBoxShape(btVector3(4.0f, 
                                   0.5f, 4.0f)); 

   btDefaultMotionState* groundMotionState = new  
     btDefaultMotionState(btTransform(btQuaternion
     (0, 0, 0, 1), btVector3(0, -2.0f, 0))); 
   btRigidBody::btRigidBodyConstructionInfo 
    groundRigidBodyCI(0.0f, new btDefaultMotionState(), 
    groundShape, btVector3(0, 0, 0)); 

   btRigidBody* groundRigidBody = new btRigidBody(
                                  groundRigidBodyCI); 

   groundRigidBody->setFriction(1.0); 
   groundRigidBody->setRestitution(0.9); 

   groundRigidBody->setCollisionFlags(btCollisionObject
     ::CF_STATIC_OBJECT); 

   dynamicsWorld->addRigidBody(groundRigidBody);  
```

Here, we first create a shape of the `btBoxShape` type with the length, height, and depth set as `4.0`, `0.5`, and `4.0` respectively. Next, we will set the motion state, where we set the rotation to zero and the position at `-2.0` in the *y*-axis and `0` along the *x*- and *z*-axis. For the construction information, we set the mass and intertia to `0`. We also set the default motion state and pass in the shape. Next, we create the rigid body by passing the rigid body information into it. Once the rigid body is created, we set the restitution and friction value. Next, we use the `setCollisionFlags` function of `rigidBody` to set the rigid body type as static. This means that it will be like a brick wall or won't move and be affected by forces from other rigid bodies, but other bodies will still be affected by it.

Finally, we add the ground rigid body to the world so that the box rigid body will be part of the physics simulation as well. We now have to create a `MeshRenderer` cube to render the ground rigid body. Create a new `MeshRenderer` object called `Ground` at the top, under which you created the sphere `MeshRenderer` object. In the `init` function, under which we added the code for the ground rigid body, add the following:

```cpp
   // Ground Mesh 
   GLuint groundTexture = tLoader.getTextureID(
                          "Assets/Textures/ground.jpg"); 
   ground = new MeshRenderer(MeshType::kCube, camera,  
            groundRigidBody); 
   ground->setProgram(texturedShaderProgram); 
   ground->setTexture(groundTexture); 
   ground->setScale(glm::vec3(4.0f, 0.5f, 4.0f));  
```

We will create a new texture by loading `ground.jpg`, so make sure you add it to the `Assets/ Textures` directory. Call the constructor and set the `meshtype` to `cube`, and then set the camera and pass in the ground rigid body. We then set the shader program, texture, and scale of the object.

17.  In the `renderScene` function, draw the ground `MeshRenderer` object as follows:

```cpp
void renderScene(){ 

   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 
   glClearColor(1.0, 1.0, 0.0, 1.0); 

   sphere->draw(); 
   ground->draw(); 
}
```

18.  Now, when you run the project, you will see the sphere bouncing on the ground box:

![](img/5aa9e70f-6bc0-4e07-8212-24d7cebc2dbf.png)

# Summary

In this chapter, we created a new class called `MeshRenderer`, which will be used to render textured 3D objects to our scene. We created a texture-loaded class, which will be used to load the textures from the images provided. Then, we added physics to the object by adding the Bullet Physics library. We then initialized the physics world and created and added the rigid body to the mesh renderer by adding the body itself to the world, causing the rendered object to be affected by physics.

In the next chapter, we will add a gameplay loop, as well as scoring and text rendering to display the score on the viewport. We will also add lighting to our world.