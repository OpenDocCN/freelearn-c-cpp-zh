# Chapter 8. Using 3D Graphics

In this chapter, we will learn how to work and draw with 3D graphics. The recipes in this chapter will cover the following:

*   Drawing 3D geometric primitives
*   Rotating, scaling, and translating
*   Drawing to an offscreen canvas
*   Drawing in 3D with the mouse
*   Adding lights
*   Picking in 3D
*   Creating a height map from an image
*   Creating a terrain with Perlin noise
*   Saving mesh data

# Introduction

In this chapter, we will learn the basics of creating graphics in 3D. We will use OpenGL and some useful wrappers that Cinder includes on some advanced OpenGL features.

# Drawing 3D geometric primitives

In this recipe, we will learn how to draw the following 3D geometric shapes:

*   Cube
*   Sphere
*   Line
*   Torus
*   Cylinder

## Getting ready

Include the necessary header to draw in OpenGL using Cinder commands and statements. Add the following code to the top of your source file:

[PRE0]

## How to do it…

We will create several geometric primitives using Cinder's methods for drawing in 3D.

1.  Declare the member variables with information of our primitives:

    [PRE1]

2.  Initialize the member variables with the position and sizes of the geometry. Add the following code in the `setup` method:

    [PRE2]

3.  Before we draw the shapes, let's also create a camera to rotate around our shapes to give us a better sense of perspective. Declare a `ci::CameraPersp` object:

    [PRE3]

4.  Initialize it in the `setup` method:

    [PRE4]

5.  In the `update` method, we will make the camera rotate around our scene. Add the following code in the `update` method:

    [PRE5]

6.  In the `draw` method, we will clear the background with black and use `mCamera` to define the window's matrices. We will also enable OpenGL to read and write to the depth buffers. Add the following code in the `draw` method:

    [PRE6]

7.  Cinder allows you to draw filled and stroked cubes, so let's draw a cube with a white fill and black stroke:

    [PRE7]

8.  Let's define the drawing color again as white, and draw a sphere with `mSphereCenter` and `mSphereRadius` as the sphere's position and radius, and the number of segments as `30`.

    [PRE8]

9.  Draw a line that begins at `mLineBegin` and ends at `mLineEnd`:

    [PRE9]

10.  Cinder draws a `Torus` at the coordinates of the origin `[0,0]`. So, we will have to translate it to the desired position at `mTorusPos`. We will be using `mTorusOuterRadius` and `mTorusInnerRadius` to define the shape's inner and outer sizes:

    [PRE10]

11.  Finally, Cinder will draw a cylinder at the origin `[0,0]`, so we will have to translate it to the position defined in `mCylinderPosition`. We will also be using `mCylinderBaseRadius` and `mCylinderTopRadius`, to set the cylinder's bottom and top sizes and `mCylinderHeight`, to set its height:

    [PRE11]

    ![How to do it…](img/8703OS_8_1.jpg)

## How it works…

Cinder's drawing methods use OpenGL calls internally to provide fast and easy drawing routines.

The method `ci::gl::color` sets the drawing color so that all shapes will be drawn with that color until another color is set by calling `ci::gl::color` again.

## See also

To learn more about OpenGL transformations such as translation, scale, and rotation, please read the recipe *Rotating, scaling, and translating*.

# Rotating, scaling, and translating

In this recipe, we will learn how to transform our graphics using OpenGL transformations.

We will draw a unit cube at `[0,0,0]` coordinates and then we will translate it to the center of the window, apply rotation, and scale it to a more visible size.

## Getting ready

Include the necessary files to draw with OpenGL and add the helpful `using` statements. Add the following code to the top of the source file:

[PRE12]

## How to do it…

We will apply rotation, translation, and scaling to alter the way our cube is rendered. We will use Cinder's wrappers for OpenGL.

1.  Let's declare variables to store our values for the translation, rotation, and scale transformations:

    [PRE13]

2.  To define the translation amount, let's translate half the window's width on the x axis and half the window's height on the y axis. This will bring anything we draw at `[0,0,0]` to the center of the window. Add the following code in the `setup` method:

    [PRE14]

3.  Let's set the scale factor to be `100` on the x axis, `200` on the y axis, and `100` on the z axis. Anything we draw will be 100 times bigger on the x and z axes and 200 times bigger on the y axis. Add the following code in the `setup` method:

    [PRE15]

4.  In the `update` method, we will animate the rotation values by incrementing the rotation on the x and y axes.

    [PRE16]

5.  In the `draw` method, let's begin by clearing the background with black, setting the windows matrices to allow for drawing in 3D, and enabling OpenGL to read and write the depth buffer:

    [PRE17]

6.  Let's add a new matrix to the stack and translate, scale, and rotate using the previously defined variables:

    [PRE18]

7.  Draw a unit quad at the origin `[0,0,0]` with a white fill and black stroke:

    [PRE19]

8.  Finally, remove the previously added matrix:

    [PRE20]

    ![How to do it…](img/8703OS_8_2.jpg)

## How it works…

The calls to `ci::gl::enableDepthRead` and `ci::gl::enableDepthWrite` respectively, enable reading and writing to the depth buffer. The depth buffer is where the depth information is stored.

When reading and writing to the depth buffer is enabled, OpenGL will sort objects so that closer objects are drawn in front of farther objects. When reading and writing to the depth buffer, the disabled objects will be drawn in the order they where created.

The methods `ci::gl::translate`, `ci::gl::rotate`, and `ci::gl::scale` are wrappers of OpenGL commands for translating, rotating, and scaling, which allow you to pass Cinder types as parameters.

Transformations in OpenGL are applied by multiplying vertex coordinates with transformation matrices. When we call the method `ci::gl::pushMatrices`, we add a copy of the current transformation matrix to the matrix stack. Calls to `ci::gl::translate`, `ci::gl::rotate`, or `ci::gl::scale` will apply the correspondent transformations to the last matrix in the stack, which will be applied to whatever geometry is created after calling the transformation methods. A call to `ci::gl::popMatrix` will remove the last transformation matrix in the stack so that transformations added to the last matrix will no longer affect our geometry.

# Drawing to an offscreen canvas

In this recipe, we will learn how to draw in an offscreen canvas using the OpenGL **Frame Buffer Object** (**FBO**).

We will draw in an FBO and draw it onscreen as well as texture a rotating cube.

## Getting ready

Include the necessary files to work with OpenGL and the FBOs as well as the useful `include` directives.

Add the following code to the top of the source file:

[PRE21]

## How to do it…

We will use a `ci::gl::Fbo` object, a wrapper to an OpenGL FBO, to draw in an offscreen destination.

1.  Declare a `ci::gl::Fbo` object as well as a `ci::Vec3f` object to define the cube's rotation:

    [PRE22]

2.  Initialize `mFbo` with a size of 256 x 256 pixels by adding the following code in the `setup` method:

    [PRE23]

3.  Animate `mCubeRotation` in the `update` method:

    [PRE24]

4.  Declare a method where we will draw to the FBO:

    [PRE25]

5.  In the implementation of `drawToFbo`, we will begin by creating a `ci::gl::SaveFramebufferBinding` object and then bind `mFbo`.

    [PRE26]

6.  Now we will clear the background with a dark gray color and set the matrices using the FBO's width and height.

    [PRE27]

7.  Now we will draw a rotating color cube at the center of the FBO with size `100` and using `mCubeRotation` to rotate the cube.

    [PRE28]

8.  Let's move to the implementation of the `draw` method. Start by calling the method `drawToFbo`, clearing the background with black, setting the window's matrices, and enable reading and writing to the depth buffer. Add the following code in the `draw` method:

    [PRE29]

    Lets draw our Fbo at the top left corner of the window using mFbo texture:

    [PRE30]

9.  Enable and bind the texture of `mFbo`:

    [PRE31]

10.  Draw a rotating cube at the center of the window using `mCubeRotation` to define its rotation:

    [PRE32]

11.  To finalize, unbind the texture of `mFbo`:

    [PRE33]

    ![How to do it…](img/8703OS_8_3.jpg)

## How it works…

The class `ci::gl::Fbo` wraps an OpenGL FBO**.**

Frame Buffer Objects are OpenGL objects that contain a collection of buffers that can be used as rendering destinations. The OpenGL context provides a default frame buffer where rendering occurs. Frame Buffer Objects allow rendering to alternative, offscreen locations.

The FBO has a color texture where the graphics are stored, and it can be bound and drawn like a regular OpenGL texture.

On step 5, we created a `ci::gl::SaveFramebufferBinding` object, which is a helper class that restores the previous FBO state. When using OpenGL ES, this object will restore and bind the previously bound FBO (usually the *screen* FBO) when it is destroyed.

## See also

See the recipe *Rotating, scaling, and translating* to learn more about OpenGL transformations.

# Drawing in 3D with the mouse

In this recipe, we will draw with the mouse on a 3D space. We will draw lines when dragging the mouse or rotate the scene in 3D when dragging and pressing the *Shift* key simultaneously.

## Getting ready

Include the necessary files to draw using OpenGL, as well as the files needed to use Cinder's perspective, Maya camera, and poly lines.

[PRE34]

Also, add the following `using` statements:

[PRE35]

## How to do it…

We will use the `ci::CameraPersp` and `ci::Ray` classes to convert the mouse coordinates to our rotated 3D scene.

1.  Declare a `ci::MayaCamUI` object and a `std::vector` object of `ci::PolyLine<ci::Vec3f>` to store the drawn lines:

    [PRE36]

2.  In the `setup` method, we will create `ci::CameraPersp` and set it up so that the point of interest is the center of the window. We will also set the camera as the current camera of `mCamera:`

    [PRE37]

3.  In the `draw` method, let's clear the background with black and use our camera to set the window's matrices.

    [PRE38]

4.  Now let's iterate `mLines` and draw each `ci::PolyLine`. Add the following code to the `draw` method:

    [PRE39]

5.  With our scene set up and the lines being drawn, we need to create the 3D perspective! Let's start by declaring a method to convert coordinates from the screen position to world position. Add the following method declaration:

    [PRE40]

6.  In the `screenToWorld` implementation, we need to generate a ray from `point` using the cameras perspective. Add the following code in `screenToWorld`:

    [PRE41]

7.  Now we need to calculate where the ray will intersect with a perpendicular plane at the camera's center of interest and then return the intersection point. Add the following code in the `screenToWorld` implementation:

    [PRE42]

8.  Let's use the previously defined method to draw with the mouse. Declare the `mouseDown` and `mouseDrag` event handlers:

    [PRE43]

9.  In the implementation of `mouseDown`, we will check if the *Shift* key is being pressed. If it is, we will call the `mouseDown` method of `mCamera`, otherwise, we will add `ci::PolyLine<ci::Vec3f>` to `mLines`, calculate the world position of the mouse cursor using `screenToWorld`, and add it:

    [PRE44]

10.  In the implementation of `mouseDrag`, we will check if the *Shift* key is being pressed. If it is, we will call the `mouseDrag` method to `mCamera`, otherwise, we will calculate the world position of the mouse cursor and add it to last line in `mLines`.

    [PRE45]

11.  Build and run the application. Press and drag the mouse to draw a line. Press the *Shift* key and press and drag the mouse to rotate the scene.

## How it works…

We use `ci::MayaCamUI` to easily rotate our scene.

The `ci::Ray` class is a representation of a ray, containing an origin, direction, and an infinite length. It provides useful methods to calculate intersections between rays and triangles or planes.

To calculate the world position of the mouse cursor we calculated a ray going from the camera's eye position in the camera's view direction.

We then calculated the intersection of the ray with the plane at the center of the scene, perpendicular to the camera.

The calculated position is then added to a `ci::PolyLine<ci::Vec3f>` object to draw the lines.

## See also

*   To learn more on how to use `ci::MayaCamUI`, please refer to the recipe *Using MayaCamUI* from [Chapter 2](ch02.html "Chapter 2. Preparing for Development"), *Preparing for Development*.
*   To learn how to draw in 2D, please read the recipe *Drawing arbitrary shapes with the mouse* from [Chapter 7](ch07.html "Chapter 7. Using 2D Graphics"), *Using 2D Graphics*.

# Adding lights

In this chapter, we will learn how to illuminate a 3D scene using OpenGL lights.

## Getting ready

Include the necessary files to use OpenGL lights, materials, and draw. Add the following code to the top of the source file:

[PRE46]

Also add the following `using` statements:

[PRE47]

## How to do it…

We will use the default OpenGL light rendering methods to illuminate our scene. We will use the `ci::gl::Material` and `ci::gl::Light` classes, which are wrappers around the OpenGL functionality.

1.  Declare `ci::gl::Material` to define the material properties of the objects being drawn and `ci::Vec3f` to define the lights position.

    [PRE48]

2.  Let's set the materials `Ambient`, `Diffuse`, `Specular`, `Emission`, and `Shininess` properties by adding the following code in the `setup` method:

    [PRE49]

3.  In the `update` method, we will use the mouse to define the light position. Add the following code in the `update` method:

    [PRE50]

4.  In the `draw` method, we will begin by clearing the background, setting the window's matrices, and enabling reading and writing to the depth buffer.

    [PRE51]

5.  Let's create an OpenGL light using a `ci::gl::Light` object. We will define it as a `POINT` light and set its ID to `0`. We will also set its position to `mLightPos` and define its attenuation.

    [PRE52]

6.  Let's enable OpenGL lighting, the previously created light, and apply the material.

    [PRE53]

7.  Let's draw a rotating `Torus` at the center of the window and use the elapsed seconds to rotate it. Add the following code to the `draw` method:

    [PRE54]

8.  Finally, disable the light:

    [PRE55]

9.  Build and run the application; you will see a red rotating torus. Move the mouse to change the lights position.![How to do it…](img/8703OS_8_4.jpg)

## How it works…

We are using the `ci::gl::Material` and `ci::gl::Light` objects, which are helper classes to define the properties of lights and materials.

The material properties defined in the `setup` method, work in the following ways:

| Material Property | Function |
| --- | --- |
| Ambient | How an object can reflect light that comes in all directions. |
| Diffuse | How an object reflects light that comes from a specific direction or position. |
| Specular | The light that an object will reflect as a result of diffuse lighting. |
| Emission | Light emitted by the object. |
| Shininess | The angle that the object will reflect specular light. Has to be a value between 1 and 128. |

The material ambient, diffuse, and specular colors will multiply with the ambient, diffuse, and specular colors coming from the light source, which are all white by default.

It is possible to define three different types of lights. In the previous example, we defined our light source to be of type `ci::gl::Light::POINT`.

Here are the available types of light and their properties:

| Light Type | Properties |
| --- | --- |
| `ci::gl::Light::POINT` | Point light is the light coming from a specific position in space and illuminating in all directions. |
| `ci::gl::Light::DIRECTION` | Directional light simulates light coming from a position so far away that all light rays are parallel and arrive in the same direction. |
| `ci::gl::Light::SPOTLIGHT` | Spotlight is the light coming from a specific position in space and a specific direction. |

We also defined the attenuation values. Lights in OpenGL allow for defining the values for the constant attenuation, linear attenuation, and quadratic attenuation. These define how the light becomes dimmer as the distance from the light source increases.

To illuminate geometry, it is necessary to calculate the normal for each vertex. All shapes created using Cinder's commands have their normal calculated for us, so we don't have to worry about that.

## There's more…

It is also possible to define the ambient, diffuse, and specular colors coming from the light source. The values defined in these colors will multiply with the correspondent colors of the material.

Here are the `ci::gl::Light` methods that allow you to define the light colors:

| Method | Light |
| --- | --- |
| `setAmbient( const Color& color )` | Color of the ambient light. |
| `setDiffuse( const Color& color )` | Color of the diffuse light. |
| `setSpecular( const Color& color )` | Color of the specular light. |

It is possible to create more than one light source. The amount of lights is dependent on the implementation of the graphics card, but it is always at least `8`.

To create more light sources, simply create more `ci::gl::Light` objects and make sure each gets a unique ID.

## See also

Please read the recipe *Calculating vertex normals* to learn how to calculate the vertex normals for user created geometry.

# Picking in 3D

In this recipe, we will calculate the intersection of the mouse cursor with a 3D model.

## Getting ready

Include the necessary files to draw using OpenGL, use textures and load images, load 3D models, define OpenGL lights and materials, and use Cinder's Maya camera.

[PRE56]

Also, add the following `using` statements:

[PRE57]

We will use a 3D model, so place a file and its texture in the `assets` folder. For this example, we will be using a mesh file named `ducky.msh` and a texture named `ducky.png`.

## How to do it…

1.  We will use the `ci::CameraPersp` and `ci::Ray` classes to convert the mouse coordinates to our rotated 3D scene and calculate the intersection with a 3D model.
2.  Declare the members to define the 3D model and its intersection with the mouse, as well as a `ci::MayaCamUI` object for easy navigation, and a `ci::gl::Material` for lighting:

    [PRE58]

3.  Declare a method where we will calculate the intersection between a `ci::Ray` class and the triangles that make up `mMesh`.

    [PRE59]

4.  In the `setup` method, lets load the model and texture and calculate its bounding box:

    [PRE60]

5.  Let's define the camera and make it look as if it's at the center of the model. Add the following code in the `setup` method:

    [PRE61]

6.  Finally, set up the material for the model's lighting.

    [PRE62]

7.  Declare the handlers for the `mouseDown` and `mouseDrag` events.

    [PRE63]

8.  Implement these methods by calling the necessary methods of `mCam`:

    [PRE64]

9.  Let's implement the `update` method and calculate the intersection between the mouse cursor and our model. Let's begin by getting the mouse position and then calculate `ci::Ray` emitting from our camera:

    [PRE65]

10.  Let's perform a fast test and check if the ray intersects with the model's bounding box. If the result is `true`, we will call the `calcIntersectionWithMeshTriangles` method.

    [PRE66]

11.  Let's implement the `calcIntersectionWithMeshTriangles` method. We will iterate over all the triangles of our model and calculate the nearest intersection and store its index.

    [PRE67]

12.  Let's check if there was any intersection and calculate its position and normal. If no intersection was found, we will simply set `mIntersects` to `false`.

    [PRE68]

13.  With the intersection calculated, let's draw the model, intersection point, and normal. Start by clearing the background with black, setting the window's matrices using our camera, and enabling reading and writing to the depth buffer. Add the following code in the `draw` method:

    [PRE69]

14.  Now let's create a light and set its position as the camera's eye position. We'll also enable the light and apply the material.

    [PRE70]

15.  Now enable and bind the models texture, draw the model, and disable both texture and lighting.

    [PRE71]

16.  Finally, we will check if `mIntersects` is `true` and draw a sphere at the intersection point and the normal vector.

    [PRE72]

    ![How to do it…](img/8703OS_8_5.jpg)

## How it works…

To calculate the intersection of the mouse with the model in 3D, we generated a ray from the mouse position towards the view direction of the camera.

For performance reasons, we first calculate if the ray intersects with the model's bounding box. In case there is an intersection with the model, we further calculate the intersection between the ray and each triangle that makes up the model. For every intersection found, we check its distance and calculate the intersection point and the normal of only the nearest intersection.

# Creating a height map from an image

In this recipe, we will learn how to create a point cloud based on an image selected by the user. We will create a grid of points where each point will correspond to a pixel. The x and y coordinates of each point will be equal to the pixel's position on the image, and the z coordinate will be calculated based on its color.

## Getting ready

Include the necessary files to work with OpenGL, image surfaces, VBO meshes, and loading images.

Add the following code to the top of the source file:

[PRE73]

Also, add the following `using` statements:

[PRE74]

## How to do it…

We will learn how to read pixel values from an image and create a point cloud.

1.  Declare `ci::Surface32f` to store the image pixels, `ci::gl::VboMesh` that we will use as the point cloud, and `ci::MayaCamUI` for easy rotation of our scene.

    [PRE75]

2.  In the `setup` method, we will first open a file load dialog and then let the user select the image to use and check if it returns a valid path.

    [PRE76]

3.  Next, let's load the image and initialize `mPointCloud`. We will set the `ci::gl::VboMesh::Layout` to have dynamic positions and colors so that we will be able to change them later.

    [PRE77]

4.  Next, we'll iterate over the image's pixels and update the vertices in `mPointCloud`.

    [PRE78]

5.  Now we will set up the camera so that it will rotate around the center of the point cloud and close the `if` statement we began on the second step.

    [PRE79]

6.  Let's declare and implement the necessary mouse event handlers to use `mCam`.

    [PRE80]

7.  And implement them:

    [PRE81]

8.  In the `draw` method, we will begin by clearing the background, setting the window's matrices defined by `mCam`, and enable reading and writing the depth buffer.

    [PRE82]

9.  Finally, we will check if `mPointCloud` is a valid object and draw it.

    [PRE83]

10.  Build and run the application. You will be prompted with a dialog box to select an image file. Select it and you will see a point cloud representation of the image. Drag the mouse cursor to rotate the scene.![How to do it…](img/8703OS_8_6.jpg)

## How it works…

We started by loading an image into `ci::Surface32f.` This surface stores pixels as float numbers in the range from `0` to `1`.

We created a grid of points where the `x` and `y` coordinates represented the pixel's position on the image and the `z` coordinate was the length of the color's vector.

The point cloud is represented by a `ci::gl::VboMesh`, which is a mesh of vertices, normal, colors, and indexes with an underlying Vertex Buffer Object. It allows for optimized drawing of geometry.

# Creating a terrain with Perlin noise

In this recipe, we will learn how to construct a surface in 3D using **Perlin noise** to create organic deformations that resemble a piece of terrain.

## Getting ready

Include the necessary files to draw using OpenGL, Perlin noise, a Maya camera for navigation, and Cinder's math utilities. Add the following code to the top of the source file:

[PRE84]

Also, add the following `using` statements:

[PRE85]

## How to do it…

We will create a grid of 3D points and use Perlin noise to calculate a smooth surface.

1.  Declare `struct` to store the vertices of the terrain by adding the following code before the applications class declaration:

    [PRE86]

2.  Add the following members to the applications class declaration:

    [PRE87]

3.  In the `setup` method, define the number of rows and lines that will make up the terrain's grid. Also, define the gap distance between each point.

    [PRE88]

4.  Add the vertices to `mTerrain` by creating a grid of points laid on the `x` and `z` axis. We will use the values generated by `ci::Perlin` to calculate each points height. We will also use the height of the points to define their color:

    [PRE89]

5.  Now let's define our camera so that it points to the center of the terrain.

    [PRE90]

6.  Declare the mouse event handlers to use `mCam`.

    [PRE91]

7.  Now let's implement the mouse handlers.

    [PRE92]

8.  In the `draw` method, let's start by clearing the background, setting the matrices using `mCam`, and enabling reading and writing of the depth buffer.

    [PRE93]

9.  Now enable OpenGL to use the `VERTEX` and `COLOR` arrays:

    [PRE94]

10.  We will use a nested `for` loop to iterate over the terrain and draw each strip of terrain as `GL_TRIANGLE_STRIP`.

    [PRE95]

    ![How to do it…](img/8703OS_8_7.jpg)

## How it works…

Perlin noise is a coherent random number generator capable of creating organic textures and transitions.

We used the values created by the `ci::Perlin` object to calculate the height of the vertices that make up the terrain and create smooth transitions between vertices.

## There's more…

We can also animate our terrain by adding an increasing offset to the coordinates used to calculate the Perlin noise. Declare the following member variables in your class declaration:

[PRE96]

In the `setup` method, initialize them.

[PRE97]

In the `update` method animate each offset value by adding `0.01`.

[PRE98]

Also in the `update` method, we will iterate over all the vertices of `mTerrain`. For each vertex we will use its `x` and `z` coordinates to calculate the `Y` coordinate with `mPerlin noise`, but we will offset the coordinates.

[PRE99]

# Saving mesh data

Provided that you are using a `TriMesh` class to store 3D geometry, we will show you how to save it in a file.

## Getting ready

We are assuming that you are using a 3D model stored in `TriMesh` object. Sample application loading 3D geometry can be found in `Cinder samples` directory in the folder: `OBJLoaderDemo`.

## How to do it…

We will implement saving a 3D mesh data.

1.  Include necessary headers:

    [PRE100]

2.  Implement your `keyDown` method as follows:

    [PRE101]

## How it works…

In Cinder we are using a `TriMesh` class to store 3D geometry. Using `TriMesh` we can store and manipulate geometry loaded from 3D model files or add each vertices with code.

Every time you hit the *S* key on the keyboard, a saving dialog pops up to ask you where to save binary data of the `TriMesh` object. When you press the *O* key, the OBJ format file will be saved into your `documents` folder. If you don't have to exchange data with other software, binary data saving and loading is usually faster.