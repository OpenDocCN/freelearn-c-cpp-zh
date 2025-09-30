# Chapter 3. Interactive 3D Data Visualization

In this chapter, we will cover the following topics:

*   Setting up a virtual camera for 3D rendering
*   Creating a 3D plot with perspective rendering
*   Creating an interactive environment with GLFW
*   Rendering a volumetric dataset – MCML simulation

# Introduction

OpenGL is a very attractive platform for creating dynamic, highly interactive tools for visualizing data in 3D. In this chapter, we will build upon the fundamental concepts discussed in the previous chapter and extend our demos to incorporate more sophisticated OpenGL features for 3D rendering. To enable 3D visualization, we will first introduce the basic steps of setting up a virtual camera in OpenGL. In addition, to create more interactive demos, we will introduce the use of GLFW callback functions for handling user inputs. Using these concepts, we will illustrate how to create an interactive 3D plot with perspective rendering using OpenGL. Finally, we will demonstrate how to render a 3D volumetric dataset generated from a Monte Carlo simulation of light transport in biological tissue. By the end of this chapter, readers will be able to visualize data in 3D with perspective rendering and interact with the environment dynamically through user inputs for a wide range of applications.

# Setting up a virtual camera for 3D rendering

Rendering a 3D scene is similar to taking a photograph with a digital camera in the real world. The steps that are taken to create a photograph can also be applied in OpenGL.

For example, you can move the camera from one position to another and adjust the viewpoint freely in space, which is known as **viewing transformation**. You can also adjust the position and orientation of the the object of interest in the scene. However, unlike in the real world, in the virtual world you can position the object at any orientation freely without any physical constraints, termed as **modeling transformation**. Finally, we can exchange camera lenses to adjust the zoom and create different perspectives the process is called **projection transformation**.

When you take a photo applying the viewing and modeling transformation, the digital camera takes the information and creates an image on your screen. This process is called **rasterization**.

These sets of matrices—encompassing the viewing transformation, modeling transformation, and projection transformation—are the fundamental elements we can adjust at run-time, which allows us to create an interactive and dynamic rendering of the scene. To get started, we will first look into the setup of the camera matrix, and how we can create a scene with different perspectives.

## Getting ready

The source code in this chapter is based on the final demo from the previous chapter. Basically, we will be modifying the previous implementation by setting up a camera model using a perspective matrix. In the upcoming chapters, we will explore the use of the **OpenGL Shading Language** (**GLSL**) to enable even more complex rendering techniques and higher performance.

## How to do it...

Let's get started on the first new requirement for handling perspective transformation in OpenGL. Since the camera parameters depend on the window size, we need to first implement a callback function that handles a window resize event and updates the matrices accordingly:

1.  Define the function prototype for the callback function:

    [PRE0]

2.  Preset the camera parameters: the vertical **field of view angle** (**fovY**), the distance to the **Near clipping plane** (front), the distance to **Far clipping plane** (back), and the screen aspect ratio (**width**/**height**):![How to do it...](img/9727OS_03_01.jpg)

    [PRE1]

3.  Set up the viewport of the virtual camera (using the window size):

    [PRE2]

4.  Specify the matrix mode as `GL_PROJECTION` and allow subsequent matrix operations to be applied to the projection matrix stack:

    [PRE3]

5.  Load the identity matrix to the current matrix (that is, reset the matrix to its default state):

    [PRE4]

6.  Set up the perspective projection matrix for the virtual camera:

    [PRE5]

## How it works...

The purpose of the `framebuffer_size_callback` function is to handle callback events from the GLFW library. Upon resizing the window, an event will be captured and the callback function provides a mechanism to update the virtual camera parameters accordingly. One important problem is that changing the aspect ratio of the screen can introduce distortion if we do not adjust our virtual camera rendering parameters appropriately. Therefore, the `update` function also calls the `glViewport` function to ensure that the graphic is rendered onto the new viewable area.

Furthermore, imagine we are taking a photo of a scene with a camera physically in the real world. The `gluPerspective` function basically controls the camera lens' zoom (that is, the field of view angle) as well as the camera sensor (that is, the image plane) aspect ratio. One major difference between the virtual and real camera is the concept of a near clipping and far clipping plane (front and back variables) that limits the viewable area of the rendered image. These constraints are related to more advanced topics (the depth buffer and depth testing) and how the graphical engine works with a virtual 3D scene. One rule of thumb is, we should never set an unnecessarily large value as it will affect the precision of the depth testing result, which can lead to z-fighting issue. **Z-fighting** is a phenomenon that occurs when objects share very similar depth values and the precision of the depth value is not sufficient to resolve the ambiguity (due to precision loss in the floating-point representation during the 3D rendering process). Setting a higher resolution depth buffer, or reducing the distance between the clipping planes, is often the simplest way to mitigate such problems.

The sample code provides perspective rendering of a scene that resembles how the human eye sees the world. For example, an object will appear larger if it is closer to the camera and smaller if it is farther away. This allows for a more realistic view of a scene. On the other hand, by controlling the field of view angle, we can exaggerate perspective distortion, similar to capturing a scene with an ultra-wide angle lens.

## There's more...

Alternatively, we can set up the camera with the `glFrustum()` function by replacing the `gluPerspective()` function with the following code:

[PRE6]

The `glFrustum` function takes the corners of the near clipping and far clipping planes to construct the projective matrix. Fundamentally, there is no difference between the `gluPerspective` and `glFrustum` functions, so they are interchangeable.

As we can see, the virtual camera in OpenGL can be updated upon changes to the screen frame buffer (window size) and these event updates are captured with the callback mechanism of the GLFW library. Of course, we can also handle other events such as keyboard and mouse inputs. Further details on how to handle additional events will be discussed later. In the next section, let's implement the rest of the demo to create our first 3D plot with perspective rendering.

# Creating a 3D plot with perspective rendering

In the previous chapter, we showed a heat map of a 2D Gaussian distribution with varying standard deviation over time. Now, we will continue with more advanced rendering of the same dataset in 3D and demonstrate the effectiveness of visualizing multi-dimensional data with OpenGL. The code base from the previous chapter will be modified to enable 3D rendering.

Instead of rendering the 2D Gaussian distribution function on a plane, we take the output of the Gaussian function ![Creating a 3D plot with perspective rendering](img/9727OS_03_13.jpg) as the z (height) value as follows:

![Creating a 3D plot with perspective rendering](img/9727OS_03_20.jpg)

Here **A** is the amplitude of the distribution centered at ![Creating a 3D plot with perspective rendering](img/9727OS_03_21.jpg), and ![Creating a 3D plot with perspective rendering](img/9727OS_03_12.jpg) are the standard deviations (spread) of the distribution in the *x* and *y* directions. In our example, we will vary the spread of the distribution over time to change its shape in 3D. Additionally, we will apply a heat map to each vertex based on the height for better visualization effect.

## Getting ready

With the camera set up using the projection model, we can render our graph in 3D with the desired effects by changing some of the virtual camera parameters such as the field of view angle for perspective distortion as well as the rotation angles for different viewing angles. To reduce coding complexity, we will re-use the `draw2DHeatMap` and `gaussianDemo` functions implemented in [Chapter 2](ch02.html "Chapter 2. OpenGL Primitives and 2D Data Visualization"), *OpenGL Primitives and 2D Data Visualization* with minor modifications. The rendering techniques will be based on the OpenGL primitives described in the previous chapter.

## How to do it...

Let's modify the final demo in [Chapter 2](ch02.html "Chapter 2. OpenGL Primitives and 2D Data Visualization"), *OpenGL Primitives and 2D Data Visualization* (`main_gaussian_demo.cpp` in the code package) to enable perspective rendering in 3D. The overall code structure is provided here to orient readers first and major changes will be discussed in smaller blocks sequentially:

[PRE7]

With the preceding framework in mind, inside the `main` function let's add the new `callback` function for handling window resizing implemented in the previous section:

[PRE8]

Let's define several global variables and initialize them for perspective rendering, including the zoom level (`zoom`) and rotation angles around the *x* (`beta`) and *z* (`alpha`) axes, respectively:

[PRE9]

In addition, outside the `main` loop, let's initialize some parameters for rendering the Gaussian distribution, including the standard deviation (sigma), sign, and step size for dynamically changing the function over time:

[PRE10]

In the `while` loop, we perform the following transformations to render the Gaussian function in 3D:

1.  Specify the matrix mode as `GL_MODELVIEW` to allow subsequent matrix operations to be applied to the `MODELVIEW` matrix stack:

    [PRE11]

2.  Perform the translation and rotation of the object:

    [PRE12]

3.  Draw the origin (with the *x*, *y*, and *z* axes) and the Gaussian function in 3D. Dynamically plot a series of Gaussian functions with varying sigma values over time and reverse the sign once a certain threshold is reached:

    [PRE13]

    For handling each of the preceding drawing tasks, we implement the origin visualizer, Guassian function generator, and 3D point visualizer in separate functions.

To visualize the origin, implement the following drawing function:

1.  Define the function prototype:

    [PRE14]

2.  Draw the *x*, *y*, and *z* axes in red, green, and blue, respectively:

    [PRE15]

For the implementation of the Gaussian function demo, we have broken down the problem into two parts: a Gaussian data generator and a heat map visualizer function with point drawing. Together with 3D rendering and the heat map, we can now clearly see the shape of the Gaussian distribution and how the samples animate and move in space over time:

1.  Generate the Gaussian distribution:

    [PRE16]

2.  Next, implement the `draw2DHeatMap` function to visualize the result. Note that, unlike in [Chapter 2](ch02.html "Chapter 2. OpenGL Primitives and 2D Data Visualization"), *OpenGL Primitives and 2D Data Visualization*, we use the z value inside the `glVertex3f` function:

    [PRE17]

The rendered result is shown in the following screenshot. We can see that the transparency (alpha blending) allows us to see through the data points and provides a visually appealing result:

![How to do it...](img/9727OS_03_02.jpg)

## How it works...

This simple example demonstrates the use of perspective rendering as well as OpenGL transformation functions to rotate and translate the rendered objects in virtual space. As you can see, the overall code structure remains the same as in [Chapter 2](ch02.html "Chapter 2. OpenGL Primitives and 2D Data Visualization"), *OpenGL Primitives and 2D Data Visualization* and the major changes primarily include setting up the camera parameters for perspective rendering (inside the `framebuffer_size_callback` function) and performing the required transformations to render the Gaussian function in 3D (after setting the matrix mode to `GL_MODELVIEW`). Two very commonly used transformation functions to manipulate virtual objects include `glRotatef` and `glTranslatef`, which allow us to position objects at any orientation and position. These functions can significantly improve the dynamics of your own application, with very minimal cost in development and computation time since they are heavily optimized.

The `glRotatef` function takes four parameters: the rotation angle and three components of the direction vector *(x, y, z)*, which define the axis of rotation. The function also replaces the current matrix with the product of the rotation matrix and the current matrix:

![How it works...](img/9727OS_03_14.jpg)

Here ![How it works...](img/9727OS_03_15.jpg) and ![How it works...](img/9727OS_03_16.jpg).

## There's more...

One may ask, what if we would like to position two objects at different orientations and positions? What if we would like to position many parts in space relative to one another? The answer to these is to use the `glPushMatrix` and `glPopMatrix` functions to control the stack of transformation matrices. The concept behind this can get relatively complex for a model with a large number of parts and keeping a history of the state machine with many components can be tedious. To address this issue, we will look into newer versions of GLSL support (OpenGL 3.x and higher).

# Creating an interactive environment with GLFW

In the previous two sections, we focused on the creation of 3D objects and on utilizing basic OpenGL rendering techniques with a virtual camera. Now, we are ready to incorporate user inputs, such as mouse and keyboard inputs, to enable more dynamic interactions using camera control features such as zoom and rotate. These features will be the fundamental building blocks for the upcoming applications and the code will be reused in later chapters.

## Getting ready

The GLFW library provides a mechanism to handle user inputs from different environments. The event handlers are implemented as callback functions in C/C++, and, in the previous tutorials, we bypassed these options for the sake of simplicity. To get started, we first need to enable these callback functions and implement basic features to control the rendering parameters.

## How to do it...

To handle keyboard inputs, we attach our own implementation of the `callback` functions back to the event handler of GLFW. We will perform the following operations in the `callback` function:

1.  Define the following global variables (including a new variable called `locked` to track whether the mouse button is pressed down, as well as the angles of rotation and zoom level) that will be updated by the `callback` functions:

    [PRE18]

2.  Define the keyboard `callback` function prototype:

    [PRE19]

3.  If we receive any event other than the key press event, ignore it:

    [PRE20]

4.  Create a `switch` statement to handle each key press case:

    [PRE21]

5.  If the *Esc* key is pressed, exit the program:

    [PRE22]

6.  If the space bar is pressed, start or stop the animation by toggling the variable:

    [PRE23]

7.  If the direction keys (up, down, left, and right) are pressed, update the variables that control the angles of rotation for the rendered object:

    [PRE24]

8.  Lastly, if the *Page Up* or *Page Down* keys are pressed, zoom in and out from the object by updating the `zoom` variable:

    [PRE25]

To handle mouse click events, we implement another `callback` function similar to the one for the keyboard. The mouse click event is rather simple as there is only a limited set of buttons available:

1.  Define the mouse press `callback` function prototype:

    [PRE26]

2.  Ignore all inputs except for the left click event for simplicity:

    [PRE27]

3.  Toggle the `lock` variable to store the mouse hold event. The `lock` variable will be used to determine whether the mouse movement is used for rotating the object:

    [PRE28]

For handling mouse movement events, we need to create another `callback` function. The `callback` function for mouse movement takes the *x* and *y* coordinates from the window instead of unique key inputs:

1.  Define the `callback` function prototype that takes in the mouse coordinates:

    [PRE29]

2.  Upon mouse press and mouse movement, we update the rotation angles of the object with the *x* and *y* coordinates of the mouse:

    [PRE30]

Finally, we will implement the mouse scroll callback function, which allows users to scroll up and down to zoom in and zoom out of the object.

1.  Define the `callback` function prototype that captures the `x` and `y` scroll variables:

    [PRE31]

2.  Take the y parameter (up and down scroll) and update the zoom variable:

    [PRE32]

With all of the `callback` functions implemented, we are now ready to link these functions to the GLFW library event handlers. The GLFW library provides a platform-independent API for handling each of these events, so the same code will run in Windows, Linux, and Mac OS X seamlessly.

To integrate the callbacks with the GLFW library, call the following functions in the `main` function:

[PRE33]

The end result is an interactive interface that allows the user to control the rendering object freely in space. First, when the user scrolls the mouse (see the following screenshots), we translate the object forward or backward. This creates the visual perception that the object is zoomed in or zoomed out of the camera:

![How to do it...](img/9727OS_03_03.jpg)

Here is another screenshot at a different zoom level:

![How to do it...](img/9727OS_03_04.jpg)

These simple yet powerful techniques allow users to manipulate virtual objects in real-time and can be extremely useful when visualizing complex datasets. Additionally, we can rotate the object at different angles by holding the mouse button and dragging the object in various directions. The screenshots below show how we can render the graph at any arbitrary angle to better understand the data distribution.

Here is a screenshot showing the side view of the Gaussian function:

![How to do it...](img/9727OS_03_05.jpg)

Here is a screenshot showing the Gaussian function from the top:

![How to do it...](img/9727OS_03_06.jpg)

Finally, here is a screenshot showing the Gaussian function from the bottom:

![How to do it...](img/9727OS_03_07.jpg)

## How it works...

This sample code illustrates the basic interface needed to build interactive applications that are highly portable across multiple platforms using OpenGL and the GLFW library. The use of `callback` functions in the GLFW library allows non-blocking calls that run in parallel with the rendering engine. This concept is particularly useful since input devices such as the mouse, keyboard, and joysticks all have different input rates and latency. These `callback` functions allow for asynchronous execution without blocking the main rendering loop.

The `glfwSetKeyCallback`, `glfwSetFramebufferSizeCallback`, `glfwSetScrollCallback`, `glfwSetMouseBcuttonCallback`, and `glfwSetCursorPosCallback` functions provide controls over the mouse buttons and scrolling wheel, keyboard inputs, and window resizing events. These are only some of the many handlers we can implement with the GLFW library support. For example, we can further extend the error handling capabilities by adding additional `callback` functions. Also, we can handle window closing and opening events, thereby enabling even more sophisticated interfaces with multiple windows. With the examples provided thus far, we have introduced the basics of how to create interactive interfaces with relatively simple API calls.

## See also

For complete coverage of GLFW library function calls, this website provides a comprehensive set of examples and documentation for all callback functions as well as the handling of inputs and other events: [http://www.glfw.org/docs/latest/](http://www.glfw.org/docs/latest/).

# Rendering a volumetric dataset – MCML simulation

In this section, we will demonstrate the rendering of a 3D volumetric dataset generated from a Monte Carlo simulation of light transport in biological tissue, called **Monte Carlo for multi-layered media** (**MCML**). For simplicity, the simulation output file is included with the code bundle for this chapter so that readers can directly run the demo without setting up the simulation code. The source code for the Monte Carlo simulation is described in detail in a series of publications listed in the *See also* section and the GPU implementation is available online for interested readers ([https://code.google.com/p/gpumcml/](https://code.google.com/p/gpumcml/)).

Light transport in biological tissue can be modeled with the **radiative transport equation** (**RTE**), which has proven difficult to solve analytically for complex geometry. The time-dependent RTE can be expressed as:

![Rendering a volumetric dataset – MCML simulation](img/9727OS_03_17.jpg)

Here ![Rendering a volumetric dataset – MCML simulation](img/9727OS_03_18.jpg) is the radiance [*W m^(−2)sr^(−1)*] defined as the radiant power [*W*] crossing an infinitesimal area at location *r* perpendicular to the direction *Ω* per unit solid angle, *μ[s]* is the scattering coefficient, *μ[a]* is the absorption coefficient, *ν* is the speed of light, and ![Rendering a volumetric dataset – MCML simulation](img/9727OS_03_19.jpg) is the source term. To solve the RTE numerically, Wilson and Adam introduced the **Monte Carlo** (**MC**) method, which is widely accepted as a gold-standard approach for photon migration modeling due to its accuracy and versatility (especially for complex tissue geometry).

The MC method is a statistical sampling technique that has been applied to a number of important problems in many different fields, ranging from radiation therapy planning in medicine to option pricing in finance. The name Monte Carlo is derived from the resort city in Monaco that is known for its casinos, among other attractions. As its name implies, the key feature of the MC method involves the exploitation of random chance (through the generation of random numbers with a particular probability distribution) to model the physical process in question.

In our case, we are interested in modeling photon propagation in biological tissue. The MCML algorithm provides an MC model of steady-state light transport in multi-layered media. In particular, we will simulate photon propagation in a homogeneous medium with a circular light source incident on the tissue surface in order to compute the light dose (absorbed energy) distribution. Such computations have a wide range of applications, including treatment planning for light therapies such as photodynamic therapy (this can be considered a light-activated chemotherapy for cancer).

Here, we demonstrate how to integrate our code base for displaying volumetric data with OpenGL rendering functions. We will take advantage of techniques such as alpha blending, perspective rendering, and heat map rendering. Together with the GLFW interface for capturing user inputs, we can create an interactive visualizer that can display a large volumetric dataset in real-time and control a slicer that magnifies a plane of data points within the volumetric dataset using a few simple key inputs.

## Getting ready

The simulation result is stored in an ASCII text file that contains a 3D matrix. Each value in the matrix represents the absorbed photon energy density at some fixed position within the voxelized geometry. Here, we will provide a simple parser that extracts the simulation output matrix from the file and stores it in the local memory.

## How to do it...

Let's get started by implementing the MCML data parser, the jet color scheme heat map generator, as well as the slicer in OpenGL:

1.  Take the data from the simulation output text file and store it in floating-point arrays:

    [PRE34]

2.  Encode the simulation output values using a custom color map for display:

    [PRE35]

3.  Implement the heat map generator with the jet color scheme:

    [PRE36]

4.  Draw all data points on screen with transparency enabled:

    [PRE37]

5.  Draw three slices of data points for cross-sectional visualization:

    [PRE38]

6.  In addition, we need to update the `key_callback` function for moving the slices:

    [PRE39]

7.  Finally, to complete the demo, simply call the `drawMCMLPoints` and `drawMCMLSlices` functions inside the `main` loop using the same code structure for perspective rendering introduced in the previous demo for plotting a Gaussian function:

    [PRE40]

The simulation results, representing the photon absorption distribution in a voxelized geometry, are displayed in 3D in the following screenshot. The light source illuminates the tissue surface (*z=0* at the bottom) and propagates through the tissue (positive *z* direction) that is modeled as an infinitely wide homogeneous medium. The photon absorption distribution follows the expected shape for a finite-sized, flat, and circular beam:

![How to do it...](img/9727OS_03_08.jpg)

## How it works...

This demo illustrates how we can take a volumetric dataset generated from a Monte Carlo simulation (and, more generally, a volumetric dataset from any application) and render it with a highly interactive interface using OpenGL. The data parser takes an ASCII text file as input. Then, we turn the floating-point data into individual vertices that can fit into our rendering pipeline. Upon initialization, the variables `mcml_vertices` and `mcml_data` store the pre-computed heat map data as well as the position of each data point. The `parser` function also computes the maximum and minimum value in the dataset for heat map visualization. The `getHeatMapColor` function takes the simulation output value and maps it to a color in the jet color scheme. The algorithm basically defines a color spectrum and we remap the value based on its range.

In the following screenshot, we show a top view of the simulation result, which allows us to visualize the symmetry of the light distribution:

![How it works...](img/9727OS_03_09.jpg)

The `drawMCMLSlices` function takes a slice (that is, a plane) of data and renders the data points at the full opacity and a larger point size. This provides a useful and very common visualization method (especially in medical imaging) that allows users to examine the volumetric data in detail by moving the cross-sectional slices. As we can see in the following screenshot, we can shift the slicer in the *x*, *y*, and *z* directions to visualize the desired regions of interest:

![How it works...](img/9727OS_03_10.jpg)

## There's more...

This demo provides an example of real-time volumetric data visualization for rendering simulation data in an interactive 3D environment. The current implementation can be easily modified for a wide range of applications that require volumetric dataset visualization. Our approach provides an intuitive way to render complex 3D datasets with a heat map generator and a slicer as well as 3D perspective rendering techniques using OpenGL.

One important observation is that this demo required a significant number of `glVertex3f` calls, which can become a major performance bottleneck. To address this, in the upcoming chapters, we will explore more sophisticated ways to handle memory transfer and draw even more complex models with **Vertex Buffer Objects** (**VBOs**), a memory buffer in your graphics card designed to store information about vertices. This will lead us towards fragment programs and custom vertex shader programs (that is, moving from OpenGL 2.0 to OpenGL 3.2 or higher). However, the simplicity of using classical OpenGL 2.0 calls is an important consideration if we are aiming for a short development cycle, minimal overhead, and backward compatibility with older hardware.

## See also

For further information, please consult the following references:

*   E. Alerstam & W. C. Y. Lo, T. Han, J. Rose, S. Andersson-Engels, and L. Lilge, "Next-generation acceleration and code optimization for light transport in turbid media using GPUs," *Biomed. Opt. Express 1*, 658-675 (2010).
*   W. C. Y. Lo, K. Redmond, J. Luu, P. Chow, J. Rose, and L. Lilge, "Hardware acceleration of a Monte Carlo simulation for photodynamic therapy treatment planning," *J. Biomed. Opt. 14*, 014019 (2009).
*   L. Wang, S. Jacques, and L. Zheng, "MCML - Monte Carlo modeling of light transport in multi-layered tissues," *Comput. Meth. Prog. Biol. 47*, 131–146 (1995).
*   B. Wilson and G. Adam, "A Monte Carlo model for the absorption and flux distributions of light in tissue," *Med. Phys. 10*, 824 (1983).