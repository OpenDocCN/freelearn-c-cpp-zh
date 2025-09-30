# Chapter 2. OpenGL Primitives and 2D Data Visualization

In this chapter, we will cover the following topics:

*   OpenGL primitives
*   Creating a 2D plot using primitives
*   Real-time visualization of time series
*   2D visualization of 3D/4D datasets

# Introduction

In the previous chapter, we provided a sample code to render a triangle on the screen using OpenGL and the GLFW library. In this chapter, we will focus on the use of OpenGL primitives, such as points, lines, and triangles, to enable the basic 2D visualization of data, including time series such as an **electrocardiogram** (**ECG**). We will begin with an introduction to each primitive, along with sample code to allow readers to experiment with the OpenGL primitives with a minimal learning curve.

One can think of primitives as the fundamental building blocks to create graphics using OpenGL. These building blocks can be easily reused in many applications and are highly portable among different platforms. Frequently, programmers struggle with displaying their results in a visually appealing manner, and an enormous amount of time may be spent on performing simple drawing tasks on screen. In this chapter, we will introduce a rapid prototyping approach to 2D data visualization using OpenGL so that impressive graphics can be created with minimal efforts. Most importantly, the proposed framework is highly intuitive and reusable, and it can be extended to be used in more sophisticated applications. Once you have mastered the basics of the OpenGL language, you will be equipped with the skills to create impressive applications that harness the true potential of OpenGL for data visualization using modern graphics hardware.

# OpenGL primitives

In the simplest terms, primitives are just basic shapes that are drawn in OpenGL. In this section, we will provide a brief overview of the main geometric primitives that are supported by OpenGL and focus specifically on three commonly used primitives (which will also appear in our demo applications): points, lines, and triangles.

## Drawing points

We begin with a simple, yet very useful, building block for many visualization problems: a point primitive. A point can be in the form of ordered pairs in 2D, or it can be visualized in the 3D space.

## Getting ready

To simplify the workflow and improve the readability of the code, we first define a structure called `Vertex`, which encapsulates the fundamental elements such as the position and color of a vertex.

[PRE0]

Now, we can treat every object and shape in terms of a set of vertices (with a specific color) in space. In this chapter, as our focus is on 2D visualization, the *z* positions of vertices are often manually set to `0.0f`.

We can create a vertex at the center of the screen (0, 0, 0) with a white color as an example:

[PRE1]

Note that the color element consists of the red, green, blue, and alpha channels. These values range from 0.0 to 1.0\. The alpha channel allows us to create transparency (0: fully transparent; 1: fully opaque) so that objects can be blended together.

## How to do it…

We can first define a function called `drawPoint` to encapsulate the complexity of OpenGL primitive functions, illustrated as follows:

1.  Create a function called `drawPoint` to draw points which takes in two parameters (the vertex and size of the point):

    [PRE2]

2.  Specify the size of the point:

    [PRE3]

3.  Set the beginning of the list of vertices to be specified and indicate the primitive type associated with the vertices (`GL_POINTS` in this case):

    [PRE4]

4.  Set the color and the vertex position using the fields from the `Vertex` structure:

    [PRE5]

5.  Set the end of the list:

    [PRE6]

6.  In addition, we can define a function called `drawPointsDemo` to encapsulate the complexity further. This function draws a series of points with an increasing size:

    [PRE7]

Finally, let's integrate these two functions into a complete OpenGL demo program (refer to identical steps in [Chapter 1](ch01.html "Chapter 1. Getting Started with OpenGL"), *Getting Started withOpenGL*):

1.  Create a source file called `main_point.cpp`, and then include the header file for the GLFW library and standard C++ libraries:

    [PRE8]

2.  Define the size of the window for display:

    [PRE9]

3.  Define the `Vertex` structure and function prototypes:

    [PRE10]

4.  Implement the `drawPoint` and `drawPointsDemo` functions, as shown previously.
5.  Initialize GLFW and create a GLFW window object:

    [PRE11]

6.  Enable anti-aliasing and smoothing:

    [PRE12]

7.  Define a loop that terminates when the window is closed. Set up the viewport (using the size of the window) and clear the color buffer at the beginning of each iteration to update with new content:

    [PRE13]

8.  Set up the camera matrix for orthographic projection:

    [PRE14]

9.  Call the `drawPointsDemo` function:

    [PRE15]

10.  Swap the front and back buffers of the window and process the event queue (such as keyboard inputs) to avoid lock-up:

    [PRE16]

11.  Release the memory and terminate the GLFW library. Then, exit the application:

    [PRE17]

Here is the result (with anti-aliasing disabled) showing a series of points with an increasing size (that is, the diameter of each point as specified by `glPointSize`):

![How to do it…](img/9727OS_02_01.jpg)

## How it works…

The `glBegin` and `glEnd` functions delimit the list of vertices corresponding to a desired primitive (`GL_POINTS` in this demo). The `glBegin` function accepts a set of symbolic constants that represent different drawing methods, including `GL_POINTS`, `GL_LINES`, and `GL_TRIANGLES`, as discussed in this chapter.

There are several ways to control the process of drawing points. First, we can set the diameter of each point (in pixels) with the `glPointSize` function. By default, a point has a diameter of 1 without anti-aliasing (a method to smooth sampling artifacts) enabled. Also, we can define the color of each point as well as the alpha channel (transparency) using the `glColor4f` function. The alpha channel allows us to overlay points and blend graphics elements. This is a powerful, yet very simple, technique used in graphics design and user interface design. Lastly, we define the position of the point in space with the `glVertex3f` function.

In OpenGL, we can define the projection transformation in two different ways: orthographic projection or perspective projection. In 2D drawing, we often use orthographic projection which involves no perspective correction (for example, the object on screen will remain the same size regardless of its distance from the camera). In 3D drawing, we use perspective projection to create more realistic-looking scenes similar to how the human eye sees. In the code, we set up an orthographic projection with the `glOrtho` function. The `glOrtho` function takes these parameters: the coordinates of the vertical clipping plane, the coordinates of the horizontal clipping plane, and the distance of the nearer and farther depth clipping planes. These parameters determine the projection matrix, and the detailed documentation can be found in [https://developer.apple.com/library/mac/documentation/Darwin/Reference/ManPages/man3/glOrtho.3.html](https://developer.apple.com/library/mac/documentation/Darwin/Reference/ManPages/man3/glOrtho.3.html).

Anti-aliasing and smoothing are necessary to produce the polished look seen in modern graphics. Most graphics cards support native smoothing and in OpenGL, it can be enabled as follows:

[PRE18]

Here is the final result with anti-aliasing enabled, showing a series of circular points with an increasing size:

![How it works…](img/9727OS_02_02.jpg)

Note that in the preceding diagram, the points are now rendered as circles instead of squares with the anti-aliasing feature enabled. Readers are encouraged to disable and enable the features of the preceding diagram to see the effects of the operation.

## See also

In this tutorial, we have focused on the C programming style due to its simplicity. In the upcoming chapters, we will migrate to an object-oriented programming style using C++. In addition, in this chapter, we focus on three basic primitives (and discuss the derivatives of these primitives where appropriate): `GL_POINTS`, `GL_LINES`, and `GL_TRIANGLES`. Here is a more extensive list of primitives supported by OpenGL (refer to [https://www.opengl.org/wiki/Primitive](https://www.opengl.org/wiki/Primitive) for more information):

[PRE19]

## Drawing line segments

One natural extension now is to connect a line between data points and then to connect the lines together to form a grid for plotting. In fact, OpenGL natively supports drawing line segments, and the process is very similar to that of drawing a point.

## Getting ready

In OpenGL, we can simply define a line segment with a set of 2 vertices, and a line will be automatically formed between them by choosing `GL_LINES` as the symbolic constant in the `glBegin` statement.

## How to do it…

Here, we define a new line drawing function called `drawLineSegment` which users can test by simply replacing the `drawPointsDemo` function in the previous section:

1.  Define the `drawLineSegment` function which takes in two vertices and the width of the line as inputs:

    [PRE20]

2.  Set the width of the line:

    [PRE21]

3.  Set the primitive type for line drawing:

    [PRE22]

4.  Set the vertices and the color of the line:

    [PRE23]

In addition, we define a new grid drawing function called `drawGrid`, built on top of the `drawLineSegment` function as follows:

[PRE24]

Finally, we can execute the full demo by replacing the call for the `drawPointsDemo` function in the previous section with the following `drawLineDemo` function:

[PRE25]

Here is a screenshot of the demo showing a grid with equal spacing and the *x* and *y* axes drawn with the line primitives:

![How to do it…](img/9727OS_02_03.jpg)

## How it works…

There are multiple ways of drawing line segments in OpenGL. We have demonstrated the use of `GL_LINES` which takes every consecutive pair of vertices in the list to form an independent line segment for each pair. On the other hand, if you would like to draw a line without gaps, you can use the `GL_LINE_STRIP` option, which connects all the vertices in a consecutive fashion. Finally, to form a closed loop sequence in which the endpoints of the lines are connected, you would use the `GL_LINE_LOOP` option.

In addition, we can modify the width and the color of a line with the `glLineWidth` and `glColor4f` functions for each vertex, respectively.

## Drawing triangles

We will now move on to another very commonly used primitive, namely a triangle, which forms the basis for drawing all possible polygons.

## Getting ready

Similar to drawing a line segment, we can simply define a triangle with a set of 3 vertices, and line segments will be automatically formed by choosing `GL_TRIANGLES` as the symbolic constant in the `glBegin` statement.

## How to do it…

Finally, we define a new function called `drawTriangle`, which users can test by simply replacing the `drawPointsDemo` function. We will also reuse the `drawGrid` function from the previous section:

1.  Define the `drawTriangle` function, which takes in three vertices as the input:

    [PRE26]

2.  Set the primitive type to draw triangles:

    [PRE27]

3.  Set the vertices and the color of the triangle:

    [PRE28]

4.  Execute the demo by replacing the call for the `drawPointsDemo` function in the full demo code with the following `drawTriangleDemo` function:

    [PRE29]

Here is the final result with a triangle rendered with 60 percent transparency overlaid on top of the grid lines:

![How to do it…](img/9727OS_02_04.jpg)

## How it works…

While the process of drawing a triangle in OpenGL appears similar to previous examples, there are some subtle differences and further complexities that can be incorporated. There are three different modes in this primitive (`GL_TRIANGLES`, `GL_TRIANGLE_STRIP`, and `GL_TRIANGLE_FAN`), and each handles the vertices in a different manner. First, `GL_TRIANGLES` takes three vertices from a given list to create a triangle. The triangles are independently formed from each triplet of the vertices (that is, every three vertices are turned into a different triangle). On the other hand, `GL_TRIANGLE_STRIP` forms a triangle with the first three vertices, and each subsequent vertex forms a new triangle using the previous two vertices. Lastly, `GL_TRIANGLE_FAN` creates an arbitrarily complex convex polygon by creating triangles that have a common vertex in the center specified by the first vertex v_1, which forms a fan-shaped structure consisting of triangles. In other words, triangles will be generated in the grouping order specified as follows:

[PRE30]

Although a different color is set for each vertex, OpenGL handles color transition (linear interpolation) automatically, as shown in the triangle drawing in the previous example. The vertices are set to red, green, and blue, but the spectrum of colors can be clearly seen. Additionally, transparency can be set using the alpha channel, which enables us to clearly see the grid behind the triangle. With OpenGL, we can also add other elements, such as the advanced handling of color and shading, which will be discussed in the upcoming chapters.

# Creating a 2D plot using primitives

Creating a 2D plot is a common way of visualizing trends in datasets in many applications. With OpenGL, we can render such plots in a much more dynamic way compared to conventional approaches (such as basic MATLAB plots) as we can gain full control over the graphics shader for color manipulation and we can also provide real-time feedback to the system. These unique features allow users to create highly interactive systems, so that, for example, time series such as an electrocardiogram can be visualized with minimal effort.

Here, we first demonstrate the visualization of a simple 2D dataset, namely a sinusoidal function in discrete time.

## Getting ready

This demo requires a number of functions (including the `drawPoint`, `drawLineSegment`, and `drawGrid` functions) implemented earlier. In addition, we will reuse the code structure introduced in the [Chapter 1](ch01.html "Chapter 1. Getting Started with OpenGL"), *Getting Started with OpenGL* to execute the demo.

## How to do it…

We begin by generating a simulated data stream for a sinusoidal function over a time interval. In fact, the data stream can be any arbitrary signal or relationship:

1.  Let's define an additional structure called `Data` to simplify the interface:

    [PRE31]

2.  Define a generic 2D data plotting function called `draw2DscatterPlot` with the input data stream and number of points as the input:

    [PRE32]

3.  Draw the *x* and *y* axes using the `drawLineSegment` function described earlier:

    [PRE33]

4.  Draw the data points one by one with the `drawPoint` function:

    [PRE34]

5.  Create a similar function called `draw2DlineSegments` to connect the dots together with the line segments so that both the curve and the data points can be shown simultaneously:

    [PRE35]

6.  Integrate everything into a full demo by creating the grid, generating the simulated data points using a cosine function and plotting the data points:

    [PRE36]

7.  Finally, in the main program, include the `math.h` header file for the cosine function and add a new variable called `phase_shift` outside the loop to execute this demo. You can download the code package from Packt Publishing website for the complete demo code:

    [PRE37]

The final result simulating a real-time input data stream with a sinusoidal shape is plotted on top of grid lines using a combination of basic primitives discussed in previous sections.

![How to do it…](img/9727OS_02_05.jpg)

## How it works…

Using the simple toolkit we created earlier using basic OpenGL primitives, we plotted a sinusoidal function with the data points (sampled at a constant time interval) overlaid on top of the curve. The smooth curve consists of many individual line segments drawn using the `draw2DlineSegments` function, while the samples were plotted using the `drawPoint` function. This intuitive interface serves as the basis for the visualization of more interesting time series for real-world applications in the next section.

# Real-time visualization of time series

In this section, we further demonstrate the versatility of our framework to plot general time series data for biomedical applications. In particular, we will display an ECG in real time. As a brief introduction, an ECG is a very commonly used diagnostic and monitoring tool to detect abnormalities in the heart. ECG surface recording essentially probes the electrical activities of the heart. For example, the biggest spike (called a QRS complex) typically corresponds to the depolarization of the ventricles of the heart (the highly muscular chambers of the heart that pump blood). A careful analysis of the ECGcan be a very powerful, noninvasive method for distinguishing many heart diseases clinically, including many forms of arrhythmia and heart attacks.

## Getting ready

We begin by importing a computer-generated ECG data stream. The ECG data stream is stored in `data_ecg.h` (only a small portion of the data stream is provided here):

[PRE38]

## How to do it…

1.  Use the following code to plot the ECG data by drawing line segments:

    [PRE39]

2.  Display multiple ECG data streams (simulating recording from different leads):

    [PRE40]

3.  Finally, in the main program, include the `data_ecg.h` header file and add the following lines of code to the loop. You can download the code package from the Packt Publishing website for the complete demo code:

    [PRE41]

Here are two snapshots of the real-time display across multiple ECG leads simulated at two different time points. If you execute the demo, you will see the ECG recording from multiple leads move across the screen as the data stream is processed for display.

![How to do it…](img/9727OS_02_06.jpg)

Here is the second snapshot at a later time point:

![How to do it…](img/9727OS_02_07.jpg)

## How it works…

This demo shows the use of the `GL_LINE_STRIP` option, described previously, to plot an ECG time series. Instead of drawing individual and independent line segments (using the `GL_LINE` option), we draw a continuous stream of data by calling the `glVertex3f` function for each data point. Additionally, the time series animates through the screen and provides dynamic updates on an interactive frame with minimal impact on the CPU computation cycles.

# 2D visualization of 3D/4D datasets

We have now learned multiple methods to generate plots on screen using points and lines. In the last section, we will demonstrate how to visualize a million data points in a 3D dataset using OpenGL in real time. A common strategy to visualize a complex 3D dataset is to encode the third dimension (for example, the *z* dimension) in the form of a heat map with a desirable color scheme. As an example, we show a heat map of a 2D Gaussian function with its height *z*, encoded using a simple color scheme. In general, a 2-D Gaussian function, ![2D visualization of 3D/4D datasets](img/9727OS_02_10.jpg), is defined as follows:

![2D visualization of 3D/4D datasets](img/9727OS_02_09.jpg)

Here, *A* is the amplitude (![2D visualization of 3D/4D datasets](img/9727OS_02_9a.jpg)) of the distribution centered at ![2D visualization of 3D/4D datasets](img/9727OS_02_11.jpg) and ![2D visualization of 3D/4D datasets](img/9727OS_02_12.jpg) are the standard deviations (spread) of the distribution in the *x* and *y* directions. To make this demo more interesting and more visually appealing, we vary the standard deviation or sigma term (equally in the *x* and *y* directions) over time. Indeed, we can apply the same method to visualize very complex 3D datasets.

## Getting ready

By now, you should be very familiar with the basic primitives described in previous sections. Here, we employ the `GL_POINTS` option to generate a dense grid of data points with different colors encoding the *z* dimension.

## How to do it…

1.  Generate a million data points (1,000 x 1,000 grid) with a 2-D Gaussian function:

    [PRE42]

2.  Draw the data points using a heat map function for color visualization:

    [PRE43]

3.  Finally, in the main program, include the `math.h` header file and add the following lines of code to the loop to vary the sigma term over time. You can download the example code from the Packt Publishing website for the complete demo code:

    [PRE44]

Here are four figures illustrating the effect of varying the sigma term of the 2-D Gaussian function over time (from 0.01 to 1):

![How to do it…](img/9727OS_02_08.jpg)

## How it works…

We have demonstrated how to visualize a Gaussian function using a simple heat map in which the maximum value is represented by red, while the minimum value is represented by blue. In total, a million data points (1,000 x 1,000) were plotted using vertices for each Gaussian function with a specific sigma term. This sigma term was varied from 0.01 to 1 to show a time-varying Gaussian distribution. To reduce the overhead, vertex buffers can be implemented in the future (we can perform the memory copy operation all at once and remove the `glVertex3f` calls). Similar techniques can be applied to the color channel as well.

## There's more…

The heat map we have described here provides a powerful visualization tool for complex 3D datasets seen in many scientific and biomedical applications. Indeed, we have actually extended our demo to the visualization of a 4D dataset, to be precise, since a time-varying 3D function; with the height encoded using a color map was displayed. This demo shows the many possibilities for displaying data in an interesting, dynamic way using just 2D techniques based on OpenGL primitives. In the next chapter, we will demonstrate the potential of OpenGL further by incorporating 3D rendering and adding user inputs to enable the 3D, interactive visualization of more complex datasets.