# Chapter 5. Rendering of Point Cloud Data for 3D Range-sensing Cameras

In this chapter, we will cover the following topics:

*   Getting started with the Microsoft Kinect (PrimeSense) 3D range-sensing camera
*   Capturing raw data from depth-sensing cameras
*   OpenGL point cloud rendering with texture mapping and overlays

# Introduction

The purpose of this chapter is to introduce the techniques to visualize another interesting and emerging class of data: depth information from 3D range-sensing cameras. Devices with 3D depth sensors are hitting the market everyday, and companies such as Intel, Microsoft, SoftKinetic, PMD, Structure Sensor, and Meta (wearable Augmented Reality eyeglasses) are all using these novel 3D sensing devices to track user inputs, such as hand gestures for interaction and/or tracking a user's environment. An interesting integration of 3D sensors with OpenGL is the ability to look at a scene in 3D from different perspectives, thereby enabling a virtual 3D fly-through of a scene captured with the depth sensors. In our case, for data visualization, being able to walk through a massive 3D dataset could be particularly powerful in scientific computing, urban planning, and many other applications that involve the visualization of 3D structures of a scene.

In this chapter, we propose a simplified pipeline that takes any 3D point data (*X*, *Y*, *Z*) with color (*r*, *g*, *b*) and renders these point clouds on the screen in real time. The point clouds will be obtained directly from real-world data using a 3D range-sensing camera. We will also provide ways to fly around the point cloud and have dynamic ways to adjust the camera's parameters. This chapter will build on the OpenGL graphics rendering pipeline discussed in the previous chapter, and we will show you a few additional tricks to filter the data with GLSL. We will display our depth information using our heat map generator to see the depth in 2D and remap this data to a 3D point cloud using texture mapping and perspective projection. This will allow us to see the real-life depth-based rendering of a scene and navigate around the scene from any perspective.

# Getting started with the Microsoft Kinect (PrimeSense) 3D range-sensing camera

The Microsoft Kinect 3D range-sensing camera based on the PrimeSense technology is an interesting piece of equipment that enables the estimation of the 3D geometry of a scene through depth-sensing using light patterns. The 3D sensor has an active infrared laser projector, which emits encoded speckle light patterns. The sensors allow users to capture color images and provide a 3D depth map at a resolution of 640 x 480\. Since the Kinect sensor is an active sensor, it is invariant to indoor lighting condition (that is, it even works in the dark) and enables many applications, such as gesture and pose tracking as well as 3D scanning and reconstruction.

In this section, we will demonstrate how to set up this type of range-sensing camera, as an example. While we do not require readers to purchase a 3D range-sensing camera for this chapter (since we will provide the raw data captured on this device for the purpose of running our demos), we will demonstrate how one can set up the device to capture data directly, primarily for those who are interested in further experimenting with real-time 3D data.

## How to do it...

Windows users can download the OpenNI 2 SDK and driver from [http://structure.io/openni](http://structure.io/openni) (or using the direct download link: [http://com.occipital.openni.s3.amazonaws.com/OpenNI-Windows-x64-2.2.0.33.zip](http://com.occipital.openni.s3.amazonaws.com/OpenNI-Windows-x64-2.2.0.33.zip)) and follow the on-screen instructions. Linux users can download the OpenNI 2 SDK from the same website at [http://structure.io/openni](http://structure.io/openni).

Mac users can install the OpenNI2 driver as follows:

1.  Install libraries with Macport:

    [PRE0]

2.  Download OpenNI2 from [https://github.com/occipital/openni2](https://github.com/occipital/openni2).
3.  Compile the source code with the following commands:

    [PRE1]

4.  Run the `SimpleViewer` executable:

    [PRE2]

If you are using a computer with a USB 3.0 interface, it is important that you first upgrade the firmware for the PrimeSense sensor to version 1.0.9 ([http://dasl.mem.drexel.edu/wiki/images/5/51/FWUpdate_RD109-112_5.9.2.zip](http://dasl.mem.drexel.edu/wiki/images/5/51/FWUpdate_RD109-112_5.9.2.zip)). This upgrade requires a Windows platform. Note that the Windows driver for the PrimeSense sensor must be installed ([http://structure.io/openni](http://structure.io/openni)) for you to proceed. Execute the `FWUpdate_RD109-112_5.9.2.exe` file, and the firmware will be automatically upgraded. Further details on the firmware can be found at [http://dasl.mem.drexel.edu/wiki/index.php/4._Updating_Firmware_for_Primesense](http://dasl.mem.drexel.edu/wiki/index.php/4._Updating_Firmware_for_Primesense).

## See also

Detailed technical specifications of the Microsoft Kinect 3D system can be obtained from [http://msdn.microsoft.com/en-us/library/jj131033.aspx](http://msdn.microsoft.com/en-us/library/jj131033.aspx), and further installation instructions and prerequisites to build OpenNI2 drivers can be found at [https://github.com/occipital/openni2](https://github.com/occipital/openni2).

In addition, Microsoft Kinect V2 is also available and is compatible with Windows. The new sensor provides higher resolution images and better depth fidelity. More information about the sensor, as well as the Microsoft Kinect SDK, can be found at [https://www.microsoft.com/en-us/kinectforwindows](https://www.microsoft.com/en-us/kinectforwindows).

# Capturing raw data from depth-sensing cameras

Now that you have installed the prerequisite libraries and drivers, we will demonstrate how to capture raw data from your depth-sensing camera.

## How to do it...

To capture sensor data directly in a binary format, implement the following function:

[PRE3]

Similarly, we also capture the raw RGB color data with the following implementation:

[PRE4]

The preceding code snippet can be added to any sample code within the OpenNI2 SDK that provides depth and color data visualization (to enable raw data capture). We recommend that you modify the `Viewer.cpp` file in the `OpenNI2-master/Samples/SimpleViewer` folder. The modified sample code is included in our code package. To capture raw data, press *R* and the data will be stored in the `depth_frame0.bin` and `color_frame0.bin` files.

## How it works...

The depth sensor returns two streams of data in real time. One data stream is a 3D depth map, which is stored in 16-bits unsigned short data type (see the following figure on the left-hand side). Another data stream is a color image (see the following figure on the right-hand side), which is stored in a 24 bits per pixel, RGB888 format (that is, the memory is aligned in the R, G, and B order, and *8 bits * 3 channels = 24 bits* are used per pixel).

![How it works...](img/9727OS_05_01.jpg)

The binary data is written directly to the hard disk without compression or modification to the data format. On the client side, we read the binary files as if there is a continuous stream of data and color data pairs arriving synchronously from the hardware device. The OpenNI2 driver provides the mechanism to interface with the PrimeSense-based sensors (Microsoft Kinect or PS1080).

The `openni::VideoFrameRef depthFrame` variable, for example, stores the reference to the depth data buffer. By calling the `depthFrame.getData()` function, we obtain a pointer to the buffer in the `DepthPixel` format, which is equivalent to the unsigned short data type. Then, we write the binary data to a file using the `write()` function in the `fstream` library. Similarly, we perform the same task with the color image, but the data is stored in the RGB888 format.

Additionally, we can enable the `setImageRegistrationMode` (`openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR`) depth map registration function in OpenNI2 to automatically compute and map a depth value onto the color image. The depth map is overlaid onto the color image and is shown in the following figure:

![How it works...](img/9727OS_05_02.jpg)

In the next section, we will assume that the raw depth map is precalibrated with image registration by OpenNI2 and can be used to compute the real-world coordinates and UV mapping indices directly.

# OpenGL point cloud rendering with texture mapping and overlays

We will build on the OpenGL framework discussed in the previous chapter for point cloud rendering in this section. The texture mapping technique introduced in the previous chapter can also be applied in the point cloud format. Basically, the depth sensor provides a set of vertices in real-world space (the depth map), and the color camera provides us with the color information of the vertices. UV mapping is a simple lookup table once the depth map and color camera are calibrated.

## Getting ready

Readers should use the raw data provided for the subsequent demo or obtain their own raw data from a 3D range-sensing camera. In either case, we assume these filenames will be used to denote the raw data files: `depth_frame0.bin` and `color_frame0.bin`.

## How to do it...

Similar to the previous chapter, we will divide the program into three major components: the main program (`main.cpp`), shader programs (`shader.cpp`, `shader.hpp`, `pointcloud.vert`, `pointcloud.frag`), and texture-mapping functions (`texture.cpp`, `texture.hpp`). The main program performs the essential tasks to set up the demo, while the shader programs perform the specialized processing. The texture-mapping functions provide a mechanism to load and map the color information onto the vertices. Finally, we modify the `control.cpp` file to provide more refined controls over the **fly-through** experience through various additional keyboard inputs (using the up, down, left, and right arrow keys to zoom in and out in addition to adjusting the rotation angles using the *a*, *s*, *x*, and *z* keys).

First, let's take a look at the shader programs. We will create two vertex and fragment shader programs inside the `pointcloud.vert` and `pointcloud.frag` files that are compiled and loaded by the program at runtime by using the `LoadShaders` function in the `shader.cpp` file.

For the `pointcloud.vert` file, we implement the following:

[PRE5]

For the `pointcloud.frag` file, we implement the following:

[PRE6]

Finally, let's put everything together with the `main.cpp` file:

1.  Include prerequisite libraries and the shader program header files inside the common folder:

    [PRE7]

2.  Create a global variable for the GLFW window:

    [PRE8]

3.  Define the width and height of the input depth dataset as well as other window/camera properties for rendering:

    [PRE9]

4.  Define helper functions to parse the raw depth and color data:

    [PRE10]

5.  Create callback functions to handle key strokes:

    [PRE11]

6.  Start the main program with the initialization of the GLFW library:

    [PRE12]

7.  Set up the GLFW window:

    [PRE13]

8.  Create the GLFW window object and make it current for the calling thread:

    [PRE14]

9.  Initialize the GLEW library and include support for experimental drivers:

    [PRE15]

10.  Set up keyboard callback:

    [PRE16]

11.  Set up the shader programs:

    [PRE17]

12.  Create the vertex (*x*, *y*, *z*) for all depth pixels:

    [PRE18]

13.  Read the raw data using the helper functions defined previously:

    [PRE19]

14.  Load the color information into a texture object:

    [PRE20]

15.  Create a set of vertices in a real-world space based on the depth map and also define the UV mapping for the color mapping:

    [PRE21]

16.  Get the location for various uniform and attribute variables:

    [PRE22]

17.  Generate the vertex array object:

    [PRE23]

18.  Initialize the vertex buffer memory:

    [PRE24]

19.  Create and bind the UV buffer memory:

    [PRE25]

20.  Use our shader program:

    [PRE26]

21.  Bind the texture in Texture Unit 0:

    [PRE27]

22.  Set up attribute buffers for vertices and UV mapping:

    [PRE28]

23.  Run the draw functions and loop:

    [PRE29]

24.  Clean up and exit the program:

    [PRE30]

25.  In `texture.cpp`, we implement the additional image-loading functions based on the previous chapter:

    [PRE31]

26.  In `texture.hpp`, we simply define the function prototypes:

    [PRE32]

27.  In `control.cpp`, we modify the `computeViewProjectionMatrices` function with the following code to support additional translation controls:

    [PRE33]

Now we have created a way to visualize the depth sensor information in a 3D fly-through style; the following figure shows the rendering of the point cloud with a virtual camera at the central position of the frame:

![How to do it...](img/9727OS_05_03.jpg)

By rotating and translating the virtual camera, we can create various representations of the scene from different perspectives. With a bird's eye view or side view of the scene, we can see the contour of the face and hand more apparently from these two angles, respectively:

![How to do it...](img/9727OS_05_04.jpg)

This is the side view of the same scene:

![How to do it...](img/9727OS_05_05.jpg)

By adding an additional condition to the remapping loop, we can render the unknown regions (holes) from the scene where the depth camera fails to reconstruct due to occlusion, field of view limitation, range limitation, and/or surface properties such as reflectance:

[PRE34]

This condition allows us to segment the region and project the regions with depth values of 0 onto a plane that is 0.2 meters away from the virtual camera, as shown in the following figure:

![How to do it...](img/9727OS_05_06.jpg)

## How it works...

In this chapter, we exploited the GLSL pipeline and texture-mapping technique to create an interactive point cloud visualization tool that enables the 3D navigation of a scene captured with a 3D range-sensing camera. The shader program also combines the result with the color image to produce our desired effect. The program reads two binary images: the calibrated depth map image and the RGB color image. The color is loaded into a texture object directly using the new `loadRGBImageToTexture()` function, which converts the data from `GL_RGB` to `GL_RGBA`. Then, the depth map data is converted into point cloud data in real-world coordinates based on the intrinsic value of the cameras as well as the depth value at each pixel, as follows:

![How it works...](img/9727OS_05_07.jpg)

Here, *d* is the depth value in millimeter, *x* and *y* are the positions of the depth value in pixel (projective) space, ![How it works...](img/9727OS_05_11.jpg) and ![How it works...](img/9727OS_05_12.jpg) are the principle axes of the depth camera, ![How it works...](img/9727OS_05_13.jpg) and ![How it works...](img/9727OS_05_14.jpg) are the focal lengths of the camera, and ![How it works...](img/9727OS_05_15.jpg) is the position of the point cloud in the real-world coordinate.

In our example, we do not require fine alignment or registration as our visualizer uses a primitive estimation of the intrinsic parameters:

![How it works...](img/9727OS_05_16.jpg)

These numbers could be estimated with the camera calibration tools in OpenCV. The details of these tools are beyond the scope of this chapter.

For our application, we are provided a set of 3D points (*x*, *y*, *z*) as well as the corresponding color information (*r*, *g*, *b*) to compute the point cloud representation. However, the point visualization does not support dynamic lighting and other more advanced rendering techniques. To address this, we can extend the point cloud further into a mesh (that is, a set of triangles to represent surfaces), which will be discussed in the next chapter.