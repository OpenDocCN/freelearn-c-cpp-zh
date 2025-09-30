# Chapter 8. Interactive Real-time Data Visualization on Mobile Devices

In this chapter, we will cover the following topics:

*   Visualizing real-time data from built-in Inertial Measurement Units (IMUs)
*   Part I – handling multi-touch interface and motion sensor inputs
*   Part II – interactive, real-time data visualization with mobile GPUs

# Introduction

In this chapter, we will demonstrate how to visualize data interactively using built-in motion sensors called **Inertial Measurement Units** (**IMUs**) and the multi-touch interface on mobile devices. We will further explore the use of shader programs to accelerate computationally intensive operations to enable real-time visualization of 3D data with mobile graphics hardware. We will assume familiarity with the basic framework for building an Android-based OpenGL ES 3.0 application introduced in the previous chapter and add significantly more complexity in the implementation in this chapter to achieve interactive, real-time 3D visualization of a Gaussian function using both motion sensors and the multi-touch gesture interface. The final demo is designed to work on any Android-based mobile device with proper sensor hardware support.

Here, we will first introduce how to extract data directly from the IMUs and plot the real-time data stream acquired on an Android device. We will divide the final demo into two parts given its complexity. In part I, we will demonstrate how to handle the multi-touch interface and motion sensor inputs on the Java side. In part II, we will demonstrate how to implement the shader program in OpenGL ES 3.0 and other components of the native code to finish our interactive demo.

# Visualizing real-time data from built-in Inertial Measurement Units (IMUs)

Many modern mobile devices now integrate a plethora of built-in sensors including various motion and position sensors (such as an accelerometer, gyroscope, and magnetometer/digital compass) to enable novel forms of user interaction (such as complex gesture and motion control) as well as other environmental sensors, which can measure environmental conditions (such as an ambient light sensor and proximity sensor) to enable smart wearable applications. The Android Sensor Framework provides a comprehensive interface to access many types of sensors, which can be either hardware-based (physical sensors) or software-based (virtual sensors that derive inputs from hardware sensors). In general, there are three major categories of sensors—motion sensors, position sensors, and environmental sensors.

In this section, we will demonstrate how to utilize the Android Sensor Framework to communicate with the sensors available on your device, register sensor event listeners to monitor changes in the sensors, and acquire raw sensor data for display on your mobile device. To create this demo, we will implement the Java code and native code using the same framework design introduced in the previous chapter. The following block diagram illustrates the core functions and the relationship among the classes that will be implemented in this demo:

![Visualizing real-time data from built-in Inertial Measurement Units (IMUs)](img/9727OS_08_01.jpg)

## Getting ready

This demo requires an Android device with OpenGL ES 3.0 support as well as physical sensor hardware support. Unfortunately, at the moment these functions cannot be simulated with an emulator shipped with the Android SDK. Specifically, an Android mobile device with the following set of sensors, which are now commonly available, would be required to run this demo: an accelerometer, gyroscope, and magnetometer (digital compass).

In addition, we assume that the Android SDK and Android NDK are configured as discussed in [Chapter 7](ch07.html "Chapter 7. An Introduction to Real-time Graphics Rendering on a Mobile Platform using OpenGL ES 3.0"), *An I* *ntroduction to Real-time Graphics Rendering on a Mobile Platform using OpenGL ES 3.0*.

## How to do it…

First, we will create the core Java source files similar to the previous chapter. Since the majority of the code is similar, we will only discuss the new and significant elements that are introduced in the current code. The rest of the code is abbreviated with the "…" notation. Please download the complete source code from the official Packt Publishing website.

In the `GL3JNIActivity.java` file, we first integrate Android Sensor Manager, which allows us to read and parse sensor data. The following steps are required to complete the integration:

1.  Import the classes for the Android Sensor Manager:

    [PRE0]

2.  Add the `SensorEventListener` interface to interact with the sensors:

    [PRE1]

3.  Define the `SensorManager` and the `Sensor` variables to handle the data from the accelerometer, gyroscope, and magnetometer:

    [PRE2]

4.  Initialize the `SensorManager` as well as all other sensor services:

    [PRE3]

5.  Register the callback functions and start listening to these events:

    [PRE4]

6.  Handle the `sensor` events. The `onSensorChanged` and `onAccuracyChanged` functions capture any changes detected and the `SensorEvent` variable holds all the information about the sensor type, time-stamp, accuracy, and so on:

    [PRE5]

Next implement the `GL3JNIView` class, which handles OpenGL rendering, in the `GL3JNIView.java` source file inside the `src/com/android/gl3jni/` directory. Since this implementation is identical to content in the [Chapter 7](ch07.html "Chapter 7. An Introduction to Real-time Graphics Rendering on a Mobile Platform using OpenGL ES 3.0"), *An Introduction to Real-time Graphics Rendering on a Mobile Platform using OpenGL ES 3.0*, we will not discuss it again here.

Finally, integrate all the new features in the `GL3JNILib` class, which handles native library loading and calling, in the `GL3JNILib.java` file inside the `src/com/android/gl3jni` directory:

[PRE6]

Now, on the JNI/C++ side, create a class called `Sensor` for managing the data buffer for each sensor, including the accelerometer, gyroscope, and magnetometer (digital compass). First, create a header file for the `Sensor` class called `Sensor.h`:

[PRE7]

Then, implement the `Sensor` class in the `Sensor.cpp` file with the following steps:

1.  Implement the constructor and destructor for the `Sensor` class. Set the default size of the buffer to `256`:

    [PRE8]

2.  Add the initialization function, which sets all default parameters, and allocate and deallocate memory at runtime:

    [PRE9]

3.  Implement the `createBuffers` function for memory allocation:

    [PRE10]

4.  Implement the `free_all` function for deallocating memory:

    [PRE11]

5.  Create routines for appending data to the data buffer of each sensor:

    [PRE12]

6.  Create routines for returning the pointer to the memory buffer of each sensor:

    [PRE13]

7.  Implement methods for displaying/plotting the data stream properly from each sensor (for example, determining the maximum value of the data stream from each sensor to scale the data properly):

    [PRE14]

Finally, we describe the implementation of the OpenGL ES 3.0 native code to complete the demo application (`main_sensor.cpp`). The code is built upon the structure introduced in the previous chapter, so only new changes and modifications will be described in the following steps:

1.  In the project directory, create a file named `main_sensor.cpp` and store it inside the `jni` directory.
2.  Include all necessary header files, including `Sensor.h` at the beginning of the file:

    [PRE15]

3.  Declare shader program handlers and variables for handling sensor data:

    [PRE16]

4.  Define the shader program code for both the vertex shader and fragment shader to render points and lines:

    [PRE17]

5.  Set up all attribute variables in the `setupGraphics` function. These variables will be used to communicate with the shader programs:

    [PRE18]

6.  Create a function for drawing 2D plots to display real-time sensor data:

    [PRE19]

7.  Set up the rendering function which draws the various 2D time series with the data stream from the sensors:

    [PRE20]

8.  Define the JNI prototypes that connect to the Java side. These calls are the interfaces for communicating between the Java code and C/C++ native code:

    [PRE21]

Finally, we need to compile and install the Android application with the same instructions as outlined in the previous chapter.

The following screenshots show the real-time sensor data stream from the accelerometer, gyroscope, and digital compass (top panel, middle panel, and bottom panel, respectively) on our Android device. Red, green, and blue are used to differentiate the channels from each sensor data stream. For example, the red plot in the top panel represents the acceleration value of the device along the *x* axis (the blue plot for the *y* axis and the green plot for the *z* axis). In the first example, we rotated the phone freely at various orientations and the plots show the corresponding changes in the sensor values. The visualizer also provides an auto-scale function, which automatically computes the maximum values to rescale the plots accordingly:

![How to do it…](img/9727OS_08_02.jpg)

Next, we positioned the phone on a stationary surface and we plotted the values of the sensors. Instead of observing constant values over time, the time series plots show that there are some very small changes (jittering) in the sensor values due to sensor noise. Depending on the application, you will often need to apply filtering techniques to ensure that the user experience is jitter-free. One simple solution is to apply a low-pass filter to smooth out any high-frequency noise. More details on the implementation of such filters can be found at [http://developer.android.com/guide/topics/sensors/sensors_motion.html](http://developer.android.com/guide/topics/sensors/sensors_motion.html).

![How to do it…](img/9727OS_08_03.jpg)

## How it works…

The Android Sensor Framework allows users to access the raw data from various types of sensors on a mobile device. This framework is part of the `android.hardware` package and the sensor package includes a set of classes and interfaces for sensor-specific features.

The `SensorManager` class provides an interface and methods for accessing and listing the available sensors from the device. Some common hardware sensors include the accelerometer, gyroscope, proximity sensor, and the magnetometer (digital compass). These sensors are represented by constant variables (such as `TYPE_ACCELEROMETER` for the accelerometer, `TYPE_MAGNETIC_FIELD` for the magnetometer, and `TYPE_GYROSCOPE` for the gyroscope) and the `getDefaultSensor` function returns an instance of the `Sensor` object based on the type requested.

To enable data streaming, we must register the sensor to the `SensorEventListener` class such that the raw data is reported back to the application upon updates. The `registerListener` function then creates the callback to handle updates to the sensor value or sensor accuracy. The `SensorEvent` variable stores the name of the sensor, the timestamp and accuracy of the event, as well as the raw data.

The raw data stream from each sensor is reported back with the `onSensorChange` function. Since sensor data may be acquired and streamed at a high rate, it is important that we do not block callback function calls or perform any computationally intensive processes within the `onSensorChange` function. In addition, it is a good practice to reduce the data rate of the sensor based on your application requirements. In our case, we set the sensor to run at the optimal rate for gaming purposes by passing the constant preset variable `SENSOR_DELAY_GAME` to the `registerListener` function.

The `GL3JNILib` class then handles all the data passing to the native code using the new functions. For simplicity, we have created separate functions for each sensor type, which makes it easier for the reader to understand the data flow for each sensor.

At this point, we have created the interfaces that redirect data to the native side. However, to plot the sensor data on the screen, we need to create a simple buffering mechanism that stores the data points over some period of time. We have created a custom `Sensor` class in C++ to handle data creation, updates, and processing needed to manage these interactions. The implementation of the class is straightforward, and we preset the buffer size to store 256 data points by default.

On the OpenGL ES side, we create the 2D plot by appending the data stream to our vertex buffer. The scale of the data stream is adjusted dynamically based on the current values to ensure that the values fit on the screen. Notice that we have also performed all data scaling and translation on the vertex shader to reduce any overhead in the CPU computation.

## See also

*   For more information on the Android Sensor Framework, consult the documentation online at [http://developer.android.com/guide/topics/sensors/sensors_overview.html](http://developer.android.com/guide/topics/sensors/sensors_overview.html).

# Part I – handling multi-touch interface and motion sensor inputs

Now that we have introduced the basics of handling sensor inputs, we will develop an interactive, sensor-based data visualization tool. In addition to using motion sensors, we will introduce a multi-touch interface for user interaction. The following is a preview of the final application, integrating all the elements in this chapter:

![Part I – handling multi-touch interface and motion sensor inputs](img/9727OS_08_06.jpg)

In this section, we will focus solely on the Java side of the implementation and the native code will be described in part II. The following class diagram illustrates the various components of the Java code (part I) that provide the basic interface for user interaction on the mobile device and demonstrates how the native code (part II) completes the entire implementation:

![Part I – handling multi-touch interface and motion sensor inputs](img/9727OS_08_04.jpg)

## How to do it…

First, we will create the core Java source files that are essential to an Android application. These files serve as a wrapper for our OpenGL ES 3.0 native code. The code structure is based on the `gl3jni` package described in the previous section. Here we will highlight the major changes made to the code and discuss the interaction of these new components.

In the project directory, modify the `GL3JNIActivity` class in the `GL3JNIActivity.java` file within the `src/com/android/gl3jni` directory. Instead of using the raw sensor data, we will utilize the Android sensor fusion algorithm, which intelligently combines all sensor data to recover the orientation of the device as a rotation vector. The steps to enable this feature are described as follows:

1.  In the `GL3JNIActivity` class, add the new variables for handling the rotation matrix and vector:

    [PRE22]

2.  Initialize the `Sensor` variable with the `TYPE_ROTATION_VECTOR` type, which returns the device orientation as a rotation vector/matrix:

    [PRE23]

3.  Register the Sensor Manager object and set the sensor response rate to `SENSOR_DELAY_GAME`, which is used for gaming or real-time applications:

    [PRE24]

4.  Retrieve the device orientation and save the event data as a rotation matrix. Then convert the rotation matrix into Euler angles that are passed to the native code:

    [PRE25]

Next, modify the `GL3JNIView` class, which handles OpenGL rendering, in the `GL3JNIView.java` file inside the `src/com/android/gl3jni/` directory. To make the application interactive, we also integrate the touch-based gesture detector that handles multi-touch events. Particularly, we add the `ScaleGestureDetector` class that enables the pinch gesture for scaling the 3D plot. To implement this feature, we make the following modifications to the `GL3JNIView.java` file:

1.  Import the `MotionEvent` and `ScaleGestureDetector` classes:

    [PRE26]

2.  Create a `ScaleGestureDetector` variable and initialize with `ScaleListener`:

    [PRE27]

3.  Pass the motion event to the gesture detector when a touch screen event occurs (`onTouchEvent`):

    [PRE28]

4.  Implement `SimpleOnScaleGestureListener` and handle the callback (`onScale`) on pinch gesture events:

    [PRE29]

Finally, in the `GL3JNILib` class, we implement the functions to handle native library loading and calling in the `GL3JNILib.java` file inside the `src/com/android/gl3jni` directory:

[PRE30]

## How it works…

Similar to the previous demo, we will use the Android Sensor Framework to handle the sensor inputs. Notice that, in this demo, we specify `TYPE_ROTATION_VECTOR` for the sensor type inside the `getDefaultSensor` function in `GL3JNIActivity.java`, which allows us to detect the device orientation. This is a software type sensor in which all IMUs data (from the accelerometer, gyroscope, and magnetometer) are fused together to create the rotation vector. The device orientation data is first stored in the rotation matrix `mRotationMatrix` using the `getRotationMatrixFromVector` function and the azimuth, pitch, and roll angles (rotation around the *x*, *y*, and *z* axes, respectively) are retrieved using the `getOrientation` function. Finally, we pass the three orientation angles to the native code portion of the implementation using the `GL3JNILib.addRotData` call. This allows us to control 3D graphics based on the device's orientation.

Next we will explain how the multi-touch interface works. Inside the `GL3JNIView` class, you will notice that we have created an instance (`mScaleDetector`) of a new class called `ScaleGestureDetector`. The `ScaleGestureDetector` class detects scaling transformation gestures (pinching with two fingers) using the `MotionEvent` class from the multi-touch screen. The algorithm returns the scale factor that can be redirected to the OpenGL pipeline to update the graphics in real time. The `SimpleOnScaleGestureListener` class provides a callback function for the `onScale` event and we pass the scale factor (`mScaleFactor`) to the native code using the `GL3JNILib.setScale` call.

## See also

*   For further information on the Android multi-touch interface, see the detailed documentation at [http://developer.android.com/training/gestures/index.html](http://developer.android.com/training/gestures/index.html).

# Part II – interactive, real-time data visualization with mobile GPUs

Now we will complete our demo with the native code implementation to create our highly interactive Android-based data visualization application with OpenGL ES 3.0 as well as the Android sensor and gesture control interface.

The following class diagram highlights what remains to be implemented on the C/C++ side:

![Part II – interactive, real-time data visualization with mobile GPUs](img/9727OS_08_05.jpg)

## How to do it…

Here, we describe the implementation of the OpenGL ES 3.0 native code to complete the demo application. We will preserve the same code structure from [Chapter 7](ch07.html "Chapter 7. An Introduction to Real-time Graphics Rendering on a Mobile Platform using OpenGL ES 3.0"), *An Introduction to Real-time Graphics Rendering on a Mobile Platform using OpenGL ES 3.0*. In the following steps, only the new codes are highlighted, and all changes are implemented in the `main.cpp` file inside the `jni` folder:

1.  Include all necessary header files, including `JNI`, OpenGL ES 3.0, and the `GLM` library:

    [PRE31]

2.  Declare the shader program variables:

    [PRE32]

3.  Declare variables for setting up the camera as well as other relevant variables such as the rotation angles and grid:

    [PRE33]

4.  Define the shader program code for both the vertex shader and fragment shader. Note the similarity in the heat map generation code between this implementation in OpenGL ES 3.0 and an earlier implementation in standard OpenGL (see chapters 4-6):

    [PRE34]

5.  Initialize the grid pattern for data visualization:

    [PRE35]

6.  Set the rotation angles that are used to control the model viewing angles. These angles (device orientation) are passed from the Java side:

    [PRE36]

7.  Compute the projection and view matrices based on camera parameters:

    [PRE37]

8.  Create a function for handling the initialization of all attribute variables for the shader program and other one-time setups, such as the memory allocation and initialization for the grid:

    [PRE38]

9.  Set up the rendering function for the 3D plot of the Gaussian function:

    [PRE39]

10.  Define the JNI prototypes that connect to the Java side. These calls are the interfaces for communicating between the Java code and C/C++ native code:

    [PRE40]

11.  Set up the internal function calls with the helper functions:

    [PRE41]

Finally, in terms of the compilation steps, modify the build files `Android.mk` and `Application.mk` accordingly as follows:

1.  Add in the GLM path to the `LOCAL_C_INCLUDES` variable in `Android.mk`:

    [PRE42]

2.  Add in `gnustl_static` to the `APP_STL` variable to use GNU STL as a static library. This allows for all runtime supports from C++, which is needed by the GLM library. See more at [http://www.kandroid.org/ndk/docs/CPLUSPLUS-SUPPORT.html](http://www.kandroid.org/ndk/docs/CPLUSPLUS-SUPPORT.html):

    [PRE43]

3.  Run the compilation script (this is similar to what we did in the previous chapter). Please note that the `ANDROID_SDK_PATH` and `ANDROID_NDK_PATH` variables should be changed to the correct directories based on the local environment setup:

    [PRE44]

4.  Install the **Android Application Package** (**APK**) on the Android phone, using the following commands in the terminal:

    [PRE45]

The final results of our implementation are shown next. By changing the orientation of the phone, the Gaussian function can be viewed from different angles. This provides a very intuitive way to visualize 3D datasets. Here is a photo showing the Gaussian function when the device is oriented parallel to the ground:

![How to do it…](img/9727OS_08_07.jpg)

Finally, we test our multi-touch gesture interface by pinching on the touch screen with 2 fingers. This provides an intuitive way to zoom into and out of the 3D data. Here is the first photo that shows the close-up view after zooming into the data:

![How to do it…](img/9727OS_08_08.jpg)

Here is another photo that shows what the data looks like when you zoom out by pinching your fingers:

![How to do it…](img/9727OS_08_09.jpg)

Finally, here is a screenshot of the demo application that shows a Gaussian distribution in 3D rendered in real-time with our OpenGL ES 3.0 shader program:

![How to do it…](img/9727OS_08_10.jpg)

## How it works…

In the second part of the demo, we demonstrated the use of a shader program written in OpenGL ES 3.0 to perform all the simulation and heat map-based 3D rendering steps to visualize a Gaussian distribution on a mobile GPU. Importantly, the shader code in OpenGL ES 3.0 is very similar to the code written in standard OpenGL 3.2 and above (see chapters 4 to 6). However, we recommend that you consult the specification to ensure that a particular feature of interest co-exists in both versions. More details on the OpenGL ES 3.0 specifications can be found at [https://www.khronos.org/registry/gles/specs/3.0/es_spec_3.0.0.pdf](https://www.khronos.org/registry/gles/specs/3.0/es_spec_3.0.0.pdf).

The hardware-accelerated portion of the code is programmed within the vertex shader program and stored inside the `g_vshader_code` variable; then the fragment shade program passes the processed color information onto the screen's color buffer. The vertex program handles the computation related to the simulation (in our case, we have a Gaussian function with a time-varying sigma value as demonstrated in [Chapter 3](ch03.html "Chapter 3. Interactive 3D Data Visualization"), *Interactive 3D Data Visualization*) in the graphics hardware. We pass in the sigma value as a uniform variable and it is used to compute the surface height. In addition, we also compute the heat map color value within the shader program based on the height value. With this approach, we have significantly improved the speed of the graphic rendering step by completely eliminating the use of the CPU cycles on these numerous floating point operations.

In addition, we have included the GLM library used in previous chapters into the Android platform by adding the headers as well as the GLM path in the build script `Android.mk`. The GLM library handles the view and projection matrix computation and also allows us to migrate most of our previous work, such as setting up 3D rendering, to the Android platform.

Finally, our Android-based application also utilizes the inputs from the multi-touch screen interface and the device orientation derived from the motion sensor data. These values are passed through the JNI directly to the shader program as uniform variables.