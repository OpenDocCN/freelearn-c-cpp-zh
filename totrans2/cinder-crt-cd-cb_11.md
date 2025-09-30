# Chapter 11. Sensing and Tracking Input from the Camera

In this chapter, we will learn how to receive and process data from input devices such as a camera or a Microsoft Kinect sensor.

The following recipes will be covered:

*   Capturing from the camera
*   Tracking an object based on color
*   Tracking motion using optical flow
*   Object tracking
*   Reading QR code
*   Building UI navigation and gesture recognition with Kinect
*   Building an augmented reality with Kinect

# Capturing from the camera

In this recipe we will learn how to capture and display frames from a camera.

## Getting ready

Include the necessary files to capture images from a camera and draw them to OpenGL textures:

[PRE0]

Also add the following `using` statements:

[PRE1]

## How to do it…

We will now capture and draw frames from the camera.

1.  Declare the following members in your application class:

    [PRE2]

2.  In the `setup` method we will initialize `mCamera`:

    [PRE3]

3.  In the `update` method, we will check if `mCamera` was successfully initialized. Also if there is any new frame available, copy the camera's image into `mTexture`:

    [PRE4]

4.  In the `draw` method, we will simply clear the background, check if `mTexture` has been initialized, and draw it's image on the screen:

    [PRE5]

## How it works…

The `ci::Capture` is a class that wraps around Quicktime on Apple computers, AVFoundation on iOS platforms, and Directshow on Windows. Under the hood it uses these lower level frameworks to access and capture frames from a webcam.

Whenever a new frame is found, it's pixels are copied into the `ci::Surface method`. In the previous code we check on every `update` method if there is a new frame by calling the `ci::Capture::checkNewFrame` method, and update our texture with its surface.

## There's more…

It is also possible to get a list of available capture devices and choose which one you wish to start with.

To ask for a list of devices and print their information, we could write the following code:

[PRE6]

To initialize `mCapture` using a specific device, you simply pass `ci::Capture::DeviceRef` as a third parameter in the constructor.

For example, if you wanted to initialize `mCapture` with the first device, you should write the following code:

[PRE7]

# Tracking an object based on color

In this recipe we will show how to track objects with a specified color using the OpenCV library.

## Getting ready

In this recipe we will use OpenCV, so please refer to the *Integrating with OpenCV* recipe from [Chapter 3](ch03.html "Chapter 3. Using Image Processing Techniques"), *Using Image Processing Techniques*. We will also need InterfaceGl which is covered in the *Setting up a GUI for parameter tweaking* recipe from [Chapter 2](ch02.html "Chapter 2. Preparing for Development"), *Preparing for Development*.

## How to do it…

We will create an application that tracks an object with a selected color.

1.  Include the necessary header files:

    [PRE8]

2.  Add members to store the original and processed frame:

    [PRE9]

3.  Add members to store the tracked object's coordinates:

    [PRE10]

4.  Add members to store the parameters that will be passed to the tracking algorithms:

    [PRE11]

5.  Add members to handle the capturing device and frame texture:

    [PRE12]

6.  In the `setup` method we will set the window dimensions and initialize capturing device:

    [PRE13]

7.  In the `setup` method we have to initialize variables and setup the GUI for a preview of the tracked color value:

    [PRE14]

8.  In the `update` method, check if there is any new frame to process and convert it to `cv::Mat`, which is necessary for further OpenCV operations:

    [PRE15]

9.  Process the captured frame:

    [PRE16]

10.  Close the `if` statement's body.

    [PRE17]

11.  Implement the method `setTrackingHSV`, which sets color's values for tracking:

    [PRE18]

12.  Implement the `mouseDown` event handler:

    [PRE19]

13.  Implement the `draw` method as follows:

    [PRE20]

## How it works…

By preparing the captured frame for processing we are converting it into a **hue, saturation, and value** (**HSV**) color space description method, which will be very useful in this case. Those are the properties describing the color in the HSV color space in a more intuitive way for color tracking. We can set a fixed hue value for detection, while saturation and value can vary with in a specified range. This can eliminate a noise caused by constantly changing light in the camera view. Take a look at the first step of the frame image processing; we are using the `cv::inRange` function to get a mask of pixels that fits our tracking color range. The range of the tracking colors is calculated from the color value picked by clicking inside the window, which is implemented inside the `mouseDown` handler and the `setTrackingHSV` method.

As you can see inside `setTrackingHSV`, we are calculating `mColorMin` and `mColorMax` by simply widening the range. You may have to adjust these calculations depending on your camera noise and lighting conditions.

## See also

*   HSV on Wikipedia: [http://en.wikipedia.org/wiki/HSL_and_HSV](http://en.wikipedia.org/wiki/HSL_and_HSV)
*   The OpenCV documentation: [http://opencv.willowgarage.com/documentation/cpp/](http://opencv.willowgarage.com/documentation/cpp/)

# Tracking motion using optical flow

In this recipe we will learn how to track motion in the images produced from a webcam using OpenCV using the popular Lucas Kanade optical flow algorithm.

## Getting ready

We will need to use OpenCV in this recipe, so please refer to the *Integrating with OpenCV* recipe from [Chapter 3](ch03.html "Chapter 3. Using Image Processing Techniques"), *Using Image Processing Techniques* and add OpenCV and it's CinderBlock to your project. Include the following files to your source file:

[PRE21]

Add the following `using` statements:

[PRE22]

## How to do it…

We will read frames from the camera and track motion.

1.  Declare the `ci::gl::Texture` and `ci::Capture` objects to display and capture from a camera. Also, declare a `cv::Mat` object as the previous frame, two `std::vector<cv::Point2f>` objects to store the current and previous features, and a `std::vector<uint8_t>` object to store the status of each feature:

    [PRE23]

2.  In the `setup` method we will initialize `mCamera`:

    [PRE24]

3.  In the `update` method we need to check if `mCamera` has been correctly initialized and whether it has a new frame available:

    [PRE25]

4.  After those `if` statements we will get a reference to `ci::Surface` of `mCamera` and then copy it to our `mTexture` for drawing:

    [PRE26]

5.  Now let's create a `cv::Mat` with the current camera frame. We will also check if `mPreviousFrame` contains any initialized data, calculate the good features to track, and calculate their motion from the previous camera frame to the current frame:

    [PRE27]

6.  Now we just need to copy the frame to `mPreviousFrame` and close the initial `if` statements:

    [PRE28]

7.  In the `draw` method we will begin by clearing the background with black and drawing `mTexture`:

    [PRE29]

8.  Next, we will draw red lines on the features we have tracked, using `mFeatureStatus` to draw the features that have been matched:

    [PRE30]

9.  Finally, we will draw a line between the previous features and the current ones, also using `mFeatureStatus` to draw one of the features that has been matched:

    [PRE31]

    In the following image, the red dots represent good features to track:

    ![How to do it…](img/8703OS_11_01.jpg)

## How it works…

The optical flow algorithm will make an estimation of how much the tracked point has moved from one frame to the other.

## There's more…

In this recipe we are using the `cv::goodFeaturesToTrack` object to calculate which features are optimal for tracking, but it is also possible to manually choose which points we wish to track. All we have to do is populate `mFeatures` manually with whatever points we wish to track and pass it to the `cv::calcOpticalFlowPyrLK`. object

# Object tracking

In this recipe, we will learn how to track specific planar objects in our webcam using OpenCV and it's corresponding CinderBlock.

## Getting ready

You will need an image depiction of the physical object you wish to track in the camera. For this recipe place that image in the `assets` folder and name it `object.jpg`.

We will use the OpenCV CinderBlock in this recipe, so please refer to the *Integrating with OpenCV* recipe from [Chapter 3](ch03.html "Chapter 3. Using Image Processing Techniques"), *Using Image Processing Techniques* and add OpenCV and it's CinderBlock to your project.

If you are using a Mac, you will need to compile the OpenCV static libraries yourself, because the OpenCV CinderBlock is missing some needed libraries on OSX (it will work fine on Windows). You can download the correct version from the following link: [http://sourceforge.net/projects/opencvlibrary/files/opencv-unix/2.3/](http://sourceforge.net/projects/opencvlibrary/files/opencv-unix/2.3/).

You will need to compile the static libraries yourself using the provided `CMake` files. Once your libraries are correctly added to your project, include the following files:

[PRE32]

Add the following `using` statements:

[PRE33]

## How to do it…

We will track an object in the camera frames based on an image depicting the object

1.  Let's begin by creating a `struct` method to store the necessary objects for feature tracking and matching. Add the following code before your application class declaration:

    [PRE34]

2.  In your class declaration add the following member objects:

    [PRE35]

3.  In the `setup` method let's start by initializing the camera:

    [PRE36]

4.  Lets resize `mCorners`, load our object image, and calculate its `image`, `keyPoints`, `texture`, and `descriptor`:

    [PRE37]

5.  In the `update` method, we will check if `mCamera` has been initialized and whether we have a new frame to process:

    [PRE38]

6.  Now let's get the surface of `mCamera` and initialize `texture` and `image` objects of `mCameraInfo`. We will create a `ci::Channel` object from `cameraSurface` that converts color surfaces to gray channel surfaces:

    [PRE39]

7.  Let's calculate `features` and `descriptor` values of `mCameraInfo`:

    [PRE40]

8.  Now let's use `mMatcher` to calculate the matches between `mObjectInfo` and `mCameraInfo`:

    [PRE41]

9.  To perform a test to check for false matches, we will calculate the minimum distance between matches:

    [PRE42]

10.  Now we will add all the points whose distance is less than `minDist*3.0` to `mObjectInfo.goodPoints.clear();`

    [PRE43]

11.  `}` With all our points calculated and matched, we need to calculate the homography between the points of `mObjectInfo` and `mCameraInfo`:

    [PRE44]

12.  Let's create `vector<cv::Point2f>` with the corners of our object and perform a perspective transform to calculate the corners of our object in the camera image:

    ### Tip

    Don't forget to close the brackets we opened earlier.

    [PRE45]

13.  Let's move to the `draw` method and begin by clearing the background and drawing the camera and object textures:

    [PRE46]

14.  Now let's iterate over `goodPoints` values in both `mObjectInfo` and `mCameraInfo` and draw them:

    [PRE47]

15.  Now let's iterate over `mCorners` and draw the corners of the found object:

    [PRE48]

16.  Build and run the application. Grab the physical object you depicted in the `object.jpg` image and put it in front of the image. The program will try to track that object in the camera image and draw it's corners in the image.

## How it works…

We are using a **Speeded Up Robust Features** (**SURF**) feature detector and descriptor to identify features. In the step 4, we are calculating the features and descriptor. We use a `cv::SurfFeatureDetect` object `or` that calculates good features to track on our object. The `cv::SurfDescriptorExtractor` object then uses these features to create a description of our object. In the step 7, we do the same for the camera image.

In the step 8, we then use a **Fast Library for Approximate Nearest Neighbor** (**FLANN**) called `cv::FlannBasedMatcher`. This matcher takes the description from both the camera frame and our object, and calculates matches between them.

In steps 9 and 10, we use the minimum distance between matches to eliminate the possible false matches. The result is passed into `mObjectInfo.goodPoints` and `mCameraInfo.goodPoints`.

In the step 11, we calculate the homography between image and camera. A homography is a projection transformation from one space to another using projective geometry. We use it in the step 12 to apply a perspective transformation to `mCorners` to identify the object corners in the camera image.

## There's more…

To learn more about what SURF is and how it works, please refer to the following web page: [http://en.wikipedia.org/wiki/SURF](http://en.wikipedia.org/wiki/SURF).

To learn more about FLANN, please refer to the web page [http://en.wikipedia.org/wiki/Nearest_neighbor_search](http://en.wikipedia.org/wiki/Nearest_neighbor_search).

To learn more about homography please refer to the following web page:

[http://en.wikipedia.org/wiki/Homography](http://en.wikipedia.org/wiki/Homography).

# Reading QR code

In this example we will use the ZXing library for QR code reading.

## Getting ready

Please download the Cinder ZXing block from GitHub and unpack it to the `blocks` folder: [https://github.com/dawidgorny/Cinder-ZXing](https://github.com/dawidgorny/Cinder-ZXing)

## How to do it…

We will now create a QR code reader:

1.  Add a header search path to the build settings of your project:

    [PRE49]

2.  Add a path from the precompiled ZXing library to the build settings of your project: `$(CINDER_PATH)/blocks/zxing/lib/macosx/libzxing.a`. For a debug configuration, use `$(CINDER_PATH)/blocks/zxing/lib/macosx/libzxing_d.a`.
3.  Add Cinder ZXing block files to your project structure as follows:![How to do it…](img/8703OS_11_02.jpg)
4.  Add the `libiconv.dylib` library to the `Link Binary With Libraries` list:![How to do it…](img/8703OS_11_03.jpg)
5.  Add the necessary header files:

    [PRE50]

6.  Add the following members to your main application class:

    [PRE51]

7.  Inside the `setup` method, set window dimensions and initialize capturing from camera:

    [PRE52]

8.  Implement the `update` function as follows:

    [PRE53]

9.  Implement the `draw` function as follows:

    [PRE54]

## How it works…

We are using regular ZXing library methods. The `SurfaceBitmapSource` class delivered by the Cinder ZXing block provides integration with Cinder `Surface` type objects. While the QR code is detected and read, the `mDetected` flag is set to `true` and the read data is stored in the `mData` member.

![How it works…](img/8703OS_11_06.jpg)

# Building UI navigation and gesture recognition with Kinect

In this recipe we will create interactive GUI controlled with a Kinect sensor.

### Tip

Since the **Kinect for Windows SDK** is available only for Windows, this recipe is written for Windows users only.

![Building UI navigation and gesture recognition with Kinect](img/8703OS_11_04.jpg)

## Getting ready

In this example we are using the `InteractiveObject` class that we covered in the *Creating an interactive object that responds to the mouse* recipe from [Chapter 10](ch10.html "Chapter 10. Interacting with the User"), *Interacting with the User*.

Download and install the Kinect for Windows SDK from [http://www.microsoft.com/en-us/kinectforwindows/](http://www.microsoft.com/en-us/kinectforwindows/).

Download the KinectSDK CinderBlock from GitHub at [https://github.com/BanTheRewind/Cinder-KinectSdk](https://github.com/BanTheRewind/Cinder-KinectSdk), and unpack it to the `blocks` directory.

## How to do it…

We will now create a Cinder application controlled with hand gestures.

1.  Include the necessary header files:

    [PRE55]

2.  Add the Kinect SDK using the following statement:

    [PRE56]

3.  Implement the class for a waving hand gesture recognition as follows:

    [PRE57]

4.  Implement `NuiInteractiveObject` extending the `InteractiveObject` class:

    [PRE58]

5.  Implement the `NuiController` class that manages the active objects:

    [PRE59]

6.  Add the members to you main application class for handling Kinect devices and data:

    [PRE60]

7.  Add members to store the calculated cursor position:

    [PRE61]

8.  Add the members that we will use for gesture recognition and user activation:

    [PRE62]

9.  Add a member to handle `NuiController`:

    [PRE63]

10.  Set window settings by implementing `prepareSettings`:

    [PRE64]

11.  In the `setup` method, set the default values for members:

    [PRE65]

12.  In the `setup` method initialize Kinect and gesture recognition for `10` users:

    [PRE66]

13.  In the `setup` method, initialize the user interface consisting of objects of type `NuiInterativeObject`:

    [PRE67]

14.  In the `update` method, we are checking if the Kinect device is capturing, getting tracked skeletons, and iterating:

    [PRE68]

15.  Inside the loop, we are checking if the skeleton is complete and deactivating the cursor controls if it is not complete:

    [PRE69]

16.  Inside the loop check if the skeleton is valid. Notice we are only processing 10 skeletons. You can modify this number, but remember to provide sufficient number of gesture controllers in `mGestureControllers`:

    [PRE70]

17.  Inside the loop and the `if` statement, check for the completed activation gesture. While the skeleton is activated, we are calculating person interaction zone:

    [PRE71]

18.  Inside the loop and the `if` statement, we are calculating cursor positions for active users:

    [PRE72]

19.  Close the opened `if` statements and the `for` loop:

    [PRE73]

20.  At the end of the `update` method, update the `NuiController` object:

    [PRE74]

21.  Implement the `draw` method as follows:

    [PRE75]

## How it works…

The application is tracking users using Kinect SDK. Skeleton data of the active user are used to calculate the cursor position by following the guidelines provided by Microsoft with Kinect SDK documentation. Activation is invoked by a hand waving gesture.

This is an example of UI responsive to cursor controlled by a user's hand. Elements of the grid light up under the cursor and fade out on roll-out.

# Building an augmented reality with Kinect

In this recipe we will learn how to combine both Kinect's depth and image frames to create augmented reality application.

### Tip

Since Kinect for Windows SDK is available only for Windows, this recipe is written for Windows users only.

## Getting ready

Download and install Kinect for Windows SDK from [http://www.microsoft.com/en-us/kinectforwindows/](http://www.microsoft.com/en-us/kinectforwindows/).

Download KinectSDK CinderBlock from GitHub at [https://github.com/BanTheRewind/Cinder-KinectSdk](https://github.com/BanTheRewind/Cinder-KinectSdk), and unpack it to the `blocks` directory.

In this example, we are using assets from one of the sample programs delivered with the Cinder package. Please copy the `ducky.mshducky.png`, `phong_vert.glsl`, and `phong_frag.glsl` files from `cinder_0.8.4_mac/samples/Picking3D/resources/` into your `assets` folder.

## How to do it…

We will now create an augmented reality application using a sample 3D model.

1.  Include the necessary header files:

    [PRE76]

2.  Add the `using` statement of the Kinect SDK:

    [PRE77]

3.  Add the members to you main application class for handling Kinect device and data:

    [PRE78]

4.  Add members to store 3D camera scene properties:

    [PRE79]

5.  Add members to store calibration settings:

    [PRE80]

6.  Add members that will store geometry, texture, and shader program for 3D object:

    [PRE81]

7.  Inside the `setup` method, set the window dimensions and initial values:

    [PRE82]

8.  Inside the `setup` method load geometry, texture, and shader program for 3D object:

    [PRE83]

9.  Inside the `setup` method, initialize the Kinect device and start capturing:

    [PRE84]

10.  At the end of the `setup` method, create GUI for parameter tweaking:

    [PRE85]

11.  Implement the `update` method as follows:

    [PRE86]

12.  Implement the `drawObject` method that will draw our 3D model with the texture and shading applied:

    [PRE87]

13.  Implement the `draw` method as follows:

    [PRE88]

14.  The last thing that is missing is the `draw3DScene` method invoked inside the `draw` method. Implement the `draw3DScene` method as follows:

    [PRE89]

15.  Implement the `shutdown` method to stop capturing from Kinect on program termination:

    [PRE90]

## How it works…

The application is tracking users using the Kinect SDK. Skeleton data of the users are used to calculate the coordinates of the 3D duck model taken from one of the Cinder sample programs. The 3D model is rendered right above the right hand of the user when the user's hand is in front of the user. The activation distance is calculated using the `mActivationDist` member value.

![How it works…](img/8703OS_11_05.jpg)

To properly overlay 3D scene onto a video frame, you have to set the camera FOV according to the Kinect video camera. To do this, we are using the `Camera FOV` property.