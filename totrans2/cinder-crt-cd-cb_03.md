# Chapter 3. Using Image Processing Techniques

In this chapter we will cover:

*   Transforming image contrast and brightness
*   Integrating with OpenCV
*   Detecting edges
*   Detecting faces
*   Detecting features in image
*   Converting images to vector graphics

# Introduction

In this chapter, we will show examples of using image processing techniques implemented in Cinder and using third-party libraries. In most of the examples, we will use the following famous test image widely used to illustrate computer vision algorithms and techniques:

![Introduction](img/8703OS_03_01.jpg)

You can download Lenna's image from Wikipedia ([http://en.wikipedia.org/wiki/File:Lenna.png](http://en.wikipedia.org/wiki/File:Lenna.png)).

# Transforming image contrast and brightness

In this recipe we will cover basic image color transformations using the `Surface` class for pixel manipulation.

## Getting ready

To change the values of contrast and brightness we will use `InterfaceGl` covered in [Chapter 2](ch02.html "Chapter 2. Preparing for Development"), *Preparing for Development in the Setting up GUI for parameters tweaking* recipe. We will need a sample image to proceed with; save it in your `assets` folder as `image.png`.

## How to do it...

We will create an application with simple GUI for contrast and brightness manipulation on the sample image. Perform the following steps to do so:

1.  Include necessary headers:

    [PRE0]

2.  Add properties to the main class:

    [PRE1]

3.  In the `setup` method an image is loaded for processing and the `Surface` object is prepared to store processed image:

    [PRE2]

4.  Set window size to default values:

    [PRE3]

5.  Add parameter controls to the `InterfaceGl` window:

    [PRE4]

6.  Implement the `update` method as follows:

    [PRE5]

7.  Lastly, we will draw the original and processed images by adding the following lines of code inside the `draw` method:

    [PRE6]

## How it works...

The most important part is inside the `update` method. In step 6 we checked if the parameters for contrast and brightness had been changed. If they have, we iterate through all the pixels of the original image and store recalculated color values in `mImageOutput`. While modifying the brightness is just increasing or decreasing each color component, calculating contrast is a little more complicated. For each color component we are using the multiplying formula, *color = (color - 0.5) * contrast + 0.5*, where contrast is a number between 0.5 and 2\. In the GUI we are setting a value between -0.5 and 1.0, which is more natural range; it is then recalculated at the beginning of step 6\. While processing the image we have to change color value of all pixels, so later in step 6, you can see that we iterate through later columns of each row of the pixels using two `while` loops. To move to the next row we invoked the `line` method on the `Surface` iterator and then the `pixel` method to move to the next pixel of the current row. This method is much faster than using, for example, the `getPixel` and `setPixel` methods.

![How it works...](img/8703OS_03_02.jpg)

Our application is rendering the original image on the left-hand side and the processed image on the right-hand side, so you can compare the results of color adjustment.

# Integrating with OpenCV

OpenCV is a very powerful open-source library for computer vision. The library is written in C++ so it can be easily integrated in your Cinder application. There is a very useful OpenCV Cinder block provided within Cinder package available at the GitHub repository ([https://github.com/cinder/Cinder-OpenCV](https://github.com/cinder/Cinder-OpenCV)).

## Getting ready

Make sure you have Xcode up and running with a Cinder project opened.

## How to do it…

We will add OpenCV Cinder block to your project, which also illustrates the usual way of adding any other Cinder block to your project. Perform the following steps to do so:

1.  Add a new group to our Xcode project root and name it `Blocks.` Next, drag the `opencv` folder inside the `Blocks` group. Be sure to select the **Create groups for any added folders** radio button, as shown in the following screenshot:![How to do it…](img/8703OS_03_03.jpg)
2.  You will need only the `include` folder inside the `opencv` folder in your project structure, so delete any reference to others. The final project structure should look like the following screenshot:![How to do it…](img/8703OS_03_04.jpg)
3.  Add the paths to the OpenCV library files in the **Other Linker Flags** section of your project's build settings, for example:

    [PRE7]

    These paths are shown in the following screenshot:

    ![How to do it…](img/8703OS_03_05.jpg)
4.  Add the paths to the OpenCV Cinder block headers you are going to use in the **User Header Search Paths** section of your project's build settings:

    [PRE8]

    This path is shown in the following screenshot:

    ![How to do it…](img/8703OS_03_06.jpg)
5.  Include OpenCV Cinder block header file:

    [PRE9]

## How it works…

OpenCV Cinder block provides the `toOcv` and `fromOcv` functions for data exchange between Cinder and OpenCV. After setting up your project you can use them, as shown in the following short example:

[PRE10]

You can use the `toOcv` and `fromOcv` functions to convert between Cinder and OpenCV types, storing image data such as `Surface` or `Channel` handled through the `ImageSourceRef` type; there are also other types, as shown in the following table:

| Cinder types | OpenCV types |
| --- | --- |
| `ImageSourceRef` | `Mat` |
| `Color` | `Scalar` |
| `Vec2f` | `Point2f` |
| `Vec2i` | `Point` |
| `Area` | `Rect` |

In this example we are linking against the following three files from the OpenCV package:

*   `libopencv_imgproc.a`: This image processing module includes image manipulation functions, filters, feature detection, and more
*   `libopencv_core.a`: This module provides core functionality and data structures
*   `libopencv_objdetect.a`: This module has object detection tools such as cascade classifiers

You can find the documentation on all OpenCV modules at [http://docs.opencv.org/index.html](http://docs.opencv.org/index.html).

## There's more…

There are some features that are not available in precompiled OpenCV libraries packaged in OpenCV Cinder block, but you can always compile your own OpenCV libraries and still use exchange functions from OpenCV Cinder block in your project.

# Detecting edges

In this recipe, we will demonstrate how to use edge detection function, which is one of the image processing functions implemented directly in Cinder.

## Getting ready

Make sure you have Xcode up and running with an empty Cinder project opened. We will need a sample image to proceed, so save it in your assets folder as `image.png`.

## How to do it…

We will process the sample image with the edge detection function. Perform the following steps to do so:

1.  Include necessary headers:

    [PRE11]

2.  Add two properties to your main class:

    [PRE12]

3.  Load the source image and set up `Surface` for processed images inside the `setup` method:

    [PRE13]

4.  Use image processing functions:

    [PRE14]

5.  Inside the `draw` method add the following two lines of code for drawing images:

    [PRE15]

## How it works…

As you can see, detecting edges in Cinder is pretty easy because of implementation of basic image processing functions directly in Cinder, so you don't have to include any third-party libraries. In this case we are using the `grayscale` function to convert the original image color space to grayscale. It is a commonly used feature in image processing because many algorithms work more efficiently on grayscale images or are even designed to work only with grayscale source images. The edge detection is implemented with the `edgeDetectSobel` function and uses the Sobel algorithm. In this case, the first parameter is the source original grayscale image and the second parameter, is the output `Surface` object in which the result will be stored.

Inside the `draw` method we are drawing both images, as shown in the following screenshot:

![How it works…](img/8703OS_03_07.jpg)

## There's more…

You may find the image processing functions implemented in Cinder insufficient, so you can also include to your project, third-party library such as OpenCV. We explained how we can use Cinder and OpenCV together in the preceding recipe, *Integrating with OpenCV*.

Other useful functions in the context of edge detection are `Canny` and `findContours`. The following is the example of how we can use them:

[PRE16]

After executing the preceding code, the points, which form the contours are stored in the `contours` variable.

# Detecting faces

In this recipe, we will examine how our application can be used to recognize human faces. Thanks to the OpenCV library, it is really easy.

## Getting ready

We will be using the OpenCV library, so please refer to the *Integrating with OpenCV* recipe for information on how to set up your project. We will need a sample image to proceed, so save it in your `assets` folder as `image.png`. Put the Haar cascade classifier file for frontal face recognition inside the `assets` directory. The cascade file can be found inside the downloaded OpenCV package or in the online public repository, located at [https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_alt.xml](https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_alt.xml).

## How to do it…

We will create an application that demonstrates the usage of cascade classifier from OpenCV with Cinder. Perform the following steps to do so:

1.  Include necessary headers:

    [PRE17]

2.  Add the following members to your main class:

    [PRE18]

3.  Add the following code snippet to the `setup` method:

    [PRE19]

4.  Also add the following code snippet at the end of the `setup` method:

    [PRE20]

5.  At the end of the `draw` method add the following code snippet:

    [PRE21]

## How it works…

In step 3 we loaded an image file for processing and an XML classifier file, which has description of the object features to be recognized. In step 4 we performed an image detection by invoking the `detectMultiScale` function on the `mFaceCC` object, where we pointed to `cvImage` as an input and stored the result in a vector structure, `cvImage` is converted from `mImage` as an 8-bit, single channel image (`CV_8UC1`). What we did next was iterating through all the detected faces and storing `Rectf` variable, which describes a bounding box around the detected face. Finally, in step 5 we drew our original image and all the recognized faces as stroked rectangles.

We are using cascade classifier implemented in OpenCV, which can be trained to detect a specific object in the image. More on training and using cascade classifier for object detection can be found in the OpenCV documentation, located at [http://docs.opencv.org/modules/objdetect/doc/cascade_classification.html](http://docs.opencv.org/modules/objdetect/doc/cascade_classification.html).

![How it works…](img/8703OS_03_08.jpg)

## There's more…

You can use a video stream from your camera and process each frame to track faces of people in real time. Please refer to the *Capturing from the camera* recipe in [Chapter 11](ch11.html "Chapter 11. Sensing and Tracking Input from the Camera"), *Sensing and Tracking Input from the Camera*.

# Detecting features in an image

In this recipe we will use one of the methods of finding characteristic features in the image. We will use the SURF algorithm implemented by the OpenCV library.

## Getting ready

We will be using the OpenCV library, so please refer to the *Integrating with OpenCV* recipe for information on how to set up your project. We will need a sample image to proceed, so save it in your `assets` folder as `image.png`, then save a copy of the sample image as `image2.png` and perform some transformation on it, for example rotation.

## How to do it…

We will create an application that visualizes matched features between two images. Perform the following steps to do so:

1.  Add the paths to the OpenCV library files in the **Other Linker Flags** section of your project's build settings, for example:

    [PRE22]

2.  Include necessary headers:

    [PRE23]

3.  In your main class declaration add the method and properties:

    [PRE24]

4.  Inside the `setup` method load the images and invoke the matching method:

    [PRE25]

5.  Now you have to implement previously declared `matchImages` method:

    [PRE26]

6.  The last thing is to visualize the matches, so put the following line of code inside the `draw` method:

    [PRE27]

## How it works…

Let's discuss the code under step 5\. First we are converting `image1` and `image2` to an OpenCV Mat structure. Then we are converting both images to grayscale. Now we can start processing images with SURF, so we are detecting keypoints – the characteristic points of the image calculated by this algorithm. We can use calculated keypoints from these two images and match them using FLANN, or more precisely the `FlannBasedMatcher` class. After filtering out the proper matches and storing them in the `good_matches` vector we can visualize them, as follows:

![How it works…](img/8703OS_03_09.jpg)

Please notice that second image is rotated, however the algorithm can still find and link the corresponding keypoints.

## There's more…

Detecting characteristic features in the images is crucial for matching pictures and is part of more advanced algorithms used in augmented reality applications.

### If images match

It is possible to determine if one of the images is a copy of another or is it rotated. You can use a number of matches returned by the `matchImages` method.

### Other possibilities

SURF is rather a slow algorithm for real-time matching so you can try the FAST algorithm for your project if you need to process frames from the camera at real time. The FAST algorithm is also included in the OpenCV library.

## See also

*   The comparison of the OpenCV's feature detection algorithms can be found at [http://computer-vision-talks.com/2011/01/comparison-of-the-opencvs-feature-detection-algorithms-2/](http://computer-vision-talks.com/2011/01/comparison-of-the-opencvs-feature-detection-algorithms-2/)

# Converting images to vector graphics

In this recipe, we will try to convert simple, hand-drawn sketches to vector graphics using image processing functions from the OpenCV library and Cairo library for vector drawing and exporting.

## Getting started

We will be using the OpenCV library, so please refer to the *Integrating with OpenCV* recipe earlier in this chapter for information on how to set up your project. You may want to prepare your own drawing to be processed. In this example we are using a photo of some simple geometric shapes sketched on paper.

![Getting started](img/8703OS_03_10.jpg)

## How to do it…

We will create an application to illustrate the conversion to vector shapes. Perform the following steps to do so:

1.  Include necessary headers:

    [PRE28]

2.  Add the following declarations to your main class:

    [PRE29]

3.  Load your drawing and set default values inside the `setup` method:

    [PRE30]

4.  At the end of the `setup` method add the following code snippet:

    [PRE31]

5.  Add implementation for the `renderDrawing` method:

    [PRE32]

6.  Implement your `draw` method as follows:

    [PRE33]

7.  Inside the `keyDown` method insert the following code snippet:

    [PRE34]

## How it works…

The key part is implemented in step 4 where we are detecting edges in the image and then finding contours. We are drawing vector representation of processed shapes in step 5, inside the `renderDrawing` method. For drawing vector graphics we are using the Cairo library, which is also able to save results into a file in several vector formats. As you can see in the following screenshot, there is an original image in the upper-left corner and just under it is the preview of the detected contours. The vector version of our simple hand-drawn image is on the right-hand side:

![How it works…](img/8703OS_03_11.jpg)

Each shape is a filled path with black color. Paths consist of points calculated in step 4\. The following is the visualization with highlighted points:

![How it works…](img/8703OS_03_12.jpg)

You can save a vector graphic as a file by pressing the *S* key. The file will be saved in the same folder as application executable under the name `output.svg`. SVG is only one of the following available exporting options:

| Method | Usage |
| --- | --- |
| `SurfaceSvg` | Preparing context for SVG file rendering |
| `SurfacePdf` | Preparing context for PDF file rendering |
| `SurfacePs` | Preparing context for PostScript file rendering |
| `SurfaceEps` | Preparing context for Illustrator EPS file rendering |

The exported graphics look as follows:

![How it works…](img/8703OS_03_13.jpg)

## See also

*   **Cairo**: [http://cairographics.org/](http://cairographics.org/)