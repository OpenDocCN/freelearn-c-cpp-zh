# Chapter 4. Using Multimedia Content

In this chapter we will learn about:

*   Loading and displaying video
*   Creating a simple video controller
*   Saving window content as an image
*   Saving window animation as video
*   Saving window content as a vector graphics image
*   Saving high resolution images with tile renderer
*   Sharing graphics between applications

# Introduction

Most interesting applications use multimedia content in some form or another. In this chapter we will start by learning how to load, manipulate, and display video. We will then move on to saving our graphics into images, image sequences, or video, and then we will move to recording sound visualization.

Lastly, we will learn how to share graphics between applications and how to save mesh data.

# Loading and displaying video

In this recipe, we will learn how to load a video from a file and display it on screen using Quicktime and OpenGL. We'll learn how to load a file as a resource or from a file selected by the user using a file open dialog.

## Getting ready

You need to have QuickTime installed and also a video file in a format compatible with QuickTime.

To load the video as a resource it is necessary to copy it to the `resources` folder in your project. To learn more on resources, please read the recipes *Using resources on Windows* and *Using resources on OSX and iOS* from [Chapter 1](ch01.html "Chapter 1. Getting Started"), *Getting Started*.

## How to do it…

We will use Cinder's QuickTime wrappers to load and display vido.

1.  Include the headers containing the Quicktime and OpenGL functionality by adding the following at the beginning of the source file:

    [PRE0]

2.  Declare a `ci::qtime::MovieGl` member in you application's class declaration. This example will only need the `setup`, `update`, and `draw` methods, so make sure at least these are declared:

    [PRE1]

3.  To load the video as a resource use the `ci::app::loadResource` method with the file name as `parameter` and pass the resulting `ci::app::DataSourceRef` when constructing the movie object. It is also good practice to place the loading resource inside a `trycatch` segment in order to catch any resource loading errors. Place the following code inside your `setup` method:

    [PRE2]

4.  You can also load the video by using a file open dialog and passing the file path as an argument when constructing the `mMovie` object. Your `setup` would instead have the following code:

    [PRE3]

5.  To play the video, call the `play` method on the movie object. You can test the successful instantiation of `mMovie` by placing it inside an `if` statement just like an ordinary pointer:

    [PRE4]

6.  In the `update` method we copy the texture of the current movie frame into our `mMovieTexture` to draw it later:

    [PRE5]

7.  To draw the movie we simply need to draw our texture on screen using the method `gl::draw`. We need to check if the texture is valid because `mMovie` may take a while to load. We'll also create `ci::Rectf` with the texture size and center it on screen to keep the drawn video centered without stretching:

    [PRE6]

## How it works…

The `ci::qtime::MovieGl` class allows playback and control of movies by wrapping around the QuickTime framework. Movie frames are copied into OpenGl textures for easy drawing. To access the texture of the current frame of the movie use the method `ci::qtime::MovieGl::getTexture()` which returns a `ci::gl::Texture` object. Textures used by `ci::qtime::MovieGl` are always bound to the `GL_TEXTURE_RECTANGLE_ARB` target.

## There's more

If you wish to do iterations over the pixels of a movie consider using the class `ci::qtime::MovieSurface`. This class allows playback of movies by wrapping around the QuickTime framework, but converts movie frames into `ci::Surface` objects. To access the current frame's surface, use the method `ci::qtime::MovieSurface::getSurface()` which returns a `ci::Surface` object.

# Creating a simple video controller

In this recipe we'll learn how to create a simple video controller using the built-in GUI functionalities of Cinder.

We'll control movie playback, if the movie loops or not, the speed rate, volume, and the position.

## Getting ready

You must have Apple's QuickTime installed and a movie file in a format compatible with QuickTime.

To learn how to load and display a movie please refer to the previous recipe *Loading and displaying Video*.

## How to do it…

We will create a simple interface using Cinder `params` classes to control a video.

1.  Include the necessary files to work with Cinder `params` (QuickTime and OpenGl) by adding the following at the top of the source file:

    [PRE7]

2.  Add the `using` statements before the application's class declaration to simplify calling Cinder commands as shown in the following code lines:

    [PRE8]

3.  Declare a `ci::qtime::MovieGl`, `ci::gl::Texture`, and a `ci::params::InterfaceGl` object to play, render, and control the video respectively. Add the following to your class declaration:

    [PRE9]

4.  Select a video file by opening an open file dialog and use that path to initialize our `mMovie`. The following code should go in the `setup` method:

    [PRE10]

5.  We'll also need some variables to store the values which we'll manipulate. Each controllable parameter of the video will have two variables to represent the current and the previous value of that parameter. Now declare the following variables:

    [PRE11]

6.  Set the default values in the `setup` method:

    [PRE12]

7.  Now let's initialize `mParams` and add a control for each of the previously defined variables and set the `max`, `min`, and `step` values when necessary. The following code must go in the `setup` method:

    [PRE13]

8.  In the `update` method we'll check if the movie was valid and compare each of the parameters to their previous state to see if they changed. If it did, we'll update `mMovie` and set the parameter to the new value. The following code lines go in the `update` method:

    [PRE14]

9.  In the `update` method it is also necessary to get a handle to the movie texture and copy it to our previously declared `mMovieTexture`. In the `update` method we write:

    [PRE15]

10.  All that is left is to draw our content. In the `draw` method we'll clear the background with black. We'll check the validity of `mMovieTexture` and draw it in a rectangle that fits on the window. We also call the `draw` command of `mParams` to draw the controls on top of the video:

    [PRE16]

11.  Draw it and you'll see the application's window with a black background along with the controls. Change the various parameters in the parameters menu and you'll see it affecting the video:![How to do it…](img/8703OS_4_1.jpg)

## How it works…

We created a `ci::params::InterfaceGl` object and added a control for each of the parameters we wanted to manipulate.

We created a variable for each of the parameters we want to manipulate and a variable to store their previous value. In the update we checked to see if these values differ, which will only happen when the user has changed their value using the `mParams` menu.

When the parameter changes we change the `mMovie` parameter with the value the user has set.

Some parameters must be kept in a specific range. The movie position is set in seconds from `0` to the maximum duration of the video in seconds. The volume must be a value between `0` and `1`, `0` meaning no audio and `1` being the maximum volume.

# Saving window content as an image

In this example we will show you how to save window content to the graphic file and how to implement this functionality in your Cinder application. This could be useful to save output of a graphics algorithm.

## How to do it…

We will add a window content saving function to your application:

1.  Add necessary headers:

    [PRE17]

2.  Add property to your application's main class:

    [PRE18]

3.  Set a default value inside the `setup` method:

    [PRE19]

4.  Implement the `keyDown` method as follows:

    [PRE20]

5.  Add the following code at the end of the `draw` method:

    [PRE21]

## How it works…

Every time you set `mMakeScreenshot` to `true` the screenshot of your application will be selected and saved. In this case the application waits for the *S* key to be pressed and then sets the flag `mMakeScreenshot` to `true`. The current application window screenshot will be saved inside your documents directory under the name `MainApp_screenshot.png`.

## There's more...

This is just the basic example of common usage of the `writeImage` function. There are many other practical applications.

### Saving window animation as image sequences

Let's say you want to record a equence of images Perform the following steps to do so:

1.  Modify the previous code snippet shown in step 5 for saving the window content as follows:

    [PRE22]

2.  You have to define `mRecordFrames` and `mFrameCounter` as properties of your main application class:

    [PRE23]

3.  Set initial values inside the `setup` method:

    [PRE24]

### Recording sound visualization

We assume that you are using `TrackRef` from the `audio` namespace to play your sound Perform the following steps:

1.  Implement the previous steps for saving window animations as image sequences.
2.  Type the following lines of code at the beginning of the `update` method:

    [PRE25]

We are calculating the desired audio track position based on the number of frames that passed. We are doing that to synchronize animation with the music track. In this case we want to produce `30` fps animation so we are dividing `mFramesCounter` by `30`.

# Saving window animations as video

In this recipe,we'll start by drawing a simple animation and learning how to export it to video. We will create a video where pressing any key will start or stop the recording.

## Getting ready

You must have Apple's QuickTime installed. Make sure you know where you want your video to be saved, as you'll have to specify its location at the beginning.

It could be anything that is drawn using OpenGl but for this example, we'll create a yellow circle at the center of the window with a changing radius. The radius is calculated by the absolute value of the sine of the elapsed seconds since the application launched. We multiply this value by `200` to scale it up. Now add the following to the `draw` method:

[PRE26]

## How to do it…

We will use the `ci::qtime::MovieWriter` class to create a video of our rendering.

1.  Include the OpenGl and QuickTime files at the beginning of the source file by adding the following:

    [PRE27]

2.  Now let's declare a `ci::qtime::MovieWriter` object and a method to initialize it. Add the following to your class declaration:

    [PRE28]

3.  In the implementation of `initMovieWriter` we start by asking the user to specify a path using a save file dialog and use it to initialize the movie writer. The movie writer also needs to know the window's width and height. Here's the implementation of `initMovieWriter`.

    [PRE29]

4.  Lets declare a key event handler by declaring the `keyUp` method.

    [PRE30]

5.  In its implementation we will see if there is already a movie being recorded by checking the validity of `mMovieWriter`. If it is a valid object then we must save the current movie by destroying the object. We can do so by calling the `ci::qtime::MovieWriter` default constructor; this will create a null instance. If `mMovieWriter` is not a valid object then we initialize a new movie writer by calling the method `initMovieWriter()`.

    [PRE31]

6.  The last two steps are to check if `mMovieWriter` is valid and to add a frame by calling the method `addFrame` with the window's surface. This method has to be called in the `draw` method, after our drawing routines have been made. Here's the final `draw` method, including the circle drawing code.

    [PRE32]

7.  Build and run the application. Pressing any key will start or end a video recording. Each time a new recording begins, the user will be presented with a save file dialog to set where the movie will be saved.![How to do it…](img/8703OS_4_2.jpg)

## How it works…

The `ci::qtime::MovieWriter` object allows for easy movie writing using Apple's QuickTime. Recordings begin by initializing a `ci::qtime::MovieWriter` object and are saved when the object is destroyed. By calling the `addFrame` method, new frames are added.

## There's more...

You can also define the format of the video by creating a `ci::qtime::MovieWriter::Format` object and passing it as an optional parameter in the movie writer's constructor. If no format is specified, the movie writer will use the default PNG codec and 30 frames per second.

For example, to create a movie writer with the H264 codec with 50 percent quality and 24 frames per second, you could write the following code:

[PRE33]

You can optionally open a **Settings** window and allow the user to define the video settings by calling the static method `qtime::MovieWriter::getUserCompressionSettings`. This method will populate a `qtime::MovieWriter::Format` object and return `true` if successful or `false` if the user canceled the change in the setting.

To use this method for defining the settings and creating a movie writer, you can write the following code:

[PRE34]

It is also possible to enable **multipass** encoding. For the current version of Cinder it is only available using the H264 codec. Multipass encoding will increase the movie's quality but at the cost of a greater performance decrease. For this reason it is disabled by default.

To write a movie with multipass encoding enabled we can write the following:

[PRE35]

There are plenty of settings and formats that can be set using the `ci::qtime::MovieWriter::Format` class and the best way to know the full list of options is to check the documentation for the class at [http://libcinder.org/docs/v0.8.4/guide__qtime___movie_writer.html](http://libcinder.org/docs/v0.8.4/guide__qtime___movie_writer.html).

# Saving window content as a vector graphics image

In this recipe we'll learn how to draw 2D graphics on screen and save it to an image in a vector graphics format using the cairo renderer.

Vector graphics can be extremely useful when creating visuals for printing as they can be scaled without losing quality.

Cinder has an integration for the cairo graphics library; a powerful and full-featured 2D renderer, capable of outputting to a variety of formats including popular vector graphics formats.

To learn more about the cairo library, please go to its official web page: [http://www.cairographics.org](http://www.cairographics.org)

In this example we'll create an application that draws a new circle whenever the user presses the mouse. When any key is pressed, the application will open a save file dialog and save the content in a format defined by the file's extension.

## Getting ready

To draw graphics created with the cairo renderer we must define our renderer to be `Renderer2d`.

At the end of the source file of our application class there's a *macro* to initialize the application where the second parameter defines the renderer. If your application is called `MyApp`, you must change the macro to be the following:

[PRE36]

The cairo renderer allows exporting of PDF, SVG, EPS, and PostScript formats. When specifying the file to save, make sure you write one of the supported extensions: `pdf`, `svg`, `eps`, or `ps`.

Include the following files at the top of your source file:

[PRE37]

## How to do it…

We will use Cinder's cairo wrappers to create images in vector formats from our rendering.

1.  To create a new circle every time the user presses the mouse we must first create a `Circle` class. This class will contain position, radius, and color parameters. Its constructor will take `ci::Vec2f` to define its position and will generate a random radius and color.

    Write the following code before the application's class declaration:

    [PRE38]

2.  We should now declare `std::vector` of circles where we'll store the created circles. Add the following code to your class declaration:

    [PRE39]

3.  Let's create a method which will draw the circles that will take `cairo::Context` as their parameter:

    [PRE40]

4.  In the method definition, iterate over `mCircles` and draw each one in the context:

    [PRE41]

5.  At this point we only need to add a circle whenever the user presses the mouse. To do this, we must implement the `mouseDown` event handler by declaring it in the class declaration.

    [PRE42]

6.  In its implementation we add a `Circle` class to `mCircles` using the mouse position.

    [PRE43]

7.  We can now draw the circles on the window by creating `cairo::Context` bound to the window's surface. This will let us visualize what we're drawing. Here's the `draw` method implementation:

    [PRE44]

8.  To save the scene to an image file we must create a context bound to a surface that represents a file in a vector graphics format. Let's do this whenever the user releases a key by declaring the `keyUp` event handler.

    [PRE45]

9.  In the `keyUp` implementation we create `ci::fs::path` and populate it by calling a save file dialog. We'll also create an empty `ci::cairo::SurfaceBase` which is the base for all the surfaces that the cairo renderer can draw to.

    [PRE46]

10.  We'll now compare the extension of the path with the supported formats and initialize the surface accordingly. It can be initialized as `ci::cairo::SurfacePdf`, `ci::cairo::SurfaceSvg`, `ci::cairo::SurfaceEps`, or as `ci::cairo::SurfacePs`.

    [PRE47]

11.  Now we can create `ci::cairo::Context` and render our scene to it by calling the `renderScene` method and passing the context as a parameter. The circles will be rendered to the context and a file will be created in the specified format. Here's the final `keyUp` method implementation:

    [PRE48]

    ![How to do it…](img/8703OS_4_3.jpg)

## How it works…

Cinder wraps and integrates the cairo 2D vector renderer. It allows use of Cinder's types to draw and interact with cairo.

The complete drawing is made by calling the drawing methods of a `ci::cairo::Context` object. The context in turn, must be created by passing a surface object extending `ci::cairo::SurfaceBase`. All drawings will be made in the surface and rasterized according to the type of the surface.

The following surfaces allow saving images in a vector graphics format:

| Surface type | Format |
| --- | --- |
| `ci::cairo::SurfacePdf` | PDF |
| `ci::cairo::SurfaceSvg` | SVG |
| `ci::cairo::SurfaceEps` | EPS |
| `ci::cairo::SurfacePs` | PostsSript |

## There's more...

It is also possible to draw using other renderers. Though the renderers aren't able to create vector images, they can be useful in other situations.

Here are the other available surfaces:

| Surface Type | Format |
| --- | --- |
| `ci::cairo::SurfaceImage` | Anti-aliased pixel-based rasterizer |
| `ci::cairo::SurfaceQuartz` | Apple's Quartz |
| `ci::cairo::SurfaceCgBitmapContext` | Apple's CoreGraphics |
| `ci::cairo::SurfaceGdi` | Windows GDI |

# Saving high resolution images with the tile renderer

In this recipe we'll learn how to export a high-resolution image of the content being drawn on screen using the `ci::gl::TileRender` class. This can be very useful when creating graphics for print.

We'll start by creating a simple scene and drawing it on screen. Next, we'll code our example so that whenever the user presses any key, a save file dialog will appear and a high-resolution image will be saved to the specified path.

## Getting ready

The `TileRender` class can create high resolution images from anything being drawn on screen using OpenGl calls.

To save an image with `TileRender` we must first draw some content on screen. It can be anything but for this example let's create a nice simple pattern with circles to fill the screen.

In the implementation of your `draw` method write the following code:

[PRE49]

Remember that this could be anything that is drawn on screen using OpenGl.

![Getting ready](img/8703OS_4_4.jpg)

## How to do it...

We will use the `ci::gl::TileRender` class to generate high-resolution images of our OpenGL rendering.

1.  Include the necessary headers by adding the following at the top of the source file:

    [PRE50]

2.  Since we'll save a high-resolution image whenever the user presses any key, let's implement the `keyUp` event handler by declaring it in the class declaration.

    [PRE51]

3.  In the `keyUp` implementation we start by creating a `ci::gl::TileRender` object and then set the width and height of the image we are going to create. We are going to set it to be four times the size of the application window. It can be of any size you want, just take in to account that if you don't respect the window's aspect ratio, the image will become stretched.

    [PRE52]

4.  We must define our scene's `Modelview` and `Projection` matrices to match our window. If we are using only 2D graphics we can call the method `setMatricesWindow`, as follows:

    [PRE53]

    To define the scene's `Modelview` and `Projection` matrices to match the window while drawing 3D content, it is necessary to call the method `setMatricesWindowPersp`:

    [PRE54]

5.  Next we'll draw our scene each time a new tile is created by using the method `nextTile`. When all the tiles have been created the method will return `false`. We can create all the tiles by redrawing our scene in a `while` loop while asking if there is a next tile, as follows:

    [PRE55]

6.  Now that the scene is fully rendered in `TileRender`, we must save it. Let's ask the user to indicate where to save by opening a save file dialog. It is mandatory to specify an extension for the image file as it will be used internally to define the image format.

    [PRE56]

7.  We check if `filePath` is not empty and write the tile render surface as an image using the `writeImage` method.

    [PRE57]

8.  After saving the image it is necessary to redefine the window's `Modelview` and `Projection` matrices. If drawing in 2D you can set the matrices to their default values by using the method `setMatricesWindow` with the window's dimensions, as follows:

    [PRE58]

## How it works…

The `ci::gl::TileRender` class makes it possible to generate high-resolution versions of our rendering by scaling individual portions of our drawing to the entire size of the window and storing them as `ci::Surface`. After the entire scene has been stored in individual portions it is stitched together as tiles to form a single high-resolution `ci::Surface`, which can then be saved as an image.

# Sharing graphics between applications

In this recipe we will show you the way of sharing graphic in real time between applications under Mac OS X. To do that, we will use **Syphon** and its implementation for Cinder. Syphon is an open source tool that allows an application to share graphics as still frames or real-time updated frame sequence. You can read more about Syphon here: [http://syphon.v002.info/](http://syphon.v002.info/)

## Getting ready

To test if the graphic shared by our application is available, we are going to use **Syphon Recorder**, which you can find here: [http://syphon.v002.info/recorder/](http://syphon.v002.info/recorder/)

## How to do it…

1.  Checkout Syphon CinderBlock from the *syphon-implementations* repository [http://code.google.com/p/syphon-implementations/](http://code.google.com/p/syphon-implementations/).
2.  Create a new group inside your project tree and name it `Blocks`.
3.  Drag-and-drop Syphon CinderBlock into your newly created `Blocks` group.![How to do it…](img/8703OS_4_5.jpg)
4.  Make sure **Syphon.framework** is added to the **Copy Files** section of **Build Phases** in the **target** settings.
5.  Add necessary header files:

    [PRE59]

6.  Add property to your main application class:

    [PRE60]

7.  At the end of `setup` method, add the following code:

    [PRE61]

8.  Inside the `draw` method add the following code:

    [PRE62]

## How it works…

Application draws a simple rotating animation and shares the whole window area via Syphon library. Our application window looks like the following screenshot:

![How it works…](img/8703OS_4_6.jpg)

To test if the graphic can be received by other applications, we will use Syphon Recorder. Run Syphon Recorder and find our Cinder application in the drop-down menu under the name: **Cinder Screen – MainApp**. We set up the first part of this name at the step 6 of this recipe in the *How to do it...* section while the second part is an executable file name. Now, the preview from our Cinder application should be available and it would looks like the following screenshot:

![How it works…](img/8703OS_4_7.jpg)

## There's more...

The Syphon library is very useful, simple to use, and is available for other applications and libraries.

### Receiving graphics from other applications

You can receive textures from other applications as well. To do this, you have to use the `syphonClient` class as shown in the following steps:

1.  Add a property to your application main class:

    [PRE63]

2.  Initialize `mClientSyphon` inside the CIT method:

    [PRE64]

3.  At the end of the `draw` method add the following line which draws graphics that the other application is sharing:

    [PRE65]