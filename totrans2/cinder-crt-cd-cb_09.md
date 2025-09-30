# Chapter 9. Adding Animation

In this chapter, we will learn the techniques of animating 2D and 3D objects. We will introduce Cinder's features in this field, such as timeline `and math` functions.

The recipes in this chapter will cover the following:

*   Animating with the timeline
*   Creating animation sequences with the timeline
*   Animating along a path
*   Aligning camera motion to a path
*   Animating text – text as a mask for a movie
*   Animating text – scrolling text lines
*   Creating a flow field with Perlin noise
*   Creating an image gallery in 3D
*   Creating a spherical flow field with Perlin noise

# Animating with the timeline

In this recipe, we will learn how we can animate values using Cinder's new feature; the timeline.

We animate the background color and a circle's position and radius whenever the user presses the mouse button.

## Getting ready

Include the necessary files to use the timeline, generate random numbers, and draw using OpenGL. Add the following code snippet at the top of the source file:

[PRE0]

Also, add the following useful `using` statements:

[PRE1]

## How to do it…

We will create several parameters that will be animated with the timeline. Perform the following steps to do so:

1.  Declare the following members to be animated:

    [PRE2]

2.  Initialize the parameters in the `setup` method.

    [PRE3]

3.  In the `draw` method, we need to clear the background using the color defined in `mBackgroundColor` and draw a circle at `mCenter` with `mRadius` as the radius.

    [PRE4]

4.  To animate the values whenever the user presses the mouse button, we need to declare the `mouseDown` event handler.

    [PRE5]

5.  Let's implement the `mouseDown` event handler and add the animations to the main timeline. We will animate `mBackgroundColor` to a new random color, set `mCenter` to the mouse cursor's position, and set `mRadius` to a new random value.

    [PRE6]

## How it works…

The timeline is a new feature of Cinder introduced in version 0.8.4\. It permits the user to animate parameters by adding them to the timeline once, and everything gets updated behind the scenes.

Animations must be objects of the template class `ci::Anim`. This class can be created using any template type that supports the `+` operator.

The main `ci::Timeline` object can be accessed by calling the `ci::app::App::timeline()` method. There is always a main timeline and the user can also create other `ci::Timeline` objects.

The fourth parameter in the `ci::Timeline::apply` method is a `functor` object that represents a Tween method. Cinder has several Tweens available that can be passed as a parameter to define the type of animation.

## There's more…

The `ci::Timeline::apply` method used in the preceding example uses the initial value of the `ci::Anim` object, but it is also possible to create an animation where both the begining and end values are passed.

For example, if we wanted to animate `mRadius` from a starting value of 10.0 to the end value of 100.0 seconds, we would call the following method:

[PRE7]

## See also

*   To see all the available easing functions, please refer to the Cinder documentation, located at [http://libcinder.org/docs/v0.8.4/_easing_8h.html](http://libcinder.org/docs/v0.8.4/_easing_8h.html).

# Creating animation sequences with the timeline

In this recipe, we will learn how to use the powerful timeline features of Cinder to create sequences of animations. We will draw a circle and animate the radius and color in a sequenced manner.

## Getting ready

Include the necessary files to use the timeline, draw in OpenGL, and generate random numbers.

[PRE8]

Also, add the following useful `using` statements:

[PRE9]

## How to do it…

We will animate several parameters sequentially using the timeline. Perform the following steps to do so:

1.  Declare the following members to define the circle's position, radius, and color:

    [PRE10]

2.  In the `setup` method, initialize the members. Set the position to be at the center of the window, the radius as 30, and a random color using the HSV color mode.

    [PRE11]

3.  In the `draw` method, we will clear the background with black and draw the circle using the previously defined members.

    [PRE12]

4.  Declare the `mouseDown` event handler.

    [PRE13]

5.  In the implementation of `mouseDown`, we will apply the animations to the main timeline.

    We will first animate `mRadius` from 30 to 200 and append another animation to `mRadius` from 200 to 30.

    Add the following code snippet to the `mouseDown` method:

    [PRE14]

6.  Let's create a random color using the HSV color mode and use it as the target color to animate `mColor` and then append this animation to `mRadius`.

    Add the following code snippet inside the `mouseDown` method:

    [PRE15]

## How it works…

Appending animations is a powerful and easy way to create complex animation sequences.

In step 5 we append an animation to `mRadius` using the following line of code:

[PRE16]

This means this animation will only occur after the previous `mRadius` animation has finished.

In step 6 we append the `mColor` animation to `mRadius` using the following line of code:

[PRE17]

This means the `mColor` animation will only occur when the previous `mRadius` animation has finished.

## There's more…

When appending two different animations, it is possible to offset the start time by defining the offset seconds as a second parameter.

So, for example, change the line in step 6 to read the following:

[PRE18]

This would mean that the `mColor` animation would begin 0.5 seconds before `mRadius` has finished.

# Animating along a path

In this recipe, we will learn how to draw a smooth B-spline in the 3D space and animate the position of an object along the calculated B-spline.

## Getting ready

To navigate in the 3D space, we will use `MayaCamUI` covered in the *Using MayaCamUI* recipe in [Chapter 2](ch02.html "Chapter 2. Preparing for Development"), *Preparing for Development*.

## How to do it…

We will create an example animation of an object moving along the spline. Perform the following steps to do so:

1.  Include necessary header files.

    [PRE19]

2.  Begin with the declaration of member variables to keep the B-spline and current object's position.

    [PRE20]

3.  Inside the `setup` method prepare a random spline:

    [PRE21]

4.  Inside the `update` method, retrieve the position of the object moving along the spline.

    [PRE22]

5.  The code snippet drawing our scene will look like the following:

    [PRE23]

## How it works…

First, have a look at step 3 where we are calculating a B-spline through points with coordinates based on the sine and cosine functions and some random points on the x axis. The path is stored in the `spline` class member.

Then we can easily retrieve the position in 3D space at any distance of our path. We are doing this in step 4; using the `getPosition` method on the `spline` member. The distance on the path is been passed as a `float` value in the range of 0.0 to 1.0 where 0.0 means the beginning of the path and 1.0 means the end.

Finally, in step 5 we are drawing an animation as a red sphere traveling along our path, represented as a black dashed line, as shown in the following screenshot:

![How it works…](img/8703OS_09_01.jpg)

## See also

*   The *Aligning camera motion to path* recipe
*   The *Animating text around curves* recipe in [Chapter 7](ch07.html "Chapter 7. Using 2D Graphics"), *Using 2D Graphics*

# Aligning camera motion to a path

In this recipe we will learn how we can animate the camera position on our path, calculated as a B-spline.

## Getting ready

In this example, we will use `MayaCamUI`, so please refer to the *Using MayaCamUI* recipe in [Chapter 2](ch02.html "Chapter 2. Preparing for Development"), *Preparing for Development*.

## How to do it…

We will create an application illustrating the mechanism. Perform the following steps to do so:

1.  Include necessary header files.

    [PRE24]

2.  Begin with the declaration of member variables.

    [PRE25]

3.  Set up the initial values of members.

    [PRE26]

4.  Inside the `update` method update the camera properties.

    [PRE27]

5.  The whole `draw` method now looks like the following code snippet:

    [PRE28]

6.  Now we have to implement the `drawScene` method, which actually draws our 3D scene.

    [PRE29]

7.  The last thing we need is the `drawGrid` method, the implementation of which can be found in the *Using 3D space guides* recipe in [Chapter 2](ch02.html "Chapter 2. Preparing for Development"), *Preparing for Development*.

## How it works…

In this example we are using a B-spline as a path that our camera is moving along. Please refer to the *Animating along a path* recipe to see the basic implementation of an object animating on a path. As you can see in step 4 we are setting the camera position by invoking the `setEyePosition` method on the `mMovingCam` member, and we have to set the camera view direction. To do that we are taking the position of the next point on the path and passing it to the `lookAt` method.

We are drawing a split screen, where on the left-hand side is a preview of our scene, and on the right-hand side we can see what is in a frustum of the camera moving along the path.

![How it works…](img/8703OS_09_02.jpg)

## See also

*   The *Animating along a path* recipe
*   The *Using 3D space guides* recipe in [Chapter 2](ch02.html "Chapter 2. Preparing for Development"), *Preparing for Development*
*   The *Using MayaCamUI* recipe in [Chapter 2](ch02.html "Chapter 2. Preparing for Development"), *Preparing for Development*

# Animating text – text as a mask for a movie

In this recipe, we will learn how we can use text as a mask for a movie using a simple shader program.

## Getting ready

In this example, we are using one of the amazing videos provided by NASA taken by an ISS crew that you can find at [http://eol.jsc.nasa.gov/](http://eol.jsc.nasa.gov/). Please download oneand save it as `video.mov` inside the `assets` folder.

## How to do it…

We will create a sample Cinder application to illustrate the mechanism. Perform the following steps to do so:

1.  Include the necessary header files.

    [PRE30]

2.  Declare the member variables.

    [PRE31]

3.  Implement the `setup` method, as follows:

    [PRE32]

4.  Inside the `update` method we have to update our `mFrameTexture` where we are keeping the current movie frame.

    [PRE33]

5.  The `draw` method will look like the following code snippet:

    [PRE34]

6.  As you can see in the `setup` method we are loading a shader to do the masking. We have to pass through vertex shader inside the `assets` folder in a file named `passThru_vert.glsl`. You can find this in the *Implementing 2D metaballs* recipe in [Chapter 7](ch07.html "Chapter 7. Using 2D Graphics"), *Using 2D Graphics*.
7.  Finally, the fragment shader program code will look like the following code snippet, and should also be inside the `assets` folder under the name `masking_frag.glsl`.

    [PRE35]

## How it works…

Inside the `setup` method in step 3 we are rendering our text as `Surface` and then converting it to `gl::Texture` that we will use later as a masking texture. It is important here to set a rectangle format for masking texture while we are using it as a mask for a movie, because `qtime::MovieGl` is creating a texture with a frame that is rectangular. To do that we are defining the `gl::Texture::Format` object named `format` and invoking the `setTargetRect` method on it. While creating `gl::Texture` we have to pass `format` to the constructor as a second parameter.

To draw a movie frame we are using our masking shader program applied on the rectangle in step 5\. We have to pass three parameters, which are the movie frame as `sourceTexture`, mask texture with text as `maskTexture`, and the position of the mask as `maskOffset`.

In step 7 you can see the fragment shader code, which simply multiplies the colors of the corresponding pixels from `sourceTexture` and `maskTexture`. Please notice that we are using `sampler2DRect` and `texture2DRect` to handle rectangular textures.

![How it works…](img/8703OS_09_03.jpg)

# Animating text – scrolling text lines

In this recipe we will learn how we can create text scrolling line-by-line.

## How to do it…

We will now create an animation with scrolling text. Perform the following steps to do so:

1.  Include the necessary header files.

    [PRE36]

2.  Add the member values.

    [PRE37]

3.  Inside the `setup` method we need to generate textures for each line of text.

    [PRE38]

4.  The `draw` method for this example looks as follows:

    [PRE39]

## How it works…

What we are doing first inside the `setup` method in step 3 is generating a texture with rendered text for each line and pushing it to the vector structure `mTextTextures`.

In step 4 you can find the code for drawing current and previous text to build a continuous looped animation.

![How it works…](img/8703OS_09_04.jpg)

# Creating a flow field with Perlin noise

In this recipe we will learn how we can animate objects using a flow field. Our flow field will be a two-dimensional grid of velocity vectors that will influence how objects move.

We will also animate the flow field using vectors calculated with Perlin noise.

## Getting ready

Include the necessary files to work with OpenGL graphics, Perlin noise, random numbers, and Cinder's math utilities.

[PRE40]

Also, add the following useful `using` statements:

[PRE41]

## How to do it…

We will create an animation using the flow field. Perform the following steps to do so:

1.  We will begin by creating a `Follower` class to define the objects that will be influenced by the flow field.

    Declare the following class before the main application class:

    [PRE42]

2.  Let's create the flow field. Declare a two-dimensional `std::vector` to define the flow field, and variables to define the gap between vectors and the number of rows and columns.

    [PRE43]

3.  In the `setup` method we will define the number of rows and columns, and calculate the gap between each vector.

    [PRE44]

4.  Based on the number of rows and columns we can initialize `mFlowField`.

    [PRE45]

5.  Let's animate the flow field using Perlin noise. To do so declare the following members:

    [PRE46]

6.  In the `setup` method initialize `mCounter` to zero.

    [PRE47]

7.  In the `update` method we will increment `mCounter` and iterate `mFlowField` using a nested `for` loop, and use `mPerlin` to animate the vectors.

    [PRE48]

8.  Now iterate over `mFlowField` and draw a line indicating the direction of the vectors.

    Add the following code snippet inside the `draw` method:

    [PRE49]

9.  Let's add some `Followers`. Declare the following member:

    [PRE50]

10.  In the `setup` method we will initialize some followers and add them at random positions in the window.

    [PRE51]

11.  In the update we will iterate `mFollowers` and calculate the corresponding vector in `mFlowField` based on its position.

    We will then update the `Follower` class using that vector.

    [PRE52]

12.  Finally, we just need to draw each `Follower` class.

    Add the following code snippet inside the `draw` method:

    [PRE53]

    The following is the result:

![How to do it…](img/8703OS_09_05.jpg)

## How it works…

The `Follower` class represents an agent that will follow the flow field. In the `Follower::update` method a new velocity vector is passed as a parameter. The `follower` object will interpolate its velocity into the passed value and use it to animate. The `Follower::update` method is also responsible for keeping each agent inside the window by warping its position whenever it is outside the window.

In step 11 we calculated the vector in the flow field that will influence the `Follower` object using it's position.

# Creating an image gallery in 3D

In this recipe we will learn how we can create an image gallery in 3D. The images will be loaded from a folder selected by the user and displayed in a three-dimensional circular fashion. Using the keyboard, the user will be able to change the selected image.

## Getting ready

When starting the application you will be asked to select a folder with images, so make sure you have one.

Also, in your code include the necessary files to use OpenGL drawing calls, textures, the timeline, and loading images.

[PRE54]

Also, add the following useful `using` statements:

[PRE55]

## How to do it…

We will display and animate images in 3D space. Perform the following steps to do so:

1.  We will start by creating an `Image` class. Add the following code snippet before the main application class:

    [PRE56]

2.  In the main application's class we will declare the following members:

    [PRE57]

3.  In the `setup` method we will ask the user to select a folder and then try to create a texture from each file in the folder. If a texture is successfully created, we will use it to create an `Image` object and add it to `mImages`.

    [PRE58]

4.  We need to iterate over `mImages` and define the angle and distance that each image will have from the center.

    [PRE59]

5.  Now we can initialize the remaining members.

    [PRE60]

6.  In the `draw` method, we will start by clearing the window, setting the window's matrices to support 3D, and enabling reading and writing in the depth buffer:

    [PRE61]

7.  Next we will draw the images. Since all our images have been displayed around the origin, we must translate them to the center of the window. We will also rotate them around the y axis using the value in `mRotationOffset`. Everything will go in an `if` statement that will check if `mImages` contains any image, in case no image was generated during the setup.
8.  Add the following code snippet inside the `draw` method:

    [PRE62]

9.  Since the user will be able to switch images using the keyboard, we must declare the `keyUp` event handler.

    [PRE63]

10.  In the implementation of `keyUp` we will move the images on to the left or right-hand side depending on whether the left or right key was released.

    If the selected image was changed, we animate `mRotationOffset` to the correspondent value, so that the correct image is now facing the user.

    Add the following code snippet inside the `keyUp` method:

    [PRE64]

11.  Build and run the application. You will be prompted to select a folder containing images that will then be displayed in a circular fashion. Press the left or right key on the keyboard to change the selected image.![How to do it…](img/8703OS_09_06.jpg)

## How it works…

The `draw` method of the `Image` class rotates the coordinate system around the y axis and then translates the image drawing on the z axis. This will extrude the image from the center facing outwards on the given angle. It is an easy and convenient way of achieving the desired effect without dealing with coordinate transformations.

The `Image::rect` member is used to draw the texture and is calculated to fit inside the rectangle passed in the constructor.

When selecting the image to be displayed in front, the value of `mRotationOffset` will be the opposite of the image's angle, making it the image being drawn in front of the view.

In the `keyUp` event we check whether the left or right key was pressed and animate `mRotationOffset` to the desired value. We also take into account if the angle wraps around, as to avoid glitches in the animation.

# Creating a spherical flow field with Perlin noise

In this recipe we will learn how to use Perlin noise with a spherical flow field and animate objects in an organic way around a sphere.

We will animate our objects using spherical coordinates and then transform them into Cartesian coordinates in order to draw them.

## Getting ready

Add the necessary files to use Perlin noise and draw with OpenGL:

[PRE65]

Add the following useful `using` statements:

[PRE66]

## How to do it…

We will create the `Follower` objects that move organically in a spherical flow field. Perform the following steps to do so:

1.  We will start by creating a `Follower` class representing an object that will follow the spherical flow field.

    Add the following code snippet before the application's class declaration:

    [PRE67]

2.  We will be using spherical to Cartesian coordinates, so declare the following method in the application's class:

    [PRE68]

3.  The implementation of this method is as follows:

    [PRE69]

4.  Declare the following members in the application's class:

    [PRE70]

5.  In the `setup` method we will begin by initializing `mRadius` and `mCounter`:

    [PRE71]

6.  Now let's create 100 followers and add them to `mFollowers`. We will also attribute random values to the `phi` and `theta` variables of the `Follower` objects and set their initial positions:

    [PRE72]

7.  In the `update` method we will animate our objects. Let's start by incrementing `mCounter`.

    [PRE73]

8.  Now we will iterate over all the objects in `mFollowers` and use Perlin noise, based on the follower's position, to calculate how much it should move on spherical coordinates. We will then calculate the correspondent Cartesian coordinates and move the object.

    Add the following code snippet inside the `update` method:

    [PRE74]

9.  Let's move to the `draw` method and begin by clearing the background, setting the windows matrices, and enabling reading and writing in the depth buffer.

    [PRE75]

10.  Since the followers are moving around the origin, we will draw them translated to the origin using a dark gray color. We will also draw a white sphere to get a better understanding of the movement.

    [PRE76]

## How it works...

We use Perlin noise to calculate the change in the `theta` and `phi` members of the `Follower` objects. We use these, together with `mRadius`, to calculate the position of the objects using the standard spherical to Cartesian coordinate transformation. Since Perlin noise gives coherent values based on coordinates by using the current position of the `Follower` objects, we get the equivalent of a flow field. The `mCounter` variable is used to animate the flow field in the third dimension.

![How it works...](img/8703OS_09_07.jpg)

## See also

*   To learn more about the Cartesian coordinate system, please refer to [http://en.wikipedia.org/wiki/Cartesian_coordinate_system](http://en.wikipedia.org/wiki/Cartesian_coordinate_system)
*   To learn more about the spherical coordinate system, please refer to [http://en.wikipedia.org/wiki/Spherical_coordinate_system](http://en.wikipedia.org/wiki/Spherical_coordinate_system)
*   To learn more about spherical to Cartesian coordinate transformations, please refer to [http://en.wikipedia.org/wiki/List_of_common_coordinate_transformations#From_spherical_coordinate](http://en.wikipedia.org/wiki/List_of_common_coordinate_transformations#From_spherical_coordinate)