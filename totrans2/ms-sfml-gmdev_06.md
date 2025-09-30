# Chapter 6. Adding Some Finishing Touches - Using Shaders

Having good art is important for any game, as it greatly compliments the content game designers bring to the table. However, simply tacking on any and all graphics to some logic and calling it a day just does not cut it anymore. Good visual aesthetics of a game are now formed by hand-in-hand cooperation of amazing art and the proper post-processing that follows. Dealing with graphics as if they are paper cut-outs feels dated, while incorporating them in the dynamic universe of your game world and making sure they react to their surroundings by properly shading them has become the new standard. For a brief moment, let us put aside gameplay and discuss the technique of that special kind of post-processing, known as shading.

In this chapter, we are going to be covering:

*   The basics of the SFML shader class
*   Implementing a unified way of drawing objects
*   Adding a day-night cycle to the game

Let us get started with giving our project that extra graphical enhancement!

# Understanding shaders

In the modern world of computer graphics, many different calculations are offloaded to the GPU. Anything from simple pixel colour calculations, to complex lighting effects can and should be handled by hardware that is specifically designed for this purpose. This is where shaders come in.

A **shader** is a little program that runs on your graphics card instead of the CPU, and controls how each pixel of a shape is rendered. The main purpose of a shader, as the name suggests, is performing lighting and shading calculations, but they can be used for much more than that. Because of the power modern GPUs have, libraries exist that are designed to perform calculations on the GPU that would usually be executed on the CPU, in order to cut down the computation time significantly. Anything from physics computations to cracking password hashes can be done on the GPU, and the entry point to that horsepower is a shader.

### Tip

GPUs are good at performing tons of very specific calculations in parallel at once. Using less predictable or unparallel algorithms is very inefficient on the GPU, which is what the CPU excels at. However, as long as the data can be processed exactly the same in parallel, the task is deemed worthy of being pushed to the GPU for further handling.

There are two main types of shader that SFML provides: **vertex** and **fragment**. Newer versions of SFML (*2.4.0* and up) have added support for geometry shaders as well, but it is not necessary to cover this for our purposes.

A **vertex shader** is executed once per vertex. This process is commonly referred to as **per-vertex** shading. For example, any given triangle has three vertices. This means that the shader would be executed once for each vertex for a grand total of three times.

A **fragment shader** is executed once per pixel (otherwise known as a fragment), which results in the process being referred to as **per-pixel** shading. This is much more taxing than simply performing per-vertex calculations, but is much more accurate and generally produces better visual results.

Both types of shader can be used at once on a single piece of geometry being drawn, and can also communicate with each other.

## Shader examples

The **OpenGL Shading Language** (**GLSL**) is extremely similar to *C* or *C++*. It even uses the same basic syntax, as seen in this vertex shader example:

[PRE0]

Notice the version on the very first line. The number `450` indicates the version of OpenGL that should be used, in this case *4.5*. Newer versions of SFML support OpenGL versions *3.3+*; however the success of running it also depends on the capabilities of your graphics card.

For now, simply ignore the first line of the `main` function. It has to do with positions transformations from one coordinate system to another, and is specific to a few possible approaches to shading. These concepts will be covered in the next chapter.

GLSL provides quite a few *hooks* that allow direct control of vertex and pixel information, such as `gl_Position` and `gl_Color`. The former is simply the position of the vertex that will be used in further calculations down the line, while the latter is the vertex colour, which is being assigned to `gl_FrontColor`, ensuring the colour is passed down the pipeline to the fragment shader.

Speaking of the fragment shader, here is a very minimal example of what it may look like:

[PRE1]

In this particular case, `gl_FragColor` is used to set a static value of the pixel being rendered. Any shape being rendered while using this shader will come out white.

### Note

The values of this vector are normalized, meaning they have to fall in the range of *0.f < n <= 1.0f*.

Keep in mind that `gl_Color` can be used here to sample the colour that is passed down from the vertex shader. However, because there may be multiple pixels in between vertices, the colour for each fragment is interpolated. In a case where each vertex of a triangle is set to colours red, green, and blue, the interpolated result would look like this:

![Shader examples](img/image_06_001.jpg)

One last thing to note about any shader is that they support communication from outside sources. This is done by using the `uniform` keyword, followed by the variable type and capped off by its name like so:

[PRE2]

In this particular example, outside code passes in three `float` that will be used as color values for the fragment. Uniforms are simply **global** variables that can be manipulated by outside code before a shader is used.

## SFML and shaders

Storing and using shaders in SFML is made simple by introducing the `sf::Shader` class. Although shaders are generally supported by most devices out there, it is still a good idea to perform a check that determines if the system the code is being executed on supports shaders as well:

[PRE3]

This shader class can hold either one of the two types of shader just by itself or a single instance of each type at the same time. Shaders can be loaded in one of two ways. The first is by simply reading a text file:

[PRE4]

### Note

File extensions of these shaders do not have to match the preceding ones. Because we are working with text files, the extension simply exists for clarity.

The second way to load a shader is by parsing a string loaded in memory:

[PRE5]

[PRE6]

Using a shader is fairly straightforward as well. Its address simply needs to be passed in to a render targets `draw()` call as the second argument when something is being rendered to it:

[PRE7]

Since our shaders may need to be communicated with through `uniform` variables, there has to be a way to set them. Enter `sf::Shader::setUniform(...)`:

[PRE8]

This simple bit of code manipulates the `r` uniform inside whatever shader(s) happen to be loaded inside the `shader` instance. The method itself supports many more types besides *float*, which we will be covering in the next chapter.

# Localizing rendering

Shading is a powerful concept. The only problem with injecting a stream of extra-graphical-fanciness to our game at this point is the fact that it is simply not architected to deal with using shaders efficiently. Most, if not all of our classes that do any kind of drawing do so by having direct access to the `sf::RenderWindow` class, which means they would have to pass in their own shader instances as arguments. This is not efficient, re-usable, or flexible at all. A better approach, such as a separate class dedicated to rendering, is a necessity.

In order to be able to switch from shader to shader with relative ease, we must work on storing them properly within the class:

[PRE9]

Because the `sf::Shader` class is a non-copyable object (inherits from `sf::NonCopyable`), it is stored as a unique pointer, resulting in avoidance of any and all move semantics. This list of shaders is directly owned by the class that is going to do all of the rendering, so let us take a look at its definition:

[PRE10]

Since shaders need to be passed in as arguments to the window `draw()` calls, it is obviously imperative for the renderer to have access to the `Window` class. In addition to that and the list of shaders that can be used at any given time, we also keep a pointer to the current shader being used in order to cut down on container access time, as well as a couple of flags that will be used when choosing the right shader to use, or determining whether the drawing is currently happening in the first place. Lastly, a fairly useful debug feature is having information about how many draw calls happen during each update. For this, a simple *unsigned integer* is going to be used.

The class itself provides the basic features of enabling/disabling additive blending instead of a regular shader, switching between all available shaders, and disabling the current shader, as well as obtaining it. The `BeginDrawing()` and `EndDrawing()` methods are going to be used by the `Window` class in order to provide us with *hooks* for obtaining information about the rendering process. Note the overloaded `Draw()` method. It is designed to take in any drawable type and draw it on either the current window, or the appropriate render target that can be provided as the second argument.

Finally, the `LoadShaders()` private method is going to be used during the initialization stage of the class. It holds all of the logic necessary to load every single shader inside the appropriate directory, and store them for later use.

## Implementing the renderer

Let us begin by quickly going over the construction of the `Renderer` object, and the initialization of all of its data members:

[PRE11]

Once the pointer to the `Window*` instance is safely stored, all of the data members of this class are initialized to their default values. The body of the constructor simply consists of a private method call, responsible for actually loading and storing all of the shader files:

[PRE12]

We begin by establishing a local variable that is going to hold the path to our `shader` directory. It is then used to obtain two lists of files with `.vert` and `.frag` extensions respectively. These will be the vertex and fragment shaders to be loaded. The goal here is to group vertex and fragment shaders with identical names, and assign them to a single instance of `sf::Shader`. Any shaders that do not have a vertex or fragment counterpart will simply be loaded alone in a separate instance.

Vertex shaders are as good a place as any to begin. After the filename is obtained and stripped of its extension, a fragment shader with the same name is attempted to be located. At the same time, a new `sf::Shader` instance is inserted into the shader container, and a reference to it is obtained. If a fragment counterpart has been found, both files are loaded into the shader. The fragment shader name is then removed from the list, as it will no longer need to be loaded in on its own.

As the first part of the code does all of the pairing, all that is really left to do at this point is load the fragment shaders. It is safe to assume that anything on the fragment shader list is a standalone fragment shader, not associated with a vertex counterpart.

Since shaders can have uniform variables that need to be initialized, it is important that outside classes have access to the shaders they use:

[PRE13]

If the shader with the provided name has not been located, `nullptr` is returned. On the other hand, a raw pointer to the `sf::Shader*` instance is obtained from the smart pointer and returned instead.

The same outside classes need to be able to instruct the `Renderer` when a specific shader should be used. For this purpose, the `UseShader()` method comes in handy:

[PRE14]

Since the `GetShader()` method already does the error-checking for us, it is used here as well. The value returned from it is stored as the pointer to the current shader, if any, and is then evaluated in order to return a *boolean* value, signifying success/failure.

The actual drawing of geometry is what we are all about here, so let us take a look at the overloaded `Draw()` method:

[PRE15]

Whether a `sf::Sprite` or `sf::Shape` is being rendered, the actual idea behind this is exactly the same. First, we check if the intention behind the method call was indeed to render to the main window by looking at the `l_target` argument. If so, a fair thing to do here is to make sure the drawable object actually is on screen. It would be pointless to draw it if it was not. Provided the test passes, the main `Draw()` method overload is invoked, with the current arguments being passed down:

[PRE16]

This is where all of the actual magic happens. The `l_target` argument is again checked for being equal to `nullptr`. If it is, the render window is stored inside the argument pointer. Whatever the target, at this point its `Draw()` method is invoked, with the drawable being passed in as the first argument, as well as the appropriate shader or blend mode passed in as the second. The additive blending obviously takes precedence here, enabling a quicker way of switching between using a shader and the additive blending modes by simply needing to use the `AdditiveBlend()` method.

Once the drawing is done, the `m_drawCalls` data member is incremented, so that we can keep track of how many drawables have been rendered in total at the end of each cycle.

Finally, we can wrap this class up by looking at a couple of essential yet basic setter/getter code:

[PRE17]

As you can see, disabling the use of shaders for whatever is being drawn currently is as simple as setting the `m_currentShader` data member to `nullptr`. Also note the `BeginDrawing()` method. It conveniently resets the `m_drawCalls` counter, which makes it easier to manage.

# Integrating the Renderer class

There is obviously no point in even having the `Renderer` class, if it is not going to be in its proper place or used at all. Since its only job is to draw things on screen with the correct effect being applied, a fitting place for it would be inside the `Window` class:

[PRE18]

Because outside classes rely on it as well, it is a good idea to provide a getter method for easy retrieval of this object.

Actually integrating it into the rest of the code is surprisingly easy. A good place to start is giving the `Renderer` access to the `Window` class like so:

[PRE19]

The renderer also has *hooks* for knowing when we begin and end the drawing process. Luckily, the `Window` class already supports this idea, so it's really easy to tap into it:

[PRE20]

Finally, in order to make use of the newest versions of OpenGL, the window needs to be instructed to create a version of the newest context available:

[PRE21]

Note the shader loading bit at the end of this code snippet. The `Renderer` class is instructed to load the shaders available in the designated directory, provided shaders are being used in the first place. These several simple additions conclude the integration of the `Renderer` class.

## Adapting existing classes

Up until this point, rendering something on screen was as simple as passing it as a drawable object to a `Draw()` method of a `Window` class. While great for smaller projects, this is problematic for us, simply because that heavily handicaps any use of shaders. A good way to upgrade from there is to simply take in `Window` pointers:

[PRE22]

Let us go over each of these classes and see what needs to be changed in order to add proper support for shaders.

### Updating the ParticleSystem

Going all the way back to [Chapter 3](ch03.html "Chapter 3.  Make It Rain! - Building a Particle System") , *Make it rain! - Building a particle system* we have already used a certain amount of shading trickery without even knowing it! The additive blending used for fire effects is a nice feature, and in order to preserve it without having to write a separate shader for it, we can simply use the `AdditiveBlend()` method of the `Renderer` class:

[PRE23]

First, note the check of the current application state. For now, we do not really need to use shaders inside any other states besides `Game` or `MapEditor`. Provided we are in one of them, the default shader is used. Otherwise, shading is disabled.

When dealing with actual particles, the `AdditiveBlend()` method is invoked with the blend mode flag being passed in as its argument, either enabling or disabling it. The particle drawable is then drawn on screen. After all of them have been processed, additive blending is turned off.

### Updating entity and map rendering

The default shader is not only used when rendering particles. As it happens, we want to be able to apply unified shading, at least to some extent, to all world objects. Let us begin with entities:

[PRE24]

The only real changes to the rendering system are the invocation of the `UseShader()` method, and the fact that a pointer to the `Window` class is being passed down to the sprite-sheets `Draw()` call as an argument, instead of the usual `sf::RenderWindow`. The `SpriteSheet` class, in turn, is also modified to use the `Renderer` class, even though it does not actually interact with or modify shaders at all.

The game map should be shaded in exactly the same way as well:

[PRE25]

The only real difference here is the fact that the `Map` class already has access to the `Window` class internally, so it does not have to be passed in as an argument.

# Creating a day/night cycle

Unifying the shading across many different world objects in our game gave us a very nice way of manipulating how the overall scene is actually represented. Many interesting effects are now possible, but we are going to focus on a rather simple yet effective one-lighting. The actual subtleties of the lighting subject will be covered in later chapters, but what we can do now is build a system that allows us to shade the world differently, based on the current time of the day, like so:

![Creating a day/night cycle](img/image_06_002.jpg)

As you can tell, this effect can add a lot to a game and make it feel very dynamic. Let us take a look at how it can be implemented.

## Updating the Map class

In order to accurately represent a day/night cycle, the game must keep a clock. Because it is relative to the world, the best place to keep track of this information is the `Map` class:

[PRE26]

For the sake of having dynamic and customizable code, two additional data members are stored: the current game time, and the overall length of the day. The latter allows the user to potentially create maps with a variable length of a day, which could offer some interesting opportunities for game designers.

Using these values is fairly simple:

[PRE27]

The actual game time is first manipulated by adding the frame time to it. It is then checked for having exceeded the boundaries of twice the value of the length of a day, in which case the game time is set back to `0.f`. This relationship represents a 1:1 proportion between the length of a day and the length of a night.

Finally, ensuring the light properly fades between day and night, a `timeNormal` local variable is established, and used to calculate the amount of darkness that should be cast over the scene. It is then checked for having exceeded the value of `1.f`, in which case it is adjusted to start moving back down, representing the fade from darkness to dawn. The value is then passed to the default shader.

### Note

It is important to remember that shaders work with normalized values most of the time. This is why we are striving to provide it with a value between `0.f` to `1.f`.

The final piece of the puzzle is actually initializing our two additional data members to their default values:

[PRE28]

As you can see, we have given the day length a value of `30.f`, which means the full day/night cycle will last a minute. This is obviously not going to be very useful for a game, but can come in handy when testing the shaders.

## Writing the shaders

With all of the *C++* code out of the way, we can finally focus on GLSL. Let us begin by implementing the default vertex shader:

[PRE29]

This is nothing different from the examples used during the introduction stage of this chapter. The purpose of adding the vertex shader now is simply to avoid having to write it again later, when something needs to be done in it. With that said, let us move on to the fragment shader:

[PRE30]

### Note

The `sampler2D` type in this instance is simply the texture being passed into the shader by SFML. Other textures may also be passed into the shader manually, by using the `shader.setUniform("texture", &texture);` call.

In order to properly draw a pixel, the fragment shader needs to sample the texture of the current object being drawn. If a simple shape is being drawn, the pixel being sampled from the texture is checked for being completely black. If that's the case, it's simply set to a white pixel. In addition to that, we also need the `timeNormal` value discussed earlier. After the current pixel of the texture has been sampled, it is multiplied by the colour passed in from the vertex shader and stored as `gl_FragColor`. The `timeNormal` value is then subtracted from all three colour channels. Finally, a slight tint of blue is added to the pixel in the end. This gives our scene a blue tint, and is purely an aesthetic choice.

# Summary

Many argue that graphics should be a secondary concern for a game developer. While it is clear that the visual side of a project should not be its primary concern, the visuals can serve a player more than simply acting as pretty backdrops. Graphical enhancements can even help tell a story better by making the player feel more engrossed in the environment, using clever visual cues, or simply controlling the overall mood and atmosphere. In this chapter, we have taken one of our first steps towards building a system that will serve as a massive helper when conquering the world of special effects.

In the next chapter, we will be delving deeper into the lower levels of graphical enhancements. See you there!