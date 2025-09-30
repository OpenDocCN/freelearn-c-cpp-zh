# Chapter 8.  Let There Be Light - An Introduction to Advanced Lighting

There is a certain standard expected of a game in this day and age. As technology progresses and the number of transistors in any given computational unit increases, there is more and more power at our disposal to do what was previously unheard of. One thing that definitely makes use of all of this extra horse power is dynamic lighting. Because of its stunning visual results, it has become an integral part of most video games, and is now one of the core technologies that are expected to come with them.

In this chapter, we're going to cover the following topics:

*   Using the technique of deferred rendering/shading
*   Implementing a multi-pass lighting shader
*   Faking geometry complexity using normal maps
*   Using specular maps to create shiny surfaces
*   Using height maps to make lighting feel more three-dimensional

Let's start shedding some light on this subject!

# Using third-party software

None of the material maps used in this chapter are hand-drawn. For the generation of certain material maps, *Crazybump* was used, which can be found at [http://crazybump.com/](http://crazybump.com/).

There are other free alternatives that can be found online.

# Deferred rendering

Deferred rendering/shading is a technique that gives us greater control over how certain effects are applied to a scene by not enabling them during the first pass. Instead, the scene can be rendered to an off screen buffer, along with other buffers that hold other material types of the same image, and then drawn on the screen in a later pass, after the effects have been applied, potentially in multiple passes as well. Using this approach allows us to separate and compartmentalize certain logic that would otherwise be entangled with our main rendering code. It also gives us an opportunity to apply as many effects to the final image as we want. Let's see what it takes to implement this technique.

## Modifying the renderer

In order to support all the fancy new techniques we're about to utilize, we need to make some changes to our renderer. It should be able to keep a buffer texture and render to it in multiple passes in order to create the lighting we're looking for:

[PRE0]

For convenience, we have a couple of methods for toggling the deferred rendering process. Also, since rendering a scene to a texture is slightly different than rendering a texture to another texture, because of where the camera (view) is positioned, we will use the `BeginSceneRendering()` and `BeginTextureRendering()` methods to properly handle the task.

Note the use of two textures in this class as well as a pointer to point to the texture that is currently in use. The essence of a multi-pass approach is being able to sample the texture holding the information of the previous render pass while drawing to the current render target.

Lastly, we'll discuss three methods for clearing the current texture, the texture of a previous render pass, and both of these. The most recent render pass texture can then be rendered by calling the `DrawBufferTexture()` method.

### Implementing changes in the Renderer class

Let's start with something simple. Implement the deferred rendering toggle methods; they will help you keep track of the current rendering state:

[PRE1]

As you can see, it's as simple as flipping a flag. In the case of enabling deferred rendering, we also need to check whether the use of shaders is allowed.

Also, the textures we're using as buffers clearly need to be created:

[PRE2]

This particular method is invoked inside the constructor of `Renderer`.

Next, we have something equally as simple, yet quite a bit more important:

[PRE3]

By using these methods at the appropriate time, we can successfully draw shapes that would have world coordinates for a standalone texture buffer the size of a window. We can also simply draw information from another window-sized texture to the buffer.

Some helpful getter methods are always useful:

[PRE4]

While the first one simply returns a pointer to the current buffer texture being used, the second method does the exact opposite. It determines which texture is *not* the current buffer; once it identifies this, it returns a pointer to that object instead. Why exactly this is useful will become apparent shortly.

Clearing these textures is just as simple as one might think:

[PRE5]

In order to prepare for another render pass and display all the changes made to the first buffer, the textures must be swapped like so:

[PRE6]

Note the call to the texture's `display` method. Calling `display` is required because we want all of the changes made to the texture to be reflected. Without calling this method, our progress would not manifest.

Another key alteration to this class is making sure the buffer texture is being used while deferred rendering is enabled:

[PRE7]

After a couple of checks to make sure we're not overwriting an already provided render target and that the use of shaders is enabled, we select the buffer texture by overwriting the `l_target` pointer with its address.

Finally, the buffer texture that has all of the render pass information can be drawn on the screen like so:

[PRE8]

This simple, yet powerful, design provides us with the possibilities of implementing almost any postprocessing effect imaginable.

## A minimal example

One of the effects incidentally, the focus of this chapter is dynamic lighting. Before we go further and implement the more advanced features or delve into more complex concepts, let's walk through the process of using the newly implemented renderer features. Let's take one step at a time.

First, the scene should be drawn to the texture buffer as usual:

[PRE9]

As you can see, once deferred rendering is enabled, the default shader is used and the scene rendering process begins. For each layer, the map, entities, and particles are all drawn as usual. The only difference now is that the buffer texture is being used behind the scenes. Once everything is rendered, the textures are swapped; this allows the current back buffer texture to display all the changes:

[PRE10]

Once the scene is rendered, we enter what from now on is going to be referred to as the light pass. This special pass uses its own shader and is responsible for the illumination of the scene. It sets up what is known as *ambient light* as well as regular omnidirectional light.

### Note

**Ambient light** is a type of light that has no position. It illuminates any part of the scene evenly, regardless of the distance.

As illustrated in the preceding code, the point light first has its world coordinates converted into screen-space coordinates, which are then passed as a uniform to the shader. Then, the light color and radius are passed to the shader along with the texture of the previous pass, which, in this case, is simply the color (diffuse) map of the scene.

### Note

**Point light** is a type of light that emits light in all directions (omnidirectional) from a single point, creating a sphere of illumination.

### Tip

The screen-space coordinate system has its *Y* axis inversed from the world coordinate format, meaning the positive *Y* values go up, not down. This is the reason the light position's *Y* coordinate has to be adjusted before it is passed to the shader.

The next portion of the code is essentially meant to just trigger a full redraw of the diffuse texture onto the buffer texture. We're making a quad comprised of two triangular strips represented as `sf::VertexArray`. It's made to be the size of the entire window so that all the pixels could surely be redrawn. Once the quad is drawn, the textures are once again swapped to reflect all the changes:

[PRE11]

The last bit of this example simply turns off deferred rendering so that all render operations from now on are done to the window. The window view is then set to its default state, so that the buffer texture can be drawn onscreen easily. Finally, we reset the view back, shortly before whatever shader is still active is disabled.

### Shader code

We're almost done! The last, but definitely not the least, important piece of this puzzle is writing the lighting pass shader correctly in order to get proper results. Given what we already know about the light pass procedure in our C++ code, let's see what GLSL has to offer:

[PRE12]

As expected, we need to process the diffuse texture in this pass in order to preserve color information. The other uniform values consist of a 3D vector that represents the ambient color, two 3D vectors for the position and color of a regular source of light, and a floating point value for the radius of the same light.

The texture that was passed to the shader is sampled at the appropriate, interpolated texture coordinates and stored in the `pixel` variable. The distance between the pixel being processed and the light's center is then calculated using the Pythagorean variant distance formula:

![Shader code](img/B05590_08_01-1.jpg)

### Note

The `gl_FragCoord` parameter holds the pixel coordinates in the screen space. Its *Z* component is a depth value, which we're not going to use for the time being.

### Tip

The `pow` function simply returns a value that is raised to the power of its second argument.

After the distance is calculated, a check is made to determine whether the distance between the light and the pixel we're working with is within the light's radius. If it is, the color information of our pixel is multiplied by the light color and added to the final pixel that's going to be written. Otherwise, the color information is simply multiplied by the ambient color.

This fairly basic principle gives us, as one should expect, fairly basic and non-realistic lighting:

![Shader code](img/image_08_002.jpg)

Although it works, in reality, light is emitted in all directions. It also slowly loses its brightness. Let's see what it takes to make this happen in our game.

# Attenuating light

Light attenuation, also known as gradual loss in intensity, is what we're going to use when creating the effect of a light source that is slowly bleeding away. It essentially comes down to using yet another formula inside the light pass shader. There are many variations of attenuating light that work for different purposes. Let's take a look:

[PRE13]

Once again, we're dealing with the same uniform values being passed in, but with one additional value of `LightFalloff`. It's a factor between *0* and *1* that determines how fast a source of light would lose its brightness.

Inside the `main()` function, the diffused pixel is sampled as usual. This is done before we calculate a vector `L` that represents the position difference between the pixel and the light's center. This vector is then converted into distance using the `length` function. This is the same type of distance that we calculated manually in the first iteration of this shader. The floating number variable `d` is then used to calculate the distance between the fragment and the outside of the light by subtracting the light's radius from it. The `max()` function simply makes sure we don't get a negative value back if the pixel is inside the light's bubble.

The attenuation itself, as mentioned before, can have many variations. This particular variation visually works best for the type of game we're dealing with.

After the calculations are performed, the final output pixel is multiplied by the ambient light (which should only be done during the first pass if there are multiple light passes). Additionally, the light information is multiplied by the diffuse pixel and the attenuation factor is added to it. This last bit of multiplication ensures that, given the pixel is outside the effective light range, no additional light is added to it. The result of this is slightly more appealing to look at:

![Attenuating light](img/image_08_003.jpg)

At this point, a very good question you could ask is 'How on earth is this going to work with multiple light input?' Luckily, this is a bit simpler than one might think.

# Multi-pass shading

Much like *C/C++* code, GLSL does support the use of data arrays. Using them can seem like an obvious choice to just push information about multiple light streams into the shader and have it all done in one pass. Unlike *C++*, however, GLSL needs to know the sizes of these arrays at compile time, which is very much like *C*. At the time of writing, dynamic size arrays aren't supported. While this information can put a damper on a naive plan of handling multiple light sources with ease, there are still options to choose from, obviously.

One approach to combat this may be to have a very large, statically sized array of data. Only some of that data would be filled in and the shader would process it by looping over the array while using a uniform integer that tells it how many lights were actually passed to it. This idea comes with a few obvious bottlenecks. First, there would be a threshold for the maximum number of light streams allowed on the screen. The second issue is performance. Sending data over to the GPU is costly and can quickly become inefficient if we send over too much information all at once.

As flawed as the first idea is, it has one component that comes in handy when considering a better strategy: the maximum number of light streams allowed. Instead of pushing tons and tons of data through to the GPU at once, why not just do it a little bit at a time in different passes. If the right number of light streams is sent each time, both the CPU and GPU performance bottlenecks can be minimized. The results of each pass can then be blended together into a single texture.

## Modifying the light pass shader

There are a couple of challenges we need to overcome in order to correctly blend the buffer textures of multiple passes. First, there's loss of information due to ambient lighting. If the light is too dark, every subsequent pass becomes less and less visible. To fix this problem, in addition to the color information of the last render pass, we're going to need access to the actual diffuse map.

The second issue is choosing the right number of light streams per shader pass. This can be benchmarked or simply gotten right through trial and error. For our purposes, we'll go with 3-4 light streams per pass. Let's take a look at how the light shader can be modified to achieve this:

[PRE14]

First, note the new `sampler2D` uniform type being passed in for the diffuse map. This is going to be invaluable in order to avoid light colors from being washed out with additional passes. The other two bits of additional information we're going to need are values that determine the number of light streams that have been sent to the shader for the current pass and the pass we're dealing with at the moment.

The actual light information is now neatly stored away in a `struct` that holds the usual data we expect. Underneath it, we need to declare a constant integer for the number of maximum light streams per shader pass and the uniform array that's going to be filled in by our C++ code for the light information.

Let's see the changes that the body of the shader needs to undergo in order to support this:

[PRE15]

First, we need to sample the diffuse pixel as well as the pixel from the previous shader pass. The `finalPixel` variable is established early on and uses the information from the previous shader pass. It is important you note this, because the previous pass would be lost otherwise. Since we have access to the pass number in the shader now, we can selectively apply the ambient light to the pixel only during the first pass.

We can then jump into a `for` loop that uses the `LightCount` uniform passed in from the C++ side. This design gives us control to only use as much data as was sent to the shader and not go overboard if the last shader pass has fewer light streams than the maximum number allowed.

Finally, let's see what needs to change when it comes to the actual shading of the fragment. All our calculations remain the same, except for using light data. The lights uniform is now accessed with the square brackets to fetch the correct information during each iteration of the loop. Note the final pixel calculation at the very bottom of the loop. It now uses the diffuse pixel instead of the pixel of a previous shader pass.

## Changes in the C++ code

None of the fanciness in the GLSL we've just finished is complete without appropriate support from our actual code base. First, let's start with something simple and conveniently represent a light stream in a proper `struct`:

[PRE16]

That's better! Now let's begin passing in all of the additional information to the shader itself:

[PRE17]

Right after we're done with drawing onto the diffuse texture, it's copied over and stored in a separate buffer. It's then flipped along the *Y* axis, as the copying process inverts it.

### Tip

The copying and flipping of the texture here is a proof of concept. It shouldn't be performed in production code, as it's highly inefficient.

At this point, we're ready to begin the light pass. Just before we start this, ensure that a couple of light streams are added to `std::vector` and are waiting to be passed in. Also, declare a constant at the very bottom that denotes how many light streams are supposed to be passed to a shader every time. This number has to match the constant inside the shader.

Let's begin with the actual light pass and see what it involves:

[PRE18]

Ambient lighting is first set up, as it's not going to change between the iterations. In this case, we're giving it a slight blue tint. Additionally, a couple of local variables for the iteration and pass are created in order to have this information handy.

As we're iterating over each light stream, a string called `id` is created with the integer of each iteration passed inside. This is meant to represent the array access analysis of the light streams' uniform inside the shader, and it will serve as a helpful way of allowing us to access and overwrite that data. The light information is then passed in using the `id` string with an attached dot operator and the name of the `struct` data member. The light's identifier `i` is incremented shortly after. At this point, we need to decide whether the required number of light streams have been processed in order to invoke the shader. If the last light stream for the pass has been added, or if we're dealing with the last light stream of the scene, the rest of the uniforms are initialized and the fullscreen `sf::VertexArray` quad we talked about earlier is drawn, invoking a shader for each visible pixel. This effectively gives us a result like this:

![Changes in the C++ code](img/image_08_004.jpg)

Now we're getting somewhere! The only downside to this is all of the mess we have to deal with in our C++ code, as none of this data is managed properly. Let's fix this now!

# Managing light input

Good data organization is important in every aspect of software design. It's hard to imagine an application that would run quickly and efficiently, yet wants have a strong, powerful, and flexible framework running in the backend. Our situation up until this point has been fairly manageable, but imagine you want to draw additional textures for the map, entities, and all your particles. This would quickly become tiresome to deal with and maintain. It's time to utilize our engineering ingenuity and come up with a better system.

## Interface for light users

First and foremost, each class that desires to use our lighting engine would need to implement their own version of drawing certain types of textures to the buffer(s). For diffuse maps, we already have the plain old regular `Draw` calls, but even if they are all lucky enough to have the same signature, that's not good enough. A common interface for these classes is needed in order to make them a successful part of the lighting family:

[PRE19]

The `LightUser` class forces any derivatives to implement a special `Draw` method that uses a material container. It also has access to the `Window` class and knows which layer it's trying to draw to. 'What's a material container?' you may ask? Let's find out by taking this design further.

## The light manager class

Before we design a grand class that would take care of all our lighting needs, let's talk about materials. As it so happens, we've already dealt with one type of material: the diffuse map. There are many other possible materials we're going to work with, so let's not beat around the bush any longer and see what they are:

[PRE20]

In addition to diffuse maps, we're going to build *height*, *normal*, and *specular* maps as well. None of these terms will probably make sense right now, but that's alright. Each one will be explained in detail as we cross that bridge.

The material map container type is simply a map that links a type to a `sf::RenderTexture`. This way, we can have a separate texture for each material type.

For the light manager, we're only going to need two type definitions:

[PRE21]

As you can see, they're extremely simple. We're going to store the light streams themselves along with pointers to the light user classes in vectors, as nothing fancier is necessary here. With that, let's take a look at the actual definition of the `LightManager` class:

[PRE22]

As you can see, this is as basic as it can be. The constructor takes in a pointer to the `Window` class. We have a couple of `add`methods for the light users, as well as the light streams themselves. We also have a few render methods for specific tasks. Note the constant integer that this class defines for the maximum number of light streams allowed per shader pass. Rendering only three light streams like we did before is a bit wasteful, so this can be upped even more, provided it doesn't become detrimental to the performance of the process.

The helper methods of which there are three--deal with clearing the buffer textures, setting their views, and displaying the changes made to them. We also store the `sf::VertexArray` of the quad that we're going to use to perform a light pass operation.

### Implementing the light manager

As always, let's begin by seeing what needs to be constructed when the light manager is created:

[PRE23]

The initializer list is useful for storing the `Window` pointer, as well as initializing the ambient lighting to absolute black. Once this is done, the window size is obtained and all the material textures are created. Lastly, the window-sized quad is set up for later use.

The adder and getter methods are quite simple, yet they are necessary:

[PRE24]

Dealing with material maps all at once can be quite wasteful typing-wise, so we need a few methods to help us do this quickly:

[PRE25]

Note the view we're using in `SetViews()`. Since these material maps are going to be used instead of the window, they must use the window's view in order to handle the world coordinates of all the visuals being drawn.

Speaking of material maps, every class that wishes to use our light manager should be able to draw to every single one of them. Luckily, we've made it easier on ourselves by making it a requirement that these classes implement a purely virtual `Draw` method:

[PRE26]

After all the textures are cleared and their views are set, each light user needs to draw something for each of the allowed layers our game engine supports. Quite literally, on top of this, any visuals that are above these elevations also need to have a chance to be rendered, which we can achieve by using the second loop. All the material textures are then updated by invoking the `DisplayAll()` method.

Once the materials are drawn, we need to go through the same process of multi-pass shading as we did in our minimal code example:

[PRE27]

This is very close to the already established model we discussed before. A couple of changes to note here are: the use of an internal data member called `m_materialMaps` for passing material information to the light pass shader and the check near the bottom where the diffuse texture is passed in as the `"LastPass"` uniform if it is the very first shader pass. This has to be done otherwise we'd be sampling a completely black texture.

### Integrating the light manager class

Once the light manager is implemented, we can add all the classes that use it to its list:

[PRE28]

In this case, we're only working with the game map, the system manager, and the particle manager classes as light users.

Setting up our previous light information is equally as easy now as it was before:

[PRE29]

Finally, we just need to make sure the material maps are drawn, just like the scene itself:

[PRE30]

Now, the only thing left to do is to adapt those pesky classes to the new lighting model we have set up here.

# Adapting classes to use lights

Obviously, each and every single class that does any rendering in our game does it differently. Rendering the same graphics to different types of material maps is no exception to this rule. Let's see how every light-supporting class should implement their respective `Draw` methods in order to stay in sync with our lighting system.

## The Map class

The first class we need to deal with is the `Map` class. It will be a bit different due to the way it handles the drawing of tiles. So let's take a look at what needs to be added in:

[PRE31]

So far, so good! The `Map` class is now using the `LightUser` interface. The `m_textures` data member is an established array that existed before all of this and it simply stores different textures for each supported elevation. One new protected member function is added though, called `CheckTextureSizes`:

[PRE32]

This is just a handy way of making sure all the future textures, as well as the current diffuse maps, have the appropriate size.

Let's see what the `Redraw` method now needs to do in order to fully support the light manager:

[PRE33]

Only a few extra lines add the support here. We just need to make sure the renderer is involved when the drawing is happening because it allows the right shader to be used in the process.

Since we're going to add more material maps quite soon, clearing of these textures also needs to be integrated into the existing code:

[PRE34]

The spaces for doing so are marked with comments, which is exactly the same for the helper methods that aid in displaying all the changes made to these buffer textures:

[PRE35]

The actual `Draw` method from the `LightUser` class can be implemented like this:

[PRE36]

Because of the way the `Map` class works, all we have to do is set up the sprite we're working with to use the right texture for the appropriate material type. In this case, all we need is the diffuse texture.

## The entity renderer system

If you recall, the `SystemManager` class is the one we added to `LightManager` as `LightUser`. Although there's only one system that does the rendering for now, we still want to keep it this way and simply forward all the arguments passed to `SystemManager`. This keeps our options for additional systems doing the same thing open in the future:

[PRE37]

The forwarded arguments are sent to `S_Renderer` and can be used like so:

[PRE38]

It's fairly similar to how the `Map` class handles its redrawing process. All we need to do is make sure the Renderer class is used to do the drawing to the diffuse texture, which is what happens under the hood, as `C_Drawable` simply passes these arguments down the line:

[PRE39]

## The particle system

Drawing particles in this way is not much different from how other `LightUser` do it:

[PRE40]

Once again, it's all about making sure the materials are passed through `Renderer`.

# Preparing for additional materials

Drawing basic light streams is fairly nifty. But let's face it, we want to do more than that! Any additional processing is going to require further material information about the surfaces we're working with. As far as storing those materials goes, the `Map` class needs to allocate additional space for textures that will be used for this purpose:

[PRE41]

These textures will also need to be checked for incorrect sizes and adjusted if it ever comes to that:

[PRE42]

Clearing the material maps is equally as simple; we just need to add a couple of extra lines:

[PRE43]

Displaying the changes that were made to the buffer textures follows the same easy and manageable approach:

[PRE44]

Finally, drawing this information to the internal buffers of `LightManager`, in the case of the `Map` class, can be done like so:

[PRE45]

Easy enough? Good! Let's keep progressing and build shaders that can handle the process of drawing these material maps.

## Preparing the texture manager

In order to automatically load the additional material maps when loading diffuse images, we need to make some very quick and painless changes to the `ResourceManager` and `TextureManager` classes:

[PRE46]

By adding the `OnRequire()` and `OnRelease()` methods and integrating them properly with the `l_notifyDerived` flag to avoid infinite recursion, `TextureManager` can safely load in both the normal and specular material maps when a diffuse texture is loaded, provided they are found. Note that the texture manager actually passes in `false` as the second argument when it needs these maps to avoid infinite recursion.

## Material pass shaders

There will be two types of material pass shaders we'll use. One type, simply referred to as *MaterialPass*, will sample the material color from a texture:

[PRE47]

It retrieves the diffuse pixel and the material texture pixel, as well as uses the diffuse alpha value to display the right color. This effectively means that if we're dealing with a transparent pixel on a diffuse map, no material color is going to be rendered for it. Otherwise, the material color is completely independent of the diffuse pixel. This is useful for drawing images that also have material maps located in a different texture.

The second type of material shader, known from here on out as *MaterialValuePass*, will also sample the diffuse pixel. Instead of using a material texture, however, it'll simply use a static color value for all the pixels that aren't transparent:

[PRE48]

Here, we first verify that the sampled pixel isn't completely black. If it is, the `alpha` value of `gl_Color` is used instead of that of the pixel. Then, we simply write the static material color value to the fragment. This type of shader is useful for drawable objects that don't have material maps and instead use a static color for every single pixel.

# Normal maps

Lighting can be used to create visually complex and breath taking scenes. One of the massive benefits of having a lighting system is the ability it provides to add extra details to your scene, which wouldn't have been possible otherwise. One way of doing so is using **normal maps**.

Mathematically speaking, the word *normal* in the context of a surface is simply a directional vector that is perpendicular to said surface. Consider the following illustration:

![Normal maps](img/image_08_005.jpg)

In this case, what's normal is facing up because that's the direction perpendicular to the plane. How is this helpful? Well, imagine you have a really complex model with many vertices; it'd be extremely taxing to render said model because of all the geometry that would need to be processed with each frame. A clever trick to work around this, known as **normal mapping**, is to take the information of all of those vertices and save them on a texture that looks similar to this one:

![Normal maps](img/image_08_006.jpg)

It probably looks extremely funky, especially if being looked at in a physical release of this book that's in grayscale, but try not to think of this in terms of colors, but directions. The red channel of a normal map encodes the *-x* and *+x* values. The green channel does the same for *-y* and *+y* values, and the blue channel is used for *-z* to *+z*. Looking back at the previous image now, it's easier to confirm which direction each individual pixel is facing. Using this information on geometry that's completely flat would still allow us to light it in such a way that it would make it look like it has all of the detail in there; yet, it would still remain flat and light on performance:

![Normal maps](img/image_08_007.jpg)

These normal maps can be hand-drawn or simply generated using software such as *Crazybump*. Let's see how all of this can be done in our game engine.

## Implementing normal map rendering

In the case of maps, implementing normal map rendering is extremely simple. We already have all the material maps integrated and ready to go, so at this time, it's simply a matter of sampling the texture of the tile sheet normals:

[PRE49]

The process is exactly the same as drawing a normal tile to a diffuse map, except that here we have to provide the material shader with the texture of the tile-sheet normal map. Also note that we're now drawing to a normal buffer texture.

The same is true for drawing entities as well:

[PRE50]

You can try obtaining a normal texture through the texture manager. If you find one, you can draw it to the normal map material buffer.

Dealing with particles isn't much different from what we've seen already, except for one small detail:

[PRE51]

As you can see, we're actually using the material value shader in order to give particles static normals, which are always sort of pointing to the camera. A normal map buffer should look something like this after you render all the normal maps to it:

![Implementing normal map rendering](img/image_08_008.jpg)

## Changing the lighting shader

Now that we have all of this information, let's actually use it when calculating the illumination of the pixels inside the light pass shader:

[PRE52]

First, the normal map texture needs to be passed to it, as well as sampled, which is where the first two highlighted lines of code come in. Once this is done, for each light we're drawing on the screen, the normal directional vector is calculated. This is done by first making sure that it can go into the negative range and then normalizing it. A normalized vector only represents a direction.

### Note

Since the color values range from *0* to *255*, negative values cannot be directly represented. This is why we first bring them into the right range by multiplying them by *2.0* and subtracting by *1.0*.

A **dot product** is then calculated between the normal vector and the normalized `L` vector, which now represents the direction from the light to the pixel. How much a pixel is lit up from a specific light is directly contingent upon the dot product, which is a value from *1.0* to *0.0* and represents magnitude.

### Note

A **dot product** is an algebraic operation that takes in *two vectors*, as well as the *cosine* of the angle between them, and produces a scalar value between *0.0* and *1.0* that essentially represents how "orthogonal" they are. We use this property to light pixels less and less, given greater and greater angles between their normals and the light.

Finally, the dot product is used again when calculating the final pixel value. The entire influence of the light is multiplied by it, which allows every pixel to be drawn differently as if it had some underlying geometry that was pointing in a different direction.

The last thing left to do now is to pass the normal map buffer to the shader in our C++ code:

[PRE53]

This effectively enables normal mapping and gives us beautiful results such as this:

![Changing the lighting shader](img/image_08_009.jpg)

The leaves, the character, and pretty much everything in this image, now look like they have a definition, ridges, and crevices; it is lit as if it had geometry, although it's paper-thin. Note the lines around each tile in this particular instance. This is one of the main reasons why normal maps for pixel art, such as tile sheets, shouldn't be automatically generated; it can sample the tiles adjacent to it and incorrectly add bevelled edges.

# Specular maps

While normal maps provide us with the possibility of faking how bumpy a surface is, specular maps allow us to do the same with the shininess of a surface. This is what the same segment of the tile sheet we used as an example for a normal map looks like in a specular map:

![Specular maps](img/image_08_010.jpg)

It's not as complex as a normal map, since it only needs to store one value: the shininess factor. We can leave it up to each light to decide how much *shine* it will cast upon the scenery by letting it have its own values:

[PRE54]

## Adding support for specularity

Similar to normal maps, we need to use the material pass shader to render to a specularity buffer texture:

[PRE55]

The texture for specularity is once again attempted to be obtained; it is passed down to the material pass shader if found. The same is true when you render entities:

[PRE56]

Particles, on the other hand, also use the material value pass shader:

[PRE57]

For now, we don't want any of them to be specular at all. This can obviously be tweaked later on, but the important thing is that we have that functionality available and yielding results, such as the following:

![Adding support for specularity](img/image_08_011.jpg)

This specularity texture needs to be sampled inside a light pass, just like a normal texture. Let's see what this involves.

## Changing the lighting shader

Just as before, a uniform `sampler2D` needs to be added to sample the specularity of a particular fragment:

[PRE58]

We also need to add in the specular exponent and strength to each light's `struct`, as it's now part of it. Once the specular pixel is sampled, we need to set up the direction of the camera as well. Since that's static, we can leave it as is in the shader.

The specularity of the pixel is then calculated by taking into account the dot product between the pixel's normal and the light, the color of the specular pixel itself, and the specular strength of the light. Note the use of a specular constant in the calculation. This is a value that can, and should, be tweaked in order to obtain the best results, as 100% specularity rarely looks good.

Then, all that's left is to make sure the specularity texture is also sent to the light pass shader, in addition to the light's specular exponent and strength values:

[PRE59]

The result may not be visible right away, but upon closer inspection of moving a light stream, we can see that correctly mapped surfaces will have a glint that will move around with the light:

![Changing the lighting shader](img/image_08_012.jpg)

While this is nearly perfect, there's still some room for improvement.

# Height maps

The main point of illuminating the world is to make all the visual details pop up in a realistic manner. We have already added artificial dynamic lighting, fake 3D geometry, and shininess, so what's left? Well, there's nothing that shows the proper height of the scene yet. Until this very moment, we've been dealing with the scene as if it's completely flat when calculating the lighting distances. Instead of this, we need to work on something referred to as the height map that will store the heights of the pixels.

## Adapting the existing code

Drawing heights properly can be quite tricky, especially in the case of tile maps. We need to know which way a tile is facing when drawing realistic heights. Consider the following illustration:

![Adapting the existing code](img/image_08_013.jpg)

The tiles right next to point **A** have no normals associated with them, while the tiles next to point **B** are all facing the camera. We can store normal data inside our map files by making these few simple alterations:

[PRE60]

The `Tile` structure itself holds on to a normal value now, which will be used later on. When tiles are being read in from a file, additional information is loaded at the very end. The last two lines here show the actual entries from a map file.

Drawing the heights of these tiles based on their normals is all done in the appropriate shader, so let's pass all of the information it needs:

[PRE61]

The height pass shader uses a value for the base height of the drawable, which, in this case, is just elevation in world coordinates. It also uses the *Y* world coordinate of the `Drawable` class and takes in the surface normal. The same values need to be set up for the entities as well:

[PRE62]

In this case, however, we're using the same normal for all the entities. This is because we want them to face the camera and be illuminated as if they're standing perpendicular to the ground. Particles, on the other hand, are not facing the camera, but instead have normals pointing up toward the positive *Y* axis:

[PRE63]

## Writing the height pass shader

The height pass is the only program we've written so far that uses both the vertex and the fragment shaders.

Let's take a look at what needs to happen in the vertex shader:

[PRE64]

There's only one line here that isn't standard from what is traditionally known as a vertex shader, outside of the uniform variable and the out variable, of course. The vertex shader outputs a floating point value called `Height` to the fragment shader. It's simply the height between the *Y* component of the vertex of a shape in world coordinates and the base *Y* position of that same shape. The height is then interpolated between all the fragments, giving a nice, gradient distribution.

### Note

The `gl_Vertex` information is stored in world coordinates. The bottom *Y* coordinates always start at the same height as the drawable, which makes the top *Y* coordinates equal to the sum of its position and height.

Finally, we can take a look at the fragment shader and actually do some filling up of fragments:

[PRE65]

As shown previously, it takes in the diffuse texture, the surface normal, the base height of the drawable, and the interpolated `Height` value from the vertex shader. The diffuse pixel is then sampled in order to use its alpha value for transparency. The height value itself is calculated by subtracting the result of the pixel height being multipled by the surface normal's *Z* component from the base height of the drawable. The whole thing is finally divided by *255* because we want to store color information in a normalized format.

## Changing the lighting shader

Finally, the light pass shader can be changed as well by sampling the height map:

[PRE66]

Once the pixel height is sampled and multiplied by *255* to bring it back to world coordinates, all we need to do is replace the `gl_FragCoord.z` value with `pixelHeight` when calculating the distance between a pixel and a fragment. Yes, that's really all it takes!

The `HeightMap` can then be actually passed to the shader for sampling, like so:

[PRE67]

This gives us a very nice effect that can actually show off the height of a particular structure, given it has elevated properly and has the right normals:

![Changing the lighting shader](img/image_08_014.jpg)

The light post on the left has no normals, while the post on the right has normals that face the *+Z* direction. The light position is exactly the same in both these images.

# Summary

If you are still here, congratulations! That was quite a bit of information to take in, but just as our world is finally beginning to take shape visually, we're about to embark on an even more stunning feature that will be discussed in the next chapter. See you there!