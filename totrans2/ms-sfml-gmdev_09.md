# Chapter 9.  The Speed of Dark - Lighting and Shadows

Contrasting differences are the very essence of existence, as the *yin-yang* symbol properly illustrates. Light and darkness are opposites, yet complementary, as they offset one another and give meaning through variety. Without darkness there can be no light, as they are never truly separate. By breathing light into our world, we are inevitably forced to add back the darkness that it creates. Let's follow the previous chapter and truly complete to our lighting engine by reintroducing the concept of darkness to it.

In this chapter, we will be covering the following topics:

*   Using OpenGL to render to and sample from cubemap textures
*   Advanced shadow mapping for omni-directional point lights
*   The use of Percentage Closer Filtering to smooth out shadow edges
*   Combating common and frustrating issues with shadow mapping

There's quite a bit of theory to get out of the way, so let's get to it!

# Use of third-party software

Before diving into such a difficult subject to debug, it's always nice to have proper tools that will ease the headaches and reduce the number of questions one might ask oneself during development. While normal code executed on the *CPU* can just be stepped through and analyzed during runtime, shader code and OpenGL resources, such as textures are a bit more difficult to handle. Most, if not all, *C++* compilers don't have native support for dealing with *GPU-bound* problems. Luckily, there is software out there that makes it easier to deal with that very predicament.

Among the few tools that exist out there to alleviate such headaches, *CodeXL* by *AMD Developer Tools Team* stands out. It's a free piece of software that can be used as a standalone application for Windows and Linux or even as a plugin for Visual Studio. Its most prominent features include being able to view OpenGL resources (including textures) while the program is running, profile the code and find bottlenecks, and even step through the shader code as it's being executed (given the right hardware). The tool can be found and downloaded here: [http://gpuopen.com/compute-product/codexl/](http://gpuopen.com/compute-product/codexl/).

# Theory behind shadowing techniques

There are a couple of different techniques that can be used when implementing realistic looking shadows in games. Choosing the right one can not only impact the kind of performance your application is going to exhibit, but can also heavily influence how good the effect is going to look in the end.

An approach that isn't at all uncommon for 2D is referred to as **ray tracing**. Depending on the type of light, a number of rays are cast in an appropriate direction. Shadows are then implemented depending on which solids these rays actually intersect with. Some simpler games tend to create an overlay mask and fill in geometrically the parts of it that are "in the shadow". This mask is later overlaid on top of the usual scene and blended in order to create the aesthetic of darkened areas meant to represent shadows. More advanced 3D games tend to allow rays to bounce around the scene, carrying different information about the particular fragments that they intersect with. By the time a ray reaches the camera, it will have enough information to do more than create simple shadows. Scenes that require extremely advanced lighting tend to use this technique, and rightly so, as it imitates the way light bounces off objects and hits the observer's eye in real life.

An older, but still widely used approach for specifically creating shadows is called **shadow mapping**. The essence of this technique comes down to simply rendering the scene to an off screen buffer from the point of view of the light. All the solids' depth information, as opposed to color information, is written to this buffer as pixel data. When the real scene is rendered, some matrix math is then used to sample the right pixels of the shadow map to figure out whether they can be directly seen by the light, thus being illuminated, or whether they're being obstructed by something, and therefore sitting in the shadow.

## Shadow mapping

The main idea behind creating a shadow map is rendering the scene from the point of view of the light, and effectively encoding the depth of a particular piece of geometry being rendered as a color value that can later be sampled. The depth value itself is nothing more than the distance between the position of the light and the position of the vertex. Consider the following diagram:

![Shadow mapping](img/image_09_001.jpg)

The distance between the light and a given vertex will be converted to a color value by simply dividing it by the frustum far distance, yielding a result in a range *[0;1]*. The frustum far value is simply the distance of how far the light/camera can see.

## Omni-directional point lights

In the previous chapter, we managed to create lights that emit in all directions from a center point. These types of lights have a very fitting name: omni-directional point lights. Dealing with shadow mapping for these lights comes with a certain layer of complexity, as the scene now needs to be drawn in all six directions, rather than just one if we were dealing with a directional light. This means we need a good way of storing the results of this process that can be accessed with relative ease. Luckily, OpenGL provides a new type of texture we can use, the **cubemap**.

### Cubemap textures

A cubemap is pretty much exactly what it sounds like. It's a special texture that really holds six textures for each face of the cube. These textures are internally stored in an *unfolded* manner, as shown here:

![Cubemap textures](img/image_09_003.jpg)

Because of this property, rendering shadow maps for omni-directional lights can be as simple as rendering the scene once for each direction of a cubemap. Sampling them is also quite easy. The shape of a cube lends itself to some useful properties we can exploit. If all of the cube's vertices are in relation to its absolute center, then the coordinates of these vertices can also be thought of as directional vectors:

![Cubemap textures](img/image_09_005.jpg)![Cubemap textures](img/image_09_006.jpg)

The direction (0, 1, 0) from the center of the cube would be pointing directly in the middle of the *+Y* face, for example. Since each face of a cubemap texture also holds a texture of its own that represents the view of the scene, it can easily be sampled using these coordinates. For a 2D texture, our shaders had to use the `sampler2D` type and provide 2D coordinates of the sampling location. Cubemaps have their own sampler type, `samplerCube`, and use a 3D vector for sampling. The consequence of this is that the largest member of the 3D vector is used to determine which face is to be sampled, and the other two members become the UV texture coordinates for that particular 2D texture/face.

### Note

Cube textures can be used for much more than shadow mapping. 3D environments can take advantage of them when implementing skyboxes and reflective/refractive materials, to name just a few techniques.

# Preparations for rendering

It's safe to say that all of this functionality is a bit beyond the scope of SFML, as it seeks to deal with simple two-dimensional concepts. While we're still going to be using SFML to render our sprites, the lighting and shadowing of the scene will have to fall back on raw OpenGL. This includes setting up and sampling cubemap textures, as well as creating, uploading, and drawing 3D primitives used to represent objects that cast shadows.

## Representing shadow casters

While SFML is great for rendering sprites, we must remember that these are two-dimensional objects. In 3D space, our character would literally be paper thin. This means that all of our game's shadow casters are going to need some 3D geometry behind them. Keep in mind that these basic rendering concepts have already been covered in [Chapter 7](ch07.html "Chapter 7.  One Step Forward, One Level Down - OpenGL Basics") , *One Step Forward, One Level Down - OpenGL Basics*. Let's start by creating some common definitions that this system will use:

[PRE0]

This is going to be a common lookup array for us, and it's important that the directional vectors here are defined correctly. It represents a direction towards each face of the cubemap texture.

Another common data structure we will be using is a list of indices used to draw the cubes/3D rectangles that represent our shadow casters:

[PRE1]

Since the cubes have 6 faces and each face uses 6 indices to enumerate the two triangles that make them up, we have a total of 36 indices.

Finally, we need an up vector for each direction of a cubemap texture:

[PRE2]

In order to get correct shadow mapping for the geometry, we're going to need to use these up directions when rendering to a shadow cubemap. Note that, unless we're rendering to *Y* faces of the cubemap, the *Y* direction is always used as up. This allows the geometry being rendered to be seen correctly by the camera.

### Implementing the shadow caster structure

Representing the literally shapeless entities of our game is the task we're going to be tackling next. In order to minimize the memory usage of this approach, it will be broken down into two parts:

*   **Prototype**: This is a structure that holds handles to uploaded geometry used by OpenGL. This kind of object represents a unique, one of a kind model.
*   **Caster**: This is a structure that holds a pointer to a prototype it's using, along with its own transform, to position, rotate, and scale it correctly.

The prototype structure needs to hold on to the resources it allocates, as follows:

[PRE3]

The constructor and destructor of this structure will take care of allocation/de-allocation of these resources:

[PRE4]

Once the internal `m_vertices` data member is properly filled out, the geometry can be submitted to the GPU as follows:

[PRE5]

Once the vertex array object and two buffers for vertices and indices are properly created, they're all bound and used to push the data to. Note the highlighted portion of the code that deals with the vertex attributes. Since this geometry is only going to be used to generate shadows, we really don't need anything else except the vertex position. The necessary math of converting all of that information into color values that represent distance from the light source is going to be done inside the shaders.

Also, note the usage of indices to render this geometry here. Doing it this way allows us to save some space by not having to upload twice as many vertices to the GPU as we would have to otherwise.

The drawing of the shadow primitives is just as simple as one would imagine:

[PRE6]

Once all of the buffers are bound, we invoke `glDrawElements`. Let it know we're drawing triangles, give the method the count of indices to use, specify their data type, and provide the proper offset for those indices, which in this case is *0*.

Finally, because we're using prototypes to store unique pieces of geometry, it's definitely useful to overload the `==` operator for easy checking of matching shapes:

[PRE7]

Each vertex of the shadow primitive is iterated over and compared to the equivalent vertex of the provided argument. So far, nothing out of the ordinary!

The prototypes are going to need to be identified in some way when they're being stored. Using string identifiers can be quite intuitive in this case, so let's define a proper storage container type for this structure:

[PRE8]

With that out of the way, we can implement our simple `ShadowCaster` structure that's going to hold all of the variable information about the prototype:

[PRE9]

As you can see, it's a very simple data structure that holds a pointer to a prototype it uses, as well as its own `GL_Transform` member, which is going to store the displacement information of an object.

The shadow casters are also going to need a proper storage data type:

[PRE10]

This effectively leaves us with the means to create and manipulate different types of shadow-casting primitives in a memory-conservative manner.

## Creating the transform class

The transform class that we're using is exactly the same as the one in [Chapter 7](ch07.html "Chapter 7.  One Step Forward, One Level Down - OpenGL Basics") , *One Step Forward, One Level Down - OpenGL Basics*. For a quick refresher, let's take a look at the most important part of it that we're going to need for this process--the generation of a model matrix:

[PRE11]

All of this should be familiar by now, and if it isn't, a quick zip through [Chapter 7](ch07.html "Chapter 7.  One Step Forward, One Level Down - OpenGL Basics") , *One Step Forward, One Level Down - OpenGL Basics* is definitely in order. The main idea, however, is combining the translation, scale, and rotation matrices in the right order to retrieve a single matrix that contains all of the information about the primitive required to bring its vertices from object space to world space.

## Creating a camera class

Similar to the `GL_Transform` class, we're also going to incorporate the `GL_Camera` class from [Chapter 7](ch07.html "Chapter 7.  One Step Forward, One Level Down - OpenGL Basics") , *One Step Forward, One Level Down - OpenGL Basics*. When we're rendering shadow maps, the projection and view matrices for all six directions will need to be submitted to the respective shaders. This makes the `GL_Camera` class perfect for representing a light in a scene that needs to draw what it sees into a cubemap texture. Once again, this has been covered already, so we're just going to breeze through it:

[PRE12]

Appropriately enough, shadow maps are going to be drawn using a perspective projection. After all the necessary information about view frustum is collected, we can begin constructing the matrices necessary to transform those vertices from world space to the light's view space, as well as to clip space:

[PRE13]

We're using `glm::lookAt` to construct a view matrix for the light's camera. Then, `glm::perspective` is used in another method to create the perspective projection matrix for the camera.

### Note

It's very important to remember that `glm::perspective` takes the field of view angle of the view frustum as the first argument. It expects this parameter to be in **radians**, not degrees! Because we're storing it in degrees, `glm::radians` is used to convert that value. This is a very easy mistake to make and many people end up having problems with their shadow maps not mapping correctly.

## Defining a cube texture class

Now that we have the storage of geometry and representation of the light's view frustum figured out, it's time to create the cube texture we're going to use to actually render the scene to.

Let's start by creating a simple class definition for it:

[PRE14]

The texture is going to be used for two distinctive actions: being rendered to and being sampled. Both of these processes have a method for binding and unbinding the texture, with the notable difference that the sampling step also requires a texture unit as an argument. We're going to cover that soon. This class also needs to have a separate method that needs to be called for each of the six faces when they're being rendered.

Although cube textures can be used for many things, in this particular instance, we're simply going to be using them for shadow mapping. The texture dimensions, therefore, are defined as constants of *1024px*.

### Tip

The size of a cubemap texture matters greatly, and can cause artifacting if left too small. Smaller textures will lead to sampling inaccuracies and will cause jagged shadow edges.

Lastly, alongside the helper methods used when creating the texture and all of the necessary buffers, we store the handles to the texture itself, the frame buffer object, and render buffer object. The last two objects haven't been covered until this point, so let's dive right in and see what they're for!

### Implementing the cube texture class

Let's start, as always, by covering the construction and destruction of this particular OpenGL asset:

[PRE15]

Similar to geometry classes, the handles are initialized to values of *0* to indicate their state of not being set up. The destructor checks those values and invokes the appropriate `glDelete` methods for the buffers/textures used.

Creating the cubemap is quite similar to a regular 2D texture, so let's take a look:

[PRE16]

First, a check is made to make sure we haven't already allocated this object. Provided that isn't the case, `glGenTextures` is used, just like for 2D textures, to create space for one texture object. Our first private helper method is then invoked to create all six faces of the cubemap, which brings us to the parameter setup. The *Min/Mag* filters are set up to use the nearest-neighbor interpolation, but can later be converted to `GL_LINEAR` for smoother results, if necessary. The texture wrapping parameters are then set up so that they're clamped to the edge, giving us a seamless transition between faces.

### Note

Note that there are three parameters for texture wrapping: R, S, and T. That's because we're dealing with a three-dimensional texture type now, so each axis must be accounted for.

Lastly, another helper method is invoked for the creation of the buffers, just before we unbind the texture as we're done with it.

The creation of the cubemap faces, once again, is similar to how we set up its 2D counterpart back in [Chapter 7](ch07.html "Chapter 7.  One Step Forward, One Level Down - OpenGL Basics") , *One Step Forward, One Level Down - OpenGL Basics*, but the trick is to do it once for each face:

[PRE17]

Once the texture is bound, we iterate over each face and use `glTexImage2D` to set the face up. Each face is treated as a 2D texture, so this should really be nothing new to look at. Note, however, the use of the `GL_TEXTURE_CUBE_MAP_POSITIVE_X` definition usage is the first argument. 2D textures would take in a `GL_TEXTURE_2D` definition, but because cubemaps are stored in an unfolded manner, getting this part right is important.

### Note

There are six definitions of `GL_TEXTURE_CUBE_MAP_`*. They're all defined in a row of *+X*, *-X*, *+Y*, *-Y*, *+Z*, and *-Z*, which is why we can use some basic arithmetic to pass in the correct face to the function by simply adding an integer to the definition.

Clearing the cubemap texture is relatively easy:

[PRE18]

Note that we're specifying the clear color as white, because that represents *infinite distance from the light* in a shadow map.

Finally, sampling the cubemap is actually not any different from sampling a regular 2D texture:

[PRE19]

Both binding and unbinding for sampling requires us to pass in the texture unit we want to use. Once the unit is active, we should enable the use of cubemaps and then bind the cubemap texture handle. The reverse of this procedure should be followed when unbinding the texture.

### Note

Keep in mind that the respective `sampler2D`/`samplerCube` uniforms inside fragment shaders are set to hold the unit ID of the texture they're sampling. When a texture is bound, the specific ID of that unit will be used to access it in a shader from then on, not the actual texture handle.

#### Rendering to an off-screen buffer

Something we didn't cover in [Chapter 7](ch07.html "Chapter 7.  One Step Forward, One Level Down - OpenGL Basics") , *One Step Forward, One Level Down - OpenGL Basics* is rendering a scene to a buffer image, rather than drawing directly onscreen. Luckily, because OpenGL operates as a giant state machine, it's just a matter of invoking the right functions at the right time, and doesn't involve us having to redesign the rendering procedures in any way.

In order to render to a texture object, we must use what is called a **framebuffer**. It's a very basic object that directs draw calls to a texture the FBO is bound to. While FBOs are useful for color information, they don't carry the depth components with them. A **renderbuffer** object is used for that very purpose of attaching additional components to the FBO.

The first step to drawing something offscreen is creating a `FRAMEBUFFER` object and a `RENDERBUFFER` object:

[PRE20]

After the buffers have been generated, the render buffer needs to have some storage allocated for any additional components it will provide. In this case, we're simply dealing with the depth component.

### Tip

The `GL_DEPTH_COMPONENT24` simply indicates that each depth pixel has a size of 24 bits. This definition can be replaced with a basic `GL_DEPTH_COMPONENT`, which will allow the application to choose the pixel size.

The depth render buffer is then attached to the FBO as a depth attachment. Finally, if there were any errors during this procedure, `glCheckFramebufferStatus` is used to catch them. The next line simply prints out the status variable using `std::cout`.

### Note

Frame buffers should always be unbound when no longer used, using `glBindFramebuffer(GL_FRAMEBUFFER, 0)`! That's the only way we're ever going to go back to rendering subsequent geometry to the screen, rather than the buffer texture.

Now that we have the buffers set up, let's use them! When drawing to a buffer texture is desired, it's first necessary to bind the frame buffer:

[PRE21]

Unbinding the FBO is necessary after we're done with it. Using `RenderingUnbind()` means that any subsequent geometry will be drawn onscreen.

Of course, just because the FBO is bound, doesn't mean we're going to magically start drawing to the cubemap. In order to do that, we must draw to one face at a time by binding the frame buffer to the desired face of the cubemap:

[PRE22]

The first argument to `glFramebufferTexture2D` simply indicates we're dealing with an FBO. We then specify that we want to use `GL_COLOR_ATTACHMENT0`. Frame buffers can have multiple attachments and use shaders to output different data to each one of them. For our purposes, we're only going to need to use one attachment.

Because we're rendering to one face of the cubemap at a time, basic definition arithmetic is, once again, used to pick the correct face of the cube to render to. Finally, the texture handle and mipmapping level are passed in at the very end, just before `Clear()` is invoked to clear the face we currently bound to complete white.

# Rendering the shadow maps

We now have everything we need in order to start rendering shadow maps of our scene. Some rather significant changes are going to have to be made to the `LightManager` class in order to support this functionality, not to mention properly store and use these shadow map textures during later passes. Let's see what changes we need to make in order to make this happen.

## Modifying the light manager

First, let's make some adjustments to the light manager class definition. We're going to need a couple of methods to add shadow caster prototypes, add actual shadow casting objects, and render the shadow maps:

[PRE23]

In addition to the aforementioned methods, the `LightManager` class is also going to need to store extra information to support these changes. A list of both shadow primitive prototypes and the primitives themselves will need to be used to manage the entities that have to cast shadows. Additionally, we need to have the camera class that will be used as the point of view of the light.

Lastly, an array of cubemap textures is required, since each light onscreen will be potentially seeing the scene from a completely different point of view, of course. The size of this array is simply the number of lights we're dealing with per shader pass, because these cubemap textures only need to exist for as long as they're being sampled. Once the lighting pass for those particular lights is over, the textures can be re-used for the next batch.

### Implementing the light manager changes

The adjustments to the constructor of the `LightManager` class are fairly simple to make this work:

[PRE24]

The first thing we need to worry about is setting up the perspective camera correctly. It's initialized to be positioned at absolute zero coordinates in the world, and has its field of view angle set to **90 degrees**. The aspect ratio of the perspective camera is obviously going to be *1*, because the width and height of the textures we're using for rendering shadow casters to are identical. The view frustum minimum value is set to *1.f*, which ensures that the geometry won't be rendered if the light is intersecting with a face. The maximum value, however, will change for each light, depending on its radius. This default value isn't really important.

### Note

Setting the field of view angle of **90** degrees for rendering a scene to a cubemap texture is important, as that's the only way the scene is going to be captured completely for each direction the camera looks at. Going too low on this value means there are going to be blind spots, and going too high will cause overlapping.

The last thing we need to do in the constructor is make sure that all cubemap textures are allocated properly.

Next, let's worry about adding shadow caster prototypes to the light manager:

[PRE25]

When adding a prototype, the caller of this particular method will provide a string identifier for it, as well as move its established and allocated smart pointer to the second argument after the vertices have been properly loaded. First, we make sure the name provided as an argument isn't already taken. If it is, that same string is returned back just after the memory for the prototype provided as an argument is released.

The second test makes sure that a prototype with the exact arrangement of vertices doesn't already exist under a different name, by iterating over every stored prototype and using the `==` operator we implemented earlier to compare the two. If something is found, the name of that prototype is returned instead, just after the `l_caster` is released.

Finally, since we can be sure that the prototype we're adding is completely unique, the render window is set to active. `UploadVertices` on the object is invoked to send the data to the GPU and the prototype is placed inside the designated container.

### Note

Using `sf::RenderWindow::setActive(true)` ensures that the main context is used while the vertices are uploaded. OpenGL **does not** share its states among different contexts, and since SFML likes to keep a number of different contexts alive internally, it's imperative to make sure the main context is selected during all operations.

Adding shadow casters themselves is relatively easy as well:

[PRE26]

This method only takes a string identifier for the prototype to be used, and allocates space for a new shadow caster object, provided the prototype with said name exists. Note the line just before the `return` statement. It ensures that the located prototype is passed to the shadow caster, so that it can use the prototype later.

Obtaining the prototypes is incredibly simple, and only requires a lookup into an `unordered_map` container:

[PRE27]

We now only have one task at hand drawing the shadow maps!

#### Drawing the actual shadow maps

In order to keep this manageable and compartmentalized, we're going to break down the `DrawShadowMap` method into smaller parts that we can discuss independently of the rest of the code. Let's start by looking at the actual blueprint of the method:

[PRE28]

First, it takes in a handle for the shadow pass shader. This is about as raw as it gets, since the handle is a simple unsigned integer we're going to bind to before drawing. The second argument is a reference to a light that we're currently drawing the shadow map for. Lastly, we have an *unsigned integer* that serves as the ID for the light that's being rendered in the current pass. In the case of having 4 lights per shader pass, this value will range from 0 to 3, and then get reset in the next pass. It is going to be used as an index for the cubemap texture lookup.

Now, it's time to really get into the actual rendering of the shadow maps, starting with enabling necessary OpenGL features:

[PRE29]

The first and most obvious feature we're going to be using here is the depth test. This ensures that different shadow caster geometry isn't rendered in the wrong order, overlapping each other. Then, we're going to be performing some face culling. Unlike normal geometry, however, we're going to be culling the front faces only. Drawing the back faces of shadow geometry will ensure that the front faces of sprites we're using will be lit, since the depth stored in the shadow map is the depth of the very back of the shadow-casting primitives.

[PRE30]

The next part here deals with actually binding the shadow pass shader and fetching locations of different shader uniform variables. We have a model matrix uniform, a view matrix uniform, a projection matrix uniform, a light position uniform, and the frustum far uniform to update.

[PRE31]

This next part of the code obtains a reference to the appropriate cubemap texture for the particular light, storing the light position, and positioning the perspective camera at that exact position.

### Note

Note the swapped *Z* and *Y* coordinates. By default, OpenGL deals with the right-hand coordinate system. It also deals with the default *up* direction being the *+Y* axis. Our lights store coordinates using the *+Z* axis as the *up* direction.

After the camera is set up, `glViewport` is invoked to resize the render target to the size of the cubemap texture. The cubemap is then bound to for rendering and we submit the light position uniform to the shaders. Just as before, the *Z* and *Y* directions here are swapped.

With the setup out of the way, we can actually begin rendering the scene for each face of the cubemap:

[PRE32]

The cubemap texture is first told which face we wish to render to in order to set up the FBO correctly. The forward and up directions for that particular face are then passed to the light's camera, along with the frustum far value, being the radius of the light. The perspective projection matrix is then recalculated, and both the view and projection matrices are retrieved from `GL_Camera` to pass to the shader, along with the frustum far value.

Lastly, for each of the 6 faces of the cubemap, we iterate over all of the shadow caster objects, retrieve their model matrices, pass them into the shader, and invoke the prototype's `Draw()` method, which takes care of the rendering.

After all of the texture's faces have been drawn to, we need to set the state back to what it was before rendering shadow maps:

[PRE33]

The texture is first unbound for rendering, which sets the FBO to 0 and allows us to draw to the screen again. The viewport is then resized back to the original size our window had, and the depth test, along with face culling, are both disabled.

## The shadow pass shaders

The C++ side of shadow mapping is finished, but we still have some logic to cover. The shaders here play an important role of actually translating the vertex information into depth. Let's take a look at the vertex shader first:

[PRE34]

The `vec3` input coordinates of a vertex position we receive on the GPU are in local space, which means they have to be passed through a number of matrices to be brought to world, view, and clip spaces in that order. The world coordinates are calculated first and stored separately, because they're used to determine the distance between the vertex and the light. That distance is stored in the local variable `d`, which is divided by the frustum far value to convert it to a range of *[0;1]*. The position of the vertex is then converted to clip space by using the world, view, and projection matrices, and the distance value is passed on to the fragment shader, where it's stored as a color for a particular pixel:

[PRE35]

Remember that the output variables from the vertex shader are interpolated between the vertices, so each fragment in between those vertices will be shaded in a gradient-like manner.

## Results

While we still don't have any actual geometry in the project to see the results of this, once we're done, it will look like the following screenshot:

![Results](img/image_09_007.jpg)

In this particular case, the primitives were extremely close to the light, so they're shaded really dark. Given greater distances, a particular face of a shadow map would look a little something like this, where *#1* is a primitive close to the camera, *#2* is further away, and *#3* is near the far end of the view frustum:

![Results](img/image_09_008.jpg)

# Adapting the light pass

With the shadow maps rendered, it may be extremely tempting to try and sample them in our existing code, since the hard part is over, right? Well, not entirely. While we were extremely close with our previous approach, sadly, sampling of cubemap textures is the only thing that we couldn't do because of SFML. The sampling itself isn't really the problem, as much as binding the cubemap textures to be sampled is. Remember that sampling is performed by setting a uniform value of the sampler inside the shader to the **texture unit ID** that's bound to the texture in our C++ code. SFML resets these units each time something is rendered either onscreen, or to a render texture. The reason we haven't had this problem before is because we can set the uniforms of the shaders through SFML's `sf::Shader` class, which keeps track of references to textures and binds them to appropriate units when a shader is used for rendering. That's all fine and good, except for when the time comes to sample other types of textures that SFML doesn't support, which includes cubemaps. This is the only problem that requires us to completely cut SFML out of the picture during the light pass and use raw OpenGL instead.

## Replacing the m_fullScreenQuad

First things first, replacing the `sf::VertexArray` object inside the `LightManager` class that's used to redraw an entire buffer texture, which we were utilizing for multipass rendering. Since SFML has to be completely cut out of the picture here, we can't use its built-in vertex array class and render a quad that covers the entire screen. Otherwise, SFML will force its own state on before rendering, which isn't going to work with our system properly as it re-assigns its own texture units each time.

### Defining a generic frame buffer object

Just like before, we need to create a frame buffer object in order to render to a texture, rather than the screen. Since we've already done this once before for a cubemap, let's breeze through the implementation of a generic FBO class for 2D textures:

[PRE36]

The main difference here is the fact that we're using variable sizes for textures now. They may vary at some point, so it's a good idea to store the size internally, rather than using constant values.

#### Implementing a generic frame buffer object

The constructor and destructor of this class, once again, deals with resource management:

[PRE37]

We're not storing a texture handle, because that too will vary depending on circumstances.

Creating the buffers for this class is pretty similar to what we've done before:

[PRE38]

Just like the cubemap textures, we need to attach a depth render buffer to the FBO. After allocation and binding, the FBO is checked for errors and both buffers are unbound.

Rendering FBO points to a 2D texture is much easier. Binding for rendering needs to take a handle to a texture, because one is not stored internally, since this is a generic class that will be used with many different textures:

[PRE39]

Once the FBO is bound, we again invoke `glFramebufferTexture2D`. This time, however, we use `GL_TEXTURE_2D` as the type of the texture, and pass in the `l_texture` argument into the function instead.

## Rendering from a buffer to another buffer in OpenGL

During our potentially numerous light passes, we're going to need a way of redrawing every pixel onscreen to the buffer texture just like we did before, except without using SFML this time. For this purpose, we're going to construct a quad that has four vertices, all positioned in screen coordinates, and covers the screen entirely. These vertices are also going to have texture coordinates that will be used to sample the buffer texture. A basic structure of such vertex, similar to the one we created in [Chapter 7](ch07.html "Chapter 7.  One Step Forward, One Level Down - OpenGL Basics") , *One Step Forward, One Level Down - OpenGL Basics* looks like this:

[PRE40]

This small structure will be used by the quad primitive that will cover the entire screen.

### Creating a basic quad primitive

The quad primitive, just like any other piece of geometry, must be pushed to the GPU for later use. Let's construct a very basic class that will break down this functionality into manageable methods we can easily call from other classes:

[PRE41]

Once again, we have methods for creating, rendering, binding, and unbinding the primitive. The class stores the `m_VAO`, `m_VBO`, and `m_indices` of this primitive, which all need to be filled out.

#### Implementing the quad primitive class

Construction and destruction of this class, once again, all take care of the resource allocation/de-allocation:

[PRE42]

Creating and uploading the primitive to the GPU is exactly the same as before:

[PRE43]

The main difference here is that we're defining the vertices inside the method, since they're never going to change. The vertex attribute pointers are set up after the data is pushed onto the GPU; indices get defined in a clockwise manner (default for SFML), and pushed to the GPU.

Binding and unbinding the buffers for rendering is, once again, exactly the same as with all of the other geometry for OpenGL:

[PRE44]

Since we're using indices, rendering the quad is achieved by calling `glDrawElements`, just like before:

[PRE45]

This concludes the necessary preparations for rendering from an offscreen buffer to the screen.

## Making the changes to the light manager

Given the complete re-architecture of our rendering process for shadows, it's obvious some things are going to have to change within the `LightManager` class. First, let's start with some new data we're going to need to store:

[PRE46]

The `MaterialHandles` and `MaterialUniformNames` containers will be used to store the names and locations of uniforms in our light pass shader. This is an effort made entirely to make the mapping of new material map types and uniforms much easier by automating it.

With that out of the way, let's take a look at the `LightManager` class definition and the changes we need to make to it:

[PRE47]

In addition to creating some new helper methods for generating material names, binding and unbinding all of the necessary 2D textures for the light pass sampling, and submitting the uniforms of a given light to the light pass shader, we're also storing the material names and handles. The `m_fullScreenQuad` class is replaced by our own class, and to accompany it, we have the `GenericFBO` object that will help us render to an offscreen buffer.

### Implementing light manager changes

The constructor of our `LightManager` class now has additional work to do in setting up all of the new data members we added:

[PRE48]

First, the FBO we'll be using is set up in the initializer list to hold the size of our window. We then ensure that the main OpenGL context is active by activating our window, and invoke the `GenerateMaterials` method that will take care of material texture allocation and storage of the texture handles for the same.

The uniform sampler2D names for all material types are then stored in the appropriate container. These names have to match the ones inside the light pass shader!

Finally, the main OpenGL context is selected again and the FBO is created. We do this one more time for the `m_fullScreenQuad` class as well.

The `GenerateMaterials()` method can be implemented like this:

[PRE49]

It iterates over each material type and creates a new texture for it, just like we did before. The only difference here is that we also store the handle of the newly created texture in `m_materialHandles`, in an effort to tie a specific `MaterialMapType` to an existing texture. We're still using SFML's render textures, because they did a fine job at managing 2D resources.

Binding all of the necessary textures to be sampled in the light pass shader would look like this:

[PRE50]

This particular method will be used inside the `RenderScene` method for rendering lights. It takes two arguments: a handler for the light pass shader, and the ID of the current pass taking place.

The finished texture handle is then obtained from the `Renderer` class. Just like before, we must pass the right texture as the `"LastPass"` uniform in the light pass shader. If we're still on the very first pass, a diffuse texture is used instead.

### Note

Passing textures to a shader for sampling simply means we're sending one integer to the shader. That integer represents the texture unit we want to sample.

The render window is then set to active once again to make sure the main OpenGL context is active. We then bind to the texture unit 0 and use it for the `"LastPass"` uniform. All of the other materials are taken care of inside a `for` loop that runs once for each material type. The texture unit `GL_TEXTURE1 + i` is activated, which ensures that we start from unit 1 and go up, since unit 0 is already being used. The appropriate texture is then bound to, and the uniform of the correct sampler for that material type is located. The uniform is then set to the texture unit we've just activated.

Unbinding these textures is easier still:

[PRE51]

Note that we're now iterating from 0 up and including the material type count. This ensures that even texture unit `0` is unbound, since we're activating `GL_TEXTURE0 + i`.

#### Re-working the light pass

Finally, we'll take a look at the `RenderScene()` method. For clarity, we're going to break it down into smaller chunks, just like before:

[PRE52]

First, let's start at the top of the method and set up some variables that are going to be used throughout:

[PRE53]

The `passes` variable works out how many passes we're going to need with the given number of lights. We then obtain a reference to the beginning of the light container, the light pass shader handle, the shadow pass shader handle, and the shader handle of the currently used shader that's set up inside the `Renderer` object, if there is one. Lastly, the `window` pointer is obtained for easy access.

Still inside the `RenderScene` method, we enter into a `for` loop that's going to iterate for each pass:

[PRE54]

Another reference to a light container iterator is obtained. This time, it points to the first light for this current pass. Also, a `LightCount` variable is set up to keep track of the number of lights rendered for the current pass so far.

Before we go on to do any actual light rendering, we need to draw the shadow maps for the lights we're going to be using in this pass:

[PRE55]

Here, we iterate over each light that belongs to this pass. A check needs to be made to make sure we haven't reached the end of the container, however. Provided that's not the case, the main OpenGL context is enabled by calling `setActive(true)`, and the shadow map for the current light is drawn to the cubemap buffer texture. The `LightCount` is then incremented to let the rest of the code know how many lights we're dealing with during this pass.

After shadow maps have been rendered, it's time to actually bind the light pass shader and begin passing information to it:

[PRE56]

After the light pass shader has been bound, we must also bind all of the 2D textures of necessary material maps. This is followed by submission of the ambient light uniform, along with the light count, and current pass uniforms.

All of this is great, but we still haven't addressed the main concept that caused a necessity for this massive redesign to begin with the cubemap textures:

[PRE57]

The texture unit for binding the very first cubemap texture is defined by simply adding *1* to the count of material map types. We have four types at this moment, and with unit *0* dedicated to the `LastPass` texture, it means units 1-4 will be used for material map textures. This leaves units 5 and up free for other samplers.

Another `for` loop is entered, this time using the `LightCount` variable for maximum value. We've already determined how many lights we're dealing with during the shadow pass, so we don't need to make that check again here.

A reference to a light is fetched and passed into the `SubmitLightUniforms()` method, along with the light pass shader handle and the light number currently being used. The cubemap texture for that specific light is then bound for sampling. Note the use of `BaseCubeMapUnit + lightID`. This ensures that each light gets its own texture unit.

Inside the light pass shader, the shadow map samplers are going to be stored inside an array. Because of this, a string name for each element of the array is constructed based on the current light ID we're working with, and the uniform for the texture unit is sent to the shader.

Finally, because all of the uniforms and textures are properly bound and updated, we can actually invoke the light-pass shader by rendering `m_fullScreenQuad`:

[PRE58]

First, the FBO is bound to the handle of the current texture being used as a buffer. The quad itself is then bound, rendered, and unbound again. This is all we need to redraw the entire finished buffer texture to the current buffer texture, so the FBO is unbound. The 2D textures are also unbound at this point, since the light pass shader has just commenced executing.

Speaking of unbinding, all of these cubemap textures need to be unbound as well:

[PRE59]

At this point, the very last thing left to do inside the lighting pass loop is to swap the buffer textures inside the `Renderer` class:

[PRE60]

This makes sure the most recent buffer is always stored as the finished texture.

Finally, once the light passes have, commenced, we must clean up the state of everything and actually render the finished buffer texture:

[PRE61]

The shader program is first reset to whatever it was before the light pass was executed. The SFML window itself has its OpenGL states reset, because our use of OpenGL functions most likely altered them. Afterwards, we obtain the current window view, reset the window to its default view, draw the buffer texture, and swap the previous view back, just as in [Chapter 8](ch08.html "Chapter 8.  Let There Be Light - An Introduction to Advanced Lighting") , *Let There Be Light! - An Introduction to Advanced Lighting*.

#### Submitting light uniforms to the shader

One more little piece of code we still haven't covered is the actual light uniform submission to the light pass shader:

[PRE62]

This chunk of code is pretty much exactly the same as in [Chapter 8](ch08.html "Chapter 8.  Let There Be Light - An Introduction to Advanced Lighting") , *Let There Be Light! - An Introduction to Advanced Lighting* *,* except it uses raw OpenGL functions to submit the uniforms.

#### The new and improved light pass shaders

Since the light pass had to be completely rewritten to use raw modern OpenGL, the shaders need to reflect those changes too. To begin with, the vertex shader is much simpler now, because it no longer uses outdated and deprecated ways of obtaining and transforming vertex information, texture coordinates, and so on:

[PRE63]

The position being passed to this shader is that of `m_fullScreenQuad`, so it's already in clip space. There's no reason to transform it. The texture coordinates are simply passed along to the fragment shader, where they get interpolated between vertices, ensuring sampling of every pixel:

[PRE64]

The fragment shader of the light pass has a couple of new values at the very top. We have a constant that's going to be used to offset the light's height, which we're going to cover very shortly. There's also the input value from the vertex shader of the texture coordinates we're going to need to sample. Lastly, we're using an array of `samplerCube` uniforms to access the shadow map information.

Let's take a look at the main body of the light pass fragment shader:

[PRE65]

Things have changed, yet oddly enough stayed the same. We're sampling all of the values from different textures just like before, only now we're using the `texCoords` variable passed down from the vertex shader.

Another small change is the pass number that gets checked for ambient lighting. It used to be *1* for clarity in the previous chapter. It's now changed to *0*.

Finally, the very reason we're here today the shadow calculations. A floating point value is obtained from the `CalculateShadow` function, that takes in coordinates of the current fragment, the position of the current light, and the number identifier of the current light as well. This value is later used when calculating the final pixel color. The pixel is simply multiplied by `ShadowValue` at the end, which determines how much in the shadow it is.

This function is for calculating the shadow value of a fragment that is implemented at the top of the shader as follows:

[PRE66]

Looks simple enough, right? Well, it is. First, the light's height is offset by the height offset constant we defined at the top of the shader. This is just a detail of further tweaking that ensures lighting looks as good as it can, and could be completely changed. The current value simply looks better than the default 0.

The difference between the fragment's position and the light's position is then calculated by subtracting one from the other. The order matters here because this is going to be used as a directional vector to determine which face of the cubemap texture should be sampled.

### Note

Keep in mind that our fragment and light positions use the *Z* component as the height. This effectively makes *Y* the depth axis, which can be visualized as the direction to and from the screen, as opposed to left/right for *X*, and up/down for *Z*.

The `currentDepth` variable is the distance from the light to the fragment being sampled. The *Y* component of the difference vector is then inverted, because in the right-hand coordinate system OpenGL uses, pointing towards the screen means going into the negatives.

Now it's time to actually sample the shadow map texture and obtain the nearest depth at that particular fragment. This is done by passing the difference vector as a directional vector. Don't worry about it not being normalized, because it doesn't have to be. Also note the *Z* and *Y* components swapped. Again, we use *Z* for height, while OpenGL uses *Y*. Finally, we check whether the depth between the fragment and the light is greater than the depth sampled from the current shadow map, and if it is, it means the fragment is in the shadow. 0 could be returned, but in order to create shadows that slowly fade out with distance, `nearestDepth` is returned instead. This is the value that the final pixel gets multiplied by, and because it's in the range *[0;1]*, we get the linear fade with distance.

### Note

Note `nearestDepth` being multiplied by the light radius, which represents the frustum far value, when it's being checked. This transforms it from the range *[0;1]*, to the actual distance at which the shadow primitive is away from the light.

Consider the following diagram to help get the point across:

![The new and improved light pass shaders](img/image_09_010.jpg)

Here, the main arrow from the sample point to the light is `currentDepth`, and the `nearestDepth` after being multiplied by the light's radius is the arrow from the black box in the middle to the light.

# Adding shadow casters to entities

Now that we have all of the rendering resolved, we still need to make sure entities can cast shadows. This will be achieved by actually attaching special components to entities that will hold pointers to 3D geometry used during shadow pass. This geometry will obviously need to be updated to match the position of the entities it represents, which is why the component data is going to be accompanied by a separate system, used to actually keep everything synced up.

## Adding the shadow caster component

First, because our entities exist within the ECS paradigm, we need to add a component that represents the shadow volume of an entity:

[PRE67]

This component will be used to load entity shadow caster primitives from the entity file, as well as update their respective `ShadowCaster` instances. The player entity file, for example, would look like this with the new component added:

[PRE68]

## Creating the shadow system

Updating these components should be done in a separate, designated system for this very purpose. Because we've done this so many times before, let's just take a look at the relevant parts of the code:

[PRE69]

The constructor of this system simply sets up the entity requirements to belong here. It requires the position and shadow caster components, obviously.

Updating these components is equally as easy:

[PRE70]

For each entity that belongs to this system, the position and shadow caster components are obtained. The shadow caster's `UpdateCaster` method is then invoked, with the 2D position and height being passed in. The constant value of `8.f` is simply used to offset the shadow primitive in order to center it properly.

### Note

Note that the *Y* and *Z* values are, once again, swapped around.

Finally, because we want to properly emplace and manage unique shadow caster prototypes in the light manager, the shadow system must implement a method that will be called when the entity has finished loading and is about to be added, in order to set everything up properly:

[PRE71]

Once the shadow caster component is retrieved, the entity type name is obtained from the entity manager. This is simply the name of the entity prototype, such as player, skeleton, and so on. The primitive prototype with the appropriate name is then attempted to be added, and should there be an exact same shadow caster prototype already in `LightManager`, that name is returned instead. The shadow caster itself is then created, passed on to the `C_ShadowCaster` component, and scaled to a decent size. For the time being, this is a constant value, but it can obviously be made to change depending on the entity type, if it's stored inside the entity file along with the rest of the component data.

# Integrating the changes made

Finally, all we have left to do in order to make this work is add the newly created component and system types to the ECS:

[PRE72]

The shadow system itself also needs a pointer to the light manager for obvious reasons. Running the game now, with all of the lights properly set up and shadow casters correctly loaded, we should have three-dimensional shadows!

![Integrating the changes made](img/image_09_012.jpg)

Because the entities can hop elevations, the lights can be made to change their heights, and the actual light pass of the scene incorporates different heights of tile layers. Moving the lights around actually creates results in three-dimensional space, allowing the shadows to flow across walls, if at a right angle. After all of that hard work, the effect is absolutely astonishing!

# Potential issues and how to address them

Although we aren't facing any of these issues at this very point, most 3D games will have to deal with them as soon as basic shadows are established using this method.

**Shadow acne** is a graphical artefact that can be summarized as horrible *tearing*, where lit areas are horribly defaced with dark and white lines closely nested together. This happens because shadow maps are of finite size and pixels that are right next to each other will end up spanning a small distance on actual, *real* geometry being shaded. It can be fixed by simply adding or subtracting a simple *bias* floating point value to or from the shadow map's depth sample inside the light pass shader. This floating point value would, ideally, not be a constant and instead depend on the slope between the point on the geometry and the light.

**Peter panning** can be described as shadows that appear to be *floating* away from the geometry that casts them. Adding the floating point bias to fix shadow acne will usually make this problem worse, especially when dealing with incredibly thin geometry. A common and easy fix for this problem is simply avoiding thin geometry and using front face culling during the shadow pass, as we did.

## Percentage closer filtering

You may have noticed that the shadows produced by our geometry are rather hard and don't exactly smooth out around the edges. As always, there is a solution that will resolve this, and it involves sampling the shadow map a couple more times per pixel.

By sampling not only the calculated pixel of the shadow map, but also the surrounding ones, we can easily take an average value of all of them and use it to *smooth* out the edge. If, for example, our sampled pixel is in the shadow but *50%* of all other sampled pixels around it are lit up, the center pixel itself should only be *50%* opaque. By eliminating this binary rule of a pixel either being completely lit or completely dark, we can successfully implement soft shadows using this technique. Higher numbers of surrounding pixels will obviously yield smoother results, but will also bog down performance.

# Summary

Congratulations on making it to the end of this chapter! Although it took quite a while to re-architect our lighting engine, the results cannot be dismissed as miniscule. The shadows created by this method add a lot of graphical diversity to our world. In the next chapter, we're going to be discussing optimizations that can be applied to make the game run as fast as it possibly can after all of the fancy, clock cycle sucking techniques used throughout this book. See you there!