# Advanced Rendering Techniques

In this chapter, we will cover the following recipes:

*   Drawing a skybox
*   Drawing billboards using geometry shaders
*   Drawing particles using compute and graphics pipelines
*   Rendering a tessellated terrain
*   Rendering a full-screen quad for post-processing
*   Using input attachments for a color correction post-process effect

# Introduction

Creating 3D applications, such as games, benchmarks or CAD tools, usually requires, from a rendering perspective, the preparation of various resources, including meshes or textures, drawing multiple objects on the scene, and implementing algorithms for object transformations, lighting calculations, and image processing. They all can be developed in any way we want, in a way that is most suitable for our purpose. But there are also many useful techniques that are commonly used in the 3D graphics industry. Descriptions for these can be found in books and tutorials with examples implemented using various 3D graphics APIs.

Vulkan is still a relatively new graphics API, so there aren't too many resources that present common rendering algorithms implemented with the Vulkan API. In this chapter, we will learn how to use Vulkan to prepare various graphics techniques. We will learn about important concepts from a collection of popular, advanced rendering algorithms found in games and benchmarks and how they match with the Vulkan resources.

In this chapter, we will focus only on the code parts that are important from the perspective of a given recipe. Resources that are not described (for example, command pool or render pass creation) are created as usual (refer to the *Rendering a geometry with a vertex diffuse lighting* recipe from [Chapter 11](45108a92-6d49-4759-9495-3f1166e69128.xhtml), *Lighting*).

# Drawing a skybox

Rendering 3D scenes, especially open world ones with vast viewing distances, requires many objects to be drawn. However, the processing power of current graphics hardware is still too limited to render as many objects as we see around us every day. So, to lower the number of drawn objects and to draw the background for our scene, we usually prepare an image (or a photo) of distant objects and draw just the image instead.

In games where players can freely move and look around, we can't draw a single image. We must draw images in all directions. Such images form a cube, and an object on which background images are placed is called a skybox. We render it in such a way that it is always in the background, at the furthest depth value available.

# Getting ready

Drawing a skybox requires the preparation of a cubemap. It contains six square images containing a view in all world directions (right, left, up, down, backward, forward), as in the following image:

![](img/image_12_001.png)

Images courtesy of Emil Persson ([h t t p ://w w w . h u m u s . n a m e](http://www.humus.name))

In Vulkan, cubemaps are special image views created for images with six array layers (or a multiple of six). Layers must contain images in the *+X*, -*X*, *+Y*, -*Y*, *+Z*, -*Z* order.

Cubemaps can be used not only for drawing skyboxes. We can use them to draw reflections or transparent objects. They can be used for lighting calculations as well (refer to the *Drawing a reflective and refractive geometry using cubemaps* recipe in Chapter 11, *Lighting*).

# How to do it...

1.  Load a 3D model of a cube from a file and store vertex data in a vertex buffer. Only vertex positions are required (refer to the *Loading a 3D model from an OBJ file* recipe in [Chapter 10](1b6b28e0-2101-47a4-8551-c30eb9bfb573.xhtml), *Helper Recipes*).

2.  Create a combined image sampler with a square `VK_IMAGE_TYPE_2D` image that has six array layers (or a multiple of six), a sampler that uses a `VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE` addressing mode for all coordinates and a `VK_IMAGE_VIEW_TYPE_CUBE` image view (refer to the *Creating a combined image sampler* recipe in [Chapter 5](fe2cb528-9d22-49db-a05b-372bce2f87ee.xhtml), *Descriptor Sets*).
3.  Load image data for all six sides of a cube and upload it to the image's memory using a staging buffer. Image data must be uploaded to six array layers in the following order: *+X*, -*X*, *+Y*, -*Y*, *+Z*, -*Z* (refer to the *Loading texture data from a file* recipe in [Chapter 10](1b6b28e0-2101-47a4-8551-c30eb9bfb573.xhtml), Helper Recipes, and to the *Using a staging buffer to update an image with a device-local memory bound* recipe in [Chapter 4](f1332ca0-b5a2-49bd-ac41-e37068e31042.xhtml), *Resources and Memory*).
4.  Create a uniform buffer in which transformation matrices will be stored (refer to the *Creating a uniform buffer* recipe in [Chapter 5](fe2cb528-9d22-49db-a05b-372bce2f87ee.xhtml), *Descriptor Sets*).
5.  Create a descriptor set layout with the uniform buffer accessed by a vertex stage and a combined image sampler accessed by a fragment stage. Allocate a descriptor set using the preceding layout. Update the descriptor set with the uniform buffer and the cubemap/combined image sampler (refer to the *Creating a descriptor set layout, Allocating descriptor sets* and *Updating descriptor sets* recipes in [Chapter 5](fe2cb528-9d22-49db-a05b-372bce2f87ee.xhtml), *Descriptor Sets*).
6.  Create a shader module with a vertex shader created from the following GLSL code (refer to the *Creating a shader module recipe* in [Chapter 8](5744ea05-b18a-4f84-a1df-250b549dfea5.xhtml), *Graphics and Compute Pipelines*):

[PRE0]

7.  Create a shader module with a fragment shader created from the following GLSL code:

[PRE1]

8.  Create a graphics pipeline from the preceding modules with vertex and fragment shaders. The pipeline should use one vertex attribute with three components (vertex positions) and a `VK_CULL_MODE_FRONT_BIT` value for the rasterization state's culling mode. Blending should be disabled. The pipeline's layout should allow access to the uniform buffer and the cubemap/combined image sampler (refer to the *Specifying pipeline shader stages*, *Specifying pipeline vertex input state*, *Specifying pipeline rasterization state*, *Specifying pipeline blend state*, *Creating a pipeline layout*, *Specifying graphics pipeline creation parameters* and *Creating a graphics pipeline* recipes from [Chapter 8](5744ea05-b18a-4f84-a1df-250b549dfea5.xhtml), *Graphics and Compute Pipelines*).
9.  Draw the cube with the rest of a rendered geometry (refer to the *Binding descriptor sets* recipe from [Chapter 5](fe2cb528-9d22-49db-a05b-372bce2f87ee.xhtml), *Descriptor Sets*, to the *Binding a pipeline object* recipe from [Chapter 8](5744ea05-b18a-4f84-a1df-250b549dfea5.xhtml), *Graphics and Compute Pipelines* and to the *Binding vertex buffers* and *Drawing a geometry* recipes from [Chapter 9](0a69f5b5-142e-422b-aa66-5cb09a6467b3.xhtml), *Command Recording and Drawing*).
10.  Update a model view matrix in the uniform buffer each time a user (a camera) moves in the scene. Update a projection matrix in the uniform buffer each time the application window is resized.

# How it works...

To render a skybox we need to load or prepare a geometry forming a cube. Only positions are required as they can also be used for texture coordinates.

Next, we load six cubemap images and create a combined image sampler with a cube image view:

[PRE2]

The created cube image view, along with a sampler, is then provided to the shaders through a descriptor set. We also need a uniform buffer in which transformation matrices will be stored and accessed in the shaders:

[PRE3]

To draw a skybox we don't need a separate, dedicated *render pass*, as we can render it along the normal geometry. What's more, to save processing power (image fill rate), we usually draw a skybox after the (opaque) geometry and before the transparent objects. It is rendered in such a way so that its vertices are always at the far clipping plane. This way it doesn't cover geometry that had been already drawn and doesn't get clipped away either. This effect is achieved with a special vertex shader. Its most important part is the following code:

[PRE4]

First, we multiply the position by a modelview matrix. We take only the rotation part of the matrix. A player should always be in the center of the skybox, or the illusion will be broken. That's why we don't want to move the skybox, we need only to rotate it as a response to the player looking around.

Next, we multiply the viewspace position of a vertex by a projection matrix. The result is stored in a 4-element vector, with the last two components being the same and equal to the z component of the result. In modern graphics hardware, a perspective projection is performed by dividing the position vector by its `w` component. After that, all vertices, whose `x` and `y` components fit into the <`-1, 1`> range (inclusive) and `z` component fits into the <`0, 1`> range (inclusive), are inside the clipping volume and are visible (unless they are obscured by something else). So, calculating the vertex position in a way that makes its last two components equal, guarantees that the vertex will lie on the far clipping plane.

Apart from the vertex shader and a cube image view, skybox needs only one additional special treatment. We need to remember polygon facingness. Usually, we draw geometry with backface culling, as we want to see its external surface. For the skybox, we want to render its internal surface, because we look at it from the inside. That's why, if we don't have a mesh prepared especially for the skybox, we probably want to cull front faces during skybox rendering. We can prepare the pipeline rasterization info like this:

[PRE5]

Apart from that, the graphics pipeline is created in the usual way. To use it for drawing, we need to bind the descriptor set, the vertex buffer, and the pipeline itself:

[PRE6]

The following images have been generated using this recipe:

![](img/image_12_002.png)

# See also

*   In [Chapter 5](fe2cb528-9d22-49db-a05b-372bce2f87ee.xhtml), *Descriptor Sets*, see the following recipes:
    *   *Creating a combined image sampler*
    *   *Creating a descriptor set layout*
    *   *Allocating descriptor sets*
    *   *Updating descriptor sets*
    *   *Binding descriptor sets*
*   In [Chapter 8](5744ea05-b18a-4f84-a1df-250b549dfea5.xhtml), *Graphics and Compute Pipelines*, see the following recipes:
    *   *Creating a shader module*
    *   *Specifying pipeline shader stages*
    *   *Creating a graphics pipeline*
    *   *Binding a pipeline object*
*   In [Chapter 9](0a69f5b5-142e-422b-aa66-5cb09a6467b3.xhtml), *Command Recording and Drawing*, see the following recipes:
    *   *Binding vertex buffers*
    *   *Drawing a geometry*
*   In [Chapter 10](1b6b28e0-2101-47a4-8551-c30eb9bfb573.xhtml), *Helper Recipes*, see the following recipes:
    *   *Loading texture data from a file*
    *   *Loading a 3D model from an OBJ file*
*   In [Chapter 11](45108a92-6d49-4759-9495-3f1166e69128.xhtml), *Lighting*, see the following recipe:
    *   *Drawing a reflective and refractive geometry using cubemaps*

# Drawing billboards using geometry shaders

Simplifying geometry drawn in a distance is a common technique for lowering the processing power needed to render the whole scene. The simplest geometry that can be drawn is a flat quad (or a triangle) with an image depicting the look of an object. For the effect to be convincing, the quad must always be facing camera:

![](img/image_12_003.png)

Flat objects that are always facing camera are called billboards. They are used not only for distant objects as the lowest level of detail of a geometry, but also for particle effects.

One straightforward technique for drawing billboards is to use geometry shaders.

# How to do it...

1.  Create a logical device with the `geometryShader` feature enabled (refer to the *Getting features and properties of a physical device* and *Creating a logical device* recipes from [Chapter 1](d10e8284-6122-4d0a-8f86-ab0bc0bba47e.xhtml), *Instance and Devices*).
2.  Prepare positions for all billboards with one vertex per single billboard. Store them in a vertex buffer (refer to the *Creating a buffer* recipe from [Chapter 4](f1332ca0-b5a2-49bd-ac41-e37068e31042.xhtml), *Resources and Memory*).
3.  Create a uniform buffer for at least two 4x4 transformation matrices (refer to the *Creating a uniform buffer* recipe from [Chapter 5](fe2cb528-9d22-49db-a05b-372bce2f87ee.xhtml), *Descriptor Sets*).
4.  If billboards should use a texture, create a combined image sampler and upload the texture data loaded from a file to the image's memory (refer to the *Creating a combined image sampler* recipe from [Chapter 5](fe2cb528-9d22-49db-a05b-372bce2f87ee.xhtml), *Descriptor Sets* and to the *Loading texture data from a file* recipe from [Chapter 10](1b6b28e0-2101-47a4-8551-c30eb9bfb573.xhtml), *Helper Recipes*).
5.  Prepare a descriptor set layout for a uniform buffer accessed by vertex and geometry stages and, if billboards need a texture, a combined image sampler accessed by a fragment shader stage. Create a descriptor set and update it with the created uniform buffer and the combined image sampler (refer to the *Creating a descriptor set layout*, *Allocating descriptor sets,* and *Updating descriptor sets* recipes from [Chapter 5](fe2cb528-9d22-49db-a05b-372bce2f87ee.xhtml), *Descriptor Sets*).
6.  Create a shader module with a vertex shader created from the following GLSL code (refer to the *Creating a shader module* recipe from [Chapter 8](5744ea05-b18a-4f84-a1df-250b549dfea5.xhtml), *Graphics and Compute Pipelines*):

[PRE7]

7.  Create a shader module containing a geometry shader created from the following GLSL code:

[PRE8]

8.  Create a shader module with a fragment shader that uses a SPIR-V assembly generated from the following GLSL code:

[PRE9]

9.  Create a graphics pipeline. It must use shader modules with the preceding vertex, geometry and fragment shaders. Only one vertex attribute (a position) is needed. It will be used to draw geometry using a `VK_PRIMITIVE_TOPOLOGY_POINT_LIST` primitive. The pipeline should have access to the uniform buffer with transformation matrices and (if needed) a combined image texture (refer to the *Specifying pipeline shader stages*, *Specifying pipeline vertex input state*, *Specifying pipeline input assembly state*, *Creating a pipeline layout*, *Specifying graphics pipeline creation parameters*, and *Creating a graphics pipeline* recipes from [Chapter 8](5744ea05-b18a-4f84-a1df-250b549dfea5.xhtml), *Graphics and Compute Pipelines*).
10.  Draw the geometry inside a render pass (refer to the *Binding descriptor sets* recipe from [Chapter 5](fe2cb528-9d22-49db-a05b-372bce2f87ee.xhtml), *Descriptor Sets*, to the *Binding a pipeline object* recipe from [Chapter 8](5744ea05-b18a-4f84-a1df-250b549dfea5.xhtml), *Graphics and Compute Pipelines*, and to the *Binding vertex buffers* and *Drawing a geometry recipes* from [Chapter 9](0a69f5b5-142e-422b-aa66-5cb09a6467b3.xhtml), *Command Recording and Drawing)*.
11.  Update a modelview matrix in the uniform buffer each time the user (a camera) moves in the scene. Update a projection matrix in the uniform buffer each time the application window is resized.

# How it works...

First, we start by preparing the positions for billboards. Billboards are drawn as point primitives, so one vertex corresponds to a one billboard. How we prepare the geometry is up to us and we don't need other attributes. A geometry shader converts a single vertex into a camera-facing quad and calculates texture coordinates.

In this example we don't use a texture, but we will use texture coordinates to draw circles. All we need to access are transformation matrices stored in a uniform buffer generated like this:

[PRE10]

The next step is to create a graphics pipeline. It uses a single vertex attribute (a position) defined in the following way:

[PRE11]

We draw vertices as points, so we need to specify an appropriate primitive type during the pipeline creation:

[PRE12]

The rest of the pipeline parameters are fairly typical. The most important parts are the shaders.

A vertex shader transforms the vertex from the local space to the view space. Billboards must always face the camera, so it is easier to perform calculations directly in the view space.

A geometry shader does almost all the work. It takes one vertex (a point) and emits a triangle strip with four vertices (a quad). Each new vertex is offset a bit to the left/right and up/down to form a quad:

![](img/image_12_004.png)

Additionally, a texture coordinate is assigned to the generated vertex based on the direction/offset. In our example, the first vertex is prepared like this:

[PRE13]

The remaining vertices are emitted in a similar way. As we transformed vertices to the view space in the vertex shader, the generated quad is always facing the screen plane. All we need to do is to multiply generated vertices by a projection matrix to transform them to the clip space.

A fragment shader is used to discard some fragments to form a circle from the quad:

[PRE14]

In the following example, we can see billboards rendered in the positions of a mesh's vertices. The circles seen in the image are flat; they are not spheres:

![](img/image_12_005.png)

# See also

*   In [Chapter 1](d10e8284-6122-4d0a-8f86-ab0bc0bba47e.xhtml), *Instances and Devices*, see the following recipes:
    *   *Getting features and properties of a physical device*
    *   *Creating a logical device*
*   In [Chapter 5](fe2cb528-9d22-49db-a05b-372bce2f87ee.xhtml), *Descriptor Sets*, see the following recipes:
    *   *Creating a uniform buffer*
    *   *Creating a descriptor set layout*
    *   *Allocating descriptor sets*
    *   *Updating descriptor sets*
    *   *Binding descriptor sets*
*   In [Chapter 7](97217f0d-bed7-4ae1-a543-b4d599f299cf.xhtml), *Shaders*, see the *Writing geometry shaders* recipe
*   In [Chapter 8](5744ea05-b18a-4f84-a1df-250b549dfea5.xhtml), *Graphics and Compute Pipelines*, see the following recipes:
    *   *Creating a shader module*
    *   *Specifying pipeline shader stages*
    *   *Specifying a pipeline vertex binding description, attribute description, and input state*
    *   *Specifying a pipeline input assembly state*
    *   *Creating a pipeline layout*
    *   *Specifying graphics pipeline creation parameters*
    *   *Creating a graphics pipeline*
    *   *Binding a pipeline object*
*   In [Chapter 9](0a69f5b5-142e-422b-aa66-5cb09a6467b3.xhtml), *Command Recording and Drawing*, see the following recipes:
    *   *Binding vertex buffers*
    *   *Drawing geometry recipes*

# Drawing particles using compute and graphics pipelines

Due to the nature of graphics hardware and the way objects are processed by the graphics pipeline, it is quite hard to display phenomena such as clouds, smoke, sparks, fire, falling rain, and snow. Such effects are usually simulated with particle systems, which are a large number of small sprites that behave according to the algorithms implemented for the system.

Because of the very large number of independent entities, it is convenient to implement the behavior and mutual interactions of particles using compute shaders. Sprites mimicking the look of each particle are usually displayed as billboards with geometry shaders.

In the following example, we can see an image generated with this recipe:

![](img/image_12_006.png)

# How to do it...

1.  Create a logical device with the `geometryShader` feature enabled. Request a queue that supports graphics operations and a queue that supports compute operations (refer to the *Getting features and properties of a physical device* and *Creating a logical device* recipes from [Chapter 1](d10e8284-6122-4d0a-8f86-ab0bc0bba47e.xhtml), *Instance and Devices*).
2.  Generate the initial data (attributes) for a particle system.
3.  Create a buffer that will serve both as a vertex buffer and a storage texel buffer. Copy the generated particle data to the buffer (refer to the *Creating a storage texel buffer* recipe from [Chapter 5](fe2cb528-9d22-49db-a05b-372bce2f87ee.xhtml), *Descriptor Sets*, and to the *Using staging buffer to update buffer with a device-local memory bound* recipe from [Chapter 4](f1332ca0-b5a2-49bd-ac41-e37068e31042.xhtml), *Resources and Memory*).
4.  Create a uniform buffer for two transformation matrices. Update it each time the camera is moved or the window is resized (refer to the *Creating a uniform buffer* recipe from [Chapter 5](fe2cb528-9d22-49db-a05b-372bce2f87ee.xhtml), *Descriptor Sets*).
5.  Create two descriptor set layouts: one with a uniform buffer accessed by vertex and geometry stages; and the second with a storage texel buffer accessed by a compute stage. Create a descriptor pool and allocate two descriptor sets using the above layouts. Update them with the uniform buffer and the storage texel buffer (refer to the *Creating a descriptor set layout, Creating a descriptor pool*, *Allocating descriptor sets* and *Updating descriptor sets*, recipes from [Chapter 5](fe2cb528-9d22-49db-a05b-372bce2f87ee.xhtml), *Descriptor Sets*).
6.  Create a shader module with a compute shader created from the following GLSL code (refer to the *Creating a shader module* recipe from [Chapter 8](5744ea05-b18a-4f84-a1df-250b549dfea5.xhtml), *Graphics and Compute Pipelines*):

[PRE15]

7.  Create a compute pipeline that uses the shader module with the compute shader and has access to the storage texel buffer and a push constant range with one floating point value (refer to the *Specifying pipeline shader stages*, *Creating a pipeline layout recipe*, and *Creating a compute pipeline* recipes from [Chapter 8](5744ea05-b18a-4f84-a1df-250b549dfea5.xhtml), *Graphics and Compute Pipelines*).
8.  Create a graphics pipeline with vertex, geometry, and fragment shaders as described in the *Drawing billboards using geometry shaders* recipe. The graphics pipeline must fetch two vertex attributes, draw vertices as `VK_PRIMITIVE_TOPOLOGY_POINT_LIST` primitives and must have blending enabled (refer to the *Specifying pipeline vertex input state*, *Specifying pipeline input assembly state*, *Specifying pipeline blend state*, and *Creating a graphics pipeline* recipes from [Chapter 8](5744ea05-b18a-4f84-a1df-250b549dfea5.xhtml), *Graphics and Compute Pipelines*).
9.  To render a frame, record a command buffer that dispatches compute work and submit it to a queue that supports compute operations. Provide a semaphore to be signaled when the queue finishes processing the submitted command buffer (refer to the *Providing data to shaders through push* *constants* and *Dispatching a compute work recipes* from [Chapter 9](0a69f5b5-142e-422b-aa66-5cb09a6467b3.xhtml), *Command Recording*, and the *Submitting command buffers to the queue* recipe from [Chapter 3](fc38e0ae-51aa-4f6f-8fb3-551861273018.xhtml), *Command Buffers and Synchronization*).
10.  Also, in each frame record a command buffer that draws billboards as described in the *Drawing billboards using geometry shaders* recipe. Submit it to the queue that supports graphics operations. During submission, provide a semaphore, which is signaled by the compute queue. Provide it as a wait semaphore (refer to the *Synchronizing two command buffers* recipe from [Chapter 3](fc38e0ae-51aa-4f6f-8fb3-551861273018.xhtml), *Command Buffers and Synchronization*).

# How it works...

Drawing particle systems can be divided in two steps:

*   We calculate and update positions of all particles with compute shaders
*   We draw particles in updated positions using graphics pipeline with vertex, geometry, and fragment shaders

To prepare a particle system, we need to think about the data needed to calculate positions and draw all particles. In this example we will use three parameters: position, speed and color. Each set of these parameters will be accessed by a vertex shader through a vertex buffer, and the same data will be read in a compute shader. A simple and convenient way to access a very large number of entries in stages other than the vertex shader is to use a texel buffer. As we want to both read and store data, we will need a storage texel buffer. It allows us to fetch data from a buffer treated as a 1-dimensional image (refer to the *Creating a storage texel buffer* recipe from [Chapter 5](fe2cb528-9d22-49db-a05b-372bce2f87ee.xhtml), *Descriptor Sets*).

First, we need to generate initial data for our particle system. For the data available in a storage texel buffer to be properly read, it must be stored according to a selected format. Storage texel buffers have a limited set of formats that are mandatory, so we need to pack the parameters of our particles to one of them. Positions and colors require at least three values each. In our example, particles will move around the center of the whole system, so the velocity can be easily calculated based on the particle's current position. We just need to differentiate the speed of particles. For this purpose one value for scaling our velocity vector is enough.

So we end up with seven values. We will pack them into two RGBA vectors of floating-point values. First we have three `X`, `Y`, `Z` components of a position attribute. The next value is unused in our particle system, but for the data to be correctly read, it needs to be included. We will store a `1.0f` value as a fourth component of the position attribute. After that there are `R`, `G`, `B` values for a color, and a value scaling the speed vector of a particle. We randomly generate all values and store them in a vector:

[PRE16]

The generated data is copied to the buffer. We create a buffer that will serve both as a vertex buffer during rendering and as a storage texel buffer during position calculations:

[PRE17]

Additionally, we need a uniform buffer, through which we will provide transformation matrices. A uniform buffer along the storage texel buffer will be provided to shaders through descriptor sets. Here we will have two separate sets. In the first set, we will have only a uniform buffer accessed by vertex and geometry shaders. The second descriptor set is used in a compute shader to access storage texel buffer. For this purpose we need two separate descriptor set layouts:

[PRE18]

Next, we need a pool from which we can allocate two descriptor sets:

[PRE19]

After that, we can allocate two descriptors sets and update them with the created buffer and buffer view:

[PRE20]

The next important step is the creation of graphics and compute pipelines. When a movement is involved, calculations must be performed based on real-time values, as we usually cannot rely on fixed time intervals. So the compute shader must have access to a value of time that has elapsed since the last frame. Such a value may be provided through a push constant range. We can see the code required to create the compute pipeline here:

[PRE21]

Compute shaders read data from the storage texel buffer defined as follows:

[PRE22]

Data from the storage texel buffer is read using the `imageLoad()` function:

[PRE23]

We read two values so we need two `imageLoad()` calls, because each such operation returns one element of a format defined for the buffer (in this case, a 4-component vector of floats). We access the buffer based on a unique value of a current compute shader instance.

Next, we perform calculations and update the positions of the vertices. Calculations are performed so the particles move around the center of the scene based on the position and an up vector. A new vector (speed) is calculated using the `cross()` function:

![](img/image_12_007.png)

This calculated speed vector is added to the fetched position and the result is stored in the same buffer using the `imageStore()` function:

[PRE24]

We don't update a color or speed, so we store only one value.

Because we access the data of only one particle, we can read values from and store values in the same buffer. In more complicated scenarios, such as when there are interactions between particles, we can't use the same buffer. The order in which compute shader invocations are executed is unknown, so we would end up with some invocations accessing unmodified values, but others would read data that has already been updated. This would impact the accuracy of performed calculations and probably result in an unpredictable system.

Graphics pipeline creation is very similar to the one presented in the *Drawing billboards using geometry shaders* recipe. The difference is that it fetches two attributes instead of one:

[PRE25]

We also render vertices as point primitives:

[PRE26]

One last difference is that here we enable additive blending, so the particles look like they are glowing:

[PRE27]

The drawing process is also divided into two steps. First, we record a command buffer that dispatches compute work. Some hardware platforms may have a queue family that is dedicated to math calculations, so it may be preferable to submit command buffers with compute shaders to that queue:

[PRE28]

Drawing is performed in the normal way. We just need to synchronize the graphics queue with a compute queue. We do this by providing an additional wait semaphore when we submit a command buffer to the graphics queue. This semaphore must be signaled by a compute queue when it finishes processing the submitted command buffer in which the
compute shaders are dispatched.

The following sample images show the same particle system rendered with different numbers of particles:

![](img/image_12_008.png)

# See also

*   In [Chapter 1](d10e8284-6122-4d0a-8f86-ab0bc0bba47e.xhtml), *Instances and Devices*, see the Getting features and properties of a physical device recipe
*   In [Chapter 5](fe2cb528-9d22-49db-a05b-372bce2f87ee.xhtml), *Descriptor Sets*, see the following recipes:
    *   *Creating a storage texel buffer*
    *   *Creating a descriptor set layout*
    *   *Creating a descriptor pool*
    *   *Allocating descriptor sets*
    *   *Updating descriptor sets*
*   In [Chapter 7](97217f0d-bed7-4ae1-a543-b4d599f299cf.xhtml), *Shaders*, see the Writing compute shaders recipe
*   In [Chapter 8](5744ea05-b18a-4f84-a1df-250b549dfea5.xhtml), *Graphics and Compute Pipelines*, see the following recipes:
    *   *Creating a shader module*
    *   *Creating a compute pipeline*
    *   *Creating a graphics pipeline*
*   In [Chapter 9](0a69f5b5-142e-422b-aa66-5cb09a6467b3.xhtml), *Command Recording and Drawing*, see the following recipes:
    *   *Providing data to shaders through push constants*
    *   *Drawing a geometry*
    *   *Dispatching a compute work*

# Rendering a tessellated terrain

3D scenes with open worlds and long rendering distances usually also contain vast terrains. Drawing ground is a very complex topic and can be performed in many different ways. Terrain in a distance cannot be too complex, as it will take up too much memory and processing power to display it. On the other hand, the area near the player must be detailed enough to look convincing and natural. That's why we need a way to lower the number of details with increasing distance or to increase the terrain's fidelity near the camera.

This is an example of how the tessellation shaders can be used to achieve high quality rendered images. For a terrain, we can use a flat plane with low number of vertices. Using tessellation shaders, we can increase the number of primitives of the ground near the camera. We can then offset generated vertices by the desired amount to increase or decrease the height of a terrain.

The following screenshot is an example of an image generated using this recipe:

![](img/image_12_009.png)

# Getting ready

Drawing a terrain usually requires the preparation of height data. This can be generated on the fly, procedurally, according to some desired formulae. However, it can also be prepared earlier in the form of a texture called a height map. It contains information about the terrain's height above (or below) a specified altitude, in which a lighter color indicates a greater height and a darker color indicates a lower height. An example of such a height map can be seen in the following image:

![](img/image_12_010.png)

# How to do it...

1.  Load or generate a model of a flat, horizontally-aligned plane. Two attributes--position and texture coordinate--will be needed. Upload the vertex data to a vertex buffer (refer to the *Loading a 3D model from an OBJ file* recipe from [Chapter 10](1b6b28e0-2101-47a4-8551-c30eb9bfb573.xhtml), *Helper Recipes* and to the *Creating a buffer* and *Using staging buffer to update buffer with a device-local memory bound* recipes from [Chapter 4](f1332ca0-b5a2-49bd-ac41-e37068e31042.xhtml), *Resources and Memory*).
2.  Create a uniform buffer for two transformation matrices (refer to the *Creating a uniform buffer* recipe from [Chapter 5](fe2cb528-9d22-49db-a05b-372bce2f87ee.xhtml), *Descriptor Sets*).
3.  Load height information from an image file (refer to the *Loading texture data from a file* recipe from [Chapter 10](1b6b28e0-2101-47a4-8551-c30eb9bfb573.xhtml), *Helper Recipes*). Create a combined image sampler and copy the loaded height data to image's memory (refer to the *Creating a combined image sampler* recipe from [Chapter 5](fe2cb528-9d22-49db-a05b-372bce2f87ee.xhtml), *Descriptor Sets* and to the *Using the staging buffer to update an image with a device-local memory bound* recipe from [Chapter 4](f1332ca0-b5a2-49bd-ac41-e37068e31042.xhtml), *Resources and Memory*).
4.  Create a descriptor set layout with one uniform buffer accessed by tessellation control and geometry stages and one combined image sampler accessed by tessellation control and evaluation stages (refer to the *Creating a descriptor set layout* recipe from [Chapter 5](fe2cb528-9d22-49db-a05b-372bce2f87ee.xhtml), *Descriptor Sets*). Allocate a descriptor set using the prepared layout. Update it with the created uniform buffer and sampler and image view handles (refer to the *Allocating descriptor sets* and *Updating descriptor sets* recipes from [Chapter 5](fe2cb528-9d22-49db-a05b-372bce2f87ee.xhtml), *Descriptor Sets*).
5.  Create a shader module with a SPIR-V assembly for a vertex shader created from the following GLSL code (refer to the *Creating a shader module* recipe from [Chapter 8](5744ea05-b18a-4f84-a1df-250b549dfea5.xhtml), *Graphics and Compute Pipelines*):

[PRE29]

6.  Create a shader module for a tessellation control stage. Use the following GLSL code to generate a SPIR-V assembly from:

[PRE30]

7.  Create a shader module for a tessellation evaluation shader created from the following GLSL code:

[PRE31]

8.  Create a shader module for a geometry shader and use the following GLSL code:

[PRE32]

9.  Create a shader module that contains a source code of a fragment shader. Generate a SPIR-V assembly from the following GLSL code:

[PRE33]

10.  Create a graphics pipeline using the above five shader modules. The pipeline should fetch two vertex attributes: a 3-component position and a 2-component texture coordinate. It must use `VK_PRIMITIVE_TOPOLOGY_PATCH_LIST` primitives. A patch should consist of three control points (refer to the *Specifying pipeline input assembly state*, *Specifying pipeline tessellation state*, *Specifying graphics pipeline creation parameters*, and *Creating a graphics pipeline* recipes from [Chapter 8](5744ea05-b18a-4f84-a1df-250b549dfea5.xhtml), *Graphics and Compute Pipelines*).
11.  Create the remaining resources and draw the geometry (refer to the *Rendering a geometry with a vertex diffuse lighting* recipe from [Chapter 11](45108a92-6d49-4759-9495-3f1166e69128.xhtml), *Lighting*).

# How it works...

We start the process of drawing a terrain by loading a model of a flat plane. It may be a simple quad with a little bit more than four vertices. Generating too many vertices in a tessellation stage may be too expensive performance-wise, so we need to find a balance between the complexity of a base geometry and the tessellation factors. We can see a plane used as a base for the tessellated terrain in the following image:

![](img/image_12_011.png)

In this example, we will load height information from a texture. We do this in the same way as we load data from files. Then we create a combined image sampler and upload loaded data to its memory:

[PRE34]

A uniform buffer with transformation matrices is also required, so the vertices can be transformed from local space to a view space and to the clip space:

[PRE35]

The next step is to create a descriptor set for the uniform buffer and the combined image sampler. A uniform buffer is accessed in the tessellation control and geometry stages. Height information is read in the tessellation control and evaluation stages:

[PRE36]

Next, we can update the descriptor set with the uniform buffer handle and with sampler and image view handles as they don't change during the lifetime of our application (that is, we don't need to recreate them when the window size is modified).

[PRE37]

The next step is to create a graphics pipeline. This time we have a very complex pipeline with all five programmable graphics stages enabled:

[PRE38]

Why do we need all five stages? A vertex shader is always required. This time it only reads two input attributes (position and texcoord) and passes it further down the pipeline.

When tessellation is enabled, we need both the control and evaluation shader stages. The tessellation control shader, as the name suggests, controls the tessellation level of processed patches (the amount of generated vertices). In this recipe, we generate vertices based on the distance from the camera: the closer the vertices of a patch are to the camera, the more vertices are generated by the tessellator. This way, the terrain in the distance is simple and doesn't take much processing power to be rendered; but, the closer to the camera, the more complex the terrain becomes.

We can't choose one tessellation level for the whole patch (in this case a triangle). When two neighboring triangles are tessellated with different factors, different number of vertices will be generated on their common edge. Vertices from each triangle will be placed in different locations and they will be offset by different values. This will create holes in our ground:

![](img/image_12_012.png)

In the preceding image we see two triangles: the left formed from vertices L0-L1-L7, and right formed from vertices R0-R1-R4\. The other vertices are generated by the tessellator. Triangles share an edge: L1-L7 or R1-R4 (points L1 and R4 indicate the same vertex; similarly points L7 and R1 indicate the same vertex); but the edge is tessellated with different factors. This causes discontinuities (indicated by stripes) in the surface formed by the two triangles.

To avoid this problem, we need to calculate a tessellation factor for each triangle edge in such a way that it is fixed across triangles that share the same edge. In this example, we will calculate tessellation factors based on the distance of a vertex from the camera. We will do this for all vertices in a triangle. Then, for a given triangle edge, we will choose a greater tessellation factor that was calculated from one of the edge's vertices:

[PRE39]

In the preceding tessellation control shader code, we calculate a distance (squared) from all vertices to the camera. We need to offset positions by the amount read from the height map, so the whole patch is in the correct place and the distance is properly calculated.

Next, for all triangle edges, we take the smaller distance of edge's two vertices. As we want a tessellation factor to increase with decreasing distance, we need to invert the calculated factor. Here we take a hardcoded value of `20` and subtract a chosen distance value. As we don't want the tessellation factor to be smaller than `1.0`, we perform additional clamping.

The tessellation factor calculated like this exaggerates the effect of decreasing the number of generated vertices with increasing distance. This is done on purpose so that we can see how triangles are tessellated and how the number of details increases near the camera. However, in real-life examples we should prepare such a formula so that the effect is barely visible.

Next, a tessellation evaluation shader takes the weights of generated vertices to calculate a valid position of the new vertices. We do the same for texture coordinates, as we need to load height information from the height map:

[PRE40]

After the position of a new vertex is calculated, we need to offset it, so the vertex is placed at an appropriate height:

[PRE41]

The tessellation evaluation shader stage is followed by the geometry shader stage. We can omit it but here we use it to calculate the normal vector of the generated triangle. We take one normal vector for all the triangle's vertices, so we will perform a flat shading in this sample.

The normal vector is calculated with the `cross()` function, which takes two vectors and returns a vector that is perpendicular to those provided. We provide vectors forming two edges of a triangle:

[PRE42]

Finally, the geometry shader calculates the clip space positions of all vertices and emits them:

[PRE43]

To simplify the recipe, a fragment shader is also simple. It mixes three colors based on the height above ground: green for grass in the lower parts, grey/brown for rocks in the middle, and white for snow in mountain tops. It also performs simple lighting calculations using the diffuse/Lambert light model.

The preceding shaders form a graphics pipeline used to draw a tessellated terrain. During pipeline creation we must remember to think about primitive topology. Because of the enabled tessellation stages, we need to use a `VK_PRIMITIVE_TOPOLOGY_PATCH_LIST` topology. We also need to provide a tessellation state during pipeline creation. As we want to operate on triangles, we specify that a patch contains three control points:

[PRE44]

The remaining parameters used for pipeline creation are defined in the usual way. We also don't need to do anything special during rendering. We just draw a plane with the preceding graphics pipeline bound, and we should see a geometry resembling a terrain. We can see examples of results generated with this recipe in the following images:

![](img/image_12_013.png)

# See also

*   In [Chapter 4](f1332ca0-b5a2-49bd-ac41-e37068e31042.xhtml), *Resources and Memory*, see the *Creating a buffer* recipe
*   In [Chapter 5](fe2cb528-9d22-49db-a05b-372bce2f87ee.xhtml), *Descriptor Sets*, see the following recipes:
    *   *Creating a combined image sampler*
    *   *Creating a uniform buffer*
    *   *Creating a descriptor set layout*
    *   *Allocating descriptor sets*
    *   *Updating descriptor sets*
*   In [Chapter 7](97217f0d-bed7-4ae1-a543-b4d599f299cf.xhtml), *Shaders*, see the following recipes:
    *   *Writing tessellation control shaders*
    *   *Writing tessellation evaluation shaders*
    *   *Writing geometry shaders*
*   In [Chapter 8](5744ea05-b18a-4f84-a1df-250b549dfea5.xhtml), *Graphics and Compute Pipelines*, see the following recipes:
    *   *Creating a shader module*
    *   *Specifying the pipeline input assembly state*
    *   *Specifying the pipeline tessellation state*
    *   *Creating a graphics pipeline*
*   In [Chapter 10](1b6b28e0-2101-47a4-8551-c30eb9bfb573.xhtml), *Helper Recipes*, see the following recipes:
    *   *Loading texture data from a file*
    *   *Loading a 3D model from an OBJ file*
*   In [Chapter 11](45108a92-6d49-4759-9495-3f1166e69128.xhtml), *Lighting*, see the *Rendering a geometry with a vertex diffuse lighting recipe*

# Rendering a full-screen quad for post-processing

Image processing is another class of techniques commonly used in 3D graphics. Human eyes perceive the world around us in a way that is almost impossible to simulate directly. There are many effects which cannot be displayed by just drawing a geometry. For example, bright areas seem larger than dark areas (this is usually referred to as bloom); objects seen at our focus point are sharp, but the further from the focus distance, these objects become more fuzzy or blurred (we call this effect a depth of field); color can be perceived differently during the day and at night, when with very little lighting, everything seems more blueish.

These phenomena are easily implemented as post-processing effects. We render the scene normally into an image. After that, we perform another rendering, this time taking the data stored in an image and processing it according to a chosen algorithm. To render an image, we need to place it on a quad that covers the whole scene. Such a geometry is usually called a fullscreen quad.

# How to do it...

1.  Prepare vertex data for the quad's geometry. Use the following values for four vertices (add texture coordinates if needed):
    *   `{ -1.0f, -1.0f, 0.0f }` for top left vertex
    *   `{ -1.0f, 1.0f, 0.0f }` for bottom left vertex
    *   `{ 1.0f, -1.0f, 0.0f }` for top right vertex
    *   `{ 1.0f, 1.0f, 0.0f }` for bottom right vertex
2.  Create a buffer that will serve as a vertex buffer. Allocate a memory object and bind it to the buffer. Upload vertex data to the buffer using a staging resource (refer to the *Creating a buffer*, *Allocating and binding memory object to a buffer*, and *Using staging buffer to update buffer with a device-local memory bound* recipes from [Chapter 4](f1332ca0-b5a2-49bd-ac41-e37068e31042.xhtml), *Resources and Memory*).
3.  Create a combined image sampler. Remember to provide valid uses that depend on the way the image will be accessed during rendering and post-processing: rendering a scene into an image requires a `VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT`; sampling an image (reading data using a sampler) requires a `VK_IMAGE_USAGE_SAMPLED_BIT`; for image load/stores we must provide a `VK_IMAGE_USAGE_STORAGE_BIT`; other uses may also be necessary (refer to the *Creating a combined image sampler* recipe from [Chapter 5](fe2cb528-9d22-49db-a05b-372bce2f87ee.xhtml), *Descriptor Sets*).
4.  Create a descriptor set layout with one combined image sampler. Create a descriptor pool and allocate a descriptor set from it using the created layout. Update the descriptor set with the image view's and sampler handles. Do it each time an application window is resized and an image is recreated (refer to the *Creating a descriptor set layout*, *Creating a descriptor pool*, *Allocating descriptor sets*, and *Updating descriptor sets* recipes from [Chapter 5](fe2cb528-9d22-49db-a05b-372bce2f87ee.xhtml), *Descriptor Sets*).

5.  If we want to access many different image coordinates, create a separate, dedicated render pass with one color attachment and at least one subpass (refer to the *Specifying attachments descriptions*, *Specifying subpass descriptions*, *Specifying dependencies between subpasses*, and *Creating a render pass* recipes from [Chapter 6](2de4339d-8912-440a-89a6-fd1f84961448.xhtml), *Render Passes and Framebuffers*).
6.  Create a shader module with a SPIR-V assembly for a vertex shader created from the following GLSL code (refer to the *Creating a shader module* recipe from [Chapter 8](5744ea05-b18a-4f84-a1df-250b549dfea5.xhtml), *Graphics and Compute Pipelines*):

[PRE45]

7.  Create a shader module for a fragment shader created from the following GLSL code:

[PRE46]

8.  Create a graphics pipeline using the preceding shader modules. It must read one vertex attribute with vertex positions (and potentially a second attribute with texture coordinates). Use a `VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP` topology and disable face culling (refer to the *Specifying pipeline vertex input state*, *Specifying pipeline input assembly state*, *Specifying pipeline rasterization state*, and *Creating a graphics pipeline* recipes from [Chapter 8](5744ea05-b18a-4f84-a1df-250b549dfea5.xhtml), *Graphics and Compute Pipelines*).
9.  Render a scene into the created image. Next, start another render pass and draw the full-screen quad using the prepared graphics pipeline (refer to the Beginning a render pass and Ending a render pass recipes from [Chapter 6](2de4339d-8912-440a-89a6-fd1f84961448.xhtml), *Render Passes and Framebuffers*, to the *Binding descriptor sets* recipe from [Chapter 5](fe2cb528-9d22-49db-a05b-372bce2f87ee.xhtml), *Descriptor Sets*, and to the *Binding vertex buffers* and *Drawing a geometry recipes* from [Chapter 9](0a69f5b5-142e-422b-aa66-5cb09a6467b3.xhtml), *Command Recording and Drawing*).

# How it works...

Image post-processing can be performed using compute shaders. However, when we want to display an image on screen, we must use a swapchain. Storing data in an image from within shaders requires images to be created with the storage image use. Unfortunately, such usage may not be supported on swapchain images, so it would require the creation of additional, intermediate resources, which further increase the complexity of a code.

Using a graphics pipeline allows us to process image data inside fragment shaders and store the results in color attachments. Such usage is mandatory for swapchain images, so this way feels more natural for image processing implemented with the Vulkan API. On the other hand, the graphics pipeline requires us to draw a geometry, so we need not only vertex data, and vertex and fragment shaders, but also a render pass and a framebuffer as well. That's why using compute shaders may be more efficient. So, everything depends on the features supported by the graphics hardware (available swapchain image usages) and the given situation.

In this recipe, we will present the method to draw a full-screen quad during an image postprocessing phase. First, we need the vertex data itself. It can be prepared directly in the clip space. This way we can create a much simpler vertex shader and avoid multiplying the vertex position by a projection matrix. After the perspective division, for the vertices to fit into a view, values stored in `x` and `y` components of their positions must fit into a <`-1, 1`> range (inclusive) and a value in a `z` component must be inside a <`0, 1`> range. So, if we want to cover the whole screen, we need the following set of vertices:

[PRE47]

We can add normalized texture coordinates if needed or we can rely on the built-in `gl_FragCoord` value (when writing GLSL shaders), which contain screen coordinates of a currently processed shader. When we use input attachments, we even don't need texture coordinates, as we can access only the sample associated with the currently processed fragment.

Vertex data needs to be stored in a buffer serving as a vertex buffer. So we need to create it, allocate a memory object and bind it to the buffer and upload vertex data to the buffer:

[PRE48]

Next, we need a way to access texel data inside the fragment shader. We can use an input attachment if we want to access data stored in a color attachment from any of the previous subpasses in the same render pass. We can use a storage image, separate the sampler and the sampled image or a combined image sampler. The latter is used in this recipe. To simplify this recipe and the code, we read texture data from a file. But usually we will have an image into which the scene will be rendered:

[PRE49]

In the preceding code, we create a combined image sampler and specify that we will access it with unnormalized texture coordinates. Usually we provide coordinates in the <0.0, 1.0> range (inclusive). This way we don't need to worry about the image's size. On the other hand, for post-processing we usually want to address the texture image using screen space coordinates, and that's when unnormalized texture coordinates are used--they correspond to the image's dimensions.

To access an image, we also need a descriptor set. We don't need a uniform buffer as we don't transform the geometry-drawn vertices are already in the correct space (the clip space). Before we can allocate a descriptor set, we create a layout with one combined image sampler accessed in a fragment shader stage. After that, a pool is created and one descriptor set is allocated from the pool:

[PRE50]

In the preceding code we also update the descriptor set with the handles of created sampler and image view. Unfortunately, the image into which we render a scene will usually fit into a screen. This means that we must recreate it when the size of an application's window is changed and, to do that, we must destroy the old image and create a new one with new dimensions. After such an operation we must update the descriptor set again with the handle of the new image (the sampler doesn't need to be recreated). So we must remember to update the descriptor set each time the application window size is changed.

One last thing is the creation of a graphics pipeline. It uses only two shader stages: vertex and fragment. The number of attributes fetched by the vertex shader depend on whether we need texture coordinates (and other dedicated attributes) or not. The full-screen quad's geometry should be drawn using a `VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP` topology. We also don't need any blending.

The most important part of the post-processing is performed inside the fragment shader. The work to be done depends on the technique we want to implement. In this recipe, we present an edge detection algorithm:

[PRE51]

In the preceding fragment shader code, we sample four values around the fragment being processed. We take a negated value from one sample to the left and add a value read from one sample to the right. This way we know the difference between samples in a horizontal direction. When the difference is big, we know there is an edge.

We do the same operation for a vertical direction to detect horizontal lines too (the vertical difference, or a gradient, is used to detect horizontal edges; the horizontal gradient allows us to detect vertical edges). After that we store a value in the output variable. We additionally take the `abs()` value, but this is done only for visualization purposes.

In the preceding fragment shader, we access multiple texture coordinates. This can be done on combined image samplers (input attachments allow us to access only a single coordinate associated with a fragment being processed). However, to bind an image to a descriptor set as a resource other than an input attachment, we must end the current render pass and start another one. In a given render pass, images cannot be used for attachments and for any other non-attachment purpose at the same time.

Using the preceding setup, we should see the following result (on the right) with the original image seen on the left:

![](img/image_12_014.png)

# See also

*   In [Chapter 4](f1332ca0-b5a2-49bd-ac41-e37068e31042.xhtml), *Resources and Memory*, see the following recipes:
    *   *Creating a buffer*
    *   *Allocating and binding a memory object to a buffer*
    *   *Using the staging buffer to update a buffer with a device-local memory bound*
*   In [Chapter 5](fe2cb528-9d22-49db-a05b-372bce2f87ee.xhtml), *Descriptor Sets*, see the following recipes:
    *   *Creating a combined image sampler*
    *   *Creating a descriptor set layout*
    *   *Allocating descriptor sets*
    *   *Binding descriptor sets*
    *   *Updating descriptor sets*
*   In [Chapter 6](2de4339d-8912-440a-89a6-fd1f84961448.xhtml), *Render Passes and Framebuffers*, see the following recipes:
    *   *Beginning a render pass*
    *   *Ending a render pass*
*   In [Chapter 8](5744ea05-b18a-4f84-a1df-250b549dfea5.xhtml), *Graphics and Compute Pipelines*, see the following recipes:
    *   *Creating a shader module*
    *   *Specifying the pipeline vertex input state*
    *   *Specifying the pipeline input assembly state*
    *   *Specifying the pipeline rasterization state*
    *   *Creating a graphics pipeline*
*   In [Chapter 9](0a69f5b5-142e-422b-aa66-5cb09a6467b3.xhtml), *Command Recording and Drawing*, see the following recipes:
    *   *Binding vertex buffers*
    *   *Drawing a geometry*

# Using input attachments for a color correction post-process effect

There are many various post-process techniques used in 3D applications. Color correction is one of them. This is relatively simple, but it can give impressive results and greatly improve the look and feel of a rendered scene. Color correction can change the mood of the scene and induce the desired feelings for the users.

Usually, a color correction effect requires us to read data of a single, currently processed sample. Thanks to this property, we can implement this effect using input attachments. This allows us to perform post-processing inside the same render pass in which the whole scene is rendered, thus improving the performance of our application.

The following is an example of an image generated with this recipe:

![](img/image_12_015.png)

# How to do it...

1.  Create a fullscreen quad with additional resources required during postprocessing phase (refer to the *Rendering a full-screen quad for post processing* recipe).
2.  Create a descriptor set layout with one input attachment accessed in a fragment shader stage. Allocate a descriptor set using the prepared layout (refer to the *Creating a descriptor set layout* and *Allocating descriptor sets* recipes from [Chapter 5](fe2cb528-9d22-49db-a05b-372bce2f87ee.xhtml), *Descriptor Sets*).
3.  Create a 2D image (along with a memory object and an image view) into which the scene will be drawn. Specify not only a `VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT` usage, but also a `VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT` usage during image creation. Recreate the image each time the application's window is resized (refer to the *Creating an input attachment* recipe from [Chapter 5](fe2cb528-9d22-49db-a05b-372bce2f87ee.xhtml), *Descriptor Sets*).

4.  Update the descriptor set with input attachment using the handle of the created image. Do it each time an application window is resized and an image is recreated (refer to the *Updating descriptor sets* recipe from Chapter 5, *Descriptor Sets*).
5.  Prepare all the resource required to normally render the scene. When creating a render pass used for rendering the scene, add one additional subpass at the end of the render pass. Specify the attachment used in previous subpasses as a color attachment to be an input attachment in the additional subpass. A swapchain image should be used as a color attachment in the additional subpass (refer to the *Specifying subpass descriptions* and *Creating a render pass* recipes from [Chapter 6](2de4339d-8912-440a-89a6-fd1f84961448.xhtml), *Render Passes and Framebuffers*).
6.  Create a shader module with a vertex shader created from the following GLSL code (refer to the *Creating a shader module* recipe from [Chapter 8](5744ea05-b18a-4f84-a1df-250b549dfea5.xhtml), *Graphics and Compute Pipelines*):

[PRE52]

7.  Create a shader module with a fragment shader created from the following GLSL code:

[PRE53]

8.  Create a graphics pipeline used for drawing a post-process phase. Use the preceding vertex and fragment shader modules. Prepare the rest of the pipeline parameters according to the *Rendering a fullscreen quad for postprocessing* recipe.
9.  In each frame of animation, draw the scene normally into a created image, then progress to the next subpass (refer to the *Progressing to the next subpass* recipe from [Chapter 6](2de4339d-8912-440a-89a6-fd1f84961448.xhtml), *Render Passes and Framebuffers*). Bind the created graphics pipeline used for post-processing, bind the descriptor set with the input attachment, bind the vertex buffer with full-screen quad data, and draw the full-screen quad (refer to the *Binding descriptor sets* recipe from [Chapter 5](fe2cb528-9d22-49db-a05b-372bce2f87ee.xhtml), *Descriptor Sets*, to the *Binding a pipeline object* recipe from [Chapter 8](5744ea05-b18a-4f84-a1df-250b549dfea5.xhtml), *Graphics and Compute Pipelines*, and to the *Binding vertex buffers* and *Drawing a geometry* recipes from [Chapter 9](0a69f5b5-142e-422b-aa66-5cb09a6467b3.xhtml), *Command Recording and Drawing*).

# How it works...

Creating a postprocessing effect that is rendered inside the same render pass as the scene is performed in two steps.

In the first step, we need to prepare resources for the base scene: its geometry, textures, descriptor sets, and pipeline objects, among others. In the second step, we do the same for the full-screen quad, as described in the *Rendering a fullscreen quad for postprocessing* recipe.

The two most important resources prepared solely for the post-processing phase are an image and a graphics pipeline. The image will serve as a color attachment when we are rendering the scene in a normal way. We just render the scene into the image instead of rendering it into a swapchain image. The image must serve both as a color attachment during scene rendering, but also as an input attachment during post-processing. We must also remember to recreate it when the size of the application's window is changed:

[PRE54]

Accessing an image as an input attachment requires us to use a descriptor set. It must contain at least our input attachment, so we need to create a proper layout. Input attachments can be accessed only inside fragment shaders, so the creation of a descriptor set layout, a descriptor pool, and an allocation of a descriptor set may look like this:

[PRE55]

We must also update the descriptor set with the handle of our color attachment/input attachment image. As the image gets recreated when the size of the application's window is changed, we must update the descriptor too:

[PRE56]

The next thing we need to describe is the preparation of a render pass. In this recipe the render pass is common for both the scene rendering and the post-processing phase. The scene is rendered in its own, dedicated subpass (or subpasses). The post-processing phase adds an additional subpass for rendering a full-screen quad.

Usually, we define two render pass attachments: a color attachment (a swapchain image) and a depth attachment (an image with a depth format). This time we need three attachments: the first one is a color attachment for which the created image will be used; the depth attachment is the same as usual; and the third attachment is also a color attachment, for which a swapchain image will be used. This way, the scene is rendered normally into two (color and depth attachments). Then, the first attachment is used as an input attachment during post-processing; and the full-screen quad is rendered into the second color attachment (a swapchain image) so the final image appears on screen.

The following code sets up the render pass attachment:

[PRE57]

The render pass has two subpasses defined as follows:

[PRE58]

We also can't forget about the render pass subpass dependencies. They are very important here as they synchronize the two subpasses. We can't read data from a texture until the data is written into it, so we need dependencies between the 0 and the 1 subpass (for the image serving as color and input attachment. Similarly, dependencies are needed for a swapchain image:

[PRE59]

The graphics pipeline used during post-processing phase is a standard one. Only two things are different: the graphics pipeline is used inside the subpass with index `1` (not `0` as in other recipes--the scene is rendered in the subpass `0`); and the fragment shader loads color data, not from the combined image sampler, but from the input attachment. The input attachment inside the fragment shader is defined as follows:

[PRE60]

We read data from it using the `subpassLoad()` function. It takes only the uniform variable. Texture coordinates are unnecessary, because through an input attachment we can read data only from the coordinate associated with the fragment being processed.

[PRE61]

The fragment shader then takes the loaded color, calculates a sepia color from it, and stores it in an output variable (a color attachment). All this combined should lead us to create the following results. On the left we see the scene rendered normally. On the right we see a post-processing effect applied:

![](img/image_12_016.png)

# See also

*   In [Chapter 5](fe2cb528-9d22-49db-a05b-372bce2f87ee.xhtml), *Descriptor Sets*, see the following recipes:
    *   *Creating an input attachment*
    *   *Creating a descriptor set layout*
    *   *Allocating descriptor sets*
    *   *Updating descriptor sets*
    *   *Binding descriptor sets*
*   In [Chapter 6](2de4339d-8912-440a-89a6-fd1f84961448.xhtml), *Render Passes and Framebuffers*, see the following recipes:
    *   *Specifying subpass descriptions*
    *   *Creating a render pass*
    *   *Progressing to the next subpass*
*   In [Chapter 8](5744ea05-b18a-4f84-a1df-250b549dfea5.xhtml), *Graphics and Compute Pipelines*, see the following recipes:
    *   *Creating a shader module*
    *   *Binding a pipeline object*
*   In [Chapter 9](0a69f5b5-142e-422b-aa66-5cb09a6467b3.xhtml), *Command Recording and Drawing*, see the following recipes:
    *   Binding vertex buffers
    *   Drawing a geometry
*   The recipe *Rendering a full-screen quad for post-processing*, in this chapter