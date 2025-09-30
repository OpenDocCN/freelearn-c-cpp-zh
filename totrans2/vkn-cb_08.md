# Graphics and Compute Pipelines

In this chapter, we will cover the following recipes:

*   Creating a shader module
*   Specifying pipeline shader stages
*   Specifying a pipeline vertex binding description, attribute description, and input state
*   Specifying a pipeline input assembly state
*   Specifying a pipeline tessellation state
*   Specifying a pipeline viewport and scissor test state
*   Specifying a pipeline rasterization state
*   Specifying a pipeline multisample state
*   Specifying a pipeline depth and stencil state
*   Specifying a pipeline blend state
*   Specifying pipeline dynamic states
*   Creating a pipeline layout
*   Specifying graphics pipeline creation parameters
*   Creating a pipeline cache object
*   Retrieving data from a pipeline cache
*   Merging multiple pipeline cache objects
*   Creating a graphics pipeline
*   Creating a compute pipeline
*   Binding a pipeline object
*   Creating a pipeline layout with a combined image sampler, a buffer, and push constant ranges
*   Creating a graphics pipeline with vertex and fragment shaders, depth test enabled, and with dynamic viewport and scissor tests
*   Creating multiple graphics pipelines on multiple threads
*   Destroying a pipeline
*   Destroying a pipeline cache
*   Destroying a pipeline layout
*   Destroying a shader module

# Introduction

Operations recorded in command buffers and submitted to queues are processed by the hardware. Processing is performed in a series of steps that form a pipeline. When we want to perform mathematical calculations, we use a compute pipeline. If we want to draw anything, we need a graphics pipeline.

Pipeline objects control the way in which geometry is drawn or computations are performed. They manage the behavior of the hardware on which our application is executed. And they are one of the biggest and most apparent differences between Vulkan and OpenGL. OpenGL used a state machine. It allowed us to change many rendering or computing parameters whenever we wanted. We could set up the state, activate a shader program, draw a geometry, then activate another shader program and draw another geometry. In Vulkan it is not possible because the whole rendering or computing state is stored in a single, monolithical object. When we want to use a different set of shaders, we need to prepare and use a separate pipeline. We can't just switch shaders.

This may be intimidating at first because many shader variations (not including the rest of the pipeline state) cause us to create multiple pipeline objects. But it serves two important goals. The first is the performance. Drivers that know the whole state in advance may optimize execution of the following operations. The second goal is the stability of the performance. Changing the state whenever we want may cause the driver to perform additional operations, such as shader recompilation, in unexpected and unpredictable moments. In Vulkan, all the required preparations, including shader compilation, are done only during the pipeline creation.

In this chapter, we will see how to set up all of the graphics or compute pipelines parameters to successfully create them. We will see how to prepare shader modules and define which shader stages are active, how to set up depth or stencil tests and how to enable blending. We will also specify what vertex attributes are used and how they are provided during drawing operations. Finally, we will see how to create multiple pipelines and how to improve the speed of their creation.

# Creating a shader module

The first step in creating a pipeline object is to prepare shader modules. They represent shaders and contain their code written in a SPIR-V assembly. A single module may contain code for multiple shader stages. When we write shader programs and convert them into SPIR-V form, we need to create a shader module (or multiple modules) before we can use shaders in our application.

# How to do it...

1.  Take the handle of a logical device stored in a variable of type `VkDevice` named `logical_device`.
2.  Load a binary SPIR-V assembly of a selected shader and store it in a variable of type `std::vector<unsigned char>` named `source_code`.
3.  Create a variable of type `VkShaderModuleCreateInfo` named `shader_module_create_info`. Use the following values to initialize its members:
    *   `VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO` value for `sType`.
    *   `nullptr` value for `pNext`
    *   `0` value for `flags`
    *   The number of elements in the `source_code` vector (size in bytes) for `codeSize`
    *   A pointer to the first element of the `source_code` variable for `pCode`
4.  Create a variable of type `VkShaderModule` named `shader_module` in which the handle of a created shader module will be stored.

5.  Make the `vkCreateShaderModule( logical_device, &shader_module_create_info, nullptr, &shader_module )` function call for which provide the `logical_device` variable, a pointer to the `shader_module_create_info`, a `nullptr` value, and a pointer to the `shader_module` variable.
6.  Make sure the `vkCreateShaderModule()` function call returned a `VK_SUCCESS` value which indicates that the shader module was properly created.

# How it works...

Shader modules contain source code--a single SPIR-V assembly--of selected shader programs. It may represent multiple shader stages but a separate entry point must be associated with each stage. This entry point is then provided as one of the parameters when we create a pipeline object (refer to the *Specifying pipeline shader stages* recipe).

When we want to create a shader module, we need to load a file with the binary SPIR-V code or acquire it in any other way. Then we provide it to a variable of type `VkShaderModuleCreateInfo` like this:

[PRE0]

Next, the pointer to such a variable is provided to the `vkCreateShaderModule()` function, which creates a module:

[PRE1]

We just need to remember that shaders are not compiled when we create a shader module; this is done when we create a pipeline object.

Shader compilation and linkage is performed during the pipeline object creation.

# See also

The following recipes in this chapter:

*   *Specifying pipeline shader stages*
*   *Creating a graphics pipeline*
*   *Creating a compute pipeline*
*   *Destroying a shader module*

# Specifying pipeline shader stages

In compute pipelines, we can use only compute shaders. But graphics pipelines may contain multiple shader stages--vertex (which is obligatory), geometry, tessellation control and evaluation, and fragment. So for the pipeline to be properly created, we need to specify what programmable shader stages will be active when a given pipeline is bound to a command buffer. And we also need to provide a source code for all the enabled shaders.

# Getting ready

To simplify the recipe and lower the number of parameters needed to prepare descriptions of all enabled shader stages, a custom `ShaderStageParameters` type is introduced. It has the following definition:

[PRE2]

In the preceding structure, `ShaderStage` defines a single pipeline stage for which the rest of the parameters are specified. `ShaderModule` is a module from which a SPIR-V source code for the given stage can be taken, associated with a function whose name is provided in the `EntryPointName` member. The `SpecializationInfo` parameter is a pointer to a variable of type `VkSpecializationInfo`. It allows values of the constant variables defined in the shader source code to be modified at runtime, during pipeline creation. But if we don't want to specify constant values, we can provide a `nullptr` value.

# How to do it...

1.  Create a shader module or modules containing source code for each shader stage that will be active in a given pipeline (refer to the *Creating a shader module* recipe).
2.  Create a `std::vector` variable named `shader_stage_create_infos` with elements of type `VkPipelineShaderStageCreateInfo`.
3.  For each shader stage that should be enabled in a given pipeline, add an element to the `shader_stage_create_infos` vector and use the following values to initialize its members:
    *   `VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO` value for `sType`
    *   `nullptr` value for `pNext`
    *   `0` value for `flags`
    *   The selected shader stage for `stage`
    *   The shader module with a source code of a given shader stage for `module`
    *   The name of the function that implements the given shader in the shader module (usually `main`) for `pName`
    *   A pointer to a variable of type `VkSpecializationInfo` with a constant value specialization or a `nullptr` value if no specialization is required for `pSpecializationInfo`

# How it works...

Defining a set of shader stages that will be active in a given pipeline requires us to prepare an array (or a vector) variable with elements of type `VkPipelineShaderStageCreateInfo`. Each shader stage requires a separate entry in which we need to specify a shader module and the name of the entry point that implements the behavior of a given shader in the provided module. We can also provide a pointer to the specialization info which allows us to modify values of shader constant variables during the pipeline creation (at runtime). This allows us to use the same shader code multiple times with slight variations.

Specifying pipeline shader stages info is obligatory for both graphics and compute pipelines.

Let's imagine we want to use only vertex and fragment shaders. We can prepare a vector with elements of a custom `ShaderStageParameters` type like this:

[PRE3]

The implementation of the preceding recipe, which uses the data from the aforementioned vector, may look like this:

[PRE4]

Each shader stage provided in the array must be unique.

# See also

The following recipes in this chapter:

*   *Creating a shader module*
*   *Creating a graphics pipeline*
*   *Creating a compute pipeline*

# Specifying a pipeline vertex binding description, attribute description, and input state

When we want to draw a geometry, we prepare vertices along with their additional attributes like normal vectors, colors, or texture coordinates. Such vertex data is chosen arbitrarily by us, so for the hardware to properly use them, we need to specify how many attributes there are, how are they laid out in memory, or where are they taken from. This information is provided through the vertex binding description and attribute description required to create a graphics pipeline.

# How to do it...

1.  Create a `std::vector` variable named `binding_descriptions` with elements of type `VkVertexInputBindingDescription`.
2.  Add a separate entry to the `binding_descriptions` vector for each vertex binding (part of a buffer bound to a command buffer as a vertex buffer) used in a given pipeline. Use the following values to initialize its members:
    *   The index of a binding (number which it represents) for `binding`
    *   The number of bytes between two consecutive elements in a buffer for `stride`
    *   Parameters indicating whether values of attributes read from a given binding should advance per vertex (`VK_VERTEX_INPUT_RATE_VERTEX`) or per instance (`VK_VERTEX_INPUT_RATE_INSTANCE`) for `inputRate`
3.  Create a `std::vector` variable named `attribute_descriptions` with elements of type `VkVertexInputAttributeDescription`.

4.  Add a separate entry to the `attribute_descriptions` vector variable for each attribute provided to a vertex shader in a given graphics pipeline. Use the following values to initialize its members:
    *   The shader location through which a given attribute is read in a vertex shader for `location`
    *   The index of a binding to which a vertex buffer with the source of this attribute's data will be bound for `binding`
    *   The format of an attribute's data for `format`
    *   The memory offset from the beginning of a given element in the binding for `offset`
5.  Create a variable of type `VkPipelineVertexInputStateCreateInfo` named `vertex_input_state_create_info`. Use the following values to initialize its members:
    *   `VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO` value for `sType`
    *   `nullptr` value for `pNext`
    *   `0` value for `flags`
    *   The number of elements in the `binding_descriptions` vector for `vertexBindingDescriptionCount`
    *   A pointer to the first element of the `binding_descriptions` vector for `pVertexBindingDescriptions`
    *   The number of elements in the `attribute_descriptions` vector for `vertexAttributeDescriptionCount`
    *   A pointer to the first element of the `attribute_descriptions` vector for `pVertexAttributeDescriptions`

# How it works...

Vertex binding defines a collection of data taken from a vertex buffer bound to a selected index. This binding is used as a numbered source of data for vertex attributes. We can use at least 16 separate bindings to which we can bind separate vertex buffers or different parts of memory of the same buffer.

The vertex input state is obligatory for a graphics pipeline creation.

Through a binding description, we specify where the data is taken from (from which binding), how it is laid out (what is the stride between consecutive elements in the buffer), and how this data is read (whether it should be fetched per vertex or per instance).

As an example, when we want to use three attributes--three element vertex positions, two element texture coordinates, and three element color values, which are read per vertex from the `0`^(th) binding--we can use the following code:

[PRE5]

Through a vertex input description we define the attributes taken from a given binding. For each attribute we need to provide a shader location (the same as in the shader source code defined through a `layout( location = <number> )` qualifier), a format of the data used for a given attribute, and a memory offset at which the given attribute starts (relative to the beginning of the data for the given element). The number of input description entries specifies the total number of attributes used during rendering.

![](img/image_08_001.png)

In the previous situation--with three component vertex positions, two component texture coordinates, and three component colors--we can use the following code to specify the vertex input description:

[PRE6]

All three attributes are taken from the `0`^(th) binding. Positions are provided to a vertex shader at the `0`^(th) location, texcoords through the first location, and color values through the second location. The position and color are three component vectors, and texcoords have two components. They all use floating-point signed values. The position is first, so it has no offset. The texture coordinate goes next, so it has an offset of three floating-point values. The color starts after the texture coordinate, so its offset is equal to five floating-point values.

The implementation of this recipe is provided in the following code:

[PRE7]

# See also

*   In [Chapter 7](97217f0d-bed7-4ae1-a543-b4d599f299cf.xhtml), *Shaders*, see the recipe:
    *   *Writing vertex shaders*
*   In [Chapter 9](0a69f5b5-142e-422b-aa66-5cb09a6467b3.xhtml), *Command Recording and Drawing*, see the following recipe:
    *   *Binding vertex buffers*
*   The recipe *Creating a graphics **pipeline* in this chapter

# Specifying a pipeline input assembly state

Drawing geometry (3D models) involves specifying the type of primitives that are formed from provided vertices. This is done through an input assembly state.

# How to do it...

1.  Create a variable of type `VkPipelineInputAssemblyStateCreateInfo` named `input_assembly_state_create_info`. Use the following values to initialize its members:
    *   `VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO` value for `sType`
    *   `nullptr` value for `pNext`
    *   `0` value for `flags`
    *   The selected type of primitives to be formed from vertices (point list, line list, line strip, triangle list, triangle strip, triangle fan, line list with adjacency, line strip with adjacency, triangle list with adjacency, triangle strip with adjacency, or patch list) for `topology`
    *   For the `primitiveRestartEnable` member, in cases of drawing commands that use vertex indices, specify whether a special index value should restart a primitive (`VK_TRUE`, can't be used for list primitives) or if a primitive restart should be disabled (`VK_FALSE`)

# How it works...

Through an input assembly state, we define what types of polygons are formed from the drawn vertices. The most commonly used primitives are triangle strips or lists, but the used topology depends on the results we want to achieve.

An input assembly state is required for the graphics pipeline creation.

![](img/image_08_002.png)

When selecting how vertices are assembled, we just need to bear in mind some requirements:

*   We can't use list primitives with a primitive restart option.
*   Primitives with adjacency can only be used with geometry shaders. For this to work correctly, a `geometryShader` feature must be enabled during the logical device creation.
*   When we want to use tessellation shaders, we can only use patch primitives. In addition, we also need to remember that a `tessellationShader` feature must be enabled during the logical device creation.

Here is an example of a source code that initializes a variable of type `VkPipelineInputAssemblyStateCreateInfo`:

[PRE8]

# See also

*   The following recipes in this chapter:
    *   *Specifying pipeline rasterization state*
    *   *Creating a graphics pipeline*

# Specifying a pipeline tessellation state

Tessellation shaders are one of the optional, additional programmable shader stages that can be enabled in a graphics pipeline. But when we want to activate them, we also need to prepare a pipeline tessellation state.

# How to do it...

1.  Create a variable of type `VkPipelineTessellationStateCreateInfo` named `tessellation_state_create_info`. Use the following to initialize its members:
    *   `VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO` value for `sType`
    *   `nullptr` value for `pNext`
    *   `0` value for `flags`
    *   The number of control points (vertices) which form a patch for `patchControlPoints`

# How it works...

To use tessellation shaders in our application, we need to enable a `tessellationShader` feature during a logical device creation, we need to write a source code for both tessellation control and evaluation shaders, we need to create a shader module (or two) for them, and we also need to prepare a pipeline tessellation state represented by a variable of type `VkPipelineTessellationStateCreateInfo`.

The tessellation state is optional--we need to specify it only when we want to use tessellation shaders in a graphics pipeline.

In the tessellation state we only provide information about the number of control points (vertices) from which a patch is formed. The specification states that patches may have up to at least 32 vertices.

The maximal supported number of control points (vertices) in a patch must be at least 32.

A patch is just a collection of points (vertices) that are used by the tessellation stages to generate typical points, lines, or polygons like triangles. It can be exactly the same as usual polygons. As an example, we can take vertices that form a triangle and draw them as patches. Results of such an operation are correct. But for the patch, we can use any other unusual order and number of vertices. This gives us much more flexibility in controlling the way new vertices are created by the tessellation engine.

To fill a variable of type `VkPipelineTessellationStateCreateInfo,` we can prepare the following code:

[PRE9]

# See also

*   In [Chapter 7](97217f0d-bed7-4ae1-a543-b4d599f299cf.xhtml), *Shaders*, see the following recipes:
    *   *Writing tessellation control shaders*
    *   *Writing tessellation evaluation shaders*
*   The recipe *Creating a graphics pipeline *in this chapter

# Specifying a pipeline viewport and scissor test state

Drawing an object on screen requires us to specify the screen parameters. Creating a swapchain is not enough--we don't always need to draw to the entire available image area. There are situations in which we just want to draw a smaller picture in the whole image, such as the reflection in the back mirror of a car or half of the image in split-screen multiplayer games. We define the area of the image to which we want to draw through a pipeline viewport and scissor test states.

# Getting ready

Specifying parameters for a viewport and scissor states requires us to provide a separate set of parameters for both the viewport and scissor test, but the number of elements in both sets must be equal. To keep parameters for both states together, a custom `ViewportInfo` type is introduced in this recipe. It has the following definition:

[PRE10]

The first member, as the name suggests, contains parameters for a set of viewports. The second is used to define the parameters for scissor tests corresponding to each viewport.

# How to do it...

1.  If rendering is to be performed to more than one viewport, create a logical device with the `multiViewport` feature enabled.

2.  Create a variable of type `std::vector<VkViewport>` named `viewports`. Add a new element to the `viewports` vector for each viewport into which rendering will be done. Use the following values to initialize its members:
    *   The position (in pixels) of the left side of the rendering area for `x`
    *   The position (in pixels) of the top side of the rendering area for `y`
    *   The width of the rendering area (in pixels) for `width`
    *   The height of the rendering area (in pixels) for `height`
    *   The value between `0.0` and `1.0` for the minimal depth of the viewport for `minDepth`
    *   The value between `0.0` and `1.0` for the maximal depth of the viewport for `maxDepth`
3.  Create a variable of type `std::vector<VkRect2D>` named `scissors`. Add a new element to the `scissors` vector variable for each viewport into which rendering will be done (the `scissors` vector must have the same number of elements as the `viewports` vector). Use the following values to initialize its members:
    *   The position of the top left corner of the scissor rectangle for the `x` and `y` members of the `offset`
    *   The width and height of the scissor rectangle for the `width` and `height` members of the `extent`
4.  Create a variable of type `VkPipelineViewportStateCreateInfo` named `viewport_state_create_info`. Use the following values to initialize its members:
    *   `VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO` value for `sType`
    *   `nullptr` value for `pNext`
    *   `0` value for `flags`
    *   The number of elements in the `viewports` vector for `viewportCount`
    *   A pointer to the first element of the `viewports` vector for `pViewports`
    *   The number of elements in the `scissors` vector for `scissorCount`
    *   A pointer to the first element of the `scissors` vector for `pScissors`

# How it works...

Vertex positions are transformed (usually inside a vertex shader) from the local space into a clip space. The hardware then performs a perspective division which generates normalized device coordinates. Next, polygons are assembled and rasterized--this process generates fragments. Each fragment has its own position defined in a framebuffer's coordinates. Also, for this position to be correctly calculated, a viewport transformation is required. Parameters of this transformation are specified in a viewport state.

The viewport and scissor test state is optional, though commonly used--we don't need to provide it when rasterization is disabled.

Through a viewport state, we define the top-left corner and the width and height of the rendering area in a framebuffer's coordinates (pixels on screen). We also define the minimal and maximal viewport depth value (floating-point values between `0.0` and `1.0`, inclusive). It is valid to specify the value of the maximal depth to be smaller than the value of the minimal depth.

A scissor test allows us to additionally clip the generated fragments to a rectangle specified in the scissor parameters. When we don't want to clip fragments, we need to specify an area equal to a viewport size.

In Vulkan, the scissor test is always enabled.

The number of set of parameters for a viewport and scissor test must be equal. That's why it may be good to define a custom type with which we can keep the number of elements of both properties equal. The following is a sample code that specifies parameters for one viewport and one scissor test through a variable of a custom `ViewportInfo` type:

[PRE11]

The preceding variable can be used to create a viewport and scissor test as defined in this recipe. The implementation of the recipe may look like this:

[PRE12]

If we want to change some of the viewport or scissor test parameters, we need to recreate a pipeline. But during the pipeline creation, we can specify that the viewport and scissor test parameters are dynamic. This way, we don't need to recreate a pipeline to change these parameters--we specify them during command buffer recording. But we need to remember that the number of viewports (and scissor tests) is always specified during the pipeline creation. We can't change it later.

It is possible to define a viewport and scissor test as dynamic states and specify their parameters during command buffer recording. The number of viewports (and scissor tests) is always specified during the graphics pipeline creation.

We also can't provide more than one viewport and scissor test, unless a `multiViewport` feature is enabled for a logical device. An index of a viewport transformation that will be used for rasterization can be changed only inside geometry shaders.

Changing the index of a viewport transformation used for rasterization requires us to use geometry shaders.

# See also

*   In [Chapter 1](d10e8284-6122-4d0a-8f86-ab0bc0bba47e.xhtml), *Instances and Devices*, see the following recipes:
    *   *Getting features and properties of a physical device*
    *   *Creating a logical device*
*   In [Chapter 7](97217f0d-bed7-4ae1-a543-b4d599f299cf.xhtml), *Shaders*, see the recipe:
    *   *Writing geometry shaders*
*   The recipe *Creating a graphics pipeline*, in this chapter

# Specifying a pipeline rasterization state

The rasterization process generates fragments (pixels) from the assembled polygons. The viewport state is used to specify where, in the framebuffer coordinates, fragments will be generated. To specify how (if at all) fragments are generated, we need to prepare a rasterization state.

# How to do it...

1.  Create a variable of type `VkPipelineRasterizationStateCreateInfo` named `rasterization_state_create_info`. Use the following values to initialize its members:
    *   `VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO` value for `sType`.
    *   `nullptr` value for `pNext`.
    *   `0` value for `flags`.
    *   For `depthClampEnable` use a `true` value if a depth value for fragments whose depth is outside of the min/max range specified in a viewport state should be clamped within this range, or use a `false` value if fragments with depth outside of the this range should be clipped (discarded); when the `depthClampEnable` feature is not enabled only a `false` value can be specified.
    *   For `rasterizerDiscardEnable` use a `false` value if fragments should be normally generated or `true` to disable rasterization.
    *   For `polygonMode` specify how assembled polygons should be rendered--fully filled or if lines or points should be rendered (lines and points modes can only be used if a `fillModeNonSolid` feature is enabled).
    *   The sides of the polygon--front, back, both or none--that should be culled for `cullMode`.
    *   The side of the polygon--drawn on screen in clockwise or counterclockwise vertex order--that should be considered as a front side for `frontFace`.
    *   For `depthBiasEnable` specify a `true` value if depth values calculated for fragments should be additionally offset or a `false` value if no such modification should be performed.
    *   The constant value that should be added to a fragment's calculated depth value when depth bias is enabled for `depthBiasConstantFactor`.
    *   The maximum (or minimum) value of a depth bias which can be added to a fragment's depth when depth bias is enabled for `depthBiasClamp`.
    *   The value added to fragment's slope in depth bias calculations when depth bias is enabled for `depthBiasSlopeFactor`.
    *   The value specifying the width of rendered lines for `lineWidth`; if a `wideLines` feature is not enabled, only a `1.0` value can be specified; otherwise, values greater than `1.0` can also be provided.

# How it works...

The rasterization state controls the parameters of a rasterization. First and foremost it defines if the rasterization is enabled or disabled. Through it we can specify which side of the polygon is the front--if it is the one in which vertices appear in a clockwise order on screen or if in a counterclockwise order. Next, we need to control if culling should be enabled for the front, back, both faces, or if it should be disabled. In OpenGL, by default, counterclockwise faces were considered front and culling was disabled. In Vulkan, there is no default state so how we define these parameters is up to us.

A rasterization state is always required during the graphics pipeline creation.

The rasterization state also controls the way polygons are drawn. Usually we want them to be fully rendered (filled). But we can specify if only their edges (lines) or points (vertices) should be drawn. Lines or points modes can only be used if the `fillModeNonSolid` feature is enabled during the logical device creation.

For the rasterization state, we also need to define how the depth value of generated fragments is calculated. We can enable depth bias--a process which offsets a generated depth value by a constant value and an additional slope factor. We also specify the maximal (or minimal) offset value that can be applied to the depth value when depth bias is enabled.

After that, we also need to define what to do with fragments whose depth value is outside the range specified in a viewport state. When the depth clamp is enabled, the depth value of such fragments is clamped to the defined range and the fragments are processed further. If the depth clamp is disabled, such fragments are discarded.

One last thing is to define the width of the rendered lines. Normally we can specify only a value of `1.0`. But if we enable the `wideLines` feature, we can provide values greater than `1.0`.

The rasterization state is defined through a variable of type `VkPipelineRasterizationStateCreateInfo`. A sample source code that fills such variable with values provided through other variables, is presented in the following code:

[PRE13]

# See also

*   The following recipes in this chapter:
    *   *Specifying pipeline viewport and scissor test state*
    *   *Creating a graphics pipeline*

# Specifying a pipeline multisample state

Multisampling is a process that eliminates jagged edges of drawn primitives. In other words, it allows us to anti-alias polygons, lines and points. We define how multisampling is performed (and if at all) through a multisample state.

# How to do it...

1.  Create a variable of type `VkPipelineMultisampleStateCreateInfo` named `multisample_state_create_info`. Use the following values to initialize its members:
    *   `VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO` value for `sType`
    *   `nullptr` value for `pNext`
    *   `0` value for `flags`
    *   The number of samples generated per pixel for `rasterizationSamples`
    *   A `true` value if per sample shading should be enabled (only if `sampleRateShading` feature is enabled) or `false` otherwise for `sampleShadingEnable`
    *   A minimum fraction of uniquely shaded samples, when sample shading is enabled, for `minSampleShading`
    *   A pointer to an array of bitmasks that controls a fragment's static coverage or a `nullptr` value to indicate that no coverage is removed from the fragments (all bits are enabled in the mask) for `pSampleMask`
    *   A `true` value if a fragment's coverage should be generated based on the fragment's alpha value or `false` otherwise for `alphaToCoverageEnable`
    *   A `true` value if an alpha component of the fragment's color should be replaced with a `1.0` value for floating-point formats or with a maximum available value of a given format for fixed-point formats (only when the `alphaToOne` feature is enabled) or `false` otherwise value for `alphaToOneEnable`

# How it works...

The multisample state allows us to enable anti-aliasing of drawn primitives. Through it we can define the number of samples generated per fragment, enable per sample shading and specify the minimal number of uniquely shaded samples, and define a fragment's coverage parameters --the sample coverage mask, whether the coverage should be generated from an alpha component of the fragment's color. We can also specify if an alpha component should be replaced with a `1.0` value.

A multisample state is required only when rasterization is enabled.

To prepare a multisample state, we need to create a variable of a `VkPipelineMultisampleStateCreateInfo` type like this:

[PRE14]

In the preceding code, the function's parameters are used to initialize members of a `multisample_state_create_info` variable.

# See also

The following recipes in this chapter:

*   *Specifying pipeline rasterization state*
*   *Creating a graphics pipeline*

# Specifying a pipeline depth and stencil state

Usually, when we render a geometry, we want to mimic the way we see the world--objects further away are smaller, objects closer to us are larger and they cover the objects behind them (obscure our view). In modern 3D graphics, this last effect (objects further away being obscured by objects being nearer) is achieved through a depth test. The way in which a depth test is performed, is specified through a depth and stencil state of a graphics pipeline.

# How to do it...

1.  Create a variable of type `VkPipelineDepthStencilStateCreateInfo` named `depth_and_stencil_state_create_info`. Use the following values to initialize its members:
    *   `VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO` value for `sType`
    *   `nullptr` value for `pNext`
    *   `0` value for `flags`
    *   A `true` value if we want to enable a depth test or otherwise `false` for `depthTestEnable`
    *   A `true` value if we want to store the depth value in a depth buffer and otherwise `false` for `depthWriteEnable`
    *   A chosen compare operator (`never`, `less`, `less and equal`, `equal`, `greater and equal`, `greater`, `not equal`, `always`) controlling how the depth test is performed for `depthCompareOp`
    *   A `true` value if we want to enable additional depth bounds tests (only if `depthBounds` feature is enabled) or otherwise `false` for `depthBoundsTestEnable`
    *   A `true` value if we want to use a stencil test or `false` if we want to disable it for `stencilTestEnable`
    *   Use the following values to initialize members of a `front` field through which we set up stencil test parameters performed for front-facing polygons:
        *   Function performed when samples fail the stencil test for `failOp`.
        *   Action performed when samples pass the stencil test for `passOp`.
        *   Action taken when samples pass the stencil test but fail the depth test for `depthFailOp`.
        *   Operator (`never`, `less`, `less and equal`, `equal`, `greater and equal`, `greater`, `not equal`, `always`) used to perform the stencil test for `compareOp`.
        *   Mask selecting the bits of stencil values that take part in the stencil test for `compareMask`.
        *   Mask selecting which bits of a stencil value should be updated in a framebuffer for `writeMask.`
        *   Reference value used for stencil test comparison for `reference`.
    *   For `back` member setup stencil test parameters as described previously for front-facing polygons but, this time, for back-facing polygons.
    *   The value between `0.0` and `1.0` (inclusive) describing the minimal value of a depth bounds test for `minDepthBounds`.
    *   The value from `0.0` to `1.0` (inclusive) describing the maximal value of a depth bounds test for `maxDepthBounds`.

# How it works...

The depth and stencil state specifies whether a depth and/or stencil test should be performed. If any of them are enabled, we also define parameters for each of these tests.

A depth and stencil state is not required when rasterization is disabled or if a given subpass in a render pass does not use any depth/stencil attachments.

We need to specify how the depth test is performed (how depth values are compared) and if the depth value of a processed fragment should be written to a depth attachment when the fragment passes the test.

When the `depthBounds` feature is enabled, we can also activate an additional depth bounds test. This test checks whether the depth value of a processed fragment is inside a specified `minDepthBounds` - `maxDepthBounds` range. If it is not, the processed fragment is discarded as if it failed the depth test.

The stencil test allows us to perform additional tests on integer values associated with each fragment. It can be used for various purposes. As an example, we can define an exact part of the screen which can be updated during drawing, but, contrary to the scissor test, this area may have any shape, even if it is very complicated. Such an approach is used in deferred shading/lighting algorithms to restrict image areas that can be lit by a given light source. Another example of a stencil test is using it to show silhouettes of objects that are hidden behind other objects or highlighting objects selected by a mouse pointer.

In the case of an enabled stencil test, we need to define its parameters separately for front- and back-facing polygons. These parameters include actions performed when a given fragment fails the stencil test, passes it but fails the depth test, and passes both the stencil and depth test. For each situation, we define that current value in a stencil attachment should be kept intact, reset to `0`, replaced with a reference value, incremented or decremented with clamping (saturation) or with wrapping, or if the current value should be inverted bitwise. We also specify how the test is performed by setting the comparison operator (similar to the operator defined in the depth test), comparison and write masks which select the stencil value's bits that should take part in the test or which should be updated in a stencil attachment, and a reference value.

The sample source code that prepares a variable of a `VkPipelineDepthStencilStateCreateInfo` type, through which the depth and stencil test is defined, is presented in the following code:

[PRE15]

# See also

*   In [Chapter 6](2de4339d-8912-440a-89a6-fd1f84961448.xhtml), *Render Passes and Framebuffers*, see the following recipes:
    *   *Specifying subpass descriptions*
    *   *Creating a framebuffer*
*   The following recipes in this chapter:
    *   *Specifying pipeline rasterization state*
    *   *Creating a graphics pipeline*

# Specifying a pipeline blend state

Transparent objects are very common in the environment we see every day around us. Such objects are also common in 3D applications. To simulate transparent materials and simplify operations that the hardware needs to perform to render transparent objects, blending was introduced. It mixes the color of a processed fragment with a color that is already stored in a framebuffer. Parameters for this operation are prepared through a graphics pipeline's blend state.

# How to do it...

1.  Create a variable of type `VkPipelineColorBlendAttachmentState` named `attachment_blend_states`.
2.  For each color attachment used in a subpass in which a given graphics pipeline is bound, add a new element to the `attachment_blend_states` vector. If the `independentBlend` feature is not enabled, all elements added to the `attachment_blend_states` vector must be exactly the same. If this feature is enabled, elements may be different. Either way, use the following values to initialize members of each added element:
    *   A `true` value whether blending should be enabled and otherwise `false` for `blendEnable`
    *   The selected blend factor for the color of the processed (source) fragment for `srcColorBlendFactor`
    *   The selected blend factor for the color already stored in an (destination) attachment for `dstColorBlendFactor`
    *   The operator used to perform the blending operation on color components for `colorBlendOp`
    *   The selected blend factor for the alpha value of an incoming (source) fragment for `srcAlphaBlendFactor`
    *   The selected blend factor for the alpha value already stored in a destination attachment for `dstAlphaBlendFactor`
    *   The function used to perform the blending operation on alpha components for `alphaBlendOp`
    *   The color mask used to select which components should be written to in an attachment (no matter if blending is enabled or disabled) for `colorWriteMask`

3.  Create a variable of type `VkPipelineColorBlendStateCreateInfo` named `blend_state_create_info`. Use these values to initialize its members:
    *   `VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO` value for `sType`
    *   `nullptr` value for `pNext`
    *   `0` value for `flags`
    *   A `true` value if a logical operation should be performed between a fragment's color and a color already stored in an attachment (which disables blending) or `false` otherwise for `logicOpEnable`.
    *   The type of the logical operation to be performed (if logical operation is enabled) for `logicOp`
    *   A number of elements in the `attachment_blend_states` vector for `attachmentCount`
    *   A pointer to the first element of the `attachment_blend_states` vector for `pAttachments`
    *   Four floating-point values defining red, green, blue, and alpha components of a blend constant used for some of the blending factors for `blendConstants[4]`

# How it works...

The blending state is optional and is not required if rasterization is disabled or when there are no color attachments in a subpass, in which a given graphics pipeline is used.

The blending state is used mainly to define the parameters of a blending operation. But it also serves other purposes. In it we specify a color mask which selects which color components are updated (written to) during rendering. It also controls the state of a logical operation. When enabled, one of the specified logical operations is performed between a fragment's color and a color already written in a framebuffer.

A logical operation is performed only for attachments with integer and normalized integer formats.

Supported logical operations include:

*   `CLEAR`: Setting the color to zero
*   `AND`: Bitwise `AND` operation between the source (fragment's) color and a destination color (already stored in an attachment)
*   `AND_REVERSE`: Bitwise `AND` operation between source and inverted destination colors
*   `COPY`: Copying the source (fragment's) color without any modifications
*   `AND_INVERTED`: Bitwise `AND` operation between destination and inverted source colors
*   `NO_OP`: Leaving the already stored color intact
*   `XOR`: Bitwise excluded `OR` between source and destination colors
*   `OR`: Bitwise `OR` operation between the source and destination colors
*   `NOR`: Inverted bitwise `OR`
*   `EQUIVALENT`: Inverted `XOR`
*   `INVERT`: Inverted destination color
*   `OR_REVERSE`: Bitwise `OR` between the source color and inverted destination color
*   `COPY_INVERTED`: Copying bitwise inverted source color
*   `OR_INVERTED`: Bitwise `OR` operation between destination and inverted source color
*   `NAND`: Inverted bitwise `AND` operation
*   `SET`: Setting all color bits to ones

Blending is controlled separately for each color attachment used during rendering in a subpass in which a given graphics pipeline is bound. This means that we need to specify blending parameters for each color attachment used in rendering. But we need to remember that if the `independentBlend` feature is not enabled, blending parameters for each attachment must be exactly the same.

For blending, we specify the source and destination factors separately for color components and an alpha component. Supported blend factors include:

*   `ZERO`: `0`
*   `ONE`: `1`
*   `SRC_COLOR`: `<source component>`
*   `ONE_MINUS_SRC_COLOR`: 1 - `<source component>`
*   `DST_COLOR`: `<destination component>`
*   `ONE_MINUS_DST_COLOR`: 1 - `<destination component>`
*   `SRC_ALPHA`: `<source alpha>`
*   `ONE_MINUS_SRC_ALPHA`: 1 - `<source alpha>`
*   `DST_ALPHA`: `<destination alpha>`
*   `ONE_MINUS_DST_ALPHA`: 1 - `<destination alpha>`
*   `CONSTANT_COLOR`: `<constant color component>`
*   `ONE_MINUS_CONSTANT_COLOR`: 1 - `<constant color component>`
*   `CONSTANT_ALPHA`: `<alpha value of a constant color>`
*   `ONE_MINUS_CONSTANT_ALPHA`: 1 - `<alpha value of a constant color>`
*   `SRC_ALPHA_SATURATE`: `min( <source alpha>, 1 - <destination alpha> )`
*   `SRC1_COLOR`: `<component of a source's second color>` (used in dual source blending)
*   `ONE_MINUS_SRC1_COLOR`: 1 - `<component of a source's second color>` (from dual source blending)
*   `SRC1_ALPHA`: `<alpha component of a source's second color>` (in dual source blending)
*   `ONE_MINUS_SRC1_ALPHA`: 1 - `<alpha component of a source's second color>` (from dual source blending)

Some of the blending factors use constant color instead of a fragment's (source) color or color already stored in an attachment (destination). This constant color may be specified statically during the pipeline creation or dynamically (as one of the dynamic pipeline states) by the `vkCmdSetBlendConstants()` function call during command buffer recording.

Blending factors that use the source's second color (SRC1) may be used only when the `dualSrcBlend` feature is enabled.

The blending function that controls how blending is performed is also specified separately for color and alpha components. Blending operators include:

*   `ADD`: `<src component> * <src factor> + <dst component> * <dst factor>`
*   `SUBTRACT`: `<src component> * <src factor> - <dst component> * <dst factor>`
*   `REVERSE_SUBTRACT`: `<dst component> * <dst factor> - <src component> * <src factor>`
*   `MIN`: `min( <src component>, <dst component> )`
*   `MAX`: `max( <src component>, <dst component> )`

Enabling a logical operation disables blending.

The following is an example of setting up a blend state with both disabled logical operation and blending:

[PRE16]

The implementation of this recipe that fills a variable of the `VkPipelineColorBlendStateCreateInfo` type may look like this:

[PRE17]

# See also

*   In [Chapter 6](2de4339d-8912-440a-89a6-fd1f84961448.xhtml), *Render Passes and Framebuffers*, see the following recipes:
    *   *Specifying subpass descriptions*
    *   *Creating a framebuffer*
*   In [Chapter 9](0a69f5b5-142e-422b-aa66-5cb09a6467b3.xhtml), *Command Recording and Drawing*, see the following recipe 
    *   *Setting blend constants states dynamically *
*   The following recipes in this chapter:
    *   *Specifying pipeline rasterization state*
    *   *Creating a graphics pipeline*

# Specifying pipeline dynamic states

Creating a graphics pipeline requires us to provide lots of parameters. What's more, once set, these parameters can't be changed. Such an approach was taken to improve the performance of our application and present a stable and predictable environment to the driver. But, unfortunately, it is also uncomfortable for developers as they may need to create many pipeline objects with almost identical states that differ only in small details.

To circumvent this problem, dynamic states were introduced. They allow us to control some of the pipeline's parameters dynamically by recording specific functions in command buffers. And in order to do that, we need to specify which parts of the pipeline are dynamic. This is done by specifying pipeline dynamic states.

# How to do it...

1.  Create a variable of type `std::vector<VkDynamicState>` named `dynamic_states`. For each (unique) pipeline state that should be set dynamically, add a new element to the `dynamic_states` vector. The following values can be used:
    *   `VK_DYNAMIC_STATE_VIEWPORT`
    *   `VK_DYNAMIC_STATE_SCISSOR`
    *   `VK_DYNAMIC_STATE_LINE_WIDTH`
    *   `VK_DYNAMIC_STATE_DEPTH_BIAS`
    *   `VK_DYNAMIC_STATE_BLEND_CONSTANTS`
    *   `VK_DYNAMIC_STATE_DEPTH_BOUNDS`
    *   `VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK`
    *   `VK_DYNAMIC_STATE_STENCIL_WRITE_MASK`
    *   `VK_DYNAMIC_STATE_STENCIL_REFERENCE`

2.  Create a variable of type `VkPipelineDynamicStateCreateInfo` named `dynamic_state_creat_info`. Use the following values to initialize its members:
    *   `VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO` value for `sType`
    *   `nullptr` value for `pNext`
    *   `0` value for `flags`
    *   The number of elements in the `dynamic_states` vector for `dynamicStateCount`
    *   A pointer to the first element of the `dynamic_states` vector for `pDynamicStates`

# How it works...

Dynamic pipeline states were introduced to allow for some flexibility in setting the state of pipeline objects. There may not be too many different parts of the pipeline that can be set during command buffer recording, but the selection is a compromise between the performance, the simplicity of a driver, the capabilities of modern hardware, and the API's ease of use.

A dynamic state is optional. If we don't want to set any part of the pipeline dynamically, we don't need to do it.

The following parts of the graphics pipeline can be set dynamically:

*   **Viewport**: Parameters for all viewports are set through the `vkCmdSetViewport()` function call, but the number of viewports is still defined during the pipeline creation (refer to the *Specifying pipeline viewport and scissor test state* recipe)
*   **Scissor**: Parameters controlling the scissor test are set through the `vkCmdSetScissor()` function call, though the number of rectangles used for the scissor test are defined statically during the pipeline creation and must be the same as the number of viewports (refer to the *Specifying pipeline viewport and scissor test state* recipe)
*   **Line width**: The width of drawn lines is specified not in a graphics pipeline's state but through the `vkCmdSetLineWidth()` function (refer to the *Specifying pipeline rasterization state* recipe)
*   **Depth bias**: When enabled, the depth bias constant factor, slope factor, and maximum (or minimum) bias applied to a fragment's calculated depth value are defined through recording the `vkCmdSetDepthBias()` function (refer to the *Specifying pipeline depth and stencil state* recipe)
*   **Depth bounds**: When the depth bounds test is enabled, minimum and maximum values used during the test are specified with the `vkCmdSetDepthBounds()` function (refer to the *Specifying pipeline depth and stencil state* recipe)
*   **Stencil compare mask**: Specific bits of stencil values used during the stencil test are defined with the `vkCmdSetStencilCompareMask()` function call (refer to the *Specifying pipeline depth and stencil state* recipe)
*   **Stencil write mask**: Specifying which bits may be updated in a stencil attachment is done through the `vkCmdSetStencilWriteMask()` function (refer to the *Specifying pipeline depth and stencil state* recipe)
*   **Stencil reference value**: Setting the reference value used during the stencil test is performed with the `vkCmdSetStencilReference()` function call (refer to the *Specifying pipeline depth and stencil state* recipe)
*   **Blend constants**: Four floating-point values for red, green, blue, and alpha components of a blend constant are specified by recording a `vkCmdSetBlendConstants()` function (refer to the *Specifying pipeline blend state* recipe)

Specifying that a given state is set dynamically is done by creating an array (or a vector) of `VkDynamicState` enums with values corresponding to the chosen states and providing the array (named `dynamic_states` in the following code) to the variable of a `VkPipelineDynamicStateCreateInfo` type like this:

[PRE18]

# See also

*   The following recipes in this chapter:
    *   *Specifying pipeline viewport and scissor test state*
    *   *Specifying pipeline rasterization state*
    *   *Specifying pipeline depth and stencil state*
    *   *Specifying pipeline blend state*
    *   *Creating a graphics pipeline*
*   In [Chapter 9](0a69f5b5-142e-422b-aa66-5cb09a6467b3.xhtml), *Command Recording and Drawing*, see the following recipes:
    *   *Setting viewport state dynamically*
    *   *Setting scissors state dynamically*
    *   *Setting depth bias state dynamically*
    *   *Setting blend constants state dynamically*

# Creating a pipeline layout

Pipeline layouts are similar to descriptor set layouts. Descriptor set layouts are used to define what types of resources form a given descriptor set. Pipeline layouts define what types of resources can be accessed by a given pipeline. They are created from descriptor set layouts and, additionally, push constant ranges.

Pipeline layouts are needed for the pipeline creation as they specify the interface between shader stages and shader resources through a set, binding, array element address. The same address needs to be used in shaders (through a layout qualifier) so they can successfully access a given resource. But even if a given pipeline doesn't use any descriptor resources, we need to create a pipeline layout to inform the driver that no such interface is needed.

# How to do it...

1.  Take the handle of a logical device stored in a variable of type `VkDevice` named `logical_device`.
2.  Create a `std::vector` variable named `descriptor_set_layouts` with elements of type `VkDescriptorSetLayout`. For each descriptor set, through which resources will be accessed from shaders in a given pipeline, add a descriptor set layout to the `descriptor_set_layouts` vector.
3.  Create a `std::vector<VkPushConstantRange>` variable named `push_constant_ranges`. Add new elements to this vector for each separate range (a unique set of push constants used by different shader stages) and use the following values to initialize its members:
    *   A logical `OR` of all shader stages that access a given push constant for `stageFlags`
    *   The value that is a multiple of 4 for the offset at which a given push constant starts in memory for `offset`
    *   The value that is a multiple of 4 for the size of a memory for a given push constant for `size`

4.  Create a variable of type `VkPipelineLayoutCreateInfo` named `pipeline_layout_create_info`. Use the following values to initialize its members:
    *   `VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO` value for `sType`
    *   `nullptr` value for `pNext`
    *   `0` value for `flags`
    *   The number of elements in the `descriptor_set_layouts` vector for `setLayoutCount`
    *   A pointer to the first element of the `descriptor_set_layouts` vector for `pSetLayouts`
    *   The number of elements in the `push_constant_ranges` vector for `pushConstantRangeCount`
    *   A pointer to the first element of the `push_constant_ranges` for `pPushConstantRanges`
5.  Create a variable of type `VkPipelineLayout` named `pipeline_layout`, in which the handle of the created pipeline layout will be stored.
6.  Make the following call: `vkCreatePipelineLayout( logical_device, &pipeline_layout_create_info, nullptr, &pipeline_layout )` for which provide the `logical_device` variable, a pointer to the `pipeline_layout_create_info` variable, a `nullptr` value, and a pointer to the `pipeline_layout` variable.
7.  Make sure the call was successful by checking if it returned the `VK_SUCCESS` value.

# How it works...

A pipeline layout defines the set of resources that can be accessed from shaders of a given pipeline. When we record command buffers, we bind descriptor sets to selected indices (refer to the *Binding descriptor sets* recipe). This index corresponds to a descriptor set layout at the same index in the array used during the pipeline layout creation (the `descriptor_set_layouts` vector from this recipe). The same index needs to be specified inside shaders through a `layout( set = <index>, binding = <number> )` qualifier for the given resource to be properly accessed.

![](img/B05542-08-03-1.png)

Usually, multiple pipelines will access different resources. During command buffer recording, we bind a given pipeline and descriptor sets. Only after that can we issue drawing commands. When we switch from one pipeline to another, we need to bind new descriptor sets according to the pipeline's needs. But frequently binding different descriptor sets may impact the performance of our application. That's why it is good to create pipelines with similar (or compatible) layouts and bind descriptor sets that do not change too often (that are common for many pipelines) to indices near the 0 (or near the start of a layout). This way, when we switch pipelines, descriptor sets near the start of the pipeline layout (from index 0 to some index N) can still be used and don't need to be updated. It is only necessary to bind the different descriptor sets--those that are placed at greater indices (after the given index N). But one additional condition must be met-- to be similar (or compatible), the pipeline layouts must use the same push constant ranges.

We should bind descriptor sets that are common for many pipelines near the start of a pipeline layout (near the `0`^(th) index).

Pipeline layouts also define the ranges of push constants. They allow us to provide a small set of constant values to shaders. They are much faster than updating descriptor sets, but memory that can be consumed by push constants is also much smaller--it is at least 128 bytes for all ranges defined in a pipeline layout. Different hardware may offer more memory for push constants, but we can't rely on it if we target hardware from various vendors.

As an example, when we want to define a different range for each stage in a graphics pipeline, we have more or less 128 / 5 = 26 bytes per stage for a push constant. Of course, we can define ranges that are common for multiple shader stages. But each shader stage may have access to only one push constant range.

The preceding example is the worst case. Usually not all stages will use different push constant ranges. Quite commonly, stages won't require access to a push constant range at all. So there should be enough memory for several 4-component vectors or a matrix or two.

Each pipeline stage can access only one push constant range.

We also need to remember that the size and an offset of a push constant range must be a multiple of 4.

In the following code, we can see a source code that implements this recipe. Descriptor set layouts and ranges of push constants are provided through `descriptor_set_layouts` and `push_constant_ranges` variables, respectively:

[PRE19]

# See also

*   In [Chapter 5](fe2cb528-9d22-49db-a05b-372bce2f87ee.xhtml), *Descriptor Sets*, see the following recipe:
    *   *Binding descriptor sets*
*   In [Chapter 7](97217f0d-bed7-4ae1-a543-b4d599f299cf.xhtml), *Shaders*, see the following recipes:
    *   *Writing a vertex shader that multiplies vertex position by a projection matrix*
    *   *Using push constants in shaders*
*   The following recipes in this chapter:
    *   *Creating a graphics pipeline*
    *   *Creating a compute pipeline*
    *   *Creating a pipeline layout with push constants, sampled image, and a buffer*
    *   *Destroying a pipeline layout*
*   In [Chapter 9](0a69f5b5-142e-422b-aa66-5cb09a6467b3.xhtml), *Command Recording and Drawing*, see the following recipe:
    *   *Providing data to shaders through push constants*

# Specifying graphics pipeline creation parameters

Creating a graphics pipeline requires us to prepare many parameters controlling its many different aspects. All these parameters are grouped into a variable of type `VkGraphicsPipelineCreateInfo` which needs to be properly initialized before we can use it to create a pipeline.

# How to do it...

1.  Create a variable of a bitfield type `VkPipelineCreateFlags` named `additional_options` through which provide additional pipeline creation options:
    *   **Disable optimization**: specifies that the created pipeline won't be optimized, but the creation process may be faster
    *   **Allow derivatives**: specifies that other pipelines may be created from it
    *   **Derivative**: specifies that this pipeline will be created based on another, already created pipeline
2.  Create a variable of type `std::vector<VkPipelineShaderStageCreateInfo>` named `shader_stage_create_infos`. For each shader stage enabled in a given pipeline, add a new element to the `shader_stage_create_infos` vector, specifying the stage's parameters. At least the vertex shader stage must be present in the `shader_stage_create_infos` vector (refer to the *Specifying pipeline shader stages* recipe).
3.  Create a variable of type `VkPipelineVertexInputStateCreateInfo` named `vertex_input_state_create_info` through which vertex bindings, attributes, and input state are specified (refer to the *Specifying pipeline vertex binding description, attribute description, and input state* recipe).
4.  Create a variable of type `VkPipelineInputAssemblyStateCreateInfo` named `input_assembly_state_create_info`. Use it to define how drawn vertices are assembled into polygons (refer to the *Specifying pipeline input assembly state* recipe).
5.  If a tessellation should be enabled in a given pipeline, create a variable of type `VkPipelineTessellationStateCreateInfo` named `tessellation_state_create_info` in which the number of control points forming a patch is defined (refer to the *Specifying pipeline tessellation state* recipe).
6.  If a rasterization process won't be disabled in a given pipeline, create a variable of type `VkPipelineViewportStateCreateInfo` named `viewport_state_create_info`. In the variable, specify viewport and scissor test parameters (refer to the *Specifying pipeline viewport and scissor test state* recipe).
7.  Create a variable of type `VkPipelineRasterizationStateCreateInfo` named `rasterization_state_create_info` that defines the properties of a rasterization (refer to the *Specifying pipeline rasterization state* recipe).

8.  If rasterization is enabled in a given pipeline, create a variable of type `VkPipelineMultisampleStateCreateInfo` named `multisample_state_create_info` that defines multisampling (anti-aliasing) parameters (refer to the *Specifying pipeline multisample state* recipe).
9.  If rasterization is active and depth and/or stencil attachments are used during drawing with a given pipeline bound, create a variable of type `VkPipelineDepthStencilStateCreateInfo` named `depth_and_stencil_state_create_info`. Use it to define parameters of depth and stencil tests (refer to the *Specifying pipeline depth and stencil state* recipe).
10.  If rasterization is not disabled, create a variable of type `VkPipelineColorBlendStateCreateInfo` named `blend_state_create_info` through which to specify parameters of operations performed on fragments (refer to the *Specifying pipeline blend state* recipe).
11.  If there are parts of the pipeline which should be set dynamically, create a variable of type `VkPipelineDynamicStateCreateInfo` named `dynamic_state_creat_info` that defines those dynamically set parts (refer to the *Specifying pipeline dynamic states* recipe).
12.  Create a pipeline layout and store its handle in a variable of type `VkPipelineLayout` named `pipeline_layout`.
13.  Take the handle of a render pass in which drawing with a given pipeline bound will be performed. Use the render pass handle to initialize a variable of type `VkRenderPass` named `render_pass` (refer to the *Creating a render pass* recipe from [Chapter 6](2de4339d-8912-440a-89a6-fd1f84961448.xhtml), *Render Passes and Framebuffers*).
14.  Create a variable of type `uint32_t` named `subpass`. Store the index of the render pass's subpass in which a given pipeline will be used during drawing operations (refer to the *Specifying subpass descriptions* recipe from [Chapter 6](2de4339d-8912-440a-89a6-fd1f84961448.xhtml), *Render Passes and Framebuffers*).
15.  Create a variable of type `VkGraphicsPipelineCreateInfo` named `graphics_pipeline_create_info`. Use the following values to initialize its members:
    *   `VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO` value for `sType`
    *   `nullptr` value for `pNext`
    *   `additional_options` variable for `flags`
    *   The number of elements in the `shader_stage_create_infos` vector for `stageCount`
    *   A pointer to the first element of the `shader_stage_create_infos` vector for `pStages`
    *   A pointer to the `vertex_input_state_create_info` variable for `pVertexInputState`
    *   A pointer to the `input_assembly_state_create_info` variable for `pInputAssemblyState`
    *   A pointer to the `tessellation_state_create_info` variable if tessellation should be active or a `nullptr` value if tessellation should be disabled for `pTessellationState`
    *   A pointer to the `viewport_state_create_info` variable if rasterization is active or a `nullptr` value if rasterization is disabled for `pViewportState`
    *   A pointer to the `rasterization_state_create_info` variable for `pRasterizationState`
    *   A pointer to the `multisample_state_create_info` variable if rasterization is enabled and a `nullptr` value otherwise for `pMultisampleState`
    *   A pointer to the `depth_and_stencil_state_create_info` variable if rasterization is enabled and there is a depth and/or stencil attachment used in the `subpass` or a `nullptr` value otherwise for `pDepthStencilState`
    *   A pointer to the `blend_state_create_info` variable if rasterization is enabled and there is a color attachment used in the `subpass` or a `nullptr` value otherwise for `pColorBlendState`
    *   A pointer to the `dynamic_state_creat_info` variable if there are parts of the pipeline that should be setup dynamically, or a `nullptr` value if the whole pipeline is prepared statically for `pDynamicState`
    *   The `pipeline_layout` variable for `layout`
    *   The `render_pass` variable for `renderPass`
    *   The `subpass` variable for `subpass`
    *   If the pipeline should derive from another, already created pipeline, provide the handle of the parent pipeline, otherwise provide a `VK_NULL_HANDLE` for `basePipelineHandle`
    *   If a pipeline should derive from another pipeline that is created within the same batch of pipelines, provide the index of a parent pipeline, otherwise provide a `-1` value for `basePipelineIndex`

# How it works...

Preparing data for a graphics pipeline creation is performed in multiple steps and each step specifies different parts of a graphics pipeline. All of these parameters are gathered in a variable of type `VkGraphicsPipelineCreateInfo`.

During the pipeline creation, we can provide many parameters of type `VkGraphicsPipelineCreateInfo`, each one specifying attributes of a single pipeline that will be created.

When a graphics pipeline is created, we can use it for drawing by binding it to the command buffer before recording a drawing command. Graphics pipelines can be bound to command buffers only inside render passes (after the beginning of a render pass is recorded). During the pipeline creation, we specify inside which render pass a given pipeline will be used. However, we are not limited only to the provided render pass. We can also use the same pipeline with other render passes if they are compatible with the specified one (refer to the Creating a render pass recipe from [Chapter 6](2de4339d-8912-440a-89a6-fd1f84961448.xhtml), *Render Passes and Framebuffers*).

It is a rare situation when each created pipeline doesn't have any common state with other pipelines. That's why, to speed up the pipeline creation, it is possible to specify that a pipeline can be a parent of other pipelines (allow derivatives) or that the pipeline will be a child of (derived from) another pipeline. To use this feature and shorten the time needed to create a pipeline, we can use `basePipelineHandle` or `basePipelineIndex` members of variables of type `VkGraphicsPipelineCreateInfo` (the `graphics_pipeline_create_info` variable in this recipe).

The `basePipelineHandle` member allows us to specify a handle of an already created pipeline, which should be a parent of the newly created one.

The `basePipelineIndex` member is used when we create multiple pipelines at once. Through it we specify an index into the array with elements of type `VkGraphicsPipelineCreateInfo` provided to the `vkCreateGraphicsPipelines()` function. This index points to a parent pipeline that will be created along with the child pipeline in the same, single function call. As they are created together, we can't provide a handle, that's why there is a separate field for an index. One requirement is that the index of a parent pipeline must be smaller than the index of a child pipeline (it must appear earlier in the list of `VkGraphicsPipelineCreateInfo` elements, before the element that describes the derived pipeline).

We can't use both `basePipelineHandle` and `basePipelineIndex` members; we can provide value only for one of them. If we want to specify a handle, we must provide a `-1` value for the `basePipelineIndex` field. If we want to specify an index, we need to provide a `VK_NULL_HANDLE` value for the `basePipelineHandle` member.

The rest of the parameters are described in earlier recipes of this chapter. The following is an example of how to use them to initialize the members of the variable of type `VkGraphicsPipelineCreateInfo`:

[PRE20]

# See also

The following recipes in this chapter:

*   *Specifying pipeline shader stages*
*   *Specifying pipeline vertex binding description, attribute description, and input state*
*   *Specifying pipeline input assembly state*
*   *Specifying pipeline tessellation state*
*   *Specifying pipeline viewport and scissor test state*
*   *Specifying pipeline rasterization state*
*   *Specifying pipeline multisample state*
*   *Specifying pipeline depth and stencil state*
*   *Specifying pipeline blend state*
*   *Specifying pipeline dynamic states*
*   *Creating a pipeline layout*

# Creating a pipeline cache object

Creating a pipeline object is a complicated and time-consuming process from the driver's perspective. A pipeline object is not a simple wrapper for parameters set during the creation. It involves preparing the states of all programmable and fixed pipeline stages, setting an interface between shaders and descriptor resources, compiling and linking shader programs, and performing error checking (that is, checking if shaders are linked properly). Results of these operations may be stored in a cache. This cache can then be reused to speed up the creation of pipeline objects with similar properties. To be able to use a pipeline cache object, we first need to create it.

# How to do it...

1.  Take the handle of a logical device and store it in a variable of type `VkDevice` named `logical_device`.
2.  If available (that is, retrieved from other caches), prepare data to initialize a newly created cache object. Store the data in a variable of type `std::vector<unsigned char>` named `cache_data`.
3.  Create a variable of type `VkPipelineCacheCreateInfo` named `pipeline_cache_create_info`. Use the following values to initialize its members:
    *   `VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO` value for `sType`.
    *   `nullptr` value for `pNext`.
    *   `0` value for `flags`.
    *   The number of elements in the `cache_data` vector (size of the initialization data in bytes) for `initialDataSize`.
    *   A pointer to the first element of the `cache_data` vector for `pInitialData`.

4.  Create a variable of type `VkPipelineCache` named `pipeline_cache` in which the handle of the created cache object will be stored.
5.  Make the following function call: `vkCreatePipelineCache( logical_device, &pipeline_cache_create_info, nullptr, &pipeline_cache )`. For the call, provide the `logical_device` variable, a pointer to the `pipeline_cache_create_info` variable, a `nullptr` value, and a pointer to the `pipeline_cache` variable.
6.  Make sure the call was successful by checking if it returned a `VK_SUCCESS` value.

# How it works...

A pipeline cache, as the name suggests, stores the results of a pipeline preparation process. It is optional and can be omitted, but when used, can significantly speed up the creation of pipeline objects.

To use a cache during the pipeline creation, we just need to create a cache object and provide it to the pipeline creating function. The driver automatically caches the results in the provided object. Also, if the cache contains any data, the driver automatically tries to use it for the pipeline creation.

The most common scenario of using a pipeline cache object, is to store its contents in a file and reuse them between separate executions of the same application. The first time we run our application, we create an empty cache and all the pipelines we need. Next, we retrieve the cache data and save it to a file. Next time the application is executed, we also create the cache, but this time we initialize it with the contents read from a previously created file. From now on, each time we run our application, the process of creating pipelines should be much shorter. Of course, when we create only small number of pipelines, we probably won't notice any improvement. But modern 3D applications, especially games, may have tens, hundreds, or sometimes even thousands of different pipelines (due to shader variations). In such situations, the cache can significantly boost the process of creating all of them.

Let's assume the cache data is stored in a vector variable named `cache_data`. It may be empty or initialized with contents retrieved from previous pipeline creations. The process of creating a pipeline cache that uses this data is presented in the following code:

[PRE21]

# See also

The following recipes in this chapter:

*   *Retrieving data from a pipeline cache*
*   *Merging multiple pipeline cache objects*
*   *Creating a graphics pipeline*
*   *Creating a compute pipeline*
*   *Creating multiple graphics pipelines on multiple threads*
*   *Destroying a pipeline cache*

# Retrieving data from a pipeline cache

A cache allows us to improve the performance of creating multiple pipeline objects. But for us to be able to use the cache each time we execute our application, we need a way to store the contents of the cache and reuse it any time we want. To do that, we can retrieve the data gathered in a cache.

# How to do it...

1.  Take the handle of a logical device and use it to initialize a variable of type `VkDevice` named `logical_device`.
2.  Store the handle of a pipeline cache, from which data should be retrieved, in a variable of type `VkPipelineCache` named `pipeline_cache`.

3.  Prepare a variable of type `size_t` named `data_size`.
4.  Call `vkGetPipelineCacheData( logical_device, pipeline_cache, &data_size, nullptr )` providing the `logical_device` and `pipeline_cache` variables, a pointer to the `data_size` variable, and a `nullptr` value.
5.  If a function call was successful (a `VK_SUCCESS` value was returned), the size of memory that can hold the cache contents is stored in the `data_size` variable.
6.  Prepare a storage space for the cache contents. Create a variable of type `std::vector<unsigned char>` named `pipeline_cache_data`.
7.  Resize the `pipeline_cache_data` vector to be able to hold at least `data_size` number of elements.
8.  Call `vkGetPipelineCacheData( logical_device, pipeline_cache, &data_size, pipeline_cache_data.data() )` but this time, apart from the previously used parameters, additionally provide a pointer to the first element of the `pipeline_cache_data` vector as the last parameter.
9.  If the function returns successfully, cache contents are stored in the `pipeline_cache_data` vector.

# How it works...

Retrieving pipeline cache contents is performed in a typical Vulkan double-call of a single function. The first call of the `vkGetPipelineCacheData()` function, stores the total number of bytes required to hold the entire data retrieved from the pipeline cache. This allows us to prepare enough storage for the data:

[PRE22]

Now, when we are ready to acquire the cache contents, we can call the `vkGetPipelineCacheData()` function once more. This time the last parameter must point to the beginning of the prepared storage. A successful call writes the provided number of bytes to the indicated memory:

[PRE23]

Data retrieved in this way can be used directly to initialize the contents of any other newly created cache object.

# See also

The following recipes in this chapter:

*   *Creating a pipeline cache object*
*   *Merging multiple pipeline cache objects*
*   *Creating a graphics pipeline*
*   *Creating a compute pipeline*
*   *Destroying a pipeline cache*

# Merging multiple pipeline cache objects

It may be a common scenario that we will have to create multiple pipelines in our application. To shorten the time needed to create them all, it may be a good idea to split the creation into multiple threads executed simultaneously. Each such thread should use a separate pipeline cache. After all the threads are finished, we would like to reuse the cache next time our application is executed. For this purpose, it is best to merge multiple cache objects into one.

# How to do it...

1.  Store the handle of a logical device in a variable of type `VkDevice` named `logical_device`.
2.  Take the cache object into which other caches will be merged. Using its handle, initialize a variable of type `VkPipelineCache` named `target_pipeline_cache`.
3.  Create a variable of type `std::vector<VkPipelineCache>` named `source_pipeline_caches`. Store the handles of all pipelines caches that should be merged in the `source_pipeline_caches` vector (make sure none of the cache objects is the same as the `target_pipeline_cache` cache).
4.  Make the following call: `vkMergePipelineCaches( logical_device, target_pipeline_cache, static_cast<uint32_t>(source_pipeline_caches.size()), source_pipeline_caches.data() )`. For the call, provide the `logical_device` and `target_pipeline_cache` variables, the number of elements in the `source_pipeline_caches` vector, and a pointer to the first element of the `source_pipeline_caches` vector.
5.  Make sure the call was successful and that it returned a `VK_SUCCESS` value.

# How it works...

Merging pipeline caches allows us to combine separate cache objects into one. This way it is possible to perform multiple pipeline creations that use separate caches in multiple threads and then merge the results into one, common cache object. Separate threads may also use the same pipeline cache object, but access to the cache may be guarded by a mutex in the driver, thus making splitting the job into multiple threads quite useless. Saving one cache data in a file is simpler than managing multiple ones. And, during the merging operation, duplicate entries should be removed by the driver, thus saving us some additional space and memory.

Merging multiple pipeline cache objects is performed like this:

[PRE24]

We need to remember that a cache, into which we merge other cache objects, cannot appear in the list of (source) caches to be merged.

# See also

The following recipes in this chapter:

*   *Creating a pipeline cache object*
*   *Retrieving data from a pipeline cache*
*   *Creating a graphics pipeline*
*   *Creating a compute pipeline*
*   *Creating multiple graphics pipelines on multiple threads*
*   *Destroying a pipeline cache*

# Creating a graphics pipeline

A graphics pipeline is the object that allows us to draw anything on screen. It controls how the graphics hardware performs all the drawing-related operations, which transform vertices provided by the application into fragments appearing on screen. Through it we specify shader programs used during drawing, the state and parameters of tests such as depth and stencil, or how the final color is calculated and written to any of the subpass attachments. It is one of the most important objects used by our application. Before we can draw anything, we need to create a graphics pipeline. If we want, we can create multiple pipelines at once.

# How to do it...

1.  Take the handle of a logical device and store it in a variable of type `VkDevice` named `logical_device`.
2.  Create a variable of type `std::vector<VkGraphicsPipelineCreateInfo>` named `graphics_pipeline_create_infos`. For each pipeline that should be created, add an element to the `graphics_pipeline_create_infos` vector describing the pipeline's parameters (refer to the *Specifying graphics pipeline creation parameters* recipe).
3.  If a pipeline cache should be used during the creation process, store its handle in a variable of type `VkPipelineCache` named `pipeline_cache`.
4.  Create a variable of type `std::vector<VkPipeline>` named `graphics_pipelines`, in which handles of the created `pipeline` will be stored. Resize the vector to hold the same number of elements as the `graphics_pipeline_create_infos` vector.
5.  Call `vkCreateGraphicsPipelines( logical_device, pipeline_cache, static_cast<uint32_t>(graphics_pipeline_create_infos.size()), graphics_pipeline_create_infos.data(), nullptr, graphics_pipelines.data() )` and provide the `logical_device` variable, the `pipeline_cache` variable or a `nullptr` value if no cache is used during the pipeline creation, the number of elements in the `graphics_pipeline_create_infos` vector, a pointer to the first element of the `graphics_pipeline_create_info` vector, a `nullptr` value, and a pointer to the first element of the `graphics_pipeline` vector.
6.  Make sure all the pipelines were successfully created by checking whether the call returned a `VK_SUCCESS` value. If any of the pipelines weren't created successfully, other values will be returned.

# How it works...

A graphics pipeline allows us to draw anything on screen. It controls the parameters of all programmable and fixed stages of the pipeline realized by the graphics hardware. A simplified diagram of a graphics pipeline is presented in the following image. White blocks represent programmable stages, gray ones are the fixed parts of the pipeline:

![](img/image_08_004.png)

Programmable stages consist of vertex, tessellation control and evaluation, and geometry and fragment shaders, of which only the vertex stage is obligatory. The rest are optional and enabling them depends on the parameters specified during the pipeline creation. As an example, if rasterization is disabled, there is no fragment shader stage. If we enable the tessellation stage, we need to provide both tessellation control and evaluation shaders.

A graphics pipeline is created with a `vkCreateGraphicsPipelines()` function. It allows us to create multiple pipelines at once. We need to provide an array of variables of type `VkGraphicsPipelineCreateInfo`, a number of elements in this array, and a pointer to an array with elements of type `VkPipeline`. This array must be large enough to hold the same number of elements as the input array with elements of type `VkGraphicsPipelineCreateInfo` (the `graphics_pipeline_create_infos` vector). When we prepare elements to the `graphics_pipeline_create_infos` vector and want to use its `basePipelineIndex` member to specify a parent pipeline created within the same function call, we provide an index into the `graphics_pipeline_create_infos` vector.

The implementation of this recipe is presented in the following code:

[PRE25]

# See also

The following recipe in this chapter:

*   *Specifying graphics pipeline creation parameters*
*   *Creating a pipeline cache object*
*   *Binding a pipeline object*
*   *Creating a graphics pipeline with vertex and fragment shaders, depth test enabled, and with dynamic viewport and scissor tests*
*   *Creating multiple graphics pipelines on multiple threads*
*   *Destroying a pipeline*

# Creating a compute pipeline

A compute pipeline is the second type of pipeline available in the Vulkan API. It is used for dispatching compute shaders, which can perform any mathematical operations. And as the compute pipeline is much simpler than the graphics pipeline, we create it by providing far fewer parameters.

# How to do it...

1.  Take the handle of a logical device and initialize a variable of type `VkDevice` named `logical_device` with it.
2.  Create a variable of a bitfield type `VkPipelineCreateFlags` named `additional_options`. Initialize it with any combination of these additional pipeline creation options:

    *   **Disable optimization**: specifies that the created pipeline won't be optimized, but the creation process may be faster
    *   **Allow derivatives**: specifies that other pipelines may be created from it
    *   **Derivative**: specifies that this pipeline will be created based on another, already created pipeline
3.  Create a variable of type `VkPipelineShaderStageCreateInfo` named `compute_shader_stage` through which specify a single compute shader stage (refer to the *Specifying pipeline shader stages* recipe).
4.  Create a pipeline layout and store its handle in a variable of type `VkPipelineLayout` named `pipeline_layout`.
5.  If a pipeline cache should be used during the pipeline creation, store the handle of a created cache object in a variable of type `VkPipelineCache` named `pipeline_cache`.
6.  Create a variable of type `VkComputePipelineCreateInfo` named `compute_pipeline_create_info`. Use the following values to initialize its members:
    *   `VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO` value for `sType`
    *   `nullptr` value for `pNext`
    *   `additional_options` variable for `flags`
    *   `compute_shader_stage` variable for `stage`
    *   `pipeline_layout` variable for `layout`
    *   If the pipeline should be a child of another pipeline, provide the handle of a parent pipeline or otherwise a `VK_NULL_HANDLE` value for `basePipelineHandle`
    *   `-1` value for `basePipelineIndex`

7.  Create a variable of type `VkPipeline` named `compute_pipeline` in which a handle of the created compute pipeline will be stored.
8.  Call `vkCreateComputePipelines( logical_device, pipeline_cache, 1, &compute_pipeline_create_info, nullptr, &compute_pipeline )` and provide the `logical_device` variable, the `pipeline_cache` variable if caching should be enabled or a `VK_NULL_HANDLE` value otherwise, `1` value, a pointer to the `compute_pipeline_create_info` variable, a `nullptr` value, and a pointer to the `compute_pipeline` variable.
9.  Make sure the call was successful by checking if it returned a `VK_SUCCESS` value.

# How it works...

We use compute pipelines when we want to dispatch compute shaders. A compute pipeline consists of only a single compute shader stage (though the hardware may implement additional stages if needed).

Compute pipelines cannot be used inside render passes.

Compute shaders don't have any input or output variables, apart from some built-in values. For the input and output data, only uniform variables (buffers or images) can be used (refer to the *Writing compute shaders* recipe from [Chapter 7](97217f0d-bed7-4ae1-a543-b4d599f299cf.xhtml), *Shaders*). That's why, though the compute pipeline is simpler, compute shaders are more universal and can be used to perform mathematical operations or operations that operate on images.

Compute pipelines, similar to graphics pipelines, can be created in bulks and multiple variables of type `VkComputePipelineCreateInfo` just need to be provided to the compute pipeline creating function. Also, compute pipelines can be parents of other compute pipelines and can derive from other parent pipelines. All this speeds up the creation process. To use this ability, we need to provide appropriate values for `basePipelineHandle` or `basePipelineIndex` members of variables of type `VkComputePipelineCreateInfo` (refer to the *Creating a graphics pipeline* recipe).

The simplified process of creating a single compute pipeline is presented in the following code:

[PRE26]

# See also

*   In [Chapter 7](97217f0d-bed7-4ae1-a543-b4d599f299cf.xhtml), *Shaders*, see the following recipe:
    *   *Writing compute shaders*
*   The following recipes in this chapter:
    *   *Specifying pipeline shader stages*
    *   *Creating a pipeline layout*
    *   *Creating a pipeline cache object*
    *   *Destroying a pipeline*

# Binding a pipeline object

Before we can issue drawing commands or dispatch computational work, we need to set up all the required states for the command to be successfully performed. One of the required states is binding a pipeline object to the command buffer--a graphics pipeline if we want to draw objects on screen or a compute pipeline if we want to perform computational work.

# How to do it...

1.  Take the handle of a command buffer and store it in a variable of type `VkCommandBuffer` named `command_buffer`. Make sure the command buffer is in the recording state.
2.  If a graphics pipeline needs to be bound, make sure the beginning of a render pass has already been recorded in the `command_buffer`. If a compute pipeline should be bound, make sure no render pass is started or any render passes are finished in the `command_buffer`.
3.  Take the handle of a pipeline object. Use it to initialize a variable of type `VkPipeline` named `pipeline`.
4.  Call `vkCmdBindPipeline( command_buffer, pipeline_type, pipeline )`. Provide the `command_buffer` variable, the type of the pipeline (graphics or compute) that is being bound to the command buffer, and the `pipeline` variable.

# How it works...

A pipeline needs to be bound before we can draw or dispatch computational work in a command buffer. Graphics pipelines can be bound only inside render passes--the one specified during pipeline creation or a compatible one. Compute pipelines cannot be used inside render passes. If we want to use them, any started render pass needs to be finished.

Binding a pipeline object is achieved with a single function call like this:

[PRE27]

# See also

*   In [Chapter 3](fc38e0ae-51aa-4f6f-8fb3-551861273018.xhtml), *Command Buffers and Synchronization*, see the recipe:
    *   *Beginning a command buffer recording operation*
*   In [Chapter 6](2de4339d-8912-440a-89a6-fd1f84961448.xhtml), *Render Passes and Framebuffers*, see the following recipes:
    *   *Beginning a render pass*
    *   *Ending a render pass*
*   The following recipes in this chapter:
    *   *Creating a graphics pipeline*
    *   *Creating a compute pipeline*

# Creating a pipeline layout with a combined image sampler, a buffer, and push constant ranges

We know how to create descriptor set layouts and use them to create a pipeline layout. Here, in this sample recipe, we will have a look at how to create a specific pipeline layout--one which allows a pipeline to access a combined image sampler, a uniform buffer, and a selected number of push constant ranges.

# How to do it...

1.  Take the handle of a logical device and store it in a variable of type `VkDevice` named `logical_device`.
2.  Create a variable of type `std::vector<VkDescriptorSetLayoutBinding>` named `descriptor_set_layout_bindings`.
3.  Add a new element to the `descriptor_set_layout_bindings` vector and use the following values to initialize its members:
    *   `0` value for `binding`.
    *   `VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE` value for `descriptorType`.
    *   `1` value for `descriptorCount`.
    *   `VK_SHADER_STAGE_FRAGMENT_BIT` value for `stageFlags`.
    *   `nullptr` value for `pImmutableSamplers`.
4.  Add a second member to the `descriptor_set_layout_bindings` vector and use the following values to initialize its members:
    *   `1` value for `binding`.
    *   `VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER` value for `descriptorType`.
    *   `1` value for `descriptorCount`.
    *   `VK_SHADER_STAGE_VERTEX_BIT` value for `stageFlags`.
    *   `nullptr` value for `pImmutableSamplers`.

5.  Create a descriptor set layout using the `logical_device` and `descriptor_set_layout_bindings` variables and store it in a variable of type `VkDescriptorSetLayout` named `descriptor_set_layout` (refer to the *Creating a descriptor set layout* recipe from [Chapter 5](fe2cb528-9d22-49db-a05b-372bce2f87ee.xhtml), *Descriptor Sets*).
6.  Create a variable of type `std::vector<VkPushConstantRange>` named `push_constant_ranges` and initialize it with the desired number of push constant ranges, each with desired values (refer to the *Creating a pipeline layout* recipe).
7.  Create a variable of type `VkPipelineLayout` named `pipeline_layout` in which the handle of the created pipeline layout will be stored.
8.  Create the pipeline layout using the `logical_device`, `descriptor_set_layout` and `push_constant_ranges` variables. Store the created handle in the `pipeline_layout` variable (refer to the *Creating a pipeline layout* recipe).

# How it works...

In this recipe, we assume that we want to create a graphics pipeline that needs access to a uniform buffer and a combined image sampler. This is a common situation--we use the uniform buffer in a vertex shader to transform vertices from the local space to the clip space. A fragment shader is used for texturing so it needs access to a combined image sampler descriptor.

We need to create a descriptor set that contains these two types of resources. For this purpose, we create a layout for it, which defines a uniform buffer used in a vertex shader and a combined image sampler accessed in a fragment shader:

[PRE28]

Using such a descriptor set layout, we can create a pipeline layout using an additional vector with information for ranges of push constants:

[PRE29]

Now, when we create a pipeline with such a layout, we can bind one descriptor set to index `0`. This descriptor set must have two descriptor resources--a combined image sampler at binding `0` and a uniform buffer at binding `1`.

# See also

*   In [Chapter 5](fe2cb528-9d22-49db-a05b-372bce2f87ee.xhtml), *Descriptor Sets*, see the recipe:
    *   *Creating a descriptor set layout*
*   *Creating a pipeline layout*, in this chapter.

# Creating a graphics pipeline with vertex and fragment shaders, depth test enabled, and with dynamic viewport and scissor tests

In this recipe, we will see how to create a commonly used graphics pipeline, in which vertex and fragment shaders are active and a depth test is enabled. We will also specify that viewport and scissor tests are set up dynamically.

# How to do it...

1.  Take the handle of a logical device. Use it to initialize a variable of type `VkDevice` named `logical_device`.
2.  Take the SPIR-V assembly of a vertex shader and use it, along with the `logical_device` variable, to create a shader module. Store it in a variable of type `VkShaderModule` named `vertex_shader_module` (refer to the *Creating a shader module* recipe).
3.  Take the SPIR-V assembly of a fragment shader and using it, along with the `logical_device` variable, create a second shader module. Store its handle in a variable of type `VkShaderModule` named `fragment_shader_module` (refer to the *Creating a shader module* recipe).
4.  Create a variable of type `std::vector` named `shader_stage_params` with elements of a custom `ShaderStageParameters` type (refer to the *Specifying pipeline shader stages* recipe).
5.  Add an element to the `shader_stage_params` vector and use the following values to initialize its members:
    *   `VK_SHADER_STAGE_VERTEX_BIT` value for `ShaderStage.`
    *   `vertex_shader_module` variable for `ShaderModule.`
    *   `main` string for `EntryPointName.`
    *   `nullptr` value for `SpecializationInfo.`
6.  Add a second element to the `shader_stage_params` vector and use the following values to initialize its members:
    *   `VK_SHADER_STAGE_FRAGMENT_BIT` value for `ShaderStage.`
    *   `fragment_shader_module` variable for `ShaderModule.`
    *   `main` string for `EntryPointName.`
    *   `nullptr` value for `SpecializationInfo.`
7.  Create a variable of type `std::vector<VkPipelineShaderStageCreateInfo>` named `shader_stage_create_infos` and initialize it using the members of the `shader_stage_params` vector (refer to the *Specifying pipeline shader stages* recipe).
8.  Create a variable of type `VkPipelineVertexInputStateCreateInfo` named `vertex_input_state_create_info`. Initialize it with the desired parameters of vertex input bindings and vertex attributes (refer to the *Specifying pipeline vertex binding description, attribute description, and input state* recipe).

9.  Create a variable of type `VkPipelineInputAssemblyStateCreateInfo` named `input_assembly_state_create_info` and initialize it using the desired primitive topology (triangle list or triangle strip, or line list, and so on) and decide whether the primitive restart should be enabled or disabled (refer to the *Specifying pipeline input assembly state* recipe).
10.  Create a variable of type `VkPipelineViewportStateCreateInfo` named `viewport_state_create_info`. Initialize it using a variable of the `ViewportInfo` type with one-element vectors for both viewport and scissor test vectors. Values stored in these vectors don't matter as viewport and stencil parameters will be defined dynamically during command buffer recording. But as the number of viewports (and scissor test states) are defined statically, both vectors need to have one element (refer to the *Specifying pipeline viewport and scissor test state* recipe).
11.  Create a variable of type `VkPipelineRasterizationStateCreateInfo` named `rasterization_state_create_info` and initialize it with selected values. Remember to provide a false value for the `rasterizerDiscardEnable` member (refer to the *Specifying pipeline rasterization state* recipe).
12.  Create a variable of type `VkPipelineMultisampleStateCreateInfo` named `multisample_state_create_info`. Specify the desired parameters of a multisampling (refer to the *Specifying pipeline multisample state* recipe).
13.  Create a variable of type `VkPipelineDepthStencilStateCreateInfo` named `depth_and_stencil_state_create_info`. Remember to enable depth writes and a depth test and to specify a `VK_COMPARE_OP_LESS_OR_EQUAL` operator for a depth test. Define the rest of the depth and stencil parameters as required (refer to the *Specifying pipeline depth and stencil state* recipe).
14.  Create a variable of type `VkPipelineColorBlendStateCreateInfo` named `blend_state_create_info` and initialize it with the desired set of values (refer to the *Specifying pipeline blend state* recipe).
15.  Create a variable of type `std::vector<VkDynamicState>` named `dynamic_states`. Add two elements to the vector, one with a `VK_DYNAMIC_STATE_VIEWPORT` value, and a second with a `VK_DYNAMIC_STATE_SCISSOR` value.
16.  Create a variable of type `VkPipelineDynamicStateCreateInfo` named `dynamic_state_create_info`. Prepare its contents using the `dynamic_states` vector (refer to the *Specifying pipeline dynamic states* recipe).

17.  Create a variable of type `VkGraphicsPipelineCreateInfo` named `graphics_pipeline_create_info`. Initialize it using the `shader_stage_create_infos`, `vertex_input_state_create_info`, `input_assembly_state_create_info`, `viewport_state_create_info`, `rasterization_state_create_info`, `multisample_state_create_info`, `depth_and_stencil_state_create_info`, `blend_state_create_info` and `dynamic_state_create_info` variables. Provide the created pipeline layout, the selected render pass, and its subpass. Use the handle or index of a parent pipeline. Provide a `nullptr` value for the tessellation state info.
18.  Create a graphics pipeline using the `logical_device` and `graphics_pipeline_create_info` variables. Provide the handle of a pipeline cache, if needed. Store the handle of the created pipeline in the one element vector variable of type `std::vector<VkPipeline>` named `graphics_pipeline`.

# How it works...

One of the most commonly used pipelines is a pipeline with only vertex and fragment shaders. To prepare parameters of vertex and fragment shader stages we can use the following code:

[PRE30]

In the preceding code, we load source codes of vertex and fragment shaders, create shader modules for them, and specify parameters of the shader stages.

Next we need to select whatever parameters we would like for vertex bindings and vertex attributes:

[PRE31]

Viewport and scissor test parameters are important. But as we want to define them dynamically, only the number of viewports matters during the pipeline creation. That's why here we can specify whatever values we want:

[PRE32]

Next we need to prepare parameters for rasterization and multisample states (rasterization must be enabled if we want to use a fragment shader):

[PRE33]

We also want to enable a depth test (and depth writes). Usually we want to simulate how people or cameras observe the world, where objects near the viewer block the view, and obscure objects that are further away. That's why for the depth test, we specify a `VK_COMPARE_OP_LESS_OR_EQUAL` operator which defines that samples with lower or equal depth values pass and those with greater depth values fail the depth test. Other depth-related parameters and parameters for the stencil test can be set as we want, but here we assume the stencil test is disabled (so the values of the stencil test parameters don't matter here):

[PRE34]

Blending parameters can be set as we want:

[PRE35]

One last thing is to prepare a list of dynamic states:

[PRE36]

Now we can create a pipeline:

[PRE37]

# See also

The following recipes in this chapter:

*   *Specifying pipeline shader stages*
*   *Specifying pipeline vertex binding description, attribute description, and input state*
*   *Specifying pipeline input assembly state*
*   *Specifying pipeline viewport and scissor test state*
*   *Specifying pipeline rasterization state*
*   *Specifying pipeline multisample state*
*   *Specifying pipeline depth and stencil state*
*   *Specifying pipeline blend state*
*   *Specifying pipeline dynamic states*
*   *Creating a pipeline layout*
*   *Specifying graphics pipeline creation parameters*
*   *Creating a pipeline cache object*
*   *Creating a graphics pipeline*

# Creating multiple graphics pipelines on multiple threads

The process of creating a graphics pipeline may take a (relatively) long time. Shader compilation takes place during the pipeline creation, the driver checks if compiled shaders can be properly linked together and if a state is properly specified for the shaders to work correctly. That's why, especially when we have lots of pipelines to create, it is good to split this process into multiple threads.

But when we have lots of pipelines to create, we should use a cache to speed up the creation even further. Here we will see how to use a cache for multiple concurrent pipeline creations and how to merge the cache afterwards.

# Getting ready

In this recipe we use a custom template wrapper class of a `VkDestroyer<>` class. It is used to automatically destroy unused resources.

# How to do it...

1.  Store the name of the file from which cache contents should be read, and into which cache contents should be written, in a variable of type `std::string` named `pipeline_cache_filename`.
2.  Create a variable of type `std::vector<unsigned char>` named `cache_data`. If the file named `pipeline_cache_filename` exists, load its contents into the `cache_data` vector.
3.  Take the handle of a logical device and store it in a variable of type `VkDevice` named `logical_device`.
4.  Create a variable of type `std::vector<VkPipelineCache>` named `pipeline_caches`. For each separate thread, create a pipeline cache object and store its handle in the `pipeline_caches` vector (refer to the *Creating a pipeline cache object* recipe).
5.  Create a variable of type `std::vector<std::thread>` named `threads`. Resize it to store the desired number of threads.
6.  Create a variable of type `std::vector<std::vector<VkGraphicsPipelineCreateInfo>>` named `graphics_pipelines_create_infos`. For each thread, add new vector to the `graphics_pipelines_create_infos` variable containing variables of type `VkGraphicsPipelineCreateInfo`, where the number of these variables should be equal to the number of pipelines that should be created on a given thread.
7.  Create a variable of type `std::vector<std::vector<VkPipeline>>` named `graphics_pipelines`. Resize each member vector that corresponds to each thread to hold the same number of pipelines created on a given thread.
8.  Create the desired number of threads where each thread creates the selected number of pipelines using the `logical_device` variable, a cache corresponding to this thread (`pipeline_caches[<thread number>]`), and a corresponding vector with elements of type `VkGraphicsPipelineCreateInfo` (`graphics_pipelines_create_infos[<thread number>]` vector variable).
9.  Wait for all threads to finish.
10.  Create new cache in a variable of type `VkPipelineCache` named `target_cache`.
11.  Merge pipeline caches stored in the `pipeline_caches` vector into the `target_cache` variable (refer to the *Merging multiple pipeline cache objects* recipe).
12.  Retrieve the cache contents of the `target_cache` variable and store it in the `cache_data` vector.
13.  Save the contents of the `cache_data` vector into the file named `pipeline_cache_filename` (replace the file's contents with the new data).

# How it works...

Creating multiple graphics pipelines requires us to provide lots of parameters for many different pipelines. But using separate threads, where each thread creates multiple pipelines, should reduce the time needed to create all the pipelines.

To speed things even more, it is good to use a pipeline cache. First we need to read the previously stored cache contents from the file, if it was created. Next we need to create the cache for each separate thread. Each cache should be initialized with the cache contents loaded from the file (if it was found):

[PRE38]

The next step is to prepare storage space in which handles of pipelines created on each thread will be stored. We also start all the threads that create multiple pipelines using the corresponding cache object:

[PRE39]

Now we need to wait until all the threads are finished. After that we can merge different cache objects (from each thread) into one, from which we retrieve the contents. These new contents we can store in the same file from which we loaded the contents at the beginning (we should replace the contents):

[PRE40]

# See also

The following recipes in this chapter:

*   *Specifying graphics pipeline creation parameters*
*   *Creating a pipeline cache object*
*   *Retrieving data from a pipeline cache*
*   *Merging multiple pipeline cache objects*
*   *Creating a graphics pipeline*
*   *Destroying a pipeline cache*

# Destroying a pipeline

When a pipeline object is no longer needed and we are sure that it is not being used by the hardware in any of the submitted command buffers, we can safely destroy it.

# How to do it...

1.  Take the handle of a logical device. Use it to initialize a variable of type `VkDevice` named `logical_device`.
2.  Take the handle of a pipeline object that should be destroyed. Store it in a variable of type `VkPipeline` named `pipeline`. Make sure it is not being referenced by any commands submitted to any of the available queues.
3.  Call `vkDestroyPipeline( logical_device, pipeline, nullptr )` for which provide the `logical_device` and `pipeline` variables and a `nullptr` value.
4.  For safety reasons, assign a `VK_NULL_HANDLE` value to the `pipeline` variable.

# How it works...

When a pipeline is no longer needed, we can destroy it by calling the `vkDestroyPipeline()` function like this:

[PRE41]

Pipeline objects are used during rendering. So before we can destroy them, we must make sure all the rendering commands that used them are already finished. This is best done by associating a fence object with a submission of a given command buffer. After that we need to wait for the fence before we destroy pipeline objects referenced in that command buffer (refer to the *Waiting for fences* recipe). However, other synchronization methods are also valid.

# See also

*   In [Chapter 3](fc38e0ae-51aa-4f6f-8fb3-551861273018.xhtml), *Command Buffers and Synchronization*, see the following recipes:
    *   *Waiting for fences*
    *   *Waiting for all submitted commands to be finished*
*   The following recipes in this chapter:
    *   *Creating a graphics pipeline*
    *   *Creating a compute pipeline*

# Destroying a pipeline cache

A pipeline cache is not used in any commands recorded in a command buffer. That's why, when we have created all the pipelines we wanted, merged cache data, or retrieved its contents, we can destroy the cache.

# How to do it...

1.  Store the handle of a logical device in a variable of type `VkDevice` named `logical_device`.
2.  Take the handle of a pipeline cache object that should be destroyed. Use the handle to initialize a variable of type `VkPipelineCache` named `pipeline_cache`.
3.  Call `vkDestroyPipelineCache( logical_device, pipeline_cache, nullptr )` and provide the `logical_device` and `pipeline_cache` variables, and a `nullptr` value.
4.  For safety reasons, store the `VK_NULL_HANDLE` value in the `pipeline_cache` variable.

# How it works...

Pipeline cache objects can be used only during the creation of pipelines, for retrieving data from it, and for merging multiple caches into one. None of these operations are recorded in the command buffers, so as soon as any function performing one the mentioned operations has finished, we can destroy the cache like this:

[PRE42]

# See also

The following recipes in this chapter:

*   *Creating a pipeline cache object*
*   *Retrieving data from a pipeline cache*
*   *Merging multiple pipeline cache objects*
*   *Creating a graphics pipeline*
*   *Creating a compute pipeline*

# Destroying a pipeline layout

When we don't need a pipeline layout anymore, and we don't intend to create more pipelines with it, bind descriptor sets or update push constants that used the given layout, and all operations using the pipeline layout are already finished, we can destroy the layout.

# How to do it...

1.  Take the handle of a logical device. Use it to initialize a variable of type `VkDevice` named `logical_device`.
2.  Take the handle of a pipeline layout stored in a variable of type `VkPipelineLayout` named `pipeline_layout`.
3.  Call `vkDestroyPipelineLayout( logical_device, pipeline_layout, nullptr )`. For the call, provide the `logical_device` and `pipeline_layout` variables and a `nullptr` value.
4.  For safety reasons, assign a `VK_NULL_HANDLE` to the `pipeline_layout` variable.

# How it works...

Pipeline layouts are used only in three situations--creating pipelines, binding descriptor sets, and updating push constants. When a given pipeline layout was used only to create a pipeline, it may be destroyed immediately after the pipeline is created. If we are using it to bind descriptor sets or update push constants, we need to wait until the hardware stops processing command buffers, in which these operations were recorded. Then, we can safely destroy the pipeline layout using the following code:

[PRE43]

# See also

*   In [Chapter 3](fc38e0ae-51aa-4f6f-8fb3-551861273018.xhtml), *Command Buffers and Synchronization*, see the following recipes:
    *   *Waiting for fences*
    *   *Waiting for all submitted commands to be finished*
*   In [Chapter 5](fe2cb528-9d22-49db-a05b-372bce2f87ee.xhtml), *Descriptor Sets*, see the following recipe:
    *   *Binding descriptor sets*
*   The following recipes in this chapter:
    *   *Creating a pipeline layout*
    *   *Creating a graphics pipeline*
    *   *Creating a compute pipeline*
    *   *Providing data to shaders through push constants*

# Destroying a shader module

Shader modules are used only for creating pipeline objects. After they are created, we can immediately destroy them, if we don't intend to use them anymore.

# How to do it...

1.  Use the handle of a logical device to initialize a variable of type `VkDevice` named `logical_device`.
2.  Take the shader module's handle stored in a variable of type `VkShaderModule` named `shader_module`.
3.  Call `vkDestroyShaderModule( logical_device, shader_module, nullptr )` providing the `logical_device` variable, the `shader_module` variable, and a `nullptr` value.
4.  Assign a `VK_NULL_HANDLE` value to the `shader_module` variable for safety reasons.

# How it works...

Shader modules are used only during the pipeline creation. They are provided as part of a shader stages state. When pipelines that use given modules are already created, we can destroy the modules (immediately after the pipeline creating functions have finished), as they are not needed for the pipeline objects to be correctly used by the driver.

Created pipelines don't need shader modules anymore to be successfully used.

To destroy a shader module, use the following code:

[PRE44]

# See also

The following recipes in this chapter:

*   *Creating a shader module*
*   *Specifying pipeline shader stages*
*   *Creating a graphics pipeline*
*   *Creating a compute pipeline*