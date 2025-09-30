# Command Recording and Drawing

In this chapter, we will cover the following recipes:

*   Clearing a color image
*   Clearing a depth-stencil image
*   Clearing render pass attachments
*   Binding vertex buffers
*   Binding an index buffer
*   Providing data to shaders through push constants
*   Setting viewport state dynamically
*   Setting scissor state dynamically
*   Setting line width state dynamically
*   Setting depth bias state dynamically
*   Setting blend constants state dynamically
*   Drawing a geometry
*   Drawing an indexed geometry
*   Dispatching compute work
*   Executing a secondary command buffer inside a primary command buffer
*   Recording a command buffer that draws a geometry with a dynamic viewport and scissor states
*   Recording command buffers on multiple threads
*   Preparing a single frame of animation
*   Increasing performance through increasing the number of separately rendered frames

# Introduction

Vulkan was designed as a graphics and compute API. Its main purpose is to allow us to generate dynamic images using a graphics hardware produced by various vendors. We already know how to create and manage resources and use them as a source of data for shaders. We learned about different shader stages and pipeline objects controlling the state of rendering or dispatching computational work. We also know how to record command buffers and order operations into render passes. One last step we must learn about is how to utilize this knowledge to render images.

In this chapter, we will see what additional commands we can record and what commands need to be recorded so we can properly render a geometry or issue computational operations. We will also learn about the drawing commands and organizing them in our source code in such a way so that it maximizes the performance of our application. Finally, we will utilize one of the greatest strengths of the Vulkan API--the ability to record command buffers in multiple threads.

# Clearing a color image

In traditional graphics APIs, we start rendering a frame by clearing a render target or a back buffer. In Vulkan, we should perform the clearing by specifying a `VK_ATTACHMENT_LOAD_OP_CLEAR` value for a `loadOp` member of the render pass's attachment description (refer to the *Specifying attachment descriptions* recipe from [Chapter 6](2de4339d-8912-440a-89a6-fd1f84961448.xhtml), *Render Passes and Framebuffers*). But sometimes, we can't clear an image inside a render pass and we need to do it implicitly.

# How to do it...

1.  Take the handle of a command buffer stored in a variable of type `VkCommandBuffer` named `command_buffer`. Make sure the command buffer is in the recording state and no render pass has started.
2.  Take the handle of an image that should be cleared. Provide it through a variable of type `VkImage` named `image`.
3.  Store the layout, in which the `image` will have during clearing, in a variable of type `VkImageLayout` named `image_layout`.

4.  Prepare a list of all mipmap levels of the `image` and array layers that should be cleared in a variable of type `std::vector<VkImageSubresourceRange>` named `image_subresource_ranges`. For each range of sub-resources of the `image`, add a new element to the `image_subresource_ranges` vector and use the following values to initialize its members:
    *   The image's aspect (color, depth, and/or stencil aspect cannot be provided) for `aspectMask`
    *   The first mipmap level to be cleared in a given range for `baseMipLevel`
    *   The number of continuous mipmap levels that should be cleared in a given range for `levelCount`
    *   The number of a first array layer that should be cleared in a given range for `baseArrayLayer`
    *   The number of continuous array layers to be cleared for `layerCount`
5.  Provide a color to which the image should be cleared using the following members of a variable type `VkClearColorValue` named `clear_color`:
    *   `int32`: When the image has a signed integer format
    *   `uint32`: When the image has an unsigned integer format
    *   `float32`: For the rest of the formats
6.  Call the `vkCmdClearColorImage( command_buffer, image, image_layout, &clear_color, static_cast<uint32_t>(image_subresource_ranges.size()), image_subresource_ranges.data() )` command for which it provides the `command_buffer`, `image`, `image_layout` variables, a pointer to the `clear_color` variable, the number of elements in the `image_subresource_ranges` vector, and a pointer to the first element of the `image_subresource_ranges` vector.

# How it works...

Clearing color images is performed by recording the `vkCmdClearColorImage()` function in a command buffer. The `vkCmdClearColorImage()` command cannot be recorded inside a render pass.

It requires us to provide the image's handle, its layout, and an array of its sub-resources (mipmap levels and/or array layers) that should be cleared. We must also specify the color to which the image should be cleared. These parameters can be used like this:

[PRE0]

Remember that by using this function, we can clear only color images (with a color aspect and one of the color formats).

The `vkCmdClearColorImage()` function can be used only for images created with **transfer dst** usage.

# See also

*   In [Chapter 3](fc38e0ae-51aa-4f6f-8fb3-551861273018.xhtml), *Command Buffers and Synchronization*, see the recipe:
    *   *Beginning a command buffer recording operation*
*   In [Chapter 4](f1332ca0-b5a2-49bd-ac41-e37068e31042.xhtml), *Resources and Memory*, see the recipe:
    *   *Creating an image*

*   In [Chapter 6](2de4339d-8912-440a-89a6-fd1f84961448.xhtml), *Render Passes and Framebuffers*, see the following recipes:
    *   *Specifying attachment descriptions*
    *   *Clearing render pass attachments*
    *   *Clearing a depth-stencil image*

# Clearing a depth-stencil image

Similarly to color images, we sometimes need to manually clear a depth-stencil image outside of a render pass.

# How to do it...

1.  Take the command buffer that is in a recording state and has no render pass currently started in it. Using its handle, initialize a variable of type `VkCommandBuffer` named `command_buffer`.
2.  Take the handle of a depth-stencil image and store it in a variable of type `VkImage` named `image`.
3.  Store the value representing the layout, in which the `image` will have during clearing, in a variable of type `VkImageLayout` named `image_layout`.
4.  Create a variable of type `std::vector<VkImageSubresourceRange>` named `image_subresource_ranges`, which will contain a list of mipmap levels of all the `image`'s and array layers, which should be cleared. For each such range, add a new element to the `image_subresource_ranges` vector and use the following values to initialize its members:
    *   The depth and/or stencil aspect for `aspectMask`
    *   The first mipmap level to be cleared in a given range for `baseMipLevel`
    *   The number of continuous mipmap levels in a given range for `levelCount`
    *   The number of a first array layer that should be cleared for `baseArrayLayer`
    *   The number of continuous array layers to be cleared in a range for `layerCount`
5.  Provide a value which should be used to clear (fill) the image using the following members of a variable of type `VkClearDepthStencilValue` named `clear_value`:
    *   `depth` when a depth aspect should be cleared
    *   `stencil` for a value used to clear the stencil aspect
6.  Call `vkCmdClearDepthStencilImage( command_buffer, image, image_layout, &clear_value, static_cast<uint32_t>(image_subresource_ranges.size()), image_subresource_ranges.data() )` and provide the `command_buffer`, `image`, and `image_layout` variables, a pointer to the `clear_value` variable, the number of elements in the `image_subresource_ranges` vector, and a pointer to the first element of the `image_subresource_ranges` vector.

# How it works...

Clearing the depth-stencil image outside of a render pass is performed like this:

[PRE1]

We can use this function only for images created with a transfer dst usage (clearing is considered as a transfer operation).

# See also

*   In [Chapter 3](fc38e0ae-51aa-4f6f-8fb3-551861273018.xhtml), *Command Buffers and Synchronization*, see the recipe:
    *   *Beginning a command buffer recording operation*
*   In [Chapter 4](f1332ca0-b5a2-49bd-ac41-e37068e31042.xhtml), *Resources and Memory*, see the recipe:
    *   *Creating an image*
*   In [Chapter 6](2de4339d-8912-440a-89a6-fd1f84961448.xhtml), *Render Passes and Framebuffers*, see the following recipes:
    *   *Specifying attachment descriptions*
    *   *Clearing render pass attachments*
*   The *Clearing a color image* recipe, in this chapter

# Clearing render pass attachments

There are situations in which we cannot rely only on implicit attachment clearings performed as initial render pass operations, and we need to clear attachments explicitly in one of the sub-passes. We can do this by calling a `vkCmdClearAttachments()` function.

# How to do it...

1.  Take a command buffer that is in a recording state and store its handle in a variable of type `VkCommandBuffer` named `command_buffer`.
2.  Create a vector variable of type `std::vector<VkClearAttachment>` named `attachments`. For each `framebuffer` attachment that should be cleared inside a current sub-pass of a render pass, add an element to the vector and initialize it with the following values:
    *   The attachment's aspect (color, depth, or stencil) for `aspectMask`
    *   If `aspectMask` is set to `VK_IMAGE_ASPECT_COLOR_BIT`, specify an index of a color attachment in the current sub-pass for `colorAttachment`; otherwise, this parameter is ignored
    *   A desired clear value for a color, depth, or stencil aspect for `clearValue`
3.  Create a variable of type `std::vector<VkClearRect>` named `rects`. For each area that should be cleared in all the specified attachments, add an element to the vector and initialize it with the following values:
    *   The rectangle to be cleared (top-left corner and a width and height) for `rect`
    *   The index of a first layer to be cleared for `baseArrayLayer`
    *   The number of layers to be cleared for `layerCount`
4.  Call `vkCmdClearAttachments( command_buffer, static_cast<uint32_t>(attachments.size()), attachments.data(), static_cast<uint32_t>(rects.size()), rects.data() )`. For the function call, provide the handle of the command buffer, the number of elements in the `attachments` vector, a pointer to its first element, the number of elements in the `rects` vector, and a pointer to its first element.

# How it works...

When we want to explicitly clear an image that is used as a framebuffer's attachment inside a started render pass, we cannot use the usual image clearing functions. We can do this only by selecting which attachments should be cleared. This is done through the `vkCmdClearAttachments()` function like this:

[PRE2]

Using this function, we can clear multiple regions of all the indicated attachments.

We can call the `vkCmdClearAttachments()` function only inside a render pass.

# See also

*   In [Chapter 3](fc38e0ae-51aa-4f6f-8fb3-551861273018.xhtml), *Command Buffers and Synchronization*, see the recipe:
    *   *Beginning a command buffer recording operation*
*   In [Chapter 6](2de4339d-8912-440a-89a6-fd1f84961448.xhtml), *Render Passes and Framebuffers*, see the recipes:
    *   *Specifying attachment descriptions*
    *   *Specifying sub-pass descriptions*
    *   *Beginning a render pass*
*   The following recipes from this chapter:
    *   *Clearing a color image*
    *   *Clearing a depth-stencil image*

# Binding vertex buffers

When we draw a geometry, we need to specify data for vertices. At the very least, vertex positions are required, but we can specify other attributes such as normal, tangent or bitangent vectors, colors, or texture coordinates. This data comes from buffers created with a **vertex buffer** usage. We need to bind these buffers to specified bindings before we can issue drawing commands.

# Getting ready

In this recipe, a custom `VertexBufferParameters` type is introduced. It has the following definition:

[PRE3]

This type is used to specify the buffer's parameters: its handle (in the `Buffer` member) and an offset from the start of the buffer's memory from which data should be taken (in the `MemoryOffset` member).

# How to do it...

1.  Take the handle of a command buffer that is in a recording state and use it to initialize a variable of type `VkCommandBuffer` named `command_buffer`.
2.  Create a variable of type `std::vector<VkBuffer>` named `buffers`. For each buffer that should be bound to a specific binding in the command buffer, add the buffer's handle to the `buffers` vector.
3.  Create a variable of type `std::vector<VkDeviceSize>` named `offsets`. For each buffer in the `buffers` vector, add a new member to the `offsets` vector with an offset value from the start of the corresponding buffer's memory (the buffer at the same index in the `buffers` vector).
4.  Call `vkCmdBindVertexBuffers( command_buffer, first_binding, static_cast<uint32_t>(buffers_parameters.size()), buffers.data(), offsets.data() )`, providing the handle of the command buffer, the number of the first binding to which the first buffer from the list should be bound, the number of elements in the `buffers` (and `offsets`) vector, and a pointer to the first element of the `buffers` vector and to the first element of the `offsets` vector.

# How it works...

During the graphics pipeline creation, we specify the vertex attributes that will be used (provided to shaders) during drawing. This is done through vertex binding and attributes descriptions (refer to the *Specifying a pipeline vertex binding description, attribute description, and input state* recipe from [Chapter 8](5744ea05-b18a-4f84-a1df-250b549dfea5.xhtml), *Graphics and Compute Pipelines*). Through them, we define the number of attributes, their formats, the location through which the shader will be able to access them, and the memory properties, such as offset and stride. We also provide the binding index from which a given attribute should be read. With this binding, we need to associate a selected buffer, in which data for a given attribute (or attributes) is stored. The association is made by binding a buffer to the selected binding index in a given command buffer, like this:

[PRE4]

In the preceding code, the handles of all the buffers that should be bound and their memory offsets are provided through a variable of type `std::vector<VertexBufferParameters>` named `buffers_parameters`.

Remember that we can only bind buffers created with a vertex buffer usage.

# See also

*   In [Chapter 3](fc38e0ae-51aa-4f6f-8fb3-551861273018.xhtml), *Command Buffers and Synchronization*, see the recipe:
    *   *Beginning a command buffer recording operation*
*   In [Chapter 4](f1332ca0-b5a2-49bd-ac41-e37068e31042.xhtml), *Resources and Memory*, see the recipe:
    *   *Creating a buffer*
*   In [Chapter 8](5744ea05-b18a-4f84-a1df-250b549dfea5.xhtml), *Graphics and Compute Pipelines*, see the following recipes:
    *   *Specifying a pipeline vertex binding description*
    *   *Attribute description and input state*
*   The following recipes in this chapter:
    *   *Drawing a geometry*
    *   *Drawing an indexed geometry*

# Binding an index buffer

To draw a geometry, we can provide the list of vertices (and their attributes) in two ways. The first way is a typical list, in which vertices are read one after another. The second method requires us to provide additional indices that indicate which vertices should be read to form polygons. This feature is known as indexed drawing. It allows us to reduce the memory consumption as we don't need to specify the same vertices multiple times. It is especially important when we have multiple attributes associated with each vertex, and when each such vertex is used across many polygons.

Indices are stored in a buffer called an **index buffer**, which must be bound before we can draw an indexed geometry.

# How to do it...

1.  Store the command buffer's handle in a variable of type `VkCommandBuffer` named `command_buffer`. Make sure it is in a recording state.
2.  Take the handle of the buffer in which the indices are stored. Use its handle to initialize a variable of type `VkBuffer` named `buffer`.
3.  Take an offset value (from the start of the buffer's memory) that indicates the beginning of the indice's data. Store the offset in a variable of type `VkDeviceSize` named `memory_offset`.
4.  Provide the type of data used for the indices. Use a `VK_INDEX_TYPE_UINT16` value for 16-bit unsigned integers or a `VK_INDEX_TYPE_UINT32` value for 32-bit unsigned integers. Store the value in a variable of type `VkIndexType` named `index_type`.
5.  Call `vkCmdBindIndexBuffer( command_buffer, buffer, memory_offset, index_type )`, and provide the handles of the command buffer and the buffer, the memory offset value, and the type of data used for the indices (the `index_type` variable as the last argument).

# How it works...

To use a buffer as a source of vertex indices, we need to create it with an *index buffer* usage and fill it with proper data--indices indicating what vertices should be used for drawing. Indices must be tightly packed (one after another) and they should just point to a given index in an array of vertex data, hence the name. This is shown in the following diagram:

![](img/image_09_01-1.png)

Before we can record an indexed drawing command, we need to bind an index buffer, like this:

[PRE5]

For the call, we need to provide a command buffer, to which we record the function and the buffer that should act as an index buffer. Also, the memory offset from the start of the buffer's memory is required. It shows from which parts of the buffer's memory the driver should start reading the indices. The last parameter, the `index_type` variable in the preceding example, specifies the data type of the indices stored in the buffer--if they are specified as unsigned integers with 16 or 32 bits.

# See also

*   In [Chapter 3](fc38e0ae-51aa-4f6f-8fb3-551861273018.xhtml), *Command Buffers and Synchronization*, see the recipe:
    *   *Beginning a command buffer recording operation*
*   In [Chapter 4](f1332ca0-b5a2-49bd-ac41-e37068e31042.xhtml), *Resources and Memory*, see the recipe:
    *   *Creating a buffer*
*   The following recipes in this chapter:
    *   *Binding vertex buffers*
    *   *Drawing an indexed geometry*

# Providing data to shaders through push constants

During drawing or dispatching computational work, specific shader stages are executed--the ones defined during the pipeline creation. So the shaders can perform their job, we need to provide data to them. Most of the time we use descriptor sets, as they allow us to provide kilobytes or even megabytes of data through buffers or images. But using them is quite complicated. And, what's more important, frequent changes of descriptor sets may impact the performance of our application. But sometimes, we need to provide a small amount of data in a fast and easy way. We can do this using push constants.

# How to do it...

1.  Store the handle of a command buffer in a variable of type `VkCommandBuffer` named `command_buffer`. Make sure it is in a recording state.
2.  Take the layout of a pipeline that uses a range of push constants. Store the handle of the layout in a variable of type `VkPipelineLayout` named `pipeline_layout`.
3.  Through a variable of type `VkShaderStageFlags` named `pipeline_stages`, define the shader stages that will access a given range of push constant data.
4.  In a variable of type `uint32_t` named `offset`, specify an offset (in bytes) from which the push constant memory should be updated. The `offset` must be a multiple of 4.
5.  Define the size (in bytes) of the part of the updated memory in a variable of type `uint32_t` named `size`. The `size` must be a multiple of 4.
6.  Using a variable of type `void *` named `data`, provide a pointer to a memory from which the data should be copied to push the constant memory.
7.  Make the following call:

[PRE6]

8.  For the call, provide (in the same order) the variables described in bullets from 1 to 6.

# How it works...

Push constants allow us to quickly provide a small amount of data to shaders (refer to the *Using push constants in shaders* recipe from [Chapter 7](97217f0d-bed7-4ae1-a543-b4d599f299cf.xhtml), *Shaders*). Drivers are required to offer at least 128 bytes of memory for push constant data. This is not much, but it is expected that push constants are much faster than updating data in a descriptor resource. This is the reason we should use them to provide data that changes very frequently, even with each drawing or dispatching of compute shaders.

Data to push constants is copied from the provided memory address. Remember that we can update only data whose size is a multiple of 4\. The offset within a push constant memory (to which we copy the data) must also be a multiple of 4\. As an example, to copy four floating-point values, we can use the following code:

[PRE7]

`ProvideDataToShadersThroughPushConstants()` is a function that implements this recipe in the following way:

[PRE8]

# See also

*   In [Chapter 7](97217f0d-bed7-4ae1-a543-b4d599f299cf.xhtml), *Shaders*, see the recipe:
    *   *Using push constants in shaders*
*   In [Chapter 8](5744ea05-b18a-4f84-a1df-250b549dfea5.xhtml), *Graphics and Compute Pipelines*, see the recipe:
    *   *Creating a pipeline layout*

# Setting viewport states dynamically

The graphics pipeline defines parameters of lots of different states used during rendering. Creating separate pipeline objects every time we need to use slightly different values of some of these parameters would be cumbersome and very impractical. That's why dynamic states are available in Vulkan. We can define a viewport transformation to be one of them. In such a situation, we specify its parameters through a function call recorded in command buffers.

# How to do it...

1.  Take the handle of a command buffer that is in a recording state. Using its handle, initialize a variable of type `VkCommandBuffer` named `command_buffer`.

2.  Specify the number of the first viewport whose parameters should be set. Store the number in a variable of type `uint32_t` named `first_viewport`.
3.  Create a variable of type `std::vector<VkViewport>` named `viewports`. For each viewport that was defined during the pipeline creation, add a new element to the `viewports` vector. Through it, specify the parameters of a corresponding viewport using the following values:
    *   The left side (in pixels) of the upper left corner for `x`
    *   The top side (in pixels) of the upper left corner for `y`
    *   The width of the viewport for `width`
    *   The height of the viewport for `height`
    *   The minimal depth value used during a fragment's depth calculations for `minDepth`
    *   The maximal value of a fragment's calculated depth for `maxDepth`
4.  Call `vkCmdSetViewport( command_buffer, first_viewport, static_cast<uint32_t>(viewports.size()), viewports.data() )` and provide the handle of the `command buffer`, the `first_viewport` variable, the number of elements in the viewports vector, and a pointer to the first element of the `viewports` vector.

# How it works...

The viewport state can be specified to be one of the dynamic pipeline states. We do this during the pipeline creation (refer to the *Specifying pipeline dynamic states* recipe from [Chapter 8](5744ea05-b18a-4f84-a1df-250b549dfea5.xhtml), *Graphics and Compute Pipelines*). Here, dimensions of the viewport are specified with a function call like this:

[PRE9]

Parameters defining the dimensions of each viewport used during rendering (refer to the *Specifying a pipeline viewport and scissor test state* recipe from [Chapter 8](5744ea05-b18a-4f84-a1df-250b549dfea5.xhtml), *Graphics and Compute Pipelines*) are specified through an array, where each element of the array corresponds to a given viewport (offset by the value specified in the `firstViewport` function parameter--`first_viewport` variable in the preceding code).

We just need to remember that the number of viewports used during rendering is always specified statically in a pipeline, no matter if the viewport state is specified as dynamic or not.

# See also

*   In [Chapter 3](fc38e0ae-51aa-4f6f-8fb3-551861273018.xhtml), *Command Buffers and Synchronization*, see the recipe:
    *   *Beginning a command buffer recording operation*
*   In [Chapter 8](5744ea05-b18a-4f84-a1df-250b549dfea5.xhtml), *Graphics and Compute Pipelines*, see the following recipes:
    *   *Specifying a pipeline viewport and scissor test state*
    *   *Specifying pipeline dynamic states*

# Setting scissor states dynamically

The viewport defines a part of an attachment (image) to which the clip's space will be mapped. The scissor test allows us to additionally confine a drawing to the specified rectangle within the specified viewport dimensions. The scissor test is always enabled; we can only set up various values for its parameters. This can be done statically during the pipeline creation, or dynamically. The latter is done with a function call recorded in a command buffer.

# How to do it...

1.  Store the handle of a command buffer that is in a recording state in a variable of type `VkCommandBuffer` named `command_buffer`.
2.  Specify the number of the first scissor rectangle in a variable of type `uint32_t` named `first_scissor`. Remember that the number of scissor rectangles corresponds to the number of viewports.
3.  Create a variable of type `std::vector<VkRect2D>` named `scissors`. For each scissor rectangle we want to specify, add an element to the `scissors` variable. Use the following values to specify its members:
    *   The horizontal offset (in pixels) from the upper left corner of the viewport for the `x` member of the `offset`
    *   The vertical offset (in pixels) from the upper left corner of the viewport for the `y` member of the `offset`
    *   The width (in pixels) of the scissor rectangle for the `width` member of the `extent`
    *   The height (in pixels) of the scissor rectangle for the `height` member of the `extent`

4.  Call `vkCmdSetScissor( command_buffer, first_scissor, static_cast<uint32_t>(scissors.size()), scissors.data() )` and provide the `command_buffer` and `first_scissor` variables, the number of elements in the `scissors` vector, and a pointer to the first element of the `scissors` vector.

# How it works...

The scissor test allows us to restrict rendering to a rectangle area specified anywhere inside the viewport. This test is always enabled and must be specified for all viewports defined during the pipeline creation. In other words, the number of specified scissor rectangles must be the same as the number of viewports. If we are providing parameters for a scissor test dynamically, we don't need to do it in a single function call. But before the drawing command is recorded, scissor rectangles for all the viewports must be defined.

To define a set of rectangles for the scissor test, we need to use the following code:

[PRE10]

The `vkCmdSetScissor()` function allows us to define scissor rectangles for only a subset of viewports. Parameters specified at index `i` in the `scissors` array (vector) correspond to a viewport at index `first_scissor + i`.

# See also

*   In [Chapter 3](fc38e0ae-51aa-4f6f-8fb3-551861273018.xhtml), *Command Buffers and Synchronization*, see the recipe:
    *   *Beginning a command buffer recording operation*
*   In [Chapter 8](5744ea05-b18a-4f84-a1df-250b549dfea5.xhtml), *Graphics and Compute Pipelines*, see the following recipes:
    *   *Specifying a pipeline viewport and scissor test state*
    *   *Specifying pipeline dynamic states*
*   *Setting viewport states dynamically*, in this chapter

# Setting line width states dynamically

One of the parameters defined during the graphics pipeline creation is the width of drawn lines. We can define it statically. But if we intend to draw multiple lines with different widths, we should specify line width as one of the dynamic states. This way, we can use the same pipeline object and specify the width of the drawn lines with a function call.

# How to do it...

1.  Take the handle of a command buffer that is being recorded and use it to initialize a variable of type `VkCommandBuffer` named `command_buffer`.
2.  Create a variable of type `float` named `line_width` through which the width of drawn lines will be provided.
3.  Call `vkCmdSetLineWidth( command_buffer, line_width )` providing the `command_buffer` and `line_width` variables.

# How it works...

Setting the width of lines dynamically for a given graphics pipeline is performed with the `vkCmdSetLineWidth()` function call. We just need to remember that to use various widths, we must enable the `wideLines` feature during the logical device creation. Otherwise, we can only specify a value of `1.0f`. In such a case, we shouldn't create a pipeline with a dynamic line width state. But, if we have enabled the mentioned feature and we want to specify various values for line widths, we can do it like this:

[PRE11]

# See also

*   In [Chapter 3](fc38e0ae-51aa-4f6f-8fb3-551861273018.xhtml), *Command Buffers and Synchronization*, see the following recipe:
    *   *Beginning a command buffer recording operation*
*   In [Chapter 8](5744ea05-b18a-4f84-a1df-250b549dfea5.xhtml), *Graphics and Compute Pipelines*, see the following recipes:
    *   *Specifying a pipeline input assembly state*
    *   *Specifying a pipeline rasterization state*
    *   *Specifying pipeline dynamic states*

# Setting depth bias states dynamically

When rasterization is enabled, each fragment that is generated during this process has its own coordinates (position on screen) and a depth value (distance from the camera). Depth value is used for the depth test, allowing for some opaque objects to cover other objects.

Enabling depth bias allows us to modify the fragment's calculated depth value. We can provide parameters for biasing a fragment's depth during the pipeline creation. But when depth bias is specified as one of the dynamic states, we do it through a function call.

# How to do it...

1.  Take the handle of a command buffer that is being recorded. Use the handle to initialize a variable of type `VkCommandBuffer` named `command_buffer`.
2.  Store the value for the constant offset added to the fragment's depth in a variable of type `float` named `constant_factor`.
3.  Create a variable of type `float` named `clamp`. Use it to provide the maximal (or minimal) depth bias that can be applied to an unmodified depth.
4.  Prepare a variable of type `float` named `slope_factor`, in which store a value applied to the fragment's slope used during depth bias calculations.
5.  Call the `vkCmdSetDepthBias( command_buffer, constant_factor, clamp, slope_factor )` function providing the prepared `command_buffer`, `constant_factor`, `clamp` and `slope_factor` variables, which are mentioned in the previous steps.

# How it works...

Depth bias is used to offset a depth value of a given fragment (or rather, all fragments generated from a given polygon). Commonly, it is used when we want to draw objects that are very near other objects; for example, pictures or posters on walls. Due to the nature of depth calculations, such objects may be incorrectly drawn (partially hidden) when viewed from a distance. This issue is known as depth-fighting or Z-fighting.

Depth bias modifies the calculated depth value--the value used during the depth test and stored in a depth attachment--but does not affect the rendered image in any way (that is, it does not increase the visible distance between the poster and the wall it is attached to). Modifications are performed based on a constant factor and fragment's slope. We also specify the maximal or minimal value of the depth bias (`clamp`) which can be applied. These parameters are provided like this:

[PRE12]

# See also

*   In [Chapter 3](fc38e0ae-51aa-4f6f-8fb3-551861273018.xhtml), *Command Buffers and Synchronization*, see the following recipe:
    *   *Beginning a command buffer recording operation*
*   In [Chapter 8](5744ea05-b18a-4f84-a1df-250b549dfea5.xhtml), *Graphics and Compute Pipelines*, see the following recipes:
    *   *Specifying pipeline rasterization states*
    *   *Specifying pipeline depth and stencil states*
    *   *Specifying pipeline dynamic states*

# Setting blend constants states dynamically

Blending is a process that mixes a color stored in a given attachment with a color of a processed fragment. It is often used to simulate transparent objects.

There are multiple ways in which a fragment's color and a color stored in an attachment can be combined--for the blending, we specify factors (weights) and operations, which generate the final color. It is also possible that an additional, constant color is used by these calculations. During the pipeline creation, we can specify that components of the constant color are provided dynamically. In such a case, we set them with a function recorded in a command buffer.

# How to do it...

1.  Take the handle of a command buffer and use it to initialize a variable of type `VkCommandBuffer` named `command_buffer`.
2.  Create a variable of type `std::array<float, 4>` named `blend_constants`. In the array's four elements, store the red, green, blue, and alpha components of the constant color used during the blending calculations.
3.  Call `vkCmdSetBlendConstants( command_buffer, blend_constants.data() )` and provide the `command_buffer` variable and a pointer to the first element of the `blend_constants` array.

# How it works...

Blending is enabled (statically) during graphics pipeline creation. When we enable it, we must provide multiple parameters that define the behavior of this process (refer to the *Specifying pipeline blend state* recipe from [Chapter 8](5744ea05-b18a-4f84-a1df-250b549dfea5.xhtml), *Graphics and Compute Pipelines*). Among these parameters are blend constants--four components of a constant color used during blending calculations. Normally, they are defined statically during the pipeline creation. But, if we enable blending and intend to use multiple different values for the blend constants, we should specify that we will provide them dynamically (refer to the *Specifying pipeline dynamic states* recipe from [Chapter 8](5744ea05-b18a-4f84-a1df-250b549dfea5.xhtml), *Graphics and Compute Pipelines*). This will allow us to avoid creating multiple similar graphics pipeline objects.

Values for the blend constants are provided with a single function call, like this:

[PRE13]

# See also

*   In [Chapter 3](fc38e0ae-51aa-4f6f-8fb3-551861273018.xhtml), *Command Buffers and Synchronization*, see the following recipe:
    *   *Beginning a command buffer recording operation*
*   In [Chapter 8](5744ea05-b18a-4f84-a1df-250b549dfea5.xhtml), *Graphics and Compute Pipelines*, see the following recipes:
    *   *Specifying pipeline blend states*
    *   *Specifying pipeline dynamic states*

# Drawing a geometry

Drawing is the operation we usually want to perform using graphics APIs such as OpenGL or Vulkan. It sends the geometry (vertices) provided by the application through a vertex buffer down the graphics pipeline, where it is processed step by step by programmable shaders and fixed-function stages.
Drawing requires us to provide the number of vertices we would like to process (display). It also allows us to display multiple instances of the same geometry at once.

# How to do it...

1.  Store the handle of a command buffer in a variable of type `VkCommandBuffer` named `command_buffer`. Make sure the command buffer is currently being recorded and that the parameters of all the states used during rendering are already set in it (bound to it). Also, make sure that the render pass is started in the command buffer.
2.  Use a variable of type `uint32_t` named `vertex_count` to hold the number of vertices we would like to draw.
3.  Create a variable of type `uint32_t` named `instance_count` and initialize it with the number of geometry instances that should be displayed.
4.  Prepare a variable of type `uint32_t` named `first_vertex`. Store the number of the first vertex from which the drawing should be performed.
5.  Create a variable of type `uint32_t` named `first_instance` in which the number of the first instance (instance offset) should be stored.
6.  Call the following function: `vkCmdDraw( command_buffer, vertex_count, instance_count, first_vertex, first_instance )`. For the call, provide all of the preceding variables in the same order.

# How it works...

Drawing is performed with a call of the `vkCmdDraw()` function:

[PRE14]

It allows us to draw any number of vertices, where vertices (and their attributes) are stored one after another in a vertex buffer (no index buffer is used). During the call we need to provide an offset--the number of the first vertex from which drawing should be started. This can be used when we have multiple models stored in one vertex buffer (for example, compounds of a model) and we want to draw only one of them.

The preceding function allows us to draw a single mesh (model), and also multiple instances of the same mesh. This is particularly useful when we have specified that some of the attributes change per instance, not per vertex (refer to the *Specifying pipeline vertex binding description, attribute description, and input state* recipe from [Chapter 8](5744ea05-b18a-4f84-a1df-250b549dfea5.xhtml), *Graphics and Compute Pipelines*). This way, each drawn instance of the same model may be a little bit different.

![](img/image_09_002.png)

Almost everything we do in Vulkan is used during drawing. So before we record a drawing command in a command buffer, we must be sure all the required data and parameters are properly set. Remember that each time we record a command buffer, it doesn't have any state. So before we can draw anything, we must set up the state accordingly.

There is no such thing as default state in Vulkan.

An example can be descriptor sets or dynamic pipeline states. Each time we start recording a command buffer, before we can draw anything, all the required descriptor sets (those used by shaders) must be bound to the command buffer. Similarly, every pipeline state that is specified as dynamic must have its parameters provided through corresponding functions. Another thing to remember is the render pass, which must be started in a command buffer for the drawing to be properly executed.

Drawing can be performed only inside the render pass.

# See also

*   In [Chapter 3](fc38e0ae-51aa-4f6f-8fb3-551861273018.xhtml), *Command Buffers and Synchronization*, see the recipe:
    *   *Beginning a command buffer recording operation*
*   In [Chapter 4](f1332ca0-b5a2-49bd-ac41-e37068e31042.xhtml), *Resources and Memory*, see the recipe:
    *   *Creating a buffer*
*   In [Chapter 5](fe2cb528-9d22-49db-a05b-372bce2f87ee.xhtml), *Descriptor Sets*, see the recipe:
    *   *Binding descriptor sets*
*   In [Chapter 6](2de4339d-8912-440a-89a6-fd1f84961448.xhtml), *Render Passes and Framebuffers*, see the following recipes:
    *   *Creating a render pass*
    *   *Creating a framebuffer*
    *   *Beginning a render pass*
*   In [Chapter 8](5744ea05-b18a-4f84-a1df-250b549dfea5.xhtml), *Graphics and Compute Pipelines*, see the following recipes:
    *   *Creating a graphics pipeline*
    *   *Binding a pipeline object*
*   The following recipes in this chapter:
    *   *Binding vertex buffers*
    *   *Setting viewport states dynamically*
    *   *Setting scissor states dynamically*

# Drawing an indexed geometry

Quite often it is more convenient to reuse vertices stored in a vertex buffer. Like the corners of a cube which belong to multiple sides, vertices in arbitrary geometry may belong to many parts of the whole model.
Drawing the object one vertex after another would require us to store the same vertex (along with all its attributes) multiple times. A better solution is to indicate which vertices should be used for drawing, no matter how they are ordered in the vertex buffer. For this purpose, indexed drawing was introduced in the Vulkan API. To draw geometry using indices stored in an index buffer, we need to call the `vkCmdDrawIndexed()` function.

# How to do it...

1.  Create a variable of type `VkCommandBuffer` named `command_buffer`, in which store the handle of a command buffer. Make sure the command buffer is in the recording state.
2.  Initialize a variable of type `uint32_t` named `index_count` with the number of indices (and vertices) that should be drawn.
3.  Use the number of instances (of the same geometry) to be drawn to initialize a variable of type `uint32_t` named `instance_count`.
4.  Store the offset (in the number of indices) from the beginning of an index buffer in a variable of type `uint32_t` named `first_index`. From this index, drawing will be started.
5.  Prepare a variable of type `uint32_t` named `vertex_offset`, in which the vertex offset (the value added to each index) should be stored.
6.  Create a variable of type `uint32_t` named `first_instance` that should hold the number of the first geometry instance to be drawn.
7.  Make the following call: `vkCmdDrawIndexed( command_buffer, index_count, instance_count, first_index, vertex_offset, first_instance )`. For the call, provide all of the preceding variables, in the same order.

# How it works...

Indexed drawing is the way to reduce the memory consumption. It allows us to remove duplicate vertices from vertex buffers, so we can allocate smaller vertex buffers. An additional index buffer is required, but usually vertex data requires much more memory space. This is especially the case in situations when each vertex has more attributes than just one position, such as normal, tangent, and bitangent vectors and two texture coordinates, which are used very often.

Indexed drawing also allows graphics hardware to reuse data from the already processed vertices through a form of vertex caching. With normal (non-indexed) drawing, hardware needs to process each vertex. When indices are used, hardware has additional information about processed vertices and knows if a given vertex was recently processed or not. If the same vertex was recently used (the last several dozens of processed vertices), in many situations the hardware may reuse the results of this vertex's previous processing.

To draw a geometry using vertex indices, we need to bind an index buffer before we record an indexed drawing command (refer to the *Binding an index buffer* recipe). We must also start a render pass, as indexed drawing (similarly to normal drawing) can be recorded only inside render passes. We also need to bind a graphics pipeline and all other required states (depending on the resources used by the graphics pipeline), and we are then good to call the following function:

[PRE15]

Indexed drawing, similarly to normal drawing, can only be performed inside a render pass.

# See also

*   In [Chapter 3](fc38e0ae-51aa-4f6f-8fb3-551861273018.xhtml), *Command Buffers and Synchronization*, see the recipe:
    *   *Beginning a command buffer recording operation*
*   In [Chapter 4](f1332ca0-b5a2-49bd-ac41-e37068e31042.xhtml), *Resources and Memory*, see the recipe:
    *   *Creating a buffer*
*   In [Chapter 5](fe2cb528-9d22-49db-a05b-372bce2f87ee.xhtml), *Descriptor Sets*, see the recipe:
    *   *Binding descriptor sets*
*   In [Chapter 6](2de4339d-8912-440a-89a6-fd1f84961448.xhtml), *Render Passes and Framebuffers*, see the following recipes:
    *   *Creating a render pass*
    *   *Creating a framebuffer*
    *   *Beginning a render pass*
*   In [Chapter 8](5744ea05-b18a-4f84-a1df-250b549dfea5.xhtml), *Graphics and Compute Pipelines*, see the following recipes:
    *   *Creating a graphics pipeline*
    *   *Binding a pipeline object*
*   The following recipes in this chapter:
    *   *Binding vertex buffers*
    *   *Binding an index buffer*
    *   *Setting viewport states dynamically*
    *   *Setting scissor states dynamically*

# Dispatching compute work

Apart from drawing, Vulkan can be used to perform general computations. For this purpose, we need to write compute shaders and execute them--this is called dispatching.
When we want to issue computational work to be performed, we need to specify how many separate compute shader instances should be executed and how they are divided into workgroups.

# How to do it...

1.  Take the handle of a command buffer and store it in a variable of type `VkCommandBuffer` named `command_buffer`. Make sure the command buffer is in the recording state and no render pass is currently started.
2.  Store the number of local workgroups along the *x* dimension in a variable of type `uint32_t` named `x_size`.
3.  The number of local workgroups in the *y* dimensions should be stored in a variable of type `uint32_t` named `y_size`.

4.  Use the number of local workgroups along the *z* dimension to initialize a variable of type `uint32_t` named `z_size`.
5.  Record the `vkCmdDispatch( command_buffer, x_size, y_size, z_size )` function using the preceding variables as its arguments.

# How it works...

When we dispatch compute work, we use compute shaders from the bound compute pipeline to perform the task they are programmed to do. Compute shaders use resources provided through descriptor sets. Results of their computations can also be stored only in resources provided through descriptor sets.

Compute shaders don't have a specific goal or use case scenario which they must fulfil. They can be used to perform any computations that operate on data read from descriptor resources. We can use them to perform image post-processing, such as color correction or blur. We can perform physical calculations and store transformation matrices in buffers or calculate new positions of a morphing geometry. The possibilities are limited only by the desired performance and hardware capabilities.

Compute shaders are dispatched in groups. The number of local invocations in `x`, `y`, and `z` dimensions are specified inside the shader source code (refer to the *Writing compute shaders* recipe from [Chapter 7](97217f0d-bed7-4ae1-a543-b4d599f299cf.xhtml), *Shaders*). The collection of these invocations is called a workgroup. During dispatching the compute shaders, we specify how many such workgroups should be executed in each *x*, *y*, and *z* dimension. This is done through the parameters of the `vkCmdDispatch()` function:

[PRE16]

We just need to remember that the number of workgroups in a given dimension cannot be larger than the value in the corresponding index of the `maxComputeWorkGroupCount[3]` physical device's limit. Currently, the hardware must allow to dispatch at least 65,535 workgroups in a given dimension.

Dispatching compute workgroups cannot be done inside render passes. In Vulkan, render passes can be used only for drawing. If we want to bind compute pipelines and perform some computations inside compute shaders, we must end a render pass.

Compute shaders cannot be dispatched inside render passes.

# See also

*   In [Chapter 3](fc38e0ae-51aa-4f6f-8fb3-551861273018.xhtml), *Command Buffers and Synchronization*, see the recipe:
    *   *Beginning a command buffer recording operation*
*   In [Chapter 5](fe2cb528-9d22-49db-a05b-372bce2f87ee.xhtml), *Descriptor Sets*, see the recipe:
    *   *Binding descriptor sets*
*   In [Chapter 6](2de4339d-8912-440a-89a6-fd1f84961448.xhtml), *Render Passes and Framebuffers*, see the recipe:
    *   *Ending a render pass*
*   In [Chapter 7](97217f0d-bed7-4ae1-a543-b4d599f299cf.xhtml), *Shaders*, see the following recipes:
    *   *Writing compute shaders*
    *   *Creating a compute pipeline*
    *   *Binding a pipeline object*

# Executing a secondary command buffer inside a primary command buffer

In Vulkan we can record two types of command buffers--primary and secondary. Primary command buffers can be submitted to queues directly. Secondary command buffers can be executed only from within primary command buffers.

# How to do it...

1.  Take a command buffer's handle. Store it in a variable of type `VkCommandBuffer` named `command_buffer`. Make sure the command buffer is in the recording state.
2.  Prepare a variable of type `std::vector<VkCommandBuffer>` named `secondary_command_buffers` containing secondary command buffers that should be executed from within the `command_buffer`.

3.  Record the following command: `vkCmdExecuteCommands( command_buffer, static_cast<uint32_t>(secondary_command_buffers.size()), secondary_command_buffers.data() )`. Provide the handle of the primary command buffer, the number of elements in the `secondary_command_buffers` vector, and a pointer to its first element.

# How it works...

Secondary command buffers are recorded in a similar way to primary command buffers. In most cases, primary command buffers should be enough to perform rendering or computing work. But there may be situations in which we need to divide work into two command buffer types. When we have recorded secondary command buffers and we want the graphics hardware to process them, we can execute them from within a primary command buffer like this:

[PRE17]

# See also

*   In [Chapter 3](fc38e0ae-51aa-4f6f-8fb3-551861273018.xhtml), *Command Buffers and Synchronization*, see the recipe:
    *   *Beginning a command buffer recording operation*

# Recording a command buffer that draws a geometry with dynamic viewport and scissor states

Now we have all the knowledge required to draw images using the Vulkan API. In this sample recipe, we will aggregate some of the previous recipes and see how to use them to record a command buffer that displays a geometry.

# Getting ready

To draw a geometry, we will use a custom structure type that has the following definition:

[PRE18]

The `Data` member contains values for all the attributes of a given vertex, one vertex after another. For example, there are three components of position attribute, three components of a normal vector and two texture coordinates of a first vertex. After that, there is data for the position, normal, and **TexCoords** of a second vertex, and so on.

The `VertexOffset` member is used to store vertex offsets of separate parts of a geometry. The `VertexCount` vector contains a number of vertices in each such part.

Before we can draw a model whose data is stored in a variable of the preceding type, we need to copy the contents of a `Data` member to a buffer that will be bound to a command buffer as a vertex buffer.

# How to do it...

1.  Take the handle of a primary command buffer and store it in a variable of type `VkCommandBuffer` named `command_buffer`.
2.  Start recording the `command_buffer` (refer to the *Beginning a command buffer recording operation* recipe from [Chapter 3](fc38e0ae-51aa-4f6f-8fb3-551861273018.xhtml), *Command Buffers and Synchronization*).
3.  Take the handle of an acquired swapchain image and use it to initialize a variable of type `VkImage` named `swapchain_image` (refer to the *Getting handles of swapchain images* and *Acquiring a swapchain image* recipes from [Chapter 2](45eb1180-672a-4745-bd85-f13c7bb658b7.xhtml), *Image Presentation*).
4.  Store the index of a queue family that is used for swapchain image presentation in a variable of type `uint32_t` named `present_queue_family_index`.
5.  Store the index of a queue family used for performing graphics operations in a variable of type `uint32_t` named `graphics_queue_family_index`.

6.  If values stored in the `present_queue_family_index` and `graphics_queue_family_index` variables are different, set up an image memory barrier in the `command_buffer` (refer to the *Setting an image memory barrier* recipe from [Chapter 4](f1332ca0-b5a2-49bd-ac41-e37068e31042.xhtml), *Resources and Memory*). Use a `VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT` value for the `generating_stages` parameter and a `VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT` value for the `consuming_stages` parameters. For the barrier, provide a single variable of type `ImageTransition` and use the following values to initialize its members:
    *   The `swapchain_image` variable for `Image`
    *   The `VK_ACCESS_MEMORY_READ_BIT` value for `CurrentAccess`
    *   The `VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT` value for `NewAccess`
    *   The `VK_IMAGE_LAYOUT_PRESENT_SRC_KHR` value for `CurrentLayout`
    *   The `VK_IMAGE_LAYOUT_PRESENT_SRC_KHR` value for `NewLayout`
    *   The `present_queue_family_index` variable for `CurrentQueueFamily`
    *   The `graphics_queue_family_index` variable for `NewQueueFamily`
    *   The `VK_IMAGE_ASPECT_COLOR_BIT` value for `Aspect`
7.  Take the handle of a `render pass` and store it in a variable of type `VkRenderPass` named `render_pass`.
8.  Store the handle of a framebuffer compatible with the `render_pass` in a variable of type `VkFramebuffer` named `framebuffer`.
9.  Store the size of the `framebuffer` in a variable of type `VkExtent2D` named `framebuffer_size`.
10.  Create a variable of type `std::vector<VkClearValue>` named `clear_values`. For each attachment used in the `render_pass` (and the `framebuffer`), add an element to the `clear_values` variable with values, to which corresponding attachments should be cleared.
11.  Record a `render pass` beginning operation in the `command_buffer`. Use the `render_pass`, `framebuffer`, `framebuffer_size`, and `clear_values` variables and a `VK_SUBPASS_CONTENTS_INLINE` value (refer to *Beginning a render pass* recipe from [Chapter 6](2de4339d-8912-440a-89a6-fd1f84961448.xhtml), *Render Passes and Framebuffers*).

12.  Take the handle of a graphics pipeline and use it to initialize a variable of type `VkPipeline` named `graphics_pipeline`. Make sure the pipeline was created with dynamic viewport and scissor states.

13.  Bind the pipeline to the `command_buffer`. Provide a `VK_PIPELINE_BIND_POINT_GRAPHICS` value and the `graphics_pipeline` variable (refer to the *Binding a pipeline object* recipe from [Chapter 8](5744ea05-b18a-4f84-a1df-250b549dfea5.xhtml), *Graphics and Compute Pipelines*).
14.  Create a variable of type `VkViewport` named `viewport`. Use the following values to initialize its members:
    *   The `0.0f` value for `x`
    *   The `0.0f` value for `y`
    *   The `width` member of the `framebuffer_size` variable for `width`
    *   The `height` member of the `framebuffer_size` variable for `height`
    *   The `0.0f` value for `minDepth`
    *   The `1.0f` value for `maxDepth`
15.  Set the viewport state dynamically in the `command_buffer`. Use a `0` value for the `first_viewport` parameter and a vector of type `std::vector<VkViewport>` with a single element containing the `viewport` variable for the `viewports` parameter (refer to the *Setting viewport state dynamically* recipe).
16.  Create a variable of type `VkRect2D` named `scissor`. Use the following values to initialize its members:
    *   The `0` value for the `x` member of the `offset`
    *   The `0` value for the `y` member of the `offset`
    *   The `framebuffer_size.width` member variable for the `width` member of the `extent`
    *   The `framebuffer_size.height` member variable for the `height` member of the `extent`
17.  Set the scissor state dynamically in the `command_buffer`. Use a `0` value for the `first_scissor` parameter and a vector of type `std::vector<VkRect2D>` with a single element containing the `scissor` variable as the `scissors` parameter (refer to the *Setting scissor states dynamically* recipe in this chapter).

18.  Create a variable of type `std::vector<VertexBufferParameters>` named `vertex_buffers_parameters`. For each buffer that should be bound to the `command_buffer` as a vertex buffer, add an element to the `vertex_buffers_parameters` vector. Use the following values to initialize the members of the new element:
    *   The handle of a buffer that should be used as the vertex buffer for `Buffer`
    *   The offset in bytes from the beginning of the buffer's memory (the memory part that should be bound for the vertex buffer) for `memoryoffset`
19.  Store the value of the first binding, to which the first vertex buffer should be bound, in a variable of type `uint32_t` named `first_vertex_buffer_binding`.
20.  Bind vertex buffers to the `command_buffer` using the `first_vertex_buffer_binding` and `vertex_buffers_parameters` variables (refer to the *Binding vertex buffers* recipe).
21.  Perform the following operations if any descriptor resources should be used during drawing:
    1.  Take the handle of a pipeline's layout and store it in a variable of type `VkPipelineLayout` named `pipeline_layout` (refer to *Creating a pipeline layout* recipe from [Chapter 8](5744ea05-b18a-4f84-a1df-250b549dfea5.xhtml), *Graphics and Compute Pipelines*).
    2.  Add each descriptor set to be used during drawing to a vector variable of type `std::vector<VkDescriptorSet>` named `descriptor_sets`.
    3.  Store an index, to which the first descriptor set should be bound, in a variable of type `uint32_t` named `index_for_first_descriptor_set`.
    4.  Bind descriptor sets to the `command_buffer` using a `VK_PIPELINE_BIND_POINT_GRAPHICS` value and the `pipeline_layout`, `index_for_first_descriptor_set` and `descriptor_sets` variables.
22.  Draw a geometry in the `command_buffer` specifying the desired values for the `vertex_count`, `instance_count`, `first_vertex`, and `first_instance` parameters (refer to the *Drawing a geometry* recipe).
23.  End a render pass in the `command_buffer` (refer to the *Ending a render pass* recipe from [Chapter 6](2de4339d-8912-440a-89a6-fd1f84961448.xhtml), *Render Passes and Framebuffers*).

24.  If values stored in the `present_queue_family_index` and `graphics_queue_family_index` variables are different, set up another image memory barrier in the `command_buffer` (refer to the *Setting an image memory barrier* recipe from [Chapter 4](f1332ca0-b5a2-49bd-ac41-e37068e31042.xhtml), *Resources and Memory*). Use the `VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT` value for the `generating_stages` parameter and the `VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT` value for the `consuming_stages` parameter. For the barrier, provide a single variable of type `ImageTransition` initialized with the following values:
    *   The `swapchain_image` variable for `Image`
    *   The `VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT` value for `CurrentAccess`
    *   The `VK_ACCESS_MEMORY_READ_BIT` value for `NewAccess`
    *   The `VK_IMAGE_LAYOUT_PRESENT_SRC_KHR` value for `CurrentLayout`
    *   The `VK_IMAGE_LAYOUT_PRESENT_SRC_KHR` value for `NewLayout`
    *   The `graphics_queue_family_index` variable for `CurrentQueueFamily` and the `present_queue_family_index` variable for `NewQueueFamily`
    *   The `VK_IMAGE_ASPECT_COLOR_BIT` value for `Aspect`
25.  Stop recording the `command_buffer` (refer to the *Ending a command buffer recording operation* recipe from [Chapter 3](fc38e0ae-51aa-4f6f-8fb3-551861273018.xhtml), *Command Buffers and Synchronization*).

# How it works...

Assume we want to draw a single object. We want the object to appear directly on screen so, before we begin, we must acquire a swapchain image (refer to the *Acquiring a swapchain image* recipe from [Chapter 2](45eb1180-672a-4745-bd85-f13c7bb658b7.xhtml), *Image Presentation*). Next, we start recording the command buffer (refer to the *Beginning a command buffer recording operation* recipe from [Chapter 3](fc38e0ae-51aa-4f6f-8fb3-551861273018.xhtml), *Command Buffers and Synchronization*):

[PRE19]

The first thing we need to record is to change the swapchain image's layout to a `VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL` layout. This operation should be performed implicitly using appropriate render pass parameters (initial and sub-pass layouts). But if queues used for the presentation and graphics operations come from two different families, we must perform ownership transfer. This cannot be done implicitly--for this we need to set up an image memory barrier (refer to the *Setting an image memory barrier* recipe from [Chapter 4](f1332ca0-b5a2-49bd-ac41-e37068e31042.xhtml), *Resources and Memory*):

[PRE20]

The next thing to do is to start a render pass (refer to the *Beginning a render pass* recipe from [Chapter 6](2de4339d-8912-440a-89a6-fd1f84961448.xhtml), *Render Passes and Framebuffers*). We also need to bind a pipeline object (refer to the *Binding a pipeline object* recipe from [Chapter 8](5744ea05-b18a-4f84-a1df-250b549dfea5.xhtml), *Graphics and Compute Pipelines*). We must do this before we can set up any pipeline related state:

[PRE21]

When a pipeline is bound, we must set up any state that was marked as dynamic during the pipeline creation. Here, we set up viewport and scissor test states respectively (refer to the *Setting viewport states dynamically* and *Setting scissor states dynamically* recipes). We also bind a buffer that should be a source of vertex data (refer to the *Binding vertex buffers* recipe). This buffer must contain data copied from a variable of type `Mesh`:

[PRE22]

One last thing to do in this example is to bind the descriptor sets, which can be accessed inside shaders (refer to the *Binding descriptor sets* recipe from [Chapter 5](fe2cb528-9d22-49db-a05b-372bce2f87ee.xhtml), *Descriptor Sets*):

[PRE23]

Now we are ready to draw a geometry. Of course, in more advanced scenarios, we would need to set up parameters of other states and bind other resources. For example, we may need to use an index buffer and provide values for push constants. But, the preceding setup is also enough for many cases:

[PRE24]

To draw a geometry, we must provide the number of geometry instances we want to draw and an index of a first instance. Vertex offsets and the number of vertices to draw are taken from the members of variables of type `Mesh`.

Before we can stop recording a command buffer, we need to end a render pass (refer to the *Ending a render pass* recipe from [Chapter 6](2de4339d-8912-440a-89a6-fd1f84961448.xhtml), *Render Passes and Framebuffers*). After that, another transition on a swapchain image is required. When we are done rendering a single frame of animation, we want to present (display) a swapchain image. For this, we need to change its layout to a `VK_IMAGE_LAYOUT_PRESENT_SRC_KHR` layout, because this layout is required for the presentation engine to correctly display an image. This transition should also be performed implicitly through render pass parameters (the final layout). But again, if the queues used for graphics operations and presentations are different, a queue ownership transfer is necessary. This is done with another image memory barrier. After that, we stop recording a command buffer (refer to the *Ending a command buffer recording operation* recipe from [Chapter 3](fc38e0ae-51aa-4f6f-8fb3-551861273018.xhtml), *Command Buffers and Synchronization*):

[PRE25]

This concludes the command buffer recording operation. We can use this command buffer and submit it to a (graphics) queue. It can be submitted only once, because it was recorded with a `VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT` flag. But, of course, we can record a command buffer without this flag and submit it multiple times.

After submitting the command buffer, we can present a swapchain image, so it is displayed on screen. But, we must remember that submission and presentation operations should be synchronized (refer to the *Preparing a single frame of animation* recipe).

# See also

*   In [Chapter 2](45eb1180-672a-4745-bd85-f13c7bb658b7.xhtml), *Image Presentation*, see the following recipes:
    *   *Acquiring a swapchain image*
    *   *Presenting an image*
*   In [Chapter 3](fc38e0ae-51aa-4f6f-8fb3-551861273018.xhtml), *Command Buffers and Synchronization*, see the following recipes:
    *   *Beginning a command buffer recording operation*
    *   *Ending a command buffer recording operation*
*   In [Chapter 4](f1332ca0-b5a2-49bd-ac41-e37068e31042.xhtml), *Resources and Memory*, see the recipe:
    *   *Setting an image memory barrier*
*   In [Chapter 5](fe2cb528-9d22-49db-a05b-372bce2f87ee.xhtml), *Descriptor Sets*, see the recipe:
    *   *Binding descriptor sets*
*   In [Chapter 6](2de4339d-8912-440a-89a6-fd1f84961448.xhtml), *Render Passes and Framebuffers*, see the following recipes:
    *   *Beginning a render pass*
    *   *Ending a render pass*
*   In [Chapter 8](5744ea05-b18a-4f84-a1df-250b549dfea5.xhtml), *Graphics and Compute Pipelines*, see the recipe:
    *   *Binding a pipeline object*
*   The following recipes in this chapter:
    *   *Binding vertex buffers*
    *   *Setting viewport states dynamically*
    *   *Setting scissor states dynamically*
    *   *Drawing a geometry*
    *   *Preparing a single frame of animation*

# Recording command buffers on multiple threads

High level graphics APIs such as OpenGL are much easier to use, but they are also limited in many aspects. One such aspect is the lack of ability to render scenes on multiple threads. Vulkan fills this gap. It allows us to record command buffers on multiple threads, utilizing as much processing power of not only the graphics hardware, but also of the main processor.

# Getting ready

For the purpose of this recipe, a new type is introduced. It has the following definition:

[PRE26]

The preceding structure is used to store parameters specific for each thread used to record command buffers. The handle of a command buffer that will be recorded on a given thread is stored in the `CommandBuffer` member. The `RecordingFunction` member is used to define a function, inside which we will record the command buffer on a separate thread.

# How to do it...

1.  Create a variable of type `std::vector<CommandBufferRecordingThreadParameters>` named `threads_parameters`. For each thread used to record a command buffer, add a new element to the preceding vector. Initialize the element with the following values:
    *   The handle of a command buffer to be recorded on a separate thread for `CommandBuffer`
    *   The function (accepting a command buffer handle) used to record a given command buffer for `RecordingFunction`
2.  Create a variable of type `std::vector<std::thread>` named `threads`. Resize it to be able to hold the same number of elements as the `threads_parameters` vector.

3.  For each element in the `threads_parameters` vector, start a new thread that will use the `RecordingFunction` and provide the `CommandBuffer` as the function's argument. Store the handle of a created thread at the corresponding position in the `threads` vector.

4.  Wait until all created threads finish their execution by joining with all elements in the `threads` vector.
5.  Gather all recorded command buffers in a variable of type `std::vector<VkCommandBuffer>` named `command_buffers`.

# How it works...

When we want to use Vulkan in a multithreaded application, we must keep in mind several rules. First, we shouldn't modify the same object on multiple threads. For example, we cannot allocate command buffers from a single pool or we cannot update a descriptor set from multiple threads.

We can access resources from multiple threads only if the access is read only or if we reference separate resources. But, as it may be hard to track which resources were created on which thread, in general, resource creation and modification should be performed only on a single *main* thread (which we can also call *the rendering thread*).

The most common scenario of utilizing multithreading in Vulkan is to concurrently record command buffers. This operation takes most of the processor time. It is also the most important operation performance-wise, so dividing it into multiple threads is very reasonable.

When we want to record multiple command buffers in parallel, we need to use not only a separate command buffer for each thread, but also a separate command pool.

We need to use a separate command pool for each thread, on which command buffers will be recorded. In other words--a command buffer recorded on each thread must be allocated from a separate command pool.

Command buffer recording doesn't affect other resources (apart from the pool). We only prepare commands that will be submitted to a queue, so we can record any operations that use any resources. For example, we can record operations that access the same images or the same descriptor sets. The same pipelines can be bound to different command buffers at the same time during recording. We can also record operations that draw into the same attachments. We only record (prepare) operations.

Recording command buffers on multiple threads may be performed like this:

[PRE27]

Here, each thread takes a separate `RecordingFunction` member, in which a corresponding command buffer is recorded. When all threads finish recording their command buffers, we need to gather the command buffers and submit them to a queue, when they are executed.

In real-life applications, we will probably want to avoid creating and destroying threads in this way. Instead, we should take an existing job/task system and use it to also record the necessary command buffers. But the presented example is easy to use and understand. And, it is also good at illustrating the steps that need to be performed to use Vulkan in multithreaded applications.

Submission can also be performed only from a single thread (queues, similarly to other resources, cannot be accessed concurrently), so we need to wait until all threads finish their jobs:

[PRE28]

Submitting command buffers to a queue can be performed only from a single thread at a time.

The preceding situation is presented in the following diagram:

![](img/image_09_003.png)

A similar situation occurs with a swapchain object. We can acquire and present swapchain images only from a single thread at a given moment. We cannot do this concurrently.

A swapchain object cannot be accessed (modified) concurrently on multiple threads. Acquiring an image and presenting it should be done on a single thread.

But, it is a valid operation to acquire a swapchain image on a single thread and then concurrently record multiple command buffers that render into this swapchain image. We just need to make sure that the first submitted command buffer performs a layout transition away from the `VK_IMAGE_LAYOUT_PRESENT_SRC_KHR` (or the `VK_IMAGE_LAYOUT_UNDEFINED`) layout. Transition back to the `VK_IMAGE_LAYOUT_PRESENT_SRC_KHR` layout must be performed inside the command buffer that was submitted to the queue at the end. The order in which these command buffers were recorded doesn't matter; only the submission order is crucial.

Of course, when we want to record operations that modify resources (for example, store values in buffers), we must also record proper synchronization operations (such as pipeline barriers). This is necessary for the proper execution, but it doesn't matter from the recording perspective.

# See also

*   In [Chapter 2](45eb1180-672a-4745-bd85-f13c7bb658b7.xhtml), *Image Presentation*, see the following recipes:
    *   *Acquiring a swapchain image*
    *   *Presenting an image*
*   In [Chapter 3](fc38e0ae-51aa-4f6f-8fb3-551861273018.xhtml), *Command Buffers and Synchronization*, see the following recipes:
    *   *Submitting command buffers to a queue*

# Preparing a single frame of animation

Usually, when we create 3D applications that render images, we would like images to be displayed on screen. For this purpose, a swapchain object is created in Vulkan. We know how to acquire images from a swapchain. We have also learned how to present them. Here, we will see how to connect image acquiring and presentation, how to record a command buffer in between, and how we should synchronize all of these operations to render a single frame of animation.

# How to do it...

1.  Take the handle of a logical device and store it in a variable of type `VkDevice` named `logical_device`.
2.  Use a handle of a created swapchain to initialize a variable of type `VkSwapchainKHR` named `swapchain`.
3.  Prepare a semaphore handle in a variable of type `VkSemaphore` named `image_acquired_semaphore`. Make sure the semaphore is unsignaled or isn't being used in any previous submissions that haven't completed yet.

4.  Create a variable of type `uint32_t` named `image_index`.
5.  Acquire an image from the `swapchain` using the `logical_device`, `swapchain`, and `image_acquired_semaphore` variables and store its index in the `image_index` variable (refer to the *Acquiring a swapchain image* recipe from [Chapter 2](45eb1180-672a-4745-bd85-f13c7bb658b7.xhtml), *Image Presentation*).
6.  Prepare a handle of a render pass that will be used during recording drawing operations. Store it in a variable of type `VkRenderPass` named `render_pass`.
7.  Prepare image views for all swapchain images. Store them in a variable of type `std::vector<VkImageView>` named `swapchain_image_views`.
8.  Store the size of the swapchain images in a variable of type `VkExtent2D` named `swapchain_size`.
9.  Create a variable of type `VkFramebuffer` named `framebuffer`.
10.  Create a framebuffer for the `render_pass` (with at least an image view corresponding to the swapchain's image at the position `image_index`) using the `logical_device`, `swapchain_image_views[image_index]` and `swapchain_size` variables. Store the created handle in the framebuffer variable (refer to the *Creating a framebuffer* recipe from [Chapter 6](2de4339d-8912-440a-89a6-fd1f84961448.xhtml), *Render Passes and Framebuffers*).
11.  Record a command buffer using the acquired swapchain image at the `image_index` position and the `framebuffer` variable. Store the handle of the recorded command buffer in a variable of type `VkCommandBuffer` named `command_buffer`.
12.  Prepare a queue that will process commands recorded in the `command_buffer`. Store the queue's handle in a variable of type `VkQueue` named `graphics_queue`.
13.  Take the handle of an unsignaled semaphore and store it in a variable of type `VkSemaphore` named `ready_to_present_semaphore`.
14.  Prepare an unsignaled fence and store its handle in a variable of type `VkFence` named `finished_drawing_fence`.
15.  Create a variable of type `WaitSemaphoreInfo` named `wait_semaphore_info` (refer to the *Submitting command buffers to a queue* recipe from [Chapter 3](fc38e0ae-51aa-4f6f-8fb3-551861273018.xhtml), *Command Buffers and Synchronization*). Initialize members of this variable using the following values:
    *   The `image_acquired_semaphore` variable for semaphore
    *   The `VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT` value for `WaitingStage`

16.  Submit the `command_buffer` to the `graphics_queue`, specifying one element vector with the `wait_semaphore_info` variable for the `wait_semaphore_infos` parameter, the `ready_to_present_semaphore` variable for the semaphore to be signaled, and the `finished_drawing_fence` variable for the fence to be signaled (refer to the *Submitting command buffers to the queue* recipe from [Chapter 3](fc38e0ae-51aa-4f6f-8fb3-551861273018.xhtml), *Command Buffers and Synchronization*).
17.  Prepare the handle of a queue used for presentation. Store it in a variable of type `VkQueue` named `present_queue`.
18.  Create a variable of type `PresentInfo` named `present_info` (refer to the *Presenting an image* recipe from [Chapter 2](45eb1180-672a-4745-bd85-f13c7bb658b7.xhtml), *Image Presentation*). Initialize members of this variable with the following values:
    *   The `swapchain` variable for `Swapchain`
    *   The `image_index` variable for `ImageIndex`
19.  Present the acquired swapchain image to the `present_queue` queue. Provide one element vector with the `ready_to_present_semaphore` variable as the `rendering_semaphores` parameter, and one element vector with the `present_info` variable as the `images_to_present` parameter (refer to the *Presenting an image* recipe from [Chapter 2](https://cdp.packtpub.com/vulkancookbook/wp-admin/post.php?post=605&action=edit#post_29), *Image Presentation*).

# How it works...

Preparing a single frame of animation can be divided into five steps:

1.  Acquiring a swapchain image.
2.  Creating a framebuffer.
3.  Recording a command buffer.
4.  Submitting the command buffer to the queue.
5.  Presenting an image.

First, we must acquire a swapchain image into which we can render. Rendering is performed inside a render pass that defines the parameters of attachments. Specific resources used for these attachments are defined in a framebuffer.

As we want to render into a swapchain image (to display the image on screen), this image must be specified as one of the attachments defined in a framebuffer. It may seem that creating a framebuffer earlier and reusing it during the rendering is a good idea. Of course, it is a valid approach but it has its drawbacks. The most important drawback is that it may be hard to maintain it during the lifetime of our application. We can render only into the image that was acquired from a swapchain. But as we don't know which image will be acquired, we need to prepare separate framebuffers for all swapchain images. What's more, we will need to recreate them each time a swapchain object is recreated. If our rendering algorithm requires more attachments to render into, we will start creating multiple variations of framebuffers for all combinations of swapchain images and images created by us. This becomes very cumbersome.

That's why it is much easier to create a framebuffer just before we start recording a command buffer. We create the framebuffer with only those resources that are needed to render this single frame. We just need to remember that we can destroy such a framebuffer only when the execution of a submitted command buffer is finished.

A framebuffer cannot be destroyed until the queue stops processing a command buffer in which the framebuffer was used.

When an image is acquired and a framebuffer is created, we can record a command buffer. These operations may be performed like this:

[PRE29]

After that, we are ready to submit the command buffer to the queue. Operations recorded in the command buffer must wait until the presentation engine allows us to use the acquired image. For this purpose, we specify a semaphore when the image is acquired. This semaphore must also be provided as one of the wait semaphores during command buffer submission:

[PRE30]

A rendered image can be presented (displayed on screen) when the queue stops processing the command buffer, but we don't want to wait and check when this happens. That's why we use an additional semaphore (the `ready_to_present_semaphore` variable in the preceding code) that will be signaled when the command buffer's execution is finished. The same semaphore is then provided when we present a swapchain image. This way, we synchronize operations internally on the GPU as this is much faster than synchronizing them on the CPU. If we weren't using the semaphore, we would need to wait until the fence is signaled and only then could we present an image. This would stall our application and hurt the performance considerably.

You may wonder why we need the fence (`finished_drawing_fence` in the preceding code), as it also gets signaled when the command buffer processing is finished. Isn't the semaphore enough? No, there are situations in which the application also needs to know when the execution of a given command buffer has ended. One such situation is when destroying the created framebuffer. We can't destroy it until the preceding fence is signaled. Only the application can destroy the resources it created, so it must know when it can safely destroy them (when they are not used anymore). Another example is re-recording of the command buffer. We can't record it again until its execution on a queue is finished. So we need to know when this happens. And, as the application cannot check the state of a semaphore, the fence must be used.

Using both a semaphore and a fence allows us to submit command buffers and present images immediately one after another, without unnecessary waits. And we can do these operations for multiple frames independently, increasing the performance even further.

# See also

*   In [Chapter 2](https://cdp.packtpub.com/vulkancookbook/wp-admin/post.php?post=605&action=edit#post_29), *Image Presentation*, see the following recipes:
    *   *Getting handles of swapchain images*
    *   *Acquiring a swapchain image*
    *   *Presenting an image*
*   In [Chapter 3](fc38e0ae-51aa-4f6f-8fb3-551861273018.xhtml), *Command Buffers and Synchronization*, see the following recipes:
    *   *Creating a semaphore*
    *   *Creating a fence*
    *   *Submitting command buffers to a queue*
    *   *Checking if processing of a submitted command buffer has finished*
*   In [Chapter 6](2de4339d-8912-440a-89a6-fd1f84961448.xhtml), *Render Passes and Framebuffers*, see the following recipes:
    *   *Creating a render pass*
    *   *Creating a framebuffer*

# Increasing the performance through increasing the number of separately rendered frames

Rendering a single frame of animation and submitting it to a queue is the goal of 3D graphics applications, such as games and benchmarks. But a single frame isn't enough. We want to render and display multiple frames or we won't achieve the effect of animation.

Unfortunately, we can't re-record the same command buffer immediately after we submit it; we must wait until the queue stops processing it. But, waiting until the command buffer processing is finished is a waste of time and it hurts the performance of our application. That's why we should render multiple frames of animation independently.

# Getting ready

For the purpose of this recipe, we will use variables of a custom `FrameResources` type. It has the following definition:

[PRE31]

The preceding type is used to define resources that manage the lifetime of a single frame of animation.

The `CommandBuffer` member stores a handle of a command buffer used to record operations of a single, independent frame of animation. In a real-life application, a single frame will be probably composed of multiple command buffers recorded in multiple threads. But for the purpose of a basic code sample, one command buffer is enough.

The `ImageAcquiredSemaphore` member is used to store a semaphore handle passed to the presentation engine when we acquire an image from a swapchain. This semaphore must then be provided as one of the wait semaphores when we submit the command buffer to a queue.

The `ReadyToPresentSemaphore` member indicates a semaphore that gets signaled when a queue stops processing our command buffer. We should use it during image presentation, so the presentation engine knows when the image is ready.

The `DrawingFinishedFence` member contains a fence handle. We provide it during the command buffer submission. Similarly to the `ReadyToPresentSemaphore` member, this fence gets signaled when the command buffer is no longer executed on a queue. But the fence is necessary to synchronize operations on the CPU side (the operations our application performs), not the GPU (and the presentation engine). When this fence is signaled, we know that we can both re-record the command buffer and destroy a framebuffer.

The `DepthAttachment` member is used to store an image view for an image serving as a depth attachment inside a sub-pass.

The `Framebuffer` member is used to store a temporary framebuffer handle created for the lifetime of a single frame of animation.

Most of the preceding members are wrapped into objects of a `VkDestroyer` type. This type is responsible for the implicit destruction of an owned object, when the object is no longer necessary.

# How to do it...

1.  Take the handle of a logical device and store it in a variable of type `VkDevice` named `logical_device`.
2.  Create a variable of type `std::vector<FrameResources>` named `frame_resources`. Resize it to hold the resources for the desired number of independently rendered frames (the recommended size is three), and initialize each element using the following values (the values stored in each element must be unique):

*   The handle of a created command buffer for `commandbuffer`
*   Two handles of created semaphores for `ImageAcquiredSemaphore` and `ReadyToPresentSemaphore`
*   The handle of a fence created in an already signaled state for `DrawingFinishedFence`

*   The handle of an image view for an image serving as a depth attachment for `DepthAttachment`
*   The `VK_NULL_HANDLE` value for `Framebuffer`

3.  Create a (potentially static) variable of type `uint32_t` named `frame_index`. Initialize it with a `0` value.
4.  Create a variable of type `FrameResources` named `current_frame` that references an element of the `frame_resources` vector pointed to by the `frame_index` variable.
5.  Wait until the `current_frame.DrawingFinishedFence` gets signaled. Provide the `logical_device` variable and a timeout value equal to `2000000000` (refer to the *Waiting for fences* recipe from [Chapter 3](fc38e0ae-51aa-4f6f-8fb3-551861273018.xhtml), *Command Buffers and Synchronization*).
6.  Reset the state of the `current_frame.DrawingFinishedFence` fence (refer to the *Resetting fences* recipe from [Chapter 3](fc38e0ae-51aa-4f6f-8fb3-551861273018.xhtml), *Command Buffers and Synchronization*).
7.  If the `current_frame.Framebuffer` member contains a handle of a created `framebuffer`, destroy it and assign a `VK_NULL_HANDLE` value to the member (refer to the *Destroying a framebuffer* recipe from [Chapter 6](2de4339d-8912-440a-89a6-fd1f84961448.xhtml), *Render Passes and Framebuffers*).
8.  Prepare a single frame of animation using all the members of the `current_frame` variable (refer to the *Preparing a single frame of animation* recipe):
    1.  Acquire a swapchain image providing the `current_frame.ImageAcquiredSemaphore` variable during this operation.
    2.  Create a framebuffer and store its handle in the `current_frame.Framebuffer` member.
    3.  Record a command buffer stored in the `current_frame.CommandBuffer` member.
    4.  Submit the `current_frame.CommandBuffer` member to a selected queue, providing the `current_frame.ImageaAquiredSemaphore` semaphore as one of the waiting semaphores, the `current_frame.ReadyToPresentSemaphore` semaphore as the semaphore to be signaled, and the `current_frame.DrawingFinishedFence` fence as the fence to be signaled when the command buffer's execution is finished.

5.  Present a swapchain image to a selected queue, providing the one element vector with the `current_frame.ReadyToPresentSemaphore` variable as the `rendering_semaphores` parameter.

9.  Increment a value stored in the `frame_index` variable. If it is equal to the number of elements in the `frame_resources` vector, reset the variable to `0`.

# How it works...

Rendering animation is performed in a loop. One frame is rendered and an image is presented, then usually the operating system messages are processed. Next, another frame is rendered and presented, and so on.

When we have only one command buffer and other resources required to prepare, render, and display a frame, we can't reuse them immediately. Semaphores cannot be used for another submission until the previous submission, in which they were used, has been finished. This situation requires us to wait for the end of the command buffer processing. But such waits are highly undesirable. The more we wait on the CPU, the more stalls we introduce to the graphics hardware and the worse performance we achieve.

To shorten the time we wait in our application (until a command buffer recorded for the previous frame is executed), we need to prepare several sets of resources required to render and present a frame. When we record and submit a command buffer for one frame and we want to prepare another frame, we just take another set of resources. For the next frame, we use yet another set of resources until we have used all of them. Then we just take the least recently used set--of course, we need to check if we can reuse it but, at this time, there is a high probability that it has already been processed by the hardware. The process of rendering animation using multiple sets of **Frame Resources** is presented in the following diagram:

![](img/image_09_004.png)

How many sets should we prepare? We may think that the more sets we have the better, because we won't need to wait at all. But unfortunately, the situation isn't that simple. First, we increase the memory footprint of our application. But, more importantly, we increase an input lag. Usually, we render animation based on the input from the user, who wants to rotate a virtual camera, view a model, or move a character. We want our application to respond to a user's input as quickly as possible. When we increase the number of independently rendered frames, we also increase the time between a user's input and the effect it has on the rendered image.

We need to balance the number of separately rendered frames, the performance of our application, its memory usage, and the input lag.

So, how many frame resources should we have? This of course depends on the complexity of the rendered scenes, the performance of the hardware on which the application is executed, and the type of rendering scenario it realizes (that is, the type of game we are creating--whether it is a fast **first-person perspective** (**FPP**) shooter or a racing game, or a more slow-paced tour based **role-playing** **game** (**RPG**)). So there is not one exact value that will fit all possible scenarios. Tests have shown that increasing the number of frame resources from one to two may increase the performance by 50%. Adding a third set increases the performance further, but the growth isn't as big this time. So, the performance gain is smaller with each additional set of frame resources. Three sets of rendering resources seems like a good choice, but we should perform our own tests and see what is best for our specific needs.

We can see three examples of recording and submitting command buffers with one, two, and three independent sets of resources needed to render frames of animations, as follows:

![](img/image_09_005.png)

Now that we know why we should use several independent numbers of frame resources, we can see how to render a frame using them.

First, we start by checking if we can use a given set of resources to prepare a frame. We do this by checking the status of a fence. If it is signaled, we are good to go. You may wonder, what should we do when we render the very first frame--we didn't submit anything to a queue yet, so the fence didn't have an opportunity to be signaled. It's true, and that's why, for the purpose of preparing frame resources, we should create fences in an already signaled state:

[PRE32]

We should also check if a framebuffer used for the frame was created. If it was, we should destroy it because it will be created later. For an acquired swapchain image, an `InitVkDestroyer()` function initializes the provided variable with a new, empty object handle and, if necessary, destroys the previously owned object. After that, we render the frame and present an image. To do this, we need a command buffer and two semaphores (refer to the *Preparing a single frame of animation* recipe):

[PRE33]

One last thing is to increase the index of the currently used set of frame resources. For the next frame of animation we will use another set, until we have used all of them, and we start from the beginning:

[PRE34]

# See also

*   In [Chapter 3](fc38e0ae-51aa-4f6f-8fb3-551861273018.xhtml), *Command Buffers and Synchronization*, see the following recipes:
    *   *Waiting for fences*
    *   *Resetting fences*
*   In [Chapter 6](2de4339d-8912-440a-89a6-fd1f84961448.xhtml), *Render Passes and Framebuffers*, see the following recipe:
    *   *Destroying a framebuffer*
*   *Preparing a single frame of animation *recipe in this chapter