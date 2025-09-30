# Render Passes and Framebuffers

In this chapter, we will cover the following recipes:

*   Specifying attachment descriptions
*   Specifying subpass descriptions
*   Specifying dependencies between subpasses
*   Creating a render pass
*   Creating a framebuffer
*   Preparing a render pass for geometry rendering and postprocess subpasses
*   Preparing a render pass and a framebuffer with color and depth attachments
*   Beginning a render pass
*   Progressing to the next subpass
*   Ending a render pass
*   Destroying a framebuffer
*   Destroying a render pass

# Introduction

In Vulkan, drawing commands are organized into render passes. A render pass is a collection of subpasses that describes how image resources (color, depth/stencil, and input attachments) are used: what their layouts are and how these layouts should be transitioned between subpasses, when we render into attachments or when we read data from them, if their contents are needed after the render pass, or if their usage is limited only to the scope of a render pass.

The aforementioned data stored in render passes is just a general description, or a metadata. The actual resources involved in the rendering process are specified with framebuffers. Through them, we define which image views are used for which rendering attachments.

We need to prepare all this information in advance, before we can issue (record) rendering commands. With that knowledge, drivers can greatly optimize the drawing process, limit the amount of memory needed for the rendering, or even use a very fast cache for some of the attachments, improving the performance even more.

In this chapter, we will learn how to organize drawing operations into a set of render passes and subpasses, which are required to draw anything with Vulkan. We will also learn how to prepare a description of render target attachments used during rendering (drawing) and how to create framebuffers, which define actual image views that will be used as these attachments.

# Specifying attachments descriptions

A render pass represents a set of resources (images) called attachments, which are used during rendering operations. These are divided into color, depth/stencil, input, or resolve attachments. Before we can create a render pass, we need to describe all the attachments used in it.

# How to do it...

1.  Create a vector with elements of type `VkAttachmentDescription`. Call the vector `attachments_descriptions`. For each attachment used in a render pass, add an element to the `attachments_descriptions` vector and use the following values for its members:
    *   `0` value for `flags`
    *   The selected format of a given attachment for `format`
    *   The number of per pixel samples for `samples`
    *   For `loadOp`, specify the type of operation that should be performed on an attachment's contents when a render pass is started--a `VK_ATTACHMENT_LOAD_OP_CLEAR` value if the attachment contents should be cleared, a `VK_ATTACHMENT_LOAD_OP_LOAD` value if its current contents should be preserved or a `VK_ATTACHMENT_LOAD_OP_DONT_CARE` value if we intend to overwrite the whole attachment by ourselves and we don't care about its current contents (this parameter is used for color attachments or for the depth aspect of depth/stencil attachments.)
    *   For `storeOp`, specify how an attachment's contents should be treated after the render pass--use a `VK_ATTACHMENT_STORE_OP_STORE` value if they should be preserved or a `VK_ATTACHMENT_STORE_OP_DONT_CARE` value if we don't need the contents after the rendering (this parameter is used for color attachments or for the depth aspect of depth/stencil attachments)
    *   Specify how the stencil aspect (component) of an attachment should be treated at the beginning of a render pass for `stencilLoadOp` (the same as for the `loadOp` member but for a stencil aspect of depth/stencil attachments)
    *   Specify how the stencil aspect (component) of an attachment should be treated after a render pass for `stencilStoreOp` (the same as for the `storeOp` but for a stencil aspect of depth/stencil attachments)
    *   Specify what layout image will have when a render pass begins for `initialLayout`
    *   Specify the layout to which image should be automatically transitioned to after a render pass for `finalLayout`

# How it works...

When we create a render pass, we have to create an array of attachment descriptions. This a general list of all the attachments used in a render pass. Indices into this array are then used for the subpass descriptions (refer to the *Specifying subpass descriptions* recipe). Similarly, when we create a framebuffer and specify exactly what image resource should be used for each attachment, we define a list where each element corresponds to the element of the attachment descriptions array.

Usually, when we draw a geometry, we render it into at least one color attachment. Probably, we also want a depth test to be enabled, so we need a depth attachment too. Attachment descriptions for such a common scenario are presented here:

[PRE0]

In the preceding example, we specify two attachments: one with a `R8G8B8A8_UNORM` and the other with a `D16_UNORM` format. Both attachments should be cleared at the beginning of a render pass (similarly to calling the OpenGL's `glClear()` function at the beginning of a frame). We also want to keep the contents of the first attachment, when the render pass is finished, but we don't need the contents of the second attachment. For both, we also specify an `UNDEFINED` initial layout. An `UNDEFINED` layout can always be used for an initial/old layout--it means that we don't need images content when a memory barrier is set up.

The value for the final layout depends on how we intend to use an image after the render pass. If we are rendering directly into a swapchain image and we want to display it on screen, we should use a `PRESENT_SRC` layout (as shown previously). For a depth attachment, if we don't intend to use a depth component after the render pass (which usually is true), we should set the same layout value as specified in the last subpass of a render pass.

It's also possible that a render pass does not use any attachments. In such a case, we don't need to specify attachment descriptions, but such a situation is rare.

# See also

The following recipes in this chapter:

*   *Specifying subpass descriptions*
*   *Creating a render pass*
*   *Creating a framebuffer*
*   *Preparing a render pass and a framebuffer with color and depth attachments*

# Specifying subpass descriptions

Operations performed in a render pass are grouped into subpasses. Each subpass represents a stage or a phase of our rendering commands in which a subset of render pass's attachments are used (into which we render or from which we read data).

A render pass always requires at least one subpass that is automatically started when we begin a render pass. And for each subpass, we need to prepare a description.

# Getting ready

To lower the number of parameters required to prepare for each subpass, a custom structure type is introduced for this recipe. It is a simplified version of a `VkSubpassDescription` structure defined in the Vulkan header. It has the following definition:

[PRE1]

The `PipelineType` member defines a type of a pipeline (graphics or compute, though only graphics pipelines are supported inside render passes at this point) that will be used during the subpass. `InputAttachments` is a collection of attachments from which we will read data during the subpass. `ColorAttachments` specifies all attachments that will be used as color attachments (into which we will render during the subpass). `ResolveAttachments` specifies which color attachments should be resolved (changed from a multisampled image to a non-multisampled/single sampled image) at the end of the subpass. `DepthStencilAttachment`, if used, specifies which attachment is used as a depth and/or stencil attachment during the subpass. `PreserveAttachments` is a set of attachments that are not used in the subpass but whose contents must be preserved during the whole subpass.

# How to do it...

1.  Create a vector variable of type `std::vector<VkSubpassDescription>` named `subpass_descriptions`. For each subpass defined in a render pass, add an element to the `subpass_descriptions` vector and use the following values for its members:
    *   `0` value for `flags`
    *   `VK_PIPELINE_BIND_POINT_GRAPHICS` value for `pipelineBindPoint` (currently only graphics pipelines are supported inside render passes)
    *   The number of input attachments used in the subpass for `inputAttachmentCount`
    *   A pointer to the first element of an array with parameters of input attachments (or a `nullptr` value if no input attachments are used in the subpass) for `pInputAttachments`; use the following values for each member of the `pInputAttachments` array:
        *   Index of the attachment in the list of all render pass attachments for `attachment`
        *   A layout given image should be automatically transitioned to at the beginning of the subpass for `layout`
    *   The number of color attachments used in the subpass for `colorAttachmentCount`
    *   A pointer to the first element of the array with parameters of the subpass's color attachments (or a `nullptr` value if no color attachments are used in the subpass) for `pColorAttachments`; for each member of the array, specify values as described in points 4a and 4b.
    *   If any of the color attachments should be resolved (changed from multisampled to single-sampled) for `pResolveAttachments,` specify a pointer to the first element of the array with same number of elements as `pColorAttachments` or use a `nullptr` value if no color attachments need to be resolved; each member of the `pResolveAttachments` array corresponds to the color attachment at the same index and specifies to which attachment a given color attachment should be resolved at the end of the subpass; for each member of the array use specified values as described in points 4a and 4b; use a `VK_ATTACHMENT_UNUSED` value for the attachment index if the given color attachment should not be resolved.
    *   For `pDepthStencilAttachment` provide a pointer to the variable of type `VkAttachmentReference` if a depth/stencil attachment is used (or a `nullptr` value if no depth/stencil attachment is used in the subpass); for members of this variable, specify values as described in points 4a and 4b
    *   The number of attachments that are not used but whose contents should be preserved for `preserveAttachmentCount`.
    *   A pointer to the first element of an array with indices of attachments whose contents should be preserved in the subpass (or a `nullptr` value if there are no attachments to be preserved) for `pPreserveAttachments`.

# How it works...

Vulkan render passes must have at least one subpass. Subpass parameters are defined in an array of `VkSubpassDescription` elements. Each such element describes how attachments are used in a corresponding subpass. There are separate lists of input, color, resolve, and preserved attachments and a single entry for depth/stencil attachments. Each of these members may be empty (or null). In this case, attachments of a corresponding type are not used in a subpass.

Each entry in one of the lists just described is a reference to the list of all attachments specified for a render pass in attachment descriptions (refer to the *Specifying attachments descriptions* recipe). Additionally, each entry specifies a layout in which an image should be during a subpass. Transitions to specified layouts are performed automatically by the driver.

Here is a code sample that uses a custom structure of a `SubpassParameters` type to specify a subpass definition:

[PRE2]

And here is a code sample defining one subpass that corresponds to an example with one color attachment: a depth/stencil attachment:

[PRE3]

First we specify a `depth_stencil_attachment` variable for a description of a depth/stencil attachment. For a depth data, the second attachment from the list of attachment descriptions is used; that's why we specify a value of `1` for its index (refer to the *Specifying attachment descriptions* recipe). And as we want to render into this attachment, we provide a `VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL` value for its layout (the driver will automatically perform a transition, if needed).

In the example, we use just one color attachment. It is the first attachment from the list of attachment descriptions, so we use a `0` value for its index. When we render into a color attachment, we should specify a `VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL` value for its layout.

One last thing--as we want to render a geometry, we need to use a graphics pipeline. This is done through a `VK_PIPELINE_BIND_POINT_GRAPHICS` value provided for a `PipelineType` member.

As we don't use input attachments and we don't want to resolve any color attachments, their corresponding vectors are empty.

# See also

The following recipes in this chapter:

*   *Specifying attachment descriptions*
*   *Creating a render pass*
*   *Creating a framebuffer*
*   *Preparing a render pass for geometry rendering and postprocess subpasses*
*   *Preparing a render pass and a framebuffer with color and depth attachments*

# Specifying dependencies between subpasses

When operations in a given subpass depend on the results of operations in one of the earlier subpasses in the same render pass, we need to specify subpass dependencies. This is also required if there are dependencies between operations recorded within a render pass and those performed before it, or between operations that are executed after a render pass and those performed within the render pass. It is also possible to define dependencies within a single subpass.

Defining subpass dependencies is similar to setting up memory barriers.

# How to do it...

1.  Create a variable of type `std::vector<VkSubpassDependency>` named `subpass_dependencies`. For each dependency, add a new element to the `subpass_dependencies` vector and use the following values for its members:
    *   The index of a subpass from which ("producing") operations should be finished before the second set of ("consuming") operations (or a `VK_SUBPASS_EXTERNAL` value for commands before the render pass) for `srcSubpass`
    *   The index of a subpass whose operations depend on the previous set of commands (or a `VK_SUBPASS_EXTERNAL` value for operations after the render pass) for `dstSubpass`
    *   The set of pipeline stages which produce the result read by the "consuming" commands for `srcStageMask`
    *   The set of pipeline stages which depend on the data generated by the "producing" commands for `dstStageMask`
    *   The types of memory operations that occurred for the "producing" commands for `srcAccessMask`
    *   The types of memory operations that will be performed in "consuming" commands for `dstAccessMask`
    *   For `dependencyFlags`, use a `VK_DEPENDENCY_BY_REGION_BIT` value if the dependency is defined by region--it means that operations generating data for a given memory region must finish before operations reading data from the same region can be executed; if this flag is not specified, dependency is global, which means that data for the whole image must be generated before "consuming" commands can be executed.

# How it works...

Specifying dependencies between subpasses (or between subpasses and commands before or after a render pass) is very similar to setting an image memory barrier and serves a similar purpose. We do this when we want to specify that commands from one subpass (or commands after the render pass) depend on results of operations performed in another subpass (or on commands executed before the render pass). We don't need to set up dependencies for the layout transitions--these are performed automatically based on the information provided for the render pass attachment and subpass descriptions. What's more, when we specify different attachment layouts for different subpasses, but in both subpasses the given attachment is used only for reading, we also don't need to specify a dependency.

Subpass dependencies are also required when we want to set up image memory barriers inside a render pass. Without specifying a so-called "self-dependency" (the source and destination subpass have the same index), we can't do that. However, if we define such a dependency for a given subpass, we can record a memory barrier in it. In other situations, the source subpass index must be lower than the target subpass index (excluding a `VK_SUBPASS_EXTERNAL` value).

There follows an example in which we prepare a dependency between two subpasses--the first draws geometry into color and depth attachments, and the second uses color data for postprocessing (it reads from the color attachment):

[PRE4]

The aforementioned dependency is set between the first and second subpasses (indices with values of 0 and 1). Writes to the color attachment are performed in the `COLOR_ATTACHMENT_OUTPUT` stage. Postprocessing is done in a fragment shader and this stage is defined as a "consuming" stage. When we draw a geometry, we perform writes to a color attachment (access mask with value of `COLOR_ATTACHMENT_WRITE`). Then the color attachment is used as an input attachment and in the postprocess subpass we read from it (so we use an access mask with a value of `INPUT_ATTACHMENT_READ`). As we don't need to read data from other parts of an image, we can specify dependency by-region (a fragment stores a color value at given coordinates in the first subpass and the same value is read in the next subpass by a fragment with the same coordinates). When we do this, we should not assume that regions are larger than the single pixel, because the size of a region may be different on various hardware platforms.

# See also

The following recipes in this chapter:

*   *Specifying attachment descriptions*
*   *Specifying subpass descriptions*
*   *Creating a render pass*
*   *Preparing a render pass for geometry rendering and postprocess subpasses*

# Creating a render pass

Rendering (drawing a geometry) can only be performed inside render passes. When we also want to perform other operations such as image postprocessing or preparing geometry and light prepass data, we need to order these operations into subpasses. For this, we specify descriptions of all the required attachments, all subpasses into which operations are grouped, and the necessary dependencies between those operations. When this data is prepared, we can create a render pass.

# Getting ready

To lower the number of parameters that need to be provided, in this recipe, we use a custom structure of type `SubpassParameters` (refer to the *Specifying subpass descriptions* recipe).

# How to do it...

1.  Create a variable of type `std::vector<VkAttachmentDescription>` named `attachments_descriptions,` in which we specify descriptions of all render pass attachments (refer to the *Specifying attachment descriptions* recipe).
2.  Prepare a variable of type `std::vector<VkSubpassDescription>` named `subpass_descriptions` and use it to define descriptions of subpasses (refer to the *Specifying subpass descriptions* recipe).

3.  Create a variable of type `std::vector<VkSubpassDependency>` named `subpass_dependencies`. Add a new member to this vector for each dependency that needs to be defined in the render pass (refer to the *Specifying dependencies between subpasses* recipe).
4.  Create a variable of type `VkRenderPassCreateInfo` named `render_pass_create_info` and initialize its member with the following values: 
    *   `VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO` value for `sType`
    *   `nullptr` value for `pNext`
    *   `0` value for `flags`
    *   The number of elements in the `attachments_descriptions` vector for `attachmentCount`
    *   A pointer to the first element of the `attachments_descriptions` vector (or a `nullptr` value if it is empty) for `pAttachments`
    *   The number of elements in the `subpass_descriptions` vector for `subpassCount`
    *   A pointer to the first element of the `subpass_descriptions` vector for `pSubpasses`
    *   The number of elements in the `subpass_dependencies` vector for `dependencyCount`
    *   A pointer to the first element of the `subpass_dependencies` vector (or a `nullptr` value if it is empty) for `pDependencies`
5.  Take the handle of a logical device for which the render pass should be created. Store it in a variable of type `VkDevice` named `logical_device`.
6.  Create a variable of type `VkRenderPass` named `render_pass` in which the handle of the created render pass will be stored.
7.  Call `vkCreateRenderPass( logical_device, &render_pass_create_info, nullptr, &render_pass )`. For the call, provide the `logical_device` variable, a pointer to the `render_pass_create_info` variable, a `nullptr` value, and a pointer to the `render_pass` variable.
8.  Make sure the call was successful by checking if it returned a `VK_SUCCESS` value.

# How it works...

A render pass defines general information about how attachments are used by operations performed in all its subpasses. This allows the driver to optimize work and improve the performance of our application.

![](img/image_06_001.png)

The most important parts of a render pass creation is a preparation of data--descriptions of all the used attachments and subpasses and a specification of dependencies between subpasses (refer to the *Specifying attachment descriptions*, *Specifying subpass descriptions,* and *Specifying dependencies between subpasses* recipes in this chapter). These steps can be presented in short as follows:

[PRE5]

This data is then used when we specify parameter for a function creating a render pass:

[PRE6]

But for the drawing operations to be performed correctly, the render pass is not enough as it only specifies how operations are ordered into subpasses and how attachments are used. There is no information about what images are used for these attachments. Such information about specific resources used for all defined attachments is stored in framebuffers.

# See also

The following recipes in this chapter:

*   *Specifying attachment descriptions*
*   *Specifying subpass descriptions*
*   *Specifying dependencies between subpasses*
*   *Creating a framebuffer*
*   *Beginning a render pass*
*   *Progressing to the next subpass*
*   *Ending a render pass*
*   *Destroying a render pass*

# Creating a framebuffer

Framebuffers are used along with render passes. They specify what image resources should be used for corresponding attachments defined in a render pass. They also define the size of a renderable area. That's why when we want to record drawing operations, we not only need to create a render pass, but also a framebuffer.

# How to do it...

1.  Take the handle of a render pass that should be compatible with the framebuffer and use it to initialize a variable of type `VkRenderPass` named `render_pass`.
2.  Prepare a list of image view handles that represent the images' subresources, which should be used for the render pass attachments. Store all the prepared image views in a variable of type `std::vector<VkImageView>` named `attachments`.
3.  Create a variable of type `VkFramebufferCreateInfo` named `framebuffer_create_info`. Use the following values to initialize its members: 
    *   `VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO` value for `sType`
    *   `nullptr` value for `pNext`
    *   `0` value for `flags`
    *   `render_pass` variable for `renderPass`
    *   The number of elements in the `attachments` vector for `attachmentCount`
    *   A pointer to the first element of the `attachments` vector (or a `nullptr` value if it is empty) for `pAttachments`
    *   The selected width of a renderable area for `width`
    *   The selected framebuffer's height for `height`
    *   The number of framebuffer layers for `layers`
4.  Take the handle of a logical device for which the framebuffer should be created and store in a variable of type `VkDevice` named `logical_device`.
5.  Create a variable of type `VkFramebuffer` named `framebuffer` that will be initialized with a handle of a created framebuffer.

6.  Call `vkCreateFramebuffer( logical_device, &framebuffer_create_info, nullptr, &framebuffer )` for which we provide the `logical_device` variable, a pointer to the `framebuffer_create_info` variable, a `nullptr` value, and a pointer to the `framebuffer` variable.
7.  Make sure the framebuffer was properly created by checking if the call returned a `VK_SUCCESS` value.

# How it works...

Framebuffers are always created in conjunction with render passes. They define specific image subresources that should be used for attachments specified in render passes, so both of these object types should correspond to each other.

![](img/image_06_002.png)

When we create a framebuffer, we provide a render pass object with which we can use the given framebuffer. However, we are not limited to using it only with the specified render pass. We can use the framebuffer also with all render passes that are compatible with the one provided.

What are compatible render passes? First, they must have the same number of subpasses. And each subpass must have a compatible set of input, color, resolve, and depth/stencil attachments. This means that formats and the number of samples of corresponding attachments must be the same. However, it is possible for the attachments to have different initial, subpasses and final layouts and different load and store operations.

Apart from that, framebuffers also define the size of a renderable area--the dimensions into which all rendering will be confined. However, what we need to remember is that it is up to us to make sure that the pixels/fragments outside of the specified range are not modified. For this purpose, we need to specify the appropriate parameters (viewport and scissor test) during the pipeline creation or when setting corresponding dynamic states (refer to the *Preparing viewport and scissor test state* recipe from [Chapter 8](5744ea05-b18a-4f84-a1df-250b549dfea5.xhtml), *Graphics and Compute Pipelines* and to the *Setting a dynamic viewport and scissors state* recipe from [Chapter 9](0a69f5b5-142e-422b-aa66-5cb09a6467b3.xhtml), *Command Recording and Drawing*).

We must ensure that rendering occurs only in the dimensions specified during the framebuffer creation.

When we begin a render pass in a command buffer and use the given framebuffer, we also need to make sure that the images' subresources specified in that framebuffer are not used for any other purpose. In other words, if we use a given portion of an image as a framebuffer attachment, we can't use it in any other way during the render pass.

Image subresources specified for render pass attachments cannot be used for any other (non-attachment) purpose between the beginning and the end of the render pass.

Here is a code sample responsible for creating a framebuffer:

[PRE7]

# See also

In [Chapter 4](f1332ca0-b5a2-49bd-ac41-e37068e31042.xhtml), *Resources and Memory*, see the following recipes:

*   *Creating an image*
*   *Creating an image view*

The following recipes in this chapter:

*   *Specifying attachment descriptions*
*   *Creating a framebuffer*

# Preparing a render pass for geometry rendering and postprocess subpasses

When developing applications such as games or CAD tools there are often situations in which we need to draw a geometry and then, when the whole scene is rendered, we apply additional image effects called postprocessing.

In this sample recipe, we will see how to prepare a render pass in which we will have two subpasses. The first subpass renders into two attachments--color and depth. The second subpass reads data from the first color attachment and renders into another color attachment--a swapchain image that can be presented (displayed on screen) after the render pass.

# Getting ready

To lower the number of parameters that need to be provided, in this recipe we use a custom structure of type `SubpassParameters` (refer to the *Specifying subpass descriptions* recipe).

# How to do it...

1.  Create a variable of type `std::vector<VkAttachmentDescription>` named `attachments_descriptions`. Add an element to the `attachments_descriptions` vector that describes the first color attachment. Initialize it with the following values:
    *   `0` value for `flags`
    *   `VK_FORMAT_R8G8B8A8_UNORM` value for `format`
    *   `VK_SAMPLE_COUNT_1_BIT` value for `samples`
    *   `VK_ATTACHMENT_LOAD_OP_CLEAR` value for `loadOp`
    *   `VK_ATTACHMENT_STORE_OP_DONT_CARE` value for `storeOp`
    *   `VK_ATTACHMENT_LOAD_OP_DONT_CARE` value for `stencilLoadOp`
    *   `VK_ATTACHMENT_STORE_OP_DONT_CARE` value for `stencilStoreOp`
    *   `VK_IMAGE_LAYOUT_UNDEFINED` value for `initialLayout`
    *   `VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL` value for `finalLayout`
2.  Add another element to the `attachments_descriptions` vector that specifies the depth/stencil attachment. Use the following values to initialize its members:
    *   `0` value for `flags`
    *   `VK_FORMAT_D16_UNORM` value for `format`
    *   `VK_SAMPLE_COUNT_1_BIT` value for `samples`
    *   `VK_ATTACHMENT_LOAD_OP_CLEAR`  value for `loadOp`
    *   `VK_ATTACHMENT_STORE_OP_DONT_CARE` value for `storeOp`
    *   `VK_ATTACHMENT_LOAD_OP_DONT_CARE` value for `stencilLoadOp`
    *   `VK_ATTACHMENT_STORE_OP_DONT_CARE` value for `stencilStoreOp`
    *   `VK_IMAGE_LAYOUT_UNDEFINED` value for `initialLayout`
    *   `VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL` value for `finalLayout`
3.  Add a third element to the `attachments_descriptions` vector. This time it will specify another color attachment. Initialize it with the following values:
    *   `0` value for `flags`
    *   `VK_FORMAT_R8G8B8A8_UNORM` value for `format`
    *   `VK_SAMPLE_COUNT_1_BIT` value for `samples`

*   `VK_ATTACHMENT_LOAD_OP_CLEAR` value for `loadOp`
*   `VK_ATTACHMENT_STORE_OP_STORE` value for `storeOp`
*   `VK_ATTACHMENT_LOAD_OP_DONT_CARE` value for `stencilLoadOp`
*   `VK_ATTACHMENT_STORE_OP_DONT_CARE` value for `stencilStoreOp`
*   `VK_IMAGE_LAYOUT_UNDEFINED` value for `initialLayout`
*   `VK_IMAGE_LAYOUT_PRESENT_SRC_KHR` value for `finalLayout`

4.  Create a variable of type `VkAttachmentReference` named `depth_stencil_attachment` and initialize it with the following values:
    *   `1` value for `attachment`
    *   `VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL` value for `layout`
5.  Create a variable of type `std::vector<SubpassParameters>` named `subpass_parameters` and add one element with the following values to this vector:
    *   `VK_PIPELINE_BIND_POINT_GRAPHICS` value for `PipelineType`
    *   An empty vector for `InputAttachments`
    *   A vector with one element and the following values for `ColorAttachments`:
        *   `0` value for `attachment`
        *   `VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL` value for `layout`
    *   An empty vector for `ResolveAttachments`
    *   A pointer to the `depth_stencil_attachment` variable for `DepthStencilAttachment`
    *   An empty vector for `PreserveAttachments`
6.  Add the second element to the `subpass_parameters` that describes the second subpass. Initialize its member using the following values:
    *   `VK_PIPELINE_BIND_POINT_GRAPHICS` value for `PipelineType`
    *   A vector with one element with the following values for `InputAttachments`:
        *   `0` value for `attachment`
        *   `VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL` value for `layout`
    *   A vector with one element with the following values for `ColorAttachments`:
        *   `2` value for `attachment`
        *   `VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL` value for `layout`
    *   An empty vector for `ResolveAttachments`
    *   `nullptr` value for `DepthStencilAttachment`
    *   An empty vector for `PreserveAttachments`
7.  Create a variable of type `std::vector<VkSubpassDependency>` named `subpass_dependencies` with a single element that uses the following values for its members:
    *   `0` value for `srcSubpass`
    *   `1` value for `dstSubpass`
    *   `VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT` value for `srcStageMask`
    *   `VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT` value for `dstStageMask`
    *   `VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT` value for `srcAccessMask`
    *   `VK_ACCESS_INPUT_ATTACHMENT_READ_BIT` value for `dstAccessMask`
    *   `VK_DEPENDENCY_BY_REGION_BIT` value for `dependencyFlags`
8.  Create the render pass using `attachments_descriptions`, `subpass_parameters` and `subpass_dependencies` variables. Store its handle in a variable of type `VkRenderPass` named `render_pass` (refer to the *Creating a render pass* recipe in this chapter).

# How it works...

In this recipe, we create a render pass with the three attachments. They are specified as follows:

[PRE8]

First there is a color attachment into which we render in the first subpass and from which we read in the second subpass. The second attachment is used for a depth data; and the third is another color attachment into which we render in the second subpass. As we don't need the contents of the first and second attachments after the render pass (we need the contents of the first attachment only in the second subpass), we specify a `VK_ATTACHMENT_STORE_OP_DONT_CARE` value for their store operations. We also don't need their contents at the beginning of the render pass, so we specify an `UNDEFINED` initial layout. We also clear all three attachments.

Next we define two subpasses:

[PRE9]

The first subpass uses a color attachment and a depth attachment. The second subpass reads from the first attachment (used here as an input attachment) and renders into the third attachment.

The last thing is to define a dependency between two subpasses for the first attachment, which is once a color attachment (we write data to it) and once an input attachment (we read data from it). After that we can create the render pass like this:

[PRE10]

# See also

The following recipes in this chapter:

*   *Specifying attachment descriptions*
*   *Specifying subpass descriptions*
*   *Specifying dependencies between subpasses*
*   *Creating a render pass*

# Preparing a render pass and a framebuffer with color and depth attachments

Rendering a 3D scene usually involves not only a color attachment, but also a depth attachment used for depth testing (we want further objects to be occluded by the objects closer to the camera).

In this sample recipe, we will see how to create images for color and depth data and a render pass with a single subpass that renders into color and depth attachments. We will also create a framebuffer that will use both images for the render pass attachments.

# Getting ready

As in earlier recipes from this chapter, in this recipe we will use a custom structure of type `SubpassParameters` (refer to the *Specifying subpass descriptions* recipe).

# How to do it...

1.  Create a 2D image and image view for it with a `VK_FORMAT_R8G8B8A8_UNORM` format, `VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT` usage and a `VK_IMAGE_ASPECT_COLOR_BIT` aspect. Choose the rest of the image's parameters. Store the created handles in variables of type `VkImage` named `color_image`, of type `VkDeviceMemory` named `color_image_memory_object,` and of type `VkImageView` named `color_image_view` (refer to the *Creating a 2D image and view* recipe from [Chapter 4](f1332ca0-b5a2-49bd-ac41-e37068e31042.xhtml), *Resources and Memory*).
2.  Create a second 2D image and image view for it with a `VK_FORMAT_D16_UNORM` format, `VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT` usage, `VK_IMAGE_ASPECT_DEPTH_BIT` aspect, and the same size as the image whose handle is stored in the `color_image` variable. Choose the rest of the image's parameters. Store the created handles in variables of type `VkImage` named `depth_image`, of type `VkDeviceMemory` named `depth_image_memory_object,` and of type `VkImageView` named `depth_image_view` (refer to the *Creating a 2D image and view* recipe from [Chapter 4](f1332ca0-b5a2-49bd-ac41-e37068e31042.xhtml), *Resources and Memory*).

3.  Create a variable of type `std::vector<VkAttachmentDescription>` named `attachments_descriptions` and add two elements to the vector. Initialize the first element with the following values:
    *   `0` value for `flags`
    *   `VK_FORMAT_R8G8B8A8_UNORM` value for `format`
    *   `VK_SAMPLE_COUNT_1_BIT` value for `samples`
    *   `VK_ATTACHMENT_LOAD_OP_CLEAR` value for `loadOp`
    *   `VK_ATTACHMENT_STORE_OP_STORE` value for `storeOp`
    *   `VK_ATTACHMENT_LOAD_OP_DONT_CARE` value for `stencilLoadOp`
    *   `VK_ATTACHMENT_STORE_OP_DONT_CARE` value for `stencilStoreOp`
    *   `VK_IMAGE_LAYOUT_UNDEFINED` value for `initialLayout`
    *   `VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL` value for `finalLayout`
4.  Use these values to initialize members of the second element of the `attachments_descriptions` vector:
    *   `0` value for `flags`
    *   `VK_FORMAT_D16_UNORM` value for `format`
    *   `VK_SAMPLE_COUNT_1_BIT` value for `samples`
    *   `VK_ATTACHMENT_LOAD_OP_CLEAR` value for `loadOp`
    *   `VK_ATTACHMENT_STORE_OP_STORE` value for `storeOp`
    *   `VK_ATTACHMENT_LOAD_OP_DONT_CARE` value for `stencilLoadOp`
    *   `VK_ATTACHMENT_STORE_OP_DONT_CARE` value for `stencilStoreOp`
    *   `VK_IMAGE_LAYOUT_UNDEFINED` value for `initialLayout`
    *   `VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL` value for `finalLayout`
5.  Create a variable of type `VkAttachmentReference` named `depth_stencil_attachment` and initialize it using the following values:
    *   `1` value for `attachment`
    *   `VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL` value for `layout`

6.  Create a vector of type `std::vector<SubpassParameters>` named `subpass_parameters`. Add a single element to this vector and use the following values to initialize it:
    *   `VK_PIPELINE_BIND_POINT_GRAPHICS` value for `PipelineType`
    *   An empty vector for `InputAttachments`
    *   A vector with just one element with these values for `ColorAttachments`:
        1.  `0` value for `attachment`
        2.  `VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL` value for `layout`
    *   An empty vector for `ResolveAttachments`
    *   A pointer to the `depth_stencil_attachment` variable for `DepthStencilAttachment`
    *   An empty vector for `PreserveAttachments`
7.  Create a vector of type `std::vector<VkSubpassDependency>` named `subpass_dependencies` with a single element initialized using these values:
    *   `0` value for `srcSubpass`
    *   `VK_SUBPASS_EXTERNAL` value for `dstSubpass`
    *   `VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT` value for `srcStageMask`
    *   `VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT` value for `dstStageMask`
    *   `VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT` value for `srcAccessMask`
    *   `VK_ACCESS_SHADER_READ_BIT` value for `dstAccessMask`
    *   `0` value for `dependencyFlags`
8.  Create a render pass using `attachments_descriptions`, `subpass_parameters` and `subpass_dependencies` vectors. Store the created render pass handle in a variable of type `VkRenderPass` named `render_pass` (refer to the *Creating a render pass* recipe in this chapter).
9.  Create a framebuffer using the `render_pass` variable and the `color_image_view` variable for its first attachment and the `depth_image_view` variable for the second attachment. Specify the same dimensions as used for the `color_image` and `depth_image` variables. Store the created framebuffer handle in a variable of type `VkFramebuffer` named `framebuffer`.

# How it works...

In this sample recipe, we want to render into two images--one for color data, and another for the depth data. We imply that after the render pass they will be used as textures (we will sample them in shaders in another render pass); that's why they are created with `COLOR_ATTACHMENT` / `DEPTH_STENCIL_ATTACHMENT` usages (so we can render into them) and `SAMPLED` usage (so they both can be sampled from in shaders):

[PRE11]

Next we specify two attachments for the render pass. They are both cleared at the beginning of the render pass and their contents are preserved after the render pass:

[PRE12]

The next step is to define a single subpass. It uses the first attachment for color writes and the second attachment for depth/stencil data:

[PRE13]

Finally, we define a dependency between the subpass and the commands that will be performed after the render pass. This is required, because we don't want other commands to start reading our images before their contents are fully written in the render pass. We also create the render pass and a framebuffer:

[PRE14]

# See also

*   In [Chapter 4](f1332ca0-b5a2-49bd-ac41-e37068e31042.xhtml), *Resources and Memory*, see the following recipe:
    *   *Creating a 2D image and view*
*   The following recipes in this chapter:
    *   *Specifying attachment descriptions*
    *   *Specifying subpass descriptions*
    *   *Specifying dependencies between subpasses*
    *   *Creating a render pass*
    *   *Creating a framebuffer*

# Beginning a render pass

When we have created a render pass and a framebuffer and we are ready to start recording commands needed to render a geometry, we must record an operation that begins the render pass. This also automatically starts its first subpass.

# How to do it...

1.  Take the handle of a command buffer stored in a variable of type `VkCommandBuffer` named `command_buffer`. Make sure the command buffer is in the recording state.
2.  Use the handle of the render pass to initialize a variable of type `VkRenderPass` named `render_pass`.

3.  Take the framebuffer that is compatible with the `render_pass`. Store its handle in a variable of type `VkFramebuffer` named `framebuffer`.
4.  Specify the dimensions of the render area into which rendering will be confined during the render pass. This area cannot be larger than the size specified for the framebuffer. Store the dimensions in a variable of type `VkRect2D` named `render_area`.
5.  Create a variable of type `std::vector<VkClearValue>` named `clear_values` with the number of elements equal to the number of attachments in the render pass. For each render pass attachment that uses a clear `loadOp`, provide the corresponding clear value at the same index as the attachment index.
6.  Prepare a variable of type `VkSubpassContents` named `subpass_contents` describing how operations in the first subpass are recorded. Use a `VK_SUBPASS_CONTENTS_INLINE` value if commands are recorded directly and no secondary command buffer will be executed, or a `VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS` value to specify that commands for the subpass are stored in the secondary command buffer and only executing a secondary command buffer command will be used (refer to the *Executing a secondary command buffer inside a primary command buffer* recipe from [Chapter 9](0a69f5b5-142e-422b-aa66-5cb09a6467b3.xhtml), *Command Recording and Drawing*).
7.  Create a variable of type `VkRenderPassBeginInfo` named `render_pass_begin_info` and initialize its members using these values:
    *   `VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO` value for `sType`
    *   `nullptr` value for `pNext`
    *   `render_pass` variable for `renderPass`
    *   `framebuffer` variable for `framebuffer`
    *   `render_area` variable for `renderArea`
    *   Number of elements in the `clear_values` vector for `clearValueCount`
    *   Pointer to the first element of the `clear_values` vector (or a `nullptr` value if it is empty) for `pClearValues`

8.  Call `vkCmdBeginRenderPass( command_buffer, &render_pass_begin_info, subpass_contents )`, providing the `command_buffer` variable, pointer to the `render_pass_begin_info` variable and the `subpass_contents` variable.

# How it works...

Starting a render pass automatically starts its first subpass. Before this is done, all attachments, for which a clear `loadOp` was specified, are cleared--filled with a single color. Values used for clearing (and the rest of the parameters required to start a render pass) are specified in a variable of type `VkRenderPassBeginInfo`:

[PRE15]

An array with clearing values must have at least as many elements so they can correspond to attachments from the start to the last cleared attachment (the attachment with the greatest index that is being cleared). It is safer to have the same number of clear values as there are attachments in the render pass, but we only need to provide values for the cleared ones. If no attachments are cleared, we can provide a `nullptr` value for the clear values array.

When we start a render pass, we also need to provide the dimensions of the render area. It can be as large as the dimensions of the framebuffer, but can be smaller. It is up to us to make sure that the rendering will be confined to the specified area, or the pixels outside of this range may become undefined.

To begin the render pass we need to call:

[PRE16]

# See also

*   In [Chapter 3](fc38e0ae-51aa-4f6f-8fb3-551861273018.xhtml), *Command Buffers and Synchronization*, see the following recipe:
    *   *Beginning a command buffer recording operation*
*   In [Chapter 9](0a69f5b5-142e-422b-aa66-5cb09a6467b3.xhtml), *Command Recording and Drawing*, see the following recipe:
    *   *Executing a secondary command buffer inside a primary command buffer*
*   The following recipes in this chapter:
    *   *Creating a render pass*
    *   *Creating a framebuffer*

# Progressing to the next subpass

Commands that are recorded inside a render pass are divided into subpasses. When a set of commands from a given subpass is already recorded and we want to record commands for another subpass, we need to switch (or progress) to the next subpass.

# How to do it...

1.  Take the handle of a command buffer that's being recorded and store it in a variable of type `VkCommandBuffer` named `command_buffer`. Make sure the operation of beginning a render pass was already recorded in the `command_buffer`.
2.  Specify how subpass commands are recorded: directly or through a secondary command buffer. Store the appropriate value in a variable of type `VkSubpassContents` named `subpass_contents` (refer to the *Beginning a render pass* recipe).
3.  Call `vkCmdNextSubpass( command_buffer, subpass_contents )`. For the call, provide the `command_buffer` and `subpass_contents` variables.

# How it works...

Progressing to the next subpass switches from the current to the next subpass in the same render pass. During this operation appropriate layout transitions are performed and memory and execution dependencies are introduced (similar to those in memory barriers). All this is performed automatically by the driver, if needed, so the attachments in the new subpass can be used in the way specified during the render pass creation. Moving to the next subpass also performs multisample resolve operations on specified color attachments.

Commands in the subpass can be recorded directly, by inlining them in the command buffer, or indirectly by executing a secondary command buffer.

To record an operation that switches from one subpass to another, we need to call a single function:

[PRE17]

# See also

The following recipes in this chapter:

*   *Specifying subpass descriptions*
*   *Creating a render pass*
*   *Beginning a render pass*
*   *Ending a render pass*

# Ending a render pass

When all commands from all subpasses are already recorded, we need to end (stop or finish) a render pass.

# How to do it...

1.  Take the handle of a command buffer and store it in a variable of type `VkCommandBuffer` named `command_buffer`. Make sure the command buffer is in a recording state and that the operation of beginning a render pass was already recorded in it.
2.  Call `vkCmdEndRenderPass( command_buffer )` for which provide the `command_buffer` variable.

# How it works...

To end a render pass, we need to call a single function:

[PRE18]

Recording this function in a command buffer performs multiple operations. Execution and memory dependencies are introduced (like the ones in memory barriers) and image layout transitions are performed--images are transitioned from layouts specified for the last subpass to the value of a final layout (refer to the *Specifying attachment descriptions* recipe). Also multisample resolving is performed on color attachments for which resolving was specified in the last subpass. Additionally, for attachments whose contents should be preserved after the render pass, attachment data may be transferred from the cache to the image's memory.

# See also

The following recipes in this chapter:

*   *Specifying subpass descriptions*
*   *Creating a render pass*
*   *Beginning a render pass*
*   *Progressing to the next subpass*

# Destroying a framebuffer

When a framebuffer is no longer used by the pending commands and we don't need it anymore, we can destroy it.

# How to do it...

1.  Initialize a variable of type `VkDevice` named `logical_device` with the handle of a logical device on which the framebuffer was created.
2.  Take the framebuffer's handle and store it in a variable of type `VkFramebuffer` named `framebuffer`.
3.  Make the following call: `vkDestroyFramebuffer( logical_device, framebuffer, nullptr )`, for which we provide the `logical_device` and `framebuffer` variables and a `nullptr` value.
4.  For safety reasons, store a `VK_NULL_HANDLE` value in the `framebuffer` variable.

# How it works...

The framebuffer is destroyed with the `vkDestroyFramebuffer()` function call. However, before we can destroy it, we must make sure that commands referencing the given framebuffer are no longer executed on the hardware.

The following code destroys a framebuffer:

[PRE19]

# See also

The following recipe in this chapter:

*   *Creating a framebuffer*

# Destroying a render pass

If a render pass is not needed and it is not used anymore by commands submitted to the hardware, we can destroy it.

# How to do it...

1.  Use the handle of a logical device, on which the render pass was created, to initialize a variable of type `VkDevice` named `logical_device`.
2.  Store the handle of the render pass that should be destroyed in a variable of type `VkRenderPass` named `render_pass`.
3.  Call `vkDestroyRenderPass( logical_device, render_pass, nullptr )` and provide the `logical_device` and `render_pass` variables and a `nullptr` value.
4.  For safety reasons, assign a `VK_NULL_HANDLE` value to the `render_pass` variable.

# How it works...

Destroying a render pass is performed with just a single function call like this:

[PRE20]

# See also

The following recipe in this chapter:

*   *Creating a render pass*