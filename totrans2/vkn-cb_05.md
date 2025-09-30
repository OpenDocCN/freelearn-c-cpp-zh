# Descriptor Sets

In this chapter, we will cover the following recipes:

*   Creating a sampler
*   Creating a sampled image
*   Creating a combined image sampler
*   Creating a storage image
*   Creating a uniform texel buffer
*   Creating a storage texel buffer
*   Creating a uniform buffer
*   Creating a storage buffer
*   Creating an input attachment
*   Creating a descriptor set layout
*   Creating a descriptor pool
*   Allocating descriptor sets
*   Updating descriptor sets
*   Binding descriptor sets
*   Creating descriptors with a texture and a uniform buffer
*   Freeing descriptor sets
*   Resetting a descriptor pool
*   Destroying a descriptor pool
*   Destroying a descriptor set layout
*   Destroying a sampler

# Introduction

In modern computer graphics, most of the rendering and processing of image data (such as vertices, pixels, or fragments) is done with a programmable pipeline and shaders. Shaders, to operate properly and to generate appropriate results, need to access additional data sources such as textures, samplers, buffers, or uniform variables. In Vulkan, these are provided through sets of descriptors.

Descriptors are opaque data structures that represent shader resources. They are organized into groups or sets and their contents are specified by descriptor set layouts. To provide resources to shaders, we bind descriptor sets to pipelines. We can bind multiple sets at once. To access resources from within shaders, we need to specify from which set and from which location within a set (called a **binding**) the given resource is acquired.

In this chapter, we will learn about the various descriptor types. We will see how to prepare resources (samplers, buffers, and images) so they can be used inside shaders. We will also look at how to set up an interface between an application and shaders and use resources inside shaders.

# Creating a sampler

Samplers define a set of parameters that control how image data is loaded inside shaders (sampled). These parameters include address calculations (that is, wrapping or repeating), filtering (linear or nearest), or using mipmaps. To use samplers from within shaders, we first need to create them.

# How to do it...

1.  Take a handle of a logical device and store it in a variable of type `VkDevice` named `logical_device`.
2.  Create a variable of type `VkSamplerCreateInfo` named `sampler_create_info` and use the following values for its members:
    *   `VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO` value for `sType`
    *   `nullptr` value for `pNext`
    *   `0` value for `flags`
    *   The desired magnification and minification filtering mode (`VK_FILTER_NEAREST` or `VK_FILTER_LINEAR`) for `magFilter` and `minFilter`
    *   The selected mipmap filtering mode (`VK_SAMPLER_MIPMAP_MODE_NEAREST` or `VK_SAMPLER_MIPMAP_MODE_LINEAR`) for `mipmapMode`
    *   The selected image addressing mode for image U, V, and W coordinates outside of the `0.0 - 1.0` range (`VK_SAMPLER_ADDRESS_MODE_REPEAT``,` `VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT``,` `VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE``VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER`, or `VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE`) for `addressModeU`, `addressModeV` and `addressModeW`
    *   The desired value to be added to the mipmap level of detail calculations for `mipLodBias`
    *   `true` value if anisotropic filtering should be enabled or otherwise `false` for `anisotropyEnable`
    *   The maximal value of the anisotropy for `maxAnisotropy`
    *   `true` value if comparison against a reference value should be enabled during image lookups or, otherwise `false` for `compareEnable`
    *   The selected comparison function applied to the fetched data (`VK_COMPARE_OP_NEVER``,` `VK_COMPARE_OP_LESS``,` `VK_COMPARE_OP_EQUAL``,` `VK_COMPARE_OP_LESS_OR_EQUAL``,` `VK_COMPARE_OP_GREATER``,` `VK_COMPARE_OP_NOT_EQUAL``,` `VK_COMPARE_OP_GREATER_OR_EQUAL`, or `VK_COMPARE_OP_ALWAYS`) for `compareOp`
    *   The minimal and maximal values to clamp the calculated image's level of detail value (mipmap number) for `minLod` and `maxLod`
    *   One of the predefined border color values (`VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK``,` `VK_BORDER_COLOR_INT_TRANSPARENT_BLACK``,` `VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK``,` `VK_BORDER_COLOR_INT_OPAQUE_BLACK``,` `VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE`, or `VK_BORDER_COLOR_INT_OPAQUE_WHITE`) for `borderColor`
    *   The `true` value if addressing should be performed using the image's dimensions or `false` if addressing should use normalized coordinates (in the `0.0`-`1.0` range) for `unnormalizedCoordinates`

3.  Create a variable of type `VkSampler` named `sampler` in which the created sampler will be stored.
4.  Call `vkCreateSampler( logical_device, &sampler_create_info, nullptr, &sampler )` and provide the `logical_device` variable, a pointer to the `sampler_create_info` variable, a `nullptr` value, and a pointer to the `sampler` variable.
5.  Make sure the call was successful by checking whether the returned value was equal to `VK_SUCCESS`.

# How it works...

Samplers control the way images are read inside shaders. They can be used separately or combined with a sampled image.

Samplers are used for a `VK_DESCRIPTOR_TYPE_SAMPLER` descriptor type.

Sampling parameters are specified with a variable of type `VkSamplerCreateInfo` like this:

[PRE0]

This variable is then provided to the function that creates the sampler:

[PRE1]

To specify a sampler inside shaders, we need to create a uniform variable with a `sampler` keyword.

An example of a GLSL code that uses a sampler, from which SPIR-V assembly can be generated, may look like this:

[PRE2]

# See also

*   See the following recipe in this chapter:
    *   *Destroying a sampler*

# Creating a sampled image

Sampled images are used to read data from images (textures) inside shaders. Usually, they are used together with samplers. And to be able to use an image as a sampled image, it must be created with a `VK_IMAGE_USAGE_SAMPLED_BIT` usage.

# How to do it...

1.  Take the handle of a physical device stored in a variable of type `VkPhysicalDevice` named `physical_device`.
2.  Select a format that will be used for an image. Initialize a variable of type `VkFormat` named `format` with the selected image format.

3.  Create a variable of type `VkFormatProperties` named `format_properties`.
4.  Call `vkGetPhysicalDeviceFormatProperties( physical_device, format, &format_properties )`, for which to provide the `physical_device` variable, the `format` variable, and a pointer to the `format_properties` variable.
5.  Make sure the selected image format is suitable for a sampled image. Do that by checking whether the `VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT` bit of an `optimalTilingFeatures` member of the `format_properties` variable is set.
6.  If the sampled image will be linearly filtered or if its mipmaps will be linearly filtered, make sure the selected format is suitable for a linearly filtered sampled image. Do that by checking whether the `VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT` bit of an `optimalTilingFeatures` member of the `format_properties` variable is set.
7.  Take the handle of the logical device created from the handle stored in the `physical_device` variable and use it to initialize a variable of type `VkDevice` named `logical_device`.
8.  Create an image using the `logical_device` and `format` variables and choose the rest of the image parameters. Don't forget to provide a `VK_IMAGE_USAGE_SAMPLED_BIT` usage during the image creation. Store the image's handle in a variable of type `VkImage` named `sampled_image` (refer to the *Creating an image* recipe from [Chapter 4](f1332ca0-b5a2-49bd-ac41-e37068e31042.xhtml), *Resources and Memory*).
9.  Allocate a memory object with a `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT` property (or use a range of an existing memory object) and bind it to the created image (refer to the *Allocating and binding memory object to an image* recipe from [Chapter 4](f1332ca0-b5a2-49bd-ac41-e37068e31042.xhtml), *Resources and Memory*).
10.  Create an image view using the `logical_device`, `sampled_image`, and `format` variables and select the rest of the view parameters. Store the image view's handle in a variable of type `VkImageView` named `sampled_image_view` (refer to the *Creating an image view* recipe from [Chapter 4](f1332ca0-b5a2-49bd-ac41-e37068e31042.xhtml), *Resources and Memory*).

# How it works...

Sampled images are used as a source of image data (textures) inside shaders. To fetch data from the image, usually we need a sampler object, which defines how the data should be read (refer to the *Creating a sampler* recipe).

Sampled images are used for a `VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE` descriptor type.

Inside shaders, we can use multiple samplers to read data from the same image in a different way. We can also use the same sampler with multiple images. But on some platforms, it may be more optimal to use combined image sampler objects, which gather a sampler and a sampled image in one object.

Not all image formats are supported for sampled images; this depends on the platform on which the application is executed. But there is a set of mandatory formats that can always be used for sampled images and linearly filtered sampled images. Examples of such formats include (but are not limited to) the following:

*   `VK_FORMAT_B4G4R4A4_UNORM_PACK16`
*   `VK_FORMAT_R5G6B5_UNORM_PACK16`
*   `VK_FORMAT_A1R5G5B5_UNORM_PACK16`
*   `VK_FORMAT_R8_UNORM` and `VK_FORMAT_R8_SNORM`
*   `VK_FORMAT_R8G8_UNORM` and `VK_FORMAT_R8G8_SNORM`
*   `VK_FORMAT_R8G8B8A8_UNORM`, `VK_FORMAT_R8G8B8A8_SNORM`, and `VK_FORMAT_R8G8B8A8_SRGB`
*   `VK_FORMAT_B8G8R8A8_UNORM` and `VK_FORMAT_B8G8R8A8_SRGB`
*   `VK_FORMAT_A8B8G8R8_UNORM_PACK32`, `VK_FORMAT_A8B8G8R8_SNORM_PACK32`, and `VK_FORMAT_A8B8G8R8_SRGB_PACK32`
*   `VK_FORMAT_A2B10G10R10_UNORM_PACK32`
*   `VK_FORMAT_R16_SFLOAT`
*   `VK_FORMAT_R16G16_SFLOAT`
*   `VK_FORMAT_R16G16B16A16_SFLOAT`
*   `VK_FORMAT_B10G11R11_UFLOAT_PACK32`
*   `VK_FORMAT_E5B9G9R9_UFLOAT_PACK32`

If we want to use some less typical format, we need to check whether it can be used for sampled images. This can be done like this:

[PRE3]

If we are sure the selected format is suitable for our needs, we can create an image, a memory object for it, and an image view (in Vulkan, images are represented with image views most of the time). We need to specify a `VK_IMAGE_USAGE_SAMPLED_BIT` usage during the image creation:

[PRE4]

When we want to use an image as a sampled image, before we load data from it inside shaders, we need to transition the image's layout to `VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL`.

To create a uniform variable that represents a sampled image inside shaders, we need to use a `texture` keyword (possibly with a prefix) with an appropriate dimensionality.

An example of a GLSL code from which SPIR-V assembly can be generated, that uses a sampled image, may look like this:

[PRE5]

# See also

*   In [Chapter 4](f1332ca0-b5a2-49bd-ac41-e37068e31042.xhtml), *Resources and Memory*, see the following recipes:
    *   *Creating an image*
    *   *Allocating and binding memory object to an image*
    *   *Creating an image view*
    *   *Destroying an image view*
    *   *Destroying an image*
    *   *Freeing a memory object*
*   In this chapter, see the following recipe:
    *   *Creating a sampler*

# Creating a combined image sampler

From the application (API) perspective, samplers and sampled images are always separate objects. But inside shaders, they can be combined into one object. On some platforms, sampling from combined image samplers inside shaders may be more optimal than using separate samplers and sampled images.

# How to do it...

1.  Create a sampler object and store its handle in a variable of type `VkSampler` named `sampler` (refer to the *Creating a sampler* recipe).
2.  Create a sampled image. Store the handle of the created image in a variable of type `VkImage` named `sampled_image`. Create an appropriate view for the sampled image and store its handle in a variable of type `VkImageView` named `sampled_image_view` (refer to the *Creating a sampled image* recipe).

# How it works...

Combined image samplers are created in our application in the same way as normal samplers and sampled images. They are just used differently inside shaders.

Combined image samplers can be bound to descriptors of a `VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER` type.

The following code uses the *Creating a sampler* and *Creating a sampled image* recipes to create necessary objects:

[PRE6]

The difference is inside the shaders.

To create a variable that represents a combined image sampler inside GLSL shaders, we need to use a `sampler` keyword (possibly with a prefix) with an appropriate dimensionality.

Don't confuse samplers and combined image samplers--both use a `sampler` keyword inside shaders, but combined image samplers additionally have a dimensionality specified like in the following example:

[PRE7]

Combined image samplers deserve a separate treatment, because applications that use them may have a better performance on some platforms. So if there is no specific reason to use separate samplers and sampled images, we should try to combine them into single objects.

# See also

In [Chapter 4](f1332ca0-b5a2-49bd-ac41-e37068e31042.xhtml), *Resources and Memory*, see the following recipes:

*   *Creating an image*
*   *Allocating and binding memory object to an image*
*   *Creating an image view*
*   *Destroying an image view*
*   *Destroying an image*
*   *Freeing a memory object*

See the following recipes in this chapter:

*   *Creating a sampler*
*   *Creating a sampled image*
*   *Destroying a sampler*

# Creating a storage image

Storage images allow us to load (unfiltered) data from images bound to pipelines. But, what's more important, they also allow us to store data from shaders in the images. Such images must be created with a `VK_IMAGE_USAGE_STORAGE_BIT` usage flag specified.

# How to do it...

1.  Take the handle of a physical device and store it in a variable of type `VkPhysicalDevice` named `physical_device`.
2.  Select a format that will be used for a storage image. Initialize a variable of type `VkFormat` named `format` with the selected format.

3.  Create a variable of type `VkFormatProperties` named `format_properties`.
4.  Call `vkGetPhysicalDeviceFormatProperties( physical_device, format, &format_properties )` and provide the `physical_device` variable, the `format` variable, and a pointer to the `format_properties` variable.
5.  Check whether the selected image format is suitable for a storage image. Do that by checking whether the `VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT` bit of an `optimalTilingFeatures` member of the `format_properties` variable is set.
6.  If atomic operations will be performed on the storage image, make sure the selected format supports them. Do that by checking whether the `VK_FORMAT_FEATURE_STORAGE_IMAGE_ATOMIC_BIT` bit of an `optimalTilingFeatures` member of the `format_properties` variable is set.
7.  Take the handle of the logical device created from a `physical_device` and use it to initialize a variable of type `VkDevice` named `logical_device`.
8.  Create an image using the `logical_device` and `format` variables and choose the rest of the image parameters. Make sure the `VK_IMAGE_USAGE_STORAGE_BIT` usage is specified during the image creation. Store the created handle in a variable of type `VkImage` named `storage_image` (refer to the *Creating an image* recipe from [Chapter 4](f1332ca0-b5a2-49bd-ac41-e37068e31042.xhtml), *Resources and Memory*).
9.  Allocate a memory object with a `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT` property (or use a range of an existing memory object) and bind it to the image (refer to the *Allocating and binding memory object to an image* recipe from [Chapter 4](f1332ca0-b5a2-49bd-ac41-e37068e31042.xhtml), *Resources and Memory*).
10.  Create an image view using the `logical_device`, `storage_image`, and `format` variables and select the rest of the view parameters. Store the image view's handle in a variable of type `VkImageView` named `storage_image_view` (refer to the *Creating an image view* recipe from [Chapter 4](f1332ca0-b5a2-49bd-ac41-e37068e31042.xhtml), *Resources and Memory*).

# How it works...

When we want to store data in images from within shaders, we need to use storage images. We can also load data from such images, but these loads are unfiltered (we can't use samplers for storage images).

Storage images correspond to descriptors of a `VK_DESCRIPTOR_TYPE_STORAGE_IMAGE` type.

Storage images are created with a `VK_IMAGE_USAGE_STORAGE_BIT` usage. We also can't forget about specifying a proper format. Not all formats may always be used for storage images. This depends on the platform our application is executed on. But there is a list of mandatory formats that all Vulkan drivers must support. It includes (but is not limited to) the following formats:

*   `VK_FORMAT_R8G8B8A8_UNORM`, `VK_FORMAT_R8G8B8A8_SNORM`, `VK_FORMAT_R8G8B8A8_UINT`, and `VK_FORMAT_R8G8B8A8_SINT`
*   `VK_FORMAT_R16G16B16A16_UINT`, `VK_FORMAT_R16G16B16A16_SINT` and `VK_FORMAT_R16G16B16A16_SFLOAT`
*   `VK_FORMAT_R32_UINT`, `VK_FORMAT_R32_SINT` and `VK_FORMAT_R32_SFLOAT`
*   `VK_FORMAT_R32G32_UINT`, `VK_FORMAT_R32G32_SINT` and `VK_FORMAT_R32G32_SFLOAT`
*   `VK_FORMAT_R32G32B32A32_UINT`, `VK_FORMAT_R32G32B32A32_SINT` and `VK_FORMAT_R32G32B32A32_SFLOAT`

If we want to perform atomic operations on storage images, the list of mandatory formats is much shorter and includes only the following ones:

*   `VK_FORMAT_R32_UINT`
*   `VK_FORMAT_R32_SINT`

If another format is required for storage images or if we need to use another format to perform atomic operations on storage images, we must check whether the selected format is supported on a platform our application is executed on. This can be done with the following code:

[PRE8]

If the format is supported, we create images as usual, but we need to specify a `VK_IMAGE_USAGE_STORAGE_BIT` usage. After the image is ready, we need to create a memory object, bind it to the image, and we also need an image view. These operations can be performed like this:

[PRE9]

Before we can load or store data in storage images from shaders, we must perform a transition to a `VK_IMAGE_LAYOUT_GENERAL` layout. It is the only layout in which these operations are supported.

Inside GLSL shaders, storage images are specified with an `image` keyword (possibly with a prefix) and an appropriate dimensionality. We also need to provide the image's format inside the `layout` qualifier.

An example of a storage image's definition in a GLSL shader is provided as follows:

[PRE10]

# See also

In [Chapter 4](f1332ca0-b5a2-49bd-ac41-e37068e31042.xhtml), *Resources and Memory*, see the following recipes:

*   *Creating an image*
*   *Allocating and binding memory object to an image*
*   *Creating an image view*
*   *Destroying an image view*
*   *Destroying an image*
*   *Freeing a memory object*

# Creating a uniform texel buffer

Uniform texel buffers allow us to read data in a way similar to reading data from images--their contents are interpreted not as an array of single (scalar) values but as formatted pixels (texel) with one, two, three, or four components. But through such buffers, we can access data that is much larger than the data provided through usual images.

We need to specify a `VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT` usage when we want to use a buffer as a uniform texel buffer.

# How to do it...

1.  Take the handle of a physical device and store it in a variable of type `VkPhysicalDevice` named `physical_device`.
2.  Select a format in which the buffer data will be stored. Use the format to initialize a variable of type `VkFormat` named `format`.
3.  Create a variable of type `VkFormatProperties` named `format_properties`.
4.  Call `vkGetPhysicalDeviceFormatProperties( physical_device, format, &format_properties )` and provide the handle of the physical device, the `format` variable, and a pointer to the `format_properties` variable.
5.  Make sure the selected format is suitable for a uniform texel buffer by checking whether the `bufferFeatures` member of the `format_properties` variable has a `VK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT` bit set.
6.  Take the handle of a logical device created from the handle of the selected physical device. Store it in a variable of type `VkDevice` named `logical_device`.
7.  Create a variable of type `VkBuffer` named `uniform_texel_buffer`.
8.  Create a buffer, using the `logical_device` variable, with a desired size and usage. Don't forget to include a `VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT` usage during the buffer's creation. Store the created handle in the `uniform_texel_buffer` variable (refer to the *Creating a buffer* recipe from [Chapter 4](f1332ca0-b5a2-49bd-ac41-e37068e31042.xhtml), *Resources and Memory*).

9.  Allocate a memory object with a `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT` property (or use an existing one) and bind it to the buffer. If the new memory object is allocated, store it in a variable of type `VkDeviceMemory` named `memory_object` (refer to the *Allocating and binding memory object to a buffer* recipe from [Chapter 4](f1332ca0-b5a2-49bd-ac41-e37068e31042.xhtml), *Resources and Memory*).
10.  Create a buffer view using the `logical_device`, `uniform_texel_buffer`, and `format` variables, and the desired offset and memory range. Store the resulting handle in a variable of type `VkBufferView` named `uniform_texel_buffer_view` (refer to the *Creating a buffer view* recipe from [Chapter 4](f1332ca0-b5a2-49bd-ac41-e37068e31042.xhtml), *Resources and Memory*).

# How it works...

Uniform texel buffers allow us to provide data interpreted as one-dimensional images. But this data may be much larger than typical images. Vulkan specification requires every driver to support 1D images of at least 4,096 texels. But for texel buffers, this minimal required limit goes up to 65,536 elements.

Uniform texel buffers are bound to descriptors of a `VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER` type.

Uniform texel buffers are created with a `VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT` usage. But apart from that, we need to select an appropriate format. Not all formats are compatible with such buffers. The list of mandatory formats that can be used with uniform texel buffers includes (but is not limited to) the following ones:

*   `VK_FORMAT_R8_UNORM`, `VK_FORMAT_R8_SNORM`, `VK_FORMAT_R8_UINT`, and `VK_FORMAT_R8_SINT`
*   `VK_FORMAT_R8G8_UNORM`, `VK_FORMAT_R8G8_SNORM`, `VK_FORMAT_R8G8_UINT`, and `VK_FORMAT_R8G8_SINT`
*   `VK_FORMAT_R8G8B8A8_UNORM`, `VK_FORMAT_R8G8B8A8_SNORM`, `VK_FORMAT_R8G8B8A8_UINT`, and `VK_FORMAT_R8G8B8A8_SINT`
*   `VK_FORMAT_B8G8R8A8_UNORM`
*   `VK_FORMAT_A8B8G8R8_UNORM_PACK32`, `VK_FORMAT_A8B8G8R8_SNORM_PACK32`, `VK_FORMAT_A8B8G8R8_UINT_PACK32`, and `VK_FORMAT_A8B8G8R8_SINT_PACK32`
*   `VK_FORMAT_A2B10G10R10_UNORM_PACK32` and `VK_FORMAT_A2B10G10R10_UINT_PACK32`
*   `VK_FORMAT_R16_UINT`, `VK_FORMAT_R16_SINT` and `VK_FORMAT_R16_SFLOAT`
*   `VK_FORMAT_R16G16_UINT`, `VK_FORMAT_R16G16_SINT` and `VK_FORMAT_R16G16_SFLOAT`
*   `VK_FORMAT_R16G16B16A16_UINT`, `VK_FORMAT_R16G16B16A16_SINT` and `VK_FORMAT_R16G16B16A16_SFLOAT`
*   `VK_FORMAT_R32_UINT`, `VK_FORMAT_R32_SINT` and `VK_FORMAT_R32_SFLOAT`
*   `VK_FORMAT_R32G32_UINT`, `VK_FORMAT_R32G32_SINT` and `VK_FORMAT_R32G32_SFLOAT`
*   `VK_FORMAT_R32G32B32A32_UINT`, `VK_FORMAT_R32G32B32A32_SINT` and `VK_FORMAT_R32G32B32A32_SFLOAT`
*   `VK_FORMAT_B10G11R11_UFLOAT_PACK32`

To check whether other formats can be used with uniform texel buffers, we need to prepare the following code:

[PRE11]

If the selected format is suitable for our needs, we can create a buffer, allocate a memory object for it, and bind it to the buffer. What is very important, is that we also need to create a buffer view:

[PRE12]

From the API perspective, the structure of the buffer's contents doesn't matter. But in the case of uniform texel buffers, we need to specify a data format which will allow shaders to interpret the buffer's contents in an appropriate way. That's why a buffer view is required.

In the GLSL shaders, uniform texel buffers are defined through variables of type `samplerBuffer` (possibly with a prefix).

An example of a uniform texel buffer variable defined in a GLSL shader is provided as follows:

[PRE13]

# See also

In [Chapter 4](f1332ca0-b5a2-49bd-ac41-e37068e31042.xhtml), *Resources and Memory*, see the following recipes:

*   *Creating a buffer*
*   *Allocating and binding memory object to a buffer*
*   *Creating a buffer view*
*   *Destroying a buffer view*
*   *Freeing a memory object*
*   *Destroying a buffer*

# Creating a storage texel buffer

Storage texel buffers, like uniform texel buffers, are a way to provide large amount of image-like data to shaders. But they also allow us to store data in them and perform atomic operations on them. For this purpose, we need to create a buffer with a `VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT`.

# How to do it...

1.  Take the handle of a physical device. Store it in a variable of type `VkPhysicalDevice` named `physical_device`.
2.  Select a format for the texel buffer's data and use it to initialize a variable of type `VkFormat` named `format`.
3.  Create a variable of type `VkFormatProperties` named `format_properties`.
4.  Call `vkGetPhysicalDeviceFormatProperties( physical_device, format, &format_properties )` and provide the handle of the selected physical device, the `format` variable, and a pointer to the `format_properties` variable.
5.  Make sure the selected format is suitable for a storage texel buffer by checking whether the `bufferFeatures` member of the `format_properties` variable has a `VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_BIT` bit set.
6.  If atomic operations will be performed on a created storage texel buffer, make sure the selected format is also suitable for atomic operations. For this purpose, check whether a `VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_ATOMIC_BIT` bit of the `bufferFeatures` member of the `format_properties` variable is set.
7.  Take the handle of a logical device created from the handle of the selected physical device. Store it in a variable of type `VkDevice` named `logical_device`.
8.  Create a variable of type `VkBuffer` named `storage_texel_buffer`.
9.  Using the `logical_device` variable, create a buffer with the chosen size and usage. Make sure a `VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT` usage is specified during the buffer's creation. Store the buffer's handle in the `storage_texel_buffer` variable (refer to the *Creating a buffer* recipe from [Chapter 4](f1332ca0-b5a2-49bd-ac41-e37068e31042.xhtml), *Resources and Memory*).
10.  Allocate a memory object with a `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT` property (or use an existing one) and bind it to the buffer. If a new memory object is allocated, store it in a variable of type `VkDeviceMemory` named `memory_object` (refer to the *Allocating and binding memory object to a buffer* recipe from [Chapter 4](f1332ca0-b5a2-49bd-ac41-e37068e31042.xhtml), *Resources and Memory*).
11.  Create a buffer view using the `logical_device`, `storage_texel_buffer`, and `format` variables, and the desired offset and memory range. Store the resulting handle in a variable of type `VkBufferView` named `storage_texel_buffer_view` (refer to the *Creating a buffer view* recipe from [Chapter 4](f1332ca0-b5a2-49bd-ac41-e37068e31042.xhtml), *Resources and Memory*).

# How it works...

Storage texel buffers allow us to access and to store data in very large arrays. Data is interpreted as if it was read or stored inside one-dimensional images. Additionally, we can perform atomic operations on such buffers.

Storage texel buffers can fill descriptors of a type equal to `VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER`.

To use a buffer as a storage texel buffer, it needs to be created with a `VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT` usage. A buffer view with an appropriate format is also required. For storage texel buffers, we can select one of the mandatory formats that include the following ones:

*   `VK_FORMAT_R8G8B8A8_UNORM`, `VK_FORMAT_R8G8B8A8_SNORM`, `VK_FORMAT_R8G8B8A8_UINT`, and `VK_FORMAT_R8G8B8A8_SINT`
*   `VK_FORMAT_A8B8G8R8_UNORM_PACK32`, `VK_FORMAT_A8B8G8R8_SNORM_PACK32`, `VK_FORMAT_A8B8G8R8_UINT_PACK32`, and `VK_FORMAT_A8B8G8R8_SINT_PACK32`
*   `VK_FORMAT_R32_UINT`, `VK_FORMAT_R32_SINT`, and `VK_FORMAT_R32_SFLOAT`
*   `VK_FORMAT_R32G32_UINT`, `VK_FORMAT_R32G32_SINT`, and `VK_FORMAT_R32G32_SFLOAT`
*   `VK_FORMAT_R32G32B32A32_UINT`, `VK_FORMAT_R32G32B32A32_SINT`, and `VK_FORMAT_R32G32B32A32_SFLOAT`

For atomic operations, the list of mandatory formats is much shorter and includes only the following:

*   `VK_FORMAT_R32_UINT` and `VK_FORMAT_R32_SINT`

Other formats may also be supported for storage texel buffers, but the support is not guaranteed and must be confirmed on the platform our application is executed on like this:

[PRE14]

For a storage texel buffer, we need to create a buffer, allocate and bind a memory object to the buffer, and also we need to create a buffer view that will define the format of the buffer's data:

![](img/image_05_001.png)

[PRE15]

We can also use an existing memory object and bind a range of its memory to the storage texel buffer.

From the GLSL perspective, storage texel buffer variables are defined using an `imageBuffer` (possibly with a prefix) keyword.

An example of a storage texel buffer defined in a GLSL shader looks like this:

[PRE16]

# See also

In [Chapter 4](f1332ca0-b5a2-49bd-ac41-e37068e31042.xhtml), *Resources and Memory*, see the following recipes:

*   *Creating a buffer*
*   *Allocating and binding memory object to a buffer*
*   *Creating a buffer view*
*   *Destroying a buffer view*
*   *Freeing a memory object*
*   *Destroying a buffer*

# Creating a uniform buffer

In Vulkan, uniform variables used inside shaders cannot be placed in a global namespace. They can be defined only inside uniform buffers. For these, we need to create buffers with a `VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT` usage.

# How to do it...

1.  Take the created logical device and use its handle to initialize a variable of type `VkDevice` named `logical_device`.
2.  Create a variable of type `VkBuffer` named `uniform_buffer`. It will hold the handle of the created buffer.

3.  Create a buffer using a `logical_device` variable and specifying the desired size and usage. The latter must contain at least a `VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT` flag. Store the handle of the buffer in the `uniform_buffer` variable (refer to the *Creating a buffer* recipe from [Chapter 4](f1332ca0-b5a2-49bd-ac41-e37068e31042.xhtml), *Resources and Memory*).
4.  Allocate a memory object with a `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT` property (or use a range of an existing memory object) and bind it to the buffer (refer to the *Allocating and binding memory object to a buffer* recipe from [Chapter 4](f1332ca0-b5a2-49bd-ac41-e37068e31042.xhtml), *Resources and Memory*).

# How it works...

Uniform buffers are used to provide values for read-only uniform variables inside shaders.

Uniform buffers can be used for `VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER` or `VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC` descriptor types.

Typically, uniform buffers contain data for parameters that don't change too often, that is, matrices (for small amounts of data, **push constants** are recommended as updating them is usually much faster; information about push constants can be found in the *Providing data to shaders through push constants* recipe in [Chapter 9](0a69f5b5-142e-422b-aa66-5cb09a6467b3.xhtml), *Command Recording and Drawing*).

Creating a buffer in which data for uniform variables will be stored requires us to specify a `VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT` flag during buffer creation. When the buffer is created, we need to prepare a memory object and bind it to the created buffer (we can also use an existing memory object and bind the part of its memory store to the buffer):

[PRE17]

After the buffer and its memory objects are ready, we can upload data to them as we do with any other kinds of buffers. We just need to remember that uniform variables must be placed at appropriate offsets. These offsets are the same as in the std140 layout from the GLSL language and are defined as follows:

*   A scalar variable of size `N` must be placed at offsets that are a multiple of `N`
*   A vector with two components, where each component has a size of `N`, must be placed at offsets that are a multiple of `2N`
*   A vector with three or four components, where each component has a size of `N`, must be placed at offsets that are a multiple of `4N`
*   An array with elements of size `N` must be placed at offsets that are a multiple of `N` rounded up to the multiple of `16`
*   A structure must be placed at an offset that is the same as the biggest offset of its members, rounded up to a multiple of `16` (offset of a member with the biggest offset requirement, rounded up to the multiple of `16`)
*   A row-major matrix must be placed at an offset equal to the offset of a vector with the number of components equal to the number of columns in the matrix
*   A column-major matrix must be placed at the same offsets as its columns

Dynamic uniform buffers differ from normal uniform buffers in the way their address is specified. During a descriptor set update, we specify the size of memory that should be used for a uniform buffer and an offset from the beginning of the buffer's memory. For normal uniform buffers, these parameters remain unchanged. For dynamic uniform buffers, the specified offset becomes a base offset that can be later modified by the dynamic offset which is added when a descriptor set is bound to a command buffer.

Inside GLSL shaders, both uniform buffers and dynamic uniform buffers are defined with a `uniform` qualifier and a block syntax.

An example of a uniform buffer's definition in a GLSL shader is provided as follows:

[PRE18]

# See also

In [Chapter 4](f1332ca0-b5a2-49bd-ac41-e37068e31042.xhtml), *Resources and Memory*, see the following recipes:

*   *Creating a buffer*
*   *Allocating and binding memory object to a buffer*
*   *Freeing a memory object*
*   *Destroying a buffer*

# Creating a storage buffer

When we want to not only read data from a buffer inside shaders, but we would also like to store data in it, we need to use storage buffers. These are created with a `VK_BUFFER_USAGE_STORAGE_BUFFER_BIT` usage.

# How to do it...

1.  Take the handle of a logical device and store it in a variable of type `VkPhysicalDevice` named `physical_device`.
2.  Create a variable of type `VkBuffer` named `storage_buffer` in which a handle of a created buffer will be stored.
3.  Create a buffer of a desired size and usage using the `logical_device` variable. Specified usage must contain at least a `VK_BUFFER_USAGE_STORAGE_BUFFER_BIT` flag. Store the created handle in the `storage_buffer` variable (refer to the *Creating a buffer* recipe from [Chapter 4](f1332ca0-b5a2-49bd-ac41-e37068e31042.xhtml), *Resources and Memory*).
4.  Allocate a memory object with a `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT` property (or use a range of an existing memory object) and bind it to the created buffer (refer to the *Allocating and binding memory object to a buffer* recipe from [Chapter 4](f1332ca0-b5a2-49bd-ac41-e37068e31042.xhtml), *Resources and Memory*).

# How it works...

Storage buffers support both read and write operations. We can also perform atomic operations on storage buffers' members which have unsigned integer formats.

Storage buffers correspond to `VK_DESCRIPTOR_TYPE_STORAGE_BUFFER` or `VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC` descriptor types.

Data for members of storage buffers must be placed at appropriate offsets. The easiest way to fulfill the requirements is to follow the rules for the std430 layout in the GLSL language. Base alignment rules for storage buffers are similar to the rules of uniform buffers with the exception of arrays and structures--their offsets don't need to be rounded up to a multiple of 16\. For convenience, these rules are specified as follows:

*   A scalar variable of size `N` must be placed at offsets that are a multiple of `N`
*   A vector with two components, where each component has a size of `N`, must be placed at offsets that are a multiple of `2N`
*   A vector with three or four components, where each component has a size of `N`, must be placed at offsets that are a multiple of `4N`
*   An array with elements of size `N` must be placed at offsets that are a multiple of `N`
*   A structure must be placed at offsets that are a multiple of the biggest offset of any of its members (a member with the biggest offset requirement)
*   A row-major matrix must be placed at an offset equal to the offset of a vector with the number of components equal to the number of columns in the matrix
*   A column-major matrix must be placed at the same offsets as its columns

Dynamic storage buffers differ in the way their base memory offset is defined. The offset and range specified during descriptor set updates remain unchanged for the normal storage buffers until the next update. In the case of their dynamic variations, the specified offset becomes a base address which is later modified by the dynamic offset specified when a descriptor set is bound to a command buffer.

In GLSL shaders, storage buffers and dynamic storage buffers are defined identically with a `buffer` qualifier and a block syntax.

An example of a storage buffer used in a GLSL shader is provided as follows:

[PRE19]

# See also

In [Chapter 4](f1332ca0-b5a2-49bd-ac41-e37068e31042.xhtml), *Resources and Memory*, see the following recipes:

*   *Creating a buffer*
*   *Allocating and binding memory object to a buffer*
*   *Freeing a memory object*
*   *Destroying a buffer*

# Creating an input attachment

Attachments are images into which we render during drawing commands, inside render passes. In other words, they are render targets.

Input attachments are image resources from which we can read (unfiltered) data inside fragment shaders. We just need to remember that we can access only one location corresponding to a processed fragment.

Usually, for input attachments, resources that were previously color or depth/stencil attachments are used. But we can also use other images (and their image views). We just need to create them with a `VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT` usage.

# How to do it...

1.  Take the physical device on which operations are performed and store its handle in a variable of type `VkPhysicalDevice` named `physical_device`.
2.  Select a format for an image and use it to initialize a variable of type `VkFormat` named `format`.
3.  Create a variable of type `VkFormatProperties` named `format_properties`.
4.  Call `vkGetPhysicalDeviceFormatProperties( physical_device, format, &format_properties )` and provide the `physical_device` and `format` variables, and a pointer to the `format_properties` variable.
5.  If the image's color data will be read, make sure the selected format is suitable for such usage. For this, check whether a `VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT` bit is set in an `optimalTilingFeatures` member of the `format_properties` variable.

6.  If the image's depth or stencil data will be read, check whether the selected format can be used for reading depth or stencil data. Do that by making sure that a `VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT` bit is set in an `optimalTilingFeatures` member of the `format_properties` variable.
7.  Take the handle of a logical device created from the used physical device. Store it in a variable of type `VkDevice` named `logical_device`.
8.  Create an image using the `logical_device` and `format` variables, and select appropriate values for the rest of the image's parameters. Make sure the `VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT` usage is specified during the image creation. Store the created handle in a variable of type `VkImage` named `input_attachment` (refer to the *Creating an image* recipe from [Chapter 4](f1332ca0-b5a2-49bd-ac41-e37068e31042.xhtml), *Resources and Memory*).
9.  Allocate a memory object with a `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT` property (or use a range of an existing memory object) and bind it to the image (refer to the *Allocating and binding memory object to an image* recipe from Chapter 4, *Resources and Memory*).
10.  Create an image view using the `logical_device`, `input_attachment`, and `format` variables, and choose the rest of the image view's parameters. Store the created handle in a variable of type `VkImageView` named `input_attachment_image_view` (refer to the *Creating an image view* recipe from [Chapter 4](f1332ca0-b5a2-49bd-ac41-e37068e31042.xhtml), *Resources and Memory*).

# How it works...

Input attachments allow us to read data inside fragment shaders from images used as render pass attachments (typically, for input attachments, images that were previously color or depth stencil attachments will be used).

Input attachments are used for descriptors of a `VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT` type.

In Vulkan, rendering operations are gathered into render passes. Each render pass has at least one subpass, but can have more. If we render to an attachment in one subpass, we can then use it as an input attachment and read data from it in subsequent subpasses of the same render pass. It is in fact the only way to read data from attachments in a given render pass--images serving as attachments in a given render pass can only be accessed through input attachments inside shaders (they cannot be bound to descriptor sets for purposes other than input attachments).

When reading data from input attachments, we are confined only to the location corresponding to the location of a processed fragment. But such an approach may be more optimal than rendering into an attachment, ending a render pass, binding an image to a descriptor set as a sampled image (texture), and starting another render pass which doesn't use the given image as any of its attachments.

For input attachments, we can also use other images (we don't have to use them as color or depth/stencil attachments). We just need to create them with a `VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT` usage and a proper format. The following formats are mandatory for input attachments from which color data will be read:

*   `VK_FORMAT_R5G6B5_UNORM_PACK16`
*   `VK_FORMAT_A1R5G5B5_UNORM_PACK16`
*   `VK_FORMAT_R8_UNORM`, `VK_FORMAT_R8_UINT` and `VK_FORMAT_R8_SINT`
*   `VK_FORMAT_R8G8_UNORM`, `VK_FORMAT_R8G8_UINT`, and `VK_FORMAT_R8G8_SINT`
*   `VK_FORMAT_R8G8B8A8_UNORM`, `VK_FORMAT_R8G8B8A8_UINT`, `VK_FORMAT_R8G8B8A8_SINT`, and `VK_FORMAT_R8G8B8A8_SRGB`
*   `VK_FORMAT_B8G8R8A8_UNORM` and `VK_FORMAT_B8G8R8A8_SRGB`
*   `VK_FORMAT_A8B8G8R8_UNORM_PACK32`, `VK_FORMAT_A8B8G8R8_UINT_PACK32`, `VK_FORMAT_A8B8G8R8_SINT_PACK32`, and `VK_FORMAT_A8B8G8R8_SRGB_PACK32`
*   `VK_FORMAT_A2B10G10R10_UNORM_PACK32` and `VK_FORMAT_A2B10G10R10_UINT_PACK32`
*   `VK_FORMAT_R16_UINT`, `VK_FORMAT_R16_SINT` and `VK_FORMAT_R16_SFLOAT`
*   `VK_FORMAT_R16G16_UINT`, `VK_FORMAT_R16G16_SINT` and `VK_FORMAT_R16G16_SFLOAT`
*   `VK_FORMAT_R16G16B16A16_UINT`, `VK_FORMAT_R16G16B16A16_SINT`, and `VK_FORMAT_R16G16B16A16_SFLOAT`
*   `VK_FORMAT_R32_UINT`, `VK_FORMAT_R32_SINT`, and `VK_FORMAT_R32_SFLOAT`
*   `VK_FORMAT_R32G32_UINT`, `VK_FORMAT_R32G32_SINT`, and `VK_FORMAT_R32G32_SFLOAT`
*   `VK_FORMAT_R32G32B32A32_UINT`, `VK_FORMAT_R32G32B32A32_SINT`, and `VK_FORMAT_R32G32B32A32_SFLOAT`

For input attachments from which depth/stencil data will be read, the following formats are mandatory:

*   `VK_FORMAT_D16_UNORM`
*   `VK_FORMAT_X8_D24_UNORM_PACK32` or `VK_FORMAT_D32_SFLOAT` (at least one of these two formats must be supported)
*   `VK_FORMAT_D24_UNORM_S8_UINT` or `VK_FORMAT_D32_SFLOAT_S8_UINT` (at least one of these two formats must be supported)

Other formats may also be supported but support for them is not guaranteed. We can check whether a given format is supported on the platform on which our application is executed like this:

[PRE20]

Next, we just need to create an image, allocate a memory object (or use an existing one) and bind it to the image, and create an image view. We can do it like this:

[PRE21]

Images and their views that are created like this can be used as input attachments. For this, we need to prepare a proper description of a render pass, and include the image views in framebuffers (refer to the *Specifying subpass descriptions* and *Creating a framebuffer* recipes from [Chapter 6](2de4339d-8912-440a-89a6-fd1f84961448.xhtml), *Render Passes and Framebuffers*).

Inside the GLSL shader code, variables that refer to input attachments are defined with a `subpassInput` (possibly with a prefix) keyword.

An example of an input attachment defined in a GLSL is provided as follows:

[PRE22]

# See also

In [Chapter 4](f1332ca0-b5a2-49bd-ac41-e37068e31042.xhtml), *Resources and Memory*, see the following recipes:

*   *Creating an image*
*   *Allocating and binding memory object to an image*
*   *Creating an image view*
*   *Destroying an image view*
*   *Destroying an image*
*   *Freeing a memory object*

In [Chapter 6](2de4339d-8912-440a-89a6-fd1f84961448.xhtml), *Render Passes and **Framebuffers*, see the following recipes:

*   *Specifying subpass descriptions*
*   *Creating a framebuffer*

# Creating a descriptor set layout

Descriptor sets gather many resources (descriptors) in one object. They are later bound to a pipeline to establish an interface between our application and the shaders. But for the hardware to know what resources are grouped in a set, how many resources of each type there are, and what their order is, we need to create a descriptor set layout.

# How to do it...

1.  Take the handle of a logical device and assign it to a variable of type `VkDevice` named `logical_device`.
2.  Create a vector variable with elements of type `VkDescriptorSetLayoutBinding` and call it `bindings`.
3.  For each resource you want to create and assign later to a given descriptor set, add an element to the `bindings` vector. Use the following values for members of each new element:
    *   The selected index of the given resource within a descriptor set for `binding`.
    *   Desired type of a given resource for `descriptorType`
    *   The number of resources of a specified type accessed through an array inside the shader (or 1 if the given resource is not accessed through an array) for `descriptorCount`
    *   The logical `OR` of all shader stages in which the resource will be accessed for `stageFlags`
    *   The `nullptr` value for `pImmutableSamplers`
4.  Create a variable of type `VkDescriptorSetLayoutCreateInfo` named `descriptor_set_layout_create_info`. Initialize its members with the following values:
    *   `VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO` value for `sType`
    *   `nullptr` value for `pNext`
    *   `0` value for `flags`
    *   The number of elements in the `bindings` vector for `bindingCount`
    *   The pointer to the first element of the `bindings` vector for `pBindings`

5.  Create a variable of type `VkDescriptorSetLayout` named `descriptor_set_layout`, in which the created layout will be stored.
6.  Call `vkCreateDescriptorSetLayout( logical_device, &descriptor_set_layout_create_info, nullptr, &descriptor_set_layout )` and provide the handle of the logical device, a pointer to the `descriptor_set_layout_create_info` variable, a `nullptr` value, and a pointer to the `descriptor_set_layout variable`.
7.  Make sure the call was successful by checking whether the return value is equal to `VK_SUCCESS`.

# How it works...

The descriptor set layout specifies the internal structure of a descriptor set and, at the same time, strictly defines what resources can be bound to the descriptor set (we can't use resources other than those specified in the layout).

When we want to create layouts, we need to know what resources (descriptor types) will be used and what their order will be. The order is specified through bindings--they define the index (position) of a resource within a given set and are also used inside shaders (with a set number through a `layout` qualifier) to specify a resource we want to access:

[PRE23]

We can choose any values for bindings, but we should keep in mind that unused indices may consume memory and impact the performance of our application.

To avoid unnecessary memory overhead and a negative performance impact, we should keep descriptor bindings as compact and as close to `0` as possible.

To create a descriptor set layout, we first need to specify a list of all the resources used in a given set:

[PRE24]

Next, we can create the layout like this:

[PRE25]

Descriptor set layouts (along with push constant ranges) also form a pipeline layout, which defines what type of resources can be accessed by a given pipeline. Created layouts, apart from pipeline layout creation, are also required during descriptor set allocation.

# See also

*   In [Chapter 8](5744ea05-b18a-4f84-a1df-250b549dfea5.xhtml), *Graphics and Compute Pipelines*, see the following recipe:
    *   *Creating a pipeline layout*
*   In this chapter, see the following recipe:
    *   *Allocating descriptor sets*

# Creating a descriptor pool

Descriptors, gathered into sets, are allocated from descriptor pools. When we create a pool, we must define which descriptors, and how many of them, can be allocated from the created pool.

# How to do it...

1.  Take the handle of a logical device on which the descriptor pool should be created. Store it in a variable of type `VkDevice` named `logical_device`.
2.  Create a vector variable named `descriptor_types` with elements of type `VkDescriptorPoolSize`. For each type of descriptor that will be allocated from the pool, add a new element to the `descriptor_types` variable defining the specified type of descriptor and the number of descriptors of a given type that will be allocated from the pool.
3.  Create a variable of type `VkDescriptorPoolCreateInfo` named `descriptor_pool_create_info`. Use the following values for members of this variable:
    *   `VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO` value for `sType`
    *   `nullptr` value for `pNext`
    *   `VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT` value if it should be possible to free individual sets allocated from this pool or a `0` value to only allow for freeing all the sets at once (through a pool reset operation) for `flags`
    *   The maximal number of sets that can be allocated from the pool for `maxSets`
    *   Number of elements in the `descriptor_types` vector for `poolSizeCount`
    *   Pointer to the first element of the `descriptor_types` vector for `pPoolSizes`
4.  Create a variable of type `VkDescriptorPool` named `descriptor_pool` in which the handle of the created pool will be stored.
5.  Call `vkCreateDescriptorPool( logical_device, &descriptor_pool_create_info, nullptr, &descriptor_pool )` and provide the `logical_device` variable, a pointer to the `descriptor_pool_create_info` variable, a `nullptr` value, and a pointer to the `descriptor_pool` variable.
6.  Make sure the pool was successfully created by checking whether the call returned a `VK_SUCCESS` value.

# How it works...

Descriptor pools manage the resources used for allocating descriptor sets (in a similar way to how command pools manage memory for command buffers). During descriptor pool creation, we specify the maximal amount of sets that can be allocated from a given pool and the maximal number of descriptors of a given type that can be allocated across all sets. This information is provided through a variable of type `VkDescriptorPoolCreateInfo` like this:

[PRE26]

In the preceding example, the types of descriptors and their total number are provided through a `descriptor_types` vector variable. It may contain multiple elements and the created pool will be big enough to allow for allocation of all the specified descriptors.

The pool itself is created like this:

[PRE27]

When we have created a pool, we can allocate descriptor sets from it. But we must remember that we can't do this in multiple threads at the same time.

We can't allocate descriptor sets from a given pool simultaneously in multiple threads.

# See also

See the following recipes in this chapter:

*   *Allocating descriptor sets*
*   *Freeing descriptor sets*
*   *Resetting a descriptor pool*
*   *Destroying a descriptor pool*

# Allocating descriptor sets

Descriptor sets gather shader resources (descriptors) in one container object. Its contents, types, and number of resources are defined by a descriptor set layout; storage is taken from pools, from which we can allocate descriptor sets.

# How to do it...

1.  Take the logical device and store its handle in a variable of type `VkDevice` named `logical_device`.
2.  Prepare a descriptor pool from which descriptor sets should be allocated. Use the pool's handle to initialize a variable of type `VkDescriptorPool` named `descriptor_pool`.
3.  Create a variable of type `std::vector<VkDescriptorSetLayout>` named `descriptor_set_layouts`. For each descriptor set that should be allocated from the pool, add a handle of a descriptor set layout that defines the structure of a corresponding descriptor set.
4.  Create a variable of type `VkDescriptorSetAllocateInfo` named `descriptor_set_allocate_info` and use the following values for its members:
    *   `VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO` value for `sType`
    *   `nullptr` value for `pNext`
    *   The `descriptor_pool` variable for `descriptorPool`
    *   The number of elements in the `descriptor_set_layouts` vector for `descriptorSetCount`
    *   The pointer to the first element of the `descriptorSetCount` vector for `pSetLayouts`
5.  Create a vector variable of type `std::vector<VkDescriptorSet>` named `descriptor_sets` and resize it to match the size of the `descriptor_set_layouts` vector.
6.  Call `vkAllocateDescriptorSets( logical_device, &descriptor_set_allocate_info, &descriptor_sets[0] )` and provide the `logical_device` variable, a pointer to the `descriptor_set_allocate_info` variable, and a pointer to the first element of the `descriptor_sets` vector.
7.  Make sure the call was successful and the `VK_SUCCESS` value was returned.

# How it works...

Descriptor sets are used to provide resources to shaders. They form an interface between the application and programmable pipeline stages. The structure of this interface is defined by the descriptor set layouts. And the actual data is provided when we update descriptor sets with image or buffer resources and later bind these descriptor sets to the command buffer during the recording operation.

Descriptor sets are allocated from pools. When we create a pool, we specify how many descriptors (resources) and of what type we can allocate from it across all descriptor sets that will be allocated from the pool. We also specify the maximum number of descriptor sets that can be allocated from the pool.

When we want to allocate descriptor sets, we need to specify layouts that will describe their internal structure--one layout for each descriptor set. This information is specified like this:

[PRE28]

Next, we allocate descriptor sets in the following way:

[PRE29]

Unfortunately, the pool's memory may become fragmented when we allocate and free separate descriptor sets. In such situations, we may not be able to allocate new sets from a given pool, even if we haven't reached the specified limits. This situation is presented in the following diagram:

![](img/B05542-05-02.png)

When we first allocate descriptors sets, the fragmentation problem will not occur. Additionally, if all descriptor sets use the same number of resources of the same type, it is guaranteed that this problem won't appear either.

To avoid problems with pool fragmentation, we can free all descriptor sets at once (by resetting a pool). Otherwise, if we can't allocate a new descriptor set and we don't want to reset the pool, we need to create another pool.

# See also

See the following recipes in this chapter:

*   *Creating a descriptor set layout*
*   *Creating a descriptor pool*
*   *Freeing descriptor sets*
*   *Resetting a descriptor pool*

# Updating descriptor sets

We have created a descriptor pool and allocated descriptor sets from it. We know their internal structure thanks to created layouts. Now we want to provide specific resources (samplers, image views, buffers, or buffer views) that should be later bound to the pipeline through descriptor sets. Defining resources that should be used is done through a process of updating descriptor sets.

# Getting ready

Updating descriptor sets requires us to provide a considerable amount of data for each descriptor involved in the process. What's more, the provided data depends on the type of descriptor. To simplify the process and lower the number of parameters that need to be specified, and also to improve error checking, custom structures are introduced in this recipe.

For samplers and all kinds of image descriptors, an `ImageDescriptorInfo` type is used which has the following definition:

[PRE30]

For uniform and storage buffers (and their dynamic variations), a `BufferDescriptorInfo` type is used. It has the following definition:

[PRE31]

For uniform and storage texel buffers, a `TexelBufferDescriptorInfo` type is introduced with the following definition:

[PRE32]

The preceding structures are used when we want to update descriptor sets with handles of new descriptors (that haven't been bound yet). It is also possible to copy descriptor data from other, already updated, sets. For this purpose, a `CopyDescriptorInfo` type is used that is defined like this:

[PRE33]

All the preceding structures define the handle of a descriptor set that should be updated, an index of a descriptor within the given set, and an index into an array if we want to update descriptors accessed through arrays. The rest of the parameters are type-specific.

# How to do it...

1.  Use the handle of a logical device to initialize a variable of type `VkDevice` named `logical_device`.
2.  Create a variable of type `std::vector<VkWriteDescriptorSet>` named `write_descriptors`. For each new descriptor that should be updated, add a new element to the vector and use the following values for its members:
    *   `VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET` value for `sType`
    *   `nullptr` value for `pNext`
    *   The handle of a descriptor set that should be updated for `dstSet`
    *   Index (binding) of a descriptor within a specified set for `dstBinding`
    *   The beginning index into an array from which descriptors should be updated if the given descriptor is accessed through an array inside shaders (or `0` value otherwise) for `dstArrayElement`
    *   The number of descriptors to be updated (number of elements in `pImageInfo`, `pBufferInfo` or `pTexelBufferView` arrays) for `descriptorCount`
    *   The type of descriptor for `descriptorType`
    *   In the case of sampler or image descriptors, specify an array with the `descriptorCount` elements and provide a pointer to its first element in `pImageInfo` (set `pBufferInfo` and `pTexelBufferView` members to `nullptr`). Use the following values for each array element:
        *   The sampler handle in the case of sampler and combined image sampler descriptors for `sampler`
        *   The image view handle in the case of the sampled image, storage image, combined image sampler, and input attachment descriptors for `imageView`
        *   The layout the given image will be in when a descriptor is accessed through shaders in the case of image descriptors for `imageLayout`
    *   In the case of uniform or storage buffers (and their dynamic variations), specify an array with the `descriptorCount` elements and provide a pointer to its first element in `pBufferInfo` (set `pImageInfo` and `pTexelBufferView` members to `nullptr`), and use the following values for each array element:
        *   The buffer's handle for `buffer`
        *   The memory offset (or base offset for dynamic descriptors) within a buffer for `offset`
        *   The buffer's memory size that should be used for a given descriptor for `range`
    *   In the case of uniform texel buffers or storage texel buffers, specify an array with the `descriptorCount` number of texel view handles, and provide a pointer to its first element in `pTexelBufferView` (set `pImageInfo` and `pBufferInfo` members to `nullptr`).

3.  Create a variable of type `std::vector<VkCopyDescriptorSet>` named `copy_descriptors`. Add an element to this vector for each descriptor data that should be copied from another, already updated, descriptor. Use the following values for the members of each new element:
    *   `VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET` value for `sType`
    *   `nullptr` value for `pNext`
    *   The handle of a descriptor set from which data should be copied for `srcSet`
    *   The binding number from within a source descriptor set for `srcBinding`
    *   The index into an array in the source descriptor set for `srcArrayElement`
    *   The handle of a descriptor set in which data should be updated for `dstSet`
    *   The binding number in the target descriptor set for `dstBinding`
    *   The array index in the target descriptor set for `dstArrayElement`
    *   The number of descriptors that should be copied from the source set and updated in the target set for `descriptorCount`
4.  Call `vkUpdateDescriptorSets( logical_device, static_cast<uint32_t>(write_descriptors.size()), &write_descriptors[0], static_cast<uint32_t>(copy_descriptors.size()), &copy_descriptors[0] )` and provide the `logical_device` variable, the number of elements in the `write_descriptors` vector, a pointer to the first element of the `write_descriptors`, the number of elements in the `copy_descriptors` vector, and a pointer to the first element of the `copy_descriptors` vector.

# How it works...

Updating descriptor sets causes specified resources (samplers, image views, buffers, or buffer views) to populate entries in the indicated sets. When the updated set is bound to a pipeline, such resources can be accessed through shaders.

We can write new (not used yet) resources to a descriptor set. In the following example, we do this by using the custom structures mentioned in the *Getting ready* section:

[PRE34]

We can also reuse descriptors from other sets. Copying already populated descriptors should be faster than writing new ones. This can be done like this:

[PRE35]

The operation of updating descriptor sets is performed through a single function call:

[PRE36]

# See also

See the following recipes in this chapter:

*   *Allocating descriptor sets*
*   *Binding descriptor sets*
*   *Creating descriptors with a texture and a uniform buffer*

# Binding descriptor sets

When a descriptor set is ready (we have updated it with all the resources that will be accessed in shaders), we need to bind it to a command buffer during the recording operation.

# How to do it...

1.  Take the handle of a command buffer that is being recorded. Store the handle in a variable of type `VkCommandBuffer` named `command_buffer`.
2.  Create a variable of type `VkPipelineBindPoint` named `pipeline_type` that will represent the type of a pipeline (graphics or compute) in which descriptor sets will be used.
3.  Take the pipeline's layout and store its handle in a variable of type `VkPipelineLayout` named `pipeline_layout` (refer to the *Creating a pipeline layout* recipe from [Chapter 8](5744ea05-b18a-4f84-a1df-250b549dfea5.xhtml), *Graphics and Compute Pipelines*).
4.  Create a variable of type `std::vector<VkDescriptorSet>` named `descriptor_sets`. For each descriptor set that should be bound to the pipeline, add a new element to the vector and initialize it with the descriptor set's handle.
5.  Select an index to which the first set from the provided list should be bound. Store the index in a variable of type `uint32_t` named `index_for_first_set`.
6.  If dynamic uniform or storage buffers are used in any of the sets being bound, create a variable of type `std::vector<uint32_t>` named `dynamic_offsets`, through which provide memory offset values for each dynamic descriptor defined in all the sets being bound. Offsets must be defined in the same order in which their corresponding descriptors appear in the layouts of each set (in order of increasing bindings).
7.  Make the following call:

[PRE37]

For this call, provide the `command_buffer`, `pipeline_type`, `pipeline_layout`, and `index_for_first_set` variables, the number of elements and a pointer to the first element of the `descriptor_sets` vector, and the number of elements and a pointer to the first element of the `dynamic_offsets` vector.

# How it works...

When we start recording a command buffer, its state is (almost entirely) undefined. So before we can record drawing operations that reference image or buffer resources, we need to bind appropriate resources to the command buffer. This is done by binding descriptor sets with the `vkCmdBindDescriptorSets()` function call like this:

[PRE38]

# See also

See the following recipes in this chapter:

*   *Creating a descriptor set layout*
*   *Allocating descriptor sets*
*   *Updating descriptor sets*

# Creating descriptors with a texture and a uniform buffer

In this sample recipe, we will see how to create the most commonly used resources: a combined image sampler and a uniform buffer. We will prepare a descriptor set layout for them, create a descriptor pool, and allocate a descriptor set from it. Then we will update the allocated set with the created resources. This way, we can later bind the descriptor set to a command buffer and access resources in shaders.

# How to do it...

1.  Create a combined image sampler (an image, image view, and a sampler) with the selected parameters--the most commonly used are `VK_IMAGE_TYPE_2D` image type, `VK_FORMAT_R8G8B8A8_UNORM` format, `VK_IMAGE_VIEW_TYPE_2D` view type, `VK_IMAGE_ASPECT_COLOR_BIT` aspect, `VK_FILTER_LINEAR` filter mode, and `VK_SAMPLER_ADDRESS_MODE_REPEAT` addressing mode for all texture coordinates. Store the created handles in a variable of type `VkSampler` named `sampler`, of type `VkImage` named `sampled_image`, and another one of type `VkImageView` named `sampled_image_view` (refer to the *Creating a combined image sampler* recipe).
2.  Create a uniform buffer with selected parameters and store the buffer's handle in a variable of type `VkBuffer` named `uniform_buffer` (refer to the *Creating a uniform buffer* recipe).
3.  Create a variable named `bindings` of type `std::vector<VkDescriptorSetLayoutBinding>`.
4.  Add one element with the following values to the `bindings` variable:
    *   `0` value for `binding`
    *   `VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER` value for `descriptorType`
    *   `1` value for `descriptorCount`
    *   The `VK_SHADER_STAGE_FRAGMENT_BIT` value for `stageFlags`
    *   The `nullptr` value for `stageFlags`
5.  Add another element to the `bindings` vector and use the following values for its members:
    *   `1` value for `binding`
    *   `VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER` value for `descriptorType`
    *   `1` value for `descriptorCount`
    *   The `VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT` value for `stageFlags`
    *   The `nullptr` value for `pImmutableSamplers`
6.  Create a descriptor set layout using the `bindings` variable and store its handle in a variable of type `VkDescriptorSetLayout` named `descriptor_set_layout` (refer to the *Creating a descriptor set layout* recipe).

7.  Create a variable of type `std::vector<VkDescriptorPoolSize>` named `descriptor_types`. Add two elements to the created vector: one with `VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER` and `1` values, the second with `VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER` and `1` values.
8.  Create a descriptor pool in which separate descriptor sets cannot be freed individually and only one descriptor set can be allocated. Use the `descriptor_types` variable during pool creation and store its handle in a variable of type `VkDescriptorPool` named `descriptor_pool` (refer to the *Creating a descriptor pool* recipe).
9.  Allocate one descriptor set from `descriptor_pool` using the `descriptor_set_layout` layout variable. Store the created handle in a one-element vector of type `std::vector<VkDescriptorSet>` named `descriptor_sets` (refer to the *Allocating descriptor sets* recipe).
10.  Create a variable of type `std::vector<ImageDescriptorInfo>` named `image_descriptor_infos`. Add one element to this vector with the following values:
    *   The `descriptor_sets[0]` for `TargetDescriptorSet`
    *   `0` value for `TargetDescriptorBinding`
    *   `0` value for `TargetArrayElement`
    *   `VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER` value for `TargetDescriptorType`
    *   Add one element to the `ImageInfos` member vector with the following values:
        *   The `sampler` variable for `sampler`
        *   The `sampled_image_view` variable for `imageView`
        *   The `VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL` value for `imageLayout`
11.  Create a variable of type `std::vector<BufferDescriptorInfo>` named `buffer_descriptor_infos` with one element initialized with the following values:
    *   The `descriptor_sets[0]` for `TargetDescriptorSet`
    *   `1` value for `TargetDescriptorBinding`
    *   `0` value for `TargetArrayElement`
    *   `VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER` value for `TargetDescriptorType`
    *   Add one element to the `BufferInfos` member vector and use the following values to initialize its members:
        *   The `uniform_buffer` variable for `buffer`
        *   The `0` for `offset`
        *   The `VK_WHOLE_SIZE` value for `range`
12.  Update the descriptor sets using the `image_descriptor_infos` and `buffer_descriptor_infos` vectors.

# How it works...

To prepare the typically used descriptors, a combined image sampler and a uniform buffer, we first need to create them:

[PRE39]

Next, we prepare a layout that will define the internal structure of a descriptor set:

[PRE40]

After that, we create a descriptor pool and allocate a descriptor set from it:

[PRE41]

The last thing to do is to update the descriptor set with the resources created at the beginning:

[PRE42]

# See also

See the following recipes in this chapter:

*   *Creating a combined image sampler*
*   *Creating a uniform buffer*
*   *Creating a descriptor set layout*
*   *Creating a descriptor pool*
*   *Allocating descriptor sets*
*   *Updating descriptor sets*

# Freeing descriptor sets

If we want to return memory allocated by a descriptor set and give it back to the pool, we can free a given descriptor set.

# How to do it...

1.  Use the handle of a logical device to initialize a variable of type `VkDevice` named `logical_device`.
2.  Take the descriptor pool that was created with a `VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT` flag. Store its handle in a variable of type `VkDescriptorPool` named `descriptor_pool`.
3.  Create a vector of type `std::vector<VkDescriptorSet>` named `descriptor_sets`. Add all the descriptor sets that should be freed to the vector.
4.  Call `vkFreeDescriptorSets( logical_device, descriptor_pool, static_cast<uint32_t>(descriptor_sets.size()), descriptor_sets.data() )`. For the call provide the `logical_device` and `descriptor_pool` variables, the number of elements in the `descriptor_sets` vector, and a pointer to the first element of the `descriptor_sets` vector.
5.  Make sure the call was successful by checking whether it returns a `VK_SUCCESS` value.
6.  Clear the `descriptor_sets` vector as we can't use the handles of freed descriptor sets any more.

# How it works...

Freeing a descriptor set releases memory used by it and gives it back to the pool. It should be possible to allocate another set of the same type from the pool but it may not be possible due to the pool's memory fragmentation (in such a situation, we may need to create another pool or reset the one from which the set was allocated).

We can free multiple descriptor sets at once, but all of them must come from the same pool. It is done like this:

[PRE43]

We cannot free descriptor sets allocated from the same pool from multiple threads at the same time.

# See also

See the following recipes in this chapter:

*   *Creating a descriptor pool*
*   *Allocating descriptor sets*
*   *Resetting a descriptor pool*
*   *Destroying a descriptor pool*

# Resetting a descriptor pool

We can free all descriptor sets allocated from a given pool at once without destroying the pool itself. To do that, we can reset a descriptor pool.

# How to do it...

1.  Take the descriptor pool that should be reset and use its handle to initialize a variable of type `VkDescriptorPool` named `descriptor_pool`.
2.  Take the handle of a logical device on which the descriptor pool was created. Store its handle in a variable of type `VkDevice` named `logical_device`.

3.  Make the following call: `vkResetDescriptorPool( logical_device, descriptor_pool, 0 )`, for which use the `logical_device` and `descriptor_pool` variables and a `0` value.
4.  Check for any error returned by the call. As successful operation should return `VK_SUCCESS`.

# How it works...

Resetting a descriptor pool returns all the descriptor sets allocated from it back to the pool. All descriptor sets allocated from the pool are implicitly freed and they can't be used any more (their handles become invalid).

If the pool is created without a `VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT` flag set, it is the only way to free descriptor sets allocated from it (apart from destroying the pool), as in such a situation, we can't free them individually.

To reset the pool, we can write code similar to the following:

[PRE44]

# See also

See the following recipes in this chapter:

*   *Creating a descriptor pool*
*   *Allocating descriptor sets*
*   *Freeing descriptor sets*
*   *Destroying a descriptor pool*

# Destroying a descriptor pool

When we don't need a descriptor pool any more, we can destroy it (with all descriptor sets allocated from the pool).

# How to do it...

1.  Take the handle of a created logical device and store it in a variable of type `VkDevice` named `logical_device`.
2.  Provide the handle of the descriptor pool through a variable of type `VkDescriptorPool` named `descriptor_pool`.
3.  Call `vkDestroyDescriptorPool( logical_device, descriptor_pool, nullptr )` and provide the `logical_device` and `descriptor_pool` variables and a `nullptr` value.
4.  For safety, assign the `VK_NULL_HANDLE` value to the `descriptor_pool` variable.

# How it works...

Destroying a descriptor pool implicitly frees all descriptor sets allocated from it. We don't need to free individual descriptor sets first. But because of this, we need to make sure that none of the descriptor sets allocated from the pool are referenced by the commands that are currently processed by the hardware.

When we are ready, we can destroy a descriptor pool like this:

[PRE45]

# See also

See the following recipe in this chapter:

*   *Creating a descriptor pool*

# Destroying a descriptor set layout

Descriptor set layouts that are no longer used should be destroyed.

# How to do it...

1.  Provide a logical device's handle using a variable of type `VkDevice` named `logical_device`.
2.  Take the handle of a created descriptor set layout and use it to initialize a variable of type `VkDescriptorSetLayout` named `descriptor_set_layout`.
3.  Call `vkDestroyDescriptorSetLayout( logical_device, descriptor_set_layout, nullptr )` and provide handles of the logical device and descriptor set layout, and a `nullptr` value.
4.  For safety, assign the `VK_NULL_HANDLE` value to the `descriptor_set_layout` variable.

# How it works...

Descriptor set layouts are destroyed with the `vkDestroyDescriptorSetLayout()` function like this:

[PRE46]

# See also

See the following recipe in this chapter:

*   *Creating a descriptor set layout*

# Destroying a sampler

When we no longer need a sampler and we are sure it is not used anymore by the pending commands, we can destroy it.

# How to do it...

1.  Take the handle of a logical device on which the sampler was created and store it in a variable of type `VkDevice` named `logical_device`.
2.  Take the handle of the sampler that should be destroyed. Provide it through a variable of type `VkSampler` named `sampler`.
3.  Call `vkDestroySampler( logical_device, sampler, nullptr )` and provide the `logical_device` and `sampler` variables, and a `nullptr` value.
4.  For safety, assign the `VK_NULL_HANDLE` value to the `sampler` variable.

# How it works...

Samplers are destroyed like this:

[PRE47]

We don't have to check whether the sampler's handle is not empty, because a deletion of a `VK_NULL_HANDLE` is ignored. We do this just to avoid an unnecessary function call. But when we delete a sampler, we must be sure that the handle (if not empty) is valid.

# See also

See the following recipe in this chapter:

*   *Creating a sampler*