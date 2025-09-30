# 描述符集

在本章中，我们将介绍以下食谱：

+   创建采样器

+   创建采样图像

+   创建组合图像采样器

+   创建存储镜像

+   创建统一纹理缓冲区

+   创建存储纹理缓冲区

+   创建统一缓冲区

+   创建存储缓冲区

+   创建输入附加

+   创建描述符集布局

+   创建描述符池

+   分配描述符集

+   更新描述符集

+   绑定描述符集

+   使用纹理和统一缓冲区创建描述符

+   释放描述符集

+   重置描述符池

+   销毁描述符池

+   销毁描述符集布局

+   销毁采样器

# 简介

在现代计算机图形学中，大多数图像数据（如顶点、像素或片段）的渲染和处理都是通过可编程管道和着色器完成的。为了正确运行并生成适当的结果，着色器需要访问额外的数据源，如纹理、采样器、缓冲区或统一变量。在 Vulkan 中，这些通过描述符集提供。

描述符是表示着色器资源的不可见数据结构。它们被组织成组或集，其内容由描述符集布局指定。为了向着色器提供资源，我们将描述符集绑定到管道上。我们可以一次绑定多个集。要从着色器内部访问资源，我们需要指定从哪个集以及从集内的哪个位置（称为**绑定**）获取给定的资源。

在本章中，我们将学习各种描述符类型。我们将了解如何准备资源（采样器、缓冲区和图像），以便它们可以在着色器中使用。我们还将探讨如何设置应用程序和着色器之间的接口，并在着色器中使用资源。

# 创建采样器

采样器定义了一组参数，这些参数控制着色器内部如何加载图像数据（采样）。这些参数包括地址计算（即，环绕或重复）、过滤（线性或最近）或使用米普映射。要从着色器内部使用采样器，我们首先需要创建它们。

# 如何操作...

1.  获取逻辑设备的句柄并将其存储在名为`logical_device`的`VkDevice`类型变量中。

1.  创建一个名为`sampler_create_info`的`VkSamplerCreateInfo`类型变量，并为其成员使用以下值：

    +   `sType`的值为`VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO`

    +   `pNext`的值为`nullptr`

    +   `flags`的值为`0`

    +   为`magFilter`和`minFilter`指定的所需放大和缩小过滤模式（`VK_FILTER_NEAREST`或`VK_FILTER_LINEAR`）

    +   为`mipmapMode`选择的米普映射过滤模式（`VK_SAMPLER_MIPMAP_MODE_NEAREST`或`VK_SAMPLER_MIPMAP_MODE_LINEAR`）

    +   用于图像 U、V 和 W 坐标超出`0.0 - 1.0`范围（`VK_SAMPLER_ADDRESS_MODE_REPEAT`、`VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT`、`VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE`、`VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER`或`VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE`）的选定图像寻址模式，对于`addressModeU`、`addressModeV`和`addressModeW`

    +   要添加到 mipmap 细节级别计算的期望值，对于`mipLodBias`

    +   如果应启用各向异性过滤，则为`true`值，否则对于`anisotropyEnable`为`false`

    +   对于`maxAnisotropy`的各向异性最大值

    +   如果在图像查找期间应启用与参考值的比较，则为`true`值，否则对于`compareEnable`为`false`

    +   用于`compareOp`的选定比较函数应用于获取的数据（`VK_COMPARE_OP_NEVER`、`VK_COMPARE_OP_LESS`、`VK_COMPARE_OP_EQUAL`、`VK_COMPARE_OP_LESS_OR_EQUAL`、`VK_COMPARE_OP_GREATER`、`VK_COMPARE_OP_NOT_EQUAL`、`VK_COMPARE_OP_GREATER_OR_EQUAL`或`VK_COMPARE_OP_ALWAYS`）

    +   用于`minLod`和`maxLod`的将计算出的图像的细节级别值（mipmap 编号）夹断的最小和最大值

    +   用于`borderColor`的预定义边界颜色值之一（`VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK`、`VK_BORDER_COLOR_INT_TRANSPARENT_BLACK`、`VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK`、`VK_BORDER_COLOR_INT_OPAQUE_BLACK`、`VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE`或`VK_BORDER_COLOR_INT_OPAQUE_WHITE`）

    +   如果寻址应使用图像的维度，则为`true`值，如果寻址应使用归一化坐标（在`0.0`-`1.0`范围内），则为`false`值，对于`unnormalizedCoordinates`

1.  创建一个名为`sampler`的`VkSampler`类型变量，其中将存储创建的采样器。

1.  调用`vkCreateSampler(logical_device, &sampler_create_info, nullptr, &sampler)`，并提供`logical_device`变量、`sampler_create_info`变量的指针、`nullptr`值和`sampler`变量的指针。

1.  通过检查返回值是否等于`VK_SUCCESS`来确保调用成功。

# 它是如何工作的...

采样器控制着色器内读取图像的方式。它们可以单独使用或与采样图像结合使用。

采样器用于`VK_DESCRIPTOR_TYPE_SAMPLER`描述符类型。

使用类型为`VkSamplerCreateInfo`的变量指定采样参数，如下所示：

```cpp
VkSamplerCreateInfo sampler_create_info = { 
  VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, 
  nullptr, 
  0, 
  mag_filter, 
  min_filter, 
  mipmap_mode, 
  u_address_mode, 
  v_address_mode, 
  w_address_mode, 
  lod_bias, 
  anisotropy_enable, 
  max_anisotropy, 
  compare_enable, 
  compare_operator, 
  min_lod, 
  max_lod, 
  border_color, 
  unnormalized_coords 
};

```

然后将此变量提供给创建采样器的函数：

```cpp
VkResult result = vkCreateSampler( logical_device, &sampler_create_info, nullptr, &sampler ); 
if( VK_SUCCESS != result ) { 
  std::cout << "Could not create sampler." << std::endl; 
  return false; 
} 
return true;

```

要在着色器中指定采样器，我们需要创建一个带有`sampler`关键字的统一变量。

一个使用采样器的 GLSL 代码示例，从中可以生成 SPIR-V 汇编，可能看起来像这样：

```cpp
layout (set=m, binding=n) uniform sampler <variable name>;

```

# 参见

+   请参阅本章中的以下配方：

    +   *销毁采样器*

# 创建一个采样图像

采样图像用于在着色器内部读取图像（纹理）中的数据。通常，它们与采样器一起使用。为了能够将图像用作采样图像，它必须使用`VK_IMAGE_USAGE_SAMPLED_BIT`使用创建。

# 如何做到这一点...

1.  获取存储在名为`physical_device`的`VkPhysicalDevice`类型变量中的物理设备的句柄。

1.  选择用于图像的格式。使用所选的图像格式初始化一个名为`format`的`VkFormat`类型变量。

1.  创建一个名为`format_properties`的`VkFormatProperties`类型变量。

1.  调用`vkGetPhysicalDeviceFormatProperties( physical_device, format, &format_properties )`，为它提供`physical_device`变量、`format`变量和`format_properties`变量的指针。

1.  确保所选的图像格式适合采样图像。通过检查`format_properties`变量中`optimalTilingFeatures`成员的`VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT`位是否设置来完成此操作。

1.  如果采样图像将被线性过滤或如果其 mipmap 将被线性过滤，请确保所选格式适合线性过滤的采样图像。通过检查`format_properties`变量中`optimalTilingFeatures`成员的`VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT`位是否设置来完成此操作。

1.  从存储在`physical_device`变量中的句柄获取逻辑设备的句柄，并使用它来初始化一个名为`logical_device`的`VkDevice`类型变量。

1.  使用`logical_device`和`format`变量创建一个图像，并选择其余的图像参数。在创建图像时，不要忘记提供`VK_IMAGE_USAGE_SAMPLED_BIT`使用。将图像的句柄存储在一个名为`sampled_image`的`VkImage`类型变量中（参考第四章，*资源和内存*中的*创建图像*配方）。

1.  分配一个具有`VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT`属性的内存对象（或使用现有内存对象的范围），并将其绑定到创建的图像上（参考第四章，*资源和内存*中的*分配和绑定内存对象到图像*配方）。

1.  使用`logical_device`、`sampled_image`和`format`变量创建一个图像视图，并选择其余的视图参数。将图像视图的句柄存储在一个名为`sampled_image_view`的`VkImageView`类型变量中（参考第四章，*资源和内存*中的*创建图像视图*配方）。

# 它是如何工作的...

采样图像用作着色器内部图像数据（纹理）的来源。要从图像中获取数据，通常需要一个采样器对象，该对象定义了数据应该如何读取（参考*创建采样器*配方）。

采样图像用于`VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE`描述符类型。

在着色器内部，我们可以使用多个采样器以不同的方式从同一图像中读取数据。我们也可以使用相同的采样器与多个图像一起使用。但在某些平台上，使用组合图像采样器对象可能更优，这些对象将采样器和采样图像组合在一个对象中。

并非所有图像格式都支持用于采样图像；这取决于应用程序执行的平台。但有一组强制性的格式，始终可以用于采样图像和线性过滤的采样图像。以下是一些此类格式的示例（但不限于）：

+   `VK_FORMAT_B4G4R4A4_UNORM_PACK16`

+   `VK_FORMAT_R5G6B5_UNORM_PACK16`

+   `VK_FORMAT_A1R5G5B5_UNORM_PACK16`

+   `VK_FORMAT_R8_UNORM` 和 `VK_FORMAT_R8_SNORM`

+   `VK_FORMAT_R8G8_UNORM` 和 `VK_FORMAT_R8G8_SNORM`

+   `VK_FORMAT_R8G8B8A8_UNORM`, `VK_FORMAT_R8G8B8A8_SNORM`, 和 `VK_FORMAT_R8G8B8A8_SRGB`

+   `VK_FORMAT_B8G8R8A8_UNORM` 和 `VK_FORMAT_B8G8R8A8_SRGB`

+   `VK_FORMAT_A8B8G8R8_UNORM_PACK32`, `VK_FORMAT_A8B8G8R8_SNORM_PACK32`, 和 `VK_FORMAT_A8B8G8R8_SRGB_PACK32`

+   `VK_FORMAT_A2B10G10R10_UNORM_PACK32`

+   `VK_FORMAT_R16_SFLOAT`

+   `VK_FORMAT_R16G16_SFLOAT`

+   `VK_FORMAT_R16G16B16A16_SFLOAT`

+   `VK_FORMAT_B10G11R11_UFLOAT_PACK32`

+   `VK_FORMAT_E5B9G9R9_UFLOAT_PACK32`

如果我们想使用一些不太典型的格式，我们需要检查它是否可以用于采样图像。这可以通过以下方式完成：

```cpp
VkFormatProperties format_properties; 
vkGetPhysicalDeviceFormatProperties( physical_device, format, &format_properties ); 
if( !(format_properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) ) { 
  std::cout << "Provided format is not supported for a sampled image." << std::endl; 
  return false; 
} 
if( linear_filtering && 
    !(format_properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) ) { 
  std::cout << "Provided format is not supported for a linear image filtering." << std::endl; 
  return false; 
}

```

如果我们确定所选格式适合我们的需求，我们可以创建一个图像、为其创建一个内存对象以及一个图像视图（在 Vulkan 中，大多数情况下用图像视图表示图像）。在创建图像时，我们需要指定`VK_IMAGE_USAGE_SAMPLED_BIT`使用：

```cpp
if( !CreateImage( logical_device, type, format, size, num_mipmaps, num_layers, VK_SAMPLE_COUNT_1_BIT, usage | VK_IMAGE_USAGE_SAMPLED_BIT, false, sampled_image ) ) { 
  return false; 
} 

if( !AllocateAndBindMemoryObjectToImage( physical_device, logical_device, sampled_image, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, memory_object ) ) { 
  return false; 
} 

if( !CreateImageView( logical_device, sampled_image, view_type, format, aspect, sampled_image_view ) ) { 
  return false; 
} 
return true;

```

当我们想在着色器内部使用图像作为采样图像之前，我们需要将图像的布局转换为`VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL`。

为了在着色器中创建一个表示采样图像的统一变量，我们需要使用一个带有适当维度的`texture`关键字（可能带有前缀）。

一个 GLSL 代码示例，从中可以生成 SPIR-V 汇编代码，该代码使用采样图像，可能看起来像这样：

```cpp
layout (set=m, binding=n) uniform texture2D <variable name>;

```

# 参见

+   在第四章，*资源和内存*，查看以下食谱：

    +   *创建一个图像*

    +   *分配和绑定内存对象到图像*

    +   *创建图像视图*

    +   *销毁图像视图*

    +   *销毁图像*

    +   *释放内存对象*

+   在本章中，查看以下食谱：

    +   *创建采样器*

# 创建组合图像采样器

从应用程序（API）的角度来看，采样器和采样图像始终是单独的对象。但在着色器内部，它们可以组合成一个对象。在某些平台上，在着色器内部从组合图像采样器中采样可能比使用单独的采样器和采样图像更优。

# 如何做到这一点...

1.  创建一个采样器对象并将它的句柄存储在名为`sampler`的`VkSampler`类型变量中（参考*创建采样器*食谱）。

1.  创建一个采样图像。将创建的图像句柄存储在名为 `sampled_image` 的 `VkImage` 类型变量中。为采样图像创建一个适当的视图，并将它的句柄存储在名为 `sampled_image_view` 的 `VkImageView` 类型变量中（参考 *创建一个采样图像* 的配方）。

# 它是如何工作的...

在我们的应用程序中，组合图像采样器与普通采样器和采样图像的创建方式相同。它们在着色器内部的使用方式不同。

组合图像采样器可以绑定到 `VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER` 类型的描述符。

以下代码使用 *创建一个采样器* 和 *创建一个采样图像* 的配方来创建必要的对象：

```cpp
if( !CreateSampler( logical_device, mag_filter, min_filter, mipmap_mode, u_address_mode, v_address_mode, w_address_mode, lod_bias, anisotropy_enable, max_anisotropy, compare_enable, compare_operator, min_lod, max_lod, border_color, unnormalized_coords, sampler ) ) { 
  return false; 
} 

bool linear_filtering = (mag_filter == VK_FILTER_LINEAR) || (min_filter == VK_FILTER_LINEAR) || (mipmap_mode == VK_SAMPLER_MIPMAP_MODE_LINEAR); 
if( !CreateSampledImage( physical_device, logical_device, type, format, size, num_mipmaps, num_layers, usage, view_type, aspect, linear_filtering, sampled_image, sampled_image_view ) ) { 
  return false; 
} 
return true;

```

差异在于着色器内部。

要在 GLSL 着色器内部创建表示组合图像采样器的变量，我们需要使用一个 `sampler` 关键字（可能带有前缀）并指定适当的维度。

不要混淆采样器和组合图像采样器--两者在着色器内部都使用 `sampler` 关键字，但组合图像采样器还额外指定了维度，如下例所示：

```cpp
layout (set=m, binding=n) uniform sampler2D <variable name>;

```

组合图像采样器需要单独处理，因为使用它们的应用程序在某些平台上可能会有更好的性能。因此，如果没有特定原因需要使用单独的采样器和采样图像，我们应该尝试将它们组合成单个对象。

# 参见

在 第四章，*资源和内存* 中，查看以下配方：

+   *创建一个图像*

+   *分配和绑定内存对象到图像*

+   *创建一个图像视图*

+   *销毁一个图像视图*

+   *销毁一个图像*

+   *释放内存对象*

在本章中查看以下配方：

+   *创建一个采样器*

+   *创建一个采样图像*

+   *销毁一个采样器*

# 创建一个存储图像

存储图像允许我们从绑定到管道的图像中加载（未过滤的）数据。但更重要的是，它们还允许我们在图像中存储着色器中的数据。此类图像必须使用指定了 `VK_IMAGE_USAGE_STORAGE_BIT` 使用标志来创建。

# 如何做到...

1.  获取物理设备的句柄并将其存储在名为 `physical_device` 的 `VkPhysicalDevice` 类型变量中。

1.  选择用于存储图像的格式。使用所选格式初始化名为 `format` 的 `VkFormat` 类型变量。

1.  创建一个名为 `format_properties` 的 `VkFormatProperties` 类型变量。

1.  调用 `vkGetPhysicalDeviceFormatProperties( physical_device, format, &format_properties )` 并提供 `physical_device` 变量、`format` 变量和 `format_properties` 变量的指针。

1.  检查所选图像格式是否适合存储图像。通过检查 `format_properties` 变量的 `optimalTilingFeatures` 成员的 `VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT` 位是否设置来完成此操作。

1.  如果将在存储图像上执行原子操作，请确保所选格式支持这些操作。通过检查 `format_properties` 变量的 `optimalTilingFeatures` 成员的 `VK_FORMAT_FEATURE_STORAGE_IMAGE_ATOMIC_BIT` 位是否设置来完成此操作。

1.  获取由 `physical_device` 创建的逻辑设备的句柄，并使用它来初始化一个名为 `logical_device` 的 `VkDevice` 类型变量。

1.  使用 `logical_device` 和 `format` 变量创建一个图像，并选择其余的图像参数。确保在创建图像时指定 `VK_IMAGE_USAGE_STORAGE_BIT` 用法。将创建的句柄存储在一个名为 `storage_image` 的 `VkImage` 类型变量中（参考第四章，*资源和内存*中的*创建图像*配方）。

1.  分配一个具有 `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT` 属性的内存对象（或使用现有内存对象的一个范围）并将其绑定到图像（参考第四章，*资源和内存*中的*将内存对象分配和绑定到图像*配方）。

1.  使用 `logical_device`、`storage_image` 和 `format` 变量创建一个图像视图，并选择其余的视图参数。将图像视图的句柄存储在一个名为 `storage_image_view` 的 `VkImageView` 类型变量中（参考第四章，*资源和内存*中的*创建图像视图*配方）。

# 它是如何工作的...

当我们想在着色器内部存储图像数据时，我们需要使用存储图像。我们也可以从这样的图像中加载数据，但这些加载是不过滤的（我们不能为存储图像使用采样器）。

存储图像对应于 `VK_DESCRIPTOR_TYPE_STORAGE_IMAGE` 类型的描述符。

存储图像是以 `VK_IMAGE_USAGE_STORAGE_BIT` 用法创建的。我们也不能忘记指定适当的格式。并非所有格式都可以始终用于存储图像。这取决于我们的应用程序执行的平台。但是，有一个必需格式的列表，所有 Vulkan 驱动程序都必须支持。它包括（但不限于）以下格式：

+   `VK_FORMAT_R8G8B8A8_UNORM`, `VK_FORMAT_R8G8B8A8_SNORM`, `VK_FORMAT_R8G8B8A8_UINT`, 和 `VK_FORMAT_R8G8B8A8_SINT`

+   `VK_FORMAT_R16G16B16A16_UINT`, `VK_FORMAT_R16G16B16A16_SINT` 和 `VK_FORMAT_R16G16B16A16_SFLOAT`

+   `VK_FORMAT_R32_UINT`, `VK_FORMAT_R32_SINT` 和 `VK_FORMAT_R32_SFLOAT`

+   `VK_FORMAT_R32G32_UINT`, `VK_FORMAT_R32G32_SINT` 和 `VK_FORMAT_R32G32_SFLOAT`

+   `VK_FORMAT_R32G32B32A32_UINT`, `VK_FORMAT_R32G32B32A32_SINT` 和 `VK_FORMAT_R32G32B32A32_SFLOAT`

如果我们想在存储图像上执行原子操作，则必需格式的列表要短得多，并且仅包括以下几种：

+   `VK_FORMAT_R32_UINT`

+   `VK_FORMAT_R32_SINT`

如果存储图像需要其他格式，或者如果我们需要使用其他格式在存储图像上执行原子操作，我们必须检查所选格式是否在应用程序执行的平台上是受支持的。这可以通过以下代码完成：

```cpp
VkFormatProperties format_properties; 
vkGetPhysicalDeviceFormatProperties( physical_device, format, &format_properties ); 
if( !(format_properties.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) ) { 
  std::cout << "Provided format is not supported for a storage image." << std::endl; 
  return false; 
} 
if( atomic_operations && 
    !(format_properties.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_ATOMIC_BIT) ) { 
  std::cout << "Provided format is not supported for atomic operations on storage images." << std::endl; 
  return false; 
}

```

如果格式受支持，我们像往常一样创建图像，但我们需要指定 `VK_IMAGE_USAGE_STORAGE_BIT` 使用方式。图像准备好后，我们需要创建一个内存对象，将其绑定到图像，并且我们还需要一个图像视图。这些操作可以像这样执行：

```cpp
if( !CreateImage( logical_device, type, format, size, num_mipmaps, num_layers, VK_SAMPLE_COUNT_1_BIT, usage | VK_IMAGE_USAGE_STORAGE_BIT, false, storage_image ) ) { 
  return false; 
} 

if( !AllocateAndBindMemoryObjectToImage( physical_device, logical_device, storage_image, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, memory_object ) ) { 
  return false; 
} 

if( !CreateImageView( logical_device, storage_image, view_type, format, aspect, storage_image_view ) ) { 
  return false; 
} 
return true;

```

在我们能够从着色器中加载或存储存储图像中的数据之前，我们必须执行到 `VK_IMAGE_LAYOUT_GENERAL` 布局的转换。这是唯一支持这些操作的布局。

在 GLSL 着色器内部，存储图像使用 `image` 关键字（可能带有前缀）和适当的维度来指定。我们还需要在 `layout` 限定符内提供图像的格式。

下面提供了一个在 GLSL 着色器中定义存储图像的示例：

```cpp
layout (set=m, binding=n, r32f) uniform image2D <variable name>;

```

# 参见

在 第四章，*资源和内存* 中，查看以下食谱：

+   *创建图像*

+   *将内存对象分配和绑定到图像*

+   *创建图像视图*

+   *销毁图像视图*

+   *销毁图像*

+   *释放内存对象*

# 创建均匀的 texel 缓冲区

均匀的 texel 缓冲区允许我们以类似于从图像读取数据的方式读取数据--它们的内 容不是解释为单个（标量）值的数组，而是解释为具有一个、两个、三个或四个组件的格式化像素（texel）。但是，通过这样的缓冲区，我们可以访问比通过常规图像提供的数据大得多的数据。

当我们想要将缓冲区用作均匀的 texel 缓冲区时，我们需要指定 `VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT` 使用方式。

# 如何做到...

1.  将物理设备的句柄存储在名为 `physical_device` 的 `VkPhysicalDevice` 类型的变量中。

1.  选择一个格式，其中将存储缓冲区数据。使用该格式初始化一个名为 `format` 的 `VkFormat` 类型的变量。

1.  创建一个名为 `format_properties` 的 `VkFormatProperties` 类型的变量。

1.  调用 `vkGetPhysicalDeviceFormatProperties( physical_device, format, &format_properties )` 并提供物理设备的句柄、`format` 变量以及 `format_properties` 变量的指针。

1.  通过检查 `format_properties` 变量的 `bufferFeatures` 成员是否设置了 `VK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT` 位，确保所选格式适合均匀的 texel 缓冲区。

1.  从所选物理设备的句柄创建一个逻辑设备。将其存储在名为 `logical_device` 的 `VkDevice` 类型的变量中。

1.  创建一个名为 `uniform_texel_buffer` 的 `VkBuffer` 类型的变量。

1.  使用`logical_device`变量创建一个具有所需大小和用途的缓冲区。不要忘记在创建缓冲区时包括`VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT`用途。将创建的句柄存储在`uniform_texel_buffer`变量中（参考第四章中的*创建缓冲区*配方，*资源和内存*）。

1.  分配一个具有`VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT`属性的内存对象（或使用现有的一个）并将其绑定到缓冲区。如果分配了新的内存对象，将其存储在名为`memory_object`的`VkDeviceMemory`类型变量中（参考第四章中的*分配和绑定内存对象到缓冲区*配方，*资源和内存*）。

1.  使用`logical_device`、`uniform_texel_buffer`和`format`变量以及所需的偏移量和内存范围创建一个缓冲区视图。将结果句柄存储在名为`uniform_texel_buffer_view`的`VkBufferView`类型变量中（参考第四章中的*创建缓冲区视图*配方，*资源和内存*）。

# 它是如何工作的...

均匀的纹理缓冲区允许我们提供解释为一维图像的数据。但这个数据可能比典型的图像大得多。Vulkan 规范要求每个驱动程序至少支持 4,096 个纹理元素的 1D 图像。但对于纹理缓冲区，这个最小要求限制增加到 65,536 个元素。

均匀的纹理缓冲区绑定到`VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER`类型的描述符。

均匀的纹理缓冲区使用`VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT`用途创建。但除此之外，我们还需要选择一个合适的格式。并非所有格式都与此类缓冲区兼容。可以与均匀纹理缓冲区一起使用的强制格式列表包括（但不限于）以下格式： 

+   `VK_FORMAT_R8_UNORM`、`VK_FORMAT_R8_SNORM`、`VK_FORMAT_R8_UINT`和`VK_FORMAT_R8_SINT`

+   `VK_FORMAT_R8G8_UNORM`、`VK_FORMAT_R8G8_SNORM`、`VK_FORMAT_R8G8_UINT`和`VK_FORMAT_R8G8_SINT`

+   `VK_FORMAT_R8G8B8A8_UNORM`、`VK_FORMAT_R8G8B8A8_SNORM`、`VK_FORMAT_R8G8B8A8_UINT`和`VK_FORMAT_R8G8B8A8_SINT`

+   `VK_FORMAT_B8G8R8A8_UNORM`

+   `VK_FORMAT_A8B8G8R8_UNORM_PACK32`、`VK_FORMAT_A8B8G8R8_SNORM_PACK32`、`VK_FORMAT_A8B8G8R8_UINT_PACK32`和`VK_FORMAT_A8B8G8R8_SINT_PACK32`

+   `VK_FORMAT_A2B10G10R10_UNORM_PACK32`和`VK_FORMAT_A2B10G10R10_UINT_PACK32`

+   `VK_FORMAT_R16_UINT`、`VK_FORMAT_R16_SINT`和`VK_FORMAT_R16_SFLOAT`

+   `VK_FORMAT_R16G16_UINT`、`VK_FORMAT_R16G16_SINT`和`VK_FORMAT_R16G16_SFLOAT`

+   `VK_FORMAT_R16G16B16A16_UINT`、`VK_FORMAT_R16G16B16A16_SINT`和`VK_FORMAT_R16G16B16A16_SFLOAT`

+   `VK_FORMAT_R32_UINT`、`VK_FORMAT_R32_SINT`和`VK_FORMAT_R32_SFLOAT`

+   `VK_FORMAT_R32G32_UINT`、`VK_FORMAT_R32G32_SINT`和`VK_FORMAT_R32G32_SFLOAT`

+   `VK_FORMAT_R32G32B32A32_UINT`、`VK_FORMAT_R32G32B32A32_SINT`和`VK_FORMAT_R32G32B32A32_SFLOAT`

+   `VK_FORMAT_B10G11R11_UFLOAT_PACK32`

要检查是否可以使用其他格式与均匀纹理缓冲区一起使用，我们需要准备以下代码：

```cpp
VkFormatProperties format_properties; 
vkGetPhysicalDeviceFormatProperties( physical_device, format, &format_properties ); 
if( !(format_properties.bufferFeatures & VK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT) ) { 
  std::cout << "Provided format is not supported for a uniform texel buffer." << std::endl; 
  return false; 
}

```

如果所选格式适合我们的需求，我们可以创建一个缓冲区，为它分配一个内存对象，并将其绑定到缓冲区。非常重要的一点是，我们还需要创建一个缓冲视图：

```cpp
if( !CreateBuffer( logical_device, size, usage | VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT, uniform_texel_buffer ) ) { 
  return false; 
} 

if( !AllocateAndBindMemoryObjectToBuffer( physical_device, logical_device, uniform_texel_buffer, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, memory_object ) ) { 
  return false; 
} 

if( !CreateBufferView( logical_device, uniform_texel_buffer, format, 0, VK_WHOLE_SIZE, uniform_texel_buffer_view ) ) { 
  return false; 
} 
return true;

```

从 API 的角度来看，缓冲区内容的结构无关紧要。但在均匀纹理缓冲区的情况下，我们需要指定一个数据格式，以便着色器能够以适当的方式解释缓冲区的内容。这就是为什么需要缓冲视图的原因。

在 GLSL 着色器中，均匀纹理缓冲区通过`samplerBuffer`类型的变量（可能带有前缀）定义。

以下是一个在 GLSL 着色器中定义的均匀纹理缓冲区变量的示例：

```cpp
layout (set=m, binding=n) uniform samplerBuffer <variable name>;

```

# 参见

在第四章，*资源和内存*中，查看以下食谱：

+   *创建缓冲区*

+   *分配和绑定内存对象到缓冲区*

+   *创建缓冲视图*

+   *销毁缓冲视图*

+   *释放内存对象*

+   *销毁缓冲*

# 创建存储纹理缓冲区

存储纹理缓冲区，就像均匀纹理缓冲区一样，是一种向着色器提供大量类似图像数据的方式。但它们还允许我们在其中存储数据并对它们执行原子操作。为此，我们需要创建一个具有`VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT`的缓冲区。

# 如何操作...

1.  获取物理设备的句柄。将其存储在名为`physical_device`的`VkPhysicalDevice`类型变量中。

1.  为纹理缓冲区的数据选择一个格式，并使用它初始化一个名为`format`的`VkFormat`类型变量。

1.  创建一个名为`format_properties`的`VkFormatProperties`类型变量。

1.  调用`vkGetPhysicalDeviceFormatProperties( physical_device, format, &format_properties )`并提供所选物理设备的句柄、`format`变量和一个指向`format_properties`变量的指针。

1.  通过检查`format_properties`变量的`bufferFeatures`成员是否设置了`VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_BIT`位，确保所选格式适用于存储纹理缓冲区。

1.  如果将在创建的存储纹理缓冲区上执行原子操作，请确保所选格式也适用于原子操作。为此，检查`format_properties`变量的`bufferFeatures`成员是否设置了`VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_ATOMIC_BIT`位。

1.  从所选物理设备的句柄创建一个逻辑设备句柄。将其存储在名为`logical_device`的`VkDevice`类型变量中。

1.  创建一个名为`storage_texel_buffer`的`VkBuffer`类型变量。

1.  使用 `logical_device` 变量，创建一个具有所选大小和用途的缓冲区。确保在创建缓冲区时指定了 `VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT` 用途。将缓冲区的句柄存储在 `storage_texel_buffer` 变量中（参考第四章 *创建缓冲区* 的配方，*资源和内存*）。

1.  分配一个具有 `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT` 属性的内存对象（或使用现有的一个）并将其绑定到缓冲区。如果分配了新的内存对象，将其存储在名为 `memory_object` 的 `VkDeviceMemory` 类型的变量中（参考第四章 *分配和绑定内存对象到缓冲区* 的配方，*资源和内存*）。

1.  使用 `logical_device`，`storage_texel_buffer` 和 `format` 变量，以及所需的偏移量和内存范围创建一个缓冲区视图，并将结果句柄存储在名为 `storage_texel_buffer_view` 的 `VkBufferView` 类型的变量中（参考第四章 *创建缓冲区视图* 的配方，*资源和内存*）。

# 它是如何工作的...

存储 texel 缓冲区允许我们访问和存储非常大的数组中的数据。数据被解释为如果它是在一维图像内部读取或存储的。此外，我们还可以对这些缓冲区执行原子操作。

存储 texel 缓冲区可以填充类型等于 `VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER` 的描述符。

要将缓冲区用作存储 texel 缓冲区，它需要以 `VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT` 用途创建。还需要一个具有适当格式的缓冲区视图。对于存储 texel 缓冲区，我们可以选择包括以下在内的强制格式之一：

+   `VK_FORMAT_R8G8B8A8_UNORM`，`VK_FORMAT_R8G8B8A8_SNORM`，`VK_FORMAT_R8G8B8A8_UINT` 和 `VK_FORMAT_R8G8B8A8_SINT`

+   `VK_FORMAT_A8B8G8R8_UNORM_PACK32`，`VK_FORMAT_A8B8G8R8_SNORM_PACK32`，`VK_FORMAT_A8B8G8R8_UINT_PACK32` 和 `VK_FORMAT_A8B8G8R8_SINT_PACK32`

+   `VK_FORMAT_R32_UINT`，`VK_FORMAT_R32_SINT` 和 `VK_FORMAT_R32_SFLOAT`

+   `VK_FORMAT_R32G32_UINT`，`VK_FORMAT_R32G32_SINT` 和 `VK_FORMAT_R32G32_SFLOAT`

+   `VK_FORMAT_R32G32B32A32_UINT`，`VK_FORMAT_R32G32B32A32_SINT`，和 `VK_FORMAT_R32G32B32A32_SFLOAT`

对于原子操作，强制格式的列表要短得多，仅包括以下内容：

+   `VK_FORMAT_R32_UINT` 和 `VK_FORMAT_R32_SINT`

其他格式也可能支持存储 texel 缓冲区，但支持并不保证，必须在应用程序执行的平台上进行确认，如下所示：

```cpp
VkFormatProperties format_properties; 
vkGetPhysicalDeviceFormatProperties( physical_device, format, &format_properties ); 
if( !(format_properties.bufferFeatures & VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_BIT) ) { 
  std::cout << "Provided format is not supported for a uniform texel buffer." << std::endl; 
  return false; 
} 

if( atomic_operations && 
    !(format_properties.bufferFeatures & VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_ATOMIC_BIT) ) { 
  std::cout << "Provided format is not supported for atomic operations on storage texel buffers." << std::endl; 
  return false; 
}

```

对于存储 texel 缓冲区，我们需要创建一个缓冲区，为缓冲区分配和绑定一个内存对象，还需要创建一个定义缓冲区数据格式的缓冲区视图：

![](img/image_05_001.png)

```cpp
if( !CreateBuffer( logical_device, size, usage | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT, storage_texel_buffer ) ) { 
  return false; 
} 

if( !AllocateAndBindMemoryObjectToBuffer( physical_device, logical_device, storage_texel_buffer, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, memory_object ) ) { 
  return false; 
} 

if( !CreateBufferView( logical_device, storage_texel_buffer, format, 0, VK_WHOLE_SIZE, storage_texel_buffer_view ) ) { 
  return false; 
} 
return true;

```

我们还可以使用现有的内存对象并将它的内存范围绑定到存储 texel 缓冲区。

从 GLSL 的角度来看，存储纹理缓冲区变量使用`imageBuffer`（可能带有前缀）关键字定义。

在 GLSL 着色器中定义的存储纹理缓冲区的一个例子如下所示：

```cpp
layout (set=m, binding=n, r32f) uniform imageBuffer <variable name>;

```

# 参见

在第四章，*资源和内存*中，查看以下配方：

+   *创建缓冲区*

+   *分配和绑定内存对象到缓冲区*

+   *创建缓冲区视图*

+   *销毁缓冲区视图*

+   *释放内存对象*

+   *销毁缓冲区*

# 创建一个统一的缓冲区

在 Vulkan 中，着色器内部使用的统一变量不能放置在全局命名空间中。它们只能定义在统一缓冲区内部。对于这些，我们需要创建具有`VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT`用途的缓冲区。

# 如何做...

1.  使用创建的逻辑设备及其句柄初始化一个名为`logical_device`的`VkDevice`类型的变量。

1.  创建一个名为`uniform_buffer`的`VkBuffer`类型的变量。它将保存创建的缓冲区的句柄。

1.  使用`logical_device`变量创建一个缓冲区，并指定所需的大小和用途。后者必须包含至少一个`VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT`标志。将缓冲区的句柄存储在`uniform_buffer`变量中（请参阅第四章，*资源和内存*中的*创建缓冲区*配方）。

1.  使用具有`VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT`属性（或使用现有内存对象的范围）的内存对象进行分配，并将其绑定到缓冲区（请参阅第四章，*资源和内存*中的*分配和绑定内存对象到缓冲区*配方）。

# 它是如何工作的...

统一缓冲区用于在着色器内部提供只读统一变量的值。

统一缓冲区可用于`VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER`或`VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC`描述符类型。

通常，统一缓冲区包含不经常更改的参数数据，即矩阵（对于少量数据，建议使用**推送常量**，因为更新它们通常要快得多；有关推送常量的信息，请参阅第九章，*命令记录和绘制*中的*通过推送常量向着色器提供数据*配方）。

创建用于存储统一变量数据的缓冲区需要我们在创建缓冲区时指定`VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT`标志。当缓冲区创建时，我们需要准备一个内存对象并将其绑定到创建的缓冲区（我们也可以使用现有的内存对象并将它的内存存储的一部分绑定到缓冲区）：

```cpp
if( !CreateBuffer( logical_device, size, usage | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, uniform_buffer ) ) { 
  return false; 
} 

if( !AllocateAndBindMemoryObjectToBuffer( physical_device, logical_device, uniform_buffer, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, memory_object ) ) { 
  return false; 
} 
return true;

```

在缓冲区和其内存对象准备就绪后，我们可以像对其他类型的缓冲区一样将数据上传到它们。我们只需记住，统一变量必须放置在适当的偏移量处。这些偏移量与 GLSL 语言中的 std140 布局相同，定义如下：

+   大小为`N`的标量变量必须放置在偏移量为`N`的倍数的位置。

+   一个具有两个组件的向量，其中每个组件的大小为`N`，必须放置在偏移量为`2N`的倍数的位置。

+   一个具有三个或四个组件的向量，其中每个组件的大小为`N`，必须放置在偏移量为`4N`的倍数的位置。

+   一个大小为`N`的数组必须放置在偏移量为`N`的倍数的位置，且向上取整到`16`的倍数。

+   结构必须放置在与其成员的最大偏移量相同的偏移量处，向上取整到`16`的倍数（具有最大偏移量要求的成员的偏移量，向上取整到`16`的倍数）。

+   一个行主序矩阵必须放置在偏移量等于具有与矩阵列数相同组件数的向量的偏移量处。

+   一个列主序矩阵必须放置在其列相同的偏移量处。

动态统一缓冲区与普通统一缓冲区在指定其地址的方式上有所不同。在描述符集更新期间，我们指定用于统一缓冲区的内存大小以及从缓冲区内存开始的偏移量。对于普通统一缓冲区，这些参数保持不变。对于动态统一缓冲区，指定的偏移量成为一个基偏移量，可以在将描述符集绑定到命令缓冲区时通过动态偏移量进行修改。

在 GLSL 着色器内部，统一缓冲区和动态统一缓冲区都使用`uniform`限定符和块语法定义。

以下提供了一个在 GLSL 着色器中定义统一缓冲区的示例：

```cpp
layout (set=m, binding=n) uniform <variable name> 
{ 
  vec4 <member 1 name>; 
  mat4 <member 2 name>; 
  // ... 
};

```

# 参见

在第四章，*资源和内存*中，查看以下配方：

+   *创建缓冲区*

+   *分配和绑定内存对象到缓冲区*

+   *释放内存对象*

+   *销毁缓冲区*

# 创建存储缓冲区

当我们不仅想要从着色器内部的缓冲区中读取数据，还希望在其中存储数据时，我们需要使用存储缓冲区。这些缓冲区使用`VK_BUFFER_USAGE_STORAGE_BUFFER_BIT`用途创建。

# 如何做到这一点...

1.  获取逻辑设备的句柄并将其存储在名为`physical_device`的类型为`VkPhysicalDevice`的变量中。

1.  创建一个名为`storage_buffer`的类型为`VkBuffer`的变量，其中将存储创建的缓冲区的句柄。

1.  使用`logical_device`变量创建一个所需大小和用途的缓冲区。指定的用途必须包含至少一个`VK_BUFFER_USAGE_STORAGE_BUFFER_BIT`标志。将创建的句柄存储在`storage_buffer`变量中（参考第四章，*资源和内存*中的*创建缓冲区*配方）。

1.  分配一个具有 `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT` 属性的内存对象（或使用现有内存对象的范围）并将其绑定到创建的缓冲区（请参阅 第四章，*资源和内存*中的*分配并将内存对象绑定到缓冲区*配方）。

# 它是如何工作的...

存储缓冲区支持读写操作。我们还可以对具有无符号整数格式的存储缓冲区成员执行原子操作。

存储缓冲区对应于 `VK_DESCRIPTOR_TYPE_STORAGE_BUFFER` 或 `VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC` 描述符类型。

存储缓冲区成员的数据必须放置在适当的偏移处。满足要求的最简单方法是遵循 GLSL 语言中 std430 布局的规则。存储缓冲区的基对齐规则与统一缓冲区的规则类似，除了数组和结构--它们的偏移量不需要向上舍入到 16 的倍数。为了方便，这些规则如下指定：

+   大小为 `N` 的标量变量必须放置在 `N` 的倍数偏移处

+   一个有两个分量的向量，其中每个分量的大小为 `N`，必须放置在 `2N` 的倍数偏移处

+   一个有三个或四个分量的向量，其中每个分量的大小为 `N`，必须放置在 `4N` 的倍数偏移处

+   一个大小为 `N` 的元素数组必须放置在 `N` 的倍数偏移处

+   一个结构必须放置在其成员中最大偏移量的倍数偏移处（具有最大偏移量要求的成员）

+   一个行主序矩阵必须放置在一个偏移量等于具有与矩阵列数相同分量的向量偏移量

+   一个列主序矩阵必须放置在其列相同的偏移处

动态存储缓冲区在定义其基内存偏移的方式上有所不同。在描述符集更新期间指定的偏移量和范围对于正常存储缓冲区保持不变，直到下一次更新。对于它们的动态变体，指定的偏移量成为一个基地址，稍后由将描述符集绑定到命令缓冲区时指定的动态偏移量修改。

在 GLSL 着色器中，存储缓冲区和动态存储缓冲区使用 `buffer` 限定符和块语法定义相同。

下面提供了一个在 GLSL 着色器中使用的存储缓冲区的示例：

```cpp
layout (set=m, binding=n) buffer <variable name> 
{ 
  vec4 <member 1 name>; 
  mat4 <member 2 name>; 
  // ... 
};

```

# 参见

在 第四章，*资源和内存*，查看以下配方：

+   *创建缓冲区*

+   *分配并将内存对象绑定到缓冲区*

+   *释放内存对象*

+   *销毁缓冲区*

# 创建输入附件

附件是在绘制命令期间，在渲染通道中渲染到其中的图像。换句话说，它们是渲染目标。

输入附件是我们可以在片段着色器内部读取（未过滤）数据的图像资源。我们只需记住，我们只能访问与处理过的片段相对应的一个位置。

通常，对于输入附件，使用之前用作颜色或深度/模板附件的资源。但我们也可以使用其他图像（及其图像视图）。我们只需使用具有`VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT`使用位的`VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT`来创建它们。

# 如何实现...

1.  获取执行操作的物理设备，并将其句柄存储在名为`physical_device`的`VkPhysicalDevice`类型变量中。

1.  为图像选择一个格式，并使用它初始化一个名为`format`的`VkFormat`类型变量。

1.  创建一个名为`format_properties`的`VkFormatProperties`类型变量。

1.  调用`vkGetPhysicalDeviceFormatProperties( physical_device, format, &format_properties )`，并提供`physical_device`和`format`变量，以及`format_properties`变量的指针。

1.  如果将读取图像的颜色数据，请确保所选格式适合此类使用。为此，请检查`format_properties`变量的`optimalTilingFeatures`成员中是否设置了`VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT`位。

1.  如果将读取图像的深度或模板数据，请检查所选格式是否可用于读取深度或模板数据。通过确保`format_properties`变量的`optimalTilingFeatures`成员中设置了`VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT`位来完成此操作。

1.  从使用的物理设备创建一个逻辑设备，并将其句柄存储在名为`logical_device`的`VkDevice`类型变量中。

1.  使用`logical_device`和`format`变量创建一个图像，并为图像的其余参数选择适当的值。确保在创建图像期间指定`VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT`使用位。将创建的句柄存储在名为`input_attachment`的`VkImage`类型变量中（参考第四章中的*创建一个图像*配方，*资源和内存*）。

1.  分配一个具有`VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT`属性的内存对象（或使用现有内存对象的范围），并将其绑定到图像上（参考第四章*资源和内存*中的*分配和绑定内存对象到图像*配方）。

1.  使用`logical_device`、`input_attachment`和`format`变量创建一个图像视图，并选择图像视图的其余参数。将创建的句柄存储在名为`input_attachment_image_view`的`VkImageView`类型变量中（参考第四章中的*创建图像视图*配方，*资源和内存*）。

# 它是如何工作的...

输入附件使我们能够从用作渲染通道附件的图像中读取片段着色器内的数据（通常，对于输入附件，将使用之前用作颜色或深度/模板附件的图像）。

输入附件用于 `VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT` 类型的描述符。

在 Vulkan 中，渲染操作被收集到渲染通道中。每个渲染通道至少有一个子通道，但可以有更多。如果我们在一个子通道中将渲染到附件，然后我们可以将其用作输入附件，并在同一渲染通道的后续子通道中从中读取数据。实际上，这是从给定渲染通道的附件中读取数据的唯一方法--在给定渲染通道中作为附件的图像只能通过着色器中的输入附件访问（它们不能绑定到描述符集用于除输入附件以外的目的）。

当从输入附件读取数据时，我们仅限于对应于处理片段位置的地点。但这种方法可能比渲染到附件、结束渲染通道、将图像绑定到描述符集作为采样图像（纹理）并开始另一个不使用给定图像作为任何附件的渲染通道更优。

对于输入附件，我们还可以使用其他图像（我们不必将它们用作颜色或深度/模板附件）。我们只需用 `VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT` 用法和适当的格式创建它们。以下格式对于从读取颜色数据的输入附件是强制性的：

+   `VK_FORMAT_R5G6B5_UNORM_PACK16`

+   `VK_FORMAT_A1R5G5B5_UNORM_PACK16`

+   `VK_FORMAT_R8_UNORM`、`VK_FORMAT_R8_UINT` 和 `VK_FORMAT_R8_SINT`

+   `VK_FORMAT_R8G8_UNORM`、`VK_FORMAT_R8G8_UINT` 和 `VK_FORMAT_R8G8_SINT`

+   `VK_FORMAT_R8G8B8A8_UNORM`、`VK_FORMAT_R8G8B8A8_UINT`、`VK_FORMAT_R8G8B8A8_SINT` 和 `VK_FORMAT_R8G8B8A8_SRGB`

+   `VK_FORMAT_B8G8R8A8_UNORM` 和 `VK_FORMAT_B8G8R8A8_SRGB`

+   `VK_FORMAT_A8B8G8R8_UNORM_PACK32`、`VK_FORMAT_A8B8G8R8_UINT_PACK32`、`VK_FORMAT_A8B8G8R8_SINT_PACK32` 和 `VK_FORMAT_A8B8G8R8_SRGB_PACK32`

+   `VK_FORMAT_A2B10G10R10_UNORM_PACK32` 和 `VK_FORMAT_A2B10G10R10_UINT_PACK32`

+   `VK_FORMAT_R16_UINT`、`VK_FORMAT_R16_SINT` 和 `VK_FORMAT_R16_SFLOAT`

+   `VK_FORMAT_R16G16_UINT`、`VK_FORMAT_R16G16_SINT` 和 `VK_FORMAT_R16G16_SFLOAT`

+   `VK_FORMAT_R16G16B16A16_UINT`、`VK_FORMAT_R16G16B16A16_SINT` 和 `VK_FORMAT_R16G16B16A16_SFLOAT`

+   `VK_FORMAT_R32_UINT`、`VK_FORMAT_R32_SINT` 和 `VK_FORMAT_R32_SFLOAT`

+   `VK_FORMAT_R32G32_UINT`、`VK_FORMAT_R32G32_SINT` 和 `VK_FORMAT_R32G32_SFLOAT`

+   `VK_FORMAT_R32G32B32A32_UINT`、`VK_FORMAT_R32G32B32A32_SINT` 和 `VK_FORMAT_R32G32B32A32_SFLOAT`

对于将读取深度/模板数据的输入附件，以下格式是强制性的：

+   `VK_FORMAT_D16_UNORM`

+   `VK_FORMAT_X8_D24_UNORM_PACK32` 或 `VK_FORMAT_D32_SFLOAT`（至少必须支持这两种格式之一）

+   `VK_FORMAT_D24_UNORM_S8_UINT` 或 `VK_FORMAT_D32_SFLOAT_S8_UINT`（至少必须支持这两种格式之一）

其他格式也可能被支持，但对它们的支持不能保证。我们可以检查在应用程序执行的平台上的给定格式是否被支持，如下所示：

```cpp
VkFormatProperties format_properties; 
vkGetPhysicalDeviceFormatProperties( physical_device, format, &format_properties ); 
if( (aspect & VK_IMAGE_ASPECT_COLOR_BIT) && 
    !(format_properties.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT) ) { 
  std::cout << "Provided format is not supported for an input attachment." << std::endl; 
  return false; 
} 
if( (aspect & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_DEPTH_BIT)) && 
  !(format_properties.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) ) { 
  std::cout << "Provided format is not supported for an input attachment." << std::endl; 
  return false; 
}

```

接下来，我们只需要创建一个图像，分配一个内存对象（或使用现有的一个）并将其绑定到图像上，然后创建一个图像视图。我们可以这样做：

```cpp
if( !CreateImage( logical_device, type, format, size, 1, 1, VK_SAMPLE_COUNT_1_BIT, usage | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT, false, input_attachment ) ) { 
  return false; 
} 

if( !AllocateAndBindMemoryObjectToImage( physical_device, logical_device, input_attachment, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, memory_object ) ) { 
  return false; 
} 

if( !CreateImageView( logical_device, input_attachment, view_type, format, aspect, input_attachment_image_view ) ) { 
  return false; 
} 
return true;

```

以这种方式创建的图像及其视图可以用作输入附件。为此，我们需要准备一个适当的渲染通道描述，并将图像视图包含在帧缓冲区中（参考第六章，*渲染通道和帧缓冲区*中的*指定子通道描述*和*创建帧缓冲区*配方）。

在 GLSL 着色器代码中，使用`subpassInput`（可能带有前缀）关键字定义引用输入附件的变量。

以下是一个在 GLSL 中定义的输入附件的示例：

```cpp
layout (input_attachment_index=i, set=m, binding=n) uniform subpassInput <variable name>;

```

# 参见

在第四章，*资源和内存*，查看以下配方：

+   *创建一个图像*

+   *将内存对象分配和绑定到图像上*

+   *创建一个图像视图*

+   *销毁一个图像视图*

+   *销毁一个图像*

+   *释放内存对象*

在第六章，*渲染通道和**帧缓冲区*，查看以下配方：

+   *指定子通道描述*

+   *创建一个帧缓冲区*

# 创建描述符集布局

描述符集在一个对象中聚集了许多资源（描述符）。它们稍后绑定到管道以建立我们的应用程序和着色器之间的接口。但是，为了让硬件知道哪些资源被分组在一个集中，每种类型的资源有多少个，以及它们的顺序是什么，我们需要创建一个描述符集布局。

# 如何做...

1.  获取逻辑设备的句柄并将其分配给名为`logical_device`的`VkDevice`类型变量。

1.  创建一个元素类型为`VkDescriptorSetLayoutBinding`的向量变量，并命名为`bindings`。

1.  对于您想要创建并稍后分配给给定描述符集的每个资源，向`bindings`向量中添加一个元素。为每个新元素的成员使用以下值：

    +   给定资源在描述符集中的选择索引用于`binding`

    +   给定资源的期望类型用于`descriptorType`

    +   通过着色器内部数组访问的指定类型的资源数量（如果给定资源不是通过数组访问，则为 1）用于`descriptorCount`

    +   资源将被访问的所有着色器阶段的逻辑“或”用于`stageFlags`

    +   `nullptr`的值用于`pImmutableSamplers`

1.  创建一个名为`descriptor_set_layout_create_info`的`VkDescriptorSetLayoutCreateInfo`类型的变量。使用以下值初始化其成员：

    +   `VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO`的值用于`sType`

    +   `nullptr`的值用于`pNext`

    +   `0`的值用于`flags`

    +   `bindings`向量中元素的数量用于`bindingCount`

    +   指向`bindings`向量第一个元素的指针用于`pBindings`

1.  创建一个名为`descriptor_set_layout`的`VkDescriptorSetLayout`类型的变量，其中将存储创建的布局。

1.  调用`vkCreateDescriptorSetLayout(logical_device, &descriptor_set_layout_create_info, nullptr, &descriptor_set_layout)`并提供逻辑设备的句柄、`descriptor_set_layout_create_info`变量的指针、一个`nullptr`值以及`descriptor_set_layout`变量的指针。

1.  通过检查返回值是否等于`VK_SUCCESS`来确保调用成功。

# 它是如何工作的...

描述符集布局指定了描述符集的内部结构，同时严格定义了可以绑定到描述符集上的资源（我们不能使用布局中未指定的资源）。

当我们想要创建布局时，我们需要知道将使用哪些资源（描述符类型）以及它们的顺序。顺序通过绑定来指定——它们定义了资源在给定集中的索引（位置），并在着色器内部（通过`layout`限定符的集合号）用于指定我们想要访问的资源：

```cpp
layout (set=m, binding=n) // variable definition

```

我们可以为绑定选择任何值，但我们应该记住，未使用的索引可能会消耗内存并影响我们应用程序的性能。

为了避免不必要的内存开销和负面的性能影响，我们应该保持描述符绑定尽可能紧凑，尽可能接近`0`。

要创建描述符集布局，我们首先需要指定给定集中使用的所有资源的列表：

```cpp
VkDescriptorSetLayoutCreateInfo descriptor_set_layout_create_info = { 
  VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, 
  nullptr, 
  0, 
  static_cast<uint32_t>(bindings.size()), 
  bindings.data() 
};

```

接下来，我们可以创建如下布局：

```cpp
VkResult result = vkCreateDescriptorSetLayout( logical_device, &descriptor_set_layout_create_info, nullptr, &descriptor_set_layout ); 
if( VK_SUCCESS != result ) { 
  std::cout << "Could not create a layout for descriptor sets." << std::endl; 
  return false; 
} 
return true;

```

描述符集布局（以及推送常量范围）也形成了一个管线布局，它定义了给定管线可以访问的资源类型。除了管线布局创建之外，创建的布局在描述符集分配期间也是必需的。

# 参见

+   在第八章“图形和计算管线”中，查看以下配方：

    +   *创建管线布局*

+   在本章中，查看以下配方：

    +   *分配描述符集*

# 创建描述符池

描述符，收集到集合中，是从描述符池中分配的。当我们创建一个池时，我们必须定义哪些描述符，以及它们中可以有多少可以从创建的池中分配。

# 如何做到...

1.  获取应该在之上创建描述符池的逻辑设备的句柄。将其存储在名为`logical_device`的`VkDevice`类型变量中。

1.  创建一个名为`descriptor_types`的向量变量，其元素类型为`VkDescriptorPoolSize`。对于将从池中分配的每种描述符类型，向`descriptor_types`变量添加一个新元素，定义指定的描述符类型以及将从池中分配的给定类型的描述符数量。

1.  创建一个名为`descriptor_pool_create_info`的`VkDescriptorPoolCreateInfo`类型的变量。为此变量的成员使用以下值：

    +   `VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO`的`sType`值

    +   `pNext`的值为`nullptr`

    +   如果应该可能释放从该池分配的单独的集，则`VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT`值为`1`，或者为`0`值以仅允许通过池重置操作一次性释放所有集（对于`flags`）。

    +   池中可以分配的最大集数`maxSets`

    +   `poolSizeCount`的`descriptor_types`向量中的元素数量

    +   `pPoolSizes`的`descriptor_types`向量的第一个元素的指针

1.  创建一个名为`descriptor_pool`的`VkDescriptorPool`类型的变量，其中将存储创建的池的句柄。

1.  调用`vkCreateDescriptorPool(logical_device, &descriptor_pool_create_info, nullptr, &descriptor_pool)`并提供`logical_device`变量、`descriptor_pool_create_info`变量的指针、一个`nullptr`值和一个指向`descriptor_pool`变量的指针。

1.  确保通过检查调用是否返回了`VK_SUCCESS`值来确认池是否成功创建。

# 它是如何工作的...

描述符池管理用于分配描述符集的资源（类似于命令池管理命令缓冲区的内存）。在创建描述符池期间，我们指定可以从给定池中分配的最大集数和可以跨所有集分配的给定类型的最大描述符数。此信息通过类型为`VkDescriptorPoolCreateInfo`的变量提供，如下所示：

```cpp
VkDescriptorPoolCreateInfo descriptor_pool_create_info = { 
  VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, 
  nullptr, 
  free_individual_sets ? 
    VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT : 0, 
  max_sets_count, 
  static_cast<uint32_t>(descriptor_types.size()), 
  descriptor_types.data() 
};

```

在前面的示例中，描述符的类型及其总数是通过`descriptor_types`向量变量提供的。它可能包含多个元素，创建的池将足够大，可以分配所有指定的描述符。

池本身创建方式如下：

```cpp
VkResult result = vkCreateDescriptorPool( logical_device, &descriptor_pool_create_info, nullptr, &descriptor_pool ); 
if( VK_SUCCESS != result ) { 
  std::cout << "Could not create a descriptor pool." << std::endl; 
  return false; 
} 
return true;

```

当我们创建了一个池，我们可以从中分配描述符集。但我们必须记住，我们不能同时以多线程的方式做这件事。

我们不能在多个线程中同时从给定的池中分配描述符集。

# 参见

在本章中查看以下食谱：

+   *分配描述符集*

+   *释放描述符集*

+   *重置描述符池*

+   *销毁描述符池*

# 分配描述符集

描述符集在一个容器对象中聚集着着色器资源（描述符）。其内容、类型和资源数量由描述符集布局定义；存储是从池中获取的，我们可以从池中分配描述符集。

# 如何操作...

1.  取逻辑设备并将其句柄存储在名为`logical_device`的`VkDevice`类型的变量中。

1.  准备一个描述符池，从该池中分配描述符集。使用池的句柄初始化一个名为`descriptor_pool`的`VkDescriptorPool`类型的变量。

1.  创建一个名为`descriptor_set_layouts`的`std::vector<VkDescriptorSetLayout>`类型的变量。对于应该从池中分配的每个描述符集，添加一个定义相应描述符集结构的描述符集布局句柄。

1.  创建一个名为 `descriptor_set_allocate_info` 的类型为 `VkDescriptorSetAllocateInfo` 的变量，并为其成员使用以下值：

    +   `VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO` 的值用于 `sType`

    +   `pNext` 的值为 `nullptr`

    +   `descriptor_pool` 变量用于 `descriptorPool`

    +   `descriptor_set_layouts` 向量中 `descriptorSetCount` 的元素数量

    +   `descriptorSetCount` 向量的第一个元素指针用于 `pSetLayouts`

1.  创建一个名为 `descriptor_sets` 的类型为 `std::vector<VkDescriptorSet>` 的向量变量，并将其大小调整为与 `descriptor_set_layouts` 向量的大小相匹配。

1.  调用 `vkAllocateDescriptorSets( logical_device, &descriptor_set_allocate_info, &descriptor_sets[0] )` 并提供 `logical_device` 变量、`descriptor_set_allocate_info` 变量的指针以及 `descriptor_sets` 向量第一个元素的指针。

1.  确保调用成功并返回了 `VK_SUCCESS` 值。

# 它是如何工作的...

描述符集用于向着色器提供资源。它们在应用程序和可编程管道阶段之间形成一个接口。该接口的结构由描述符集布局定义。实际数据是在我们使用图像或缓冲区资源更新描述符集时提供，并在记录操作期间将这些描述符集绑定到命令缓冲区时提供。

描述符集是从池中分配的。当我们创建池时，我们指定可以从池中分配多少描述符（资源）以及其类型，以及可以从中分配的最大描述符集数量。

当我们想要分配描述符集时，我们需要指定将描述其内部结构的布局——每个描述符集一个布局。此信息指定如下：

```cpp
VkDescriptorSetAllocateInfo descriptor_set_allocate_info = { 
  VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, 
  nullptr, 
  descriptor_pool, 
  static_cast<uint32_t>(descriptor_set_layouts.size()), 
  descriptor_set_layouts.data() 
};

```

接下来，我们按照以下方式分配描述符集：

```cpp
descriptor_sets.resize( descriptor_set_layouts.size() ); 

VkResult result = vkAllocateDescriptorSets( logical_device, &descriptor_set_allocate_info, descriptor_sets.data() ); 
if( VK_SUCCESS != result ) { 
  std::cout << "Could not allocate descriptor sets." << std::endl; 
  return false; 
} 
return true;

```

不幸的是，当我们分配和释放单独的描述符集时，池的内存可能会变得碎片化。在这种情况下，我们可能无法从给定的池中分配新的集合，即使我们没有达到指定的限制。这种情况在以下图中展示：

![](img/B05542-05-02.png)

当我们首次分配描述符集时，不会出现碎片化问题。此外，如果所有描述符集使用相同类型的资源数量，则可以保证不会出现此类问题。

为了避免池碎片化问题，我们可以一次性释放所有描述符集（通过重置池）。否则，如果我们无法分配新的描述符集，并且不想重置池，我们需要创建另一个池。

# 参见

在本章中查看以下食谱：

+   *创建描述符集布局*

+   *创建描述符池*

+   *释放描述符集*

+   *重置描述符池*

# 更新描述符集

我们已经创建了一个描述符池，并从中分配了描述符集。由于创建了布局，我们知道了它们的内部结构。现在我们想要提供特定的资源（采样器、图像视图、缓冲区或缓冲区视图），这些资源稍后应通过描述符集绑定到管线。定义要使用的资源是通过更新描述符集的过程完成的。

# 准备工作

更新描述符集需要我们为每个参与过程描述符提供相当数量的数据。更重要的是，提供的数据取决于描述符的类型。为了简化过程并减少需要指定的参数数量，以及为了提高错误检查，在此配方中引入了自定义结构。

对于采样器和各种图像描述符，使用 `ImageDescriptorInfo` 类型，其定义如下：

```cpp
struct ImageDescriptorInfo { 
  VkDescriptorSet                     TargetDescriptorSet; 
  uint32_t                            TargetDescriptorBinding; 
  uint32_t                            TargetArrayElement; 
  VkDescriptorType                    TargetDescriptorType; 
  std::vector<VkDescriptorImageInfo>  ImageInfos; 
};

```

对于统一和存储缓冲区（及其动态变体），使用 `BufferDescriptorInfo` 类型。其定义如下：

```cpp
struct BufferDescriptorInfo { 
  VkDescriptorSet                     TargetDescriptorSet; 
  uint32_t                            TargetDescriptorBinding; 
  uint32_t                            TargetArrayElement; 
  VkDescriptorType                    TargetDescriptorType; 
  std::vector<VkDescriptorBufferInfo> BufferInfos; 
};

```

对于统一和存储纹理缓冲区，引入了 `TexelBufferDescriptorInfo` 类型，其定义如下：

```cpp
struct TexelBufferDescriptorInfo { 
  VkDescriptorSet                     TargetDescriptorSet; 
  uint32_t                            TargetDescriptorBinding; 
  uint32_t                            TargetArrayElement; 
  VkDescriptorType                    TargetDescriptorType; 
  std::vector<VkBufferView>           TexelBufferViews; 
};

```

当我们想要使用新描述符的句柄更新描述符集时，使用前面的结构。也可以从其他已更新的集中复制描述符数据。为此，使用 `CopyDescriptorInfo` 类型，其定义如下：

```cpp
struct CopyDescriptorInfo { 
  VkDescriptorSet     TargetDescriptorSet; 
  uint32_t            TargetDescriptorBinding; 
  uint32_t            TargetArrayElement; 
  VkDescriptorSet     SourceDescriptorSet; 
  uint32_t            SourceDescriptorBinding; 
  uint32_t            SourceArrayElement; 
  uint32_t            DescriptorCount; 
};

```

所有前面的结构定义了应更新的描述符集的句柄、给定集中描述符的索引，以及如果我们想通过数组访问描述符，则数组中的索引。其余参数是类型特定的。

# 如何操作...

1.  使用逻辑设备的句柄初始化一个名为 `logical_device` 的 `VkDevice` 类型的变量。

1.  创建一个名为 `write_descriptors` 的 `std::vector<VkWriteDescriptorSet>` 类型的变量。对于每个需要更新的新描述符，向向量中添加一个新元素，并为其成员使用以下值：

    +   `VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET` 的 `sType` 值

    +   `pNext` 的值为 `nullptr`

    +   应更新的描述符集的句柄为 `dstSet`

    +   在指定集中描述符的索引（绑定）为 `dstBinding`

    +   如果在着色器内部通过数组访问给定的描述符，则从数组中更新的描述符的起始索引为 `dstArrayElement`（否则为 `0` 值）

    +   对于 `descriptorCount`，要更新的描述符数量（`pImageInfo`、`pBufferInfo` 或 `pTexelBufferView` 数组中的元素数量）

    +   描述符的类型为 `descriptorType`

    +   在采样器或图像描述符的情况下，指定一个包含 `descriptorCount` 个元素的数组，并在 `pImageInfo` 中提供其第一个元素的指针（将 `pBufferInfo` 和 `pTexelBufferView` 成员设置为 `nullptr`）。为每个数组元素使用以下值：

        +   在采样器描述符的情况下，用于`sampler`的组合图像采样器描述符的采样器句柄

        +   在采样图像、存储图像、组合图像采样器和输入附件描述符的情况下，用于`imageView`的图像视图句柄

        +   当通过着色器访问描述符时，给定图像将处于的布局情况，对于图像描述符的`imageLayout`

    +   在均匀或存储缓冲区（及其动态变体）的情况下，指定一个包含`descriptorCount`个元素的数组，并在`pBufferInfo`中提供其第一个元素的指针（将`pImageInfo`和`pTexelBufferView`成员设置为`nullptr`），并为每个数组元素使用以下值：

        +   缓冲区的句柄用于`buffer`

        +   缓冲区内的内存偏移量（或动态描述符的基偏移量）用于`offset`

        +   对于给定描述符的`range`，应使用的缓冲区内存大小

    +   在均匀的 texel 缓冲区或存储 texel 缓冲区的情况下，指定一个包含`descriptorCount`个 texel 视图句柄的数组，并在`pTexelBufferView`中提供其第一个元素的指针（将`pImageInfo`和`pBufferInfo`成员设置为`nullptr`）。

1.  创建一个名为`copy_descriptors`的`std::vector<VkCopyDescriptorSet>`类型的变量。对于应从另一个已更新的描述符复制的数据，向此向量添加一个元素。为每个新元素的成员使用以下值：

    +   `VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET`值用于`sType`

    +   `nullptr`值用于`pNext`

    +   应从其中复制数据的描述符集的句柄用于`srcSet`

    +   源描述符集中用于`srcBinding`的绑定编号

    +   在源描述符集中用于`srcArrayElement`的数组索引

    +   应在其中更新数据的描述符集的句柄用于`dstSet`

    +   目标描述符集中用于`dstBinding`的绑定编号

    +   在目标描述符集中用于`dstArrayElement`的数组索引

    +   应从源集复制并更新到目标集的描述符数量用于`descriptorCount`

1.  调用`vkUpdateDescriptorSets(logical_device, static_cast<uint32_t>(write_descriptors.size()), &write_descriptors[0], static_cast<uint32_t>(copy_descriptors.size()), &copy_descriptors[0])`并提供`logical_device`变量、`write_descriptors`向量的元素数量、`write_descriptors`的第一个元素的指针、`copy_descriptors`向量的元素数量和`copy_descriptors`向量的第一个元素的指针。

# 它是如何工作的...

更新描述符集会导致指定的资源（采样器、图像视图、缓冲区或缓冲区视图）填充指示集中的条目。当更新的集被绑定到管线时，这些资源可以通过着色器访问。

我们可以将新的（尚未使用）资源写入描述符集。在以下示例中，我们通过使用*准备就绪*部分中提到的自定义结构来实现这一点：

```cpp
std::vector<VkWriteDescriptorSet> write_descriptors; 

for( auto & image_descriptor : image_descriptor_infos ) { 
  write_descriptors.push_back( { 
    VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, 
    nullptr, 
    image_descriptor.TargetDescriptorSet, 
    image_descriptor.TargetDescriptorBinding, 
    image_descriptor.TargetArrayElement, 
    static_cast<uint32_t>(image_descriptor.ImageInfos.size()), 
    image_descriptor.TargetDescriptorType, 
    image_descriptor.ImageInfos.data(), 
    nullptr, 
    nullptr 
  } ); 
} 

for( auto & buffer_descriptor : buffer_descriptor_infos ) { 
  write_descriptors.push_back( { 
    VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, 
    nullptr, 
    buffer_descriptor.TargetDescriptorSet, 
    buffer_descriptor.TargetDescriptorBinding, 
    buffer_descriptor.TargetArrayElement, 
    static_cast<uint32_t>(buffer_descriptor.BufferInfos.size()), 
    buffer_descriptor.TargetDescriptorType, 
    nullptr, 
    buffer_descriptor.BufferInfos.data(), 
    nullptr 
  } ); 
} 

for( auto & texel_buffer_descriptor : texel_buffer_descriptor_infos ) { 
  write_descriptors.push_back( { 
    VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, 
    nullptr, 
    texel_buffer_descriptor.TargetDescriptorSet, 
    texel_buffer_descriptor.TargetDescriptorBinding, 
    texel_buffer_descriptor.TargetArrayElement, 
    static_cast<uint32_t>(texel_buffer_descriptor.TexelBufferViews.size()), 
    texel_buffer_descriptor.TargetDescriptorType, 
    nullptr, 
    nullptr, 
    texel_buffer_descriptor.TexelBufferViews.data() 
  } ); 
}

```

我们还可以重用其他集的描述符。复制已填充的描述符应该比编写新的描述符更快。这可以通过以下方式完成：

```cpp
std::vector<VkCopyDescriptorSet> copy_descriptors; 

for( auto & copy_descriptor : copy_descriptor_infos ) { 
  copy_descriptors.push_back( { 
    VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET, 
    nullptr, 
    copy_descriptor.SourceDescriptorSet, 
    copy_descriptor.SourceDescriptorBinding, 
    copy_descriptor.SourceArrayElement, 
    copy_descriptor.TargetDescriptorSet, 
    copy_descriptor.TargetDescriptorBinding, 
    copy_descriptor.TargetArrayElement, 
    copy_descriptor.DescriptorCount 
  } ); 
}

```

更新描述符集的操作通过单个函数调用执行：

```cpp
vkUpdateDescriptorSets( logical_device, static_cast<uint32_t>(write_descriptors.size()), write_descriptors.data(), static_cast<uint32_t>(copy_descriptors.size()), copy_descriptors.data() );

```

# 参见

参见本章中的以下配方：

+   *分配描述符集*

+   *绑定描述符集*

+   *使用纹理和统一缓冲区创建描述符*

# 绑定描述符集

当描述符集准备就绪（我们已使用将在着色器中访问的所有资源更新了它）时，我们需要在记录操作期间将其绑定到命令缓冲区。

# 如何做到...

1.  捕获正在记录的命令缓冲区的句柄。将句柄存储在名为 `command_buffer` 的 `VkCommandBuffer` 类型变量中。

1.  创建一个名为 `pipeline_type` 的 `VkPipelineBindPoint` 类型变量，它将表示描述符集将使用的管道类型（图形或计算）。

1.  获取管道布局并将句柄存储在名为 `pipeline_layout` 的 `VkPipelineLayout` 类型变量中（参考第八章 *创建管道布局* 配方，*图形和计算管道*）。

1.  创建一个名为 `descriptor_sets` 的 `std::vector<VkDescriptorSet>` 类型变量。对于每个需要绑定到管道的描述符集，向向量中添加一个新元素，并用描述符集的句柄初始化它。

1.  选择一个索引，将提供的列表中的第一个集绑定到该索引。将索引存储在名为 `index_for_first_set` 的 `uint32_t` 类型变量中。

1.  如果在任何被绑定的集中使用了动态统一或存储缓冲区，创建一个名为 `dynamic_offsets` 的 `std::vector<uint32_t>` 类型变量，通过它提供所有被绑定的集中定义的每个动态描述符的内存偏移量。偏移量必须在每个集的布局中按其对应的描述符出现的顺序定义（按递增的绑定顺序）。

1.  执行以下调用：

```cpp
      vkCmdBindDescriptorSets( command_buffer, pipeline_type, 
      pipeline_layout, index_for_first_set, static_cast<uint32_t>
      (descriptor_sets.size()), descriptor_sets.data(), 
      static_cast<uint32_t>(dynamic_offsets.size()), 
      dynamic_offsets.data() )

```

对于此调用，提供 `command_buffer`、`pipeline_type`、`pipeline_layout` 和 `index_for_first_set` 变量，元素数量以及 `descriptor_sets` 向量第一个元素指针，以及元素数量和 `dynamic_offsets` 向量第一个元素指针。

# 它是如何工作的...

当我们开始记录命令缓冲区时，其状态（几乎全部）是未定义的。因此，在我们能够记录引用图像或缓冲区资源的绘制操作之前，我们需要将适当的资源绑定到命令缓冲区。这是通过使用 `vkCmdBindDescriptorSets()` 函数调用绑定描述符集来完成的，如下所示：

```cpp
vkCmdBindDescriptorSets( command_buffer, pipeline_type, pipeline_layout, index_for_first_set, static_cast<uint32_t>(descriptor_sets.size()), descriptor_sets.data(), static_cast<uint32_t>(dynamic_offsets.size()), dynamic_offsets.data() )

```

# 参见

参见本章中的以下配方：

+   *创建描述符集布局*

+   *分配描述符集*

+   *更新描述符集*

# 使用纹理和统一缓冲区创建描述符

在本示例配方中，我们将了解如何创建最常用的资源：组合图像采样器和统一缓冲区。我们将为它们准备描述符集布局，创建描述符池，并从中分配描述符集。然后我们将使用创建的资源更新分配的集。这样，我们就可以稍后绑定描述符集到命令缓冲区，并在着色器中访问资源。

# 如何操作...

1.  使用所选参数创建一个组合图像采样器（一个图像、图像视图和一个采样器）--最常用的参数包括`VK_IMAGE_TYPE_2D`图像类型、`VK_FORMAT_R8G8B8A8_UNORM`格式、`VK_IMAGE_VIEW_TYPE_2D`视图类型、`VK_IMAGE_ASPECT_COLOR_BIT`属性、`VK_FILTER_LINEAR`过滤器模式以及所有纹理坐标的`VK_SAMPLER_ADDRESS_MODE_REPEAT`寻址模式。将创建的句柄存储在名为`sampler`的`VkSampler`类型变量中、名为`sampled_image`的`VkImage`类型变量中，以及名为`sampled_image_view`的`VkImageView`类型变量中（参考*创建组合图像采样器*配方）。

1.  使用所选参数创建一个统一缓冲区，并将缓冲区的句柄存储在名为`uniform_buffer`的`VkBuffer`类型变量中（参考*创建统一缓冲区*配方）。

1.  创建一个名为`bindings`的`std::vector<VkDescriptorSetLayoutBinding>`类型变量。

1.  向`bindings`变量添加一个具有以下值的元素：

    +   `binding`的值为`0`。

    +   `descriptorType`的值为`VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER`。

    +   `descriptorCount`的值为`1`。

    +   `stageFlags`的值为`VK_SHADER_STAGE_FRAGMENT_BIT`。

    +   `stageFlags`的值为`nullptr`。

1.  向`bindings`向量添加另一个元素，并为其成员使用以下值：

    +   `binding`的值为`1`。

    +   `descriptorType`的值为`VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER`。

    +   `descriptorCount`的值为`1`。

    +   `stageFlags`的值为`VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT`。

    +   `pImmutableSamplers`的值为`nullptr`。

1.  使用`bindings`变量创建一个描述符集布局，并将句柄存储在名为`descriptor_set_layout`的`VkDescriptorSetLayout`类型变量中（参考*创建描述符集布局*配方）。

1.  创建一个名为`descriptor_types`的`std::vector<VkDescriptorPoolSize>`类型变量。向创建的向量添加两个元素：一个具有`VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER`和`1`的值，另一个具有`VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER`和`1`的值。

1.  创建一个描述符池，其中单独的描述符集不能单独释放，只能分配一个描述符集。在创建池时使用`descriptor_types`变量，并将句柄存储在名为`descriptor_pool`的`VkDescriptorPool`类型变量中（参考*创建描述符池*配方）。

1.  使用 `descriptor_pool` 和 `descriptor_set_layout` 布局变量分配一个描述符集。将创建的句柄存储在名为 `descriptor_sets` 的 `std::vector<VkDescriptorSet>` 类型的单元素向量中（参见图 *分配描述符集* 食谱）。

1.  创建一个名为 `image_descriptor_infos` 的 `std::vector<ImageDescriptorInfo>` 类型的变量。向此向量添加一个元素，并使用以下值：

    +   `TargetDescriptorSet` 的 `descriptor_sets[0]`

    +   `TargetDescriptorBinding` 的 `0` 值

    +   `TargetArrayElement` 的 `0` 值

    +   `VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER` 的值用于 `TargetDescriptorType`

    +   向 `ImageInfos` 成员向量添加一个元素，并使用以下值：

        +   `sampler` 的变量用于 `sampler`

        +   为 `imageView` 创建 `sampled_image_view` 变量

        +   `imageLayout` 的 `VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL` 值

1.  创建一个名为 `buffer_descriptor_infos` 的 `std::vector<BufferDescriptorInfo>` 类型的变量，并使用以下值初始化其一个元素：

    +   `TargetDescriptorSet` 的 `descriptor_sets[0]`

    +   `TargetDescriptorBinding` 的 `1` 值

    +   `TargetArrayElement` 的 `0` 值

    +   `VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER` 的值用于 `TargetDescriptorType`

    +   向 `BufferInfos` 成员向量添加一个元素，并使用以下值初始化其成员：

        +   `buffer` 的 `uniform_buffer` 变量

        +   `offset` 的 `0` 值

        +   `range` 的 `VK_WHOLE_SIZE` 值

1.  使用 `image_descriptor_infos` 和 `buffer_descriptor_infos` 向量更新描述符集。

# 它是如何工作的...

为了准备通常使用的描述符，一个组合图像采样器和统一缓冲区，我们首先需要创建它们：

```cpp
if( !CreateCombinedImageSampler( physical_device, logical_device, VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, sampled_image_size, 1, 1, VK_IMAGE_USAGE_TRANSFER_DST_BIT, 
  VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_REPEAT, 
  VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_SAMPLER_ADDRESS_MODE_REPEAT, 0.0f, false, 1.0f, false, VK_COMPARE_OP_ALWAYS, 0.0f, 0.0f, VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK, false, 
  sampler, sampled_image, sampled_image_memory_object, sampled_image_view ) ) { 
  return false; 
} 

if( !CreateUniformBuffer( physical_device, logical_device, uniform_buffer_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, uniform_buffer, uniform_buffer_memory_object ) ) { 
  return false; 
}

```

接下来，我们准备一个将定义描述符集内部结构的布局：

```cpp
std::vector<VkDescriptorSetLayoutBinding> bindings = { 
  { 
    0, 
    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 
    1, 
    VK_SHADER_STAGE_FRAGMENT_BIT, 
    nullptr 
  }, 
  { 
    1, 
    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 
    1, 
    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 
    nullptr 
  } 
}; 
if( !CreateDescriptorSetLayout( logical_device, bindings, descriptor_set_layout ) ) { 
  return false; 
}

```

之后，我们创建一个描述符池并从其中分配一个描述符集：

```cpp
std::vector<VkDescriptorPoolSize> descriptor_types = { 
  { 
    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 
    1 
  }, 
  { 
    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 
    1 
  } 
}; 
if( !CreateDescriptorPool( logical_device, false, 1, descriptor_types, descriptor_pool ) ) { 
  return false; 
} 

if( !AllocateDescriptorSets( logical_device, descriptor_pool, { descriptor_set_layout }, descriptor_sets ) ) { 
  return false; 
}

```

最后一件要做的事情是使用最初创建的资源更新描述符集：

```cpp
std::vector<ImageDescriptorInfo> image_descriptor_infos = { 
  { 
    descriptor_sets[0], 
    0, 
    0, 
    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 
    { 
      { 
        sampler, 
        sampled_image_view, 
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL 
      } 
    } 
  } 
}; 

std::vector<BufferDescriptorInfo> buffer_descriptor_infos = { 
  { 
    descriptor_sets[0], 
    1, 
    0, 
    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 
    { 
      { 
        uniform_buffer, 
        0, 
        VK_WHOLE_SIZE 
      } 
    } 
  } 
}; 

UpdateDescriptorSets( logical_device, image_descriptor_infos, buffer_descriptor_infos, {}, {} ); 
return true;

```

# 参见

参见本章中的以下食谱：

+   *创建组合图像采样器*

+   *创建统一缓冲区*

+   *创建描述符集布局*

+   *创建描述符池*

+   *分配描述符集*

+   *更新描述符集*

# 释放描述符集

如果我们想要返回由描述符集分配的内存并将其放回池中，我们可以释放一个给定的描述符集。

# 如何做到这一点...

1.  使用逻辑设备的句柄初始化名为 `logical_device` 的 `VkDevice` 类型的变量。

1.  使用带有 `VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT` 标志创建的描述符池。将其句柄存储在名为 `descriptor_pool` 的 `VkDescriptorPool` 类型的变量中。

1.  创建一个名为 `descriptor_sets` 的 `std::vector<VkDescriptorSet>` 类型的向量。将所有应释放的描述符集添加到该向量中。

1.  调用`vkFreeDescriptorSets(logical_device, descriptor_pool, static_cast<uint32_t>(descriptor_sets.size()), descriptor_sets.data())`。对于调用，提供`logical_device`和`descriptor_pool`变量，`descriptor_sets`向量的元素数量，以及`descriptor_sets`向量第一个元素的指针。

1.  通过检查它是否返回`VK_SUCCESS`值来确保调用成功。

1.  由于我们不能再使用已释放描述符集的句柄，因此清除`descriptor_sets`向量。

# 它是如何工作的...

释放描述符集会释放它所使用的内存并将其归还到池中。应该可以从池中分配相同类型的另一组描述符集，但由于池的内存碎片化，这可能不可行（在这种情况下，我们可能需要创建另一个池或重置分配该集的池）。

我们可以一次性释放多个描述符集，但所有这些描述符集都必须来自同一池。这样做：

```cpp
VkResult result = vkFreeDescriptorSets( logical_device, descriptor_pool, static_cast<uint32_t>(descriptor_sets.size()), descriptor_sets.data() ); 
if( VK_SUCCESS != result ) { 
  std::cout << "Error occurred during freeing descriptor sets." << std::endl; 
  return false; 
} 

descriptor_sets.clear(); 
return true;

```

我们不能同时从多个线程释放来自同一池的描述符集。

# 参见

请参阅本章中的以下食谱：

+   *创建描述符池*

+   *分配描述符集*

+   *重置描述符池*

+   *销毁描述符池*

# 重置描述符池

我们可以一次性释放从给定池分配的所有描述符集，而不销毁池本身。为此，我们可以重置描述符池。

# 如何做到这一点...

1.  获取应重置的描述符池并使用其句柄初始化一个名为`descriptor_pool`的`VkDescriptorPool`类型变量。

1.  获取创建描述符池的逻辑设备的句柄。将其句柄存储在名为`logical_device`的`VkDevice`类型变量中。

1.  进行以下调用：`vkResetDescriptorPool(logical_device, descriptor_pool, 0)`，其中使用`logical_device`和`descriptor_pool`变量以及一个`0`值。

1.  检查调用返回的任何错误。因为成功的操作应该返回`VK_SUCCESS`。

# 它是如何工作的...

重置描述符池会将从该池分配的所有描述符集返回到池中。从池中分配的所有描述符集都将隐式释放，并且不能再使用（它们的句柄变为无效）。

如果池是在没有设置`VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT`标志的情况下创建的，那么这是释放从该池分配的描述符集的唯一方法（除了销毁池之外），因为在这样的情况下，我们无法单独释放它们。

要重置池，我们可以编写类似于以下代码的代码：

```cpp
VkResult result = vkResetDescriptorPool( logical_device, descriptor_pool, 0 ); 
if( VK_SUCCESS != result ) { 
  std::cout << "Error occurred during descriptor pool reset." << std::endl; 
  return false; 
} 
return true;

```

# 参见

请参阅本章中的以下食谱：

+   *创建描述符池*

+   *分配描述符集*

+   *释放描述符集*

+   *销毁描述符池*

# 销毁描述符池

当我们不再需要描述符池时，我们可以销毁它（包括从池中分配的所有描述符集）。

# 如何做到这一点...

1.  获取创建的逻辑设备的句柄并将其存储在名为`logical_device`的`VkDevice`类型变量中。

1.  通过名为 `descriptor_pool` 的 `VkDescriptorPool` 类型变量提供描述符池的句柄。

1.  调用 `vkDestroyDescriptorPool(logical_device, descriptor_pool, nullptr)` 并提供 `logical_device` 和 `descriptor_pool` 变量以及一个 `nullptr` 值。

1.  为了安全起见，将 `VK_NULL_HANDLE` 值分配给 `descriptor_pool` 变量。

# 它是如何工作的...

销毁描述符池隐式释放了从它分配的所有描述符集。我们不需要首先释放单个描述符集。但是，由于这个原因，我们需要确保从池中分配的任何描述符集都没有被当前由硬件处理的命令引用。

当我们准备好时，可以像这样销毁描述符池：

```cpp
if( VK_NULL_HANDLE != descriptor_pool ) { 
  vkDestroyDescriptorPool( logical_device, descriptor_pool, nullptr ); 
  descriptor_pool = VK_NULL_HANDLE; 
}

```

# 参见

在本章中查看以下配方：

+   *创建描述符池*

# 销毁描述符集布局

不再使用的描述符集布局应该被销毁。

# 如何操作...

1.  使用名为 `logical_device` 的 `VkDevice` 类型变量提供逻辑设备的句柄。

1.  获取已创建的描述符集布局的句柄，并使用它来初始化一个名为 `descriptor_set_layout` 的 `VkDescriptorSetLayout` 类型的变量。

1.  调用 `vkDestroyDescriptorSetLayout(logical_device, descriptor_set_layout, nullptr)` 并提供逻辑设备和描述符集布局的句柄，以及一个 `nullptr` 值。

1.  为了安全起见，将 `VK_NULL_HANDLE` 值分配给 `descriptor_set_layout` 变量。

# 它是如何工作的...

使用 `vkDestroyDescriptorSetLayout()` 函数销毁描述符集布局，如下所示：

```cpp
if( VK_NULL_HANDLE != descriptor_set_layout ) { 
  vkDestroyDescriptorSetLayout( logical_device, descriptor_set_layout, nullptr ); 
  descriptor_set_layout = VK_NULL_HANDLE; 
}

```

# 参见

在本章中查看以下配方：

+   *创建描述符集布局*

# 销毁采样器

当我们不再需要采样器，并且我们确信它不再被挂起的命令使用时，我们可以销毁它。

# 如何操作...

1.  获取创建采样器的逻辑设备的句柄，并将其存储在名为 `logical_device` 的 `VkDevice` 类型变量中。

1.  获取应该被销毁的采样器的句柄。通过名为 `sampler` 的 `VkSampler` 类型变量提供它。

1.  调用 `vkDestroySampler(logical_device, sampler, nullptr)` 并提供 `logical_device` 和 `sampler` 变量，以及一个 `nullptr` 值。

1.  为了安全起见，将 `VK_NULL_HANDLE` 值分配给 `sampler` 变量。

# 它是如何工作的...

采样器是这样被销毁的：

```cpp
if( VK_NULL_HANDLE != sampler ) { 
  vkDestroySampler( logical_device, sampler, nullptr ); 
  sampler = VK_NULL_HANDLE; 
}

```

我们不需要检查采样器的句柄是否为空，因为删除一个 `VK_NULL_HANDLE` 被忽略。我们这样做只是为了避免不必要的函数调用。但是，当我们删除采样器时，我们必须确保句柄（如果非空）是有效的。

# 参见

在本章中查看以下配方：

+   *创建采样器*
