

# 第二章：改进资源管理

在本章中，我们将改进资源管理，使其更容易处理可能具有不同数量纹理的材料。这种技术通常被称为无绑定，尽管这并不完全准确。我们仍然会绑定一个资源列表；然而，我们可以通过使用索引来访问它们，而不是在特定绘制过程中必须指定确切要使用哪些资源。

我们将要进行的第二个改进是自动生成管线布局。大型项目有成百上千个着色器，根据特定应用程序使用的材料组合编译出许多不同的变体。如果开发者每次更改都要手动更新他们的管线布局定义，那么很少会有应用程序能够上市。本章中提出的实现依赖于 SPIR-V 二进制格式提供的信息。

最后，我们将向我们的 GPU 设备实现中添加管线缓存。此解决方案提高了首次运行后管线对象的创建时间，并且可以显著提高应用程序的加载时间。

总结来说，在本章中，我们将涵盖以下主要主题：

+   解锁并实现无绑定资源

+   自动化管线布局生成

+   使用管线缓存改进加载时间

到本章结束时，你将了解如何在 Vulkan 中启用和使用无绑定资源。你还将能够解析 SPIR-V 二进制数据来自动生成管线布局。最后，你将能够通过使用管线缓存来加快应用程序的加载时间。

# 技术要求

本章的代码可以在以下网址找到：[`github.com/PacktPublishing/Mastering-Graphics-Programming-with-Vulkan/tree/main/source/chapter2`](https://github.com/PacktPublishing/Mastering-Graphics-Programming-with-Vulkan/tree/main/source/chapter2).

# 解锁并实现无绑定渲染

在上一章中，我们必须手动绑定每个材料的纹理。这也意味着，如果我们想支持需要不同数量纹理的不同类型的材料，我们就需要单独的着色器和管线。

Vulkan 提供了一种机制，可以绑定一个可用于多个着色器的纹理数组。然后，每个纹理都可以通过索引访问。在以下章节中，我们将突出显示我们对 GPU 设备实现所做的更改，以启用此功能，并描述如何使用它。

在以下章节中，我们首先将检查启用无绑定资源所需的扩展是否在给定的 GPU 上可用。然后，我们将展示对描述符池创建和描述符集更新的更改，以利用无绑定资源。最后一步将是更新我们的着色器，以便在纹理数组中使用索引进行渲染。

## 检查支持

大多数桌面 GPU，即使相对较旧，只要您有最新的驱动程序，都应该支持`VK_EXT_descriptor_indexing`扩展。仍然是一个好习惯来验证扩展是否可用，并且在生产实现中，如果扩展不可用，提供使用标准绑定模型的替代代码路径。

要验证您的设备是否支持此扩展，您可以使用以下代码，或者您可以使用 Vulkan SDK 提供的`vulkaninfo`应用程序。参见*第一章*，*介绍 Raptor 引擎和 Hydra*，了解如何安装 SDK。

第一步是查询物理设备以确定 GPU 是否支持此扩展。以下代码段完成了这项任务：

```cpp
VkPhysicalDeviceDescriptorIndexingFeatures indexing
_features{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR
           _INDEXING_FEATURES, nullptr };
    VkPhysicalDeviceFeatures2 device_features{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            &indexing_features };
    vkGetPhysicalDeviceFeatures2( vulkan_physical_device,
                                  &device_features );
    bindless_supported = indexing_features.
                         descriptorBindingPartiallyBound &&
                         indexing_features.
                         runtimeDescriptorArray;
```

我们必须填充`VkPhysicalDeviceDescriptorIndexingFeatures`结构并将其链接到`VkPhysicalDeviceFeatures2`结构。驱动程序在调用`vkGetPhysicalDeviceFeatures2`时将填充`indexing_features`变量成员。为了验证描述符索引扩展是否受支持，我们检查`descriptorBindingPartiallyBound`和`runtimeDescriptorArray`的值是否为`true`。

一旦我们确认扩展受支持，我们可以在创建设备时启用它：

```cpp
VkPhysicalDeviceFeatures2 physical_features2 = {
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
vkGetPhysicalDeviceFeatures2( vulkan_physical_device,
                              &physical_features2 );
VkDeviceCreateInfo device_create_info = {};
// same code as chapter 1
device_create_info.pNext = &physical_features2;
if ( bindless_supported ) {
    physical_features2.pNext = &indexing_features;
}
vkCreateDevice( vulkan_physical_device,
                &device_create_info,
                vulkan_allocation_callbacks,
                &vulkan_device );
```

我们必须将`indexing_features`变量链接到创建设备时使用的`physical_features2`变量。其余的代码与*第一章*中的代码相同，*介绍 Raptor 引擎*和*Hydra*。

## 创建描述符池

下一步是从中可以分配支持在绑定后更新纹理内容的描述符集的描述符池：

```cpp
VkDescriptorPoolSize pool_sizes_bindless[] =
{
    { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      k_max_bindless_resources },
      { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      k_max_bindless_resources },
};
pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE
                  _AFTER_BIND_BIT_EXT;
pool_info.maxSets = k_max_bindless_resources * ArraySize(
                    pool_sizes_bindless );
pool_info.poolSizeCount = ( u32 )ArraySize(
                            pool_sizes_bindless );
pool_info.pPoolSizes = pool_sizes_bindless;
vkCreateDescriptorPool( vulkan_device, &pool_info,
                        vulkan_allocation_callbacks,
                        &vulkan_bindless_descriptor_pool);
```

与*第一章*中的代码相比，主要区别是添加了`VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT_EXT`标志。此标志是允许创建在绑定后可以更新的描述符集所必需的。

接下来，我们必须定义描述符集布局绑定：

```cpp
const u32 pool_count = ( u32 )ArraySize(
                         pool_sizes_bindless );
VkDescriptorSetLayoutBinding vk_binding[ 4 ];
VkDescriptorSetLayoutBinding& image_sampler_binding =
    vk_binding[ 0 ];
image_sampler_binding.descriptorType = VK_DESCRIPTOR
                                       _TYPE_COMBINED
                                       _IMAGE_SAMPLER;
image_sampler_binding.descriptorCount =
    k_max_bindless_resources;
image_sampler_binding.binding = k_bindless_texture_binding;
VkDescriptorSetLayoutBinding& storage_image_binding =
    vk_binding[ 1 ];
storage_image_binding.descriptorType = VK_DESCRIPTOR
                                       _TYPE_STORAGE_IMAGE;
storage_image_binding.descriptorCount =
    k_max_bindless_resources;
storage_image_binding.binding = k_bindless_texture_binding
                                + 1;
```

注意，`descriptorCount`不再具有`1`的值，而必须容纳我们可以使用的最大纹理数量。现在我们可以使用这些数据来创建描述符集布局：

```cpp
VkDescriptorSetLayoutCreateInfo layout_info = {
    VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
layout_info.bindingCount = pool_count;
layout_info.pBindings = vk_binding;
layout_info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE
                    _UPDATE_AFTER_BIND_POOL_BIT_EXT;
VkDescriptorBindingFlags bindless_flags =
    VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT_EXT |
        VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT_EXT;
VkDescriptorBindingFlags binding_flags[ 4 ];
binding_flags[ 0 ] = bindless_flags;
binding_flags[ 1 ] = bindless_flags;
VkDescriptorSetLayoutBindingFlagsCreateInfoEXT
extended_info{
    VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT
        _BINDING_FLAGS_CREATE_INFO_EXT, nullptr };
extended_info.bindingCount = pool_count;
extended_info.pBindingFlags = binding_flags;
layout_info.pNext = &extended_info;
vkCreateDescriptorSetLayout( vulkan_device, &layout_info,
                             vulkan_allocation_callbacks,
                             &vulkan_bindless
                             _descriptor_layout );
```

代码与上一章中看到的版本非常相似；然而，我们添加了`bindless_flags`值以启用描述符集的部分更新。我们还需要将`VkDescriptorSetLayoutBindingFlagsCreateInfoEXT`结构链接到`layout_info`变量。最后，我们可以创建将在应用程序生命周期内使用的描述符集：

```cpp
VkDescriptorSetAllocateInfo alloc_info{
    VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
alloc_info.descriptorPool = vulkan_bindless
                            _descriptor_pool;
alloc_info.descriptorSetCount = 1;
alloc_info.pSetLayouts = &vulkan_bindless_descriptor
                         _layout;
vkAllocateDescriptorSets( vulkan_device, &alloc_info,
                          &vulkan_bindless_descriptor_set
                         );
```

我们只需用我们之前定义的值填充`VkDescriptorSetAllocateInfo`结构并调用`vkAllocateDescriptorSets`。

## 更新描述符集

到目前为止，我们已经完成了大部分繁重的工作。当我们调用 `GpuDevice::create_texture` 时，新创建的资源会被添加到 `texture_to_update_bindless` 数组中：

```cpp
if ( gpu.bindless_supported ) {
    ResourceUpdate resource_update{
        ResourceDeletionType::Texture,
            texture->handle.index, gpu.current_frame };
    gpu.texture_to_update_bindless.push( resource_update );
}
```

还可以将特定的采样器关联到给定的纹理上。例如，当我们为某个材质加载纹理时，我们添加以下代码：

```cpp
gpu.link_texture_sampler( diffuse_texture_gpu.handle,
                          diffuse_sampler_gpu.handle );
```

这将散布的纹理与其采样器关联起来。这个信息将在下一节代码中用来确定我们是否使用默认采样器或刚刚分配给纹理的采样器。

在处理下一帧之前，我们使用上一节创建的描述符集更新任何已上传的新纹理：

```cpp
for ( i32 it = texture_to_update_bindless.size - 1;
  it >= 0; it-- ) {
    ResourceUpdate& texture_to_update =
        texture_to_update_bindless[ it ];
   Texture* texture = access_texture( {
                      texture_to_update.handle } );
    VkWriteDescriptorSet& descriptor_write =
        bindless_descriptor_writes[ current_write_index ];
    descriptor_write = {
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    descriptor_write.descriptorCount = 1;
    descriptor_write.dstArrayElement =
        texture_to_update.handle;
    descriptor_write.descriptorType =
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    descriptor_write.dstSet =
        vulkan_bindless_descriptor_set;
    descriptor_write.dstBinding =
        k_bindless_texture_binding;
    Sampler* vk_default_sampler = access_sampler(
                                  default_sampler );
    VkDescriptorImageInfo& descriptor_image_info =
        bindless_image_info[ current_write_index ];
    if ( texture->sampler != nullptr ) {
        descriptor_image_info.sampler =
        texture->sampler->vk_sampler;
    }
    else {
        descriptor_image_info.sampler =
        vk_default_sampler->vk_sampler;
    }
descriptor_image_info.imageView = 
        texture->vk_format != VK_FORMAT_UNDEFINED ? 
        texture->vk_image_view : vk_dummy_texture-> 
        vk_image_view;
    descriptor_image_info.imageLayout =
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    descriptor_write.pImageInfo = &descriptor_image_info;
    texture_to_update.current_frame = u32_max;
    texture_to_update_bindless.delete_swap( it );
    ++current_write_index;
}
```

上述代码与上一个版本非常相似。我们已突出显示主要差异：采样器选择，正如我们上段所述，以及如果槽位为空时使用虚拟纹理。我们仍然需要为每个槽位分配一个纹理，因此如果未指定，则使用虚拟纹理。这也有助于在场景中查找任何缺失的纹理。

如果你更喜欢紧密打包的纹理数组，另一个选项是启用 `VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT_EXT` 标志，并在创建描述符集时链式连接一个 `VkDescriptorSetVariableDescriptorCountAllocateInfoEXT` 结构。我们已经有了一些启用此功能的初步代码，并鼓励你完成实现！

## 更新着色器代码

使用绑定无关渲染的最后一部分是在着色器代码中，因为它需要以不同的方式编写。

对于所有使用绑定无关资源的着色器，步骤都是相似的，并且将它们定义在公共头文件中将是有益的。不幸的是，这并不完全由 **OpenGL 着色语言** 或 **GLSL** 支持。

我们建议自动化这一步骤，因为它可以在编译引擎代码中的着色器时轻松添加。

首先要做的是在 GLSL 代码中启用非均匀限定符：

```cpp
#extension GL_EXT_nonuniform_qualifier : enable
```

这将在当前着色器中启用扩展，而不是全局；因此，它必须在每个着色器中编写。

以下代码是正确绑定无关纹理的声明，但有一个限制：

```cpp
layout ( set = 1, binding = 10 ) uniform sampler2D global_textures[];
layout ( set = 1, binding = 10 ) uniform sampler3D global_textures_3d[];
```

这是一个已知的技巧，可以将纹理声明别名到相同的绑定点。这允许我们拥有一个全局的绑定无关纹理数组，但一次支持所有类型的纹理（一维、二维、三维及其数组对应物）！

这简化了在引擎和着色器中绑定无关纹理的使用。

最后，为了读取纹理，着色器中的代码需要按以下方式修改：

```cpp
texture(global_textures[nonuniformEXT(texture_index)],
        vTexcoord0)
```

让我们按以下顺序进行：

1.  首先，我们需要来自常量的整数索引。在这种情况下，`texture_index` 将包含与绑定无关数组中纹理位置相同的数字。

1.  第二点，这是一个关键的变化，我们需要用`nonuniformEXT`限定符（[`github.com/KhronosGroup/GLSL/blob/master/extensions/ext/GL_EXT_nonuniform_qualifier.txt`](https://github.com/KhronosGroup/GLSL/blob/master/extensions/ext/GL_EXT_nonuniform_qualifier.txt)）包装索引；这基本上会在不同的执行之间同步程序，以确保正确读取纹理索引，以防索引在不同线程的同一着色器调用中不同。

这可能一开始听起来很复杂，但把它看作是一个需要同步的多线程问题，以确保每个线程都能正确读取适当的纹理索引，从而使用正确的纹理。

1.  最后，使用我们从`global_textures`数组中读取的同步索引，我们终于得到了我们想要的纹理样本！

我们现在已将无绑定纹理支持添加到 Raptor 引擎中！我们首先检查 GPU 是否支持此功能。然后我们详细说明了我们对描述符池和描述符集创建所做的更改。

最后，我们展示了如何随着新纹理上传到 GPU，更新描述符集，以及必要的着色器修改以使用无绑定纹理。从现在开始的所有渲染都将使用这个功能；因此，这个概念将变得熟悉。

接下来，我们将通过解析着色器的二进制数据来添加自动管线生成，以提升我们的引擎功能。

# 自动化管线布局生成

在本节中，我们将利用 SPIR-V 二进制格式提供的数据来提取创建管线布局所需的信息。SPIR-V 是着色器源代码在传递给 GPU 之前编译成的**中间表示**（**IR**）。

与标准的 GLSL 着色器源代码（纯文本）相比，SPIR-V 是一种二进制格式。这意味着它是一个在分发应用程序时更紧凑的格式。更重要的是，开发者不必担心他们的着色器根据其代码运行的 GPU 和驱动程序被编译成不同的一组高级指令。

然而，SPIR-V 二进制文件不包含 GPU 将要执行的最终指令。每个 GPU 都会将 SPIR-V 数据块进行最终编译成 GPU 指令。这一步仍然是必需的，因为不同的 GPU 和驱动程序版本可以为相同的 SPIR-V 二进制文件生成不同的汇编代码。

将 SPIR-V 作为中间步骤仍然是一个巨大的改进。着色器代码的验证和解析是在离线完成的，开发者可以将他们的着色器与他们的应用程序代码一起编译。这允许我们在尝试运行着色器代码之前发现任何语法错误。

拥有中间表示形式的另一个好处是能够将不同语言编写的着色器编译为 SPIR-V，以便它们可以与 Vulkan 一起使用。例如，可以将用 HLSL 编写的着色器编译为 SPIR-V，并在 Vulkan 渲染器中重用它。

在此选项可用之前，开发者要么必须手动移植代码，要么必须依赖将着色器从一种语言重写到另一种语言的工具。

到现在为止，您应该已经相信 SPIR-V 的引入为开发者和 Vulkan API 带来了优势。

在接下来的章节中，我们将使用我们的一个着色器来向您展示如何将其编译为 SPIR-V，并解释如何使用二进制数据中的信息自动生成管线布局。

## 将 GLSL 编译为 SPIR-V

我们将使用我们在*第一章*，*介绍 Raptor 引擎和 Hydra*中开发的顶点着色器代码。之前，我们将着色器代码字符串存储在`main.cpp`文件中，并且在将其传递给 Vulkan API 以创建管线之前，我们没有将其编译为 SPIR-V。

从本章开始，我们将把所有着色器代码存储在每个章节的`shaders`文件夹中。对于*第二章*，*改进资源管理*，您将找到两个文件：`main.vert`用于顶点着色器，`main.frag`用于片段着色器。以下是`main.vert`的内容：

```cpp
#version 450
layout ( std140, binding = 0 ) uniform LocalConstants {
    mat4        model;
    mat4        view_projection;
    mat4        model_inverse;
    vec4        eye;
    vec4        light;
};
layout(location=0) in vec3 position;
layout(location=1) in vec4 tangent;
layout(location=2) in vec3 normal;
layout(location=3) in vec2 texCoord0;
layout (location = 0) out vec2 vTexcoord0;
layout (location = 1) out vec3 vNormal;
layout (location = 2) out vec4 vTangent;
layout (location = 3) out vec4 vPosition;
void main() {
    gl_Position = view_projection * model * vec4(position,
                                                 1);
    vPosition = model * vec4(position, 1.0);
    vTexcoord0 = texCoord0;
    vNormal = mat3(model_inverse) * normal;
    vTangent = tangent;
}
```

这段代码对于一个顶点着色器来说相当标准。我们有四个数据流，用于位置、切线、法线和纹理坐标。我们还定义了一个`LocalConstants`统一缓冲区，用于存储所有顶点的公共数据。最后，我们定义了将传递给片段着色器的`out`变量。

Vulkan SDK 提供了将 GLSL 编译为 SPIR-V 以及将生成的 SPIR-V 反汇编成人类可读形式的工具。这可以用于调试表现不佳的着色器。

要编译我们的顶点着色器，我们运行以下命令：

```cpp
glslangValidator -V main.vert -o main.vert.spv
```

这将生成一个包含二进制数据的`main.vert.spv`文件。要查看此文件的内容以人类可读格式，我们运行以下命令：

```cpp
spirv-dis main.vert.spv
```

此命令将在终端上打印出反汇编的 SPIR-V。我们现在将检查输出的相关部分。

## 理解 SPIR-V 输出

从输出的顶部开始，以下是我们提供的第一组信息：

```cpp
      OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
      OpMemoryModel Logical GLSL450
      OpEntryPoint Vertex %main "main" %_ %position
      %vPosition %vTexcoord0 %texCoord0 %vNormal %normal
      %vTangent %tangent
      OpSource GLSL 450
      OpName %main "main"
```

这个前缀定义了编写着色器所使用的 GLSL 版本。`OpEntryPoint`指令引用了主函数，并列出了着色器的输入和输出。惯例是变量以`%`为前缀，并且可以提前声明稍后定义的变量。

下一个部分定义了在此着色器中可用的输出变量：

```cpp
OpName %gl_PerVertex "gl_PerVertex"
OpMemberName %gl_PerVertex 0 "gl_Position"
OpMemberName %gl_PerVertex 1 "gl_PointSize"
OpMemberName %gl_PerVertex 2 "gl_ClipDistance"
OpMemberName %gl_PerVertex 3 "gl_CullDistance"
OpName %_ ""
```

这些是由编译器自动注入的变量，由 GLSL 规范定义。我们可以看到一个 `gl_PerVertex` 结构体，它反过来有四个成员：`gl_Position`，`gl_PointSize`，`gl_ClipDistance` 和 `gl_CullDistance`。还有一个未命名的变量定义为 `%_`。我们很快就会发现它指的是什么。

现在，我们继续到我们定义的结构体：

```cpp
OpName %LocalConstants "LocalConstants"
OpMemberName %LocalConstants 0 "model"
OpMemberName %LocalConstants 1 "view_projection"
OpMemberName %LocalConstants 2 "model_inverse"
OpMemberName %LocalConstants 3 "eye"
OpMemberName %LocalConstants 4 "light"
OpName %__0 ""
```

在这里，我们有我们的 `LocalConstants` 统一缓冲区的条目，其成员以及它们在结构体中的位置。我们再次看到了一个未命名的 `%__0` 变量。我们很快就会了解它。SPIR-V 允许你定义成员装饰来提供有助于确定数据布局和结构体内位置的信息：

```cpp
OpMemberDecorate %LocalConstants 0 ColMajor
OpMemberDecorate %LocalConstants 0 Offset 0
OpMemberDecorate %LocalConstants 0 MatrixStride 16
OpMemberDecorate %LocalConstants 1 ColMajor
OpMemberDecorate %LocalConstants 1 Offset 64
OpMemberDecorate %LocalConstants 1 MatrixStride 16
OpMemberDecorate %LocalConstants 2 ColMajor
OpMemberDecorate %LocalConstants 2 Offset 128
OpMemberDecorate %LocalConstants 2 MatrixStride 16
OpMemberDecorate %LocalConstants 3 Offset 192
OpMemberDecorate %LocalConstants 4 Offset 208
OpDecorate %LocalConstants Block
```

从这些条目中，我们可以开始对结构体中每个成员的类型有所了解。例如，我们可以识别前三个条目为矩阵。最后一个条目只有一个偏移量。

对于我们的目的来说，偏移量值是最相关的值，因为它允许我们知道每个成员的确切起始位置。当从 CPU 向 GPU 转移数据时，这一点至关重要，因为每个成员的对齐规则可能不同。

接下来的两行定义了我们的结构体的描述符集和绑定：

```cpp
OpDecorate %__0 DescriptorSet 0
OpDecorate %__0 Binding 0
```

如您所见，这些装饰项引用了未命名的 `%__0` 变量。我们现在已经到达了定义变量类型的部分：

```cpp
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%uint = OpTypeInt 32 0
%uint_1 = OpConstant %uint 1
%_arr_float_uint_1 = OpTypeArray %float %uint_1
%gl_PerVertex = OpTypeStruct %v4float %float
                %_arr_float_uint_1 %_arr_float_uint_1
%_ptr_Output_gl_PerVertex = OpTypePointer Output
                            %gl_PerVertex
%_ = OpVariable %_ptr_Output_gl_PerVertex Output
```

对于每个变量，我们都有其类型，并且根据类型，还有与之相关的附加信息。例如，`%float` 变量是 32 位 `float` 类型；`%v4float` 变量是 `vector` 类型，并且包含 4 个 `%float` 值。

这对应于 GLSL 中的 `vec4`。然后我们有一个无符号值 `1` 的常量定义和一个长度为 `1` 的固定大小的 `float` 类型的数组。

`%gl_PerVertex` 变量的定义如下。它是 `struct` 类型，并且，正如我们之前看到的，它有四个成员。它们的类型是 `vec4` 用于 `gl_Position`，`float` 用于 `gl_PointSize`，以及 `float[1]` 用于 `gl_ClipDistance` 和 `gl_CullDistance`。

SPIR-V 规范要求每个可读或可写的变量都通过指针来引用。这正是我们看到的 `%_ptr_Output_gl_PerVertex`：它是指向 `gl_PerVertex` 结构体的指针。最后，我们可以看到未命名的 `%_` 变量的类型是指向 `gl_PerVertex` 结构体的指针。

最后，我们有我们自己的统一数据的类型定义：

```cpp
%LocalConstants = OpTypeStruct %mat4v4float %mat4v4float
                  %mat4v4float %v4float %v4float
%_ptr_Uniform_LocalConstants = OpTypePointer Uniform
                               %LocalConstants
%__0 = OpVariable %_ptr_Uniform_LocalConstants
       Uniform
```

如前所述，我们可以看到 `%LocalConstants` 是一个具有五个成员的结构体，其中三个是 `mat4` 类型，两个是 `vec4` 类型。然后我们有我们统一结构体的指针类型定义，最后是此类型的 `%__0` 变量。请注意，此变量具有 `Uniform` 属性。这意味着它是只读的，我们将在以后利用这个信息来确定要添加到管道布局中的描述符类型。

解析的其余部分包含输入和输出变量定义。它们的定义结构与迄今为止我们所看到的变量相同，因此我们在这里不会分析它们。

解析还包含着着色器主体的指令。虽然看到 GLSL 代码如何被转换成 SPIR-V 指令很有趣，但这与管道创建无关，我们在这里不会涉及这个细节。

接下来，我们将展示如何利用所有这些数据来自动化管道创建。

## 从 SPIR-V 到管道布局

Khronos 已经提供了解析 SPIR-V 数据以创建管道布局的功能。您可以在 [`github.com/KhronosGroup/SPIRV-Reflect`](https://github.com/KhronosGroup/SPIRV-Reflect) 找到其实施。对于这本书，我们决定编写一个简化的解析器版本，我们认为它更容易跟随，因为我们只对一小部分条目感兴趣。

您可以在 `source\chapter2\graphics\spirv_parser.cpp` 中找到实现。让我们看看如何使用这个 API 以及它在底层是如何工作的：

```cpp
spirv::ParseResult parse_result{ };
spirv::parse_binary( ( u32* )spv_vert_data,
                       spv_vert_data_size, name_buffer,
                       &parse_result );
spirv::parse_binary( ( u32* )spv_frag_data,
                       spv_frag_data_size, name_buffer,
                       &parse_result );
```

在这里，我们假设顶点和片段着色器的二进制数据已经读取到 `spv_vert_data` 和 `spv_frag_data` 变量中。我们必须定义一个空的 `spirv::ParseResult` 结构，它将包含解析的结果。其定义相当简单：

```cpp
struct ParseResult {
    u32 set_count;
    DescriptorSetLayoutCreation sets[MAX_SET_COUNT];
};
```

它包含了我们从二进制数据中识别出的集合数量以及每个集合的条目列表。

解析的第一步是确保我们正在读取有效的 SPIR-V 数据：

```cpp
u32 spv_word_count = safe_cast<u32>( data_size / 4 );
u32 magic_number = data[ 0 ];
RASSERT( magic_number == 0x07230203 );
u32 id_bound = data[3];
```

我们首先计算二进制中包含的 32 位单词数量。然后我们验证前四个字节是否匹配标识 SPIR-V 二进制的魔数。最后，我们检索二进制中定义的 ID 数量。

接下来，我们遍历二进制中的所有单词以检索所需的信息。每个 ID 定义都以 `Op` 类型及其组成的单词数量开始：

```cpp
SpvOp op = ( SpvOp )( data[ word_index ] & 0xFF );
u16 word_count = ( u16 )( data[ word_index ] >> 16 );
```

`Op` 类型存储在单词的最低 16 位中，单词计数在最高 16 位中。接下来，我们解析我们感兴趣的 `Op` 类型的数据。在本节中，我们不会涵盖所有 `Op` 类型，因为所有类型的结构都是相同的。我们建议您参考 SPIR-V 规范（在 *进一步阅读* 部分链接），以获取每个 `Op` 类型的更多详细信息。

我们从当前正在解析的着色器类型开始：

```cpp
case ( SpvOpEntryPoint ):
{
    SpvExecutionModel model = ( SpvExecutionModel )data[
                                word_index + 1 ];
    stage = parse_execution_model( model );
    break;
}
```

我们提取执行模型，将其转换为 `VkShaderStageFlags` 值，并将其存储在 `stage` 变量中。

接下来，我们解析描述符集索引和绑定：

```cpp
case ( SpvOpDecorate ):
{
    u32 id_index = data[ word_index + 1 ];
    Id& id= ids[ id_index ];
    SpvDecoration decoration = ( SpvDecoration )data[
                                 word_index + 2 ];
    switch ( decoration )
    {
        case ( SpvDecorationBinding ):
        {
            id.binding = data[ word_index + 3 ];
            break;
        }
        case ( SpvDecorationDescriptorSet ):
        {
            id.set = data[ word_index + 3 ];
            break;
        }
    }
    break;
}
```

首先，我们检索 ID 的索引。如前所述，变量可以是前向声明的，我们可能需要多次更新相同 ID 的值。接下来，我们检索装饰的值。我们只对描述符集索引（`SpvDecorationDescriptorSet`）和绑定（`SpvDecorationBinding`）感兴趣，并将它们的值存储在这个 ID 的条目中。

我们接着用一个变量类型的例子来说明：

```cpp
case ( SpvOpTypeVector ):
{
    u32 id_index = data[ word_index + 1 ];
    Id& id= ids[ id_index ];
    id.op = op;
    id.type_index = data[ word_index + 2 ];
    id.count = data[ word_index + 3 ];
    break;
}
```

正如我们在反汇编中看到的，一个向量由其条目类型和计数定义。我们将其存储在 ID 结构体的`type_index`和`count`成员中。在这里，我们还可以看到如果需要，ID 可以引用另一个 ID。`type_index`成员存储对`ids`数组中另一个条目的索引，并且可以在以后用于检索额外的类型信息。

接下来，我们有一个样本定义：

```cpp
case ( SpvOpTypeSampler ):
{
    u32 id_index = data[ word_index + 1 ];
    RASSERT( id_index < id_bound );
    Id& id= ids[ id_index ];
    id.op = op;
    break;
}
```

我们只需要存储这个条目的`Op`类型。最后，我们有变量类型的条目：

```cpp
case ( SpvOpVariable ):
{
    u32 id_index = data[ word_index + 2 ];
    Id& id= ids[ id_index ];
    id.op = op;
    id.type_index = data[ word_index + 1 ];
    id.storage_class = ( SpvStorageClass )data[
                         word_index + 3 ];
    break;
}
```

这个条目的相关信息是`type_index`，它将始终引用一个`pointer`类型的条目和存储类。存储类告诉我们哪些是我们感兴趣的变量条目，哪些可以跳过。

这正是代码的下一部分所做的事情。一旦我们解析完所有 ID，我们就遍历每个 ID 条目并识别我们感兴趣的条目。我们首先识别所有变量：

```cpp
for ( u32 id_index = 0; id_index < ids.size; ++id_index ) {
    Id& id= ids[ id_index ];
    if ( id.op == SpvOpVariable ) {
```

接下来，我们使用变量存储类来确定它是否是一个统一变量：

```cpp
switch ( id.storage_class ) {
    case ( SpvStorageClassUniform ):
    case ( SpvStorageClassUniformConstant ):
    {
```

我们只对`Uniform`和`UniformConstant`变量感兴趣。然后我们检索`uniform`类型。记住，检索变量实际类型存在双重间接引用：首先，我们获取`pointer`类型，然后从`pointer`类型获取变量的实际类型。我们已突出显示执行此操作的代码：

```cpp
Id& uniform_type = ids[ ids[ id.type_index ].type_index ];
DescriptorSetLayoutCreation& setLayout =
parse_result->sets[ id.set ];
setLayout.set_set_index( id.set );
DescriptorSetLayoutCreation::Binding binding{ };
binding.start = id.binding;
binding.count = 1;
```

在检索类型后，我们获取这个变量所属集合的`DescriptorSetLayoutCreation`条目。然后我们创建一个新的`binding`条目并存储`binding`值。我们总是假设每个资源有一个`1`的计数。

在这个最后步骤中，我们确定这个绑定的资源类型，并将其条目添加到集合布局中：

```cpp
switch ( uniform_type.op ) {
    case (SpvOpTypeStruct):
    {
        binding.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        binding.name = uniform_type.name.text;
        break;
    }
    case (SpvOpTypeSampledImage):
    {
        binding.type = VK_DESCRIPTOR_TYPE_COMBINED
        _IMAGE_SAMPLER;
        binding.name = id.name.text;
        break;
    }
}
setLayout.add_binding_at_index( binding, id.binding );
```

我们使用`Op`类型来确定我们找到的资源类型。到目前为止，我们只对统一缓冲区的`Struct`和纹理的`SampledImage`感兴趣。在本书的剩余部分，如果需要，我们将添加对更多类型的支持。

虽然可以在统一缓冲区和存储缓冲区之间进行区分，但二进制数据无法确定缓冲区是动态的还是静态的。在我们的实现中，应用程序代码需要指定这个细节。

另一个选择是使用命名约定（例如，在动态缓冲区前加上`dyn_`），以便可以自动识别动态缓冲区。

这就结束了我们对 SPIR-V 二进制格式的介绍。可能需要阅读几遍才能完全理解它是如何工作的，但不用担心，我们确实花了一些迭代时间才能完全理解它！

知道如何解析 SPIR-V 数据是自动化图形开发其他方面的重要工具。例如，它可以用来自动化生成 C++头文件，以保持 CPU 和 GPU 结构同步。我们鼓励你扩展我们的实现，以添加你可能需要的功能支持！

在本节中，我们解释了如何将着色器源编译成 SPIR-V。我们展示了 SPIR-V 二进制格式的组织方式以及如何解析这些数据以帮助我们自动创建管道布局。

在本章的下一节和最后一节中，我们将向我们的 GPU 设备实现中添加管道缓存。

# 使用管道缓存提高加载时间

每次我们创建一个图形管道，以及在较小程度上创建一个计算管道时，驱动程序都必须分析并编译我们提供的着色器。它还必须检查我们在创建结构中定义的状态，并将其转换为编程 GPU 不同单元的指令。这个过程相当昂贵，这也是为什么在 Vulkan 中我们必须提前定义大部分管道状态的原因之一。

在本节中，我们将向我们的 GPU 设备实现中添加管道缓存以提高加载时间。如果你的应用程序需要创建成千上万的管道，它可能会产生显著的启动时间，或者对于游戏来说，在关卡之间的加载时间可能会很长。

本节中描述的技术将有助于减少创建管道所需的时间。你首先会注意到`GpuDevice::create_pipeline`方法接受一个新可选参数，该参数定义了管道缓存文件的路径：

```cpp
GpuDevice::create_pipeline( const PipelineCreation&
                            creation, const char*
                            cache_path )
```

然后我们需要定义`VkPipelineCache`结构：

```cpp
VkPipelineCache pipeline_cache = VK_NULL_HANDLE;
VkPipelineCacheCreateInfo pipeline_cache_create_info {
    VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO };
```

下一步是检查管道缓存文件是否已经存在。如果存在，我们加载文件数据并将其添加到管道缓存创建中：

```cpp
FileReadResult read_result = file_read_binary( cache_path,
                                               allocator );
pipeline_cache_create_info.initialDataSize =
  read_result.size;
pipeline_cache_create_info.pInitialData = read_result.data;
```

如果文件不存在，我们不需要对创建结构做任何进一步的修改。现在我们可以调用`vkCreatePipelineCache`：

```cpp
vkCreatePipelineCache( vulkan_device,
                       &pipeline_cache_create_info,
                       vulkan_allocation_callbacks,
                       &pipeline_cache );
```

这将返回一个指向`VkPipelineCache`对象的句柄，我们将在创建管道对象时使用它：

```cpp
vkCreateGraphicsPipelines( vulkan_device, pipeline_cache,
                           1, &pipeline_info,
                           vulkan_allocation_callbacks,
                           &pipeline->vk_pipeline );
```

我们可以对计算管道做同样的操作：

```cpp
vkCreateComputePipelines( vulkan_device, pipeline_cache, 1,
                          &pipeline_info,
                          vulkan_allocation_callbacks,
                          &pipeline->vk_pipeline );
```

如果我们已加载管道缓存文件，驱动程序将使用这些数据来加速管道创建。另一方面，如果我们是第一次创建给定的管道，我们现在可以查询并存储管道缓存数据以供以后重用：

```cpp
sizet cache_data_size = 0;
vkGetPipelineCacheData( vulkan_device, pipeline_cache,
                        &cache_data_size, nullptr );
void* cache_data = allocator->allocate( cache_data_size, 64 );
vkGetPipelineCacheData( vulkan_device, pipeline_cache,
                        &cache_data_size, cache_data );
file_write_binary( cache_path, cache_data, cache_data_size );
```

我们首先使用`nullptr`调用`vkGetPipelineCacheData`来获取数据成员的缓存数据大小。然后，我们分配存储缓存数据的内存，并再次调用`vkGetPipelineCacheData`，这次使用一个指向将要存储缓存数据的内存的指针。最后，我们将这些数据写入在调用`GpuDevice::create_pipeline`时指定的文件中。

现在我们已经完成了管道缓存数据结构，可以销毁它：

```cpp
vkDestroyPipelineCache( vulkan_device, pipeline_cache,
                        vulkan_allocation_callbacks );
```

在我们总结之前，我们想提到管道缓存的一个缺点。缓存中的数据由每个供应商的驱动程序实现控制。当发布新的驱动程序版本时，缓存的数据格式可能会改变，变得与之前存储在缓存文件中的数据不兼容。在这种情况下，拥有缓存文件可能不会带来任何好处，因为驱动程序无法使用它。

因此，每个驱动程序都必须在缓存数据前加上以下头部信息：

```cpp
struct VkPipelineCacheHeaderVersionOne {
    uint32_t                       headerSize;
    VkPipelineCacheHeaderVersion   headerVersion;
    uint32_t                       vendorID;
    uint32_t                       deviceID;
    uint8_t                        pipeline
                                   CacheUUID[VK_UUID_SIZE];
}
```

当我们从磁盘加载缓存数据时，我们可以将头部中的值与驱动程序和 GPU 返回的值进行比较：

```cpp
VkPipelineCacheHeaderVersionOne* cache_header =
    (VkPipelineCacheHeaderVersionOne*)read_result.data;
if ( cache_header->deviceID == vulkan_physical
     _properties.deviceID && cache_header->vendorID ==
     vulkan_physical_properties.vendorID &&
     memcmp( cache_header->pipelineCacheUUID,
     vulkan_physical_properties.pipelineCacheUUID,
     VK_UUID_SIZE ) == 0 ) {
    pipeline_cache_create_info.initialDataSize =
    read_result.size;
    pipeline_cache_create_info.pInitialData =
    read_result.data;
}
else
{
    cache_exists = false;
}
```

如果头部的值与我们正在运行的设备上的值匹配，我们就像以前一样使用缓存数据。如果不匹配，我们将像缓存不存在一样操作，并在管道创建后存储一个新的版本。

在本节中，我们展示了如何利用管道缓存来在运行时加快管道创建速度。我们强调了我们对 GPU 设备实现所做的更改，以利用此功能，以及它在本章代码中的应用。

# 摘要

在本章中，我们改进了我们的 GPU 设备实现，使其更容易管理大量使用无绑定资源的纹理。我们解释了需要哪些扩展，并详细说明了在创建描述符集布局以允许使用无绑定资源时需要哪些更改。然后，我们展示了在创建描述符集以更新正在使用的纹理数组时所需的更改。

然后，我们通过解析`glslang`编译器为我们着色器生成的 SPIR-V 二进制文件，添加了自动管道布局生成。我们提供了 SPIR-V 二进制数据格式的概述，并解释了如何解析它以提取绑定到着色器的资源，以及如何使用这些信息来创建管道布局。

最后，我们通过添加管道缓存来增强我们的管道创建 API，以改善应用程序首次运行后的加载时间。我们介绍了生成或加载管道缓存数据所需的 Vulkan API。我们还解释了管道缓存的局限性以及如何处理它们。

本章中提出的所有技术都有一个共同的目标，那就是使处理大型项目更容易，并在修改我们的着色器或材质时将手动代码更改减少到最低。

我们将在下一章通过添加多线程来记录多个命令缓冲区或并行提交多个工作负载到 GPU，继续扩展我们的引擎。

# 进一步阅读

我们只涵盖了 SPIR-V 规范的一小部分。如果您想根据您的需求扩展我们的解析器实现，我们强烈建议您查阅官方规范：[`www.khronos.org/registry/SPIR-V/specs/unified1/SPIRV.xhtml`](https://www.khronos.org/registry/SPIR-V/specs/unified1/SPIRV.xhtml)。

我们为这一章节编写了一个定制的 SPIR-V 解析器，主要是为了教育目的。对于您自己的项目，我们建议使用 Khronos 提供的现有反射库：[`github.com/KhronosGroup/SPIRV-Reflect`](https://github.com/KhronosGroup/SPIRV-Reflect)。

它提供了本章所述的功能，用于推断着色器二进制的管道布局以及许多其他特性。
