# 渲染通道和帧缓冲区

在本章中，我们将介绍以下食谱：

+   指定附件描述

+   指定子通道描述

+   指定子通道之间的依赖关系

+   创建渲染通道

+   创建一个帧缓冲区

+   准备用于几何渲染和后处理子通道的渲染通道

+   准备带有颜色和深度附件的渲染通道和帧缓冲区

+   开始一个渲染通道

+   转进下一个子通道

+   结束渲染通道

+   销毁帧缓冲区

+   销毁渲染通道

# 简介

在 Vulkan 中，绘图命令被组织到渲染通道中。渲染通道是一组子通道的集合，它描述了图像资源（颜色、深度/模板、输入附件）的使用：它们的布局是什么，以及这些布局如何在子通道之间转换，当我们向附件中渲染或从它们读取数据时，如果它们的内 容在渲染通道后还需要，或者如果它们的用途仅限于渲染通道的范围。

在渲染通道中存储的上述数据只是一个一般描述，或者说是元数据。实际参与渲染过程的资源是通过帧缓冲区指定的。通过它们，我们定义了哪些图像视图用于哪些渲染附件。

我们需要在发出（记录）渲染命令之前提前准备所有这些信息。有了这些知识，驱动程序可以大大优化绘图过程，限制渲染所需的内存量，或者甚至为某些附件使用非常快速的缓存，从而进一步提高性能。

在本章中，我们将学习如何将绘图操作组织成一系列渲染通道和子通道，这是使用 Vulkan 绘制任何内容所必需的。我们还将学习如何准备在渲染（绘图）过程中使用的渲染目标附件的描述以及如何创建帧缓冲区，这些帧缓冲区定义了将用作这些附件的实际图像视图。

# 指定附件描述

渲染通道代表一组称为附件的资源（图像），这些资源在渲染操作期间使用。这些分为颜色、深度/模板、输入或解析附件。在我们能够创建渲染通道之前，我们需要描述其中使用的所有附件。

# 如何操作...

1.  创建一个类型为 `VkAttachmentDescription` 的向量。将向量命名为 `attachments_descriptions`。对于渲染通道中使用的每个附件，向 `attachments_descriptions` 向量中添加一个元素，并使用以下值为其成员：

    +   `0` 对 `flags` 的值

    +   给定附件的 `format` 所选格式

    +   `samples` 的每像素样本数

    +   对于`loadOp`，指定在渲染过程开始时应在附件内容上执行的操作类型--如果附件内容应该被清除，则使用`VK_ATTACHMENT_LOAD_OP_CLEAR`值；如果其当前内容应该被保留，则使用`VK_ATTACHMENT_LOAD_OP_LOAD`值；如果打算自己覆盖整个附件并且不关心其当前内容，则使用`VK_ATTACHMENT_LOAD_OP_DONT_CARE`值（此参数用于颜色附件或深度/模板附件的深度方面）。

    +   对于`storeOp`，指定在渲染过程结束后如何处理附件的内容--如果应该保留，则使用`VK_ATTACHMENT_STORE_OP_STORE`值；如果渲染后不需要内容，则使用`VK_ATTACHMENT_STORE_OP_DONT_CARE`值（此参数用于颜色附件或深度/模板附件的深度方面）。

    +   指定在渲染过程开始时，附件的模板（组件）应该如何处理，对于`stencilLoadOp`（与`loadOp`成员相同，但用于深度/模板附件的模板方面）

    +   指定在渲染过程结束后，附件的模板（组件）应该如何处理，对于`stencilStoreOp`（与`storeOp`相同，但用于深度/模板附件的模板方面）

    +   指定渲染过程开始时图像将具有的布局，对于`initialLayout`

    +   指定图像在渲染过程结束后应自动过渡到的布局，对于`finalLayout`

# 它是如何工作的...

当我们创建渲染过程时，我们必须创建一个附件描述数组。这是一个渲染过程中使用的所有附件的通用列表。然后，此数组中的索引用于子过程描述（参考*指定子过程描述*配方）。同样，当我们创建帧缓冲区并指定每个附件应使用的确切图像资源时，我们定义了一个列表，其中每个元素对应于附件描述数组中的元素。

通常，当我们绘制几何体时，我们至少将其渲染到一个颜色附件中。可能我们还想启用深度测试，因此还需要一个深度附件。此类常见场景的附件描述如下：

```cpp
std::vector<VkAttachmentDescription> attachments_descriptions = { 
  { 
    0, 
    VK_FORMAT_R8G8B8A8_UNORM, 
    VK_SAMPLE_COUNT_1_BIT, 
    VK_ATTACHMENT_LOAD_OP_CLEAR, 
    VK_ATTACHMENT_STORE_OP_STORE, 
    VK_ATTACHMENT_LOAD_OP_DONT_CARE, 
    VK_ATTACHMENT_STORE_OP_DONT_CARE, 
    VK_IMAGE_LAYOUT_UNDEFINED, 
    VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, 
  }, 
  { 
    0, 
    VK_FORMAT_D16_UNORM, 
    VK_SAMPLE_COUNT_1_BIT, 
    VK_ATTACHMENT_LOAD_OP_CLEAR, 
    VK_ATTACHMENT_STORE_OP_STORE, 
    VK_ATTACHMENT_LOAD_OP_DONT_CARE, 
    VK_ATTACHMENT_STORE_OP_DONT_CARE, 
    VK_IMAGE_LAYOUT_UNDEFINED, 
    VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 
  } 
};

```

在前面的示例中，我们指定了两个附件：一个使用`R8G8B8A8_UNORM`格式，另一个使用`D16_UNORM`格式。在渲染过程的开始时，这两个附件都应该被清除（类似于在帧开始时调用 OpenGL 的`glClear()`函数）。我们还想在渲染过程完成后保留第一个附件的内容，但不需要第二个附件的内容。对于两者，我们也指定了一个`UNDEFINED`初始布局。`UNDEFINED`布局始终可以用于初始/旧布局--这意味着当设置内存屏障时，我们不需要图像内容。

最终布局的值取决于我们打算在渲染通道之后如何使用图像。如果我们直接将渲染到交换链图像，并且希望在屏幕上显示它，我们应该使用`PRESENT_SRC`布局（如之前所示）。对于深度附件，如果我们不打算在渲染通道之后使用深度组件（这通常是真的），我们应该设置与渲染通道的最后一个子通道中指定的相同布局值。

也有可能渲染通道不使用任何附件。在这种情况下，我们不需要指定附件描述，但这种情况很少见。

# 参见

本章中的以下配方：

+   *指定子通道描述*

+   *创建渲染通道*

+   *创建帧缓冲区*

+   *准备渲染通道和具有颜色和深度附件的帧缓冲区*

# 指定子通道描述

在渲染通道中执行的操作被分组到子通道中。每个子通道代表我们渲染命令的一个阶段或一个阶段，其中使用渲染通道附件的一个子集（我们将数据渲染到其中或从中读取数据）。

渲染通道始终需要至少一个子通道，当开始一个渲染通道时，它会自动启动。对于每个子通道，我们需要准备一个描述。

# 准备工作

为了降低为每个子通道准备所需的参数数量，为此配方引入了一个自定义结构类型。它是 Vulkan 头文件中定义的`VkSubpassDescription`结构的一个简化版本。它具有以下定义：

```cpp
struct SubpassParameters { 
  VkPipelineBindPoint                  PipelineType; 
  std::vector<VkAttachmentReference>   InputAttachments; 
  std::vector<VkAttachmentReference>   ColorAttachments; 
  std::vector<VkAttachmentReference>   ResolveAttachments; 
  VkAttachmentReference const        * DepthStencilAttachment; 
  std::vector<uint32_t>                PreserveAttachments; 
};

```

`PipelineType`成员定义了在子通道期间将使用的管道类型（图形或计算，尽管目前渲染通道内部仅支持图形管道）。`InputAttachments`是我们将在子通道期间读取数据的附件集合。`ColorAttachments`指定所有将用作颜色附件的附件（我们将在此期间将其渲染）。`ResolveAttachments`指定在子通道结束时应该解析哪些颜色附件（从多采样图像更改为非多采样/单采样图像）。如果使用`DepthStencilAttachment`，则指定在子通道期间用作深度和/或模板附件的附件。`PreserveAttachments`是一组在子通道中未使用但必须在整个子通道期间保留内容的附件。

# 如何操作...

1.  创建一个名为`subpass_descriptions`的`std::vector<VkSubpassDescription>`类型的向量变量。对于在渲染通道中定义的每个子通道，向`subpass_descriptions`向量添加一个元素，并使用以下值为其成员：

    +   `flags`的`0`值

    +   `pipelineBindPoint`的`VK_PIPELINE_BIND_POINT_GRAPHICS`值（目前渲染通道内部仅支持图形管道）

    +   子通道中使用的输入附件数量为`inputAttachmentCount`

    +   指向具有输入附件参数的数组中第一个元素的指针（如果子通行中没有使用输入附件，则为`nullptr`值）用于`pInputAttachments`；对于`pInputAttachments`数组的每个成员，使用以下值：

        +   附件在所有渲染通行附件列表中的索引`attachment`。

        +   应在子通行开始时自动将给定的图像布局转换为`layout`。

    +   子通行中使用的颜色附件数量`colorAttachmentCount`。

    +   指向具有子通行颜色附件参数的数组中第一个元素的指针（如果子通行中没有使用颜色附件，则为`nullptr`值）用于`pColorAttachments`；对于数组的每个成员，指定如第 4a 点和第 4b 点所述的值。

    +   如果任何颜色附件应该被解析（从多采样变为单采样）对于`pResolveAttachments`，指定与`pColorAttachments`具有相同元素数量的数组中第一个元素的指针，或者如果不需要解析任何颜色附件，则使用`nullptr`值；`pResolveAttachments`数组的每个成员对应于相同索引的颜色附件，并指定在子通行结束时给定颜色附件应解析到的附件；对于数组的每个成员，使用如第 4a 点和第 4b 点所述的指定值；如果给定的颜色附件不应解析，则使用`VK_ATTACHMENT_UNUSED`值作为附件索引。

    +   对于`pDepthStencilAttachment`，如果使用了深度/模板附件，则提供一个指向类型为`VkAttachmentReference`的变量的指针（如果子通行中没有使用深度/模板附件，则为`nullptr`值）；对于此变量的成员，指定如第 4a 点和第 4b 点所述的值。

    +   应保留内容的未使用附件数量`preserveAttachmentCount`。

    +   指向数组中第一个元素的指针，该数组包含应保留内容的附件索引（如果没有附件需要保留，则为`nullptr`值）用于`pPreserveAttachments`。

# 它是如何工作的...

Vulkan 渲染通行必须至少有一个子通行。子通行参数定义在一个`VkSubpassDescription`元素数组中。每个这样的元素描述了在相应的子通行中如何使用附件。有单独的输入、颜色、解析和保留附件列表，以及深度/模板附件的单个条目。这些成员可能为空（或 null）。在这种情况下，对应类型的附件在子通行中不被使用。

描述的列表中的每个条目都是对在附件描述中为渲染通行指定的所有附件列表的引用（参见图*指定附件描述*）。此外，每个条目指定了一个图像在子通行期间应处于的布局。到指定布局的转换由驱动程序自动执行。

这里是一个使用自定义的`SubpassParameters`类型结构的代码示例，用于指定子通道定义：

```cpp
subpass_descriptions.clear(); 

for( auto & subpass_description : subpass_parameters ) { 
  subpass_descriptions.push_back( { 
    0, 
    subpass_description.PipelineType, 
static_cast<uint32_t>(subpass_description.InputAttachments.size()), 
    subpass_description.InputAttachments.data(), 
    static_cast<uint32_t>(subpass_description.ColorAttachments.size()), 
    subpass_description.ColorAttachments.data(), 
    subpass_description.ResolveAttachments.data(), 
    subpass_description.DepthStencilAttachment, 
    static_cast<uint32_t>(subpass_description.PreserveAttachments.size()), 
    subpass_description.PreserveAttachments.data() 
  } ); 
}

```

以下是一个定义一个与具有一个颜色附加项的示例相对应的子通道的代码示例：一个深度/模板附加项：

```cpp
VkAttachmentReference depth_stencil_attachment = { 
  1, 
  VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 
}; 

std::vector<SubpassParameters> subpass_parameters = { 
  { 
    VK_PIPELINE_BIND_POINT_GRAPHICS, 
    {}, 
    { 
      { 
        0, 
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL 
      } 
    }, 
    {}, 
    &depth_stencil_attachment, 
    {} 
  } 
};

```

首先，我们为描述深度/模板附加的`depth_stencil_attachment`变量指定一个值。对于深度数据，使用附加描述列表中的第二个附加项；这就是为什么我们为其索引指定了`1`的值（参考*指定附加描述*配方）。并且因为我们想渲染到这个附加项，所以我们为其布局提供了`VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL`的值（驱动程序将自动执行转换，如果需要的话）。

在示例中，我们只使用一个颜色附加项。它是附加描述列表中的第一个附加项，因此我们为其索引使用`0`的值。当我们渲染到颜色附加项时，我们应该为其布局指定`VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL`的值。

最后一点——因为我们想渲染几何图形，我们需要使用图形管道。这是通过为`PipelineType`成员提供一个`VK_PIPELINE_BIND_POINT_GRAPHICS`值来完成的。

由于我们不使用输入附加项，并且我们不希望解析任何颜色附加项，因此它们对应的向量是空的。

# 参见

本章中的以下配方：

+   *指定附加描述*

+   *创建渲染通道*

+   *创建帧缓冲区*

+   *准备用于几何渲染和后处理子通道的渲染通道*

+   *准备带有颜色和深度附加的渲染通道和帧缓冲区*

# 指定子通道之间的依赖关系

当给定子通道中的操作依赖于同一渲染通道中较早子通道中的操作的结果时，我们需要指定子通道依赖。如果记录在渲染通道内的操作与之前执行的操作之间或执行在渲染通道之后的操作与在渲染通道内执行的操作之间存在依赖关系，这也需要。也可以在单个子通道内定义依赖关系。

定义子通道依赖类似于设置内存屏障。

# 如何做到这一点...

1.  创建一个名为`subpass_dependencies`的`std::vector<VkSubpassDependency>`类型的变量。对于每个依赖项，向`subpass_dependencies`向量添加一个新元素，并为其成员使用以下值：

    +   对于`srcSubpass`，在第二组（“消耗”）操作之前（或对于渲染通道之前的`VK_SUBPASS_EXTERNAL`值）应该完成（“产生”）操作的子通道的索引

    +   对于`dstSubpass`，依赖于之前命令集（或渲染通道之后的`VK_SUBPASS_EXTERNAL`值）的操作的子通道的索引

    +   生成由`srcStageMask`的“消耗”命令读取结果的管道阶段集合

    +   依赖于`dstStageMask`的“生产”命令生成的数据的管道阶段集合

    +   对于`srcAccessMask`的“生产”命令发生的记忆操作类型

    +   在`dstAccessMask`的“消费”命令中将要执行的记忆操作类型

    +   对于`dependencyFlags`，如果依赖关系由区域定义，则使用`VK_DEPENDENCY_BY_REGION_BIT`值——这意味着为给定内存区域生成数据的操作必须在从同一区域读取数据的操作可以执行之前完成；如果没有指定此标志，则依赖关系是全局的，这意味着必须先生成整个图像的数据，然后才能执行“消费”命令。

# 它是如何工作的...

指定子阶段之间的依赖关系（或子阶段与渲染通道之前或之后的命令之间的依赖关系）与设置图像内存屏障非常相似，并且具有类似的目的。我们在想要指定一个子阶段的命令（或渲染通道之后的命令）依赖于另一个子阶段（或渲染通道之前执行的命令）的操作结果时这样做。我们不需要设置布局转换的依赖关系——这些是基于渲染通道附件和子阶段描述提供的信息自动执行的。更重要的是，当我们为不同的子阶段指定不同的附件布局，但在两个子阶段中，给定的附件都仅用于读取时，我们也不需要指定依赖关系。

当我们想在渲染通道内设置图像内存屏障时，也需要子阶段依赖关系。如果没有指定所谓的“自依赖关系”（源子阶段和目标子阶段的索引相同），我们无法做到这一点。然而，如果我们为给定的子阶段定义了这样的依赖关系，我们可以在其中记录一个内存屏障。在其他情况下，源子阶段索引必须低于目标子阶段索引（不包括`VK_SUBPASS_EXTERNAL`值）。

下面是一个示例，其中我们准备两个子阶段之间的依赖关系——第一个将几何体绘制到颜色和深度附件中，第二个使用颜色数据进行后处理（它从颜色附件中读取）：

```cpp
std::vector<VkSubpassDependency> subpass_dependencies = { 
  { 
    0, 
    1, 
    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 
    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 
    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, 
    VK_ACCESS_INPUT_ATTACHMENT_READ_BIT, 
    VK_DEPENDENCY_BY_REGION_BIT 
  } 
};

```

上述依赖关系设置在第一个和第二个子通道之间（索引值为 0 和 1）。对颜色附件的写入操作在 `COLOR_ATTACHMENT_OUTPUT` 阶段执行。后处理在片段着色器中完成，此阶段被定义为“消耗”阶段。当我们绘制几何体时，我们对颜色附件执行写入操作（访问掩码的值为 `COLOR_ATTACHMENT_WRITE`）。然后颜色附件被用作输入附件，在后处理子通道中我们从它读取（因此我们使用一个值为 `INPUT_ATTACHMENT_READ` 的访问掩码）。由于我们不需要从图像的其他部分读取数据，我们可以通过区域指定依赖关系（一个片段在第一个子通道中存储给定坐标的颜色值，下一个子通道中具有相同坐标的片段读取相同的值）。当我们这样做时，我们不应假设区域大于单个像素，因为区域的大小可能在不同的硬件平台上不同。

# 参见

本章中以下配方：

+   *指定附件描述*

+   *指定子通道描述*

+   *创建渲染通道*

+   *准备几何体渲染和后处理子通道的渲染通道*

# 创建渲染通道

渲染（绘制几何体）只能在渲染通道内执行。当我们还想要执行其他操作，例如图像后处理或准备几何体和光照预通道数据时，我们需要将这些操作排序到子通道中。为此，我们指定所有必需的附件描述、所有将操作分组到其中的子通道，以及这些操作之间必要的依赖关系。当这些数据准备就绪后，我们可以创建一个渲染通道。

# 准备工作

为了减少需要提供的参数数量，在本配方中，我们使用一个自定义的 `SubpassParameters` 类型结构（参考 *指定子通道描述* 配方）。

# 如何操作...

1.  创建一个名为 `attachments_descriptions` 的 `std::vector<VkAttachmentDescription>` 类型的变量，在其中指定所有渲染通道附件的描述（参考 *指定附件描述* 配方）。

1.  准备一个名为 `subpass_descriptions` 的 `std::vector<VkSubpassDescription>` 类型的变量，并使用它来定义子通道的描述（参考 *指定子通道描述* 配方）。

1.  创建一个名为 `subpass_dependencies` 的 `std::vector<VkSubpassDependency>` 类型的变量。为渲染通道中需要定义的每个依赖关系向此向量添加一个新成员（参考 *指定子通道之间的依赖关系* 配方）。

1.  创建一个名为 `render_pass_create_info` 的 `VkRenderPassCreateInfo` 类型的变量，并使用以下值初始化其成员：

    +   `sType` 的值为 `VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO`

    +   `pNext` 的值为 `nullptr`

    +   `flags` 的值为 `0`

    +   `attachments_descriptions` 向量中 `attachmentCount` 元素的个数

    +   `attachments_descriptions` 向量第一个元素的指针（如果为空，则为 `nullptr`）用于 `pAttachments`

    +   `subpass_descriptions` 向量中 `subpassCount` 的元素数量

    +   `subpass_descriptions` 向量第一个元素的指针用于 `pSubpasses`

    +   `subpass_dependencies` 向量中 `dependencyCount` 的元素数量

    +   `subpass_dependencies` 向量第一个元素的指针（如果为空，则为 `nullptr`）用于 `pDependencies`

1.  获取应创建渲染通道的逻辑设备的句柄。将其存储在名为 `logical_device` 的 `VkDevice` 类型变量中。

1.  在名为 `render_pass` 的 `VkRenderPass` 类型变量中创建一个变量，其中将存储创建的渲染通道的句柄。

1.  调用 `vkCreateRenderPass(logical_device, &render_pass_create_info, nullptr, &render_pass)`。对于调用，提供 `logical_device` 变量、`render_pass_create_info` 变量的指针、一个 `nullptr` 值以及 `render_pass` 变量的指针。

1.  通过检查是否返回了 `VK_SUCCESS` 值来确保调用成功。

# 它是如何工作的...

渲染通道定义了所有子通道中操作使用附件的通用信息。这允许驱动程序优化工作并提高我们应用程序的性能。

![](img/image_06_001.png)

渲染通道创建最重要的部分是数据准备——所有使用附件和子通道的描述以及子通道之间依赖关系的指定（参考本章中的 *指定附件描述*、*指定子通道描述* 和 *指定子通道之间的依赖关系* 章节中的食谱）。以下步骤可以简要表示如下：

```cpp
SpecifyAttachmentsDescriptions( attachments_descriptions ); 

std::vector<VkSubpassDescription> subpass_descriptions; 
SpecifySubpassDescriptions( subpass_parameters, subpass_descriptions ); 

SpecifyDependenciesBetweenSubpasses( subpass_dependencies );

```

然后当指定创建渲染通道函数的参数时使用这些数据：

```cpp
VkRenderPassCreateInfo render_pass_create_info = { 
  VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO, 
  nullptr, 
  0, 
  static_cast<uint32_t>(attachments_descriptions.size()), 
  attachments_descriptions.data(), 
  static_cast<uint32_t>(subpass_descriptions.size()), 
  subpass_descriptions.data(), 
  static_cast<uint32_t>(subpass_dependencies.size()), 
  subpass_dependencies.data() 
}; 

VkResult result = vkCreateRenderPass( logical_device, &render_pass_create_info, nullptr, &render_pass ); 
if( VK_SUCCESS != result ) { 
  std::cout << "Could not create a render pass." << std::endl; 
  return false; 
} 
return true;

```

但是，为了正确执行绘图操作，渲染通道是不够的，因为它只指定了操作如何按顺序进入子通道以及如何使用附件。没有关于用于这些附件的图像的信息。关于所有定义的附件使用的特定资源的信息存储在帧缓冲区中。

# 参见

本章中的以下食谱：

+   *指定附件描述*

+   *指定子通道描述*

+   *指定子通道之间的依赖关系*

+   *创建帧缓冲区*

+   *开始渲染通道*

+   *推进到下一个子通道*

+   *结束渲染通道*

+   *销毁渲染通道*

# 创建帧缓冲区

帧缓冲区与渲染通道一起使用。它们指定了在渲染通道中定义的相应附件应使用哪些图像资源。它们还定义了可渲染区域的大小。这就是为什么当我们想要记录绘图操作时，我们不仅需要创建渲染通道，还需要创建帧缓冲区。

# 如何操作...

1.  获取应与帧缓冲区兼容的渲染通道句柄，并使用它初始化一个名为 `render_pass` 的 `VkRenderPass` 类型变量。

1.  准备一个表示图像子资源的图像视图句柄列表，这些子资源应用于渲染通道附件。将所有准备好的图像视图存储在名为 `attachments` 的 `std::vector<VkImageView>` 类型变量中。

1.  创建一个名为 `framebuffer_create_info` 的 `VkFramebufferCreateInfo` 类型变量。使用以下值初始化其成员：

    +   `VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO` 值用于 `sType`

    +   `pNext` 的 `nullptr` 值

    +   `0` 值用于 `flags`

    +   `render_pass` 变量用于 `renderPass`

    +   `attachments` 向量中的元素数量用于 `attachmentCount`

    +   指向 `attachments` 向量第一个元素的指针（如果为空，则为 `nullptr` 值）用于 `pAttachments`

    +   为 `width` 选择可渲染区域的宽度

    +   选择帧缓冲区的高度用于 `height`

    +   `layers` 的帧缓冲区层数

1.  获取用于创建并存储在名为 `logical_device` 的 `VkDevice` 类型变量中的帧缓冲区句柄。

1.  创建一个名为 `framebuffer` 的 `VkFramebuffer` 类型变量，它将使用创建的帧缓冲区的句柄进行初始化。

1.  调用 `vkCreateFramebuffer( logical_device, &framebuffer_create_info, nullptr, &framebuffer )`，我们提供 `logical_device` 变量、`framebuffer_create_info` 变量的指针、一个 `nullptr` 值和一个指向 `framebuffer` 变量的指针。

1.  确保帧缓冲区已正确创建，通过检查调用是否返回了 `VK_SUCCESS` 值。

# 它是如何工作的...

帧缓冲区总是与渲染通道一起创建。它们定义了应用于渲染通道中指定附件的特定图像子资源，因此这两个对象类型应相互对应。

![](img/image_06_002.png)

当我们创建帧缓冲区时，我们提供一个渲染通道对象，我们可以使用该对象使用给定的帧缓冲区。然而，我们不仅限于仅与指定的渲染通道一起使用它。我们还可以使用与提供的渲染通道兼容的所有渲染通道。

兼容的渲染通道是什么？首先，它们必须具有相同数量的子通道。每个子通道必须具有兼容的输入、颜色、解析和深度/模板附件集合。这意味着相应的附件的格式和样本数必须相同。然而，附件可以具有不同的初始、子通道和最终布局以及不同的加载和存储操作。

除了这些，帧缓冲区还定义了可渲染区域的尺寸--所有渲染都将被限制的维度。然而，我们需要记住的是，确保指定范围之外的像素/片段不被修改的责任在我们身上。为此，我们需要在管道创建期间或设置相应的动态状态时指定适当的参数（视口和裁剪测试）（参考第八章，*图形和计算管道*中的*准备视口和裁剪测试状态*食谱，以及第九章，*命令记录和绘制*中的*设置动态视口和裁剪状态*食谱）。

我们必须确保渲染只发生在在帧缓冲区创建期间指定的维度内。

当我们在命令缓冲区中开始一个渲染通道并使用给定的帧缓冲区时，我们还需要确保在该帧缓冲区中指定的图像子资源不用于任何其他目的。换句话说，如果我们将图像的某个部分用作帧缓冲区附件，那么在渲染通道期间我们不能以任何其他方式使用它。

为渲染通道附件指定的图像子资源不能在渲染通道的开始和结束之间用于任何其他（非附件）目的。

下面是一个负责创建帧缓冲区的代码示例：

```cpp
VkFramebufferCreateInfo framebuffer_create_info = { 
  VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, 
  nullptr, 
  0, 
  render_pass, 
  static_cast<uint32_t>(attachments.size()), 
  attachments.data(), 
  width, 
  height, 
  layers 
}; 

VkResult result = vkCreateFramebuffer( logical_device, &framebuffer_create_info, nullptr, &framebuffer ); 
if( VK_SUCCESS != result ) { 
  std::cout << "Could not create a framebuffer." << std::endl; 
  return false; 
} 
return true;

```

# 参见

在第四章，*资源和内存*，查看以下食谱：

+   *创建一个图像*

+   *创建一个图像视图*

本章中的以下食谱：

+   *指定附件描述*

+   *创建一个帧缓冲区*

# 准备几何渲染和后处理子通道的渲染通道

在开发游戏或 CAD 工具等应用程序时，我们经常需要先绘制几何图形，然后在整个场景渲染完毕后，应用称为后处理的附加图像效果。

在这个示例食谱中，我们将看到如何准备一个渲染通道，其中我们将有两个子通道。第一个子通道渲染到两个附件中--颜色和深度。第二个子通道从第一个颜色附件中读取数据并渲染到另一个颜色附件中--一个可以在渲染通道之后呈现（显示在屏幕上）的交换链图像。

# 准备工作

为了减少需要提供的参数数量，在这个食谱中我们使用一个自定义的结构体类型`SubpassParameters`（参考*指定子通道描述*食谱）。

# 如何做...

1.  创建一个名为`attachments_descriptions`的类型为`std::vector<VkAttachmentDescription>`的变量。向`attachments_descriptions`向量添加一个元素，描述第一个颜色附件。使用以下值初始化它：

    +   `flags`的`0`值

    +   `format`的`VK_FORMAT_R8G8B8A8_UNORM`值

    +   `samples`的`VK_SAMPLE_COUNT_1_BIT`值

    +   `loadOp`的`VK_ATTACHMENT_LOAD_OP_CLEAR`值

    +   `storeOp` 的 `VK_ATTACHMENT_STORE_OP_DONT_CARE` 值

    +   `stencilLoadOp` 的 `VK_ATTACHMENT_LOAD_OP_DONT_CARE` 值

    +   `stencilStoreOp` 的 `VK_ATTACHMENT_STORE_OP_DONT_CARE` 值

    +   `initialLayout` 的 `VK_IMAGE_LAYOUT_UNDEFINED` 值

    +   `finalLayout` 的 `VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL` 值

1.  向 `attachments_descriptions` 向量中添加另一个元素，指定深度/模板附件。使用以下值初始化其成员：

    +   `flags` 的 `0` 值

    +   `format` 的 `VK_FORMAT_D16_UNORM` 值

    +   `samples` 的 `VK_SAMPLE_COUNT_1_BIT` 值

    +   `loadOp` 的 `VK_ATTACHMENT_LOAD_OP_CLEAR` 值

    +   `storeOp` 的 `VK_ATTACHMENT_STORE_OP_DONT_CARE` 值

    +   `stencilLoadOp` 的 `VK_ATTACHMENT_LOAD_OP_DONT_CARE` 值

    +   `stencilStoreOp` 的 `VK_ATTACHMENT_STORE_OP_DONT_CARE` 值

    +   `initialLayout` 的 `VK_IMAGE_LAYOUT_UNDEFINED` 值

    +   `finalLayout` 的 `VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL` 值

1.  向 `attachments_descriptions` 向量中添加第三个元素。这次它将指定另一个颜色附件。使用以下值初始化它：

    +   `flags` 的 `0` 值

    +   `format` 的 `VK_FORMAT_R8G8B8A8_UNORM` 值

    +   `samples` 的 `VK_SAMPLE_COUNT_1_BIT` 值

+   `loadOp` 的 `VK_ATTACHMENT_LOAD_OP_CLEAR` 值

+   `storeOp` 的 `VK_ATTACHMENT_STORE_OP_STORE` 值

+   `stencilLoadOp` 的 `VK_ATTACHMENT_LOAD_OP_DONT_CARE` 值

+   `stencilStoreOp` 的 `VK_ATTACHMENT_STORE_OP_DONT_CARE` 值

+   `initialLayout` 的 `VK_IMAGE_LAYOUT_UNDEFINED` 值

+   `finalLayout` 的 `VK_IMAGE_LAYOUT_PRESENT_SRC_KHR` 值

1.  创建一个名为 `depth_stencil_attachment` 的 `VkAttachmentReference` 类型的变量，并使用以下值初始化它：

    +   `attachment` 的 `1` 值

    +   `layout` 的 `VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL` 值

1.  创建一个名为 `subpass_parameters` 的 `std::vector<SubpassParameters>` 类型的变量，并向此向量添加一个具有以下值的元素：

    +   `PipelineType` 的 `VK_PIPELINE_BIND_POINT_GRAPHICS` 值

    +   用于 `InputAttachments` 的空向量

    +   `ColorAttachments` 的一个元素和一个以下值的向量：

        +   `attachment` 的 `0` 值

        +   `layout` 的 `VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL` 值

    +   用于 `ResolveAttachments` 的空向量

    +   `DepthStencilAttachment` 的 `depth_stencil_attachment` 变量的指针

    +   用于 `PreserveAttachments` 的空向量

1.  向 `subpass_parameters` 中添加第二个元素，描述第二个子通道。使用以下值初始化其成员：

    +   `PipelineType` 的 `VK_PIPELINE_BIND_POINT_GRAPHICS` 值

    +   `InputAttachments` 的一个元素和一个以下值的向量：

        +   `attachment` 的 `0` 值

        +   `layout` 的 `VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL` 值

    +   `ColorAttachments` 的一个元素和一个以下值的向量：

        +   `attachment` 的 `2` 值

        +   `layout` 的 `VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL` 值

    +   用于 `ResolveAttachments` 的空向量

    +   `DepthStencilAttachment` 的 `nullptr` 值

    +   用于 `PreserveAttachments` 的空向量

1.  创建一个名为 `subpass_dependencies` 的 `std::vector<VkSubpassDependency>` 类型的变量，它只有一个元素，其成员使用以下值：

    +   `srcSubpass` 的 `0` 值

    +   `dstSubpass` 的 `1` 值

    +   `srcStageMask` 的 `VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT` 值

    +   `dstStageMask` 的 `VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT` 值

    +   `srcAccessMask` 的 `VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT` 值

    +   `dstAccessMask` 的 `VK_ACCESS_INPUT_ATTACHMENT_READ_BIT` 值

    +   `dependencyFlags` 的 `VK_DEPENDENCY_BY_REGION_BIT` 值

1.  使用 `attachments_descriptions`、`subpass_parameters` 和 `subpass_dependencies` 变量创建渲染通道。将句柄存储在名为 `render_pass` 的 `VkRenderPass` 类型的变量中（参考本章中的 *创建渲染通道* 配方）。

# 它是如何工作的...

在这个配方中，我们创建了一个包含三个附件的渲染通道。它们被指定如下：

```cpp
std::vector<VkAttachmentDescription> attachments_descriptions = { 
  { 
    0, 
    VK_FORMAT_R8G8B8A8_UNORM, 
    VK_SAMPLE_COUNT_1_BIT, 
    VK_ATTACHMENT_LOAD_OP_CLEAR, 
    VK_ATTACHMENT_STORE_OP_DONT_CARE, 
    VK_ATTACHMENT_LOAD_OP_DONT_CARE, 
    VK_ATTACHMENT_STORE_OP_DONT_CARE, 
    VK_IMAGE_LAYOUT_UNDEFINED, 
    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 
  }, 
  { 
    0, 
    VK_FORMAT_D16_UNORM, 
    VK_SAMPLE_COUNT_1_BIT, 
    VK_ATTACHMENT_LOAD_OP_CLEAR, 
    VK_ATTACHMENT_STORE_OP_DONT_CARE, 
    VK_ATTACHMENT_LOAD_OP_DONT_CARE, 
    VK_ATTACHMENT_STORE_OP_DONT_CARE, 
    VK_IMAGE_LAYOUT_UNDEFINED, 
    VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 
  }, 
  { 
    0, 
    VK_FORMAT_R8G8B8A8_UNORM, 
    VK_SAMPLE_COUNT_1_BIT, 
    VK_ATTACHMENT_LOAD_OP_CLEAR, 
    VK_ATTACHMENT_STORE_OP_STORE, 
    VK_ATTACHMENT_LOAD_OP_DONT_CARE, 
    VK_ATTACHMENT_STORE_OP_DONT_CARE, 
    VK_IMAGE_LAYOUT_UNDEFINED, 
    VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, 
  }, 
};

```

首先有一个颜色附件，我们在第一个子阶段向其中渲染，并在第二个子阶段从中读取。第二个附件用于深度数据；第三个是另一个颜色附件，我们在第二个子阶段向其中渲染。由于我们不需要在渲染通道之后第一个和第二个附件的内容（我们只需要在第二个子阶段中第一个附件的内容），所以我们为它们的存储操作指定了 `VK_ATTACHMENT_STORE_OP_DONT_CARE` 值。我们也不需要在渲染通道开始时需要它们的内容，所以我们指定了一个 `UNDEFINED` 初始布局。我们还清除了所有三个附件。

接下来我们定义两个子阶段：

```cpp
VkAttachmentReference depth_stencil_attachment = { 
  1, 
  VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 
}; 

std::vector<SubpassParameters> subpass_parameters = { 
  // #0 subpass 
  { 
    VK_PIPELINE_BIND_POINT_GRAPHICS, 
    {}, 
    { 
      { 
        0, 
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL 
      } 
    }, 
    {}, 
    &depth_stencil_attachment, 
    {} 
  }, 
  // #1 subpass 
  { 
    VK_PIPELINE_BIND_POINT_GRAPHICS, 
    { 
      { 
        0, 
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL 
      } 
    }, 
    { 
      { 
        2, 
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL 
      } 
    }, 
    {}, 
    nullptr, 
    {} 
  } 
};

```

第一个子阶段使用颜色附件和深度附件。第二个子阶段从第一个附件（在此用作输入附件）读取，并将渲染到第三个附件中。

最后，我们需要定义两个子阶段之间第一个附件的依赖关系，该附件最初是颜色附件（我们向其中写入数据），然后是输入附件（我们从其中读取数据）。之后，我们可以像这样创建渲染通道：

```cpp
std::vector<VkSubpassDependency> subpass_dependencies = { 
  { 
    0, 
    1, 
    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 
    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 
    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, 
    VK_ACCESS_INPUT_ATTACHMENT_READ_BIT, 
    VK_DEPENDENCY_BY_REGION_BIT 
  } 
}; 

if( !CreateRenderPass( logical_device, attachments_descriptions, subpass_parameters, subpass_dependencies, render_pass ) ) { 
  return false; 
} 
return true;

```

# 参见

本章中的以下配方：

+   *指定附件描述*

+   *指定子阶段描述*

+   *指定子阶段之间的依赖关系*

+   *创建渲染通道*

# 准备带有颜色和深度附件的渲染通道和帧缓冲区

渲染 3D 场景通常不仅涉及颜色附件，还涉及用于深度测试的深度附件（我们希望远离相机的对象被靠近相机的对象遮挡）。

在这个示例配方中，我们将看到如何创建用于颜色和深度数据的图像，以及一个具有单个子阶段的渲染通道，该子阶段将渲染到颜色和深度附件中。我们还将创建一个帧缓冲区，该帧缓冲区将使用这两个图像作为渲染通道的附件。

# 准备工作

与本章中较早的配方一样，在这个配方中，我们将使用类型为 `SubpassParameters` 的自定义结构（参考 *指定子阶段描述* 配方）。

# 如何做到这一点...

1.  创建一个带有 `VK_FORMAT_R8G8B8A8_UNORM` 格式、`VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT` 用法和 `VK_IMAGE_ASPECT_COLOR_BIT` 方面的 2D 图像及其视图。选择图像的其余参数。将创建的句柄存储在名为 `color_image` 的 `VkImage` 类型的变量中，名为 `color_image_memory_object` 的 `VkDeviceMemory` 类型的变量中，以及名为 `color_image_view` 的 `VkImageView` 类型的变量中（参考第四章 Creating a 2D image and view 中的配方，*资源和内存*）。

1.  创建一个带有 `VK_FORMAT_D16_UNORM` 格式、`VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT` 用法、`VK_IMAGE_ASPECT_DEPTH_BIT` 方面和与存储在 `color_image` 变量中的图像相同大小的第二个 2D 图像及其视图。选择图像的其余参数。将创建的句柄存储在名为 `depth_image` 的 `VkImage` 类型的变量中，名为 `depth_image_memory_object` 的 `VkDeviceMemory` 类型的变量中，以及名为 `depth_image_view` 的 `VkImageView` 类型的变量中（参考第四章 Creating a 2D image and view 中的配方，*资源和内存*）。

1.  创建一个名为 `attachments_descriptions` 的 `std::vector<VkAttachmentDescription>` 类型的变量，并向该向量添加两个元素。使用以下值初始化第一个元素：

    +   `0` 的值用于 `flags`

    +   `VK_FORMAT_R8G8B8A8_UNORM` 的值用于 `format`

    +   `VK_SAMPLE_COUNT_1_BIT` 的值用于 `samples`

    +   `VK_ATTACHMENT_LOAD_OP_CLEAR` 的值用于 `loadOp`

    +   `VK_ATTACHMENT_STORE_OP_STORE` 的值用于 `storeOp`

    +   `VK_ATTACHMENT_LOAD_OP_DONT_CARE` 的值用于 `stencilLoadOp`

    +   `VK_ATTACHMENT_STORE_OP_DONT_CARE` 的值用于 `stencilStoreOp`

    +   `VK_IMAGE_LAYOUT_UNDEFINED` 的值用于 `initialLayout`

    +   `VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL` 的值用于 `finalLayout`

1.  使用这些值初始化 `attachments_descriptions` 向量第二个元素的成员：

    +   `0` 的值用于 `flags`

    +   `VK_FORMAT_D16_UNORM` 的值用于 `format`

    +   `VK_SAMPLE_COUNT_1_BIT` 的值用于 `samples`

    +   `VK_ATTACHMENT_LOAD_OP_CLEAR` 的值用于 `loadOp`

    +   `VK_ATTACHMENT_STORE_OP_STORE` 的值用于 `storeOp`

    +   `VK_ATTACHMENT_LOAD_OP_DONT_CARE` 的值用于 `stencilLoadOp`

    +   `VK_ATTACHMENT_STORE_OP_DONT_CARE` 的值用于 `stencilStoreOp`

    +   `VK_IMAGE_LAYOUT_UNDEFINED` 的值用于 `initialLayout`

    +   `VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL` 的值用于 `finalLayout`

1.  创建一个名为 `depth_stencil_attachment` 的 `VkAttachmentReference` 类型的变量，并使用以下值进行初始化：

    +   `1` 的值用于 `attachment`

    +   `VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL` 的值用于 `layout`

1.  创建一个名为 `subpass_parameters` 的 `std::vector<SubpassParameters>` 类型的向量。向该向量添加一个元素，并使用以下值进行初始化：

    +   `VK_PIPELINE_BIND_POINT_GRAPHICS` 的值用于 `PipelineType`

    +   `InputAttachments` 的一个空向量

    +   一个只有一个元素的向量，这些值用于 `ColorAttachments`：

        1.  `0`的值用于`attachment`

        1.  `VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL`的值用于`layout`

    +   一个空的向量用于`ResolveAttachments`

    +   一个指向`depth_stencil_attachment`变量的指针，用于`DepthStencilAttachment`

    +   一个空的向量用于`PreserveAttachments`

1.  创建一个名为`subpass_dependencies`的`std::vector<VkSubpassDependency>`类型的向量，使用这些值初始化单个元素：

    +   `0`的值用于`srcSubpass`

    +   `VK_SUBPASS_EXTERNAL`的值用于`dstSubpass`

    +   `VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT`的值用于`srcStageMask`

    +   `VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT`的值用于`dstStageMask`

    +   `VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT`的值用于`srcAccessMask`

    +   `VK_ACCESS_SHADER_READ_BIT`的值用于`dstAccessMask`

    +   `0`的值用于`dependencyFlags`

1.  使用`attachments_descriptions`、`subpass_parameters`和`subpass_dependencies`向量创建一个名为`render_pass`的`VkRenderPass`类型的变量。将创建的渲染通道句柄存储在名为`render_pass`的变量中（参考本章中的*创建渲染通道*配方）。

1.  使用`render_pass`变量和`color_image_view`变量作为其第一个附件，以及`depth_image_view`变量作为第二个附件创建一个帧缓冲区。指定与`color_image`和`depth_image`变量相同的维度。将创建的帧缓冲区句柄存储在名为`framebuffer`的`VkFramebuffer`类型的变量中。

# 它是如何工作的...

在这个示例配方中，我们希望将渲染到两个图像中——一个用于颜色数据，另一个用于深度数据。我们暗示在渲染通道之后它们将被用作纹理（我们将在另一个渲染通道的着色器中采样它们）；这就是为什么它们被创建为`COLOR_ATTACHMENT` / `DEPTH_STENCIL_ATTACHMENT`用法（以便我们可以将渲染到它们中）和`SAMPLED`用法（以便它们都可以在着色器中被采样）：

```cpp
if( !Create2DImageAndView( physical_device, logical_device, VK_FORMAT_R8G8B8A8_UNORM, { width, height }, 1, 1, VK_SAMPLE_COUNT_1_BIT, 

  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT, color_image, color_image_memory_object, color_image_view ) ) { 

  return false; 
} 

if( !Create2DImageAndView( physical_device, logical_device, VK_FORMAT_D16_UNORM, { width, height }, 1, 1, VK_SAMPLE_COUNT_1_BIT, 

  VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, depth_image, depth_image_memory_object, depth_image_view ) ) { 
  return false; 
}

```

接下来，我们指定渲染通道的两个附件。它们都在渲染通道的开始时清除，并在渲染通道之后保留其内容：

```cpp
std::vector<VkAttachmentDescription> attachments_descriptions = { 
  { 
    0, 
    VK_FORMAT_R8G8B8A8_UNORM, 
    VK_SAMPLE_COUNT_1_BIT, 
    VK_ATTACHMENT_LOAD_OP_CLEAR, 
    VK_ATTACHMENT_STORE_OP_STORE, 
    VK_ATTACHMENT_LOAD_OP_DONT_CARE, 
    VK_ATTACHMENT_STORE_OP_DONT_CARE, 
    VK_IMAGE_LAYOUT_UNDEFINED, 
    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 
  }, 
  { 
    0, 
    VK_FORMAT_D16_UNORM, 
    VK_SAMPLE_COUNT_1_BIT, 
    VK_ATTACHMENT_LOAD_OP_CLEAR, 
    VK_ATTACHMENT_STORE_OP_STORE, 
    VK_ATTACHMENT_LOAD_OP_DONT_CARE, 
    VK_ATTACHMENT_STORE_OP_DONT_CARE, 
    VK_IMAGE_LAYOUT_UNDEFINED, 
    VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, 
  } 
};

```

下一步是定义一个单独的子通道。它使用第一个附件进行颜色写入，第二个附件进行深度/模板数据：

```cpp
VkAttachmentReference depth_stencil_attachment = { 
  1, 
  VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 
}; 

std::vector<SubpassParameters> subpass_parameters = { 
  { 
    VK_PIPELINE_BIND_POINT_GRAPHICS, 
    {}, 
    { 
      { 
        0, 
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL 
      } 
    }, 
    {}, 
    &depth_stencil_attachment, 
    {} 
  } 
};

```

最后，我们定义子通道与渲染通道之后将执行的命令之间的依赖关系。这是必需的，因为我们不希望其他命令在渲染通道的内容完全写入之前开始读取我们的图像。我们还创建了渲染通道和帧缓冲区：

```cpp
std::vector<VkSubpassDependency> subpass_dependencies = { 
  { 
    0, 
    VK_SUBPASS_EXTERNAL, 
    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 
    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 
    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, 
    VK_ACCESS_SHADER_READ_BIT, 
    0 
  } 
}; 

if( !CreateRenderPass( logical_device, attachments_descriptions, subpasses_parameters, subpasses_dependencies, render_pass ) ) { 
  return false; 
} 

if( !CreateFramebuffer( logical_device, render_pass, { color_image_view, depth_image_view }, width, height, 1, framebuffer ) ) { 
  return false; 
} 
return true;

```

# 参见

+   在第四章，*资源和内存*中，查看以下配方：

    +   *创建二维图像和视图*

+   本章中的以下配方：

    +   *指定附件描述*

    +   *指定子通道描述*

    +   *指定子通道之间的依赖关系*

    +   *创建渲染通道*

    +   *创建帧缓冲区*

# 开始渲染通道

当我们创建了一个渲染通道和帧缓冲区，并且准备开始记录渲染几何形状所需的命令时，我们必须记录一个开始渲染通道的操作。这也会自动开始其第一个子通道。

# 如何操作...

1.  获取存储在类型为 `VkCommandBuffer` 的变量 `command_buffer` 中的命令缓冲区句柄。确保命令缓冲区处于记录状态。

1.  使用渲染通道的句柄初始化一个类型为 `VkRenderPass` 的变量 `render_pass`。

1.  获取与 `render_pass` 兼容的帧缓冲区，并将其句柄存储在类型为 `VkFramebuffer` 的变量 `framebuffer` 中。

1.  指定渲染通道期间渲染将被限制的渲染区域的尺寸。此区域不能大于帧缓冲区指定的尺寸。将尺寸存储在类型为 `VkRect2D` 的变量 `render_area` 中。

1.  创建一个类型为 `std::vector<VkClearValue>` 的变量，命名为 `clear_values`，其元素数量等于渲染通道中附件的数量。对于每个使用清除 `loadOp` 的渲染通道附件，提供与附件索引相同的索引处的相应清除值。

1.  准备一个类型为 `VkSubpassContents` 的变量 `subpass_contents`，描述第一个子通道中操作的记录方式。如果命令直接记录且没有执行二级命令缓冲区，则使用 `VK_SUBPASS_CONTENTS_INLINE` 值；或使用 `VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS` 值来指定子通道的命令存储在二级命令缓冲区中，并且仅使用执行二级命令缓冲区命令（参考第九章 *执行主命令缓冲区内的二级命令缓冲区* 的配方，*命令记录和绘制*）。

1.  创建一个类型为 `VkRenderPassBeginInfo` 的变量，命名为 `render_pass_begin_info`，并使用这些值初始化其成员：

    +   `VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO` 值用于 `sType`

    +   `nullptr` 值用于 `pNext`

    +   `render_pass` 变量用于 `renderPass`

    +   `framebuffer` 变量用于 `framebuffer`

    +   `render_area` 变量用于 `renderArea`

    +   `clear_values` 向量中的元素数量用于 `clearValueCount`

    +   `clear_values` 向量第一个元素的指针（如果为空，则为 `nullptr` 值）用于 `pClearValues`

1.  调用 `vkCmdBeginRenderPass(command_buffer, &render_pass_begin_info, subpass_contents)`，提供 `command_buffer` 变量、`render_pass_begin_info` 变量的指针和 `subpass_contents` 变量。

# 它是如何工作的...

开始渲染通道会自动开始其第一个子通道。在此操作完成之前，所有指定了清除 `loadOp` 的附件都会被清除--填充为单一颜色。用于清除的值（以及启动渲染通道所需的其他参数）在类型为 `VkRenderPassBeginInfo` 的变量中指定：

```cpp
VkRenderPassBeginInfo render_pass_begin_info = { 
  VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, 
  nullptr, 
  render_pass, 
  framebuffer, 
  render_area, 
  static_cast<uint32_t>(clear_values.size()), 
  clear_values.data() 
};

```

清除值的数组必须有至少与从开始到最后清除的附件（正在清除的具有最大索引的附件）对应的元素。最好有与渲染通道中附件数量相同的清除值，但我们只需要为清除的提供值。如果没有附件被清除，我们可以为清除值数组提供一个`nullptr`值。

当我们开始渲染通道时，我们还需要提供渲染区域的尺寸。它可以与帧缓冲区的尺寸一样大，但可以更小。确保渲染被限制在指定的区域取决于我们，或者此范围之外的像素可能成为未定义的。

要开始渲染通道，我们需要调用：

```cpp
vkCmdBeginRenderPass( command_buffer, &render_pass_begin_info, subpass_contents );

```

# 参见

+   在第三章，*命令缓冲区和同步*中，查看以下配方：

    +   *开始命令缓冲区记录操作*

+   在第九章，*命令记录和绘制*中，查看以下配方：

    +   *在主命令缓冲区内部执行二级命令缓冲区*

+   本章中的以下配方：

    +   *创建渲染通道*

    +   *创建帧缓冲区*

# 进入下一个子通道

在渲染通道内记录的命令被分为子通道。当给定子通道的一组命令已经记录，而我们想要记录另一个子通道的命令时，我们需要切换（或进入）下一个子通道。

# 如何操作...

1.  获取正在记录的命令缓冲区的句柄，并将其存储在名为`command_buffer`的`VkCommandBuffer`类型变量中。确保开始渲染通道的操作已经记录在`command_buffer`中。

1.  指定子通道命令的记录方式：直接或通过二级命令缓冲区。将适当的值存储在名为`subpass_contents`的`VkSubpassContents`类型变量中（参考*开始渲染通道*配方）。

1.  调用`vkCmdNextSubpass(command_buffer, subpass_contents)`。对于调用，提供`command_buffer`和`subpass_contents`变量。

# 它是如何工作的...

进入下一个子通道会将当前渲染通道切换到下一个子通道。在此操作过程中，将执行适当的布局转换，并引入内存和执行依赖（类似于内存屏障）。所有这些操作都由驱动程序自动执行，如果需要，以便新子通道中的附件可以按照在渲染通道创建期间指定的方式使用。进入下一个子通道还会对指定的颜色附件执行多采样解析操作。

子通道中的命令可以直接记录，通过在命令缓冲区中内联它们，或者间接通过执行二级命令缓冲区。

要记录从一个子通道切换到另一个子通道的操作，我们需要调用一个单一函数：

```cpp
vkCmdNextSubpass( command_buffer, subpass_contents );

```

# 参见

本章中的以下食谱：

+   *指定子通道描述*

+   *创建渲染通道*

+   *开始渲染通道*

+   *结束渲染通道*

# 结束渲染通道

当所有子通道的所有命令都已记录时，我们需要结束（停止或完成）渲染通道。

# 如何操作...

1.  取出命令缓冲区的句柄并将其存储在名为`command_buffer`的`VkCommandBuffer`类型变量中。确保命令缓冲区处于记录状态，并且开始渲染通道的操作已经记录在其中。

1.  调用`vkCmdEndRenderPass( command_buffer )`，并提供`command_buffer`变量。

# 它是如何工作的...

要结束渲染通道，我们需要调用一个单一函数：

```cpp
vkCmdEndRenderPass( command_buffer );

```

在命令缓冲区中记录此函数执行多个操作。引入执行和内存依赖（如内存屏障中的那些）并执行图像布局转换——图像从为最后一个子通道指定的布局转换到最后布局的值（参考*指定附件描述*食谱）。此外，对于在最后一个子通道中指定了解决的彩色附件，执行多采样解决。另外，对于在渲染通道之后应保留内容的附件，可能将附件数据从缓存传输到图像的内存中。

# 参见

本章中的以下食谱：

+   *指定子通道描述*

+   *创建渲染通道*

+   *开始渲染通道*

+   *推进到下一个子通道*

# 销毁帧缓冲区

当帧缓冲区不再被挂起的命令使用，并且我们不再需要它时，我们可以销毁它。

# 如何操作...

1.  使用创建帧缓冲区的逻辑设备的句柄初始化一个名为`logical_device`的`VkDevice`类型变量。

1.  取出帧缓冲区的句柄并将其存储在名为`framebuffer`的`VkFramebuffer`类型变量中。

1.  执行以下调用：`vkDestroyFramebuffer( logical_device, framebuffer, nullptr )`，其中我们提供`logical_device`和`framebuffer`变量以及一个`nullptr`值。

1.  出于安全原因，在`framebuffer`变量中存储`VK_NULL_HANDLE`值。

# 它是如何工作的...

使用`vkDestroyFramebuffer()`函数调用销毁帧缓冲区。然而，在我们能够销毁它之前，我们必须确保不再在硬件上执行引用给定帧缓冲区的命令。

以下代码销毁了一个帧缓冲区：

```cpp
if( VK_NULL_HANDLE != framebuffer ) { 
  vkDestroyFramebuffer( logical_device, framebuffer, nullptr ); 
  framebuffer = VK_NULL_HANDLE; 
}

```

# 参见

本章中的以下食谱：

+   *创建帧缓冲区*

# 销毁渲染通道

如果渲染通道不再需要，并且它不再被提交到硬件的命令使用，我们可以销毁它。

# 如何操作...

1.  使用创建渲染通道的逻辑设备的句柄来初始化一个名为`logical_device`的`VkDevice`类型变量。

1.  将应销毁的渲染通道的句柄存储在名为`render_pass`的`VkRenderPass`类型变量中。

1.  调用 `vkDestroyRenderPass( logical_device, render_pass, nullptr )` 并提供 `logical_device` 和 `render_pass` 变量以及一个 `nullptr` 值。

1.  由于安全原因，将 `VK_NULL_HANDLE` 值分配给 `render_pass` 变量。

# 它是如何工作的...

删除渲染通道只需一个函数调用，如下所示：

```cpp
if( VK_NULL_HANDLE != render_pass ) { 
  vkDestroyRenderPass( logical_device, render_pass, nullptr ); 
  render_pass = VK_NULL_HANDLE; 
}

```

# 参见

本章中的以下配方：

+   *创建渲染通道*
