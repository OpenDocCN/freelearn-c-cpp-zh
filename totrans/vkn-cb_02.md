# 图像呈现

在本章中，我们将介绍以下食谱：

+   创建启用 WSI 扩展的 Vulkan 实例

+   创建呈现表面

+   选择支持向给定表面呈现的队列家族

+   创建启用 WSI 扩展的逻辑设备

+   选择所需的呈现模式

+   获取呈现表面的功能

+   选择交换链图像的数量

+   选择交换链图像的大小

+   选择交换链图像的期望使用场景

+   选择交换链图像的转换

+   选择交换链图像的格式

+   创建交换链

+   获取交换链图像的句柄

+   创建具有 R8G8B8A8 格式和邮箱呈现模式的交换链

+   获取交换链图像

+   展示图像

+   销毁交换链

+   销毁呈现表面

# 简介

如 Vulkan 这样的 API 可以用于许多不同的目的，例如数学和物理计算、图像或视频流处理以及数据可视化。但 Vulkan 被设计的主要目的及其最常见的用途是高效渲染 2D 和 3D 图形。当我们的应用程序生成图像时，我们通常希望将其显示在屏幕上。

起初，可能会觉得令人惊讶，Vulkan API 的核心不允许在应用程序窗口中显示生成的图像。这是因为 Vulkan 是一个可移植的、跨平台的 API，但不幸的是，由于不同的操作系统具有截然不同的架构和标准，因此在不同的操作系统上显示图像没有通用的标准。

因此，为 Vulkan API 引入了一组扩展，使我们能够在应用程序的窗口中呈现生成的图像。这些扩展通常被称为窗口系统集成（WSI）。Vulkan 可用的每个操作系统都有自己的扩展集，这些扩展将 Vulkan 与特定操作系统的窗口系统集成在一起。

最重要的扩展是允许我们创建交换链的那个。交换链是一组可以呈现给用户的图像。在本章中，我们将为在屏幕上绘制图像做准备——设置图像参数，如格式、大小等。我们还将查看各种可用的 **呈现** 模式，这些模式决定了图像的显示方式，即定义是否启用垂直同步。最后，我们将了解如何展示图像——在应用程序窗口中显示它们。

# 创建启用 WSI 扩展的 Vulkan 实例

为了能够正确地在屏幕上显示图像，我们需要启用一组 WSI 扩展。根据它们引入的功能，它们分为实例级和设备级。第一步是创建一个带有启用扩展的 Vulkan 实例，这些扩展允许我们创建一个呈现表面——这是应用程序窗口的 Vulkan 表示。

# 如何操作...

在 Windows 操作系统家族中执行以下步骤：

1.  准备一个名为`instance`的类型为`VkInstance`的变量。

1.  准备一个名为`desired_extensions`的类型为`std::vector<char const *>`的变量。将您想要启用的所有扩展的名称存储在`desired_extensions`变量中。

1.  将`VK_KHR_SURFACE_EXTENSION_NAME`值添加到`desired_extensions`向量中另一个元素。

1.  将`VK_KHR_WIN32_SURFACE_EXTENSION_NAME`值添加到`desired_extensions`向量中另一个元素。

1.  为`desired_extensions`变量中指定的所有扩展创建一个 Vulkan 实例对象（参考第一章中的*创建 Vulkan 实例*配方，*实例和设备*）。

在具有**X11**窗口系统的 Linux 操作系统家族中通过**XLIB**接口执行以下步骤：

1.  准备一个名为`instance`的类型为`VkInstance`的变量。

1.  准备一个名为`desired_extensions`的类型为`std::vector<char const *>`的变量。将您想要启用的所有扩展的名称存储在`desired_extensions`变量中。

1.  将`VK_KHR_SURFACE_EXTENSION_NAME`值添加到`desired_extensions`向量中另一个元素。

1.  将`VK_KHR_XLIB_SURFACE_EXTENSION_NAME`值添加到`desired_extensions`向量中另一个元素。

1.  为`desired_extensions`变量中指定的所有扩展创建一个 Vulkan 实例对象（参考第一章中的*创建 Vulkan 实例*配方，*实例和设备*）。

在具有 X11 窗口系统的 Linux 操作系统家族中通过**XCB**接口执行以下步骤：

1.  准备一个名为`instance`的类型为`VkInstance`的变量。

1.  准备一个名为`desired_extensions`的类型为`std::vector<char const *>`的变量。将您想要启用的所有扩展的名称存储在`desired_extensions`变量中。

1.  将`VK_KHR_SURFACE_EXTENSION_NAME`值添加到`desired_extensions`向量中另一个元素。

1.  将`VK_KHR_XCB_SURFACE_EXTENSION_NAME`值添加到`desired_extensions`向量中另一个元素。

1.  为`desired_extensions`变量中指定的所有扩展创建一个 Vulkan 实例对象（参考[第一章](https://cdp.packtpub.com/vulkancookbook/wp-admin/post.php?post=29&action=edit#post_42)中的*创建 Vulkan 实例*配方，*实例和设备*）。

# 它是如何工作的...

实例级别的扩展负责管理、创建和销毁一个呈现表面。它是应用程序窗口的（跨平台）表示。通过它，我们可以检查我们是否能够向窗口绘制（显示图像、呈现是队列家族的附加属性），它的参数是什么，或者支持哪些呈现模式（如果我们想启用或禁用垂直同步）。

显示表面直接连接到我们的应用程序窗口，因此只能以特定于给定操作系统的特定方式创建。这就是为什么这种功能是通过扩展引入的，每个操作系统都有自己的扩展来创建显示表面。在 Windows 操作系统家族中，这个扩展称为 `VK_KHR_win32_surface`。在具有 X11 窗口系统的 Linux 操作系统家族中，这个扩展称为 `VK_KHR_xlib_surface`。在具有 XCB 窗口系统的 Linux 操作系统家族中，这个扩展称为 `VK_KHR_xcb_surface`。

销毁显示表面的功能是通过一个名为 `VK_KHR_surface` 的附加扩展启用的。它在所有操作系统上都是可用的。因此，为了正确管理显示表面、检查其参数并验证向其呈现的能力，我们需要在创建 Vulkan 实例时启用两个扩展。

`VK_KHR_win32_surface` 和 `VK_KHR_surface` 扩展引入了在 Windows 操作系统家族中创建和销毁显示表面的能力。

`VK_KHR_xlib_surface` 和 `VK_KHR_surface` 扩展引入了在 Linux 操作系统家族中使用 X11 窗口系统和 XLIB 接口创建和销毁显示表面的能力。

`VK_KHR_xcb_surface` 和 `VK_KHR_surface` 扩展引入了在 Linux 操作系统家族中使用 X11 窗口系统和 XCB 接口创建和销毁显示表面的能力。

为了创建一个支持创建和销毁显示表面过程的 Vulkan 实例，我们需要准备以下代码：

```cpp
desired_extensions.emplace_back( VK_KHR_SURFACE_EXTENSION_NAME ); 
desired_extensions.emplace_back( 
#ifdef VK_USE_PLATFORM_WIN32_KHR 
  VK_KHR_WIN32_SURFACE_EXTENSION_NAME 

#elif defined VK_USE_PLATFORM_XCB_KHR 
  VK_KHR_XCB_SURFACE_EXTENSION_NAME 

#elif defined VK_USE_PLATFORM_XLIB_KHR 
  VK_KHR_XLIB_SURFACE_EXTENSION_NAME 
#endif 
); 

return CreateVulkanInstance( desired_extensions, application_name, instance );

```

在前面的代码中，我们从一个向量变量开始，其中存储了我们想要启用的所有扩展的名称。然后我们将所需的 WSI 扩展添加到向量中。这些扩展的名称通过方便的预处理器定义提供。它们在 `vulkan.h` 文件中定义。有了它们，我们不需要记住扩展的确切名称，如果出错，编译器会告诉我们。

在我们准备好所需扩展的列表后，我们可以像在 第一章 的 *创建 Vulkan 实例* 配方中描述的那样创建 Vulkan 实例对象。

# 参见

+   在 第一章 的 *实例和设备* 部分中，查看以下配方：

    +   *检查可用的实例扩展*

    +   *创建 Vulkan 实例*

+   本章中的以下配方：

    +   *启用 WSI 扩展创建逻辑设备*

# 创建显示表面

显示表面代表应用程序的窗口。它允许我们获取窗口的参数，例如尺寸、支持的颜色格式、所需图像数量或显示模式。它还允许我们检查给定的物理设备是否能够在给定的窗口中显示图像。

因此，在我们想在屏幕上显示图像的情况下，我们首先需要创建一个显示表面，因为它将帮助我们选择一个符合我们需求的物理设备。

# 准备工作

要创建一个显示表面，我们需要提供应用程序窗口的参数。为了做到这一点，窗口必须已经创建。在这个菜谱中，我们将通过一个类型为 `WindowParameters` 的结构来提供其参数。其定义看起来像这样：

```cpp
struct WindowParameters { 
#ifdef VK_USE_PLATFORM_WIN32_KHR 
  HINSTANCE          HInstance; 
  HWND               HWnd; 
#elif defined VK_USE_PLATFORM_XLIB_KHR 
  Display          * Dpy; 
  Window             Window; 
#elif defined VK_USE_PLATFORM_XCB_KHR 
  xcb_connection_t * Connection; 
  xcb_window_t       Window; 
#endif 
};

```

在 Windows 上，该结构包含以下参数：

+   一个名为 `HInstance` 的类型为 `HINSTANCE` 的变量，其中我们存储使用 `GetModuleHandle()` 函数获取的值

+   一个名为 `HWnd` 的类型为 `HWND` 的变量，其中我们存储了 `CreateWindow()` 函数返回的值

在 Linux 的 X11 窗口系统和 XLIB 接口中，该结构包含以下成员：

+   一个名为 `Dpy` 的类型为 `Display*` 的变量，其中存储了 `XOpenDisplay()` 函数调用的值

+   一个名为 `Window` 的类型为 `Window` 的变量，我们将 `XCreateWindow()` 或 `XCreateSimpleWindow()` 函数返回的值赋给它

在 Linux 的 X11 窗口系统和 XCB 接口中，`WindowParameters` 结构包含以下成员：

+   一个名为 `Connection` 的类型为 `xcb_connection_t*` 的变量，其中存储了 `xcb_connect()` 函数返回的值

+   一个名为 `Window` 的类型为 `xcb_window_t` 的变量，其中存储了 `xcb_generate_id()` 函数返回的值

# 如何做到这一点...

在 Windows 操作系统家族中，执行以下步骤：

1.  取一个名为 `instance` 的类型为 `VkInstance` 的变量，其中存储了一个创建的 Vulkan 实例的句柄。

1.  创建一个名为 `window_parameters` 的类型为 `WindowParameters` 的变量。为其成员分配以下值：

    +   `CreateWindow()` 函数为 `HWnd` 返回的值

    +   `GetModuleHandle(nullptr)` 函数返回的值用于 `HInstance`

1.  创建一个名为 `surface_create_info` 的类型为 `VkWin32SurfaceCreateInfoKHR` 的变量，并使用以下值初始化其成员：

    +   `VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR` 的值用于 `sType`

    +   `nullptr` 的值用于 `pNext`

    +   `0` 的值用于 `flags`

    +   `window_parameters.HInstance` 成员用于 `hinstance`

    +   `window_parameters.HWnd` 成员用于 `hwnd`

1.  创建一个名为 `presentation_surface` 的类型为 `VkSurfaceKHR` 的变量，并将其赋值为 `VK_NULL_HANDLE`。

1.  调用 `vkCreateWin32SurfaceKHR(instance, &surface_create_info, nullptr, &presentation_surface)`。在第一个参数中提供一个创建的实例的句柄，在第二个参数中提供一个指向 `surface_create_info` 变量的指针，在第三个参数中使用 `nullptr` 值，在最后一个参数中提供一个指向 `presentation_surface` 变量的指针。

1.  确保调用 `vkCreateWin32SurfaceKHR()` 函数成功，通过检查其返回值是否等于 `VK_SUCCESS` 以及 `presentation_surface` 变量的值是否不等于 `VK_NULL_HANDLE`。

在具有 X11 窗口系统和 XLIB 接口的 Linux 操作系统家族上，执行以下步骤：

1.  取名为 `instance` 的类型为 `VkInstance` 的变量，其中存储了一个创建的 Vulkan 实例的句柄。

1.  创建一个名为 `window_parameters` 的类型为 `WindowParameters` 的变量。为其成员分配以下值：

    +   `XOpenDisplay()` 函数为 `Dpy` 返回的值

    +   `XCreateSimpleWindow()` 或 `XCreateWindow()` 函数为 `Window` 返回的值

1.  创建一个名为 `surface_create_info` 的类型为 `VkXlibSurfaceCreateInfoKHR` 的变量，并使用以下值初始化其成员：

    +   `VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR` 的值用于 `sType`

    +   `pNext` 的值为 `nullptr`

    +   `flags` 的值为 `0`

    +   `dpy` 的 `window_parameters.Dpy` 成员

    +   `window` 的 `window_parameters.Window` 成员

1.  创建一个名为 `presentation_surface` 的类型为 `VkSurfaceKHR` 的变量，并将其值设置为 `VK_NULL_HANDLE`。

1.  调用 `vkCreateXlibSurfaceKHR(instance, &surface_create_info, nullptr, &presentation_surface)`。在第一个参数中提供一个创建的实例的句柄，在第二个参数中提供一个指向 `surface_create_info` 变量的指针，在第三个参数中使用 `nullptr` 值，在最后一个参数中提供一个指向 `presentation_surface` 变量的指针。

1.  通过检查 `vkCreateXlibSurfaceKHR()` 函数返回的值是否等于 `VK_SUCCESS` 以及 `presentation_surface` 变量的值是否不等于 `VK_NULL_HANDLE`，确保 `vkCreateXlibSurfaceKHR()` 函数调用成功。

在具有 X11 窗口系统和 XCB 接口的 Linux 操作系统家族上，执行以下步骤：

1.  取名为 `instance` 的类型为 `VkInstance` 的变量，其中存储了一个创建的 Vulkan 实例的句柄。

1.  创建一个名为 `window_parameters` 的类型为 `WindowParameters` 的变量。为其成员分配以下值：

    +   `xcb_connect()` 函数为 `Connection` 返回的值

    +   `xcb_generate_id()` 函数为 `Window` 返回的值

1.  创建一个名为 `surface_create_info` 的类型为 `VkXcbSurfaceCreateInfoKHR` 的变量，并使用以下值初始化其成员：

    +   `sType` 的值为 `VK_STRUCTURE_TYPE_XCB_SURFACE_CREATE_INFO_KHR`

    +   `pNext` 的值为 `nullptr`

    +   `flags` 的值为 `0`

    +   `connection` 的 `window_parameters.Connection` 成员

    +   `window` 的 `window_parameters.Window` 成员

1.  创建一个名为 `presentation_surface` 的类型为 `VkSurfaceKHR` 的变量，并将其值设置为 `VK_NULL_HANDLE`。

1.  调用 `vkCreateXcbSurfaceKHR(instance, &surface_create_info, nullptr, &presentation_surface)`。在第一个参数中提供一个创建的实例的句柄，在第二个参数中提供一个指向 `surface_create_info` 变量的指针，在第三个参数中使用 `nullptr` 值，在最后一个参数中提供一个指向 `presentation_surface` 变量的指针。

1.  通过检查`vkCreateXcbSurfaceKHR()`函数返回的值是否等于`VK_SUCCESS`以及`presentation_surface`变量的值是否不等于`VK_NULL_HANDLE`，确保该函数调用成功。

# 它是如何工作的...

展示表面的创建在很大程度上依赖于特定于给定操作系统的参数。在每种操作系统上，我们需要创建不同类型的变量并调用不同的函数。以下是在 Windows 上创建展示表面的代码：

```cpp
#ifdef VK_USE_PLATFORM_WIN32_KHR 

VkWin32SurfaceCreateInfoKHR surface_create_info = { 
  VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR, 
  nullptr, 
  0, 
  window_parameters.HInstance, 
  window_parameters.HWnd 
}; 

VkResult result = vkCreateWin32SurfaceKHR( instance, &surface_create_info, nullptr, &presentation_surface );

```

这里是 Linux 操作系统上使用 X11 窗口系统时执行相同操作的代码片段：

```cpp
#elif defined VK_USE_PLATFORM_XLIB_KHR 

VkXlibSurfaceCreateInfoKHR surface_create_info = { 
  VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR, 
  nullptr, 
  0, 
  window_parameters.Dpy, 
  window_parameters.Window 
}; 

VkResult result = vkCreateXlibSurfaceKHR( instance, &surface_create_info, nullptr, &presentation_surface );

```

最后，这是 Linux 上的 XCB 窗口系统的部分，同样是在 Linux 上：

```cpp
#elif defined VK_USE_PLATFORM_XCB_KHR 

VkXcbSurfaceCreateInfoKHR surface_create_info = { 
  VK_STRUCTURE_TYPE_XCB_SURFACE_CREATE_INFO_KHR, 
  nullptr, 
  0, 
  window_parameters.Connection, 
  window_parameters.Window 
}; 

VkResult result = vkCreateXcbSurfaceKHR( instance, &surface_create_info, nullptr, &presentation_surface ); 

#endif

```

上述代码示例非常相似。在每个示例中，我们创建了一个结构类型变量，并用创建的窗口的参数初始化其成员。接下来，我们调用一个`vkCreate???SurfaceKHR()`函数来创建展示表面，并将其句柄存储在`presentation_surface`变量中。之后，我们应该检查一切是否按预期工作：

```cpp
if( (VK_SUCCESS != result) || 
    (VK_NULL_HANDLE == presentation_surface) ) { 
  std::cout << "Could not create presentation surface." << std::endl; 
  return false; 
} 
return true;

```

# 参见

本章中的以下食谱：

+   *获取展示表面的能力*

+   *创建交换链*

+   *销毁展示表面*

# 选择支持向给定表面展示的队列家族

在屏幕上显示图像是通过向设备的队列提交一个特殊命令来完成的。我们不能使用任何我们想要的队列来显示图像，换句话说，我们不能将此操作提交给任何队列。这是因为可能不支持。图像展示，连同图形、计算、传输和稀疏操作，是队列家族的另一个属性。并且与所有类型的操作类似，并非所有队列都支持它，更重要的是，甚至并非所有设备都支持它。这就是为什么我们需要检查哪个物理设备所属的队列家族允许我们在屏幕上展示图像。

# 如何做到这一点...

1.  获取由`vkEnumeratePhysicalDevices()`函数返回的物理设备的句柄。将其存储在名为`physical_device`的`VkPhysicalDevice`类型变量中。

1.  将创建的展示表面及其句柄存储在名为`presentation_surface`的`VkSurfaceKHR`类型变量中。

1.  创建一个包含`VkQueueFamilyProperties`类型元素的`std::vector`，并将其命名为`queue_families`。

1.  枚举由`physical_device`变量表示的物理设备上可用的所有队列家族（参考第一章中的*检查可用的队列家族及其属性*食谱）。将此操作的结果存储在`queue_families`变量中。

1.  创建一个名为`queue_family_index`的`uint32_t`类型变量。

1.  创建一个名为`index`的`uint32_t`类型变量。使用它来遍历`queue_families`向量的所有元素。对于`queue_families`变量的每个元素，执行以下步骤：

    1.  创建一个名为`presentation_supported`的`VkBool32`类型的变量。将值`VK_FALSE`分配给这个变量。

    1.  调用`vkGetPhysicalDeviceSurfaceSupportKHR(physical_device, index, presentation_surface, &presentation_supported)`。在第一个参数中提供物理设备的句柄，在第二个参数中提供当前循环迭代的数字，在第三个参数中提供呈现表面的句柄。同时，在最后一个参数中提供`presentation_supported`变量的指针。

    1.  检查`vkGetPhysicalDeviceSurfaceSupportKHR()`函数返回的值是否等于`VK_SUCCESS`，以及`presentation_supported`变量的值是否等于`VK_TRUE`。如果是，将当前循环迭代的值（`index`变量）存储在`queue_family_index`变量中，并结束循环。

# 它是如何工作的...

首先，我们需要检查给定物理设备暴露了哪些队列家族。这个操作与第一章的*实例和设备*部分中描述的*检查可用的队列家族及其属性*食谱中的方式相同：

```cpp
std::vector<VkQueueFamilyProperties> queue_families; 
if( !CheckAvailableQueueFamiliesAndTheirProperties( physical_device, queue_families ) ) { 
  return false; 
}

```

接下来，我们可以遍历所有可用的队列家族，并检查给定的家族是否支持图像呈现。这是通过调用`vkGetPhysicalDeviceSurfaceSupportKHR()`函数来完成的，该函数将信息存储在指定的变量中。如果支持图像呈现，我们可以记住给定家族的索引。从这个家族的所有队列都将支持图像呈现：

```cpp
for( uint32_t index = 0; index < static_cast<uint32_t>(queue_families.size()); ++index ) { 
  VkBool32 presentation_supported = VK_FALSE; 
  VkResult result = vkGetPhysicalDeviceSurfaceSupportKHR( physical_device, index, presentation_surface, &presentation_supported ); 
  if( (VK_SUCCESS == result) && 
      (VK_TRUE == presentation_supported) ) { 
    queue_family_index = index; 
    return true; 
  } 
} 
return false;

```

当没有队列家族由给定的物理设备导出以支持图像呈现时，我们必须检查此操作是否可在另一个物理设备上执行。

# 参考内容

在第一章的*实例和设备*部分，查看以下食谱：

+   *检查可用的队列家族及其属性*

+   *选择具有所需功能的队列家族的索引*

+   *创建逻辑设备*

# 创建启用了 WSI 扩展的逻辑设备

当我们创建了一个启用了 WSI 扩展的实例，并且找到了一个支持图像呈现的队列家族时，就是时候创建一个启用了另一个扩展的逻辑设备了。设备级别的 WSI 扩展允许我们创建一个 swapchain。这是一个由呈现引擎管理的图像集合。为了使用这些图像并将它们渲染到其中，我们需要获取它们。完成之后，我们将它们归还给呈现引擎。这个操作被称为呈现，它通知驱动程序我们想要向用户展示一个图像（在屏幕上呈现或显示它）。呈现引擎根据在 swapchain 创建期间定义的参数来显示它。我们只能在启用了 swapchain 扩展的逻辑设备上创建它。

# 如何操作...

1.  获取一个物理设备的句柄，该设备有一个支持图像呈现的队列家族，并将其存储在一个名为 `VkPhysicalDevice` 的变量中。

1.  准备一个队列家族列表和每个家族的队列数量。为每个家族中的每个队列分配一个优先级（一个介于 `0.0f` 和 `1.0f` 之间的浮点值）。将这些参数存储在一个名为 `queue_infos` 的 `std::vector` 变量中，其元素为自定义类型 `QueueInfo`（参考第一章中的 *创建逻辑设备* 菜单，*实例和设备*）。请记住，至少包含一个支持图像呈现的家族的队列。

1.  准备一个应启用扩展的列表。将其存储在一个名为 `desired_extensions` 的 `std::vector<char const *>` 类型的变量中。

1.  将另一个元素添加到 `desired_extensions` 变量中，其值等于 `VK_KHR_SWAPCHAIN_EXTENSION_NAME`。

1.  使用在 `physical_device` 和 `queue_infos` 变量中准备的参数，以及从 `desired_extensions` 向量中启用的所有扩展来创建一个逻辑设备（参考第一章中的 *创建逻辑设备* 菜单，*实例和设备*）。

# 它是如何工作的...

当我们想在屏幕上显示图像时，在创建逻辑设备期间需要启用一个设备级扩展。这被称为 `VK_KHR_swapchain`，它允许我们创建交换链。

交换链定义了与 OpenGL API 中默认绘制缓冲区参数非常相似的参数。它指定了我们要渲染到的图像的格式、图像的数量（可以认为是双缓冲或三缓冲），或呈现模式（启用或禁用 v-sync）。在交换链中创建的图像由呈现引擎拥有和管理。我们不允许自己创建或销毁它们。我们甚至不能使用它们，直到我们请求这样做。当我们想在屏幕上显示图像时，我们需要请求交换链中的一个图像（获取它），将其渲染进去，然后将图像交还给呈现引擎（呈现它）。

指定一组可显示的图像、获取它们以及在屏幕上显示它们的功能由 `VK_KHR_swapchain` 扩展定义。

描述的功能由 `VK_KHR_swapchain` 扩展定义。要在创建逻辑设备时启用它，我们需要准备以下代码：

```cpp
desired_extensions.emplace_back( VK_KHR_SWAPCHAIN_EXTENSION_NAME ); 

return CreateLogicalDevice( physical_device, queue_infos, desired_extensions, desired_features, logical_device );

```

逻辑设备创建的代码与第一章中描述的 *创建逻辑设备* 菜单中的操作相同，*实例和设备*。在这里，我们只需要记住，我们必须检查给定的物理设备是否支持 `VK_KHR_swapchain` 扩展，然后我们需要将其包含在应启用的扩展列表中。

扩展的名称通过`VK_KHR_SWAPCHAIN_EXTENSION_NAME`预处理器定义指定。它在`vulkan.h`头文件中定义，帮助我们避免在扩展名称中出错。

# 参见

+   在第一章中的以下配方，*实例和设备：*

    +   *检查可用的设备扩展*

    +   *创建逻辑设备*

+   在本章中启用 WSI 扩展的*创建 Vulkan 实例*配方

# 选择期望的显示模式

在屏幕上显示图像是 Vulkan 交换链最重要的功能之一——实际上，这也是交换链被设计的目的。在 OpenGL 中，当我们完成对后缓冲区的渲染后，我们只需将其与前缓冲区交换，渲染的图像就会显示在屏幕上。我们只能确定是否要显示图像以及是否要使用空白间隔（如果我们想启用 v-sync）。

在 Vulkan 中，我们不仅限于只能渲染一个图像（后缓冲区）。而且，我们可以在图像在屏幕上显示的方式中选择一种或多种方式，而不是两种（启用或禁用 v-sync）。这被称为显示模式，我们需要在创建交换链时指定它。

# 如何实现...

1.  使用`vkEnumeratePhysicalDevices()`函数枚举物理设备。将其存储在名为`physical_device`的`VkPhysicalDevice`类型的变量中。

1.  将创建的显示表面及其句柄存储在名为`presentation_surface`的`VkSurfaceKHR`类型的变量中。

1.  创建一个名为`desired_present_mode`的`VkPresentModeKHR`类型的变量。将期望的显示模式存储在这个变量中。

1.  准备一个名为`present_modes_count`的`uint32_t`类型的变量。

1.  调用`vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, presentation_surface, &present_modes_count, nullptr)`。将物理设备的句柄和显示表面的句柄作为前两个参数提供。在第三个参数中，提供一个指向`present_modes_count`变量的指针。

1.  如果函数调用成功，则`present_modes_count`变量将包含支持的显示模式数量。

1.  创建一个名为`present_modes`的`std::vector<VkPresentModeKHR>`类型的变量。将向量的大小调整为至少包含`present_modes_count`个元素。

1.  再次调用`vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, presentation_surface, &present_modes_count, &present_modes[0])`，但这次，在最后一个参数中，提供一个指向`present_modes`向量第一个元素的指针。

1.  如果函数返回`VK_SUCCESS`值，则`present_modes`变量将包含在给定平台上支持的显示模式。

1.  遍历`present_modes`向量的所有元素。检查这些元素中是否有与存储在`desired_present_mode`变量中的期望显示模式相等的元素。

1.  如果所需的展示模式不受支持（`present_modes`向量的任何元素都不等于`desired_present_mode`变量），则选择 FIFO 展示模式--`VK_PRESENT_MODE_FIFO_KHR`的值--这始终应该被支持。

# 它是如何工作的...

展示模式定义了图像在屏幕上显示的方式。目前，在 Vulkan API 中定义了四种模式。

最简单的是**立即**模式。在这里，当展示一个图像时，它立即替换正在显示的图像。没有等待，没有队列，也没有从应用程序角度应该考虑的其他参数。正因为如此，可能会观察到屏幕撕裂：

![](img/image_02_001.png)

必须支持的展示模式，即每个 Vulkan API 实现都必须支持的，是**FIFO 模式**。在这里，当展示一个图像时，它被添加到先进先出队列（这个队列的长度等于 swapchain 中的图像数量减一，*n - 1*）。从这个队列中，图像在同步消隐周期（v-sync）的情况下显示在屏幕上，始终按照它们被添加到队列中的相同顺序。在这个模式下没有撕裂，因为 v-sync 已启用。这种模式类似于 OpenGL 的缓冲区交换，其中交换间隔设置为 1。

FIFO 展示模式必须始终支持。

还有一种 FIFO 模式的轻微修改，称为**FIFO RELAXED**。这两种模式之间的区别在于，在**RELAXED**模式下，只有当图像展示得足够快，快于刷新率时，图像才会与消隐周期同步显示在屏幕上。如果一个图像被应用程序展示，并且从上次展示以来经过的时间大于两个消隐周期之间的刷新时间（FIFO 队列是空的），则图像将立即展示。所以如果我们足够快，就没有屏幕撕裂，但如果我们绘制速度慢于显示器的刷新率，屏幕撕裂将可见。这种行为类似于 OpenGL 的`EXT_swap_control_tear`扩展中指定的：

![](img/image_02_002.png)

最后一种展示模式被称为**邮箱**模式。它可以被视为三缓冲。在这里，也涉及到一个队列，但它只包含一个元素。在这个队列中等待的图像将与消隐周期（v-sync 已启用）同步显示。但是，当应用程序展示一个图像时，新的图像将替换队列中等待的图像。因此，展示引擎总是显示最新的、最新的可用图像。而且没有屏幕撕裂：

![](img/image_02_003.png)

要选择所需的展示模式，我们需要检查当前平台上有哪些模式可用。首先，我们需要获取所有支持展示模式的总数。这是通过调用一个`vkGetPhysicalDeviceSurfacePresentModesKHR()`函数来完成的，其中最后一个参数设置为`nullptr`：

```cpp
uint32_t present_modes_count = 0; 
VkResult result = VK_SUCCESS; 

result = vkGetPhysicalDeviceSurfacePresentModesKHR( physical_device, presentation_surface, &present_modes_count, nullptr ); 
if( (VK_SUCCESS != result) || 
    (0 == present_modes_count) ) { 
  std::cout << "Could not get the number of supported present modes." << std::endl; 
  return false; 
}

```

接下来我们可以为所有支持的模式准备存储，然后再次调用相同的函数，但这次将最后一个参数指向分配的存储：

```cpp
std::vector<VkPresentModeKHR> present_modes( present_modes_count ); 
result = vkGetPhysicalDeviceSurfacePresentModesKHR( physical_device, presentation_surface, &present_modes_count, &present_modes[0] ); 
if( (VK_SUCCESS != result) || 
  (0 == present_modes_count) ) { 
  std::cout << "Could not enumerate present modes." << std::endl; 
  return false; 
}

```

现在我们知道了可用的展示模式，我们可以检查所选模式是否可用。如果不可用，我们可以从获取的列表中选择另一个展示模式，或者我们退回到强制性和始终可用的默认 FIFO 模式：

```cpp
for( auto & current_present_mode : present_modes ) { 
  if( current_present_mode == desired_present_mode ) { 
    present_mode = desired_present_mode; 
    return true; 
  } 
} 

std::cout << "Desired present mode is not supported. Selecting default FIFO mode." << std::endl; 
for( auto & current_present_mode : present_modes ) { 
  if( current_present_mode == VK_PRESENT_MODE_FIFO_KHR ) { 
    present_mode = VK_PRESENT_MODE_FIFO_KHR; 
    return true; 
  } 
}

```

# 参见

本章中的以下食谱：

+   *选择交换链图像的数量*

+   *创建交换链*

+   *使用 R8G8B8A8 格式和邮箱展示模式创建交换链*

+   *获取交换链图像*

+   *展示图像*

# 获取展示表面的能力

当我们创建交换链时，我们需要指定创建参数。但我们不能选择我们想要的任何值。我们必须提供适合支持限制的值，这些值可以从展示表面获得。因此，为了正确创建交换链，我们需要获取表面的能力。

# 如何做到...

1.  使用 `vkEnumeratePhysicalDevices()` 函数枚举的所选物理设备的句柄，并将其存储在名为 `physical_device` 的类型为 `VkPhysicalDevice` 的变量中。

1.  获取创建的展示表面的句柄。将其存储在名为 `presentation_surface` 的类型为 `VkSurfaceKHR` 的变量中。

1.  创建一个名为 `surface_capabilities` 的类型为 `VkSurfaceCapabilitiesKHR` 的变量。

1.  调用 `vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, presentation_surface, &surface_capabilities)`，其中提供物理设备的句柄和一个展示表面，以及指向 `surface_capabilities` 变量的指针。

1.  如果函数调用成功，`surface_capabilities` 变量将包含用于创建交换链的展示表面的参数、限制和能力。

# 它是如何工作的...

在创建交换链时获取支持的特性和参数范围非常直接：

```cpp
VkResult result = vkGetPhysicalDeviceSurfaceCapabilitiesKHR( physical_device, presentation_surface, &surface_capabilities ); 

if( VK_SUCCESS != result ) { 
  std::cout << "Could not get the capabilities of a presentation surface." << std::endl; 
  return false; 
} 
return true;

```

我们只需调用一个 `vkGetPhysicalDeviceSurfaceCapabilitiesKHR()` 函数，它将参数存储在类型为 `VkSurfaceCapabilitiesKHR` 的变量中。这是一个结构，其中包含定义以下参数的成员：

+   允许的最小和最大交换链图像数量

+   展示表面的最小、最大和当前范围

+   支持的图像转换（可以在展示之前应用）和当前正在使用的转换

+   支持的最大图像层数量

+   支持的使用方式

+   支持的表面 alpha 值组合列表（图像的 alpha 成分应该如何影响应用程序的窗口桌面合成）

# 参见

本章中的以下食谱：

+   *创建展示表面*

+   *选择交换链图像的数量*

+   *选择交换链图像的大小*

+   *选择交换链图像的期望使用场景*

+   *选择交换链图像的转换*

+   *选择 swapchain 图像的格式*

+   *创建 swapchain*

# 选择 swapchain 图像的数量

当应用程序想要渲染到 swapchain 图像时，它必须从展示引擎中获取它。应用程序可以获取更多图像；我们不仅仅限制于一次获取一张图像。但是，可用图像的数量（在给定时间未被展示引擎使用）取决于指定的展示模式、应用程序当前的状态（渲染/展示历史）以及图像的数量——当我们创建 swapchain 时，我们必须指定应该创建的（最小）图像数量。

# 如何实现...

1.  获取展示表面的能力（参考 *获取展示表面能力* 脚本）。将它们存储在名为 `surface_capabilities` 的 `VkSurfaceCapabilitiesKHR` 类型变量中。

1.  创建一个名为 `number_of_images` 的 `uint32_t` 类型变量。

1.  将 `surface_capabilities.minImageCount + 1` 的值分配给 `number_of_images` 变量。

1.  检查 `surface_capabilities` 变量的 `maxImageCount` 成员是否大于零。如果是，这意味着创建图像的最大允许数量有限制。在这种情况下，检查 `number_of_images` 变量的值是否大于 `surface_capabilities.maxImageCount` 的值。如果是，将 `number_of_images` 变量的值限制在 `surface_capabilities` 变量的 `maxImageCount` 成员中定义的限制内。

# 它是如何工作的...

与 swapchain 一起创建（自动）的图像主要用于展示目的。但它们也允许展示引擎正常工作。屏幕上始终显示一张图像。应用程序必须用另一张图像替换它后才能使用它。展示的图像会立即替换显示的图像，或者根据所选模式等待队列中的适当时刻（v-sync）来替换它。被展示并正在被替换的图像变为未使用状态，可以被应用程序获取。

应用程序只能获取当前处于未使用状态（参考 *选择所需展示模式* 脚本）的图像。我们可以获取所有这些图像。但是，一旦所有未使用的图像都被获取，我们就需要展示其中至少一张，以便能够获取另一张。如果我们不这样做，获取操作可能会无限期地阻塞。

未使用图像的数量主要取决于展示模式和与 swapchain 一起创建的总图像数量。因此，我们想要创建的图像数量应根据我们想要实现的渲染场景（应用程序希望同时拥有的图像数量）和所选的展示模式来选择。

选择最小数量的图像可能如下所示：

```cpp
number_of_images = surface_capabilities.minImageCount + 1; 
if( (surface_capabilities.maxImageCount > 0) && 
    (number_of_images > surface_capabilities.maxImageCount) ) { 
  number_of_images = surface_capabilities.maxImageCount; 
} 
return true;

```

通常，在最常见的渲染场景中，我们将在给定时间内渲染到单个图像。因此，最小支持的图像数量可能就足够了。创建更多图像允许我们同时获取更多图像，但更重要的是，如果实现了适当的渲染算法，这也可能提高我们应用程序的性能。但我们不能忘记图像会消耗相当大的内存。因此，我们为 swapchain 选择的图像数量应该在我们的需求、内存使用和应用程序性能之间取得平衡。

在前面的示例中，展示了这样一个折衷方案，即应用程序选择比允许呈现引擎正常工作的最小值多一个图像。之后，我们还需要检查是否存在上限，以及我们是否超过了它。如果我们超过了，我们需要将选定的值限制在支持的范围内。

# 参见

本章中的以下配方：

+   *选择所需的呈现模式*

+   *获取呈现表面的能力*

+   *创建 swapchain*

+   *获取 swapchain 图像*

+   *呈现图像*

# 选择 swapchain 图像的大小

通常，为 swapchain 创建的图像应该适合应用程序的窗口。支持的尺寸在呈现表面的能力中可用。但在某些操作系统上，图像的大小定义了窗口的最终大小。我们也应该记住这一点，并检查 swapchain 图像的适当尺寸。

# 如何做到这一点...

1.  获取呈现表面的能力（参考*获取呈现表面能力*配方）。将它们存储在名为`surface_capabilities`的`VkSurfaceCapabilitiesKHR`类型的变量中。

1.  创建一个名为`size_of_images`的`VkExtent2D`类型的变量，我们将存储所需的 swapchain 图像的大小。

1.  检查`surface_capabilities`变量的`currentExtent.width`成员是否等于`0xFFFFFFFF`（`-1`转换为`uint32_t`类型的无符号值）。如果是，这意味着图像的大小决定了窗口的大小。在这种情况下：

    +   为`size_of_images`变量的`width`和`height`成员分配所需的值

    +   将`size_of_images`变量的`width`成员的值限制在由`surface_capabilities.minImageExtent.width`和`surface_capabilities.maxImageExtent.width`定义的范围内

    +   将`size_of_images`变量的`height`成员的值限制在由`surface_capabilities.minImageExtent.height`和`surface_capabilities.maxImageExtent.height`定义的范围内

1.  如果`surface_capabilities`变量的`currentExtent.width`成员的值不等于`0xFFFFFFFF`，在`size_of_images`变量中存储`surface_capabilities.currentExtent`的值。

# 它是如何工作的...

交换链图像的大小必须符合支持的极限。这些极限由表面能力定义。在大多数典型场景中，我们希望渲染到与应用程序窗口客户端区域相同尺寸的图像。这个值在表面的能力成员`currentExtent`中指定。

但有一些操作系统，窗口的大小由交换链图像的大小决定。这种情况通过表面能力的`currentExtent.width`或`currentExtent.height`成员的`0xFFFFFFFF`值来表示。在这种情况下，我们可以定义图像的大小，但它仍然必须在一个指定的范围内：

```cpp
if( 0xFFFFFFFF == surface_capabilities.currentExtent.width ) { 
  size_of_images = { 640, 480 }; 

  if( size_of_images.width < surface_capabilities.minImageExtent.width ) { 
    size_of_images.width = surface_capabilities.minImageExtent.width; 
  } else if( size_of_images.width > surface_capabilities.maxImageExtent.width ) { 
    size_of_images.width = surface_capabilities.maxImageExtent.width; 
  } 

  if( size_of_images.height < surface_capabilities.minImageExtent.height ) { 
    size_of_images.height = surface_capabilities.minImageExtent.height; 
  } else if( size_of_images.height > surface_capabilities.maxImageExtent.height ) { 
    size_of_images.height = surface_capabilities.maxImageExtent.height; 
  } 
} else { 
  size_of_images = surface_capabilities.currentExtent; 
} 
return true;

```

# 参见

本章中的以下食谱：

+   *获取呈现表面的能力*

+   *创建交换链*

# 选择交换链图像的期望使用场景

使用交换链创建的图像通常用作颜色附件。这意味着我们希望将渲染到它们中（将它们用作渲染目标）。但我们不仅限于这种场景。我们可以将交换链图像用于其他目的--我们可以从它们中采样，将它们用作复制操作中的数据源，或者将数据复制到它们中。这些都是不同的图像使用方式，我们可以在创建交换链期间指定它们。但是，再次强调，我们需要检查这些使用是否受支持。

# 如何做到这一点...

1.  获取呈现表面的能力（参考*获取呈现表面的能力*食谱）。将它们存储在一个名为`surface_capabilities`的`VkSurfaceCapabilitiesKHR`类型变量中。

1.  选择所需的图像使用方式，并将它们存储在一个名为`desired_usages`的位字段类型`VkImageUsageFlags`的变量中。

1.  在一个名为`image_usage`的`VkImageUsageFlags`类型变量中创建一个变量，其中将存储在给定平台上受支持的请求使用列表。将`image_usage`变量的值设置为`0`。

1.  遍历`desired_usages`位字段变量的所有位。对于变量中的每个位：

    +   检查位是否设置（等于一）

    +   检查`surface_capabilities`变量的`supportedUsageFlags`成员中的对应位是否设置

    +   如果前面的检查是正确的，则在`image_usage`变量中设置相同的位。

1.  通过检查`desired_usages`和`image_usage`变量的值是否相等，确保在给定平台上所有请求的使用都是受支持的。

# 它是如何工作的...

可以为交换链图像选择的用法列表在呈现表面能力的`supportedUsageFlags`成员中可用。该成员是一个位字段，其中每个位对应于特定的用法。如果给定位被设置，则表示给定的用法受支持。

颜色附件使用（`VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT`）必须始终受支持。

`VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT`用法是强制性的，所有 Vulkan 实现都必须支持它。其他用法是可选的。这就是为什么我们不应该依赖于它们的可用性。同样，我们也不应该请求我们不需要的用法，因为这可能会影响我们应用程序的性能。

选择所需的用法可能看起来像这样：

```cpp
image_usage = desired_usages & surface_capabilities.supportedUsageFlags; 

return desired_usages == image_usage;

```

我们只取所需用法和支持用法的公共部分。然后检查是否所有请求的用法都受支持。我们通过比较请求的用法和“最终”用法的值来完成此操作。如果它们的值不同，我们知道并非所有所需的用法都受支持。

# 参考以下内容

本章中的以下食谱：

+   *获取呈现表面的能力*

+   *创建 swapchain*

# 选择 swapchain 图像的转换

在某些（特别是移动）设备上，图像可以从不同的方向查看。有时我们可能希望能够指定图像在屏幕上显示时应如何定位。在 Vulkan 中，我们有这样的可能性。在创建 swapchain 时，我们需要指定在呈现之前应用于图像的转换。

# 如何操作...

1.  获取呈现表面的能力（参考*获取呈现表面的能力*食谱）。将它们存储在名为`surface_capabilities`的`VkSurfaceCapabilitiesKHR`类型的变量中。

1.  将所需的转换存储在名为`desired_transform`的`VkSurfaceTransformFlagBitsKHR`类型的位字段变量中。

1.  创建一个名为`surface_transform`的`VkSurfaceTransformFlagBitsKHR`类型的变量，我们将存储受支持的转换。

1.  检查在`desired_transform`变量中设置的位是否也在呈现表面能力的`supportedTransforms`成员中设置。如果是，将`desired_transform`变量的值分配给`surface_transform`变量。

1.  如果不是所有所需的转换都受支持，则通过将`surface_capabilities.currentTransform`的值分配给`surface_transform`变量来回退到当前转换。

# 它是如何工作的...

呈现表面能力的`supportedTransforms`成员定义了在给定平台上可用的所有图像转换列表。转换定义了在屏幕上显示之前图像应该如何旋转或镜像。在创建 swapchain 期间，我们可以指定所需的转换，并且呈现引擎将其应用于图像作为显示过程的一部分。

我们可以选择任何受支持的值。以下是一个代码示例，如果可用则选择所需的转换，否则仅使用当前使用的转换：

```cpp
if( surface_capabilities.supportedTransforms & desired_transform ) { 
  surface_transform = desired_transform; 
} else { 
  surface_transform = surface_capabilities.currentTransform; 
}

```

# 参考以下内容

本章中的以下食谱：

+   *获取呈现表面的能力*

+   *创建 swapchain*

# 选择 swapchain 图像的格式

格式定义了颜色组件的数量、每个组件的位数以及使用的数据类型。在创建交换链时，我们必须指定是否要使用带有或不带有 alpha 组件的红、绿、蓝通道，颜色值是否应该使用无符号整数或浮点数据类型进行编码，以及它们的精度是多少。我们还必须选择是否使用线性或非线性颜色空间进行颜色值的编码。但与其他交换链参数一样，我们只能使用由显示表面支持的值。

# 准备工作

在这个菜谱中，我们使用了一些可能看起来相同的术语，但实际上它们指定了不同的参数：

+   图像格式用于描述图像像素的组件数量、精度和数据类型。它对应于 `VkFormat` 类型的变量。

+   颜色空间决定了硬件解释颜色组件值的方式，是使用线性或非线性函数进行编码或解码。颜色空间对应于 `VkColorSpaceKHR` 类型的变量。

+   表面格式是图像格式和颜色空间的一对，由 `VkSurfaceFormatKHR` 类型的变量表示。

# 如何做到这一点...

1.  取 `vkEnumeratePhysicalDevices()` 函数返回的物理设备的句柄。将其存储在名为 `physical_device` 的 `VkPhysicalDevice` 类型变量中。

1.  将创建的显示表面存储在其名为 `presentation_surface` 的 `VkSurfaceKHR` 类型变量中。

1.  选择所需的图像格式和颜色空间，并将它们分配给名为 `desired_surface_format` 的 `VkSurfaceFormatKHR` 类型变量的成员。

1.  创建一个名为 `formats_count` 的 `uint32_t` 类型变量。

1.  调用 `vkGetPhysicalDeviceSurfaceFormatsKHR( physical_device, presentation_surface, &formats_count, nullptr )`，在第一个参数中提供物理设备的句柄，在第二个参数中提供显示表面的句柄，在第三个参数中提供一个指向 `formats_count` 变量的指针。将最后一个参数的值设置为 `nullptr`。

1.  如果函数调用成功，`formats_count` 变量将包含所有支持的格式-颜色空间对的数量。

1.  创建一个名为 `surface_formats` 的 `std::vector<VkSurfaceFormatKHR>` 类型的变量。调整向量大小，使其至少能容纳 `formats_count` 个元素。

1.  执行以下调用，`vkGetPhysicalDeviceSurfaceFormatsKHR( physical_device, presentation_surface, &formats_count, &surface_formats[0] )`。为前三个参数提供相同的参数。在最后一个参数中，提供一个指向 `surface_formats` 向量第一个元素的指针。

1.  如果调用成功，所有可用的图像格式-颜色空间对都将存储在 `surface_formats` 变量中。

1.  创建一个名为`image_format`的`VkFormat`类型变量和一个名为`image_color_space`的`VkColorSpaceKHR`类型变量，我们将存储在创建 swapchain 时使用的格式和颜色空间的选择值。

1.  检查`surface_formats`向量的元素数量。如果它只包含一个值为`VK_FORMAT_UNDEFINED`的元素，这意味着我们可以选择我们想要的任何表面格式。将`desired_surface_format`变量的成员分配给`image_format`和`image_color_space`变量。

1.  如果`surface_formats`向量包含更多元素，遍历向量的每个元素，并将`format`和`colorSpace`成员与`desired_surface_format`变量的相同成员进行比较。如果我们找到一个两个成员都相同的元素，这意味着所需的表面格式受支持，并且我们可以用它来创建 swapchain。将`desired_surface_format`变量的成员分配给`image_format`和`image_color_space`变量。

1.  如果还没有找到匹配项，遍历`surface_formats`向量的所有元素。检查其任何元素的`format`成员是否与所选`surface_format.format`的值相同。如果存在这样的元素，将`desired_surface_format.format`的值分配给`image_format`变量，但将从当前查看的`surface_formats`向量的元素中获取相应的颜色空间并分配给`image_color_space`变量。

1.  如果`surface_formats`变量不包含任何具有所选图像格式的元素，则取向量的第一个元素，并将它的`format`和`colorSpace`成员分配给`image_format`和`image_color_space`变量。

# 它是如何工作的...

要获取所有支持表面格式的列表，我们需要执行两次`vkGetPhysicalDeviceSurfaceFormatsKHR()`函数的调用。首先，我们获取所有支持格式-颜色空间对的数目：

```cpp
uint32_t formats_count = 0; 
VkResult result = VK_SUCCESS; 

result = vkGetPhysicalDeviceSurfaceFormatsKHR( physical_device, presentation_surface, &formats_count, nullptr ); 
if( (VK_SUCCESS != result) || 
    (0 == formats_count) ) { 
  std::cout << "Could not get the number of supported surface formats." << std::endl; 
  return false; 
}

```

接下来，我们可以为实际值准备存储空间，并执行第二次调用以获取它们：

```cpp
std::vector<VkSurfaceFormatKHR> surface_formats( formats_count ); 
result = vkGetPhysicalDeviceSurfaceFormatsKHR( physical_device, presentation_surface, &formats_count, &surface_formats[0] ); 
if( (VK_SUCCESS != result) || 
    (0 == formats_count) ) { 
  std::cout << "Could not enumerate supported surface formats." << std::endl; 
  return false; 
}

```

之后，我们可以选择一个最适合我们需求的受支持表面格式。如果只返回了一个表面格式并且它的值为`VK_FORMAT_UNDEFINED`，这意味着对支持的格式-颜色空间对没有限制。在这种情况下，我们可以选择我们想要的任何表面格式并在创建 swapchain 时使用它：

```cpp
if( (1 == surface_formats.size()) && 
    (VK_FORMAT_UNDEFINED == surface_formats[0].format) ) { 
  image_format = desired_surface_format.format; 
  image_color_space = desired_surface_format.colorSpace; 
  return true; 
}

```

如果`vkGetPhysicalDeviceSurfaceFormatsKHR()`函数返回了更多元素，我们需要从中选择一个。首先，我们检查所选表面格式是否完全受支持--所选图像格式和颜色空间都是可用的：

```cpp
for( auto & surface_format : surface_formats ) { 
  if( (desired_surface_format.format == surface_format.format) && 
      (desired_surface_format.colorSpace == surface_format.colorSpace) ) { 
    image_format = desired_surface_format.format; 
    image_color_space = desired_surface_format.colorSpace; 
    return true; 
  } 
}

```

如果找不到匹配项，我们寻找一个具有相同图像格式但其他颜色空间的成员。我们不能选择任何支持的格式和任何支持的颜色空间--我们必须选择与给定格式相对应的相同颜色空间：

```cpp
for( auto & surface_format : surface_formats ) { 
  if( (desired_surface_format.format == surface_format.format) ) { 
    image_format = desired_surface_format.format; 
    image_color_space = surface_format.colorSpace; 
    std::cout << "Desired combination of format and colorspace is not supported. Selecting other colorspace." << std::endl; 
    return true; 
  } 
}

```

最后，如果我们想使用的格式不受支持，我们只需取第一个可用的图像格式-颜色空间对：

```cpp
image_format = surface_formats[0].format; 
image_color_space = surface_formats[0].colorSpace; 
std::cout << "Desired format is not supported. Selecting available format - colorspace combination." << std::endl; 
return true;

```

# 参见

本章中的以下食谱：

+   *创建交换链*

+   *使用 R8G8B8A8 格式和邮箱展示模式创建交换链*

# 创建交换链

交换链用于在屏幕上显示图像。它是一个图像数组，应用程序可以获取这些图像，然后在我们的应用程序窗口中展示。每个图像都有相同的定义属性集。当我们准备好了所有这些参数，意味着我们选择了一个数字、大小、格式和交换链图像的使用场景，并且获取并选择了一个可用的展示模式，我们就准备好创建一个交换链了。

# 如何做到这一点...

1.  拿到一个创建的逻辑设备对象的句柄。将其存储在一个名为 `logical_device` 的 `VkDevice` 类型变量中。

1.  将创建的展示表面句柄分配给名为 `VkSurfaceKHR` 类型的变量 `presentation_surface`。

1.  将分配给变量 `uint32_t` 类型的 `image_count` 的所需数量的交换链图像句柄。

1.  将所选图像格式和颜色空间的值存储在名为 `VkSurfaceFormatKHR` 类型的变量 `surface_format` 中。

1.  准备所需图像大小并将其分配给名为 `VkExtent2D` 类型的变量 `image_size`。

1.  选择交换链图像的所需使用场景，并将它们存储在名为 `VkImageUsageFlags` 类型的位字段变量 `image_usage` 中。

1.  将存储在名为 `VkSurfaceTransformFlagBitsKHR` 类型的变量 `surface_transform` 中的所选表面转换。

1.  准备一个名为 `present_mode` 的 `VkPresentModeKHR` 类型变量，并将其分配一个所需的展示模式。

1.  创建一个名为 `old_swapchain` 的 `VkSwapchainKHR` 类型变量。如果有之前创建的交换链，将那个交换链的句柄存储在这个变量中。否则，将 `VK_NULL_HANDLE` 值分配给此变量。

1.  创建一个名为 `swapchain_create_info` 的 `VkSwapchainCreateInfoKHR` 类型变量。将以下值分配给此变量的成员：

    +   为 `sType` 的 `VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR` 值

    +   为 `pNext` 的 `nullptr` 值

    +   为 `flags` 的 `0` 值

    +   为 `surface` 的 `presentation_surface` 变量

    +   为 `minImageCount` 的 `image_count` 变量

    +   为 `imageFormat` 的 `surface_format.format` 成员

    +   为 `imageColorSpace` 的 `surface_format.colorSpace` 成员

    +   为 `imageExtent` 的 `image_size` 变量

    +   为 `imageArrayLayers` 的 `1` 值（或更多，如果我们想进行分层/立体渲染）

    +   为 `imageUsage` 的 `image_usage` 变量

    +   为 `imageSharingMode` 的 `VK_SHARING_MODE_EXCLUSIVE` 值

    +   为 `queueFamilyIndexCount` 的 `0` 值

    +   为 `pQueueFamilyIndices` 的 `nullptr` 值

    +   为 `preTransform` 的 `surface_transform` 变量

    +   为 `compositeAlpha` 的 `VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR` 值

    +   为 `presentMode` 的 `present_mode` 变量

    +   为 `clipped` 的 `VK_TRUE`

    +   为 `oldSwapchain` 的 `old_swapchain` 变量

1.  创建一个名为 `swapchain` 的 `VkSwapchainKHR` 类型变量。

1.  调用`vkCreateSwapchainKHR( logical_device, &swapchain_create_info, nullptr, &swapchain )`。使用已创建的逻辑设备句柄、`swapchain_create_info`变量的指针、`nullptr`值以及`swapchain`变量的指针作为函数的参数。

1.  通过将返回值与`VK_SUCCESS`值进行比较，确保调用成功。

1.  调用`vkDestroySwapchainKHR( logical_device, old_swapchain, nullptr )`来销毁旧 swapchain。提供一个已创建的逻辑设备句柄、旧 swapchain 的句柄，以及函数调用中的`nullptr`值。

# 它是如何工作的...

如前所述，swapchain 是一组图像。它们会自动与 swapchain 一起创建。当 swapchain 被销毁时，它们也会被销毁。尽管应用程序可以获取这些图像的句柄，但不允许创建或销毁它们。

创建 swapchain 的过程并不太复杂，但在我们能够创建它之前，我们需要准备相当多的数据：

```cpp
VkSwapchainCreateInfoKHR swapchain_create_info = { 
  VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR, 
  nullptr, 
  0, 
  presentation_surface, 
  image_count, 
  surface_format.format, 
  surface_format.colorSpace, 
  image_size, 
  1, 
  image_usage, 
  VK_SHARING_MODE_EXCLUSIVE, 
  0, 
  nullptr, 
  surface_transform, 
  VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR, 
  present_mode, 
  VK_TRUE, 
  old_swapchain 
}; 

VkResult result = vkCreateSwapchainKHR( logical_device, &swapchain_create_info, nullptr, &swapchain ); 
if( (VK_SUCCESS != result) || 
    (VK_NULL_HANDLE == swapchain) ) { 
  std::cout << "Could not create a swapchain." << std::endl; 
  return false; 
}

```

只能有一个 swapchain 与一个特定应用程序的窗口相关联。当我们创建一个新的 swapchain 时，我们需要销毁之前为同一窗口创建的任何 swapchain：

```cpp
if( VK_NULL_HANDLE != old_swapchain ) { 
  vkDestroySwapchainKHR( logical_device, old_swapchain, nullptr ); 
  old_swapchain = VK_NULL_HANDLE; 
}

```

当 swapchain 准备就绪时，我们可以获取其图像并执行适合指定使用场景的任务。我们不仅限于获取单个图像，就像我们习惯在 OpenGL API（单个后缓冲区）中那样。图像的数量取决于应与 swapchain 一起创建的最小指定图像数量、选择的显示模式以及当前的渲染历史（当前获取和最近显示的图像数量）。

在我们获取到一个图像后，我们可以在我们的应用程序中使用它。最常见的用法是将渲染到图像中（将其用作颜色附件），但我们不仅限于这种用法，我们还可以使用 swapchain 图像执行其他任务。但我们必须确保在给定的平台上可用的相应用法，并且在创建 swapchain 时已指定。并非所有平台都支持所有用法。仅颜色附件用法是强制性的。

当我们完成对图像（或图像）的渲染或其他任务后，我们可以通过显示图像来显示它。此操作将图像返回到显示引擎，根据指定的显示模式用新图像替换当前显示的图像。

# 参见

本章中的以下配方：

+   *创建显示表面*

+   *选择支持向给定表面进行显示的队列家族*

+   *创建启用 WSI 扩展的逻辑设备*

+   *选择所需的显示模式*

+   *获取显示表面的能力*

+   *选择多个 swapchain 图像*

+   *选择 swapchain 图像的大小*

+   *选择 swapchain 图像的期望使用场景*

+   *选择 swapchain 图像的转换*

+   *选择 swapchain 图像的格式*

# 获取交换链图像的句柄

当交换链对象被创建时，获取与交换链一起创建的所有图像的数量和句柄可能非常有用。

# 如何做...

1.  获取创建的逻辑设备对象的句柄。将其存储在名为`logical_device`的`VkDevice`类型变量中。

1.  将创建的交换链的句柄分配给名为`swapchain`的`VkSwapchainKHR`类型变量。

1.  创建一个名为`images_count`的`uint32_t`类型变量。

1.  调用`vkGetSwapchainImagesKHR(logical_device, swapchain, &images_count, nullptr)`，其中在第一个参数中提供创建的逻辑设备句柄，在第二个参数中提供交换链的句柄，在第三个参数中提供指向`images_count`变量的指针。在最后一个参数中提供`nullptr`值。

1.  如果调用成功，即返回值等于`VK_SUCCESS`，则`images_count`变量将包含为给定交换链对象创建的图像总数。

1.  创建一个元素类型为`VkImage`的`std::vector`。命名为`swapchain_images`，并调整大小以便能够容纳至少`images_count`个元素。

1.  调用`vkGetSwapchainImagesKHR(logical_device, swapchain, &images_count, &swapchain_images[0])`，并为前三个参数提供与之前相同的参数。在最后一个参数中，提供指向`swapchain_images`向量第一个元素的指针。

1.  成功时，该向量将包含所有交换链图像的句柄。

# 它是如何工作的...

驱动程序可能创建比交换链创建参数中请求的更多图像。在那里，我们只定义了所需的最小数量，但 Vulkan 实现被允许创建更多。

我们需要知道创建的图像总数，以便能够获取它们的句柄。在 Vulkan 中，当我们想要将渲染输出到图像时，我们需要知道它的句柄。需要创建一个图像视图来包装图像，并在创建帧缓冲区时使用。帧缓冲区，就像 OpenGL 一样，指定了在渲染过程中使用的图像集合（大多数情况下是我们将渲染到它们）。

但这并不是唯一需要知道与交换链一起创建的图像的情况。据说当应用程序想要使用可呈现的图像时，它必须从显示引擎中获取它。图像获取的过程返回一个数字，而不是句柄本身。提供的数字代表使用`vkGetSwapchainImagesKHR()`函数（一个`swapchain_images`变量）获取的图像数组中的图像索引。因此，了解图像的总数、它们的顺序和它们的句柄对于正确使用交换链及其图像是必要的。

要获取图像的总数，我们需要使用以下代码：

```cpp
uint32_t images_count = 0; 
VkResult result = VK_SUCCESS; 

result = vkGetSwapchainImagesKHR( logical_device, swapchain, &images_count, nullptr ); 
if( (VK_SUCCESS != result) || 
    (0 == images_count) ) { 
  std::cout << "Could not get the number of swapchain images." << std::endl; 
  return false; 
}

```

接下来，我们可以为所有图像准备存储空间并获取它们的句柄：

```cpp
swapchain_images.resize( images_count ); 
result = vkGetSwapchainImagesKHR( logical_device, swapchain, &images_count, &swapchain_images[0] ); 
if( (VK_SUCCESS != result) || 
  (0 == images_count) ) { 
  std::cout << "Could not enumerate swapchain images." << std::endl; 
  return false; 
} 

return true;

```

# 参见

本章中的以下内容：

+   *选择交换链图像的数量*

+   *创建交换链*

+   *获取 swapchain 图像*

+   *显示图像*

# 创建一个具有 R8G8B8A8 格式和存在邮箱显示模式的 swapchain

要创建 swapchain，我们需要获取大量的附加信息和准备相当数量的参数。为了展示准备阶段所需的所有步骤的顺序以及如何使用获取的信息，我们将使用任意选择的参数创建一个 swapchain。为此，我们将设置邮箱显示模式，最常用的 R8G8B8A8 颜色格式，具有无符号归一化值（类似于 OpenGL 的 RGBA8 格式），无转换，以及标准颜色附加图像使用。

# 如何做到这一点...

1.  准备一个物理设备句柄。将其存储在名为`physical_device`的类型为`VkPhysicalDevice`的变量中。

1.  获取创建的显示表面的句柄并将其分配给名为`presentation_surface`的类型为`VkSurfaceKHR`的变量。

1.  从`physical_device`变量表示的句柄中获取逻辑设备。将逻辑设备的句柄存储在名为`logical_device`的类型为`VkDevice`的变量中。

1.  创建一个名为`old_swapchain`的类型为`VkSwapchainKHR`的变量。如果之前创建了 swapchain，则将其句柄分配给`old_swapchain`变量。否则，将其分配给`VK_NULL_HANDLE`。

1.  创建一个名为`desired_present_mode`的类型为`VkPresentModeKHR`的变量。

1.  检查`VK_PRESENT_MODE_MAILBOX_KHR`显示模式是否受支持，并将其分配给`desired_present_mode`变量。如果不支持此模式，则使用`VK_PRESENT_MODE_FIFO_KHR`模式（参考*选择期望的显示模式*配方）。

1.  创建一个名为`surface_capabilities`的类型为`VkSurfaceCapabilitiesKHR`的变量。

1.  获取显示表面的能力并将它们存储在`surface_capabilities`变量中。

1.  创建一个名为`number_of_images`的类型为`uint32_t`的变量。根据获取的表面能力，将所需的最小图像数量分配给`number_of_images`变量（参考*选择 swapchain 图像数量*配方）。

1.  创建一个名为`image_size`的类型为`VkExtent2D`的变量。根据获取的表面能力，将 swapchain 图像的大小分配给`image_size`变量（参考*选择 swapchain 图像大小*配方）。

1.  确保变量`image_size`的`width`和`height`成员大于零。如果它们不是，不要尝试创建 swapchain，但不要关闭应用程序——这种情况下可能发生在窗口最小化时。

1.  创建一个名为`image_usage`的类型为`VkImageUsageFlags`的变量。将`VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT`图像使用分配给它（参考*选择 swapchain 图像的期望使用场景*配方）。

1.  创建一个名为 `surface_transform` 的 `VkSurfaceTransformFlagBitsKHR` 类型的变量。将一个单位变换（`VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR` 的值）存储在该变量中。根据获取的表面能力，检查它是否受支持。如果不支持，将获取的能力中的 `currentTransform` 成员赋值给 `surface_transform` 变量（参考 *选择交换链图像的变换* 章节中的配方）。

1.  创建一个名为 `image_format` 的 `VkFormat` 类型的变量和一个名为 `image_color_space` 的 `VkColorSpaceKHR` 类型的变量。

1.  使用获取的能力，尝试使用 `VK_FORMAT_R8G8B8A8_UNORM` 图像格式和 `VK_COLOR_SPACE_SRGB_NONLINEAR_KHR` 色彩空间。如果格式或色彩空间，或两者都不受支持，则从表面能力中选择其他值（参考 *选择交换链图像的格式* 章节中的配方）。

1.  创建一个名为 `swapchain` 的 `VkSwapchainKHR` 类型的变量。

1.  使用 `logical_device`、`presentation_surface`、`number_of_images`、`image_format`、`image_color_space`、`size_of_images`、`image_usage`、`surface_transform`、`desired_present_mode` 和 `old_swapchain` 变量创建一个交换链，并将句柄存储在 `swapchain` 变量中。请记住检查交换链创建是否成功。（参考 *创建交换链* 章节中的配方）。

1.  创建一个名为 `swapchain_images` 的 `std::vector<VkImage>` 类型的变量，并将创建的交换链图像的句柄存储在其中（参考 *掌握交换链图像的句柄* 章节中的配方）。

# 它是如何工作的...

当我们想要创建一个交换链时，我们首先需要考虑我们想要使用哪种展示模式。由于邮箱模式允许我们在不出现屏幕撕裂的情况下展示最新的图像（它类似于三缓冲），这似乎是一个不错的选择：

```cpp
VkPresentModeKHR desired_present_mode; 
if( !SelectDesiredPresentationMode( physical_device, presentation_surface, VK_PRESENT_MODE_MAILBOX_KHR, desired_present_mode ) ) { 
  return false; 
}

```

接下来，我们需要获取展示表面的能力，并使用它们来设置所需图像的数量、大小（维度）、使用场景、展示期间应用的变换以及它们的格式和色彩空间：

```cpp
VkSurfaceCapabilitiesKHR surface_capabilities; 
if( !GetCapabilitiesOfPresentationSurface( physical_device, presentation_surface, surface_capabilities ) ) { 
  return false; 
} 

uint32_t number_of_images; 
if( !SelectNumberOfSwapchainImages( surface_capabilities, number_of_images ) ) { 
  return false; 
} 

VkExtent2D image_size; 
if( !ChooseSizeOfSwapchainImages( surface_capabilities, image_size ) ) { 
  return false; 
} 
if( (0 == image_size.width) || 
    (0 == image_size.height) ) { 
  return true; 
} 

VkImageUsageFlags image_usage; 
if( !SelectDesiredUsageScenariosOfSwapchainImages( surface_capabilities, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, image_usage ) ) { 
  return false; 
} 

VkSurfaceTransformFlagBitsKHR surface_transform; 
SelectTransformationOfSwapchainImages( surface_capabilities, VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR, surface_transform ); 

VkFormat image_format; 
VkColorSpaceKHR image_color_space; 
if( !SelectFormatOfSwapchainImages( physical_device, presentation_surface, { VK_FORMAT_R8G8B8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR }, image_format, image_color_space ) ) { 
  return false; 
}

```

最后，在完成所有这些准备工作后，我们可以创建一个交换链，销毁旧的交换链（如果我们想用新的交换链替换之前创建的交换链），并获取与其一起创建的图像句柄：

```cpp
if( !CreateSwapchain( logical_device, presentation_surface, number_of_images, { image_format, image_color_space }, image_size, image_usage, surface_transform, desired_present_mode, old_swapchain, swapchain ) ) { 
  return false; 
} 

if( !GetHandlesOfSwapchainImages( logical_device, swapchain, swapchain_images ) ) { 
  return false; 
} 
return true;

```

# 参见

本章中的以下配方：

+   *创建一个展示表面*

+   *创建启用 WSI 扩展的逻辑设备*

+   *选择期望的展示模式*

+   *获取展示表面的能力*

+   *选择交换链图像的数量*

+   *选择交换链图像的大小*

+   *选择交换链图像的期望使用场景*

+   *选择交换链图像的变换*

+   *选择交换链图像的格式*

+   *创建交换链*

+   *掌握交换链图像的句柄*

# 获取交换链图像

在我们能够使用交换链图像之前，我们需要向显示引擎请求它。这个过程被称为**图像获取**。它返回一个图像的索引，该索引是 `vkGetSwapchainImagesKHR()` 函数返回的图像数组中的索引，如 *获取交换链图像句柄* 菜单中所述。

# 准备就绪

要在 Vulkan 中获取图像，我们需要指定两种尚未描述的对象之一。这些是信号量和栅栏。

信号量用于同步设备的队列。这意味着当我们提交命令进行处理时，这些命令可能需要另一个任务完成。在这种情况下，我们可以指定这些命令应该在执行之前等待其他命令。这正是信号量的作用。它们用于内部队列同步，但我们不能使用它们来同步应用程序与提交的命令（请参阅 第三章，*命令缓冲区和同步*中的 *创建信号量* 菜单）。

要这样做，我们需要使用栅栏。它们用于通知应用程序某些工作已完成。应用程序可以获取栅栏的状态，并根据获取的信息检查某些命令是否仍在处理中，或者它们是否已经完成了分配的任务（请参阅 第三章，*命令缓冲区和同步*中的 *创建栅栏* 菜单）。

# 如何操作...

1.  获取创建的逻辑设备的句柄，并将其存储在类型为 `VkDevice` 的变量 `logical_device` 中。

1.  准备一个交换链对象的句柄，并将其分配给名为 `swapchain` 的 `VkSwapchainKHR` 变量。

1.  准备一个类型为 `VkSemaphore` 的变量 `semaphore` 的信号量，或者准备一个栅栏并将它的句柄分配给类型为 `VkFence` 的变量 `fence`。您可以准备这两个同步对象，但至少需要其中一个（无论哪个）。

1.  创建一个名为 `image_index` 的 `uint32_t` 类型的变量。

1.  调用 `vkAcquireNextImageKHR( logical_device, swapchain, <timeout>, semaphore, fence, &image_index )`。在第一个参数中提供逻辑设备的句柄，在第二个参数中提供交换链对象的句柄。对于第三个参数，名为 `<timeout>`，提供函数返回超时错误的时间值。您还需要提供一个或两个同步原语——交换链和/或栅栏。对于最后一个参数，提供一个指向 `image_index` 变量的指针。

1.  检查`vkAcquireNextImageKHR()`函数返回的值。如果返回值等于`VK_SUCCESS`或`VK_SUBOPTIMAL_KHR`，则调用成功，`image_index`变量将包含一个指向由`vkGetSwapchainImagesKHR()`函数返回的交换链图像数组的索引（参考*获取交换链图像句柄*过程）。但如果返回了`VK_ERROR_OUT_OF_DATE_KHR`值，则无法使用交换链中的任何图像。你必须销毁给定的交换链，并重新创建它以获取图像。

# 它是如何工作的...

`vkAcquireNextImageKHR()`函数返回一个索引，该索引指向由`vkGetSwapchainImagesKHR()`函数返回的交换链图像数组。它不返回该图像的句柄。以下代码说明了这个过程：

```cpp
VkResult result; 

result = vkAcquireNextImageKHR( logical_device, swapchain, 2000000000, semaphore, fence, &image_index ); 
switch( result ) { 
  case VK_SUCCESS: 
  case VK_SUBOPTIMAL_KHR: 
    return true; 
default: 
  return false; 
}

```

在代码示例中，我们调用`vkAcquireNextImageKHR()`函数。有时由于演示引擎的内部机制，图像可能无法立即可用。甚至可能无限期地等待！这种情况发生在我们想要获取比演示引擎能提供的更多图像时。这就是为什么在前一个函数的第三个参数中，我们提供了一个以纳秒为单位的超时值。它告诉硬件我们可以等待图像多长时间。在此时间之后，函数将通知我们获取图像花费了太长时间。在前面的示例中，我们通知驱动程序我们不想等待超过 2 秒钟来获取图像。

其他有趣的参数是一个信号量和一个栅栏。当我们获取一个图像时，我们仍然可能无法立即为我们自己的目的使用它。我们需要等待所有之前提交的引用此图像的操作完成。为此，可以使用栅栏，通过它应用程序可以检查何时修改图像是安全的。但我们也可以告诉驱动程序在开始处理使用给定图像的新命令之前应该等待。为此，使用信号量，这通常是一个更好的选择。

在应用程序端等待比仅在 GPU 端等待对性能的伤害更大。

在交换链图像获取过程中，返回值也非常重要。当函数返回`VK_SUBOPTIMAL_KHR`值时，这意味着我们仍然可以使用该图像，但它可能不再最适合演示引擎。我们应该从获取图像的交换链中重新创建交换链。但我们不需要立即这样做。当函数返回`VK_ERROR_OUT_OF_DATE_KHR`值时，我们不能再使用给定交换链中的图像，并且我们需要尽快重新创建它。

关于交换链图像获取的最后一点是，在我们能够使用图像之前，我们需要更改（转换）其布局。布局是图像的内部内存组织，这可能会根据图像当前使用的目的而有所不同。如果我们想以不同的方式使用图像，我们需要更改其布局。

例如，展示引擎使用的图像必须具有 `VK_IMAGE_LAYOUT_PRESENT_SRC_KHR` 布局。但如果我们想要将图像渲染到图像中，它必须具有 `VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL` 布局。改变布局的操作称为转换（参考第四章 *设置图像内存屏障* 的配方，*资源和内存*）。

# 参见

+   本章中的以下配方：

    +   *选择所需的展示模式*

    +   *创建一个交换链*

    +   *获取交换链图像的句柄*

    +   *展示一个图像*

+   在第四章 *资源和内存* 中，查看以下配方：

    +   *设置图像内存屏障*

+   在第三章 *命令缓冲区和同步* 中，查看以下配方：

    +   *创建一个信号量*

    +   *创建一个栅栏*

# 展示一个图像

在我们将图像渲染到交换链图像或用于其他任何目的之后，我们需要将图像归还给展示引擎。这个操作称为展示，它会在屏幕上显示图像。

# 准备工作

在这个配方中，我们将使用以下定义的自定义结构：

```cpp
struct PresentInfo { 
  VkSwapchainKHR  Swapchain; 
  uint32_t        ImageIndex; 
};

```

它用于定义我们想要展示图像的交换链，以及我们想要显示的图像（其索引）。对于每个交换链，我们一次只能展示一个图像。

# 如何进行...

1.  准备一个支持展示的队列的句柄。将其存储在名为 `queue` 的 `VkQueue` 类型的变量中。

1.  准备一个名为 `rendering_semaphores` 的 `std::vector<VkSemaphore>` 类型的变量。将与此相关的信号量插入到这个向量中，这些信号量与引用我们想要展示的图像的渲染命令相关联。

1.  创建一个名为 `swapchains` 的 `std::vector<VkSwapchainKHR>` 类型的变量，用于存储我们想要展示图像的所有交换链的句柄。

1.  创建一个名为 `image_indices` 的 `std::vector<uint32_t>` 类型的变量。将向量的大小调整为与 `swapchains` 向量相同。对于 `image_indices` 变量的每个元素，分配来自相应交换链（在 `swapchains` 向量中的相同位置）的图像的索引。

1.  创建一个名为 `present_info` 的 `VkPresentInfoKHR` 类型的变量。为其成员分配以下值：

    +   `VK_STRUCTURE_TYPE_PRESENT_INFO_KHR` 对应的 `sType` 值

    +   `pNext` 的 `nullptr` 值

    +   `rendering_semaphores` 向量中的元素数量用于 `waitSemaphoreCount`

    +   指向 `rendering_semaphores` 向量第一个元素的指针用于 `pWaitSemaphores`

    +   `swapchainCount` 的 `swapchains` 向量中的元素数量

    +   `pSwapchains`的`swapchains`向量的第一个元素的指针

    +   `pImageIndices`的`image_indices`向量的第一个元素的指针

    +   `pResults`的`nullptr`值

1.  调用`vkQueuePresentKHR(queue, &present_info)`并提供我们想要提交此操作的队列的句柄，以及`present_info`变量的指针。

1.  通过将返回值与`VK_SUCCESS`进行比较，确保调用成功。

# 它是如何工作的...

展示操作将图像返回给展示引擎，展示引擎根据展示模式显示图像。我们可以同时展示多个图像，但只能从给定的 swapchain 中选择一个图像。要展示一个图像，我们需要提供其索引，该索引由`vkGetSwapchainImagesKHR()`函数返回的数组中（参考*获取 swapchain 图像句柄*配方）：

```cpp
VkPresentInfoKHR present_info = { 
  VK_STRUCTURE_TYPE_PRESENT_INFO_KHR, 
  nullptr, 
  static_cast<uint32_t>(rendering_semaphores.size()), 
  rendering_semaphores.size() > 0 ? &rendering_semaphores[0] : nullptr, 
  static_cast<uint32_t>(swapchains.size()), 
  swapchains.size() > 0 ? &swapchains[0] : nullptr, 
  swapchains.size() > 0 ? &image_indices[0] : nullptr, 
  nullptr 
};  

result = vkQueuePresentKHR( queue, &present_info ); 
switch( result ) { 
case VK_SUCCESS: 
  return true; 
default: 
  return false; 
}

```

在前面的示例中，我们想要展示图像的 swapchain 的句柄和图像的索引被放置在`swapchains`和`image_indices`向量中。

在我们可以提交一个图像之前，我们需要将其布局更改为`VK_IMAGE_LAYOUT_PRESENT_SRC_KHR`，否则展示引擎可能无法正确显示此类图像。

信号量用于通知硬件何时可以安全地显示图像。当我们提交一个渲染命令时，我们可以将一个信号量与这样的提交关联。然后，当命令完成时，这个信号量将改变其状态为已触发。我们应该创建一个信号量并将其与引用可展示图像的命令关联。这样，当我们展示一个图像并提供这样的信号量时，硬件将知道何时图像不再使用，并且展示它不会中断任何先前发出的操作。

# 参见

+   本章中的以下配方：

    +   *选择一个期望的展示模式*

    +   *创建 swapchain*

    +   *获取 swapchain 图像句柄*

    +   *获取 swapchain 图像*

+   在第三章，*命令缓冲区和同步*中查看以下配方：

    +   *创建一个信号量*

    +   *创建一个栅栏*

+   在[第四章](https://cdp.packtpub.com/vulkancookbook/wp-admin/post.php?post=29&action=edit#post_207)，*资源和内存*中查看以下配方：

    +   *设置图像内存屏障*

# 销毁 swapchain

当我们完成使用 swapchain 时，因为我们不再想要展示图像，或者因为我们只是关闭我们的应用程序，我们应该销毁它。在销毁用于给定 swapchain 创建的展示表面之前，我们需要销毁它。

# 如何操作...

1.  获取逻辑设备的句柄并将其存储在名为`logical_device`的`VkDevice`类型的变量中。

1.  获取需要销毁的 swapchain 对象的句柄。将其存储在名为`swapchain`的`VkSwapchainKHR`类型的变量中。

1.  调用`vkDestroySwapchainKHR(logical_device, swapchain, nullptr)`，并将`logical_device`变量作为第一个参数，swapchain 句柄作为第二个参数。将最后一个参数设置为`nullptr`。

1.  为了安全起见，将`VK_NULL_HANDLE`值分配给`swapchain`变量。

# 它是如何工作的...

要销毁 swapchain，我们可以准备类似于以下示例的代码：

```cpp
if( swapchain ) { 
  vkDestroySwapchainKHR( logical_device, swapchain, nullptr ); 
  swapchain = VK_NULL_HANDLE; 
}

```

首先，我们检查是否真的创建了一个 swapchain（如果其句柄不为空）。接下来，我们调用`vkDestroySwapchainKHR()`函数，然后将`VK_NULL_HANDLE`值赋给`swapchain`变量以确保我们不会尝试删除同一个对象两次。

# 参见

+   本章中的*创建 swapchain*配方

# 销毁展示表面

展示表面代表我们应用程序的窗口。它在创建 swapchain 的过程中被使用，以及其他目的。这就是为什么在基于给定表面的 swapchain 被销毁完成后，我们应该销毁展示表面。

# 如何操作...

1.  准备一个 Vulkan 实例的句柄，并将其存储在名为`instance`的`VkInstance`类型变量中。

1.  获取展示表面的句柄，并将其分配给名为`presentation_surface`的`VkSurfaceKHR`类型变量。

1.  调用`vkDestroySurfaceKHR(instance, presentation_surface, nullptr)`，并将`instance`和`presentation_surface`变量作为前两个参数，最后一个参数设置为`nullptr`。

1.  为了安全起见，将`VK_NULL_HANDLE`值分配给`presentation_surface`变量。

# 它是如何工作的...

展示表面的销毁与其他到目前为止展示的 Vulkan 资源的销毁非常相似。我们确保不提供`VK_NULL_HANDLE`值，并调用`vkDestroySurfaceKHR()`函数。之后，我们将`VK_NULL_HANDLE`值分配给`presentation_surface`变量：

```cpp
if( presentation_surface ) { 
  vkDestroySurfaceKHR( instance, presentation_surface, nullptr ); 
  presentation_surface = VK_NULL_HANDLE; 
}

```

# 参见

+   本章中的*创建展示表面*配方
