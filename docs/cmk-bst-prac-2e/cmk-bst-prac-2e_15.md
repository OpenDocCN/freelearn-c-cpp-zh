# 第十二章：跨平台编译自定义工具链

CMake 的一个强大特性是它对跨平台软件构建的支持。简单来说，这意味着通过 CMake，可以将任何平台的项目构建为任何其他平台的软件，只要在运行 CMake 的系统上提供必要的工具。在构建软件时，我们通常谈论编译器和链接器，它们当然是构建软件的必需工具。然而，如果我们仔细看看，构建软件时通常还涉及一些其他工具、库和文件。统称这些工具、库和文件通常被称为 CMake 中的工具链。

到目前为止，本书中的所有示例都是针对 CMake 运行所在的系统构建的。在这些情况下，CMake 通常能很好地找到正确的工具链。然而，如果软件是为另一个平台构建的，通常必须由开发者指定工具链。工具链定义可能相对简单，仅指定目标平台，或者可能复杂到需要指定单个工具的路径，甚至是为了为特定芯片组创建二进制文件而指定特定的编译器标志。

在交叉编译的上下文中，工具链通常伴随有`root`文件夹，用于查找编译和链接软件所需的库和文件，以便将软件编译到预期的目标平台。

虽然交叉编译一开始可能让人感到害怕，但使用 CMake 正确配置时，它通常并不像看起来那样困难。本章将介绍如何使用工具链文件以及如何自己编写工具链文件。我们将详细探讨在软件构建的不同阶段涉及哪些工具。最后，我们将介绍如何设置 CMake，使其能够通过模拟器运行测试。

本章将涵盖以下主要内容：

+   使用现有的跨平台工具链文件

+   创建工具链文件

+   测试交叉编译的二进制文件

+   测试工具链的支持特性

本章结束时，你将熟练掌握如何处理现有的工具链，并了解如何使用 CMake 为不同平台构建和测试软件。我们将深入探讨如何测试编译器的某个特性，以确定它是否适合我们的用途。

# 技术要求

与前几章一样，示例是用 CMake 3.25 进行测试的，并在以下任一编译器上运行：

+   **GNU 编译器集合 9**（**GCC 9**）或更新版本，包括用于**arm 硬浮动**（**armhf**）架构的交叉编译器

+   Clang 12 或更新版本

+   **Microsoft Visual Studio C++ 19**（**MSVC 19**）或更新版本

+   对于 Android 示例，**Android 原生开发工具包**（**Android NDK**）23b 或更新版本是必需的。安装说明可以在官方的 Android 开发文档中找到：[`developer.android.com/studio/projects/install-ndk`](https://developer.android.com/studio/projects/install-ndk)。

+   对于 Apple 嵌入式示例，建议使用 Xcode 12 或更新版本，以及**iOS 软件开发工具包 12.4**（**iOS SDK 12.4**）。

本书的所有示例和源代码都可以在 GitHub 仓库中找到。如果缺少任何软件，相应的示例将从构建中排除。仓库地址在这里：[`github.com/PacktPublishing/CMake-Best-Practices---2nd-Edition/`](https://github.com/PacktPublishing/CMake-Best-Practices---2nd-Edition/)。

# 使用现有的跨平台工具链文件

当为多个平台构建软件时，最直接的方法是直接在目标系统上进行编译。其缺点是每个开发者必须有一个正在运行的目标系统来进行构建。如果这些是桌面系统，可能会相对顺利，但在不同的安装环境之间迁移以开发软件也会使开发者的工作流程变得非常繁琐。像嵌入式系统这样不太强大的设备，由于缺乏适当的开发工具，或者因为编译软件非常耗时，可能会非常不方便。

因此，从开发者的角度来看，更便捷的方式是使用交叉编译。这意味着软件工程师在自己的机器上编写代码并构建软件，但生成的二进制文件是为不同平台的。构建软件的机器和平台通常称为*主机机器*和*主机平台*，而软件应运行的平台称为*目标平台*。例如，开发者在运行 Linux 的*x64*桌面机器上编写代码，但生成的二进制文件是为运行在*arm64*处理器上的嵌入式 Linux 系统设计的。因此，主机平台是*x64 Linux*，目标平台是*arm64 Linux*。要进行交叉编译软件，以下两项是必需的：

+   一个能够生成正确格式二进制文件的工具链

+   为目标系统编译的项目依赖项

工具链是一组工具，如编译器、链接器和归档器，用于生成在主机系统上运行，但为目标系统生成输出的二进制文件。依赖项通常会收集在一个*sysroot*目录中。Sysroot 是包含根文件系统精简版的目录，所需的库会存储在其中。对于交叉编译，这些目录作为搜索依赖项的根目录。

一些工具，例如 `CMAKE_TOOLCHAIN_FILE` 变量，或者从 CMake 3.21 开始，使用 `--toolchain` 选项，像这样：

```cpp
cmake -DCMAKE_TOOLCHAIN_FILE=toolchain.cmake -S <SourceDir> -B
  <BuildDir>
cmake  --toolchain arm64.toolchain.cmake -S <SourceDir> -B <BuildDir>
```

这些调用是等效的。如果 `CMAKE_TOOLCHAIN_FILE` 被设置为环境变量，CMake 也会进行解析。如果使用 CMake 预设，配置预设可能会通过 `toolchainFile` 选项配置工具链文件，像这样：

```cpp
{
    "name": "arm64-build-debug",
    "generator" : "Ninja",
    "displayName": "Arm 64 Debug",
    "toolchainFile": "${sourceDir}/arm64.toolchain.cmake",
    "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Debug"
    }
},
```

`toolchainFile`选项支持宏扩展，具体描述请参见*第九章*，*创建可复现的构建环境*。如果工具链文件的路径是相对路径，CMake 会先在`build`目录下查找，如果在那里没有找到文件，它会从源目录开始查找。由于`CMAKE_TOOLCHAIN_FILE`是一个缓存变量，它只需要在第一次运行 CMake 时指定；之后的运行将使用缓存的值。

在第一次运行时，CMake 会执行一些内部查询来确定工具链支持哪些功能。这无论是否使用工具链文件指定工具链，或使用默认系统工具链时，都会发生。有关这些测试是如何执行的更深入的介绍，请参考*测试工具链支持的功能*部分。CMake 将在第一次运行时输出各种功能和属性的测试结果，类似如下：

```cpp
-- The CXX compiler identification is GNU 9.3.0
-- The C compiler identification is GNU 9.3.0
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/arm-linux-gnueabihf-g++-
  9 - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /usr/bin/arm-linux-gnueabi-gcc-9 -
  skipped
-- Detecting C compile features
-- Detecting C compile features - done
```

功能检测通常发生在`CMakeLists.txt`文件中的第一次调用`project()`时。但是，任何启用先前禁用的语言的后续`project()`调用都会触发进一步的检测。如果在`CMakeLists.txt`文件中使用`enable_language()`来启用额外的编程语言，也会发生同样的情况。

由于工具链的功能和测试结果是被缓存的，因此无法更改已配置构建目录的工具链。CMake 可能会检测到工具链已经更改，但通常情况下，替换缓存变量是不完全的。因此，在更改工具链之前，应该完全删除构建目录。

配置后切换工具链

在切换工具链之前，请始终完全清空构建目录。仅删除`CMakeCache.txt`文件是不够的，因为与工具链相关的内容可能会被缓存到不同的位置。如果你经常为多个平台构建项目，使用为每个工具链分配的独立构建目录可以显著加快开发过程。

CMake 的工作方式是一个项目应该使用相同的工具链来进行所有操作。因此，直接支持使用多个工具链的方式并不存在。如果确实需要这样做，那么需要将需要不同工具链的项目部分配置为子构建，具体方法请参见*第十章*，*在超构建中处理分布式仓库和依赖关系*。

工具链应尽可能保持精简，并且与任何项目完全解耦。理想情况下，它们可以在不同的项目中复用。通常，工具链文件是与用于交叉编译的任何 SDK 或 sysroot 一起捆绑的。然而，有时它们需要手动编写。

# 创建工具链文件

工具链文件一开始可能看起来令人害怕，但仔细检查后，它们通常相对简单。定义跨编译工具链很难的误解源于互联网上存在许多过于复杂的工具链文件示例。许多示例是为早期版本的 CMake 编写的，因此实现了许多额外的测试和检查，而这些现在已经是 CMake 的一部分。CMake 工具链文件基本上做以下几件事：

+   定义目标系统和架构。

+   提供构建软件所需的任何工具的路径，这些工具通常只是编译器。

+   为编译器和链接器设置默认标志。

+   如果是跨编译，指向 sysroot 并可能指向任何暂存目录。

+   设置 CMake `find_` 命令的搜索顺序提示。更改搜索顺序是项目可能定义的内容，是否应将其放在工具链文件中或由项目处理是有争议的。有关 `find_` 命令的详细信息，请参见 *第五章*，*集成第三方库和依赖管理*。

一个执行所有这些操作的示例工具链可能如下所示：

```cpp
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)
set(CMAKE_C_COMPILER /usr/bin/arm-linux-gnueabi-gcc-9)
set(CMAKE_CXX_COMPILER /usr/bin/arm-linux-gnueabihf-g++-9)
set(CMAKE_C_FLAGS_INIT -pedantic)
set(CMAKE_CXX_FLAGS_INIT -pedantic)
set(CMAKE_SYSROOT /home/builder/raspi-sysroot/)
set(CMAKE_STAGING_PREFIX /home/builder/raspi-sysroot-staging/)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)
```

这个示例会定义一个工具链，目标是为在主机系统的 `/usr/bin/` 文件夹上运行的 Linux 操作系统进行构建。接着，编译器标志设置为打印由严格的 `-pedantic` 标志要求的所有警告。然后，设置 sysroot 以查找任何所需的库，路径为 `/home/builder/raspi-sysroot/`，并设置跨编译时用于安装内容的暂存目录为 `/home/builder/raspi-sysroot-staging/`。最后，改变 CMake 的搜索行为，使得程序仅在主机系统上搜索，而库、`include` 文件和包仅在 sysroot 中搜索。关于工具链文件是否应该影响搜索行为，存在争议。通常，只有项目知道它正在尝试查找什么，因此在工具链文件中做假设可能会破坏这一点。然而，只有工具链知道应该使用哪个系统根目录以及其中包含哪些类型的文件，因此让工具链来定义这一点可能会更方便。一个好的折中方法是使用 CMake 预设来定义工具链和搜索行为，而不是将其放在项目文件或工具链文件中。

## 定义目标系统

跨编译的目标系统由以下三个变量定义 – `CMAKE_SYSTEM_NAME`、`CMAKE_SYSTEM_PROCESSOR` 和 `CMAKE_SYSTEM_VERSION`。它们分别对应 `CMAKE_HOST_SYSTEM_NAME`、`CMAKE_HOST_SYSTEM_PROCESSOR` 和 `CMAKE_HOST_SYSTEM_VERSION` 变量，这些变量描述了构建所在平台的系统信息。

`CMAKE_SYSTEM_NAME` 变量描述了要构建软件的目标操作系统。设置这个变量很重要，因为它会导致 CMake 将 `CMAKE_CROSSCOMPILING` 变量设置为 `true`。常见的值有 `Linux`、`Windows`、`Darwin`、`Android` 或 `QNX`，你也可以使用更具体的平台名称，例如 `WindowsPhone`、`WindowsCE` 或 `WindowsStore`。对于裸机嵌入式设备，`CMAKE_SYSTEM_NAME` 变量设置为 `Generic`。不幸的是，在写这篇文档时，CMake 文档中没有官方的支持系统列表。然而，如果需要，可以查看本地 CMake 安装中的 `/Modules/Platform` 文件夹中的文件。

`CMAKE_SYSTEM_PROCESSOR` 变量用于描述平台的硬件架构。如果未指定，将假定使用 `CMAKE_HOST_SYSTEM_PROCESSOR` 变量的值。在从 64 位平台交叉编译到 32 位平台时，即使处理器类型相同，也应该设置目标处理器架构。对于 Android 和 Apple 平台，通常不指定处理器。当为 Apple 目标交叉编译时，实际设备由使用的 SDK 定义，SDK 由 `CMAKE_OSX_SYSROOT` 变量指定。为 Android 交叉编译时，使用诸如 `CMAKE_ANDROID_ARCH_ABI`、`CMAKE_ANDROID_ARM_MODE` 和（可选的）`CMAKE_ANDROID_ARM_NEON` 等专用变量来控制目标架构。关于 Android 的构建会在 *为 Android 交叉编译* 部分中详细介绍。

定义目标系统的最后一个变量是 `CMAKE_SYSTEM_VERSION`。它的内容取决于构建的系统。对于 `WindowsCE`、`WindowsStore` 和 `WindowsPhone`，它用于定义使用哪个版本的 Windows SDK。在 Linux 上，通常省略此项，或者如果相关，可能包含目标系统的内核版本。

使用 `CMAKE_SYSTEM_NAME`、`CMAKE_SYSTEM_PROCESSOR` 和 `CMAKE_SYSTEM_VERSION` 变量，通常可以完全指定目标平台。然而，一些生成器，如 Visual Studio，直接支持其本地平台。对于这些平台，可以通过 CMake 的 `-A` 命令行选项来设置架构，方法如下：

```cpp
cmake -G "Visual Studio 2019" -A Win32 -T host=x64
```

当使用预设时，`architecture` 设置可以在配置预设中使用，以达到相同的效果。一旦定义了目标系统，就可以定义用于实际构建软件的工具。

一些编译器，如 Clang 和 `CMAKE_<LANG>_COMPILER_TARGET` 变量也被使用。对于 Clang，值是目标三元组，如 `arm-linux-gnueabihf`，而对于 QNX GCC，编译器名称和目标的值如 `gcc_ntoarmv7le`。Clang 的支持三元组在其官方文档中有描述，网址为 [`clang.llvm.org/docs/CrossCompilation.html`](https://clang.llvm.org/docs/CrossCompilation.html)。

对于 QNX 可用的选项，应该参考 QNX 文档，网址为 [`www.qnx.com/developers/docs/`](https://www.qnx.com/developers/docs/)。

所以，使用 Clang 的工具链文件可能如下所示：

```cpp
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)
set(CMAKE_C_COMPILER /usr/bin/clang)
set(CMAKE_C_COMPILER_TARGET arm-linux-gnueabihf)
set(CMAKE_CXX_COMPILER /usr/bin/clang++)
set(CMAKE_CXX_COMPILER_TARGET arm-linux-gnueabihf)
```

在这个例子中，Clang 被用来编译运行在 ARM 处理器上的 Linux 系统的 C 和 C++ 代码，并且该系统支持硬件浮点运算。定义目标系统通常会直接影响将使用的构建工具。在下一节中，我们将探讨如何为交叉编译选择编译器及相关工具。

## 选择构建工具

在构建软件时，编译器通常是首先想到的工具，在大多数情况下，仅设置工具链文件中的编译器就足够了。编译器的路径由 `CMAKE_<LANG>_COMPILER` 缓存变量设置，可以在工具链文件中设置，也可以手动传递给 CMake。如果路径是绝对路径，则会直接使用；否则，将使用与 `find_program()` 相同的搜索顺序，这也是为什么在工具链文件中更改搜索行为时需要谨慎的原因之一。如果工具链文件和用户都没有指定编译器，CMake 将尝试根据指定的目标平台和生成器自动选择一个编译器。此外，编译器还可以通过与 `<LANG>` 对应的环境变量来设置。所以，`C` 用来设置 C 编译器，`CXX` 用来设置 C++ 编译器，`ASM` 用来设置汇编器，依此类推。

一些生成器，如 Visual Studio，可能支持其自定义的工具集定义，这些定义的工作方式不同。它们可以通过 `-T` 命令行选项进行设置。以下命令将告诉 CMake 为 Visual Studio 生成代码，以便为 32 位系统生成二进制文件，但使用 64 位编译器进行编译：

```cpp
cmake -G "Visual Studio 2019" -A Win32 -T host=x64
```

这些值也可以通过工具链文件中的 `CMAKE_GENERATOR_TOOLSET` 变量进行设置。这个变量不应该在项目中设置，因为它显然不符合 CMake 项目文件与生成器和平台无关的原则。

对于 Visual Studio 用户，通过安装同一版本的预览版和正式版，可以在计算机上同时安装多个相同版本的 Visual Studio 实例。如果是这种情况，可以在工具链文件中将 `CMAKE_GENERATOR_INSTANCE` 变量设置为 Visual Studio 的绝对安装路径。

通过指定要使用的编译器，CMake 将为编译器和链接器选择默认标志，并通过设置`CMAKE_<LANG>_FLAGS`和`CMAKE_<LANG>_FLAGS_<CONFIG>`使其在项目中可用，其中 `<LANG>` 代表相应的编程语言，`<CONFIG>` 代表构建配置，如调试或发布。默认的链接器标志由 `CMAKE_<TARGETTYPE>_LINKER_FLAGS` 和 `CMAKE_<TARGETTYPE>_LINKER_FLAGS_<CONFIG>` 变量设置，其中 `<TARGETTYPE>` 可以是 `EXE`、`STATIC`、`SHARED` 或 `MODULE`。

要向默认标志添加自定义标志，可以使用带有 `_INIT` 后缀的变量—例如，`CMAKE_<LANG>_FLAGS_INIT`。在使用工具链文件时，`_INIT` 变量用于设置任何必要的标志。一个从 64 位主机为 32 位目标进行 GCC 编译的工具链文件可能如下所示：

```cpp
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR i686)
set(CMAKE_C_COMPILER  gcc)
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_C_FLAGS_INIT -m32)
set(CMAKE_CXX_FLAGS_INIT -m32)
set(CMAKE_EXE_LINKER_FLAGS_INIT -m32)
set(CMAKE_SHARED_LINKER_FLAGS_INIT -m32)
set(CMAKE_STATIC_LINKER_FLAGS_INIT -m32)
set(CMAKE_MODULE_LINKER_FLAGS_INIT -m32)
```

对于简单的项目，设置目标系统和工具链可能已经足够开始创建二进制文件，但对于更复杂的项目，它们可能需要访问目标系统的库和头文件。对于这种情况，可以在工具链文件中指定 sysroot。

## 设置 sysroot

在进行交叉编译时，所有链接的依赖项显然也必须与目标平台匹配，一种常见的处理方法是创建一个 sysroot，它是目标系统的根文件系统，存储在一个文件夹中。虽然 sysroot 可以包含完整的系统，但通常会被精简到仅提供所需内容。sysroot 的详细描述见于 *第九章*，*创建可重现的* *构建环境*。

设置 sysroot 通过将 `CMAKE_SYSROOT` 设置为其路径来完成。如果设置了该值，CMake 默认会首先在 sysroot 中查找库和头文件，除非另有说明，正如在 *第五章*，*集成第三方库和依赖管理* 中所述。在大多数情况下，CMake 还会自动设置必要的编译器和链接器标志，以便工具与 sysroot 一起工作。

如果构建产物不应直接安装到 sysroot 中，可以设置 `CMAKE_STAGING_PREFIX` 变量以提供替代的安装路径。通常在以下情况时需要这样做：sysroot 应保持干净或当它被挂载为只读时。请注意，`CMAKE_STAGING_PREFIX` 设置不会将该目录添加到 `CMAKE_SYSTEM_PREFIX_PATH`，因此，只有当工具链中的 `CMAKE_FIND_ROOT_PATH_MODE_PACKAGE` 变量设置为 `BOTH` 或 `NEVER` 时，暂存目录中安装的内容才能通过 `find_package()` 找到。

定义目标系统并设置工具链配置、sysroot 和暂存目录通常是进行交叉编译所需的所有内容。两个例外是针对 Android 和 Apple 的 iOS、tvOS 或 watchOS 进行交叉编译。

## 针对 Android 进行交叉编译

过去，Android 的 NDK 与不同 CMake 版本之间的兼容性有时关系并不顺畅，因为 NDK 的新版本往往不再以与以前版本相同的方式与 CMake 协作。然而，从 r23 版本开始，这一情况得到了极大的改善，因为 Android NDK 现在使用 CMake 内部对工具链的支持。结合 CMake 3.21 或更高版本，为 Android 构建变得相对方便，因此推荐使用这些或更新的版本。关于 Android NDK 与 CMake 集成的官方文档可以在此处找到：[`developer.android.com/ndk/guides/cmake`](https://developer.android.com/ndk/guides/cmake)。

从 r23 版本开始，NDK 提供了自己的 CMake 工具链文件，位于`<NDK_ROOT>/build/cmake/android.toolchain.cmake`，可以像任何常规的工具链文件一样使用。NDK 还包括所有必要的工具，以支持基于 Clang 的工具链，因此通常不需要定义其他工具。要控制目标平台，应通过命令行或使用 CMake 预设传递以下 CMake 变量：

+   `ANDROID_ABI`：指定`armeabi-v7a`、`arm64-v8a`、`x86`和`x86_64`。在为 Android 进行交叉编译时，这个变量应该始终设置。

+   `ANDROID_ARM_NEON`：为`armeabi-v7a`启用 NEON 支持。该变量不会影响其他 ABI 版本。使用 r21 版本以上的 NDK 时，默认启用 NEON 支持，通常不需要禁用它。

+   `ANDROID_ARM_MODE`：指定是否为`armeabi-v7a`生成 ARM 或 Thumb 指令。有效值为`thumb`或`arm`。该变量不会影响其他 ABI 版本。

+   `ANDROID_LD`：决定使用默认的链接器还是来自`llvm`的实验性`lld`。有效的值为`default`或`lld`，但由于`lld`处于实验阶段，这个变量通常在生产构建中被省略。

+   `ANDROID_PLATFORM`：指定最低的`$API_LEVEL`，`android-$API_LEVEL`，或`android-$API_LETTER`格式，其中`$API_LEVEL`是一个数字，`$API_LETTER`是平台的版本代码。`ANDROID_NATIVE_API_LEVEL`是该变量的别名。虽然设置 API 级别并非严格必要，但通常会进行设置。

+   `ANDROID_STL`：指定使用哪种`c++_static`（默认值）、`c++_shared`、`none`或`system`。现代 C++支持需要使用`c++_shared`或`c++_static`。`system`库仅提供`new`和`delete`以及 C 库头文件的 C++封装，而`none`则完全不提供 STL 支持。

调用 CMake 来配置 Android 构建的命令可能如下所示：

```cpp
cmake -S . -B build --toolchain <NDK_DIR>/build/cmake/android
  .toolchain.cmake -DANDROID_ABI=armeabi-v7a -DANDROID_PLATFORM=23
```

这个调用将指定需要 API 级别 23 或更高的构建，这对应于 Android 6.0 或更高版本的 32 位 ARM **中央处理** **单元** (**CPU**)。

使用 NDK 提供的工具链的替代方案是将 CMake 指向 Android NDK 的位置，这对于 r23 版本之后的 NDK 是推荐的方式。然后，目标平台的配置通过相应的 CMake 变量进行。通过将`CMAKE_SYSTEM_NAME`变量设置为`android`，并将`CMAKE_ANDROID_NDK`变量设置为 Android NDK 的位置，CMake 会被告知使用 NDK。这可以通过命令行或在工具链文件中完成。或者，如果设置了`ANDROID_NDK_ROOT`或`ANDROID_NDK` *环境变量*，它们将被用作`CMAKE_ANDROID_NDK`的值。

当以这种方式使用 NDK 时，配置是通过定义变量来实现的，而不是直接调用 NDK 工具链文件时所用的`CMAKE_`等效变量，如下所示：

+   `CMAKE_ANDROID_API`或`CMAKE_SYSTEM_VERSION`用于指定要构建的最低 API 级别

+   `CMAKE_ANDROID_ARCH_ABI`用于指示要使用的 ABI 模式

+   `CMAKE_ANDROID_STL_TYPE`指定要使用的 STL

配置 CMake 与 Android NDK 的示例工具链文件可能如下所示：

```cpp
set(CMAKE_SYSTEM_NAME Android)
set(CMAKE_SYSTEM_VERSION 21)
set(CMAKE_ANDROID_ARCH_ABI arm64-v8a)
set(CMAKE_ANDROID_NDK /path/to/the/android-ndk-r23b)
set(CMAKE_ANDROID_STL_TYPE c++_static)
```

当使用 Visual Studio 生成器为 Android 进行交叉编译时，CMake 要求使用*NVIDIA Nsight Tegra Visual Studio Edition*或*Visual Studio for Android 工具*，它们使用 Android NDK。使用 Visual Studio 构建 Android 二进制文件时，可以通过将`CMAKE_ANDROID_NDK`变量设置为 NDK 的位置，利用 CMake 的内置 Android NDK 支持。

随着 NDK 的最近版本和 3.20 及更高版本的 CMake，Android 的本地代码交叉编译变得更加简单。交叉编译的另一个特殊情况是当目标是 Apple 的 iOS、tvOS 或 watchOS 时。

## 为 iOS、tvOS 或 watchOS 进行交叉编译

推荐的为 Apple 的 iPhone、Apple TV 或 Apple 手表进行交叉编译的方式是使用 Xcode 生成器。苹果对用于这些设备构建应用的工具有相当严格的限制，因此需要使用 macOS 或运行 macOS 的**虚拟机**（**VM**）。虽然使用 Makefiles 或 Ninja 文件也是可能的，但它们需要更深入的苹果生态系统知识才能正确配置。

为这些设备进行交叉编译时，需要使用 Apple 设备的 SDK，并将`CMAKE_SYSTEM_NAME`变量设置为`iOS`、`tvOS`或`watchOS`，如下所示：

```cpp
cmake -S <SourceDir> -B <BuildDir> -G Xcode -DCMAKE_SYSTEM_NAME=iOS
```

对于合理现代的 SDK 和 CMake 版本为 3.14 或更高版本时，通常这就是所需要的所有配置。默认情况下，系统上可用的最新设备 SDK 将被使用，但如果需要，可以通过将`CMAKE_OSX_SYSROOT`变量设置为 SDK 路径来选择不同的 SDK。如果需要，还可以通过`CMAKE_OSX_DEPLOYMENT_TARGET`变量指定最低目标平台版本。

在为 iPhone、Apple TV 或 Apple Watch 进行交叉编译时，目标可以是实际设备，也可以是随不同 SDK 提供的设备模拟器。然而，Xcode 内置支持在构建过程中切换目标，因此 CMake 不需要运行两次。如果选择了 Xcode 生成器，CMake 会内部使用 `xcodebuild` 命令行工具，该工具支持 `-sdk` 选项来选择所需的 SDK。在通过 CMake 构建时，可以像这样传递此选项：

```cpp
cmake -build <BuildDir> -- -sdk <sdk>
```

这将把指定的 `-sdk` 选项传递给 `xcodebuild`。允许的值包括 iOS 的 `iphoneos` 或 `iphonesimulator`，Apple TV 设备的 `appletvos` 或 `appletvsimulator`，以及 Apple Watch 的 `watchos` 或 `watchsimulator`。

Apple 嵌入式平台要求对某些构建产物进行强制签名。对于 Xcode 生成器，开发团队 `CMAKE_XCODE_ATTRIBUTE_DEVELOPMENT_TEAM` 缓存变量。

在为 Apple 嵌入式设备构建时，模拟器非常有用，可以在无需每次都将代码部署到设备上的情况下进行测试。在这种情况下，测试最好通过 Xcode 或 `xcodebuild` 本身来完成，但对于其他平台，交叉编译的代码可以直接通过 CMake 和 CTest 进行测试。

# 测试交叉编译的二进制文件

能够轻松地为不同架构交叉编译二进制文件，为开发者的工作流程带来了极大的便利，但通常这些工作流程不仅仅局限于构建二进制文件，还包括运行测试。如果软件也可以在主机工具链上编译，并且测试足够通用，那么在主机上运行测试可能是测试软件的最简单方式，尽管这可能会在切换工具链和频繁重建时浪费一些时间。如果这不可行或过于耗时，当然可以选择在实际目标硬件上运行测试，但这取决于硬件的可用性和在硬件上设置测试的工作量，这可能会变得相当繁琐。因此，通常可行的折中方法是，如果有模拟器可用，在目标平台的模拟器中运行测试。

要定义用于运行测试的仿真器，使用`CROSSCOMPILING_EMULATOR`目标属性。它可以为单个目标设置，也可以通过设置`CMAKE_CROSSCOMPILING_EMULATOR`缓存变量来全局设置，该变量包含一个用分号分隔的命令和参数列表，用于运行仿真器。如果全局设置，则该命令将被添加到`add_test()`、`add_custom_command()`和`add_custom_target()`中指定的所有命令之前，并且它将用于运行任何由`try_run()`命令生成的可执行文件。这意味着所有用于构建的自定义命令也必须能够在仿真器中访问并运行。`CROSSCOMPILING_EMULATOR`属性不一定必须是一个实际的仿真器——它可以是任何任意程序，例如一个将二进制文件复制到目标机器并在那里执行的脚本。

设置`CMAKE_CROSSCOMPILING_EMULATOR`应该通过工具链文件、命令行或配置的前缀进行。一个用于交叉编译 C++代码到 ARM 的工具链文件示例如下，它使用流行的开源仿真器*QEMU*来运行测试：

```cpp
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)
set(CMAKE_SYSROOT /path/to/arm/sysroot/)
set(CMAKE_CXX_COMPILER /usr/bin/clang++)
set(CMAKE_CXX_COMPILER_TARGET arm-linux-gnueabihf)
set(CMAKE_CROSSCOMPILING_EMULATOR "qemu-arm;-L;${CMAKE_SYSROOT}")
```

除了设置目标系统和工具链的交叉编译信息外，示例中的最后一行将`emulator`命令设置为`qemu-arm -L /path/to/arm/sysroot`。假设一个`CMakeLists.txt`文件中包含如下定义的测试：

```cpp
add_test(NAME exampleTest COMMAND exampleExe)
```

当运行 CTest 时，不是直接运行`exampleExe`，而是将`test`命令转换为如下形式：

```cpp
qemu-arm "-L" "/path/to/arm/sysroot/" "/path/to/build-dir/
  exampleExe"
```

在仿真器中运行测试可以显著加速开发人员的工作流程，因为它可能消除了在主机工具链和目标工具链之间切换的需要，并且不需要将构建产物移动到目标硬件进行每个表面测试。像这样的仿真器也非常适合**持续集成**（**CI**）构建，因为在真实的目标硬件上构建可能会很困难。

有关`CMAKE_CROSSCOMPILING_EMULATOR`的一个技巧是，它也可以用来临时将测试包装在诊断工具中，例如*valgrind*或类似的诊断工具。由于运行指定的仿真器可执行文件并不依赖于`CMAKE_CROSSCOMPILING`变量（该变量指示一个项目是否是交叉编译的），因此使用这个变通方法的一个常见陷阱是，设置`CMAKE_CROSSCOMPILING_EMULATOR`变量会影响`try_run()`命令，该命令通常用于测试工具链或任何依赖项是否支持某些功能，并且由于诊断工具可能导致编译器测试失败，因此可能需要在已经缓存的构建上运行它，其中`try_run()`的任何结果已经被缓存。因此，使用`CMAKE_CROSSCOMPILING_EMULATOR`变量运行诊断工具不应永久进行，而应在特定的开发情况下使用，例如在寻找缺陷时。

在本节中，我们提到过 CMake 的 `try_run()` 命令，它与密切相关的 `try_compile()` 命令一起，用于检查编译器或工具链中某些功能的可用性。在下一节中，我们将更详细地探讨这两个命令以及功能测试工具链。

## 测试工具链支持的功能

当 CMake 在项目树上首次运行时，它会执行各种编译器和语言功能的测试。每次调用 `project()` 或 `enable_language()` 都会重新触发测试，但测试结果可能已经从之前的运行中缓存。缓存也是为什么在现有构建中切换工具链不推荐的原因。

正如我们将在本节中看到的，CMake 可以开箱即检查许多功能。大多数检查将内部使用 `try_compile()` 命令来执行这些测试。该命令本质上使用检测到的或由用户提供的工具链构建一个小的二进制文件。所有相关的全局变量，如 `CMAKE_<LANG>_FLAGS`，都将传递给 `try_compile()`。

与 `try_complie()` 密切相关的是 `try_run()` 命令，它内部调用 `try_compile()`，如果成功，它将尝试运行程序。对于常规的编译器检查，不使用 `try_run()`，任何调用它的地方通常都在项目中定义。

为了编写自定义检查，建议使用 `CheckSourceCompiles` 或 `CheckSourceRuns` 模块，而不是直接调用 `try_compile()` 和 `try_run()`，这两个模块和相应的函数 `check_source_compiles()` 和 `check_source_runs()` 命令自 CMake 3.19 版本以来就已可用。在大多数情况下，它们足以提供必要的信息，而无需更复杂地处理 `try_compile()` 或 `try_run()`。这两个命令的签名非常相似，如下所示：

```cpp
check_source_compiles(<lang> <code> <resultVar>
    [FAIL_REGEX <regex1> [<regex2>...]]  [SRC_EXT <extension>])
check_source_runs(<lang> <code> <resultVar>
[SRC_EXT <extension>])
```

`<lang>` 参数指定 CMake 支持的语言之一，如 C 或 C++。`<code>` 是作为字符串链接为可执行文件的代码，因此它必须包含一个 `main()` 函数。编译的结果将作为布尔值存储在 `<resultVar>` 缓存变量中。如果为 `check_source_compiles` 提供了 `FAIL_REGEX`，则将检查编译输出是否符合提供的表达式。代码将保存在具有与所选语言匹配的扩展名的临时文件中；如果文件的扩展名与默认值不同，则可以通过 `SRC_EXT` 选项指定。

还有语言特定版本的模块，称为 `Check<LANG>SourceCompiles` 和 `Check<LANG>SourceRuns`，它们提供相应的命令，如以下示例所示：

```cpp
include(CheckCSourceCompiles)
check_c_source_compiles(code resultVar
  [FAIL_REGEX regexes...]
)
include(CheckCXXSourceCompiles)
check_cxx_source_compiles(code resultVar
  [FAIL_REGEX regexes...]
)
```

假设有一个 C++ 项目，它可能使用标准库的原子功能，或者如果不支持该功能，则回退到其他实现。针对这个功能的编译器检查可能如下所示：

```cpp
include(CheckSourceCompiles)
check_source_compiles(CXX "
#include <atomic>
int main(){
    std::atomic<unsigned int> x;
    x.fetch_add(1);
    x.fetch_sub(1);
}" HAS_STD_ATOMIC)
```

在包含该模块后，`check_source_compiles()` 函数会与一个使用待检查功能的小程序一起调用。如果代码成功编译，`HAS_STD_ATOMIC` 将被设置为 `true`；否则，将被设置为 `false`。该测试会在项目配置期间执行，并打印类似如下的状态信息：

```cpp
[cmake] -- Performing Test HAS_STD_ATOMIC
[cmake] -- Performing Test HAS_STD_ATOMIC - Success
```

结果会被缓存，以便后续运行 CMake 时不会再次执行该测试。在很多情况下，检查程序是否编译已经能提供足够的信息，表明工具链的某些特性，但有时需要运行底层程序以获取所需的信息。为此，`check_source_runs()` 类似于 `check_source_compiles()`。`check_source_runs()` 的一个注意事项是，如果设置了 `CMAKE_CROSSCOMPILING` 但未设置模拟器命令，那么测试将只编译测试程序，而不会运行，除非设置了 `CMAKE_CROSSCOMPILING_EMULATOR`。

有许多以 `CMAKE_REQUIRED_` 开头的变量可以控制检查如何编译代码。请注意，这些变量缺少特定语言的部分，如果在进行跨语言测试时需要特别小心。以下是一些这些变量的解释：

+   `CMAKE_REQUIRED_FLAGS` 用于在 `CMAKE_<LANG>_FLAGS` 或 `CMAKE_<LANG>_FLAGS_<CONFIG>` 变量中指定的任何标志之后，向编译器传递附加标志。

+   `CMAKE_REQUIRED_DEFINITIONS` 指定了多个编译器定义，形式为 `-DFOO=bar`。

+   `CMAKE_REQUIRED_INCLUDES` 指定了一个目录列表，用于搜索额外的头文件。

+   `CMAKE_REQUIRED_LIBRARIES` 指定在链接程序时要添加的库列表。这些可以是库的文件名或导入的 CMake 目标。

+   `CMAKE_REQUIRED_LINK_OPTIONS` 是一个附加链接器标志的列表。

+   `CMAKE_REQUIRED_QUIET` 可以设置为 `true`，以抑制检查时的任何状态信息。

在需要将检查彼此隔离的情况下，`CMakePushCheckState` 模块提供了 `cmake_push_check_state()`、`cmake_pop_check_state()` 和 `cmake_reset_check_state()` 函数，用于存储配置、恢复先前的配置和重置配置，以下例子演示了这一点：

```cpp
include(CMakePushCheckState)
cmake_push_check_state()
# Push the state and clean it to start with a clean check state
cmake_reset_check_state()
include(CheckCompilerFlag)
check_compiler_flag(CXX -Wall WALL_FLAG_SUPPORTED)
if(WALL_FLAG_SUPPORTED)
    set(CMAKE_REQUIRED_FLAGS -Wall)
    # Preserve -Wall and add more things for extra checks
    cmake_push_check_state()
        set(CMAKE_REQUIRED_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/include)
        include(CheckSymbolExists)
        check_symbol_exists(hello "hello.hpp" HAVE_HELLO_SYMBOL)
    cmake_pop_check_state()
endif()
# restore all CMAKE_REQUIRED_VARIABLEs to original state
cmake_pop_check_state()
```

用于检查编译或运行测试程序的命令是更复杂的 `try_compile()` 和 `try_run()` 命令。虽然它们可以使用，但主要用于内部，因此我们这里不做解释，而是参考命令的官方文档。

通过编译和运行程序来检查编译器特性是一种非常灵活的方法，用于检查工具链特性。有些检查非常常见，CMake 提供了专门的模块和函数来执行这些检查。

## 工具链和语言特性的常见检查

对于一些最常见的功能检查，例如检查编译器标志是否支持或头文件是否存在，CMake 提供了方便的模块。从 CMake 3.19 版本开始，提供了通用模块，可以将语言作为参数，但相应的`Check<LANG>...`特定语言模块仍然可以使用。

一个非常基础的测试，用于检查某个语言的编译器是否可用，可以通过`CheckLanguage`模块来完成。如果未设置`CMAKE_<LANG>_COMPILER`变量，它可以用来检查某个语言的编译器是否可用。例如，检查 Fortran 是否可用的示例如下：

```cpp
include(CheckLanguage)
check_language(Fortran)
if(CMAKE_Fortran_COMPILER)
   enable_language(Fortran)
else()
   message(STATUS "No Fortran support")
endif()
```

如果检查成功，则会设置相应的`CMAKE_<LANG>_COMPILER`变量。如果在检查之前该变量已设置，则不会产生任何影响。

`CheckCompilerFlag`提供了`check_compiler_flag()`函数，用于检查当前编译器是否支持某个标志。在内部，会编译一个非常简单的程序，并解析输出以获取诊断信息。该检查假设`CMAKE_<LANG>_FLAGS`中已存在的任何编译器标志都能成功运行；否则，`check_compiler_flag()`函数将始终失败。以下示例检查 C++编译器是否支持`-Wall`标志：

```cpp
include(CheckCompilerFlag)
check_compiler_flag(CXX -Wall WALL_FLAG_SUPPORTED)
```

如果`-Wall`标志被支持，则`WALL_FLAG_SUPPORTED`缓存变量将为`true`；否则为`false`。

用于检查链接器标志的相应模块叫做`CheckLinkerFlag`，其工作方式与检查编译器标志类似，但链接器标志不会直接传递给链接器。由于链接器通常是通过编译器调用的，因此传递给链接器的额外标志可以使用如`-Wl`或`-Xlinker`等前缀，告诉编译器将该标志传递过去。由于该标志是编译器特定的，CMake 提供了`LINKER:`前缀来自动替换命令。例如，要向链接器传递生成执行时间和内存消耗统计信息的标志，可以使用以下命令：

```cpp
include(CheckLinkerFlag)
check_linker_flag(CXX LINKER:-stats LINKER_STATS_FLAG_SUPPORTED)
```

如果链接器支持`-stats`标志，则`LINKER_STATS_FLAG_SUPPORTED`变量将为`true`。

其他有用的模块用于检查各种内容，包括`CheckLibraryExists`、`CheckIncludeFile`和`CheckIncludeFileCXX`模块，用于检查某个库或包含文件是否存在于某些位置。

CMake 还提供了更多详细的检查，可能非常特定于某个项目——例如，`CheckSymbolExists`和`CheckSymbolExistsCXX`模块检查某个符号是否存在，无论它是作为预处理器定义、变量还是函数。`CheckStructHasMember`将检查结构体是否具有某个成员，而`CheckTypeSize`可以检查非用户类型的大小，并使用`CheckPrototypeDefinition`检查 C 和 C++函数原型的定义。

正如我们所见，CMake 提供了很多检查，随着 CMake 的发展，可用的检查列表可能会不断增加。虽然在某些情况下检查是有用的，但我们应该小心不要让测试的数量过多。检查的数量和复杂度将对配置步骤的速度产生很大影响，同时有时并不会带来太多好处。在一个项目中有很多检查，也可能意味着该项目存在不必要的复杂性。

# 总结

CMake 对交叉编译的广泛支持是其显著特点之一。在本章中，我们探讨了如何定义一个用于交叉编译的工具链文件，以及如何使用 sysroot 来使用不同目标平台的库。交叉编译的一个特殊案例是 Android 和苹果移动设备，它们依赖于各自的 SDK。通过简要介绍使用模拟器或仿真器测试其他平台，现在你已经掌握了所有必要的信息，可以开始为各种目标平台构建优质软件。

本章的最后部分讨论了测试工具链某些特性的高级话题。虽然大多数项目不需要关注这些细节，但了解这些内容依然很有用。

下一章将讨论如何让 CMake 代码在多个项目之间可重用，而不需要一遍又一遍地重写所有内容。

# 问题

1.  工具链文件如何传递给 CMake？

1.  通常在交叉编译的工具链文件中定义了什么？

1.  在 sysroot 的上下文中，什么是中间目录？

1.  如何将模拟器传递给 CMake 进行测试？

1.  什么触发了编译器特性的检测？

1.  如何存储和恢复编译器检查的配置上下文？

1.  `CMAKE_CROSSCOMPILING` 变量对编译器检查有什么影响？

1.  为什么在切换工具链时应该完全清除构建目录，而不仅仅是删除缓存？

# 答案

1.  工具链文件可以通过 `--toolchain` 命令行标志、`CMAKE_TOOLCHAIN_FILE` 变量，或者通过 CMake 预设中的 `toolchainFile` 选项传递。

1.  通常，交叉编译的工具链文件中会做以下几件事：

    1.  定义目标系统和架构

    1.  提供构建软件所需的任何工具的路径

    1.  为编译器和链接器设置默认标志

    1.  指定 sysroot 和可能的任何中间目录（如果是交叉编译的话）

    1.  为 CMake 的 `find_` 命令设置搜索顺序的提示

1.  中间目录通过 `CMAKE_STAGING_PREFIX` 变量设置，作为安装任何已构建的工件的地方，如果 sysroot 不应被修改。

1.  模拟器命令作为分号分隔的列表传递给 `CMAKE_CROSSCOMPILING_EMULATOR` 变量。

1.  在项目中对 `project()` 或 `enable_language()` 的任何调用都会触发特性检测。

1.  编译器检查的配置上下文可以通过`cmake_push_check_state()`存储，并通过`cmake_pop_check_state()`恢复到先前的状态。

1.  如果设置了`CMAKE_CROSSCOMPILING`，任何对`try_run()`的调用将会编译测试但不会运行，除非设置了模拟器命令。

1.  构建目录应该完全清理，因为仅删除缓存时，编译器检查的临时产物可能不会正确重建。
