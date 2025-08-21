# 第四章：为 FetchContent 创建库

在*第三章*《使用 FetchContent 处理外部依赖》中，我们详细了解了如何作为应用程序开发者使用 `FetchContent`。这是非常有用的，如果你不打算创建自己的库，那么这些知识会对你大有帮助。然而，如果你对创建库以在多个项目间共享（或者更好的是，与更广泛的开源社区共享）充满兴趣，那么本章将适合你。

在本章中，我们将介绍用于创建库的 CMake 命令，并通过 `FetchContent` 使其易于访问。你将在这里学到的技能不仅对你的库有帮助，还可以应用到其他不使用 CMake 的项目中。根据库的大小和复杂性，通常只需几个命令就能为库添加 `FetchContent` 支持。

在本章中，我们将讨论以下主要主题：

+   使库兼容 FetchContent

+   将生命游戏移到库中

+   将生命游戏做成共享库

+   最终的跨平台补充

+   接口库

# 技术要求

为了跟上进度，请确保你已经满足*第一章*《入门》的要求。包括以下内容：

+   一台运行最新 **操作系统**（**OS**）的 Windows、Mac 或 Linux 机器

+   一个可工作的 C/C++ 编译器（如果你还没有的话，建议使用系统默认的编译器，适用于每个平台）

本章中的代码示例可以在 https://github.com/PacktPublishing/Minimal-CMake 找到。

# 使库兼容 FetchContent

回到我们正在进行的项目，让我们从识别一块可以重用的代码开始：`array`。我们将把这个功能提取到一个独立的库中，以便从主应用程序中使用，并且将来可能在其他项目中重用（或与其他开发者共享，供他们尝试）。

## 项目结构

在我们查看 `CMakeLists.txt` 文件之前，先对项目结构做一些小的调整，以确保我们的库遵循常见的惯例。这些调整并非严格必要（我们在*第三章*《使用 FetchContent 处理外部依赖》中包含的库（`timer_lib` 和 `as-c-math`）并未遵循这些指南），但了解这些惯例是有用的，并且它们将帮助我们在项目不断发展时保持整洁和有序。

从我们在*第二章*《你好，CMake！》和*第三章*《使用 FetchContent 处理外部依赖》中看到的 `array/` 文件夹开始，结构如下：

```cpp
.
├── CMakeLists.txt
├── array
│   ├── array.c
│   └── array.h
├── build
│   └── ...
└── main.c
```

为了支持重用，我们将把`array.h`和`array.c`移到我们“生命游戏”应用程序之外的新文件夹中（如果你在跟随教程，请在`minimal-cmake`仓库之外创建一个名为`minimal-cmake-array`的新文件夹，并将`array.h`和`array.c`复制到接下来展示的位置）。

为了使一切保持自包含在*Minimal CMake*书籍仓库中（[`github.com/PacktPublishing/Minimal-CMake`](https://github.com/PacktPublishing/Minimal-CMake)），我们暂时将内容移至`ch4/part-1/lib/array`（可以将其视为顶级 CMake 项目的同义词）。

结构如下：

```cpp
.
├── CMakeLists.txt
├── build
│   └── ...
├── include
│   └── minimal-cmake
│      └── array.h
└── src
   └── array.c
```

请注意引入了两个新目录，`include`和`src`。这些名称在开源生态系统中已经被广泛采用（为什么`include`和`source`或`inc`和`src`没有更常见，可能是历史上的偶然结果）。根据惯例，`include`文件夹用于公共头文件（那些需要被客户端包含的头文件）；任何仅在库内部使用的头文件（私有头文件）应保存在`src`文件夹中，与源文件本身一起。

另一种可能性是将`array.h`和`array.c`保留在根目录中，如下所示：

```cpp
├── CMakeLists.txt
├── build
    └── ...
└── array.h
└── array.c
```

这种方式对于小型库来说无疑是可行的，但也有一些缺点。如果我们想添加更多的源文件，它们可能会使根目录变得杂乱，并增加导航的难度。将实现细节保存在`src`文件夹下，可以给库的用户一个清晰的信号，让他们将注意力集中在其他地方。

创建一个名为项目名称的`include`文件夹及子目录的一个优势是，可以使消费应用程序或库中的`#include`指令更加清晰。

以下方式会更加有帮助：

```cpp
#include <minimal-cmake/array.h>
```

将前面的代码与以下代码进行对比，后者更难理解：

```cpp
#include <array.h>
```

使用第一种方法，可以明确知道依赖项的来源。这还减少了与其他库发生命名冲突的可能性（这种方法属于*代码卫生*的范畴）。

另一种选择是为库文件添加前缀。例如，我们本可以选择将`array.h`重命名为`mc-array.h`，或`minimal-cmake-array.h`，并省略子文件夹。为文件、函数和类型名称（例如，`mc_array_push`）添加项目标识符作为前缀，也是避免与其他库命名冲突的好做法。对于 C++，命名空间是首选的机制，但在 C 语言中，我们必须依赖显式的函数和类型前缀。这也是我们在数组实现中将采用的方法。

在这里展示的示例中，`src` 文件夹没有任何子文件夹。这是随意的，具体如何安排由库的作者决定。对于一个较小的库来说，`src` 下没有层级的扁平结构可能是可以的。而对于较大的库，我们可能会决定将某些文件分组以便更好地组织。由于 `src` 文件夹下的所有内容都可以视为库的私有部分，因此 `src` 下的结构不应影响库的使用者，所以它可以是你喜欢的任何结构。

关于我们的 C 实现有一个简短的说明，我们可能希望将这个库与未来的 C++ 应用程序一起使用。为了适应这种需求，我们需要使用 `extern "C"` 来包装或注解所有函数，确保当我们用 C++ 编译这个库时，*名称修饰*（C++ 中支持函数重载的过程）不会启动（在 C 中，你不能重载函数，符号名称保持不变）。我们还需要在编译为普通 C 代码时忽略 `extern "C"`。为了实现这一点，我们可以使用 `__cplusplus` 宏来检查我们是否在编译 C++ 代码（`__cplusplus` 只有在使用 C++ 时才会定义）。将这一切结合起来，我们得到了如下代码：

```cpp
#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
// our implementation
#ifdef __cplusplus
}
#endif // __cplusplus
```

最后，采用之前讨论的文件夹结构将使得安装时的工作变得更轻松。实际上，对于较小的库来说，这可能被认为是一种过度工程，特别是如果你从不打算安装这些库的话，但我们还是为了完整性考虑介绍了这一部分，因为这是我们后面需要的内容。

## `CMakeLists.txt` 文件

在设定好文件夹结构后，我们可以查看新 `array` 库的 `CMakeLists.txt` 文件。此处包含了完整的 `CMakeLists.txt` 文件。我们将像之前的章节那样，逐行分析：

```cpp
cmake_minimum_required(VERSION 3.28)
project(mc-array LANGUAGES C)
add_library(${PROJECT_NAME})
target_sources(${PROJECT_NAME} PRIVATE src/array.c)
target_include_directories(
  ${PROJECT_NAME} PUBLIC $<BUILD_LOCAL_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_features(${PROJECT_NAME} PRIVATE c_std_17)
```

让我们略过前两行，它们与之前相同：

```cpp
cmake_minimum_required(VERSION 3.28)
project(mc-array LANGUAGES C)
```

这里是强制要求的 `cmake_minimum_required` 命令，后面紧跟着同样重要的 `project` 命令。唯一的区别是，我们为我们的库命名为与其功能相匹配的名称（一个数组接口），并且我们还包含了我们在项目中打算使用的前缀（在这种情况下是 `mc`，代表*Minimal CMake*）。这可能有些过头，CMake 也提供了其他方法让你通过使用 `ALIAS` 来为库加上*命名空间*。我们将在后面的章节回到这个话题，但目前我们所做的已经足够了。

## 创建库

接下来，我们将介绍一条很久没见过的新命令：

```cpp
add_library(${PROJECT_NAME})
```

由于我们创建的是一个库，而不是一个应用程序，我们必须使用`add_library`命令而不是`add_executable`。默认情况下，CMake 会为我们创建一个静态库（对于静态库，内容将会被打包进我们的可执行文件并在编译时链接）。为了覆盖这个行为，在配置 CMake 项目时（运行`cmake -B build`），可以传递`-DBUILD_SHARED_LIBS=ON`来切换到构建共享库。为了确保在所有平台（Windows、macOS 和 Linux）上都能正常工作，我们需要做一些额外的工作，所以我们暂时不做处理。为了提供不同于默认的设置，可以在我们的`CMakeLists.txt`文件中添加一个选项，如下所示：

```cpp
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
```

更多关于`BUILD_SHARED_LIBS`选项的信息，请参见[`cmake.org/cmake/help/latest/variable/BUILD_SHARED_LIBS.html`](https://cmake.org/cmake/help/latest/variable/BUILD_SHARED_LIBS.html)。

为了硬编码静态或共享库，可以通过在库名后传递`STATIC`或`SHARED`来提供库类型给`add_library`。以下是一个示例：

```cpp
add_library(${PROJECT_NAME} STATIC can be a good approach. If you’re creating a library that will be built and installed separately from the main application (something we’ll cover in *Chapter 7*, *Adding Install Support for Your Libraries*), giving a user the flexibility to decide to use either static or shared is a nice feature. Unfortunately, BUILD_SHARED_LIBS doesn’t play nicely when composing multiple libraries using FetchContent. Luckily for us, there is a workaround that builds on the topics we’ve covered here. We’ll cover this a little later in the chapter.
			Next up, we have `target_sources`, which has been updated to reference the new location of `array.c`:

```

target_sources(${PROJECT_NAME} PRIVATE PRIVATE，这里作为`array.c`是实现细节，我们不希望（也不需要）它重新编译。唯一的区别是我们在新的位置引用它。

            剩下的新命令（我们在 *第三章* ，*使用 FetchContent 与外部依赖项* 中简要提到过，在查看依赖项链接时）是`target_include_directories`：

```cpp
target_include_directories(
  ${PROJECT_NAME} PUBLIC 
  $<BUILD_LOCAL_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
```

            这个命令告诉依赖项包含文件相对于的位置。我们直接在目标上设置它，并且希望这个属性对客户端或者库的用户可见，这就是为什么我们指定`PUBLIC`而不是`PRIVATE`的原因。

            生成器表达式

            看看之前提到的`target_include_directories`命令，它的第三行可能一开始看起来有点陌生。你看到的是 CMake 提供的一项功能，称为**生成器表达式**。如果我们暂时移除生成器表达式，命令看起来是这样的：

```cpp
target_include_directories(
  ${PROJECT_NAME} PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
```

            让我们回顾一下之前检查过的文件结构：

```cpp
.
├── include
    └── minimal-cmake
        └── array.h
```

            这样可以确保应用程序通过`#include <minimal-cmake/array.h>`来包含`array.h`。这非常棒，因为这意味着客户端不需要自己设置`include`目录；他们只需链接到目标，并自动继承这个属性。

            在你的项目的`README`文件中包含一个示例，要么是一个小应用程序，要么是一个代码片段，展示如何包含依赖项以及包含路径是什么，这是个不错的主意。用户虽然可以自己搞定，但你提供的信息越多，就越能让使用这个库变得简单，也能降低他们在使用过程中卡住的几率。

            让我们回到之前看到的生成器表达式：

```cpp
$<BUILD_LOCAL_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
```

            在最简单的形式下，结构为`$<condition:value>`。如果`condition`被设置（即存在），则提供`value`；否则，表达式的结果为空。生成器表达式有点像 C 或 C++中的三元操作符（`<condition> ? <true> : <false>`）。它本质上是一种简洁、声明式的方式，用来在`CMakeLists.txt`脚本中编写条件，而不需要依赖更冗长的`if`/`else`分支，这种分支采用的是更命令式的编程风格。

            使用生成器表达式时需要找到一个平衡点；它们可以方便并简化`CMakeLists.txt`文件，但如果过度使用，可能会让代码更难理解。要明智地使用它们，如果你认为使用显式的`if`/`else`语句更清晰，就应当选择这种方式。通过使用多个 CMake 变量将复杂的生成器表达式拆解开来也可以是一种有价值的方式，而不是试图将所有内容都写成一个单一的表达式。

            命令`cmake -B build`中，CMake 首先执行配置步骤，然后执行生成步骤。这时，生成器表达式会被求值，项目文件会被创建。如下所示，这是`cmake`命令的输出：

```cpp
-- Configuring done (8.7s)
-- Generating done (0.0s)
```

            使用生成器表达式可能会很困难，能够调试表达式的结果是非常有用的。不幸的是，普通的 CMake `message`语句无法与生成器表达式一起输出日志到控制台，因为它们的求值时间不同（配置时间与生成时间不同）。为了解决这个问题，可以通过以下方法将表达式的结果写入文件：

```cpp
file(GENERATE OUTPUT <filename> CONTENT "$<...>")
```

            运行`cmake -B build`时，这将把生成器表达式（`"$<...>"`）的结果写入指定的文件名（如果提供了相对路径，它将位于`build/`文件夹内）。然后可以检查文件的内容，确认结果是否符合预期。

            想要了解更多关于生成器表达式及其支持的多种变体，可以访问[`cmake.org/cmake/help/latest/manual/cmake-generator-expressions.7.html`](https://cmake.org/cmake/help/latest/manual/cmake-generator-expressions.7.html)。

            包含接口

            我们已经讨论了为什么指定`target_include_directories`很重要以及什么是生成器表达式，但没有解释为什么特别需要`BUILD_LOCAL_INTERFACE`。原因在于，这使得我们能够根据是否在构建库或在安装后使用它来使用不同的包含路径。安装对库来说很重要，这是我们将在*第七章*《*为你的库添加安装支持*》中详细讲解的内容，但现在，只需知道有这种替代方案即可。在库的`CMakeLists.txt`文件中，通常会看到类似这样的内容：

```cpp
target_include_directories(
  ${PROJECT_NAME} PUBLIC 
  $<BUILD_LOCAL_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
```

            根据上下文，目标将在以下情况下设置不同的包含路径：如果它依赖于并在同一构建树中构建（如`FetchContent`或`add_subdirectory`），或者安装到另一个位置并从那里依赖（称为导入目标）。安装库的包含文件通常与库本身的不同（开发者可能希望将包含层次结构扁平化，使库接口更易于使用）。通常在创建库时指定`BUILD_LOCAL_INTERFACE`是一个不错的主意。如果以后决定添加安装支持，可以再添加`INSTALL_INTERFACE`。通过明确这一点，您可以避免将来需要匹配构建和安装接口。

            BUILD_LOCAL_INTERFACE 与 BUILD_INTERFACE

            你可能会遇到`BUILD_INTERFACE`，除此之外还有`BUILD_LOCAL_INTERFACE`。`BUILD_LOCAL_INTERFACE`是一个较新的生成表达式（在 CMake `3.26`版本中添加），它仅在同一构建系统中的另一个目标使用时才会展开其内容，而`BUILD_INTERFACE`会在同一构建系统中的另一个目标使用时展开其内容，并且当属性通过`export`命令导出时也会展开。由于我们不打算从构建树中导出目标，因此我们选择了这两个命令中限制性更强的那个。

            最后，我们将编译特性设置为标准版本，以确保在不同编译器之间获得一致的行为：

```cpp
target_compile_features(${PROJECT_NAME} PRIVATE c_std_17)
```

            这就是我们通过`FetchContent`将我们的库提供给其他用户所需的一切。

            使用我们的库

            现在，我们可以更新应用程序的现有`CMakeLists.txt`文件，将新的数组库引入：

```cpp
...
FetchContent_Declare(
  minimal-cmake-array
  GIT_REPOSITORY https://github.com/PacktPublishing/Minimal-CMake.git
  GIT_TAG 2b5ca4e58a967b27674a62f22ece4f846bc0aa78
  SOURCE_SUBDIR ch4/part-1/lib/array) # look just in array folder
FetchContent_MakeAvailable(timer_lib as-c-math minimal-cmake-array)
target_link_libraries(
  ${PROJECT_NAME} PRIVATE timer_lib as-c-math CMakeLists.txt file to a new app folder, with the array library moving to a new lib folder. The folder structure now looks like this:

```

├── app

│   ├── CMakeLists.txt

│   └── main.c

└── lib

└── array

├── CMakeLists.txt

├── include

└── src

```cpp

			This means we need to run our CMake configure and build commands (`cmake -B build` and `cmake --build build`) from `part-<n>/app`, instead of `part-<n>` (you could also use the `-S` option and pass the source folder explicitly, as discussed in *Chapter 2*, *Hello, CMake!* if preferred).
			A complete example is presented in `ch4/part-1/app` to show how everything fits together. A small detail to note is the use of `SOURCE_SUBDIR` in the `FetchContent_Declare` command. This lets us specify a subdirectory in the repository as the root to use for `FetchContent`. As we’ve extracted our `array` type to a library in the *Minimal CMake* repository, we can treat that folder as the root of the CMake project (for completeness, the full repository will be downloaded, but only the files specified under `SOURCE_SUBDIR` will be used in the build).
			We can also use `SOURCE_DIR` and a relative path, which can be useful when we’re working on the library and application together. This would look like the following:

```

FetchContent_Declare(

minimal-cmake-array

SOURCE_DIR ../../lib/array)

```cpp

			This means any changes to `array` will immediately be reflected in the main application. Just remember to pick a commit for the library when you’re committing your changes to make it easier to go back to earlier points in your project history for reproducible builds.
			Moving Game of Life to a library
			We started by extracting the `array` type from our application as it was a simpler piece of functionality to start with. At this point, we’d like to pull out the core *Game of Life* logic to a separate library. We’re going to make it possible to build it as either a static or shared library, in preparation for potentially integrating it with other languages in the future. This will require us to provide an interface and move the functionality to separate files.
			To prepare for our *Game of Life* code being used as a shared library, we’ll keep the concrete implementation of the *Game of Life* board hidden and expose functionality through a series of functions. The interface looks as follows:

```

// 前向声明板

typedef struct mc_gol_board_t mc_gol_board_t;

// 生命周期

mc_gol_board_t* mc_gol_create_board(int32_t width, int32_t height);

void mc_gol_destroy_board(mc_gol_board_t* board);

// 处理

void mc_gol_update_board(mc_gol_board_t* board);

// 查询

int32_t mc_gol_board_width(const mc_gol_board_t* board);

int32_t mc_gol_board_height(const mc_gol_board_t* board);

bool mc_gol_board_cell(

const mc_gol_board_t* board, int32_t x, int32_t y);

// 变异

void mc_gol_set_board_cell(

mc_gol_board_t* board, int32_t x, int32_t y, bool alive);

```cpp

			This is a C-style interface where we forward declare the Game of Life board type (`mc_gol_board_t`) and provide create and destroy functions to manage the lifetime. By hiding concrete types, we make it easier to integrate our library with other languages in the future and avoid potential **application binary interface** (**ABI**) incompatibilities across different compilers (such as layout, padding, or alignment). Function interfaces also help with encapsulation and backward compatibility.
			With our interface defined, we can follow the same approach that we did with `array` and create a static library encapsulating our Game of Life implementation. If you review `ch4/part-2/lib/gol`, you’ll see the updated structure. We’ve also been able to move `as-c-math` and `mc-array` so that they’re private dependencies of the new *Game of Life* library (`mc-gol`) and remove them from the main app’s `CMakeLists.txt` file. To disambiguate the application and library, we’ll also rename our app to `minimal-cmake_game-of-life_console`.
			With this in place, we can focus on the changes necessary to make this a shared library.
			Making Game of Life a shared library
			We will start by working through the changes between `ch4/part-2` and `ch4/part-3` to see what updates are needed to make `mc_gol` a shared library. The focus will be `ch4/part-3/lib/gol/CMakeLists.txt`, but we’ll also need to update `ch4/part-3/lib/gol/include/minimal-cmake-gol/gol.h` and `ch4/part-3/app/CMakeLists.txt`.
			Visual Studio Code – Compare Active File With...
			This is a quick reminder to use the Visual Studio Code feature known as `ch4/part-2/lib/gol/CMakeLists.txt` and `ch4/part-3/lib/gol/CMakeLists.txt`). The `diff` view makes the changes clear without needing to switch back and forth.
			The first difference is the addition of a new `option` command for `mc-gol`:

```

option(MC_GOL_SHARED "启用共享库（动态链接）" OFF)

```cpp

			The CMake `option` command allows the library user to compile `mc-gol` as either `STATIC` or `SHARED` (it defaults to `OFF` to match the CMake default of static libraries). The `option` name is also prefixed with `MC_GOL` to help with readability and reduce the chance of name collisions in other projects.
			We’ve refrained from using `BUILD_SHARED_LIBS` in this case because using this would apply to all libraries we’re building (including `mc-array` and `as-c-math`). We would like those libraries to be compiled statically as normal and only allow `mc-gol` to be explicitly compiled as a shared library.
			If we were only building our library and linking to external dependencies that had already been built, `BUILD_SHARED_LIBS` would work well, but this isn’t what we want when composing libraries with `FetchContent`.
			To support only building `mc-gol` as `SHARED`, we need a little more logic before the `add_library` command:

```

set(MC_GOL_LIB_TYPE STATIC)

if(MC_GOL_SHARED)

set(MC_GOL_LIB_TYPE SHARED)

endif()

```cpp

			Here, we introduce a new CMake variable called `MC_GOL_LIB_TYPE`, which we default to `STATIC`. Only if the `MC_GOL_SHARED` option is turned on do we set it to `SHARED`. We then pass this CMake variable to the `add_library` command to decide the library type:

```

add_library(${PROJECT_NAME} ${MC_GOL_LIB_TYPE})

```cpp

			We’ll skip over the change to `target_include_directories` for now as it’s a side effect of what we’ll talk about next.
			Here, we’re focusing on making our library cross-platform. To ensure our shared library works consistently across macOS, Windows, and Linux, we need to take some extra steps to support this. With the preceding change, if we try to build and run our project on Windows with `MC_GOL_SHARED` set to `ON` (`cmake -B build -DMC_GOL_SHARED=ON`), our application will fail to link. This is because Windows requires symbols from a shared library (in our case, functions) to be explicitly exported; otherwise, they are hidden, and they’re only available internally to the library. This contrasts with macOS and Linux, where all symbols are usually exported by default.
			To work around this, we must explicitly annotate the functions we want to make available to other applications with special compiler directives. These are different across Windows and macOS/Linux (Visual Studio versus GCC/Clang). Fortunately, CMake provides an incredibly useful feature called `generate_export_header` that provides a cross-platform solution for us. To use it, add the following to your `CMakeLists.txt` file:

```

include(GenerateExportHeader)

generate_export_header(${PROJECT_NAME} BASE_NAME mc_gol)

```cpp

			First, we bring in the `GenerateExportHeader` module, which provides the `generate_export_header` command, and then we call it while providing the project name and a base name for the library (`mc_gol`). This will create a file called `mc_gol_export.h` in the `mc-gol` build folder.
			This briefly brings us back to the change to `target_include_directories` we skipped over earlier. To ensure our header (`gol.h`) can include `mc_gol_export.h`, we need to ensure it is added to the target’s include path. To achieve this, we’ll add `${CMAKE_CURRENT_BINARY_DIR}` to `target_include_directories`.
			This can be done in one of two ways. First, we can pass two generator expressions like so:

```

target_include_directories(

${PROJECT_NAME}

PUBLIC $<BUILD_LOCAL_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>

$<BUILD_LOCAL_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/>)

```cpp

			Alternatively, we can wrap the generator expression in quotes and pass the second directory as a list (separated by semicolons):

```

target_include_directories(

${PROJECT_NAME}

PUBLIC "$<BUILD_LOCAL_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/build/mc_gol_export.h，我们会看到几个宏已为我们生成。对我们而言，最重要的一个是 MC_GOL_EXPORT。按照我们当前在 macOS 或 Linux 上的设置，它目前不会展开任何内容（因为默认所有符号都是可见/公共的），但在 Windows 上，当构建共享库时，我们会看到已经生成了以下内容：

```cpp
#    ifdef mc_gol_EXPORTS
        /* We are building this library */
#      define MC_GOL_EXPORT __declspec(dllexport)
#    else
        /* We are using this library */
#      define MC_GOL_EXPORT __declspec(dllimport)
#    endif
```

            编译指令 `__declspec(dllexport)` 和 `__declspec(dllimport)` 是微软特有的。当构建共享库时，`__declspec(dllexport)` 用于使符号可供库外部使用，而在使用库时，必须存在 `__declspec(dllimport)` 来显示哪些符号正在被导入。利用 CMake 为我们生成这些宏非常方便；它保证无论我们为哪个平台构建，或者启用了哪些编译器设置，都会做出正确的处理。

            如果我们决定再次将 `mc-gol` 构建为静态库，那么 `MC_GOL_EXPORT` 将不会展开。构建静态版本库时，我们可以设置一个额外的 `#define`，在这种情况下是 `MC_GOL_STATIC_DEFINE`。我们可以这样定义：

```cpp
target_compile_definitions(
  ${PROJECT_NAME}
  PUBLIC $<$<NOT:$<BOOL:${MC_GOL_SHARED}>>:MC_GOL_STATIC_DEFINE, but only if we’re not building a shared library. This will guarantee that MC_GOL_EXPORT won’t be expanded when building as a static library (see ch4/part-5/lib/CMakeLists.txt for an example). This can be useful if you’re reusing a generated version of mc_gol_export.h that has MC_GOL_EXPORT set to something you don’t want. In our case, it’s not strictly necessary but it can be a good failsafe to keep in place.
			To learn more about `GenerateExportHeader`, you can read the full documentation, which is available at [`cmake.org/cmake/help/latest/module/GenerateExportHeader.html`](https://cmake.org/cmake/help/latest/module/GenerateExportHeader.html).
			With `mc_gol_export.h` created, and our `target_include_directories` command updated, all that remains is to annotate our symbols (in the case of `gol.h`, our functions) with `MC_GOL_EXPORT`. Here’s an example:

```

MC_GOL_EXPORT mc_gol_board_t* mc_gol_create_board(

int32_t width, int32_t height);

```cpp

			On Windows, when `mc_gol` is built, the macro is substituted with `__declspec(dllexport)`, and when it’s later used as a dependency from our application, `MC_GOL_EXPORT` is substituted with `__declspec(dllimport)`.
			Making things work on Windows
			We’re nearly there! The last change we need to make is to our application’s `CMakeLists.txt` file (`ch4/part-3/app/CMakeLists.txt`) to ensure things work correctly on Windows.
			Let’s configure and build our project with `MC_GOL_SHARED` set to `ON`, like so:

```

cmake -B build -DMC_GOL_SHARED=ON

cmake --build build

```cpp

			Assuming Visual Studio is picked as the default generator (it being a multi-config generator, our executable will end up in the `Debug/` folder unless a different config is provided), we can try to run our application with the following command:

```

./build/Debug/minimal-cmake_game-of-life_console.exe

```cpp

			The unwelcome news is this will fail on startup with the following error:

```

C:/Path/to/minimal-cmake/ch4/part-3/app/build/Debug/minimal-cmake_game-of-life_console.exe: 加载共享库时出错：?: 无法打开共享对象文件：没有这样的文件或目录

```cpp

			The reason for this is that our application cannot find `mc-gol.dll` to load. This has happened because, on Windows, an application will search for a shared library (called a `PATH` environment variable. We haven’t told our executable where to search for `mc-gol.dll` or moved the DLL next to our executable, so it can’t find it.
			To get things working, we could update the `PATH` variable from the terminal:

```

set PATH=C:\Path\to\minimal-cmake\ch4\part-3\app\build\_deps\minimal-cmake-gol-build\Debug;%PATH%

```cpp

			This, however, is a tedious manual step and deals with absolute paths (not exactly portable). A much better idea is just to copy or move the DLL to the same folder as the executable.
			There are two ways to do this in our example. The first is to update `RUNTIME_OUTPUT_DIRECTORY` of `mc_gol` to that of our current executable. In our application’s `CMakeLists.txt` file, we can add this line:

```

if(WIN32)

set_target_properties(

mc-gol 属性 RUNTIME_OUTPUT_DIRECTORY

${CMAKE_CURRENT_BINARY_DIR})

endif()

```cpp

			As we’re building `mc-gol` ourselves, we can set properties on it as if we’d added the library locally. The preceding command will ensure `mc-gol.dll` will be written directly to `build\Debug`, instead of `build\_deps\minimal-cmake-gol-build\Debug`. This command also handles single and multi-config generators correctly (if we were to switch to the Ninja single-config generator, `mc-gol.dll` would end up in the `build\` folder).
			As a brief aside, it’s worth mentioning that `RUNTIME_OUTPUT_DIRECTORY` refers to `.dll` files on Windows (as well as executable files), but on macOS and Linux, it is `LIBRARY_OUTPUT_DIRECTORY`, which refers to `.dylib` (macOS) and `.so` (Linux) shared library files. This can be a little counterintuitive and will be important a little later when we return to `ch4/part4`.
			The second way to copy `mc-gol.dll` to the same directory as our executable is to use a CMake custom command. Here is the one we’ll use:

```

if(WIN32)

add_custom_command(

TARGET ${PROJECT_NAME}

POST_BUILD

COMMAND

${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:mc-gol>

$<TARGET_FILE_DIR:${PROJECT_NAME}>

VERBATIM)

endif()

```cpp

			This sets up a custom command to run immediately after the build completes (`POST_BUILD`). The target the command is bound to is our application, and the command copies the target file (`$<TARGET_FILE:mc-gol>`) to the directory of our application’s target binary file (`$<TARGET_FILE_DIR:${PROJECT_NAME}>`). In this case, when `mc-gol.dll` is built, it is written to `build\_deps\minimal-cmake-gol-build\Debug\mc-gol.dll` first, after which it is copied to `build\Debug` once our application (`minimal-cmake_game-of-life_console`) has finished building.
			One advantage of this approach over using the `set_target_properties(... RUNTIME_OUTPUT_DIRECTORY` method is that this works for libraries outside the current build (for example, installed libraries found using `find_package`, something we’ll cover in *Chapter 6*, *Installing Dependencies and ExternalProject_Add*). This consistency is one reason to prefer this approach; however, it depends on the type of application you’re building. If you know the library will always be included in the main build using `FetchContent` or `add_subdirectory`, then sticking with setting `RUNTIME_OUTPUT_DIRECTORY` is a fine choice.
			Making things relocatable on macOS and Linux
			We spent a bit of time dealing with DLL loading issues on Windows, but both macOS and Linux also need some attention to work reliably across different locations. The reason we had to copy `mc-gol.dll` to the application folder on Windows was that our application wouldn’t start without it there. The good news is that on macOS and Linux, we don’t need to do that because when we build the project, our application will record the location of the shared library and know where to load it from.
			This works great until we decide to move our library to another location. Suppose we want to zip up the contents of our project and share it with a friend, or just check it runs on another machine. If we try this as-is, chances are you’ll see the following error:

```

dyld[10168]: 未加载库：@rpath/libmc-gol.dylib

原因：尝试了：'/path/to/minimal-cmake/ch4/part-3/app/build/_deps/minimal-cmake-gol-build/libmc-gol.dylib'（没有这个文件）

```cpp

			This is because the absolute path of where the library was found when it was built is baked into our application. This means we can move our application (`minimal-cmake_game-of-life_console`), but if we move `mc-gol.dylib` (macOS) or `mc-gol.so` (Linux), things will break. Fortunately, there is a straightforward way to solve this.
			What we’re going to rely on is changing the `RPATH` (runtime search path) variable of our executable to include `@loader_path` (on macOS) and `$ORIGIN` (on Linux). This is effectively a way to refer to the application wherever it is on the filesystem. What this means is that just like on Windows, our application will search for the shared library in the folder it’s running from, so we simply need to copy the shared library (`.dylib`/`.so`) to the application folder. We only need to do this when we want to distribute the application, and we can either use `set_target_properties(... LIBRARY_OUTPUT_DIRECTORY)` or rely on the same method we used to copy the Windows `.dll` file to the same folder.
			To change the `RPATH` variable, we can use the following CMake commands:

```

set_target_properties(

${PROJECT_NAME} 属性 BUILD_RPATH @loader_path) # 仅限 macOS

set_target_properties(

${PROJECT_NAME} 属性 BUILD_RPATH $ORIGIN) # 仅限 Linux

set_target_properties(

${PROJECT_NAME}

PROPERTIES

BUILD_RPATH

"$<$<PLATFORM_ID:Linux>:$ORIGIN>$<$<PLATFORM_ID:Darwin>:@loader_path>")分别为 macOS 和 Linux 设置 set_target_properties，然后使用生成器表达式来设置正确的 RPATH 值，以便根据平台进行调整（在此情况下不会在 Windows 上设置任何内容）。

            要检查 `RPATH` 的值，可以在 macOS 上使用 `otool` 工具或在 Linux 上使用 `readelf` 工具（这两个工具分别显示其平台的对象文件）。在 macOS 上使用 `otool -l minimal-cmake_game-of-life_console` 命令，以及在 Linux 上使用 `readelf -d minimal-cmake_game-of-life_console` 命令，将显示列出的值。

            以下是在 macOS 上使用 `otool` 的输出片段：

```cpp
Load command 16
          cmd LC_RPATH
      cmdsize 32
         readelf on Linux:

```

0x..001 (NEEDED)  共享库：[ld-linux-aarch64.so.1]

0x..01d @loader_path 和 $ORIGIN 出现如预期。

            要了解有关 `CMake` 中 `RPATH` 处理的更多信息，请访问 [`gitlab.kitware.com/cmake/community/-/wikis/doc/cmake/RPATH-handling`](https://gitlab.kitware.com/cmake/community/-/wikis/doc/cmake/RPATH-handling)。在配置共享库时，有许多不同的方法，我们只是初步探讨了一个可能的解决方案。这是一个可以继续探索的领域，具体取决于您将要创建的应用程序类型。在本书后面讨论安装库和打包项目时，我们一定会重新讨论这些主题。

            最终跨平台增强

            在结束之前，让我们来介绍一些小更新，以确保我们的库在不同平台上更为一致。我们可以使用现在熟悉的 `set_target_properties` 命令，仅对我们的库应用这些设置。

            前两个相关属性是 `C_VISIBILITY_PRESET` 和 `VISIBILITY_INLINES_HIDDEN`。我们将 `C_VISIBILITY_PRESET` 设置为 `hidden`，将 `VISIBILITY_INLINES_HIDDEN` 设置为 `ON`。这可以确保在 Windows 上的 Visual Studio 编译器（MSVC）和 macOS/Linux 上的 Clang/GCC 编译器之间，默认情况下，除非使用 `MC_GOL_EXPORT` 显式注释符号，否则它们将保持隐藏。这有助于防止不同平台之间的不兼容性。

            启用这些设置后，如果我们在 macOS 或 Linux 上像往常一样运行 `cmake -B build` 来重新生成我们的导出头文件，我们将看到以下内容：

```cpp
#    ifdef mc_gol_EXPORTS
        /* We are building this library */
#      define MC_GOL_EXPORT
__attribute__((visibility("default")))
#    else
        /* We are using this library */
#      define MC_GOL_EXPORT
__attribute__((visibility("default")))
#    endif
```

            这比看到以下内容要好：

```cpp
#    ifdef mc_gol_EXPORTS
        /* We are building this library */
#      define MC_GOL_EXPORT
#    else
        /* We are using this library */
#      define MC_GOL_EXPORT
#    endif
```

            启用这些设置后，如果我们尝试在 macOS 或 Linux 上使用尚未明确导出的符号（类型或函数），我们将会得到链接错误，就像在 Windows 上一样。如果我们正在开发跨平台库，建议尽可能保持行为在各个平台上的一致性。不自动导出所有符号默认有很好的理由，可以减少导出符号表的大小和整体二进制大小。

            接下来的两个属性是 `C_STANDARD_REQUIRED` 和 `C_EXTENSIONS`。我们将 `C_STANDARD_REQUIRED` 设置为 `ON`，将 `C_EXTENSIONS` 设置为 `OFF`。

            将 `C_STANDARD_REQUIRED` 设置为 `ON` 确保我们能够获取到在 `target_compile_features` 中使用 `c_std_17` 指定的最小 C 语言版本。也可以通过 `set_target_properties` 和 `C_STANDARD 17` 来设置语言版本，尽管可以说，`target_compile_features` 更加清晰，这也是为什么本书中更倾向于使用它的原因。

            将 `C_EXTENSIONS` 设置为 `OFF` 确保我们不会不小心使用不同编译器厂商添加的、不符合 C 标准（或如果我们使用了 `CXX_EXTENSIONS` 则是 C++ 标准）的语言特性。同样，这是为了帮助强制执行跨平台代码，使其不依赖于仅在某个编译器或平台上可用的特性。如果你打算只为一个平台或编译器进行构建，这一点不那么重要，但养成这个习惯是个好做法。特别是如果有一天你决定将代码移植到另一个平台，避免依赖特定编译器的特性将让这个过程变得更加容易。

            最终的表达式如下所示：

```cpp
set_target_properties(
  ${PROJECT_NAME}
  PROPERTIES C_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             C_STANDARD_REQUIRED ON
             C_EXTENSIONS OFF)
```

            为了更保险起见，如果我们不是以共享库的方式构建 `mc-gol`，我们还会添加 `MC_GOL_STATIC_DEFINE`（尽管在这种情况下，这并不是严格必要的，但这是一个很好的、低成本的防御性措施，可以避免将来可能出现的链接时问题，这取决于 `mc_gol_export.h` 的状态）。

            若想查看所有内容，可以访问 [`github.com/PacktPublishing/Minimal-CMake`](https://github.com/PacktPublishing/Minimal-CMake) 并查看 `ch4/part-5/lib/gol/CMakeLists.txt`。

            这就完成了我们对 *生命游戏* 库的所有修改！在进入下一章之前，我们还有一个重要的主题尚未讨论。

            接口库

            除了静态库和共享库外，还有另一种常见的库类型，通常被称为 `.h` 文件）。它不会在编译或链接时预先处理，`.h` 文件只是被包含进去，然后与主应用程序的源代码一起编译。

            仅头文件库因为其易于集成而非常受欢迎（你只需将 `.h` 文件包含到项目中，通常一切就能正常工作）。缺点是，每当你更改代码时，你必须重新编译该库，这会带来额外的开销，这种开销根据库的复杂度可能会很大。仅头文件库在 C++ 中尤其常见，尤其是模板库，因为它们的实现必须出现在头文件中。

            幸运的是，CMake 提供了一种直接的方法来创建仅头文件库，这些库可以像其他库一样使用。这里展示了一个完整的仅头文件 `CMakeLists.txt` 文件：

```cpp
cmake_minimum_required(VERSION 3.28)
project(mc-utils LANGUAGES C)
add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(
  ${PROJECT_NAME}
  INTERFACE $<BUILD_LOCAL_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>)
target_compile_features(${PROJECT_NAME} INTERFACE c_std_17)
```

            该文件应该与我们之前看到的 `CMakeLists.txt` 文件非常相似。主要的不同点是添加了 `INTERFACE` 关键字，取代了 `add_library` 命令中的 `STATIC` 或 `SHARED`，以及特定的 `target_...` 命令中的 `PUBLIC` 或 `PRIVATE`。

            `INTERFACE`关键字告知 CMake 这个目标没有源文件需要构建，也不会生成任何工件（库文件）。它所做的只是提供使用它的要求（在我们的例子中，我们指定了包含文件的位置，并要求使用 `c_std_17` 或更高版本）。`INTERFACE` 关键字还允许我们通过 `target_sources` 命令为依赖的目标指定一组源文件进行编译（我们将在*第九章*中看到此用途，*为项目编写测试*）。

            上面的代码是一个人为的示例，我们提取了一个不特定于*生命游戏*的单一有用工具函数，未来可能会使用（并且可能会添加）。这个函数是 `try_wrap`，它本质上是一个更强大的取模函数，能在处理负数时更好地进行环绕运算。

            现在，我们可以像下面这样在`mc-gol`中使用这个库：

```cpp
FetchContent_Declare(
  minimal-cmake-utils
  GIT_REPOSITORY <path/to/git-repo>
  GIT_TAG <commit-hash>)
FetchContent_MakeAvailable(minimal-cmake-utils)
target_link_libraries(<main-app> PRIVATE mc-utils)
```

            我们技术上并没有链接到这个库，但我们必须将目标添加为 `target_link_libraries` 的依赖项，以便为我们的目标应用程序填充包含搜索路径。然后，我们只需要在 `gol.c` 中添加 `#include <minimal-cmake/utils.h>` 以访问该函数。

            由于这仍然是一个 C 语言的仅头文件库，我们需要用 `static` 来注解我们的函数实现，以避免链接错误。这将导致在每个翻译单元（`.c` 文件）中生成函数的副本，这并不理想，但在这个简单的例子中是可行的。C++ 对仅头文件库的支持要好得多。在这种情况下，应该首选 `inline` 关键字（`inline` 在 C 语言中也受支持，但它在 C 中的含义与 C++ 中有所不同，使用起来也稍微复杂一些）。

            以这种方式使用仅包含头文件的库提供了在*第三章*中讨论的所有优势，*使用 FetchContent 处理外部依赖项*，包括将代码和依赖项分开，并使设置包含路径变得更加简单。

            你可以在 `ch4/part6/lib/utils/CMakeLists.txt` 和 `ch4/part6/app/CMakeLists.txt` 中找到完整的示例。

            摘要

            如果你已经走到这一步，给自己一个值得的鼓励——你已经走了很长一段路！在本章中，我们讨论了如何使库与`FetchContent`兼容。这包括回顾项目的物理结构、如何创建库，以及如何使用生成器表达式来控制包含接口。接着，我们查看了如何使用我们的新库。在此基础上，我们将我们的*生命游戏*逻辑提取到一个具有新接口的独立库中。我们深入探讨了如何将其制作成共享库，以及在 Windows、macOS 和 Linux 之间需要考虑的许多问题，还探讨了 CMake 如何帮助我们（通过导出头文件、在 Windows 上为 DLL 复制创建自定义命令，以及如何定制目标属性以帮助在 macOS 和 Linux 上创建可移动的库）。最后，我们通过做一些小的改进来帮助避免跨平台问题，并查看了接口（或仅头文件）库以及如何使用 CMake 创建它们。

            如果你还没有，请花一些时间通过访问[`github.com/PacktPublishing/Minimal-CMake`](https://github.com/PacktPublishing/Minimal-CMake)来熟悉本章讨论的示例，并尝试配置和构建这些项目（请参见`ch4`中的逐步示例）。实际的示例对于构建对这些概念的理解和熟悉非常有帮助。希望其中一些示例应该很容易提取并用于你的项目。了解如何创建库是一个重要的里程碑，并且为编写别人可以轻松使用的代码提供了令人兴奋的机会。

            现在你已经对创建库有了扎实的理解，是时候看看如何利用一些有用的 CMake 功能，使日常开发更快、更简单和更可靠了。我们将在下一章中做具体介绍。

```cpp

```

```cpp

```

```cpp

```

```cpp

```
