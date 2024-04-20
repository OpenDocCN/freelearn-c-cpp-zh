# 第一章：1. 可移植 C++软件的解剖学

## 学习目标

在本章结束时，您将能够：

+   建立代码构建测试流程

+   描述编译的各个阶段

+   解密复杂的 C++类型系统

+   配置具有单元测试的项目

+   将源代码转换为目标代码

+   编写可读的代码并调试它

在本章中，我们将学习建立贯穿全书使用的代码构建测试模型，编写优美的代码并进行单元测试。

## 介绍

C++是最古老和最流行的语言之一，您可以使用它来编写高效的代码。它既像 C 一样“接近底层”，又具有高级的面向对象特性，就像 Java 一样。作为一种高效的低级语言，C++是效率至关重要的领域的首选语言，例如游戏、模拟和嵌入式系统。同时，作为一种具有高级特性的面向对象语言，例如泛型、引用和无数其他特性，使其适用于由多人开发和维护的大型项目。

几乎任何编程经验都涉及组织您的代码库并使用他人编写的库。C++也不例外。除非您的程序很简单，否则您将把代码分发到多个文件中，并且需要组织这些文件，您将使用各种库来完成任务，通常比您的代码更有效和更可靠。不使用任何第三方库的 C++项目是不代表大多数项目的边缘情况，大多数项目都使用许多库。这些项目及其库预期在不同的硬件架构和操作系统上工作。因此，如果您要使用 C++开发任何有意义的东西，花时间进行项目设置并了解用于管理依赖关系的工具是很重要的。

大多数现代和流行的高级语言都有标准工具来维护项目、构建项目并处理其库依赖关系。其中许多都有托管库和工具的存储库，可以自动下载并使用这些库。例如，Python 有`pip`，它负责下载和使用程序员想要使用的库的适当版本。同样，JavaScript 有`npm`，Java 有`maven`，Dart 有`pub`，C#有`NuGet`。在这些语言中，您列出要使用的库的名称和版本，工具会自动下载并使用兼容版本的库。这些语言受益于程序在受控环境中构建和运行，其中满足一定级别的硬件和软件要求。另一方面，C++预期在各种上下文中使用，具有不同的架构，包括非常原始的硬件。因此，当涉及构建程序和执行依赖管理时，C++程序员受到的关注较少。

## 管理 C++项目

在 C++世界中，我们有几种工具可帮助管理项目源代码及其依赖关系。例如，`pkg-config`、`Autotools`、`make`和`CMake`是社区中最值得注意的工具。与其他高级语言的工具相比，这些工具使用起来要复杂得多。`CMake`已成为管理 C++项目及其依赖关系的事实标准。与`make`相比，它更具有主观性，并且被接受为大多数集成开发环境（IDE）的直接项目格式。

虽然`CMake`有助于管理项目及其依赖关系，但体验仍远远不及高级语言，其中您列出要使用的库及其版本，其他一切都会为您处理。使用 CMake，您仍需负责在开发环境中正确安装库，并且您需要使用每个库的兼容版本。在流行的 Linux 发行版中，有广泛的软件包管理器，您可以轻松安装大多数流行库的二进制版本。然而，有时您可能需要自行编译和安装库。这是 C++开发者体验的一部分，您将通过学习更多关于您选择的开发平台的开发平台来了解。在这里，我们将更专注于如何正确设置我们的 CMake 项目，包括理解和解决与库相关的问题。

### 代码构建测试运行循环

为了以坚实的基础展开讨论，我们将立即从一个实际示例开始。我们将从一个 C++代码基础模板开始，您可以将其用作自己项目的起点。我们将看到如何使用 CMake 在命令行上构建和编译它。我们还将为 C/C++开发人员设置 Eclipse IDE，并导入我们的 CMake 项目。使用 IDE 将为我们提供便利设施，以便轻松创建源代码，并使我们能够逐行调试我们的程序，查看程序执行过程中到底发生了什么，并以明智的方式纠正错误，而不是靠试错和迷信。

### 构建一个 CMake 项目

C++项目的事实标准是使用 CMake 来组织和构建项目。在这里，我们将使用一个基本的模板项目作为起点。以下是一个示例模板的文件夹结构：

![图 1.1：示例模板的文件夹结构](img/C14508_01_01.jpg)

###### 图 1.1：示例模板的文件夹结构

在上图中，`git`版本控制系统。这些被忽略的文件包括构建过程的输出，这些文件是在本地创建的，不应在计算机之间共享。

不同平台的`make`文件中的文件。

使用 CMake 构建项目是一个两步过程。首先，我们让 CMake 生成平台相关的配置文件，用于本地构建系统编译和构建项目。然后，我们将使用生成的文件来构建项目。CMake 可以为平台生成配置文件的构建系统包括`UNIX` `Makefiles`、`Ninja` `build files`、`NMake` `Makefiles`和`MinGW` `Makefiles`。选择取决于所使用的平台、这些工具的可用性和个人偏好。`UNIX` `Makefiles`是`Unix`和`Linux`的事实标准，而`NMake`是其`Windows`和`Visual Studio`的对应物。另一方面，`MinGW`是`Windows`中的`Unix`-like 环境，也在使用`Makefiles`。`Ninja`是一个现代的构建系统，与其他构建系统相比速度异常快，同时支持多平台，我们选择在这里使用。此外，除了这些命令行构建系统，我们还可以为`Visual Studio`、`XCode`、`Eclipse CDT`等生成 IDE 项目，并在 IDE 中构建我们的项目。因此，`CMake`是一个元工具，将为另一个实际构建项目的系统创建配置文件。在下一节中，我们将解决一个练习，其中我们将使用`CMake`生成`Ninja` `build files`。

### 练习 1：使用 CMake 生成 Ninja 构建文件

在这个练习中，我们将使用`CMake`生成`Ninja build files`，用于构建 C++项目。我们将首先从`git`存储库下载我们的源代码，然后使用 CMake 和 Ninja 来构建它。这个练习的目的是使用 CMake 生成 Ninja 构建文件，构建项目，然后运行它们。

#### 注意

GitHub 仓库的链接可以在这里找到：[`github.com/TrainingByPackt/Advanced-CPlusPlus/tree/master/Lesson1/Exercise01/project`](https://github.com/TrainingByPackt/Advanced-CPlusPlus/tree/master/Lesson1/Exercise01/project)。

执行以下步骤完成练习：

1.  在终端窗口中，输入以下命令，将`CxxTemplate`仓库从 GitHub 下载到本地系统：

```cpp
git clone https://github.com/TrainingByPackt/Advanced-CPlusPlus/tree/master/Lesson1/Exercise01/project
```

上一个命令的输出类似于以下内容：

![图 1.2：从 GitHub 检出示例项目](img/C14508_01_02.jpg)

###### 图 1.2：从 GitHub 检出示例项目

现在你已经在`CxxTemplate`文件夹中有了源代码。

1.  通过在终端中输入以下命令，进入`CxxTemplate`文件夹：

```cpp
cd CxxTemplate
```

1.  现在你可以通过在终端中输入以下命令来列出项目中的所有文件：

```cpp
find .
```

1.  在`CxxTemplate`文件夹中使用`cmake`命令生成我们的 Ninja 构建文件。为此，输入以下命令：

```cpp
cmake -Bbuild -H. -GNinja
```

上一个命令的输出如下：

![图 1.3：生成 Ninja 构建文件](img/C14508_01_03.jpg)

###### 图 1.3：生成 Ninja 构建文件

让我们解释一下上一个命令的部分。使用`-Bbuild`，我们告诉 CMake 使用`build`文件夹来生成构建产物。由于这个文件夹不存在，CMake 会创建它。使用`-H.`，我们告诉 CMake 使用当前文件夹作为源。通过使用单独的`build`文件夹，我们将保持我们的源文件干净，所有的构建产物都将存放在`build`文件夹中，这得益于我们的`.gitignore`文件而被 Git 忽略。使用`-GNinja`，我们告诉 CMake 使用 Ninja 构建系统。

1.  运行以下命令来列出项目文件并检查在`build`文件夹中创建的文件：

```cpp
ls
ls build
```

上一个命令将在终端中显示以下输出：

![图 1.4：构建文件夹中的文件](img/C14508_01_04.jpg)

###### 图 1.4：构建文件夹中的文件

很明显，上一个文件将存在于构建文件夹中。上一个输出中的**build.ninja**和**rules.ninja**是 Ninja 构建文件，实际上可以在这个平台上构建我们的项目。

#### 注意

通过使用 CMake，我们不必编写 Ninja 构建文件，并避免了对 Unix 平台的提交。相反，我们有一个可以为其他平台生成低级构建文件的元构建系统，比如 UNIX/Linux、MinGW 和 Nmake。

1.  现在，进入`build`文件夹，并通过在终端中输入以下命令来构建我们的项目：

```cpp
cd build
ninja
```

你应该看到最终输出如下：

![图 1.5：使用 ninja 构建](img/C14508_01_05.jpg)

###### 图 1.5：使用 ninja 构建

1.  在`CxxTemplate`可执行文件中键入`ls`或不键入：

```cpp
ls
```

上一个命令在终端中产生以下输出：

![图 1.6：运行 ninja 后构建文件夹中的文件](img/C14508_01_06.jpg)

###### 图 1.6：运行 ninja 后构建文件夹中的文件

在上一个图中，你可以看到`CxxTemplate`可执行文件已经生成。

1.  在终端中，输入以下命令来运行`CxxTemplate`可执行文件：

```cpp
./CxxTemplate
```

终端中的上一个命令将提供以下输出：

![](img/C14508_01_07.jpg)

###### 图 1.7：运行可执行文件

`src/CxxTemplate.cpp`文件中的以下行负责写入上一个输出：

```cpp
std::cout << "Hello CMake." << std::endl;
```

现在你已经成功在 Linux 中构建了一个 CMake 项目。Ninja 和 CMake 在一起工作得很好。你只需要运行一次 CMake，Ninja 就会检测是否需要再次调用 CMake，并会自动为你调用。例如，即使你向`CMakeLists.txt`文件中添加新的源文件，你只需要在终端中输入`ninja`命令，它就会自动运行 CMake 来更新 Ninja 构建文件。现在你已经了解了如何在 Linux 中构建 CMake 项目，在下一节中，我们将看看如何将 CMake 项目导入 Eclipse CDT。

## 将 CMake 项目导入 Eclipse CDT

Ninja 构建文件对于在 Linux 中构建我们的项目非常有用。但是，CMake 项目是可移植的，并且也可以与其他构建系统和 IDE 一起使用。许多 IDE 接受 CMake 作为其配置文件，并在您修改和构建项目时提供无缝体验。在本节中，我们将讨论如何将 CMake 项目导入 Eclipse CDT，这是一款流行的跨平台 C/C++ IDE。

使用 Eclipse CDT 与 CMake 有多种方法。CMake 提供的默认方法是单向生成 IDE 项目。在这里，您只需创建一次 IDE 项目，对 IDE 项目进行的任何修改都不会改变原始的 CMake 项目。如果您将项目作为 CMake 项目进行管理，并且只在 Eclipse CDT 中进行一次性构建，则这很有用。但是，如果您想在 Eclipse CDT 中进行开发，则不是理想的方法。

使用 Eclipse CDT 与 CMake 的另一种方法是使用自定义的`cmake4eclipse`插件。使用此插件时，您不会放弃您的`CMakeLists.txt`文件并单向切换到 Eclipse CDT 的项目管理器。相反，您将继续通过`CMakeLists.txt`文件管理项目，该文件将继续是项目的主要配置文件。Eclipse CDT 会积极与您的`CMakeLists.txt`文件合作构建项目。您可以在`CMakeLists.txt`中添加或删除源文件并进行其他更改，`cmake4eclipse`插件会在每次构建时将这些更改应用于 Eclipse CDT 项目。您将拥有良好的 IDE 体验，同时保持您的 CMake 项目处于最新状态。这种方法的好处是您始终可以停止使用 Eclipse CDT，并使用您的`CMakeLists.txt`文件切换到另一个构建系统（如 Ninja）。我们将在以下练习中使用这种第二种方法。

### 练习 2：将 CMake 文件导入 Eclipse CDT

在上一个练习中，您开发了一个 CMake 项目，并希望开始使用 Eclipse CDT IDE 来编辑和构建该项目。在本练习中，我们将使用`cmake4eclipse`插件将我们的 CMake 项目导入 Eclipse CDT IDE。执行以下步骤完成练习：

1.  打开 Eclipse CDT。

1.  在当前项目的位置（包含`CMakeLists.txt`文件和**src**文件夹的文件夹）中创建一个新的 C++项目。转到**文件** | **新建** | **项目**。将出现一个类似以下截图的**新建项目**对话框：![图 1.8：新建项目对话框](img/C14508_01_08.jpg)

###### 图 1.8：新建项目对话框

1.  选择**C++项目**选项，然后点击**下一步**按钮。将出现一个类似以下截图的**C++项目**对话框：![图 1.9：C++项目对话框](img/C14508_01_09.jpg)

###### 图 1.9：C++项目对话框

1.  接受一切，包括切换到 C/C++视角，然后点击**完成**。

1.  点击左上角的**还原**按钮查看新创建的项目：![图 1.10：还原按钮](img/C14508_01_10.jpg)

###### 图 1.10：还原按钮

1.  点击**CxxTemplate**项目。转到**项目** | **属性**，然后在左侧窗格下选择**C/C++构建**下的**工具链编辑器**，将**当前构建器**设置为**CMake Builder (portable)**。然后，点击**应用并关闭**按钮：![图 1.11：项目属性](img/C14508_01_11.jpg)

###### 图 1.11：项目属性

1.  然后，选择**项目** | **构建全部**菜单项来构建项目：![图 1.12：构建项目](img/C14508_01_12.jpg)

###### 图 1.12：构建项目

1.  在接下来的`make all`中实际构建我们的项目：![图 1.13：构建输出](img/C14508_01_13.jpg)

###### 图 1.13：构建输出

1.  如果在之前的步骤中没有出现任何错误，您可以使用菜单项**运行** | **运行**来运行项目。如果给出了一些选项，请选择**本地 C/C++应用程序**和**CxxTemplate**作为可执行文件：![图 1.14：运行项目](img/C14508_01_14.jpg)

###### 图 1.14：运行项目

1.  当运行时，你会在**控制台**窗格中看到程序的输出如下：

![图 1.15：项目的输出](img/C14508_01_15.jpg)

###### 图 1.15：项目的输出

你已经成功地使用 Eclipse CDT 构建和运行了一个 CMake 项目。在下一个练习中，我们将通过添加新的源文件和新类来频繁地更改我们的项目。

### 练习 3：向 CMake 和 Eclipse CDT 添加新的源文件

随着 C++项目的不断扩大，你会倾向于向其中添加新的源文件，以满足预期的要求。在这个练习中，我们将向我们的项目中添加一个新的`.cpp`和`.h`文件对，并看看 CMake 和 Eclipse CDT 如何处理这些更改。我们将使用新类向项目中添加这些文件，但你也可以使用任何其他文本编辑器创建它们。执行以下步骤将新的源文件添加到 CMake 和 Eclipse CDT 中：

1.  首先，打开我们一直在使用的项目。在左侧的**项目资源管理器**窗格中，展开根条目**CxxTemplate**，你会看到我们项目的文件和文件夹。右键单击**src**文件夹，从弹出菜单中选择**新建** | **类**：![图 1.16：创建一个新类](img/C14508_01_16.jpg)

###### 图 1.16：创建一个新类

1.  在打开的对话框中，为类名输入**ANewClass**。当你点击**完成**按钮时，你会看到**src**文件夹下生成了**ANewClass.cpp**和**ANewClass.h**文件。

1.  现在，让我们在`ANewClass`类中写一些代码，并从`ANewClass.cpp`中访问它，并更改文件的开头以匹配以下内容，然后保存文件：

```cpp
#include "ANewClass.h"
#include <iostream>
void ANewClass::run() {
    std::cout << "Hello from ANewClass." << std::endl;
}
```

你会看到 Eclipse 用`ANewClass.h`文件警告我们。这些警告是由 IDE 中的分析器实现的，非常有用，因为它们可以在你输入代码时帮助你修复代码，而无需运行编译器。

1.  打开`ANewClass.h`文件，添加以下代码，并保存文件：

```cpp
public:
    void run(); // we added this line
    ANewClass();
```

你应该看到`.cpp`文件中的错误消失了。如果没有消失，可能是因为你可能忘记保存其中一个文件。你应该养成按*Ctrl + S*保存当前文件的习惯，或者按*Shift + Ctrl + S*保存你编辑过的所有文件。

1.  现在，让我们从我们的另一个类`CxxTemplate.cpp`中使用这个类。打开该文件，进行以下修改，并保存文件。在这里，我们首先导入头文件，在`CxxApplication`的构造函数中，我们向控制台打印文本。然后，我们创建了`ANewClass`的一个新实例，并调用了它的`run`方法：

```cpp
#include "CxxTemplate.h"
#include "ANewClass.h"
#include <string>
...
CxxApplication::CxxApplication( int argc, char *argv[] ) {
  std::cout << "Hello CMake." << std::endl;
  ::ANewClass anew;
  anew.run();
}
```

#### 注意

这个文件的完整代码可以在这里找到：[`github.com/TrainingByPackt/Advanced-CPlusPlus/blob/master/Lesson1/Exercise03/src/CxxTemplate.cpp`](https://github.com/TrainingByPackt/Advanced-CPlusPlus/blob/master/Lesson1/Exercise03/src/CxxTemplate.cpp)。

1.  尝试通过点击`CMakeLists.txt`文件来构建项目，进行以下修改，并保存文件：

```cpp
add_executable(CxxTemplate
  src/CxxTemplate.cpp  
  src/ANewClass.cpp
)
```

尝试再次构建项目。这次你不应该看到任何错误。

1.  使用**运行** | **运行**菜单选项运行项目。你应该在终端中看到以下输出：

![图 1.18：程序输出](img/C14508_01_18.jpg)

###### 图 1.18：程序输出

你修改了一个 CMake 项目，向其中添加了新文件，并成功地运行了它。请注意，我们在`src`文件夹中创建了文件，并让`CMakeLists.txt`文件知道了 CPP 文件。如果你不使用 Eclipse，你可以继续使用通常的 CMake 构建命令，你的程序将成功运行。到目前为止，我们已经从 GitHub 检出了示例代码，并且用纯 CMake 和 Eclipse IDE 构建了它。我们还向 CMake 项目中添加了一个新类，并在 Eclipse IDE 中重新构建了它。现在你知道如何构建和修改 CMake 项目了。在下一节中，我们将进行一个活动，向项目添加一个新的源文件-头文件对。

### 活动 1：向项目添加新的源文件-头文件对

在开发 C++项目时，随着项目的增长，您会向其中添加新的源文件。您可能出于各种原因想要添加新的源文件。例如，假设您正在开发一个会计应用程序，在其中需要在多个地方计算利率，并且您希望创建一个单独的文件中的函数，以便在整个项目中重用它。为了保持简单，在这里我们将创建一个简单的求和函数。在这个活动中，我们将向项目添加一个新的源文件和头文件对。执行以下步骤完成该活动：

1.  在 Eclipse IDE 中打开我们在之前练习中创建的项目。

1.  将`SumFunc.cpp`和`SumFunc.h`文件对添加到项目中。

1.  创建一个名为`sum`的简单函数，它返回两个整数的和。

1.  从`CxxTemplate`类构造函数中调用该函数。

1.  在 Eclipse 中构建并运行项目。

预期输出应该类似于以下内容：

![图 1.19：最终输出](img/C14508_01_19.jpg)

###### 图 1.19：最终输出

#### 注意

此活动的解决方案可在第 620 页找到。

在接下来的部分中，我们将讨论如何为我们的项目编写单元测试。将项目分成许多类和函数，并让它们一起工作以实现期望的目标是很常见的。您必须使用单元测试来管理这些类和函数的行为，以确保它们以预期的方式运行。

## 单元测试

单元测试在编程中是一个重要的部分。基本上，单元测试是使用我们的类在各种场景下进行测试的小程序，预期结果是在我们的项目中的一个并行文件层次结构中，不会最终出现在实际的可执行文件中，而是在开发过程中由我们单独执行，以确保我们的代码以预期的方式运行。我们应该为我们的 C++程序编写单元测试，以确保它们在每次更改后都能按照预期的方式运行。

### 为单元测试做准备

有几个 C++测试框架可以与 CMake 一起使用。我们将使用**Google Test**，它比其他选项有几个优点。在下一个练习中，我们将准备我们的项目以便使用 Google Test 进行单元测试。

### 练习 4：为单元测试准备我们的项目

我们已经安装了 Google Test，但我们的项目还没有设置好以使用 Google Test 进行单元测试。除了安装之外，在我们的 CMake 项目中还需要进行一些设置才能进行 Google Test 单元测试。按照以下步骤执行此练习：

1.  打开 Eclipse CDT，并选择我们一直在使用的 CxxTemplate 项目。

1.  创建一个名为**tests**的新文件夹，因为我们将在那里执行所有的测试。

1.  编辑我们的基本`CMakeLists.txt`文件，以允许在`GTest`包中进行测试，该包为 CMake 带来了`GoogleTest`功能。我们将在此之后添加我们的新行：

```cpp
find_package(GTest)
if(GTEST_FOUND)
set(Gtest_FOUND TRUE)
endif()
if(GTest_FOUND)
include(GoogleTest)
endif()
# add these two lines below
enable_testing()
add_subdirectory(tests)
```

这就是我们需要添加到我们主要的`CMakeLists.txt`文件中的所有内容。

1.  在我们主要的`CMakeLists.txt`文件中的`add_subdirectory(tests)`行内创建另一个`CMakeLists.txt`文件。这个`tests/CMakeLists.txt`文件将管理测试源代码。

1.  在`tests/CMakeLists.txt`文件中添加以下代码：

```cpp
include(GoogleTest)
add_executable(tests CanTest.cpp)
target_link_libraries(tests GTest::GTest)
gtest_discover_tests(tests)
```

让我们逐行解析这段代码。第一行引入了 Google Test 功能。第二行创建了`tests`可执行文件，其中将包括所有我们的测试源文件。在这种情况下，我们只有一个`CanTest.cpp`文件，它将验证测试是否有效。之后，我们将`GTest`库链接到`tests`可执行文件。最后一行标识了`tests`可执行文件中的所有单独测试，并将它们添加到`CMake`作为一个测试。这样，各种测试工具将能够告诉我们哪些单独的测试失败了，哪些通过了。

1.  创建一个`tests/CanTest.cpp`文件。添加这段代码来简单验证测试是否运行，而不实际测试我们实际项目中的任何内容：

```cpp
#include "gtest/gtest.h"
namespace {
class CanTest: public ::testing::Test {};
TEST_F(CanTest, CanReallyTest) {
  EXPECT_EQ(0, 0);
}
}  
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
```

`TEST_F`行是一个单独的测试。现在，`EXPECT_EQ(0, 0)`正在测试零是否等于零，如果我们实际运行测试，它将始终成功。稍后，我们将在这里添加我们自己类的结果，以便对各种值进行测试。现在我们的项目中已经具备了 Google Test 的必要设置。接下来，我们将构建和运行这些测试。

### 构建、运行和编写单元测试

现在，我们将讨论如何构建、运行和编写单元测试。到目前为止，我们所拥有的示例是一个简单的虚拟测试，已准备好进行构建和运行。稍后，我们将添加更有意义的测试，并查看通过和失败测试的输出。在接下来的练习中，我们将为上一个练习中创建的项目构建、运行和编写单元测试。

### 练习 5：构建和运行测试

到目前为止，您已经创建了一个设置好的`GoogleTest`的项目，但没有构建或运行我们创建的测试。在这个练习中，我们将构建和运行我们创建的测试。由于我们使用`add_subdirectory`添加了我们的`tests`文件夹，构建项目将自动构建测试。运行测试将需要更多的努力。执行以下步骤完成练习：

1.  在 Eclipse CDT 中打开我们的 CMake 项目。

1.  构建测试，只需像以前一样构建项目即可。以下是在 Eclipse 中进行完整构建后再次构建项目的输出，使用**Project** | **Build All**：![图 1.20：构建操作及其输出](img/C14508_01_20.jpg)

###### 图 1.20：构建操作及其输出

1.  如果您没有看到此输出，则可能是因为您的控制台处于错误的视图中。您可以按照以下图示进行更正：![图 1.21：查看正确的控制台输出](img/C14508_01_21.jpg)

###### 图 1.21：查看正确的控制台输出

![图 1.22：查看正确的控制台输出](img/C14508_01_22.jpg)

###### 图 1.22：查看正确的控制台输出

如您所见，我们的项目现在有两个可执行目标。它们都位于`build`文件夹中，与任何其他构建产物一样。它们的位置分别是`build/Debug/CxxTemplate`和`build/Debug/tests/tests`。由于它们是可执行文件，我们可以直接运行它们。

1.  我们之前运行了`CxxTemplate`，现在不会看到任何额外的输出。通过在项目文件夹中输入以下命令，我们可以运行其他可执行文件：

```cpp
./build/Debug/tests/tests
```

前面的代码在终端中生成了以下输出：

![图 1.23：运行测试可执行文件](img/C14508_01_23.jpg)

###### 图 1.23：运行测试可执行文件

这是我们的`tests`可执行文件的简单输出。如果您想查看测试是否通过，您可以简单地运行它。但是，测试远不止于此。

1.  您可以通过使用`ctest`命令之一来运行测试。在项目文件夹中的终端中输入以下命令。我们进入`tests`可执行文件所在的文件夹，运行`ctest`，然后返回：

```cpp
cd build/Debug/tests
ctest
cd ../../..
```

以下是您将看到的输出：

![图 1.24：运行 ctest](img/C14508_01_24.jpg)

###### 图 1.24：运行 ctest

#### 注意

`ctest`命令可以使用多种选项运行您的`tests`可执行文件，包括自动将测试结果提交到在线仪表板的功能。在这里，我们将简单地运行`ctest`命令；其更多功能留给感兴趣的读者作为练习。您可以输入`ctest --help`或访问在线文档以了解更多关于`ctest`的信息，网址为[`cmake.org/cmake/help/latest/manual/ctest.1.html#`](https://cmake.org/cmake/help/latest/manual/ctest.1.html#)。

1.  另一种运行测试的方法是在 Eclipse 中以漂亮的图形报告格式运行它们。为此，我们将创建一个测试感知的运行配置。在 Eclipse 中，单击**Run** | **Run Configurations…**，在左侧右键单击**C/C++ Unit**，然后选择**New Configuration**。

1.  将名称从**CxxTemplate Debug**更改为**CxxTemplate Tests**如下所示：![图 1.25：更改运行配置的名称](img/C14508_01_25.jpg)

###### 图 1.25：更改运行配置的名称

1.  在**C/C++ Application**下，选择**Search Project**选项：![图 1.26：运行配置](img/C14508_01_26.jpg)

###### 图 1.26：运行配置

1.  在新对话框中选择**tests**：![图 1.27：创建测试运行配置并选择测试可执行文件](img/C14508_01_27.jpg)

###### 图 1.27：创建测试运行配置并选择测试可执行文件

1.  接下来，转到**C/C++ Testing**选项卡，并在下拉菜单中选择**Google Tests Runner**。点击对话框底部的**Apply**，然后点击第一次运行的测试的**Run**选项：![图 1.28：运行配置](img/C14508_01_28.jpg)

###### 图 1.28：运行配置

1.  在即将进行的运行中，您可以单击工具栏中播放按钮旁边的下拉菜单，或选择**Run** | **Run History**来选择**CxxTemplate Tests**：

![图 1.29：完成运行配置设置并选择要运行的配置](img/C14508_01_29.jpg)

###### 图 1.29：完成运行配置设置并选择要运行的配置

结果将类似于以下截图：

![图 1.30：单元测试的运行结果](img/C14508_01_30.jpg)

###### 图 1.30：单元测试的运行结果

这是一个很好的报告，包含了所有测试的条目，现在只有一个。如果您不想离开 IDE，您可能会更喜欢这个。此外，当您有许多测试时，此界面可以帮助您有效地对其进行过滤。现在，您已经构建并运行了使用 Google Test 编写的测试。您以几种不同的方式运行了它们，包括直接执行测试，使用`ctest`和使用 Eclipse CDT。在下一节中，我们将解决一个练习，其中我们将实际测试我们代码的功能。

### 练习 6：测试代码功能

您已经运行了简单的测试，但现在您想编写有意义的测试来测试功能。在初始活动中，我们创建了`SumFunc.cpp`，其中包含`sum`函数。现在，在这个练习中，我们将为该文件编写一个测试。在这个测试中，我们将使用`sum`函数来添加两个数字，并验证结果是否正确。让我们回顾一下之前包含`sum`函数的以下文件的内容：

+   `src/SumFunc.h`：

```cpp
#ifndef SRC_SUMFUNC_H_
#define SRC_SUMFUNC_H_
int sum(int a, int b);
#endif /* SRC_SUMFUNC_H_ */
```

+   `src/SumFunc.cpp`：

```cpp
#include "SumFunc.h"
#include <iostream>
int sum(int a, int b) {
  return a + b;
}
```

+   `CMakeLists.txt`的相关行：

```cpp
add_executable(CxxTemplate
  src/CxxTemplate.cpp  
  src/ANewClass.cpp
  src/SumFunc.cpp
)
```

另外，让我们回顾一下我们的`CantTest.cpp`文件，它包含了我们单元测试的`main()`函数：

```cpp
#include "gtest/gtest.h"
namespace {
class CanTest: public ::testing::Test {};
TEST_F(CanTest, CanReallyTest) {
  EXPECT_EQ(0, 0);
}
}  
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
```

执行以下步骤完成练习：

1.  在 Eclipse CDT 中打开我们的 CMake 项目。

1.  添加一个新的测试源文件（`tests/SumFuncTest.cpp`），内容如下：

```cpp
#include "gtest/gtest.h"
#include "../src/SumFunc.h"
namespace {
  class SumFuncTest: public ::testing::Test {};
  TEST_F(SumFuncTest, CanSumCorrectly) {
    EXPECT_EQ(7, sum(3, 4));
  }
}
```

请注意，这里没有`main()`函数，因为`CanTest.cpp`有一个，它们将被链接在一起。其次，请注意，这包括`SumFunc.h`，它在测试中使用了`sum(3, 4)`。这是我们在测试中使用项目代码的方式。

1.  在`tests/CMakeLists.txt`文件中进行以下更改以构建测试：

```cpp
include(GoogleTest)
add_executable(tests CanTest.cpp SumFuncTest.cpp ../src/SumFunc.cpp) # added files here
target_link_libraries(tests GTest::GTest)
gtest_discover_tests(tests)
```

请注意，我们将测试（`SumFuncTest.cpp`）和它测试的代码（`../src/SumFunc.cpp`）都添加到可执行文件中，因为我们的测试代码正在使用实际项目中的代码。

1.  构建项目并像以前一样运行测试。您应该看到以下报告：![图 1.31：运行测试后的输出](img/C14508_01_31.jpg)

###### 图 1.31：运行测试后的输出

我们可以将这样的测试添加到我们的项目中，所有这些测试都将显示在屏幕上，就像前面的截图所示的那样。

1.  现在，让我们添加一个实际失败的测试。在`tests/SumFuncTest.cpp`文件中，进行以下更改：

```cpp
TEST_F(SumFuncTest, CanSumCorrectly) {
  EXPECT_EQ(7, sum(3, 4));
}
// add this test
TEST_F(SumFuncTest, CanSumAbsoluteValues) {
  EXPECT_EQ(6, sum(3, -3));
}
```

请注意，此测试假定输入的绝对值被求和，这是不正确的。这次调用的结果是`0`，但在这个例子中预期是`6`。这是我们在项目中必须做的唯一更改，以添加这个测试。

1.  现在，构建项目并运行测试。您应该会看到这个报告：![图 1.32：构建报告](img/C14508_01_32.jpg)

###### 图 1.32：构建报告

如前图所示，前两个测试通过了，最后一个测试失败了。当我们看到这个输出时，有两种选择：要么我们的项目代码有问题，要么测试有问题。在这种情况下，我们的测试有问题。这是因为我们的`6`等于`sum(3, -3)`。这是因为我们假设我们的函数对提供的整数的绝对值求和。然而，事实并非如此。我们的函数只是简单地添加给定的数字，无论它们是正数还是负数。因此，这个测试有一个错误的假设，所以失败了。

1.  让我们改变测试并修复它。修改测试，使我们期望`-3`和`3`的和为`0`。重命名测试以反映这个测试实际上做了什么：

```cpp
TEST_F(SumFuncTest, CanSumCorrectly) {
  EXPECT_EQ(7, sum(3, 4));
}
// change this part
TEST_F(SumFuncTest, CanUseNegativeValues) {
  EXPECT_EQ(0, sum(3, -3));
}
```

1.  现在运行它，并观察报告中所有测试是否都通过了：

![图 1.33：测试执行成功](img/C14508_01_33.jpg)

###### 图 1.33：测试执行成功

最后，我们已经在系统和项目中使用 CMake 设置了 Google Test。我们还使用 Google Test 编写、构建和运行了单元测试，无论是在终端还是在 Eclipse 中。理想情况下，您应该为每个类编写单元测试，并覆盖每种可能的用法。您还应该在每次重大更改后运行测试，并确保不会破坏现有代码。在下一节中，我们将执行一个添加新类及其测试的活动。

### 活动 2：添加新类及其测试

在开发 C++项目时，随着项目的增长，我们会向其中添加新的源文件。我们还会为它们编写测试，以确保它们正常工作。在这个活动中，我们将添加一个模拟`1D`线性运动的新类。该类将具有`position`和`velocity`的 double 字段。它还将有一个`advanceTimeBy()`方法，接收一个 double `dt`参数，根据`velocity`的值修改`position`。对于 double 值，请使用`EXPECT_DOUBLE_EQ`而不是`EXPECT_EQ`。在这个活动中，我们将向项目中添加一个新类及其测试。按照以下步骤执行此活动：

1.  在 Eclipse IDE 中打开我们创建的项目。

1.  将`LinearMotion1D.cpp`和`LinearMotion1D.h`文件对添加到包含`LinearMotion1D`类的项目中。在这个类中，创建两个 double 字段：`position`和`velocity`。另外，创建一个`advanceTimeBy(double dt)`函数来修改`position`。

1.  在`tests/LinearMotion1DTest.cpp`文件中为此编写测试。编写两个代表两个不同方向运动的测试。

1.  在 Eclipse IDE 中构建并运行它。

1.  验证测试是否通过。

最终的测试结果应该类似于以下内容：

![图 1.34：最终测试结果](img/C14508_01_34.jpg)

###### 图 1.34：最终测试结果

#### 注意

这个活动的解决方案可以在第 622 页找到。

在 C++开发中，添加新类及其测试是一项非常常见的任务。我们出于各种原因创建类。有时，我们有一个很好的软件设计计划，我们创建它所需的类。其他时候，当一个类变得过大和单一时，我们以有意义的方式将一些责任分离到另一个类中。使这项任务变得实际是很重要的，以防止拖延和最终得到庞大的单一类。在接下来的部分中，我们将讨论编译和链接阶段发生了什么。这将让我们更好地了解 C++程序底层发生了什么。

## 理解编译、链接和目标文件内容

使用 C++的主要原因之一是效率。C++使我们能够控制内存管理，这就是为什么理解对象在内存中的布局很重要的原因。此外，C++源文件和库被编译为目标硬件的对象文件，并进行链接。通常，C++程序员必须处理链接器问题，这就是为什么理解编译步骤并能够调查对象文件很重要的原因。另一方面，大型项目是由团队在长时间内开发和维护的，这就是为什么创建清晰易懂的代码很重要的原因。与任何其他软件一样，C++项目中会出现错误，需要通过观察程序行为来仔细识别、分析和解决。因此，学习如何调试 C++代码也很重要。在接下来的部分中，我们将学习如何创建高效、与其他代码协作良好且易于维护的代码。

### 编译和链接步骤

C++项目是一组源代码文件和项目配置文件，用于组织源文件和库依赖关系。在编译步骤中，这些源文件首先被转换为对象文件。在链接步骤中，这些对象文件被链接在一起，形成项目的最终输出可执行文件。项目使用的库也在这一步中被链接。

在即将进行的练习中，我们将使用现有项目来观察编译和链接阶段。然后，我们将手动重新创建它们以更详细地查看这个过程。

### 练习 7：识别构建步骤

您一直在构建项目而没有调查构建操作的详细信息。在这个练习中，我们将调查我们项目的构建步骤的详细信息。执行以下操作完成练习：

1.  打开终端。

1.  通过输入以下命令导航到`build`文件夹，其中我们的`Makefile`文件位于其中：

```cpp
cd build/Debug
```

1.  使用以下命令清理项目并以`VERBOSE`模式运行构建：

```cpp
make clean 
make VERBOSE=1 all
```

您将在终端中获得构建过程的详细输出，可能会显得有点拥挤：

![图 1.35：构建过程第 1 部分](img/C14508_01_35.jpg)

###### 图 1.35：构建过程第 1 部分

![图 1.36：构建过程第 2 部分](img/C14508_01_36.jpg)

###### 图 1.36：构建过程第 2 部分

![图 1.37：完整的构建输出](img/C14508_01_37.jpg)

###### 图 1.37：完整的构建输出

以下是此输出中的一些行。以下行是与主可执行文件的编译和链接相关的重要行：

```cpp
/usr/bin/c++    -g   -pthread -std=gnu++1z -o CMakeFiles/CxxTemplate.dir/src/CxxTemplate.cpp.o -c /home/username/Packt/Cpp2019/CxxTemplate/src/CxxTemplate.cpp
/usr/bin/c++    -g   -pthread -std=gnu++1z -o CMakeFiles/CxxTemplate.dir/src/ANewClass.cpp.o -c /home/username/Packt/Cpp2019/CxxTemplate/src/ANewClass.cpp
/usr/bin/c++    -g   -pthread -std=gnu++1z -o CMakeFiles/CxxTemplate.dir/src/SumFunc.cpp.o -c /home/username/Packt/Cpp2019/CxxTemplate/src/SumFunc.cpp
/usr/bin/c++    -g   -pthread -std=gnu++1z -o CMakeFiles/CxxTemplate.dir/src/LinearMotion1D.cpp.o -c /home/username/Packt/Cpp2019/CxxTemplate/src/LinearMotion1D.cpp
/usr/bin/c++  -g   CMakeFiles/CxxTemplate.dir/src/CxxTemplate.cpp.o CMakeFiles/CxxTemplate.dir/src/ANewClass.cpp.o CMakeFiles/CxxTemplate.dir/src/SumFunc.cpp.o CMakeFiles/CxxTemplate.dir/src/LinearMotion1D.cpp.o  -o CxxTemplate -pthread 
```

1.  这里的`c++`命令只是`g++`编译器的符号链接。要查看它实际上是一系列符号链接，输入以下命令：

```cpp
namei /usr/bin/c++
```

您将看到以下输出：

![图 1.38：/usr/bin/c++的符号链接链](img/C14508_01_38.jpg)

###### 图 1.38：/usr/bin/c++的符号链接链

因此，在我们的讨论中，我们将交替使用`c++`和`g++`。在我们之前引用的构建输出中，前四行是编译每个`.cpp`源文件并创建相应的`.o`对象文件。最后一行是将这些对象文件链接在一起以创建`CxxTemplate`可执行文件。以下图形形象地展示了这个过程：

![图 1.39：C++项目的执行阶段](img/C14508_01_39.jpg)

###### 图 1.39：C++项目的执行阶段

如前面的图所示，作为目标的一部分添加到 CMake 中的 CPP 文件以及它们包含的头文件被编译为对象文件，然后将它们链接在一起以创建目标可执行文件。

1.  为了进一步了解这个过程，让我们自己执行编译步骤。在终端中，转到项目文件夹并使用以下命令创建一个名为`mybuild`的新文件夹：

```cpp
cd ~/CxxTemplate
mkdir mybuild
```

1.  然后，运行以下命令将 CPP 源文件编译为对象文件：

```cpp
/usr/bin/c++ src/CxxTemplate.cpp -o mybuild/CxxTemplate.o -c 
/usr/bin/c++ src/ANewClass.cpp -o mybuild/ANewClass.o -c 
/usr/bin/c++ src/SumFunc.cpp -o mybuild/SumFunc.o -c 
/usr/bin/c++ src/LinearMotion1D.cpp -o mybuild/LinearMotion1D.o -c 
```

1.  进入`mybuild`目录，并使用以下命令查看其中的内容：

```cpp
cd mybuild
ls 
```

我们看到了预期的以下输出。这些是我们的目标文件：

![图 1.40：已编译的目标文件](img/C14508_01_40.jpg)

###### 图 1.40：已编译的目标文件

1.  在下一步中，将目标文件链接在一起形成我们的可执行文件。输入以下命令：

```cpp
/usr/bin/c++  CxxTemplate.o ANewClass.o SumFunc.o LinearMotion1D.o  -o CxxTemplate 
```

1.  现在，通过输入以下命令，让我们在文件列表中看到我们的可执行文件：

```cpp
ls 
```

这显示了以下图中的新`CxxTemplate`文件：

![图 1.41：链接可执行文件](img/C14508_01_41.jpg)

###### 图 1.41：链接可执行文件

1.  现在，通过输入以下命令运行我们的可执行文件：

```cpp
./CxxTemplate
```

然后看看我们之前的输出：

![图 1.42：可执行文件输出](img/C14508_01_42.jpg)

###### 图 1.42：可执行文件输出

现在您已经检查了构建过程的细节，并自己重新创建了它们，在下一节中，让我们探索链接过程。

### 链接步骤

在本节中，让我们看一下两个源文件之间的联系以及它们如何最终出现在同一个可执行文件中。看看以下图中的**sum**函数：

![图 1.43：链接过程](img/C14508_01_43.jpg)

###### 图 1.43：链接过程

**sum**函数的主体在**SumFunc.cpp**中定义。它在**SumFunc.h**中有一个前向声明。这样，想要使用**sum**函数的源文件可以了解其签名。一旦它们知道了它的签名，它们就可以调用它，并相信在运行时将会有实际的函数定义，而实际上并没有与**SumFunc.cpp**交互。

编译后，调用**sum**函数的**CxxTemplate.cpp**将该调用传递到其目标文件中。但它不知道函数定义在哪里。**SumFunc.cpp**的目标文件具有该定义，但与**CxxTemplate.o**无关。

在链接步骤中，链接器将**CxxTemplate.o**中的调用与**SumFunc.o**中的定义进行匹配。结果，可执行文件中的调用正常工作。如果链接器找不到**sum**函数的定义，它将产生链接器错误。

链接器找到了`无法解析符号`错误。

这使我们经历了构建过程的两个阶段：`编译`和`链接`。请注意，与手动编译源文件时相比，我们使用了相当简单的命令。随时输入`man g++`以查看所有选项。稍后，我们将讨论链接以及符号是如何解析的。我们还讨论了链接步骤可能出现的问题。在下一节中，我们将学习有关目标文件的知识。

### 深入挖掘：查看目标文件

为了使链接步骤能够正常工作，我们需要使所有符号引用与符号定义匹配。大多数情况下，我们可以通过查看源文件来分析解决方案将如何解析。有时，在复杂情况下，我们可能难以理解为什么符号未能解析。在这种情况下，查看目标文件的内容以调查引用和定义可能有助于解决问题。除了链接器错误外，了解目标文件的内容以及链接工作的一般原理对于 C++程序员来说是有用的。了解底层发生的事情可能有助于程序员更好地理解整个过程。

当我们的源代码编译为目标文件时，我们的语句和表达式将转换为汇编代码，这是 CPU 理解的低级语言。汇编中的每条指令都包含一个操作，后跟寄存器，这些寄存器是 CPU 的寄存器。有指令用于将数据加载到寄存器中并从寄存器中加载数据，并对寄存器中的值进行操作。Linux 中的`objdump`命令可帮助我们查看这些目标文件的内容。

#### 注意

我们将利用 Compiler Explorer，这是一个很好用的在线工具，您可以在左侧窗口上编写代码，在右侧可以看到编译后的汇编代码。这是 Compiler Explorer 的链接：[`godbolt.org`](https://godbolt.org)。

### 练习 8：探索编译代码

在这个练习中，我们将使用 Compiler Explorer 编译一些简单的 C++代码，其中我们定义并调用一个函数。我们将调查编译后的汇编代码，以了解名称是如何解析和调用是如何进行的。这将让我们更好地理解发生了什么以及我们的代码在可执行格式中是如何工作的。执行以下步骤完成练习：

1.  在`call sum(int, int)`行中添加以下代码可以实现您的预期：它调用前面的`sum`函数并将参数放入一些寄存器中。这里的重要一点是，函数是通过它们的名称和参数类型按顺序标识的。链接器会寻找具有这个签名的适当函数。请注意，返回值不是签名的一部分。

1.  禁用`_Z`，数字告诉我们函数名的长度，以便正确解释后面的字母。在函数名之后，我们有`v`表示没有参数，`i`表示一个`int`参数。您可以更改这些函数签名以查看其他可能的类型。

1.  现在，让我们看看类是如何编译的。将以下代码添加到**Compiler Explorer**的现有代码下：

```cpp
class MyClass {
private:
    int a = 5;
    int myPrivateFunc(int i) {
        a = 4;
        return i + a;
    }
public:
    int b = 6;
    int myFunc(){ 
        return sum(1, myPrivateFunc(b));
    }
};
MyClass myObject;
int main() {
    myObject.myFunc();
}
```

这是这些添加行的编译版本：

![图 1.46：编译版本](img/C14508_01_46.jpg)

###### 图 1.46：编译版本

您可能会惊讶地发现编译代码中没有类定义。这些方法类似于全局函数，但有一个变化：它们的混淆名称包含类名，并将对象实例作为参数接收。创建实例只是为类的字段分配空间。

在链接器阶段，这些混淆的函数名用于将调用者与被调用者匹配。对于找不到被调用者的调用者，我们会得到链接器错误。大多数链接器错误可以通过仔细检查源代码来解决。然而，在某些情况下，使用`objdump`查看目标文件内容可以帮助找到问题的根源。

## 调试 C++代码

在开发 C++项目时，您可能会遇到不同级别的问题：

+   首先，您可能会收到编译器错误。这可能是因为您在语法上犯了错误，或者选择了错误的类型等。编译器是您必须跨越的第一个障碍，它会捕捉到您可能犯的一些错误。

+   第二个障碍是链接器。在那里，一个常见的错误是使用声明但实际上未定义的内容。当您使用错误的库头文件时，这种情况经常发生——头文件宣传了某个不存在于任何源文件或库中的签名。一旦您也通过了链接器的障碍，您的程序就准备好执行了。

+   现在，下一个要跨越的障碍是避免任何运行时错误。您的代码可能已经编译和链接成功，但可能会出现一些不起作用的情况，比如解引用空指针或除以零。

要查找和修复运行时错误，您必须以某种方式与正在运行的应用程序进行交互和监视。一个经常使用的技术是向代码中添加`print`语句，并监视它生成的日志，希望将应用程序行为与日志相关联，以确定代码中存在问题的区域。虽然这对某些情况有效，但有时您需要更仔细地查看执行情况。

调试器是一个更好的工具来解决运行时错误。调试器可以让你逐行运行代码，继续运行并在你想要的行上暂停，调查内存的值，并在错误上暂停，等等。这让你可以在程序运行时观察内存的具体情况，并确定导致不良行为的代码行。

`gdb`是一个经典的命令行调试器，可以调试 C++程序。然而，它可能难以使用，因为调试本质上是一项视觉任务——你希望能够同时查看代码行、变量值和程序的输出。幸运的是，Eclipse CDT 包含了一个易于使用的可视化调试器。

### 练习 9：使用 Eclipse CDT 进行调试

你之前只是简单地运行项目并查看输出。现在你想要学习如何详细调试你的代码。在这个练习中，我们将探索 Eclipse CDT 的调试能力。按照以下步骤完成练习：

1.  在 Eclipse CDT 中打开 CMake 项目。

1.  为了确保我们有一个现有的运行配置，点击**运行** | **运行配置**。在那里，你应该在**C/C++应用程序**下看到一个**CxxTemplate**条目。

#### 注意

由于我们之前运行了项目，它应该在那里。如果没有，请返回并重新创建。

1.  关闭对话框以继续。

1.  要启动调试器，找到看起来像昆虫（虫子）的工具栏条目，并点击旁边的下拉菜单。选择`main()`函数，它在代码视图中央显示为绿色高亮和箭头。在左侧，我们看到正在运行的线程，其中只有一个。在右侧，我们看到在这个上下文中可访问的变量。在底部，我们看到 Eclipse 在后台使用的**gdb**输出来实际调试可执行文件。现在，我们的主函数没有太多需要调试的地方。

1.  点击`libc-start.c`库，它是`main`函数的调用者。当完成后，你可以关闭它并切换到你的源文件。当你不再看到红色停止按钮时，你就知道程序执行结束了。

1.  通过添加以下代码编辑我们的`main`函数：

```cpp
int i = 1, t = 0;
do {
  t += i++;
} while (i <= 3);
std::cout << t << std::endl;
```

后增量运算符与偶尔的`do-while`循环对一些人来说可能是一个难题。这是因为我们试图在脑海中执行算法。然而，我们的调试器完全能够逐步运行它，并显示在执行过程中到底发生了什么。

1.  在添加了上述代码后开始调试。点击工具栏上**调试**按钮旁边的下拉菜单，选择**CxxTemplate**。按下*F6*几次来逐步执行代码。它会显示变量的变化以及将要执行的代码行：![图 1.48：跳过代码](img/C14508_01_48.jpg)

###### 图 1.48：跳过代码

1.  在执行每行代码后看到变量的变化，可以更清楚地理解算法。按下*F6*，注意在执行`t += i++;`这行代码后的值：![图 1.49：变量状态随时间变化](img/C14508_01_49.jpg)

###### 图 1.49：变量状态随时间变化

前面的输出清楚地解释了值是如何变化的，以及为什么最后打印出`6`。

1.  探索调试器的其他功能。虽然变量视图很有用，但你也可以悬停在任何变量上并浏览它的值：![图 1.50：调试器的视图选项](img/C14508_01_50.jpg)

###### 图 1.50：调试器的视图选项

此外，**表达式**视图帮助你计算那些从浏览的值中不清楚的东西。

1.  在右侧点击**表达式**，然后点击**添加**按钮：![图 1.51：添加表达式](img/C14508_01_51.jpg)

###### 图 1.51：添加表达式

1.  输入**t+i**并按*Enter*。现在你可以在表达式列表中看到总和：![图 1.52：带有新表达式的表达式视图](img/C14508_01_52.jpg)

###### 图 1.52：带有新表达式的表达式视图

您可以在工具栏中按下红色方块，或选择**运行** | **终止**随时停止调试。另一个功能是断点，它告诉调试器每当它到达带有断点的行时暂停。到目前为止，我们一直在逐行执行我们的代码，这在一个大型项目中可能非常耗时。相反，通常您希望继续执行，直到到达您感兴趣的代码。

1.  现在，不是逐行进行，而是在进行打印的行中添加一个断点。为此，请双击此行行号左侧的区域。在下图中，点表示断点：![图 1.53：使用断点](img/C14508_01_53.jpg)

###### 图 1.53：使用断点

1.  现在启动调试器。通常情况下，它将开始暂停。现在选择**运行** | **恢复**或单击工具栏按钮。它将运行循环的三次执行，并在我们的断点处暂停。这样，我们通过跳过我们不调查的代码来节省时间：![图 1.54：使用调试器](img/C14508_01_54.jpg)

###### 图 1.54：使用调试器

1.  当我们处理添加的循环时，我们忽略了创建`app`对象的行。**步过**命令跳过了这行。但是，我们也有选择进入这行中的构造函数调用的选项。为此，我们将使用**运行** | **步入**或相应的工具栏按钮。

1.  停止调试器，然后再次启动。单击**步过**以转到创建应用程序的行：![图 1.55：使用调试器 - 步过选项](img/C14508_01_55.jpg)

###### 图 1.55：使用调试器 - 步过选项

1.  如果我们再次步过，高亮显示的是下一行将执行的行。相反，按下步入按钮。这将带我们进入构造函数调用：

![图 1.56：使用调试器 - 步入选项](img/C14508_01_56.jpg)

###### 图 1.56：使用调试器 - 步入选项

这是一个方便的功能，可以更深入地了解函数，而不仅仅是跳过它。还要注意左侧调试视图中的调用堆栈。您可以随时单击较低的条目以再次查看调用者的上下文。

这是对 Eclipse CDT 调试器的简要介绍，它在内部使用 GDB 为您提供可视化调试体验。在尝试更好地理解运行时错误并纠正导致这些错误的错误时，您可能会发现调试非常有用。

## 编写可读的代码

虽然可视化调试器非常有用，可以识别和消除运行时错误或意外的程序行为，但更好的做法是编写更不太可能出现问题的代码。其中一种方法是努力编写更易读和理解的代码。然后，在代码中找问题更像是识别英语句子之间的矛盾，而不是解决神秘的谜题。当您以一种易于理解的方式编写代码时，您的错误通常在制造时就会显现出来，并且在您回来解决滑过的问题时更容易发现。

经历了一些令人不愉快的维护经验后，你意识到你编写的程序的主要目的不是让计算机按照你的意愿去做，而是告诉读者程序运行时计算机将会做什么。这通常意味着你需要输入更多的内容，而集成开发环境可以帮助你。这也可能意味着你有时会编写在执行时间或内存使用方面不是最优的代码。如果这与你所学的知识相悖，考虑到你可能在以微不足道的效率换取错误的风险。在我们拥有的庞大处理能力和内存的情况下，你可能会使你的代码变得不必要地晦涩，可能会在追求效率的虚无之中产生错误。在接下来的章节中，我们将列出一些经验法则，这些法则可能会帮助你编写更易读的代码。

### 缩进和格式化

C++代码，就像许多其他编程语言一样，由程序块组成。一个函数有一组语句组成它的主体作为一个块。循环的块语句将在迭代中执行。如果给定条件为真，则`if`语句的块将执行，相应的`else`语句的块将在条件为假时执行。

花括号，或者对于单语句块的缺失，通知计算机，而缩进形式的空白则通知人类读者关于块结构。缺乏缩进或者误导性的缩进会使读者非常难以理解代码的结构。因此，我们应该努力保持我们的代码缩进良好。考虑以下两个代码块：

```cpp
// Block 1
if (result == 2) 
firstFunction();
secondFunction();
// Block 2
if (result == 2) 
  firstFunction();
secondFunction();
```

虽然从执行的角度来看它们是相同的，但在第二个示例中更清楚地表明`firstFunction()`只有在`result`是`2`的情况下才会被执行。现在考虑以下代码：

```cpp
if (result == 2) 
  firstFunction();
  secondFunction();
```

这只是误导。如果读者不小心，他们可能会很容易地假设`secondFunction()`只有在`result`是`2`的情况下才会被执行。然而，从执行的角度来看，这段代码与前两个示例是相同的。

如果你觉得纠正缩进在减慢你的速度，你可以使用编辑器的格式化工具来帮助你。在 Eclipse 中，你可以选择一段代码并使用**源码** | **纠正缩进**来修复该选择的缩进，或者使用**源码** | **格式化**来修复代码的其他格式问题。

除了缩进之外，其他格式规则，比如将花括号放在正确的行上，在二元运算符周围插入空格，以及在每个逗号后插入一个空格，也是非常重要的格式规则，你应该遵守这些规则，以保持你的代码格式良好，易于阅读。

在 Eclipse 中，你可以在**窗口** | **首选项** | **C/C++** | **代码样式** | **格式化程序**中为每个工作空间设置格式化规则，或者在**项目** | **属性** | **C/C++常规** | **格式化程序**中为每个项目设置格式化规则。你可以选择行业标准样式，比如 K&R 或 GNU，或者修改它们并创建自己的样式。当你使用**源码** | **格式化**来格式化你的代码时，这变得尤为重要。例如，如果你选择使用空格进行缩进，但 Eclipse 的格式化规则设置为制表符，你的代码将成为制表符和空格的混合体。

### 使用有意义的标识符名称

在我们的代码中，我们使用标识符来命名许多项目——变量、函数、类名、类型等等。对于计算机来说，这些标识符只是一系列字符，用于区分它们。然而，对于读者来说，它们更重要。标识符应该完全且明确地描述它所代表的项目。同时，它不应该过长。此外，它应该遵守正在使用的样式标准。

考虑以下代码：

```cpp
studentsFile File = runFileCheck("students.dat");
bool flag = File.check();
if (flag) {
    int Count_Names = 0;
    while (File.CheckNextElement() == true) {
        Count_Names += 1;
    }
    std::cout << Count_Names << std::endl;
}
```

虽然这是一段完全有效的 C++代码，但它很难阅读。让我们列出它的问题。首先，让我们看看标识符的风格问题。`studentsFile`类名以小写字母开头，而应该是大写字母。`File`变量应该以小写字母开头。`Count_Names`变量应该以小写字母开头，而且不应该有下划线。`CheckNextElement`方法应该以小写字母开头。虽然这些规则可能看起来是武断的，但在命名上保持一致会携带关于名称的额外信息——当你看到一个以大写字母开头的单词时，你立刻明白它必须是一个类名。此外，拥有不遵守使用标准的名称只会分散注意力。

现在，让我们超越风格，检查名称本身。第一个有问题的名称是`runFileCheck`函数。方法是返回值的动作：它的名称应该清楚地解释它的作用以及它的返回值。 “Check”是一个过度使用的词，在大多数情况下都太模糊了。是的，我们检查了，它在那里——那么我们接下来该怎么办呢？在这种情况下，似乎我们实际上读取了文件并创建了一个`File`对象。在这种情况下，`runFileCheck`应该改为`readFile`。这清楚地解释了正在进行的操作，返回值是你所期望的。如果你想对返回值更具体，`readAsFile`可能是另一种选择。同样，`check`方法太模糊了，应该改为`exists`。`CheckNextElement`方法也太模糊了，应该改为`nextElementExists`。

另一个过度使用的模糊词是`flag`，通常用于布尔变量。名称暗示了一个开/关的情况，但并没有提示其值的含义。在这种情况下，它的`true`值表示文件存在，`false`值表示文件不存在。命名布尔变量的技巧是设计一个问题或语句，当变量的值为`true`时是正确的。在这个例子中，`fileExists`和`doesFileExist`是两个不错的选择。

我们下一个命名不当的变量是`Count_Names`，或者正确的大写形式`countNames`。这对于整数来说是一个糟糕的名称，因为名称并没有暗示一个数字，而是暗示导致一个数字的动作。相反，诸如`numNames`或`nameCount`这样的标识符会清楚地传达内部数字的含义。

### 保持算法清晰简单

当我们阅读代码时，所采取的步骤和流程应该是有意义的。间接进行的事情——函数的副产品，为了效率而一起执行的多个操作等等——这些都会让读者难以理解你的代码。例如，让我们看看以下代码：

```cpp
int *input = getInputArray();
int length = getInputArrayLength();
int sum = 0;
int minVal = 0;
for (int i = 0; i < length; ++i) {
  sum += input[i];
  if (i == 0 || minVal > input[i]) {
    minVal = input[i];
  }
  if (input[i] < 0) {
    input[i] *= -1;
  }
}
```

在这里，我们有一个在循环中处理的数组。乍一看，很难确定循环到底在做什么。变量名帮助我们理解正在发生的事情，但我们必须在脑海中运行算法，以确保这些名称所宣传的确实发生在这里。在这个循环中进行了三种不同的操作。首先，我们找到所有元素的总和。其次，我们找到数组中的最小元素。第三，我们在这些操作之后取每个元素的绝对值。

现在考虑这个替代版本：

```cpp
int *input = getInputArray();
int length = getInputArrayLength();
int sum = 0;
for (int i = 0; i < length; ++i) {
  sum += input[i];
}
int minVal = 0;
for (int i = 0; i < length; ++i) {
  if (i == 0 || minVal > input[i]) {
    minVal = input[i];
  }
}
for (int i = 0; i < length; ++i) {
  if (input[i] < 0) {
    input[i] *= -1;
  }
}
```

现在一切都清晰多了。第一个循环找到输入的总和，第二个循环找到最小的元素，第三个循环找到每个元素的绝对值。虽然现在更清晰、更易理解，但你可能会觉得自己在做三个循环，因此浪费了 CPU 资源。创造更高效的代码的动力可能会促使你合并这些循环。请注意，这里的效率提升微乎其微；你的程序的时间复杂度仍然是 O(n)。

在创建代码时，可读性和效率是经常竞争的两个约束条件。如果你想开发可读性强、易于维护的代码，你应该始终优先考虑可读性。然后，你应该努力开发同样高效的代码。否则，可读性低的代码可能难以维护，甚至可能存在难以识别和修复的错误。当你的程序产生错误结果或者添加新功能的成本变得太高时，程序的高效性就变得无关紧要了。

### 练习 10：使代码更易读

以下代码存在样式和缩进问题。空格使用不一致，缩进不正确。此外，关于单语句`if`块是否使用大括号的决定也不一致。以下代码存在缩进、格式、命名和清晰度方面的问题：

```cpp
//a is the input array and Len is its length
void arrayPlay(int *a, int Len) { 
    int S = 0;
    int M = 0;
    int Lim_value = 100;
    bool flag = true;
    for (int i = 0; i < Len; ++i) {
    S += a[i];
        if (i == 0 || M > a[i]) {
        M = a[i];
        }
        if (a[i] >= Lim_value) {            flag = true;
            }
            if (a[i] < 0) {
            a[i] *= 2;
        }
    }
}
```

让我们解决这些问题，使其符合常见的 C++代码风格。执行以下步骤完成这个练习：

1.  打开 Eclipse CDT。

1.  创建一个新的`a`，其长度为`Len`。对这些更好的命名应该是`input`和`inputLength`。

1.  让我们首先做出这个改变，将`a`重命名为`input`。如果你正在使用 Eclipse，你可以选择`Len`并将其重命名为`inputLength`。

1.  更新后的代码将如下所示。请注意，由于参数名是不言自明的，我们不再需要注释：

```cpp
void arrayPlay(int *input, int inputLength) {
    int S = 0;
    int M = 0;
    int Lim_value = 100;
    bool flag = true;
    for (int i = 0; i < inputLength; ++i) {
        S += input[i];
        if (i == 0 || M > input[i]) {
            M = input[i];
        }
        if (input[i] >= Lim_value) {
            flag = true;
        }
        if (input[i] < 0) {
            input[i] *= 2;
        }
    }
}
```

1.  在循环之前我们定义了一些其他变量。让我们试着理解它们。它似乎只是将每个元素添加到`S`中。因此，`S`必须是`sum`。另一方面，`M`似乎是最小的元素——让我们称它为`smallest`。

1.  `Lim_value`似乎是一个阈值，我们只是想知道它是否被越过。让我们将其重命名为`topThreshold`。如果越过了这个阈值，`flag`变量被设置为 true。让我们将其重命名为`isTopThresholdCrossed`。在这些更改后，代码的状态如下所示：**重构** | **重命名**：

```cpp
void arrayPlay(int *input, int inputLength) {
    int sum = 0;
    int smallest = 0;
    int topThreshold = 100;
    bool isTopThresholdCrossed = true;
    for (int i = 0; i < inputLength; ++i) {
        sum += input[i];
        if (i == 0 || smallest > input[i]) {
            smallest = input[i];
        }
        if (input[i] >= topThreshold) {
            isTopThresholdCrossed = true;
        }
        if (input[i] < 0) {
            input[i] *= 2;
        }
    }
}
```

现在，让我们看看如何使这段代码更简单、更易理解。前面的代码正在做这些事情：计算输入元素的总和，找到最小的元素，确定是否越过了顶部阈值，并将每个元素乘以 2。

1.  由于所有这些都是在同一个循环中完成的，现在算法不太清晰。修复这个问题，将其分为四个独立的循环：

```cpp
void arrayPlay(int *input, int inputLength) {
    // find the sum of the input
    int sum = 0;
    for (int i = 0; i < inputLength; ++i) {
        sum += input[i];
    }
    // find the smallest element
    int smallest = 0;
    for (int i = 0; i < inputLength; ++i) {
        if (i == 0 || smallest > input[i]) {
            smallest = input[i];
        }
    }
    // determine whether top threshold is crossed
    int topThreshold = 100;
    bool isTopThresholdCrossed = true;
    for (int i = 0; i < inputLength; ++i) {
        if (input[i] >= topThreshold) {
            isTopThresholdCrossed = true;
        }
    }
    // multiply each element by 2
    for (int i = 0; i < inputLength; ++i) {
        if (input[i] < 0) {
            input[i] *= 2;
        }
    }
}
```

现在代码清晰多了。虽然很容易理解每个块在做什么，但我们还添加了注释以使其更清晰。在这一部分，我们更好地理解了我们的代码是如何转换为可执行文件的。然后，我们讨论了识别和解决可能的代码错误的方法。我们最后讨论了如何编写可读性更强、更不容易出现问题的代码。在下一部分，我们将解决一个活动，我们将使代码更易读。

### 活动 3：使代码更易读

你可能有一些难以阅读并且包含错误的代码，要么是因为你匆忙写成的，要么是因为你从别人那里收到的。你想改变代码以消除其中的错误并使其更易读。我们有一段需要改进的代码。逐步改进它并使用调试器解决问题。执行以下步骤来实施这个活动：

1.  下面是`SpeedCalculator`类的源代码。将这两个文件添加到你的项目中。

1.  在你的`main()`函数中创建这个类的一个实例，并调用它的`run()`方法。

1.  修复代码中的风格和命名问题。

1.  简化代码以使其更易理解。

1.  运行代码并观察运行时的问题。

1.  使用调试器来解决问题。

这是**SpeedCalculator.cpp**和**SpeedCalculator.h**的代码，你将把它们添加到你的项目中。你将修改它们作为这个活动的一部分：

```cpp
// SpeedCalculator.h
#ifndef SRC_SPEEDCALCULATOR_H_
#define SRC_SPEEDCALCULATOR_H_
class SpeedCalculator {
private:
    int numEntries;
    double *positions;
    double *timesInSeconds;
    double *speeds;
public:
    void initializeData(int numEntries);
    void calculateAndPrintSpeedData();
};
#endif /* SRC_SPEEDCALCULATOR_H_ */

//SpeedCalculator.cpp
#include "SpeedCalculator.h"
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <cassert>
void SpeedCalculator::initializeData(int numEntries) {
    this->numEntries = numEntries;
    positions = new double[numEntries];
    timesInSeconds = new double[numEntries];
    srand(time(NULL));
    timesInSeconds[0] = 0.0;
    positions[0] = 0.0;
    for (int i = 0; i < numEntries; ++i) {
    positions[i] = positions[i-1] + (rand()%500);
    timesInSeconds[i] = timesInSeconds[i-1] + ((rand()%10) + 1);
    }
}
void SpeedCalculator::calculateAndPrintSpeedData() {
    double maxSpeed = 0;
    double minSpeed = 0;
    double speedLimit = 100;
    double limitCrossDuration = 0;
    for (int i = 0; i < numEntries; ++i) {
        double dt = timesInSeconds[i+1] - timesInSeconds[i];
        assert (dt > 0);
        double speed = (positions[i+1] - positions[i]) / dt;
            if (maxSpeed < speed) {
                maxSpeed = speed;
            }
            if (minSpeed > speed) {
                minSpeed = speed;
            }
        if (speed > speedLimit) {
            limitCrossDuration += dt;
        }
        speeds[i] = speed;
    }
    std::cout << "Max speed: " << maxSpeed << std::endl;
        std::cout << "Min speed: " << minSpeed << std::endl;
        std::cout << "Total duration: " << 
timesInSeconds[numEntries - 1] - timesInSeconds[0] << " seconds" << std::endl;
    std::cout << "Crossed the speed limit for " << limitCrossDuration << " seconds"<< std::endl;
    delete[] speeds;
}
```

#### 注意

这个活动的解决方案可以在第 626 页找到。

## 总结

在本章中，我们学习了如何创建可移植和可维护的 C++项目。我们首先学习了如何创建 CMake 项目以及如何将它们导入到 Eclipse CDT，从而使我们可以选择使用命令行或者 IDE。本章的其余部分侧重于消除项目中的各种问题。首先，我们学习了如何向项目添加单元测试，以及如何使用它们来确保我们的代码按预期工作。然后，我们讨论了代码经历的编译和链接步骤，并观察了目标文件的内容，以更好地理解可执行文件。接着，我们学习了如何在 IDE 中以可视化方式调试我们的代码，以消除运行时错误。我们用一些经验法则结束了这个讨论，这些法则有助于创建可读、易懂和可维护的代码。这些方法将在你的 C++之旅中派上用场。在下一章中，我们将更多地了解 C++的类型系统和模板。
