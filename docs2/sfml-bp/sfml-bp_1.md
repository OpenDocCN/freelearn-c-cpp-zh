# 第一章. 准备环境

通过这本书，我将尝试教你一些使用 SFML 库构建视频游戏的基本元素。每一章都会涵盖一个不同的主题，并且需要掌握前一章的知识。

在本章中，我们将介绍未来所需的基础要点，例如：

+   安装 C++11 的编译器

+   安装 CMake

+   安装 SFML 2.2

+   构建一个最小的 SFML 项目

在开始之前，让我们谈谈每种技术以及为什么我们会使用它们。

# C++11

C++编程语言是一个非常强大的工具，具有真正出色的性能，但它也非常复杂，即使经过多年的实践也是如此。它允许我们在低级和高级进行编程。它有助于对我们的程序进行一些优化，例如直接操作内存的能力。利用 C++库构建软件使我们能够在高级别工作，当性能至关重要时，在低级别工作。此外，C/C++编译器在优化代码方面非常高效。结果是，目前，C++在速度方面是最强大的语言，多亏了零成本抽象，你不会为不使用的内容或提供的抽象付费。

我会尝试以现代的方式使用这种语言，采用面向对象的方法。有时，我会绕过这种方法，使用 C 语言的方式进行优化。所以，当你看到一些“老式代码”时，请不要感到惊讶。此外，现在所有的主流编译器都支持 2011 年发布的标准语言，因此我们可以毫无困难地使用它。这个版本在语言中添加了一些非常实用的功能，这些功能将在本书中使用，例如以下内容：

+   关键字是这样一个重要特性。以下是一些例子：

    +   `auto`：这个关键字可以自动检测新变量的类型。对于迭代器的实例化来说，它非常有用。`auto`关键字过去就已经存在，但已经废弃很长时间了，其含义现在已改变。

    +   `nullptr`：这是一个引入旧 NULL 值强类型的全新关键字。你总是可以使用 NULL，但最好使用`nullptr`，它是指针类型，其值为 0。

    +   `override`和`final`：这两个关键字已经存在于一些语言中，如 Java。这些关键字不仅对编译器，也对程序员有简单的指示意义，但并不指定它们指示的内容。不要犹豫使用它们。你可以查看它们的文档[`en.cppreference.com/w/cpp/language/override`](http://en.cppreference.com/w/cpp/language/override)和[`en.cppreference.com/w/cpp/language/final`](http://en.cppreference.com/w/cpp/language/final)。

+   基于范围的`for`循环是语言`foreach`中的一种新循环类型。此外，你可以使用新的`auto`关键字来大幅度减少你的代码。以下语法非常简单：

    ```cpp
    for(auto& var : table){...}.
    ```

    在这个例子中，`table` 是一个容器（vector 和 list），而 `var` 是存储变量的引用。使用 `&` 允许我们修改表内包含的变量，并避免复制。

+   C++11 引入了智能指针。有多个指针对应于它们不同的可能用途。查看官方文档，这真的很有趣。主要思想是管理内存，当没有更多引用时，在运行时删除创建的对象，这样你就不必自己删除它或确保没有发生双重释放损坏。在栈上创建的智能指针具有速度快且在方法/代码块结束时自动删除的优点。但重要的是要知道，对这种指针的强烈使用，尤其是 `shared_ptr`，会降低程序的执行速度，所以请谨慎使用。

+   Lambda 表达式或匿名函数是一种引入了特定语法的全新类型。现在，你可以创建函数，例如，将其作为另一个函数的参数。这对于回调来说非常有用。在过去，我们使用函数对象（functor）来实现这种行为。以下是一个函数对象和 Lambda 的示例：

    ```cpp
    class Func(){ void operator()(){/* code here */}};
    auto f = [](){/* code here*/};
    ```

+   如果你已经熟悉了使用省略号运算符（`...`）的变长参数函数，那么这个概念可能会让你感到困惑，因为它的用法是不同的。变长模板只是使用省略号运算符对任何数量的参数进行优化的模板。一个很好的例子是元组类。元组可以包含编译时已知的任何类型和数量的值。如果没有变长模板，实际上无法构建此类，但现在这变得非常简单。顺便说一下，元组类是在 C++11 中引入的。还有其他一些特性，如线程、对等、等等。

# SFML

**SFML** 代表 **Simple and Fast Multimedia Library**。这是一个用 C++ 编写的框架，其图形渲染部分基于 OpenGL。这个名字很好地描述了其目标，即拥有一个用户友好的界面（API），提供高性能，并且尽可能便携。SFML 库分为五个模块，这些模块分别编译到单独的文件中：

+   **系统（System）**：这是主要模块，所有其他模块都需要它。它提供了时钟、线程以及所有二维和三维的逻辑（数学运算）。

+   **窗口（Window）**：此模块允许应用程序通过管理窗口以及来自鼠标、键盘和游戏手柄的输入与用户交互。

+   **图形（Graphics）**：此模块允许用户使用所有基本的图形元素，如纹理、形状、文本、颜色、着色器等。

+   **音频（Audio）**：此模块允许用户使用一些声音。多亏了它，我们将能够播放一些主题、音乐和声音。

+   **网络**: 此模块不仅管理套接字和类型安全的传输，还管理 HTTP 和 FTP 协议。它对于在不同程序之间进行通信也非常有用。

我们程序使用的每个模块在编译时都需要与它们链接。如果不需要，我们不需要链接它们。本书将涵盖每个模块，但不会涵盖所有 SFML 类。我建议您查看 SFML 文档[`www.sfml-dev.org/documentation.php`](http://www.sfml-dev.org/documentation.php)，因为它非常有趣且内容完整。每个模块和类都在不同的部分中得到了很好的描述。

现在已经介绍了主要技术，让我们安装使用它们所需的所有内容。

# 安装 C++11 编译器

如前所述，我们将使用 C++11，因此我们需要为其安装编译器。对于每个操作系统，都有几个选项；选择您喜欢的。

## 对于 Linux 用户

如果您是 Linux 用户，您可能已经安装了 GCC/G++。在这种情况下，请检查您的版本是否为 4.8 或更高版本。否则，您可以使用您喜欢的包管理器安装 GCC/G++（版本 4.8+）或 Clang（版本 3.4+）。在基于 Debian 的发行版（如 Ubuntu 和 Mint）下，使用以下命令行：

```cpp
sudo apt-get install gcc g++ clang -y

```

## 对于 Mac 用户

如果您是 Mac 用户，您可以使用 Clang (3.4+)。这是 Mac OS X 下的默认编译器。

## 对于 Windows 用户

最后，如果您是 Windows 用户，您可以通过下载来使用 Visual Studio (2013)、Mingw-gcc (4.8+)或 Clang (3.4+)。我建议您不要使用 Visual Studio，因为它对于 C99 标准并不完全兼容，而是使用另一个 IDE，例如 Code::Blocks（见下一段）。

## 对于所有用户

我假设在两种情况下，您都已经能够安装编译器并配置系统以使用它（通过将其添加到系统路径）。如果您无法做到这一点，另一种解决方案是安装一个像 Code::Blocks 这样的 IDE，它具有以下优点：默认安装编译器，与 C++11 兼容，且不需要任何系统配置。

我将在本书的其余部分选择带有 Code::Blocks 的 IDE 选项，因为它不依赖于特定的操作系统，每个人都能导航。您可以在[`www.codeblocks.org/downloads/26`](http://www.codeblocks.org/downloads/26)下载它。安装非常简单；您只需按照向导操作即可。

# 安装 CMake

CMake 是一个非常实用的工具，它以编译器无关的方式管理任何操作系统中的构建过程。此配置非常简单。我们将需要它来构建 SFML（如果您选择此安装方案）以及构建本书的所有未来项目。使用 CMake 为我们提供了一个跨平台解决方案。我们需要 CMake 的 2.8 或更高版本。目前，最后一个稳定版本是 3.0.2。

## 对于 Linux 用户

如果您使用 Linux 系统，您可以使用包管理器安装 CMake 及其 GUI。例如，在 Debian 下，使用以下命令行：

```cpp
sudo apt-get install cmake cmake-gui -y

```

## 对于其他操作系统

你可以在[`www.cmake.org/download/`](http://www.cmake.org/download/)下载适合你系统的 CMake 二进制文件。按照向导操作，然后安装完成。CMake 现在已安装并准备好使用。

# 安装 SFML 2.2

获取 SFML 库有两种方法。更简单的方法是下载预构建版本，可以在[`sfml-dev.org/download/sfml/2.2/`](http://sfml-dev.org/download/sfml/2.2/)找到，但请确保你下载的版本与你的编译器兼容。

第二种选择是自行编译库。与之前的方法相比，这种方法更可取，可以避免任何麻烦。

## 自行编译 SFML

编译 SFML 并不像我们想象的那么困难，对每个人来说都是可行的。首先，我们需要安装一些依赖项。

### 安装依赖项

SFML 依赖于几个库。在开始编译之前，请确保你已经安装了所有依赖项及其开发文件。以下是依赖项列表：

+   `pthread`

+   `opengl`

+   `xlib`

+   `xrandr`

+   `freetype`

+   `glew`

+   `jpeg`

+   `sndfile`

+   `openal`

### Linux

在 Linux 上，我们需要安装这些库的开发版本。包的确切名称取决于每个发行版，但这里是为 Debian 的命令行：

```cpp
sudo apt-get install libglu1-mesa-dev freeglut3-dev mesa-common-dev libxrandr-dev libfreetype6-dev libglew-dev libjpeg-dev libsndfile1-dev libopenal-dev -y

```

### 其他操作系统

在 Windows 和 Mac OS X 上，所有需要的依赖项都直接由 SFML 提供，因此你不需要下载或安装任何东西。编译将直接完成。

### SFML 的编译

如前所述，SFML 的编译过程非常简单。我们只需按照以下步骤使用 CMake：

1.  在[`sfml-dev.org/download/sfml/2.2/`](http://sfml-dev.org/download/sfml/2.2/)下载源代码并解压。

1.  打开 CMake，指定源代码目录和构建目录。按照惯例，构建目录称为`build`，位于源目录的根级别。

1.  点击**配置**按钮，并选择适合你系统的**Code::Blocks**。

    在 Linux 下，选择**Unix Makefiles**。它应该看起来像这样：

    ![SFML 的编译](img/8477OS_01_01.jpg)

    在 Windows 下，选择**MinGW Makefiles**。它应该看起来像这样：

    ![SFML 的编译](img/8477OS_01_02.jpg)

1.  最后，点击**生成**按钮。你会得到如下输出：![SFML 的编译](img/8477OS_01_03.jpg)

现在 Code::Blocks 文件已构建，可以在你的构建目录中找到。用 Code::Blocks 打开它，并点击**构建**按钮。所有二进制文件都将构建并放置在`build/lib`目录中。此时，你将有一些依赖于你系统的文件。如下所示：

+   `libsfml-system`

+   `libsfml-window`

+   `libsfml-graphics`

+   `libsfml-audio`

+   `libsfml-network`

每个文件对应于一个不同的 SFML 模块，这些模块将是我们未来游戏运行所需的。

现在是时候配置我们的系统以便能够找到它们了。我们所需做的只是将 `build/lib` 目录添加到我们的系统路径中。

#### Linux

要在 Linux 中编译，首先打开一个终端并运行以下命令：

```cpp
cd /your/path/to/SFML-2.2/build

```

以下命令将在 `/usr/local/lib/` 下安装二进制文件，并在 `/usr/local/include/SFML/` 中安装头文件：

```cpp
sudo make install

```

默认情况下，`/usr/local/` 已在您的系统路径中，因此无需进行更多操作。

#### Windows

在 Windows 上，您需要按照以下方式将 `/build/lib/` 目录添加到您的系统路径中：

1.  在 **系统属性** 的 **高级** 选项卡中，点击 **环境变量** 按钮：![Windows](img/8477OS_01_04.jpg)

1.  然后，在 **系统变量** 表中选中 **路径**，并点击 **编辑...** 按钮：![Windows](img/8477OS_01_05.jpg)

1.  现在编辑 **变量值** 输入文本，添加 `;C:\your\path\to\SFML-2.2\build\lib`，然后通过在所有打开的窗口中点击 **确定** 来验证它：![Windows](img/8477OS_01_06.jpg)

到目前为止，您的系统已配置为查找 SFML `dll` 模块。

## Code::Blocks 和 SFML

现在您的系统已配置为查找 SFML 二进制文件，是时候配置 Code::Blocks 并最终测试您的全新安装是否一切正常了。为此，请按照以下步骤操作：

1.  运行 Code::Blocks，转到 **文件** | **新建** | **项目**，然后选择 **控制台应用程序**。

1.  点击 **GO**。

1.  选择 **C++** 作为编程语言，并按照说明操作，直到创建项目。现在已创建一个包含典型 `Hello world` 程序的默认 `main.cpp` 文件。尝试构建并运行它以检查您的编译器是否正确检测到。![Code::Blocks 和 SFML](img/8477OS_01_07.jpg)

如果一切正常，将创建一个新窗口，其中包含 `Hello world!` 消息，如下所示：

![Code::Blocks 和 SFML](img/8477OS_01_08.jpg)

如果您看到这个输出，那么一切正常。在任何其他情况下，请确保您已遵循所有安装步骤。

现在，我们将配置 Code::Blocks 以查找 SFML 库，并在编译结束时将其链接到我们的程序。为此，请执行以下步骤：

1.  转到 **项目** | **构建选项** 并在根级别选择您的项目（不是调试或发布）。

1.  转到 **搜索目录**。在这里，我们必须添加编译器和链接器可以找到 SFML 的路径。

1.  对于编译器，添加您的 SFML 文件夹。

1.  对于链接器，添加 `build/lib` 文件夹，如下所示：![Code::Blocks 和 SFML](img/8477OS_01_09.jpg)

现在我们需要让链接器知道我们的项目需要哪些库。我们所有的未来 SFML 项目都需要系统、窗口和图形模块，因此我们将添加它们：

1.  转到 **链接器设置** 选项卡。

1.  在 **其他链接器选项** 列表中添加 `-lsfml-system`、`-lsfml-window` 和 `-lsfml-graphics`。

1.  现在点击 **确定**。![Code::Blocks 和 SFML](img/8477OS_01_10.jpg)

好消息，所有的配置现在都已经完成。我们最终可能需要在链接器中添加一个库（音频、网络），但仅此而已。

## 一个最小示例

现在是我们用一个非常基本的示例来测试 SFML 的时候了。这个应用程序将展示如下截图中的窗口：

![一个最小示例](img/8477OS_01_11.jpg)

以下代码片段生成了这个窗口：

```cpp
int main(int argc,char* argv[])
{
    sf::RenderWindow window(sf::VideoMode(400, 
400),"01_Introduction");
    window.setFramerateLimit(60);

    //create a circle
    sf::CircleShape circle(150);
    circle.setFillColor(sf::Color::Blue);
    circle.setPosition(10, 20);

    //game loop
    while (window.isOpen())
    {
       //manage the events
        sf::Event event;
        while(window.pollEvent(event))
        {
            if ((event.type == sf::Event::Closed)
                or (event.type == sf::Event::KeyPressed and 
event.key.code == sf::Keyboard::Escape))
                window.close(); //close the window
        }
        window.clear(); //clear the windows to black
        window.draw(circle); //draw the circle
        window.display(); //display the result on screen
    }
    return 0;
}
```

这个应用程序所做的一切就是创建一个宽度和高度为 400 像素的窗口，其标题为 `01_Introduction`。然后创建一个半径为 150 像素的蓝色圆圈，并在窗口打开时绘制。最后，在每次循环中检查用户事件。在这里，我们验证是否请求了关闭事件（关闭按钮或点击 *Alt* + *F4*），或者用户是否按下了键盘上的 *Esc* 按钮。在两种情况下，我们都会关闭窗口，这将导致程序退出。

# 摘要

在本章中，我们讨论了我们将使用哪些技术以及为什么使用它们。我们还学习了在不同环境中安装 C++11 编译器，了解了如何安装 CMake 以及它将如何帮助我们构建本书中的 SFML 项目。然后我们安装了 SFML 2.2，并继续构建了一个非常基本的 SFML 应用程序。

在下一章中，我们将了解如何构建游戏结构，管理用户输入，并跟踪我们的资源。
