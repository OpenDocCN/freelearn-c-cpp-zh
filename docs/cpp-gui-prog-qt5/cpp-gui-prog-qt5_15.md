# 第十五章：跨平台开发

自从第一次发布以来，Qt 就以其跨平台能力而闻名。这也是创始人在决定创建这个框架时的主要目标之一，早在它被**诺基亚**和后来的**Qt 公司**接管之前。

在本章中，我们将涵盖以下主题：

+   编译器

+   构建设置

+   部署到 PC 平台

+   部署到移动平台

让我们开始吧。

# 了解编译器

在本章中，我们将学习从 Qt 项目生成可执行文件的过程。这个过程就是我们所谓的**编译**或**构建**。用于此目的的工具称为**编译器**。在接下来的部分中，我们将学习编译器是什么，以及如何使用它为我们的 Qt 项目生成可执行文件。

# 什么是编译器？

当我们开发一个应用程序时，无论是使用 Qt 还是其他任何软件开发工具包，我们经常需要将项目编译成可执行文件，但实际上在我们编译项目时到底发生了什么呢？

**编译器**是一种软件，它将用高级编程语言编写的计算机代码或计算机指令转换为计算机可以读取和执行的机器代码或较低级别形式。这种低级机器代码在操作系统和计算机处理器上都有很大的不同，但你不必担心，因为编译器会为你转换它。

这意味着你只需要担心用人类可读的编程语言编写逻辑代码，让编译器为你完成工作。理论上，通过使用不同的编译器，你应该能够将代码编译成可在不同操作系统和硬件上运行的可执行程序。我在这里使用“理论上”这个词是因为实际上要比使用不同的编译器更困难，你可能还需要实现支持目标平台的库。然而，Qt 已经为你处理了所有这些，所以你不必做额外的工作。

在当前版本中，Qt 支持以下编译器：

+   **GNU 编译器集合（GCC）**：GCC 是用于 Linux 和 macOS 的编译器

+   **MinGW（Windows 的最小 GNU）**：MinGW 是 GCC 和 GNU Binutils（二进制工具）的本地软件端口，用于在 Windows 上开发应用程序

+   **Microsoft Visual C++（MSVC）**：Qt 支持 MSVC 2013、2015 和 2017 用于构建 Windows 应用程序

+   **XCode**：XCode 是开发者为 macOS 和 iOS 开发应用程序时使用的主要编译器

+   **Linux ICC（英特尔 C++编译器）**：Linux ICC 是英特尔为 Linux 应用程序开发开发的一组 C 和 C++编译器

+   **Clang**：Clang 是 LLVM 编译器的 C、C++、Objective C 和 Objective C++前端，适用于 Windows、Linux 和 macOS

+   **Nim**：Nim 是适用于 Windows、Linux 和 macOS 的 Nim 编译器

+   **QCC**：QCC 是用于在 QNX 操作系统上编译 C++应用程序的接口

# 使用 Make 进行构建自动化

在软件开发中，**Make**是一种构建自动化工具，它通过读取名为**Makefiles**的配置文件自动从源代码构建可执行程序和库，这些配置文件指定如何生成目标平台。简而言之，Make 程序生成构建配置文件，并使用它们告诉编译器在生成最终可执行程序之前要做什么。

Qt 支持两种类型的 Make 程序：

+   **qmake**：它是 Qt 团队开发的本地 Make 程序。它在 Qt Creator 上效果最好，我强烈建议在所有 Qt 项目中使用它。

+   **CMake**：另一方面，尽管这是一个非常强大的构建系统，但它并不像 qmake 那样专门为 Qt 项目做所有事情，比如：

+   运行**元对象编译器**（**MOC**）

+   告诉编译器在哪里查找 Qt 头文件

+   告诉链接器在哪里查找 Qt 库

在 CMake 上手动执行上述步骤，以便成功编译 Qt 项目。只有在以下情况下才应使用 CMake：

+   您正在处理一个非 Qt 项目，但希望使用 Qt Creator 编写代码

+   您正在处理一个需要复杂配置的大型项目，而 qmake 无法处理

+   您真的很喜欢使用 CMake，并且您确切地知道自己在做什么

在选择适合项目的正确工具时，Qt 真的非常灵活。它不仅限于自己的构建系统和编译器。它给开发人员自由选择最适合其项目的工具。

# 构建设置

在项目编译或构建之前，编译器需要在继续之前了解一些细节。这些细节被称为**构建设置**，是编译过程中非常重要的一个方面。在接下来的部分中，我们将学习构建设置是什么，以及如何以准确的方式配置它们。

# Qt 项目（.pro）文件

我相信您已经了解**Qt 项目文件**，因为我们在整本书中已经提到了无数次。`.pro`文件实际上是*qmake*用来构建应用程序、库或插件的项目文件。它包含了所有信息，例如链接到头文件和源文件，项目所需的库，不同平台/环境的自定义构建过程等。一个简单的项目文件可能如下所示：

```cpp
QT += core gui widgets 

TARGET = MyApp 
TEMPLATE = app 

SOURCES +=  
        main.cpp  
        mainwindow.cpp 

HEADERS +=  
        mainwindow.h 

FORMS +=  
        mainwindow.ui 

RESOURCES +=  
    resource.qrc 
```

它只是告诉 qmake 应该在项目中包含哪些 Qt 模块，可执行程序的名称是什么，应用程序的类型是什么，最后是需要包含在项目中的头文件、源文件、表单声明文件和资源文件的链接。所有这些信息对于 qmake 生成配置文件并成功构建应用程序至关重要。对于更复杂的项目，您可能希望为不同的操作系统不同地配置项目。在 Qt 项目文件中也可以轻松实现这一点。

要了解如何为不同的操作系统配置项目，请参阅以下链接：[`doc.qt.io/qt-5/qmake-language.html#scopes-and-conditions.`](http://doc.qt.io/qt-5/qmake-language.html#scopes-and-conditions)

# 评论

您可以在项目文件中添加自己的注释，以提醒自己添加特定配置行的目的，这样您在一段时间不接触后就不会忘记为什么添加了一行。注释以井号（`#`）开头，之后您可以写任何内容，因为构建系统将简单地忽略整行文本。例如：

```cpp
# The following define makes your compiler emit warnings if you use 
# any feature of Qt which has been marked as deprecated (the exact warnings 
# depend on your compiler). Please consult the documentation of the 
# deprecated API in order to know how to port your code away from it. 
DEFINES += QT_DEPRECATED_WARNINGS 
```

您还可以添加虚线或使用空格使您的评论脱颖而出：

```cpp
#------------------------------------------------- 
# 
# Project created by QtCreator 2018-02-18T01:59:44 
# 
#------------------------------------------------- 
```

# 模块、配置和定义

您可以向项目添加不同的 Qt 模块、配置选项和定义。让我们看看我们如何实现这些。要添加额外的模块，只需在`QT +=`后面添加`module`关键字，如下所示：

```cpp
QT += core gui sql printsupport charts multimedia 
```

或者您还可以在前面添加条件来确定何时向项目添加特定模块：

```cpp
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets 
```

您还可以向项目添加配置设置。例如，我们希望明确要求编译器在编译我们的项目时遵循 C++规范的 2011 版本（称为 C++11），并使其成为多线程应用程序：

```cpp
CONFIG += qt c++11 thread
```

您必须使用`+=`，而不是`=`，否则 qmake 将无法使用 Qt 的配置来确定项目所需的设置。或者，您也可以使用`-=`来从项目中删除模块、配置和定义。

至于向编译器添加定义（或变量），我们使用`DEFINES`关键字，如下所示：

```cpp
DEFINES += QT_DEPRECATED_WARNINGS 
```

在编译项目之前，qmake 将此变量的值作为编译器 C 预处理宏（`-D`选项）添加到项目中。前面的定义告诉 Qt 编译器，如果您使用了已标记为弃用的 Qt 功能，则会发出警告。

# 特定于平台的设置

您可以为不同的平台设置不同的配置或设置，因为并非每个设置都适用于所有用例。例如，如果我们想为不同的操作系统包含不同的头文件路径，可以执行以下操作：

```cpp
win32:INCLUDEPATH += "C:/mylibs/extra headers" 
unix:INCLUDEPATH += "/home/user/extra headers" 
```

或者，您还可以将设置放在花括号中，这类似于编程语言中的`if`语句：

```cpp
win32 { 
    SOURCES += extra_code.cpp 
} 
```

您可以通过访问以下链接查看项目文件中可以使用的所有设置：[`doc.qt.io/qt-5/qmake-variable-reference.html.`](http://doc.qt.io/qt-5/qmake-variable-reference.html)

# 部署到 PC 平台

让我们继续学习如何在 Windows、Linux 和 macOS 等平台上部署我们的应用程序。

# Windows

在本节中，我们将学习如何将我们的应用程序部署到不同的操作系统。尽管 Qt 默认支持所有主要平台，但可能需要设置一些配置，以便使您的应用程序能够轻松部署到所有平台。

我们将要介绍的第一个操作系统是最常见的**Microsoft Windows**。

从 Qt 5.6 开始，Qt 不再支持**Windows XP**。

在您尝试部署的 Windows 版本上可能有某些插件无法正常工作，因此在决定处理项目之前，请查看文档。但可以肯定的是，大多数功能在 Qt 上都可以直接使用。

默认情况下，当您将 Qt 安装到 Windows PC 时，**MinGW** 32 位编译器会一起安装。不幸的是，除非您从源代码编译 Qt，否则默认情况下不支持 64 位。如果您需要构建 64 位应用程序，可以考虑在**Microsoft Visual Studio**旁边安装 MSVC 版本的 Qt。可以从以下链接免费获取 Microsoft Visual Studio：[`www.visualstudio.com/vs`](https://www.visualstudio.com/vs)。

您可以通过转到 Tools | Options，然后转到 Build & Run 类别并选择 Kits 选项卡，在 Qt Creator 中设置编译器设置：

![](img/4061d513-9767-4724-9382-a1c09d727cc1.png)

如您所见，有多个运行在不同编译器上的工具包，您可以进行配置。默认情况下，Qt 已经配备了五个工具包——一个用于 Android，一个用于 MinGW，三个用于 MSVC（版本 2013、2015 和 2017）。Qt 将自动检测这些编译器的存在，并相应地为您配置这些设置。

如果您尚未安装**Visual Studio**或**Android SDK**，则在工具包选项前会出现带有感叹号的红色图标。安装所需的编译器后，请尝试重新启动 Qt Creator。它现在将检测到新安装的编译器。您应该可以毫无问题地为 Windows 平台编译 Qt 将为您处理其余部分。我们将在另一节中更多地讨论 Android 平台。

编译应用程序后，打开安装 Qt 的文件夹。将相关的 DLL 文件复制到应用程序文件夹中，并在分发给用户之前将其打包在一起。没有这些 DLL 文件，用户可能无法运行 Qt 应用程序。

有关更多信息，请访问以下链接：[`doc.qt.io/qt-5/windows-deployment.html.`](http://doc.qt.io/qt-5/windows-deployment.html)

要为应用程序设置自定义图标，必须将以下代码添加到项目（`.pro`）文件中：

```cpp
win32:RC_ICONS = myappico.ico 
```

前面的代码仅适用于 Windows 平台，因此我们必须在其前面添加`win32`关键字。

# Linux

**Linux**（或 GNU/Linux）通常被认为是主导云/服务器市场的主要操作系统。由于 Linux 不是单一操作系统（Linux 以不完全兼容的不同 Linux 发行版的形式由不同供应商提供），就像 Windows 或 macOS 一样，开发人员很难构建他们的应用程序并期望它们在不同的 Linux 发行版（**distros**）上无缝运行。但是，如果您在 Qt 上开发 Linux 应用程序，只要目标系统上存在 Qt 库，它就有很高的机会在大多数发行版上运行，如果不是所有主要发行版。

在 Linux 上的默认套件选择比 Windows 简单得多。由于 64 位应用程序已经成为大多数 Linux 发行版的主流和标准已经有一段时间了，我们在安装 Qt 时只需要包括**GCC** 64 位编译器。还有一个 Android 选项，但我们稍后会详细讨论：

![](img/39cbb752-788b-4f19-be8e-80a6fe79aecb.png)

如果您是第一次在 Qt Creator 上编译 Linux 应用程序，我相当肯定您会收到以下错误：

![](img/8cb011d6-5884-41c7-9359-949a3da431c1.png)

这是因为您尚未安装构建 Linux 应用程序所需的相关工具，例如 Make、GCC 和其他程序。

不同的 Linux 发行版安装程序的方法略有不同，但我不会在这里解释每一个。在我的情况下，我使用的是 Ubuntu 发行版，所以我首先打开终端并键入以下命令来安装包含 Make 和 GCC 的`build-essential`软件包：

```cpp
sudo apt-get install build-essential 
```

前面的命令仅适用于继承自**Debian**和**Ubuntu**的发行版，可能不适用于其他发行版，如**Fedora**、**Gentoo**、**Slackware**等。您应该搜索您的 Linux 发行版使用的适当命令来安装这些软件包，如下图所示：

![](img/ca215015-a2a1-4372-bc16-b295382402fb.png)

一旦安装了适当的软件包，请重新启动 Qt Creator 并转到工具|选项。然后，转到“构建和运行”类别，打开“套件”选项卡。现在，您应该能够为您的桌面套件选择 C 和 C ++选项的编译器：

![](img/d8efa23d-958b-49c4-a0af-9d9ab6ef91b5.png)

但是，当您再次尝试编译时，可能会遇到另一个错误，即找不到-lGL：

![](img/3fa99b4a-7ff3-4fbb-a194-4fadbb2b8cc1.png)

这是因为 Qt 试图寻找`OpenGL`库，但在您的系统上找不到它们。通过使用以下命令安装`Mesa 开发`库软件包，可以轻松解决这个问题：

```cpp
sudo apt-get install libgl1-mesa-dev 
```

同样，前面的命令仅适用于 Debian 和 Ubuntu 变体。如果您没有运行 Debian 或 Ubuntu 分支之一，请寻找适合您的 Linux 发行版的命令：

![](img/469afd43-8058-490f-87dd-f89d97955bd7.png)

安装了软件包后，您应该能够编译和运行 Qt 应用程序而无任何问题：

![](img/f90b95a8-a587-4cfd-8c60-c291a40177ae.png)

至于使用其他不太流行的编译器，如**Linux ICC**、**Nim**或**QCC**，您必须通过单击位于 Kits 界面右侧的“添加”按钮来手动设置，然后输入所有适当的设置以使其正常工作。大多数人不使用这些编译器，所以我们暂时跳过它们。

在分发 Linux 应用程序时，比 Windows 或 macOS 要复杂得多。这是因为 Linux 不是单一操作系统，而是一堆具有自己依赖项和配置的不同发行版，这使得分发程序非常困难。

最安全的方法是静态编译程序，这有其优缺点。您的程序将变得非常庞大，这使得对于互联网连接速度较慢的用户来说，更新软件将成为一个巨大的负担。除此之外，如果您不是在进行开源项目并且没有 Qt 商业许可证，Qt 许可证也禁止您进行静态构建。要了解有关 Qt 许可选项的更多信息，请访问以下链接：[`www1.qt.io/licensing-comparison`](https://www1.qt.io/licensing-comparison)

另一种方法是要求用户在运行应用程序之前安装正确版本的 Qt，但这将在用户端产生大量问题，因为并非每个用户都非常精通技术，并且有耐心去避免依赖地狱。

因此，最好的方法是将 Qt 库与应用程序一起分发，就像我们在 Windows 平台上所做的那样。该库可能在某些 Linux 发行版上无法工作（很少见，但有一点可能性），但可以通过为不同的发行版创建不同的安装程序来轻松克服这个问题，现在每个人都很满意。

然而，出于安全原因，Linux 应用程序通常不会默认在其本地目录中查找其依赖项。您必须在您的 qmake 项目（.pro）文件中使用可执行文件的`rpath`设置中的`$ORIGIN`关键字：

```cpp
unix:!mac{ 
QMAKE_LFLAGS += -Wl,--rpath=$$ORIGIN 
QMAKE_RPATH= 
} 
```

设置`QMAKE_RPATH`会清除 Qt 库的默认`rpath`设置。这允许将 Qt 库与应用程序捆绑在一起。如果要将`rpath`包括在 Qt 库的路径中，就不要设置`QMAKE_RPATH`。

之后，只需将 Qt 安装文件夹中的所有库文件复制到应用程序的文件夹中，并从文件名中删除其次版本号。例如，将`libQtCore.so.5.8.1`重命名为`libQtCore.so.5`，现在应该能够被您的 Linux 应用程序检测到。

至于应用程序图标，默认情况下无法为 Linux 应用程序应用任何图标，因为不受支持。尽管某些桌面环境（如 KDE 和 GNOME）支持应用程序图标，但必须手动安装和配置图标，这对用户来说并不是很方便。它甚至可能在某些用户的 PC 上无法工作，因为每个发行版的工作方式都有些不同。为应用程序设置图标的最佳方法是在安装过程中创建桌面快捷方式（符号链接）并将图标应用于快捷方式。

# macOS

在我看来，**macOS**是软件世界中最集中的操作系统。它不仅设计为仅在 Macintosh 机器上运行，您还需要从 Apple 应用商店下载或购买软件。

毫无疑问，这对一些关心选择自由的人造成了不安，但另一方面，这也意味着开发人员在构建和分发应用程序时遇到的问题更少。

除此之外，macOS 应用程序的行为与 ZIP 存档非常相似，每个应用程序都有自己的目录，其中包含适当的库。因此，用户无需预先在其操作系统上安装 Qt 库，一切都可以直接使用。

至于 Kit Selection，Qt for macOS 支持 Android、clang 64 位、iOS 和 iOS 模拟器的工具包：

![](img/735061cd-1346-48d1-bc11-915a94b1f452.png)

从 Qt 5.10 及更高版本开始，Qt 不再支持 macOS 的 32 位构建。此外，Qt 不支持 PowerPC 上的 OS X；由于 Qt 在内部使用 Cocoa，因此也不可能构建 Carbon，请注意这一点。

在编译您的 macOS 应用程序之前，请先从 App Store 安装 Xcode。Xcode 是 macOS 的集成开发环境，包含了由苹果开发的一套用于开发 macOS 和 iOS 软件的软件开发工具。一旦安装了 Xcode，Qt Creator 将检测到其存在，并自动为您设置编译器设置，这非常棒：

![](img/5998a25c-8025-4cab-a703-0ebc982da29c.png)

编译项目后，生成的可执行程序是一个单个的应用程序包，可以轻松地分发给用户。由于所有库文件都打包在应用程序包中，因此它应该可以在用户的 PC 上直接运行。

为 Mac 设置应用程序图标是一项非常简单的任务。只需将以下代码添加到您的项目（`.pro`）文件中，我们就可以开始了：

```cpp
ICON = myapp.icns 
```

请注意，图标格式为`.icns`，而不是我们通常用于 Windows 的`.ico`。

# 在移动平台上部署

除了 Windows、Linux 和 macOS 等平台外，移动平台同样重要。许多开发人员希望将他们的应用程序部署到移动平台。让我们看看如何做到这一点。我们将涵盖两个主要平台，即 iOS 和 Android。

# iOS

在 iOS 上部署 Qt 应用程序非常简单。就像我们之前为 macOS 所做的那样，您需要首先在开发 PC 上安装 Xcode：

![](img/3be1b23b-4409-4291-8b3a-50b98101ecde.png)

然后，重新启动 Qt Creator。它现在应该能够检测到 Xcode 的存在，并且会自动为您设置编译器设置：

![](img/3a8e54b9-b260-4f38-8b25-33541d29e9a0.png)

之后，只需将 iPhone 连接并点击运行按钮！

在 Qt 上构建 iOS 应用程序确实很容易。然而，分发它们并不容易。这是因为 iOS 就像一个有围墙的花园一样，是一个非常封闭的生态系统。您不仅需要在 Apple 注册为应用程序开发人员，还需要在能够将其分发给用户之前对 iOS 应用程序进行代码签名。如果您想为 iOS 构建应用程序，您无法避开这些步骤。

您可以通过访问以下链接了解更多信息：[`developer.apple.com/app-store/submissions.`](https://developer.apple.com/app-store/submissions)

# Android

尽管 Android 是基于 Linux 的操作系统，但与您在 PC 上运行的 Linux 平台相比，它非常不同。要在 Qt 上构建 Android 应用程序，无论您是在 Windows、Linux 还是 macOS 上运行，都必须先将**Android SDK**、**Android NDK**和**Apache ANT**安装到开发 PC 上：

![](img/c84e6e83-7ac5-46fc-b538-4f7013df7fa0.png)

这三个软件包在构建 Qt 上的 Android 应用程序时至关重要。一旦它们都安装好了，重新启动 Qt Creator，它应该已经检测到它们的存在，并且构建设置现在应该已经自动设置好了：

![](img/670eda7a-b1cb-49ef-80f4-e32d4bf20ced.png)

最后，您可以通过使用 Qt Creator 打开`AndroidManifect.xml`文件来配置您的 Android 应用程序：

![](img/d5039d76-b634-4e6b-9116-3b4ef38c82c0.png)

您可以在这里设置一切，如包名称、版本代码、SDK 版本、应用程序图标、权限等。

与 iOS 相比，Android 是一个开放的系统，因此在将应用程序分发给用户之前，您无需做任何事情。但是，如果您希望在 Google Play 商店上分发您的应用程序，可以选择注册为 Google Play 开发人员。

# 总结

在本章中，我们已经学习了如何为不同平台（如 Windows、Linux、macOS、Android 和 iOS）编译和分发我们的 Qt 应用程序。在下一章中，我们将学习不同的调试方法，这可以节省开发时间。让我们来看看吧！
