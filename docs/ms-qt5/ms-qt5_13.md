# 第十三章。全部打包，准备部署

在上一章中，您学习了如何创建具有单元测试的健壮应用程序。应用程序的最终步骤是打包。Qt 框架使您能够开发跨平台应用程序，但打包实际上是一个特定于平台的任务。此外，当您的应用程序准备发货时，您需要一个一步到位的流程来生成和打包您的应用程序。

在本章中，我们将重用画廊应用程序（包括桌面和移动平台）来学习打包 Qt 应用程序所需的步骤。准备应用程序打包的方法有很多。在本章中，我们想要打包画廊应用程序，从 第四章，*征服桌面 UI* 和 第五章，*主宰移动 UI* 在支持的平台上（Windows、Linux、Mac、Android 和 iOS）。

本章涵盖了以下主题：

+   在 Windows 上打包 Qt 应用程序

+   在 Linux 上打包 Qt 应用程序

+   在 Mac 上打包 Qt 应用程序

+   在 Android 上打包 Qt 应用程序

+   在 iOS 上打包 Qt 应用程序

# 打包您的应用程序

您将为每个平台创建一个专门的脚本，以执行构建独立应用程序所需的所有任务。根据操作系统类型，打包的应用程序将是 `gallery-desktop` 或 `gallery-mobile`。因为整个画廊项目必须编译，所以它还必须包含 `gallery-core`。因此，我们将创建一个包含 `gallery-core`、`gallery-desktop` 和 `gallery-mobile` 的父项目。

对于每个平台，我们将准备要打包的项目并创建一个特定的脚本。所有脚本遵循相同的流程：

1.  设置输入和输出目录。

1.  使用 `qmake` 创建 Makefiles。

1.  构建项目。

1.  仅在输出目录中重新组合必要的文件。

1.  使用平台特定的任务打包应用程序。

1.  将打包的应用程序存储在输出目录中。

这些脚本可以在开发计算机或运行类似 Jenkins 等软件的持续集成服务器上运行，只要打包计算机的操作系统与脚本目标操作系统相匹配（除了移动平台）。换句话说，您需要在运行 Windows 的计算机上运行 Windows 脚本，才能为 Windows 打包 Qt 应用程序。

技术上，您可以进行交叉编译（给定适当的工具链和库），但这超出了本书的范围。当您在 Linux 上交叉编译 RaspberryPI 时，这很容易，但当您想在 Windows 上编译 MacOS 时，情况就不同了。

### 注意

从 Linux，您可以使用 MXE 等工具在 [`mxe.cc/`](http://mxe.cc/) 上交叉编译 Qt。

创建一个名为 `ch13-gallery-packaging` 的新子目录项目，具有以下层次结构：

+   `ch13-gallery-packaging`:

    +   `gallery-core`

    +   `gallery-desktop`

    +   `gallery-mobile`

即使您现在是 Qt 子目录项目的专家，这里也有 `ch13-gallery-packaging.pro` 文件：

```cpp
TEMPLATE = subdirs 

SUBDIRS += \ 
    gallery-core \ 
    gallery-desktop \ 
    gallery-mobile 

gallery-desktop.depends = gallery-core 
gallery-mobile.depends = gallery-core 

```

您现在可以开始处理以下任何部分，具体取决于您要针对的平台。

# Windows 的打包

要在 Windows 上打包独立应用程序，您需要提供可执行文件的所有依赖项。`gallery-core.dll` 文件、Qt 库（例如，`Qt5Core.dll`）和特定编译器的库（例如，`libstdc++-6.dll`）是我们可执行文件所需的一些依赖项示例。如果您忘记提供库，则在运行 `gallery-desktop.exe` 程序时将显示错误。

### 注意

在 Windows 上，您可以使用实用工具依赖关系查看器 (`depends`)。它将为您提供应用程序所需的所有库的列表。您可以从这里下载：[www.dependencywalker.com](http://www.dependencywalker.com)。

对于本节，我们将创建一个脚本，通过命令行界面构建项目。然后我们将使用 Qt 工具 `windeployqt` 收集应用程序所需的所有依赖项。此示例适用于 MinGW 编译器，但您可以轻松地将其适应 MSVC 编译器。

以下是 `winqtdeploy` 收集的所需文件和文件夹列表，以便在 Windows 上正确运行 `gallery-desktop`：

+   `iconengines`:

    +   `qsvgicon.dll`

+   `imageformats`:

    +   `qjpeg.dll`

    +   `qwbmp.dll`

    +   `...`

+   `Platforms`:

    +   `qwindows.dll`

+   `translations`:

    +   `qt_en.qm`

    +   `qt_fr.qm`

    +   `...`

+   `D3Dcompiler_47.dll`

+   `gallery-core.dll`

+   `gallery-desktop.exe`

+   `libEGL.dll`

+   `libgcc_s_dw2-1.dll`

+   `libGLESV2.dll`

+   `libstdc++-6.dll`

+   `libwinpthread-1.dll`

+   `opengl32sw.dll`

+   `Qt5Core.dll`

+   `Qt5Gui.dll`

+   `Qt5Svg.dll`

+   `Qt5Widgets.dll`

检查您的环境变量是否设置正确：

![Windows 的打包](img/image00446.jpeg)

在 `scripts` 目录中创建一个名为 `package-windows.bat` 的文件：

```cpp
@ECHO off 

set DIST_DIR=dist\desktop-windows 
set BUILD_DIR=build 
set OUT_DIR=gallery 

mkdir %DIST_DIR% && pushd %DIST_DIR% 
mkdir %BUILD_DIR% %OUT_DIR% 

pushd %BUILD_DIR% 
%QTDIR%\bin\qmake.exe ^ 
  -spec win32-g++ ^ 
  "CONFIG += release" ^ 
  ..\..\..\ch13-gallery-packaging.pro 

%MINGWROOT%\bin\mingw32-make.exe qmake_all 

pushd gallery-core 
%MINGWROOT%\bin\mingw32-make.exe && popd 

pushd gallery-desktop 
%MINGWROOT%\bin\mingw32-make.exe && popd 

popd 
copy %BUILD_DIR%\gallery-core\release\gallery-core.dll %OUT_DIR% 
copy %BUILD_DIR%\gallery-desktop\release\gallery-desktop.exe %OUT_DIR% 
%QTDIR%\bin\windeployqt %OUT_DIR%\gallery-desktop.exe %OUT_DIR%\gallery-core.dll 

popd 

```

让我们讨论一下执行步骤：

1.  设置主路径变量。输出目录是 `DIST_DIR`。所有文件都在 `dist/desktop-windows/build` 目录中生成。

1.  创建所有目录并启动 `dist/desktop-windows/build`。

1.  在 Win32 平台上以发布模式执行 `qmake` 以生成父项目 `Makefile`。`win32-g++` 规范适用于 MinGW 编译器。如果您想使用 MSVC 编译器，应使用 `win32-msvc` 规范。

1.  运行 `mingw32-make qmake_all` 命令以生成子项目的 Makefile。如果您使用 MSVC 编译器，必须将 `mingw32-make` 替换为 `nmake` 或 `jom`。

1.  执行 `mingw32-make` 命令以构建每个所需的子项目。

1.  将生成的文件 `gallery-desktop.exe` 和 `gallery-core.dll` 复制到 `gallery` 目录。

1.  在两个文件上调用 Qt 工具 `windeployqt` 并复制所有必需的依赖项（例如，`Qt5Core.dll`、`Qt5Sql.dll`、`libstdc++-6.dll`、`qwindows.dll` 等）。

# 使用发行版包的 Linux 打包

为 Linux 发行版打包应用程序是一条坎坷的道路。因为每个发行版都可以有自己的打包格式（`.deb`、`.rpm` 等），首先要回答的问题是：你希望针对哪个发行版？涵盖每一个主要的打包格式需要几章内容。甚至详细说明一个单一的发行版也可能是不公平的（你想要为 RHEL 打包？很遗憾，我们只覆盖了 Arch Linux！）。毕竟，从 Qt 应用程序开发者的角度来看，你想要的是将你的产品发送给你的用户，你（目前）并不打算成为官方 Debian 仓库维护者。

考虑到所有这些，我们决定专注于一个为你为每个分发打包应用程序的工具。没错，你不需要学习 Debian 或 Red Hat 的内部结构！我们仍然会解释打包系统中的共同原则，而不会过度详细。

对于我们的目的，我们将演示如何在 Ubuntu 机器上使用 `.deb` 格式进行打包，但正如你将看到的，它可以很容易地更新以生成 `.rpm`。

我们将要使用的工具名为 `fpm`（**eFfing Package Management**）。

### 注意

`fpm` 工具可在 [`github.com/jordansissel/fpm`](https://github.com/jordansissel/fpm) 获取。

`fpm` 工具是一个 Ruby 应用程序，旨在完成我们需要的任务：处理特定于分发的细节并生成最终的包。首先，花时间在你的机器上安装 `fpm` 并确保它正在运行。

简而言之，Linux 打包是一种文件格式，它包含了你想要部署的所有文件以及大量的元数据。它可以包含内容的描述、变更日志、许可文件、依赖项列表、校验和、安装前和安装后触发器等等。

### 注意

如果你想要学习如何手动打包 Debian 二进制文件，请访问 [`tldp.org/HOWTO/html_single/Debian-Binary-Package-Building-HOWTO/`](http://tldp.org/HOWTO/html_single/Debian-Binary-Package-Building-HOWTO/)。

在我们的案例中，我们仍然需要进行一些项目准备，以便 `fpm` 执行其工作。我们想要部署的文件必须与目标文件系统相匹配。以下是部署应该看起来像这样：

+   `gallery-desktop`：这个二进制文件应该部署在 `/usr/bin`

+   `libgallery-core.so`：这个文件应该部署在 `/usr/lib`

为了实现这一点，我们将按照以下方式在 `dist/desktop-linux` 中组织我们的输出：

+   `build` 目录将包含编译后的项目（这是我们发布的影子构建）

+   `root` 目录将包含待打包的文件，即二进制文件和库文件在适当的层次结构中（`usr/bin` 和 `usr/lib`）。

为了生成根目录，我们将依赖 Qt 和 `.pro` 文件的力量。当编译 Qt 项目时，目标文件已经跟踪。我们只需要为 `gallery-core` 和 `gallery-desktop` 添加一个额外的安装目标。

在 `gallery-core/gallery-core.pro` 中添加以下作用域：

```cpp
linux { 
    target.path = $$_PRO_FILE_PWD_/../dist/desktop-linux/root/usr/lib/ 
    INSTALLS += target 
} 

```

在这里，我们定义了一个新的 `target.path`，它将部署 `DISTFILES`（`.so` 文件）到我们期望的根目录。注意使用 `$$_PRO_FILE_PWD_`，它指向当前 `.pro` 文件存储的目录。

在 `gallery-desktop/gallery-desktop.pro` 中执行几乎相同的程序：

```cpp
linux { 
    target.path = $$_PRO_FILE_PWD_/../dist/desktop-linux/root/usr/bin/ 
    INSTALLS += target 
} 

```

通过这些行，当我们调用 make install 时，文件将被部署到 `dist/desktop-linux/root/...`。

现在项目配置已完成，我们可以切换到打包脚本。我们将分两部分介绍脚本：

+   项目编译和 `root` 准备

+   使用 `fpm` 生成 `.deb` 软件包

首先，检查你的环境变量是否设置正确：

![使用发行版包对 Linux 进行打包](img/image00447.jpeg)

使用以下内容创建 `scripts/package-linux-deb.sh`：

```cpp
#!/bin/bash 

DIST_DIR=dist/desktop-linux 
BUILD_DIR=build 
ROOT_DIR=root 

BIN_DIR=$ROOT_DIR/usr/bin 
LIB_DIR=$ROOT_DIR/usr/lib 

mkdir -p $DIST_DIR && cd $DIST_DIR 
mkdir -p $BIN_DIR $LIB_DIR $BUILD_DIR 

pushd $BUILD_DIR 
$QTDIR/bin/qmake \ 
    -spec linux-g++ \ 
    "CONFIG += release" \ 
    ../../../ch13-gallery-packaging.pro 

make qmake_all 
pushd gallery-core && make && make install ; popd 
pushd gallery-desktop && make && make install ; popd 
popd 

```

让我们分解一下：

1.  设置主路径变量。输出目录是 `DIST_DIR`。所有文件都在 `dist/desktop-linux/build` 文件夹中生成。

1.  创建所有目录并启动 `dist/desktop-linux/build`。

1.  在 Linux 平台上以发布模式执行 `qmake` 以生成父项目 `Makefile`。

1.  执行 `make qmake_all` 命令以生成子项目的 Makefile。

1.  执行 `make` 命令来构建每个所需的子项目。

1.  使用 `make install` 命令将二进制文件和库部署到 `dist/desktop-linux/root` 目录。

如果你执行 `scripts/package-linux-deb.sh`，`dist/desktop-linux` 中的最终文件树看起来像这样：

+   `build/`

    +   `gallery-core/*.o`

    +   `gallery-desktop/*.p`

    +   `Makefile`

+   `root/`

    +   `usr/bin/gallery-desktop`

    +   `usr/lib/libgallery-core.so`

现在一切准备就绪，`fpm` 可以工作了。`scripts/package-linux-deb.sh` 的最后一部分包含以下内容：

```cpp
fpm --input-type dir \ 
    --output-type deb \ 
    --force \ 
    --name gallery-desktop \ 
    --version 1.0.0 \ 
    --vendor "Mastering Qt 5" \ 
    --description "A Qt gallery application to organize and manage your pictures in albums" \ 
    --depends qt5-default \ 
    --depends libsqlite3-dev \ 
    --chdir $ROOT_DIR \ 
    --package gallery-desktop_VERSION_ARCH.deb 

```

大多数参数都很明确。我们将重点关注最重要的几个：

+   `--input-type`：此参数表示 `fpm` 将与之交互的内容。它可以接受 `deb`、`rpm`、`gem`、`dir` 等格式，并将其重新包装为另一种格式。在这里，我们使用 `dir` 选项告诉 `fpm` 使用目录树作为输入源。

+   `--output-type`：此参数表示期望的输出类型。查看官方文档以了解支持多少平台。

+   `--name`：这是分配给软件包的名称（如果你想卸载它，你可以写 `apt-get remove gallery-desktop`）。

+   `--depends`：此参数表示项目的库包依赖。你可以添加任意多的依赖。在我们的例子中，我们只依赖于 `qt5 -default` 和 `sqlite3-dev`。此选项非常重要，确保应用程序能够在目标平台上运行。你可以使用 `--depends library >= 1.2.3` 来指定依赖的版本。

+   `--chdir`：此参数表示 `fpm` 将从中运行的基准目录。我们将其设置为 `dist/desktop-linux/root`，我们的文件树已准备好加载！

+   `--package`：此参数是最终软件包的名称。`VERSION` 和 `ARCH` 是占位符，将根据您的系统自动填充。

其余的选项纯粹是信息性的；您可以指定一个变更日志、许可文件等等。只需将 `--output-type` 的 `deb` 更改为 `rpm`，软件包格式就会正确更新。`fpm` 工具还提供了特定的软件包格式选项，让您可以精细控制生成的内容。

如果您现在执行 `scripts/package-linux-deb.sh`，应该会得到一个新的 `dist/desktop-linux/gallery-desktop_1.0.0_amd64.deb` 文件。尝试使用以下命令安装它：

```cpp
sudo dpkg -i  dist/desktop-linux/gallery-desktop_1.0.0_amd64.deb
sudo apt-get install -f

```

第一个命令将在您的系统中部署该软件包。现在您应该拥有文件 `/usr/bin/gallery-desktop` 和 `/usr/lib/libgallery-core.so`。

然而，因为我们使用 `dpkg` 命令安装了软件包，所以依赖项并没有自动安装。如果软件包是由 Debian 仓库提供的（因此，使用 `apt-get install gallery-desktop` 安装软件包），则会这样做。缺失的依赖项仍然“标记”着，`apt-get install -f` 会安装它们。

现在，您可以使用命令 `gallery-desktop` 从系统中的任何位置启动 `gallery-desktop`。当我们于 2016 年编写这一章节时，如果在“全新”的 Ubuntu 上执行它，可能会遇到以下问题：

```cpp
$ gallery-desktop 
gallery-desktop: /usr/lib/x86_64-linux-gnu/libQt5Core.so.5: version `Qt_5.7' not found (required by gallery-desktop)
gallery-desktop: /usr/lib/x86_64-linux-gnu/libQt5Core.so.5: version `Qt_5' not found (required by gallery-desktop)
...
gallery-desktop: /usr/lib/x86_64-linux-gnu/libQt5Core.so.5: version `Qt_5' not found (required by /usr/lib/libgallery-core.so.1)

```

发生了什么？我们使用 `apt-get install -f` 安装了依赖项！在这里，我们遇到了 Linux 软件包管理的一个主要痛点。我们在 `.deb` 文件中指定的依赖项可能指向 Qt 的特定版本，但实际情况是我们依赖于上游维护的软件包版本。换句话说，每当 Qt 发布新版本时，发行版维护者（Ubuntu、Fedora 等等）必须重新打包它，以便在官方仓库中提供。这可能是一个漫长的过程，维护者需要移植大量的软件包！

为了确保我们所说的内容准确无误，让我们使用 `ldd` 命令查看 `gallery-desktop` 的库依赖项：

```cpp
$ ldd /usr/bin/gallery-desktop
 libgallery-core.so.1 => /usr/lib/libgallery-core.so.1 (0x00007f8110775000)
 libQt5Widgets.so.5 => /usr/lib/x86_64-linux-gnu/libQt5Widgets.so.5 (0x00007f81100e8000)
 libQt5Gui.so.5 => /usr/lib/x86_64-linux-gnu/libQt5Gui.so.5 (0x00007f810fb9f000)
 libQt5Core.so.5 => /usr/lib/x86_64-linux-gnu/libQt5Core.so.5 (0x00007f810f6c9000)
 ...
 libXext.so.6 => /usr/lib/x86_64-linux-gnu/libXext.so.6 (0x00007f810966e000)

```

如您所见，`libgallery-core.so` 在 `/usr/lib` 中被正确解析，Qt 的依赖项也在 `/usr/lib/x86_64-linux-gnu` 中。但使用了哪个版本的 Qt 呢？答案在于库的详细信息：

```cpp
$ ll /usr/lib/x86_64-linux-gnu/libQt5Core.*
-rw-r--r-- 1 root root    1014 may    2 15:37 libQt5Core.prl
lrwxrwxrwx 1 root root      19 may    2 15:39 libQt5Core.so -> libQt5Core.so.5.5.1
lrwxrwxrwx 1 root root      19 may    2 15:39 libQt5Core.so.5 -> libQt5Core.so.5.5.1
lrwxrwxrwx 1 root root      19 may    2 15:39 libQt5Core.so.5.5 -> libQt5Core.so.5.5.1
-rw-r--r-- 1 root root 5052920 may    2 15:41 libQt5Core.so.5.5.1

```

`libQt5Core.so` 文件是 `libQt5Core.so.5.5.1` 的软链接，这意味着系统版本的 Qt 是 5.5.1，而 `gallery-desktop` 依赖于 Qt 5.7。您可以配置系统，使系统 Qt 指向您的 Qt 安装（通过 Qt 安装程序完成）。然而，您的客户手动安装 Qt 只为了让 `gallery-desktop` 运行几乎是不可能的。

更糟糕的是，对于较旧的发行版，经过一段时间后，软件包通常根本不会更新。只需尝试在 Ubuntu 14.04 上安装 Qt 5.7 Debian 软件包，就能理解事情变得多么复杂。我们甚至还没有提到不兼容的依赖项。如果我们依赖于特定版本的`libsqlite3-dev`，而另一个应用程序需要另一个版本，事情就会变得很糟糕，只有一个能够幸存。

如果你想要 Linux 官方仓库中有这个软件包或者你有特定的需求，一个 Linux 软件包有很多优点。在 Linux 上安装应用程序通常使用官方仓库，这样你的用户就不会感到困惑。如果你能将 Qt 版本限制在 Linux 发行版上部署的版本，这可能是一个不错的解决方案。

不幸的是，这也带来了巨大的头痛：你需要支持多个发行版，处理依赖关系而不会破坏系统，并确保你的应用程序有足够的旧依赖项，等等。

不要担心，一切还没有失去；聪明的人已经在 Linux 上通过自包含的软件包来解决这个问题。实际上，我们将要介绍一个自包含的软件包。

# 使用 AppImage 对 Linux 进行软件打包

在 Windows 或 Mac 上，一个应用程序是自给自足的：它包含执行所需的所有依赖项。一方面，这造成了更多的文件重复，另一方面也简化了开发者的打包工作。

基于这个前提，人们已经努力在 Linux 上实现相同的模式（与特定仓库/发行版的软件包相反）。今天，有几个解决方案在 Linux 上提供自包含的软件包。我们建议你研究这些解决方案之一：AppImage。这个特定的工具在 Linux 社区中越来越受欢迎。越来越多的开发者依赖 AppImage 来打包和部署他们的应用程序。

AppImage 是一种包含所有库的应用程序的文件格式。你下载一个单一的 AppImage 文件，执行它，就完成了：应用程序正在运行。幕后，AppImage 是一个强化版的 ISO 文件，在你执行时即时挂载。AppImage 文件本身是只读的，也可以在沙盒中运行，例如 Firejail（一个 SUID 沙盒程序，通过限制应用程序的运行环境来降低安全漏洞的风险）。

### 注意

关于 AppImage 的更多信息可以在[`appimage.org/`](http://appimage.org/)找到。

将`gallery-desktop`打包成 AppImage 有两个主要步骤：

1.  收集`gallery-desktop`的所有依赖项。

1.  将`gallery-desktop`及其依赖项打包成 AppImage 格式。

幸运的是，整个流程可以通过使用一个巧妙的小工具来完成：`linuxdeployqt`。它最初是一个爱好项目，后来成为了 AppImage 文档中打包 Qt 应用程序的官方方法。

### 注意

从[`github.com/probonopd/linuxdeployqt/`](https://github.com/probonopd/linuxdeployqt/)获取`linuxdeployqt`。

我们将要编写的脚本假设二进制文件`linuxdeployqt`在您的`$PATH`变量中可用。请确保您的环境变量设置正确：

![使用 AppImage 为 Linux 打包](img/image00448.jpeg)

创建`scripts/package-linux-appimage.sh`并更新如下：

```cpp
#!/bin/bash 

DIST_DIR=dist/desktop-linux 
BUILD_DIR=build 

mkdir -p $DIST_DIR && cd $DIST_DIR 
mkdir -p $BUILD_DIR 

pushd $BUILD_DIR 
$QTDIR/bin/qmake \ 
    -spec linux-g++ \ 
    "CONFIG += release" \ 
    ../../../ch13-gallery-packaging.pro 
make qmake_all 
pushd gallery-core && make ; popd 
pushd gallery-desktop && make ; popd 
popd 

export QT_PLUGIN_PATH=$QTDIR/plugins/ 
export LD_LIBRARY_PATH=$QTDIR/lib:$(pwd)/build/gallery-core 

linuxdeployqt \ 
    build/gallery-desktop/gallery-desktop \ 
    -appimage 

mv build/gallery-desktop.AppImage . 

```

第一部分是项目的编译：

1.  设置主路径变量。输出目录是`DIST_DIR`。所有文件都在`dist/desktop-linux/build`文件夹中生成。

1.  创建所有目录并进入`dist/desktop-linux/build`。

1.  在 Linux 平台上以发布模式执行`qmake`以生成父项目`Makefile`。

1.  运行`make qmake_all`命令以生成子项目的 Makefiles。

1.  执行`make`命令来构建每个所需的子项目。

脚本的第二部分涉及`linuxdeployqt`。我们首先必须导出一些路径，以便`linuxdeployqt`能够正确地找到`gallery-desktop`的所有依赖项（Qt 库和`gallery-core`库）。

之后，我们通过指定要处理的源二进制文件和目标文件类型（AppImage）来执行`linuxdeployqt`。生成的文件是一个单独的`gallery-desktop.AppImage`，无需安装任何 Qt 包即可在用户的计算机上启动！

# Mac OS X 的打包

在 OS X 上，应用程序是通过一个包来构建和运行的：一个包含应用程序二进制文件及其所有依赖项的单个目录。在 Finder 中，这些包被视为`.app`特殊目录。

当从 Qt Creator 运行`gallery-desktop`时，应用程序已经打包在一个`.app`文件中。因为我们使用的是自定义库`gallery-core`，所以这个`gallery-desktop.app`不包含所有依赖项，Qt Creator 会为我们处理。

我们旨在创建一个脚本，将`gallery-desktop`（包括`gallery-core`）完全打包在一个`.dmg`文件中，这是一个 Mac OS X 磁盘映像文件，在执行时挂载，并允许用户轻松安装应用程序。

为了实现这一点，Qt 提供了`macdeployqt`工具，它收集依赖项并创建`.dmg`文件。

首先，检查您的环境变量是否设置正确：

![使用 Mac OS X 打包](img/image00449.jpeg)

创建`scripts/package-macosx.sh`文件，内容如下：

```cpp
#!/bin/bash 

DIST_DIR=dist/desktop-macosx 
BUILD_DIR=build 

mkdir -p $DIST_DIR && cd $DIST_DIR 
mkdir -p $BUILD_DIR 

pushd $BUILD_DIR 
$QTDIR/bin/qmake \ 
  -spec macx-clang \ 
  "CONFIG += release x86_64" \ 
  ../../../ch13-gallery-packaging.pro 
make qmake_all 
pushd gallery-core && make ; popd 
pushd gallery-desktop && make ; popd 

cp gallery-core/*.dylib \ 
    gallery-desktop/gallery-desktop.app/Contents/Frameworks/ 

install_name_tool -change \ 
  libgallery-core.1.dylib \ 
  @rpath/libgallery-core.1.dylib \ 
  gallery-desktop/gallery-desktop.app/Contents/MacOS/gallery-desktop 
popd 

$QTDIR/bin/macdeployqt \ 
    build/gallery-desktop/gallery-desktop.app \ 
    -dmg 

mv build/gallery-desktop/gallery-desktop.dmg . 

```

我们可以将脚本分成两部分。第一部分是为`macdeployqt`准备应用程序：

1.  设置主路径变量。输出目录是`DIST_DIR`。所有文件都在`dist/desktop-macosx/build`文件夹中生成。

1.  创建所有目录并进入`dist/desktop-macosx/build`。

1.  在 Mac OS X 平台上以发布模式执行`qmake`以生成父项目`Makefile`。

1.  运行`make qmake_all`命令以生成子项目的 Makefiles。

1.  执行`make`命令来构建每个所需的子项目。

以下部分包括在生成的`gallery-desktop.app`中的`gallery-core`库。如果我们不执行脚本中提到的`cp`命令及其之后的内容，我们可能会对`gallery-desktop`的二进制内容感到非常惊讶。让我们通过执行以下命令来查看它：

```cpp
$ otool -L dist/desktop-macosx/build/gallery-desktop/gallery-desktop.app/Contents/MacOS/gallery-desktop 
dist/desktop-macosx/build/gallery-desktop/gallery-desktop.app/Contents/MacOS/gallery-desktop:
 libgallery-core.1.dylib (compatibility version 1.0.0, current version 1.0.0)
 @rpath/QtWidgets.framework/Versions/5/QtWidgets (compatibility version 5.7.0, current version 5.7.0)
...
 /usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1226.10.1)

```

如您所见，`libgallery-core.1.dylib`在本地路径中解析，但不在特殊依赖路径中，就像为`QtWidget`使用`@rpath`（即`Contents/Frameworks/`）那样。为了减轻这种情况，`package-macosx.sh`将`.dylib`文件复制到`gallery-desktop.app/Contents/Frameworks/`，并使用`install_name_tool`重新生成二进制文件的依赖项索引。

最后，在`package-macosx.sh`中，使用更新的`gallery-deskop.app`和目标`dmg`格式调用`macdeployqt`。生成的`gallery-desktop.dmg`可以部署到您的用户计算机上。

# Android 打包

本节的目标是为`gallery-mobile`应用程序生成一个独立的 APK 文件。为 Android 打包和部署应用程序需要多个步骤：

1.  配置 Android 构建细节。

1.  生成一个密钥库和一个证书。

1.  从模板中自定义 Android 清单。

1.  创建一个脚本来自动化打包。

您可以直接从 Qt Creator 中完成大多数任务。在底层，Qt 工具`androiddeployqt`被调用以生成 APK 文件。转到**项目** | **armeabi-v7a 的 Android** | **构建步骤**。您应该看到一个特殊的构建步骤：**构建 Android APK**。细节如下截图：

![Android 打包](img/image00450.jpeg)

首件事是选择您想用于生成**应用程序**的 Android API 级别。在我们的案例中，我们选择了**android-23**作为 Android API Level 23。尽量始终使用可用的最新 SDK 版本构建您的应用程序。

要在 Play Store 上发布您的应用程序，您必须对包进行签名。要能够更新应用程序，当前版本和新版本的签名必须相同。此程序是一个保护措施，以确保任何未来的应用程序版本确实是由您创建的。第一次您应该创建密钥库，下次您可以使用**浏览...**按钮重用它。现在，点击**签名包** | **密钥库**行上的**创建...**按钮。您将得到以下弹出窗口：

![Android 打包](img/image00451.jpeg)

按照以下步骤生成新的密钥库：

1.  密钥库必须通过密码进行保护。不要忘记它，否则你将无法为未来的版本使用此密钥库。

1.  为证书指定一个**别名名称**。对于**密钥大小**和**有效期（天）**的默认值是合适的。您可以指定不同的密码用于证书或使用密钥库的密码。

1.  在**证书区分名称**组中，输入有关您和您公司的信息。

1.  将密钥库文件保存在安全的地方。

1.  输入 keystore 密码以验证其选择用于部署。

下一个部分是关于 **Qt 部署**。确实，您的应用程序需要一些 Qt 库。Qt 支持三种部署方式：

+   创建一个依赖 **Ministro** 的最小 APK，Ministro 是一个可以从 Play Store 下载的 Android 应用程序。它充当 Android 上所有 Qt 应用程序的 Qt 共享库安装程序/提供者。

+   创建一个嵌入 Qt 库的独立 **bundle** APK。

+   创建一个依赖于 Qt 库位于特定目录的 APK。在第一次部署期间，库被复制到 **临时目录**。

在开发和调试阶段，您应该选择 **临时目录** 方式以减少打包时间。对于部署，您可以使用 **Ministro** 或 **bundle** 选项。在我们的案例中，我们选择了独立的 bundle 来生成完整的 APK。

**高级操作** 面板提供了三个选项：

+   **使用 Gradle**：此选项生成 Gradle 包装器和脚本，如果您计划在 Android Studio 等 IDE 中自定义 Java 部分，则非常有用。

+   **构建后打开包位置**：此选项将打开由 `androiddeployqt` 生成的包所在的目录。

+   **详细输出**：此选项显示有关 `androiddeployqt` 处理的附加信息。

Android 构建细节和签名选项已完成。我们现在可以自定义 Android 清单。点击 **创建模板**，选择 `gallery-mobile.pro` 文件，然后点击 **完成**。向导为您创建一个包含多个文件的 `android` 子目录；例如，`AndroidManifest.xml`。`gallery-mobile.pro` 文件必须自动更新这些文件。然而，不要忘记添加 `android` 范围，如下面的代码片段所示：

```cpp
TEMPLATE = app 
... 
android { 
    contains(ANDROID_TARGET_ARCH,x86) { 
        ANDROID_EXTRA_LIBS = \ 
            $$[QT_INSTALL_LIBS]/libQt5Sql.so 
    } 

    DISTFILES += \ 
        android/AndroidManifest.xml \ 
        android/gradle/wrapper/gradle-wrapper.jar \ 
        android/gradlew \ 
        android/res/values/libs.xml \ 
        android/build.gradle \ 
        android/gradle/wrapper/gradle-wrapper.properties \ 
        android/gradlew.bat 

    ANDROID_PACKAGE_SOURCE_DIR = $$PWD/android 
} 

```

您现在可以编辑 `AndroidManifest.xml` 文件。Qt Creator 提供了一个专门的编辑器。您也可以小心地使用纯文本编辑器进行编辑。您可以从分层项目视图打开它：**gallery-mobile** | **其他文件** | **android**。

这里是我们在 Qt Creator 中的 Android 清单文件：

![Android 打包](img/image00452.jpeg)

这里是最重要的步骤：

1.  将默认的 **包名** 替换为您自己的。

1.  **版本代码** 是一个整数，必须为每个官方版本增加。

1.  **版本名称** 是用户显示的版本。

1.  选择 **最低要求的 SDK**。使用较旧版本的用户将无法安装您的应用程序。

1.  选择用于编译应用程序的 **目标 SDK** 将使用的 SDK。

1.  更改应用程序和活动名称。

1.  根据屏幕 DPI（每英寸点数）选择 **应用程序图标**。从左到右：低、中、高 DPI 图标。

1.  最后，如果您的应用程序需要，您可以添加一些 Android 权限。

你已经可以从 Qt Creator 中构建和部署你的签名应用程序。你应该能在你的 Android 手机或模拟器上看到新的应用程序名称和图标。然而，我们现在将创建一个脚本，以便从命令行轻松生成和打包签名 APK。

Android 和 Qt 工具以及脚本本身需要几个环境变量。以下是一个带有示例的总结：

![Android 打包](img/image00453.jpeg)

这个例子是一个 bash 脚本，但如果你在 Windows 上，请随意将其适配为 `.bat` 文件。在 `scripts` 目录中创建一个 `package-android.sh` 文件：

```cpp
#!/bin/bash 

DIST_DIR=dist/mobile-android 
BUILD_DIR=build 
APK_DIR=apk 
KEYSTORE_PATH="$(pwd)/scripts/android-data" 
ANDROID_BUILD_PATH="$(pwd)/$DIST_DIR/$BUILD_DIR/android-build" 

mkdir -p $DIST_DIR && cd $DIST_DIR 
mkdir -p $APK_DIR $BUILD_DIR 

pushd $BUILD_DIR 
$QTDIR_ANDROID/bin/qmake \ 
    -spec android-g++ \ 
    "CONFIG += release" \ 
    ../../../ch13-gallery-packaging.pro 
make qmake_all 
pushd gallery-core && make ; popd 
pushd gallery-mobile && make ; popd 
pushd gallery-mobile && make INSTALL_ROOT=$ANDROID_BUILD_PATH install ; popd 

$QTDIR_ANDROID/bin/androiddeployqt 
    --input ./gallery-mobile/android-libgallery-mobile.so-deployment-settings.json \ 
    --output $ANDROID_BUILD_PATH \ 
    --deployment bundled \ 
    --android-platform android-23 \ 
    --jdk $JAVA_HOME \ 
    --ant $ANT_ROOT/ant \ 
    --sign $KEYSTORE_PATH/android.keystore myandroidkey \ 
    --storepass 'masteringqt' 

cp $ANDROID_BUILD_PATH/bin/QtApp-release-signed.apk ../apk/cute-gallery.apk 
popd 

```

让我们一起分析这个脚本：

1.  设置主要路径变量。输出目录是 `DIST_DIR`。所有文件都在 `dist/mobile-android/build` 目录中生成。最终的签名 APK 被复制到 `dist/mobile-android/apk` 目录。

1.  创建所有目录并进入 `dist/mobile-android/build`。

1.  为 Android 平台执行发布模式的 `qmake` 以生成父项目 Makefile。

1.  执行 `make qmake_all` 命令来生成子项目的 Makefiles。

1.  执行 `make` 命令来构建每个所需的子项目。

1.  在 `gallery-mobile` 目录中运行 `make install` 命令，指定 `INSTALL_ROOT` 以复制 APK 生成所需的全部二进制文件和文件。

脚本的最后部分调用 `androiddeployqt` 二进制文件，这是一个用于生成 APK 的 Qt 工具。查看以下选项：

+   这里使用的 `--deployment` 选项是 `bundled`，就像我们在 Qt Creator 中使用的那样。

+   `--sign` 选项需要两个参数：密钥库文件的 URL 和证书的别名。

+   `--storepass` 选项用于指定密钥库密码。在我们的例子中，密码是 "masteringqt"。

最后，生成的签名 APK 被复制到 `dist/mobile-android/apk` 目录，文件名为 `cute-gallery.apk`。

# iOS 打包

为 iOS 打包 Qt 应用程序依赖于 XCode。当你从 Qt Creator 中构建和运行 gallery-mobile 时，XCode 将在后台被调用。最后，生成一个 `.xcodeproj` 文件并将其传递给 XCode。

了解这一点后，打包部分将相当有限：唯一可以自动化的就是 `.xcodeproj` 文件的生成。

首先，检查你的环境变量是否设置正确：

![iOS 打包](img/image00454.jpeg)

创建 `scripts/package-ios.sh` 并将以下片段添加到其中：

```cpp
#!/bin/bash 

DIST_DIR=dist/mobile-ios 
BUILD_DIR=build 

mkdir -p $DIST_DIR && cd $DIST_DIR 
mkdir -p $BIN_DIR $LIB_DIR $BUILD_DIR 

pushd $BUILD_DIR 
$QTDIR_IOS/bin/qmake \ 
  -spec macx-ios-clang \ 
  "CONFIG += release iphoneos device" \ 
  ../../../ch13-gallery-packaging.pro 
make qmake_all 
pushd gallery-core && make ; popd 
pushd gallery-mobile && make ; popd 

popd 

```

脚本执行以下步骤：

1.  设置主要路径变量。输出目录是 `DIST_DIR`。所有文件都在 `dist/mobile-ios/build` 文件夹中生成。

1.  创建所有目录并进入 `dist/mobile-ios/build`。

1.  为 iPhone 设备（与 iPhone 模拟器平台相对）执行发布模式的 `qmake` 以生成父项目 `Makefile`。

1.  执行 `make qmake_all` 命令来生成子项目的 Makefiles。

1.  执行 `make` 命令来构建每个所需的子项目。

一旦执行了这个脚本，`dist/mobile-ios/build/gallery-mobile/gallery-mobile.xcodeproj` 就可以打开在 XCode 中。接下来的步骤完全在 XCode 中完成：

1.  在 XCode 中打开 `gallery-mobile.xcodeproj`。

1.  为 iOS 设备编译应用程序。

1.  按照苹果的流程分发您的应用程序（通过 App Store 或作为独立文件）。

之后，`gallery-mobile` 将为您的用户准备好！

# 摘要

即使您的应用程序在您的电脑上运行良好，您的开发环境也可能影响这种行为。其打包必须正确，以便在用户的硬件上运行您的应用程序。您已经学习了部署应用程序之前所需的打包步骤。某些平台需要遵循特定的任务。如果您的应用程序正在运行独特的脚本，您现在可以制作一个独立的包。

下一章描述了一些在开发 Qt 应用程序时可能有用的技巧。您将学习一些关于 Qt Creator 的提示。
