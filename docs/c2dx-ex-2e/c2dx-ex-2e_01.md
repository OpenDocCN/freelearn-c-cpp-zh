# 第一章. 安装 Cocos2d-x

*在本章中，我们将让你的机器开始运行，以便你能够充分利用本书中的示例。这包括有关下载框架和创建项目的信息，以及 Cocos2d-x 应用程序基本结构的概述。*

*我还会向你推荐一些额外的工具，你可以考虑获取以帮助你的开发过程，例如用于构建精灵表、粒子效果和位图字体的工具。尽管这些工具是可选的，而且你只需通过跟随本书中给出的示例就可以学习如何使用精灵表、粒子效果和位图字体，但你可能仍会考虑这些工具用于你的项目。*

你在本章中将学习的内容如下：

+   如何下载 Cocos2d-x

+   如何运行你的第一个多平台应用程序

+   基本项目的外观以及如何熟悉它

+   如何使用测试项目作为主要参考来源

# 下载和安装 Cocos2d-x

本书中的所有示例都是在 Mac 上使用 Xcode 和/或 Eclipse 开发的。最后一章的示例使用 Cocos2d-x 自带的 IDE 进行脚本编写。虽然你可以使用 Cocos2d-x 在其他平台上使用不同的系统开发你的游戏，但示例是在 Mac 上构建的，并部署到 iOS 和 Android。

Xcode 是免费的，可以从 Mac App Store ([`developer.apple.com/xcode/index.php`](https://developer.apple.com/xcode/index.php)) 下载，但为了在 iOS 设备上测试你的代码并发布你的游戏，你需要一个 Apple 开发者账户，这需要每年支付 99 美元。你可以在他们的网站上找到更多信息：[`developer.apple.com/`](https://developer.apple.com/)。

对于 Android 部署，我建议你从 Google 获取 Eclipse 和 ADT 套件，你可以在 [`developer.android.com/sdk/installing/installing-adt.html`](http://developer.android.com/sdk/installing/installing-adt.html) 找到。你将能够免费在 Android 设备上测试你的游戏。

因此，假设你有互联网连接，让我们开始吧！

# 行动时间 – 下载，下载，下载

我们首先下载必要的 SDK、NDK 和一些通用的小工具：

1.  访问 [`www.cocos2d-x.org/download`](http://www.cocos2d-x.org/download) 并下载 Cocos2d-x 的最新稳定版本。对于本书，我将使用 Cocos2d-x-3.4 版本。

1.  将文件解压缩到你的机器上某个你可以记住的地方。我建议你将我们现在要下载的所有文件都添加到同一个文件夹中。

1.  顺便下载 Code IDE 也一样。我们将在本书的最后一章中使用它。

1.  然后，前往[`developer.android.com/sdk/installing/installing-adt.html`](http://developer.android.com/sdk/installing/installing-adt.html)下载 Eclipse ADT 插件（如果你还没有安装 Eclipse 或 Android SDK，请分别从[`eclipse.org/downloads/`](https://eclipse.org/downloads/)和[`developer.android.com/sdk/installing/index.html?pkg=tools`](http://developer.android.com/sdk/installing/index.html?pkg=tools)下载）。

    ### 注意

    如果你在安装 ADT 插件时遇到任何问题，你可以在[`developer.android.com/sdk/installing/installing-adt.html`](http://developer.android.com/sdk/installing/installing-adt.html)找到完整的说明。

1.  现在，对于 Apache Ant，请访问[`ant.apache.org/bindownload.cgi`](http://ant.apache.org/bindownload.cgi)并查找压缩文件的链接，然后下载`.zip`版本。

1.  最后，前往[`developer.android.com/tools/sdk/ndk/index.html`](https://developer.android.com/tools/sdk/ndk/index.html)下载针对目标系统的最新 NDK 版本。按照同一页上的安装说明提取文件，因为某些系统不允许这些文件自动解压。提醒一下：你必须使用 NDK r8e 以上的版本与 Cocos2d-x 3.x 一起使用。

## *发生了什么？*

你已经成功下载了在机器上设置 Cocos2d-x 和开始开发所需的所有内容。如果你使用的是 Mac，你可能需要更改**系统偏好设置**中的安全设置，以允许 Eclipse 运行。此外，通过转到**窗口-Android SDK 管理器**菜单，在 Eclipse 中打开 Android SDK 管理器，并安装至少版本 2.3.3 的包以及你可能希望针对的任何更高版本。

此外，请确保你的机器上已安装 Python。在终端或命令提示符中，只需输入单词`python`并按回车键。如果你还没有安装，请访问[`www.python.org/`](https://www.python.org/)并按照那里的说明操作。

因此，到这一步结束时，你应该有一个文件夹，其中包含了 Cocos2d-x、CocosIDE、Android SDK、NDK 和 Apache Ant 的所有提取文件。

现在，让我们安装 Cocos2d-x。

# 行动时间 - 安装 Cocos2d-x

打开终端或命令提示符，导航到 Cocos2d-x 提取的文件夹：

1.  你可以通过输入`cd`（即`cd`后跟一个空格）并将文件夹拖到终端窗口中，然后按*Enter*键来完成此操作。在我的机器上，这看起来是这样的：

    ```cpp
    cd /Applications/Dev/cocos2d-x-3.4
    ```

1.  接下来，输入`python setup.py`。

1.  按*Enter*。你将被提示输入 NDK、SDK 和 Apache ANT 根目录的路径。你必须将每个文件夹都拖到终端窗口中，确保删除路径末尾的任何额外空格，然后按*Enter*。所以对于 NDK，我得到：

    ```cpp
    /Applications/Dev/android-ndk-r10c
    ```

1.  接下来，是 SDK 的路径。再次，我将存储在 Eclipse 文件夹中的文件夹拖动：

    ```cpp
    /Applications/eclipse/sdk
    ```

1.  接下来是 ANT 的路径。如果您已经在您的机器上正确安装了它，路径将类似于`usr/local/bin`，设置脚本会为您找到它。否则，您可以使用您下载并解压的版本。只需指向其中的`bin`文件夹：

    ```cpp
    /Applications/Dev/apache-ant-1.9.4/bin
    ```

1.  最后一步是将这些路径添加到您的系统中。按照窗口中的最后一条指令操作：**请执行以下命令："source /Users/YOUR_USER_NAME/.bash_profile" 以使添加的系统变量生效**。您可以将引号内的命令复制，粘贴，然后按*Enter*键。

## *发生了什么？*

您现在已经在您的机器上安装了 Cocos2d-x，并且准备开始使用了。是时候创建我们的第一个项目了！

# Hello-x World-x

让我们创建计算机编程中的老生常谈：`hello world`示例。

# 开始行动 - 创建应用程序

再次打开终端并按照以下简单步骤操作：

1.  您应该已经将 Cocos2d-x 控制台路径添加到您的系统中。您可以通过在终端中使用`cocos`命令来测试这一点。为了创建一个名为`HelloWorld`的新项目，使用 C++作为其主要语言并将其保存在您的桌面上，您需要运行以下命令，将`YOUR_BUNDLE_INDETIFIER`替换为您选择的包名，将`PATH_TO_YOUR_PROJECT`替换为您希望保存项目的路径：

    ```cpp
    cocos new HelloWorld -p YOUR_BUNDLE_IDENTIFIER -l cpp -d PATH_TO_YOUR_PROJECT

    ```

1.  例如，在我的机器上，我输入的行如下：

    ```cpp
    cocos new HelloWorld -p com.rengelbert.HelloWorld -l cpp -d /Users/rengelbert/Desktop/HelloWorld

    ```

    然后按*Enter*。如果您选择不提供目录参数（`-d`），Cocos 控制台将项目保存在`Cocos2d-x`文件夹内。

1.  现在，您可以前往您的桌面或您选择保存项目的任何位置，导航到`HelloWorld`项目中的`proj.ios_mac`文件夹。在该文件夹中，您将找到 Xcode 项目文件。一旦在 Xcode 中打开项目，您就可以点击**运行**按钮，这样就完成了。![开始行动 - 创建应用程序](img/00002.jpeg)

### 注意

当您在 Xcode 中运行**cocos2d-x**应用程序时，程序通常会发布一些关于您的代码或最可能的是框架的警告。这些警告大多会引用已弃用的方法或不符合当前 SDK 更近和更严格规则的语句。但这没关系。尽管这些警告确实令人烦恼，但可以忽略。

## *发生了什么？*

您已经创建了您的第一个 Cocos2d-x 应用程序。命令行上使用的参数是：

+   `-p` 用于包或捆绑标识符

+   `-l` 用于语言，这里您有`cpp`、`lua`或 JavaScript 选项

现在让我们在 Android 上运行这个应用程序。

### 提示

**下载示例代码**

您可以从[`www.packtpub.com`](http://www.packtpub.com)下载您购买的所有 Packt Publishing 书籍的示例代码文件。如果您在其他地方购买了这本书，您可以访问[`www.packtpub.com/support`](http://www.packtpub.com/support)并注册，以便将文件直接通过电子邮件发送给您。

# 开始行动 - 部署到 Android

我们将在 Eclipse 中打开项目：

1.  打开 Eclipse。

1.  我们需要修复 NDK 的路径；在您的系统中，这一步可能是可选的，但在任何情况下，它必须只执行一次。在 Eclipse 中，转到 **Eclipse-首选项**，然后在 **C/C++** 选项下选择 **构建环境**。

1.  您需要添加 NDK 路径，并且它必须命名为 `NDK_ROOT`。为了做到这一点，您必须点击 **添加…**，并使用 `NDK_ROOT` 作为名称，然后点击 **值** 字段以确保鼠标光标在其中处于活动状态，然后拖动您下载的 NDK 文件夹到该字段中。在我的机器上，结果看起来像这样：![部署到 Android 的时间 - 部署到 Android](img/00003.jpeg)

1.  点击 **应用**。重启 Eclipse 可能是个好主意。（如果您在 **首选项** 中看不到 **C/C++** 选项，这意味着您没有安装 CDT 插件。请查找有关如何安装它们的完整说明，请参阅 [`www.eclipse.org/cdt/`](http://www.eclipse.org/cdt/)。）

1.  现在，我们准备将项目导入到 Eclipse 中。选择 **文件** | **导入…**。

1.  在对话框中，选择 **Android** 选项，然后选择 **将现有 Android 代码导入工作空间** 选项并点击 **下一步**：![部署到 Android 的时间 - 部署到 Android](img/00004.jpeg)

1.  点击 **浏览** 按钮，导航到 `HelloWorld` 项目，并选择其中的 `proj.android` 文件夹，然后点击 **下一步**。

1.  您应该看到项目正在编译。整个框架库将被编译，并且基础模板中使用的类也将被编译。

1.  很遗憾，在框架的 3.4 版本中，这里有一个额外的步骤。在 3.3 版本中已经没有了，但现在又回来了。您必须将项目引用的 Cocos2d-x 库导入到 Eclipse 的包资源管理器中。重复步骤 8，但不是选择 `proj.android` 文件夹，而是选择 `cocos2d/cocos/platform/android/java`，然后点击 **下一步**。

1.  这将选择一个名为 `libcocos2dx` 的库；点击 **完成**。

1.  一旦完成，运行一个构建以检查您的项目是否未能生成正确的资源文件可能是个好主意。因此，导航到 **项目** | **构建所有**。

1.  现在，连接您的 Android 设备并确保 Eclipse 已经识别它。您可能需要在设备上打开 **开发** 选项，或者在与计算机连接并 Eclipse 运行时重启您的设备。

1.  右键单击您的项目文件夹，然后选择 **运行方式** | **Android 应用程序**。

## **发生了什么？**

您已运行了您的第一个 Cocos2d-x 应用程序在 Android 上。对于您的 Android 构建不需要模拟器；这是浪费时间。如果您没有现成的设备，考虑投资一个。

或者，您可以在终端（或命令提示符）中打开项目的根文件夹，并使用 Cocos2d-x 控制台的 `compile` 命令：

```cpp
cocos compile -p android

```

Cocos2d-x 背后的人宣布，他们将在框架的未来版本中移除构建 Python 脚本，所以做好准备并了解如何在没有它的情况下进行操作是好的。

当您使用 Eclipse 工作时，您很快就会面临可怕的`java.lang.NullPointerException`错误。这可能与 ADT、CDT 或 NDK 中的冲突有关！

### 注意

当您遇到这个错误时，您除了重新安装 Eclipse 指向的任何东西作为罪魁祸首外别无选择。这可能在更新后发生，或者如果您出于某种原因安装了使用 NDK 或 ADT 路径的另一个框架。如果错误与特定项目或库相关，只需从 Eclipse 的项目资源管理器中删除所有项目，然后重新导入它们。

现在让我们来查看示例应用程序及其文件。

## 文件夹结构

首先是`Classes`文件夹；它将包含您的应用程序类，并且完全用 C++编写。在其下方是`Resources`文件夹，您可以在其中找到应用程序使用的图像、字体和任何类型的媒体。

`ios`文件夹包含了您的应用程序与 iOS 之间的必要底层连接。对于其他平台，您将在各自的平台文件夹中找到必要的链接文件。

维护这种文件结构很重要。因此，您的类将放入`Classes`文件夹，而所有您的图像、声音文件、字体和关卡数据应放置在`Resources`文件夹中。

![文件夹结构](img/00005.jpeg)

现在让我们来查看基本应用程序的主要类。

## iOS 链接类

`AppController`和`RootViewController`负责在 iOS 中设置 OpenGL，并通知底层操作系统您的应用程序即将说`Hello... To the World`。

这些类是用 Objective-C 和 C++混合编写的，正如所有漂亮的括号和`.mm`扩展名所示。您对这些类几乎不会做任何修改；再次，这将在 iOS 处理您的应用程序的方式中反映出来。因此，其他目标将需要相同的指令或根本不需要，具体取决于目标。

例如，在`AppController`中，我可以添加对多点触控的支持。在`RootViewController`中，我可以限制应用程序支持的屏幕方向，例如。

## AppDelegate 类

这个类标志着您的 C++应用程序第一次与底层操作系统通信。它试图映射我们想要分发和监听的主要移动设备事件。从现在开始，您所有的应用程序都将用 C++编写（除非您需要针对特定目标的其他东西）并且从这一点开始，您可以添加针对不同目标的条件代码。

在`AppDelegate`中，您应该设置`Director`对象（它是 Cocos2d-x 功能强大的单例管理对象），以按照您想要的方式运行您的应用程序。您可以：

+   移除应用程序状态信息

+   改变应用程序的帧率

+   告诉`Director`您的高清图像在哪里，您的标准定义图像在哪里，以及使用哪个

+   您可以更改应用程序的整体缩放比例，使其最适合不同的屏幕

+   `AppDelegate` 类也是开始任何预加载过程的最佳位置

+   最重要的是，在这里你告诉 `Director` 对象以哪个 `Scene` 开始你的应用程序

在这里，你将处理操作系统决定杀死、推到一边或倒挂晾干你的应用程序时会发生什么。你所需做的只是将你的逻辑放在正确的事件处理程序中：`applicationDidEnterBackground` 或 `applicationWillEnterForeground`。

## HelloWorldScene 类

当你运行应用程序时，你会在屏幕上看到一个写着 `Hello World` 和一个角上的数字；那些是你决定在 `AppDelegate` 类周围显示的显示统计信息。

实际的屏幕是由名为 `HelloWorldScene` 的奇特类创建的。它是一个 `Layer` 类，它创建自己的场景（如果你不知道 `Layer` 或 `Scene` 类是什么，不用担心；你很快就会知道）。

当它初始化时，`HelloWorldScene` 在屏幕上放置了一个按钮，你可以按下它来退出应用程序。实际上，这个按钮是一个 `Menu` 对象的一部分，该对象只有一个按钮，按钮有两个图像状态，当按下该按钮时有一个回调事件。

`Menu` 对象自动处理针对其成员的触摸事件，所以你不会看到任何代码漂浮。然后，还有必要的 `Label` 对象来显示 `Hello World` 消息和背景图像。

## 谁生谁？

如果你之前从未使用过 Cocos2d 或 Cocos2d-x，初始 `scene()` 方法的实例化方式可能会让你感到头晕。为了回顾，在 `AppDelegate` 中你有：

```cpp
auto scene = HelloWorld::createScene();
director->runWithScene(scene);
```

`Director` 需要一个 `Scene` 对象来运行，你可以将其视为你的应用程序，基本上。`Scene` 需要显示一些内容，在这种情况下，一个 `Layer` 对象就可以。然后说 `Scene` 包含一个 `Layer` 对象。

在这里，通过 `Layer` 派生类中的静态方法 `scene` 创建了一个 `Scene` 对象。因此，层创建了场景，场景立即将层添加到自身。嗯？放松。这种类似乱伦的实例化可能只会发生一次，当它发生时，你无权过问。所以你可以轻松地忽略所有这些有趣的举动，并转过身去。我保证在第一次之后，实例化将变得容易得多。

# 寻找更多参考资料

按照以下步骤访问 Cocos2d-x 参考资料的最好来源之一：它的 `Test` 项目。

# 是时候行动了——运行测试样本

你可以像打开任何其他 Xcode/Eclipse 项目一样打开测试项目：

1.  在 Eclipse 中，你可以从下载的 Cocos2d-x 文件夹中导入测试项目。你会在 `tests/cpp-tests/proj.android` 中找到它。

1.  你可以按照之前的步骤构建此项目。

1.  在 Xcode 中，你必须打开位于 `build` 文件夹中的 Cocos2d-x 框架文件夹内的测试项目文件：`build/cocos2d_tests.xcodeproj`。

1.  一旦在 Xcode 中打开项目，您必须在**运行**按钮旁边选择正确的目标，如下所示：![执行测试样本的时间](img/00006.jpeg)

1.  为了实际查看测试中的代码，您可以导航到`tests/cpp-tests/Classes`以查看 C++测试或`tests/lua-tests/src`以查看 Lua 测试。更好的是，如果您有一个像`TextWrangler`或类似程序，您可以在**磁盘浏览器**窗口中打开这些整个目录，并将所有这些信息准备好以便在您的桌面上直接引用。

## *发生了什么？*

使用测试样本，您可以可视化 Cocos2d-x 中的大多数功能，了解它们的作用，以及看到一些初始化和自定义它们的方法。

我将经常引用测试中找到的代码。像编程一样，总是有完成同一任务的不同方法，所以有时在向您展示一种方法之后，我会引用另一种您可以在`Test`类中找到的方法（并且那时您可以轻松理解）。

# 其他工具

接下来是您可能需要花费更多钱来获取一些极其有用的工具（并做一些额外的学习）的部分。在这本书的示例中，我使用了其中四个：

+   一个帮助构建精灵图集的工具：我将使用**TexturePacker**([`www.codeandweb.com/texturepacker`](http://www.codeandweb.com/texturepacker))。还有其他替代品，如**Zwoptex**([`zwopple.com/zwoptex/`](http://zwopple.com/zwoptex/))，它们通常提供一些免费功能。Cocos2d-x 现在提供了一个名为**CocosStudio**的免费程序，它与**SpriteBuilder**（以前称为**CocosBuilder**）有些相似，并提供构建精灵图集、位图字体以及许多其他好东西的方法。在撰写本文时，Windows 版本在某种程度上优于 Mac 版本，但它们都是免费的！

+   一个帮助构建粒子效果的工具：我将使用粒子设计师([`www.71squared.com/en/particledesigner`](http://www.71squared.com/en/particledesigner))。根据您的操作系统，您可能在网上找到免费工具。Cocos2d-x 捆绑了一些常见的粒子效果，您可以自定义它们。但盲目地做这个过程我不推荐。CocosStudio 也允许您创建自己的粒子效果，但您可能会发现其界面有点令人望而却步。它确实需要自己的教程书籍！

+   一个帮助构建位图字体的工具：我将使用 Glyph Designer([`www.71squared.com/en/glyphdesigner`](http://www.71squared.com/en/glyphdesigner))。但还有其他选择：bmGlyph（价格不那么昂贵）和 FontBuilder（免费）。构建位图字体并不特别困难——远不如从头开始构建粒子效果困难——但做一次就足以让您快速获取这些工具之一。再次提醒，您也可以尝试 CocosStudio。

+   生成音效的工具：毫无疑问——Windows 上的 sfxr 或其 Mac 版本 cfxr。两者都是免费的（分别见[`www.drpetter.se/project_sfxr.html`](http://www.drpetter.se/project_sfxr.html)和[`thirdcog.eu/apps/cfxr`](http://thirdcog.eu/apps/cfxr)）。

# 摘要

你刚刚学习了如何安装 Cocos2d-x 并创建一个基本应用程序。你也学到了足够的基本 Cocos2d-x 应用程序结构，可以开始构建你的第一个游戏，并且你知道如何部署到 iOS 和 Android。

在阅读本书中的示例时，请将`Test`类放在身边，你很快就会成为 Cocos2d-x 专家！

但首先，让我们回顾一下关于框架及其本地语言的一些内容。
