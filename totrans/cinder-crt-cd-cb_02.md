# 第二章 准备开发

在本章中，我们将涵盖：

+   设置用于调整参数的图形用户界面（GUI）

+   保存和加载配置

+   捕获当前参数状态的快照

+   使用MayaCamUI

+   使用3D空间指南

+   与其他软件通信

+   为iOS准备您的应用程序

# 简介

在本章中，我们将介绍一些在开发过程中非常有用的简单配方。

# 设置用于调整参数的GUI

**图形用户界面**（**GUI**）通常用于控制和调整您的Cinder应用程序。在许多情况下，您花费更多的时间调整应用程序参数以实现所需的结果，而不是编写代码。这在您正在处理一些生成图形时尤其如此。

Cinder通过`InterfaceGl`类提供了一个方便且易于使用的GUI。

![设置用于调整参数的GUI](img/8703OS_02_01.jpg)

## 准备就绪

要使`InterfaceGl`类在您的Cinder应用程序中可用，您只需包含一个头文件即可。

[PRE0]

## 如何操作…

按照此处给出的步骤将GUI添加到您的Cinder应用程序中。

1.  让我们从在我们的主类中准备不同类型的变量开始，我们将使用GUI来操作这些变量。

    [PRE1]

1.  接下来，声明`InterfaceGl`类成员如下：

    [PRE2]

1.  现在我们转向`setup`方法，并初始化我们的GUI窗口，将`"Parameters"`作为窗口标题传递给`InterfaceGl`构造函数：

    [PRE3]

1.  现在我们可以添加和配置变量的控件：

    [PRE4]

    查看`addParam`方法和其参数。第一个参数只是字段标题。第二个参数是存储值的变量的指针。有许多支持的变量类型，例如`bool`、`float`、`double`、`int`、`Vec3f`、`Quatf`、`Color`、`ColorA`和`std::string`。

    可能的变量类型及其接口表示在以下表中列出：

    | 类型 | 表示 |
    | --- | --- |
    | `std::string` | ![如何操作…](img/8703OS_02_02.jpg) |
    | `Numerical: int, float, double` | ![如何操作…](img/8703OS_02_03.jpg) |
    | `bool` | ![如何操作…](img/8703OS_02_04.jpg) |
    | `ci::Vec3f` | ![如何操作…](img/8703OS_02_05.jpg) |
    | `ci::Quatf` | ![如何操作…](img/8703OS_02_06.jpg) |
    | `ci::Color` | ![如何操作…](img/8703OS_02_07.jpg) |
    | `ci::ColorA` | ![如何操作…](img/8703OS_02_08.jpg) |
    | 枚举参数 | ![如何操作…](img/8703OS_02_09.jpg) |

    第三个参数定义了控制选项。在下面的表中，您可以找到一些常用选项及其简短说明：

    | 名称 | 说明 |
    | --- | --- |
    | `min` | 数值变量的可能最小值 |
    | `max` | 数值变量的可能最大值 |
    | `step` | 定义浮点变量小数点后打印的显著数字的数量 |
    | `key` | 调用按钮回调的键盘快捷键 |
    | `keyIncr` | 增加值的键盘快捷键 |
    | `keyDecr` | 减少值的键盘快捷键 |
    | `readonly` | 将值设置为`true`使变量在GUI中为只读 |
    | `precision` | 定义浮点变量小数点后打印的显著数字的数量 |

    ### 提示

    您可以在以下地址的AntTweakBar页面找到可用选项的完整文档：[http://anttweakbar.sourceforge.net/doc/tools:anttweakbar:varparamsyntax](http://anttweakbar.sourceforge.net/doc/tools:anttweakbar:varparamsyntax)。

1.  最后一件要做的事情是调用`InterfaceGl::draw()`方法。我们将在主类中的`draw`方法末尾通过输入以下代码行来完成此操作：

    [PRE5]

## 它是如何工作的...

在`setup`方法中，我们将设置GUI窗口并添加控件，在`addParam`方法的第一个参数中设置一个名称。在第二个参数中，我们指向我们想要链接GUI元素的变量。每次我们通过GUI更改值时，链接的变量都会更新。

## 更多...

对于`InterfaceGl`，如果您需要更多控制内置GUI机制，请参阅*AntTweakBar*文档，您可以在本菜谱的*也见*部分提到的项目页面上找到。

### 按钮

您还可以向InterfaceGl (CIT)面板添加按钮，并为其分配一些函数的回调。例如：

[PRE6]

在GUI中点击**开始**按钮将触发`MainApp`类的`start`方法。

### 面板位置

控制GUI面板位置的便捷方式是通过使用*AntTweekBar*工具。您必须包含一个额外的头文件：

[PRE7]

现在您可以使用以下代码行更改GUI面板的位置：

[PRE8]

在这种情况下，`Parameters`是GUI面板名称，`position`选项接受x和y作为值。

## 也见

CinderBlocks中提供了一些看起来不错的GUI库。Cinder有一个名为blocks的扩展系统。CinderBlocks背后的理念是提供与许多第三方库的易于使用的集成。您可以在*与其他软件通信*菜谱中找到如何将CinderBlocks示例添加到您的项目的说明。

### SimpleGUI

您可以在[https://github.com/vorg/MowaLibs/tree/master/SimpleGUI](https://github.com/vorg/MowaLibs/tree/master/SimpleGUI)找到由*Marcin Ignac*开发的作为CinderBlock的替代GUI。

### ciUI

您可以查看由*Reza Ali*开发的作为CinderBlock的替代用户界面，地址为[http://www.syedrezaali.com/blog/?p=2366](http://www.syedrezaali.com/blog/?p=2366)。

### AntTweakBar

Cinder中的`InterfaceGl`是在*AntTweakBar*之上构建的；您可以在[http://www.antisphere.com/Wiki/tools:anttweakbar](http://www.antisphere.com/Wiki/tools:anttweakbar)找到其文档。

# 保存和加载配置

你将要开发的大多数应用程序都会操作用户设置的输入参数。例如，这可能是某些图形元素的色彩或位置，或者用于设置与其他应用程序通信的参数。从外部文件读取配置对于你的应用程序是必要的。我们将使用 Cinder 内置的读取和写入 XML 文件的支持来实现配置持久化机制。

## 准备工作

在主类中创建两个可配置的变量：我们正在与之通信的主机的 IP 地址和端口号。

[PRE9]

## 如何操作...

现在，我们将实现 `loadConfig` 和 `saveConfig` 方法，并在应用程序启动时加载配置，在关闭时保存更改。

1.  包含以下两个额外的头文件：

    [PRE10]

1.  我们将为加载和保存 XML 配置文件准备两种方法。

    [PRE11]

1.  现在，在主类的 `setup` 方法中，我们将放置以下内容：

    [PRE12]

1.  在此之后，我们将按照以下方式实现 `shutdown` 方法：

    [PRE13]

1.  并且不要忘记在主类中声明 `shutdown` 方法：

    [PRE14]

## 它是如何工作的...

前两个方法，`loadConfig` 和 `saveConfig`，是必不可少的。`loadConfig` 方法尝试打开 `config.xml` 文件并找到 `general` 节点。在 `general` 节点中应该有 `hostIP` 和 `hostPort` 节点。这些节点的值将被分配到我们应用程序中相应的变量：`mHostIP` 和 `mHostPort`。

`shutdown` 方法在 Cinder 应用程序关闭前自动触发，因此当我们退出应用程序时，我们的配置值将被存储在 XML 文件中。最后，我们的配置 XML 文件看起来像这样：

[PRE15]

你可以清楚地看到节点正在引用应用程序变量。

## 参见

你可以编写自己的配置加载器和保存器，或者使用现有的 CinderBlock。

### Cinder-Config

Cinder-Config 是一个小的 CinderBlock，用于创建配置文件以及 `InterfaceGl`。

[https://github.com/dawidgorny/Cinder-Config](https://github.com/dawidgorny/Cinder-Config)

# 制作当前参数状态的快照

我们将实现一个简单但有用的机制来保存和加载参数的状态。示例中使用的代码将基于之前的配方。

## 准备工作

假设我们有一个频繁更改的变量。在这种情况下，它将是我们在绘图中更改的某个元素的色彩，主类将具有以下成员变量：

[PRE16]

## 如何操作...

我们将使用内置的 XML 解析器和 `fileDrop` 事件处理器。

1.  我们必须包含以下额外的头文件：

    [PRE17]

1.  首先，我们实现两个用于加载和保存参数的方法：

    [PRE18]

1.  现在，我们声明一个类成员。它将是一个触发快照创建的标志：

    [PRE19]

1.  在 `setup` 方法中为其赋值：

    [PRE20]

1.  在 `draw` 方法的末尾，我们在 `params::InterfaceGl::draw();` 行之前放置以下代码：

    [PRE21]

1.  我们想在 `InterfaceGl` 窗口中创建一个按钮：

    [PRE22]

    正如你所见，我们还没有`makeSnapshotClick`方法。实现起来很简单：

    [PRE23]

1.  最后一步将是添加以下方法以支持*拖放*：

    [PRE24]

## 它是如何工作的...

我们有两种方法用于在XML文件中加载和存储`mColor`值。这些方法是`loadParameters`和`saveParameters`。

我们放在`draw`方法内部的代码需要一些解释。我们正在等待`mMakeSnapshot`方法被设置为`true`，然后我们创建一个时间戳以避免覆盖之前的快照。接下来的两行通过调用`saveParameters`方法存储所选值，并使用`writeImage`函数将当前窗口视图保存为PNG文件。请注意，我们在调用`InterfaceGl::draw`之前放置了这段代码，所以我们保存的窗口视图没有GUI。

这里有一个很好的功能是加载快照文件的*拖放*功能。它在`fileDrop`方法中实现；每当文件被拖放到你的应用程序窗口时，Cinder都会调用此方法。首先，我们获取被拖放文件的路径；在多个文件的情况下，我们只取一个。然后我们使用被拖放文件的路径作为参数调用`loadParameters`方法。

# 使用MayaCamUI

我们将向你的3D场景添加一个导航功能，这是我们自从建模3D软件以来就熟知的。使用`MayaCamUI`，你只需几行代码就能做到这一点。

## 准备工作

我们需要在场景中有些3D对象。你可以使用Cinder提供的某些原语，例如：

[PRE25]

一个彩色立方体是一个每个面都有不同颜色的立方体，因此很容易确定方向。

![准备工作](img/8703OS_02_10.jpg)

## 如何做到这一点...

执行以下步骤以创建相机导航：

1.  我们需要`MayaCam.h`头文件：

    [PRE26]

1.  我们还需要在主类中添加一些成员声明：

    [PRE27]

1.  在`setup`方法内部，我们将设置相机的初始状态：

    [PRE28]

1.  现在我们必须实现三个方法：

    [PRE29]

1.  在`draw`方法内部应用相机矩阵：

    [PRE30]

## 它是如何工作的...

在`setup`方法内部，我们设置初始的相机设置。当窗口调整大小时，我们必须更新相机的纵横比，因此我们将这段代码放在`resize`方法中。每当我们的应用程序窗口调整大小时，Cinder都会自动调用此方法。我们在`mouseDown`和`mouseDrag`方法内部捕获鼠标事件。你可以点击并拖动鼠标进行旋转，右键点击进行缩放，使用中间按钮进行平移。现在你已经在自己的应用程序中拥有了类似于常见3D建模软件的交互功能。

# 使用3D空间指南

我们将尝试使用内置的Cinder方法来可视化我们正在工作的场景的一些基本信息。这应该会使在3D空间中工作更加舒适。

## 准备工作

我们将需要我们在上一个菜谱中实现的`MayaCamUI`导航。

## 如何做到这一点...

我们将绘制一些有助于可视化和找到 3D 场景方向的物体。

1.  我们将在 `MayaCamUI` 之外添加另一个相机。让我们先在主类中添加成员声明：

    [PRE31]

1.  然后，我们将在 `setup` 方法内部设置初始值：

    [PRE32]

1.  我们必须在 `resize` 方法中更新 `mSceneCamera` 的纵横比：

    [PRE33]

1.  现在，我们将实现 `keyDown` 方法，通过按键盘上的 *1* 或 *2* 键在两个相机之间切换：

    [PRE34]

1.  我们将要使用的方法是 `drawGrid`，它看起来是这样的：

    [PRE35]

1.  之后，我们可以实现我们的主要绘图程序，所以这里是整个 `draw` 方法：

    [PRE36]

## 它是如何工作的...

我们有两个相机；`mSceneCam` 用于最终渲染，`mMayaCam` 用于场景中物体的预览。你可以通过按 *1* 或 *2* 键在它们之间切换。默认相机是 `MayaCam`。

![它是如何工作的...](img/8703OS_02_11.jpg)

在前面的截图中，你可以看到整个场景设置，包括坐标系统的原点、帮助你轻松保持 3D 空间方向的构造网格，以及 `mSceneCam` 之间的视锥体和向量可视化。你可以使用 `MayaCamUI` 在这个空间中导航。

如果你按下 *2* 键，你将切换到 `mSceneCam` 的视图，因此你将只看到你的 3D 对象，没有引导，如下面的截图所示：

![它是如何工作的...](img/8703OS_02_12.jpg)

# 与其他软件通信

我们将实现两个 Cinder 应用程序之间的示例通信，以说明我们如何发送和接收信号。这两个应用程序中的每一个都可以很容易地被非 Cinder 应用程序替换。

我们将要使用 **Open Sound Control** (**OSC**) 消息格式，它是为网络中广泛的多媒体设备之间的通信而设计的。OSC 使用 UDP 协议，提供灵活性和性能。每个消息由类似 URL 的地址和整数、浮点或字符串类型的参数组成。OSC 的流行使其成为连接使用不同技术开发的网络或本地机器上的不同环境或应用程序的绝佳工具。

## 准备工作

在下载 Cinder 包的同时，我们也在下载四个主要块。其中之一是位于 `blocks` 目录中的 `osc` 块。首先，我们将向我们的 XCode 项目根目录添加一个新的组，并将其命名为 `Blocks`，然后我们将拖动 `osc` 文件夹到 `Blocks` 组中。确保选中 **Create groups for any added folders** 选项和 **MainApp** 在 **Add to targets** 部分中。

![准备工作](img/8703OS_02_13.jpg)

我们只需要包含 `osc` 文件夹中的 `src`，因此我们将从我们的项目树中删除对 `lib` 和 `samples` 文件夹的引用。最终的项目结构应该看起来像下面的截图：

![准备工作](img/8703OS_02_14.jpg)

现在，我们必须在项目的构建设置中将 `OSC` 库文件的路径添加为另一个链接器标志的位置：

[PRE37]

### 小贴士

**CINDER_PATH** 应该在项目的构建设置中设置为用户定义的设置，并且它应该是 Cinder 根目录的路径。

## 如何实现...

首先，我们将介绍关于 *发送者* 的说明，然后是 *监听者*。

### 发送者

我们将实现一个发送 OSC 消息的应用程序。

1.  我们必须包含一个额外的头文件：

    [PRE38]

1.  之后，我们可以使用 `osc::Sender` 类，因此让我们在主类中声明所需的属性：

    [PRE39]

1.  现在，我们必须在 `setup` 方法中设置我们的发送者：

    [PRE40]

1.  将 `mObjectPosition` 的默认值设置为窗口的中心：

    [PRE41]

1.  我们现在可以实现 `mouseDrag` 方法，它包括两个主要操作——根据鼠标位置更新对象位置，并通过 OSC 发送位置信息。

    [PRE42]

1.  我们最后需要做的是绘制一个方法，仅用于可视化对象的位置：

    [PRE43]

### 监听者

我们将实现一个接收 `OSC` 消息的应用程序。

1.  我们必须包含一个额外的头文件：

    [PRE44]

1.  之后，我们可以使用 `osc::Listener` 类，因此让我们在主类中声明所需的属性：

    [PRE45]

1.  现在，我们必须在 `setup` 方法中设置我们的监听者对象，传递监听端口作为参数：

    [PRE46]

1.  并且将 `mObjectPosition` 的默认值设置为窗口的中心：

    [PRE47]

1.  在 `update` 方法内部，我们将监听传入的 `OSC` 消息：

    [PRE48]

1.  我们的 `draw` 方法将与发送者版本几乎相同，但我们将绘制一个填充的圆圈而不是描边的圆圈：

    [PRE49]

# 工作原理...

我们已经实现了发送应用程序，该应用程序通过 OSC 协议发送鼠标位置。这些消息，带有地址 `/obj/position`，可以被任何在许多其他框架和编程语言中实现的非 Cinder OSC 应用程序接收。消息的第一个参数是鼠标的 x 轴位置，第二个参数是 y 轴位置。两者都是 `float` 类型。

![工作原理...](img/8703OS_02_15.jpg)

在我们的例子中，接收消息的应用程序是另一个 Cinder 应用程序，它会在你指向发送应用程序窗口中的确切位置绘制一个填充的圆圈。

![工作原理...](img/8703OS_02_16.jpg)

# 还有更多...

这只是 OSC 提供的可能性的一个简短示例。这种简单的通信方法甚至可以应用于非常复杂的项目。当多个设备作为独立单元工作时，OSC 工作得非常好。但到了某个时候，来自它们的数据需要被处理；例如，来自摄像机的帧可以被计算机视觉软件处理，并将结果通过网络发送到另一台投影可视化的机器。基于 UDP 协议的实现不仅因为传输数据比使用 TCP 快而提供性能，而且由于不需要连接握手，实现也更为简单。

## 广播

您可以通过设置一个广播地址作为目标主机来向您网络上的所有主机发送 OSC 消息：`255.255.255.255`。例如，在子网的情况下，您可以使用 `192.168.1.255`。

### 小贴士

如果您在 Mac OS X 10.7 下编译时遇到链接错误，请尝试在您的项目构建设置中将 **内联方法隐藏** 设置为 **否**。

# 参见

您可以通过查看以下链接来获取有关 OSC 实现的更多信息。

## Flash 中的 OSC

要在您的 ActionScript 3.0 代码中支持接收和发送 OSC 消息，您可以使用以下库：[http://bubblebird.at/tuioflash/](http://bubblebird.at/tuioflash/)

## 处理中的 OSC

要在您的 **Processing** 草图中支持 **OSC** 协议，您可以使用以下库：[http://www.sojamo.de/libraries/oscP5/](http://www.sojamo.de/libraries/oscP5/)

## openFrameworks 中的 OSC

要在您的 `openFrameworks` 项目中支持接收和发送 OSC 消息，您可以使用 `ofxOsc` 扩展程序：[http://ofxaddons.com/repos/112](http://ofxaddons.com/repos/112)

## OpenSoundControl 协议

您可以在其官方网站上找到有关 OSC 协议和相关工具的更多信息：[http://opensoundcontrol.org/](http://opensoundcontrol.org/)。

# 为 iOS 准备应用程序

使用 Cinder 的主要好处是生成的多平台代码。在大多数情况下，您的应用程序可以在 Windows、Mac OS X 和 iOS 上编译，而无需进行重大修改。

## 准备工作

如果您想在 iOS 设备上运行应用程序，您需要注册为 Apple 开发者并购买 iOS 开发者计划。

## 如何操作...

在注册为 Apple 开发者或购买 iOS 开发者计划后，您可以使用 Tinderbox 创建一个初始的 XCode iOS 项目。

1.  在运行 Tinderbox 后，您必须将 **目标** 设置为 **Cocoa Touch**。![如何操作...](img/8703OS_02_18.jpg)

1.  它将为您生成一个项目结构，支持针对多点触控屏幕的特定 iOS 事件。

    我们可以使用事件来处理多个触摸操作，并轻松访问加速度计数据。触摸事件和鼠标事件之间的主要区别在于，在只有一个鼠标光标的情况下，可以有多个活跃的触摸点。正因为如此，每个触摸会话都有一个 ID，可以从 `TouchEvent` 对象中读取。

    | 方法 | 描述 |
    | --- | --- |
    | `touchesBegan( TouchEvent event )` | 多指触摸序列的开始 |
    | `touchesMoved( TouchEvent event )` | 在多指触摸序列中拖动 |
    | `touchesEnded( TouchEvent event )` | 多指触摸序列的结束 |
    | `getActiveTouches()` | 返回所有活动触摸 |
    | `accelerated( AccelEvent event )` | 加速度方向的3D向量 |

## 相关内容

我建议你查看Cinder包中包含的示例项目：`MultiTouchBasic` 和 `iPhoneAccelerometer`。

### 苹果开发者中心

你可以在这里找到有关iOS开发者计划的更多信息：[https://developer.apple.com/](https://developer.apple.com/)
