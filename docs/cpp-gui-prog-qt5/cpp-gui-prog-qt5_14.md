# Qt Quick 和 QML

在这一章中，我们将学习与本书其他章节非常不同的内容。Qt 包括两种不同的应用开发方法。第一种方法是 Qt Widgets 和 C++，这是我们在之前所有章节中都涵盖过的内容。第二种方法是使用 Qt Quick 控件和 QML 脚本语言，这将在本章中介绍。

在本章中，我们将涵盖以下主题：

+   介绍 Qt Quick 和 QML

+   Qt Quick 控件和控制

+   Qt Quick 设计师

+   Qt Quick 布局

+   基本的 QML 脚本

准备好了吗？让我们开始吧！

# 介绍 Qt Quick 和 QML

在接下来的部分，我们将学习 Qt Quick 和 QML 是什么，以及如何利用它们来开发 Qt 应用程序，而无需编写 C++代码。

# 介绍 Qt Quick

**Qt Quick**是 Qt 中的一个模块，为开发面向触摸和视觉的应用程序提供了一整套用户界面引擎和语言基础设施。开发人员选择 Qt Quick 后，将使用 Qt Quick 对象和控件，而不是通常的 Qt Widgets 进行用户界面设计。

此外，开发人员将使用类似于**JavaScript**的 QML 语言编写代码，而不是使用 C++代码。但是，您可以使用 Qt 提供的 C++ API 来扩展 QML 应用程序，通过相互调用每种语言的函数（在 QML 中调用 C++函数，反之亦然）。

开发人员可以通过在创建项目时选择正确的选项来选择他们喜欢的开发应用程序的方法。开发人员可以选择 Qt Quick 应用程序而不是通常的 Qt Widgets 应用程序选项，这将告诉 Qt Creator 为您的项目创建不同的起始文件和设置，从而增强 Qt Quick 模块：

![](img/43e54681-0742-4983-9a4e-70c933538d25.png)

当您创建 Qt Quick 应用程序项目时，Qt Creator 将要求您选择项目的最低要求 Qt 版本：

![](img/3077a542-a06f-4c6a-96d6-2b84826ebc78.png)

选择了 Qt 版本后，Qt Quick 设计师将确定要启用哪些功能，并在 QML 类型窗口上显示哪些小部件。我们将在后面的部分中更多地讨论这些内容。

# 介绍 QML

**QML**（**Qt 建模语言**）是一种用于设计触摸友好用户界面的用户界面标记语言，类似于 CSS 在 HTML 上的工作方式。与 C++或 JavaScript 不同，它们都是命令式语言，QML 是一种声明式语言。在声明式编程中，您只需在脚本中表达逻辑，而不描述其控制流。它只是告诉计算机要做什么，而不是如何做。然而，命令式编程需要语句来指定操作。

当您打开新创建的 Qt Quick 项目时，您将在项目中看到`main.qml`和`MainForm.ui.qml`，而不是通常的`mainwindow.h`和`mainwindow.cpp`文件。您可以在以下截图中的项目目录中看到这一点：

![](img/fc36ad7d-787e-4a09-bde7-f95ab7c362df.png)

这是因为整个项目主要将在 QML 上运行，而不是在 C++上。您将看到的唯一 C++文件是`main.cpp`，它在应用程序启动时只是加载`main.qml`文件。`main.cpp`中执行此操作的代码如下所示：

```cpp
int main(int argc, char *argv[]) 
{ 
   QGuiApplication app(argc, argv); 

   QQmlApplicationEngine engine; 
   engine.load(QUrl(QStringLiteral("qrc:/main.qml"))); 
   if (engine.rootObjects().isEmpty()) 
         return -1; 

   return app.exec(); 
} 
```

您应该已经意识到有两种类型的 QML 文件，一种是扩展名为`.qml`，另一种是扩展名为`.ui.qml`。尽管它们都使用相同的语法等，但它们在项目中的作用是非常不同的。

首先，`.ui.qml`文件（在开头多了一个`.ui`）用作基于 Qt Quick 的用户界面设计的声明文件。您可以使用 Qt Quick Designer 可视化编辑器编辑`.ui.qml`文件，并轻松设计应用程序的 GUI。您也可以向文件添加自己的代码，但对文件中可以包含的代码有一些限制，特别是与逻辑代码相关的限制。当运行 Qt Quick 应用程序时，Qt Quick 引擎将阅读`.ui.qml`文件中存储的所有信息，并相应地构建用户界面，这与 Qt Widgets 应用程序中使用的`.ui`文件非常相似。

然后，我们有另一个只有`.qml`扩展名的文件。这个文件仅用于构建 Qt Quick 应用程序中的逻辑和功能，就像 Qt Widget 应用程序中使用的`.h`和`.cpp`文件一样。这两种不同的格式将应用程序的视觉定义与其逻辑块分开。这使开发人员能够将相同的逻辑代码应用于不同的用户界面模板。您不能使用 Qt Quick Designer 打开`.qml`文件，因为它不用于 GUI 声明。`.qml`文件是由开发人员手动编写的，对他们使用的 QML 语言特性没有限制。

让我们首先打开`MainForm.ui.qml`，看看这两个 QML 文件的区别。默认情况下，Qt Creator 将打开用户界面设计师（Qt Quick Designer）；然而，让我们通过按左侧面板上的编辑按钮切换到代码编辑模式：

![](img/d2c87dda-ef9d-45f7-89c1-0bc053484c75.png)

然后，您将能够看到形成您在设计模式中看到的用户界面的 QML 脚本。让我们分析这段代码，看看 QML 与 C++相比是如何工作的。在`MainForm.ui.qml`中，您首先看到的是这行代码：

```cpp
import QtQuick 2.6 
```

这非常简单明了；我们需要导入带有适当版本号的`Qt Quick`模块。不同的 Qt Quick 版本可能具有不同的功能，并支持不同的部件控件。有时，甚至语法可能略有不同。请确保为您的项目选择正确的版本，并确保它支持您需要的功能。如果不知道要使用哪个版本，请考虑使用最新版本。

接下来，我们将看到在两个大括号之间声明的不同 GUI 对象（我们称之为 QML 类型）。我们首先看到的是`Rectangle`类型：

```cpp
    Rectangle { 
       property alias mouseArea: mouseArea 
       property alias textEdit: textEdit 

       width: 360 
       height: 360 
       ... 
```

在这种情况下，`Rectangle`类型是窗口背景，类似于 Qt Widget 应用程序项目中使用的中央窗口部件。让我们看看`Rectangle`下面的其他 QML 类型：

```cpp
    MouseArea { 
        id: mouseArea 
        anchors.fill: parent 
    } 

    TextEdit { 
        id: textEdit 
        text: qsTr("Enter some text...") 
        verticalAlignment: Text.AlignVCenter 
        anchors.top: parent.top 
        anchors.horizontalCenter: parent.horizontalCenter 
        anchors.topMargin: 20 
        Rectangle { 
            anchors.fill: parent 
            anchors.margins: -10 
            color: "transparent" 
            border.width: 1 
        } 
    } 
```

`MousArea`类型，顾名思义，是一个检测鼠标点击和触摸事件的无形形状。您基本上可以通过在其上放置`MouseArea`将任何东西变成按钮。之后，我们还有一个`TextEdit`类型，其行为与 Qt Widget 应用程序中的`Line Edit`部件完全相同。

您可能已经注意到，在`Rectangle`声明中有两个带有`alias`关键字的属性。这两个属性公开了`MouseArea`和`TextEdit`类型，并允许其他 QML 脚本与它们交互，接下来我们将学习如何做到这一点。

现在，打开`main.qml`并查看其代码：

```cpp
import QtQuick 2.6 
import QtQuick.Window 2.2 

Window { 
    visible: true 
    width: 640 
    height: 480 
    title: qsTr("Hello World") 

    MainForm { 
        anchors.fill: parent 
        mouseArea.onClicked: { 
            console.log(qsTr('Clicked on background. Text: "' + 
            textEdit.text + '"')) 
        } 
    } 
} 
```

在上面的代码中，有一个`Window`类型，只能通过导入`QtQuick.Window`模块才能使用。设置了`Window`类型的属性后，声明了`MainForm`类型。这个`MainForm`类型实际上就是我们之前在`MainForm.ui.qml`中看到的整个用户界面。由于`MouseArea`和`TextEdit`类型已在`MainForm.ui.qml`中公开，我们现在可以在`main.qml`中访问并使用它们。

QML 还使用 Qt 提供的信号和槽机制，但写法略有不同，因为我们不再编写 C++代码。例如，我们可以在上面的代码中看到`onClicked`的使用，这是一个内置信号，相当于 Qt Widgets 应用程序中的`clicked()`。由于`.qml`文件是我们定义应用程序逻辑的地方，我们可以定义`onClicked`被调用时发生的事情。另一方面，我们不能在`.ui.qml`中做同样的事情，因为它只允许与视觉相关的代码。如果你尝试在`.ui.qml`文件中编写逻辑相关的代码，Qt Creator 会发出警告。

就像 Qt Widgets 应用程序一样，您也可以像以前一样构建和运行项目。默认示例应用程序看起来像这样：

![](img/cfbfa37c-2ba3-4e65-b456-4735c5c90efa.png)

您可能会意识到构建过程非常快。这是因为 QML 代码默认不会被编译成二进制代码。QML 是一种解释性语言，就像 JavaScript 一样，因此不需要编译就可以执行。在构建过程中，所有 QML 文件将被打包到应用程序的资源系统中。然后，在应用程序启动时，Qt Quick 引擎将加载和解释 QML 文件。

但是，您仍然可以选择使用包含在 Qt 中的`Qt Quick Compiler`程序将您的 QML 脚本编译成二进制代码，以使代码执行速度略快于通常情况。这是一个可选步骤，除非您要在资源非常有限的嵌入式系统上运行应用程序，否则不需要。

现在我们已经了解了**Qt Quick**和**QML**语言是什么，让我们来看看 Qt 提供的所有不同的 QML 类型。

# Qt Quick 小部件和控件

在 Qt Quick 的领域中，小部件和控件被称为`QML 类型`。默认情况下，**Qt Quick Designer**为我们提供了一组基本的 QML 类型。您还可以导入随不同模块提供的其他 QML 类型。此外，如果没有现有的类型符合您的需求，甚至可以创建自定义的 QML 类型。

让我们来看看 Qt Quick Designer 默认提供的 QML 类型。首先，这是基本类别下的 QML 类型：

![](img/87221c3c-f5cb-4409-a1aa-2e1b86f76030.png)

让我们看看不同的选项：

+   **Border Image**：Border Image 是一个设计用来创建可维持其角形状和边框的可伸缩矩形形状的 QML 类型。

+   **Flickable**：Flickable 是一个包含所有子类型的 QML 类型，并在其裁剪区域内显示它们。Flickable 还被`ListView`和`GridView`类型扩展和用于滚动长内容。它也可以通过触摸屏轻扫手势移动。

+   **Focus Scope**：Focus Scope 是一个低级别的 QML 类型，用于促进其他 QML 类型的构建，这些类型在被按下或释放时可以获得键盘焦点。我们通常不直接使用这种 QML 类型，而是使用直接从它继承的其他类型，如`GroupBox`、`ScrollView`、`StatusBar`等。

+   **Image**：`Image`类型基本上是不言自明的。它可以加载本地或网络上的图像。

+   **Item**：`Item`类型是 Qt Quick 中所有可视项的最基本的 QML 类型。Qt Quick 中的所有可视项都继承自这个`Item`类型。

+   **MouseArea**：我们已经在默认的 Qt Quick 应用程序项目中看到了`MouseArea`类型的示例用法。它在预定义区域内检测鼠标点击和触摸事件，并在检测到时调用 clicked 信号。

+   **Rectangle**：`Rectangle` QML 类型与`Item`类型非常相似，只是它有一个可以填充纯色或渐变的背景。您还可以选择使用自己的颜色和厚度添加边框。

+   **文本**：`Text` QML 类型也很容易理解。它只是在窗口上显示一行文本。您可以使用它来显示特定字体系列和字体大小的纯文本和富文本。

+   **文本编辑**：文本编辑 QML 类型相当于 Qt Widgets 应用程序中的`文本编辑`小部件。当焦点在它上面时，允许用户输入文本。它可以显示纯文本和格式化文本，这与`文本输入`类型非常不同。

+   **文本输入**：文本输入 QML 类型相当于 Qt Widgets 应用程序中的行编辑小部件，因为它只能显示单行可编辑的纯文本，这与`文本编辑`类型不同。您还可以通过验证器或输入掩码对其应用输入约束。通过将`echoMode`设置为`Password`或`PasswordEchoOnEdit`，它也可以用于密码输入字段。

我们在这里讨论的 QML 类型是 Qt Quick Designer 默认提供的最基本的类型。这些也是用于构建其他更复杂的 QML 类型的基本构建块。Qt Quick 还提供了许多额外的模块，我们可以将其导入到我们的项目中，例如，如果我们在`MainForm.ui.qml`文件中添加以下行：

```cpp
import QtQuick.Controls 2.2
```

当您切换到设计模式时，Qt Quick Designer 将在您的 Qt Quick Designer 上显示一堆额外的 QML 类型：

![](img/64d47d41-00c4-45c9-9bbe-80aff6dc8bd3.png)

我们不会逐一介绍所有这些 QML 类型，因为它们太多了。如果您有兴趣了解更多关于这些 QML 类型的信息，请访问以下链接：[`doc.qt.io/qt-5.10/qtquick-controls-qmlmodule.html`](https://doc.qt.io/qt-5.10/qtquick-controls-qmlmodule.html)

# Qt Quick Designer

接下来，我们将看一下 Qt Quick Designer 对 Qt Quick 应用程序项目的布局。当您打开一个`.ui.qml`文件时，Qt Quick Designer，即包含在 Qt Creator 工具集中的设计工具，将自动为您启动。

自从本书第一章以来一直跟随所有示例项目的读者可能会意识到，Qt Quick Designer 看起来与我们一直在使用的设计工具有些不同。这是因为 Qt Quick 项目与 Qt Widgets 项目非常不同，因此设计工具自然也应该有所不同以适应其需求。

让我们看看 Qt Quick 项目中的 Qt Quick Designer 是什么样子的：

![](img/a20022be-e9e5-4630-baa7-4c0b82b689c5.png)

1.  库：库窗口显示当前项目可用的所有 QML 类型。您可以单击并将其拖动到画布窗口中以将其添加到您的 UI 中。您还可以创建自己的自定义 QML 类型并在此处显示。

1.  资源：资源窗口以列表形式显示所有资源，然后可以在 UI 设计中使用。

1.  导入：导入窗口允许您将不同的 Qt Quick 模块导入到当前项目中。

1.  导航器：导航器窗口以树形结构显示当前 QML 文件中的项目。它类似于 Qt Widgets 应用程序项目中的对象操作器窗口。

1.  连接：连接窗口由几个不同的选项卡组成：连接、绑定、属性和后端。这些选项卡允许您在不切换到编辑模式的情况下向您的 QML 文件添加连接（信号和槽）、绑定和属性。

1.  状态窗格：状态窗格显示 QML 项目中的不同状态，通常描述 UI 配置，例如 UI 控件、它们的属性和行为以及可用操作。

1.  画布：画布是您设计应用程序 UI 的工作区。

1.  属性窗格：与我们在 Qt Widgets 应用程序项目中使用的属性编辑器类似，QML 设计师中的属性窗格显示所选项目的属性。在更改这里的值后，您可以立即在 UI 中看到结果。

# Qt Quick 布局

与 Qt Widget 应用程序一样，Qt Quick 应用程序中也存在布局系统。唯一的区别是在 Qt Quick 中称为定位器：

![](img/ee431b85-661f-47d1-9709-8f4d7a64297b.png)

最显著的相似之处是列和行定位器。这两者与 Qt Widgets 应用程序中的垂直布局和水平布局完全相同。除此之外，网格定位器也与网格布局相同。

在 Qt Quick 中唯一额外的是 Flow 定位器。Flow 定位器中包含的项目会像页面上的单词一样排列，项目沿一个轴排成一行，然后沿另一个轴放置项目行。

![](img/931898a3-240c-472c-91f7-58409ec5cbc9.png)

# 基本的 QML 脚本

在接下来的部分中，我们将学习如何使用 Qt Quick Designer 和 QML 创建我们的第一个 Qt Quick 应用程序！

# 设置项目

话不多说，让我们动手使用 QML 创建一个 Qt Quick 应用程序吧！在这个示例项目中，我们将使用 Qt Quick Designer 和一个 QML 脚本创建一个虚拟登录界面。首先，让我们打开 Qt Creator，并通过转到文件|新建文件或项目...来创建一个新项目。

在那之后，选择 Qt Quick 应用程序并按“选择”....之后，一直按“下一步”直到项目创建完成。我们将在这个示例项目中使用所有默认设置，包括最小所需的 Qt 版本：

![](img/f61ed1c4-6c26-438d-a9d0-adfe3d663049.png)

项目创建完成后，我们需要向项目中添加一些图像文件，以便稍后使用它们：

![](img/984d0ff3-798c-46fe-9d4c-5b9745e3590c.png)

您可以在我们的 GitHub 页面上获取源文件（包括这些图像）：[`github.com/PacktPublishing/Hands-On-GUI-Programming-with-C-QT5`](http://github.com/PacktPublishing/Hands-On-GUI-Programming-with-C-QT5)

我们可以通过右键单击项目窗格中的`qml.qrc`文件并选择在编辑器中打开来将这些图像添加到我们的项目中。添加一个名为`images`的新前缀，并将所有图像文件添加到该前缀中：

![](img/44cb6357-0d0f-4a8c-83c4-c8182c2cafbb.png)

在那之后，打开`MainForm.ui.qml`，并删除 QML 文件中的所有内容。我们通过向画布添加一个 Item 类型，将其大小设置为 400 x 400，并将其命名为`loginForm`来重新开始。之后，在其下方添加一个`Image`类型，并将其命名为`background`。然后将背景图像应用到`Image`类型上，画布现在看起来像这样：

![](img/885c2157-4e28-477d-8ccd-ebc2c3e669ec.png)

然后，在`Image`类型（背景）下添加一个`Rectangle`类型，并在属性窗格中打开布局选项卡。启用垂直和水平锚定选项。之后，将`width`设置为`402`，`height`设置为`210`，将`vertical anchor margin`设置为`50`：

![](img/4b63ff3b-64b7-4cf3-919d-de2c5407db44.png)

接着，我们将矩形的颜色设置为`#fcf9f4`，边框颜色设置为`#efedeb`，然后将边框值设置为`1`。到目前为止，用户界面看起来像这样：

![](img/2046715a-edfc-44b7-bd96-490ea0da78b6.png)

接下来，在矩形下添加一个 Image QML 类型，并将其锚定设置为顶部锚定和水平锚定。然后将其顶部锚定边距设置为`-110`，并将 logo 图像应用到其`image source`属性上。您可以通过单击位于画布顶部的小按钮来打开和关闭 QML 类型的边界矩形和条纹，这样在画布上充满内容时更容易查看结果：

![](img/fd8bc259-88d1-458d-bcae-9d0aa18e09ab.png)

然后，我们在`loginRect`矩形下的画布中添加了三个`Rectangle`类型，并将它们命名为`emailRect`、`passwordRect`和`loginButton`。矩形的锚定设置如下所示：

![](img/b5ff6a01-25f6-4081-92bb-6e0607e3bf95.png)

然后，将`emailRect`和`passwordRect`的`border`值设置为`1`，`color`设置为`#ffffff`，`bordercolor`设置为`#efedeb`。至于`loginButton`，我们将`border`设置为`0`，`radius`设置为`2`，`color`设置为`#27ae61`。登录屏幕现在看起来像这样：

![](img/d5578076-0799-402b-a013-554a56330e25.png)

看起来不错。接下来，我们将在`emailRect`和`passwordRect`中添加`TextInput`、`Image`、`MouseArea`和`Text` QML 类型。由于这里有许多 QML 类型，我将列出需要设置的属性：

+   TextInput：

+   选择颜色设置为`#4f0080`

+   启用左锚点、右锚点和垂直锚点

+   左锚点边距`20`，右锚点边距`40`，垂直边距`3`

+   为密码输入设置 echoMode 为 Password

+   Image：

+   启用右锚点和垂直锚点

+   右锚点边距设置为`10`

+   将图像源设置为电子邮件图标或密码图标

+   将图像填充模式设置为 PreserveAspectFit

+   MouseArea：

+   启用填充父项

+   Text：

+   将文本属性分别设置为`E-Mail`和`Password`

+   文本颜色设置为`#cbbdbd`

+   将文本对齐设置为左对齐和顶部对齐

+   启用左锚点、右锚点和垂直锚点

+   左锚点边距`20`，右锚点边距`40`，垂直边距`-1`

完成后，还要为`loginButton`添加`MouseArea`和`Text`。为`MouseArea`启用`fill parent item`，为`Text` QML 类型启用`vertical`和`horizontal anchors`。然后，将其`text`属性设置为`LOGIN`。

您不必完全按照我的步骤进行，它们只是指导您实现与上面截图类似的结果的指南。但是，最好您应用自己的设计并创建独特的东西！

哦！经过上面漫长的过程，我们的登录屏幕现在应该看起来像这样：

![](img/b17e64f5-faef-47a9-abd5-88669f766d47.png)

在转到`main.qml`之前，我们还需要做一件事，那就是公开我们登录屏幕中的一些 QML 类型，以便我们可以将其链接到我们的`main.qml`文件进行逻辑编程。实际上，我们可以直接在设计工具上做到这一点。您只需点击对象名称旁边的小矩形图标，并确保图标上的三条线穿过矩形框，就像这样：

![](img/882dcceb-99d9-4358-9e59-a444ab53d9d3.png)

我们需要公开/导出的 QML 类型是`emailInput`（TextInput）、`emailTouch`（MouseArea）、`emailDisplay`（Text）、`passwordInput`（TextInput）、`passwordTouch`（MouseArea）、`passwordDisplay`（Text）和`loginMouseArea`（MouseArea）。完成所有这些后，让我们打开`main.qml`。

首先，我们的`main.qml`应该看起来像这样，它只会打开一个空窗口：

```cpp
import QtQuick 2.6 
import QtQuick.Window 2.2 

Window { 
    id: window 
    visible: true 
    width: 800 
    height: 600 
    title: qsTr("My App") 
} 
```

之后，添加`MainForm`对象，并将其锚点设置为`anchors.fill: parent`。然后，当点击（或触摸，如果在触摸设备上运行）`loginButton`时，在控制台窗口上打印一行文本`Login pressed`：

```cpp
Window { 
    id: window 
    visible: true 
    width: 800 
    height: 600 
    title: qsTr("My App") 

    MainForm 
    { 
        anchors.fill: parent 

        loginMouseArea.onClicked: 
        { 
            console.log("Login pressed"); 
        } 
    } 
} 
```

之后，我们将编写`MouseArea`在电子邮件输入上被点击/触摸时的行为。由于我们手动创建自己的文本字段，而不是使用`QtQuick.Controls`模块提供的`TextField` QML 类型，我们必须手动隐藏和显示`E-Mail`和`Password`文本显示，并在用户点击/触摸`MouseArea`时更改输入焦点。

我选择不使用`TextField`类型的原因是，我几乎无法自定义`TextField`的视觉呈现，那么为什么不创建自己的呢？手动为电子邮件输入设置焦点的代码如下：

```cpp
emailTouch.onClicked: 
{ 
    emailDisplay.visible = false;      // Hide emailDisplay 
    emailInput.forceActiveFocus();     // Focus emailInput 
    Qt.inputMethod.show();       // Activate virtual keyboard 
} 

emailInput.onFocusChanged: 
{ 
    if (emailInput.focus == false && emailInput.text == "") 
    { 
        emailDisplay.visible = true;   // Show emailDisplay if 
        emailInput is empty when loses focus 
    } 
} 
```

之后，对密码字段执行相同操作：

```cpp
passwordTouch.onClicked: 
{ 
    passwordDisplay.visible = false;   // Hide passwordDisplay 
    passwordInput.forceActiveFocus();  // Focus passwordInput 
    Qt.inputMethod.show();       // Activate virtual keyboard 
} 

passwordInput.onFocusChanged: 
{ 
    if (passwordInput.focus == false && passwordInput.text == "") 
    { 
        passwordDisplay.visible = true;      // Show passwordDisplay if  
        passwordInput is empty when loses focus 
    } 
} 
```

就是这样，我们完成了！现在您可以编译和运行程序。您应该会得到类似这样的结果：

![](img/e4f5430c-1afb-4481-842b-fa0dcb61ffea.png)

如果您没有看到图片，并且收到错误消息说 Qt 无法打开图片，请返回到您的`MainForm.ui.qml`，并在源属性的前面添加前缀`image/`。这是因为 Qt Quick Designer 加载图片时没有前缀，而您的最终程序需要前缀。添加了前缀后，您可能会意识到在 Qt Quick Designer 中不再看到图片显示，但在最终程序中将正常工作。

我不确定这是一个错误还是他们有意这样做的。希望 Qt 的开发人员可以解决这个问题，这样我们就不必再做额外的步骤了。就是这样，希望您已经理解了 Qt Widgets 应用程序和 Qt Quick 应用程序之间的相似之处和不同之处。现在您可以从这两者中选择最适合您项目需求的选项了！

# 总结

在本章中，我们学习了 Qt Quick 是什么，以及如何使用 QML 语言创建程序。在接下来的章节中，我们将学习如何将我们的 Qt 项目轻松导出到不同的平台。让我们开始吧！
