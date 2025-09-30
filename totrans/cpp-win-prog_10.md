# 第10章。框架

本书剩余章节解释了Small Windows实现的细节。本章涵盖以下主题：

+   Small Windows类的概述

+   我们在本书开头介绍的Hello World应用程序的示例，使用Win32 API编写

+   `MainWindow`和`WinMain`函数

+   Small Windows主要类的实现：`Application`、`Window`和`Graphics`

# Small Windows的概述

这里是Small Windows类的简要描述：

| **章节** | **类** | **描述** |
| --- | --- | --- |
| 10 | `Application` | 这是Small Windows的`main`类。它管理消息循环和Windows类的注册。 |
| 10 | `Window` | 这是根`Window`类。它创建单个窗口并提供基本的窗口功能，如鼠标、触摸和键盘输入、绘图、缩放、计时器、焦点、大小和坐标系。 |
| 10 | `Graphics` | 这是用于在窗口客户端区域绘制线条、矩形、椭圆和文本的类。 |
| 11 | `Document` 扩展 `Window` | 这扩展了窗口以包含文档功能，如滚动、光标处理和拖放文件。 |
| 11 | `Menu` | 这处理菜单栏、菜单、菜单项和菜单分隔符。 |
| 11 | `Accelerator` | 这从菜单项文本中提取加速器信息。 |
| 11 | `StandardDocument` 扩展 `Document` | 这提供了一个基于文档的框架，包含常见的**文件**、**编辑**和**帮助**菜单项。 |
| 12 | `Size` `Point` `Rect` | 这些是处理二维点（x和y）、大小（宽度和高度）或矩形四个角的辅助类。 |
| 12 | `Font` | 这封装了`LOGFONT`结构，该结构包含有关字体名称、大小以及是否为粗体或斜体的信息。 |
| 12 | `Cursor` | 这设置光标并提供一组标准光标。 |
| 12 | `DynamicList` 模板 | 这是一个动态大小的列表和一组回调方法。 |
| 12 | `Tree` 模板 | 这是一个树结构，其中每个节点都有一个（可能为空）子节点列表。 |
| 12 | `InfoList` | 这是一个通用信息列表，可以转换成和从内存缓冲区。 |
| 13 | `Registry` | 这提供了一个与Windows注册表的接口。 |
| 13 | `Clipboard` | 这提供了一个与Windows剪贴板的接口。 |
| 13 | `StandardDialog` | 这显示保存和打开文件、选择字体或颜色以及打印的标准对话框。 |
| 13 | `PreviewDocument` 扩展 `Document` | 这设置了一个逻辑大小固定（无论其物理大小如何）的文档。 |
| 14 | `Dialog` 扩展 `Window` | 这提供了一个模态对话框。下面的控件被添加到对话框中。 |
| 14 | `Control` 抽象 | 这是对话框控件的基础类。 |
| 14 | `ButtonControl` 扩展 `Control` | 这是按钮控件的基础类。 |
| 14 | `GroupBox`、`PushButton`、`CheckBox`、`RadioButton` 扩展 `ButtonControl` | 这些是用于分组框、按钮、复选框和单选按钮的类。 |
| 14 | `ListControl` 扩展 `Control` | 这是列表控件的基础类。 |
| 14 | `ListBox`、`MultipleListBox` 扩展 `ListControl` | 这些是用于单选和复选列表框的类。 |
| 14 | `ComboBox` 扩展 `Control` | 这是一个组合（下拉）框的类。 |
| 14 | `Label` 扩展 `Control` | 这是一个简单的标签类，通常用作 `TextField` 的提示。 |
| 14 | `TextField` 模板扩展 `Control` | 这是一个可编辑字段的类，其中转换器可以在字符串和任何类型之间进行转换。 |
| 14 | `Converter` 模板 | 这是一个可以指定为任何类型的转换器类。 |
| 14 | `PageSetupDialog` 扩展 `Dialog` | 这是一个用于页面设置设置的对话框，例如页边距、页眉和页脚文本。 |
| 14 | `PageSetupInfo` | 这包含页面设置信息，我们之前已经看到过。 |

# "Hello" 窗口用于 Win32 API

首先，让我们看看这本书第一章中的 Hello 应用程序。以下代码片段是使用 Win32 API 直接编写的相同应用程序，没有使用 Small Windows。请注意，代码是用 C 编写的，而不是 C++，因为 Win32 API 是一个 C 函数库，而不是 C++ 类库。正如你所看到的，与第一章中的应用程序相比，代码要复杂得多。

如果看起来很复杂，请不要担心。它的目的实际上是演示 Win32 API 的复杂性；我们将在本章和下一章中讨论细节。

**MainWindow.c**

[PRE0]

当应用程序开始执行时，会调用 `WinMain` 方法。它对应于标准 C 中的 `main`。

[PRE1]

首先，我们需要为我们的窗口注册 `Windows` 类。请注意，`Windows` 类不是 C++ 类：

[PRE2]

当窗口在水平和垂直方向上改变大小的时候，`Windows` 类的样式将被重新绘制：

[PRE3]

窗口的图标是标准应用程序图标，光标是标准箭头光标，客户端区域的背景是白色。

[PRE4]

`WindowProc` 函数是一个回调函数，每次窗口收到消息时都会被调用：

[PRE5]

`Windows` 类的名称是 `window`，在这里用于 `CreateWindowEx` 调用中：

[PRE6]

`CreateWindowEx` 方法创建一个具有默认位置和大小的窗口。请注意，我们可以使用相同的 `Windows` 类创建许多窗口：

[PRE7]

`GetMessage` 方法等待下一个消息，该消息被翻译并发送到具有输入焦点的窗口。`GetMessage` 方法对所有消息返回 `true`，除了退出消息，该消息在用户关闭窗口时最终发送：

[PRE8]

在绘制客户端区域时，我们需要创建一个绘图结构和设备上下文，这是通过 `BeginPaint` 创建的：

[PRE9]

由于我们想使用逻辑单位（毫米的百倍），我们需要通过调用 `SetWindowExtEx` 和 `SetViewportExtEx` 来设置设备上下文：

[PRE10]

由于我们还想考虑滚动动作，所以我们也会调用 `SetWindowOrgEx`:

[PRE11]

此外，由于我们想考虑滚动动作，我们调用 `SetWindowOrgEx` 来设置客户端区域的逻辑原点：

[PRE12]

我们需要设置一个 `LOGFONT` 结构来创建12磅粗体的 `Times New Roman` 字体：

[PRE13]

由于我们使用的是毫米级的逻辑单位，一个排版点等于1英寸除以72，1英寸等于25.4毫米。我们将字体大小乘以2540然后除以72：

[PRE14]

当我们在客户端区域使用字体写入文本时，我们需要间接创建字体并将其添加为图形对象。我们还需要保存先前的对象以便稍后恢复：

[PRE15]

文本颜色为黑色，背景颜色为白色。`RGB` 是一个宏，它将颜色的红色、绿色和蓝色部分转换为一个 `COLORREF` 值：

[PRE16]

最后，`DrawText` 在客户端区域的中间绘制文本：

[PRE17]

由于字体是系统资源，我们需要恢复先前的字体对象并删除新的字体对象。我们还需要恢复绘图结构：

[PRE18]

由于我们已经处理了 `WM_PAINT` 消息，我们返回零。

[PRE19]

对于除 `WM_PAINT` 以外的所有消息，我们调用 `DefWindowProc` 来处理消息：

[PRE20]

# 主窗口函数

在常规 C 和 C++ 中，应用程序的执行从 `main` 函数开始。然而，在小型 Windows 中，`main` 被替换为 `MainWindow`。`MainWindow` 由小型 Windows 的用户为每个项目实现。其任务是定义应用程序名称并创建主窗口对象。

**MainWindow.h**

[PRE21]

# WinMain 函数

在 Win32 API 中，`WinMain` 是与 `main` 等效的函数。每个应用程序都必须包含 `WinMain` 函数的定义。为了使小型 Windows 工作，`WinMain` 作为小型 Windows 的一部分实现，而 `MainWindow` 必须由小型 Windows 的用户为每个项目实现。总结一下，这里有三种主函数：

| **常规 C/C++** | **Win32 API** | **小型 Windows** |
| --- | --- | --- |
| main | WinMain | MainWindow |

`WinMain` 函数由 Windows 系统调用，并接受以下参数：

+   `instanceHandle`：这包含应用程序的句柄

+   `prevInstanceHandle`：由于向后兼容性，它存在，但始终为 `null`

+   `commandLine`：这是一个以空字符终止的字符（`char`，不是 `TCHAR`）数组，包含应用程序的参数，由空格分隔

+   `commandShow`：这包含主窗口的首选外观

**WinMain.cpp**

[PRE22]

`WinMain` 函数执行以下任务：

+   它通过调用 `GenerateArgumentList` 将命令行中空格分隔的单词划分为一个 `String` 列表。请参阅[第12章](ch12.html "第12章。辅助类")，*辅助类*，以了解 `CharPtrToGenericString` 和 `Split` 的定义。

+   它实例化一个 `Application` 对象。

+   它调用 `MainWindow` 函数，该函数创建应用程序的主窗口并设置其名称。

+   它调用 `Application` 的 `RunMessageLoop` 方法，该方法继续处理 Windows 消息，直到收到退出消息。

[PRE23]

# `Application` 类

`Application` 类处理应用程序的消息循环。消息循环等待从 Windows 系统接收下一个消息并将其发送到正确的窗口。`Application` 类还定义了 `Window`、`Document`、`StandardDocument` 和 `Dialog` C++ 类的 `Windows` 类（它们不是 C++ 类）。由于 `Application` 不打算实例化，因此类的字段是静态的。

从 Small Windows 的这个点开始，Small Windows 实现的每一部分都包含在 `SmallWindows` 命名空间中。命名空间是 C++ 的一个特性，用于封装类和函数。我们之前看到的 `MainWindow` 的声明不包括在 `Smallwindows` 命名空间中，因为 C++ 语言规则规定它不能包含在命名空间中。`WinMain` 的定义也不包含在命名空间中，因为它需要放在命名空间外部才能被 Windows 系统调用。

**Application.h**

[PRE24]

`RegisterWindowClasses` 方法为 `Window`、`Document`、`StandardDocument` 和 `Dialog` C++ 类定义 Windows 类。`RunMessageLoop` 方法运行 Windows 消息系统的消息循环。它等待下一个消息并将其发送到正确的窗口。当接收到特殊的退出消息时，它会中断消息循环，从而导致 `Application` 类的终止：

[PRE25]

在 Windows 中，每个应用程序都持有应用程序实例的 **句柄**。句柄在 Win32 API 中很常见，用于访问 Windows 系统的对象。它们类似于指针，但提供标识而不透露任何位置信息。

在创建以下 `Window` 类的构造函数中以及在第 14 章[对话框、控件和页面设置](ch14.html "第14章。对话框、控件和页面设置")的“标准对话框”部分显示标准对话框时，使用实例句柄（`HINSTANCE` 类型）：

[PRE26]

应用程序名称由每个应用程序设置，并通过标准 **文件**、**帮助** 和 **关于** 菜单、**打开** 和 **保存** 对话框以及注册表进行引用：

[PRE27]

当用户关闭窗口时，会引用应用程序主窗口的指针。如果是主窗口，则应用程序退出。此外，当用户选择 **退出** 菜单项时，在应用程序退出之前会关闭主窗口：

[PRE28]

**Application.cpp**

[PRE29]

## Win32 API的Windows类

`Windows`类在`Application`中注册。一个Windows类只需要注册一次。注册后，可以为每个`Windows`类创建多个窗口。再次注意，窗口类不是C++类。每个`Windows`类通过其名称：`lpszClassName`存储。`lpfnWndProc`字段定义了接收来自消息循环的窗口消息的独立函数。每个窗口都允许双击以及水平和垂直重绘样式，这意味着每次用户更改窗口大小时，都会向窗口发送`WM_PAINT`消息并调用`OnPaint`方法。此外，每个窗口在其右上角都有标准的应用程序图标和标准的箭头光标。客户端区域为白色，除了对话框，其客户端区域为浅灰色：

[PRE30]

## 消息循环

`RunMessageLoop`方法保留了经典的Windows消息循环。有两种情况：如果主窗口指针指向`Window`类的对象，我们只需使用Win32 API函数`GetMessage`、`TranslateMessage`和`DispatchMessage`处理消息队列，而不关心加速器。然而，如果它指向`Document`或其任何子类的对象，消息循环变得更加复杂，因为我们需要考虑加速器：

[PRE31]

如果主窗口指针指向`Document`对象或其任何子类，我们将在`Document`中定义的加速器表中设置一个缓冲区，我们在消息循环中使用它。Win32 API的`TranslateAccelerator`函数查找加速器并决定是否将按键消息视为与加速器关联的菜单项：

[PRE32]

`TranslateAccelerator`方法需要一个ACCEL结构的数组，因此我们将加速器集转换为数组：

[PRE33]

当使用加速器数组时，它将被删除：

[PRE34]

当消息循环完成后，我们返回最后一条消息：

[PRE35]

# `Window`类

`Window`类是文档类的根类；它处理基本窗口功能，如计时器、输入焦点、坐标变换、窗口大小和位置、文本度量以及消息框以及鼠标、键盘和触摸屏输入。此外，`Window`定义了窗口样式和外观、按钮、图标和坐标系统的枚举：

**Window.h**

[PRE36]

存在大量的窗口样式。窗口可能配备有边框、厚边框、滚动条或最小化和最大化框：

[PRE37]

窗口可以以最小化、最大化或正常模式显示：

[PRE38]

鼠标可以按下左键、中键和右键。鼠标滚轮可以向上或向下滚动：

[PRE39]

有四种类型的坐标系如下：

+   `LogicalWithScroll`：在这种情况下，每个单位是毫米的一百分之一，无论物理屏幕分辨率如何，都考虑当前的滚动条设置

+   `LogicalWithoutScroll`：这与`LogicalWithScroll`相同，只是忽略了滚动条设置

+   `PreviewCoordinate`：在这种情况下，窗口客户端区域始终保持特定的逻辑大小，这意味着当窗口大小改变时，逻辑单位的大小也会改变

[PRE40]

消息框配备了按钮组合、图标和答案。请注意，对应于**OK**按钮的答案在`Answer`枚举中命名为`OkAnswer`，以避免与`ButtonGroup`枚举中的`OK`按钮名称冲突：

[PRE41]

`OnPaint`和`OnPrint`的默认定义都调用`OnDraw`。为了区分这两种情况，`OnDraw`参数的值为`Paint`或`Print`：

[PRE42]

第一个`Window`构造函数是公开的，用于直接创建窗口时使用。`pageSize`字段指的是窗口客户端区域的大小。构造函数还接受指向窗口父窗口的指针（如果没有父窗口则为`null`），窗口的基本样式和扩展样式，以及其初始外观、位置和大小。如果位置或大小为零，窗口将根据系统的默认设置定位或调整尺寸。

注意在`PreviewCoordinate`中文档和窗口大小之间的区别：文档大小是窗口坐标系定义的单位中的客户端区域大小，而窗口的大小和位置是在父窗口的坐标系中给出，如果没有父窗口，则使用设备单位。此外，文档大小指的是客户端区域的大小，而窗口大小指的是整个窗口的大小：

[PRE43]

第二个构造函数是受保护的，用于由子类的构造函数调用。与第一个构造函数相比，它的区别在于它将`window`类的名称作为其第一个参数。根据`Application`类的定义，类名可以是`Window`、`Document`、`StandardDocument`或`Dialog`：

[PRE44]

在绘制客户端区域、在逻辑单位和设备单位之间转换以及计算文本大小时使用**设备上下文**。它是连接到窗口的客户端区域或打印机的连接。然而，由于它附带了一套用于绘制图形对象文本的函数，它也可以被视为一个绘图工具箱。但是，在使用之前，它需要根据当前坐标系进行准备和调整：

[PRE45]

析构函数会销毁窗口并退出应用程序，如果窗口是应用程序的主窗口：

[PRE46]

窗口可以是可见的或不可见的；它也可以被启用，以便捕获鼠标、触摸和键盘输入：

[PRE47]

当用户更改窗口大小或移动窗口时，会调用`OnSize`和`OnMove`方法。大小和位置以逻辑坐标给出。当用户在消息框中按下*帮助*按钮的*F1*键时，会调用`OnHelp`方法。这些方法旨在被子类覆盖，并且它们的默认行为是不做任何事情：

[PRE48]

`WindowHandle`方法返回Win32 API窗口句柄，它被标准对话框函数使用。`ParentWindowPtr`方法返回父窗口的指针，它是`null`，表示没有父窗口。`SetHeader`方法设置窗口的标题，该标题在窗口的上边框中可见：

[PRE49]

窗口的客户端区域根据缩放因子进行缩放；1.0对应于正常大小：

[PRE50]

只要`timerId`参数的值不同，就可以设置或删除多个计时器。`OnTimer`方法根据毫秒间隔被调用；它的默认行为是不做任何事情。

[PRE51]

`SetFocus`方法将输入焦点设置到这个窗口。输入焦点将键盘输入和剪贴板指向窗口。然而，鼠标指针可能指向另一个窗口。之前拥有输入焦点的窗口会失去焦点；在给定时间内只能有一个窗口拥有焦点。`HasFocus`方法返回`true`，如果窗口有输入焦点。

[PRE52]

当窗口获得或失去输入焦点时，会调用`OnGainFocus`和`OnLoseFocus`方法。它们旨在被子类覆盖，并且它们的默认行为是不做任何事情。

[PRE53]

在Windows中，鼠标被视为有三个按钮，即使它实际上没有这样做。可以按下或释放鼠标按钮，并且可以移动鼠标。当用户按下或释放鼠标按钮或至少按下一个按钮移动鼠标时，会调用`OnMouseDown`、`OnMouseUp`和`OnMouseMove`方法。用户可以同时按下***Shift***或***Ctrl***键，在这种情况下`shiftPressed`或`controlPressed`为`true`：

[PRE54]

用户还可以双击鼠标按钮，在这种情况下会调用`OnDoubleClick`。双击的定义由Windows系统决定，可以在控制面板中设置。当用户单击按钮时，会调用`OnMouseDown`，如果可能发生鼠标移动，则随后调用`OnMouseMove`，最后调用`OnMouseUp`。然而，在双击的情况下，不会调用`OnMouseDown`，其调用被`OnDoubleClick`所取代：

[PRE55]

当用户向上或向下滚动鼠标滚轮一步时，会调用`OnMouseWheel`方法。

[PRE56]

当用户触摸屏幕时，会调用`OnTouchDown`、`OnTouchMove`和`OnTouchUp`方法。与鼠标点击不同，用户可以同时触摸屏幕的多个位置。因此，参数是点的列表而不是单个点。这些方法打算由子类重写。它们的默认行为是模拟每个触摸点的一个鼠标点击，没有按钮按下，且没有按下***Shift***或***Ctrl***键：

[PRE57]

当用户按下和释放键时，会调用`OnKeyDown`和`OnKeyUp`方法。如果键是一个图形字符（ASCII编号在32到127之间，包括127），则在之间会调用`OnChar`。`OnKeyDown`和`OnKeyUp`方法返回`bool`；其思路是，如果使用了键，则方法返回`true`。如果没有使用，则返回`false`，调用方法可以自由使用该键，例如，控制滚动操作：

[PRE58]

当窗口的客户区域需要部分或完全重绘时，会调用`OnPaint`方法，而当用户选择**打印**菜单项时，会调用`OnPrint`方法。在这两种情况下，默认定义都会调用`OnDraw`，它执行实际的绘制；当由`OnPaint`调用时，`drawMode`为`Paint`，当由`OnPrint`调用时，`drawMode`为`Print`。其思路是我们让`OnPaint`和`OnPrint`执行与绘画和打印相关的特定操作，并调用`OnDraw`进行共同绘制。`Graphics`类将在下一节中描述：

[PRE59]

`Invalidate`方法使客户区域无效，部分或全部；也就是说，它准备由`OnPaint`或`OnDraw`重绘的区域。如果`clear`为`true`，则首先清除该区域（用窗口客户端颜色绘制）。`UpdateWindow`方法强制重绘客户区域中被无效化的部分：

[PRE60]

当用户尝试关闭窗口时，会调用`OnClose`方法；其默认行为是调用`TryClose`。如果`TryClose`返回`true`（在其默认定义中确实如此），则窗口将被关闭。如果发生这种情况，会调用`OnDestroy`，其默认行为是不做任何操作：

[PRE61]

以下方法在设备单位和逻辑单位之间转换`Point`、`Rectangle`或`Size`对象。它们是受保护的，因为它们打算只由子类调用：

[PRE62]

以下方法在设备单位中获取或设置窗口和客户区域的大小和位置：

[PRE63]

以下方法根据窗口的坐标系，在逻辑单位中获取或设置窗口和客户区域的逻辑大小和位置：

[PRE64]

`CreateTextMetric`方法初始化并返回一个Win32 API `TEXTMETRIC`结构，然后由文本度量方法使用，以计算文本的逻辑大小。它是私有的，因为它打算只由`Window`方法调用：

[PRE65]

以下方法计算并返回具有给定字体的字符或文本的宽度、高度、上升或平均宽度，单位为逻辑单位：

[PRE66]

`MessageBox` 方法显示一个包含消息、标题、一组按钮、图标以及可选的 **帮助** 按钮的消息框：

[PRE67]

`pageSize` 字段存储窗口客户端在 `PreviewCoordinate` 坐标系中的逻辑大小，该坐标系用于在逻辑坐标和设备坐标之间转换坐标。在 `LogicalWithScroll` 和 `LogicalWithoutScroll` 坐标系中，`pageSize` 存储文档的逻辑大小，这不一定等于客户端区域的大小，且在窗口大小调整时不会改变。它是受保护的，因为它在下一章的 `Document` 和 `StandardDocument` 子类中也被使用：

[PRE68]

在上一节中，有一个应用程序实例的句柄。`windowHandle` 是一个类型为 `HWND` 的 Win32 API 窗口句柄；`parentPtr` 是父窗口的指针，如果没有父窗口，则为 `null`：

[PRE69]

窗口选择的坐标系存储在 `system` 中。`zoom` 字段存储窗口的缩放因子，其中 1.0 是默认值：

[PRE70]

每次窗口接收消息时都会调用 `WindowProc` 方法。它是 `Window` 的朋友，因为它需要访问其私有成员：

[PRE71]

最后，`WindowMap` 将 `HWND` 句柄映射到 `Window` 指针，这些指针在 `WindowProc` 中如下使用：

[PRE72]

**Window.cpp**

[PRE73]

## 初始化

第一个构造函数只是用类名 `window` 调用第二个构造函数：

[PRE74]

第二个构造函数初始化 `parentPtr`、`system` 和 `pageSize` 字段：

[PRE75]

如果窗口是子窗口（父指针不是 `null`），则将其坐标转换为父窗口的坐标系：

[PRE76]

Win32 API 窗口创建过程分为两个步骤。首先，需要注册一个 Windows 类，这已经在之前的 `Application` 构造函数中完成。然后，使用 `Windows` 类名调用 Win32 API 的 `CreateWindowEx` 函数，该函数返回窗口的句柄。如果大小或位置为零，则使用默认值：

[PRE77]

为了使 `WindowProc` 能够接收消息并识别接收窗口，句柄被存储在 `WindowMap` 中：

[PRE78]

调用 Win32 API 函数 `ShowWindow` 和 `RegisterTouchWindow` 以根据 `windowShow` 参数使窗口可见，并使窗口能够响应触摸移动：

[PRE79]

析构函数调用 `OnDestroy` 并从 `windowMap` 中删除窗口。如果窗口有父窗口，它将接收输入焦点：

[PRE80]

如果窗口是应用程序的主窗口，则调用 Win32 API 的 `PostQuitMessage` 函数。它发布一个退出消息，该消息最终由 `Application` 类中的 `RunMessageLoop` 捕获，从而终止执行。最后，销毁窗口：

[PRE81]

## 标题和可见性

`ShowWindow` 和 `EnableWindow` 方法使用窗口句柄作为第一个参数调用 Win32 API 的 `ShowWindow` 和 `EnableWindow` 函数：

[PRE82]

注意，`EnableWindow` 的第二个参数是 Win32 API 类型 `BOOL` 的值，这不一定与 C++ 类型 `bool` 相同。因此，由于 `enable` 持有类型 `bool`，我们需要将其转换为 `BOOL`：

[PRE83]

`SetHeader` 方法通过调用 Win32 API 函数 `SetWindowText` 来设置窗口的标题。由于 `headerText` 是一个 `String` 对象，而 `SetWindowText` 需要一个 C 字符串（一个以零终止的字符指针）作为参数，因此我们需要调用 `c_str` 函数：

[PRE84]

`SetTimer` 和 `DropTimer` 方法通过调用 Win32 API 函数 `SetTimer` 和 `KillTimer` 来开启和关闭具有给定标识符的计时器。`SetTimer` 调用中的间隔以毫秒为单位给出：

[PRE85]

`SetFocus` 方法通过调用相应的 Win32 API 函数 `SetFocus` 来设置焦点。`HasFocus` 方法通过调用 `GetFocus` Win32 API 函数返回 `true`，如果窗口通过该函数获得了输入焦点，该函数返回窗口句柄，与窗口句柄进行比较：

[PRE86]

## 触摸屏

`OnTouchDown`、`OnTouchMove` 和 `OnTouchUp` 的默认行为是调用每个触摸点的相应鼠标输入方法，没有按钮按下，也没有 ***Shift*** 或 ***Ctrl*** 键被按下：

[PRE87]

在现代屏幕上，用户可以以类似于鼠标点击的方式触摸屏幕。然而，用户可以同时触摸屏幕的几个位置，并且其位置存储在一个点列表中。`OnTouch` 方法是一个辅助方法，当用户触摸屏幕时调用 `OnTouchDown`、`OnTouchMove` 和 `OnTouchUp`。它创建一个逻辑坐标中的点列表：

[PRE88]

如果触摸标识符不等于输入数组中的第一个值，我们有一个触摸下事件；如果它相等，我们有一个触摸移动事件：

[PRE89]

## 无效化和窗口更新

当窗口的客户区域需要（部分或全部）重绘时，会调用 `Invalidate` 方法之一。`Invalidate` 方法调用 Win32 API 函数 `InvalicateRect`，该函数在调用 `UpdateWindow` 时发送一个消息，导致调用 `OnPaint`。`clear` 参数指示在重绘之前是否应该清除（用窗口客户区域的颜色重绘）无效区域，这通常是情况。类似于我们之前看到的 `EnableWindow` 方法，我们需要将 `clear` 从类型 `bool` 转换为 `BOOL`：

[PRE90]

`Invalidate` 方法在调用 Win32 API 函数 `InvalidateRect` 之前将区域从逻辑坐标转换为设备坐标，并将大小存储在 `RECT` 结构中：

[PRE91]

`UpdateWindow` 方法调用 Win32 API 函数 `UpdateWindow`，这最终导致调用 `OnPaint`：

[PRE92]

## 准备设备上下文

当绘制窗口的客户端区域时，我们需要一个设备上下文，我们需要根据坐标系来准备它，以便使用逻辑坐标进行绘制。Win32 API 函数 `SetMapMode` 设置逻辑坐标系统的映射模式。`MISOTROPIC` 强制 *x* 和 *y* 轴具有相同的单位长度（导致非椭圆形的圆），这对于 `LogicalWithScroll` 和 `LogicalWithoutScroll` 系统是合适的，而 `MANISOTROPIC` 允许不同的单位长度，这对于 `PreviewCoordinate` 系统是合适的。我们通过调用 `SetWindowExtEx` 函数建立逻辑和设备系统之间的映射，它接受客户端区域的逻辑大小，以及调用 `SetViewportExtEx` 函数，它接受其物理（设备）大小。

在 `PreviewCoordinate` 坐标系的情况下，我们只需将客户端区域（`pageSize`）的逻辑大小与由 Win32 API 函数 `GetClientRect` 给出的设备大小（`clientDeviceRect`）相匹配，从而使得客户端区域始终具有相同的逻辑大小，无论其实际大小如何：

[PRE93]

在逻辑坐标系的情况下，我们需要找到逻辑坐标（数百毫米）和设备坐标（像素）之间的比率。换句话说，我们需要确定像素的逻辑大小。我们可以通过调用带有 `HORZSIZE` 和 `VERTSIZE` 的 Win32 API 函数 `GetDeviceCaps` 来找到屏幕上的像素数，以及使用 `HORZRES` 和 `VERTRES` 的毫米级屏幕大小。由于我们的逻辑单位是数百毫米，我们需要将逻辑大小乘以 100。我们还需要考虑窗口的缩放因子，这通过将物理大小乘以 `zoom` 来实现。

注意，只有在 `PreviewCoordinate` 系统中，客户端区域始终具有相同的逻辑大小。在其他系统中，当窗口大小改变时，逻辑大小也会改变。在 `LogicalWithScroll` 和 `LogicalWithoutScroll` 中，逻辑单位始终相同：数百毫米：

[PRE94]

在 `LogicalWithScroll` 逻辑坐标系的情况下，我们还需要通过调用 Win32 API 函数 `SetWindowOrg` 来根据当前的滚动设置调整窗口的原点：

[PRE95]

## 单位转换

`DeviceToLogical` 方法通过准备设备上下文并调用 Win32 API 函数 `DPtoLP`（设备点到逻辑点）将一个点、矩形或大小的设备坐标转换为逻辑坐标。请注意，我们通过调用 Win32 API 函数 `GetDC` 来建立设备上下文，并且需要通过调用 `ReleaseDC` 来返回它。另外，请注意，我们需要将 `Point` 对象转换为 `POINT` 结构，然后再转换回来，因为 `DPtoLP` 接受一个指向 `POINT` 的指针：

[PRE96]

在转换矩形时，我们使用点方法转换其左上角和右下角。在转换大小时，我们创建一个矩形，调用矩形方法，然后将矩形转换为大小：

[PRE97]

`LogicalToDevice` 方法将点、矩形或大小从逻辑坐标转换为设备坐标，调用 Win32 API 函数 `LPtoDP`（逻辑点到设备点）的方式与早期方法相同。唯一的区别是它们调用 `LPtoDP` 而不是 `DPtoLP`：

[PRE98]

## 窗口大小和位置

`GetWindowDevicePosition`、`SetWindowDevicePosition`、`GetWindowDeviceSize`、`SetWindowDeviceSize` 和 `GetClientDeviceSize` 方法调用相应的 Win32 API 函数 `GetWindowRect`、`GetClientRect` 和 `SetWindowPos`：

[PRE99]

`GetWindowPosition`、`SetWindowPosition`、`GetWindowSize`、`SetWindowSize` 和 `GetClientSize` 方法与 `LogicalToDevice` 或 `DeviceToLogical` 一起调用相应的设备方法：

[PRE100]

## 文本度量

给定一个字体，`CreateTextMetric` 创建一个包含字体字符的高度、基线上升线和平均宽度的度量结构。`CreateFontIndirect` 和 `SelectObject` 方法为 `GetTextExtentPoint` 准备字体：

[PRE101]

注意，`CreateFontIndirect` 必须与 `DeleteObject` 匹配，并且第一个 `SelectObject` 调用必须与第二个 `SelectObject` 调用匹配以重新安装原始对象：

[PRE102]

此外，请注意，从 `GetDC` 收到的设备上下文必须使用 `ReleaseDC` 释放：

[PRE103]

`GetCharacterHeight`、`GetCharacterAscent` 和 `GetCharacterAverageWidth` 方法调用 `CreateTextMetric` 并返回相关信息：

[PRE104]

`GetCharacterWidth` 方法调用 `GetTextExtentPoint` 以确定给定字体的字符宽度。由于字体高度是以排版点（1 点 = 1/72 英寸 = 1/72 * 25.4 毫米 ≈≈ 0.35 毫米）给出的，并且需要以毫米为单位给出，我们调用 `PointsToLogical`。类似于我们在 `CreateTextMetric`、`CreateFontIndirect` 和 `SelectObject` 中所做的，`CreateFontIndirect` 和 `SelectObject` 方法为 `GetTextExtentPoint` 准备字体：

[PRE105]

## 关闭窗口

当用户尝试关闭窗口时，如果 `TryClose` 返回 `true`，则删除 `Window` 对象（`this`）：

[PRE106]

## `MessageBox` 方法

`MessageBox` 方法显示一个包含标题、消息、按钮组合（**确定**、**确定-取消**、**重试-取消**、**是-否**、**是-否-取消**、**取消-重试-继续**或**中止-重试-忽略**）、可选图标（**信息**、**停止**、**警告**或**问题**）和可选的 **帮助** 按钮的消息框。它返回 **确定答案**（因为 **确定** 已经被 `ButtonGroup` 枚举占用）、**取消**、**是**、**否**、**重试**、**继续**、**中止** 或 **忽略**：

[PRE107]

当通过 `Window` 类构造函数中的 `CreateWindowEx` 调用创建窗口时，由 `Application` 类构造函数先前给出的 `Windows` 类的名称被包含在内。当类被注册时，还会提供一个独立的函数。对于 `Window` 类，该函数是 `WindowProc`，因此每当窗口收到消息时都会调用它：

`wordParam` 和 `longParam` 参数（`WPARAM` 和 `LPARAM` 都是 4 字节）包含消息特定的信息，这些信息可能被分为低字和高字（2 字节），使用 `LOWORD` 和 `HIWORD` 宏来区分：

[PRE108]

首先，我们需要通过在静态字段 `WindowMap` 中查找句柄来找到与窗口句柄关联的 `Window` 对象：

[PRE109]

当接收到 `WSETFOCUS`、`WKILLFOCUS` 和 `WTIMER` 消息时，`Window` 中的相应方法被简单地调用。当消息被处理完毕后，它们不需要进一步处理；因此，返回零：

[PRE110]

计时器的身份（`SetTimer` 和 `DropTimer` 中的 `timerId` 参数）存储在 `wordParam` 中：

[PRE111]

当接收到 `WMOVE` 和 `WSIZE` 消息时，存储在 `longParam` 中的 `Point` 值是以设备单位给出的，需要通过在 `Window` 中的 `OnMove` 和 `OnSize` 调用中调用 `DeviceToLogical` 来转换为逻辑单位：

[PRE112]

如果用户在消息框中按下 ***F1*** 键或 **帮助** 按钮，则发送 `WM_HELP` 消息。我们在 `Window` 中调用 `OnHelp`：

[PRE113]

在处理鼠标或键盘输入消息时，决定用户是否同时按下 ***Shift*** 或 ***Ctrl*** 键是有用的。这可以通过调用 Win32 API 函数 `GetKeyState` 来实现，如果使用 `VK_SHIFT` 或 `VK_CONTROL` 调用，则当键被按下时，它返回一个小于零的整数值：

[PRE114]

如果 `OnKeyDown` 返回 `true`，则表示键消息已被处理，我们返回零。如果它返回 `false`，则将调用如这里所示的 Win32 API 函数 `DefWindowProc`，该函数将进一步处理消息：

[PRE115]

如果按下的键是一个图形字符（ASCII 码在 32 到 127 之间，包括 127），则调用 `OnChar`：

[PRE116]

所有存储在 `longParam` 中的鼠标输入点都是以设备坐标给出的，需要通过 `DeviceToLogical` 转换为逻辑坐标。鼠标按下消息通常随后是相应的鼠标抬起消息。不幸的是，如果用户在一个窗口中按下鼠标按钮并在另一个窗口中释放它，那么鼠标抬起消息将被发送到另一个窗口。然而，可以通过 Win32 API 函数 `SetCapture` 解决这个问题，该函数确保在调用 `ReleaseCapture` 之前将每个鼠标消息发送到窗口：

[PRE117]

当用户移动鼠标时，他们可能同时按下按钮组合，这些按钮存储在 `buttonMask` 中：

[PRE118]

注意，`ReleaseCapture`是在鼠标抬起方法结束时被调用的，目的是释放窗口的鼠标消息，并使鼠标消息能够发送到其他窗口：

[PRE119]

当发送触摸消息时，会调用`OnTouch`，这需要窗口在设备单位中的位置：

[PRE120]

在响应绘图消息创建设备上下文时，我们使用Win32 API 函数`BeginPaint`和`EndPaint`而不是`GetDC`和`ReleaseDC`来处理设备上下文。然而，设备上下文仍然需要为窗口的坐标系统做好准备，这是通过`PrepareDeviceContext`完成的：

[PRE121]

当用户尝试通过点击右上角的关闭框来关闭窗口时，会调用`OnClose`。它调用`TryClose`，如果`TryClose`返回true，则关闭窗口：

[PRE122]

如果我们达到这一点，Win32 API 函数`DefWindowProc`会被调用，它执行默认的消息处理：

[PRE123]

# `Graphics`类

`Graphics`类是一个设备上下文的包装类。它还提供了绘制线条、矩形和椭圆；写入文本；保存和恢复图形状态；设置设备上下文的起点；以及裁剪绘图区域的功能。构造函数是私有的，因为`Graphics`对象旨在仅由小窗口内部创建。

**Graphics.h**

[PRE124]

在绘制线条时，可以是实线、虚线、点线、点划线，以及点划双线：

[PRE125]

`Save`方法保存`Graphics`对象当前的状态，而`Restore`恢复它：

[PRE126]

`SetOrigin`方法设置坐标系统的原点，而`IntersectClip`限制要绘制的区域：

[PRE127]

以下方法绘制线条、矩形、椭圆，并写入文本：

[PRE128]

`GetDeviceContextHandle`方法返回由`Graphics`对象包装的设备上下文：

[PRE129]

`windowPtr`字段持有指向要绘制客户端区域的窗口的指针，而`deviceContextHandle`持有设备上下文的句柄，类型为`HDC`：

[PRE130]

`WindowProc`和`DialogProc`函数是`Graphics`类的朋友，因为它们需要访问其私有成员。对于`StandardDialog`类的`PrintDialog`方法也是如此：

[PRE131]

**Graphics.cpp**

[PRE132]

构造函数初始化窗口指针和设备上下文：

[PRE133]

有时，可能希望使用`Save`保存`Graphics`对象的当前状态，它返回一个可以用来使用`Restore`恢复`Graphics`对象的身份号码：

[PRE134]

坐标系的默认原点（x = 0 和 y = 0）是窗口客户端区域的左上角。这可以通过`SetOrigin`来改变，它接受新的原点在逻辑单位中。Win32 API 函数`SetWindowOrgEx`设置新的原点：

[PRE135]

可以使用`IntersectClip`限制要绘制的客户端区域的部分，结果是在给定矩形外的区域不受影响。Win32 API 函数`IntersectClip`设置限制区域：

[PRE136]

可以使用笔绘制线条、矩形和椭圆，笔是通过 Win32 API 函数 `CreatePen` 和 `SelectObject` 获取的。请注意，我们保存了上一个对象以便稍后恢复：

[PRE137]

顺便说一下，使用 `MoveToEx` 和 `LineTo` 将笔移动到起点并绘制到终点的技术被称为**海龟**图形，指的是笔在客户端区域内上提或放下移动的乌龟：

[PRE138]

与 `Window` 中的 `CreateTextMetrics` 和 `GetCharacterWidth` 类似，我们需要选择上一个对象并恢复笔：

[PRE139]

在绘制矩形时，我们需要一个实心笔和一个空心画刷，我们使用带有 `LOGBRUSH` 结构参数的 Win32 API 函数 `CreateBrushIndirect` 创建它们：

[PRE140]

在填充矩形时，我们还需要一个实心画刷，我们使用 Win32 API 函数 `CreateSolidBrush` 创建它：

[PRE141]

`DrawEllipse` 和 `FillEllipse` 方法与 `DrawRectangle` 和 `FillRectangle` 类似。唯一的区别是它们调用 Win32 API 函数 `Ellipse` 而不是 `Rectangle`：

[PRE142]

在绘制文本时，我们首先需要检查字体是否以排版点给出并需要转换为逻辑单位（如果 `pointToMeters` 为真），这在 `LogicalWithScroll` 和 `LogicalWithoutScroll` 坐标系中是这种情况。然而，在 `PreviewCoordinate` 系统中，文本的大小已经以逻辑单位给出，不应进行转换。此外，在我们写入文本之前，我们需要创建并选择一个字体对象，并设置文本和背景颜色。Win32 的 `DrawText` 函数在给定的矩形内居中文本：

[PRE143]

# 摘要

在本章中，我们探讨了小型窗口的核心：`MainWindow` 函数以及 `Application`、`Window` 和 `Graphics` 类。在[第11章](ch11.html "第11章。文档")《文档》中，我们探讨了小型窗口的文档类：`Document`、`Menu`、`Accelerator` 和 `StandardDocument`。
