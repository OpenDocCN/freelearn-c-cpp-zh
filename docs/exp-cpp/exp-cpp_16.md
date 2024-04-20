# 使用 Qt 进行图形用户界面

C++并不直接提供**图形用户界面**（**GUI**）编程。首先，我们应该了解 GUI 与特定的**操作系统**（**OS**）密切相关。您可以使用 Windows API 在 Windows 中编写 GUI 应用程序，或者使用 Linux 特定的 API 在 Linux 中编写 GUI 应用程序，依此类推。每个操作系统都有自己特定的窗口和 GUI 组件形式。

我们在第一章中提到了不同平台及其差异。在讨论 GUI 编程时，平台之间的差异更加令人望而生畏。跨平台开发已经成为 GUI 开发人员生活中的一大痛苦。他们不得不专注于特定的操作系统。为其他平台实现相同的应用程序几乎需要同样多的工作。这是一个不合理的巨大时间和资源浪费。诸如*Java*之类的语言提供了在虚拟环境中运行应用程序的智能模型。这使得开发人员可以专注于一种语言和一个项目，因为环境负责在不同的平台上运行应用程序。这种方法的一个主要缺点是强制用户安装虚拟机，以及与特定平台应用程序相比较慢的执行时间。

为了解决这些问题，Qt 框架被创建了。在本章中，我们将了解 Qt 框架如何支持跨平台 GUI 应用程序开发。为此，您需要熟悉 Qt 及其关键特性。这将使您能够使用您喜爱的编程语言——C++来开发 GUI 应用程序。我们将首先了解 Qt 的 GUI 开发方法，然后我们将涵盖其概念和特性，如信号和槽，以及模型/视图编程。

在本章中，我们将涵盖以下主题：

+   跨平台 GUI 编程的基础

+   Qt 核心组件

+   使用 Qt 小部件

+   使用 Qt Network 设计网络应用程序

# 技术要求

您需要安装最新的 Qt 框架才能运行本章的示例。我们建议使用 Qt Creator 作为项目的 IDE。要下载 Qt 及相应的工具，请访问[qt.io](https://www.qt.io/)网站，并选择框架的开源版本。本章的代码可以在以下网址找到：[`github.com/PacktPublishing/Expert-CPP`](https://github.com/PacktPublishing/Expert-CPP)。

# 了解跨平台 GUI 编程

每个操作系统都有自己的 API。它与 GUI 特别相关。当公司计划设计、实现和发布桌面应用程序时，他们应该决定专注于哪个平台。一个团队的开发人员在一个平台上工作，几乎需要花同样多的时间为另一个平台编写相同的应用程序。这最大的原因是操作系统提供的不同方法和 API。API 的复杂性也可能在按时实现应用程序方面起到重要作用。例如，官方文档中的以下片段显示了如何使用 C++在 Windows 中创建按钮：

```cpp
HWND hwndButton = CreateWindow(
  L"BUTTON", // Predefined class; Unicode assumed      
  L"OK", // Button text      
  WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON, // Styles      
  10, // x position      
  10, // y position      
  100, // Button width     
  100, // Button height     
  m_hwnd, // Parent window     
  NULL, // No menu.     
  (HINSTANCE)GetWindowLong(m_hwnd, GWL_HINSTANCE),     
  NULL); // Pointer not needed.
```

解决 Windows GUI 编程需要你使用`HWND`、`HINSTACNCE`和许多其他奇怪命名和令人困惑的组件。

.NET Framework 对 Windows GUI 编程进行了重大改进。如果您想支持除 Windows 之外的操作系统，使用.NET Framework 之前要三思。

然而，为了支持多个操作系统，您仍然需要深入了解 API 来实现相同的应用程序，以满足所有操作系统的用户。以下代码显示了在 Linux 中使用*Gtk+* GUI 工具包创建按钮的示例：

```cpp
GtkWidget* button = gtk_button_new_with_label("Linux button");
```

与 Windows API 相比，它似乎更容易理解。但是，您应该深入了解`GtkWidgets`和其他带有*Gtk*前缀的组件，以了解更多关于它们的信息。

正如我们已经提到的，诸如 Java 和.NET Core 之类的跨平台语言使用虚拟机在不同平台上运行代码。Qt 框架支持使用基于平台的编译方法进行跨平台 GUI 编程。让我们就 C++语言讨论这两种方法。

# 使用 C++作为 Java

诸如 Java 或 C#之类的语言有不同的编译模型。本书的第一章介绍了 C++的编译模型。首先，我们认为 C++是一种完全可编译的语言，而 Java 保持了混合模型。它将源代码编译成称为**字节码**的中间表示，然后虚拟机通过将其翻译成特定平台的机器代码来运行它。

以下图表描述了 C++和 Java 编译模型之间的差异：

![](img/c0552bd5-d588-48b5-b7ec-9491231fbe30.png)

**Java 虚拟机**（JVM）充当中间层。它对每个平台有一个独特的实现。用户需要在运行 Java 程序之前安装特定实现的虚拟机。安装过程只发生一次。另一方面，C++程序被翻译成机器代码，而不需要像 JVM 这样的中间层环境。这是 C++应用程序通常更快的原因之一。当我们在某个平台上编译 C++程序时，编译器会输出一个由特定于该平台的格式的指令组成的可执行文件。当我们将应用程序移动到另一个平台时，它就无法运行。

其他平台无法识别它的格式，也无法识别它的指令（尽管它们可能在某种程度上相似）。Java 方法通过提供一些字节码来工作，这些字节码对于所有虚拟机的实现都是相同的。但是虚拟机确切地知道他们应该为作为输入提供的字节码生成哪些指令。如果安装了虚拟机，相同的字节码可以在许多计算机上运行。以下图表演示了 Java 应用程序编译模型：

![](img/4395de50-bc9b-4255-a882-8203041b2429.png)

如您所见，源代码被编译成可以在每个操作系统上运行的字节码。然而，每个操作系统必须提供其自己的虚拟机实现。这意味着如果我们为该操作系统安装了专门为该操作系统实现的 JVM，我们就可以在任何操作系统上运行 Java 应用程序。

尽管 C++是一种跨平台语言，也就是说我们不需要修改代码就可以在其他平台上编译它，但是这种语言并不直接支持 GUI 编程。为了编写 GUI 应用程序，正如我们之前提到的，我们需要直接从代码中访问操作系统 API。这使得 C++ GUI 应用程序依赖于平台，因为你需要修改代码基础才能在其他平台上编译它。以下图表显示了 GUI 是如何破坏语言的跨平台性的：

![](img/2cba1673-03ed-49cd-89b0-123652d0f46b.png)

尽管应用程序的逻辑、名称和任务可能相同，但现在它有三种不同的实现，有三种不同的可执行文件。要将应用程序交付给最终用户，我们需要发现他们的操作系统并交付正确的可执行文件。您可能在从网上下载应用程序时遇到了类似的情况。它们基于操作系统提供下载应用程序。这就是 Qt 发挥作用的地方。让我们看看它是如何做到的。

# Qt 的跨平台模型

Qt 是一个用于创建 GUI 应用程序的流行的小部件工具包。它还允许我们创建在各种系统上运行的跨平台应用程序。Qt 包括以下模块：

+   **Qt 核心**：核心类

+   **Qt GUI**：GUI 组件的基本类

+   **Qt 小部件**：用于扩展 Qt GUI 的 C++小部件的类

+   **Qt 多媒体**：音频、视频、广播和摄像功能的类

+   **Qt 多媒体小部件**：实现多媒体功能的类

+   **Qt 网络**：网络编程的类（我们将在本章中使用它们）

+   **Qt 建模语言**（**QML**）：用于构建具有自定义用户界面的声明性框架

+   **Qt SQL**：使用 SQL 进行数据库集成的类

+   **Qt Quick 模块系列**：一个与 QML 相关的模块列表，本书不会讨论

+   **Qt 测试**：用于单元测试 Qt 应用程序的类

我们在程序中使用的每个模块都通过具有`.pro`扩展名的项目文件连接到编译器。该文件描述了`qmake`构建应用程序所需的一切。*qmake*是一个旨在简化构建过程的工具。我们在项目的`.pro`文件中描述项目组件（源文件、Qt 模块、库等）。例如，一个使用 Qt 小部件和 Qt 网络，由`main.cpp`和`test.cpp`文件组成的项目将在`.pro`文件中具有以下内容：

```cpp
QT += widgets
QT += network
SOURCES += test.cpp
SOURCES += main.cpp
```

我们也可以在`.pro`文件中指定特定于平台的源文件，如下所示：

```cpp
QT += widgets
QT += network
SOURCES += test.cpp
SOURCES += main.cpp
win32 {
 SOURCES += windows_specific.cpp
}
unix {
 SOURCES += linux_world.cpp
}
```

当我们在 Windows 环境中构建应用程序时，`windows_specific.cpp`文件将参与构建过程。相反，当在 Unix 环境中构建时，将包括`linux_world.cpp`文件，而`windows_specific.cpp`文件将被忽略。通过这样，我们已经了解了 Qt 应用程序的编译模型。

Qt 强大的跨平台编程能力的整个重点在于元编译源代码；也就是说，在代码传递给 C++编译器之前，Qt 编译器通过引入或替换特定于平台的组件来清理它。例如，当我们使用按钮组件（`QPushButton`）时，如果在 Windows 环境中编译，它将被替换为特定于 Windows 的按钮组件。这就是为什么`.pro`文件也可以包含项目的特定于平台的修改。以下图表描述了这个编译过程：

![](img/ea855027-5cfa-4dbb-861e-3c3c908077f2.png)

元编译器通常被称为**元对象编译器**（**MOC**）。这种方法的美妙之处在于产生的输出代表了我们可以直接运行的相同机器代码，而无需虚拟机。我们可以立即发布可执行文件。这种方法的缺点是，我们再次为不同的平台有不同的可执行文件。然而，我们只编写一个应用程序 - 无需使用不同的语言，深入研究特定于操作系统的 API，或学习特定于操作系统的 GUI 组件类名称。正如 Qt 所说，*一次编写，到处编译*。现在，让我们继续构建一个简单的 GUI 应用程序。

# 编写一个简单的应用程序

我们不会在本书中讨论我们之前提到的所有模块，因为这需要一本全新的书。您可以在本章末尾列出的书籍中的*进一步阅读*部分中查阅更多信息。`main`函数如下所示：

```cpp
#include <QtWidgets>

int main(int argc, char** argv)
{
  QApplication app(argc, argv);

  QPushButton btn("Click me!");
  btn.show();

  return app.exec();
}
```

让我们来看看我们在代码中使用的各种组件。第一个是`QtWidgets`头文件。它包含了我们可以用来为应用程序构建细粒度 GUI 的小部件组件。接下来是`QPushButton`类，它代表一个可点击按钮的包装器。我们故意在这里引入它作为一个包装器，这样我们可以在本章后面讨论 Qt 程序的编译过程时解释它。这是运行上述代码的结果：

![](img/10a68757-4e71-4c70-a722-b05add3fda63.png)

正如您所看到的，我们只声明了`QPushButton`类，但它出现为一个具有标准 OS 的关闭和最小化按钮的窗口（在本例中是 macOS）。这是因为`QPushButton`间接继承自`QWidget`，它是一个带有框架的小部件；也就是说，一个窗口。按钮几乎占据了窗口的所有空间。我们可以调整窗口的大小，看看按钮如何随之调整大小。我们将在本章后面更详细地讨论小部件。 

当我们运行`app.exec()`时，GUI 构建完成。注意`app`对象的类型。它是一个`QApplication`对象。这是 Qt 应用程序的起点。当我们调用`exec()`函数时，我们启动了 Qt 的事件循环。我们对程序执行的感知应该有所改变，以理解 GUI 应用程序的生命周期。重新定义程序构建和执行的感知在第七章之后对你来说应该不足为奇，*函数式编程*。这次并不那么困难。这里需要知道的主要事情是，GUI 应用程序在主程序之外还有一个额外的实体在运行。这个实体被称为**事件循环**。

回想一下我们在第十一章中讨论过的事件循环，*使用设计模式设计策略游戏*。游戏代表了用户密集交互的可视组件的程序。同样适用于具有按钮、标签和其他图形组件的常规 GUI 应用程序。

用户与应用程序交互，每个用户操作都被解释为一个事件。然后将每个事件推送到队列中。事件循环逐个处理这些事件。处理事件意味着调用与事件相关联的特殊处理程序函数。例如，每当单击按钮时，将调用`keyPressedEvent()`函数。它是一个虚函数，因此在设计自定义按钮时可以重写它，如下面的代码所示：

```cpp
class MyAwesomeButton : public QPushButton
{
  Q_OBJECT
public:
 void keyPressedEvent(QKeyEvent* e) override
 {
 // anything that we need to do when the button is pressed
 }
};
```

事件的唯一参数是指向`QKeyEvent`的指针，它是`QEvent`的子类型。`QEvent`是 Qt 中所有事件类的基类。注意在类的开头块之后放置的奇怪的`Q_OBJECT`。这是一个 Qt 特定的宏，如果你想让它们被 Qt 的 MOC 发现，应该将它放在自定义类的第一行。

在下一节中，我们将介绍特定于 Qt 对象的信号和槽的机制。为了使我们的自定义对象支持该机制，我们在类定义中放置`Q_OBJECT`宏。

现在，让我们构建比简单按钮更大的东西。以下示例创建了一个标题为“精通 C ++”的窗口：

```cpp
#include <QtWidgets>

int main(int argc, char** argv)
{
  QApplication app(argc, argv);
 QWidget window;
 window.resize(120, 100);
 window.setWindowTitle("Mastering C++");
 window.show();

  return app.exec();
}
```

通过执行上述程序，我们得到以下结果：

![](img/9d9f3077-f55d-4e9d-969c-943ce57ae816.png)

标题被截断了；我们只能看到“Mast...”部分的“Mastering C ++”。现在，如果我们手动调整大小，或者更改源代码，使第二个参数的`resize()`函数具有更大的值，我们会得到以下结果：

![](img/e176177a-de4c-4ab2-8043-e63400a3a368.png)

`window`对象是`QWidget`类型。`QWidget`是所有用户界面对象的中心类。每当您想要创建自定义小部件或扩展现有小部件时，您都会直接或间接地继承自`QWidget`。它有很多函数适用于每种用例。您可以使用`move()`函数在屏幕上移动它，可以通过调用`showFullScreen()`使窗口全屏，等等。在上面的代码中，我们调用了`resize()`函数，它接受宽度和高度来调整小部件的大小。还要注意`setWindowTitle()`函数，它正如其名-将传递的字符串参数设置为窗口的标题。在代码中使用字符串值时，最好使用`QApplication::translate()`函数。这样做可以使程序本地化变得更容易，因为当语言设置更改时，Qt 会自动用正确的翻译替换文本。`QObject::tr()`提供了几乎相同的功能。

`QObject`是所有 Qt 类型的基类。在诸如 Java 或 C＃之类的语言中，每个对象都直接或间接地继承自一个通用类型，通常命名为`Object`。C ++没有包含一个公共基类。另一方面，Qt 引入了`QObject`，它具有所有对象应支持的基本功能。

现在我们已经了解了 Qt 应用程序开发的基础知识，让我们深入了解框架并发现其关键特性。

# 发现 Qt

Qt 随着时间的推移不断发展，在撰写本书时，其版本为 5.14。它的第一个公共预发布版本是在 1995 年宣布的。已经过去了二十多年，现在 Qt 在几乎所有平台上都有许多强大的功能，包括 Android 和 iOS 等移动系统。除了少数例外，我们可以自信地为所有平台使用 C++和 Qt 编写功能齐全的 GUI 应用程序。这是一个重大的变革，因为公司可以雇佣专门从事一种技术的小团队，而不是为每个特定平台都有几个团队。

如果你是 Qt 的新手，强烈建议你尽可能熟悉它（在本章的末尾有书籍参考）。除了 GUI 框架提供的常规组件外，Qt 还引入了一些在框架中新的或精心实现的概念。其中一个概念是使用信号和槽进行对象之间的通信。

# 掌握信号和槽

Qt 引入了信号和槽的概念作为对象之间灵活的通信机制。信号和槽的概念及其实现机制是将 Qt 与其他 GUI 框架区分开的特性之一。在之前的章节中，我们讨论了观察者模式。这个模式的主要思想是有一个对象通知其他对象（订阅者）一个事件。信号和槽的机制类似于观察者模式的实现。这是一种对象通知另一个对象其变化的方式。Qt 提供了一个通用接口，可以用来通过将一个对象的信号与另一个对象的槽绑定来连接对象。信号和槽都是对象的常规成员函数。信号是在对象的指定动作上调用的函数。槽是作为订阅者的另一个函数。它由信号函数调用。

正如我们之前提到的，Qt 向我们介绍了所有对象的基本类型`QObject`。支持信号和槽的基本功能在`QObject`中实现。你在代码中声明的任何对象，`QWidget`、`QPushButton`等都继承自`QObject`，因此它们都支持信号和槽。QObject 为我们提供了两个用于管理对象通信的函数。这些对象是`connect()`和`disconnect()`：

```cpp
bool connect(const QObject* sender, const char* signal, 
  const QObject* receiver, const char* method, 
  Qt::ConnectionType type = Qt::AutoConnect);

bool disconnect(const QObject* sender, const char* signal, 
  const QObject* receiver, const char* method);
```

正如你所看到的，`connect()`函数将`receiver`和`sender`对象作为参数。它还接受信号和槽的名称。`signal`与发送者相关联，而`slot`是接收者提供的。以下图表显示了这一点：

![](img/2583b4a4-6b3d-4aa9-b879-885fcb19b63b.png)

当编写 Qt 应用程序时，操作信号和槽将变得自然，迟早你会认为每个其他框架都支持信号和槽，因为它们很方便。还要注意，在`connect()`和`disconnect()`函数中，信号和槽被处理为字符串。在连接对象时指定信号和槽，我们使用另外两个宏，分别是`SIGNAL()`和`SLOT()`。从现在开始不会再介绍更多的宏 - 我们保证。

这是我们如何连接两个对象的方式。假设我们想要改变标签（`QLabel`的一个实例）的文本，使其在按钮被点击时接收一个信号。为了实现这一点，我们将`QPushButton`的`clicked()`信号连接到`QLabel`的槽，如下所示：

```cpp
QPushButton btn("Click me!");
QLabel lbl;
lbl.setText("No signal received");
QObject::connect(&btn, SIGNAL(clicked()), &lbl, SLOT(setText(const QString&)));
```

前面的代码可能看起来有点冗长，但你会习惯的。把它看作是信号和槽的便利机制的代价。然而，前面的例子不会给我们所需的结果；也就是说，它不会将标签的文本设置为接收到信号。我们应该以某种方式将该字符串传递给标签的槽。`clicked()`信号不会为我们做到这一点。实现这一点的一种方法是通过扩展`QLabel`，使其实现一个自定义槽，将文本设置为`received a signal`。下面是我们可以这样做的方法：

```cpp
class MyLabel : public QLabel
{
Q_OBJECT
public slots:
  void setCustomText() { 
    this->setText("received a signal");
  }
};
```

要声明一个槽，我们像在前面的代码中所做的那样指定部分。信号的声明方式几乎相同：通过指定一个带有`signals：`的部分。唯一的区别是信号不能是私有或受保护的。我们只是按原样声明它们：

```cpp
class Example
{
Q_OBJECT:
public:
  // member functions
public slots:
  // public slots
private slots:
  // private slots
signals: // no public, private, or protected
  // signals without any definition, only the prototype
};
```

现在，我们只需要更新前面的代码，以更改标签的信号（以及标签对象的类型）：

```cpp
QPushButton btn("Click me!");
MyLabel lbl;
lbl.setText("No signal received");
QOBject::connect(&btn, SIGNAL(clicked()), &lbl, SLOT(setCustomText()));
```

我们说槽将在信号被发射时被调用。您还可以在对象内部声明和发射信号。与 GUI 事件循环无关的信号和槽的一个重要细节。

当信号被发射时，连接的槽立即执行。但是，我们可以通过将`Qt::ConnectionType`之一作为`connect()`函数的第五个参数来指定连接的类型。它包括以下值：

+   `AutoConnection`

+   `DirectConnection`

+   `QueuedConnection`

+   `BlockingQueuedConnection`

+   `UniqueConnection`

在`DirectConnection`中，当信号被发射时，槽立即被调用。另一方面，当使用`QueuedConnection`时，当执行返回到接收对象线程的事件循环时，槽被调用。`BlockingQueuedConnection`类似于`QueuedConnection`，只是信号线程被阻塞，直到槽返回一个值。`AutoConnection`可以是`DirectConnection`或`QueuedConnection`。当信号被发射时，类型被确定。如果接收者和发射者在同一线程中，使用`DirectConnection`；否则，连接使用`QueuedConnection`。最后，`UniqueConnection`与前面描述的任何连接类型一起使用。它与其中一个使用按位或组合。它的唯一目的是使`connect()`函数在信号和线程之间的连接已经建立时失败。

信号和槽构成了 Qt 在 GUI 编程中出色的机制。我们介绍的下一个机制在框架中很受欢迎，与我们在应用程序中操作数据的方式有关。

# 理解模型/视图编程

模型/视图编程根植于**模型视图控制器**（MVC）设计模式。该模式的主要思想是将问题分解为三个松散耦合的组件，如下所示：

+   模型负责存储和操作数据

+   视图负责渲染和可视化数据

+   控制器负责额外的业务逻辑，并从模型向视图提供数据

通过其演变，我们现在有了一种简化和更便利的编程方法，称为**模型/视图编程**。它类似于 MVC 模式，只是通过使视图和模型更关注手头的功能来省略了控制器。我们可以说视图和控制器在模型/视图架构中合并在一起。看一下以下架构图：

![](img/f039425a-e465-49bd-81db-34ca31d2918f.png)

模型代表数据，与其来源通信，并为架构中的其他组件提供方便的接口。模型的实现及其与其他组件的通信基于手头数据的类型。

视图通过获取所谓的模型索引来引用数据项。视图可以从模型检索和提供数据。关键是，数据项可以使用视图进行编辑，委托起到了与模型通信以保持数据同步的作用。

介绍的每个组件——模型、视图和委托——都由提供共同接口的抽象类定义。在某些情况下，类还提供了功能的默认实现。要编写专门的组件，我们从抽象类继承。当然，模型、视图和委托使用我们在上一节中介绍的信号和槽进行通信。

当模型遇到数据变化时，它会通知视图。另一方面，渲染数据项的用户交互由视图发出的信号通知。最后，委托发出的信号通知模型和视图有关数据编辑状态的信息。

模型基于`QAbstractItemModel`类，该类定义了视图和委托使用的接口。Qt 提供了一组现有的模型类，我们可以在不进行修改的情况下使用；但是，如果需要创建新模型，应该从`QAbstractItemModel`继承您的类。例如，`QStringListModel`、`QStandardItemModel`和`QFileSystemModel`类已经准备好处理数据项。`QStringListModel`用于存储字符串项列表（表示为`QString`对象）。此外，还有方便的模型类用于处理 SQL 数据库。`QSqlQueryModel`、`QSqlTableModel`和`QSqlRelationalTableModel`允许我们在模型/视图约定的上下文中访问关系数据库。

视图和委托也有相应的抽象类，即`QAbstractItemView`和`QAbstractItemDelegate`。Qt 提供了现有的视图，可以立即使用，例如`QListView`、`QTableView`和`QTreeView`。这些是大多数应用程序处理的基本视图类型。`QListView`显示项目列表，`QTableView`以表格形式显示数据，`QTreeView`以分层列表形式显示数据。如果要使用这些视图类，Qt 建议从`QAbstractListModel`或`QAbstractTableModel`继承自定义模型，而不是对`QAbstractItemModel`进行子类化。

`QListView`、`QTreeView`和`QTableView`被认为是核心和低级别的类。还有更方便的类，为新手 Qt 程序员提供更好的可用性——`QListWidget`、`QTreeWidget`和`QTableWidget`。我们将在本章的下一节中看到使用小部件的示例。在那之前，让我们看一个`QListWidget`的简单示例：

```cpp
#include <QListWidget>

int main(int argc, char** argv)
{
  QApplication app(argc, argv);
  QListWidget* listWgt{new QListWidget};
  return app.exec();
}
```

向列表窗口小部件添加项目的一种方法是通过创建它们，我们可以通过将列表窗口小部件设置为其所有者来实现。在下面的代码中，我们声明了三个`QListWidgetItem`对象，每个对象都包含一个名称，并与我们之前声明的列表窗口小部件相关联：

```cpp
new QListWidgetItem("Amat", listWgt);
new QListWidgetItem("Samvel", listWgt);
new QListWidgetItem("Leia", listWgt);
```

或者，我们可以声明一个项目，然后将其插入到列表窗口小部件中：

```cpp
QListWidgetItem* newName{new QListWidgetItem};
newName->setText("Sveta");
listWgt->insertItem(0, newName);
```

`insertItem()`成员函数的第一个参数是要将项目插入的`row`的数量。我们将`Sveta`项目放在列表的第一个位置。

现在我们已经涉及了行的概念，我们应该回到模型和它们的索引。模型将数据封装为数据项的集合。模型中的每个项都有一个由`QModelIndex`类指定的唯一索引。这意味着模型中的每个项都可以通过关联的模型索引访问。要获取模型索引，我们需要使用`index()`函数。以下图表描述了一个以表格结构组织其数据的模型：

![](img/394f992b-dd5a-4d0b-a1cf-ad364cf58851.png)

视图使用这种约定来访问模型中的数据项。但是，请注意，视图在呈现数据给用户方面并没有限制。视图的实现方式取决于如何以对用户方便的方式呈现和展示数据。以下图表显示了数据在模型中的组织方式：

![](img/8cadf899-4035-46fc-9457-dc68738f5099.png)

这是我们如何使用模型索引访问第 1 行第 2 列的特定数据项：

```cpp
QModelIndex itemAtRow1Col2 = model->index(1, 2);
```

最后，让我们声明一个视图并为其设置一个模型，以查看模型/视图编程的实际效果：

```cpp
QStringList lst;
lst << "item 1" << "item 2" << "item 3";

QStringListModel model;
model.setStringList(lst);

QListView lview;
lview.setModel(model);
```

一旦我们熟悉了 Qt 提供的各种小部件，我们将在下一节继续这个示例。

# 使用 Qt 小部件

小部件是可视化 GUI 组件。如果一个小部件没有父级，它将被视为一个窗口，也就是**顶级小部件**。在本章的前面，我们创建了 Qt 中最简单的窗口，如下所示：

```cpp
#include <QtWidgets>

int main(int argc, char** argv)
{
  QApplication app(argc, argv);
 QWidget window;
 window.resize(120, 100);
 window.setWindowTitle("Mastering C++");
 window.show();

  return app.exec();
}
```

正如您所看到的，`window`对象没有父级。问题是，`QWidget`的构造函数接受另一个`QWidget`作为当前对象的父级。因此，当我们声明一个按钮并希望它成为`window`对象的子级时，我们可以这样做：

```cpp
#include <QtWidgets>

int main(int argc, char** argv)
{
  QApplication app(argc, argv);
QWidget window;
  window.resize(120, 100);
  window.setWindowTitle("Mastering C++");
  window.show();

 QPushButton* btn = new QPushButton("Click me!", &window);

  return app.exec();
}
```

观察`QPushButton`构造函数的第二个参数。我们将`window`对象的引用作为其父级传递。当父对象被销毁时，其子对象将自动被销毁。Qt 支持许多其他小部件；让我们看看其中一些。

# 常见的 Qt 小部件

在上一节中，我们介绍了`QPushButton`类，并指出它间接继承了`QWidget`类。要创建一个窗口，我们使用了`QWidget`类。事实证明，QWidget 代表了向屏幕渲染的能力，它是所有小部件都继承的基本类。它具有许多属性和函数，例如`enabled`，一个布尔属性，如果小部件启用则为 true。要访问它，我们使用`isEnabled()`和`setEnabled()`函数。要控制小部件的大小，我们使用它的`height`和`width`，分别表示小部件的高度和宽度。要获取它们的值，我们分别调用`height()`和`width()`。要设置新的高度和宽度，我们应该使用`resize()`函数，它接受两个参数 - 宽度和高度。您还可以使用`setMinimumWidth()`、`setMinimumHeight()`、`setMaximumWidth()`和`setMaximumHeight()`函数来控制小部件的最小和最大大小。当您在布局中设置小部件时，这可能会很有用（请参阅下一节）。除了属性和函数，我们主要对 QWidget 的公共槽感兴趣，它们如下：

+   `close()`: 关闭小部件。

+   `hide()`: 等同于`setVisible(false)`，此函数隐藏小部件。

+   `lower()`和`raise()`: 将小部件移动到父小部件的堆栈中（到底部或顶部）。每个小部件都可以有一个父小部件。没有父小部件的小部件将成为独立窗口。我们可以使用`setWindowTitle()`和`setWindowIcon()`函数为此窗口设置标题和图标。

+   `style`: 该属性保存小部件的样式。要修改它，我们使用`setStyleSheet()`函数，通过传递描述小部件样式的字符串。另一种方法是调用`setStyle()`函数，并传递封装了与样式相关属性的`QStyle`类型的对象。

Qt 小部件几乎具备所有必要的属性，可以直接使用。很少遇到需要构建自定义小部件的情况。然而，一些团队为他们的软件创建了整套自定义小部件。如果您计划为程序创建自定义外观和感觉，那是可以的。例如，您可以整合扁平风格的小部件，这意味着您需要修改框架提供的默认小部件的样式。自定义小部件应该继承自`QWidget`（或其任何后代），如下所示：

```cpp
class MyWidget : public QWidget
{}; 
```

如果您希望小部件公开信号和插槽，您需要在类声明的开头使用`Q_OBJECT`宏。更新后的`MyWidget`类的定义如下：

```cpp
class MyWidget : public QWidget
{
Q_OBJECT
public:
  // public section

signals: 
  // list of signals

public slots:
  // list of public slots
};
```

正如您可能已经猜到的那样，信号没有访问修饰符，而插槽可以分为公共、私有和受保护部分。正如我们之前提到的，Qt 提供了足够的小部件。为了了解这些小部件，Qt 提供了一组将小部件组合在一起的示例。如果您已安装了 Qt Creator（用于开发 Qt 应用程序的 IDE），您应该能够通过单击一次来查看示例。在 Qt Creator 中的样子如下：

![](img/477bcb65-d63d-444f-b137-11aa76371cd2.png)

配置和运行地址簿示例将给我们提供以下界面：

![](img/d0e7b66e-749a-4d36-9520-e04b702e354a.png)

单击“添加”按钮将打开一个对话框，以便我们可以向地址簿添加新条目，如下所示：

![](img/5bd497c5-53ec-4f80-8c12-1b1143f90e84.png)

添加了几个条目后，主窗口将以表格形式显示条目，如下所示：

![](img/91bc9193-20c4-4f05-8660-b43300552cd6.png)

前面的屏幕截图显示了在一个应用程序中组合在一起的各种小部件。以下是我们在 GUI 应用程序开发中经常使用的一些常见小部件：

+   `QCheckBox`：表示带有文本标签的复选框。

+   `QDateEdit`：表示可以用来输入日期的小部件。如果还要输入时间，也可以使用`QDateTimeEdit`。

+   `QLabel`：文本显示。也用于显示图像。

+   `QLineEdit`：单行编辑框。

+   `QProgressBar`：渲染垂直或水平进度条。

+   `QTabWidget`：标签式小部件的堆栈。这是许多组织小部件中的一个。其他组织者包括`QButtonGroup`、`QGroupBox`和`QStackedWidget`。

前面的列表并非最终版本，但它给出了 Qt 的基本功能的基本概念。我们在这里使用的地址簿示例使用了许多这些小部件。`QTabWidget`表示一个组织小部件。它将几个小部件组合在一起。另一种组织小部件的方法是使用布局。在下一节中，我们将介绍如何将小部件组织在一起。

# 使用布局组合小部件

Qt 为我们提供了一个灵活和简单的平台，我们可以在其中使用布局机制来安排小部件。这有助于确保小部件内部的空间被高效地使用，并提供友好的用户体验。

让我们来看看布局管理类的基本用法。使用布局管理类的优势在于，当容器小部件更改大小时，它们会自动调整小部件的大小和位置。Qt 的布局类的另一个优势是，它们允许我们通过编写代码来安排小部件，而不是使用 UI 组合器。虽然 Qt Creator 提供了一种通过手工组合小部件的好方法（在屏幕上拖放小部件），但大多数程序员在实际编写安排小部件外观和感觉的代码时会感到更舒适。假设您也喜欢后一种方法，我们将介绍以下布局类：

+   `QHBoxLayout`

+   `QVBoxLayout`

+   `QGridLayout`

+   `QFormLayout`

所有这些类都继承自`QLayout`，这是几何管理的基类。`QLayout`是一个抽象基类，继承自`QObject`。它不继承自`QWidget`，因为它与渲染无关；相反，它负责组织应该在屏幕上呈现的小部件。您可能不需要实现自己的布局管理器，但如果需要，您应该从`QLayout`继承您的类，并为以下函数提供实现：

+   `addItem()`

+   `sizeHint()`

+   `setGeometry()`

+   `itemAt()`

+   `takeAt()`

+   `minimumSize()`

这里列出的类已经足够组成几乎任何复杂的小部件。更重要的是，我们可以将一个布局放入另一个布局中，从而更灵活地组织小部件。使用`QHBoxLayout`，我们可以从左到右水平地组织小部件，如下面的屏幕截图所示：

![](img/41c40a4b-c3e7-48f1-b380-1e50488376d7.png)

要实现上述组织，我们需要使用以下代码：

```cpp
QWidget *window = new QWidget;
QPushButton *btn1 = new QPushButton("Leia");
QPushButton *btn2 = new QPushButton("Patrick");
QPushButton *btn3 = new QPushButton("Samo");
QPushButton *btn4 = new QPushButton("Amat");

QHBoxLayout *layout = new QHBoxLayout;
layout->addWidget(btn1);
layout->addWidget(btn2);
layout->addWidget(btn3);
layout->addWidget(btn4);

window->setLayout(layout);
window->show();
```

看一下我们在小部件上调用`setLayout()`函数的那一行。每个小部件都可以分配一个布局。布局本身没有太多作用，除非有一个容器，所以我们需要将其设置为一个作为组织小部件（在我们的情况下是按钮）容器的小部件。`QHBoxLayout`继承自`QBoxLayout`，它有另一个我们之前列出的后代——`QVBoxLayout`。它类似于`QHBoxLayout`，但是垂直地组织小部件，如下面的屏幕截图所示：

![](img/a74e6782-1f0b-4911-a676-6caff1a6e7cf.png)

在上述代码中，我们唯一需要做的是将`QHBoxLayout`替换为`QVBoxLayout`，如下所示：

```cpp
QVBoxLayout* layout = new QVBoxLayout;
```

`GridLayout`允许我们将小部件组织成网格，如下面的屏幕截图所示：

![](img/76ec7f99-4a11-45e6-938a-9693af4e5f5f.png)

以下是相应的代码块：

```cpp
QGridLayout *layout = new QGridLayout;
layout->addWidget(btn1, 0, 0);
layout->addWidget(btn2, 0, 1);
layout->addWidget(btn3, 1, 0);
layout->addWidget(btn4, 1, 1);
```

最后，类似于`QGridLayout`，`QFormLayout`在设计输入表单时更有帮助，因为它以两列描述的方式布置小部件。

正如我们之前提到的，我们可以将一个布局组合到另一个布局中。为此，我们需要使用`addItem()`函数，如下所示：

```cpp
QVBoxLayout *vertical = new QVBoxLayout;
vertical->addWidget(btn1);
vertical->addWidget(btn2);

QHBoxLayout *horizontal = new QHBoxLayout;
horizontal->addWidget(btn3);
horizontal->addWidget(btn4);

vertical->addItem(horizontal);

```

布局管理器足够灵活，可以构建复杂的用户界面。

# 总结

如果您是 Qt 的新手，本章将作为对框架的一般介绍。我们涉及了 GUI 应用程序开发的基础知识，并比较了 Java 方法和 Qt 方法。使用 Qt 的最大优点之一是它支持跨平台开发。虽然 Java 也可以做到，但 Qt 通过生成与平台原生的可执行文件而更进一步。这使得使用 Qt 编写的应用程序比集成虚拟机的替代方案快得多。

我们还讨论了 Qt 的信号和槽作为对象间通信的灵活机制。通过使用这个机制，您可以在 GUI 应用程序中设计复杂的通信机制。虽然本章中我们只看了一些简单的例子，但您可以自由地尝试各种使用信号和槽的方式。我们还熟悉了常见的 Qt 小部件和布局管理机制。现在您已经有了基本的理解，可以设计甚至最复杂的 GUI 布局。这意味着您可以通过应用本章介绍的技术和小部件来实现复杂的 Qt 应用程序。在下一章中，我们将讨论一个当今流行的话题——人工智能和机器学习。

# 问题

1.  为什么 Qt 不需要虚拟机？

1.  `QApplication::exec()`函数的作用是什么？

1.  如何更改顶层小部件的标题？

1.  给定`m`模型，如何访问第 2 行第 3 列的项目？

1.  给定`wgt`小部件，如何将其宽度更改为 400，高度更改为 450？

1.  从`QLayout`继承以创建自己的布局管理器类时，应该实现哪些函数？

1.  如何将信号连接到槽？

# 进一步阅读

+   *Qt5 C++ GUI Programming Cookbook* by Lee Zhi Eng: [`www.packtpub.com/application-development/qt5-c-gui-programming-cookbook-second-edition`](https://www.packtpub.com/application-development/qt5-c-gui-programming-cookbook-second-edition)

+   *Mastering Qt5* by Guillaume Lazar, Robin Penea: [`www.packtpub.com/web-development/mastering-qt-5-second-edition`](https://www.packtpub.com/web-development/mastering-qt-5-second-edition)
