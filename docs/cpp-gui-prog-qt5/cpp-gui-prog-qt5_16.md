# 测试和调试

在阅读与编程相关的教程或文章时，我们经常看到*调试*这个词。但是您知道调试是什么意思吗？在编程术语中，*bug*表示计算机程序中的错误或缺陷，导致软件无法正常运行，通常会导致不正确的输出甚至崩溃。

在本章中，我们将涵盖以下主题，并学习如何调试我们的 Qt 项目：

+   调试技术

+   Qt 支持的调试器

+   单元测试

让我们开始吧。

# 调试技术

在开发过程中经常会出现技术问题。为了解决这些问题，我们需要在将应用程序发布给用户之前找出所有这些问题并解决它们，以免影响公司/团队的声誉。用于查找技术问题的方法称为调试。在本节中，我们将介绍专业人士常用的常见调试技术，以确保他们的程序可靠且质量高。

# 识别问题

在调试程序时，无论编程语言或平台如何，最重要的是知道代码的哪一部分导致了问题。您可以通过几种方式来识别问题代码：

+   询问用户出现错误的位置；例如，按下了哪个按钮，导致崩溃的步骤是什么，等等。

+   注释掉代码的一部分，然后重新构建和运行程序，以检查问题是否仍然存在。如果问题仍然存在，继续注释更多的代码，直到找到问题所在的代码行。

+   使用内置调试器通过设置数据断点来检查目标函数中的变量更改。您可以轻松地发现您的变量是否已更改为意外值，或者对象指针是否已变为未定义指针。

+   确保您为用户安装程序中包含的所有库与项目中使用的库具有匹配的版本号。

# 使用 QDebug 打印变量

您还可以使用`QDebug`类将变量的值打印到应用程序输出窗口。`QDebug`与标准库中的`std::cout`非常相似，但使用`QDebug`的优势在于，由于它是 Qt 的一部分，它支持 Qt 类，而且能够在不需要任何转换的情况下输出其值。

要启用`QDebug`，我们必须首先包含其头文件：

```cpp
#include <QDebug> 
```

之后，我们可以调用`qDebug()`将变量打印到应用程序输出窗口：

```cpp
int amount = 100; 
qDebug() << "You have obtained" << amount << "apples!"; 
```

结果将如下所示：

![](img/085d30a2-eaa0-43d5-8887-4df07ebf0ed9.png)

通过使用`QDebug`，我们将能够检查我们的函数是否正常运行。在检查完问题后，您可以注释掉包含`qDebug()`的特定代码行。

# 设置断点

设置断点是调试程序的另一种好方法。当您在 Qt Creator 中右键单击脚本的行号时，将会弹出一个包含三个选项的菜单，您可以在下面的截图中看到：

![](img/6ee7547c-f999-4056-b55d-41a1ea786a99.png)

第一个选项称为在行处设置断点...，允许您在脚本的特定行上设置断点。一旦创建了断点，该行号旁边将出现一个红色圆点图标：

![](img/59956e75-08f7-4699-8bd7-2d45b6d49ee0.png)

第二个选项称为在行处设置消息跟踪点...，当程序到达特定代码行时打印消息。一旦创建了断点，该行号旁边将出现一个眼睛图标：

![](img/488f3bb1-e9de-42ac-8600-e4b592256270.png)

第三个选项是切换书签，允许您为自己设置书签。让我们创建一个名为`test()`的函数来尝试断点：

```cpp
void MainWindow::test() 
{ 
   int amount = 100; 
   amount -= 10; 
   qDebug() << "You have obtained" << amount << "apples!"; 
} 
```

之后，我们在`MainWindow`构造函数中调用`test()`函数：

```cpp
MainWindow::MainWindow(QWidget *parent) : 
   QMainWindow(parent), 
   ui(new Ui::MainWindow) 
{ 
   ui->setupUi(this); 
   test(); 
} 
```

然后，按下位于 Qt Creator 窗口左下角的开始调试按钮：

![](img/dc999d58-3ca2-4a24-91b9-0474314d3908.png)

您可能会收到类似于这样的错误消息：

![](img/d8c0529c-a614-491e-ac8f-0089a0b6a7c5.png)

在这种情况下，请确保您的项目工具包已连接到调试器。如果仍然出现此错误，请关闭 Qt Creator，转到您的项目文件夹并删除`.pro.user`文件。然后，用 Qt Creator 打开您的项目。Qt Creator 将重新配置您的项目，并且调试模式现在应该可以工作了。

让我们给我们的代码添加两个断点并运行它。一旦我们的程序启动，我们将看到一个黄色箭头出现在第一个红点的顶部：

![](img/d5b0fc54-92ce-48ef-a6e6-7ce0af651b97.png)

这意味着调试器已经停在了第一个断点处。现在，位于 Qt Creator 右侧的本地和表达式窗口将显示变量及其值和类型：

![](img/401c5d9f-2626-4871-ac40-f7bba1272cab.png)

在上图中，您可以看到值仍然为 100，因为此时减法操作尚未运行。接下来，我们需要做的是单击位于 Qt Creator 底部的堆栈窗口顶部的“步入”按钮：

![](img/fb1565d7-beb1-4642-ae50-737e7795057c.png)

之后，调试器将移动到下一个断点，这里我们可以看到值已经减少到了 90，正如预期的那样：

![](img/fca1e49a-7ed7-4b7b-b17e-56aed8a01921.png)

您可以使用这种方法轻松检查您的应用程序。要删除断点，只需再次单击红点图标。

请注意，您必须在调试模式下运行此操作。这是因为在调试模式下编译时，将额外的调试符号嵌入到您的应用程序或库中，使您的调试器能够访问来自二进制源代码的信息，例如标识符、变量和例程的名称。这也是为什么在调试模式下编译的应用程序或库的文件大小会更大的原因。

# Qt 支持的调试器

Qt 支持不同类型的调试器。根据您的项目运行的平台和编译器，使用的调试器也会有所不同。以下是 Qt 通常支持的调试器列表：

+   **Windows (MinGW):** GDB (GNU 调试器)

+   **Windows (MSVC):** CDB (Windows 调试工具)

+   **macOS**: LLDB (LLVM 调试器), FSF GDB (实验性)

+   **Linux**: GDB, LLDB (实验性)

+   **Unix** (FreeBSD, OpenBSD, 等): GDB

+   **Android**: GDB

+   **iOS**: LLDB

# PC 的调试

对于**GDB (GNU 调试器)**，如果您在 Windows 上使用 MinGW 编译器，则无需进行任何手动设置，因为它通常与您的 Qt 安装一起提供。如果您运行其他操作系统，如 Linux，则可能需要在将其与 Qt Creator 链接之前手动安装它。Qt Creator 会自动检测 GDB 的存在并将其与您的项目链接起来。如果没有，您可以轻松地在 Qt 目录中找到 GDB 可执行文件并自行链接。

另一方面，需要在 Windows 机器上手动安装**CDB (Windows 调试工具)**。请注意，Qt 不支持 Visual Studio 的内置调试器。因此，您需要通过在安装 Windows SDK 时选择一个名为“调试工具”的可选组件来单独安装 CDB 调试器。Qt Creator 通常会识别 CDB 的存在，并将其放在调试器选项页面下的调试器列表中。您可以转到“工具”|“选项”|“构建和运行”|“调试器”查找设置，如下面的屏幕截图所示：

![](img/9b474cf0-8099-4386-860c-4ef15f5e5e40.png)

# 针对 Android 设备的调试

针对 Android 设备的调试比 PC 稍微复杂一些。您必须安装所有必要的 Android 开发包，如 JDK（6 或更高版本）、Android SDK 和 Android NDK。然后，您还需要在 Windows 平台上安装 Android 调试桥（ADB）驱动程序，以启用 USB 调试，因为 Windows 上的默认 USB 驱动程序不允许调试。

# macOS 和 iOS 的调试

至于 macOS 和 iOS，使用的调试器是**LLDB（LLVM 调试器）**，它默认随 Xcode 一起提供。Qt Creator 也会自动识别其存在并将其与您的项目链接起来。

每个调试器都与另一个略有不同，并且在 Qt Creator 上可能表现不同。如果您熟悉这些工具并知道自己在做什么，还可以在其各自的 IDE（Visual Studio、XCode 等）上运行非 GDB 调试器。

如果您需要向项目添加其他调试器，可以转到“工具”|“选项”|“构建和运行”|“工具包”，然后单击“克隆”以复制现有工具包。然后，在“调试器”选项卡下，单击“添加”按钮以添加新的调试器选择：

![](img/471d3646-ff16-4764-a526-92ea5bf8f6b4.png)

在“名称”字段中，输入调试器的描述性名称，以便您可以轻松记住其目的。然后，在“路径”字段中指定调试器二进制文件的路径，以便 Qt Creator 知道在启动调试过程时要运行哪个可执行文件。除此之外，“类型”和“版本”字段由 Qt Creator 用于识别调试器的类型和版本。此外，Qt Creator 还在“ABIs”字段中显示将在嵌入式设备上使用的 ABI 版本。

要了解如何在 Qt 中设置不同调试器的详细信息，请访问以下链接：

[`doc.qt.io/qtcreator/creator-debugger-engines.html.`](http://doc.qt.io/qtcreator/creator-debugger-engines.html)

# 单元测试

单元测试是一个自动化的过程，用于测试应用程序中的单个模块、类或方法。单元测试可以在开发周期的早期发现问题。这包括程序员实现中的错误和单元规范中的缺陷或缺失部分。

# Qt 中的单元测试

Qt 带有一个内置的单元测试模块，我们可以通过在项目文件（.pro）中添加`testlib`关键字来使用它：

```cpp
QT += core gui testlib 
```

之后，将以下标题添加到我们的源代码中：

```cpp
#include <QtTest/QtTest> 
```

然后，我们可以开始测试我们的代码。我们必须将测试函数声明为私有槽。除此之外，该类还必须继承自`QOBject`类。例如，我创建了两个文本函数，分别称为`testString()`和`testGui()`，如下所示：

```cpp
private slots: 
   void testString(); 
   void testGui(); 
```

函数定义看起来像这样：

```cpp
void MainWindow::testString() 
{ 
   QString text = "Testing"; 
   QVERIFY(text.toUpper() == "TESTING"); 
} 

void MainWindow::testGui() 
{ 
   QTest::keyClicks(ui->lineEdit, "testing gui"); 
   QCOMPARE(ui->lineEdit->text(), QString("testing gui")); 
} 
```

我们使用`QTest`类提供的一些宏，如`QVERIFY`、`QCOMPARE`等，来评估作为其参数传递的表达式。如果表达式求值为`true`，则测试函数的执行将继续。否则，将向测试日志附加描述失败的消息，并且测试函数停止执行。

我们还使用了`QTest::keyClicks()`来模拟鼠标在我们的应用程序中的点击。在前面的示例中，我们模拟了在主窗口小部件上的行编辑小部件上的点击。然后，我们输入一行文本到行编辑中，并使用`QCOMPARE`宏来测试文本是否已正确插入到行编辑小部件中。如果出现任何问题，Qt 将在应用程序输出窗口中显示问题。

之后，注释掉我们的`main()`函数，而是使用`QTEST_MAIN()`函数来开始测试我们的`MainWindow`类：

```cpp
/*int main(int argc, char *argv[]) 
{ 
   QApplication a(argc, argv); 
   MainWindow w; 
   w.show(); 

   return a.exec(); 
}*/ 
QTEST_MAIN(MainWindow) 
```

如果我们现在构建和运行我们的项目，我们应该会得到类似以下的结果：

```cpp
********* Start testing of MainWindow ********* 
Config: Using QtTest library 5.9.1, Qt 5.9.1 (i386-little_endian-ilp32 shared (dynamic) debug build; by GCC 5.3.0) 
PASS   : MainWindow::initTestCase() 
PASS   : MainWindow::_q_showIfNotHidden() 
PASS   : MainWindow::testString() 
PASS   : MainWindow::testGui() 
PASS   : MainWindow::cleanupTestCase() 
Totals: 5 passed, 0 failed, 0 skipped, 0 blacklisted, 880ms 
********* Finished testing of MainWindow ********* 
```

还有许多宏可以用来测试应用程序。

有关更多信息，请访问以下链接：

[`doc.qt.io/qt-5/qtest.html#macros`](http://doc.qt.io/qt-5/qtest.html#macros)

# 总结

在这一章中，我们学习了如何使用多种调试技术来识别 Qt 项目中的技术问题。除此之外，我们还了解了 Qt 在不同操作系统上支持的不同调试器。最后，我们还学会了如何通过单元测试自动化一些调试步骤。

就是这样！我们已经到达了本书的结尾。希望你在学习如何使用 Qt 从头开始构建自己的应用程序时找到了这本书的用处。你可以在 GitHub 上找到所有的源代码。祝你一切顺利！
