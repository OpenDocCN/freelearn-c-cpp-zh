# Qt 图形应用程序

在 第四章，*指针库管理系统* 中，我们开发了抽象数据类型和库管理系统。然而，那些应用程序是基于文本的。在本章中，我们将探讨我们将使用 Qt 图形库开发的三个图形应用程序：

+   **时钟**：我们将开发一个带有时针、分针和秒针的模拟时钟，以及标记小时、分钟和秒的线条

+   **绘图程序**：一个可以以不同颜色绘制线条、矩形和椭圆的程序

+   **编辑器**：一个用户可以输入和编辑文本的程序

我们还将了解 Qt 库：

+   窗口和小部件

+   菜单和工具栏

+   在窗口中绘制图形和写入文本

+   如何捕获鼠标和键盘事件

# 创建时钟应用程序

在本章和下一章中，我们将使用 Qt，它是一个面向对象的类库，用于图形应用程序。我们还将使用 Qt Creator，而不是 Visual Studio，它是一个集成开发环境。

# 设置环境

在 Qt Creator 中创建新的图形项目时，我们在文件菜单中选择新建文件或项目，这将使新建文件或项目对话框窗口可见。我们选择 Qt Widgets 应用程序，并点击选择按钮。

然后出现简介和项目位置对话框。我们命名项目为 `Clock`，将其放置在适当的位置，并点击下一步按钮。在 KitSelection 对话框中，我们选择 Qt 库的最新版本，并点击下一步。在类信息对话框中，我们命名应用程序的基类为 `clock`。通常，图形应用程序的窗口继承自 `window` 类。然而，在这种情况下，我们处理的是一个相对简单的应用程序。因此，我们继承 Qt 类 `QWidget`，尽管小部件通常指的是经常嵌入窗口中的较小的图形对象。在 Qt Creator 中，可以添加表单。然而，我们本章不使用该功能。因此，我们取消选中生成表单选项。

Qt 中的所有类名都以字母 `Q` 开头。

最后，在项目管理对话框中，我们简单地接受默认值并点击完成以生成项目，包括文件 `Clock.h` 和 `Clock.cpp`。

# `Clock` 类

项目由文件 `Clock.h`、`Clock.cpp` 和 `Main.cpp` 组成。与前面章节中的类相比，类的定义看起来略有不同。我们使用 *include guards* 来包围类的定义。也就是说，我们必须使用预处理指令 `ifndef`、`define` 和 `endif` 来包围类的定义。预处理程序执行文本替换。

`ifndef` 和 `endif` 指令在 C++ 中工作方式类似于 `if` 语句。如果条件不成立，则省略指令之间的代码。在这种情况下，只有当 `CLOCK_H` 宏之前未定义时，才会包含代码。如果包含代码，则使用 `define` 指令在下一行定义宏。这样，类定义只包含在项目中一次。此外，我们还在 `Clock.h` 头文件中而不是 `Clock.cpp` 定义文件中包含系统头文件 `QWidget` 和 `QTimer`。

**Clock.h:** 

```cpp
    #ifndef CLOCK_H 
    #define CLOCK_H 

    #include <QWidget> 
    #include <QTimer> 
```

由于 `Clock` 是 Qt `QWidget` 类的子类，必须包含 `Q_OBJECT` 宏，它包含来自 Qt 库的某些代码。我们需要它来使用这里显示的 `SIGNAL` 和 `SLOT` 宏：

```cpp
    class Clock : public QWidget { 
      Q_OBJECT 
```

构造函数接受对其父小部件的指针，默认为 `nullptr`：

```cpp
    public: 
      Clock(QWidget* parentWidgetPtr = nullptr); 
```

每当窗口需要重新绘制时，框架都会调用 `paintEvent` 方法。它接受一个指向 `QPaintEvent` 对象的指针作为参数，可以用来确定以何种方式执行重新绘制：

```cpp
    void paintEvent(QPaintEvent *eventPtr); 
```

`QTimer` 是一个 Qt 系统类，用于处理计时器。我们将使用它来移动时钟的指针：

```cpp
      private: 
        QTimer m_timer; 
    }; 

    #endif // CLOCK_H 
```

定义文件主要由 `paintEvent` 方法组成，该方法处理时钟的绘制。

**Clock.cpp:** 

```cpp
    #include <QtWidgets> 
    #include "Clock.h"
```

在构造函数中，我们使用 `parentWidgetPtr` 参数（可能为 `nullptr`）调用基类 `QWidget`：

```cpp
    Clock::Clock(QWidget* parentWidgetPtr /* = nullptr */) 
     :QWidget(parentWidgetPtr) { 
```

我们将窗口标题设置为 `Clock`。在 Qt 中，我们始终使用 `tr` 函数用于文本字面量，它反过来调用 Qt `QCoreApplication` 类中的 `translate` 方法，确保文本被转换为适合显示的形式。我们还调整窗口大小为 1000 x 500 像素，这对于大多数屏幕来说都是合适的：

```cpp
    setWindowTitle(tr("Clock")); 
    resize(1000, 500); 
```

我们需要一种方法将计时器与时钟小部件连接起来：当计时器完成倒计时后，时钟应该更新。为此，Qt 为我们提供了信号和槽系统。当计时器达到倒计时结束时，它调用其 `timeout` 方法。我们使用 `connect` 方法以及 `SIGNAL` 和 `SLOT` 宏将 `timeout` 的调用与 Qt `QWidget` 类中的 `update` 方法的调用连接起来，该调用更新时钟的绘制。`SIGNAL` 宏注册了调用 `timeout` 将引发一个信号，`SLOT` 宏注册了当信号被引发时将调用更新方法，`connect` 方法将信号与槽连接起来。我们已经设置了计时器的超时与时钟更新的连接：

```cpp
      m_timer.setParent(this); 
      connect(&m_timer, SIGNAL(timeout()), this, SLOT(update())); 
      m_timer.start(100); 
    }
```

每当窗口需要重新绘制时，都会调用 `paintEvent` 方法。这可能是由于某些外部原因，例如用户调整窗口大小。也可能是由于对 `QMainWindow` 类的 `update` 方法的调用，这最终会调用 `paintEvent`。

在这种情况下，我们不需要任何关于事件的信息，所以我们用注释包围了 `eventPtr` 参数。`width` 和 `height` 方法给出窗口可绘制部分的宽度和高度，以像素为单位。我们调用 `qMin` 方法来决定窗口的最小边长，并调用 `QTime` 类的 `currentTime` 方法来找到时钟的当前时间：

```cpp
    void Clock::paintEvent(QPaintEvent* /* eventPtr */) { 
      int side = qMin(width(), height()); 
      QTime time = QTime::currentTime();
```

`QPainter` 类可以被视为一个绘图画布。我们首先将其初始化为适当的抗锯齿。然后我们调用 `translate` 和 `scale` 方法将像素中的物理大小转换为 `200` * `200` 单位的逻辑大小：

```cpp
      QPainter painter(this); 
      painter.setRenderHint(QPainter::Antialiasing); 
      painter.setRenderHint(QPainter::TextAntialiasing); 
      painter.translate(width() / 2, height() / 2); 
      painter.scale(side / 200.0, side / 200.0); 
```

我们为分钟绘制 60 条线。每隔第五条线会稍微长一些，以标记当前的小时。对于每一分钟，我们绘制一条线，然后调用 Qt 的 `rotate` 方法，该方法将绘图旋转 `6` 度。这样，我们每次旋转绘图 `6` 度，总共旋转 `60` 次，累计达到 `360` 度，即一整圈：

```cpp
      for (int second = 0; second <= 60; ++second) { 
        if ((second % 5) == 0) { 
          painter.drawLine(QPoint(0, 81), QPoint(0, 98)); 
        } 
        else { 
          painter.drawLine(QPoint(0, 90), QPoint(0, 98)); 
        } 
```

一个完整的跳跃是 `360` 度。对于每条线我们旋转 `6` 度，因为 `360` 除以 `60` 等于 `6` 度。当我们完成旋转后，绘图将重置到其原始设置：

```cpp
        painter.rotate(6); 
      }  
```

我们从 `QTime` 对象中获取当前的小时、分钟、秒和毫秒：

```cpp
      double hours = time.hour(), minutes = time.minute(), 
             seconds = time.second(), milliseconds = time.msec(); 
```

我们将画笔颜色设置为黑色，背景颜色设置为灰色：

```cpp
      painter.setPen(Qt::black); 
      painter.setBrush(Qt::gray);
```

我们定义时针的端点。时针比分针和秒针略粗短。我们定义构成时针端点的三个点。时针的底部长度为 `16` 单位，位于原点 `8` 单位处。因此，我们将底部点的 x 坐标设置为 `8` 和 `-8`，y 坐标为 `8`。最后，我们定义时针的长度为 `60` 单位。这个值是负的，以便与当前旋转相对应：

```cpp
      { static const QPoint hourHand[3] = 
          {QPoint(8, 8), QPoint(-8, 8), QPoint(0, -60)};
```

`save` 方法用于保存 `QPointer` 对象的当前设置。这些设置稍后可以通过 `restore` 方法恢复：

```cpp
        painter.save(); 
```

我们通过计算小时、分钟、秒和毫秒来找出当前时针的确切角度。然后我们旋转以设置时针。每个小时对应 `30` 度，因为我们有 `12` 个小时，`360` 度除以 `12` 等于 `30` 度：

```cpp
        double hour = hours + (minutes / 60.0) + (seconds / 3600.0) + 
                      (milliseconds / 3600000.0); 
        painter.rotate(30.0 * hour); 
```

我们使用时针的三个点调用 `drawConvexPloygon` 方法：

```cpp
        painter.drawConvexPolygon(hourHand, 3); 
        painter.restore(); 
      } 
```

我们以相同的方式绘制分针。它比时针细长一些。另一个区别是，我们之前有 12 个小时，而现在有 60 分钟。这导致每一分钟对应 `6` 度，因为 `360` 度除以 `60` 等于 `6` 度：

```cpp
      { static const QPoint minuteHand[3] = 
          {QPoint(6, 8), QPoint(-6, 8), QPoint(0, -70)}; 
        painter.save(); 
```

在计算当前分钟角度时，我们使用分钟、秒和毫秒：

```cpp
        double minute = minutes + (seconds / 60.0) + 
                        (milliseconds / 60000.0); 
        painter.rotate(6.0 * minute); 
        painter.drawConvexPolygon(minuteHand, 3); 
        painter.restore(); 
      }
```

秒针的绘制几乎与上一分钟针的绘制相同。唯一的区别是我们只使用秒和毫秒来计算秒的角度：

```cpp
      { static const QPoint secondHand[3] = 
          {QPoint(4, 8), QPoint(-4, 8), QPoint(0, -80)}; 

        painter.save(); 
        double second = seconds + (milliseconds / 1000); 
        painter.rotate(6.0 * second); 
        painter.drawConvexPolygon(secondHand, 3); 
        painter.restore(); 
      } 
    } 
```

# 主函数

在 `main` 函数中，我们初始化并启动 Qt 应用程序。`main` 函数可以接受 `argc` 和 `argv` 参数。它包含应用程序的命令行参数；`argc` 包含参数的数量，而 `argv` 数组包含参数本身。`argv` 的第一个条目始终包含执行文件的路径，最后一个条目始终是 `nullptr`。`QApplication` 类接受 `argc` 和 `argv` 并初始化 Qt 应用程序。我们创建了一个 `Clock` 类的对象，并调用 `show` 使其可见。最后，我们调用 `QApplication` 对象的 `exec`。

**Main.cpp:**

```cpp
    #include <QApplication> 
    #include "Clock.h" 

    int main(int argc, char *argv[]) { 
      QApplication application(argc, argv); 
      Clock Clock; 
      Clock.show(); 
      return application.exec(); 
    }
```

要执行应用程序，我们选择项目的运行选项：

![图片](img/4e8a7f8f-c0bc-4cfb-b4b7-8aa2fa5e81f1.png)

执行将继续，直到用户通过按下右上角的关闭按钮关闭 `Clock` 窗口：

![图片](img/f489094c-a214-400d-bdb7-32e03e57e385.png)

# 设置窗口和控件的可重用类

在图形应用中，有窗口和控件。窗口通常是一个完整的窗口，包含一个带有标题、菜单栏和关闭及调整窗口大小的按钮的框架。控件通常是一个较小的图形对象，通常嵌入在窗口中。在 *Clock* 项目中，我们只使用了继承自 `QWidget` 类的 `widget` 类。然而，在本节中，我们将离开 *Clock* 项目，探讨带有窗口和控件的更高级应用。窗口包含带有菜单栏和工具栏的框架，而控件位于窗口内，负责图形内容。

在本章的后续部分，我们将探讨绘图程序和编辑器。这些应用是典型的文档应用，其中我们打开和保存文档，以及剪切、复制、粘贴和删除文档元素。为了向窗口添加菜单和工具栏，我们需要继承两个 Qt 类，`QMainWindow` 和 `QWidget`。我们需要 `QMainWindow` 来向窗口框架添加菜单和工具栏，以及 `QWidget` 来在窗口区域绘制图像。

为了在本书剩余部分和下一章介绍的应用程序中重用文档代码，在本节中，我们定义了 `MainWindow` 和 `DocumentWidget` 类。这些类将在本章后续部分中的绘图程序和编辑器中使用。`MainWindow` 设置了一个带有 `文件` 和 `编辑` 菜单和工具栏的窗口，而 `DocumentWidget` 提供了一个框架，为 `新建`、`打开`、`保存`、`另存为`、`剪切`、`复制`、`粘贴`、`删除` 和 `退出` 项目设置了基本代码。在本节中，我们不会创建一个新的 Qt 项目，我们只会编写 `MainWindow` 和 `DocumentWidget` 类，这些类将在本章后续部分的绘图程序和编辑器中作为基类使用，以及 `LISTENER` 宏，它用于设置菜单和工具栏项。

# 添加监听器

监听器是在用户选择菜单项或工具栏项时被调用的方法。`Listener` 宏将监听器添加到类中。

**Listener.h:**

```cpp
    #ifndef LISTENER_H 
    #define LISTENER_H 

    #include <QObject>
```

由于 Qt 关于菜单和工具栏的规则，Qt 框架在响应用户操作时调用的监听器必须是一个函数而不是一个方法。

方法属于一个类，而函数是独立的。

`DefineListener` 宏定义了一个友好的函数和一个方法。Qt 框架调用该函数，该函数随后调用该方法：

```cpp
    #define DEFINE_LISTENER(BaseClass, Listener)           
      friend bool Listener(QObject* baseObjectPtr) {       
         return ((BaseClass*) baseObjectPtr)->Listener();  
      }                                                    
      bool Listener()                                      
```

`Listener` 宏定义为指向方法的指针：

```cpp
    #define LISTENER(Listener) (&::Listener) 
```

监听器方法接受一个 `QObject` 指针作为参数，并返回一个布尔值：

```cpp
    typedef bool (*Listener)(QObject*); 
    #endif // LISTENER_H 
```

# 基础窗口类

`MainWindow` 类使用 `File` 和 `Edit` 菜单和工具栏设置文档窗口。它还提供了 `addAction` 方法，该方法旨在供子类添加特定于应用程序的菜单和工具栏。

**MainWindow.h:**

```cpp
    #ifndef MAINWINDOW_H 
    #define MAINWINDOW_H 

    #include <QMainWindow> 
    #include <QActionGroup> 
    #include <QPair> 
    #include <QMap> 

    #include "Listener.h" 
    #include "DocumentWidget.h" 

    class MainWindow : public QMainWindow { 
      Q_OBJECT 

      public: 
        MainWindow(QWidget* parentWidgetPtr = nullptr); 
        ~MainWindow(); 

      protected: 
        void addFileMenu(); 
        void addEditMenu(); 
```

`addAction` 方法添加一个带有潜在快捷键、工具栏图标和监听器的菜单项，用于标记该项为复选框或单选按钮：

```cpp
      protected: 
        void addAction(QMenu* menuPtr, QString text, 
                       const char* onSelectPtr, 
                       QKeySequence acceleratorKey = 0, 
                       QString iconName = QString(), 
                       QToolBar* toolBarPtr = nullptr, 
                       QString statusTip = QString(), 
                       Listener enableListener = nullptr, 
                       Listener checkListener = nullptr, 
                       QActionGroup* groupPtr = nullptr); 
```

我们使用 `DefineListener` 宏添加一个监听器来决定菜单项是否应该启用。如果项应该启用，监听器返回 `true`。`DocumentWidget` 是 Qt 类 `QWidget` 的子类，我们将在下一节中定义它。使用 `DEFINE_LISTENER` 宏，我们将 `isSaveEnabled`、`isCutEnabled`、`isCopyEnabled`、`isPasteEnabled` 和 `isDeleteEnabled` 方法添加到 `MainWindow` 类中。当用户选择菜单项时，它们将被调用：

```cpp
        DEFINE_LISTENER(DocumentWidget, isSaveEnabled); 
        DEFINE_LISTENER(DocumentWidget, isCutEnabled); 
        DEFINE_LISTENER(DocumentWidget, isCopyEnabled); 
        DEFINE_LISTENER(DocumentWidget, isPasteEnabled); 
        DEFINE_LISTENER(DocumentWidget, isDeleteEnabled); 
```

`onMenuShow` 方法在菜单变得可见之前被调用；它调用菜单项的监听器来决定它们是否应该被禁用或用复选框或单选按钮进行注释。它还由框架调用以禁用工具栏图标：

```cpp
      public slots: 
        void onMenuShow();
```

`m_enableMap` 和 `m_checkMap` 字段包含菜单项的监听器映射。前面的 `onMenuShow` 方法使用它们来决定是否禁用项，或用复选框或单选按钮对其进行注释：

```cpp
      private: 
        QMap<QAction*,QPair<QObject*,Listener>> m_enableMap, 
                                                m_checkMap; 
    }; 

    #endif // MAINWINDOW_H 
```

**MainWindow.cpp:**

```cpp
    #include "MainWindow.h" 
    #include <QtWidgets> 
```

构造函数调用 Qt `QMainWindow` 类的构造函数，将父小部件指针作为其参数：

```cpp
    MainWindow::MainWindow(QWidget* parentWidgetPtr /*= nullptr*/) 
     :QMainWindow(parentWidgetPtr) { 
    } 
```

当添加菜单项时，它连接到一个动作。析构函数释放菜单栏的所有动作：

```cpp
    MainWindow::~MainWindow() { 
      for (QAction* actionPtr : menuBar()->actions()) { 
        delete actionPtr; 
      } 
    } 
```

`addFileMenu` 方法将标准 `File` 菜单添加到菜单栏；`menubar` 是 Qt 方法，它返回窗口菜单栏的指针：

```cpp
    void MainWindow::addFileMenu() { 
      QMenu* fileMenuPtr = menuBar()->addMenu(tr("&File"));
```

与下面的代码片段中的`connect`方法类似，该方法将菜单项与`onMenuShow`方法连接。Qt 宏`SIGNAL`和`SLOT`确保在菜单变得可见之前调用`onMenuShow`。`onMenuShow`方法在菜单变得可见之前为菜单中的每个项设置启用、复选框和单选按钮状态。它还设置工具栏图像的启用状态。`aboutToShow`方法在每次菜单变得可见之前被调用，以启用或禁用项，并可能用复选框或单选按钮标记它们：

```cpp
      connect(fileMenuPtr, SIGNAL(aboutToShow()), this, 
              SLOT(onMenuShow())); 
```

Qt 的`addToolBar`方法将工具栏添加到窗口的框架中。当我们在这里调用`addAction`时，菜单项将被添加到菜单中，如果存在，也将添加到工具栏中：

```cpp
      QToolBar *fileToolBarPtr = addToolBar(tr("File")); 
```

`addAction`方法添加`New`、`Open`、`Save`、`SaveAs`和`Exit`菜单项。它接受以下参数：

+   指向该项应属于的菜单的指针。

+   项文本。文本前的符号`&`（例如`&New`）表示下一个字母（`N`）将被下划线标记，并且用户可以通过按下 *Alt*-*N* 来选择该项。

+   加速器信息。`QKeySequence`是 Qt 枚举，包含加速键组合。`QKeySequence::New`表示用户可以通过按下 *Ctrl*-*N* 来选择该项。文本`Ctrl+N`也将添加到项文本中。

+   图标文件的名称（`new`）。文件图标既显示在项文本的左侧，也显示在工具栏上。图标文件本身是在 Qt Creator 中添加到项目中的。

+   指向工具栏的指针，如果该项未连接到工具栏则为`nullptr`。

+   当用户将鼠标悬停在工具栏项上时显示的文本。如果该项未连接到工具栏，则忽略。

+   在菜单和工具栏变得可见之前被调用的监听器（默认`nullptr`），并决定该项是否启用或用复选框或单选按钮标记：

```cpp
  addAction(fileMenuPtr, tr("&New"), SLOT(onNew()), 
            QKeySequence::New, tr("new"), fileToolBarPtr, 
            tr("Create a new file")); 

  addAction(fileMenuPtr, tr("&Open"), SLOT(onOpen()), 
            QKeySequence::Open, tr("open"), fileToolBarPtr, 
            tr("Open an existing file")); 
```

如果自上次保存以来文档没有变化，则不需要保存文档，并且应禁用“保存”项。因此，我们添加了一个额外的参数，表示应调用`isSaveEnabled`方法来启用或禁用菜单和工具栏项：

```cpp
      addAction(fileMenuPtr, tr("&Save"), SLOT(onSave()), 
                QKeySequence::Save, tr("save"), fileToolBarPtr, 
                tr("Save the document to disk"), 
                LISTENER(isSaveEnabled)); 
```

`SaveAs`菜单项没有快捷键序列。此外，它没有工具栏条目。因此，图标文件名和工具栏文本是默认的`QString`对象，工具栏指针是`nullptr`：

```cpp
      addAction(fileMenuPtr, tr("Save &As"), SLOT(onSaveAs()), 
                0, QString(), nullptr, QString(), 
                LISTENER(isSaveEnabled)); 
```

`addSeparator`方法在两个项之间添加一条水平线：

```cpp
      fileMenuPtr->addSeparator(); 
      addAction(fileMenuPtr, tr("E&xit"), 
                SLOT(onExit()), QKeySequence::Quit); 
    } 
```

`addEditMenu`方法以与前面的`File`菜单相同的方式将`Edit`菜单添加到窗口的菜单栏中：

```cpp
    void MainWindow::addEditMenu() { 
      QMenu* editMenuPtr = menuBar()->addMenu(tr("&Edit")); 
      QToolBar* editToolBarPtr = addToolBar(tr("Edit")); 
      connect(editMenuPtr, SIGNAL(aboutToShow()), 
              this, SLOT(onMenuShow())); 

      addAction(editMenuPtr, tr("&Cut"), SLOT(onCut()), 
                QKeySequence::Cut, tr("cut"), editToolBarPtr, 
          tr("Cut the current selection's contents to the clipboard"), 
                LISTENER(isCutEnabled)); 

      addAction(editMenuPtr, tr("&Copy"), SLOT(onCopy()), 
                QKeySequence::Copy, tr("copy"), editToolBarPtr, 
         tr("Copy the current selection's contents to the clipboard"), 
                LISTENER(isCopyEnabled)); 

      addAction(editMenuPtr, tr("&Paste"), SLOT(onPaste()), 
                QKeySequence::Paste, tr("paste"), editToolBarPtr, 
        tr("Paste the current selection's contents to the clipboard"), 
                LISTENER(isPasteEnabled)); 

      editMenuPtr->addSeparator(); 
      addAction(editMenuPtr, tr("&Delete"), SLOT(onDelete()), 
                QKeySequence::Delete, tr("delete"), editToolBarPtr, 
                tr("Delete the current selection"), 
                LISTENER(isDeleteEnabled)); 
    } 
```

`addAction`方法将菜单项添加到菜单栏，并将工具栏图标添加到工具栏。它还将项目与当用户选择项目时被调用的`onSelectPtr`方法连接起来，以及启用项目并使用复选框或单选按钮进行标注的方法。除非为零，否则将添加一个加速器到动作中。`groupPtr`参数定义了项目是否是组的一部分。如果`checkListener`不为`nullptr`，则如果`groupPtr`为`nullptr`，项目将用复选框标注，如果不是，则用单选按钮标注。在单选按钮的情况下，组中只有一个单选按钮会被同时标记：

```cpp
    void MainWindow::addAction(QMenu* menuPtr, QString itemText, 
                               const char* onSelectPtr, 
                               QKeySequence acceleratorKey /* = 0 */, 
                               QString iconName /*= QString()*/, 
                               QToolBar* toolBarPtr /*= nullptr*/, 
                               QString statusTip /*= QString()*/, 
                               Listener enableListener /*= nullptr*/, 
                               Listener checkListener /*= nullptr*/, 
                               QActionGroup* groupPtr /*= nullptr*/) { 
      QAction* actionPtr; 
```

如果`iconName`不为空，我们从项目资源中的文件加载图标，然后创建一个新的带有图标的`QAction`对象：

```cpp
      if (!iconName.isEmpty()) { 
        const QIcon icon = QIcon::fromTheme("document-" + iconName, 
                           QIcon(":/images/" + iconName + ".png")); 
        actionPtr = new QAction(icon, itemText, this); 
      } 
```

如果`iconName`为空，我们创建一个新的不带图标的`QAction`对象：

```cpp
      else { 
        actionPtr = new QAction(itemText, this); 
      }
```

我们将菜单项连接到选择方法。当用户选择项目或点击工具栏图标时，会调用`onSelectPtr`：

```cpp
      connect(actionPtr, SIGNAL(triggered()), 
              centralWidget(), onSelectPtr); 
```

如果加速键不是零，我们将它添加到动作指针中：

```cpp
      if (acceleratorKey != 0) { 
        actionPtr->setShortcut(acceleratorKey); 
      } 
```

最后，我们将动作指针添加到菜单指针中，以便它能够处理用户的项选择：

```cpp
      menuPtr->addAction(actionPtr); 
```

如果`toolBarPtr`不是`nullptr`，我们将动作添加到窗口的工具栏中：

```cpp
      if (toolBarPtr != nullptr) { 
        toolBarPtr->addAction(actionPtr); 
      } 
```

如果状态提示不为空，我们将它添加到工具栏的提示和状态提示中：

```cpp
      if (!statusTip.isEmpty()) { 
          actionPtr->setToolTip(statusTip); 
          actionPtr->setStatusTip(statusTip); 
      } 
```

如果启用监听器不为空，我们将一个由窗口中心部件的指针和监听器组成的对添加到`m_enableMap`中。同时，我们调用监听器以初始化菜单项和工具栏图标的启用状态：

```cpp
      if (enableListener != nullptr) { 
        QWidget* widgetPtr = centralWidget(); 
        m_enableMap[actionPtr] = 
          QPair<QObject*,Listener>(widgetPtr, enableListener); 
        actionPtr->setEnabled(enableListener(widgetPtr)); 
      } 
```

同样，如果检查监听器不为空，我们将窗口中心小部件的指针和监听器添加到`m_checkMap`中。`m_enableMap`和`m_checkMap`都由`onMenuShow`使用，如下所示。我们还调用监听器以初始化菜单项的检查状态（工具栏图标不会被勾选）：

```cpp
      if (checkListener != nullptr) { 
        actionPtr->setCheckable(true); 
        QWidget* widgetPtr = centralWidget(); 
        m_checkMap[actionPtr] = 
          QPair<QObject*,Listener>(widgetPtr, checkListener); 
        actionPtr->setChecked(checkListener(widgetPtr)); 
      } 
```

最后，如果组指针不为空，我们将动作添加到其中。这样，菜单项将通过单选按钮而不是复选框进行标注。框架也会跟踪组，并确保每个组中只有一个单选按钮被同时标记：

```cpp
      if (groupPtr != nullptr) { 
        groupPtr->addAction(actionPtr); 
      } 
    } 
```

`onMenuShow`方法在菜单或工具栏图标变得可见之前被调用。它确保每个项目都被启用或禁用，并且项目被标注为复选框或单选按钮。

我们首先遍历启用映射。对于映射中的每个条目，我们查找小部件和启用函数。我们调用该函数，它返回`true`或`false`，然后通过在动作对象指针上调用`setEnabled`来使用结果启用或禁用项目：

```cpp
    void MainWindow::onMenuShow() { 
      for (QMap<QAction*,QPair<QObject*,Listener>>::iterator i = 
           m_enableMap.begin(); i != m_enableMap.end(); ++i) { 
        QAction* actionPtr = i.key(); 
        QPair<QObject*,Listener> pair = i.value(); 
        QObject* baseObjectPtr = pair.first; 
        Listener enableFunction = pair.second; 
        actionPtr->setEnabled(enableFunction(baseObjectPtr)); 
      } 
```

同样地，我们遍历检查映射。对于映射中的每个条目，我们查找小部件和检查函数。我们调用该函数，并使用结果通过在动作对象指针上调用 `setCheckable` 和 `setChecked` 来检查项目。Qt 框架确保如果项目属于一个组，则通过单选按钮进行注释，如果不属于，则通过复选框进行注释：

```cpp
      for (QMap<QAction*,QPair<QObject*,Listener>>::iterator i = 
           m_checkMap.begin(); i != m_checkMap.end(); ++i) { 
        QAction* actionPtr = i.key(); 
        QPair<QObject*,Listener> pair = i.value(); 
        QObject* baseObjectPtr = pair.first; 
        Listener checkFunction = pair.second; 
        actionPtr->setCheckable(true); 
        actionPtr->setChecked(checkFunction(baseObjectPtr)); 
      } 
    } 
```

# 基础小部件类

`DocumentWidget` 是处理文档的应用程序的骨架框架。它处理文档的加载和保存，并为子类提供覆盖 `Cut`、`Copy`、`Paste` 和 `Delete` 菜单项的方法。

当前的 `MainWindow` 类处理窗口框架，包括其菜单和工具栏，而 `DocumentWidget` 类则负责绘制窗口内容。其理念是 `MainWindow` 的子类创建一个 `DocumentWidget` 子类的对象，并将其放置在窗口的中心。请参阅下一节中 `DrawingWindow` 和 `EditorWindow` 的构造函数。

**DocumentWidget.h:**

```cpp
    #ifndef DOCUMENTWIDGET_H 
    #define DOCUMENTWIDGET_H 

    #include "Listener.h" 
    #include <QWidget> 
    #include <QtWidgets> 
    #include <FStream> 
    using namespace std; 

    class DocumentWidget : public QWidget { 
      Q_OBJECT 
```

构造函数接受要显示在窗口顶部横幅中的应用程序名称，用于加载和存储文档的标准文件对话框的文件名掩码，以及指向潜在父小部件的指针（通常是包含的主窗口）：

```cpp
      public: 
        DocumentWidget(const QString& name, const QString& fileMask, 
                       QWidget* parentWidgetPtr); 
        ~DocumentWidget();
```

`setFilePath` 方法设置当前文档的路径。路径在窗口的顶部横幅中显示，并在标准加载和保存对话框中作为默认路径给出：

```cpp
      protected: 
        void setFilePath(QString filePath); 
```

当文档被修改时，会设置修改标志（有时称为脏标志）。这会导致在窗口顶部横幅的文件路径旁边出现一个星号（`*`），并启用 `Save` 和 `SaveAs` 菜单项：

```cpp
      public: 
        void setModifiedFlag(bool flag); 
```

`setMainWindowTitle` 方法是一个辅助方法，用于组合窗口的标题。它由文件路径和一个潜在的星号（`*`）组成，以指示是否设置了修改标志：

```cpp
      private: 
        void setMainWindowTitle(); 
```

`closeEvent` 方法是从 `QWidget` 重写的，当用户关闭窗口时被调用。通过设置 `eventPtr` 参数的字段，可以阻止关闭。例如，如果文档尚未保存，可以询问用户是否要保存文档或取消窗口的关闭：

```cpp
      public: 
        virtual void closeEvent(QCloseEvent* eventPtr); 
```

`isClearOk` 方法是一个辅助方法，如果用户尝试在不保存文档的情况下关闭窗口或退出应用程序，则会显示消息框：

```cpp
      private: 
        bool isClearOk(QString title); 
```

当用户选择菜单项或点击工具栏图标时，框架会调用以下方法。为了使其工作，我们将这些方法标记为槽，这是在 `connect` 调用中的 `SLOT` 宏所必需的：

```cpp
      public slots: 
        virtual void onNew(); 
        virtual void onOpen(); 
        virtual bool onSave(); 
        virtual bool onSaveAs(); 
        virtual void onExit();
```

当文档没有更改时，没有必要保存它。在这种情况下，`Save` 和 `SaveAs` 菜单项和工具栏图像应禁用。`isSaveEnabled` 方法在 `File` 菜单可见之前由 `onMenuShow` 调用。它仅在文档已更改且需要保存时返回 true：

```cpp
    virtual bool isSaveEnabled(); 
```

`tryWriteFile` 方法是一个辅助方法，它尝试写入文件。如果失败，将显示一个消息框显示错误信息：

```cpp
    private: 
        bool tryWriteFile(QString filePath); 
```

以下方法是虚拟方法，旨在由子类重写。当用户选择 `New`、`Save`、`SaveAs` 和 `Open` 菜单项时，会调用这些方法：

```cpp
      protected: 
        virtual void newDocument() = 0; 
        virtual bool writeFile(const QString& filePath) = 0; 
        virtual bool readFile(const QString& filePath) = 0; 
```

在编辑菜单可见之前，会调用以下方法，并决定是否启用 `Cut`、`Copy`、`Paste` 和 `Delete` 项目：

```cpp
      public: 
        virtual bool isCutEnabled(); 
        virtual bool isCopyEnabled(); 
        virtual bool isPasteEnabled(); 
        virtual bool isDeleteEnabled(); 
```

当用户选择 `Cut`、`Copy`、`Paste` 和 `Delete` 项目或工具栏图标时，会调用以下方法：

```cpp
      public slots: 
        virtual void onCut(); 
        virtual void onCopy(); 
        virtual void onPaste(); 
        virtual void onDelete(); 
```

`m_applicationName` 字段包含应用程序的名称，而不是文档的名称。在下一节中，名称将是 *Drawing* 和 *Editor*。`m_fileMask` 字段包含在标准对话框中加载和保存文档时使用的掩码。例如，假设我们有以 `.abc` 结尾的文档。那么掩码可以是 `Abc files (.abc)`。`m_filePath` 字段包含当前文档的路径。当文档是新的且尚未保存时，该字段包含空字符串。

最后，当文档已被修改且在应用程序退出之前需要保存时，`m_modifiedFlag` 为真：

```cpp
      private: 
        QString m_applicationName, m_fileMask, m_filePath; 
        bool m_modifiedFlag = false; 
    }; 
```

最后，还有一些重载的辅助运算符。加法和减法运算符将一个点与一个大小相加或相减，以及一个具有大小的矩形：

```cpp
    QPoint& operator+=(QPoint& point, const QSize& size); 
    QPoint& operator-=(QPoint& point, const QSize& size); 

    QRect& operator+=(QRect& rect, int size); 
    QRect& operator-=(QRect& rect, int size); 
```

`writePoint` 和 `readPoint` 方法从输入流中写入和读取一个点：

```cpp
    void writePoint(ofstream& outStream, const QPoint& point); 
    void readPoint(ifstream& inStream, QPoint& point); 
```

`writeColor` 和 `readColor` 方法从输入流中写入和读取一个颜色：

```cpp
    void writeColor(ofstream& outStream, const QColor& color); 
    void readColor(ifstream& inStream, QColor& color); 
```

`makeRect` 方法创建一个以 `point` 为中心，`size` 为大小的矩形：

```cpp
    QRect makeRect(const QPoint& centerPoint, int halfSide); 
    #endif // DOCUMENTWIDGET_H 
```

**DocumentWidget.cpp:**

```cpp
    #include <QtWidgets> 
    #include <QMessageBox> 

    #include "MainWindow.h" 
    #include "DocumentWidget.h" 
```

构造函数设置应用程序的名称、保存和加载标准对话框的文件掩码，以及指向封装父小部件的指针（通常是封装的主窗口）：

```cpp
    DocumentWidget::DocumentWidget(const QString& name, 
                    const QString& fileMask, QWidget* parentWidgetPtr) 
     :m_applicationName(name), 
      m_fileMask(fileMask), 
      QWidget(parentWidgetPtr) { 
        setMainWindowTitle(); 
      } 
```

析构函数不执行任何操作，仅为了完整性而包含：

```cpp
    DocumentWidget::~DocumentWidget() { 
      // Empty. 
    } 
```

`setFilePath` 方法调用 `setMainWindowTitle` 来更新窗口顶部横幅上的文本：

```cpp
    void DocumentWidget::setFilePath(QString filePath) { 
      m_filePath = filePath; 
      setMainWindowTitle(); 
    } 
```

`setModifiedFlag` 方法还会调用 `setMainWindowTitle` 来更新窗口顶部横幅上的文本。此外，它还会在父小部件上调用 `onMenuShow` 以更新工具栏的图标：

```cpp
    void DocumentWidget::setModifiedFlag(bool modifiedFlag) { 
      m_modifiedFlag = modifiedFlag; 
      setMainWindowTitle(); 
      ((MainWindow*) parentWidget())->onMenuShow(); 
    } 
```

工具栏顶部横幅上显示的标题是应用程序名称、文档文件路径（如果非空），以及如果文档未经保存而修改，则显示一个星号：

```cpp
    void DocumentWidget::setMainWindowTitle() { 
      QString title= m_applicationName + 
              (m_filePath.isEmpty() ? "" : (" [" + m_filePath + "]"))+ 
              (m_modifiedFlag ? " *" : ""); 
      this->parentWidget()->setWindowTitle(title); 
    } 
```

`isClearOk` 方法会在文档未经保存而修改时显示一个消息框。用户可以选择以下按钮之一：

+   是：保存文档并退出应用程序。但是，如果保存失败，将显示错误消息，并且应用程序不会退出。

+   否：应用程序退出而不保存文档。

+   取消：取消应用程序的关闭。文档不会被保存。

```cpp
    bool DocumentWidget::isClearOk(QString title) { 
      if (m_modifiedFlag) { 
        QMessageBox messageBox(QMessageBox::Warning, 
                               title, QString()); 
        messageBox.setText(tr("The document has been modified.")); 
        messageBox.setInformativeText( 
                   tr("Do you want to save your changes?")); 
        messageBox.setStandardButtons(QMessageBox::Yes | 
                              QMessageBox::No | QMessageBox::Cancel); 
        messageBox.setDefaultButton(QMessageBox::Yes); 

        switch (messageBox.exec()) { 
          case QMessageBox::Yes: 
            return onSave(); 

          case QMessageBox::No: 
            return true; 

          case QMessageBox::Cancel: 
            return false; 
        } 
      } 

      return true; 
    } 
```

如果文档被清除，则会调用`newDocument`，该函数旨在被子类覆盖以执行特定于应用程序的初始化。此外，修改标志和文件路径也会被清除。最后，调用 Qt 的 `update` 方法来强制重绘窗口内容：

```cpp
    void DocumentWidget::onNew() { 
      if (isClearOk(tr("New File"))) { 
        newDocument(); 
        setModifiedFlag(false); 
        setFilePath(QString()); 
        update(); 
      } 
    } 
```

如果文档被清除，`onOpen` 会使用标准打开对话框来获取文档的文件路径：

```cpp
    void DocumentWidget::onOpen() { 
      if (isClearOk(tr("Open File"))) { 
        QString file = 
          QFileDialog::getOpenFileName(this, tr("Open File"), 
                       tr("C:\Users\Stefan\Documents\" 
                          "A A_Cpp_By_Example\Draw"), 
                  m_fileMask + tr(";;Text files (*.txt)")); 
```

如果文件成功读取，则清除修改标志，设置文件路径，并调用 `update` 来强制重绘窗口内容：

```cpp
        if (!file.isEmpty()) { 
          if (readFile(file)) { 
            setModifiedFlag(false); 
            setFilePath(file); 
            update(); 
          } 
```

然而，如果读取不成功，将显示一个包含错误信息的消息框：

```cpp
          else { 
            QMessageBox messageBox; 
            messageBox.setIcon(QMessageBox::Critical); 
            messageBox.setText(tr("Read File")); 
            messageBox.setInformativeText(tr("Could not read "") + 
                                          m_filePath  + tr(""")); 
            messageBox.setStandardButtons(QMessageBox::Ok); 
            messageBox.setDefaultButton(QMessageBox::Ok); 
            messageBox.exec(); 
          } 
        } 
      } 
    } 
```

`ifSaveEnabled` 方法简单地返回 `m_modifiedFlag` 的值。但是，我们需要这个方法以便监听器可以工作：

```cpp
    bool DocumentWidget::isSaveEnabled() { 
      return m_modifiedFlag; 
    } 
```

当用户选择 `Save` 或 `SaveAs` 菜单项或工具栏图标时，会调用 `onSave` 方法。如果文档已经有一个名称，我们只需尝试写入文件。但是，如果没有给出名称，我们则调用 `OnSaveAs`，这会为用户显示标准的保存对话框：

```cpp
    bool DocumentWidget::onSave() { 
      if (!m_filePath.isEmpty()) { 
        return tryWriteFile(m_filePath); 
      } 
      else { 
        return onSaveAs(); 
      } 
    } 
```

当用户选择 `SaveAs` 菜单项（此项目没有工具栏图标）时，会调用 `onSaveAs` 方法。它打开标准打开对话框并尝试写入文件。如果写入不成功，则返回 `false`。这是因为 `isClearOk` 只在写入成功时关闭窗口：

```cpp
    bool DocumentWidget::onSaveAs() { 
      QString filePath = 
              QFileDialog::getSaveFileName(this, tr("Save File"), 
                   tr("C:\Users\Stefan\Documents\" 
                      "A A_Cpp_By_Example\Draw"), 
                m_fileMask + tr(";;Text files (*.txt)")); 

      if (!filePath.isEmpty()) { 
        return tryWriteFile(filePath); 
      } 
      else { 
        return false; 
      } 
    } 
```

`tryWriteFile` 方法尝试通过调用 `write` 来写入文件，该函数旨在被子类覆盖。如果成功，则设置修改标志和文件路径。如果文件未能成功写入，将显示一个包含错误信息的消息框：

```cpp
    bool DocumentWidget::tryWriteFile(QString filePath) { 
      if (writeFile(filePath)) { 
        setModifiedFlag(false); 
        setFilePath(filePath); 
        return true; 
      } 
      else { 
        QMessageBox messageBox; 
        messageBox.setIcon(QMessageBox::Critical); 
        messageBox.setText(tr("Write File")); 
        messageBox.setInformativeText(tr("Could not write "") + 
                                      filePath  + tr(""")); 
        messageBox.setStandardButtons(QMessageBox::Ok); 
        messageBox.setDefaultButton(QMessageBox::Ok); 
        messageBox.exec(); 
        return false; 
      } 
    } 
```

当用户选择 `Exit` 菜单项时，会调用 `onExit` 方法。它会检查是否可以关闭窗口，如果可以，则退出应用程序：

```cpp
    void DocumentWidget::onExit() { 
      if (isClearOk(tr("Exit"))) { 
        qApp->exit(0); 
      } 
    } 
```

`isCutEnabled` 和 `isDeleteEnabled` 的默认行为是调用 `isCopyEnabled`，因为它们通常在相同的条件下被启用：

```cpp
    bool DocumentWidget::isCutEnabled() { 
      return isCopyEnabled(); 
    } 

    bool DocumentWidget::isDeleteEnabled() { 
      return isCopyEnabled(); 
    } 
```

`onCut` 的默认行为是简单地调用 `onCopy` 和 `onDelete`：

```cpp
    void DocumentWidget::onCut() { 
      onCopy(); 
      onDelete(); 
    } 
```

其他剪切和复制方法的默认行为是返回 `false` 并不执行任何操作，这将使菜单项处于禁用状态，除非子类覆盖这些方法：

```cpp
    bool DocumentWidget::isCopyEnabled() { 
      return false; 
    } 

    void DocumentWidget::onCopy() { 
      // Empty. 
    } 

    bool DocumentWidget::isPasteEnabled() { 
      return false; 
    } 

    void DocumentWidget::onPaste() { 
      // Empty. 
    } 

    void DocumentWidget::onDelete() { 
      // Empty. 
} 
```

最后，当用户尝试关闭窗口时，会调用 `closeEvent`。如果窗口准备就绪可以清除，则会在 `eventPtr` 上调用 `accept`，这将导致窗口关闭，并在全局 `qApp` 对象上调用 `exit`，导致应用程序退出：

```cpp
    void DocumentWidget::closeEvent(QCloseEvent* eventPtr) { 
      if (isClearOk(tr("Close Window"))) { 
        eventPtr->accept(); 
        qApp->exit(0); 
      } 
```

然而，如果窗口尚未准备好清除，则会在 `eventPtr` 上调用 `ignore`，这将导致窗口保持打开状态（并且应用程序继续运行）：

```cpp
      else { 
        eventPtr->ignore(); 
      } 
    } 
```

此外，还有一组处理点、大小、矩形和颜色的辅助函数。以下运算符将一个点与一个大小相加或相减，并返回结果点：

```cpp
    QPoint& operator+=(QPoint& point, const QSize& size) { 
      point.setX(point.x() + size.width()); 
      point.setY(point.y() + size.height()); 
      return point; 
    } 

    QPoint& operator-=(QPoint& point, const QSize& size) { 
      point.setX(point.x() - size.width()); 
      point.setY(point.y() - size.height()); 
      return point; 
    } 
```

以下运算符将一个整数加到或从矩形中减去，并返回结果矩形。加法运算符在所有方向上扩展矩形的大小，而减法运算符在所有方向上缩小矩形的大小：

```cpp
    QRect& operator+=(QRect& rect, int size) { 
      rect.setLeft(rect.left() - size); 
      rect.setTop(rect.top() - size); 
      rect.setWidth(rect.width() + size); 
      rect.setHeight(rect.height() + size); 
      return rect; 
    } 

    QRect& operator-=(QRect& rect, int size) { 
      rect.setLeft(rect.left() + size); 
      rect.setTop(rect.top() + size); 
      rect.setWidth(rect.width() - size); 
      rect.setHeight(rect.height() - size); 
      return rect; 
    } 
```

`writePoint` 和 `readPoint` 函数用于从文件写入和读取一个点。它们分别写入和读取 *x* 和 *y* 坐标：

```cpp
    void writePoint(ofstream& outStream, const QPoint& point) { 
      int x = point.x(), y = point.y(); 
      outStream.write((char*) &x, sizeof x); 
      outStream.write((char*) &y, sizeof y); 
    } 

    void readPoint(ifstream& inStream, QPoint& point) { 
      int x, y; 
      inStream.read((char*) &x, sizeof x); 
      inStream.read((char*) &y, sizeof y); 
      point = QPoint(x, y); 
    } 
```

`writeColor` 和 `readColor` 函数用于从文件写入和读取一个颜色。一个颜色由 `red`（红色）、`green`（绿色）和 `blue`（蓝色）三个分量组成。每个分量是一个介于 `0` 和 `255`（包含）之间的整数。这些方法从文件流中写入和读取分量：

```cpp
    void writeColor(ofstream& outStream, const QColor& color) { 
      int red = color.red(), green = color.green(), 
      blue = color.blue(); 
      outStream.write((char*) &red, sizeof red); 
      outStream.write((char*) &green, sizeof green); 
      outStream.write((char*) &blue, sizeof blue); 
    } 

    void readColor(ifstream& inStream, QColor& color) { 
      int red, green, blue; 
      inStream.read((char*) &red, sizeof red); 
      inStream.read((char*) &green, sizeof green); 
      inStream.read((char*) &blue, sizeof blue);
```

当组件被读取后，我们创建一个 `QColor` 对象，并将其分配给 `color` 参数：

```cpp
      color = QColor(red, green, blue); 
    } 
```

`makeRect` 函数创建一个以点为中心的矩形：

```cpp
    QRect makeRect(const QPoint& centerPoint, int halfSide) { 
      return QRect(centerPoint.x() - halfSide, 
                   centerPoint.y() - halfSide, 
                   2 * halfSide, 2 * halfSide); 
    } 
```

# 构建绘图程序

现在我们开始一个新的项目，利用上一节中提到的主窗口和文档小部件类——*绘图程序*。我们将在本章中从基本版本开始，并在下一章中继续构建更高级的版本。使用本章的绘图程序，我们可以用不同的颜色绘制线条、矩形和椭圆。我们还可以保存和加载我们的绘图。请注意，在这个项目中，窗口和小部件类继承自上一节中的 `MainWindow` 和 `DocumentWidget` 类。

# 图形基类

应用程序中的图形构成一个类层次结构，其中 `Figure` 是基类。其子类是 `Line`、`RectangleX` 和 `EllipseX`，这些将在后面进行描述。我们不能使用 *Rectangle* 和 *Ellipse* 作为我们类的名称，因为这会与具有相同名称的 Qt 方法冲突。我选择简单地在名称中添加一个 '`X`'。

`Figure` 类是抽象的，这意味着我们不能创建该类的对象。我们只能将其用作基类，子类从中继承。

**Figure.h:** 

```cpp
    #ifndef FIGURE_H 
    #define FIGURE_H 

    enum FigureId {LineId, RectangleId, EllipseId}; 

    #include <QtWidgets> 
    #include <FStream> 
    using namespace std; 

    class Figure { 
      public: 
        Figure(); 
```

以下方法都是纯虚的，这意味着它们不需要被定义。包含至少一个纯虚方法的一个类变成抽象类。子类必须定义其所有基类的所有纯虚方法，或者它们自己也成为抽象类。这样，可以保证所有非抽象类的所有方法都被定义。

每个子类定义 `getId` 并返回其类的身份枚举：

```cpp
    virtual FigureId getId() const = 0; 
```

每个图形都有一个起始点和结束点，具体由每个子类来定义：

```cpp
    virtual void initializePoints(QPoint point) = 0; 
    virtual void setLastPoint(QPoint point) = 0; 
```

`isClick` 方法如果图形被点击则返回 `true`：

```cpp
    virtual bool isClick(QPoint mousePoint) = 0; 
```

`move` 方法将图形移动一定距离：

```cpp
    virtual void move(QSize distance) = 0; 
```

`draw` 方法在绘图区域上绘制图形：

```cpp
    virtual void draw(QPainter &painter) const = 0; 
```

`write` 和 `read` 方法将图形从文件中写入和读取；`write` 是常量，因为它不会改变图形：

```cpp
    virtual bool write(ofstream& outStream) const; 
    virtual bool read(ifstream& inStream); 
```

`color` 方法返回图形的颜色。它有两种版本，其中第一种是常量版本，返回一个常量 `QColor` 对象的引用，而第二种是非常量版本，返回一个非常量对象的引用：

```cpp
    const QColor& color() const {return m_color;} 
    QColor& color() {return m_color;}
```

`filled` 方法仅适用于二维图形（矩形和椭圆）。如果图形被填充，则返回 `true`。请注意，第二个版本返回 `m_filled` 字段的引用，允许方法调用者修改 `m_filled` 的值：

```cpp
    virtual bool filled() const {return m_filled;} 
    virtual bool& filled() {return m_filled;} 
```

当图形被标记时，它会在其角落绘制小正方形。正方形的边长由静态字段 `Tolerance` 定义：

```cpp
    static const int Tolerance; 
```

`writeColor` 和 `readColor` 方法是辅助方法，用于读取和写入颜色。由于它们由 `Figure` 类层次结构之外的方法调用，因此它们是静态的：

```cpp
        static void writeColor(ofstream& outStream, 

                               const QColor& color); 
        static void readColor(ifstream& inStream, QColor& color); 
```

每个图形都有一个颜色，它可以被标记或填充：

```cpp
      private: 
        QColor m_color; 
        bool m_marked = false, m_filled = false; 
    }; 

    #endif 
```

`Figure.cpp` 文件包含了 `Figure` 类的定义。它定义了 `Tolerance` 字段以及 `write` 和 `read` 方法。

**Figure.cpp:**

```cpp
    #include "..\MainWindow\DocumentWidget.h" 
    #include "Figure.h" 
```

由于 `Tolerance` 是静态的，必须在全局空间中定义和初始化。我们定义标记正方形的尺寸为 `6` 像素：

```cpp
    const int Figure::Tolerance(6); 
```

仅当从文件读取图形时才调用默认构造函数：

```cpp
    Figure::Figure() { 
      // Empty. 
    }
```

`write` 和 `read` 方法写入和读取图形的颜色以及图形是否被填充：

```cpp
    bool Figure::write(ofstream& outStream) const { 
      writeColor(outStream, m_color); 
      outStream.write((char*) &m_filled, sizeof m_filled); 
      return ((bool) outStream); 
    } 

    bool Figure::read(ifstream& inStream) { 
      readColor(inStream, m_color); 
      inStream.read((char*) &m_filled, sizeof m_filled); 
      return ((bool) inStream); 
    } 
```

# `Line` 子类

`Line` 类是 `Figure` 的子类。通过定义 `Figure` 的每个纯虚方法，它变得非抽象。一条线通过 `Line` 中的 `m_firstPoint` 到 `m_lastPoint` 字段在两个端点之间绘制：

![](img/4723c801-7788-48b8-8399-9bc2e7f4d2b2.png)

**Line.h:**

```cpp
    #ifndef LINE_H 
    #define LINE_H 

    #include <FStream> 
    using namespace std; 

    #include "Figure.h" 

    class Line : public Figure { 
      public:
```

默认构造函数仅在从文件读取 `Line` 对象时调用；`getId` 简单地返回线的身份枚举：

```cpp
    Line(); 
    FigureId getId() const {return LineId;} 
```

一条线有两个端点。当创建线时，这两个点都被设置，当用户移动它时，第二个点被修改：

```cpp
    void initializePoints(QPoint point); 
    void setLastPoint(QPoint point); 
```

`isClick` 方法如果鼠标点击位于线上（带有一些容差），则返回 `true`：

```cpp
    bool isClick(QPoint mousePoint); 
```

`move` 方法将线（及其两个端点）移动给定的距离：

```cpp
    void move(QSize distance); 
```

`draw` 方法在 `QPainter` 对象上绘制线：

```cpp
    void draw(QPainter& painter) const; 
```

`write` 和 `read` 方法将线的端点从文件流中写入和读取：

```cpp
    bool write(ofstream& outStream) const; 
    bool read(ifstream& inStream); 
```

线的第一个和最后一个点存储在 `Line` 对象中：

```cpp
    private: 
      QPoint m_firstPoint, m_lastPoint; 
    }; 

    #endif 
```

`Line.cpp` 文件定义了 `Line` 类的方法。

**Line.cpp:**

```cpp
    #include "..\MainWindow\DocumentWidget.h" 
    #include "Line.h" 

    Line::Line() { 
      // Empty. 
    }
```

当用户向绘图添加新线时，会调用 `initializePoints` 方法。它设置其两个端点：

```cpp
    void Line::initializePoints(QPoint point) { 
      m_firstPoint = point; 
      m_lastPoint = point; 
    } 
```

当用户添加线并修改其形状时，会调用 `setLastPoint` 方法。它设置最后一个点：

```cpp
    void Line::setLastPoint(QPoint point) { 
      m_lastPoint = point; 
    } 
```

`isClick`方法测试用户是否在线条上用鼠标点击。我们需要考虑两种情况。第一种情况是当线条完全垂直时发生的特殊情况，此时端点的`x`坐标相等。我们使用 Qt 的`QRect`类创建一个围绕线条的矩形，并测试该点是否在矩形内：

![](img/d9377e8e-8a2a-4fdf-8c0f-3e6f23276884.png)

```cpp
    bool Line::isClick(QPoint mousePoint) { 
      if (m_firstPoint.x() == m_lastPoint.x()) { 
        QRect lineRect(m_firstPoint, m_lastPoint); 
        lineRect.normalized(); 
        lineRect += Tolerance; 
        return lineRect.contains(mousePoint); 
      }
```

在一般情况中，即线条不是垂直的情况下，我们首先创建一个包含的矩形并测试鼠标指针是否在其中。如果是，我们将`leftPoint`设置为`firstPoint`和`lastPoint`的最左端点，将`rightPoint`设置为最右端点。然后我们计算包含矩形的宽度（`lineWidth`）和高度（`lineHeight`），以及`rightPoint`和`mousePoint`在`x`和`y`方向上的距离（`diffWidth`和`diffHeight`）。

![](img/f051abb6-3200-45a4-bd8e-1ce97c0443a2.png)

由于一致性，如果鼠标指针击中线条，以下等式是正确的：

![](img/94c97888-f928-420d-9f23-9f7308d283d9.png)

然而，为了使左手表达式正好为零，用户必须精确地点击在线条上。因此，我们可以允许有一定的容差。让我们使用`Tolerance`字段：

![](img/fb006091-e45c-4af5-9ac4-13234e8b1bfe.png)

```cpp
        else { 
          QPoint leftPoint = (m_firstPoint.x() < m_lastPoint.x()) 
                             ? m_firstPoint : m_lastPoint, 
                 rightPoint = (m_firstPoint.x() < m_lastPoint.x()) 
                              ? m_lastPoint : m_firstPoint; 

          if ((leftPoint.x() <= mousePoint.x()) && 
              (mousePoint.x() <= rightPoint.x())) { 
            int lineWidth = rightPoint.x() - leftPoint.x(), 
                lineHeight = rightPoint.y() - leftPoint.y(); 

            int diffWidth = mousePoint.x() - leftPoint.x(), 
                diffHeight = mousePoint.y() - leftPoint.y(); 
```

我们必须将`lineHeight`转换为双精度浮点数，以便执行非整数除法：

```cpp
          return (fabs(diffHeight - (((double) lineHeight) / 
                       lineWidth) * diffWidth) <= Tolerance); 
        } 
```

如果鼠标指针位于包含线条的矩形外部，我们直接返回`false`：

```cpp
        return false; 
      } 
    } 
```

`move`方法简单地移动线条的两个端点：

```cpp
    void Line::move(QSize distance) { 
      m_firstPoint += distance; 
      m_lastPoint += distance; 
    } 
```

当绘制线条时，我们设置画笔颜色并绘制线条。`Figure`类的`color`方法返回线条的颜色：

```cpp
    void Line::draw(QPainter& painter) const { 
      painter.setPen(color()); 
      painter.drawLine(m_firstPoint, m_lastPoint); 
    } 
```

当绘制线条时，我们首先在`Figure`中调用`write`来绘制图形的颜色。然后我们写入线条的端点。最后，我们返回输出流的布尔值，如果写入成功则为`true`：

```cpp
    bool Line::write(ofstream& outStream) const { 
      Figure::write(outStream); 
      writePoint(outStream, m_firstPoint); 
      writePoint(outStream, m_lastPoint); 
      return ((bool) outStream); 
    }
```

以同样的方式，当读取线条时，我们首先在`Figure`中调用`read`来读取线条的颜色。然后我们读取线条的端点并返回输入流的布尔值：

```cpp
    bool Line::read(ifstream& inStream) { 
      Figure::read(inStream); 
      readPoint(inStream, m_firstPoint); 
      readPoint(inStream, m_lastPoint); 
      return ((bool) inStream); 
    } 
```

# `Rectangle`子类

`RectangleX`是`Figure`的子类，用于处理矩形。与`Line`类似，它持有两个点，这两个点持有矩形的对角线：

**Rectangle.h**

```cpp
    #ifndef RECTANGLE_H 
    #define RECTANGLE_H 

    #include <FStream> 
    using namespace std; 

    #include "Figure.h" 

    class RectangleX : public Figure { 
      public: 
```

与前面的`Line`类类似，`RectangleX`有一个默认构造函数，用于从文件中读取对象时使用：

```cpp
        RectangleX(); 
        virtual FigureId getId() const {return RectangleId;} 

        RectangleX(const RectangleX& rectangle); 

        virtual void initializePoints(QPoint point); 
        virtual void setLastPoint(QPoint point); 

        virtual bool isClick(QPoint mousePoint); 
        virtual void move(QSize distance); 
        virtual void draw(QPainter& painter) const; 

        virtual bool write(ofstream& outStream) const; 
        virtual bool read(ifstream& inStream); 

      protected: 
        QPoint m_topLeft, m_bottomRight; 
    }; 

    #endif 
```

**Rectangle.cpp**

```cpp
    #include "..\MainWindow\DocumentWidget.h" 
    #include "Rectangle.h" 

    RectangleX::RectangleX() { 
      // Empty. 
    } 
```

`initializePoints`和`setLastPoint`方法的工作方式与`Line`中的对应方法类似：`initializePoints`设置两个角点，而`setLastPoint`设置最后一个角点：

```cpp
    void RectangleX::initializePoints(QPoint point) { 
      m_topLeft = point; 
      m_bottomRight = point; 
    } 

    void RectangleX::setLastPoint(QPoint point) { 
      m_bottomRight = point; 
    } 
```

`isClick`方法比其在`Line`中的对应方法简单：

```cpp
    bool RectangleX::isClick(QPoint mousePoint) { 
      QRect areaRect(m_topLeft, m_bottomRight); 
```

如果矩形被填充，我们简单地通过在`QRect`中调用`contains`来检查鼠标点击是否击中了矩形：

```cpp
      if (filled()) { 
        return areaRect.contains(mousePoint); 
      } 
```

如果矩形没有被填充，我们需要检查鼠标是否点击了矩形的边界。为此，我们创建两个稍微小一些和大一些的矩形。如果鼠标点击击中了较大的矩形，但没有击中较小的矩形，我们认为矩形边界被击中：

```cpp
      else { 
        QRect largeAreaRect(areaRect), smallAreaRect(areaRect); 

        largeAreaRect += Tolerance; 
        smallAreaRect -= Tolerance; 

        return largeAreaRect.contains(mousePoint) && 
               !smallAreaRect.contains(mousePoint); 
      } 

      return false; 
    } 
```

当移动矩形时，我们只需移动第一个和最后一个角：

```cpp
    void RectangleX::move(QSize distance) { 
      addSizeToPoint(m_topLeft, distance); 
      addSizeToPoint(m_bottomRight, distance); 
    } 
```

当绘制矩形时，我们首先通过调用`Figure`中的`color`来设置笔的颜色：

```cpp
    void RectangleX::draw(QPainter& painter) const { 
      painter.setPen(color()); 
```

如果矩形被填充，我们只需在`QPainter`对象上调用`fillRect`：

```cpp
      if (filled()) { 
        painter.fillRect(QRect(m_topLeft, m_bottomRight), color()); 
      } 
```

如果矩形没有被填充，我们禁用画笔使矩形空心，然后调用`QPainter`对象的`drawRect`来绘制矩形的边界：

```cpp
      else { 
        painter.setBrush(Qt::NoBrush); 
        painter.drawRect(QRect(m_topLeft, m_bottomRight)); 
      } 
    } 
```

`write`方法首先在`Figure`中调用`write`，然后写入矩形的第一个和最后一个角：

```cpp
    bool RectangleX::write(ofstream& outStream) const { 
      Figure::write(outStream); 
      writePoint(outStream, m_topLeft); 
      writePoint(outStream, m_bottomRight); 
      return ((bool) outStream); 
    }
```

同样地，`read`首先在`Figure`中调用`read`，然后读取矩形的第一个和最后一个角：

```cpp
    bool RectangleX::read (ifstream& inStream) { 
      Figure::read(inStream); 
      readPoint(inStream, m_topLeft); 
      readPoint(inStream, m_bottomRight); 
      return ((bool) inStream); 
    } 
```

# 椭圆子类

`EllipseX`是处理椭圆的`RectangleX`子类。`RectangleX`的部分功能在`EllipseX`中被重用。更具体地说，`initializePoints`、`setLastPoint`、`move`、`write`和`read`是从`RectangleX`中重写的。

**Ellipse.h:**

```cpp
    #ifndef ELLIPSE_H 
    #define ELLIPSE_H 

    #include "Rectangle.h" 

    class EllipseX : public RectangleX { 
      public: 
        EllipseX(); 
        FigureId getId() const {return EllipseId;} 

        EllipseX(const EllipseX& ellipse); 

        bool isClick(QPoint mousePoint); 
        void draw(QPainter& painter) const; 
    }; 

    #endif 
```

**Ellipse.cpp:**

```cpp
    #include "..\MainWindow\DocumentWidget.h" 
    #include "Ellipse.h" 

    EllipseX::EllipseX() { 
      // Empty. 
    }
```

`EllipseX`类的`isClick`方法与其在`RectangleX`中的对应方法类似。我们使用 Qt 的`QRegion`类创建椭圆对象，并将其与鼠标点击进行比较：

```cpp
    bool EllipseX::isClick(QPoint mousePoint) { 
      QRect normalRect(m_topLeft, m_bottomRight); 
      normalRect.normalized(); 
```

如果椭圆被填充，我们创建一个椭圆区域并测试鼠标点击是否击中了该区域：

```cpp
      if (filled()) { 
        QRegion normalEllipse(normalRect, QRegion::Ellipse); 
        return normalEllipse.contains(mousePoint); 
      } 
```

如果椭圆没有被填充，我们创建稍微小一些和大一些的椭圆区域。如果鼠标点击击中了较小的区域，但没有击中较大的区域，我们认为椭圆的边界被击中：

```cpp
      else { 
        QRect largeRect(normalRect), smallRect(normalRect); 
        largeRect += Tolerance; 
        smallRect -= Tolerance; 

        QRegion largeEllipse(largeRect, QRegion::Ellipse), 
                smallEllipse(smallRect, QRegion::Ellipse); 

        return (largeEllipse.contains(mousePoint) && 
                !smallEllipse.contains(mousePoint)); 
      } 
    } 
```

当绘制椭圆时，我们首先通过调用`Figure`中的`color`来设置笔的颜色：

```cpp
    void EllipseX::draw(QPainter& painter) const { 
      painter.setPen(color()); 
```

如果椭圆被填充，我们设置画笔并绘制椭圆：

```cpp
      if (filled()) { 
        painter.setBrush(color()); 
        painter.drawEllipse(QRect(m_topLeft, m_bottomRight)); 
      }
```

如果椭圆没有被填充，我们设置画笔为空心并绘制椭圆边界：

```cpp
      else { 
        painter.setBrush(Qt::NoBrush); 
        painter.drawEllipse(QRect(m_topLeft, m_bottomRight)); 
      } 
    }
```

# 绘制窗口

`DrawingWindow`类是上一节中`MainWindow`类的子类。

**DrawingWindow.h:**

```cpp
    #ifndef DRAWINGWINDOW_H 
    #define DRAWINGWINDOW_H 

    #include <QMainWindow> 
    #include <QActionGroup> 

    #include "..\MainWindow\MainWindow.h" 
    #include "DrawingWidget.h" 

    class DrawingWindow : public MainWindow { 
      Q_OBJECT 

      public: 
        DrawingWindow(QWidget* parentWidgetPtr = nullptr); 
        ~DrawingWindow(); 

      public: 
        void closeEvent(QCloseEvent *eventPtr)
             { m_drawingWidgetPtr->closeEvent(eventPtr); } 

      private: 
        DrawingWidget* m_drawingWidgetPtr; 
        QActionGroup* m_figureGroupPtr; 
    }; 

    #endif // DRAWINGWINDOW_H 
```

**DrawingWindow.cpp:**

```cpp
    #include "..\MainWindow\DocumentWidget.h" 
    #include "DrawingWindow.h"
```

构造函数将窗口的大小设置为`1000` * `500`像素：

```cpp
    DrawingWindow::DrawingWindow(QWidget* parentWidgetPtr 
                                 /* = nullptr */) 
     :MainWindow(parentWidgetPtr) { 
      resize(1000, 500); 
```

`m_drawingWidgetPtr`字段被初始化为指向`DrawingWidget`类的一个对象，然后将其设置为窗口的中心部分：

```cpp
      m_drawingWidgetPtr = new DrawingWidget(this); 
      setCentralWidget(m_drawingWidgetPtr); 
```

标准文件菜单被添加到窗口菜单栏：

```cpp
      addFileMenu(); 
```

然后我们添加应用程序特定的格式菜单。它连接到上一节中`DocumentWidget`类的`onMenuShow`方法：

```cpp
      { QMenu* formatMenuPtr = menuBar()->addMenu(tr("F&ormat")); 
        connect(formatMenuPtr, SIGNAL(aboutToShow()), 
                this, SLOT(onMenuShow())); 
```

格式菜单包含颜色和填充项：

```cpp
        addAction(formatMenuPtr, tr("&Color"), 
                  SLOT(onColor()), QKeySequence(Qt::ALT + Qt::Key_C), 
                  QString(), nullptr, tr("Figure Color")); 
```

当绘图程序的下一个图形是一个二维图形（矩形或椭圆）时，填充项将被启用：

```cpp
        addAction(formatMenuPtr, tr("&Fill"), 
                  SLOT(onFill()), QKeySequence(Qt::CTRL + Qt::Key_F), 
                  QString(), nullptr, tr("Figure Fill"), 
                  LISTENER(isFillEnabled)); 
      } 
```

对于图形菜单，我们为线、矩形和椭圆项创建一个新的动作组。它们中只能同时标记一个：

```cpp
      { m_figureGroupPtr = new QActionGroup(this); 

        QMenu* figureMenuPtr = menuBar()->addMenu(tr("F&igure")); 
        connect(figureMenuPtr, SIGNAL(aboutToShow()), 
                this, SLOT(onMenuShow()));
```

当前选中的项应使用单选按钮标记：

```cpp
        addAction(figureMenuPtr, tr("&Line"), 
                  SLOT(onLine()), QKeySequence(Qt::CTRL + Qt::Key_L), 
                  QString(), nullptr, tr("Line Figure"), nullptr, 
                  LISTENER(isLineChecked), m_figureGroupPtr); 
        addAction(figureMenuPtr, tr("&Rectangle"), 
                  SLOT(onRectangle()), 
                  QKeySequence(Qt::CTRL + Qt::Key_R), 
                  QString(), nullptr, tr("Rectangle Figure"), nullptr, 
                  LISTENER(isRectangleChecked), m_figureGroupPtr); 
        addAction(figureMenuPtr, tr("&Ellipse"), 
                  SLOT(onEllipse()), 
                  QKeySequence(Qt::CTRL + Qt::Key_E), 
                  QString(), nullptr, tr("Ellipse Figure"), nullptr, 
                  LISTENER(isEllipseChecked), m_figureGroupPtr); 
      } 
    } 
```

析构函数释放了在构造函数中动态分配的图形组：

```cpp
    DrawingWindow::~DrawingWindow() { 
      delete m_figureGroupPtr; 
    } 
```

# 绘制小部件

`DrawingWidget`是上一节中`DocumentWidget`的子类。它处理鼠标输入、图形的绘制以及绘图的保存和加载。它还提供了决定何时标记和启用菜单项的方法。

**DrawingWidget.h:**

```cpp
    #ifndef DRAWINGWIDGET_H 
    #define DRAWINGWIDGET_H 

    #include "..\MainWindow\MainWindow.h" 
    #include "..\MainWindow\DocumentWidget.h" 
    #include "Figure.h" 

    class DrawingWidget : public DocumentWidget { 
      Q_OBJECT 

      public: 
        DrawingWidget(QWidget* parentWidgetPtr); 
        ~DrawingWidget(); 
```

当用户按下或释放鼠标键或移动鼠标时，会调用重写的`mousePressEvent`、`mouseReleaseEvent`和`mouseMoveEvent`方法：

```cpp
      public: 
        void mousePressEvent(QMouseEvent *eventPtr); 
        void mouseReleaseEvent(QMouseEvent *eventPtr); 
        void mouseMoveEvent(QMouseEvent *eventPtr); 
```

当窗口需要重新绘制时，会调用`paintEvent`方法。这可以由几个原因引起。例如，用户可以修改窗口的大小。重新绘制也可以通过调用`update`方法强制执行，这最终会导致调用`paintEvent`：

```cpp
        void paintEvent(QPaintEvent *eventPtr); 
```

当用户选择新菜单项时，会调用`newDocument`方法，当用户选择保存或另存为项时，会调用`writeFile`，当用户选择打开项时，会调用`readFile`：

```cpp
      private: 
        void newDocument() override; 
        bool writeFile(const QString& filePath); 
        bool readFile(const QString& filePath); 
        Figure* createFigure(FigureId figureId); 
```

当用户选择颜色和填充菜单项时，会调用`onColor`和`onFill`方法：

```cpp
      public slots: 
        void onColor(); 
        void onFill(); 
```

在用户选择格式菜单之前会调用`isFillEnabled`方法。如果它返回`true`，则填充项变为启用：

```cpp
        DEFINE_LISTENER(DrawingWidget, isFillEnabled);
```

在图形菜单可见之前也会调用`isLineChecked`、`isRectangleChecked`和`isEllipseChecked`方法。如果方法返回`true`，则项目会带有单选按钮：

```cpp
        DEFINE_LISTENER(DrawingWidget, isLineChecked); 
        DEFINE_LISTENER(DrawingWidget, isRectangleChecked); 
        DEFINE_LISTENER(DrawingWidget, isEllipseChecked); 
```

当用户选择线条、矩形和椭圆菜单项时，会调用`onLine`、`onRectangle`和`isEllipse`方法：

```cpp
        void onLine(); 
        void onRectangle(); 
        void onEllipse(); 
```

当应用程序运行时，它可以保持`Idle`、`Create`或`Move`模式：

+   `Idle`：当应用程序等待用户输入时。

+   `Create`：当用户向绘图添加新图形时。发生在用户按下左鼠标按钮而没有击中图形时。添加一个新图形，并修改其端点，直到用户释放鼠标按钮。

+   `Move`：当用户移动图形时。发生在用户按下左鼠标按钮并击中图形时。图形被移动，直到用户释放鼠标按钮。

```cpp
      private: 
        enum ApplicationMode {Idle, Create, Move}; 
        ApplicationMode m_applicationMode = Idle; 
        void setApplicationMode(ApplicationMode mode); 
```

`m_currColor`字段保存用户将要添加的下一个图形的颜色；`m_currFilled`决定下一个图形（如果是矩形或椭圆）是否应该填充。`m_addFigureId`方法保存用户将要添加的下一个图形类型（线条、矩形或椭圆）的标识整数：

```cpp
        QColor m_currColor = Qt::black; 
        bool m_currFilled = false; 
        FigureId m_addFigureId = LineId; 
```

当用户按下鼠标按钮并移动图形时，我们需要存储上一个鼠标点，以便计算图形自上次鼠标事件以来移动的距离：

```cpp
        QPoint m_mousePoint;
```

最后，`m_figurePtrList`保存指向绘图图形的指针。绘图中最顶层的图形位于列表的末尾：

```cpp
        QList<Figure*> m_figurePtrList; 
    }; 

    #endif // DRAWINGWIDGET_H 
```

**DrawingWidget.cpp:**

```cpp
    #include "..\MainWindow\DocumentWidget.h" 
    #include "DrawingWidget.h" 

    #include "Line.h" 
    #include "Rectangle.h" 
    #include "Ellipse.h" 
```

构造函数调用基类`DocumentWidget`的构造函数，标题为`Drawing`。它还将保存和加载掩码设置为`Drawing files (*.drw)`，这意味着标准保存和加载对话框中默认选择的文件具有`drw`后缀：

```cpp
    DrawingWidget::DrawingWidget(QWidget* parentWidgetPtr) 
     :DocumentWidget(tr("Drawing"), tr("Drawing files (*.drw)"), 
                     parentWidgetPtr) { 
      // Empty. 
    } 
```

析构函数释放图形指针列表中的图形指针：

```cpp
    DrawingWidget::~DrawingWidget() { 
      for (Figure* figurePtr : m_figurePtrList) { 
        delete figurePtr; 
      } 
    } 
```

`setApplicationMode`方法设置应用程序模式，并在主窗口中调用`onMenuShow`以正确启用工具栏图标：

```cpp
    void DrawingWidget::setApplicationMode(ApplicationMode mode) { 
      m_applicationMode = mode; 
      ((MainWindow*) parent())->onMenuShow(); 
    }
```

当用户选择新菜单项时，会调用`newDocument`。图形指针列表中的图形被释放，列表本身被清除：

```cpp
    void DrawingWidget::newDocument() { 
      for (Figure* figurePtr : m_figurePtrList) { 
        delete figurePtr; 
      } 
      m_figurePtrList.clear(); 
```

用户将要添加的下一个图形是一条黑色线条，并且填充状态为`false`：

```cpp
      m_currColor = Qt::black; 
      m_addFigureId = LineId; 
      m_currFilled = false; 
    } 
```

当用户选择保存或另存为菜单项时，会调用`writeFile`方法：

```cpp
    bool DrawingWidget::writeFile(const QString& filePath) { 
      ofstream outStream(filePath.toStdString()); 
```

我们首先写入当前颜色和填充状态。然后继续写入图形指针列表的大小，以及图形本身：

```cpp
      if (outStream) { 
        writeColor(outStream, m_currColor); 
        outStream.write((char*) &m_currFilled, sizeof m_currFilled); 

        int size = m_figurePtrList.size(); 
        outStream.write((char*) &size, sizeof size); 
```

对于每个图形，我们首先写入其身份编号，然后写入图形本身：

```cpp
        for (Figure* figurePtr : m_figurePtrList) { 
          FigureId figureId = figurePtr->getId(); 
          outStream.write((char*) &figureId, sizeof figureId); 
          figurePtr->write(outStream); 
        } 

        return ((bool) outStream); 
      } 
```

如果文件无法打开，则返回`false`：

```cpp
      return false; 
    }
```

当用户选择打开菜单项时，会调用`readFile`方法。与之前的`writeFile`方法相同，我们读取颜色和填充状态，图形指针列表的大小，然后是图形本身：

```cpp
    bool DrawingWidget::readFile(const QString& filePath) { 
      ifstream inStream(filePath.toStdString()); 

      if (inStream) { 
        readColor(inStream, m_currColor); 
        inStream.read((char*) &m_currFilled, sizeof m_currFilled); 

        int size; 
        inStream.read((char*) &size, sizeof size); 
```

在读取图形时，我们首先读取其身份编号，并调用`createFigure`来创建与图形身份编号对应的类的对象。然后通过调用其指针上的`read`来读取图形的字段。请注意，我们实际上并不真正知道（或关心）它是哪种图形。我们只是调用图形指针的`read`，该指针实际上指向`Line`、`RectangleX`或`EllipseX`类的对象：

```cpp
        for (int count = 0; count < size; ++count) { 
          FigureId figureId = (FigureId) 0; 
          inStream.read((char*) &figureId, sizeof figureId); 
          Figure* figurePtr = createFigure(figureId); 
          figurePtr->read(inStream); 
          m_figurePtrList.push_back(figurePtr); 
        } 

        return ((bool) inStream); 
      } 

      return false; 
    } 
```

根据`figureId`参数的值，`createFigure`方法动态创建`Line`、`RectangleX`或`EllipseX`类的对象：

```cpp
    Figure* DrawingWidget::createFigure(FigureId figureId) { 
      Figure* figurePtr = nullptr; 

      switch (figureId) { 
        case LineId: 
          figurePtr = new Line(); 
          break; 

        case RectangleId: 
          figurePtr = new RectangleX(); 
          break; 

        case EllipseId: 
          figurePtr = new EllipseX(); 
          break; 
      } 

      return figurePtr; 
    } 
```

当用户选择颜色菜单项时，会调用`onColor`方法。它设置用户将要添加的下一个图形的颜色：

```cpp
    void DrawingWidget::onColor() { 
      QColor newColor = QColorDialog::getColor(m_currColor, this); 

      if (newColor.isValid() && (m_currColor != newColor)) { 
        m_currColor = newColor; 
        setModifiedFlag(true); 
      } 
    } 
```

在格式菜单可见之前，会调用`isFillEnabled`方法，如果用户将要添加的下一个图形是矩形或椭圆，则返回`true`：

```cpp
    bool DrawingWidget::isFillEnabled() { 
      return (m_addFigureId == RectangleId) || 
             (m_addFigureId == EllipseId); 
    } 
```

当用户选择填充菜单项时，会调用`onFill`方法。它反转`m_currFilled`字段。它还设置修改标志，因为文档已被影响：

```cpp
    void DrawingWidget::onFill() { 
      m_currFilled = !m_currFilled; 
      setModifiedFlag(true); 
    } 
```

在图形菜单可见之前，会调用`isLineChecked`、`isRectangleChecked`和`isEllipseChecked`方法。如果它们返回`true`，则如果下一个要添加的图形是所涉及的图形，则项目会通过单选按钮被选中：

```cpp
    bool DrawingWidget::isLineChecked() { 
      return (m_addFigureId == LineId); 
    } 

    bool DrawingWidget::isRectangleChecked() { 
      return (m_addFigureId == RectangleId); 
    } 

    bool DrawingWidget::isEllipseChecked() { 
      return (m_addFigureId == EllipseId); 
    } 
```

当用户选择图形菜单中的项目时，会调用`onLine`、`onRectangle`和`onEllipse`方法。它们将用户将要添加的下一个图形设置为所涉及的图形：

```cpp
    void DrawingWidget::onLine() { 
      m_addFigureId = LineId; 
    } 

    void DrawingWidget::onRectangle() { 
      m_addFigureId = RectangleId; 
    } 

    void DrawingWidget::onEllipse() { 
      m_addFigureId = EllipseId; 
    } 
```

每次用户按下鼠标键时，都会调用`mousePressEvent`方法。首先，我们需要检查他们是否按下了左鼠标键：

```cpp
    void DrawingWidget::mousePressEvent(QMouseEvent* eventPtr) { 
      if (eventPtr->buttons() == Qt::LeftButton) { 
```

在以下代码片段中对`mouseMoveEvent`的调用中，我们需要跟踪最新的鼠标点，以便计算鼠标移动之间的距离。因此，我们将`m_mousePoint`设置为鼠标点：

```cpp
        m_mousePoint = eventPtr->pos(); 
```

我们遍历图形指针列表，并对每个图形，我们通过调用`isClick`来检查图形是否被鼠标点击。我们需要以相当尴尬的方式反向迭代，以便首先找到最顶部的图形。我们使用`reverse_iterator`类和`rbegin`和`rend`方法来反向迭代：

```cpp
        for (QList<Figure*>::reverse_iterator iterator = 
             m_figurePtrList.rbegin(); 
             iterator != m_figurePtrList.rend(); ++iterator) {
```

我们使用解引用运算符（`*`）来获取列表中的图形指针：

```cpp
              Figure* figurePtr = *iterator; 
```

如果图形被鼠标点击，我们将应用程序模式设置为移动。我们还通过在列表上调用`removeOne`和`push_back`来将图形放置在列表的末尾，使其看起来是绘图的顶层。最后，我们中断循环，因为我们已经找到了我们正在寻找的图形：

```cpp
          if (figurePtr->isClick(m_mousePoint)) { 
            setApplicationMode(Move); 
            m_figurePtrList.removeOne(figurePtr); 
            m_figurePtrList.push_back(figurePtr); 
            break; 
          } 
        } 
```

如果应用程序模式仍然是空闲状态（没有移动），我们没有找到被鼠标点击的图形。在这种情况下，我们将应用程序模式设置为创建，并调用`createFigure`来找到一个要复制的图形。然后，我们设置图形的颜色和填充状态以及图形的点。最后，通过调用`push_back`（为了使其出现在绘图的顶部，将其添加到列表的末尾）并将修改标志设置为`true`，因为绘图已被修改：

```cpp
        if (m_applicationMode == Idle) { 
          setApplicationMode(Create); 
          Figure* newFigurePtr = createFigure(m_addFigureId); 
          newFigurePtr->color() = m_currColor; 
          newFigurePtr->filled() = m_currFilled; 
          newFigurePtr->initializePoints(m_mousePoint); 
          m_figurePtrList.push_back(newFigurePtr); 
          setModifiedFlag(true); 
        } 
      } 
    } 
```

每当用户移动鼠标时，都会调用`mouseMoveEvent`。首先，我们需要检查用户在移动鼠标时是否按下了鼠标左键：

```cpp
    void DrawingWidget::mouseMoveEvent(QMouseEvent* eventPtr) { 
      if (eventPtr->buttons() == Qt::LeftButton) { 
        QPoint newMousePoint = eventPtr->pos();
```

然后，我们检查应用程序模式。如果我们正在将新图形添加到绘图的过程中，我们修改其最后一个点：

```cpp
        switch (m_applicationMode) { 
          case Create: 
            m_figurePtrList.back()->setLastPoint(m_mousePoint); 
            break; 
```

如果我们在移动一个图形的过程中，我们将计算自上次鼠标事件以来的距离，并将位于图形指针列表末尾的图形移动。记住，被鼠标点击的图形是在前一个`mousePressEvent`中放置在图形指针列表末尾的：

```cpp
          case Move: { 
              QSize distance(newMousePoint.x() - m_mousePoint.x(), 
                             newMousePoint.y() - m_mousePoint.y()); 
              m_figurePtrList.back()->move(distance); 
              setModifiedFlag(true); 
            } 
            break; 
        } 
```

最后，我们更新当前鼠标点，以便下一次调用`mouseMoveEvent`。我们还调用更新方法来强制窗口重新绘制：

```cpp
        m_mousePoint = newMousePoint; 
        update(); 
      } 
    } 
```

当用户释放鼠标按钮之一时，会调用`mouseReleaseEvent`方法。我们将应用程序模式设置为空闲：

```cpp
    void DrawingWidget::mouseReleaseEvent(QMouseEvent* eventPtr) { 
      if (eventPtr->buttons() == Qt::LeftButton) { 
        setApplicationMode(Idle); 
      } 
    } 
```

每当窗口需要重新绘制时，都会调用`paintEvent`方法。这可能是由于几个原因。例如，用户可能已经改变了窗口的大小。这也可能是由于在 Qt 的`QWidget`类中调用`update`的结果，这强制窗口重新绘制，并最终调用`paintEvent`。

我们首先创建一个`QPainter`对象，这可以被视为绘画的画布，并设置合适的渲染。然后，我们遍历图形指针列表，并绘制每个图形。这样，列表中的最后一个图形就会绘制在绘图的顶部：

```cpp
    void DrawingWidget::paintEvent(QPaintEvent* /* eventPtr */) { 
      QPainter painter(this); 
      painter.setRenderHint(QPainter::Antialiasing); 
      painter.setRenderHint(QPainter::TextAntialiasing); 

      for (Figure* figurePtr : m_figurePtrList) { 
        figurePtr->draw(painter); 
      } 
    } 
```

# 主函数

最后，我们在 `main` 函数中通过创建应用程序对象、显示主窗口并执行应用程序来启动应用程序。

**Main.cpp:**

```cpp
    #include "DrawingWindow.h" 
    #include <QApplication> 

    int main(int argc, char *argv[]) { 
      QApplication application(argc, argv); 
      DrawingWindow drawingWindow; 
      drawingWindow.show(); 
      return application.exec(); 
    }
```

接收到的输出如下：

![](img/25b6b2a9-d07d-4e9e-b8c8-28b2475353a8.png)

# 构建编辑器

下一个应用程序是一个编辑器，用户可以在其中输入和编辑文本。当前输入位置由光标指示。可以使用箭头键和鼠标点击来移动光标。

# 光标类

`Caret` 类处理光标；即标记下一个要输入的字符位置的闪烁垂直线。

**Caret.h:**

```cpp
    #ifndef CARET_H 
    #define CARET_H 

    #include <QObject> 
    #include <QWidget> 
    #include <QTimer> 

    class Caret : public QObject { 
      Q_OBJECT 

      public: 
        Caret(QWidget* parentWidgetPtr = nullptr);

```

`show` 和 `hide` 方法用于显示和隐藏光标。在本应用中，光标永远不会被隐藏。然而，在下一章的高级版本中，在某些情况下光标将被隐藏：

```cpp
        void show(); 
        void hide(); 
```

`set` 方法设置光标的当前尺寸和位置，而 `paint` 方法将其绘制在 `QPainter` 对象上：

```cpp
        void set(QRect rect); 
        void paint(QPainter& painter); 
```

每次光标闪烁时都会调用 `onTimer` 方法：

```cpp
      public slots: 
        void onTimer(void); 

      private: 
        QWidget* m_parentWidgetPtr; 
```

当光标可见时，`m_visible` 字段为真：

```cpp
        bool m_visible, m_blink; 
```

`m_rect` 字段处理使光标闪烁的计时器：

```cpp
        QRect m_rect; 
```

`m_timer` 字段处理使光标闪烁的计时器：

```cpp
        QTimer m_timer; 
    }; 

    #endif // CARET_H 
```

`Caret.cpp` 文件包含 `Caret` 类方法的定义。

**Caret.cpp:**

```cpp
    #include "Caret.h" 
    #include <QPainter>
```

构造函数将计时器信号连接到 `onTimer`，结果为每次超时都会调用 `onTimer`。然后计时器初始化为 `500` 毫秒。也就是说，`onTimer` 将每 `500` 毫秒被调用一次，光标每 `500` 毫秒显示和隐藏：

```cpp
    Caret::Caret(QWidget* parentWidgetPtr) 
      :m_parentWidgetPtr(parentWidgetPtr) { 
      m_timer.setParent(this); 
      connect(&m_timer, SIGNAL(timeout()), this, SLOT(onTimer())); 
      m_timer.start(500); 
    } 
```

`show` 和 `hide` 方法设置 `m_visible` 字段并通过在父窗口上调用 `update` 强制重绘光标区域：

```cpp
    void Caret::show() { 
      m_visible = true; 
      m_parentWidgetPtr->update(m_rect); 
    } 

    void Caret::hide() { 
      m_visible = false; 
      m_parentWidgetPtr->update(m_rect); 
    } 
```

`set` 方法设置光标的尺寸和位置。然而，光标的宽度始终设置为 1，这使得它看起来像一条细长的垂直线：

```cpp
    void Caret::set(QRect rect) { 
      m_rect = rect; 
      m_rect.setWidth(1); 
      m_parentWidgetPtr->update(m_rect); 
    } 
```

`onTimer` 方法每 500 毫秒被调用一次。它反转 `m_blink` 并强制重绘光标。这导致光标以一秒的间隔闪烁：

```cpp
    void Caret::onTimer(void) { 
      m_blink = !m_blink; 
      m_parentWidgetPtr->update(m_rect); 
    }
```

每次需要重绘光标时都会调用 `paint` 方法。如果 `m_visible` 和 `m_blink` 都为真，则绘制光标；如果光标被设置为可见且光标正在闪烁，即光标在闪烁间隔内可见，则它们为真。在调用 `paint` 之前清除光标区域，以便如果没有发生绘制，则清除光标：

```cpp
    void Caret::paint(QPainter& painter) { 
      if (m_visible && m_blink) { 
        painter.save(); 
        painter.setPen(Qt::NoPen); 
        painter.setBrush(Qt::black); 
        painter.drawRect(m_rect); 
        painter.restore(); 
      } 
    } 
```

# 绘制编辑器窗口

`EditorWindow` 是上一节中 `MainWindow` 的子类。它处理窗口的关闭操作。此外，它还处理按键事件。

**EditorWindow.h:**

```cpp
    #ifndef EDITORWINDOW_H 
    #define EDITORWINDOW_H 

    #include <QMainWindow> 
    #include <QActionGroup> 
    #include <QPair> 
    #include <QMap> 

    #include "..\MainWindow\MainWindow.h" 
    #include "EditorWidget.h" 

    class EditorWindow : public MainWindow { 
      Q_OBJECT 

      public: 
        EditorWindow(QWidget* parentWidgetPtr = nullptr); 
        ~EditorWindow(); 
```

每次用户按下键时都会调用 `keyPressEvent` 方法，当用户尝试关闭窗口时调用 `closeEvent`：

```cpp
      protected: 
        void keyPressEvent(QKeyEvent* eventPtr); 
        void closeEvent(QCloseEvent* eventPtr); 

      private: 
        EditorWidget* m_editorWidgetPtr; 
    }; 

    #endif // EDITORWINDOW_H 
```

`EditorWindow` 类实际上相当小。它只定义了构造函数和析构函数，以及 `keyPressEvent` 和 `closePressEvent` 方法。

**EditorWindow.cpp:**

```cpp
    #include "EditorWindow.h" 
    #include <QtWidgets> 
```

构造函数将窗口大小设置为 `1000` * `500` 像素，并将标准文件菜单添加到菜单栏：

```cpp
    EditorWindow::EditorWindow(QWidget* parentWidgetPtr /*= nullptr*/) 
     :MainWindow(parentWidgetPtr) { 
      resize(1000, 500); 
      m_editorWidgetPtr = new EditorWidget(this); 
      setCentralWidget(m_editorWidgetPtr); 
      addFileMenu(); 
    } 

    EditorWindow::~EditorWindow() { 
      // Empty. 
    } 
```

`keyPressEvent` 和 `closeEvent` 方法只是将消息传递给编辑器小部件中的对应方法，该小部件位于窗口中心：

```cpp
    void EditorWindow::keyPressEvent(QKeyEvent* eventPtr) { 
      m_editorWidgetPtr->keyPressEvent(eventPtr); 
    } 

    void EditorWindow::closeEvent(QCloseEvent* eventPtr) { 
      m_editorWidgetPtr->closeEvent(eventPtr); 
    }
```

# 绘制编辑器小部件

`EditorWidget` 类是上一节中 `DocumentWidget` 的子类。它捕获键、鼠标、调整大小和关闭事件。它还重写了保存和加载文档的方法。

**EditorWidget.h:** 

```cpp
    #ifndef EDITORWIDGET_H 
    #define EDITORWIDGET_H 

    #include <QWidget> 
    #include <QMap> 
    #include <QMenu> 
    #include <QToolBar> 
    #include <QPair> 
    #include "Caret.h" 

    #include "..\MainWindow\DocumentWidget.h" 

    class EditorWidget : public DocumentWidget { 
      Q_OBJECT 

      public: 
        EditorWidget(QWidget* parentWidgetPtr); 
```

当用户按下键时调用 `keyPressEvent`，当用户用鼠标点击时调用 `mousePressEvent`：

```cpp
        void keyPressEvent(QKeyEvent* eventPtr); 
        void mousePressEvent(QMouseEvent* eventPtr); 
```

`mouseToIndex` 方法是一个辅助方法，用于计算用户用鼠标点击的字符的索引：

```cpp
      private: 
        int mouseToIndex(QPoint point); 
```

当窗口需要重绘时调用 `paintEvent` 方法，当用户调整窗口大小时调用 `resizeEvent`。我们在此应用程序中捕获调整大小事件，因为我们想要重新计算每行可以容纳的字符数：

```cpp
      public: 
        void paintEvent(QPaintEvent* eventPtr); 
        void resizeEvent(QResizeEvent* eventPtr);
```

与上一节中的绘图程序类似，当用户选择“新建”菜单项时调用 `newDocument`，当用户选择“保存”或“另存为”项时调用 `writeFile`，当用户选择“打开”项时调用 `readFile`：

```cpp
      private: 
        void newDocument(void); 
        bool writeFile(const QString& filePath); 
        bool readFile(const QString& filePath); 
```

调用 `setCaret` 方法以响应用户输入或鼠标点击来设置光标：

```cpp
      private: 
        void setCaret(); 
```

当用户移动光标上下时，我们需要找到光标上方或下方的字符索引。完成此操作的最简单方法是模拟鼠标点击：

```cpp
        void simulateMouseClick(int x, int y); 
```

`calculate` 方法是一个辅助方法，用于计算行数以及每行中每个字符的位置：

```cpp
      private: 
        void calculate(); 
```

`m_editIndex` 字段持有用户输入文本的位置索引。该位置也是光标可见的位置：

```cpp
        int m_editIndex = 0; 
```

`m_caret` 字段持有应用程序的光标：

```cpp
        Caret m_caret; 
```

编辑器的文本存储在 `m_editorText` 中：

```cpp
        QString m_editorText; 
```

编辑器的文本可能分布在多行；`m_lineList` 跟踪每行的第一个和最后一个索引：

```cpp
        QList<QPair<int,int>> m_lineList; 
```

之前的 `calculate` 方法计算编辑器文本中每个字符的矩形，并将它们放置在 `m_rectList` 中：

```cpp
        QList<QRect> m_rectList;
```

在本章的应用中，所有字符都使用相同的字体，该字体存储在 `TextFont` 中：

```cpp
        static const QFont TextFont; 
```

`FontWidth` 和 `FontHeight` 持有 `TextFont` 中字符的宽度和高度：

```cpp
         int FontWidth, FontHeight; 
    }; 

    #endif // EDITORWIDGET_H 
```

`EditorWidget` 类相当大。它定义了编辑器的功能。

**EditorWidget.cpp:** 

```cpp
    #include "EditorWidget.h" 
    #include <QtWidgets> 
    using namespace std; 
```

我们将文本字体初始化为 12 点的 `Courier New`：

```cpp
    const QFont EditorWidget::TextFont("Courier New", 12); 
```

构造函数将标题设置为 `Editor`，并将标准加载和保存对话框的文件后缀设置为 `edi`。使用 Qt `QMetrics` 类设置文本字体中字符的高度和平均宽度（以像素为单位）。计算每个字符的矩形，并将光标设置为文本中的第一个字符：

```cpp
    EditorWidget::EditorWidget(QWidget* parentWidgetPtr) 
     :DocumentWidget(tr("Editor"), tr("Editor files (*.edi)"), 
                     parentWidgetPtr), 
      m_caret(this), 
      m_editorText(tr("Hello World")) { 
      QFontMetrics metrics(TextFont);
      FontHeight = metrics.height();
      FontWidth = metrics.averageCharWidth();
      calculate(); 
      setCaret(); 
      m_caret.show(); 
    } 
```

当用户选择新菜单项时，会调用`newDocument`方法。它会清除文本，设置光标，并重新计算字符矩形：

```cpp
    void EditorWidget::newDocument(void) { 
      m_editIndex = 0; 
      m_editorText.clear(); 
      calculate(); 
      setCaret(); 
    } 
```

当用户选择保存或另存为菜单项时，会调用`writeFile`方法。它简单地写入编辑器的当前文本：

```cpp
    bool EditorWidget::writeFile(const QString& filePath) { 
      QFile file(filePath); 
      if (file.open(QIODevice::WriteOnly | QIODevice::Text)) { 
        QTextStream outStream(&file); 
        outStream << m_editorText; 
```

我们使用输入流的`Ok`字段来决定写入是否成功：

```cpp
        return ((bool) outStream.Ok); 
      } 
```

如果无法打开文件进行写入，则返回`false`：

```cpp
      return false; 
    } 
```

当用户选择加载菜单项时，会调用`readFile`方法。它通过在输入流上调用`readAll`来读取编辑器的所有文本：

```cpp
    bool EditorWidget::readFile(const QString& filePath) { 
      QFile file(filePath); 

      if (file.open(QIODevice::ReadOnly | QIODevice::Text)) { 
        QTextStream inStream(&file); 
        m_editorText = inStream.readAll(); 
```

当文本被读取后，计算字符矩形，并设置光标：

```cpp
        calculate(); 
        setCaret(); 
```

我们使用输入流的`Ok`字段来决定读取是否成功：

```cpp
        return ((bool) inStream.Ok); 
      } 
```

如果无法打开文件进行读取，则返回`false`：

```cpp
      return false; 
    }
```

当用户按下鼠标按钮之一时，会调用`mousePressEvent`。如果用户按下左键，我们调用`mouseToIndex`来计算点击的字符索引，并将光标设置到该索引：

```cpp
    void EditorWidget::mousePressEvent(QMouseEvent* eventPtr) { 
      if (eventPtr->buttons() == Qt::LeftButton) { 
        m_editIndex = mouseToIndex(eventPtr->pos()); 
        setCaret(); 
      } 
    } 
```

当用户按下键时，会调用`keyPressEvent`。首先，我们检查它是否是箭头键、删除键、退格键或回车键。如果不是，我们将在光标指示的位置插入字符：

```cpp
    void EditorWidget::keyPressEvent(QKeyEvent* eventPtr) { 
      switch (eventPtr->key()) { 
```

如果键是向左箭头键，并且如果编辑光标尚未位于文本开头，我们减少编辑索引：

```cpp
        case Qt::Key_Left: 
          if (m_editIndex > 0) { 
            --m_editIndex; 
          } 
          break; 
```

如果键是向右箭头键，并且如果编辑光标尚未位于文本末尾，我们增加编辑索引：

```cpp
        case Qt::Key_Right: 
          if (m_editIndex < m_editorText.size()) { 
            ++m_editIndex; 
          } 
          break; 
```

如果键是向上箭头键，并且如果编辑光标尚未位于编辑器的顶部，我们调用`similateMouseClick`来模拟用户在当前索引稍上方点击鼠标。这样，新的编辑索引将位于当前行的上方行：

```cpp
        case Qt::Key_Up: { 
            QRect charRect = m_rectList[m_editIndex]; 

            if (charRect.top() > 0) { 
              int x = charRect.left() + (charRect.width() / 2), 
                  y = charRect.top() - 1; 
              simulateMouseClick(x, y); 
            } 
          } 
          break; 
```

如果键是向下箭头键，我们调用`similateMouseClick`来模拟用户在当前索引稍下方点击鼠标。这样，编辑光标将位于当前字符直接下方的字符。注意，如果索引已经在底部行，则不会发生任何操作：

```cpp
        case Qt::Key_Down: { 
            QRect charRect = m_rectList[m_editIndex]; 
            int x = charRect.left() + (charRect.width() / 2), 
                y = charRect.bottom() + 1; 
            simulateMouseClick(x, y); 
          } 
          break; 
```

如果用户按下删除键，并且编辑索引尚未超出文本末尾，则移除当前字符：

```cpp
        case Qt::Key_Delete: 
          if (m_editIndex < m_editorText.size()) { 
            m_editorText.remove(m_editIndex, 1); 
            setModifiedFlag(true); 
          } 
          break; 
```

如果用户按下退格键，并且编辑索引尚未位于文本开头，则移除当前字符之前的字符：

```cpp
        case Qt::Key_Backspace: 
          if (m_editIndex > 0) { 
            m_editorText.remove(--m_editIndex, 1); 
            setModifiedFlag(true); 
          } 
          break; 
```

如果用户按下回车键，则插入换行字符（`n`）：

```cpp
        case Qt::Key_Return: 
          m_editorText.insert(m_editIndex++, 'n'); 
          setModifiedFlag(true); 
          break;
```

如果用户按下可读字符，它由`text`方法提供，我们将它的第一个字符插入到编辑索引处：

```cpp
        default: { 
            QString text = eventPtr->text(); 

            if (!text.isEmpty()) { 
              m_editorText.insert(m_editIndex++, text[0]); 
              setModifiedFlag(true); 
            } 
          } 
          break; 
      }  
```

当文本被修改后，我们需要计算字符矩形，设置光标，并通过调用`update`强制重绘：

```cpp
      calculate(); 
      setCaret(); 
      update(); 
    } 
```

`similateMouseClick`方法通过调用`mousePressEvent`和`mousePressRelease`以及给定的点来模拟鼠标点击：

```cpp
    void EditorWidget::simulateMouseClick(int x, int y) { 
      QMouseEvent pressEvent(QEvent::MouseButtonPress, QPointF(x, y), 
                       Qt::LeftButton, Qt::NoButton, Qt::NoModifier); 
      mousePressEvent(&pressEvent); 
      QMouseEvent releaseEvent(QEvent::MouseButtonRelease, 
                               QPointF(x, y), Qt::LeftButton, 
                               Qt::NoButton, Qt::NoModifier); 
      mousePressEvent(&releaseEvent); 
    } 
```

`setCaret` 方法创建一个包含光标大小和位置的矩形，然后隐藏、设置并显示光标：

```cpp
    void EditorWidget::setCaret() { 
      QRect charRect = m_rectList[m_editIndex]; 
      QRect caretRect(charRect.left(), charRect.top(), 
                      1, charRect.height()); 
      m_caret.hide(); 
      m_caret.set(caretRect); 
      m_caret.show(); 
    }
```

`mouseToIndex` 方法计算给定鼠标点的编辑索引：

```cpp
    int EditorWidget::mouseToIndex(QPoint mousePoint) { 
      int x = mousePoint.x(), y = mousePoint.y(); 
```

首先，我们将 `y` 坐标设置为文本，以防它在文本下方：

```cpp
      if (y > (FontHeight * m_lineList.size())) { 
        y = ((FontHeight * m_lineList.size()) - 1); 
      } 
```

我们计算鼠标点的行：

```cpp
      int lineIndex = y / FontHeight; 
      QPair<int,int> lineInfo = m_lineList[lineIndex]; 
      int firstIndex = lineInfo.first, lastIndex = lineInfo.second; 
```

我们在该行找到索引：

```cpp
      if (x > ((lastIndex - firstIndex + 1) * FontWidth)) { 
        return (lineIndex == (m_lineList.size() - 1)) 
               ? (lineInfo.second + 1) : lineInfo.second; 
      } 
      else { 
        return firstIndex + (x / FontWidth); 
      } 

      return 0; 
    } 
```

当用户更改窗口大小时，会调用 `resizeEvent` 方法。由于线条可能变短或变长，因此会重新计算字符矩形：

```cpp
    void EditorWidget::resizeEvent(QResizeEvent* eventPtr) { 
      calculate(); 
      DocumentWidget::resizeEvent(eventPtr); 
    } 
```

每当文本发生变化或窗口大小发生变化时，都会调用 `calculate` 方法。它会遍历文本并为每个字符计算矩形：

```cpp
    void EditorWidget::calculate() { 
      m_lineList.clear(); 
      m_rectList.clear(); 
      int windowWidth = width();
```

首先，我们需要将文本分成行。每行继续直到它不适合窗口，直到我们达到一个新行，或者直到文本结束：

```cpp
      { int firstIndex = 0, lineWidth = 0; 
        for (int charIndex = 0; charIndex < m_editorText.size(); 
             ++charIndex) { 
          QChar c = m_editorText[charIndex]; 

          if (c == 'n') { 
            m_lineList.push_back 
                       (QPair<int,int>(firstIndex, charIndex)); 
            firstIndex = charIndex + 1; 
            lineWidth = 0; 
          } 
          else { 
            if ((lineWidth + FontWidth) > windowWidth) { 
              if (firstIndex == charIndex) { 
                m_lineList.push_back 
                           (QPair<int,int>(firstIndex, charIndex)); 
                firstIndex = charIndex + 1; 
              } 
              else { 
                m_lineList.push_back(QPair<int,int>(firstIndex, 
                                                    charIndex - 1)); 
                firstIndex = charIndex; 
              } 

              lineWidth = 0; 
            } 
            else { 
              lineWidth += FontWidth; 
            } 
          } 
        } 

        m_lineList.push_back(QPair<int,int>(firstIndex, 
                                            m_editorText.size() - 1)); 
      } 
```

然后，我们遍历这些行，并对每一行计算每个字符的矩形：

```cpp
      { int top = 0; 
        for (int lineIndex = 0; lineIndex < m_lineList.size(); 
             ++lineIndex) { 
          QPair<int,int> lineInfo = m_lineList[lineIndex]; 
          int firstIndex = lineInfo.first, 
              lastIndex = lineInfo.second, left = 0; 

          for (int charIndex = firstIndex; 
               charIndex <= lastIndex; ++charIndex){ 
            QRect charRect(left, top, FontWidth, FontHeight); 
            m_rectList.push_back(charRect); 
            left += FontWidth; 
          } 

          if (lastIndex == (m_editorText.size() - 1)) { 
            QRect lastRect(left, top, 1, FontHeight); 
            m_rectList.push_back(lastRect); 
          } 

          top += FontHeight; 
        } 
      } 
    } 
```

当窗口需要重绘时，会调用 `paintEvent` 方法：

```cpp
    void EditorWidget::paintEvent(QPaintEvent* /*eventPtr*/) { 
      QPainter painter(this); 
      painter.setRenderHint(QPainter::Antialiasing); 
      painter.setRenderHint(QPainter::TextAntialiasing); 
      painter.setFont(TextFont); 
      painter.setPen(Qt::black); 
      painter.setBrush(Qt::white); 
```

我们遍历编辑器的文本，并对每个字符（除了换行符）在其适当的位置写入：

```cpp
      for (int index = 0; index < m_editorText.length(); ++index) { 
        QChar c = m_editorText[index]; 

        if (c != 'n') { 
          QRect rect = m_rectList[index]; 
          painter.drawText(rect, c); 
        } 
      } 

      m_caret.paint(painter); 
    }
```

# `main` 函数

最后，`main` 函数的工作方式与本章之前的应用类似——我们创建一个应用程序，创建一个编辑窗口，并执行该应用程序。

**Main.cpp:**

```cpp
#include "EditorWindow.h" 
#include <QApplication> 

int main(int argc, char *argv[]) { 
  QApplication application(argc, argv); 
  EditorWindow editorWindow; 
  editorWindow.show(); 
  return application.exec(); 
} 
```

得到以下输出：

![图片](img/a562a9ab-9bc0-4cbc-99f8-d3d7382a9605.png)

# 概述

在本章中，我们使用 Qt 库开发了三个图形应用程序——一个模拟时钟、一个绘图程序和一个编辑器。时钟显示当前的小时、分钟和秒。在绘图程序中，我们可以绘制线条、矩形和椭圆，在编辑器中，我们可以输入和编辑文本。

在下一章中，我们将继续与这些应用程序一起工作，并开发更高级的版本。
