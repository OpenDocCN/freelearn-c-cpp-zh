# 第十一章：实现图形编辑器

Qt 为我们提供了使用`QPainter`类进行低级图形渲染的功能。Qt 能够渲染位图和矢量图像。在本章中，我们将学习如何使用 Qt 绘制形状，并最终创建我们自己的绘图程序。

在本章中，我们将涵盖以下主题：

+   绘制矢量形状

+   将矢量图像保存为 SVG 文件

+   创建绘图程序

准备好了吗？让我们开始吧！

# 绘制矢量形状

在接下来的部分，我们将学习如何在我们的 Qt 应用程序中使用 QPainter 类渲染矢量图形。

# 矢量与位图

计算机图形中有两种格式——位图和矢量。位图图像（也称为光栅图像）是以一系列称为**像素**的微小点存储的图像。每个像素将被分配一种颜色，并且以存储的方式显示在屏幕上——像素与屏幕上显示的内容之间是一一对应的关系。

另一方面，矢量图像不是基于位图模式，而是使用数学公式来表示可以组合成几何形状的线条和曲线。

这里列出了两种格式的主要特点：

+   位图：

+   通常文件大小较大

+   不能放大到更高分辨率，因为图像质量会受到影响

+   用于显示颜色丰富的复杂图像，如照片

+   矢量：

+   文件大小非常小

+   图形可以调整大小而不影响图像质量

+   每个形状只能应用有限数量的颜色（单色、渐变或图案）

+   复杂形状需要高处理能力才能生成

这里的图表比较了位图和矢量图形：

![](img/94527953-3456-480e-92b6-2303f304d7c4.png)

在本节中，我们将专注于学习如何使用 Qt 绘制矢量图形，但我们也将在本章后面介绍位图图形。

# 使用 QPainter 绘制矢量形状

首先，通过转到文件|新建文件或项目来创建另一个 Qt 项目。然后在应用程序类别下选择 Qt Widget 应用程序。创建项目后，打开`mainwindow.h`并添加`QPainter`头文件：

```cpp
#include <QMainWindow> 
#include <QPainter> 
```

之后，我们还声明了一个名为`paintEvent()`的虚函数，这是 Qt 中的标准事件处理程序，每当需要绘制东西时都会调用它，无论是 GUI 更新、窗口调整大小，还是手动调用`update()`函数时：

```cpp
public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    virtual void paintEvent(QPaintEvent *event); 
```

然后，打开`mainwindow.cpp`并添加`paintEvent()`函数：

```cpp
void MainWindow::paintEvent(QPaintEvent *event) 
{ 
   QPainter painter; 
   painter.begin(this); 

   // Draw Line 
   painter.drawLine(QPoint(50, 60), QPoint(100, 100)); 

   // Draw Rectangle 
   painter.setBrush(Qt::BDiagPattern); 
   painter.drawRect(QRect(40, 120, 80, 30)); 

   // Draw Ellipse 
   QPen ellipsePen; 
   ellipsePen.setColor(Qt::red); 
   ellipsePen.setStyle(Qt::DashDotLine); 
   painter.setPen(ellipsePen); 
   painter.drawEllipse(QPoint(80, 200), 50, 20); 

   // Draw Rectangle 
   QPainterPath rectPath; 
   rectPath.addRect(QRect(150, 20, 100, 50)); 
   painter.setPen(QPen(Qt::red, 1, Qt::DashDotLine, Qt::FlatCap, 
   Qt::MiterJoin)); 
   painter.setBrush(Qt::yellow); 
   painter.drawPath(rectPath); 

   // Draw Ellipse 
   QPainterPath ellipsePath; 
   ellipsePath.addEllipse(QPoint(200, 120), 50, 20); 
   painter.setPen(QPen(QColor(79, 106, 25), 5, Qt::SolidLine, 
   Qt::FlatCap, Qt::MiterJoin)); 
   painter.setBrush(QColor(122, 163, 39)); 
   painter.drawPath(ellipsePath); 

   painter.end(); 
} 
```

如果现在构建程序，你应该会看到以下内容：

![](img/24aed423-bb6e-4adc-b33f-3804ca1972c2.png)

上面的代码真的很长。让我们把它分解一下，这样你就更容易理解了。每当调用`paintEvent()`时（通常在窗口需要绘制时会调用一次），我们调用`QPainter::begin()`告诉 Qt 我们要开始绘制东西了，然后在完成时调用`QPainter::end()`。因此，绘制图形的代码将包含在`QPainter::begin()`和`QPainter::end()`之间。

让我们看看以下步骤：

1.  我们绘制的第一件事是一条直线，这很简单——只需调用`QPainter::drawLine()`并将起点和终点值插入函数中。请注意，Qt 使用的坐标系统是以像素格式的。它的原点从应用程序窗口的左上角开始，并向右和向下方向增加，取决于*x*和*y*的值。*x*值的增加将位置移动到右方向，而*y*值的增加将位置移动到下方向。

1.  接下来，绘制一个矩形，在形状内部有一种阴影图案。这次，我们调用了`QPainter::setBrush()`来设置图案，然后调用`drawRect()`。

1.  之后，我们用虚线轮廓和图案在形状内部绘制了一个椭圆形。由于我们已经在上一步中设置了图案，所以我们不必再次设置。相反，我们使用`QPen`类在调用`drawEllipse()`之前设置轮廓样式。只需记住，在 Qt 的术语中，刷子用于定义形状的内部颜色或图案，而笔用于定义轮廓。

1.  接下来的两个形状基本上与之前的相似；我们只是改变了不同的颜色和图案，这样你就可以看到它们与之前的例子之间的区别。

# 绘制文本

此外，您还可以使用`QPainter`类来绘制文本。在调用`QPainter::drawText()`之前，您只需要调用`QPainter::setFont()`来设置字体属性，就像这样：

```cpp
QPainter painter; 
painter.begin(this); 

// Draw Text 
painter.setFont(QFont("Times", 14, QFont::Bold)); 
painter.drawText(QPoint(20, 30), "Testing"); 

// Draw Line 
painter.drawLine(QPoint(50, 60), QPoint(100, 100)) 
```

`setFont()`函数是可选的，如果您不指定它，将获得默认字体。完成后，构建并运行程序。您应该在窗口中看到“Hello World！”这个词显示出来：

![](img/69667eda-dc36-4753-be48-1c8ac3a0143f.png)

在这里你可以看到，矢量形状基本上是由 Qt 实时生成的，无论你如何重新调整窗口大小和改变它的纵横比，它看起来都很好。如果你渲染的是位图图像，当它与窗口一起重新调整大小或改变纵横比时，它的视觉质量可能会下降。

# 将矢量图像保存到 SVG 文件

除了绘制矢量图形，Qt 还允许我们将这些图形保存为矢量图像文件，称为**SVG**（可缩放矢量图形）文件格式。SVG 格式是许多软件使用的开放格式，包括 Web 浏览器用于显示矢量图形。实际上，Qt 也可以读取 SVG 文件并在屏幕上呈现它们，但我们暂时跳过这一点。让我们看看如何将我们的矢量图形保存为 SVG 文件！

这个例子继续了我们在上一节中留下的地方。因此，我们不必创建一个新的 Qt 项目，只需坚持之前的项目即可。

首先，如果主窗口还没有菜单栏，让我们为主窗口添加一个菜单栏。然后，打开`mainwindow.ui`，在表单编辑器中，右键单击层次结构窗口上的 MainWindow 对象，然后选择创建菜单栏：

![](img/6041e5ce-79df-4fd0-8b7f-0308f37da1b9.png)

完成后，将文件添加到菜单栏，然后在其下方添加“另存为 SVG”：

![](img/22dbdd71-1359-46bb-8a7e-6537dc52034e.png)

然后，转到底部的操作编辑器，右键单击我们刚刚添加的菜单选项，并选择转到槽...：

![](img/e6fe895f-60a5-4fd8-9328-d937ea068f9a.png)

将弹出一个窗口询问您选择一个信号。选择`triggered()`，然后点击确定。这样就会在`mainwindow.cpp`中为您创建一个新的槽函数。在打开`mainwindow.cpp`之前，让我们打开我们的`项目文件`（`.pro`）并添加以下`svg`模块：

```cpp
QT += core gui svg 
```

`svg`关键字告诉 Qt 向您的项目添加相关类，可以帮助您处理 SVG 文件格式。然后，我们还需要在`mainwindow.h`中添加另外两个头文件：

```cpp
#include <QtSvg/QSvgGenerator> 
#include <QFileDialog> 
```

之后，打开`mainwindow.cpp`并将以下代码添加到我们刚刚在上一步中添加的槽函数中：

```cpp
void MainWindow::on_actionSave_as_SVG_triggered() 
{ 
    QString filePath = QFileDialog::getSaveFileName(this, "Save SVG", "", "SVG files (*.svg)"); 

    if (filePath == "") 
        return; 

    QSvgGenerator generator; 
    generator.setFileName(filePath); 
    generator.setSize(QSize(this->width(), this->height())); 
    generator.setViewBox(QRect(0, 0, this->width(), this->height())); 
    generator.setTitle("SVG Example"); 
    generator.setDescription("This SVG file is generated by Qt."); 

    paintAll(&generator); 
} 
```

在前面的代码中，我们使用`QFileDialog`让用户选择他们想要保存 SVG 文件的位置。然后，我们使用`QSvgGenerator`类将图形导出到 SVG 文件中。最后，我们调用`paintAll()`函数，这是我们将在下一步中定义的自定义函数。

实际上，我们需要修改现有的`paintAll()`方法并将我们的渲染代码放入其中。然后，将`QSvgGenerator`对象作为绘制设备传递到函数输入中：

```cpp
void MainWindow::paintAll(QSvgGenerator *generator) 
{ 
    QPainter painter; 

    if (generator) 
        painter.begin(generator); 
    else 
        painter.begin(this); 

   // Draw Text 
    painter.setFont(QFont("Times", 14, QFont::Bold)); 
   painter.drawText(QPoint(20, 30), "Hello World!"); 
```

因此，我们的`paintEvent()`现在在`mainwindow.cpp`中看起来像这样：

```cpp
void MainWindow::paintEvent(QPaintEvent *event) 
{ 
   paintAll(); 
} 
```

这里的过程可能看起来有点混乱，但它的基本作用是在创建窗口时调用`paintAll()`函数一次绘制所有图形，然后当您想要将图形保存到 SVG 文件时再次调用`paintAll()`。

唯一的区别是绘图设备——一个是主窗口本身，我们将其用作绘图画布，对于后者，我们将`QSvgGenerator`对象传递为绘图设备，它将把图形保存到 SVG 文件中。

现在构建并运行程序，单击文件|保存 SVG 文件，您应该能够将图形保存到 SVG 文件中。尝试用网络浏览器打开文件，看看它是什么样子的：

![](img/982756a6-c52f-45d4-ab97-f9429e05366c.png)

看起来我的网络浏览器（Firefox）不支持填充图案，但其他东西都很好。由于矢量图形是由程序生成的，形状不存储在 SVG 文件中（只存储数学公式及其变量），您可能需要确保用户平台支持您使用的功能。

在下一节中，我们将学习如何创建我们自己的绘画程序，并使用它绘制位图图像！

# 创建绘画程序

在接下来的部分，我们将转向像素领域，并学习如何使用 Qt 创建绘画程序。用户将能够通过使用不同大小和颜色的画笔来表达他们的创造力，绘制像素图像！

# 设置用户界面

同样，对于这个例子，我们将创建一个新的 Qt Widget 应用程序。之后，打开`mainwindow.ui`并在主窗口上添加一个菜单栏。然后，在菜单栏中添加以下选项：

![](img/6c4c1e46-259b-4888-a009-0a1ddbbac18c.png)

我们的菜单栏上有三个菜单项——文件、画笔大小和画笔颜色。在文件菜单下有将画布保存为位图文件的功能，以及清除整个画布的功能。画笔大小类别包含不同的画笔大小选项；最后，画笔颜色类别包含设置画笔颜色的几个选项。

您可以选择更像*绘画*或*Photoshop*的 GUI 设计，但出于简单起见，我们现在将使用这个。

完成所有这些后，打开`mainwindow.h`并在顶部添加以下头文件：

```cpp
#include <QMainWindow> 
#include <QPainter> 
#include <QMouseEvent> 
#include <QFileDialog> 
```

之后，我们还声明了一些虚拟函数，如下所示：

```cpp
public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    virtual void mousePressEvent(QMouseEvent *event); 
    virtual void mouseMoveEvent(QMouseEvent *event); 
    virtual void mouseReleaseEvent(QMouseEvent *event); 
    virtual void paintEvent(QPaintEvent *event); 
    virtual void resizeEvent(QResizeEvent *event); 
```

除了我们在上一个示例中使用的`paintEvent()`函数之外，我们还可以添加一些用于处理鼠标事件和窗口调整事件的函数。然后，我们还向我们的`MainWindow`类添加以下变量：

```cpp
private: 
    Ui::MainWindow *ui; 
 QImage image; 
    bool drawing; 
    QPoint lastPoint; 
    int brushSize; 
    QColor brushColor; 
```

之后，让我们打开`mainwindow.cpp`并从类构造函数开始：

```cpp
MainWindow::MainWindow(QWidget *parent) : 
    QMainWindow(parent), 
    ui(new Ui::MainWindow) 
{ 
    ui->setupUi(this); 

 image = QImage(this->size(), QImage::Format_RGB32); 
    image.fill(Qt::white); 

    drawing = false; 
    brushColor = Qt::black; 
    brushSize = 2; 
} 
```

我们需要首先创建一个`QImage`对象，它充当画布，并将其大小设置为与我们的窗口大小相匹配。然后，我们将默认画笔颜色设置为黑色，其默认大小设置为`2`。之后，我们将看一下每个事件处理程序及其工作原理。

首先，让我们看一下`paintEvent()`函数，这也是我们在矢量图形示例中使用的。这一次，它所做的就是调用`QPainter::drawImage()`并在我们的主窗口上渲染`QImage`对象（我们的图像缓冲区）：

```cpp
void MainWindow::paintEvent(QPaintEvent *event)
{
    QPainter canvasPainter(this);
    canvasPainter.drawImage(this->rect(), image, image.rect());
}
```

接下来，我们将看一下`resizeEvent()`函数，每当用户调整主窗口大小时都会触发该函数。为了避免图像拉伸，我们必须调整图像缓冲区的大小以匹配新的窗口大小。这可以通过创建一个新的`QImage`对象并设置其大小与调整后的主窗口相同来实现，然后复制先前的 QImage 的像素信息，并将其放置在新图像缓冲区的完全相同的位置。

这意味着如果窗口大小小于绘图，您的图像将被裁剪，但至少画布不会被拉伸和扭曲图像，当窗口调整大小时。让我们看一下代码：

```cpp
void MainWindow::resizeEvent(QResizeEvent *event) 
{ 
    QImage newImage(event->size(), QImage::Format_RGB32); 
    newImage.fill(qRgb(255, 255, 255)); 

    QPainter painter(&newImage); 
    painter.drawImage(QPoint(0, 0), image); 
    image = newImage; 
} 
```

接下来，我们将看一下鼠标事件处理程序，我们将使用它来在画布上应用颜色。首先是`mousePressEvent()`函数，当我们开始按下鼠标按钮（在这种情况下是左鼠标按钮）时将触发该函数。在这一点上我们仍然没有画任何东西，但是将绘图布尔值设置为`true`并将我们的光标位置保存到`lastPoint`变量中。

```cpp
void MainWindow::mousePressEvent(QMouseEvent *event) 
{ 
    if (event->button() == Qt::LeftButton) 
    { 
        drawing = true; 
        lastPoint = event->pos(); 
    } 
} 
```

然后，这是`mouseMoveEvent()`函数，当鼠标光标移动时将被调用：

```cpp
void MainWindow::mouseMoveEvent(QMouseEvent *event) 
{ 
    if ((event->buttons() & Qt::LeftButton) && drawing) 
    { 
        QPainter painter(&image); 
        painter.setPen(QPen(brushColor, brushSize, Qt::SolidLine, 
        Qt::RoundCap, Qt::RoundJoin)); 
        painter.drawLine(lastPoint, event->pos()); 

        lastPoint = event->pos(); 
        this->update(); 
    } 
} 
```

在前面的代码中，我们检查是否确实在按住鼠标左键移动鼠标。如果是，那么我们就从上一个光标位置画一条线到当前光标位置。然后，我们保存当前光标位置到`lastPoint`变量，并调用`update()`通知 Qt 触发`paintEvent()`函数。

最后，当我们释放鼠标左键时，将调用`mouseReleaseEvent()`。我们只需将绘图变量设置为`false`，然后完成：

```cpp
void MainWindow::mouseReleaseEvent(QMouseEvent *event) 
{ 
    if (event->button() == Qt::LeftButton) 
    { 
        drawing = false; 
    } 
} 
```

如果我们现在构建并运行程序，我们应该能够在我们的小绘画程序上开始绘制一些东西：

![](img/89598a8c-02fa-4d93-9868-37aa3a30d6f8.png)

尽管现在我们可以绘制一些东西，但都是相同的笔刷大小和相同的颜色。这有点无聊！让我们在主菜单的“笔刷大小”类别上右键单击每个选项，然后选择“转到槽...”，然后选择“触发()”选项，然后按“确定”。然后 Qt 将为我们创建相应的槽函数，我们需要在这些函数中做的就是基本上改变 brushSize 变量，就像这样：

```cpp
void MainWindow::on_action2px_triggered() 
{ 
    brushSize = 2; 
} 

void MainWindow::on_action5px_triggered() 
{ 
    brushSize = 5; 
} 

void MainWindow::on_action10px_triggered() 
{ 
    brushSize = 10; 
} 
```

在“笔刷颜色”类别下的所有选项也是一样的。这次，我们相应地设置了`brushColor`变量：

```cpp
void MainWindow::on_actionBlack_triggered() 
{ 
    brushColor = Qt::black; 
} 

void MainWindow::on_actionWhite_triggered() 
{ 
    brushColor = Qt::white; 
} 

void MainWindow::on_actionRed_triggered() 
{ 
    brushColor = Qt::red; 
} 

void MainWindow::on_actionGreen_triggered() 
{ 
    brushColor = Qt::green; 
} 

void MainWindow::on_actionBlue_triggered() 
{ 
    brushColor = Qt::blue; 
} 
```

如果您再次构建和运行程序，您将能够使用各种笔刷设置绘制图像：

![](img/a9b9ff12-8980-45c0-8cf2-ed3f9eaab8fe.png)

除此之外，我们还可以将现有的位图图像添加到我们的画布上，以便我们可以在其上绘制。假设我有一个企鹅图像，以 PNG 图像的形式存在（名为`tux.png`），我们可以在类构造函数中添加以下代码：

```cpp
MainWindow::MainWindow(QWidget *parent) : 
    QMainWindow(parent), 
    ui(new Ui::MainWindow) 
{ 
    ui->setupUi(this); 

    image = QImage(this->size(), QImage::Format_RGB32); 
    image.fill(Qt::white); 

    QImage tux; 
    tux.load(qApp->applicationDirPath() + "/tux.png"); 
    QPainter painter(&image); 
    painter.drawImage(QPoint(100, 100), tux); 

    drawing = false; 
    brushColor = Qt::black; 
    brushSize = 2; 
} 
```

前面的代码基本上打开图像文件并将其移动到位置 100 x 100，然后将图像绘制到我们的图像缓冲区上。现在，每当我们启动程序时，我们就可以在画布上看到一个企鹅：

![](img/1eb1fc12-4d31-4fd6-be34-8c5c174ff48d.png)

接下来，我们将看一下“文件”下的“清除”选项。当用户在菜单栏上点击此选项时，我们使用以下代码清除整个画布（包括企鹅）并重新开始：

```cpp
void MainWindow::on_actionClear_triggered() 
{ 
    image.fill(Qt::white); 
    this->update(); 
} 
```

最后，当用户点击“文件”下的“保存”选项时，我们打开一个文件对话框，让用户将他们的作品保存为位图文件。在以下代码中，我们过滤图像格式，只允许用户保存 PNG 和 JPEG 格式：

```cpp
void MainWindow::on_actionSave_triggered() 
{ 
    QString filePath = QFileDialog::getSaveFileName(this, "Save Image", "", "PNG (*.png);;JPEG (*.jpg *.jpeg);;All files (*.*)"); 

    if (filePath == "") 
        return; 

    image.save(filePath); 
} 
```

就是这样，我们成功地使用 Qt 从头开始创建了一个简单的绘画程序！您甚至可以将从本章学到的知识与上一章结合起来，创建一个在线协作白板！唯一的限制就是您的创造力。最后，我要感谢所有读者使用我们新创建的绘画程序创建了以下杰作：

![](img/2d43a36a-906f-4b53-8e78-f4a72b9416c6.jpg)

# 总结

在这一章中，我们学习了如何绘制矢量和位图图形，随后我们使用 Qt 创建了自己的绘画程序。在接下来的章节中，我们将研究创建一个将数据传输并存储到云端的程序的方面。
