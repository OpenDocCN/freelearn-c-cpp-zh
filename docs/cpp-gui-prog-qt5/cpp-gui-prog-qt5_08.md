# 第八章：Graphics View

在上一章中，我们学习了通过在地图上显示坐标数据来为用户提供视觉呈现的重要性。在本章中，我们将进一步探索使用 Qt 的`Graphics View`框架来表示图形数据的可能性。

在本章中，我们将涵盖以下主题：

+   Graphics View 框架

+   可移动的图形项

+   创建一个组织图表

在本章结束时，你将能够使用 C++和 Qt 的 API 创建一个组织图表显示。让我们开始吧！

# Graphics View 框架

`Graphics View`框架是 Qt 中的小部件模块的一部分，因此它已经默认支持，除非你运行的是 Qt 控制台应用程序，它不需要小部件模块。

在 Qt 中，`Graphics View`视图的工作方式基本上就像一个白板，你可以使用 C/C++代码在上面画任何东西，比如绘制形状、线条、文本，甚至图像。对于初学者来说，这一章可能有点难以理解，但肯定会是一个有趣的项目。让我们开始吧！

# 设置一个新项目

首先，创建一个新的 Qt Widgets 应用程序项目。之后，打开`mainwindow.ui`，将`Graphics View`小部件拖放到主窗口上，就像这样：

![](img/01a52e3d-f0ef-4e70-b7dd-c390e5edc2b1.png)

然后，通过点击画布顶部的垂直布局按钮为图形视图创建一个布局。之后，打开`mainwindow.h`并添加以下头文件和变量：

```cpp
#include <QGraphicsScene> 
#include <QGraphicsRectItem> 
#include <QGraphicsEllipseItem> 
#include <QGraphicsTextItem> 
#include <QBrush> 
#include <QPen> 

private:
  Ui::MainWindow *ui;
  QGraphicsScene* scene;
```

之后，打开`mainwindow.cpp`。一旦打开，添加以下代码：

```cpp
MainWindow::MainWindow(QWidget *parent) : 
   QMainWindow(parent), 
   ui(new Ui::MainWindow) 
{ 
   ui->setupUi(this); 

   scene = new QGraphicsScene(this); 
   ui->graphicsView->setScene(scene); 

   QBrush greenBrush(Qt::green); 
   QBrush blueBrush(Qt::blue); 
   QPen pen(Qt::black); 
   pen.setWidth(2); 

   QGraphicsRectItem* rectangle = scene->addRect(80, 0, 80, 80, pen, greenBrush); 
   QGraphicsEllipseItem* ellipse = scene->addEllipse(0, -80, 200, 60, pen, blueBrush); 
   QGraphicsTextItem* text = scene->addText("Hello World!", QFont("Times", 25)); 
} 
```

现在构建并运行程序，你应该会看到类似这样的东西：

![](img/8b6246e1-dedd-4df3-b865-75e323337c8a.png)

代码有点长，所以让我向你解释一下它的作用以及它如何将图形绘制到屏幕上。

正如我之前所说，`Graphics View`小部件就像一个画布或白板，允许你在上面画任何你想要的东西。然而，我们还需要一个叫做 Graphics Scene 的东西，它本质上是一个场景图，它在显示在`Graphics View`上之前以父子层次结构存储所有图形组件。场景图层次结构就像在之前的截图中出现的图像，每个对象都可以有一个链接在一起的父对象或子对象：

![](img/25c1c1d4-4bba-4b7d-9fe3-ec7e109bfcbb.png)

在上面的代码中，我们首先创建了一个`QGraphicsScene`对象，并将其设置为我们的`Graphics View`小部件的 Graphics Scene：

```cpp
scene = new QGraphicsScene(this); 
ui->graphicsView->setScene(scene); 
```

然而，在这个例子中，我们不必将图形项链接在一起，所以我们只需独立创建它们，就像这样：

```cpp
QBrush greenBrush(Qt::green); 
...
QGraphicsTextItem* text = scene->addText("Hello World!", QFont("Times", 25)); 
```

`QPen`和`QBrush`类用于定义这些图形项的渲染样式。`QBrush`通常用于定义项目的背景颜色和图案，而`QPen`通常影响项目的轮廓。

Qt 提供了许多类型的图形项，用于最常见的形状，包括：

+   `QGraphicsEllipseItem` – 椭圆项

+   `QGraphicsLineItem` – 线条项

+   `QGraphicsPathItem` – 任意路径项

+   `QGraphicsPixmapItem` – 图像项

+   `QGraphicsPolygonItem` – 多边形项

+   `QGraphicsRectItem` – 矩形项

+   `QGraphicsSimpleTextItem` – 简单文本标签项

+   `QGraphicsTextItem` – 高级格式化文本项

更多信息，请访问此链接：[`doc.qt.io/archives/qt-5.8/qgraphicsitem.html#details.`](http://doc.qt.io/archives/qt-5.8/qgraphicsitem.html#details)

# 可移动的图形项

在上一个例子中，我们成功地将一些简单的形状和文本绘制到了`Graphics View`小部件上。然而，这些图形项是不可交互的，因此不适合我们的目的。我们想要的是一个交互式的组织图表，用户可以使用鼠标移动项目。在 Qt 下，使这些项目可移动实际上非常容易；让我们看看我们如何通过继续我们之前的项目来做到这一点。

首先，确保不要更改我们的图形视图小部件的默认交互属性，即启用（复选框已选中）：

![](img/01e7d066-00f9-4c4e-ae4b-9352ecf34437.png)

在那之后，在之前的`Hello World`示例中创建的每个图形项下面添加以下代码：

```cpp
QGraphicsRectItem* rectangle = scene->addRect(80, 0, 80, 80, pen, greenBrush); 
rectangle->setFlag(QGraphicsItem::ItemIsMovable); 
rectangle->setFlag(QGraphicsItem::ItemIsSelectable); 

QGraphicsEllipseItem* ellipse = scene->addEllipse(0, -80, 200, 60, pen, blueBrush); 
ellipse->setFlag(QGraphicsItem::ItemIsMovable); 
ellipse->setFlag(QGraphicsItem::ItemIsSelectable); 

QGraphicsTextItem* text = scene->addText("Hello World!", QFont("Times", 25)); 
text->setFlag(QGraphicsItem::ItemIsMovable); 
text->setFlag(QGraphicsItem::ItemIsSelectable); 
```

再次构建和运行程序，这次您应该能够在图形视图中选择和移动项目。请注意，`ItemIsMovable`和`ItemIsSelectable`都会给您不同的行为——前者标志将使项目可以通过鼠标移动，而后者使项目可选择，通常在选择时使用虚线轮廓进行视觉指示。每个标志都独立工作，不会影响其他标志。

我们可以通过使用 Qt 中的信号和槽机制来测试`ItemIsSelectable`标志的效果。让我们回到我们的代码并添加以下行：

```cpp
ui->setupUi(this); 
scene = new QGraphicsScene(this); 
ui->graphicsView->setScene(scene); 
connect(scene, &QGraphicsScene::selectionChanged, this, &MainWindow::selectionChanged); 
```

`selectionChanged()`信号将在您在图形视图小部件上选择项目时触发，然后`MainWindow`类下的`selectionChanged()`槽函数将被调用（我们需要编写）。让我们打开`mainwindow.h`并添加另一个头文件以显示调试消息：

```cpp
#include <QDebug> 
```

然后，我们声明槽函数，就像这样：

```cpp
private: 
   Ui::MainWindow *ui; 

public slots: 
 void selectionChanged(); 
```

之后打开`mainwindow.cpp`并定义槽函数，就像这样：

```cpp
void MainWindow::selectionChanged() 
{ 
   qDebug() << "Item selected"; 
} 
```

现在尝试再次运行程序；您应该看到一行调试消息，每当单击图形项时会出现“项目选择”。这真的很简单，不是吗？

至于`ItemIsMovable`标志，我们将无法使用信号和槽方法进行测试。这是因为所有从`QGraphicsItem`类继承的类都不是从`QObject`类继承的，因此信号和槽机制不适用于这些类。这是 Qt 开发人员有意为之，以使其轻量级，从而提高性能，特别是在屏幕上渲染数千个项目时。

尽管信号和槽对于这个选项不是一个选择，我们仍然可以使用事件系统，这需要对`itemChange()`虚函数进行重写，我将在下一节中演示。

# 创建组织图表

让我们继续学习如何使用 Graphics View 创建组织图表。组织图表是一种显示组织结构和员工职位关系层次结构的图表。通过使用图形表示来理解公司的结构是很容易的；因此最好使用 Graphics View 而不是表格。

这一次，我们需要为图形项创建自己的类，以便我们可以利用 Qt 的事件系统，并且更好地控制它的分组和显示方式。

首先，通过转到文件 | 新建文件或项目来创建一个 C/C++类：

![](img/a86a053b-8bab-4827-b081-a2858e1b1d66.png)

接下来，在点击下一步和完成按钮之前，将我们的类命名为`profileBox`：

![](img/e257c658-4e37-45d9-a89a-e0ef788161b7.png)

之后，打开`mainwindow.h`并添加这些头文件：

```cpp
#include <QWidget> 
#include <QDebug> 
#include <QBrush> 
#include <QPen> 
#include <QFont> 
#include <QGraphicsScene> 
#include <QGraphicsItemGroup> 
#include <QGraphicsItem> 
#include <QGraphicsRectItem> 
#include <QGraphicsTextItem> 
#include <QGraphicsPixmapItem> 
```

然后，打开`profilebox.h`并使我们的`profileBox`类继承`QGraphicsItemGroup`：

```cpp
class profileBox : public QGraphicsItemGroup 
{ 
public: 
   explicit profileBox(QGraphicsItem* parent = nullptr); 
```

在那之后，打开`profilebox.cpp`并在类的构造函数中设置`QBrush`、`QPen`和`QFont`，这将在稍后用于渲染：

```cpp
profileBox::profileBox(QGraphicsItem *parent) : QGraphicsItemGroup(parent) 
{ 
   QBrush brush(Qt::white); 
   QPen pen(Qt::black); 
   QFont font; 
   font.setFamily("Arial"); 
   font.setPointSize(12); 
} 
```

之后，在构造函数中，创建一个`QGraphicsRectItem`、`QGraphicsTextItem`和一个`QGraphicsPixmapItem`：

```cpp
QGraphicsRectItem* rectangle = new QGraphicsRectItem(); 
rectangle->setRect(0, 0, 90, 100); 
rectangle->setBrush(brush); 
rectangle->setPen(pen); 

nameTag = new QGraphicsTextItem(); 
nameTag->setPlainText(""); 
nameTag->setFont(font); 

QGraphicsPixmapItem* picture = new QGraphicsPixmapItem(); 
QPixmap pixmap(":/images/person-icon-blue.png"); 
picture->setPixmap(pixmap); 
picture->setPos(15, 30); 
```

然后，将这些项目添加到组中，这是当前类，因为这个类是从`QGraphicsItemGroup`类继承的：

```cpp
this->addToGroup(rectangle); 
this->addToGroup(nameTag); 
this->addToGroup(picture); 
```

最后，为当前类设置三个标志，即`ItemIsMovable`、`ItemIsSelectable`和`ItemSendsScenePositionChanges`：

```cpp
this->setFlag(QGraphicsItem::ItemIsMovable); 
this->setFlag(QGraphicsItem::ItemIsSelectable); 
this->setFlag(QGraphicsItem::ItemSendsScenePositionChanges); 
```

这些标志非常重要，因为它们默认情况下都是禁用的，出于性能原因。我们在上一节中已经涵盖了`ItemIsMovable`和`ItemIsSelectable`，而`ItemSendsPositionChanges`是一些新的东西。此标志使图形项在用户移动时通知图形场景，因此得名。

接下来，创建另一个名为`init()`的函数，用于设置员工个人资料。为简单起见，我们只设置了员工姓名，但是如果您愿意，还可以进行更多操作，例如根据职级设置不同的背景颜色，或更改其个人资料图片：

```cpp
void profileBox::init(QString name, MainWindow *window, QGraphicsScene* scene) 
{ 
   nameTag->setPlainText(name); 
   mainWindow = window; 
   scene->addItem(this); 
} 
```

请注意，我们还在这里设置了主窗口和图形场景指针，以便以后使用。在将其呈现在屏幕上之前，我们必须将`QGraphicsItem`添加到场景中。在这种情况下，我们将所有图形项分组到`QGraphicsItemGroup`中，因此我们只需要将组添加到场景中，而不是单个项。

请注意，您必须在`profilebox.h`中的`#include "mainwindow.h"`之后进行`MainWindow`类的前向声明，以避免递归头文件包含错误。同时，我们还在`profilebox.h`中放置了`MainWindow`和`QGraphicsTextItem`指针，以便以后调用它们：

```cpp
#include "mainwindow.h" 

class MainWindow; 

class profileBox : public QGraphicsItemGroup 
{ 
public: 
   explicit profileBox(QGraphicsItem* parent = nullptr); 
   void init(QString name, MainWindow* window, QGraphicsScene* scene); 

private: 
   MainWindow* mainWindow; 
   QGraphicsTextItem* nameTag; 

```

您还会注意到，我在`QGraphicsPixmapItem`中使用了一个图标作为装饰图标：

![](img/9787b7b9-c914-42cd-a823-622a852bea88.png)

此图标是存储在资源文件中的 PNG 图像。您可以从我们在 GitHub 页面上的示例项目文件中获取此图像：[`github.com/PacktPublishing/Hands-On-GUI-Programming-with-C-QT5`](http://github.com/PacktPublishing/Hands-On-GUI-Programming-with-C-QT5)

为您的项目创建一个资源文件。转到文件|新建文件或项目，然后在 Qt 类别下选择 Qt 资源文件选项：

![](img/b48d892f-781d-4781-a7d1-2548d6d5dca4.png)

创建空的资源文件后，通过添加|添加前缀添加一个新前缀。我们将只称此前缀为`images`：

![](img/8a5addd2-28d8-4bb3-b595-9acd7b2a0531.png)

然后，选择新创建的`images`前缀，单击添加|添加文件。将图标图像添加到资源文件并保存。您现在已成功将图像添加到项目中。

![](img/1803de41-0a6d-4761-bf75-a9da8ff984bf.png)

如果您的前缀名称或文件名与本书中的前缀名称或文件名不同，您可以右键单击资源文件中的图像，然后选择复制资源路径到剪贴板，并用您的路径替换代码中的路径。

![](img/27bfcbda-33d3-4330-8ed7-33cd8082e990.png)

之后，打开`mainwindow.h`并添加：

```cpp
#include "profilebox.h"
```

然后，打开`mainwindow.cpp`并添加以下代码以手动创建个人资料框：

```cpp
MainWindow::MainWindow(QWidget *parent) : 
   QMainWindow(parent), 
   ui(new Ui::MainWindow) 
{ 
   ui->setupUi(this); 

   scene = new QGraphicsScene(this); 
   ui->graphicsView->setScene(scene); 

   connect(scene, &QGraphicsScene::selectionChanged, this, &MainWindow::selectionChanged); 

   profileBox* box = new profileBox(); 
   box->init("John Doe", this, scene); 
} 
```

现在构建和运行项目，您应该看到类似于这样的东西：

![](img/af8e4e65-97ea-4571-a6f0-b00a43980191.png)

看起来整洁；但我们还远未完成。还有一些事情要做——我们必须允许用户通过用户界面添加或删除个人资料框，而不是使用代码。同时，我们还需要添加连接不同个人资料框的线条，以展示不同员工之间的关系以及他们在公司内的职位。

让我们从简单的部分开始。再次打开`mainwindow.ui`，并在图形视图小部件底部添加一个推送按钮，并将其命名为`addButton`：

![](img/03f5b860-526e-425f-b3d7-d24c86c84ebd.png)

然后，右键单击推送按钮，选择转到插槽...之后，选择单击选项，然后单击确定。将自动为您创建一个新的插槽函数，名为`on_addButton_clicked()`。添加以下代码以允许用户在单击添加按钮时创建个人资料框：

```cpp
void MainWindow::on_addButton_clicked() 
{ 
   bool ok; 
   QString name = QInputDialog::getText(this, tr("Employee Name"), 
   tr("Please insert employee's full name here:"), QLineEdit::Normal,  
   "John Doe", &ok); 
   if (ok && !name.isEmpty()) 
   { 
         profileBox* box = new profileBox(); 
         box->init(name, this, scene); 
   } 
} 
```

现在，用户不再需要使用代码创建每个个人资料框，他们可以通过单击添加按钮轻松创建任意数量的个人资料框。还将出现一个消息框，让用户在创建个人资料框之前输入员工姓名：

![](img/7f6b3cf5-f348-49a5-a43e-71172aab6166.png)

接下来，我们将创建另一个名为`profileLine`的类。这次，我们将使这个类继承`QGraphicsLineItem`。`profileline.h`基本上看起来像这样：

```cpp
#include <QWidget> 
#include <QGraphicsItem> 
#include <QPen> 

class profileLine : public QGraphicsLineItem 
{ 
public: 
   profileLine(QGraphicsItem* parent = nullptr); 
   void initLine(QGraphicsItem* start, QGraphicsItem* end); 
   void updateLine(); 

   QGraphicsItem* startBox; 
   QGraphicsItem* endBox; 

private: 
}; 
```

与`profileBox`类类似，我们还为`profileLine`类创建了一个`init`函数，称为`initLine()`函数。此函数接受两个`QGraphicsItem`对象作为渲染行的起点和终点。此外，我们还创建了一个`updateLine()`函数，以便在配置框移动时重新绘制行。

接下来，打开`profileline.cpp`并将以下代码添加到构造函数中：

```cpp
profileLine::profileLine(QGraphicsItem *parent) : QGraphicsLineItem(parent) 
{ 
   QPen pen(Qt::black); 
   pen.setWidth(2); 
   this->setPen(pen); 

   this->setZValue(-999); 
} 
```

我们使用`QPen`将线的颜色设置为黑色，宽度设置为`2`。之后，我们还将线的`Zvalue`设置为`-999`，这样它将始终保持在配置框的后面。

之后，将以下代码添加到我们的`initLine()`函数中，使其看起来像这样：

```cpp
void profileLine::initLine(QGraphicsItem* start, QGraphicsItem* end) 
{ 
   startBox = start; 
   endBox = end; 

   updateLine(); 
} 
```

它的作用基本上是设置框的起点和终点位置。之后，调用`updateLine()`函数来渲染行。

最后，`updateLine()`函数看起来像这样：

```cpp
void profileLine::updateLine() 
{ 
   if (startBox != NULL && endBox != NULL) 
   { 
         this->setLine(startBox->pos().x() + startBox->boundingRect().width() / 2, startBox->pos().y() + startBox->boundingRect().height() / 2, endBox->pos().x() + endBox->boundingRect().width() / 2, endBox->pos().y() + endBox->boundingRect().height() / 2); 
   } 
} 
```

前面的代码看起来有点复杂，但如果我这样说，它就真的很简单：

```cpp
this->setLine(x1, y1, x2, y2); 
```

值`x1`和`y1`基本上是第一个配置框的中心位置，而`x2`和`y2`是第二个配置框的中心位置。由于从调用`pos()`获取的位置值从左上角开始，我们必须获取配置框的边界大小并除以二以获取其中心位置。然后，将该值添加到左上角位置以将其偏移至中心。

完成后，让我们再次打开`mainwindow.cpp`并将以下代码添加到`on_addButton_clicked()`函数中：

```cpp
void MainWindow::on_addButton_clicked() 
{ 
   bool ok; 
   QString name = QInputDialog::getText(this, tr("Employee Name"), tr("Please insert employee's full name here:"), QLineEdit::Normal, "John Doe", &ok); 
   if (ok && !name.isEmpty()) 
   { 
         profileBox* box = new profileBox(); 
         box->init(name, this, scene); 

         if (scene->selectedItems().size() > 0) 
         { 
               profileLine* line = new profileLine(); 
               line->initLine(box, scene->selectedItems().at(0)); 
               scene->addItem(line); 

               lines.push_back(line); 
         } 
   } 
} 
```

在前面的代码中，我们检查用户是否选择了任何配置框。如果没有，我们就不必创建任何线。否则，创建一个新的`profileLine`对象，并将新创建的配置框和当前选择的配置框设置为`startBox`和`endBox`属性。

之后，将该`profileLine`对象添加到我们的图形场景中，以便它出现在屏幕上。最后，将此`profileLine`对象存储到`QList`数组中，以便我们以后使用。在`mainwindow.h`中，数组声明如下所示：

```cpp
private: 
   Ui::MainWindow *ui; 
   QGraphicsScene* scene; 
   QList<profileLine*> lines; 
```

现在构建和运行项目。当您点击“添加”按钮创建第二个配置框时，您应该能够看到线出现，并在选择第一个框时保持选中。但是，您可能会注意到一个问题，即当您将配置框移出原始位置时，线根本不会更新自己！：

![](img/115bd268-e78e-49b4-95d5-f58d962cd051.png)

这是我们将行放入`QList`数组的主要原因，这样我们就可以在用户移动配置框时更新这些行。

为此，首先，我们需要重写`profileBox`类中的虚函数`itemChanged()`。让我们打开`profilebox.h`并添加以下代码行：

```cpp
class profileBox : public QGraphicsItemGroup 
{ 
public: 
   explicit profileBox(QGraphicsItem* parent = nullptr); 
   void init(QString name, MainWindow* window, QGraphicsScene* scene); 
   QVariant itemChange(GraphicsItemChange change, const QVariant 
   &value) override; 
```

然后，打开`profilebox.cpp`并添加`itemChanged()`的代码：

```cpp
QVariant profileBox::itemChange(GraphicsItemChange change, const QVariant &value) 
{ 
   if (change == QGraphicsItem::ItemPositionChange) 
   { 
         qDebug() << "Item moved"; 

         mainWindow->updateLines(); 
   } 

   return QGraphicsItem::itemChange(change, value); 
} 
```

`itemChanged()`函数是`QGraphicsItem`类中的虚函数，当图形项发生变化时，Qt 的事件系统将自动调用它，无论是位置变化、可见性变化、父级变化、选择变化等等。

因此，我们所需要做的就是重写该函数并向函数中添加我们自己的自定义行为。在前面的示例代码中，我们所做的就是在我们的主窗口类中调用`updateLines()`函数。

接下来，打开`mainwindow.cpp`并定义`updateLines()`函数。正如函数名所示，您要在此函数中做的是循环遍历存储在行数组中的所有配置行对象，并更新每一个，如下所示：

```cpp
void MainWindow::updateLines() 
{ 
   if (lines.size() > 0) 
   { 
         for (int i = 0; i < lines.size(); i++) 
         { 
               lines.at(i)->updateLine(); 
         } 
   } 
} 
```

完成后，再次构建和运行项目。这次，您应该能够创建一个组织图表，如下所示：

![](img/72762a59-c68a-4e39-8bb7-097924dd8425.png)

这只是一个更简单的版本，向您展示了如何利用 Qt 强大的图形视图系统来显示一组数据的图形表示，这些数据可以被普通人轻松理解。

在完成之前还有一件事-我们还没有讲解如何删除配置档框。实际上很简单，让我们打开`mainwindow.h`并添加`keyReleaseEvent()`函数，看起来像这样：

```cpp
public: 
   explicit MainWindow(QWidget *parent = 0); 
   ~MainWindow(); 

   void updateLines(); 
   void keyReleaseEvent(QKeyEvent* event); 
```

这个虚函数在键盘按钮被按下和释放时也会被 Qt 的事件系统自动调用。函数的内容在`mainwindow.cpp`中看起来像这样：

```cpp
void MainWindow::keyReleaseEvent(QKeyEvent* event) 
{ 
   qDebug() << "Key pressed: " + event->text(); 

   if (event->key() == Qt::Key_Delete) 
   { 
         if (scene->selectedItems().size() > 0) 
         { 
               QGraphicsItem* item = scene->selectedItems().at(0); 
               scene->removeItem(item); 

               for (int i = lines.size() - 1; i >= 0; i--) 
               { 
                     profileLine* line = lines.at(i); 

                     if (line->startBox == item || line->endBox == 
                     item) 
                     { 
                           lines.removeAt(i); 
                           scene->removeItem(line); 
                           delete line; 
                     } 
               } 
               delete item; 
         } 
   } 
} 
```

在这个函数中，我们首先要检测用户按下的键盘按钮。如果按钮是`Qt::Key_Delete (删除按钮)`，那么我们将检查用户是否选择了任何配置档框，通过检查`scene->selectedItems().size()`是否为空来判断。如果用户确实选择了一个配置档框，那么就从图形场景中移除该项。之后，循环遍历线数组，并检查是否有任何配置线连接到已删除的配置档框。从场景中移除连接到配置档框的任何线，然后我们就完成了：

![](img/6fb5678c-13c1-4f9e-a849-3abd267b209c.png)

这个截图显示了从组织结构图中删除`Jane Smith`配置档框的结果。请注意，连接配置框的线已经被正确移除。就是这样，本章到此结束；希望您觉得这很有趣，也许会继续创造比这更好的东西！

# 总结

在本章中，我们学习了如何使用 Qt 创建一个应用程序，允许用户轻松创建和编辑组织结构图。我们学习了诸如`QGraphicsScene`、`QGrapicsItem`、`QGraphicsTextItem`、`QGraphicsPixmapItem`等类，这些类帮助我们在短时间内创建一个交互式组织结构图。在接下来的章节中，我们将学习如何使用网络摄像头捕捉图像！
