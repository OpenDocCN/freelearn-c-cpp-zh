# 第五章：项目视图和对话框

在上一章中，我们学习了如何使用不同类型的图表显示数据。图表是向用户在屏幕上呈现信息的许多方式之一。对于您的应用程序来说，向用户呈现重要信息非常重要，这样他们就可以准确地了解应用程序的情况——无论数据是否已成功保存，或者应用程序正在等待用户的输入，或者用户应该注意的警告/错误消息等等——这些都非常重要，以确保您的应用程序的用户友好性和可用性。

在本章中，我们将涵盖以下主题：

+   使用项目视图部件

+   使用对话框

+   使用文件选择对话框

+   图像缩放和裁剪

Qt 为我们提供了许多类型的部件和对话框，我们可以轻松使用它们来向用户显示重要信息。让我们看看这些部件是什么！

# 使用项目视图部件

除了使用不同类型的图表显示数据外，我们还可以使用不同类型的项目视图来显示这些数据。项目视图部件通过在垂直轴上呈现数据来将数据可视化呈现。

二维项目视图，通常称为**表视图**，在垂直和水平方向上显示数据。这使它能够在紧凑的空间内显示大量数据，并使用户能够快速轻松地搜索项目。

在项目视图中显示数据有两种方法。最常见的方法是使用**模型-视图架构**，它使用三个不同的组件，模型、视图和委托，从数据源检索数据并在项目视图中显示它。这些组件都利用 Qt 提供的**信号-槽架构**来相互通信：

+   模型的信号通知视图有关数据源保存的数据的更改

+   视图的信号提供有关用户与正在显示的项目的交互的信息

+   委托的信号在编辑期间用于告诉模型和视图有关编辑器状态的信息

另一种方法是手动方式，程序员必须告诉 Qt 哪些数据放在哪一列和行。与模型-视图相比，这种方法要简单得多，但在性能上要慢得多。然而，对于少量数据，性能问题可以忽略不计，这是一个很好的方法。

如果您打开 Qt Designer，您将看到两种不同的项目视图部件类别，即项目视图（基于模型）和项目部件（基于项目）：

![](img/6f63f909-cc29-4299-baf3-b34e7655cf7d.png)

尽管它们看起来可能相同，但实际上这两个类别中的部件工作方式非常不同。在本章中，我们将学习如何使用后一类别，因为它更直观、易于理解，并且可以作为前一类别的先决知识。

在项目部件（基于项目）类别下有三种不同的部件，称为列表部件、树部件和表部件。每个项目部件以不同的方式显示数据。选择适合您需求的部件：

![](img/a50b4415-472d-4c74-b1e1-f735f0a5bd21.png)

正如您从前面的图表中所看到的，**列表部件**以一维列表显示其项目，而**表部件**以二维表格显示其项目。尽管**树部件**几乎与**列表部件**类似，但其项目以分层结构显示，其中每个项目下可以递归地有多个子项目。一个很好的例子是我们操作系统中的文件系统，它使用树部件显示目录结构。

为了说明这些区别，让我们创建一个新的 Qt Widgets 应用程序项目，并自己试一试。

# 创建我们的 Qt Widgets 应用程序

创建项目后，打开`mainwindow.ui`并将三种不同的项目小部件拖到主窗口中。之后，选择主窗口并点击位于顶部的垂直布局按钮：

![](img/e0e15392-5def-4f64-accd-075c8e6d2778.png)

然后，双击列表小部件，将弹出一个新窗口。在这里，您可以通过单击+图标向列表小部件添加一些虚拟项目，或者通过选择列表中的项目并单击-图标来删除它们。单击“确定”按钮将最终结果应用于小部件：

![](img/4a126e2c-2059-4faf-a325-fafa3b81ce9d.png)

您可以对树形小部件执行相同的操作。它几乎与列表小部件相同，只是您可以向项目添加子项目，递归地。您还可以向树形小部件添加列并命名这些列：

![](img/486ef112-0666-4f94-a753-eac1472c352e.png)

最后，双击表格小部件以打开编辑表格小部件窗口。与其他两个项目视图不同，表格小部件是一个二维项目视图，这意味着您可以像电子表格一样向其添加列和行。可以通过在“列”或“行”选项卡中设置所需的名称来为每列和行加标签：

![](img/c3e07ba3-fb5c-4017-90db-caf9edae65e1.png)

通过使用 Qt Designer，了解小部件的工作原理非常容易。只需将小部件拖放到窗口中并调整其设置，然后构建并运行项目以查看结果。

在这种情况下，我们已经演示了三种不同的项目视图小部件之间的区别，而不需要编写一行代码：

![](img/cb94f990-0b6d-435e-8bea-c09205c56bf3.png)

# 使我们的列表小部件功能化

然而，为了使小部件在应用程序中完全可用，仍然需要编写代码。让我们学习如何使用 C++代码向我们的项目视图小部件添加项目！

首先，打开`mainwindow.cpp`并在`ui->setupui(this)`之后的类构造函数中编写以下代码：

```cpp
ui->listWidget->addItem("My Test Item"); 
```

就这么简单，您已成功向列表小部件添加了一个项目！

![](img/2f54393e-6a48-42c9-90c3-0d36ae463ad2.png)

还有另一种方法可以向列表小部件添加项目。但在此之前，我们必须向`mainwindow.h`添加以下头文件：

```cpp
#ifndef MAINWINDOW_H 
#define MAINWINDOW_H 

#include <QMainWindow> 
#include <QDebug> 
#include <QListWidgetItem> 
```

`QDebug`头文件用于打印调试消息，`QListWidgetItem`头文件用于声明列表小部件的项目对象。接下来，打开`mainwindow.cpp`并添加以下代码：

```cpp
QListWidgetItem* listItem = new QListWidgetItem; 
listItem->setText("My Second Item"); 
listItem->setData(100, 1000); 
ui->listWidget->addItem(listItem); 
```

前面的代码与前一个一行代码相同。不同的是，这次我向项目添加了额外的数据。`setData()`函数接受两个输入变量——第一个变量是项目的数据角色，指示 Qt 应如何处理它。如果放入与`Qt::ItemDataRole`枚举器匹配的值，数据将影响显示、装饰、工具提示等，这可能会改变其外观。

在我的情况下，我只是简单地设置了一个与`Qt::ItemDataRole`中的任何枚举器都不匹配的数字，以便我可以将其存储为以后使用的隐藏数据。要检索数据，您只需调用`data()`并插入与您刚刚设置的数字匹配的数字：

```cpp
qDebug() << listItem->data(100); 
```

构建并运行项目；您应该能够看到新项目现在已添加到列表小部件中：

![](img/bf13b2e0-4637-4f30-a7c4-2c79b541baad.png)

有关`Qt::ItemDataRole`枚举器的更多信息，请查看以下链接：[`doc.qt.io/qt-5/qt.html#ItemDataRole-enum`](http://doc.qt.io/qt-5/qt.html#ItemDataRole-enum)

如前所述，可以将隐藏数据附加到列表项目以供以后使用。例如，您可以使用列表小部件显示准备由用户购买的产品列表。每个项目都可以附加其产品 ID，以便当用户选择该项目并将其放入购物车时，您的系统可以自动识别已添加到购物车的产品 ID 作为数据角色存储。 

在上面的例子中，我在我的列表项中存储了自定义数据`1000`，并将其数据角色设置为`100`，这与任何`Qt::ItemDataRole`枚举器都不匹配。这样，数据就不会显示给用户，因此只能通过 C++代码检索。

# 向树部件添加功能

接下来，让我们转到树部件。实际上，它与列表部件并没有太大的不同。让我们看一下以下代码：

```cpp
QTreeWidgetItem* treeItem = new QTreeWidgetItem; 
treeItem->setText(0, "My Test Item"); 
ui->treeWidget->addTopLevelItem(treeItem); 
```

它与列表部件几乎相同，只是我们必须在`setText()`函数中设置列 ID。这是因为树部件介于列表部件和表部件之间——它可以有多个列，但不能有任何行。

树部件与其他视图部件最明显的区别是，所有的项都可以递归地包含子项。让我们看一下以下代码，看看我们如何向树部件中的现有项添加子项：

```cpp
QTreeWidgetItem* treeItem2 = new QTreeWidgetItem; 
treeItem2->setText(0, "My Test Subitem"); 
treeItem->addChild(treeItem2); 
```

就是这么简单！最终结果看起来像这样：

![](img/3580d596-4c97-4c34-9699-f54ddf816393.png)

# 最后，我们的表部件

接下来，让我们对表部件做同样的操作。从技术上讲，当列和行被创建时，表部件中的项已经存在并被保留。我们需要做的是创建一个新项，并用特定列和行的（当前为空的）项替换它，这就是为什么函数名叫做`setItem()`，而不是列表部件使用的`addItem()`。

让我们看一下代码：

```cpp
QTableWidgetItem* tableItem = new QTableWidgetItem; 
tableItem->setText("Testing1"); 
ui->tableWidget->setItem(0, 0, tableItem); 

QTableWidgetItem* tableItem2 = new QTableWidgetItem; 
tableItem2->setText("Testing2"); 
ui->tableWidget->setItem(1, 2, tableItem2); 
```

从代码中可以看出，我在两个不同的位置添加了两个数据部分，这将转化为以下结果：

![](img/05cecb5e-908c-4668-8afa-dde23dae413d.png)

就是这样！使用 Qt 中的项视图来显示数据是如此简单和容易。如果你正在寻找与项视图相关的更多示例，请访问以下链接：[`doc.qt.io/qt-5/examples-itemviews.html`](http://doc.qt.io/qt-5/examples-itemviews.html)

# 使用对话框

创建用户友好的应用程序的一个非常重要的方面是，在发生某个事件（有意或无意）时，能够显示关于应用程序状态的重要信息。为了显示这样的信息，我们需要一个外部窗口，用户可以在确认信息后将其关闭。

Qt 具有这个功能，它全部驻留在`QMessageBox`类中。在 Qt 中，你可以使用几种类型的消息框；最基本的一种只需要一行代码，就像这样：

```cpp
QMessageBox::information(this, "Alert", "Just to let you know, something happened!"); 
```

对于这个函数，你需要提供三个参数。第一个是消息框的父窗口，我们已经将其设置为主窗口。第二个参数是窗口标题，第三个参数是我们想要传递给用户的消息。上述代码将产生以下结果：

![](img/2b92e3f2-4363-46db-aa12-afc721e665e8.png)

这里显示的外观是在 Windows 系统上运行的。在不同的操作系统（Linux、macOS 等）上，外观可能会有所不同。正如你所看到的，对话框甚至带有文本之前的图标。你可以使用几种类型的图标，比如信息、警告和严重。以下代码向你展示了调用带有图标的不同消息框的代码：

```cpp
QMessageBox::question(this, "Alert", "Just to let you know, something happened!"); 
QMessageBox::warning(this, "Alert", "Just to let you know, something happened!"); 
QMessageBox::information(this, "Alert", "Just to let you know, something happened!"); 
QMessageBox::critical(this, "Alert", "Just to let you know, something happened!"); 
```

上述代码产生以下结果：

![](img/84024277-2f97-4651-b89e-b3a4e9528f8a.png)

如果你不需要任何图标，只需调用`QMessageBox::about()`函数。你还可以通过从 Qt 提供的标准按钮列表中选择来设置你想要的按钮，例如：

```cpp
QMessageBox::question(this, "Serious Question", "Am I an awesome guy?", QMessageBox::Ignore, QMessageBox::Yes); 
```

上述代码将产生以下结果：

![](img/e95537ee-6b14-4da6-9e38-af1eaaf6fda3.png)

由于这些是 Qt 提供的内置函数，用于轻松创建消息框，它不会给开发人员完全自定义消息框的自由。但是，Qt 允许您使用另一种方法手动创建消息框，这种方法比内置方法更可定制。这需要更多的代码行，但编写起来仍然相当简单：

```cpp
QMessageBox msgBox; 
msgBox.setWindowTitle("Alert"); 
msgBox.setText("Just to let you know, something happened!"); 
msgBox.exec(); 
```

上述代码将产生以下结果：

![](img/f79a9076-7112-4fa4-bed4-5bc3b9bc5628.png)

“看起来完全一样”，你告诉我。那么添加我们自己的图标和自定义按钮呢？这没有问题：

```cpp
QMessageBox msgBox; 
msgBox.setWindowTitle("Serious Question"); 
msgBox.setText("Am I an awesome guy?"); 
msgBox.addButton("Seriously Yes!", QMessageBox::YesRole); 
msgBox.addButton("Well no thanks", QMessageBox::NoRole); 
msgBox.setIcon(QMessageBox::Question); 
msgBox.exec(); 
```

上述代码产生以下结果：

![](img/04e2e8f6-5139-4bfb-aced-564cedaf5d2d.png)

在上面的代码示例中，我已经加载了 Qt 提供的问题图标，但如果您打算这样做，您也可以从资源文件中加载自己的图标：

```cpp
QMessageBox msgBox; 
msgBox.setWindowTitle("Serious Question"); 
msgBox.setText("Am I an awesome guy?"); 
msgBox.addButton("Seriously Yes!", QMessageBox::YesRole); 
msgBox.addButton("Well no thanks", QMessageBox::NoRole); 
QPixmap myIcon(":/images/icon.png"); 
msgBox.setIconPixmap(myIcon); 
msgBox.exec(); 
```

现在构建并运行项目，您应该能够看到这个奇妙的消息框：

![](img/ce170a25-75c4-448e-ab26-b82691eda029.png)

一旦您了解了如何创建自己的消息框，让我们继续学习消息框附带的事件系统。

当用户被呈现具有多个不同选择的消息框时，他/她会期望在按下不同按钮时应用程序有不同的反应。

例如，当消息框弹出并询问用户是否希望退出程序时，按钮“是”应该使程序终止，而“否”按钮将不起作用。

Qt 的`QMessageBox`类为我们提供了一个简单的解决方案来检查按钮事件。当消息框被创建时，Qt 将等待用户选择他们的选择；然后，它将返回被触发的按钮。通过检查哪个按钮被点击，开发人员可以继续触发相关事件。让我们看一下示例代码：

```cpp
if (QMessageBox::question(this, "Question", "Some random question. Yes or no?") == QMessageBox::Yes) 
{ 
   QMessageBox::warning(this, "Yes", "You have pressed Yes!"); 
} 
else 
{ 
   QMessageBox::warning(this, "No", "You have pressed No!"); 
} 
```

上述代码将产生以下结果：

![](img/4ec73206-502c-4051-aa2a-ba175f839f16.png)

如果您更喜欢手动创建消息框，检查按钮事件的代码会稍微长一些：

```cpp
QMessageBox msgBox; 
msgBox.setWindowTitle("Serious Question"); 
msgBox.setText("Am I an awesome guy?"); 
QPushButton* yesButton = msgBox.addButton("Seriously Yes!", QMessageBox::YesRole); 
QPushButton* noButton = msgBox.addButton("Well no thanks", QMessageBox::NoRole); 
msgBox.setIcon(QMessageBox::Question); 
msgBox.exec(); 

if (msgBox.clickedButton() == (QAbstractButton*) yesButton) 
{ 
   QMessageBox::warning(this, "Yes", "Oh thanks! :)"); 
} 
else if (msgBox.clickedButton() == (QAbstractButton*) noButton) 
{ 
   QMessageBox::warning(this, "No", "Oh why... :("); 
} 
```

尽管代码稍微长一些，但基本概念基本相同——被点击的按钮始终可以被开发人员检索以触发适当的操作。然而，这次，Qt 直接检查按钮指针，而不是检查枚举器，因为前面的代码没有使用`QMessageBox`类的内置标准按钮。

构建项目，您应该能够获得以下结果：

![](img/9c49eca5-6076-4609-9e1c-e3f2e7a1a762.png)

有关对话框的更多信息，请访问以下链接的 API 文档：[`doc.qt.io/qt-5/qdialog.html`](http://doc.qt.io/qt-5/qdialog.html)

# 创建文件选择对话框

既然我们已经讨论了消息框的主题，让我们也了解一下另一种类型的对话框——文件选择对话框。文件选择对话框也非常有用，特别是如果您的应用程序经常处理文件。要求用户输入他们想要打开的文件的绝对路径是非常不愉快的，因此文件选择对话框在这种情况下非常方便。

Qt 为我们提供了一个内置的文件选择对话框，看起来与我们在操作系统中看到的一样，因此，对用户来说并不陌生。文件选择对话框本质上只做一件事——让用户选择他们想要的文件或文件夹，并返回所选文件或文件夹的路径；就这些。实际上，它不负责打开文件和读取其内容。

让我们看看如何触发文件选择对话框。首先，打开`mainwindow.h`并添加以下头文件：

```cpp
#ifndef MAINWINDOW_H 
#define MAINWINDOW_H 

#include <QMainWindow> 
#include <QFileDialog> 
#include <QDebug> 
```

接下来，打开`mainwindow.cpp`并插入以下代码：

```cpp
QString fileName = QFileDialog::getOpenFileName(this); 
qDebug() << fileName; 
```

就是这么简单！现在构建并运行项目，您应该会得到这个：

![](img/cb0d3f1e-1e61-4d02-b280-9a4935ead609.png)

如果用户选择了文件并按下打开，`fileName` 变量将填充为所选文件的绝对路径。如果用户单击取消按钮，`fileName` 变量将为空字符串。

文件选择对话框在初始化步骤中还包含几个可以设置的选项。例如：

```cpp
QString fileName = QFileDialog::getOpenFileName(this, "Your title", QDir::currentPath(), "All files (*.*) ;; Document files (*.doc *.rtf);; PNG files (*.png)"); 
qDebug() << fileName; 
```

在前面的代码中，我们设置了三件事，它们如下：

+   文件选择对话框的窗口标题

+   对话框创建时用户看到的默认路径

+   文件类型过滤

文件类型过滤在您只允许用户选择特定类型的文件时非常方便（例如，仅允许 JPEG 图像文件），并隐藏其他文件。除了 `getOpenFileName()`，您还可以使用 `getSaveFileName()`，它将允许用户指定尚不存在的文件名。

有关文件选择对话框的更多信息，请访问以下链接的 API 文档：[`doc.qt.io/qt-5/qfiledialog.html`](http://doc.qt.io/qt-5/qfiledialog.html)

# 图像缩放和裁剪

由于我们在上一节中学习了文件选择对话框，我想这次我们应该学习一些有趣的东西！

首先，让我们创建一个新的 Qt Widgets 应用程序。然后，打开 `mainwindow.ui` 并创建以下用户界面：

![](img/867c3332-9f87-40c0-933b-05190c15dd8e.png)

让我们将这个用户界面分解成三个部分：

+   顶部—图像预览：

+   首先，在窗口中添加一个水平布局。

+   然后，将一个标签小部件添加到我们刚刚添加的水平布局中，然后将文本属性设置为 `empty`。将标签的 minimumSize 和 maximumSize 属性都设置为 150x150。最后，在 QFrame 类别下设置 frameShape 属性为 Box。

+   在标签的两侧添加两个水平间隔器，使其居中。

+   中部—用于调整的滑块：

+   在窗口中添加一个表单布局，放在我们在步骤 1 中刚刚添加的水平布局下方。

+   将三个标签添加到表单布局中，并将它们的文本属性分别设置为 `比例：`、`水平：` 和 `垂直：`。

+   将三个水平滑块添加到表单布局中。将最小属性设置为 `1`，最大属性设置为 `100`。然后，将 pageStep 属性设置为 `1`。

+   将比例滑块的值属性设置为 `100`。

+   底部—浏览按钮和保存按钮：

+   在窗口中添加一个水平布局，放在我们在步骤 2 中添加的表单布局下方。

+   将两个按钮添加到水平布局中，并将它们的文本属性分别设置为 `浏览` 和 `保存`。

+   +   最后，从中央小部件中删除菜单栏、工具栏和状态栏。

现在我们已经创建了用户界面，让我们开始编码吧！首先，打开 `mainwindow.h` 并添加以下头文件：

```cpp
#ifndef MAINWINDOW_H 
#define MAINWINDOW_H 

#include <QMainWindow> 
#include <QMessageBox> 
#include <QFileDialog> 
#include <QPainter> 
```

然后，将以下变量添加到 `mainwindow.h`：

```cpp
private: 
   Ui::MainWindow *ui; 
   bool canDraw; 
   QPixmap* pix; 
   QSize imageSize; 
   QSize drawSize; 
   QPoint drawPos; 
```

然后，返回到 `mainwindow.ui`，右键单击浏览按钮，然后选择转到槽。然后，一个窗口将弹出并要求您选择一个信号。选择位于列表顶部的 `clicked()` 信号，然后按下 OK 按钮：

![](img/e63f85a0-57c5-4097-a3aa-ddba2adc9e17.png)

在您的源文件中将自动添加一个新的 `slot` 函数。现在，添加以下代码以在单击浏览按钮时打开文件选择对话框。对话框仅列出 JPEG 图像并隐藏其他文件：

```cpp
void MainWindow::on_browseButton_clicked() 
{ 
   QString fileName = QFileDialog::getOpenFileName(this, tr("Open   
   Image"), QDir::currentPath(), tr("Image Files (*.jpg *.jpeg)")); 

   if (!fileName.isEmpty()) 
   { 
         QPixmap* newPix = new QPixmap(fileName); 

         if (!newPix->isNull()) 
         { 
               if (newPix->width() < 150 || newPix->height() < 150) 
               { 
                     QMessageBox::warning(this, tr("Invalid Size"), 
                     tr("Image size too small. Please use an image  
                     larger than 150x150.")); 
                     return; 
               } 

               pix = newPix; 
               imageSize = pix->size(); 
               drawSize = pix->size(); 

               canDraw = true; 

         } 
         else 
         { 
               canDraw = false; 

               QMessageBox::warning(this, tr("Invalid Image"), 
               tr("Invalid or corrupted file. Please try again with  
               another image file.")); 
         } 
   } 
} 
```

如您所见，代码检查用户是否选择了任何图像。如果选择了图像，它会再次检查图像分辨率是否至少为 150 x 150。如果没有问题，我们将保存图像的像素映射到名为 `pix` 的指针中，然后将图像大小保存到 `imageSize` 变量中，并将初始绘图大小保存到 `drawSize` 变量中。最后，我们将 `canDraw` 变量设置为 `true`。

之后，再次打开 `mainwindow.h` 并声明以下两个函数：

```cpp
public: 
   explicit MainWindow(QWidget *parent = 0); 
   ~MainWindow(); 
   virtual void paintEvent(QPaintEvent *event); 
   void paintImage(QString fileName, int x, int y); 
```

第一个函数`paintEvent()`是一个虚函数，每当 Qt 需要刷新用户界面时（例如当主窗口被调整大小时），它就会自动调用。我们将重写这个函数，并将新加载的图像绘制到图像预览部件上。在这种情况下，我们将在`paintEvent()`虚函数中调用`paintImage()`函数：

```cpp
void MainWindow::paintEvent(QPaintEvent *event) 
{ 
   if (canDraw) 
   { 
         paintImage("", ui->productImage->pos().x(), ui->productImage-
         >pos().y()); 
   } 
} 
```

之后，我们将在`mainwindow.cpp`中编写`paintImage()`函数：

```cpp
void MainWindow::paintImage(QString fileName, int x, int y) 
{ 
   QPainter painter; 
   QImage saveImage(150, 150, QImage::Format_RGB16); 

   if (!fileName.isEmpty()) 
   { 
         painter.begin(&saveImage); 
   } 
   else 
   { 
         painter.begin(this); 
   } 

   if (!pix->isNull()) 
   { 
         painter.setClipRect(x, y, 150, 150); 
         painter.fillRect(QRect(x, y, 150, 150), Qt::SolidPattern); 
         painter.drawPixmap(x - drawPos.x(), y - drawPos.y(), 
         drawSize.width(), drawSize.height(), *pix); 
   } 

   painter.end(); 

   if (fileName != "") 
   { 
         saveImage.save(fileName); 
         QMessageBox::information(this, "Success", "Image has been 
         successfully saved!"); 
   } 
} 
```

此函数有两个作用——如果我们不设置`fileName`变量，它将继续在图像预览部件上绘制图像，否则，它将根据图像预览部件的尺寸裁剪图像，并根据`fileName`变量将其保存到磁盘上。

当单击保存按钮时，我们将再次调用此函数。这次，我们将设置`fileName`变量为所需的目录路径和文件名，以便`QPainter`类可以正确保存图像：

```cpp
void MainWindow::on_saveButton_clicked() 
{ 
   if (canDraw) 
   { 
         if (!pix->isNull()) 
         { 
               // Save new pic from painter 
               paintImage(QCoreApplication::applicationDirPath() + 
               "/image.jpg", 0, 0); 
         } 
   } 
} 
```

最后，右键单击三个滑块中的每一个，然后选择“转到槽”。然后，选择`valueChanged(int)`并单击“确定”。

![](img/b605edff-3b8d-4ce0-9661-b08ae23bbb5d.png)

之后，我们将编写从上一步骤中得到的`slot`函数的代码：

```cpp
void MainWindow::on_scaleSlider_valueChanged(int value) 
{ 
   drawSize = imageSize * value / 100; 
   update(); 
} 

void MainWindow::on_leftSlider_valueChanged(int value) 
{ 
   drawPos.setX(value * drawSize.width() / 100 * 0.5); 
   update(); 
} 

void MainWindow::on_topSlider_valueChanged(int value) 
{ 
   drawPos.setY(value * drawSize.height() / 100 * 0.5); 
   update(); 
} 
```

比例滑块基本上是供用户在图像预览部件内调整所需比例的。左侧滑块是供用户水平移动图像的，而顶部滑块是供用户垂直移动图像的。通过组合这三个不同的滑块，用户可以在将图像上传到服务器之前，或者用于其他目的之前，调整和裁剪图像以满足他们的喜好。

如果您现在构建并运行项目，您应该能够获得以下结果：

![](img/2b7d67e4-2cc1-434a-87b7-8e09130c019a.png)

您可以单击“浏览”按钮选择要加载的 JPG 图像文件。之后，图像应该会出现在预览区域。然后，您可以移动滑块来调整裁剪大小。一旦您对结果满意，点击“保存”按钮将图像保存在当前目录中。

如果您想详细了解，请查看本书附带的示例代码。您可以在以下 GitHub 页面找到源代码：[`github.com/PacktPublishing/Hands-On-GUI-Programming-with-C-QT5`](https://github.com/PacktPublishing/Hands-On-GUI-Programming-with-C-QT5)

# 摘要

**输入和输出（I/O）**是现代计算机软件的本质。Qt 允许我们以许多直观和引人入胜的方式显示我们的数据给最终用户。除此之外，Qt 提供的事件系统使得作为程序员的我们的生活变得更加轻松，因为它倾向于通过强大的信号和槽机制自动捕获用户输入，并触发自定义行为。没有 Qt，我们将很难想出如何重新发明这个老生常谈的轮子，并最终可能会创建一个不太用户友好的产品。

在本章中，我们学习了如何利用 Qt 提供的出色功能——视图部件、对话框和文件选择对话框，用于向用户显示重要信息。此外，我们还通过一个有趣的小项目学习了如何使用 Qt 部件对用户输入进行缩放和裁剪图像。在下一章中，我们将尝试更高级（也更有趣）的内容，即使用 Qt 创建我们自己的网络浏览器！
