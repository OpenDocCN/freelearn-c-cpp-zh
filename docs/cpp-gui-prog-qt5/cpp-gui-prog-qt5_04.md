# 图表和图形

在上一章中，我们学习了如何使用 Qt 的`sql`模块从数据库中检索数据。有许多方法可以向用户呈现这些数据，例如以表格或图表的形式显示。在本章中，我们将学习如何进行后者——使用 Qt 的图表模块以不同类型的图表和图形呈现数据。

在本章中，我们将涵盖以下主题：

+   Qt 中的图表和图形类型

+   图表和图形实现

+   创建仪表板页面

自 Qt 5.7 以来，以前只有商业用户才能使用的几个模块已经免费提供给所有开源软件包用户，其中包括 Qt Charts 模块。因此，对于那些没有商业许可证的大多数 Qt 用户来说，这被认为是一个非常新的模块。

请注意，与大多数可在 LGPLv3 许可下使用的 Qt 模块不同，Qt Chart 模块是根据 GPLv3 许可提供的。与 LGPLv3 不同，GPLv3 许可要求您发布应用程序的源代码，同时您的应用程序也必须在 GPLv3 下获得许可。这意味着您不允许将 Qt Chart 与您的应用程序进行静态链接。它还阻止了该模块在专有软件中的使用。

要了解有关 GNU 许可的更多信息，请访问以下链接：[`www.gnu.org/licenses/gpl-faq.html.`](https://www.gnu.org/licenses/gpl-faq.html)

让我们开始吧！

# Qt 中的图表和图形类型

Qt 支持最常用的图表，并且甚至允许开发人员自定义它们的外观和感觉，以便可以用于许多不同的目的。Qt Charts 模块提供以下图表类型：

+   线性和样条线图

+   条形图

+   饼图

+   极坐标图

+   区域和散点图

+   箱形图

+   蜡烛图

# 线性和样条线图

第一种类型的图表是**线性和样条线图**。这些图表通常呈现为一系列通过线连接的点/标记。在线图中，点通过直线连接以显示变量随时间变化的情况。另一方面，样条线图与线图非常相似，只是点是通过样条线/曲线连接而不是直线：

![](img/629caf81-65dc-4ade-bca7-c83446b9563a.png)

# 条形图

**条形图**是除线图和饼图之外最常用的图表之一。条形图与线图非常相似，只是它不沿轴连接数据。相反，条形图使用单独的矩形形状来显示其数据，其中其高度由数据的值决定。这意味着数值越高，矩形形状就会变得越高：

![](img/da0e850e-1370-4f92-9b2f-59d6ff87010f.png)

# 饼图

**饼图**，顾名思义，是一种看起来像饼的图表类型。饼图以饼片的形式呈现数据。每个饼片的大小将由其值的整体百分比决定，与其余数据相比。因此，饼图通常用于显示分数、比率、百分比或一组数据的份额：

![](img/bef47cfa-ec84-4d74-9a31-7fccac977da3.jpg)

有时，饼图也可以以甜甜圈形式显示（也称为甜甜圈图）：

![](img/21decb70-9994-4aa5-9201-0d617a5577f0.png)

# 极坐标图

**极坐标图**以圆形图表的形式呈现数据，其中数据的放置基于角度和距离中心的距离，这意味着数据值越高，点距离图表中心就越远。您可以在极坐标图中显示多种类型的图表，如线性、样条线、区域和散点图来可视化数据：

![](img/12341292-4158-439e-a319-746511e60aab.png)

如果您是游戏玩家，您应该已经注意到在一些视频游戏中使用了这种类型的图表来显示游戏角色的属性：

![](img/388b962c-afac-421e-afe0-076d47706e35.png)

# 区域和散点图

**面积图**将数据显示为面积或形状，以指示体积。通常用于比较两个或多个数据集之间的差异。

![](img/fb091f5c-a7aa-4329-9faf-40f1f7e1ead0.png)

**散点图**，另一方面，用于显示一组数据点，并显示两个或多个数据集之间的非线性关系。

![](img/e2ce41e1-47a7-4fdf-832a-df1bc9b62b47.png)

# 箱线图

**箱线图**将数据呈现为四分位数，并延伸出显示值的变异性的须。箱子可能有垂直延伸的线，称为*须*。这些线表示四分位数之外的变异性，任何超出这些线或须的点都被视为异常值。箱线图最常用于统计分析，比如股票市场分析：

![](img/bd88e23e-128d-4f49-828e-6e548e6f83ce.png)

# 蜡烛图

**蜡烛图**在视觉上与箱线图非常相似，只是用于表示开盘和收盘价之间的差异，同时通过不同的颜色显示值的方向（增加或减少）。如果特定数据的值保持不变，矩形形状将根本不会显示：

![](img/1b4c98ab-bbee-4f7f-8168-c054e28be15d.png)

有关 Qt 支持的不同类型图表的更多信息，请访问以下链接：[`doc.qt.io/qt-5/qtcharts-overview.html.`](https://doc.qt.io/qt-5/qtcharts-overview.html)

Qt 支持大多数你项目中需要的图表类型。在 Qt 中实现这些图表也非常容易。让我们看看如何做到！

# 实现图表和图形

Qt 通过将复杂的绘图算法放在不同的抽象层后面，使得绘制不同类型的图表变得容易，并为我们提供了一组类和函数，可以用来轻松创建这些图表，而不需要知道绘图算法在幕后是如何工作的。这些类和函数都包含在 Qt 的图表模块中。

让我们创建一个新的 Qt Widgets 应用程序项目，并尝试在 Qt 中创建我们的第一个图表。

创建新项目后，打开项目文件（.pro）并将`charts`模块添加到项目中，如下所示：

```cpp
QT += core gui charts 
```

然后，打开`mainwindow.h`并添加以下内容以包含使用`charts`模块所需的头文件：

```cpp
#include <QtCharts> 
#include <QChartView> 
#include <QBarSet> 
#include <QBarSeries> 
```

`QtCharts`和`QtChartView`头文件对于 Qt 的`charts`模块都是必不可少的。你必须包含它们两个才能让任何类型的图表正常工作。另外两个头文件，即`QBarSet`和`QBarSeries`，在这里被使用是因为我们将创建一个条形图。根据你想创建的图表类型不同，项目中包含的头文件也会有所不同。

接下来，打开`mainwindow.ui`并将垂直布局或水平布局拖到中央窗口部件。然后，选择中央窗口部件，点击水平布局或垂直布局。布局方向并不是特别重要，因为我们这里只会创建一个图表：

![](img/4e4f032b-86fb-4548-a497-f60076f9a6d3.png)

之后，右键单击刚刚拖到中央窗口部件的布局部件，选择转换为 | QFrame。这将把布局部件更改为 QFrame 部件，同时保持其布局属性。如果从 Widget Box 创建 QFrame，它将没有我们需要的布局属性。这一步很重要，这样我们才能将其设置为稍后图表的父级：

![](img/9ef83d89-2839-43c4-9537-cb34557dddec.png)

现在打开`mainwindow.cpp`并添加以下代码：

```cpp
MainWindow::MainWindow(QWidget *parent) : 
   QMainWindow(parent), 
   ui(new Ui::MainWindow) 
{ 
   ui->setupUi(this); 

   QBarSet *set0 = new QBarSet("Jane"); 
   QBarSet *set1 = new QBarSet("John"); 
   QBarSet *set2 = new QBarSet("Axel"); 
   QBarSet *set3 = new QBarSet("Mary"); 
   QBarSet *set4 = new QBarSet("Samantha"); 

   *set0 << 10 << 20 << 30 << 40 << 50 << 60; 
   *set1 << 50 << 70 << 40 << 45 << 80 << 70; 
   *set2 << 30 << 50 << 80 << 13 << 80 << 50; 
   *set3 << 50 << 60 << 70 << 30 << 40 << 25; 
   *set4 << 90 << 70 << 50 << 30 << 16 << 42; 

   QBarSeries *series = new QBarSeries(); 
   series->append(set0); 
   series->append(set1); 
   series->append(set2); 
   series->append(set3); 
   series->append(set4); 
} 
```

上面的代码初始化了将显示在条形图中的所有类别。然后，我们还为每个类别添加了六个不同的数据项，这些数据项稍后将以条形/矩形形式表示。

`QBarSet`类表示条形图中的一组条形。它将几个条形组合成一个条形集，然后可以加标签。另一方面，`QBarSeries`表示按类别分组的一系列条形。换句话说，颜色相同的条形属于同一系列。

接下来，初始化`QChart`对象并将系列添加到其中。我们还设置了图表的标题并启用了动画：

```cpp
QChart *chart = new QChart(); 
chart->addSeries(series); 
chart->setTitle("Student Performance"); 
chart->setAnimationOptions(QChart::SeriesAnimations); 
```

之后，我们创建了一个条形图类别轴，并将其应用于条形图的*x*轴。我们使用了一个`QStringList`变量，类似于数组，但专门用于存储字符串。然后，`QBarCategoryAxis`将获取字符串列表并填充到*x*轴上：

```cpp
QStringList categories; 
categories << "Jan" << "Feb" << "Mar" << "Apr" << "May" << "Jun"; 
QBarCategoryAxis *axis = new QBarCategoryAxis(); 
axis->append(categories); 
chart->createDefaultAxes(); 
chart->setAxisX(axis, series); 
```

然后，我们为 Qt 创建一个图表视图来渲染条形图，并将其设置为主窗口中框架小部件的子级；否则，它将无法在主窗口上渲染：

```cpp
QChartView *chartView = new QChartView(chart); 
chartView->setParent(ui->verticalFrame); 
```

在 Qt Creator 中点击运行按钮，你应该会看到类似这样的东西：

![](img/ca8c434b-348e-442c-83fc-d763be3e71c3.png)

接下来，让我们做一个饼图；这真的很容易。首先，我们包括`QPieSeries`和`QPieSlice`，而不是`QBarSet`和`QBarSeries`：

```cpp
#include <QPieSeries> 
#include <QPieSlice> 
```

然后，创建一个`QPieSeries`对象，并设置每个数据的名称和值。之后，将其中一个切片设置为不同的视觉样式，并使其脱颖而出。然后，创建一个`QChart`对象，并将其与我们创建的`QPieSeries`对象链接起来：

```cpp
QPieSeries *series = new QPieSeries(); 
series->append("Jane", 10); 
series->append("Joe", 20); 
series->append("Andy", 30); 
series->append("Barbara", 40); 
series->append("Jason", 50); 

QPieSlice *slice = series->slices().at(1); 
slice->setExploded(); // Explode this chart 
slice->setLabelVisible(); // Make label visible 
slice->setPen(QPen(Qt::darkGreen, 2)); // Set line color 
slice->setBrush(Qt::green); // Set slice color 

QChart *chart = new QChart(); 
chart->addSeries(series); 
chart->setTitle("Students Performance"); 
```

最后，创建`QChartView`对象，并将其与我们刚刚创建的`QChart`对象链接起来。然后，将其设置为框架小部件的子级，我们就可以开始了！

```cpp
QChartView *chartView = new QChartView(chart);
chartView->setParent(ui->verticalFrame);
```

现在按下运行按钮，你应该能看到类似这样的东西：

![](img/a5355056-5e99-4777-804c-117005d6848d.png)

有关如何在 Qt 中创建不同图表的更多示例，请查看以下链接的示例代码：[`doc.qt.io/qt-5/qtcharts-examples.html`](https://doc.qt.io/qt-5/qtcharts-examples.html)。

现在我们已经看到使用 Qt 创建图表和图形是很容易的，让我们扩展前几章开始的项目，并为其创建一个仪表板！

# 创建仪表板页面

在上一章中，我们创建了一个功能性的登录页面，允许用户使用他们的用户名和密码登录。接下来我们需要做的是创建仪表板页面，用户成功登录后将自动跳转到该页面。

仪表板页面通常用作用户快速了解其公司、业务、项目、资产和/或其他统计数据的概览。以下图片展示了仪表板页面可能的外观：

![](img/00d8ae97-eb16-42e8-87b2-e6fc98288a8a.jpg)

正如你所看到的，仪表板页面使用了相当多的图表和图形，因为这是在不让用户感到不知所措的情况下显示大量数据的最佳方式。此外，图表和图形可以让用户轻松了解整体情况，而无需深入细节。

让我们打开之前的项目并打开`mainwindow.ui`文件。用户界面应该看起来像这样：

![](img/d94821f7-5f65-4794-824b-f819318c9b22.png)

正如你所看到的，我们现在已经有了登录页面，但我们还需要添加另一个页面作为仪表板。为了让多个页面在同一个程序中共存，并能够随时在不同页面之间切换，Qt 为我们提供了一种叫做**QStackedWidget**的东西。

堆叠窗口就像一本书，你可以不断添加更多页面，但一次只显示一页。每一页都是完全不同的 GUI，因此不会干扰堆叠窗口中的其他页面。

由于之前的登录页面并不是为堆叠窗口而设计的，我们需要对其进行一些调整。首先，从小部件框中将堆叠窗口拖放到应用程序的中央小部件下，然后，我们需要将之前在中央小部件下的所有内容移动到堆叠窗口的第一页中，我们将其重命名为 loginPage：

![](img/4e44e2d8-9594-4cea-88c6-970b2e7fb0b7.png)

接下来，将中央窗口部件的所有布局设置为`0`，这样它就完全没有边距，就像这样：

![](img/4870550e-3423-44cc-9e59-782f31959dd6.png)

在那之后，我们必须将中央窗口部件的样式表属性中的代码剪切，并粘贴到登录页面的样式表属性中。换句话说，背景图片、按钮样式和其他视觉设置现在只应用于登录页面。

完成后，切换页面时，你应该会得到两个完全不同的 GUI（仪表板页面目前为空）：

![](img/8c047da2-209c-4102-94c5-ac0e8ec76c60.png)

接下来，将网格布局拖放到仪表板页面，并将布局垂直应用到仪表板页面：

![](img/ed2ed650-4139-46e9-99f8-f2375414f6f1.png)

在那之后，将六个垂直布局拖放到网格布局中，就像这样：

![](img/cc848c1e-1aa3-4a26-a6e1-966834507a66.png)

然后，选择我们刚刚添加到网格布局中的每个垂直布局，并将其转换为 QFrame：

![](img/d5761610-be0c-4854-a859-603a27326ffc.png)

就像我们在图表实现示例中所做的那样，我们必须将布局转换为`QFrame`（或`QWidget`），以便我们可以将图表附加到它作为子对象。如果你直接从部件框中拖动`QFrame`并且不使用变形，那么`QFrame`对象就没有布局属性，因此图表可能无法调整大小以适应`QFrame`的几何形状。此外，将这些`QFrame`对象命名为`chart1`到`chart6`，因为我们将在接下来的步骤中需要它们。完成后，让我们继续编写代码。

首先，打开你的项目（`.pro`）文件，并添加`charts`模块，就像我们在本章的早期示例中所做的那样。然后，打开`mainwindow.h`并包含所有所需的头文件。这一次，我们还包括了用于创建折线图的`QLineSeries`头文件：

```cpp
#include <QtCharts> 
#include <QChartView> 

#include <QBarSet> 
#include <QBarSeries> 

#include <QPieSeries> 
#include <QPieSlice> 

#include <QLineSeries> 
```

在那之后，声明图表的指针，就像这样：

```cpp
QChartView *chartViewBar; 
QChartView *chartViewPie; 
QChartView *chartViewLine; 
```

然后，我们将添加创建柱状图的代码。这是我们之前在图表实现示例中创建的相同的柱状图，只是现在它附加到名为`chart1`的`QFrame`对象上，并在渲染时设置为启用*抗锯齿*。抗锯齿功能可以消除所有图表的锯齿状边缘，从而使渲染看起来更加平滑：

```cpp
MainWindow::MainWindow(QWidget *parent) : 
   QMainWindow(parent), 
   ui(new Ui::MainWindow) 
{ 
   ui->setupUi(this); 

   ////////BAR CHART///////////// 
   QBarSet *set0 = new QBarSet("Jane"); 
   QBarSet *set1 = new QBarSet("John"); 
   QBarSet *set2 = new QBarSet("Axel"); 
   QBarSet *set3 = new QBarSet("Mary"); 
   QBarSet *set4 = new QBarSet("Samantha"); 

   *set0 << 10 << 20 << 30 << 40 << 50 << 60; 
   *set1 << 50 << 70 << 40 << 45 << 80 << 70; 
   *set2 << 30 << 50 << 80 << 13 << 80 << 50; 
   *set3 << 50 << 60 << 70 << 30 << 40 << 25; 
   *set4 << 90 << 70 << 50 << 30 << 16 << 42; 

   QBarSeries *seriesBar = new QBarSeries(); 
   seriesBar->append(set0); 
   seriesBar->append(set1); 
   seriesBar->append(set2); 
   seriesBar->append(set3); 
   seriesBar->append(set4); 

   QChart *chartBar = new QChart(); 
   chartBar->addSeries(seriesBar); 
   chartBar->setTitle("Students Performance"); 
   chartBar->setAnimationOptions(QChart::SeriesAnimations); 

   QStringList categories; 
   categories << "Jan" << "Feb" << "Mar" << "Apr" << "May" << "Jun"; 
   QBarCategoryAxis *axis = new QBarCategoryAxis(); 
   axis->append(categories); 
   chartBar->createDefaultAxes(); 
   chartBar->setAxisX(axis, seriesBar); 

   chartViewBar = new QChartView(chartBar); 
   chartViewBar->setRenderHint(QPainter::Antialiasing); 
   chartViewBar->setParent(ui->chart1); 
} 
```

接下来，我们还要添加饼图的代码。同样，这是来自先前示例的相同饼图：

```cpp
QPieSeries *seriesPie = new QPieSeries(); 
seriesPie->append("Jane", 10); 
seriesPie->append("Joe", 20); 
seriesPie->append("Andy", 30); 
seriesPie->append("Barbara", 40); 
seriesPie->append("Jason", 50); 

QPieSlice *slice = seriesPie->slices().at(1); 
slice->setExploded(); 
slice->setLabelVisible(); 
slice->setPen(QPen(Qt::darkGreen, 2)); 
slice->setBrush(Qt::green); 

QChart *chartPie = new QChart(); 
chartPie->addSeries(seriesPie); 
chartPie->setTitle("Students Performance"); 

chartViewPie = new QChartView(chartPie); 
chartViewPie->setRenderHint(QPainter::Antialiasing); 
chartViewPie->setParent(ui->chart2); 
```

最后，我们还向仪表板添加了一个折线图，这是新的内容。代码非常简单，非常类似于饼图：

```cpp
QLineSeries *seriesLine = new QLineSeries(); 
seriesLine->append(0, 6); 
seriesLine->append(2, 4); 
seriesLine->append(3, 8); 
seriesLine->append(7, 4); 
seriesLine->append(10, 5); 
seriesLine->append(11, 10); 
seriesLine->append(13, 3); 
seriesLine->append(17, 6); 
seriesLine->append(18, 3); 
seriesLine->append(20, 2); 

QChart *chartLine = new QChart(); 
chartLine->addSeries(seriesLine); 
chartLine->createDefaultAxes(); 
chartLine->setTitle("Students Performance"); 

chartViewLine = new QChartView(chartLine); 
chartViewLine->setRenderHint(QPainter::Antialiasing); 
chartViewLine->setParent(ui->chart3); 
```

完成后，我们必须为主窗口类添加一个 resize-event 槽，并在主窗口调整大小时使图表跟随其各自父级的大小。首先，进入`mainwindow.h`并添加事件处理程序声明：

```cpp
protected: 
   void resizeEvent(QResizeEvent* event); 
```

然后，打开`mainwindow.cpp`并添加以下代码：

```cpp
void MainWindow::resizeEvent(QResizeEvent* event) 
{ 
   QMainWindow::resizeEvent(event); 

   chartViewBar->resize(chartViewBar->parentWidget()->size()); 
   chartViewPie->resize(chartViewPie->parentWidget()->size()); 
   chartViewLine->resize(chartViewLine->parentWidget()->size()); 
} 
```

请注意，必须首先调用`QMainWindow::resizeEvent(event)`，以便在调用自定义方法之前触发默认行为。`resizeEvent()`是 Qt 提供的许多事件处理程序之一，用于对其事件做出反应，例如鼠标事件、窗口事件、绘制事件等。与信号和槽机制不同，你需要替换事件处理程序的虚函数，以使其在调用事件时执行你想要的操作。

如果我们现在构建并运行项目，应该会得到类似这样的东西：

![](img/2a440a24-0d6a-4d47-b54b-a50aee5eaffc.png)

看起来相当整洁，不是吗！然而，为了简单起见，也为了不让读者感到困惑，图表都是硬编码的，并且没有使用来自数据库的任何数据。如果你打算使用来自数据库的数据，在程序启动时不要进行任何 SQL 查询，因为如果你加载的数据非常大，或者你的服务器非常慢，这将使你的程序冻结。

最好的方法是只在从登录页面切换到仪表板页面（或切换到任何其他页面时）加载数据，以便加载时间对用户不太明显。要做到这一点，右键单击堆叠窗口，然后选择转到槽。然后，选择 currentChanged(int)并单击确定。

![](img/531baa65-d083-4a70-a57c-aeff6def670a.png)

之后，Qt 会自动创建一个新的槽函数。当堆叠窗口在页面之间切换时，此函数将自动调用。您可以通过检查`arg1`变量来查看它当前切换到的页面。如果目标页面是堆叠窗口中的第一页，则`arg1`的值将为`0`，如果目标是第二页，则为`1`，依此类推。

只有在堆叠窗口显示仪表板页面时，才能提交 SQL 查询，这是第二页（`arg1`等于`1`）：

```cpp
void MainWindow::on_stackedWidget_currentChanged(int arg1) 
{ 
   if (arg1 == 1) 
   { 
      // Do it here 
   } 
} 
```

哎呀！这一章内容真是太多了！希望这一章能帮助您了解如何为您的项目创建一个美丽而丰富的页面。

# 摘要

Qt 中的图表模块是功能和视觉美学的结合。它不仅易于实现，而且无需编写非常长的代码来显示图表，而且还可以根据您的视觉要求进行定制。我们真的需要感谢 Qt 开发人员开放了这个模块，并允许非商业用户免费使用它！

在本章中，我们学习了如何使用 Qt 图表模块创建一个真正漂亮的仪表板，并在其上显示不同类型的图表。在接下来的章节中，我们将学习如何使用视图部件、对话框和文件选择对话框。
