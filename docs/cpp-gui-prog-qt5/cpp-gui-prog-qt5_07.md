# 第七章：地图查看器

用户位置和地图显示是如今变得更加常见的两个功能，已经被用于各种类型的应用程序。它们通常用于后端分析和前端显示目的。

地图查看器可用于导航、附近的兴趣点查找、基于位置的服务（如叫出租车）等等。你可以使用 Qt 来实现大部分功能，但如果你要做更复杂的东西，就需要一个先进的数据库系统。

在上一章中，我们学习了如何将 Web 浏览器嵌入到应用程序中。在本章中，我们将尝试一些更有趣的东西，涵盖以下主题：

+   创建地图显示

+   标记和形状显示

+   获取用户位置

+   地理路由请求

让我们继续创建我们自己的地图查看器！

# 地图显示

Qt 位置模块为开发者提供了地理编码和导航信息的访问权限。它还可以允许用户进行地点搜索，需要从服务器或用户设备中检索数据。

目前，Qt 的地图视图不支持 C++，只支持 QML。这意味着我们只能使用 QML 脚本来改变与可视化相关的任何内容——显示地图，添加标记等等；另一方面，我们可以使用模块提供的 C++类来从数据库或服务提供商获取信息，然后通过 QML 将其显示给用户。

简单来说，**QML**（**Qt 建模语言**）是用于 Qt Quick 应用程序的用户界面标记语言。由于 QML 由 JavaScript 框架驱动，其编码语法几乎与 JavaScript 相似。如果你需要深入学习 QML 和 Qt Quick，请继续阅读第十四章，*Qt Quick 和 QML*，因为这是一个专门的章节。

有许多教程教你如何使用 Qt Quick 和 QML 语言创建一个完整的地图查看器，但并没有很多教你如何将 C++与 QML 结合使用。让我们开始吧！

# 设置 Qt 位置模块

1.  首先，创建一个新的 Qt Widgets 应用程序项目。

1.  之后，打开项目文件（`.pro`）并将以下模块添加到你的 Qt 项目中：

```cpp
QT += core gui location qml quickwidgets 
```

除了`location`模块，我们还添加了`qml`和`quickwidgets`模块，这些模块是下一节地图显示小部件所需的。这就是我们在项目中启用`Qt Location`模块所需要做的。接下来，我们将继续向项目中添加地图显示小部件。

# 创建地图显示

准备好后，让我们打开`mainwindow.ui`，并移除 menuBar、toolBar 和 statusBar，因为在这个项目中我们不需要这些东西：

![](img/c0b7b1d8-e5fe-4bcc-a390-a313956cddb6.png)

然后，从小部件框中拖动一个 QQuickWidget 到 UI 画布上。然后，点击画布顶部的水平布局按钮，为其添加布局属性：

![](img/28d2e932-e3fa-4392-a2b3-0426a394ff24.png)

然后，将中央小部件的所有边距属性设置为 0：

![](img/3a817668-af69-4e72-9b67-377db36a1240.png)

接下来，我们需要创建一个名为`mapview.qml`的新文件，方法是转到文件 | 新建文件或项目... 然后选择 Qt 类别并选择 QML 文件（Qt Quick 2）：

![](img/6a51fac1-dd56-4e67-bf24-8ac694b71266.png)

一旦 QML 文件创建完成，打开它并添加以下代码以包含`location`和`positioning`模块，以便稍后可以使用其功能：

```cpp
import QtQuick 2.0 
import QtLocation 5.3 
import QtPositioning 5.0 
```

之后，我们创建一个`Plugin`对象并命名为**osm**（**Open Street Map**），然后创建一个 Map 对象并将插件应用到其`plugin`属性上。我们还将起始坐标设置为（`40.7264175，-73.99735`），这是纽约的某个地方。除此之外，默认的`缩放级别`设置为`14`，足以让我们有一个良好的城市视图：

```cpp
Item 
{ 
    Plugin 
    { 
        id: mapPlugin 
        name: "osm" 
    } 

    Map 
    { 
        id: map 
        anchors.fill: parent 
        plugin: mapPlugin 
        center: QtPositioning.coordinate(40.7264175,-73.99735) 
        zoomLevel: 14 
    } 
} 
```

在我们能够在应用程序上显示地图之前，我们必须先创建一个资源文件并将 QML 文件添加到其中。这可以通过转到文件 | 创建新文件或项目...来完成。然后，选择 Qt 类别并选择 Qt 资源文件。

资源文件创建完成后，添加一个名为`qml`的前缀，并将 QML 文件添加到前缀中，如下所示：

![](img/ccdd7e2f-cb58-4ebd-b4f3-dc5320be49f4.png)

现在我们可以打开`mainwindow.ui`并将 QQuickWidget 的`source`属性设置为`qrc:/qml/mapview.qml`。您还可以点击源属性后面的按钮，直接从资源中选择 QML 文件。

完成后，让我们编译并运行项目，看看我们得到了什么！您也可以尝试使用鼠标在地图上平移和放大缩小：

![](img/7151aaa3-59be-4e8a-8dee-883c039e2c05.png)

即使我们可以通过使用 web 视图小部件来实现相同的结果，但这将使我们编写大量的 JavaScript 代码来显示地图。通过使用 Qt Quick，我们只需要编写几行简单的 QML 代码就可以了。

# 标记和形状显示

在前面的部分中，我们成功创建了地图显示，但这只是这个项目的开始。我们需要能够以标记或形状的形式显示自定义数据，以便用户能够理解这些数据。

# 在地图上显示位置标记

如果我告诉你我的最喜欢的餐厅位于（`40.7802655, -74.108644`），你可能无法理解。然而，如果这些坐标以位置标记的形式显示在地图视图上，你会立刻知道它在哪里。让我们看看如何向地图视图添加位置标记！

首先，我们需要一个标记图像，应该看起来像这样，或者更好的是，设计你自己的标记：

![](img/04d42016-db00-4884-a2c8-0c8edfb9052d.png)

之后，我们需要将这个图像注册到我们项目的资源文件中。用 Qt Creator 打开`resource.qrc`，创建一个名为`images`的新前缀。然后，将标记图像添加到新创建的前缀中。确保图像具有透明背景，以便在地图上显示良好。

![](img/935d7d97-990d-4453-a247-1a9a05867f34.png)

接下来，打开`mapview.qml`并用以下代码替换原来的代码：

```cpp
Item 
{ 
    id: window 

    Plugin 
    { 
        id: mapPlugin 
        name: "osm" 
    } 

    Image 
    { 
        id: icon 
        source: "qrc:///images/map-marker-icon.png" 
        sourceSize.width: 50 
        sourceSize.height: 50 
    } 

    MapQuickItem 
    { 
        id: marker 
        anchorPoint.x: marker.width / 4 
        anchorPoint.y: marker.height 
        coordinate: QtPositioning.coordinate(40.7274175,-73.99835) 

        sourceItem: icon 
    } 

    Map 
    { 
        id: map 
        anchors.fill: parent 
        plugin: mapPlugin 
        center: QtPositioning.coordinate(40.7264175,-73.99735) 
        zoomLevel: 14 

        Component.onCompleted: 
        { 
            map.addMapItem(marker) 
        } 
    } 
} 
```

在上面的代码中，我们首先添加了一个图像对象，它将用作标记的图像。由于原始图像非常庞大，我们必须通过将`sourceSize`属性设置为`50x50`来调整其大小。我们还必须将标记图像的锚点设置为图像的`中心底部`，因为那是标记的尖端所在的位置。

之后，我们创建一个`MapQuickItem`对象，它将作为标记本身。将标记图像设置为`MapQuickItem`对象的`sourceItem`，然后通过调用`map.addMapItem()`将标记添加到地图上。这个函数必须在地图创建并准备好显示之后调用，这意味着我们只能在`Component.onCompleted`事件触发后调用它。

现在我们完成了代码，让我们编译并查看结果：

![](img/6977a437-56a1-4f90-bcc9-55b5005d9fda.png)

尽管现在看起来一切都很好，但我们不想在 QML 中硬编码标记。想象一下向地图添加数百个标记，手动使用不同的代码添加每个标记是不可能的。

为了创建一个允许我们动态创建位置标记的函数，我们需要先将标记的 QML 代码从`mapview.qml`中分离出来，放到一个新的 QML 文件中。让我们创建一个名为`marker.qml`的新 QML 文件，并将其添加到资源文件中：

![](img/2dd8239b-895d-482c-985e-cde898426da4.png)

接下来，从`mapview.qml`中删除`MapQuickItem`和`Image`对象，并将其移动到`marker.qml`中：

```cpp
import QtQuick 2.0 
import QtLocation 5.3 

MapQuickItem 
{ 
    id: marker 
    anchorPoint.x: marker.width / 4 
    anchorPoint.y: marker.height 
    sourceItem: Image 
    { 
        id: icon 
        source: "qrc:///images/map-marker-icon.png" 
        sourceSize.width: 50 
        sourceSize.height: 50 
    } 
} 
```

从上述代码中，您可以看到我已经将`Image`对象与`MapQuickItem`对象合并。坐标属性也已被删除，因为我们只会在将标记放在地图上时设置它。

现在，再次打开`mapview.qml`，并将此函数添加到`Item`对象中：

```cpp
Item 
{ 
    id: window 

    Plugin 
    { 
        id: mapPlugin 
        name: "osm" 
    } 

    function addMarker(latitude, longitude) 
    { 
        var component = Qt.createComponent("qrc:///qml/marker.qml") 
        var item = component.createObject(window, { coordinate: 
        QtPositioning.coordinate(latitude, longitude) }) 
        map.addMapItem(item) 
    } 
```

从上述代码中，我们首先通过加载`marker.qml`文件创建了一个组件。然后，我们通过调用`createObject()`从组件创建了一个对象/项。在`createObject()`函数中，我们将窗口对象设置为其父对象，并将其位置设置为`addMarker()`函数提供的坐标。最后，我们将项目添加到地图中以进行渲染。

每当我们想要创建一个新的位置标记时，我们只需调用这个`addMarker()`函数。为了演示这一点，让我们通过三次调用`addMarker()`来创建三个不同的标记：

```cpp
Map 
{ 
    id: map 
    anchors.fill: parent 
    plugin: mapPlugin 
    center: QtPositioning.coordinate(40.7264175,-73.99735) 
    zoomLevel: 14 

    Component.onCompleted: 
    { 
        addMarker(40.7274175,-73.99835) 
        addMarker(40.7276432,-73.98602) 
        addMarker(40.7272175,-73.98935) 
    } 
} 
```

再次构建和运行项目，您应该能够看到类似于这样的东西：

![](img/581d34bc-8dfb-46c4-80e3-fbea4706da85.png)

我们甚至可以进一步为每个标记添加文本标签。要做到这一点，首先打开`marker.qml`，然后添加另一个名为`QtQuick.Controls`的模块：

```cpp
import QtQuick 2.0 
import QtQuick.Controls 2.0 
import QtLocation 5.3 
```

之后，向`MapQuickItem`对象添加一个自定义属性称为`labelText`：

```cpp
MapQuickItem 
{ 
    id: marker 
    anchorPoint.x: marker.width / 4 
    anchorPoint.y: marker.height 
    property string labelText 
```

一旦完成，将其`sourceItem`属性更改为：

```cpp
sourceItem: Item 
{ 
        Image 
        { 
            id: icon 
            source: "qrc:///images/map-marker-icon.png" 
            sourceSize.width: 50 
            sourceSize.height: 50 
        } 

        Rectangle 
        { 
            id: tag 
            anchors.centerIn: label 
            width: label.width + 4 
            height: label.height + 2 
            color: "black" 
        } 

        Label 
        { 
            id: label 
            anchors.centerIn: parent 
            anchors.horizontalCenterOffset: 20 
            anchors.verticalCenterOffset: -12 
            font.pixelSize: 16 
            text: labelText 
            color: "white" 
        } 
} 
```

从上述代码中，我们创建了一个`Item`对象来将多个对象组合在一起。然后，我们创建了一个`Rectangle`对象作为标签背景，以及一个文本的`Label`对象。`Label`对象的`text`属性将链接到`MapQuickItem`对象的`labelText`属性。我们可以为`addMarker()`函数添加另一个输入，用于设置`labelText`属性，如下所示：

```cpp
function addMarker(name, latitude, longitude) 
{ 
        var component = Qt.createComponent("qrc:///qml/marker.qml") 
        var item = component.createObject(window, { coordinate: QtPositioning.coordinate(latitude, longitude), labelText: name }) 
        map.addMapItem(item) 
} 
```

因此，当我们创建标记时，我们可以像这样调用`addMarker()`函数：

```cpp
Component.onCompleted: 
{ 
   addMarker("Restaurant", 40.7274175,-73.99835) 
   addMarker("My Home", 40.7276432,-73.98602) 
   addMarker("School", 40.7272175,-73.98935) 
} 
```

再次构建和运行项目，您应该会看到这个：

![](img/8b2f536b-7199-4fd7-bd36-b15745c7e285.png)

相当棒，不是吗？但是，我们还没有完成。由于我们很可能使用 C++通过 Qt 的 SQL 模块从数据库获取数据，我们需要找到一种方法从 C++调用 QML 函数。

为了实现这一点，让我们在`mapview.qml`中注释掉三个`addMarker()`函数，并打开`mainwindow.h`和以下头文件：

```cpp
#include <QQuickItem> 
#include <QQuickView> 
```

之后，打开`mainwindow.cpp`并调用`QMetaObject::invokeMethod()`函数，如下所示：

```cpp
MainWindow::MainWindow(QWidget *parent) : 
   QMainWindow(parent), 
   ui(new Ui::MainWindow) 
{ 
   ui->setupUi(this); 

 QObject* target = qobject_cast<QObject*>(ui->quickWidget->rootObject()); 
   QString functionName = "addMarker"; 

   QMetaObject::invokeMethod(target, functionName, Qt::AutoConnection, Q_ARG(QVariant, "Testing"), Q_ARG(QVariant, 40.7274175), Q_ARG(QVariant, -73.99835)); 
} 
```

上述代码可能看起来复杂，但如果我们分解并分析每个参数，实际上非常简单。上述函数的第一个参数是我们要从中调用函数的对象，在这种情况下，它是地图视图小部件中的根对象（`mapview.qml`中的`Item`对象）。接下来，我们要告诉要调用的函数名称是什么，它是`addMarker()`函数。之后，第三个参数是信号和槽系统使用的连接类型来调用此方法。对于这一点，我们将让它保持默认设置，即`Qt::AutoConnection`。其余的是`addMarker()`函数所需的参数。我们使用`Q_ARG`宏来指示数据的类型和值。

最后，再次构建和运行应用程序。您将看到一个带有标签的标记已经添加到地图上，但这次是从我们的 C++代码而不是 QML 中调用的：

![](img/3372e8f3-0227-467a-ba57-3a7033d395e1.png)

# 在地图上显示形状

除了在地图上添加标记，我们还可以在地图上绘制不同类型的形状，以指示感兴趣的区域或作为地理围栏，当目标进入或离开形状覆盖的区域时发出警告。地理围栏是在地图上定义感兴趣区域或虚拟地理边界的多边形形状，用于基于位置的服务。通常，地理围栏用于在设备进入和/或离开地理围栏时触发警报。使用地理围栏的一个很好的例子是当你需要购物提醒时，你可以在超市周围画一个地理围栏，并附上购物清单。当你（和你的手机）进入地理围栏区域时，你将收到一条提醒你要买什么的手机通知。那不是很棒吗？

有关地理围栏的更多信息，请访问：`https://en.wikipedia.org/wiki/Geo-fence`

在本章中，我们不会创建一个功能性的地理围栏，因为这是一个相当高级的话题，通常作为服务器端服务运行，用于检查和触发警报。我们只会使用 Qt 来绘制形状并在屏幕上显示它。

为了在地图视图小部件上绘制形状，我们将为每种类型的形状创建一些新的 QML 文件，并将它们添加到程序的资源中：

![](img/7ea97aa7-6e6d-4218-8515-adc2ad7578b5.png)

对于每个新创建的 QML 文件，我们将类似于位置标记的操作。对于`circle.qml`，它看起来像这样：

```cpp
import QtQuick 2.0 
import QtLocation 5.3 

MapCircle 
{ 
    property int borderWidth 
    border.width: borderWidth 
} 
```

我们只在这个文件中声明`borderWidth`，因为当调用`createCircle()`函数时，我们可以直接设置其他属性。对于`rectangle.qml`也是一样的：

```cpp
import QtQuick 2.0 
import QtLocation 5.3 

MapRectangle 
{ 
    property int borderWidth 
    border.width: borderWidth 
} 
```

对于`polygon.qml`，重复类似的步骤：

```cpp
import QtQuick 2.0 
import QtLocation 5.3 

MapPolygon 
{ 
    property int borderWidth 
    border.width: borderWidth 
} 
```

如果你愿意，你可以设置其他属性，但为了演示，我们只改变了一些属性，比如颜色、形状和边框宽度。完成后，让我们打开`mapview.qml`并定义一些函数来添加形状：

```cpp
Item 
{ 
    id: window 

    Plugin 
    { 
        id: mapPlugin 
        name: "osm" 
    } 

    function addCircle(latitude, longitude, radius, color, borderWidth) 
    { 
       var component = Qt.createComponent("qrc:///qml/circle.qml") 
       var item = component.createObject(window, { center: 
       QtPositioning.coordinate(latitude, longitude), radius: radius, 
       color: color, borderWidth: borderWidth }) 
       map.addMapItem(item) 
    } 

    function addRectangle(startLat, startLong, endLat, endLong, color, 
    borderWidth) 
    { 
        var component = Qt.createComponent("qrc:///qml/rectangle.qml") 
        var item = component.createObject(window, { topLeft: 
       QtPositioning.coordinate(startLat, startLong), bottomRight: 
       QtPositioning.coordinate(endLat, endLong), color: color, 
       borderWidth: borderWidth }) 
        map.addMapItem(item) 
    } 

    function addPolygon(path, color, borderWidth) 
    { 
        var component = Qt.createComponent("qrc:///qml/polygon.qml") 
        var item = component.createObject(window, { path: path, color: 
        color, borderWidth: borderWidth }) 
        map.addMapItem(item) 
    } 
```

这些函数与`addMarker()`函数非常相似，只是它接受稍有不同的参数，稍后传递给`createObject()`函数。之后，让我们尝试使用前面的函数创建形状：

```cpp
addCircle(40.7274175,-73.99835, 250, "green", 3); 
addRectangle(40.7274175,-73.99835, 40.7376432, -73.98602, "red", 2) 
var path = [{ latitude: 40.7324281, longitude: -73.97602 }, 
            { latitude: 40.7396432, longitude: -73.98666 }, 
            { latitude: 40.7273266, longitude: -73.99835 }, 
            { latitude: 40.7264281, longitude: -73.98602 }]; 
addPolygon(path, "blue", 3); 
```

以下是使用我们刚刚定义的函数创建的形状。我分别调用了每个函数来演示其结果，因此有三个不同的窗口：

![](img/c6a0e1b2-e88a-4a65-b6a0-c5c4ea11274d.png)

# 获取用户位置

Qt 为我们提供了一组函数来获取用户的位置信息，但只有在用户的设备支持地理定位时才能工作。这应该适用于所有现代智能手机，也可能适用于一些现代计算机。 

要使用`Qt Location`模块获取用户位置，首先让我们打开`mainwindow.h`并添加以下头文件：

```cpp
#include <QDebug> 
#include <QGeoPositionInfo> 
#include <QGeoPositionInfoSource> 
```

在同一个文件中声明以下的`slot`函数：

```cpp
private slots: 
   void positionUpdated(const QGeoPositionInfo &info); 
```

就在那之后，打开`mainwindow.cpp`并将以下代码添加到你希望开始获取用户位置的地方。出于演示目的，我只是在`MainWindow`构造函数中调用它：

```cpp
QGeoPositionInfoSource *source = QGeoPositionInfoSource::createDefaultSource(this); 
if (source) 
{ 
   connect(source, &QGeoPositionInfoSource::positionUpdated, 
         this, &MainWindow::positionUpdated); 
   source->startUpdates(); 
} 
```

然后，实现我们之前声明的`positionUpdated()`函数，就像这样：

```cpp
void MainWindow::positionUpdated(const QGeoPositionInfo &info) 
{ 
   qDebug() << "Position updated:" << info; 
} 
```

如果现在构建并运行应用程序，根据你用于运行测试的设备，你可能会或者不会获得任何位置信息。如果你收到这样的调试消息：

```cpp
serialnmea: No serial ports found
Failed to create Geoclue client interface. Geoclue error: org.freedesktop.DBus.Error.Disconnected
```

![](img/d9095bd8-9aa1-4369-998f-0cc65b69698d.png)

然后你可能需要找一些其他设备进行测试。否则，你可能会得到类似于这样的结果：

```cpp
Position updated: QGeoPositionInfo(QDateTime(2018-02-22 19:13:05.000 EST Qt::TimeSpec(LocalTime)), QGeoCoordinate(45.3333, -75.9))
```

我在这里给你留下一个作业，你可以尝试使用我们迄今为止创建的函数来完成。由于你现在可以获取你的位置坐标，尝试通过在地图显示上添加一个标记来进一步增强你的应用程序。这应该很有趣！

# 地理路由请求

还有一个重要的功能叫做**地理路由请求**，它是一组函数，帮助你绘制从 A 点到 B 点的路线（通常是最短路线）。这个功能需要一个服务提供商；在这种情况下，我们将使用**Open Street Map**（**OSM**），因为它是完全免费的。

请注意，OSM 是一个在线协作项目，这意味着如果你所在地区没有人向 OSM 服务器贡献路线数据，那么你将无法获得准确的结果。作为可选项，你也可以使用付费服务，如 Mapbox 或 ESRI。

让我们看看如何在 Qt 中实现地理路由请求！首先，将以下头文件包含到我们的`mainwindow.h`文件中：

```cpp
#include <QGeoServiceProvider>
#include <QGeoRoutingManager>
#include <QGeoRouteRequest>
#include <QGeoRouteReply>
```

之后，向`MainWindow`类添加两个槽函数，分别是`routeCalculated()`和`routeError()`：

```cpp
private slots:
    void positionUpdated(const QGeoPositionInfo &info);
    void routeCalculated(QGeoRouteReply *reply);
    void routeError(QGeoRouteReply *reply, QGeoRouteReply::Error error, const QString &errorString);
```

完成后，打开`mainwindow.cpp`并在`MainWindow`构造方法中创建一个服务提供商对象。我们将使用 OSM 服务，因此在初始化`QGeoServiceProvider`类时，我们将放置缩写`"osm"`：

```cpp
QGeoServiceProvider* serviceProvider = new QGeoServiceProvider("osm");
```

接着，我们将从刚刚创建的服务提供商对象中获取路由管理器的指针：

```cpp
QGeoRoutingManager* routingManager = serviceProvider->routingManager();
```

然后，将路由管理器的`finished()`信号和`error()`信号与我们刚刚定义的`slot`函数连接起来：

```cpp
connect(routingManager, &QGeoRoutingManager::finished, this, &MainWindow::routeCalculated);
connect(routingManager, &QGeoRoutingManager::error, this, &MainWindow::routeError);
```

当成功请求后，这些槽函数将在服务提供商回复时被触发，或者当请求失败并返回错误消息时被触发。`routeCalculated()`槽函数看起来像这样：

```cpp
void MainWindow::routeCalculated(QGeoRouteReply *reply)
{
    qDebug() << "Route Calculated";
    if (reply->routes().size() != 0)
    {
        // There could be more than 1 path
        // But we only get the first route
        QGeoRoute route = reply->routes().at(0);
        qDebug() << route.path();
    }
    reply->deleteLater();
}
```

正如你所看到的，`QGeoRouteReply`指针包含了服务提供商在成功请求后发送的路线信息。有时它会有多条路线，所以在这个例子中，我们只获取第一条路线并通过 Qt 的应用程序输出窗口显示出来。或者，你也可以使用这些坐标来绘制路径或沿着路线动画移动你的标记。

至于`routeError()`槽函数，我们将只输出服务提供商发送的错误字符串：

```cpp
void MainWindow::routeError(QGeoRouteReply *reply, QGeoRouteReply::Error error, const QString &errorString)
{
    qDebug() << "Route Error" << errorString;
    reply->deleteLater();
}
```

完成后，让我们在`MainWindow`构造方法中发起一个地理路由请求并将其发送给服务提供商：

```cpp
QGeoRouteRequest request(QGeoCoordinate(40.675895,-73.9562151), QGeoCoordinate(40.6833154,-73.987715));
routingManager->calculateRoute(request);
```

现在构建并运行项目，你应该能看到以下结果：

![](img/849cb2c6-e347-4f25-b95e-743695a488fc.png)

这里有另一个具有挑战性的任务——尝试将所有这些坐标放入一个数组中，并创建一个`addLine()`函数，该函数接受数组并绘制一系列直线，代表地理路由服务描述的路线。

自从 GPS 导航系统发明以来，地理路由一直是最重要的功能之一。希望在完成本教程后，你能够创造出一些有用的东西！

# 摘要

在本章中，我们学习了如何创建类似于谷歌地图的自己的地图视图。我们学习了如何创建地图显示，将标记和形状放在地图上，最后找到用户的位置。请注意，你也可以使用 Web 视图并调用谷歌的 JavaScript 地图 API 来创建类似的地图显示。然而，使用 QML 更简单，轻量级（我们不必加载整个 Web 引擎模块来使用地图），在移动设备和触摸屏上运行得非常好，并且也可以轻松移植到其他地图服务上。希望你能利用这些知识创造出真正令人印象深刻和有用的东西。

在下一章中，我们将探讨如何使用图形项显示信息。让我们继续吧！
