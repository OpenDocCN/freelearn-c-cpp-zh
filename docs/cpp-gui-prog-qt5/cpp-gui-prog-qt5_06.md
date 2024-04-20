# 集成网络内容

在上一章中，我们学习了如何在 Qt 中使用项目视图和对话框。在这一章中，我们将学习如何将网络内容集成到我们的 Qt 应用程序中。

从 90 年代末和 21 世纪初的互联网时代开始，我们的世界变得越来越被互联网连接。自然地，运行在我们计算机上的应用程序也朝着这个方向发展。如今，我们大多数——如果不是全部——的软件在某种程度上都与互联网连接，通常是为了检索有用的信息并将其显示给用户。最简单的方法是将网络浏览器显示（也称为网络视图）嵌入到应用程序的用户界面中。这样，用户不仅可以查看信息，而且可以以美观的方式进行查看。

通过使用网络视图，开发人员可以利用其渲染能力，并使用**HTML**（超文本标记语言）和**CSS**（层叠样式表）的强大组合来装饰他们的内容。在这一章中，我们将探索 Qt 的 web 引擎模块，并创建我们自己的网络浏览器。

在这一章中，我们将涵盖以下主题：

+   创建你自己的网络浏览器

+   会话、cookie 和缓存

+   集成 JavaScript 和 C++

话不多说，让我们看看如何在 Qt 中创建我们自己的网络浏览器！

# 创建你自己的网络浏览器

从前，Qt 使用一个名为**WebKit**的不同模块在其用户界面上渲染网络内容。然而，自 5.5 版本以来，WebKit 模块已完全被弃用，并被一个名为**WebEngine**的新模块所取代。

新的 WebEngine 模块是基于谷歌构建的**Chromium**框架，它只能在 Windows 平台上的**Visual C++**编译器上运行。因此，如果你在运行 Windows，确保你已经在你的计算机上安装了**Microsoft Visual Studio**以及与你的计算机上安装的 Visual Studio 版本匹配的 Qt 的**MSVC**组件。除此之外，这个特定章节还需要 Qt WebEngine 组件。如果你在 Qt 的安装过程中跳过了这些组件，你只需要再次运行相同的安装程序并在那里安装它：

![](img/02f1ed42-5efc-43a4-b0d7-40c1610c382e.png)

# 添加网络视图小部件

一旦你准备好了，让我们开始吧！首先，打开 Qt Creator 并创建一个新的 Qt Widgets 应用程序项目。之后，打开项目（`.pro`）文件并添加以下文本以启用模块：

```cpp
QT += core gui webengine webenginewidgets 
```

如果你没有安装 MSVC 组件（在 Windows 上）或 Qt WebEngine 组件，如果你尝试构建项目，此时将会出现错误消息。如果是这种情况，请再次运行 Qt 安装程序。

接下来，打开`mainwindow.h`并添加以下头文件：

```cpp
#ifndef MAINWINDOW_H 
#define MAINWINDOW_H 

#include <QMainWindow> 
#include <QWebEngineView> 
```

之后，打开`mainwindow.h`并添加以下代码：

```cpp
private: 
   Ui::MainWindow *ui; 
 QWebEngineView* webview; 
```

然后，添加以下代码：

```cpp
MainWindow::MainWindow(QWidget *parent) : 
   QMainWindow(parent), 
   ui(new Ui::MainWindow) 
{ 
   ui->setupUi(this); 

   webview = new QWebEngineView(ui->centralWidget); 
   webview->load(QUrl("http://www.kloena.com")); 
} 
```

现在构建并运行程序，你应该看到以下结果：

![](img/d9118fa0-3227-49b7-a2bb-714e581d8784.png)

就是这么简单。你现在已经成功地在你的应用程序上放置了一个网络视图！

我们使用 C++代码创建网络视图的原因是，Qt Creator 使用的默认 Qt Designer 在小部件框中没有网络视图。前面的代码简单地创建了`QWebEngineView`对象，设置了它的父对象（在这种情况下是中央小部件），并在显示网络视图小部件之前设置了网页的 URL。如果你想使用 Qt Designer 在你的 UI 上放置一个 web 引擎视图，你必须运行独立的 Qt Designer，它位于你的 Qt 安装目录中。例如，如果你在 Windows 上运行，它位于`C:QtQt5.10.25.10.2msvc2017_64bin`。请注意，它位于支持 web 引擎的编译器名称的目录中：

![](img/0ff18712-2f53-4ce9-a66c-939f467147e6.png)

# 为网络浏览器创建用户界面

接下来，我们将把它变成一个合适的网络浏览器。首先，我们需要添加一些布局小部件，以便稍后可以放置其他小部件。将垂直布局(1)拖放到 centralWidget 上，并从对象列表中选择 centralWidget。然后，点击位于顶部的 Lay Out Vertically 按钮(2)：

![](img/bf92deaf-aeb4-44b2-9ae7-84a690721694.png)

完成后，选择新添加的垂直布局，右键单击，选择 Morph into | QFrame。我们这样做的原因是，我们希望将 web 视图小部件放在这个 QFrame 对象下，而不是中心小部件下。我们必须将布局小部件转换为 QFrame(或任何继承自 QWidget 的)对象，以便它可以*采用*web 视图作为其子对象。最后，将 QFrame 对象重命名为`webviewFrame`：

![](img/4c9ede35-a9f9-4f09-9158-12a6773bb646.png)

完成后，让我们将水平布局小部件拖放到 QFrame 对象上方。现在我们可以看到水平布局小部件和 QFrame 对象的大小是相同的，我们不希望这样。接下来，选择 QFrame 对象，并将其垂直策略设置为 Expanding：

![](img/41a44341-e29f-4869-be90-c8c26479e052.png)

然后，您会看到顶部布局小部件现在非常窄。让我们暂时将其高度设置为`20`，如下所示：

![](img/804fd92d-c9dd-4e0c-8085-4f938384d1a4.png)

完成后，将三个按钮拖放到水平布局中，现在我们可以将其顶部边距设置回`0`：

![](img/073c8c36-2d6a-4ab1-9c34-a62fdefd8695.png)

将按钮的标签分别设置为`Back`、`Forward`和`Refresh`。您也可以使用图标而不是文本显示在这些按钮上。如果您希望这样做，只需将文本属性设置为空，并从图标属性中选择一个图标。为了简单起见，我们将在本教程中只在按钮上显示文本。

接下来，在三个按钮的右侧放置一个行编辑小部件，然后再添加另一个带有`Go`标签的按钮：

![](img/7c621b7f-d77a-4bab-8c9f-a4f45f2ca11c.png)

完成后，右键单击每个按钮，然后选择转到插槽。窗口将弹出，选择 clicked()，然后按 OK。

![](img/ba8acc39-41ae-4c97-9b68-464ec3008baa.png)

这些按钮的信号函数将看起来像这样：

```cpp
void MainWindow::on_backButton_clicked() 
{ 
   webview->back(); 
} 

void MainWindow::on_forwardButton_clicked() 
{ 
   webview->forward(); 
} 

void MainWindow::on_refreshButton_clicked() 
{ 
   webview->reload(); 
} 

void MainWindow::on_goButton_clicked() 
{ 
   loadPage(); 
} 
```

基本上，`QWebEngineView`类已经为我们提供了`back()`、`forward()`和`reload()`等函数，所以我们只需在按下相应按钮时调用这些函数。然而，`loadPage()`函数是我们将编写的自定义函数。

```cpp
void MainWindow::loadPage() 
{ 
   QString url = ui->addressInput->text(); 
   if (!url.startsWith("http://") && !url.startsWith("https://")) 
   { 
         url = "http://" + url; 
   } 
   ui->addressInput->setText(url); 
   webview->load(QUrl(url)); 
} 
```

记得在`mainwindow.h`中添加`loadPage()`的声明。

我们不应该只调用`load()`函数，我认为我们应该做更多的事情。通常，用户在输入网页 URL 时不会包括`http://`(或`https://`)方案，但当我们将 URL 传递给 web 视图时，这是必需的。为了解决这个问题，我们会自动检查方案的存在。如果没有找到任何方案，我们将手动将`http://`方案添加到 URL 中。还要记得在开始时调用它来替换`load()`函数：

```cpp
MainWindow::MainWindow(QWidget *parent) : 
   QMainWindow(parent), 
   ui(new Ui::MainWindow) 
{ 
   ui->setupUi(this); 

 webview = new QWebEngineView(ui->webviewFrame); 
   loadPage(); 
} 
```

接下来，右键单击文本输入，然后选择转到插槽。然后，选择 returnPressed()，点击 OK 按钮：

![](img/a722c4f2-e653-4d97-bb37-5547c61835d4.png)

用户在完成输入网页 URL 后，按键盘上的*Return*键时，将调用此插槽函数。从逻辑上讲，用户希望页面开始加载，而不必每次输入 URL 后都要按 Go 按钮。代码非常简单，我们只需调用前面步骤中创建的`loadPage()`函数：

```cpp
void MainWindow::on_addressInput_returnPressed() 
{ 
   loadPage(); 
} 
```

现在我们已经完成了大量的代码，让我们构建并运行我们的项目，看看结果如何：

![](img/e0729566-832f-4121-9b6c-d68ddf187c50.png)

显示的结果看起来并不是很好。由于某种原因，新的 Web 视图似乎在扩展大小策略上也无法正确缩放，至少在编写本书时使用的 Qt 版本 5.10 上是如此。这个问题可能会在将来的版本中得到修复，但让我们找到解决这个问题的方法。我所做的是重写主窗口中继承的函数`paintEvent()`。在`mainwindow.h`中，只需添加函数声明，就像这样：

```cpp
public: 
   explicit MainWindow(QWidget *parent = 0); 
   ~MainWindow(); 
 void paintEvent(QPaintEvent *event); 
```

然后，在`mainwindow.cpp`中编写其定义，就像这样：

```cpp
void MainWindow::paintEvent(QPaintEvent *event) 
{ 
   QMainWindow::paintEvent(event); 
   webview->resize(ui->webviewFrame->size()); 
} 
```

当主窗口需要重新渲染其部件时（例如当窗口被调整大小时），Qt 会自动调用`paintEvent()`函数。由于这个函数在应用程序初始化时和窗口调整大小时都会被调用，我们将使用这个函数手动调整 Web 视图的大小以适应其父部件。

再次构建和运行程序，你应该能够让 Web 视图很好地适应，无论你如何调整主窗口的大小。此外，我还删除了菜单栏、工具栏和状态栏，以使整个界面看起来更整洁，因为我们在这个应用程序中没有使用这些功能：

![](img/244f4c48-0ee6-4dab-9873-101bfac0d247.png)

接下来，我们需要一个进度条来显示用户当前页面加载的进度。为此，首先我们需要在 Web 视图下方放置一个进度条部件：

![](img/d092b79b-87d6-4b15-b1a9-f91c1cfb5e94.png)

然后，在`mainwindow.h`中添加这两个槽函数：

```cpp
private slots: 
   void on_backButton_clicked(); 
   void on_forwardButton_clicked(); 
   void on_refreshButton_clicked(); 
   void on_goButton_clicked(); 
   void on_addressInput_returnPressed(); 
   void webviewLoading(int progress); 
   void webviewLoaded(); 
```

它们在`mainwindow.cpp`中的函数定义如下：

```cpp
void MainWindow::webviewLoading(int progress) 
{ 
   ui->progressBar->setValue(progress); 
} 

void MainWindow::webviewLoaded() 
{ 
   ui->addressInput->setText(webview->url().toString()); 
} 
```

第一个函数`webviewLoading()`简单地从 Web 视图中获取进度级别（以百分比值的形式）并直接提供给进度条部件。

第二个函数`webviewLoaded()`将用 Web 视图加载的网页的实际 URL 替换地址输入框上的 URL 文本。如果没有这个函数，地址输入框在你按下返回按钮或前进按钮后将不会显示正确的 URL。完成后，让我们再次编译和运行项目。结果看起来很棒：

![](img/39428977-f800-4408-bbd9-451b45561382.png)

你可能会问我，如果我不是使用 Qt 制作 Web 浏览器，这有什么实际用途？将 Web 视图嵌入到应用程序中还有许多其他用途，例如，通过精美装饰的 HTML 页面向用户展示产品的最新新闻和更新，这是游戏市场上大多数在线游戏使用的常见方法。例如，流媒体客户端也使用 Web 视图来向玩家展示最新的游戏和折扣。

这些通常被称为混合应用程序，它们将 Web 内容与本地 x 结合在一起，因此你可以利用来自 Web 的动态内容以及具有高性能和一致外观和感觉优势的本地运行的代码。

除此之外，你还可以使用它来以 HTML 格式显示可打印的报告。你可以通过调用`webview->page()->print()`或`webview->page()->printToPdf()`轻松地将报告发送到打印机，或将其保存为 PDF 文件。

要了解更多关于从 Web 视图打印的信息，请查看以下链接：[`doc.Qt.io/Qt-5/qwebenginepage.html#print.`](http://doc.Qt.io/Qt-5/qwebenginepage.html#print)

你可能还想使用 HTML 创建程序的整个用户界面，并将所有 HTML、CSS 和图像文件嵌入到 Qt 的资源包中，并从 Web 视图本地运行。可能性是无限的，唯一的限制是你的想象力！

要了解更多关于 Qt WebEngine 的信息，请查看这里的文档：[`doc.Qt.io/Qt-5/qtwebengine-overview.html.`](https://doc.Qt.io/Qt-5/qtwebengine-overview.html)

# 管理浏览器历史记录

Qt 的 Web 引擎将用户访问过的所有链接存储在一个数组结构中以供以后使用。Web 视图部件使用这个结构通过调用`back()`和`forward()`在历史记录中来回移动。

如果需要手动访问此浏览历史记录，请在`mainwindow.h`中添加以下头文件：

```cpp
#include <QWebEnginePage> 
```

然后，使用以下代码以获取以`QWebEngineHistory`对象形式的浏览历史记录：

```cpp
QWebEngineHistory* history = QWebEnginePage::history(); 
```

您可以从`history->items()`获取访问链接的完整列表，或者使用`back()`或`forward()`等函数在历史记录之间导航。要清除浏览历史记录，请调用`history->clear()`。或者，您也可以这样做：

```cpp
QWebEngineProfile::defaultProfile()->clearAllVisitedLinks();
```

要了解更多关于`QWebEngineHistory`类的信息，请访问以下链接：[`doc.Qt.io/Qt-5/qwebenginehistory.html.`](http://doc.Qt.io/Qt-5/qwebenginehistory.html)

# 会话、cookie 和缓存

与任何其他网络浏览器一样，`WebEngine`模块还支持用于存储临时数据和持久数据的机制，用于会话和缓存。会话和缓存非常重要，因为它们允许网站记住您的上次访问并将您与数据关联，例如购物车。会话、cookie 和缓存的定义如下所示：

+   **会话**：通常，会话是包含用户信息和唯一标识符的服务器端文件，从客户端发送以将它们映射到特定用户。然而，在 Qt 中，会话只是指没有任何过期日期的 cookie，因此当程序关闭时它将消失。

+   **Cookie**：Cookie 是包含用户信息或任何您想要保存的其他信息的客户端文件。与会话不同，cookie 具有过期日期，这意味着它们将保持有效，并且可以在到达过期日期之前检索，即使程序已关闭并重新打开。

+   **缓存**：缓存是一种用于加快页面加载速度的方法，通过在首次加载时将页面及其资源保存到本地磁盘。如果用户在下次访问时再次加载同一页面，Web 浏览器将重用缓存的资源，而不是等待下载完成，这可以显著加快页面加载时间。

# 管理会话和 cookie

默认情况下，`WebEngine`不保存任何 cookie，并将所有用户信息视为临时会话，这意味着当您关闭程序时，您在网页上的登录会话将自动失效。

要在 Qt 的`WebEngine`模块上启用 cookie，首先在`mainwindow.h`中添加以下头文件：

```cpp
#include <QWebEngineProfile> 
```

然后，只需调用以下函数以强制使用持久性 cookie：

```cpp
QWebEngineProfile::defaultProfile()->setPersistentCookiesPolicy(QWebEngineProfile::ForcePersistentCookies);
```

调用上述函数后，您的登录会话将在关闭程序后继续存在。要恢复为非持久性 cookie，我们只需调用：

```cpp
QWebEngineProfile::defaultProfile()->setPersistentCookiesPolicy(QWebEngineProfile::NoPersistentCookies); 
```

除此之外，您还可以更改 Qt 程序存储 cookie 的目录。要做到这一点，请将以下代码添加到您的源文件中：

```cpp
QWebEngineProfile::defaultProfile()->setPersistentStoragePath("your folder");  
```

如果出于某种原因，您想手动删除所有 cookie，请使用以下代码：

```cpp
QWebEngineProfile::defaultProfile()->cookieStore()->deleteAllCookies(); 
```

# 管理缓存

接下来，让我们谈谈缓存。在 Web 引擎模块中，有两种类型的缓存，即内存缓存和磁盘缓存。内存缓存使用计算机的内存来存储缓存，一旦关闭程序就会消失。另一方面，磁盘缓存将所有文件保存在硬盘中，因此它们将在关闭计算机后仍然存在。

默认情况下，Web 引擎模块将所有缓存保存到磁盘，如果需要将它们更改为内存缓存，请调用以下函数：

```cpp
QWebEngineProfile::defaultProfile()->setHttpCacheType(QWebEngineProfile::MemoryHttpCache); 
```

或者，您也可以通过调用完全禁用缓存：

```cpp
QWebEngineProfile::defaultProfile()->setHttpCacheType(QWebEngineProfile::NoCache); 
```

要更改程序保存缓存文件的文件夹，请调用`setCachePath()`函数：

```cpp
QWebEngineProfile::defaultProfile()->setCachePath("your folder"); 
```

最后，要删除所有缓存文件，请调用`clearHttpCache()`：

```cpp
QWebEngineProfile::defaultProfile()->clearHttpCache(); 
```

还有许多其他函数可用于更改与 cookie 和缓存相关的设置。

您可以在以下链接中了解更多信息：[`doc.Qt.io/Qt-5/qwebengineprofile.html`](https://doc.Qt.io/Qt-5/qwebengineprofile.html)

# 集成 JavaScript 和 C++

使用 Qt 的 Web 引擎模块的一个强大功能是它可以从 C++调用 JavaScript 函数，以及从 JavaScript 调用 C++函数。这使它不仅仅是一个 Web 浏览器。您可以使用它来访问 Web 浏览器标准不支持的功能，例如文件管理和硬件集成。这些功能在 W3C 标准中是不可能的；因此，无法在原生 JavaScript 中实现。但是，您可以使用 C++和 Qt 来实现这些功能，然后简单地从 JavaScript 中调用 C++函数。让我们看看如何在 Qt 中实现这一点。

# 从 C++调用 JavaScript 函数

之后，将以下代码添加到我们刚创建的 HTML 文件中：

```cpp
<!DOCTYPE html><html> 
   <head> 
      <title>Page Title</title> 
   </head> 
   <body> 
      <p>Hello World!</p> 
   </body> 
</html> 
```

这些是基本的 HTML 标记，除了显示一行文字`Hello World!`之外，什么也不显示。您可以尝试使用 Web 浏览器加载它：

![](img/84001c1c-aabc-4ff1-80bf-b05771ab51cf.png)

之后，让我们返回到我们的 Qt 项目中，然后转到文件|新建文件或项目，并创建一个 Qt 资源文件：

![](img/1d9d3c88-e775-4c5e-bd46-ab54e7a7ab81.png)

然后，打开我们刚创建的 Qt 资源文件，并在 HTML 文件中添加`/html`前缀，然后将 HTML 文件添加到资源文件中，就像这样：

![](img/8c17bb18-d44b-4989-a04d-0e8b7ea3b91e.png)

在资源文件仍然打开的情况下，右键单击`text.html`，然后选择复制资源路径到剪贴板。然后，立即更改您的 Web 视图的 URL：

```cpp
webview->load(QUrl("qrc:///html/test.html")); 
```

您可以使用刚从资源文件中复制的链接，但请确保在链接前面添加 URL 方案`qrc://`。现在构建并运行您的项目，您应该能够立即看到结果：

![](img/3908ba70-8603-4631-bfef-3994e2929583.png)

接下来，我们需要在 JavaScript 中设置一个函数，稍后将由 C++调用。我们将创建一个简单的函数，当调用时弹出一个简单的消息框并将`Hello World!`文本更改为其他内容：

```cpp
<!DOCTYPE html> 
<html> 
   <head> 
         <title>Page Title</title> 
         <script> 
               function hello() 
               { 
                  document.getElementById("myText").innerHTML =       
                  "Something happened!"; 
                  alert("Good day sir, how are you?"); 
               } 
         </script> 
   </head> 
   <body> 
         <p id="myText">Hello World!</p> 
   </body> 
</html> 
```

请注意，我已经为`Hello World!`文本添加了一个 ID，以便我们能够找到它并更改其文本。完成后，让我们再次转到我们的 Qt 项目。

让我们继续向程序 UI 添加一个按钮，当按钮被按下时，我们希望我们的 Qt 程序调用我们刚刚在 JavaScript 中创建的`hello()`函数。在 Qt 中做到这一点实际上非常容易；您只需从`QWebEnginePage`类中调用`runJavaScript()`函数，就像这样：

```cpp
void MainWindow::on_pushButton_clicked() 
{ 
   webview->page()->runJavaScript("hello();"); 
} 
```

结果非常惊人，您可以从以下截图中看到：

![](img/e860b594-575b-49c9-81dc-922f1dbb9067.png)

您可以做的远不止更改文本或调用消息框。例如，您可以在 HTML 画布中启动或停止动画，显示或隐藏 HTML 元素，触发 Ajax 事件以从 PHP 脚本中检索信息，等等...无限的可能性！

# 从 JavaScript 调用 C++函数

接下来，让我们看看如何从 JavaScript 中调用 C++函数。为了演示，我将在 Web 视图上方放置一个文本标签，并使用 JavaScript 函数更改其文本：

![](img/a962a250-b7c4-4945-93ad-ddc7ae12a78b.png)

通常，JavaScript 只能在 HTML 环境中工作，因此只能更改 HTML 元素，而不能更改 Web 视图之外的内容。但是，Qt 允许我们通过使用 Web 通道模块来做到这一点。因此，让我们打开我们的项目（`.pro`）文件并将 Web 通道模块添加到项目中：

```cpp
QT += core gui webengine webenginewidgets webchannel 
```

之后，打开`mainwindow.h`并添加`QWebChannel`头文件：

```cpp
#include <QMainWindow> 
#include <QWebEngineView> 
#include <QWebChannel> 
```

同时，我们还声明一个名为`doSomething()`的函数，并在其前面加上`Q_INVOKABLE`宏：

```cpp
Q_INVOKABLE void doSomething(); 
```

`Q_INVOKABLE`宏告诉 Qt 将函数暴露给 JavaScript 引擎，因此该函数可以从 JavaScript（以及 QML，因为 QML 也基于 JavaScript）中调用。

然后在`mainwindow.cpp`中，我们首先需要创建一个`QWebChannel`对象，并将我们的主窗口注册为 JavaScript 对象。只要从`QObject`类派生，就可以将任何 Qt 对象注册为 JavaScript 对象。

由于我们将从 JavaScript 中调用`doSomething（）`函数，因此我们必须将主窗口注册到 JavaScript 引擎。之后，我们还需要将刚刚创建的`QWebChannel`对象设置为我们的 web 视图的 web 通道。代码如下所示：

```cpp
QWebChannel* channel = new QWebChannel(this); 
channel->registerObject("mainwindow", this); 
webview->page()->setWebChannel(channel); 
```

完成后，让我们定义`doSomething（）`函数。我们只是做一些简单的事情——改变我们的 Qt GUI 上的文本标签，就这样：

```cpp
void MainWindow::doSomething() 
{ 
   ui->label->setText("This text has been changed by javascript!"); 
} 
```

我们已经完成了 C++代码，让我们打开 HTML 文件。我们需要做一些事情才能使其工作。首先，我们需要包含默认嵌入在 Qt 程序中的`qwebchannel.js`脚本，这样您就不必在 Qt 目录中搜索该文件。在`head`标签之间添加以下代码：

```cpp
<script type="text/javascript" src="img/qwebchannel.js"></script> 
```

然后，在 JavaScript 中，当文档成功被 web 视图加载时，我们创建一个`QWebChannel`对象，并将`mainwindow`变量链接到之前在 C++中注册的实际主窗口对象。这一步必须在网页加载后才能完成（通过`window.onload`回调）；否则，可能会出现创建 web 通道的问题：

```cpp
var mainwindow; 
window.onload = function() 
{ 
   new QWebChannel(Qt.webChannelTransport,function(channel) 
   { 
         mainwindow = channel.objects.mainwindow; 
   }); 
} 
```

之后，我们创建一个调用`doSomething（）`函数的 JavaScript 函数：

```cpp
function myFunction() 
{ 
   mainwindow.doSomething(); 
} 
```

最后，在 HTML 主体中添加一个按钮，并确保在按下按钮时调用`myFunction（）`：

```cpp
<body> 
   <p id="myText">Hello World!</p> 
   <button onclick="myFunction()">Do Something</button> 
</body> 
```

现在构建并运行程序，您应该能够获得以下结果：

![](img/eebf2409-5486-476d-b772-06ec44cbed98.png)

除了更改 Qt 小部件的属性之外，您可以使用此方法做很多有用的事情。例如，将文件保存到本地硬盘，从条形码扫描仪获取扫描数据等。本地和 Web 技术之间不再有障碍。但是，请格外注意此技术可能带来的安全影响。正如古话所说：

“伟大的力量带来伟大的责任。”

# 摘要

在本章中，我们已经学会了如何创建自己的网络浏览器，并使其与本地代码交互。Qt 为我们提供了 Web 通道技术，使 Qt 成为软件开发的一个非常强大的平台。

它充分利用了 Qt 的强大功能和 Web 技术的美感，这意味着在开发时你可以有更多的选择，而不仅仅局限于 Qt 的方法。我非常兴奋，迫不及待地想看看你能用这个技术实现什么！

加入我们的下一章，学习如何创建一个类似 Google Maps 的地图查看器，使用 Qt！
