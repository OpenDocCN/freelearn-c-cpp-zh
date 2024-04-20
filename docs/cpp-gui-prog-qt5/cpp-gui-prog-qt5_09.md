# 第九章：摄像头模块

在通过许多难度逐渐增加的章节后，让我们尝试一些更简单和更有趣的东西！我们将学习如何通过 Qt 的多媒体模块访问我们的摄像头并使用它拍照。

在本章中，我们将涵盖以下主题：

+   Qt 多媒体模块

+   连接到摄像头

+   将摄像头图像捕获到文件

+   将摄像头视频录制到文件

您可以使用这个功能创建视频会议应用程序、安全摄像头系统等。让我们开始吧！

# Qt 多媒体模块

Qt 中的多媒体模块处理平台的多媒体功能，如媒体播放和摄像头和收音机设备的使用。这个模块涵盖了很多主题，但是在本章中我们只会专注于摄像头。

# 设置一个新项目

首先，创建一个新的 Qt Widgets 应用程序项目。

首先，我们需要打开项目文件（.pro）并添加两个关键字——`multimedia`和`multimediawidgets`：

```cpp
QT += core gui multimedia multimediawidgets 
```

通过在项目文件中检测这些关键字，Qt 在编译时将包含多媒体模块和所有与多媒体相关的部件到您的项目中。多媒体模块包括四个主要组件，列举如下：

+   音频

+   视频

+   摄像头

+   收音机

每个组件都包括一系列提供相应功能的类。通过使用这个模块，您不再需要自己实现低级别的平台特定代码。让 Qt 来为您完成这项工作。真的很简单。

在添加了多媒体模块后，让我们打开`mainwindow.ui`并将一个水平布局拖放到主窗口上，如下所示：

![](img/59be5c52-c020-4ae8-8db0-3485497ad386.png)

然后，在我们刚刚添加的水平布局中添加一个标签、下拉框（命名为`deviceSelection`）和一个按钮。之后，在下拉框和按钮之间添加一个水平间隔。完成后，选择中央窗口部件并点击工作区上方的垂直布局按钮。

然后，在上一个水平布局的底部添加另一个水平布局，右键单击它并选择转换为 | QFrame。然后，将其 sizePolicy（水平策略和垂直策略）设置为扩展。参考以下截图：

![](img/afc05a4b-5788-4b1d-ac84-33e5cd81fd92.png)

到目前为止，您的程序用户界面应该看起来像这样：

![](img/e31cada6-b831-4f34-bad4-3c3096644d80.png)

我们将布局转换为框架的原因是为了将 sizePolicy（水平策略和垂直策略）设置为扩展。但是，如果我们只是从部件框中添加一个框架部件（本质上是 QFrame），我们就无法得到所需的用于稍后附加取景器的布局组件。

接下来，再次右键单击 QFrame 并选择更改样式表。将弹出一个窗口来设置该部件的样式表。添加以下样式表代码以使背景变为黑色：

![](img/64d7d9eb-31d7-463e-b806-0c13f4de32b4.png)

这一步是可选的；我们将其背景设置为黑色，只是为了指示取景器的位置。完成后，让我们在 QFrame 上方再添加一个水平布局，如下所示：

![](img/bb45ce27-fc97-4962-a84f-f3e7f3cac303.png)

然后，在水平布局中添加两个按钮和一个水平间隔以使它们右对齐：

![](img/cd4977c9-a179-4245-a431-8f3ef6622ba3.png)

到此为止；我们已经完成了使用多媒体模块设置项目，并为下一节精心布置了用户界面。

# 连接到摄像头

最激动人心的部分来了。我们将学习如何使用 Qt 的多媒体模块访问我们的摄像头。首先，打开`mainwindow.h`并添加以下头文件：

```cpp
#include <QMainWindow> 
#include <QDebug> 
#include <QCameraInfo> 
#include <QCamera> 
#include <QCameraViewfinder> 
#include <QCameraImageCapture> 
#include <QMediaRecorder> 
#include <QUrl> 
```

接下来，添加以下变量，如下所示：

```cpp
private: 
   Ui::MainWindow *ui; 
   QCamera* camera; 
   QCameraViewfinder* viewfinder; 
   bool connected; 
```

然后，打开`mainwindow.cpp`并将以下代码添加到类构造函数中以初始化`QCamera`对象。然后，我们使用`QCameraInfo`类检索连接摄像头的列表，并将该信息填充到组合框小部件中：

```cpp
MainWindow::MainWindow(QWidget *parent) : 
   QMainWindow(parent), 
   ui(new Ui::MainWindow) 
{ 
   ui->setupUi(this); 

   connected = false; 
   camera = new QCamera(); 

   qDebug() << "Number of cameras found:" << QCameraInfo::availableCameras().count(); 

   QList<QCameraInfo> cameras = QCameraInfo::availableCameras(); 
   foreach (const QCameraInfo &cameraInfo, cameras) 
   { 
         qDebug() << "Camera info:" << cameraInfo.deviceName() << 
         cameraInfo.description() << cameraInfo.position(); 

         ui->deviceSelection->addItem(cameraInfo.description()); 
   } 
} 
```

现在构建并运行项目。之后，检查调试输出以查看计算机上检测到的摄像头。检测到的摄像头也应显示在下拉框中。如果您在支持摄像头的笔记本电脑上运行，您应该能够看到它在列表中。如果您在没有内置摄像头的系统上运行，则调试输出可能不会显示任何内容，下拉框也将保持为空。如果是这种情况，请尝试插入一个廉价的 USB 摄像头并重新运行程序：

![](img/653beee6-57b1-4705-8987-5387abe142b4.png)

之后，打开`mainwindow.ui`，右键单击连接按钮，然后选择转到槽.... 选择`clicked()`选项，然后单击确定。Qt Creator 将自动为您创建一个`slot`函数；将以下代码添加到函数中：

```cpp
void MainWindow::on_connectButton_clicked() 
{ 
   if (!connected) 
   { 
         connectCamera(); 
   } 
   else 
   { 
         camera->stop(); 
         viewfinder->deleteLater(); 
         ui->connectButton->setText("Connect"); 
         connected = false; 
   } 
} 
```

当单击连接按钮时，我们首先检查`camera`是否已连接，方法是检查`connect`变量。如果尚未连接，我们运行`connectCamera()`函数，我们将在下一步中定义。如果摄像头已连接，我们停止摄像头，删除`viewfinder`并将连接按钮的文本设置为`Connect`。最后，将`connected`变量设置为`false`。请注意，这里我们使用`deleteLater()`而不是`delete()`，这是删除内存指针的推荐方法。如果在没有运行事件循环的线程中调用`deleteLater()`，则对象将在线程完成时被销毁。

接下来，我们将在`MainWindow`类中添加一个名为`connectCamera()`的新函数。该函数如下所示：

```cpp
void MainWindow::connectCamera() 
{ 
   QList<QCameraInfo> cameras = QCameraInfo::availableCameras(); 
   foreach (const QCameraInfo &cameraInfo, cameras) 
   { 
         qDebug() << cameraInfo.description() << ui->deviceSelection-
         >currentText(); 

         if (cameraInfo.description() == ui->deviceSelection- 
         >currentText()) 
         { 
               camera = new QCamera(cameraInfo); 
               viewfinder = new QCameraViewfinder(this); 
               camera->setViewfinder(viewfinder); 
               ui->webcamLayout->addWidget(viewfinder); 

               connected = true; 
               ui->connectButton->setText("Disconnect"); 

               camera->start(); 

               return; 
         } 
   } 
} 
```

在`connectCamera()`函数中，我们重复了构造中的操作，并获取当前连接摄像头的列表。然后，我们循环遍历列表，并将摄像头的名称（存储在`description`变量中）与组合框小部件上当前选择的设备名称进行比较。

如果有匹配的名称，这意味着用户打算连接到该特定摄像头，因此我们将通过初始化`QCamera`对象和新的`QCameraViewFinder`对象来连接到该摄像头。然后，我们将`viewfinder`链接到`camera`，并将`viewfinder`添加到具有黑色背景的布局中。然后，我们将`connected`变量设置为`true`，并将连接按钮的文本设置为`Disconnect`。最后，调用`start()`函数来启动摄像头运行。

现在构建并运行项目。选择要连接的摄像头，然后单击连接按钮。您应该能够连接到摄像头并在程序中看到自己：

![](img/c6d70bb6-30f2-426f-815c-a92e80f674e0.png)

如果您的摄像头无法连接，请执行以下步骤以显示操作系统返回的任何错误。首先，打开`mainwindow.h`并添加以下`slot`函数：

```cpp
private slots: 
   void cameraError(QCamera::Error error); 
```

之后，打开`mainwindow.cpp`并将以下代码添加到`connectCamera()`函数中，将`error()`信号连接到`cameraError()`槽函数：

```cpp
void MainWindow::connectCamera() 
{ 
   QList<QCameraInfo> cameras = QCameraInfo::availableCameras(); 
   foreach (const QCameraInfo &cameraInfo, cameras) 
   { 
         qDebug() << cameraInfo.description() << ui->deviceSelection-
         >currentText(); 

         if (cameraInfo.description() == ui->deviceSelection-
         >currentText()) 
         { 
               camera = new QCamera(cameraInfo); 
               viewfinder = new QCameraViewfinder(this); 
               camera->setViewfinder(viewfinder); 
               ui->webcamLayout->addWidget(viewfinder); 

               connect(camera, SIGNAL(error(QCamera::Error)), this, 
               SLOT(cameraError(QCamera::Error))); 

               connected = true; 
               ui->connectButton->setText("Disconnect"); 

               camera->start(); 

               return; 
         } 
   } 
} 
```

`cameraError()`槽函数如下所示：

```cpp
void MainWindow::cameraError(QCamera::Error error) 
{ 
   qDebug() << "Camera error:" << error; 

   connected = false; 
   camera->stop(); 
   ui->connectButton->setText("Connect"); 
} 
```

在上述代码中，我们显示错误消息，并确保摄像头已完全停止，以防万一。通过查看错误消息，您应该能够更轻松地调试问题。

# 将摄像头图像捕获到文件

在上一节中，我们已经学习了如何使用 Qt 的多媒体模块连接到摄像头。现在，我们将尝试从摄像头中捕获静态图像并将其保存为 JPEG 文件。使用 Qt 实际上非常简单。

首先，打开`mainwindow.h`并添加以下变量：

```cpp
private: 
   Ui::MainWindow *ui; 
   QCamera* camera; 
   QCameraViewfinder* viewfinder; QCameraImageCapture* imageCapture; bool connected; 
```

然后，在`mainwindow.ui`中右键单击 Capture 按钮，选择转到槽...。然后，选择`clicked()`并按 OK。现在，在`mainwindow.cpp`中为您创建了一个新的`slot`函数。添加以下代码以从摄像头捕获图像：

```cpp
void MainWindow::on_captureButton_clicked() 
{ 
   if (connected) 
   { 
         imageCapture = new QCameraImageCapture(camera); 
         camera->setCaptureMode(QCamera::CaptureStillImage); 
         camera->searchAndLock(); 
         imageCapture->capture(qApp->applicationDirPath()); 
         camera->unlock(); 
   } 
} 
```

在前面的代码中，我们基本上创建了一个新的`QCameraImageCapture`对象，并将其媒体对象设置为活动摄像头。然后，将其捕获模式设置为静态图像。在要求`QCameraImageCapture`对象捕获图像之前，我们必须锁定摄像头，以便在捕获图像过程中设置保持不变。成功捕获图像后，您可以通过调用`camera->unlock()`来解锁它。

我们使用了`qApp->applicationDirPath()`来获取应用程序目录，以便图像将保存在可执行文件旁边。您可以将其更改为任何您想要的目录。您还可以将所需的文件名放在目录路径后面；否则，它将使用默认文件名格式按顺序保存图像，从`IMG_00000001.jpg`开始，依此类推。

# 将摄像头视频录制到文件

在学习了如何从我们的摄像头捕获静态图像之后，让我们继续学习如何录制视频。首先，打开`mainwindow.h`并添加以下变量：

```cpp
private: 
   Ui::MainWindow *ui; 
   QCamera* camera; 
   QCameraViewfinder* viewfinder; 
   QCameraImageCapture* imageCapture; 
   QMediaRecorder* recorder; 

   bool connected; 
   bool recording; 
```

接下来，再次打开`mainwindow.ui`，右键单击 Record 按钮。从菜单中选择转到槽...，然后选择`clicked()`选项，然后单击 OK 按钮。将为您创建一个`slot`函数；然后继续将以下代码添加到`slot`函数中：

```cpp
void MainWindow::on_recordButton_clicked() 
{ 
   if (connected) 
   { 
         if (!recording) 
         { 
               recorder = new QMediaRecorder(camera); 
               camera->setCaptureMode(QCamera::CaptureVideo); 
               recorder->setOutputLocation(QUrl(qApp-
               >applicationDirPath())); 
               recorder->record(); 
               recording = true; 
         } 
         else 
         { 
               recorder->stop(); 
               recording = false; 
         } 
   } 
} 
```

这次，我们使用`QMediaRecorder`来录制视频。在调用`recorder->record()`之前，我们还必须将摄像头的捕获模式设置为`QCamera::CaptureVideo`。

要检查媒体录制器在录制阶段产生的错误消息，您可以将媒体录制器的`error()`信号连接到`slot`函数，如下所示：

```cpp
void MainWindow::on_recordButton_clicked() 
{ 
   if (connected) 
   { 
         if (!recording) 
         { 
               recorder = new QMediaRecorder(camera); 
               connect(recorder, SIGNAL(error(QMediaRecorder::Error)), 
               this, SLOT(recordError(QMediaRecorder::Error))); 
               camera->setCaptureMode(QCamera::CaptureVideo); 
               recorder->setOutputLocation(QUrl(qApp-
               >applicationDirPath())); 
               recorder->record(); 
               recording = true; 
         } 
         else 
         { 
               recorder->stop(); 
               recording = false; 
         } 
   } 
} 
```

然后，只需在`slot`函数中显示错误消息：

```cpp
void MainWindow::recordError(QMediaRecorder::Error error) 
{ 
   qDebug() << errorString(); 
} 
```

请注意，在撰写本章时，`QMediaRecorder`类仅支持 macOS、Linux、移动平台和 Windows XP 上的视频录制。目前在 Windows 8 和 Windows 10 上不起作用，但将在即将推出的版本之一中移植过去。主要原因是 Qt 在 Windows 平台上使用 Microsoft 的`DirectShow` API 来录制视频，但自那时起已经从 Windows 操作系统中停用。希望在您阅读本书时，这个功能已经完全在 Qt 中为 Windows 8 和 10 实现。

如果没有，您可以使用使用`OpenCV` API 进行视频录制的第三方插件，例如**Qt 媒体编码库**（**QtMEL**）API，作为临时解决方案。请注意，QtMEL 中使用的代码与我们在本章中展示的代码完全不同。

有关 QtMEL 的更多信息，请查看以下链接：

[`kibsoft.ru`](http://kibsoft.ru)。

# 摘要

在本章中，我们学习了如何使用 Qt 连接到我们的摄像头。我们还学习了如何从摄像头捕获图像或录制视频。在下一章中，我们将学习有关网络模块，并尝试使用 Qt 制作即时通讯工具！
