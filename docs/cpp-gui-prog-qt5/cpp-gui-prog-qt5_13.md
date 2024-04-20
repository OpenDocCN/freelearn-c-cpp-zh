# 多媒体查看器

在上一章中，我们学习了如何通过云存储上传和下载文件。现在，在本章中，我们将学习如何使用 Qt 的多媒体模块打开这些文件，特别是媒体文件，如图像、音乐和视频。

在本章中，我们将涵盖以下主题：

+   重新访问多媒体模块

+   图像查看器

+   音乐播放器

+   视频播放器

让我们开始！

# 重新访问多媒体模块

在本章中，我们将再次使用多媒体模块，这在第九章中已经介绍过，*相机模块*。但是，这一次我们将使用模块的其他部分，所以我认为剖析模块并看看里面有什么是个好主意。

# 剖析模块

多媒体模块是一个非常庞大的模块，包含许多不同的部分，提供非常不同的功能和功能。主要类别如下：

+   音频

+   视频

+   相机

+   收音机

请注意，处理图像格式的类，如`QImage`、`QPixmap`等，不是多媒体模块的一部分，而是 GUI 模块的一部分。这是因为它们是 GUI 的重要组成部分，不能分开。尽管如此，我们仍将在本章中介绍`QImage`类。

在每个类别下都有一些子类别，看起来像下面这样：

+   音频：

+   音频输出

+   音频录制器

+   视频：

+   视频录制器

+   视频播放器

+   视频播放列表

+   相机：

+   相机取景器

+   相机图像捕获

+   相机视频录制器

+   收音机：

+   收音机调谐器（适用于支持模拟收音机的设备）

每个类都设计用于实现不同的目的。例如，`QSoundEffect`用于播放低延迟音频文件（如 WAV 文件）。另一方面，`QAudioOutput`将原始音频数据输出到特定的音频设备，这使您可以对音频输出进行低级控制。最后，`QMediaPlayer`是一个高级音频（和视频）播放器，支持许多不同的高延迟音频格式。在选择项目的正确类之前，您必须了解所有类之间的区别。

Qt 中的多媒体模块是一个庞大的怪兽，经常会让新手感到困惑，但如果您知道该选择哪个，它可能会带来好处。多媒体模块的另一个问题是，它可能会或可能不会在您的目标平台上工作。这是因为在所有这些类的底层都有特定平台的本机实现。如果特定平台不支持某个功能，或者尚未对其进行实现，那么您将无法使用这些功能。

有关 Qt 多媒体模块提供的不同类的更多信息，请访问以下链接：

[`doc.qt.io/qt-5.10/qtmultimedia-index.html`](https://doc.qt.io/qt-5.10/qtmultimedia-index.html)

# 图像查看器

数字图像已经成为我们日常生活中的重要组成部分。无论是自拍、毕业晚会照片还是有趣的表情包，我们花费大量时间查看数字图像。在接下来的部分中，我们将学习如何使用 Qt 和 C++创建我们自己的图像查看器。

# 为图像查看器设计用户界面

让我们开始创建我们的第一个多媒体程序。在本节中，我们将创建一个图像查看器，正如其名称所示，它会打开一个图像文件并在窗口上显示它：

1.  让我们打开 Qt Creator 并创建一个新的 Qt Widgets 应用程序项目。

1.  之后，打开`mainwindow.ui`并向中央窗口添加一个`Label`（命名为`imageDisplay`），它将用作渲染图像的画布。然后，通过选择中央窗口并按下位于画布顶部的垂直布局按钮，向 centralWidget 添加一个布局：

![](img/5e2e8370-e62f-4fe3-a04e-3b95358c4be8.png)

1.  您可以删除工具栏和状态栏以给`Label`腾出空间。此外，将中央窗口的布局边距设置为`0`：

1.  之后，双击菜单栏，添加一个文件操作，然后在其下方添加打开文件：

![](img/8cb08727-5a95-4356-8c66-787f8a8a9aeb.png)

1.  然后，在操作编辑器下，右键单击打开文件操作，选择转到槽...：

![](img/b5e3338b-a8c9-4402-af8c-a5f030de6057.png)

1.  将弹出一个窗口，询问您选择一个信号，因此选择`triggered()`，然后点击确定：

![](img/c9f8f7e1-2970-42cf-adf4-726ef91fae7b.png)

一个`slot`函数将自动为您创建，但我们将在下一部分保留它。我们已经完成了用户界面，而且真的很简单。接下来，让我们继续并开始编写我们的代码！

# 为图像查看器编写 C++代码

让我们通过以下步骤开始：

1.  首先，打开`mainwindow.h`并添加以下头文件：

```cpp
#include <QMainWindow> 
#include <QFileDialog> 
#include <QPixmap> 
#include <QPainter>
```

1.  然后，添加以下变量，称为`imageBuffer`，它将作为指向重新缩放之前的实际图像数据的指针。然后，也添加函数：

```cpp
private: 
   Ui::MainWindow *ui; 
 QPixmap* imageBuffer; 

public:
   void resizeImage();
 void paintEvent(QPaintEvent *event);

public slots:
   void on_actionOpen_triggered();
```

1.  接下来，打开`mainwindow.cpp`并在类构造函数中初始化`imageBuffer`变量：

```cpp
MainWindow::MainWindow(QWidget *parent) : 
   QMainWindow(parent), 
   ui(new Ui::MainWindow) 
{ 
   ui->setupUi(this); 
   imageBuffer = nullptr; 
} 
```

1.  之后，在上一部分中 Qt 为我们创建的`slot`函数中添加以下代码：

```cpp
void MainWindow::on_actionOpen_triggered() 
{ 
   QString fileName = QFileDialog::getOpenFileName(this, "Open Image File", qApp->applicationDirPath(), "JPG (*.jpg *.jpeg);;PNG (*.png)"); 

   if (!fileName.isEmpty()) 
   { 
         imageBuffer = new QPixmap(fileName); 
         resizeImage(); 
   } 
}
```

1.  上述代码基本上打开了文件选择对话框，并创建了一个`QPixmap`对象，其中包含所选的图像文件。完成所有这些后，它将调用`resizeImage()`函数，代码如下所示：

```cpp
void MainWindow::resizeImage() 
{ 
   if (imageBuffer != nullptr) 
   { 
         QSize size = ui->imageDisplay->size(); 
         QPixmap pixmap = imageBuffer->scaled(size, 
            Qt::KeepAspectRatio); 

         // Adjust the position of the image to the center 
         QRect rect = ui->imageDisplay->rect(); 
         rect.setX((this->size().width() - pixmap.width()) / 2); 
         rect.setY((this->size().height() - pixmap.height()) / 2); 

         QPainter painter; 
         painter.begin(this); 
         painter.drawPixmap(rect, pixmap, ui->imageDisplay->rect()); 
         painter.end(); 
   } 
} 
```

`resizeImage()`函数的作用是简单地从`imageBuffer`变量中复制图像数据，并将图像调整大小以适应窗口大小，然后显示在窗口的画布上。您可能打开的图像比屏幕分辨率大得多，我们不希望在打开这样一个大图像文件时裁剪图像。

我们使用`imageBuffer`变量的原因是，这样我们可以保留原始数据的副本，并且不会通过多次调整大小来影响图像质量。

最后，我们还在`paintEvent()`函数中调用`resizeImage()`函数。每当主窗口被调整大小或从最小化状态恢复时，`paintEvent()`将自动被调用，`resizeImage()`函数也将被调用，如下所示：

```cpp
void MainWindow::paintEvent(QPaintEvent *event) 
{ 
   resizeImage(); 
} 
```

就是这样。如果现在构建并运行项目，您应该会得到一个看起来像下面这样的漂亮的图像查看器：

![](img/fca8f4b0-48cb-4037-ba35-3518c6beac66.png)

# 音乐播放器

在接下来的部分中，我们将学习如何使用 Qt 和 C++构建自定义音乐播放器。

# 为音乐播放器设计用户界面

让我们继续下一个项目。在这个项目中，我们将使用 Qt 构建一个音频播放器。执行以下步骤：

1.  与上一个项目一样，我们将创建一个`Qt Widgets 应用程序`项目。

1.  打开`项目文件(.pro)`，并添加`multimedia`模块：

```cpp
QT += core gui multimedia 
```

1.  我们添加了`multimedia`文本，以便 Qt 在我们的项目中包含与多媒体模块相关的类。接下来，打开`mainwindow.ui`，并参考以下截图构建用户界面：![](img/e87dedd9-939f-4e7d-a19f-3bb2bad50497.png)

我们基本上在顶部添加了一个标签，然后添加了一个水平滑块和另一个标签来显示音频的当前时间。之后，我们在底部添加了三个按钮，分别是播放按钮、暂停按钮和停止按钮。这些按钮的右侧是另一个水平布局，用于控制音频音量。

如您所见，所有按钮目前都没有图标，很难分辨每个按钮的用途。

1.  要为按钮添加图标，让我们转到文件 | 新建文件或项目，并在 Qt 类别下选择 Qt 资源文件。然后，创建一个名为`icons`的前缀，并将图标图像添加到前缀中：![](img/d2370cbd-83c0-45ae-99b9-47fd81a252d7.png)

1.  之后，通过设置其图标属性并选择选择资源...，将这些图标添加到推按钮。然后，将位于音量滑块旁边的标签的`pixmap`属性设置为音量图标：![](img/ab91665f-5ce4-4f6b-b1f4-9b2772ab7fa2.png)

1.  在您将图标添加到推按钮和标签之后，用户界面应该看起来更好了！![](img/cd321651-a9b0-45bb-8e91-72c15d5d11b3.png)

我们已经完成了用户界面，让我们继续进行编程部分！

# 为音乐播放器编写 C++代码

要为音乐播放器编写 C++代码，请执行以下步骤：

1.  首先，打开`mainwindow.h`并添加以下标头：

```cpp
#include <QMainWindow> 
#include <QDebug> 
#include <QFileDialog> 
#include <QMediaPlayer> 
#include <QMediaMetaData> 
#include <QTime> 
```

1.  之后，添加`player`变量，它是一个`QMediaPlayer`指针。然后，声明我们将稍后定义的函数：

```cpp
private: 
   Ui::MainWindow *ui; 
   QMediaPlayer* player; 

public:
 void stateChanged(QMediaPlayer::State state);
 void positionChanged(qint64 position);
```

1.  接下来，打开`mainwindow.cpp`并初始化播放器变量：

```cpp
MainWindow::MainWindow(QWidget *parent) : 
   QMainWindow(parent), 
   ui(new Ui::MainWindow) 
{ 
   ui->setupUi(this); 

   player = new QMediaPlayer(this); 
   player->setVolume(ui->volume->value()); 
   connect(player, &QMediaPlayer::stateChanged, this, &MainWindow::stateChanged); 
   connect(player, &QMediaPlayer::positionChanged, this, &MainWindow::positionChanged); 
} 
```

`QMediaPlayer`类是我们的应用程序用来播放由其加载的任何音频文件的主要类。因此，我们需要知道音频播放的状态及其当前位置。我们可以通过将其`stateChanged()`和`positionChanged()`信号连接到我们的自定义`slot`函数来获取这些信息。

1.  `stateChanged()`信号允许我们获取有关音频播放的当前状态的信息。然后，我们相应地启用和禁用推按钮：

```cpp
void MainWindow::stateChanged(QMediaPlayer::State state) 
{ 
   if (state == QMediaPlayer::PlayingState) 
   { 
         ui->playButton->setEnabled(false); 
         ui->pauseButton->setEnabled(true); 
         ui->stopButton->setEnabled(true); 
   } 
   else if (state == QMediaPlayer::PausedState) 
   { 
         ui->playButton->setEnabled(true); 
         ui->pauseButton->setEnabled(false); 
         ui->stopButton->setEnabled(true); 
   } 
   else if (state == QMediaPlayer::StoppedState) 
   { 
         ui->playButton->setEnabled(true); 
         ui->pauseButton->setEnabled(false); 
         ui->stopButton->setEnabled(false); 
   } 
} 

```

1.  至于`positionChanged()`和`slot`函数，我们使用它们来设置时间轴滑块以及计时器显示：

```cpp
 void MainWindow::positionChanged(qint64 position) 
{ 
   if (ui->progressbar->maximum() != player->duration()) 
         ui->progressbar->setMaximum(player->duration()); 

   ui->progressbar->setValue(position); 

   int seconds = (position/1000) % 60; 
   int minutes = (position/60000) % 60; 
   int hours = (position/3600000) % 24; 
   QTime time(hours, minutes,seconds); 
   ui->durationDisplay->setText(time.toString()); 
} 

```

1.  完成后，打开`mainwindow.ui`，右键单击每个推按钮，然后选择转到槽...然后选择`clicked()`信号。这将为每个推按钮生成一个`slot`函数。这些`slot`函数的代码非常简单：

```cpp
void MainWindow::on_playButton_clicked() 
{  
   player->play(); 
} 

void MainWindow::on_pauseButton_clicked() 
{ 
   player->pause(); 
} 

void MainWindow::on_stopButton_clicked() 
{ 
   player->stop(); 
} 
```

1.  之后，在两个水平滑块上右键单击，并选择转到槽...然后选择`sliderMoved()`信号，然后单击确定：![](img/e66e337a-4d88-42fb-a93e-7499babbe61d.png)

1.  每当用户拖动滑块更改其位置时，都会调用`sliderMoved()`信号。我们需要将此位置发送到媒体播放器，并告诉它调整音频音量或更改当前音频位置。请注意不要将音量滑块的默认位置设置为零。考虑以下代码：

```cpp
void MainWindow::on_volume_sliderMoved(int position) 
{ 
   player->setVolume(position); 
} 

void MainWindow::on_progressbar_sliderMoved(int position) 
{ 
   player->setPosition(position); 
} 
```

1.  然后，我们需要向菜单栏添加文件和打开文件操作，就像我们在上一个示例项目中所做的那样。

1.  然后，在操作编辑器中右键单击打开文件操作，选择转到槽...之后，选择`triggered()`，让 Qt 为您生成一个`slot`函数。将以下代码添加到用于选择音频文件的`slot`函数中：

```cpp
 void MainWindow::on_actionOpen_File_triggered() 
{ 
   QString fileName = QFileDialog::getOpenFileName(this,
      "Select Audio File", qApp->applicationDirPath(), 
       "MP3 (*.mp3);;WAV (*.wav)"); 
   QFileInfo fileInfo(fileName); 

   player->setMedia(QUrl::fromLocalFile(fileName)); 

   if (player->isMetaDataAvailable()) 
   { 
         QString albumTitle = player-
         >metaData(QMediaMetaData::AlbumTitle).toString(); 
         ui->songNameDisplay->setText("Playing " + albumTitle); 
   } 
   else 
   { 
         ui->songNameDisplay->setText("Playing " + 
           fileInfo.fileName()); 
   } 

   ui->playButton->setEnabled(true); 
   ui->playButton->click(); 
} 

```

上述简单地打开一个文件选择对话框，只接受 MP3 和 WAV 文件。如果您愿意，也可以添加其他格式，但支持的格式因平台而异；因此，您应该测试以确保您想要使用的格式受支持。

之后，它将选定的音频文件发送到媒体播放器进行预加载。然后，我们尝试从元数据中获取音乐的标题，并在`Labelwidget`上显示它。但是，此功能（获取元数据）可能会或可能不会受到您的平台支持，因此，以防它不会显示，我们将其替换为音频文件名。最后，我们启用播放按钮并自动开始播放音乐。

就是这样。如果您现在构建并运行项目，您应该能够获得一个简单但完全功能的音乐播放器！

![](img/7da6ec50-1bc9-4ef7-8120-2c9b755c11fd.png)

# 视频播放器

在上一节中，我们已经学习了如何创建音频播放器。在本章中，我们将进一步改进我们的程序，并使用 Qt 和 C++创建视频播放器。

# 为视频播放器设计用户界面

下一个示例是视频播放器。由于`QMediaPlayer`还支持视频输出，我们可以使用上一个音频播放器示例中的相同用户界面和 C++代码，只需对其进行一些小的更改。

1.  首先，打开`项目文件（.pro）`并添加另一个关键字，称为`multimediawidgets`：

```cpp
QT += core gui multimedia multimediawidgets 
```

1.  然后，打开`mainwindow.ui`，在时间轴滑块上方添加一个水平布局（将其命名为`movieLayout`）。之后，右键单击布局，选择转换为 | QFrame。然后将其 sizePolicy 属性设置为 Expanding, Expanding：

![](img/c0c92ef1-df28-4145-86a6-361aae7a70db.png)

1.  之后，我们通过设置其`styleSheet`属性将 QFrame 的背景设置为黑色：

```cpp
background-color: rgb(0, 0, 0); 
```

1.  用户界面应该看起来像下面这样，然后我们就完成了：

![](img/eebc0672-a33c-47c9-98f3-237b4dc4e74c.png)

# 为视频播放器编写 C++代码

要为视频播放器编写 C++代码，我们执行以下步骤：

1.  对于`mainwindow.h`，对它的更改并不多。我们只需要在头文件中包含`QVideoWidget`：

```cpp
#include <QMainWindow> 
#include <QDebug> 
#include <QFileDialog> 
#include <QMediaPlayer> 
#include <QMediaMetaData> 
#include <QTime> 
#include <QVideoWidget> 
```

1.  然后，打开`mainwindow.cpp`。在将其添加到我们在上一步中添加的`QFrame`对象的布局之前，我们必须定义一个`QVideoWidget`对象并将其设置为视频输出目标：

```cpp
MainWindow::MainWindow(QWidget *parent) : 
   QMainWindow(parent), 
   ui(new Ui::MainWindow) 
{ 
   ui->setupUi(this); 

   player = new QMediaPlayer(this); 

   QVideoWidget* videoWidget = new QVideoWidget(this); 
   player->setVideoOutput(videoWidget); 
   ui->movieLayout->addWidget(videoWidget); 

   player->setVolume(ui->volume->value()); 
   connect(player, &QMediaPlayer::stateChanged, this, &MainWindow::stateChanged); 
   connect(player, &QMediaPlayer::positionChanged, this, &MainWindow::positionChanged); 
} 
```

1.  在`slot`函数中，当“打开文件”操作被触发时，我们只需将文件选择对话框更改为仅接受`MP4`和`MOV`格式。如果您愿意，也可以添加其他视频格式：

```cpp
QString fileName = QFileDialog::getOpenFileName(this, "Select Movie File", qApp->applicationDirPath(), "MP4 (*.mp4);;MOV (*.mov)"); 
```

就是这样。代码的其余部分与音频播放器示例几乎相同。这个示例的主要区别在于我们定义了视频输出小部件，Qt 会为我们处理其余部分。

如果我们现在构建和运行项目，应该会得到一个非常流畅的视频播放器，就像您在这里看到的那样：

![](img/45c1d750-a1a7-4261-8ce5-c7821acb069e.png)

在 Windows 系统上，有一个情况是视频播放器会抛出错误。这个问题类似于这里报告的问题：[`stackoverflow.com/questions/32436138/video-play-returns-directshowplayerservicedoseturlsource-unresolved-error-cod`](https://stackoverflow.com/questions/32436138/video-play-returns-directshowplayerservicedoseturlsource-unresolved-error-cod)

要解决此错误，只需下载并安装 K-Lite_Codec_Pack，您可以在此处找到：[`www.codecguide.com/download_k-lite_codec_pack_basic.htm`](https://www.codecguide.com/download_k-lite_codec_pack_basic.htm)。之后，视频应该可以正常播放！

# 总结

在本章中，我们已经学会了如何使用 Qt 创建自己的多媒体播放器。接下来的内容与我们通常的主题有些不同。在接下来的章节中，我们将学习如何使用 QtQuick 和 QML 创建触摸友好、移动友好和图形导向的应用程序。
