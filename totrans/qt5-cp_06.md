# 第六章：连接网络和管理下载

网络模块在当今已成为关键部分，也是开发框架必备的功能；因此，Qt 为网络编程提供了 API。请耐心等待，我们将连接网络并下载文件。此外，本章还包括了线程，这是避免阻塞的重要编程技能。本章的主题如下：

+   介绍 Qt 网络编程

+   利用 `QNetworkAccessManager`

+   使用进度条

+   编写多线程应用程序

+   管理系统网络会话

# 介绍 Qt 网络编程

Qt 支持网络编程并提供大量高级 API 以简化您的开发工作。`QNetworkRequest`、`QNetworkReply` 和 `QNetworkAccessManager` 使用通用协议执行网络操作。Qt 还提供表示低级网络概念的底层类。

在本章中，我们将利用 Qt 提供的高级 API 来编写一个下载器，用于检索互联网文件并将它们保存到您的磁盘上。正如我之前提到的，该应用程序将需要 `QNetworkRequest`、`QNetworkReply` 和 `QNetworkAccessManager` 类。

首先，所有网络请求都由 `QNetworkRequest` 类表示，它是一个用于与请求相关信息的通用容器，包括头信息和加密。目前，支持 HTTP、FTP 和本地文件 URL 的上传和下载。

一旦创建了一个请求，就使用 `QNetworkAccessManager` 类来分发它并发出信号，报告进度。然后，它创建一个网络请求的回复，由 `QNetworkReply` 类表示。同时，`QNetworkReply` 提供的信号可以用来单独监控每个回复。尽管如此，一些开发者会丢弃回复的引用，并使用 `QNetworkAccessManager` 类的信号来达到这个目的。所有回复都可以同步或异步处理，因为 `QNetworkReply` 是 `QIODevice` 的子类，这意味着可以实现非阻塞操作。

下面是一个描述这些类之间关系的图示：

![介绍 Qt 网络编程](img/4615OS_06_01.jpg)

同样，网络相关的内容在网络模块中提供。要使用此模块，您需要编辑项目文件并将网络添加到 QT。现在，创建一个新的 Qt Widget 应用程序项目并编辑项目文件。在我们的 `Downloader_Demo` 示例中，`downloader_demo.pro` 项目文件如下所示：

```cpp
QT       += core gui network

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = Downloader_Demo
TEMPLATE = app

SOURCES +=  main.cpp\
            mainwindow.cpp \
            downloader.cpp \
            downloaddialog.cpp

HEADERS  += mainwindow.h \
            downloader.h \
            downloaddialog.h

FORMS    += mainwindow.ui \
            downloaddialog.ui
```

# 利用 QNetworkAccessManager

现在，我们将探讨如何编写一个能够从其他位置下载文件的应用程序。这里的“其他位置”意味着您可以从本地位置下载文件；它不一定是互联网地址，因为 Qt 也支持本地文件 URL。

首先，让我们创建一个`Downloader`类，它将使用`QNetworkAccessManager`为我们执行下载工作。以下是将`downloader.h`头文件粘贴显示的内容：

```cpp
#ifndef DOWNLOADER_H
#define DOWNLOADER_H

#include <QObject>
#include <QNetworkAccessManager>
#include <QNetworkRequest>
#include <QNetworkReply>

class Downloader : public QObject
{
  Q_OBJECT
public:
  explicit Downloader(QObject *parent = 0);

public slots:
  void download(const QUrl &url, const QString &file);

signals:
  void errorString(const QString &);
  void available(bool);
  void running(bool);
  void downloadProgress(qint64, qint64);

private:
  QNetworkAccessManager *naManager;
  QString saveFile;

  void saveToDisk(QNetworkReply *);

private slots:
  void onDownloadFinished(QNetworkReply *);
};

#endif // DOWNLOADER_H
```

我们暴露下载槽以获取 URL 和保存目标。相应地，`saveFile`用于存储保存目标。除此之外，我们使用`QNetworkAccessManager`类的`naManager`对象来管理下载过程。

让我们检查`downloader.cpp`文件中这些函数的定义。在以下构造函数中，我们将`naManager`对象的`finished`信号连接到`onDownloadFinished`槽。因此，当网络连接完成时，将通过此信号传递一个相关的`QNetworkReply`引用。

```cpp
Downloader::Downloader(QObject *parent) :
  QObject(parent)
{
  naManager = new QNetworkAccessManager(this);
  connect(naManager, &QNetworkAccessManager::finished, this, &Downloader::onDownloadFinished);
}
```

相应地，在`onDownloadFinished`槽中，我们必须谨慎处理`QNetworkReply`。如果有任何错误，这意味着下载失败，我们通过`errorString`信号暴露`errorString()`函数。否则，我们调用`saveToDisk`函数将文件保存到磁盘。然后，我们使用`deleteLater()`安全地释放`QNetworkReply`对象。正如 Qt 文档中所述，直接使用`delete`语句是不安全的；因为它已经完成，我们发出可用性和运行信号。这些信号将后来用于更改用户界面。

```cpp
void Downloader::onDownloadFinished(QNetworkReply *reply)
{
  if (reply->error() != QNetworkReply::NoError) {
    emit errorString(reply->errorString());
  }
  else {
    saveToDisk(reply);
  }
  reply->deleteLater();
  emit available(true);
  emit running(false);
}
```

在`saveToDisk`函数中，我们只是实现`QFile`将所有下载的数据保存到磁盘。这是可行的，因为`QNetworkReply`继承自`QIODevice`。因此，除了网络 API 之外，您可以将`QNetworkReply`视为一个普通的`QIODevice`对象。在这种情况下，使用`readAll()`函数获取所有数据：

```cpp
void Downloader::saveToDisk(QNetworkReply *reply)
{
  QFile f(saveFile);
  f.open(QIODevice::WriteOnly | QIODevice::Truncate);
  f.write(reply->readAll());
  f.close();
}
```

最后，让我们看看将被`MainWindow`后来使用的`download`函数内部。首先，我们将保存的文件存储到`saveFile`中。然后，我们使用`QUrl`对象`url`构建`QNetworkRequest req`。接下来，我们将`req`发送到`QNetworkAccessManager`的`naManager`对象，同时将创建的`QNetworkManager`对象的引用保存到`reply`中。之后，我们将两个`downloadProgress`信号连接在一起，这仅仅是暴露了回复的`downloadProgress`信号。最后，我们发出两个信号，分别表示可用性和运行状态。

```cpp
void Downloader::download(const QUrl &url, const QString &file)
{
  saveFile = file;
  QNetworkRequest req(url);
  QNetworkReply *reply = naManager->get(req);
  connect(reply, &QNetworkReply::downloadProgress, this, &Downloader::downloadProgress);
  emit available(false);
  emit running(true);
}
```

我们描述了`Downloader`类。现在，我们将通过导航到**Qt Designer** | **带有底部按钮的对话框**来添加`DownloadDialog`。这个类用于获取用户输入的 URL 和保存路径。对于`downloaddialog.ui`的设计，我们使用两个`QLineEdit`对象分别获取 URL 和保存路径。其中一个对象的名字是`urlEdit`，另一个是`saveAsEdit`。为了打开文件对话框让用户选择保存位置，我们在`saveAsEdit`的右侧添加了一个`QPushButton`的`saveAsButton`属性。以下截图显示了此 UI 文件的布局：

![利用 QNetworkAccessManager](img/4615OS_06_02.jpg)

您需要将此对话框的布局更改为**网格布局**。与之前类似，为了将值传递到主窗口，我们需要在**信号与槽编辑器**中删除默认的`accepted`信号和槽连接。

此类的`downloaddialog.h`头文件内容如下所示：

```cpp
#ifndef DOWNLOADDIALOG_H
#define DOWNLOADDIALOG_H

#include <QDialog>

namespace Ui {
  class DownloadDialog;
}

class DownloadDialog : public QDialog
{
  Q_OBJECT

public:
  explicit DownloadDialog(QWidget *parent = 0);
  ~DownloadDialog();

signals:
  void accepted(const QUrl &, const QString &);

private:
  Ui::DownloadDialog *ui;

private slots:
  void onButtonAccepted();
  void onSaveAsButtonClicked();
};

#endif // DOWNLOADDIALOG_H
```

如您所见，添加了一个名为`accepted`的新信号，用于传递 URL 和保存位置。此外，两个`private`槽分别用于处理按钮框的`accepted`事件和`saveAsButtonClicked`信号。

定义在`downloaddialog.cpp`源文件中，如下所示：

```cpp
#include <QFileDialog>
#include "downloaddialog.h"
#include "ui_downloaddialog.h"

DownloadDialog::DownloadDialog(QWidget *parent) :
  QDialog(parent),
  ui(new Ui::DownloadDialog)
{
  ui->setupUi(this);

  connect(ui->buttonBox, &QDialogButtonBox::accepted, this, &DownloadDialog::onButtonAccepted);
  connect(ui->saveAsButton, &QPushButton::clicked, this, &DownloadDialog::onSaveAsButtonClicked);
}

DownloadDialog::~DownloadDialog()
{
  delete ui;
}

void DownloadDialog::onButtonAccepted()
{
  emit accepted(QUrl(ui->urlEdit->text()), ui->saveAsEdit->text());
  this->accept();
}

void DownloadDialog::onSaveAsButtonClicked()
{
  QString str = QFileDialog::getSaveFileName(this, "Save As");
  if (!str.isEmpty()) {
    ui->saveAsEdit->setText(str);
  }
}
```

在`DownloadDialog`的构造函数中，仅连接信号和槽。在`onButtonAccepted`槽中，我们发出`accepted`信号，用于传递 URL 和保存路径，其中使用`urlEdit`的文本构造一个临时的`QUrl`类。然后，调用`accept`函数关闭对话框。同时，在`onSaveAsButtonClicked`槽函数中，我们使用`QFileDialog`类提供的`static`函数获取保存位置。如果`QString`返回值为空，则不执行任何操作；这意味着用户可能在文件对话框中点击了**取消**。

# 利用进度条

使用进度条直观地指示下载进度是一种方法。在 Qt 中，提供水平或垂直进度条小部件的是`QProgressBar`类。它使用`minimum`、`value`和`maximum`来确定完成百分比。百分比通过以下公式计算，`(value – minimum) / (maximum – minimum)`。我们将在示例应用程序中通过以下步骤使用这个有用的组件：

1.  返回到`MainWindow`类。

1.  在**设计**模式下编辑`mainwindow.ui`文件。

1.  拖动**按钮**并将其重命名为`newDownloadButton`，其文本为`New Download`。

1.  将**进度条**拖到`newDownloadButton`下方。

1.  将布局更改为**垂直布局**。

1.  在`progressBar`小部件的属性中取消选中`textVisible`。

推出按钮`newDownloadButton`用于弹出`DownloadDialog`以获取新的下载任务。我们需要根据以下建议对`mainwindow.h`进行一些修改：

```cpp
#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "downloader.h"
#include "downloaddialog.h"

namespace Ui {
  class MainWindow;
}

class MainWindow : public QMainWindow
{
  Q_OBJECT

public:
  explicit MainWindow(QWidget *parent = 0);
  ~MainWindow();

private:
  Ui::MainWindow *ui;
 Downloader *downloader;
 DownloadDialog *ddlg;

private slots:
  void onNewDownloadButtonPressed();
  void showMessage(const QString &);
  void onDownloadProgress(qint64, qint64);
};

#endif // MAINWINDOW_H
```

为了使用`Downloader`和`DownloadDialog`类，我们必须在`header`文件中包含它们。然后，我们可以将它们作为`private`指针包含。对于`private`槽，`onNewDownloadButtonPressed`用于处理`newDownloadButton`点击信号。而`showMessage`是一个槽函数，用于在状态栏上显示消息，最后一个`onDownloadProgress`用于更新进度条。

类似地，对于`mainwindow.cpp`源文件，我们在构造函数中连接信号和槽，如下所示：

```cpp
MainWindow::MainWindow(QWidget *parent) :
  QMainWindow(parent),
  ui(new Ui::MainWindow)
{
  ui->setupUi(this);
  ui->progressBar->setVisible(false);

  downloader = new Downloader(this);

  connect(ui->newDownloadButton, &QPushButton::clicked, this, &MainWindow::onNewDownloadButtonPressed);
  connect(downloader, &Downloader::errorString, this, &MainWindow::showMessage);
  connect(downloader, &Downloader::downloadProgress, this, &MainWindow::onDownloadProgress);
  connect(downloader, &Downloader::available, ui->newDownloadButton, &QPushButton::setEnabled);
  connect(downloader, &Downloader::running, ui->progressBar, &QProgressBar::setVisible);
}
```

在开始创建这些连接之前，我们需要隐藏进度条并创建一个新的`Downloader`类，使用`MainWindow`作为`QObject`父类。同时，在这些连接中，第一个是将`newDownloadButton`点击信号连接起来。然后，我们将下载器的`errorString`信号连接到`showMessage`，这样状态栏就可以直接显示错误消息。接下来，我们将`downloadProgress`信号连接到我们的`onDownloadProgress`处理程序。至于可用和运行信号，它们分别连接到控制`newDownloadButton`和`progressBar`的可用性和可见性。

在`onNewDownloadButtonPressed`槽函数内部，我们构建一个`DownloadDialog`对象`ddlg`，然后将`DownloadDialog`的接受信号连接到`Downloader`类的下载槽。然后使用`exec`运行对话框并阻塞事件循环。之后，我们调用`deleteLater`来安全地释放为`ddlg`分配的资源。

```cpp
void MainWindow::onNewDownloadButtonPressed()
{
  ddlg = new DownloadDialog(this);
  connect(ddlg, &DownloadDialog::accepted, downloader, &Downloader::download);
  ddlg->exec();
  ddlg->deleteLater();
}
```

对于`showMessage`槽函数，它只是简单地调用`statusBar`的`showMessage`函数，并设置三秒的超时时间，如下所示：

```cpp
void MainWindow::showMessage(const QString &es)
{
  ui->statusBar->showMessage(es, 3000);
}
```

最后，我们可以通过`onDownloadProgress`函数更新进度条，如下面的代码所示。由于`minimum`值默认为`0`，我们不需要更改它。相反，我们将`maximum`值更改为下载的总字节数，并将`value`更改为当前已下载的字节数。请注意，如果总大小未知，则总大小的值为`-1`，这将导致进度条以忙碌样式显示。

```cpp
void MainWindow::onDownloadProgress(qint64 r, qint64 t)
{
  ui->progressBar->setMaximum(t);
  ui->progressBar->setValue(r);
}
```

现在，运行应用程序并点击**新下载**按钮。将弹出**添加新下载**对话框，您可以在其中添加新的下载任务，如下所示：

![利用进度条](img/4615OS_06_03.jpg)

点击**确定**，如果没有错误；预期将显示进度条并显示当前的下载进度，如下所示：

![利用进度条](img/4615OS_06_04.jpg)

如您所见，**新下载**按钮目前不可用，因为它与`downloader`的可用信号相关联。此外，如果`downloader`没有运行，进度条甚至不会显示。

虽然这个下载器演示仍然缺少一个基本功能，即取消下载，但实际上很容易实现。在`QNetworkReply`类中有一个名为`abort`的槽函数。您可能需要存储`QNetworkReply`的引用，然后在`MainWindow`中的某个按钮被点击时调用`abort`。这里不会演示这个功能。它已经留给了您自己练习。

# 编写多线程应用程序

我敢打赌，多线程或线程对于您来说并不陌生。使用其他线程可以防止 GUI 应用程序冻结。如果应用程序在单个线程上运行，并且有一个同步的耗时操作，它将会卡住。多线程可以使应用程序运行得更加流畅。尽管大多数 Qt 网络 API 都是非阻塞的，但实践起来并不困难。

Qt 提供了一个 `QThread` 类，用于在所有支持的平台上实现多线程。换句话说，我们不需要编写特定于平台的代码，利用 POSIX 线程或 Win32 API。相反，`QThread` 提供了一种平台无关的方式来管理线程。一个 `QThread` 对象在程序中管理一个线程，该线程从 `run()` 函数开始执行，并在调用 `quit()` 或 `exit()` 时结束。

由于某些历史原因，仍然可以子类化 `QThread` 并将阻塞或耗时的代码放入重新实现的 `run()` 函数中。然而，这被认为是不正确的做法，并且不建议这样做。正确的方法是使用 `QObject::moveToThread`，稍后将会演示。

我们打算将 `Downloader::download` 函数放入一个新的线程中。实际上，是 `QNetworkAccessManager::get` 函数将被移动到另一个线程。让我们创建一个新的 C++ 类，`DownloadWorker`，其 `downloadworker.h` 头文件如下所示：

```cpp
#ifndef DOWNLOADWORKER_H
#define DOWNLOADWORKER_H

#include <QObject>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QNetworkAccessManager>

class DownloadWorker : public QObject
{
  Q_OBJECT

public slots:
  void doDownload(const QUrl &url, QNetworkAccessManager *nm);

signals:
  void downloadProgress(qint64, qint64);
};

#endif // DOWNLOADWORKER_H
```

由于我们无法创建将在另一个线程中存在的子对象，因此已从代码中删除构造函数。这是 `QThread` 几乎唯一的限制。相比之下，您可以在不同线程之间连接信号和槽而不会出现任何问题。

不要在线程之间分割父对象和子对象。父对象和子对象只能位于同一个线程中。

我们声明了 `doDownload` 插槽函数，用于为我们执行 `QNetworkAccessManager::get` 函数的工作。另一方面，`downloadProgress` 信号用于公开 `QNetworkReply` 的 `downloadProgress` 信号，就像我们之前做的那样。`downloadworker.cpp` 的内容如下所示：

```cpp
#include "downloadworker.h"

void DownloadWorker::doDownload(const QUrl &url, QNetworkAccessManager *nm)
{
  QNetworkRequest req(url);
  QNetworkReply *reply = nm->get(req);
  connect(reply, &QNetworkReply::downloadProgress, this, &DownloadWorker::downloadProgress);
}
```

上述代码是一个简单的 `worker` 类的示例。现在，我们必须将 `Downloader` 类更改为使用 `DownloadWorker` 类。`Downloader` 类的 `header` 文件 `downloader.h` 需要一些修改，如下所示：

```cpp
#ifndef DOWNLOADER_H
#define DOWNLOADER_H

#include <QObject>
#include <QNetworkAccessManager>
#include <QNetworkRequest>
#include <QNetworkReply>
#include <QThread>
#include "downloadworker.h"

class Downloader : public QObject
{
  Q_OBJECT
public:
  explicit Downloader(QObject *parent = 0);
  ~Downloader();

public slots:
  void download(const QUrl &url, const QString &file);

signals:
  void errorString(const QString &);
  void available(bool);
  void running(bool);
  void downloadProgress(qint64, qint64);

private:
  QString saveFile;
  QNetworkAccessManager *naManager;
  DownloadWorker *worker;
  QThread workerThread;

  void saveToDisk(QNetworkReply *);

private slots:
  void onDownloadFinished(QNetworkReply *);
};

#endif // DOWNLOADER_H
```

如您所见，我们已声明了一个新的 `private` 成员 `workerThread`，它是一种 `QThread` 类型。同时，还声明了一个 `DownloadWorker` 对象 `worker`。在 `downloader.cpp` 源文件中还有更多更改，如下所示：

```cpp
#include <QFile>
#include "downloader.h"

Downloader::Downloader(QObject *parent) :
  QObject(parent)
{
  naManager = new QNetworkAccessManager(this);
  worker = new DownloadWorker;
  worker->moveToThread(&workerThread);

  connect(naManager, &QNetworkAccessManager::finished, this, &Downloader::onDownloadFinished);
  connect(&workerThread, &QThread::finished, worker, &DownloadWorker::deleteLater);
  connect(worker, &DownloadWorker::downloadProgress, this, &Downloader::downloadProgress);

  workerThread.start();
}

Downloader::~Downloader()
{
  workerThread.quit();
  workerThread.wait();
}

void Downloader::download(const QUrl &url, const QString &file)
{
  saveFile = file;
  worker->doDownload(url, naManager);
  emit available(false);
  emit running(true);
}

void Downloader::onDownloadFinished(QNetworkReply *reply)
{
  if (reply->error() != QNetworkReply::NoError) {
    emit errorString(reply->errorString());
  }
  else {
    saveToDisk(reply);
  }
  reply->deleteLater();
  emit available(true);
  emit running(false);
}

void Downloader::saveToDisk(QNetworkReply *reply)
{
  QFile f(saveFile);
  f.open(QIODevice::WriteOnly | QIODevice::Truncate);
  f.write(reply->readAll());
  f.close();
}
```

在构造函数中，我们将创建一个新的 `DownloadWorker` 类，并将其移动到另一个线程 `workerThread`。通过将 `workerThread` 的 `finished` 信号连接到 `worker` 的 `deleteLater` 函数，可以在 `workerThread` 退出后安全地删除 `worker` 的资源。然后，我们需要再次公开 `downloadProgress`，因为它被移动到了 `worker` 中。最后，我们调用 `start()` 函数，以启动 `workerThread`。

作为反向操作，我们调用`quit()`函数退出`workerThread`，然后使用`wait()`确保其成功退出。

由于大量的代码已经移动到`worker`的`doDownload`函数中，我们在这里只需要调用`worker`的`doDownload`。实际上，函数调用是跨线程的，这意味着主线程不会因为那个语句而被阻塞。

由于`get`不是阻塞的，你可能感觉不到区别。然而，我相信你有一些应用程序已经冻结了，因此需要修改以适应`QThread`。始终记得只将后台阻塞操作放在另一个线程中。这主要是因为这些操作很容易从 GUI 中分离成没有父或子对象的单个对象。由于这种限制，几乎所有 GUI 对象都必须在同一个线程中，在大多数情况下是主线程。

# 管理系统网络会话

除了网络应用程序之外，Qt 还为你提供了跨平台的 API 来控制网络接口和接入点。尽管控制网络状态并不常见，但在某些情况下确实需要这样做。

首先，我想向你介绍`QNetworkConfigurationManager`。这个类管理由系统提供的网络配置。它使你能够访问它们，并在运行时检测系统的能力。网络配置由`QNetworkConfiguration`类表示，它抽象了一组配置选项，这些选项涉及如何配置网络接口以连接到目标网络。要控制网络会话，你需要使用`QNetworkSession`类。这个类为你提供了对系统接入点的控制，并允许会话管理。它还允许你控制由`QNetworkInterface`类表示的网络接口。为了帮助你理解这种关系，这里显示了一个图表：

![管理系统网络会话](img/4615OS_06_05.jpg)

如你所见，结构类似于`QNetworkAccessManager`、`QNetworkReply`和`QNetworkRequest`。特别是，还有一个另一个管理类。让我们看看在实际中如何处理这些类。

按照常规创建一个新的 Qt Widgets Application 项目。关于这个主题的示例称为`NetworkManager_Demo`。记得在你的项目文件中将网络添加到 Qt 中，就像我们在前面的示例中所做的那样。然后，在**Design**模式下编辑`mainwindow.ui`并执行以下步骤：

1.  由于我们在这个应用程序中不需要它们，请移除状态栏、菜单栏和工具栏。

1.  在**Item Views (Model-Based)**类别下添加**List View**。

1.  将**Vertical Layout**拖到`listView`的右侧。

1.  将**MainWindow**中的**Lay out**改为**Lay Out Horizontally**。

1.  将**Label**拖入`verticalLayout`并重命名为`onlineStatus`。

1.  将**进度条**拖动到`verticalLayout`中。将其`maximum`值更改为`0`并取消选中`textVisible`，以便它可以作为忙碌指示器使用。

1.  添加三个**按钮**：**刷新**、**连接**和**断开连接**；在进度条下方。它们的对象名称分别是`refreshButton`、`connectButton`和`disconnectButton`。

1.  最后，将**垂直间隔**拖动到`progressBar`和`onlineStatus`之间以分隔它们。

如同往常，我们需要在`mainwindow.h`头文件中进行一些声明，如下所示：

```cpp
#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QNetworkConfigurationManager>
#include <QNetworkConfiguration>
#include <QNetworkSession>
#include <QStandardItemModel>

namespace Ui {
  class MainWindow;
}

class MainWindow : public QMainWindow
{
  Q_OBJECT

public:
  explicit MainWindow(QWidget *parent = 0);
  ~MainWindow();

private:
  Ui::MainWindow *ui;
  QNetworkConfigurationManager *networkConfManager;
  QStandardItemModel *confListModel;

private slots:
  void onOnlineStateChanged(bool isOnline);
  void onConfigurationChanged(const QNetworkConfiguration &config);
  void onRefreshClicked();
  void onRefreshCompleted();
  void onConnectClicked();
  void onDisconnectClicked();
};

#endif // MAINWINDOW_H
```

在这种情况下，我们只利用`QNetworkConfigurationManager`、`QNetworkConfiguration`和`QNetworkSession`类来管理系统网络会话。因此，我们需要在适当的位置包含它们。

### 注意

注意，我们只需要声明一个`private`成员，在这种情况下是`networkConfManager`，`QNetworkConfigurationManager`类，因为可以从这个管理器中检索`QNetworkConfiguration`，而`QNetworkSession`绑定到`QNetworkConfiguration`。

至于`QStandardItemModel`，记得第三章中的模型/视图内容，*使用 Qt Quick 制作 RSS 阅读器*。这一章和这一章之间的唯一区别是我们之前写了 QML。然而，在这一章中，我们使用的是 C++ 应用程序。尽管工具不同，但它们共享相同的概念。`QStandardItemModel *confListModel`是 UI 文件中`listView`的确切模型。

最后，但同样重要的是，是一些槽的声明。除了按钮点击处理程序之外，前两个用于监控网络系统。这将在后面解释。

让我们编辑`mainwindow.cpp`文件，看看`MainWindow`的构造函数：

```cpp
MainWindow::MainWindow(QWidget *parent) :
  QMainWindow(parent),
  ui(new Ui::MainWindow)
{
  ui->setupUi(this);

  networkConfManager = new QNetworkConfigurationManager(this);
  confListModel = new QStandardItemModel(0, 1, this);

  ui->listView->setModel(confListModel);
  ui->progressBar->setVisible(false);

  connect(networkConfManager, &QNetworkConfigurationManager::onlineStateChanged, this, &MainWindow::onOnlineStateChanged);
  connect(networkConfManager, &QNetworkConfigurationManager::configurationChanged, this, &MainWindow::onConfigurationChanged);
  connect(networkConfManager, &QNetworkConfigurationManager::updateCompleted, this, &MainWindow::onRefreshCompleted);

  connect(ui->refreshButton, &QPushButton::clicked, this, &MainWindow::onRefreshClicked);
  connect(ui->connectButton, &QPushButton::clicked, this, &MainWindow::onConnectClicked);
  connect(ui->disconnectButton, &QPushButton::clicked, this, &MainWindow::onDisconnectClicked);

  onOnlineStateChanged(networkConfManager->isOnline());
  onRefreshClicked();
}
```

我们使用此对象，也称为`MainWindow`作为其`QObject`父对象来构建`QNetworkConfigurationManager`。然后，我们来看`confListModel`的构建。参数包括行数、列数以及`QObject`父对象，通常情况下就是它。我们将只使用一列，因为我们使用**列表视图**来显示数据。如果你使用**表格视图**，你可能需要使用更多的列。然后，我们将此模型绑定到`ui`的`listView`。在此之后，我们隐藏`progressBar`，因为它是一个`忙碌`指示器，只有在有工作运行时才会显示。在我们显式调用两个成员函数之前，将会有几个`connect`语句。其中，你可能想查看`QNetworkConfigurationManager`的信号。如果系统的`online`状态发生变化，即从`online`变为`offline`，则`onlineStateChanged`信号会被发出。每当`QNetworkConfiguration`的状态发生变化时，`configurationChanged`信号会被发出。一旦`QNetworkConfigurationManager`完成`updateConfigurations`，`updateCompleted`信号将被发出。在构造函数的末尾，我们直接调用`onOnlineStateChanged`以设置`onlineStatus`的文本。同样，调用`onRefreshClicked`可以使应用程序在启动时扫描所有网络配置。

如前所述，`onOnlineStateChanged`函数用于设置`onlineStatus`。如果系统被认为通过一个活动的网络接口连接到另一个设备，它将显示`Online`；否则，它将显示`Offline`。此函数的定义如下所示：

```cpp
void MainWindow::onOnlineStateChanged(bool isOnline)
{
  ui->onlineStatus->setText(isOnline ? "Online" : "Offline");
}
```

在以下代码中显示的`onConfigurationChanged`槽函数内部，我们更改项的背景颜色以指示配置是否活动。我们使用`findItems`函数获取`itemList`，它只包含一些与`config.name()`完全匹配的`QStandardItem`。然而，配置名称可能不是唯一的。这就是为什么我们使用一个`foreach`循环来比较`config`的标识符，它是一个唯一的字符串，其中使用`data`函数检索特定数据，其类型为`QVariant`。然后，我们使用`toString`将其转换回`QString`。`QStandardItem`使我们能够将多个数据设置到一个项中。

```cpp
void MainWindow::onConfigurationChanged(const QNetworkConfiguration &config)
{
  QList<QStandardItem *> itemList = confListModel->findItems(config.name());
  foreach (QStandardItem *i, itemList) {
    if (i->data(Qt::UserRole).toString().compare(config.identifier()) == 0) {
      if (config.state().testFlag(QNetworkConfiguration::Active)) {
        i->setBackground(QBrush(Qt::green));
      }
      else {
        i->setBackground(QBrush(Qt::NoBrush));
      }
    }
  }
}
```

这意味着我们将`identifier`存储为`Qt::UserRole`数据。它不会显示在屏幕上；相反，它作为一个特定的数据载体，在这种情况下非常有助于我们。因此，在此之后，如果它是活动的，我们将背景颜色设置为绿色；否则，不使用画笔，这意味着默认背景。请注意，`QNetworkConfiguration`的`state`函数返回`StateFlags`，这实际上是一个`QFlag`模板类，其中最佳实践是检查是否设置了标志，可以使用`testFlag`函数。

让我们检查`onRefreshClicked`函数，该函数在`onRefreshCompleted`之前显示。它将调用`QNetworkConfigurationManager *networkConfManager`的`updateConfigurations`函数。这个函数是一个耗时的函数，特别是如果它需要扫描 WLAN。因此，我们显示`progressBar`来告诉用户要有耐心，并禁用`refreshButton`，因为它正在刷新。

```cpp
void MainWindow::onRefreshClicked()
{
  ui->progressBar->setVisible(true);
  ui->refreshButton->setEnabled(false);
  networkConfManager->updateConfigurations();
}
```

更新完成后，将发出`updateCompleted`信号，并执行与`onRefreshCompleted`绑定的槽。检查以下函数，在这里我们需要清除列表。然而，我们不是调用`clear`函数，而是使用`removeRows`，这样可以保留列。如果你调用`clear`，请注意将列添加回来；否则，实际上就没有列了，这意味着没有地方放置项目。在`foreach`循环中，我们将`networkConfManager`找到的所有配置添加到`confListModel`中。正如我之前提到的，我们使用名称作为显示的`text`，而将其标识符设置为隐藏的用户角色数据。循环结束后，隐藏`progressBar`，因为刷新已完成，然后启用`refreshButton`。

```cpp
void MainWindow::onRefreshCompleted()
{
  confListModel->removeRows(0, confListModel->rowCount());
  foreach(QNetworkConfiguration c, networkConfManager->allConfigurations()) {
    QStandardItem *item = new QStandardItem(c.name());
    item->setData(QVariant(c.identifier()), Qt::UserRole);
    if (c.state().testFlag(QNetworkConfiguration::Active)) {
      item->setBackground(QBrush(Qt::green));
    }
    confListModel->appendRow(item);
  }
  ui->progressBar->setVisible(false);
  ui->refreshButton->setEnabled(true);
}
```

剩下的两个处理程序是对`connect`和`disconnect`按钮的处理。对于`connectButton`，我们显示`progressBar`，因为从路由器获取 IP 地址可能需要很长时间。然后，我们直接从`confListModel`的数据中获取`identifier`并将其保存为`QString ident`，其中`listView`的`currentIndex`函数将返回当前视图的`QModelIndex`。通过使用此索引，我们可以从模型中获取当前选中的数据。然后，我们通过调用`networkConfManager`的`configurationFromIdentifier`从`ident`构建`QNetworkConfiguration`。最后，使用`QNetworkConfiguration`构建`QNetworkSession`会话，并打开此网络会话，等待 1,000 毫秒。然后，调用`deleteLater`以安全地释放会话。最后，在这些工作完成后，隐藏`progressBar`。

```cpp
void MainWindow::onConnectClicked()
{
  ui->progressBar->setVisible(true);
  QString ident = confListModel->data(ui->listView->currentIndex(), Qt::UserRole).toString();
  QNetworkConfiguration conf = networkConfManager->configurationFromIdentifier(ident);
  QNetworkSession *session = new QNetworkSession(conf, this);
  session->open();
  session->waitForOpened(1000);
  session->deleteLater();
  ui->progressBar->setVisible(false);
}

void MainWindow::onDisconnectClicked()
{
  QString ident = confListModel->data(ui->listView->currentIndex(), Qt::UserRole).toString();
  QNetworkConfiguration conf = networkConfManager->configurationFromIdentifier(ident);
  QNetworkSession *session = new QNetworkSession(conf, this);
  if (networkConfManager->capabilities().testFlag(QNetworkConfigurationManager::SystemSessionSupport)) {
    session->close();
  }
  else {
    session->stop();
  }
  session->deleteLater();
}
```

对于`disconnectButton`，`onDisconnectClicked`处理程序将执行相反的操作，即停止网络会话。前三行与`onConnectClicked`中的相同。然而，我们需要测试平台是否支持进程外会话。正如 Qt 文档中所述，调用`close`的结果如下：

> *void QNetworkSession::close() [slot]*
> 
> *减少关联网络配置的会话计数器。如果会话计数器达到零，则关闭活动网络接口。这也意味着，只有当当前会话是最后一个打开的会话时，状态()才会从 Connected 变为 Disconnected。*

然而，如果平台不支持进程外会话，`close`函数将不会停止接口，在这种情况下，我们需要使用`stop`代替。

因此，我们调用 `networkConfManager` 的 `capabilities` 函数来检查它是否具有 `SystemSessionSupport`。如果有，则调用 `close`，否则调用 `stop`。然后，我们只需调用 `deleteLater` 来安全地释放会话。

现在，运行这个应用程序，你期望它的工作方式如下截图所示：

![管理系统网络会话](img/4615OS_06_06.jpg)

在 Windows 上，网络架构与 Unix 世界不同。因此，你可能会在列表中找到一些奇怪的配置，例如截图中的**Teredo 隧道伪接口**。不用担心这些配置，只需忽略它们！此外，没有 Qt API 允许你连接到一个新发现的加密 Wi-Fi 接入点。这是因为没有实现用于访问 WLAN 系统密码的功能。换句话说，它只能用来控制系统已知的网络会话。

# 摘要

在本章中，你有机会在掌握前几章所学内容的同时，学习 Qt 的新技能。到目前为止，你已经对 Qt 的常见架构有了深入了解，这是其子模块共享的。毕竟，网络和线程技术将使你的应用程序达到更高的水平。

在下一章中，除了解析 XML 和 JSON 文档外，我们还将用 Qt 来震撼 Android！
