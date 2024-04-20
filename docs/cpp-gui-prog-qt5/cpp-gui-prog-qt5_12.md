# 云存储

在上一章中，我们学习了如何使用 Qt 在屏幕上绘制图像。然而，在本章中，我们将学习完全不同的东西，即设置我们自己的文件服务器并将其链接到我们的 Qt 应用程序。

在本章中，我们将涵盖以下主题：

+   设置 FTP 服务器

+   在列表视图上显示文件列表

+   将文件上传到 FTP 服务器

+   从 FTP 服务器下载文件

让我们开始吧！

# 设置 FTP 服务器

在接下来的部分，我们将学习如何设置 FTP 服务器，该服务器存储用户上传的所有文件，并允许他们随时下载。这一部分与 Qt 无关，因此如果您已经运行了 FTP 服务器，请跳过此部分并继续本章的下一部分。

# 介绍 FTP

**FTP**是**文件传输协议**的缩写。FTP 用于在网络上从一台计算机传输文件到另一台计算机，通常是通过互联网。FTP 只是云存储技术的众多形式之一，但它也是一种简单的形式，您可以轻松地在自己的计算机上设置。

有许多不同的 FTP 服务器是由不同的人群为特定操作系统开发的。在本章的这一部分，我们将学习如何设置运行在 Windows 操作系统上的 FileZilla 服务器。如果您运行其他操作系统，如 GNU、Linux 或 macOS，还有许多其他 FTP 服务器程序可供使用，如 VSFTP 和 Pure-FTPd。

在 Debian、Ubuntu 或其他类似的 Linux 变体上，在终端上运行`sudo apt-get install vsftpd`将安装和配置 FTP 服务器。在 macOS 上，从苹果菜单中打开“系统偏好设置”，然后选择“共享”。然后，点击“服务”选项卡，选择 FTP 访问。最后，点击“启动”按钮启动 FTP 服务器。

如果您已经运行了 FTP 服务器，请跳过到下一节，我们将开始学习 C++编程。

# 下载 FileZilla

FileZilla 真的很容易设置和配置。它提供了一个完全功能的、易于使用的用户界面，不需要任何先前的操作经验。我们需要做的第一件事是下载 FileZilla。我们将按照以下步骤进行：

1.  打开浏览器，跳转到[`filezilla-project.org`](https://filezilla-project.org)。您将在主页上看到两个下载按钮。

1.  点击“下载 FileZilla 服务器”，它将带我们到下载页面：

![](img/4bfbf211-e454-4edf-bc21-d4658021c8eb.png)

1.  一旦您到达下载页面，点击“下载 FileZilla 服务器”按钮并开始下载软件。我们不会使用 FileZilla 客户端，所以您不需要下载它。一切准备就绪后，让我们继续安装软件。

1.  像大多数 Windows 软件一样，安装过程非常简单。保持一切默认，然后一直点击下一步，直到安装过程开始。安装过程最多只需要几分钟。

1.  完成后，点击“关闭”按钮，我们完成了！：

![](img/1f2e50ce-e947-4859-9cb8-7b4325f307d9.png)

# 设置 FileZilla

安装完 FileZilla 后，控制面板很可能会自动打开。

1.  由于这是您第一次启动 FileZilla，它将要求您设置服务器。将服务器 IP 地址保持为`127.0.0.1`（即**localhost**），将管理员端口设置为`14147`。

1.  输入您想要的服务器管理密码，并勾选“始终连接到此服务器”选项。点击连接，FTP 服务器现在将启动！如下截图所示：

![](img/6c47f55c-7b14-4f7e-bb5f-43c5af4b817c.png)

1.  FTP 服务器启动后，我们需要创建一个用户帐户。点击左侧的第四个图标打开“用户”对话框：

![](img/90d51fc3-b15a-43fd-afc2-b99511c8b1f6.png)

1.  然后，在常规页面下，单击窗口右侧的添加按钮。通过设置用户名创建一个帐户，然后单击确定。

1.  我们现在不必为用户设置任何组，因为用户组仅在您有许多具有相同特权设置的用户时才有用，因为这样可以更容易地一次更改所有用户的设置或将用户移动到不同的组中。创建用户后，选中密码选项并输入所需的密码。将密码放在您的 FTP 帐户上始终是一个好习惯：

![](img/93277ddc-baa0-4121-a281-25dcef5ebd9e.png)

1.  之后，我们将继续到共享文件夹页面，并为我们新创建的用户添加一个共享目录。

1.  确保删除和追加选项已选中，以便可以替换具有相同名称的文件。我们将在稍后使用它来更新我们的文件列表：

![](img/51573cbf-7f90-4144-8677-cba2ec6bad13.png)

1.  如果单击从左起的第三个图标，将出现 FileZilla 服务器选项对话框。您基本上可以在这里配置一切以满足您的需求。例如，如果您不想使用默认端口号`21`，您可以在选项窗口中简单地更改它，在常规设置页面下：

![](img/697d6dcd-114e-4477-bc2e-7437b0905d2a.png)

1.  您还可以在速度限制页面为所有用户或特定用户设置速度限制。这可以防止您的服务器在许多用户同时下载大文件时性能下降：

![](img/1868a1d7-c12b-482b-9536-b844f2e7d50c.png)

接下来，让我们继续创建我们的 Qt 项目！

# 在列表视图上显示文件列表

在上一节中，我们成功地设置了一个 FTP 服务器并使其保持运行。在接下来的部分中，我们将学习如何创建一个 FTP 客户端程序，该程序显示文件列表，将文件上传到 FTP 服务器，最后从中下载文件。

# 设置项目

像往常一样，让我们使用**Qt Creator**创建一个新项目。以下步骤将有所帮助：

1.  我们可以通过转到文件|新文件或项目并选择 Qt 小部件应用程序来创建一个新项目。

1.  创建项目后，打开您的项目（`.pro`）文件，并添加`network`关键字，以便 Qt 知道您的项目需要网络模块：

```cpp
QT += core gui network
```

# 设置用户界面

之后，打开`mainwindow.ui`并执行以下步骤来设计用户界面的上半部分以上传文件：

1.  放置一个标签，上面写着上传文件：放在其他小部件的顶部。

1.  在标签下方放置一个水平布局和两个按钮，分别写着打开和上传。

1.  在水平布局下放置一个进度条。

1.  在底部放置一个水平线，然后是垂直间隔器：

![](img/8a026044-0639-4540-af80-4de768b78ffa.jpg)

接下来，我们将构建用户界面的底部部分，用于下载文件：

![](img/3130468f-e520-4536-8be3-2b8c472857ee.jpg)

这次，我们的用户界面与上半部分非常相似，只是我们在第二个进度条之前添加了一个列表视图来显示文件列表。我们将所有内容放在同一页上，以便更简单和不易混淆地解释这个示例程序。

# 显示文件列表

接下来，我们将学习如何保存并显示 FTP 服务器上的文件列表。实际上，FTP 服务器默认提供文件列表，并且 Qt 能够在旧版本中使用`qtftp`模块显示它。但是，自从版本 5 以来，Qt 已经完全放弃了`qtftp`模块，这个功能不再存在。

如果您仍然对旧的`qtftp`模块感兴趣，您仍然可以通过访问以下链接在 GitHub 上获取其源代码：[`github.com/qt/qtftp`](https://github.com/qt/qtftp)

在 Qt 中，我们使用`QNetworkAccessManager`类与我们的 FTP 服务器通信，因此不再使用专门为 FTP 设计的功能。但是，不用担心，我们将研究一些其他替代方法来实现相同的结果。

在我看来，最好的方法是使用在线数据库来存储文件列表及其信息（文件大小、格式、状态等）。如果您有兴趣学习如何将 Qt 应用程序连接到数据库，请参阅第三章，*数据库连接*。然而，为了简单起见，我们将使用另一种方法，它可以正常工作，但不够安全——直接将文件名保存在文本文件中，并将其存储在 FTP 服务器上。

如果您正在为客户或公司做一个严肃的项目，请不要使用这种方法。查看第三章，*数据库连接*，并学习使用实际数据库。

好吧，假设除了使用文本文件之外没有其他办法；我们该怎么做呢？很简单：创建一个名为`files.txt`的文本文件，并将其放入我们在本章开头创建的 FTP 目录中。

# 编写代码

接下来，打开`mainwindow.h`并添加以下头文件：

```cpp
#include <QMainWindow> 
#include <QDebug> 
#include <QNetworkAccessManager> 
#include <QNetworkRequest> 
#include <QNetworkReply> 
#include <QFile> 
#include <QFileInfo> 
#include <QFileDialog> 
#include <QListWidgetItem> 
#include <QMessageBox> 
```

之后，添加以下变量和函数：

```cpp
private: 
   Ui::MainWindow *ui; 
 QNetworkAccessManager* manager; 

   QString ftpAddress; 
   int ftpPort; 
   QString username; 
   QString password; 

   QNetworkReply* downloadFileListReply; 
   QNetworkReply* uploadFileListReply; 

   QNetworkReply* uploadFileReply; 
   QNetworkReply* downloadFileReply; 

   QStringList fileList; 
   QString uploadFileName; 
   QString downloadFileName; 

public:
   void getFileList();
```

完成上一步后，打开`mainwindow.cpp`并将以下代码添加到类构造函数中：

```cpp
MainWindow::MainWindow(QWidget *parent) : 
   QMainWindow(parent), 
   ui(new Ui::MainWindow) 
{ 
   ui->setupUi(this); 

 manager = new QNetworkAccessManager(this); 

   ftpAddress = "ftp://127.0.0.1/"; 
   ftpPort = 21; 
   username = "tester"; // Put your FTP user name here
   password = "123456"; // Put your FTP user password here 
   getFileList(); 
} 
```

我们所做的基本上是初始化`QNetworkAccessManager`对象并设置存储我们的 FTP 服务器信息的变量，因为我们将在后续步骤中多次使用它。之后，我们将调用`getFileList()`函数开始从 FTP 服务器下载`files.txt`。`getFileList()`函数如下所示：

```cpp
void MainWindow::getFileList() 
{ 
   QUrl ftpPath; 
   ftpPath.setUrl(ftpAddress + "files.txt"); 
   ftpPath.setUserName(username); 
   ftpPath.setPassword(password); 
   ftpPath.setPort(ftpPort); 

   QNetworkRequest request; 
   request.setUrl(ftpPath); 

   downloadFileListReply = manager->get(request); 
   connect(downloadFileListReply, &QNetworkReply::finished, this, 
   &MainWindow::downloadFileListFinished); 
} 
```

我们使用`QUrl`对象来存储有关我们的服务器和我们试图下载的文件位置的信息，然后将其提供给`QNetworkRequest`对象，然后通过调用`QNetworkAccessManager::get()`将其发送出去。由于我们不知道何时所有文件将完全下载，因此我们利用了 Qt 的`SIGNAL`和`SLOT`机制。

我们连接了来自`downloadFileListReply`指针（指向`mainwindow.h`中的`QNetworkReply`对象）的`finished()`信号，并将其链接到`slot`函数`downloadFileListFinished()`，如下所示：

```cpp
void MainWindow::downloadFileListFinished() 
{ 
   if(downloadFileListReply->error() != QNetworkReply::NoError) 
   { 
         QMessageBox::warning(this, "Failed", "Failed to load file 
         list: " + downloadFileListReply->errorString()); 
   } 
   else 
   { 
         QByteArray responseData; 
         if (downloadFileListReply->isReadable()) 
         { 
               responseData = downloadFileListReply->readAll(); 
         } 

         // Display file list 
         ui->fileList->clear(); 
         fileList = QString(responseData).split(","); 

         if (fileList.size() > 0) 
         { 
               for (int i = 0; i < fileList.size(); i++) 
               { 
                     if (fileList.at(i) != "") 
                     { 
                           ui->fileList->addItem(fileList.at(i)); 
                     } 
               } 
         } 
   } 
} 
```

代码有点长，所以我将函数分解为以下步骤：

1.  如果在下载过程中出现任何问题，请显示一个消息框，告诉我们问题的性质。

1.  如果一切顺利并且下载已经完成，我们将尝试通过调用`downloadFileListReply` | `readAll()`来读取数据。

1.  然后，清空列表窗口并开始解析文本文件的内容。我们在这里使用的格式非常简单；我们只使用逗号符号来分隔每个文件名：`filename1,filename2,filename,...`。重要的是我们不要在实际项目中这样做。

1.  一旦我们调用`split(",")`将字符串拆分为字符串列表，就进行`for`循环并在列表窗口中显示每个文件名。

测试前面的代码是否有效，创建一个名为`files.txt`的文本文件，并将以下文本添加到文件中：

```cpp
filename1,filename2,filename3 
```

然后，将文本文件放到 FTP 目录中并运行项目。您应该能够在应用程序中看到它出现如下：

![](img/515e1c2d-015e-4fd2-88e6-9588304c21a9.png)

一旦它工作正常，我们可以清空文本文件的内容并继续下一节。

# 将文件上传到 FTP 服务器

由于我们的 FTP 目录中还没有任何文件（除了文件列表），让我们编写代码以允许我们上传我们的第一个文件。

1.  首先，打开`mainwindow.ui`，右键单击“打开”按钮。然后，选择“转到槽”并选择“clicked()”选项：

![](img/9101a98a-64cd-4902-b2ea-d463509b03d9.png)

1.  将自动为您创建一个`slot`函数。然后，将以下代码添加到函数中，以打开文件选择器窗口，让用户选择要上传的文件：

```cpp
void MainWindow::on_openButton_clicked() 
{ 
   QString fileName = QFileDialog::getOpenFileName(this, "Select 
   File", qApp->applicationDirPath()); 
   ui->uploadFileInput->setText(fileName); 
}
```

1.  之后，重复此步骤，并对“上传”按钮执行相同操作。这次，其`slot`函数的代码看起来像下面这样：

```cpp
void MainWindow::on_uploadButton_clicked() 
{ 
   QFile* file = new QFile(ui->uploadFileInput->text()); 
   QFileInfo fileInfo(*file); 
   uploadFileName = fileInfo.fileName(); 

   QUrl ftpPath; 
   ftpPath.setUrl(ftpAddress + uploadFileName); 
   ftpPath.setUserName(username); 
   ftpPath.setPassword(password); 
   ftpPath.setPort(ftpPort); 

   if (file->open(QIODevice::ReadOnly)) 
   { 
         ui->uploadProgress->setEnabled(true); 
         ui->uploadProgress->setValue(0); 

         QNetworkRequest request; 
         request.setUrl(ftpPath); 

         uploadFileReply = manager->put(request, file); 
         connect(uploadFileReply, 
         SIGNAL(uploadProgress(qint64,qint64)), this, 
         SLOT(uploadFileProgress(qint64,qint64))); 
         connect(uploadFileReply, SIGNAL(finished()), this,  
         SLOT(uploadFileFinished())); 
   } 
   else 
   { 
         QMessageBox::warning(this, "Invalid File", "Failed to open 
         file for upload."); 
   } 
} 

```

代码看起来有点长，所以让我们分解一下：

1.  我们使用`QFile`类打开我们要上传的文件（文件路径取自`ui->uploadFileInput->text()`）。如果文件不存在，显示一个消息框通知用户。

1.  然后，我们将 FTP 服务器和上传目的地的信息填入一个`QUrl`对象中，然后将其提供给`QNetworkRequest`对象。

1.  之后，我们开始读取文件的内容，并将其提供给`QNetworkAccessManager::put()`函数。

1.  由于我们不知道文件何时会完全上传，我们使用了 Qt 提供的`SIGNAL`和`SLOT`机制。我们将`uploadProgress()`和`finished()`信号链接到我们的两个自定义`slot`函数`uploadFileProgress()`和`uploadFileFinised()`。

`slot`函数`uploadFileProgress()`将告诉我们上传的当前进度，因此我们可以用它来设置进度条：

```cpp
void MainWindow::uploadFileProgress(qint64 bytesSent, qint64 bytesTotal) 
{ 
   qint64 percentage = 100 * bytesSent / bytesTotal; 
   ui->uploadProgress->setValue((int) percentage); 
} 
```

与此同时，当文件完全上传时，`uploadFileFinished()`函数将被触发：

```cpp
void MainWindow::uploadFileFinished() 
{ 
   if(uploadFileReply->error() != QNetworkReply::NoError) 
   { 
         QMessageBox::warning(this, "Failed", "Failed to upload file: " 
         + uploadFileReply->errorString()); 
   } 
   else 
   { 
         QMessageBox::information(this, "Success", "File successfully 
         uploaded."); 
   } 
} 

```

我们还没有完成前面的函数。由于已向 FTP 服务器添加了新文件，我们必须更新现有文件列表，并替换存储在 FTP 目录中的`files.txt`文件。由于代码稍微长一些，我们将把代码分成几个部分，这些部分都发生在显示文件成功上传消息框之前。

1.  首先，让我们检查新上传的文件是否已经存在于我们的文件列表中（替换 FTP 服务器上的旧文件）。如果存在，我们可以跳过整个过程；否则，将文件名追加到我们的`fileList`字符串列表中，如下所示：

```cpp
// Add new file to file list array if not exist yet 
bool exists = false; 
if (fileList.size() > 0) 
{ 
   for (int i = 0; i < fileList.size(); i++) 
   { 
         if (fileList.at(i) == uploadFileName) 
         { 
               exists = true; 
         } 
   } 
} 

if (!exists) 
{ 
   fileList.append(uploadFileName); 
} 
```

1.  之后，在我们应用程序的目录中创建一个临时文本文件（`files.txt`），并将新文件列表保存在文本文件中：

```cpp
// Create new files.txt 
QString fileName = "files.txt"; 
QFile* file = new QFile(qApp->applicationDirPath() + "/" + fileName); 
file->open(QIODevice::ReadWrite); 
if (fileList.size() > 0) 
{ 
   for (int j = 0; j < fileList.size(); j++) 
   { 
         if (fileList.at(j) != "") 
         { 
               file->write(QString(fileList.at(j) + ",").toUtf8()); 
         } 
   } 
} 
file->close(); 
```

1.  最后，我们使用`QFile`类打开我们刚创建的文本文件，并将其再次上传到 FTP 服务器以替换旧的文件列表：

```cpp
// Re-open the file 
QFile* newFile = new QFile(qApp->applicationDirPath() + "/" + fileName); 
if (newFile->open(QIODevice::ReadOnly)) 
{ 
   // Update file list to server 
   QUrl ftpPath; 
   ftpPath.setUrl(ftpAddress + fileName); 
   ftpPath.setUserName(username); 
   ftpPath.setPassword(password); 
   ftpPath.setPort(ftpPort); 

   QNetworkRequest request; 
   request.setUrl(ftpPath); 
   uploadFileListReply = manager->put(request, newFile); 
   connect(uploadFileListReply, SIGNAL(finished()), this, SLOT(uploadFileListFinished())); 
   file->close(); 
} 
```

1.  再次使用`SIGNAL`和`SLOT`机制，以便在文件列表上传完成时得到通知。`slot`函数`uploadFileListFinished()`看起来像下面这样：

```cpp
void MainWindow::uploadFileListFinished() 
{ 
   if(uploadFileListReply->error() != QNetworkReply::NoError) 
   { 
         QMessageBox::warning(this, "Failed", "Failed to update file list: " + uploadFileListReply->errorString()); 
   } 
   else 
   { 
         getFileList(); 
   } 
} 

```

1.  我们基本上只是在更新文件列表到 FTP 服务器后再次调用`getFileList()`。如果现在构建和运行项目，您应该能够将第一个文件上传到本地 FTP 服务器，万岁！

![](img/9f62c8f3-6cf9-42aa-8a69-8d79ea69d13b.png)

# 从 FTP 服务器下载文件

现在我们已经成功将第一个文件上传到 FTP 服务器，让我们创建一个功能，将文件下载回我们的计算机！

1.  首先，再次打开`mainwindow.ui`，右键单击“设置文件夹”按钮。选择转到槽... 并选择 clicked()信号以创建一个`slot`函数。`slot`函数非常简单；它只会打开一个文件选择对话框，但这次它只允许用户选择一个文件夹，因为我们为其提供了一个`QFileDialog::ShowDirsOnly`标志：

```cpp
void MainWindow::on_setFolderButton_clicked() 
{ 
   QString folder = QFileDialog::getExistingDirectory(this, tr("Open Directory"), qApp->applicationDirPath(), QFileDialog::ShowDirsOnly); 
   ui->downloadPath->setText(folder); 
} 
```

1.  然后，在列表窗口上右键单击并选择转到槽... 这一次，我们将选择`itemDoubleClicked(QListWidgetItem*)`选项：

![](img/ccd5fed6-24d3-4345-a7b5-06be7331f314.png)

1.  当用户在列表窗口中双击项目时，将触发以下函数，启动下载。文件名可以通过调用`item->text()`从`QListWidgetItem`对象中获取：

```cpp
void MainWindow::on_fileList_itemDoubleClicked(QListWidgetItem *item) 
{ 
   downloadFileName = item->text(); 

   // Check folder 
   QString folder = ui->downloadPath->text(); 
   if (folder != "" && QDir(folder).exists()) 
   { 
         QUrl ftpPath; 
         ftpPath.setUrl(ftpAddress + downloadFileName); 
         ftpPath.setUserName(username); 
         ftpPath.setPassword(password); 
         ftpPath.setPort(ftpPort); 

         QNetworkRequest request; 
         request.setUrl(ftpPath); 

         downloadFileReply = manager->get(request); 
         connect(downloadFileReply, 
         SIGNAL(downloadProgress(qint64,qint64)), this, 
         SLOT(downloadFileProgress(qint64,qint64))); 
         connect(downloadFileReply, SIGNAL(finished()), this, 
         SLOT(downloadFileFinished())); 
   } 
   else 
   { 
         QMessageBox::warning(this, "Invalid Path", "Please set the 
         download path before download."); 
   } 
} 
```

1.  就像我们在`upload`函数中所做的那样，我们在这里也使用了`SIGNAL`和`SLOT`机制来获取下载过程的进展以及完成信号。`slot`函数`downloadFileProgress()`将在下载过程中被调用，我们用它来设置第二个进度条的值：

```cpp
void MainWindow::downloadFileProgress(qint64 byteReceived,qint64 bytesTotal) 
{ 
   qint64 percentage = 100 * byteReceived / bytesTotal; 
   ui->downloadProgress->setValue((int) percentage); 
} 
```

1.  然后，当文件完全下载时，`slot`函数`downloadFileFinished()`将被调用。之后，我们将读取文件的所有数据并将其保存到我们想要的目录中：

```cpp
void MainWindow::downloadFileFinished() 
{ 
   if(downloadFileReply->error() != QNetworkReply::NoError) 
   { 
         QMessageBox::warning(this, "Failed", "Failed to download 
         file: " + downloadFileReply->errorString()); 
   } 
   else 
   { 
         QByteArray responseData; 
         if (downloadFileReply->isReadable()) 
         { 
               responseData = downloadFileReply->readAll(); 
         } 

         if (!responseData.isEmpty()) 
         { 
               // Download finished 
               QString folder = ui->downloadPath->text(); 
               QFile file(folder + "/" + downloadFileName); 
               file.open(QIODevice::WriteOnly); 
               file.write((responseData)); 
               file.close(); 

               QMessageBox::information(this, "Success", "File 
               successfully downloaded."); 
         } 
   } 
}
```

1.  现在构建程序，你应该能够下载文件列表上列出的任何文件！

![](img/9b76d2da-fc18-4ff7-9e3f-1a559ee1d2cf.png)

# 总结

在本章中，我们学习了如何使用 Qt 的网络模块创建自己的云存储客户端。在接下来的章节中，我们将学习更多关于多媒体模块，并使用 Qt 从头开始创建自己的多媒体播放器。
