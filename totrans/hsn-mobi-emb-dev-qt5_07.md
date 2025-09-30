# Qt 网络用于通信

网络对于移动设备来说几乎和移动设备本身一样重要。没有网络，数据就必须从物理上一个地方移动到另一个地方。幸运的是，Qt 在`QNetwork`中提供了广泛的网络功能。在本章中，我们将讨论以下 API：

+   `QNetworkReply`

+   `QNetworkRequest`

+   `QDnsLookup`

+   `QHostInfo`

+   `QLocalServer`

+   `QTcpSocket`

要显示附近的可用 Wi-Fi 网络，我们还将介绍以下内容：

+   `QNetworkSession`

+   `QNetworkConfiguration`

你还将学习如何使用 Qt API 进行标准网络任务，例如**域名服务**（**DNS**）查找、下载和上传文件，以及如何使用 Qt 的套接字类进行通信。

# 高级 – 请求、回复和访问

Qt 中的网络功能非常丰富。Qt Quick 中的网络比 Qt 更隐蔽。在**Qt 建模语言**（**QML**）中，你可以下载远程组件并在你的应用程序中使用它们，但任何其他任意下载或网络功能你将不得不在 C++后端中实现或使用 JavaScript。

尽管`QNetworkRequest`、`QNetworkReply`和`QNetworkAccessManager`都用于制作网络请求，但让我们分开来看如何使用它们。

# QNetworkRequest

`QNetworkRequest`是访问功能的一部分。它构建一个`request`，可以是以下动词之一：

+   `GET`: `get(...)`

+   `POST`: `post(...)`

+   `PUT`: `put(...)`

+   `DELETE`: `deleteResource(...)`

+   `HEAD`: `head(...)`

你还可以使用`sendCustomRequest`发送自定义动词，它接受自定义动词作为`QByteArray`参数。

可以使用`setHeader`将头设置为已知头，可以是以下之一：

+   `ContentDispositionHeader`

+   `ContentTypeHeader`

+   `ContentLengthHeader`

+   `LocationHeader`

+   `LastModifiedHeader`

+   `CookieHeader`

+   `SetCookieHeader`

+   `UserAgentHeader`

+   `ServerHeader`

可以使用`setRawHeader`设置原始或自定义头。HTTP 属性可以帮助控制请求缓存、重定向和 cookies。它们可以用`setAttribute`设置。

让我们把这段代码放到以下代码中。

源代码可以在`Chapter05-1`目录下的`cp5`分支中的 Git 仓库中找到。

要使用网络模块，在`.pro`项目中，将`network`添加到`QT`变量中，如下所示：

```cpp
QT += network
```

我们现在可以使用 Qt 网络。

`QNetworkRequest`是需要用于从网络请求操作的部分，例如`get`和`put`。

一个简单的实现如下：

```cpp
QNetworkRequest request;
request.setUrl(QUrl("http://www.example.com"));
```

`QNetworkRequest`也可以将`QUrl`作为其参数。`QNetworkRequest`不是基于`QObject`的，因此它没有父对象，也没有任何自己的信号。所有通信都是通过`QNetworkAccessManager`完成的。

你想要连接的一个信号是`finished`信号。

假设我有一些需要传输的表单数据；我需要使用`setHeader`添加一个标准头。我还可以添加以下自定义头，我称之为`X-UUID`：

```cpp
request.setHeader(QNetworkRequest::ContentTypeHeader, "application/x-www-form-urlencoded");
request.setRawHeader(QByteArray("X-UUID"), QUuid::createUuid().toByteArray());
```

现在我们有一个有效的`QNetworkRequest`，我们需要将其发送到`QNetworkAccessManager`。让我们看看我们如何做到这一点。

# QNetworkAccessManager

引入管理器——`QNetworkAccessManager`（**QNAM**）。它用于通过网络发送和接收异步请求。通常，一个应用程序中只有一个 QNAM 实例，如下所示：

```cpp
QNetworkAccessManager *manager = new QNetworkAccessManager(this);
```

在最简单的情况下，你可以使用`get`、`put`、`post`、`deleteResource`或`head`函数来创建一个 QNAM 请求。

QNAM 使用信号来传输数据和请求信息，而`finished()`信号用于表示请求已完成。

让我们为它添加一个信号处理程序，如下所示：

```cpp
 connect(manager, &QNetworkAccessManager::finished, 
        this, &MainWindow::replyFinished);
```

这将调用你的`replyFinished`槽，其中包含`QNetworkReply`参数中的数据和头信息，如下所示：

```cpp
void MainWindow::replyFinished(QNetworkReply *reply) 
{ 
    if (reply->error()) 
        ui->textEdit->insertPlainText( reply->errorString()); 
    else { 
        QList<QByteArray> headerList = reply->rawHeaderList(); 
        ui->textEdit->insertPlainText(headerList.join("\n") +"\n"); 
        QByteArray responsData = reply->readAll(); 
        ui->textEdit->insertHtml(responsData); 
    } 
}
```

然后，按照以下方式在`QNetworkAccessManager`上调用`get`方法：

```cpp
manager->get(request);
```

下载东西就这么简单！QNAM 将施展其魔法并下载 URL。

创建文件上传也是同样简单的方法。当然，你的 Web 服务器需要支持`put`方法，如下所示：

```cpp
    QFileDialog dialog(this); 
    dialog.setFileMode(QFileDialog::AnyFile); 
    QString filename = QFileDialog::getOpenFileName(this, tr("Open File"), QDir::homePath()); 

    if (!filename.isEmpty()) { 
        QFile file(filename); 
        if (file.open(QIODevice::ReadOnly | QIODevice::Text)) { 
            QByteArray fileBytes = file.readAll(); 
            manager->put(request, fileBytes); 
        } 
    }
```

源代码可以在 Git 仓库的`Chapter05-2`目录下的`cp5`分支中找到。

如果你需要在 URL 中发送一些查询参数，你可以使用`QUrlQuery`来构建`form`查询数据，然后按照以下方式发送`request`：

```cpp
QNetworkRequest request;
QUrl url("http://www.example.com");

QUrlQuery formData;
formData.addQueryItem("login", "me");
formData.addQueryItem("password", "123");
formData.addQueryItem("submit", "Send");
url.setQuery(formData);
request.setUrl(url);
manager->get(request);
```

可以使用`post`函数将表单数据作为`QByteArray`上传，如下所示：

```cpp
QByteArray postData;
postData.append("?login=me&password=123&submit=Send");
manager->post(request, postData);
```

要发送多部分表单数据，例如表单数据和图片，你可以使用`QHttpMultiPart`如下所示：

```cpp
QFile *file = new QFile(filename); 
    if (file->open(QIODevice::ReadOnly)) { 
        QByteArray fileBytes = file->readAll(); 
         QHttpMultiPart *multiPart = 
             new QHttpMultiPart(QHttpMultiPart::FormDataType); 

        QHttpPart textPart; 
        textPart.setHeader(QNetworkRequest::ContentDispositionHeader, QVariant("form-data; name=\"filename\"")); 
        textPart.setBody(filename.toLocal8Bit()); 

        QHttpPart filePart; 
        filePart.setHeader(QNetworkRequest::ContentDispositionHeader, QVariant("form-data; name=\"file\"")); 

        filePart.setHeader(QNetworkRequest::ContentTypeHeader, QVariant("application/zip")); 

        filePart.setBodyDevice(file); 

        file->setParent(multiPart); 

        multiPart->append(textPart); 
        multiPart->append(filePart); 

        manager->put(request, multiPart); 
    }
```

当然，这些示例都没有跟踪回复。`QNetworkReply`是`QNetworkAccessManager`的`get`、`post`和`put`方法的返回值，可以用来跟踪下载或上传进度或是否有错误。

# QNetworkReply

对 QNAM 的`get`、`post`等所有调用都将返回`QNetworkReply`。

你需要删除这个指针，否则它将导致内存泄漏，但不要在`finished`信号处理程序中删除它。你可以使用`deleteLater()`。

`QNetworkReply`有一个有趣的信号，我们很可能需要处理。让我们从两个最重要的信号开始——`error`和`readyRead`。

因此，让我们正确地处理那个`QNetworkReply`。由于我们事先没有有效的对象，我们需要在网络请求操作之后连接信号。这对我来说似乎有点反直觉，但这是必须这样做的方式，并且它有效。代码如下：

```cpp
QNetworkReply *networkReply = manager->get(request);
connect(networkReply, SIGNAL(finished()), this, SLOT(requestFinished()));
connect(networkReply, SIGNAL(error(QNetworkReply::NetworkError)),
    this,SLOT(networkReplyError(QNetworkReply::NetworkError)));
 connect(networkReply, SIGNAL(readyRead()), this, SLOT(readyRead()));

```

我正在使用传统的信号连接方式，但你也可以并应该编写如下所示的连接，因为它允许编译时检查语法和其他错误：

```cpp
connect(networkReply, &QNetworkReply::error, this, &MyClass::networkReplyError);
 connect(networkReply, QOverload<QNetworkReply::NetworkError>::of(&QNetworkReply::error),this, &MyClass::networkReplyError);
connect(networkReply, &QNetworkReply::finished, this, &MyClass::requestFinished);
connect(networkReply, &QNetworkReply::readyRead, this, &MyClass::readyRead);
```

因此，我们现在已经发送了一个请求，正在等待服务器的回复。让我们逐一查看信号处理程序。

当出现错误并带有错误代码作为参数时，会发出 `error(QNetworkReply::NetworkError)`。如果您需要一个用户友好的字符串，可以使用 `QNetworkReply::errorString()` 获取。当请求完成时，会发出 `finished()`。回复仍然打开，因此您可以在这里读取它：`readyRead()`。由于回复是从 `QIODevice` 派生的，它具有 `readyRead` 信号，该信号在可以读取更多数据时发出。

在大文件下载时，您可能想要监控下载进度，这是一件常见的事情。通常，每个下载都有一个进度条。`QNetworkReply` 会发出 `downloadProgress(qint64 bytesReceived, qint64 bytesTotal)` 信号，如下所示：

```cpp
connect(networkReply, &QNetworkReply::downloadProgress, this, &MyClass::onDownloadProgress);
```

对于上传，存在相应的 `uploadProgress`。

当下载需要身份验证时，会发出 `preSharedKeyAuthenticationRequired(QSslPreSharedKeyAuthenticator *authenticator)`。`QSslPreSharedKeyAuthenticator` 对象应加载预共享密钥和其他详细信息以验证用户。

当 **安全套接字层** (**SSL**) 遇到问题时，会调用 `sslErrors(const QList<QSslError> &errors)` 信号，包括证书验证错误。

`QNetworkManager` 也可以执行简单的 **文件传输协议** (**FTP**) 转发。

# QFtp

使用 Qt 进行 FTP 有两种方式。`QNetworkAccessManager` 提供简单的 FTP `get` 和 `put` 支持，我们可以轻松使用它。

FTP 服务器通常需要某种类型的用户名和密码。我们使用 `QUrl` 的 `setUserName()` 和 `setPassword()` 来设置这些，如下所示：

```cpp
 QUrl url("ftp://llornkcor.com/");
 url.setUserName("guest@llornkcor.com");
 url.setPassword("handsonmobileandembedded");
```

源代码可以在 Git 仓库的 `Chapter05-5` 目录下的 `cp5` 分支中找到。

一旦我们知道文件名，我们需要将其添加到 `url` 中，因为它将使用此信息写入失败，如下所示：

```cpp
        url.setPath(QFileInfo(file).fileName());
```

然后，设置请求 `url`，如下所示：

```cpp
 request.setUrl(url);
```

一旦我们在 QNAM 上调用 `put`，我们就可以将槽连接到 `QNetworkReply` 信号。

```cpp
QNetworkReply *networkReply = manager->put(request, fileBytes);

connect(networkReply, &QNetworkReply::downloadProgress, 
    this, &MainWindow::onDownloadProgress);
connect(networkReply, &QNetworkReply::downloadProgress, 
    this, &MainWindow::onUploadProgress);
```

不要忘记 `error` 信号需要 `QOverload`，如下所示：

```cpp
connect(networkReply, QOverload<QNetworkReply::NetworkError>::of(&QNetworkReply::error),               ={
    qDebug() << Q_FUNC_INFO << code << networkReply->errorString(); });

connect(networkReply, &QNetworkReply::finished, 
    this, &MainWindow::requestFinished);
```

如果您需要执行除 `get` 和 `put` 之外更复杂的事情，您将需要使用除 `QNetworkAccessManager` 之外的其他东西。

`QFtp` 不包含在 Qt 中，但您可以使用从 Qt 4 移植的独立 `QFtp` 模块，如下运行与 Qt 5：

```cpp
git clone -b 5.12 git://code.qt.io/qt/qtftp.git

```

我们需要构建 `QFtp`，因此可以在 Qt Creator 中打开 `qtftp.pro`。运行构建并安装它。

使用命令行，命令如下：

```cpp
cd qtftp
qmake
make
make install
```

我们需要将其安装到 Qt 5.12 中，因此请在 Qt Creator 中导航到 Projects | Build | Build Steps 并选择 Add Build Step | Make。在参数字段中，键入 `install`。

构建此文件，它也会安装。

在项目的 `.pro` 文件中，为了告诉 `qmake` 使用 `network` 和 `ftp` 模块，请添加以下内容：

`QT += network ftp`

`QFtp` 的工作方式非常典型；登录，执行操作，然后登出，如下所示：

```cpp
    connect(ftp, SIGNAL(commandFinished(int,bool)), 
            this, SLOT(qftpCommandFinished(int,bool))); 

    connect(ftp, SIGNAL(stateChanged(int)), 
            this, SLOT(stateChanged(int))); 

    connect(ftp, SIGNAL(dataTransferProgress(qint64,qint64)), 
            this, SLOT(qftpDataTransferProgress(qint64,qint64))); 

    QUrl url(URL); 
    ftp->connectToHost(url.host(), 21); 
    ftp->login(USER, PASS);

```

我们连接到 `commandFinished` 信号，它可以告诉我们是否发生了错误。

`stateChanged` 信号将告诉我们何时登录，而 `dataTransferProgress` 信号将告诉我们何时正在传输字节。

`QFtp` 支持其他操作，包括以下：

+   `list`

+   `cd`

+   `remove`

+   `mkdir`

+   `rmdir`

+   `rename`

QNAM 还触及了我最喜欢的 Qt 网络部分——承载管理。让我们继续学习承载管理。

# 带来好消息的承载管理

承载管理旨在方便用户对网络连接的控制。有 `open` 和 `close` 函数用于找到的连接。它不做的一件事是实际配置这些连接。它们必须已经由系统配置。

它还旨在能够将连接分组，以便更容易地在连接之间平滑切换，例如从 Wi-Fi 迁移到移动蜂窝数据，有点像 **媒体无关切换**（**MIH**）或 **未授权移动接入**（**UMA**）规范。如果您对帮助切换的开源库感兴趣，请查看 SourceForge 上的 Open MIH。

在 Qt 的承载管理最初开发时，Symbian 是最常用的，可以说是最重要的移动操作系统。Symbian 有能力在不停机或丢失数据的情况下，在技术之间无缝迁移连接，有点像手机连接从基站迁移到基站的方式。

苹果似乎称之为 Wi-Fi 助手；三星有自动网络切换。

几年前，移动数据连接非常昂贵，因此一旦发生特定的上传或下载，连接通常会关闭。连接的开启和关闭更加动态，需要自动控制。

无论如何，`QtConfigurationManager` 将使用系统支持的功能；它不会实现自己的连接数据迁移。

Qt 有以下三个主要类构成了承载管理：

+   `QNetworkConfiguration`

+   `QNetworkConfigurationManager`

+   `QNetworkSession`

此外，还有 `QBearerEngine`，它是承载插件的基类。

# QNetworkConfiguration

`QNetworkConfiguration` 表示一个网络连接配置，例如连接到特定接入点的 Wi-Fi，其 **服务集标识符**（**SSID**）作为配置名称。

网络配置可以是以下类型之一：

+   `QNetworkConfiguration::InternetAccessPoint`:

    +   这种类型是一个典型的接入点，例如 Wi-Fi **接入点**（**AP**）或它可能代表以太网或移动网络。

+   `QNetworkConfiguration::ServiceNetwork`:

    +   `ServiceNetwork` 类型是一组称为 **服务网络接入点**（**SNAP**）的接入点。系统将根据成本、速度和可用性等标准确定连接到哪个服务网络最好。`QNetworkConfiguration::ServiceNetwork` 类型的配置也可能在其子 `QNetworkConfiguration::InternetAccessPoint` 之间漫游。

+   `QNetworkConfiguration::UserChoice`:

    +   此类型可以表示用户首选的配置。它曾被诺基亚的 Maemo 和 Symbian 平台使用，其中系统可以弹出一个对话框询问用户哪个 AP 最好。当前的所有承载后端都不使用这种类型的`QNetworkConfiguration`。

通常，我们需要知道承载类型，也就是说，连接使用的是哪种通信协议。让我们了解一下`BearerType`。

# QNetworkConfiguration::BearerType

这是一个`enum`，指定了`QNetworkConfiguration`的底层技术，可以是以下之一：

+   `BearerEthernet`

+   `BearerWLAN`

+   `Bearer2G`

+   `BearerCDMA2000`

+   `BearerWCDMA`

+   `BearerHSPA`

+   `BearerBluetooth`

+   `BearerWiMAX`

+   `BearerEVDO`

+   `BearerLTE`

+   `Bearer3G`

+   `Bearer4G`

这可以通过调用`QNetworkConfiguration`对象的`bearerType()`函数来确定，如下所示：

```cpp
QNetworkConfiguration config;
if (config.bearerType() == QNetworkConfiguration::Bearer4G)
    qWarning() << "Config is using 4G";
```

你可以打开或连接。

# QNetworkConfiguration::StateFlags

`StateFlags`是`StateFlag`值的 OR'd `||`组合，如下所示：

+   `Defined`: 已知于系统但尚未配置

+   `Discovered`: 已知并配置，可用于`open()`

+   `Active`: 当前在线

一个具有`Active`标志的`QNetworkConfiguration`也将具有`Discovered`和`Defined`标志。你可以通过以下方式检查配置是否处于活动状态：

```cpp
QNetworkConfiguration config;
if (config.testFlag(QNetworkConfiguration::Active))
    qWarning() << "Config is active";
```

# QNetworkConfigurationManager

`QNetworkConfigurationManager`允许你获取系统的`QNetworkConfigurations`，如下所示：

```cpp
QNetworkConfigurationManager manager;
QNetworkConfiguration default = manager.defaultConfiguration();
```

在使用它之前，总是明智的等待`QNetworkConfigurationManager`的`updateCompleted`信号，以确保配置已正确设置。

默认配置是系统定义的默认配置。它可能处于`Active`状态或只是`Discovered`状态。

如果你只需要确定系统当前是否在线，`manager->isOnline();`如果系统被认为是在线的，将返回`true`。在线是指通过网络连接到另一个设备，这可能或可能不是互联网，可能或可能不是正确路由。因此，它可能是在线的，但不能访问互联网。

你可能需要调用`updateConfigurations()`，这将要求系统更新配置列表，然后你需要监听`updateCompleted`信号才能继续。

你可以通过调用`allConfigurations()`获取系统已知的所有配置，或者通过`allConfigurations(QNetworkConfiguration::Discovered);`过滤到具有特定状态的配置。

在这种情况下，它返回一个`Discovered`配置的列表。

你可以通过调用`capabilities()`来检查系统的能力，它可以是以下之一：

+   `CanStartAndStopInterfaces`: 系统允许用户启动和停止连接

+   `DirectConnectionRouting`: 连接路由直接绑定到指定的设备接口

+   `SystemSessionSupport`: 系统保持连接打开，直到所有会话都关闭

+   `ApplicationLevelRoaming`: 应用程序可以控制漫游/迁移

+   `ForcedRoaming`：系统在漫游/迁移时将重新连接

+   `DataStatics`：系统提供有关已传输和接收的数据的信息

+   `NetworkSessionRequired`：系统需要会话

# QNetworkSession

`QNetworkSession`提供了一种启动和停止连接以及管理连接会话的方法。在用`QNetworkConfiguration`实例化`QNetworkSession`且该配置为`ServiceNetwork`类型时，它可以提供漫游功能。在大多数系统中，漫游将涉及实际断开连接然后连接新的接口和/或连接。在其他系统中，漫游可以是无缝的，不会干扰用户的流量。

如果`QNetworkConfigurationManager`的能力报告它支持`CanStartAndStopInterfaces`，那么您可以使用`QNetworkSession`来`open()`（连接）和`stop()`（关闭）`QNetworkConfigurations`。

QNAM 在幕后进行网络请求时会使用`QNetworkSession`。您可以使用`QNetworkSession`如下监控连接：

源代码可以在 Git 仓库的`Chapter05-3`目录下的`cp5`分支中找到。

```cpp
QNetworkAccessManager manager;
QNetworkConfiguration config = manager.configuration();
QNetworkSession *networkSession = new QNetworkSession(config, this);
connect(networkSession, &QNetworkSession::opened, this, &SomeClass::sessionOpened);
networkSession->open();
```

要监控从 QNAM 请求接收和发送的字节，连接到`bytesReceived`和`bytesWritten`信号，如下所示：

```cpp
connect(networkSession, &QNetworkSession::bytesReceived, this, &SomeClass::bytesReceived); 
connect(networkSession, &QNetworkSession::bytesWritten, this, &SomeClass::bytesWritten); 

QNetworkRequest request(QUrl("http://example.com"));
manager->get(request);
```

# 漫游

当我提到漫游时，我指的是在 Wi-Fi 和移动数据之间的漫游，而不是在家庭网络之外漫游，这可能是一个非常昂贵的移动数据使用。

为了方便漫游，客户端应用可以连接到`preferredConfigurationChanged`信号，然后通过调用`migrate()`开始漫游过程，或者通过调用`ignore()`取消漫游。迁移连接可能就像暂停下载、断开连接并重新连接到新连接，然后继续下载一样简单。这种方法被称为强制漫游。在某些平台上，它可以将数据流无缝迁移到新连接，类似于手机在通话迁移到另一个基站时的操作。

目前没有支持迁移会话的后端。系统集成商可以实现一个支持真正连接迁移和切换的后端。如果系统允许这样做，那也会有所帮助。

话虽如此，三星的 Android 和 iOS 的漫游功能似乎已经赶上了几年前诺基亚的水平。三星称之为自适应 Wi-Fi，之前被称为智能网络切换。iOS 称之为 Wi-Fi Assist。这些都是在系统级别发生的，允许在 Wi-Fi 和移动数据连接之间漫游。这两个平台都不允许应用程序控制切换。

# QBearerEngine

Qt 基于`QBearerEngine`类提供了以下承载后端插件：

+   `Android`：Android

+   `Connman`：Linux 桌面和嵌入式，SailfishOS

+   `Corewlan`：Mac OS 和 iOS

+   `Generic`：所有

+   `NativeWifi`：Windows

+   `NetworkManager`：Linux

+   `NLA`：Windows

根据平台的不同，其中一些与通用后端协同工作。

# 低级 – 网络套接字和服务器

`QTcpSocket` 和 `QTcpServer` 是 Qt 中用于套接字的两个类。它们的工作方式与你的网络浏览器和 WWW 服务器非常相似。它们连接到网络地址主机，而 `QLocalSocket` 和 `QLocalServer` 连接到本地文件描述符。

让我们先看看 `QLocalServer` 和 `QLocalSocket`。

在套接字服务器编程中，基本步骤如下：

1.  创建套接字

1.  设置套接字选项

1.  绑定套接字地址

1.  监听连接

1.  接受新连接

Qt 将这些步骤简化为以下步骤：

1.  创建套接字

1.  监听连接

1.  接受新连接

# QLocalServer

如果你需要在同一台机器上进行通信，那么 `QLocalServer` 将比使用基于 TCP 的套接字服务器稍微高效一些。它可以用于 **进程间通信** (**IPC**)。

首先，我们创建服务器，然后使用客户端用于连接的字符串名称调用 `listen` 函数。我们将连接到 `newConnection` 信号，这样我们就能知道何时有新的客户端连接。

源代码可以在 Git 仓库的 `Chapter05-5a` 目录下的 `cp5` 分支中找到。

当客户端尝试连接时，我们使用 `write` 函数发送一条小消息，最后使用 `flush` 发送消息，如下所示：

```cpp
QLocalServer *localServer = new QLocalServer(this);
localServer->listen("localSocketName");

connect(localServer, &QLocalServer::newConnection, this,      
    &SomeClass::newLocalConnection);

void SomeClass::newLocalConnection()
{
    QLocalSocket *local = localServer->nextPendingConnection();
    local->write("Client OK\r\n");
    local->flush();
}
```

这很简单！任何时候你需要向客户端写入，只需使用 `nextPendingConnection()` 获取下一个 `QLocalSocket` 对象，并使用 `write` 发送数据。确保在所有需要发送的行中添加 `\r\n`，包括最后一行。调用 `flush()` 不是必需的，但它会立即发送数据。

你可以保留这个对象，以便在需要时发送更多消息。

我们的应用程序现在正在等待并监听连接。让我们接下来做这件事。

# QLocalSocket

`QLocalSocket` 用于与 `QLocalServer` 通信。你将想要连接到 `readyRead` 信号。其他信号包括 `connected()`、`disconnected()`、`error(...)` 和 `stateChanged(...)`，如下所示：

源代码可以在 Git 仓库的 `Chapter05-5b` 目录下的 `cp5` 分支中找到。

```cpp
QLocalSocket *lSocket = new QLocalSocket(this);
connect(lSocket, &QLocalSocket::connected, this, &SomeClass::connected);

connect(lSocket, &QLocalSocket::disconnected, this,
    &SomeClass::disconnected);

connect(lSocket, &QLocalSocket::error, this, &SomeClass::error);
connect(lSocket, &QLocalSocket::readyRead, this, &SomeClass::readMessage);

void SomeClass::readMessage()
{
    if (lSocket->bytesAvailable())
        QByteArray msg = lSocket->readAll();
}
```

如果你需要状态变化，你将连接到 `stateChanged`，并且会在以下状态变化时收到通知：

+   `UnconnectedState`

+   `ConnectingState`

+   `ConnectedState`

+   `ClosingState`

现在，我们需要实际连接到服务器，如下所示：

```cpp
lSocket->connectToHost("localSocketName");
```

与 `QLocalServer` 类似，`QLocalSocket` 使用 `write` 函数向服务器发送消息，如下所示：

```cpp
lSocket->write("local socket OK\r\n");
```

记得添加 **行结束符** (**EOL**) `\r\n` 来标记数据馈送行的结束。

这就是一个简单的基于本地套接字的通信。现在，让我们看看基于网络的 TCP 套接字。

# QTcpServer

`QTcpServer` API 与`QLocalServer`非常相似，几乎可以无缝替换，只需进行一些小的更改。最值得注意的是，监听调用的参数略有不同，你需要为`QTcpServer`指定`QHostAddress`而不是`QString`名称和一个端口号。在这里，我使用`QHostAddress::Any`，这意味着它将在所有网络接口上监听。如果你不关心使用哪个端口，将其设置为`0`，如下所示：

```cpp
QTcpServer *tcpServer = new QTcpServer(this);
tcpServer->listen(QHostAddress::Any, 8888);

connect(tcpServer, &QTcpServer::newConnection, this,      
    &SomeClass::newLocalConnection);

void SomeClass::newLocalConnection()
{
    QTcpSocket *tSocket = tcpServer->nextPendingConnection();
    tSocket->write("Client OK\r\n");
    tSocket->flush();
}
```

这看起来熟悉吗？`QHostAddress`可以是 IPv4 或 IPv6 地址。你也可以通过使用`QHostAddress::SpecialAddress` `枚举`来指定不同的地址范围，就像我这样做的一样，它可以有以下之一：

+   `LocalHost`: `127.0.0.1`

+   `LocalHostIPv6`: `::1`

+   `Broadcast`: `255.255.255.255`

+   `AnyIPv4`: `0.0.0.0`

+   `AnyIPv6`: `::`

+   `Any`: `所有 IPv4 和 IPv6 地址`

`QTcpServer`有一个额外的信号`acceptError`，当新连接的接受阶段发生错误时会被触发。你还可以`pauseAccepting()`和`resumeAccepting()`待处理连接队列中的连接接受。

# QTcpSocket

`QTcpSocket`与`QLocalSocket`类似。除了其他方面之外，`QTcpSocket`有`connectToHost`作为连接到服务器的方式，如下所示：

```cpp
QTcpSocket *tSocket = new QTcpSocket(this);
connect(tSocket, &QTcpSocket::connected, this, &SomeClass::connected);

connect(tSocket, &QTcpSocket::disconnected, this,
    &SomeClass::disconnected);

connect(tSocket, &QTcpSocket::error, this, &SomeClass::error);
connect(tSocket, &QTcpSocket::readyRead, this, &SomeClass::readData);

```

要发送一个简单的`HTTP`请求，我们可以在连接后向套接字写入，如下所示：

```cpp
void SomeClass:connected()
{
    QString requestLine = QStringLiteral("GET \index.html HTTP/1.1\r\nhost: www.example.com\r\n\r\n");
    QByteArray ba;
    ba.append(requestLine);
    tSocket->write(ba);
    tSocket->flush();
}
```

这将请求服务器上的`index.html`文件。数据可以在`readyRead`信号处理程序中读取，如下所示：

```cpp
void SomeClass::readData()
{
    if (tSocket->bytesAvailable())
        QByteArray msg = tSocket->readAll();
}
```

如果你不想使用更同步的方式，你也可以使用`waitForConnected`、`waitForBytesWritten`和`waitForReadyRead`函数，如下所示：

```cpp
QTcpSocket *tSocket = new QTcpSocket(this);
if (!tSocket->waitForConnected(3000)) {
    qWarning() << "Not connected";
    return;
}

tSocket->write("GET \index.html HTTP/1.1\r\nhost: www.example.com\r\n\r\n");
tSocket->waitForBytesWritten(1000);
tSocket->waitForReadyRead(3000);
if (tSocket->bytesAvailable())
    QByteArray msg = tSocket->readAll();
```

然后，使用以下命令关闭连接：

```cpp
tSocket->close();
```

# QSctpServer

**SCTP**代表**流控制传输协议**。`QSctpServer`将消息作为字节数组组发送，就像 UDP 一样，而不是像 TCP 套接字一样发送字节数据流。它还确保数据包的可靠传输，就像 TCP 一样。它可以并行或同时发送多个消息。它是通过使用多个连接来做到这一点的。

`QSctpServer`也可以通过将`setMaximumChannelCount`设置为`-1`来发送字节流，就像 TCP 一样。在创建`QSctpServer`对象后，你首先想要做的是`setMaximumChannelCount`。将其设置为`0`将允许它使用客户端使用的通道数，如下所示：

```cpp
QSctpServer *sctpServer = new QSctpServer(this);
sctpServer->setMaximumChannelCount(8);
```

如果你打算使用 TCP 字节流，你可以像`QTcpServer`一样使用`nextPendingConnection()`函数来获取一个`QTcpSocket`对象进行通信。`QSctpServer`有额外的`nextPendingDatagramConnection()`来与`QSctpSocket`通信。

要在`newConnection`信号处理程序中接收字节，请使用以下代码：

```cpp
QSctpSocket *sSocket = sctpServer->nextPendingDatagramConnection();
```

# QSctpSocket

`QSctpSocket`也有对通道数的控制，并且与`QSctpServer`一样，如果你将最大通道数设置为`-1`，它将表现得更像 TCP 套接字，发送数据流而不是消息包。消息块被称为`datagram`。

要读取和写入这些数据报，请使用 `readDatagram()` 和 `writeDatagram()`。让我们来检查 `QNetworkDatagram`。

要构建 `QNetworkDatagram`，您需要一个包含数据消息的 `QByteArray`，一个目标 `QHostAddress`，以及可选的端口号。它可以像以下这样简单：

```cpp
QNetworkDatagram datagram("Hello Mobile!", QHostAddress("10.0.0.50"), 8888);
sSocket->writeDatagram(datagram);
```

这将发送 `"Hello Mobile!"` 消息到相应的服务器。

# QUdpSocket

`QUdpSocket` 发送数据报，例如 `QSctpSocket`，但它们是不可靠的，这意味着它不会重试发送任何数据报。它也是无连接的，并且对数据长度有 65,536 字节的限制。

设置 `QUdpSocket` 有两种方式——`bind(...)` 和 `connectToHost(...)`。

如果您使用 `connectToHost`，您可以使用 `QIODevice` 的 `read()`、`write()`、`readAll()` 方法来发送和接收数据报。使用 `bind(...)` 方法，您需要使用 `readDatagram` 和 `writeDatagram`，如下所示：

```cpp
QUdpSocket *uSocket = new QUdpSocket(this);
uSocket->bind(QHostAddress::LocalHost, 8888);
connect(uSocket, &QUdpSocket::readyRead, this, &SomeClass::readMessage);

void SomeClass::readMessage()
{
  while (udpSocket->hasPendingDatagrams()) {
        QNetworkDatagram datagram = uSocket->receiveDatagram();
        qWarning() << datagram.data();
    }
}
```

# QSslSocket

加密套接字通信可以通过 `QSslSocket` 处理，它使用 SSL 加密 TCP 连接。当连接安全时，将发出加密信号，如下所示：

```cpp
QSslSocket *sslSocket = new QSslSocket(this);
connect(sslSocket, &QSslSocket::encrypted, this, SomeClass::socketEncrypted);
sslSocket->connectToHostEncrypted("example.com", 943);
```

源代码可以在 Git 仓库的 `Chapter05-6a` 目录下的 `cp5` 分支中找到。

这将启动连接并立即开始安全的握手过程。一旦握手完成且没有错误，将发出加密信号，连接将准备就绪。

您需要将密钥/证书对添加到 `QSslSocket` 以利用加密功能。您可以通过使用此网站轻松生成密钥证书失败对进行测试：[`www.selfsignedcertificate.com/`](https://www.selfsignedcertificate.com/)。

由于我们使用的是自签名证书，因此我们需要在错误处理槽中添加 `ignoreSslErrors`：

```cpp
sslSocket->ignoreSslErrors();
```

要添加加密密钥和证书，您需要打开并读取这两个文件，并使用生成的 `QByteArrays` 创建 `QSslKey` 和 `QSslCertificate`：

```cpp
void MainWindow::initCerts() 
{ 
    QByteArray key; 
    QByteArray cert; 

    QString keyPath = 
QFileDialog::getOpenFileName(0, tr("Open Key File"), 
                            QDir::homePath(), 
                            "Key file (*.key)"); 

    if (!keyPath.isEmpty()) { 
        QFile keyFile(keyPath); 
        if (keyFile.open(QIODevice::ReadOnly)) { 
            key = keyFile.readAll(); 
            keyFile.close(); 
        }    
    }    

    QString certPath = 
QFileDialog::getOpenFileName(0, tr("Open cert File"), 
                             QDir::homePath(), 
                            "Cert file (*.cert)"); 

    if (!certPath.isEmpty()) { 
        QFile certFile(certPath); 
        if (certFile.open(QIODevice::ReadOnly)) { 
            cert = certFile.readAll(); 
            certFile.close(); 
        }    
    }    

    QSslKey sslKey(key, QSsl::Rsa,    QSsl::Pem,QSsl::PrivateKey,"localhost"); 
    sslSocket->setPrivateKey(sslKey); 

    QSslCertificate sslCert(cert); 
    sslSocket->addCaCertificate(sslCert); 
    sslSocket->setLocalCertificate(sslCert); 
}

```

当您运行此代码时，您需要使用 `QFileDialog` 导航并找到源目录中的 `localhost.key` 和 `localhost.cert` 文件。

然后，我们使用 `setPrivateKey` 来设置密钥文件，并使用 `addCaCertificate` 和 `setLocalCertificate` 来添加证书。

要从套接字读取，您可以连接到 `readReady` 信号，就像在 `QTcpSocket` 中一样。

要向服务器发送的套接字写入，只需使用 `write` 函数：

```cpp
 sslSocket->write(ui->lineEdit->text().toUtf8() +"\r\n");
```

然后，您可以使用 `QSslSocket` 连接到打开 `QSslSocket` 的 `QTcpServer`。这带我们到下一步。

# QSslServer

好的，没有 `QSslServer` 类，但由于 `QSslSocket` 类只是从 `QTcpSocket` 派生而来，并在其顶部添加了一些额外的 SSL 功能，您可以使用 `QSslSocket` 的函数创建自己的 SSL 服务器。

您需要生成 SSL 密钥和证书。如果它们是自签名的，同样适用以下规则，我们需要设置以下内容：

```cpp
server->ignoreSslErrors()
```

您可以通过继承 `QTcpServer` 并重写 `incomingConnection()` 方法来创建一个 SSL 服务器，如下所示。

源代码可以在 Git 仓库的 `Chapter05-6` 目录下的 `cp5` 分支中找到。

我们使用 `override` 函数实现 `header` 文件，以及一个在服务器进入加密模式时连接的槽：

```cpp
class MySslServer : public QTcpServer
{
public:
    MySslServer();
protected:
    void incomingConnection(qintptr handle) override;
private slots:
    void socketEncrypted();
};
```

在 SSL 服务器类的实现中，请注意对 `startServerEncryption()` 的调用。这将启动 `server` 通道的加密并创建一个 `Server`，如下所示：

```cpp
MySslServer::MySslServer()
{
    server = new QSslSocket(this);
    initCerts();
}
```

我们还需要添加加密密钥和证书，因为这与上一节中的 `QSslSocket` 类似，*QSslSocket*：

```cpp

void MySslServer::incomingConnection(qintptr sd)
{
 if (server->setSocketDescriptor(sd)) {
 addPendingConnection(server);
 connect(server, &QSslSocket::encrypted, this, &MySslServer::socketEncrypted);
 server->startServerEncryption();
 } else {
 delete server;
 }
}

void MySslServer::socketEncrypted()
{
 // entered encrypted mode, time to write secure transmissions
}
```

在这里，我们连接到 `QSslSocket` 的 `encrypted` 信号，该信号在 `QSslSocket` 进入加密模式时发出。从那时起，所有发送或接收的字节都将被加密。

错误通过连接到 `sslErrors(const QList<QSslError> &errors)` 信号来处理：

```cpp
connect(server, QOverload<const QList<QSslError> &>::of(&QSslSocket::sslErrors), 
                ={ 
            for (QSslError error : errors) { 
                emit messageOutput(error.errorString()); 
            } 
        });

```

我们还需要连接到 `QAbstractSocket::socketError` 信号来处理这些错误：

```cpp
connect(server, SIGNAL(error(QAbstractSocket::SocketError)), SLOT(error(QAbstractSocket::SocketError)));

```

你还希望连接的其他信号如下：

+   `QSslSocket::connected`

+   `QSslSocket::disconnected`

+   `QSslSocket::encrypted`

+   `QSslSocket::modeChanged`

+   `QSslSocket::stateChanged`

到目前为止，我们一直在使用本地 IP 地址，但当服务器是远程的，我们不仅需要服务器名，还需要它的 IP 地址时会发生什么？让我们探索我们如何使用 Qt 来执行域名查找。

# 查找——查找我

计算机网络，如互联网，依赖于 **域名服务** (**DNS**) 查找。这通常在远程中央服务器上完成，但也可以在本地使用。

有两个类用于执行网络查找——`QDnsLookup` 和 `QHostInfo`。`QHostInfo` 将为主机名提供简单的 IP 地址查找。这实际上只是使用主机名查找 IP 地址。让我们看看我们如何使用它。

# QHostInfo

`QHostInfo` 是平台系统提供的一个简单的用于地址查找的类。它有同步、阻塞的查找方法，或者你可以使用信号/槽，如下所示：

```cpp
QHostInfo hInfo = QHostInfo::fromName("www.packtpub.com");
```

此方法会阻塞，直到收到响应。

`lookupHost` 函数执行异步查找，并接受一个槽作为参数，如下所示：

```cpp
QHostInfo::lookupHost("www.packtpub.com", this, SLOT(lookupResult(QHostInfo)));
```

我们需要实现的槽接收 `QHostInfo` 作为参数，如下所示：

```cpp

void SomeClass::lookupResult(QHostInfo info) 
{
    if (!hInfo.addresses().isEmpty()) {
        QHostAddress address = info.addresses().first();
        qWarning() << address.toString();
    }
}
```

要从这些响应中的任何一个获取地址，可以执行如下操作：

```cpp
if (!hInfo.addresses().isEmpty()) {
    QHostAddress address = info.addresses().first();
    qWarning() << address.toString();
}
```

现在我们继续到 `QDnsLookup`。

# QDnsLookup

`QDnsLookup` 可以查找不同类型的记录，而不仅仅是 IP 地址。你可以使用的值来设置查找类型如下：

+   `A`: IPv4 地址，通过 `hostAddressRecords()` 访问

+   `AAAA`: IPv6 地址，通过 `hostAddressRecords()` 访问

+   `ANY`: 任何记录

+   `CNAME`: 规范名称，通过 `canonicalNameRecords()` 访问

+   `MX`: 邮件交换，通过 `mailExchangeRecords()` 访问

+   `NS`: 名称服务器，通过 `nameServerRecords()` 访问

+   `PTR`: 指针，通过 `pointerRecords()` 访问

+   `SRV`: 服务，通过 `serviceRecords()` 访问

+   `TXT`: 文本，通过 `textRecords()` 访问

让我们看看如何实现这一点。我们将名为`finished`的`QDnsLookup`信号连接到我们的`lookupFinished`槽。我们将类型设置为`TXT`以访问文本记录：

```cpp
QDnsLookup *lookup = new QDnsLookup(this);
connect(lookup, &QDnsLookup::finished, this, &SomeClass::lookupFinished);
lookup->setType(QDnsLookup::TXT);
lookup->setName("example.com");
lookup->lookup();
```

对`lookup()`的调用将开始对设置的名称`example.com`的文本记录进行查找。我们仍然需要处理响应，如下所示：

```cpp
void SomeClass:: lookupFinished()
{
    QDnsLookup *lookup = qobject_cast<QDnsLookup *>(sender());
    if (!lookup)
        return;
    if (lookup->error() != QDnsLookup::NoError) {
        lookup->deleteLater();
        return;
    }
    const QList<QDnsTextRecord> txtRecords = lookup->textRecords();
    for (const QDnsTextRecord &record: txtRecords) {
        const QString recordName = record->name();
        const QList <QByteArray> recordValues = record->values();
        ...
    }
}
```

然后，你可以按需使用这些记录。

# 摘要

`QNetwork`的功能非常广泛。我已经提到了一些特性，例如`QNetworkRequest`、`QNetworkAccessManager`和`QNetworkReply`，它们用于发起网络请求，如`get`和`put`。你可以使用 Qt 的承载管理功能来控制在线状态，以及使用`QNetworkSession`将连接分组在一起，以便在连接之间漫游。我们讨论了使用`QLocalSocket`、`QLocalServer`、`QTcpSocket`和`QTcpServer`进行套接字开发。你可以使用`QHostInfo`和`QDnsLookup`进行主机和 DNS 查找。

连接性可以意味着几件事情，在下一章中，我们将探讨使用蓝牙**低功耗**（**LE**）进行连接性。
