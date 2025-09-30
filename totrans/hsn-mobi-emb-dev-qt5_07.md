# Qt Network for Communication

Networking is almost as important to mobile devices as the device being mobile. Without networking, data would have to be physically moved from one place to another. Luckily, Qt has extensive networking features in `QNetwork`. In this chapter, we will discuss the following APIs:

*   `QNetworkReply`
*   `QNetworkRequest`
*   `QDnsLookup`
*   `QHostInfo`
*   `QLocalServer`
*   `QTcpSocket`

To show available Wi-Fi networks that are nearby, we will also go over the following:

*   `QNetworkSession`
*   `QNetworkConfiguration`

You will also learn how to use Qt APIs for standard networking tasks, such as **Domain Name Service** (**DNS**) lookups, download and upload files, and how to use Qt's socket classes for communication.

# High level – request, reply, and access

Networking in Qt is quite feature-rich. Networking in Qt Quick is more behind the scenes than in your face. In **Qt Modeling Language** (**QML**), you can download remote components and use them in your application, but any other arbitrary download or network functionality you will have to bake yourself in the C++ backend or use JavaScript.

Even though `QNetworkRequest`, `QNetworkReply`, and `QNetworkAccessManager` are all used to make network requests, let's split them up and see how to use them.

# QNetworkRequest

`QNetworkRequest` is a part of the access functionality. It constructs a `request`, which can be one of the following verbs:

*   `GET`: `get(...)`
*   `POST`: `post(...)`
*   `PUT`: `put(...)`
*   `DELETE`: `deleteResource(...)`
*   `HEAD`: `head(...)`

You can also send custom verbs using `sendCustomRequest`, which takes the custom verb as a `QByteArray` argument.

Headers can be set as known headers using `setHeader` and can be one of the following:

*   `ContentDispositionHeader`
*   `ContentTypeHeader`
*   `ContentLengthHeader`
*   `LocationHeader`
*   `LastModifiedHeader`
*   `CookieHeader`
*   `SetCookieHeader`
*   `UserAgentHeader`
*   `ServerHeader`

Raw or custom headers can be set with `setRawHeader`. HTTP attributes can help to control the request cache, redirect, and cookies. They can be set with, you guessed it, `setAttribute`.

Let's put this into the following code.

The source code can be found on the Git repository under the `Chapter05-1` directory, in the `cp5` branch.

To use the networking module, in the `.pro` project, add `network` to the `QT` variable as follows:

```cpp
QT += network
```

We can now use Qt Networking.

`QNetworkRequest` is what needs to be used to request operations from the network such as `get` and `put`.

A simple implementation looks like this:

```cpp
QNetworkRequest request;
request.setUrl(QUrl("http://www.example.com"));
```

`QNetworkRequest` can also take `QUrl` as its argument. `QNetworkRequest` is not based on `QObject`, so it has no parent, nor does it have any of its own signals. All communication is done through `QNetworkAccessManager`.

The one signal you want to connect to is the `finished` signal.

Suppose I have some form data I need to transfer; I would need to add a standard header with `setHeader`. I could also add the following custom header I call `X-UUID`:

```cpp
request.setHeader(QNetworkRequest::ContentTypeHeader, "application/x-www-form-urlencoded");
request.setRawHeader(QByteArray("X-UUID"), QUuid::createUuid().toByteArray());
```

Now that we have a viable `QNetworkRequest`, we need to send it to `QNetworkAccessManager`. Let's take a look at how we can do that.

# QNetworkAccessManager

Bring in the manager—`QNetworkAccessManager` (**QNAM**). It is used to send and receive asynchronous requests over a network. Usually, there is one instance of QNAM in an application, as here:

```cpp
QNetworkAccessManager *manager = new QNetworkAccessManager(this);
```

At its simplest, you can make a QNAM request using the `get`, `put`, `post`, `deleteResource`, or `head` functions.

QNAM uses signals to transfer data and request information and the `finished()` signal is used to signal when a request has finished.

Let's add a signal handler for that, as follows:

```cpp
 connect(manager, &QNetworkAccessManager::finished, 
        this, &MainWindow::replyFinished);
```

This would call your `replyFinished` slot with the data and headers within the `QNetworkReply` argument, as follows:

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

Then, call the `get` method on `QNetworkAccessManager` as follows:

```cpp
manager->get(request);
```

It's as simple as that to download something! QNAM will work its magic and download the URL.

It is also just as easy a method to create a file upload. Of course, your web server needs to support the `put` method, as follows:

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

The source code can be found on the Git repository under the `Chapter05-2` directory, in the `cp5` branch.

If you need to send some query parameters in the URL, you can use `QUrlQuery` to construct the `form` query data, and then send the `request` as follows:

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

Form data can be uploaded with the `post` function as a `QByteArray` as follows:

```cpp
QByteArray postData;
postData.append("?login=me&password=123&submit=Send");
manager->post(request, postData);
```

To send a multipart form data, such as form data and an image, you can use `QHttpMultiPart` as follows:

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

Of course, none of these examples keeps track of the reply. `QNetworkReply` is returned by the `get`, `post`, and `put` methods of `QNetworkAccessManager`, which can be used to track download or upload progress or if there are any errors.

# QNetworkReply

All calls to QNAM's `get`, `post`, and so on, will return `QNetworkReply`.

You will need to delete this pointer, otherwise it will leak memory, but do not delete it in the `finished` signal handler. You can use `deleteLater()`.

`QNetworkReply` has an interesting signal we would most likely need to handle. Let's start with the two most important—`error` and `readyRead`.

So, let's handle that `QNetworkReply` properly. Since we do not have the valid object beforehand, we need to connect the signals after the network request action. It seems a bit backward to me, but this is the way it needs to be and it works. The code is as follows:

```cpp
QNetworkReply *networkReply = manager->get(request);
connect(networkReply, SIGNAL(finished()), this, SLOT(requestFinished()));
connect(networkReply, SIGNAL(error(QNetworkReply::NetworkError)),
    this,SLOT(networkReplyError(QNetworkReply::NetworkError)));
 connect(networkReply, SIGNAL(readyRead()), this, SLOT(readyRead()));

```

I am using the legacy style of signal connections, but you could and should write connections like the following because it allows compile time checking for syntax and other errors:

```cpp
connect(networkReply, &QNetworkReply::error, this, &MyClass::networkReplyError);
 connect(networkReply, QOverload<QNetworkReply::NetworkError>::of(&QNetworkReply::error),this, &MyClass::networkReplyError);
connect(networkReply, &QNetworkReply::finished, this, &MyClass::requestFinished);
connect(networkReply, &QNetworkReply::readyRead, this, &MyClass::readyRead);
```

So, now we have done a request and are waiting for a reply from the server. Let's look at the signal handlers one by one.

`error(QNetworkReply::NetworkError)` is emitted when there is an error with the error code as argument. If you need a user-friendly string, you can retrieve that with `QNetworkReply::errorString()`. `finished()` is emitted when the request is finished. The reply is still open, so you can read it here: `readyRead()` .Since the reply is derived from `QIODevice`, it has the `readyRead` signal, which is emitted whenever more data is ready to read.

On large downloads, you might want to monitor the progress of the download, which is a common thing to do. Usually, every download has some kind of progress bar. `QNetworkReply` emits the `downloadProgress(qint64 bytesReceived, qint64 bytesTotal)` signal as follows:

```cpp
connect(networkReply, &QNetworkReply::downloadProgress, this, &MyClass::onDownloadProgress);
```

There is the corresponding `uploadProgress` for uploads.

`preSharedKeyAuthenticationRequired(QSslPreSharedKeyAuthenticator *authenticator)` gets emitted when the download needs authentication. The `QSslPreSharedKeyAuthenticator` object should be loaded with the pre-shader key and other details to authenticate the user.

The `sslErrors(const QList<QSslError> &errors)` signal is called when **Secure Sockets Layer** (**SSL**) encounters problems, including certificate verification errors.

`QNetworkManager` can also do simple **File Transfer Protocol** (**FTP**) transfers.

# QFtp

There are two ways to use FTP with Qt. `QNetworkAccessManager` has simple FTP `get` and `put` support, we can easily use that.

FTP servers usually require some sort of username and password. We use `setUserName()` and `setPassword()` of `QUrl` to set these, as follows:

```cpp
 QUrl url("ftp://llornkcor.com/");
 url.setUserName("guest@llornkcor.com");
 url.setPassword("handsonmobileandembedded");
```

The source code can be found on the Git repository under the `Chapter05-5` directory, in the `cp5` branch.

Once we know the file's name, we need to add that to the `url`, as it will use this to write the fail, as follows:

```cpp
        url.setPath(QFileInfo(file).fileName());
```

Then, set the request `url`, as follows:

```cpp
 request.setUrl(url);
```

We can hook up slots to the `QNetworkReply` signals, once we call `put` on the QNAM, as follows:

```cpp
QNetworkReply *networkReply = manager->put(request, fileBytes);

connect(networkReply, &QNetworkReply::downloadProgress, 
    this, &MainWindow::onDownloadProgress);
connect(networkReply, &QNetworkReply::downloadProgress, 
    this, &MainWindow::onUploadProgress);
```

Do not forget that `error` signal needs `QOverload` as follows:

```cpp
connect(networkReply, QOverload<QNetworkReply::NetworkError>::of(&QNetworkReply::error),               [=](QNetworkReply::NetworkError code){
    qDebug() << Q_FUNC_INFO << code << networkReply->errorString(); });

connect(networkReply, &QNetworkReply::finished, 
    this, &MainWindow::requestFinished);
```

If you need to do more complicated things other than `get` and `put`, you will need to use something else besides `QNetworkAccessManager`.

`QFtp` is not included with Qt, but you can access the standalone `QFtp` module that was ported from Qt 4 to run with Qt 5 as follows:

```cpp
git clone -b 5.12 git://code.qt.io/qt/qtftp.git

```

We will need to build `QFtp`, so we can open the `qtftp.pro` in Qt Creator. Run Build and install that.

Using the command line the commands would be as follows:

```cpp
cd qtftp
qmake
make
make install
```

We will need to install this into Qt 5.12, so in Qt Creator, navigate to Projects | Build | Build Steps and select Add Build Step | Make. In the arguments field, type `install`.

Build this and it will also install.

In the project's `.pro` file, to tell `qmake` to use the `network` and `ftp` modules, add the following:

`QT += network ftp`

`QFtp` works very typically; log in, do operations, and then log out, as follows:

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

We connect to the `commandFinished` signal, which can tell us whether there was an error.

The `stateChanged` signal will tell us when we are logged in and the `dataTransferProgress` signal will tell us when bytes are being transferred.

`QFtp` supports other operations, including the following:

*   `list`
*   `cd`
*   `remove`
*   `mkdir`
*   `rmdir`
*   `rename`

QNAM also touches upon my favorite part of Qt Network—Bearer Management. Let's move on to learning about Bearer Management.

# Bearer Management of good news

Bearer Management was meant to facilitate user control over the network connections. There are `open` and `close` functions for found connections. One thing it does not do is actually configure these connections. They must already be configured by the system.

It is also meant to be able to group connections together to make it easier to smoothly switch between connections, such as migrating from Wi-Fi to mobile cellular data, somewhat like **Media Independent Handover** (**MIH**) or also **Unlicensed Mobile Access** (**UMA**) specification. If you are interested in an open source library to help with handovers, look at Open MIH at SourceForge.

At the time Qt's Bearer Management was first developed, Symbian was the most used and arguably the most important mobile OS. Symbian had the ability to seamlessly migrate connections between technologies without dropping the connection or data, kind of like the way mobile phone connections get migrated from cell tower to cell tower.

Apple seems to call this Wi-Fi Assist; Samsung has Auto Network Switching.

Years ago, mobile data connections were very expensive, so the connection was often closed after a specific upload or download happened. The opening and closing of connections was more dynamic and needed automatic controls.

At any rate, `QtConfigurationManager` will use what the system supports; it does not implement its own connection data migration.

Qt has the following three main classes that make up Bearer Management:

*   `QNetworkConfiguration`
*   `QNetworkConfigurationManager`
*   `QNetworkSession`

There is also `QBearerEngine`, which is the base class for bearer plugins.

# QNetworkConfiguration

`QNetworkConfiguration` represents a network connection configuration, such as a Wi-Fi connection to a particular access point with its **Service Set Identifier** (**SSID**) as the configuration name.

The network configuration can be one of the following types:

*   `QNetworkConfiguration::InternetAccessPoint`:
    *   This type is a typical access point, such as a Wi-Fi **Access Point** (**AP**) or it could represent an Ethernet or mobile network.

*   `QNetworkConfiguration::ServiceNetwork`:
    *   A `ServiceNetwork` type is a group of access points known as a **Service Network Access Point** (**SNAP**). The system will determine which of the service networks is best to connect with based on criteria such as cost, speed, and availability. A configuration of the `QNetworkConfiguration::ServiceNetwork` type may also roam between its children `QNetworkConfiguration::InternetAccessPoint`.
*   `QNetworkConfiguration::UserChoice`:
    *   This type can represent a user preferred configuration. It was used by Nokia's Maemo and Symbian platforms in which the system could pop up a dialog asking the user to choose which AP was best. None of the current bearer backends use this type of `QNetworkConfiguration`.

Often, we need to know the type of bearer, which is to say, what communication protocol the connection is using. Let's find out about `BearerType`.

# QNetworkConfiguration::BearerType

This is an `enum` that specifies what the underlying technology of `QNetworkConfiguration` is and can be one of the following:

*   `BearerEthernet`
*   `BearerWLAN`
*   `Bearer2G`
*   `BearerCDMA2000`
*   `BearerWCDMA`
*   `BearerHSPA`
*   `BearerBluetooth`
*   `BearerWiMAX`
*   `BearerEVDO`
*   `BearerLTE`
*   `Bearer3G`
*   `Bearer4G`

This can be determined by calling the `bearerType()` function of the `QNetworkConfiguration` object, as follows:

```cpp
QNetworkConfiguration config;
if (config.bearerType() == QNetworkConfiguration::Bearer4G)
    qWarning() << "Config is using 4G";
```

You can open or connect.

# QNetworkConfiguration::StateFlags

`StateFlags` are an OR'd `||`,combination of the `StateFlag` values, which are as follows:

*   `Defined`: Known to the system but not yet configured
*   `Discovered`: Known and configured, can be used to `open()`
*   `Active`: Currently online

A `QNetworkConfiguration` that has an `Active` flag will also have the `Discovered` and `Defined` flags as well. You can check to see whether a configuration is active by doing this:

```cpp
QNetworkConfiguration config;
if (config.testFlag(QNetworkConfiguration::Active))
    qWarning() << "Config is active";
```

# QNetworkConfigurationManager

`QNetworkConfigurationManager` allows you to obtain `QNetworkConfigurations` of the system, as follows:

```cpp
QNetworkConfigurationManager manager;
QNetworkConfiguration default = manager.defaultConfiguration();
```

It's always wise to wait for the `updateCompleted` signal from `QNetworkConfigurationManager` before using it, to be sure the configurations are set up properly.

A default configuration is the configuration that the system defines as the default. It could have a state of `Active` or just `Discovered`.

If you need to simply determine whether the system is currently online, `manager->isOnline();` will return `true` if the system is considered online. Online is when it is connected to another device via a network, which may or may nor be the internet, and may or may not be routed correctly. So, it could be online, but cannot access the internet.

You may need to call `updateConfigurations()`, which will ask the system to update the list of configurations, and then you need to listen for the `updateCompleted` signal before proceeding.

You can get all configurations known to the system with a call to `allConfigurations()`, or filter it to the ones that have a certain state with `allConfigurations(QNetworkConfiguration::Discovered);`.

In this case, it returns a list of `Discovered` configurations.

You can check the system's capabilities with a call to `capabilities()`, which can be one of the following:

*   `CanStartAndStopInterfaces`: System allows user to start and stop connections
*   `DirectConnectionRouting`: Connection routing is bound directly to a specified device interface
*   `SystemSessionSupport`: System keeps connection open until all sessions are closed
*   `ApplicationLevelRoaming`: Apps can control roaming/migrating
*   `ForcedRoaming`: System will reconnect when roaming/migrating
*   `DataStatics`: System provides information about transmitted and received data
*   `NetworkSessionRequired`: System requires a session

# QNetworkSession

`QNetworkSession` provides a way to start and stop connections as well as providing management of connection sessions. In the case of instantiating `QNetworkSession` with a `QNetworkConfiguration` that is a `ServiceNetwork` type, it can provide roaming features. On most systems, roaming will entail actually disconnecting and then connecting a new interface and/or connection. On others, roaming can be seamless and without disturbing the user's data stream.

If the capabilities of `QNetworkConfigurationManager` reports that it supports `CanStartAndStopInterfaces`, then you use `QNetworkSession` to `open()` (connect) and `stop()` (close) `QNetworkConfigurations`.

The QNAM will use `QNetworkSession` when making network requests behind the scenes. You can use `QNetworkSession` to monitor the connection as follows:

The source code can be found on the Git repository under the `Chapter05-3` directory, in the `cp5` branch.

```cpp
QNetworkAccessManager manager;
QNetworkConfiguration config = manager.configuration();
QNetworkSession *networkSession = new QNetworkSession(config, this);
connect(networkSession, &QNetworkSession::opened, this, &SomeClass::sessionOpened);
networkSession->open();
```

To monitor bytes received and sent from a QNAM request, connect up to the `bytesReceived` and `bytesWritten` signals, as follows:

```cpp
connect(networkSession, &QNetworkSession::bytesReceived, this, &SomeClass::bytesReceived); 
connect(networkSession, &QNetworkSession::bytesWritten, this, &SomeClass::bytesWritten); 

QNetworkRequest request(QUrl("http://example.com"));
manager->get(request);
```

# Roaming

By roaming, I mean roaming between Wi-Fi and mobile data, not roaming as in out of the home network, which can be very expensive mobile data to use.

In order to facilitate roaming, a client app can connect to the `preferredConfigurationChanged` signal and then begin the process by calling `migrate()` or cancel it by calling `ignore()`. Migrating a connection could be as simple as pausing the download, disconnecting and reconnecting to the new connection, and then resuming the download. This method is called forced roaming. It can, on some platforms, seamlessly migrate the data stream to the new connection, similar to what a mobile phone does when a call gets migrated to another cell tower.

At this time, there are no currently supported backends that support migrating sessions. A system integrator could implement a backend that does true connection migration and handovers. It would also help if the system allows this.

That said, both Samsung's Android and iOS support roaming features seem to have caught up to where Nokia was years ago. Samsung calls it Adaptive Wi-Fi, previously known as Smart Network Switch. iOS calls it Wi-Fi Assist. These happen at the system level and allow roaming between Wi-Fi and mobile data connections. Neither of these platforms allows applications to control the handover.

# QBearerEngine

Qt comes with the following bearer backend plugins based off of the `QBearerEngine` class:

*   `Android`: Android
*   `Connman`: Linux desktop & embedded, SailfishOS
*   `Corewlan`: Mac OS and iOS

*   `Generic`: All
*   `NativeWifi`: Windows
*   `NetworkManager`: Linux
*   `NLA`: Windows

Depending on the platform, some of these work in conjunction with the generic backend.

# Low level – of sockets and servers

`QTcpSocket` and `QTcpServer` are two classes for sockets used in Qt. They work in much the same way as your web browser and a WWW server. These connect to a network address host, whereas `QLocalSocket` and `QLocalServer` connect to a local file descriptor.

Let's look at `QLocalServer` and `QLocalSocket` first.

In socket server programming, the basic procedure is as follows:

1.  Create a socket
2.  Set socket options
3.  Bind a socket address
4.  Listen for connections
5.  Accept new connection

Qt simplifies these steps to the following:

1.  Create a socket
2.  Listen for connections
3.  Accept new connection

# QLocalServer

If you need communication on the same machine, then `QLocalServer` will be slightly more performant than using a TCP-based socket server. It can be used for **Inter-process communication** (**IPC**).

First, we create the server, and then call the `listen` function with a string name that clients use to connect. We hook up to the `newConnection` signal, so we know when a new client connects.

The source code can be found on the Git repository under the `Chapter05-5a` directory, in the `cp5` branch.

When a client tries to connect, we then send a small message using the `write` function, and finally `flush` the message, as follows:

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

It's that simple! Anytime you need to write to the client, simply use `nextPendingConnection()` to get the next `QLocalSocket` object and use `write` to send the data. Be sure to add `\r\n` to all lines you need to send, including the last line. The call to `flush()` is not required, but it sends the data immediately.

You can keep this object around to send more messages when needed.

Our app is now waiting and listening for connections. Let's do that next.

# QLocalSocket

`QLocalSocket` is used to communicate with `QLocalServer`. You will want to connect to the `readyRead` signal. Other signals are `connected()`, `disconnected()`, `error(...)`, and `stateChanged(...)`, as follows:

The source code can be found on the Git repository under the `Chapter05-5b` directory, in the `cp5` branch.

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

If you need state changes, you connect to `stateChanged` and will be notified when the following states change:

*   `UnconnectedState`
*   `ConnectingState`
*   `ConnectedState`
*   `ClosingState`

Now, we need to actually connect to the server, as follows:

```cpp
lSocket->connectToHost("localSocketName");
```

Like `QLocalServer`, `QLocalSocket` uses the `write` function to send messages to the server, as follows:

```cpp
lSocket->write("local socket OK\r\n");
```

Remember to add the **End Of Line** (**EOL**) `\r\n` to mark the end of the data feed line.

That is a simple local sockets based communication. Now, let's look at a TCP-based socket over a network.

# QTcpServer

The `QTcpServer` API is much like `QLocalServer` and can be pretty much a drop-in replacement with a few small changes. Most notably, the arguments for the listen call are slightly different, in which you need to specify `QHostAddress` for `QTcpServer` instead of a `QString` name, and a port number. Here, I use `QHostAddress::Any`, which means it will listen on all network interfaces. If you don't care about which port is used, set it to `0` as follows:

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

Does it look familiar? `QHostAddress` can be an IPv4 or IPv6 address. You can also specify different ranges of address by using the `QHostAddress::SpecialAddress` `enum` as I did, which can be one of the following:

*   `LocalHost`: `127.0.0.1`
*   `LocalHostIPv6`: `::1`
*   `Broadcast`: `255.255.255.255`
*   `AnyIPv4`: `0.0.0.0`
*   `AnyIPv6`: `::`
*   `Any`: `all IPv4 and IPv6 addresses`

`QTcpServer` has an additional signal to `QLocalServer`—`acceptError`, which gets emitted when an error occurs during the accept phase of a new connection. You can also `pauseAccepting()` and `resumeAccepting()` the accepting of the connections in the pending connection queue.

# QTcpSocket

`QTcpSocket` is similar to `QLocalSocket` as well. Except, among other things, `QTcpSocket` has `connectToHost` as a way to connect to a server, as follows:

```cpp
QTcpSocket *tSocket = new QTcpSocket(this);
connect(tSocket, &QTcpSocket::connected, this, &SomeClass::connected);

connect(tSocket, &QTcpSocket::disconnected, this,
    &SomeClass::disconnected);

connect(tSocket, &QTcpSocket::error, this, &SomeClass::error);
connect(tSocket, &QTcpSocket::readyRead, this, &SomeClass::readData);

```

To make a simple `HTTP` request, we can write to the socket after we are connected, as follows:

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

This will request the `index.html` file from the server. The data can be read in the `readyRead` signal handler, as follows:

```cpp
void SomeClass::readData()
{
    if (tSocket->bytesAvailable())
        QByteArray msg = tSocket->readAll();
}
```

You can also use the `waitForConnected`, `waitForBytesWritten`, and `waitForReadyRead` functions if you do not want to use this more synchronously, as follows:

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

Then, close the connection with the following command:

```cpp
tSocket->close();
```

# QSctpServer

**SCTP** stands for **Stream Control Transmission Protocol**. `QSctpServer` sends messages as groups of bytes like UDP, rather than a stream of bytes like a TCP socket. It also ensures reliable delivery of the packets, like TCP. It can send several messages in parallel or at the same time. It does this by using several connections.

`QSctpServer` can also send a stream of bytes like TCP by setting `setMaximumChannelCount` to `-1`. The first thing you want to do after creating the `QSctpServer` object is `setMaximumChannelCount`. Setting this to `0` will let this use the number of channels that the client uses, as follows:

```cpp
QSctpServer *sctpServer = new QSctpServer(this);
sctpServer->setMaximumChannelCount(8);
```

If you intend to use TCP byte streams, you use the `nextPendingConnection()` function like `QTcpServer` to get a `QTcpSocket` object to communicate with. `QSctpServer` has the additional `nextPendingDatagramConnection()` to communicate with `QSctpSocket`.

To receive bytes in the `newConnection` signal handler, use the following code:

```cpp
QSctpSocket *sSocket = sctpServer->nextPendingDatagramConnection();
```

# QSctpSocket

`QSctpSocket` also has controls for channel count, and as with `QSctpServer`, if you set the maximum channel count to `-1`, it will behave more like TCP sockets and send a data stream instead of message packets. The message blob is called a `datagram`.

To read and write these datagrams, use `readDatagram()` and `writeDatagram()`. Let's examine `QNetworkDatagram`.

To construct `QNetworkDatagram`, you need a `QByteArray` that holds the data message, a `QHostAddress` for the destination, and optionally, a port number. It can be as simple as the following:

```cpp
QNetworkDatagram datagram("Hello Mobile!", QHostAddress("10.0.0.50"), 8888);
sSocket->writeDatagram(datagram);
```

This will send the `"Hello Mobile!"` message to the corresponding server.

# QUdpSocket

`QUdpSocket` sends datagrams such as `QSctpSocket`, but they are not reliable, which means it will not retry to send any datagrams. It is also connectionless and has a restriction on data length of 65,536 bytes.

There are two ways to set up `QUdpSocket`—`bind(...)` and `connectToHost(...)`.

If you use `connectToHost`, you can use `QIODevice` `read()`, `write()`, `readAll()` to send and receive datagrams. Using the `bind(...)` method, you need to use `readDatagram` and `writeDatagram` instead, as follows:

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

Encrypted socket communications can be handled by `QSslSocket`, which uses SSL to encrypt the TCP connection. The encrypted signal is emitted when the connection is secured, as follows:

```cpp
QSslSocket *sslSocket = new QSslSocket(this);
connect(sslSocket, &QSslSocket::encrypted, this, SomeClass::socketEncrypted);
sslSocket->connectToHostEncrypted("example.com", 943);
```

The source code can be found on the Git repository under the `Chapter05-6a` directory, in the `cp5` branch.

This will start the connection and immediately start the secure handshake procedure. Once the handshake is finished with no error, the encrypted signal will be emitted and the connection will be ready.

You will need to add key/certificate pair to `QSslSocket` to utilize the encryption capabilities. You can easily generate key-certificate fail pair for testing by using this web site: [https://www.selfsignedcertificate.com/](https://www.selfsignedcertificate.com/).

Because we are using a self-signed certificate, we will need to add `ignoreSslErrors` in our error handling slot:

```cpp
sslSocket->ignoreSslErrors();
```

To add the encryption key and certificate, you need to open and read both files, and use the resulting `QByteArrays` to create `QSslKey` and `QSslCertificate`:

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

When you run this code, you will need to use `QFileDialog` to navigate and find the `localhost.key` and `localhost.cert` files in the source directory.

Then, we use `setPrivateKey` to set the key file, and `addCaCertificate` and `setLocalCertificate` to add the certificate.

To read from the socket, you can connect to the `readReady` signal like in `QTcpSocket`.

To write to the socket, which transmits to the server, simply use the `write` function:

```cpp
 sslSocket->write(ui->lineEdit->text().toUtf8() +"\r\n");
```

You can then use `QSslSocket` to connect to `QTcpServer` that opens `QSslSocket`. This brings us to our next step.

# QSslServer

Ok, there is no `QSslServer` class, but since the `QSslSocket` class is just derived from `QTcpSocket` with some extra SSL stuff on top, you can create your own SSL server using the functions from `QSslSocket`.

You will need to generate SSL key and certificates. If they are self-signed, the same rules apply, in which we need to set the following:

```cpp
server->ignoreSslErrors()
```

You can create an SSL server by subclassing `QTcpServer` and overriding `incomingConnection()` ,as follows.

The source code can be found on the Git repository under the `Chapter05-6` directory, in the `cp5` branch.

We implement the `header` file with the `override` function, as well as a slot to connect to when the server changes into encrypted mode:

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

In the implementation of the SSL server class, pay attention to the call to `startServerEncryption()`. This will initiate the encryption of the `server` channels and create a `Server`, as follows:

```cpp
MySslServer::MySslServer()
{
    server = new QSslSocket(this);
    initCerts();
}
```

We also need to add the encruption key and certificate, as this uses `QSslSocket` like in the last section, *QSslSocket*:

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

Here, we connect to the `QSslSocket` `encrypted` signal, which signals when `QSslSocket` enters encrypted mode. From then on, all bytes sent or received are encrypted.

Errors are handled by connecting to the `sslErrors(const QList<QSslError> &errors)` signal:

```cpp
connect(server, QOverload<const QList<QSslError> &>::of(&QSslSocket::sslErrors), 
                [=](const QList<QSslError> &errors){ 
            for (QSslError error : errors) { 
                emit messageOutput(error.errorString()); 
            } 
        });

```

We also need to connect to the `QAbstractSocket::socketError` signal to handle those errors as well:

```cpp
connect(server, SIGNAL(error(QAbstractSocket::SocketError)), SLOT(error(QAbstractSocket::SocketError)));

```

Other signals you will also want to connect to are the following:

*   `QSslSocket::connected`
*   `QSslSocket::disconnected`
*   `QSslSocket::encrypted`
*   `QSslSocket::modeChanged`
*   `QSslSocket::stateChanged`

Up till now, we have been using local IP addresses, but what happens when the server is remote and we need not just the server name, but it's IP address? Let's explore how we can use Qt to do domain name lookups.

# Lookups – look me up

Computer networks such as the internet rely on **Domain Name Service** (**DNS**) lookups. This is usually done on remote central servers, but can also be used locally.

There are two classes for doing network lookups—`QDnsLookup` and `QHostInfo`. `QHostInfo` will provide simple IP address lookups for a hostname. It is really just looking up an IP address using a hostname. Let's look at how we can use this.

# QHostInfo

`QHostInfo` is a simple class to do address lookups provided by the platform system. It has synchronous, blocking method for lookup, or you can use signal/slots, as follows:

```cpp
QHostInfo hInfo = QHostInfo::fromName("www.packtpub.com");
```

This method blocks until a response is received.

The `lookupHost` function does asynchronous lookups and takes a slot as an argument, as follows:

```cpp
QHostInfo::lookupHost("www.packtpub.com", this, SLOT(lookupResult(QHostInfo)));
```

The slot we need to implement receives `QHostInfo` as an argument, as such:

```cpp

void SomeClass::lookupResult(QHostInfo info) 
{
    if (!hInfo.addresses().isEmpty()) {
        QHostAddress address = info.addresses().first();
        qWarning() << address.toString();
    }
}
```

To get an address from either of these responses, do something like the following:

```cpp
if (!hInfo.addresses().isEmpty()) {
    QHostAddress address = info.addresses().first();
    qWarning() << address.toString();
}
```

Let's now proceed to `QDnsLookup`.

# QDnsLookup

`QDnsLookup` can look up different types of records, not just IP addresses. The values you can use to set the type of lookup are as follows:

*   `A`: IPv4 addresses, access with `hostAddressRecords()`
*   `AAAA`: IPv6 addresses, access with `hostAddressRecords()`
*   `ANY`: Any record
*   `CNAME`: Canonical name, access with `canonicalNameRecords()`
*   `MX`: Mail exchange, access with `mailExchangeRecords()`
*   `NS`: Name server, access with `nameServerRecords()`
*   `PTR`: Pointer, access with `pointerRecords()`
*   `SRV`: Service, access with `serviceRecords()`
*   `TXT`: Text, access with `textRecords()`

Let's look at how this can be implemented. We connect the `QDnsLookup` signal named `finished` to our `lookupFinished` slot. We set the type here to `TXT` to access text records:

```cpp
QDnsLookup *lookup = new QDnsLookup(this);
connect(lookup, &QDnsLookup::finished, this, &SomeClass::lookupFinished);
lookup->setType(QDnsLookup::TXT);
lookup->setName("example.com");
lookup->lookup();
```

The call to `lookup()` will start a lookup of the text records for the name that we set, which is `example.com`. We still need to handle the response, as follows:

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

You can then use these records in the manner you need.

# Summary

`QNetwork` is quite extensive in what it can do. I have touched upon a few features, such as `QNetworkRequest`, `QNetworkAccessManager`, and `QNetworkReply` to make network requests, such as `get` and `put`. You can use Qt's Bearer Management features to control the online state and `QNetworkSession` to group connections together to roam between connections. We discussed socket development with `QLocalSocket`, `QLocalServer`, `QTcpSocket`, and `QTcpServer`. You can perform host and DNS lookups with `QHostInfo` and `QDnsLookup`.

Connectivity can mean a few things, and in the next chapter, we explore connectivity using Bluetooth **Low Energy** (**LE**).