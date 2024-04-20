# 即时通讯

企业软件的一个重要特性是与员工进行通信的能力。因此，内部即时通讯系统是软件的一个关键部分。通过在 Qt 中整合网络模块，我们可以轻松地创建一个聊天系统。

在本章中，我们将涵盖以下主题：

+   Qt 网络模块

+   创建即时通讯服务器

+   创建即时通讯客户端

使用 Qt 创建即时通讯系统比你想象的要容易得多。让我们开始吧！

# Qt 网络模块

在接下来的部分，我们将学习 Qt 的网络模块以及它如何帮助我们通过 TCP 或 UDP 连接协议实现服务器-客户端通信。

# 连接协议

Qt 的网络模块提供了低级网络功能，如 TCP 和 UDP 套接字，以及用于网络集成和网络通信的高级网络类。

在本章中，我们将使用 TCP（传输控制协议）互联网协议，而不是 UDP（用户数据报协议）协议。主要区别在于 TCP 是一种面向连接的协议，要求所有客户端在能够相互通信之前必须与服务器建立连接。

另一方面，UDP 是一种无连接的协议，不需要连接。客户端只需将需要发送到目的地的任何数据发送出去，而无需检查数据是否已被另一端接收。两种协议都有利弊，但 TCP 更适合我们的示例项目。我们希望确保每条聊天消息都被接收者接收到，不是吗？

两种协议之间的区别如下：

+   TCP：

+   面向连接的协议

+   适用于需要高可靠性的应用程序，对数据传输时间不太关键

+   TCP 的速度比 UDP 慢

+   在发送下一个数据之前，需要接收客户端的确认收据

+   绝对保证传输的数据保持完整，并按发送顺序到达目的地

+   UDP：

+   无连接协议

+   适用于需要快速、高效传输的应用程序，如游戏和 VOIP

+   UDP 比 TCP 轻量且更快，因为不会尝试错误恢复

+   也适用于需要从大量客户端回答小查询的服务器

+   没有保证发送的数据是否到达目的地，因为没有跟踪连接，也不需要接收客户端的任何确认

由于我们不打算采用点对点连接的方法，我们的聊天系统将需要两个不同的软件部分——服务器程序和客户端程序。服务器程序将充当中间人（就像邮递员一样），接收所有用户的消息并将它们发送给相应的接收者。服务器程序将被锁定在服务器房间的一台计算机中，普通用户无法接触。

另一方面，客户端程序是所有用户使用的即时通讯软件。这个程序将安装在用户的计算机上。用户可以使用这个客户端程序发送消息，并查看其他人发送的消息。我们的消息系统的整体架构看起来像这样：

![](img/26bb7700-45cf-4482-9232-4eb2ce750839.png)

让我们继续设置我们的项目并启用 Qt 的网络模块！对于这个项目，我们将先从服务器程序开始，然后再处理客户端程序。

# 设置新项目

首先，创建一个新的 Qt 控制台应用程序项目。然后，打开项目文件（.pro）并添加以下模块：

```cpp
QT += core network 
Qt -= gui 
```

你应该已经注意到，这个项目没有任何`gui`模块（我们确保它被明确删除），因为服务器程序不需要任何用户界面。这也是为什么我们选择了 Qt 控制台应用程序而不是通常的 Qt 小部件应用程序的原因。

实际上，就是这样——你已经成功地将网络模块添加到了你的项目中。在下一节中，我们将学习如何为我们的聊天系统创建服务器程序。

# 创建即时通讯服务器

在接下来的部分，我们将学习如何创建一个即时通讯服务器，接收用户发送的消息并将其重新分发给各自的接收者。

# 创建 TCP 服务器

在这一部分，我们将学习如何创建一个 TCP 服务器，不断监听特定端口以接收传入的消息。为了简单起见，我们将创建一个全局聊天室，其中每个用户都可以看到聊天室内每个用户发送的消息，而不是一个一对一的消息系统带有好友列表。一旦你了解了聊天系统的运作方式，你可以很容易地将这个系统改进为后者。

首先，转到文件|新建文件或项目，并在 C++类别下选择 C++类。然后，将类命名为`server`，并选择 QObject 作为基类。在创建自定义类之前，确保选中包含 QObject 选项。你也应该注意到了`mainwindow.ui`、`mainwindow.h`和`mainwindow.cpp`的缺失。这是因为在控制台应用程序项目中没有用户界面。

一旦服务器类被创建，让我们打开`server.h`并添加以下头文件、变量和函数：

```cpp
#ifndef SERVER_H 
#define SERVER_H 

#include <QObject> 
#include <QTcpServer> 
#include <QTcpSocket> 
#include <QDebug> 
#include <QVector> 

private: 
   QTcpServer* chatServer; 
   QVector<QTcpSocket*>* allClients; 

public:
   explicit server(QObject *parent = nullptr);
 void startServer();
   void sendMessageToClients(QString message); public slots: void newClientConnection();
  void socketDisconnected();
  void socketReadyRead();
  void socketStateChanged(QAbstractSocket::SocketState state);
```

接下来，创建一个名为`startServer()`的函数，并将以下代码添加到`server.cpp`中的函数定义中：

```cpp
void server::startServer() 
{ 
   allClients = new QVector<QTcpSocket*>; 

   chatServer = new QTcpServer(); 
   chatServer->setMaxPendingConnections(10); 
   connect(chatServer, SIGNAL(newConnection()), this, 
   SLOT(newClientConnection())); 

   if (chatServer->listen(QHostAddress::Any, 8001)) 
   { 
         qDebug() << "Server has started. Listening to port 8001."; 
   } 
   else 
   { 
         qDebug() << "Server failed to start. Error: " + chatServer-
         >errorString(); 
   } 
} 
```

我们创建了一个名为`chatServer`的`QTcpServer`对象，并使其不断监听端口`8001`。你可以选择从`1024`到`49151`范围内的任何未使用的端口号。此范围之外的其他数字通常保留用于常见系统，如 HTTP 或 FTP 服务，因此最好不要使用它们以避免冲突。我们还创建了一个名为`allClients`的`QVector`数组，用于存储所有连接的客户端，以便我们以后可以利用它来将传入的消息重定向到所有用户。

我们还使用了`setMaxPendingConnections()`函数来限制最大挂起连接数为 10 个客户端。你可以使用这种方法来保持活动客户端的数量，以便服务器的带宽始终在其限制范围内。这可以确保良好的服务质量并保持积极的用户体验。

# 监听客户端

每当客户端连接到服务器时，`chatServer`将触发`newConnection()`信号，因此我们将该信号连接到我们的自定义槽函数`newClientConnection()`。槽函数如下所示：

```cpp
void server::newClientConnection() 
{ 
   QTcpSocket* client = chatServer->nextPendingConnection(); 
   QString ipAddress = client->peerAddress().toString(); 
   int port = client->peerPort(); 

   connect(client, &QTcpSocket::disconnected, this, &server::socketDisconnected); 
   connect(client, &QTcpSocket::readyRead, this, &server::socketReadyRead); 
   connect(client, &QTcpSocket::stateChanged, this, &server::socketStateChanged); 

   allClients->push_back(client); 

   qDebug() << "Socket connected from " + ipAddress + ":" + QString::number(port); 
} 
```

每个连接到服务器的新客户端都是一个`QTcpSocket`对象，可以通过调用`nextPendingConnection()`从`QTcpServer`对象中获取。你可以通过调用`peerAddress()`和`peerPort()`分别获取有关客户端的信息，如其 IP 地址和端口号。然后我们将每个新客户端存储到`allClients`数组中以供将来使用。我们还将客户端的`disconnected()`、`readyRead()`和`stateChanged()`信号连接到其相应的槽函数。

当客户端从服务器断开连接时，将触发`disconnected()`信号，随后将调用`socketDisconnected()`槽函数。在这个函数中，我们只是在服务器控制台上显示消息，当它发生时，什么都不做。你可以在这里做任何你喜欢的事情，比如将用户的离线状态保存到数据库等。为了简单起见，我们将在控制台窗口上打印出消息：

```cpp
void server::socketDisconnected() 
{ 
   QTcpSocket* client = qobject_cast<QTcpSocket*>(QObject::sender()); 
   QString socketIpAddress = client->peerAddress().toString(); 
   int port = client->peerPort(); 

   qDebug() << "Socket disconnected from " + socketIpAddress + ":" + 
   QString::number(port); 
} 
```

接下来，每当客户端向服务器发送消息时，`readyRead()`信号将被触发。我们已经将该信号连接到一个名为`socketReadyRead()`的槽函数，它看起来像这样：

```cpp
void server::socketReadyRead() 
{ 
   QTcpSocket* client = qobject_cast<QTcpSocket*>(QObject::sender()); 
   QString socketIpAddress = client->peerAddress().toString(); 
   int port = client->peerPort(); 

   QString data = QString(client->readAll()); 

   qDebug() << "Message: " + data + " (" + socketIpAddress + ":" + 
   QString::number(port) + ")"; 

   sendMessageToClients(data); 
} 
```

在上述代码中，我们只是简单地将消息重定向到一个名为`sendMessageToClients()`的自定义函数中，该函数处理将消息传递给所有连接的客户端。我们将在一分钟内看看这个函数是如何工作的。我们使用`QObject::sender()`来获取发出`readyRead`信号的对象的指针，并将其转换为`QTcpSocket`类，以便我们可以访问其`readAll()`函数。

之后，我们还将另一个名为`stateChanged()`的信号连接到`socketStateChanged()`槽函数。慢函数看起来像这样：

```cpp
void server::socketStateChanged(QAbstractSocket::SocketState state) 
{ 
   QTcpSocket* client = qobject_cast<QTcpSocket*>(QObject::sender()); 
   QString socketIpAddress = client->peerAddress().toString(); 
   int port = client->peerPort(); 

   QString desc; 

   if (state == QAbstractSocket::UnconnectedState) 
         desc = "The socket is not connected."; 
   else if (state == QAbstractSocket::HostLookupState) 
         desc = "The socket is performing a host name lookup."; 
   else if (state == QAbstractSocket::ConnectingState) 
         desc = "The socket has started establishing a connection."; 
   else if (state == QAbstractSocket::ConnectedState) 
         desc = "A connection is established."; 
   else if (state == QAbstractSocket::BoundState) 
         desc = "The socket is bound to an address and port."; 
   else if (state == QAbstractSocket::ClosingState) 
         desc = "The socket is about to close (data may still be 
         waiting to be written)."; 
   else if (state == QAbstractSocket::ListeningState) 
         desc = "For internal use only."; 

   qDebug() << "Socket state changed (" + socketIpAddress + ":" + 
   QString::number(port) + "): " + desc; 
} 
```

此函数在客户端的网络状态发生变化时触发，例如连接、断开连接、监听等。我们将根据其新状态简单地打印出相关消息，以便更轻松地调试我们的程序。

现在，让我们看看`sendMessageToClients()`函数的样子：

```cpp
void server::sendMessageToClients(QString message) 
{ 
   if (allClients->size() > 0) 
   { 
         for (int i = 0; i < allClients->size(); i++) 
         { 
               if (allClients->at(i)->isOpen() && allClients->at(i)-
               >isWritable()) 
               { 
                     allClients->at(i)->write(message.toUtf8()); 
               } 
         } 
   } 
} 
```

在上述代码中，我们只是简单地循环遍历`allClients`数组，并将消息数据传递给所有连接的客户端。

最后，打开`main.cpp`并添加以下代码来启动我们的服务器：

```cpp
#include <QCoreApplication> 
#include "server.h" 

int main(int argc, char *argv[]) 
{ 
   QCoreApplication a(argc, argv); 

   server* myServer = new server(); 
   myServer->startServer(); 

   return a.exec(); 
} 
```

现在构建并运行程序，你应该看到类似这样的东西：

![](img/07666326-5c7d-4633-8a02-641e3ae73af5.png)

除了显示服务器正在监听端口`8001`之外，似乎没有发生任何事情。别担心，因为我们还没有创建客户端程序。让我们继续！

# 创建即时通讯客户端

在接下来的部分中，我们将继续创建我们的即时通讯客户端，用户将使用它来发送和接收消息。

# 设计用户界面

在本节中，我们将学习如何为即时通讯客户端设计用户界面并为其创建功能：

1.  首先，通过转到文件|新建文件或项目来创建另一个 Qt 项目。然后在应用程序类别下选择 Qt Widget 应用程序。

1.  项目创建后，打开`mainwindow.ui`并将一个行编辑和文本浏览器拖放到窗口画布中。然后，选择中央窗口小部件并单击位于上方小部件栏上的“垂直布局”按钮，以将垂直布局效果应用到小部件上：

![](img/e12c2a26-b9f7-4a29-be49-3e3e35eaa0c8.png)

1.  之后，在底部放置一个水平布局，并将行编辑放入布局中。然后，从小部件框中拖放一个按钮到水平布局中，并将其命名为`sendButton`；我们还将其标签设置为`Send`，就像这样：

![](img/b6569033-78ae-4f24-8c92-52d0d24ce323.png)

1.  完成后，将另一个水平布局拖放到文本浏览器顶部。然后，将标签、行编辑和一个按钮放入水平布局中，就像这样：

![](img/65759c61-0e68-4ef7-803d-91247e04d7ae.png)

我们将行编辑小部件称为`nameInput`，并将其默认文本设置为`John Doe`，这样用户就有了默认名称。然后，我们将推按钮称为`connectButton`，并将其标签更改为`Connect`。

我们已经完成了一个非常简单的即时通讯程序的用户界面设计，它将执行以下任务：

1.  连接到服务器

1.  让用户设置他们的名字

1.  可以看到所有用户发送的消息

1.  用户可以输入并发送他们的消息供所有人查看

现在编译并运行项目，你应该看到你的程序看起来类似这样：

![](img/b412d5d1-4f08-44fb-bcd5-3da628cdfb2a.png)

请注意，我还将窗口标题更改为`Chat Client`，这样看起来稍微更专业一些。您可以通过在层次结构窗口中选择`MainWindow`对象并更改其`windowTitle`属性来实现。

在下一节中，我们将开始进行编程工作，并实现上面列表中提到的功能。

# 实现聊天功能

在我们开始编写任何代码之前，我们必须通过打开项目文件（`.pro`）并在那里添加 `network` 关键字来启用网络模块：

```cpp
QT += core gui network 
```

接下来，打开 `mainwindow.h` 并添加以下头文件和变量：

```cpp
#ifndef MAINWINDOW_H 
#define MAINWINDOW_H 

#include <QMainWindow> 
#include <QDebug> 
#include <QTcpSocket> 

private: 
   Ui::MainWindow *ui; 
   bool connectedToHost; 
   QTcpSocket* socket; 
```

我们在 `mainwindow.cpp` 中默认将 `connectedToHost` 变量设置为 `false`：

```cpp
MainWindow::MainWindow(QWidget *parent) : 
   QMainWindow(parent), 
   ui(new Ui::MainWindow) 
{ 
   ui->setupUi(this); 
   connectedToHost = false; 
} 
```

完成此操作后，我们需要实现的第一个功能是服务器连接。打开 `mainwindow.ui`，右键单击连接按钮，然后选择转到槽...，然后选择 `clicked()`。之后，将自动为您创建一个槽函数。在 `SLOT` 函数中添加以下代码：

```cpp
void MainWindow::on_connectButton_clicked() 
{ 
   if (!connectedToHost) 
   { 
         socket = new QTcpSocket(); 

         connect(socket, SIGNAL(connected()), this, 
         SLOT(socketConnected())); 
         connect(socket, SIGNAL(disconnected()), this, 
         SLOT(socketDisconnected())); 
         connect(socket, SIGNAL(readyRead()), this, 
         SLOT(socketReadyRead())); 

         socket->connectToHost("127.0.0.1", 8001); 
   } 
   else 
   { 
         QString name = ui->nameInput->text(); 
         socket->write("<font color="Orange">" + name.toUtf8() + " has 
         left the chat room.</font>"); 

         socket->disconnectFromHost(); 
   } 
} 
```

在前面的代码中，我们基本上是检查了 `connectedToHost` 变量。如果变量为 `false`（表示客户端未连接到服务器），则创建一个名为 `socket` 的 `QTcpSocket` 对象，并使其连接到端口 `8801` 上的 `127.0.0.1` 主机。IP 地址 `127.0.0.1` 代表本地主机。由于这仅用于测试目的，我们将客户端连接到位于同一台计算机上的测试服务器。如果您在另一台计算机上运行服务器，则可以根据需要将 IP 地址更改为局域网或广域网地址。

当 `connected()`、`disconnected()` 和 `readReady()` 信号被触发时，我们还将 `socket` 对象连接到其相应的槽函数。这与我们之前所做的服务器代码完全相同。如果客户端已连接到服务器并且单击了连接（现在标记为 `Disconnect`）按钮，则向服务器发送断开连接消息并终止连接。

接下来，我们将看看槽函数，这些槽函数在上一步中连接到了 `socket` 对象。第一个是 `socketConnected()` 函数，当客户端成功连接到服务器时将被调用：

```cpp
void MainWindow::socketConnected() 
{ 
   qDebug() << "Connected to server."; 

   printMessage("<font color="Green">Connected to server.</font>"); 

   QString name = ui->nameInput->text(); 
   socket->write("<font color="Purple">" + name.toUtf8() + " has joined 
   the chat room.</font>"); 

   ui->connectButton->setText("Disconnect"); 
   connectedToHost = true; 
} 
```

首先，客户端将在应用程序输出和文本浏览器小部件上显示 `Connected to server.` 消息。我们马上就会看到 `printMessage()` 函数是什么样子。然后，我们从输入字段中获取用户的名称，并将其合并到文本消息中，然后将其发送到服务器，以便通知所有用户。最后，将连接按钮的标签设置为 `Disconnect`，并将 `connectedToHost` 变量设置为 `true`。

接下来，让我们看看 `socketDisconnected()`，正如其名称所示，每当客户端从服务器断开连接时都会被调用：

```cpp
void MainWindow::socketDisconnected() 
{ 
   qDebug() << "Disconnected from server."; 

   printMessage("<font color="Red">Disconnected from server.</font>"); 

   ui->connectButton->setText("Connect"); 
   connectedToHost = false; 
} 
```

前面的代码非常简单。它只是在应用程序输出和文本浏览器小部件上显示断开连接的消息，然后将断开按钮的标签设置为 `Connect`，将 `connectedToHost` 变量设置为 `false`。请注意，由于此函数仅在客户端从服务器断开连接后才会被调用，因此我们无法在那时向服务器发送任何消息以通知它断开连接。您应该在服务器端检查断开连接并相应地通知所有用户。

然后是 `socketReadyRead()` 函数，每当服务器向客户端发送数据时都会触发该函数。这个函数比之前的函数更简单，因为它只是将传入的数据传递给 `printMessage()` 函数，什么都不做：

```cpp
void MainWindow::socketReadyRead() 
{ 
   ui->chatDisplay->append(socket->readAll()); 
} 
```

最后，让我们看看 `printMessage()` 函数是什么样子。实际上，它就是这么简单。它只是将消息附加到文本浏览器中，然后完成：

```cpp
void MainWindow::printMessage(QString message) 
{ 
   ui->chatDisplay->append(message); 
} 
```

最后但同样重要的是，让我们看看如何实现向服务器发送消息的功能。打开 `mainwindow.ui`，右键单击发送按钮，选择转到槽...，然后选择 `clicked()` 选项。一旦为您创建了槽函数，将以下代码添加到函数中：

```cpp
void MainWindow::on_sendButton_clicked() 
{ 
   QString name = ui->nameInput->text(); 
   QString message = ui->messageInput->text(); 
   socket->write("<font color="Blue">" + name.toUtf8() + "</font>: " + 
   message.toUtf8()); 

   ui->messageInput->clear(); 
} 
```

首先，我们获取用户的名称并将其与消息组合在一起。然后，在将整个内容发送到服务器之前，我们将名称设置为蓝色，通过调用`write()`来发送。之后，清除消息输入字段，完成。由于文本浏览器默认接受富文本，我们可以使用`<font>`标签来为文本着色。

现在编译并运行项目；您应该能够在不同的客户端之间进行聊天！在连接客户端之前，不要忘记打开服务器。如果一切顺利，您应该会看到类似于这样的内容：

![](img/3e596014-2a86-4f29-b996-da0d30ff5cd9.png)

同时，您还应该在服务器端看到所有的活动：

![](img/0fc60d96-fc57-4fe5-b4cd-a459653d6dcf.png)

到此为止！我们已经成功使用 Qt 创建了一个简单的聊天系统。欢迎您在此基础上进行改进，创建一个完整的消息传递系统！

# 总结

在本章中，我们学习了如何使用 Qt 的网络模块创建即时消息传递系统。在接下来的章节中，我们将深入探讨使用 Qt 进行图形渲染的奇妙之处。
