# 第八章. 从零开始构建实时塔防游戏 - 第二部分，网络

在上一章中，我们从零开始构建了一个完整的游戏。我们遇到的唯一限制是没有真正的敌人可以击败。在本章中，我们将通过添加网络到我们的游戏中来解决这个问题，使其能够与除你之外的其他玩家交互。在本章结束时，你将能够与一些朋友一起玩这个游戏。本章将涵盖以下主题：

+   网络架构

+   使用套接字进行网络通信

+   创建通信协议

+   通过应用客户端-服务器概念修改我们的游戏

+   保存和加载我们的游戏

现在，让我们深入探讨这个相当复杂的章节。

# 网络架构

在构建我们的架构之前，我们需要了解一些关于在游戏中常用哪些网络架构以及它们的具体特点的信息。在游戏编程中使用了不同类型的架构。它们在很大程度上取决于游戏和开发者的需求。我们将看到两种常见的架构：对等网络（P2P）和客户端-服务器。它们各自都有优势和劣势。让我们分别分析它们。

## 对等网络架构

这种架构在过去被广泛使用，至今仍在使用。在这种架构中，玩家知道彼此的地址，并直接相互通信，无需任何中介。例如，对于一个有四个不同玩家的游戏，网络可以表示为以下图表：

![对等网络架构](img/8477OS_08_02.jpg)

这种组织方式允许玩家直接与任何或所有其他玩家互动。当客户端执行某个动作时，它会通知其他玩家这个动作，然后他们相应地更新模拟（游戏）。

这种方法在通信方面效率很高，但也有一些不能忽视的限制。主要的一个是，无法避免作弊。客户端可以通过通知其他玩家该动作来执行任何它想做的事情，即使这是不可能的，比如通过发送任意位置来传送自己。可能的结果是，其他玩家的游戏乐趣完全被破坏。

为了避免这种作弊行为，我们必须改变架构，以便能够有一个可以决定一个动作是否合法的裁判。

## 客户端-服务器架构

在游戏编程中，避免作弊非常重要，因为它可以完全破坏玩家的游戏体验。为了能够减少作弊的可能性，所使用的架构可以提供帮助。使用客户端-服务器架构，游戏可以检测到这些漏洞的大部分。这是证明这一部分重要性的一个原因。另一个观点是，这是我们游戏将使用的架构。玩家之间不会相互通信，他们只会与一个称为服务器的单一主机通信。因为所有其他玩家也会这样做，我们将能够与他们通信，但有一个中介。

此外，这个中介将充当法官，将决定一个动作是否合法。而不是在所有不同玩家的电脑上进行全面模拟，真正的模拟由服务器完成。它持有必须考虑的真实游戏状态；客户端只是我们可以与之交互的一种显示。以下图表表示了架构：

![客户端-服务器架构](img/8477OS_08_03.jpg)

如您所见，我们现在需要通过服务器来传播任何类型的动作给其他玩家。

它的主要缺点是服务器必须对所有玩家（客户端）做出反应，如果你的游戏有大量玩家，这可能会变得很困难。将任务分配到不同的线程对于确保服务器的反应性非常重要。

有些游戏需要的资源太多，以至于它只能处理有限数量的玩家，结果是，你必须为一场游戏管理多个服务器；例如，一个用于登录，另一个用于聊天，另一个用于地图的特定区域，等等。我们现在将看看如何使用这种架构来构建我们的游戏。

在创建多人架构时，首先要考虑的是，我们必须将我们的游戏分成两个不同的程序：一个客户端和一个服务器。我们将有一个服务器托管多个游戏实例和任意数量的客户端，可能在不同场比赛中。

为了能够得到这种结果，我们首先考虑每个部分需要什么。

### 客户端

每个玩家必须启动一个客户端程序才能开始一场比赛。这个程序必须执行以下操作：

+   显示游戏状态

+   处理不同的用户输入

+   播放效果（声音、血腥场面等）

+   根据从服务器接收到的信息更新其游戏状态

+   向服务器发送请求（构建、销毁）

这些不同的功能已经存在于我们的实际游戏中，因此我们需要适应它们；但也有一些新功能：

+   请求创建一个新的比赛

+   请求加入比赛

我在这里使用“请求”这个词，因为它确实如此。作为一个玩家不会完全处理游戏，它只能向服务器发送请求以采取行动。然后服务器将判断它们并做出反应。现在让我们来看看服务器。

### 服务器

另一方面，服务器只需要启动一次，并需要管理以下功能：

+   存储所有不同的比赛

+   处理每个游戏的步骤

+   向玩家发送游戏更新

+   处理玩家请求

但服务器还必须注意以下事项：

+   管理连接/断开

+   游戏创建

+   将玩家添加为团队的控制者

如你所见，不需要任何类型的显示，因此服务器输出将仅在控制台。它还必须判断来自客户端的所有不同请求。在分布式环境中，对于 Web 开发也是如此，请记住这个规则：*不要信任用户输入*。

如果你记住这一点，它将为你节省很多麻烦和调试时间。一些用户，即使是非常少数的用户，也可能发送随机数据，如作弊或其他你不应该接收的内容。所以不要直接接受输入。

现在功能已经暴露出来，我们需要一种在客户端和服务器之间进行通信的方式。这是我们接下来要讨论的主题。

# 使用套接字进行网络通信

为了能够与其他玩家互动，我们需要一种与他们通信的方法，无论使用的是哪种架构。为了能够与任何计算机通信，我们必须使用套接字。简而言之，套接字通过网络与其他进程/计算机进行通信，只要双方之间存在现有的连接方式（局域网或互联网）。套接字主要有两种类型：非连接（UDP）或连接（TCP）。这两种都需要 IP 地址和端口号才能与目的地通信。

注意，计算机上可用的端口号范围在 0 到 65535 之间。一条建议是避免使用小于 1024 的端口号。原因是大多数端口号都被系统保留或被常用应用程序使用，例如 80 用于网页浏览器，21 用于 FTP 等。你还要确保通信双方使用相同的端口号才能交换数据。现在让我们详细看看之前提到的两种套接字。

## UDP

正如之前所说，**用户数据报协议**（**UDP**）是一种在网络中发送数据而不建立连接的方式。我们可以将这种协议实现的通信可视化，例如发送信件。每次你想向某人发送消息时，你必须指定目标地址（IP 和端口）。然后可以发送消息，但你不知道它是否真的到达了目的地。这种通信非常快，但也有一些限制：

+   你甚至不知道消息是否到达了目的地

+   消息可能会丢失

+   一个大消息将被分割成更小的消息

+   消息的接收顺序可能与原始顺序不同

+   消息可能会重复

由于这些限制，消息不能在收到后立即被利用。需要进行验证。解决这些麻烦的一个简单方法是在您的数据中添加一个包含唯一消息标识符的小型标题。这个标识符将允许我们精确地识别一个消息，删除可能的重复，并按正确的顺序处理每个消息。您还可以确保您的消息不是太大，以避免分割和丢失数据的一部分。

SFML 为我们提供了用于通过 UDP 协议通信的 `sf::UdpSocket` 类。本章将不涉及这种套接字，但如果您对此感兴趣，请查看官方网站上的 SFML 教程（[www.sfml-dev.org](http://www.sfml-dev.org)）。

## TCP

**传输控制协议**（**TCP**）是一个连接协议。这可以比作电话对话。理解这个协议有一些步骤需要遵循：

+   请求连接到地址（电话铃声响起）

+   接受连接（接起电话）

+   交换数据（交谈）

+   停止对话（挂断电话）

由于协议是连接的，它确保到达目的地的数据与源地的顺序、结构和一致性相同。顺便说一句，我们只需要在连接期间指定一次目标地址。此外，如果连接中断（问题在另一边，例如），我们可以在发生时立即检测到。这种协议的缺点是通信速度会降低。

SFML 为我们提供了 `sf::TcpSocket` 类来轻松处理 TCP 协议。这是我们将在我们的项目中使用的类。我将在下一节讨论其用法。

## 选择器

SFML 为我们提供了一个另一个实用工具类：`sf::SocketSelector`。这个类就像任何类型的套接字的观察者，并持有指向管理套接字的指针，如下面的步骤中解释的那样：

1.  使用 `sf::SocketSelector::add(sf::Socket)` 方法将套接字添加到观察列表中。

1.  然后，当观察到的一个或多个套接字接收到数据时，`sf::SocketSelector::wait()函数` 返回。最后，使用 `sf::SocketSelector::isReady(sf::Socket)`，我们可以确定哪个套接字接收到了数据。这使我们能够避免池化并使用实时反应。

我们将在本章中使用这个类与 `sf::TcpSocket` 配对。

## 连接类

现在我们已经介绍了所有基本网络组件，是时候我们考虑我们的游戏了。我们需要决定我们的游戏将如何与其他玩家交换数据。我们需要发送和接收数据。为了实现这一点，我们将使用 `sf::TcpSocket` 类。由于每个套接字上的操作都会阻塞我们游戏的执行，我们需要创建一个系统来禁用阻塞。SFML 提供了 `sf::Socket::setBlocking()` 函数，但我们的解决方案将使用不同的方法。

### 连接类的目标

如果你还记得，在第六章中，*使用多线程提升代码*，我告诉你网络主要是在一个专用线程中管理的。我们的解决方案将遵循这条路径；想法是尽可能透明地管理一个线程，对用户来说。此外，我们将设计 API，使其类似于`sf::Window`类的 SFML 事件管理。这些约束的结果是构建一个`Connection`类。然后，这个类将由我们选择的架构（在下一节中描述）进行特殊化。

现在我们来看看这个新类的头文件：

```cpp
class Connection
{
  public:
  Connection();
  virtual ~Connection();

  void run();
  void stop();
  void wait();

  bool pollEvent(sf::Packet& event);
  bool pollEvent(packet::NetworkEvent*& event);

  void send(sf::Packet& packet);
  void disconnect();
  int id()const;
  virtual sf::IpAddress getRemoteAddress()const = 0;

  protected:
  sf::TcpSocket _sockIn;
  sf::TcpSocket _sockOut;

  private:
  bool _isRunning;

  void _receive();
  sf::Thread _receiveThread;
  sf::Mutex _receiveMutex;
  std::queue<sf::Packet> _incoming;

  void _send();
  sf::Thread _sendThread;
  sf::Mutex _sendMutex;
  std::queue<sf::Packet> _outgoing;

  static int _numberOfCreations;
  const int _id;
};
```

让我们一步一步地解释这个类：

1.  我们首先定义了一个构造函数和一个析构函数。请注意，析构函数被设置为虚拟的，因为该类将被特殊化。

1.  然后我们定义了一些常见的函数来处理内部线程以解决同步问题。

1.  然后定义了一些处理事件的方法。我们构建了两个方法来处理传入的事件，一个方法来处理发出的消息。`pollEvent()`函数的重载使我们能够使用原始数据或解析数据。`packet::NetworkEvent`类将在本章后面进行描述。现在，将其视为类似于`sf::Event`的消息，具有类型和数据，但来自网络。

1.  我们定义了一个函数来正确地关闭通信。

1.  最后，我们定义了一些函数来获取有关连接的信息。

为了能够工作，所有这些函数都需要一些对象。此外，为了尽可能响应，我们将使用两个套接字：一个用于传入消息，另一个用于传出消息。这将允许我们同时发送和接收数据，并加速游戏的响应速度。由于这个选择，我们需要复制所有其他要求（线程、互斥锁、队列等）。让我们讨论每个目标：

+   `sf::TcpSocket`：它处理两端的通信。

+   `sf::Thread`：它允许我们像之前展示的那样非阻塞。它将保持与连接实例的生命周期一致。

+   `sf::Mutex`：它保护数据队列，以避免数据竞争或之后免费使用。

+   `std::queue<sf::Packet>`：这是要处理的事件队列。每次访问它时，都会锁定相关的互斥锁。

现在已经解释了不同的对象，我们可以继续实现类的实现，如下所示：

```cpp
Connection::Connection() :_isRunning(false), _receiveThread(&Connection::_receive,this), _sendThread(&Connection::_send,this),_id(++_numberOfCreations) {}
Connection::~Connection() {}
```

构造函数没有特定的功能。它只是使用正确的值进行初始化，而不启动不同的线程。我们有一个函数来做这件事，如下所示：

```cpp
void Connection::run()
{
  _isRunning = true;
  _receiveThread.launch();
  _sendThread.launch();
}

void Connection::stop() {_isRunning  = false;}

void Connection::wait()
{
  _receiveThread.wait();
  _sendThread.wait();
}
```

这三个函数通过启动、停止或保持它们等待来管理不同线程的生命周期。请注意，不需要互斥锁来保护`_isRunning`，因为我们不会在这些函数之外写入它。

```cpp
int Connection::id()const {return _id;}

bool Connection::pollEvent(sf::Packet& event)
{
  bool res = false;
  sf::Lock guard(_receiveMutex);
  if(_incoming.size() > 0)
  {
    std::swap(event,_incoming.front());
    _incoming.pop();
    res = true;
  }
  return res;
}

bool Connection::pollEvent(packet::NetworkEvent*& event)
{
  bool res = false;
  sf::Packet msg;
  if(Connection::pollEvent(msg))
  {
    event = packet::NetworkEvent::makeFromPacket(msg);
    if(event != nullptr)
      res = true;
  }
  return res;
}
```

这两个函数非常重要，并复制了`sf::Window::pollEvent()`函数的行为，所以它们的用法不会让您感到惊讶。我们在这里做的是，如果有一个启用的事件，就从输入队列中获取一个事件。第二个函数还将接收到的消息解析为`NetworkEvent`函数。通常，我们更倾向于在代码中使用第二种方法，因为所有验证都已经完成，以便能够利用事件。这个函数只是将数据包添加到输出队列。然后，由`_sendThread`对象完成工作，如下面的代码片段所示：

```cpp
void Connection::send(sf::Packet& packet)
{
  sf::Lock guard(_sendMutex);
  _outgoing.emplace(packet);
}
```

这个函数关闭了使用的不同套接字。因为我们使用了连接协议，通信的另一端将能够检测到这一点，并在方便的时候处理。

```cpp
void Connection::disconnect()
{
  _sockIn.disconnect();
  _sockOut.disconnect();
}
```

这个函数是两个最重要的函数之一。它在自己的线程中运行——这就是循环的原因。此外，我们使用`sf::SocketSelector`函数来观察我们的套接字。使用它，我们避免了消耗 CPU 功率的无用操作。相反，我们锁定线程，直到在输入套接字上收到消息。我们还添加了一秒钟的超时，以避免死锁，如下面的代码片段所示：

```cpp
void Connection::_receive()
{
  sf::SocketSelector selector;
  selector.add(_sockIn);
  while(_isRunning)
  {
if(not selector.wait(sf::seconds(1)))
  continue;
if(not selector.isReady(_sockIn))
  continue;
    sf::Packet packet;
    sf::Socket::Status status = _sockIn.receive(packet);
    if(status == sf::Socket::Done)
    {
      sf::Lock guard(_receiveMutex);
      _incoming.emplace(std::move(packet));
    }
    else if (status == sf::Socket::Disconnected)
    {
      packet.clear();
      packet<<packet::Disconnected();
      sf::Lock guard(_receiveMutex);
      _incoming.emplace(std::move(packet));
      stop();
    }
  }
}
```

### 注意

死锁是在多线程程序中遇到的一种情况，其中两个线程无限期地等待，因为它们都在等待只有另一个线程才能释放的资源。最常见的是同一线程中对同一个互斥锁的双重锁定，例如递归调用。在当前情况下，假设您使用了`stop()`函数。线程没有意识到这种变化，仍然会等待数据，可能永远如此，因为套接字上不会收到新的数据。一个简单的解决方案是添加超时，以避免无限期等待，而是等待一小段时间，这样我们就可以重新检查循环条件，并在必要时退出。

一旦收到数据包或检测到断开连接，我们就将相应的数据包添加到队列中。然后用户将能够从自己的线程中检索它，并按自己的意愿处理。断开连接会显示一个特定的`NetworkEvent`：`Disconnected`函数。在后面的章节中，我将详细解释其背后的逻辑。

```cpp
void Connection::_send()
{
  while(_isRunning)
  {
    _sendMutex.lock();
    if(_outgoing.size() > 0)
    {
      sf::Packet packet = _outgoing.front();
      _outgoing.pop();
      _sendMutex.unlock();
      _sockOut.send(packet);
    }
    else
    {
      _sendMutex.unlock();
    }
  }
}
```

这个函数补充了前面的一个函数。它从输出队列中获取事件，并通过其套接字通过网络发送。

如您所见，通过使用类，我们可以在多线程环境中非常容易地发送和接收数据。此外，断开连接就像管理任何其他事件一样，不需要用户进行任何特殊处理。这个类的另一个优点是它非常通用，可以在很多情况下使用，包括客户端和服务器端。

总结一下，我们可以将这个类的使用可视化如下图表：

![连接类的目标](img/8477OS_08_01.jpg)

现在我们已经设计了一个类来管理不同的消息，让我们构建我们的自定义协议。

# 创建通信协议

现在是我们创建自己的自定义协议的时候了。我们将使用 SFML 类 `sf::Packet` 来传输数据，但我们必须定义它们的形状。让我们首先关注 `sf::Packet` 类，然后是形状。

## 使用 sf::Packet 类

`sf::Packet` 类就像一个包含我们数据的缓冲区。它自带了允许我们序列化原始类型的函数。我不知道你是否熟悉计算机的内部内存存储，但请记住，这种排列并非在所有地方都相同。这被称为字节序。你可以把它想象成从右到左或从左到右读取。当你通过网络发送数据时，你不知道目标端的字节序。正因为如此，网络上的数据发送通常采用大端字节序。我建议你查看维基百科页面（[`en.wikipedia.org/wiki/Endianness`](https://en.wikipedia.org/wiki/Endianness)）以获取更多详细信息。

感谢 SFML，有一些预定义的函数让我们的工作变得简单。唯一的麻烦是我们必须使用 SFML 类型而不是原始类型。以下是一个表格，展示了原始类型以及与 `sf::Packet` 一起使用的对应类型：

| 原始类型 | SFML 重载 |
| --- | --- |
| `char` | `sf::Int8` |
| `unsigned char` | `sf::Uint8` |
| `short int` | `sf::Int16` |
| `unsigned short int` | `sf::Uint16` |
| `Int` | `sf::int32` |
| `unsigned int` | `sf::Uint32` |
| `float` | `float` |
| `double` | `double` |
| `char*` | `char*` |
| `std::string` | `std:string` |
| `bool` | `bool` |

`sf::Packet` 类的使用方式类似于标准的 C++ I/O 流，使用 `>>` 和 `<<` 操作符来提取和插入数据。以下是一个直接从 SFML 文档中摘取的 `sf::Packet` 类示例，展示了其在使用上的简单性：

```cpp
void sendDatas(sf::Socket& socket)
{
  sf::Uint32 x = 24;
  std::string s = "hello";
  double d = 5.89;
  // Group the variables to send into a packet
  sf::Packet packet;
  packet << x << s << d;
  // Send it over the network (socket is a valid sf::TcpSocket)
  socket.send(packet);
}

void receiveDatas(sf::Socket& socket)
{
  sf::Packet packet;
  socket.receive(packet);
  // Extract the variables contained in the packet
  sf::Uint32 x;
  std::string s;
  double d;
  if (packet >> x >> s >> d)
  {
    // Data extracted successfully...
  }
}
```

即使这种用法很简单，还有另一种方法可以更轻松地发送结构/类数据，即使用操作符重载。这是我们用来发送/接收数据的技术，以下是一个示例：

```cpp
struct MyStruct
{
  float number;
  sf::Int8 integer;
  std::string str;
};

sf::Packet& operator <<(sf::Packet& packet, const MyStruct& m){
  return packet << m.number << m.integer << m.str;
}

sf::Packet& operator >>(sf::Packet& packet, MyStruct& m){
  return packet >> m.number >> m.integer >> m.str;
}

int main()
{
  MyStruct toSend;
  toSend.number = 18.45f;
  toSend.integer = 42;
  toSend.str = "Hello world!";

  sf::Packet packet;
  packet << toSend;

  // create a socket

  socket.send(packet);
  //...
}
```

使用这种技术，有两个操作符需要重载，然后序列化和反序列化对用户来说是透明的。此外，如果结构发生变化，只需在一个地方更新：操作符。

现在我们已经看到了传输数据的系统，让我们思考一种尽可能通用的构建方式。

## 类似 RPC 的协议

我们现在需要精确地考虑发送数据的需求。我们已经在本章的第一部分通过分离客户端和服务器任务而完成了大部分工作，但这还不够。我们现在需要一个包含所有不同可能性的列表，这些可能性已在此列出。

双方：

+   连接

+   断开连接

+   客户端事件

登出

+   获取游戏列表

+   创建游戏的请求（比赛）

+   加入游戏的请求

+   创建实体的请求

+   销毁实体的请求

服务器事件

+   实体更新

+   实体的事件（onHit，onHitted，onSpawn）

+   更新团队（金牌，游戏结束）

+   响应客户端事件

好消息是事件种类并不多；坏消息是这些事件不需要相同的信息，所以我们不能只构建一个事件，而必须构建与可能动作数量一样多的多个事件，每个事件都有自己的数据。

但现在又出现了另一个问题。我们如何识别使用哪一个？嗯，我们需要一个允许这样做的标识符。一个`enum`函数将完美地完成这项工作，如下所示：

```cpp
namespace FuncIds{
  enum FUNCIDS {
    //both side
    IdHandler = 0, IdDisconnected, IdLogOut,
    //client
    IdGetListGame, IdCreateGame, IdJoinGame,IdRequestCreateEntity, IdRequestDestroyEntity,
    //server events
    IdSetListGame, IdJoinGameConfirmation, IdJoinGameReject, IdDestroyEntity, IdCreateEntity,  IdUpdateEntity, IdOnHittedEntity, IdOnHitEntity,  IdOnSpawnEntity, IdUpdateTeam
  };
}
```

现在我们有了区分动作的方法，我们必须发送一个包含所有这些动作公共部分的包。这个部分（头部）将包含动作的标识符。然后所有动作都将添加它们自己的数据。这正是`sf::Event`与`sf::Event::type`属性一起工作的方式。

我们将复制这个机制到我们自己的系统中，通过构建一个新的类，称为`NetworkEvent`。这个类的工作方式与`sf::Event`类似，但它还增加了与`sf::Packet`类的序列化/反序列化，使我们能够轻松地将数据发送到网络上。现在让我们看看这个新类。

## 网络事件类

`NetworkEvent`类是在`book::packet`命名空间内部构建的。现在我们已经对我们的发送数据的全局形状有了概念，是时候构建一些帮助我们处理它们的类了。

我们将为每个事件构建一个类，它们有一个共同的父类，即`NetworkEvent`类。这个类将允许我们使用多态。以下是其头文件：

```cpp
class NetworkEvent
{
  public:
  NetworkEvent(FuncIds::FUNCIDS type);
  virtual ~NetworkEvent();

  FuncIds::FUNCIDS type()const;
  static NetworkEvent* makeFromPacket(sf::Packet& packet);

  friend sf::Packet& operator>>(sf::Packet&, NetworkEvent& self);
  friend sf::Packet& operator<<(sf::Packet&, const NetworkEvent& self);

  protected:
  const FuncIds::FUNCIDS _type;
};
```

如您所见，这个类非常短，只包含其类型。原因是它是所有不同事件唯一的共同点。它还包含一些默认运算符和一个重要的函数：`makeFromPacket()`。这个函数，正如您将看到的，根据作为参数接收到的`sf::Packet`内部存储的数据构建正确的事件。现在让我们看看实现：

```cpp
NetworkEvent::NetworkEvent(FuncIds::FUNCIDS type) : _type(type){}
NetworkEvent::~NetworkEvent(){}
```

如同往常，构造函数和析构函数非常简单，应该很熟悉：

```cpp
NetworkEvent* NetworkEvent::makeFromPacket(sf::Packet& packet)
{
  sf::Uint8 type;
  NetworkEvent* res = nullptr;
  packet>>type;
  switch(type)
  {
    case FuncIds::IdDisconnected :
    {
      res = new Disconnected();
      packet>>(*static_cast<Disconnected*>(res));
    }break;

    //... test all the different  FuncIds

    case FuncIds::IdUpdateTeam :
    {
      res = new UpdateTeam();
      packet>>(*static_cast<UpdateTeam*>(res));
    }break;
  }
return res;
}
```

前面的函数非常重要。这个函数会将从网络接收到的数据解析为`NetworkEvent`实例，具体取决于接收到的类型。程序员将使用这个实例而不是`sf::Packet`。请注意，在这个函数内部进行了分配，因此在使用后必须对返回的对象进行删除：

```cpp
FuncIds::FUNCIDS NetworkEvent::type()const {return _type;}
```

前一个函数返回与`NetworkEvent`关联的类型。它允许程序员将实例转换为正确的类。

```cpp
sf::Packet& operator>>(sf::Packet& packet, NetworkEvent& self)
{
    return packet;
}

sf::Packet& operator<<(sf::Packet& packet, const NetworkEvent& 
  self)
{
  packet<<sf::Uint8(self._type);
  return packet;
}
```

这两个函数负责序列化/反序列化功能。因为反序列化函数（`>>`运算符）仅在`makeFromPacket()`函数内部调用，并且类型已经被提取，所以这个函数不做任何事情。另一方面，序列化函数（`<<`运算符）将事件的类型添加到包中，因为没有其他数据。

我现在将向您展示一个事件类。所有其他类都是基于相同的逻辑构建的，我相信您已经理解了它是如何实现的。

让我们看看`RequestCreateEntity`类。这个类包含了请求在战场上创建实体的不同数据：

```cpp
namespace EntityType {
  enum TYPES {IdMain = 0,IdEye,IdWormEgg,IdWorm,IdCarnivor,};
}

class RequestCreateEntity : public NetworkEvent
{
  public :
  RequestCreateEntity();
  RequestCreateEntity(short int type,const sf::Vector2i& coord);

  short int getType()const;
  const sf::Vector2i& getCoord()const;

  friend sf::Packet& operator>>(sf::Packet&, RequestCreateEntity& self);
  friend sf::Packet& operator<<(sf::Packet&, const RequestCreateEntity& self);

  private:
  short int _entitytype;
  sf::Vector2i _coord;
};
```

首先，我们定义一个`enum`函数，它将包含所有实体的标识符，然后是请求它们构建的类。`RequestCreateEntity`类继承自之前的`NetworkEvent`类，并定义了相同的函数，以及特定于事件的函数。请注意，这里有两个构造函数。默认构造函数用于`makeFromPacket()`函数，另一个由程序员用来发送事件。现在让我们看看以下实现：

```cpp
RequestCreateEntity::RequestCreateEntity() : NetworkEvent(FuncIds::IdRequestCreateEntity){}

RequestCreateEntity::RequestCreateEntity(short int type,const sf::Vector2i& coord) : NetworkEvent(FuncIds::IdRequestCreateEntity), _entitytype(type), _coord(coord) {}

short int RequestCreateEntity::getType()const
{
    return _entitytype;
}

const sf::Vector2i& RequestCreateEntity::getCoord()const {return _coord;}

sf::Packet& operator>>(sf::Packet& packet, RequestCreateEntity& self)
{
  sf::Int8 type;
  sf::Int32 x,y;
  packet>>type>>x>>y;

  self._entitytype = type;
  self._coord.x = x;
  self._coord.y = y;
  return packet;
}
```

此函数解包特定于事件的不同的数据，并将其存储在内部。就是这样：

```cpp
sf::Packet& operator<<(sf::Packet& packet, const RequestCreateEntity& self)
{
  packet<<sf::Uint8(self._type)
  <<sf::Int8(self._entitytype)
  <<sf::Int32(self._coord.x)
  <<sf::Int32(self._coord.y);
  return packet;
}
```

此函数使用与用于原始类型的 SFML 对象序列化不同的数据。

如您所见，使用这个系统创建事件真的很简单。它只需要为其类提供一个标识符以及一些解析函数。所有其他事件都是基于这个模型构建的，所以我就不再解释它们了。如果您想查看完整的代码，可以查看`include/SFML-Book/common/Packet.hpp`文件。

现在我们已经拥有了构建多人游戏部分所需的所有键，是时候修改我们的游戏了。

# 修改我们的游戏

为了将此功能添加到我们的游戏中，我们需要稍微重新思考一下内部结构。首先，我们需要将我们的代码分成两个不同的程序。所有通用类（例如用于通信的类）将放入一个通用目录中。所有其他功能将根据其用途放入服务器或客户端文件夹中。让我们从最复杂的部分开始：服务器。

## 服务器

服务器将负责所有模拟。实际上，我们的整个游戏都将驻留在服务器上。此外，它还必须确保能够同时运行多个比赛。它还必须处理连接/断开连接和玩家事件。

因为服务器不会渲染任何内容，所以我们在这边不再需要任何图形类。因此，`CompSkin`组件中的`AnimatedSprite`函数以及`CompHp`函数中的`sf::RectangleShape`组件都需要被移除。

因为实体的位置是由`CompSkin`组件（更确切地说，是`_sprite`）存储的，所以我们必须在每个实体中添加一个`sf::Vector2f`函数来存储其位置。

主循环也将有很大的变化。记住，我们需要管理多个客户端和比赛，并在特定端口上监听新的连接。因此，为了能够做到这一点，我们将构建一个`Server`类，每个比赛将有一个自己的游戏实例在其自己的线程中运行。所以让我们这样做：

### 构建服务器入口点

服务器类将负责管理新客户端，创建新比赛并将客户端添加到现有比赛中。这个类可以看作是游戏的主菜单。顺便说一下，玩家屏幕上的相应显示如下：

![构建服务器入口点](img/8477OS_08_04.jpg)

因此，我们需要做的是：

+   存储正在进行的比赛（游戏）

+   存储新客户端

+   监听新客户端

+   响应一些请求（创建新比赛、加入比赛、获取正在进行的比赛列表）

现在我们来构建服务器类。

```cpp
class Server
{
    public:
        Server(int port);
        ~Server();
        void run(); 
    private:
        const unsigned int _port;
        void runGame();
        void listen();

        sf::Thread _gameThread;
        sf::Mutex _gameMutex;
        std::vector<std::shared_ptr<Game>> _games;

        sf::Mutex _clientMutex;
        std::vector<std::shared_ptr<Client>> _clients;

        sf::Thread _listenThread;
        sf::TcpListener _socketListener;
        std::shared_ptr<Client> _currentClient;
};
```

这个类处理上述所有信息，以及一些线程来独立运行不同的功能（日志和请求）。现在让我们看看它的实现：

首先，我们需要声明一些全局变量和函数，如下所示：

```cpp
sig_atomic_t stop = false;
void signalHandler(int sig) {stop = true;}
```

当用户按下*Ctrl* + *C*键请求停止服务器时，将调用之前的函数。这个机制在`Server::run()`函数中初始化，你很快就会看到。

```cpp
Server::Server(int port) : 
  _port(port),_gameThread(&Server::runGame,this),_listenThread(&Server::listen,this)
{
  rand_init();
  _currentClient = nullptr;
}
```

之前的函数初始化了不同的线程和随机函数。

```cpp
Server::~Server()
{
  _gameMutex.lock();
  for(Game* game : _games)
  game->stop()
  _gameMutex.unlock();
  _clientMutex.lock();
  for(Client* client : _clients)
  client->stop();
  _clientMutex.unlock();
}
```

在这里，我们销毁所有正在进行的比赛和客户端，以正确地停止服务器。

```cpp
void Server::run()
{
  std::signal(SIGINT,signalHandler);
  _gameThread.launch();
  _listenThread.launch();
  _gameThread.wait();
  _listenThread.terminate();
}
```

这个函数启动服务器，直到接收到`SIGINT`（*Ctrl* + *c*）信号：

```cpp
void Server::runGame()
{
  while(!stop)
  {
    sf::Lock guard(_clientMutex);
    for(auto it = _clients.begin(); it != _clients.end();++it)//loop on clients
    {
      std::shared_ptr<Client> client = *it; //get iteration current client
      packet::NetworkEvent* msg;
      while(client and client->pollEvent(msg)) //some events incomings
      {
        switch(msg->type()) //check the type
        {
          case FuncIds::IdGetListGame :
          {
            sf::Packet response;
            packet::SetListGame list;
            sf::Lock guard(_gameMutex);
            for(Game* game : _games) { //send match informations
            list.add(game->id(),game->getPlayersCount(),game->getTeamCount());
          }
          response<<list;
          client->send(response);
        }break;
        case FuncIds::IdCreateGame :
        {
          sf::Packet response;
          packet::SetListGame list;
          sf::Lock guard(_gameMutex);
          _games.emplace_back(new Game("./media/map.json")); //create a new match
          for(Game* game : _games){ //send match informations
          list.add(game->id(),game->getPlayersCount(),game->getTeamCount());
        }
        //callback when a client exit a match
        _games.back()->onLogOut = this{
          _clients.emplace_back(client);
        };
        _games.back()->run(); //start the match
        response<<list;
        for(auto it2 = _clients.begin(); it2 != _clients.end();++it2){ //send to all client
        (*it2)->send(response);
      }
    }break;
    case FuncIds::IdJoinGame :
    {
      int gameId = static_cast<packet::JoinGame*>(msg)->gameId()
      sf::Lock guard(_gameMutex); 
      //check if the player can really join the match
      for(auto game : _games) {
        if(game->id() == gameId) {
          if(game->addClient(client)){ //yes he can
          client = nullptr;
          it = _clients.erase(it); //stop to manage the client here. Now the game do it
          --it;
        }
        break;
      }
    }
  }break;
  case FuncIds::IdDisconnected : //Oups, the client leave the game
  {
    it = _clients.erase(it);
    --it;
    client = nullptr;
  }break;
  default : break;
}
delete msg;
}
```

这个函数是服务器最重要的函数。这是处理来自玩家的所有事件的函数。对于每个客户端，我们检查是否有等待处理的事件，然后根据其类型，采取不同的行动。多亏了我们的`NetworkEvent`类，对事件的解析变得简单，我们可以将代码缩减到仅包含功能的部分：

```cpp
void Server::listen()
{
  if(_socketListener.listen(_port) != sf::Socket::Done) {
    stop = true;
    return;
  }
  _currentClient =   new Client;
  while(!stop)
  {
    if (_socketListener.accept(_currentClient->getSockIn()) == sf::Socket::Done) {
      if(_currentClient->connect()) {
        sf::Lock guard(_clientMutex);
        _clients.emplace_back(_currentClient);
        _currentClient->run();
        _currentClient = new Client;
      }
      else {
        _currentClient->disconnect();
      }
    }
  }
}
```

这个函数是服务器的最终函数。它的任务是等待新的连接，初始化客户端，并将其添加到先前函数管理的列表中。

在这个类中不需要做其他任何事情，因为一旦客户端加入比赛，处理它的将不再是`Server`类，而是每个比赛都由一个`Game`实例管理。现在让我们来看看它。

### 在比赛中对玩家动作做出反应

`Game`类没有太大变化。事件处理已经改变，但仍然非常类似于原始系统。我们不再使用`sf::Event`，而是现在使用`NetworkEvent`。由于 API 非常接近，它不应该给你带来太多麻烦。

与玩家交互的第一个函数是接收比赛信息的那个。例如，我们需要将其发送到地图文件和所有不同的实体。这个任务是由`Game::addClient()`函数创建的，如下所示：

```cpp
bool Game::addClient(Client* client)
{
    sf::Lock guard(_teamMutex);
    Team* clientTeam = nullptr;
    for(Team* team : _teams)
    {
        // is there any team for the player
        if(team->getClients().size() == 0 and team->isGameOver())
        { //find it
            clientTeam = team;
            break;
        }
    }

    sf::Packet response;
    if(clientTeam != nullptr)
    {
        //send map informations
        std::ifstream file(_mapFileName);
        //get file content to as std::string
        std::string content((std::istreambuf_iterator<char>(file)),(std::istreambuf_iterator<char>()));

        packet::JoinGameConfirmation conf(content,clientTeam->id());//send confirmation

        for(Team* team : _teams)
        { //send team datas
            packet::JoinGameConfirmation::Data data;
            data.team = team->id();
            data.gold = team->getGold();
            data.color = team->getColor();
            conf.addTeam(std::move(data));
        }

        response<<conf;
        client->send(response);
        {
            //send initial content
            response.clear();
            sf::Lock gameGuard(_gameMutex);
            packet::CreateEntity datas; //entites informations
            for(auto id : entities)
                addCreate(datas,id);
            response<<datas;
            client->send(response);
        }

        client->setTeam(clientTeam);
        sf::Lock guardClients(_clientsMutex);
        _clients.emplace_back(client);
    }
    else
    { //Oups, someone the match is already full
        response<<packet::JoinGameReject(_id);
        client->send(response);
    }
    return clientTeam != nullptr;
}
```

这个函数分为四个部分：

1.  检查是否可以添加新玩家到比赛中。

1.  发送地图数据。

1.  发送实体信息。

1.  将客户端添加到团队中。

1.  一旦客户端被添加到游戏中，我们必须管理其接收的事件。这个任务由新的函数`processNetworkEvents()`完成。它的工作方式与旧的`processEvents()`函数完全相同，但使用`NetworkEvent`而不是`sf::Events`：

    ```cpp
    void Game::processNetworkEvents()
    {
        sf::Lock guard(_clientsMutex);
         for(auto it = _clients.begin(); it != _clients.end();++it)
        {
             auto client = *it;
             packet::NetworkEvent* msg;
              while(client and client->pollEvent(msg))
             {
                  switch(msg->type())
                  {
                       case FuncIds::IdDisconnected :
                       {
                           it = _clients.erase(it);
                           --it;
                           delete client;
                           client = nullptr;
                       }break;

                       case FuncIds::IdLogOut :
                       {
                           it = _clients.erase(it);
                           --it;
                           client->getTeam()->remove(client);
                           onLogOut(client); //callback to the server
                            client = nullptr;
                       }break;

                       case FuncIds::IdRequestCreateEntity :
                       {
                           packet::RequestCreateEntity* event = static_cast<packet::RequestCreateEntity*>(msg);
                           sf::Lock gameGuard(_teamMutex);
                            // create the entity is the team as enough money
                       }break;

                       case FuncIds::IdRequestDestroyEntity :
                       {
                           packet::RequestDestroyEntity* event = static_cast<packet::RequestDestroyEntity*>(msg);
                           // destroy the entity if it shares the same team as the client
                       }break;
                        default : break;
                   } //end switch           } //end while       } //end for 
    }
    ```

这并不令人惊讶。我们必须处理客户端断开连接/登出，以及所有不同的事件。我无需放置不同事件的全部代码，因为那里没有复杂的东西。但如果你感兴趣，可以查看`src/SFML-Book/server/Game.cpp`文件。

注意，我们从未向客户端发送任何请求的确认。游戏的同步将确保这一点。

### 客户端与服务器之间的同步

`Game`类中的一个重大变化是管理客户端与服务器之间同步的方式。在前一章中，只有一个客户端接收数据。现在我们有一些客户端，逻辑发生了变化。为了确保同步，我们必须向客户端发送更新。

为了能够发送更新，我们必须在游戏循环中记住每个变化，然后将它们发送给所有玩家。因为请求将改变游戏，它将包含在更新中。这就是为什么在前面的点中我们不向玩家发送任何请求的响应。在游戏中，我们需要跟踪以下内容：

+   实体创建

+   实体销毁

+   实体更新

+   实体事件（onHitted, onHit, onSpawn）

+   更新团队状态、金币数量等

这些事件中的大多数只需要实体 ID，而不需要其他信息（销毁实体事件）。对于其他事件，需要一些额外的数据，但逻辑仍然是相同的：将信息添加到容器中。

然后，在`Game::update()`函数中，我们必须向所有玩家发送更新。为此，我们将输出事件添加到队列中（与`Connection`类中的方式相同）。另一个线程将负责它们的传播。

这里是一个创建销毁事件的代码片段：

```cpp
if(_destroyEntityId.size() > 0)
{
  packet::DestroyEntity update;
  for(auto id : _destroyEntityId)
  update.add(id);
  sf::Packet packet;
  packet<<update;
  sendToAll(packet);
  _destroyEntityId.clear();
}
```

如你所见，这里没有复杂性，所有的魔法都是由`sendToAll()`函数完成的。正如你所猜测的，它的目的是通过将数据包添加到输出队列来向所有不同的玩家广播消息。然后另一个线程将进入该队列以广播消息。

在游戏逻辑方面，没有其他变化。我们仍然使用实体系统和地图来管理关卡。只是图形元素被删除了。这是客户端的任务，向玩家显示游戏状态，说到这里，让我们现在详细看看这一部分。

## 客户端类

这是本章的最后一部分。客户端比服务器简单得多，因为它只需要管理一个玩家，但仍然有点复杂。客户端将具有图形渲染，但没有更多的游戏逻辑。客户端的唯一任务是处理玩家输入和更新游戏状态，使用接收到的网络事件。

由于现在仅启动客户端不足以开始比赛，我们必须与服务器通信以初始化游戏，甚至创建一个新的比赛。实际上，客户端由两个主要组件组成：连接菜单和游戏。客户端游戏类为了处理新的功能而发生了很大变化，这就是为什么我现在在继续解释之前会先展示新的`Game`头文件：

```cpp
class Game
{
  public:
  Game(int x=1600, int y=900);
  ~Game();
  bool connect(const sf::IpAddress& ip, unsigned short port,sf::Time timeout=sf::Time::Zero);
  void run(int frame_per_seconds=60);
  private:
  void processEvents();
  void processNetworkEvents();
  void update(sf::Time deltaTime);
  void render();
  bool _asFocus; 
  sf::RenderWindow _window;
  sf::Sprite _cursor;
  Client _client;
  bool _isConnected;
  enum Status {StatusMainMenu,StatusInGame, StatusDisconnected} _status;
  MainMenu _mainMenu;
  GameMenu _gameMenu;
  Level* _level;
  Level::FuncType _onPickup;
  int _team;
};
```

如你所见，有一些新的函数用于管理网络，GUI 已经被分离到其他类中（`MainMenu`，`GameMenu`）。另一方面，一些类如`Level`并没有改变。

现在让我们看看主菜单。

### 服务器连接

在开始比赛之前，需要连接到服务器，连接成功后，我们必须选择我们想要玩哪场比赛。连接方式与服务器上完全相同，但顺序相反（将接收改为发送，反之亦然）。

然后由玩家选择比赛。他必须能够创建一个新的比赛并加入其中。为了简化这个过程，我们将通过创建一个`MainMenu`类来使用我们的 GUI：

```cpp
class MainMenu : public sfutils::Frame
{
  public:
  MainMenu(sf::RenderWindow& window,Client& client);
  void fill(packet::SetListGame& list);
  private:
  Client& _client;
};
```

这个类非常小。它是一个带有几个按钮的框架，正如你在以下图片中可以看到的：

![服务器连接](img/8477OS_08_04.jpg)

这个类的实现并不太复杂；而是具有更大的影响：

```cpp
MainMenu::MainMenu(sf::RenderWindow& window,Client& client) : sfutils::Frame(window,Configuration::guiInputs), _client(client)
{
        setLayout(new sfutils::Vlayout);
}

void MainMenu::fill(packet::SetListGame& list)
{
    clear();
    sfutils::VLayout* layout = static_cast<sfutils::VLayout*>(Frame::getLayout());
    {
        sfutils::TextButton* button = new sfutils::TextButton("Create game");
        button->setCharacterSize(20);
        button->setOutlineThickness(1);
        button->setFillColor(sf::Color(48,80,197));
        button->on_click = this{
            sf::Packet event;
            event<<packet::CreateGame();
            _client.send(event);
        };
        layout->add(button);
    }

    {
        sfutils::TextButton* button = new sfutils::TextButton("Refresh");
        button->setCharacterSize(20);
        button->setOutlineThickness(1);
        button->setFillColor(sf::Color(0,88,17));
        button->on_click = this{
            sf::Packet event;
            event<<packet::GetListGame();
            _client.send(event);
        };
        layout->add(button);
    }

    for(const autoe& game : list.list())
    {
        std::stringstream ss;
        ss<<"Game ["<<game.id<<"] Players: "<<game.nbPlayers<<"/"<<game.nbTeams;
        sfutils::TextButton* button = new sfutils::TextButton(ss.str());
        button->setCharacterSize(20);
        button->setOutlineThickness(1);
        button->on_click = this,game{
            sf::Packet event;
            event<<packet::JoinGame(game.id);
            _client.send(event);
        };
        layout->add(button);
    } //end for
}
```

类的所有逻辑都编码在`fill()`函数中。这个函数接收服务器上正在运行的比赛列表，并将其显示为按钮给玩家。然后玩家可以按下其中一个按钮加入比赛或请求创建游戏。

当玩家请求加入游戏时，如果服务器端一切正常，客户端将接收到一个`JoinGameConfirmation`事件，其中包含初始化其级别的数据（记住服务器中的`addClient()`函数）：

```cpp
void Game::processNetworkEvents()
{
  packet::NetworkEvent* msg;
  while(_client.pollEvent(msg))
  {
    if(msg->type() == FuncIds::IdDisconnected) {
      _isConnected = false;
      _status = StatusDisconnected;
    }
    else
    {
      switch(_status)
      {
        case StatusMainMenu:
        {
          switch(msg->type())
          {
            case FuncIds::IdSetListGame :
            {
              packet::SetListGame* event = static_cast<packet::SetListGame*>(msg);
              _mainMenu.fill(*event);
            }break;
            case FuncIds::IdJoinGameConfirmation :
            {
              packet::JoinGameConfirmation* event = static_cast<packet::JoinGameConfirmation*>(msg);
              // create the level from event
              if(_level != nullptr) {
                _team = event->getTeamId();
                // initialize the team menu
                _status = StatusInGame;
              }
            }break;
            case FuncIds::IdJoinGameReject :
            {
              //...
            }break;
            default : break;
          }
        }break;
        case StatusInGame :
        {
          _gameMenu.processNetworkEvent(msg);
          _level->processNetworkEvent(msg);
        }break;
        case StatusDisconnected :
        {
          // ...
                }break;
            } //end switch
        } //end else
        delete msg;
    } //end while
}
```

这个函数处理来自服务器的各种事件，并根据内部状态进行分发。正如你所见，一个`JoinGameConfirmation`事件会启动级别的创建，并改变内部状态，这通过向玩家显示游戏来体现。

### 级别类

对`Level`类进行了一些添加，以处理网络事件。我们仍然需要处理构建/销毁请求，但现在我们还需要管理来自服务器的各种事件，例如位置更新、实体创建/销毁和实体事件。

这种管理非常重要，因为这是添加游戏动态并使其与服务器同步的地方。请看以下函数：

```cpp
void Level::processNetworkEvent(packet::NetworkEvent* msg)
{
  switch(msg->type())
  {
    case FuncIds::IdDestroyEntity :
    {//need to destroy an entity
      packet::DestroyEntity* event = static_cast<packet::DestroyEntity*>(msg);
         for(auto id : event->getDestroy())
           {
                destroyEntity(id);
            }
    }break;
    case FuncIds::IdCreateEntity :
    {//need to create an entity
      packet::CreateEntity* event = static_cast<packet::CreateEntity*>(msg);
      for(const autoa& data : event->getCreates())
      {
        Entity& e = createEntity(data.entityId,data.coord); //create the entity
        makeAs(data.entityType,e,&_teamInfo.at(data.entityTeam),*this,data); //add the components
      }
    }break;
    case FuncIds::IdUpdateEntity :
    {//an entity has changed
      packet::UpdateEntity* event = static_cast<packet::UpdateEntity*>(msg);
      for(const auto& data : event->getUpdates())
      {
        if(entities.isValid(data.entityId)) //the entity is still here, so we have to update it
        {
          CompSkin::Handle skin = entities.getComponent<CompSkin>(data.entityId);
          CompHp::Handle hp = entities.getComponent<CompHp>(data.entityId);
          //... and other updates
          hp->_hp = data.hp;
        }
      }
    }break;
    case FuncIds::IdOnHittedEntity :
    {//entity event to launch
      packet::OnHittedEntity* event = static_cast<packet::OnHittedEntity*>(msg);
      for(const auto& data : event->getHitted())
      {
        if(entities.isValid(data.entityId))
        {
          Entity& e = entities.get(data.entityId);
          if(e.onHitted and entities.isValid(data.enemyId)) //to avoid invalid datas
          {
            Entity& enemy = entities.get(data.enemyId);
            //call the callback
            e.onHitted(e,_map->mapPixelToCoords(e.getPosition()), enemy, _map->mapPixelToCoords(enemy.getPosition()),*this);
          }
        }
      }
    }break;
    case FuncIds::IdOnHitEntity :
    {//another event
      //same has previous with e.onHit callback
    }break;
    case FuncIds::IdOnSpawnEntity :
    { //other event
      packet::OnSpawnEntity* event = static_cast<packet::OnSpawnEntity*>(msg);
      for(auto id : event->getSpawn())
      {
        if(entities.isValid(id))
        {
          Entity& e = entities.get(id);
          CompAISpawner::Handle spawn = entities.getComponent<CompAISpawner>(id);
          if(spawn.isValid() and spawn->_onSpawn) //check data validity
          {//ok, call the call back
            spawn->_onSpawn(*this,_map->mapPixelToCoords(e.getPosition()));
          }
        }
      }
    }break;
    default : break;
  }
}
```

如您所见，这个函数有点长。这是因为我们必须管理六种不同类型的事件。实体的销毁和创建很容易实现，因为大部分工作都是由`EntityManager`函数完成的。更新也很简单。我们必须逐个更改每个值，或者激活实体事件的回调，并进行所有必要的验证；记住*不要相信用户输入*，即使它们来自服务器。

现在游戏的主要部分已经完成，我们只需要从客户端清理掉所有不必要的组件，只保留`CompTeam`、`CompHp`和`CompSkin`。其他所有组件都只由服务器用于实体的行为。

本章的最终结果与上一章不会有太大变化，但现在你将能够和朋友一起玩游戏，因为难度现在是真实的：

![The Level class](img/8477OS_08_05.jpg)

# 将数据持久化添加到游戏中

如果你像我一样，无法想象一个没有保存选项的游戏，这部分将对你更有吸引力。在这本书的最后一部分，我将向你介绍数据的持久化。数据持久化是程序保存其内部状态以供将来恢复的能力。这正是游戏中保存选项所做的事情。在我们的特定情况下，因为客户端直接从服务器接收数据，所有的工作都必须在服务器部分完成。首先，让我们稍微思考一下我们需要保存什么：

+   实体及其组件

+   团队

+   游戏

接下来我们需要一种方法来存储这些数据，以便以后能够恢复。解决方案是使用文件或其他可以随时间增长且易于复制的东西。为此功能，我选择了使用`Sqlite`。这是一个作为库提供的数据库引擎。更多信息可以在[`sqlite.org/`](https://sqlite.org/)网站上找到。

使用数据库引擎对我们的项目来说有点过度，但在这里的目标是向你展示它在我们的实际游戏中的应用。然后你将能够将其用于你创建的更复杂的项目。持久化数据将存储在一个数据库文件中，这个文件可以很容易地使用一些 GUI 工具来复制或修改`Sqlite`。

这个解决方案的唯一缺点是需要一些关于 SQL 语言的知识。因为这本书的目标不是涵盖这个主题，我建议你使用另一种用法：**对象关系映射**（**ORM**）。

## 什么是 ORM？

简单地说，ORM 位于数据库引擎和程序 API 之间，并在需要时自动生成 SQL 查询，而不需要手动编写。此外，大多数 ORM 支持多种数据库引擎，允许你通过一行或两行代码更改引擎。

以下是一个示例，将有助于说明我的话（伪代码）。首先，使用标准库：

```cpp
String sql = "SELECT * from Entity WHERE id = 10"
SqlQuery query(sql);
SqlResults res = query.execute();
Entity e;
e.color = res["color"];
//.. other initializations
```

现在使用 ORM：

```cpp
Entity e = Entity::get(10);
// color is already load and set
```

如你所见，所有这些都是由 ORM 完成的，无需编写任何代码。保存数据时也是如此。只需使用 `save()` 方法，就是这样。

## 使用 cpp-ORM

我们将使用我编写的 `cpp-ORM` 库，所以在我们的项目中使用它不会有任何问题。它可以在 [`github.com/Krozark/cpp-ORM`](https://github.com/Krozark/cpp-ORM) 找到。

为了能够工作，库需要一些关于你类的信息；这就是为什么必须使用一些自定义类型来保存你想要保存的数据。

| ORM 类型 | C++ 类型 |
| --- | --- |
| orm::BooleanField | bool |
| orm::CharField<N> | std::string (of length N) |
| orm::DateTimeField | struct tm |
| orm::AutoDateTimeField |
| orm::AutoNowDateTimeField |
| orm::IntegerField | int |
| orm::FloatField | float |
| orm::DoubleField | double |
| orm::TextField | std::string |
| orm::UnsignedIntegerField | unsigned int |
| orm::FK<T,NULLABLE=true> | std::shared_ptr<T> NULLABLE 指定 T 是否可以为空 |
| orm::ManyToMany<T,U> | std::vector<std::shared_ptr<U>> 当 T 需要保留对 U 类的未知数量的引用时使用 |

此外，你的类将需要一个不带参数的默认构造函数，并扩展自 `orm::SqlObject<T>`，其中 `T` 是你的类名。为了更好地理解，让我们构建一个持久化的组件，例如 `CompHp`：

```cpp
class CompHp : public sfutils::Component<CompHp,Entity>, public orm::SqlObject<CompHp>
{
  public:
  CompHp(); //default constructor
  explicit CompHp(int hp);
  orm::IntegerField _hp; //change the type to be persistent
  orm::IntegerField _maxHp; //here again
  //create column for the query ability (same name as your attributes)
  MAKE_STATIC_COLUMN(_hp,_maxHp); 
};
```

没有多少需要解释的。我们只需将 `orm::SqlObject<CompHp>` 作为父类添加，并将 `int` 改为 `orm::IntegerField`。`MAKE_STATIC_COLUMN` 用于创建一些额外的字段，这些字段将包含数据库中每个字段的列名。关于实现，还有一个宏来避免重复工作：`REGISTER_AND_CONSTRUCT`。其用法如下：

```cpp
REGISTER_AND_CONSTRUCT(CompHp,"CompHp",\
_hp,"hp",\
_maxHp,"maxHp")
```

这个宏将构建整个默认构造函数实现。然后，在你的代码中，像往常一样使用字段。不需要更改任何关于你类的内容。

最后的要求是引用要使用的默认数据库。在我们的例子中，我们将使用 `Sqlite3` 引擎，因此我们需要在某个地方创建它，例如在 `main.cpp` 文件中：

```cpp
#include <ORM/backends/Sqlite3.hpp>
orm::Sqlite3DB def("./08_dataPersistence.sqlite"); //create the database (need to be include before file that use SqlObject)
orm::DB& orm::DB::Default = def;//set the default connection (multi connection is possible)
#include <ORM/core/Tables.hpp>
#include <SFML-Book/server/Server.hpp>
int main(int argc, char* argv[])
{
  // get port parameter
  orm::DB::Default.connect(); //connect to the database
  orm::Tables::create(); //create all the tables if needed
  book::Server server(port);
  server.run();
  orm::DB::Default.disconnect(); //disconnect the database
  return 0;
}
```

在这个简短的例子中，数据库已创建，连接已连接到它。重要的是要记住，默认情况下，所有对数据库的访问都将使用默认连接。

## 将我们的对象持久化

现在数据库已经创建，我们不再需要触碰它了。现在让我们关注如何将我们的对象保存到数据库中或恢复它们。

### 在数据库中保存对象

多亏了实体系统，这个功能非常简单。让我们以我们之前的 `CompHp` 类为例。创建其实例，并在其上调用 `.save()` 方法。如果你想要更新数据库中已经存储的对象，也可以使用 `save()`。只有更改的字段将被更新：

```cpp
CompHp chp;
chp._hp = 42;
chp.save();
//oups I've forgotten the other field
chp._maxHp = 42;
chp.save();
std::cout<<"My id is now "<<chp.getPk()<<std::endl;
```

现在让我们继续到对象加载。

### 从数据库中加载数据对象

加载对象基本上有两种方式。第一种是你知道它的主键（标识符），第二种是搜索符合特定标准的所有对象：

```cpp
CompHp::type_ptr chp = CompHp::get(10); //load from database
//chp.getPk() = -1 on error, but chp is a valid object so you can use it
std::cout<<"My id is "<<chp->getPk()<<" And my content is "<<*chp<<std::endl;
```

这两行代码从数据库中加载一个对象，并将其内容显示到控制台输出。另一方面，如果你不知道标识符值，但有一个特定的标准，你也可以以下这种方式加载对象：

```cpp
CompHp::result_type res;
CompHp::query()
.filter(
  orm::Q<CompHp>(25,orm::op::gt,CompHp::$_hp)
  and orm::Q<CompHp>(228,orm::op::lte,CompHp::$_maxHp)
  or (orm::Q<CompHp>(12,orm::op::gt,CompHp::$_hp) and orm::Q<CompHp>(25,orm::op::exact,CompHp::$_maxHp))
)// (_hp > 25) and (_maxHp <= 228) or (_hp > 12 and _maxHp ==25 )
. orderBy(CompHp::$_hp,'+')// could be +,-,?
.limit(12) //only the first 12 objects
.get(res);
for(auto chp : res)
std::cout<<"My id is "<<chp->getPk()<<" And my content is "<<*chp<<std::endl;
```

在这个例子中，我们通过一个复杂的查询获取整个`CompHp`组件，并将其内容显示到控制台输出。

现在你已经掌握了所有必要的键，可以在我们的实际游戏中轻松地添加加载/保存功能，所以我就不会进一步深入到实现细节中。

# 摘要

在最后一章中，你学习了如何使用套接字、选择器和甚至创建自定义协议来添加基本网络功能。你已经将新知识整合到之前的游戏中，并将其转变为实时多人游戏。

你还学会了如何使用 ORM 给你的数据添加持久性，以及如何为游戏添加保存/加载选项。到目前为止，你已经看到了游戏编程的许多方面，现在你手头有所有必要的键，可以构建你想要的任何类型的 2D 游戏。

我希望这本书能给你提供有用的工具。如果你想重用这本书中制作的框架的某些部分，代码可在 GitHub 上找到：[`github.com/Krozark/SFML-utils`](https://github.com/Krozark/SFML-utils)。

我希望你喜欢阅读这本书，并且游戏开发得很好。祝你未来游戏好运！
