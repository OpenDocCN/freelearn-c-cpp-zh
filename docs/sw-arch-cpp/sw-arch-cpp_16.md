# 第十二章：面向服务的架构

分布式系统的一个非常常见的架构是**面向服务的架构**（**SOA**）。这不是一个新的发明，因为这种架构风格几乎和计算机网络一样古老。SOA 有许多方面，从**企业服务总线**（**ESB**）到云原生微服务。

如果您的应用程序包括 Web、移动或**物联网**（**IoT**）接口，本章将帮助您了解如何以模块化和可维护性为重点构建它们。由于大多数当前系统以客户端-服务器（或其他网络拓扑）方式工作，学习 SOA 原则将帮助您设计和改进这样的系统。

本章将涵盖以下主题：

+   理解 SOA

+   采用消息传递原则

+   使用 Web 服务

+   利用托管服务和云提供商

# 技术要求

本章中提出的大多数示例不需要任何特定的软件。对于 AWS API 示例，您将需要**AWS SDK for C++**，可以在[`aws.amazon.com/sdk-for-cpp/`](https://aws.amazon.com/sdk-for-cpp/)找到。

本章中的代码已放在 GitHub 上，网址为[`github.com/PacktPublishing/Software-Architecture-with-Cpp/tree/master/Chapter12`](https://github.com/PacktPublishing/Software-Architecture-with-Cpp/tree/master/Chapter12)。

# 理解面向服务的架构

面向服务的架构是一个特征松散耦合的组件提供服务给彼此的软件设计的例子。这些组件使用共享的通信协议，通常是通过网络。在这种设计中，服务意味着可以在原始组件之外访问的功能单元。一个组件的例子可能是一个提供地理坐标响应区域地图的映射服务。

根据定义，服务具有四个属性：

+   它是业务活动的一种表现，具有明确定义的结果。

+   它是自包含的。

+   它对用户是不透明的。

+   它可能由其他服务组成。

## 实施方法

面向服务的架构并不规定如何处理服务定位。这是一个可以应用于许多不同实现的术语。关于一些方法是否应该被视为面向服务的架构存在讨论。我们不想参与这些讨论，只是强调一些经常被提及为 SOA 方法的方法。

让我们比较一些。

### 企业服务总线

当有人提到面向服务的架构时，ESB 往往是第一个联想到的。这是实现 SOA 的最古老方法之一。

ESB 从计算机硬件架构中得到类比。硬件架构使用计算机总线，如 PCI，以实现模块化。这样，第三方提供商可以独立于主板制造商实现模块（如图形卡、声卡或 I/O 接口），只要每个人都遵守总线所需的标准。

与 PCI 类似，ESB 架构旨在构建一种标准的通用方式，以允许松散耦合服务的交互。这些服务预计将独立开发和部署。还应该可以组合异构服务。

与 SOA 本身一样，ESB 没有由任何全局标准定义。要实现 ESB，需要在系统中建立一个额外的组件。这个组件就是总线本身。ESB 上的通信是事件驱动的，通常是通过消息导向中间件和消息队列实现的，我们将在后面的章节中讨论。

企业服务总线组件扮演以下角色：

+   控制服务的部署和版本控制

+   维护服务冗余

+   在服务之间路由消息

+   监控和控制消息交换

+   解决组件之间的争执

+   提供常见服务，如事件处理、加密或消息队列

+   强制服务质量（**QOS**）

既有专有商业产品，也有实现企业服务总线功能的开源产品。一些最受欢迎的开源产品如下：

+   Apache Camel

+   Apache ServiceMix

+   Apache Synapse

+   JBoss ESB

+   OpenESB

+   Red Hat Fuse（基于 Apache Camel）

+   Spring 集成

最受欢迎的商业产品如下：

+   IBM 集成总线（取代 IBM WebSphere ESB）

+   Microsoft Azure 服务总线

+   Microsoft BizTalk Server

+   Oracle 企业服务总线

+   SAP 过程集成

与本书中介绍的所有模式和产品一样，您在决定采用特定架构之前，必须考虑其优势和劣势。引入企业服务总线的一些好处如下：

+   更好的服务可扩展性

+   分布式工作负载

+   可以专注于配置而不是在服务中实现自定义集成

+   设计松散耦合服务的更简单方法

+   服务是可替换的

+   内置冗余能力

另一方面，缺点主要围绕以下方面：

+   单点故障-ESB 组件的故障意味着整个系统的故障。

+   配置更复杂，影响维护。

+   消息队列、消息转换以及 ESB 提供的其他服务可能会降低性能甚至成为瓶颈。

### Web 服务

Web 服务是面向服务的架构的另一种流行实现。根据其定义，Web 服务是一台机器向另一台机器（或操作者）提供的服务，通信是通过万维网协议进行的。尽管万维网的管理机构 W3C 允许使用 FTP 或 SMTP 等其他协议，但 Web 服务通常使用 HTTP 作为传输协议。

虽然可以使用专有解决方案实现 Web 服务，但大多数实现都基于开放协议和标准。尽管许多方法通常被称为 Web 服务，但它们在本质上是不同的。稍后在本章中，我们将详细描述各种方法。现在，让我们专注于它们的共同特点。

#### Web 服务的优缺点

Web 服务的好处如下：

+   使用流行的 Web 标准

+   大量的工具

+   可扩展性

以下是缺点：

+   大量开销。

+   一些实现过于复杂（例如 SOAP/WSDL/UDDI 规范）。

### 消息和流

在介绍企业服务总线架构时，我们已经提到了消息队列和消息代理。除了作为 ESB 实现的一部分外，消息系统也可以作为独立的架构元素。

#### 消息队列

消息队列是用于**进程间通信**（**IPC**）的组件。顾名思义，它们使用队列数据结构在不同进程之间传递消息。通常，消息队列是**面向消息的中间件**（**MOM**）设计的一部分。

在最低级别上，消息队列在 UNIX 规范中都有，包括 System V 和 POSIX。虽然它们在单台机器上实现 IPC 时很有趣，但我们想要专注于适用于分布式计算的消息队列。

目前在开源软件中有三种与消息队列相关的标准：

1.  **高级消息队列协议**（**AMQP**），一种在 7 层 OSI 模型的应用层上运行的二进制协议。流行的实现包括以下内容：

+   Apache Qpid

+   Apache ActiveMQ

+   RabbitMQ

+   Azure 事件中心

+   Azure 服务总线

1.  **流文本定向消息协议**（**STOMP**），一种类似于 HTTP 的基于文本的协议（使用诸如`CONNECT`、`SEND`、`SUBSCRIBE`等动词）。流行的实现包括以下内容：

+   Apache ActiveMQ

+   RabbitMQ

+   syslog-ng

1.  **MQTT**，一个面向嵌入式设备的轻量级协议。流行的实现包括以下家庭自动化解决方案：

+   OpenHAB

+   Adafruit IO

+   IoT Guru

+   Node-RED

+   Home Assistant

+   Pimatic

+   AWS IoT

+   Azure IoT Hub

#### 消息代理

消息代理处理消息系统中消息的翻译、验证和路由。与消息队列一样，它们是 MOM 的一部分。

使用消息代理，您可以最大程度地减少应用程序对系统其他部分的感知。这导致设计松散耦合的系统，因为消息代理承担了与消息上的常见操作相关的所有负担。这被称为**发布-订阅**（**PubSub**）设计模式。

代理通常管理接收者的消息队列，但也能执行其他功能，例如以下功能：

+   将消息从一种表示形式转换为另一种

+   验证消息发送者、接收者或内容

+   将消息路由到一个或多个目的地

+   聚合、分解和重组传输中的消息

+   从外部服务检索数据

+   通过与外部服务的交互增强和丰富消息

+   处理和响应错误和其他事件

+   提供不同的路由模式，如发布-订阅

消息代理的流行实现包括以下内容：

+   Apache ActiveMQ

+   Apache Kafka

+   Apache Qpid

+   Eclipse Mosquitto MQTT Broker

+   NATS

+   RabbitMQ

+   Redis

+   AWS ActiveMQ

+   AWS Kinesis

+   Azure Service Bus

### 云计算

云计算是一个广义的术语，有很多不同的含义。最初，**云**这个术语指的是架构不应该过于担心的抽象层。例如，这可能意味着由专门的运维团队管理的服务器和网络基础设施。后来，服务提供商开始将云计算这个术语应用到他们自己的产品上，这些产品通过抽象底层基础设施及其所有复杂性。不必单独配置每个基础设施部分，只需使用简单的**应用程序编程接口**（**API**）即可设置所有必要的资源。

如今，云计算已经发展到包括许多新颖的应用架构方法。它可能包括以下内容：

+   托管服务，如数据库、缓存层和消息队列

+   可扩展的工作负载编排

+   容器部署和编排平台

+   无服务器计算平台

考虑云采用时最重要的一点是，将应用程序托管在云中需要专门为云设计的架构。通常还意味着专门为特定云提供商设计的架构。

这意味着选择云提供商不仅仅是在某一时刻做出一个选择是否比另一个更好的决定。这意味着未来切换提供商的成本可能太大，不值得搬迁。在提供商之间迁移需要架构变更，对于一个正常运行的应用程序来说，这可能会超过迁移带来的预期节省。

云架构设计还有另一个后果。对于传统应用程序来说，这意味着为了利用云的好处，应用程序首先必须重新设计和重写。迁移到云并不仅仅是将二进制和配置文件从本地托管复制到由云提供商管理的虚拟机。这种方法只会意味着浪费金钱，因为只有当您的应用程序是可扩展的并且具备云意识时，云计算才是划算的。

云计算并不一定意味着使用外部服务并从第三方提供商租用机器。还有一些解决方案，比如运行在本地的 OpenStack，它允许您利用已经拥有的服务器来享受云计算的好处。

我们将在本章后面讨论托管服务。容器、云原生设计和无服务器架构将在本书的后面有专门的章节。

### 微服务

关于微服务是否属于 SOA 存在一些争议。大多数情况下，SOA 这个术语几乎等同于 ESB 设计。在许多方面，微服务与 ESB 相反。这导致了微服务是 SOA 的一个独特模式的观点，是软件架构演进的下一步。

我们认为，实际上，这些是一种现代的 SOA 方法，旨在消除 ESB 中出现的一些问题。毕竟，微服务非常符合面向服务的架构的定义。

微服务是下一章的主题。

## 面向服务的架构的好处

将系统的功能分割到多个服务中有几个好处。首先，每个服务可以单独维护和部署。这有助于团队专注于特定任务，而无需了解系统内的每种可能的交互。它还实现了敏捷开发，因为测试只需覆盖特定服务，而不是整个系统。

第二个好处是服务的模块化有助于创建分布式系统。通过网络（通常基于互联网协议）作为通信手段，服务可以分布在不同的机器之间，以提供可伸缩性、冗余性和更好的资源利用率。

当每个服务有许多生产者和许多消费者时，实施新功能和维护现有软件是一项困难的任务。这就是为什么 SOA 鼓励使用文档化和版本化的 API。

另一种使服务生产者和消费者更容易互动的方法是使用已建立的协议，描述如何在不同服务之间传递数据和元数据。这些协议可能包括 SOAP、REST 或 gRPC。

使用 API 和标准协议可以轻松创建提供超出现有服务的附加值的新服务。考虑到我们有一个返回地理位置的服务 A，另一个服务 B，它提供给定位置的当前温度，我们可以调用 A 并在请求 B 中使用其响应。这样，我们就可以获得当前位置的当前温度，而无需自己实现整个逻辑。

我们对这两个服务的所有复杂性和实现细节一无所知，并将它们视为黑匣子。这两个服务的维护者也可以引入新功能并发布新版本的服务，而无需通知我们。

测试和实验面向服务的架构也比单片应用更容易。单个地方的小改变不需要重新编译整个代码库。通常可以使用客户端工具以临时方式调用服务。

让我们回到我们的天气和地理位置服务的例子。如果这两个服务都暴露了 REST API，我们可以仅使用 cURL 客户端手动发送适当的请求来构建原型。当我们确认响应令人满意时，我们可以开始编写代码，自动化整个操作，并可能将结果公开为另一个服务。

要获得 SOA 的好处，我们需要记住所有服务都必须松散耦合。如果服务依赖于彼此的实现，这意味着它们不再是松散耦合，而是紧密耦合。理想情况下，任何给定的服务都应该可以被不同的类似服务替换，而不会影响整个系统的运行。

在我们的天气和位置示例中，这意味着在不同语言中重新实现位置服务（比如，从 Go 切换到 C++）不应影响该服务的下游消费者，只要他们使用已建立的 API。

通过发布新的 API 版本仍然可能引入 API 的破坏性变化。连接到 1.0 版本的客户端将观察到传统行为，而连接到 2.0 版本的客户端将受益于错误修复，更好的性能和其他改进，这些改进是以兼容性为代价的。

对于依赖 HTTP 的服务，API 版本通常发生在 URI 级别。因此，当调用[`service.local/v1/customer`](https://service.local/v1/customer)时，可以访问 1.0、1.1 或 1.2 版本的 API，而 2.0 版本的 API 位于[`service.local/v2/customer`](https://service.local/v2/customer)。然后，API 网关、HTTP 代理或负载均衡器能够将请求路由到适当的服务。

## SOA 的挑战

引入抽象层总是有成本的。同样的规则适用于面向服务的体系结构。当看到企业服务总线、Web 服务或消息队列和代理时，可以很容易地看到抽象成本。可能不太明显的是微服务也有成本。它们的成本与它们使用的远程过程调用（RPC）框架以及与服务冗余和功能重复相关的资源消耗有关。

与 SOA 相关的另一个批评目标是缺乏统一的测试框架。开发应用程序服务的个人团队可能使用其他团队不熟悉的工具。与测试相关的其他问题是组件的异构性和可互换性意味着有大量的组合需要测试。一些组合可能会引入通常不会观察到的边缘情况。

由于关于特定服务的知识大多集中在一个团队中，因此要理解整个应用程序的工作方式要困难得多。

当应用程序的 SOA 平台在应用程序的生命周期内开发时，可能会引入所有服务更新其版本以针对最新平台开发的需求。这意味着开发人员不再是引入新功能，而是专注于确保他们的应用程序在对平台进行更改后能够正确运行。在极端情况下，对于那些没有看到新版本并且不断修补以符合平台要求的服务，维护成本可能会急剧上升。

面向服务的体系结构遵循康威定律，详见第二章，*架构风格*。

# 采用消息传递原则

正如我们在本章前面提到的，消息传递有许多不同的用例，从物联网和传感器网络到在云中运行的基于微服务的分布式应用程序。

消息传递的好处之一是它是一种连接使用不同技术实现的服务的中立方式。在开发 SOA 时，每个服务通常由专门的团队开发和维护。团队可以选择他们感觉舒适的工具。这适用于编程语言、第三方库和构建系统。

维护统一的工具集可能会适得其反，因为不同的服务可能有不同的需求。例如，一个自助应用可能需要一个像 Qt 这样的图形用户界面（GUI）库。作为同一应用程序的一部分的硬件控制器将有其他要求，可能链接到硬件制造商的第三方组件。这些依赖关系可能会对不能同时满足两个组件的一些限制（例如，GUI 应用程序可能需要一个较新的编译器，而硬件对应可能被固定在一个较旧的编译器上）。使用消息系统来解耦这些组件让它们有单独的生命周期。

消息系统的一些用例包括以下内容：

+   金融业务

+   车队监控

+   物流捕捉

+   处理传感器

+   数据订单履行

+   任务排队

以下部分重点介绍了为低开销和使用经纪人的消息系统设计的部分。

## 低开销的消息系统

低开销的消息系统通常用于需要小占地面积或低延迟的环境。这些通常是传感器网络、嵌入式解决方案和物联网设备。它们在基于云的和分布式服务中较少见，但仍然可以在这些解决方案中使用。

### MQTT

**MQTT**代表**消息队列遥测传输**。它是 OASIS 和 ISO 下的开放标准。MQTT 通常使用 PubSub 模型，通常在 TCP/IP 上运行，但也可以与其他传输协议一起工作。

正如其名称所示，MQTT 的设计目标是低代码占用和在低带宽位置运行的可能性。还有一个名为**MQTT-SN**的单独规范，代表**传感器网络的 MQTT**。它专注于没有 TCP/IP 堆栈的电池供电的嵌入式设备。

MQTT 使用消息经纪人接收来自客户端的所有消息，并将这些消息路由到它们的目的地。QoS 提供了三个级别：

+   至多一次交付（无保证）

+   至少一次交付（已确认交付）

+   确保交付一次（已确认交付）

MQTT 特别受到各种物联网应用的欢迎并不奇怪。它受 OpenHAB、Node-RED、Pimatic、Microsoft Azure IoT Hub 和 Amazon IoT 的支持。它在即时通讯中也很受欢迎，在 ejabberd 和 Facebook Messanger 中使用。其他用例包括共享汽车平台、物流和运输。

支持此标准的两个最流行的 C++库是 Eclipse Paho 和基于 C++14 和 Boost.Asio 的 mqtt_cpp。对于 Qt 应用程序，还有 qmqtt。

### ZeroMQ

ZeroMQ 是一种无经纪人的消息队列。它支持常见的消息模式，如 PubSub、客户端/服务器和其他几种。它独立于特定的传输，并可以与 TCP、WebSockets 或 IPC 一起使用。

ZeroMQ 的主要思想是，它需要零经纪人和零管理。它也被宣传为提供零延迟，这意味着来自经纪人存在的延迟为零。

低级库是用 C 编写的，并且有各种流行编程语言的实现，包括 C++。C++的最受欢迎的实现是 cppzmq，这是一个针对 C++11 的仅头文件库。

## 经纪人消息系统

两个最受欢迎的不专注于低开销的消息系统是基于 AMQP 的 RabbitMQ 和 Apache Kafka。它们都是成熟的解决方案，在许多不同的设计中都非常受欢迎。许多文章都集中在 RabbitMQ 或 Apache Kafka 在特定领域的优越性上。

这是一个略微不正确的观点，因为这两种消息系统基于不同的范例。Apache Kafka 专注于流式传输大量数据并将流式存储在持久内存中，以允许将来重播。另一方面，RabbitMQ 通常用作不同微服务之间的消息经纪人或用于处理后台作业的任务队列。因此，在 RabbitMQ 中的路由比 Apache Kafka 中的路由更先进。Kafka 的主要用例是数据分析和实时处理。

虽然 RabbitMQ 使用 AMQP 协议（并且还支持其他协议，如 MQTT 和 STOMP），Kafka 使用基于 TCP/IP 的自己的协议。这意味着 RabbitMQ 与基于这些支持的协议的其他现有解决方案是可互操作的。如果您编写一个使用 AMQP 与 RabbitMQ 交互的应用程序，应该可以将其稍后迁移到使用 Apache Qpid、Apache ActiveMQ 或来自 AWS 或 Microsoft Azure 的托管解决方案。

扩展问题也可能会驱使选择一个消息代理而不是另一个。Apache Kafka 的架构允许轻松进行水平扩展，这意味着向现有工作机群添加更多机器。另一方面，RabbitMQ 的设计考虑了垂直扩展，这意味着向现有机器添加更多资源，而不是添加更多相似大小的机器。

# 使用 Web 服务

正如本章前面提到的，Web 服务的共同特点是它们基于标准的 Web 技术。大多数情况下，这将意味着**超文本传输协议**（**HTTP**），这是我们将重点关注的技术。尽管可能实现基于不同协议的 Web 服务，但这类服务非常罕见，因此超出了我们的范围。

## 用于调试 Web 服务的工具

使用 HTTP 作为传输的一个主要好处是工具的广泛可用性。在大多数情况下，测试和调试 Web 服务可以使用的工具不仅仅是 Web 浏览器。除此之外，还有许多其他程序可能有助于自动化。这些包括以下内容：

+   标准的 Unix 文件下载器`wget`

+   现代 HTTP 客户端`curl`

+   流行的开源库，如 libcurl、curlpp、C++ REST SDK、cpr（C++ HTTP 请求库）和 NFHTTP

+   测试框架，如 Selenium 或 Robot Framework

+   浏览器扩展，如 Boomerang

+   独立解决方案，如 Postman 和 Postwoman

+   专用测试软件，包括 SoapUI 和 Katalon Studio

基于 HTTP 的 Web 服务通过返回 HTTP 响应来处理使用适当的 HTTP 动词（如 GET、POST 和 PUT）的 HTTP 请求。请求和响应的语义以及它们应传达的数据在不同的实现中有所不同。

大多数实现属于两类：基于 XML 的 Web 服务和基于 JSON 的 Web 服务。基于 JSON 的 Web 服务目前正在取代基于 XML 的 Web 服务，但仍然常见到使用 XML 格式的服务。

对于处理使用 JSON 或 XML 编码的数据，可能需要额外的工具，如 xmllint、xmlstarlet、jq 和 libxml2。

## 基于 XML 的 Web 服务

最初获得关注的第一个 Web 服务主要基于 XML。**XML**或**可扩展标记语言**当时是分布式计算和 Web 环境中的交换格式选择。有几种不同的方法来设计带有 XML 有效负载的服务。

您可能希望与已经存在的基于 XML 的 Web 服务进行交互，这些服务可能是在您的组织内部开发的，也可能是外部开发的。但是，我们建议您使用更轻量级的方法来实现新的 Web 服务，例如基于 JSON 的 Web 服务、RESTful Web 服务或 gRPC。

### XML-RPC

最早出现的标准之一被称为 XML-RPC。该项目的理念是提供一种与当时盛行的**公共对象模型**（**COM**）和 CORBA 竞争的 RPC 技术。其目标是使用 HTTP 作为传输协议，并使格式既可读又可写，并且可解析为机器。为了实现这一点，选择了 XML 作为数据编码格式。

在使用 XML-RPC 时，想要执行远程过程调用的客户端向服务器发送 HTTP 请求。请求可能有多个参数。服务器以单个响应回答。XML-RPC 协议为参数和结果定义了几种数据类型。

尽管 SOAP 具有类似的数据类型，但它使用 XML 模式定义，这使得消息比 XML-RPC 中的消息不可读得多。

#### 与 SOAP 的关系

由于 XML-RPC 不再得到积极维护，因此没有现代的 C++实现标准。如果您想从现代代码与 XML-RPC Web 服务进行交互，最好的方法可能是使用支持 XML-RPC 和其他 XML Web 服务标准的 gSOAP 工具包。

XML-RPC 的主要批评是它在使消息显着变大的同时，没有比发送纯 XML 请求和响应提供更多价值。

随着标准的发展，它成为了 SOAP。作为 SOAP，它构成了 W3C Web 服务协议栈的基础。

### SOAP

**SOAP**的原始缩写是**Simple Object Access Protocol**。该缩写在标准的 1.2 版本中被取消。它是 XML-RPC 标准的演变。

SOAP 由三部分组成：

+   **SOAP 信封**，定义消息的结构和处理规则

+   **SOAP 头**规定应用程序特定数据类型的规则（可选）

+   **SOAP 主体**，携带远程过程调用和响应

这是一个使用 HTTP 作为传输的 SOAP 消息示例：

```cpp
POST /FindMerchants HTTP/1.1
Host: www.domifair.org
Content-Type: application/soap+xml; charset=utf-8
Content-Length: 345
SOAPAction: "http://www.w3.org/2003/05/soap-envelope"

<?xml version="1.0"?>
<soap:Envelope >
 <soap:Header>
 </soap:Header>
 <soap:Body >
    <m:FindMerchants>
      <m:Lat>54.350989</m:Lat>
      <m:Long>18.6548168</m:Long>
      <m:Distance>200</m:Distance>
    </m:FindMerchants>
  </soap:Body>
</soap:Envelope>
```

该示例使用标准的 HTTP 头和 POST 方法来调用远程过程。SOAP 特有的一个头是`SOAPAction`。它指向标识操作意图的 URI。由客户端决定如何解释此 URI。

`soap:Header`是可选的，所以我们将其留空。与`soap:Body`一起，它包含在`soap:Envelope`中。主要的过程调用发生在`soap:Body`中。我们引入了一个特定于多米尼加展会应用程序的 XML 命名空间。该命名空间指向我们域的根。我们调用的过程是`FindMerchants`，并提供三个参数：纬度、经度和距离。

由于 SOAP 被设计为可扩展、传输中立和独立于编程模型，它也导致了其他相关标准的产生。这意味着通常需要在使用 SOAP 之前学习所有相关的标准和协议。

如果您的应用程序广泛使用 XML，并且您的开发团队熟悉所有术语和规范，那么这不是问题。但是，如果您只是想为第三方公开 API，一个更简单的方法是构建 REST API，因为它对生产者和消费者来说更容易学习。

#### WSDL

**Web 服务描述语言**（**WSDL**）提供了服务如何被调用以及消息应该如何形成的机器可读描述。与其他 W3C Web 服务标准一样，它以 XML 编码。

它通常与 SOAP 一起使用，以定义 Web 服务提供的接口及其使用方式。

一旦在 WSDL 中定义了 API，您可以（而且应该！）使用自动化工具来帮助您从中创建代码。对于 C++，具有此类工具的一个框架是 gSOAP。它配备了一个名为`wsdl2h`的工具，它将根据定义生成一个头文件。然后您可以使用另一个工具`soapcpp2`，将接口定义生成到您的实现中。

不幸的是，由于消息的冗长，SOAP 服务的大小和带宽要求通常非常巨大。如果这不是问题，那么 SOAP 可以有其用途。它允许同步和异步调用，以及有状态和无状态操作。如果您需要严格、正式的通信手段，SOAP 可以提供。只需确保使用协议的 1.2 版本，因为它引入了许多改进。其中之一是服务的增强安全性。另一个是服务本身的改进定义，有助于互操作性，或者正式定义传输手段（允许使用消息队列）等，仅举几例。

#### UDDI

在记录 Web 服务接口之后的下一步是服务发现，它允许应用程序找到并连接到其他方实现的服务。

**通用描述、发现和集成**（**UDDI**）是用于 WSDL 文件的注册表，可以手动或自动搜索。与本节讨论的其他技术一样，UDDI 使用 XML 格式。

UDDI 注册表可以通过 SOAP 消息查询自动服务发现。尽管 UDDI 提供了 WSDL 的逻辑扩展，但其在开放中的采用令人失望。仍然可能会发现公司内部使用 UDDI 系统。

#### SOAP 库

SOAP 最流行的两个库是**Apache Axis**和**gSOAP**。

Apache Axis 适用于实现 SOAP（包括 WSDL）和 REST Web 服务。值得注意的是，该库在过去十年中没有发布新版本。

gSOAP 是一个工具包，允许创建和与基于 XML 的 Web 服务进行交互，重点是 SOAP。它处理数据绑定、SOAP 和 WSDL 支持、JSON 和 RSS 解析、UDDI API 等其他相关的 Web 服务标准。尽管它不使用现代 C++特性，但它仍在积极维护。

## 基于 JSON 的 Web 服务

**JSON**代表**JavaScript 对象表示法**。与名称所暗示的相反，它不仅限于 JavaScript。它是与语言无关的。大多数编程语言都有 JSON 的解析器和序列化器。JSON 比 XML 更紧凑。

它的语法源自 JavaScript，因为它是基于 JavaScript 子集的。

JSON 支持的数据类型如下：

+   数字：确切的格式可能因实现而异；在 JavaScript 中默认为双精度浮点数。

+   字符串：Unicode 编码。

+   布尔值：使用`true`和`false`值。

+   数组：可能为空。

+   对象：具有键值对的映射。

+   `null`：表示空值。

在第九章中介绍的`Packer`配置，即*持续集成/持续部署*，是 JSON 文档的一个示例：

```cpp
{
  "variables": {
    "aws_access_key": "",
    "aws_secret_key": ""
  },
  "builders": [{
    "type": "amazon-ebs",
    "access_key": "{{user `aws_access_key`}}",
    "secret_key": "{{user `aws_secret_key`}}",
    "region": "eu-central-1",
    "source_ami": "ami-5900cc36",
    "instance_type": "t2.micro",
    "ssh_username": "admin",
    "ami_name": "Project's Base Image {{timestamp}}"
  }],
  "provisioners": [{
    "type": "ansible",
    "playbook_file": "./provision.yml",
    "user": "admin",
    "host_alias": "baseimage"
  }],
  "post-processors": [{
    "type": "manifest",
    "output": "manifest.json",
    "strip_path": true
  }]
}
```

使用 JSON 作为格式的标准之一是 JSON-RPC 协议。

### JSON-RPC

JSON-RPC 是一种基于 JSON 编码的远程过程调用协议，类似于 XML-RPC 和 SOAP。与其 XML 前身不同，它需要很少的开销。它也非常简单，同时保持了 XML-RPC 的人类可读性。

这是我们之前的示例在 SOAP 调用中使用 JSON-RPC 2.0 的样子：

```cpp
{
  "jsonrpc": "2.0",
  "method": "FindMerchants",
  "params": {
    "lat": "54.350989",
    "long": "18.6548168",
    "distance": 200
  },
  "id": 1
}
```

这个 JSON 文档仍然需要适当的 HTTP 标头，但即使有标头，它仍然比 XML 对应物要小得多。唯一存在的元数据是带有 JSON-RPC 版本和请求 ID 的文件。`method`和`params`字段几乎是不言自明的。SOAP 并非总是如此。

尽管该协议轻量级、易于实现和使用，但与 SOAP 和 REST Web 服务相比，它并没有得到广泛的采用。它发布得比 SOAP 晚得多，大约与 REST 服务开始流行的时间相同。虽然 REST 迅速取得成功（可能是因为其灵活性），但 JSON-RPC 未能获得类似的推动力。

C++的两个有用的实现是 libjson-rpc-cpp 和 json-rpc-cxx。json-rpc-cxx 是先前库的现代重新实现。

## 表述性状态转移（REST）

Web 服务的另一种替代方法是**表述性状态转移（REST）。**符合这种架构风格的服务通常被称为 RESTful 服务。REST 与 SOAP 或 JSON-RPC 的主要区别在于 REST 几乎完全基于 HTTP 和 URI 语义。

REST 是一种在实现 Web 服务时定义一组约束的架构风格。符合这种风格的服务称为 RESTful。这些约束如下：

+   必须使用客户端-服务器模型。

+   无状态性（客户端和服务器都不需要存储与它们的通信相关的状态）。

+   可缓存性（响应应定义为可缓存或不可缓存，以从标准 Web 缓存中获益，以提高可伸缩性和性能）。

+   分层系统（代理和负载均衡器绝对不能影响客户端和服务器之间的通信）。

REST 使用 HTTP 作为传输协议，URI 表示资源，HTTP 动词操作资源或调用操作。关于每个 HTTP 方法应如何行为没有标准，但最常同意的语义是以下内容：

+   POST - 创建新资源。

+   GET - 检索现有资源。

+   PATCH - 更新现有资源。

+   DELETE - 删除现有资源。

+   PUT - 替换现有资源。

由于依赖于 Web 标准，RESTful Web 服务可以重用现有组件，如代理、负载均衡器和缓存。由于开销低，这样的服务也非常高效和有效。

### 描述语言

就像基于 XML 的 Web 服务一样，RESTful 服务可以以机器和人可读的方式描述。有几种竞争标准可用，其中 OpenAPI 是最受欢迎的。

#### OpenAPI

OpenAPI 是由 Linux Foundation 的 OpenAPI 计划监督的规范。它以前被称为 Swagger 规范，因为它曾经是 Swagger 框架的一部分。

该规范与语言无关。它使用 JSON 或 YAML 输入来生成方法、参数和模型的文档。这样，使用 OpenAPI 有助于保持文档和源代码的最新状态。

有许多与 OpenAPI 兼容的工具可供选择，例如代码生成器、编辑器、用户界面和模拟服务器。OpenAPI 生成器可以使用 cpp-restsdk 或 Qt 5 生成 C++代码进行客户端实现。它还可以使用 Pistache、Restbed 或 Qt 5 QHTTPEngine 生成服务器代码。还有一个方便的在线 OpenAPI 编辑器可用：[`editor.swagger.io/`](https://editor.swagger.io/)。

使用 OpenAPI 记录的 API 将如下所示：

```cpp
{
  "openapi": "3.0.0",
  "info": {
    "title": "Items API overview",
    "version": "2.0.0"
  },
  "paths": {
    "/item/{itemId}": {
      "get": {
        "operationId": "getItem",
        "summary": "get item details",
        "parameters": [
          "name": "itemId",
          "description": "Item ID",
          "required": true,
          "schema": {
            "type": "string"
          }
        ],
        "responses": {
          "200": {
            "description": "200 response",
            "content": {
              "application/json": {
                "example": {
                  "itemId": 8,
                  "name", "Kürtőskalács",
                  "locationId": 5
                }
              }
            }
          }
        }
      }
    }
  }
}
```

前两个字段（`openapi`和`info`）是描述文档的元数据。`paths`字段包含与 REST 接口的资源和方法对应的所有可能路径。在上面的示例中，我们只记录了一个路径（`/item`）和一个方法（`GET`）。此方法将`itemId`作为必需参数。我们提供了一个可能的响应代码，即`200`。200 响应包含一个 JSON 文档作为其本身的主体。与`example`键相关联的值是成功响应的示例有效负载。

#### RAML

一种竞争规范，RAML，代表 RESTful API 建模语言。它使用 YAML 进行描述，并实现了发现、代码重用和模式共享。

建立 RAML 的理念是，虽然 OpenAPI 是一个很好的工具来记录现有的 API，但在当时，它并不是设计新 API 的最佳方式。目前，该规范正在考虑成为 OpenAPI 计划的一部分。

RAML 文档可以转换为 OpenAPI 以利用可用的工具。

以下是使用 RAML 记录的 API 的示例：

```cpp
#%RAML 1.0

title: Items API overview
version: 2.0.0

annotationTypes:
  oas-summary:
    type: string
    allowedTargets: Method

/item:
  get:
    displayName: getItem
    queryParameters:
      itemId:
        type: string
    responses:
      '200':
        body:
          application/json:
            example: |
              {
                "itemId": 8,
                "name", "Kürtőskalács",
                "locationId": 5
              }
        description: 200 response
    (oas-summary): get item details
```

此示例描述了先前使用 OpenAPI 记录的相同接口。当以 YAML 序列化时，OpenAPI 3.0 和 RAML 2.0 看起来非常相似。主要区别在于，OpenAPI 3.0 要求使用 JSON 模式来记录结构。使用 RAML 2.0，可以重用现有的 XML 模式定义（XSD），这样更容易从基于 XML 的 Web 服务迁移或包含外部资源。

#### API Blueprint

API Blueprint 提出了与前两个规范不同的方法。它不依赖于 JSON 或 YAML，而是使用 Markdown 来记录数据结构和端点。

其方法类似于测试驱动的开发方法论，因为它鼓励在实施功能之前设计合同。这样，更容易测试实现是否真正履行了合同。

就像 RAML 一样，可以将 API Blueprint 规范转换为 OpenAPI，反之亦然。还有一个命令行界面和一个用于解析 API Blueprint 的 C++库，名为 Drafter，您可以在您的代码中使用。

使用 API Blueprint 记录的简单 API 示例如下：

```cpp
FORMAT: 1A

# Items API overview

# /item/{itemId}

## GET

+ Response 200 (application/json)

        {
            "itemId": 8,
            "name": "Kürtőskalács",
            "locationId": 5
        }
```

在上文中，我们看到针对`/item`端点的`GET`方法应该产生一个`200`的响应代码。在下面是我们的服务通常会返回的 JSON 消息。

API Blueprint 允许更自然的文档编写。主要缺点是它是迄今为止描述的格式中最不受欢迎的。这意味着文档和工具都远远不及 OpenAPI 的质量。

#### RSDL

类似于 WSDL，**RSDL**（或**RESTful Service Description Language**）是用于 Web 服务的 XML 描述。它与语言无关，旨在既适合人类阅读又适合机器阅读。

它比之前介绍的替代方案要不受欢迎得多。而且，它也要难得多，特别是与 API Blueprint 或 RAML 相比。

### 超媒体作为应用状态的引擎

尽管提供诸如基于*gRPC*的二进制接口可以提供出色的性能，但在许多情况下，您仍然希望拥有 RESTful 接口的简单性。如果您想要一个直观的基于 REST 的 API，**超媒体作为应用状态的引擎**（**HATEOAS**）可能是一个有用的原则。

就像您打开网页并根据显示的超媒体导航一样，您可以使用 HATEOAS 编写您的服务来实现相同的功能。这促进了服务器和客户端代码的解耦，并允许客户端快速了解哪些请求是有效的，这通常不适用于二进制 API。发现是动态的，并且基于提供的超媒体。

如果您使用典型的 RESTful 服务，在执行操作时，您会得到包含对象状态等数据的 JSON。除此之外，除此之外，您还会得到一个显示您可以在该对象上运行的有效操作的链接（URL）列表。这些链接（超媒体）是应用的引擎。换句话说，可用的操作由资源的状态确定。虽然在这种情况下，超媒体这个术语可能听起来很奇怪，但它基本上意味着链接到资源，包括文本、图像和视频。

例如，如果我们有一个 REST 方法允许我们使用 PUT 方法添加一个项目，我们可以添加一个返回参数，该参数链接到以这种方式创建的资源。如果我们使用 JSON 进行序列化，这可能采用以下形式：

```cpp
{
    "itemId": 8,
    "name": "Kürtőskalács",
    "locationId": 5,
    "links": [
        {
            "href": "item/8",
            "rel": "items",
            "type" : "GET"
        }
    ]
}
```

没有普遍接受的 HATEOAS 超媒体序列化方法。一方面，这样做可以更容易地实现，而不受服务器实现的影响。另一方面，客户端需要知道如何解析响应以找到相关的遍历数据。

HATEOAS 的好处之一是，它使得可以在服务器端实现 API 更改，而不一定会破坏客户端代码。当一个端点被重命名时，新的端点会在随后的响应中被引用，因此客户端会被告知在哪里发送进一步的请求。

相同的机制可能提供诸如分页或者使得发现给定对象可用方法变得容易的功能。回到我们的项目示例，这是我们在进行`GET`请求后可能收到的一个可能的响应：

```cpp
{
    "itemId": 8,
    "name": "Kürtőskalács",
    "locationId": 5,
    "stock": 8,
    "links": [
        {
            "href": "item/8",
            "rel": "items",
            "type" : "GET"
        },
        {
            "href": "item/8",
            "rel": "items",
            "type" : "POST"
        },
        {
            "href": "item/8/increaseStock",
            "rel": "increaseStock",
            "type" : "POST"
        }, 
        {
            "href": "item/8/decreaseStock",
            "rel": "decreaseStock",
            "type" : "POST"
        }
    ]
}
```

在这里，我们得到了两个负责修改库存的方法的链接。如果库存不再可用，我们的响应将如下所示（请注意，其中一个方法不再被广告）：

```cpp
{
    "itemId": 8,
    "name": "Kürtőskalács",
    "locationId": 5,
    "stock": 0,
    "links": [
        {
            "href": "items/8",
            "rel": "items",
            "type" : "GET"
        },
        {
            "href": "items/8",
            "rel": "items",
            "type" : "POST"
        },
        {
            "href": "items/8/increaseStock",
            "rel": "increaseStock",
            "type" : "POST"
        }
    ]
}
```

与 HATEOAS 相关的一个重要问题是，这两个设计原则似乎相互矛盾。如果遍历超媒体总是以相同的格式呈现，那么它将更容易消费。这里的表达自由使得编写不了解服务器实现的客户端变得更加困难。

并非所有的 RESTful API 都能从引入这一原则中受益-通过引入 HATEOAS，您承诺以特定方式编写客户端，以便它们能够从这种 API 风格中受益。

### C++中的 REST

Microsoft 的 C++ REST SDK 目前是在 C++应用程序中实现 RESTful API 的最佳方法之一。也被称为 cpp-restsdk，这是我们在本书中使用的库，用于说明各种示例。

## GraphQL

REST Web 服务的一个最新替代品是 GraphQL。名称中的**QL**代表**查询语言**。GraphQL 客户端直接查询和操作数据，而不是依赖服务器来序列化和呈现必要的数据。除了责任的逆转，GraphQL 还具有使数据处理更容易的机制。类型、静态验证、内省和模式都是规范的一部分。

有许多语言的 GraphQL 服务器实现，包括 C++。其中一种流行的实现是来自 Microsoft 的 cppgraphqlgen。还有许多工具可帮助开发和调试。有趣的是，由于 Hasura 或 PostGraphile 等产品在 Postgres 数据库上添加了 GraphQL API，您可以使用 GraphQL 直接查询数据库。

# 利用托管服务和云提供商

面向服务的架构可以延伸到当前的云计算趋势。虽然企业服务总线通常具有内部开发的服务，但使用云计算可以使用一个或多个云提供商提供的服务。

在为云计算设计应用程序架构时，您应该在实施任何替代方案之前始终考虑提供商提供的托管服务。例如，在决定是否要使用自己选择的插件托管自己的 PostgreSQL 数据库之前，确保您了解与提供商提供的托管数据库托管相比的权衡和成本。

当前的云计算环境提供了许多旨在处理流行用例的服务，例如以下内容：

+   存储

+   关系数据库

+   文档（NoSQL）数据库

+   内存缓存

+   电子邮件

+   消息队列

+   容器编排

+   计算机视觉

+   自然语言处理

+   文本转语音和语音转文本

+   监控、日志记录和跟踪

+   大数据

+   内容传送网络

+   数据分析

+   任务管理和调度

+   身份管理

+   密钥和秘钥管理

由于可用的第三方服务选择很多，很明显云计算如何适用于面向服务的架构。

## 云计算作为 SOA 的延伸

云计算是虚拟机托管的延伸。区别云计算提供商和传统 VPS 提供商的是两个东西：

+   云计算通过 API 可用，这使其成为一个服务本身。

+   除了虚拟机实例，云计算还提供额外的服务，如存储、托管数据库、可编程网络等。所有这些服务也都可以通过 API 获得。

有几种方式可以使用云提供商的 API 在您的应用程序中使用，我们将在下面介绍。

### 直接使用 API 调用

如果您的云提供商提供了您选择的语言可访问的 API，您可以直接从应用程序与云资源交互。

例如：您有一个允许用户上传自己图片的应用程序。该应用程序使用云 API 为每个新注册用户创建存储桶：

```cpp
#include <aws/core/Aws.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/CreateBucketRequest.h>

#include <spdlog/spdlog.h>

const Aws::S3::Model::BucketLocationConstraint region =
    Aws::S3::Model::BucketLocationConstraint::eu_central_1;

bool create_user_bucket(const std::string &username) {
  Aws::S3::Model::CreateBucketRequest request;

  Aws::String bucket_name("userbucket_" + username);
  request.SetBucket(bucket_name);

  Aws::S3::Model::CreateBucketConfiguration bucket_config;
  bucket_config.SetLocationConstraint(region);
  request.SetCreateBucketConfiguration(bucket_config);

  Aws::S3::S3Client s3_client;
  auto outcome = s3_client.CreateBucket(request);

  if (!outcome.IsSuccess()) {
    auto err = outcome.GetError();
    spdlog::error("ERROR: CreateBucket: {}: {}", 
                  err.GetExceptionName(),
                  err.GetMessage());
    return false;
  }

  return true;
}
```

在这个例子中，我们有一个 C++函数，它创建一个名为提供参数中的用户名的 AWS S3 存储桶。该存储桶配置为驻留在特定区域。如果操作失败，我们希望获取错误消息并使用`spdlog`记录。

### 通过 CLI 工具使用 API 调用

有些操作不必在应用程序运行时执行。它们通常在部署期间运行，因此可以在 shell 脚本中自动化。一个这样的用例是调用 CLI 工具来创建一个新的 VPC：

```cpp
gcloud compute networks create database --description "A VPC to access the database from private instances"
```

我们使用 Google Cloud Platform 的 gcloud CLI 工具创建一个名为`database`的网络，该网络将用于处理来自私有实例到数据库的流量。

### 使用与云 API 交互的第三方工具

让我们看一个例子，运行 HashiCorp Packer 来构建一个预先配置了你的应用程序的虚拟机实例镜像：

```cpp
{
   variables : {
     do_api_token : {{env `DIGITALOCEAN_ACCESS_TOKEN`}} ,
     region : fra1 ,
     packages : "customer"
     version : 1.0.3
  },
   builders : [
    {
       type : digitalocean ,
       api_token : {{user `do_api_token`}} ,
       image : ubuntu-20-04-x64 ,
       region : {{user `region`}} ,
       size : 512mb ,
       ssh_username : root
    }
  ],
  provisioners: [
    {
       type : file ,
       source : ./{{user `package`}}-{{user `version`}}.deb ,
       destination : /home/ubuntu/
    },
    {
       type : shell ,
       inline :[
         dpkg -i /home/ubuntu/{{user `package`}}-{{user `version`}}.deb
      ]
    }
  ]
}
```

在前面的代码中，我们提供了所需的凭据和区域，并使用构建器为我们准备了一个来自 Ubuntu 镜像的实例。我们感兴趣的实例需要有 512MB 的 RAM。然后，我们首先通过发送一个`.deb`包给它来提供实例，然后通过执行一个 shell 命令来安装这个包。

### 访问云 API

通过 API 访问云计算资源是区别于传统托管的最重要特性之一。使用 API 意味着你能够随意创建和删除实例，而无需操作员的干预。这样，就可以非常容易地实现基于负载的自动扩展、高级部署（金丝雀发布或蓝绿发布）以及应用程序的自动开发和测试环境。

云提供商通常将他们的 API 公开为 RESTful 服务。此外，他们通常还为几种编程语言提供客户端库。虽然三个最受欢迎的提供商都支持 C++作为客户端库，但来自较小供应商的支持可能有所不同。

如果你考虑将你的 C++应用程序部署到云上，并计划使用云 API，请确保你的提供商已发布了 C++ **软件开发工具包**（**SDK**）。也可以在没有官方 SDK 的情况下使用云 API，例如使用 CPP REST SDK 库，但请记住，这将需要更多的工作来实现。

要访问**Cloud SDK**，你还需要访问控制。通常，你的应用程序可以通过两种方式进行云 API 的身份验证：

+   **通过提供 API 令牌**

API 令牌应该是秘密的，永远不要存储在版本控制系统的一部分或编译后的二进制文件中。为了防止被盗，它也应该在静态时加密。

将 API 令牌安全地传递给应用程序的一种方法是通过像 HashiCorp Vault 这样的安全框架。它是可编程的秘密存储，内置租赁时间管理和密钥轮换。

+   **通过托管在具有适当访问权限的实例上**

许多云提供商允许给予特定虚拟机实例访问权限。这样，托管在这样一个实例上的应用程序就不必使用单独的令牌进行身份验证。访问控制是基于云 API 请求的实例。

这种方法更容易实现，因为它不必考虑秘密管理的需求。缺点是，当实例被入侵时，访问权限将对所有在那里运行的应用程序可用，而不仅仅是你部署的应用程序。

### 使用云 CLI

云 CLI 通常由人类操作员用于与云 API 交互。或者，它可以用于脚本编写或使用官方不支持的语言与云 API 交互。

例如，以下 Bourne Shell 脚本在 Microsoft Azure 云中创建一个资源组，然后创建属于该资源组的虚拟机：

```cpp
#!/bin/sh
RESOURCE_GROUP=dominicanfair
VM_NAME=dominic
REGION=germanynorth

az group create --name $RESOURCE_GROUP --location $REGION

az vm create --resource-group $RESOURCE_GROUP --name $VM_NAME --image UbuntuLTS --ssh-key-values dominic_key.pub
```

当寻找如何管理云资源的文档时，你会遇到很多使用云 CLI 的例子。即使你通常不使用 CLI，而更喜欢像 Terraform 这样的解决方案，有云 CLI 在手可能会帮助你调试基础设施问题。

### 使用与云 API 交互的工具

您已经了解了在使用云提供商的产品时出现供应商锁定的危险。通常，每个云提供商都会为所有其他提供商提供不同的 API 和不同的 CLI。也有一些较小的提供商提供抽象层，允许通过类似于知名提供商的 API 访问其产品。这种方法旨在帮助将应用程序从一个平台迁移到另一个平台。

尽管这样的情况很少见，但通常用于与一个提供商的服务进行交互的工具与另一个提供商的工具不兼容。当您考虑从一个平台迁移到另一个平台时，这不仅是一个问题。如果您想在多个提供商上托管应用程序，这也可能会成为一个问题。

为此，有一套新的工具，统称为**基础设施即代码**（**IaC**）工具，它们在不同提供商的顶部提供了一个抽象层。这些工具不一定仅限于云提供商。它们通常是通用的，并有助于自动化应用程序架构的许多不同层。

在[第九章](https://cdp.packtpub.com/hands_on_software_architecture_with_c__/wp-admin/post.php?post=33&action=edit)，*持续集成和持续部署*，我们简要介绍了其中一些。

## 云原生架构

新工具使架构师和开发人员能够更加抽象地构建基础架构，首先并且主要是考虑云。流行的解决方案，如 Kubernetes 和 OpenShift，正在推动这一趋势，但该领域还包括许多较小的参与者。本书的最后一章专门讨论了云原生设计，并描述了这种构建应用程序的现代方法。

# 总结

在本章中，我们了解了实施面向服务的体系结构的不同方法。由于服务可能以不同的方式与其环境交互，因此有许多可供选择的架构模式。我们了解了最流行的架构模式的优缺点。

我们专注于一些广受欢迎的方法的架构和实施方面：消息队列，包括 REST 的 Web 服务，以及使用托管服务和云平台。我们将在独立章节中更深入地介绍其他方法，例如微服务和容器。

在下一章中，我们将研究微服务。

# 问题

1.  面向服务的体系结构中服务的属性是什么？

1.  Web 服务的一些好处是什么？

1.  何时微服务不是一个好选择？

1.  消息队列的一些用例是什么？

1.  选择 JSON 而不是 XML 有哪些好处？

1.  REST 如何建立在 Web 标准之上？

1.  云平台与传统托管有何不同？

# 进一步阅读

+   *SOA 简化*：[`www.packtpub.com/product/soa-made-simple/9781849684163`](https://www.packtpub.com/product/soa-made-simple/9781849684163)

+   *SOA 食谱*：[`www.packtpub.com/product/soa-cookbook/9781847195487`](https://www.packtpub.com/product/soa-cookbook/9781847195487)
