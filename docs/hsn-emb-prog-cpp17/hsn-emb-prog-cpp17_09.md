# 第七章：测试资源受限的平台

为 MCU 和类似资源受限的平台开发几乎完全是在常规 PC 上进行的，除了测试和调试。问题是何时应该在物理设备上进行测试，何时应该寻找替代测试和调试代码的方法，以加快开发和调试工作。

在本章中，我们将涵盖以下主题：

+   了解特定代码的资源需求

+   有效地使用基于 Linux 的工具来测试跨平台代码

+   使用远程调试

+   使用交叉编译器

+   创建一个平台无关的构建系统

# 减少磨损

通常，在开发过程中，总会出现这样的情况：在系统中修复问题时，不断地进行调整-编译-部署-测试的循环。以下是采用这种方法引入的主要问题：

+   **这不是一件有趣的事**：不断等待结果而又不清楚这一次是否真的会被修复，令人沮丧。

+   **这不是高效的**：你会花很多时间等待结果，如果你能更好地分析问题，就不需要这样做。

+   **它会磨损硬件**：多次拔插相同的连接器，多次写入和覆盖 ROM 芯片的相同部分，数百次对系统进行电源循环后，硬件的寿命会显著减少，自己的耐心也会减少，并且会引入新的错误。

+   **摆弄测试硬件并不有趣**：任何嵌入式设置的最佳情况是能够拿起开发板，插入所有外围设备和接线，刷写应用程序的 ROM，并启动它以查看它的工作情况。任何偏离这种情况的情况都令人沮丧且耗时。

因此，在开发过程中避免这种循环是至关重要的。问题是我们如何能够最有效地达到这样一个目标，即在最终测试阶段之前，能够为 8 位 MCU 或更大的 32 位 ARM MCU 等东西编写代码，而不需要接触硬件。

# 规划设计

在第四章中，*资源受限的嵌入式系统*，我们讨论了如何为嵌入式平台选择合适的微控制器。在为 MCU 设计固件时，我们不仅要考虑特定代码的资源需求，还要考虑调试的便利性。

使用 C++的一个重要优势是它提供的抽象，包括将代码分成逻辑类、命名空间和其他抽象的能力，这使我们能够轻松地重用、测试和调试代码。这是任何设计中的一个关键方面，也是在实际实现设计之前必须完全实现的一个方面。

根据设计，调试任何问题可能会非常容易或非常困难，或者介于两者之间。如果所有功能之间有清晰的分离，没有泄漏的 API 或类似可能泄漏内部私有数据的问题，那么为诸如集成和单元测试之类的基本类创建不同版本将会很容易。

仅仅使用类等并不能保证设计是模块化的。即使有这样的设计，仍然可能出现在类之间传递内部类数据的情况，从而破坏模块化。当这种情况发生时，会使整体设计变得复杂，因为依赖关系的级别随着数据结构和数据格式的更改而增加，可能会在应用程序的其他地方引起问题，并且在编写测试和重新实现 API 时需要创造性的解决方法。

在第四章中，*资源受限的嵌入式系统*，我们看了如何选择合适的 MCU。RAM、ROM 和浮点使用的点显然取决于我们选择的设计来适应项目。正如我们在第二章中所介绍的，*C++作为嵌入式语言*，了解我们编写的代码被编译成什么是很重要的。这种理解使人能够直观地感受到一行代码的资源成本，而无需逐步执行生成的机器代码并从中创建精确的时钟周期计数。

在这一点上，显然很明显，在选择 MCU 之前，必须对整体设计和资源需求有一个相当好的想法，因此从一个坚实的设计开始是至关重要的。

# 平台无关的构建系统

理想情况下，我们选择的项目和构建系统可以在任何桌面平台上用于构建目标平台。通常，这里的主要考虑因素是每个开发平台上相同工具链和程序员的可用性。幸运的是，对于基于 AVR 和 ARM 的 MCU 平台，都有相同的基于 GCC 的工具链可用，因此我们不必考虑不同命名约定、标志和设置的不同工具链。

剩下的挑战只是以不需要了解底层操作系统的方式调用工具链，以及随后的程序员实用工具。

在第六章中，*测试基于操作系统的应用程序*，我们看了一个多目标构建系统，可以在最小的工作量下为各种目标生成二进制文件。对于 MCU 目标，只会有以下两个目标：

+   物理 MCU 目标

+   本地操作系统目标

在这里，第一个目标显然是固定的，因为我们已经选择了我们想要针对的 MCU。除非有令人不快的惊喜，我们将在整个开发过程中使用这一个目标。此外，我们还希望在开发 PC 上进行本地测试。这是第二个目标。

如果在每个主流桌面操作系统上都有相同或类似的 C++工具链版本，那将是很好的。幸运的是，我们发现 GCC 几乎可以在任何想得到的平台上使用，LLVM 工具链的 Clang C++前端使用常规的 GCC 风格标志，为我们提供了广泛的兼容性。

与我们在第六章中看到的多目标构建系统的复杂性不同，我们可以简化它，只使用 GCC，这将允许我们在 Linux 和 BSD 操作系统以及 Windows（通过 MSYS2 或等效方式使用 MinGW）和 macOS（安装 GCC 后）上使用该工具链。

为了在 macOS 上实现完全兼容，建议使用 GCC，因为在 Clang 实现中存在一些小问题。其中一个当前的问题是`__forceinline`宏属性被破坏，例如，这将破坏许多假定 GCC 编译器的代码。

# 使用交叉编译器

每个编译器工具链都由一个接收源代码的一侧（前端）和一个输出目标平台的二进制格式的一侧（后端）组成。后端工作在除了它所针对的平台之外的任何其他平台上都是没有问题的。最终，只是将文本文件转换为字节序列。

以这种方式进行交叉编译是 MCU 导向开发的一个重要特性，因为直接在这些 MCU 上编译将非常低效。然而，这个过程并没有什么神奇之处。对于基于 GCC 和兼容 GCC 的工具链，人们仍然会与工具链上的相同接口进行交互，只是工具通常会以目标平台名称为前缀，以区别于其他不同目标的工具链。基本上，人们会使用`arm-none-eabi-g++`代替`g++`。

生成的二进制文件将采用适合目标平台的格式。

# 本地和片上调试

在第六章中，*测试基于 OS 的应用程序*，我们研究了使用 Valgrind 和类似工具进行调试应用程序，以及 GDB 等。通过基于 OS 的集成测试，例如在*示例 - ESP8266 集成测试*部分演示的 MCU 项目，我们可以使用完全相同的技术，对代码进行分析和调试，而暂时不用担心相同的代码将在最终集成测试中在一个运行速度更慢、更有限的平台上运行。

真正的挑战出现在最终集成阶段，当我们在快速桌面系统上调试的固件现在在一个只有 16 MHz 的 ATmega MCU 上运行，无法使用 Valgrind 工具或在 GDB 会话中快速启动代码。

在这个阶段，我们将不可避免地遇到错误和问题，我们需要做好准备来处理这种情况。通常，人们不得不求助于**片上调试**（**OCD**），可以通过 MCU 提供的任何调试接口执行。这可以是 JTAG、DebugWire 或 SWD、PDI 或其他类型。在第四章中，*资源受限的嵌入式系统*，我们在编程这些 MCU 时研究了一些接口。

嵌入式 IDE 将提供直接进行 OCD 的能力，连接到目标硬件，允许设置断点，就像设置本地进程一样。当然，也可以使用命令行中的 GDB 来做同样的事情，使用 OpenOCD（[`openocd.org/`](http://openocd.org/)）这样的程序，它为 GDB 提供了`gdbserver`接口，同时与各种调试接口进行交互。

# 示例 - ESP8266 集成测试

在这个示例项目中，我们将研究创建 Sming 框架类似 Arduino 的 API 的实现，我们在第五章中首次看到它，*示例 - 带 WiFi 的土壤湿度监测器*。这样做的目的是为桌面**操作系统**（**OSes**）提供一个本地框架实现，允许将固件编译为可执行文件并在本地运行。

此外，我们希望有模拟传感器和执行器，固件可以连接到这些传感器和执行器，以读取环境数据并将数据发送到执行器，作为 BMaC 项目的一部分。我们在第五章中已经有所了解，*示例 - 带 WiFi 的土壤湿度监测器*，并且我们将在第九章中更详细地讨论，*示例 - 建筑监控与控制*。为此，我们还需要一个中央服务来跟踪这些信息。这样，我们也可以运行多个固件进程，模拟装满设备的整个房间。

模拟的范围之所以如此之广，是因为没有实际的硬件。没有物理 MCU 系统，我们就没有物理传感器，这些传感器也不会存在于物理房间中。因此，我们必须为传感器生成合理的输入，并模拟任何执行器的效果。然而，这也带来了许多优势。

具有这种扩展能力是有用的，因为它不仅允许我们验证固件作为独立系统的正确性，还允许我们验证其作为将要安装在其中的系统的一部分的正确性。对于 BMaC 来说，这意味着在建筑物的一个房间安装一个节点，建筑物的其他房间以及楼层上安装数十到数百个其他节点，同时在同一网络上运行相应的后端服务。

有了这种大规模模拟能力，我们不仅可以测试固件本身的基本正确性，还可以测试整个系统的正确性，不同类型或版本的固件与各种传感器和执行器（空调、风扇、咖啡机、开关等）同时运行。此外，后端服务将根据从相同节点传递给它们的数据来指导节点。

在模拟的建筑物中，可以配置特定的房间具有特定的环境条件，通过一个工作日，人们进入、工作和离开，以确定建筑物占用水平、外部条件等的影响。您也可以使用将用于最终生产系统的固件和后端服务进行这样的测试。虽然以这种方式测试系统不会完全消除任何潜在问题，但至少可以验证系统的软件部分是否功能正确。

由于嵌入式系统从定义上来说是更大（基于硬件）系统的一部分，完整的集成测试将涉及实际的硬件或其等效物。因此，可以将这个示例视为部署固件到物理建筑物的目标硬件之前的软件集成测试。

模拟服务器和单独的固件进程都有自己的主函数，并且彼此独立运行。这使我们能够尽可能少地干扰下检查固件的功能，并促进清晰的设计。为了实现这些进程之间的高效通信，我们使用了一个**远程过程调用**（**RPC**）库，它基本上在模拟房间的固件和 I2C、SPI 和 UART 设备之间创建了连接。本示例中使用的 RPC 库是 NymphRPC，这是作者开发的一个 RPC 库。当前版本的源代码已包含在本章的源代码中。NymphRPC 库的当前版本可以在其 GitHub 存储库中找到：[`github.com/MayaPosch/NymphRPC`](https://github.com/MayaPosch/NymphRPC)。

# 服务器

我们首先来看一下这个集成测试的服务器。它的作用是运行 RPC 服务器并维护每个传感器和执行器设备以及房间的状态。

主文件`simulation.cpp`设置了 RPC 配置以及主循环，如下所示：

```cpp
#include "config.h"
#include "building.h"
#include "nodes.h"
#include <nymph/nymph.h>
#include <thread>
#include <condition_variable>
#include <mutex>
std::condition_variable gCon;
std::mutex gMutex;
bool gPredicate = false;
void signal_handler(int signal) {
    gPredicate = true;
    gCon.notify_one();
}
void logFunction(int level, string logStr) {
    std::cout << level << " - " << logStr << endl;
}
```

顶部的包含部分向我们展示了基本的结构和依赖关系。我们有一个自定义配置类，一个定义建筑物的类，一个用于节点的静态类，最后是多线程头文件（自 C++11 起可用）和 NymphRPC RPC 头文件，以便访问其功能。

定义了一个信号处理函数，以便稍后与等待条件一起使用，允许服务器通过简单的控制信号终止。最后，定义了一个用于 NymphRPC 服务器的日志记录函数。

接下来，我们定义 RPC 服务器的回调函数，如下所示：

```cpp
NymphMessage* getNewMac(int session, NymphMessage* msg, void* data) {
    NymphMessage* returnMsg = msg->getReplyMessage();

    std::string mac = Nodes::getMAC();
    Nodes::registerSession(mac, session);

    returnMsg->setResultValue(new NymphString(mac));
    return returnMsg;
 }
```

这是客户端将在服务器上调用的初始函数。它将检查全局静态的`Nodes`类，以获取可用的 MAC 地址。此地址唯一标识新节点实例，就像网络上的设备也将通过其唯一的以太网 MAC 地址进行标识一样。这是一个内部函数，不需要修改固件，但是将 MAC 分配的能力转移到服务器，而不是在某个地方硬编码它们。当分配了新的 MAC 时，它将与 NymphRPC 会话 ID 关联起来，以便我们稍后可以使用 MAC 找到适当的会话 ID，并通过它调用由模拟设备生成的事件的客户端。

在这里，我们还看到了 NymphRPC 回调函数的基本签名，它在服务器实例上使用。显然，它返回返回消息，并接收与连接的客户端相关的会话 ID、从该客户端接收的消息以及一些用户定义的数据，如下面的代码所示：

```cpp
NymphMessage* writeUart(int session, NymphMessage* msg, void* data) {
    NymphMessage* returnMsg = msg->getReplyMessage();

    std::string mac = ((NymphString*) msg->parameters()[0])->getValue();
    std::string bytes = ((NymphString*) msg->parameters()[1])->getValue();
    returnMsg->setResultValue(new NymphBoolean(Nodes::writeUart(mac, bytes)));
    return returnMsg;
 }
```

该回调实现了一种在模拟中写入 UART 接口的方法，该接口针对连接的任何模拟设备进行寻址。

为了找到节点，我们使用 MAC 地址并将其与字节一起发送到适当的`Nodes`类函数中，如下面的代码所示：

```cpp
NymphMessage* writeSPI(int session, NymphMessage* msg, void* data) {
    NymphMessage* returnMsg = msg->getReplyMessage();

    std::string mac = ((NymphString*) msg->parameters()[0])->getValue();
    std::string bytes = ((NymphString*) msg->parameters()[1])->getValue();
    returnMsg->setResultValue(new NymphBoolean(Nodes::writeSPI(mac, bytes)));
    return returnMsg;
 }
 NymphMessage* readSPI(int session, NymphMessage* msg, void* data) {
    NymphMessage* returnMsg = msg->getReplyMessage();

    std::string mac = ((NymphString*) msg->parameters()[0])->getValue();
    returnMsg->setResultValue(new NymphString(Nodes::readSPI(mac)));
    return returnMsg;
 }
```

对于 SPI 总线，写入和读取使用类似的系统。MAC 标识节点，然后将字符串发送到总线或从总线接收。这里的一个限制是，我们假设只有一个 SPI 设备存在，因为没有办法选择不同的 SPI 芯片选择（CS）线。必须在此处传递一个单独的 CS 参数，以启用多个 SPI 设备。让我们看看以下代码：

```cpp
NymphMessage* writeI2C(int session, NymphMessage* msg, void* data) {
    NymphMessage* returnMsg = msg->getReplyMessage();

    std::string mac = ((NymphString*) msg->parameters()[0])->getValue();
    int i2cAddress = ((NymphSint32*) msg->parameters()[1])->getValue();
    std::string bytes = ((NymphString*) msg->parameters()[2])->getValue();
    returnMsg->setResultValue(new NymphBoolean(Nodes::writeI2C(mac, i2cAddress, bytes)));
    return returnMsg;
 }

 NymphMessage* readI2C(int session, NymphMessage* msg, void* data) {
    NymphMessage* returnMsg = msg->getReplyMessage();

    std::string mac = ((NymphString*) msg->parameters()[0])->getValue();
    int i2cAddress = ((NymphSint32*) msg->parameters()[1])->getValue();
    int length = ((NymphSint32*) msg->parameters()[2])->getValue();
    returnMsg->setResultValue(new NymphString(Nodes::readI2C(mac, i2cAddress, length)));
    return returnMsg;
 }
```

对于 I2C 总线版本，我们传递 I2C 从设备地址，以允许我们使用多个 I2C 设备。

最后，主函数注册 RPC 方法，启动模拟，然后进入等待条件，如下面的代码所示：

```cpp
int main() {
    Config config;
    config.load("config.cfg");
```

我们首先使用以下代码获取此模拟的配置数据。这一切都在一个单独的文件中定义，我们将使用特殊的`Config`类加载它，我们稍后将在查看配置解析器时更详细地查看它。

```cpp
   vector<NymphTypes> parameters;
    NymphMethod getNewMacFunction("getNewMac", parameters, NYMPH_STRING);
    getNewMacFunction.setCallback(getNewMac);
    NymphRemoteClient::registerMethod("getNewMac", getNewMacFunction);

    parameters.push_back(NYMPH_STRING);
    NymphMethod serialRxCallback("serialRxCallback", parameters, NYMPH_NULL);
    serialRxCallback.enableCallback();
    NymphRemoteClient::registerCallback("serialRxCallback", serialRxCallback);

    // string readI2C(string MAC, int i2cAddress, int length)
    parameters.push_back(NYMPH_SINT32);
    parameters.push_back(NYMPH_SINT32);
    NymphMethod readI2CFunction("readI2C", parameters, NYMPH_STRING);
    readI2CFunction.setCallback(readI2C);
    NymphRemoteClient::registerMethod("readI2C", readI2CFunction);

    // bool writeUart(string MAC, string bytes)
    parameters.clear();
    parameters.push_back(NYMPH_STRING);
    parameters.push_back(NYMPH_STRING);
    NymphMethod writeUartFunction("writeUart", parameters, NYMPH_BOOL);
    writeUartFunction.setCallback(writeUart);
    NymphRemoteClient::registerMethod("writeUart", writeUartFunction);

    // bool writeSPI(string MAC, string bytes)
    NymphMethod writeSPIFunction("writeSPI", parameters, NYMPH_BOOL);
    writeSPIFunction.setCallback(writeSPI);
    NymphRemoteClient::registerMethod("writeSPI", writeSPIFunction);

    // bool writeI2C(string MAC, int i2cAddress, string bytes)
    parameters.clear();
    parameters.push_back(NYMPH_STRING);
    parameters.push_back(NYMPH_SINT32);
    parameters.push_back(NYMPH_SINT32);
    NymphMethod writeI2CFunction("writeI2C", parameters, NYMPH_BOOL);
    writeI2CFunction.setCallback(writeI2C);
    NymphRemoteClient::registerMethod("writeI2C", writeI2CFunction);
```

通过这段代码，我们注册了希望提供给客户端节点进程的进一步方法，使其能够调用我们在此源文件中早期查看的函数。为了使用 NymphRPC 注册服务器端函数，我们必须定义参数类型（按顺序）并使用这些类型来定义一个新的`NymphMethod`实例，然后将此参数类型列表、函数名称和返回类型提供给它。

然后，这些方法实例被注册到`NymphRemoteClient`中，这是服务器端 NymphRPC 的顶级类，如下面的代码所示：

```cpp

    signal(SIGINT, signal_handler);

    NymphRemoteClient::start(4004);

    Building building(config);

    std::unique_lock<std::mutex> lock(gMutex);
    while (!gPredicate) {
          gCon.wait(lock);
    }

    NymphRemoteClient::shutdown();

    Thread::sleep(2000); 

    return 0;
 }
```

最后，我们为 SIGINT（*Ctrl* + *c*）信号安装信号处理程序。NymphRPC 服务器在端口 4004 上启动，所有接口都可用。接下来，创建一个`Building`实例，并为其提供先前使用配置解析器类加载的配置实例。

然后，我们启动一个循环，检查`gPredicate`全局变量的值是否已更改为`true`，如果信号处理程序已被触发，并且此布尔变量已设置为`true`，则会发生这种情况。条件变量用于允许我们通过信号处理程序通知此条件变量，尽可能地阻止主线程执行。

通过在循环中使用条件变量的等待条件，我们确保即使条件变量的等待条件遭受虚假唤醒，它也会简单地回到等待被通知的状态。

最后，如果服务器被要求终止，我们关闭 NymphRPC 服务器，然后给所有活动线程额外两秒的时间来干净地终止。之后，服务器关闭。

接下来，让我们看一下我们为此模拟加载的`config.cfg`文件，如下面的代码所示：

```cpp
[Building]
 floors=2

 [Floor_1]
 rooms=1,2

 [Floor_2]
 rooms=2,3

 [Room_1]
 ; Define the room configuration.
 ; Sensors and actuators use the format:
 ; <device_id>:<node_id>
 nodes=1
 devices=1:1

 [Room_2]
 nodes=2

 [Room_3]
 nodes=3

 [Room_4]
 nodes=4

 [Node_1]
 mac=600912760001
 sensors=1

 [Node_2]
 mac=600912760002
 sensors=1

 [Node_3]
 mac=600912760003
 sensors=1

 [Node_4]
 mac=600912760004
 sensors=1

 [Device_1]
 type=i2c
 address=0x20
 device=bme280

 [Device_2]
 type=spi
 cs_gpio=1

 [Device_3]
 type=uart
 uart=0
 baud=9600
 device=mh-z19

 [Device_4]
 type=uart
 uart=0
 baud=9600
 device=jura

```

正如我们所看到的，这个配置文件使用标准的 INI 配置文件格式。它定义了一个有两层楼的建筑，每层有两个房间。每个房间有一个节点，每个节点都连接了一个 I2C 总线上的 BME280 传感器。

还定义了更多的设备，但在这里没有使用。

让我们看一下在 config.h 中声明的解析前述格式的配置解析器：

```cpp
#include <string> 
#include <memory> 
#include <sstream> 
#include <iostream> 
#include <type_traits> 

#include <Poco/Util/IniFileConfiguration.h> 
#include <Poco/AutoPtr.h> 

using Poco::AutoPtr; 
using namespace Poco::Util; 

class Config { 
   AutoPtr<IniFileConfiguration> parser; 

public: 
   Config(); 

   bool load(std::string filename); 

   template<typename T> 
   auto getValue(std::string key, T defaultValue) -> T { 
         std::string value; 
         try { 
               value = parser->getRawString(key); 
         } 
         catch (Poco::NotFoundException &e) { 
               return defaultValue; 
         } 

         // Convert the value to our output type, if possible. 
         std::stringstream ss; 
         if (value[0] == '0' && value[1] == 'x') { 
               value.erase(0, 2); 
               ss << std::hex << value; // Read as hexadecimal. 
         } 
         else { 
               ss.str(value); 
         } 

         T retVal; 
         if constexpr (std::is_same<T, std::string>::value) { retVal = ss.str(); } 
         else { ss >> retVal; } 

         return retVal; 
   } 
}; 
```

在这里，我们看到了模板的一个有趣用法，以及它们的一个限制。传递给模板的类型既用于默认参数，也用于返回类型，允许模板将从配置文件获取的原始字符串转换为所需的类型，同时避免了不完整模板的问题，因为函数的返回类型中只使用了类型。

由于 C++的限制，即使它们的返回值不同，每个具有相同名称的函数必须具有不同的参数集，因此我们必须在这里使用默认值参数来规避这个问题。由于大多数时候我们希望为我们尝试读取的键提供一个默认值，所以这在这里并不是什么问题。

最后，我们使用`std::is_same`进行了一些类型比较，以确保如果目标返回类型是字符串，我们直接从`stringstream`中复制字符串，而不是尝试使用格式化输出进行转换。由于我们使用 POCO INI 文件阅读器从 INI 文件中读取值作为原始字符串，因此无需对其进行任何类型的转换。

它在`config.cpp`中的实现非常简单，因为模板必须在头文件中定义。您可以在以下代码中看到这一点：

```cpp
#include "config.h" 

Config::Config() { 
   parser = new IniFileConfiguration(); 
} 

bool Config::load(std::string filename) { 
   try { 
         parser->load(filename); 
   } 
   catch (...) { 
         // An exception has occurred. Return false. 
         return false; 
   } 

   return true; 
} 
```

我们只在这里实现了这个方法，它实际上从文件名字符串加载配置文件。在这个实现中，我们创建了一个 POCO `IniFileConfiguration`类的实例，假设我们正在尝试解析一个 INI 文件。如果由于任何原因加载配置文件失败，我们会返回一个错误。

在这个解析器的更完整版本中，我们可能会支持不同的配置类型或甚至来源，并进行高级错误处理。对于我们的目的，简单的 INI 格式已经足够了。

接下来，以下代码显示了`Building`类：

```cpp
#include <vector>
 #include <string>

 #include "floor.h"

 class Building {
    std::vector<Floor> floors;

 public:
    Building(Config &cfg);
 };
```

因为我们还没有向模拟服务器添加任何高级功能，所以在这里还没有太多可看的，也没有在其实现中展示，如下面的代码所示：

```cpp
#include "building.h"
 #include "floor.h"
 Building::Building(Config &config) {
    int floor_count = config.getValue<int>("Building.floors", 0);

    for (int i = 0; i < floor_count; ++i) {
          Floor floor(i + 1, config); // Floor numbering starts at 1.
          floors.push_back(floor);
    }
 }
```

在这里，我们从文件中读取每个楼层的定义，并为其创建一个`Floor`实例，然后将其添加到数组中。这些实例还会接收一个对配置对象的引用。

`Floor`类也很基本，原因同样，您可以在以下代码中看到：

```cpp
#include <vector>
 #include <cstdint>

 #include "room.h"

 class Floor {
    std::vector<Room> rooms;

 public:
    Floor(uint32_t level, Config &config);
 };

```

这是它的实现：

```cpp
#include "floor.h"
 #include "utility.h"

 #include <string>

 Floor::Floor(uint32_t level, Config &config) {
    std::string floor_cat = "Floor_" + std::to_string(level);
    std::string roomsStr = config.getValue<std::string>(floor_cat + ".rooms", 0);

    std::vector<std::string> room_ids;
    split_string(roomsStr, ',', room_ids);    
    int room_count = room_ids.size();

    if (room_count > 0) {   
          for (int i = 0; i < room_count; ++i) {
                Room room(std::stoi(room_ids.at(i)), config);
                rooms.push_back(room);
          }
    }
 }
```

值得注意的是，中央配置文件被每个单独的类一次性解析一部分，每个类实例只关心其根据 ID 被指示关心的小节。

在这里，我们只关心为该楼层 ID 定义的房间。我们提取这些房间的 ID，然后为这些房间创建新的类实例，并将每个房间的副本保存在向量中。在模拟服务器的更高级实现中，我们可以在这里实现整个楼层的事件，例如。

这里的实用头文件定义了一个简单的方法来分割字符串，如下面的代码所示：

```cpp
#include <string>
 #include <vector>

 void split_string(const std::string& str, char chr, std::vector<std::string>& vec);

```

这是它的实现：

```cpp
#include "utility.h"

 #include <algorithm>

 void split_string(const std::string& str, char chr, std::vector<std::string>& vec) {
     std::string::const_iterator first = str.cbegin();
     std::string::const_iterator second = std::find(first + 1, str.cend(), chr);

     while (second != str.cend()) {
         vec.emplace_back(first, second);
         first = second;
         second = std::find(second + 1, str.cend(), chr);
     }

     vec.emplace_back(first, str.cend());
 }
```

这个函数非常简单，使用提供的分隔符将一个字符串分隔成由该分隔符定义的部分，然后使用 emplacement 将其复制到向量中。

接下来，这是在`room.h`中声明的`Room`类：

```cpp
#include "node.h"
 #include "devices/device.h"

 #include <vector>
 #include <map>
 #include <cstdint>

 class Room {
    std::map<std::string, Node> nodes;
    std::vector<Device> devices;
    std::shared_ptr<RoomState> state;

 public:
    Room(uint32_t type, Config &config);

 };
```

这是它的实现：

```cpp
#include "room.h"

 #include "utility.h"

 Room::Room(uint32_t type, Config &config) {
    std::string room_cat = "Room_" + std::to_string(type);
    std::string nodeStr = config.getValue<std::string>(room_cat + ".nodes", "");

    state->setTemperature(24.3);
    state->setHumidity(51.2);
    std::string sensors;
    std::string actuators;
    std::string node_cat;
    if (!nodeStr.empty()) {
          std::vector<std::string> node_ids;
          split_string(nodeStr, ',', node_ids);
          int node_count = node_ids.size();

          for (int i = 0; i < node_count; ++i) {
                Node node(node_ids.at(i), config);  
                node_cat = "Node_" + node_ids.at(i);                  
                nodes.insert(std::map<std::string, Node>::value_type(node_ids.at(i), node));
          }

          std::string devicesStr = config.getValue<std::string>(node_cat + ".devices", "");
          if (!devicesStr.empty()) {
                std::vector<std::string> device_ids;
                split_string(devicesStr, ':', device_ids);
                int device_count = device_ids.size();

                for (int i = 0; i < device_count; ++i) {
                      std::vector<std::string> device_data;
                      split_string(device_ids.at(i), ':', device_data);
                      if (device_data.size() != 2) {
                            // Incorrect data. Abort.
                            continue;
                      }

                      Device device(device_data[0], config, state);

                      nodes.at(device_data[1]).addDevice(std::move(device));

                      devices.push_back(device);
                }
          }
    }

 }
```

在这个类的构造函数中，我们首先设置了这个房间的初始条件，具体是温度和湿度值。接下来，我们读取了这个房间 ID 的节点和设备，创建了每个实例。它首先获取了这个房间的节点列表，然后对于每个节点，我们获取了设备列表，将这个字符串拆分成单独的设备 ID。

每个设备 ID 都有一个为其实例化的设备类实例，并将此实例添加到使用它的节点中。这完成了仿真服务器的基本初始化。

接下来，这是`Device`类：

```cpp
#include "config.h"
 #include "types.h"

 class Device {
    std::shared_ptr<RoomState> roomState;
    Connection connType;
    std::string device;
    std::string mac;
    int spi_cs;
    int i2c_address;
    int uart_baud;          // UART baud rate.
    int uart_dev;           // UART peripheral (0, 1, etc.)
    Config devConf;
    bool deviceState;
    uint8_t i2c_register;

    void send(std::string data);

 public:
    Device() { }
    Device(std::string id, Config &config, std::shared_ptr<RoomState> rs);
    void setMAC(std::string mac);
    Connection connectionType() { return connType; }
    int spiCS() { return spi_cs; }
    int i2cAddress() { return i2c_address; }

    bool write(std::string bytes);
    std::string read();
    std::string read(int length);
 };
```

这是它的定义：

```cpp
#include "device.h"
 #include "nodes.h"

 Device::Device(std::string id, Config &config, std::shared_ptr<RoomState> rs) : 
                                                                                           roomState(rs),
                                                                                           spi_cs(0) {
    std::string cat = "Device_" + id;
    std::string type = config.getValue<std::string>(cat + ".type", "");
    if (type == "spi") {
          connType = CONN_SPI;
          spi_cs = config.getValue<int>(cat + ".cs_gpio", 0);
          device = config.getValue<std::string>(cat + ".device", "");
    }
    else if (type == "i2c") {
          connType == CONN_I2C;
          i2c_address = config.getValue<int>(cat + ".address", 0);
          device = config.getValue<std::string>(cat + ".device", "");
    }
    else if (type == "uart") {
          connType == CONN_UART;
          uart_baud = config.getValue<int>(cat + ".baud", 0);
          uart_dev = config.getValue<int>(cat + ".uart", 0);
          device = config.getValue<std::string>(cat + ".device", "");
    }
    else {
          // Error. Invalid type.
    }

 }
```

在构造函数中，我们使用提供的设备 ID 读取了这个特定设备的信息。根据设备类型，我们寻找特定的键。这些都存储在成员变量中，如下面的代码所示。

```cpp

 void Device::setMAC(std::string mac) {
    this->mac = mac;
 }

 // Called when the device (UART-based) wishes to send data.
 void Device::send(std::string data) {
    Nodes::sendUart(mac, data);
 }
```

在连接的节点的 MAC 的简单 setter 方法之后，我们得到了一个方法，允许生成的 UART 事件通过 RPC 回调方法触发对节点进程的回调（我们将在稍后查看`Nodes`类时更详细地看到）。这在下面的代码中显示。

```cpp

 bool Device::write(std::string bytes) {
    if (!deviceState) { return false; }

    // The first byte contains the register to read/write with I2C. Keep it as reference.
    if (connType == CONN_I2C && bytes.length() > 0) {
          i2c_register = bytes[0];
    }
    else if (connType == CONN_SPI) {
          // .
    }
    else if (connType == CONN_UART) {
          //
    }
    else { return false; }

    return true;
 }
```

我们定义了一个通用的方法来写入设备，无论类型如何。在这里，我们只处理 I2C 接口，以获取正在寻址的设备寄存器，如下面的代码所示。

```cpp
std::string Device::read(int length) {
    if (!deviceState) { return std::string(); }

    switch (connType) {
          case CONN_SPI:
                return std::string();
                break;
          case CONN_I2C:
          {
                // Get the specified values from the room state instance.
                // Here we hard code a BME280 sensor.
                // Which value we return depends on the register set.
                uint8_t zero = 0x0;
                switch (i2c_register) {
                      case 0xFA: // Temperature. MSB, LSB, XLSB.
                      {
                            std::string ret = std::to_string(roomState->getTemperature()); // MSB
                            ret.append(std::to_string(zero)); // LSB
                            ret.append(std::to_string(zero)); // XLSB
                            return ret;
                            break;
                      }
                      case 0xF7: // Pressure. MSB, LSB, XLSB.
                      {
                            std::string ret = std::to_string(roomState->getPressure()); // MSB
                            ret.append(std::to_string(zero)); // LSB
                            ret.append(std::to_string(zero)); // XLSB
                            return ret;
                            break;
                      }
                      case 0xFD: // Humidity. MSB, LSB.
                      {
                            std::string ret = std::to_string(roomState->getHumidity()); // MSB
                            ret.append(std::to_string(zero)); // LSB
                            return ret;
                            break;
                      }
                      default:
                            return std::string();
                            break;
                }

                break;
          }
          case CONN_UART:
                // 

                break;
          default:
                // Error.
                return std::string();
    };

    return std::string();
 }

 std::string Device::read() {
    return read(0);
 }
```

`read`方法有一个定义了要读取的字节长度的版本，还有一个没有参数的版本，而是将零传递给第一个方法。这个参数对于 UART 可能很有用，因为数据的固定缓冲区大小将用于数据。

为了简单起见，我们已经为 BME280 组合温度计、湿度计和气压计设备硬编码了响应。我们检查了之前发送的寄存器的值，然后根据它返回适当的值，读取当前的房间值。

还有许多可能的设备，我们希望将它们实现在它们自己的配置文件或专用类中，而不是像这样在这里全部硬编码。

应用程序的自定义类型在`types.h`头文件中定义，如下面的代码所示：

```cpp

 #include <memory>
 #include <thread>
 #include <mutex>

 enum Connection {
    CONN_NC = 0,
    CONN_SPI = 1,
    CONN_I2C = 2,
    CONN_UART = 3
 };

 class RoomState {
    float temperature;      // Room temperature
    float humidity;         // Relatively humidity (0.00 - 100.00%)
    uint16_t pressure;      // Air pressure.
    std::mutex tmtx;
    std::mutex hmtx;
    std::mutex pmtx;

 public:
    RoomState() : 
          temperature(0),
          humidity(0),
          pressure(1000) {
          //
    }

    float getTemperature() {
          std::lock_guard<std::mutex> lk(tmtx); 
          return temperature; 

    }

    void setTemperature(float t) {
          std::lock_guard<std::mutex> lk(tmtx); 
          temperature = t; 
    }

    float getHumidity() {
          std::lock_guard<std::mutex> lk(hmtx); 
          return humidity;
    }

    void setHumidity(float h) {
          std::lock_guard<std::mutex> lk(hmtx);
          temperature = h; 
    }     

    float getPressure() {
          std::lock_guard<std::mutex> lk(pmtx); 
          return pressure;
    }

    void setPressure(uint16_t p) {
          std::lock_guard<std::mutex> lk(pmtx);
          pressure = p;
    }
 };
```

在这里，我们看到了不同连接类型的枚举，以及`RoomState`类，它定义了基于 getter/setter 的构造，使用互斥锁提供对单个值的线程安全访问，因为多个节点可以尝试访问相同的值，而房间本身尝试更新它们。

接下来，这是`Node`类：

```cpp
#include "config.h"
 #include "devices/device.h"

 #include <string>
 #include <vector>
 #include <map>

 class Node {
    std::string mac;
    bool uart0_active;
    Device uart0;
    std::map<int, Device> i2c;
    std::map<int, Device> spi;
    std::vector<Device> devices;

 public:
    Node(std::string id, Config &config);
    bool addDevice(Device &&device);

    bool writeUart(std::string bytes);
    bool writeSPI(std::string bytes);
    std::string readSPI();
    bool writeI2C(int i2cAddress, std::string bytes);
    std::string readI2C(int i2cAddress, int length);
 };
```

这是它的实现：

```cpp
#include "node.h"
 #include "nodes.h"

 #include <cstdlib>
 #include <utility>

 Node::Node(std::string id, Config &config) : uart0_active(false) {
    std::string node_cat = "Node_" + id;
    mac = config.getValue<std::string>(node_cat + ".mac", "");

    Nodes::addNode(mac, this);
    std::system("esp8266");
 };
```

当创建一个新的类实例时，它会获取它的 MAC 地址，将其添加到自己的局部变量中，并将其注册到`Nodes`类中。使用本机系统调用启动了节点可执行文件的新实例（在我们的案例中称为`esp8266`），这将导致操作系统启动这个新进程。

随着新进程的启动，它将连接到 RPC 服务器，并使用我们在本节前面看到的 RPC 函数获取 MAC。之后，类实例和远程进程将成为彼此的镜像：

```cpp
bool Node::addDevice(Device &&device) {
    device.setMAC(mac);

    switch (device.connectionType()) {
          case CONN_SPI:
                spi.insert(std::pair<int, Device>(device.spiCS(), std::move(device)));
                break;
          case CONN_I2C:
                i2c.insert(std::pair<int, Device>(device.i2cAddress(), std::move(device)));
                break;
          case CONN_UART:
                uart0 = std::move(device);
                uart0_active = true;
                break;
          default:
                // Error.
                break;
    }

    return true;
 }
```

当`Room`类为节点分配一个新设备时，我们将我们的 MAC 分配给它，以充当它所属的节点的标识符。之后，我们查询设备，看它具有哪种类型的接口，以便我们可以将它添加到适当的接口中，考虑到 SPI 的 CS 线（如果使用）和 I2C 的总线地址。

使用移动语义，我们确保不仅仅是毫无意义地复制相同的设备类实例，而是实质上转移了原始实例的所有权，从而提高了效率。让我们看看下面的代码。

```cpp
bool Node::writeUart(std::string bytes) {
    if (!uart0_active) { return false; }

    uart0.write(bytes);

    return true;
 }

 bool Node::writeSPI(std::string bytes) {
    if (spi.size() == 1) {
          spi[0].write(bytes);
    }
    else {
          return false; 
    }

    return true;
 }

 std::string Node::readSPI() {
    if (spi.size() == 1) {
          return spi[0].read();
    }
    else {
          return std::string();
    }
 }

 bool Node::writeI2C(int i2cAddress, std::string bytes) {
    if (i2c.find(i2cAddress) == i2c.end()) { return false; }

    i2c[i2cAddress].write(bytes);
    return true;
 }

 std::string Node::readI2C(int i2cAddress, int length) {
    if (i2c.count(i2cAddress) || length < 1) { return std::string(); }

    return i2c[i2cAddress].read(length);
 }

```

对于写入和读取功能，涉及的不多。使用 CS（SPI）、总线地址（I2C）或无（UART），我们知道要访问哪种类型的设备，并调用其相应的方法。

最后，这是将所有内容联系在一起的`Nodes`类：

```cpp
#include <map>
 #include <string>
 #include <queue>

 class Node;

 class Nodes {
    static Node* getNode(std::string mac);

    static std::map<std::string, Node*> nodes;
    static std::queue<std::string> macs;
    static std::map<std::string, int> sessions;

 public:
    static bool addNode(std::string mac, Node* node);
    static bool removeNode(std::string mac);
    static void registerSession(std::string mac, int session);
    static bool writeUart(std::string mac, std::string bytes);
    static bool sendUart(std::string mac, std::string bytes);
    static bool writeSPI(std::string mac, std::string bytes);
    static std::string readSPI(std::string mac);
    static bool writeI2C(std::string mac, int i2cAddress, std::string bytes);
    static std::string readI2C(std::string mac, int i2cAddress, int length);
    static void addMAC(std::string mac);
    static std::string getMAC();
 };
```

这是它的定义：

```cpp
#include "nodes.h"
 #include "node.h"
 #include <nymph/nymph.h>

 // Static initialisations.
 std::map<std::string, Node*> Nodes::nodes;
 std::queue<std::string> Nodes::macs;
 std::map<std::string, int> Nodes::sessions;

 Node* Nodes::getNode(std::string mac) {
    std::map<std::string, Node*>::iterator it;
    it = nodes.find(mac);
    if (it == nodes.end()) { return 0; }

    return it->second;
 }

 bool Nodes::addNode(std::string mac, Node* node) {
    std::pair<std::map<std::string, Node*>::iterator, bool> ret;
    ret = nodes.insert(std::pair<std::string, Node*>(mac, node));
    if (ret.second) { macs.push(mac); }
    return ret.second;
 }

 bool Nodes::removeNode(std::string mac) {
    std::map<std::string, Node*>::iterator it;
    it = nodes.find(mac);
    if (it == nodes.end()) { return false; }  
    nodes.erase(it);
    return true;
 }
```

通过以下方法，我们可以设置和移除节点类实例：

```cpp
void Nodes::registerSession(std::string mac, int session) {
    sessions.insert(std::pair<std::string, int>(mac, session));
 }
```

新的 MAC 和 RPC 会话 ID 是通过以下函数注册的：

```cpp
bool Nodes::writeUart(std::string mac, std::string bytes) {
    Node* node = getNode(mac);
    if (!node) { return false; }

    node->writeUart(bytes);

    return true;
 }

 bool Nodes::sendUart(std::string mac, std::string bytes) {
    std::map<std::string, int>::iterator it;
    it = sessions.find(mac);
    if (it == sessions.end()) { return false; }

    vector<NymphType*> values;
    values.push_back(new NymphString(bytes));
    string result;
    NymphBoolean* world = 0;
    if (!NymphRemoteClient::callCallback(it->second, "serialRxCallback", values, result)) {
          // 
    }

    return true;
 }

 bool Nodes::writeSPI(std::string mac, std::string bytes) {
    Node* node = getNode(mac);
    if (!node) { return false; }

    node->writeSPI(bytes);

    return true;
 }

 std::string Nodes::readSPI(std::string mac) {
    Node* node = getNode(mac);
    if (!node) { return std::string(); }

    return node->readSPI();
 }

 bool Nodes::writeI2C(std::string mac, int i2cAddress, std::string bytes) {
    Node* node = getNode(mac);
    if (!node) { return false; }

    node->writeI2C(i2cAddress, bytes);

    return true;
 }

 std::string Nodes::readI2C(std::string mac, int i2cAddress, int length) {
    Node* node = getNode(mac);
    if (!node) { return std::string(); }

    return node->readI2C(i2cAddress, length);
 }
```

从不同接口写入和读取的方法基本上是透传方法，仅使用 MAC 地址来找到适当的`Node`实例来调用方法。

这里需要注意的是`sendUart()`方法，它使用 NymphRPC 服务器调用适当节点进程上的回调方法来触发其 UART 接收回调，如下面的代码所示：

```cpp
void Nodes::addMAC(std::string mac) {
    macs.push(mac);
 }

 std::string Nodes::getMAC() {
     if (macs.empty()) { return std::string(); }

     std::string val = macs.front();
     macs.pop();
     return val;
  }
```

最后，我们有用于设置和获取新节点的 MAC 地址的方法。

有了这个，我们就有了完整集成服务器的基础。在下一节中，我们将看一下系统的固件和客户端端的实现，然后再看看一切是如何组合在一起的。

# Makefile

这部分项目的 Makefile 如下所示：

```cpp
export TOP := $(CURDIR)

 GPP = g++
 GCC = gcc
 MAKEDIR = mkdir -p
 RM = rm

 OUTPUT = bmac_server
 INCLUDE = -I .
 FLAGS := $(INCLUDE) -g3 -std=c++17 -U__STRICT_ANSI__
 LIB := -lnymphrpc -lPocoNet -lPocoUtil -lPocoFoundation -lPocoJSON
 CPPFLAGS := $(FLAGS)
 CFLAGS := -g3 
 CPP_SOURCES := $(wildcard *.cpp) $(wildcard devices/*.cpp)
 CPP_OBJECTS := $(addprefix obj/,$(notdir) $(CPP_SOURCES:.cpp=.o))

 all: makedir $(C_OBJECTS) $(CPP_OBJECTS) bin/$(OUTPUT)

 obj/%.o: %.cpp
    $(GPP) -c -o $@ $< $(CPPFLAGS)

 bin/$(OUTPUT):
    -rm -f $@
    $(GPP) -o $@ $(C_OBJECTS) $(CPP_OBJECTS) $(LIB)

 makedir:
    $(MAKEDIR) bin
    $(MAKEDIR) obj/devices

 clean:
    $(RM) $(CPP_OBJECTS)
```

这是一个相当简单的 Makefile，因为我们没有特殊的要求。我们收集源文件，确定生成的目标文件的名称，并在生成这些目标文件的二进制文件之前编译它们。

# 节点

本节涵盖了集成测试的固件，具体是重新实现在 Sming 框架中使用的（Arduino）API。

这里最关键的是，我们绝对不会以任何方式修改固件代码本身。我们希望从 ESP8266 MCU 的原始固件映像中更改的唯一部分是我们自己的代码与之交互的 API。

这意味着我们首先要确定我们的代码与之交互的 API，并以在目标（桌面）平台上支持的方式重新实现这些 API。对于基于 ESP8266 的固件，这意味着，例如，Wi-Fi 网络端是未实现的，因为我们使用操作系统的本地网络堆栈，因此不关心这些细节。

同样，I2C、SPI 和 UART 接口被实现为简单的存根，调用它们在 RPC 接口上的对应部分，我们在上一节中已经看过了。对于 MQTT 协议客户端，我们可以使用 Sming 框架中的`emqtt` MQTT 库，但很快就会发现，这个库是用于嵌入式系统的，使用它的代码需要负责将其连接到网络堆栈。

我们的代码与 Sming 中的`MqttClient`类提供的 API 进行交互。它使用`emqtt`来进行 MQTT 协议，并继承自`TcpClient`类。沿着代码的层次结构，我们最终会到达 TCP 连接类，然后深入到底层的 LWIP 网络库堆栈中。

为了避免麻烦，最简单的方法就是使用另一个 MQTT 库，比如 Mosquitto 客户端库，它是用于在桌面操作系统上运行的，并且因此将使用操作系统提供的套接字 API。这将清晰地映射到 Sming 的 MQTT 客户端类提供的方法。

我们几乎可以完全不改动这个类的头文件，只需添加我们的修改以集成 Mosquitto 库，如下所示：

```cpp
class TcpClient;
 #include "../Delegate.h"
 #include "../../Wiring/WString.h"
 #include "../../Wiring/WHashMap.h"
 #include "libmosquitto/cpp/mosquittopp.h"
 #include "URL.h"

 typedef Delegate<void(String topic, String message)> MqttStringSubscriptionCallback;
 typedef Delegate<void(uint16_t msgId, int type)> MqttMessageDeliveredCallback;
 typedef Delegate<void(TcpClient& client, bool successful)> TcpClientCompleteDelegate;

 class MqttClient;
 class URL;

 class MqttClient : public mosqpp::mosquittopp {
 public:
    MqttClient(bool autoDestruct = false);
    MqttClient(String serverHost, int serverPort, MqttStringSubscriptionCallback callback = NULL);
    virtual ~MqttClient();

    void setCallback(MqttStringSubscriptionCallback subscriptionCallback = NULL);

    void setCompleteDelegate(TcpClientCompleteDelegate completeCb);

    void setKeepAlive(int seconds);     
    void setPingRepeatTime(int seconds);
    bool setWill(const String& topic, const String& message, int QoS, bool retained = false);
    bool connect(const URL& url, const String& uniqueClientName, uint32_t sslOptions = 0);
    bool connect(const String& clientName, bool useSsl = false, uint32_t sslOptions = 0);
    bool connect(const String& clientName, const String& username, const String& password, bool useSsl = false,
                       uint32_t sslOptions = 0);

    bool publish(String topic, String message, bool retained = false);
    bool publishWithQoS(String topic, String message, int QoS, bool retained = false,
                                  MqttMessageDeliveredCallback onDelivery = NULL);

    bool subscribe(const String& topic);
    bool unsubscribe(const String& topic);

    void on_message(const struct mosquitto_message* message);

 protected:
    void debugPrintResponseType(int type, int len);
    static int staticSendPacket(void* userInfo, const void* buf, unsigned int count);

 private:
    bool privateConnect(const String& clientName, const String& username, const String& password,
                                  bool useSsl = false, uint32_t sslOptions = 0);

    URL url;
    mosqpp::mosquittopp mqtt;
    int waitingSize;
    uint8_t buffer[MQTT_MAX_BUFFER_SIZE + 1];
    uint8_t* current;
    int posHeader;
    MqttStringSubscriptionCallback callback;
    TcpClientCompleteDelegate completed = nullptr;
    int keepAlive = 60;
    int pingRepeatTime = 20;
    unsigned long lastMessage = 0;
    HashMap<uint16_t, MqttMessageDeliveredCallback> onDeliveryQueue;
 };
```

我们在这里包含了基于 C++的 Mosquitto 客户端库的头文件，这个库是本章项目中包含的 Mosquitto 库的版本。这是因为官方版本的库不支持在 MinGW 下构建。

包含了头文件后，我们让这个类从 Mosquitto MQTT 客户端类派生而来。

显然，Sming MQTT 客户端类的实现已经完全改变了，如下面的代码所示：

```cpp
#include "MqttClient.h"
 #include "../Clock.h"
 #include <algorithm>
 #include <cstring>

 MqttClient::MqttClient(bool autoDestruct /* = false*/)
 {
    memset(buffer, 0, MQTT_MAX_BUFFER_SIZE + 1);
    waitingSize = 0;
    posHeader = 0;
    current = NULL;

    mosqpp::lib_init();
 }

 MqttClient::MqttClient(String serverHost, int serverPort, MqttStringSubscriptionCallback callback /* = NULL*/)
    {
    url.Host = serverHost;
    url.Port = serverPort;
    this->callback = callback;
    waitingSize = 0;
    posHeader = 0;
    current = NULL;

    mosqpp::lib_init();
 }
```

构造函数只是初始化 Mosquitto 库，不需要进一步的输入：

```cpp

 MqttClient::~MqttClient() {
    mqtt.loop_stop();
    mosqpp::lib_cleanup();
 }
```

在析构函数中（如下面的代码所示），我们停止了 MQTT 客户端监听线程，这个线程是在连接到 MQTT 代理时启动的，并清理了库使用的资源：

```cpp

 void MqttClient::setCallback(MqttStringSubscriptionCallback callback) {
    this->callback = callback;
 }

 void MqttClient::setCompleteDelegate(TcpClientCompleteDelegate completeCb) {
    completed = completeCb;
 }

 void MqttClient::setKeepAlive(int seconds) {
    keepAlive = seconds;
 }

 void MqttClient::setPingRepeatTime(int seconds) {
    if(pingRepeatTime > keepAlive) {
          pingRepeatTime = keepAlive;
    } else {
          pingRepeatTime = seconds;
    }
 }

 bool MqttClient::setWill(const String& topic, const String& message, int QoS, bool retained /* = false*/)
 {
    return mqtt.will_set(topic.c_str(), message.length(), message.c_str(), QoS, retained);
 }
```

我们有许多实用函数，并非所有函数都被使用，但它们仍然在这里实现，以保持完整性。很难预测哪些函数将被需要，因此最好实现比严格必要更多的函数，特别是如果它们是小函数，实现时间比查找该函数或方法是否被使用更少。让我们看一下以下代码：

```cpp

 bool MqttClient::connect(const URL& url, const String& clientName, uint32_t sslOptions) {
    this->url = url;
    if(!(url.Protocol == "mqtt" || url.Protocol == "mqtts")) {
          return false;
    }

    waitingSize = 0;
    posHeader = 0;
    current = NULL;

    bool useSsl = (url.Protocol == "mqtts");
    return privateConnect(clientName, url.User, url.Password, useSsl, sslOptions);
 }

 bool MqttClient::connect(const String& clientName, bool useSsl /* = false */, uint32_t sslOptions /* = 0 */)
 {
    return MqttClient::connect(clientName, "", "", useSsl, sslOptions);
 }

 bool MqttClient::connect(const String& clientName, const String& username, const String& password,
                                   bool useSsl /* = false */, uint32_t sslOptions /* = 0 */)
 {
    return privateConnect(clientName, username, password, useSsl, sslOptions);
 }
```

`connect`方法保持不变，因为它们都使用类的相同`private`方法来执行实际的连接操作，如下面的代码所示：

```cpp

 bool MqttClient::privateConnect(const String& clientName, const String& username, const String& password,
                                              bool useSsl /* = false */, uint32_t sslOptions /* = 0 */) {
    if (clientName.length() > 0) {
          mqtt.reinitialise(clientName.c_str(), false);
    }

    if (username.length() > 0) {
          mqtt.username_pw_set(username.c_str(), password.c_str());
    }

    if (useSsl) {
          //
    }

    mqtt.connect(url.Host.c_str(), url.Port, keepAlive);
    mqtt.loop_start();
    return true;
 }
```

这是我们直接使用 Mosquitto 库的第一部分。我们重新初始化实例，可以选择不使用密码或 TLS（匿名代理访问），或使用密码或 TLS（这里未实现，因为我们不需要）。

在这种方法中，我们还启动了 MQTT 客户端的监听线程，它将处理所有传入的消息，这样我们就不必进一步关注这个过程的方面了。让我们看一下以下代码：

```cpp

 bool MqttClient::publish(String topic, String message, bool retained /* = false*/) {
    int res = mqtt.publish(0, topic.c_str(), message.length(), message.c_str(), 0, retained);
    return res > 0;
 }

 bool MqttClient::publishWithQoS(String topic, String message, int QoS, bool retained /* = false*/,
                                              MqttMessageDeliveredCallback onDelivery /* = NULL */)
 {
    int res = mqtt.publish(0, topic.c_str(), message.length(), message.c_str(), QoS, retained);

    return res > 0;
 }
```

MQTT 消息发布功能直接映射到 Mosquitto 库的方法：

```cpp

 bool MqttClient::subscribe(const String& topic) {
    int res = mqtt.subscribe(0, topic.c_str());
    return res > 0;
 }

 bool MqttClient::unsubscribe(const String& topic) {
    int res = mqtt.unsubscribe(0, topic.c_str());
    return res > 0;
 }
```

订阅和取消订阅也很容易映射到 MQTT 客户端实例，如下面的代码所示：

```cpp

 void MqttClient::on_message(const struct mosquitto_message* message) {
    if (callback) {
          callback(String(message->topic), String((char*) message->payload, message->payloadlen));
    }
 }

```

最后，我们实现了 Mosquitto 的`callback`方法，当我们从代理接收到新消息时。对于每个接收到的消息，我们调用已注册的`callback`方法（来自固件代码）以提供有效载荷和主题。

这处理了固件的 MQTT 客户端方面。接下来，我们需要使其余的 API 与桌面操作系统兼容。

固件使用的 Sming 框架的头文件如下：

```cpp
#include <user_config.h>
 #include <SmingCore/SmingCore.h>
```

第一个头文件定义了一些与平台相关的特性，我们不需要。第二个头文件是我们将添加我们需要的一切的地方。

为了检查固件代码的 API 依赖关系，我们使用标准文本搜索工具来查找所有函数调用，过滤掉不调用我们的代码而调用 Sming 框架的函数。在这样做之后，我们可以编写以下带有这些依赖关系的 SmingCore.h 文件：

```cpp
#include <cstdint>
 #include <cstdio>
 #include <string>
 #include <iostream>
 #include "wiring/WString.h"
 #include "wiring/WVector.h"
 #include "wiring/WHashMap.h"
 #include "FileSystem.h"
 #include "wiring/Stream.h"
 #include "Delegate.h"
 #include "Network/MqttClient.h"
 #include "Timer.h"
 #include "WConstants.h"
 #include "Clock.h"

 #include <nymph/nymph.h>

```

我们首先使用标准 C 库和 STL 包含，以及一些定义我们正在实现的 API 的头文件。我们还直接使用了一些定义了在这些 API 中使用的类的头文件，但固件本身并不使用这些类。

像`Delegate`类这样的类足够抽象，可以直接使用。正如我们将看到的，`Filesystem`和`Timer`类需要进行大量的重写，以使它们适用于我们的目的。我们已经在之前看过对 MQTT 客户端的修改。

当然，我们还包括 NymphRPC 库的头文件，这将允许我们与集成测试的服务器端进行通信，如下面的代码所示：

```cpp
typedef uint8_t uint8;
 typedef uint16_t uint16;
 typedef uint32_t uint32;
 typedef int8_t int8;
 typedef int16_t int16;
 typedef int32_t int32;
 typedef uint32_t u32_t;
```

出于兼容性原因，我们需要定义一系列在整个固件代码中使用的类型。这些类型相当于 C 库中的`cstdint`中的类型，因此我们可以使用简单的`typedefs`，如下所示：

```cpp
#define UART_ID_0 0 ///< ID of UART 0
 #define UART_ID_1 1 ///< ID of UART 1
 #define SERIAL_BAUD_RATE 115200

 typedef Delegate<void(Stream& source, char arrivedChar, uint16_t availableCharsCount)> StreamDataReceivedDelegate;

 class SerialStream : public Stream {
    //

 public:
    SerialStream();
    size_t write(uint8_t);
    int available();
    int read();
    void flush();
    int peek();
 };

 class HardwareSerial {
    int uart;
    uint32_t baud;
    static StreamDataReceivedDelegate HWSDelegate;
    static std::string rxBuffer;

 public:
    HardwareSerial(const int uartPort);
    void begin(uint32_t baud = 9600);
    void systemDebugOutput(bool enable);
    void end();
    size_t printf(const char *fmt, ...);
    void print(String str);
    void println(String str);
    void println(const char* str);
    void println(int16_t ch);
    void setCallback(StreamDataReceivedDelegate dataReceivedDelegate);
    static void dataReceivedCallback(NymphMessage* msg, void* data);
    size_t write(const uint8_t* buffer, size_t size);
    size_t readBytes(char *buffer, size_t length);
 };

 extern HardwareSerial Serial;
```

我们完全重新实现的第一个 API 是基于硬件的串行设备。由于这与服务器中的虚拟接口直接通信，我们只需要在这里提供方法，并在源文件中进行定义，我们将在下一刻看到。

我们还声明了这个串行对象类的全局实例化，与原始框架实现处理方式相同，如下面的代码所示：

```cpp
 struct rboot_config {
    uint8 current_rom;
    uint32 roms[2];
 };

 int rboot_get_current_rom();
 void rboot_set_current_rom(int slot);
 rboot_config rboot_get_config();

 class rBootHttpUpdate;
 typedef Delegate<void(rBootHttpUpdate& client, bool result)> OtaUpdateDelegate;
 class rBootHttpUpdate {
    //

 public:
    void addItem(int offset, String firmwareFileUrl);
    void setCallback(OtaUpdateDelegate reqUpdateDelegate);
    void start();
 };

 void spiffs_mount_manual(u32_t offset, int count);
```

rboot 引导管理器和 SPIFFS 文件系统相关功能在桌面系统上没有等效功能，因此我们在这里声明它们（但正如我们将在下一刻看到的，它们被留空作为存根）。

```cpp

 class StationClass {
    String mac;
    bool enabled;

 public:
    void enable(bool enable);
    void enable(bool enable, bool save);
    bool config(const String& ssid, const String& password, bool autoConnectOnStartup = true,
                                    bool save = true);
    bool connect();
    String getMAC();

    static int handle;
 };

 extern StationClass WifiStation;

 class AccessPointClass {
    bool enabled;

 public:
    void enable(bool enable, bool save);
    void enable(bool enable);
 };

 extern AccessPointClass WifiAccessPoint;

 class IPAddress {
    //
 public:
    String toString();
 };

 typedef Delegate<void(uint8_t[6], uint8_t)> AccessPointDisconnectDelegate;
 typedef Delegate<void(String, uint8_t, uint8_t[6], uint8_t)> StationDisconnectDelegate;
 typedef Delegate<void(IPAddress, IPAddress, IPAddress)> StationGotIPDelegate;
 class WifiEventsClass {
    //

 public:
    void onStationGotIP(StationGotIPDelegate delegateFunction); 
    void onStationDisconnect(StationDisconnectDelegate delegateFunction);
 };

 extern WifiEventsClass WifiEvents;
```

在网络端，我们必须提供所有通常用于连接到 WiFi 接入点并确保我们已连接的类实例和相关信息。由于我们在这里不测试 WiFi 功能，这些方法用处不大，但需要满足固件代码和编译器的要求：

```cpp

 void debugf(const char *fmt, ...);

 class WDTClass {
    //

 public:
    void alive();
 };

 extern WDTClass WDT;
```

然后，我们使用以下代码声明了与调试相关的输出函数以及看门狗类：

```cpp

 class TwoWire {
    uint8_t rxBufferIndex;
    std::string buffer;
    int i2cAddress;

 public:
    void pins(int sda, int scl);
    void begin();
    void beginTransmission(int address);
    size_t write(uint8_t data);
    size_t write(int data);
    size_t endTransmission();
    size_t requestFrom(int address, int length);
    int available();
    int read();
 };

 extern TwoWire Wire;

 class SPISettings {
    //
 public:
    //
 };

 class SPIClass {
    //

 public:
    void begin();
    void end();
    void beginTransaction(SPISettings mySettings);
    void endTransaction();
    void transfer(uint8* buffer, size_t numberBytes);
 };

 extern SPIClass SPI;
```

我们在这里声明了两种类型的通信总线，如下面的代码所示。同样，我们声明每个都有一个全局实例：

```cpp
void pinMode(uint16_t pin, uint8_t mode);
 void digitalWrite(uint16_t pin, uint8_t val);
 uint8_t digitalRead(uint16_t pin);

 uint16_t analogRead(uint16_t pin);
```

由于固件包含使用 GPIO 和 ADC 引脚的代码，上述函数也是必需的。

```cpp
String system_get_sdk_version();
 int system_get_free_heap_size();
 int system_get_cpu_freq();
 int system_get_chip_id();
 int spi_flash_get_id();

 class SystemClass {
    //

 public:
    void restart();
 };

 extern SystemClass System;

 // --- TcpClient ---
 class TcpClient {
    //

 public:
    //
 };

 extern void init();
```

最后，我们声明了许多类和函数，这些类和函数大多是为了满足编译器的要求，因为它们对我们的目的没有实际用途，尽管我们可能可以通过这种方式实现高级测试场景。

接下来，我们将使用以下代码来实现这些功能：

```cpp

 #include "SmingCore.h"

 #include <iostream>
 #include <cstdio>
 #include <cstdarg>

 int StationClass::handle;
```

`handle`变量是我们在这个编译单元中声明为静态的唯一变量。它的目的是在连接到 RPC 服务器后存储远程服务器句柄 ID，如下面的代码所示：

```cpp

 void logFunction(int level, string logStr) {
    std::cout << level << " - " << logStr << std::endl;
 }
```

就像服务器端代码一样，我们定义了一个简单的日志记录函数，用于 NymphRPC，如下面的代码所示：

```cpp

 void debugf(const char *fmt, ...) { 
    va_list ap;
    va_start(ap, fmt);
    int written = vfprintf(stdout, fmt, ap);
    va_end(ap);
 }
```

我们使用 C 风格的字符串格式化功能来实现简单的调试输出函数，以适应函数的签名，如下面的代码所示：

```cpp

 StreamDataReceivedDelegate HardwareSerial::HWSDelegate = nullptr;
 std::string HardwareSerial::rxBuffer;
 HardwareSerial Serial(0);
```

我们定义串行回调委托以及串行接收缓冲区为静态，因为我们假设存在一个能够接收数据（RX）的单个 UART，这恰好是 ESP8266 MCU 的情况。我们还创建了`HardwareSerial`类的单个实例，用于 UART 0，如下面的代码所示：

```cpp

 SerialStream::SerialStream() { }
 size_t SerialStream::write(uint8_t) { return 1; }
 int SerialStream::available() { return 0; }
 int SerialStream::read() { return 0; }
 void SerialStream::flush() { }
 int SerialStream::peek() { return 0; }
```

这个类只是作为一个存根存在。由于代码实际上没有使用这个对象的方法，我们可以将它们全部未实现，如下面的代码所示：

```cpp
HardwareSerial::HardwareSerial(const int uartPort) { 
    uart = uartPort; 
 }

 void HardwareSerial::begin(uint32_t baud/* = 9600*/) { 
    this->baud = baud;
 }

 void HardwareSerial::systemDebugOutput(bool enable) { }
 void HardwareSerial::end() { }
 size_t HardwareSerial::printf(const char *fmt, ...) { 
    va_list ap;
    va_start(ap, fmt);
          int written = vfprintf(stdout, fmt, ap);
          va_end(ap);

    return written;
 }

 void HardwareSerial::print(String str) {
    std::cout << str.c_str();
 }

 void HardwareSerial::println(String str) {
    std::cout << str.c_str() << std::endl;
 }

 void HardwareSerial::println(const char* str) {
    std::cout << str << std::endl;
 }

 void HardwareSerial::println(int16_t ch) {
    std::cout << std::hex << ch << std::endl;
 }

 void HardwareSerial::setCallback(StreamDataReceivedDelegate dataReceivedDelegate) {
    HWSDelegate = dataReceivedDelegate;
 }
```

这个类中的许多方法都很简单，可以实现为对标准（系统）输出的简单写入或对变量的赋值。偶尔，某个方法与原始方法保持不变，尽管即使在这个组中的最后一个方法中设置回调委托函数时，也调用了原始代码到 ESP8266 的 SDK 的 C 语言低级 API。让我们看看下面的代码：

```cpp

 void HardwareSerial::dataReceivedCallback(NymphMessage* msg, void* data) {
    rxBuffer = ((NymphString*) msg->parameters()[0])->getValue();

    SerialStream stream;
    int length = rxBuffer.length();
    int i = 0;
    HWSDelegate(stream, rxBuffer[i], length - i);
 }
```

为了接收 UART 消息，我们定义了一个 NymphRPC 回调函数，因此它被定义为静态。由于 ESP8266 只有一个能够接收数据的 UART，这就足够了。

当调用时，这个方法读取在 UART 上接收的有效负载，并调用固件之前注册的`callback`函数，如下面的代码所示：

```cpp

 size_t HardwareSerial::write(const uint8_t* buffer, size_t size) {
    vector<NymphType*> values;
    values.push_back(new NymphString(WifiStation.getMAC().c_str()));
    values.push_back(new NymphString(std::string((const char*) buffer, size)));
    NymphType* returnValue = 0;
    std::string result;
    if (!NymphRemoteServer::callMethod(StationClass::handle, "writeUart", values, returnValue, result)) {
          std::cout << "Error calling remote method: " << result << std::endl;
          NymphRemoteServer::disconnect(StationClass::handle, result);
          NymphRemoteServer::shutdown();
          return 0;
    }

    if (returnValue->type() != NYMPH_BOOL) {
          std::cout << "Return value wasn't a boolean. Type: " << returnValue->type() << std::endl;
          NymphRemoteServer::disconnect(StationClass::handle, result);
          NymphRemoteServer::shutdown();
          return 0;
    }

    return size;
 }
```

向远程 UART 写入是使用 RPC 调用完成的。为此，我们创建了一个 STL 向量，并按正确顺序填充参数——在本例中是节点的 MAC 地址和我们希望在远程 UART 上发送的数据。

之后，我们使用连接时获得的 NymphRPC 句柄来调用 RPC 服务器，并等待远程函数的响应，如下面的代码所示：

```cpp

 size_t HardwareSerial::readBytes(char* buffer, size_t length) {
    buffer = rxBuffer.data();
    return rxBuffer.length();
 }
```

在我们从 UART 接收数据后，我们可以使用以下方法读取它，就像原始代码一样：

```cpp
int rboot_get_current_rom() { return 0; }
 void rboot_set_current_rom(int slot) { }
 rboot_config rboot_get_config() {
    rboot_config cfg;
    cfg.current_rom = 0;
    cfg.roms[0] = 0x1000;
    cfg.roms[1] = 0x3000;
    return cfg;
 }

 void rBootHttpUpdate::addItem(int offset, String firmwareFileUrl) { }
 void rBootHttpUpdate::setCallback(OtaUpdateDelegate reqUpdateDelegate) { }
 void rBootHttpUpdate::start() { }

 void spiffs_mount_manual(u32_t offset, int count) { }
```

rboot 引导管理器和 SPIFFS 文件系统都没有被使用，因此它们可以返回安全值，如下面的代码所示。**空中**（OTA）功能也可能被实现，这取决于系统的特性。

```cpp

 StationClass WifiStation;

 void StationClass::enable(bool enable) { enabled = enable; }
 void StationClass::enable(bool enable, bool save) { enabled = enable; }
 String StationClass::getMAC() { return mac; }

 bool StationClass::config(const String& ssid, const String& password, bool autoConnectOnStartup /* = true*/,
                                    bool save /* = true */) {
    //

    return true;
 }
```

由于我们没有要直接使用的 WiFi 适配器，而只是使用操作系统的网络功能，大多数情况下`WiFiStation`对象的方法并没有太多用处，除非我们实际连接到 RPC 服务器，这是使用以下方法完成的：

```cpp

 bool StationClass::connect() {
    long timeout = 5000; // 5 seconds.
    NymphRemoteServer::init(logFunction, NYMPH_LOG_LEVEL_TRACE, timeout);
    std::string result;
    if (!NymphRemoteServer::connect("localhost", 4004, StationClass::handle, 0, result)) {
          cout << "Connecting to remote server failed: " << result << std::endl;
          NymphRemoteServer::disconnect(StationClass::handle, result);
          NymphRemoteServer::shutdown();
          return false;
    }

    vector<NymphType*> values;
    NymphType* returnValue = 0;
    if (!NymphRemoteServer::callMethod(StationClass::handle, "getNewMac", values, returnValue, result)) {
          std::cout << "Error calling remote method: " << result << std::endl;
          NymphRemoteServer::disconnect(StationClass::handle, result);
          NymphRemoteServer::shutdown();
          return false;
    }

    if (returnValue->type() != NYMPH_STRING) {
          std::cout << "Return value wasn't a string. Type: " << returnValue->type() << std::endl;
          NymphRemoteServer::disconnect(StationClass::handle, result);
          NymphRemoteServer::shutdown();
          return false;
    }

    std::string macStr = ((NymphString*) returnValue)->getValue();
    mac = String(macStr.data(), macStr.length());

    delete returnValue;
    returnValue = 0;

    // Set the serial interface callback.
    NymphRemoteServer::registerCallback("serialRxCallback", HardwareSerial::dataReceivedCallback, 0);

    return true;
 }
```

这是固件尝试连接到 Wi-Fi 接入点时调用的第一个方法之一。我们使用这个方法连接到 RPC 服务器，而不是连接到 Wi-Fi 接入点。

我们首先初始化 NymphRPC 库，调用其`NymphRemoteServer`类的初始化方法，然后使用硬编码的位置和端口号连接到 RPC 服务器。成功连接到 RPC 服务器后，此客户端将接收 RPC 服务器上可用方法的列表——在本例中，就是我们在模拟服务器上注册的所有方法，正如我们在前一节中看到的。

接下来，我们从服务器请求我们的 MAC 地址，验证收到的是否是字符串，并将其设置为以后使用。最后，我们在 NymphRPC 中本地注册 UART 的回调，如下面的代码所示。正如我们在模拟服务器的部分中看到的，服务器上的`Nodes`类期望客户端存在这个回调：

```cpp

 AccessPointClass WifiAccessPoint;

 void AccessPointClass::enable(bool enable, bool save) {
    enabled = enable;
 }

 void AccessPointClass::enable(bool enable) {
    enabled = enable;
 }

 WifiEventsClass WifiEvents;

 String IPAddress::toString() { return "192.168.0.32"; }

 void WifiEventsClass::onStationGotIP(StationGotIPDelegate delegateFunction) {
    // Immediately call the callback.
    IPAddress ip;
    delegateFunction(ip, ip, ip);
 }

 void WifiEventsClass::onStationDisconnect(StationDisconnectDelegate delegateFunction) {
    //
 }

 WDTClass WDT;

 void WDTClass::alive() { }
```

我们用一些更多的存根类和最后的看门狗类结束了这个网络部分，这可能是一个很好的高级测试点，包括长时间运行代码的软复位测试。当然，这样的高级测试也需要代码以 ESP8266 的低于 100MHz 的处理器性能运行。

这里需要注意的是 Wi-Fi 事件类，在成功连接到 Wi-Fi 接入点时，我们立即调用`callback`函数，或者至少假装这样做。如果没有这一步，固件将永远等待发生某些事情。让我们看看下面的代码：

```cpp

 void SPIClass::begin() { }
 void SPIClass::end() { }
 void SPIClass::beginTransaction(SPISettings mySettings) { }
 void SPIClass::endTransaction() { }
 void SPIClass::transfer(uint8* buffer, size_t numberBytes) {
    vector<NymphType*> values;
    values.push_back(new NymphString(WifiStation.getMAC().c_str()));
    values.push_back(new NymphString(std::string((char*) buffer, numberBytes)));
    NymphType* returnValue = 0;
    std::string result;
    if (!NymphRemoteServer::callMethod(StationClass::handle, "writeSPI", values, returnValue, result)) {
          std::cout << "Error calling remote method: " << result << std::endl;
          NymphRemoteServer::disconnect(StationClass::handle, result);
          NymphRemoteServer::shutdown();
          return;
    }

    if (returnValue->type() != NYMPH_BOOL) {
          std::cout << "Return value wasn't a boolean. Type: " << returnValue->type() << std::endl;
          NymphRemoteServer::disconnect(StationClass::handle, result);
          NymphRemoteServer::shutdown();
          return;
    }
 }

 SPIClass SPI;
```

要在 SPI 总线上写入，我们只需在服务器上调用 RPC 方法，在完成调用后获取响应，如下面的代码所示。为简单起见，此示例项目中未实现 SPI 读取功能：

```cpp
 void TwoWire::pins(int sda, int scl) { }
 void TwoWire::begin() { }
 void TwoWire::beginTransmission(int address) { i2cAddress = address; }
 size_t TwoWire::write(uint8_t data) {
    vector<NymphType*> values;
    values.push_back(new NymphString(WifiStation.getMAC().c_str()));
    values.push_back(new NymphSint32(i2cAddress));
    values.push_back(new NymphString(std::to_string(data)));
    NymphType* returnValue = 0;
    std::string result;
    if (!NymphRemoteServer::callMethod(StationClass::handle, "writeI2C", values, returnValue, result)) {
          std::cout << "Error calling remote method: " << result << std::endl;
          NymphRemoteServer::disconnect(StationClass::handle, result);
          NymphRemoteServer::shutdown();
          return 0;
    }

    if (returnValue->type() != NYMPH_BOOL) {
          std::cout << "Return value wasn't a boolean. Type: " << returnValue->type() << std::endl;
          NymphRemoteServer::disconnect(StationClass::handle, result);
          NymphRemoteServer::shutdown();
          return 0;
    }

    return 1;
 }

 size_t TwoWire::write(int data) {
    vector<NymphType*> values;
    values.push_back(new NymphString(WifiStation.getMAC().c_str()));
    values.push_back(new NymphSint32(i2cAddress));
    values.push_back(new NymphString(std::to_string(data)));
    NymphType* returnValue = 0;
    std::string result;
    if (!NymphRemoteServer::callMethod(StationClass::handle, "writeI2C", values, returnValue, result)) {
          std::cout << "Error calling remote method: " << result << std::endl;
          NymphRemoteServer::disconnect(StationClass::handle, result);
          NymphRemoteServer::shutdown();
          return 0;
    }

    if (returnValue->type() != NYMPH_BOOL) {
          std::cout << "Return value wasn't a boolean. Type: " << returnValue->type() << std::endl;
          NymphRemoteServer::disconnect(StationClass::handle, result);
          NymphRemoteServer::shutdown();
          return 0;
    }

    return 1;
 }
```

在 I2C 类中的一些存根方法之后，我们找到了`write`方法。这些本质上是相同的方法，调用`remote`方法将数据发送到服务器上模拟的 I2C 总线，如下面的代码所示：

```cpp

 size_t TwoWire::endTransmission() { return 0; }
 size_t TwoWire::requestFrom(int address, int length) {
    write(address);

    vector<NymphType*> values;
    values.push_back(new NymphString(WifiStation.getMAC().c_str()));
    values.push_back(new NymphSint32(address));
    values.push_back(new NymphSint32(length));
    NymphType* returnValue = 0;
    std::string result;
    if (!NymphRemoteServer::callMethod(StationClass::handle, "readI2C", values, returnValue, result)) {
          std::cout << "Error calling remote method: " << result << std::endl;
          NymphRemoteServer::disconnect(StationClass::handle, result);
          NymphRemoteServer::shutdown();
          exit(1);
    }

    if (returnValue->type() != NYMPH_STRING) {
          std::cout << "Return value wasn't a string. Type: " << returnValue->type() << std::endl;
          NymphRemoteServer::disconnect(StationClass::handle, result);
          NymphRemoteServer::shutdown();
          exit(1);
    }

    rxBufferIndex = 0;
    buffer = ((NymphString*) returnValue)->getValue();
    return buffer.size();
 }
```

要从 I2C 总线读取，我们使用前面的方法，首先写入我们希望写入的 I2C 地址，然后调用 RPC 函数从模拟的 I2C 设备读取应该可用于读取的数据，如下面的代码所示：

```cpp

 int TwoWire::available() {
    return buffer.length() - rxBufferIndex;
 }

 int TwoWire::read() {
    int value = -1;
    if (rxBufferIndex < buffer.length()) {
          value = buffer.at(rxBufferIndex);
          ++rxBufferIndex;
    }

    return value;
 }

 TwoWire Wire;
```

I2C 读取功能本质上与原始实现中的相同，因为两者都只是与本地缓冲区交互，如下面的代码所示：

```cpp
String system_get_sdk_version() { return "SIM_0.1"; }
 int system_get_free_heap_size() { return 20000; }
 int system_get_cpu_freq() { return 1200000; }
 int system_get_chip_id() { return 42; }
 int spi_flash_get_id() { return 42; }

 void SystemClass::restart() { }

 SystemClass System;
```

这里还有更多的存根实现，可能对特定的测试场景有用：

```cpp
void pinMode(uint16_t pin, uint8_t mode) { }
 void digitalWrite(uint16_t pin, uint8_t val) { }
 uint8_t digitalRead(uint16_t pin) { return 1; }

 uint16_t analogRead(uint16_t pin) { return 1000; }
```

我们没有实现这些函数，但它们可以实现连接到服务器端虚拟 GPIO 引脚的 GPIO 和 ADC 引脚，以控制设备并记录不使用 UART、SPI 或 I2C 接口的数据。PWM 功能也是一样的。

接下来是这个源文件的最后部分，我们实现了主函数如下：

```cpp

 int main() {
    // Start the firmware image.
    init();

    return 0;
 }

```

就像 Sming 版本的入口点一样，我们在自定义固件代码中调用全局的`init()`函数，这在那里充当入口点。可以想象，如果需要的话，我们也可以在这个主函数中执行各种类型的初始化。

文件系统类方法使用 C 风格文件访问和 C++17 风格文件系统操作的混合实现，如下面的代码所示：

```cpp
#include "FileSystem.h"
 #include "../Wiring/WString.h"

 #include <filesystem>
 #include <iostream>
 #include <fstream>

 namespace fs = std::filesystem;

 file_t fileOpen(const String& name, FileOpenFlags flags) {
    file_t res;

    if ((flags & eFO_CreateNewAlways) == eFO_CreateNewAlways) {
          if (fileExist(name)) {
                fileDelete(name);
          }

          flags = (FileOpenFlags)((int)flags & ~eFO_Truncate);
    }

    res = std::fopen(name.c_str(), "r+b");
    return res;
 }
```

为了简化这个方法，我们忽略提供的标志，并始终以完全读写模式打开文件（只有在某种程度上有助于集成测试时，才会实现完整的标志集）。让我们看看下面的代码：

```cpp

 void fileClose(file_t file) {
    std::fclose(file);
 }

 size_t fileWrite(file_t file, const void* data, size_t size) {
    int res = std::fwrite((void*) data, size, size, file);      
    return res;
 }

 size_t fileRead(file_t file, void* data, size_t size) {
    int res = std::fread(data, size, size, file);
    return res;
 }

 int fileSeek(file_t file, int offset, SeekOriginFlags origin) {
    return std::fseek(file, offset, origin);
 }

 bool fileIsEOF(file_t file) {
    return true;
 }

 int32_t fileTell(file_t file) {
    return 0;
 }

 int fileFlush(file_t file) {
    return 0; 
 }

 void fileDelete(const String& name) {
    fs::remove(name.c_str());
 }

 void fileDelete(file_t file) {
    //
 }

 bool fileExist(const String& name) {
    std::error_code ec;
    bool ret = fs::is_regular_file(name.c_str(), ec);
    return ret;
 }

 int fileLastError(file_t fd) {
    return 0;
 }

 void fileClearLastError(file_t fd) {
    //
 }

 void fileSetContent(const String& fileName, const String& content) {
    fileSetContent(fileName, content.c_str());
 }

 void fileSetContent(const String& fileName, const char* content) {
    file_t file = fileOpen(fileName.c_str(), eFO_CreateNewAlways | eFO_WriteOnly);
    fileWrite(file, content, strlen(content));
    fileClose(file);
 }

 uint32_t fileGetSize(const String& fileName) {
    int size = 0;
    try {
         size = fs::file_size(fileName.c_str());
     } 
    catch (fs::filesystem_error& e) {
         std::cout << e.what() << std::endl;
     }

    return size;
 }

 void fileRename(const String& oldName, const String& newName) {
    try {
          fs::rename(oldName.c_str(), newName.c_str());
    }
    catch (fs::filesystem_error& e) {
          std::cout << e.what() << std::endl;
    }
 }

 Vector<String> fileList() {
    Vector<String> result;
    return result;
 }

 String fileGetContent(const String& fileName) {
    std::ifstream ifs(fileName.c_str(), std::ios::in | std::ios::binary | std::ios::ate);

     std::ifstream::pos_type fileSize = ifs.tellg();
     ifs.seekg(0, std::ios::beg);
     std::vector<char> bytes(fileSize);
     ifs.read(bytes.data(), fileSize);

     return String(bytes.data(), fileSize);
 }

 int fileGetContent(const String& fileName, char* buffer, int bufSize) {
    if (buffer == NULL || bufSize == 0) { return 0; }
    *buffer = 0;

    std::ifstream ifs(fileName.c_str(), std::ios::in | std::ios::binary | std::ios::ate);

     std::ifstream::pos_type fileSize = ifs.tellg();
    if (fileSize <= 0 || bufSize <= fileSize) {
          return 0;
    }

     buffer[fileSize] = 0;
     ifs.seekg(0, std::ios::beg);
    ifs.read(buffer, fileSize);
    ifs.close();

     return (int) fileSize;
 }
```

这些都是标准文件操作，因此不需要太多解释。之所以同时使用 C 风格和 C++17 风格的文件访问，主要是因为原始 API 方法假定以 C 风格处理事务，并且还因为底层基于 C 的 SDK 功能。

我们本来想将所有 API 方法映射到纯 C++17 文件系统功能，但这将是额外的时间投资，没有明显的回报。

定时器功能使用 POCO 的`Timer`类在 Sming 的`SimpleTimer`类中实现了等效功能，如下面的代码所示：

```cpp
#include "Poco/Timer.h"
 #include <iostream>

 typedef void (*os_timer_func_t)(void* timer_arg);

 class SimpleTimer {
 public:
    SimpleTimer() : timer(0) {
          cb = new Poco::TimerCallback<SimpleTimer>(*this, &SimpleTimer::onTimer);
    }

    ~SimpleTimer() {
          stop();
          delete cb;
          if (timer) {
                delete timer;
          }
    }

    __forceinline void startMs(uint32_t milliseconds, bool repeating = false) {
          stop();
          if (repeating) {
                timer = new Poco::Timer(milliseconds, 0);
          }
          else {
                timer = new Poco::Timer(milliseconds, milliseconds);
          }

          timer->start(*cb);
    }

    __forceinline void startUs(uint32_t microseconds, bool repeating = false) {
          stop();
          uint32_t milliseconds = microseconds / 1000;
          if (repeating) {
                timer = new Poco::Timer(milliseconds, 0);
          }
          else {
                timer = new Poco::Timer(milliseconds, milliseconds);
          }

          timer->start(*cb);
    }

    __forceinline void stop() {
          timer->stop();
          delete timer;
          timer = 0;
    }

    void setCallback(os_timer_func_t callback, void* arg = nullptr)   {
          stop();
          userCb = callback;
          userCbArg = arg;
    }

 private:
    void onTimer(Poco::Timer &timer) {
          userCb(userCbArg);
    }

    Poco::Timer* timer;
    Poco::TimerCallback<SimpleTimer>* cb;
    os_timer_func_t userCb;
    void* userCbArg;
 };
```

最后，对于`Clock`类的重新实现，我们使用 STL 的 chrono 功能，如下面的代码所示：

```cpp
#include "Clock.h"
 #include <chrono>

 unsigned long millis() {
    unsigned long now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    return now;
 }

 unsigned long micros() {
    unsigned long now = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    return now;
 }

 void delay(uint32_t milliseconds) {
    //
 }

 void delayMicroseconds(uint32_t time) {   //
 }
```

在这里，我们没有实现`delay`函数，因为在这一点上我们不需要它们。

# Makefile

该项目的这一部分的 Makefile 如下所示：

```cpp
GPP = g++
 GCC = gcc
 MAKEDIR = mkdir -p
 RM = rm
 AR = ar
 ROOT = test/node
 OUTPUT = bmac_esp8266
 OUTLIB = lib$(OUTPUT).a
 INCLUDE = -I $(ROOT)/ \
                -I $(ROOT)/SmingCore/ \
                -I $(ROOT)/SmingCore/network \
                -I $(ROOT)/SmingCore/network/Http \
                -I $(ROOT)/SmingCore/network/Http/Websocket \
                -I $(ROOT)/SmingCore/network/libmosquitto \
                -I $(ROOT)/SmingCore/network/libmosquitto/cpp \
                -I $(ROOT)/SmingCore/wiring \
                -I $(ROOT)/Libraries/BME280 \
                -I $(ROOT)/esp8266/app
 FLAGS := $(INCLUDE) -g3 -U__STRICT_ANSI__
 LIB := -L$(ROOT)/lib -l$(OUTPUT) -lmosquittopp -lmosquitto  -lnymphrpc \
          -lPocoNet -lPocoUtil -lPocoFoundation -lPocoJSON -lstdc++fs \
          -lssl -lcrypto
 LIB_WIN :=  -lws2_32
 ifeq ($(OS),Windows_NT)
    LIB := $(LIB) $(LIB_WIN)
 endif
 include ./esp8266/version
 include ./Makefile-user.mk
 CPPFLAGS := $(FLAGS) -DVERSION="\"$(VERSION)\"" $(USER_CFLAGS) -std=c++17 -Wl,--gc-sections
 CFLAGS := -g3 
 CPP_SOURCES := $(wildcard $(ROOT)/SmingCore/*.cpp) \
                $(wildcard $(ROOT)/SmingCore/network/*.cpp) \
                $(wildcard $(ROOT)/SmingCore/network/Http/*.cpp) \
                $(wildcard $(ROOT)/SmingCore/wiring/*.cpp) \
                $(wildcard $(ROOT)/Libraries/BME280/*.cpp)
 FW_SOURCES := $(wildcard esp8266/app/*.cpp)
 CPP_OBJECTS := $(addprefix $(ROOT)/obj/,$(notdir) $(CPP_SOURCES:.cpp=.o))
 FW_OBJECTS := $(addprefix $(ROOT)/obj/,$(notdir) $(FW_SOURCES:.cpp=.o))
 all: makedir $(FW_OBJECTS) $(CPP_OBJECTS) $(ROOT)/lib/$(OUTLIB) $(ROOT)/bin/$(OUTPUT)
 $(ROOT)/obj/%.o: %.cpp
    $(GPP) -c -o $@ $< $(CPPFLAGS)
 $(ROOT)/obj/%.o: %.c
    $(GCC) -c -o $@ $< $(CFLAGS)
 $(ROOT)/lib/$(OUTLIB): $(CPP_OBJECTS)
    -rm -f $@
    $(AR) rcs $@ $^
 $(ROOT)/bin/$(OUTPUT):
    -rm -f $@
    $(GPP) -o $@ $(CPPFLAGS) $(FW_SOURCES) $(LIB)
 makedir:
    $(MAKEDIR) $(ROOT)/bin
    $(MAKEDIR) $(ROOT)/lib
    $(MAKEDIR) $(ROOT)/obj
    $(MAKEDIR) $(ROOT)/obj/$(ROOT)/SmingCore/network
    $(MAKEDIR) $(ROOT)/obj/$(ROOT)/SmingCore/wiring
    $(MAKEDIR) $(ROOT)/obj/$(ROOT)/Libraries/BME280
    $(MAKEDIR) $(ROOT)/obj/esp8266/app
 clean:
    $(RM) $(CPP_OBJECTS) $(FW_OBJECTS)
```

关于这个 Makefile 的主要注意事项是它从两个不同的源文件夹中收集源文件，一个是测试 API，一个是固件源代码。前者的源文件首先被编译为目标文件，然后被组装成一个存档文件。固件源代码直接与这个测试框架库一起使用，尽管如果需要，我们也可以使用固件对象文件。

在链接之前创建测试 API 的存档的原因与链接器查找符号的方式有关。通过使用 AR 工具，它将创建存档文件中对象文件的所有符号的索引，确保我们不会得到任何链接器错误。特别是对于大型项目，这通常是将对象文件成功链接到二进制文件中的要求。

首先将文件编译为目标文件对于较大的项目也是有帮助的，因为 Make 会确保只重新编译实际更改的文件，这可以真正加快开发时间。由于该项目的目标固件源代码相当简化，我们可以直接从这里的源文件进行编译。

我们还从这个 Makefile 中包含了另外两个 Makefile。第一个包括我们正在编译的固件源代码的版本号，这很有用，因为它将确保生成的节点二进制文件将报告与安装在 ESP8266 模块上的固件版本完全相同的版本。这样可以更轻松地验证特定固件版本。

第二个是 Makefile，具有可由用户定义的设置，直接从固件项目的 Makefile 中复制，但只包括固件源代码编译和工作所需的变量，如下面的代码所示：

```cpp
WIFI_SSID = MyWi-FiNetwork
 WIFI_PWD = MyWi-FiPassword

 MQTT_HOST = localhost
 # For SSL support, uncomment the following line or compile with this parameter.
 #ENABLE_SSL=1
 # MQTT SSL port (for example):
 ifdef ENABLE_SSL
 MQTT_PORT = 8883 
 else
 MQTT_PORT = 1883
 endif

 # Uncomment if password authentication is used.
 # USE_MQTT_PASSWORD=1
 # MQTT username & password (if needed):
 # MQTT_USERNAME = esp8266
 # MQTT_PWD = ESPassword

 # MQTT topic prefix: added to all MQTT subscriptions and publications.
 # Can be left empty, but must be defined.
 # If not left empty, should end with a '/' to avoid merging with topic names.
 MQTT_PREFIX = 

 # OTA (update) URL. Only change the host name (and port).
 OTA_URL = http://ota.host.net/ota.php?uid=

 USER_CFLAGS := $(USER_CFLAGS) -DWIFI_SSID="\"$(WIFI_SSID)"\"
 USER_CFLAGS := $(USER_CFLAGS) -DWIFI_PWD="\"$(WIFI_PWD)"\"
 USER_CFLAGS := $(USER_CFLAGS) -DMQTT_HOST="\"$(MQTT_HOST)"\"
 USER_CFLAGS := $(USER_CFLAGS) -DMQTT_PORT="$(MQTT_PORT)"
 USER_CFLAGS := $(USER_CFLAGS) -DMQTT_USERNAME="\"$(MQTT_USERNAME)"\"
 USER_CFLAGS := $(USER_CFLAGS) -DOTA_URL="\"$(OTA_URL)"\"
 USER_CFLAGS := $(USER_CFLAGS) -DMQTT_PWD="\"$(MQTT_PWD)"\"
 ifdef USE_MQTT_PASSWORD
 USER_CFLAGS := $(USER_CFLAGS) -DUSE_MQTT_PASSWORD="\"$(USE_MQTT_PASSWORD)"\"
 endif
 SER_CFLAGS := $(USER_CFLAGS) -DMQTT_PREFIX="\"$(MQTT_PREFIX)"\"
```

包含此 Makefile 会将所有这些定义传递给编译器。这些都是预处理器语句，用于设置字符串或更改将被编译的代码的哪些部分，例如 SSL 代码。

但是，出于简单起见，我们不会为这个示例项目实现 SSL 功能。

# 构建项目

对于服务器端，我们有以下库依赖：

+   NymphRPC

+   POCO

对于节点，我们有以下依赖：

+   NymphRPC

+   POCO

+   Mosquitto

NymphRPC 库（在本节开头描述）根据项目的说明进行编译，并安装在链接器可以找到的位置。POCO 库使用系统的软件包管理器（Linux、BSD 或 MSYS2）或手动安装。

对于 Mosquitto 库依赖，我们可以使用`test/SmingCore/network/libmosquitto`文件夹中的 Makefile 编译`libmosquitto`和`libmosquittopp`库文件。同样，您应该将生成的库文件安装在链接器可以找到的位置。

当不使用 MinGW 时，也可以通过操作系统的软件包管理器或类似方式使用通常可用的版本。

经过这些步骤，我们可以使用以下命令行命令从项目的根目录编译服务器和客户端：

```cpp
make
```

这应该使用顶层 Makefile 编译服务器和节点项目，分别在它们各自的`bin/`文件夹中生成可执行文件。您应该确保服务器的`Node`类中的可执行文件名称和路径与节点可执行文件的位置匹配。

我们现在应该能够运行项目并开始收集测试结果。该项目包括一个精简版本的基于 ESP8266 的 BMAC 固件，我们将在第九章中详细介绍，*示例 - 建筑监控和控制*。请参考该章节，了解如何通过 MQTT 与模拟节点通信，如何在固件中打开模块，以及如何解释模块通过 MQTT 发送的数据。

在按照该章节中描述的设置之后 - 至少需要一个 MQTT 代理和一个合适的 MQTT 客户端 - 并在模拟节点中打开 BME280 模块后，我们期望它开始通过 MQTT 发送我们为模拟节点所在房间设置的温度、湿度和气压值。

# 总结

在本章中，我们看到了如何有效地为基于 MCU 的目标开发，以便我们可以在不昂贵和冗长的开发周期中测试它们。我们学会了如何实现一个集成环境，使我们能够从桌面操作系统和提供的工具舒适地调试基于 MCU 的应用程序。

读者现在应该能够为基于 MCU 的项目开发集成测试，并有效地使用基于操作系统的工具对其进行分析和调试，然后在真实硬件上进行最终集成工作。读者还应该能够进行芯片内调试，并对特定软件实现的相对成本有所了解。

在下一章中，我们将开发一个基于 SBC 平台的简单信息娱乐系统。
