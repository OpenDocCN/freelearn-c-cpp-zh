# 示例 - 带 Wi-Fi 的土壤湿度监测器

保持室内植物存活并非易事。本章中的示例项目将向您展示如何创建一个带有泵或类似执行器的 Wi-Fi 土壤湿度监测器，例如阀门和重力供水水箱。使用内置的 Web 服务器，我们将能够使用基于浏览器的用户界面来监控植物健康和控制系统功能，或者使用其基于 HTTP 的 REST API 将其集成到更大的系统中。

本章涵盖的主题如下：

+   编程 ESP8266 微控制器

+   将传感器和执行器连接到 ESP8266

+   在此平台上实现 HTTP 服务器

+   开发一个基于网页的用户界面以进行监控和控制

+   将项目集成到更大的网络中

# 让植物快乐

为了让植物存活，你需要许多东西：

+   营养

+   光

+   水

在这些方法中，前两种通常通过富含营养的土壤和将植物放置在光照充足的地方来处理。在满足这两个条件后，保持植物存活的主要问题通常是第三个问题，因为这需要每天处理。

在这里，不仅仅是简单地保持水分充足的问题，而是要保持在土壤中有足够但不过多的水分范围内。土壤中水分过多会影响植物通过根部吸收的氧气量。因此，当土壤中水分过多时，植物会枯萎死亡。

另一方面，水分过少意味着植物无法吸收足够的水来补偿通过其叶子蒸发的水分，也无法将养分输送到根部。在这种情况下，植物也会枯萎死亡。

当手动给植物浇水时，我们倾向于使用对植物可能需要更多水的粗略估计，以及用手指对表层土壤湿度的表面测试。这告诉我们很少关于实际上在植物根部周围有多少水分，远在土壤上层之下。

为了更精确地测量土壤湿度，我们可以使用多种方法：

| **类型** | **原理** | **注意事项** |
| --- | --- | --- |
| 硫酸钙块 | 电阻—— | 水被硫酸钙吸收，部分溶解，这允许电流在两个电极之间流动。电阻值指示土壤水分张力。 |
| 张力计 | 真空 | 一根空心管的一端有一个真空计，另一端有一个多孔尖端，允许水自由进出。土壤通过管子吸水会增加真空传感器的读数，表明从土壤中提取水分对植物来说更困难（水分张力）。 |
| 电容探头 | **频域反射测量法** (**FDR**) | 利用振荡电路中两个金属电极（在土壤中）之间的介电常数来测量由于水分含量变化而引起的该常数的改变。指示水分含量。 |
| 微波传感器 | **时域反射计**（TDR） | 测量微波信号从平行探头的一端传播到另一端所需的时间，这取决于土壤的介电常数。测量水分含量。 |
| ThetaProbe | RF 幅度阻抗 | 通过四个包围土壤圆柱体的探头发送一个 100 MHz 的正弦波无线电信号。使用正弦波阻抗的变化来计算土壤中的水分。 |
| 电阻探头 | 电阻 | 这与石膏块类似，只是有电极。因此，这仅测量水的存在（及其导电性），而不是土壤水分张力。 |

所有这些传感器类型都带有它们自己的优缺点。在石膏块和张力计的情况下，维护量很大，因为前者依赖于足够的石膏残留物来溶解，而不会破坏校准，而在后者的情况下，必须确保气密密封保持完好，以防止空气进入管中。任何密封的缝隙都会立即使真空传感器失效。

另一个重要的问题是成本。虽然基于 FDR 和 TDR 的探测器可能非常准确，但它们也往往非常昂贵。这通常导致只想尝试土壤湿度传感器的用户选择基于电阻或电容的传感器。在这里，前一种传感器类型的主要缺点在一个月或更短的时间内变得明显：腐蚀。

在含有离子的溶液中悬挂两个电极，并给其中一个电极施加电流时，简单的化学反应会导致其中一个电极迅速腐蚀（材料损失），直到它不再功能。这也会使土壤受到金属分子的污染。在单个电极上使用交流电（AC）而不是直流电可以稍微减少腐蚀作用，但这仍然是一个问题。

在价格低廉且仍然准确的土壤湿度传感器中，只有电容探头符合所有要求。它的精度足够进行合理的测量和比较（校准后），它不受土壤中水分的影响，也不会以任何方式影响土壤。

要实际上将水输送到植物，我们需要有一种方法来提供恰好适量的水。在这里，系统的规模主要决定了水输送的选择。对于灌溉整个田地，我们可以使用基于叶轮的泵，每分钟可以输送许多升的水。

对于单株植物，我们最多需要每分钟提供几百毫升的水。在这里，某种像蠕动泵这样的设备几乎是理想的。这种泵也用于实验室和医疗应用中，在这些应用中，你必须提供少量液体，并且需要高精度。

# 我们的解决方案

为了保持简单，我们只需构建一个能够照顾单一植物的系统。这将为我们提供最大的灵活性，因为我们只需在每个植物旁边放置一个系统，无论它是在窗台上、桌子上还是某个地方的露台上。

除了测量土壤湿度水平外，我们还想能够让系统在设定的触发水平下自动给植物浇水，并能够监控这个过程。这需要某种形式的网络访问，最好是无线访问，这样我们就不需要再铺设任何电缆，除了电源连接器。

这使得 ESP8266 MCU 非常吸引人，NodeMCU 开发板是开发和调试系统的理想目标。我们将连接一个土壤湿度传感器和一个蠕动泵。

通过使用网络浏览器连接到 ESP8266 系统的 IP 地址，我们可以看到系统的当前状态，包括土壤湿度水平以及可选的更多信息。配置系统和更多操作将通过常用的紧凑型二进制 MQTT 协议完成，系统也会发布当前系统状态，以便我们可以将其读入数据库进行显示和分析。

这样，我们还可以在以后编写后端服务，将许多这样的节点组合成一个具有集中控制和管理的完整系统。这是我们将在第九章（示例 - 构建监控和控制）中详细探讨的内容。

# 硬件

我们理想的解决方案是拥有最精确的传感器，同时不花费太多。这意味着我们基本上必须使用电容式传感器，正如我们在本章前面所看到的。这些传感器可以作为电容式土壤湿度传感器以几欧元或美元的价格获得，例如这些基于简单 555 定时器 IC 的设计：

![图片](img/9473a55a-2149-4762-9a3d-1580277a0462.png)

您只需将这些设备插入土壤中，直到电路开始的地方，然后连接电源以及连接到 MCU 的模拟到数字转换器。

大多数可购买的蠕动泵都需要 12V。这意味着我们需要有一个能够提供 5V 和 12V 的电源，或者使用所谓的升压转换器将 5V 转换为 12V。无论如何，我们还需要有一种方法来打开或关闭泵。使用升压转换器，我们可以使用其*使能*引脚通过我们的 MCU 上的 GPIO 引脚来打开或关闭其输出。

对于原型设计，我们可以使用这些常见的 5V 到 12V 升压转换模块，这些模块使用 ME2149 升压开关稳压器：

![图片](img/97a91881-f1ca-4bbf-a786-fd122d1bba40.png)

这些模块没有以任何方式引出使能引脚，但我们可以轻松地焊接一根线到相应的引脚上：

![图片](img/d992fb6a-fb74-480f-9d5c-4cec48efd14b.png)

这个升压转换模块的输出然后连接到蠕动泵：

![](img/9e0cb710-095e-4f43-b2ba-b9c0145085ea.png)

在这里，我们需要获取一些正确直径的软管，将其连接到水储罐和植物。泵本身可以朝任意方向旋转。由于它本质上是一组位于内部管道段的滚轮，这些滚轮推动液体向一个方向流动，因此泵的任意一侧都可以是输入或输出。

在测试之前，务必使用两个容器和一些水测试流向，并在泵外壳上标记方向，以及正负终端连接。

除了这些组件之外，我们还想连接一个 RGB LED，用于一些信号指示以及外观。为此，我们将使用**APA102** RGB LED 模块，该模块通过 SPI 总线连接到 ESP8266：

![](img/080f68db-2083-4eff-b67e-536e6aee587c.png)

我们可以使用一个可以提供 5V 电压和 1A 或更多电流的单电源，同时还能应对每次泵被激活时从升压转换器中突然抽取的电力。

整个系统看起来会是这样：

![](img/541c3d8a-ca18-49c7-96cc-1e863fb62a79.png)

# 固件

对于这个项目，我们将实现一个模块，该模块将用于我们在第九章，“示例 - 构建监控和控制”中将要使用的相同固件。因此，本章将仅涵盖这个植物灌溉模块的独特部分。

在我们可以开始固件本身之前，我们首先必须设置开发环境。这包括安装 ESP8266 SDK 和 Sming 框架。

# 设置 Sming

基于 Sming 的 ESP8266 开发环境可以在 Linux、Windows 和 macOS 上使用。你最好使用 Sming 的开发分支，因为在 Linux（或在 Linux 虚拟机中，或在 Windows 10 的**Windows Subsystem for Linux**（**WSL**）中）使用它是最简单的方式，并且强烈推荐。在 Linux 中，建议在`/opt`文件夹中安装，以确保与 Sming 快速入门指南的一致性。

Linux 的快速入门指南可以在[`github.com/SmingHub/Sming/wiki/Linux-Quickstart`](https://github.com/SmingHub/Sming/wiki/Linux-Quickstart)找到。

在 Linux 上，我们可以使用 ESP8266 的 Open SDK，它采用官方的 Espressif（非 RTOS）SDK，并用开源替代品替换了它所能替换的所有非开放组件。可以使用以下代码进行安装：

```cpp
    git clone --recursive https://github.com/pfalcon/esp-open-sdk.git
    cd esp-open-sdk
    make VENDOR_SDK=1.5.4 STANDALONE=y  
```

这将获取 Open SDK 的当前源代码并编译它，目标是官方 SDK 的 1.5.4 版本。虽然 SDK 的 2.0 版本已经存在，但 Sming 框架中可能仍然存在一些兼容性问题。使用 1.5.4 版本提供了几乎相同的体验，同时使用了经过良好测试的代码。当然，随着时间的推移，这将会发生变化，所以请务必查看官方 Sming 文档以获取更新的说明。

`STANDALONE` 选项意味着 SDK 将作为一个独立的 SDK 和工具链安装，没有其他依赖。这是与 Sming 一起使用的理想选项。

安装 `Sming` 就像这样简单：

```cpp
    git clone https://github.com/SmingHub/Sming.git
    cd Sming
    make  
```

这将构建 Sming 框架。如果我们正在向 Sming 的 `Libraries` 文件夹中添加新库，我们必须再次执行最后一步，以便构建和安装新的 Sming 共享库实例。

对于本项目，在编译 Sming 之前，将本章软件项目的 `libs` 文件夹中的文件夹复制到 `Sming/Sming/Libraries` 文件夹中，否则项目代码将无法编译。

我们还可以编译带有 SSL 支持的 Sming。这需要我们使用 `ENABLE_SSL=1` 参数编译 Make。这将使 axTLS 基于加密支持在编译 Sming 库时启用。

完成这些步骤后，我们只需安装 `esptool.py` 和 `esptool2`。在 `/opt` 文件夹中，执行以下命令以获取 esptool：

```cpp
    wget https://github.com/themadinventor/esptool/archive/master.zip
    unzip master.zip
    mv esptool-master esp-open-sdk/esptool  
```

`Esptool.py` 是一个 Python 脚本，它允许我们与每个 ESP8266 模块中作为其一部分的 SPI ROM 进行通信。这是我们用我们的代码闪存 MCU ROM 的方式。此工具由 Sming 自动使用：

```cpp
    cd  $ESP_HOME
    git clone https://github.com/raburton/esptool2
    cd esptool2
    make  
```

`esptool2` 工具是官方 SDK 中将链接器输出转换为我们可以写入 ESP8266 的 ROM 格式的脚本集的替代品。当我们在编译我们的应用程序时，Sming 会调用它。

最后，假设我们已经将 SDK 和 Sming 安装在 `/opt` 下，我们可以添加以下全局变量和将以下内容添加到系统 `PATH` 变量中：

```cpp
    export ESP_HOME=/opt/esp-open-sdk
    export SMING_HOME=/opt/Sming/Sming
    export PATH=$PATH:$ESP_HOME/esptool2
    export PATH=$PATH:$ESP_HOME/xtensa-lx106-elf/bin  
```

最后一行将工具链的二进制文件添加到路径中，当我们调试 ESP8266 应用程序时，我们将需要它，正如我们在第七章测试资源受限平台中看到的。在此阶段，我们可以使用 Sming 进行开发并创建可以写入 MCU 的 ROM 映像。

# 植物模块代码

在本节中，我们将查看本项目的源代码基础，从核心模块 `OtaCore` 开始，接着是 `BaseModule` 类，所有固件模块都会注册该类。最后，我们来看 `PlantModule` 类本身，它包含了本章讨论的项目需求中的业务逻辑。

值得注意的是，对于本项目，我们在项目的 Makefile 中启用了 rBoot bootmanager 和 rBoot Big Flash 选项。这样做会在我们的 ESP8266 模块（即所有 ESP-12E/F 模块）的 4 MB ROM 中创建四个 1 MB 的块，其中两个用于固件映像，其余两个用于文件存储（使用 SPIFFS 文件系统）。

然后将 rBoot 引导加载程序写入 ROM 的开始处，以便在每次引导时首先加载。在固件槽位中，任何给定时间只有一个槽位是活动的。此设置的便利功能是它允许我们轻松执行**空中**（**OTA**）更新，通过将新的固件映像写入非活动固件槽位，更改活动槽位，并重新启动 MCU。如果 rBoot 无法从新的固件映像引导，它将回退到另一个固件槽位，这是我们已知的工作固件，我们从中执行了 OTA 更新。|

# Makefile-user.mk

在 `project` 文件夹的根目录中，我们找到了此 Makefile。它包含了一些我们可能想要设置以适应我们目的的设置：

| **名称** | **描述** |
| --- | --- |
| `COM_PORT` | 如果我们总是将板连接到相同的串行端口，我们可以在此处将其硬编码以节省一些输入。 |
| `SPI_MODE` | 这设置了在将固件映像闪存到 SPI ROM 时使用的 SPI 模式。使用 `dio` 仅有两个数据线（`SD_D0`、`D1`）或四个（`SD_D0-3`）。并非所有 SPI ROM 都连接了所有四个数据线。`qio` 模式更快，但 `dio` 应始终工作。 |
| `RBOOT_ENABLED` | 当设置为 1 时，此选项启用 rBoot 引导加载程序支持。我们希望启用此功能。 |
| `RBOOT_BIG_FLASH` | 有 4 MB 的 ROM 可用，我们希望使用所有这些。请启用此选项。 |
| `RBOOT_TWO_ROMS` | 如果我们希望将两个固件映像放置在单个 1 MB ROM 芯片中，可以使用此选项。这适用于一些 ESP8266 模块及其衍生产品。 |
| `SPI_SIZE` | 在这里，我们设置 SPI ROM 芯片的大小，对于此项目应该是 4M。 |
| `SPIFF_FILES` | 包含将放置在写入 MCU 的 SPIFFS ROM 映像中的文件的文件夹的位置。 |
| `SPIFFS_SIZE` | 要创建的 SPIFFS ROM 映像的大小。在这里，64 KB 是标准，但如果我们需要使用启用了 `RBOOT_BIG_FLASH` 选项的 4 MB ROM，我们可以使用高达 1 MB。 |
| `WIFI_SSID` | 我们希望连接的 Wi-Fi 网络的 SSID。 |
| `WIFI_PWD` | Wi-Fi 网络的密码。 |
| `MQTT_HOST` | 要使用的 MQTT 服务器（代理）的 URL 或 IP 地址。 |
| `ENABLE_SSL` | 启用此选项以在 Sming 中编译 SSL 支持，使固件使用 TLS 加密连接与 MQTT 代理通信。 |
| `MQTT_PORT` | MQTT 代理的端口。这取决于是否启用了 SSL。 |
| `USE_MQTT_PASSWORD` | 如果您希望使用用户名和密码连接到 MQTT 代理，请设置为 true。 |
| `MQTT_USERNAME` | 如果需要，MQTT 代理的用户名。 |
| `MQTT_PWD` | 如果需要，MQTT 代理的密码。 |
| `MQTT_PREFIX` | 如果需要，您可以在固件使用的每个 MQTT 主题之前可选地添加一个前缀。如果为空，它必须以斜杠结尾。 |
| `OTA_URL` | 当请求 OTA 更新时，固件将使用的硬编码 URL。 |

在这些设置中，Wi-Fi、MQTT 和 OTA 设置是必不可少的，因为它们将允许应用程序连接到网络和 MQTT 代理，以及接收固件更新，而无需通过串行接口对 MCU 进行烧录。

# Main

主要源文件以及与之相关的应用程序入口点相当无趣：

```cpp
#include "ota_core.h"
void onInit() {
    // 
}
void init() {
         OtaCore::init(onInit);
 }
```

在包含主要应用程序逻辑的 `OtaCore` 类中，我们只需调用其静态初始化函数，如果我们希望在核心类完成设置网络、MQTT 和其他功能之后执行任何进一步逻辑，我们可以提供一个回调函数。

# OtaCore

在这个类中，我们为特定功能模块设置了所有基本网络功能，并提供用于日志记录和 MQTT 功能的实用函数。此类还包含通过 MQTT 接收的命令的主要命令处理器：

```cpp
#include <user_config.h>
#include <SmingCore/SmingCore.h>
```

这两个包含是使用 Sming 框架所必需的。通过它们，我们包含了 SDK 的主要头文件（`user_config.h`）和 Sming 的头文件（`SmingCore.h`）。这也定义了多个预处理器语句，例如使用开源 **轻量级 IP 堆栈**（**LWIP**）以及处理官方 SDK 中的某些问题。

值得注意的是 `esp_cplusplus.h` 头文件，它以这种方式间接包含。其源文件实现了 `new` 和 `delete` 函数，以及一些与类相关功能的处理程序，例如使用虚拟类时的 `vtables`。这使它与 STL 兼容：

```cpp
enum {
          LOG_ERROR = 0,
          LOG_WARNING,
          LOG_INFO,
          LOG_DEBUG,
          LOG_TRACE,
          LOG_XTRACE
 };

 enum ESP8266_pins {
          ESP8266_gpio00 = 0x00001,     // Flash
          ESP8266_gpio01 = 0x00002,     // TXD 0
          ESP8266_gpio02 = 0x00004,     // TXD 1
          ESP8266_gpio03 = 0x00008,     // RXD 0
          ESP8266_gpio04 = 0x00010,     // 
          ESP8266_gpio05 = 0x00020,     // 
          ESP8266_gpio09 = 0x00040,     // SDD2 (QDIO Flash)
          ESP8266_gpio10 = 0x00080,     // SDD3 (QDIO Flash)
          ESP8266_gpio12 = 0x00100,     // HMISO (SDO)
          ESP8266_gpio13 = 0x00200,     // HMOSI (SDI)
          ESP8266_gpio14 = 0x00400,     // SCK
          ESP8266_gpio15 = 0x00800,     // HCS
          ESP8266_gpio16 = 0x01000,     // User, Wake
          ESP8266_mosi = 0x02000,
          ESP8266_miso = 0x04000,
          ESP8266_sclk = 0x08000,
          ESP8266_cs = 0x10000
 };
```

这两个枚举定义了日志级别，以及我们可能想要使用的 ESP8266 的单个 GPIO 和其他引脚。ESP8266 引脚枚举的值对应于位掩码中的位置：

```cpp
#define SCL_PIN 5
#define SDA_PIN 4
```

在这里，我们定义了 I2C 总线的固定引脚。这些对应于 GPIO 4 和 5，也称为 NodeMCU 板上的 **D1** 和 **D2**。这些引脚预定义的主要原因是因为它们是 ESP8266 上为数不多的 **安全** 引脚之一。

ESP8266 的许多引脚在启动过程中会改变电平，然后在稳定之前，这可能会对任何连接的外设造成不希望的行为。

```cpp
typedef void (*topicCallback)(String);
typedef void (*onInitCallback)();
```

我们定义了两个函数指针，一个用于功能模块在它们希望注册 MQTT 主题及其回调函数时使用。另一个是我们看到的主函数中的回调函数。

```cpp

class OtaCore {
         static Timer procTimer;
         static rBootHttpUpdate* otaUpdater;
         static MqttClient* mqtt;
         static String MAC;
         static HashMap<String, topicCallback>* topicCallbacks;
         static HardwareSerial Serial1;
         static String location;
         static String version;
         static int sclPin;
         static int sdaPin;
         static bool i2c_active;
         static bool spi_active;
         static uint32 esp8266_pins;

         static void otaUpdate();
         static void otaUpdate_CallBack(rBootHttpUpdate& update, bool result);
         static void startMqttClient();
         static void checkMQTTDisconnect(TcpClient& client, bool flag);
         static void connectOk(IPAddress ip, IPAddress mask, IPAddress gateway);
         static void connectFail(String ssid, uint8_t ssidLength, uint8_t *bssid,    uint8_t reason);
         static void onMqttReceived(String topic, String message);
         static void updateModules(uint32 input);
         static bool mapGpioToBit(int pin, ESP8266_pins &addr);

public:
         static bool init(onInitCallback cb);
         static bool registerTopic(String topic, topicCallback cb);
         static bool deregisterTopic(String topic);
         static bool publish(String topic, String message, int qos = 1);
         static void log(int level, String msg);
         static String getMAC() { return OtaCore::MAC; }
         static String getLocation() { return OtaCore::location; }
         static bool starti2c();
         static bool startSPI();
         static bool claimPin(ESP8266_pins pin);
         static bool claimPin(int pin);
         static bool releasePin(ESP8266_pins pin);
         static bool releasePin(int pin);
};
```

类声明本身提供了对这个类提供的功能的好概述。我们首先注意到的是，它是完全静态的。这确保了当固件启动时，这个类的功能立即初始化，并且可以全局访问，而无需担心特定实例。

我们还可以看到 `uint32` 类型的首次使用，它与其他整数类型一样，在 `cstdint` 头文件中定义。

接下来，这里是实现方式：

```cpp
#include <ota_core.h>

#include "base_module.h"

#define SPI_SCLK 14
#define SPI_MOSI 13
#define SPI_MISO 12
#define SPI_CS 15

Timer OtaCore::procTimer;
rBootHttpUpdate* OtaCore::otaUpdater = 0;
MqttClient* OtaCore::mqtt = 0;
String OtaCore::MAC;
HashMap<String, topicCallback>* OtaCore::topicCallbacks = new HashMap<String, topicCallback>();
HardwareSerial OtaCore::Serial1(UART_ID_1); // UART 0 is 'Serial'.
String OtaCore::location;
String OtaCore::version = VERSION;
int OtaCore::sclPin = SCL_PIN; // default.
int OtaCore::sdaPin = SDA_PIN; // default.
bool OtaCore::i2c_active = false;
bool OtaCore::spi_active = false;
uint32 OtaCore::esp8266_pins = 0x0;
```

我们在这里包含 `BaseModule` 类的头部，这样我们就可以在我们完成基本功能设置之后调用其自己的初始化函数。静态类成员也在这里初始化，在相关位置分配了多个默认值。

值得注意的是，除了默认的 Serial 对象实例之外，还初始化了一个第二个串行接口对象。这些对应于 ESP8266 上的第一个（UART0，Serial）和第二个（UART1，Serial1）UART。

在 Sming 的较旧版本中，与 SPIFFS 相关的文件函数在处理二进制数据时存在问题（由于内部假设字符串以空字符终止），这就是为什么添加了以下替代函数的原因。它们的命名是原始函数名的轻微倒置版本，以防止命名冲突。

由于 TLS 证书和其他存储在 SPIFFS 上的二进制数据文件必须能够被写入和读取，以便固件能够正确运行，因此这是一个必要的折衷方案。

```cpp
String getFileContent(const String fileName) {
         file_t file = fileOpen(fileName.c_str(), eFO_ReadOnly);

         fileSeek(file, 0, eSO_FileEnd);
         int size = fileTell(file);
         if (size <= 0)    {
                fileClose(file);
                return "";
         }

         fileSeek(file, 0, eSO_FileStart);
         char* buffer = new char[size + 1];
         buffer[size] = 0;
         fileRead(file, buffer, size);
         fileClose(file);
         String res(buffer, size);
         delete[] buffer;
         return res;
}
```

此函数将指定文件的全部内容读取到一个返回的 `String` 实例中。

```cpp
void setFileContent(const String &fileName, const String &content) {
          file_t file = fileOpen(fileName.c_str(),                                                   eFO_CreateNewAlways | eFO_WriteOnly);
          fileWrite(file, content.c_str(), content.length());
          fileClose(file);
 }
```

此函数用提供的 `String` 实例中的新数据替换文件中的现有内容。

```cpp
bool readIntoFileBuffer(const String filename, char* &buffer, unsigned int &size) {
         file_t file = fileOpen(filename.c_str(), eFO_ReadOnly);

         fileSeek(file, 0, eSO_FileEnd);
         size = fileTell(file);
         if (size == 0)    {
                fileClose(file);
                return true;
         }

         fileSeek(file, 0, eSO_FileStart);
         buffer = new char[size + 1];
         buffer[size] = 0;
         fileRead(file, buffer, size);
         fileClose(file);
         return true;
}
```

此函数与 `getFileContent()` 类似，但返回一个简单的字符缓冲区而不是 `String` 实例。它主要用于读取证书数据，这些数据被传递到一个基于 C 的 TLS 库（称为 axTLS）中，在那里将转换为 `String` 实例会因涉及复制而造成浪费，尤其是当证书大小可能达到几个 KB 时。

接下来是这个类的初始化函数：

```cpp
bool OtaCore::init(onInitCallback cb) {
         Serial.begin(9600);

         Serial1.begin(SERIAL_BAUD_RATE); 
         Serial1.systemDebugOutput(true);
```

我们首先在 NodeMCU 中初始化两个 UART（串行接口）。尽管官方 ESP8266 中有两个 UART，但第二个 UART 只包含一个 TX 输出线（默认为 GPIO 2）。因此，我们希望保留第一个 UART 以供需要完整串行线的应用程序使用，例如一些传感器。

因此，第一个 UART（`Serial`）被初始化，以便我们可以在以后使用它与功能模块一起使用，而第二个 UART（`Serial1`）被初始化为默认波特率 115,200，同时将系统的调试输出（WiFi/IP 堆栈等）也导向这个串行输出。因此，这个第二个串行接口将仅用于日志输出。

```cpp
         BaseModule::init(); 
```

接下来，初始化了 `BaseModule` 静态类。这会导致在此固件中激活的所有功能模块被注册，允许它们稍后激活。

```cpp
         int slot = rboot_get_current_rom();
         u32_t offset;
         if (slot == 0) { offset = 0x100000; }
         else { offset = 0x300000; }
         spiffs_mount_manual(offset, 65536);
```

在使用 rBoot 启动加载程序的同时自动挂载 SPIFFS 文件系统在 Sming 的较旧版本中不起作用，这就是为什么我们在这里手动进行此操作的原因。为此，我们获取 rBoot 的当前固件槽，使用它可以选择适当的偏移量，要么在 ROM 的第二个兆字节的开始处，要么在第四个兆字节处。

确定偏移量后，我们使用带有我们的偏移量和 SPIFFS 部分大小的 SPIFFS 手动挂载功能。我们现在能够读取和写入我们的存储。

```cpp

          Serial1.printf("\r\nSDK: v%s\r\n", system_get_sdk_version());
     Serial1.printf("Free Heap: %d\r\n", system_get_free_heap_size());
     Serial1.printf("CPU Frequency: %d MHz\r\n", system_get_cpu_freq());
     Serial1.printf("System Chip ID: %x\r\n", system_get_chip_id());
     Serial1.printf("SPI Flash ID: %x\r\n", spi_flash_get_id());
```

接下来，我们将一些系统详细信息打印到串行调试输出中。这包括我们针对编译的 ESP8266 SDK 版本、当前空闲堆大小、CPU 频率、MCU ID（32 位 ID）以及 SPI ROM 芯片的 ID。

```cpp
         mqtt = new MqttClient(MQTT_HOST, MQTT_PORT, onMqttReceived);
```

我们在堆上创建一个新的 MQTT 客户端，提供当接收到新消息时将被调用的回调函数。MQTT 代理的主机和端口由预处理器根据我们在用户 Makefile 中为项目添加的详细信息填充。

```cpp

         Serial1.printf("\r\nCurrently running rom %d.\r\n", slot);

         WifiStation.enable(true);
         WifiStation.config(WIFI_SSID, WIFI_PWD);
         WifiStation.connect();
         WifiAccessPoint.enable(false);

        WifiEvents.onStationGotIP(OtaCore::connectOk);
        WifiEvents.onStationDisconnect(OtaCore::connectFail);

          (*cb)();
}
```

在初始化的最后几个步骤中，我们输出当前正在运行的固件槽，然后启用 Wi-Fi 客户端，同时禁用**无线接入点**（**WAP**）功能。Wi-Fi 客户端被告知连接到我们在 Makefile 中之前指定的 Wi-Fi SSID。

最后，我们在调用作为参数提供的回调函数之前，定义了成功 Wi-Fi 连接和失败连接尝试的处理程序。

在固件 OTA 更新后，以下回调函数将被调用：

```cpp

void OtaCore::otaUpdate_CallBack(rBootHttpUpdate& update, bool result) {
         OtaCore::log(LOG_INFO, "In OTA callback...");
         if (result == true) { // success
               uint8 slot = rboot_get_current_rom();
               if (slot == 0) { slot = 1; } else { slot = 0; }

               Serial1.printf("Firmware updated, rebooting to ROM slot %d...\r\n",                                                                                                                        slot);
               OtaCore::log(LOG_INFO, "Firmware updated, restarting...");
               rboot_set_current_rom(slot);
               System.restart();
         } 
         else {
               OtaCore::log(LOG_ERROR, "Firmware update failed.");
         }
}
```

在此回调中，如果 OTA 更新成功，我们更改活动 ROM 槽，然后重启系统。否则，我们简单地记录错误，并不重启。

接下来是一些与 MQTT 相关的函数：

```cpp
bool OtaCore::registerTopic(String topic, topicCallback cb) {
         OtaCore::mqtt->subscribe(topic);
         (*topicCallbacks)[topic] = cb;
         return true;
}

bool OtaCore::deregisterTopic(String topic) {
         OtaCore::mqtt->unsubscribe(topic);
         if (topicCallbacks->contains(topic)) {
               topicCallbacks->remove(topic);
         }

         return true;
}
```

这两个函数允许功能模块分别注册和注销一个 MQTT 主题及其回调。MQTT 代理使用订阅或取消订阅请求进行调用，并相应地更新`HashMap`实例：

```cpp
bool OtaCore::publish(String topic, String message, int qos /* = 1 */) {
         OtaCore::mqtt->publishWithQoS(topic, message, qos);
         return true;
}
```

任何功能模块都可以使用此函数在任何主题上发布 MQTT 消息。**服务质量**（**QoS**）参数确定发布模式。默认情况下，消息以*保留*模式发布，这意味着代理将保留特定主题的最后一个发布消息。

OTA 更新功能的入口点位于以下函数中：

```cpp
void OtaCore::otaUpdate() {
         OtaCore::log(LOG_INFO, "Updating firmware from URL: " + String(OTA_URL));

         if (otaUpdater) { delete otaUpdater; }
         otaUpdater = new rBootHttpUpdate();

         rboot_config bootconf = rboot_get_config();
         uint8 slot = bootconf.current_rom;
         if (slot == 0) { slot = 1; } else { slot = 0; }

         otaUpdater->addItem(bootconf.roms[slot], OTA_URL + MAC);

         otaUpdater->setCallback(OtaCore::otaUpdate_CallBack);
         otaUpdater->start();
}
```

对于 OTA 更新，我们需要创建一个干净的`rBootHttpUpdate`实例。然后我们需要使用从 rBoot 获得的配置以及当前的固件槽号来配置此实例，这个槽号用于将其他固件槽的编号提供给 OTA 更新器。

在这里，我们只配置它更新固件槽，但也可以通过这种方式更新其他固件槽的 SPIFFS 部分。固件将通过我们之前设置的固定 URL 通过 HTTP 获取。ESP8266 的 MAC 地址附加到末尾作为唯一的查询字符串参数，以便更新服务器知道哪个固件映像适合此系统。

在设置我们之前查看的`callback`函数之后，我们开始更新：

```cpp
void OtaCore::checkMQTTDisconnect(TcpClient& client, bool flag) {
         if (flag == true) { Serial1.println("MQTT Broker disconnected."); }
         else { 
               String tHost = MQTT_HOST;
               Serial1.println("MQTT Broker " + tHost + " unreachable."); }

         procTimer.initializeMs(2 * 1000, OtaCore::startMqttClient).start();
}
```

在这里，我们定义 MQTT 断开连接处理程序。每当与 MQTT 代理的连接失败时，它都会被调用，这样我们就可以在两秒延迟后尝试重新连接。

如果我们之前已经连接，则将标志参数设置为 true，如果初始 MQTT 代理连接失败（无网络访问、地址错误等），则设置为 false。

接下来是配置和启动 MQTT 客户端的函数：

```cpp
void OtaCore::startMqttClient() {
         procTimer.stop();
         if (!mqtt->setWill("last/will",                                 "The connection from this device is lost:(",    1, true)) {
               debugf("Unable to set the last will and testament. Most probably there is not enough memory on the device.");
         }
```

如果我们是从重连计时器中被调用，我们将停止 procTimer 计时器。接下来，我们为该设备设置**遗嘱和遗言**（**LWT**），这样我们就可以设置一个消息，当 MQTT 代理与客户端（我们）断开连接时，MQTT 代理将发布该消息。

接下来，我们定义了三条不同的执行路径，其中只有一条会被编译，具体取决于我们是否使用 TLS（SSL）、用户名/密码登录或匿名访问：

```cpp
#ifdef ENABLE_SSL
         mqtt->connect(MAC, MQTT_USERNAME, MQTT_PWD, true);
         mqtt->addSslOptions(SSL_SERVER_VERIFY_LATER);

       Serial1.printf("Free Heap: %d\r\n", system_get_free_heap_size());

         if (!fileExist("esp8266.client.crt.binary")) {
               Serial1.println("SSL CRT file is missing: esp8266.client.crt.binary.");
               return;
         }
         else if (!fileExist("esp8266.client.key.binary")) {
               Serial1.println("SSL key file is missing: esp8266.client.key.binary.");
               return;
         }

         unsigned int crtLength, keyLength;
         char* crtFile;
         char* keyFile;
         readIntoFileBuffer("esp8266.client.crt.binary", crtFile, crtLength);
         readIntoFileBuffer("esp8266.client.key.binary", keyFile, keyLength);

         Serial1.printf("keyLength: %d, crtLength: %d.\n", keyLength, crtLength);
         Serial1.printf("Free Heap: %d\r\n", system_get_free_heap_size());

         if (crtLength < 1 || keyLength < 1) {
               Serial1.println("Failed to open certificate and/or key file.");
               return;
         }

         mqtt->setSslClientKeyCert((const uint8_t*) keyFile, keyLength,
                                                (const uint8_t*) crtFile, crtLength, 0, true);
         delete[] keyFile;
         delete[] crtFile;

    Serial1.printf("Free Heap: %d\r\n", system_get_free_heap_size());
```

如果我们使用 TLS 证书，我们使用我们的 `MAC` 作为客户端标识符与 MQTT 代理建立连接，然后启用连接的 SSL 选项。为了调试目的，将可用的堆空间打印到串行日志输出。通常，在这个时候，我们应该剩下大约 25 KB 的 RAM，这对于在内存中存储证书和密钥以及 TLS 握手的 RX 和 TX 缓冲区（如果 SSL 端点配置为使用 SSL 片段大小选项的可接受大小）是足够的。我们将在第九章示例 - 建筑管理和控制中更详细地探讨这一点。

接下来，我们从 SPIFFS 读取 DER 编码（二进制）的证书和密钥文件。这些文件具有固定的名称。对于每个文件，我们打印出文件大小，以及当前的空闲堆大小。如果任一文件大小为零字节，我们认为读取尝试失败，并终止连接尝试。

否则，我们使用密钥和证书数据与 MQTT 连接，这应该会导致成功的握手并建立与 MQTT 代理的加密连接。

在删除密钥和证书文件数据后，我们打印出空闲堆大小，以便我们可以检查清理是否成功：

```cpp
#elif defined USE_MQTT_PASSWORD
          mqtt->connect(MAC, MQTT_USERNAME, MQTT_PWD);
```

当使用 MQTT 用户名和密码登录到代理时，我们只需在 MQTT 客户端实例上调用前面的函数，提供我们的 MAC 作为客户端标识符，以及用户名和密码：

```cpp
#else
         mqtt->connect(MAC);
#endif
```

要匿名连接，我们设置与代理的连接，并将我们的 `MAC` 作为客户端标识符传递：

```cpp
         mqtt->setCompleteDelegate(checkMQTTDisconnect);

         mqtt->subscribe(MQTT_PREFIX"upgrade");
         mqtt->subscribe(MQTT_PREFIX"presence/tell");
         mqtt->subscribe(MQTT_PREFIX"presence/ping");
         mqtt->subscribe(MQTT_PREFIX"presence/restart/#");
         mqtt->subscribe(MQTT_PREFIX"cc/" + MAC);

         delay(100);

         mqtt->publish(MQTT_PREFIX"cc/config", MAC);
}
```

在这里，我们首先设置 MQTT 断开连接处理程序。然后，我们订阅我们希望响应的一些主题。这些都与该固件的管理系统功能相关，允许通过 MQTT 查询和配置系统。

在订阅后，我们短暂地（100 毫秒）等待，给代理一些时间处理这些订阅，然后在我们发布到中心通知主题之前，使用我们的 `MAC` 让任何感兴趣的客户端和服务器知道该系统刚刚上线。

接下来是 WiFi 连接处理程序：

```cpp
void OtaCore::connectOk(IPAddress ip, IPAddress mask, IPAddress gateway) {
          Serial1.println("I'm CONNECTED. IP: " + ip.toString());

          MAC = WifiStation.getMAC();
          Serial1.printf("MAC: %s.\n", MAC.c_str());

          if (fileExist("location.txt")) {
                location = getFileContent("location.txt");
          }
          else {
                location = MAC;
          }

          if (fileExist("config.txt")) {
                String configStr = getFileContent("config.txt");
                uint32 config;
                configStr.getBytes((unsigned char*) &config, sizeof(uint32), 0);
                updateModules(config);
          }

          startMqttClient();
 }
```

当我们使用提供的凭据成功连接到配置的 WiFi 网络时，将调用此处理程序。连接后，我们在内存中保留我们的`MAC`副本作为我们的唯一 ID。

此固件还支持指定一个用户定义的字符串作为我们的位置或类似标识符。如果之前已经定义了一个，我们就从 SPIFFS 加载它并使用它；否则，我们的位置字符串简单地是`MAC`。

类似地，如果存在，我们从 SPIFFS 加载定义功能模块配置的 32 位位掩码。如果没有，所有功能模块最初都保持未激活状态。否则，我们读取位掩码并将其传递给`updateModules()`函数，以便相关的模块被激活：

```cpp
void OtaCore::connectFail(String ssid, uint8_t ssidLength, 
                                                   uint8_t* bssid, uint8_t reason) {
          Serial1.println("I'm NOT CONNECTED. Need help :(");
          debugf("Disconnected from %s. Reason: %d", ssid.c_str(), reason);

          WDT.alive();

          WifiEvents.onStationGotIP(OtaCore::connectOk);
          WifiEvents.onStationDisconnect(OtaCore::connectFail);
 }
```

如果连接到 Wi-Fi 网络失败，我们将记录这一事实，然后告诉 MCU 的看门狗定时器我们仍然存活，以防止在我们再次尝试连接之前发生软重启。

这完成了所有初始化函数。接下来是正常活动期间使用的函数，首先是 MQTT 消息处理程序：

```cpp
void OtaCore::onMqttReceived(String topic, String message) {
         Serial1.print(topic);
         Serial1.print(":\n");
         Serial1.println(message);

         log(LOG_DEBUG, topic + " - " + message);

         if (topic == MQTT_PREFIX"upgrade" && message == MAC) {
                otaUpdate();
         }
         else if (topic == MQTT_PREFIX"presence/tell") {
                mqtt->publish(MQTT_PREFIX"presence/response", MAC);
         }
         else if (topic == MQTT_PREFIX"presence/ping") {
               mqtt->publish(MQTT_PREFIX"presence/pong", MAC);
         }
         else if (topic == MQTT_PREFIX"presence/restart" && message == MAC) {
               System.restart();
         }
         else if (topic == MQTT_PREFIX"presence/restart/all") {
               System.restart();
         }
```

我们在最初创建 MQTT 客户端实例时注册了这个回调。每次我们订阅的主题在代理上收到新消息时，我们都会收到通知，并且这个回调将接收到一个包含主题的字符串和另一个包含实际消息（有效载荷）的字符串。

我们可以将该主题与已注册的主题进行比较，并执行所需的操作，无论是执行 OTA 更新（如果指定了我们的`MAC`），通过返回包含我们的`MAC`的 pong 响应来响应 ping 请求，还是重启系统。

下一个主题是一个更通用的维护主题，允许配置活动功能模块，设置位置字符串，并请求系统的当前状态。有效载荷格式由命令字符串后跟一个分号组成，然后是有效载荷字符串：

```cpp
   else if (topic == MQTT_PREFIX"cc/" + MAC) {
          int chAt = message.indexOf(';');
          String cmd = message.substring(0, chAt);
          ++chAt;

          String msg(((char*) &message[chAt]), (message.length() - chAt));

          log(LOG_DEBUG, msg);

          Serial1.printf("Command: %s, Message: ", cmd.c_str());
          Serial1.println(msg);
```

我们首先使用简单的查找和子字符串方法从有效载荷字符串中提取命令。然后我们读取剩余的有效载荷字符串，注意将其作为二进制字符串读取。为此，我们使用剩余字符串的长度作为起始位置，分号后的字符作为起始位置。

到目前为止，我们已经提取了命令和有效载荷，可以看到我们必须做什么：

```cpp

         if (cmd == "mod") {
               if (msg.length() != 4) {
                     Serial1.printf("Payload size wasn't 4 bytes: %d\n", msg.length());
                     return; 
               }

               uint32 input;
               msg.getBytes((unsigned char*) &input, sizeof(uint32), 0);
               String byteStr;
               byteStr = "Received new configuration: ";
               byteStr += input;
               log(LOG_DEBUG, byteStr);
               updateModules(input);               
          }
```

此命令设置哪些功能模块应该处于活动状态。其有效载荷应该是一个无符号 32 位整数，形成一个位掩码，我们检查以确保我们收到了正好四个字节。

在位掩码中，每个位都与一个模块相对应，此时这些模块如下：

| **位位置** | **值** |
| --- | --- |
| 0x01 | THPModule |
| 0x02 | CO2Module |
| 0x04 | JuraModule |
| 0x08 | JuraTermModule |
| 0x10 | MotionModule |
| 0x20 | PwmModule |
| 0x40 | IOModule |
| 0x80 | SwitchModule |
| 0x100 | PlantModule |

其中，CO2、Jura 和 JuraTerm 模块是互斥的，因为它们都使用第一个 UART（`Serial`）。如果在位掩码中仍然指定了两个或更多这些模块，则只有第一个模块将被启用，其他模块将被忽略。我们将在第九章中更详细地查看这些其他功能模块，*示例 - 建筑管理与控制*。

在我们获得新的配置位掩码后，我们将其发送到`updateModules()`函数：

```cpp
        else if (cmd == "loc") {
               if (msg.length() < 1) { return; }
               if (location != msg) {
                     location = msg;
                     fileSetContent("location.txt", location);
               }
         }
```

使用此命令，我们设置新的位置字符串，如果它与当前的不同，也会将其保存到 SPIFFS 中的位置文件中，以便在重启后持久化：

```cpp
         else if (cmd == "mod_active") {
               uint32 active_mods = BaseModule::activeMods();
               if (active_mods == 0) {
                     mqtt->publish(MQTT_PREFIX"cc/response", MAC + ";0");
                     return;
               }

               mqtt->publish(MQTT_PREFIX"cc/response", MAC + ";"                                                         + String((const char*) &active_mods, 4));
         }
         else if (cmd == "version") {
               mqtt->publish(MQTT_PREFIX"cc/response", MAC + ";" + version);
         }
         else if (cmd == "upgrade") {
               otaUpdate();
         }
   }
```

本节最后三个命令返回活动功能模块的当前位掩码、固件版本，并触发 OTA 升级：

```cpp
         else {
               if (topicCallbacks->contains(topic)) {
                     (*((*topicCallbacks)[topic]))(message);
                }
         }
}
```

`if...else`块中的最后一个条目检查主题是否在我们的功能模块回调列表中。如果找到，则使用 MQTT 消息字符串调用回调。

自然，这意味着只有一个功能模块可以注册到特定的主题。由于每个模块通常在自己的 MQTT 子主题下运行以隔离消息流，这通常不是问题：

```cpp
void OtaCore::updateModules(uint32 input) {
         Serial1.printf("Input: %x, Active: %x.\n", input, BaseModule::activeMods());

         BaseModule::newConfig(input);

         if (BaseModule::activeMods() != input) {
               String content(((char*) &input), 4);
               setFileContent("config.txt", content);
         }
}
```

此函数相当简单。它主要作为`BaseModule`类的中继，但它还确保我们保持 SPIFFS 中的配置文件是最新的，当它发生变化时，将其写入新位掩码。

我们绝对必须防止对 SPIFFs 进行不必要的写入，因为底层 Flash 存储器具有有限的写入周期。限制写入周期可以显著延长硬件的使用寿命，同时减少整体系统负载：

```cpp
bool OtaCore::mapGpioToBit(int pin, ESP8266_pins &addr) {
          switch (pin) {
                case 0:
                      addr = ESP8266_gpio00;
                      break;
                case 1:
                      addr = ESP8266_gpio01;
                      break;
                case 2:
                      addr = ESP8266_gpio02;
                      break;
                case 3:
                      addr = ESP8266_gpio03;
                      break;
                case 4:
                      addr = ESP8266_gpio04;
                      break;
                case 5:
                      addr = ESP8266_gpio05;
                      break;
                case 9:
                      addr = ESP8266_gpio09;
                      break;
                case 10:
                      addr = ESP8266_gpio10;
                      break;
                case 12:
                      addr = ESP8266_gpio12;
                      break;
                case 13:
                      addr = ESP8266_gpio13;
                      break;
                case 14:
                      addr = ESP8266_gpio14;
                      break;
                case 15:
                      addr = ESP8266_gpio15;
                      break;
                case 16:
                      addr = ESP8266_gpio16;
                      break;
                default:
                      log(LOG_ERROR, "Invalid pin number specified: " + String(pin));
                      return false;
          };

          return true;
 }
```

此函数将给定的 GPIO 引脚号映射到内部位掩码中的位置。它使用我们查看的该类头文件的枚举。通过这种映射，我们可以使用单个 uint32 值设置 ESP8266 模块的 GPIO 引脚的已用/未用状态：

```cpp
void OtaCore::log(int level, String msg) {
         String out(lvl);
         out += " - " + msg;

         Serial1.println(out);
         mqtt->publish(MQTT_PREFIX"log/all", OtaCore::MAC + ";" + out);
}
```

在日志记录方法中，我们在将消息写入串行输出之前，将日志级别追加到消息字符串中，以及将其发布在 MQTT 上。在这里，我们发布在一个单一的主题上，但作为一个改进，你可以根据指定的级别在不同的主题上记录日志。

这里有什么意义在很大程度上取决于你为监听和处理运行此固件的 ESP8266 系统日志输出而设置的后端类型：

```cpp
bool OtaCore::starti2c() {
         if (i2c_active) { return true; }

         if (!claimPin(sdaPin)) { return false; }
         if (!claimPin(sclPin)) { return false; }

         Wire.pins(sdaPin, sclPin);
         pinMode(sclPin, OUTPUT);
         for (int i = 0; i < 8; ++i) {
               digitalWrite(sclPin, HIGH);
               delayMicroseconds(3);
               digitalWrite(sclPin, LOW);
               delayMicroseconds(3);
         }

         pinMode(sclPin, INPUT);

         Wire.begin();
         i2c_active = true;
}
```

此函数如果尚未启动，则启动 I2C 总线。它尝试注册它希望用于 I2C 总线的引脚。如果这些引脚可用，它将时钟线（SCL）设置为输出模式，并首先脉冲它八次以解冻总线上的任何 I2C 设备。

在像他那样脉冲时钟线之后，我们在引脚上启动 I2C 总线，并记录下该总线的激活状态。

如果 MCU 在 I2C 设备没有时循环电源，并且保持不确定状态，则可能会出现冻结的 I2C 设备。通过这种脉冲，我们确保系统不会最终处于非功能状态，需要手动干预：

```cpp
bool OtaCore::startSPI() {
    if (spi_active) { return true; }

    if (!claimPin(SPI_SCLK)) { return false; }
    if (!claimPin(SPI_MOSI)) { return false; }
    if (!claimPin(SPI_MISO)) { return false; }
    if (!claimPin(SPI_CS)) { return false; }

    SPI.begin();
    spi_active = true;
 }
```

启动 SPI 总线与启动 I2C 总线类似，只是没有类似的恢复机制：

```cpp
bool OtaCore::claimPin(int pin) {
          ESP8266_pins addr;
          if (!mapGpioToBit(pin, addr)) { return false; }

          return claimPin(addr);
    }

    bool OtaCore::claimPin(ESP8266_pins pin) {
          if (esp8266_pins & pin) {
                log(LOG_ERROR, "Attempting to claim an already claimed pin: "                                                                                                      + String(pin));
                log(LOG_DEBUG, String("Current claimed pins: ") + String(esp8266_pins));
                return false;
          }

          log(LOG_INFO, "Claiming pin position: " + String(pin));

          esp8266_pins |= pin;

          log(LOG_DEBUG, String("Claimed pin configuration: ") + String(esp8266_pins));

          return true;
 }
```

这个重载函数用于在功能模块启动之前通过功能模块注册 GPIO 引脚，以确保没有两个模块尝试同时使用相同的引脚。一个版本接受一个引脚号（GPIO），并使用我们之前查看的映射函数在`esp8266_pins`位掩码中获取位地址，然后再将其传递给函数的另一个版本。

在那个函数中，使用引脚枚举进行位运算的`AND`比较。如果位尚未设置，则将其切换并返回 true。否则，函数返回 false，调用模块知道它不能继续其初始化：

```cpp
bool OtaCore::releasePin(int pin) {
          ESP8266_pins addr;
          if (!mapGpioToBit(pin, addr)) { return false; }

          return releasePin(addr);
    }

    bool OtaCore::releasePin(ESP8266_pins pin) {
          if (!(esp8266_pins & pin)) {
                log(LOG_ERROR, "Attempting to release a pin which has not been set: "                                                                                                      + String(pin));
                return false;
          }

          esp8266_pins &= ~pin;

          log(LOG_INFO, "Released pin position: " + String(pin));
          log(LOG_DEBUG, String("Claimed pin configuration: ") + String(esp8266_pins));

          return true;
 }
```

这个重载函数，在功能模块关闭时释放引脚，以类似的方式工作。一个使用映射函数获取位地址，另一个执行位运算的`AND`操作以检查引脚是否确实已设置，如果已设置，则使用位运算的`OR`赋值运算符将其切换到关闭位置。

# BaseModule

这个类包含注册和跟踪当前哪些功能模块是活动或非活动的逻辑。其头文件如下所示：

```cpp
#include "ota_core.h"

enum ModuleIndex {
   MOD_IDX_TEMPERATURE_HUMIDITY = 0,
   MOD_IDX_CO2,
   MOD_IDX_JURA,
   MOD_IDX_JURATERM,
   MOD_IDX_MOTION,
   MOD_IDX_PWM,
   MOD_IDX_IO,
   MOD_IDX_SWITCH,
   MOD_IDX_PLANT
};

typedef bool (*modStart)();
typedef bool (*modShutdown)();
```

包含`OtaCore`头文件是为了让我们能够使用日志功能。除此之外，我们创建另一个枚举，它将特定的功能模块映射到功能模块位掩码（`active_mods`）中的特定位。

最后，定义了函数指针，它们分别用于启动和关闭功能模块。这些将由功能模块在它们注册时定义：

```cpp
#include "thp_module.h"
#include "jura_module.h"
#include "juraterm_module.h"
#include "co2_module.h"
#include "motion_module.h"
#include "pwm_module.h"
#include "io_module.h"
#include "switch_module.h"
#include "plant_module.h"
```

这些是目前为这个固件存在的功能模块。由于我们只需要这个项目中的植物模块，我们可以注释掉其他模块的所有头文件，以及这个类初始化函数中的它们的初始化。

这将不会以任何方式影响最终生成的固件镜像，除了我们无法启用那些模块，因为它们不存在。

最后，这里是类声明本身：

```cpp
class BaseModule {   
         struct SubModule {
               modStart start;
               modShutdown shutdown;
               ModuleIndex index;
               uint32 bitmask;
               bool started;
         };

         static SubModule modules[32];
         static uint32 active_mods;
         static bool initialized;
         static uint8 modcount;

public:
         static void init();
         static bool registerModule(ModuleIndex index, modStart start,                                                                                    modShutdown shutdown);
```

```cpp

         static bool newConfig(uint32 config);
         static uint32 activeMods() { return active_mods; }
};
```

每个功能模块在内部都由一个`SubModule`实例表示，其细节我们可以在类定义中稍后看到：

```cpp
#include "base_module.h"

BaseModule::SubModule BaseModule::modules[32];
uint32 BaseModule::active_mods = 0x0;
bool BaseModule::initialized = false;
uint8 BaseModule::modcount = 0;
```

由于这是一个静态类，我们首先初始化其类变量。我们有一个可以容纳 32 个`SubModule`实例的数组，以适应完整的位掩码。在此之外，没有模块是活动的，所以一切都被初始化为零和 false：

```cpp
void BaseModule::init() {
    CO2Module::initialize();
    IOModule::initialize();
    JuraModule::initialize();
    JuraTermModule::initialize();
    MotionModule::initialize();
    PlantModule::initialize();
    PwmModule::initialize();
    SwitchModule::initialize();
    THPModule::initialize();
}
```

当我们在`OtaCore`中调用此函数时，我们还触发了在此定义的功能模块的注册。通过在此函数中选择性删除或注释模块，我们可以从最终的固件映像中移除它们。这里调用的模块将调用以下函数来注册自己：

```cpp
bool BaseModule::registerModule(ModuleIndex index, modStart start, modShutdown shutdown) {
         if (!initialized) {
               for (uint8 i = 0; i < 32; i++) {
                     modules[i].start = 0;
                     modules[i].shutdown = 0;
                     modules[i].index = index;
                     modules[i].bitmask = (1 << i);
                     modules[i].started = false;
               }

               initialized = true;
         }

         if (modules[index].start) {
               return false;
         }

         modules[index].start = start;
         modules[index].shutdown = shutdown;
         ++modcount;

         return true;
}
```

调用此函数的第一个功能模块将触发`SubModule`数组的初始化，将所有值设置为中性设置，同时为数组中的此位置创建位掩码，这允许我们更新`active_mods`位掩码，正如我们稍后将看到的。

在初始化数组后，我们检查数组中此位置是否已注册了模块。如果有，我们返回 false。否则，我们注册模块的启动和关闭函数指针，并在返回 true 之前增加活动模块计数：

```cpp
bool BaseModule::newConfig(uint32 config) {
    OtaCore::log(LOG_DEBUG, String("Mod count: ") + String(modcount));
    uint32 new_config = config ^ active_mods;
    if (new_config == 0x0) {
        OtaCore::log(LOG_INFO, "New configuration was 0x0\. No 
        change.");
        return true; 
    }
    OtaCore::log(LOG_INFO, "New configuration: " + new_config);
    for (uint8 i = 0; i < 32; ++i) {
        if (new_config & (1 << i)) {
            OtaCore::log(LOG_DEBUG, String("Toggling module: ") + 
            String(i));
            if (modules[i].started) { 
                if ((modules[i]).shutdown()) { 
                    modules[i].started = false; 
                    active_mods ^= modules[i].bitmask;
                }
                else { 
                    OtaCore::log(LOG_ERROR, "Failed to shutdown 
                    module.");
                    return false; 
                }
            }
            else { 
                if ((modules[i].start) && (modules[i]).start()) { 
                    modules[i].started = true;
                    active_mods |= modules[i].bitmask;
                }
                else { 
                    OtaCore::log(LOG_ERROR, "Failed to start module.");
                    return false;
                }
            }
        }
    }
    return true;
 }
```

此函数的输入参数是从`OtaCore`中的 MQTT 有效负载中提取的位掩码。在这里，我们使用位异或比较与活动模块位掩码，以获得表示要进行的更改的新位掩码。如果结果是零，我们知道它们是相同的，我们可以返回而不需要进一步的操作。

我们因此获得的`uint32`位掩码指示了哪些模块应该开启或关闭。为此，我们检查掩码中的每个位。如果它是`1`（AND 运算符返回非零值），我们检查数组中该位置的模块是否存在并且已经启动。

如果模块已经启动，我们尝试将其关闭。如果模块的`shutdown()`函数成功（返回 true），我们在`active_mods`位掩码中切换位以更新其状态。同样，如果模块尚未启动，该位置已注册了模块，我们尝试启动它，如果成功则更新活动模块。

我们检查是否已注册启动函数回调，以确保我们不会意外调用未正确注册的模块并导致系统崩溃。

# PlantModule

到目前为止，我们已经详细查看了解决方案背后的支持代码，这使得编写新模块变得容易，因为我们不需要自己完成所有维护工作。我们还没有看到的是实际的模块或与本章项目直接相关的代码。

在本节中，我们将查看拼图的最后部分，即`PlantModule`本身：

```cpp
#include "base_module.h"
#include <Libraries/APA102/apa102.h>

#define PLANT_GPIO_PIN 5
#define NUM_APA102 1

class PlantModule {
         static int pin;
         static Timer timer;
         static uint16 humidityTrigger;
         static String publishTopic;
         static HttpServer server;
         static APA102* LED;

         static void onRequest(HttpRequest& request, HttpResponse& response);

public:
         static bool initialize();
         static bool start();
         static bool shutdown();
         static void readSensor();
         static void commandCallback(String message);
};
```

在此类声明中值得注意的是 APA102 库头文件的包含。这是一个简单的库，允许我们将颜色和亮度数据写入 APA102 RGB（全光谱）LED，通过 SPI 总线。

我们还定义了我们希望用于触发蠕动泵（GPIO 5）的引脚以及连接的 APA102 LED 模块的数量（1）。如果您想串联多个 APA102 LED，只需更新定义以匹配数量即可。

接下来是类实现：

```cpp
#include "plant_module.h"

int PlantModule::pin = PLANT_GPIO_PIN;
Timer PlantModule::timer;
uint16 PlantModule::humidityTrigger = 530;
String PlantModule::publishTopic;
HttpServer PlantModule::server;
APA102* PlantModule::LED = 0;

enum {
         PLANT_SOIL_MOISTURE = 0x01,
         PLANT_SET_TRIGGER = 0x02,
         PLANT_TRIGGER = 0x04
};
```

在本节中，我们初始化静态类成员，设置 GPIO 引脚，并定义泵应触发的初始传感器值。此触发值应更新以匹配您自己的传感器校准结果。

最后，我们定义一个枚举，包含此模块可以通过 MQTT 发送的可能的命令：

```cpp
bool PlantModule::initialize() {
          BaseModule::registerModule(MOD_IDX_PLANT, PlantModule::start,                                                                                                                 PlantModule::shutdown);
}
```

这是 `BaseModule` 在启动时调用的初始化函数。正如我们所见，它使此模块以预设值注册自己，包括其启动和关闭回调：

```cpp
bool PlantModule::start() {
         OtaCore::log(LOG_INFO, "Plant Module starting...");

         if (!OtaCore::claimPin(pin)) { return false; }

         publishTopic = MQTT_PREFIX + "plant/response/" + OtaCore::getLocation();
         OtaCore::registerTopic(MQTT_PREFIX + String("plants/") +                                                             OtaCore::getLocation(), PlantModule::commandCallback);

         pinMode(pin, OUTPUT);

         server.listen(80);
         server.setDefaultHandler(PlantModule::onRequest);

         LED = new APA102(NUM_APA102);
         LED->setBrightness(15);
         LED->clear();
         LED->setAllPixel(0, 255, 0);
         LED->show();

         timer.initializeMs(60000, PlantModule::readSensor).start();
         return true;
}
```

当此模块启动时，我们尝试声明用于触发泵的引脚，以及注册一个 MQTT 主题的回调，以便我们可以使用命令处理程序回调接受命令。我们将在处理命令后响应的主题也在此定义。

设置输出引脚模式，然后启动端口 80 上的 HTTP 服务器，注册一个基本的客户端请求处理程序。接下来，我们创建一个新的 `APA102` 类实例，并使用它来获取连接的 LED，以大约一半的全亮度显示绿色。

最后，我们启动一个定时器，该定时器将每分钟触发读取连接的土壤传感器：

```cpp
bool PlantModule::shutdown() {
         if (!OtaCore::releasePin(pin)) { return false; }

         server.shutdown();

         if (LED) {
               delete LED;
               LED = 0;
         }

         OtaCore::deregisterTopic(MQTT_PREFIX + String("plants/") +                                                                                            OtaCore::getLocation());

         timer.stop();
         return true;
}
```

当关闭此模块时，我们释放之前注册的引脚，停止网络服务器，删除 RGB LED 类实例（并检查删除是否必要），注销我们的 MQTT 主题，并最终停止传感器定时器。

```cpp

void PlantModule::commandCallback(String message) {
         OtaCore::log(LOG_DEBUG, "Plant command: " + message);

         if (message.length() < 1) { return; }
         int index = 0;
         uint8 cmd = *((uint8*) &message[index++]);

         if (cmd == PLANT_SOIL_MOISTURE) {
               readSensor();
         }
         else if (cmd == PLANT_SET_TRIGGER) {               
                if (message.length() != 3) { return; }
               uint16 payload = *((uint16*) &message[index]);
               index += 2;

               humidityTrigger = payload;
         }
         else if (cmd == PLANT_TRIGGER) {
               OtaCore::publish(publishTopic, OtaCore::getLocation() + ";" 
                                                                + String(((char*) &humidityTrigger), 2));
         }
}
```

此回调会在我们注册的 MQTT 主题上发布消息时被调用。在我们的消息中，我们期望找到一个单字节（uint8）值，该值定义了命令，最多八个不同的命令。对于此模块，我们之前定义了三个命令。

这些命令如下定义：

| **命令** | **含义** | **负载** | **返回值** |
| --- | --- | --- | --- |
| 0x01 | 获取土壤湿度 | - | 0xXXXX |
| 0x02 | 设置触发级别 | uint16（新触发级别） | - |
| 0x04 | 获取触发级别 | - | 0xXXXX |

在这里，每个命令都会返回请求的值（如果适用）。

在确认我们收到的消息字符串中至少有一个字节后，我们提取第一个字节，并尝试将其解释为命令。如果我们正在设置新的触发点，我们也会从消息中提取新的值作为 uint16，前提是我们有一个格式正确的消息。

最后，这里是我们一直在本项目中努力实现的所有魔法发生的函数：

```cpp
void PlantModule::readSensor() {
    int16_t val = 0;
    val = analogRead(A0); // calls system_adc_read().

    String response = OtaCore::getLocation() + ";" + val;
    OtaCore::publish(MQTT_PREFIX"nsa/plant/moisture_raw", response);
```

作为第一步，我们从 ESP8266 的模拟输入读取当前传感器值，并将其发布到以下 MQTT 主题：

```cpp
        if (val >= humidityTrigger) {
               digitalWrite(pin, HIGH);

               LED->setBrightness(31);
               LED->setAllPixel(0, 0, 255);
               LED->show();

               for (int i = 0; i < 10; ++i) {
                     LED->directWrite(0, 0, 255, 25);
                     delay(200);
                     LED->directWrite(0, 0, 255, 18);
                     delay(200);
                     LED->directWrite(0, 0, 255, 12);
                     delay(200);
                     LED->directWrite(0, 0, 255, 5);
                     delay(200);
                     LED->directWrite(0, 0, 255, 31);
                     delay(200);
               }

               digitalWrite(pin, LOW);
         }
}
```

在使用土壤湿度传感器校准一个原型时，发现一个完全干燥的传感器（悬在空中）的值大约为 766，而将相同的传感器浸入水中得到的值为 379。由此我们可以推断，60% 的湿度含量大约在 533 的读数左右，这与我们在静态初始化步骤中设置的初始值相匹配。当然，理想的触发点和目标土壤湿度水平取决于土壤类型和特定植物。

当达到这个触发水平时，我们将连接到升压转换器使能引脚的输出引脚设置为高，使其开启输出，这反过来又启动了泵。我们希望让它泵大约十秒钟。

在这段时间内，我们将 LED 颜色设置为蓝色，然后在每一秒内将其亮度从 100% 降低到几乎关闭，然后再回到全亮度，从而产生脉冲效果。

然后，我们将输出引脚设置回低，这会禁用泵，并等待下一次土壤湿度传感器的读数：

```cpp
void PlantModule::onRequest(HttpRequest& request, HttpResponse& response) {
         TemplateFileStream* tmpl = new TemplateFileStream("index.html");
         TemplateVariables& vars = tmpl->variables();
         int16_t val = analogRead(A0);
         int8_t perc = 100 - ((val - 379) / 3.87);
         vars["raw_value"] = String(val);
         vars["percentage"] = String(perc);

         response.sendTemplate(tmpl);
}
```

最后，我们在这里看到我们网络服务器的请求处理器。它所做的是从 SPIFFS 中读取一个模板文件（下一节将详细介绍），获取该模板文件中的变量列表，然后继续读取当前的传感器值。

使用这个值，它计算当前的土壤湿度百分比，并使用原始和计算出的数字在返回之前填充模板中的两个变量。

# Index.html

为了与 PlantModule 的网络服务器一起使用，我们必须将以下模板文件添加到 SPIFFS 中：

```cpp
<!DOCTYPE html>
<html>
<head>
         <title>Plant soil moisture readings</title>
   </head>
   <body>
         Current value: {raw_value}<br>
         Percentage: {percentage}%
</body>
</html>
```

# 编译和烧录

在完成我们应用程序的代码后，我们可以在项目的根目录下使用单个命令编译它：

```cpp
make  
```

完成此操作后，我们可以在 `out` 文件夹中找到包括 ROM 镜像在内的二进制文件。由于我们同时使用 rBoot 引导加载程序和 SPIFFs，`firmware` 文件夹中总共有三个 ROM 镜像。

在这一点上，我们可以连接一个 ESP8266 模块，无论是 NodeMCU 板的形式还是许多替代品之一，并注意它将连接到的串行端口。在 Windows 上，这可能是 `COM3`；在 Linux 上，USB 到串行适配器通常注册为 `/dev/ttyUSB0` 或类似。

如果我们没有在用户 Makefile 中指定串行端口 (`COM_PORT`)，那么在向 ESP8266 模块烧录时，我们必须明确指定它：

```cpp
    make flash COM_PORT=/dev/ttyUSB0  
```

执行此命令后，我们应该看到 `esptool.py` 工具的输出，因为它连接到 ESP8266 的 ROM 并开始将其写入 ROM 镜像。

一旦完成，MCU 将重新启动，并且应该直接进入新的固件镜像，在那里它将等待我们的命令来配置它。

# 首次配置

如本章前面所述，此固件设计为可以通过 MQTT 进行配置和维护。这需要有一个 MQTT 代理可用。Mosquitto（[`mosquitto.org/`](http://mosquitto.org/））是一个流行的 MQTT 代理。由于它是一个轻量级服务器，它可以安装在桌面系统、小型 SBC、虚拟机等上。

除了代理和运行固件的 ESP8266，我们还需要我们自己的客户端

与固件交互。由于我们使用二进制协议，我们的选择相对有限，因为大多数常见的 MQTT 桌面客户端都假设基于文本的消息。一个

有限，因为大多数常见的 MQTT 桌面客户端都假设基于文本的消息。一个

可以用来发布二进制消息的方法是使用 MQTT 发布客户端。

随着 Mosquitto 一起提供的 **echo** 命令行工具的十六进制输入，我们可以发送

将二进制数据作为流发送到客户端工具

因此，本书的作者开发了一个新的 MQTT 桌面客户端（基于 C++ 和 Qt），该客户端旨在围绕 MQTT 上的二进制协议的使用和调试进行设计：[https://github.com/MayaPosch/MQTTCute](https://github.com/MayaPosch/MQTTCute)。

当三个组件都到位——ESP8266 运行项目、MQTT 代理和桌面客户端——我们可以组装整个植物监控和灌溉系统，并发送命令以启用植物模块。

当我们在 cc/config 主题上监控消息时，我们应该看到 ESP8266 通过发布其 `MAC` 来报告其存在。我们也可以通过将 USB 到 TTL 串行适配器连接到串行日志输出引脚（NodeMCU 上的 `D4`）来获取这个信息。通过查看我们的串行控制台输出，我们将看到系统的 IP 地址和 `MAC`。

当我们创建一个格式为 `cc/<MAC>` 的新主题时，我们可以然后向固件发布命令，例如：

```cpp
    log;plant001  
```

这将设置系统的位置名为 `plant001`。

当使用 MQTTCute 客户端时，我们可以使用 echo 风格的二进制输入，使用十六进制输入来激活植物模块：

```cpp
mod;\x00\x01\x00\x00  
```

这将向固件发送 `mod` 命令，以及一个值为 0x100 的位掩码。在此之后，植物模块应该被激活并运行。由于我们正在持久化位置字符串和配置，我们不需要再次重复此步骤，除非我们进行 OTA 更新，此时新固件将具有空的 SPIFFS 文件系统，除非我们在 ROM 的两个 SPIFFS 插槽上闪存相同的 SPIFFS 图像。

在这里，我们可以扩展 OTA 代码，除了下载固件之外，还可以下载 SPIFFS ROM 图像，尽管这可能会增加覆盖现有 SPIFFS 文件的复杂性。

在这一点上，我们应该有一个工作的植物监控和灌溉系统。

# 使用该系统

我们可以通过订阅 `nsa/plant/moisture_raw` 主题来使用测量值并将它们存储在数据库中。通过向 `plant/<location string>` 主题发送新命令，我们可以调整触发点。

设备上的 Web 服务器可以通过获取 IP 地址来访问，我们可以通过查看串行控制台输出（如前所述）或查看路由器中的活动 IP 地址来找到它。

在浏览器中打开这个 IP 地址，我们应该会看到填入当前值的 HTML 模板。

# 进一步探索

您还需要考虑以下因素：

+   在这一点上，您可以通过实现植物浇水配置文件来进一步细化系统，以添加干燥期或调整特定土壤类型。您还可以添加新的 RGB LED 模式，充分利用可用的颜色选择。

+   整个硬件可以集成到一个外壳中，使其融入背景，或者可能使其更加显眼。

+   Web 界面可以扩展，允许通过浏览器控制触发点等，而不是必须使用 MQTT 客户端。

+   除了湿度传感器外，您还可以添加亮度传感器、温度传感器等，以测量更多影响植物健康方面的因素。

+   为了加分，您还可以自动化向植物施加（液体）肥料的操作。

# 复杂性

您可能会遇到的一个可能的复杂性是 ESP8266 的 ADC 问题，在 NodeMCU 板上，紧挨着 ADC 引脚的第一个保留（RSV）引脚直接连接到 ESP8266 模块的 ADC 输入。这可能会因静电放电 ESD 暴露而引起问题。本质上是在 MCU 中放电高压、低电流。在这个 RSV 引脚上添加一个小电容器可以帮助降低这种风险。

这个系统显然无法帮助的是保持植物无病虫害。这意味着尽管浇水可以自动化，但这并不意味着您可以忽视植物。定期检查植物是否有任何问题，以及系统是否有任何可能发展的问题（如断开的软管、由于猫而倒下的东西等），仍然是一项重要的任务。

# 摘要

在本章中，我们探讨了如何将一个基于 ESP8266 的简单项目从理论和简单需求转变为一个具有多功能固件和一系列输入输出选项的运行设计，通过这些我们可以确保连接的植物获得适量的水分以保持健康。我们还看到了如何为 ESP8266 设置开发环境。

读者现在应该能够为 ESP8266 创建项目，用新的固件编程 MCU，并且对这一开发平台的优势和局限性有一个稳固的掌握。

在下一章中，我们将探讨如何测试为 SoC 和其他大型嵌入式平台编写的嵌入式软件。
