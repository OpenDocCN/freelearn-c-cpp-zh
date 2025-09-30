# 前言

欢迎来到 *Arduino IoT Cloud for Developers*。在一个日益通过 **物联网（IoT**）相互连接的世界中，本书作为您掌握使用多才多艺的 Arduino 平台进行物联网开发的全面指南。无论您是物联网的新手还是希望扩展技能的资深开发者，本书都提供了一种动手、分步的方法，以 Arduino IoT Cloud 构建有意义的物联网项目。

物联网（IoT）彻底改变了我们与世界互动的方式。从家用电器到工业机械，日常物品现在都能够连接到互联网并交换数据。这种变革为创新和效率提供了无限的机会。

以其简单性和灵活性著称的 Arduino 已成为物联网爱好者和专业人士的首选平台。Arduino IoT Cloud，Arduino 生态系统的重要组成部分，提供了一个强大且用户友好的平台来构建和管理物联网项目。本书是您开启 Arduino IoT Cloud 全部潜能的门户。

通过理论与实践的结合，您将开始一段旅程，从物联网基础开始，最终达到高级物联网应用。我们将探索 Arduino IoT Cloud 的核心功能，深入研究各种通信技术，并创建实际的物联网解决方案。

本书旨在让所有人都能轻松阅读，无论是刚开始接触物联网还是希望深化知识。每一章都提供了清晰的解释、代码片段和动手项目，以巩固您的学习。

在这段旅程结束时，您不仅将牢固掌握物联网概念，还将具备使用 Arduino IoT Cloud 设计和定制物联网解决方案的能力。让我们共同踏上这个激动人心的物联网开发之旅。

# 本书面向对象

*Arduino IoT Cloud for Developers* 面向多样化的受众，包括使用 Arduino 进入物联网的物联网和 Arduino 爱好者，以及那些希望深入了解物联网应用的具有先前 Arduino 经验的开发者。对于追求电子学、计算机科学或工程学位的学生和学者，本书也很有价值，他们寻求实际的物联网知识和项目灵感。此外，热衷于为个人实验制作实际物联网项目的创客、业余爱好者和 DIY 爱好者会发现这本书很有益。对于物联网专家、工程师和开发者等专业人士，本书提供了一条途径，以创建高质量、商业级解决方案。

本书适合初学者和有经验的读者，提供从物联网基础到使用 Arduino IoT Cloud 进行高级物联网开发的逐步学习路径。无论您的背景如何，本书都为您提供了设计和实施创新物联网解决方案的技能。

# 本书涵盖内容

*第一章*, *物联网和 Arduino 简介*，涵盖了物联网架构、安全和 Arduino。

*第二章*, *初探 Arduino IoT 云平台*，通过 MKR Wi-Fi 1010 和 Node-RED 的实际演示，探讨了如何将旧设备（较老/不兼容的设备）连接到 Arduino IoT 云平台。

*第三章*, *Arduino IoT 云平台和云编辑器的见解*，提供了 Arduino IoT 云平台功能的概述，包括事物、设备和仪表板小部件，最后，我们将更深入地探讨云编辑器。

*第四章*, *项目 #1 – 为环境感知设置更智能的方案*，提供了一个项目，将实际演示如何使用 WeMos D1 Mini、DHT22 和 MQ-135 部署智能感应节点，并将环境数据发送到 Arduino IoT 云平台，并通过令人惊叹的小部件进行可视化。

*第五章*, *项目 #2 – 使用 MKR GSM 1400 创建便携式物品追踪器*，考察了资产追踪，并演示了如何使用 MKR GSM 1400 和 GPS 模块追踪资产，并通过 Arduino IoT 云平台中的地图小部件可视化数据。

*第六章*, *项目 #3 – 使用 LoRaWAN 的远程资产追踪应用*，探讨了使用 LoRaWAN 技术的远程通信。我们将设置 LoRaWAN 节点，包括 LoRaWAN 网关，使用 MKR WAN 1300 和 GPS 模块追踪资产。

*第七章*, *实现不同设备之间的通信*，提供了一个实际演示，说明如何设置跨多个事物的云变量同步，通过简单的图形用户界面实现设备之间的通信，无需任何编码。

*第八章*, *使用 Arduino IoT Cloud SDK 和 JavaScript 进行操作*，探讨了开发者如何使用 JavaScript SDK 与 Arduino IoT 云平台交互，该 SDK 包括 API 密钥设置，并使用 JavaScript 对事物、设备、云变量和仪表板进行操作。

*第九章*, *项目 #4 – 通过土壤和环境收集数据以实现智能农业*，专注于智能农业。我们将探讨如何感知土壤湿度、土壤温度等，使我们的农业更智能、更稳健。

*第十章*, *项目 #5 – 通过语音助手使您的家更智能*，深入探讨了智能家居。我们将使用 WS2812 RGB 环和 XIAO ESP32C3 创建一个智能灯，并将其连接到 Arduino IoT 云平台，同时将 Arduino IoT 云平台与 Amazon Alexa 集成，以获得语音助手体验。

*第十一章*，*实现 Arduino IoT Cloud 调度器和空中更新功能*，提供了一个如何在 Arduino IoT Cloud 中使用云调度器来自动化依赖时间的操作的实用演示。在第二部分中，我们将探讨**空中更新**（**OTA**）功能，这些功能帮助我们通过空中发送更新，而不需要任何物理连接到设备。

*第十二章*，*项目#6 – 跟踪和通知心率*，提供了一个智能健康项目的实际演示，我们将使用 XIAO ESP32C3 和心率监测传感器来构建一个可穿戴产品，该产品将数据发送到 Arduino IoT Cloud。然后，我们将使用 Webhooks 将我们的**每分钟心跳次数**（**BPM**）发送到 Zapier 进行通知警报。

*第十三章*，*使用 Cloud CLI 编写 Arduino IoT Cloud 脚本*，教您如何使用命令行工具在 Arduino IoT Cloud 上执行操作，包括对设备、事物和仪表板的操作，以及 OTA。这一章将帮助您学习命令，并为您创建自己的批处理脚本以实现自动化操作做好准备。

*第十四章*，*在 Arduino IoT Cloud 中前进*，提供了有关不同 Arduino IoT Cloud 定价计划的详细信息，包括每个计划的特性。它还探讨了 Arduino PRO 硬件和软件在工业物联网中的应用，并提供了进一步探索的完整资源列表。

# 为了充分利用这本书

*在开始这本书之前，您应该拥有以下软件和开发板，包括完整的传感器列表，以充分利用这本书*：

| **书中涵盖的软件/硬件** | **操作系统要求** |
| --- | --- |
| Arduino IDE 桌面版 | Windows、macOS 或 Linux |
| Fritzing 原理图设计软件 | Windows、macOS 或 Linux |
| 一个 Arduino IoT Cloud 账户或一个 Zapier 账户 | N/A |
| Amazon Alexa 语音助手 | N/A |

**开发板**:

+   MKR Wi-Fi 1010

+   MKR GSM 1400

+   MKR WAN 1300

+   WeMos D1 Mini ESP8266

+   XIAO ESP32C3

+   ESP32 Dev Kit V1

+   Things Indoor Gateway for LoRaWAN

**传感器**:

+   DHT22/DHT11

+   MQT-135

+   The NEO 6-M u-blox GPS module

+   电容式土壤湿度传感器

+   DS18B20 防水探头，带长线

+   WS2812 RGB 环形灯

+   脉搏传感器

**如果您使用的是这本书的数字版，我们建议您自己输入代码或从书的 GitHub 仓库（下一节中有一个链接）获取代码。这样做将帮助您避免与代码复制和粘贴相关的任何潜在错误。**

# 下载示例代码文件

您可以从 GitHub 下载本书的示例代码文件：[`github.com/PacktPublishing/Arduino-IoT-Cloud-for-Developers`](https://github.com/PacktPublishing/Arduino-IoT-Cloud-for-Developers)。如果代码有更新，它将在 GitHub 仓库中更新。

我们还有其他来自我们丰富图书和视频目录的代码包，可在[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)找到。查看它们！

# 使用的约定

本书使用了多种文本约定。

`文本中的代码`：表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称。以下是一个示例：“这个云变量声明在`thingProperties.h`文件中可用。”

代码块应如下设置：

```cpp
TinyGPSPlus gps;
unsigned long previousMillis = 0;
const long interval = 30000; //milliseconds
```

任何命令行输入或输出都应如下编写：

```cpp
npm install @arduino/arduino-iot-client
npm i request-promise
```

**粗体**：表示新术语、重要单词或您在屏幕上看到的单词。例如，菜单或对话框中的单词以**粗体**显示。以下是一个示例：“设置配置后，点击**获取新访问****令牌**按钮。”

小贴士或重要注意事项

看起来是这样的。

# 联系我们

我们读者提供的反馈总是受欢迎的。

**一般反馈**：如果您对本书的任何方面有疑问，请通过 customercare@packtpub.com 给我们发邮件，并在邮件主题中提及书名。

**勘误**：尽管我们已经尽最大努力确保内容的准确性，但错误仍然可能发生。如果您在这本书中发现了错误，我们将非常感激您向我们报告。请访问[www.packtpub.com/support/errata](http://www.packtpub.com/support/errata)并填写表格。

**盗版**：如果您在互联网上以任何形式遇到我们作品的非法副本，我们将非常感激您提供位置地址或网站名称。请通过 copyright@packtpub.com 与我们联系，并提供材料的链接。

**如果您有兴趣成为作者**：如果您在某个领域有专业知识，并且您有兴趣撰写或为本书做出贡献，请访问[authors.packtpub.com](http://authors.packtpub.com)。

# 分享您的想法

读完《Arduino IoT Cloud for Developers》后，我们非常乐意听到您的想法！请[点击此处直接进入本书的亚马逊评论页面](https://packt.link/r/1837637172)并分享您的反馈。

您的评论对我们和科技社区非常重要，并将帮助我们确保我们提供高质量的内容。

# 下载本书的免费 PDF 副本

感谢您购买本书！

您喜欢在路上阅读，但又无法携带您的印刷书籍到处走吗？

您选择的电子书购买是否与设备不兼容？

别担心，现在，随着每本 Packt 书籍，您都可以免费获得该书的 DRM 免费 PDF 版本。

在任何地方、任何设备上阅读。直接从您最喜欢的技术书籍中搜索、复制和粘贴代码到您的应用程序中。

优惠远不止于此，您还可以获得独家折扣、新闻通讯以及每天收件箱中的优质免费内容。

按照以下简单步骤获取福利：

1.  扫描二维码或访问以下链接

![](img/B19752_QR_Free_PDF.jpg)

[`packt.link/free-ebook/9781837637171`](https://packt.link/free-ebook/9781837637171)

1.  提交您的购买证明

1.  就这些！我们将直接将您的免费 PDF 和其他福利发送到您的电子邮件。

# 第一部分：物联网和通信技术及 Arduino IoT Cloud 简介

第一部分介绍了**物联网**（**IoT**）的原则，概述了 Arduino IoT Cloud 平台，并展示了如何首次使用它。然后提供了其功能的详细描述。

本部分包含以下章节：

+   *第一章*，*物联网和 Arduino 简介*

+   *第二章*，*初探 Arduino IoT Cloud*

+   *第三章*，*深入了解 Arduino IoT Cloud 平台和云编辑器*
