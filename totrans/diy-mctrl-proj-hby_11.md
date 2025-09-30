# *第十一章*：物联网太阳能（电压）测量

面对全球变暖的挑战，太阳能被认为是**可再生能源**中最有希望的一种。它被认为是在减少对化石燃料的依赖和满足不断增长的电力需求方面最好的替代品之一（Ryan，2005）。为了实现这一点，阳光被转化为电能，而阳光是通过太阳能板收集的。

在本章中，您将继续为 STM32 Blue Pill 微控制器板创建物联网软件，使用电压传感器来测量太阳能板收集的太阳能。该应用将通过 NodeMCU ESP8266 微控制器板将感应数据发送到互联网。

在本章中，我们将涵盖以下主要主题：

+   将太阳能板连接到 Blue Pill 板

+   从电压传感器模块读取数据

+   编写程序将感应数据发送到互联网

+   在互联网上显示传感器数据结果

在本章之后，您将具备开发物联网应用和提升您的简历的扎实技能，因为它是工业 4.0 的核心元素。您将学习的第一个技能是从连接到 STM32 Blue Pill 的传感器读取太阳能板电压。此外，您将学习如何通过 NodeMCU 8266 开发板将读取的信息发送到互联网。最后，您将了解如何在移动物联网应用上可视化传感器值。

# 技术要求

开发太阳能能量测量系统所需的硬件组件如下：

+   一个无焊面包板。

+   一个 Blue Pill 微控制器板。

+   一个 NodeMCU 微控制器。

+   一个 ST-Link/V2 电子接口，用于将编译后的代码上传到 Blue Pill 板。请注意，ST-Link/V2 需要四根公对公跳线。

+   一个 B25 电压传感器。

+   一个太阳能板。

+   公对公跳线。

+   公对母跳线。

+   电源。

所有组件都可以在您偏好的电子供应商处轻松找到。记住，您将需要 Arduino IDE 和 GitHub 仓库来完成本章：[`github.com/PacktPublishing/DIY-Microcontroller-Projects-for-Hobbyists/tree/master/Chapter11`](https://github.com/PacktPublishing/DIY-Microcontroller-Projects-for-Hobbyists/tree/master/Chapter11)

本章的“代码在行动”视频可以在这里找到：[`bit.ly/2U4YMsT`](https://bit.ly/2U4YMsT)

下一节将介绍太阳能板和 B25 电压测量传感器，以及如何将它们与 STM32 Blue Pill 微控制器板接口连接。

# 将太阳能板连接到 Blue Pill 板

首先，我们需要了解两个组件：太阳能板和电压测量传感器。在了解基础知识后，我们可以构建我们的太阳能能量测量系统。

## 介绍太阳能板

太阳光携带能量。当阳光与半导体碰撞时，部分能量会转化为移动的电子，从而产生电流。太阳能电池（也称为光伏板或 PV 板）是为了利用到达我们星球的所有阳光而设计的。当阳光反射在 PV 板上时，电流输出保持恒定；这被称为**直流**（**DC**）电。这种直流电可以用来给电池充电并为微控制器如 STM32 Blue Pill 提供电力。

以下截图显示了用于电子组件（如我们的太阳能演示系统）的太阳能板：

![图 11.1 – 太阳能板](img/B16413_Figure_11.1.jpg)

图 11.1 – 太阳能板

为了方便与该太阳能板连接和操作，我们将焊接一个引脚头到板上，这样我们就可以直接连接跳线到它。以下图显示了引脚头和焊接后的 PV 板的外观：

![图 11.2 – 将引脚头焊接至太阳能板](img/B16413_Figure_11.2.jpg)

图 11.2 – 将引脚头焊接至太阳能板

您还可以在市场上找到已经集成电缆的太阳能板，以方便使用，如下图所示：

![图 11.3 – 带集成线的太阳能板](img/B16413_Figure_11.3.jpg)

图 11.3 – 带集成线的太阳能板

在了解了太阳能板的外观和功能后，让我们继续下一小节，我们将探讨我们将用来测量电压的传感器。

## B25 电压传感器

如果我们需要测量电压，我们可以使用 STM32 Blue Pill 板的模拟输入。这些输入的电压限制为 5V，因此如果需要测量更高的电压，就必须使用外部传感器来完成。**B25 传感器**（见 *图 11.4*）测量 5V 至 25V 范围的电压，使其成为这项任务的非常受欢迎的传感器：

![图 11.4 – B25 电压传感器扩展板](img/B16413_Figure_11.4.jpg)

图 11.4 – B25 电压传感器扩展板

如所示，该模块有两个终端，外部电源将连接到这两个终端，一个连接到 GND，另一个连接到 VCC，必须用螺丝调整。

此外，扩展板通过 3 个引脚头连接到 STM32 Blue Pill。它们如下：

+   **S**：此引脚产生模拟信号，必须连接到微控制器的模拟输入。

+   **+**：未连接。

+   **-**：接地连接。

在这个信息的基础上，我们将在下一小节学习如何将电压传感器连接到 STM32 Blue Pill 板上。

## 连接组件

我们将使用无焊面包板连接传感器和 STM32 Blue Pill 微控制器，并最终用电线连接组件。以下是我们的接线方式和组件连接方法：

1.  将电压传感器和 STM32 Blue Pill 放在无焊面包板上。留出一些空位以添加电线。

1.  将传感器的**地**（**GND**）引脚连接到 STM32 Blue Pill 的 GND 端子。

1.  接下来，你需要将传感器的模拟输出连接到 STM32 Blue Pill 卡上的模拟输入，并将传感器的 S 连接到 Blue Pill 的**A0**引脚，如图*图 11.5*所示：![图 11.5 – 电压传感器连接到 Blue Pill](img/B16413_Figure_11.5.jpg)

    图 11.5 – 电压传感器连接到 Blue Pill

1.  最后，你需要一个电源连接来将太阳能板连接到板上。使用 STLink 将脚本上传到 STM32 Blue Pill 微控制器板。*图 11.6*总结了所有硬件连接：

![图 11.6：电压传感器连接电路](img/B16413_Figure_11.6.jpg)

图 11.6：电压传感器连接电路

以下截图展示了本项目的原理图：

![图 11.7 – 电压传感器连接的原理图](img/B16413_Figure_11.7.jpg)

图 11.7 – 电压传感器连接的原理图

原理图显示了电气连接。光伏板的 VCC 和地端子连接到传感器的 VCC 和 GND 引脚。为了将 Blue Pill 与传感器接口，其地引脚连接到 Blue Pill 的 GND 总线，最后，传感器的模拟输出（**S**）插入到 STM32 微控制器的 A0 引脚。*图 11.8*显示了太阳能测量系统：

![图 11.8 – 太阳能测量系统](img/B16413_Figure_11.8.jpg)

图 11.8 – 太阳能测量系统

现在我们已经完成了组件的连接，我们创建了一个简单的电路用于我们的电压测量系统，如前图所示。

在本节中，我们了解了太阳能板，并遇到了电压传感器及其组件。我们还学习了如何将太阳能电池连接到电压传感器，以及如何将电压传感器连接到 STM32 Blue Pill。

是时候进入下一节了，该节将展示如何编写 C 代码来完成我们的物联网太阳能监测系统的第一个功能。

# 从电压传感器模块读取数据

是时候学习如何编写一个程序来读取电压传感器的信息并在串行监视器上显示其读数了。

让我们编写程序来接收 STM32 Blue Pill 的传感器数据：

1.  声明 STM32 Blue Pill 卡上哪个引脚将用作传感器数据的输入：

    ```cpp
    0 (labeled A0 on the Blue Pill).
    ```

1.  接下来，在`setup()`部分，开始串行数据传输，并将传输速度设置为 9600 bps，并告知微控制器分配给 A0 的引脚类型：

    ```cpp
    void setup() {
      Serial.begin(9600);
      pinMode(sensorPin, INPUT);
    }
    ```

1.  现在，在`loop()`中，首先读取输入引脚的传感器数据，将其值发送到串行端口，并等待一秒钟：

    ```cpp
    void loop() {
      int sensorValue = analogRead(sensorPin);
      Serial.print("Voltage: ");
      Serial.println(sensorValue);
      delay(1000);
    }
    ```

1.  我们将程序加载到 STM32 板上，并查看 Arduino IDE 的串行绘图器，以了解我们从传感器读取的模拟信号的波形，结果可见于 *图 11.9*：![图 11.9 – 串行绘图器中的传感器信号波形](img/B16413_Figure_11.9.jpg)

    图 11.9 – 串行绘图器中的传感器信号波形

    形成传感器信号的波形可以取 `0` 到 `1023` 的值。然后，需要将此值转换为电压。

1.  我们将在脚本中添加两行来显示电压值，并对将信号值发送到串行端口的行进行注释：

    ```cpp
    void loop() {
      int sensorValue = analogRead(sensorPin);
    map() function transforms a number from one range to another:

    ```

    map(value, fromLow, fromHigh, toLow, toHigh)

    ```cpp

    The first parameter that `map()` receives is the value to be converted. In our program, it is the value read from the sensor. The value of `fromLow` will be mapped to `toLow`, and `fromHigh` to `toHigh`, and all values within the range.Now, upload it to the Blue Pill board. Now you can see in the **serial monitor** the voltage value as shown in *Figure 11.10*:
    ```

![图 11.10 – 串行监视器读数](img/B16413_Figure_11.10.jpg)

图 11.10 – 串行监视器读数

对于完整的草图，请参阅 GitHub 仓库中的 `Chapter11/voltage` 文件夹。

我们到目前为止学到了什么？我们介绍了 B25 传感器来测量电压，并了解了太阳能板。我们学习了如何将它们连接到我们的 STM32 Blue Pill 微控制器，编写代码读取传感器数据，在串行监视器上显示它，并在串行绘图器中绘制它。

在本节中，我们获得了一些新技能，这些技能将帮助您构建需要监测电压水平的电子系统。

接下来，我们将使用 NodeMCU 微控制器将感应数据发送到互联网。

# 编写程序将感应数据发送到互联网

在本节中，我们将继续使用 NodeMCU 开发板从 STM32 接收数据并将其发送到互联网。然而，与 *第十章* 中的 *IoT 植物花盆湿度传感器* 不同，其中两个微控制器之间直接发送数字值（1 或 0），我们现在需要在这些微控制器之间使用串行通信发送电压值。

串行传输是通过使用 RX/TX 引脚发送数据来完成的。

让我们创建连接 NodeMCU 和 STM32 的程序：

1.  在 `setup()` 中，我们需要添加新的串行数据传输到 115200 bps。这是 NodeMCU 板的推荐速度：

    ```cpp
    void setup() {
      serial.begin(9600);
      Serial1.begin(115200);
    }
    ```

1.  `loop()` 实例需要在读取传感器和电压转换后添加新的一行。`write()` 函数将数据作为整数值发送：

    ```cpp
    void loop() {
      int sensorValue = analogRead(sensorPin);
      double voltageValue = map(sensorValue, 0, 1023, 0, 25);
      Serial.print("Voltage: ");
      //Serial.println(sensorValue);
      Serial.println(voltageValue);
      Serial1.write((int)voltageValue);
      delay(1000);
    }
    ```

1.  要完成 NodeMCU 和 STM32 之间的通信，需要添加 *图 11.11* 和 *图 11.12* 中显示的附加连接：![图 11.11 – 微控制器串行通信电路](img/B16413_Figure_11.11.jpg)

    图 11.11 – 微控制器串行通信电路

    *图 11.12* 显示了 STM32 和 NodeMCU 微控制器之间电路接口的原理图：

    ![图 11.12 – 微控制器串行通信原理图](img/B16413_Figure_11.12.jpg)

    图 11.12 – 微控制器串行通信原理图

    将 NodeMCU 的 RX 引脚连接到 STM32 的 TX 引脚（B6），并将 NodeMCU 的 TX 引脚连接到 STM32 的 RX 引脚（B7）。

    *图 11.13*显示了实际系统中所有连接的方式，包括电压传感器：

    ![图 11.13 – STM32 和 NodeMCU 串行连接](img/B16413_Figure_11.13.jpg)

    图 11.13 – STM32 和 NodeMCU 串行连接

1.  现在，为了完成 NodeMCU 和 STM32 之间的串行连接，我们将创建一个新的草图，`Chapter11/voltage_iot.`

1.  在`setup()`中，指示串行数据传输：

    ```cpp
    void setup() {
      Serial.begin(115200);
    }
    ```

1.  最后一步是`loop()`：

    ```cpp
    void loop() {
      double data = Serial.read();
      Serial.print("Voltage: ");
      Serial.println(data);
      delay(1000);
    } 
    ```

    使用前面的代码，NodeMCU 将从 STM32 接收传感器值，并将其显示在串行监视器上。

草图现在已完成。将其上传到 NodeMCU 板，并在上传完成后重置它。现在您可以看到，在**串行监视器**中，传感器值，如下面的截图所示：

现在是时候进入下一节了，该节将向您展示如何在网上可视化数据。

# 在互联网上显示传感器数据结果

在*第九章*，“物联网温度记录系统”，和*第十章*，“物联网植物花盆湿度传感器”中，我们学习了如何在本地网络内编程物联网应用。在本章的这一节中，我们将学习如何将数据发送到本地网络之外的云。

众多云平台允许我们将我们的物联网设备连接到它们的服务。大多数平台允许我们免费使用基本服务。如果需要更完整的服务，则可能需要付费，通常是每月支付。这次我们将使用 Blynk 平台，它有几个免费选项，我们将使用这些选项。

Blynk 为 Android 和 iOS 都提供了应用程序，这将使我们能够监控太阳能电池板上的电压值。

让我们看看如何使用移动应用程序从互联网发送和查看我们的信息：

1.  下载 Blynk 应用程序。

    对于 Android，从[`play.google.com/store/apps/details?id=cc.blynk&hl=en_US`](https://play.google.com/store/apps/details?id=cc.blynk&hl=en_US)下载。

    对于 iOS，从[`apps.apple.com/us/app/blynk-iot-for-arduino-esp32/id808760481`](https://apps.apple.com/us/app/blynk-iot-for-arduino-esp32/id808760481)下载。

1.  创建一个新账户：![图 11.14 – Blynk，主页屏幕](img/B16413_Figure_11.14.jpg)

    图 11.14 – Blynk，主页屏幕

1.  一旦创建账户，创建一个新的项目。输入名称，选择 ESP8266 作为设备，并将 WiFi 设置为连接类型。然后点击**创建项目**：![图 11.15 – Blynk，创建新账户](img/B16413_Figure_11.15.jpg)

    图 11.15 – Blynk，创建新账户

1.  您将收到一封包含应用程序所需令牌的电子邮件，您也可以在**设置**中找到：![图 11.16 – Blynk，菜单屏幕](img/B16413_Figure_11.16.jpg)

    图 11.16 – Blynk，菜单屏幕

1.  输入一个名称，选择**ESP8266**作为设备，并选择**WiFi**作为连接类型。点击**创建项目**：![图 11.17 – Blynk，创建新项目](img/B16413_Figure_11.17.jpg)

    图 11.17 – Blynk，创建新项目

1.  你将收到一封包含应用程序所需令牌的电子邮件，你也可以在**设置**中找到它。

1.  按压屏幕，Widget 工具箱将出现：![图 11.18 – Blynk，小部件框](img/B16413_Figure_11.18.jpg)

    图 11.18 – Blynk，小部件框

1.  添加一个**仪表**组件。配置它并按下**确定**按钮：![图 11.19 – Blynk，太阳能应用](img/B16413_Figure_11.19.jpg)

    图 11.19 – Blynk，太阳能应用

1.  最后，将 `Chapter11/voltage_iot` 程序上传到 NodeMCU 并执行它。

我们已经到达了 *第十一章* 的结尾，“物联网太阳能（电压）测量”。恭喜！

# 摘要

在本章专门介绍物联网的章节中，我们学习了一些基本主题。首先，我们了解了用于为小型电子设备供电的太阳能电池。接下来，我们学习了 B25 电压传感器以及如何将其连接到 STM32。

之后，我们学习了如何创建一个程序来读取电压传感器的数据。有了电压读数，我们通过串行通信将 STM32 连接到 NodeMCU 板。我们创建了一个程序，在微控制器之间发送电压值。最后，我们使用一个应用程序在云端可视化传感器数据。

在物联网主题的结尾，你拥有了创建连接到互联网和内网的程序和设备的扎实技能。你的项目组合得到了加强，这将使你更容易在这个增长领域找到工作机会。

在下一章中，你将开始开发项目，这些项目将帮助你创建电子支持设备，以协助应对 COVID-19 大流行。

# 进一步阅读

Ryan, V., *什么是太阳能？* 技术学生，2005: [`technologystudent.com/energy1/solar1.htm`](https://technologystudent.com/energy1/solar1.htm)
