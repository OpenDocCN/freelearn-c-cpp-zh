# 前言

C++是最受欢迎的传统编程语言之一，用于机器人技术，许多领先的行业都使用 C++和机器人硬件的组合。本书将弥合树莓派和 C/C++编程之间的差距，并使您能够为树莓派开发应用程序。要跟随本书中涵盖的项目，您可以使用 wiringPi 库在树莓派上实现 C 程序。

通过本书，您将开发一个完全功能的小车机器人，并编写程序以不同的方向移动它。然后，您将使用超声波传感器创建一个避障机器人。此外，您将了解如何使用您的 PC/Mac 无线控制机器人。本书还将帮助您使用 OpenCV 处理对象检测和跟踪，并指导您探索人脸检测技术。最后，您将创建一个 Android 应用程序，并使用 Android 智能手机无线控制机器人。

通过本书，您将获得使用树莓派和 C/C++编程开发机器人的经验。

# 这本书适合谁

本书适用于希望利用 C++构建激动人心的机器人应用程序的开发人员、程序员和机器人爱好者。需要一些 C++的先验知识。

# 本书涵盖的内容

第一章，*树莓派简介*，介绍了树莓派的不同模式和 GPIO 引脚配置。然后，我们将设置树莓派 B+和树莓派 Zero，并在其上安装 Raspbian 操作系统。我们还将学习如何通过 Wi-Fi 网络将树莓派无线连接到笔记本电脑。

第二章，*使用 wiringPi 实现 Blink*，介绍了 wiringPi 库的安装。在本章中，我们将了解树莓派的 wiringPi 引脚连接。然后，我们将编写两个 C++程序，并将它们上传到我们的树莓派上。

第三章，*编程机器人*，介绍了选择机器人底盘的标准。之后，我们将构建我们的小车，将电机驱动器连接到树莓派，并了解 H 桥电路的工作原理。最后，我们将编写程序，使机器人向前、向后、向左和向右移动。

第四章，*构建避障机器人*，介绍了超声波传感器的工作原理，并编写了一个测量距离值的程序。接下来，我们将编程 16 x 2 LCD 以读取超声波距离值。我们还将研究 I2C LCD，它将 16 个 LCD 引脚作为输入，并提供四个引脚作为输出，从而简化了接线连接。最后，我们将在机器人上安装超声波传感器，创建我们的避障机器人。当附近没有障碍物时，这个机器人将自由移动，如果它接近障碍物，它将通过转弯来避开。

第五章，*使用笔记本电脑控制机器人*，介绍了使用笔记本电脑控制机器人的两种不同技术。在第一种技术中，我们将使用 ncurses 库从键盘接收输入，以相应地移动机器人。在第二种技术中，我们将使用 QT Creator IDE 创建 GUI 按钮，然后使用这些按钮以不同的方向移动机器人。

第六章，*使用 OpenCV 访问 Rpi 相机*，重点介绍了在树莓派上安装 OpenCV。您还将了解树莓派相机模块，并在设置 Pi 相机后，使用 Pi 相机拍照和录制短视频剪辑。

第七章，*使用 OpenCV 构建一个目标跟随机器人*，介绍了 OpenCV 库中的一些重要功能。之后，我们将对这些功能进行测试，并尝试从图像中识别对象。然后，我们将学习如何从 Pi 摄像头读取视频源，如何对彩色球进行阈值处理，以及如何在其上放置一个红点。最后，我们将使用 Pi 摄像头和超声波传感器来检测球并跟随它。

第八章，*使用 Haar 分类器进行面部检测和跟踪*，使用 Haar 面部分类器从视频源中检测面部并在其周围绘制一个矩形。接下来，我们将检测给定面部上的眼睛和微笑，并创建一个围绕眼睛和嘴的圆圈。在使用这些面部和眼睛检测知识后，我们将在检测到眼睛和微笑时首先打开/关闭 LED。接下来，通过在面部中心创建一个白点，我们将使机器人跟随面部。

第九章，*构建语音控制机器人*，从创建我们的第一个 Android 应用程序 Talking Pi 开始，其中文本框中的文本将显示在标签中，并由智能手机朗读出来。然后，我们将为机器人开发一个语音控制的 Android 应用程序，该应用程序将识别我们的声音并通过蓝牙将文本发送到 RPi。之后，我们将使用终端窗口将 Android 智能手机的蓝牙与 RPi 的蓝牙配对。最后，我们将研究套接字编程，并编写 VoiceBot 程序，以建立与 Android 智能手机蓝牙的连接，以控制机器人。

# 为了充分利用本书

要使用本书中的代码，需要 Raspberry Pi 3B+或 Raspberry Pi Zero 板。每章的*技术要求*部分中提到了额外的硬件和软件。

# 下载示例代码文件

您可以从您在[www.packt.com](http://www.packt.com)的帐户中下载本书的示例代码文件。如果您在其他地方购买了本书，您可以访问[www.packt.com/support](http://www.packt.com/support)并注册，以便文件直接通过电子邮件发送给您。

您可以按照以下步骤下载代码文件：

1.  在[www.packt.com](http://www.packt.com)登录或注册。

1.  选择 SUPPORT 选项卡。

1.  单击“代码下载和勘误”。

1.  在搜索框中输入书名，然后按照屏幕上的说明操作。

下载文件后，请确保使用最新版本的解压缩或提取文件夹：

+   WinRAR/7-Zip 适用于 Windows

+   Mac 上的 Zipeg/iZip/UnRarX

+   7-Zip/PeaZip 适用于 Linux

该书的代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Hands-On-Robotics-Programming-with-Cpp`](https://github.com/PacktPublishing/Hands-On-Robotics-Programming-with-Cpp)。如果代码有更新，将在现有的 GitHub 存储库上进行更新。

我们还提供了来自我们丰富书籍和视频目录的其他代码包，网址为**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**。请查看！

# 下载彩色图像

我们还提供了一个 PDF 文件，其中包含本书中使用的屏幕截图/图表的彩色图像。您可以在这里下载：[`www.packtpub.com/sites/default/files/downloads/9781789139006_ColorImages.pdf`](http://www.packtpub.com/sites/default/files/downloads/9781789139006_ColorImages.pdf)。

# 使用的约定

本书中使用了许多文本约定。

`CodeInText`：指示文本中的代码词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 句柄。这是一个例子：“将轴向和径向转向的代码添加到`RobotMovement.cpp`程序中。”

代码块设置如下：

```cpp
digitalWrite(0,HIGH);           //PIN O & 2 will STOP the Left Motor
digitalWrite(2,HIGH);
digitalWrite(3,HIGH);          //PIN 3 & 4 will STOP the Right Motor
digitalWrite(4,HIGH);
delay(3000);
```

当我们希望引起您对代码块的特定部分的注意时，相关行或项目会以粗体显示：

```cpp
digitalWrite(0,HIGH);           //PIN O & 2 will STOP the Left Motor
digitalWrite(2,HIGH);
digitalWrite(3,HIGH);          //PIN 3 & 4 will STOP the Right Motor
digitalWrite(4,HIGH);
delay(3000);
```

任何命令行输入或输出都将以以下方式书写：

```cpp
sudo nano /boot/config.txt
```

**粗体**: 表示一个新术语、一个重要词或者屏幕上看到的词。例如，菜单或对话框中的单词会以这种方式出现在文本中。这里有一个例子："选择 记住密码 选项 然后按 确定。"

警告或重要说明会显示在这样。

提示和技巧会显示在这样。
