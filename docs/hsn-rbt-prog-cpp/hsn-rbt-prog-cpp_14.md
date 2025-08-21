# 第十章：评估

# 第一章：树莓派简介

1.  Broadcom BCM2837 四核 1.4 GHz 处理器

1.  40

1.  VNC 查看器

1.  用户名：`pi`，密码：`raspberry`

1.  `sudo raspi-config`

# 第二章：使用 wiringPi 实现闪烁

1.  八个（引脚号 **6**, **9**, **14**, **20**, **25**, **30**, **34**, 和 **39**）

1.  高

1.  `digitalRead(pinnumber);`

1.  `for (int i=0; i<6;i++)`

1.  1V

# 第三章：编程机器人

1.  L298N 电机驱动 IC

1.  H 桥

1.  `digitalWrite(0,HIGH);`

`digitalWrite(2,LOW);`

`digitalWrite(3,HIGH);`

`digitalWrite(4,LOW);`

1.  逆时针方向

1.  `digitalWrite(0,HIGH);`

`digitalWrite(2,HIGH);`

`digitalWrite(3,HIGH);`

`digitalWrite(4,LOW);`

1.  `digitalWrite(0,HIGH);`

`digitalWrite(2,LOW);`

`digitalWrite(3,LOW);`

`digitalWrite(4,HIGH);`

# 第四章：构建避障机器人

1.  超声波脉冲以 340 m/s 的速度传播

1.  液晶显示

1.  180 厘米

1.  第 4 列和第 1 行

1.  它们用于调节 LCD 的背光

# 第五章：使用笔记本电脑控制机器人

1.  `initscr()` 和 `endwin()`

1.  `initscr()` 函数初始化屏幕。它设置内存并清除命令窗口屏幕

1.  `gcc -o Programname-lncurses Programname.cpp`

1.  GCC

1.  `pressed()` 按下按钮时移动机器人，`released()` 松开按钮时停止移动

# 第六章：使用 OpenCV 构建一个目标跟随机器人

1.  开源计算机视觉

1.  3,280 x 2,464 像素

1.  `raspistill`

1.  `raspivid`

1.  8GB - 50% and  32 GB - 15%

# 第七章：使用 OpenCV 访问 RPi 相机

1.  阈值处理

1.  `flip(original_image, new_image, 0)`

1.  右下角的区块

1.  `resize(original_image , resized_image , cvSize(640,480));`

1.  在屏幕的左上部分

# 第八章：使用 Haar 分类器进行人脸检测和跟踪

1.  `haarcascade_frontalface_alt2.xml`。

1.  水平线特征。

1.  `haarcascade_lefteye_2splits.xml`。

1.  感兴趣的区域。

1.  `equalizeHist` 函数改善图像的亮度和对比度。这很重要，因为在光线不足的情况下，相机可能无法从图像中区分出脸部。

# 第九章：构建语音控制机器人

1.  **无线电频率通信** (**RFCOMM**)

1.  **媒体访问控制** (**MAC**) 地址

1.  ListPicker 显示了所有已经与您的智能手机蓝牙配对的蓝牙设备列表

1.  连接

1.  `raspberrypi`
