# 玩转灯光

我们创建的大部分大型项目将使用一个或多个 LED 作为指示器。这些 LED 可以指示诸如电源、接收数据、警告或任何我们需要视觉反馈的其他事物。我们已经看到了如何使用基本的单色 LED，但如果我们需要多个 LED 或甚至多色 LED 怎么办？在本章中，我们将探讨其他将 LED 添加到项目中的方法。

在本章中，你将学习：

+   什么是 NeoPixel

+   RGB LED 的工作原理

+   如何在项目中使用 NeoPixel

+   如何在项目中使用 RGB LED

# 简介

在本章中，我们将探讨如何使用 RGB LED 和**WS2812 40 RGB LED Pixel Arduino 屏蔽板**。让我们先了解一下 RGB LED。

多色或 RGB LED 实际上并不是一个可以改变颜色的单个 LED，它实际上是三个 LED。RGB LED 包含三个 LED，分别是红色、绿色和蓝色。LED 产生的颜色是这三个 LED 产生的颜色的组合。

RGB LED 有两种类型。这些是公共阳极和公共阴极 LED。在公共阴极 RGB LED 中，三个 LED 共享一个公共接地源，而在公共阳极 RGB LED 中，三个 LED 共享一个公共电源源。

RGB LED 有四个引脚，每个颜色一个，第四个用于公共阴极或阳极连接。以下图示显示了公共阴极和公共阳极 RGB LED 的引脚：

![图片](img/2f9d2d26-bbc0-49f9-ad94-1566ff4478b3.png)

要产生各种颜色，我们可以通过 Arduino 上的 PWM 引脚调整三个不同 LED 的强度。由于 LED 非常靠近，光线会混合在一起，从而产生我们想要的颜色。现在让我们看看 WS2812 集成光源，或者，如它们在 Adafruit 网站上所知，称为**NeoPixel**。在本章的大部分内容中，我们将把 WS2812 集成光源称为 NeoPixel，因为它更简洁，听起来也更酷。

如你所想，如果我们想在项目中包含 10 个 RGB LED，每个 LED 需要三个输入引脚，项目会很快变成一团糟。更不用说 Arduino 上的引脚也会很快用完。我们可以解决这个问题的方法之一是使用 NeoPixel。NeoPixel 将红色、绿色和蓝色 LED 以及驱动芯片集成在一个微型表面贴装组件上。这个组件可以通过一根线控制，可以单独使用或作为一组使用。NeoPixel 有多种形式，包括条带、环形、Arduino 屏蔽板，甚至珠宝。

NeoPixel 的一个优点是，没有固有的限制可以连接多少个 NeoPixel。然而，根据你使用的控制器的 RAM 和电源限制，有一些实际限制。

在本章中，我们将使用 NeoPixel 屏蔽板。如果你使用单个 NeoPixel，有几件事你需要记住：

+   在将 NeoPixels 连接到电源之前，您可能需要添加一个 1000 微法拉、6.3V 或更高电压的电容。

+   您还希望在 Arduino 数据输出和第一个 NeoPixel 的输入线之间添加一个 470 欧姆电阻。

+   如果可能的话，在电路通电时避免连接/断开 NeoPixels。如果必须将它们连接到通电电路，请始终先连接地线。如果必须从通电电路中断开它们，请始终先断开 5V 电源。

+   NeoPixels 应始终从 5V 电源供电。

在本章中，我们将使用**Keyestudio 40 RGB LED 2812 像素矩阵盾**。这个盾已经包含了电容器和电阻，所以我们只需要将盾放置在 Arduino Uno 的顶部，就可以开始了。Keyestudio 盾与 Arduino 的连接方式如图所示：

![图片](img/37ad2196-1591-40d9-b5e6-a9298a8678d6.png)

当使用其他 NeoPixel 形式时，在将其连接到 Arduino 之前，请务必阅读制造商的数据表。损坏 NeoPixels 很容易，所以请确保遵循制造商的建议。

# 需要的组件

我们将需要以下组件来完成本章的项目：

+   一块 Arduino Uno 或兼容板

+   一个 RGB LED，要么是常见的阳极，要么是常见的阴极

+   三个 330 欧姆电阻

+   一块 Keyestudio 40 RGB LED 2812 像素矩阵盾

+   跳线

+   一块面包板

# 电路图

以下图示展示了我们如何将一个常见的阳极 RGB LED 连接到 Arduino：

![图片](img/da63fa99-99f0-45ca-89ac-9c5ea6863a23.png)

在此图中，我们展示了如何连接一个常见的阳极 RGB LED。我们可以看到这一点，因为公共引脚连接到面包板上的电源轨。如果您使用的 RGB LED 是常见的阴极 LED，那么请将 LED 上的公共引脚连接到地轨而不是电源轨。每个 RGB 引脚都通过一个 330 欧姆电阻连接到 Arduino 的 PWM 引脚。

我们没有展示 NeoPixel 盾的电路图，因为我们只需要将盾连接到 Arduino。现在让我们看看代码。

# 代码

让我们从查看 RGB LED 的代码开始。

# RGB LED

我们将首先定义 Arduino 上的哪些引脚连接到 LED 的 RGB 引脚：

```cpp
#define REDPIN 11
#define BLUEPIN 10
#define GREENPIN 9
```

此代码显示红色引脚连接到 Arduino 11 PWM 引脚，蓝色引脚连接到 Arduino 10 PWM 引脚，绿色引脚连接到 Arduino 9 PWM 引脚。我们将定义一个空宏，让应用程序代码知道我们是否有常见的阳极或阴极 RGB LED。以下代码将做到这一点：

```cpp
#define COMMON_ANODE
```

如果您使用的是常见的阴极 RGB LED，那么请注释或删除此行代码。当我们查看设置 LED 颜色的函数时，我们将看到如何使用它。现在让我们看看`setup()`函数。

```cpp
void setup() {
  pinMode(REDPIN, OUTPUT);
  pinMode(GREENPIN, OUTPUT);
  pinMode(BLUEPIN, OUTPUT);
}
```

`setup()` 函数将设置连接到 LED 上 RGB 引脚的引脚模式为输出。这将允许我们使用 PWM 引脚来设置组成 RGB LED 的三个颜色 LED 的光强度。接下来，我们需要创建一个设置这些颜色的函数。我们将把这个函数命名为 `setColor()`，它将接受三个参数，这些参数将定义每个 RGB LED 的强度，并包含以下代码：

```cpp
void setColor(int red, int green, int blue) {
  #ifdef COMMON_ANODE
  red = 255 - red;
  green = 255 - green;
  blue = 255 - blue;
  #endif
  analogWrite(REDPIN, red);
  analogWrite(GREENPIN, green);
  analogWrite(BLUEPIN, blue);
}
```

这个函数中的代码从 `#ifdef` 语句开始。这个语句表示如果定义了 `COMMON_ANODE` 宏，则执行 `#ifdef` 和 `#endif` 语句之间的代码；否则，跳过该代码。因此，如果我们定义 `COMMON_ANODE` 宏在代码的开头，那么我们将每个参数从 `255` 减去以获得正确的强度。然后我们使用 `analogWrite()` 函数将值写入 RGB 引脚。

在本章的开头，我们解释了 RGB LED 的工作原理是通过调整 RGB LED 内部三个 RGB LED 的强度。如果我们向一个共阴极 LED 写入 `255` 的值，那么 LED 将达到最亮。对于一个共阳极 LED，我们需要写入 `0` 的值来使 LED 最亮。这就是为什么如果定义了 `COMMON_ANODE` 宏，我们就从每个参数的值中减去 `255`。 

在 `loop()` 函数中，我们循环通过几种颜色来演示 LED 如何显示不同的颜色。以下显示了 `loop()` 函数的代码：

```cpp
void loop() {
  setColor(255, 0, 0); // Red
  delay(1000);
  setColor(0, 255, 0); // Green
  delay(1000);
  setColor(0, 0, 255); // Blue
  delay(1000);
  setColor(255, 255, 255); // White
  delay(1000);
  setColor(255, 0, 255); // Purple
  delay(1000);
}
```

在 `loop()` 函数中，我们调用 `setColor()` 函数五次来改变 LED 的颜色。我们显示的颜色有红色、绿色、蓝色、白色和紫色。每次颜色改变后，在显示下一个颜色之前会有一个一秒的暂停。这个暂停是由 `delay()` 函数实现的。

我们在 RGB LED 中显示颜色的方式与点亮一个普通 LED 的方式非常相似，只是我们为三种颜色定义了光强度（亮度）。现在让我们看看 NeoPixel 面板的代码。

# NeoPixel 面板

在我们开始编码之前，我们需要安装 **Adafruit NeoPixel** 库。以下截图显示了应该通过库管理器安装的库。如果你不记得安装库的步骤，请参阅第九章 *环境传感器*：

![图片](img/be910d93-d7d8-46d7-9f51-78daf098172e.png)

我们安装 DHT11 温湿传感器的库的传感器位置。

一旦安装了库，我们需要在代码顶部添加以下行来包含它：

```cpp
#include <Adafruit_NeoPixel.h>
```

当我们使用 Adafruit NeoPixel 库时，我们需要告诉它 NeoPixel 连接到哪个引脚以及连接了多少个 NeoPixel。因此，我们将定义包含这些值的宏：

```cpp
#define SHIELD_PIN 13
#define MAX_PIXELS 40 
```

根据 Keyestudio 盾牌的数据表，盾牌连接到 Arduino 的 13 号引脚，盾牌包含 40 个 NeoPixels；因此，我们在宏中定义这些值。现在我们将使用这些值来初始化`Adafruit_NeoPixel`类的一个实例，如下面的代码所示：

```cpp
Adafruit_NeoPixel pixels = Adafruit_NeoPixel(MAX_PIXELS, SHIELD_PIN, NEO_GRB + NEO_KHZ800);
```

第一个参数是盾牌上的像素数量，第二个参数是 NeoPixels 连接的引脚。最后一个参数是像素类型标志。本例中显示的值是最常见的。以下是一些可能的值：

+   `NEO_KHZ800`: 800 KHz 比特流（大多数 NeoPixel 产品带有 WS2812 LED）

+   `NEO_KHZ400`: 400 KHz (经典 v1（非 v2）FLORA 像素，WS2811 驱动器)

+   `NEO_GRB`: 像素以 GRB 比特流连接（大多数 NeoPixel 产品）

+   `NEO_RGB`: 像素以 RGB 比特流连接（v1 FLORA 像素，非 v2）

在这个例子中，我们将逐个将每个像素转换为特定的颜色。因此，我们需要一个全局变量来指向我们所在的像素，另一个全局变量来定义要使用的颜色。在这个例子中，我们将使用两种颜色并在两者之间切换。以下代码定义了这个全局变量：

```cpp
int num = 0;
boolean color = 0;
```

在`setup()`函数中，我们需要初始化 NeoPixels。以下代码展示了包含初始化 NeoPixels 代码的`setup()`函数：

```cpp
void setup() {
  pixels.begin();
  pixels.show();
  pixels.setBrightness(50);
}
```

`begin()`函数准备 Arduino 上的数据引脚，以便输出到 NeoPixels。`show()`函数将数据推送到 NeoPixels，在这里并不是绝对必要的；我发现，无论何时我们向 NeoPixels 写入任何内容，都包括这个函数是一种好的做法。第三个函数控制像素的亮度。我通常将其设置为 50%，因为 NeoPixels 非常亮。

现在让我们看看将每个像素逐个设置为颜色的`loop()`函数。

```cpp
void loop() { 
  num++; 
  if (num > (MAX_PIXELS -1)) { 
    num = 0; 
    color = !color; 
  } 
  if (color) { 
    pixels.setPixelColor(num, 170, 255, 10); 
  } else { 
    pixels.setPixelColor(num, 10, 255, 170); 
  } 
  pixels.show(); 
  delay(500); 
}
```

在`loop()`函数中，我们首先将`num`变量增加一，然后检查是否到达了最后一个像素。如果我们到达了最后一个像素，我们将`num`变量重置为零，并交换`color`变量。在`color = !color`这一行中，`!`运算符是 NOT 运算符，它使得`color`变量在 true 和 false 之间切换。这是因为 NOT 运算符返回`color`变量当前值的相反数。因此，如果以`color`变量当前为 false 为例，那么`!color`操作将返回 true。

然后，我们使用`setPixelColor()`函数将当前像素设置为两种颜色之一，这取决于`color`变量是 true 还是 false。`setPixelColor()`函数有两种版本。我们在这里看到的版本使用第一个参数作为我们设置的像素编号，然后接下来的三个数字定义了组成我们想要的颜色的红色、绿色和蓝色强度的值。如果我们使用 RGBW NeoPixel，我们还需要定义白色颜色。因此，这个函数将添加一个额外的参数，如下所示：

```cpp
 setPixelColor(n, red, green, blue, white);
```

调用`setPixelColor()`函数的第二种方式是传递两个参数，其中第一个参数是像素编号，第二个参数是一个 32 位数字，它结合了红色、绿色和蓝色值。这个版本的函数看起来像这样：

```cpp
setPixelColor(n, color);
```

颜色值可以从 0 到 16,777,216。

在我们设置像素颜色后，我们接着调用`show()`函数将值推送到像素，然后使用延时函数在代码中插入半秒的暂停。

# 运行项目

如果我们运行 RGB LED 的草图，我们会看到 LED 缓慢地在五种颜色之间循环。NeoPixel 的代码将逐个翻转像素，在两种颜色之间切换。

# 挑战

这将是书中最难的挑战之一。Keyestudio NeoPixel 盾牌有八列像素，每列包含五个像素，像素的编号如下：

![](img/c0bdca92-c921-4780-848e-fd36259f20f3.png)

对于这个挑战，将每一列设置为不同的颜色，并让颜色从左到右在盾牌上旋转。以下是一些帮助你开始的提示。第一个是 Adafruit NeoPixel 库，它有一个名为`Color()`的函数，可以根据三个红色、绿色和蓝色值返回 32 位颜色。因此，你可以使用以下代码将 8 位数字转换为 32 位颜色。

```cpp
uint32_t colorNum(int color) {
  colorPos = 255 - colorPos;
  if(colorPos < 85) {
    return pixels.Color(255 - colorPos * 3, 0, colorPos * 3);
  }
  if(colorPos < 170) {
    colorPos -= 85;
    return pixels.Color(0, colorPos * 3, 255 - colorPos * 3);
  }
  colorPos -= 170;
  return pixels.Color(colorPos * 3, 255 - colorPos * 3, 0);
}
```

然后，我们可以使用以下代码，该代码将列中的所有像素设置为它们的颜色：

```cpp
for (int j=0; j<5; j++) {
  int pixNum = (j*8) + i;
  pixels.setPixelColor(pixNum, colorNum((tmpColorMode * 30) & 255));
}
```

`tmpColorMode`变量是一个从 1 到 8 的数字，将用于选择该列的颜色。这应该为你开始这个挑战提供了基础知识。答案可以在本书的可下载代码中找到。

# 摘要

在本章中，我们学习了 RGB LED 的工作原理，如何使用它们，并探讨了共阳极和共阴极 RGB LED 之间的区别。我们还学习了 WS2812（NeoPixel）的工作原理以及如何使用它。NeoPixel 有多种不同的形式，几乎可以用于需要大量 RGB LED 的任何地方。

在下一章中，我们将探讨如何使用 Arduino 和一个小蜂鸣器来产生声音。
