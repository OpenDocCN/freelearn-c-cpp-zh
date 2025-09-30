# 第十三章。提高你的 C 编程技能和创建库

这是本书的最后一章，也是最先进的，但不是最复杂的。你将通过几个典型的示例学习 C 代码优化，这些示例将使你更进一步，并使你在使用 Arduino 的未来项目中更有能力。我将讨论库以及它们如何提高你代码的可重用性，以节省未来的时间。我将描述一些使用位操作而不是常规操作符以及使用一些内存管理技术来提高代码性能的技巧。然后，我将讨论重新编程 Arduino 芯片本身以及使用外部硬件编程器调试我们的代码。

让我们开始吧。

# 编程库

我已经在第二章中提到了库，*与 C 语言的初次接触*。我们可以将其定义为一组使用特定语言编写的已实现的行为，该语言通过一些接口提供了一些方法，可以通过这些方法调用所有可用的行为。

基本上，库是一些已经编写好并且可以重复使用的代码，我们可以在自己的代码中通过遵循一些规范来使用它们。例如，我们可以引用 Arduino 核心中包含的一些库。从历史上看，其中一些库是独立编写的，随着时间的推移，Arduino 团队以及整个 Arduino 社区将它们纳入不断增长的核心库中，作为原生可用的库。

让我们以 EEPROM 库为例。为了检查与之相关的文件，我们必须在我们的计算机上找到正确的文件夹。例如，在 OS X 上，我们可以浏览`Arduino.app`文件本身的内容。我们可以进入`Contents`/`Resources`/`Java`/`libraries`/中的`EEPROM`文件夹。在这个文件夹中，我们有三个文件和一个名为`examples`的文件夹，它包含所有与 EEPROM 库相关的示例：

![编程库](img/7584_13_001.jpg)

我们计算机上的 EEPROM 库（一个 OS X 系统）

我们有以下文件：

+   `EEPROM.h`，包含库的头文件

+   `EEPROM.cpp`，包含实际的代码

+   `keywords.txt`，包含一些参数来着色库的关键字

由于这些文件在文件夹层次结构中的位置，它们可以作为核心 EEPROM 库的一部分使用。这意味着我们一旦在我们的计算机上安装了 Arduino 环境，就可以包含这个库，而无需下载其他任何东西。

简单的语句`include <EEPROM.h>`将库包含到我们的代码中，并使这个库的所有功能都可以进一步使用。

让我们在这些文件中编写代码。

## 头文件

让我们打开`EEPROM.h`：

![头文件](img/7584_13_002.jpg)

在 Xcode IDE 中显示的 EEPROM.h

在这个文件中，我们可以看到一些以`#`字符开头的预处理器指令。这是我们用来在 Arduino 代码中包含库的同一个指令。在这里，这是一种很好的方法，可以避免重复包含相同的头文件。有时，在编码过程中，我们会包含很多库，在编译时，我们必须检查我们没有重复包含相同的代码。这些指令，尤其是`ifndef`指令意味着：“如果`EEPROM_h`常量尚未定义，则执行以下语句”。

这是一个众所周知的技术，称为**包含保护**。在这项测试之后，我们首先定义`EEPROM_h`常量。如果在我们的代码中我们或某些其他库包含了 EEPROM 库，预处理器就不会在第二次看到这个指令时重新处理以下语句。

我们必须使用`#endif`指令完成`#ifndef`指令。这是头文件中的一个常见块，如果您打开其他库头文件，您会看到它很多次。这个块里包含什么？我们还有一个与 C 整数类型相关的包含：`#include <inttypes.h>`。

Arduino IDE 包含库中所有必需的 C 头文件。正如我们之前提到的，我们可以在固件中使用纯 C 和 C++代码。我们之前没有这样做，因为我们一直在使用的函数和类型已经编码到 Arduino 核心中。但请记住，您可以选择在固件中包含其他纯 C 代码，在本章的最后，我们还将讨论您也可以遵循纯 AVR 处理器类型代码的事实。

现在我们有一个类定义。这是一个 C++特性。在这个类内部，我们声明了两个函数原型：

+   `uint8_t read(int)`

+   `void write(int, uint8_t)`

有一个函数用于读取，它接受一个整数作为参数，并返回一个 8 位无符号整数（即字节）。然后，还有一个函数用于写入，它接受一个整数和一个字节，并返回空值。这些原型指的是在其他`EEPROM.cpp`文件中这些函数的定义。

## 源文件

让我们打开`EEPROM.cpp`文件：

![源文件](img/7584_13_003.jpg)

EEPROM 库的源文件在 Xcode IDE 中显示

文件开始时包含了一些头文件。`avr/eeprom.h`指的是 AVR 类型处理器的 EEPROM 库本身。在这个库示例中，我们只是有一个库，它引用并为我们提供了比原始纯 AVR 代码更好的 Arduino 编程风格接口。这就是为什么我选择了这个库示例。这是最短但最明确的示例，它教会了我们很多。

然后我们包含`Arduino.h`头文件，以便访问 Arduino 语言本身的标凈类型和常量。最后，当然，我们还要包含 EEPROM 库本身的头文件。

在以下语句中，我们定义了这两个函数。它们在其块定义中调用其他函数：

+   `eeprom_read_byte()`

+   `eeprom_write_byte()`

这些函数直接来自 AVR EEPROM 库本身。EEPROM Arduino 库只是 AVR EEPROM 库的一个接口。我们为什么不尝试自己创建一个库呢？

# 创建自己的 LED 数组库

我们将创建一个非常小的库，并用一个包括六个非复用 LED 的基本电路来测试它。

## 将六个 LED 连接到板子上

下面是电路图。它基本上包含六个连接到 Arduino 的 LED：

![将六个 LED 连接到板子上](img/7584_13_005.jpg)

六个 LED 连接到板子上

电路图如下所示：

![将六个 LED 连接到板子上](img/7584_13_006.jpg)

另一个将六个 LED 直接连接到 Arduino 的电路图

我不会讨论电路本身，只是提一下我放入了一个 1 kΩ的电阻。我考虑了最坏的情况，即所有 LED 同时点亮。这将驱动大量的电流，因此这为我们的 Arduino 提供了安全保障。一些作者可能不会使用它。我更愿意让一些 LED 稍微暗一些，以保护我的 Arduino。

## 创建一些漂亮的灯光图案

下面是按照某些模式点亮 LED 的代码，所有这些都是硬编码的。每个图案显示之间都有一个暂停：

```cpp
void setup() {

  for (int i = 2 ; i <= 7 ; i++)
  {
    pinMode(i, OUTPUT);
  }
}

void loop(){

  // switch on everything progressively
  for (int i = 2 ; i <= 7 ; i++)
  {
    digitalWrite(i, HIGH);
    delay(100);
  }

  delay(3000);

  // switch off everything progressively
  for (int i = 7 ; i >=2 ; i--)
  {
    digitalWrite(i, LOW);
    delay(100);
  }

  delay(3000);

  // switch on even LEDS
  for (int i = 2 ; i <= 7 ; i++)
  {
    if ( i % 2 == 0 ) digitalWrite(i, HIGH);
    else digitalWrite(i, LOW);
  }

  delay(3000);

  // switch on odd LEDS
  for (int i = 2 ; i <= 7 ; i++)
  {
    if ( i % 2 != 0 ) digitalWrite(i, HIGH);
    else digitalWrite(i, LOW);
  }

  delay(3000);
}
```

这段代码运行正确。但我们如何让它更优雅，尤其是更易于重用呢？我们可以将`for()`循环块嵌入到函数中。但它们只在这个代码中可用。我们必须通过记住我们设计它们的那个项目来复制和粘贴它们，以便在另一个项目中重用它们。

通过创建一个我们可以反复使用的库，我们可以在未来的编码和数据处理中节省时间。通过一些定期的修改，我们可以达到为特定任务设计的完美模块，它将越来越好，直到不需要再触碰它，因为它比其他任何东西都表现得更好。至少这是我们希望看到的。

## 设计一个小型的 LED 图案库

首先，我们可以在头文件中设计我们函数的原型。让我们把这个库叫做`LEDpatterns`。

### 编写 LEDpatterns.h 头文件

下面是一个可能的头文件示例：

```cpp
/*
  LEDpatterns - Library for making cute LEDs Pattern.
  Created by Julien Bayle, February 10, 2013.
*/
#ifndef LEDpatterns_h
#define LEDpatterns_h

#include "Arduino.h"

class LEDpatterns
{
  public:
    LEDpatterns(int firstPin, int ledsNumber);
    void switchOnAll();
    void switchOffAll();
    void switchEven();
    void switchOdd();
  private:
    int _firstPin;
    int _ledsNumber;
};
#endif
```

我们首先编写我们的 include guards。然后包含 Arduino 库。然后，我们定义一个名为`LEDpatterns`的类，其中包含与类本身同名的构造函数等`public`函数。

我们还有两个与第一个连接 LED 的引脚和与连接 LED 总数相关的内部（`private`）变量。在示例中，LED 必须连续连接。

### 编写 LEDpatterns.cpp 源文件

这是 C++库的源代码：

```cpp
/*
  LEDpatterns.cpp - Library for making cute LEDs Pattern.
 Created by Julien Bayle, February 10, 2013.
 */
#include "Arduino.h"
#include "LEDpatterns.h"

LEDpatterns::LEDpatterns(int firstPin, int ledsNumber)
{
  for (int i = firstPin ; i < ledsNumber + firstPin ; i++)
  {
    pinMode(i, OUTPUT);
  }

  _ledsNumber = ledsNumber;
  _firstPin = firstPin;
}

void LEDpatterns::switchOnAll()
{
  for (int i = _firstPin ; i < _ledsNumber + _firstPin ; i++)
  {
    digitalWrite(i, HIGH);
    delay(100);
  }
}

void LEDpatterns::switchOffAll()
{
  for (int i = _ledsNumber + _firstPin -1 ; i >= _firstPin   ; i--)
  {
    digitalWrite(i, LOW);
    delay(100);
  }
}

void LEDpatterns::switchEven()
{
  for (int i = _firstPin ; i < _ledsNumber + _firstPin ; i++)
  {
    if ( i % 2 == 0 ) digitalWrite(i, HIGH);
    else digitalWrite(i, LOW);
  }
}

void LEDpatterns::switchOdd()
{
  for (int i = _firstPin ; i < _ledsNumber + _firstPin ; i++)
  {
    if ( i % 2 != 0 ) digitalWrite(i, HIGH);
    else digitalWrite(i, LOW);
  }
}
```

在开始时，我们检索所有`include`库。然后我们有构造函数，这是一个与库同名的特殊方法。这是这里的重要点。它接受两个参数。在其主体内部，我们将从第一个到最后的所有引脚（将 LED 视为数字输出）放入其中。然后，我们将构造函数的参数存储在之前在头文件`LEDpatterns.h`中定义的`private`变量中。

我们可以声明所有与第一个示例中创建的函数相关的函数，而不需要库。注意每个函数的`LEDpatterns::`前缀。我不会在这里讨论这种纯类相关语法，但请记住结构。

### 编写 keyword.txt 文件

当我们查看我们的源代码时，如果某些内容能够跳出来而不是融入背景，那就非常有帮助。为了正确地着色与我们新创建的库相关的不同关键字，我们必须使用`keyword.txt`文件。让我们检查一下这个文件：

```cpp
#######################################
# Syntax Coloring Map For Messenger
#######################################

#######################################
# Datatypes (KEYWORD1)
#######################################

LEDpatterns	KEYWORD1

#######################################
# Methods and Functions (KEYWORD2)
#######################################
switchOnAll	KEYWORD2
switchOffAll	KEYWORD2
switchEven	KEYWORD2
switchOdd	KEYWORD2

#######################################
# Instances (KEYWORD2)
#######################################

#######################################
# Constants (LITERAL1)
#######################################
```

在前面的代码中，我们可以看到以下内容：

+   所有跟随`KEYWORD1`的内容都将被染成橙色，通常用于类

+   所有跟随`KEYWORD2`的内容都将被染成棕色，用于函数

+   所有跟随`LITERAL1`的内容都将被染成蓝色，用于常量

使用这些来着色你的代码并使其更易于阅读是非常有用的。

## 使用 LEDpatterns 库

该库位于`Chapter13`的`LEDpatterns`文件夹中，你必须将它放在与其他库相同的正确文件夹中，我们已经这样做了。我们必须重新启动 Arduino IDE 以使库可用。完成之后，你应该能够在菜单**Sketch** | **Import Library**中检查它。现在`LEDpatterns`已经出现在列表中：

![使用 LEDpatterns 库](img/7584_13_007.jpg)

该库是一个贡献的库，因为它不是 Arduino 核心的一部分

现在我们使用这个库来检查新的代码。你可以在`Chapter13`/`LEDLib`文件夹中找到它：

```cpp
#include <LEDpatterns.h>
LEDpatterns ledpattern(2,6);

void setup() {
}

void loop(){

  ledpattern.switchOnAll();
  delay(3000);

  ledpattern.switchOffAll();
  delay(3000);

  ledpattern.switchEven();
  delay(3000);

  ledpattern.switchOdd();
  delay(3000);
}
```

在第一步中，我们包含`LEDpatterns`库。然后，我们创建名为`ledpattern`的`LEDpatterns`实例。我们使用之前设计的带有两个参数的构造函数：

+   第一个 LED 的第一个引脚

+   LED 的总数

`ledpattern`是`LEDpatterns`类的一个实例。它在我们的代码中被引用，如果没有`#include`，它将无法工作。我们已调用这个实例的每个方法。

如果代码看起来更干净，这种设计的真正好处是我们可以在我们的任何项目中重用这个库。如果我们想修改和改进库，我们只需要修改库的头文件和源文件。

# 内存管理

这个部分非常短，但绝对不是不重要。我们必须记住，我们在 Arduino 上有以下三个内存池：

+   闪存（程序空间），其中存储固件

+   **静态随机存取存储器**（**SRAM**），其中草图在运行时创建和操作变量

+   EEPROM 是一个用于存储长期信息的内存空间

与 SRAM 相比，Flash 和 EEPROM 是非易失性的，这意味着即使断电后数据也会持续存在。每个不同的 Arduino 板都有不同数量的内存：

+   ATMega328 (UNO) 具有：

    +   Flash 32k 字节（引导程序占用 0.5k 字节）

    +   SRAM 2k 字节

    +   EEPROM 1k 字节

+   ATMega2560 (MEGA) 具有：

    +   Flash 256k 字节（引导程序占用 8k 字节）

    +   SRAM 8k 字节

    +   EEPROM 4k 字节

一个经典的例子是引用一个字符串的基本声明：

```cpp
char text[] = "I love Arduino because it rocks.";
```

这将占用 32 字节到 SRAM 中。这似乎并不多，但 UNO 只提供了 2048 字节。想象一下，如果你使用了一个大的查找表或大量的文本。以下是一些节省内存的技巧：

+   如果你的项目同时使用 Arduino 和计算机，你可以尝试将一些计算步骤从 Arduino 移动到计算机本身，使 Arduino 只在计算机上触发计算并请求结果，例如。

+   总是使用可能的最小数据类型来存储你需要的数据。例如，如果你需要存储介于 0 和 255 之间的数据，不要使用占用 2 字节的 `int` 类型，而应使用 `byte` 类型

+   如果你使用了一些不会更改的查找表或数据，你可以将它们存储在 Flash 内存中而不是 SRAM 中。你必须使用 `PROGMEM` 关键字来完成此操作。

+   你可以使用 Arduino 板的原生 EEPROM，这将需要编写两个小程序：第一个用于将信息存储在 EEPROM 中，第二个用于使用它。我们在第九章 Making Things Move and Creating Sounds 中使用 PCM 库做到了这一点，*使事物移动和创造声音*。

# 掌握位移操作

C++ 中有两个位移操作符：

+   `<<` 是左移操作符

+   `>>` 是右移操作符

这些在 SRAM 内存中非常有用，并且通常可以优化你的代码。`<<` 可以理解为左操作数乘以 2 的右操作数次幂。

`>>` 与之相同，但类似于除法。位操作的能力通常非常有用，并且可以在许多情况下使你的代码更快。

## 用 2 的倍数进行乘除

让我们使用位移来乘以一个变量。

```cpp
int a = 4;
int b = a << 3;
```

第二行将变量 `a` 乘以 `2` 的三次方，因此 `b` 现在包含 `32`。同样，除法可以按以下方式进行：

```cpp
int a = 12 ;
int b = a >> 2;
```

`b` 包含 `3`，因为 `>> 2` 等于除以 4。使用这些操作符可以使代码更快，因为它们是直接访问二进制操作，而不需要使用 Arduino 核心的任何函数，如 `pow()` 或其他操作符。

## 将多个数据项打包到字节中

例如，而不是使用一个大型的二维表来存储，比如以下显示的位图：

```cpp
const prog_uint8_t BitMap[5][7] = {   
// store in program memory to save RAM         
{1,1,0,0,0,1,1},         
{0,0,1,0,1,0,0},         
{0,0,0,1,0,0,0},         
{0,0,1,0,1,0,0},         
{1,1,0,0,0,1,1}     }; 
```

我们可以使用以下代码：

```cpp
const prog_uint8_t BitMap[5] = {   
// store in program memory to save RAM         
B1100011,         
B0010100,         
B0001000,         
B0010100,         
B1100011     }; 
```

在第一种情况下，每个位图需要 7 x 5 = 35 字节。在第二种情况下，只需要 5 字节。我想你已经刚刚发现了一些重大的事情，不是吗？

## 在控制和端口寄存器中打开/关闭单个位

以下是前一条技巧的直接后果。如果我们想将引脚 8 到 13 设置为输出，我们可以这样做：

```cpp
void setup()     {         
  int pin;         

  for (pin=8; pin <= 13; ++pin) {             
    pinMode (pin, LOW);         
  } 
}
```

但这样做会更好：

```cpp
void setup()     {         
 DDRB = B00111111 ; // DDRB are pins from 8 to 15
}
```

在一次遍历中，我们直接在内存中将整个包配置到一个变量中，并且不需要编译 `pinMode` 函数、结构或变量名。

# 重新编程 Arduino 板

Arduino 本地使用著名的引导加载程序。这为我们通过 USB 上的虚拟串行端口上传固件提供了一种很好的方式。但我们也可能对在没有引导加载程序的情况下继续前进感兴趣。如何以及为什么？首先，这将节省一些闪存内存。它还提供了一种避免在我们打开或重置板子并使其活跃并开始运行之前的小延迟的方法。这需要一个外部编程器。

我可以引用 AVR-ISP、STK500，甚至并行编程器（并行编程器在 [`arduino.cc/en/Hacking/ParallelProgrammer`](http://arduino.cc/en/Hacking/ParallelProgrammer) 中有描述）。你可以在 Sparkfun 电子找到 AVR-ISP。

我在 2013 年的一个名为 The Village 的连接城市的项目中，用这个编程器编程了 Arduino FIO 型板，用于特定的无线应用。

![重新编程 Arduino 板](img/7584_13_008.jpg)

Sparkfun 电子的口袋 AVR 编程器

这个编程器可以通过 2 x 5 连接器连接到 Arduino 板上的 ICSP 端口。

![重新编程 Arduino 板](img/7584_13_009.jpg)

Arduino 的 ICSP 连接器

为了重新编程 Arduino 的处理器，我们首先必须关闭 Arduino IDE，然后检查首选项文件（Mac 上的 `preferences.txt`，位于 `Arduino.app` 包内的 `Contents`/`Resources`/`Java`/`lib` 中）。在 Windows 7 PC 和更高版本上，此文件位于：`c:\Users\<USERNAME>\AppData\Local\Arduino\preferences.txt`。在 Linux 上位于：`~/arduino/preferences.ard`。

我们必须将最初设置为引导加载程序的 `upload.using` 值更改为适合你的编程器的正确标识符。这可以在 OS X 的 Arduino 应用程序包内容中找到，或者在 Windows 的 Arduino 文件夹中。例如，如果你显示 `Arduino.app` 内容，你可以找到这个文件：`Arduino.app/Contents/Resources/Java/hardware/arduino/programmers.txt`。

然后，我们可以启动 Arduino IDE，使用我们的编程器上传草图。要恢复到正常的引导加载程序行为，我们首先必须重新上传适合我们硬件的引导加载程序。然后，我们必须更改 `preferences.txt` 文件，它将像初始板一样工作。

# 摘要

在这一章中，我们学习了更多关于设计库的知识，我们现在能够以不同的方式设计我们的项目，考虑到未来项目中代码或代码部分的复用性。这可以节省时间，同时也提高了可读性。

我们也可以通过使用现有的库，通过修改它们，使它们符合我们的需求，来探索开源世界。这是一个真正开放的世界，我们刚刚迈出了第一步。

# 结论

我们已经到了这本书的结尾。你可能已经读完了所有的内容，也用你自己的硬件测试了一些代码片段，我相信你现在能够想象出你未来的高级项目了。

我想感谢你如此专注和感兴趣。我知道你现在几乎和我处于同一条船上，你想要学习更多，测试更多，检查和使用新技术来实现你疯狂的项目。我想说最后一件事：去做，现在就去做！

在大多数情况下，人们在开始之前，仅仅想象到将要面对的大量工作就会感到害怕。但你要相信我，不要过多地考虑细节或优化。试着做一些简单的东西，一些能工作起来的东西。然后你会有方法去优化和改进它。

最后一条建议给你：不要想太多，多做一些。我见过太多的人因为想要先想、想、想，而不是先开始做，结果项目没有完成。

保重，继续探索！
