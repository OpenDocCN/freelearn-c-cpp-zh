# 前言

音频在视频游戏中无疑是我们手头最强大的工具之一，它可以在许多不同的方面发挥作用，比如通过音效提供反馈、通过环境音轨增加沉浸感、通过录制的语音讲述故事，或通过背景音乐传达各种情感。

自早期以来，视频游戏一直在利用声音。例如，1972 年的经典游戏《乒乓球》使用了蜂鸣音效来提供反馈，每当球与物体碰撞时，不同的音高用于区分与墙壁的碰撞、与球拍的碰撞或球离开游戏场地。

另一方面，《太空侵略者》通过逐渐加快歌曲的速度，巧妙地利用了其基本的背景音乐，随着外星人入侵的危险越来越近，从而增强了玩家内心的紧张感。研究表明，没有声音玩游戏的玩家没有感受到同样的紧迫感，他们的心率也没有像打开声音的玩家那样上升。

自那时以来，技术取得了许多进步，使得游戏中的音频得以显著发展。大多数游戏开始使用录制的音频而不是粗糙的合成音调，而新技术如 3D 音频现在允许玩家感觉到声音来自四面八方，并与游戏环境互动。

音乐在视频游戏中也扮演着非常重要的角色。流行的《最终幻想》游戏在情感上的巨大影响归功于植松伸夫创作的宏大、电影般的配乐。系列中最令人难忘的场景如果没有伴随着音乐，将不会是同样的。

许多开发者和作曲家也研究了使音乐与游戏玩法互动的方法。例如，从《猴岛小英雄 2：勒船长的复仇》开始，卢卡斯艺术公司创造的每个图形冒险游戏都使用了一种名为 iMUSE 的自定义交互式音乐系统，它允许主题之间的音乐过渡在玩家从一个房间移动到另一个房间时无缝进行。

甚至有一些游戏直接将音频概念融入到它们的主要游戏机制中，比如玩家必须记忆并在《塞尔达传说：时光之笛》中演奏的歌曲，以及完全围绕声音展开的游戏，最流行的例子是节奏游戏，比如《帕拉帕大冒险》、《舞力全开》或《吉他英雄》。

然而，尽管音频是视频游戏中如此重要的一部分，许多游戏开发书籍都只是粗略地涉及音频编程这个主题。即使是那些专门用一章来讲解音频的书籍，通常也只教授一些非常基础的知识，比如加载和播放音频文件，或者使用过时的音频引擎，而不是现在行业中使用的引擎。此外，其他游戏开发主题，如图形、物理或人工智能往往更吸引初级游戏开发者，学习音频变得不那么重要。

这本书的主要目标是通过使用一种流行且成熟的音频引擎，从几个不同的抽象级别涵盖音频编程，给你一个关于游戏音频编程的速成课程。我希望这种方法能够为你提供足够的知识，以实现大多数视频游戏通常需要的音频功能，并为你打下基础，使你能够追求更高级的主题。

# 这本书涵盖了什么

第一章，“音频概念”，涵盖了一些最重要的音频概念，如声波、模拟和数字音频、多声道音频和音频文件格式。

第二章，“音频播放”，展示了如何使用 FMOD 加载和播放音频文件，以及如何开始创建一个简单的音频管理器类。

第三章，“音频控制”，展示了如何控制声音的播放和参数，以及如何将声音分组到类别并同时控制它们。

第四章，“3D 音频”，涵盖了 3D 音频的最重要概念，比如定位音频、混响、遮挡/遮蔽，以及一些 DSP 效果。

第五章，“智能音频”，提供了使用 FMOD Designer 工具进行高级声音设计的概述，以及如何创建自适应和交互式声音事件和音乐的示例。

第六章，“低级音频”，提供了关于如何在非常低级别上处理音频的基本信息，通过直接操作和编写音频数据。

# 阅读本书所需的内容

阅读本书，您需要以下软件：

+   **C++ IDE**：提供了 Microsoft Visual Studio 的说明，但您也可以使用任何 C++ IDE 或编译器。Visual Studio 的 Express 版本是免费的，可以从微软网站下载。

+   **FMOD Ex**：第 2-4 章和第六章需要，可从[www.fmod.org](http://www.fmod.org)免费下载。

+   **FMOD Designer**：第五章需要。可从[www.fmod.org](http://www.fmod.org)免费下载。

+   **SFML**：网站上的所有代码示例也使用 SFML（2.0 版本）来处理其他任务，比如窗口管理、图形和输入处理。可从[www.sfml-dev.org](http://www.sfml-dev.org)免费下载。

# 本书的受众

本书面向具有少量或没有音频编程经验的 C++游戏开发人员，他们希望快速了解集成音频到游戏中所需的最重要主题。

您需要具备中级的 C++知识才能够理解本书中的代码示例，包括对基本的 C++标准库特性的理解，比如字符串、容器、迭代器和流。同时也建议具备一些游戏编程经验，但这不是必需的。

# 约定

在本书中，您会发现一些不同类型信息的文本样式。以下是一些样式的示例，以及它们的含义解释。

文本中的代码词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 账号显示如下：“注意函数通过参数返回`system`对象。”

代码块设置如下：

```cpp
#include <math.h>

float ChangeOctave(float frequency, float variation) {
  static float octave_ratio = 2.0f;
  return frequency * pow(octave_ratio, variation);
}
float ChangeSemitone(float frequency, float variation) {
  static float semitone_ratio = pow(2.0f, 1.0f / 12.0f);
  return frequency * pow(semitone_ratio, variation);
}
```

当我们希望引起您对代码块的特定部分的注意时，相关行或项目会以粗体显示：

```cpp
#include <SFML/Window.hpp>
#include "SimpleAudioManager.h"

int main() {
  sf::Window window(sf::VideoMode(320, 240), "AudioPlayback");
  sf::Clock clock;

  // Place your initialization logic here
 SimpleAudioManager audio;
 audio.Load("explosion.wav");

  // Start the game loop
  while (window.isOpen()) {
    // Only run approx 60 times per second
    float elapsed = clock.getElapsedTime().asSeconds();
    if (elapsed < 1.0f / 60.0f) continue;
    clock.restart();
    sf::Event event;
    while (window.pollEvent(event)) {
      // Handle window events
      if (event.type == sf::Event::Closed) 
        window.close();

      // Handle user input
      if (event.type == sf::Event::KeyPressed &&
          event.key.code == sf::Keyboard::Space)
 audio.Play("explosion.wav");
    }
    // Place your update and draw logic here
 audio.Update(elapsed);
  }
  // Place your shutdown logic here
  return 0;
}
```

**新术语**和**重要单词**以粗体显示。您在屏幕上看到的单词，比如菜单或对话框中的单词，会以这样的方式出现在文本中：“对于接下来的所有步骤，请确保**配置**选项设置为**所有配置**。”

### 注意

警告或重要提示会以这样的方式出现。

### 提示

提示和技巧会以这种形式出现。
