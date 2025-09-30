# 第七章。音频肾上腺素

这是关于我们一直在工作的 2D 游戏的最后一章。尽管我们的 Robo Racer 2D 游戏几乎完成，但我们还没有包括一个元素来使其成为一个完整的游戏。除非你喜欢无声电影，否则你可能已经注意到我们在这个游戏中没有任何音频。大多数游戏都依赖于音频，我们的也不例外。在本章中，我们将介绍音频以及一些其他维护事项。

+   **音频格式**：了解音频在计算机中的表示方式以及它在游戏中的应用方式非常重要。我们将讨论采样率和比特数，并帮助你理解音频是如何工作的。

+   **音频引擎**：我们需要某种音频引擎来将音频集成到我们的游戏中。我们将讨论 FMOD，这是一个非常流行的引擎，它允许你使用 C++轻松集成音频。

+   **音效**：音效在大多数游戏中扮演着巨大的角色，我们将为我们的游戏添加音效，使其栩栩如生。

+   **音乐**：大多数游戏都使用某种形式的音乐。音乐的处理方式与音效不同，你将学习这两者之间的区别。

+   **最后的维护**：最后一点，对于我们的游戏，我们在这个章节中保留了游戏关闭。我们并不是很好的程序员，因为我们没有正确释放游戏中的对象。我们将学习为什么这样做很重要，以及如何做到这一点。

# 比特和字节

音频本质上是一种模拟体验。声音是通过压缩波在空气中传播并与我们的耳膜相互作用而创造的。直到最近，用于重现音频的技术也是严格属于音频领域的。例如，麦克风记录声音的方式与我们的耳朵相似，通过捕捉空气压力的变化并将其转换为电脉冲。扬声器通过将电信号转换回空气压力的波来实现相反的过程。

相反，计算机是数字的。计算机通过采样音频将音频样本转换为比特和字节。为了简化，让我们考虑一个系统，其中声音波的当前频率（即波移动的速度）被捕获为一个 16 位（2 字节）的数字。结果是，16 位数字可以捕获从 0 到 65,536 的数字范围。每个声音波的样本都必须编码在这个范围内。此外，由于我们实际上每次捕获两个样本（用于立体声），我们需要 4 个字节来捕获每个样本。

下一个重要因素是您多久采样一次声音。音频频率的范围大致从 20 赫兹到 20,000 赫兹（*Hz = 每秒周期数*）。一位名叫尼奎斯特的非常聪明的人发现，为了准确捕捉波形，我们必须以两倍于音频频率的频率采样。这意味着我们每秒至少需要捕捉 40,000 个样本才能准确捕捉声音。相反，我们必须以相同的频率播放声音。这就是为什么光盘上的音频采样频率为 44,100 赫兹。

你现在应该能够看出，处理声音需要大量的磁盘空间和内存。一分钟的音频文件大约需要 10 MB 的存储空间！这意味着，如果我们一次性加载整个音频文件，相同的音频将需要 10 MB 的内存。

你可能会想知道现代游戏是如何运作的。有些游戏的配乐时长以小时计算，而不是分钟。同样，可能会有数百甚至数千个音效，更不用说录音的语音，这些都以音频的形式记录。

## 音效的别称

音频文件可以存储在许多格式中。我们将处理两种在游戏中常用的常见格式：WAV 文件和 MP3 文件。WAV 文件以未压缩的格式存储音频数据。

虽然 WAV 文件可以用于所有音频，但它们通常用于音效。音效通常非常短，通常不到 1 秒。这意味着文件的大小将会相对较小，因为音频文件非常短。

虽然音效通常以 WAV 文件保存，但音乐通常不是。这是因为音乐的长度往往比音效长得多。将一个三到五分钟长的音乐文件加载到内存中会消耗大量的内存。

处理较大音频文件的主要技术有两种。首先，可以使用数据压缩来减小音频文件的大小。提供数据压缩的最常见音频格式之一是 MP3 格式。通过数学技巧，MP3 文件在不牺牲任何音质的情况下，将声音数据存储在更小的空间中。

处理大文件的第二种技术是流式传输。不是将整个声音文件一次性加载到内存中，而是将文件分批次作为连续的数据流发送，然后在游戏中播放。

流式传输有一些局限性。首先，从硬盘或另一个存储设备传输数据比从内存中传输数据慢得多。流式音频可能会出现延迟，这是从触发播放到声音实际播放所需的时间。

对于音效来说，延迟比音乐更重要。这是因为特定的音效通常与游戏中刚刚发生的事情相吻合。如果子弹发射后半秒才听到子弹的声音，那会让人感到不安！另一方面，音乐通常开始后持续几分钟。音乐开始时的轻微延迟通常可以忽略不计。

## 制造噪音

进入一个关于创建声音和音乐的全面课程，当然超出了本书的范围。然而，我确实想给你提供一些资源，帮助你开始学习。

你可能会问的第一个问题是哪里可以找到声音。实际上，有成千上万的网站提供可以在游戏中使用的声音和音乐。许多网站收费，而一些网站提供免费音频。

需要记住的一点是，*免版税*并不一定意味着免费。免版税音频意味着一旦你获得使用音频的许可，你就不必为使用音乐支付任何额外费用。

所以，这里是我的大贴士。我找到的每个网站都为音效和音乐收取一小笔费用。但有一种方法，我找到了免费获取声音的方法，那就是使用**Unity 资产商店**。前往[`unity3d.com`](http://unity3d.com)并安装免费的 Unity 版本。一旦你启动了 Unity，执行以下步骤：

1.  通过点击**Unity 项目向导**中的**创建新项目**标签创建一个新的项目。点击**浏览**并导航到或创建一个文件夹来存储你的项目。然后点击**选择文件夹**。

1.  一旦 Unity 加载了项目，点击**窗口**然后从菜单中选择**资产商店**。

1.  当**资产商店**窗口出现时，在**搜索资产商店**文本框中输入相关搜索词（例如，音乐或 SFX），然后按*Enter*键。

1.  浏览免费资产的结果。点击任何列表以获取更多详细信息。如果你找到了你喜欢的东西，点击**下载**链接。

1.  一旦 Unity 下载了资产，将出现名为**导入包**的屏幕。点击**导入**按钮。

1.  现在，你可以退出 Unity 并导航到你创建新项目的文件夹。然后导航到`Assets`文件夹。从这里，取决于你导入的包的结构，但如果你四处浏览，你应该能够找到音频文件。

    ### 小贴士

    实际上，我们正在使用 Robson Cozendey 提供的名为 Jolly Bot 的音乐作品。[www.cozendey.com](http://www.cozendey.com)。我们还找到了一个很棒的 SFX 包。

1.  现在，你可以将音频文件复制到你的项目中！

    ### 小贴士

    在浏览音频文件时，你可能会遇到一些带有`ogg`扩展名的文件。这是一种类似于 MP3 的常见音频格式。然而，我们将使用的引擎不支持 ogg 文件，因此你需要将它们转换为 MP3 文件。接下来要介绍的 Audacity 将允许你将音频文件从一种格式转换为另一种格式。

你可能会发现你想编辑或混合你的音频文件。或者，你可能需要将你的音频文件从一种格式转换为另一种格式。我发现最适合处理音频的免费工具是**Audacity**，你可以在[`audacity.sourceforge.net/`](http://audacity.sourceforge.net/)下载它。Audacity 是一个功能齐全的音频混音器，它将允许你播放、编辑和转换音频文件。

### 小贴士

要将文件导出为 MP3 格式，你需要在你的系统上安装一份**LAME**。你可以在[`lame.buanzo.org/#lamewindl`](http://lame.buanzo.org/#lamewindl)下载 LAME。

# 提高你的引擎

现在你已经更好地理解了计算机中音频的工作方式，是时候编写一些代码将音频引入你的游戏了。我们通常不直接处理音频。相反，有音频引擎为我们做所有艰苦的工作，其中最受欢迎的一个是**FMOD**。

FMOD 是一个 C 和 C++ API，它允许我们加载、管理和播放音频源。FMOD 对学生和独立项目免费使用，因此它是我们游戏的完美音频引擎。要使用 FMOD，你必须访问 FMOD 网站，下载适当的 API 版本，并将其安装到你的系统上：

1.  要下载 FMOD，请访问[`www.FMOD.org/download/`](http://www.FMOD.org/download/)。

1.  有几个下载选项可供选择。向下滚动到**FMOD Ex 程序员 API**，然后点击 Windows 的**下载**按钮。

1.  你必须找到你刚刚下载的 exe 文件并安装它。记下 FMOD 安装的文件夹。

1.  下载 FMOD 后，你必须将其集成到游戏项目中。首先打开`RoboRacer2D`项目。

### 小贴士

我相信你很想看到**FMOD API**的完整文档。如果你在默认位置安装了 FMOD，你将在`C:\Program Files (x86)\FMOD SoundSystem\FMOD Programmers API Windows\documentation`找到文档。主要文档位于文件 fmodex.chm 中。

现在，是时候设置我们的游戏以使用 FMOD 了。类似于大多数第三方库，连接东西有三个步骤：

1.  访问`.dll`文件。

1.  链接到库。

1.  指向包含文件。

让我们一步步来。

## 访问 FMOD .dll 文件

FMOD 包含几个`.dll`文件，使用正确的文件很重要。以下表格总结了随 FMOD 提供的 dll 文件及其相关的库文件：

| Dll | 描述 | 库 |
| --- | --- | --- |
| `fmodex.dll` | 32 位 FMOD API | `fmodex_vc.lib` |
| `fmodexL.dll` | 32 位带调试日志的 FMOD API | `fmodexL_vc.lib` |
| `fmodex64.dll` | 64 位 FMOD API | `fmodex64_vc.lib` |
| `fmodexL64.dll` | 64 位带调试日志的 FMOD API | `fmodexL64_vc.lib` |

由你决定是否使用库的 32 位或 64 位版本。库的调试版本会将日志信息写入文件。你可以在文档中找到更多信息。

我们将在游戏中使用 32 位文件。我们可以将文件放置在几个地方，但最简单的方法是将`.dll`文件直接复制到我们的项目中：

1.  导航到`C:\Program Files (x86)\FMOD SoundSystem\FMOD Programmers API Windows\api`。

    ### 小贴士

    前面的路径假设你使用了默认的安装位置。如果你选择了另一个位置，你可能需要修改路径。

1.  将`fmodex.dll`复制到包含`RoboRacer2D`源代码的项目文件夹中。

## 链接到库

下一步是告诉 Visual Studio 我们想要访问 FMOD 库。这是通过将库添加到项目属性中完成的：

1.  右键单击项目并选择**属性**。

1.  在**配置属性**下的**链接器**分支中打开，然后点击**输入**。链接到库

1.  在**添加依赖项**条目中点击，然后点击下拉箭头并选择**<编辑…**>。

1.  将`fmodex_vc.lib`添加到依赖项列表中。链接到库

1.  点击**确定**关闭`附加依赖项`窗口。

1.  点击**确定**关闭`属性页`窗口。

现在，我们必须告诉 Visual Studio 在哪里可以找到库：

1.  右键单击项目并选择**属性**。链接到库

1.  在**配置属性**下的**链接器**分支中打开，然后点击**常规**。

1.  在**附加库目录**条目中点击，然后点击下拉箭头并选择**<编辑…**>：链接到库

1.  点击**新行**图标，然后点击出现的省略号（**…**）。

1.  导航到`C:\Program Files (x86)\FMOD SoundSystem\FMOD Programmers API Windows\api\lib`并点击**选择文件夹**。

1.  点击**确定**关闭`附加库目录`窗口。

1.  点击**确定**关闭`属性页`窗口。

## 指向包含文件

无论何时使用第三方代码，通常你都必须在代码中包含 C++头文件。有时，我们只是将相关的头文件复制到项目文件夹中（例如，这就是我们处理`SOIL.h`的方式）。

对于像 FMOD 这样的大型代码库，我们将 Visual Studio 指向头文件安装的位置：

1.  右键单击项目并选择**属性**。指向包含文件

1.  在**配置属性**下的**C/C++**分支中打开，然后点击**常规**。

1.  点击**附加包含目录**条目，然后点击下拉箭头，选择**<编辑…**>。指向包含文件

1.  点击**新行**图标，然后点击出现的省略号（**…**）。

1.  导航到`C:\Program Files (x86)\FMOD SoundSystem\FMOD Programmers API Windows\api\inc`并点击**选择文件夹**。

1.  点击**确定**关闭`附加包含目录`窗口。

1.  点击**确定**关闭`属性页`窗口。

最后一步是将头文件包含到我们的程序中。打开`RoboRacer2D.cpp`并将以下行添加到包含头文件：

```cpp
#include "fmod.hpp"
```

你终于准备好使用我们的音频引擎了！

# 初始化 FMOD

我们需要添加的第一段代码是初始化音频引擎的代码。就像我们必须初始化 OpenGL 一样，这段代码将设置 FMOD 并检查过程中是否有任何错误。

打开`RoboRacer2D.cpp`并将以下代码添加到变量声明区域：

```cpp
FMOD::System* audiomgr;
```

然后添加以下函数：

```cpp
bool InitFmod()
{
  FMOD_RESULT result;
  result = FMOD::System_Create(&audiomgr);
  if (result != FMOD_OK)
  {
    return false;
  }
  result = audiomgr->init(50, FMOD_INIT_NORMAL, NULL);
  if (result != FMOD_OK)
  {
    return false;
  }
  return true;
}
```

此函数创建 FMOD 系统并初始化它：

+   首先，我们定义一个变量来捕获 FMOD 错误代码

+   `System_Create` 调用创建引擎并将结果存储在 `audiomgr`

+   然后我们用 50 个虚拟通道、正常模式初始化 FMOD，

最后，我们需要调用`InitAudio`函数。修改`GameLoop`函数，添加高亮行：

```cpp
void GameLoop(const float p_deltatTime)
{
  if (m_gameState == GameState::GS_Splash)
  {
    InitFmod();
    BuildFont();
    LoadTextures();
    m_gameState = GameState::GS_Loading;
  }
  Update(p_deltatTime);
  Render();
}
```

## 虚拟通道

FMOD 为我们提供的最重要的功能是**虚拟通道**。每个播放的声音都必须有自己的通道来播放。播放音频的物理通道数量因设备而异。早期的声卡一次只能处理两到四个声道的声音。现代声卡可能能够处理八个、十六个甚至更多的声道。

以前，确保在任何时候播放的声音数量不超过硬件通道数量是由开发者负责的。如果游戏触发了一个新的声音，但没有可用通道，那么声音就不会播放。这导致了音频的断断续续和不可预测。

幸运的是，FMOD 为我们处理了所有这些。FMOD 使用虚拟通道，并允许你决定想要使用多少虚拟通道。在幕后，FMOD 决定在任何给定时间需要分配给硬件通道的虚拟通道。

在我们的代码示例中，我们用 50 个虚拟通道初始化了 FMOD。这实际上比我们在这个游戏中会用到的要多得多，但对于一个完整游戏来说这并不夸张。在考虑分配多少虚拟通道时，你应该考虑在任何特定时间将加载多少音频源。这些声音不会同时播放，只是可供播放。

## 通道优先级

FMOD 无法使你的硬件播放比物理声道更多的同时声音，因此你可能想知道为什么你总是会分配比硬件通道更多的虚拟通道。

这个问题的第一个答案是，你实际上不知道玩家实际在系统中玩游戏时会有多少硬件通道可用。虚拟通道的使用消除了你的这个担忧。

第二个答案是，虚拟通道允许你设计你的音频，就像你真的有 50（或 100）个通道可用一样。然后 FMOD 在幕后负责管理这些通道。

那么，如果你的游戏需要播放第九个声音，而只有八个物理通道会发生什么？FMOD 使用优先级系统来决定当前八个通道中哪一个不再需要。例如，第七个通道可能被分配给一个不再播放的声音效果。然后 FMOD 将第七个通道分配给想要播放的新声音。

如果所有物理通道现在都在播放声音，而 FMOD 需要播放一个新的声音，那么它将选择优先级最低的通道，停止在该通道上播放声音，并播放新的声音。决定优先级的因素包括：

+   声音被触发的时间有多久

+   声音是否被设置为连续循环

+   程序员使用 `Channel:setPriority` 或 `Sound::setDefaults` 函数分配的优先级

+   在 3D 声音中，声音距离的远近

+   声音的当前音量

因此，如果您的声音设计超过了同时物理通道的数量，您最终可能会得到丢失的声音。但 FMOD 会尽力限制这种影响。

# 喇叭声和嘟嘟声

想象一下观看一个没有声音的电影。当主要角色沿着小巷跑时，没有脚步声。当他手臂摩擦夹克时，没有摩擦声。当一辆车在他即将撞到他之前停下来时，没有尖叫声。

没有声音的电影会相当无聊，大多数游戏也是如此。声音让游戏栩栩如生。最好的声音设计是玩家实际上并没有意识到有声音设计。这意味着以补充游戏而不令人讨厌的方式制作音效和音乐。

## 音效

音效通常对应于游戏中发生的一些事件或动作。特定的声音通常对应于玩家可以看到的东西，但音效也可能发生在玩家看不到的地方，比如在角落附近。

让我们向游戏中添加第一个音效。我们将保持简单，并添加以下声音：

+   当 Robo 在屏幕上移动时发出的滚动声音

+   当 Robo 跳起或跳下时发出的声音

+   当他与油罐碰撞时发出的欢快声音

+   当他与水瓶碰撞时发出的不太愉快的声音

### 设置声音

我们将首先设置一些变量作为指向我们的声音的指针。打开 `RoboRacer2D.cpp` 并在变量声明部分添加以下代码：

```cpp
FMOD::Sound* sfxWater;
FMOD::Sound* sfxOilcan;
FMOD::Sound* sfxJump; 
FMOD::Sound* sfxMovement;
FMOD::Channel* chMovement;
```

我们有三个指向声音的指针和一个指向通道的指针。我们只需要一个通道指针，因为只有一个声音（`sfxMovement`）将是循环声音。循环声音需要一个持久的通道指针，而一次性声音则不需要。

接下来，我们将加载这些声音。将以下函数添加到 `RoboRacer2D.cpp`：

```cpp
const bool LoadAudio()
{
  FMOD_RESULT result;
  result = audiomgr-> createSound ("resources/oil.wav", FMOD_DEFAULT, 0, &sfxOilcan);
  result = audiomgr-> createSound ("resources/water.wav", FMOD_DEFAULT, 0, &sfxWater);
  result = audiomgr-> createSound ("resources/jump.wav", FMOD_DEFAULT, 0, &sfxJump);
  result = audiomgr->createSound("resources/movement.wav", FMOD_LOOP_NORMAL | FMOD_2D | FMOD_HARDWARE, 0, &sfxMovement);
  result = audiomgr->playSound(FMOD_CHANNEL_FREE, sfxMovement, true, &chMovement);
return true; }
```

### 小贴士

您可以从本书的网站下载这些声音，或者您可以用自己的声音替换它们。只需确保您使用非常短的声音来模拟油、水和跳跃，因为它们旨在快速播放。

此函数将我们的三个音效文件加载到音频系统中。

+   `createSound` 函数为声音分配内存并设置声音的 FMOD 属性。

+   `FMOD_DEFAULT` 设置以下 FMOD 属性：

    +   `FMOD_LOOP_OFF`：声音播放一次，不会循环

    +   `FMOD_2D`：这是一个 2D 声音

    +   `FMOD_HARDWARE`：这使用设备的硬件功能来处理音频

+   结果变量捕获返回值。在生产游戏中，你每次都会测试这个功能，以确保声音已成功加载（我们在这里省略了那些错误检查，以节省空间）。

+   注意我们在移动 SFX 上调用`playSound`。我们将开始这个声音，将其分配给下一个空闲的硬件通道（`FMOD_CHANNEL_FREE`），但告诉 FMOD 立即暂停它（因此`true`参数）。当我们想要播放声音时，我们将播放它，当我们想要停止它时，我们将暂停它。

+   我们将根据需要调用其他 SFX 的`playSound`。由于它们不是循环声音，我们不需要管理它们的暂停状态。

注意，我们将`sfxJump`、`sfxOilcan`和`sfxWater`设置为使用`FMOD_DEFAULT`设置。然而，我们需要`sfxMovement`循环，因此我们必须单独设置其设置标志。

有几个标志可以用来设置声音的属性，并且可以使用 OR 运算符（`|`）来组合标志：

+   `FMOD_HARDWARE`：这使用设备硬件来处理音频。

+   `FMOD_SOFTWARE`：这使用 FMOD 的软件模拟来处理音频（较慢，但可能可以访问设备不支持的功能）。

+   `FMOD_2D`：这是一个 2D 声音。这是我们将在游戏中使用的格式！

+   `FMOD_3D`：这是一个 3D 声音。3D 声音可以放置在 3D 空间中，并似乎具有距离（例如，声音随着距离的增加而变弱）和位置（左、右、前面、后面）。

+   `FMOD_LOOP_OFF`：声音播放一次且不循环。

+   `FMOD_LOOP_NORMAL`：声音播放后重新开始，无限循环。

有许多其他可以设置的标志。请查看 FMOD 文档以获取更多详细信息。

现在我们已经有了加载我们的声音的函数，我们必须将其连接到游戏的初始化中。修改`GameLoop`函数，添加以下突出显示的行：

```cpp
void GameLoop(const float p_deltatTime)
{
  if (m_gameState == GameState::GS_Splash)
  {
    InitFmod();
 LoadAudio();
    BuildFont();
    LoadTextures();
    m_gameState = GameState::GS_Loading;
  }
  Update(p_deltatTime);
  Render();
}
```

### 播放声音

现在，我们需要在适当的时间触发音效。让我们从 Robo 的移动音效开始。基本上，我们希望在 Robo 实际移动时播放这个声音。

我们将修改`ProcessInput`函数中的`CM_STOP`、`CM_LEFT`和`CM_RIGHT`情况。通过插入以下突出显示的行来更新代码：

```cpp
case Input::Command::CM_STOP:
player->SetVelocity(0.0f);
background->SetVelocity(0.0f);
chMovement->setPaused(true);
break;

case Input::Command::CM_LEFT:
if (player == robot_right)
{
  robot_right->IsActive(false);
  robot_right->IsVisible(false);
  robot_left->SetPosition(robot_right->GetPosition());
  robot_left->SetValue(robot_right->GetValue());
}
player = robot_left;
player->IsActive(true);
player->IsVisible(true);
player->SetVelocity(-50.0f);
background->SetVelocity(50.0f);
chMovement->setPaused(false);
break;

case Input::Command::CM_RIGHT:
if (player == robot_left)
{
  robot_left->IsActive(false);
  robot_left->IsVisible(false);
  robot_right->SetPosition(robot_left->GetPosition());
  robot_right->SetValue(robot_left->GetValue());
}
player = robot_right;
player->IsActive(true);
player->IsVisible(true);
player->SetVelocity(50.0f);
background->SetVelocity(-50.0f);
chMovement->setPaused(false);
break;
```

记住，我们已加载`sfxMovement`并将其分配给一个虚拟通道（`chMovement`），然后告诉它以暂停状态开始播放。实际上，在 FMOD 中，我们暂停和播放通道，而不是声音。因此，我们现在只需在 Robo 移动时调用`chMovement->setPaused(true)`，在他不移动时调用`chMovement->setPaused(false)`。

现在，我们需要处理油和水收集。这两个都可以在`CheckCollisions`函数中处理。通过添加以下突出显示的代码行来修改`CheckCollisions`：

```cpp
void CheckCollisions()
{
  if (player->IntersectsCircle(pickup))
  {
 FMOD::Channel* channel;
 audiomgr->playSound(FMOD_CHANNEL_FREE, sfxOilcan, false, &channel);
    pickup->IsVisible(false);
    pickup->IsActive(false);
    player->SetValue(player->GetValue() + pickup->GetValue());
    pickupSpawnTimer = 0.0f;
    pickupsReceived++;
  }

  if (player->IntersectsRect(enemy))
  {
 FMOD::Channel* channel;
 audiomgr->playSound(FMOD_CHANNEL_FREE, sfxWater, false, &channel);
    enemy->IsVisible(false);
    enemy->IsActive(false);
    player->SetValue(player->GetValue() + enemy->GetValue());
    enemySpawnTimer = 0.0f;
  }
}
```

最后，我们将为 Robo 跳跃或下落时添加音效。这些更改将应用于`ProcessInput`函数中的`CM_UP`和`CM_DOWN`情况。使用以下突出显示的行修改现有代码：

```cpp
case Input::Command::CM_UP:
{
  FMOD::Channel* channel;
 audiomgr->playSound(FMOD_CHANNEL_FREE, sfxJump, false, &channel);
  player->Jump(Sprite::SpriteState::UP);
}
break;

case Input::Command::CM_DOWN:
{
  FMOD::Channel* channel;
 audiomgr->playSound(FMOD_CHANNEL_FREE, sfxJump, false, &channel);
  player->Jump(Sprite::SpriteState::DOWN);
}
break;
```

这些音效是单次声音。当它们播放完毕后，我们不需要再担心它们，直到再次播放它们的时候。对于这种类型的音效，我们创建一个通道（`FMOD::channel* channel`），然后使用以下方式调用 `playSound`：

+   `FMOD_CHANNEL_FREE`：这允许 FMOD 选择下一个可用的硬件声音通道。

+   音效指针：`sfxWater` 用于水瓶，`sfxOilcan` 用于油，`sfxJump` 用于跳跃音效。

+   `false`：不要暂停声音！

+   `&channel`：这是虚拟通道句柄。请注意，这只是一个局部变量。对于一次性音效，我们不需要将其存储在任何地方。

就这样！如果你现在玩游戏，四个音效应该会根据我们的设计触发。

## UI 反馈

到目前为止，我们已经创建了音效来响应当前游戏中的事件和动作。音效也用于从用户界面提供反馈。例如，当玩家点击按钮时，应该播放某种音频，以便玩家立即知道点击已被注册。

幸运的是，我们已经捕捉到每次用户点击 UI 按钮的情况，所以每次发生时触发声音很容易。让我们首先添加一个新的声音指针。在 `RoboRacer2D.cpp` 中，将以下行添加到变量声明中：

```cpp
FMOD::Sound* sfxButton;
```

然后在 `LoadAudio` 中添加以下代码：

```cpp
result = audiomgr->createSound("resources/button.wav", FMOD_DEFAULT, 0, &sfxButton);
```

最后，将以下高亮显示的代码行添加到 `ProcessInput` 中的 `CM_UI` 情况：

```cpp
case Input::Command::CM_UI:
FMOD::Channel* channel;
if (pauseButton->IsClicked())
{
 audiomgr->playSound(FMOD_CHANNEL_FREE, sfxButton, false, &channel);
  pauseButton->IsClicked(false);
  pauseButton->IsVisible(false);
  pauseButton->IsActive(false);

  resumeButton->IsClicked(false);
  resumeButton->IsVisible(true);
  resumeButton->IsActive(true);
  m_gameState = GS_Paused;
}

if (resumeButton->IsClicked())
{
 audiomgr->playSound(FMOD_CHANNEL_FREE, sfxButton, false, &channel);
  resumeButton->IsClicked(false);
  resumeButton->IsVisible(false);
  resumeButton->IsActive(false);

  pauseButton->IsClicked(false);
  pauseButton->IsVisible(true);
  pauseButton->IsActive(true);
  m_gameState = GS_Running;
}

if (playButton->IsClicked())
{
 audiomgr->playSound(FMOD_CHANNEL_FREE, sfxButton, false, &channel);
  playButton->IsClicked(false);
  exitButton->IsActive(false);
  playButton->IsActive(false);
  creditsButton->IsActive(false);
  m_gameState = GameState::GS_Running;
}

if (creditsButton->IsClicked())
{
 audiomgr->playSound(FMOD_CHANNEL_FREE, sfxButton, false, &channel);
  creditsButton->IsClicked(false);
  exitButton->IsActive(false);
  playButton->IsActive(false);
  creditsButton->IsActive(false);
  m_gameState = GameState::GS_Credits;
}

if (exitButton->IsClicked())
{
 audiomgr->playSound(FMOD_CHANNEL_FREE, sfxButton, false, &channel);
  playButton->IsClicked(false);
  exitButton->IsActive(false);
  playButton->IsActive(false);
  creditsButton->IsActive(false);
  PostQuitMessage(0);
}

if (menuButton->IsClicked())
{
 audiomgr->playSound(FMOD_CHANNEL_FREE, sfxButton, false, &channel);
  menuButton->IsClicked(false);
  menuButton->IsActive(false);
  m_gameState = GameState::GS_Menu;
}

if (continueButton->IsClicked())
{
 audiomgr->playSound(FMOD_CHANNEL_FREE, sfxButton, false, &channel);
  continueButton->IsClicked(false);
  continueButton->IsActive(false);
  m_gameState = GameState::GS_Running;
}

if (replayButton->IsClicked())
{
 audiomgr->playSound(FMOD_CHANNEL_FREE, sfxButton, false, &channel);
  replayButton->IsClicked(false);
  replayButton->IsActive(false);
  exitButton->IsActive(false);
  RestartGame();
  m_gameState = GameState::GS_Running;
}
break;
```

到目前为止，当你运行游戏时，每次点击按钮你都会听到一个音效。

# 音乐的声音

我们现在转向我们游戏的音频音轨。就像电影配乐一样，在游戏中播放的音乐为游戏设定了基调。许多游戏有巨大的、编排精良的制作，而其他游戏则有合成或 8 位音乐。

正如我们已经讨论过的，音乐文件和音效的处理方式不同。这是因为音效通常是很短的声音，最好以 wav 文件的形式存储。音乐文件通常要长得多，并以 MP3 文件的形式存储，因为数据可以被压缩，占用更少的存储空间和内存。

我们将向我们的游戏添加一条单独的音乐音轨。为了使事情简单，我们将告诉音轨循环播放，以便它在整个游戏中持续运行。

我们首先添加一个声音指针。打开 `RoboRacer2D.cpp` 并在变量声明中添加以下代码行：

```cpp
FMOD::Sound* musBackground;
```

接下来，转到 `LoadAudio` 函数并添加以下行：

```cpp
result = audiomgr->createSound("resources/jollybot.mp3", FMOD_LOOP_NORMAL | FMOD_2D | FMOD_HARDWARE, 0, &musBackground);
FMOD::Channel* channel;
result = audiomgr->playSound(FMOD_CHANNEL_FREE, musBackground, false, &channel);
```

注意，我们使用 `createStream` 而不是 `createSound` 来加载我们的音乐文件。由于音乐比音效长得多，音乐是从存储中流式传输的，而不是直接加载到内存中。

我们希望音轨在游戏开始时启动，所以我们在加载后立即使用 `playSound` 开始播放音乐。

就这么多了！我们的游戏现在通过生动的声音景观得到了增强。

# 打扫房子

我们有一个相当完整的游戏。当然，它不会打破任何记录或使任何人致富，但如果这是你的第一个游戏，那么恭喜你！

我们在某个方面有所疏忽：良好的编程实践要求我们每次创建一个对象后，在使用完毕时都应将其删除。到目前为止，你可能想知道我们是否真的会这样做！好吧，现在就是时候了。

我们在`EndGame`函数中为所有这些操作留了一个占位符。现在，我们将添加必要的代码来正确释放我们的资源。

## 释放精灵

让我们从清理我们的精灵开始。重要的是要记住，当我们移除任何资源时，我们需要确保它也释放了自己的资源。这就是类析构函数的目的。让我们以`Sprite`类为例。打开`Sprite.cpp`，你应该会看到使用以下代码定义的析构函数：

```cpp
Sprite::~Sprite()
{
  for (int i = 0; i < m_textureIndex; i++)
  {
    glDeleteTextures(1, &m_textures[i]);
  }
  delete[] m_textures;
  m_textures = NULL;
}
```

我们首先想要释放`m_textures`数组中的所有纹理。然后我们使用`delete[]`来释放`m_textures`数组。一旦对象被删除，将变量设置为`NULL`也是良好的编程实践。

当我们在精灵对象上调用`delete`时，将调用`Sprite`析构函数。因此，我们需要首先在`EndGame`中添加对为我们的游戏创建的每个精灵的`delete`操作。在`EndGame`函数中添加以下代码行：

```cpp
delete robot_left;
delete robot_right;
delete robot_right_strip;
delete robot_left_strip;
delete background;
delete pickup;
delete enemy;
delete pauseButton;
delete resumeButton;
delete splashScreen;
delete menuScreen;
delete creditsScreen;
delete playButton;
delete creditsButton;
delete exitButton;
delete menuButton;
delete nextLevelScreen;
delete continueButton;
delete gameOverScreen;
delete replayButton;
```

### 小贴士

如果你仔细观察，你会注意到我们没有删除玩家对象。这是因为玩家仅用作指向已创建精灵的指针。换句话说，我们从未使用玩家来创建新的精灵。一个很好的经验法则是，对于每个新创建的对象，应该恰好有一个删除操作。

## 释放输入

我们接下来要关闭的系统是输入系统。首先，让我们完成`Input`析构函数。在`Input`类的析构函数中添加以下高亮代码：

```cpp
Input::~Input()
{
  delete[] m_uiElements;
 m_uiElements = NULL;
}
```

我们必须删除`uiElements`数组，这是一个指向输入系统中的精灵的指针数组。请注意，我们在这里没有删除实际的精灵，因为它们不是由输入系统创建的。

现在，在`EndGame`中添加以下代码行：

```cpp
delete inputManager;
```

## 释放字体

添加以下行以释放我们用于存储字体的显示列表：

```cpp
KillFont();
```

## 释放音频

我们最后的清理工作是音频系统。在`EndGame`中添加以下代码行：

```cpp
sfxWater->release();
sfxOilcan->release();
sfxJump->release();
sfxMovement->release();
sfxButton->release();
musBackground->release();
audiomgr->release(); 
```

恭喜！你的房子已经全部清理干净了。

# 概述

我们在本章中涵盖了大量的内容，在这个过程中，我们完成了我们的 2D 游戏。你了解了一些关于计算机中音频表示的知识。然后我们安装了 FMOD API，并学习了如何将其集成到我们的项目中。最后，我们使用 FMOD 在游戏中设置和播放音效和音乐。

本章完成了我们对 2D 游戏编程的讨论。你应该现在已经清楚，完成一个游戏不仅仅使用 OpenGL 库。记住，OpenGL 是一个渲染库。我们必须编写自己的类来处理输入，并使用第三方类来处理音频。

在下一章，我们开始探索 3D 编程的世界！
