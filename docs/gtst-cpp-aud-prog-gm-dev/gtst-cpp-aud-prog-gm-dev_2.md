# 第二章：音频播放

在本章中，我们将执行音频编程中最基本的两个操作——加载和播放音频文件。这可能看起来不像什么，但已经足够让我们开始将音频添加到我们的游戏中了。

如今有许多不同的音频库可用，如 DirectSound、Core Audio、PortAudio、OpenAL、FMOD 或 Wwise。有些仅在特定平台上可用，而其他一些几乎在任何地方都可以工作。有些是非常低级的，几乎只提供了用户和声卡驱动程序之间的桥梁，而其他一些则提供了高级功能，如 3D 音效或交互式音乐。

对于本书，我们将使用 FMOD，这是由 Firelight Technologies 开发的跨平台音频中间件，非常强大，但易于使用。然而，你应该更专注于所涵盖的概念，而不是 API，因为理解它们将使你更容易适应其他库，因为很多知识是可以互换的。

首先，我们将学习如何安装 FMOD，如何初始化和更新音频系统，以及如何让它播放音频文件。在本章结束时，我们将通过创建一个非常简单的音频管理器类来完成这些任务，它将所有这些任务封装在一个极简的接口后面。

# 理解 FMOD

我选择 FMOD 作为本书的主要原因之一是它包含两个单独的 API——FMOD Ex 程序员 API，用于低级音频播放，以及 FMOD Designer，用于高级数据驱动音频。这将使我们能够在不必使用完全不同的技术的情况下，以不同的抽象级别涵盖游戏音频编程。

### 提示

**下载示例代码**

你可以从[`www.packtpub.com`](http://www.packtpub.com)的帐户中下载你购买的所有 Packt 图书的示例代码文件。如果你在其他地方购买了这本书，你可以访问[`www.packtpub.com/support`](http://www.packtpub.com/support)并注册，文件将直接通过电子邮件发送给你。

除此之外，FMOD 也是一款优秀的软件，对游戏开发者有几个优势：

+   **许可证**：它可以免费用于非商业用途，并且对于商业项目有合理的许可证。

+   **跨平台**：它可以在令人印象深刻的多个平台上运行。你可以在 Windows、Mac、Linux、Android、iOS 上运行它，并且在索尼、微软和任天堂的大多数现代游戏主机上也可以运行。

+   **支持的格式**：它原生支持大量的音频文件格式，这样就不用包含其他外部库和解码器了。

+   **编程语言**：你不仅可以使用 C 和 C++来使用 FMOD，还有其他编程语言的绑定可用，比如 C#和 Python。

+   **流行度**：它非常受欢迎，被广泛认为是目前的行业标准。它被用于游戏如 BioShock，Crysis，Diablo 3，Guitar Hero，Start Craft II 和 World of Warcraft。它还被用来驱动几个流行的游戏引擎，如 Unity3D 和 CryEngine。

+   **特点**：它功能齐全，涵盖了从简单的音频播放、流式传输和 3D 音效到交互式音乐、DSP 效果和低级音频编程的一切。

# 安装 FMOD Ex 程序员 API

首次安装 C++库可能有点令人生畏。好的一面是，一旦你第一次完成了这个过程，对于其他库来说，通常过程是一样的。如果你使用的是 Microsoft Visual Studio，你应该遂循以下步骤：

1.  从[`www.fmod.org`](http://www.fmod.org)下载 FMOD Ex 程序员 API 并将其安装到一个你可以记住的文件夹，比如`C:\FMOD`。

1.  创建一个新的空项目，并至少向其中添加一个`.cpp`文件。然后，在**解决方案资源管理器**上右键单击项目节点，并从列表中选择**属性**。对于接下来的所有步骤，请确保**配置**选项设置为**所有配置**。

1.  导航到**C/C++** | **常规**，并将`C:\FMOD\api\inc`添加到**附加包含目录**列表中（条目用分号分隔）。

1.  导航到**链接器** | **常规**，并将`C:\FMOD\api\lib`添加到**附加库目录**列表中。

1.  导航到**链接器** | **输入**，并将`fmodex_vc.lib`添加到**附加依赖项**列表中。

1.  导航到**生成事件** | **后期生成事件**，并将`xcopy /y "C:\FMOD\api\fmodex.dll" "$(OutDir)"`添加到**命令行**列表中。

1.  从您的代码中包含`<fmod.hpp>`头文件。

# 创建和管理音频系统

FMOD 内发生的一切都由一个名为`FMOD::System`的类管理，我们必须通过`FMOD::System`的`m_Create()`函数实例化它：

```cpp
FMOD::System* system;
FMOD::System_Create(&system);
```

请注意，该函数通过参数返回`system`对象。每当 FMOD 函数需要返回一个值时，您都会看到这种模式，因为它们都将常规返回值保留给错误代码。我们将在稍后讨论错误检查，但现在让我们让音频引擎运行起来。

现在我们已经实例化了一个`system`对象，我们还需要通过调用`init()`方法来初始化它：

```cpp
system->init(100, FMOD_INIT_NORMAL, 0);
```

第一个参数指定要分配的最大通道数。这控制了您能够同时播放多少个声音。您可以为此参数选择任何数字，因为系统在幕后执行一些聪明的优先级管理，并使用可用资源分配通道。第二个和第三个参数自定义了初始化过程，通常可以将它们保留为示例中所示。

我们将使用的许多功能只有在每帧更新`system`对象时才能正常工作。这是通过在游戏循环内调用`update()`方法来完成的：

```cpp
system->update();
```

您还应该记住在游戏结束之前关闭`system`对象，以便它可以处理所有资源。这是通过调用`release()`方法来完成的：

```cpp
system->release();
```

# 加载和流式传输音频文件

FMOD 最伟大的一点是，你可以用一个方法调用加载几乎任何音频文件格式。要将音频文件加载到内存中，请使用`createSound()`方法：

```cpp
FMOD::Sound* sound;
system->createSound("sfx.wav", FMOD_DEFAULT, 0, &sound);
```

要从磁盘流式传输音频文件而无需将其存储在内存中，请使用`createStream()`方法：

```cpp
FMOD::Sound* stream;
system->createStream("song.ogg", FMOD_DEFAULT, 0, &stream);
```

这两种方法都将音频文件的路径作为第一个参数，并通过第四个参数返回一个指向`FMOD::Sound`对象的指针，您可以使用它来播放声音。以前示例中的路径是相对于应用程序路径的。如果您在 Visual Studio 中运行这些示例，请确保将音频文件复制到输出文件夹中（例如，使用后期构建事件，如`xcopy /y "$(ProjectDir)*.ogg" "$(OutDir)"`）。

加载和流式传输之间的选择主要是内存和处理能力之间的权衡。当您加载音频文件时，所有数据都会被解压缩并存储在内存中，这可能会占用大量空间，但计算机可以轻松播放它。另一方面，流式传输几乎不使用任何内存，但计算机必须不断访问磁盘，并即时解码音频数据。另一个区别（至少在 FMOD 中）是，当您流式传输声音时，您一次只能播放一个实例。这种限制存在是因为每个流只有一个解码缓冲区。因此，对于必须同时播放多次的音效，您必须将它们加载到内存中，或者打开多个并发流。作为一个经验法则，流式传输非常适合音乐曲目、语音提示和环境曲目，而大多数音效应该加载到内存中。

第二个和第三个参数允许我们自定义声音的行为。有许多不同的选项可用，但以下列表总结了我们将要使用最多的选项。使用`FMOD_DEFAULT`等同于组合每个类别的第一个选项：

+   `FMOD_LOOP_OFF`和`FMOD_LOOP_NORMAL`：这些模式控制声音是否应该只播放一次，或者在达到结尾时循环播放

+   `FMOD_HARDWARE`和`FMOD_SOFTWARE`：这些模式控制声音是否应该在硬件中混合（性能更好）或软件中混合（更多功能）

+   `FMOD_2D`和`FMOD_3D`：这些模式控制是否使用 3D 声音

我们可以使用按位`OR`运算符组合多个模式（例如，`FMOD_DEFAULT | FMOD_LOOP_NORMAL | FMOD_SOFTWARE`）。我们还可以告诉系统在使用`createSound()`方法时流式传输声音，通过设置`FMOD_CREATESTREAM`标志。实际上，`createStream()`方法只是这样做的一个快捷方式。

当我们不再需要声音（或者游戏结束时），我们应该通过调用声音对象的`release()`方法来处理它。无论音频系统是否也被释放，我们都应该释放我们创建的声音。

```cpp
sound->release();
```

# 播放声音

将声音加载到内存中或准备好进行流式传输后，剩下的就是告诉系统使用`playSound()`方法来播放它们：

```cpp
FMOD::Channel* channel;
system->playSound(FMOD_CHANNEL_FREE, sound, false, &channel);
```

第一个参数选择声音将在哪个通道播放。通常应该让 FMOD 自动处理，通过将`FMOD_CHANNEL_FREE`作为参数传递。

第二个参数是指向要播放的`FMOD::Sound`对象的指针。

第三个参数控制声音是否应该在暂停状态下开始，让您有机会修改一些属性，而这些更改不会被听到。如果您将其设置为 true，您还需要使用下一个参数，以便稍后取消暂停。

第四个参数是一个输出参数，返回`FMOD::Channel`对象的指针，声音将在其中播放。您可以使用此句柄以多种方式控制声音，这将是下一章的主要内容。

如果您不需要对声音进行任何控制，可以忽略最后一个参数，并在其位置传递`0`。这对于非循环的一次性声音很有用。

```cpp
system->playSound(FMOD_CHANNEL_FREE, sound, false, 0);
```

# 检查错误

到目前为止，我们假设每个操作都会顺利进行，没有错误。然而，在实际情况下，有很多事情可能会出错。例如，我们可能会尝试加载一个不存在的音频文件。

为了报告错误，FMOD 中的每个函数和方法都有一个`FMOD_RESULT`类型的返回值，只有当一切顺利时才会等于`FMOD_OK`。用户需要检查这个值并做出相应的反应：

```cpp
FMOD_RESULT result = system->init(100, FMOD_INIT_NORMAL, 0);
if (result != FMOD_OK) {
  // There was an error, do something about it
}
```

首先，了解错误是什么将是有用的。然而，由于`FMOD_RESULT`是一个枚举，如果尝试打印它，您只会看到一个数字。幸运的是，在`fmod_errors.h`头文件中有一个名为`FMOD_ErrorString()`的函数，它将为您提供完整的错误描述。

您可能还想创建一个辅助函数来简化错误检查过程。例如，以下函数将检查错误，将错误描述打印到标准输出，并退出应用程序：

```cpp
#include <iostream>
#include <fmod_errors.h>

void ExitOnError(FMOD_RESULT result) {
  if (result != FMOD_OK) {
    std::cout << FMOD_ErrorString(result) << std::endl;
    exit(-1);
  }
}
```

然后，您可以使用该函数来检查是否有任何应该导致应用程序中止的关键错误：

```cpp
ExitOnError(system->init(100, FMOD_INIT_NORMAL, 0));
```

前面描述的初始化过程也假设一切都会按计划进行，但真正的游戏应该准备好处理任何错误。幸运的是，FMOD 文档中提供了一个模板，向您展示如何编写健壮的初始化序列。这里涵盖的内容有点长，所以我建议您参考文档文件夹中名为`Getting started with FMOD for Windows.pdf`的文件，以获取更多信息。

为了清晰起见，所有的代码示例将继续在没有错误检查的情况下呈现，但在实际项目中，你应该始终检查错误。

# 项目 1 - 构建一个简单的音频管理器

在这个项目中，我们将创建一个`SimpleAudioManager`类，它结合了本章涵盖的所有内容。创建一个仅公开我们需要的操作的底层系统的包装器被称为**外观设计模式**，在保持事情简单的同时非常有用。

由于我们还没有看到如何操作声音，不要指望这个类足够强大，可以用于复杂的游戏。它的主要目的将是让你用非常少的代码加载和播放一次性音效（实际上对于非常简单的游戏可能已经足够了）。

它还会让你摆脱直接处理声音对象（并且需要释放它们）的责任，通过允许你通过文件名引用任何加载的声音。以下是如何使用该类的示例：

```cpp
SimpleAudioManager audio;
audio.Load("explosion.wav");
audio.Play("explosion.wav");
```

从教育角度来看，也许更重要的是，你可以将这个练习作为一种获取一些关于如何调整技术以满足你需求的想法的方式。它还将成为本书后续章节的基础，我们将构建更复杂的系统。

## 类定义

让我们从检查类定义开始：

```cpp
#include <string>
#include <map>
#include <fmod.hpp>

typedef std::map<std::string, FMOD::Sound*> SoundMap;

class SimpleAudioManager {
 public:
  SimpleAudioManager();
  ~SimpleAudioManager();
  void Update(float elapsed);
  void Load(const std::string& path);
  void Stream(const std::string& path);
  void Play(const std::string& path);
 private:
  void LoadOrStream(const std::string& path, bool stream);
  FMOD::System* system;
  SoundMap sounds;
};
```

通过浏览公共类成员列表，应该很容易推断它能做什么：

+   该类可以使用`Load()`方法加载音频文件（给定路径）

+   该类可以使用`Stream()`方法流式传输音频文件（给定路径）

+   该类可以使用`Play()`方法播放音频文件（前提是它们已经被加载或流式传输）

+   还有一个`Update()`方法和一个构造函数/析构函数对来管理声音系统

另一方面，私有类成员可以告诉我们很多关于类内部工作的信息：

+   该类的核心是一个`FMOD::System`实例，负责驱动整个声音引擎。该类在构造函数中初始化声音系统，并在析构函数中释放它。

+   声音存储在一个关联容器中，这允许我们根据文件路径搜索声音。为此，我们将依赖于 C++标准模板库（STL）关联容器之一，`std::map`类，以及用于存储键的`std::string`类。查找字符串键有点低效（例如与整数相比），但对于我们的需求来说应该足够快。将所有声音存储在单个容器中的优势是我们可以轻松地遍历它们并从类析构函数中释放它们。

+   由于加载和流式传输音频文件的代码几乎相同，公共功能已经被提取到一个名为`LoadOrStream()`的私有方法中，`Load()`和`Stream()`将所有工作委托给它。这样可以避免不必要地重复代码。

## 初始化和销毁

现在，让我们逐一实现每个方法。首先是类构造函数，非常简单，因为它唯一需要做的就是初始化`system`对象。

```cpp
SimpleAudioManager::SimpleAudioManager() {
  FMOD::System_Create(&system);
  system->init(100, FMOD_INIT_NORMAL, 0);
}
```

更新更简单，只需要一个方法调用：

```cpp
void SimpleAudioManager::Update(float elapsed) {
  system->update();
}
```

另一方面，析构函数需要负责释放`system`对象，以及创建的所有声音对象。不过，这个过程并不复杂。首先，我们遍历声音的映射，依次释放每一个，并在最后清除映射。如果你以前从未使用过 STL 迭代器，语法可能会显得有点奇怪，但它的意思只是从容器的开头开始，不断前进直到达到末尾。然后我们像往常一样释放`system`对象。

```cpp
SimpleAudioManager::~SimpleAudioManager() {
  // Release every sound object and clear the map
  SoundMap::iterator iter;
  for (iter = sounds.begin(); iter != sounds.end(); ++iter)
    iter->second->release();
  sounds.clear();

  // Release the system object
  system->release();
  system = 0;
}
```

## 加载或流式传输声音

接下来是`Load()`和`Stream()`方法，但让我们先来看一下私有的`LoadOrStream()`方法。这个方法以音频文件的路径作为参数，并检查它是否已经被加载（通过查询声音映射）。如果声音已经被加载，就没有必要再次加载，所以该方法返回。否则，文件将被加载（或流式传输，取决于第二个参数的值），并存储在声音映射中的适当键下。

```cpp
void SimpleAudioManager::LoadOrStream(const std::string& path, bool stream) {
  // Ignore call if sound is already loaded
  if (sounds.find(path) != sounds.end()) return;

  // Load (or stream) file into a sound object
  FMOD::Sound* sound;
  if (stream)
    system->createStream(path.c_str(), FMOD_DEFAULT, 0, &sound);
  else
    system->createSound(path.c_str(), FMOD_DEFAULT, 0, &sound);

  // Store the sound object in the map using the path as key
  sounds.insert(std::make_pair(path, sound));
}
```

有了前面的方法，`Load()`和`Stream()`方法可以轻松实现如下：

```cpp
void SimpleAudioManager::Load(const std::string& path) {
  LoadOrStream(path, false);
}
void SimpleAudioManager::Stream(const std::string& path) {
  LoadOrStream(path, true);
}
```

## 播放声音

最后，还有`Play()`方法，它的工作方式相反。它首先检查声音是否已经加载，如果在地图上找不到声音，则不执行任何操作。否则，使用默认参数播放声音。

```cpp
void SimpleAudioManager::Play(const std::string& path) {
  // Search for a matching sound in the map
  SoundMap::iterator sound = sounds.find(path);

  // Ignore call if no sound was found
  if (sound == sounds.end()) return;

  // Otherwise play the sound
  system->playSound(FMOD_CHANNEL_FREE, sound->second, false, 0);
}
```

我们本可以尝试在找不到声音时自动加载声音。一般来说，这不是一个好主意，因为加载声音是一个昂贵的操作，我们不希望在关键的游戏过程中发生这种情况，因为这可能会减慢游戏速度。相反，我们应该坚持分开加载和播放操作。

## 关于代码示例的说明

尽管这是一本关于音频的书，但所有示例都需要一个运行环境。为了尽可能清晰地保持示例的音频部分，我们还将使用**Simple and Fast Multimedia Library 2.0**（SFML）（[`www.sfml-dev.org`](http://www.sfml-dev.org)）。这个库可以非常容易地处理所有杂项任务，比如窗口创建、定时、图形和用户输入，这些任务在任何游戏中都会找到。

例如，这里有一个使用 SFML 和`SimpleAudioManager`类的完整示例。它创建一个新窗口，加载一个声音，以 60 帧每秒的速度运行游戏循环，并在用户按下空格键时播放声音。

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

# 总结

在本章中，我们已经看到了使用 FMOD 音频引擎的一些优势。我们看到了如何在 Visual Studio 中安装 FMOD Ex 程序员 API，如何初始化、管理和释放 FMOD 音频系统，如何从磁盘加载或流式传输任何类型的音频文件，如何播放先前由 FMOD 加载的声音，如何检查每个 FMOD 函数中的错误，以及如何创建一个简单的音频管理器，它封装了加载和播放音频文件的操作背后的简单接口。
