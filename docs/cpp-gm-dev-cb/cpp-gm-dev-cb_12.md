# 第十二章：游戏开发中的音频

在本章中，将介绍以下教程：

+   安装 FMOD

+   添加背景音乐

+   添加音效

+   创建音效管理器

+   处理多个音频文件名

# 介绍

游戏开发中最重要的方面之一是音频编程。然而，令人奇怪的是，它也是游戏开发中最被忽视和低估的部分之一。要理解音频在游戏中的影响，可以尝试玩一款带有声音的游戏，比如*反恐精英*或*雷神*，然后再尝试没有声音的游戏。这会产生巨大的影响。如果音频编程没有正确完成，可能会导致游戏崩溃和许多其他问题。

因此，学习正确的音频编程方式非常重要。大多数引擎都会有内置的声音组件。对于其他引擎，我们需要添加音频组件。在本章中，我们将介绍最流行的声音引擎之一。我们还将看看如何将 SDL 集成到我们的 C++代码中，以播放音频和音效。

# 安装 FMOD

开始的第一件事是安装 FMOD。这是最流行的音频引擎之一，几乎所有现代游戏引擎都在使用。它也可以添加到您选择的任何游戏引擎中。另一个流行的音频引擎叫做**Wwise**。这用于集成控制台编程的音频，例如在 PS4 上。

## 准备工作

要完成本教程，您需要一台运行 Windows 的计算机。

## 操作方法…

在本教程中，我们将看到不同类型的源控制可用于我们：

1.  访问[`www.fmod.org/`](http://www.fmod.org/)。

1.  要下载 FMOD，请访问[`www.fmod.org/download/`](http://www.fmod.org/download/)。

有一个编辑音频文件的创作工具。但是，我们应该下载 FMOD Studio 程序员 API 和低级程序员 API。

它还为所有现代引擎提供插件，如 Cocos2d-x、Unreal Engine 和 Unity3D。

## 工作原理…

FMOD 是一个低级 API，因此它提供了回调，帮助我们使用 FMOD 的接口来播放声音，暂停声音，以及执行许多其他操作。因为我们有源文件，我们可以构建库并在我们自己的引擎中使用它。FMOD 还为 Unity3D 提供了 API，这意味着代码也暴露给 C#，使其在 Unity3D 中更容易使用。

# 添加背景音乐

如果游戏没有背景音乐，就会显得不完整。因此，非常重要的是我们将播放音乐的方式集成到我们的 C++引擎中。有各种各样的方法可以做到这一点。我们将使用 SDL 在游戏中播放音乐。

## 准备工作

您需要一台运行 Windows 的计算机和一个可用的 Visual Studio 副本。还需要 SDL 库。

## 操作方法…

在本教程中，我们将了解播放背景音乐有多容易：

1.  添加一个名为`Source.cpp`的源文件。

1.  将以下代码添加到其中：

```cpp
#include <iostream>
#include "../AudioDataHandler.h"

#include "../lib/SDL2/include/SDL2/SDL.h"

#include "iaudiodevice.hpp"
#include "iaudiocontext.hpp"
#include "audioobject.hpp"

#include "sdl/sdlaudiodevice.hpp"
#include "sdl/sdlaudiocontext.hpp"

#define FILE_PATH "./res/audio/testClip.wav"

int main(int argc, char** argv)
{
  SDL_Init(SDL_INIT_AUDIO);

  IAudioDevice* device = new SDLAudioDevice();
  IAudioContext* context = new SDLAudioContext();

  IAudioData* data = device->CreateAudioFromFile(FILE_PATH);

  SampleInfo info;
  info.volume = 1.0;
  info.pitch = 0.7298149802137;

  AudioObject sound(info, data);
  sound.SetPos(0.0);

  char in = 0;
  while(in != 'q')
  {
    std::cin >> in;
    switch(in)
    {
      case 'a':
        context->PlayAudio(sound);
        break;
      case 's':
        context->PauseAudio(sound);
        break;
      case 'd':
        context->StopAudio(sound);
        break;
    }
  }

  device->ReleaseAudio(data);
  delete context;
  delete device;

  SDL_Quit();
  return 0;
}

int main()
{
  AudioDataHandler _audioData;
  cout<<_audioData.GetAudio(AudioDataHandler::BACKGROUND);
}
```

## 工作原理…

在这个例子中，我们正在为我们的游戏播放背景音乐。我们需要创建一个接口作为现有 SDL 音频库的包装器。接口还可以提供一个基类以便将来派生。我们需要`SDLAudioDevice`，这是播放音乐的主处理对象。除此之外，我们还创建了一个指向音频数据对象的指针，它可以从提供的文件路径创建音频。设备处理对象有一个内置函数叫做`CreateAudioFromFile`，可以帮助我们完成这个过程。最后，我们有一个音频上下文类，它有播放、暂停和停止音频的功能。每个函数都以音频对象作为引用，用于对我们的音频文件执行操作。

# 添加音效

音效是向游戏添加一些紧张感或成就感的好方法。播放、暂停和停止音效的操作方式与我们在上一个示例中看到的背景音乐相同。然而，我们可以通过控制它们的位置、音量和音调来为声音文件增加一些变化。

## 准备工作

你需要一台正常工作的 Windows 机器。

## 如何做…

添加一个名为`Source.cpp`的源文件，并将以下代码添加到其中：

```cpp
struct SampleInfo
{
  double volume;
  double pitch;
};

SampleInfo info;
info.volume = 1.0;
info.pitch = 0.7298149802137;

AudioObject sound(info, data);
sound.SetPos(0.0);
```

## 它是如何工作的…

在这个示例中，我们只关注游戏中涉及修改声音文件音调、音量和位置的部分。这三个属性可以被视为声音文件的属性，但还有其他属性。因此，首先要做的是创建一个结构。结构用于存储声音的所有属性。我们只需要在需要时填充结构的值。最后，我们创建一个音频对象，并将`SampleInfo`结构作为对象的参数之一传递进去。构造函数然后初始化声音具有这些属性。因为我们将属性附加到对象上，这意味着我们也可以在运行时操纵它们，并在需要时动态降低音量。音调和其他属性也可以以同样的方式进行操纵。

# 创建音效管理器

尽管不是最佳实践之一，但处理音频的最常见方法之一是创建一个管理器类。管理器类应确保整个游戏中只有一个音频组件，控制要播放、暂停、循环等哪种声音。虽然有其他编写管理器类的方法，但这是最标准的做法。

## 准备工作

对于这个示例，你需要一台 Windows 机器和 Visual Studio。

## 如何做…

在这个示例中，我们将了解如何使用以下代码片段轻松添加音效管理器：

```cpp
#pragma once
#include <iostream>
#include "../lib/SDL2/include/SDL2/SDL.h"

#include "iaudiodevice.hpp"
#include "iaudiocontext.hpp"
#include "audioobject.hpp"

#include "sdl/sdlaudiodevice.hpp"
#include "sdl/sdlaudiocontext.hpp"

#define FILE_PATH "./res/audio/testClip.wav"

class GlobalAudioClass
{
private:

  AudioObject* _audObj;
  IAudioDevice* device = new SDLAudioDevice();
  IAudioContext* context = new SDLAudioContext();

  IAudioData* data = device->CreateAudioFromFile(FILE_PATH);

  SampleInfo info;

  static GlobalAudioClass *s_instance;

  GlobalAudioClass()
  {
    info.volume = 1.0;
   info.pitch = 0.7298149802137;
    _audObj = new AudioObject(info,data);
  }
  ~GlobalAudioClass()
  {
    //Delete all the pointers here
  }
public:
  AudioObject* get_value()
  {
    return _audObj;
  }
  void set_value(AudioObject* obj)
  {
    _audObj = obj;
  }
  static GlobalAudioClass *instance()
  {
    if (!s_instance)
      s_instance = new GlobalAudioClass();
    return s_instance;
  }
};

// Allocating and initializing GlobalAudioClass's
// static data member.  The pointer is being
// allocated - not the object inself.
GlobalAudioClass *GlobalAudioClass::s_instance = 0;
```

## 它是如何工作的…

在这个示例中，我们编写了一个单例类来实现音频管理器。单例类具有所有必要的`sdl`头文件和其他播放声音所需的设备和上下文对象。所有这些都是私有的，因此无法从其他类中访问。我们还创建了一个指向该类的静态指针，并将构造函数也设为私有。如果我们需要这个音频管理器的实例，我们必须使用静态的`GlobalAudioClass *instance()`函数。该函数会自动检查是否已经创建了一个实例，然后返回该实例，或者创建一个新的实例。因此，管理器类始终只存在一个实例。我们还可以使用管理器来设置和获取声音文件的数据，例如设置声音文件的路径。

# 处理多个声音文件名称

在游戏中，不会只有一个声音文件，而是多个声音文件需要处理。每个文件都有不同的名称、类型和位置。因此，单独定义所有这些并不明智。虽然这样做可以工作，但如果游戏中有超过 20 个音效，那么编码将会非常混乱，因此需要对代码进行轻微改进。

## 准备工作

在这个示例中，你需要一台 Windows 机器和安装了 SVN 客户端的版本化项目。

## 如何做…

在这个示例中，你将看到处理多个声音文件名称有多么容易。你只需要添加一个名为`Source.cpp`的源文件。将以下代码添加到其中：

```cpp
#pragma once

#include <string>
using namespace std;

class AudioDataHandler
{
public:
  AudioDataHandler();
  ~AudioDataHandler();
  string GetAudio(int data) // Set one of the enum values here from the driver program
  {
    return Files[data];
  }

  enum AUDIO
  {
    NONE=0,
    BACKGROUND,
    BATTLE,
    UI
  };
private:
  string Files[] =
  {
    "",
    "Hello.wav",
    "Battlenn.wav",
    "Click.wav"
  }

};

int main()
{
  AudioDataHandler _audioData;
  cout<<_audioData.GetAudio(AudioDataHandler::BACKGROUND);
}
```

## 它是如何工作的…

在这个例子中，我们创建了一个音频数据处理类。该类有一个`enum`，其中存储了所有声音的逻辑名称，例如`battle_music`，`background_music`等。我们还有一个字符串数组，其中存储了声音文件的实际名称。顺序很重要，它必须与我们编写的`enum`的顺序相匹配。现在这个`enum`被创建了，我们可以创建这个类的对象，并设置和获取音频文件名。`enum`被存储为整数，并默认从`0`开始，名称作为字符串数组的索引。所以`Files[AudioDataHandler::Background]`实际上是`Files[1]`，即`Hello.wav`，因此将播放正确的文件。这是一种非常整洁的组织音频数据文件的方式。在游戏中处理音频的另一种方式是在 XML 或 JSON 文件中具有音频文件的名称和其位置的属性，并有一个读取器解析这些信息，然后以与我们相同的方式填充数组。这样，代码就变得非常数据驱动，因为设计师或音频工程师可以只更改 XML 或 JSON 文件的值，而无需对代码进行任何更改。
