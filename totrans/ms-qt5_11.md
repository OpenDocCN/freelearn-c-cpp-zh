# 第十一章. 与序列化一起玩乐

上一章充满了线程、套接字和工作者的内容。我们希望你的小兵们一直在努力工作。在这一章中，我们将关注使用 Qt 的序列化。你将学习如何使用灵活的系统以多种格式序列化数据。示例项目将是一个虚拟鼓机，你可以创作自己的鼓点，录制、播放、保存和重新加载它。你的鼓点可能非常棒，以至于你想与他人分享：你现在可以以各种格式做到了。

本章将涵盖以下主题：

+   如何构建一个播放和录制声音的应用程序架构

+   `QVariant` 类及其内部机制

+   一个灵活的序列化系统

+   JSON 序列化

+   XML 序列化

+   二进制序列化

+   Qt 多媒体框架

+   使用 Qt 处理拖放

+   从键盘触发按钮

# 构建鼓机项目

如往常一样，在深入代码之前，让我们研究一下项目的结构。项目的目标是能够：

+   从鼓机播放和录制声音轨道

+   将此轨道保存到文件中并加载以播放

为了播放声音，我们将布局四个大按钮，点击（或键盘事件）时将播放特定的鼓声：一个踢鼓、一个军鼓、一个高脚鼓和一个钹。这些声音将由应用程序加载的 `.wav` 文件。用户将能够录制他的声音序列并重新播放。

对于序列化部分，我们不仅希望将轨道保存到单个文件格式，我们更愿意做三个：

+   **JSON**（**JavaScript 对象表示法**）

+   **XML**（**可扩展标记语言**）

+   **二进制**

不仅涵盖三种格式更有趣，而且也给了我们了解每种格式的优缺点以及它们如何在 Qt 框架中定位的机会。我们将要实施的架构将力求灵活以应对未来的演变。你永远不知道一个项目会如何发展！

类的组织结构如下：

![构建鼓机项目架构](img/image00441.jpeg)

让我们回顾一下这些类的作用：

+   `SoundEvent` 类是轨道的基本构建块。它是一个简单的类，包含 `timestamp`（声音播放的时间）和 `soundId` 变量（播放了什么声音）。

+   `Track` 类包含一个 `SoundEvents` 列表、一个持续时间和一个状态（播放、录制、停止）。每次用户播放声音时，都会创建一个 `SoundEvent` 类并将其添加到 `Track` 类。

+   `PlaybackWorker` 类是一个在单独线程中运行的工人类。它负责遍历 `Track` 类的 `soundEvents` 并在其 `timestamp` 达到时触发适当的音效。

+   `Serializable` 类是一个接口，每个想要被序列化的类都必须实现（在我们的案例中：`SoundEvent` 和 `Track`）。

+   `Serializer` 类是一个接口，每个特定格式的实现类都必须实现它。

+   `JsonSerializer`、`XmlSerializer` 和 `BinarySerializer` 是 `Serializer` 类的子类，它们执行特定格式的序列化和反序列化任务。

+   `SoundEffectWidget` 类是包含播放单个声音信息的窗口小部件。它显示我们四个声音中的一个按钮。它还拥有一个 `QSoundEffect` 类，该类将声音发送到音频卡。

+   `MainWindow` 类将一切整合在一起。它拥有 `Track` 类，生成 `PlaybackWorker` 线程，并触发序列化和反序列化。

输出格式应该易于更换。为了实现这一点，我们将依赖一个修改过的桥接设计模式，这将允许 `Serializable` 和 `Serializer` 类独立演进。

整个项目围绕模块之间独立性的概念展开。它甚至到了在播放过程中可以即时替换声音的程度。比如说，你正在听你令人难以置信的节奏，并想尝试另一个鼓声。你将能够通过简单地将 `.wav` 文件拖放到持有鼓声的 `SoundEffectWidget` 类上来替换它。

# 创建鼓点轨道

让我们系好安全带，开始这个项目！创建一个新的 **Qt Widgets Application** 项目，命名为 `ch11-drum-machine`。像往常一样，在 `ch11-drum-machine.pro` 中添加 `CONFIG += c++14`。

现在创建一个新的 C++ 类，命名为 `SoundEvent`。以下是 `SoundEvent.h` 中移除了函数的部分：

```cpp
#include <QtGlobal> 

class SoundEvent 
{ 

public: 
    SoundEvent(qint64 timestamp = 0, int soundId = 0); 
    ~SoundEvent(); 

    qint64 timestamp; 
    int soundId; 
}; 

```

这个类只包含两个公共成员：

+   `timestamp`：一个包含从轨道开始以来 `SoundEvent` 当前时间的 `qint64` (`long long` 类型)，单位为毫秒。

+   `soundId`：播放过的声音的 ID

在录制模式下，每次用户播放声音时，都会创建一个带有适当数据的 `SoundEvent`。`SoundEvent.cpp` 文件非常无聊，所以我们不会把它强加给你。

下一个要构建的类是 `Track`。再次创建一个新的 C++ 类。让我们回顾一下 `Track.h` 中的成员：

```cpp
#include <QObject> 
#include <QVector> 
#include <QElapsedTimer> 

#include "SoundEvent.h" 

class Track : public QObject 
{ 
    Q_OBJECT 
public: 
    enum class State { 
        STOPPED, 
        PLAYING, 
        RECORDING, 
    }; 

    explicit Track(QObject *parent = 0); 
    ~Track(); 

private: 
    qint64 mDuration; 
       std::vector<std::unique_ptr<SoundEvent>> mSoundEvents; 
    QElapsedTimer mTimer; 
    State mState; 
    State mPreviousState; 
}; 

```

我们现在可以详细地介绍它们：

+   `mDuration`：这个变量持有 `Track` 类的持续时间。当开始录制时，这个成员被重置为 0，并在录制停止时更新。

+   `mSoundEvents`：这个变量是给定 `Track` 的 `SoundEvents` 列表。正如 `unique_ptr` 语义所表明的，`Track` 是声音事件的拥有者。

+   `mTimer`：每次播放或录制 `Track` 时，这个变量都会启动。

+   `mState`：这个变量是 `Track` 类的当前 `State`，它可以有三个可能的值：`STOPPED`、`PLAYING`、`RECORDING`。

+   `mPreviousState`：这个变量是 `Track` 的上一个 `State`。当你想知道在新的 `STOPPED` 状态上要执行什么操作时，这很有用。如果 `mPreviousState` 处于 `PLAYING` 状态，我们将不得不停止播放。

`Track`类是项目业务逻辑的核心。它持有`mState`，这是整个应用程序的状态。其内容将在播放你令人惊叹的音乐表演时被读取，并将其序列化到文件中。

让我们用函数丰富`Track.h`：

```cpp
class Track : public QObject 
{ 
    Q_OBJECT 
public: 
    ... 
    qint64 duration() const; 
    State state() const; 
    State previousState() const; 
    quint64 elapsedTime() const; 
    const std::vector<std::unique_ptr<SoundEvent>>& soundEvents() const; 

signals: 
    void stateChanged(State state); 

public slots: 
    void play(); 
    void record(); 
    void stop(); 
    void addSoundEvent(int soundEventId); 

private: 
    void clear(); 
    void setState(State state); 

private: 
    ... 
}; 

```

我们将跳过简单的获取器，专注于重要的函数：

+   `elapsedTime()`：这个函数返回`mTimer.elapsed()`的值。

+   `soundEvents()`：这个函数是一个更复杂的获取器。`Track`类是`mSoundEvents`内容的拥有者，我们确实想强制执行这一点。为此，获取器返回`mSoundEvents`的`const &`。

+   `stateChanged()`：当`mState`值更新时，会发出这个函数。新的`State`作为参数传递。

+   `play()`：这个函数是一个槽，开始播放`Track`。这种播放完全是逻辑上的，真正的播放将由`PlaybackWorker`触发。

+   `record()`：这个函数是一个槽，开始`Track`的录制状态。

+   `stop()`：这个函数是一个槽，停止当前开始或录制状态。

+   `addSoundEvent()`：这个函数使用给定的`soundId`创建一个新的`SoundEvent`并将其添加到`mSoundEvents`。

+   `clear()`：这个函数重置`Track`的内容：它清除`mSoundEvents`并将`mDuration`设置为`0`。

+   `setState()`：这个函数是一个私有辅助函数，用于更新`mState`、`mPreviousState`并发出`stateChanged()`信号。

现在已经覆盖了头文件，我们可以研究`Track.cpp`中的有趣部分：

```cpp
void Track::play() 
{ 
    setState(State::PLAYING); 
    mTimer.start(); 
} 

```

调用`Track.play()`只是将状态更新为`PLAYING`并启动`mTimer`。`Track`类不包含与 Qt 多媒体 API 相关的任何内容；它仅限于一个进化的数据持有者（因为它还管理状态）。

现在来看`record()`函数，它带来了很多惊喜：

```cpp
void Track::record() 
{ 
    clearSoundEvents(); 
    setState(State::RECORDING); 
    mTimer.start(); 
} 

```

它首先清除数据，将状态设置为`RECORDING`，并启动`mTimer`。现在考虑`stop()`，这是一个轻微的变化：

```cpp
void Track::stop() 
{ 
    if (mState == State::RECORDING) { 
        mDuration = mTimer.elapsed(); 
    } 
    setState(State::STOPPED); 
} 

```

如果我们在`RECORDING`状态下停止，`mDuration`将被更新。这里没有什么特别之处。我们看到了三次`setState()`调用，但没有看到其主体：

```cpp
void Track::setState(Track::State state) 
{ 
    mPreviousState = mState; 
    mState = state; 
    emit stateChanged(mState); 
} 

```

在更新之前，当前`mState`的值存储在`mPreviousState`中。最后，使用新值发出`stateChanged()`。

`Track`的状态系统被完全覆盖。最后缺失的部分是`SoundEvents`的交互。我们可以从`addSoundEvent()`片段开始：

```cpp
void Track::addSoundEvent(int soundEventId) 
{ 
    if (mState != State::RECORDING) { 
        return; 
    } 
    mSoundEvents.push_back(make_unique<SoundEvent>( 
                               mTimer.elapsed(), 
                               soundEventId)); 
} 

```

只有在我们处于`RECORDING`状态时，才会创建`soundEvent`。之后，将带有当前`mTimer`的流逝时间和传递的`soundEventId`的`SoundEvent`添加到`mSoundEvents`。

现在来看`clear()`函数：

```cpp
void Track::clear() 
{ 
    mSoundEvents.clear(); 
    mDuration = 0; 
} 

```

因为我们在`mSoundEvents`中使用`unique_ptr<SoundEvent>`，所以`mSoundEvents.clear()`函数足以清空向量并删除每个`SoundEvent`。这减少了你需要担心智能指针的事情。

`SoundEvent`和`Track`是包含你未来节奏信息的基类。我们将看到负责读取这些数据以播放的类：`PlaybackWorker`。

创建一个新的 C++类，并像这样更新`PlaybackWorker.h`：

```cpp
#include <QObject> 
#include <QAtomicInteger> 

class Track; 

class PlaybackWorker : public QObject 
{ 
    Q_OBJECT 
public: 
    explicit PlaybackWorker(const Track& track, QObject *parent = 0); 

signals: 
    void playSound(int soundId); 
    void trackFinished(); 

public slots: 
    void play(); 
    void stop(); 

private: 
    const Track& mTrack; 
    QAtomicInteger<bool> mIsPlaying; 
}; 

```

`PlaybackWorker`类将在不同的线程中运行。如果你的记忆需要刷新，请回到第九章，*使用多线程保持你的理智*。它的作用是遍历`Track`类的内容以触发声音。让我们分解这个头文件：

+   `mTrack`: 这个函数是`PlaybackWorker`正在工作的`Track`类的引用。它作为`const`引用传递给构造函数。有了这个信息，你已经知道`PlaybackWorker`不能以任何方式修改`mTrack`。

+   `mIsPlaying`: 这个函数是一个标志，用于能够从另一个线程停止工作。它是一个`QAtomicInteger`，以保证对变量的原子访问。

+   `playSound()`: 这个函数由`PlaybackWorker`在需要播放声音时发出。

+   `trackFinished()`: 当播放被播放到结束时，这个函数会被发出。如果在途中停止，这个信号将不会被发出。

+   `play()`: 这个函数是`PlaybackWorker`的主要函数。在其中，将查询`mTrack`内容以触发声音。

+   `stop()`: 这个函数是更新`mIsPlaying`标志并导致`play()`退出其循环的函数。

类的核心在于`play()`函数：

```cpp
void PlaybackWorker::play() 
{ 
    mIsPlaying.store(true); 
    QElapsedTimer timer; 
    size_t soundEventIndex = 0; 
    const auto& soundEvents = mTrack.soundEvents(); 

    timer.start(); 
    while(timer.elapsed() <= mTrack.duration() 
          && mIsPlaying.load()) { 
        if (soundEventIndex < soundEvents.size()) { 
            const auto& soundEvent =   
                                  soundEvents.at(soundEventIndex); 

            if (timer.elapsed() >= soundEvent->timestamp) { 
                emit playSound(soundEvent->soundId); 
                soundEventIndex++; 
            } 
        } 
        QThread::msleep(1); 
    } 

    if (soundEventIndex >= soundEvents.size()) { 
        emit trackFinished(); 
    } 
} 

```

`play()`函数首先做的事情是准备它的读取：将`mIsPlaying`设置为`true`，声明一个`QElapsedTimer`类，并初始化一个`soundEventIndex`。每次调用`timer.elapsed()`时，我们都会知道是否应该播放声音。

为了知道应该播放哪个声音，`soundEventIndex`将用来知道我们在`soundEvents`向量中的位置。

随后，启动`timer`对象，我们进入`while`循环。这个`while`循环有两个条件以继续：

+   `timer.elapsed() <= mTrack.duration()`: 这个条件表明我们没有完成播放曲目

+   `mIsPlaying.load()`: 这个条件返回**true**：没有人要求`PlaybackWorker`停止

直观地讲，你可能在`while`条件中添加了`soundEventIndex < soundEvents.size()`条件。这样做的话，你会在最后一个声音播放完毕后立即退出`PlaybackWorker`。技术上，这是可行的，但这样就不会尊重用户记录的内容。

考虑一个用户创建了一个复杂的节奏（不要低估你用四个声音能做什么！）并在歌曲结束时决定有一个 5 秒的长时间暂停。当他点击停止按钮时，时间显示为 00:55（55 秒）。然而，当他播放他的表演时，最后一个声音在 00:50 结束。播放在 00:50 停止，程序不尊重他记录的内容。

因此，`soundEventIndex < size()`测试被移动到`while`循环内部，并仅用作通过`soundEvents`读取的保险丝。

在这个条件内部，我们检索当前`soundEvent`的引用。然后我们将已过时间与`soundEvent`的`timestamp`进行比较。如果`timer.elapsed()`大于或等于`soundEvent->timestamp`，则使用`soundId`发出`playSound()`信号。

这只是一个播放声音的请求。`PlaybackWorker`类仅限于读取`soundEvents`并在适当的时候触发`playSound()`。真正的声音将在稍后由`SoundEffectWidget`类处理。

在`while`循环的每次迭代中，都会执行`QThread::msleep(1)`以避免忙等待。我们尽量减少睡眠时间，因为我们希望回放尽可能忠实于原始乐谱。睡眠时间越长，回放时可能遇到的时差就越大。

最后，如果整个`soundEvents`已经被处理，将发出`trackFinished`信号。

# 使用 QVariant 使你的对象可序列化

现在我们已经在业务类中实现了逻辑，我们必须考虑我们将要序列化什么以及我们将如何进行序列化。用户与包含所有要记录和回放数据的`Track`类进行交互。

从这里开始，我们可以假设要序列化的对象是`Track`，它应该以某种方式携带其包含`SoundEvent`实例列表的`mSoundEvents`。为了实现这一点，我们将大量依赖`QVariant`类。

你可能之前已经使用过`QVariant`。它是一个用于任何原始类型（`char`、`int`、`double`等）以及复杂类型（`QString`、`QDate`、`QPoint`等）的通用占位符。

### 注意

QVariant 支持的所有类型完整列表可在[`doc.qt.io/qt-5/qmetatype.html#Type-enum`](http://doc.qt.io/qt-5/qmetatype.html#Type-enum)找到。

`QVariant`的一个简单例子是：

```cpp
QVariant variant(21); 

int answer = variant.toInt() * 2; 

qDebug() << "what is the meaning of the universe,  
             life and everything?" 
         << answer; 

```

我们在`variant`中存储`21`。从这里，我们可以要求`variant`拥有一个值副本，该值被转换为我们期望的类型。这里我们想要一个`int`值，所以我们调用`variant.toInt()`。`variant.toX()`语法已经提供了很多转换。

我们可以快速了解一下`QVariant`背后的情况。它是如何存储我们给它提供的内容的？答案在于 C++类型的`union`。`QVariant`类是一种超级`union`。

`union`是一种特殊类类型，一次只能持有其非静态数据成员中的一个。一个简短的代码片段可以说明这一点：

```cpp
union Sound 
{ 
    int duration; 
    char code; 
}; 

Sound s = 10; 
qDebug() << "Sound duration:" << s.duration; 
// output= Sound duration: 10 

s.code = 'K'; 
qDebug() << "Sound code:" << s.code; 
// output= Sound code: K 

```

首先，声明一个类似于`struct`的`union`类。默认情况下，所有成员都是`public`的。`union`的特殊之处在于它在内存中只占用最大成员的大小。在这里，`Sound`将只占用内存中`int duration`空间的大小。

因为`union`只占用这个特定的空间，每个成员变量共享相同的内存空间。因此，一次只有一个成员是可用的，除非你想要有未定义的行为。

当使用`Sound`片段时，我们首先使用默认值`10`（默认情况下第一个成员被初始化）进行初始化。从这里，`s.duration`是可访问的，但`s.code`被认为是未定义的。

一旦我们给`s.code`赋值，`s.duration`就变为未定义，而`s.code`现在是可访问的。

`union`类使得内存使用非常高效。在`QVariant`中，当你存储一个值时，它被存储在一个私有的`union`中：

```cpp
union Data 
{ 
    char c; 
    uchar uc; 
    short s; 
    signed char sc; 
    ushort us; 
    ... 
    qulonglong ull; 
    QObject *o; 
    void *ptr; 
    PrivateShared *shared; 
} data; 

```

注意基本类型列表，最后是复杂类型`QObject*`和`void*`。

除了`Data`，一个`QMetaType`对象被初始化以了解存储对象的类型。`union`和`QMetaType`的结合让`QVariant`知道应该使用哪个`Data`成员来转换值并将其返回给调用者。

现在你已经知道了`union`是什么以及`QVariant`如何使用它，你可能会问：为什么还要创建一个`QVariant`类呢？一个简单的`union`不是足够了吗？

答案是否定的。这还不够，因为`union`类不能有没有默认构造函数的成员。这极大地减少了你可以放入`union`中的类的数量。Qt 开发者想要在`union`中包含许多没有默认构造函数的类。为了减轻这个问题，`QVariant`应运而生。

`QVariant`非常有趣的地方在于它可以存储自定义类型。如果我们想将`SoundEvent`类转换为`QVariant`类，我们会在`SoundEvent.h`中添加以下内容：

```cpp
class SoundEvent 
{ 
    ... 
}; 
Q_DECLARE_METATYPE(SoundEvent); 

```

我们已经在第十章中使用了`Q_DECLARE_METATYPE`宏，*需要 IPC？让你的仆从去工作*。这个宏有效地将`SoundEvent`注册到`QMetaType`注册表中，使其对`QVariant`可用。因为`QDataStream`依赖于`QVariant`，所以我们不得不在上一个章节中使用这个宏。

现在要使用`QVariant`进行转换：

```cpp
SoundEvent soundEvent(4365, 0); 
QVariant stored; 
stored.setValue(soundEvent); 

SoundEvent newEvent = stored.value<SoundEvent>(); 
qDebug() << newEvent.timestamp; 

```

如你所猜，这个片段的输出是`4365`，这是存储在`soundEvent`中的原始`timestamp`。

如果我们只想进行二进制序列化，这种方法将完美无缺。数据可以轻松地写入和读取。然而，我们希望将`Track`和`SoundEvents`输出到标准格式：JSON 和 XML。

`Q_DECLARE_METATYPE`/`QVariant`组合有一个主要问题：它不存储序列化类字段的任何键。我们可以预见，`SoundEvent`类的 JSON 对象将看起来像这样：

```cpp
{ 
    "timestamp": 4365, 
    "soundId": 0 
} 

```

`QVariant`类不可能知道我们想要一个`timestamp`键。它只会存储原始的二进制数据。同样的原则也适用于 XML 对应物。

由于这个原因，我们将使用`QVariant`的一个变体，配合`QVariantMap`。`QVariantMap`类仅仅是`QMap<QString, QVariant>`的一个`typedef`。这个映射将用于存储字段的键名和`QVariant`类中的值。反过来，这些键将由 JSON 和 XML 序列化系统使用，以输出格式化的文件。

由于我们旨在拥有一个灵活的序列化系统，我们必须能够以多种格式序列化和反序列化这个`QVariantMap`。为了实现这一点，我们将定义一个接口，它赋予一个类在`QVariantMap`中序列化/反序列化其内容的能力。

这个`QVariantMap`将被用作一个中间格式，与最终的 JSON、XML 或二进制格式无关。

创建一个名为`Serializer.h`的 C++头文件。以下是内容：

```cpp
#include <QVariant> 

class Serializable { 
public: 
    virtual ~Serializable() {} 
    virtual QVariant toVariant() const = 0; 
    virtual void fromVariant(const QVariant& variant) = 0; 
}; 

```

通过实现这个抽象基类，一个类将变为`Serializable`。这里只有两个虚拟纯函数：

+   `toVariant()`函数，其中类必须返回一个`QVariant`（或者更精确地说是一个`QVariantMap`，它可以因为`QMetaType`系统而转换为`QVariant`）

+   `fromVariant()`函数，其中类必须从作为参数传递的变体中初始化其成员

通过这样做，我们将加载和保存其内容的责任交给最终类。毕竟，谁比`SoundEvent`本身更了解`SoundEvent`呢？

让我们看看`Serializable`在`SoundEvent`上的实际应用。像这样更新`SoundEvent.h`：

```cpp
#include "Serializable.h" 

class SoundEvent : public Serializable 
{ 
    SoundEvent(qint64 timestamp = 0, int soundId = 0); 
    ~SoundEvent(); 

    QVariant toVariant() const override; 
    void fromVariant(const QVariant& variant) override; 

    ... 
}; 

```

`SoundEvent`类现在是`Serializable`。让我们在`SoundEvent.cpp`中做实际的工作：

```cpp
QVariant SoundEvent::toVariant() const 
{ 
    QVariantMap map; 
    map.insert("timestamp", timestamp); 
    map.insert("soundId", soundId); 
    return map; 
} 

void SoundEvent::fromVariant(const QVariant& variant) 
{ 
    QVariantMap map = variant.toMap(); 
    timestamp = map.value("timestamp").toLongLong(); 
    soundId = map.value("soundId").toInt(); 
} 

```

在`toVariant()`中，我们简单地声明一个`QVariantMap`，并用`timestamp`和`soundId`填充它。

在另一方面，在`fromVariant()`中，我们将`variant`转换为`QVariantMap`，并使用与`toVariant()`中相同的键检索其内容。就这么简单！

下一个需要实现序列化（`Serializable`）的类是`Track`。在使`Track`继承自`Serializable`之后，更新`Track.cpp`：

```cpp
QVariant Track::toVariant() const 
{ 
    QVariantMap map; 
    map.insert("duration", mDuration); 

    QVariantList list; 
    for (const auto& soundEvent : mSoundEvents) { 
        list.append(soundEvent->toVariant()); 
    } 
    map.insert("soundEvents", list); 

    return map; 
} 

```

原则相同，尽管稍微复杂一些。`mDuration`变量以我们在`SoundEvent`中看到的方式存储在`map`对象中。对于`mSoundEvents`，我们必须生成一个`QVariant`（一个`QVariantList`）列表，其中每个项都是`soundEvent`键转换后的`QVariant`版本。

要做到这一点，我们只需遍历`mSoundEvents`，并用之前提到的`soundEvent->toVariant()`结果填充`list`。

现在来看`fromVariant()`：

```cpp
void Track::fromVariant(const QVariant& variant) 
{ 
    QVariantMap map = variant.toMap(); 
    mDuration = map.value("duration").toLongLong(); 

    QVariantList list = map.value("soundEvents").toList(); 
    for(const QVariant& data : list) { 
        auto soundEvent = make_unique<SoundEvent>(); 
        soundEvent->fromVariant(data); 
        mSoundEvents.push_back(move(soundEvent)); 
    } 
} 

```

在这里，对于`soundEvents`键的每个元素，我们创建一个新的`SoundEvent`，用`data`的内容加载它，并将其最终添加到`mSoundEvents`向量中。

# 以 JSON 格式序列化对象

`Track`和`SoundEvent`类现在可以被转换成通用的 Qt 格式`QVariant`。我们现在需要将`Track`（及其`SoundEvent`对象）类写入一个文本或二进制格式的文件中。这个示例项目允许你处理所有格式。它将允许你在一行中切换保存的文件格式。那么具体格式代码应该放在哪里呢？这是一个价值百万的问题！这里有一个主要的方法：

![以 JSON 格式序列化对象](img/image00442.jpeg)

在这个提议中，特定的文件格式序列化代码位于一个专门的子类中。好吧，它工作得很好，但如果我们添加两种新的文件格式，层次结构会是什么样子？此外，每次我们添加一个要序列化的新对象时，我们必须创建所有这些子类来处理不同的序列化文件格式。这个庞大的继承树很快就会变得混乱不堪。代码将变得难以维护。你不想这样做。所以，这就是桥接模式可以成为一个好解决方案的地方：

![以 JSON 格式序列化对象](img/image00443.jpeg)

在桥接模式中，我们解耦了两个继承层次结构中的类：

+   与文件格式无关的组件。`SoundEvent`和`Track`对象不关心 JSON、XML 或二进制格式。

+   文件格式实现。`JsonSerializer`、`XmlSerializer`和`BinarySerializer`处理一个通用格式`Serializable`，而不是特定的组件，如`SoundEvent`或`Track`。

注意，在经典的桥接模式中，一个抽象（`Serializable`）应该包含一个实现者（`Serializer`）变量。调用者只处理抽象。然而，在这个项目示例中，`MainWindow`拥有`Serializable`和`Serializer`的所有权。这是在保持功能类解耦的同时使用设计模式力量的个人选择。

`Serializable`和`Serializer`的架构是清晰的。`Serializable`类已经实现，因此你现在可以创建一个新的 C++头文件，名为`Serializer.h`：

```cpp
#include <QString> 

#include "Serializable.h" 

class Serializer 
{ 
public: 
    virtual ~Serializer() {} 

    virtual void save(const Serializable& serializable, 
        const QString& filepath,  
        const QString& rootName = "") = 0; 
    virtual void load(Serializable& serializable,  
        const QString& filepath) = 0; 
}; 

```

`Serializer`类是一个接口，一个只包含纯虚函数而没有数据的抽象类。让我们来谈谈`save()`函数：

+   这个函数将`Serializable`保存到硬盘驱动器上的文件中。

+   `Serializable`类是`const`的，不能被这个函数修改。

+   `filepath`函数指示要创建的目标文件

+   一些`Serializer`实现可以使用`rootName`变量。例如，如果我们请求保存一个`Track`对象，`rootName`变量可以是字符串`track`。这是用于写入根元素的标签。XML 实现需要这个信息。

`load()`函数也容易理解：

+   这个函数从文件中加载数据以填充`Serializable`类

+   这个函数将更新`Serializable`类

+   `filepath`函数指示要读取的文件

接口 `Serializer` 已准备就绪，等待一些实现！让我们从 JSON 开始。创建一个 C++ 类，`JsonSerializer`。以下是 `JsonSerializer.h` 的头文件：

```cpp
#include "Serializer.h" 

class JsonSerializer : public Serializer 
{ 
public: 
    JsonSerializer(); 

    void save(const Serializable& serializable,  
        const QString& filepath, 
        const QString& rootName) override; 
    void load(Serializable& serializable, 
        const QString& filepath) override; 
}; 

```

这里没有困难；我们必须提供 `save()` 和 `load()` 的实现。以下是 `save()` 的实现：

```cpp
void JsonSerializer::save(const Serializable& serializable, 
    const QString& filepath, const QString& /*rootName*/) 
{ 
    QJsonDocument doc =     
        QJsonDocument::fromVariant(serializable.toVariant()); 
    QFile file(filepath); 
    file.open(QFile::WriteOnly); 
    file.write(doc.toJson()); 
    file.close(); 
} 

```

Qt 框架提供了一个很好的方式来使用 `QJsonDocument` 类读取和写入 JSON 文件。我们可以从 `QVariant` 类创建一个 `QJsonDocument` 类。请注意，`QJsonDocument` 接受的 `QVariant` 必须是 `QVariantMap`、`QVariantList` 或 `QStringList`。不用担心，`Track` 类和 `SoundEvent` 的 `toVariant()` 函数会生成一个 `QVariantMap`。然后，我们可以使用目标 `filepath` 创建一个 `QFile` 文件。`QJsonDocument::toJson()` 函数将其转换为 UTF-8 编码的文本表示。我们将此结果写入 `QFile` 文件并关闭文件。

### 提示

`QJsonDocument::toJson()` 函数可以生成 `Indented` 或 `Compact` JSON 格式。默认情况下，格式是 `QJsonDocument::Indented`。

`load()` 的实现也很简短：

```cpp
void JsonSerializer::load(Serializable& serializable, 
    const QString& filepath) 
{ 
    QFile file(filepath); 
    file.open(QFile::ReadOnly); 
    QJsonDocument doc = QJsonDocument::fromJson(file.readAll()); 
    file.close(); 
    serializable.fromVariant(doc.toVariant()); 
} 

```

我们使用源 `filepath` 打开一个 `QFile`。使用 `QFile::readAll()` 读取所有数据。然后我们可以使用 `QJsonDocument::fromJson()` 函数创建一个 `QJsonDocument` 类。最后，我们可以用转换为 `QVariant` 类的 `QJsonDocument` 填充我们的目标 `Serializable`。请注意，`QJsonDocument::toVariant()` 函数可以返回 `QVariantList` 或 `QVariantMap`，具体取决于 JSON 文档的性质。

这里是使用这个 `JsonSerializer` 保存的 `Track` 类的示例：

```cpp
{ 
    "duration": 6205, 
    "soundEvents": [ 
        { 
            "soundId": 0, 
            "timestamp": 2689 
        }, 
        { 
            "soundId": 2, 
            "timestamp": 2690 
        }, 
        { 
            "soundId": 2, 
            "timestamp": 3067 
        } 
    ] 
} 

```

根元素是一个 JSON 对象，由包含两个键的映射表示：

+   `Duration`: 这是一个简单的整数值

+   `soundEvents`: 这是一个对象的数组。每个对象是一个包含以下键的映射：

+   `soundId`: 这是一个整数

+   `timestamp`: 这也是一个整数值

# 以 XML 格式序列化对象

JSON 序列化是 C++ 对象的直接表示，Qt 已经提供了我们所需的一切。然而，C++ 对象的序列化可以用各种表示形式在 XML 格式中完成。因此，我们必须自己编写 XML 到 `QVariant` 的转换。我们决定使用以下 XML 表示形式：

```cpp
<[name]> type="[type]">[data]</[name]> 

```

例如，`soundId` 类型给出以下 XML 表示形式：

```cpp
<soundId type="int">2</soundId> 

```

创建一个继承自 `Serializer` 的 C++ 类 `XmlSerializer`。让我们从 `save()` 函数开始，以下是 `XmlSerializer.h`：

```cpp
#include <QXmlStreamWriter> 
#include <QXmlStreamReader> 

#include "Serializer.h" 

class XmlSerializer : public Serializer 
{ 
public: 
    XmlSerializer(); 

    void save(const Serializable& serializable,  
        const QString& filepath,  
        const QString& rootName) override; 
}; 

```

现在，我们可以看到 `XmlSerializer.cpp` 中的 `save()` 实现：

```cpp
void XmlSerializer::save(const Serializable& serializable, const QString& filepath, const QString& rootName) 
{ 
    QFile file(filepath); 
    file.open(QFile::WriteOnly); 
    QXmlStreamWriter stream(&file); 
    stream.setAutoFormatting(true); 
    stream.writeStartDocument(); 
    writeVariantToStream(rootName, serializable.toVariant(), 
        stream); 
    stream.writeEndDocument(); 
    file.close(); 
} 

```

我们使用 `filepath` 目的地创建一个 `QFile` 文件。我们构造一个写入 `QFile` 的 `QXmlStreamWriter` 对象。默认情况下，写入器将生成紧凑的 XML；你可以使用 `QXmlStreamWriter::setAutoFormatting()` 函数生成格式化的 XML。`QXmlStreamWriter::writeStartDocument()` 函数写入 XML 版本和编码。我们使用 `writeVariantToStream()` 函数将我们的 `QVariant` 写入 XML 流中。最后，我们结束文档并关闭 `QFile`。如前所述，将 `QVariant` 写入 XML 流取决于你如何表示数据。因此，我们必须编写转换函数。请更新你的类，添加如下 `writeVariantToStream()`：

```cpp
//XmlSerializer.h 
private: 
    void writeVariantToStream(const QString& nodeName, 
        const QVariant& variant, QXmlStreamWriter& stream); 

//XmlSerializer.cpp 
void XmlSerializer::writeVariantToStream(const QString& nodeName, 
    const QVariant& variant, QXmlStreamWriter& stream) 
{ 
    stream.writeStartElement(nodeName); 
    stream.writeAttribute("type", variant.typeName()); 

    switch (variant.type()) { 
        case QMetaType::QVariantList: 
            writeVariantListToStream(variant, stream); 
            break; 
        case QMetaType::QVariantMap: 
            writeVariantMapToStream(variant, stream); 
            break; 
        default: 
            writeVariantValueToStream(variant, stream); 
            break; 
    } 

    stream.writeEndElement(); 
} 

```

这个 `writeVariantToStream()` 函数是一个通用入口点。每次我们想要将一个 `QVariant` 放入 XML 流时，它都会被调用。`QVariant` 类可以是列表、映射或数据。因此，如果 `QVariant` 是一个容器（`QVariantList` 或 `QVariantMap`），我们将应用特定的处理。所有其他情况都被视为数据值。以下是此函数的步骤：

1.  使用 `writeStartElement()` 函数开始一个新的 XML 元素。`nodeName` 将用于创建 XML 标签。例如，`<soundId>`。

1.  在当前元素中写入一个名为 `type` 的 XML 属性。我们使用存储在 `QVariant` 中的类型名称。例如，`<soundId type="int" />`。

1.  根据数据类型 `QVariant`，我们调用我们的一个 XML 序列化函数。例如，`<soundId type="int">2`。

1.  最后，我们使用 `writeEndElement()` 结束当前 XML 元素：

    +   最终结果是：`<soundId type="int">2</soundId>`

    +   在这个函数中，我们调用我们现在将创建的三个辅助函数。其中最容易的是 `writeVariantValueToStream()`。请更新你的 `XmlSerializer` 类：

        ```cpp
        //XmlSerializer.h 
        void writeVariantValueToStream(const QVariant& variant, 
            QXmlStreamWriter& stream); 

        //XmlSerializer.cpp 
        void XmlSerializer::writeVariantValueToStream( 
            const QVariant& variant, QXmlStreamWriter& stream) 
        { 
            stream.writeCharacters(variant.toString()); 
        } 

        ```

如果 `QVariant` 是一个简单类型，我们检索它的 `QString` 表示形式。然后我们使用 `QXmlStreamWriter::writeCharacters()` 将这个 `QString` 写入 XML 流中。

第二个辅助函数是 `writeVariantListToStream()`。以下是它的实现：

```cpp
//XmlSerializer.h 
private: 
    void writeVariantListToStream(const QVariant& variant, 
        QXmlStreamWriter& stream); 

//XmlSerializer.cpp 
void XmlSerializer::writeVariantListToStream( 
    const QVariant& variant, QXmlStreamWriter& stream) 
{ 
    QVariantList list = variant.toList(); 

    for(const QVariant& element : list) { 
        writeVariantToStream("item", element, stream); 
    } 
} 

```

在这一步，我们已经知道 `QVariant` 是一个 `QVariantList`。我们调用 `QVariant::toList()` 来检索列表。然后我们遍历列表的所有元素并调用我们的通用入口点，`writeVariantToStream()`。请注意，我们从列表中检索元素，因此我们没有元素名称。但是，对于列表项的序列化，标签名称并不重要，所以插入任意标签 `item`。

最后一个写入辅助函数是 `writeVariantMapToStream()`：

```cpp
//XmlSerializer.h 
private: 
    void writeVariantMapToStream(const QVariant& variant, 
        QXmlStreamWriter& stream); 

//XmlSerializer.cpp 
void XmlSerializer::writeVariantMapToStream( 
    const QVariant& variant, QXmlStreamWriter& stream) 
{ 
    QVariantMap map = variant.toMap(); 
    QMapIterator<QString, QVariant> i(map); 

    while (i.hasNext()) { 
        i.next(); 
        writeVariantToStream(i.key(), i.value(), stream); 
    } 
} 

```

`QVariant` 是一个容器，但这次是 `QVariantMap`。我们对每个找到的元素调用 `writeVariantToStream()`。标签名称对于映射很重要。我们使用 `QMapIterator::key()` 作为节点名称。

保存部分已经完成。现在我们可以实现加载部分。它的架构与保存函数遵循相同的理念。让我们从 `load()` 函数开始：

```cpp
//XmlSerializer.h 
public: 
    void load(Serializable& serializable,  
        const QString& filepath) override; 

//XmlSerializer.cpp 
void XmlSerializer::load(Serializable& serializable, 
    const QString& filepath) 
{ 
    QFile file(filepath); 
    file.open(QFile::ReadOnly); 
    QXmlStreamReader stream(&file); 
    stream.readNextStartElement(); 
    serializable.fromVariant(readVariantFromStream(stream)); 
} 

```

首先要做的事情是创建一个包含源 `filepath` 的 `QFile`。我们使用 `QFile` 构造一个 `QXmlStreamReader`。`QXmlStreamReader ::readNextStartElement()` 函数读取直到 XML 流中的下一个起始元素。然后我们可以使用我们的读取辅助函数 `readVariantFromStream()` 从 XML 流中创建一个 `QVariant` 类。最后，我们可以使用我们的 `Serializable::fromVariant()` 来填充目标 `serializable`。让我们实现辅助函数 `readVariantFromStream()`：

```cpp
//XmlSerializer.h 
private: 
    QVariant readVariantFromStream(QXmlStreamReader& stream); 

//XmlSerializer.cpp 
QVariant XmlSerializer::readVariantFromStream(QXmlStreamReader& stream) 
{ 
    QXmlStreamAttributes attributes = stream.attributes(); 
    QString typeString = attributes.value("type").toString(); 

    QVariant variant; 
    switch (QVariant::nameToType( 
            typeString.toStdString().c_str())) { 
        case QMetaType::QVariantList: 
            variant = readVariantListFromStream(stream); 
            break; 
        case QMetaType::QVariantMap: 
            variant = readVariantMapFromStream(stream); 
            break; 
        default: 
            variant = readVariantValueFromStream(stream); 
            break; 
    } 

    return variant; 
} 

```

这个函数的作用是创建一个 `QVariant`。首先，我们从 XML 属性中检索 `"type"`。在我们的例子中，我们只有一个属性需要处理。然后，根据类型，我们将调用我们三个读取辅助函数中的一个。让我们实现 `readVariantValueFromStream()` 函数：

```cpp
//XmlSerializer.h 
private: 
    QVariant readVariantValueFromStream(QXmlStreamReader& stream); 

//XmlSerializer.cpp 
QVariant XmlSerializer::readVariantValueFromStream( 
    QXmlStreamReader& stream) 
{ 
    QXmlStreamAttributes attributes = stream.attributes(); 
    QString typeString = attributes.value("type").toString(); 
    QString dataString = stream.readElementText(); 

    QVariant variant(dataString); 
    variant.convert(QVariant::nameToType( 
        typeString.toStdString().c_str())); 
    return variant; 
} 

```

这个函数创建一个根据类型数据而定的 `QVariant`。和之前的函数一样，我们从 XML 属性中检索类型。我们同样使用 `QXmlStreamReader::readElementText()` 函数读取文本数据。使用这个 `QString` 数据创建一个 `QVariant` 类。在这个步骤中，`QVariant` 类型是 `QString`。因此，我们使用 `QVariant::convert()` 函数将 `QVariant` 转换为实际类型（`int`、`qlonglong` 等）。

第二个读取辅助函数是 `readVariantListFromStream()`：

```cpp
//XmlSerializer.h 
private: 
    QVariant readVariantListFromStream(QXmlStreamReader& stream); 

//XmlSerializer.cpp 
QVariant XmlSerializer::readVariantListFromStream(QXmlStreamReader& stream) 
{ 
    QVariantList list; 
    while(stream.readNextStartElement()) { 
        list.append(readVariantFromStream(stream)); 
    } 
    return list; 
} 

```

我们知道流元素包含一个数组。因此，这个函数创建并返回一个 `QVariantList`。`QXmlStreamReader::readNextStartElement()` 函数读取直到下一个起始元素，如果当前元素内找到起始元素则返回 `true`。我们为每个元素调用入口点函数 `readVariantFromStream()`。最后，我们返回 `QVariantList`。

最后要覆盖的辅助函数是 `readVariantMapFromStream()`。更新你的文件，使用以下片段：

```cpp
//XmlSerializer.h 
private: 
    QVariant readVariantMapFromStream(QXmlStreamReader& stream); 

//XmlSerializer.cpp 
QVariant XmlSerializer::readVariantMapFromStream( 
    QXmlStreamReader& stream) 
{ 
    QVariantMap map; 
    while(stream.readNextStartElement()) { 
        map.insert(stream.name().toString(), 
                   readVariantFromStream(stream)); 
    } 
    return map; 
} 

```

这个函数听起来像 `readVariantListFromStream()`。这次我们必须创建一个 `QVariantMap`。用于插入新项的键是元素名称。我们使用 `QXmlStreamReader::name()` 函数检索名称。

使用 `XmlSerializer` 序列化的 `Track` 类看起来像这样：

```cpp
<?xml version="1.0" encoding="UTF-8"?> 
<track type="QVariantMap"> 
    <duration type="qlonglong">6205</duration> 
    <soundEvents type="QVariantList"> 
        <item type="QVariantMap"> 
            <soundId type="int">0</soundId> 
            <timestamp type="qlonglong">2689</timestamp> 
        </item> 
        <item type="QVariantMap"> 
            <soundId type="int">2</soundId> 
            <timestamp type="qlonglong">2690</timestamp> 
        </item> 
        <item type="QVariantMap"> 
            <soundId type="int">2</soundId> 
            <timestamp type="qlonglong">3067</timestamp> 
        </item> 
    </soundEvents> 
</track> 

```

# 以二进制格式序列化对象

XML 序列化已经完全可用！我们现在可以切换到本章中介绍的序列化的最后一种类型。

二进制序列化比较简单，因为 Qt 提供了一个直接的方法来做这件事。请创建一个继承自 `Serializer` 的 `BinarySerializer` 类。头文件是通用的，我们只有重写的函数，`save()` 和 `load()`。以下是 `save()` 函数的实现：

```cpp
void BinarySerializer::save(const Serializable& serializable, 
    const QString& filepath, const QString& /*rootName*/) 
{ 
    QFile file(filepath); 
    file.open(QFile::WriteOnly); 
    QDataStream dataStream(&file); 
    dataStream << serializable.toVariant(); 
    file.close(); 
} 

```

我们希望你能认出在 第十章 中使用的 `QDataStream` 类，*需要 IPC？让你的小弟去工作*。这次我们使用这个类在目标 `QFile` 中序列化二进制数据。`QDataStream` 类接受一个带有 `<<` 操作符的 `QVariant` 类。请注意，`rootName` 变量在二进制序列化器中没有被使用。

这里是 `load()` 函数：

```cpp
void BinarySerializer::load(Serializable& serializable, const QString& filepath) 
{ 
    QFile file(filepath); 
    file.open(QFile::ReadOnly); 
    QDataStream dataStream(&file); 
    QVariant variant; 
    dataStream >> variant; 
    serializable.fromVariant(variant); 
    file.close(); 
} 

```

多亏了 `QVariant` 和 `QDataStream` 机制，任务变得简单。我们使用源 `filepath` 打开 `QFile`。然后，我们使用这个 `QFile` 构造一个 `QDataStream` 类。然后，我们使用 `>>` 操作符读取根 `QVariant`。最后，我们使用 `Serializable::fromVariant()` 函数填充源 `Serializable`。

不要担心，我们不会包含使用 `BinarySerializer` 类序列化的 `Track` 类的示例。

序列化部分已完成。本例项目的 GUI 部分在本书的前几章中已经多次介绍。以下章节将仅涵盖在 `MainWindow` 和 `SoundEffectWidget` 类中使用的特定功能。如果需要完整的 C++ 类，请检查源代码。

# 使用 QSoundEffect 播放低延迟声音

项目应用程序 `ch11-drum-machine` 显示了四个 `SoundEffectWidget` 小部件：`kickWidget`、`snareWidget`、`hihatWidget` 和 `crashWidget`。

每个 `SoundEffectWidget` 小部件显示一个 `QLabel` 和一个 `QPushButton`。标签显示声音名称。如果按钮被点击，就会播放声音。

Qt 多媒体模块提供了两种主要方式来播放音频文件：

+   `QMediaPlayer`：这个文件可以播放歌曲、电影和互联网广播，支持各种输入格式

+   `QSoundEffect`：这个文件可以播放低延迟的 `.wav` 文件

这个项目示例是一个虚拟鼓机，所以我们使用了一个 `QSoundEffect` 对象。使用 `QSoundEffect` 的第一步是更新你的 `.pro` 文件，如下所示：

```cpp
QT       += core gui multimedia 

```

然后，你可以初始化声音。以下是一个示例：

```cpp
QUrl urlKick("qrc:/sounds/kick.wav"); 
QUrl urlBetterKick = QUrl::fromLocalFile("/home/better-kick.wav"); 

QSoundEffect soundEffect; 
QSoundEffect.setSource(urlBetterKick); 

```

第一步是为你的声音文件创建一个有效的 `QUrl`。`urlKick` 从 `.qrc` 资源文件路径初始化，而 `urlBetterKick` 是从本地文件路径创建的。然后我们可以创建 `QSoundEffect` 并使用 `QSoundEffect::setSource()` 函数设置要播放的 URL 声音。

现在我们已经初始化了一个 `QSoundEffect` 对象，我们可以使用以下代码片段来播放声音：

```cpp
soundEffect.setVolume(1.0f); 
soundEffect.play(); 

```

# 使用键盘触发 QButton

让我们来探索 `SoundEffectWidget` 类中的公共槽，`triggerPlayButton()`：

```cpp
//SoundEffectWidget.h 
class SoundEffectWidget : public QWidget 
{ 
... 
public slots: 
    void triggerPlayButton(); 
    ... 

private: 
    QPushButton* mPlayButton; 
    ... 
}; 

//SoundEffectWidget.cpp 
void SoundEffectWidget::triggerPlayButton() 
{ 
   mPlayButton->animateClick(); 
} 

```

这个小部件有一个名为 `mPlayButton` 的 `QPushButton`。`triggerPlayButton()` 槽调用 `QPushButton::animateClick()` 函数，默认情况下通过 100 毫秒模拟按钮点击。所有信号都将像真实点击一样发送。按钮看起来确实被按下了。如果你不想有动画，可以调用 `QPushButton::click()`。

现在我们来看看如何使用键盘触发这个槽。每个 `SoundEffectWidget` 都有一个 `Qt:Key`：

```cpp
//SoundEffectWidget.h 
class SoundEffectWidget : public QWidget 
{ 
... 
public: 
    Qt::Key triggerKey() const; 
    void setTriggerKey(const Qt::Key& triggerKey); 
}; 

//SoundEffectWidget.cpp 
Qt::Key SoundEffectWidget::triggerKey() const 
{ 
    return mTriggerKey; 
} 

void SoundEffectWidget::setTriggerKey(const Qt::Key& triggerKey) 
{ 
    mTriggerKey = triggerKey; 
} 

```

`SoundEffectWidget` 类提供了一个获取器和设置器来获取和设置成员变量 `mTriggerKey`。

`MainWindow` 类初始化其四个 `SoundEffectWidget` 的键如下：

```cpp
ui->kickWidget->setTriggerKey(Qt::Key_H); 
ui->snareWidget->setTriggerKey(Qt::Key_J); 
ui->hihatWidget->setTriggerKey(Qt::Key_K); 
ui->crashWidget->setTriggerKey(Qt::Key_L); 

```

默认情况下，`QObject::eventFilter()` 函数不会被调用。为了启用它并拦截这些事件，我们需要在 `MainWindow` 上安装一个事件过滤器：

```cpp
installEventFilter(this); 

```

因此，每次 `MainWindow` 接收到事件时，都会调用 `MainWindow::eventFilter()` 函数。

这里是 `MainWindow.h` 头文件：

```cpp
class MainWindow : public QMainWindow 
{ 
    Q_OBJECT 
public: 
    ... 
    bool eventFilter(QObject* watched, QEvent* event) override; 

private: 
    QVector<SoundEffectWidget*> mSoundEffectWidgets; 
    ... 
}; 

```

`MainWindow` 类有一个 `QVector`，包含四个 `SoundEffectWidgets` (`kickWidget`、`snareWidget`、`hihatWidget` 和 `crashWidget`)。让我们看看 `MainWindow.cpp` 中的实现：

```cpp
bool MainWindow::eventFilter(QObject* watched, QEvent* event) 
{ 
    if (event->type() == QEvent::KeyPress) { 
        QKeyEvent* keyEvent = static_cast<QKeyEvent*>(event); 
        for(SoundEffectWidget* widget : mSoundEffectWidgets) { 
            if (keyEvent->key() == widget->triggerKey()) { 
                widget->triggerPlayButton(); 
                return true; 
            } 
        } 
    } 
    return QObject::eventFilter(watched, event); 
} 

```

首先要做的是检查 `QEvent` 类是否为 `KeyPress` 类型。我们不关心其他事件类型。如果事件类型正确，我们继续下一步：

1.  将 `QEvent` 类转换为 `QKeyEvent`。

1.  然后我们搜索按下的键是否属于 `SoundEffectWidget` 类。

1.  如果 `SoundEffectWidget` 类与键相对应，我们调用我们的 `SoundEffectWidget::triggerPlayButton()` 函数，并返回 `true` 以指示我们已消费该事件，并且它不得传播到其他类。

1.  否则，我们调用 `QObject` 类的 `eventFilter()` 实现。

# 使 `PlaybackWorker` 生动起来

用户可以通过鼠标点击或键盘键实时播放声音。但是，当他录制一个令人惊叹的节奏时，应用程序必须能够使用 `PlaybackWorker` 类再次播放它。让我们看看 `MainWindow` 如何使用这个工作器。以下是与 `PlaybackWorker` 类相关的 `MainWindow.h`：

```cpp
class MainWindow : public QMainWindow 
{ 
... 
private slots: 
    void playSoundEffect(int soundId); 
    void clearPlayback(); 
    void stopPlayback(); 
    ... 

private: 
    void startPlayback(); 
    ... 

private: 
    PlaybackWorker* mPlaybackWorker; 
    QThread* mPlaybackThread; 
    ... 
}; 

```

如您所见，`MainWindow` 有 `PlaybackWorker` 和一个 `QThread` 成员变量。让我们看看 `startPlayback()` 的实现：

```cpp
void MainWindow::startPlayback() 
{ 
    clearPlayback(); 

    mPlaybackThread = new QThread(); 

    mPlaybackWorker = new PlaybackWorker(mTrack); 
    mPlaybackWorker->moveToThread(mPlaybackThread); 

    connect(mPlaybackThread, &QThread::started, 
            mPlaybackWorker, &PlaybackWorker::play); 
    connect(mPlaybackThread, &QThread::finished, 
            mPlaybackWorker, &QObject::deleteLater); 

    connect(mPlaybackWorker, &PlaybackWorker::playSound, 
            this, &MainWindow::playSoundEffect); 

    connect(mPlaybackWorker, &PlaybackWorker::trackFinished, 
            &mTrack, &Track::stop); 

    mPlaybackThread->start(QThread::HighPriority); 
} 

```

让我们分析所有步骤：

1.  我们使用 `clearPlayback()` 函数清除当前播放，这个功能很快就会介绍。

1.  新的 `QThread` 和 `PlaybackWorker` 被构造。此时，当前曲目被传递给工作器。像往常一样，工作器随后被移动到其专用线程。

1.  我们希望尽快播放曲目。因此，当 `QThread` 发出 `started()` 信号时，会调用 `PlaybackWorker::play()` 插槽。

1.  我们不想担心 `PlaybackWorker` 的内存。因此，当 `QThread` 结束并发送了 `finished()` 信号时，我们调用 `QObject::deleteLater()` 插槽，该插槽安排工作器进行删除。

1.  当 `PlaybackWorker` 类需要播放声音时，会发出 `playSound()` 信号，并调用我们的 `MainWindow::playSoundEffect()` 插槽。

1.  最后一个连接覆盖了当 `PlaybackWorker` 类播放完整个曲目时的情况。会发出一个 `trackFinished()` 信号，然后我们调用 `Track::Stop()` 插槽。

1.  最后，以高优先级启动线程。请注意，某些操作系统（例如 Linux）不支持线程优先级。

现在我们可以看到 `stopPlayback()` 函数体：

```cpp
void MainWindow::stopPlayback() 
{ 
    mPlaybackWorker->stop(); 
    clearPlayback(); 
} 

```

我们从我们的线程中调用 `PlaybackWorker` 的 `stop()` 函数。因为我们使用 `QAtomicInteger` 在 `stop()` 中，所以该函数是线程安全的，可以直接调用。最后，我们调用我们的辅助函数 `clearPlayback()`。这是我们第二次使用 `clearPlayback()`，所以让我们来实现它：

```cpp
void MainWindow::clearPlayback() 
{ 
    if (mPlaybackThread) { 
        mPlaybackThread->quit(); 
        mPlaybackThread->wait(1000); 
        mPlaybackThread = nullptr; 
        mPlaybackWorker = nullptr; 
    } 
} 

```

没有任何惊喜。如果线程有效，我们要求线程退出并等待 1 秒。然后，我们将线程和工作者设置为 `nullptr`。

`PlaybackWorker::PlaySound` 信号连接到 `MainWindow::playSoundEffect()`。以下是其实施：

```cpp
void MainWindow::playSoundEffect(int soundId) 
{ 
   mSoundEffectWidgets[soundId]->triggerPlayButton(); 
} 

```

此槽获取与 `soundId` 对应的 `SoundEffectWidget` 类。然后，我们调用 `triggerPlayButton()`，这是当你按下键盘上的触发键时调用的相同方法。

因此，当你点击按钮、按下一个键，或者当 `PlaybackWorker` 类请求播放声音时，`SoundEffectWidget` 的 `QPushButton` 会发出 `clicked()` 信号。这个信号连接到我们的 `SoundEffectWidget::play()` 槽。下一个片段描述了这个槽：

```cpp
void SoundEffectWidget::play() 
{ 
    mSoundEffect.play(); 
    emit soundPlayed(mId); 
} 

```

没有什么特别之处。我们在已经覆盖的 `QSoundEffect` 上调用 `play()` 函数。最后，如果我们处于 `RECORDING` 状态，我们发出 `soundPlayed()` 信号，该信号由 `Track` 用于添加新的 `SoundEvent`。

# 接受鼠标拖放事件

在这个项目示例中，如果你将 `.wav` 文件拖放到 `SoundEffectWidget` 上，你可以更改播放的声音。`SoundEffectWidget` 的构造函数执行特定任务以允许拖放：

```cpp
setAcceptDrops(true); 

```

我们现在可以覆盖拖放回调。让我们从 `dragEnterEvent()` 函数开始：

```cpp
//SoundEffectWidget.h 
class SoundEffectWidget : public QWidget 
{ 
... 
protected: 
    void dragEnterEvent(QDragEnterEvent* event) override; 
... 
}; 

//SoundEffectWidget.cpp 
void SoundEffectWidget::dragEnterEvent(QDragEnterEvent* event) 
{ 
    if (event->mimeData()->hasFormat("text/uri-list")) { 
        event->acceptProposedAction(); 
    } 
} 

```

`dragEnterEvent()` 函数会在用户在部件上拖动对象时被调用。在我们的例子中，我们只想允许拖放那些 MIME 类型为 `"text/uri-list"`（URI 列表，可以是 `file://`、`http://` 等等）的文件。在这种情况下，尽管我们可以调用 `QDragEnterEvent::acceptProposedAction()` 函数来通知我们接受这个对象进行拖放。

我们现在可以添加第二个函数，`dropEvent()`：

```cpp
//SoundEffectWidget.h 
class SoundEffectWidget : public QWidget 
{ 
... 
protected: 
    void dropEvent(QDropEvent* event) override; 
... 
}; 

//SoundEffectWidget.cpp 
void SoundEffectWidget::dropEvent(QDropEvent* event) 
{ 
    const QMimeData* mimeData = event->mimeData(); 
    if (!mimeData->hasUrls()) { 
        return; 
    } 
    const QUrl url = mimeData->urls().first(); 
    QMimeType mime = QMimeDatabase().mimeTypeForUrl(url); 
    if (mime.inherits("audio/wav")) { 
        loadSound(url); 
    } 
} 

```

第一步是进行合理性检查。如果事件没有 URL，我们就不做任何事情。`QMimeData::hasUrls()` 函数仅在 MIME 类型为 `"text/uri-text"` 时返回 `true`。注意，用户可以一次性拖放多个文件。在我们的例子中，我们只处理第一个 URL。你可以检查文件是否为 `.wav` 文件，通过其 MIME 类型。如果 MIME 类型是 `"audio/wav"`，我们调用 `loadSound()` 函数，该函数更新分配给此 `SoundEffectWidget` 的声音。

以下截图显示了 `ch11-drum-machine` 的完整应用程序：

![接受鼠标拖放事件](img/image00444.jpeg)

# 摘要

序列化是在你关闭应用程序时使你的数据持久化的好方法。在本章中，你学习了如何使用 `QVariant` 使你的 C++ 对象可序列化。你使用桥接模式创建了一个灵活的序列化结构。你将对象保存为不同的文本格式，如 JSON 或 XML，以及二进制格式。

你还学会了使用 Qt 多媒体模块来播放一些音效。这些声音可以通过鼠标点击或键盘按键触发。你实现了友好的用户交互，允许你通过文件拖放来加载新的声音。

在下一章中，我们将发现 `QTest` 框架以及如何组织你的项目，使其具有清晰的应用程序/测试分离。
