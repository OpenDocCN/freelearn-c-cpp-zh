# 示例 - 基于 Linux 的信息娱乐系统

本章提供了一个使用基于 Linux 的**单板计算机**（**SBC**）实现信息娱乐系统的示例。它还展示了如何使用蓝牙连接到远程设备，以及如何使用在线流媒体服务。该设备将能够从各种来源播放音频，而无需复杂的用户界面。特别是，我们将涵盖以下主题：

+   为基于 Linux 的 SBC 开发

+   在 Linux 下使用蓝牙

+   从各种来源播放音频和录制音频

+   使用 GPIO 进行简单的输入和语音识别

+   连接到在线流媒体音频服务

# 一个能做所有事情的盒子

信息娱乐系统已成为我们日常生活中的一种常见功能，始于**车载娱乐**（**ICE**）系统（也称为**车载信息娱乐**或**IVI**），它从基本的收音机和卡式录音机发展到包括导航功能，并通过蓝牙连接到智能手机以访问音乐库等功能。另一个重要功能是向驾驶员提供免提功能，这样他们就可以在不离开路面或方向盘的情况下开始电话通话和控制收音机。

随着智能手机的普及，为用户提供持续的新闻、天气和娱乐访问，车载助手的出现，这些助手在智能手机和 ICE 上使用语音驱动界面，最终导致了面向家庭使用的语音驱动信息娱乐系统的出现。这些系统通常包括一个扬声器和麦克风，以及语音驱动界面所需的硬件和访问所需基于互联网的服务。

本章将主要关注此类语音驱动信息娱乐系统。在第十章《使用 Qt 开发嵌入式系统》中，我们将深入探讨如何添加图形用户界面。

我们想要实现的目标如下：

+   从蓝牙源播放音乐，例如智能手机

+   从在线流媒体服务播放音乐

+   从本地文件系统播放音乐，包括 USB 闪存盘

+   录制音频片段并在需要时重复播放

+   使用语音控制所有操作，对于某些操作使用按钮

在接下来的几节中，我们将探讨这些目标以及如何实现它们。

# 所需硬件

对于这个项目，任何能够运行 Linux 的 SBC 都应该可以工作。它还需要以下功能才能进行完整实现：

+   一个互联网连接（无线或有线）以访问在线内容。

+   具有内置或作为附加模块的蓝牙功能，使系统能够作为蓝牙扬声器使用。

+   提供免费的 GPIO 输入，以便连接按钮。

+   用于语音输入和音频播放的功能麦克风输入和音频输出。

+   用于连接存储设备（如硬盘驱动器）的 SATA 连接或类似连接。

+   I2C 总线的 I2C 显示屏外设。

对于本章的示例代码，我们只需要麦克风输入和音频输出，以及一些用于本地媒体文件的存储空间。

我们可以将多个按钮连接到 GPIO 引脚，以便在不使用语音激活系统的情况下控制信息娱乐系统。这在使用语音激活系统会显得尴尬的情况下很方便，例如在接电话时暂停或静音音乐。

在本例中不会演示按钮的连接，但可以在第三章中找到一个早期项目的示例，*为嵌入式 Linux 和类似系统开发*。在那里，我们使用了 WiringPi 库将开关连接到 GPIO 引脚，并配置了中断例程来处理这些开关的变化。

如果想要显示当前信息，例如当前歌曲的名称或其他相关状态信息，也可以将一个小型显示屏连接到系统。16x2 字符的廉价显示屏，可以通过 I2C 接口控制，非常普遍；这些显示屏，以及一系列 OLED 和其他小型显示屏，由于其硬件要求最小，因此非常适合此目的。

在第三章，*为嵌入式 Linux 和类似系统开发*中，我们简要地看了看可能用于此类信息娱乐系统的硬件类型，以及一些可能的用户界面和存储选项。当然，合适的硬件配置取决于个人需求。如果想要存储大量音乐进行播放，将一个大型的 SATA 硬盘连接到系统将会非常方便。

然而，在本章的示例中，我们不会做出这样的假设，而是作为一个易于扩展的起点。因此，硬件要求非常有限，除了明显的麦克风和音频输出需求之外。

# 软件要求

对于这个项目，我们假设 Linux 已经安装在了目标 SBC 上，并且硬件功能（如麦克风和音频输出）的驱动程序已经安装并配置好。

由于我们在这个项目中使用 Qt 框架，因此所有依赖项也应该满足。这意味着，在运行项目生成的二进制文件的系统上应该存在共享库。Qt 框架可以通过操作系统的包管理器或通过 Qt 网站[`qt.io/`](http://qt.io/)获取。

在第十章，*使用 Qt 开发嵌入式系统*中，我们将更详细地探讨使用 Qt 在嵌入式平台上进行开发。本章将简要介绍 Qt API 的使用。

根据我们是在 SBC 上直接编译应用程序还是在我们的开发 PC 上编译，我们可能需要在 SBC 上安装编译器工具链和进一步的依赖项，或者是在目标 SBC（ARM、x86 或其他架构）上的 Linux 交叉编译工具链。在第六章测试基于 OS 的应用程序中，我们探讨了 SBC 系统的交叉编译以及本地测试系统。

由于本章的示例项目不需要任何特殊硬件，它可以直接在任何由 Qt 框架支持的系统上编译。这是在部署到 SBC 之前测试代码的推荐方式。

# 蓝牙音频源和接收器

蓝牙是一种不幸的技术，尽管它无处不在，但由于其专有性质，支持蓝牙功能的完整范围（以配置文件的形式）不足。我们在这个项目中感兴趣的是称为**高级音频分配配置文件**（**A2DP**）。这是从蓝牙耳机到蓝牙扬声器等各种设备用来流式传输音频的配置文件。

任何实现了 A2DP 的设备都可以将音频流式传输到 A2DP 接收器，或者本身可以作为接收器（取决于蓝牙堆栈的实现）。理论上，这允许某人通过智能手机或类似设备连接到我们的信息娱乐系统并播放音乐，就像他们使用独立的蓝牙扬声器一样。

A2DP 配置文件中的接收器是一个 A2DP 接收器，而另一边是 A2DP 源。蓝牙耳机或扬声器设备始终是接收设备，因为它们只能消费音频流。PC、SBC 或类似的多用途设备可以被配置为接收器或源。

如前所述，在主流操作系统上实现完整蓝牙堆栈的复杂性导致了除了蓝牙基本串行通信功能之外的其他功能的支持不佳。

尽管 FreeBSD、macOS、Windows 和 Android 都拥有蓝牙堆栈，但它们在支持的蓝牙适配器数量（Windows 上只有一个，并且仅限 USB 适配器）、支持的配置文件（FreeBSD 仅限数据传输）和可配置性（Android 基本上仅针对智能手机）方面都有限制。

对于 Windows 10，由于蓝牙堆栈的变化，A2DP 配置文件支持已经从 Windows 7 中的功能性退化到写作时无法使用。在 macOS 上，其蓝牙堆栈在 OS 10.5 版本（2007 年的 Leopard）中添加了 A2DP 支持，应该可以正常工作。

Linux 官方蓝牙堆栈的 BlueZ 蓝牙堆栈最初由高通公司开发，现在包含在官方 Linux 内核发行版中。它是功能最齐全的蓝牙堆栈之一。

随着从 BlueZ 版本 4 到 5 的迁移，ALSA 声音 API 支持被取消，并转移到 PulseAudio 音频系统，同时旧 API 被重命名。这意味着使用旧（版本 4）API 实现的应用程序和代码不再工作。不幸的是，网上找到的大量示例代码和教程仍然针对版本 4，这是需要注意的，因为它们的工作方式非常不同。

BlueZ 通过 D-Bus Linux 系统 IPC（进程间通信）系统进行配置，或者通过直接编辑配置文件。实际上，在类似本章项目中的应用程序中实现 BlueZ 支持以编程方式配置它将会相当复杂，这是因为 API 的范围很大，以及设置配置选项的限制，这些选项不仅限于蓝牙堆栈，还需要访问基于文本的配置文件。因此，应用程序必须以正确的权限运行，以访问某些属性和文件，直接编辑后者或手动执行这些步骤。

对于信息娱乐项目来说，另一个复杂的问题是设置自动配对模式，否则远程设备（智能手机）将无法实际连接到信息娱乐系统。这还需要与蓝牙堆栈进行持续交互，以轮询在此期间可能已连接的新设备。

每个新的设备都需要检查是否支持 A2DP 源模式，如果是的话，它将被添加到系统的音频输入中。然后可以连接到音频系统以利用这个新的输入。

由于此实现的复杂性和范围，它被省略在本章的示例代码中。然而，它可以添加到代码中。例如，Raspberry Pi 3 等 SBC 带有内置的蓝牙适配器。其他设备可以通过 USB 设备添加蓝牙适配器。

# 在线流媒体

有许多在线流媒体服务可以集成到本章所讨论的信息娱乐系统中。它们都使用类似的流式传输 API（通常是基于 HTTP 的 REST API），这要求用户在服务中创建一个账户，通过这个账户可以获得一个特定应用程序的令牌，该令牌允许用户访问该 API，从而可以查询特定艺术家、音乐曲目、专辑等。

使用 HTTP 客户端，例如 Qt 框架中找到的客户端，实现必要的控制流程将会相当容易。由于这些流媒体服务需要注册的应用程序 ID，它被省略在示例代码中。

从 REST API 流式传输的基本序列通常看起来像这样，它围绕 HTTP 调用有一个简单的包装类：

```cpp
#include "soundFoo"
// Create a client object with your app credentials.
client = soundFoo.new('YOUR_CLIENT_ID');
// Fetch track to stream.
track = client.get("/tracks/293")
// Get the tracks streaming URL.
stream_url = client.get(track.stream_url, true); 
// stream URL, allow redirects
// Print the tracks stream URL
std::cout << stream_url.location;
```

# 语音驱动用户界面

这个项目采用了一个完全可以通过语音命令控制的用户界面。为此，它实现了一个由 PocketSphinx 库（见 [`cmusphinx.github.io/`](https://cmusphinx.github.io/)）提供的语音到文本接口，该接口使用关键词检测和语法搜索来识别和解释给它提供的命令。

我们使用随 PocketSphinx 分发版一起提供的默认美国英语语言模型。这意味着任何 spoken 的命令都应该带有美国英语口音，以便能够被准确理解。要更改这一点，可以加载针对不同语言和口音的不同语言模型。各种模型可通过 PocketSphinx 网站获得，并且可以通过一些努力创建自己的语言模型。

# 使用场景

我们不希望信息娱乐系统在每次语音用户界面识别到命令词时都激活，而这些命令词并不是有意为之。防止这种情况发生的常见方法是使用一个激活命令界面的关键词。如果在一定时间内没有识别到命令，系统将恢复到关键词检测模式。

对于这个示例项目，我们使用关键词 `计算机`。当系统检测到这个关键词后，我们可以使用以下命令：

| **命令** | **结果** |
| --- | --- |
| 播放蓝牙 | 从任何连接的 A2DP 源设备开始播放（未实现）。 |
| 停止蓝牙播放 | 停止从任何蓝牙设备播放。 |
| 播放本地 | 播放（硬编码的）本地音乐文件。 |
| 停止本地播放 | 如果正在播放，则停止播放本地音乐文件。 |
| 播放远程 | 从在线流媒体服务或服务器播放（未实现）。 |
| 停止远程播放 | 如果正在播放，则停止播放。 |
| 录制消息 | 记录一条消息。记录直到出现一定秒数的静音。 |
| 播放消息 | 如果有记录的消息，则播放。 |

# 源代码

这个应用程序是使用 Qt 框架实现的，作为一个 GUI 应用程序，这样我们也得到了一个图形界面，便于调试。这个调试 UI 是使用 Qt Creator IDE 中的 Qt Designer 设计的单个 UI 文件。 

我们首先创建一个 GUI 应用程序的实例：

```cpp
#include "mainwindow.h" 
#include <QApplication> 

int main(int argc, char *argv[]) { 
    QApplication a(argc, argv); 
    MainWindow w; 
    w.show(); 

    return a.exec(); 
} 
```

这创建了一个 `MainWindow` 类的实例，我们在其中实现了应用程序，以及一个 `QApplication` 实例，这是一个由 Qt 框架使用的包装类。

接下来，这是 `MainWindow` 的头文件：

```cpp
#include <QMainWindow> 

#include <QAudioRecorder> 
#include <QAudioProbe> 
#include <QMediaPlayer> 

namespace Ui { 
    class MainWindow; 
} 

class MainWindow : public QMainWindow { 
    Q_OBJECT 

public: 
    explicit MainWindow(QWidget *parent = nullptr); 
    ~MainWindow(); 

public slots: 
    void playBluetooth(); 
    void stopBluetooth(); 
    void playOnlineStream(); 
    void stopOnlineStream(); 
    void playLocalFile(); 
    void stopLocalFile(); 
    void recordMessage(); 
    void playMessage(); 

    void errorString(QString err); 

    void quit(); 

private: 
    Ui::MainWindow *ui; 

    QMediaPlayer* player; 
    QAudioRecorder* audioRecorder; 
    QAudioProbe* audioProbe; 

    qint64 silence; // Microseconds of silence recorded so far. 

private slots: 
    void processBuffer(QAudioBuffer); 
}; 
```

其实现包含大部分核心功能，声明了音频录制器和播放器实例，只有语音命令处理是在一个单独的类中处理的：

```cpp
#include "mainwindow.h" 
#include "ui_mainwindow.h" 

#include "voiceinput.h" 

#include <QThread> 
#include <QMessageBox> 

#include <cmath> 

#define MSG_RECORD_MAX_SILENCE_US 5000000 

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent), 
    ui(new Ui::MainWindow) { 
    ui->setupUi(this); 

    // Set up menu connections. 
    connect(ui->actionQuit, SIGNAL(triggered()), this, SLOT(quit())); 

    // Set up UI connections. 
    connect(ui->playBluetoothButton, SIGNAL(pressed), this, SLOT(playBluetooth)); 
    connect(ui->stopBluetoothButton, SIGNAL(pressed), this, SLOT(stopBluetooth)); 
    connect(ui->playLocalAudioButton, SIGNAL(pressed), this, SLOT(playLocalFile)); 
    connect(ui->stopLocalAudioButton, SIGNAL(pressed), this, SLOT(stopLocalFile)); 
    connect(ui->playOnlineStreamButton, SIGNAL(pressed), this, SLOT(playOnlineStream)); 
    connect(ui->stopOnlineStreamButton, SIGNAL(pressed), this, SLOT(stopOnlineStream)); 
    connect(ui->recordMessageButton, SIGNAL(pressed), this, SLOT(recordMessage)); 
    connect(ui->playBackMessage, SIGNAL(pressed), this, SLOT(playMessage)); 

    // Defaults 
    silence = 0; 

    // Create the audio interface instances. 
    player = new QMediaPlayer(this); 
    audioRecorder = new QAudioRecorder(this); 
    audioProbe = new QAudioProbe(this); 

    // Configure the audio recorder. 
    QAudioEncoderSettings audioSettings; 
    audioSettings.setCodec("audio/amr"); 
    audioSettings.setQuality(QMultimedia::HighQuality);     
    audioRecorder->setEncodingSettings(audioSettings);     
    audioRecorder->setOutputLocation(QUrl::fromLocalFile("message/last_message.amr")); 

    // Configure audio probe. 
    connect(audioProbe, SIGNAL(audioBufferProbed(QAudioBuffer)), this, SLOT(processBuffer(QAudioBuffer))); 
    audioProbe->setSource(audioRecorder); 

    // Start the voice interface in its own thread and set up the connections. 
    QThread* thread = new QThread; 
    VoiceInput* vi = new VoiceInput(); 
    vi->moveToThread(thread); 
    connect(thread, SIGNAL(started()), vi, SLOT(run())); 
    connect(vi, SIGNAL(finished()), thread, SLOT(quit())); 
    connect(vi, SIGNAL(finished()), vi, SLOT(deleteLater())); 
    connect(thread, SIGNAL(finished()), thread, SLOT(deleteLater())); 

    connect(vi, SIGNAL(error(QString)), this, SLOT(errorString(QString))); 
    connect(vi, SIGNAL(playBluetooth), this, SLOT(playBluetooth)); 
    connect(vi, SIGNAL(stopBluetooth), this, SLOT(stopBluetooth)); 
    connect(vi, SIGNAL(playLocal), this, SLOT(playLocalFile)); 
    connect(vi, SIGNAL(stopLocal), this, SLOT(stopLocalFile)); 
    connect(vi, SIGNAL(playRemote), this, SLOT(playOnlineStream)); 
    connect(vi, SIGNAL(stopRemote), this, SLOT(stopOnlineStream)); 
    connect(vi, SIGNAL(recordMessage), this, SLOT(recordMessage)); 
    connect(vi, SIGNAL(playMessage), this, SLOT(playMessage)); 

    thread->start(); 
} 
```

在构造函数中，我们设置了 GUI 窗口中所有按钮的 UI 连接，这样我们可以在不使用语音用户界面的情况下触发应用程序的功能。这对于测试目的很有用。

此外，我们创建了一个音频记录器和媒体播放器的实例，以及一个与音频记录器链接的音频探针，这样我们就可以查看它所记录的音频样本并对它们进行操作。

最后，我们创建了一个语音输入接口类的实例，并在启动之前将其推送到自己的线程。我们将它的信号连接到特定的命令，以及其他事件到它们各自的槽中：

```cpp
MainWindow::~MainWindow() { 
    delete ui; 
} 

void MainWindow::playBluetooth() { 
    // Use the link with the BlueZ Bluetooth stack in the Linux kernel to 
    // configure it to act as an A2DP sink for smartphones to connect to. 
} 

// --- STOP BLUETOOTH --- 
void MainWindow::stopBluetooth() { 
    // 
} 
```

如蓝牙技术章节中所述，我们出于该章节中解释的原因，没有实现蓝牙功能。

```cpp
void MainWindow::playOnlineStream() { 
    // Connect to remote streaming service's API and start streaming. 
} 

void MainWindow::stopOnlineStream() { 
    // Stop streaming from remote service. 
} 
```

对于在线流媒体功能也是如此。有关如何实现此功能的详细信息，请参阅本章前面关于在线流媒体的章节。

```cpp
void MainWindow::playLocalFile() { 
    player->setMedia(QUrl::fromLocalFile("music/coolsong.mp3")); 
    player->setVolume(50); 
    player->play(); 
} 

void MainWindow::stopLocalFile() { 
    player->stop(); 
} 
```

要播放本地文件，我们期望在硬编码的路径中找到一个 MP3 文件。然而，通过读取文件名并逐个播放，只需进行少量修改，也可以播放特定文件夹中的所有音乐。

```cpp
void MainWindow::recordMessage() { 
    audioRecorder->record(); 
} 

void MainWindow::playMessage() { 
    player->setMedia(QUrl::fromLocalFile("message/last_message.arm")); 
    player->setVolume(50); 
    player->play(); 
} 
```

在构造函数中，我们配置了记录器将记录保存到名为 `message` 的子文件夹中的文件。如果进行新的录音，这将会被覆盖，从而允许留下可以稍后回放的留言。可选的显示屏或其他附件可以用来指示何时进行了新的录音且尚未收听：

```cpp
void MainWindow::processBuffer(QAudioBuffer buffer) { 
    const quint16 *data = buffer.constData<quint16>(); 

    // Get RMS of buffer, if silence, add its duration to the counter. 
    int samples = buffer.sampleCount(); 
    double sumsquared = 0; 
    for (int i = 0; i < samples; i++) { 
        sumsquared += data[i] * data[i]; 
    } 

    double rms = sqrt((double(1) / samples)*(sumsquared)); 

    if (rms <= 100) { 
        silence += buffer.duration(); 
    } 

    if (silence >= MSG_RECORD_MAX_SILENCE_US) { 
        silence = 0; 
        audioRecorder->stop(); 
    } 
} 
```

当记录器处于活动状态时，我们的音频探针会调用此方法。在这个函数中，我们计算音频缓冲区的**均方根**（**RMS**）值以确定它是否充满了静音。在这里，静音是相对的，可能需要根据录音环境进行调整。

在检测到五秒钟的静音后，消息的录音停止：

```cpp
void MainWindow::errorString(QString err) { 
    QMessageBox::critical(this, tr("Error"), err); 
} 

void MainWindow::quit() { 
    exit(0); 
} 
```

剩余的方法处理可能在其他地方发出的错误消息的报道，以及终止应用程序。

`VoiceInput` 类头定义了语音输入界面的功能：

```cpp
#include <QObject> 
#include <QAudioInput> 

extern "C" { 
#include "pocketsphinx.h" 
} 

class VoiceInput : public QObject { 
    Q_OBJECT 

    QAudioInput* audioInput; 
    QIODevice* audioDevice; 
    bool state; 

public: 
    explicit VoiceInput(QObject *parent = nullptr); 
    bool checkState() { return state; } 

signals: 
    void playBluetooth(); 
    void stopBluetooth(); 
    void playLocal(); 
    void stopLocal(); 
    void playRemote(); 
    void stopRemote(); 
    void recordMessage(); 
    void playMessage(); 

    void error(QString err); 

public slots: 
    void run(); 
}; 
```

由于 PocketSphinx 是一个 C 库，我们必须确保它以 C 风格的链接使用。除此之外，我们创建了语音输入将使用的音频输入和相关 IO 设备的类成员。

接下来是类的定义：

```cpp
#include <QDebug> 
#include <QThread> 

#include "voiceinput.h" 

extern "C" { 
#include <sphinxbase/err.h> 
#include <sphinxbase/ad.h> 
} 

VoiceInput::VoiceInput(QObject *parent) : QObject(parent) { 
    // 
} 
```

构造函数没有做任何特殊的事情，因为下一个方法负责所有的初始化和主循环的设置：

```cpp
void VoiceInput::run() { 
    const int32 buffsize = 2048; 
    int16 adbuf[buffsize]; 
    uint8 utt_started, in_speech; 
    uint32 k = 0; 
    char const* hyp; 

    static ps_decoder_t *ps; 

    state = true; 

    QAudioFormat format; 
    format.setSampleRate(16000); 
    format.setChannelCount(1); 
    format.setSampleSize(16); 
    format.setCodec("audio/pcm"); 
    format.setByteOrder(QAudioFormat::LittleEndian); 
    format.setSampleType(QAudioFormat::UnSignedInt); 

    // Check that the audio device supports this format. 
    QAudioDeviceInfo info = QAudioDeviceInfo::defaultInputDevice(); 
    if (!info.isFormatSupported(format)) { 
       qWarning() << "Default format not supported, aborting."; 
       state = false; 
       return; 
    } 

    audioInput = new QAudioInput(format, this); 
    audioInput->setBufferSize(buffsize * 2);    
    audioDevice = audioInput->start(); 

    if (ps_start_utt(ps) < 0) { 
        E_FATAL("Failed to start utterance\n"); 
    } 

    utt_started = FALSE; 
    E_INFO("Ready....\n"); 
```

此方法的第一部分设置了音频接口，配置它使用 PocketSphinx 所需的音频格式设置：单声道、小端序、16 位有符号 PCM 音频，频率为 16,000 赫兹。在确认音频输入支持此格式后，我们创建一个新的音频输入实例：

```cpp
    const char* keyfile = "COMPUTER/3.16227766016838e-13/\n"; 
    if (ps_set_kws(ps, "keyword_search", keyfile) != 0) { 
        return; 
    } 

    if (ps_set_search(ps, "keyword_search") != 0) { 
        return; 
    } 

    const char* gramfile = "grammar asr;\ 
            \ 
            public <rule> = <action> [<preposition>] [<objects>] [<preposition>] [<objects>];\ 
            \ 
            <action> = STOP | PLAY | RECORD;\ 
            \ 
            <objects> = BLUETOOTH | LOCAL | REMOTE | MESSAGE;\ 
            \ 
            <preposition> = FROM | TO;"; 
    ps_set_jsgf_string(ps, "jsgf", gramfile); 
```

接下来，我们设置了在音频样本处理过程中将使用的关键字检测和 JSGF 语法文件。通过第一个 `ps_set_search()` 函数调用，我们开始关键字检测搜索。接下来的循环将一直处理样本，直到检测到 `computer` 这个词：

```cpp
    bool kws = true; 
    for (;;) { 
        if ((k = audioDevice->read((char*) &adbuf, 4096))) { 
            E_FATAL("Failed to read audio.\n"); 
        } 

        ps_process_raw(ps, adbuf, k, FALSE, FALSE); 
        in_speech = ps_get_in_speech(ps); 

        if (in_speech && !utt_started) { 
            utt_started = TRUE; 
            E_INFO("Listening...\n"); 
        } 
```

在每个循环中，我们读取另一个缓冲区的音频样本，然后让 PocketSphinx 处理这些样本。它还为我们进行静音检测，以确定是否有人开始对着麦克风说话。如果有人说话但我们还没有开始解释，我们会开始一个新的语音：

```cpp
        if (!in_speech && utt_started) { 
            ps_end_utt(ps); 
            hyp = ps_get_hyp(ps, nullptr); 
            if (hyp != nullptr) { 
                // We have a hypothesis. 

                if (kws && strstr(hyp, "computer") != nullptr) { 
                    if (ps_set_search(ps, "jsgf") != 0) { 
                        E_FATAL("ERROR: Cannot switch to jsgf mode.\n"); 
                    } 

                    kws = false; 
                    E_INFO("Switched to jsgf mode \n");                             
                    E_INFO("Mode: %s\n", ps_get_search(ps)); 
                } 
                else if (!kws) { 
                    if (hyp != nullptr) { 
                        // Check each action. 
                        if (strncmp(hyp, "play bluetooth", 14) == 0) { 
                            emit playBluetooth(); 
                        } 
                        else if (strncmp(hyp, "stop bluetooth", 14) == 0) { 
                            emit stopBluetooth(); 
                        } 
                        else if (strncmp(hyp, "play local", 10) == 0) { 
                            emit playLocal(); 
                        } 
                        else if (strncmp(hyp, "stop local", 10) == 0) { 
                            emit stopLocal(); 
                        } 
                        else if (strncmp(hyp, "play remote", 11) == 0) { 
                            emit stopBluetooth(); 
                        } 
                        else if (strncmp(hyp, "stop remote", 11) == 0) { 
                            emit stopBluetooth(); 
                        } 
                        else if (strncmp(hyp, "record message", 14) == 0) { 
                            emit stopBluetooth(); 
                        } 
                        else if (strncmp(hyp, "play message", 12) == 0) { 
                            emit stopBluetooth(); 
                        } 
                    } 
                    else { 
                        if (ps_set_search(ps, "keyword_search") != 0){ 
                            E_FATAL("ERROR: Cannot switch to kws mode.\n"); 
                        } 

                        kws = true; 
                        E_INFO("Switched to kws mode.\n"); 
                    } 
                }                 
            } 

            if (ps_start_utt(ps) < 0) { 
                E_FATAL("Failed to start utterance\n"); 
            } 

            utt_started = FALSE; 
            E_INFO("Ready....\n"); 
        } 

        QThread::msleep(100); 
    } 

} 
```

该方法的其余部分检查我们是否有可用的假设可以分析。根据我们是否处于关键词或语法模式，在前者情况下检查关键词的检测，并切换到语法模式。如果我们已经在语法模式中，我们尝试将语音缩小到特定的命令，此时我们将发出相关的信号，这将触发连接的功能。

每当 PocketSphinx 检测到至少一秒钟的静音时，就会开始一个新的语音。在执行命令后，系统会切换回关键词检测模式。

# 构建项目

要构建项目，首先必须构建 PocketSphinx 项目。在本章附带示例项目的源代码中，`sphinx`文件夹下有两个 Makefile，一个在`pocketsphinx`文件夹中，另一个在`sphinxbase`文件夹中。使用这些 Makefile，将构建形成 PocketSphinx 的两个库。

然后，可以从 Qt Creator 或通过执行以下命令从命令行构建 Qt 项目：

```cpp
mkdir build
cd build
qmake ..
make
```

# 扩展系统

除了音频格式，还可以添加播放视频的功能，并集成使用蓝牙 API 进行拨打电话和接听电话的能力。有人可能希望扩展应用程序，使其更加灵活和模块化，例如，可以添加一个模块来添加语音命令和相应的动作。

有语音输出也会很方便，使其更符合当前的商业产品。为此，可以使用 Qt 框架中可用的文本到语音 API。

通过查询远程 API 获取更多*信息*，例如当前天气、新闻更新，甚至可能对正在进行的足球比赛进行更新，这将非常有用。基于语音的用户界面可以用来设置计时器和任务提醒，集成日历，等等。

最后，正如本章的示例代码所示，无法指定想要播放的曲目名称、特定的专辑或艺术家名称。允许这种自由式输入非常有用，但也会带来它自己的一系列问题。

主要问题是语音到文本系统的识别率，尤其是对于它字典中没有的单词。我们中的一些人可能已经体验过在手机、汽车或智能手机上尝试让语音驱动的用户界面理解某个单词的乐趣。

到目前为止，这仍然是一个重要的研究方向，没有快速简便的解决方案。理论上可以通过使用本地音频文件名和艺术家索引以及作为字典一部分的其他元数据来暴力破解这种识别，从而获得更高的准确性。同样，对于远程流媒体服务，也可以通过查询其 API 来实现。然而，这可能会给识别过程增加相当大的延迟。

# 摘要

在本章中，我们探讨了如何相对容易地构建一个基于 SBC 的信息娱乐系统，使用语音到文本构建语音驱动用户界面。我们还探讨了如何扩展它以添加更多功能。

预期读者能够在这个阶段实现一个类似的系统，并将其扩展以连接到在线和网络服务。读者还应了解更高级的语音驱动用户界面的实现、文本到语音的添加以及基于 A2DP 的蓝牙设备的使用。

在下一章中，我们将探讨如何使用微控制器和本地网络实现整个建筑的监控和控制系统。
