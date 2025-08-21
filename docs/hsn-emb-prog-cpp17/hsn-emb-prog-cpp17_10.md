# 第八章：示例-基于 Linux 的信息娱乐系统

本章提供了一个使用基于 Linux 的单板计算机（SBC）实现信息娱乐系统的示例。它还展示了如何使用蓝牙连接远程设备，以及如何使用在线流媒体服务。最终的设备将能够从各种来源播放音频，而无需复杂的用户界面。具体来说，我们将涵盖以下主题：

+   为基于 Linux 的 SBC 开发

+   在 Linux 下使用蓝牙

+   从各种来源播放音频和录制音频

+   使用 GPIO 进行简单输入和语音识别

+   连接到在线流媒体音频服务

# 一个能够完成所有功能的盒子

信息娱乐系统已经成为我们日常生活中的常见功能，从车载娱乐系统（ICE）开始（也称为车载信息娱乐或 IVI），它们从基本的收音机和磁带播放器发展到包括诸如导航和通过蓝牙连接智能手机以访问音乐库等功能。另一个重要功能是为驾驶员提供免提功能，使他们可以开始电话并控制收音机，而无需将目光从道路上移开或双手离开方向盘。

随着智能手机的普及，为用户提供持续访问新闻、天气和娱乐，使用语音驱动界面的车载助手的到来，最终导致了面向家庭使用的语音驱动信息娱乐系统的到来。这些通常包括扬声器和麦克风，以及用于语音驱动界面和访问所需的互联网服务的必要硬件。

本章将主要关注这种类型的语音驱动信息娱乐系统。在第十章中，*使用 Qt 开发嵌入式系统*，我们将深入研究添加图形用户界面。

我们想要在这里实现的目标是：

+   从蓝牙源（如智能手机）播放音乐

+   从在线流媒体服务播放音乐

+   从本地文件系统（包括 USB 存储设备）播放音乐

+   录制音频片段并在需要时重复播放

+   用语音控制所有操作，并为一些操作配备按钮

在接下来的章节中，我们将看看这些目标以及如何实现它们。

# 所需的硬件

对于这个项目，任何能够运行 Linux 的 SBC 都应该可以。它还需要具备以下功能以进行完整实现：

+   互联网连接（无线或有线）以访问在线内容。

+   蓝牙功能（内置或作为附加模块）以使系统能够充当蓝牙扬声器。

+   释放 GPIO 输入以允许连接按钮。

+   用于语音输入和音频播放的功能麦克风输入和音频输出。

+   SATA 连接或类似连接用于连接硬盘等存储设备。

+   I2C 总线外设用于 I2C 显示器。

在本章的示例代码中，我们只需要麦克风输入和音频输出，以及一些用于本地媒体文件存储的存储空间。

对于 GPIO 引脚，我们可以连接一些按钮，用于控制信息娱乐系统，而无需使用语音激活系统。这在使用语音激活系统会很尴尬的情况下非常方便，比如在接听电话时暂停或静音音乐。

连接按钮在本示例中不会进行演示，但可以在第三章的早期项目中找到示例，即*开发嵌入式 Linux 和类似系统*。在那里，我们使用 WiringPi 库将开关连接到 GPIO 引脚，并配置中断例程来处理这些开关上的变化。

如果需要显示当前信息，比如当前歌曲的名称或其他相关状态信息，也可以将小型显示器连接到系统上。16x2 字符的廉价显示器可以通过 I2C 接口进行控制，而且有各种 OLED 和其他小型显示器可供选择，由于其最低硬件要求，它们非常适合这个用途。

在第三章《开发嵌入式 Linux 和类似系统》中，我们简要介绍了为这样的信息娱乐系统使用什么样的硬件，以及一些可能的用户界面和存储选项。当然，正确的硬件配置取决于个人的需求。如果想要本地存储大量音乐进行播放，连接到系统的大型 SATA 硬盘将非常方便。

然而，对于本章的示例，我们不会做出这样的假设，而是更像一个易于扩展的起点。因此，硬件要求非常低，除了明显需要麦克风和音频输出之外。

# 软件需求

对于这个项目，我们假设 Linux 已经安装在目标 SBC 上，并且硬件功能的驱动程序，如麦克风和音频输出的驱动程序已经安装和配置好。

由于我们在这个项目中使用 Qt 框架，因此所有依赖项也应该得到满足。这意味着生成项目的二进制文件所在的系统上应该存在共享库。Qt 框架可以通过操作系统的软件包管理器获得，也可以通过 Qt 网站[`qt.io/`](http://qt.io/)获得。

在第十章《使用 Qt 开发嵌入式系统》中，我们将更详细地研究在嵌入式平台上使用 Qt 进行开发。本章将简要介绍 Qt API 的使用。

根据我们是否想要直接在 SBC 上编译应用程序，还是在开发 PC 上编译应用程序，我们可能需要在 SBC 上安装编译器工具链和其他依赖项，或者在目标 SBC（ARM、x86 或其他架构）上安装交叉编译工具链。在第六章《测试基于操作系统的应用程序》中，我们研究了为 SBC 系统进行交叉编译，以及在本地测试系统。

由于本章的示例项目不需要任何特殊的硬件，因此可以直接在任何受 Qt 框架支持的系统上进行编译。这是在将代码部署到 SBC 之前测试代码的推荐方式。

# 蓝牙音频源和接收器

不幸的是，蓝牙是一种专有技术，尽管它无处不在，但由于其专有性质，它缺乏对蓝牙功能的全面支持（以配置文件的形式）。我们在这个项目中感兴趣的配置文件称为**高级音频分发配置文件**（**A2DP**）。这是一种用于流式传输音频的配置文件，从蓝牙耳机到蓝牙音箱都在使用。

任何实现 A2DP 的设备都可以将音频流式传输到 A2DP 接收器，或者可以自己充当接收器（取决于 BT 堆栈的实现）。理论上，这将允许某人连接智能手机或类似设备到我们的信息娱乐系统，并在其上播放音乐，就像连接独立的蓝牙音箱一样。

A2DP 配置文件中的接收器是 A2DP 接收器，而另一侧是 A2DP 源。蓝牙耳机或音箱设备始终是接收器设备，因为它们只能消耗音频流。PC、SBC 或类似的多用途设备可以配置为充当接收器或源。

正如前面提到的，主流操作系统上实现完整蓝牙堆栈的复杂性导致对蓝牙的基本串行通信功能以外的支持不足。

虽然 FreeBSD、macOS、Windows 和 Android 都有蓝牙堆栈，但它们在支持的蓝牙适配器数量（Windows 只支持一个，而且只支持 USB 适配器）、支持的配置文件（FreeBSD 只支持数据传输）和可配置性方面存在限制（Android 基本上只针对智能手机）。

对于 Windows 10，A2DP 配置文件支持目前已经从 Windows 7 中的功能性退化到了在撰写本文时不再起作用，这是由于其蓝牙堆栈的更改。而 macOS 的蓝牙堆栈在操作系统的 10.5 版本（2007 年的 Leopard）中添加了 A2DP 支持，并应该可以正常工作。

BlueZ 蓝牙堆栈已成为 Linux 的官方蓝牙堆栈，最初由高通开发，现在已包含在官方 Linux 内核发行版中。它是最全面的蓝牙堆栈之一。

从 BlueZ 版本 4 到 5 的转变中，ALSA 音频 API 支持被删除，而是转移到了 PulseAudio 音频系统，并且旧 API 的名称也被更改。这意味着使用旧（版本 4）API 实现的应用程序和代码不再起作用。不幸的是，许多在线找到的示例代码和教程仍然针对版本 4，这是需要注意的，因为它们的工作方式有很大不同。

BlueZ 通过 D-Bus Linux 系统 IPC（进程间通信）系统进行配置，或者直接编辑配置文件。然而，在像本章项目中那样以编程方式配置它实际上会相当复杂，因为 API 的范围非常广，而且在设置超出蓝牙堆栈的配置选项时需要访问基于文本的配置文件。因此，应用程序必须以正确的权限运行，以访问某些属性和文件，直接编辑后者或手动执行这些步骤。

信息娱乐项目的另一个复杂之处是设置自动配对模式，否则远程设备（智能手机）将无法连接到信息娱乐系统。这还需要与蓝牙堆栈进行持续交互，以便在此期间轮询任何新连接的设备。

每个新设备都需要检查是否支持 A2DP 源模式，如果支持，它将被添加到系统的音频输入中。然后可以连接到音频系统，利用新的输入。

由于这个实现的复杂性和范围，它被省略在本章的示例代码中。但是，它可以被添加到代码中。像树莓派 3 这样的 SBC 带有内置蓝牙适配器。其他设备可以使用 USB 设备添加蓝牙适配器。

# 在线流媒体

有许多在线流媒体服务可以集成到类似于本章所研究的信息娱乐系统中。它们都使用类似的流媒体 API（通常是基于 HTTP 的 REST API），需要用户使用该服务创建一个帐户，从而获取一个特定于应用程序的令牌，以便访问该 API，允许用户查询特定的艺术家、音乐曲目、专辑等。

使用 HTTP 客户端，比如在 Qt 框架中找到的客户端，实现必要的控制流将会相当容易。由于需要为这些流媒体服务注册应用程序 ID，因此它被省略在示例代码中。

从 REST API 流式传输的基本顺序通常是这样的，使用一个简单的 HTTP 调用包装类：

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

该项目采用了完全可通过语音命令控制的用户界面。为此，它实现了由 PocketSphinx 库提供动力的语音到文本接口（参见[`cmusphinx.github.io/`](https://cmusphinx.github.io/)），它使用关键词识别和语法搜索来识别和解释给定的命令。

我们使用了随 PocketSphinx 发行的默认的美式英语语言模型。这意味着任何口语命令都应该以美式英语口音发音，以便准确理解。要更改这一点，可以加载针对不同语言和口音的不同语言模型。通过 PocketSphinx 网站可以获得各种模型，也可以通过一些努力制作自己的语言模型。

# 使用场景

我们不希望信息娱乐系统在语音用户界面识别到命令词时每次都被激活，而这些命令词并非有意为之。防止这种情况发生的常见方法是有一个关键词来激活命令界面。如果在关键词之后一定时间内没有识别到命令，系统将恢复到关键词识别模式。

对于这个示例项目，我们使用关键词`computer`。系统识别到这个关键词后，我们可以使用以下命令：

| **命令** | **结果** |
| --- | --- |
| 播放蓝牙 | 从任何连接的 A2DP 源设备开始播放（未实现）。 |
| 停止蓝牙 | 停止从任何蓝牙设备播放。 |
| 播放本地 | 播放（硬编码的）本地音乐文件。 |
| 停止本地 | 如果当前正在播放本地音乐文件，则停止播放。 |
| 播放远程 | 从在线流媒体服务或服务器播放（未实现）。 |
| 停止远程 | 如果正在播放，则停止播放。 |
| 录制消息 | 录制一条消息。录制直到发生一定时间的静音。 |
| 播放消息 | 如果有录制的消息，则播放回。 |

# 源代码

该应用程序是使用 Qt 框架实现的 GUI 应用程序，因此我们还获得了一个用于调试的图形界面。这个调试 UI 是使用 Qt Creator IDE 的 Qt Designer 设计的单个 UI 文件。

我们首先创建了 GUI 应用程序的实例：

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

这在`MainWindow`类中创建了一个实例，我们在其中实现了应用程序，以及`QApplication`的实例，这是 Qt 框架使用的包装类。

接下来，这是`MainWindow`的标题：

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

其实现包含大部分核心功能，声明了音频录制器和播放器实例，只是声音命令处理是在一个单独的类中处理的：

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

在构造函数中，我们为 GUI 窗口中的按钮设置了所有 UI 连接，以便我们可以触发应用程序的功能，而无需使用语音用户界面。这对于测试目的很有用。

此外，我们创建了音频录制器和媒体播放器的实例，以及与音频录制器链接的音频探针，以便我们可以查看它正在录制的音频样本并对其进行操作。

最后，我们创建了语音输入接口类的实例，并在启动之前将其推送到自己的线程上。我们将其信号连接到特定命令，其他事件连接到它们各自的插槽：

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

正如在蓝牙技术部分中提到的，出于该部分所述的原因，我们未实现了蓝牙功能。

```cpp
void MainWindow::playOnlineStream() { 
    // Connect to remote streaming service's API and start streaming. 
} 

void MainWindow::stopOnlineStream() { 
    // Stop streaming from remote service. 
} 
```

在线流功能也是如此。有关如何实现此功能的详细信息，请参阅本章前面关于在线流的部分。

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

要播放本地文件，我们期望在硬编码路径中找到一个 MP3 文件。但是，也可以通过读取文件名并逐个播放它们来播放特定文件夹中的所有音乐。

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

在构造函数中，我们配置了录音机将记录到一个名为`message`的子文件夹中的文件中。如果进行新的录音，这将被覆盖，允许留下一条可以稍后播放的消息。可选的显示器或其他附件可以用来指示是否已经进行了新的录音并且尚未被听过：

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

每当录音机处于活动状态时，我们的音频探测器就会调用这个方法。在这个函数中，我们计算音频缓冲区的**均方根**（**RMS**）值，以确定它是否充满了静默。在这里，静默是相对的，可能需要根据录音环境进行调整。

在检测到五秒的静默后，消息的录制将停止：

```cpp
void MainWindow::errorString(QString err) { 
    QMessageBox::critical(this, tr("Error"), err); 
} 

void MainWindow::quit() { 
    exit(0); 
} 
```

其余的方法处理可能在应用程序的其他地方发出的错误消息的报告，以及终止应用程序。

`VoiceInput`类头文件定义了语音输入接口的功能：

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

由于 PocketSphinx 是一个 C 库，我们必须确保它使用 C 风格的链接。除此之外，我们为音频输入和相关 IO 设备创建了类成员，语音输入将使用这些成员。

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

构造函数并没有做任何特殊的事情，因为接下来的方法将初始化和设置主循环：

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

这个方法的第一部分设置了音频接口，配置它使用 PocketSphinx 所需的音频格式设置进行录制：单声道，小端，16 位有符号 PCM 音频，采样率为 16,000 赫兹。在检查音频输入是否支持这种格式后，我们创建了一个新的音频输入实例：

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

接下来，我们设置了在处理音频样本时将使用的关键词检测和 JSGF 语法文件。通过第一个`ps_set_search()`函数调用，我们开始了关键词检测搜索。接下来的循环将持续处理样本，直到检测到`computer`这个话语：

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

每个周期，我们读入另一个缓冲区的音频样本，然后让 PocketSphinx 处理这些样本。它还为我们进行了静默检测，以确定是否有人开始对麦克风说话。如果有人在说话，但我们还没有开始解释，我们就开始一个新的话语：

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

方法的其余部分检查我们是否有可用的假设可以分析。根据我们是处于关键词模式还是语法模式，我们在前一种情况下检查关键词的检测并切换到语法模式。如果我们已经处于语法模式，我们尝试将话语缩小到特定的命令，此时我们将发出相关信号，触发连接的功能。

每当 PocketSphinx 检测到至少一秒的静默时，就会开始一个新的话语。执行命令后，系统会切换回关键词检测模式。

# 构建项目

要构建项目，必须先构建 PocketSphinx 项目。在本章附带的示例项目源代码中，`sphinx`文件夹下有两个 Makefile，一个在`pocketsphinx`文件夹中，另一个在`sphinxbase`文件夹中。通过这些，将构建形成 PocketSphinx 的两个库。

在此之后，可以通过执行以下命令从 Qt Creator 或命令行构建 Qt 项目：

```cpp
mkdir build
cd build
qmake ..
make
```

# 扩展系统

除了音频格式，还可以添加播放视频和集成制作和回复电话的能力（使用蓝牙 API）。可能希望扩展应用程序，使其更灵活和模块化，例如，可以添加一个模块，用于添加语音命令和相应的操作。

具有语音输出也将很方便，使其更符合当前的商业产品。为此，可以使用 Qt 框架中提供的文本到语音 API。

通过查询远程 API 获取更多*信息*，例如当前天气、新闻更新，甚至是当前足球比赛的更新，也将非常有用。基于语音的用户界面可以用于设置定时器和任务提醒，集成日历等等。

最后，正如本章示例代码所示，人们无法指定要播放的曲目名称，或特定的专辑或艺术家名称。允许这种自由输入非常有用，但也带来了一系列问题。

主要问题是语音转文本系统的识别率，特别是对于其词典中没有的单词。我们中的一些人可能已经有幸提高了声音，试图让手机、汽车或智能手机上的语音驱动用户界面理解某个词。

在这一点上，这仍然是一个需要大量研究的重点，没有快速简单的解决方案。可以想象通过使用本地音频文件名和艺术家的索引，以及其他元数据作为词典的一部分，来强制进行这种识别，并通过查询其 API 来对远程流媒体服务进行更准确的识别。然而，这可能会给识别工作增加相当大的延迟。

# 总结

在本章中，我们看了如何相当容易地构建基于 SBC 的信息娱乐系统，使用语音转文本来构建语音驱动用户界面。我们还看了如何扩展它以添加更多功能。

预计读者能够在这一点上实施类似的系统，并将其扩展到连接在线和基于网络的服务。读者还应该阅读更高级的语音驱动用户界面的实施，添加文本到语音，以及使用基于 A2DP 的蓝牙设备。

在下一章中，我们将看看如何使用微控制器和本地网络实现建筑范围的监控和控制系统。
