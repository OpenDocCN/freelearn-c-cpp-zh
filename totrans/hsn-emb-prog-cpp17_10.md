# Example - Linux-Based Infotainment System

This chapter provides an example of how to implement an infotainment system using a Linux-based **single-board computer** (**SBC**). It also shows how to connect to remote devices using Bluetooth, and how to use online streaming services. The resulting device will be able to play back audio from a variety of sources without a complex UI. In particular, we will be covering the following topics:

*   Developing for a Linux-based SBC
*   Using Bluetooth under Linux
*   Playing back audio from a variety of sources and recording audio
*   Using GPIO for both simple input and voice recognition
*   Connecting to online streaming audio services

# One box that does everything

Infotainment systems have become a common feature in our daily lives, starting with **in-car entertainment** (**ICE**) systems (also known as **In-Vehicle Infotainment** or **IVI**), which evolved from the basic radios and cassette players to include features such as navigation and connecting to smartphones over Bluetooth for access to one's music library, and much more. Another big feature is to provide the driver with hands-free functionality so that they can start a phone call and control the radio without having to take their eyes off the road or their hands off the steering wheel.

As smartphones became more popular, providing their users with constant access to news, weather, and entertainment, the arrival of onboard assistants that use a voice-driven interface, both on smartphones and ICEs, ultimately led to the arrival of speech-driven infotainment systems aimed at in-home use. These usually consist of a speaker and microphone, along with the required hardware for the voice-driven interface and access to the required internet-based services.

This chapter will mostly focus on this type of voice-driven infotainment system. In [Chapter 10](c3f58bee-de57-4934-95e8-18c78b667373.xhtml), *Developing Embedded Systems with Qt*, we will take an in-depth look at adding a graphical user interface.

The goals which we want to achieve here are the following:

*   Play music from a Bluetooth source, such as a smartphone
*   Play music from an online streaming service
*   Play music from the local filesystem, including USB sticks
*   Record an audio clip and repeat it when asked
*   Control all actions with one's voice, with buttons for some actions

In the next sections, we'll look at these goals and how to accomplish them.

# Hardware needed

For this project, any SBC that's capable of running Linux should work. It also needs to have the following features for a full implementation:

*   An internet connection (wireless or wired) to access online content.
*   Bluetooth functionality (built-in or as an add-on module) to allow the system to act as a Bluetooth speaker.
*   Free GPIO input to allow for buttons to be hooked up.
*   A functioning microphone input and audio output for the voice input and audio playback, respectively.
*   SATA connectivity or similar for connecting storage devices like hard-drives.
*   I2C bus peripheral for an I2C display.

For the example code in this chapter we only require the microphone input and audio output, along with some storage for local media files.

To the GPIO pins, we can connect a number of buttons that can be used to control the infotainment system without having to use the voice-activated system. This is convenient for situations where using the voice-activated system would be awkward, such as when pausing or muting music when taking a phone call.

Connecting the buttons will not be demonstrated in this example, but an example can be found in an earlier project in [Chapter 3](47e0b6fb-cb68-43c3-9453-2dc7575b1a46.xhtml), *Developing for Embedded Linux and Similar Systems*. There, we used the WiringPi library to connect switches to GPIO pins and configured interrupt routines to handle changes on these switches.

One could also connect a small display to the system, if one wanted to show current information, such as the name of the current song or other relevant status information. Cheap displays of 16x2 characters, which can be controlled over an I2C interface, are widely available; these, along with a range of OLED and other small displays, would be suitable for this purpose thanks to their minimal hardware requirements.

In [Chapter 3](47e0b6fb-cb68-43c3-9453-2dc7575b1a46.xhtml), *Developing for Embedded Linux and Similar Systems*, we had a brief look at what kind of hardware one might want to use for an infotainment system such as this, along with a number of possible user interfaces and storage options. What the right hardware configuration is, of course, depends on one's requirements. If one wants to store a lot of music locally for playback, having a large SATA hard drive connected to the system would be highly convenient.

For the example in this chapter, however, we will make no such assumptions, acting more as an easily extensible starting point. The hardware requirements are therefore very minimal, beyond the obvious need for a microphone and an audio output.

# Software requirements

For this project, we are assuming that Linux has been installed on the target SBC, and that the drivers for the hardware functionality, such as the microphone and audio output, have been installed and configured.

Since we use the Qt framework for this project, all dependencies there should be met as well. This means that the shared libraries should be present on the system on which the resulting binary for the project will be run. The Qt framework can be obtained via the package manager of the OS, or via the Qt website at [http://qt.io/](http://qt.io/).

In [Chapter 10](c3f58bee-de57-4934-95e8-18c78b667373.xhtml), *Developing Embedded Systems with Qt*, we will look at developing on embedded platforms with Qt in more detail. This chapter will briefly touch upon the use of Qt APIs.

Depending on whether we want to compile the application directly on the SBC or on our development PC, we might have to install the compiler toolchain and further dependencies on the SBC, or the cross-compiling toolchain for Linux on the target SBC (ARM, x86, or other architecture). In [Chapter 6](7d5d654f-a027-4825-ab9e-92c369b576a8.xhtml), *Testing OS-Based Applications*, we looked at cross-compiling for SBC systems, as well as testing the system locally.

As the example project in this chapter doesn't require any special hardware, it can be compiled directly on any system that's supported by the Qt framework. This is the recommended way to test out the code prior to deploying it on the SBC.

# Bluetooth audio sources and sinks

Bluetooth is unfortunately a technology that, despite being ubiquitous, suffers from its proprietary nature. As a result, support for the full range of Bluetooth functionality (in the form of profiles) is lacking. The profile that we are interested in for this project is called **Advanced Audio Distribution Profile** (**A2DP**). This is the profile used by everything from Bluetooth headphones to Bluetooth speakers in order to stream audio.

Any device that implements A2DP can stream audio to an A2DP receiver or can themselves act as a receiver (depending on the BT stack implementation). Theoretically, this would allow someone to connect with a smartphone or similar device to our infotainment system and play back music on it, as they would with a standalone Bluetooth speaker.

A receiver in the A2DP profile is an A2DP sink, whereas the other side is the A2DP source. A Bluetooth headphone or speaker device would always be a sink device as they can only consume an audio stream. A PC, SBC, or similar multi-purpose device can be configured to act as either a sink or a source.

As mentioned earlier, the complications surrounding the implementation of a full Bluetooth stack on mainstream OSes has led to lackluster support for anything more than the basic serial communication functionality of Bluetooth.

While FreeBSD, macOS, Windows, and Android all have Bluetooth stacks, they are limited in the number of Bluetooth adapters they can support (just one on Windows, and only USB adapters), the profiles they support (FreeBSD is data-transfer-only), and configurability (Android is essentially only targeted at smartphones).

For Windows 10, A2DP profile support has currently regressed from being functional in Windows 7 to not being functional as of the time of writing due to changes to its Bluetooth stack. With macOS, its Bluetooth stack added A2DP support with version 10.5 of the OS (Leopard, in 2007) and should function.

The BlueZ Bluetooth stack that has become the official Bluetooth stack for Linux was originally developed by Qualcomm and is now included with official Linux kernel distributions. It's one of the most full-featured Bluetooth stacks.

With the move from BlueZ version 4 to 5, ALSA sound API support was dropped, and instead moved to the PulseAudio audio system, along with the renaming of the old APIs. This means that applications and code implemented using the old (version 4) API no longer work. Unfortunately a lot of the example code and tutorials one finds online still targets the version 4, which is something to be aware of, as they work very differently.

BlueZ is configured via the D-Bus Linux system IPC (interprocess communication) system, or by editing configuration files directly. Actually implementing BlueZ support in an application like that in this chapter's project to configure it programmatically would be fairly complicated however, due to the sheer scope of the APIs, as well the limitations in setting configuration options that go beyond just the Bluetooth stack and require access to text-based configuration files. The application would therefore have to run with the correct permissions to access certain properties and files, editing the latter directly or performing those steps manually.

Another complication for the infotainment project is setting up an automatic pairing mode, as otherwise the remote device (smartphone) would be unable to actually connect to the infotainment system. This would require constant interaction with the Bluetooth stack as well, to poll it for any new devices that may have connected in the meantime.

Each new device would have to be checked to see whether it supports the A2DP source mode, in which case it would be added to the audio input for the system. One could then hook into the audio system to make use of that new input.

Due to the complexity and scope of this implementation, it was left out of the example code in this chapter. It could, however, be added to the code. SBCs such as the Raspberry Pi 3 come with a built-in Bluetooth adapter. Others can have a Bluetooth adapter added using a USB device.

# Online streaming

There are a number of online streaming services which one could integrate into an infotainment system like the type which are looking at in this chapter. All of them use a similar streaming API (usually an HTTP-based REST API), which requires one to create an account with the service, using which one can obtain an application-specific token that gives one access to that API, allowing one to query it for specific artists, music tracks, albums, and so on.

Using an HTTP client, such as the one found in the Qt framework, it would be fairly easy to implement the necessary control flow. Due to the requirement of having a registered application ID for those streaming services, it was left out of the example code.

The basic sequence to stream from a REST API usually looks like this, with a simple wrapper class around the HTTP calls:

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

# Voice-driven user interface

This project employs a user interface that is fully controllable by voice commands. For this, it implements a voice-to-text interface powered by the PocketSphinx library (see [https://cmusphinx.github.io/](https://cmusphinx.github.io/)) that uses both keyword-spotting and a grammar search in order to recognize and interpret commands given to it.

We use the default US-English language model that comes with the PocketSphinx distribution. This means that any commands spoken should be pronounced with a US-English accent in order to be accurately understood. To change this, one can load a different language model aimed at different languages and accents. Various models are available via the PocketSphinx website, and it is possible to make one's own language model with some effort.

# Usage scenarios

We don't want the infotainment system to be activated every single time that the voice user interface recognizes command words when they are not intended as such. The common way to prevent this from happening is by having a keyword that activates the command interface. If no command is recognized after the keyword within a certain amount of time, the system reverts to the keyword-spotting mode.

For this example project, we use the keyword `computer`. After the system spots this keyword, we can use the following commands:

| **Command** | **Result** |
| Play Bluetooth | Starts playing from any connected A2DP source device (unimplemented). |
| Stop Bluetooth | Stops playing from any Bluetooth device. |
| Play local | Plays the (hardcoded) local music file. |
| Stop local | Stops playing the local music file, if currently playing. |
| Play remote | Plays from an online streaming service or server (unimplemented). |
| Stop remote | Stops playing, if active. |
| Record message | Records a message. Records until a number of seconds of silence occurs. |
| Play message | Plays back the recorded message, if any. |

# Source code

This application has been implemented using the Qt framework, as a GUI application, so that we also get a graphical interface for ease of debugging. This debugging UI was designed using the Qt Designer of the Qt Creator IDE as a single UI file.

We start by creating an instance of the GUI application:

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

This creates an instance of the `MainWindow` class in which we have implemented the application, along with an instance of `QApplication`, which is a wrapper class used by the Qt framework.

Next, this is the `MainWindow` header:

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

Its implementation contains most of the core functionality, declaring the audio recorder and player instances, with just the voice command processing being handled in a separate class:

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

In the constructor, we set up all of the UI connections for the buttons in the GUI window that allow us to trigger the application's functionality without having to use the voice user interface. This is useful for testing purposes.

In addition, we create an instance of the audio recorder and media player, along with an audio probe that is linked with the audio recorder, so that we can look at the audio samples it's recording and act on them.

Finally, we create an instance of the voice input interface class and push it onto its own thread before starting it. We connect its signals to specific commands, and other events to their respective slots:

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

As mentioned in the section on Bluetooth technology, we have left the Bluetooth functionality unimplemented for the reasons explained in that section. 

```cpp
void MainWindow::playOnlineStream() { 
    // Connect to remote streaming service's API and start streaming. 
} 

void MainWindow::stopOnlineStream() { 
    // Stop streaming from remote service. 
} 
```

The same is true for the online streaming functionality. See the section on online streaming earlier in this chapter for details on how to implement this functionality.

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

To play a local file, we expect to find an MP3 file present in the hardcoded path. This could, however, also play all of the music in a specific folder with just a few modifications by reading in the filenames and playing them back one by one.

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

In the constructor, we configured the recorder to record to a file in a sub-folder called `message`. This will be overwritten if a new recording is made, allowing one to leave a message that can be played back later. The optional display or another accessory could be used to indicate when a new recording has been made and hasn't been listened to yet:

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

This method is called by our audio probe whenever the recorder is active. In this function, we calculate the **root-mean square** (**RMS**) value of the audio buffer to determine whether it's filled with silence. Here, silence is relative and might have to be adjusted depending on the recording environment.

After five seconds of silence have been detected, the recording of the message is stopped:

```cpp
void MainWindow::errorString(QString err) { 
    QMessageBox::critical(this, tr("Error"), err); 
} 

void MainWindow::quit() { 
    exit(0); 
} 
```

The remaining methods handle the reporting of error messages that may be emitted elsewhere in the application, as well as terminating the application.

The `VoiceInput` class header defines the functionality for the voice input interface:

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

As PocketSphinx is a C library, we have to make sure that it is used with C-style linkage. Beyond this, we create the class members for the audio input and related IO device that the voice input will use.

Next, the class definition:

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

The constructor doesn't do anything special, as the next method does all of the initializing and setting up of the main loop:

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

The first part of this method sets up the audio interface, configuring it to record using the audio format settings PocketSphinx requires: mono, little-endian, 16-bit signed PCM audio at 16,000 Hertz. After checking that the audio input supports this format, we create a new audio input instance:

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

Next, we set up the keyword-spotting and JSGF grammar file that will be used during the processing of the audio sample. With the first `ps_set_search()` function call, we start the keyword-spotting search. The following loop will keep processing samples until the utterance `computer` is detected:

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

Each cycle, we read in another buffer worth of audio samples, to then have PocketSphinx process these samples. It also does silence detection for us to determine whether someone has started speaking into the microphone. If someone is speaking but we haven't started interpreting it yet, we start a new utterance:

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

The rest of the method checks whether we have a usable hypothesis we can analyze. Depending on whether we are in keyword or grammar mode, we check for the detection of the keyword in the former case and switch to grammar mode. If we're already in grammar mode, we try to narrow the utterance down to a specific command, at which point we will emit the relevant signal that will trigger the connected functionality.

A new utterance is started whenever PocketSphinx detects at least one second of silence. After executing a command, the system switches back to keyword-spotting mode.

# Building the project

To build the project, the PocketSphinx project has to be built first. In the example project's source code that comes with this chapter, there are two Makefiles underneath the `sphinx` folder, one in the `pocketsphinx` folder and one in the `sphinxbase` folder. With these, the two libraries that form PocketSphinx will be built.

After this, one can build the Qt project, either from Qt Creator or from the command line, by executing the following command:

```cpp
mkdir build
cd build
qmake ..
make
```

# Extending the system

In addition to audio formats, one could also add the ability to play back videos and integrate the ability to make and respond to phone calls (using the Bluetooth API). One may want to extend the application to make it more flexible and modular, so that, for example, one could add a module that would add the voice commands and resulting actions.

Having voice output would be convenient as well, making it more aligned with the current commercial offerings. For this, one could use the text-to-speech API that's available in the Qt framework.

It would also be useful to add more *information* to the infotainment system by querying remote APIs for things such as the current weather, news updates, and maybe even running updates on a current football game. The voice-based UI could be used to set up timers and task reminders, integrate a calendar, and much more.

Finally, as can be seen in this chapter's example code, one cannot specify the name of the track that one wants to play, or a specific album or artist name. Allowing such freestyle input is incredibly useful, but comes with its own set of issues.

The main problem is the recognition rate of a voice-to-text system, especially for words it doesn't have in its dictionary. Some of us may already have had the pleasure of raising our voice in trying to make a voice-driven user interface on the phone, in the car, or on our smartphones understand a certain word.

At this point, it's still a big point of research, without a quick and easy solution. One could conceivably brute-force such recognition and get much better accuracy by using an index of local audio filenames and artists, along with other metadata, as part of the dictionary. The same could be done for a remote streaming service, through querying its API. This might add considerable latency to the recognition effort, however.

# Summary

In this chapter, we looked at how one can fairly easily construct an SBC-based infotainment system, using voice-to-text to construct a voice-driven user interface. We also looked at ways that we could extend it to add even more functionality.

The reader is expected to be able to implement a similar system at this point, and to extend it to connect it to online and network-based services. The reader should also read up on the implementation of more advanced voice-driven user interfaces, the addition of text-to-speech, and the use of A2DP-based Bluetooth devices.

In the next chapter, we'll be taking a look at how to implement a building-wide monitoring and control system using microcontrollers and the local network.