# Developing Embedded Systems with Qt

Qt (pronounced *cute*) is an advanced C++-based framework that covers a wide range of APIs, allowing you to implement networking, graphical user interfaces, parsing of data formats, the playing back and recording of audio, and much more. This chapter primarily covers the graphical aspect of Qt, and how to create advanced GUIs for embedded devices to provide an attractive and functional UI to users.

The topics covered in this chapter are as follows:

*   Creating advanced GUIs with Qt for embedded systems
*   Using Qt's 3D designer to create an infotainment UI
*   Extending an existing embedded system with a GUI

# The power of the right framework

A **framework** is essentially a collection of code aimed at easing the development of software for a specific application. It provides the developer with a range of classes—or the language equivalent—to allow you to implement the application logic without having to worry about interfacing with the underlying hardware, or using the OS's APIs.

In previous chapters, we used a number of frameworks to make our development efforts easier, from the No date Framework ([Chapter 4](bb67db6a-7c71-4519-80c3-7cd571cddfc0.xhtml), *Resource-Restricted Embedded Systems*) and CMSIS to Arduino for **microcontrollers** (**MCUs**), and from the low-level POCO framework for cross-platform development to the higher-level Qt framework.

Each of these frameworks has a specific type of system that they are intended for. For No date, CMSIS, and Arduino, the target is MCUs, ranging from 8-bit AVR MCUs to 32-bit ARM MCUs. These target the bare-metal systems, without any intermediate **operating system** (**OS**) or similar. Above those in terms of complexity, we find **real-time OS frameworks** (**RTOS**), which include a full OS in the framework.

Frameworks such as POCO and Qt target OSes in general, from desktop and server platforms to SoC platforms. Here they function primarily as an abstraction layer between the OS-specific APIs, while providing additional functionality alongside this abstraction. This allows you to quickly build up a full-featured application, without having to spend much time on each feature.

This is particularly important for networking functionality, where you do not want to write a TCP sockets-based server from scratch, but ideally just want to instantiate a ready-made class and use it. In the case of Qt, it also provides graphical user interface-related APIs to make the development of cross-platform GUIs easier. Other frameworks that also provide this kind of functionality include GTK+ and WxWidgets. In this chapter, however, we'll just be looking at developing with Qt.

In [Chapter 8](4416b2de-d86a-4001-863d-b167635a0e10.xhtml), *Example - Linux-Based Infotainment System*, we got a good look at how to develop with the Qt framework. There, we mostly ignored the **graphical user interface** (**GUI**) part, even though this is probably the most interesting part of Qt relative to other OS-based frameworks. Being able to use the same GUI across multiple OSes can be incredibly useful and convenient.

This is mostly the case for desktop-based applications, where the GUI is a crucial part of the application, and thus not having to spend the time and trouble of porting it between OSes is a major time saver. For embedded platforms, this is also true, though here you have the option of integrating even deeper into the system than on a desktop system, as we will see in a moment.

We'll also look at the various types of Qt applications that you can develop, starting with a simple **command-line interface** (**CLI**) application.

# Qt for command-line use

Even though the graphical user interface is a big selling point of the Qt framework, it is also possible to use it to develop command-line-only applications. For this, we just use the `QCoreApplication` class to create an input and an event loop handler, as in this example:

```cpp
#include <QCoreApplication> 
#include <core.h> 

int main(int argc, char *argv[]) { 
   QCoreApplication app(argc, argv); 
   Core core; 

   connect(&core, &Core::done, &app, &app::quit, Qt::QueuedConnection); 
   core.start(); 

   return app.exec(); 
} 
```

Here, our code is implemented in a class called `Core`. In the main function, we create a `QCoreApplication` instance, which receives the command-line parameters. We then instantiate an instance of our class.

We connect a signal from our class to the `QCoreApplication` instance, so that if we signal that we have finished, it will trigger a slot on the latter to clean up and terminate the application.

After this, we call the method on our class to start its functionality and finally start the event loop by calling `exec()` on the `QCoreApplication` instance. At this point, we can use signals.

Note here that it is also possible to use Qt4-style connection syntax, instead of the earlier Qt5-style:

```cpp
connect(core, SIGNAL(done()), &app, SLOT(quit()), Qt::QueuedConnection); 
```

Functionally, this makes no difference, and using either is fine for most situations.

Our class appears as follows:

```cpp
#include <QObject> 

class Core : public QObject { 
   Q_OBJECT 
public: 
   explicit Core(QObject *parent = 0); 

signals: 
   void done(); 
public slots: 
   void start(); 
}; 
```

Every class in a Qt-based application that wants to make use of the signal-slot architecture of Qt is required to derive from the `QObject` class, and to include the `Q_OBJECT` macro within the class declaration. This is needed for Qt's `qmake preprocessor` tool to know which classes to process before the application code is compiled by the toolchain.

Here is the implementation:

```cpp
#include "core.h" 
#include <iostream> 

Core::Core(QObject *parent) : QObject(parent) { 
   // 
} 

void hang::start() { 
   std::cout << "Start emitting done()" << std::endl; 
   emit done(); 
} 
```

Of note is the fact that we can let the constructor of any QObject-derived class know what the encapsulating parent class is, allowing said parent to own these child classes and invoke their destructor when it itself is destroyed.

# GUI-based Qt applications

Returning to the Qt-based example project from [Chapter 8](4416b2de-d86a-4001-863d-b167635a0e10.xhtml), *Example - Linux-Based Infotainment System*, we can now compare its main function to the preceding command-line-only version to see what changes once we add a GUI to the project:

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

The most obvious change here is that we use `QApplication` instead of `QCoreApplication`. The other big change is that we do not use a completely custom class, but one that derives from `QMainWindow`:

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

    qint64 silence; 

private slots: 
    void processBuffer(QAudioBuffer); 
}; 
```

Here, we can see that the `MainWindow` class indeed derives from `QMainWindow`, which also gives it its `show()` method. Of note is the `MainWindow` instance being declared in the UI namespace. This is connected to the auto-generated code that is created when we run the qmake tool on the UI file, as we will see in a moment. Next is the constructor:

```cpp
MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent), 
    ui(new Ui::MainWindow) { 
    ui->setupUi(this); 
```

The first thing of note here is how we inflate the GUI from the UI description file. This file is usually created by visually laying out the GUI with the Qt Designer tool, which is part of the Qt Creator IDE. This UI file contains a description of each widget's properties, along with the layout applied to them, and so on.

It's also possible to programmatically create these widgets and add them to layouts, of course. This gets quite tedious for larger layouts, however. Generally, you create a single UI file for the main window, and an additional one for each sub window and dialog. These can then be inflated into a window or dialog in a similar fashion:

```cpp
    connect(ui->actionQuit, SIGNAL(triggered()), this, SLOT(quit())); 
```

Menu actions in the GUI are connected to internal slots by specifying the specific signal on the menu action (`QAction` instance). We can see here that they are in the `ui` object, which is found in the auto-generated source code for the UI file, as we mentioned earlier:

```cpp
    connect(ui->playBluetoothButton, SIGNAL(pressed), this, SLOT(playBluetooth)); 
    connect(ui->stopBluetoothButton, SIGNAL(pressed), this, SLOT(stopBluetooth)); 
    connect(ui->playLocalAudioButton, SIGNAL(pressed), this, SLOT(playLocalFile)); 
    connect(ui->stopLocalAudioButton, SIGNAL(pressed), this, SLOT(stopLocalFile)); 
    connect(ui->playOnlineStreamButton, SIGNAL(pressed), this, SLOT(playOnlineStream)); 
    connect(ui->stopOnlineStreamButton, SIGNAL(pressed), this, SLOT(stopOnlineStream)); 
    connect(ui->recordMessageButton, SIGNAL(pressed), this, SLOT(recordMessage)); 
    connect(ui->playBackMessage, SIGNAL(pressed), this, SLOT(playMessage)); 
```

Button widgets in the GUI are connected in a similar manner, though they of course emit a different signal on account of them being a different type of widget:

```cpp
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
    audioProbe→setSource(audioRecorder); 
```

We're free to do anything we would do in any other constructor, including setting defaults and creating instances of classes we will need later on:

```cpp
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

One crucial thing to remember here is that this class runs on the UI thread, meaning that we should not do anything intensive in here. That's why we move such class instances off to their own thread, as shown here:

```cpp
MainWindow::~MainWindow() { 
    delete ui; 
} 
```

In the constructor, we delete the UI and all associated elements.

# Embedded Qt

A major target of the Qt framework next to desktop systems are **embedded systems**, specifically **Embedded Linux**, where there are a few different ways to use Q. The main point of embedded Qt is to optimize the software stock by allowing you to boot straight into a Qt-optimized environment, and by allowing for a variety of ways to render to the display.

Qt for Embedded Linux supports the following platform plugins for rendering:

| **Plugin** | **Description** |
| EGLFS | Provides an interface to OpenGL ES or similar 3D rendering API. Usually, the default configuration for Embedded Linux. More details about EGL can be found at the following address: [https://www.khronos.org/egl](https://www.khronos.org/egl). |
| LinuxFB | Writes directly to the framebuffer via Linux's fbdev subsystem. Only software-rendered content is supported. As a result, on some setups the display performance is likely to be limited. |
| DirectFB | Directly writes to the graphic card's framebuffer using the DirectFB library. |
| Wayland | Uses the Wayland windowing system. This allows for multiple concurrent windows, but is of course more demanding on the hardware. |

In addition to this, Qt for Embedded Linux comes with a variety of APIs for handling touch and pen input, and so on. To optimize the system for a Qt-based application, any unrelated services, processes, and libraries are generally removed, resulting in a system that boots within a matter of seconds into the embedded application.

# Custom GUIs with stylesheets

The standard widget-based GUIs that desktop systems tend to use do not lend themselves that readily to customization. As a result, you are generally faced with having to either override the painting function in a `QWidget` instance and handle every single pixel of the widget drawing, or to use stylesheet-based customization.

Qt stylesheets allow you to tweak the look and feel of individual widgets, even dynamically. They are essentially written using **Cascading Style Sheet** (**CSS**) syntax as used with HTML pages. They allow you to change elements of a widget, such as the borders, rounding corners, or the thickness and color of the elements.

# QML

**Qt Modeling Language** (**QML**) is a user interface markup language. It is strongly based on JavaScript and even uses inline JavaScript. It can be used to create dynamic and completely custom user interfaces, and is usually used together with the Qt Quick module.

Later in this chapter, we will take an in-depth look at how a dynamic GUI is created.

# 3D designer

With Qt 5, the Qt 3D module was introduced, which streamlined access to the OpenGL rendering API. This new module was used as the foundation for the Qt 3D Designer editor and the accompanying runtime. It can be used to create highly dynamic GUIs, featuring a combination of 2D and 3D elements.

It is quite similar to hand-crafted QML-based GUIs, but provides a more streamlined workflow, ease of adding animations, and previewing the project. It's similar to the Qt Designer Studio, which focuses more on 2D GUIs, but this one is not available for free, instead requiring you to purchase a license.

# An example of adding a GUI to the infotainment system

In this example, we will be using C++, Qt, and QML to create a graphical user interface that is capable of showing the current track that is playing, performing an audio visualization, indicating the playback progress, and allowing you to toggle different input modes using onscreen buttons.

This example is based on the *Audio Visualizer* example from the Qt documentation. This can be found in the Qt installation folder (if examples got installed), as well as on the Qt site: [https://doc.qt.io/qt-5/qt3d-audio-visualizer-qml-example.html.](https://doc.qt.io/qt-5/qt3d-audio-visualizer-qml-example.html.)

The main difference between this code and the official example is that the `QMediaPlayer` media player was moved into the C++ code, along with a number of other functions. Instead, a number of signals and slots between the QML UI and C++ backend are used in the new `QmlInterface` class for button presses, updating the UI, and interaction with the media player.

A GUI such as this could be wired into the existing infotainment project code to control its functionality, using the GUI in addition to the voice-driven interface.

The GUI we're putting together in this example looks like this in action:

![](img/abcd27b7-3f26-455b-8fa1-3e70263869cf.png)

# Main

The main source file appears as follows:

```cpp
#include "interface.h" 
#include <QtGui/QGuiApplication> 
#include <QtGui/QOpenGLContext> 
#include <QtQuick/QQuickView> 
#include <QtQuick/QQuickItem> 
#include <QtQml/QQmlContext> 
#include <QObject> 

int main(int argc, char* argv[]) { 
    QGuiApplication app(argc, argv); 

    QSurfaceFormat format; 
    if (QOpenGLContext::openGLModuleType() == QOpenGLContext::LibGL) { 
        format.setVersion(3, 2); 
        format.setProfile(QSurfaceFormat::CoreProfile); 
    } 

    format.setDepthBufferSize(24); 
    format.setStencilBufferSize(8); 

    QQuickView view; 
    view.setFormat(format); 
    view.create(); 

    QmlInterface qmlinterface; 
    view.rootContext()->setContextProperty("qmlinterface", &qmlinterface); 
    view.setSource(QUrl("qrc:/main.qml")); 

    qmlinterface.setPlaying(); 

    view.setResizeMode(QQuickView::SizeRootObjectToView); 
    view.setMaximumSize(QSize(1820, 1080)); 
    view.setMinimumSize(QSize(300, 150)); 
    view.show(); 

    return app.exec(); 
} 
```

Our custom class is added to the QML viewer (`QQuickView`) as a context class. This serves as the proxy between the QML UI and our C++ code, as we will see in a moment. The viewer itself uses an OpenGL surface to render the UI on.

# QmlInterface

The header of our custom class features a number of additions to make properties and methods visible to the QML code:

```cpp
#include <QtCore/QObject> 
#include <QMediaPlayer> 
#include <QByteArray> 

class QmlInterface : public QObject { 
    Q_OBJECT     
    Q_PROPERTY(QString durationTotal READ getDurationTotal NOTIFY durationTotalChanged) 
    Q_PROPERTY(QString durationLeft READ getDurationLeft NOTIFY durationLeftChanged) 

```

The `Q_PROPERTY` tag tells the qmake parser that this class contains a property (variable) that should be made visible to the QML code, with the parameters specifying the name of the variable, the methods used for reading and writing the variable (if desired), and finally the signal that is emitted whenever the property has changed.

This allows for an automatic update feature to be set up to keep this property synchronized between the C++ code and the QML side:

```cpp

    QString formatDuration(qint64 milliseconds); 

    QMediaPlayer mediaPlayer; 
    QByteArray magnitudeArray; 
    const int millisecondsPerBar = 68; 
    QString durationTotal; 
    QString durationLeft; 
    qint64 trackDuration; 

public: 
    explicit QmlInterface(QObject *parent = nullptr); 

    Q_INVOKABLE bool isHoverEnabled() const; 
    Q_INVOKABLE void setPlaying(); 
   Q_INVOKABLE void setStopped(); 
   Q_INVOKABLE void setPaused(); 
    Q_INVOKABLE qint64 duration(); 
    Q_INVOKABLE qint64 position(); 
    Q_INVOKABLE double getNextAudioLevel(int offsetMs); 

    QString getDurationTotal() { return durationTotal; } 
    QString getDurationLeft() { return durationLeft; } 

public slots: 
    void mediaStatusChanged(QMediaPlayer::MediaStatus status); 
    void durationChanged(qint64 duration); 
    void positionChanged(qint64 position); 

signals: 
    void start(); 
    void stopped(); 
    void paused(); 
    void playing(); 
    void durationTotalChanged(); 
    void durationLeftChanged(); 
}; 
```

Similarly, the `Q_INVOKABLE` tag ensures that these methods are made visible to the QML side and can be called from there.

Here is the implementation:

```cpp
#include "interface.h" 
#include <QtGui/QTouchDevice> 
#include <QDebug> 
#include <QFile> 
#include <QtMath> 

QmlInterface::QmlInterface(QObject *parent) : QObject(parent) { 
    // Set track for media player. 
    mediaPlayer.setMedia(QUrl("qrc:/music/tiltshifted_lost_neon_sun.mp3")); 

    // Load magnitude file for the audio track. 
    QFile magFile(":/music/visualization.raw", this); 
    magFile.open(QFile::ReadOnly); 
    magnitudeArray = magFile.readAll(); 

    // Media player connections. 
    connect(&mediaPlayer, SIGNAL(mediaStatusChanged(QMediaPlayer::MediaStatus)), this, SLOT(mediaStatusChanged(QMediaPlayer::MediaStatus))); 
    connect(&mediaPlayer, SIGNAL(durationChanged(qint64)), this, SLOT(durationChanged(qint64))); 
    connect(&mediaPlayer, SIGNAL(positionChanged(qint64)), this, SLOT(positionChanged(qint64))); 
} 
```

The constructor got changed considerably from the original example project, with the media player instance being created here, along with its connections.

We load the same music file here as was used with the original project. When integrating the code into the infotainment project or a similar project, you would make this dynamic. Similarly, the file that we are also loading here to get the amplitude for the music file with the visualization would likely be omitted in a full integration, instead opting to generate the amplitude values dynamically:

```cpp
bool QmlInterface::isHoverEnabled() const { 
#if defined(Q_OS_IOS) || defined(Q_OS_ANDROID) || defined(Q_OS_QNX) || defined(Q_OS_WINRT) 
    return false; 
#else 
    bool isTouch = false; 
    foreach (const QTouchDevice *dev, QTouchDevice::devices()) { 
        if (dev->type() == QTouchDevice::TouchScreen) { 
            isTouch = true; 
            break; 
        } 
    } 

    bool isMobile = false; 
    if (qEnvironmentVariableIsSet("QT_QUICK_CONTROLS_MOBILE")) { 
        isMobile = true; 
    } 

    return !isTouch && !isMobile; 
#endif 
} 
```

This was the only method that previously existed in the QML context class. It's used to detect whether the code runs on a mobile device with a touch screen:

```cpp
void QmlInterface::setPlaying() { 
   mediaPlayer.play(); 
} 

void QmlInterface::setStopped() { 
   mediaPlayer.stop(); 
} 

void QmlInterface::setPaused() { 
   mediaPlayer.pause(); 
} 
```

We got a number of control methods that connect to the buttons in the UI to allow for control of the media player instance:

```cpp
void QmlInterface::mediaStatusChanged(QMediaPlayer::MediaStatus status) { 
    if (status == QMediaPlayer::EndOfMedia) { 
        emit stopped(); 
    } 
} 
```

This slot method is used to detect when the media player has reached the end of the active track, so that the UI can be signaled that it should update to indicate this:

```cpp
void QmlInterface::durationChanged(qint64 duration) { 
    qDebug() << "Duration changed: " << duration; 

    durationTotal = formatDuration(duration); 
    durationLeft = "-" + durationTotal; 
    trackDuration = duration; 
    emit start(); 
    emit durationTotalChanged(); 
    emit durationLeftChanged(); 
} 

void QmlInterface::positionChanged(qint64 position) { 
    qDebug() << "Position changed: " << position; 
    durationLeft = "-" + formatDuration((trackDuration - position)); 
    emit durationLeftChanged(); 
} 
```

These two slot methods are connected to the media player instance. The duration slot is required because the length (duration) of a newly loaded track will not be immediately available. Instead, it's an asynchronously updated property.

As a result, we have to wait until the media player has finished with this and emits the signal that it has completed this process.

Next, to allow us to update the time remaining on the current track, we also get constant updates on the current position from the media player so that we can update the UI with the new value.

Both the duration and position properties are updated in the UI using the linkage method we saw in the description of the header file for this class.

Finally, we emit a `start()` signal, which is linked into a slot in the QML code that will start the visualization process, as we will see later on in the QML code:

```cpp
qint64 QmlInterface::duration() { 
    qDebug() << "Returning duration value: " << mediaPlayer.duration(); 
    return mediaPlayer.duration(); 
} 

qint64 QmlInterface::position() { 
    qDebug() << "Returning position value: " << mediaPlayer.position(); 
    return mediaPlayer.position(); 
} 
```

The duration property is also used by the visualization code. Here, we allow it to be obtained directly. Similarly, we make the position property available as well with a direct call:

```cpp
double QmlInterface::getNextAudioLevel(int offsetMs) { 
    // Calculate the integer index position in to the magnitude array 
    qint64 index = ((mediaPlayer.position() + offsetMs) / millisecondsPerBar) | 0; 

    if (index < 0 || index >= (magnitudeArray.length() / 2)) { 
        return 0.0; 
    } 

    return (((quint16*) magnitudeArray.data())[index] / 63274.0); 
} 
```

This method was ported from the JavaScript code in the original project, performing the same task of determining the audio level based on the amplitude data we read in previously from the file:

```cpp
QString QmlInterface::formatDuration(qint64 milliseconds) { 
    qint64 minutes = floor(milliseconds / 60000); 
    milliseconds -= minutes * 60000; 
    qint64 seconds = milliseconds / 1000; 
    seconds = round(seconds); 
    if (seconds < 10) { 
        return QString::number(minutes) + ":0" + QString::number(seconds); 
    } 
    else { 
        return QString::number(minutes) + ":" + QString::number(seconds); 
    } 
} 
```

Similarly, this method was also ported from the original project's JavaScript code, since we moved the code that relies on it into the C++ code. It takes in the millisecond count for the track duration or position and converts it into a string containing the minutes and seconds, matching the original value.

# QML

Moving on, we are done with the C++ side of things and can now look at the QML UI.

First, here is the main QML file:

```cpp
import QtQuick 2.0 
import QtQuick.Scene3D 2.0 
import QtQuick.Layouts 1.2 
import QtMultimedia 5.0 

Item { 
    id: mainview 
    width: 1215 
    height: 720 
    visible: true 
    property bool isHoverEnabled: false 
    property int mediaLatencyOffset: 68 
```

The QML file consists out of a hierarchy of elements. Here, we define the top element, giving it its dimensions and name:

```cpp
    state: "stopped" 
    states: [ 
        State { 
            name: "playing" 
            PropertyChanges { 
                target: playButtonImage 
                source: { 
                    if (playButtonMouseArea.containsMouse) 
                        "qrc:/images/pausehoverpressed.png" 
                    else 
                        "qrc:/images/pausenormal.png" 
                } 
            } 
            PropertyChanges { 
                target: stopButtonImage 
                source: "qrc:/images/stopnormal.png" 
            } 
        }, 
        State { 
            name: "paused" 
            PropertyChanges { 
                target: playButtonImage 
                source: { 
                    if (playButtonMouseArea.containsMouse) 
                        "qrc:/images/playhoverpressed.png" 
                    else 
                        "qrc:/images/playnormal.png" 
                } 
            } 
            PropertyChanges { 
                target: stopButtonImage 
                source: "qrc:/images/stopnormal.png" 
            } 
        }, 
        State { 
            name: "stopped" 
            PropertyChanges { 
                target: playButtonImage 
                source: "qrc:/images/playnormal.png" 
            } 
            PropertyChanges { 
                target: stopButtonImage 
                source: "qrc:/images/stopdisabled.png" 
            } 
        } 
    ]    
```

A number of states for the UI are defined, along with the changes that should be triggered if the state should change to it:

```cpp
    Connections { 
        target: qmlinterface 
        onStopped: mainview.state = "stopped" 
        onPaused: mainview.state = "paused" 
        onPlaying: mainview.state = "started" 
        onStart: visualizer.startVisualization() 
    } 
```

These are the connections that link the signals from the C++ side to a local handler. We target our custom class as the source of these signals, then define the handler for each signal we wish to handle by prefixing it and adding the code that should be executed.

Here, we see that the start signal is linked to a handler that triggers the function in the visualization module that starts that module:

```cpp
    Component.onCompleted: isHoverEnabled = qmlinterface.isHoverEnabled() 

    Image { 
        id: coverImage 
        anchors.fill: parent 
        source: "qrc:/images/albumcover.png" 
    } 
```

This `Image` element defines the background image, which we load from the resources that were added to the executable when the project was built:

```cpp
    Scene3D { 
        anchors.fill: parent 

        Visualizer { 
            id: visualizer 
            animationState: mainview.state 
            numberOfBars: 120 
            barRotationTimeMs: 8160 // 68 ms per bar 
        } 
    } 
```

The 3D scene that will be filled with the visualizer's content is defined:

```cpp
    Rectangle { 
        id: blackBottomRect 
        color: "black" 
        width: parent.width 
        height: 0.14 * mainview.height 
        anchors.bottom: parent.bottom 
    } 

    Text { 
        text: qmlinterface.durationTotal 
        color: "#80C342" 
        x: parent.width / 6 
        y: mainview.height - mainview.height / 8 
        font.pixelSize: 12 
    } 

    Text { 
        text: qmlinterface.durationLeft 
        color: "#80C342" 
        x: parent.width - parent.width / 6 
        y: mainview.height - mainview.height / 8 
        font.pixelSize: 12 
    } 
```

These two text elements are linked with the property in our custom C++ class, as we saw earlier. These values will be kept updated with the value in the C++ class instance as it changes:

```cpp
    property int buttonHorizontalMargin: 10 
    Rectangle { 
        id: playButton 
        height: 54 
        width: 54 
        anchors.bottom: parent.bottom 
        anchors.bottomMargin: width 
        x: parent.width / 2 - width - buttonHorizontalMargin 
        color: "transparent" 

        Image { 
            id: playButtonImage 
            source: "qrc:/images/pausenormal.png" 
        } 

        MouseArea { 
            id: playButtonMouseArea 
            anchors.fill: parent 
            hoverEnabled: isHoverEnabled 
            onClicked: { 
                if (mainview.state == 'paused' || mainview.state == 'stopped') 
                    mainview.state = 'playing' 
                else 
                    mainview.state = 'paused' 
            } 
            onEntered: { 
                if (mainview.state == 'playing') 
                    playButtonImage.source = "qrc:/images/pausehoverpressed.png" 
                else 
                    playButtonImage.source = "qrc:/images/playhoverpressed.png" 
            } 
            onExited: { 
                if (mainview.state == 'playing') 
                    playButtonImage.source = "qrc:/images/pausenormal.png" 
                else 
                    playButtonImage.source = "qrc:/images/playnormal.png" 
            } 
        } 
    } 

    Rectangle { 
        id: stopButton 
        height: 54 
        width: 54 
        anchors.bottom: parent.bottom 
        anchors.bottomMargin: width 
        x: parent.width / 2 + buttonHorizontalMargin 
        color: "transparent" 

        Image { 
            id: stopButtonImage 
            source: "qrc:/images/stopnormal.png" 
        } 

        MouseArea { 
            anchors.fill: parent 
            hoverEnabled: isHoverEnabled 
            onClicked: mainview.state = 'stopped' 
            onEntered: { 
                if (mainview.state != 'stopped') 
                    stopButtonImage.source = "qrc:/images/stophoverpressed.png" 
            } 
            onExited: { 
                if (mainview.state != 'stopped') 
                    stopButtonImage.source = "qrc:/images/stopnormal.png" 
            } 
        } 
    } 
} 
```

The rest of the source serves to set up the individual buttons for controlling the playback, with play, stop, and pause buttons, which get swapped over as needed.

Next, we will look at the amplitude bar file:

```cpp
import Qt3D.Core 2.0 
import Qt3D.Render 2.0 
import Qt3D.Extras 2.0 
import QtQuick 2.4 as QQ2 

Entity { 
    property int rotationTimeMs: 0 
    property int entityIndex: 0 
    property int entityCount: 0 
    property int startAngle: 0 + 360 / entityCount * entityIndex 
    property bool needsNewMagnitude: true 
    property real magnitude: 0 
    property real animWeight: 0 

    property color lowColor: "black" 
    property color highColor: "#b3b3b3" 
    property color barColor: lowColor 

    property string entityAnimationsState: "stopped" 
    property bool entityAnimationsPlaying: true 

    property var entityMesh: null 
```

A number of properties are defined before we dive into the animation state change handler:

```cpp
    onEntityAnimationsStateChanged: { 
        if (animationState == "paused") { 
            if (angleAnimation.running) 
                angleAnimation.pause() 
            if (barColorAnimations.running) 
                barColorAnimations.pause() 
        } else if (animationState == "playing"){ 
            needsNewMagnitude = true; 
            if (heightDecreaseAnimation.running) 
                heightDecreaseAnimation.stop() 
            if (angleAnimation.paused) { 
                angleAnimation.resume() 
            } else if (!entityAnimationsPlaying) { 
                magnitude = 0 
                angleAnimation.start() 
                entityAnimationsPlaying = true 
            } 
            if (barColorAnimations.paused) 
                barColorAnimations.resume() 
        } else { 
            if (animWeight != 0) 
                heightDecreaseAnimation.start() 
            needsNewMagnitude = true 
            angleAnimation.stop() 
            barColorAnimations.stop() 
            entityAnimationsPlaying = false 
        } 
    } 
```

Every time the audio playback is stopped, paused, or started, the animation has to be updated to match this state change:

```cpp
    property Material barMaterial: PhongMaterial { 
        diffuse: barColor 
        ambient: Qt.darker(barColor) 
        specular: "black" 
        shininess: 1 
    } 
```

This defines the look of the amplitude bars, using Phong shading:

```cpp
    property Transform angleTransform: Transform { 
        property real heightIncrease: magnitude * animWeight 
        property real barAngle: startAngle 

        matrix: { 
            var m = Qt.matrix4x4() 
            m.rotate(barAngle, Qt.vector3d(0, 1, 0)) 
            m.translate(Qt.vector3d(1.1, heightIncrease / 2 - heightIncrease * 0.05, 0)) 
            m.scale(Qt.vector3d(0.5, heightIncrease * 15, 0.5)) 
            return m; 
        } 

        property real compareAngle: barAngle 
        onBarAngleChanged: { 
            compareAngle = barAngle 

            if (compareAngle > 360) 
                compareAngle = barAngle - 360 

            if (compareAngle > 180) { 
                parent.enabled = false 
                animWeight = 0 
                if (needsNewMagnitude) { 
                    // Calculate the ms offset where the bar will be at the center point of the 
                    // visualization and fetch the correct magnitude for that point in time. 
                    var offset = (90.0 + 360.0 - compareAngle) * (rotationTimeMs / 360.0) 
                    magnitude = qmlinterface.getNextAudioLevel(offset) 
                    needsNewMagnitude = false 
                } 
            } else { 
                parent.enabled = true 
                // Calculate a power of 2 curve for the bar animation that peaks at 90 degrees 
                animWeight = Math.min((compareAngle / 90), (180 - compareAngle) / 90) 
                animWeight = animWeight * animWeight 
                if (!needsNewMagnitude) { 
                    needsNewMagnitude = true 
                    barColorAnimations.start() 
                } 
            } 
        } 
    } 
```

As the amplitude bars move across the screen, they change relative to the camera, so we need to keep calculating the new angle and display height.

In this section, we also replaced the original call to the audio level method with a call to the new method in our C++ class:

```cpp
    components: [entityMesh, barMaterial, angleTransform] 

    QQ2.NumberAnimation { 
        id: angleAnimation 
        target: angleTransform 
        property: "barAngle" 
        duration: rotationTimeMs 
        loops: QQ2.Animation.Infinite 
        running: true 
        from: startAngle 
        to: 360 + startAngle 
    } 

    QQ2.NumberAnimation { 
        id: heightDecreaseAnimation 
        target: angleTransform 
        property: "heightIncrease" 
        duration: 400 
        running: false 
        from: angleTransform.heightIncrease 
        to: 0 
        onStopped: barColor = lowColor 
    } 

    property int animationDuration: angleAnimation.duration / 6 

    QQ2.SequentialAnimation on barColor { 
        id: barColorAnimations 
        running: false 

        QQ2.ColorAnimation { 
            from: lowColor 
            to: highColor 
            duration: animationDuration 
        } 

        QQ2.PauseAnimation { 
            duration: animationDuration 
        } 

        QQ2.ColorAnimation { 
            from: highColor 
            to: lowColor 
            duration: animationDuration 
        } 
    } 
} 
```

The rest of the file contains a few more animation transforms.

Finally, here is the visualization module:

```cpp
import Qt3D.Core 2.0 
import Qt3D.Render 2.0 
import Qt3D.Extras 2.0 
import QtQuick 2.2 as QQ2 

Entity { 
    id: sceneRoot 
    property int barRotationTimeMs: 1 
    property int numberOfBars: 1 
    property string animationState: "stopped" 
    property real titleStartAngle: 95 
    property real titleStopAngle: 5 

    onAnimationStateChanged: { 
        if (animationState == "playing") { 
            qmlinterface.setPlaying(); 
            if (progressTransformAnimation.paused) 
                progressTransformAnimation.resume() 
            else 
                progressTransformAnimation.start() 
        } else if (animationState == "paused") { 
            qmlinterface.setPaused(); 
            if (progressTransformAnimation.running) 
                progressTransformAnimation.pause() 
        } else { 
            qmlinterface.setStopped(); 
            progressTransformAnimation.stop() 
            progressTransform.progressAngle = progressTransform.defaultStartAngle 
        } 
    } 
```

This section got changed from interacting with the local media player instance to the new one in the C++ code. Beyond that, we left it unchanged. This is the main handler for when anything changes in the scene due to user interaction, or a track starting or ending:

```cpp
    QQ2.Item { 
        id: stateItem 

        state: animationState 
        states: [ 
            QQ2.State { 
                name: "playing" 
                QQ2.PropertyChanges { 
                    target: titlePrism 
                    titleAngle: titleStopAngle 
                } 
            }, 
            QQ2.State { 
                name: "paused" 
                QQ2.PropertyChanges { 
                    target: titlePrism 
                    titleAngle: titleStopAngle 
                } 
            }, 
            QQ2.State { 
                name: "stopped" 
                QQ2.PropertyChanges { 
                    target: titlePrism 
                    titleAngle: titleStartAngle 
                } 
            } 
        ] 

        transitions: QQ2.Transition { 
            QQ2.NumberAnimation { 
                property: "titleAngle" 
                duration: 2000 
                running: false 
            } 
        } 
    } 
```

A number of property changes and transitions are defined for the track title object:

```cpp
    function startVisualization() { 
        progressTransformAnimation.duration = qmlinterface.duration() 
        mainview.state = "playing" 
        progressTransformAnimation.start() 
    } 
```

This function is what starts the entire visualization sequence. It uses the track duration, as obtained via our C++ class instance, to determine the dimensions of the progress bar for the track playback animation before starting the visualization animation:

```cpp
    Camera { 
        id: camera 
        projectionType: CameraLens.PerspectiveProjection 
        fieldOfView: 45 
        aspectRatio: 1820 / 1080 
        nearPlane: 0.1 
        farPlane: 1000.0 
        position: Qt.vector3d(0.014, 0.956, 2.178) 
        upVector: Qt.vector3d(0.0, 1.0, 0.0) 
        viewCenter: Qt.vector3d(0.0, 0.7, 0.0) 
    } 
```

A camera is defined for the 3D scene:

```cpp
    Entity { 
        components: [ 
            DirectionalLight { 
                intensity: 0.9 
                worldDirection: Qt.vector3d(0, 0.6, -1) 
            } 
        ] 
    } 

    RenderSettings { 
        id: external_forward_renderer 
        activeFrameGraph: ForwardRenderer { 
            camera: camera 
            clearColor: "transparent" 
        } 
    } 
```

A renderer and light for the scene are created:

```cpp
    components: [external_forward_renderer] 

    CuboidMesh { 
        id: barMesh 
        xExtent: 0.1 
        yExtent: 0.1 
        zExtent: 0.1 
    } 
```

A mesh is created for the amplitude bars:

```cpp
    NodeInstantiator { 
        id: collection 
        property int maxCount: parent.numberOfBars 
        model: maxCount 

        delegate: BarEntity { 
            id: cubicEntity 
            entityMesh: barMesh 
            rotationTimeMs: sceneRoot.barRotationTimeMs 
            entityIndex: index 
            entityCount: sceneRoot.numberOfBars 
            entityAnimationsState: animationState 
            magnitude: 0 
        } 
    } 
```

The number of bars, along with other properties, is defined:

```cpp
    Entity { 
        id: titlePrism 
        property real titleAngle: titleStartAngle 

        Entity { 
            id: titlePlane 

            PlaneMesh { 
                id: titlePlaneMesh 
                width: 550 
                height: 100 
            } 

            Transform { 
                id: titlePlaneTransform 
                scale: 0.003 
                translation: Qt.vector3d(0, 0.11, 0) 
            } 

            NormalDiffuseMapAlphaMaterial { 
                id: titlePlaneMaterial 
                diffuse: TextureLoader { source: "qrc:/images/demotitle.png" } 
                normal: TextureLoader { source: "qrc:/images/normalmap.png" } 
                shininess: 1.0 
            } 

            components: [titlePlaneMesh, titlePlaneMaterial, titlePlaneTransform] 
        } 
```

This plane contains the title object whenever there's no track playing:

```cpp
        Entity { 
            id: songTitlePlane 

            PlaneMesh { 
                id: songPlaneMesh 
                width: 550 
                height: 100 
            } 

            Transform { 
                id: songPlaneTransform 
                scale: 0.003 
                rotationX: 90 
                translation: Qt.vector3d(0, -0.03, 0.13) 
            } 

            property Material songPlaneMaterial: NormalDiffuseMapAlphaMaterial { 
                diffuse: TextureLoader { source: "qrc:/images/songtitle.png" } 
                normal: TextureLoader { source: "qrc:/images/normalmap.png" } 
                shininess: 1.0 
            } 

            components: [songPlaneMesh, songPlaneMaterial, songPlaneTransform] 
        } 
```

This plane contains the song title whenever a track is active:

```cpp
        property Transform titlePrismPlaneTransform: Transform { 
            matrix: { 
                var m = Qt.matrix4x4() 
                m.translate(Qt.vector3d(-0.5, 1.3, -0.4)) 
                m.rotate(titlePrism.titleAngle, Qt.vector3d(1, 0, 0)) 
                return m; 
            } 
        } 

        components: [titlePlane, songTitlePlane, titlePrismPlaneTransform] 
    } 
```

To transform the planes between playing and non-playing transitions, this transform is used:

```cpp
    Mesh { 
        id: circleMesh 
        source: "qrc:/meshes/circle.obj" 
    } 

    Entity { 
        id: circleEntity 
        property Material circleMaterial: PhongAlphaMaterial { 
            alpha: 0.4 
            ambient: "black" 
            diffuse: "black" 
            specular: "black" 
            shininess: 10000 
        } 

        components: [circleMesh, circleMaterial] 
    } 
```

A circle mesh that provides a reflection effect is added:

```cpp
    Mesh { 
        id: progressMesh 
        source: "qrc:/meshes/progressbar.obj" 
    } 

    Transform { 
        id: progressTransform 
        property real defaultStartAngle: -90 
        property real progressAngle: defaultStartAngle 
        rotationY: progressAngle 
    } 

    Entity { 
        property Material progressMaterial: PhongMaterial { 
            ambient: "purple" 
            diffuse: "white" 
        } 

        components: [progressMesh, progressMaterial, progressTransform] 
    } 

    QQ2.NumberAnimation { 
        id: progressTransformAnimation 
        target: progressTransform 
        property: "progressAngle" 
        duration: 0 
        running: false 
        from: progressTransform.defaultStartAngle 
        to: -270 
        onStopped: if (animationState != "stopped") animationState = "stopped" 
    } 
} 
```

Finally, this mesh creates the progress bar, which moves from the left to the right to indicate playback progress.

The entire project is compiled by running qmake followed by make, or by opening the project in Qt Creator and building it from there. When run, it will automatically start playing the included song and show the amplitude visualization, while being controllable via the buttons in the UI.

# Summary

In this chapter, we looked at the myriad ways in which the Qt framework can be used to develop for embedded systems. We briefly looked at how it compares with other frameworks and how Qt is optimized for these embedded platforms, before working through an example of a QML-based GUI that could be added to the infotainment system we previously created.

You should now be able to create basic Qt applications, both purely command line-based and with a GUI. You should also have a clear idea of which options Qt offers to develop GUIs with.

In the next chapter, we will be taking a look at the next evolution of embedded platforms, using **field-programmable gate arrays** (**FPGAs**) to add custom, hardware-based functionality to speed up embedded platforms.