# Input and Touch

Not all devices have a readily available keyboard. With a touchscreen device, users can easily use buttons and other **User Interface** (**UI**) features. What do you do when there is no keyboard or mouse, like on a kiosk or interactive signage? Virtual keyboards and touch interaction define mobile and embedded applications these days.

In this chapter, we will cover the following topics:

*   We will discover Qt's graphical solutions to incorporating user input.
*   The reference Qt Virtual Keyboard will be examined.
*   We will demonstrate Touch components, such as `TouchPoints`, `Flickable`, and `PinchArea`.

# What to do when there's no keyboard

Dude, where's my keyboard?

Computer information kiosks and cars do not usually come with keyboard inputs. They use virtual inputs, such as a virtual keyboard, voice inputs, or even gesture recognition.

People at the Qt Company have created a virtual input method they named **Qt Virtual Keyboard** (**QtVK**). It's more than just an onscreen keyboard, as it also has handwriting recognition. It is available under a commercial license as well as the open source GPL version 3.

There are other virtual keyboards that will work with Qt apps. On a desktop computer that also has a touchscreen, such as a two-in-one laptop, the system might already have a virtual keyboard. These should work as an input method for Qt apps, although they may or may not automatically pop up when the user wants to input into a text area. 

There are two ways to integrate Qt's Virtual Keyboard:

| Desktop system | Fully integrated into applications |
| Application | Qt Widget apps: Set environmental `QT_IM_MODULE=qtvirtualkeyboard` variableQt Quick: Use `InputPanel` in your application |

I have a Raspberry Pi setup here for Boot to Qt, which is fully integrated into the Qt Creator, so I can build and run Qt apps on the Raspberry Pi from the Qt Creator. You can also grab the sources and build it yourself from `git://code.qt.io/qt/qtvirtualkeyboard.git`.

To build the QtVK, download the following source:

`git clone git://code.qt.io/qt/qtvirtualkeyboard.git`

QtVK build can be configured by `qmake`, using `CONFIG+=<configuration>` and the following configuration options:

*   `lang <code>`
    *   `form of language_country`
        *   Language is lowercase, a two-letter language code
        *   Country is uppercase, a two-letter country code
*   `lang-all`
*   `handwriting`

    *   Handles custom engines
*   Arrow-key navigation

For example, to configure only Australian-English and add handwriting support, you would run `qmake CONFIG+=lang-en_AU CONFIG+=handwriting` and then `make && make install`.

There are many other configurations available. You can disable layouts for creating a custom layout and desktop integration, among other configurations.

QtVK can be used in C++ or QML. Let's start by using Qt Quick by ticking the Use Qt Virtual Keyboard in the Qt Creator project wizard when creating a new Qt Quick Application from a template:

![](img/3a8ba860-c0ee-4071-a651-76cb98a33e95.png)

This is the boiler plate code you get when you use Boot to Qt for Device Creation:

```cpp
    InputPanel {
        id: inputPanel
        z: 99
        x: 0
        y: window.height
        width: window.width
        states: State {
            name: "visible"
            when: inputPanel.active
            PropertyChanges {
                target: inputPanel
                y: window.height - inputPanel.height
            }
        }
        transitions: Transition {
            from: ""
            to: "visible"
            reversible: true
            ParallelAnimation {
                NumberAnimation {
                    properties: "y"
                    duration: 250
                    easing.type: Easing.InOutQuad
                }
            }
        }
    }
```

The source code can be found on the Git repository under the `Chapter04-1` directory, in the `cp4` branch.

Let's add something that takes text as input, such as a `TextField` element:

```cpp
    TextField {
        anchors {
            bottom: inputPanel.top
            top: parent.top
            right: parent.right
            left: parent.left
        }
        placeholderText: "Enter something"
    }

```

The `anchors` here are used to resize this `TextField` element when QtVK is opened automatically when the user taps on `TextField`. 

Here is what this should look like:

![](img/8ace3bbe-11dc-4241-8646-63059ba98f12.png)

Implementing a touch screen can have many benefits, which we can do by using the Qt event loop. Let's look in detail at using touch screens as an input.

# Using touch input

Touch screens are ubiquitous these days. They are everywhere. While essential in a mobile phone or tablet, you can also get a laptop or desktop computer with one. Refrigerators and cars also commonly have touchscreens. Knowing how to utilize these in your Qt app is also essential.

On mobile phone and tablet platforms, touchscreen support comes from the system and is often built-in. If you are creating your own embedded device, you will most likely need to tell Qt how to use the touchscreen. Qt has support for various touchscreen systems on embedded devices.

# QEvent

`QEvent` is the way to get access to the touch input events in C++. It comes through an event filter you can add to your application. There are a few different ways to access this data.

We can use an event filter or an event loop. We will start by looking at an event filter.

# Event filter

One way you can access the event loop is by using an event filter. You first need to call the following function: 

```cpp
qApp->installEventFilter(this);
```

The source code can be found on the Git repository under the `Chapter04-2` directory in the `cp4` branch.

You then need to override the function named `eventFilter(QObject* obj, QEvent* event)`, which returns a `bool `value:

```cpp
bool MainWindow::eventFilter(QObject* obj, QEvent* event);
```

You will then receive any and all events. You can also handle these touch events by using the following: 

*   `QEvent::TouchBegin`
*   `QEvent::TouchCancel`
*   `QEvent::TouchEnd`
*   `QEvent::TouchUpdate`

Using a `switch` statement in a `eventFilter` is an effective way to go through different options:

```cpp
bool MainWindow::eventFilter(QObject* obj, QEvent* event)
{
    switch(event->type()) {
        case QEvent::TouchBegin:
        case QEvent::TouchCancel:
        case QEvent::TouchEnd:
        case QEvent::TouchUpdate:
            qWarning("Touch event %d", event->type());
            break;
        default:
            break;
    };
    return false;
}
```

Be sure to pass these events on to the parent class unless you need to intercept them. To not pass these on, return `true`. Using an event loop is another way to access events. Let's take a look.

# Event loop

To use the event loop, you need to override `event(QEvent *ev)`:

```cpp
bool MainWindow::event(QEvent *ev)
{
  switch (ev->type()) {
    case QEvent::TouchBegin:
      qWarning("TouchBegin event %d", ev->type());
        break;
    case QEvent::TouchEnd:
      qWarning("TouchEnd event %d", ev->type());
        break;
    case QEvent::TouchUpdate:
      qWarning("TouchUpdate event %d", ev->type());
        break;
};}
```

You also need to add `setAttribute(Qt::WA_AcceptTouchEvents, true);` to the class constructor, otherwise your application will not receive touch events.

Let's take a look at how touchscreen support is handled in Qt and how you can use Qt to access a lower level of the touchscreen input stack.

# Touchscreen support

Touchscreen support for Qt is done through the **Qt Platform Abstraction** (**QPA)** platform plugins. 

Qt configure will auto-detect the correct platform and determine whether or not the development files are installed. If it finds the development files, it will use them.

Let's see how touchscreens work for various operating systems, starting out with mobile phones.

# Windows, iOS and Android

On Windows, iOS and Android, touchscreens are supported through the Qt Event system.

Using the Qt event system and allowing the platform plugins to do the scanning and reading, we can use `QEvent` if we need access to those events.

Let's look at how we can access a low level of the input system using Qt on embedded Linux.

# Linux

On the Linux operating system there are a variety of input systems that can be used with Qt.

Qt has built-in support for these types of touchscreen interfaces:

*   `evdev`: an event device interface
*   `libinput`: a library to handle input devices
*   `tslib`: a typescript runtime library

We will start by learning about the Linux `evdev` system to read the device files directly.

# evdev

Qt has built-in support for the `evdev` standard event-handling system for Linux and embedded Linux . This is what you will get by default if no other system is configured or detected. It handles keyboard, mouse, and touch. You can then use Qt as normal with respect to keyboard, touch, and mouse events.

You can assign startup parameters, such as device file path and default rotation of the screen, like this: 

`QT_QPA_EVDEV_TOUCHSCREEN_PARAMETERS=/dev/input/input2:rotate=90`

Other parameters available are `invertx` and `inverty`. Of course, you do not need to reply on Qt for these input events, and can access them directly in the stack below Qt. I call them raw events, but they are really just reading the special Linux kernel device files.

Let's take a look at handling these `evdev` input events yourself while using Qt. This is low-level system file access, so you might need root or administrator permissions to run the applications that use it this way.

Input events on Linux are accessed through the kernel's `dev` nodes, typically found at `/dev/input`, but they could be anywhere under the `/dev` directory tree, depending on the driver. `QFile` should not be used for actually reading from these special device node files.

`QFile` is not suited for reading Unix device node files. This is because `QFile` has no signals and the device node files report a size of zero and only have data when you read them.

The main `include` file to read input nodes is as follows:

```cpp
#include <linux/input.h>
```

The source code can be found on the Git repository under the `Chapter04-3` directory in the `cp4` branch.

You will want to scan the device files to detect which file the touchscreen produces. In Linux, these device nodes are named dynamically, so you need to use some other method to discern the correct file other than just the filename. So, you have to open the file and ask it to tell you its name.

We can use `QDir` and its filters to at least filter out some of the files we know are not what we are looking for:

```cpp
     QDir inputDir = QDir("/dev/input");
     QStringList filters;
     filters << "event*";
     QStringList eventFiles = inputDir.entryList(filters,
QDir::System);
     int fd = -1;
     char name[256];
     for (QString file : eventFiles) {
         file.prepend(inputDir.absolutePath());
         fd = ::open(file.toLocal8Bit().constData(), O_RDONLY|O_NONBLOCK);
 if (fd >= 0) {
 ioctl(fd, EVIOCGNAME(sizeof(name)), name);
 ::close(fd);
 }
}
```

Be sure to include the `O_NONBLOCK` argument for `open`.

At this point, we have a list of the names for the different input devices. You might have to just guess which name to use and then do a `String` compare to find the correct device. Sometimes, the driver will have correct `id` information, which can be obtained using `EVIOCGID` like this:

```cpp
unsigned short id[4];
ioctl(fd, EVIOCGID, &id);
```

Sometimes, you can detect certain features using `EVIOCGBIT`. This will tell us which buttons or keys the hardware driver supports. The touchscreen driver outputs a keycode of `0x14a (BTN_TOUCH)` when you touch it, so we can use this to detect which input event will be our touchscreen:

```cpp
bool MainWindow::isTouchDevice(int fd)
{
    unsigned short id[4];
    long bitsKey[LONG_FIELD_SIZE(KEY_CNT)];
    memset(bitsKey, 0, sizeof(bitsKey));
    ioctl(fd, EVIOCGBIT(EV_KEY, sizeof(bitsKey)), bitsKey);
    if (testBit(BTN_TOUCH, bitsKey)) {
        return true;
    }
    return false;
}
```

We can now be fairly certain that we have the proper device file. Now, we can set up a `QSocketNotifier` object to notify us when that file is activated, and then we can read it to get the `X` and `Y` values of the touch. We use the `QSocketNotifier` class because we cannot use `QFile`, as it doesn't have any signals to tell us when the Linux device files get changed, so this makes it much easier:

```cpp
int MainWindow::doScan(int fd)
{
    QSocketNotifier *notifier
        = new QSocketNotifier(fd, QSocketNotifier::Read,
         this);
        auto c = connect(notifier,  &QSocketNotifier::activated,
                     [=]( int /*socket*/ ) {
        struct input_event ev;
        unsigned int size;
        size = read(fd, &ev, sizeof(struct input_event));
        if (size < sizeof(struct input_event)) {
            qWarning("expected %u bytes, got %u\n", sizeof(struct
            input_event), size);
            perror("\nerror reading");
            return EXIT_FAILURE;
        }
        if (ev.type == EV_KEY && ev.code == BTN_TOUCH)
            qWarning("Touchscreen value: %i\n", ev.value);
        if (ev.type == EV_ABS && ev.code == ABS_MT_POSITION_X)
             qWarning("X value: %i\n", ev.value);
         if (ev.type == EV_ABS && ev.code == ABS_MT_POSITION_Y)
             qWarning("Y value: %i\n", ev.value);
          return 0;
     });
 return true;
}
```

We also use the standard `read()` function instead of `QFile` to read this. 

The `BTN_TOUCH` event value tells us when the touchscreen was pressed or released.

The `ABS_MT_POSITION_X` value will be the touchscreen's `X` position, and the `ABS_MT_POSITION_Y` value will be the `Y` position.

There is a library that can be used to do the very same thing, which might be a little easier.

# libevdev 

When you use the library `libevdev`, you won't have to access such low level filesystem functions like a `QSocketNotifier` and read files yourself.

To use `libevdev`, we start by adding to the `LIBS` entry in our projects `.pro` file.

```cpp
LIBS += -levdev
```

The source code can be found on the Git repository under the `Chapter04-4` directory in the `cp4` branch.

This allows `qmake` to set up proper linker arguments. The `include` header would be as follows: 

```cpp
#include <libevdev-1.0/libevdev/libevdev.h>
```

We can borrow the initial code to scan the directory for device files from the preceding code, but the `isTouchDevice` function gets cleaner code:

```cpp
bool MainWindow::isTouchDevice(int fd)
{
    int rc = 1;
    rc = libevdev_new_from_fd(fd, &dev);
    if (rc < 0) {
        qWarning("Failed to init libevdev (%s)\n", strerror(-rc));
        return false;
    }
    if (libevdev_has_event_code(dev, EV_KEY, BTN_TOUCH)) {
        qWarning("Device: %s\n", libevdev_get_name(dev));
        return true;
    }
    libevdev_free(dev);
    return false;
}
```

`Libevdev` has the nice `libevdev_has_event_code` function that can be used to easily detect whether or not the device has a certain event code. This is just what we needed to identify the touchscreen! Notice the `libevdev_free` function, which will free the memory being used that we do not need.

The `doScan` function loses the call to read, but substitutes a call to `libevdev_next_event` instead. It can also output a nice message with the actual name of the event code by calling `libevdev_event_code_get_name`: 

```cpp
int MainWindow::doScan(int fd)
{
    QSocketNotifier *notifier
            = new QSocketNotifier(fd, QSocketNotifier::Read,
this);
    auto c = connect(notifier,  &QSocketNotifier::activated,
                     [=]( int /*socket*/ ) {
        int rc = -1;
        do {            struct input_event ev;
            rc = libevdev_next_event(dev,
LIBEVDEV_READ_FLAG_NORMAL, &ev);
            if (rc == LIBEVDEV_READ_STATUS_SYNC) {
                while (rc == LIBEVDEV_READ_STATUS_SYNC) {
                    rc = libevdev_next_event(dev, LIBEVDEV_READ_FLAG_SYNC, &ev);
                }
            } else if (rc == LIBEVDEV_READ_STATUS_SUCCESS) {
                if ((ev.type == EV_KEY && ev.code == BTN_TOUCH) ||
                        (ev.type == EV_ABS && ev.code ==
ABS_MT_POSITION_X) ||
                        (ev.type == EV_ABS && ev.code ==
ABS_MT_POSITION_Y)) {
                    qWarning("%s value: %i\n",
libevdev_event_code_get_name(ev.type, ev.code), ev.value);
                }
            }
        } while (rc == 1 || rc == 0 || rc == -EAGAIN);
        return 0;
    });
    return 0;
}
```

The library `libinput` also uses `evdev`, and is a bit more up to date than the others.

# libinput 

The `libinput` library is the input handling for Wayland compositors and X.Org window system. Wayland is a display server protocol a bit like a newer version of the ancient Unix standard X11\. `Libinput` depends on `libudev` and supports the following input types:

*   `Keyboard`: Standard hardware keyboard
*   `Gesture`: Touch gestures
*   `Pointer`: Mouse events
*   `Touch`: Touchscreen events
*   `Switch`: Laptop lid switch events
*   `Tablet`: Tablet tool events
*   `Tablet pad:` Tablet pad events

The `libinput` library has a build-time dependency upon `libudev ` ; therefore, to configure Qt, you will need `libudev` as well as the `libinput` development files or packages installed. If you need hardware keyboard support, the `xcbcommon` package is also needed.

Yet another touch library is `tslib`, which is specifically used in embedded devices, as it has a small filesystem footprint and minimal dependencies.

# Tslib

`Tslib` is a library used to access and filter touchscreen events on Linux devices; it supports multi-touch and Qt has support for using it. You will need to have the `tslib` development files installed. Qt will auto-detect this, or you can explicitly configure Qt with the following:

```cpp
configure -qt-mouse-tslib
```

It can then be enabled by setting `QT_QPA_EGLFS_TSLIB` or `QT_QPA FB_TSLIB` to `1`. You can change the actual device file path by setting the environmental variable called `TSLIB_TSDEVICE` to the path of the device node like this:

```cpp
 export TSLIB_TSDEVICE=/dev/input/event4
```

Let's now move on to see how we can use higher level APIs in Qt to utilize the touchscreen.

# Using the touchscreen

There are two ways the Qt backend uses the touchscreen. The events come in as a mouse using a point with `click` and `drag` events, or as multi-point touch-to-handle gestures, such as `pinch` and `swipe`. Let's get a better understanding about multi-point touch.

# MultiPointTouchArea

As I have mentioned earlier, to use multi-point touchscreens in QML, there is the `MultiPointTouchArea` type. If you want to use gestures in QML, you either have to use`MultiPointTouchArea` and do it yourself, or use `QGesture` in your C++ and handle custom signals in your QML components.

The source code can be found on the Git repository under the `/Chapter04-5` directory in the cp4 branch.

```cpp
    MultiPointTouchArea { 
          anchors.fill: parent 
          touchPoints: [ 
              TouchPoint { id: finger1 }, 
              TouchPoint { id: finger2 }, 
              TouchPoint { id: finger3 }, 
              TouchPoint { id: finger4 }, 
              TouchPoint { id: finger5 } 
          ] 
      } 

```

You declare the `touchPoints` property of `MultiPointTouchArea` with a `TouchPoint` element for each finger you want to deal with. Here, we are using five-finger points. 

You can use the `x` and `y` properties to move things around:

```cpp
      Rectangle { 
          width: 30; height: 30 
          color: "green" 
          radius: 50 
          x: finger1.x 
          y: finger1.y 
      }

```

You can also use touchscreen gestures in your app. 

# Qt Gestures

Gestures are a great way of utilizing user input. As I mentioned gestures in [Chapter 2](ab2105ed-232b-4c99-8fd8-4ca295f6f5f9.xhtml), *Fluid UI with Qt Quick*, I will mention the C++ API here, which is much more feature-rich than gestures in QML. Keep in mind that these are touchscreen gestures and not device or sensor gestures, which I will examine in a later chapter. `QGesture` supports the following built-in gestures:

*   `QPanGesture`
*   `QPinchGesture`
*   `QSwipeGesture`
*   `QTapGesture`
*   `QTapAndHoldGesture`

QGesture is an event-based API, so it will come through the event filter, which means you need to re-implement your `event(QEvent *event)` widgets as the gesture will target your widget. It also supports custom gestures by subclassing `QGestureRecognizer` and re-implementing `recognize`.

To use gestures in your app, you need to first tell Qt that you want to receive touch events. If you are using built-in gestures, this is done internally by Qt, but if you have custom gestures, you need to do this:

```cpp
setAttribute(Qt::WA_AcceptTouchEvents);
```

To accept touch events, you then need to call `QGraphicsItem::setAcceptTouchEvent(bool)` with `true` as the argument.

If you want to use unhandled mouse events for touch events, you can also set the `Qt::WA_SynthesizeTouchEventsForUnhandledMouseEvents` attribute.

You then need to define to Qt that you want to use certain gestures by calling the `grabGesture` function of your `QWidget` or `QGraphicsObject`class:

```cpp
grabGesture(Qt::SwipeGesture);
```

`QGesture` events are delivered to a specific `QWidget` class and not the current `QWidget` class that holds the focus like a mouse event would.

In your `QWidget` derived class, you need to re-implement the `event` function and then handle the `gesture` event when it happens:

```cpp
bool MyWidget::event(QEvent *event)
{
    if (event->type() == QEvent::Gesture)
        handleSwipe();
    return QWidget::event(event);
}
```

Since we are handling only one `QGesture` type, we know it is our target gesture. You can check whether or not this event is caused by a certain gesture by checking for its `pointer` using the `gesture` function that is defined as follows: 

`QGesture * QGestureEvent::gesture(Qt::GestureType type) const`

This can be implemented by the following:

```cpp
if (QGesture *swipe = event->gesture(Qt::SwipeGesture))
```

If the `QGesture` object called `swipe` is `nullptr`, then this event is not our target gesture.

It is also a good idea to check on the gesture's `state()`, which can be one of the following:

*   `Qt::NoGesture`
*   `Qt::GestureStarted`
*   `Qt::GestureUpdated` 
*   `Qt::GestureFinished`
*   `Qt::GestureCanceled`

You can create your own gesture using `QGestureRecognizer` by sub-classing `QGestureRecognizer` and re-implementing `recognize()`. This is where most of the work will be, as you will need to detect your gesture and are more likely detect what is not your gesture. Your `recognize()` function will need to return one of the values of the `enum` value, `QGestureRecognizer::Result`, which can be any of the following:

*   `QGestureRecognizer::Ignore`
*   `QGestureRecognizer::MayBeGesture`
*   `QGestureRecognizer::TriggerGesture`
*   `QGestureRecognizer::FinishGesture`
*   `QGestureRecognizer::CancelGesture`
*   `QGestureRecognizer::ConsumeEventHint`

There are heaps of edge cases you need to handle here to discern exactly what is and what is not your gesture. Do not be afraid if this function is complicated or long.

Another form of input that is becoming more popular is using your voice. Let's look at that next.

# Voice as input

Voice recognition and Qt has been around for a while now. IBM's ViaVoice was ported to KDE and was being ported to Trolltech's phone software suite Qtopia at the time that I became the Qtopia Community Liaison in 2003\. Consequently, it was worked on by the same developer who later dreamed-up what became Qt Quick. While the concept has essentially stayed the same, the technology has gotten better, and now you can find voice control in many different devices, including automobiles.

There are many competing systems, such as Alexa, Google Voice, Cortana, and Siri, as well as some open source APIs. Combined with voice search, voice input is an invaluable tool.

At the time of writing, Qt Company and one of its partners, **Integrated Computer Solutions** (**ICS**), have announced that they are working together on integrating the Amazon Alexa system with Qt. I have been told that it will be called `QtAlexaAuto` and be released under the lgpl v3 license. Since this has not been released at the time of writing, I cannot go into very much detail about how to use this implementation just yet. I am sure that, if or when it gets released, the API will be quite easy to use.

Amazon's **Alexa Voice Service** (**AVS**) **Software Development Kit** (**SDK**) works on Windows, Linux, Android and MacOS. You can even use a mic array component, such as the MATRIX creator with the Raspberry Pi.  Siri works on iOS and MacOS. Cortana works on Windows.

While none of these voice systems are integrated into Qt, they can be used with a custom integration. It is worth looking into them, depending on what your application will be doing and what device it will run on.

Alexa, Google Assistant, and Cortana have C++ APIs, and Siri can be used as well with its Objective-C API:

*   Alexa: [https://github.com/alexa/avs-device-sdk.git](https://github.com/alexa/avs-device-sdk.git)
*   Google Assistant: [https://developers.google.com/assistant/sdk/guides/library/python/embed/install-sample](https://developers.google.com/assistant/sdk/guides/library/python/embed/install-sample)

*   Cortana: [https://developer.microsoft.com/en-us/cortana](https://developer.microsoft.com/en-us/cortana)
*   Siri: [https://developer.apple.com/sirikit/](https://developer.apple.com/sirikit/) 

# QtAlexaAuto

QtAlexaAuto is a module created by Integrated Computer Solutions (ICS) and the Qt Company to enable the use of Amazon's Alexa Voice Service (AVS) from within Qt and QtQuick applications. You can use Raspberry Pi, Linux, Android, or other machines to prototype an application that uses voice as input.

At the time of writing this book, QtAlexaAuto is yet to be released, so you will need to search the internet for the url to download the source code. Some things may have changed in the official release from what it written in this book.

You will need to download, build and install the following SDKs from Amazon:

*   AVS Device SDK: git clone -b v1.9 [https://github.com/alexa/avs-device-sdk](https://github.com/alexa/avs-device-sdk)
*   Alexa Auto SDK: git clone -b 1.2.0 [https://github.com/alexa/aac-sdk](https://github.com/alexa/aac-sdk)

To build these you should follow instructions for your platform from this URL:

[https://github.com/alexa/avs-device-sdk/wiki](https://github.com/alexa/avs-device-sdk/wiki)

The basic steps for building QtAlexaAuto are the following:

1.  Sign-up for an Amazon developer account, and register a product. [https://developer.amazon.com/docs/alexa-voice-service/register-a-product.html](https://developer.amazon.com/docs/alexa-voice-service/register-a-product.html)
2.  Add your clientID and productID
3.  Install the requirements mentioned in the Alexa wiki
4.  Apply patches as detailed in install_aac_sdk.md
5.  Build and install AVS Device SDK
6.  Build and install Alexa Auto SDK
7.  Edit AlexaSamplerConfig.json
8.  Build QtAlexaAuto

There are a few patches you need to apply. Luckily there are instructions in install_aac_sdk.md, which instruct you on how to apply the patches from aac-sdk to the avs-device-sdk.

The file AlexaSamplerConfig.json needs to be edited and renamed to  AlexaClientSDKConfig_new_version_linux.json

The file then needs to be put into the directory where you are running the example from.

The QtAlexaAuto main QML component is named alexaApp, which corresponds to the QtAlexaAuto class.

When you run the example app, you will need to sign in to your Amazon developer account and link this application by entering a code the application gives you when you start it for the first time. These can be provided to the user by calling acctLinkUrl() and acctLinkCode(), or in QML by the alexaApp properties named accountLinkCode and accountLinkUrl.

Once this is linked to an account, you use voice input and the Alexa Voice Service by tapping on the button.

The function that runs when the user presses the talk button is tapToTalk(), and the startTapToTalk signal gets emitted.

![](img/2267a64d-0f9e-4de9-86fd-0edb84f65ade.png)

AVS has a notion of a RenderTemplate, which gets passed from the service so the application will be able to show the user visual information about the response. QtAlexaAuto handles Weather, media player templates, as well as some generic multipurpose templates. The RenderTemplate gets emitted as JSON documents and shown in the example application using QML components, which then parse and display the data.

This was just a quick look at QtAlexaAuto, as I did not have enough time to really dig into this new API before the publication of this book.

# Summary

User input is important, and there are various ways for users to interact with applications. If you are creating your own embedded device, you will need to decide what input methods to use. Touchscreens can increase usability because touching things is a very natural thing to do. Babies and even cats can use touchscreen devices! Gestures are a fantastic way to use touch input and you can even develop custom gestures for your application. Voice input is taking off right now. Whilst adding support for it might take a little work, it can be the right thing to do on some devices that require hands-free usage.

In the following chapter, we will learn about networking and its features.