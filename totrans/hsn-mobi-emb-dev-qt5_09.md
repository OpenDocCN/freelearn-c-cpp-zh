# Machines Talking

Machine automation and IoT use various APIs for communication with each other. 

I like to say that you cannot have IoT without sensors. They truly define IoT. Sensors are everywhere these days. Cars, lights, and mobile phones all have a myriad of sensors. Laptop computers have led, light, touch, and proximity sensors. 

MQTT and WebSockets are communication and messaging protocols. One use of them is to send sensors to remote locations.

You will learn about using Qt APIs for machine-to-machine automation and communication to web applications using the `QWebSocket` and `QWebSocketServer` classes.

MQTT is a publish-and-subscribe-based TCP/IP protocol for sending sensor data over a limited bandwidth network using `QMqttMessage` to a `QMqttClient` and `QMqttSubscription`.

We will be covering the following topics:

*   **Sensory control **- QtSensor data
*   **WebSockets **- Bi-directional web communication
*   **QMqtt **- Brokers of machine talk

# Sensory control – QtSensor data

The Qt Sensors API started with Qt Mobility, which itself grew from Qtopia, which was later renamed Qt Extended.

Qt Mobility was a collection of APIs useful for mobile and embedded devices. It was intended specifically for use in Nokia phones. Some of the Mobility API was integrated into Qt 4 and later into Qt 5.

Qt Sensors, on the other hand, was put into its own repository when Qt 5 split into modules. Qt Sensors started out targeting mobile phone platforms, but as computers, such as laptops and Raspberry Pis, gained sensors, the backends expanded. You can find backends for iOS, Android, WinRT, generic Linux, Sensorfw, as well as Texas Instrument's SensorTag. At my GitHub repository, you can find additional sensor backends for Raspberry Pi Sense HAT, and MATRIX Creator for Raspberry Pi.

**Sensor Framework** (**SensorFW**) is a framework and backend for configuring and reading sensor data in a variety of ways. It is tried, tested, and used on some of the best alternative mobile devices. It has support for Hybris (which is used in Sailfish OS), Linux IIO sensors, as well as for reading directly from the Linux filesystem. If you are integrating a new device and need to read various sensors, I recommend using Sensor Framework, available from [https://git.merproject.org/mer-core/sensorfw/](https://git.merproject.org/mer-core/sensorfw/)[. ](https://git.merproject.org/mer-core/sensorfw/)

There are dozens of different sensors for monitoring the environment. Qt Sensors handles the most common sensors found in mobile phones and tablets, and provides tools to help implement new sensor types that may be developed and become popular.

Not only are sensors for monitoring the environment; they can also be used as an input to the system. The Qt Sensor API includes an ad-hoc `QSensorGestures`, which is an API for various device gestures, such as shake, free-fall, hover, cover, turnover, and pickup.

Qt Sensors has the C++ and QML APIs. Let's start with the C++ API.

There are actually three ways to use this API. The first is the generic way. All the sensor classes are derived from `QSensor`. A more generic way to use them is to just use `QSensor`.

# QSensor

`QSensor` has two static functions that we can use. `QSensor::sensorTypes()` which returns a `QList` of sensor types; for example, it could be `QLightSensor` or `QOrientationSensor`. You can then use `QSensor::sensorForType` or `defaultSensorForType`. Usually there is only one sensor for a type, so, using the latter will suffice.

But first, we need to tell `qmake` that we want to use the `sensors` module, so in the `.pro` file, do the following:

The source code can be found on the Git repository under the `Chapter07-1` directory, in the `cp7` branch.

```cpp
QT += sensors
```

To include all `QSensors` headers, the include file line is `#include <QtSensors>`, so let's add this to our file.

Get a list of all sensor types known to the system by using `QSensor::sensorTypes()`:

```cpp
 for (const QByteArray &type : QSensor::sensorTypes()) {
        const QByteArray &identifier = QSensor::defaultSensorForType(type);
```

`QSensor` is created by supplying a `QSensor::type` argument, and then you call the `setIdentifier` function with a `String` indicating the sensor you want to use.

```cpp
        QSensor* sensor = new QSensor(type, this);
        sensor->setIdentifier(identifier);

```

We now have a `QSensor`. You must then call `connectToBackend()` if you are using `QSensor` directly:

```cpp
if (!sensor->connectToBackend())
    qWarning() << "Could not connect to sensor backend";
```

You can then connect to the `readingChanged()` signal and read the values from there. To get the `QSensor`, you can use the `sender()` function in any slot, and then `qobject_cast` to cast to a `QSensor`:

```cpp
connect(sensor, &QSensor::readingChanged, this, &SomeClass::readingChanged);
```

The `readingChanged()` slot looks like this:

```cpp
void SomeClass::readingChanged()
{
    QSensor *sensor = qobject_cast<QSensor *>(sender());
    QSensorReading *reading = sensor->reading()
    QString values;
    for (int i = 0; i < reading->valueCount(); i++) {
        values += QString::number(reading->value(i).toReal()) + " ";
    }
  ui->textEdit->insertPlainText(sensor->type() +" " + sensor->identifier() + " "+ values + "\n");
}
```

We cast the `QSensor` using the `sender()` function, which returns the object that the slot is connected to. We then use that to get the `QSensorReading` using the `reading()` function. From the reading, we can get the values the sensor signaled to us.

We still need to call `start()` on the sensor, so we will add this somewhere after we connect to the `readingChanged()` signal. This will activate the sensor's backend and start reading data from the device.

```cpp
if (!sensor->isActive())
  sensor->start();
```

There is another way to access a sensor, and that is by using a `QSensor` subclass. Let's have a look at how we will use `QSensor` as a subclass:

# The QSensor subclass

A more popular way to use Qt Sensors is to use the standard `QSensors` derived classes, such as `QLightSensor` or `QAccelerometer`. This is useful if you know exactly which sensors your device has or what you are going to use. It also reduces the need for type-casting. In this way, it is also easier to use a class's sensor-specific properties:

```cpp
QLightSensor *lightSensor = new QLightSensor(this);
if (!lightSensor->connectToBackend()) {
    qWarning() << "Could not connect to light sensor backend";
    return;
}
connect(lightSensor, &QLightSensor::readingChanged, &SomeClass::lightSensorChanged);

```

Instead of a generic `QSensorReading`, we get a sensor specific reading, `QLightReading` in this case, with a sensor-specific value accessor:

```cpp
SomeClass::lightSensorChanged(const QLightReading *reading)
{
    qWarning() << reading->lux();
}
```

Another way to access sensor data is to use a `QSensorFilter`. Let's go there.

# QSensorFilter

There is a third way to access sensor data in C++, which is to use the sensor-specific filter class. This provides an efficient callback when signals and slots might be too slow, as in the case of `QAccelerometer` and other motion sensors which might be running at 200 cycles per second. It also provides a way to apply one or more filters that affect the values before they get emitted by the sensor reading signals. You could provide additional smoothing and noise reduction, or amplify the signal to a greater range.

 In our case, our class would inherit from `QLightFilter`.

```cpp
class LightFilter : public QLightFilter
{
public:
```

We then implement the filter override.

If the `filter` function returns `true`, it will store the `QLightReading` of the `QLightSensor` and the new values will be emitted by, in our case, the `QLightSensor` class. Let's apply a simple moving-average filter to our light sensor data:

```cpp

 bool filter(QLightReading *reading)
 { 
      int lux = 0;
        int averageLux = 0; 
        if (averagingList.count() <= 4) {
            averagingList.append(reading->lux());
        } else {
            for (int i = 0; i < averagingList.count(); i++) {
                lux += averagingList.at(i);
            }
            averageLux = lux / (averagingList.count());
            reading->setLux(averageLux);
            averagingList.append(averageLux);
            return true; // store the reading in the sensor
        }
        return false;
    };
    QList<int> averagingList;

};
```

You can then create a new `LightFilter` object and set `QLightSensor` to use it. Add this before the call to `start()`:

```cpp
            if (type == QByteArray("QLightSensor")) {
                LightFilter *filter = new LightFilter();
                sensor->addFilter(filter);
            }
```

Now let's find out about the `QSensor` data and how to access it.

# QSensor data

`QSensor` has values that are specific to the respective sensor. You can access them either generically with `QSensor`, or by sensor value.

# QSensorReading

If you are using the more generic `QSensor` class, there is a corresponding `QSensorReading` that you can use to retrieve the generic data. For getting any sensor-specific data you will need to use the corresponding sensors' `QSensorReading` subclass, such as `QAccelerometerReading`. For example, if we are using the `QSensor` to grab accelerometer data, we could do the following:

```cpp
QSensorReading reading;
QList <qreal> data;
qreal x = reading.at(0);
qreal y = reading.at(1);
if (reading.valueCount() == 3)
    qreal z = reading.at(2);
qreal timestamp = reading.timestamp;
```

However, using the `QAccelerometer` and `QAccelerometerReading` classes to do the same thing, would look like this.

```cpp
QAccelerometer accel;
QAccelerometerReading accelReading = accel.reading();
qreal x = accelReading.x();
qreal y = accelReading.y();
qreal z = accelReading.z();
```

Here are some data explanations for common sensors:

| **Sensor reading** | **Values** | **Unit** | **Description** |
| `QAccelerometerReading` | `x`, `y`, `z` | ms², meters per second squared | Linear acceleration along x, y, z axis |
| `QAltimeterReading` | `altitude` | Meters | Meters above average sea level |
| `QAmbientLightReading` | `lightLevel` | Dark, Twilight, Light, Bright, Sunny | General light level |
| `QAmbientTemperatureReading` | `temperature` | Celsius | Degrees Celsius |
| `QCompassReading` | `azimuth` | Degrees | Degrees from magnetic north |
| `QGyroscopeReading` | `x`, `y`, `z` | Degrees per second | Angular velocity around the axis |
| `QHumidityReading` | `absoluteHumidity`,`relativeHumidty` | gm³, grams per cubic meter | Water vapor in air |
| `QIrProximityReading` | `reflectance` | Decimal fraction 0 - 1 | How much infrared light was returned |
| `QLidReading` | `backLidClosed`,`frontLidClosed` | Bool | Laptop lid |
| `QLightReading` | `lux` | Lux | Light measured in lux |
| `QMagnetometerReading` | `x`, `y`, `z` |  Magnetic flux density | Raw flux |
| `QOrientationReading` | `orientation` | TopUp, TopDown, LeftUp, RightUp, FaceUp, FaceDown | Enum device orientation |
| `QPressureReading` | `pressure`, `temperature` | Pascals, Celsius | Atmospheric pressure |
| `QProximityReading` | `close` | Bool | Close or far away |
| `QRotationReading` | `x`, `y`, `z` | Degrees | Rotation around axis in degrees |

Some of these have sensor-specific readings, such as `QCompass` and `QMagnetometer`—both contain calibration levels.

Of course, C++ is not the only way to implement Qt's sensors; you can use them in QML as well. Let's find out how.

# QML sensors

Of course, you can also use Qt Sensors from QML. In a lot of ways, it is easier to implement them this way, as the Qt Quick API has been optimized and simplified, so it takes less time to get the sensor up and running. Following our previous use of the light sensors, we will continue here. First off is the ever-present `import` statement: instead of calling a `start()` function to get it rolling, there is an `active` property. Instead of a `Lux` value, the property is `illuminance`. Not quite sure why there's a difference, but there you go:

```cpp
import QtSensors 5.12
LightSensor {    
    id: lightSensor
    active: true
    onReadingChanged {
        console.log("Lux "+ illuminance);
 }
```

It cannot get much simpler than that. `QtSensors` QML has no filter, so if you need to filter anything, you will have to use C++.

# Custom QSensor and the backend engine

I want to briefly touch on how to create a custom sensor and engine backend. If you are on an embedded device, Qt Sensors may not have support for your sensor if it is a moisture or an air-quality sensor. You would need to implement your own `QSensor` and `QSensorBackend` engine.

There is a script in the directory, `src/sensors/make_sensor.pl`, that you can run which will generate a simple `QSensor` derived class, but additionally this script will generate Qt Quick classes that derive from `QmlSensor` . The `make_sensor.pl` script needs to be run from the `src/sensors` directory. For this exercise, we are going to create a sensor for monitoring salt concentrations in our saltwater swimming pool, so the name will be `QSaltSensor`.

You can then open these files in an editor, such as Qt Creator, and add what you need. Having a new `QSensor` type will also require a new backend that reads from the sensor. 

# Custom QSensor

There is a helper script named `QtSensors/src/sensors/make_sensor.pl`, which will create a basic template for a new `QSensor`, `QSensorReading`. It generates a simple `QSensor` derived class, but also classes for `QmlSensor` derived classes. 

If you do not have it in your source directory, it can be found in the Git repository at [https://code.qt.io/cgit/qt/qtsensors.git/tree/src/sensors/make_sensor.pl](https://code.qt.io/cgit/qt/qtsensors.git/tree/src/sensors/make_sensor.pl).

The `make_sensor.pl` script needs to be run from the `src/sensors` directory. 

You will have to edit the resulting files and fill in a thing. For this example, I chose `QSaltSensor` as a class name. Execute the script with the class name as the first argument: `make_sensor.pl QSaltSensor`.

It creates the following files:

*   `<sensorname>.h`
*   `<sensorname>.cpp`
*   `<sensorname>_p.h`
*   `imports/sensors/<sensorname>.h`
*   `imports/sensors/<sensorname>.cpp`

The output of using the `make_sensor.pl` command will appear like this:

```cpp
cd src/sensors
$perl ./make_sensor.pl QSaltSensor 
Creating ../imports/sensors/qmlsaltsensor.h 
Creating ../imports/sensors/qmlsaltsensor.cpp 
Creating qsaltsensor_p.h 
Creating qsaltsensor.h 
Creating qsaltsensor.cpp
You will need to add qsaltsensor to the src/sensors.pro file to the SENSORS and the qmlsaltsensor files to src/imports/sensors/sensors.pro
```

Like the output says, you will need to add `qsaltsensor` to the `src/sensors.pro` file to the `SENSORS` variable that is used there. Add the `qmlsaltsensor` filepaths in the file, `src/imports/sensors/sensors.pro`.

Start by editing `qsaltsensor.cpp`, which is the class we will use as our `QSensorBackend`. The `perl` script we used to create the template has added comments where you should edit to customize. You will also need to add any properties.

# Custom QSensorBackend

There are many reasons why you might want to implement your own sensor backend. One of these might be if you have a new type of sensor.

You would need to start implementing a backend engine for your new `QSensor` type. Begin by deriving from `QSensorBackend`:

```cpp
#ifndef LINUXSALT_H
#define LINUXSALT_H

#include <qsensorbackend.h>
#include <qsaltsensor.h>

class LinuxSaltSensor : public QSensorBackend
{
```

The class `QSensorBackend`, has two pure virtual functions you need to implement: `start()` and `stop()`:

```cpp

public:
    static char const * const id;
    LinuxSaltSensor(QSensor *sensor);
    ~LinuxSaltSensor();
    void start() override;
    void stop() override;
private:
    QSaltReading m_reading;
};
#endif // LINUXSALT_H
```

The source code can be found on the Git repository under the `Chapter07-2` directory, in the `cp7` branch.

Implementing the backend functionality is up to you, based on if you have a salt sensor device you want to use. Of course you will have to compile and deploy your own Qt Sensors when you do so.

For more information about custom `QSensors` and the backend, look at the Grue sensor example in Qt sensors. There is some rather amusing documentation on how to implement a custom sensor at [https://doc.qt.io/qt-5/qtsensors-grue-example.html](https://doc.qt.io/qt-5/qtsensors-grue-example.html)

Sometimes there are more than one sensor plugin on a system for any sensor. In this case we will need to tell the system which sensor to use. Let's look at how to configure `QSensors`. 

# Sensors.conf

If there is more than one backend for a particular sensor, you might need to specify which is the default.

You can add `qsaltsensor` to the `Sensors.conf` configuration file so the system can determine which sensor type is the default for a certain sensor. Of course, developers are free to choose whichever registered sensor on the system they want to use. The config file's format is `SensorType = sensor.id`, where `SensorType` is the base sensor class name, such as `QLightSensor`, and `sensor.id` is a `String` identifier for the specific sensor backend. The following code uses our `saltsensor` from the Linux backend and the list sensor from the `sensorfw` backend:

```cpp
[Default]
QSaltSensor = linux.saltsensor
QLightSensor = sensorfw.lightsensor
```

# QSensorGesture

`QSensorGesture` is an API for device gestures using sensors. As I mentioned in the introduction, they use ad-hoc gestures, which is to say there is no machine learning involved. Qt Sensors offers the following already-baked gestures:

*   `cover`
*   `doubletap`
*   `freefall`
*   `hover`
*   `pickup`
*   `slam`
*   `shake`
*   `turnover`
*   `twist`
*   `whip`

Instructions on how to perform specific gestures in Qt Sensors are detailed at [http://doc.qt.io/qt-5/sensorgesture-plugins-topics.html](http://doc.qt.io/qt-5/sensorgesture-plugins-topics.html).

It is worth noting that `QSensorGesture` uses signals specific to the recognizer. The `slam` gesture has the `slam()` signal, which gets emitted when it detects the `slam` gesture. It also has the standard `detected("<gesture>")` signal. The `shake2` gesture has the `shakeLeft`, `shakeRight`, `shakeUp` and `shakeDown` signals, but also the corresponding `detected` signals.

The `QSensorGesture` class does not have the `Q_OBJECT` macro, and creates its signals at runtime directly on the `meta` object. As such, `qobject_cast` and subclassing `QSensorGesture` while using `Q_OBJECT` will not work. 

`QSensorGestureManager` has the `recognizerSignals` function, which takes a `gestureId` so you can discover signals specific to the gesture, if you need to.

The source code can be found on the Git repository under the `Chapter07-3` directory, in the `cp7` branch.

To use `QSensorGestures`, create a `QSensorGesture` object, which takes a `QStringList` argument of a list of gesture IDs you want to use. You can specify directly which gestures you want using a `QStringList` like this:

```cpp
QSensorGesture *gesture = new QSensorGesture(QStringList() << "QtSensors.slam", this);
connect(gesture, SIGNAL(detected(QString)), this, SLOT(detectedGesture(QString)));

```

Alternatively you can also use `QSensorGestureManager` to get a list of all the registered gestures, calling `gestureIds()`.

Because of the atypical implementation of `QSensorGesture` (because the signals get dynamically created at runtime), using the new style connect syntax, `connect(gesture, &QSensorGesture::detected, this, &SomeClass::detectedGesture);`, will result in a compiler error, as the new style syntax has compile-time checks. 

Once you have these signals connected correctly, you call `startDetection()` for `QSensorGesture`:

```cpp
gesture->startDetection();
```

# QSensorGestureManager

You can get a list of all sensor gestures registered on the system by using `QSensorGestureManager`:

```cpp
QSensorGestureManager manager;

    for (const QString gestureId :  manager.gestureIds()) {
        qDebug() << gestureId;

       QStringList recognizerSignals = manager.recognizerSignals(gestureId);

        for (const QString signalId : recognizerSignals ) {
            qDebug() << " Has signal " << signalId;
        }
 }
```

You can use the `gestureId` from the preceding code to create a new `QSensorGesture` object and connect to the detected signal:

```cpp
QSensorGesture *gesture = new QSensorGesture(QStringList() << gestureId, this);
        connect(gesture,SIGNAL(detected(QString)), this,SLOT(detectedGesture(QString)));
```

# SensorGesture

Of course, sensor gestures can be used from QML. The API is slightly different in that there is only one type, `SensorGesture`, so it is like using the generic `QSensor` class, except that, instead of one gesture per object, `SensorGesture` can represent one or more gestures.

`SensorGesture` does not have its own import, and is lumped into `QtSensors`, so we need to use that to indicate we are using the `QtSensors` module:

```cpp
import QtSensors 5.12
```

You specify which gestures you want by writing to the `gestures` property, which takes a list of strings of the `id` recognizer:

```cpp
SensorGesture {
    id: sensorGesture
    gestures : [ "QtSensors.slam", "QtSensors.pickup" ]
}
```

Since there is only one generic `SensorGesture`, there are no gesture-specific signals. The gesture signal is `onDetected`, and a string of which gesture was detected is set in the `gesture` property. You will have to use some logic to filter for a certain gesture if you are using the component for more than one gesture:

```cpp
onDetected: {
    if (gesture == "slam") {
        console.log("slam gesture detected!")
    }
} 
```

To start the detection, write `true` to the `SensorGesture` enabled property:

```cpp
sensor.gesture.enabled
```

You can grab your device and perform the slam gesture as outlined in the Qt documentation at [http://doc.qt.io/qt-5/sensorgesture-plugins-topics.html](http://doc.qt.io/qt-5/sensorgesture-plugins-topics.html). Depending on your device, it will detect a slam.

# WebSockets – Bi-directional web communication

Now we are starting to get into the realm of network and the internet. WebSockets are a protocol that allows two-way data exchange between a web browser or client and a server without polling. You can stream data or send data at any time. Qt has support for WebSockets through the use of the `QWebSocket` API. Like normal TCP sockets, `QWebSockets` needs a server.

# QWebSocketServer

`QWebSocketServer` can work in two modes: non-secure and SSL. We start by adding `websockets` to the `.pro` file so `qmake` sets up the proper library and header paths:

```cpp
QT += websockets
```

Then include the `QWebSocketServer` header file:

```cpp
#include <QtWebSockets/QWebSocketServer>

```

The source code can be found on the Git repository under the `Chapter07-3` directory, in the `cp7` branch.

To create a `QWebSocketServer`, it takes a server name as a string, a mode, and a parent object. The mode can be `SecureMode` or `NonSecureMode`.

`SecureMode` is is like HTTPS, uses SSL, and the protocol is wss. `NonSecureMode` is like HTTPS with the ws protocol:

```cpp
const QWebSocketServer *socketServer = new QWebSocketServer("MobileSocketServer",
                                                            QWebSocketServer::NonSecureMode,this);
connect(sockerServer, &QWebSocketServer::newConnection, this, &SomeClass::newConnection);
connect(sockerServer, &QWebSocketServer::closed, this, &SomeClass::closed);
```

Like `QSocket`, there is a `newConnection` signal that gets emitted when a client attempts to connect to this server. If you are using `SecureMode`, you will want to connect to the `sslErrors(const QList<QSslError> &errors)` signal. Once the signals you want to use are connected, call `listen` to start the server, with a `QHostAddress` and a port number. `QHostAddress::Any` will listen to all network interfaces. You can specify one interface's address. The port number is optional and a port of 0 will be assigned a port automatically:

```cpp
socketServer->listen(QHostAddress::Any, 7532);
```

Now we have a `QWebSocketServer` object that listens to incoming connections. We can handle this much like we did with the `QSocketServer` using `nextPendingConnection` in the corresponding slot we used to connect with the `newConnection` signal. That will give us a `QWebSocket` object that represents the connecting client.

# QWebSocket

When a new connection comes in, `QWebSocketServer` emits the `newConnection` signal, which, here, calls the `newConnection` slot. We grab `QWebSocket` using the `nextPendingConnection` of the server object. With this, we connect to the `QWebSocket` signals:

```cpp
QWebSocket *socket = socketServer->nextPendingConnection();
```

The first signal I like to connect is the error signal, as it can help debug. Like the `QBluetooth` class, the `error` function is overloaded, so special syntax is needed to use this signal.

The `QWebSocket` `error` signal is overloaded, so it needs unique handling to compile. `QOverload` is what you need to use.

```cpp
connect(socket, QOverload<QAbstractSocket::SocketError>::of(&QWebSocket::error),
           this, &SomeClass::socketError);
```

There are two types of messages that can be sent and received: `text` and `binary`. We have to deal with those differently, so there are signals for each. They get emitted by the server when the client sends a `text` or `binary` message:

```cpp
connect(socket, &QWebSocket::textMessageReceived, 
    this, &SomeClass::textMessageReceived);
connect(socket, &QWebSocket::binaryMessageReceived, 
    this, &SomeClass::binaryReceived);
```

One difference between `binary` and `text` messages in WebSockets is that the `text` messages are terminated with the `0xFF` character.

The `textMessageReceived` signal sends a `QString`, while the `binaryMessageReceived` sends a `QByteArray`:

```cpp
SomeClass:binaryMessageReceived(const QByteArray &message) {
}
SomeClass:textMessageReceived(const QString &message) {
}
```

They also work on the frame level, but we are simply handling the entire message. If you have continuous streaming data of some kind, you might choose the `textFrameReceived` or `binaryFrameReceived` signals.

# Client

A WebSocket client would simply use `QWebSocket` and connect to a server that supports WebSockets. One use case would be a web page (client) that shows sensor data sent though a `QWebSocketServer`.

# QtQuick

Of course, the `QWebSockets` API provides QML components – WebSocket and `WebSocketServer` to be exact. As usual, it is quicker than using C++.

# WebSocketServer

Add the following import line for your `qml` file to use WebSockets:

The source code can be found on the Git repository under the `Chapter07-4` directory, in the `cp7` branch.

```cpp
import QtWebSockets 1.0
```

To start listening with `WebSocketServer`, set the `listen` property to `true`. The `url` property, which takes a string, can be set to the address that clients will connect to:

```cpp
WebSocketServer {
    id: socketServer
    url : "ws://127.0.0.1:33343"
    listen: true
}
```

When a client connects, the `onClientConnected` signal gets emitted, and its `webSocket` property represents the incoming WebSocket client. You also want to be able to do error checking so `WebSocketServer` has the `onErrorStringChanged` signal, with the `errorString` property. To do so, in the `WebSocketServer` component, implement it like this.

```cpp
onClientConnected {
  ...
}

onErrorStringChanged {
    console.log(errorString)
}
```

Let's see how to handle the WebSocket for both server and client.

# WebSocket

Both the client and server use WebSocket element. In the server, as I outlined in the *WebSocketServer* section, the client's `WebSocket` object can be obtained via the `onClientConnect` signal.

Check out how this works in the `WebSocketServer` component, as compared to the client:

```cpp
WebSocketServer {
      id: socketServer
      host : "127.0.0.1"
      port: 33343
      listen: false
      onClientConnected {
         webSocket.onTextMessageReceived.connect(function(message)         {
              console.log(message)
         });
      }
  }
```

The client requires the `url` property to be populated so it knows which server it will connect to:

```cpp

WebSocket {
    id: webSocket
    url: "ws://localhost"

    onTextMessageReceived {
        console.log(message)

    }
}
```

The incoming message appears in the `onTextMessageReceived` signal with the `message` property.

To send a message to the server or client, `WebSocket` has the `sendMessage` function. If this is the server, the `webSocket` would be used to send a message of text like this. 

```cpp
webSocket.sendTextMessage("socket connected ok!")
```

WebSockets for Qt Quick does not handle binary messages in the true sense of the word. It does happen to have an `onBinaryMessageReceived` signal, but the `message` object that gets received is a `String`. I would suggest that if your binary message will get messed up by being converted to UTF16-encoded `QString`, you might consider using the C++ API.

# QMqtt – Brokers of machine talk

MQTT is a publish-and-subscribe messaging transport. There was a similar framework in the Qt Mobility stack called Publish and Subscribe, which is now part of the officially unsupported `QSystems` API framework, which also includes `QSystemInfo` and `QSystemFramework`.

`QMqtt` is a framework for writing MQTT clients. You will need to install and run an MQTT broker, such as Mosquitto or HiveMQ, or use an internet-based service. For my development and testing purposes, I chose HiveMQ. You can download it from [https://www.hivemq.com/](https://www.hivemq.com/).

They also have a public broker at [http://www.mqtt-dashboard.com/index.html](http://www.mqtt-dashboard.com/index.html).

MQTT has a broker, or server that one or more clients connect to. The clients can then publish and/or subscribe to different topics.

You can use `QWebSockets` to access a broker, and there is an example in Qt, which uses the `WebSocketIODevice` class in the `examples/mqtt/websocketsubscription` directory.

# QMqttClient

To start developing a `QMqttClient`, you will have to build it yourself, as it does not get distributed with Qt itself, unless you get the commercial Qt for Automation.

You download the open source licensed version from `git://code.qt.io/qt/qtmqtt.git`.

Luckily it is a straightforward and easy build. Once you run `qmake; make && make install;`, you are ready to use it. 

In your `pro` file, we need to add the `mqtt` module.

```cpp
QT += mqtt
```

The header file is named `QtMqtt/QMqttClient`, so let's include that:

```cpp
#include <QtMqtt/QMqttClient>
```

The source code can be found on the Git repository under the `Chapter07-5` directory, in the `cp7` branch.

The main class we use to access the broker is named `QMqttClient`. It can be thought of as the manager. It has a simple construction. You need to specify the host and port with the `setHostname` and `setPort` functions. We will use the `hivemq` public broker and port `1883`:

```cpp
mqttClient = new QMqttClient(this);
mqttClient->setHostname(broker.hivemq.com);
mqttClient->setPort(1883);

```

It is a good idea connect to any error signals to help debugging when things go wrong; let's do that first:

```cpp
connect(mqttClient, &QMqttClient::errorChanged, this, &SomeClass::errorChanged);
connect(mqttClient, &QMqttClient::stateChanged, this, &SomeClass::stateChanged);
connect(mqttClient, &QMqttClient::messageReceived, this, &SomeClass::messageReceived);
```

To connect to the `mqtt` broker, call `connectToHost();`:

```cpp
mqttClient->connectToHost();
```

Since we connected to the `stateChanged` signal, we can wait until we are connected to the broker to subscribe to any topics:

```cpp
void SomeClass::stateChanged(QMqttClient::ClientState state)
{
   switch(state) {
    case QMqttClient::Connecting:
        qDebug() << "Connecting...";
        break;
    case QMqttClient::Connected:
        qDebug() << "Connected.";
        subscribe();
        break;
    case QMqttClient::Disconnected:
        qDebug() << "Disconnected."
        break;
    }
}
```

The `QMqttClient::subscribe` function takes a topic in the form of `QMqttTopicFilter`. Here, I assign it the `"Qt"` string.

It returns a `QMqttSubscription` pointer, which we can use to connect to the `stateChanged` signal. We will then simply subscribe to the topic we just published.

Our `subscribe` function looks like this:

```cpp
void MainWindow::subscribe()
{
    QMqttTopicFilter topicName("Qt");
    subscription = mqttClient->subscribe(topicName, 0);
    connect(subscription, &QMqttSubscription::stateChanged,this,    
            &SomeClass::subscriptionStateChanged);
    publish();
}
```

We simply call our function that will then publish something to that topic.

`QMqttClient::publish` takes a topic name in the form of a `QMqttTopicName`, and the message is just a standard `QByteArray`.

The `publish` function looks like this:

```cpp
void MainWindow::publish()
{
    QMqttTopicName topicName("Qt");
    QByteArray topicMessage("Everywhere!");
    mqttClient->publish(topicName, topicMessage);
}
```

You should then see the message we published in the `messageReceived` slot: 

![](img/5b5ad8cf-4496-41a9-a675-caa991a0bc0e.png)

# Putting it all together

I have a Raspberry Pi and a Sense HAT board that I can use to collect sensor data. Luckily, I previously wrote a Qt Sensors plugin for the Sense HAT. It happens to be in a standalone Git repository and not in any Qt Sensors version, unlike the TI SensorTag backend plugin.

If you don't want to write your own Sense HAT sensor plugin you can get my standalone Sense HAT plugin from [https://github.com/lpotter/qsensors-sensehatplugin.git](https://github.com/lpotter/qsensors-sensehatplugin.git).

The version of Qt Sensors on the Raspbian distribution is 5.7 and does not have the pressure and humidity sensors that the Sense HAT has. They were added in later Qt Sensors versions.

Cross-compiling on a desktop is so much faster than compiling on the device—days on the **Raspberry Pi** (**rpi**) as opposed to a few minutes on a good development machine. I had some trouble getting the cross-compiling `toolchain` to work, so I had to opt for the on-board native compile, which of course takes a very long time on a Raspberry Pi. The easiest way is to get Qt's commercial `Boot2Qt` and `Automation` packages, as they package it up nicely, and provide binaries and support.

 Since this book uses Qt 5.12, we need to get the explicit version of the following Qt module repositories, by using the following Git commands:

*   Qt Base: `git clone http://code.qt.io/qt/qtbase.git -b 5.12`
*   Qt WebSockets: `git clone http://code.qt.io/qt/qtwebsockets.git -b 5.12`
*   Qt MQTT: `git clone http://code.qt.io/qt/qtmqtt -b 5.12`
*   Qt Sensors: `git clone http://code.qt.io/qt/qtsensors -b 5.12`

We are going to create an app for Raspberry Pi that grabs the Sense HAT's temperature and pressure data and distributes them via `QMqtt` and `QWebSockets` to a broker running on HiveMQ.

The source code can be found on the Git repository under the `Chapter07-6` directory, in the `cp7` branch.

Start by implementing a `SensorServer` class, which is typically a `QObject` derived class.

```cpp
SensorServer::SensorServer(QObject *parent)
   : QObject(parent),
      humiditySensor(new QHumiditySensor(this)),
      temperatureSensor(new QAmbientTemperatureSensor(this))
{
    initSensors();
    initWebsocket();
}
```

We then implement the `QWebSockeIODevice` that we declared as `mDevice` and connect to its `socketConnected` signal.

```cpp
void SensorServer::initWebsocket() 
{ 
    mDevice.setUrl("broker.hivemq.com:8000"); 
    mDevice.setProtocol("mqttv3.1"); 

    connect(&mDevice, &WebSocketIODevice::socketConnected, this, &SensorServer::websocketConnected); 
} 

```

Next we call the `connectToBackend()` function of the sensors we want to use.

```cpp
void SensorServer::initSensors() 
{ 
    if (!humiditySensor->connectToBackend()) { 
        qWarning() << "Could not connect to humidity backend"; 
    } else { 
        humiditySensor->setProperty("alwaysOn",true); 
        connect(humiditySensor,SIGNAL(readingChanged()), 
                this, SLOT(humidityReadingChanged())); 
    }    
    if (!temperatureSensor->connectToBackend()) { 
        qWarning() << "Could not connect to humidity backend"; 
    } else { 
        temperatureSensor->setProperty("alwaysOn",true); 
        connect(temperatureSensor,SIGNAL(readingChanged()), 
                this, SLOT(temperatureReadingChanged())); 
    } 
}

```

The call to `initSensors()` connects to the sensor's backend and sets up `readingChanged` signal connections.

To use `QWebSockets` for the `QMqtt` client, we need to create a `QIODevice` that uses `QWebSockets`. Luckily, there is one already written in the `QMqtt` `examples/mqtt/websocketssubscription` directory, named `websocketsiodevice`, so we will import that into the project:

```cpp
SOURCES += websocketiodevice.cpp
HEADERS += websocketiodevice.h
```

In our header file, we include `websocketdevice.h`.

```cpp
#include "websocketiodevice.h"
```

In the class declaration, we instantiate the `WebSocketIODevice` as a class member.

```cpp
WebSocketIODevice mDevice;
```

To actually use `WebSocketIODevice`, we need to set it as the `QMqttClient` transport.

We first set up our `WebSocketIODevice` and connect to its `socketConnected` signal to set up `QMqtt`.

The `mqtt` broker at `hivemq` uses a different port number, so we set it in the URL:

```cpp
void SensorServer::initWebsocket()
{
    mDevice.setUrl(QUrl("broker.hivemq.com:8000"));
    connect(&mDevice, &WebSocketIODevice::socketConnected, this, &SensorServer::websocketConnected);
    mDevice.open(QIODevice::ReadWrite);
}
```

Now we set up `QMqtt` and set its transport to use `WebSocketIODevice`. We are using a transport with its own connection, so we do not set the URL for the `QMqtt` object, but rely on the `websocket` for connection. We then set up `mqttClient` as usual:

```cpp
void SensorServer::websocketConnected()
{
    mqttClient = new QMqttClient(this);
    mqttClient->setProtocolVersion(QMqttClient::MQTT_3_1);
    mqttClient->setTransport(&mDevice, QMqttClient::IODevice);
    connect(mqttClient, &QMqttClient::errorChanged,
            this, &SensorServer::errorChanged);
    connect(mqttClient, &QMqttClient::stateChanged,
            this, &SensorServer::stateChanged);
    connect(mqttClient, &QMqttClient::messageReceived,
            this, &SensorServer::messageReceived);

    mqttClient->connectToHost();
}
```

We monitor the changing state and act when it becomes `Connected`. We will start the `humidity` and `temperature` sensor, and then call subscribe so we can monitor when the `mqtt` broker is publishing:

```cpp
void SensorServer::stateChanged(QMqttClient::ClientState state)
{
    switch(state) {
    case QMqttClient::Connecting:
        qDebug() << "Connecting...";
        break;
    case QMqttClient::Connected:
        qDebug() << "Connected.";
        humiditySensor->start();
        temperatureSensor->start();
        subscribe();
        break;
    case QMqttClient::Disconnected:
        qDebug() << "Disconnected.";
        break;
    }
}
```

In our sensor's `readingChanged` slots, we will publish the data to the `mqtt` broker:

```cpp
void SensorServer::humidityReadingChanged()
{
    qDebug() << Q_FUNC_INFO << __LINE__;
    QHumidityReading *humidityReading = humiditySensor->reading();
    QByteArray data;
    data.setNum(humidityReading->relativeHumidity());
    QMqttTopicName topicName("Humidity");
    QByteArray topicMessage(data);
    mqttClient->publish(topicName, topicMessage);
}

void SensorServer::temperatureReadingChanged()
{
    qDebug() << Q_FUNC_INFO << __LINE__;
    QAmbientTemperatureReading *tempReading = temperatureSensor
>reading();
    QByteArray data;
    data.setNum(tempReading->temperature());
    QMqttTopicName topicName("Temperature");
    QByteArray topicMessage(data);
    mqttClient->publish(topicName, topicMessage);
}
```

Finally, let's see any subscribed messages:

```cpp
void SensorServer::messageReceived(const QByteArray &message, const QMqttTopicName &topic)
{
    qDebug() << Q_FUNC_INFO  << topic << message;
}
```

# Summary

In this chapter, we looked at the different ways of using `QSensors` to read a device's sensor data. There are many supported platforms for Qt Sensors: Android, iOS, WinRT , SensorTag, Sensorfw, Linux generic, and Linux iio-sensor-proxy. Sensorfw also has support for Linux's IIO sensors.

I described how to implement custom `QSensor` and `QSensorBackend` to add support for sensors not currently supported in Qt Sensors.

We went through the steps involved in using `QtMqtt` to talk to an `mqtt` broker, and we looked at how to use `QWebsockets` to communicate to a web server web page.

Then I threw them all together to grab sensor data from a Sense HAT, and publish them to `mqtt` broker by way of WebSockets.

In the next chapter, we will discuss using GPS data comprising of location and position and mapping.