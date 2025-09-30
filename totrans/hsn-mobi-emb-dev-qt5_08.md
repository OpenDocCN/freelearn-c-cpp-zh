# Connectivity with Qt Bluetooth LE

You will learn about using Qt Bluetooth **Low Energy** (**LE**) to build connectivity to devices that have LE Bluetooth radios. Bluetooth is more than mice, keyboards, and audio. Device discovery, data exchange, and other tasks involving Bluetooth Low Energy will be examined. We will use the `QBluetoothUuid`, `QBluetoothCharacteristic`, `QLowEnergyController`, and `QLowEnergyService` classes.

We will cover the following topics in this chapter:

*   What is Bluetooth Low Energy
*   Discovering and connecting with devices
*   Advertising services
*   Retrieving sensor data from remote device

# What is Bluetooth Low Energy?

**Bluetooth Low Energy** (**BLE**), or Bluetooth Smart as it is also called, was originally developed by Nokia under the name Wibree and was originally released in 2006\. It was integrated into the Bluetooth 4.0 specification and released in 2010.

Bluetooth is a wireless connection technology that operates in the 2,400-2,483.5 MHz range of the 2.4 GHz frequency band. There are 79 data channels it can choose for transmitting packets. BLE limits the data channels to 40.

BLE is targeted at mobile and embedded devices that require lower power consumption. Unlike Bluetooth, BLE is designed for devices that exchange small amounts of data periodically, as opposed to regular Bluetooth that was designed for continuous data streams. Most importantly, BLE has a sleep mode that it uses to conserve power.

Qt has support for BLE in the Qt Connectivity module alongside **Near-field Communication** (NFC). BLE has a number of profiles and services:

*   Alerts
*   Battery
*   Fitness
*   Health
*   HID
*   Internet
*   Mesh
*   Sensors

**Generic Attribute** (**GATT**) is used to store profiles, services, characteristics, and other data. Each entry is a unique 16-bit ID. The BLE connection is exclusive in that it can only connect to one computer at a time. The BLE peripheral device is known as the GATT server, and the computer it connects to is the GATT client.

Each profile can have a number of services. Each service can have a number of characteristics. A profile is just the collection of pre-defined services in the specification.

A service is just a group of characteristics defined by a unique 16 or 128-bit ID. A characteristic is a single data point, which may contain an array of data, such as with an accelerometer.

Now that you know a little bit of the background, let's get started.

# Implementing a BLE GATT server

I guess we really need a BLE server now.

Let's say you have an embedded device that has a few environmental sensors attached, such as humidity and temperature. You need to send this data over Bluetooth to another handheld device once in a while. On the embedded sensor device, you need to setup the device. The basic procedure to set up the BLE server is as follows:

1.  Supply advertisement data (`QLowEnergyAdvertisingData`)
2.  Supply characteristic data (`QLowEnergyCharacteristicData`)
3.  Set up the service data (`QLowEnergyServiceData`)
4.  Start advertising and listening for connections

# QLowEnergyAdvertisingData

`QLowEnergyAdvertisingData` is the class you use to tell the server what and how the data is going to be presented.

Here's how we would use `QLowEnergyAdvertisingData.`

Construct a `QLowEnergyAdvertisingData` object:

```cpp
QLowEnergyAdvertisingData *leAdd = new QLowEnergyAdvertisingData;
```

Set `Discoverability` options:

```cpp
leAdd->setDiscoverability(
QLowEnergyAdvertisingData::DiscoverabilityGeneral);
```

Set a `Name` for our service:

```cpp
leAdd->setLocalName("SensorServer");
```

Add a list of services we are interested it:

```cpp
QList<QBluetoothUuid> servicesList 
<< QBluetoothUuid::EnvironmentalSensing;
leAdd->setServices(servicesList);
```

The source code can be found on the Git repository under the `Chapter06-1` directory, in the `cp6` branch.

We need to create some characteristic data now. Let's create a `Characteristic` that handles temperature, so we set its `uuid` to `TemperatureMearurement`. We need to also let it be configurable for notifications.

# QLowEnergyCharacteristicData

`QLowEnergyCharacteristicData` represents a **Generic Attribute Profile** (**GATT**) characteristic, which defines a single data point in the Bluetooth transfer. You use it to set up service data:

```cpp
QLowEnergyCharacteristicData chData;
chData.setUuid(QBluetoothUuid::TemperatureMeasurement);
chData.setValue(QByteArray(2,0));
chData.setProperties(QLowEnergyCharacteristic::Notify);
const QLowEnergyDescriptorData descriptorData(QBluetoothUuid::ClientCharacteristicConfiguration, QByteArray(2, 0));
chData.addDescriptor(descriptorData);
```

# QLowEnergyServiceData

Here, we set up the `Temperature` service data as a `Primary` service, and add `Characteristic` to `service`:

```cpp
QLowEnergyServiceData serviceData;
serviceData.setUuid(QBluetoothUuid::Temperature);
serviceData.setType(QLowEnergyServiceData::ServiceTypePrimary);
serviceData.addCharacteristic(chData);
```

Now, let's supply the temperature data. We construct `QLowEnergyCharacteristic` with the `TemperatureMeasurement` type, and supply to it some data. The first bit specifies that we are supplying the `temperature` unit in Celsius:

```cpp
QLowEnergyCharacteristic characteristic = service->characteristic(QLowEnergyCharacteristic::TemperatureMeasurement);
quint8 temperature = 35;

QByteArray currentTempValue;
value.append(char(0));
value.append(char(temperature));
service->writeCharacteristic(characteristic, currentTempValue);
```

We are all set up now, and all we need is to start `Advertising` to listen for connections:

```cpp
controller->startAdvertising(QLowEnergyAdvertisingParameters(), leAdd, leAdd);
```

# Discovery and Pair-ity – search and connect for BLE devices

The first thing you need to do is search for devices, which is called discovery. It entails putting the Bluetooth device into search, or discovery mode. You then receive a list of devices address with which you can connect or pair to be able to access and share data.

Let's look at how that is done in Qt using `QBluetoothDeviceDiscoveryAgent`.

# QBluetoothDeviceDiscoveryAgent

The `QBluetoothDeviceDiscoveryAgent` class is responsible for the device discovery search. It will emit the `deviceDiscovered` signal when any Bluetooth is found:

```cpp
QBluetoothDeviceDiscoveryAgent *discoveryAgent = new QBluetoothDeviceDiscoveryAgent(this); connect(discoveryAgent, SIGNAL(deviceDiscovered(QBluetoothDeviceInfo)),
          this, SLOT(newDevice(QBluetoothDeviceInfo)));
discoveryAgent->start(QBluetoothDeviceDiscoveryAgent::LowEnergyMethod));
```

The source code can be found on the Git repository under the `Chapter06-1a` directory, in the `cp6` branch.

The call to `start()` will initiate the discovery process. The `QBluetoothDeviceDiscoveryAgent::LowEnergyMethod` argument will set a filter to only discover `LowEnergy` devices. Once you find the device you want, you can call `stop()` to stop the device search.

You can wait for errors by connecting to the error (`QBluetoothDeviceDiscoveryAgent::Error error`) signal.

The `error` signal in the `QBluetoothDeviceDiscoveryAgent` class is overloaded, so special care needs to happen in order to connect to the signal. Qt provides `QOverload` and can be implemented like this:

```cpp
connect(discoveryAgent, QOverload<QBluetoothDeviceDiscoveryAgent::Error>::of(&QBluetoothDeviceDiscoveryAgent::error), this, &SomeClass::deviceDiscoveryError);
```

If you would rather get a list of devices all at one time, connect to the `Finished` signal and use the `discoveryDevices()` call, which returns `QList <QBluetoothDeviceInfo>`:

![](img/fa9c6ddf-5c99-47b2-a1af-e024d5852294.png)

You might want to check for the remote devices pairing status, so call `pairingStatus` of `QLocalBluetoothDevice`.

You can pair with a device by then calling the `requestPairing` function of `QBluetoothLocalDevice`, with `QBluetoothAddress` of the remote Bluetooth device:

```cpp
SomeClass::newDevice(const QBluetoothDeviceInfo &info)
{
    QBluetoothLocalDevice::Pairing pairingStatus = localDevice->pairingStatus(info.address()); 
    if (pairingStatus == QBluetoothLocalDevice::Unpaired) {
        QMessageBox msgBox; 
        msgBox.setText("Bluetooth Pairing."); 
        msgBox.setInformativeText("Do you want to pair with device: " + item->data(Qt::UserRole).toString()); 
        msgBox.setStandardButtons(QMessageBox::Ok | QMessageBox::Cancel); 
        msgBox.setDefaultButton(QMessageBox::Cancel); 
        int ret = msgBox.exec(); 
        if (ret == QMessageBox::Ok) { 
            qDebug() << Q_FUNC_INFO << "Pairing..."; 
            localDevice->requestPairing(address, QBluetoothLocalDevice::Paired); 
     } 

    }
}
```

Our example app asks to pair the device before we execute the `requestPairing` procedure:

![](img/02444217-bc7a-4b09-84bd-a389e3f211f8.png)

You can then call `requestPairing` on `QBluetoothLocalDevice` with the `QBluetoothAddress` of the device you wish to pair with. Let's take a look at `QBluetoothLocalDevice`

#  QBluetoothLocalDevice

`QBluetoothLocalDevice` represents the Bluetooth on your device. You use this class to initiate pairing to another device, but also to handle pairing requests from remote Bluetooth devices. It has a few signals to help with that:

*   `pairingDisplayConfirmation`: This is a signal the remote device requests to show user a PIN and ask whether it is the same on both devices. You must call `pairingConfirmation` with `true` or `false` on `QBluetoothLocalDevice`.
*   `pairingDisplayPinCode`: This is a request to enter a PIB.
*   `pairingFinished`: Pairing is completed successfully.

We then connect to these signals, if the user allows it when they click on the OK button:

```cpp
        if (ret == QMessageBox::Ok) { 

            connect(localDevice, &QBluetoothLocalDevice::pairingDisplayPinCode, this, &MainWindow::displayPin);     
            connect(localDevice, &QBluetoothLocalDevice::pairingDisplayConfirmation, this, &MainWindow::displayConfirmation);     
            connect(localDevice, &QBluetoothLocalDevice::pairingFinished, this, &MainWindow::pairingFinished);     
            connect(localDevice, &QBluetoothLocalDevice::error, this, &MainWindow::pairingError);

            localDevice->requestPairing(address, QBluetoothLocalDevice::Paired); 
        }
```

When the remote device only needs a PIN confirmation, the `pairingDisplayConfirmation` signal is called:

```cpp
SomeClass::displayConfirmation(const QBluetoothAddress &address, const QString &pin)
{
    QMessageBox msgBox; 
    msgBox.setText("Confirm pin"); 
    msgBox.setInformativeText("Confirm the pin is the same as on the device.");
    msgBox.setStandardButtons(QMessageBox::Ok | QMessageBox::Cancel);
    msgBox.setDefaultButton(QMessageBox::Cancel);
    int ret = msgBox.exec(); 
    if (ret == QMessageBox::Ok) {
        localDevice->pairingConfirmed(true);
     } else {
        localDevice->pairingConfirmed(false);
    }
}
```

When the remote device needs user to enter a PIN, the `pairingDisplayPinCode` signal is called with a PIN to be displayed and entered on the remote device:

```cpp
SomeClass::displayPin(const QBluetoothAddress &address, const QString &pin) {
{
    QMessageBox msgBox; 
    msgBox.setText(pin);
    msgBox.setInformativeText("Enter pin on remote device"); 
    msgBox.setStandardButtons(QMessageBox::Ok); 
    msgBox.exec(); 
}
```

On the other side, to receive pairing, you need to put `QBluetoothLocalDevice` into the `Discoverable` mode:

```cpp
   localDevice->setHostMode(QBluetoothLocalDevice::HostDiscoverable);
```

The device can then be seen by other devices that are in the Bluetooth `Discovery` mode.

# Specifying and getting client data

Once you have connected to a BLE device peripheral, you need to discover its characteristics to be able to read and write them. You do that by using `QLowEnergyController`. Let's take a look at what `QLowEnergyController` is.

# QLowEnergyController

`QLowEnergyController` is the central place to access BLE devices both local and remote.

The local `QLowEnergyController` can be created by using the static `QLowEnergyController::createPeripheral(QObject *parent)` function.

Creating a `QLowEnergyController` object that represents the remote device is done by calling the static class `QLowEnergyController::createCentral` using the `QBluetoothDeviceInfo` object that you receive when you discover remote devices.

The `QLowEnergyController` object has several signals:

*   `discoveryFinished`
*   `serviceDiscovered`
*   `connected`
*   `disconnected`

Connect to the `connected` signal and start connecting by calling `connectToDevice()`:

```cpp
SomeClass::newDevice(const QBluetoothDeviceInfo &device) 
{
    QLowEnergyController *controller = new QLowEnergyController(device.address());
    connect(controller, &QLowEnergyController::connected, this, &SomeClass::controllerConnected);

    controller->connectToDevice();
}

SomeClass::controllerConnected()
{
    QLowEnergyController *controller = qobject_cast<QLowEnergyController *>(sender());
    if (controller) {
        connect(controller, &QLowEnergyController::serviceDiscovered, this, &SomeClass::newServiceFound);
        controller->discoverServices();
}
```

Once the device is connected, it's time to discover its services, so we connect to the `serviceDiscovered` signal and initiate the service discovery by calling `discoverServices()`.

# QLowEnergyService

You can also connect to the `discoveryFinished()` signal, which returns a list of discovered services by calling `services()`. With either of these, you will get the `QBluetoothUuid` that belongs to that service, with which you can then create a `QLowEnergyService` object:

```cpp
SomeClass::newServiceFound(const QBluetoothUuid &gatt)
{
    QLowEnergyController *controller = qobject_cast<QLowEnergyController *>(sender());
    QLowEnergyService *myLEService = controller->createServiceObject(gatt, this);
}
```

We now have a `QLowEnergyService` object, which gives us details about it. We can only read its service details when its state becomes `ServiceDiscovered`, so now call the `discoverDetails()` function of the service to start the discovery process:

```cpp
   QLowEnergyService *myLEService = controller->createServiceObject(gatt, this);
    connect(myLEService, &QLowEnergyService::stateChanged, this, &SomeClass::serviceStateChanged);
    myLEService->discoverDetails();
```

Let's have a look at `QLowEnergyCharacteristic`.

# QLowEnergyCharacteristic

Once the service details or `characteristics` are discovered, we can perform actions with `QLowEnergyCharacteristic`, such as enabling notifications:

```cpp
void SomeClass::serviceStateChanged(QLowEnergyService::ServiceState state))
{
    if (state != QLowEnergyService::ServiceDiscovered)
        return;
    QLowEnergyService *myLEService = qobject_cast<QLowEnergyService *>(sender());
    QList <QLowEnergyCharacteristic> characteristics = myLEService->characteristics();

}
```

Using `QLowEnergyCharacteristic`, we can get a `QLowEnergyDescriptor` that we use to enable or disable notifications.

Sometimes, a `characteristic` on the remote device needs to be written to as well, such as enabling a specific sensor. In this case, you need to use the `writeCharacteristic` function of the service with `characteristic` as the first argument and the value to be written as the second:

```cpp
QLowEnergyCharacteristic *movementCharacteristic = myLEService->characteristic(someUuid);
myLEService->writeCharacteristic(movementCharacteristic, QLowEnergyCharacteristic::Read);
```

Writing to `QLowEnergyDescriptor` is just as easy; let's take a look.

# QLowEnergyDescriptor

From the Bluetooth specifications, a descriptor is defined as attributes that describe a characteristic value. It contains additional information about a characteristic. `QLowEnergyDescriptor` encapsulates a GATT descriptor. Descriptors and characteristics can have notifications when changes happen.

To enable notifications, we might need to write a value to the descriptor. Here are some possible values:

| **GATT term** | **Description** | **Value** | **Qt constant** |
| Broadcast | Permits broadcast | `0x01`  | `QLowEnergyCharacteristic::Broadcasting` |
| Read | Permits reading  | `0x02`  | `QLowEnergyCharacteristic::Read` |
| Write without response | Permits writing with any response  | `0x04` | `QLowEnergyCharacteristic::WriteNoResponse` |
| Write | Permits writing with a response  | `0x08` | `QLowEnergyCharacteristic::Write` |
| Notify | Permits notifications  | `0x10` | `QLowEnergyCharacteristic::Notify` |
| Indicate | Permits notification with client confirmation required | `0x20` | `QLowEnergyCharacteristic::Indicate` |
| Authenticated signed writes | Permits signed writes  | `0x40` | `QLowEnergyCharacteristic::WriteSigned` |
| Extended properties | Queued writes and writable auxiliaries  | `0x80` | `QLowEnergyCharacteristic::ExtendedProperty` |

The difference between notifications and indications is that, with indications, the server requires the client to confirm that it has received the message, whereas with a notification, the server doesn't care whether the client receives it.

Qt does not currently have support to use authenticated signed writes (`0x40`) with Qt, nor does it have support to use indications (`0x20`).

We want to be notified when the characteristic values change. To enable this, we need to write a value of `0x10` or `QLowEnergyCharacteristic::Notify` to `descriptor`:

```cpp
for ( const QLowEnergyCharacteristic character :  characteristics) {
    QLowEnergyDescriptor descriptor = character.descriptor(QBluetoothUuid::ClientCharacteristicConfiguration);
    connect(myLEService, &QLowEnergyService::characteristicChanged, this, &SomeClass::characteristicUpdated);
    myLEService->writeDescriptor(descriptor, QByteArrayLiteral("\x01\x00"));
}
```

Or we can use the predefined `QLowEnergyCharacteristic::Notify`, like so:

```cpp
myLEService->writeDescriptor(descriptor, QLowEnergyCharacteristic::Notify));
```

Now, we can finally get values out of our Bluetooth LE device:

```cpp
void SomeClass::characteristicUpdated(const QLowEnergyCharacteristic &ch, const QByteArray &value)
{
    qWarning() << ch.name() << "value changed!" << value;
}
```

# Bluetooth QML

There are Bluetooth QML components you can use as a client to scan and connect to Bluetooth devices as well. They are simple but functional.

The source code can be found on the Git repository under the `Chapter06-2` directory, in the `cp6` branch.

1.  Add the `bluetooth` module to your `pro` file:

```cpp
QT += bluetooth
```

2.  In your `qml` file, use the `QtBluetooth` import:

```cpp
import QtBluetooth 5.12
```

The most important element is `BluetoothDiscoveryModel`. 

# BluetoothDiscoveryModel

`BluetoothDiscoveryModel` provides a data model of available Bluetooth devices nearby. You can use it in various model-based Qt Quick components, such as `GridView`, `ListView`, and `PathView`. Setting the `discoveryMode` property tells the local Bluetooth device the level of service discovery, which is one of the following:

*   `FullServiceDiscovery`: Discovers all services of all devices
*   `MinimalServiceDiscovery`: Minimal discovery only includes device and UUID information
*   `DeviceDiscovery`: Discovers only devices and no services

The discovery process will take various amounts of time according to the number of services that need to be discovered. To speed up the discovery of a specific device, you can set the `discoveryMode` property to `BluetoothDiscoveryModel.DeviceDiscovery`, which will allow you to discover the target device address. In the following example, I have commented out the device's target Bluetooth address so it will at least show some devices when you run it:

```cpp
BluetoothDiscoveryModel {
    id: discoveryModel
    discoveryMode: BluetoothDiscoveryModel.DeviceDiscovery
    onDeviceDiscovered: {
        if (/*device == "01:01:01:01:01:01" && */ discoveryMode == BluetoothDiscoveryModel.DeviceDiscovery) {
            discoveryModel.running = false
            discoveryModel.discoveryMode = BluetoothDiscoveryModel.FullServiceDiscovery
            discoveryModel.remoteAddress = device
            discoveryModel.running = true     
        }
    }   
}
```

To discover all services of all nearby devices, set `discoveryMode` to `BluetoothDiscoveryModel.FullServiceDiscovery`. If you set the `remoteAddress` property with a device address, you can target that one specific device. You will then have to toggle the `running` property off and then on to start a new scan.

We have a basic data model, but we need somewhere to display it. Qt Quick has a few options for viewing model data:

*   `GridView`
*   `ListView`
*   `PathView`

`PathView` is best written with Qt Creator QML designer, as you can visually adjust its path.

Let's choose a `ListView` for simplicity although I really wanted to use `PathView`:

```cpp
ListView {
    id: mainList
    anchors.top: busy.bottom
    anchors.fill: parent
    model: discoveryModel   
}
```

It's not going to show anything without defining `delegate`:

```cpp
delegate: Rectangle {
    id: btDelegate
    width: parent.width
    height: column.height + 10
    focus: true
    Column {
        id: column
        anchors.horizontalCenter: parent.horizontalCenter
        Text {
            id: btText
            text: deviceName ? deviceName : name
            font.pointSize: 14
        }
    }
}
```

Scanning for devices can take a while to complete sometimes, so I want to add a busy indicator. Qt Quick Control 2 has `BusyIndicator`:

```cpp
BusyIndicator {
    id: busy
    width: mainWindow.width *.6
    anchors.horizontalCenter: parent.horizontalCenter
    anchors.top: mainWindow.top
    height: mainWindow.height / 8
    running: discoveryModel.running
}
```

When you discover remote services, you will get a `BluetoothService` object.

# BluetoothService

When you specify `BluetoothDiscoveryModel.FullServiceDiscovery` for a discovery scan and when `BluetoothDiscoveryModel` locates a new service, the `serviceDiscovered` signal will be emitted. When we connect to that signal, we will receive the `BluetoothService` object in the slot.

We can the get the **universal unique identifier** (**uuid**), device and service name, service description, and other details. You can use this `BluetoothService` to connect to `BluetoothSocket`.

# BluetoothSocket

The `BluetoothSocket` component can be used to send and receive `String` messages.

To implement this component, at it's simplest would be the following:

```cpp

BluetoothSocket {
    id: btSocket
}
```

`BluetoothSocket` does not handle binary data. For that, you will have to use the C++ `QBluetoothSocket` class.

In `BluetoothDiscoveryModel`, handle the `serviceDiscovered` signal. You will get a `BluetoothService` object named `service`. You can then set `Socket` to use the service with the `setService` method:

```cpp

onServiceDiscovered {
    if (service.serviceName == "Magical Service")
       btSocket.setService(service)

}
```

First, you might want to handle the `stateChanged` signals:

```cpp
onSocketStateChanged: {
 switch (socketState) {
 case BluetoothSocket.Unconnected:
 case BluetoothSocket.NoServiceSet:
 break;
 case BluetoothSocket.Connected:
 console.log("Connected");
 break;
 case BluetoothSocket.Connecting:
 console.log("Connecting...");
 break;
 case BluetoothSocket.ServiceLookup:
 console.log("Looking up Service");
 break;
 case BluetoothSocket.Closing:
 console.log("Closing connection");
 break;
 case BluetoothSocket.Listening:
 console.log("Listening for incoming connections");
 break;
 case BluetoothSocket.Bound:
 console.log("Bound to local address")
 break;
 }
 }
```

To connect to the service, write `true` to the `connected` property:

```cpp
btSocket.connected = true
```

Once the `socketState` property is `Connected`, you can transmit a message or string data using the `stringData` property:

```cpp
btSocket.stringData = "Message Ok"
```

Qt Quick offers a simple way to send string messages over Bluetooth.

# Summary

Bluetooth Low Energy is meant to have lower energy requirements for mobile and embedded devices. Qt offers both C++ and QML classes and components to use it. You should now be able to discover and connect to a Bluetooth Low Energy device. 

Advertising GATT services so users and clients can receive and send data was also covered.

In the next chapter, we will go over some of the main components for the **Internet of Things** (**IoT**), such as sensors and automation communication protocols.