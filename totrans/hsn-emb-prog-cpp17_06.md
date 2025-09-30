# Example - Soil Humidity Monitor with Wi-Fi

Keeping indoor plants alive is no small feat. The example project in this chapter will show you how to create a Wi-Fi-enabled soil humidity monitor with actuator options for a pump or similar, like a valve and gravity-fed water tank. Using the built-in web server, we will be able to use its browser-based UI for monitoring the plant health and control system features, or integrate it into a larger system using its HTTP-based REST API.

The topics covered in this chapter are as follows:

*   Programming an ESP8266 microcontroller
*   Connecting sensors and actuators to an ESP8266
*   Implementing an HTTP server on this platform
*   Developing a web-based UI for monitoring and control
*   Integrating the project into a larger network

# Keeping plants happy

To keep plants alive, you need a number of things:

*   Nutrients
*   Light
*   Water

Of these, the first two are usually handled by nutrient-rich soil and putting the plant in a well-lit place, respectively. The main issue with keeping plants alive after satisfying those two points is usually the third point, as this has to be handled on a daily basis.

Here, it's not just a simple matter of keeping the water topped up, but instead of staying within the range where the soil has enough but not too much water. The presence of too much water in the soil affects how much oxygen the plant can absorb via its roots. As a result, with too much water in the soil, the plant will wither and die.

On the other hand, too little water means that the plant cannot take up enough water to compensate for the water that is being evaporated through its leaves, nor can it get the nutrients into its roots. In this case, the plant will also wither and die.

When manually watering plants, we tend to use rough estimates of when the plant will likely need more water, along with a superficial testing of the humidity of the top soil, using our fingers. This tells us little about how much water is actually present around the roots of the plant, far below the upper layer of soil.

To measure the humidity of the soil with more precision, we can use a number of methods:

| **Type** | **Principle** | **Notes** |
| Gypsum block | Resistance—– | Water is absorbed by the gypsum, dissolving some of it, which allows a current to flow between two electrodes. The resistance value indicates soil moisture tension. |
| Tensiometer | Vacuum | A hollow tube has a vacuum meter on one end and a porous tip at the other, allowing water to enter and leave freely. Water getting sucked out of the tube by the soil increases the vacuum sensor readings, indicating that it's harder to extract moisture from the soil for plants (moisture tension). |
| Capacitance probe | **Frequency Domain Reflectometry** (**FDR**) | Uses the dielectric constant between two metal electrodes (in the soil) in an oscillator circuit to measure changes to this constant due to changing moisture levels. Indicates moisture content. |
| Microwave sensor | **Time Domain Reflectometry** (**TDR**) | Measures the time required for a microwave signal to travel to the end of the parallel probes and back, which differs depending on the dielectric constant of the soil. Measures moisture content. |
| ThetaProbe | RF amplitude impedance | A 100 MHz sine wave radio signal is sent among four probes enclosing a soil cylinder. The change in the impedance of the sine wave is used to calculate the water in the soil. |
| Resistance probe | Resistance | This is similar to the gypsum block, except with just the electrodes. Thus, this only measures water presence (and its conductivity) instead of soil moisture tension. |

All of these sensor types come with their own sets of advantages of disadvantages. In the case of the gypsum block and tensiometer, there is a significant amount of maintenance, as the former relies on there being enough of the gypsum remaining to dissolve and not throw off the calibration, whereas in the latter case, it is imperative that the airtight seal remains so as not to let air into the tube. Any gap in this seal would immediately render the vacuum sensor useless.

Another big point is that of cost. While FDR- and TDR-based probes may be quite accurate, they also tend to be very expensive. This usually leads people who just want to experiment with soil moisture sensors to pick either the resistance or capacitance-based sensors. Here, the main disadvantage of the former sensor type becomes obvious in a month or less of usage: corrosion.

With two electrodes suspended in a solution containing ions and a current being applied to one of the electrodes, simple chemistry results in one of the electrodes rapidly corroding (losing material), until it is no longer functional. This also pollutes the soil with metal molecules. The use of an **alternating current** (**AC**) instead of a direct current on a single electrode can reduce the corrosive effect somewhat, but it remains an issue.

Among cheap and still accurate soil moisture sensors, only the capacitance probe ticks all of the boxes. Its accuracy is decent enough for sensible measurements and comparisons (after calibration), it is unaffected by the moisture in the soil, and it does not affect the soil in any manner either.

To actually get water to the plant, we need to have a way to get just the right amount to it. Here, it's mostly the scale of the system that determines the choice of water delivery. For watering an entire field, we could use an impeller-based pump, capable of delivering many liters of water per minute.

For a single plant, we would need to be able to deliver in the order of a few hundred milliliters per minute at most. Here, something such as a peristaltic pump would be pretty much ideal. This is the kind of pump you would also use in laboratories and medical applications where you have to provide a small amount of fluid with high accuracy.

# Our solution

To keep things simple, we will just be building something that can take care of a single plant. This will provide us with the most flexibility in terms of placement, as we would just have a single system next to each plant, no matter whether it's on a windowsill, table, or terrace somewhere.

In addition to measuring the soil moisture level, we would also want to be able to have the system automatically water the plant at set trigger levels and for us to be able to monitor this process. This requires some kind of network access, preferably wireless so that we don't have to run any more cables than the power connector.

This makes the ESP8266 MCU very attractive, with the NodeMCU development board an attractive target for developing and debugging the system. We'd hook up a soil moisture sensor to it, along with a peristaltic pump.

By connecting to the ESP8266 system's IP address using a web browser, we would see the current status of the system, with the soil humidity level and optionally much more. Configuring the system and more would be done over the commonly used, compact binary MQTT protocol, with the system also publishing the current system status so that we can read it into a database for display and analysis.

This way, we can also write a backend service later on that combines many of such nodes into a coherent system with central control and management. This is something that we will actually look at in great detail in [Chapter 9](d165297b-8be7-44f5-90b5-53b3bcb51d3b.xhtml), *Example - Building Monitoring and Control*.

# The hardware

Our ideal solution would have the most accurate sensor, without breaking the bank. This means that we pretty much have to use a capacitance sensor, as we saw earlier in this chapter. These sensors can be obtained as capacitive soil moisture sensors for little more than a few euros or dollars for a simple 555 timer IC-based design such as these:

![](img/9473a55a-2149-4762-9a3d-1580277a0462.png)

You would simply stick these into the soil up to the point where the circuitry begins, then connect it with a power source as well as the connection to the analog to digital converter of the MCU.

Most peristaltic pumps one can purchase require 12V. This means that we need to either have a power supply that can provide both 5V and 12V, or use a so-called boost converter to convert the 5V to 12V. Either way, we would also need to have some method to turn the pump on or off. With the boost converter, we can use its *enable* pin to turn its output on or off using a GPIO pin on our MCU.

For prototyping, we can use one of these common 5V to 12V boost converter modules that use an ME2149 step-up switching regulator:

![](img/97a91881-f1ca-4bbf-a786-fd122d1bba40.png)

These do not have the enable pin broken out in any way, but we can easily solder on a wire to the pin in question:

![](img/d992fb6a-fb74-480f-9d5c-4cec48efd14b.png)

This boost-converter module's outputs are then connected to the peristaltic pump:

![](img/9e0cb710-095e-4f43-b2ba-b9c0145085ea.png)

Here, we need to get some tubing of the right diameter to connect it to the water reservoir and the plant. The pump itself will rotate either direction. As it consists of essentially a set of rollers on the section of internal tubing, which push the liquid inside one way, either side of the pump can be the input or output.

Be sure to test the flow direction beforehand with two containers and some water, and mark the direction on the pump casing, along with the positive and negative terminal connections used.

In addition to these components, we also want to have an RGB LED connected for some signaling and just for looks. For this, we will use the **APA102** RGB LED module, which connects to the ESP8266 over the SPI bus:

![](img/080f68db-2083-4eff-b67e-536e6aee587c.png)

We can use a single power supply that can be provide 5V with 1A or more, as well as cope with the sudden power draw from the boost converter every time that the pump is activated.

The whole system would look like this:

![](img/541c3d8a-ca18-49c7-96cc-1e863fb62a79.png)

# The firmware

For this project, we will be implementing a module for the same firmware that we will be using in [Chapter 9](d165297b-8be7-44f5-90b5-53b3bcb51d3b.xhtml), *Example - Building Monitoring and Control*. Therefore, this chapter will only cover the parts that are unique to this plant-watering module.

Before we can start with the firmware itself, we first have to set up the development environment. This involves the installation of the ESP8266 SDK and the Sming framework.

# Setting up Sming

The Sming-based ESP8266 development environment can be used on Linux, Windows, and macOS. You want to preferably use the development branch of Sming, however, which is where using it on Linux (or in a Linux VM, or Windows 10's **Windows Subsystem for Linux** (**WSL**)) is the easiest way, and definitely recommended. On Linux installing in the `/opt` folder is recommended for consistency with the Sming quick start guide.

This quick start guide for Linux can be found at [https://github.com/SmingHub/Sming/wiki/Linux-Quickstart](https://github.com/SmingHub/Sming/wiki/Linux-Quickstart).

On Linux, we can use the Open SDK for ESP8266, which takes the official Espressif (non-RTOS) SDK, and replaces all the non-open components it can with open source alternatives. This can be installed using this code:

```cpp
    git clone --recursive https://github.com/pfalcon/esp-open-sdk.git
    cd esp-open-sdk
    make VENDOR_SDK=1.5.4 STANDALONE=y  
```

This will get the current source for the Open SDK and compile it, targeting version 1.5.4 of the official SDK. While a 2.0 version of the SDK already exists, some compatibility issues within the Sming framework can remain. Using the 1.5.4 version offers pretty much the same experience while using well-tested code. This will of course change over time, so be sure to check the official Sming documentation for updated instructions.

The `STANDALONE` option means that the SDK will be built as a standalone installation of the SDK and the toolchain, without further dependencies. This is the desired option for use with Sming.

Installing `Sming` is as easy as this:

```cpp
    git clone https://github.com/SmingHub/Sming.git
    cd Sming
    make  
```

This will build the Sming framework. If we are adding new libraries to Sming in its `Libraries` folder, we have to execute the last step again to have a new Sming shared library instance to be built and installed.

For this project, copy the folders in the `libs` folder of the software project for this chapter to the `Sming/Sming/Libraries` folder prior to compiling Sming, or the project code will not compile.

We can also compile Sming with SSL support. This requires us to compile it with the `ENABLE_SSL=1` parameter to Make. This will enable the axTLS-based encryption support throughout the Sming library as it is compiled.

With these steps complete, we just have to install `esptool.py` and `esptool2`. While in the `/opt` folder, execute these commands to obtain esptool:

```cpp
    wget https://github.com/themadinventor/esptool/archive/master.zip
    unzip master.zip
    mv esptool-master esp-open-sdk/esptool  
```

`Esptool.py` is a Python script that allows us to communicate with the SPI ROM that is part of each ESP8266 module. It is the way we will flash the MCU's ROM with our code. This tool is automatically used by Sming:

```cpp
    cd  $ESP_HOME
    git clone https://github.com/raburton/esptool2
    cd esptool2
    make  
```

The `esptool2` utility is an alternative to the set of scripts in the official SDK that turn the linker output into a ROM format that we can write to the ESP8266\. It is called by Sming when we are compiling our application.

Finally, assuming that we installed the SDK and Sming under `/opt`, we can add the following global variables and addition to the system `PATH` variable:

```cpp
    export ESP_HOME=/opt/esp-open-sdk
    export SMING_HOME=/opt/Sming/Sming
    export PATH=$PATH:$ESP_HOME/esptool2
    export PATH=$PATH:$ESP_HOME/xtensa-lx106-elf/bin  
```

The last line adds the toolchain's binaries to the path, which we will need when debugging ESP8266 applications, as will see in [Chapter 7](d8237285-fcb7-4bbc-84f3-e45568598865.xhtml), *Testing Resource-Restricted Platforms*. At this point, we can develop with Sming and create ROM images that we can write to the MCU.

# Plant module code

In this section, we will look at the basic source code for this project, starting with the core module, `OtaCore`, and continuing with the `BaseModule` class, which all firmware modules register with. Finally, we look at the `PlantModule` class itself, which contains the business logic for the project requirements that we discussed in this chapter.

Also of note is that for this project we enabled both the rBoot bootmanager and the rBoot Big Flash options in the project Makefile. What this does is create four 1 MB blocks in the 4 MB of ROM that we have available on our ESP8266 module (which is all ESP-12E/F modules), of which two are used for firmware images and the remaining two for file storage (using the SPIFFS filesystem).

The rBoot bootloader is then written to the beginning of the ROM, so that it will be loaded first on each boot. Of the firmware slots, only one is active at any given time. A handy feature of this setup is that it allows us to easily perform **over-the-air** (**OTA**) updates, by writing the new firmware image to the inactive firmware slot, changing the active slot, and restarting the MCU. If rBoot fails to boot from the new firmware image, it will fall back on the other firmware slot, which is our known working firmware that we performed the OTA update from.

# Makefile-user.mk

In the root of the `project` folder, we find this Makefile. It contains a number of settings that we may want to set to suit our purposes:

| **Name** | **Description** |
| `COM_PORT` | If we always connect the board to the same serial port, we can hardcode it here to save ourselves some typing. |
| `SPI_MODE` | This sets the SPI mode used while flashing the firmware images to the SPI ROM. With `dio` only two data lines (`SD_D0`, `D1`) or four (`SD_D0-3`). Not all SPI ROMs have all four data lines connected. The `qio` mode is faster, but `dio` should always work. |
| `RBOOT_ENABLED` | When set to 1, this enables rBoot bootloader support. We want this enabled. |
| `RBOOT_BIG_FLASH` | With 4 MB of ROM available, we wish to use all of this. Enable this as well. |
| `RBOOT_TWO_ROMS` | This option can be used if we wish to place two firmware images in a single 1 MB ROM chip instead. This applies to some ESP8266 modules and derivatives. |
| `SPI_SIZE` | Here, we set the size of the SPI ROM chip, which should be 4M for this project. |
| `SPIFF_FILES` | The location of the folder containing the files that will be put on the SPIFFS ROM image that will be written to the MCU. |
| `SPIFFS_SIZE` | The size of the SPIFFS ROM image to create. Here, 64 KB is standard, but we could use up to 1 MB if we needed to when using a 4 MB ROM with the `RBOOT_BIG_FLASH` option enabled. |
| `WIFI_SSID` | The SSID of the Wi-Fi network that we wish to connect to. |
| `WIFI_PWD` | The password for the Wi-Fi network. |
| `MQTT_HOST` | The URL or IP address of the MQTT server (broker) to use. |
| `ENABLE_SSL` | Enable this with SSL support compiled into Sming to make the firmware use TLS-encrypted connections with the MQTT broker. |
| `MQTT_PORT` | The port of the MQTT broker. This depends on whether SSL is enabled. |
| `USE_MQTT_PASSWORD` | Set to true if you wish to connect to the MQTT broker with a username and password. |
| `MQTT_USERNAME` | The MQTT broker username, if required. |
| `MQTT_PWD` | The MQTT broker password, if required. |
| `MQTT_PREFIX` | A prefix you can optionally add in front of each MQTT topic used by the firmware, if necessary. It has to end with a slash if not left empty. |
| `OTA_URL` | The hardcoded URL that will be used by the firmware whenever an OTA update is requested. |

Of these, the Wi-Fi, MQTT, and OTA settings are essential, as they will allow the application to connect to the network and MQTT broker, as well as receive firmware updates without having to flash the MCU over its serial interface.

# Main

The main source file and with it the application entry point is pretty uneventful:

```cpp
#include "ota_core.h"
void onInit() {
    // 
}
void init() {
         OtaCore::init(onInit);
 }
```

With the `OtaCore` class containing the main application logic, we merely call its static initialization function while providing a callback function if we wish to execute any further logic after the core class has finished setting up the network, MQTT, and other functionality.

# OtaCore

In this class, we set up all of the basic network functionality for the specific feature modules, in addition to providing utility functions for logging and MQTT functionality. This class also contains the main command processor for commands received over MQTT:

```cpp
#include <user_config.h>
#include <SmingCore/SmingCore.h>
```

These two includes are required to make use of the Sming framework. With them, we include the main headers of the SDK (`user_config.h`) and those of Sming (`SmingCore.h`). This also defines a number of preprocessor statements, such as to use the open source **Light-Weight IP stack** (**LWIP**) and to deal with some issues in the official SDK.

Also of note is the `esp_cplusplus.h` header, which is indirectly included this way. Its source file implements the `new` and `delete` functions, as well as a few handlers for class-related functionality, such as `vtables` when using virtual classes. This enables compatibility with the STL:

```cpp
enum {
          LOG_ERROR = 0,
          LOG_WARNING,
          LOG_INFO,
          LOG_DEBUG,
          LOG_TRACE,
          LOG_XTRACE
 };

 enum ESP8266_pins {
          ESP8266_gpio00 = 0x00001,     // Flash
          ESP8266_gpio01 = 0x00002,     // TXD 0
          ESP8266_gpio02 = 0x00004,     // TXD 1
          ESP8266_gpio03 = 0x00008,     // RXD 0
          ESP8266_gpio04 = 0x00010,     // 
          ESP8266_gpio05 = 0x00020,     // 
          ESP8266_gpio09 = 0x00040,     // SDD2 (QDIO Flash)
          ESP8266_gpio10 = 0x00080,     // SDD3 (QDIO Flash)
          ESP8266_gpio12 = 0x00100,     // HMISO (SDO)
          ESP8266_gpio13 = 0x00200,     // HMOSI (SDI)
          ESP8266_gpio14 = 0x00400,     // SCK
          ESP8266_gpio15 = 0x00800,     // HCS
          ESP8266_gpio16 = 0x01000,     // User, Wake
          ESP8266_mosi = 0x02000,
          ESP8266_miso = 0x04000,
          ESP8266_sclk = 0x08000,
          ESP8266_cs = 0x10000
 };
```

These two enumerations define the logging levels, and the individual GPIO and other pins of the ESP8266 that we may want to use. The values for the ESP8266 pin enumeration correspond to positions in a bitmask:

```cpp
#define SCL_PIN 5
#define SDA_PIN 4
```

Here, we define the fixed pins for the I2C bus. These correspond to GPIO 4 and 5, also known as **D1** and **D2** on NodeMCU boards. The main reason for having these pins predefined is that they are two of the few *safe* pins on the ESP8266.

Many pins of the ESP8266 will change levels during startup before settling, which can cause unwanted behavior with any connected peripherals.

```cpp
typedef void (*topicCallback)(String);
typedef void (*onInitCallback)();
```

We define two function pointers, one to be used by feature modules when they wish to register an MQTT topic, along with a callback function. The other is the callback we saw in the main function.

```cpp

class OtaCore {
         static Timer procTimer;
         static rBootHttpUpdate* otaUpdater;
         static MqttClient* mqtt;
         static String MAC;
         static HashMap<String, topicCallback>* topicCallbacks;
         static HardwareSerial Serial1;
         static String location;
         static String version;
         static int sclPin;
         static int sdaPin;
         static bool i2c_active;
         static bool spi_active;
         static uint32 esp8266_pins;

         static void otaUpdate();
         static void otaUpdate_CallBack(rBootHttpUpdate& update, bool result);
         static void startMqttClient();
         static void checkMQTTDisconnect(TcpClient& client, bool flag);
         static void connectOk(IPAddress ip, IPAddress mask, IPAddress gateway);
         static void connectFail(String ssid, uint8_t ssidLength, uint8_t *bssid,    uint8_t reason);
         static void onMqttReceived(String topic, String message);
         static void updateModules(uint32 input);
         static bool mapGpioToBit(int pin, ESP8266_pins &addr);

public:
         static bool init(onInitCallback cb);
         static bool registerTopic(String topic, topicCallback cb);
         static bool deregisterTopic(String topic);
         static bool publish(String topic, String message, int qos = 1);
         static void log(int level, String msg);
         static String getMAC() { return OtaCore::MAC; }
         static String getLocation() { return OtaCore::location; }
         static bool starti2c();
         static bool startSPI();
         static bool claimPin(ESP8266_pins pin);
         static bool claimPin(int pin);
         static bool releasePin(ESP8266_pins pin);
         static bool releasePin(int pin);
};
```

The class declaration itself gives a good overview of the functionality provided by this class. The first thing we notice is that it is completely static. This ensures that this class's functionality is immediately initialized when the firmware starts, and that it can be accessed globally without having to worry about specific instances.

We can also see the first use of the `uint32` type, which along with other integer types is defined similar to those in the `cstdint` header.

Moving on, here is the implementation:

```cpp
#include <ota_core.h>

#include "base_module.h"

#define SPI_SCLK 14
#define SPI_MOSI 13
#define SPI_MISO 12
#define SPI_CS 15

Timer OtaCore::procTimer;
rBootHttpUpdate* OtaCore::otaUpdater = 0;
MqttClient* OtaCore::mqtt = 0;
String OtaCore::MAC;
HashMap<String, topicCallback>* OtaCore::topicCallbacks = new HashMap<String, topicCallback>();
HardwareSerial OtaCore::Serial1(UART_ID_1); // UART 0 is 'Serial'.
String OtaCore::location;
String OtaCore::version = VERSION;
int OtaCore::sclPin = SCL_PIN; // default.
int OtaCore::sdaPin = SDA_PIN; // default.
bool OtaCore::i2c_active = false;
bool OtaCore::spi_active = false;
uint32 OtaCore::esp8266_pins = 0x0;
```

We include the `BaseModule` class's header here, so that we can call its own initialization function later on after we have finished setting up the basic functionality. The static class members are also initialized here, with a number of default values assigned where relevant.

Of note here is the initializing of a second serial interface object in addition to the default Serial object instance. These correspond to the first (UART0, Serial) and second (UART1, Serial1) UART on the ESP8266.

With older versions of Sming, the SPIFFS-related file functions had trouble with binary data (due to internally assuming null-terminated strings), which is why the following alternative functions were added. Their naming is a slightly inverted version from the original function name to prevent naming collisions.

Since TLS certificates and other binary data files stored on SPIFFS have to be able to be written and read for the firmware to function correctly, this was a necessary compromise.

```cpp
String getFileContent(const String fileName) {
         file_t file = fileOpen(fileName.c_str(), eFO_ReadOnly);

         fileSeek(file, 0, eSO_FileEnd);
         int size = fileTell(file);
         if (size <= 0)    {
                fileClose(file);
                return "";
         }

         fileSeek(file, 0, eSO_FileStart);
         char* buffer = new char[size + 1];
         buffer[size] = 0;
         fileRead(file, buffer, size);
         fileClose(file);
         String res(buffer, size);
         delete[] buffer;
         return res;
}
```

This function reads the entire contents of the specified file into a `String` instance that is returned.

```cpp
void setFileContent(const String &fileName, const String &content) {
          file_t file = fileOpen(fileName.c_str(),                                                   eFO_CreateNewAlways | eFO_WriteOnly);
          fileWrite(file, content.c_str(), content.length());
          fileClose(file);
 }
```

This function replaces the existing content in a file with the new data in the provided `String` instance.

```cpp
bool readIntoFileBuffer(const String filename, char* &buffer, unsigned int &size) {
         file_t file = fileOpen(filename.c_str(), eFO_ReadOnly);

         fileSeek(file, 0, eSO_FileEnd);
         size = fileTell(file);
         if (size == 0)    {
                fileClose(file);
                return true;
         }

         fileSeek(file, 0, eSO_FileStart);
         buffer = new char[size + 1];
         buffer[size] = 0;
         fileRead(file, buffer, size);
         fileClose(file);
         return true;
}
```

This function is similar to `getFileContent()`, but returns a simple character buffer instead of a `String` instance. It's mostly used for reading in the certificate data, which is passed into a C-based TLS library (called axTLS), where converting to a `String` instance would be wasteful with the copying involved, especially where certificates can be a few KB in size.

Next is the initialization function for this class:

```cpp
bool OtaCore::init(onInitCallback cb) {
         Serial.begin(9600);

         Serial1.begin(SERIAL_BAUD_RATE); 
         Serial1.systemDebugOutput(true);
```

We first initialize the two UARTs (serial interfaces) in the NodeMCU. Although officially there are two UARTs in the ESP8266, the second one consists only out of a TX output line (GPIO 2, by default). Because of this, we want to keep the first UART free for applications requiring a full serial line, such as some sensors.

The first UART (`Serial`) is thus initialized so that we can later use it with feature modules, while the second UART (`Serial1`) is initialized to the default baud rate of 115,200, along with the system's debug output (WiFi/IP stack, and so on) being directed to this serial output as well. This second serial interface will thus be used solely for logging output.

```cpp
         BaseModule::init(); 
```

Next, the `BaseModule` static class is initialized as well. This causes all feature modules active in this firmware to be registered, allowing them to be activated later on.

```cpp
         int slot = rboot_get_current_rom();
         u32_t offset;
         if (slot == 0) { offset = 0x100000; }
         else { offset = 0x300000; }
         spiffs_mount_manual(offset, 65536);
```

Automatically mounting the SPIFFS filesystem while using the rBoot bootloader did not work with older releases of Sming, which is why we are doing it manually here. To do this, we get the current firmware slot from rBoot, using which we can pick the proper offset, either at the start of the second megabyte in the ROM, or of the fourth megabyte.

With the offset determined, we use the SPIFFS manual-mounting function with our offset and the size of the SPIFFS section. We are now able to read and write to our storage.

```cpp

          Serial1.printf("\r\nSDK: v%s\r\n", system_get_sdk_version());
     Serial1.printf("Free Heap: %d\r\n", system_get_free_heap_size());
     Serial1.printf("CPU Frequency: %d MHz\r\n", system_get_cpu_freq());
     Serial1.printf("System Chip ID: %x\r\n", system_get_chip_id());
     Serial1.printf("SPI Flash ID: %x\r\n", spi_flash_get_id());
```

Next, we print out a few system details to the serial debug output. This includes the ESP8266 SDK version we compiled against, the current free heap size, CPU frequency, the MCU ID (32-bit ID), and the ID of the SPI ROM chip.

```cpp
         mqtt = new MqttClient(MQTT_HOST, MQTT_PORT, onMqttReceived);
```

We create a new MQTT client on the heap, providing the callback that will be called when we receive a new message. The MQTT broker host and port are filled in by the preprocessor from the details we added in the user Makefile for the project.

```cpp

         Serial1.printf("\r\nCurrently running rom %d.\r\n", slot);

         WifiStation.enable(true);
         WifiStation.config(WIFI_SSID, WIFI_PWD);
         WifiStation.connect();
         WifiAccessPoint.enable(false);

        WifiEvents.onStationGotIP(OtaCore::connectOk);
        WifiEvents.onStationDisconnect(OtaCore::connectFail);

          (*cb)();
}
```

As the final steps in the initialization, we output the current firmware slot that we are running from, then enable the Wi-Fi client while disabling the **wireless access point** (**WAP**) functionality. The WiFi client is told to connect to the WiFi SSID with the credentials that we specified previously in the Makefile.

Finally, we define the handlers for a successful WiFi connection and for a failed connection attempt, before calling the callback function we were provided with as a parameter.

After an OTA update of the firmware, the following callback function will be called:

```cpp

void OtaCore::otaUpdate_CallBack(rBootHttpUpdate& update, bool result) {
         OtaCore::log(LOG_INFO, "In OTA callback...");
         if (result == true) { // success
               uint8 slot = rboot_get_current_rom();
               if (slot == 0) { slot = 1; } else { slot = 0; }

               Serial1.printf("Firmware updated, rebooting to ROM slot %d...\r\n",                                                                                                                        slot);
               OtaCore::log(LOG_INFO, "Firmware updated, restarting...");
               rboot_set_current_rom(slot);
               System.restart();
         } 
         else {
               OtaCore::log(LOG_ERROR, "Firmware update failed.");
         }
}
```

In this callback, we change the active ROM slot if the OTA update was successful, followed by a reboot of the system. Otherwise, we simply log an error and do not restart.

Next are a few MQTT-related functions:

```cpp
bool OtaCore::registerTopic(String topic, topicCallback cb) {
         OtaCore::mqtt->subscribe(topic);
         (*topicCallbacks)[topic] = cb;
         return true;
}

bool OtaCore::deregisterTopic(String topic) {
         OtaCore::mqtt->unsubscribe(topic);
         if (topicCallbacks->contains(topic)) {
               topicCallbacks->remove(topic);
         }

         return true;
}
```

These two functions allow feature modules to respectively register and deregister an MQTT topic along with a callback. The MQTT broker is called with a subscription or unsubscribe request and the `HashMap` instance is updated accordingly:

```cpp
bool OtaCore::publish(String topic, String message, int qos /* = 1 */) {
         OtaCore::mqtt->publishWithQoS(topic, message, qos);
         return true;
}
```

Any feature modules can publish an MQTT message on any topic using this function. The **Quality of Service** (**QoS**) parameter determines the publish mode. By default, messages are published in *retain* mode, meaning that the broker will retain the last published message for a particular topic.

The entry point for the OTA update functionality is found in the following function:

```cpp
void OtaCore::otaUpdate() {
         OtaCore::log(LOG_INFO, "Updating firmware from URL: " + String(OTA_URL));

         if (otaUpdater) { delete otaUpdater; }
         otaUpdater = new rBootHttpUpdate();

         rboot_config bootconf = rboot_get_config();
         uint8 slot = bootconf.current_rom;
         if (slot == 0) { slot = 1; } else { slot = 0; }

         otaUpdater->addItem(bootconf.roms[slot], OTA_URL + MAC);

         otaUpdater->setCallback(OtaCore::otaUpdate_CallBack);
         otaUpdater->start();
}
```

For an OTA update, we need to create a clean `rBootHttpUpdate` instance. We then need to configure this instance with the details of the current firmware slot, for which we obtain the configuration from rBoot and with it the current firmware slot number. This we use to give the number of the other firmware slot to the OTA updater.

Here, we only configure it to update the firmware slot, but we could also update the SPIFFS section for the other firmware slot as well this way. The firmware will be fetched over HTTP from the fixed URL we set before. The ESP8266's MAC address is affixed to the end of it as a unique query string parameter so that the update server knows which firmware image fits this system.

After setting the `callback` function that we looked at earlier, we start the update:

```cpp
void OtaCore::checkMQTTDisconnect(TcpClient& client, bool flag) {
         if (flag == true) { Serial1.println("MQTT Broker disconnected."); }
         else { 
               String tHost = MQTT_HOST;
               Serial1.println("MQTT Broker " + tHost + " unreachable."); }

         procTimer.initializeMs(2 * 1000, OtaCore::startMqttClient).start();
}
```

Here, we define the MQTT disconnection handler. It is called whenever the connection with the MQTT broker fails so that we can try reconnecting after a two-second delay.

The flag parameter is set to true if we previously were connected, and false if the initial MQTT broker connection failed (no network access, wrong address, and so on).

Next is the function to configure and start the MQTT client:

```cpp
void OtaCore::startMqttClient() {
         procTimer.stop();
         if (!mqtt->setWill("last/will",                                 "The connection from this device is lost:(",    1, true)) {
               debugf("Unable to set the last will and testament. Most probably there is not enough memory on the device.");
         }
```

We stop the procTimer timer if it's running in case we were being called from a reconnect timer. Next, we set the **last will and testament** (**LWT**) for this device, which allows us to set a message that the MQTT broker will publish when it loses the connection with the client (us).

Next, we define three different execution paths, only one of which will be compiled, depending on whether we are using TLS (SSL), a username/password login, or anonymous access:

```cpp
#ifdef ENABLE_SSL
         mqtt->connect(MAC, MQTT_USERNAME, MQTT_PWD, true);
         mqtt->addSslOptions(SSL_SERVER_VERIFY_LATER);

       Serial1.printf("Free Heap: %d\r\n", system_get_free_heap_size());

         if (!fileExist("esp8266.client.crt.binary")) {
               Serial1.println("SSL CRT file is missing: esp8266.client.crt.binary.");
               return;
         }
         else if (!fileExist("esp8266.client.key.binary")) {
               Serial1.println("SSL key file is missing: esp8266.client.key.binary.");
               return;
         }

         unsigned int crtLength, keyLength;
         char* crtFile;
         char* keyFile;
         readIntoFileBuffer("esp8266.client.crt.binary", crtFile, crtLength);
         readIntoFileBuffer("esp8266.client.key.binary", keyFile, keyLength);

         Serial1.printf("keyLength: %d, crtLength: %d.\n", keyLength, crtLength);
         Serial1.printf("Free Heap: %d\r\n", system_get_free_heap_size());

         if (crtLength < 1 || keyLength < 1) {
               Serial1.println("Failed to open certificate and/or key file.");
               return;
         }

         mqtt->setSslClientKeyCert((const uint8_t*) keyFile, keyLength,
                                                (const uint8_t*) crtFile, crtLength, 0, true);
         delete[] keyFile;
         delete[] crtFile;

    Serial1.printf("Free Heap: %d\r\n", system_get_free_heap_size());
```

If we are using TLS certificates, we establish a connection with the MQTT broker, using our `MAC` as client identifier, then enable the SSL option for the connection. The available heap space is printed to the serial logging output for debugging purposes. Usually, at this point, we should have around 25 KB of RAM left, which is sufficient for holding the certificate and key in memory, along with the RX and TX buffers for the TLS handshake if the latter are configured on the SSL endpoint to be an acceptable size using the SSL fragment size option. We will look at this in more detail in [Chapter 9](d165297b-8be7-44f5-90b5-53b3bcb51d3b.xhtml), *Example - Building Management and Control*.

Next, we read the DER-encoded (binary) certificate and key files from SPIFFS. These files have a fixed name. For each file, we print out the file size, along with the current free heap size. If either file size is zero bytes, we consider the read attempt to have failed and we abort the connection attempt.

Otherwise, we use the key and certificate data with the MQTT connection, which should lead to a successful handshake and establishing an encrypted connection with the MQTT broker.

After deleting the key and certificate file data, we print out the free heap size to allow us to check that the cleanup was successful:

```cpp
#elif defined USE_MQTT_PASSWORD
          mqtt->connect(MAC, MQTT_USERNAME, MQTT_PWD);
```

When using an MQTT username and password to log in to the broker, we just need to call the previous function on the MQTT client instance, providing our MAC as client identifier along with the username and password:

```cpp
#else
         mqtt->connect(MAC);
#endif
```

To connect anonymously, we set up a connection with the broker and pass our `MAC` as the client identifier:

```cpp
         mqtt->setCompleteDelegate(checkMQTTDisconnect);

         mqtt->subscribe(MQTT_PREFIX"upgrade");
         mqtt->subscribe(MQTT_PREFIX"presence/tell");
         mqtt->subscribe(MQTT_PREFIX"presence/ping");
         mqtt->subscribe(MQTT_PREFIX"presence/restart/#");
         mqtt->subscribe(MQTT_PREFIX"cc/" + MAC);

         delay(100);

         mqtt->publish(MQTT_PREFIX"cc/config", MAC);
}
```

Here, we first set the MQTT disconnect handler. Then, we subscribe to a number of topics that we wish to respond to. These all relate to management functionality for this firmware, allowing the system to be queried and configured over MQTT.

After subscribing, we briefly (100 ms) wait to give the broker some time to process these subscriptions before we publish on the central notification topic, using our `MAC` to let any interested clients and servers know that this system just came online.

Next are the WiFi connection handlers:

```cpp
void OtaCore::connectOk(IPAddress ip, IPAddress mask, IPAddress gateway) {
          Serial1.println("I'm CONNECTED. IP: " + ip.toString());

          MAC = WifiStation.getMAC();
          Serial1.printf("MAC: %s.\n", MAC.c_str());

          if (fileExist("location.txt")) {
                location = getFileContent("location.txt");
          }
          else {
                location = MAC;
          }

          if (fileExist("config.txt")) {
                String configStr = getFileContent("config.txt");
                uint32 config;
                configStr.getBytes((unsigned char*) &config, sizeof(uint32), 0);
                updateModules(config);
          }

          startMqttClient();
 }
```

This handler is called when we have successfully connected to the configured WiFi network using the provided credentials. After connecting, we keep a copy of our `MAC` in memory as our unique ID.

This firmware also supports specifying a user-defined string as our location or similar identifier. If one has been defined before, we load it from SPIFFS and use it; otherwise, our location string is simply the `MAC`.

Similarly, we load the 32-bit bitmask that defines the feature module configuration from SPIFFS if it exists. If not, all feature modules are initially left deactivated. Otherwise, we read the bitmask and pass it to the `updateModules()` function so that the relevant modules will be activated:

```cpp
void OtaCore::connectFail(String ssid, uint8_t ssidLength, 
                                                   uint8_t* bssid, uint8_t reason) {
          Serial1.println("I'm NOT CONNECTED. Need help :(");
          debugf("Disconnected from %s. Reason: %d", ssid.c_str(), reason);

          WDT.alive();

          WifiEvents.onStationGotIP(OtaCore::connectOk);
          WifiEvents.onStationDisconnect(OtaCore::connectFail);
 }
```

If connecting to the Wi-Fi network fails, we log this fact, then tell the MCU's watchdog timer that we are still alive to prevent a soft restart before we attempt to connect again.

This finishes all of the initialization functions. Next up are the functions used during normal activity, starting with the MQTT message handler:

```cpp
void OtaCore::onMqttReceived(String topic, String message) {
         Serial1.print(topic);
         Serial1.print(":\n");
         Serial1.println(message);

         log(LOG_DEBUG, topic + " - " + message);

         if (topic == MQTT_PREFIX"upgrade" && message == MAC) {
                otaUpdate();
         }
         else if (topic == MQTT_PREFIX"presence/tell") {
                mqtt->publish(MQTT_PREFIX"presence/response", MAC);
         }
         else if (topic == MQTT_PREFIX"presence/ping") {
               mqtt->publish(MQTT_PREFIX"presence/pong", MAC);
         }
         else if (topic == MQTT_PREFIX"presence/restart" && message == MAC) {
               System.restart();
         }
         else if (topic == MQTT_PREFIX"presence/restart/all") {
               System.restart();
         }
```

We registered this callback when we initially created the MQTT client instance. Every time a topic that we subscribed to receives a new message on the broker, we are notified and this callback receives a string containing the topic and another string containing the actual message (payload).

We can compare the topic with the topics we registered for, and perform the required operation, whether it is to perform an OTA update (if it specifies our `MAC`), respond to a ping request by returning a pong response with our `MAC`, or to restart the system.

The next topic is a more generic maintenance one, allowing one to configure active feature modules, set the location string, and request the current status of the system. The payload format consists out of the command string followed by a semicolon, and then the payload string:

```cpp
   else if (topic == MQTT_PREFIX"cc/" + MAC) {
          int chAt = message.indexOf(';');
          String cmd = message.substring(0, chAt);
          ++chAt;

          String msg(((char*) &message[chAt]), (message.length() - chAt));

          log(LOG_DEBUG, msg);

          Serial1.printf("Command: %s, Message: ", cmd.c_str());
          Serial1.println(msg);
```

We start by extracting the command from the payload string using a simple find and substring approach. We then read in the remaining payload string, taking care to read it in as a binary string. For this, we use the remaining string's length and as starting position, the character right after the semicolon.

At this point, we have extracted the command and payload and can see what we have to do:

```cpp

         if (cmd == "mod") {
               if (msg.length() != 4) {
                     Serial1.printf("Payload size wasn't 4 bytes: %d\n", msg.length());
                     return; 
               }

               uint32 input;
               msg.getBytes((unsigned char*) &input, sizeof(uint32), 0);
               String byteStr;
               byteStr = "Received new configuration: ";
               byteStr += input;
               log(LOG_DEBUG, byteStr);
               updateModules(input);               
          }
```

This command sets which feature modules should be active. Its payload should be an unsigned 32-bit integer forming a bitmask, which we check to make sure that we have received exactly four bytes.

In the bitmask, the bits each match up with a module, which at this point are the following:

| **Bit position** | **Value** |
| 0x01 | THPModule |
| 0x02 | CO2Module |
| 0x04 | JuraModule |
| 0x08 | JuraTermModule |
| 0x10 | MotionModule |
| 0x20 | PwmModule |
| 0x40 | IOModule |
| 0x80 | SwitchModule |
| 0x100 | PlantModule |

Of these, the CO2, Jura, and JuraTerm modules are mutually exclusive, since they all use the first UART (`Serial`). If two or more of these are still specified in the bitmask, only the first module will be enabled and the others ignored. We will look at these other feature modules in more detail in [Chapter 9](d165297b-8be7-44f5-90b5-53b3bcb51d3b.xhtml), *Example - Building Management and Control*.

After we obtain the new configuration bitmask, we send it to the `updateModules()` function:

```cpp
        else if (cmd == "loc") {
               if (msg.length() < 1) { return; }
               if (location != msg) {
                     location = msg;
                     fileSetContent("location.txt", location);
               }
         }
```

With this command, we set the new location string, if it is different then the current one, also saving it to the location file in SPIFFS to persist it across a reboot:

```cpp
         else if (cmd == "mod_active") {
               uint32 active_mods = BaseModule::activeMods();
               if (active_mods == 0) {
                     mqtt->publish(MQTT_PREFIX"cc/response", MAC + ";0");
                     return;
               }

               mqtt->publish(MQTT_PREFIX"cc/response", MAC + ";"                                                         + String((const char*) &active_mods, 4));
         }
         else if (cmd == "version") {
               mqtt->publish(MQTT_PREFIX"cc/response", MAC + ";" + version);
         }
         else if (cmd == "upgrade") {
               otaUpdate();
         }
   }
```

The last three commands in this section return the current bitmask for the active feature modules, the firmware version, and trigger an OTA upgrade:

```cpp
         else {
               if (topicCallbacks->contains(topic)) {
                     (*((*topicCallbacks)[topic]))(message);
                }
         }
}
```

The last entry in the `if...else` block looks at whether the topic is perhaps found in our list of callbacks for the feature modules. If found, the callback is called with the MQTT message string.

Naturally, this means that only one feature module can register itself to a specific topic. Since each module tends to operate under its own MQTT sub-topic to segregate the message flow, this is generally not a problem:

```cpp
void OtaCore::updateModules(uint32 input) {
         Serial1.printf("Input: %x, Active: %x.\n", input, BaseModule::activeMods());

         BaseModule::newConfig(input);

         if (BaseModule::activeMods() != input) {
               String content(((char*) &input), 4);
               setFileContent("config.txt", content);
         }
}
```

This function is pretty simple. It mostly serves as a pass-through for the `BaseModule` class, but it also ensures that we keep the configuration file in SPIFFS up to date, writing the new bitmask to it when it has changed.

We absolutely must prevent unnecessary writes to SPIFFs, as the underlying Flash storage has finite write cycles. Limiting write cycles can significantly extend the lifespan of the hardware, as well as reduce overall system load:

```cpp
bool OtaCore::mapGpioToBit(int pin, ESP8266_pins &addr) {
          switch (pin) {
                case 0:
                      addr = ESP8266_gpio00;
                      break;
                case 1:
                      addr = ESP8266_gpio01;
                      break;
                case 2:
                      addr = ESP8266_gpio02;
                      break;
                case 3:
                      addr = ESP8266_gpio03;
                      break;
                case 4:
                      addr = ESP8266_gpio04;
                      break;
                case 5:
                      addr = ESP8266_gpio05;
                      break;
                case 9:
                      addr = ESP8266_gpio09;
                      break;
                case 10:
                      addr = ESP8266_gpio10;
                      break;
                case 12:
                      addr = ESP8266_gpio12;
                      break;
                case 13:
                      addr = ESP8266_gpio13;
                      break;
                case 14:
                      addr = ESP8266_gpio14;
                      break;
                case 15:
                      addr = ESP8266_gpio15;
                      break;
                case 16:
                      addr = ESP8266_gpio16;
                      break;
                default:
                      log(LOG_ERROR, "Invalid pin number specified: " + String(pin));
                      return false;
          };

          return true;
 }
```

This function maps the given GPIO pin number to its position in the internal bitmask. It uses the enumeration we looked at for the header file for this class. With this mapping, we can set the used/unused state of GPIO pins of the ESP8266 module using just a single uint32 value:

```cpp
void OtaCore::log(int level, String msg) {
         String out(lvl);
         out += " - " + msg;

         Serial1.println(out);
         mqtt->publish(MQTT_PREFIX"log/all", OtaCore::MAC + ";" + out);
}
```

In the logging method, we append the log level to the message string before writing it to the serial output, as well as publishing it on MQTT. Here, we publish on a single topic, but as a refinement you could log on a different topic depending on the specified level.

What makes sense here depends a great deal on what kind of backend you have set up to listen for and process logging output from the ESP8266 systems running this firmware:

```cpp
bool OtaCore::starti2c() {
         if (i2c_active) { return true; }

         if (!claimPin(sdaPin)) { return false; }
         if (!claimPin(sclPin)) { return false; }

         Wire.pins(sdaPin, sclPin);
         pinMode(sclPin, OUTPUT);
         for (int i = 0; i < 8; ++i) {
               digitalWrite(sclPin, HIGH);
               delayMicroseconds(3);
               digitalWrite(sclPin, LOW);
               delayMicroseconds(3);
         }

         pinMode(sclPin, INPUT);

         Wire.begin();
         i2c_active = true;
}
```

This function starts the I2C bus if it hasn't been started already. It tries to register the pins it wishes to use for the I2C bus. If these are available, it will set the clock line (SCL) to output mode and first pulse it eight times to unfreeze any I2C devices on the bus.

After pulsing the clock line like his, we start the I2C bus on the pins and make a note of the active state of this bus.

Frozen I2C devices can occur if the MCU power cycles when the I2C devices do not, and remain in an indeterminate state. With this pulsing, we make sure that the system won't end up in a non-functional state, requiring manual intervention:

```cpp
bool OtaCore::startSPI() {
    if (spi_active) { return true; }

    if (!claimPin(SPI_SCLK)) { return false; }
    if (!claimPin(SPI_MOSI)) { return false; }
    if (!claimPin(SPI_MISO)) { return false; }
    if (!claimPin(SPI_CS)) { return false; }

    SPI.begin();
    spi_active = true;
 }
```

Starting the SPI bus is similar to staring the I2C bus, except without a similar recovery mechanism:

```cpp
bool OtaCore::claimPin(int pin) {
          ESP8266_pins addr;
          if (!mapGpioToBit(pin, addr)) { return false; }

          return claimPin(addr);
    }

    bool OtaCore::claimPin(ESP8266_pins pin) {
          if (esp8266_pins & pin) {
                log(LOG_ERROR, "Attempting to claim an already claimed pin: "                                                                                                      + String(pin));
                log(LOG_DEBUG, String("Current claimed pins: ") + String(esp8266_pins));
                return false;
          }

          log(LOG_INFO, "Claiming pin position: " + String(pin));

          esp8266_pins |= pin;

          log(LOG_DEBUG, String("Claimed pin configuration: ") + String(esp8266_pins));

          return true;
 }
```

This overloaded function is used to register a GPIO pin by a feature module before it starts, to ensure that no two modules attempt to use the same pins at the same time. One version accepts a pin number (GPIO) and uses the mapping function we looked at earlier to get the bit address in the `esp8266_pins` bitmask before passing it on to the other version of the function.

In that function, the pin enumeration is used to do a bitwise `AND` comparison. If the bit has not been set yet, it is toggled and true is returned. Otherwise, the function returns false and the calling module knows that it cannot proceed with its initialization:

```cpp
bool OtaCore::releasePin(int pin) {
          ESP8266_pins addr;
          if (!mapGpioToBit(pin, addr)) { return false; }

          return releasePin(addr);
    }

    bool OtaCore::releasePin(ESP8266_pins pin) {
          if (!(esp8266_pins & pin)) {
                log(LOG_ERROR, "Attempting to release a pin which has not been set: "                                                                                                      + String(pin));
                return false;
          }

          esp8266_pins &= ~pin;

          log(LOG_INFO, "Released pin position: " + String(pin));
          log(LOG_DEBUG, String("Claimed pin configuration: ") + String(esp8266_pins));

          return true;
 }
```

This overloaded function, to release a pin when a feature module is shutting down, works in a similar manner. One uses the mapping function to get the bit address, the other performs a bitwise `AND` operation to check that the pin has in fact been set, and toggles it to an off position with the bitwise `OR` assignment operator if it was set.

# BaseModule

This class contains the logic for registering and keeping track of which feature modules are currently active or inactive. Its header file looks as follows:

```cpp
#include "ota_core.h"

enum ModuleIndex {
   MOD_IDX_TEMPERATURE_HUMIDITY = 0,
   MOD_IDX_CO2,
   MOD_IDX_JURA,
   MOD_IDX_JURATERM,
   MOD_IDX_MOTION,
   MOD_IDX_PWM,
   MOD_IDX_IO,
   MOD_IDX_SWITCH,
   MOD_IDX_PLANT
};

typedef bool (*modStart)();
typedef bool (*modShutdown)();
```

The inclusion of the `OtaCore` header is to allow us to use the logging feature. Beyond this, we create another enumeration, which maps a specific feature module to a particular bit in the feature module bitmask (`active_mods`).

Finally, function pointers are defined, which are used for respectively starting and shutting down a feature module. These will be defined by the feature modules as they register themselves:

```cpp
#include "thp_module.h"
#include "jura_module.h"
#include "juraterm_module.h"
#include "co2_module.h"
#include "motion_module.h"
#include "pwm_module.h"
#include "io_module.h"
#include "switch_module.h"
#include "plant_module.h"
```

These are the feature modules that currently exist for this firmware as of writing. Since we only need the plant module for this project, we could comment out all header files for the other modules, along with their initialization in the initialization function of this class.

This would not affect the resulting firmware image in any way other than that we cannot enable those modules since they do not exist.

Finally, here is the class declaration itself:

```cpp
class BaseModule {   
         struct SubModule {
               modStart start;
               modShutdown shutdown;
               ModuleIndex index;
               uint32 bitmask;
               bool started;
         };

         static SubModule modules[32];
         static uint32 active_mods;
         static bool initialized;
         static uint8 modcount;

public:
         static void init();
         static bool registerModule(ModuleIndex index, modStart start,                                                                                    modShutdown shutdown);
```

```cpp

         static bool newConfig(uint32 config);
         static uint32 activeMods() { return active_mods; }
};
```

Each feature module is represented internally by a `SubModule` instance, the details of which we can see in a moment in the class definition:

```cpp
#include "base_module.h"

BaseModule::SubModule BaseModule::modules[32];
uint32 BaseModule::active_mods = 0x0;
bool BaseModule::initialized = false;
uint8 BaseModule::modcount = 0;
```

Since this is a static class, we first initialize its class variables. We have an array with space for 32 `SubModule` instances, to fit the full bitmask. Beyond this, no modules are active, so everything is initialized to zero and false:

```cpp
void BaseModule::init() {
    CO2Module::initialize();
    IOModule::initialize();
    JuraModule::initialize();
    JuraTermModule::initialize();
    MotionModule::initialize();
    PlantModule::initialize();
    PwmModule::initialize();
    SwitchModule::initialize();
    THPModule::initialize();
}
```

When we called this function in `OtaCore`, we also triggered the registration of the feature modules defined here. By selectively removing or commenting out modules in this function, we can remove them from the final firmware image. Those modules that are called here will call the following function to register themselves:

```cpp
bool BaseModule::registerModule(ModuleIndex index, modStart start, modShutdown shutdown) {
         if (!initialized) {
               for (uint8 i = 0; i < 32; i++) {
                     modules[i].start = 0;
                     modules[i].shutdown = 0;
                     modules[i].index = index;
                     modules[i].bitmask = (1 << i);
                     modules[i].started = false;
               }

               initialized = true;
         }

         if (modules[index].start) {
               return false;
         }

         modules[index].start = start;
         modules[index].shutdown = shutdown;
         ++modcount;

         return true;
}
```

The first feature module that calls this function will trigger the initialization of the `SubModule` array, setting all of its values to a neutral setting, while also creating the bitmask for this position in the array, which allows us to update the `active_mods` bitmask, as we will see in a moment.

After initializing the array, we check whether this position in the array already has a module registered for it. If it has, we return false. Otherwise, we register the module's function pointers for starting and shutting down, and increase the active module count before returning true:

```cpp
bool BaseModule::newConfig(uint32 config) {
    OtaCore::log(LOG_DEBUG, String("Mod count: ") + String(modcount));
    uint32 new_config = config ^ active_mods;
    if (new_config == 0x0) {
        OtaCore::log(LOG_INFO, "New configuration was 0x0\. No 
        change.");
        return true; 
    }
    OtaCore::log(LOG_INFO, "New configuration: " + new_config);
    for (uint8 i = 0; i < 32; ++i) {
        if (new_config & (1 << i)) {
            OtaCore::log(LOG_DEBUG, String("Toggling module: ") + 
            String(i));
            if (modules[i].started) { 
                if ((modules[i]).shutdown()) { 
                    modules[i].started = false; 
                    active_mods ^= modules[i].bitmask;
                }
                else { 
                    OtaCore::log(LOG_ERROR, "Failed to shutdown 
                    module.");
                    return false; 
                }
            }
            else { 
                if ((modules[i].start) && (modules[i]).start()) { 
                    modules[i].started = true;
                    active_mods |= modules[i].bitmask;
                }
                else { 
                    OtaCore::log(LOG_ERROR, "Failed to start module.");
                    return false;
                }
            }
        }
    }
    return true;
 }
```

The input parameter to this function is the bitmask we extracted from the MQTT payload in `OtaCore`. Here, we use a bitwise XOR comparison with the active modules bitmask to obtain a new bitmask indicating any changes to be made. If the result is zero, we know that they're identical and we can return without further action being required.

The `uint32` bitmask we have thus obtained indicates which modules should be toggled on or off. For this, we check each bit of the mask. If it is a `1` (AND operator returns a value that's not zero), we check whether the module at that position in the array exists and has been started yet.

If the module has been started, we attempt to shut it down. If the module's shutdown() function succeeds (returns true), we toggle the bit in the `active_mods` bitmask to update its status. Similarly, if the module has not been started yet, a module has been registered at that location, we attempt to start it, updating the active modules if this succeeds.

We check that a start function callback has been registered to ensure that we do not accidentally call an improperly registered module and crash the system.

# PlantModule

At this point, we have had a detailed look at the underlying, supporting code that makes life easy when writing a new module because we don't have to do all of the housekeeping ourselves. The only thing we haven't seen yet is an actual module, or code directly pertaining to this chapter's project.

In this section, we will look at the last part of the puzzle, the `PlantModule` itself:

```cpp
#include "base_module.h"
#include <Libraries/APA102/apa102.h>

#define PLANT_GPIO_PIN 5
#define NUM_APA102 1

class PlantModule {
         static int pin;
         static Timer timer;
         static uint16 humidityTrigger;
         static String publishTopic;
         static HttpServer server;
         static APA102* LED;

         static void onRequest(HttpRequest& request, HttpResponse& response);

public:
         static bool initialize();
         static bool start();
         static bool shutdown();
         static void readSensor();
         static void commandCallback(String message);
};
```

Of note in this class declaration is the inclusion of the APA102 library header. This is a simple library that allows us to write color and brightness data to APA102 RGB (full-spectrum) LEDs, over the SPI bus.

We also define the pin that we wish to use to trigger the peristaltic pump (GPIO 5) and the number of connected APA102 LED modules (1). You can add multiple APA102 LEDs in series if you want, simply updating the definition to match the count.

Next is the class implementation:

```cpp
#include "plant_module.h"

int PlantModule::pin = PLANT_GPIO_PIN;
Timer PlantModule::timer;
uint16 PlantModule::humidityTrigger = 530;
String PlantModule::publishTopic;
HttpServer PlantModule::server;
APA102* PlantModule::LED = 0;

enum {
         PLANT_SOIL_MOISTURE = 0x01,
         PLANT_SET_TRIGGER = 0x02,
         PLANT_TRIGGER = 0x04
};
```

In this section, we initialize the static class members, setting the GPIO pin and defining the initial sensor value at which the pump should be triggered. This trigger value should be updated to match your own sensor calibration results.

Finally, we define an enumeration containing the possible commands for this module that can be sent to it over MQTT:

```cpp
bool PlantModule::initialize() {
          BaseModule::registerModule(MOD_IDX_PLANT, PlantModule::start,                                                                                                                 PlantModule::shutdown);
}
```

This is the initialization function the `BaseModule` calls on startup. As we can see, it causes this module to register itself with preset values, including its start and shutdown callbacks:

```cpp
bool PlantModule::start() {
         OtaCore::log(LOG_INFO, "Plant Module starting...");

         if (!OtaCore::claimPin(pin)) { return false; }

         publishTopic = MQTT_PREFIX + "plant/response/" + OtaCore::getLocation();
         OtaCore::registerTopic(MQTT_PREFIX + String("plants/") +                                                             OtaCore::getLocation(), PlantModule::commandCallback);

         pinMode(pin, OUTPUT);

         server.listen(80);
         server.setDefaultHandler(PlantModule::onRequest);

         LED = new APA102(NUM_APA102);
         LED->setBrightness(15);
         LED->clear();
         LED->setAllPixel(0, 255, 0);
         LED->show();

         timer.initializeMs(60000, PlantModule::readSensor).start();
         return true;
}
```

When this module starts, we attempt to claim the pin we wish to use for triggering the pump, as well as register a callback for an MQTT topic so that we can accept commands using the command handler callback. The topic on which we will responses after processing a command is also defined here.

The output pin mode is set, followed by the starting of the HTTP server on port 80, registering a basic handler for client requests. Next, we create a new `APA102` class instance and use it to get the connected LED to display green at about half of full brightness.

Finally, we start a timer that will trigger the reading out of the connected soil sensor every minute:

```cpp
bool PlantModule::shutdown() {
         if (!OtaCore::releasePin(pin)) { return false; }

         server.shutdown();

         if (LED) {
               delete LED;
               LED = 0;
         }

         OtaCore::deregisterTopic(MQTT_PREFIX + String("plants/") +                                                                                            OtaCore::getLocation());

         timer.stop();
         return true;
}
```

When shutting down this module, we release the pin we registered previously, stop the web server, delete the RGB LED class instance (with a check to see that deleting it is necessary), deregister our MQTT topic, and finally stop the sensor timer.

```cpp

void PlantModule::commandCallback(String message) {
         OtaCore::log(LOG_DEBUG, "Plant command: " + message);

         if (message.length() < 1) { return; }
         int index = 0;
         uint8 cmd = *((uint8*) &message[index++]);

         if (cmd == PLANT_SOIL_MOISTURE) {
               readSensor();
         }
         else if (cmd == PLANT_SET_TRIGGER) {               
                if (message.length() != 3) { return; }
               uint16 payload = *((uint16*) &message[index]);
               index += 2;

               humidityTrigger = payload;
         }
         else if (cmd == PLANT_TRIGGER) {
               OtaCore::publish(publishTopic, OtaCore::getLocation() + ";" 
                                                                + String(((char*) &humidityTrigger), 2));
         }
}
```

This callback is called whenever a message is published on the MQTT topic we registered. In our messages, we expect to find a single byte (uint8) value that defines the command, up to eight distinct commands. For this module, we earlier defined three commands.

These commands are defined as follows:

| **Command** | **Meaning** | **Payload** | **Return value** |
| 0x01 | Get soil moisture | - | 0xXXXX |
| 0x02 | Set trigger level | uint16 (new trigger level) | - |
| 0x04 | Get trigger level | - | 0xXXXX |

Here, every command returns the requested value, if applicable.

After checking that the message string we got has at least one byte in it, we extract the first byte and try to interpret it as a command. If we are setting a new trigger point, we also extract the new value as a uint16 from the message after making sure that we have a properly formed message.

Finally, here is the function in which all of the magic happens that we have been working toward in this project:

```cpp
void PlantModule::readSensor() {
    int16_t val = 0;
    val = analogRead(A0); // calls system_adc_read().

    String response = OtaCore::getLocation() + ";" + val;
    OtaCore::publish(MQTT_PREFIX"nsa/plant/moisture_raw", response);
```

As the first step, we read out the current sensor value from the analog input of the ESP8266 and publish it on the MQTT topic for this:

```cpp
        if (val >= humidityTrigger) {
               digitalWrite(pin, HIGH);

               LED->setBrightness(31);
               LED->setAllPixel(0, 0, 255);
               LED->show();

               for (int i = 0; i < 10; ++i) {
                     LED->directWrite(0, 0, 255, 25);
                     delay(200);
                     LED->directWrite(0, 0, 255, 18);
                     delay(200);
                     LED->directWrite(0, 0, 255, 12);
                     delay(200);
                     LED->directWrite(0, 0, 255, 5);
                     delay(200);
                     LED->directWrite(0, 0, 255, 31);
                     delay(200);
               }

               digitalWrite(pin, LOW);
         }
}
```

During calibration of one prototype with a soil moisture sensor, it was found that the value for a completely dry sensor (held in the air) was approximately 766, whereas having the same sensor submerged in water got 379 as a value. From this, we can deduce that 60% moisture content should be roughly around a reading of 533, which matches the initial value we set during the static initialization step. The ideal trigger point and target soil moisture level of course depends on the soil type and specific plant.

With this trigger level reached, we set the output pin that is connected to the enable pin of the boost converter to high, causing it to turn on its output, which in turn starts the pump. We wish to let it pump for about ten seconds.

During this time we set the LED color to blue, then during each second we drop its brightness from 100% to nearly off and then back to full brightness again, creating a pulsating effect.

After this, we set the output pin back to low, which disables the pump, and we await the next soil moisture sensor reading:

```cpp
void PlantModule::onRequest(HttpRequest& request, HttpResponse& response) {
         TemplateFileStream* tmpl = new TemplateFileStream("index.html");
         TemplateVariables& vars = tmpl->variables();
         int16_t val = analogRead(A0);
         int8_t perc = 100 - ((val - 379) / 3.87);
         vars["raw_value"] = String(val);
         vars["percentage"] = String(perc);

         response.sendTemplate(tmpl);
}
```

Finally, we see here the request handler for our web server. What it does is read in a template file from SPIFFS (detailed in the next section), gets the list of variables in this template file, and then proceeds to read out the current sensor value.

Using this value, it calculates the current soil moisture percentage and uses both the raw and calculated numbers to fill in the two variables in the template before returning it.

# Index.html

For use with the PlantModule's web server, we have to add the following template file to SPIFFS:

```cpp
<!DOCTYPE html>
<html>
<head>
         <title>Plant soil moisture readings</title>
   </head>
   <body>
         Current value: {raw_value}<br>
         Percentage: {percentage}%
</body>
</html>
```

# Compiling and flashing

After finishing the code for our application, we can compile it with a single command in the project's root folder:

```cpp
make  
```

After this completes, we can find the binaries including the ROM images in the `out` folder. Since we are using both the rBoot bootloader and SPIFFs, we get three ROM images in total in the `firmware` folder.

At this point, we can connect an ESP8266 module, either in the form of a NodeMCU board or one of the many alternatives, and note the serial port that it will be connected to. On Windows, this will be something like `COM3`; on Linux, USB-to-serial adapters usually are registered as `/dev/ttyUSB0` or similar.

Unless we have specified the serial port (`COM_PORT`) in the user Makefile, we have to specify it explicitly when we flash to the ESP8266 module:

```cpp
    make flash COM_PORT=/dev/ttyUSB0  
```

After executing this command, we should see the output from the `esptool.py` utility, as it connects to the ESP8266's ROM and starts writing the ROM images to it.

Once this is complete, the MCU will restart and it should boot straight into the new firmware image, where it will await our commands to configure it.

# First-time configuration

As noted earlier in this chapter, this firmware is designed to be configured and maintained over MQTT. This requires that an MQTT broker is available. An MQTT broker such as Mosquitto ([http://mosquitto.org/](http://mosquitto.org/)) is popular. Since it's a lightweight server, it can be installed on a desktop system, a small SBC, inside a virtual machine, and so on.

In addition the broker and the ESP8266 running the firmware, we also need our own client
to interact with the firmware. Since we use binary protocols, our choice there is somewhat
limited, as most common MQTT desktop clients assume text-based messages. One
approach one can use to publish binary messages is to use the MQTT publish client that
comes with Mosquitto and use the **echo** command-line tool's hexadecimal input to send
binary data to it as a stream to be published by the client tool

Because of this, the author of this book has developed a new MQTT desktop client (based on C++ and Qt) that is designed around the use and debugging of binary protocols on MQTT: [https://github.com/MayaPosch/MQTTCute](https://github.com/MayaPosch/MQTTCute).

With all three components in place—ESP8266 running the project, the MQTT broker and desktop client—we can have the entire plant monitoring and watering system assembled and send it the command to enable the plant module.

While monitoring the cc/config topic for messages, we should see the ESP8266 report its presence by publishing its `MAC`. We can also get this by hooking up a USB to TTL serial adapter to the serial logging output pin (`D4` on NodeMCU). By looking at the output on our serial console, we will see both the IP address and the `MAC` of the system.

When we compose a new topic of the format `cc/<MAC>`, we can then publish commands to the firmware, for example:

```cpp
    log;plant001  
```

This would set the location name of the system to `plant001`.

When using the MQTTCute client, we can use echo-style binary input, using hexadecimal input, to activate the plant module:

```cpp
mod;\x00\x01\x00\x00  
```

This would send the `mod` command to the firmware, along with a bitmask with the value 0 x 100\. After this, the plant module should be activated and running. Since we are persisting both the location string and the configuration, we do not have to repeat this step any more unless we do an OTA update, at which point the new firmware will have an empty SPIFFS filesystem, unless we flash the same SPIFFS image on both SPIFFS slots on the ROM.

Here, we could expand the OTA code to also download a SPIFFS ROM image in addition to the firmware one, though this might add the complication of possibly overwriting the existing SPIFFS files.

At this point, we should have a working plant-monitoring and -watering system.

# Using the system

We can use the measured values and store them in a database by subscribing to the `nsa/plant/moisture_raw` topic. The trigger point can be adjusted by sending a new command to the `plant/<location string>` topic.

The web server on the device can be accessed by taking the IP address, which we can find either by looking at the output on the serial console, as described in the previous section, or by looking at the active IP addresses in your router.

By opening this IP address in the browser, we should see the HTML template filled in with the current values.

# Taking it further

You also need to have the following considerations:

*   At this point, you can further refine the system by implementing plant-watering profiles to add dry periods or to adjust for certain soil types. You can add new RGB LED modes to make full use of the color choices available.
*   The entire hardware could be built into an enclosure, to make it blend into the background, or maybe to make it more visible.
*   The web interface could be extended to allow for controlling the trigger point and such from the browser, instead of having to use an MQTT client.
*   In addition to the moisture sensor, you could also add a brightness sensor, a temperature sensor, and so on, to measure more aspects that affect the plant's health.
*   For bonus points, you could automate the applying of (liquid) fertilizer to the plant.

# Complications

One possible complication you may encounter with the ESP8266's ADC is that on the NodeMCU boards, the first reserved (RSV) pin that is right next to the ADC pin is directly connected to the ESP8266 module's ADC input. This can potentially cause issues with electrostatic discharge ESD exposure. Essentially the discharging of a high voltage, but low current, into the MCU. Adding a small capacitor on this RSV pin to ground can help reduce this risk.

One thing that this system obviously cannot help with is to keep your plants pest-free. This means that though the watering may be automated, that doesn't mean that you can just ignore the plants. The regular checking of the plants for any issues, as well as the system for any issues that may be developing (disconnected tubing, things that have fallen over due to cats, and so on) remains an important task.

# Summary

In this chapter, we looked at how to take a simple ESP8266-based project from theory and simple requirements to a functioning design with a versatile firmware and a collection of input and output options, using which we can ensure that a connected plant gets just the right amount of water to stay healthy. We also saw how to set up a development environment for the ESP8266.

The reader should now be able to create projects for the ESP8266, program the MCU with new firmware, and have a solid grasp on both the strengths and limitations of this development platform.

In the next chapter, we will be looking at how to test embedded software written for SoCs and other large, embedded platforms.