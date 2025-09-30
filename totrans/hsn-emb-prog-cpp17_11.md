# Example - Building Monitoring and Control

The monitoring of conditions within a building, including the temperature, humidity, and CO[2] levels is becoming increasingly more common, with the goal being to adjust heating, cooling, and ventilation systems to keep the occupants of the building as comfortable as possible. In this chapter, such a system is explored and implemented. The following topics will be covered:

*   Creating complex firmware for the ESP8266
*   Integrating an MCU into an IP-based network
*   Adding CO[2] and I2C-based sensors
*   Using GPIO and PWM to control relays and DC voltage-controlled fans
*   Connecting networked nodes using a central controller

# Plants, rooms, and beyond

In [Chapter 5](886aecf2-8926-4aec-8045-a07ae2cdde84.xhtml), *Example - Soil Humidity Monitor with Wi-Fi*, we looked at developing firmware for the ESP8266 MCU to complement a soil humidity sensor and pump, to ensure that a connected plant would be provided with sufficient water from the water tank.

As we noted in that chapter, the firmware used is highly modular and has the highly flexible MQTT-based interface so that it can be used for a wide variety of modules. This chapter covers the system in which the firmware originated: **Building Management and Control** (**BMaC**), originally developed just to monitor rooms for their temperature, humidity, and CO² levels, but later expanded to keep tabs on coffee machines and meeting room occupancy, and ultimately to control the air-conditioning throughout the building.

The BMaC project's current development status can be found at the author's GitHub account at [https://github.com/MayaPosch/BMaC](https://github.com/MayaPosch/BMaC). The version we are covering here is as it exists at this point, with us covering how this system came to be and what it looks like today, and why.

# Developmental history

The BMaC project started when sensors were to be added around an office building in order to measure temperature and other parameters, such as relative humidity. After deciding to use ESP8266 MCUs along with DHT22 temperature and humidity sensors, a simple prototype was put together, using a basic firmware written using the Sming framework.

It was found that DHT22 sensors were generally rather bulky and not very precise. The breakout boards used also had an improper resistor mounted on them, leading to the wrong temperature being reported. This sensor type also had the disadvantage of using its own one-wire protocol, instead of a standard interface method.

The DHT22 sensors got swapped out with BME280 MEMS sensors, which measure temperature, humidity, and also air pressure. A CO[2] sensor was added as well, in the form of the MH-Z19\. This required the firmware to support these additional sensors too. The sensor readings would be sent as MQTT messages, with a backend service subscribing to these topics, and writing them to a time series database (InfluxDB), for viewing and analysis.

Decisions had to be made when the possibility of reading out the counters for products from the fully automatic Jura coffee machines was considered, and with it whether separate firmware would have to be developed.

Instead of separate firmware, the decision was made to use the same firmware for all ESP8266 nodes. This meant that they needed to have the functionality to somehow enable individual features and to support specific sensors and other features. This led to the development of new firmware, which allowed remote commands, sent over MQTT, to toggle feature modules on or off, along with other management features.

Along with the new firmware, a **command and control** (**C&C**) server was added, used by the individual nodes to retrieve their configuration, along with an administration application to be used to add new nodes and add or edit the node configuration.

With this framework in place, it became possible to add new features quickly. These included the addition of motion sensors, for detecting the presence of people in a room, to ultimately the controlling of air-conditioning units, as the existing centralized control in the office building was found to be inadequate.

The system as a whole can be visualized like this:

![](img/d784b62d-39db-497b-91d4-8914218b0cde.png)

In the upcoming sections, we will be taking a detailed look at each of these aspects.

# Functional modules

Here is a list of modules in this firmware:

| **Name** | **Feature** | **Description** |
| THP | Temperature, Humidity, Pressure | Central class for THP sensors. Enables BME280 functionality by default. |
| CO[2] | CO[2] value | Measures CO[2] <indexentry content="functional modules, Building Management and Control (BMaC):CO[2]">values using an MH-Z19 or compatible sensor. |
| Jura | TopTronics EEPROM counters | Reads out the counters for various products from the EEPROM. |
| JuraTerm | TopTronics remote control | Allows a remote service to send TopTronics (classic, v5-style) commands to a supported coffee machine. |
| Motion | Motion detection | Uses an HC-SR501 PIR motion sensor or compatible to detect motion. |
| PWM | PWM output | Sets a pulse-width modulation output on one or more pins. |
| I/O | I/O expansion | Supports the MCP23008 eight-channel I/O expander module via I2C. |
| Switch | Persistent switch | Controls a switch that uses a latching relay or equivalent for switching. |
| Plant | Watering plants | Reads out an analog soil sensor to determine soil humidity, activating a pump when needed. |

# Firmware source

In this section, we look at the source code for the ESP8266 firmware as used with the BMaC system.

# Core

The core of the firmware we already looked at in [Chapter 5](886aecf2-8926-4aec-8045-a07ae2cdde84.xhtml), *Example - Soil Humidity Monitor with Wi-Fi*, including the entry point, the `OtaCore` class, and the `BaseModule` class, which provide all of the functionality needed to make individual modules initialize and to allow them to be enabled and disabled using the MQTT interface.

# Modules

Of the firmware modules, we already looked at the plant module in [Chapter 5](886aecf2-8926-4aec-8045-a07ae2cdde84.xhtml), *Example - Soil Humidity Monitor with Wi-Fi*. Here we will look at the remaining modules, starting with the THP module:

```cpp
#include "base_module.h"
class THPModule {
    public:
    static bool initialize();
    static bool start();
    static bool shutdown();
};
#include "thp_module.h"
#include "dht_module.h"
#include "bme280_module.h"
bool THPModule::initialize() {
    BaseModule::registerModule(MOD_IDX_TEMPERATURE_HUMIDITY, 
    THPModule::start, THPModule::shutdown);
    return true;
}
bool THPModule::start() {
    BME280Module::init();
    return true;
}
bool THPModule::shutdown() {
    BME280Module::shutdown();
    return true;
}
```

This module has the provisions to act as a generic interface to a wide variety of temperature, humidity, and air-pressure sensors. As this was not a requirement at the time, it merely acted as a pass-through for the BME280 module. It registers itself with the base module when called and calls the respective functions on the BME280 module when its own are called.

To make it more versatile, the class would be extended to allow for commands to be received—possibly over MQTT as well on its own topic—which would then enable a specific sensor module, or even a collection of them, when using separate temperature and air pressure sensors, for example.

Regardless of whether it is being used or not in this firmware, let's take a look at the DHT module so that we can compare it with the BME280 module later.

```cpp
#include "ota_core.h"

 #include <Libraries/DHTesp/DHTesp.h>

 #define DHT_PIN 5 // DHT sensor: GPIO5 ('D1' on NodeMCU)

 class DHTModule {
    static DHTesp* dht;
    static int dhtPin;
    static Timer dhtTimer;

 public:
    static bool init();
    static bool shutdown();
    static void config(String cmd);
    static void readDHT();
 };
```

Of note is that while the class is static, any variables that would take up considerable memory—such as library class instances—are defined as pointers. This forms a compromise between having the module available for easy use and going for a more complicated, fully dynamic solution. As most MCUs tend to keep as much of the program code as possible in the ROM until it is used, this should keep both SRAM and ROM usage to a minimum.

```cpp
#include "dht_module.h"

 DHTesp* DHTModule::dht = 0;
 int DHTModule::dhtPin = DHT_PIN;
 Timer DHTModule::dhtTimer;

 bool DHTModule::init() {
    if (!OtaCore::claimPin(dhtPin)) { return false; }
    if (!dht) { dht = new DHTesp(); dht->setup(dhtPin, DHTesp::DHT22); }
    dhtTimer.initializeMs(2000, DHTModule::readDHT).start();    
    return true;
 }
```

To initialize the module, we ensure that we can safely use the **general-purpose input/output** (**GPIO**) pins we intend to use, create a new instance of the sensor class from the library, and set it up before creating the 2-second timer that will perform the scheduled sensor read-out.

Since we create a new instance of the  sensor class upon initializing there should never be an existing instance of this class, but we check in case the init() function gets called again a second time for some reason. Calling the initialization function on the timer a second time could also be included in this block, but isn't strictly required as there is no harmful effect from initializing the timer again.

```cpp
bool DHTModule::shutdown() {
    dhtTimer.stop();
    if (!OtaCore::releasePin((ESP8266_pins) dhtPin)) { delete dht; return false; }
    delete dht;
    dht = 0;    
    return true;
 }
```

To shut down the module, we stop the timer and release the GPIO pin we were using, before cleaning up all resources we used. As we have claimed the pin we're using previously when we initialized the module we should have no issues releasing it again, but we check to make sure.

```cpp

 void DHTModule::config(String cmd) {
    Vector<String> output;
    int numToken = splitString(cmd, '=', output);
    if (output[0] == "set_pin" && numToken > 1) {
          dhtPin = output[1].toInt();
    }
 }
```

This is an example of how one could later change the GPIO pin used by a module, here using the old text-based command format that early versions of the BMaC firmware used to use. We could also receive this information via an MQTT topic, or by actively querying the command and control server. 

Note that to change the pin used by the sensor one would have to restart the sensor by deleting the class instance and creating a new instance.

```cpp
 void DHTModule::readDHT() {
    TempAndHumidity th;
    th = dht->getTempAndHumidity();

    OtaCore::publish("nsa/temperature", OtaCore::getLocation() + ";" + th.temperature);
    OtaCore::publish("nsa/humidity", OtaCore::getLocation() + ";" + th.humidity);
 }
```

Next, for the `BME280` sensor module, its code looks like this:

```cpp
#include "ota_core.h"

 #include <Libraries/BME280/BME280.h>

 class BME280Module {
    static BME280* bme280;
    static Timer timer;

 public:
    static bool init();
    static bool shutdown();
    static void config(String cmd);
    static void readSensor();
 };
```

Finally, it's familiar-looking implementation:

```cpp
#include "bme280_module.h"

 BME280* BME280Module::bme280 = 0;
 Timer BME280Module::timer;

 bool BME280Module::init() {
    if (!OtaCore::starti2c()) { return false; }
    if (!bme280) { bme280 = new BME280(); }

    if (bme280->EnsureConnected()) {
          OtaCore::log(LOG_INFO, "Connected to BME280 sensor.");
          bme280->SoftReset();
          bme280->Initialize();
    }
    else {
          OtaCore::log(LOG_ERROR, "Not connected to BME280 sensor.");
          return false;
    }

    timer.initializeMs(2000, BME280Module::readSensor).start();

    return true;
 }

 bool BME280Module::shutdown() {
    timer.stop();
    delete bme280;
    bme280 = 0;

    return true;
 }

 void BME280Module::config(String cmd) {
    Vector<String> output;
    int numToken = splitString(cmd, '=', output);
    if (output[0] == "set_pin" && numToken > 1) {
          //
    }
 }

 void BME280Module::readSensor() {
    float t, h, p;
    if (bme280->IsConnected) {
          t = bme280->GetTemperature();
          h = bme280->GetHumidity();
          p = bme280->GetPressure
          OtaCore::publish("nsa/temperature", OtaCore::getLocation() + ";" + t);
          OtaCore::publish("nsa/humidity", OtaCore::getLocation() + ";" + h);
          OtaCore::publish("nsa/pressure", OtaCore::getLocation() + ";" + p);
    }
    else {
          OtaCore::log(LOG_ERROR, "Disconnected from BME280 sensor.");
    }
 }

```

As we can see, this module was basically copied from the DHT one, and then modified to fit the BME280 sensor. The similarities between those two modules was one of the motivations behind developing the THP module, in order to exploit these similarities.

As with the DHT module, we can see that we rely on an external library to do the heavy lifting for us, with us merely having to call the functions on the library class to set up the sensor and get the data from it.

# CO2 module

For the CO[2] module, no attempt has been made yet to make it work with multiple types of CO[2] sensors. The first CO[2] sensor used was the MH-Z14, before it <indexentry content="modules, Building Management and Control (BMaC):CO[2] module">switched to the more compact MH-Z19 sensor. Both of these use the same protocol on their **universal asynchronous receiver/transmitter** (**UART**) interface, however.

On the ESP8266 there are two UARTs, though only one is complete, with a receive (RX) and send (TX) line. The second UART only has a TX line. This essentially limits this MCU to a single UART and thus single UART-based sensor.

These sensors also have a single-wire interface in addition to the UART-based interface, where the sensor outputs the current sensor reading using a specific encoding that has to be received and decoded using the specific distance between pulses on that signal wire. This is similar to the DHT-22's single-wire protocol.

Using the UART is obviously significantly easier, and it's what we ended up using with this module:

```cpp
#include "base_module.h"

 class CO2Module {
    static Timer timer;
    static uint8_t readCmd[9];
    static uint8 eventLevel;
    static uint8 eventCountDown;
    static uint8 eventCountUp;

    static void onSerialReceived(Stream &stream, char arrivedChar, unsigned short availableCharsCount);

 public:
    static bool initialize();
    static bool start();
    static bool shutdown();
    static void readCO2();
    static void config(String cmd);
 };
```

We can see here the callback function that will be used with the UART when we receive data. We also have a few other variables whose meaning will <indexentry content="modules, Building Management and Control (BMaC):CO[2] module">become clear in a moment:

```cpp
#include "CO2_module.h"

 Timer CO2Module::timer;
 uint8_t CO2Module::readCmd[9] = { 0xFF,0x01,0x86,0x00,0x00,0x00,0x00,0x00,0x79};
 uint8 CO2Module::eventLevel = 0;
 uint8 CO2Module::eventCountDown = 10;
 uint8 CO2Module::eventCountUp = 0;

```

In the static initializations, we define the command that we will be sending to the CO[2] sensor, which will tell it to send us its currently measured value. We define a number of counters and the related timer instance, which we will be using to analyze the CO[2] levels we receive.

```cpp
bool CO2Module::initialize() {
    BaseModule::registerModule(MOD_IDX_CO2, CO2Module::start, CO2Module::shutdown);
    return true;
 }

 bool CO2Module::start() {
    if (!OtaCore::claimPin(ESP8266_gpio03)) { return false; }
    if (!OtaCore::claimPin(ESP8266_gpio01)) { return false; }

    Serial.end();
    delay(10);
    Serial.begin(9600);
    Serial.setCallback(&CO2Module::onSerialReceived);

    timer.initializeMs(30000, CO2Module::readCO2).start();
    return true;
 }
```

Starting this module triggers the registering of the pins, which we need for the UART, with the UART started at a baud rate of 9,600\. Our receive callback is also registered. The pin registration routine in the core class is meant for housekeeping and therefore cannot really fail. In case of an overlapping pin mapping with another module, we might want to release the first pin registration if the second registration fails.

The GPIO pins used by the serial interface are set in the same core class and would have to be modified there. The main reason behind this lack of configurability is that the GPIO pins on the ESP8266 are fairly limited in what features they support, which is why the hardware UART is basically always found on these two pins, leaving the other pins for other functionality.

The timer we start will read out the sensor every 30 seconds, keeping in mind that the first 3 minutes of sensor readings are useless as the sensor takes <indexentry content="modules, Building Management and Control (BMaC):CO[2] module">about that long to warm up.

```cpp
bool CO2Module::shutdown() {
    if (!OtaCore::releasePin(ESP8266_gpio03)) { return false; }
    if (!OtaCore::releasePin(ESP8266_gpio01)) { return false; }

    timer.stop();
    Serial.end();
    return true;
 }

 void CO2Module::readCO2() {
    Serial.write(readCmd, 9);
 }
```

Reading out the sensor is as easy as writing the series of bytes we defined during the static initialization phase to the sensor, and waiting for the sensor to respond by sending data back to us into our RX buffer, which will trigger our callback function.

```cpp
 void CO2Module::config(String cmd) {
    Vector<String> output;
    int numToken = splitString(cmd, '=', output);
    if (output[0] == "event" && numToken > 1) {
          // 
    }
 }

```

The configuration method was also left unimplemented here, but could be used to disable events (explained in the next part) and make various adjustments dynamically:

```cpp
void CO2Module::onSerialReceived(Stream &stream, char arrivedChar, unsigned short availableCharsCount) {
    if (availableCharsCount >= 9) {
          char buff[9];
          Serial.readBytes(buff, 9);

          int responseHigh = (int) buff[2];
          int responseLow = (int) buff[3];
          int ppm = (responseHigh * 0xFF) + responseLow;
          String response = OtaCore::getLocation() + ";" + ppm;
          OtaCore::publish("nsa/CO2", response);

          if (ppm > 1000) { // T3
                if (eventLevel < 2 && eventCountUp < 10) {
                      if (++eventCountUp == 10) {
                            eventLevel = 2;
                            eventCountDown = 0;
                            eventCountUp = 0;
                            response = OtaCore::getLocation() + ";" + eventLevel + ";1;" + ppm;
                            OtaCore::publish("nsa/events/CO2", response);
                      }
                }
          }
          else if (ppm > 850) { // T2
                if (eventLevel == 0 && eventCountUp < 10) {
                      if (++eventCountUp == 10) {
                            eventLevel = 1;
                            eventCountDown = 0;
                            eventCountUp = 0;
                            response = OtaCore::getLocation() + ";" + eventLevel + ";1;" + ppm;
                            OtaCore::publish("nsa/events/CO2", response);
                      }
                }
                else if (eventLevel == 2 && eventCountDown < 10) {
                      if (++eventCountDown == 10) {
                            eventLevel = 1;
                            eventCountUp = 0;
                            eventCountDown = 0;
                            response = OtaCore::getLocation() + ";" + eventLevel + ";0;" + ppm;
                            OtaCore::publish("nsa/events/CO2", response);
                      }
                }
          }
          else if (ppm < 750) { // T1
                if (eventLevel == 1 && eventCountDown < 10) {
                      if (++eventCountDown == 10) {
                            eventLevel = 0;
                            eventCountDown = 0;
                            eventCountUp = 0;
                            response = OtaCore::getLocation() + ";" + eventLevel + ";0;" + ppm;
                            OtaCore::publish("nsa/events/CO2", response);
                      }
                }
          }
    }
 }
```

In the callback, we get the characters as they come in on the RX line. We wait until we have nine characters waiting for us in the RX buffer, which is the <indexentry content="modules, Building Management and Control (BMaC):CO[2] module">number of bytes we are expecting to receive from the CO[2] sensor. We could also validate the checksum for the received data, for which the MH-Z19 datasheet gives the following C code:

```cpp
char getCheckSum(char* packet) { 
    char i, checksum; 
    for ( i = 1; i < 8; i++) { 
        checksum += packet[i]; 
    } 

    checksum = 0xff – checksum; 
    checksum += 1; 
    return checksum; 
}
```

This routine calculates the checksum for the received data as a single byte, which we can then compare with the value contained in the 9th byte of the received data to see whether the values match.

Returning to our own code, we process the bytes to calculate the **parts per million** (**PPM**) of the CO[2] molecules the sensor detected. This value is immediately published to its respective MQTT topic.

After this, we compare the new PPM value to see whether we have crossed any of the three preset trigger levels, the first one of which indicates a safe CO[2 ]level, the second an elevated CO[2 ]level, and the third a very high CO[2] level that requires attention. When we exceed or return to a lower trigger level, an event is published for this on the MQTT topic.

# Jura

This is another module which uses the UART. It was used with a number of Jura coffee machines, which used the common TopTronics electronics used by other coffee machine manufacturers as well. To enable reading out these coffee machines, an ESP8266 module was integrated into a small, plastic enclosure which just had a serial connector on one side. This connected with a standard nine-pin serial cable to the so-called service port on the back of the machine.

The serial port on the machine provided 5V when it was powered on, which thus also turned on the ESP8266 node when the coffee machine was on. The plastic enclosure could then be hidden away behind the machine.

The module for this feature looks like this:

```cpp
#include "base_module.h"

 class JuraModule {
    static String mqttTxBuffer;
    static Timer timer;

    static bool toCoffeemaker(String cmd);
    static void readStatistics();
    static void onSerialReceived(Stream &stream, char arrivedChar, unsigned short availableCharsCount);

 public:
    static bool initialize();
    static bool start();
    static bool shutdown();
 };
```

The only really noticeable thing about this class declaration is the method name involving a coffee maker. We'll see in a second what it does:

```cpp
#include "jura_module.h"
 #include <stdlib.h>
 Timer JuraModule::timer;
 String JuraModule::mqttTxBuffer;
 bool JuraModule::initialize() {
    BaseModule::registerModule(MOD_IDX_JURA, JuraModule::start, JuraModule::shutdown);
 }
 bool JuraModule::start() {
    if (!OtaCore::claimPin(ESP8266_gpio03)) { return false; }
    if (!OtaCore::claimPin(ESP8266_gpio01)) { return false; }
    Serial.end();
    delay(10);
    Serial.begin(9600);
    Serial.setCallback(&JuraModule::onSerialReceived);
    timer.initializeMs(60000, JuraModule::readStatistics).start();
    return true;
 }
```

As is common, the coffee machine's UART runs at 9,600 baud. We set the serial callback method, and start a timer for reading out the EEPROM's product counters. Since we are talking about a coffee machine, reading out the counters more than once a minute is somewhat silly:

```cpp
bool JuraModule::shutdown() {
    if (!OtaCore::releasePin(ESP8266_gpio03)) { return false; } // RX 0
    if (!OtaCore::releasePin(ESP8266_gpio01)) { return false; } // TX 0
    timer.stop();
    Serial.end();
    return true;
 }
 void JuraModule::readStatistics() {
    String message = "RT:0000";
    JuraModule::toCoffeemaker(message);
 }
```

To read out the EEPROM's counters, we need to send the command for this to the machine's UART. This command will tell it to send us the contents of the first row in the EEPROM. Unfortunately, the machine's protocol doesn't use plain text, but requires a bit of special encoding, which we do in the next method:

```cpp
bool JuraModule::toCoffeemaker(String cmd) {
    OtaCore::log(LOG_DEBUG, "Sending command: " + cmd);
    cmd += "\r\n";
    for (int i = 0; i < cmd.length(); ++i) {
          uint8_t ch = static_cast<uint8_t>(cmd[i]);
          uint8_t d0 = 0xFF;
          uint8_t d1 = 0xFF;
          uint8_t d2 = 0xFF;
          uint8_t d3 = 0xFF;
          bitWrite(d0, 2, bitRead(ch, 0));
          bitWrite(d0, 5, bitRead(ch, 1));
          bitWrite(d1, 2, bitRead(ch, 2));
          bitWrite(d1, 5, bitRead(ch, 3));
          bitWrite(d2, 2, bitRead(ch, 4));
          bitWrite(d2, 5, bitRead(ch, 5));
          bitWrite(d3, 2, bitRead(ch, 6)); 
          bitWrite(d3, 5, bitRead(ch, 7));
          delay(1); 
          Serial.write(d0);
          delay(1); 
          Serial.write(d1);
          delay(1); 
          Serial.write(d2);
          delay(1); 
          Serial.write(d3);
          delay(7);
    }     
    return true;
 }
```

This method takes in a string, appending the required EOL characters and encoding each byte into four bytes, putting the data bits into each new byte's second and fifth bit, the rest of the bits all being a 1\. These four bytes are then sent to the machine's UART with a small delay between each write to ensure correct reception:

```cpp
void JuraModule::onSerialReceived(Stream &stream, char arrivedChar, 
unsigned short availableCharsCount) {

    OtaCore::log(LOG_DEBUG, "Receiving UART 0.");
    while(stream.available()){

        delay(1);
        uint8_t d0 = stream.read();
        delay(1);
        uint8_t d1 = stream.read();
        delay(1);
        uint8_t d2 = stream.read();
        delay(1);
        uint8_t d3 = stream.read();
        delay(7);

        uint8_t d4;
        bitWrite(d4, 0, bitRead(d0, 2));
        bitWrite(d4, 1, bitRead(d0, 5));
        bitWrite(d4, 2, bitRead(d1, 2));
        bitWrite(d4, 3, bitRead(d1, 5));
        bitWrite(d4, 4, bitRead(d2, 2));
        bitWrite(d4, 5, bitRead(d2, 5));
        bitWrite(d4, 6, bitRead(d3, 2));
        bitWrite(d4, 7, bitRead(d3, 5));
        OtaCore::log(LOG_TRACE, String(d4));
        mqttTxBuffer += (char) d4;

        if ('\n' == (char) d4) {
            long int espressoCount = strtol(mqttTxBuffer.substring(3, 
            7).c_str(), 0, 16);
            long int espresso2Count = strtol(mqttTxBuffer.substring(7, 
            11).c_str(), 0, 16);
            long int coffeeCount = strtol(mqttTxBuffer.substring(11, 
            15).c_str(), 0, 16);
            long int coffee2Count = strtol(mqttTxBuffer.substring(15, 
            19).c_str(), 0, 16);
            OtaCore::publish("nsa/espresso", OtaCore::getLocation() + 
            ";" + espressoCount);
            OtaCore::publish("nsa/espresso2", OtaCore::getLocation() + 
            ";" + espresso2Count);
            OtaCore::publish("nsa/coffee", OtaCore::getLocation() + ";" 
            + coffeeCount);
            OtaCore::publish("nsa/coffee2", OtaCore::getLocation() + 
            ";" + coffee2Count);
            mqttTxBuffer = "";
          }
    }
 }
```

In the serial receive callback, we decode each byte we receive using the same process we used to encode the data we sent to the machine, buffering the decoded bytes until we detect the end of the response (linefeed, LF) character. We then read out the 16-bit counters, which we then publish on the MQTT topic for them.

# JuraTerm

The JuraTerm module is similar to the Jura one, but it accepts remote commands, encodes them in the same way as the Jura module, and returns the decoded response. In the project it used to be the Jura class until it got superseded by the new Jura class and this one was delegated to just a terminal class. In a future revision this module's functionality will therefore be merged into the main Jura class.

```cpp
#include "base_module.h" 

class JuraTermModule {
    static String mqttTxBuffer;

    static bool toCoffeemaker(String cmd);
    static void onSerialReceived(Stream &stream, char arrivedChar, unsigned short availableCharsCount);

 public:
    static bool initialize();
    static bool start();
    static bool shutdown();
    static void commandCallback(String message);
 };#include "juraterm_module.h"

 String JuraTermModule::mqttTxBuffer;

 bool JuraTermModule::initialize() {
    BaseModule::registerModule(MOD_IDX_JURATERM, JuraTermModule::start, JuraTermModule::shutdown);
 }

 bool JuraTermModule::start() {
    if (!OtaCore::claimPin(ESP8266_gpio03)) { return false; } // RX 0
    if (!OtaCore::claimPin(ESP8266_gpio01)) { return false; } // TX 0

    OtaCore::registerTopic("coffee/command/" + OtaCore::getLocation(), 
                            JuraTermModule::commandCallback); 
    Serial.end();
    delay(10);
    Serial.begin(9600);
    Serial.setCallback(&JuraTermModule::onSerialReceived);

    return true;
 }

 bool JuraTermModule::shutdown() {
    if (!OtaCore::releasePin(ESP8266_gpio03)) { return false; } // RX 0
    if (!OtaCore::releasePin(ESP8266_gpio01)) { return false; } // TX 0

    Serial.end();
    OtaCore::deregisterTopic("coffee/command/" + OtaCore::getLocation());
    return true;
 }

 void JuraTermModule::commandCallback(String message) {
    if (message == "AN:0A") { return; }

    JuraTermModule::toCoffeemaker(message);
 }
```

When we start this module, we register an MQTT topic to receive commands. This allows us to receive the coffee machine commands. We basically act as a straight pass-through for these commands, except for this one particular command. This command that we filter out would erase the machine's EEPROM, which is something which we are unlikely to want.

Again, we use the same method to encode the command:

```cpp
 bool JuraTermModule::toCoffeemaker(String cmd) {
    OtaCore::log(LOG_DEBUG, "Sending command: " + cmd);

    cmd += "\r\n";

    for (int i = 0; i < cmd.length(); ++i) {
          uint8_t ch = static_cast<uint8_t>(cmd[i]);
          uint8_t d0 = 0xFF;
          uint8_t d1 = 0xFF;
          uint8_t d2 = 0xFF;
          uint8_t d3 = 0xFF;

          bitWrite(d0, 2, bitRead(ch, 0));
          bitWrite(d0, 5, bitRead(ch, 1));
          bitWrite(d1, 2, bitRead(ch, 2));
          bitWrite(d1, 5, bitRead(ch, 3));
          bitWrite(d2, 2, bitRead(ch, 4));
          bitWrite(d2, 5, bitRead(ch, 5));
          bitWrite(d3, 2, bitRead(ch, 6)); 
          bitWrite(d3, 5, bitRead(ch, 7));

          delay(1); 
          Serial.write(d0);
          delay(1); 
          Serial.write(d1);
          delay(1); 
          Serial.write(d2);
          delay(1); 
          Serial.write(d3);
          delay(7);
    }     

    return true;
 }

 void JuraTermModule::onSerialReceived(Stream &stream, char arrivedChar, unsigned short availableCharsCount) {
    OtaCore::log(LOG_DEBUG, "Receiving UART 0.");

    while(stream.available()){
          delay(1);
          uint8_t d0 = stream.read();
          delay(1);
          uint8_t d1 = stream.read();
          delay(1);
          uint8_t d2 = stream.read();
          delay(1);
          uint8_t d3 = stream.read();
          delay(7);

          uint8_t d4;
          bitWrite(d4, 0, bitRead(d0, 2));
          bitWrite(d4, 1, bitRead(d0, 5));
          bitWrite(d4, 2, bitRead(d1, 2));
          bitWrite(d4, 3, bitRead(d1, 5));
          bitWrite(d4, 4, bitRead(d2, 2));
          bitWrite(d4, 5, bitRead(d2, 5));
          bitWrite(d4, 6, bitRead(d3, 2));
          bitWrite(d4, 7, bitRead(d3, 5));

          OtaCore::log(LOG_TRACE, String(d4));

          mqttTxBuffer += (char) d4;
          if ('\n' == (char) d4) {
                OtaCore::publish("coffee/response", OtaCore::getLocation() + ";" + mqttTxBuffer);
                mqttTxBuffer = "";
          }
    }
 }
```

Instead of interpreting the data in any way, we merely return the response on its respective MQTT topic.

# Motion

The motion module is intended to work with **passive infrared** (**PIR**) sensors. These have onboard logic that determine when a trigger point has been reached, at which point they change an interrupt pin into a high signal. We can use this to determine whether a person is in a room, or is walking through a hallway.

Its code looks as follows:

```cpp
#include "base_module.h"

 #define GPIO_PIN 0

 class MotionModule {
    static int pin;
    static Timer timer;
    static Timer warmup;
    static bool motion;
    static bool firstLow;

 public:
    static bool initialize();
    static bool start();
    static bool shutdown();
    static void config(String cmd);
    static void warmupSensor();
    static void readSensor();
    static void IRAM_ATTR interruptHandler();
 };
```

Of note here is that we explicitly move the interrupt handler method into the MCU's SRAM with the IRAM_ATTR keyword, to prevent any delay when the interrupt gets called.

Its implementation is as follows:

```cpp
#include "motion_module.h"
int MotionModule::pin = GPIO_PIN;
Timer MotionModule::timer;
Timer MotionModule::warmup;
bool MotionModule::motion = false;
bool MotionModule::firstLow = true;
bool MotionModule::initialize() {
      BaseModule::registerModule(MOD_IDX_MOTION, MotionModule::start, 
      MotionModule::shutdown);
}
bool MotionModule::start() {
    if (!OtaCore::claimPin(ESP8266_gpio00)) { return false; }
    pinMode(pin, INPUT);
    warmup.initializeMs(60000, MotionModule::warmupSensor).start();
   return true;
}
```

A PIR sensor requires warm-up time to stabilize its readings. We give it a minute using the warm-up timer. We also set the mode for the GPIO pin we're using.

```cpp

 bool MotionModule::shutdown() {
    if (!OtaCore::releasePin(ESP8266_gpio00)) { return false; } // RX 0

    timer.stop();
    detachInterrupt(pin);

    return true;
 }

 void MotionModule::config(String cmd) {
    Vector<String> output;
    int numToken = splitString(cmd, '=', output);
    if (output[0] == "set_pin" && numToken > 1) {
          //
    }
 }

 void MotionModule::warmupSensor() {
    warmup.stop();
    attachInterrupt(pin, &MotionModule::interruptHandler, CHANGE);

    timer.initializeMs(5000, MotionModule::readSensor).start();
 }
```

After the sensor has finished warming up, we stop its timer and attach the interrupt to handle any signals from the sensor. We'll check up on the shared variable with the interrupt routine, to see whether the value has changed, publishing the current value every 5 seconds:

```cpp
 void MotionModule::readSensor() {
    if (!motion) {
          if (firstLow) { firstLow = false; }
          else {
                OtaCore::publish("nsa/motion", OtaCore::getLocation() + ";0");
                firstLow = true;
          }
    }
    else if (motion) {
          OtaCore::publish("nsa/motion", OtaCore::getLocation() + ";1");
          firstLow = true;
    }
 }
```

When checking the current sensor value, we make it a point to ignore the first time that the sensor reports `LOW`. This in order to ensure that we ignore moments when people do not move a lot in the room. The resulting value is then published on the MQTT topic:

```cpp
void IRAM_ATTR MotionModule::interruptHandler() {
    int val = digitalRead(pin);
    if (val == HIGH) { motion = true; }
    else { motion = false; }
 }
```

The interrupt handler merely updates the local Boolean value. Because of the relatively long transition times for most processing circuits for PIR sensor there is quite a bit of time (seconds) before the sensor will detect motion again, creating dead zones. Here we keep track of the last registered value.

# PWM

The reason why the PWM module was developed was to have a way to generate an analog output voltage using an external RC filter circuit. This was in order to control the fan of the ceiling-mounted air-conditioning units, whose fan controller accepts a voltage of between 0 and 10 volts.

An interesting feature of this module is that it has its own binary protocol to allow for remote control, which is how the air-conditioning service can directly control the fan speeds via the ceiling-mounted nodes:

```cpp
#include "base_module.h"

 #include <HardwarePWM.h>

 class PwmModule {
    static HardwarePWM* hw_pwm;
    static Vector<int> duty;
    static uint8 pinNum;
    static Timer timer;
    static uint8* pins;

 public:
    static bool initialize();
    static bool start();
    static bool shutdown();
    static void commandCallback(String message);
 };
```

The implementation is as follows:

```cpp
#include "pwm_module.h"

 HardwarePWM* PwmModule::hw_pwm = 0;
 uint8 PwmModule::pinNum = 0;
 Timer PwmModule::timer;
 uint8* PwmModule::pins = 0;

 enum {
    PWM_START = 0x01,
    PWM_STOP = 0x02,
    PWM_SET_DUTY = 0x04,
    PWM_DUTY = 0x08,
    PWM_ACTIVE = 0x10
 };
```

We define the commands that will be available with the PWM module here as an enumeration:

```cpp

 bool PwmModule::initialize() {
    BaseModule::registerModule(MOD_IDX_PWM, PwmModule::start, PwmModule::shutdown);
 }

 bool PwmModule::start() {
    OtaCore::registerTopic(MQTT_PREFIX + String("pwm/") + OtaCore::getLocation(), PwmModule::commandCallback);

    return true;
 }

 bool PwmModule::shutdown() {
    OtaCore::deregisterTopic(MQTT_PREFIX + String("pwm/") + OtaCore::getLocation());

    if (hw_pwm) {
          delete hw_pwm;
          hw_pwm = 0;
    }

    return true;
 }
```

When we start this module, we register the MQTT topic on which the module will be able to receive commands. When shutting down, we deregister this topic again. We use the `HardwarePWM` class from Sming to enable PWM on individual pins.

The rest of the module is simply the command processor:

```cpp

 void PwmModule::commandCallback(String message) {
    OtaCore::log(LOG_DEBUG, "PWM command: " + message);
    if (message.length() < 1) { return; }
    int index = 0;
    uint8 cmd = *((uint8*) &message[index++]);

    if (cmd == PWM_START) {
          if (message.length() < 2) { return; }
          uint8 num = *((uint8*) &message[index++]);

          OtaCore::log(LOG_DEBUG, "Pins to add: " + String(num));

          if (message.length() != (2 + num)) { return; }

          pins = new uint8[num];
          for (int i = 0; i < num; ++i) {
                pins[i] = *((uint8*) &message[index++]);
                if (!OtaCore::claimPin(pins[i])) {
                      OtaCore::log(LOG_ERROR, "Pin is already in use: " + String(pins[i]));

                      OtaCore::publish("pwm/response", OtaCore::getLocation() + ";0", 1);

                      return; 
                }

                OtaCore::log(LOG_INFO, "Adding GPIO pin " + String(pins[i]));
          }

          hw_pwm = new HardwarePWM(pins, num);
          pinNum = num;

          OtaCore::log(LOG_INFO, "Added pins to PWM: " + String(pinNum));

          OtaCore::publish("pwm/response", OtaCore::getLocation() + ";1", 1);
    }
    else if (cmd == PWM_STOP) {
          delete hw_pwm;
          hw_pwm = 0;

          for (int i = 0; i < pinNum; ++i) {
                if (!OtaCore::releasePin(pins[i])) {
                      OtaCore::log(LOG_ERROR, "Pin cannot be released: " + String(pins[i]));

                      OtaCore::publish("pwm/response", OtaCore::getLocation() + ";0", 1);

                      return; 
                }

                OtaCore::log(LOG_INFO, "Removing GPIO pin " + String(pins[i]));
          }

          delete[] pins;
          pins = 0;

          OtaCore::publish("pwm/response", OtaCore::getLocation() + ";1");
    }
    else if (cmd == PWM_SET_DUTY) {
          if (message.length() < 3) { return; }

          uint8 pin = *((uint8*) &message[index++]);
          uint8 duty = *((uint8*) &message[index++]);
          bool ret = hw_pwm->setDuty(pin, ((uint32) 222.22 * duty));
          if (!ret) {
                OtaCore::publish("pwm/response", OtaCore::getLocation() + ";0");

                return;
          }

          OtaCore::publish("pwm/response", OtaCore::getLocation() + ";1");
    }
    else if (cmd == PWM_DUTY) {
          if (message.length() < 2) { return; }

          uint8 pin = *((uint8*) &message[index++]);
          uint32 duty = hw_pwm->getDuty(pin);

          uint8 dutyp = (duty / 222.22) + 1;
          String res = "";
          res += (char) pin;
          res += (char) dutyp;
          OtaCore::publish("pwm/response", OtaCore::getLocation() + ";" + res);
    }
    else if (cmd == PWM_ACTIVE) {
          String res;
          if (pins && pinNum > 0) {
                res = String((char*) pins, pinNum);
          }

          OtaCore::publish("pwm/response", OtaCore::getLocation() + ";" + res);
    }
 }
```

The protocol implemented by the preceding method is the following:

| **Command** | **Meaning** | **Payload** | **Return value** |
| 0x01 | Start the module | uint8 (number of pins)uint8* (one byte per pin number) | 0x00/0x01 |
| 0x02 | Stop the module | - | 0x00/0x01 |
| 0x04 | Set the PWM duty level | uint8 (pin number)uint8 (duty cycle, 0 - 100) | 0x00/0x01 |
| 0x08 | Get the PWM duty level | uint8 (pin number). | uint8 (duty level) |
| 0x10 | Returns the active pins | - | uint8* (one pin number per byte) |

For each command, we parse the string of bytes we receive, checking the number of bytes to see whether we get the expected number, and then interpreting them as commands and their payload. We either return a 0 (failure) or a 1 (success), or a payload with the desired information.

One obvious addition that could be made here would be to add some kind of checksum to the received command, along with sanity checks on the received data. While code like this will work great in a secure environment with encrypted MQTT links and a reliable network connection, other environments may be less forgiving, with corrupted data and false data being injected.

# I/O

Sometimes all we need is just a lot of GPIO pins that connect to things like relays, so that we can turn heating valves on or off. This was the reason behind this module. The nodes that were being installed on the ceiling had not just an I2C bus being used for the environmental sensors, but also the UART for CO[2] measurements and four pins for PWM output.

As more GPIO was needed to turn the relays that controlled the valves on the water lines to the air-conditioning units on or off, a dedicated GPIO expander chip was added to the I2C bus to provide eight more GPIO pins.

This module allows for an external service like the air-conditioning service to directly set these new GPIO pins as high or low:

```cpp
#include "base_module.h"

 #include <Libraries/MCP23008/MCP23008.h>

 class IOModule {
    static MCP23008* mcp;
    static uint8 iodir;
    static uint8 gppu;
    static uint8 gpio;
    static String publishTopic;

 public:
    static bool initialize();
    static bool start();
    static bool shutdown();
    static void commandCallback(String message);
 };
```

This class wraps the MCP23008 I/O expander device, keeping a local copy of its direction, pull-up, and GPIO state registers for easy updating and control:

```cpp
#include "io_module.h"

 #include <Wire.h>

 MCP23008* IOModule::mcp = 0;
 uint8 IOModule::iodir;     
 uint8 IOModule::gppu;
 uint8 IOModule::gpio;      
 String IOModule::publishTopic;
```

We keep a local copy of three registers on the I2C GPIO expander device—the I/O direction (`iodir`), pull-up register (`gppu`), and the pin I/O level (`gpio`):

```cpp

 enum {
    IO_START = 0x01,
    IO_STOP = 0x02,
    IO_STATE = 0x04,
    IO_SET_MODE = 0x08,
    IO_SET_PULLUP = 0x10,
    IO_WRITE = 0x20,
    IO_READ = 0x40,
    IO_ACTIVE = 0x80
 };

 enum {
    MCP_OUTPUT = 0,
    MCP_INPUT = 1
 };
```

We again define a number of commands in the form of an enumeration, along with one for the pin direction of the GPIO expander:

```cpp
bool IOModule::initialize() {
    BaseModule::registerModule(MOD_IDX_IO, IOModule::start, IOModule::shutdown);
 }

 bool IOModule::start() {   
    publishTopic = "io/response/" + OtaCore::getLocation();
    OtaCore::registerTopic("io/" + OtaCore::getLocation(), IOModule::commandCallback);

    OtaCore::starti2c();
 }

 bool IOModule::shutdown() {
    OtaCore::deregisterTopic("io/" + OtaCore::getLocation());
    if (mcp) {
          delete mcp;
          mcp = 0;
    }
 }
```

Initializing and starting the module is similar to the PWM module, with us registering an MQTT topic to receive commands on. The difference here is that since we are using an I2C device, we have to make sure that the I2C functionality has been started already.

Next, we address the command-processing method:

```cpp
void IOModule::commandCallback(String message) {
    OtaCore::log(LOG_DEBUG, "I/O command: " + message);
    uint32 mlen = message.length();
    if (mlen < 1) { return; }
    int index = 0;
    uint8 cmd = *((uint8*) &message[index++]);
    if (cmd == IO_START) {
        if (mlen > 2) {
            OtaCore::log(LOG_INFO, "Enabling I/O Module failed: too 
            many parameters.");
            OtaCore::publish(publishTopic, OtaCore::getLocation() + 
            ";" + (char) 0x01 + (char) 0x00);
            return; 
        }
        // Read out the desired address, or use the default.
        uint8 addr = 0;
        if (mlen == 2) {
            addr = *((uint8*) &message[index++]);
            if (addr > 7) {                     
            // Report failure. QoS 1.
            OtaCore::log(LOG_INFO, "Enabling I/O Module failed: invalid 
            i2c address.");
            OtaCore::publish(publishTopic, OtaCore::getLocation() + ";" 
            + (char) 0x01 + (char) 0x00);
            return;
        }
    }
    if (!mcp) {
        mcp = new MCP23008(0x40);
    }           
    // Set all pins to output (0) and low (0)
    mcp->writeIODIR(0x00);
    mcp->writeGPIO(0x00);
    // Read in current chip values.
    iodir = mcp->readIODIR();
    gppu = mcp->readGPPU();
    gpio = mcp->readGPIO();
    // Validate IODIR and GPIO registers.
    if (iodir != 0 || gpio != 0) {
        delete mcp;
        mcp = 0;
        OtaCore::log(LOG_INFO, "Enabling I/O Module failed: not 
        connected.");
         OtaCore::publish(publishTopic, OtaCore::getLocation() + ";" +
         (char) 0x01 + (char) 0x00);
         return;
    }
    OtaCore::log(LOG_INFO, "Enabled I/O Module.");
    OtaCore::publish(publishTopic, OtaCore::getLocation() + ";" +                                                                        
    (char) 0x01 + (char) 0x01);
}
    else if (cmd == IO_STOP) {
        if (mlen > 1) {
            OtaCore::log(LOG_INFO, "Disabling I/O Module failed: too 
            many parameters.");
            OtaCore::publish(publishTopic, OtaCore::getLocation() + ";" 
            + (char) 0x02 + (char) 0x00);
            return; 
        }
        if (mcp) {
            delete mcp;
            mcp = 0;
        }
        OtaCore::log(LOG_INFO, "Disabled I/O Module.");
        OtaCore::publish(publishTopic, OtaCore::getLocation() + ";" + 
        (char) 0x02 + (char) 0x01);
    }
    else if (cmd == IO_STATE) {
          if (mlen > 1) {
                OtaCore::log(LOG_INFO, "Reading state failed: too many parameters.");
                OtaCore::publish(publishTopic, OtaCore::getLocation() + ";" + 
                                                                      (char) 0x04 + (char) 0x00);
                return; 
          }

          OtaCore::publish(publishTopic, OtaCore::getLocation() + ";" + 
                                                                      (char) 0x04 + (char) 0x01 + 
                                                                      ((char) iodir) + ((char) gppu) +
                                                                      ((char) gpio));
    }
    else if (cmd == IO_SET_MODE) {
          if (mlen != 3) {
                OtaCore::log(LOG_INFO, "Reading state failed: incorrect number of parameters.");
                OtaCore::publish(publishTopic, OtaCore::getLocation() + ";" + 
                                                                      (char) 0x08 + (char) 0x00);
                return; 
          }

          uint8 pnum = *((uint8*) &message[index++]);
          uint8 pstate = *((uint8*) &message[index]);
          if (pnum > 7) {
                OtaCore::log(LOG_INFO, "Setting pin mode failed: unknown pin.");
                OtaCore::publish(publishTopic, OtaCore::getLocation() + ";" + 
                                                                      (char) 0x08 + (char) 0x00);
                return; 
          }

          if (pstate > 1) {
                // Report failure. QoS 1.
                OtaCore::log(LOG_INFO, "Setting pin mode failed: invalid pin mode.");
                OtaCore::publish(publishTopic, OtaCore::getLocation() + ";" + 
                                                                      (char) 0x08 + (char) 0x00);
                return; 
          }

          // Set new state of IODIR register.
          if (pstate == MCP_INPUT) { iodir |= 1 << pnum; } 
          else { iodir &= ~(1 << pnum); }

          if (mcp) {
                OtaCore::log(LOG_DEBUG, "Setting pinmode in library...");
                mcp->writeIODIR(iodir);
          }

          OtaCore::log(LOG_INFO, "Set pin mode for I/O Module.");
          OtaCore::publish(publishTopic, OtaCore::getLocation() + ";" + 
                                                                      (char) 0x08 + (char) 0x01);
    }
    else if (cmd == IO_SET_PULLUP) {          
          if (mlen != 3) {
                OtaCore::log(LOG_INFO, "Reading state failed: incorrect number of parameters.");
                OtaCore::publish(publishTopic, OtaCore::getLocation() + ";" + 
                                                                      (char) 0x10 + (char) 0x00);
                return; 
          }

          uint8 pnum = *((uint8*) &message[index++]);
          uint8 pstate = *((uint8*) &message[index]);
          if (pnum > 7) {
                OtaCore::log(LOG_INFO, "Setting pull-up failed: unknown pin.");
                OtaCore::publish(publishTopic, OtaCore::getLocation() + ";" + 
                                                                      (char) 0x10 + (char) 0x00);
                return; 
          }

          if (pstate > 1) {
                OtaCore::log(LOG_INFO, "Setting pull-up failed: invalid state.");
                OtaCore::publish(publishTopic, OtaCore::getLocation() + ";" + 
                                                                      (char) 0x10 + (char) 0x00);
                return; 
          }

          if (pstate == HIGH) { gppu |= 1 << pnum; } 
          else { gppu &= ~(1 << pnum); }

          if (mcp) {
                OtaCore::log(LOG_DEBUG, "Setting pull-up in library...");
                mcp->writeGPPU(gppu);
          }

          OtaCore::log(LOG_INFO, "Changed pull-up for I/O Module.");
          OtaCore::publish(publishTopic, OtaCore::getLocation() + ";" + 
                                                                      (char) 0x10 + (char) 0x01);
    }
    else if (cmd == IO_WRITE) {
          if (mlen != 3) {
                OtaCore::log(LOG_INFO, "Writing pin failed: incorrect number of parameters.");
                OtaCore::publish(publishTopic, OtaCore::getLocation() + ";" + 
                                                                      (char) 0x20 + (char) 0x00);
                return; 
          }
          // Set the new GPIO pin level.
          uint8 pnum = *((uint8*) &message[index++]);
          uint8 pstate = *((uint8*) &message[index]);
          if (pnum > 7) {
                OtaCore::log(LOG_INFO, "Writing pin failed: unknown pin.");
                OtaCore::publish(publishTopic, OtaCore::getLocation() + ";" + 
                                                                      (char) 0x20 + (char) 0x00);
                return; 
          }
          if (pstate > 1) {
                OtaCore::log(LOG_INFO, "Writing pin failed: invalid state.");
                OtaCore::publish(publishTopic, OtaCore::getLocation() + ";" + 
                                                                      (char) 0x20 + (char) 0x00);
                return; 
          }
          String state = "low";
          if (pstate == HIGH) { gpio |= 1 << pnum; state = "high"; } 
          else { gpio &= ~(1 << pnum); }

          OtaCore::log(LOG_DEBUG, "Changed GPIO to: " + ((char) gpio));

          if (mcp) {
                OtaCore::log(LOG_DEBUG, "Setting state to " + state + 
                                        " in library for pin " + ((char) pnum));
                mcp->writeGPIO(gpio);
          }

          OtaCore::log(LOG_INFO, "Wrote pin state for I/O Module.");
          OtaCore::publish(publishTopic, OtaCore::getLocation() + ";" + 
                                                                      (char) 0x20 + (char) 0x01);
    }

    else if (cmd == IO_READ) {

          if (mlen > 2) {
                OtaCore::log(LOG_INFO, "Reading pin failed: too many 
                parameters.");
                OtaCore::publish(publishTopic, OtaCore::getLocation()
                                                                     (char) 0x40 + (char) 0x00);
                return; 
          }
          // Read the GPIO pin status and return it.
          uint8 pnum = *((uint8*) &message[index]);

        if (pnum > 7) {
            OtaCore::log(LOG_INFO, "Reading pin failed: unknown pin.");
            OtaCore::publish(publishTopic, OtaCore::getLocation() + ";" 
            + (char) 0x40 + (char) 0x00);
        }
          uint8 pstate;

        if (mcp) {
            OtaCore::log(LOG_DEBUG, "Reading pin in library...");
            pstate = (mcp->readGPIO() >> pnum) & 0x1;
        }
        OtaCore::log(LOG_INFO, "Read pin state for I/O Module.");
        OtaCore::publish(publishTopic, OtaCore::getLocation() + ";" + 
        (char) 0x40 + (char) 0x01 + (char) pnum + (char) pstate);
    }

    else if (cmd == IO_ACTIVE) {

        if (mlen > 1) {
            OtaCore::log(LOG_INFO, "Reading active status failed: too 
            many parameters.");
            OtaCore::publish(publishTopic, OtaCore::getLocation() + 
            ";" + (char) 0x80 + (char) 0x00);
            return; 
        }
        uint8 active = 0;
        if (mcp) { active = 1; }
        char output[] = { 0x80, 0x01, active };
        OtaCore::publish(publishTopic, OtaCore::getLocation() + ";" + 
        String(output, 3));
    }
}
```

Its protocol looks as follows:

| **Command** | **Meaning** | **Payload** | **Return value** |
| 0x01 | Start the module | uint8 I2C address offset (0-7, optional) | 0x010x00/0x01 |
| 0x02 | Stop the module | - | 0x020x00/0x01 |
| 0x04 | Returns I/O mode, pull-up, and level state | - | 0x040x00/0x01 (result)uint8 (iodir register)uint8 (gppu register)uint8 (gpio register) |
| 0x08 | Set a pin to a specific mode (In/Out) | uint8 (pin number, 0 - 7)uint8 (0: output, 1: input) | 0x080x00/0x01 |
| 0x10 | Set a pin's pull-up resistor (Low/High) | uint8 (pin number, 0 - 7)uint8 (pin pull-up state, 0/1) | 0x100x00/0x01 |
| 0x20 | Set a pin to either Low or High | uint8 (pin number, 0-7)uint8 (pin state, 0/1) | 0x20 0x00/0x01 |
| 0x40 | Read the current pin value (Low, High) | uint8 (pin number) | 0x40 0x00/0x01 uint8 (pin number) uint8 (pin value) |
| 0x80 | Return whether this module has been initialized | - | 0x80 0x00/0x01 uint8 (module state, 0/1). |

Similar to the protocol for the PWM module, either a Boolean value is returned to indicate success, or the requested payload is returned. We also return the command that was called in the response.

The command is a single byte, allowing for a maximum of eight commands since we are using bit flags. This could be extended to 256 commands if we wanted to.

Possible improvements to this module's code include consolidating duplicated code into (inline) function calls and conceivably the use of a sub-class that would manage the setting and toggling of individual bits with a more higher-level API.

# Switch

Since each section of the office had its own central switch that would switch the water in the pipes that flowed to the FCUs, this had to be controllable from the backend server as well. Using a latching relay configuration, it was possible to both switch between heating and cooling configurations, as well as have a memory element that could be read out by the node:

This system was assembled on a single board that was used to replace the original manual switch, using the following module to control it:

```cpp
#include "base_module.h"

 class SwitchModule {
    static String publishTopic;

 public:
    static bool initialize();
    static bool start();
    static bool shutdown();
    static void commandCallback(String message);
 };
```

Its implementation is as follows:

```cpp
#include "switch_module.h"
#include <Wire.h>
#define SW1_SET_PIN 5 
#define SW2_SET_PIN 4 
#define SW1_READ_PIN 14 
#define SW2_READ_PIN 12 
String SwitchModule::publishTopic;
enum {
    SWITCH_ONE = 0x01,//Switch the first connected load on, second off.
    SWITCH_TWO = 0x02,//Switch the second connected load on, first off.
    SWITCH_STATE = 0x04,//Returns position of the switch (0x01/0x02).
};
bool SwitchModule::initialize() {
    BaseModule::registerModule(MOD_IDX_SWITCH, SwitchModule::start, 
    SwitchModule::shutdown);
}
bool SwitchModule::start() {
    // Register pins.
    if (!OtaCore::claimPin(ESP8266_gpio05)) { return false; }
    if (!OtaCore::claimPin(ESP8266_gpio04)) { return false; }
    if (!OtaCore::claimPin(ESP8266_gpio14)) { return false; }
    if (!OtaCore::claimPin(ESP8266_gpio12)) { return false; }
    publishTopic = "switch/response/" + OtaCore::getLocation();
    OtaCore::registerTopic("switch/" + OtaCore::getLocation(), 
    SwitchModule::commandCallback);
// Set the pull-ups on the input pins and configure the output pins.
    pinMode(SW1_SET_PIN, OUTPUT);
    pinMode(SW2_SET_PIN, OUTPUT);
    pinMode(SW1_READ_PIN, INPUT_PULLUP);
    pinMode(SW2_READ_PIN, INPUT_PULLUP);
    digitalWrite(SW1_SET_PIN, LOW);
    digitalWrite(SW2_SET_PIN, LOW);
 }
 bool SwitchModule::shutdown() {
    OtaCore::deregisterTopic("switch/" + OtaCore::getLocation());
    // Release the pins.
    if (!OtaCore::releasePin(ESP8266_gpio05)) { return false; }
    if (!OtaCore::releasePin(ESP8266_gpio04)) { return false; }
    if (!OtaCore::releasePin(ESP8266_gpio14)) { return false; }
    if (!OtaCore::releasePin(ESP8266_gpio12)) { return false; }
 }

 void SwitchModule::commandCallback(String message) {
    // Message is the command.
    OtaCore::log(LOG_DEBUG, "Switch command: " + message);

    uint32 mlen = message.length();
    if (mlen < 1) { return; }
    int index = 0;
    uint8 cmd = *((uint8*) &message[index++]);
    if (cmd == SWITCH_ONE) {
          if (mlen > 1) {
                // Report failure. QoS 1.
                OtaCore::log(LOG_INFO, "Switching to position 1 failed: too many parameters.");
                OtaCore::publish(publishTopic, OtaCore::getLocation() + ";" + 
                                                                      (char) 0x01 + (char) 0x00);
                return; 
          }

          // Set the relay to its first position (reset condition).
          // This causes pins 3 & 10 on the latching relay to become active.
          digitalWrite(SW1_SET_PIN, HIGH);
          delay(1000); // Wait 1 second for the relay to switch position.
          digitalWrite(SW1_SET_PIN, LOW);

          OtaCore::log(LOG_INFO, "Switched to position 1.");
          OtaCore::publish(publishTopic, OtaCore::getLocation() + ";" + 
                                                                      (char) 0x01 + (char) 0x01);
    }
    else if (cmd == SWITCH_TWO) {
          if (mlen > 1) {
                OtaCore::log(LOG_INFO, "Switching to position 2 failed: too many parameters.");
                OtaCore::publish(publishTopic, OtaCore::getLocation() + ";" + 
                                                                      (char) 0x02 + (char) 0x00);
                return; 
          }

          // Set the relay to its first position (reset condition).
          // This causes pins 3 & 10 on the latching relay to become active.
          digitalWrite(SW2_SET_PIN, HIGH);
          delay(1000); // Wait 1 second for the relay to switch position.
          digitalWrite(SW2_SET_PIN, LOW);

          OtaCore::log(LOG_INFO, "Switched to position 1.");
          OtaCore::publish(publishTopic, OtaCore::getLocation() + ";" + 
                                                                      (char) 0x02 + (char) 0x01);
    }
    else if (cmd == SWITCH_STATE) {
          if (mlen > 1) {
                OtaCore::log(LOG_INFO, "Reading state failed: too many parameters.");
                OtaCore::publish(publishTopic, OtaCore::getLocation() + ";" + 
                                                                      (char) 0x04 + (char) 0x00);
                return; 
          }

          // Check the value of the two input pins. If one is low, then that
          // is the active position.
          uint8 active = 2;
          if (digitalRead(SW1_READ_PIN) == LOW) { active = 0; }
          else if (digitalRead(SW2_READ_PIN) == LOW) { active = 1; }

          if (active > 1) {
                OtaCore::log(LOG_INFO, "Reading state failed: no active state found.");
                OtaCore::publish(publishTopic, OtaCore::getLocation() + ";" + 
                                                                      (char) 0x04 + (char) 0x00);
                return; 
          }

          OtaCore::publish(publishTopic, OtaCore::getLocation() + ";" + 
                                                                      (char) 0x04 + (char) 0x01 + 
                                                                      (char) active);
    }
 }
```

This module is very similar to the PWM and I/O modules, with the registering of an MQTT topic to allow communication using its own binary protocol. Here, the device that is being controlled is fairly simple. It is a latching relay with two sides, one of which is connected to the connections that are being switched between, while the other side is used as a one-bit memory cell.

As both sides of this type of relay will switch simultaneously, we can count on the side connected to the MCU to match the position of that on the side connected to the rest of the system. Even after a power failure or reset of the MCU, we can simply read out the values of the pins connected to the relay to find out the state of the system.

The resulting protocol looks like this:

| **Command** | **Meaning** | **Payload** | **Return value** |
| 0x01 | Switch to Position 1 | - | 0x010x00/0x01 |
| 0x02 | Switch to Position 2 | - | 0x020x00/0x01 |
| 0x04 | Return the current state | - | 0x040x00/0x01 (result)uint8 (active pin 0x00, 0x01) |

# Command and control server

As alluded to earlier in this chapter, a so-called **command and control** (**C&C**) server is essentially a database containing information on individual nodes and their configuration, for use by the nodes themselves and administration tools like the one in the next section.

It also includes an HTTP server, for use with HTTP-based **over-the-air** (**OTA**) updates. Since the BMaC system is MQTT-based, this server is also written as an MQTT client:

```cpp
#include "listener.h"  
#include <iostream> 
#include <string> 

using namespace std; 

#include <Poco/Util/IniFileConfiguration.h> 
#include <Poco/AutoPtr.h> 
#include <Poco/Net/HTTPServer.h> 

using namespace Poco::Util; 
using namespace Poco; 
using namespace Poco::Net; 

#include "httprequestfactory.h" 

int main(int argc, char* argv[]) { 
   cout << "Starting MQTT BMaC Command & Control server...\n"; 

   int rc; 
   mosqpp::lib_init(); 

   cout << "Initialised C++ Mosquitto library.\n"; 

   string configFile; 
   if (argc > 1) { configFile = argv[1]; } 
   else { configFile = "config.ini"; } 

   AutoPtr<IniFileConfiguration> config(new IniFileConfiguration(configFile)); 
   string mqtt_host = config->getString("MQTT.host", "localhost"); 
   int mqtt_port = config->getInt("MQTT.port", 1883); 
   string defaultFirmware = config->getString("Firmware.default", "ota_unified.bin"); 

   Listener listener("Command_and_Control", mqtt_host, mqtt_port, defaultFirmware); 

   UInt16 port = config->getInt("HTTP.port", 8080); 
   HTTPServerParams* params = new HTTPServerParams; 
   params->setMaxQueued(100); 
   params->setMaxThreads(10); 
   HTTPServer httpd(new RequestHandlerFactory, port, params); 
   httpd.start(); 

   cout << "Created listener, entering loop...\n"; 

   while(1) { 
         rc = listener.loop(); 
         if (rc){ 
               cout << "Disconnected. Trying to reconnect...\n"; 
               listener.reconnect(); 
         } 
   } 

   cout << "Cleanup...\n"; 

   mosqpp::lib_cleanup(); 

   return 0; 
} 
```

We're using the Mosquitto C++ MQTT client along with the POCO framework to provide us with the required functionality.

The `Listener` class is next:

```cpp
#include <mosquittopp.h> 
#include <string> 

using namespace std; 

#include <Poco/Data/Session.h> 
#include <Poco/Data/SQLite/Connector.h> 

using namespace Poco; 

class Listener : public mosqpp::mosquittopp { 
   Data::Session* session; 
   string defaultFirmware; 

public: 
   Listener(string clientId, string host, int port, string defaultFirmware); 
   ~Listener(); 

   void on_connect(int rc); 
   void on_message(const struct mosquitto_message* message); 
   void on_subscribe(int mid, int qos_count, const int* granted_qos); 
}; 
```

We include the headers from POCO for the SQLite database functionality, which forms the database backend for this application. The class itself derives from the Mosquitto C++ class, providing us with all the basic MQTT functionalities along with a few function stubs, which we still have to implement in a moment:

```cpp
#include "listener.h" 

#include <iostream> 
#include <fstream> 
#include <sstream> 

using namespace std; 

#include <Poco/StringTokenizer.h> 
#include <Poco/String.h> 
#include <Poco/Net/HTTPSClientSession.h> 
#include <Poco/Net/HTTPRequest.h> 
#include <Poco/Net/HTTPResponse.h> 
#include <Poco/File.h> 

using namespace Poco::Data::Keywords; 

struct Node { 
   string uid; 
   string location; 
   UInt32 modules; 
   float posx; 
   float posy; 
}; 
```

We define a structure for a single node:

```cpp
Listener::Listener(string clientId, string host, int port, string defaultFirmware) : mosquittopp(clientId.c_str()) { 
   int keepalive = 60; 
   connect(host.c_str(), port, keepalive); 

   Data::SQLite::Connector::registerConnector(); 
   session = new Poco::Data::Session("SQLite", "nodes.db"); 

   (*session) << "CREATE TABLE IF NOT EXISTS nodes (uid TEXT UNIQUE, \ 
         location TEXT, \ 
         modules INT, \ 
         posx FLOAT, \ 
         posy FLOAT)", now; 

   (*session) << "CREATE TABLE IF NOT EXISTS firmware (uid TEXT UNIQUE, \ 
         file TEXT)", now; 

   this->defaultFirmware = defaultFirmware; 
} 
```

In the constructor, we attempt to connect to the MQTT broker, using the provided host and port. We also set up a connection with the SQLite database, and ensure that it has valid nodes and a firmware table:

```cpp
Listener::~Listener() { 
   // 
} 

void Listener::on_connect(int rc) { 
   cout << "Connected. Subscribing to topics...\n"; 

   if (rc == 0) { 
         string topic = "cc/config";   // announce by nodes coming online. 
         subscribe(0, topic.c_str()); 
         topic = "cc/ui/config";       // C&C client requesting configuration. 
         subscribe(0, topic.c_str()); 
         topic = "cc/nodes/new";       // C&C client adding new node. 
         subscribe(0, topic.c_str()); 
         topic = "cc/nodes/update";    // C&C client updating node. 
         subscribe(0, topic.c_str()); 
         topic = "nsa/events/CO2";     // CO2-related events. 
         subscribe(0, topic.c_str()); 
         topic = "cc/firmware";  // C&C client firmware command. 
         subscribe(0, topic.c_str()); 
   } 
   else { 
         cerr << "Connection failed. Aborting subscribing.\n"; 
   } 
} 
```

We reimplement the callback for when a connection has been established with the MQTT broker. In this method, we subscribe to all the MQTT topics in which we are interested.

The next method is called whenever we receive an MQTT message on one of the topics which we subscribed to:

```cpp
void Listener::on_message(const struct mosquitto_message* message) { 
   string topic = message->topic; 
   string payload = string((const char*) message->payload, message->payloadlen); 

   if (topic == "cc/config") { 
         if (payload.length() < 1) { 
               cerr << "Invalid payload: " << payload << ". Reject.\n"; 
               return; 
         } 
```

We validate the payload we receive for each topic. For this first topic, we expect its payload to contain the MAC address of the node which wants to receive its configuration. We make sure that this seems to be the case, then continue:

```cpp
         Data::Statement select(*session); 
         Node node; 
         node.uid = payload; 
         select << "SELECT location, modules FROM nodes WHERE uid=?", 
                     into (node.location), 
                     into (node.modules), 
                     use (payload); 

         size_t rows = select.execute(); 

         if (rows == 1) { 
               string topic = "cc/" + payload; 
               string response = "mod;" + string((const char*) &node.modules, 4); 
               publish(0, topic.c_str(), response.length(), response.c_str()); 
               response = "loc;" + node.location; 
               publish(0, topic.c_str(), response.length(), response.c_str()); 
         } 
         else if (rows < 1) { 
               // No node with this UID found. 
               cerr << "Error: No data set found for uid " << payload << endl; 
         } 
         else { 
               // Multiple data sets were found, which shouldn't be possible... 
               cerr << "Error: Multiple data sets found for uid " << payload << "\n"; 
         } 
   } 
```

We attempt to find the MAC address in the database, reading out the node's configuration if found and making it the payload for the return message.

The next topics are used with the administration tool:

```cpp
else if (topic == "cc/ui/config") { 

    if (payload == "map") {

        ifstream mapFile("map.png", ios::binary); 

        if (!mapFile.is_open()) { 

            cerr << "Failed to open map file.\n"; 

            return; 

        } 

        stringstream ss; 

        ss << mapFile.rdbuf(); 

        string mapData = ss.str(); 

        publish(0, "cc/ui/config/map", mapData.length(), 

        mapData.c_str()); 

} 
```

In the case of this payload string, we return the binary data for a map image that should exist in the local folder. This map contains the layout of the building we are administrating, for displaying in the tool. 

```cpp
         else if (payload == "nodes") { 
               Data::Statement countQuery(*session); 
               int rowCount; 
               countQuery << "SELECT COUNT(*) FROM nodes", 
                     into(rowCount), 
                     now; 

               if (rowCount == 0) { 
                     cout << "No nodes found in database, returning...\n"; 
                     return; 
               } 

               Data::Statement select(*session); 
               Node node; 
               select << "SELECT uid, location, modules, posx, posy FROM nodes", 
                           into (node.uid), 
                           into (node.location), 
                           into (node.modules), 
                           into (node.posx), 
                           into (node.posy), 
                           range(0, 1); 

               string header; 
               string nodes; 
               string nodeStr; 
               UInt32 nodeCount = 0; 
               while (!select.done()) { 
                     select.execute(); 
                     nodeStr = "NODE"; 
                     UInt8 length = (UInt8) node.uid.length(); 
                     nodeStr += string((char*) &length, 1); 
                     nodeStr += node.uid; 
                     length = (UInt8) node.location.length(); 
                     nodeStr += string((char*) &length, 1); 
                     nodeStr += node.location; 
                     nodeStr += string((char*) &node.posx, 4); 
                     nodeStr += string((char*) &node.posy, 4); 
                     nodeStr += string((char*) &node.modules, 4); 
                     UInt32 segSize = nodeStr.length(); 

                     nodes += string((char*) &segSize, 4); 
                     nodes += nodeStr; 
                     ++nodeCount; 
               } 

               UInt64 messageSize = nodes.length() + 9; 
               header = string((char*) &messageSize, 8); 
               header += "NODES"; 
               header += string((char*) &nodeCount, 4); 
               header += nodes; 

               publish(0, "cc/nodes/all", header.length(), header.c_str()); 
         } 
   } 
```

The preceding section reads out every single node in the database and returns it in a binary, serialized format.

Next, we create a new node and add it to the database:

```cpp
   else if (topic == "cc/nodes/new") { 
         UInt32 index = 0; 
         UInt32 msgLength = *((UInt32*) payload.substr(index, 4).data()); 
         index += 4; 
         string signature = payload.substr(index, 4); 
         index += 4; 

         if (signature != "NODE") { 
               cerr << "Invalid node signature.\n"; 
               return; 
         } 

         UInt8 uidLength = (UInt8) payload[index++]; 
         Node node; 
         node.uid = payload.substr(index, uidLength); 
         index += uidLength; 
         UInt8 locationLength = (UInt8) payload[index++]; 
         node.location = payload.substr(index, locationLength); 
         index += locationLength; 
         node.posx = *((float*) payload.substr(index, 4).data()); 
         index += 4; 
         node.posy = *((float*) payload.substr(index, 4).data()); 
         index += 4; 
         node.modules = *((UInt32*) payload.substr(index, 4).data()); 

         cout << "Storing new node for UID: " << node.uid << "\n"; 

         Data::Statement insert(*session); 
         insert << "INSERT INTO nodes VALUES(?, ?, ?, ?, ?)", 
                     use(node.uid), 
                     use(node.location), 
                     use(node.modules), 
                     use(node.posx), 
                     use(node.posy), 
                     now; 

         (*session) << "INSERT INTO firmware VALUES(?, ?)", 
                     use(node.uid), 
                     use(defaultFirmware), 
                     now; 
   } 
```

Updating a node's configuration is also possible:

```cpp
   else if (topic == "cc/nodes/update") { 
         UInt32 index = 0; 
         UInt32 msgLength = *((UInt32*) payload.substr(index, 4).data()); 
         index += 4; 
         string signature = payload.substr(index, 4); 
         index += 4; 

         if (signature != "NODE") { 
               cerr << "Invalid node signature.\n"; 
               return; 
         } 

         UInt8 uidLength = (UInt8) payload[index++]; 
         Node node; 
         node.uid = payload.substr(index, uidLength); 
         index += uidLength; 
         UInt8 locationLength = (UInt8) payload[index++]; 
         node.location = payload.substr(index, locationLength); 
         index += locationLength; 
         node.posx = *((float*) payload.substr(index, 4).data()); 
         index += 4; 
         node.posy = *((float*) payload.substr(index, 4).data()); 
         index += 4; 
         node.modules = *((UInt32*) payload.substr(index, 4).data()); 

         cout << "Updating node for UID: " << node.uid << "\n"; 

         Data::Statement update(*session); 
         update << "UPDATE nodes SET location = ?, posx = ?, posy = ?, modules = ? WHERE uid = ?", 
                     use(node.location), 
                     use(node.posx), 
                     use(node.posy), 
                     use(node.modules), 
                     use(node.uid), 
                     now; 
   } 
```

Next, we look at the topic handler for deleting a node's configuration:

```cpp
   else if (topic == "cc/nodes/delete") { 
         cout << "Deleting node with UID: " << payload << "\n"; 

         Data::Statement del(*session); 
         del << "DELETE FROM nodes WHERE uid = ?", 
                     use(payload), 
                     now; 

         (*session) << "DELETE FROM firmware WHERE uid = ?", 
                     use(payload), 
                     now; 
   } 
```

When we looked at the CO[2] module of the firmware earlier, we saw that it generated CO[2] events. These also end up here in this example, in order to generate events in JSON format, which we send to some HTTP-based API. We then use the HTTPS client in POCO to send this JSON to the remote server (here set to localhost):

```cpp
   else if (topic == "nsa/events/CO2") { 
         StringTokenizer st(payload, ";", StringTokenizer::TOK_TRIM | StringTokenizer::TOK_IGNORE_EMPTY); 
         if (st.count() < 4) { 
               cerr << "CO2 event: Wrong number of arguments. Payload: " << payload << "\n"; 
               return; 
         } 

         string state = "ok"; 
         if (st[1] == "1") { state = "warn"; } 
         else if (st[1] == "2") { state = "crit"; } 
         string increase = (st[2] == "1") ? "true" : "false"; 
         string json = "{ \"state\": \"" + state + "\", \ 
                                 \"location\": \"" + st[0] + "\", \ 
                                 \"increase\": " + increase + ", \ 
                                 \"ppm\": " + st[3] + " }"; 

         Net::HTTPSClientSession httpsClient("localhost"); 
         try { 
               Net::HTTPRequest request(Net::HTTPRequest::HTTP_POST, 
                                                   "/", 
                                                   Net::HTTPMessage::HTTP_1_1); 
               request.setContentLength(json.length()); 
               request.setContentType("application/json"); 
               httpsClient.sendRequest(request) << json; 

               Net::HTTPResponse response; 
               httpsClient.receiveResponse(response); 
         } 
         catch (Exception& exc) { 
               cout << "Exception caught while attempting to connect." << std::endl; 
               cerr << exc.displayText() << std::endl; 
               return; 
         } 
   } 
```

Finally, for managing the stored firmware images, we can use the following topic. Which node uses which firmware version can be set in each node's configuration, though as we saw earlier, the default is to use the latest firmware.

Using this topic, we can list the available firmware images or upload a new one:

```cpp
   else if (topic == "cc/firmware") { 
         if (payload == "list") { 
               std::vector<File> files; 
               File file("firmware"); 
               if (!file.isDirectory()) { return; } 

               file.list(files); 
               string out; 
               for (int i = 0; i < files.size(); ++i) { 
                     if (files[i].isFile()) { 
                           out += files[i].path(); 
                           out += ";"; 
                     } 
               } 

               out.pop_back(); 

               publish(0, "cc/firmware/list", out.length(), out.c_str()); 
         } 
         else { 
               StringTokenizer st(payload, ";", StringTokenizer::TOK_TRIM | StringTokenizer::TOK_IGNORE_EMPTY); 

               if (st[0] == "change") { 
                     if (st.count() != 3) { return; } 
                     (*session) << "UPDATE firmware SET file = ? WHERE uid = ?", 
                                             use (st[1]), 
                                             use (st[2]), 
                                             now; 
               } 
               else if (st[0] == "upload") { 
                     if (st.count() != 3) { return; } 

                     // Write file & truncate if exists. 
                     string filepath = "firmware/" + st[1];                       
                     ofstream outfile("firmware/" + st[1], ofstream::binary | ofstream::trunc); 
                     outfile.write(st[2].data(), st[2].size()); 
                     outfile.close(); 
               } 
         } 
   } 
} 
void Listener::on_subscribe(int mid, int qos_count, const int* granted_qos) { 
   // 
} 
```

On each successful MQTT topic subscription, this method is called, allowing us to do something else if needed.

Next, we look at the HTTP server component, starting with the HTTP request handler factory:

```cpp
#include <Poco/Net/HTTPRequestHandlerFactory.h> 
#include <Poco/Net/HTTPServerRequest.h> 

using namespace Poco::Net; 

#include "datahandler.h" 

class RequestHandlerFactory: public HTTPRequestHandlerFactory { 
public: 
   RequestHandlerFactory() {} 
   HTTPRequestHandler* createRequestHandler(const HTTPServerRequest& request) { 
         return new DataHandler(); 
   } 
}; 
```

This handler will always return an instance of the following class:

```cpp
#include <iostream> 
#include <vector> 

using namespace std; 

#include <Poco/Net/HTTPRequestHandler.h> 
#include <Poco/Net/HTTPServerResponse.h> 
#include <Poco/Net/HTTPServerRequest.h> 
#include <Poco/URI.h> 
#include <Poco/File.h> 

#include <Poco/Data/Session.h> 
#include <Poco/Data/SQLite/Connector.h> 

using namespace Poco::Data::Keywords; 

using namespace Poco::Net; 
using namespace Poco; 

class DataHandler: public HTTPRequestHandler { 
public: 
   void handleRequest(HTTPServerRequest& request, HTTPServerResponse& response) { 
         cout << "DataHandler: Request from " + request.clientAddress().toString() << endl; 

         URI uri(request.getURI()); 
         string path = uri.getPath(); 
         if (path != "/") { 
               response.setStatus(HTTPResponse::HTTP_NOT_FOUND); 
               ostream& ostr = response.send(); 
               ostr << "File Not Found: " << path; 
               return; 
         } 

         URI::QueryParameters parts; 
         parts = uri.getQueryParameters(); 
         if (parts.size() > 0 && parts[0].first == "uid") { 
               Data::SQLite::Connector::registerConnector(); 
               Data::Session* session = new Poco::Data::Session("SQLite", "nodes.db"); 

               Data::Statement select(*session); 
               string filename; 
               select << "SELECT file FROM firmware WHERE uid=?", 
                                 into (filename), 
                                 use (parts[0].second); 

               size_t rows = select.execute(); 

               if (rows != 1) { 
                     response.setStatus(HTTPResponse::HTTP_NOT_FOUND); 
                     ostream& ostr = response.send(); 
                     ostr << "File Not Found: " << parts[0].second; 
                     return; 
               } 

               string fileroot = "firmware/"; 
               File file(fileroot + filename); 

               if (!file.exists() || file.isDirectory()) { 
                     response.setStatus(HTTPResponse::HTTP_NOT_FOUND); 
                     ostream& ostr = response.send(); 
                     ostr << "File Not Found."; 
                     return; 
               } 

               string mime = "application/octet-stream"; 
               try { 
                     response.sendFile(file.path(), mime); 
               } 
               catch (FileNotFoundException &e) { 
                     cout << "File not found exception triggered..." << endl; 
                     cerr << e.displayText() << endl; 

                     response.setStatus(HTTPResponse::HTTP_NOT_FOUND); 
                     ostream& ostr = response.send(); 
                     ostr << "File Not Found."; 
                     return; 
               } 
               catch (OpenFileException &e) { 
                     cout << "Open file exception triggered..." << endl; 
                     cerr << e.displayText() << endl; 

                     response.setStatus(HTTPResponse::HTTP_INTERNAL_SERVER_ERROR); 
                     ostream& ostr = response.send(); 
                     ostr << "Internal Server Error. Couldn't open file."; 
                     return; 
               } 
         } 
         else { 
               response.setStatus(HTTPResponse::HTTP_BAD_REQUEST); 
               response.send(); 
               return; 
         } 
   } 
}; 
```

This class looks fairly impressive, yet mostly does just an SQLite database lookup for the node ID (MAC address) and returns the appropriate firmware image if found.

# Administration tool

Using the APIs implemented by the C&C server, a GUI-based administration tool was created using the Qt5 framework and the Mosquitto MQTT client library was developed, allowing for the basic management of nodes. They were overlaid on top of a layout graphic of buildings.

While basically usable, it was found that a graphical tool was fairly complicated to develop. It was also limited to a single floor of a building, unless one were to have a really large map containing all of the floors with the nodes mapped onto this. This would have been quite clumsy, obviously.

In the source code provided with this chapter, the administration tool can be found as well, to serve as an example of how one could implement it. For the sake of brevity, the code for it has been omitted here.

# Air-conditioning service

To control air-conditioning units, a service much like the C&C one was developed, using the same basic template. The interesting parts of its source are the following:

```cpp
#include <string>
 #include <vector>

 using namespace std;

 #include <Poco/Data/Session.h>
 #include <Poco/Data/SQLite/Connector.h>

 #include <Poco/Net/HTTPClientSession.h>
 #include <Poco/Net/HTTPSClientSession.h>

 #include <Poco/Timer.h>

 using namespace Poco;
 using namespace Poco::Net;

 class Listener;

 struct NodeInfo {
    string uid;
    float posx;
    float posy;
    float current;    
    float target;
    bool ch0_state;
    UInt8 ch0_duty;
    bool ch0_valid;
    bool ch1_state;
    UInt8 ch1_duty;
    bool ch1_valid;
    bool ch2_state;
    UInt8 ch2_duty;
    bool ch2_valid;
    bool ch3_state;
    UInt8 ch3_duty;
    bool ch3_valid;
    UInt8 validate;
 };

 struct ValveInfo {
    string uid;
    UInt8 ch0_valve;
    UInt8 ch1_valve;
    UInt8 ch2_valve;
    UInt8 ch3_valve;
 };

 struct SwitchInfo {
    string uid;
    bool state;
 };

 #include "listener.h"

 class Nodes {
    static Data::Session* session;
    static bool initialized;
    static HTTPClientSession* influxClient;
    static string influxDb;
    static bool secure;
    static Listener* listener;
    static Timer* tempTimer;
    static Timer* nodesTimer;
    static Timer* switchTimer;
    static Nodes* selfRef;

 public:
    static void init(string influxHost, int influxPort, string influxDb, string influx_sec, Listener* listener);
    static void stop();
    static bool getNodeInfo(string uid, NodeInfo &info);
    static bool getValveInfo(string uid, ValveInfo &info);
    static bool getSwitchInfo(string uid, SwitchInfo &info);
    static bool setTargetTemperature(string uid, float temp);
    static bool setCurrentTemperature(string uid, float temp);
    static bool setDuty(string uid, UInt8 ch0, UInt8 ch1, UInt8 ch2, UInt8 ch3);
    static bool setValves(string uid, bool ch0, bool ch1, bool ch2, bool ch3);
    static bool setSwitch(string uid, bool state);
    void updateCurrentTemperatures(Timer& timer);
    void checkNodes(Timer& timer);

    void checkSwitch(Timer& timer);
    static bool getUIDs(vector<string> &uids);
    static bool getSwitchUIDs(vector<string> &uids);
 };
```

The definition for this class in the AC service gives a good overview of the functionality of this class. It's essentially a wrapper around an SQLite database, containing information on nodes, valves, and cooling/heating switches. It also contains the timers that will keep triggering the application to check the status of the system, to compare it to the target state, and to make adjustments if necessary.

This class is used extensively by the `Listener` class of this application for keeping track of the status of nodes and the connected AC units, along with those switches and valves controlling the water flow:

```cpp
#include <mosquittopp.h>

#include <string>
#include <map>

using namespace std;

#include <Poco/Mutex.h>

using namespace Poco;

struct NodeInfo;
struct ValveInfo;
struct SwitchInfo;

 #include "nodes.h"

 class Listener : public mosqpp::mosquittopp {
    map<string, NodeInfo> nodes;
    map<string, ValveInfo> valves;
    map<string, SwitchInfo> switches;
    Mutex nodesLock;
    Mutex valvesLock;
    Mutex switchesLock;
    bool heating;
    Mutex heatingLock;

 public:
    Listener(string clientId, string host, int port);
    ~Listener();

    void on_connect(int rc);
    void on_message(const struct mosquitto_message* message);
    void on_subscribe(int mid, int qos_count, const int* granted_qos);
    bool checkNodes();
    bool checkSwitch();
 };
```

The way that this application works is that the `Nodes` class timers will cause the `Listener` class to publish on the topics for the PWM, IO, and Switch modules, inquiring about the state of the devices that are supposed to be active.

This kind of active loop system is common in industrial applications, as it provides constant validation of the system to detect quickly if something isn't working as intended.

# InfluxDB for recording sensor readings

Recording the sensor readings and later the statistics read from the coffee machines was a priority from the beginning. The ideal database for this kind of data is a time series database, of which Influx is a common one. The biggest problem with this database is that it does not support MQTT, only offering its HTTP and native interface.

To fix this, a simple MQTT-to-Influx HTTP line protocol bridge was written, again using the Mosquitto client library as well as the POCO framework's HTTP functionality:

```cpp
#include "mth.h"

#include <iostream>

using namespace std;

#include <Poco/Net/HTTPRequest.h>
#include <Poco/Net/HTTPResponse.h>
#include <Poco/StringTokenizer.h>
#include <Poco/String.h>

using namespace Poco;

MtH::MtH(string clientId, string host, int port, string topics, string influxHost, 
                int influxPort, string influxDb, string influx_sec) : mosquittopp(clientId.c_str()) {
    this->topics  = topics;
    this->influxDb = influxDb;
    if (influx_sec == "true") { 
          cout << "Connecting with HTTPS..." << std::endl;
          influxClient = new Net::HTTPSClientSession(influxHost, influxPort);
          secure = true; 
    } 
    else {
          cout << "Connecting with HTTP..." << std::endl;
          influxClient = new Net::HTTPClientSession(influxHost, influxPort);
          secure = false; 
    }

    int keepalive = 60;
    connect(host.c_str(), port, keepalive);
 }
```

In the constructor, we connect to the MQTT broker, and create either an HTTP or HTTPS client, depending on which protocol has been set in the configuration file:

```cpp

 MtH::~MtH() {
    delete influxClient;
 }

 void MtH::on_connect(int rc) {
    cout << "Connected. Subscribing to topics...\n";

    if (rc == 0) {
          StringTokenizer st(topics, ",", StringTokenizer::TOK_TRIM | StringTokenizer::TOK_IGNORE_EMPTY);
          for (StringTokenizer::Iterator it = st.begin(); it != st.end(); ++it) {
                string topic = string(*it);
                cout << "Subscribing to: " << topic << "\n";
                subscribe(0, topic.c_str());

                // Add name of the series to the 'series' map.
                StringTokenizer st1(topic, "/", StringTokenizer::TOK_TRIM | StringTokenizer::TOK_IGNORE_EMPTY);
                string s = st1[st1.count() - 1]; // Get last item.
                series.insert(std::pair<string, string>(topic, s));
          }
    }
    else {
          cerr << "Connection failed. Aborting subscribing.\n";
    }
 }
```

Instead of fixed MQTT topics to subscribe to, we use the topics that are defined in the configuration file, here provided to us as a single string with each topic separated by a comma.

We also create an STL map containing the name of the time series to record for the topic, taking the final part of the MQTT topic after the last slash. One could make this further configurable, but for the topics used in the BMaC system this limitation was no consideration as it not necessary to have more complex topics.

```cpp
void MtH::on_message(const struct mosquitto_message* message) {
    string topic = message->topic;      
    map<string, string>::iterator it = series.find(topic);
    if (it == series.end()) { 
          cerr << "Topic not found: " << topic << "\n";
          return; 
    }

    if (message->payloadlen < 1) {
          cerr << "No payload found. Returning...\n";
          return;
    }

    string payload = string((const char*) message->payload, message-
    >payloadlen);
    size_t pos = payload.find(";");
    if (pos == string::npos || pos == 0) {
        cerr << "Invalid payload: " << payload << ". Reject.\n";
        return;
    }

    string uid = payload.substr(0, pos);
    string value = payload.substr(pos + 1);
    string influxMsg; 
    influxMsg = series[topic];
    influxMsg += ",location=" + uid;
    influxMsg += " value=" + value;
    try {
        Net::HTTPRequest request(Net::HTTPRequest::HTTP_POST, 
        "/write?db=" + influxDb, Net::HTTPMessage::HTTP_1_1);
        request.setContentLength(influxMsg.length());
        request.setContentType("application/x-www-form-urlencoded");
        influxClient->sendRequest(request) << influxMsg;

        Net::HTTPResponse response;
        influxClient->receiveResponse(response);
    }
    catch (Exception& exc) {
        cout << "Exception caught while attempting to connect." << 
        std::endl;
        cerr << exc.displayText() << std::endl;
        return;
    }
```

When we get a new MQTT message in, we find the name of the Influx time series for it, then create a string to send to the InfluxDB server. The assumption here is that the payload consists of the MAC address of the node which sent the message followed by a semi-colon.

We simply get the part after the semi-colon to set it as the value, and use the MAC as the location. This we then send to the database server.

# Security aspects

During the development of this system it became soon obvious that security would be a paramount aspect of the system. For that reason we looked at adding transport layer security (TLS) encryption. This would use the integrated axTLS encryption library in the Sming framework together with AES certificates (host and client) to provide both verification that the host (servers) and clients (nodes) are who they say they are, but also provide a secure encrypted link.

In [Chapter 5](886aecf2-8926-4aec-8045-a07ae2cdde84.xhtml), *Example - Soil Humidity Monitor with Wi-Fi*, we already looked at the handling of these client certificates and setting up of an encrypted MQTT connection. One detail which is not obvious from that were the troubles which we encountered while setting up this certificate system. As mentioned in [Chapter 5](886aecf2-8926-4aec-8045-a07ae2cdde84.xhtml), *Example - Soil Humidity Monitor with Wi-Fi*, the ESP8266 does not have enough memory to allocate the default TLS handshake buffers and requires the use of the SSL fragment size extension on the side of the server (host).

Unfortunately we found that the commonly used MQTT broker we were using (Mosquitto) did not support this SSL extension and would therefore require that clients used the default double 16 kB buffer. The first solution to this would be to recompile the Mosquitto broker after making a few changes to its source code to change this setting.

The better solution and the one which we ultimately implemented was to install a proxy software (HAProxy) which functioned as the TLS endpoint, handling the certificates and redirecting the decrypted traffic to the MQTT broker via the local loopback (localhost) interface.

With the SSL fragment size option set to 1-2 kB everything worked as intended and we had a building-wide, wireless monitoring and control system that allowed for secure communications of sensitive information and delicate control commands.

# Future developments

There are still many additions that can be made to this system. From the number of sensors that could be supported, further GPIO expander chips, air-conditioning system configurations, room occupancy detection linked into a calendar backend, to clearing out scheduled meetings at an office where nobody showed up, and so on.

There is also the option of switching from ESP8266 as the MCU to a different one, such as ARM-based MCUs, to get wired Ethernet options, along with better debug and development tools. As convenient as it is to have an MCU with Wi-Fi, which one can just stick anywhere and theoretically have it work, the development tools for the ESP8266 aren't that great, and the lack of wired communication options (without using external chips) means that everything either works or doesn't depending on the quality of the Wi-Fi network.

As BMaC involves the automation of a building, it is desirable to have a certain level of reliability, which is hard to guarantee with a Wi-Fi network, though for less crucial components (coffee machine statistics, sensor readings, and so on) this is unlikely to be an issue. Conceivably a hybrid network with both wired and wireless options could be the future.

# Summary

In this chapter, we looked at how a building-wide monitoring and management system was developed, what its components looked like, and what lessons were learned during its development.

The reader is expected to understand now how such a large-scale embedded system is constructed and functions, and should be able either to use the BMaC system themselves or implement a similar system.

In the next chapter we will look at developing embedded projects using the Qt framework.