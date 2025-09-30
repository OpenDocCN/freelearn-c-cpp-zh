# *Chapter 12*: COVID-19 Digital Body Temperature Measurement (Thermometer)

This chapter describes an interesting project where you will develop a touchless thermometer to measure human body temperature. This digital thermometer could be useful for supporting the diagnosis of people with COVID-19\. The electronics project explained in this chapter involves the use of a very capable **infrared** (**IR**) temperature sensor that will check the body temperature. Additionally, you will learn and practice how to connect an IR temperature sensor to a microcontroller board using the **Inter-Integrated Circuit** (**I2C**) data transmission protocol.

Important note

The body temperature measurement project described in this chapter should not be used as a definitive and accurate way to determine whether a person has COVID-19 or not. It is for demonstration and learning purposes only.

In this chapter, we will cover the following main topics:

*   Programming the I2C interface
*   Connecting an IR temperature sensor to the microcontroller board
*   Showing the temperature on an LCD
*   Testing the thermometer

By the end of this chapter, you will have learned how to get useful data from an IR temperature sensor and how to effectively show body temperature data on an LCD connected to a microcontroller board. You will also learn how the I2C data transmission protocol works to get data from the IR sensor, and how to properly test an IR temperature sensor.

# Technical requirements

The software tool that you will be using in this chapter is the **Arduino IDE** for editing and uploading your programs to the Blue Pill microcontroller board. The code used in this chapter can be found in the book's GitHub repository:

[https://github.com/PacktPublishing/DIY-Microcontroller-Projects-for-Hobbyists/tree/master/Chapter12](https://github.com/PacktPublishing/DIY-Microcontroller-Projects-for-Hobbyists/tree/master/Chapter12)

The Code in Action video for this chapter can be found here: [https://bit.ly/2SMUkPw](https://bit.ly/2SMUkPw)

In this chapter, we will use the following pieces of hardware:

*   One solderless breadboard.
*   One Blue Pill microcontroller board.
*   One micro-USB cable for connecting your microcontroller board to a computer and a power bank.
*   One Arduino Uno microcontroller board.
*   One USB 2.0 A to B cable for the Arduino Uno board.
*   Two USB power banks.
*   One ST-Link/V2 electronic interface, needed for uploading the compiled code to the Blue Pill. Bear in mind that the ST-Link/V2 requires four female-to-female DuPont wires.
*   One MLX90614ESF-DCA-000 temperature sensor (it works with 3.3 volts).
*   One 0.1 microfarad capacitor. It generally has a 104 label on it.
*   One 1602 16x2 I2C LCD.
*   A dozen male-to-male and a dozen male-to-female DuPont wires.

The next section describes how to code the I2C protocol and the code that will run on the Blue Pill and Arduino Uno microcontroller boards. The Arduino Uno is used for getting data from the IR sensor.

# Programming the I2C interface

In this section, we will review how to obtain useful data from the MLX90614 temperature sensor to be transmitted using the **I2C** protocol, also known as **IIC**. It is a serial data communication protocol that is practical for interfacing sensors, LCDs, and other devices to microcontroller boards that support I2C. The next section defines what I2C is.

## The I2C protocol

I2C is a synchronous serial communication protocol that allows interconnecting sensors, microcontrollers, displays, **analog-to-digital converters** (**ADCs**), and so on, at a short distance using a common **bus** (a bus works as a main digital road). The I2C bus is composed of a few lines (wires) that all the devices share and use for transmitting and exchanging data. The I2C protocol is practical and beneficial because it uses only two wires for data communication. Another benefit of I2C is that in theory, it can support up to 1,008 devices connected to the same I2C bus! It is also worth mentioning that more than one microcontroller can be connected to the same bus, although they must take turns to access data from the I2C bus. *Figure 12.1* shows an overview of the I2C bus configuration:

![Figure 12.1 – An overview of the I2C bus](img/Figure_12.1_B16413.jpg)

Figure 12.1 – An overview of the I2C bus

As *Figure 12.1* shows, the bus allows the connection of two main types of devices: a **controller** (also known as a **master**) and a **peripheral** (also known as a **slave**). The controller is usually a microcontroller board, but it can also be a personal computer or another type of device that will take, send, process, and use data from and to the peripheral(s). A peripheral can be a sensor that will provide data to a controller, or a display (such as an LCD) where data coming from a controller is displayed. There are other types of peripherals that can be connected to an I2C bus.

Note

Not all LCDs can be directly connected to an I2C bus. To do that, the LCD must have an I2C adapter connected to it, often called an **I2C backpack** or **I2C module**. This is a small electronic circuit that handles I2C communications, typically attached at the back of some LCDs.

An I2C bus contains two data lines (wires) called **SDA** and **SCL**. The SCL wire transports the clock signal necessary for synchronizing the transmitted data on the bus, and the SDA wire is the data signal transporting all the data between the controller(s) and the peripheral(s).

The devices connected to the I2C bus also have two more wires. One of them is the ground (sometimes labeled `GND` or `Vss`). This wire should be connected to the common ground of the electronic circuit where the microcontroller board is connected to, and a voltage wire (labeled `Vdd`). This is connected to 5 volts, but it sometimes works with 3.3 volts. The I2C protocol is quite robust, allowing a bit rate of up to 5 Mbit/s. Generally, either 5 or 3.3 volts are provided by the microcontroller board, used by the devices connected to the I2C bus.

Many microcontroller boards support the I2C protocol. Fortunately, the Blue Pill includes I2C pins for that. In fact, the Blue Pill has three sets of pins for connecting three I2C devices directly. In this chapter, we will use the Blue Pill's B6 (SCL) and B7 (SDA) pins for the I2C communication only.

Note

The Arduino microcontroller boards also support the I2C protocol. For example, the pins A4 and A5 from the Arduino Uno board provide the connection for the SDA and SCL wires, respectively.

It is worth noting that the I2C protocol is handled by the `Wire.h` library, found by default in the Arduino IDE configuration. You don't need to install this library.

In the next section, we will review how to code the I2C protocol to get temperature data from the IR sensor using an *Arduino Uno* microcontroller board.

## I2C coding

In this section, we review the code for reading the temperature data from the MLX90614 IR sensor by an Arduino Uno microcontroller board, working as a peripheral (slave). This board will send the temperature data to the Blue Pill through the I2C bus, and the Blue Pill (working as a controller) will display it on an LCD. This LCD is also connected to the I2C bus. The full description of the Blue Pill and the Arduino Uno connections is found in the *Connecting an IR temperature sensor to the microcontroller board* section.

The next section explains the necessary code that runs on the Arduino Uno board.

## Coding the Arduino Uno software (peripheral)

In order to code an application on the Arduino Uno for reading the MLX90614 IR sensor data, we will use a library to be included in our Arduino IDE program, called `Adafruit_MLX90614.h`. You can install this library from the Arduino IDE:

1.  Go to **Tools** | **Manage Libraries**.
2.  Set **Type** as **All** and **Topic** as **All**.
3.  In the search field, type `Adafruit MLX90614`.

Install the latest version of the `Adafruit MLX90614` library (do not install the Mini one). The `Wire.h` library controls the I2C protocol, which is already installed on the Arduino IDE's files.

The following is the code that runs on the Arduino Uno (its file is called `peripheral.ino`; you can find it on the GitHub page). `0X8` is the hexadecimal address assigned to the Arduino Uno board as the peripheral (slave) for the I2C protocol. We assigned 0x8 arbitrarily; it can be any hexadecimal address, but make sure to use the same address for both the master and the slave:

```cpp
#include <Wire.h> 
#include <Adafruit_MLX90614.h> 
Adafruit_MLX90614 mlx = Adafruit_MLX90614(); 
#define SLAVEADDRESS 0x8  
float AmbientobjC=0.0; 
```

This function sets up the Arduino Uno as a peripheral, assigning it the 0x8 address. The Blue Pill will identify the Arduino Uno with this address. This function also sets up the interrupt that handles incoming requests from the controller (master), which is the Blue Pill. The Blue Pill will be acquiring the temperature data from the sensor:

```cpp
void setup() {  
  Wire.begin(SLAVEADDRESS);
  Wire.onRequest(requestEvent); 
}
```

This function will read the temperature from the sensor continuously. The `delay` function makes a small pause to allow the sensor to get new temperature readings:

```cpp
void loop() {
  AmbientobjC=mlx.readObjectTempC();
  delay(100);
}
```

This function runs every time the controller (master) requests data from the peripheral (slave):

```cpp
void requestEvent() { 
  union floatToBytes {
    char buffer[4];
    float objtempReading;
  } converter;
  converter.objtempReading = AmbientobjC+3;
  Wire.write(converter.buffer, 4); 
}
```

From the previous code snippet, we see that the `write()` function is used to send the temperature data to the controller (the Blue Pill).

We added the value of `3` to the `AmbientobjC` variable to compensate for the ambient temperature. It is important to clarify that the temperature readings are not absolute and will slightly change depending on a number of factors, including the ambient temperature, whether the person is outside. So, you may need to test the temperature readings a number of times and adjust the `AmbientobjC` variable accordingly, perhaps comparing the sensor readings against medical body thermometer readings.

Note

The code uploaded to the GitHub platform contains extensive comments that explain most of the code lines.

The next section explains the necessary code for running the Blue Pill as a controller.

## Coding the Blue Pill software (controller)

The following code (its file is called `controller.ino`; you can find it on the GitHub page) will run on the Blue Pill microcontroller board (the controller). This code will serve to obtain the temperature data sent by the the Arduino Uno board, and then display it on an LCD connected to the I2C bus:

```cpp
#include <LiquidCrystal_I2C.h> 
LiquidCrystal_I2C lcd(0x27, 16, 2);
#include <Wire.h> 
#define SLAVEADDRESS 0x8  
```

The previous code snippet shows the `LiquidCrystal_I2C.h` library for controlling the LCD using the I2C protocol. Its next line sets up the LCD address to `0x27` for a 16-character and 2-line (16x2) LCD. The library can be downloaded from [https://github.com/fdebrabander/Arduino-LiquidCrystal-I2C-library](https://github.com/fdebrabander/Arduino-LiquidCrystal-I2C-library). Download the `LiquidCrystal_I2C.h` file and copy it to the Arduino libraries folder, usually `Arduino/libraries`.

The next code snippet starts an I2C connection using a peripheral (slave) address, initializing the Serial Monitor and the LCD:

```cpp
void setup() {
  Wire.begin(SLAVEADDRESS); 
  Serial.begin(9600); 
  lcd.begin();  
  lcd.backlight(); 
}
```

This `loop()` function continuously reads the temperature data sent by the peripheral (the Arduino Uno):

```cpp
void loop() {
   Wire.requestFrom(8, 4);
   uint8_t index = 0;
   union floatToBytes {
       char buffer[4];
       float objtempReading; 
   } converter;
   while (Wire.available()){ 
      converter.buffer[index] = Wire.read(); 
      index++;
   }
   Serial.println(converter.objtempReading); 
   lcd.setCursor(0, 0);  
   lcd.print("Body Temp.:"); 
   lcd.print(converter.objtempReading);
   delay(500);
}
```

The previous code snippet shows how to read the data in bytes from the Arduino Uno. Remember that we can't transmit float values directly over the I2C bus. The code that runs on the Arduino Uno converts each floating-point temperature data reading into four character bytes. The code running on the Blue Pill converts the four bytes back to a floating-point number.

The code submitted to the GitHub page contains many comments explaining the code lines.

The following section explains how to connect the IR sensor to the Blue Pill microcontroller board.

# Connecting an IR temperature sensor to the microcontroller board

This section explains the main technical characteristics of the MLX90614 temperature sensor. It also shows how to connect it to the Blue Pill microcontroller board using the *I2C protocol*.

First, let's explain the main characteristics of the MLX90614 sensor.

## The MLX90614 IR sensor

The MLX90614 sensor, manufactured by the company Melexis, is a powerful yet compact IR sensor. This sensor uses IR rays to measure the amount of heat generated by the human body or by basically any object. Being a non-contact thermometer, it reduces chance of spreading disease when checking the body temperature, and you don't need to clean it.

The MLX0-614 is technically a sensor contained in an **integrated circuit** (**IC**), since it has extra electronic components and smaller circuits, including an ADC, a voltage regulator, and a **digital-signal processor** (**DSP**).

The following are some of the sensor's technical characteristics:

*   Body temperature range of -40 to +125 degrees Celsius.
*   Ambient temperature range of -70 to 382.2 degrees Celsius.
*   Medical (high) accuracy calibration.
*   Measurement resolution close to 0.02 degrees Celsius.
*   This sensor is available in both 3-volt and 5-volt versions.
*   A convenient sleep mode that reduces power consumption.

The MLX90614 family of sensors' datasheet can be downloaded from the following link:

[https://www.melexis.com/-/media/files/documents/datasheets/mlx90614-datasheet-melexis.pdf](https://www.melexis.com/-/media/files/documents/datasheets/mlx90614-datasheet-melexis.pdf)

The MLX90614 IR sensor generates two types of temperature measurements: ambient temperature and an object temperature reading. The **ambient temperature** is the temperature registered on the IR-sensitive part of the sensor (an internal component), which is close to room temperature. The **object temperature** measures how much IR light is emitted by an object, which can be used for measuring body temperature. In this chapter, we will use only the object temperature measurement.

*Figure 12.2* shows the MLX90614 sensor:

![Figure 12.2 – The MLX90614 sensor showing its four pins](img/Figure_12.2_B16413.jpg)

Figure 12.2 – The MLX90614 sensor showing its four pins

In *Figure 12.2*, please note that there is a round transparent window on the sensor, where the IR light passes through, hitting an internal sensitive part. This part will convert IR light into electrical impulses.

*Figure 12.3* shows the *top* view of the MLX90614 sensor diagram pinout:

![Figure 12.3 – MLX90614 IR sensor pinout](img/Figure_12.3_B16413.jpg)

Figure 12.3 – MLX90614 IR sensor pinout

In *Figure 12.3*, you will notice that there is a small notch on top of the sensor. This notch helps you identify which pin is which.

The MLX90614's SCL and SDA pins are connected to a microcontroller board's SCL and SDA, respectively. The ground pin is connected to the microcontroller board's ground, and the voltage pin is connected to either 3.3 or 5 volts.

The MLX90614 sensor is manufactured in different types or versions. One of the most important differences is its supply voltage. It is important to verify its part number:

*   **MLX90614ESF-Axx**, **MLX90614KSF-Axx**: Their supply voltage is 5.5 volts.
*   **MLX90614ESF-Bxx**, **MLX90614ESF-Dxx**: Their supply voltage is 3.6 volts.

For example, the IR sensor that we use in this chapter is the MLX90614ESF-DCA-000, which, according to the manufacturer's datasheet, requires 3.6 volts to work. So, you can use the 3.3 volts supplied by many microcontroller boards for using this type of sensor.

Important note

Always check the IR sensor part number to determine what type of supply voltage it will need. If you apply a voltage higher than its required supply voltage, you can damage the sensor.

Another technical aspect that you should take into account is the sensor's field of view. It is a relationship between the distance between the sensor and an object that its temperature is being measured. This will determine the sensing area being observed by the sensor. For every 1 centimeter that the object moves away from the sensor's surface, the sensing area grows by 2 centimeters. Ideally, the distance between the sensor and the object (for example, human skin) to be measured should be between 2 and 5 centimeters.

The next section explains the GY-906 module, containing an IR sensor.

## The GY-906 sensor module

This section describes the main characteristics and pinout of the GY-906 sensor module. The MLX90614 sensor (see *Figure 12.4*) is also sold embedded in a module called **GY-906**:

![Figure 12.4 – The GY-906 module](img/Figure_12.4_B16413.jpg)

Figure 12.4 – The GY-906 module

As *Figure 12.4* shows, the GY-906 module has four pins dedicated to I2C communication. The VIN pin is connected to voltage, either 3.3 volts or 5 volts depending on the type of MLX90641 sensor that it contains. Please refer to the *The MLX90614 IR sensor* section to help identify what voltage the embedded sensor will need in the module. You will also need to consult the module's datasheet. For example, a GY-906 module may have an MLX90614ESF-DCA-000 IR sensor embedded in it, needing 3.3 volts to work. So, GY-906 modules could have any MLX90614 sensor type embedded in them. The module's GND pin is connected to the microcontroller board's ground. The SCL pin is for transmitting the signal clock, and the SDA pin transmits the signal data.

Besides the MLX90614 sensor, the GY-906 module also contains other electronic components, such as pull-up resistors.

The decision on using either an MLX90614 sensor or a GY-906 module rests on a number of factors, including cost, type of application, and size. Bear in mind that the GY-906 module is somewhat larger than the MLX90614 sensor. In this chapter, we will use the MLX90614 sensor and not the module, because the sensor alone is a cost-effective option and to demonstrate how to directly connect the sensor to an I2C bus. However, both the GY-906 module and the MLX90614 sensor have the same functions.

The next section shows how to connect the MLX90614 sensor to a microcontroller board.

## Connection of the IR sensor to an Arduino Uno

In this section, we describe how to connect the MLX90614 sensor to the I2C pins of an Arduino Uno microcontroller board, which will be working as a peripheral (slave). The Blue Pill board will be the controller (master), receiving temperature data sent by the Arduino Uno through the I2C bus, as shown in *Figure 12.5*:

![Figure 12.5 – The controller and peripherals](img/Figure_12.5_B16413.jpg)

Figure 12.5 – The controller and peripherals

As *Figure 12.5* shows, both the Blue Pill and the Arduino Uno boards are connected to the I2C bus.

We connected the MLX90614 to an Arduino Uno board working as a peripheral (slave) for the following reasons:

*   For taking advantage of the I2C data transmission protocol, which only requires two data wires.
*   For practicing the implementation of the I2C protocol using controller (master) and peripheral (slave) devices.
*   For using MLX90614 software libraries that are fully compatible with other microcontroller boards (for example, the Arduino family), but not with the Blue Pill board. That is why we are connecting the IR sensor to an Arduino Uno board using its respective library. This way we could use the Arduino Uno as a peripheral (slave) that is 100% compatible with the IR sensor.
*   For freeing up the Blue Pill (the controller) from processing the sensor's data directly, so the Blue Pill can be used to do other processing-intensive tasks. In addition, the Blue Pill can be dedicated to obtain the sensor's data and display it on an LCD.

First, we will explain how to connect the MLX90614 IR sensor to an Arduino Uno microcontroller board. The connections are shown in *Figure 12.6*:

![Figure 12.6 – The Arduino Uno and IR sensor's connections](img/Figure_12.6_B16413.jpg)

Figure 12.6 – The Arduino Uno and IR sensor's connections

As you can see from *Figure 12.6*, the sensor is connected to analog ports A4 and A5, which are also the Arduino Uno's I2C pins. The 0.1 microfarad capacitor is recommended by the sensor's datasheet to smooth out any high-frequency or very-high frequency electrical noise that may be present in the sensor that may affect the temperature readings. Here are the steps for connecting everything:

1.  Connect the sensor's voltage (VDD) pin to Arduino Uno's 3.3 volts pin. Do this if you are using an MLX90614ESF-DCA-000 or an MLX90614ESF-BAA-000 sensor. If you are using an MLX90614ESF-AAA-000 or an MLX90614KSF-ACC-000 sensor, connect it to Arduino Uno's 5 volts pin.
2.  Connect the sensor's ground (VSS) pin to Arduino Uno's ground (GND) pin.
3.  Connect the sensor's SDA pin to Arduino Uno's A4 analog port.
4.  Connect the sensor's SCL pin to Arduino Uno's A5 analog port.
5.  Connect one leg of the 0.1 microfarad capacitor to the sensor's voltage pin and the other capacitor's leg to the sensor's ground pin.

Once you connected the wires and components, upload and run the code to the Arduino Uno board called `peripheral.ino`, explained in the *Programming the I2C interface* section. It should be showing the temperature data on the Arduino IDE's Serial Monitor. Follow these steps for opening the Serial Monitor for the Arduino Uno:

1.  Open **Tools** from the IDE's main menu.
2.  Select the **Serial Monitor** option.
3.  From the Serial Monitor, make sure to select **9600** bauds.
4.  Don't forget to select **Arduino Uno** from **Tools** | **Board**.
5.  Select the right USB port where the Arduino Uno is connected, clicking on **Tool** | **Port** from the Arduino IDE.

The next section describes how to connect a Blue Pill to the Arduino Uno using the I2C bus and how to transmit temperature data from the Arduino Uno to the Blue Pill.

## Connecting the Blue Pill to the Arduino Uno

This section shows how to connect the Arduino Uno and the Blue Pill through the I2C bus. The Arduino Uno will send the IR temperature data to the Blue Pill. Remember that the Blue Pill works as a controller (master) and the Arduino Uno is the peripheral (slave). *Figure 12.7* shows a Fritzing diagram with the two microcontroller boards:

![Figure 12.7 – The Blue Pill and Arduino Uno I2C connections](img/Figure_12.7_B16413.jpg)

Figure 12.7 – The Blue Pill and Arduino Uno I2C connections

As *Figure 12.7* shows, here are the steps for connecting the Blue Pill to the I2C bus:

1.  Connect Arduino Uno's ground (GND) to Blue Pill's ground (G or `GND`).
2.  Connect Blue Pill's B7 pin to MLX90614's SDA pin.
3.  Connect Blue Pill's B6 pin to MLX90614's SCL pin.

As you can see from all the connections from *Figure 12.7*, the Blue Pill, the IR sensor, and the Arduino Uno board are all connected by the SDA and the SCL pins. This is the I2C bus in our application.

Important note

Make sure to connect Blue Pill's ground (`G` or `GND`) to Arduino Uno's ground (`GND`). This will allow for correct data transmission across the I2C bus between the two microcontroller boards.

The following section describes how to show the IR temperature measurements on an LCD using the I2C bus.

# Showing the temperature on an LCD

This section describes how to display the IR temperature measurement on an LCD through the I2C bus. The temperature data is sent by the Arduino Uno to the Blue Pill, as shown in the previous section. *Figure 12.8* shows a Fritzing diagram containing the microcontroller boards, the LCD, and the IR temperature sensor:

![Figure 12.8 – The LCD connected to the I2C bus](img/Figure_12.8_B16413.jpg)

Figure 12.8 – The LCD connected to the I2C bus

As *Figure 12.8* shows, the LCD connection is simple. It requires four wires only, because the LCD used in this chapter is I2C-capable, having an I2C interface in the back. The following are the steps for connecting the LCD to the Blue Pill:

1.  Connect the LCD's ground (`GND`) pin to Blue Pill's ground (`G` or `GND`) pin.
2.  Connect the LCD's voltage (`VCC`) pin to Blue Pill's 5-volt (5V) pin.
3.  Connect the LCD's SDA pin to Blue Pill's B7 pin.
4.  Connect the LCD's SCL pin to Blue Pill's B6 pin.

*Figure 12.9* depicts the back of the LCD, showing its I2C interface backpack attached to it:

![Figure 12.9 – The back of the LCD](img/Figure_12.9_B16413.jpg)

Figure 12.9 – The back of the LCD

From *Figure 12.9*, you can see that the LCD's I2C interface has a small variable resistor, just right of the big IC. You can rotate it to adjust the LCD contrast.

The LCD can display up to 16 characters on each of its two rows. This is enough to show the IR sensor temperature with two-digit precision and two decimals.

Important note

Make sure to connect the LCD's voltage (`VCC`) to Blue Pill's `5V`. If you connect it to a 3.3-volt pin, it may not work properly.

*Figure 12.10* shows how everything is connected:

![Figure 12.10 – The microcontroller boards, the sensor, and the LCD](img/Figure_12.10_B16413.jpg)

Figure 12.10 – The microcontroller boards, the sensor, and the LCD

*Figure 12.10* shows that the LCD is showing a temperature of 28.53 degrees Celsius, because the IR sensor was unintentionally measuring the temperature of an LED lamp when the photo was taken! However, this circuit is intended to be used for body temperature measurement. The next section shows how to test the sensor by checking the temperature on different parts of the human body. We used two power banks connected to the Blue Pill and the Arduino Uno to try them out. If you finished the connection of the temperature sensor and the microcontroller boards, and if your LCD is displaying a temperature value, congratulations! You know how to use a touchless IR temperature sensor.

# Testing the thermometer

In this section, we will test out how the IR sensor works as a thermometer by measuring the temperature of a human body. It seems that different body parts will get you slightly different temperature measurements. You should do a number of tests by measuring the body temperature from different parts, such as the forehead and the earlobe of a person. Remember that the distance between the sensor and the skin should be between 2 and 5 centimeters, although you should try out different distances and see what happens.

Before testing the thermometer, make sure that the skin is dry, clean, and unobstructed. In addition, confirm that the person has not been exposed to high heat, such as being out on a hot and sunny day, because this will change your measurements. If you are measuring skin temperature with the thermometer, make sure that the person is out of direct sunlight or you will get incorrect readings.

Medical studies indicate that the average skin surface temperature of the human body is in the range of 36.3 to 36.6 degrees Celsius. This is considered normal. However, a temperature higher than 37 degrees Celsius is suggestive of fever.

With these tips in mind, let's see how our thermometer works. *Figure 12.11* shows a person testing the IR sensor:

![Figure 12.11 – Measuring the temperature](img/Figure_12.11_B16413.jpg)

Figure 12.11 – Measuring the temperature

As *Figure 12.11* shows, the measured temperature was 36.55 degrees Celsius, which is within the normal range for adults. The tester needed to place her forefront close to the IR sensor to get reliable measurements. The distance between the forehead and the sensor can be increased by using a mirror tube placed on top of the sensor to steer the IR light through it.

Now that we have tested the chapter and realized that the thermometer works perfectly, let's recap in the *Summary* what we have learned.

# Summary

In this chapter, you learned the basics of connecting an IR temperature sensor to a microcontroller board using the I2C serial data transmission protocol. As you could see, the I2C bus is an important part of the IR thermometer that we built in this chapter. This IR thermometer can be effective for checking human body temperature, like a regular thermometer will do. Since the temperature measurement is contactless, this can prevent the human body from touching the sensor and thus avoiding spreading viruses such as SARS-CoV-2.

This touchless thermometer may help to check whether a person has a fever and thus determine (along with other measurements) whether the person has contracted COVID-19\. However, the IR temperature measurements explained in this chapter should not be definitive data to determine whether a person has COVID-19.

The next chapter explains another COVID-19-related project about measuring the recommended 2-meter distance between two people using an ultrasonic sensor.

# Further reading

*   Body Temperature (2020). Body temperature: What is (and isn't) normal? Available from [https://health.clevelandclinic.org/body-temperature-what-is-and-isnt-normal/](https://health.clevelandclinic.org/body-temperature-what-is-and-isnt-normal/)
*   Gay, W. (2018) I2C, *In: Beginning STM32*, Apress, Berkeley, CA
*   I2C (2021), *I2C tutorial*. Available from [https://learn.sparkfun.com/tutorials/i2c/all](https://learn.sparkfun.com/tutorials/i2c/all)
*   Mankar, J., Darode, C., Trivedi, K., Kanoje, M., and Shahare, P. (2014), *Review of I2C protocol*, International Journal of Research in Advent Technology, 2(1)