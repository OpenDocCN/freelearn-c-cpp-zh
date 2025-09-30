# Chapter 10. Some Advanced Techniques

In this chapter, we are going to learn different techniques that can be used either together or independently. Each technique developed here is a new tool for your future or current projects. We are going to use EEPROMs to provide Arduino boards with a small memory system that is readable and writable.

We are also going to test communications between the Arduino boards themselves, use GPS modules, make our boards autonomous, and more.

# Data storage with EEPROMs

Until now, we learned and used the Arduino boards as totally electricity dependent devices. Indeed, they need current in order to execute tasks compiled in our firmware.

As we noticed, when we switch them off, every living variable and data is lost. Fortunately, the firmware isn't.

## Three native pools of memory on the Arduino boards

The Arduino boards based on the ATmega168 chipset own three different pools of memory:

*   Flash memory
*   SRAM
*   EEPROM

The flash memory is also named program space. This is the place where our firmware is stored.

The **SRAM** stands for **Static** **Random** **Access** **Memory** and is the place where the running firmware stores, reads, and manipulates variables.

The **EEPROM** stands for **Electrically** **Erasable** **Programmable** **Read-Only** **Memory**. It is the place where we, programmers, can store things for long-term purposes. This is the place where our firmware sits, and anything in the EEPROM isn't erased should the board be switched off.

ATmega168 has:

*   16000 bytes of Flash (2000 bytes are used for the bootloader)
*   1024 bytes of SRAM
*   512 bytes of EEPROM

Here we won't discuss the fact that we have to take care of the memory while programming; we will do that in the last chapter of this book [Chapter 13](ch13.html "Chapter 13. Improving your C Programming and Creating Libraries"), *Improving your C Programming and Creating Libraries*.

The interesting part here is the EEPROM space. It allows us to store data on the Arduino and we didn't even know that until now. Let's test the EEPROM native library.

### Writing and reading with EEPROM core library

Basically, this example doesn't require any wiring. We are going to use the internal EEPROM of 512 bytes. Here is some code that reads all the bytes of the EEPROM and prints it to the computer's Serial Monitor:

[PRE0]

This code is in the public domain and provided as an example for the EEPROM library. You can find it in your `examples` folder in the **File** menu of the Arduino IDE, under the folder **Examples** | **EEPROM**.

At first, we include the library itself. Then we define a variable for storing the current read address. We initialize it at 0, the beginning of the memory register. We also define a variable as a byte type.

In the `setup()` function, we initialize the serial communication. In `loop()`, we read the byte at the current address and store it in the variable `value`. Then we print the result to the serial port. Notice the `\t` value in the second `Serial.print()` statement. This stands for tabulation (as in the *Tab* key on a computer keyboard). This writes tabulation to the serial port between the current address printed and the value itself in order to make things more readable.

We advance to the next address. We check if the address equals 512, if that is the case, we restart the address counter to 0 and so on.

We add a small delay. We can write bytes in the same way using `EEPROM.write(addr, val);` where `addr` is the address where you want to write the value `val`.

Be careful, these are bytes (8 bits = 256 possible values). Read and write operations are quite easy on the internal EEPROM, so let's see how it goes with external EEPROMs wired by an I2C connection.

## External EEPROM wiring

There are a lot of cheap EEPROM components available in electronics markets. We are going to use the classic 24LC256, an EEPROM implementing I2C for read/write operations and providing 256 kilobits (32 kilobytes) of memory space.

You can find it at Sparkfun: [https://www.sparkfun.com/products/525](https://www.sparkfun.com/products/525). Here is how we can wire its bigger cousin 24LC1025 (1024k bytes) using I2C:

![External EEPROM wiring](img/7584_10_001.jpg)

A 24LC256 EEPROM wired to the Arduino via I2C communication

The corresponding diagram is the one shown as follows:

![External EEPROM wiring](img/7584_10_002.jpg)

A 24LC256 EEPROM wired to the Arduino via I2C communication

Let's describe the EEPROM.

**A0**, **A1**, and **A2** are chip address inputs. **+V** and **0V** are **5V** and ground. WP is the write protect pin. If it is wired to ground, we can write to the EEPROM. If it is wired to 5V, we cannot.

SCL and SDA are the two wires involved in the I2C communication and are wired to **SDA** / **SCL**. **SDA** stands for **Serial** **Data** **Line** and **SCL** stands for **Serial** **Clock** **Line**. Be careful about the SDA/SCL pins. The following depends on your board:

*   The Arduino UNO before R3 and Ethernet's I2C pins are A4 (SDA) and A5 (SCL)
*   Mega2560, pins 20 (SDA) and 21 (SCL)
*   Leonardo, pin 2 (SDA) and pin 3 (SCL)
*   Due Pins, pins 20 (SDA) and 21 (SCL) and also another one SDA1 and SCL1

## Reading and writing to the EEPROM

The underlying library that we can use for I2C purposes is `Wire`. You can find it directly in the Arduino core. This library takes care of the raw bits, but we have to look at it more closely.

The `Wire` library takes care of many things for us. Let's check the code in the folder `Chapter10/readWriteI2C`:

[PRE1]

We include the `Wire` library at first. Then we define 2 functions:

*   `eepromWrite()`
*   `eepromRead()`

These functions write and read bytes to and from the external EEPROM using the `Wire` library.

The `Setup()` function instantiates the `Wire` and the `Serial` communication. Then using a `for` loop, we write data to a specific address. This data is basically a character 'a' plus a number. This structure writes characters from a to a + 9 which means 'j'. This is an example to show how we can store things quickly, but of course we could have written more meaningful data.

We then print a message to the Serial Monitor in order to tell the user that Arduino has finished writing to the EEPROM.

In the `loop()` function, we then read the EEPROM. It is quite similar to the EEPROM library.

Obviously, we still haven't spoken about addresses. Here is an I2C message format:

![Reading and writing to the EEPROM](img/7584_10_003.jpg)

An I2C message

`Wire` library takes care of **Start** **Bit** and **Acknowledge** **Bit**. The control code is fixed and you can change the **Chip** **Select** **Bits** by wiring **A0**, **A1**, and **A2** pins to ground or +V. That means there are 8 possibilities of addresses from 0 to 7.

1010000 1010001… until 1010111\. 1010000 binary means 0x50 in hexadecimal, and 1010111 means 0x57.

In our case, we wired **A0**, **A1**, and **A2** to ground, then the EEPROM address on the I2C bus is 0x50\. We could use more than one on the I2C bus, but only if we need more storage capacity. Indeed, we would have to address the different devices inside our firmware.

We could now imagine storing many things on that EEPROM space, from samples for playing PCM audio to, eventually, huge lookup tables or whatever requiring more memory than available on Arduino itself.

# Using GPS modules

**GPS** stands for **Global** **Positioning** **System**. This system is based on satellite constellations.

Basically, a receiver that receives signals from at least 4 satellites embedded with a special atomic clock can, by calculating propagation time of these signals between them and itself, calculate precisely its tri-dimensional position. That sounds magical; it is just trigonometric.

We won't get into the details of this process; instead focus on the parsing of data coming from the GPS modules. You can get more information from Wikipedia: [http://en.wikipedia.org/wiki/Global_Positioning_System](http://en.wikipedia.org/wiki/Global_Positioning_System).

## Wiring the Parallax GPS receiver module

The Parallax GPS Receiver is based on the PMB-248 specification and provides a very easy way to add position detection to the Arduino with its small footprint and low cost.

![Wiring the Parallax GPS receiver module](img/7584_10_004.jpg)

The Parallax GPS Receiver: small footprint and accurate

It provides standard raw NMEA01823 strings or even specific user-requested data via the serial command interface. It can track 12 satellites and even WAAS (system only available in USA and Hawaii for helping the GPS Signal calculation).

NMEA0183 is a combined hardware and logic specification for communication between marine electronic devices such as sonars, anemometers, and many others including GPS. A great description of this protocol can be found here: [http://aprs.gids.nl/nmea/](http://aprs.gids.nl/nmea/).

The module provides current time, date, latitude, longitude, altitude speed, and travel direction/heading, among other data.

We can write data to the GPS modules in order to request specific strings. However, if we pull the **/RAW** pin low, some strings are automatically transmitted by the modules. These strings are:

*   $GPGGA: Global Positioning System Fix Data
*   $GPGSV: GPS satellites in view
*   $GPGSA: GPS DOP and active satellites
*   $GPRMC: Recommended minimum specific GPS/Transit data

This data has to be grabbed by the Arduino and eventually used. Let's check the wiring first:

![Wiring the Parallax GPS receiver module](img/7584_10_005.jpg)

The Parallax GPS Receiver wired to the Arduino in automatic mode by pulling low the /RAW pin

The wiring is quite easy.

Yes, the Parallax GPS Receiver only consumes one data pin: digital pin 0\. Let's stop here for two seconds. Didn't we talk about the fact that we cannot use the USB port for serial monitoring and pins 0 and 1 for other serial feature at the same time on Arduino?

With serial communication using Rx/Tx 2 wires, serial software implementation can be full duplex.

In our case, the GPS device sends data to the Arduino Rx pin. This pin (digital pin 0) is wired to the USB Rx pin. At the same time, the Arduino sends data to the computer using the USB Tx pin that is wired to digital pin 1.

Is there a problem in our case here? No. We just have to take care of interferences. We must not send data from the computer via USB to the Arduino because it already receives data on its serial pin 0 from the GPS device. This is the only thing we have to be careful of.

The `Serial.write()` function will write to digital pin 1, and the USB Tx digital pin 1 isn't wired to anything. Therefore, no problem, data will be sent to the USB. The `Serial.read()` function reads from digital pin 0 and USB and we don't send anything from the computer to USB, so it can read digital pin 0 without any problem.

We pull the /RAW pin to low. In this mode, the device pops data out to the Arduino automatically; I mean, without having to request it.

## Parsing GPS location data

Before building any firmware that will be able to use GPS data, we have to know a bit more about what the device is able to transmit.

We can read the datasheet of the GPS device at: [http://www.rcc.ryerson.ca/media/2008HCLParallaxGPSReceiverModuledatasheet.pdf](http://www.rcc.ryerson.ca/media/2008HCLParallaxGPSReceiverModuledatasheet.pdf).

Here is an example of data that can be transmitted:

`$GPRMC,220516,A,5133.82,N,00042.24,W,173.8,231.8,130694,004.2,W*70`

`$GPRMC` defines the type of information sequence sent. The comma is a separator that separates each data field.

Here is the meaning of each field:

1.  UTC time of fix
2.  Data status (`A` means valid position and `V` means warning)
3.  Latitude of the fix
4.  North or South latitude
5.  Longitude of the fix
6.  East or West longitude
7.  Speed over ground (in knots)
8.  Track made good in degrees
9.  UTC date of fix
10.  Magnetic variation in degrees
11.  East or West magnetic variation
12.  Checksum

As soon as we know what data is sent, we can code a parser in our firmware. Here is a possible firmware. You can find it in folder `Chapter10/locateMe`:

[PRE2]

Let's explain the code a bit. At first, I'm defining several variables:

*   `rxPin` is the digital input where the GPS device is wired
*   `byteGPS` is the latest byte read from the GPS using serial communication
*   `line` is a buffer array
*   `commandGPR` is a string related to messages we want to parse
*   `counter` is the index of the index array
*   `correctness` stores the validity of the message
*   `lineCounter` is a counter keeping track of the buffer position of the data
*   `index` stores the position of each of the separators in the GPS data string (",")

In the `setup()` function, we first define digital pin 0 as an input, and then start the serial communication with a rate of 4800 baud as required by serial interface of the Parallax GPS Receiver (remember to always check your date sheets). Then, we are clearing our `line` array buffer by filling it with a space character.

In the `loop()` function, we begin by reading byte from serial input, the digital pin being 0\. If the port isn't empty, we enter it in the second part of the `if` conditional test defined by the `else` block. If it is empty, we just wait for 100 ms then try to read it again.

At first, the parsing begins by putting the data read in the line buffer at this particular index of the array: `lineCounter`. Then, we increment the latter in order to store the data received.

We then print the data read as a raw line to the USB port. It is at this moment that the Serial Monitor can receive and display it as the raw data row we quoted before as an example.

Then, we test the data itself, comparing it to 13\. If it equals 13, it means data communication is finished and we can begin to parse.

We reset the `counter` and `correctness` variables and check if the first 6 characters in the buffer equals `$GPRMC`. For each match, we increment the `correctness` variable.

This is a classic pattern. Indeed, if all the tests are true, it means `correctness` equals `6` at the end. Then we just have to check if `correctness` equals `6` to see if all the tests have been true, and if the first 6 characters equals `$GPRMC`.

If this is the case, we can be sure we have a correct NMEA raw sequence of the type `$GPRMC`, and we can start to actually parse the payload part of the data.

At first, we split our raw string by storing the position in the string of each comma separator. We then do the same with the last part separator, the "*" character. At this point, we are able to distinguish which character belongs to which part of the string, I mean, which part of the raw message.

It is a loop between each value of the raw message, and we test each value using a switch/case structure in order to display the correct sentence introducing each value of the GPS Data message.

The most tricky part, finally, is the last `for()` loop. We don't start as usual. Indeed, we start the `j` index in the loop using the array `index` at the specific position `i`.

Here is a small schematic showing indexes around the raw message:

![Parsing GPS location data](img/7584_10_006.jpg)

Progressively parsing each part of the message according to each separator

We increment progressively according to each separator's position, and we display each value. This is one way of parsing and using location data using a GPS module. This data can be used in many ways, depending on your purpose. I like data visualization, and I made small projects for students with a GPS module grabbing location every 30s while walking in the street and writing it on an EEPROM. Then, I used this data to make some graphs. One I liked a lot is the following:

![Parsing GPS location data](img/7584_10_007.jpg)

Data visualization designed with Processing from a data set provided by a GPS Arduino module

Each line is a timestamp. The size of the line represents the time we spent between two measures of my Arduino-based GPS module. The longer the line is, the more time I spent at this step of the travel.

Your question could be: How did you supply power to your Arduino + GPS module walking in the street?

Now, let's check how we can make the Arduino autonomous using batteries.

# Arduino, battery, and autonomy

The Arduino boards can supply power in two ways:

*   A USB wire from a computer
*   An external power supply

We already used USB for supplying power to the Arduino since the beginning of the section. This is a pretty nice way to begin (and even to make a great project). This is easy and works for many purposes.

We can also use an external power supply when we need more autonomy and mobility with our Arduino devices.

In any case, we have to keep in mind that both our Arduino and our circuits wired to it need power. Usually, the Arduino consumption is no more than 50mA. Add some LEDs and you'll see the consumption increase.

Let's check some cases of real use.

## Classic cases of USB power supplying

Why and when would we use a USB power supply?

Obviously, if we need our computer connected to our Arduino for data communication purposes, we can naturally supply power to the Arduino through the USB.

This is the main reason for using a USB power supply.

There are also some cases where we cannot have a lot of power sockets. Sometimes, there are many constraints in installation design projects and we don't have a lot of power sockets. This is also one case of supplying power using the USB.

Basically, the first thing to bear in mind before using power supplied by the USB port is the global consumption amount of our circuit.

Indeed, as we have already learned, the maximum current a USB port can provide is around 500mA. Be sure you don't exceed this value. Above this limit of consumption, things become totally unpredictable and some computers can even reboot while some others can disable all USB ports. We have to keep that in mind.

## Supplying external power

There are two different ways to supply power to an Arduino-based system. We can state the two main power supplies as:

*   Batteries
*   Power adapters

### Supplying with batteries

If we remember correctly, the Arduino Uno and Mega for instance can operate on an external power supply of 6 V to 20 V. For stable use, the recommended range is 7 V to 12 V. 9 V is an ideal voltage.

In order to set the board to external power supply, you have to take care of the power jumper. We have to put it on the external power supply side, named EXT. This setup is for the Arduino Diecimilla and older the Arduino boards:

![Supplying with batteries](img/7584_10_008.jpg)

The power supply jumper put on the EXT side, meaning set up to external power supply

Let's check the basic wiring with a 9 V battery:

![Supplying with batteries](img/7584_10_009.jpg)

A 9V battery wired to an Arduino board UNO R3

This simple wiring provides a way to supply power to the Arduino board. If you plug some other circuits to the Arduino, the battery through the Arduino will feed them.

There are also some other types of batteries that we can use. Coin cell batteries are a nice way to save space while supplying power externally:

![Supplying with batteries](img/7584_10_010.jpg)

A classic coin cell battery

There are many type of coin cell holders to use this type of battery in our circuits. Usually, coin cell batteries provide 3.6 V at 110 mAh. If this cannot supply power to the Arduino Uno, it can easily supply the Arduino Pro Mini working at a voltage of 3.3 V:

![Supplying with batteries](img/7584_10_011.jpg)

Arduino Pro Mono

The Arduino Pro Mini board is really interesting as it can be embedded in many circuits that need to be discrete and sometimes hidden in walls for digital art installations or put into a small plastic box that can be carried in a pocket when they are used as a mobile tool.

We can also use polymer lithium-ion batteries. I used them a couple of times for an autonomous device project.

However, we can have some projects that require more power.

## Power adapter for Arduino supply

For projects requiring more power, we have to use an external power supply. The setup of the Arduino stays the same as with batteries. The off-the-shelf Arduino adapter has to meet some requirements:

*   DC adapter (No AC adapter here!)
*   Output voltage between 9V and 12V DC
*   Able to output a minimum current of 250mA but aim at 500mA at least or preferably 1A
*   Must have a centre positive 2.1mm power plug

Here are the patterns you have to look for on your adapter before plugging in the Arduino.

First, the center of the connector has to be the positive part; check out the following diagram. You should see that on an Arduino-compatible adapter:

![Power adapter for Arduino supply](img/7584_10_012.jpg)

The symbol showing the center positive plug

Then, the voltage and current characteristics. This has to show something like: OUTPUT: 12 VDC 1 A. This is an example; 12 VDC and 5 A is also fine. Don't forget that current is only driven by what's there in your circuit. A power adapter that puts out a higher current will not harm your circuit, because a circuit will only draw what it needs.

A lot of adapters are available in the market and can be used with our Arduino boards.

## How to calculate current consumption

In order to calculate current in your circuit, you have to use Ohm's law as described in the first chapter of this book.

When you check the datasheet of a component, like an LED, you can see that the current passed through it.

Let's check the RGB LED Common Cathode with this datasheet: [https://www.sparkfun.com/datasheets/Components/YSL-R596CR3G4B5C-C10.pdf](https://www.sparkfun.com/datasheets/Components/YSL-R596CR3G4B5C-C10.pdf)

We can see a forward current of 20 mA and a peak forward current of 30 mA. If we have five LEDs like that switched on at the maximum brightness (that is red, blue, and green lighted up), we have: 5 x (20 + 20 + 20) = 300 mA needed for normal use and even peaks would consume 5 x (30 + 30 + 30) = 450 mA.

This is in the case where all LEDs are fully switched on at the same time.

You must have understood the strategy we already used in power supply cycling, switching on each LED one after the other in quick succession. This provides a way to reduce the power consumption and also allow some projects to use a lot of LEDs without requiring an external power adapter.

I won't describe the calculations for each case here, but you'd have to refer to electricity rules to precisely calculate the consumption.

By experience, there is nothing better than your voltmeter and Ampere meter, the former measuring voltage between two points and the latter measuring current at some points along the circuit.

I'd suggest that you make some calculations to be sure to:

*   Not override the Arduino capacity per pins
*   Not override USB 450mA limit, in case you use a USB power supply

Then, after that, begin to wire and measure at the same time with voltmeter and Ampere meter.

At last, a classic reference for most of the Arduino boards is available at this page: [http://playground.arduino.cc/Main/ArduinoPinCurrentLimitations](http://playground.arduino.cc/Main/ArduinoPinCurrentLimitations).

We can find the limitations for current consumption for each part of the Arduino.

# Drawing on gLCDs

Drawing is always fun. Drawing and handling LCD displays instead of LEDs matrices is really interesting too, because we have devices with high-density points we can switch on and off easily.

LCDs exist in many types. The two main types are the character and graphical type.

We are talking about the graphical type here, especially those based on the KS0108 graphics-only controller used in many regular gLCD devices.

We are going to use a nice library that is available on Google. It has code by Michael Margolis and Bill Perry, and it is named `glcd-arduino`. This library is licensed under the GNU Lesser GPL.

Let's download it here: [http://code.google.com/p/glcd-arduino/downloads/list](http://code.google.com/p/glcd-arduino/downloads/list). Download the most recent version.

Unzip it, put it in the place where all your libraries are, and restart or start your Arduino IDE.

You should now see a lot of examples related to the gLCD library.

We won't check all the nice features and functions provided by this library here, but you can check this page on the Arduino website: [http://playground.arduino.cc/Code/GLCDks0108](http://playground.arduino.cc/Code/GLCDks0108).

## Wiring the device

We are going to check the wiring of a KS0108 based gLCD type Panel B:

![Wiring the device](img/7584_10_013.jpg)

A lot of wires wiring the gLCD to Arduino and the potentiometer to adjust LCD contrast

The corresponding electrical diagram is as follows:

![Wiring the device](img/7584_10_014.jpg)

KS0108 based gLCD type Panel B wired to an Arduino Uno R3

These are a lot of wires. Of course, we can multiply things. We can also use an Arduino MEGA and keep using the other digital pin available for other purposes, but that is not the point here. Let's check some of the functions of this powerful library.

## Demoing the library

Take the example named `GLCDdemo`. It shows you almost all the functions available in the library.

There is very good PDF documentation provided with the library. It explains each available method. You can find it in the `library` folder in the `doc` subfolder:

![Demoing the library](img/7584_10_015.jpg)

The documentation of gLCD-Arduino showing the screen coordinates system

At first, we have to include `glcd.h` in order to use the library. Then, we have to include some other headers, in this example, fonts and bitmap in order to use the font typographic methods and the bitmap objects too.

## Some useful method's families

I'd suggest ordering learning methods into three parts:

*   Global GLCD methods
*   Drawing methods
*   Text methods

### Global GLCD methods

The first is the `init()` function. This one initializes the library and has to be called before any other gLCD methods.

The `SetDisplayMode()` function is useful because it sets up the use of the LCD as normal (writing in black over white background) or inverted. White just means not black. The real color depends on the backlight color, of course.

The `ClearScreen()` function erases the screen, filling it with white background in normal mode, or black in inverted mode.

The `ReadData()` and `WriteData()` functions are really raw methods that get and set the byte of data at particular coordinates.

### Drawing methods

These are a set of functions dedicated to drawing on the screen.

The set of constants are as follows:

*   `GLCD.Width` is the display width in pixels
*   `GLCD.Height` is the display height in pixels
*   `GLCD.Right` is the last pixel column at the right (equals GLCD.Width – 1)
*   `GLCD.Bottom` is the last pixel row at the bottom (equals GLCD. Height – 1)
*   `GLCD.CenterX` and `GLCD.CenterY` are the coordinates of the pixel in the middle

Basically, you can draw by moving the graphics cursor and by drawing primitive shapes:

| Function | Description |
| --- | --- |
| `GotoXY()` | Moves the cursor to specific coordinates |
| `DrawVLine()` | Draws a vertical line from a point to another point in the same pixel column but above or below the initial point |
| `DrawHLine()` | Works the same as `DrawVLine()` but on the same pixel row |
| `DrawLine()` | Draws a line between two coordinates |

Some other, more complex shapes can be drawn too:

| Functions | Descriptions |
| --- | --- |
| `DrawRect()` | Draws a rectangle from a point when provided with a width and height. |
| `FillRect()` | Works the same as `DrawRect()`, but by filling the rectangle shape with black (or white) pixels. |
| `DrawRoundRect()` | Draws a nice rectangle with rounded corners. |
| `DrawCircle()` and `FillCircle()` | Draws a circle from coordinates and a radius, and a circle filled with black (or white) pixels. |
| `DrawBitmap()` | Draws a whole bitmap at a particular position on the screen. It uses a pointer to that bitmap in memory. |

With this set of functions, you can basically draw anything you want.

### Text methods

These are a set of functions dedicated to typography on the screen:

| Functions | Descriptions |
| --- | --- |
| `SelectFont()` | At first, this chooses the font to be used in the next functions calls. |
| `SetFontColor()` | Chooses the color. |
| `SetTextMode()` | Chooses a scrolling direction. |
| `CursorTo()` | Moves the cursor to a specific column and row. The column calculation uses the width of the widest character. |
| `CursorToXY()` | Moves the cursor to particular pixel coordinate. |

One important feature to know about, is the fact that Arduino's print functions can be used with gLCD library; `GLCD.print()` works fine, for instance. There are also a couple of other functions available that can be found on the official website.

At last, I'd suggest you to test the example named `life`. This is based on the John Conway's Game of Life. This is a nice example of what you can do and implement some nice and useful logic.

Drawing on gLCD is nice, but we could also use a small module handling VGA.

# Using VGA with the Gameduino Shield

Gameduino is an Arduino Shield. This is the first one we are using here in this book. Basically, a shield is a PCB (printed circuit board) that can be plugged to another PCB, here our Arduino.

Arduino Shields are pre-made circuits including components and sometimes processors too. They add features to our Arduino board by handling some specific tasks.

Here, the Gameduino will add VGA drawing abilities to our Arduino that can't be done on its own.

The Gameduino adds a VGA port, a mini-jack for the sound, and also includes an FPGA Xilling Spartan3A. FPGA Xilling Spartan3A can process graphical data faster than the Arduino itself. Arduino can control this graphical hardware driver by SPI interface.

Let's see how it works:

![Using VGA with the Gameduino Shield](img/7584_10_016.jpg)

The Gameduino controller Arduino Shield

Arduino Shields can be plugged in Arduino boards directly. Check the following screenshot:

![Using VGA with the Gameduino Shield](img/7584_10_017.jpg)

The Gameduino plugged in the Arduino board

Here are some characteristics of the Gameduino:

*   Video output is 400 x 300 pixels in 512 colors
*   All color processed internally at 15 bit precision
*   Compatible with any standard VGA monitor (800 x 600 @ 72 Hz)
*   Background graphics (512 x 512 pixel character, 256 characters)
*   Foreground graphics (sprite 16 x 16 abilities, transparency, rotate/flip, and sprite collision detection)
*   Audio output as stereo; 12-bit frequency synthesizer
*   64 independent voices at 10 to 8000 hz
*   Sample playback channel

The underlying concept is to plug it in the Arduino and to control it using our Arduino firmware with the library taking care of all SPI communication between the Arduino and Gameduino.

We cannot describe all the examples right here in this book, but I want to point you in the right direction. At first, the official website: [http://excamera.com/sphinx/gameduino/](http://excamera.com/sphinx/gameduino/).

You can find the library here: [http://excamera.com/files/gameduino/synth/sketches/Gameduino.zip](http://excamera.com/files/gameduino/synth/sketches/Gameduino.zip).

You can also check and use the quick reference poster here: [http://excamera.com/files/gameduino/synth/doc/gen/poster.pdf](http://excamera.com/files/gameduino/synth/doc/gen/poster.pdf).

For your information, I'm currently designing a piece on digital art installation based on this shield. I intend to describe it on my own website [http://julienbayle.net](http://julienbayle.net) and the whole schematics will be provided too.

# Summary

In this first, advanced chapter, we learned a bit more about how to deal with new concrete concepts such as storing data on non-volatile memories (internal and external EEPROM), use GPS module receivers, draw on graphical LCD, and use a nice Arduino Shield named Gameduino to add new features and power to our Arduino. This allowed it to display a VGA signal and also to produce audio. We also learned the use of Arduino as a very portable and mobile device, autonomous from the power supply point of view.

In the next chapter, we are going to talk about networking concepts. Creating and using networks are usual ways of communication today. We will describe wired and wireless network use with our Arduino projects in the next chapter.