# Chapter 7. Talking over Serial

We already saw that using Arduino is all about talking and sharing signals. Indeed, from the most basic component in Arduino, reacting to some physical world values by changing its environment and propagating the change as a basic message to its neighbors, to the now classic serial communication, electronic entities are talking among themselves and to us.

As with the many concepts in this book, we have already used serial communication and the underlying Serial protocol a couple of times as a black-boxed tool, that is, a tool I have introduced but not explained.

We are going to dive into it in this small chapter. We will discover that serial communication is used not only for machine-to-human communication but also for "component-to-component" discussions inside machines. By components, I mean small systems, and I could use the term peripheral to describe them.

# Serial communication

Typically, serial communication in computer science and telecommunications is a type of communication where data is sent one bit at a time over a communication bus.

Nowadays, we can see serial communication all around us, and often we don't even realize this. J The "S" in the **USB** acronym means Serial (USB is **Universal Serial Bus**), and represents the underlying serial communication bus used by every higher protocol.

Let's dig into that right now.

## Serial and parallel communication

Serial communication is often defined by its opposite form of communication, **parallel communication**, where several bits of data are sent out over a link made by several parallel channels at the same time. Look at the following figure:

![Serial and parallel communication](img/7584_07_01.jpg)

Basic, unidirectional serial communication between a speaker and a listener

Now let's compare this to a parallel case:

![Serial and parallel communication](img/7584_07_02.jpg)

Basic, unidirectional parallel communication between a speaker and a listener

In these two figures, a speaker is sending the following data byte: `0 1 1 0 0 0 1 1`. These eight bits of data are sent sequentially over one channel in the case where serial communication has been used, and simultaneously over eight different channels in the case where parallel communication has been used.

Right from small-distance to long-distance communications, even if the parallel approach seems faster at first glance because more than one bit of data is sent at the same time during a clock cycle, serial communication has progressively outperformed other forms of communication.

The first reason for this is the number of wires involved. For example, the parallel approach used in our small example requires eight channels to drive our eight bits of data at the same time, while the serial requires only one. We'll discuss what a channel is very soon, but with one wire, the ratio 1:8 would save us money if we were to use serial communication.

The second major reason is the fact that we finally achieved to make serial communication very fast. This has been achieved due to the following:

*   Firstly, **propagation time** is easier to handle with a smaller number of wires
*   Secondly, **crosstalk** is less with fewer channels than with a higher density of channels such as those found in parallel links
*   Thirdly, because there are fewer wires involved, we can save space (and money) and often use this saved space to shield our wires better

Nowadays, serial communication bandwidths range from several megabits per second to more than 1 terabit per second (which means 1,000 gigabits per second), and a lot of media can be used from wire-driven fibers to wireless, and from copper cables to optical fibers. As you might suspect, there are many serial protocols that are used.

## Types and characteristics of serial communications

Whether it be synchronism or duplex mode or bus or peering, serial communication can be defined differently, and we have to dig into that point here.

### Synchronous or asynchronous

Serial communication can either be synchronous or not.

**Synchronous** communication involves a clock, which we can call a master clock, that keeps a reference time for all the participants of the communication. The first example that comes to mind is phone communication.

**Asynchronous** communication doesn't require that the clock's data be sent over the serial channel(s); this makes it easier to communicate but it can lead to some issues with understandability at times. Mailing and texting are asynchronous types of communication.

### Duplex mode

The duplex mode is a particular characteristic of a communication channel. It can be:

*   **Simplex**: Unidirectional only (data is passed only in one direction, between two points)
*   **Half-duplex** : Bidirectional, but only in one direction at the same time
*   **Full-duplex** : Bidirectional simultaneously

Half-duplex is obviously more useful than simplex, but it has to run a collision detection and retransmission process. Indeed, when you are talking to your friend, you are also sharing the same media (the room and air inside the room that carries vibrations from your mouth to his ears), and if you are talking at the same time, usually one checks that and stops and tells the other to repeat.

Full-duplex requires more channels. That way no collisions occur and all the collision detection and retransmission processes can be dropped. The detection of other errors and fixing is still involved, but usually it is much easier.

### Peering and bus

In a **peering** system, the speakers are linked to listeners either physically or logically. There is no master, and these kinds of interfaces are most often asynchronous.

In a **bus**, they will all get connected physically at some point and some logical commutations will occur.

![Peering and bus](img/7584_07_08.jpg)

An example of a multibus system

#### Master and slave buses

In master/slave buses, one device is the master and the others are the slaves, and this usually involves synchronism where the master participant generates the timing clock.

The main difficulty with serial communication is to avoid collisions and misunderstandings.

There are a lot of solutions that can be implemented to solve these problems, such as using multiple physical link types and specific preexisting communication protocols. Let's check some of these, and especially those we can use with Arduino of course.

### Data encoding

The most important things to define when we use serial protocols for our communication are as follows:

*   The word length in bits
*   Whether a stop bit is present or not (defines a blank moment in time)
*   Whether a parity bit is present or not (defines the simplest, error-detecting, code-based solution)

Indeed, especially in asynchronous communication, how could a listener know where a word begins or ends without these properties? Usually, we hardcode this behavior in both the participants' brains in order to be sure we have a valid communication protocol.

In the first figure of this chapter, I sent eight bits of data over the channel. This equals 1 byte.

We often write the types of serial communication as `<word length><parity><stop>`. For instance, 8 bit without parity but one stop is written as `8N1`.

I won't describe the parity bit completely, but you should know that it is basically a checksum. Using this concept, we transmit a word and checksum, after which we verify the binary sum of all the bits in my received word. In this way, the listener can check the integrity of the words that were received quite easily, but in a very primitive way. An error can occur, but this is the cheapest way; it can avoid a lot of errors and is statistically right.

A global frame of data with the `8N1` type serial communication contains 10 bits:

*   One start bit
*   Eight bits for each characters
*   One stop bit

Indeed, only 80 percent of the data sent is the real payload. We are always trying to reduce the amount of flow control data that is sent because it can save bandwidth and ultimately time.

# Multiple serial interfaces

I won't describe all the serial protocols, but I'd like to talk about some important ones, and sort them into families.

## The powerful Morse code telegraphy ancestor

I give you one of the oldest Serial protocols: the Morse code telegraphy protocol. Telecommunications operators have been using this one since the second half of the 19th century.

I have to say that Samuel F. B. Morse was not only an inventor but also an accomplished artist and painter. It is important to mention this here because I'm really convinced that art and technology are finally one and the same thing that we used to see with two different points of view. I could quote more artist/inventor persons but I guess it would be a bit off topic.

By sending long and short pulses separated by blanks, Morse's operators can send words, sentences, and information. This can happen over multiple types of media, such as:

*   Wires (electrical pulses)
*   Air (electromagnetic wave carriers, light, sounds)

It can be sorted into a peered, half-duplex, and asynchronous communication system.

There are some rules about the duration of pulses ranging from long to short to blank, but this remains asynchronous because there isn't really a clock shared between both participants.

### The famous RS-232

RS-232 is a common interface that you will find on all personal computers. It defines a complete standard for electrical to physical (and electrical to mechanical) characteristics, such as connection hardware, pins, and signal names. RS-232 was introduced in 1962 and is still widely used. This point-to-point interface can drive data up to 20 Kbps (kilobit per second = 20,000 bits per second) for moderate distances. Even though it isn't specified in the standard, we will usually find instances where the speed is greater than 115.2 Kbps on short and shielded wires.

I myself use cables that are 20 meters long with sensors that transmit their data over serial to Arduino for different installations. Some friends use cables that are 50 meters long, but I don't do that and prefer other solutions such as Ethernet.

#### From 25 wires to 3

If the standard defines a 25-pin connector and link, we can reduce this huge number required for multiple hardware flow control, error detections, and more to only three wires:

*   Transmit data (usually written as TX)
*   Receive data (usually written as RX)
*   Ground

The connector with 25 pins/wires is named DB25 and has been used a lot, for peripherals such as printers. There is another type of connector named DB9 with 9 pins/wires only. This is a variant that omits more wires than DB25\. This DB9 has been used a lot for connecting mouse devices.

But how can we omit a large number of wires/signals and keep the serial communication working well? Basically, as with many standards, it has been designed to fit a lot of use cases. For instance, in the full version of DB25, there are pins 8 and 22 that are dedicated to phone lines: the first one is the **Data Carrier Detect** and the second one is the **Ring Indicator** . The signal sent over pins 4 and 5 is used for a handshake between the participants.

In this standard, pin 7 is the common ground and 2 and 3 are respectively TX and RX. With only these three, we can make our serial asynchronous communication correctly.

![From 25 wires to 3](img/7584_07_03.jpg)

The DB25 connector

![From 25 wires to 3](img/7584_07_04.jpg)

The DB9 connector

Our precious Arduino provides this three-wire serial alternative. Of course, each type of board doesn't provide the same number of serial interfaces, but the principle remains the same: a serial interface based on three-wire is available.

Arduino Uno and Leonardo provide the three wires TX, RX, and ground, while freshly released Arduino Mega 2560 and Arduino Due ([http://arduino.cc/en/Main/ArduinoBoardDue](http://arduino.cc/en/Main/ArduinoBoardDue)) provide four different serial communication interface names right from RX0 and TX0 to RX3 and TX3.

We are going to describe another type of serial interface standard, and we'll come back to RS-232 with the famous integrated circuit made by FTDI that provides a very efficient way to convert RS-232 to USB.

### The elegant I2C

The I2C multimaster serial single-ended computer bus has been designed by Philips and requires a license for any hardware implementations.

One of its advantages is the fact that it uses only two wires: **SDA** (**Serial Data Line**) with a 7-bit addressing system and **SCL** (**Serial Clock Line**).

This interface is really nice considering its addressing system. In order to use it, we have to build the two wire-based bus from Arduino, which is the master here.

![The elegant I2C](img/7584_07_06.jpg)

BlinkM modules connected as an I2C bus to the Arduino Uno R3

In order to know which pins have to be used for each Arduino board, you can directly check the information at [http://www.arduino.cc/en/Reference/Wire](http://www.arduino.cc/en/Reference/Wire).

**BlinkM** modules ([http://thingm.com/products/blinkm](http://thingm.com/products/blinkm)) are RGB LED modules with a small form factor that are quite easy to manipulate on I2C buses. I also used it a lot to more or less control big LCDs with Arduino.

This is also the page of the `Wire` library for Arduino. Nowadays, this library is included with the Arduino core. Considering the complexity of the standard, the cost increases when you have a lot of elements on the buses. Because of its two wires and the precision of data integrity, this is still an elegant solution for short-distance and intermittent communication inside the same box. The **Two Wire Interface** (**TWI**) is principally the same standard as I2C. It was known by another name when the patents on I2C were still running.

I2C has been the base for many other interface protocols, such as VESA DDC (a digital link between screens and graphical card), SMBus by Intel, and some others.

### The synchronous SPI

**SPI** stands for **Serial Peripheral Interface**, which has been developed by Motorola. It uses the following four wires:

*   **SCLK**: This is the serial clock driven by the master
*   **MOSI**: This is the master output / slave input driven by the master
*   **MISO**: This is the master input / slave output driven by the master
*   **SS**: This is the slave-selection wire

It is very useful in point-to-point communication where there is only one master and one slave, even if we find many applications with more than one slave on the SPI bus.

Since SPI is a full-duplex, mode-based interface, we can achieve higher data rates than we can with I2C. It is often used for communication between a coder/decoder and a digital signal processor; this communication consists of sending samples in and out at the same time. SPI lacking device addressing is also a huge advantage as it makes it much lighter and thus faster in case you don't need this feature. Indeed, I2C and SPI are really complementary to each other depending on what you want to achieve.

There is information available online regarding SPI in the Arduino boards ([http://arduino.cc/en/Reference/SPI](http://arduino.cc/en/Reference/SPI)), but you have to know that we can easily use any digital pins as one of the four wires included in SPI.

I personally have often used it in projects involving a lot of shift registers that are all daisy-chained to have a lot of inputs and/or outputs with Arduino Uno and even Arduino Mega, this latter offering more outputs and inputs natively.

We'll describe the use of shift registers in the next chapter when I show you how to multiplex outputs quite easily with some smart and, ultimately, very simple integrated circuits linked to Arduino through SPI.

### The omnipresent USB

USB is the Universal Serial Bus standard. This is probably the one you use the most.

The main advantage of this standard is the Plug and Play feature of USB devices. You can plug and unplug devices without restarting your computer.

USB has been designed to standardize the connection of a wide variety of computer peripherals, including the following:

*   Audio (speaker, microphone, sound card, MIDI)
*   Communications (modem, Wi-Fi, and Ethernet)
*   Human interface device (HID, keyboard, mouse, joystick)
*   Image and video (webcam, scanner)
*   Printer
*   Mass storage (flash drive, memory card, drive)
*   Wireless (infrared)

And there are many more types too. The standard is actually Version 3.0\. A USB bus can contain up to 127 peripherals and can supply a maximum of 500 to 900 mA for general devices.

#### USB system design

The architecture of USB is an asymmetrical topology consisting of one host and a multitude of downstream USB ports and multiple peripheral devices connected in a tiered-star topology.

USB hubs can be included in the tiers allowing branching up to five tier levels. This results in a tree topology. This is why you can stack hubs on hubs.

Device classes provide a way of having an adaptable and device-independent host to support new devices. An ID that the host can recognize defines each class. You can find all the approved classes at [http://www.usb.org/developers/devclass](http://www.usb.org/developers/devclass) on the official USB standard website.

#### USB connectors and cables

A USB standard plug contains four wires ([http://en.wikipedia.org/wiki/Universal_Serial_Bus](http://en.wikipedia.org/wiki/Universal_Serial_Bus)):

*   Vcc (+5 V)
*   Data-
*   Data+
*   Ground![USB connectors and cables](img/7584_07_07.jpg)

    The USB standard A and B plugs

Cables are shielded; their usual maximal lengths are around two to five meters. I already used a 12-meter cable for the USB port. It worked totally fine with a cable that I myself soldered in an electromagnetic-safe environment, I mean, in a place where my cable was alone behind a wall and not mixed with a lot of other cables, especially the ones supplying power.

There are some other types of plug that are somewhat bigger, but the requirement of having at least four wires remains the same.

#### FTDI IC converting RS-232 to USB

Except for some versions, such as the Arduino Pro Mini, Arduino boards provide a USB connector, as you already know and have used.

This provides the basic power supply feature for a computer or the hubs connected to a computer, and it is used for communication too.

The FTDI integrated circuit EEPROM named FT232 provides a way of converting USB into an RS-232 interface. This is why we can use the serial communication features of the Arduino boards over USB without the need for an external serial port interface from the Arduino pins related to serial communication, which are TX and RX. New boards include an Atmega16U2 that provides serial communication features.

Indeed, as soon as you connect your Arduino board to a computer, you will have a serial communication feature available. We already used it with:

*   Arduino IDE (Serial Monitor)
*   Processing (with the serial library)
*   Max 6 (with the serial object)

I guess you also recall that we weren't able to use the Serial Monitor while using Max 6's serial object polling feature.

Do you understand why now? Only one point-to-point link can be active at the same time on the wires and in the virtual world of computers. It's the same for physical links, too. I warned you not to use the digital pins 0 and 1 as soon as you needed to use serial communication with the Arduino board, especially the Diecimilla version. These pins are directly connected to the corresponding RX and TX pins of the FTDI USB-to-TTL serial chip.

### Note

If you use serial communication over the USB feature, you have to avoid using the digital pins 0 and 1.

# Summary

In this chapter, we talked about serial communication. This is a very common mode of communication both inside and between electronic devices. This chapter is also a nice introduction to other communication protocol in general, and I'm sure that you are now ready to understand more advanced features.

In the next chapter, we'll use some of the different types of serial protocol that were introduced here. In particular, we are going to talk about Arduino outputs; this means that not only will we be able to add feedback and reactions to our Arduino boards, considering behavior pattern designs such as stimulus and response for deterministic ways, but we will also see more chaotic behaviors such as those including constrained chance, for instance.