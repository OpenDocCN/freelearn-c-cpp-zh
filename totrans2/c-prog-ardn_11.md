# Chapter 11. Networking

In this chapter we are going to talk about linking objects and making them talk by creating communication networks. We are going to learn how we can make multiple Arduinos and computers communicate using network links and protocols.

After defining what a network is (specifically, a data network), we'll describe ways to use wired Ethernet links between Arduinos and computers. This will open the Arduino world to the Internet. Then, we'll discover how to create Bluetooth communications.

We will learn how to use Ethernet Wi-Fi in order to connect the Arduino to computers or other Arduinos without being tethered by network cables.

At last, we will study a couple of examples from the one in which we will fire message to the micro-blogging service Twitter, to the one in which we will parse and react to data received from the Internet.

We will also introduce the OSC exchange protocol, widely used in anything related to interaction design, music, and multimedia.

# An overview of networks

A network is a system of elements linked together. There are many networks around us such as highway systems, electrical grids, and data networks. Data networks surround us. They relate to video services networks, phone and global telecommunication networks, computer networks, and so on. We are going to focus on these types of networks by talking about how we can share data over different types of media such as wires transporting electric pulses or electromagnetic waves facilitating wireless communication.

Before we dive into the details of network implementations for Arduino boards, we are going to describe a model named the OSI model. It is a very useful representation of what a data network is and what it involves.

## Overview of the OSI model

The **Open** **Systems** **Interconnection** model (**OSI** model) has been initiated in 1977 by the International Organization for Standardization in order to define prescriptions and requirements around the functions of communication systems in terms of abstract layers.

Basically, this is a layers-based model describing what features are necessary to design communicating systems. Here is the OSI model with seven layers:

![Overview of the OSI model](img/7584_11_001.jpg)

OSI model describing communication system requirements with seven abstraction layers

## Protocols and communications

A communications protocol is a set of message formats and rules providing a way of communication between at least two participants. Within each layer, one or more entities implements its functionality and each entity interacts directly and only with the layer just beneath it and at the same time provides facilities for use by the layer above it. A protocol enables an entity in one host to interact with a corresponding entity at the same layer in another host. This can be represented by the following diagram:

![Protocols and communications](img/7584_11_002.jpg)

Protocols helping hosts' layers to communicate together

## Data encapsulation and decapsulation

If the application of one host needs to send data to another application of another host, the effective data, also named the payload, is passed down directly to the layer beneath it. In order to make the application able to retrieve its data, a header and footer are added to this data depending on the protocol used at each layer. This is called **encapsulation** and it happens until the lowest layer, which is the physical one. At this point, a flow of bits is modulated on the medium for the receiver.

The receiver has to make the data progressively climb the layer stack, passing data from a layer to a higher layer and addressing it to the right entities in each layer using previously added headers and footers. These headers and footers are removed all along the path; this is called **decapsulation**.

At the end of the journey, the application of the receiver receives its data and can process it. This whole process can be represented by the following diagram:

![Data encapsulation and decapsulation](img/7584_11_003.jpg)

Encapsulation and decapsulation all along the layers' stack

We can also represent these processes as shown in the following figure. The small gray rectangle is the data payload for the layer N+1.

![Data encapsulation and decapsulation](img/7584_11_004.jpg)

Adding and removing specific headers and footers according to the protocols used

At each level, two hosts interact using a protocol transmitted, which we call **Protocol Data Unit** or **PDU**. We also call **Service Data Unit** or **SDU**, a specific unit of data passed down from a layer to a lower layer and that has not yet been encapsulated.

Each layer considers the data received as data for it and adds/removes headers and footers according to the protocol used.

We are now going to illustrate each layer and protocol by examples.

## The roles of each layer

We are going to describe the purpose and roles of each layer here.

### Physical layer

The physical layer defines electrical and physical specifications required for communication.

Pin layout, voltages and line impedance, signal timing, network adapters, or host bus adapters are defined in this layer. Basically, this layer performs three major functions/services:

*   Initialization and termination of a connection to a communication medium
*   Participation in shared resources control processes
*   Conversion between the data communicated and the electrical signals which carry them

We can quote some known standards being in this physical layer:

*   ADSL and ISDN (network and phony provider services)
*   Bluetooth
*   IEEE 1394 (FireWire)
*   USB
*   IrDA (data transfer over infrared links)
*   SONET, SDH (wide area optic fiber networks operated by providers)

### Data link layer

This layer is made of two sublayers:

*   Logical Link Control (LLC)
*   Media Access Control (MAC)

Both are responsible for transferring data between network entities and to detect errors that may occur at the physical layer, and eventually to fix them. Basically, this layer provides these functions/services:

*   Framing
*   Physical addressing
*   Flow control
*   Error control
*   Access control
*   Media access control

We can quote some known standards of this data link layer:

*   Ethernet
*   Wi-Fi
*   PPP
*   I2C

We have to keep in mind that the second layer is also the domain of local area networks with only physical addresses. It can be federated using LAN switches.

By the way, we often need to segment networks and also communicate wider and so we need another addressing concept; this introduces the network layer.

### Network layer

This layer provides the way to transfer data sequences between hosts that can be in different networks. It provides the following functions/services:

*   Routing
*   Fragmentation and reassembly
*   Delivery error reports

Routing provides a way to make hosts on a different network able to communicate by using a network addressing system.

Fragmentation and reassembly also occur at this level. These provide a way to chop data streams into pieces and to be able to reassemble parts after the transmission. We can quote some known standards in this layer:

*   ARP (resolving and translating physical MAC address into network address)
*   BOOTP (providing a way for the host to boot over the network)
*   BGP, OSPF, RIP, and other routing protocols
*   IPv4 and IPv6 (Internet Protocol)

Routers are usually the gear where the routing occurs. They are connected to more than one network and make data going from one network to another. This is also the place where we can put some access lists in order to control access based on IP addresses.

### Transport layer

This layer is in charge of the data transfer between end users, being at the crossroads of network layers and application layers. This layer provides the following functions/services:

*   Flow control to assure reliability of the link used
*   Segmentation/desegmentation of data units
*   Error control

Usually, we order protocols in two categories:

*   State-oriented
*   Connection-oriented

This means this layer can keep track of segments emitted and eventually retransmit them in case of previously failed transmission.

We can quote the two well-known standards of the IP suite in this layer:

*   TCP
*   UDP

TCP is the connection-oriented one. It keeps the communication reliable by checking a lot of elements at each transmission or at each x segments transmitted.

UDP is simpler and stateless. It doesn't provide a communication state control and thus is lighter. It is more suited for transaction-oriented query/response protocol such as DNS (Domain Name System) or NTP (Network Time Protocol). If there is something wrong, such as a segment not transmitted well, the above layer has to take care of resending a request, for instance.

### Application/Host layers

I grouped the highest three layers under the terms application and host.

Indeed, they aren't considered as network layers, but they are part of OSI model because they are often the final purpose of any network communication.

We find a lot of client/server applications there:

*   FTP for basic and light file transfers
*   POP3, IMAP, and SMTP for mail services
*   SSH for secure remote shell communication
*   HTTP for web server browsing and downloading (and nowadays much more)

We also find a lot of standards related to encryption and security such as TLS (Transport Layer Security). Our firmware, an executing Processing code, Max 6 running a patch are in this layer.

If we want to make them communicate through a wide variety of networks, we need some OSI stack. I mean, we need a transport and network protocol and a medium to transport our data.

If our modern computers own the whole network stack ready to use, we have to build this later in our Arduino's firmware if we want them to be able to communicate with the world. This is what we are going to do in the next subchapter.

## Some aspects of IP addresses and ports

One of the protocol stacks we tend to use each day is the TCP/IP one. TCP is the layer 4 transport protocol, and IP the layer 3 network.

This is the most used network protocol in the world both for end users and for companies.

We are going to explain a little bit more about the IP addressing system, subnet masks, and communication ports. I won't be describing a complete network course.

### The IP address

An IP address is a numerical address referenced by any devices wanting to communicate over an IP network. IP is currently used in 2 versions: IPv4 and IPv6\. We are considering IPv4 here because it is currently the only one used by end users. IPv4 addresses are coded over 32 bits. They are often written as a human-readable set of 4 bytes separated by a point. 192.168.1.222 is the current IP address of my computer. There are 2^(32) possible unique addresses and all aren't routable over the Internet. Some are reserved for private use. Some companies assign Internet-routable addresses. Indeed, we cannot use both addresses as this is handled by global organizations. Each country has sets of addresses attributed for their own purposes.

### The subnet

A subnet is a way to segment our network into multiple smaller ones. A device network's configuration contains usually the address, the subnet mask, and a gateway.

The address and the subnet mask define the network range. It is necessary to know if a transmitter can communicate directly with a receiver. Indeed, if the latter is inside the same network, communication can occur directly; if it is on another network, the transmitter has to send its data to the gateway that will route data to the correct next node on the networks in order to reach, if possible, the receiver.

The gateway knows about the networks to which it is connected. It can route data across different networks and eventually filter some data according to some rules.

Usually, the subnet mask is written as a human-readable set of 4 bytes too. There is obviously a bit notation, more difficult for those not used to manipulating the numbers.

The subnet mask of my computer is 255.255.255.0\. This information and my IP address defines that my home network begins at 192.168.1.0 (which is the base network address) and finish at 192.168.1.255 (which is the broadcast address). I cannot use these addresses for my device, but only those from 192.168.1.1 to 192.168.1.254.

### The communication port

A communication port is something defined and related to layer 4, the transport layer.

Imagine you want to address a message to a host for a particular application. The receiver has to be in a listening mode for the message he wants to receive.

This means it has to open and reserve a specific socket for the connection, and that is a communication port. Usually, applications open specific ports for their own purpose, and once a port has been opened and reserved by an application, it cannot be used by another application while it is opened by the first one.

This provides a powerful system for data exchange. Indeed, if we want to send data to a host for more than one application, we can specifically address our messages to this host on a different port to reach different applications.

Of course, standards had to be defined for global communications.

TCP port 80 is used for the HTTP protocol related to data exchange with web-servers.

UDP port 53 is used for anything related to DNS.

If you are curious, you can read the following huge official text file containing all declared and reserved port and the related services: [http://www.ietf.org/assignments/service-names-port-numbers/service-names-port-numbers.txt](http://www.ietf.org/assignments/service-names-port-numbers/service-names-port-numbers.txt).

These are conventions. Someone can easily run a web server on a port other than 80\. Then, the specific clients of this web server would have to know about the port used. This is why conventions and standards are useful.

# Wiring Arduino to wired Ethernet

Ethernet is the local area network most used nowadays.

Usual Arduino boards don't provide Ethernet ability. There is one board named Arduino Ethernet that provides native Ethernet and network features. By the way, it doesn't provide any USB-native features.

You can find the reference page here: [http://arduino.cc/en/Main/ArduinoBoardEthernet](http://arduino.cc/en/Main/ArduinoBoardEthernet).

![Wiring Arduino to wired Ethernet](img/7584_11_005.jpg)

Arduino Ethernet board with the Ethernet connector

We are going to use the Arduino Ethernet Shield and a 100BASE-T cable with the Arduino UNO R3\. It keeps the USB features and adds Ethernet network connectivity and provides a nice way to link our computer to the Arduino with a much longer cable that USB ones.

![Wiring Arduino to wired Ethernet](img/7584_11_006.jpg)

The Arduino Ethernet Shield

If you look for the Arduino Ethernet module, you must know they are sold either with or without the PoE module.

**PoE** stands for **Power** **over** **Ethernet** and is a way to supply power to devices through Ethernet connections. This requires two parts:

*   A module on the device that has to be supplied
*   A network equipment able to provide PoE support

In our case here, we won't use PoE.

## Making Processing and Arduino communicate over Ethernet

Let's design a basic system showing how to set up a communication over Ethernet between the Arduino board and a processing applet.

Here, we are going to use an Arduino board wired to our computer using Ethernet. We push a button that triggers the Arduino to send a message over UDP to the Processing applet on the computer. The applet reacts by drawing something and sends back a message to the Arduino, which switches on its built-in LED.

### Basic wiring

Here, we are wiring a switch and using the built-in LED board. We have to connect our Arduino board to our computer using an Ethernet cable.

This wiring is very similar to the MonoSwitch project in [Chapter 5](ch05.html "Chapter 5. Sensing with Digital Inputs") except that we are wiring the Arduino Ethernet Shield here instead of the Arduino board itself.

![Basic wiring](img/7584_11_007.jpg)

The switch and the pull-down resistor wired to the Arduino Ethernet Shield

The corresponding circuit diagram is as follows:

![Basic wiring](img/7584_11_008.jpg)

The switch and the pull-down resistor wired to the Arduino Ethernet Shield

### Coding network connectivity implementation in Arduino

As we described, if we want to give our Arduino the ability to communicate over the Ethernet cable, and more generally over an Ethernet network, we have to implement the required standards in the firmware.

There is a library called `Ethernet` that can provide a great number of features.

As usual, we have to include this native library itself. You can choose to do that by navigating to **Sketch** **|** **Import** **Library**, which includes almost everything you need.

However, since Arduino version 0018, because of the implementation of SPI and because the Arduino Ethernet Shield communicates with the Arduino board through SPI, we have to include something more. Be careful about that.

For this code, you need:

[PRE0]

This is a example of the Arduino code, followed by an explanation.

You can find the complete Arduino code at `Chapter11/WiredEthernet`.

[PRE1]

In the previous block of code, at first we include the `Ethernet` library. Then we declare the complete set of variables related to switch debouncing and LED handling. After these statements, we define some variables related to network features.

At first, we have to set the MAC address related to our own shield. This unique identifier is usually indicated on a sticker on your Ethernet shield. Please don't forget to put yours in the code.

Then, we set up the IP address of the Arduino. We can use any address as long as it respects the IP address schema and as long as it is reachable by our computer. That means on the same network or on another network, but with a router between both. However, be careful, as the IP address you chose has to be unique on a local network segment.

We also choose a UDP port for our communication. We are using the same definition with network parameters related to our computer, the second set of participants in the communication.

We declare a buffer to store the current received messages at each time. Notice the constant `UDP_TX_PACKET_MAX_SIZE`. It is defined in the Ethernet library. Basically, it is defined as 24 bytes in order to save memory. We could change that. Then, we instantiate the `EthernetUDP` object in order to receive and send datagram over UDP. The `setup()` function block contains statements for switch and LED, then for Ethernet itself.

We begin the Ethernet communication using the MAC and IP addresses. Then we open and listen at the UDP port defined in the definition, which is 9999 in our case. The `loop()` function seems a bit thick, but we can divide it in 2 parts.

In the first part, we check if the Arduino has received a packet. If it has, it is checked by calling the `parsePacket()` function of the Ethernet library and checking if that one returns a packet size different than zero. We read the data and store it into the `packetBuffer` variable.

Then we check if this variable equals `Light` or `Dark` and act accordingly by switching on or off the LED on the Arduino board.

In the second part, we can see the same debouncing structure as we have seen in [Chapter 5](ch05.html "Chapter 5. Sensing with Digital Inputs"). At the end of this part, we check if the switch is pushed or released and depending on the state send a UDP message to the computer.

Let's check the Processing/Computer part now.

### Coding a Processing Applet communicating on Ethernet

Let's check the code at `Chapter11/WiredEthernetProcessing`.

We need the library hypermedia. We can find it at [http://ubaa.net/shared/processing/udp](http://ubaa.net/shared/processing/udp).

[PRE2]

We import the library first. Then we define the UDP object and a String variable for the current received message.

Here too, we have to define the IP address of the remote participant, the Arduino. We also define the port opened and available for the communication on the Arduino side, here it is 9999.

Of course, this has to match the one defined in the Arduino firmware. In the `setup()` function, we define some drawing parameters and then instantiate the UDP socket on the UDP port 10000 and we set it to listening mode, waiting for incoming messages.

In the `draw()` function, we draw a circle. The `receive()` function is a callback used by the code when packets are incoming. We test the length of packets in bytes because we want to react to only two different messages here (`Pushed` or `Released`), so we check if the length is 6 or 8 bytes. All other packets won't be processed. We could implement a better checking mechanism, but this one works fine.

As soon as one of these lengths match, we concatenate each byte into the String variable `currentMessage`. This provides an easy way to compare the content to any other string.

Then, we compare it to `Pushed` and `Released` and act accordingly by sending back the message `Light` to the Arduino and filling our drawn circle with white color, or by sending back the message `Dark` to the Arduino and filling our drawn circle with black color.

We just designed our first basic communication protocol using Ethernet and UDP.

## Some words about TCP

In my own design, I often use UDP for communication between systems. It is much lighter than TCP and is quite sufficient for our purposes.

In some cases, you would need to have the flow control provided by TCP. The Ethernet library we just used provides TCP features too. You can find the reference page at [http://arduino.cc/en/Reference/Ethernet](http://arduino.cc/en/Reference/Ethernet).

`Server` and `Client` classes can be used for this purpose especially, implementing function testing if a connection has been opened, if it is still valid, and so on.

We will learn how to connect our Arduino to some live server on the Internet at the end of this chapter.

# Bluetooth communications

Bluetooth is a wireless technology standard. It provides a way to exchange data over short distances using short-wavelength radio transmissions in the band 2,400 to 2,480 MHz.

It allows to create PANs (Personal Area Networks) with the "correct" level of security. It is implemented on various types of devices such as computers, smartphones, sound systems that can read digital audio from a remote source, and so on.

Arduino BT board natively implements this technology. It is now supplied with ATmega328 and a Bluegiga WT11 Bluetooth module. The reference page is `http://www.arduino.cc/en/Main/ArduinoBoardBluetooth`.

In my opinion, the best way to proceed in many projects is to keep a general purpose board at the core of our designs and to add new features by adding only what we need as external modules. Following this, we are going to use the Arduino UNO R3 here with an external Bluetooth module.

We are going to make a small project using Processing again. You click somewhere over the Processing canvas and the Processing applet sends a message over Bluetooth to the Arduino, which reacts by switching its built-in LED on or off.

## Wiring the Bluetooth module

Check the following figures:

![Wiring the Bluetooth module](img/7584_11_009.jpg)

RN41 Bluetooth module wired to the Arduino via a serial link

The corresponding circuit diagram is as follows:

![Wiring the Bluetooth module](img/7584_11_010.jpg)

Roving Networks RN41 module wired to the Arduino board

There is a Roving Networks RN41 Bluetooth module wired to the Arduino board.

You can find it at[https://www.sparkfun.com/products/10559](https://www.sparkfun.com/products/10559).

Here we are using the basic serial link communication between the Arduino itself and the Bluetooth module.

We suppose our computer has Bluetooth capabilities and that those are activated.

## Coding the firmware and the Processing applet

The firmware is as follows. You can find it in at `Chapter11/Bluetooth`.

[PRE3]

We basically instantiate the `Serial` communication with the Bluetooth module, then we check if any bytes are available from it and parse them. If a message is available and equals 1, we switch on the LED; if it equals 0, we switch off the LED.

The processing code is as follows:

[PRE4]

We first include the serial library. In the `setup()` function, we define some drawing bits, then we print the list of serial device to the Processing log area. This displays a list and we have to find the right Bluetooth module of our computer. In my case, this was the third one and I used this to instantiate the `Serial` communication in the latest statement of the `setup()` function:

[PRE5]

The `draw()` function only sets up:

*   Background color according to the variable `bgcolor`
*   Stroke color according to the variable `fgcolor`
*   Fill color according to the variable `fgcolor`

Then we draw a square.

The `mousePressed()` and `mouseReleased()` functions are two Processing callbacks respectively which are called when a mouse event occurs, when you push a button on the mouse and release it.

As soon as the mouse is pressed, we check where the cursor was when it was pressed. In my case, I defined the area inside the square.

If we press the button in the square, a visual feedback occurs in order to tell us the order has been received, but the most important thing is the `digitalWrite('1')` function of course.

We write the value 1 to the Bluetooth module.

In the same way, as soon as we release the mouse button, a "0" is written to the Bluetooth module of the computer. Of course, these messages are sent to the Arduino and the latter switches the LED on or off.

We just checked a nice example of an external module providing wireless Bluetooth communication feature to the Arduino.

As we noticed, we don't have to use a particular library for this purpose because the module itself is able to connect and send/receive data by itself only if we send serial data to it. Indeed, the communication between Arduino and the module is a basic serial one.

Let's improve our data communication over the air using Ethernet Wi-Fi.

# Playing with Wi-Fi

We previously learned how to use the Ethernet library. Then, we tested Bluetooth for short-range network communications. Now, let's test Wi-Fi for medium range communications still without any wire.

## What is Wi-Fi?

Wi-Fi is a set of communication protocols wireless driven by standards of IEEE 802.11\. These standards describe characteristics of Wireless Local Area Networks (WLANs).

Basically, multiple hosts having Wi-Fi modules can communicate using their IP stacks without wire. There are multiple networking modes used by Wi-Fi.

### Infrastructure mode

In that mode, Wi-Fi hosts can communicate between each other via an access point.

This access point and hosts have to be set up with the same **Service** **Set** **Identifier** (SSID), which is a network name used as a reference.

This mode is interesting because it provides security by the fact that each host has to pass by the access point in order to access the global network. We can configure some access lists in order to control which host can connect and which cannot.

![Infrastructure mode](img/7584_11_011.jpg)

Hosts exchanging data through an access point in infrastructure mode

### Ad hoc mode

In this mode, each host can connect to each one directly without access points. It is very useful to quickly connect two hosts in order to share documents and exchange data.

![Ad hoc mode](img/7584_11_012.jpg)

Two hosts directly connected in ad hoc mode

### Other modes

There are also two other modes. **Bridge** **mode** is a way to link multiple access points. We can imagine a work group sparse in two buildings; we could use two different access points and connect them together using a bridge mode.

There is also a trivial mode named **range-extender mode**. It is used to repeat the signal and provide a connection between two hosts, two access points or a host, and an access point when those are too far.

## The Arduino Wi-Fi shield

This shield adds the wireless networking capabilities to the Arduino board. The official shield also contains an SD card slot providing storing features too. It provides:

*   Connection via 802.11b/g networks
*   Encryption using WEP or WPA2 personal
*   FTDI connection for serial debugging of the shield itself
*   Mini-USB to update the Wi-Fi shield firmware itself

![The Arduino Wi-Fi shield](img/7584_11_013.jpg)

The Arduino Wi-Fi Shield

It is based on the HDG104 Wireless LAN 802.11b/g system in-package. A proper Atmega 32 UC3 provides the network IP stack.

A dedicated native library named **WiFi library** provides all that we need to connect our board wireless to any network. The reference is provided at [http://arduino.cc/en/Reference/WiFi](http://arduino.cc/en/Reference/WiFi).

This shield is available from many distributors as well as from the Arduino store: [http://store.arduino.cc/ww/index.php?main_page=product_info&cPath=11_5&products_id=237](http://store.arduino.cc/ww/index.php?main_page=product_info&cPath=11_5&products_id=237).

Let's try to connect our Arduino to our Wi-Fi network.

## Basic Wi-Fi connection without encryption

Here, we don't have to draw any schematic. Basically, we connect the shield to the Arduino and upload our code to it. We are going to test a basic connection without encrypting anything, at first.

The Accept Point has to provide a DHCP server; the latter will deliver an IP address to our Arduino-based system.

Let's check the example `ConnectNoEncryption` provided with the `WiFi` library.

[PRE6]

At first, we include the `WiFi` library. Then, we set the name of our network, the SSID. Please be careful to change it to your own SSID.

In the `setup()` function, we instantiate the `Serial` connection. Then, we check the presence of the shield by calling the function `WiFi.status()`.

If the latter returns the value `WL_NO_SHIELD` (which is a constant defined inside the WiFi library), that means there is no shield. In that case, an infinite loop is executed with a `while(true)` statement without the `break` keyword inside.

If it returns a value different than `WL_CONNECTED`, then we print a statement in order to inform that it is trying to connect. Then, `WiFi.begin()` tries to connect. This is a usual structure providing a way to try to connect while it isn't connected, constantly, and each 10 s considering the `delay()` function is called.

Then, if the connection occurs, the status becomes `WL_CONNECTED`, we exit from the `while` loop and continue.

There is something printed to serial too, saying the board has achieved connection status.

We also call two functions. These functions print to serial many elements related to network parameters and status. I'll let you discover each of them using the [http://arduino.cc/en/Reference/WiFi](http://arduino.cc/en/Reference/WiFi) reference quoted earlier.

After this connection, we can begin to exchange data. As you probably know, using Wi-Fi especially without security can lead to problems. Indeed, it is very easy to capture packets from an unprotected Wi-Fi network.

Let's use the `WiFi` library with more security.

## Arduino Wi-Fi connection using WEP or WPA2

If you open both code `ConnectWithWEP` and `ConnectWithWPA`, there are minor differences with the preceding example.

### Using WEP with Wi-Fi library

If we use a 40-bit WEP, we need a key containing 10 characters that must be hexadecimal. If we use 128-bit WEP, we need a key containing 26 characters, also hexadecimal. This key must be specified within the code.

We replaced the call to `WiFi.begin()`, which had only one argument, by two new arguments related to WEP encryption. This is the only difference.

For many reasons that we won't discuss here, WEP is considered too weak in terms of security, so most people and organizations have moved to the more secure WPA2 alternative.

### Using WPA2 with Wi-Fi library

Following the same schema, we need only a password here. Then, we call `WiFi.begin()` with 2 arguments: the SSID and the password.

In both cases we just checked, we only had to pass some additional arguments with `WiFi.begin()` in order to secure things a bit more.

## Arduino has a (light) web server

Here, we use the code `WifiWebServer` provided with the library.

In this example, Arduino acts as a web server after having been connected to a WEP or WPA Wi-Fi network.

[PRE7]

Let's explain the underlying concepts in these statements.

We explain only the new part of the code, not the autoconnect and encryption statements, because we did that earlier.

The `WiFiServer server(80)` statement instantiates a server on a specific port. Here, the TCP port chosen is 80, the standard HTTP server TCP port.

In the `setup()` function, we auto-connect the Arduino to the Wi-Fi network, then we start the server. Basically, it opens a socket on TCP port 80 and begins to listen on this port.

In the `loop()` function, we check if there is an incoming client to our web server embedded on the Arduino. This is done with `WiFiClient client = server.available();`

Then, we have a condition on client instance. If there is no client, we basically do nothing, and execute the loop again until we have a client.

As soon as we have one, we print this to serial in order to give feedback. Then, we check if the client is effectively connected and if there is data in the reading buffer. We then print this data if it is available and answer the client by sending the standard HTTP response header. This is done basically by printing bytes to the client instance itself.

The code includes some dynamic features and sends some values read on the board itself like the value coming from the ADC of each analog input.

We could try to connect some sensors and provide values of each of them through a webpage directly handled by the Arduino itself. I'll let you check the other part of the code. This deals with standard HTTP messages.

# Tweeting by pushing a switch

Connecting the Arduino to networks obviously brings the Internet to mind. We could try to create a small system that can send messages over the Internet. I choose to use the micro-blogging service Twitter because it provides a nice communication API.

We are going to use the same circuit that we used in the *Wiring Arduino to wired Ethernet* section except that here we are using the Arduino MEGA related to some memory constraints with a smaller board.

## An overview of APIs

**API** stands for **Application** **Programming** **Interface**. Basically, it defines ways to exchange data with the considered system. We can define APIs in our systems in order to make them communicate with others.

For instance, we could define an API in our Arduino firmware that would explain how and what to send in order to make the LED on the board switch on and off. We won't describe the whole firmware, but we would provide to the world a basic document explaining precisely the format and data to send from the Internet, for instance, to use it remotely. That would be an API.

## Twitters API

Twitter, as do many other social network-related systems on the Internet, provides an API. Other programmers can use it to get data and send data too. All data specifications related to Twitters API are available at [https://dev.twitter.com](https://dev.twitter.com).

In order to use the API, we have to create an application on Twitters developer website. There are some special security parameters to set up, and we have to agree upon some rules of use that respect data requests rate and other technical specifications.

We can create an application by going to [https://dev.twitter.com/apps/new](https://dev.twitter.com/apps/new).

That will provide us with some credential information, in particular an access token and a token secret. These are strings of characters that have to be used following some protocols to be able to access the API.

## Using the Twitter library with OAuth support

*Markku Rossi* created a very powerful and reliable library embedding the OAuth support and intended for sending tweets directly from the Arduino. The official library website is [http://www.markkurossi.com/ArduinoTwitter](http://www.markkurossi.com/ArduinoTwitter).

This library needs to be used with a board with more than the usual amount of memory. The Arduino MEGA runs it perfectly.

OAuth is an open protocol to allow secure authorization in a simple and standard method from web, mobile, and desktop applications. This is defined at [http://oauth.net](http://oauth.net).

Basically, this is a way to enable third-party application to obtain limited access to an HTTP service. By sending some specific string of characters, we can grant access to a host and make it communicate with the API.

This is what we are going to do together as a nice example that you could reuse for other APIs on the Web.

### Grabbing credentials from Twitter

Markku's library implements the OAuth request signing, but it doesn't implement the OAuth Access Token retrieval flow. We can retrieve our token by using this guide on the Twitter website where we created our application: [https://dev.twitter.com/docs/auth/tokens-devtwittercom](https://dev.twitter.com/docs/auth/tokens-devtwittercom).

You need to keep handy the Access token and Access token secret, as we are going to include them in our firmware.

### Coding a firmware connecting to Twitter

Markku's library is easy to use. Here is a possible code connecting the Arduino to your Ethernet network so that you can tweet messages directly.

You can find it at `Chapter11/tweetingButton/`.

[PRE8]

Let's explain things here. Please note, this is a code including many things we already discovered and learned together:

*   Button push with debouncing system
*   Ethernet connection with the Arduino Ethernet Shield
*   Twitter library example

We first include a lot of library headers:

*   SPI and Ethernet for network connection
*   Sha1 for credentials encryption
*   Time for time and date specific functions used by Twitter library
*   EEPROM to store credentials in EEPROM of the board
*   Twitter library itself

Then, we include the variable related to the button itself and the debouncing system.

We configure the network parameters. Please notice you have to put your own elements here, considering your network and Ethernet shield. Then, we define the IP address of Twitter.

We define the `TWEET_DELTA` constant for further use, with respect to the Twitter API use that forbids us from sending too many tweets at a time. Then, we store our credentials. Please use yours, related to the application you created on the Twitter website for our purpose. At last we create the object twitter.

In the `setup()` function, we start the `Serial` connection in order to send some feedback to us. We configure the digital pin of the switch and start the Ethernet connection. Then, we have all the wizardry about Twitter. We first choose the entry point defined by the Twitter API docs itself. We have to put our Access token and Token secret here too. Then, we have a compilation condition: `#if TOKEN_IN_MEMORY`.

`TOKEN_IN_MEMORY` is defined before as 0 or 1\. Depending on its value, the compilation occurs in one manner or another.

In order to store credentials to the EEPROM of the board, we first have to put the value 0\. We compile it and run it on the board. The firmware runs and writes the tokens in memory. Then, we change the value to 1 (because tokens are now in memory) and we compile it and run it on the board. From now, the firmware will read credentials from EEPROM.

Then, the `loop()` function is quite simple considering what we learned before.

We first test if the twitter connection to the API is okay. If it is okay, we store the time and the time of the last tweet at an initial value. We read the debounce value of the digital input.

If we push the button, we test to see if we did that in less than the `TWEET_DELTA` amount of time. If it is the case, we are safe with respect to the Twitter API rules and we can tweet.

At last, we store a message in the char array `msg`. And we tweet the message by using `twitter.post_status()` function. While using it, we also test what it returns. If it returns `1`, it means the tweet occurred. That provides this information to the user through serial monitor.

All API providers work in the same way. Here, we were very helped by the Twitter library we used, but there are other libraries also for other services on the Internet. Also, each service provides the complete documentation to use their API. Facebook API resources are available here: [https://developers.facebook.com/](https://developers.facebook.com/). Google+ API resources are available here: [https://developers.google.com/+/api/](https://developers.google.com/+/api/). Instagram API resources are available here: [http://instagram.com/developer](http://instagram.com/developer). And we could find a lot of others.

# Summary

In this chapter, we learned how to extend the area of communication of our Arduino boards. We were used to making very local connections; we are now able to connect our board to the Internet and potentially communicate with the whole planet.

We described Wired Ethernet, Wi-Fi, Bluetooth connections, and how to use Twitters API.

We could have described the Xbee board, which uses radio frequencies, too, but I preferred to describe IP-related stuff because I consider them to be the safest way to transmit data. Of course, Xbees shield solution is a very nice one too and I used it myself in many projects.

In the next chapter, we are going to describe and dig into the Max 6 framework. This is a very powerful programming tool that can generate and parse data and we are going to explain how we can use it with Arduino.