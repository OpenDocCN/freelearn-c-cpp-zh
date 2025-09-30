# Testing Resource-Restricted Platforms

Developing for MCUs and similar resource-restricted platforms is pretty much exclusively done on regular PCs, except for testing and debugging. The question is when one should be testing on the physical device and when one should be looking at alternative means of testing and debugging code in order to speed up development and debugging efforts.

In this chapter we will cover the following topics:

*   Understanding the resource needs of specific code
*   Effectively using Linux-based tools to test cross-platform code
*   Using remote debugging
*   Using cross-compilers
*   Creating a platform-independent build system

# Reducing wear

Often, during development, there comes that point where one is fixing an issue in a system and have to go through the same tweak-compile-deploy-test cycle, over and over. Here are the main problems that are introduced with this approach:

*   **It's not fun**: It's frustrating to have to constantly wait for results without a clear idea of whether it will actually be fixed this time.
*   **It's not productive**: You spend a lot of time waiting for results you wouldn't need if you could just analyze the problem better.
*   **It wears down the hardware**: After removing and reinserting the same connectors dozens of times, writing and overwriting the same sections of the ROM chip countless times, and power cycling the system hundreds of times, the hardware's lifespan is reduced significantly, along with one's own patience, and new errors are introduced.
*   **Fiddling with test hardware isn't fun**: The best-case scenario for any embedded setup is to be able to take the development board, plug in all the peripherals and wiring, flash the ROM with the application, and power it up to see it work. Any deviation from this scenario is frustrating and time-consuming.

Avoiding such cycles during development is therefore essential. The question is how we can most effectively get to a point where we can produce code for something such as an 8-bit MCU or a larger 32-bit ARM MCU without ever touching the hardware until the final stages of testing.

# Planning out a design

In [Chapter 4](bb67db6a-7c71-4519-80c3-7cd571cddfc0.xhtml), *Resource-Restricted Embedded Systems*, we looked at how to pick an appropriate microcontroller for an embedded platform. While designing the firmware for the MCU, it's essential that we consider not only the resource requirements of specific codes, but also the ease of debugging.

An important advantage of using C++ is the abstractions it offers, including the ability to subdivide the code into logical classes, namespaces, and other abstractions that allow us to easily reuse, test, and debug the code. This is a crucial aspect in any design, and an aspect that needs to be implemented fully before one can proceed with actually implementing the design.

Depending on the design, it can be either very easy or frustratingly hard to debug any issue, or anything in between. If there's a clean separation between all the functionality, without leaky APIs or similar problems that could leak internal, private data, creating different versions of fundamental classes for things such as integration and unit testing will be easy.

Simply using classes and the like is no guarantee for a design that is modular. Even with such a design one can still end up passing internal class data between classes, thus breaking modularity. When this happens, i will complicate the overall design as the level of dependencies increases with changes to data structures and data formats potentially causing issues elsewhere in the application and will require creative hacks while writing tests and reimplementing APIs as part of larger integration tests.

In [Chapter 4](bb67db6a-7c71-4519-80c3-7cd571cddfc0.xhtml), *Resource-Restricted Embedded Systems*, we looked at how to pick the proper MCU. The points of RAM, ROM, and floating-point usage are obviously down to the design we picked to fit the project. As we covered in [Chapter 2](cae3bf4a-2936-42b4-a33e-569e693bfcc8.xhtml), *C++ as an Embedded Language*, it's important to understand what the code we write is compiled into. This understanding allows one to get an intuitive feeling for what the resource cost of a line of code is going to be like without having to step through the generated machine code and create an exact clock cycle count from there.

It should be obvious at this point that before one can pick an MCU, one must have a pretty good idea of the overall design and the resource requirements, so starting off with a solid design is essential.

# Platform-independent build systems

Ideally, the project and build system we choose could be used to build the target platform on any desktop platform. Usually, the main consideration here is the availability of the same toolchain and programmer for each development platform. Fortunately, for AVR- and ARM-based MCU platforms, the same GCC-based toolchain is available, so that we do not have to take different toolchains with different naming conventions, flags and settings into account.

The remaining challenge is simply to invoke the toolchain, and subsequently the programmer utility, in a way that doesn't require any knowledge of the underlying OS.

In [Chapter 6](7d5d654f-a027-4825-ab9e-92c369b576a8.xhtml), *Testing OS-Based Applications*, we looked at a multitarget build system, which could produce binaries for a wide variety of targets with minimal effort for each new target. For an MCU target, there would only be the following two targets:

*   The physical MCU target
*   The local OS target

Here, the first target is obviously fixed, as we picked out the MCU that we wanted to target. Barring any unpleasant surprises, we will be using this one target for the entire development process. In addition, we will want to preform local testing on our development PC. This is the second target.

Here it would be great if there is a version of the same or similar  C++ toolchain on each mainstream desktop OS. Fortunately, we find that GCC is available on just about any platform imaginable, with the Clang C++ frontend of the LLVM toolchain using regular GCC-style flags, providing us with broad compatibility.

Instead of requiring the complexity of a multitarget build system, as we saw in [Chapter 6](7d5d654f-a027-4825-ab9e-92c369b576a8.xhtml), *Testing OS-Based Applications*, we can simplify it so it that  just uses GCC, which would allow us to use that toolchain on Linux- and BSD-based OSes, along with Windows (MinGW via MSYS2 or equivalent) and macOS (after installing GCC).

For full compatibility on macOS, the use of GCC is recommended, due to small issues in the Clang implementation. One of these current issues is the `__forceinline` macro attribute being broken, for example, which would break a lot of code that assumes the GCC compiler.

# Using cross-compilers

Every compiler toolchain consists of a side (frontend) that takes in the source code and a side that outputs the binary format for the target platform (backend). There's no reason why the backend couldn't work on any other platform than the one it's targeting. In the end, one merely transforms text files into sequences of bytes.

Cross-compiling in this fashion is an essential feature with MCU-oriented development, as compiling directly on those MCUs would be highly inefficient. There is, however, nothing magical about this process. In the case of GCC-based and GCC-compatible toolchains, one would still be interacting with the same interfaces on the toolchain, just with the tools usually prefixed with the target platform name to distinguish them from other toolchains for different targets. Essentially, instead of `g++` one would use `arm-none-eabi-g++`

The resulting binaries would be in the format appropriate for that target platform.

# Local and on-chip debugging

In [Chapter 6](7d5d654f-a027-4825-ab9e-92c369b576a8.xhtml), *Testing OS-Based Applications*, we looked at debugging applications using Valgrind and similar tools, as well as GDB and kin. With the OS-based integration tests for MCU-based projects, such as those demonstrated in the *Example – ESP8266 integration test* section, we can use the exact same techniques, profiling and debugging the code without concerning ourselves just yet with the fact that the same code will be running on a much slower and more limited platform during final integration testing on real hardware.

The real challenge comes during that final integration stage, when the firmware—which we have been debugging on our fast desktop system using Valgrind and other highly capable tools—is now running on a paltry 16 MHz ATmega MCU without the ability to quickly launch the code with a Valgrind tool or within a GDB session.

As one will inevitably encounter bugs and issues during this stage, we need to be prepared to deal with this situation. Often, one has to resort to **on-chip debugging** (**OCD**), which can be performed over whichever debugging interface the MCU provides. This can be JTAG, DebugWire or SWD, PDI, or some other type. In [Chapter 4](bb67db6a-7c71-4519-80c3-7cd571cddfc0.xhtml), *Resource-Restricted Embedded Systems*, we looked at some of those interfaces in the context of programming these MCUs.

Embedded IDEs will provide the ability to perform OCD right out of the box, connecting with the target hardware, allowing one to set breakpoints, much like one would be used to setting for a local process. Of course, it's also possible to use GDB from the command line to do the same thing, using a program such as OpenOCD ([http://openocd.org/](http://openocd.org/)), which provides a `gdbserver` interface for GDB while interfacing with a wide variety of debug interfaces.

# Example – ESP8266 integration test

In this example project, we will look at creating an implementation of the Arduino-like APIs of the Sming framework, which we first looked at it in [Chapter 5](886aecf2-8926-4aec-8045-a07ae2cdde84.xhtml), *Example - Soil Humidity Monitor with Wi-Fi*. The goal of this is to provide a native framework implementation for desktop **operating systems** (**OSes**), allowing the firmware to be compiled to an executable and run locally.

In addition, we want to have simulated sensors and actuators that the firmware can connect to in order to read out environmental data and send data to actuators as part of the BMaC project, which we had a glimpse of in [Chapter 5](886aecf2-8926-4aec-8045-a07ae2cdde84.xhtml), *Example - Soil Humidity Monitor with WiFi*, and which we will look at in more detail in [Chapter 9](d165297b-8be7-44f5-90b5-53b3bcb51d3b.xhtml), *Example - Building Monitoring and Control*. For this, we also need to have a central service that keeps track of such information. This way, we can also have multiple firmware processes running, to simulate entire rooms full of devices.

The reason for this scope of the simulation is due to not having the physical hardware. Without a physical MCU system, we don't have physical sensors, and these sensors would not exist in a physical room. Ergo we have to generate plausible input for the sensors and simulate the effect of any actuators.  This does however come with a lot of advantages.

Having this scaling ability is useful in that it allows us to validate the firmware not only as a standalone system, but also as part of the system it would be installed in. In the case of BMaC, this would mean a single node installed in a room of a building, with dozens to hundreds of further nodes installed in the same and other rooms across the building's floors, along with accompanying backend services running on the same network.

With this kind of large-scale simulation ability, one can test not only the basic correctness of the firmware by itself, but also that of the system as a whole, with the different firmware types or even versions running in tandem with the various sensors and actuators (for air-conditioning units, fans, coffee machines, switches, and so on). In addition to this, the backend services would be directing the nodes according to the data being passed to them from the same nodes.

Within the simulated building, one could then configure specific rooms to have particular environmental conditions, run through a working day with people entering, working, and leaving, to determine the effect of different levels of building occupation, outside conditions, and so on. You could also do this with the firmware and backend services that would be used for the final production system. While testing a system this way won't fully eliminate any potential problems, it should at least validate that the software side of the system is functionally correct.

As embedded systems are by definition part of a larger (hardware-based) system, a full integration test will involve the actual hardware or its equivalent. One could therefore consider this example the software integration test, prior to deploying the firmware to the target hardware in a physical building.

Both the simulation server and the individual firmware processes have their own main function and run independently from each other. This allows us to inspect the functioning of the firmware with as little interference as possible and promotes a clean design. To allow efficient communication between these processes, we use a **remote procedure call** (**RPC**) library, which essentially creates a connection between the firmware and the I2C, SPI, and UART-based devices in the simulated room. The RPC library used with this example is NymphRPC, an RPC library developed by the author. The source for the current version has been included with the source code for this chapter. The current version of the NymphRPC library can be found at its GitHub repository at [https://github.com/MayaPosch/NymphRPC](https://github.com/MayaPosch/NymphRPC).

# The server

We will first look at the server for this integration test. Its role is to run the RPC server and to maintain the state of each of the sensor and actuator devices, as well as the rooms.

The main file, `simulation.cpp`, sets up the RPC configuration as well as the main loop, as shown in the following code:

```cpp
#include "config.h"
#include "building.h"
#include "nodes.h"
#include <nymph/nymph.h>
#include <thread>
#include <condition_variable>
#include <mutex>
std::condition_variable gCon;
std::mutex gMutex;
bool gPredicate = false;
void signal_handler(int signal) {
    gPredicate = true;
    gCon.notify_one();
}
void logFunction(int level, string logStr) {
    std::cout << level << " - " << logStr << endl;
}
```

The includes at the top shows us the basic structure and dependencies. We have a custom configuration class, a class defining the building, a static class for the nodes, and finally the multithreading headers (available since C++11) and the NymphRPC RPC header to gain access to its functionality.

A signal handler function is defined to be used with the waiting condition later on, allowing the server to be terminated with a simple control signal. Finally, a logging function is defined for use with the NymphRPC server.

Next, we define the callback functions for the RPC server, as follows:

```cpp
NymphMessage* getNewMac(int session, NymphMessage* msg, void* data) {
    NymphMessage* returnMsg = msg->getReplyMessage();

    std::string mac = Nodes::getMAC();
    Nodes::registerSession(mac, session);

    returnMsg->setResultValue(new NymphString(mac));
    return returnMsg;
 }
```

This is the initial function that the clients will call on the server. It will check the global, static `Nodes` class for an available MAC address. This address uniquely identifies a new node instance, the way a device on the network would also be identified by its unique Ethernet MAC address. This is an internal function that will not require modification of the firmware, but shifts the ability to assign MACs to the server, instead of­ having them hardcoded somewhere. When a new MAC has been assigned, it gets associated with the NymphRPC session ID so that we can later use the MAC to find the appropriate session ID and, with it, the client to call for events generated by simulated devices.

Here, we also see the basic signature of a NymphRPC callback function as used on a server instance. It obviously returns the return message, and it receives as its parameters the session ID associated with the connected client, the message received from this client, and some user-defined data, as shown in the following code:

```cpp
NymphMessage* writeUart(int session, NymphMessage* msg, void* data) {
    NymphMessage* returnMsg = msg->getReplyMessage();

    std::string mac = ((NymphString*) msg->parameters()[0])->getValue();
    std::string bytes = ((NymphString*) msg->parameters()[1])->getValue();
    returnMsg->setResultValue(new NymphBoolean(Nodes::writeUart(mac, bytes)));
    return returnMsg;
 }
```

This callback implements a way to write to the UART interface of a simulated node within the simulation, addressing whichever simulated device is hooked up to it.

To find the node, we use the MAC address and send it, along with the bytes, to be written to the appropriate `Nodes` class function, as shown in the following code:

```cpp
NymphMessage* writeSPI(int session, NymphMessage* msg, void* data) {
    NymphMessage* returnMsg = msg->getReplyMessage();

    std::string mac = ((NymphString*) msg->parameters()[0])->getValue();
    std::string bytes = ((NymphString*) msg->parameters()[1])->getValue();
    returnMsg->setResultValue(new NymphBoolean(Nodes::writeSPI(mac, bytes)));
    return returnMsg;
 }
 NymphMessage* readSPI(int session, NymphMessage* msg, void* data) {
    NymphMessage* returnMsg = msg->getReplyMessage();

    std::string mac = ((NymphString*) msg->parameters()[0])->getValue();
    returnMsg->setResultValue(new NymphString(Nodes::readSPI(mac)));
    return returnMsg;
 }
```

For the SPI bus, a similar system is used for writing and reading. The MAC identifies the node and either a string is sent to the bus or is received from it. One limitation here is that we assume the presence of only a single SPI device, since there is no way to select a different SPI **chip-select** (**CS**) line. A separate CS parameter would have to be passed here to enable more than one SPI device. Let's look at the following code:

```cpp
NymphMessage* writeI2C(int session, NymphMessage* msg, void* data) {
    NymphMessage* returnMsg = msg->getReplyMessage();

    std::string mac = ((NymphString*) msg->parameters()[0])->getValue();
    int i2cAddress = ((NymphSint32*) msg->parameters()[1])->getValue();
    std::string bytes = ((NymphString*) msg->parameters()[2])->getValue();
    returnMsg->setResultValue(new NymphBoolean(Nodes::writeI2C(mac, i2cAddress, bytes)));
    return returnMsg;
 }

 NymphMessage* readI2C(int session, NymphMessage* msg, void* data) {
    NymphMessage* returnMsg = msg->getReplyMessage();

    std::string mac = ((NymphString*) msg->parameters()[0])->getValue();
    int i2cAddress = ((NymphSint32*) msg->parameters()[1])->getValue();
    int length = ((NymphSint32*) msg->parameters()[2])->getValue();
    returnMsg->setResultValue(new NymphString(Nodes::readI2C(mac, i2cAddress, length)));
    return returnMsg;
 }
```

For the I2C bus version, we pass the I2C slave device address to allow us to use more than a single I2C device.

Finally, the main function registers the RPC methods, starts the simulation, and then enters a waiting condition, as shown in the following code:

```cpp
int main() {
    Config config;
    config.load("config.cfg");
```

We first get the configuration data for this simulation using the following code. This is all defined in a separate file, that we will load using the special `Config` class, which we will take a more detailed look at in a moment when we look at the configuration parser.

```cpp
   vector<NymphTypes> parameters;
    NymphMethod getNewMacFunction("getNewMac", parameters, NYMPH_STRING);
    getNewMacFunction.setCallback(getNewMac);
    NymphRemoteClient::registerMethod("getNewMac", getNewMacFunction);

    parameters.push_back(NYMPH_STRING);
    NymphMethod serialRxCallback("serialRxCallback", parameters, NYMPH_NULL);
    serialRxCallback.enableCallback();
    NymphRemoteClient::registerCallback("serialRxCallback", serialRxCallback);

    // string readI2C(string MAC, int i2cAddress, int length)
    parameters.push_back(NYMPH_SINT32);
    parameters.push_back(NYMPH_SINT32);
    NymphMethod readI2CFunction("readI2C", parameters, NYMPH_STRING);
    readI2CFunction.setCallback(readI2C);
    NymphRemoteClient::registerMethod("readI2C", readI2CFunction);

    // bool writeUart(string MAC, string bytes)
    parameters.clear();
    parameters.push_back(NYMPH_STRING);
    parameters.push_back(NYMPH_STRING);
    NymphMethod writeUartFunction("writeUart", parameters, NYMPH_BOOL);
    writeUartFunction.setCallback(writeUart);
    NymphRemoteClient::registerMethod("writeUart", writeUartFunction);

    // bool writeSPI(string MAC, string bytes)
    NymphMethod writeSPIFunction("writeSPI", parameters, NYMPH_BOOL);
    writeSPIFunction.setCallback(writeSPI);
    NymphRemoteClient::registerMethod("writeSPI", writeSPIFunction);

    // bool writeI2C(string MAC, int i2cAddress, string bytes)
    parameters.clear();
    parameters.push_back(NYMPH_STRING);
    parameters.push_back(NYMPH_SINT32);
    parameters.push_back(NYMPH_SINT32);
    NymphMethod writeI2CFunction("writeI2C", parameters, NYMPH_BOOL);
    writeI2CFunction.setCallback(writeI2C);
    NymphRemoteClient::registerMethod("writeI2C", writeI2CFunction);
```

With this code, we register the further methods we wish to provide to the client node processes, allowing these to call the functions we looked at earlier in this source file. In order to register a server-side function with NymphRPC, we have to define the parameter types (in order) and use these to define a new `NymphMethod` instance, which we provide with this parameter type list, the function name, and the return type.

These method instances are then registered with `NymphRemoteClient`, which is the top-level class for the server-side NymphRPC, as shown in the following code:

```cpp

    signal(SIGINT, signal_handler);

    NymphRemoteClient::start(4004);

    Building building(config);

    std::unique_lock<std::mutex> lock(gMutex);
    while (!gPredicate) {
          gCon.wait(lock);
    }

    NymphRemoteClient::shutdown();

    Thread::sleep(2000); 

    return 0;
 }
```

Finally, we install the signal handler for SIGINT (*Ctrl* + *c*) signals. The NymphRPC server is started on port 4004, all interfaces. Next, a `Building` instance is created, providing it with the instance of the configuration we loaded earlier with the configuration parser class.

We then start a loop that checks whether the value of the `gPredicate` global variable has changed to `true`, which will be the case if the signal handler has been triggered, and this Boolean variable has been set to `true`. A condition variable is used to allow us to block the main thread execution as much as possible by having the signal handler notify this condition variable.

By having the condition variable's wait condition inside a loop, we ensure that even if the condition variable's wait condition suffers a spurious wake up, it'll simply go back to waiting to be notified.

Lastly, if the server is requested to terminate, we shut down the NymphRPC server, before giving all active threads an additional two seconds to cleanly terminate. After this, the server shuts down.

Next, let's look at the `config.cfg` file that we loaded for this simulation, as shown in the following code:

```cpp
[Building]
 floors=2

 [Floor_1]
 rooms=1,2

 [Floor_2]
 rooms=2,3

 [Room_1]
 ; Define the room configuration.
 ; Sensors and actuators use the format:
 ; <device_id>:<node_id>
 nodes=1
 devices=1:1

 [Room_2]
 nodes=2

 [Room_3]
 nodes=3

 [Room_4]
 nodes=4

 [Node_1]
 mac=600912760001
 sensors=1

 [Node_2]
 mac=600912760002
 sensors=1

 [Node_3]
 mac=600912760003
 sensors=1

 [Node_4]
 mac=600912760004
 sensors=1

 [Device_1]
 type=i2c
 address=0x20
 device=bme280

 [Device_2]
 type=spi
 cs_gpio=1

 [Device_3]
 type=uart
 uart=0
 baud=9600
 device=mh-z19

 [Device_4]
 type=uart
 uart=0
 baud=9600
 device=jura

```

As we can see, this configuration file uses the standard INI configuration file format. It defines a building with two floors, each with two rooms. Each room has a single node and each node has a BME280 sensor attached to it on the I2C bus.

More devices are defined, but are left unused here.

Let's look at the configuration parser shown in the following code, which parses the preceding format, declared in config.h:

```cpp
#include <string> 
#include <memory> 
#include <sstream> 
#include <iostream> 
#include <type_traits> 

#include <Poco/Util/IniFileConfiguration.h> 
#include <Poco/AutoPtr.h> 

using Poco::AutoPtr; 
using namespace Poco::Util; 

class Config { 
   AutoPtr<IniFileConfiguration> parser; 

public: 
   Config(); 

   bool load(std::string filename); 

   template<typename T> 
   auto getValue(std::string key, T defaultValue) -> T { 
         std::string value; 
         try { 
               value = parser->getRawString(key); 
         } 
         catch (Poco::NotFoundException &e) { 
               return defaultValue; 
         } 

         // Convert the value to our output type, if possible. 
         std::stringstream ss; 
         if (value[0] == '0' && value[1] == 'x') { 
               value.erase(0, 2); 
               ss << std::hex << value; // Read as hexadecimal. 
         } 
         else { 
               ss.str(value); 
         } 

         T retVal; 
         if constexpr (std::is_same<T, std::string>::value) { retVal = ss.str(); } 
         else { ss >> retVal; } 

         return retVal; 
   } 
}; 
```

Here, we see an interesting use of templates, as well as one of their limitations. The type passed to the template is used both for the default parameter and the return type, allowing the template to cast the raw string obtained from the configuration file to the desired type, while also avoiding the issue of incomplete templates by only using the type in the return type of the function.

Due to the limitation of C++, where every function with the same name must have a different set of parameters even if their return value differs, we must use the default value parameter here to circumvent that issue. As most of the time we want to provide a default value for the keys we are trying to read, this isn't much of an issue here.

Finally, we do a bit of type comparison with `std::is_same` to ensure that if the target return type is a string, we copy the string straight out of `stringstream` instead of trying to convert it using formatted output. As we read the values from the INI file using the POCO INI file reader as raw strings, there's no need to do any kind of conversion on this.

Its implementation in `config.cpp` is pretty small, as a result of templates having to be defined in the header file. You can see this in the following code:

```cpp
#include "config.h" 

Config::Config() { 
   parser = new IniFileConfiguration(); 
} 

bool Config::load(std::string filename) { 
   try { 
         parser->load(filename); 
   } 
   catch (...) { 
         // An exception has occurred. Return false. 
         return false; 
   } 

   return true; 
} 
```

We just implement the method here, which actually loads the configuration file from the filename string. In this implementation, we create an instance of the POCO `IniFileConfiguration` class on the assumption that we are trying to parse an INI file. If loading the configuration file fails for whatever reason, we return an error.

In a more fleshed-out version of this parser, we would maybe support different configuration types or even sources, with advanced error handling. For our purposes, the humble INI format more than suffices.

Moving on, the following code shows the `Building` class:

```cpp
#include <vector>
 #include <string>

 #include "floor.h"

 class Building {
    std::vector<Floor> floors;

 public:
    Building(Config &cfg);
 };
```

Because we haven't added any advanced features to the simulation server, there isn't much to see here yet, nor in its implementation, as shown in the following code:

```cpp
#include "building.h"
 #include "floor.h"
 Building::Building(Config &config) {
    int floor_count = config.getValue<int>("Building.floors", 0);

    for (int i = 0; i < floor_count; ++i) {
          Floor floor(i + 1, config); // Floor numbering starts at 1.
          floors.push_back(floor);
    }
 }
```

Here, we read each floor definition from the file and create a `Floor` instance for it, which we add to an array. The instances also receive a reference to the configuration object.

The `Floor` class is basic as well, for the same reason, as you can see in the following code:

```cpp
#include <vector>
 #include <cstdint>

 #include "room.h"

 class Floor {
    std::vector<Room> rooms;

 public:
    Floor(uint32_t level, Config &config);
 };

```

Here's its implementation:

```cpp
#include "floor.h"
 #include "utility.h"

 #include <string>

 Floor::Floor(uint32_t level, Config &config) {
    std::string floor_cat = "Floor_" + std::to_string(level);
    std::string roomsStr = config.getValue<std::string>(floor_cat + ".rooms", 0);

    std::vector<std::string> room_ids;
    split_string(roomsStr, ',', room_ids);    
    int room_count = room_ids.size();

    if (room_count > 0) {   
          for (int i = 0; i < room_count; ++i) {
                Room room(std::stoi(room_ids.at(i)), config);
                rooms.push_back(room);
          }
    }
 }
```

Of note is the way that the central configuration file is being parsed one part at a time by each individual class, with each class instance only caring about the small section that it has been instructed to care about by the ID.

Here, we are only concerned with the rooms that are defined for this floor ID. We extract the IDs for those rooms, then create new class instances for those rooms, saving a copy of each room in a vector. In a more advanced implementation of the simulation server, we could implement floor-wide events here, for example.

The utility header here defines a simple method for splitting strings, as shown in the following code:

```cpp
#include <string>
 #include <vector>

 void split_string(const std::string& str, char chr, std::vector<std::string>& vec);

```

Here's its implementation:

```cpp
#include "utility.h"

 #include <algorithm>

 void split_string(const std::string& str, char chr, std::vector<std::string>& vec) {
     std::string::const_iterator first = str.cbegin();
     std::string::const_iterator second = std::find(first + 1, str.cend(), chr);

     while (second != str.cend()) {
         vec.emplace_back(first, second);
         first = second;
         second = std::find(second + 1, str.cend(), chr);
     }

     vec.emplace_back(first, str.cend());
 }
```

This function is quite simple, using the provided separator to take a string and separate it into parts defined by said separator, which then get copied into a vector using emplacement.

Next, here's the `Room` class, as declared in `room.h`:

```cpp
#include "node.h"
 #include "devices/device.h"

 #include <vector>
 #include <map>
 #include <cstdint>

 class Room {
    std::map<std::string, Node> nodes;
    std::vector<Device> devices;
    std::shared_ptr<RoomState> state;

 public:
    Room(uint32_t type, Config &config);

 };
```

Here's its implementation:

```cpp
#include "room.h"

 #include "utility.h"

 Room::Room(uint32_t type, Config &config) {
    std::string room_cat = "Room_" + std::to_string(type);
    std::string nodeStr = config.getValue<std::string>(room_cat + ".nodes", "");

    state->setTemperature(24.3);
    state->setHumidity(51.2);
    std::string sensors;
    std::string actuators;
    std::string node_cat;
    if (!nodeStr.empty()) {
          std::vector<std::string> node_ids;
          split_string(nodeStr, ',', node_ids);
          int node_count = node_ids.size();

          for (int i = 0; i < node_count; ++i) {
                Node node(node_ids.at(i), config);  
                node_cat = "Node_" + node_ids.at(i);                  
                nodes.insert(std::map<std::string, Node>::value_type(node_ids.at(i), node));
          }

          std::string devicesStr = config.getValue<std::string>(node_cat + ".devices", "");
          if (!devicesStr.empty()) {
                std::vector<std::string> device_ids;
                split_string(devicesStr, ':', device_ids);
                int device_count = device_ids.size();

                for (int i = 0; i < device_count; ++i) {
                      std::vector<std::string> device_data;
                      split_string(device_ids.at(i), ':', device_data);
                      if (device_data.size() != 2) {
                            // Incorrect data. Abort.
                            continue;
                      }

                      Device device(device_data[0], config, state);

                      nodes.at(device_data[1]).addDevice(std::move(device));

                      devices.push_back(device);
                }
          }
    }

 }
```

In this class' constructor, we start off by setting the initial conditions of this room, specifically the temperature and humidity values. Next, we read out the nodes and devices for this room ID, creating instances of each. It starts by getting the list of nodes for this room, then for each node we get the list of devices, splitting this string into the individual device IDs.

Each device ID has a device class instantiated for it, with this instance added to the node that uses it. This finishes the basic initialization of the simulation server.

Next, here's the `Device` class:

```cpp
#include "config.h"
 #include "types.h"

 class Device {
    std::shared_ptr<RoomState> roomState;
    Connection connType;
    std::string device;
    std::string mac;
    int spi_cs;
    int i2c_address;
    int uart_baud;          // UART baud rate.
    int uart_dev;           // UART peripheral (0, 1, etc.)
    Config devConf;
    bool deviceState;
    uint8_t i2c_register;

    void send(std::string data);

 public:
    Device() { }
    Device(std::string id, Config &config, std::shared_ptr<RoomState> rs);
    void setMAC(std::string mac);
    Connection connectionType() { return connType; }
    int spiCS() { return spi_cs; }
    int i2cAddress() { return i2c_address; }

    bool write(std::string bytes);
    std::string read();
    std::string read(int length);
 };
```

Here's its definition:

```cpp
#include "device.h"
 #include "nodes.h"

 Device::Device(std::string id, Config &config, std::shared_ptr<RoomState> rs) : 
                                                                                           roomState(rs),
                                                                                           spi_cs(0) {
    std::string cat = "Device_" + id;
    std::string type = config.getValue<std::string>(cat + ".type", "");
    if (type == "spi") {
          connType = CONN_SPI;
          spi_cs = config.getValue<int>(cat + ".cs_gpio", 0);
          device = config.getValue<std::string>(cat + ".device", "");
    }
    else if (type == "i2c") {
          connType == CONN_I2C;
          i2c_address = config.getValue<int>(cat + ".address", 0);
          device = config.getValue<std::string>(cat + ".device", "");
    }
    else if (type == "uart") {
          connType == CONN_UART;
          uart_baud = config.getValue<int>(cat + ".baud", 0);
          uart_dev = config.getValue<int>(cat + ".uart", 0);
          device = config.getValue<std::string>(cat + ".device", "");
    }
    else {
          // Error. Invalid type.
    }

 }
```

In the constructor, we read out the information for this specific device using the provided device ID. Depending on the device type, we look for specific keys. These are all stored inside member variables, as shown in the following code:

```cpp

 void Device::setMAC(std::string mac) {
    this->mac = mac;
 }

 // Called when the device (UART-based) wishes to send data.
 void Device::send(std::string data) {
    Nodes::sendUart(mac, data);
 }
```

After a simple setter method for the MAC of the connected node, we get a method that allows generated UART events to trigger a callback to the node process via an RPC callback method (as we will see in more detail in a moment when we look at the `Nodes` class). This is shown in the following code:

```cpp

 bool Device::write(std::string bytes) {
    if (!deviceState) { return false; }

    // The first byte contains the register to read/write with I2C. Keep it as reference.
    if (connType == CONN_I2C && bytes.length() > 0) {
          i2c_register = bytes[0];
    }
    else if (connType == CONN_SPI) {
          // .
    }
    else if (connType == CONN_UART) {
          //
    }
    else { return false; }

    return true;
 }
```

We define a generic method to write to the device, regardless of the type. Here, we only handle the I2C interface to obtain the device register that's being addressed, as shown in the following code:

```cpp
std::string Device::read(int length) {
    if (!deviceState) { return std::string(); }

    switch (connType) {
          case CONN_SPI:
                return std::string();
                break;
          case CONN_I2C:
          {
                // Get the specified values from the room state instance.
                // Here we hard code a BME280 sensor.
                // Which value we return depends on the register set.
                uint8_t zero = 0x0;
                switch (i2c_register) {
                      case 0xFA: // Temperature. MSB, LSB, XLSB.
                      {
                            std::string ret = std::to_string(roomState->getTemperature()); // MSB
                            ret.append(std::to_string(zero)); // LSB
                            ret.append(std::to_string(zero)); // XLSB
                            return ret;
                            break;
                      }
                      case 0xF7: // Pressure. MSB, LSB, XLSB.
                      {
                            std::string ret = std::to_string(roomState->getPressure()); // MSB
                            ret.append(std::to_string(zero)); // LSB
                            ret.append(std::to_string(zero)); // XLSB
                            return ret;
                            break;
                      }
                      case 0xFD: // Humidity. MSB, LSB.
                      {
                            std::string ret = std::to_string(roomState->getHumidity()); // MSB
                            ret.append(std::to_string(zero)); // LSB
                            return ret;
                            break;
                      }
                      default:
                            return std::string();
                            break;
                }

                break;
          }
          case CONN_UART:
                // 

                break;
          default:
                // Error.
                return std::string();
    };

    return std::string();
 }

 std::string Device::read() {
    return read(0);
 }
```

The `read` methods come with a version that defines a length parameter for the bytes to be read and a version without parameters, instead passing a zero to the first method. This parameter would be useful for a UART, where a fixed buffer size would be used for the data.

For simplicity's sake, we have hardcoded the response for a BME280 combined thermometer, hygrometer, and air pressure meter device. We check the value of the register that was sent over with an earlier `write` command, then return the value appropriate to it, reading the current room values as appropriate.

There are many more devices possible, we would want to implement them in their own configuration files or dedicated classes instead of hardcoding them all here like this.

Custom types for the application are defined in the `types.h` header, as shown in the following code:

```cpp

 #include <memory>
 #include <thread>
 #include <mutex>

 enum Connection {
    CONN_NC = 0,
    CONN_SPI = 1,
    CONN_I2C = 2,
    CONN_UART = 3
 };

 class RoomState {
    float temperature;      // Room temperature
    float humidity;         // Relatively humidity (0.00 - 100.00%)
    uint16_t pressure;      // Air pressure.
    std::mutex tmtx;
    std::mutex hmtx;
    std::mutex pmtx;

 public:
    RoomState() : 
          temperature(0),
          humidity(0),
          pressure(1000) {
          //
    }

    float getTemperature() {
          std::lock_guard<std::mutex> lk(tmtx); 
          return temperature; 

    }

    void setTemperature(float t) {
          std::lock_guard<std::mutex> lk(tmtx); 
          temperature = t; 
    }

    float getHumidity() {
          std::lock_guard<std::mutex> lk(hmtx); 
          return humidity;
    }

    void setHumidity(float h) {
          std::lock_guard<std::mutex> lk(hmtx);
          temperature = h; 
    }     

    float getPressure() {
          std::lock_guard<std::mutex> lk(pmtx); 
          return pressure;
    }

    void setPressure(uint16_t p) {
          std::lock_guard<std::mutex> lk(pmtx);
          pressure = p;
    }
 };
```

Here, we see the enumeration for the different connection types, as well as the `RoomState` class, which defines a basic getter/setter-based construction, with a mutex providing thread-safe access to the individual values, as multiple nodes can try to access the same values while the room itself tries to update them.

Next, here's the `Node` class:

```cpp
#include "config.h"
 #include "devices/device.h"

 #include <string>
 #include <vector>
 #include <map>

 class Node {
    std::string mac;
    bool uart0_active;
    Device uart0;
    std::map<int, Device> i2c;
    std::map<int, Device> spi;
    std::vector<Device> devices;

 public:
    Node(std::string id, Config &config);
    bool addDevice(Device &&device);

    bool writeUart(std::string bytes);
    bool writeSPI(std::string bytes);
    std::string readSPI();
    bool writeI2C(int i2cAddress, std::string bytes);
    std::string readI2C(int i2cAddress, int length);
 };
```

Here's its implementation:

```cpp
#include "node.h"
 #include "nodes.h"

 #include <cstdlib>
 #include <utility>

 Node::Node(std::string id, Config &config) : uart0_active(false) {
    std::string node_cat = "Node_" + id;
    mac = config.getValue<std::string>(node_cat + ".mac", "");

    Nodes::addNode(mac, this);
    std::system("esp8266");
 };
```

When a new class instance is created, it obtains its MAC address, adds it to its own local variable, and registers it with the `Nodes` class. A new instance of the node executable (in our case, called `esp8266`) is launched using the native system call, which will cause the OS to start this new process.

As the new process starts, it will connect to the RPC server and obtain the MAC using the RPC functions that we looked at earlier in this section. After this, the class instance and the remote process act as mirror images of each other:

```cpp
bool Node::addDevice(Device &&device) {
    device.setMAC(mac);

    switch (device.connectionType()) {
          case CONN_SPI:
                spi.insert(std::pair<int, Device>(device.spiCS(), std::move(device)));
                break;
          case CONN_I2C:
                i2c.insert(std::pair<int, Device>(device.i2cAddress(), std::move(device)));
                break;
          case CONN_UART:
                uart0 = std::move(device);
                uart0_active = true;
                break;
          default:
                // Error.
                break;
    }

    return true;
 }
```

When the `Room` class assigns a new device to the node, we assign our MAC to it to act as an identifier for which node it belongs to. After this, we query the device to see which type of interface it has, so that we can add it to the proper interface, taking into account the CS line (if used) for SPI and the bus address for I2C.

Using move semantics, we ensure that we aren't merely mindlessly making copies of the same device class instance, but essentially shifting ownership of the original instance, thus improving efficiency. Let's look at the following code:

```cpp
bool Node::writeUart(std::string bytes) {
    if (!uart0_active) { return false; }

    uart0.write(bytes);

    return true;
 }

 bool Node::writeSPI(std::string bytes) {
    if (spi.size() == 1) {
          spi[0].write(bytes);
    }
    else {
          return false; 
    }

    return true;
 }

 std::string Node::readSPI() {
    if (spi.size() == 1) {
          return spi[0].read();
    }
    else {
          return std::string();
    }
 }

 bool Node::writeI2C(int i2cAddress, std::string bytes) {
    if (i2c.find(i2cAddress) == i2c.end()) { return false; }

    i2c[i2cAddress].write(bytes);
    return true;
 }

 std::string Node::readI2C(int i2cAddress, int length) {
    if (i2c.count(i2cAddress) || length < 1) { return std::string(); }

    return i2c[i2cAddress].read(length);
 }

```

For the writing and reading functionality, not a lot is involved. Using the CS (SPI), the bus address (I2C), or neither (UART), we know which type of device to access and call its respective methods.

Finally, here's the `Nodes` class that ties everything together:

```cpp
#include <map>
 #include <string>
 #include <queue>

 class Node;

 class Nodes {
    static Node* getNode(std::string mac);

    static std::map<std::string, Node*> nodes;
    static std::queue<std::string> macs;
    static std::map<std::string, int> sessions;

 public:
    static bool addNode(std::string mac, Node* node);
    static bool removeNode(std::string mac);
    static void registerSession(std::string mac, int session);
    static bool writeUart(std::string mac, std::string bytes);
    static bool sendUart(std::string mac, std::string bytes);
    static bool writeSPI(std::string mac, std::string bytes);
    static std::string readSPI(std::string mac);
    static bool writeI2C(std::string mac, int i2cAddress, std::string bytes);
    static std::string readI2C(std::string mac, int i2cAddress, int length);
    static void addMAC(std::string mac);
    static std::string getMAC();
 };
```

Here's its definition:

```cpp
#include "nodes.h"
 #include "node.h"
 #include <nymph/nymph.h>

 // Static initialisations.
 std::map<std::string, Node*> Nodes::nodes;
 std::queue<std::string> Nodes::macs;
 std::map<std::string, int> Nodes::sessions;

 Node* Nodes::getNode(std::string mac) {
    std::map<std::string, Node*>::iterator it;
    it = nodes.find(mac);
    if (it == nodes.end()) { return 0; }

    return it->second;
 }

 bool Nodes::addNode(std::string mac, Node* node) {
    std::pair<std::map<std::string, Node*>::iterator, bool> ret;
    ret = nodes.insert(std::pair<std::string, Node*>(mac, node));
    if (ret.second) { macs.push(mac); }
    return ret.second;
 }

 bool Nodes::removeNode(std::string mac) {
    std::map<std::string, Node*>::iterator it;
    it = nodes.find(mac);
    if (it == nodes.end()) { return false; }  
    nodes.erase(it);
    return true;
 }
```

With the following methods, we can set and remove node class instances:

```cpp
void Nodes::registerSession(std::string mac, int session) {
    sessions.insert(std::pair<std::string, int>(mac, session));
 }
```

New MAC and RPC session IDs are registered with the following function:

```cpp
bool Nodes::writeUart(std::string mac, std::string bytes) {
    Node* node = getNode(mac);
    if (!node) { return false; }

    node->writeUart(bytes);

    return true;
 }

 bool Nodes::sendUart(std::string mac, std::string bytes) {
    std::map<std::string, int>::iterator it;
    it = sessions.find(mac);
    if (it == sessions.end()) { return false; }

    vector<NymphType*> values;
    values.push_back(new NymphString(bytes));
    string result;
    NymphBoolean* world = 0;
    if (!NymphRemoteClient::callCallback(it->second, "serialRxCallback", values, result)) {
          // 
    }

    return true;
 }

 bool Nodes::writeSPI(std::string mac, std::string bytes) {
    Node* node = getNode(mac);
    if (!node) { return false; }

    node->writeSPI(bytes);

    return true;
 }

 std::string Nodes::readSPI(std::string mac) {
    Node* node = getNode(mac);
    if (!node) { return std::string(); }

    return node->readSPI();
 }

 bool Nodes::writeI2C(std::string mac, int i2cAddress, std::string bytes) {
    Node* node = getNode(mac);
    if (!node) { return false; }

    node->writeI2C(i2cAddress, bytes);

    return true;
 }

 std::string Nodes::readI2C(std::string mac, int i2cAddress, int length) {
    Node* node = getNode(mac);
    if (!node) { return std::string(); }

    return node->readI2C(i2cAddress, length);
 }
```

The methods for writing and reading from the different interfaces are basically pass-through methods, merely using the MAC address to find the appropriate `Node` instance to call the method on.

Of note here is the `sendUart()` method, which uses the NymphRPC server to call the callback method on the appropriate node process to trigger its UART receive callback, as shown in the following code:

```cpp
void Nodes::addMAC(std::string mac) {
    macs.push(mac);
 }

 std::string Nodes::getMAC() {
     if (macs.empty()) { return std::string(); }

     std::string val = macs.front();
     macs.pop();
     return val;
  }
```

Finally, we got the methods used to set and get the MAC address for new nodes.

With this, we have the basics of the full integration server. In the next section, we will take a look at the firmware and client side of the system before looking at how everything fits together.

# Makefile

The Makefile for this part of the project looks as follows:

```cpp
export TOP := $(CURDIR)

 GPP = g++
 GCC = gcc
 MAKEDIR = mkdir -p
 RM = rm

 OUTPUT = bmac_server
 INCLUDE = -I .
 FLAGS := $(INCLUDE) -g3 -std=c++17 -U__STRICT_ANSI__
 LIB := -lnymphrpc -lPocoNet -lPocoUtil -lPocoFoundation -lPocoJSON
 CPPFLAGS := $(FLAGS)
 CFLAGS := -g3 
 CPP_SOURCES := $(wildcard *.cpp) $(wildcard devices/*.cpp)
 CPP_OBJECTS := $(addprefix obj/,$(notdir) $(CPP_SOURCES:.cpp=.o))

 all: makedir $(C_OBJECTS) $(CPP_OBJECTS) bin/$(OUTPUT)

 obj/%.o: %.cpp
    $(GPP) -c -o $@ $< $(CPPFLAGS)

 bin/$(OUTPUT):
    -rm -f $@
    $(GPP) -o $@ $(C_OBJECTS) $(CPP_OBJECTS) $(LIB)

 makedir:
    $(MAKEDIR) bin
    $(MAKEDIR) obj/devices

 clean:
    $(RM) $(CPP_OBJECTS)
```

This is a rather simple Makefile as we have no special demands. We gather the source files, determine the names of the resulting object files, and compile all of them before generating a binary out of these object files.

# The node

This section covers the firmware for the integration test specifically the reimplementation of the (Arduino) APIs used in the Sming framework.

Most crucial here is that we don't in any way modify the firmware code itself. The only parts that we wish to change from the original firmware image for the ESP8266 MCU are the APIs that our own code interacts with.

This means that we have to first determine the APIs that our code interacts with and reimplement these in a way that is supported on the target (desktop) platform. For our ESP8266-based firmware, this means, for example, that the Wi-Fi network side is left unimplemented, as we are using the local network stack of the OS and therefore don't care about such details.

Similarly, the I2C, SPI, and UART interfaces are implemented as mere stubs that call their respective counterparts on the RPC interface, which we looked at in the previous section. For the MQTT protocol client, we could use the `emqtt` MQTT library that is part of the Sming framework, but as one will quickly find out, this library is meant to be used on embedded systems where the code using it is responsible for connecting it to the network stack.

Our code interacts with the API offered by the `MqttClient` class in Sming. It uses `emqtt` for the MQTT protocol, and inherits from the `TcpClient` class. Following the code down the hierarchy, one will end up at the TCP connection class before diving into the underlying LWIP network library stack.

In order to save ourselves a lot of trouble, it's easiest to just use an alternative MQTT library, such as the Mosquitto client library, which is meant to be run on a desktop OS, and will therefore use the OS-provided sockets API. This will cleanly map to the methods provided by Sming's MQTT client class.

We can leave the header for this class almost entirely untouched, just adding our modifications to integrate the Mosquitto library, as follows:

```cpp
class TcpClient;
 #include "../Delegate.h"
 #include "../../Wiring/WString.h"
 #include "../../Wiring/WHashMap.h"
 #include "libmosquitto/cpp/mosquittopp.h"
 #include "URL.h"

 typedef Delegate<void(String topic, String message)> MqttStringSubscriptionCallback;
 typedef Delegate<void(uint16_t msgId, int type)> MqttMessageDeliveredCallback;
 typedef Delegate<void(TcpClient& client, bool successful)> TcpClientCompleteDelegate;

 class MqttClient;
 class URL;

 class MqttClient : public mosqpp::mosquittopp {
 public:
    MqttClient(bool autoDestruct = false);
    MqttClient(String serverHost, int serverPort, MqttStringSubscriptionCallback callback = NULL);
    virtual ~MqttClient();

    void setCallback(MqttStringSubscriptionCallback subscriptionCallback = NULL);

    void setCompleteDelegate(TcpClientCompleteDelegate completeCb);

    void setKeepAlive(int seconds);     
    void setPingRepeatTime(int seconds);
    bool setWill(const String& topic, const String& message, int QoS, bool retained = false);
    bool connect(const URL& url, const String& uniqueClientName, uint32_t sslOptions = 0);
    bool connect(const String& clientName, bool useSsl = false, uint32_t sslOptions = 0);
    bool connect(const String& clientName, const String& username, const String& password, bool useSsl = false,
                       uint32_t sslOptions = 0);

    bool publish(String topic, String message, bool retained = false);
    bool publishWithQoS(String topic, String message, int QoS, bool retained = false,
                                  MqttMessageDeliveredCallback onDelivery = NULL);

    bool subscribe(const String& topic);
    bool unsubscribe(const String& topic);

    void on_message(const struct mosquitto_message* message);

 protected:
    void debugPrintResponseType(int type, int len);
    static int staticSendPacket(void* userInfo, const void* buf, unsigned int count);

 private:
    bool privateConnect(const String& clientName, const String& username, const String& password,
                                  bool useSsl = false, uint32_t sslOptions = 0);

    URL url;
    mosqpp::mosquittopp mqtt;
    int waitingSize;
    uint8_t buffer[MQTT_MAX_BUFFER_SIZE + 1];
    uint8_t* current;
    int posHeader;
    MqttStringSubscriptionCallback callback;
    TcpClientCompleteDelegate completed = nullptr;
    int keepAlive = 60;
    int pingRepeatTime = 20;
    unsigned long lastMessage = 0;
    HashMap<uint16_t, MqttMessageDeliveredCallback> onDeliveryQueue;
 };
```

We're including the header file for the C++-based wrapper for the Mosquitto client library here from the version of the Mosquitto library that is included in the project for this chapter. This is because the official version of the library doesn't support building with MinGW.

With the header included, we have the class derive from the Mosquitto MQTT client class instead.

Naturally, the implementation of the Sming MQTT client class has been completely changed, as you can see in the following code:

```cpp
#include "MqttClient.h"
 #include "../Clock.h"
 #include <algorithm>
 #include <cstring>

 MqttClient::MqttClient(bool autoDestruct /* = false*/)
 {
    memset(buffer, 0, MQTT_MAX_BUFFER_SIZE + 1);
    waitingSize = 0;
    posHeader = 0;
    current = NULL;

    mosqpp::lib_init();
 }

 MqttClient::MqttClient(String serverHost, int serverPort, MqttStringSubscriptionCallback callback /* = NULL*/)
    {
    url.Host = serverHost;
    url.Port = serverPort;
    this->callback = callback;
    waitingSize = 0;
    posHeader = 0;
    current = NULL;

    mosqpp::lib_init();
 }
```

The constructor simply initializes the Mosquitto library, with no further input required:

```cpp

 MqttClient::~MqttClient() {
    mqtt.loop_stop();
    mosqpp::lib_cleanup();
 }
```

In the destructor (shown in the following code) we stop the MQTT client-listening thread that we launched when we connect to an MQTT broker and clean up the resources that were used by the library:

```cpp

 void MqttClient::setCallback(MqttStringSubscriptionCallback callback) {
    this->callback = callback;
 }

 void MqttClient::setCompleteDelegate(TcpClientCompleteDelegate completeCb) {
    completed = completeCb;
 }

 void MqttClient::setKeepAlive(int seconds) {
    keepAlive = seconds;
 }

 void MqttClient::setPingRepeatTime(int seconds) {
    if(pingRepeatTime > keepAlive) {
          pingRepeatTime = keepAlive;
    } else {
          pingRepeatTime = seconds;
    }
 }

 bool MqttClient::setWill(const String& topic, const String& message, int QoS, bool retained /* = false*/)
 {
    return mqtt.will_set(topic.c_str(), message.length(), message.c_str(), QoS, retained);
 }
```

We have a number of utility functions, not all of which are being utilized, but they are still implemented here for the sake of completeness. It's also hard to predict which ones will be required, therefore it's often better to implement more than strictly necessary, especially if they are small functions that take less time to implement than to find out whether that function or method is used at all. Let's look at the following code:

```cpp

 bool MqttClient::connect(const URL& url, const String& clientName, uint32_t sslOptions) {
    this->url = url;
    if(!(url.Protocol == "mqtt" || url.Protocol == "mqtts")) {
          return false;
    }

    waitingSize = 0;
    posHeader = 0;
    current = NULL;

    bool useSsl = (url.Protocol == "mqtts");
    return privateConnect(clientName, url.User, url.Password, useSsl, sslOptions);
 }

 bool MqttClient::connect(const String& clientName, bool useSsl /* = false */, uint32_t sslOptions /* = 0 */)
 {
    return MqttClient::connect(clientName, "", "", useSsl, sslOptions);
 }

 bool MqttClient::connect(const String& clientName, const String& username, const String& password,
                                   bool useSsl /* = false */, uint32_t sslOptions /* = 0 */)
 {
    return privateConnect(clientName, username, password, useSsl, sslOptions);
 }
```

The `connect` methods remain the same, as they all use the same `private` method of the class to perform the actual connection operation, as shown in the following code:

```cpp

 bool MqttClient::privateConnect(const String& clientName, const String& username, const String& password,
                                              bool useSsl /* = false */, uint32_t sslOptions /* = 0 */) {
    if (clientName.length() > 0) {
          mqtt.reinitialise(clientName.c_str(), false);
    }

    if (username.length() > 0) {
          mqtt.username_pw_set(username.c_str(), password.c_str());
    }

    if (useSsl) {
          //
    }

    mqtt.connect(url.Host.c_str(), url.Port, keepAlive);
    mqtt.loop_start();
    return true;
 }
```

This is the first section where we directly use the Mosquitto library. We reinitialize the instance either without a password or TLS (anonymous broker access), or with a password, or with TLS (left unimplemented here, as we don't need it).

In this method, we also start the listening thread for the MQTT client, which will handle all incoming messages so that we don't have to further concern ourselves with this aspect of the process. Let's look at the following code:

```cpp

 bool MqttClient::publish(String topic, String message, bool retained /* = false*/) {
    int res = mqtt.publish(0, topic.c_str(), message.length(), message.c_str(), 0, retained);
    return res > 0;
 }

 bool MqttClient::publishWithQoS(String topic, String message, int QoS, bool retained /* = false*/,
                                              MqttMessageDeliveredCallback onDelivery /* = NULL */)
 {
    int res = mqtt.publish(0, topic.c_str(), message.length(), message.c_str(), QoS, retained);

    return res > 0;
 }
```

The MQTT message-publish functionality directly maps to the Mosquitto library's methods:

```cpp

 bool MqttClient::subscribe(const String& topic) {
    int res = mqtt.subscribe(0, topic.c_str());
    return res > 0;
 }

 bool MqttClient::unsubscribe(const String& topic) {
    int res = mqtt.unsubscribe(0, topic.c_str());
    return res > 0;
 }
```

Subscribing and unsubscribing both also map easily to the MQTT client instance, as shown in the following code:

```cpp

 void MqttClient::on_message(const struct mosquitto_message* message) {
    if (callback) {
          callback(String(message->topic), String((char*) message->payload, message->payloadlen));
    }
 }

```

Finally, we implement the Mosquitto `callback` method for when we receive a new message from the broker. For each received message, we then call the registered `callback` method (from the firmware code) to provide it with the payload and topic.

This takes care of the MQTT client aspect of the firmware. Next, we need to make the rest of the APIs compatible with a desktop OS.

The headers of the Sming framework that the firmware uses are as follows:

```cpp
#include <user_config.h>
 #include <SmingCore/SmingCore.h>
```

The first header file defines some platform-related features that we don't need. The second header is the one that we will add everything that we need to.

To check the firmware's code for API dependencies, we use standard text searching tools to find all function calls, filtering out any that do not call into our code but into the Sming framework. After doing this we can write the following SmingCore.h file with these dependencies:

```cpp
#include <cstdint>
 #include <cstdio>
 #include <string>
 #include <iostream>
 #include "wiring/WString.h"
 #include "wiring/WVector.h"
 #include "wiring/WHashMap.h"
 #include "FileSystem.h"
 #include "wiring/Stream.h"
 #include "Delegate.h"
 #include "Network/MqttClient.h"
 #include "Timer.h"
 #include "WConstants.h"
 #include "Clock.h"

 #include <nymph/nymph.h>

```

We start off with a combination of standard C library and STL includes, along with a number of headers that define the rest of the API that we are implementing. We also directly use a number of header files that define classes that are used throughout these APIs, but not by the firmware itself.

A class like the `Delegate` class is sufficiently abstract that it can be used as is. As we will see, the `Filesystem` and `Timer` classes required a fair bit of reworking to make them work for our purposes. We already looked at the modifications to the MQTT client earlier.

Naturally, we also include the header file for the NymphRPC library, which will allow us to communicate with the server side of the integration test, as shown in the following code:

```cpp
typedef uint8_t uint8;
 typedef uint16_t uint16;
 typedef uint32_t uint32;
 typedef int8_t int8;
 typedef int16_t int16;
 typedef int32_t int32;
 typedef uint32_t u32_t;
```

For compatibility reasons, we need to define a range of types that are used throughout the firmware code. These are equivalent to the types in `cstdint` from the C library, so we can use simple `typedefs`, as follows:

```cpp
#define UART_ID_0 0 ///< ID of UART 0
 #define UART_ID_1 1 ///< ID of UART 1
 #define SERIAL_BAUD_RATE 115200

 typedef Delegate<void(Stream& source, char arrivedChar, uint16_t availableCharsCount)> StreamDataReceivedDelegate;

 class SerialStream : public Stream {
    //

 public:
    SerialStream();
    size_t write(uint8_t);
    int available();
    int read();
    void flush();
    int peek();
 };

 class HardwareSerial {
    int uart;
    uint32_t baud;
    static StreamDataReceivedDelegate HWSDelegate;
    static std::string rxBuffer;

 public:
    HardwareSerial(const int uartPort);
    void begin(uint32_t baud = 9600);
    void systemDebugOutput(bool enable);
    void end();
    size_t printf(const char *fmt, ...);
    void print(String str);
    void println(String str);
    void println(const char* str);
    void println(int16_t ch);
    void setCallback(StreamDataReceivedDelegate dataReceivedDelegate);
    static void dataReceivedCallback(NymphMessage* msg, void* data);
    size_t write(const uint8_t* buffer, size_t size);
    size_t readBytes(char *buffer, size_t length);
 };

 extern HardwareSerial Serial;
```

The first API we fully reimplement is the hardware-based serial device. Since this communicates directly with the virtual interface in the server, we just need to provide the methods here, with the definition in the source file, as we will see in a moment.

We also declare a global instantiation of this serial object class, identical to how the original framework implementation handles it, as shown in the following code:

```cpp
 struct rboot_config {
    uint8 current_rom;
    uint32 roms[2];
 };

 int rboot_get_current_rom();
 void rboot_set_current_rom(int slot);
 rboot_config rboot_get_config();

 class rBootHttpUpdate;
 typedef Delegate<void(rBootHttpUpdate& client, bool result)> OtaUpdateDelegate;
 class rBootHttpUpdate {
    //

 public:
    void addItem(int offset, String firmwareFileUrl);
    void setCallback(OtaUpdateDelegate reqUpdateDelegate);
    void start();
 };

 void spiffs_mount_manual(u32_t offset, int count);
```

The rboot boot manager and SPIFFS filesystem-related functionality has no equivalent on a desktop system, so we declare them here (but as we'll see in a moment, they are left as empty stubs):

```cpp

 class StationClass {
    String mac;
    bool enabled;

 public:
    void enable(bool enable);
    void enable(bool enable, bool save);
    bool config(const String& ssid, const String& password, bool autoConnectOnStartup = true,
                                    bool save = true);
    bool connect();
    String getMAC();

    static int handle;
 };

 extern StationClass WifiStation;

 class AccessPointClass {
    bool enabled;

 public:
    void enable(bool enable, bool save);
    void enable(bool enable);
 };

 extern AccessPointClass WifiAccessPoint;

 class IPAddress {
    //
 public:
    String toString();
 };

 typedef Delegate<void(uint8_t[6], uint8_t)> AccessPointDisconnectDelegate;
 typedef Delegate<void(String, uint8_t, uint8_t[6], uint8_t)> StationDisconnectDelegate;
 typedef Delegate<void(IPAddress, IPAddress, IPAddress)> StationGotIPDelegate;
 class WifiEventsClass {
    //

 public:
    void onStationGotIP(StationGotIPDelegate delegateFunction); 
    void onStationDisconnect(StationDisconnectDelegate delegateFunction);
 };

 extern WifiEventsClass WifiEvents;
```

On the network side, we have to provide all of the class instances and related information that are normally used to connect to a WiFi access point and ensure that we are connected. As we aren't testing WiFi functionality here, these methods are of little use, but are needed to satisfy the firmware code and the compiler:

```cpp

 void debugf(const char *fmt, ...);

 class WDTClass {
    //

 public:
    void alive();
 };

 extern WDTClass WDT;
```

We then declare the debug-related output function as well as the watchdog class using the following code:

```cpp

 class TwoWire {
    uint8_t rxBufferIndex;
    std::string buffer;
    int i2cAddress;

 public:
    void pins(int sda, int scl);
    void begin();
    void beginTransmission(int address);
    size_t write(uint8_t data);
    size_t write(int data);
    size_t endTransmission();
    size_t requestFrom(int address, int length);
    int available();
    int read();
 };

 extern TwoWire Wire;

 class SPISettings {
    //
 public:
    //
 };

 class SPIClass {
    //

 public:
    void begin();
    void end();
    void beginTransaction(SPISettings mySettings);
    void endTransaction();
    void transfer(uint8* buffer, size_t numberBytes);
 };

 extern SPIClass SPI;
```

We declare the two types of communication buses here, as shown in the following code. Again, we declare that there is a global instantiation of each:

```cpp
void pinMode(uint16_t pin, uint8_t mode);
 void digitalWrite(uint16_t pin, uint8_t val);
 uint8_t digitalRead(uint16_t pin);

 uint16_t analogRead(uint16_t pin);
```

Since the firmware contains code that uses the GPIO and ADC pins, the above functions are needed as well.

```cpp
String system_get_sdk_version();
 int system_get_free_heap_size();
 int system_get_cpu_freq();
 int system_get_chip_id();
 int spi_flash_get_id();

 class SystemClass {
    //

 public:
    void restart();
 };

 extern SystemClass System;

 // --- TcpClient ---
 class TcpClient {
    //

 public:
    //
 };

 extern void init();
```

Finally, we declare a number of classes and functions that are mostly there to satisfy the compiler as they have no practical use for our purposes, though we could potentially implement advanced test scenarios this way.

Next, we'll look at the implementation of these functions using the following code:

```cpp

 #include "SmingCore.h"

 #include <iostream>
 #include <cstdio>
 #include <cstdarg>

 int StationClass::handle;
```

The `handle` variable is the one variable we declare as being static in this compile unit. Its purpose is to store the remote server handle ID for future operations after we connect to the RPC server, as shown in the following code:

```cpp

 void logFunction(int level, string logStr) {
    std::cout << level << " - " << logStr << std::endl;
 }
```

Just like in the server-side code, we define a simple logging function to use with NymphRPC, as shown in the following code:

```cpp

 void debugf(const char *fmt, ...) { 
    va_list ap;
    va_start(ap, fmt);
    int written = vfprintf(stdout, fmt, ap);
    va_end(ap);
 }
```

We implement the simple debug output function using C-style string formatting features to fit the function's signature, as shown in the following code:

```cpp

 StreamDataReceivedDelegate HardwareSerial::HWSDelegate = nullptr;
 std::string HardwareSerial::rxBuffer;
 HardwareSerial Serial(0);
```

We define the serial callback delegate along with the serial receive buffer as static, as we assume the presence of a single UART capable of **receiving data** (RX), which happens to be the case on the ESP8266 MCU. We also create a single instance of the `HardwareSerial` class, for UART 0, as shown in the following code:

```cpp

 SerialStream::SerialStream() { }
 size_t SerialStream::write(uint8_t) { return 1; }
 int SerialStream::available() { return 0; }
 int SerialStream::read() { return 0; }
 void SerialStream::flush() { }
 int SerialStream::peek() { return 0; }
```

This class is just there to act as a stub. As none of the code actually uses this object's methods, we can leave them all unimplemented, as shown in the following code:

```cpp
HardwareSerial::HardwareSerial(const int uartPort) { 
    uart = uartPort; 
 }

 void HardwareSerial::begin(uint32_t baud/* = 9600*/) { 
    this->baud = baud;
 }

 void HardwareSerial::systemDebugOutput(bool enable) { }
 void HardwareSerial::end() { }
 size_t HardwareSerial::printf(const char *fmt, ...) { 
    va_list ap;
    va_start(ap, fmt);
          int written = vfprintf(stdout, fmt, ap);
          va_end(ap);

    return written;
 }

 void HardwareSerial::print(String str) {
    std::cout << str.c_str();
 }

 void HardwareSerial::println(String str) {
    std::cout << str.c_str() << std::endl;
 }

 void HardwareSerial::println(const char* str) {
    std::cout << str << std::endl;
 }

 void HardwareSerial::println(int16_t ch) {
    std::cout << std::hex << ch << std::endl;
 }

 void HardwareSerial::setCallback(StreamDataReceivedDelegate dataReceivedDelegate) {
    HWSDelegate = dataReceivedDelegate;
 }
```

A lot of the methods in this class are simple enough that they can be implemented as a simple write to the standard (system) output or with an assignment to a variable. Occasionally a method is left unaltered from the original, though even for the setting of the callback delegate function in the last method in this group, the original code is called into the C-based low-level APIs of the ESP8266's SDK. Let's look at the following code:

```cpp

 void HardwareSerial::dataReceivedCallback(NymphMessage* msg, void* data) {
    rxBuffer = ((NymphString*) msg->parameters()[0])->getValue();

    SerialStream stream;
    int length = rxBuffer.length();
    int i = 0;
    HWSDelegate(stream, rxBuffer[i], length - i);
 }
```

To receive UART messages, we define a NymphRPC callback function, which for that reason is defined as being static. Since the ESP8266 only has a single UART capable of receiving data this suffices.

When called, this method reads out the payload being received on the UART and calls the `callback` function that the firmware registered previously, as shown in the following code:

```cpp

 size_t HardwareSerial::write(const uint8_t* buffer, size_t size) {
    vector<NymphType*> values;
    values.push_back(new NymphString(WifiStation.getMAC().c_str()));
    values.push_back(new NymphString(std::string((const char*) buffer, size)));
    NymphType* returnValue = 0;
    std::string result;
    if (!NymphRemoteServer::callMethod(StationClass::handle, "writeUart", values, returnValue, result)) {
          std::cout << "Error calling remote method: " << result << std::endl;
          NymphRemoteServer::disconnect(StationClass::handle, result);
          NymphRemoteServer::shutdown();
          return 0;
    }

    if (returnValue->type() != NYMPH_BOOL) {
          std::cout << "Return value wasn't a boolean. Type: " << returnValue->type() << std::endl;
          NymphRemoteServer::disconnect(StationClass::handle, result);
          NymphRemoteServer::shutdown();
          return 0;
    }

    return size;
 }
```

Writing to the remote UART is done using an RPC call. To do this, we create an STL vector and fill it with the parameters to pass in the proper order—in this case, the node's MAC address and the data that we wish to send on the remote UART.

After this, we use the NymphRPC handle that we got when we connected to call the RPC server and wait for the response from the remote function, as shown in the following code:

```cpp

 size_t HardwareSerial::readBytes(char* buffer, size_t length) {
    buffer = rxBuffer.data();
    return rxBuffer.length();
 }
```

Reading from the UART is done after we've received data on the UART, after which we can read it out with the following method, just as we would with the original code:

```cpp
int rboot_get_current_rom() { return 0; }
 void rboot_set_current_rom(int slot) { }
 rboot_config rboot_get_config() {
    rboot_config cfg;
    cfg.current_rom = 0;
    cfg.roms[0] = 0x1000;
    cfg.roms[1] = 0x3000;
    return cfg;
 }

 void rBootHttpUpdate::addItem(int offset, String firmwareFileUrl) { }
 void rBootHttpUpdate::setCallback(OtaUpdateDelegate reqUpdateDelegate) { }
 void rBootHttpUpdate::start() { }

 void spiffs_mount_manual(u32_t offset, int count) { }
```

Both the rboot boot manager and the SPIFFS filesystem are not used, so they can just return safe values, as shown in the following code. The **over-the-air** (**OTA**) functionality could potentially be implemented as well, depending on the kind of features of the system one would want to test:

```cpp

 StationClass WifiStation;

 void StationClass::enable(bool enable) { enabled = enable; }
 void StationClass::enable(bool enable, bool save) { enabled = enable; }
 String StationClass::getMAC() { return mac; }

 bool StationClass::config(const String& ssid, const String& password, bool autoConnectOnStartup /* = true*/,
                                    bool save /* = true */) {
    //

    return true;
 }
```

Since we don't have a Wi-Fi adapter that we want to use directly and are just using the OS's network capabilities, the `WiFiStation` object doesn't do a lot for most of its methods, except for when we actually connect to the RPC server, which is done using the following method:

```cpp

 bool StationClass::connect() {
    long timeout = 5000; // 5 seconds.
    NymphRemoteServer::init(logFunction, NYMPH_LOG_LEVEL_TRACE, timeout);
    std::string result;
    if (!NymphRemoteServer::connect("localhost", 4004, StationClass::handle, 0, result)) {
          cout << "Connecting to remote server failed: " << result << std::endl;
          NymphRemoteServer::disconnect(StationClass::handle, result);
          NymphRemoteServer::shutdown();
          return false;
    }

    vector<NymphType*> values;
    NymphType* returnValue = 0;
    if (!NymphRemoteServer::callMethod(StationClass::handle, "getNewMac", values, returnValue, result)) {
          std::cout << "Error calling remote method: " << result << std::endl;
          NymphRemoteServer::disconnect(StationClass::handle, result);
          NymphRemoteServer::shutdown();
          return false;
    }

    if (returnValue->type() != NYMPH_STRING) {
          std::cout << "Return value wasn't a string. Type: " << returnValue->type() << std::endl;
          NymphRemoteServer::disconnect(StationClass::handle, result);
          NymphRemoteServer::shutdown();
          return false;
    }

    std::string macStr = ((NymphString*) returnValue)->getValue();
    mac = String(macStr.data(), macStr.length());

    delete returnValue;
    returnValue = 0;

    // Set the serial interface callback.
    NymphRemoteServer::registerCallback("serialRxCallback", HardwareSerial::dataReceivedCallback, 0);

    return true;
 }
```

This is one of the first methods that gets called in the firmware when it tries to connect to the Wi-Fi access point. Instead of connecting to a Wi-Fi access point, we use this method to connect to the RPC server instead.

We start by initializing the NymphRPC library, calling the initialization method on its `NymphRemoteServer` class, and then connecting to the RPC server using the hardcoded location and port number. Upon successfully connecting to the RPC server, this client will receive a list of the available methods on the RPC server—in this case, all of the methods we registered, as we saw in the previous section on the simulation server.

Next, we request our MAC address from the server, verify that it's a string that we received, and set it for later use. Finally, we locally register the callback for the UART with NymphRPC, as shown in the following code. As we saw in the simulation server's section, the `Nodes` class on the server expects this callback to exist on the client:

```cpp

 AccessPointClass WifiAccessPoint;

 void AccessPointClass::enable(bool enable, bool save) {
    enabled = enable;
 }

 void AccessPointClass::enable(bool enable) {
    enabled = enable;
 }

 WifiEventsClass WifiEvents;

 String IPAddress::toString() { return "192.168.0.32"; }

 void WifiEventsClass::onStationGotIP(StationGotIPDelegate delegateFunction) {
    // Immediately call the callback.
    IPAddress ip;
    delegateFunction(ip, ip, ip);
 }

 void WifiEventsClass::onStationDisconnect(StationDisconnectDelegate delegateFunction) {
    //
 }

 WDTClass WDT;

 void WDTClass::alive() { }
```

We conclude this networking section with some more stub classes and, finally, the watchdog class, which might make for a nice point for advanced testing, including soft reset testing for long-running code. Of course, such advanced tests would also require that the code runs with the performance of the ESP8266's sub-100 MHz processor.

Of note here is the Wi-Fi events class, where we immediately call the `callback` function for a successful connection to the Wi-Fi access point, or at least pretend to. Without this step, the firmware would forever wait for something to happen. Let's look at the following code:

```cpp

 void SPIClass::begin() { }
 void SPIClass::end() { }
 void SPIClass::beginTransaction(SPISettings mySettings) { }
 void SPIClass::endTransaction() { }
 void SPIClass::transfer(uint8* buffer, size_t numberBytes) {
    vector<NymphType*> values;
    values.push_back(new NymphString(WifiStation.getMAC().c_str()));
    values.push_back(new NymphString(std::string((char*) buffer, numberBytes)));
    NymphType* returnValue = 0;
    std::string result;
    if (!NymphRemoteServer::callMethod(StationClass::handle, "writeSPI", values, returnValue, result)) {
          std::cout << "Error calling remote method: " << result << std::endl;
          NymphRemoteServer::disconnect(StationClass::handle, result);
          NymphRemoteServer::shutdown();
          return;
    }

    if (returnValue->type() != NYMPH_BOOL) {
          std::cout << "Return value wasn't a boolean. Type: " << returnValue->type() << std::endl;
          NymphRemoteServer::disconnect(StationClass::handle, result);
          NymphRemoteServer::shutdown();
          return;
    }
 }

 SPIClass SPI;
```

To write on the SPI bus, we again just call the RPC method on the server, getting the response once that call has been completed, as shown in the following code. For simplicity's sake, no SPI read functionality is implemented in this example project:

```cpp
 void TwoWire::pins(int sda, int scl) { }
 void TwoWire::begin() { }
 void TwoWire::beginTransmission(int address) { i2cAddress = address; }
 size_t TwoWire::write(uint8_t data) {
    vector<NymphType*> values;
    values.push_back(new NymphString(WifiStation.getMAC().c_str()));
    values.push_back(new NymphSint32(i2cAddress));
    values.push_back(new NymphString(std::to_string(data)));
    NymphType* returnValue = 0;
    std::string result;
    if (!NymphRemoteServer::callMethod(StationClass::handle, "writeI2C", values, returnValue, result)) {
          std::cout << "Error calling remote method: " << result << std::endl;
          NymphRemoteServer::disconnect(StationClass::handle, result);
          NymphRemoteServer::shutdown();
          return 0;
    }

    if (returnValue->type() != NYMPH_BOOL) {
          std::cout << "Return value wasn't a boolean. Type: " << returnValue->type() << std::endl;
          NymphRemoteServer::disconnect(StationClass::handle, result);
          NymphRemoteServer::shutdown();
          return 0;
    }

    return 1;
 }

 size_t TwoWire::write(int data) {
    vector<NymphType*> values;
    values.push_back(new NymphString(WifiStation.getMAC().c_str()));
    values.push_back(new NymphSint32(i2cAddress));
    values.push_back(new NymphString(std::to_string(data)));
    NymphType* returnValue = 0;
    std::string result;
    if (!NymphRemoteServer::callMethod(StationClass::handle, "writeI2C", values, returnValue, result)) {
          std::cout << "Error calling remote method: " << result << std::endl;
          NymphRemoteServer::disconnect(StationClass::handle, result);
          NymphRemoteServer::shutdown();
          return 0;
    }

    if (returnValue->type() != NYMPH_BOOL) {
          std::cout << "Return value wasn't a boolean. Type: " << returnValue->type() << std::endl;
          NymphRemoteServer::disconnect(StationClass::handle, result);
          NymphRemoteServer::shutdown();
          return 0;
    }

    return 1;
 }
```

After some stub methods in the I2C class, we find the `write` methods. These are essentially the same methods, calling the `remote` method to send the data to the simulated I2C bus on the server, as shown in the following code:

```cpp

 size_t TwoWire::endTransmission() { return 0; }
 size_t TwoWire::requestFrom(int address, int length) {
    write(address);

    vector<NymphType*> values;
    values.push_back(new NymphString(WifiStation.getMAC().c_str()));
    values.push_back(new NymphSint32(address));
    values.push_back(new NymphSint32(length));
    NymphType* returnValue = 0;
    std::string result;
    if (!NymphRemoteServer::callMethod(StationClass::handle, "readI2C", values, returnValue, result)) {
          std::cout << "Error calling remote method: " << result << std::endl;
          NymphRemoteServer::disconnect(StationClass::handle, result);
          NymphRemoteServer::shutdown();
          exit(1);
    }

    if (returnValue->type() != NYMPH_STRING) {
          std::cout << "Return value wasn't a string. Type: " << returnValue->type() << std::endl;
          NymphRemoteServer::disconnect(StationClass::handle, result);
          NymphRemoteServer::shutdown();
          exit(1);
    }

    rxBufferIndex = 0;
    buffer = ((NymphString*) returnValue)->getValue();
    return buffer.size();
 }
```

To read from the I2C bus, we use the preceding method, first writing the I2C address we wish to write to, then calling the RPC function to read from the simulated I2C device that should have data available to read, as shown in the following code:

```cpp

 int TwoWire::available() {
    return buffer.length() - rxBufferIndex;
 }

 int TwoWire::read() {
    int value = -1;
    if (rxBufferIndex < buffer.length()) {
          value = buffer.at(rxBufferIndex);
          ++rxBufferIndex;
    }

    return value;
 }

 TwoWire Wire;
```

The I2C read functionality is essentially the same as it was in the original implementation, as both just interact with a local buffer, as shown in the following code:

```cpp
String system_get_sdk_version() { return "SIM_0.1"; }
 int system_get_free_heap_size() { return 20000; }
 int system_get_cpu_freq() { return 1200000; }
 int system_get_chip_id() { return 42; }
 int spi_flash_get_id() { return 42; }

 void SystemClass::restart() { }

 SystemClass System;
```

Here are more stub implementations that could be of use for specific test scenarios:

```cpp
void pinMode(uint16_t pin, uint8_t mode) { }
 void digitalWrite(uint16_t pin, uint8_t val) { }
 uint8_t digitalRead(uint16_t pin) { return 1; }

 uint16_t analogRead(uint16_t pin) { return 1000; }
```

We left these functions unimplemented, but they could implement GPIO and ADC pins that are connected to virtual GPIO pins on the server side, to control devices and record data that does not use a UART, SPI, or I2C interface. The same would work for PWM functionality.

Moving on to the final part in this source file, we implement the main function as follows:

```cpp

 int main() {
    // Start the firmware image.
    init();

    return 0;
 }

```

Just like the Sming version of the entry point, we call the global `init()` function in the custom firmware code, which serves as the entrance point there. Conceivably, we could also perform various types of initialization in this main function if we needed to.

The filesystem class methods are implemented using a mixture of C-style file access and C++17-style filesystem operations, as shown in the following code:

```cpp
#include "FileSystem.h"
 #include "../Wiring/WString.h"

 #include <filesystem>
 #include <iostream>
 #include <fstream>

 namespace fs = std::filesystem;

 file_t fileOpen(const String& name, FileOpenFlags flags) {
    file_t res;

    if ((flags & eFO_CreateNewAlways) == eFO_CreateNewAlways) {
          if (fileExist(name)) {
                fileDelete(name);
          }

          flags = (FileOpenFlags)((int)flags & ~eFO_Truncate);
    }

    res = std::fopen(name.c_str(), "r+b");
    return res;
 }
```

To simplify this method, we ignore the provided flags and always open the file in full read and write mode (one would only implement the full set of flags if it contributed to the integration test in some way). Let's look at the following code:

```cpp

 void fileClose(file_t file) {
    std::fclose(file);
 }

 size_t fileWrite(file_t file, const void* data, size_t size) {
    int res = std::fwrite((void*) data, size, size, file);      
    return res;
 }

 size_t fileRead(file_t file, void* data, size_t size) {
    int res = std::fread(data, size, size, file);
    return res;
 }

 int fileSeek(file_t file, int offset, SeekOriginFlags origin) {
    return std::fseek(file, offset, origin);
 }

 bool fileIsEOF(file_t file) {
    return true;
 }

 int32_t fileTell(file_t file) {
    return 0;
 }

 int fileFlush(file_t file) {
    return 0; 
 }

 void fileDelete(const String& name) {
    fs::remove(name.c_str());
 }

 void fileDelete(file_t file) {
    //
 }

 bool fileExist(const String& name) {
    std::error_code ec;
    bool ret = fs::is_regular_file(name.c_str(), ec);
    return ret;
 }

 int fileLastError(file_t fd) {
    return 0;
 }

 void fileClearLastError(file_t fd) {
    //
 }

 void fileSetContent(const String& fileName, const String& content) {
    fileSetContent(fileName, content.c_str());
 }

 void fileSetContent(const String& fileName, const char* content) {
    file_t file = fileOpen(fileName.c_str(), eFO_CreateNewAlways | eFO_WriteOnly);
    fileWrite(file, content, strlen(content));
    fileClose(file);
 }

 uint32_t fileGetSize(const String& fileName) {
    int size = 0;
    try {
         size = fs::file_size(fileName.c_str());
     } 
    catch (fs::filesystem_error& e) {
         std::cout << e.what() << std::endl;
     }

    return size;
 }

 void fileRename(const String& oldName, const String& newName) {
    try {
          fs::rename(oldName.c_str(), newName.c_str());
    }
    catch (fs::filesystem_error& e) {
          std::cout << e.what() << std::endl;
    }
 }

 Vector<String> fileList() {
    Vector<String> result;
    return result;
 }

 String fileGetContent(const String& fileName) {
    std::ifstream ifs(fileName.c_str(), std::ios::in | std::ios::binary | std::ios::ate);

     std::ifstream::pos_type fileSize = ifs.tellg();
     ifs.seekg(0, std::ios::beg);
     std::vector<char> bytes(fileSize);
     ifs.read(bytes.data(), fileSize);

     return String(bytes.data(), fileSize);
 }

 int fileGetContent(const String& fileName, char* buffer, int bufSize) {
    if (buffer == NULL || bufSize == 0) { return 0; }
    *buffer = 0;

    std::ifstream ifs(fileName.c_str(), std::ios::in | std::ios::binary | std::ios::ate);

     std::ifstream::pos_type fileSize = ifs.tellg();
    if (fileSize <= 0 || bufSize <= fileSize) {
          return 0;
    }

     buffer[fileSize] = 0;
     ifs.seekg(0, std::ios::beg);
    ifs.read(buffer, fileSize);
    ifs.close();

     return (int) fileSize;
 }
```

These are all standard file operations, so they don't require a lot of explanation. The main reason why both C-style and C++17-style file access are used is because the original API methods assume a C-style way of handling things, and also because of the underlying, C-based SDK functionality.

We would map all API methods to a pure C++17 filesystem functionality, but this would be an additional time investment without any obvious payoff.

The timer functionality uses POCO's `Timer` class in Sming's `SimpleTimer` class to implement an equivalent functionality, as shown in the following code:

```cpp
#include "Poco/Timer.h"
 #include <iostream>

 typedef void (*os_timer_func_t)(void* timer_arg);

 class SimpleTimer {
 public:
    SimpleTimer() : timer(0) {
          cb = new Poco::TimerCallback<SimpleTimer>(*this, &SimpleTimer::onTimer);
    }

    ~SimpleTimer() {
          stop();
          delete cb;
          if (timer) {
                delete timer;
          }
    }

    __forceinline void startMs(uint32_t milliseconds, bool repeating = false) {
          stop();
          if (repeating) {
                timer = new Poco::Timer(milliseconds, 0);
          }
          else {
                timer = new Poco::Timer(milliseconds, milliseconds);
          }

          timer->start(*cb);
    }

    __forceinline void startUs(uint32_t microseconds, bool repeating = false) {
          stop();
          uint32_t milliseconds = microseconds / 1000;
          if (repeating) {
                timer = new Poco::Timer(milliseconds, 0);
          }
          else {
                timer = new Poco::Timer(milliseconds, milliseconds);
          }

          timer->start(*cb);
    }

    __forceinline void stop() {
          timer->stop();
          delete timer;
          timer = 0;
    }

    void setCallback(os_timer_func_t callback, void* arg = nullptr)   {
          stop();
          userCb = callback;
          userCbArg = arg;
    }

 private:
    void onTimer(Poco::Timer &timer) {
          userCb(userCbArg);
    }

    Poco::Timer* timer;
    Poco::TimerCallback<SimpleTimer>* cb;
    os_timer_func_t userCb;
    void* userCbArg;
 };
```

Finally, for the reimplementation of the `Clock` class, we use STL's chrono functionality, as shown in the following code:

```cpp
#include "Clock.h"
 #include <chrono>

 unsigned long millis() {
    unsigned long now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    return now;
 }

 unsigned long micros() {
    unsigned long now = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    return now;
 }

 void delay(uint32_t milliseconds) {
    //
 }

 void delayMicroseconds(uint32_t time) {   //
 }
```

Here, we leave the `delay` functions unimplemented since we don't need them at this point.

# Makefile

The Makefile for this part of the project looks like this:

```cpp
GPP = g++
 GCC = gcc
 MAKEDIR = mkdir -p
 RM = rm
 AR = ar
 ROOT = test/node
 OUTPUT = bmac_esp8266
 OUTLIB = lib$(OUTPUT).a
 INCLUDE = -I $(ROOT)/ \
                -I $(ROOT)/SmingCore/ \
                -I $(ROOT)/SmingCore/network \
                -I $(ROOT)/SmingCore/network/Http \
                -I $(ROOT)/SmingCore/network/Http/Websocket \
                -I $(ROOT)/SmingCore/network/libmosquitto \
                -I $(ROOT)/SmingCore/network/libmosquitto/cpp \
                -I $(ROOT)/SmingCore/wiring \
                -I $(ROOT)/Libraries/BME280 \
                -I $(ROOT)/esp8266/app
 FLAGS := $(INCLUDE) -g3 -U__STRICT_ANSI__
 LIB := -L$(ROOT)/lib -l$(OUTPUT) -lmosquittopp -lmosquitto  -lnymphrpc \
          -lPocoNet -lPocoUtil -lPocoFoundation -lPocoJSON -lstdc++fs \
          -lssl -lcrypto
 LIB_WIN :=  -lws2_32
 ifeq ($(OS),Windows_NT)
    LIB := $(LIB) $(LIB_WIN)
 endif
 include ./esp8266/version
 include ./Makefile-user.mk
 CPPFLAGS := $(FLAGS) -DVERSION="\"$(VERSION)\"" $(USER_CFLAGS) -std=c++17 -Wl,--gc-sections
 CFLAGS := -g3 
 CPP_SOURCES := $(wildcard $(ROOT)/SmingCore/*.cpp) \
                $(wildcard $(ROOT)/SmingCore/network/*.cpp) \
                $(wildcard $(ROOT)/SmingCore/network/Http/*.cpp) \
                $(wildcard $(ROOT)/SmingCore/wiring/*.cpp) \
                $(wildcard $(ROOT)/Libraries/BME280/*.cpp)
 FW_SOURCES := $(wildcard esp8266/app/*.cpp)
 CPP_OBJECTS := $(addprefix $(ROOT)/obj/,$(notdir) $(CPP_SOURCES:.cpp=.o))
 FW_OBJECTS := $(addprefix $(ROOT)/obj/,$(notdir) $(FW_SOURCES:.cpp=.o))
 all: makedir $(FW_OBJECTS) $(CPP_OBJECTS) $(ROOT)/lib/$(OUTLIB) $(ROOT)/bin/$(OUTPUT)
 $(ROOT)/obj/%.o: %.cpp
    $(GPP) -c -o $@ $< $(CPPFLAGS)
 $(ROOT)/obj/%.o: %.c
    $(GCC) -c -o $@ $< $(CFLAGS)
 $(ROOT)/lib/$(OUTLIB): $(CPP_OBJECTS)
    -rm -f $@
    $(AR) rcs $@ $^
 $(ROOT)/bin/$(OUTPUT):
    -rm -f $@
    $(GPP) -o $@ $(CPPFLAGS) $(FW_SOURCES) $(LIB)
 makedir:
    $(MAKEDIR) $(ROOT)/bin
    $(MAKEDIR) $(ROOT)/lib
    $(MAKEDIR) $(ROOT)/obj
    $(MAKEDIR) $(ROOT)/obj/$(ROOT)/SmingCore/network
    $(MAKEDIR) $(ROOT)/obj/$(ROOT)/SmingCore/wiring
    $(MAKEDIR) $(ROOT)/obj/$(ROOT)/Libraries/BME280
    $(MAKEDIR) $(ROOT)/obj/esp8266/app
 clean:
    $(RM) $(CPP_OBJECTS) $(FW_OBJECTS)
```

The main thing to note about this Makefile is that it gathers source files from two different source folders, both for the test API and for the firmware source. The former source files are first compiled to object files, which are assembled into an archive. The firmware source is used directly along with this test framework library, though we also have the firmware object files available if we need them.

The reason for creating an archive of the test API before linking it has to do with the way that the linker finds symbols. By using the AR tool, it will create an index of all symbols in the object files inside the archive, ensuring that we will not get any linker errors. Especially for large projects this is often a requirement to have the object files successfully link into a binary.

Compiling to object files first is also helpful with larger projects, as Make will ensure that only files that have actually changed will be recompiled, which can really speed up development time. Since the target firmware source for this project is fairly minimal, we can compile directly from the source files here.

We also include two more Makefiles from this one. The first includes the version number of the firmware source we are compiling with, which is useful since it'll ensure that the produced node binary will report the exact same version as the version installed on an ESP8266 module would. This making validation of a specific firmware version much easier.

The second is the Makefile with user-definable settings, copied *verbatim* from the firmware project Makefile, but with just the variables we need for the firmware source to compile and work, as shown in the following code:

```cpp
WIFI_SSID = MyWi-FiNetwork
 WIFI_PWD = MyWi-FiPassword

 MQTT_HOST = localhost
 # For SSL support, uncomment the following line or compile with this parameter.
 #ENABLE_SSL=1
 # MQTT SSL port (for example):
 ifdef ENABLE_SSL
 MQTT_PORT = 8883 
 else
 MQTT_PORT = 1883
 endif

 # Uncomment if password authentication is used.
 # USE_MQTT_PASSWORD=1
 # MQTT username & password (if needed):
 # MQTT_USERNAME = esp8266
 # MQTT_PWD = ESPassword

 # MQTT topic prefix: added to all MQTT subscriptions and publications.
 # Can be left empty, but must be defined.
 # If not left empty, should end with a '/' to avoid merging with topic names.
 MQTT_PREFIX = 

 # OTA (update) URL. Only change the host name (and port).
 OTA_URL = http://ota.host.net/ota.php?uid=

 USER_CFLAGS := $(USER_CFLAGS) -DWIFI_SSID="\"$(WIFI_SSID)"\"
 USER_CFLAGS := $(USER_CFLAGS) -DWIFI_PWD="\"$(WIFI_PWD)"\"
 USER_CFLAGS := $(USER_CFLAGS) -DMQTT_HOST="\"$(MQTT_HOST)"\"
 USER_CFLAGS := $(USER_CFLAGS) -DMQTT_PORT="$(MQTT_PORT)"
 USER_CFLAGS := $(USER_CFLAGS) -DMQTT_USERNAME="\"$(MQTT_USERNAME)"\"
 USER_CFLAGS := $(USER_CFLAGS) -DOTA_URL="\"$(OTA_URL)"\"
 USER_CFLAGS := $(USER_CFLAGS) -DMQTT_PWD="\"$(MQTT_PWD)"\"
 ifdef USE_MQTT_PASSWORD
 USER_CFLAGS := $(USER_CFLAGS) -DUSE_MQTT_PASSWORD="\"$(USE_MQTT_PASSWORD)"\"
 endif
 SER_CFLAGS := $(USER_CFLAGS) -DMQTT_PREFIX="\"$(MQTT_PREFIX)"\"
```

Including this Makefile sets all of these defines to be passed to the compiler. These are all preprocessor statements that are used to set strings or to change which parts of the code will be compiled, such as the SSL code.

However, for simplicity's sake, we aren't implementing SSL functionality for this example project.

# Building the project

For the server side, we have the following library dependencies:

*   NymphRPC
*   POCO

For the node, we have the following dependencies:

*   NymphRPC
*   POCO
*   Mosquitto

The NymphRPC library (described at the beginning of this section) is compiled according to the project's instructions and installed in a place where the linker can find it. The POCO libraries are installed using the system's package manager (Linux, BSD, or MSYS2) or by hand.

For the Mosquitto library dependency, we can compile the `libmosquitto` and `libmosquittopp` library files using the project's library version by using the Makefile in the `test/SmingCore/network/libmosquitto` folder. Again you should install the resulting library files where the linker can find them.

When not using MinGW, one can also use the generally available version via the OS's package manager or similar.

After these steps, we can compile the server and client using the following command-line command from the root of the project:

```cpp
make
```

This should compile both the server and node projects using the top-level Makefile, resulting in an executable for each in their respective `bin/` folder. You should ensure that the executable name and path in the server's `Node` class match that of the node executable's location.

We should now be able to run the project and start to collect test results. The project includes a stripped version of the ESP8266-based BMAC firmware, which we'll be covering in detail in [Chapter 9](d165297b-8be7-44f5-90b5-53b3bcb51d3b.xhtml), *Example - Building Monitoring and Control*. Please refer to that chapter to understand how to communicate with the simulated nodes via MQTT, how to turn on modules inside the firmware and how to interpret the data sent over MQTT by the modules.

After setting things up as described in that chapter - requiring at the least an MQTT broker and a suitable MQTT client - and turning on the BME280 module in the simulated node, we expect it to start sending over MQTT the temperature, humidity and air pressure values we set for the room the simulated node is in.

# Summary

In this chapter, we looked at how to effectively develop for MCU-based targets in a way that allows us to test them without expensive and long-winded development cycles. We learned how to implement an integration environment that allows us to debug MCU-based applications from the comfort of a desktop OS and the tools it provides.

The reader should now be able to develop integration tests for MCU-based projects and effectively use OS-based tools to profile and debug them before doing final integration work on real hardware. The reader should also be able to perform on-chip debugging, and have a feel for the relative cost of specific software implementations.

In the next chapter, we'll develop a simple infotainment system, based on an SBC platform.