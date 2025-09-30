# Manual Robotics with Intel Edison

In [Chapter 5](45fccd6a-a75e-465d-89dc-dad31f528ac1.xhtml), *Autonomous Robotics with Intel Edison*, we dealt with robotics and the autonomous side of it. Here, we are going to deep dive into the field of manual robotics. A manual robot may not typically be called a robot, so more specifically, we will deal with the manual control of robots that have some autonomous characteristics. We are primarily dealing with the development of UGVs and its control using WPF applications. WPF applications have already been discussed in [Chapter 3](3bd53219-a287-4d8f-9a58-5a06c5b14062.xhtml), *Intel Edison and IoT (Home Automation)*, where we communicated with Edison using the MQTT protocol. Here, we are going to do the same using serial port communication. We will also learn how to make our bot fully wireless. The topics we will be covering are as follows:

*   Manual robotic system—architecture and overview
*   2WD and 4WD mechanisms
*   Serial port communication with Intel Edison
*   Making the bot wireless in robotics
*   A simple WPF application to switch an LED on and off using Intel Edison
*   High performance motor driver example with code
*   Black: e-track platform for UGV
*   Universal robot controller for UGV

All the codes for this chapter will be written in Arduino IDE, and for the software side in Visual Studio we are using C# and xaml.

# Manual robotic system

We have had a look at the autonomous robotic architecture. Manual robotics also deal with a similar architecture; the only difference being that we have a fully-fledged controller that is responsible for most of the action:

![](img/image_06_001.jpg)

Manual robot architecture

There isn't much of a difference between the architecture discussed here and the one discussed in [Chapter 5](45fccd6a-a75e-465d-89dc-dad31f528ac1.xhtml), *Autonomous Robotics with Intel Edison*. We've added a receiver and a transmitter unit here which would have been present in the earlier use case as well. When dealing with robotics, the entire architecture falls under the same roof.

Manual robotics may not be limited to only manual robots. It may be a blend of manual and autonomous functionality, because a fully manual robot may not typically be called a robot. However, we are aware of **Unmanned Ground Vehicles** (**UGVs**) and **Unmanned Aerial Vehicles** (**UAVs**). Sometimes the terminology may define them as robots, but until, and unless, they don't have at least some manual functionality, they may not be referred to as robots. This chapter mainly deals with UGVs, and like every robot or UGV, we need a sturdy chassis.

# Chassis in robotics: 2WD and 4WD

The reader is expected to develop their own robot, and thus you will be required to learn about drive mechanisms and a choice of chassis. Ideally, there are two types of drive mechanisms and the choice for the chassis is done on the basis of the drive mechanism used. Normally we don't want a chassis that over-stresses our motors, nor do we want one that may get stuck while exposed to the outdoor environment. In a typical line follower robot, as discussed in [Chapter 5](45fccd6a-a75e-465d-89dc-dad31f528ac1.xhtml), *Autonomous Robotics with Intel Edison,* the most common and the most widely-used drive mechanism is a two-wheel drive, as normally these operate on smooth surfaces and in indoor environments.

# Two-wheel drive

**Two-wheel drive** (**2WD**) refers to the driving mechanism involving two motors and two wheels, and it may typically contain a castor for balancing:

![](img/image_06_002.jpg)

2WD typical layout

The rear motors provide mobility and also acts as a steering mechanism for the robot. For the robot to move right, you may switch off the right motor and let the left motor do the work. However, in that way, the turning radius may be more extreme and the power consumption for the left motor increases as it needs to overcome the force of friction provided by the right wheel. The castor being omni-directional provides less resistance, but this isn't necessarily preferred. The other way is to rotate the right wheel backwards while the left wheel moves forwards; this method allows the robot to turn on its own axis and provide a zero turning radius:

![](img/image_06_003.jpg)

2WD turning of robot on its own axis

When we follow the preceding method, there is a lot less stress on the motors, and with the castor being omni-directional, the robot executes an almost perfect turn.

A chassis can be built with any material and the design should be such that it provides as less stress as possible on the motors.

However, when dealing with four-wheel drives, design plays a factor:

![](img/image_06_004.jpg)

4WD typical drive

Typically, these are powered by four motors (here, motor 1 are the left-side motors, whereas motor 2 are the right-side motors), which can be controlled independently. Usually during rotation, we don't stop the motors on the other side because it creates a lot of pressure on those motors. The other possible option is to rotate on opposite sides—but there is a catch.

Usually, the length of the robot in these cases needs to be either equal to the breadth or even less so. Otherwise, a condition may arise called wheel slip. To prevent such a condition, the design is normally such that the entire model, along with its wheels, fits in a circle, shown as follows:

![](img/image_06_005.jpg)

4WD—design

There is another parameter that may be considered and that is the distance between two wheels, as it must be less than the diameter of the wheels. This is because if we are exposed to rough terrain, the bot will be able to come out.

This will happen if the structure of the bot fits in a circle and the length and the distance between the front and rear wheels are less than the distance between the left and right side. Here, while wheel slip happens, it's reduced considerably and is almost negligible. Have a look at the following image for more information:

![](img/image_06_006.jpg)

Rotation of a 4WD robot

In the preceding image, the concept should become clearer as the robot tends to stay in the circle it's enclosed in, executing more or less a pivotal turn or a zero radius turn.

Now that we know how the robot chassis can be designed, let's have a look at the ready-made designs available. On sites such as Amazon and eBay, a lot of chassis are available pre-fabricated, following existing design patterns. If you want to fabricate your own chassis, then it's better to follow the preceding design pattern, especially in a 4WD configuration.

# Serial port communication with Intel Edison

When we have a manual robot, we need to control it. So, to control it we need some mode of communication. This is attained by using serial port communication. In Intel Edison, we have three serial ports; let's call it Serialx, where x stands for 1 or 2\. These serial ports can be accessed by the Arduino IDE:

*   **Serial**:
*   *   **Name**: Multi-gadget, firmware programming, serial console, or OTG port
    *   **Location**: USB-micro connector near the center of the Arduino breakout board
    *   **ArduinoSWname**: Serial
    *   **Linuxname**: `/dev/ttyGS0`

This port allows us to program Intel Edison and is also the default port for the Arduino IDE. In the Arduino breakout board, this is activated when the toggle switch or SW1 is towards the OTG port and away from the USB slot.

*   **Serial1**:
*   *   **Name**: UART1, the general-purpose TTL-level port (Arduino shield compatibility)
    *   **Location**: Pins 0 (RX) and 1 (TX) on the Arduino shield interface headers.
    *   **ArduinoSWname**: Serial1
    *   **Linuxname**: `/dev/ttyMFD1`

This port is the pin numbers 0 and 1, which are used as Rx and Tx. This port is used for the remote control of Edison over an RF network or any external Bluetooth device.

*   **Serial2**:
    *   **Name**: UART2, Linux kernel debug, or debug spew port
    *   **Location**: USB-micro connector near the edge of the Arduino board
    *   **ArduinoSWname**: Serial2
    *   **Linuxname**: `/dev/ttyMFD2`

This is one of the most useful ports whose communication baud rate is 115200\. This is usually the port that is accessed through the PuTTY console and is used to isolate boot problems. When the Serial2 object is created and initialized with `Serial2.begin()`, the kernel's access to the port is removed and the Arduino sketch is given control until `Serial2.end()` is invoked.

*   **Virtual ports**:
    *   **Name**: VCP or virtual communications port (appears only when the Serial-over-USB device is connected)
    *   **Location**: Big type A USB port nearest the Arduino power connector
    *   **ArduinoSWname**: Not supported by default
    *   **Linuxname**: `/dev/ttyACMx` or `/dev/ttyUSBx`

This is the USB port of your Intel Edison's Arduino breakout board. The switch must be towards the USB port for enabling the device. Multiple USB devices can be connected using a USB hub.

Consider the following example of code:

[PRE0]

This will just print Hi, Reporting in from Intel Edison in the serial monitor. From the code, it's evident that `Serial` has been used, which is the default one.

# Making the system wireless

For making systems wireless in robotics, there are many options available. The choice of hardware and protocol depends on certain factors, which are as follows:

*   Availability of mobile network coverage
*   Rules and regulations over RF in your operating country
*   Maximum distance required
*   Availability of Internet connectivity

If we use a GSM module, then mobile network coverage is a must. We may need to get clearance for the RF and ensure that it does not interfere with other signals. The maximum distance is another factor to consider, as distance is limited when using Bluetooth. Bluetooth connectivity can be hampered if the distance exceeds. The same goes for RF, but RF coverage can be increased based on the antenna used. If there is Internet connectivity over an area, then MQTT itself can be used, which was again discussed in [Chapter 3](3bd53219-a287-4d8f-9a58-5a06c5b14062.xhtml), *Intel Edison and IoT (Home Automation)*.

RF, or radio frequency, can be used for small applications. Wi-Fi can also be used with Edison, but let's cover a wide spectrum of devices and take a look into how RF can be used.

Normally, RF modules follow a **Universal Asynchronous Receiver Transmitter** (**UART**) protocol. These generally have a USB link and a serial link. A serial link can be converted with a USB link using a serial to USB converter. There are many options to choose from when buying an RF module set.

Make a note of what the maximum range and the operating frequency are. All details can be obtained from the place you buy the product.

Normally, the pin out of a RF serial link is shown as follows:

![](img/image_06_007.jpg)

RF serial link pin out

Here is a product of [http://robokits.co.in/](http://robokits.co.in/), which we used in our projects:

![](img/image_06_008.jpg)

RF USB Serial link. Picture source: [http://robokits.co.in/](http://robokits.co.in/)

The module can consist of five pins. We only need to deal with the four pins, as mentioned in the preceding figure.

An RF kit is used to manually control the robot wirelessly by sending commands. These are sent using serial port communication. The controller may use an RF module that has a USB link, or you can use a serial to USB converter to connect it to your PC. The connections of an RF serial link with a serial to USB converter is shown as follows:

![](img/image_06_009.jpg)

Connections of RF serial link to a serial to USB converter

The connection shown earlier is for connecting an RF serial link to a USB. This applies to the computer side as we want to control it by a PC. We must use two RF modules; one is for Edison and the other is for the controller app or the PC. To connect the RF module to Intel Edison, have a look at the following image:

![](img/image_06_010.jpg)

Connections of a RF serial link to Intel Edison

Intel Edison has Rx and Tx pins, which are pins 0 and 1 respectively. The overall architecture is shown as follows:

![](img/image_06_011.jpg)

Wireless control of Intel Edison

Now that we know how the hardware pieces are used for wireless communication, the programming part of the preceding model in Intel Edison is ridiculously simple. Just replace `Serial` with `Serial1`, as we are using the Rx and Tx pins:

[PRE1]

The preceding code sends data to a controller app by using the Rx and Tx pins over an RF network. Now we will have a look on the controller application side, where we will develop a WPF application to control our device.

# WPF application for LED on and off

In [Chapter 3](3bd53219-a287-4d8f-9a58-5a06c5b14062.xhtml), *Intel Edison and IoT (Home Automation), *we looked at using a WPF application and MQTT connection, learning that we could control our Intel Edison using MQTT protocol. However, here, we'll be dealing with serial port communication. Since we have already discussed WPF applications and how to create projects, and created an hello world application, we won't discuss the basics in this chapter, and will instead get into the application directly. Our problem statement in this chapter is to switch an LED on and off using a WPF application via serial port communication.

Start with creating a new WPF project and name it `RobotController`:

![](img/image_06_012.png)

RobotController—1

Next, in MainWindow.xaml, we'll design the UI. We'll use the following controls:

*   `Buttons`
*   `TextBox`
*   `TextBlocks`

Design your UI as follows:

![](img/image_06_013.png)

RobotController—2

The xaml code for the preceding UI is as follows:

[PRE2]

By default, we have written `COM13`; however, that might change. A total of four buttons are added, which are on, off, connect, and disconnect. We also have a `TextBlock` to display the status. You can tamper with this code for more customization.

Now our job is to write the backend for this code, which will also include the logic behind it.

Let's first create event handlers. Double click on each of the buttons to create an event. The preceding code contains the event handlers. Once done, include the following namespace for the use of the `SerialPort `class:

[PRE3]

Next, create an object of the `SerialPort` class:

[PRE4]

Now navigate to the connect button's event handler method, and here add the code required to connect your app to Intel Edison via a serial port. A try catch block is added to prevent crashes while connecting. The most common reason for a crash is an incorrect port number or the USB is not connected:

[PRE5]

In the preceding code, we stored the `com` port number in a string type variable. Next, we assign the object's `PortName` member with the `portName`. We also set the baud rate to `9600`. Finally, we open the port and write in the status box `connected`.

Next, we write the code for the disconnect event handler:

[PRE6]

`sp.close()` disconnects the connection. It's safe to write these under a try catch block.

Finally, we write the code for the on and off buttons' event handlers:

[PRE7]

In the preceding code, we used the `WriteLine` method and sent a string. The device, which is connected with the application using a serial port, receives the string and an action may be triggered. This sums up the entire process. The entire code for `MainWindow.xaml.cs` is provided as follows:

[PRE8]

Now we have the application ready to control our Intel Edison. Let's test it out. Open up the Arduino IDE. We'll write a small code for Intel Edison that will read serial data from the application so that the on board LED will turn on and off based on the incoming data.

Write the following code to do the same:

[PRE9]

When you burn this code, go to Visual Studio and run your WPF application. Enter the port number; it must be the same as your Arduino programming port, that is, the serial port. After that, press the on button. The on board LED should glow. It should turn off when you press the off button. Thus, we now have a very basic understanding of how to communicate with Edison using serial port communication via a WPF application. As the chapter progresses, we'll see how to efficiently control a robot with keyboard controls.

# High performance motor driver sample with code

In [Chapter 5](45fccd6a-a75e-465d-89dc-dad31f528ac1.xhtml), *Autonomous Robotics with Intel Edison,* we saw an application of L293D and we also wrote some code for it to control motors. However, L293D fails in high performance applications. To tackle this, we had a brief discussion about an alternative high-power driver.

Here, we'll deep dive into the driver, as it has been my personal favorite and is used in virtually all our robots:

![](img/image_06_014.jpg)

Dual motor driver high power. Picture source: [http://robokits.co.in/motor-drives/dual-dc-motor-driver-20a](http://robokits.co.in/motor-drives/dual-dc-motor-driver-20a)

The driver has the following five control pins:

*   **Gnd**: Ground
*   **DIR**: When low, the motor rotates in one direction; when high, it rotates in another direction
*   **PWM**: Pulse width modulation to control the speed of the motor; the recommended frequency range is 20Hz - 400Hz
*   **BRK**: When high, it halts the motor in operation
*   **5V**: Regulated 5V output from motor driver board

Now let's write a simple code to operate this driver with all the circuitry:

![](img/image_06_015.jpg)

Circuit diagram for motor driver

The preceding circuit is really simple to understand. You don't need to connect the 5V pin. You may use a single ground by shorting two wires of the grounds from the board. Let's now write a code to operate this. This motor driver is very efficient in controlling high torque motors. Since PWM functionality is used, we will therefore use half of the original speed of `122`:

[PRE10]

In the preceding code, it is worth noting the functionalities of `Brake` and the `pwm`. Even if you are using a low torque motor, the motor won't rotate if the brake is set to high. Similarly, efficient speed control can be achieved by `pwm` pins. So, by default, we have set everything else on the `pwm` to low. This again depends on the polarity of your motors. Feel free to tamper with the connections so that everything is set with the preceding code. Reverse the connections of your motor if you find an opposite rotation of both sides in the forward condition.

Observe how efficiently motors are controlled by a very simple code.

Now that we know how to control motors more effectively, we'll now move forward with our special black-e-track UGV platform where we developed our own controller for controlling the robot. Almost all the parts were bought from [http://robokits.co.in](http://robokits.co.in).

# 4WD UGV (black-e-track)

The name might be a bit fancy but this UGV is quite simple with the only difference being that it contained four high torque motors powered by a single dual driver motor driver circuit. Initially, we used two driver circuits but then we shifted to one. It was powered by a Li-Po battery but all tests were conducted using an SMPS. The UGV was controlled by a WPF application with the name of universal remote controller. This UGV was also fitted with a camera with an operating frequency of 5.8 GHz. The UGV was also wireless using a 2.4 GHz RF module. Let's have a look at the hardware required apart from the Intel Edison:

*   30 cm by 30 cm chassis(1)
*   10 cm diameter wheels(4)
*   High torque motors 300 RPM 12V(4)
*   20A dual motor driver(1)
*   RF 2.4 GHz USB link(1)
*   RF 2.4 GHz Serial link(1)
*   Li-Po battery (minimum voltage supply: 12V; maximum current drawn: 3-4A)

This section will cover the hardware aspect of it and how to develop the controller application using WPF. The chassis combined with the wheels falls under the deign principle discussed in preceding figure. Let's have a look at the circuit diagram of the UGV. If the robot is made using the earlier mentioned hardware, then the robot will perform well in rough terrain and also be able to climb a steep slope of 60-65 degrees (tested):

![](img/image_06_016.jpg)

Circuit Diagram for UGV

Motor 1 represents the left hand side motors while motor 2 represents the right hand side motors. Both the left hand side motors are shorted and same goes for the left hand side motors as well. This particular UGV was programmed to receive certain characters through serial port communication and provide some action based on that. Now, let's have a look at the code of the Intel Edison:

[PRE11]

The preceding code executes functions based on the data received. The following table summarises the characters responsible for the data received:

| **Character Received** | **Action Undertaken** |
| `0` | Fast back |
| `1` | Fast front |
| `3` | Fast right |
| `4` | Fast left |
| `5` | Stop |
| `6` | Slow front |
| `7` | Slow back |
| `8` | Slow right |
| `9` | Slow left |

We have created two macros for max and slow speed. The parameters for methods of motion execution is the speed that is passed based on the data received. You can test it using your serial monitor. Now, that we have the hardware lets write a software for it. This software will be able to control the robot using keyboard as well.

# Universal robot controller for UGV

Before deep diving into the controller, clone the following GitHub repository to your PC. The code is itself around 350+ lines so some parts are to be discussed:

[https://github.com/avirup171/bet_controller_urc](https://github.com/avirup171/bet_controller_urc)

So initially let's design the UI first:

![](img/image_06_017.png)

Screenshot of URC

For simplicity two sections of fast and slow controls are included. However it can be merged into one and using a checkbox. We have a connection pane on the right hand top side. The commands are displayed. A default password for `12345` was added which was done to avoid crashes and unauthorized use. However it's a simple controller and can be used with UGVs pretty much efficiently.

If you have a close look over the UI, then you will find a button named Press to activate keyboard control. Once you click on the button, the keyboard control gets activated. Now here you need to assign keyboard pressed and keyboard release event. This can be done by selecting the control and in the properties windows, click on the following icon. This manages all the event handlers for the selected control:

![](img/image_06_018.jpg)

Properties window

When we press the key on a keyboard two events are triggered. The first is when we press the key and the second is when we release the key

Now when you click on it you will get all the possible events associated with it. Scroll down to KeyDown and KeyUp. Double click on both to create the associated event handlers. The control buttons have a different event associated with them. That is we send data when the button is pressed. When the button is released, 5 which is for stop is sent. You can assign the events by the properties window as shown earlier:

[PRE12]

Assign names to all the buttons and create their respective event handlers. We have also created three progress bars. A xaml code for a progress bar is shown as follows:

[PRE13]

For the keyboard up and down events, the respective xaml code is:

[PRE14]

Two events are created. One for the key pressed and another for key released.

The xaml code for the preceding UI is given as follows:

[PRE15]

Now that the UI is ready, let's go to the main C# code. The event handlers are also in place. Initially include the `System.IO.Ports` namespace and create an object of that class. After that the keyboard pressed event will be handled with our code:

[PRE16]

In the preceding code, we used the following keys:

| **Serial No** | **Keyboard keys** | **Commands executed** |
| `1` | *W* | Fast forward |
| `2` | *A* | Fast left turn |
| `3` | *S* | Fast backward |
| `4` | *D* | Fast right turn |
| `5` | Numpad *8* | Slow forward |
| `6` | Numpad *2* | Slow Backward |
| `7` | Numpad *4* | Slow left turn |
| `8` | Numpad *6* | Slow right turn |

Based on the input, we sent that particular character. While for the key up or key released event, we simply send `5` which means stop:

[PRE17]

The connect and disconnect events are same as before. Now each button will have two methods. The first one is of `GotMouseCapture` and the second one is of `LostMouseCapture`.

Take the example of the front button under fast control:

[PRE18]

Similarly apply it for the other controls. Only 360 degree left and right is associated with a button click event. The entire code is pasted as shown below of `MainWindow.xaml.cs`:

[PRE19]

In the preceding code, some facts to be noted are as follows:

*   If the password is not entered, all the buttons are disabled
*   The password in `12345`
*   All buttons are associated with `gotMouseCapture` and `lostMouseCapture`
*   Only 360 degree rotation button follows a click event

Once you are able to successfully develop the project, test it out. Connect the RF USB link to your PC. Install all the required drivers and test it out.The entire process is mentioned as follows:

*   Connect the RF USB link to your PC.
*   Make sure your Intel Edison is powered on and connected to our bot. You can use a USB hub to power the Intel Edison and connect the hub to a power bank.
*   After you click on connect, the WPF application should get connected to your RF device.
*   Test whether your robot is working. Use a FPV 5.8 GHz camera to get a live view from your UGV.

# Open-ended question for the reader

What we have developed so far is a kind of UGV and not typically a robot, although we can configure it to be one. To develop an autonomous and manual robot, we normally design a robot to perform a certain task, however we keep manual control as well so that we can take back control whenever we desire. More appropriately, it may not be fully manual nor fully autonomous. Think of a drone. We just specify the waypoints on the map and the drone follows the waypoints. That's one of the classic examples. Now the reader's job is to combine the line follower robot discussed previously and manual robot discussed here and combine it into a single platform.

# Summary

We have come to the end of the chapter as well as to the end of the book. In this chapter, we had a chance to have a look at some in-depth concepts of manual robotics and UGVs. We developed our own software for robot controlling. We also learned how to make our robots wireless and the ways to access multiple serial ports. Finally, we controlled our robot using our own controller. In [Chapter 3](3bd53219-a287-4d8f-9a58-5a06c5b14062.xhtml), *Intel Edison and IoT (Home Automation)*,we have learned how to control the Edison using an Android app with the MQTT protocol. That technique can also be used to control a robot by using the `mraa` library.

The entire book has covered multiple topics related to Intel Edison. Now it's your job to use the concepts discussed to come up with new projects and explore even further. The last two chapters purely concentrated on robotics based on Intel Edison, but these concepts may be applied to other devices, such as an Arduino. Visit [https://software.intel.com/en-us/iot/hardware/edison/documentation](https://software.intel.com/en-us/iot/hardware/edison/documentation) for more details and more in-depth study about the hardware.