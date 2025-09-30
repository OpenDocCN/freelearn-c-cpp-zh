# Setting up Intel Edison

In every **Internet of Things** (**IoT**) or robotics project, we have a controller that is the brain of the entire system. Similarly, we have the Intel Edison. The Intel Edison computing module comes in two different packages: one is a mini-breakout board; the other is an Arduino-compatible board. One can use the board in its native state as well, but in that case the we have to fabricate our own expansion board. The Edison is basically the size of an SD card. Due to its tiny size, it's perfect for wearable devices. However, it's capabilities makes it suitable for IoT applications; and above all, the powerful processing capability, makes it suitable for robotics applications. However we don't simply use the device in this state. We hook up the board with an expansion board. The expansion board provides the user with enough flexibility and compatibility for interfacing with other units. The Edison has an operating system that runs the entire system. It runs a Linux image. So, to set up your device, you initially need to configure your device both at the hardware and at the software level.

In this chapter, we will be covering the following topics:

*   Setting up the Intel Edison
*   Setting up the developer environment
*   Running sample programs on the board using Arduino IDE, Intel XDK, and others
*   Interacting with the board by using our PC

# Initial hardware setup

We'll concentrate on the Edison package that comes with an Arduino expansion board. Initially, you will get two different pieces:

*   The Intel® Edison board
*   The Arduino expansion board

The following figure shows the architecture of the device:

![](img/image_01_001.png)

Architecture of Intel Edison. Picture Credits: [http://www.software.intel.com](http://www.software.intel.com/)

We need to hook these two pieces up in a single unit. Place the Edison board on top of the expansion board so that the GPIO interfaces meet at a single point. Gently push the Edison against the expansion board. You will hear a click. Use the screws that come with the package to tighten the setup. Once this is done, we'll now set up the device both at hardware level and software level to be used further. The following are the steps we'll cover in detail:

1.  Downloading the necessary software packages
2.  Connecting your Intel® Edison to your PC
3.  Flashing your device with the Linux image
4.  Connecting to a Wi-Fi network
5.  SSH-ing your Intel® Edison device

# Downloading the necessary software packages

To move forward with the development on this platform, we need to download and install a couple of software packages, which includes the drivers and the IDEs. The following is the list of the software along with the links that are required:

*   Intel® Platform Flash Tool Lite ([https://01.org/android-ia/downloads/intel-platform-flash-tool-lite](https://01.org/android-ia/downloads/intel-platform-flash-tool-lite))
*   PuTTY ([http://www.chiark.greenend.org.uk/~sgtatham/putty/download.html](http://www.chiark.greenend.org.uk/~sgtatham/putty/download.html))
*   Intel XDK for IoT ([https://software.intel.com/en-us/intel-xdk](https://software.intel.com/en-us/intel-xdk))
*   Arduino IDE ([https://www.arduino.cc/en/Main/Software](https://www.arduino.cc/en/Main/Software))
*   FileZilla FTP client ([https://filezilla-project.org/download.php](https://filezilla-project.org/download.php))
*   Notepad ++ or any other editor ([https://notepad-plus-plus.org/download/v7.3.html](https://notepad-plus-plus.org/download/v7.3.html))

# Drivers and miscellaneous downloads

Drivers and miscellaneous can be downloaded from:

*   Latest Yocto Poky image
*   Windows standalone driver for the Intel Edison
*   FTDI drivers ([http://www.ftdichip.com/Drivers/VCP.htm](http://www.ftdichip.com/Drivers/VCP.htm))

The first and the second packages can be downloaded from [https://software.intel.com/en-us/iot/hardware/edison/downloads](https://software.intel.com/en-us/iot/hardware/edison/downloads). [](https://software.intel.com/en-us/iot/hardware/edison/downloads) 

# Plugging in your device

After the software and drivers have all been installed, we'll connect the device to a PC. You need two Micro-B USB cables(s) to connect your device to the PC. You can also use a 9V power adapter and a single Micro-B USB cable, but for now we won't use the power adapter. The main use of the power adapter will come in a later section of this book, especially when we'll be interfacing with other devices that require USB.The following figure shows different sections of an Arduino expansion board of the Intel Edison:

![](img/image_01_002.png)

Different sections of an Arduino expansion board of Intel Edison

A small switch exists between the USB port and the OTG port. This switch must be towards the OTG port because we're going to power the device from the OTG port and not through the DC power port. Once it is connected to your PC, open your device manager and expand the ports section. If all the installations of the drivers were successful, then you'll see two ports:

*   Intel Edison virtual com port
*   USB serial port

# Flashing your device

Once your device is successfully detected and installed, you need to flash your device with the Linux image. For this we'll use the flash tool provided by Intel:

1.  Open the flash lite tool and connect your device to the PC:

![](img/image_01_003.png)

Intel phone flash lite tool

2.  Once the flash tool is opened, click on Browse... and browse to the `.zip` file of the Linux image you have downloaded.
3.  After you click on OK, the tool will automatically unzip the file.

4.  Next, click on Start to flash:

![](img/image_01_004.png)

Intel® Phone flash lite tool — stage 1

5.  You will be asked to disconnect and reconnect your device. Do this, and the board should start flashing. It may take some time before the flashing is completed. Don't tamper with the device during this process.
6.  Once the flashing is completed, we can configure the device:

![](img/image_01_005.png)

Intel® Phone flash lite tool — complete

# Configuring the device

After flashing successfully, we'll now configure the device. We're going to use the PuTTY console for the configuration. PuTTY is an SSH and telnet client, developed originally by Simon Tatham for the Windows platform. We're going to use the Serial section here.

Before opening the PuTTY console, open up the Device manager and note the port number for the USB serial port. This will be used in your PuTTY console:

![](img/image_01_006.png)

Ports for Intel® Edison in PuTTY

Next, select Serial on the PuTTY console and enter the port number. Use a baud rate of `115,200`. Press Open to open the window for communicating with the device:

![](img/image_01_007.png)

PuTTY console — login screen

Once you are in the PuTTY console, you can execute commands to configure your Edison. The following is the set of tasks we'll do in the console to configure the device:

1.  Provide a name for your device.
2.  Provide a root password (SSH your device).
3.  Connect your device to Wi-Fi.

Initially, when in the console, you will be asked to log in. Type in `root` and press *Enter*. You will see root@edison, which means that you are in the `root` directory:

![](img/image_01_008.png)

PuTTY console — login success

Now, we are in the Linux Terminal of the device. Firstly, we'll enter the following command for the setup:

[PRE0]

Press *Enter* after entering the command, and the entire configuration will be straightforward:

![](img/image_01_009.png)

PuTTY console — set password

Firstly, you will be asked to set a password. Type in a password and press *Enter*. You need to type in your password again for confirmation. Next, we'll set up a name for the device:

![](img/image_01_010.png)

PuTTY console — set name

Give a name for your device. Please note that this is not the login name for your device. It's just an alias for your device. Also the name should be atleast five characters long. Once you've entered the name, it will ask for confirmation: press *y* to confirm. Then it will ask you to set up Wi-Fi. Again select *y* to continue. It's not mandatory to set up Wi-Fi, but it's recommended. We need the Wi-Fi for file transfer, downloading packages, and so on:

![](img/image_01_011.png)

PuTTY console — set Wi-Fi

Once the scanning is completed, we'll get a list of available networks. Select the number corresponding to your network and press *Enter*. In this case, it's `5`, which corresponds to avirup171 which is my Wi-Fi. Enter the network credentials. After you do that, your device will be connected to Wi-Fi. You should get an IP address after your device is connected:

![](img/image_01_012.png)

PuTTY console — set Wi-Fi -2

After successful connection, you should get this screen. Make sure your PC is connected to the same network. Open up the browser in your PC, and enter the IP address shown in the console. You should get a screen similar to this:

![](img/image_01_013.png)

Wi-Fi setup — completed

Now, we've finished with the initial setup. However, the Wi-Fi setup normally doesn't happen in one go. Sometimes your device doesn't get connected to Wi-Fi and sometimes we cannot get the page shown previously. In those cases, you need to start `wpa_cli` to manually configure Wi-Fi.

Refer to the following link for the details:

[http://www.intel.com/content/www/us/en/support/boards-and-kits/000006202.html](http://www.intel.com/content/www/us/en/support/boards-and-kits/000006202.html)

With Wi-Fi setup completed, we can move forward to set up our developer environment. We'll cover the following programming languages and the respective IDEs:

*   Arduino processor language (C/C++)
*   Python
*   Node.js

# Arduino IDE

The Arduino IDE is a famous, and widely used, integrated developer environment that not only covers Arduino boards but also many other boards of Intel including Galileo, Edison, Node MCU, and so on. The language is based on C/C++. Once you download the Arduino IDE from the link mentioned at the beginning of this chapter, you may not receive the Edison board package. We need to manually download the package from the IDE itself. To do that, open up your Arduino IDE, and then go to Tools | Board: "Arduino/Genuino Uno" | Board Manager...:

![](img/image_01_014.png)

Arduino IDE

You now need to click on Boards Manager and select Intel i686 Boards. Click on the version number and then click on Install. Boards Manager is an extremely important component of the IDE. We use the Boards Manager to add external Arduino-compatible boards:

![](img/image_01_015.png)

Boards Manager

Once installed, you should see your board displayed under Tools Boards:

![](img/image_01_016.png)

Board installation successful

Once successfully installed, you will now be able to program the device using the IDE. Like every starter program, we'll also be burning a simple program into the Intel Edison which will blink the on-board LED at certain intervals set by us. Through this, the basic structure of the program using the Arduino IDE will also be clear. When we initially open the IDE, we get two functions:

*   `void setup()`
*   `void loop()`

The setup function is the place where we declare whether the pins are to be configured in output mode or input mode. We also start various other services, such as serial port communication, in the setup method. Depending on the usecase, the implementation changes. The loop method is that segment of the code that executes repeatedly in an infinite sequence. Our main logic goes in here. Now we need to blink an LED with an interval of 1 second:

[PRE1]

In the preceding code, the line `#define LED_PIN 13` is a macro for defining the LED pin. In the Arduino expansion board, an LED and a resistor is already attached to `pin 13`, so we do not need to attach any additional LEDs for now. In the setup function, we have defined the configuration of the pin as output using the pinMode function with two parameters. In the loop function, we have initially set the pin to high by using the `digitalWrite` function with two parameters, and then we've defined a delay of 1,000 miliseconds which is equivalent of 1 second. After the delay, we set the pin to low and then again define a delay of 1 second. The preceding code explains the basic structure of the Arduino code written in the Arduino IDE.

To burn this program to the Edison device, first compile the code using the compile button, then select the port number of your device, and finally click the Upload button to upload the code:

![](img/image_01_017.png)

Arduino IDE — blink

The port number can be selected under Tools | port.

Now that we know how to program using an Arduino, let's have a look at how it actually works or what's happening inside the Arduino IDE.

A number of steps actually happen while uploading the code:

1.  First, the Arduino environment performs some small transformations to make sure that the code is correct C or C++ (two common programming languages).
2.  It then gets passed to a compiler (`avr-gcc`), which turns the human readable code into machine readable instructions (or object files).
3.  Then, your code gets combined with (linked against), the standard Arduino libraries that provide basic functions such as `digitalWrite()` or `Serial.print()`. The result is a single Intel hex file, which contains the specific bytes that need to be written to the program memory of the chip on the Arduino board.
4.  This file is then uploaded to the board, transmitted over the USB or serial connection via the bootloader already on the chip or with external programming hardware.

# Python

Edison can also be programmed in Python. The code needs to be run on the device directly. We can either directly program the device, using any editor, such as the VI editor, or write the code in the PC first, and then transfer it using any FTP client, like FileZilla. Here we'll first write the code using Notepad++ and then transfer the script. Here also, we'll be executing a simple script which will blink the on-board LED. While dealing with Python and hardware, we need to use the MRAA library to interface with the GPIO pins. This is a low-level skeleton library for communication on GNU/Linux platforms. It supports almost all of the widely-used Linux-based boards. So, initially you need to install the library on the board.

Open up PuTTY and log in to your device. Once logged in, we'll add AlexT's unofficial `opkg` repository.

To do that, add the following lines to `/etc/opkg/base-feeds.conf` using the VI editor:

[PRE2]

Next, `update` the package manager and `install git` by executing the following commands:

[PRE3]

We'll clone Edison-scripts from GitHub to simplify certain things:

[PRE4]

Next we'll add `~/edison-scripts` to the path:

[PRE5]

We'll now run the following scripts to complete the process. Please note that the previous steps will not only configure the device for MRAA, but will also set up the environment for later projects in this book.

Firstly, run the following script. Just type:

[PRE6]

The previous package is the Python package manager. This will be used to install essential Python packages to be used in a later part of this book. Finally, we'll install Mraa by executing the following command:

[PRE7]

MRAA is a low-level skeleton library for communication on GNU/Linux platforms. `Libmraa` is a C/C++ library with bindings to Java, Python, and JavaScript to interface with the IO on Galileo, Edison, and other platforms. In simple words, it allows us to operate on the IO pins.

Once the preceding steps have completed, we are good to go with the code for Python. For that, open up any code editor, such as Notepad++, and type in the following code:

[PRE8]

Please save the preceding code as a `.py` extension such as `blink.py`, and now, we'll explain it line by line.

Initially, using the import statements, we import two libraries: MRAA and time. MRAA is required for interfacing with the GPIO pins:

[PRE9]

Here we initialize the LED pin and set it to the output mode:

[PRE10]

In the preceding block, we put our main logic in an infinite loop block. Now, we will transfer this to our device. To do that again, go to the PuTTY console and type `ifconfig`. Under the `wlan0` section, note down your IP address:

![](img/image_01_018.jpg)

IP address to be used

Now open up FileZilla and enter your credentials. Make sure your device and your PC are on the same network:

*   Host: The IP address you got according to the preceding screenshot: `192.168.0.101`
*   Username: `root` because you will be logging in to the root directory
*   Password: Your Edison password
*   Port: `22`

Once entered, you will get the folder structure of the device. We'll now transfer the Python code from our PC to the device. To do that, just locate your `.py` file in Windows Explorer and drag and drop the file in the FileZilla console's Edison's folder. For now, just paste the file under `root`. Once you do that and if it's a success, the file should be visible in your Edison device by accessing the PuTTY console and executing the `ls` command.

Another alternative is to locate your file on the left-hand side of FileZilla; once located, just right-click on the file and click Upload. The following is the typical screenshot of the FileZilla windows:

![](img/image_01_019.png)

FileZilla application

Once transferred and successfully listed using the `ls` command, we are going to run the script. To run the script, in the PuTTY console, go to your `root` directory and type in the following command:

[PRE11]

If the file is present, then you should get the LED blinking on your device. Congrats! You have successfully written a Python script on your Edison board.

# Intel XDK for IoT (Node.js)

Another IDE we will be covering is the powerful cross-platform development tool by Intel: Intel XDK. This will be used to run our Node.js scripts. Ideally we run our Node.js scripts from the XDK, but there is always an option to do the same by just transferring the `.js` file to your device using an FTP client such as FileZilla and use node `FileName.js` to run your script. From the list of downloaded software provided at the beginning of this chapter, download and install the XDK and open it. You may be required to sign in to the Intel developer zone. Once done, open your XDK. Then, under IoT embedded applications, select a Blank IoT Node.js Template:

![](img/image_01_020.jpg)

Screenshot for XDK

Once opened, replace all the existing code with the following code:

[PRE12]

If you have a close look at the code, then you may notice that the structure of the code remains more or less similar as that of the other two platforms. We initially import the `MRAA` library:

[PRE13]

We also display the version of `MRAA` installed (you can skip this step). The next task is to initialize and configure the pin to be in output or input mode:

[PRE14]

We use `ledState` to get the present state of the LED. Next, we define the logic in a separate function for blinking:

[PRE15]

Finally, we call the function. On close inspection of the code, it's evident that the we have used only one delay in milliseconds as we are checking the present state using the tertiary operator. In order to execute the code on the device, we need to connect our device first.

To connect your device to the XDK, go to the IoT Device section, and click on the dropdown. You may see your device in the dropdown. If you see it, then click on Connect:

![](img/image_01_021.jpg)

XDK screenshot — connection pane

If the device is not listed, then we need to add a manual connection. Click on Add Manual Connection, then add the credentials:

![](img/image_01_022.png)

Screenshot for manual connection

In the address, put in the IP which was used in FileZilla. In the Username, insert `root`, and the password is the password that was set before. Click on Connect and your device should be connected. Click on Upload to upload the program and Run to run the program:

![](img/image_01_023.jpg)

Screenshot for uploading and executing the code

After uploading, the LED that is attached to pin `13` should blink. Normally, when dealing with complex projects, we go for blank templates so that it's easy to customize and do the stuff we need.

For more examples and details on the XDK are available at: [https://software.intel.com/en-us/getting-started-with-xdk-and-iot](https://software.intel.com/en-us/getting-started-with-xdk-and-iot)

# Summary

In this chapter, we've covered the initial setup of the Intel Edison and configuring it to the network. We have also looked at how to transfer files to and from the Edison, and set up the developer environment for Arduino, Python, and Node.js. We did some sample programming, blinking an LED, using all three platforms. Through this, we've gained a fair knowledge of operating the Edison and developing simple to complex projects.

In [Chapter 2](bada9944-ec60-4e8f-8d88-0085dd1c8210.xhtml), *Weather Station (IoT)*, we'll build a mini-weather station and will be able to deploy a project on the Intel Edison.