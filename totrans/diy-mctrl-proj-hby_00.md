# Preface

This book will introduce you to microcontroller technology. It focuses particularly on two very capable microcontroller boards, the Blue Pill and the Curiosity Nano, and how to connect sensors to them to solve problems and to support everyday life situations. In addition, this book covers the use of **light-emitting diodes** (**LEDs**) and **liquid-crystal displays** (**LCDs**) for showing sensor information to its microcontroller board users.

Microcontroller boards are practical small computers used for getting information from an environment using sensors. In this book, each chapter will focus on a specific problem to be solved with microcontroller technology, incorporating the use of practical sensors.

Many people from the intended audience would like to start with a microcontroller-based project but they may not know how to begin with it, including what kind of basic hardware and software tools and electronic components they will need to use. This book will cover that.

A chapter in this book introduces you to the field of electronics, examining and reviewing common electronic components that you will use in this book. Another chapter provides an introduction to C and C++, which will be used for coding Blue Pill and Curiosity Nano applications in most of the chapters.

One of the most important aspects of this book is that sensor programming via microcontroller boards is becoming more effective and easier than before because several easy coding libraries support them, which saves time and effort when getting either analog or digital data from them. This book explains common sensor-programming libraries.

# Who this book is for

This book is intended for students, hobbyists, geeks, and engineers alike who wish to dive into the world of microcontroller board programming. In addition, this book is suitable for digital electronics and microcontroller board beginners. If you are already a skilled electronics hobbyist and/or programmer, you may find this book helpful if you want to use and code efficient sensors with microcontroller boards.

People that use other types of microcontroller boards (such as Arduino boards) may find this book useful because it includes an introduction to the Blue Pill and Curiosity Nano microcontroller boards, facilitating the skills transfer required to understand and apply them in electronics projects requiring Arduino microcontroller boards.

Basic knowledge of digital circuits, and C and C++ programming language is desirable but not necessary. This is an introductory book on microcontroller boards for people who are starting with digital electronics projects.

# What this book covers

This book covers technical topics on the programming of the Blue Pill and Curiosity Nano microcontroller boards using C++, including descriptions of commonly used sensors and how they are electronically connected to the microcontroller boards. The book consists of 14 chapters, as follows:

[*Chapter 1*](B16413_01_Final_NM_ePub.xhtml#_idTextAnchor014), *Introduction to Microcontrollers and Microcontroller Boards,* introduces the reader to microcontroller technology and explains how to install the **integrated development environments** (**IDEs**) necessary for programming the Blue Pill and Curiosity Nano microcontroller boards that are used in the book.

[*Chapter 2*](B16413_02_Final_NM_ePub.xhtml#_idTextAnchor029), *Software Setup and C Programming for Microcontroller Boards,* provides an overview of C and an introduction to Blue Pill and Curiosity Nano microcontroller board programming, which are used for coding examples in most of the book chapters.

[*Chapter 3*](B16413_03_Final_NM_ePub.xhtml#_idTextAnchor041), *Turning an LED On or Off Using a Push Button,* explains how to use push buttons with microcontroller boards to start a process, such as turning an LED on or off, and how electrical noise from a push button can be minimized.

[*Chapter 4*](B16413_04_Final_NM_ePub.xhtml#_idTextAnchor053), *Measuring the Amount of Light with a Photoresistor,* focuses on how to connect a photoresistor to the Blue Pill and Curiosity Nano microcontroller boards to measure the amount of light within an environment. The result is shown on red, green, and blue LEDs also connected to those boards.

[*Chapter 5*](B16413_05_Final_NM_ePub.xhtml#_idTextAnchor069), *Humidity and Temperature Measurement,* describes how to connect a practical DHT11 sensor to measure the humidity and temperature of an environment, how to display its values on a computer, and also how to use the easy-to-use LM35 temperature sensor, showing its values on two LEDs.

[*Chapter 6*](B16413_06_Final_NM_ePub.xhtml#_idTextAnchor087), *Morse Code SOS Visual Alarm with a Bright LED,* shows how to code the Blue Pill and Curiosity Nano microcontroller boards to display a Morse code SOS signal using a high-intensity LED, increasing its visibility. This chapter also explains how to use a transistor as a switch to increase the LED's brightness.

[*Chapter 7*](B16413_07_Final_NM_ePub.xhtml#_idTextAnchor099)*, Creating a Clap Switch, describes to the reader how to make an electronic wireless control using sounds (claps). When two claps are detected by a microphone connected to a microcontroller board, a signal will be transmitted to activate a device connected to it and an LED will light up.*

[*Chapter 8*](B16413_08_Final_NM_ePub.xhtml#_idTextAnchor110)*, Gas Sensor**, introduces the reader to the use of a sensor connected to a microcontroller board that reacts with the presence of a specific gas in an environment.*

[*Chapter 9*](B16413_09_Final_NM_ePub.xhtml#_idTextAnchor122), *IoT Temperature-Logging System*, shows the reader how to build an **Internet of Things** (**IoT**) temperature logger using the Blue Pill microcontroller board and a temperature sensor. Its data will be transmitted via Wi-Fi using an ESP8266 module.

[*Chapter 10*](B16413_10_Final_NM_ePub.xhtml#_idTextAnchor135), *IoT Plant Pot Moisture Sensor*, explains how to build a digital device with a microcontroller board and a moisture sensor to monitor a plant pot's soil and determine if it needs water, sending an alert wirelessly to notify the user if it's too dry.

[*Chapter 11*](B16413_11_Final_NM_ePub.xhtml#_idTextAnchor145), *IoT Solar Energy (Voltage) Measurement,* continues applying IoT software running on a microcontroller board using the ESP8266 WiFi module to measure voltage obtained from a solar panel through a sensor. The application will send sensor data to the internet using the ESP8266 WiFi signal.

[*Chapter 12*](B16413_12_Final_NM_ePub.xhtml#_idTextAnchor157)*, COVID-19 Digital Body Temperature Measurement (Thermometer), looks at an interesting project to develop a contactless thermometer using an infrared temperature sensor. Its measured temperature data is sent through the I2C protocol to a Blue Pill microcontroller board, displaying it on an I2C LCD.*

[*Chapter 13*](B16413_13_Final_NM_ePub.xhtml#_idTextAnchor173)*, COVID-19 Social Distancing Alert,* explains how to program a microcontroller board that measures a distance of two meters between two or more people. Within the new normal of COVID-19, we need to maintain social distance due to the higher risk of catching the virus if you are close to someone who is infected. The World Health Organization recommends keeping a distance of at least two meters; this rule varies depending on the country, but it is generally accepted that a distance of two meters is safe.

[*Chapter 14*](B16413_14_Final_NM_ePub.xhtml#_idTextAnchor183)*, COVID-19 20-Second Hand Washing Timer*, contains a practical project to make a timer running on a Blue Pill microcontroller board that ensures that people wash their hands for twenty seconds, as per World Health Organization recommendations, to prevent COVID-19 infection. This project shows the time count on a **liquid-crystal display** (**LCD**). An ultrasonic sensor detects if the user is waving at it to initiate the count.

# To get the most out of this book

In order to use this book to the full, the reader will need basic knowledge of computer programming and the major operating systems (such as Windows or macOS), although there is a chapter that contains an introduction to C. In order to compile and run the programming examples described in this book, the reader should have the latest Arduino IDE previously installed on their computer (the Blue Pill board can be programmed using the Arduino IDE) and the MPLAB X IDE used for programming the Curiosity Nano microcontroller board; one of the chapters explains how to install and use them. All the program examples contained in this book for the Blue Pill microcontroller board should run on Windows, macOS, and Linux operating systems. The programs that run for the Curiosity Nano microcontroller board were tested on computers running both Windows and Linux operating systems.

**If you are using the digital version of this book, we advise you to type the code yourself or access the code via the GitHub repository (link available in the next section). Doing so will help you avoid any potential errors related to the copying and pasting of code.**

Some pre-requisites for this book include having basic knowledge of computer programming and electronics, and having some materials, such as a solderless breadboard, many DuPont wires, LEDs, and resistors.

After reading this book, you can continue experimenting with the sensors used in the chapters and perhaps programming and applying other sensors to be connected to microcontroller boards, since this book provides a solid foundation for microcontroller boards programming and use.

# Download the example code files

You can download the example code files for this book from GitHub at [https://github.com/PacktPublishing/DIY-Microcontroller-Projects-for-Hobbyists](https://github.com/PacktPublishing/DIY-Microcontroller-Projects-for-Hobbyists). In case there is an update to the code, it will be updated on the existing GitHub repository.

We also have other code bundles from our rich catalog of books and videos available at [https://github.com/PacktPublishing/](https://github.com/PacktPublishing/). Check them out!

# Code in Action

Code in Action videos for this book can be viewed at [https://bit.ly/3cZJHQ5](https://bit.ly/3cZJHQ5).

# Download the color images

We also provide a PDF file that has color images of the screenshots/diagrams used in this book. You can download it here: [https://static.packt-cdn.com/downloads/9781800564138_ColorImages.pdf](https://static.packt-cdn.com/downloads/9781800564138_ColorImages.pdf).

# Conventions used

There are a number of text conventions used throughout this book.

`Code in text`: Indicates code words in text, database table names, folder names, filenames, file extensions, pathnames, dummy URLs, user input, and Twitter handles. Here is an example: "Mount the downloaded `WebStorm-10*.dmg` disk image file as another disk in your system."

A block of code is set as follows:

```cpp
html, body, #map {
 height: 100%; 
 margin: 0;
 padding: 0
}
```

When we wish to draw your attention to a particular part of a code block, the relevant lines or items are set in bold:

```cpp
[default]
exten => s,1,Dial(Zap/1|30)
exten => s,2,Voicemail(u100)
exten => s,102,Voicemail(b100)
exten => i,1,Voicemail(s0)
```

Any command-line input or output is written as follows:

```cpp
$ mkdir css
$ cd css
```

Tips or important notes

Appear like this.

# Get in touch

Feedback from our readers is always welcome.

**General feedback**: If you have questions about any aspect of this book, mention the book title in the subject of your message and email us at [customercare@packtpub.com](mailto:customercare@packtpub.com).

**Errata**: Although we have taken every care to ensure the accuracy of our content, mistakes do happen. If you have found a mistake in this book, we would be grateful if you would report this to us. Please visit [www.packtpub.com/support/errata](http://www.packtpub.com/support/errata), selecting your book, clicking on the Errata Submission Form link, and entering the details.

**Piracy**: If you come across any illegal copies of our works in any form on the Internet, we would be grateful if you would provide us with the location address or website name. Please contact us at [copyright@packt.com](mailto:copyright@packt.com) with a link to the material.

**If you are interested in becoming an author**: If there is a topic that you have expertise in and you are interested in either writing or contributing to a book, please visit [authors.packtpub.com](http://authors.packtpub.com).

# Share Your Thoughts

Once you've read *DIY Microcontroller Projects for Hobbyists*, we'd love to hear your thoughts! Please [click here to go straight to the Amazon review page](https://packt.link/r/1-800-56413-9) for this book and share your feedback.

Your review is important to us and the tech community and will help us make sure we're delivering excellent quality content.