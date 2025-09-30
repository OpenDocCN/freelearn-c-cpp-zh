# Chapter 5. Sensing with Digital Inputs

Arduino boards have inputs and outputs. Indeed, this is also one of the strengths of this platform: to directly provide headers connecting the ATMega chipset legs. We can then directly wire an input or output to any other external component or circuit without having to solder.

In case you need it here, I'm reminding you of some points:

*   Arduino has digital and analog inputs
*   Arduino has digital outputs that can also be used to mimic analog outputs

We are going to talk about digital inputs in this chapter.

We'll learn about the global concept of sensing the world. We are going to meet a new companion named **Processing** because it is a nice way to visualize and illustrate all that we are going to do in a more graphical way. It is also a pretext to show you this very powerful and open source tool. Then, it will drive us to design the first serial communication protocol between the board and a piece of software.

We'll specifically play with switches, but we will also cover some useful hardware design patterns.

# Sensing the world

In our over-connected world, a lot of systems don't even have sensors. We, humans, own a bunch of biological sensors directly in and over our body. We are able to feel temperature with our skin, light with our eyes, chemical components with both our nose and mouth, and air movement with ears. From a characteristic of our world, we are able to sense, integrate this feeling, and eventually to react.

If I go a bit further, I can remember a definition for senses from my early physiological courses at university (you remember, I was a biologist in one of my previous lives):

> "Senses are physiological capacities that provide data for perception"

This basic physiological model is a nice way to understand how we can work with an Arduino board to make it sense the world.

Indeed, it introduces three elements we need:

*   A capacity
*   Some data
*   A perception

## Sensors provide new capacities

A sensor is a physical converter, able to measure a physical quantity and to translate it into a signal understandable directly or indirectly by humans or machines.

A thermometer, for example, is a sensor. It is able to measure the local temperature and to translate it into a signal. Alcohol-based or Mercury-based thermometers provide a scale written on them and the contraction/dilatation of the chemical matter according to the temperature makes them easy to read.

In order to make our Arduino able to sense the world, temperature for instance, we would have to connect a sensor.

### Some types of sensors

We can find various types of types of sensors. We often think about environmental sensors when we use the term sensor.

I'll begin by quoting some environmental quantities:

*   Temperature
*   Humidity
*   Pressure
*   Gas sensors (gas-specific or not, smoke)
*   Electromagnetic fields
*   Anemometer (wind speed)
*   Light
*   Distance
*   Capacitance
*   Motion

This is a non-exhaustive list. For almost each quantity, we can find a sensor. Indeed, for each quantifiable physical or chemical phenomenon, there is a way to measure and track it. Each of them provides data related to the quantity measured.

## Quantity is converted to data

When we use sensors, the reason is that we need to have a numeric value coming from a physical phenomenon, such as temperature or movement. If we could directly measure the temperature with our skin's thermal sensors, we would have been able to understand the relationship between the volume of chemical components and temperature itself. Because we know this relationship from other physical measures or calculations, we have been able to design thermometers.

Indeed, thermometers are converting a quantity (here a volume) related to the temperature into a value readable on the scale of the thermometer. In fact, we have a double conversion here. The volume is a function depending on the temperature. The height of the liquid inside the thermometer is a function depending on the volume of the liquid. Thus, we can understand that the height and temperature are related. This is the double conversion.

Anyway, the thermometer is a nice module that integrates all this mathematical and physical wizardry to provide data, a value: the temperature. As shown in the following figure, volume is used to provide a temperature:

![Quantity is converted to data](img/7584_05_01.jpg)

All sensors work like that. They are modules measuring physical phenomenon and providing a value. We'll see later that those values can be very different, and eventually encoded too.

## Data has to be perceived

The data provided by a sensor makes more sense if it is read. This can be obvious but imagine that the reader isn't a human but is instead an instrument, a machine, or in our case, an Arduino board.

Indeed, let's take an electronic thermal sensor. At first, this one has to be supplied with electricity in order to work. Then, if we are able to supply it but unable to physically measure the electric potential generated by it from its pins, we couldn't appreciate the main value it tries to provide us: the temperature.

In our case, the Arduino would be the device that is able to convert the electric potential to something readable or at least easier to understand for us, humans. This is again a conversion. From the physical phenomenon that we want to translate, to the device displaying the value explaining the physical phenomenon, there are conversions and perceptions.

I can simplify the process as shown in the following figure:

![Data has to be perceived](img/7584_05_02.jpg)

# What does digital mean?

Let's define precisely what the digital term means here.

## Digital and analog concepts

Digital, in the computer and electronic worlds, means discrete, which is the opposite of analog/continuous. It is also a mathematical definition. We often talk about domains to define the cases for use of digital and analog.

Usually, the analog domain is the domain related to physical measures. Our temperature can have all the values that are possible and that exist, even if our measuring equipment dosen't have an infinite resolution.

The digital domain is the one of computers. Because of the encoding and finite memory size, computers translates analog/continuous values into digital representations.

On a graph, this could be visualized as follows:

![Digital and analog concepts](img/7584_05_03.jpg)

## Inputs and outputs of Arduino

Arduino owns inputs and outputs. We can also distinguish analog and digital pins.

You have to remember the following points:

*   Arduino provides digital pins that can be both an input or an output
*   Arduino provides only analog input, not output

Inputs and outputs are pins provided by the board to communicate with external peripherals.

### Note

Inputs provide the ability to feel the world.

Outputs provide the ability to alter the world.

We often talk about *reading pins* for inputs and *writing pins* for outputs. Indeed, from the Arduino board point of view, we are reading from the world and writing to the world, aren't we?

A digital input is a digital pin set up like an input and providing the capacity for electrical potential reading and conversion to 0 or 1 to the Arduino board. We'll illustrate this very soon using switches.

But before manipulating this directly, let me introduce a new friend named **Processing**. We'll use it to easily illustrate our Arduino tests further in the book.

# Introducing a new friend – Processing

Processing is an open source programming language and Integrated Development Environment (IDE) for people who want to create images, animations, and interaction.

This major open source project was initiated in 2001 by Ben Fry and Casey Reas, two gurus and former students of John Maeda at the Aesthetics and Computation Group at the MIT Media Lab.

It is a programming framework most used by non-programmers. Indeed, it has been designed primarily for this purpose. One of the first targets of Processing is to provide an easy way of programming for non-programmers through the instant gratification of visual feedback. Indeed, as we know, programming can be very abstract. Processing natively provides a canvas on which we can draw, write, and do more. It also provides a very user-friendly IDE that we are going to see on the official website at [http://processing.org](http://processing.org).

You'll probably also find the term Processing written as **Proce55ing** as the domain name `processing.org` was already taken at the time of its inception.

## Is Processing a language?

Processing isn't a language in the strictest sense. It's a subset of Java with some external libraries and a custom IDE.

Programming with Processing is usually performed using the native IDE comes with the download as we will see in this section.

Processing uses the Java language but provides simplified syntax and graphics programming. It also simplifies all compilations steps into a one-click action like Arduino IDE.

Like Arduino core, it provides a huge set of ready-to-use functions. You can find all references at [http://processing.org/reference](http://processing.org/reference).

There is now more than one way to use Processing. Indeed, because JavaScript runtimes integrated in web browsers became more and more powerful, we can use a JavaScript derived project. You still continue to code using Java, you include this code in your webpage, and as the official website says "*Processing.js does the rest. It's not magic, but almost*." The website is [http://processingjs.org](http://processingjs.org).

There is also something very interesting: You can package applications coded using Processing for Android mobile OS. You can read this if you are interested at [http://processing.org/learning/android](http://processing.org/learning/android).

I will avoid going on a tangent with the JS and Android applications, but I felt it was important enough to mention these usages.

## Let's install and launch it

Like the Arduino framework, the Processing framework doesn't include installation program. You just have to put it somewhere and run it from there.

The download URL is: [http://processing.org/download](http://processing.org/download).

First, download the package corresponding to your OS. Please refer to the website for the install process for your specific OS.

On OS X, you have to deflate the zip file and run the resulting file with the icon:

![Let's install and launch it](img/7584_05_004.jpg)

Processing icon

Double-click on the icon, and you'll see a pretty nice splash screen:

![Let's install and launch it](img/7584_05_005.jpg)

Then you'll see the Processing IDE as shown in the following image:

![Let's install and launch it](img/7584_05_006.jpg)

Processing's IDE looks like others

## A very familiar IDE

Indeed, the Processing IDE looks like the Arduino one. The Processing IDE is like the father of the Arduino IDE.

This is totally normal because the Arduino IDE has been forked from the Processing IDE. Now, we are going to check that we'll be very comfortable with the Processing IDE as well.

Let's explore it and run a small example:

1.  Go to **Files** | **Examples Basics** | **Arrays** | **ArraysObjects**.
2.  Then, click on the first icon (the play symbol arrow). You should see the following screenshot:![A very familiar IDE](img/7584_05_007.jpg)

    Running ArrayObjects native example in Processing

3.  Now click on the small square (stop symbol). Yes, this new playground is very familiar.![A very familiar IDE](img/7584_05_008.jpg)

    Processing IDE with ArrayObjects example opened

At the top we can see some familiar icons.

From left to right, they are as follows:

*   **Run** (small arrow): This is used to compile and run your program
*   **Stop** (small square): This is used to stop the program when it is running
*   **New project** (small page): This is used to open a blank canvas
*   **Open project** (top arrow): This is used to open an existing project
*   **Save project** (down arrow): This is used to save a project
*   **Export application** (right arrow): This is used to create an application

No Upload button of course. There is no need to upload anything here; we are on the computer and we only want to code applications, compile them, and run them.

With Processing, you have everything in hand to code, compile, and run.

You can have some tabs if you use more than one file in your project (especially if you use some separate Java classes).

Under this tab zone, you have the text area where you type your code. Code is colored as in the Arduino IDE, and this is very useful.

At last, at the bottom, you have the log console area where all the messages can be output, from errors to our own tracer messages.

### Alternative IDEs and versioning

If you are interested in digging some IDE alternatives, I'd suggest that you use the universal and open source software development environment Eclipse. I suggest that to all the students I meet who want to go further in pure-development fields. This powerful IDE can be easily set up to support versioning.

Versioning is a very nice concept providing an easy way to track versions of your code. You can, for instance, code something, test it, back it up in your versioning system, then continue your code design. If you run it and have a nice and cute crash at some point, you can easily check the differences between your working code and the new non working one and make your troubleshooting much easier! I won't describe versioning systems in detail, but I want to introduce you to the two main systems that are widely used a

*   [http://subversion.apache.org](http://subversion.apache.org)
*   **Git**: [http://git-scm.com](http://git-scm.com)

## Checking an example

Here is a small piece of code showing some cheap and easy design patterns. You can also find this code in the folder `Chapter05` `/p` `rocessingMultipleEasing/` in the code bundle:

[PRE0]

You can run this piece of code. Then, you can move the mouse into the canvas and enjoy what is happening.

![Checking an example](img/7584_05_010.jpg)

processingMultipleEasing code running and showing a strange series of particles following the mouse

First, check the code. Basically, this is Java. I guess you aren't shocked too much, are you? Indeed, Java derives from C.

You can see three main parts in your code:

*   Variable declarations/definitions
*   The `setup()`function that runs only once at the beginning
*   The `draw()` function that runs infinitely until you press stop

Ok. You can see the `setup()` functions in the Arduino core and Processing have similar roles, and `loop()` and `draw()` too.

This piece of code shows some usual design patterns with Processing. I first initiate a variable storing the global number of particles, then I initiate some arrays for each particle I want to create. Please notice all these arrays are empty at this step!

This pattern is usual because it offers good readability and works fine too. I could have used classes or even multidimensional arrays, but in this latter case, I would not even have benefits except a shorter (but less readable) code. In all those arrays, the *N*th indexed value represents the particle *N*. In order to store/retrieve the parameters of particle *N*, I have to manipulate the *N*th value for each array. The parameters are spread inside each array but are easy to store and retrieve, aren't they?

In `setup()`, I define and instantiate the canvas and its size of 600 x 600\. Then, I'm defining that there will be no stroke in any of my drawings. The stroke of a circle, for instance, is its border.

Then, I'm filling the `easing` and `radii` arrays using a `for` loop structure. This is a very usual pattern where we can use `setup()` to initialize a bunch of parameters at the beginning. Then we can check the `draw()` loop. I'm defining a color for the background. This function also erases the canvas and fills it with the color in argument. Check the background function on the reference page to understand how we can use it. This erase/fill is a nice way to erase each frame and to reset the canvas.

After this erase/fill, I'm storing the current position of the mouse for each coordinate in the local variables `targetX` and `targetY`.

The core of the program sits in the `for` loop. This loop walks over each particle and makes something for each of them. The code is quite self-explanatory. I can add here that I'm checking the distance between the mouse and each particle for each frame (each run of `draw()`), and I draw each particle by moving them a bit, according to its easing.

This is a very simple example but a nice one I used to show to illustrate the power of Processing.

## Processing and Arduino

Processing and Arduino are very good friends.

Firstly, they are both open source. It is a very friendly characteristic bringing a lot of advantages like code source sharing and gigantic communities, among others. They are available for all OSes: Windows, OS X, and Linux. We also can download them for free and run them in a couple of clicks.

I began to program primarily with Processing, and I use it a lot for some of my own data visualization projects and art too. Then, we can illustrate complex and abstract data flows by smooth and primitive shapes on a screen.

What we are going to do together now is display Arduino activity on the Processing canvas. Indeed, this is a common use of Processing as an eye-friendly software for Arduino.

We are going to design a very trivial and cheap protocol of communication between the hardware and the software. This will show you the path that we'll dig further in the next chapters of this book. Indeed, if you want to get your Arduino talking with another software framework (I'm thinking about Max 6, openFrameworks, Cinder, and many others), you'll have to follow the same ways of design.

![Processing and Arduino](img/7584_05_011.jpg)

Arduino and some software friends

I often say Arduino can work as a very smart *organ* of software. If you want to connect some software to the real, physical world, Arduino is the way to go. Indeed, that way, software can sense the world, providing your computer with new features. Let's move on by displaying some physical world events on the computer.

# Pushing the button

We are going to have fun. Yes, this is the very special moment when we are going to link the physical world to the virtual world. Arduino is all about this.

## What is a button, a switch?

A **switch** is an electrical component that is able to break an electrical circuit. There are a lot of different types of switches.

### Different types of switches

Some switches are called **toggles**. Toggles are also named continuous switches. In order to act on the circuit, the toggle can be pushed and released each time you want to act and when you release it, the action continues.

Some others are called **momentaries**. Momentaries are named **push for action** too. In order to act on the circuit, you have to push and keep the switch pushed to continue the action. If you release it, the action stops.

Usually, all our switches at home are toggles. Except the one for the mixer that you have to push to cut and release to stop it, which means it is a momentary.

## A basic circuit

Here is a basic circuit with an Arduino, a momentary switch and a resistor.

We want to turn the board's built-in LED ON when we push the momentary switch and turn it OFF when we release it.

![A basic circuit](img/7584_05_12.jpg)

A small circuit

I'm presenting you with the circuit on which we are going to work right now. This is also a nice pretext to make you more familiar with circuit diagrams.

### Wires

Each line represents a link between two components. By definition, a line is a wire and there is no electrical potential from one side to the other. It can also be defined as follows: a wire has a resistance of 0 ohm. Then we can say that two points linked by a wire have the same electrical potential.

### The circuit in the real world

Of course, I didn't want to show you the next diagram directly. Now we have to build the real circuit, so please take some wires, your breadboard, and the momentary switch, and wire the whole circuit as shown in the next diagram.

You can take a resistor around 10 Kohms. We'll explain the purpose of the resistor in the next pages.

![The circuit in the real world](img/7584_05_014.jpg)

The momentary switch in a real circuit

Let's explain things a bit more.

Let's remember the breadboard wiring; I'm using cold and hot rails at the top of the breadboard (cold is blue and means ground, hot is red and means +5 V). After I have wired the ground and +5 V from the Arduino to the rails, I'm using rails to wire the other parts of the board; it is easier and requires shorter cables.

There is a resistor between the ground and the digital pin 2\. There is a momentary switch between the +5 V line and the pin 2 as well. The pin 2 will be set up as an input, which means it will be able to sink current.

Usually, switches are *push-to-on*. Pushing them closes the circuit and lets the current flow. So, in that case, if I don't push the switch, there is no current from +5 V to the pin 2.

For the duration it is pressed, the circuit is closed. Then, current flows from the +5 V to the pin 2\. It is a bit metaphoric and abusive, and I should say we have created an electrical potential between +5 V and the pin 2, but I need to be shorter to hit the point home.

And this resistor, why is it here?

## The pull-up and pull-down concept

If the global circuit is easy, the resistor part can be a bit tricky at first sight.

A digital pin set up as an input provides the ability to *sink* current. This means it behaves like the ground. Indeed, and in fact, internally, it works exactly as if the concerned pin was connected to the ground.

With a properly coded firmware, we would have the ability to check pin 2\. This means we could test it and read the value of the electrical potential. Because it is a digital input, an electrical potential near +5 V would be translated as the value HIGH, and if it is near 0 V, it will be translated as the value LOW. Both values are constants defined inside the Arduino core. But if everything seems totally perfect in a perfect digital world, it is not true.

Indeed, the input signal noise could potentially be read as a button press.

To be sure and safe, we use what we call a *pull-down resistor*. This is usually a high impedance resistance that provides a current sink to the digital pin considered, making it safer at the value 0 V if the switch is not pressed. Pull down to be more consistently recognized as a LOW value, pull up to be more consistently recognized as the HIGH value.

Of course, the global energy consumption increases a bit. In our case, this is not important here but you have to know that. On this same concept, a pull-up resistor can be used to link the +5 V to the digital output. Generally, you should know that a chipset's I/O shouldn't be floating.

Here is what you have to remember:

| Type of Digital Pin | Input | Output |
| --- | --- | --- |
| Pull Resistor | Pull-down resistor | Pull-up resistor |

We want to push a switch, and particularly, this action has to turn the LED ON. We are going to write a pseudocode first.

### The pseudocode

Here is a possible pseudocode. Following are the steps we want our firmware to follow:

1.  Define the pins.
2.  Define a variable for the current switch state.
3.  Set up the LED pin as an output.
4.  Set up the switch pin as an input.
5.  Set up an infinite loop. In the infinite loop do the following:

    1.  Read the input state and store it.
    2.  If the input state is HIGH, turn the LED ON.
    3.  Else turn the LED OFF.

### The code

Here is a translation of this pseudocode in valid C code:

[PRE1]

As usual you can also find the code in the `Chapter05/MonoSwitch/` folder available for download along with other code files on Packt Publishing's site.

Upload it and see what happens. You should have a nice system on which you can push a switch and turn on an LED. Splendid!

Now let's make the Arduino board and Processing communicate with each other.

## Making Arduino and Processing talk

Let's say we want to visualize our switch's manipulations on the computer.

We have to define a small communication protocol between Arduino and Processing. Of course, we'll use a serial communication protocol because it is quite easy to set it up and it is light.

We could design a protocol as a library of communication. We only design a protocol using the native Arduino core at the moment. Then, later in this book, we will design a library.

### The communication protocol

A communication protocol is a system of rules and formats designed for exchanging messages between two entities. Those entities can be humans, computers and maybe more.

Indeed, I'd use a basic analogy with our language. In order to understand each other, we have to follow some rules:

*   Syntactic and grammatical rules (I have to use words that you know)
*   Physical rules (I have to talk loud enough)
*   Social rules (I shouldn't insult you just before asking you for the time)

I could quote many other rules like the speed of talking, the distance between the two entities, and so on. If each rule is agreed upon and verified, we can talk together. Before designing a protocol, we have to define our requirements.

#### Protocol requirements

What do we want to do?

We need a communication protocol between our Arduino and Processing inside the computer. Right! These requirements are usually the same for a lot of communication protocols you'll design.

Here is a short list of very important ones:

*   The protocol must be expandable without having to rewrite everything each time I want to add new message types
*   The protocol must be able to send enough data quite quickly
*   The protocol must be easy to understand and well commented, especially for open source and collaborative projects

#### Protocol design

Each message will be 2 bytes in size. This is a common data packet size and I propose to organize data like this:

*   **Byte 1**: switch number
*   **Byte 2**: switch state

The fact that I defined byte 1 as a representation of the switch number is typically because of the requirement of expandability. With one switch, the number will be 0.

I can easily instantiate serial communication between the board and the computer. Indeed, we already made that when we used Serial Monitoring at least on the Arduino side.

How can we do that using Processing?

### The Processing code

Processing comes with very useful set of libraries already integrated into its core. Specifically, we are going to use the serial library.

Let's sketch a pseudocode first, as usual.

#### Sketching a pseudocode

What do we want the program to do?

I propose to have a big circle. Its color will represent the switch's state. *Dark* will mean released, and *green* will mean pushed.

The pseudocode can be created as follows:

1.  Define and instantiate the serial port.
2.  Define a current drawing color to dark.
3.  In the infinite loop, do the following:

    1.  Check if the serial port and grab data have been received.
    2.  If data indicates that state is off, change current drawing from color to dark.
    3.  Else, change current drawing color to green.
    4.  Draw the circle with the current drawing color.

#### Let's write that code

Let's open a new processing canvas.

Because the Processing IDE works like the Arduino IDE and needs to create all saved project files in a folder, I'd suggest that you directly save the canvas, even empty, in the right place on your disk. Call it `processingOneButtonDisplay`.

You can find the code in the `Chapter05/processingOneButtonDisplay/` folder available for download along with other code files on Packt's site.

![Let's write that code](img/7584_05_015.jpg)

Making a library inclusion in your code

To include the serial library from the Processing core, you can go to **Sketch | Import Library… | serial**. It adds this row to your code: `processing.serial.*;`

You could also type this statement by yourself.

Following is the code, with a lot of comments:

[PRE2]

#### Variable definitions

`theSerialPort` is an object of the `Serial` library. I have to create it first.

`serialBytesArray` is an array of two integers used to store messages coming from Arduino. Do you remember? When we designed the protocol, we talked about 2 byte messages.

`switchState` and `switchID` are global but temporary variables used to store the switch state and the switch ID corresponding to the message coming from the board. Switch ID has been put there for (close) future implementation to distinguish the different switches in case we use more than one.

`bytesCount` is a useful variable tracking the current position in our message reading.

`init` is defined to `false` at the beginning and becomes `true` when the first byte from the Arduino (and a special one, `Z`) has been received for the first time. It is a kind of first-contact purpose.

Then, we keep a trace of the fill color and the initial one is `40`. `40` is only an integer and will be used a bit further as an argument of the function `fill()`.

#### setup()

We define the canvas (size, background color, and no stroke).

We print a list of all the serial ports available on your computer. This is debug information for the next statement where we store the name of the first serial port into a String. Indeed, you could be led to change the array element from 0 to the correct one according to the position of your Arduino's port in the printed list.

This String is then used in the very important statement that instantiates serial communication at 9600 bauds.

This `setup()` function, of course, runs only once.

#### draw()

The draw function is very light here.

We pass the variable `fillColor` to the `fill()` function, setting up the color with which all further shapes will be filled.

Then, we draw the circle with the ellipse function. This function takes four arguments:

*   x coordinates of the center of the ellipse (here `width/2`)
*   y coordinates of the center of the ellipse (here `height/2`)
*   Width of the ellipse (here `230`)
*   Height of the ellipse (here `230`)

`width` and `height` colored in blue in the Processing IDE are the current width and height of the canvas. It is very useful to use them because if you change the `setup()` statement by choosing a new size for the canvas, all `width` and `height` in your code will be updated automatically without needing to change them all manually.

Please keep in mind that an ellipse with same values for `width` and `height` is a circle (!). Ok. But where is the magic here? It will only draw a circle, every time the same one (size and position). `fillColor` is the only variable of the `draw()` function. Let's see that strange callback named `serialEvent()`.

#### The serialEvent() callback

We talked about callbacks in [Chapter 4](ch04.html "Chapter 4. Improve Programming with Functions, Math, and Timing"), *Improve Programming with Functions, Math, and Timing*.

Here, we have a pure callback method in Processing. This is an event-driven callback. It is useful and efficient not to have to poll every time our serial port wants to know if there is something to read. Indeed, user interfaces related events are totally less numerous than the number of Arduino board's processor cycles. It is smarter to implement a callback in that case; as soon as a serial event occurs (that is, a message is received), we execute a series of statements.

`myPort.read()` will first read the bytes received. Then we make the test with the `init` variable. Indeed, if this is the very first message, we want to check if the communication has already begun.

In the case where it is the first hello (`init == false`), if the message coming from the Arduino Board is `Z`, Processing program clear its own serial port, stores the fact the communication has just started, and resends back `Z` to the Arduino board. It is not so tricky.

It can be illustrated as follows:

Imagine we can begin to talk only if we begin by saying "hello" to each other. We aren't watching each other (no event). Then I begin to talk. You turn your head to me (serial event occurs) and listen. Am I saying "hello" to you? (whether the message is `Z`?). If I'm not, you just turn your head back (no `else` statement). If I am, you answer "hello" (sending back `Z`) and the communication begins.

What happens then?

If communication has already begun, we have to store bytes read into the `serialBytesArray` and increment the `bytesCount`. While bytes are being received and `bytesCount` is smaller or equal to 1, this means we don't have a complete message (a complete message is two bytes) and we store more bytes in the array.

As soon as the bytes count equals `2`, we have a complete message and we can "split" it into the variables `switchID` and `switchState`. Here's how we do that:

[PRE3]

This next statement is a debug one: we print each variable. Then, the core of the method is the test of the `switchState` variable. If it is `0`, it means the switch is released, and we modify the `fillColor` to `40` (dark color, `40` means the value 40 for each RGB component; check `color()` method in Processing reference at [http://processing.org/reference/color_.html](http://processing.org/reference/color_.html)). If it isn't `0`, we modify the `fillColor` to `255`, which means white. We could be a bit safer by not using only `else`, but `else if (switchState ==1)` also.

Why? Because if we are not sure about all the messages that can be sent (lack of documentation or whatever else making us unsure), we can modify the color to white *only* if `switchState` equals `1`. This concept can be done at the optimization state too, but here, it is quite light so we can leave it like that.

Ok. It is a nice, heavy piece, right? Now, let's see how we have to modify the Arduino code. Do you remember? It isn't communication ready yet.

### The new Arduino firmware talk-ready

Because we now have a nice way to display our switch state, I'll remove all things related to the built-in LED of the board and following is the result:

[PRE4]

What do we have to add? All the `Serial` stuff. I also want to add a small function dedicated to the first "hello".

Here is the result, then we will see the explanations:

[PRE5]

I'm defining one new variable first: `inByte`. This stores the bytes read. Then inside the `setup()` method, I'm instantiating serial communication as we already learned to do with Arduino. I'm setting up the `pinMode` method of the switch pin then, I'm calling `sayHello()`.

This function just waits for something. Please focus on this.

I'm calling this function in `setup()`. This is a *simple* call, not a callback or whatever else. This function contains a `while` loop while `Serial.available()` is smaller or equal to zero. What does this mean? It means this function pauses the `setup()` method while the first byte comes to the serial port of the Arduino board. The `loop()` done doesn't run while the `setup()` done has finished, so this is a nice trick to wait for the first external event; in this case, the first communication. Indeed, the board is sending the message `Z` (that is, the "hello") while Processing doesn't answer.

The consequence is that when you can plug in your board, it sends `Z` continuously while you run your Processing program. Then the communication begins and you can push the switch and see what is happening. Indeed, as soon as the communication begins, `loop()` begins its infinite loop. At first a test is made at each cycle and we only test if a byte is being received. Whatever the byte received (Processing only sends `Z` to the board), we read the digital pin of the switch and send back two bytes. Here too, pay attention please: each byte is written to the serial port using `Serial.write()`. You have to send 2 bytes, so you stack two `Serial.write()`. The first byte is the number (ID) of the switch that is pushed/released; here, it is not a variable because we have one and only one switch, so it is an integer 0\. The second byte is the switch state. We just saw here a nice design pattern involving the board, an external program running on a computer and a communication between both of them.

Now, let's go further and play with more than one switch.

# Playing with multiple buttons

We can extrapolate our previously designed logic with more than one switch.

There are many ways to use multiple switches, and, in general, multiple inputs on the Arduino. We're going to see a cheap and easy first way right now. This way doesn't involve multiplexing a lot of inputs on only a couple of Arduino inputs but a basic one to one wiring where each switch is wired to one input. We'll learn multiplexing a bit later (in the next chapter).

## The circuit

Following is the circuit diagram required to work with multiple switches:

![The circuit](img/7584_05_16.jpg)

Wiring three momentary switches to the Arduino board

The schematic is an extrapolation of the previous one that showed only one switch. We can see the three switches between the +5 V and the three pull-down resistors. Then we can also see the three wires going to digital inputs 2 to 4 again.

Here is a small memory refresh: Why didn't I use the digital pins 0 or 1?

Because I'm using serial communication from the Arduino, we cannot use the digital pins 0 and 1 (each one respectively corresponding to RX and TX used in serial communication). Even if we are using the USB link as the physical support for our serial messages, the Arduino board is designed like that and we have to be very careful with it.

Here is the circuit view with the breadboard. I voluntarily didn't align every wire. Why? Don't you remember that I want you to be totally autonomous after reading this book and yes, you'll find many schematics in the real world made sometimes like that. You have to become familiar with them too. It could be an (easy) homework assignment.

![The circuit](img/7584_05_017.jpg)

The preceding circuit shows the three switches, the three pull-down resistors, and the Arduino board.

Both source codes have to be modified to provide a support for the new circuit.

Let's add things there.

## The Arduino code

Here is the new code; of course, you can find it in the `Chapter05/MultipleSwitchesWithProcessing/` folder available for download along with other code files on Packt's site:

[PRE6]

Let's explain this code.

At first, I defined a constant `switchesNumber` to the number `3`. This number can be changed to any other number from `1` to `12`. This number represents the current number of switches wired to the board from digital pin 2 to digital pin 14\. All switches have to be linked without an empty pin between them.

Then, I defined an array to store the switch's states. I declared it using the `switchesNumber` constant as the length. I have to fill this array with zeroes in the `setup()` method, that I made with a `for` loop. It provides a safe way to be sure that all switches have a release state in the code.

I still use the `sayHello()` function, to set up the communication start with Processing.

Indeed, I have to fill each switch state in the array `switchesStates` so I added the `for` loop. Please notice the index trick in each `for` loop. Indeed, because it seems to be more convenient to start from 0 and because in the real world we mustn't use digital pins 0 and 1 while using serial communications, I added `2` as soon as I dealt with the real number of the digital pin, that is, with the two functions `pinMode()` and `digitalRead()`.

Now, let's upgrade the Processing code too.

## The Processing code

Here is the new code; you can find it in the `Chapter05/MultipleSwitchesWithProcessing/` folder available for download along with other code files on Packt's site:

[PRE7]

Following is a screenshot of the render of this code used with five switches while I was pushing on the fourth button:

![The Processing code](img/7584_05_020.jpg)

So, what did I alter?

Following the same concept as with the Arduino code, I added a variable (not a constant here), named `switchesNumber`. A nice evolution could be to add something to the protocol about the number of the switch. For instance, the Arduino board could inform Processing about the switch's number according to only one constant defined in the Arduino firmware. This would save the manual update of the processing code when we change this number.

I also transformed the variable `switchState` into an array of integers `switchesStates`. This one stores all the switches' states. I added two variables related to the display: `distanceCircles` and `radii`. Those are used for dynamically displaying the position of circles according to the number of switches. Indeed, we want one circle per switch.

The `setup()` function is almost the same as before.

I'm calculating here the distance between two circles by dividing the width of the canvas by the number of circles. Then, I'm calculating the radii of each circle by using the distance between them divided by 2\. These numbers can be changed. You could have a very different aesthetical choice.

Then the big difference here is also the `for` loop. I'm filling the whole `switchesStates` array with zeroes to initialize it. At the beginning, none of the switches are pushed. The `draw()` function now also includes a `for` loop. Pay attention here. I removed the `fillColor` method because I moved the fill color choice to the draw. This is an alternative, showing you the flexibility of the code.

In the same for loop, I'm drawing the circle number *i*. I will let you check for yourself how I have placed the circles. The `serialEvent()` method doesn't change a lot either. I removed the fill color change as I wrote before. I also used the `switchesStates` array, and the index provided by the first byte of the message that I stored in `switchID`.

Now, you can run the code on each side after you have uploaded the firmware on the Arduino board.

Magic? I guess you now know that it isn't magic at all, but beautiful, maybe.

Let's go a bit further talking about something important about switches, but also related to other switches.

# Understanding the debounce concept

Now here is a small section that is quite cool and light compared to analog inputs, which we will dive into in the next chapter.

We are going to talk about something that happens when someone pushes a button.

## What? Who is bouncing?

Now, we have to take our microscopic biocybernetic eyes to zoom into the switch's structure.

A switch is made with pieces of metal and plastic. When you push the cap, a piece of metal moves and comes into contact with another piece of metal, closing the circuit. Microscopically and during a very small time interval, things aren't that clean. Indeed, the moving piece of metal bounces against the other part. With an oscilloscope measuring the electrical potential at the digital pin of the Arduino, we can see some noise in the voltage curve around 1 ms after the push.

These oscillations could generate incorrect inputs in some programs. Imagine, that you want to count the states transitions in order, for instance, to run something when the user pushed the switch seven times. If you have a bouncing system, by pushing only once, the program could count a lot of transitions even if the user pushed the switch only once.

Check the next graph. It represents the voltage in relation to time. The small arrows on the time axis show the moment when the switch has been pushed:

![What? Who is bouncing?](img/7584_05_18.jpg)

How can we deal with these oscillations?

## How to debounce

We have two distinct elements on which we can act:

*   The circuit itself
*   The firmware

The circuit itself can be modified. I could quote some solutions such as adding diodes, capacitors, and some Schmitt trigger inverters. I won't explain that solution in detail because we are going to do that in software, but I can explain the global concept. The capacitor in that case will be charged and discharged while the switch will be bouncing, smoothing those peaks of noise. Of course, some tests are needed in order to find the perfect components fitting your precise needs.

The firmware can also be modified.

Basically, we can use a time-based filter, because the bounce occurs during a particular amount of time.

Following is the code, then will come explanations:

[PRE8]

Following is an example of the debouncing cycle.

At the beginning, I defined some variables:

*   `lastSwitchState`: This stores the last read state
*   `lastDebounceTime`: This stores the moment when the last debounce occurred
*   `debounceDelay`: This is the value during which nothing is taken as a safe value

We are using `millis()` here in order to measure the time. We already talked about this time function in [Chapter 4](ch04.html "Chapter 4. Improve Programming with Functions, Math, and Timing"), *Improve Programming with Functions, Math, and Timing*.

Then, at each `loop()` cycle, I read the input but basically I don't store it in the `switchState` variable that is used to the test to turning ON or OFF the LED. Basically, I used to say that `switchState` is the official variable that I don't want to modify before the debounce process. Using other terms, I can say that I'm storing something in `switchState` only when I'm sure about the state, not before.

So I read the input at each cycle and I store it in `readInput`. I compare `readInput` to the `lastSwitchState` variable that is the last read value. If both variables are different, what does it mean? It means a change occurs, but it can be a bounce (unwanted event) or a real push. Anyway, in that case, we reset the counter by putting the current time provided by `millis()` to `lastDebounceTime`.

Then, we check if the time since the last debounce is greater than our delay. If it is, then we can consider the last `readInput` in this cycle as the real switch state and we can store it into the corresponding variable. In the other case, we store the last read value into `lastSwitchState` to keep it for the next cycle comparison.

This method is a general concept used to smooth inputs.

We can find here and there some examples of software debouncing used not only for switches but also for noisy inputs. In everything related to a user-driven event, I would advise using this kind of debouncer. But for everything related to system communication, debounce can be very useless and even a problem, because we can ignore some important messages and data. Why? Because a communication system is much faster than any user, and if we can use 50 ms as the time during which nothing is considered as a real push or a real release with users, we cannot do that for very fast chipset signals and other events that could occurs between systems themselves.

# Summary

We have learnt a bit more about digital inputs. Digital inputs can be used *directly*, as we just did, or also *indirectly*. I'm using this term because indeed, we can use other peripherals for encoding data before sending them to digital inputs. I used some distance sensors that worked like that, using digital inputs and not analog inputs. They encoded distance and popped it out using the I2C protocol. Some specific operations were required to extract and use the distance. In this way, we are making an indirect use of digital inputs.

Another nice way to sense the world is the use of analog inputs. Indeed, this opens a new world of continuous values. Let's move on.