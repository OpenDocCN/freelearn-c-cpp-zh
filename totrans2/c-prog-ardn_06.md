# Chapter 6. Sensing the World – Feeling with Analog Inputs

The real world isn't digital. My digital-art-based vision shows me *The Matrix* behind things and huge digital waterfalls between things. In this chapter, however, I need to convey to you the relationship between digital and analog, and we need to understand it well.

This chapter is a good one but a huge one. Don't be afraid. We'll also discuss new concepts a lot while writing and designing pure C++ code.

We are going to describe together what analog inputs are. I'm also going to introduce you to a new and powerful friend worthy of respect, Max 6 framework. Indeed, it will help us a bit like Processing did—to communicate with the Arduino board. You'll realize how important this is for computers, especially when they have to sense the world. A computer with the Max 6 framework is very powerful, but a computer with the Max 6 framework and the Arduino plugin can feel much characteristics of the physical world, such as pressure, temperature, light, color, and many more. Arduino, as we have already seen, behaves a bit like a very powerful organ able to…*feel*.

If you like this concept of feeling things, and especially that of making other things react to these feelings, you'll love this chapter.

# Sensing analog inputs and continuous values

There's no better way to define analog than by comparing it to digital. We just talked about digital inputs in the previous chapter, and you now know well about the only two values those kind of inputs can read. It is a bit exhausting to write it, and I apologize because this is indeed more a processor constraint than a pure input limitation. By the way, the result is that a digital input can only provide 0 or 1 to our executed binary firmware.

Analog works totally differently. Indeed, analog inputs can continuously provide variable values by measuring voltage from 0 V to 5 V. It means a value of 1.4 V and another value of 4.9 V would be interpreted as totally different values. This is very different from a digital input that could interpret them as…1\. Indeed, as we already saw, a voltage value greater than 0 is usually understood as 1 by digital inputs. 0 is understood as 0, but 1.4 would be understood as 1; this we can understand as HIGH, the ON value, as opposed to the OFF, which comes from the 0 V measure.

Here, in the continuous world of analog inputs, we can sense a flow between the different values, where digital inputs can provide only steps. This is one of the reasons why I'm always using the term "feeling". Yes, when you can measure a lot of values, this is near to sensing and feeling. This is a bit of humanization of the electronic hardware, and I totally assume that.

## How many values can we distinguish?

The term "a lot" isn't precise. Even if we are in a new continuous field of measure, we are still in the digital world, the one of the computers. So how many values can be distinguished by Arduino's analog inputs? 1024.

Why 1024? The reason is easy to understand if you understand how Arduino can feel continuous values.

Because Arduino's chip works in the digital domain for all calculations, we have to convert analog values from 0 V to 5 V to a digital one. The purpose of the **analog-to-digital converter** , housed within the chipset itself, is exactly this. This device is also referred to using the acronym ADC.

Arduino's ADCs have a 10-bit resolution. This means that every analog value is encoded and mapped to a 10-bit, encoded integer value. The maximum number encodable using this encoding system is 1111111111 in the binary system, which means 1023 in the decimal system. If I consider the first number to be 0, we have a total of 1024 values represented. A 1024-value resolution provides a very comfortable field of sensing as we are going to see in the next few pages.

Let's see how we can use these precious inputs with Arduino.

## Reading analog inputs

Because we are now more familiar with circuits and code, we can work with a small project while still explaining concepts. I'm going to describe a simple example of circuits and code using a **potentiometer** only.

### The real purpose of the potentiometer

First, let's grab a potentiometer. A potentiometer is, if you remember correctly from the first chapter of this book, a variable resistor.

Considering Ohm's law, which links voltage, current, and resistance value, we can understand that, for a constant current, we can make the voltage vary by changing the value of the resistance of the potentiometer. Indeed, because some of us haven't dusted off our elementary electronics course textbook in many years, how about a refresher? Here's Ohm's law:

V = R * I

Here, V is the voltage in Volts, R the resistance in Ohms, and I the current in Amperes.

So now, to define the purpose of a potentiometer:

### Note

The potentiometer is your way to change continuously a variable in your running code from the physical world.

### Tip

**Always remember:**

Use 10-bit resolution, and you'll be the master of analog inputs!

### Changing the blinking delay of an LED with a potentiometer

The following figure is the most basic circuit to illustrate the concept of analog inputs with the Arduino board:

![Changing the blinking delay of an LED with a potentiometer](img/7584_06_01.jpg)

A potentiometer connected to the Arduino board

Check the corresponding electrical diagram for connections:

![Changing the blinking delay of an LED with a potentiometer](img/7584_06_02.jpg)

Analog input 0 is measuring the voltage

Now let's see the code we have to use.

Like the function `digitalRead()`, which can read the value of digital inputs on the Arduino, there is `analogRead()` for doing the same with analog inputs.

The intention here is to read the value as a pause value in our program for the purpose of controlling the blink rate of an LED. In code, we'll be using the `delay()` function.

Here's an example:

[PRE0]

Upload the code. Then turn the pot a bit, and observe the output.

After the variable definition, I'm defining the `ledPin` pin as output in the `setup()` function in order to be able to drive current to this pin. Actually, I'm using pin 13 in order to simplify our tests. Don't forget pin 13 is the surface-mounted LED on the Arduino board.

Then, the magic happens in the `loop()` function.

I'm first reading the value at the `potPin` pin. As we discussed before, the value returned by this function is an integer between 0 and 1023\. I'm storing it in the `potValue` variable to keep the LED ON, but also to keep it OFF.

Then, I'm turning OFF and ON the LED with some delay between status changes. The smart thing here is to use `potValue` as the delay. Turned on one side completely, the potentiometer provides a value of 0\. Turned on the other side completely, it provides 1023, which is a reasonable and user-friendly delay value in milliseconds.

The higher the value is, the longer the delay.

In order to be sure you understood the physical part of this, I'd like to explain a bit more about voltage.

The +5 V and ground pins of the Arduino supply the potentiometer the voltage. Its third leg provides a way to vary the voltage by varying the resistance. The Arduino's analog inputs are able to read this voltage. Please notice that analog pins on the Arduino are inputs only. This is also why, with analog pins, we don't have to worry about precision in the code like we have for digital pins.

So let's modify the code a bit in order to read a voltage value.

### How to turn the Arduino into a low voltage voltmeter?

Measuring voltage requires two different points on a circuit. Indeed, a voltage is an electrical potential. Here, we have (only) that analog pin involved in our circuit to measure voltage. What's that ?!

Simple! We're using the +5 V supply from Vcc as a reference. We control the resistance provided by the potentiometer and supply the voltage from the Vcc pin to have something to demonstrate.

If we want to use it as a real potentiometer, we have to supply another part of a circuit with Vcc too, and then connect our A0 pin to another point of the circuit.

As we saw, the `analogRead()` function only provides integers from 0 to 1023\. How can we have real electrical measures displayed somewhere?

Here's how it works:

The range 0 to 1023 is mapped to 0 to 5V. That comes built into the Arduino. We can then calculate the voltage as follows:

V = 5 * (analogRead() value / 1023)

Let's implement it and display it on our computer by using the serial monitor of the Arduino IDE:

[PRE1]

The code is almost the same as the previous code.

I added a variable to store the calculated voltage. I also added the serial communication stuff, which you see all the time: `Serial.begin(9600)` to instantiate the serial communication and `Serial.println()` to write the current calculated voltage value to the serial communication port, followed by a carriage return.

In order to see a result on your computer, you have to turn on the serial monitor, of course. Then, you can read the voltage values.

#### Calculating the precision

Please note that we are using an ADC here in order to convert an analog value to digital; then, we are making a small calculation on that digital value in order to have a voltage value. This is a very expensive method compared to a basic analog voltage controller.

It means our precision depends on the ADC itself, which has a resolution of 10 bits. It means we can only have 1024 values between 0 V and 5 V. 5 divided by 1024 equals 0.00488, which is approximated.

It basically means we won't be able to distinguish between values such as 2.01 V and 2.01487 V, for instance. However, it should be precise enough for the purposes of our learning.

Again, it was an example because I wanted to point out to you the precision/resolution concept. You have to know and consider it. It will prove very important and could deliver strange results in some cases. At least, you have been warned.

Let's discover another neat way of interacting with the Arduino board.

# Introducing Max 6, the graphical programming framework

Now, let me introduce you to the framework known as Max 6\. This is a whole universe in itself, but I wanted to write some pages about it in this book because you'll probably come across it in your future projects; maybe you'll be a Max 6 developer one day, like me, or perhaps you'll have to interface your smart physical objects with Max 6-based systems.

The following is one of the patches of my 3D universe project with Max 6:

![Introducing Max 6, the graphical programming framework](img/7584_06_03.jpg)

## A brief history of Max/MSP

Max is a visual programming language for multimedia purposes. It is actually developed and maintained by Cycling '74\. Why call it Max? It was named after Max Matthews ([http://en.wikipedia.org/wiki/Max_Mathews](http://en.wikipedia.org/wiki/Max_Mathews)), one of the great pioneers of computer music.

The original version of Max was written by Miller Puckette; it was initially an editor named Patcher for Macintosh. He wrote it at **The European Institut de Recherche et Coordination Acoustique/Musique** (**IRCAM**), an avant-garde science institute based near the Centre Pompidou in Paris, France.

In 1989, the software was licensed by IRCAM to Opcode Systems, a private company, and ever since then, has been developed and extended by David Zicarelli. In the mid-'90s, Opcode Systems ceased all development for it.

Puckette released a totally free and open source version of Max named Pure Data (often seen as Pd). This version is actually used a lot and maintained by the community that uses it.

Around 1997, a whole module dedicated to sound processing and generation has been added, named **MSP** , for **Max Signal Processing** and, apparently, for the initials of Miller S. Puckette.

Since 1999, the framework commonly known as Max/MSP has been developed and distributed by Cycling '74, Mr. Zicarelli's company.

Because the framework architecture was very flexible, some extensions have progressively been added, such as Jitter (a huge and efficient visual synthesis), Processing, real-time matrix calculations modules, and 3D engine too. This happened around 2003\. At that time, Jitter was released and could be acquired separately but required Max, of course.

In 2008, a major update was released under the name Max 5\. This version too did not include Jitter natively but as an add-on module.

And the most giant upgrade, in my humble opinion, released in November 2011 as Max 6, included Jitter natively and provided huge improvements such as:

*   A redesigned user interface
*   A new audio engine compatible with 64-bit OSs
*   High-quality sound filter design features
*   A new data structure
*   New movement handling for 3D models
*   New 3D material handling
*   The Gen extension

Max 4 was already totally usable and efficient, but I have to give my opinion about Max 6 here. Whatever you have to build, interfaces, complex, or easy communication protocols including HID-based (**HID**=**human interface device**) USB devices such as Kinect, MIDI, OSC, serial, HTTP, and anything else, 3D-based sound engine or basic standalone applications for Windows or OS X platform, you can make it with Max 6, and it is a safe way to build.

Here is my own short history with Max: I personally began to play with Max 4\. I specially built some macro MIDI interfaces for my first hardware MIDI controllers in order to control my software tools in very specific ways. It has taught me much, and it opened my mind to new concepts. I use it all the time, for almost every part of my artistic creation.

Now, let's understand a little bit more about what Max is.

## Global concepts

Of course, I hesitated to begin the part about Max 6 in the preceding section. But I guess the little story was a good starting point to describing the framework itself.

### What is a graphical programming framework?

A **graphical programming framework** is a programming language that provides a way for users to create programs by manipulating elements graphically instead of by typing text.

Usually, graphical programming languages are also called **visual programming languages** , but I'll use "graphical" because, to many, "visual" is used for the product rendered by frameworks; I mean, the 3D scene for instance. Graphical is more related to **GUI** , that is, **graphical user interface**, which is, from the developer point of view, our editor interface (I mean, the IDE part).

Frameworks using this strong graphical paradigm include many ways of programming in which we can find data, data types, operator and functions, input and output, and a way of connecting hardware too.

Instead of typing long source codes, you add objects and connect them together in order to build software architectures. Think Tinker Toys or Legos.

A global software architecture, which is a system of objects connected and related on our 2D screen, is called **Patch** in the Max world. By the way, other graphical programming frameworks use this term too.

If this paradigm can be understood at first as a way of simplification, it is not the first purpose, I mean that not only is it easier, but it also provides a totally new approach for programmers and non-programmers alike. It also provides a new type of support task. Indeed, if we don't program in the same way we patch, we don't troubleshoot problems in the same way too.

I can quote some other major graphical programming software in our fields:

*   **Quartz Composer**: This is a graphical rendering framework for OS X and is available at [https://developer.apple.com/technologies/mac/graphics-and-animation.html](https://developer.apple.com/technologies/mac/graphics-and-animation.html)
*   **Reaktor**: This is a DSP and MIDI-processing framework by Native Instruments and is available at [http://www.native-instruments.com/#/en/products/producer/reaktor-5](http://www.native-instruments.com/#/en/products/producer/reaktor-5)
*   **Usine**: This is a universal audio software for live and studio recording and is available at [http://www.sensomusic.com/usine](http://www.sensomusic.com/usine)
*   **vvvv**: This is a real-time video synthesis tool for Windows and is available at [http://vvvv.org](http://vvvv.org)
*   **SynthMa****ker**: This is a VST device design for Windows and is available at [http://synthmaker.co.uk](http://synthmaker.co.uk)

I'd like to make a special mention of Usine. It is a very interesting and powerful framework that provides graphical programming to design patches usable inside Usine software itself or even as standalone binaries. But one of the particularly powerful features is the fact you can export your patch as a fully-functional and optimized VST plugin. **VST** (**Virtual Studio Technology**) is a powerful standard created by the Steinberg company. It provides a huge list of specifications and is implemented in almost all digital audio workstations. Usine provides a one-click-only export feature that packs your graphically programmed patch into a standard VST plugin for people who haven't even heard about Usine or patching styles. The unique multitouch feature of Usine makes it a very powerful framework too. Then, you can even code your own modules using their C++ **SDKs** (**software development kits**).

![What is a graphical programming framework?](img/7584_06_04.jpg)

Usine big patch connecting the real world to many virtual objects

### Max, for the playground

Max is the playground and the core structure in which everything will be placed, debugged, and shown. This is the place where you put objects, connect them together, create a user interface (UI), and project some visual rendering too.

Here is a screenshot with a very basic patch designed to help you understand where things go:

![Max, for the playground](img/7584_06_05.jpg)

A small and easy calculation system patch with Max 6

As I described, with a graphical programming framework, we don't need to type code to make things happen. Here, I'm just triggering a calculation.

The box with the number **17** inside is a numbox. It holds an integer and it is also a UI object, providing a neat way to change the value by dragging and dropping with a mouse. You then connect the output of one object to the input of another. Now when you change the value, it is sent through the wire to the object connected to the numboxes. Magic!

You see two other objects. One with a:

*   **+** sign inside followed by the number **5**
*   **-** sign inside followed by the number **3**

Each one takes the number sent to them and makes the calculation of + 5 and - 3 respectively.

You can see two other numboxes displaying basically the resulting numbers sent by the objects with the **+** and **–** signs.

Are you still with me? I guess so. Max 6 provides a very well documented help system with all references to each object and is directly available in the playground. It is good to tell that to students when you teach them this framework, because it really helps the students teach themselves. Indeed, they can be almost autonomous in seeking answers to small questions and about stuff they have forgotten but don't dare to ask.

Max part provides quite an advanced task scheduler, and some objects can even modify priority to, say, `defer` and `deferlow` for a neat granularity of priorities inside your patch, for instance, for the UI aspect and the calculation core aspect that each require very different scheduling.

Max gives us a nifty debugging system too with a console-like window called the **Max window**.

![Max, for the playground](img/7584_06_06.jpg)

The Max window showing debugging information about the expr object's error in the patch

Max drives many things. Indeed, it is Max that owns and leads the access to all modules, activated or not, provides autocompletion when you create new objects, and also gives access to many things that can extend the power of Max, such as:

*   JavaScript API to Max itself and specific parts, such as Jitter, too
*   Java through the mxj object that instantiates directly inside Max 6 Java classes
*   MSP core engine for everything related to signal rate stuff, including audio
*   Jitter core engine for everything related to matrix processing and much more, such as visuals and video
*   Gen engine for efficient and on-the-fly code compilation directly from the patch

This is not an exhaustive list, but it gives you an insight of what Max provides.

Let's check the other modules.

### MSP, for sound

Where Max objects communicate by sending messages triggered by user or by the scheduler itself, MSP is the core engine that calculates signals at any particular instant, as written in the documentation.

Even if we can patch (or connect) MSP objects in the same way as pure Max objects, the concept underneath is different. At each moment, a signal element is calculated, making an almost continuous data flow through what we call a signal network. The signal network is easy to identify in the patcher window; the wires are different.

Here is an image of a very simple patch producing a cosine-based audio wave in your ears:

![MSP, for sound](img/7584_06_07.jpg)

Indeed, even the patch cords have a different look, showing cool, striped yellow-and-black, bee-like colors, and the names of the MSP objects contain a tilde `~` as a suffix, symbolizing…a wave of course!

The signal rate is driven by the audio sampling rate and some dark parameters in the MSP core settings window. I won't describe that, but you have to know that Max usually provides, by default, parameters related to your soundcard, which include the sampling rate (44110 Hz, the standard sampling rate for audio CDs, means a fast processing rate of 44100 times per second for each audio channel).

![MSP, for sound](img/7584_06_08.jpg)

The Audio Status window is the place where you set up some important MSP parameters

### Jitter, for visuals

Jitter is the core engine for everything related to visual processing and synthesis in Max 6.

It provides a very efficient framework of matrix processing initially designed for fast pixel value calculations to display pictures, animated or not.

We are talking about matrix calculation for everything related with Jitter processing matrices. And indeed, if you need to trigger very fast calculations of huge arrays in Max 6, you can use Jitter for that, even if you don't need to display any visuals.

Jitter provides much more than only matrix calculation. It gives full access to an OpenGL ([http://en.wikipedia.org/wiki/OpenGL](http://en.wikipedia.org/wiki/OpenGL)) implementation that works at the speed of the light. It also provides a way for designing and handling particle systems, 3D worlds, OpenGL materials, and physics-based animation. Pixel processing is also one of the powerful features provided with many objects designed and optimized for pixel processing itself.

![Jitter, for visuals](img/7584_06_09.jpg)

A basic Jitter-core-based patch generating a good resolution 400x400 noise pixel map

In order to summarize this massive load of information, Max schedules events or waits for the user to trigger something, MSP (for audio signal processing)—as soon as it is activated—calculates signal elements at each instant in its signal networks, and Jitter processes calculations when Jitter objects are triggered by **bangs**.

Indeed, Jitter objects need to be triggered in order to do their jobs, which can be very different, such as popping out a matrix that contains pixel color values, matrix processing for each cell of a matrix, and popping out the resulting matrix, for instance.

Bangs are special messages used to kinda say "*Hey, let's start your job!*" to objects. Objects in Max can behave differently, but almost every one can understand the bang message.

In **Patch003** (pictured in the previous screenshot), the Max object `qmetro` provides a bang every 20 ms from a low priority scheduler queue to a Jitter object named `jit.noise`. This latter object calculates a matrix filled with a random value in each cell. Then, the result goes through a new green-and-black-striped patch cord to a UI object in which we can see a name, the `jit.pwindow`, a kind of display we can include in our patchers.

Jitter can be controlled via powerful Java and JavaScript APIs for some tasks that require typing big loops in code , which are easy to design using code.

Still here?

For the bravest among the brave, some other rows about Gen, the latest and most efficient module of Max 6.

### Gen, for a new approach to code generation

If you understood that there was a kind of compilation/execution behind our patches, I'd disappoint you by saying it doesn't really work like that. Even if everything works real time, there isn't a real compilation.

By the way, there are many ways to design patch bits using code, with JavaScript for instance. Directly inside Max patcher, you can create a `.js` object and put your JavaScript code inside; it is indeed compiled on the fly (it is called **JS JIT** compiler, for JavaScript just-in-time compiler). It is really fast. Believe me, I tested it a lot and compared it to many other frameworks. So, as the documentation said, "we are not confined to writing Max externals in C" even if it is totally possible using the Max 6 SDK ([http://cycling74.com/products/sdk](http://cycling74.com/products/sdk)).

Gen is a totally new concept.

Gen provides a way of patching patch bits that are compiled on the fly, and this is a real compilation from your patch. It provides a new type of patcher with specific objects, quite similar to Max objects.

It works for MSP, with the `gen~` Max object, providing a neat way to design signal-rate related to audio patches architecture. You can design DSP and sound generators like that. The `gen~` patches are like a zoom in time; you have to consider them as sample processors. Each sample is processed by those patches inside the `gen~` patchers. There are smart objects to accumulate things over time, of course, in order to have signal processing windows of time.

It works also for Jitter with three main Max objects:

*   `jit.gen` is the fast matrix processor, processing each cell of a matrix at each turn
*   `jit.pix` is the CPU-based pixel processor, processing each pixel of a pixel map
*   `jit.gl.pix` is the GPU-based version of `jit.pix`

A GPU (graphics processor unit), and is basically a dedicated graphics processor on your video card. Usually, and this is a whole different universe, OpenGL pipeline provides an easy way to modify pixels from the software definitions to the screen just before they are displayed on the screen. It is called **shader process**.

You may already know that term in relation with the world of gaming. These are those shaders that are some of the last steps to improving graphics and visual renders in our games too.

Shaders are basically small programs that can be modified on the fly by passing arguments processed by the GPU itself. These small programs use specific languages and run vary fast on dedicated processors on our graphic cards.

Max 6 + Gen provides direct access to this part of the pipeline by patching only; if we don't want to write shaders based on **OpenGL GLSL** ([http://www.opengl.org/documentation/glsl](http://www.opengl.org/documentation/glsl)), **Microsoft DirectX HLSL** ([85).aspx">http://msdn.microsoft.com/en-us/library/bb509635(v=VS.).aspx">85).aspx](http://msdn.microsoft.com/en-us/library/bb509635(v=VS.).aspx)), or **Nvidia Cg** ([http://http.developer.nvidia.com/CgTutorial/cg_tutorial_chapter01.html](http://http.developer.nvidia.com/CgTutorial/cg_tutorial_chapter01.html)), Gen is your friend.

All patches based on `jit.gl.pix` are specifically compiled and sent for GPU-based execution.

You can then design your own fragment shaders (or pixel shaders) by patching and you can even grab the source code in GLSL or WebGL language in order to use it in another framework, for instance.

Geometry shaders aren't available using Gen, but with other Jitter objects they already exists.

I guess I lost some of you. Relax I won't ask you questions about Gen in Arduino exams!

### Summarizing everything in one table

Everything related to Max 6 is on the Cycling 74's website at [http://cycling74.com](http://cycling74.com). Also, almost 99 percent of the documentation is online too, at [http://cycling74.com/docs/max6/dynamic/c74_docs.html#docintro](http://cycling74.com/docs/max6/dynamic/c74_docs.html#docintro).

The following table summarizes everything we did until now:

| Parts | What? | Cable color | Distinctive sign |
| --- | --- | --- | --- |
| Max | The playground | Gray by default and no stripes | Basic names |
| MSP | Everything related to audio and signal rate | Yellow-and-black stripes | `~` suffixed to the namesignal-rate processing |
| Jitter | Everything related to visuals and matrices | Green-and-black stripes for matrix cablesBlue-and-black stripes for pixel map cables | `jit.` prefixed to the name |
| Gen | Specific patchers (DSP-related and matrix and texture processing) compiled on the fly | Like MSP for `gen~` and Jitter for `jit.gen`, `jit.pix`, and `jit.gl.pix` | Very very fast! |

## Installing Max 6

Max 6 is available as a 30-day trial. Installing Max 6 is quite easy as it comes with an installer for both platforms, Windows and OS X, downloadable at [http://cycling74.com/downloads](http://cycling74.com/downloads). Download and install it. Then, launch it. That's all. (The following examples will only work when you have installed Max.)

You should see a blank playground

![Installing Max 6](img/7584_06_10.jpg)

Max 6 blank-page anxiety can occur right now, can't it?

## The very first patch

Here is a basic patch you can find also in the `Chapter06/` folder under the name `Patcher004_Arduino.maxpat`. Usually, if you double-click on it, it is opened directly by Max 6.

This patch is a very basic one, but not that basic actually!

It is a basic noise-based sequencer modifying an oscillator's frequency regularly in real time. This produces a sequence of strange sounds, more or less pretty, the modifications of the frequency being controlled by chance. So, turn on your speakers and the patch will produce sounds.

![The very first patch](img/7584_06_11.jpg)

The noise-based sequencer

Basically, patches are stored in files. You can share patches with other friends quite easily. Of course, bigger projects would involve some dependency issues; if you added some libraries to your Max 6 framework, if you use them in a patch, or if you basically send your patch files to a friend who doesn't have those libraries installed, your friend will have some errors in the Max Window. I won't describe these kinds of issues here, but I wanted to warn you.

Other neat ways to share patches in the Max 6 world are the copy/paste and copy compressed features. Indeed, if you select objects in your patcher (whatever the layer, including a subpatcher, inside a subpatcher, and so on) and go to **Edit** | **Copy**, text-based content is put in your clipboard. This can then be repasted into another patcher or inside a text file.

The smartest way is the use of copy compress, which as the well-chosen name means, copies and compresses the JSON code to something much more compact and easy to copy into the text area on forums, for instance.

Wait, let me show you what it looks like.

I just selected all objects in my patch and went to **Edit** | **Copy Compressed**.

![The very first patch](img/7584_06_12.jpg)

The copy compressed feature

And the following figure is the result of pasting directly into a text file.

Those familiar with HTML would notice something funny; Cycling '74 developers include two HTML tags (`pre` and `code`) in order to directly provide code that is pastable inside a text field on (any) forums on the Web.

![The very first patch](img/7584_06_41.jpg)

Copy-compressed code

So you can also copy that code into your clipboard and paste it into a new patch. You create a new empty patch by by going to **File** | **New** (or hitting *Ctrl* + *N* on Windows and *command* + *N* on OS X).

### Playing sounds with the patch

As you can see, I put some comments in the patcher. You can follow them in order to produce some electronic sounds from your computer.

Before you begin, be sure to lock the patch by clicking on the padlock icon in the lower-left corner. To hear the results of the patch, you'll also need to click on the speaker icon. To zoom out, go to the **View** menu and click on **Zoom Out**.

First, note and check the toggle at the top. It will send the value `1` to the connected object metro.

A metro is a pure Max object that sends a bang every *n* milliseconds. Here, I hardcoded an argument: `100`. As soon as the metro receives the message `1` from the toggle, it begins to be active and, following the Max timing scheduler, it will send its bangs every 100 ms to the next connected object.

When the `random` object receives a bang, it pops out a random integer from within a range. Here, I put `128`, which means `random` will send values from `0` to `127`. Directly after `random`, I put a `zmap` object that works like a scaler. I harcoded four arguments, minimum and maximum values for inputs and minimum and maximum values for output.

Basically, here, `zmap` maps my values `0` to `127` sent by `random` to another values from `20` to `100`. It produces an implicit stretch and loss of resolution that I like.

Then, this resulting number is sent to the famous and important `mtof` object. This converts a MIDI note pitch standard to a frequency according to the MIDI standard. It is often used to go from the MIDI world into the real sound world. You can also read the frequency in the UI object `flonum` displaying the frequency as a float number in Hz (hertz, a measure of frequency).

Then, at last, this frequency is sent to the `cycle~` object, producing a signal (check the yellow-and-black striped cord). Sending numbers to this object makes it to change the frequency of the signal produced. This one is multiplied by a signal multiply operator `*~`, producing another signal but with a lower amplitude to protect our precious ears.

The last destination of that signal is the big gray box on which you have to click once in order to hear or not hear the sounds produced by the upper signal network.

Now you're ready to check the toggle box. Activate the speaker icon by clicking on the gray box, and then you can dance. Actually, electronic sounds produced are a bit shuffly about the frequency (that is, the note) but it can be interesting.

Of course, controlling this cheap patch with the Arduino in order to not use the mouse/cursor would be very great.

Let's do that with the same circuit that we designed previously.

# Controlling software using hardware

Coming from pure digital realms where everything can be wrapped into software and virtual worlds, we often need physical interfaces. This can sound like a paradox; we want everything in one place, but that place is so small and user-unfriendly for everything related to pure creation and feelings that we need more or less big external (physical) interfaces. I love this paradox.

But, why do we need such interfaces? Sometimes, the old mouse and QWERTY keyboard don't cut it. Our computers are fast, but these interfaces to control our programs are slow and clunky.

We need interfaces between the real world and the virtual world. Whatever they are, we need them to focus on our final purpose, which is usually not the interface or even the software itself.

Personally, I write books and teach art-related technology courses, but as a live performer, I need to focus on the final rendering. While performing, I want to black-box as much as possible the technology under the hood. I want to feel more than I want to calculate. I want a controller interface to help me operate at the speed and level of flexibility to make the types of changes I want.

As I already said in this book, I needed a huge MIDI controller, heavy, solid, and complex, in order to control only one software on my computer. So, I built Protodeck ([http://julienbayle.net/protodeck](http://julienbayle.net/protodeck))). This was my interface.

So, how can we use Arduino to control software? I guess you have just a part of your answer because we already sent data to our computer by turning a potentiometer.

Let's improve our Max 6 patch to make it receive our Arduino's data while we turn the potentiometer.

## Improving the sequencer and connecting the Arduino

We are going to create a very cheap and basic project that will involve our Arduino board as a small sound controller. Indeed, we'll directly use the firmware we just designed with the potentiometer, and then we'll modify our patch. This is a very useful base for you to continue to build things and even create bigger controller machines.

### Let's connect the Arduino to Max 6

Arduino can communicate using the serial protocol. We already did that. Our latest firmware already does that, sending the voltage value.

Let's modify it a bit and make it send only the analog value read, within the range `0` to `1023`. Here is the code, available in `Chapter06/maxController`:

[PRE2]

I removed everything unnecessary and added a delay of 2 ms at the end of the loop (before the loop restarts) This is often used with analog input and especially ADC. It provides a break to let it stabilize a bit. I didn't do that in previous code involving analog read because there were already two `delay()` methods involved in the LED blinking.

This basic one sends the value read at the analog input pin where the potentiometer is connected. No more, but no less.

Now, let's learn how to receive that somewhere other than the Serial Monitor of our precious IDE, especially in Max 6.

### The serial object in Max 6

There is a Max object named `serial`. It provides a way to communicate using a serial port with any other type of device using serial communication.

The next figure describes the new Max 6 patch including the part necessary to communicate with our small hardware controller.

Now, let's plug the Arduino in, if this has not been done already, and upload the `maxController` firmware.

### Note

Be careful to switch off serial monitoring for the IDE.

Otherwise, there would be a conflict on your computer; only one serial communication can be instantiated on one port.

Then here is another patch you can find, also in the `Chapter06/` folder, with the name `Patcher005_Arduino.maxpat`.

![The serial object in Max 6](img/7584_06_13.jpg)

The Max patch including the Arduino communication module

Double-click on the file, and you'll see this patch.

Let's describe it a bit. I added everything in green and orange.

Everything necessary to understand the Arduino messages and to convert them in terms understandable easily by our sequencer patch is in green. Some very useful helpers that are able to write to the Max window at every step of the data flow, from raw to converted data, are in orange.

Let's describe both parts, beginning with the helpers.

### Tracing and Debugging easily in Max 6

Max 6 provides many ways to debug and trace things. I won't describe them all in this Arduino book, but some need a few words.

Check your patch, especially the orange-colored objects.

`print` objects are the way to send messages directly to the Max window. Everything sent to them is written to the Max window as soon it has been received. The argument you can pass to these objects is very useful too; it helps to discern which `print` object sends what in cases where you use more than one `print` object. This is the case here and check: I name all my `print` objects considering the object from which comes the message:

*   `fromSerial`: This is for all messages coming from the `serial` object itself
*   `fromZl`: This is for all messages coming from the `zl` object
*   `fromitoa`: This is for all messages coming from the `itoa` object
*   `fromLastStep`: This is for all messages coming from the `fromsymbol` object

The `gate` objects are just small doors, gates that we can enable or disable by sending `1` or `0` to the leftmost input. The `toggle` objects are nice UI objects to do that by clicking. As soon as you check the toggle, the related `gate` object will let any message sent to the right input pass through them to the only one output.

We are going to use this trace system in several minutes.

### Understanding Arduino messages in Max 6

What is required to be understood is that the previous toggle is now connected to a new `qmetro` object too. This is the low priority `metro` equivalent. Indeed, this one will poll the `serial` object every 20 ms, and considering how our Arduino's firmware currently works by sending the analog value read at every turn in the loop, even if this polling lags a bit, it won't matter; the next turn, the update will occur.

The `serial` object is the important one here.

I hardcoded some parameters related to serial communication with the Arduino:

*   `9600` sets the clock to 9600 bauds
*   `8` sets the word length at 8 bit
*   `1` means there is a stop bit
*   `0` means there is no parity (parity is sometimes useful in error checking)

This object needs to be banged in order to provide the current content of the serial port buffer. This is the reason why I feed it by the `qmetro` object.

The `serial` object pops out a raw list of values. Those values need to be a bit parsed and organized before reading the analog value sent. This is what the `select`, `zl`, `itoa`, and `fromsymbol` objects stand for.

### Note

Directly read the help information for any object in Max 6 by pushing the *Alt* key on your keyboard and then clicking on the object.

![Understanding Arduino messages in Max 6](img/7584_06_14.jpg)

The serial object's help patch

Every 20 ms, if the serial communication has been instantiated, the `serial` object will provide what will be sent by the Arduino, the current and most recently read analog value of the pin where the potentiometer is connected. This value going from 0 to 1023, I'm using a `scale` object as I did with the `zmap` object for the sequencer/sound part of the patch. This `scale` object recasts the scale of values from 0 to 1023 at input to an inverted range of 300 down to 20, letting the range to go opposite direction (be careful, current and future Max patchers, `zmap` doesn't behave like that). I did that in order to define the maximum range of the note-per-minute rate. The `expr` object calculates this. `qmetro` needs the interval between two bangs. I'm making this vary between 400 ms and 20 ms while turning my potentiometer. Then, I calculate the note-per-minute rate and display it in another `flonum` UI object.

Then, I also added this strange `loadbang` object and the `print` one. `loadbang` is the specific object that sends a bang as soon as the patcher is opened by Max 6\. It is often used to initialize some variable inside our patcher, a bit like we are doing with the declarations in the first rows of our Arduino sketches.

`print` is only text inside an object named `message`. Usually, each Max 6 object can understand specific messages. You can create a new empty message by typing `m` anywhere in a patcher. Then, with the autocomplete feature, you can fill it with text by selecting it and clicking on it again.

Here, as soon as the patch is loaded and begins to run, the `serial` object receives the print message triggered by `loadbang`. The `serial` object is able to send the list of all serial port messages to the computer that runs the patch to the console (that is, the Max window). This happens when we send the print message to it. Check the Max window of the figure showing the `Patcher005_Arduino.maxpat` patch.

We can see a list of…things. `serial` pops out a list of serial port letter abbreviations with the corresponding serial ports often representing the hardware name. Here, as we already saw in the Arduino IDE, the one corresponding to the Arduino is `usbmodemfa131`.

The corresponding reference in Max is the letter `c` on my computer. This is only an internal reference.

![Understanding Arduino messages in Max 6](img/7584_06_15.jpg)

Result of the print message sent to the serial object: the list of port letters / names of serial ports

Let's change the hardcoded letter put as argument for the `serial` object in the patch.

Select the `serial` object. Then, re-click inside and swap `a` with the letter corresponding to the Arduino serial port on your computer. As soon as you hit *Enter*, the object is instantiated again with new parameters.

![Understanding Arduino messages in Max 6](img/7584_06_16.jpg)

Changing the reference letter in the serial object to match the one corresponding to the serial port of the Arduino

Now, everything is ready. Check the toggle, enable the gray box with the speaker, and turn your potentiometer. You are going to hear your strange noises from the sequencer, and you can now change the note rate (I mean the interval between each sound) because I abusively used the term note to fit better to the sequencer's usual definition.

### What is really sent on the wire?

You will have noticed that, as usual, I mentioned the series of objects: `select`, `zl`, `itoa`, and `fromsymbol`. The time has come to explain them.

When you use the `Serial.println()` function in your Arduino's firmware source code, the Arduino doesn't send only the value passed as argument to the function. Check the first orange toggle at the top of the series of toggle/gate systems.

![What is really sent on the wire?](img/7584_06_17.jpg)

The serial object pops out strange series of numbers

You can see the name of the printing object in the first column named **Object**, and in the **Message** column, the message sent by the related object. And we can see the `serial` object popping out strange series of numbers in a repetitive way: **51**, **53**, **48**, **13**, **10**, and so on.

### Note

Arduino transmits its values as ASCII, exactly as if we were typing them on our computer.

This is very important. Let's check the *Appendix E, ASCII Table*, in order to find the corresponding characters:

*   51 means the character 3
*   53 means 5
*   48 means 0
*   13 means a carriage return
*   10 means line feed, which itself means new line

Of course, I cheated a bit by sorting the series as I did. I knew about the `10 13` couple of numbers. It is a usual marker meaning *a carriage return followed by a new line*.

So it seems that my Arduino sent a message a bit like this:

[PRE3]

Here, `<CR>` and `<LF>` are carriage return and new line characters.

If I had used the `Serial.print()` function instead of `Serial.println()`, I wouldn't have had the same result. Indeed, the `Serial.print()` version doesn't add the `<CR>` and `<NL>` characters at the end of a message. How could I have known whether `3`, `5`, or `0` would be the first character if I didn't have an end marker?

The design pattern to keep in mind is as follows:

*   Build the message
*   Send the message after it is completely built (using the `Serial.println()` function.)

If you want to send it while building it, here's what you can use:

*   Send the first byte using `Serial.print()`
*   Send the second byte using `Serial.print()`
*   Continue to send until the end
*   Send the `<CR><LF>` at the end by using `Serial.println()` with no argument

### Extracting only the payload?

In many fields related to communication, we talk about payload. This is the message, the purpose of the communication itself. Everything else is very important but can be understood as a carrier; without these signals and semaphores, the message couldn't travel. However, we are interested in the message itself.

We need to parse the message coming from the serial object.

We have to accumulate each ASCII code into the same message, and when we detect the `<CR><LF>` sequence, we have to pop out the message block and then restart the process.

This is done with the `select` and `zl` objects.

`select` is able to detect messages equaling one of its arguments. When `select 10 13` receives a 10, it will send a bang to the first output. If it is a 13, it will send a bang to the second output. Then, if anything else comes, it will just pass the message from the last output to the right.

`zl` is such a powerful list processor with so many usage scenarios that it would make up a book by itself! Using argument operator, we can even use it to parse the data, cut lists into pieces, and much more. Here, with the group 4 argument, `zl` receives an initial message and stores it; when it receives a second message, it stores the message, and so on, until the fourth message. At the precise moment that this is received, it will send a bigger message composed of the four messages received and stored. Then, it clears its memory.

Here, if we check the corresponding toggle and watch the Max window, we can see **51 53 48** repeated several times and sent by the `zl` object.

The `zl` object does a great job; it passes all ASCII characters except `<CR>` and `<LF>`, and as soon as it receives `<LF>`, `zl` sends a bang. We have just built a message processor that *resets* the `zl` buffer each time it receives `<LF>`, that is, when a new message is going to be sent.

![Extracting only the payload?](img/7584_06_18.jpg)

The zl list processor pops out a series of integers

### ASCII conversions and symbols

We have now a series of three integers directly equaling the ASCII message sent by the Arduino, in my case, `51 53 48`.

If you turn the potentiometer, you'll change this series, of course.

But look at this, where is the value between 0 and 1023 we so expected? We have to convert the ASCII integer message into a real character one. This can be done using the `itoa` object (which means integer to ASCII).

Check the related toggle, and watch the Max window.

![ASCII conversions and symbols](img/7584_06_19.jpg)

Here is our important value

This value is the important one; it is the message sent by the Arduino over the wire and is transmitted as a symbol. You cannot distinguish a symbol from another type of message, such as an integer or a float in the Max window.

I placed two empty messages in the patch. Those are really useful for debugging purposes too. I connect them to the `itoa` and `fromsymbol` objects to their right input. Each time you send a message to another message on its right input, the value of the destination message is changed by the content of the other one. We can then display what message is really sent by `itoa` and `fromsymbol`.

![ASCII conversions and symbols](img/7584_06_20.jpg)

"350" doesn't equal exactly 350

`fromsymbol` transforms each symbol into its component parts, which here make up an integer, `350`.

This final value is the one we can use with every object able to understand and process numbers. This value is scaled by the scale object and sent, at last, to the metro object. Turning the potentiometer changes the value sent, and depending upon the value, the metro sends bangs faster or slower.

This long example taught you two main things:

*   You have to carefully know what is sent and received
*   How an Arduino communicates

Now, let's move on to some other examples relating to analog inputs.

### Playing with sensors

What I don't want to write in this book is a big catalog. Instead of that, I want to give you keys and the feel of all the concepts. Of course, we have to be precise and learn about particular techniques you didn't invent yourself, but I especially want you to learn best practices, to think about huge projects by yourself, and to be able to have a global vision.

I'll give you some examples here, but I won't cover every type of sensor for the previously mentioned reason.

### Measuring distances

When I design installations for others or myself, I often have the idea of measuring distance between moving things and a fixed point. Imagine you want to create a system with a variable light intensity depending on the proximity of some visitors.

I used to play with a Sharp GP2Y0A02YK infrared long range sensor.

![Measuring distances](img/7584_06_21.jpg)

The infrared Sharp GP2Y0A-family sensor

This cool analog sensor provides good results for distances from 20 to 150 cm. There are other types of sensors on the market, but I like this one for its stability.

As with any distance sensors, the subject/target has to theoretically be perpendicular to the infrared beam's direction for maximum accuracy, but in the real world, it works fine even otherwise.

The datasheet is a first object to take care about.

### Reading a datasheet?

First, you have to find the datasheet. A search engine can help a lot. This sensor's datasheet is at [http://sharp-world.com/products/devvice/lineup/data/pdf/datasheet/gp2y0a02_e.pdf](http://sharp-world.com/products/devvice/lineup/data/pdf/datasheet/gp2y0a02_e.pdf).

You don't have to understand everything. I know some fellows would blame me here for not explaining the datasheet, but I want my students to be relaxed about that. You have to filter information.

Ready? Let's go!

Generally, on the first page, you have all the features summarized.

Here, we can see this sensor seems to be quite independent considering the color of the target. Ok, good. The distance output type is very important here. Indeed, it means it outputs the distance directly and needs no additional circuitry to utilize its analog data output.

There are often some schematics of all dimensions of the outline of the sensor. This can be very useful if you want to be sure the sensor fits your box or installation before ordering it.

In the next figure, we can see a graph. This is a curve illustrating how the output voltage varies according to the distance of the target.

![Reading a datasheet?](img/7584_06_22.jpg)

Mathematical relation between distance and analog output voltage from the sensor

This information is precious. Indeed, as we discussed in the previous chapter, a sensor converts a physical parameter into something measurable by Arduino (or any other type of equipment). Here, a distance is converted into a voltage.

Because we measure the voltage with the analog input of our Arduino board, we need to know how the conversion works. And I'm going to use a shortcut here because I made the calculation for you.

Basically, I used another graph similar to the one we saw but mathematically generated. We need a formula to code our firmware.

If the output voltage increases, the distance decreases following *a kind of* exponential function. I had been in touch with some Sharp engineers at some point and they confirmed my thoughts about the type of formula, providing me with this:

![Reading a datasheet?](img/7584_06_42.jpg)

Here, D is the distance in centimeters and V the voltage measured; and a = 0.008271, b = 939.65, c = -3.398, and d = 17.339

This formula will be included in Arduino's logic in order to make it directly provide the distance to anyone who would like to know it. We could also make this calculation on the other side of the communication chain, in a Max 6 patch for instance, or even in Processing. Either way, you want to make sure your distance parameter data scales well when comparing the output from the sensor to the input where that data will be used.

### Let's wire things

This next circuit will remind you very much of the previous one. Indeed, the range sensor replaces the potentiometer, but it is wired in exactly the same way:

*   The Vcc and ground of the Arduino board connected respectively to +5 V and ground
*   The signal legs connected to the analog input 0

![Let's wire things](img/7584_06_23.jpg)

The Sharp sensor connected to the Arduino board

The circuit diagram is as follows:

![Let's wire things](img/7584_06_24.jpg)

The sensor range supplied by the Arduino itself and sending voltage to the Analog Input 0

### Coding the firmware

The following code is the firmware I designed:

[PRE4]

Is it not gratifying to know you understood every line of this code? Just in case though, I will provide a brief explanation.

I need some variables to store the sensor value (that is, the values from `0` to `1023`) coming from the ADC. Then, I need to store the voltage calculated from the sensor value, and of course, the distance calculated from the voltage value.

I only initiate serial communication in the `setup()` function. Then, I make every calculation in the `loop()` method.

I started by reading the current ADC value measured and encoded from the sensor pin. I use this value to calculate the voltage using the formula we already used in a previous firmware. Then, I inject this voltage value into the formula for the Sharp sensor and I have the distance.

At last, I send the distance calculated through serial communication with the `Serial.println()` function.

### Reading the distance in Max 6

`Patcher006_Arduino.maxpat` is the patch related to this distance measurement project. Here it is:

![Reading the distance in Max 6](img/7584_06_25.jpg)

The distance reading patch

As we learnt previously, this patcher contains the whole design pattern to read messages coming from the Arduino board.

The only new thing here is the strange UI element at the bottom. It is called a **slider** . Usually, sliders are used to control things. Indeed, when you click and drag a slider object, it pops out values. It looks like sliders on mixing boards or lighting dimmers, which provide control over some parameters.

Obviously, because I want to transmit a lot of data myself here, I'm using this slider object as a display device and not as a control device. Indeed, the slider object also owns an input port. If you send a number to a slider, the slider takes it and updates its internal current value; it also transmits the value received. I'm only using it here as a display.

Each object in Max 6 has its own parameters. Of course a lot of parameters are common to all objects, but some aren't. In order to check those parameters:

*   Select the object
*   Check the inspector by choosing the **Inspector** tab or typing *Ctrl* + *I* on Windows or *command* + *I* on OS X![Reading the distance in Max 6](img/7584_06_26.jpg)

    The inspector window showing the attributes and properties of the selected slider object

I won't describe all parameters, only the two at the bottom. In order to produce a relevant result, I had to scale the value coming from the `fromsymbol` object. I know the range of values transmitted by the Arduino (though this could require some personal verification), having calculated them from the Sharp datasheet. I considered this range as 20 to 150 cm. I mean a number between 20 and 150.

I took this range and compressed and translated it a bit, using the `scale` object, into a `0-to-100` range of float numbers. I chose the same range for my slider object. Doing that, the result displayed by the slider is coherent and represents the real value.

I didn't write any increment marks on the slider but only made two comments: `near` and `far`. It is a bit poetic in this world of numbers.

Let's check some other examples of sensors able to pop out continuous voltage variations.

## Measuring flexion

Flexi sensors are also very much in use. Where the distance sensor is able to convert a measured distance into voltage, the flexi sensor measures flexion and provides a voltage.

Basically, the device flexion is related to a variable resistance able to make a voltage vary according to the amount of flexion.

![Measuring flexion](img/7584_06_27.jpg)

A standard flexi sensor with two connectors only

A flexi sensor can be used for many purposes.

I like to use it to inform computer through Arduino about door position in digital installations I design. People wanted initially to know only about whether doors are open or closed, but I proposed to use a flexi and got very good information about the angle of openness.

The following figure illustrates how the sensor works:

![Measuring flexion](img/7584_06_28.jpg)

Now, I'm directly giving you the wiring schematic made again with Fritzing:

![Measuring flexion](img/7584_06_29.jpg)

Flexi sensor connected to Arduino board with the pull-down resistor

I put a pull-down resistor. If you didn't read [Chapter 5](ch05.html "Chapter 5. Sensing with Digital Inputs"), *Sensing with Digital Inputs*, about pull-up and pull-down resistors, I suggest you to do that now.

Usually, I use resistors about 10K Ω and they work fine.

The circuit diagram is shown in the following figure:

![Measuring flexion](img/7584_06_30.jpg)

The flexi sensor and its pull-down resistor wired to the Arduino

### Resistance calculations

For this project, I won't give you the code because it is very similar to the previous one, except for the calculation formulas. It is these resistance calculation formulas I'd like to discuss here.

What do we do if we don't have the graph the Sharp Co. was kind enough to include with their infrared sensor? We have to resort to some calculations.

Usually, the flexi sensor documentation provides resistance values for it when it is not bent and when it is bent at 90 degrees. Let's say some usual values of 10K Ω and 20K Ω, respectively.

What are the voltage values we can expect for these resistances values, including the pull-down one too?

Considering the electrical schematic, the voltage at the analog pin 0 is:

![Resistance calculations](img/7584_06_45.jpg)

If we choose the same resistance for the pull-down as the one for the flexi when it is not flexed, we can expect the voltage to behave according to this formula:

![Resistance calculations](img/7584_06_43.jpg)

Obviously, by using the same formula when it isn't bent, we can expect:

![Resistance calculations](img/7584_06_44.jpg)

This means we found our range of voltage values.

We can now convert that into digital 10-bit, encoded values, I mean the famous 0-to-1023 range of Arduino's ADC.

A small, easy calculation provides us with the values:

*   `511` when the voltage is 2.5 (when the flexi isn't bent)
*   `347` when the voltage is 1.7 (when the flexi is bent at around a 90-degree angle)

Because the voltage at Arduino's pin depends on the inverse of the resistance, we don't have a perfectly linear variation.

Experience tells me I can almost approximate this to a linear variation, and I used a scale function in Arduino firmware in order to map `[347,511]` to a simplerange of `[0,90]`. `map(value, fromLow, fromHigh, toLow, toHigh)` is the function to use here.

Do you remember the `scale` object in Max 6? `map()` works basically the same way, but for the Arduino. The statement here would be `map(347,511,90,0)`. This would give a fairly approximated value for the physical angle of bend.

The `map` function works in both directions and can map number segments going in the opposite direction. I guess you begin to see what steps to follow when you have to work with analog inputs on the Arduino.

Now, we are going to meet some other sensors.

## Sensing almost everything

Whatever the physical parameter you want to measure, there's a sensor for it.

Here is a small list:

*   Light color and light intensity
*   Sound volume
*   Radioactivity intensity
*   Humidity
*   Pressure
*   Flexion
*   Liquid level
*   Compass and direction related to magnetic north
*   Gas-specific detection
*   Vibration intensity
*   Acceleration on three axes (x, y, z)
*   Temperature
*   Distance
*   Weight (different for a pure flexion sensor)

It isn't an exhaustive list, but it is quite complete.

Prices are really variable from a few dollars to $50 or $60\. I found one of the cheaper Geiger counters for around $100\. You can find a huge list of companies available on the Internet to buy sensors from in *Appendix G, List of Components' Distributors*.

Now, let's move further. How can we handle multiple analog sensors? The first answer is by wiring everything to many analog inputs of the Arduino. Let's check if we can be smarter than that.

# Multiplexing with a CD4051 multiplexer/demultiplexer

We are going to explore a technique called **multiplexing** . This is a major subchapter because we are going to learn how to make our real-life project more concrete, more real.

We often have many constraints in the real world. One can be the number of Arduinos available. This constraint can also come from having a computer that has only one USB port. Yes, this happens in real life, and I would be lying if I said you can have every connector you want, whenever you want, within the budget you want.

Imagine that you have to plug more than eight sensors to you Arduino's analog input. How would you do it?

We will learn to multiplex signals.

## Multiplexing concepts

Multiplexing is quite common in the telecommunications world. Multiplexing defines techniques providing efficient ways to make multiple signals share a single medium.

![Multiplexing concepts](img/7584_06_31.jpg)

Basic multiplexing concept showing the shared medium

This technique provides a very helpful concept in which you only need one shared medium to bring many channels of information as we can see in the previous figure.

Of course, it involves multiplexing (named mux in the figure) and demultiplexing (demux) processes.

Let's dig into those processes a bit.

## Multiple multiplexing/demultiplexing techniques

When we have to multiplex/demultiplex signals, we basically have to find a way to separate them using physical quantities that we can control.

I can list at least three types of multiplexing techniques:

*   space-division multiplexing
*   frequency-division multiplexing
*   time-division multiplexing

### Space-division multiplexing

This is the easiest to grasp.

![Space-division multiplexing](img/7584_06_32.jpg)

Space-division multiplexing physically agglomerates all wires into the same place

This concept is the basic phone network multiplexing in your flat, for instance.

Your phone wires go out, as those from your neighbors, and all those wires are joined into one shielded, big multipair cable containing, for instance, all phone wires for the whole building in which you live. This huge multipair cable goes into the street, and it is easier to catch it as a single global cable than if you had to catch each cable coming from your neighbors plus yours.

This concept is easily transposable to Wi-Fi communications. Indeed, some Wi-Fi routers today provide more than one Wi-Fi antenna. Each antenna would be able, for instance, to handle one Wi-Fi link. Every communication would be transmitted using the same medium: air transporting electromagnetic waves.

### Frequency-division multiplexing

This type of multiplexing is very common in everything related to DSL and cable TV connections.

Service providers can (and do) provide more than one service on the same cable using this technique.

![Frequency-division multiplexing](img/7584_06_33.jpg)

Frequency-division multiplexing plays with frequencies of transmission and bandwidths

Imagine the **1**, **2**, and **3** frequency bands on the figure would be three different services. 1 could be voice, 2 could be internet, and 3 TV. The reality isn't too far from this.

Of course, what we multiplex at one end, we have to demultiplex at the other in order to address our signals correctly. I wouldn't try to convert a TV modulated signal into voice, but I'm guessing it wouldn't be a very fruitful experience.

### Time-division multiplexing

This is the case we are going to dig into the deepest because this is the one we are going to use with the Arduino to multiplex many signals.

![Time-division multiplexing](img/7584_06_34.jpg)

Time-division multiplexing illustrated with an example of one cycle of four steps

Sequentially, only one channel between the multiplexer and the demultiplexer is fully used for the first signal, then the second, and so on, until the last one.

This kind of system often involves a clock. This helps in setting the right cycle for each participant so they know at which step of communication we are. It is critical that we preserve the safety and integrity of communications.

Serial communications work like that, and for many reasons—even if you think you know them a lot after previous chapters—we'll dig a bit deeper into them in the next chapter.

Let's check how we can deal with eight sensors and only one analog input for our Arduino board.

## The CD4051B analog multiplexer

The CD4051B analog multiplexer is a very cheap one and is very useful. It is basically an analog and digital multiplexer and demultiplexer. This doesn't mean you can use it as a multiplexer and a demultiplexer at the same time. You have to identify in what case you are and wire and design the code for this proper case. But it is always useful to have a couple of CD4051B devices.

Used as a multiplexer, you can connect, say eight potentiometers to the CD4051B and only one Arduino analog input, and you'll be able, by code, to read all 8 values.

Used as a demultiplexer, you could write to eight analog outputs by writing from only one Arduino pin. We'll talk about that a bit later in this book, when we approach the output pin and especially the **pulse-width modulation** (**PWM**) trick with LEDs.

### What is an integrated circuit?

An **integrated circuit** (**IC**) is an electronic circuit miniaturized and all included in a small box of plastic. This is the simplest definition.

Basically, we cannot talk about integrated circuits without bringing to mind their small size. It is one of the more interesting features of IC.

The other one is what I am naming the **black box abstraction** . I also define it like the programming-like classes of the hardware world. Why? Because you don't have to know exactly how it works but only how you can use it. It means all the circuits inside don't really matter if the legs outside make sense for your own purpose.

Here are two among several type of IC packages:

*   **Dual in-line package** (**DIP**, also named **DIL**)
*   **Small** **outline** (**SO**)

You can find a useful guide at [http://how-to.wikia.com/wiki/Guide_to_IC_packages](http://how-to.wikia.com/wiki/Guide_to_IC_packages).

The more commonly used of the two ICs are definitely DIPs. They are also called through-holes. We can easily manipulate and plug them into a breadboard or **printed** **circuit board** (**PCB**).

SO requires more dexterity and finer tools.

### Wiring the CD4051B IC?

The first question is about *what* it looks like? In this case, the answer is that it looks like a DIP package.

![Wiring the CD4051B IC?](img/7584_06_35.jpg)

The CD4051B DIP case version

Here is the face of this nice little integrated circuit. The datasheet is easy to find on the Internet. Here is one by Texas Instruments:

[http://www.ti.com/lit/ds/symlink/cd4051b.pdf](http://www.ti.com/lit/ds/symlink/cd4051b.pdf)

I redrew the global package in the next figure.

![Wiring the CD4051B IC?](img/7584_06_36.jpg)

A schematic of the CD4051B with all pin descriptions

#### Identifying pin number 1

It is easy easy to find out which pin is pin number 1\. As standard, there is a small circle engraved in front of one of the corner pins. This is the pin number 1.

There is also a small hole shaped as a half circle. When you place the IC with this half circle at the top (as shown on the previous figure), you know which pin number 1 is; the first pin next to pin number 1 is pin number 2, and so on, until the last pin of the left column which, in our case, is pin number 8\. Then, continue with the pin opposite to the last one in the left column; this is pin number 9, and the next pin is pin number 10, and so on, until the top of the right column.

![Identifying pin number 1](img/7584_06_37.jpg)

Numbering the pins of an IC

Of course, it would be much too simple if the first input was pin 1\. The only real way you can know for sure is to check the specs.

### Supplying the IC

The IC itself has to be supplied. This is to make it active but also, in some cases, to drive the current too.

*   Vdd is the positive supply voltage pin. It has to be wired to the 5 V supply.
*   Vee is the negative supply voltage pin. Here, we'll wire it to Ground.
*   Vss is the ground pin, connected to Ground too.

### Analog I/O series and the common O/I

Check the order of the I and the O in this title.

If you choose to use the CD4051B as a multiplexer, you'll have multiple analog inputs and one common output.

On the other hand, if you choose to use it as a demultiplexer, you'll have one common input and multiple analog outputs.

How does the selection/commutation work? Let's check the selector's digital pins, A, B, and C.

### Selecting the digital pin

Now comes the most important part.

There are three pins, named A (pin 11), B (pin10), and C (pin 9), that have to be driven by digital pins of the Arduino. What? Aren't we in the analog inputs part? We totally are, but we'll introduce a new method of control using these three selected pins.

The multiplexing engine under the hood isn't that hard to understand.

Basically, we send some signal to make the CD4051B commute the inputs to the common output. If we wanted to use it as a demultiplexer, the three selected pins would have to be controlled exactly in the same way.

In the datasheet, I found a table of truth. What is that? It is just a table where we can check which A, B, and C combinations commute the inputs to the common output.

The following table describes the combination:

![Selecting the digital pin](img/7584_06_38.jpg)

The truth table for the CD4051B

In other words, it means that, if we write 1 to the digital output on Arduino corresponding to A, 1 to that corresponding to B and 0 to that corresponding to C, the commuted input would be the third channel.

Of course, there is something good in this. If you *read* the binary number corresponding to the inputs on C, B, and A (in that order), you'll have a nice surprise; it will be equivalent to the decimal number of the input pin commuted by the common output.

Indeed, 0 0 0 in binary equals 0 in decimal. Refer the table for the binary values of decimal numbers:

| 0 0 0 | 0 |
| 0 0 1 | 1 |
| 0 1 0 | 2 |
| 0 1 1 | 3 |
| 1 0 0 | 4 |
| 1 0 1 | 5 |
| 1 1 0 | 6 |
| 1 1 1 | 7 |

Here is how we could wire things:

![Selecting the digital pin](img/7584_06_39.jpg)

The circuit including the CD4051B multiplexer with its common output wired to the analog pin 0

And the following figure is the electrical diagram:

![Selecting the digital pin](img/7584_06_40.jpg)

The electrical diagram

All devices we'd like to read with this system should be wired to I/O 0, 1, 2, and so on, on the CD4051B.

Considering what we know about table of truth and how the device works, if we want to read sequentially all pins from 0 to 7, we will have to make a loop containing both types of statements: 

*   One for commuting the multiplexer
*   One for reading the Arduino analog input 0

The source code would look like this (you can find it in the `Chapter6/analogMuxReader` folder):

[PRE5]

After you've defined all the variables, we set up the serial port in `setup()` and also the three pins related to the selector pin of the CD4051B as outputs. Then, in each cycle, I first select the commuted input by either driving the current or not to pins A, B, and C of the CD4051B. I'm using a nested function in my statement in order to save some rows.

`bitRead(number,n)` is a new function able to return the *nth* bit of a number. It is the perfect function for us in our case.

We make a loop over the input commuted from 0 to 7, more precisely to `devicesNumber - 1`.

By writing those bits to pins A, B, and C of the CD4051B device, it selects the analog input at each turn and pops the value read at the serial port for further processing in Processing or Max 6 or whatever software you want to use.

# Summary

In this chapter, we learnt at least how to approach a very powerful graphical framework environment named Max 6\. We'll use it in several further examples in this book as we continue to use Processing too.

We learnt some reflexes for when we want to handle sensors providing continuous voltage variations to our Arduino analog inputs.

Then, we also discovered a very important technique, the multiplexing/demultiplexing.

We are going to talk about it in the next chapter about serial communication. We'll dig deeper into this type of communication now that we have used a lot of time already.