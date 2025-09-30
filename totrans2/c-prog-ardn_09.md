# Chapter 9. Making Things Move and Creating Sounds

If the Arduino board can listen and feel with sensors, it can also react by making things move.

By the movement concept, I mean both of the following:

*   Object movements
*   Air movements producing sounds

We are going to learn how we can control small motors named **servo**, and how we can deal with high-current control by using transistors.

Then we'll start talking about the basics of sound generation. This is a requirement before trying to produce any sounds, even the simplest ones. This is the part where we'll describe analog and digital concepts.

At last, we'll design a very basic random synthesizer controllable using MIDI. We'll also introduce a very nice library called **PCM** that provides a simple way to add sample playing features to your 8-bit microcontroller.

# Making things vibrate

One of the simplest projects we can introduce here is the use of a small piezoelectric sensor.

This is the first basic tangible action we design here. Of course, we already designed many of the visual feedback, but this is our first real-world object that moves the firmware.

This kind of feedback can be very useful in nonvisual contexts. I designed a small project for a person who wanted to send a feedback to visitors in his reactive installation. The visitor had to put on a t-shirt that included some electronics attached, such as a LilyPad and some piezoelectric sensors. The LED feedback wasn't the solution we used before to send feedback to the wearer, and we decided to send a vibration. These piezoelectric sensors were distributed on each side of the t-shirt to produce different feedback in response to different interactions.

But wouldn't I have made a mistake talking about sensors vibrating?

## The piezoelectric sensor

A piezoelectric sensor is a component that uses the piezoelectric effect.

This effect is defined as the linear electromechanical interaction between the mechanical and electrical state in some specific materials.

Basically, a mechanical action on this device generates electricity, making it usable for movement and vibration detection. But the nice thing here is that the effect is reciprocal—if you apply a current to it, it will vibrate.

This is why we are using a piezoelectric sensor here. We are using it as a vibration generator.

Piezoelectric sensors are also often used as a tone generator. We will dig deeper into the relationship between air vibrations and sound a bit later, but it is important to mention it here too.

## Wiring a vibration motor

Piezoelectric sensors usually consume around 10 mA to 15 mA, which is very small.

Of course, you need to check the proper datasheet of the device you are going to use. I have had good results with the one from **Sparkfun** ([https://www.sparkfun.com/products/10293](https://www.sparkfun.com/products/10293)). The wiring could not be simpler—there are only two legs. The following image shows how the piezoelectric sensor/vibrator is wired to Arduino via a PWM-capable digital pin:

![Wiring a vibration motor](img/7584_09_001.jpg)

Please note that I have wired the piezoelectric device to a PWM-capable digital pin. I explained PWM in the previous chapter.

Here is the circuit schematic. This piezoelectric component is displayed as a small buzzer/speaker:

![Wiring a vibration motor](img/7584_09_002.jpg)

Of course, since we are going to use PWM, it means that we are going to simulate an analog output current. Considering the duty-cycle concept, we can feed the piezoelectric device using the `analogWrite()` function and then feed it with different voltages.

## Firmware generating vibrations

Check the firmware. It is also available in the `Chapter09/vibrations/` folder.

[PRE0]

We are using the `analogWrite()` function here again. This function takes the digital pin as an argument and value. This value from 0 to 255 is the duty cycle. It basically simulates an analog output.

We use it the usual way with the `incdec` (stands for increment-decrement) parameter. We store the increment value we want to use at each `loop()` execution.

This increment changes when the value reaches its boundaries, 0 or 255, and is inverted, providing a cheap way to make a cycle from 0 to 255, then to 0, then to 255, and so on.

This firmware makes the piezoelectric device vibrate cyclically from a low rate to a higher rate.

Let's control bigger motors now.

# Higher current driving and transistors

We talked about transistors in the previous chapter. We used them as digital switches. They can also be used as amplifiers, voltage stabilizers, and many other related applications.

You can find transistors almost everywhere and they are quite cheap. You can find the complete datasheet at [http://www.fairchildsemi.com/ds/BC/BC547.pdf](http://www.fairchildsemi.com/ds/BC/BC547.pdf).

The following is a basic diagram explaining how transistors work:

![Higher current driving and transistors](img/7584_09_003.jpg)

The transistor used as a digital switch in logical circuits

A transistor has the following legs:

*   The collector
*   The base
*   The emitter

If we saturate the base by applying a 5 V power supply to it, all the current coming from the collector will be transmitted through the emitter.

When used like this, the NPN transistor is a nice way to switch on/off high current that Arduino wouldn't have been able to drive by itself. By the way, this switch is controllable with Arduino because it only requires a very small amount of current to be provided to the base of the transistor.

### Note

Sending 5 V to the transistor base closes the circuit. Putting the transistor base to ground opens the circuit.

In any case, where you need to have an external power supply to drive motors, we use this kind of design pattern.

Let's now learn about small current servos and then move further using transistors.

# Controlling a servo

A **servomotor** is also defined as a rotary actuator that allows for very fine control of angular positions.

Many servos are widely available and quite cheap. I have had nice results with a 43 R servo, by Spring Model Electronics. You can find the datasheet at [http://www.sparkfun.com/datasheets/Robotics/servo-360_e.pdf](http://www.sparkfun.com/datasheets/Robotics/servo-360_e.pdf).

Servos can drive a great amount of current. This means that you wouldn't be able to use more than one or two on your Arduino board without using an external source of power.

## When do we need servos?

Whenever we need a way to control a position related to a rotation angle, we can use servos.

Servos can not only be used to move small parts and make objects rotate, but can also be used to move the object including them. Robots work in this fashion, and there are many Arduino-related robot projects on the Web that are very interesting.

In the case of robots, the servo device case is fixed to a part of an arm, for instance, and the other part of the arm is fixed to the rotating part of the servo.

## How to control servos with Arduino

There is a nice library that should be used at first, named `Servo`.

This library supports up to 12 motors on most Arduino boards and 48 on the Arduino Mega.

By using other Arduino boards over Mega, we can figure out some software limitations. For instance, pins 9 and 10 cannot be used for PWM's `analogWrite()`method ([http://arduino.cc/en/Reference/analogWrite](http://arduino.cc/en/Reference/analogWrite)).

Servos are provided in three-pin packages:

*   5 V
*   Ground
*   Pulse; that is, control pin

Basically, the power supply can be easily provided by an external battery, and the pulse still remains the Arduino board.

Let's check the basic wiring.

## Wiring one servo

The following diagram is that of a servo wired to an Arduino for both power supply and control:

![Wiring one servo](img/7584_09_004.jpg)

The corresponding circuit diagram is as follows:

![Wiring one servo](img/7584_09_005.jpg)

One servo and Arduino

We are basically in a very common digital output-based control pattern.

Let's check the code now.

## Firmware controlling one servo using the Servo library

Here is a firmware that provides a cyclic movement from 0 degrees to 180 degrees. It is also available in the `Chapter09/OneServo/` folder.

[PRE1]

We first include the `Servo` library header.

Then we instantiate a `Servo` object instance named `myServo`.

In the `setup()` block, we have to make something special. We attach pin 9 to the `myServo` object. This explicitly defines the pin as the control pin for the `Servo` instance `myServo`.

In the `loop()` block, we have two `for()` loops, and it looks like the previous example with the piezoelectric device. We define a cycle, progressively incrementing the angle variable from 0 to 180 and then decrementing it from 180 to 0, and each time we pause for 20 ms.

There is also a function not used here that I want to mention, `Servo.read()`.

This function reads the current angle of the servo (that is, the value passed to the last call to `write()`). This can be useful if we are making some dynamic stuff without storing it at each turn.

# Multiple servos with external power supply

Let's imagine we need three servos. As explained before, servos are motors, and motors convert current into movement, driving more current than other kinds of devices such as LEDs or sensors.

If your Arduino project requires a computer, you can supply power to it with the USB as long as you don't go beyond the 500 mA limit. Beyond this, you'd need to use an external power supply for some or all parts of your circuit.

Let's see how it goes with three servos.

## Three servos and an external power supply

An external power supply can be batteries or a wall adapter power supply.

We are going to use basic AA batteries here. This is also a way to supply Arduino if you don't need a computer and want Arduino to be autonomous. We will consider this option in the third part of this book about more advanced concepts.

Let's check the wiring for now:

![Three servos and an external power supply](img/7584_09_006.jpg)

Three servos wired to an Arduino, and power supplied by two AA batteries

In cases like this, we have to wire the grounds together. Of course, there is only one current source supply for the servos—the two AA batteries.

Let's check the circuit diagram:

![Three servos and an external power supply](img/7584_09_007.jpg)

Three servos, two AA batteries, and an Arduino

## Driving three servos with firmware

Here is an example of firmware for driving three servos:

[PRE2]

This very minimal firmware is also available in the `Chapter09/Servos/` folder.

We first instantiate our three servos and attach one pin for each in the `setup()` block.

In `loop()`, we play with angles. As a new approach for generative creation, I defined one variable only for the angle. This variable cyclically goes from 0 to 180 in each `loop()` turn.

The servo attached to pin 9 is driven with the angle value itself.

The servo attached to pin 10 is driven with the value [135-(angle/2)], varying itself from 135 to 45.

Then, the servo attached to pin 11 is driven with the value [180-angle], which is the opposite movement of the servo attached to pin 9.

This is also an example to show you how we can easily control one variable only, and program variations around this variable each time; here, we are making angles vary and we are combining the angle variable in different expressions.

Of course, we could control the servo position by using an external parameter, such as a potentiometer position or distance measured. This will combine concepts taught here with those in [Chapter 5](ch05.html "Chapter 5. Sensing with Digital Inputs"), *Sensing with Digital Inputs*, and [Chapter 6](ch06.html "Chapter 6. Sensing the World – Feeling with Analog Inputs"), *Sensing the World–Feeling with Analog Inputs*.

Let's learn a bit more about step motors.

# Controlling stepper motors

**Stepper motor** is the common name for a **step motor**. They are motors that are controllable using small steps.

The full rotation is divided into a number of equal steps and the motors' positions can be controlled to move and hold at one of these steps easily with a high degree of accuracy, without any feedback mechanism.

There are a series of electromagnetic coils that can be charged positively or negatively in a specific sequence. Controlling the sequence provides control about the movement, forward or backward in small steps.

Of course, we can do that using Arduino boards.

We are going to examine the unipolar stepper here.

## Wiring a unipolar stepper to Arduino

Unipolar steppers usually consist of a center shaft part and four electromagnetic coils. We call them unipolar because power comes in through one pole. We can draw it as follows:

![Wiring a unipolar stepper to Arduino](img/7584_09_010.jpg)

A six-pin unipolar step motor

Let's check how it can be wired to our Arduino.

We need to supply power to the stepper from an external source. One of the best practices here is the use of a wall adapter. Pins 5 and 6 have to be fed a source of current.

Then, we need to control each pin from 1 to 4 with the Arduino. This will be done using the sink current system ULN2004, which is very similar to ULN2003 which we used in the previous chapter with our LED matrix. ULN2004 is suited for voltage from 6 V to 15 V. When ULN2003 is 5 V, the stepper datasheet shows that we have to use this system instead of ULN2003.

![Wiring a unipolar stepper to Arduino](img/7584_09_008.jpg)

A unipolar stepper connected to Arduino through the Darlington transistor array, ULN2004

Let's check the corresponding circuit diagram:

![Wiring a unipolar stepper to Arduino](img/7584_09_009.jpg)

A circuit diagram showing Arduino, the ULN2004 Darlington transistors array, and the stepper

We are using an external power supply here again. All the grounds are wired together too.

Please notice that the **COM** pin (pin number 9) has to be wired to the power supply source (+V).

If you remember correctly from the previous chapter, when we fed an input of the ULN200x Darlington Transistor array, the corresponding output sinks the current to the ground.

In our case here, each pin of Arduino connected to the ULN2004 shift register can commute each pin of the stepper to the ground.

Let's design firmware for stepper control.

## Firmware controlling the stepper motor

There is a very nice library that can save us from providing sequences of the HIGH and LOW pins, considering the movements we want to drive.

In order to control precise movements, we normally have to deal with specific sequences. These sequences are usually described in the datasheet.

Let's check the one available at [http://www.sparkfun.com/datasheets/Robotics/StepperMotor.pdf](http://www.sparkfun.com/datasheets/Robotics/StepperMotor.pdf).

Sparkfun Electronics provides it for a model designed by Robotics.

We can see a table similar to the following one, named **Drive Sequence Model**:

| STEP | A | B | C | D |
| --- | --- | --- | --- | --- |
| 1 | HIGH | HIGH | LOW | LOW |
| 2 | LOW | HIGH | HIGH | LOW |
| 3 | LOW | LOW | HIGH | HIGH |
| 4 | HIGH | LOW | LOW | HIGH |

If you want to make a clockwise rotation, you should generate a sequence from 1 to 4, then 1, and so on, cyclically. Counterclockwise rotations require generating sequences from 4 to 1 and so on.

Instead of writing a lot of sequences like these, with some function, we can directly use the library named `Stepper`, which is now included in Arduino Core.

Here is the code, followed by the discussion. It is also available in the `Chapter09/StepperMotor/` folder.

[PRE3]

We first include the `Stepper` library.

Then we define the number of steps that are equivalent to one whole turn. In our datasheet, we can see that the first step is an angle of 1.8 degrees, with a 5 percent error room. We won't consider that error; we will take 1.8 degrees. This means we need 200 steps (200 * 1.8 = 360°) in order to make a whole turn.

We then instantiate a `Stepper` object by pushing five arguments, which are the step numbers for a whole turn, and the four pins of the Arduino wired to the stepper.

We then declare two helper variables for tracing and, sometimes, changing the rotation direction.

In the `setup()` block, we usually define the speed of the current instance handling the stepper. Here, I have set `30` (which stands for 30 rounds per minute). This can also be changed in the `loop()` block, considering specific conditions or whatever.

At last, in the `loop()` block, we move the stepper to an amount equal to the multiplier value, which is initially `1`. This means that at each run of the `loop()` method, the stepper rotates from step 1 (that is, 1.8 degrees) in the clockwise direction.

I added a logic test, which checks each time if the counter has completed the number of steps required to make a whole turn. If it hasn't, I increment it; otherwise, as soon as it reaches the limit (that is, the motor makes a whole turn since the beginning of the program execution), I reset the counter and invert the multiplier in order to make the stepper continue its walk, but in the other direction.

This is another pattern that you should keep in mind. These are all small patterns that will give you a lot of cheap and efficient ideas to use in each one of your future projects.

With servos and steppers, we can now make things move.

In some of my projects, I used two steppers, with one string bound to each and both these strings bound to a hanging pencil. We can draw on a wall by controlling the amount of string hanging on each side.

# Air movement and sounds

Making the air move can generate nice audible sounds, and we are going learn a bit more about this in the following sections.

If you can make things move with Arduino, you will probably be able to make the air move too.

In fact, we have already done this, but we probably didn't move it enough to produce a sound.

This part is just a short introduction to some definitions and not a complete course about sound synthesis. These are the basic elements that we will use in the next few sections of the book, and as far as possible there will be references of websites or books provided that you can refer to if you are interested in learning more about those specific parts.

## What is sound actually?

Sound can be defined as a mechanical wave. This wave is an oscillation of pressure and can be transmitted through solid, liquid, or gas. By extension, we can define sound as the audible result of these oscillations on our ear.

Our ear, combined with further brain processes, is an amazing air-pressure sensor. It is able to evaluate the following:

*   Amplitude of a sound (related to the amount of air moving)
*   Frequency of a sound (related to the air oscillation amount)

Of course, all these processes are real time, assuming higher or lower frequencies mix at this particular moment.

I'd really suggest that you read the amazing and efficient introduction to *How Digital Audio Works?*, by cycling 74, the maker of the Max 6 framework. You can read it online at [http://www.cycling74.com/docs/max6/dynamic/c74_docs.html#mspdigitalaudio](http://www.cycling74.com/docs/max6/dynamic/c74_docs.html#mspdigitalaudio).

A sound can contain more than one frequency, and it is generally a combination of the frequency content and the global perception of each frequency amplitude that gives the feeling of what we call the timbre of a sound. Psychoacoustics studies the perception of sound.

## How to describe sound

We can describe sound in many ways.

Usually, there are two representations of sound:

*   Variation of the amplitude over time. This description can be put on a graph and defined as a time-domain representation of sounds.
*   Variation of the amplitude depending on the frequency content. This is called the frequency-domain representation of sounds.

There is a mathematical operation that provides an easy way to pass from one to the other, known as the Fourier transform ([http://en.wikipedia.org/wiki/Fast_Fourier_transform](http://en.wikipedia.org/wiki/Fast_Fourier_transform)). Many implementations of this operation are available on computers, in the form of the **Fast Fourier Transform** (**FFT**), which is an efficient method that provides fast approximate calculations.

Let's consider a sinusoidal variation of air pressure. This is one of the most simple sound waves.

Here are the two representations in the two domains:

![How to describe sound](img/7584_09_011.jpg)

Two representations of the same elementary sound produced by a sinusoidal variation of air pressure

Let's describe the two graphs of the preceding image.

In the time-domain representation, we can see a cyclical variation with a period. The period is the time equivalent of the spatial wavelength.

The period is the time needed to complete a complete vibrational cycle. Basically, if you can describe the variation over a period, you are able to totally draw the representation of the sound in time. Here, it is a bit obvious because we are watching a pure sine-based sound.

If you draw and observe a sound produced by a source, the amplitude variation over time will correspond directly to a variation of air pressure.

Considering the orientation of the axis, we first have what we call a high-pressure front. This is the part of the curve above zero (represented by the time axis). This means that the pressure is high and our tympanum is pushed a bit more inside our ear.

Then, after a semi-period, the curve crosses zero and goes below, meaning that the air pressure is lower than the normal atmospheric pressure. Our tympanum also feels this variation. It is pulled a little bit.

In the frequency-domain representation, there is only a vertical line. This pulse-like graph in the previous figure represents the unique frequency contained in this sine-based sound. It is directly related to its period by a mathematical equation, as follows:

![How to describe sound](img/7584_09_inline01.jpg)

Here, `T` is the period in seconds and `f` is the frequency in Hertz.

The higher the frequency, the more the sound is felt as high-pitched. The lesser it is, the more the sound is felt as low-pitched.

Of course, a high frequency means a short period and faster oscillations over time.

These are the basic steps in understanding how sound can be represented and felt.

## Microphones and speakers

Microphones are devices that are sensitive to the subtle variation of air pressure. Yes, they are sensors. They can translate air-pressure variations into voltage variations.

Speakers are devices that implement a part that can move, pushing and pulling masses of air, making it vibrate and produce sounds. The movement is induced by voltage variations.

In both these cases, we have:

*   A membrane
*   An electrical transducer system

In the microphone case, we change the air pressure and that produces an electrical signal.

In the speaker case, we change the electrical signal and that produces an air pressure variation.

In each case, we have analog signals.

## Digital and analog domains

Sounds sources can be very different. If you knock on a table, you'll hear a sound. This is a basic analog- and physical-based sound. Here, you physically make the table vibrate a bit, pushing and pulling air around it; and because you are near it, your tympanum feels these subtle variations.

As soon as we talk about digital equipment, we have some limitations considering storage and memory. Even if these are large and sufficient now, they aren't infinite.

And how can we describe something analog in that case? We already spoke about this situation when we described analog and digital input and output pins of Arduino.

### How to digitalize sound

Imagine a system that could sample the voltage variation of your microphones periodically. A sampling concept usually used is sample and hold.

The system is able to read the analog value at regular intervals of time. It takes a value, holds it as a constant until the next value, and so on.

We are talking about the sampling rate to define the sampling frequency. If the sampling rate is low, we will have a lower approximation of the analog signal than if what we would have had if the sampling rate was high.

A mathematical theorem provides us a limit that we have to keep in mind—the Nyquist frequency.

In order to keep our sampling system process a safe artifact induced by the system itself, we have to sample at a minimum of two times the higher frequency in our original analog signal.

![How to digitalize sound](img/7584_09_012.jpg)

Example illustrating the sampling rate while sampling a sine wave

A higher sampling rate not only means more precision and fidelity to the original analog wave, but also more points to store in the digital system. The result would be a heavier file, in terms of disks and filesystems.

Another element to keep in mind while sampling is the bit depth.

I voluntarily omitted it in the previous figure in order to not overload the drawings.

Indeed, we sampled a value over time, but how can you represent the value itself, the amplitude I mean? We use a bit-based coding system, as usual, with the digital equipment.

The **bit depth** is the resolution of the amplitude values from `-1` (the minimum possible) to `1` (the maximum possible).

The higher the bit depth, the more the subtle variations we can encode and record into our digital systems. Conversely, if we have a very low bit-depth sampler and we make a progressively decreasing amplitude variation, the sound will decrease considerably in a manner similar to the Doppler effect. For instance, we wouldn't be able to distinguish values from `0.5` to `0.6`; everything would only be `0.5` or `0.7` but never `0.6`. The sound would lose subtlety.

Usual sampling rates and bit depth depends on the purpose of the final rendering.

Here are two commonly used quality standards:

*   CD quality is 44.1 kHz and 16-bit
*   DAT quality is 48 kHz and 16-bit

Some recording and mastering studios use audio interfaces and internal processing at 96 kHz and 24 bits. Some people who love old-school sound engines still use lo-fi systems to produce their own sound and music at 16 kHz and 8 bits.

The process from analog to digital conversion is handled by the **analog to digital converter** (**ADC**). Its quality is the key to achieving good conversion. This process is similar to the one involved in Arduino when we use an analog input. Its ADC is 10 bits and it can read a value once every 111 microseconds, which is a sampling rate frequency of 9 kHz.

Buffers are used to smoothly process times and make things smoother in time.

### How to play digital bits as sounds

We can also convert digital encoded sounds into analog sounds. This process is achieved by the **digital to analog converter** (**DAC**).

If the processor sends bits of data from the encoded sound to the DAC as a continuous flow of discrete values, the DAC takes all these values and converts them as an analog electrical signal. It interpolates values between each digital value, which often involves some processes (for example, low-pass filtering), in order to remove some artifacts such as harmonics above the Nyquist frequency.

In the world of digital audio, DAC power and quality is one of the most important aspects of our audio workstation. They have to provide high resolutions, a high sampling rate, a small total harmonic distortion and noise, and a great dynamic range.

## How Arduino helps produce sounds

Let's come back to Arduino.

Arduino can read and write digital signals. It can also read analog signals and simulate analog output signals through PWM.

Wouldn't it be able to produce and even listen to sounds? Of course it would.

We can even use some dedicated components to make things better. For instance, we can use an ADC with a higher sampling rate in order to store sounds and a high-quality DAC too, if required. Today, we often use electronic hardware equipment to control software. We can, for instance, build a device based on Arduino, full of knobs and buttons and interface it with a software on the computer. This has to be mentioned here.

We can also use Arduino as a sound trigger. Indeed, it is quite easy to turn it into a small sequencer, popping out specific MIDI or OSC messages to an external synthesizer, for instance. Let's move further and go deeper into audio concepts specifically with the Arduino board.

# Playing basic sound bits

Playing a sound requires a sound source and a speaker. Of course, it also requires a listener who is able to hear sounds.

Natively, Arduino is able to produce 8 kHz and 8-bit audio playback sounds on small PC speakers.

We are going to use the `tone()` function available natively in the Arduino Core. As written at [http://arduino.cc/en/Reference/Tone](http://arduino.cc/en/Reference/Tone), we have to take care of the pins used when using this function, because it will interfere with PWM output on pins 3 and 11 (except for the Arduino MEGA).

This technique is also named **bit-banging**. It is based on I/O pin toggling at a specific frequency.

## Wiring the cheapest sound circuit

We are going to design the cheapest sound generator ever with a small 8-ohm speaker, a resistor, and an Arduino board.

![Wiring the cheapest sound circuit](img/7584_09_013.jpg)

A small sound generator

The connections made here ensure an audible sound. Let's program the chip now.

The corresponding circuit diagram is as follows:

![Wiring the cheapest sound circuit](img/7584_09_014.jpg)

The diagram of the sound generator

## Playing random tones

As a digital artist and specifically as an electronic musician, I like to be free of the notes. I often use frequencies instead of notes; if you are interested, you can read about the microtonal concept at [http://en.wikipedia.org/wiki/Microtonal_music](http://en.wikipedia.org/wiki/Microtonal_music).

In this example, we don't use notes but frequencies to define and trigger our electronic music.

The code is also available in the `Chapter09/ ToneGenerator/` folder.

[PRE4]

We initialize the pseudorandom number generator at first by reading the analog input `0`.

In the loop, we generate two numbers:

*   The pitch is a number from 30 to 4,999; this is the frequency of the sound
*   The duration is a number from 1 ms to 1 s; this is the duration of the sound

These two arguments are required by the `tone()` function.

Then, we call `tone()`. The first argument is the pin where you feed the speaker.

The `tone()` function generates a square wave of the specified frequency on a pin as explained in its reference page at [http://arduino.cc/en/Reference/Tone](http://arduino.cc/en/Reference/Tone).

If we don't provide a duration, the sound continues until the `noTone()` function is called. The latter takes an argument that was used by the pin as well.

Now, listen to and enjoy this microtonal pseudorandom melody coming from your 8-bit chip.

# Improving the sound engine with Mozzi

The bit-banging technique is very cheap and it's nice to learn how it works. However, I can quote some annoying things here:

*   **No pure sound**: Square waves are a sum of all odd harmonics at the fundamental frequency
*   **No amplitude control available**: Each note sounds at the same volume

We are going to use a very nice library called Mozzi, by Tim Barrass. The official website is directly hosted on GitHub at [http://sensorium.github.com/Mozzi/](http://sensorium.github.com/Mozzi/). It includes the `TimerOne` library, a very fast timer handler.

Mozzi provides a very nice 16,384 kHz, 8-bit audio output. There is also a nice basic audio toolkit containing oscillators, samples, lines and envelopes, and filtering too.

Everything is available without external hardware and by only using two pins of the Arduino.

We are going to design a small sound engine based on it.

## Setting up a circuit and Mozzi library

Setting up the circuit is easy; it is the same as the latest one except that pin 9 has to be used.

Mozzi's documentation says:

> To hear Mozzi, connect a 3.5 mm audio jack with the centre wire to the PWM output on Digital Pin 9* on Arduino, and the black ground to the Ground on the Arduino. Use this as a line out which you can plug into your computer and listen to with a sound program like Audacity.
> 
> It is really easy to set up the hardware. You can find many 3.5 mm audio jack connector like that all over the Internet. In the following circuit diagram, I put a speaker instead of a jack connector but it works exactly the same with a jack connector, that latter having 2 pins, one ground and one signal related. Ground has to be connected to the Arduino's ground and the other pin to the digital pin 9 of the Arduino.
> 
> Then we have to install the library itself.
> 
> Download it from their website: [http://sensorium.github.com/Mozzi](http://sensorium.github.com/Mozzi)
> 
> Unzip it and rename the folder as Mozzi.
> 
> Then put it as usual in the place you put your libraries; in my case it is:
> 
> /Users/julien/Documents/Arduino/libraries/
> 
> Restart or just start your Arduino IDE and you'll be able to see the library in the IDE.
> 
> It is provided with a bunch of examples.
> 
> We are going to use the one about the sine wave.

This is what the Mozzi library looks like:

![Setting up a circuit and Mozzi library](img/7584_09_015.jpg)

A Mozzi installation revealing a lot of examples

## An example sine wave

As with any library, we have to learn how to use the sine wave.

There are a lot of examples, and these are useful to learn how to design our own firmware step-by-step. Obviously, I won't describe all these examples, but only those in which I'll grab elements to make your own sound generator.

Let's check the sine wave example. It is also available in the `Chapter09/ MozziSoundGenerator/` folder.

[PRE5]

At first, some inclusions are done.

`MozziGuts.h` is the basic header to include in any case.

`Oscil.h` is the header to use if you need an oscillator.

We then include a wave table (sine wave).

### Oscillators

In the sound synthesis world, an **oscillator** is a basic unit that is capable of producing oscillations. It is often used not only for direct sound generation with frequencies varying from 20 Hz to 20 kHz (audible spectrum), but also as a modulator (usually with frequencies lower than 50 Hz). It has been used as the latter in this case. An oscillator is usually called a **Low Frequency Oscillator** (**LFO**).

### Wavetables

A **wavetable** is a very nice and efficient way to store whole pieces of sounds, generally cyclical or looped sounds.

We basically used this as a lookup table. Do you remember using it?

Instead of calculating our sine value over time in real time, we basically precalculate each value of a whole period, and then add the results into a table; each time we need it, we just have to scan the table from the beginning to the end to retrieve each value.

Of course, this IS definitely an approximation. But it saves a lot of CPU work.

A wavetable is defined by its size, the sample rate related, and of course the whole values.

Let's check what we can find in the `sin2048_int8.h` file:

![Wavetables](img/7584_09_016revised.jpg)

We can indeed find the number of cells: 2048 (that is, there are 2048 values in the table). Then, the sample rate is defined as 2048.

Let's go back to the example.

We then define the Oscil object that creates an oscillator.

After the second `define` keyword related to the variable update frequency, we have the usual structure of `setup()` and `loop()`.

We also have `updateControl()` and `updateAudio()` and those aren't defined in the code. Indeed, they are related to Mozzi and are defined in the library files themselves.

The `setup()` block starts the Mozzi library at the specific control rate defined before. Then, we set up the oscillator defined before at a frequency of 440 Hz. 440 Hz is the frequency of the universal A note. In this context, it can be thought of as the audio equivalent of the Hello World example.

Nothing more about `updateControl()` here.

We return `aSin.next()` in `updateAudio()`. It reads and returns the next sample, which is understood as the next element, which is the next bit of sound.

In `loop()`, we call the `audioHook()` function.

The global pattern is usual. Even if you use another library related to sound, inside or outside the Arduino world, you'll have to deal with this kind of pattern in four steps (generally, but it may differ):

*   Definitions in the header with some inclusions
*   Start of the audio engine
*   Permanent loop of a hook
*   Updating functions for rendering things before a commit, then in the hook

If you upload this, you'll hear a nice A440 note, which may make you hum a little.

## Frequency modulation of a sine wave

Let's now merge some concepts—sine wave generation, modulation, and input reading.

We are going to use two oscillators, one modulating the frequency of the other.

With a potentiometer, we can control the frequency of the modulating oscillator.

Let's first improve the circuit by adding a potentiometer.

### Adding a pot

In the following circuit diagram, we have added a potentiometer in the sound generator circuit:

![Adding a pot](img/7584_09_017.jpg)

The circuit diagram is as follows:

![Adding a pot](img/7584_09_018.jpg)

Improving the sound generator

### Upgrading the firmware for input handling

This code is also available in the `Chapter09/MozziFMOnePot/` folder.

[PRE6]

In this example, we use two oscillators, both based on a cosine wavetable:

*   `aCos` stands for the sound itself
*   `aVibrato` is the modulator

Since we have a potentiometer here, we need to scale things a bit.

`intensityMax` is the maximum intensity of the modulation effect. I chose 500 after testing it myself.

We often use the following technique to scale things: use a constant (or even a "real" variable) and then multiply it by the value you can vary. This can be done in one pass by using the `map()` function. We already used it in [Chapter 6](ch06.html "Chapter 6. Sensing the World – Feeling with Analog Inputs"), *Sensing the World–Feeling with Analog Inputs*, for the same purpose—scaling an analog input value.

In that case, at the maximum value, your potentiometer (more generally your input) changes the parameter you want to alter to its maximum value.

Let's continue the review of the code.

We define the potentiometer pin n and the variable `potPin`. We also define `potValue` to `0`.

In the `setup()` block, we start Mozzi. We define the frequency of the oscillator as `aCos`. The frequency itself is the result of the `mtof()` function. `mtof` stands for **MIDI to Frequency**.

As we are going to describe it a bit later, MIDI protocol codes many bytes of values, including the pitch of notes it uses to transport from sequencers to instruments. Each MIDI note fits with real note values in the real world, and each note fits with a particular frequency. There are tables that show the frequency of each MIDI note, and Mozzi includes that for us.

We can pass a MIDI note pitch as argument to the `mtof()` function, and it will return the right frequency. Here, we use the `random(21,80)` function to generate a MIDI note pitch from 21 to 79, which means from A0 to A5.

Of course, this use case is a pretext to begin introducing MIDI. We could have directly used a `random()` function to generate a frequency.

We then read the current value of the analog input A0 and use it to calculate a scaled value of the frequency of the modulating oscillator, `aVibrato`. This is only to provide more randomness and weirdness. Indeed, if your pot isn't at the same place each time you restart Arduino, you'll have a different modulation frequency.

The `loop()` block then executes the `audioHook()` method constantly to produce audio.

And the smart thing here is the `updateControl()` method. We add the `analogRead()` function that reads the value of the analog input. Doing this in `updateControl()` is better, considering the purpose of this function. Indeed, the Mozzi framework separates the audio rendering time-critical tasks from the control (especially human control) pieces of code.

You'll come across this situation very often in many frameworks, and it can confuse you the first time. It is all about the task and its scheduling. Without reverse-engineering the Mozzi concepts here, I would like to say only that time-critical events have to be handled more carefully than human actions.

Indeed, even if it seems as if we can be very fast at turning a knob, it is really slow compared to the sample rate of Mozzi, for instance (16,384 kHz). This means we cannot stop the whole process only to test and check, if we change the value of this potentiometer constantly. Things are separated; keep this in mind and use the framework carefully.

Here, we read the value in `updateControl()` and store it in the `potValue` variable.

Then, in `updateAudio()`, we calculate the vibrato value as the value of `potValue` scaled from `0` to the value of `intensityMax`, multiplied by the next value of the oscillator in its wavetable.

This value is then used in a new method named `phMod`. This method applies a phase modulation to the oscillator for which it is called. This modulation is a nice way to produce a frequency modulation effect.

Now, upload the firmware, add the earphone, and turn the potentiometer. You should be able to hear the effect and control it with the potentiometer.

# Controlling the sound using envelopes and MIDI

We are now okay to design small bits of a sound engine using Mozzi. There are other libraries around, and what we learned will be used with those two. Indeed, these are patterns.

Let's check how we can control our Arduino-based sound engine using a standard protocol from a computer or other device. Indeed, it would be interesting to be able to trigger notes to change sound parameters using a computer, for instance.

Both are protocols used in the music and new media related projects and works.

## An overview of MIDI

**MIDI** is short for **Musical Instrument Digital Interface**. It is a specification standard that enables digital music instruments, computers, and all required devices to connect and communicate with one another. It was introduced in 1983, and at the time of writing has just celebrated its 30th anniversary. The reference website is [http://www.midi.org](http://www.midi.org).

MIDI can transport the following data over a basic serial link:

*   Notes (on/off, after touch)
*   Parameter changes (control change, program change)
*   Real-time messages (clock, transport state such as start/stop/continue)
*   System exclusives, allowing manufacturers to create their message

A new protocol appeared and is used very widely today: OSC. It isn't a proper protocol, by the way.

**OSC** stands for **Open Sound Control** and is a content format developed by two people at the **Center for New Music and Audio Technologies** (**CNMAT**) at University of Berkeley, California. It was originally intended for sharing gestures, parameters, and sequences of notes during musical performances. It is very widely used as a replacement for MIDI today, providing a higher resolution and faster transfer. Its main feature is the native network transport possibility. OSC can be transported over UDP or TCP in an IP environment, making it easy to be used over Wi-Fi networks and even over the Internet.

## MIDI and OSC libraries for Arduino

I'd suggest two libraries here. I tested them myself and they are stable and efficient. You can check the one about MIDI at [http://sourceforge.net/projects/arduinomidilib](http://sourceforge.net/projects/arduinomidilib). You can check this one about OSC at [https://github.com/recotana/ArdOSC](https://github.com/recotana/ArdOSC). You shouldn't have too many difficulties installing them now. Let's install at least MIDI, and restart the IDE.

## Generating envelopes

In the audio field, an **envelope** is a shape used to modify something. For instance, imagine an amplitude envelope shaping a waveform.

You have a waveform first. I generated this sine with Operator synthesizer in Ableton Live ([https://www.ableton.com](https://www.ableton.com)), the famous digital audio workstation. Here is a screenshot:

![Generating envelopes](img/7584_09_019.jpg)

A basic sine wave generated by an operator in Ableton Live's Operator FM synth

The sine doesn't show very well due to aliasing; here is another screenshot, which is the same wave but more zoomed in:

![Generating envelopes](img/7584_09_020.jpg)

A sine wave

This sine wave has a global constant amplitude. Of course, the air pressure push and pull constantly, but the global maximums and minimums are constant over time.

Musicians always want to make their sounds evolve over time, subtly or harshly.

Let's apply an envelope to this same wave that will make it increase the global volume progressively, then decrease it a bit, and then decrease quickly to zero:

![Generating envelopes](img/7584_09_021.jpg)

A sine wave altered by an envelope with a long attack

Here is the result with another envelope:

![Generating envelopes](img/7584_09_022.jpg)

A sine wave altered by an envelope with a very short attack

Basically, an envelope is a series of points in time. At each moment, we multiply the value of the original signal by the value of the envelope.

This produces a sound evolution over time.

We can use envelopes in many cases because they can modulate amplitude, as we just learned. We can also use them to alter the pitch (that is, the frequency) of a sound.

Usually, envelopes are triggered (that is, applied to the sound) at the same time the sound is triggered, but of course we can use the offset retrigger feature to retrigger the envelope during the same triggered sound and do much more.

Here is a last example showing a pitch envelope. The envelope makes the frequency of the sound decrease. As you can see, the waves are tighter on the left than on the right. The sound changes from high-pitched to low-pitched.

![Generating envelopes](img/7584_09_023.jpg)

An envelope modulating the pitch of a sound

## Implementing envelopes and MIDI

We are going to design a very cheap sound synthesizer that will be able to trigger notes when it receives a MIDI note message and alter the sound when it receives a particular MIDI Control Change message.

The MIDI part will be handled by the library and the envelope will be explicated and coded.

You can check the following code. This code is also available in the `Chapter09/MozziMIDI/` folder.

[PRE7]

At first, we include the MIDI library. Then we include the Mozzi library.

Of course, the right bits of Mozzi to include are a bit different for each project. Studying examples helps to understand what goes where. Here, we not only need Oscil for the basic features of the oscillator, but also need Line. Line is related to interpolation functions in Mozzi. Generating an envelope deals with this. Basically, we choose two values and a time duration, and it starts from the first one and reaches the second one in the time duration you choose.

We also include the wavetable related to a sine.

We define a control rate higher than before, at 128\. That means the `updateControl()` function is called 128 times per second.

Then we define the oscillator as `aSin`.

After these bits, we define an envelope by declaring an instance of the Line object.

We define two variables that store the release part of the envelope duration, one for the control part in one second (that is, the number of steps will be the value of `CONTROL_RATE`) and one for the audio part in one second too (that is, 16,384 steps). Lastly, a variable named `fade_counter` is defined.

`HandleControlChange()` is a function that is called when a MIDI Control Change message is sent to Arduino. The message comes with these bytes:

*   MIDI channel
*   CC number
*   Value

These arguments are passed to the `HandleControlChange()` function, and you can access them directly in your code.

This is a very common way to use event handlers. Almost all event listener frameworks are made like this. You have some function and you can use them and put whatever you want inside them. The framework itself handles the functions that have to be called, saving as much CPU time as possible.

Here, we add a `switch` statement with only one case over the `CCNumber` variable.

This means if you send a MIDI Control Change 100 message, this case being matched, the value of `CC` will be processed and the `vol` variable will be altered and modified. This Control Change will control the master output volume of the synth.

In the same way, `HandleNoteOn()` and `HandleNoteOff()` handle MIDI note messages.

Basically, a MIDI Note On message is sent when you push a key on your MIDI keyboard. As soon as you release that key, a MIDI Note Off message pops out.

Here, we have two functions handling these messages.

`HandleNoteOn()` parses the message, takes the velocity part, bit shifts it on the left to 8 bits, and passes it to `aGain` through the `set()` method. When a MIDI Note On message is received, the envelope `aGain` is triggered to its maximum value. When a MIDI Note Off message is received, the envelope is triggered to reach 0 in one second via the number of audio steps discussed before. The `fade` counter is also reset to its maximum value at the moment the key is released.

In this way, we have a system responding to the MIDI Note On and MIDI Note Off messages. When we push a key, a sound is produced until we release the key. When we release it, the sound decreases linearly to 0, taking one second.

The `setup()` method includes the setup of the MIDI library:

*   `MIDI.begin()` instantiates the communication
*   `MIDI.setHandleControlChange()` lets you define the name of the function called when a control change message is coming
*   `MIDI.setHandleNoteOn()` lets you define the name of the function called when a Note On message is coming
*   `MIDI.setHandleNoteOff()` lets you define the name of the function called when a Note Off message is coming

It also includes the setup of Mozzi.

The `loop()` function is quite familiar now.

The `updateControl()` function does not contain the time-critical part of the sound generator. It doesn't mean this function is called rarely; it is called less than `updateAudio()`—128 times per second for control and 16,384 per second for audio, as we have seen before.

This is the perfect place to read our MIDI flow, with the `MIDI.read()` function.

This is where we can trigger our decreasing envelope to 0 as soon as the `fade` counter reaches 0 and not before, making the sound in one second, as we checked before.

Lastly, the `updateAudio()` function returns the value of the oscillator multiplied by the envelope value too. This is the purpose of the envelope. Then, `vol` multiplies the first result in order to add a key to control the master output volume.

The `<<8` and `>>8` expressions here are for setting a high-resolution linear fade on Note Off, and this is a nice trick provided by Tim Barrass himself.

## Wiring a MIDI connector to Arduino

This schematic is based on the MIDI electrical specification diagram at [http://www.midi.org/techspecs/electrispec.php](http://www.midi.org/techspecs/electrispec.php).

![Wiring a MIDI connector to Arduino](img/7584_09_024.jpg)

The MIDI-featured sound generator based on Arduino

The corresponding circuit diagram is as follows:

![Wiring a MIDI connector to Arduino](img/7584_09_025.jpg)

The MIDI connector wired to the Arduino-based sound generator

As you can see, the digital pin 0 (serial input) is involved. This means we won't be able to use the serial communication over USB. In fact, we want to use our MIDI interface.

Let's upload the code and start this small sequencer in Max 6.

![Wiring a MIDI connector to Arduino](img/7584_09_026.jpg)

The *cheap sequencer for chips* fires MIDI notes and MIDI control changes

The sequencer is quite self-explanatory. Toggle on the toggle button at the top-left and it starts the sequencer, reading each step in the multislider object. The higher a slider is, the higher the pitch of this note into that step will be.

You can click on the button under the multislider on the left, and it will generate a random sequence of 16 elements.

Choose the correct MIDI output bus from the list menu on the top-right.

Connect your Arduino circuit and your MIDI interface with a MIDI cable, and listen to the music. Change the multislider content and the sequence played. If you turn the dial, the volume will change.

Everything here is transmitted by MIDI. The computer is a sequencer and a remote controller and the Arduino is the synthesizer.

# Playing audio files with the PCM library

Another way to play sounds is by reading already digitalized sounds.

Audio samples define digital content, often stored as files on filesystems that can be read and converted into audible sound.

Samples can be very heavy from the memory size point of view.

We are going to use the PCM library set up by David A. Mellis from MIT. Like other collaborators, he is happy to be a part of this book.

The reference page is [http://hlt.media.mit.edu/?p=1963](http://hlt.media.mit.edu/?p=1963).

Download the library and install it.

Imagine that we have enough space in the Arduino memory spaces. How can we do the installation if we want to convert a sample on our disks as a C-compatible structure?

## The PCM library

Check this code. It is also available in the `Chapter09/PCMreader/` folder.

![The PCM library](img/7584_09_027.jpg)

Our PCM reader

There is an array of `unsigned char` datatypes declared as `const`, and especially with the `PROGMEM` keyword named `sample`.

`PROGMEM` forces this constant to be put in the program space instead of RAM, because the latter is much smaller. Basically, this is the sample. The `startPlayback()` function is able to play a sample from an array. The `sizeof()` method calculates the size of the memory of the array.

## WAV2C – converting your own sample

Since we have already played with wavetable, and this is what we will be doing hereafter, we can store our sample waveforms in the Arduino code directly.

Even if dynamic reading of the audio file from an SD card would seem smarter, PCM provides an even easier way to proceed—directly reading an analog conversion of an array while storing a waveform into a sound.

We first have to transform a sample as C data.

David Ellis made an open source, small processing-based program that provides a way to do this; it can be found at [https://github.com/damellis/EncodeAudio](https://github.com/damellis/EncodeAudio).

You can download it from the reference project page directly compiled for your OS.

Launch it, choose a WAV file (PCM-based encoded sample), and then it will copy something huge in your clipboard.

Then, you only have to copy-paste this content into the array defined before.

Be careful to correctly paste it between the curly brackets.

Here is the content copied from the clipboard after converting a `wav` sample that I made myself:

![WAV2C – converting your own sample](img/7584_09_028.jpg)

A huge amount of data to paste in a C array

In the same folder, I have put a `.wav` file I designed. It is a short rhythm recorded in 16 bits.

## Wiring the circuit

The circuit is similar to the one in the *Playing basic sound bits* section, except that we have to use the digital pin 11 here. And we cannot use PWM on pins 3, 9, and 10 because the timers involved in the library consume them.

![Wiring the circuit](img/7584_09_029.jpg)

Wiring our PCM reader

The circuit diagram is easy too.

![Wiring the circuit](img/7584_09_030.jpg)

Don't forget to use pin 11 with the PCM library

Now, let's play the music.

## Other reader libraries

There are also other libraries providing ways to read and decode the MP3 format or other formats.

You can find a lot on the Internet; but be careful as some of them require some shields, like the one on the Sparkfun website at [https://www.sparkfun.com/products/10628](https://www.sparkfun.com/products/10628).

This provides a shield with an SD Card reader, a 3.5 mm stereo headphone jack, a VS1053 shift register, and very versatile decoder chips for MP3, WMA, AAC, and other formats.

It is a very dedicated solution and we only have to interface the shield with Arduino.

Arduino only sends and receives bits from the shield, which takes care of the decoding of the encoded files, the conversion to analog signals, and so on.

I'd really suggest testing it. There are many examples on the Sparkfun website.

# Summary

We learned how to make things move right here with Arduino. In particular, we learned about:

*   Moving solid things with motors
*   Moving air with sound generators

Of course, unfortunately, I cannot describe more on how to make things move.

If you need help with sound, please contact me at `<[book@cprogrammingforarduino.com](mailto:book@cprogrammingforarduino.com)>`. I will be a happy to help you with sound inputs too, for instance.

This is the end of the second part of the book. We discovered a lot of concepts together. And now we are going to dig into some more advanced topics.

We are able to understand firmware design and inputs and outputs, so let's move further.

We are going to dig deeper into precise examples with I2C/SPI communication to use GPS modules, 7-segment LED systems, and more. We are also going to dig into Max 6, and especially how we can use Arduino to control some OpenGL visuals on the computer. We'll discover network protocols and how to use Arduino even without any network cables, with Wi-Fi. At last, we'll design a small library together and check some nice tips and tricks to improve our C code.