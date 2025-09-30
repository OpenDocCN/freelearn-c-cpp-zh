# Autonomous Robotics with Intel Edison

Robotics is a branch of engineering that deals with the design, development, and
application of robots. While robots can take the form of human beings, most robots are
designed to perform a specific set of tasks, and it may just look like a machine or
anything that is not human. What we are interested in is how the Intel Edison can be used to
develop robots. This chapter will cover the autonomous aspect of robotics and will mainly
cover the following topics:

*   Architecture of a robotic system
*   Intel Edison as a controller
*   Connecting sensors with the Intel Edison
*   Calibration of sensors with real-time environment
*   Actuators: motors, servos, and so on
*   Motordrivers: Dual H bridge configuration
*   Speed control
*   Patching everything together: Line follower robot
*   More advanced line follower robots based on the PID control system

This chapter will deal with a line follower robot and will explore all the components of it and discuss some tips and tricks for advanced line following. All the code in this chapter will be written using the Arduino IDE.

# Architecture of a typical robotic system

In a typical autonomous robotic system, the **sensor** does the job of gathering data. Based on the data, the controller initially processes it and then performs an action that results in an action by the use of **actuators**:

![](img/image_05_001-1.jpg)

Architecture of a robotic system

We can also have a **dashboard** that may just display what exactly is happening, or sometimes can provide some commands to the robot. Different models exist of the architecture. The preceding architecture is such that the components of a robot are said to be horizontally organized. The information gathered through sensors goes through multiple steps before getting executed as an action. Beside the horizontal flow, there can be vertical flow, where the control may shift to the actuator at any time without completion of the entire process. It's like a queue of cars moving on a highway, and then a single car takes a exit ramp and exits. The line follower robot that we will develop will use the horizontal approach, which is the most primitive.

# Intel Edison as a controller

Throughout the book, we have seen the use of the Intel Edison in various applications. In every case, we have stressed on the use of it as a controller because of its features. Now, in the field of robotics, we are again stressing on it as a microcontroller, but the question is, why?

Well, when we are dealing with robotics, in some cases we may need the core capabilities of the Intel Edison. The Intel Edison has a built-in BLE, Wi-Fi, and some features that make it perfect for use in robotics. The Arduino compatible expansion board allows us to use generic Arduino shields for peripherals and motor driving, while the core capabilities provide us some extra functionality for the robot, such as voice-based commands and uploading data to the cloud.

All this comes in a single unit. So, instead of using multiple peripherals, the Intel Edison provides us the flexibility to use everything under a single roof. The ability to attach a camera also adds more spice to it:

![](img/image_05_002-1.jpg)

Spider bot by Intel. It uses the Intel Edison and a custom expansion board for controlling servos. Picture credit: [https://i.ytimg.com/vi/3NeJisPvHcU/maxresdefault.jpg](https://i.ytimg.com/vi/3NeJisPvHcU/maxresdefault.jpg)

# Connecting sensors to the Intel Edison

In [Chapter 2](bada9944-ec60-4e8f-8d88-0085dd1c8210.xhtml), *Weather Station (IoT)*, we had a brief discussion about sensors. Here we are more focused on sensors that will help the robot find where exactly it is in an environment. We are going to deal with two sensors and how you can calibrate them:

*   Ultrasonic sensors (HCSR04)
*   Infrared sensors for detection of lines

The reason behind the use of the preceding sensors is that we use these sensors in robotics. However, others are also used, but these are the most commonly used. Now the next question that arises is "how do they operate and how do we hook up with the Intel Edison?"

Let's have a look at each of them.

# Ultrasonic sensor (HCSR04)

The main purpose of using this sensor is for the measurement of distances. In our case it may not be needed, but it is very useful in beginner robotics:

![](img/image_05_003-1.jpg)

Ultrasonic sensor HCSR04

It emits ultrasonic sound waves, which, if anything is present, then gets back a reflected and picked up by the receiver. Based on the duration of pulse output and the pulse received, we calculate the distance.

The sensor can provide readings ranging from 2 cm to 400 cm. It contains the emitter and the receiver, and thus we can just plug and play with our Intel Edison. The operating voltage is +5V DC. The operation isn't affected by sunlight or dark materials, and is thus efficient for finding distances. Another thing to be noted is that the beam angle is 15 degrees:

![](img/image_05_004-1.jpg)

Beam angle

Let's have a look at some sample code for using the preceding module.

There is no direct method of calculating the distance directly from the sensors. We need to use two GPIO pins. The first will be the trigger pin, which will be configured as the output, while the second will be the echo pin, which will be in input mode. The moment we send a pulse through the trigger pin, we wait for an incoming pulse in the echo pin. The timing difference and a little bit of mathematics gives us the distance:

[PRE0]

In the preceding code, concentrate on the loop method. Initially, we send a low pulse to get a clean high pulse. Next, we send a high pulse for `10` seconds. Finally, we get the high pulse in the echo pin and use the `pulseIn()` method to calculate the duration. The duration is sent to another method, where we use the known parameter of the speed of sound to calculate the distance. To use this code on the Edison, connect the HCSR04 sensor to the Edison by following this circuit diagram:

![](img/image_05_005-1.jpg)

Circuit diagram

Use the preceding circuitry to connect your HCSR04 to your Edison. Next, upload the code and open the serial monitor:

![](img/image_05_006.png)

Serial monitor reading for HCSR04

The preceding screenshot shows us the serial monitor where we get the readings of the distance measured.

# Applications of HCSR04

This sensor has lots of applications, especially in autonomous robotics. Even mapping can be accomplished using this sensor. When a HCSR04 is placed on top of a servo motor, the HCSR04 can be used to map the entire 360 degrees. These are extremely useful when we want to perform **simultaneous localization and mapping** (**SLAM**).

# Infrared sensors

These sensors have multiple utilities. Starting from line detection to edge detection, these sensors can even be optimized to be used as proximity sensors. Even our smartphones use these sensors. They are usually located near the front speakers. These sensors also work on the principle of sending and receiving signals.

# Working methodology

The following image is a commonly used infrared sensor:

![](img/image_05_007-1.jpg)

Infrared sensor: Picture source: [http://www.amazon.in/Robosoft-Systems-Single-Sensor-Module/dp/B00U3LKGTG](http://www.amazon.in/Robosoft-Systems-Single-Sensor-Module/dp/B00U3LKGTG)

It has an emitter and a receiver. The output can either be digital or analog. The emitter sends an infrared wave. Based on the object, it is reflected back and the signal is picked up by the receiver. Based on that, we know that we have something in close proximity.

There are multiple applications for infrared sensors:

*   Detection of objects in close proximity
*   Measuring temperature
*   Infrared cameras
*   Passive infrared for motion detection

The first was already discussed. The second one is the use of infrared detectors for detection of the infrared spectrum, and based on that, we calculate the temperature. Infrared cameras are widely used in the military, firefighting, and many places where we need to know from a distance that temperatures are high. The third one is the PIR. Passive infrared modules are used for motion detection and are used for automating homes.

# Digital and analog outputs for infrared sensors

We have already mentioned that these sensors either give a digital output or an analog output. But why are we so concerned about this? It's mainly because of the reason that in a typical line following we may use digital values, but in high-speed line follower robots, we opt for analog values. The greatest advantage is that there will be a smooth transition from a white surface to a black surface, and in a high-speed line following, we control the speed of the motors based on the sensor analog input, and therefore have more efficient control. For simple purposes, we go for digital output.

# Calibration of the infrared sensor module

Infrared sensor modules need to be calibrated for the following reasons:

*   The environment may be too bright for the sensor to actually detect anything
*   Due to changes in environmental parameters, we may need to calibrate them to suit our needs

Let us consider a line follower robot under two sets of conditions:

*   In a normally lit environment (indoor)
*   In a sunny environment (outdoor)

When the robot runs efficiently in an indoor environment under ambient light, it is not necessarily true that it will do the same outdoors. The reason is that the sensors are infrared and in a brightly lit environment, especially in sunlight, it's harder to detect. Now, how do we calibrate the sensors? Every sensor module has a potentiometer that controls the sensitivity of the sensor. This potentiometer needs to be adjusted according to the surroundings so that you get appropriate readings.

Now, when you are dealing with robotics, keep some code handy, because it might be required. Similarly for calibration, we follow very simple steps.

Let us consider a simple line follower use case. We need to follow a black line on a white track. The steps are as follows:

1.  Place the robot or the sensor in the ideal position (ideal position stands for the position where your robot needs to be, that is, at the center of the line).
2.  If the robot is not yet constructed, then cover your sensor with paper and hold it at or about 2 cm above the track.
3.  Now check the values of the sensor with the code to be shown in the following steps.
4.  There should be a stark difference between the sensor that is above the white line and the sensor that is above the black line. If you have both the sensors above white line then the values should be exactly similar else adjust the potentiometer.
5.  Similarly, carry out the process by placing both the sensors over the black line.
6.  Carry out the entire process again if the environment or, more precisely, the lighting conditions change.

# Hardware setup for calibration and sensor reading

Follow the following circuit diagram for the hardware setup:

![](img/image_05_008-1.jpg)

Sensor calibration circuit

The circuit is pretty straightforward and simple to understand. Just attach two sensors to the **A0** and **A1** pins and the common **Vcc** and **Gnd** connections.

Now, let's get on with the code:

[PRE1]

The preceding code returns us the values and readings from both the sensors. After you burn the code into the Intel Edison, you will start receiving the values from the sensors. Now, it is required to perform the tests as mentioned in calibration to calibrate the sensors.

# Actuators - DC motors and servos

In every robotics project the robot needs to get some mobility. For mobility, we use DC motors or servos. This section will deal with motors and types of motor. The next section will deal with how we can control the motors using the Intel Edison.

Electrical motors are electromechanical devices that convert electrical energy to mechanical energy. The two electrical components in a motor are the field windings and the armature. It's a two pole device. The armature is usually on the rotor, while the field windings are usually on the stator.

Permanent magnet motors use permanent magnets to supply field flux:

![](img/image_05_009.jpg)

Permanent magnet motor

These motors are restricted by the load they can drive, and that's a disadvantage. However, considering the fact that we are focusing on robotics and on a small-scale basis, we are opting for this kind of motor.

These motors generally operate at 9V-12V, with a maximum full-load current of around 2-3 A. In robotics, we mainly deal with three types of DC motor:

*   Continuous DC
*   Stepper
*   Servo

Continuous DC motors are continuous rotation devices. When the power is switched on the motor rotates, and if polarity is reversed, it rotates in the opposite direction. These motors are widely used for providing mobility to the robot. Normally, DC motors come with gears, which provide some more torque:

![](img/image_05_010-1.jpg)

A DC motor. Picture source: [http://img.directindustry.com/images_di/photo-g/61070-3667645.jpg](http://img.directindustry.com/images_di/photo-g/61070-3667645.jpg)

The next category is a servo motor. The speed of these motors is controlled by varying the width of pulses with a technique known as pulse width modulation. Servo motors are generally an assembly of four things: a DC motor, a gearing set, a control circuit and a position sensor (usually a potentiometer). Generally, a servo motor contains three set of wires: Vcc, Gnd, and signal. Due to thier precise nature, servo motors have a complete different use case, where position is an important parameter. Servo motors do not rotate freely like a standard DC motor. Instead, the angle of rotation is limited to 180 degrees (or so) back and forth. The control signal comes as a **Pulse Width Modulation** (**PWM**) signal:

![](img/image_05_011.png)

A servo motor: Picture source: [https://electrosome.com/wp-content/uploads/2012/06/Servo-Motor.gif](https://electrosome.com/wp-content/uploads/2012/06/Servo-Motor.gif)

Finally, stepper motors are an advanced form of servo motor. They provide a full 360-degree rotation and it's a continuous motion. The stepper motor utilizes multiple toothed electromagnets arranged together around a central gear to define the position. They require an external controller circuit to individually excite each electromagnet. Stepper motors are available in two varieties; unipolar and bipolar. Bipolar motors are the strongest type of stepper motor and usually have four or eight leads:

![](img/image_05_012-1.jpg)

Stepper motor

In the project of line following, we'll be dealing with DC motors. But motors cannot be directly hooked up to the Intel Edison, and therefore we need an interfacing circuit, more commonly known as a motor driver.

# Motor drivers

Since motors consume voltage and current that cannot be supplied by the GPIO pins alone, we opt for motor drivers. These have an external power supply and use the microcontroller's GPIO pins to receive the control signal. Based on the control signals received, the motor rotates at a particular speed. There are lots of motor drivers available on the market; we will be concentrating on L293D initially, and then a custom and a high-power driver.

The target of any motor driver is to receive control signals from the controller here, the Intel Edison, and send the final output to the motor. Typically, a motor driver can rotate the motor in both directions and also control the speed.

# L293D

L293D is a typical motor driver integrated circuit that can drive two motors in both directions. It's like a starter for every robotics project. It's a 16-bit IC:

![](img/image_05_013.png)

Pinout for L293D. Picture source: [http://www.gadgetronicx.com](http://www.gadgetronicx.com)

The maximum voltage for Vs motor supply is 36V. It can supply a max current of 600mA per channel.

It works on the concept of H bridge. The circuit allows the flow of current in either direction. Let's have a look at the H bridge circuit first:

![](img/image_05_014.png)

Simple layout of an H bridge circuit. Picture source: [https://en.wikipedia.org/](https://en.wikipedia.org/)

Here, **S1**, **S2**, **S3**, and **S4** are switches that in real life contain transistors. The operation is extremely simple. When **S1** and **S4** are on, the central motor rotates in one direction while the reverse happens when **S2** and **S3** are on. **S1**, **S2**, **S3**, and **S4** receive control signals from the microcontroller and operate the direction of the motor accordingly.

The L293D consists of two such circuits. Thus, we can control up to two motors. One can use the L293D as a module or just as a standalone IC. Normally what we need to worry about are four pins, where we'll send control signals. We have the pin layout for L293D. Let's have a look at what signals will result in what kind of action.

Pins 1-8 are responsible for one motor, while pins 9-16 are responsible for the other.

Enable pin is set to logic high for the operation. The same goes for the other side.

What needs to be tampered with are the input pins 2, 7, 10, and 15\. The motors are connected to pins 3, 6, 11, and 14\. Vss is for the power supply for the motor, while Vss or Vcc is for the internal power supply. While enable 1 and enable 2, that is pins 1 and 9, are set to high depending on the condition:

| **Pin 2 or 10** | **Pin 7 or 15** | **Motor** |
| High | Low | Clockwise |
| High | High | Stop |
| Low | Low | Stop |
| Low | High | Anti-clockwise |

The preceding table summarizes the action triggered based on the input. It is to be noticed that if both the input pins are either on or off then the motor won't rotate. Now in typical L293D modules, only four control pins are exposed and the main voltage supply and Gnd pins are exposed. In the preceding table, it is mentioned as **2 or 10** and **7 or 15**. The pair goes as 2 and 7 or 10 and 15\. It means that 2 is on or 10 is off and the same goes for the other as well. The motion of the motor, which is designated as **clockwise** and **anti-clockwise**, depends on the connection of the motor. Assume that the rotation direction reverses when the control signal changes.

# Circuit diagram

Construct the following temporary circuit to test out the motors. The enable pins are to be connected to **5V** of the Intel Edison and **Gnd** to **Gnd**.

It's recommended to use motor driver modules instead of a standalone IC. This allows us to accomplish things in a much simpler way.

**CP1** and **CP2** are the control pins for the first motor, while **CP3** and **CP4** are the control pins for the second motor:

![](img/image_05_015-1.jpg)

Circuit diagram for motor testing

When dealing with robotics, it is advised to keep sample code with which you can test that your motor driver is working or not. The code discussed here will explain how one can do so.

This part is especially important because it is necessary as the motor testing unit and also for calibration. This code also depends on your motor connection and it may require trial and error:

[PRE2]

The preceding code is very simple to understand. Here, we have broken it into several methods for `forward`, `backward`, `left`, `right`, and `stop`. These methods will be responsible for sending control signals to the motor driver using `digitalWrite`. Point to be noted is that this code may not work. It all depends on how your motor is connected. So, when you burn this code to your Intel Edison, make a note of which direction your motor is rotating for the initial 10 seconds. If both motors rotate in the same direction, well then you are lucky enough of not tampering with the connections. If, however, you observe rotation in opposite directions, then reverse the connection of one of the motors from its motor driver. This will allows the motor to rotate in the other direction, so both the motors will rotate in the same direction.

Another important point to be noted is that considering the methods `left` and `right`, we notice that motors rotate in different directions. That's how the steering system of a robot is implemented. It will be discussed in a later section of this chapter.

While dealing with motor drivers, please go through the specifications first. Then go for connections and coding.

# Speed control of DC motors

Now that we know how to control a motor and its direction of rotation, let's have a look at controlling the speed of a motor, which is necessary for advanced maneuvering.

The speed control happens through the PWM technique, where we vary the width of pulses to control the speed.

This technique is used to get analog results by digital means. The digital pins on the Intel Edison produce a square wave:

![](img/image_05_016.png)

Typical square wave

The on and off pattern can simulate voltages between a full on **5V** and a full off **0V**. This is manipulated by altering the time the signal spends on and the time the signal spends off. The duration of the on time is called **pulse width**. In order for us to get varying pulse values, we change or modulate the pulse width. If this is done fast enough, then the result is a value between 0-5V.

There are PWM pins on the Arduino breakout board for the Intel Edison. These pins are used:

![](img/image_05_017.jpg)

PWM samples. Picture source: [www.arduino.cc](http://www.arduino.cc)

Now we can implement this to control the speed of our own motors. In the preceding L293D IC, the enable pins can be used for PWM input.

Modules of L293D mainly expose the following pins:

*   Input 1 for motor 1
*   Input 2 for motor 1
*   Input 1 for motor 2
*   Input 2 for motor 2
*   Enable for motor 1
*   Enable for motor 2
*   Vcc
*   Gnd

Take a look at the following module:

![](img/image_05_018-1.jpg)

L293D motor driver module:

A total of eight pins are exposed, as mentioned previously.

Connect the enable pins to any of the PWM pins on the Intel Edison:

*   Enable 1 to digital pin 6
*   Enable 2 to digital pin 9

Next, to control the speed we need to use the `analogWrite` method in those enabled PWM pins on the Intel Edison.

To set the frequency of the Intel Edison's PWM control, use the example shown in the following link and clone it:

[https://github.com/MakersTeam/Edison/blob/master/Arduino-Examples/setPWM_Edison.ino](https://github.com/MakersTeam/Edison/blob/master/Arduino-Examples/setPWM_Edison.ino)

The range of values of the `analogWrite` method is 0-255, where 0 is always off and 255 is always on.

Now using this modify the preceding code to set the enable pin values. An example of using it is shown here. The task of controlling the pins in a fully-fledged motion is left to the reader:

[PRE3]

In the preceding code, stress on the `forward` method, where we've used `analogWrite (PWM, 122)`. This means that the motor should now rotate half of its original speed. This technique can be used for faster line following robots and speed control.

# More advanced motor drivers

While dealing with robotics, there may be some cases where L293D isn't quite a good option due to its current limitation. In those cases, we opt for more powerful drivers. Let's have a look at another product from robokits, which can pretty much drive powerful high torque motors:

![](img/image_05_019-1.jpg)

Dual motor driver high power. Picture source: [http://robokits.co.in/motor-drives/dual-dc-motor-driver-20a](http://robokits.co.in/motor-drives/dual-dc-motor-driver-20a)

The preceding motor driver is my personal favorite. It has multiple controls and can drive high torque motors. The driver has the following five control pins:

*   **Gnd**: Ground.
*   **DIR**: When low, the motor rotates in one direction; when high, it rotates in another direction.
*   **PWM**: Pulse width modulation to control the speed of the motor. Recommended frequency range is 20 Hz - 400 Hz.
*   **BRK**: When high, it halts the motor in operation.
*   **5V**: Regulated 5V output from the motor driver board.

From the description of the pins discussed here, it should be clear as to why this is a better choice.

The voltage and current specifications are as follows:

*   Voltage range: 6V to 18V
*   Max current: 20 A

The current and the voltage rating help us to drive motors with max load. We have used this motor driver for many of our applications, and it serves without fail. However, there are other motor drivers also on the market that can provide a similar functionality. The choice of which motor driver to use depends on certain factors, as discussed here:

*   **Power**: It all depends on how much power the motor needs to run at full capacity. The current drawn at full load and at no load condition. If you are going to use a high torque motor driver with an L293D, you may end up frying your motor driver.
*   **Maneuvering**: According to the use case of the problem, the choice of motor is yours and ultimately the choice of the motor driver. In high speed line following, we require PWM capability, thus we need a driver that is capable of handling PWM signals.

Ultimately based on your use case choose your motor driver.

The following is an image of a small yet high performance UGV powered using the previous motor driver that we developed:

![](img/image_05_020-1.jpg)

Black-e-Track UGV. The UGV's motors are high torque 300 RPM and can climb steep slopes of up to 75 degrees.

Now we have a fairly good idea of how motor drivers work and how we can choose a good motor driver.

# Line follower robot (patching everything together)

Based on the previous sections of this chapter, we have got a fairly good idea as to how everything needs to be brought under a single platform. Let's go in to the steps and have a look at how a line follower robot works.

# Fundamental concepts of a line follower

The following figure shows the concept of a line follower robot:

![](img/image_05_021-1.jpg)

Line follower concept

In the preceding figure, we have two conditions. The first is when the left sensor is over the line and the second is the same for the right sensor. The black line is the track that the robot should follow. There are sensors represented by blocks. Let's consider the left-hand side of the preceding figure.

Initially, both the sensors are over the white surface that is in position 1, as shown in the preceding figure. Next, consider position 2 on the left-hand side of the image. The left sensor comes over the black line and the sensor gets activated. Based on this, we intercept sensor readings and relay control signals to the motor driver. In the preceding case, specific to the left-hand side of the image, we get that the left sensor detected the line and thus the robot should not go forward, instead it should take a slight left turn until and unless both the sensors return the same value.

The same is the case for the right-hand side of the figure where the right sensor detects the black line and the motion execution command is triggered. This is the case of a simple line follower robot where a single colored line needs to be followed. Things get a bit different with multiple colored lines that need to be followed.

Now that we know the exact process of following a line, we can now focus more on how the robot executes turns and the robot structure.

# Robot motion execution

To understand how a robot executes its motions, let's consider a four-wheel drive robot:

![](img/image_05_022-1.jpg)

Typical 4WD robot structure

The robot can follow a differential drive-based system where a turn is executed the side where the robot turns, that side's wheel rotates either in the reverse direction, or slows down, or stops. If you are dealing with two wheels, then stopping one side and rotating the other will also do. However, if you are using four wheels, then it's always safe to go for reverse rotation, or it will give rise to wheel slip condition. Again, when using two wheels, we opt for the use of a castor wheel, which keeps the robot balanced. Details of wheel slip condition and other structural elements will be discussed in [Chapter 6](56a788c0-bef4-43e2-91cb-02b2d981c5c0.xhtml), *Manual Robotics with Intel Edison*.

# Hardware requirements for line follower robots

The list of hardware required for a simple two-wheel drive line follower robot is as follows:

*   Robot chassis
*   9V DC motors
*   Motor driver L293D
*   Two infrared sensors
*   9V DC power supply
*   Two wheels
*   One castor
*   Intel Edison, which is used as a controller

The process of attaching the motors to the chassis and the castor won't be shown. A circuit diagram will be shown, and arranging all the components depends on the reader.

Use a two or a four-wheel drive robot chassis, and as we are using two wheels we will fit the castor on the front of the robot. The the sensors should be at the front, on either side of the robot. Let's consider a 2D model of our robot:

![](img/image_05_023-1.jpg)

2D robot model of a typical line follower

The preceding figure is a 2D model of the line follower robot with 2WD. The sensors should be on either side and the castor in the middle. While the L293D and the Intel Edison can be located anywhere, the position of the castor, sensors, motors, and obviously the wheels should be the same or similar to the structure shown in the preceding figure.

Make sure that the distance from the sensor to the ground is optimum for detection of the line. This distance must be calculated during the calibration of the sensors.

The hardware setup usually takes a bit of time as it involves a lot of tickling of wires and loose soldering joints usually add in more problems. Now, before moving forward with the code, let's wire everything up with the following circuit diagram:

![](img/image_05_024-1.jpg)

Circuit diagram for line follower robot

The preceding circuit is a combination of what we've done so far in this chapter. A common ground, a **Vcc** line, is created that connects the **L293D**, **Intel Edison**, and the **sensors**. The **9V DC** power supply powers the motors, while the control pins are responsible for sending control signals from the Intel Edison. The output of the sensors is connected to the digital pins of the Intel Edison. Finally, the motor driver controls the motors based on the control signals.

If you have a close look at the preceding circuit diagram, then everything fits into the typical robotics architecture:

![](img/image_05_025-1.jpg)

Robotics architecture

But the dashboard is missing. We may add a dashboard, but as of now, we aren't interested in that aspect.

Now that the hardware is done with all the connections and circuitry, let's add a code to it to make it run.

The algorithm is very simple, as follows:

1.  Check the left sensor value.
2.  If detected, turn `left`.
3.  Or else, check the `right` sensor value.
4.  If detected, turn `right`.
5.  Else if both detected.
6.  Stop the motion.
7.  Or else, move `forward`.

The following is the code for this:

[PRE4]

In the preceding code, which is very similar to that of the motor testing, only the `void loop()` is replaced by the main logic, as described in the algorithm. We've used macros for defining sensor and motor pins.

The code initially sets the pins to either input mode or output mode. Next, we store the input values of the sensor. Finally, based on the sensor input, we process the robot motion.

After you burn the code on your Intel Edison, the robot should run. Try a simple track initially, and once your robot runs, then go for a more tight turns. Again, it should be kept in mind that in the preceding code, our right sensor may be your left sensor. In that case you must change the position or just change the condition.

Thus, through a combination of sensors and very simple processing, we can control the motors of a robot and follow a line. Now, if the problem statement asked you to reverse the condition and follow a white line on a black surface, we'd need to tamper with the code a bit, especially in condition checking. The result will be as follows:

[PRE5]

Just the 1s and 0s need to be interchanged.

Now that we have fairly basic knowledge of developing a basic line follower robot, let's have a brief look at an advanced form of line following and tackle some of the basic concepts.

# Advanced line follower robot concepts

So far, we have focused on basic line follower robots of following a single line. Now let's complicate things a bit and try to solve the following section of a track using the previous logic:

![](img/image_05_026-1.jpg)

Intersection track

In the preceding track, if we use two sensors, then things will get out of hand because the robot should go forward, but according to the algorithm discussed before when both sensors are returning `1`, the robot should stop. Then how do we tackle such cases?

The answer lies in the use of more than two sensors. Let's have a look at the following figure:

![](img/image_05_027-1.jpg)

Intersection 3 sensor concept

In the preceding figure, we have shown the use of three sensors. Let's consider the following values:

*   If the sensor is on black, it sends `1`
*   If the sensor is on white, it sends `0`

Now, in **position 1**, we get a value of `010` and on **position 1**, the value is `111`. This means that `111` represents an intersection. We can have it for left and right junctions too:

![](img/image_05_028-1.jpg)

Intersection - 2

Here, the value for left (**position 1**) will be `110` and for right it will be `011`. Now it's easier to detect intersecting points and also at the same time prevent the robot from executing false turns. To implement this in code, it's very simple:

[PRE6]

It should be noted that the preceding method needs to be applied based on the scenario of the track. Sometimes it is even required to ignore intersections. It totally depends on the track. The placement of the sensors can also play a very crucial role in line following.

There is a popular line follower robot that has a curved arrangement of sensors. It's the Polulu 3 pi robot:

![](img/image_05_029-1.jpg)

Polulu 3pi robot. Image source: [https://www.pololu.com/product/975](https://www.pololu.com/product/975)

Normally we use five or six sensors for a line follower robot. It comes as a module known as a line sensor array:

![](img/image_05_030-1.jpg)

Line sensor array. Image source: [http://robokits.co.in/sensors/line-sensor-array](http://robokits.co.in/sensors/line-sensor-array)

# Proportional integral derivative - based control

**Proportional integral derivative** (**PID**) is a control loop feedback mechanism. The main point of using a PID-based control is for efficiently controlling motors or actuators. The main task of the PID is to minimize the error of whatever we are controlling.

It takes an input and then calculates the deviation or error from the intended behavior and ultimately adjusts the output.

Line following may be accurate at lower speeds, but at higher speeds, things may get out of hand or out of control. That's when the PID comes into the picture. Let's have a look at some of the terminology:

*   **Error**: The error is something that the device isn't doing the right way. If the RPM of a motor is 380 and the desired RPM is 372 then the error is 380-372=8.
*   **Proportional** (**P**): It is directly proportional to the error.
*   **Integral** (**I**): It depends on the cumulative error over a period of time.
*   **Derivative** (**D**): It depends on the rate of change of error.
*   **Constant factor**: When the terms P, I, and D are included in code, it is done by multiplying with some constant factors:
*   *   P: Factor (Kp)
    *   I: Factor (Ki)
    *   D: Factor (Kd)
*   **Error measurement**: Consider a line follower robot with a five-sensor array that returns digital values; let's have a look at the error measurement. The input obtained from the sensors needs to be weighted depending on the possible combinations of the input. Consider the following table:

| **                    Binary value** | **Weighted value** |
|                     00001 | 4 |
|                     00011 | 3 |
|                     00010 | 2 |
|                     00110 | 1 |
|                     00100 | 0 |
|                     01100 | -1 |
|                     01000 | -2 |
|                     11000 | -3 |
|                     10000 | -4 |
|                     00000 | -5 |

The range of values is from -5 to +5\. The measurement of the position of the robot is taken several times in a second and then, with the average, we determine the errors.

*   **PID formulae**: The error value calculated needs to affect the real motion of the robot. We need to simply add the error value to the output to adjust the robot's motion.
*   **Proportional**:

*Difference = Target position - Present position*
*Proportional = Kp * Difference*

The preceding approach works, but it is found that for a quick response time, if we use a large constant or if the error value is large then the output overshoots the set point. In order to avoid that, the derivative is brought into the picture.

*   **Derivative**: Derivative is the rate of change of error.

*Rate of change = (Difference - Previous difference) / Time interval*
*Derivative = Kd * Rate of change*

The timing interval is obtained by using the timer control of the Intel Edison. This helps us to calculate how quickly the error changes, and based on that, the output is set.

*   **Integral:**

*Integral = Integral + Difference*
*Integral = Ki * Integral*

The integral improves the steady state performance. All the errors are thus added together and the result is applied on the motion of the robot.

The final control signal is obtained from this:

*Proportional + Integral + Derivative*

*   **Tuning**: PID implementation can't help you, rather it will degrade the motion of the robot unless and until it is tuned. The tuning parameters are Kp, Ki, and Kd. The tuning value depends on various parameters, such as the friction of the ground, the light conditions, the center of mass, and many more. It varies from one robot to the other.

Set everything to zero and start with Kp first. Set Kp to `1` and see the condition of the robot. The goal is to make the robot follow the line even if it's wobbly. If the robot overshoots and loses the line, decrease Kp. If it's not able to take turns or seems sluggish, increase Kp.

Once the robot follows the line more or less correctly, assign `1` to Kd. For now, skip Ki. Increase the value of Ki until you see less wobbling. Once the robot is fairly stable and able to follow the line more or less correctly, assign a value ranging from 0.5-1 in Ki. If the value is too low, not much of a difference would be found. If the value is too high, then the robot may jerk left and right quickly. You may end up incrementing by `.01`.

PID doesn't implement effective results unless it's properly tuned, so coding only won't yield proper results.

# Open-ended question for the reader

The PID use case was explained. Try to implement it in code and write and implement a line follower algorithm with the use of five-six sensors. These practice use cases will explain all the concepts behind line following.

# Summary

In this chapter about autonomous robotics, we have covered multiple topics, including dealing from sensors and motor drivers, and how to calibrate sensors and test motor drivers. We also covered line follower robot use cases in detail and also had a chance to look at more advanced controls, and ultimately ended with the PID-based control system.

In [Chapter 6](56a788c0-bef4-43e2-91cb-02b2d981c5c0.xhtml), *Manual Robotics with Intel Edison*, we'll cover manual robotics and develop some controller software. We'll also cover more hardware topics pertaining to a robot.