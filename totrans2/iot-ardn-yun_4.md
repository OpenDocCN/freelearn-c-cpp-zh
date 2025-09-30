# Chapter 4. Wi-Fi-controlled Mobile Robot

In this last chapter of the book, we are going to use the Arduino Yún in a completely different field: robotics. You will learn how to interface DC motors, as well as how to build your own mobile robot with the Arduino Yún as the brain of the robot, a distance sensor for the robot, and wireless control using Wi-Fi and a simple web interface. You will also be able to get a live display of the measurements done by the robot, for example, the distance that is measured in front of the robot by the ultrasonic sensor.

# Building the mobile robot

Arduino boards are widely used in mobile robots because they are easy to interface with the different parts of a robot, such as sensors, actuators such as DC motors, and other components such as LCD screens. Arduino even released their own robot recently so people can experiment on a common robotic platform. These robots are usually programmed once and then left alone to perform certain tasks, such as moving around without hitting obstacles or picking up objects.

In this project, we are going to make things differently. What we want is to build a mobile robot that has the Arduino Yún as its "brain" and control it entirely via Wi-Fi from a computer or mobile device, such as a smartphone or a tablet. To do so, we will program an Arduino sketch for the robot that will receive commands and send data back, and program a graphical interface on your computer. This way, if you want to build more complex applications in the future, you simply need to change the software running on your computer and leave the robot untouched.

We are first going to build the robot using some basic mechanical and electrical parts. We will not only show you how to build the robot using a specific kit, but also give you a lot of advice on building your own robot using other equivalent components. To give you an idea about what we are going to build, the following is an image of the assembled robot:

![Building the mobile robot](img/8007_04_01.jpg)

At the bottom of the robot, you have most of the mechanical parts, such as the chassis, the wheels, the DC motors, and the ultrasonic sensor. You also have the battery at the center of the base of the robot. Then, you can see the different Arduino boards on top. Starting from the bottom, you have the Arduino Yún board, an Arduino Uno board, a motor shield, and a prototyping shield.

Assembling components in this project will be slightly different than before because we will actually have two Arduino boards in the project: the Yún, which will receive commands directly from the outside world, and an Arduino Uno board, which will be connected to the motor shield.

We will then perform the usual test on the individual parts of the robot, such as testing the two DC motors of the robot and the ultrasonic distance sensor that is located at the front of the robot. To test the motor, we are simply going to make them accelerate gradually to see whether or not the command circuit is working correctly. The measurements being received from the ultrasonic distance sensor will simply be displayed on the serial monitor.

The next step is to build the Arduino software that will receive commands from the computer and transmit them to the motors that move the robot around. At this point, we are also going to code the part that will transmit the distance information back to the computer. Because we want to standardize our code and make it usable by other projects, we will build this part with inspiration from the REST API of the Arduino Yún board that we already used in [Chapter 2](ch02.html "Chapter 2. Creating a Remote Energy Monitoring and Control Device"), *Creating a Remote Energy Monitoring and Control Device*.

Finally, we are going to build the server-side graphical interface on your computer, so you can easily control the robot from your computer or a mobile device and receive some data about the robot, such as the readings from the ultrasonic sensor. This server-side software will again use HTML to display the interface, JavaScript to handle the users' actions, and PHP to talk directly to your Arduino Yún board via the `cURL` function.

# The required hardware and software components

You will need several mechanical and electrical components for this project apart from the Arduino Yún. The first set of components is for the robot itself. You basically need three things: a robot base or chassis that will support all the components, two DC motors with wheels so the robot can move around, and at least one ultrasonic sensor in front of the robot. We used a mobile robot kit from DFRobot ([http://www.dfrobot.com/](http://www.dfrobot.com/)) that you can see in the following image:

![The required hardware and software components](img/8007_04_02.jpg)

The kit is called the **2 Wheels miniQ Balancing Robot chassis** and costs $32.20 at the time of writing this book. Of course, you don't need this kit specifically to build this project. As long as you have a kit that includes the three kinds of components we mentioned before, you are probably good to go on this project.

For the motors, note that the circuit we used in the motor shield can handle up to 12V DC, so use motors that are made to work at a voltage under 12V. Also, use motors that have an integrated speed reducer. This way, you will increase the available torque of your motors (to make the robot move more easily).

For the ultrasonic sensor, you have many options available. We used one that can be interfaced via the `pulseIn()` function of Arduino, so any sensor that works this way should be compatible with the code we will see in the rest of this chapter. The reference of this sensor at DFRobot is URM37\. If you plan to use other kinds of distance sensors, such as sensors that work with the I2C interface, you will have to modify the code accordingly.

Then, you need an Arduino board that will directly interface with the DC motors via a motor shield. At this point, you might ask why we are not connecting all the components directly to the Arduino Yún without having another Arduino board in the middle. It is indeed possible to do with the sensors of the robot, but not the motors.

We can't connect the motors directly to an Arduino board; they usually require more current than what the Arduino pins can deliver. This is why we will use a motor shield that is specialized in that task. Usually, the Arduino Yún can't use these motor shields without being damaged, at least at the time of writing this book. This is due to the fact that motor shields are usually designed for Arduino Uno boards and the wrong pins on the shield can be connected to the wrong pins on the Yún. Of course, it would also be possible to do that with external components on a breadboard, but using a shield here really simplifies things.

This is why we will interface all the components with a standard Arduino board and then make the Yún board communicate with the standard Arduino board. We used a DFRduino board for this project, which is the name that DFRobot gave this clone of the Arduino Uno board. This is as shown in the following image:

![The required hardware and software components](img/8007_04_03.jpg)

Of course, any equivalent board will work as well, as long as it's compatible with the official Arduino Uno board. You could also use other boards, such as an Arduino Leonardo, but our code has not been tested on other boards.

Then, you need a motor shield to interface the two DC motors with the Arduino Uno board. We also used a motor shield from DFRobot for this project. The reference on the DFRobot website is **1A Motor Shield For Arduino**, as shown in the following image:

![The required hardware and software components](img/8007_04_04.jpg)

Again, most motor shields will work for this project. You basically need one shield that can command at least two motors. The shield also needs to be able to handle the motors you want to control in terms of voltage and current. In our case, we needed a shield that can handle the two 6V DC motors of the robot, with a maximum current of 1A.

Usually, you can look for motor shields that include the L293D motor driver IC. This integrated circuit is a chip dedicated to controlling DC motors. It can handle up to two 12V DC motors with 1A of current, which will work for the mobile robot we are trying to build here. Of course, if your shield can handle more current or voltage, that would work as well. The important point to look for is how to set the speed of the robot: the IC I mentioned can directly take a PWM command that comes from the Arduino board, so if you want to use the code prescribed in this chapter, you will need to use a shield that uses a similar type of command to set the motor's speed.

Finally, we added a simple prototyping shield on top of the robot to make power connections easier and so we can add more components in the future, as shown in the following image:

![The required hardware and software components](img/8007_04_05.jpg)

Again, you can use any equivalent prototyping shield, for example, the official prototype shield from Arduino. It is mainly so you don't have many cables lying around, but you can also use it to extend your robot project with more components, such as an accelerometer or a gyroscope.

You will also need a power source for your robot. As the DC motors can use quite a lot of current, we really recommend that you don't use power coming from your computer USB port when testing the robot or you will risk damaging it. That's why we will always use a battery when working with the motors of the robot. We used a 7.2V battery with a DC jack connector, so it can be easily inserted into the Arduino Uno board. This battery pack can also be found on the DFRobot website. You can also use some AA batteries instead of a battery pack. You will have to make sure that the total voltage of these batteries is greater than the nominal voltage of your DC motors.

As for the software itself, you don't need anything other than the Arduino IDE and a web server installed on your computer.

# Robot assembly

It's now time to assemble the robot. We will show you the steps you need to follow on the robot kit we used for this project, but they can be applied to any other equivalent robot kit. The first step is to put the battery at the base of the robot, as shown in the following image:

![Robot assembly](img/8007_04_06.jpg)

Note that some metal spacers were also used at the base of the robot to maintain the battery in place and to provide support for the rest of the components. These spacers can also be found on the DFRobot website. Then, you can screw on two more spacers and the Arduino Yún board to the top of the chassis, as shown in the following image:

![Robot assembly](img/8007_04_07.jpg)

Then, we added the Arduino Uno compatible board on top of the two metallic spacers. At this point, you can screw on the Arduino Uno board; all the other components will just be plugged into these boards, as shown in the following image:

![Robot assembly](img/8007_04_08.jpg)

Then, you can simply plug the motor shield on top of the Arduino Uno board. At this point, you can also connect the cables that come from the DC motors to the motor shield screw headers. Be careful with this step; it is quite easy to plug the wrong cables from the DC motors. You need to connect each motor on a different connector on the motor shield board, as shown in the following image:

![Robot assembly](img/8007_04_09.jpg)

Finally, you can plug the prototyping shield on top of the robot. At this point, we already connected the ultrasonic sensor: ground goes to Arduino ground, VCC to Arduino's 5V pin on the prototype shield, and the signal pin goes into pin A0 of the Arduino board. If your ultrasonic sensor works with a digital interface, for example, you might want to use different pins. Please read the datasheet of your ultrasonic sensor for more information. The following image shows the state of the robot at this step:

![Robot assembly](img/8007_04_10.jpg)

# Connecting the Arduino Yún and Uno boards

We are not done yet! For now, there are no connections between the Arduino Yún and the Arduino Uno board, so the Yún board won't be able to access the DC motors and the sensors of the robot. To solve this issue, the first step is to connect the power from the Arduino Uno board to the Yún board. This way, when we power the project using the battery, the Yún board will be powered as well.

To do so, simply connect the ground pins together and plug the Vin pin on the Arduino Yún to the 5V rail of the Arduino Uno, as shown in the following image:

![Connecting the Arduino Yún and Uno boards](img/8007_04_11.jpg)

To finish connecting the two Arduino boards, we need to connect them so they can speak together when the project is under operation. For this, we are going to use the I2C interface of the Arduino boards so they can send messages to each other. I2C stands for **Inter Integrated Circuit** and is a simple communication protocol that was developed for communication between circuits, and is widely used in electronics. There are two wires to connect for that purpose: SDA and SCL. To do so, simply connect pin 2 of the Yún board to pin A4 of the Uno board, and pin 3 of the Yún board to pin A5 of the Uno board, as shown in the following image:

![Connecting the Arduino Yún and Uno boards](img/8007_04_12.jpg)

The following image summarizes the connection between both boards:

![Connecting the Arduino Yún and Uno boards](img/8007OS_04_13.jpg)

Finally, you can power up the project by inserting the DC jack connector of the battery into the power connector of the Uno board as shown in the following image:

![Connecting the Arduino Yún and Uno boards](img/8007OS_04_14.jpg)

If everything was done correctly in this step, you should see that both boards (the Yún and the Uno) are powered up, with some of their LEDs on. To help you build the robot, we also included two pictures of the sides of the robot that show you the different connections. The following is an image of a side of the robot that shows the power connections to the Yún:

![Connecting the Arduino Yún and Uno boards](img/8007OS_04_15.jpg)

The following image shows the connections from the I2C interface to the Yún:

![Connecting the Arduino Yún and Uno boards](img/8007OS_04_16.jpg)

# Testing the robot's hardware connections

Before building the remote control part of the project, we want to make sure that the hardware is wired correctly, especially between the Arduino Uno board and the different motors and sensors. This is why we are first going to build a simple sketch for the Arduino Uno board to test the different components.

At this point, we are going to turn the motors of the robot on; so make sure the robot is standing on a small platform, for example, to prevent it from moving around while you are testing your different Arduino sketches with the USB cable connected to your computer.

The sketch starts by declaring the pins for the motors, as shown in the following code. Note that these pins are specifically for the motor shield we are using; please refer to the datasheet of your shield if you are using a different one.

[PRE0]

Declare the pin used by the ultrasonic sensor as follows:

[PRE1]

We also want to make the speed of the motor vary during operation, so we declare the variable as follows:

[PRE2]

In the `setup()` part of the sketch, we need to specify that the motor pins will behave as output pins, as shown in the following code:

[PRE3]

We also need to set a starting speed for the robot. Note that the speed of each motor will be set by PWM commands coming from the Arduino, so we have to specify a value between 0 (no voltage applied to the motor) and 255 (maximum voltage applied to the motor). Also, because of mechanical resistance on the motors, there is no linear relation between the value of the PWM command and the speed of the motor.

We used the value 75 as a starting speed, which is a very slow speed on our DC motors. However, depending on your own setup, this value will have a completely different effect. At this point, you can also experiment to see what the maximum PWM value is that will give you exactly zero speed on your DC motors. Make sure that the robot is not on the floor just yet as it would start to move forward and possibly damage things. We put it on a small stand so the wheels don't touch anything.

In the `loop()` part, everything is done by the function `send_motor_command`, which will be called for both motors. For example:

[PRE4]

Let's see the details of this function. It starts by writing the speed of the motor on the correct pin as follows:

[PRE5]

Then, we need to set the the direction pin to the correct direction. This is done by a simple `digitalWrite` function as follows:

[PRE6]

Still in the `loop()` function, we call a function to measure the distance in front of the robot and print the result on the `Serial` port:

[PRE7]

Let's see the details of this function. It starts by getting the raw measurement from the sensor using the `pulseIn` function. Basically, the sensor returns a pulse whose length is proportional to the measured distance. The length of the pulse is measured with the following function of Arduino dedicated for that purpose:

[PRE8]

Then, we check whether the reading is valid and if it is, we convert it to centimeters using the following formula:

[PRE9]

This is returned with the following code:

[PRE10]

Finally, we update the speed at every iteration of the loop by increasing it by one unit, and we reset it if it reaches 255, as shown in the following code:

[PRE11]

The code for this section is available at the GitHub repository of the book and is stored in a file called `robot_test`: [https://github.com/openhomeautomation/geeky-projects-yun/tree/master/chapter4/robot_test](https://github.com/openhomeautomation/geeky-projects-yun/tree/master/chapter4/robot_test)

It's now time to upload the code to the robot. Before doing so, please make sure that the robot is powered by the battery. Both motors of the robot should gradually accelerate upon reaching the maximum speed and then start again at a lower speed.

You can also open the serial monitor at this point to check the readings from the distance sensor. Try moving your hand or an object in front of the robot; you should see the distance changing accordingly on the serial monitor.

# Building the Arduino sketch

It's now time to build the final sketch for our project. To be really precise, we should say sketches because we will have to develop two of them: one for the Uno board and one for the Yún board. You just have to make one simple change to the hardware at this point: connect the ultrasonic sensor directly to the Yún board by connecting the signal pin to the pin A0 of the Yún board.

Let's first focus on the Arduino Uno sketch. The sketch is inspired by the test sketch we wrote before, so it already includes the functions to control the two DC motors. To communicate between the two boards, we have to include the Wire library that is in charge of handling I2C communications:

[PRE12]

Then, in the `setup()` part of the sketch, we need to declare that we are connecting to the I2C bus and start listening for incoming events. The Uno board will be configured as a slave, receiving commands from the Yún board, which will act as the master. This is done by the following piece of code:

[PRE13]

Let's see the details of this `receiveEvent` part, which is actually a function that is passed as an argument to the `onReceive()` function of the Wire library. This function will be called whenever an event is received on the I2C bus. What this function does is basically read the incoming data from the Yún, which has to follow a specific format like you can see in the following example:

[PRE14]

For example, the first part of the previous message is read back with the following code:

[PRE15]

These commands that come from the Yún are then applied to the motors as follows:

[PRE16]

Let's now focus on the Yún sketch. This sketch is inspired by the Bridge sketch that comes with the Arduino IDE and is based on the REST API of the Arduino Yún. To make things easier, we are going to create a new kind of REST call named robot. This way, we are going to be able to command the robot by executing calls like the following in your browser:

[PRE17]

First, we need to include the correct libraries for the sketch as follows:

[PRE18]

Then, create a web server on the board:

[PRE19]

In the `setup()` function, we also join the I2C bus:

[PRE20]

Then, we start the bridge:

[PRE21]

The `setup()` function ends by starting the web server as follows:

[PRE22]

Then, the `loop()` function consists of listening to incoming connections as follows:

[PRE23]

The requests that come from these clients can be processed with the following command:

[PRE24]

If a client is connected, we process it to check whether or not a robot command was received, as follows:

[PRE25]

This function processes the REST call to see what we need to do with the motors of the robot. For example, let's consider the case where we want to make the robot go forward at full speed. We need to send the following message to the Arduino Uno board:

[PRE26]

This is done by the following piece of code:

[PRE27]

We included three other commands for this simple REST API: `stop` (which obviously stops the robot), `turnleft` (which makes the robot turn left at moderate speed), `turnright` (which makes the robot turn right), and `getdistance` to return the distance coming from the ultrasonic sensor. We also inserted the `measure_distance` function in the sketch to read data that comes from the ultrasonic sensor.

We are now ready to upload the code to the robot. Remember that you have to upload two sketches here: one for the Uno board and one for the Yún board. The order doesn't matter that much, just upload the two Arduino sketches successfully by carefully ensuring that you are uploading the correct code to the correct board.

Both sketches are available in the following repository on GitHub: [https://github.com/openhomeautomation/geeky-projects-yun/tree/master/chapter4/remote_control](https://github.com/openhomeautomation/geeky-projects-yun/tree/master/chapter4/remote_control).

You can then test that the Yún board is correctly relaying commands to the Uno board. At this point, you can disconnect all cables and power the robot with the battery only. Then, go to a web browser and type the following code:

[PRE28]

The robot should instantly start turning to the right. To stop the robot, you can simply type the following code:

[PRE29]

You can also type the following code:

[PRE30]

This should print the value of the distance in front of the robot on your web browser. If you can see a realistic distance being printed on your web browser, it means that the command is working correctly.

# Building the computer interface

We are now going to build an interface so you can control the robot remotely from your computer or a mobile device. This is actually quite similar to what we did for the relay control project, the main difference being that we also want to read some data back from the robot (in the present case the distance measurement from the ultrasonic sensor). There will be an HTML file that will host the different elements of the interface, some PHP code to communicate with the Yún board, some JavaScript to establish the link between HTML and PHP, and finally some CSS to give some style to the interface.

The first step is to create the HTML file that will be our access point to the robot control. This file basically hosts four buttons that we will use to control our robot and a field to continuously display the distance measured by the ultrasonic sensor. The buttons are declared inside a form; the following is the code for one button:

[PRE31]

The distance information will be displayed using the following line of code:

[PRE32]

The following field will be updated with some JavaScript:

[PRE33]

Let's see the content of this PHP file. It basically makes a call to the REST API of the Yún board, and returns the answer to be displayed on the interface. Again, it will make use of the `curl` function of PHP.

It starts by making the `cURL` call to your Yún board with the `getdistance` parameter we defined in the sketch before:

[PRE34]

It then prepares the call with the following code:

[PRE35]

We get the answer with the following code:

[PRE36]

We then print it back with the `echo` function of PHP:

[PRE37]

The PHP script that commands the motors is quite similar, so we won't detail it here.

Let's see the JavaScript file that handles the different buttons of the interface. Each button of the interface is basically linked to a JavaScript function that sends the correct parameter to the Arduino Yún, via the PHP file. For example, the `stop` button calls the following function:

[PRE38]

The same is done with the function to make the robot go full speed forward. To make it turn left or right, we can implement a more complex behavior. What we usually want is not for the robot to turn continuously by itself, but for example, to turn off a quarter of a turn. This is where the approach we took in this project becomes powerful. We can do that right on the server side without having to change the sketch on the Arduino board.

That's why to turn right for a given amount of time, for example, we will implement a series of commands on the server side and then stop. This is done by the following code:

[PRE39]

The `sleep` function itself is implemented in the same file and works by comparing the time that passed since the function was called, as shown in the following code:

[PRE40]

Of course, we invite you to play with this sleep function to get the desired angle. For example, we set our sleep function such that the robot turns off about a quarter of a turn whenever we press the **Turn Right** button.

The code for the interface is available on the GitHub repository of the project: [https://github.com/openhomeautomation/geeky-projects-yun/tree/master/chapter4/remote_control](https://github.com/openhomeautomation/geeky-projects-yun/tree/master/chapter4/remote_control)

Now, it's time to start the project. Be sure to place all the files at the root of your web server and make sure that the web server is running. Then, go to the folder of your web server in your browser (usually by typing `localhost`) and open the HTML file. The project also contains a CSS sheet to make the interface look better. The following is what you should see in your browser:

![Building the computer interface](img/8007OS_04_17.jpg)

The field that displays the distance reading from the ultrasonic sensor should be updated automatically every second, so you can see whether or not this is working right away. Try moving your hand or an object in front of the robot and the value should change accordingly.

Before making the robot move around, we recommend that you test the different buttons while the robot is still on a small stand so it cannot move. Indeed, if something is wrongly coded on your server or within the Arduino sketch, your robot will not respond anymore and will randomly hit objects in your home.

You can now also test the different buttons. You can especially focus on the buttons that make the robot turn left or right and adjust the `sleep()` function in the PHP code to make them do exactly what you want. Notice that while your robot is moving around, the distance detected by the ultrasonic sensor in front of the robot is updated accordingly.

# Summary

Let's see what the major takeaways of this chapter are:

*   We started the project by building the robot from the different components, such as the robot base, the DC motors, the ultrasonic sensor, and the different Arduino boards.
*   Then, we built a simple sketch to test the DC motors and the ultrasonic distance sensor.
*   The next step was to build two Arduino sketches to control the robot remotely: one for the Arduino Uno board and one for the Yún board.
*   At the end of the project, we built a simple web interface to control the robot remotely. The interface is composed of several buttons to make the robot move around, and one field that continuously displays the measurement that comes from the ultrasonic sensor mounted in front of the robot.

Let's now see what else you can do to improve this project. You can, for example, use the ultrasonic sensor data to make the robot act accordingly, for instance, to avoid hitting into walls.

Finally, you can also add many hardware components to the robot. The first thing you can do is add more ultrasonic sensors around the robot so you can detect obstacles to the sides of the robot as well. You can also imagine adding an accelerometer and/or a gyroscope to the robot so you will know exactly where it is going and at what speed.

You can even imagine combining the project with the one from the [Chapter 3](ch03.html "Chapter 3. Making Your Own Cloud-connected Camera"), *Making Your Own Cloud-connected Camera*, and plug a USB camera to the robot. This way, you can live stream what the robot is seeing while you control it with the web interface!

I hope this book gave you a good overview of what the Arduino Yún can add to your Arduino projects. Through the four projects in the book, we used the three main features of the Arduino Yún: the powerful embedded Linux machine, the onboard Wi-Fi connection, and the Temboo libraries to interface the board with web services. You can now use what you learned in this book to build your own applications based on the Arduino Yún!