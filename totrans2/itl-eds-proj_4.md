# Intel Edison and Security System

In previous chapters, we learned how we can use the Intel Edison to develop applications related to IoT where we displayed live sensor data and also controlled the Edison itself. We also learned the development of an Android and a WPF app that was used to control the Intel Edison. Well, this chapter is more on the local front of the Intel Edison where we are going to use the built-in features of the device. This chapter is concentrated mainly on two key points:

*   Speech and voice processing with the Intel Edison
*   Image processing with the Intel Edison

All the codes are to be written in Python, so some parts of the chapter will concentrate on Python programming as well. In this chapter, we'll operate the Intel Edison using voice commands and then ultimately detect faces using the Intel Edison and a webcam. This chapter will thus explore the core capabilities of the Intel Edison. Since most of the code is in Python, it is advisable to download Python for your PC from the following website:

[https://www.python.org/downloads/](https://www.python.org/downloads/)

This chapter will be divided into two parts. The first part will concentrate on only speech or voice processing and we'll do a mini-project based on that while the second part, which will be a bit lengthy, will concentrate on the image processing aspect of it using OpenCV.

# Speech/voice processing using Edison

Speech processing typically refers to the various mathematical techniques that are applied on an audio signal to process it. It may be some simple mathematical operation or some complex operation. It's a special case of digital signal processing. However, we are not typically dealing with speech processing as a whole entity. We are interested only in a specific area of speech to text conversion. It is to be noted that everything in this chapter is to be performed by the Edison itself without accessing any cloud services. The scenario that this chapter will tackle initially is that we'll make the Edison perform some tasks based on our voice commands. We'll be using a lightweight speech processing tool, but before we proceed further with all the code and circuits, make sure you have the following devices with you. Initially, we'll walk you through switching an LED on and off. Next we'll deal with controlling a servo motor using voice commands.

# Devices required

Along with the Intel Edison, we need a couple of more devices, as listed here:

*   Power adapter of 9V-1 A for the Intel Edison
*   USB sound card
*   USB hub, preferably powered

This project will use the Edison on external power and the USB port will be used for the sound card. A non-powered USB hub also works, but because of the current it's recommended to use a powered USB hub.

Make sure that the USB sound card is supported on a Linux environment. The selector switch should be towards the USB port. That is because the Edison will be powered through the DC adapter and we need power in the USB port that is activated only when we provide DC power.

# Speech processing library

For this project we are going to use PocketSphinx. It's a lightweight version of CMU Sphinx, a project created by Carnegie Mellon University. It's a lightweight speech recognition engine meant for mobile and handheld devices and wearables. The greatest advantage of using this over any cloud-based service is that it is available offline.

More information about PocketSphinx can be accessed from the following links:

[http://cmusphinx.sourceforge.net/wiki/develop](http://cmusphinx.sourceforge.net/wiki/develop)

[https://github.com/cmusphinx/pocketsphinx](https://github.com/cmusphinx/pocketsphinx)

Setting up the library will be discussed in a later section of this chapter.

# Initial configuration

In the first chapter, we performed some very basic configuration for the Intel Edison. Here we need to configure our device with the required libraries and sound setup with the sound card. For this you need to connect the Intel Edison to only one micro USB port. This will be used to communicate using the PuTTY console and transfer files using the FileZilla FTP client:

![](img/6639_04_01.jpg)

Arduino expansion board components

Connect the Intel Edison to the Micro B USB for serial interface to your PC.

Some of the steps were covered in [Chapter 1](c225d705-919a-4442-adc8-7b22d33437fc.xhtml), *Setting up Intel Edison;* however, we'll show all the steps from the beginning. Open your PuTTY console and log in to your device. Use the `configure_edison -wifi` to connect to your Wi-Fi network.

Initially, we'll add AlexT's unofficial `opkg` repository. To add this, edit the `/etc/opkg/base-feeds.conf` file.

Add these lines to the preceding file:

[PRE0]

To do that, execute the following command:

[PRE1]

Update the package manager:

[PRE2]

Install `git` using the package manager:

[PRE3]

We will now install Edison helper scripts to simplify things a bit:

1.  First `clone` the package:

[PRE4]

2.  Now we have to add `~/edison-scripts` to the path:

[PRE5]

3.  Next we will run the following scripts:

[PRE6]

The initial configuration is done. Now we'll configure the Edison for sound.

4.  Now `install` the modules for USB devices, including USB webcams, microphone, and speakers. Make sure that your sound card is connected to the Intel Edison:

[PRE7]

5.  The next target is to check whether the USB device is getting detected or not. To check that, type the `lsusb` command:

![](img/6639_04_02.jpg)

USB sound card

The device that is connected to the Intel Edison is shown in the preceding screenshot. It is highlighted in the box. Once we get the device that is connected to the Edison, we can proceed further.

Now we'll check whether `alsa` is able to detect the sound card or not. Type in the following command:

[PRE8]

![](img/6639_04_03.png)

Alsa device check

It is noted that our device is getting detected as card 2, named as `Device`.

Now we have to create a `~/.asoundrc` file where we need to add the following line. Please note that `Device` must be replaced with the device name that is detected on your system:

[PRE9]

Now, once this is done, exit and save the file. Next, to test whether everything is working or not, execute the following command and you must hear something on the headphone connected:

[PRE10]

You should hear the words `Front Center`.

Now, our target is to record something and interpret the result. So let's test whether recording is working or not.

To record a clip, type in the following command:

[PRE11]

Press *Ctrl *+ *c* to stop recording. To play the preceding recording, type the following:

[PRE12]

You must hear what you have recorded. If you are not able to hear the sound, type `alsamixer` and adjust the playback and record volumes. Initially, you need to select the device:

![](img/6639_04_04.png)

Alsamixer—1

Next, adjust the volume using the arrow keys:

![](img/6639_04_05.png)

Alsamixer—2

Now everything related to sound is set up. The next aim is to `install` the packages for speech recognition.

Initially, use Python's `pip` to `install cython`:

[PRE13]

The preceding package takes a lot of time to install. Once that's done, there are some shell scripts that are required to be executed. I have created a GitHub repository for this that contains the required files and the code. Use the git command to clone the repository ([https://github.com/avirup171/Voice-Recognition-using-Intel-Edison.git](https://github.com/avirup171/Voice-Recognition-using-Intel-Edison.git)):

[PRE14]

Next in the bin folder, you will find the packages. Before typing the commands to execute those shell scripts, we need to provide permissions. Type the following command to add permissions:

[PRE15]

Next type the filename to execute them. Installation of the packages may take a bit of time:

[PRE16]

Next type these for adding to the path:

[PRE17]

Next install `Pocketsphinx`:

[PRE18]

Finally, install `PyAudio`:

[PRE19]

After this step, all the configurations are set up and we are good to go with the coding. PocketSphinx works with some specific sets of commands. We need to create a language mode and a dictionary for the words to be used. We'll do that using the Sphinx knowledge base tool:

[http://www.speech.cs.cmu.edu/tools/lmtool-new.html](http://www.speech.cs.cmu.edu/tools/lmtool-new.html)

Upload a text file containing the set of commands that we want the engine to decode. Then click on COMPILE KNOWLEDGE BASE. Download the `.tgz` file that contains the necessary files that are required. Once we have those files, copy it to the Edison using FileZilla. Note the names of the files that contain the following extension. Ideally each file should have the same name:

*   `.dic`
*   `.lm`

Move the entire set to the Edison.

# Writing the code

**Problem statement**: To turn on and off an LED using voice commands such as `ON` and `OFF`.

Before writing the code, let us discuss the algorithm first. Please note that I am writing the algorithm in plain text so that it is easier for the reader to understand.

# Let's start with the algorithm

Perform the following steps to begin with the algorithm:

1.  Import all the necessary packages.
2.  Set the LED pin.
3.  Start an infinite loop. From here on, all the parts or blocks will be inside the while loop.
4.  Store two variables in the path for the `.lm` and `.dic` files.
5.  Record and save a `.wav` file for `3` seconds.
6.  Pass the `.wav` file as a parameter to the speech recognition engine.
7.  Get the resultant text.
8.  With an `if else` block test for the `ON` and `OFF` texts and use the `mraa` library to turn on and off an LED.

The algorithm is pretty much straightforward. Compare the following code with the preceding algorithm to get a full grip of it:

[PRE20]

Let's go line by line:

[PRE21]

The preceding segment is just to `import` all the libraries and packages:

[PRE22]

We set the LED pin and set its direction as the output. Next we will begin the infinite while loop:

[PRE23]

The preceding chunk of code is just the parameters for PocketSphinx and for audio recording. We will be recording for `3` seconds. We have also provided the path for the `.lmd` and `.dic` files and some other audio recording parameters:

[PRE24]

In the preceding code, we record the audio for the specific time interval.

Next, we save it as a `.wav` file:

[PRE25]

The final step contains the decoding of the file and comparing it to affect the LED:

[PRE26]

In the preceding code, we initially pass the `.wav` file as a parameter to the speech processing engine and then use the result to compare the output. Finally, we switch on and off the LEDs based on the output of the speech processing engine. Another activity carried out by the preceding code is that whatever is recognized is spoken back using `espeak`. `espeak` is a text to speech engine. It uses spectral formant synthesis by default, which sounds robotic, but can be configured to use Klatt formant synthesis or MBROLA to give it a more natural sound.

Transfer the code to your device using FileZilla. Let's assume that the code is saved by the file named `VoiceRecognitionTest.py`.

Before executing the code, you may want to attach an LED to GPIO pin 13 or just use the on board LED for the purpose.

To execute the code, type the following:

[PRE27]

Initially, the console says `*recording`, speak `on`:

![](img/image_04_009.png)

Voice recognition—1

Then, after you speak, the speech recognition engine will recognize the word that you spoke from the existing language model:

![](img/image_04_010.png)

Voice recognition—2

It is noted that on is displayed. That means that the speech recognition engine has successfully decoded the speech we just spoke. Similarly, the other option stands when we speak off on the microphone:

![](img/image_04_011.png)

Voice recognition—3

So now we have a voice recognition proof of concept ready. Now, we are going to use this concept with small modifications to lock and unlock the door.

# Door lock/unlock based on voice commands

In this section, we'll just open and close a door based on voice commands. Similar to the previous section, where we switched an LED on and off using voice commands such as `ON` and `OFF`, here we are going to do a similar thing using a servo motor. The main target is to make the readers understand the core concepts of the Intel Edison where we use voice commands to perform different tasks. The question may arise, why are we using servo motors?

A servo motor, unlike normal DC motors, rotates up to a specific angle set by the operator. In normal scenarios, controlling the lock of a door may use a relay. The usage of relays was discussed in [Chapter 3](3bd53219-a287-4d8f-9a58-5a06c5b14062.xhtml), *Intel Edison and IoT (Home Automation)*.

Let us also explore the use of servo motors so that we can widen the spectrum of controlling devices. In this case, when a servo is set to `0` degrees, it is unlocked and when it is set to `90` degrees, it is locked. The control of servo motors requires the use of pulse width modulation pins. Intel Edison has four PWM pins:

![](img/6639_04_09.jpg)

Servo motor. Picture credits: [https://circuitdigest.com](https://circuitdigest.com)

There are three operating lines to operate a servo:

*   Vcc
*   Gnd
*   Signal

The typical color coding goes like this:

*   Black—ground
*   Red or brown—power supply
*   Yellow or white—control signal

We are using a 5V servo motor; therefore the Edison is enough to supply power. The Edison and the servo motor must share a common ground. Finally, the signal pin is connected to the PWM pin on the Intel Edison. As we move further with this mini-project, things will get clearer.

# Circuit diagram

The following is the circuit diagram for voice recognition:

![](img/6639_04_10-1.jpg)

Circuit diagram for voice recognition

As already mentioned, the servo motor requires PWM pins for operation and the Intel Edison has a total of six PWM pins. Here we are using digital pin 6 for servo control and digital pin 13 for the LED. As far as the peripheral devices are concerned, connect your USB sound card to the USB of the Intel Edison and you are all set.

# Configuring the servo library for Python

To control a servo, we need to send some signals through the PWM pins. We opt for using a library for controlling the servo motors.

Use the following link to get access to a `Servo.py` Python script from a GitHub repository:

[https://github.com/MakersTeam/Edison/blob/master/Python-Examples/Servo/Servo.py](https://github.com/MakersTeam/Edison/blob/master/Python-Examples/Servo/Servo.py)

Download the file and push it to your Edison device. After that, just execute the file similar to executing a Python script:

[PRE28]

Now once you have done, you are ready to use the servo with your Intel Edison using Python.

Getting back to the hardware, the servo must be connected to digital pin `6`, which is a PWM pin. Let's write a Python script that will test if the library is functioning or not:

[PRE29]

The preceding code basically sweeps from `0` to `180` degrees and back to `0` degrees. The circuit remains the same as discussed before. Initially, we attach the servo to the servo pin. Then as the standard goes, we put the entire logic in an infinite loop. To rotate the servo to a specific angle, we use `.write(angle)`. In the two for loops, initially we rotate from `0` to `180` degrees and in the second one, we rotate from `180` to `0` degrees.

It is also to be noted that `time.sleep(time_interval)` is used to pause the code for some miliseconds. When you execute the preceding code, the servo should rotate and come back to the initial position.

Now, we have all the things in place. We'll just put them in the right place and your voice controlled door will be ready. Initially, we controlled an LED and then we learned how we can operate a servo using Python. Now let's create a language model using the Sphinx knowledge base tool.

# Language model

For this project, we'll be using the following set of commands. To keep things simple, we're using only two sets of commands:

*   `door open`
*   `door close`

Follow the process that was discussed earlier and create a text file and just write the three unique words:

[PRE30]

Save it and upload it to the Sphinx knowledge base tool and compile it.

Once you have the compressed file downloaded, move on to the next step with this code:

[PRE31]

The preceding code is more or less similar to the code for switching an LED on and off. The only difference is that the servo control mechanism is added into the existing code. In a simple if else block, we check for the `door open` and `door close` conditions. Finally based on what is triggered, we set the LED and the servo to a `90` degrees or `0` degree position.

# Conclusion of speech processing using the Intel Edison

From the projects discussed before, we explored one of the core capabilities of the Intel Edison and explored a whole new scenario of controlling the Intel Edison by voice. A popular use case that implements the preceding procedure can be the case of home automation, which was implemented in the earlier chapter. Another use case is building a virtual voice based assistant using your Intel Edison. There are multiple opportunities that can be used using voice-based control. It's up to the reader's imagination as to what they want to explore.

In the next part, we'll be dealing with the implementation of image processing using the Intel Edison.

# Image processing using the Intel Edison

Image processing or computer vision is one such field that requires tremendous amounts of research. However, we're not going to do rocket science here. We are opting for an open source computer vision library called `OpenCV`. `OpenCV` supports multiple languages and we are going to use Python as our programming language to perform face detection.

Typically, an image processing application has an input image; we process the input image and we get an output processed image.

Intel Edison doesn't have a display unit. So essentially we will run the Python script on our PC first. Then after the successful working of the code in the PC, we'll modify the code to run on the Edison. Things will get clearer when we do the practical implementation.

Our target is to perform face detection and, if detected, perform some action.

# Initial configuration

The initial configuration will include installing the `openCV` package both on the Edison device as well as the PC.

For the PC, download Python from [https://www.python.org/downloads/windows/](https://www.python.org/downloads/windows/). Next install Python on your system. Also download the latest version of openCV from [https://sourceforge.net/projects/opencvlibrary/](https://sourceforge.net/projects/opencvlibrary/).

After you download openCV, move the extracted folder to `C:\`. Next, browse to `C:\opencv\build\python\2.7\x86`.

Finally, copy the `cv2.pyd` file to `C:\Python27\Lib\site-packages`.

We need to install `numpy` as well. Numpy stands for **Numerical Python**. Download and install it.

Once you install all the components, we need to test whether everything is installed or not. To do that, open up the idle Python GUI and type the following:

[PRE32]

If this proceeds without any error, then everything is installed and in place as far as the PC configuration is concerned. Next, we'll configure for our device.

To configure your Edison with openCV, initially execute the following:

[PRE33]

Finally, after the preceding is successfully executed, run the following:

[PRE34]

This should install all the necessary components. To check whether everything is set up or not, type the following:

[PRE35]

And press the *Enter* key. This should enter into the Python shell mode. Next, type the following:

[PRE36]

Here is the screenshot of this:

![](img/6639_04_11.png)

Python shell

If this doesn't return any error message, then you are all set to go.

At first we will be covering everything in the PC and after that we'll move on to deploy it to the Intel Edison.

# Real-time video display using OpenCV

Before we move on to face detection, let's first see whether we can access our camera or not. To do that, let's write a very simple Python script to display the webcam video feed:

[PRE37]

In the preceding code, we initially import the openCV module as `import cv2`.

Next we initialize the video capture device and set the index to zero as we're using the default webcam that comes with the laptop. For desktop users, you may need to vary the parameter.

After the initialization, in an infinite loop, we read the incoming video frame by frame using `cap.read()`:

[PRE38]

Next we apply some operations on the incoming video feed. Here in the sample, we convert the RGB video frame to a grayscale image:

[PRE39]

Finally, the frames are displayed in a separate window:

[PRE40]

In the preceding two lines, we implement the mechanism of keyboard interrupts. When someone presses *q* or presses the *Esc* key, the display will close.

Once you get the incoming video feed, then we are ready to move to face detection.

# Face detection theory

Face detection is a very specific case of object recognition. There are many approaches to face recognition. However, we are going to discuss the two given here:

*   Segmentation based on color
*   Feature-based recognition

# Segmentation based on color

In this technique, the face is segmented out based on skin color. The input of this is typically an RGB image, while in the processing stage we shift it to **Hue saturation value** (**HSV**)or YIQ ( Luminance (Y), In-phase Quadrature) color formats. In this process, each pixel is classified as a skin-color pixel or a non-skin-color pixel. The reason behind the use of other color models other than RGB is that sometimes RGB isn't able to distinguish skin colors in different light conditions. This significantly improves while using other color models.

This algorithm won't be used here.

# Feature-based recognition

In this technique, we go for certain features and based on that we do the recognition. Use of the haar feature-based cascade for face detection is an effective object detection method proposed by Paul Viola and Michael Jones in their paper "*Rapid Object Detection using a Boosted Cascade of Simple Features*" in 2001\. It is a machine learning based approach where a cascade function is trained against a set of positive and negative images. Then it is used to detect objects in other images.

The algorithm initially needs a lot of positive images. In our case, these are images of faces, while negative images which don't contain images of faces. Then we need to extract features from it.

For this purpose, the haar features shown in the following figure are used. Each of the features is a single value obtained by subtracting the sum of pixels under a white rectangle from sum of pixels under a black rectangle:

![](img/6639_04_12.jpg)

Haar features

The haar classifiers need to be trained for face, eyes, smile, and so on. OpenCV contains a set of predefined classifiers. They are available in the `C:\opencv\build\etc\haarcascades` folder. Now that we know how we can approach face detection, we are going to use the pre-trained haar classifiers for face detection.

# Code for face detection

The following is the code for face detection:

[PRE41]

Let's look at the code line by line:

[PRE42]

Import all the required modules:

[PRE43]

We select the cascade classifier file. Also we select the video capture device. Make sure you mention the path correctly:

[PRE44]

In the preceding lines, which are inside the infinite while loop, we read the video frame and convert it from RGB to grayscale:

[PRE45]

The preceding line is the most important part of the code. We have actually applied the operation on the incoming feed.

`detectMultiScale` consists of three important parameters. It is a general function for detecting images and since we are applying the face haar cascade, therefore we are detecting faces:

*   The first parameter is the input image that needs to be processed. Here we have passed the grayscale version of the original image.
*   The second parameter is the scale factor, which provides us with the factor for the creation of a scale pyramid. Typically, around 1.01-1.5 is an appropriate one. The higher the value, the speed increases, but the accuracy decreases.
*   The third parameter is `minNeighbours` which affects the quality of the detected regions. A higher value results in less detection. A range of 3-6 is good:

[PRE46]

The preceding lines simply draw rectangles around the faces.

Finally, we display the resultant frame and use the keyboard interrupts to release the video capture device and destroy the window.

Now press *F5* to run the code. Initially, it will ask to save the file, and then the execution will begin:

![](img/image_04_017.png)

Screenshot of the image window with a face detected

Until now, if everything is carried out in a proper way, you must have a brief idea about face detection and how it can be accomplished using openCV. But now, we need to transfer it to the Intel Edison. Also we need to alter certain parts to meet the capabilities of the device as it doesn't have a display unit and above all it has a RAM of 1 GB.

# Intel Edison code

For the Intel Edison, let's find out what is actually possible. We don't have a display, so we can rely only on console messages and LED, perhaps, for visual signals. Next, we may need to optimize the code to run on the Intel Edison. But first let's edit the code discussed previously to include an LED and some kind of messages to the picture:

[PRE47]

Since the Intel Edison has only one USB port, therefore we have mentioned the parameter of `cv2.VideoCapture` as `0`. Also notice the following line:

[PRE48]

You will notice that the parameters have been changed to optimize them for the Intel Edison. You can easily tamper with the parameters to get a good result.

We have included some lines for LED on and off:

![](img/6639_04_14.png)

Console output for face detection in images using openCV

This is when you begin to notice that the Intel Edison is simply not meant for image processing because of the RAM.

Now when you are dealing with high processing applications, we cannot rely on the processing power of the Intel Edison alone.

In those cases, we opt for cloud-based solutions. For cloud-based solutions, there are multiple frameworks that exist. One of them is Project Oxford by Microsoft ([https://www.microsoft.com/cognitive-services](https://www.microsoft.com/cognitive-services)).

Microsoft Cognitive Services provides us with APIs for face detection, recognition, speech recognition, and many more. Use the preceding link to learn more about them.

After all the discussions that we've had in this chapter, we now know that voice recognition performs reasonably well. However, things are not so good with image processing. But why are we focused on using it? The answer lies in that the Intel Edison can definitely be used as an image gathering device while other processing can be carried out on the cloud:

![](img/6639_04_15.jpg)

Security-based systems architecture at a glance

Processing can either be performed at the device end or at the cloud end. It all depends on the use case and the availability of resources.

# Open-ended task for the reader

The task for this chapter may require a bit of time, but the end result is going to be awesome. Implement Microsoft Cognitive Services to perform facial recognition. Use the Edison to gather data from the user and send it to the service for processing and perform some actions based on the result.

# Summary

Throughout this chapter, we have learned some techniques of voice recognition using the Intel Edison. We also learned how image processing can be done in Python and implemented the same on the Intel Edison. Finally, we explored how a real life security-based system would look like and an open-ended question for Microsoft Cognitive Services.

[Chapter 5](45fccd6a-a75e-465d-89dc-dad31f528ac1.xhtml), *Autonomous Robotics with Intel Edison*, will be entirely dedicated to robotics and how the Intel Edison can be used with robotics. We'll be covering both autonomous and manual robotics.