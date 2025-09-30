# Chapter 4. Using Media Files

JUCE provides its own classes for reading and writing files and many helper classes for specific media formats. This chapter introduces the main examples of these classes. In this chapter we will cover the following topics:

*   Using simple input and output streams
*   Reading and writing image files
*   Playing audio files
*   Working with the Binary Builder tool to turn binary files into source code

By the end of this chapter, you will be able to manipulate a range of media files using JUCE.

# Using simple input and output streams

In [Chapter 3](ch03.html "Chapter 3. Essential Data Structures"), *Essential Data Structures*, we introduced the JUCE `File` class, which is used for specifying file paths in a cross-platform manner. In addition, the `File` class includes some convenience functions for reading and writing files as blocks of data or strings of text. In many cases these functions are sufficient, but in others, raw access to input and output streams may be more useful.

## Reading and writing text files

First, create a console application project in the Introjucer application and name it `Chapter04_01`. In this simple example, we will declare two functions, one for writing text to the file—`writeFile()` , and one for reading the contents of the file—`readFile()` . Each of these functions is passed the same file path reference created in the way we did in [Chapter 3](ch03.html "Chapter 3. Essential Data Structures"), *Essential Data Structures*. Replace the contents of the `Main.cpp` file with the following, where we declare the file reading and writing functions, and define a `main()` function:

[PRE0]

Then, add the definition for the `writeFile()` function:

[PRE1]

Here, we create a `FileOutputStream` object, passing it the `File` object that refers to the file path. The `FileOutputStream` class inherits from the base class `OutputStream` that represents the general notions of writing data to a stream. There can be other types of output stream, such as the `MemoryOutputStream` class for writing data to areas of computer memory in a stream-like manner. The default behavior of the `FileOutputStream` class on construction is to position the stream's write position at the end of the file if the file already exists (or to create an empty file if it doesn't). The calls to the `FileOutputStream::setPosition()` and `FileOutputStream::truncate()` functions effectively empty the file each time before we write it. Of course, in a real application you may not want to do this each time. The call to the `FileOutputStream::writeText()` function is almost equivalent to the `File::appendText()` function, although the flags for controlling the output in Unicode UTF16 format are implicit for the `File::appendText()` function, but need to be specified explicitly for the `FileOutputStream::writeText()` function. Here, we write the data in UTF8 format by setting both flags to `false`.

### Tip

The UFT8 format is probably most convenient, because the text we are writing is plain ASCII text, which is compatible with UTF8 encoding.

Finally, add the definition for the `readFile()` function:

[PRE2]

Here, we attempt to read the entire stream into a `String`, and post it to the log. We use a `FileInputStream` object, which inherits from the more general `InputStream` class. In both the `writeFile()` and `readFile()` functions we check that the streams opened successfully before proceeding. In addition to this, the stream objects gracefully close the streams when they go out of scope.

## Reading and writing binary files

The output and input streams can be used for binary data too, and offer much greater functionality over the `File` class convenience functions. Here, you can write raw numerical data, and choose the byte order for multibyte data types.

Create a new console application in the Introjucer application and name it `Chapter04_02`. The following example writes `int`, `float`, and `double` data types to a file, and then reads this data back in, posting the result to the log. Replace the contents of `Main.cpp` file with the following code:

[PRE3]

The `OutputStream` and `InputStream` classes and their respective subclasses, support writing and reading the various built-in types using functions `writeInt()`, `writeFloat()`, `readInt()`, `readFloat()`, and so on. These versions of the functions write these multi-byte types using little endian byte order. For file formats requiring big endian storage, there are equivalent functions `writeIntBigEndian()`, `writeFloatBigEndian()`, `readIntBigEndian` `()`, `readFloatBigEndian()` , and so on.

The JUCE stream classes are useful but quite low level. For many purposes, JUCE already includes high-level classes for reading and writing specific file types. Of course, these are built on top of the stream classes, but, unless you are dealing with a custom data format, it is likely to be more sensible to use built-in functionality for handling things such as images, audio, and other formats such as **Extensible Markup Language** (**XML**) and **JavaScript Object Notation** (**JSON**).

# Reading and writing image files

JUCE includes built-in support for reading and writing GIF, PNG, and JPEG image files. JUCE also includes its own `Image` class for holding bitmap images. The following example illustrates how to present a native file browser to choose an image file, load the image file, and display it in an `ImageComponent` object. Create a new GUI project in the Introjucer application with a basic window named `Chapter04_03`. and make the window resizable in the `Main.cpp` file, as we did in earlier chapters. You should then change the `MainComponent.h` file to contain:

[PRE4]

Change `MainComponent.cpp` to contain:

[PRE5]

Here, we create a `FileChooser` object in response to the user clicking on the **Read Image File…** button. This presents a native dialog window that allows the user to choose a file. We use the `ImageFileFormat::loadFrom()` function to attempt to load the file as an image. Because we didn't limit the file types displayed or enabled in the file chooser, the user may not have chosen a valid image file. We check the validity of the image, and if it is valid we pass the loaded image to the `ImageComponent` object for display. The `ImageComponent` class has various options to control the way the image is positioned and scaled, depending on how the original image size and component rectangle compare. These can be controlled using the `ImageComponent::setImagePlacement()` function. The following screenshot shows the application that reads an image file:

![Reading and writing image files](img/3316_04_01.jpg)

The `Image` class is similar to the `String` class, in that it uses a reference-counted object internally such that several `Image` objects may share the same internal data.

## Manipulating image data

In the next example we will add a slider to control the brightness of the displayed image and a button to write this processed image as a PNG file. Change the contents of the `MainComponent.h` file, where the changes are highlighted in the following code listing:

[PRE6]

Now replace the `MainComponent.cpp` file with the include directive and the constructor:

[PRE7]

Add the `resized()` function that positions the components:

[PRE8]

Add the `buttonClicked()` function that responds to the button interactions:

[PRE9]

Finally, add the `sliderValueChanged()` function that responds to the slider interaction:

[PRE10]

Here, we keep a copy of the original image and a processed version. Each time the slider changes, the image is updated with the new brightness by iterating over each of the pixels. When the **Write Image File…** button is clicked, we create a `FileChooser` object and present this to the user with the `FileChooser::browseForFileToSave()` function, rather than the `FileChooser::browseForFileToOpen()` function as we did for reading the file. Then the `PNGImageFormat` class is used to write the processed image to the selected file as a file stream. The image processing here could be significantly optimized, but that is beyond the scope of this book.

# Playing audio files

JUCE provides a sophisticated set of classes for dealing with audio. This includes: sound file reading and writing utilities, interfacing with the native audio hardware, audio data conversion functions, and a cross-platform framework for creating audio plugins for a range of well-known host applications. Covering all of these aspects is beyond the scope of this book, but the examples in this section will outline the principles of playing sound files and communicating with the audio hardware. In addition to showing the audio features of JUCE, in this section we will also create the GUI and autogenerate some other aspects of the code using the Introjucer application.

## Creating a GUI to control audio file play

Create a new GUI application Introjucer project named `Chapter04_04`, selecting the option to create a basic window. In the Introjucer application, select the **Config** panel, and select **Modules** in the hierarchy.

For this project we need the `juce_audio_utils` module (which contains a special `Component` class for configuring the audio device hardware); therefore, turn `ON` this module. Even though we created a basic window and a basic component, we are going to create the GUI using the Introjucer application in a similar way to that at the end of [Chapter 2](ch02.html "Chapter 2. Building User Interfaces"), *Building User Interfaces*.

Navigate to the **Files** panel and right-click (on the Mac, press *control* and click) on the **Source** folder in the hierarchy, and select **Add New GUI Component…** from the contextual menu.

When asked, name the header `MediaPlayer.h` and click on **Save**. In the **Files** hierarchy, select the `MediaPlayer.cpp` file. First select the **Class** panel and change the **Class name** from `NewComponent` to `MediaPlayer`. We will need four buttons for this basic project: a button to open an audio file, a **Play** button, a **Stop** button, and an audio device settings button. Select the **Subcomponents** panel, and add four `TextButton` components to the editor by right-clicking to access the contextual menu. Space the buttons equally near the top of the editor, and configure each button as outlined in the table as follows:

| Purpose | member name | name | text | background (normal) |
| --- | --- | --- | --- | --- |
| Open file | `openButton` | `open` | `Open…` | Default |
| Play/pause file | `playButton` | `play` | `Play` | Green |
| Stop playback | `stopButton` | `stop` | `Stop` | Red |
| Configure audio | `settingsButton` | `settings` | `Audio Settings…` | Default |

Arrange the buttons as shown in the following screenshot:

![Creating a GUI to control audio file play](img/3316_04_02.jpg)

For each button, access the **mode** pop-up menu for the **width** setting, and choose **Subtracted from width of parent**. This will keep the right-hand side of the buttons the same distance from the right-hand side of the window if the window is resized. There are more customizations to be done in the Introjucer project, but for now, make sure that you have saved the `MediaPlayer.h` file, the `MediaPlayer.cpp` file, and the Introjucer project before you open your native IDE project.

### Tip

Make sure that you have saved all of these files in the Introjucer application; otherwise the files may not get correctly updated in the file system when the project is opened in the IDE.

In the IDE we need to replace the `MainContentComponent` class code to place a `MediaPlayer` object within it. Change the `MainComponent.h` file as follows:

[PRE11]

Then, change the `MainComponent.cpp` file to:

[PRE12]

Finally, make the window resizable in the `Main.cpp` file (as we did in the *Adding child components* section of [Chapter 2](ch02.html "Chapter 2. Building User Interfaces"), *Building User Interfaces*), and build and run the project to check that the window appears as expected.

## Adding audio file playback support

Quit the application and return to the `Introjucer` project. Select the `MediaPlayer.cpp` file in the **Files** panel hierarchy and select its **Class** panel. The **Parent classes** setting already contains `public Component`. We are going to be listening for state changes from two of our member objects that are `ChangeBroadcaster` objects. To do this, we need our `MediaPlayer` class to inherit from the `ChangeListener` class. Change the **Parent classes** setting such that it reads:

[PRE13]

Save the `MediaPlayer.h` file, the `MediaPlayer.cpp` file, and the `Introjucer` project again, and open it into your IDE. Notice in the `MediaPlayer.h` file that the parent classes have been updated to reflect this change. For convenience, we are going to add some enumerated constants to reflect the current playback state of our `MediaPlayer` object, and a function to centralize the change of this state (which will, in turn, update the state of various objects, such as the text displayed on the buttons). The `ChangeListener` class also has one pure virtual function, which we need to add. Add the following code to the `[UserMethods]` section of `MediaPlayer.h`:

[PRE14]

We also need some additional member variables to support our audio playback. Add these to the `[UserVariables]` section:

[PRE15]

The `AudioDeviceManager` object will manage our interface between the application and the audio hardware. The `AudioFormatManager` object will assist in creating an object that will read and decode the audio data from an audio file. This object will be stored in the `ScopedPointer<AudioFormatReaderSource>` object. The `AudioTransportSource` object will control the playback of the audio file and perform any sampling rate conversion that may be required (if the sampling rate of the audio file differs from the audio hardware sampling rate). The `AudioSourcePlayer` object will stream audio from the `AudioTransportSource` object to the `AudioDeviceManager` object. The `state` variable will store one of our enumerated constants to reflect the current playback state of our `MediaPlayer` object.

Now add some code to the `MediaPlayer.cpp` file. In the `[Constructor]` section of the constructor, add following two lines:

[PRE16]

This sets the **Play** and **Stop** buttons to be disabled (and grayed out) initially. Later, we enable the **Play** button once a valid file is loaded, and change the state of each button and the text displayed on the buttons, depending on whether the file is currently playing or not. In this `[Constructor]` section you should also initialize the `AudioFormatManager` as follows:

[PRE17]

This allows the `AudioFormatManager` object to detect different audio file formats and create appropriate file reader objects. We also need to connect the `AudioSourcePlayer`, `AudioTransportSource` and `AudioDeviceManager` objects together, and initialize the `AudioDeviceManager` object. To do this, add the following lines to the `[Constructor]` section:

[PRE18]

The first line connects the `AudioTransportSource` object to the `AudioSourcePlayer` object. The second line connects the `AudioSourcePlayer` object to the `AudioDeviceManager` object. The final line initializes the `AudioDeviceManager` object with:

*   The number of required audio input channels (`0` in this case).
*   The number of required audio output channels (`2` in this case, for stereo output).
*   An optional "saved state" for the `AudioDeviceManager` object (`nullptr` initializes from scratch).
*   Whether to open the default device if the saved state fails to open. As we are not using a saved state, this argument is irrelevant, but it is useful to set this to `true` in any case.

The final three lines to add to the `[Constructor]` section to configure our `MediaPlayer` object as a listener to the `AudioDeviceManager` and `AudioTransportSource` objects, and sets the current state to `Stopped`:

[PRE19]

In the `buttonClicked()` function we need to add some code to the various sections. In the `[UserButtonCode_openButton]` section, add:

[PRE20]

When the `openButton` button is clicked, this will create a `FileChooser` object that allows the user to select a file using the native interface for the platform. The types of files that are allowed to be selected are limited using the wildcard `*.wav` to allow only files with the `.wav` file extension to be selected.

If the user actually selects a file (rather than cancels the operation), the code can call the `FileChooser::getResult()` function to retrieve a reference to the file that was selected. This file is then passed to the `AudioFormatManager` object to create a file reader object, which in turn is passed to create an `AudioFormatReaderSource` object that will manage and own this file reader object. Finally, the `AudioFormatReaderSource` object is connected to the `AudioTransportSource` object and the **Play** button is enabled.

The handlers for the `playButton` and `stopButton` objects will make a call to our `changeState()` function depending on the current transport state. We will define the `changeState()` function in a moment where its purpose should become clear.

In the `[UserButtonCode_playButton]` section, add the following code:

[PRE21]

This changes the state to `Starting` if the current state is either `Stopped` or `Paused`, and changes the state to `Pausing` if the current state is `Playing`. This is in order to have a button with combined play and pause functionality.

In the `[UserButtonCode_stopButton]` section, add the following code:

[PRE22]

This sets the state to `Stopped` if the current state is `Paused`, and sets it to `Stopping` in other cases. Again, we will add the `changeState()` function in a moment, where these state changes update various objects.

In the `[UserButtonCode_settingsButton]` section add the following code:

[PRE23]

This presents a useful interface to configure the audio device settings.

We need to add the `changeListenerCallback()` function to respond to changes in the `AudioDeviceManager` and `AudioTransportSource` objects. Add the following to the `[MiscUserCode]` section of the `MediaPlayer.cpp` file:

[PRE24]

If our `MediaPlayer` object receives a message that the `AudioDeviceManager` object changed in some way, we need to check that this change wasn't to disable all of the audio output channels, by obtaining the setup information from the device manager. If the number of output channels is zero, we disconnect our `AudioSourcePlayer` object from the `AudioTransportSource` object (otherwise our application may crash) by setting the source to `nullptr`. If the number of output channels becomes nonzero again, we reconnect these objects.

If our `AudioTransportSource` object has changed, this is likely to be a change in its playback state. It is important to note the difference between requesting the transport to start or stop, and this change actually taking place. This is why we created the enumerated constants for all the other states (including transitional states). Again we issue calls to the `changeState()` function depending on the current value of our `state` variable and the state of the `AudioTransportSource` object.

Finally, add the important `changeState()` function to the `[MiscUserCode]` section of the `MediaPlayer.cpp` file that handles all of these state changes:

[PRE25]

After checking that the `newState` value is different from the current value of the `state` variable, we update the `state` variable with the new value. Then, we perform the appropriate actions for this particular point in the cycle of state changes. These are summarized as follows:

*   In the `Stopped` state, the buttons are configured with the **Play** and **Stop** labels, the **Stop** button is disabled, and the transport is positioned to the start of the audio file.
*   In the `Starting` state, the `AudioTransportSource` object is told to start. Once the `AudioTransportSource` object has actually started playing, the system will be in the `Playing` state. Here we update the `playButton` button to display the text **Pause**, ensure the `stopButton` button displays the text **Stop**, and we enable the **Stop** button.
*   If the **Pause** button is clicked, the state becomes `Pausing`, and the transport is told to stop. Once the transport has actually stopped, the state changes to `Paused`, the `playButton` button is updated to display the text **Resume** and the `stopButton` button is updated to display **Return to Zero**.
*   If the **Stop** button is clicked, the state is changed to `Stopping`, and the transport is told to stop. Once the transport has actually stopped, the state changes to `Stopped` (as described in the first point).
*   If the **Return to Zero** button is clicked, the state is changed directly to `Stopped` (again, as previously described).
*   When the audio file reaches the end of the file, the state is also changed to `Stopped`.

Build and run the application. You should be able to select a `.wav` audio file after clicking the **Open...** button, play, pause, resume, and stop the audio file using the respective buttons, and configure the audio device using the **Audio Settings…** button. The audio settings window allows you to select the input and output device, the sample rate, and the hardware buffer size. It also provides a **Test** button that plays a tone through the selected output device.

# Working with the Binary Builder tool

One problem with writing cross-platform applications is the packaging of binary files for use within the application. JUCE includes the **Binary Builder** tool that transforms binary files into source code, which is then compiled into the application's code. This ensures that the files will behave identically on all platforms, rather than relying on peculiarities of the runtime machine. Although the Binary Builder is available as a separate project (in `juce/extras/binarybuilder`), its functionality is available within the Introjucer application's GUI component editor.

## Embedding an image file using the Introjucer application

Create a new Introjucer project named `Chapter04_05` with a basic window. Add a new GUI component as before; this time name it `EmbeddedImage` (remembering to also change the name in its **Class** panel). In its **Subcomponents** panel, right-click in the canvas and choose **New Generic Component** and resize it to fill the canvas with a small border around the edge. Change the **member name** and **name** to `image`, and change the **class** to `ImageComponent`. In the **Resources** panel, choose **Add new resource…** and select an image file to add. This will create a resource that is the binary file converted to code. It will be given a variable name within this component based on the original filename, and will be stored as a static variable. For example, a file named `sample.png` will be named `sample_png`. In addition to this a static variable storing this resource's size as an integer will be created and will have `Size` appended to this name, for example, `sample_pngSize`. Save the project and open it into your IDE. Update the `MainComponent` file's contents as before. Change the `MainComponent.h` file as follows:

[PRE26]

Then change the `MainComponent.cpp` file to:

[PRE27]

Finally in the `EmbeddedImage.cpp` file notice the large arrays of numbers at the end of the file, this is the image file converted to code. In the `[Constructor]` section, add the following two lines (although you may need to use different names from `sample_png`, `sample_pngSize`, depending on the file resource you added previously):

[PRE28]

This creates a memory stream from our resource, providing the data pointer and the data size (the final `false` argument tells the memory stream not to copy the data). Then we load the image as before using the `ImageFileFormat` class. Build and run the application, and the image should be displayed into the application's window.

# Summary

This chapter has covered a range of techniques for dealing with files in JUCE, focusing in particular on image and audio files. You are encouraged to explore the online JUCE documentation, which provides even more detail on many of the possibilities introduced here. We have also introduced the Binary Builder tool that provides a means of transforming media files into source code that is suitable for cross-platform use. You are encouraged to read the online JUCE documentation for each of the classes introduced in this chapter. This chapter has given only an introduction to get you started; there are many other options and alternative approaches, which may suit different circumstances. The JUCE documentation will take you through each of these and point you to related classes and functions. The next chapter covers some useful utilities available within JUCE for creating cross-platform applications.