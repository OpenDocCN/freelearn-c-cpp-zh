# Sounds and Visions - Qt Multimedia

Applications that need to play sounds or show videos are usually games, while others are full-blown multimedia apps. Qt Multimedia can handle both.

Qt Multimedia can be used with both Qt Widgets and Qt Quick, or even without a GUI interface. It has both C++ and QML APIs, but the QML API has a few special treats and tricks. A little-known fact is that Qt can also play 3D positional audio in Qt Quick. You can control the gain and pitch with three dimensions.

We will cover the following topics in this chapter:

*   Sonic vibrations – audio
*   Image sensors – camera
*   Visual media – playing video
*   Tuning it in – FM radio tuner

# Sonic vibrations – audio

I have a relationship with audio that goes way back—before computers were household things, when Mylar tape and magnets ruled the sonic realms. Things have progressed since then. Now, mobile phones fit into our pockets and light bulbs can play music.

3D audio in Qt is supported through the OpenAL API. If you are using Linux, the default Qt binaries from Qt Company do not ship with the needed Qt Audio Engine API. You will have to install the OpenAL development package and then compile Qt Multimedia for yourself. OpenAL is not supported on Android, so no joy there. Luckily, it is supported by default on Apple Mac and iOS. So, that is where I am going to develop this next section. Let's grab the nearest MacBook and head over there.

3D audio is audio in three dimensions, just like 3D graphics—not just left and right, but also up, down, front, and back placement of audio. The term *positional audio* might explain this better. 

With Qt, 3D audio is only supported using Qt Quick. 

The source code for this chapter can be found in the Git repository under the `Chapter09-3dAudio` directory, in the `cp9` branch.

To use Qt Multimedia, you need to edit the project `.pro` file and add the following line:

```cpp
QT += multimedia
```

Edit the `qml` file that you want to use the 3D audio in, and add the `import` line:

```cpp
import QtAudioEngine 1.0
```

3D space is made up of three axes named x, y, and z, which correspond to horizontal/vertical and up/down in 3-dimensional space.

`AudioEngine` and other associated classes use the `Qt.vector3d` value type. It is essential to understand this element to use 3D audio.

# Qt.vector3d

`Qt.vector3d` is an array of values that represents the x, y, and z axes—x being horizontal, y being vertical, and z being up or down. Each value is a single-precision `qreal`.

It can be used like `Qt.vector3d(15, -5, 0)` or `"15, -5, 0"` as a `String`. 

The positioning of the audio is controlled through the use of the `vector3d` property value.

`Qt.vector3d` is used to position the audio in 3 dimensional space.

The main component for using 3D audio in QML is called `AudioEngine`. The other components we will use can be children of this component.

# AudioEngine

`AudioEngine` is the central container for the other 3D audio items that you will use.

We can set up the component easily enough:

```cpp
    AudioEngine {
        id: audioEngine
        dopplerFactor: 1
        speedOfSound: 343.33
}
```

The `dopplerFactor` property creates a Doppler shift effect. The `speedOfSound` value reflects the speed of sound in which the Doppler effect is calculated.

You assign a `listener` through the `listener` property. We will get to that later in the *AudioListener* section.

We have an audio sample we want to load and use, so we declare at least one `AudioSample`.

# AudioSample

`AudioSample` can be defined as a child of an `AudioEngine` component:

```cpp
 AudioEngine {
        id: audioEngine
        dopplerFactor: 1
        speedOfSound: 343.33
        AudioSample {
            name:"plink"
            source: "thunder.wav"
            preloaded: true
        }
}
```

It can also be added using the `AudioEngine.addAudioSample()` method:

```cpp
 AudioEngine {
        id: audioEngine
        dopplerFactor: 1
        speedOfSound: 343.33
        addAudioSample(plinkSound)
}
AudioSample {
    id: plinkSound
    name:"plink"
    source: "thunder.wav"
    preloaded: true
}
```

The `source` property holds the sample's filename and a name to refer to it with.

Now, we are ready to play the sound using the `Sound` component.

# Sound

The `Sound` element is a container for one or more samples that will play with different parameters and variances. In other words, you can define a `PlayVariation` item, which defines how a `Sound` plays an `AudioSample`, with maximum or minimum values in pitch and gain. You can also declare the sample to be `looping`, which means it plays over and over:

```cpp
Sound {
    name: "thunderengine"
    attenuationModel: "thunderModel"
    PlayVariation {
        looping: true
        sample: "plink"
        maxGain: 0.5
        minGain: 0.3
     }
}
```

The `attenuationModel` property controls the way the sound volume level falls off, or fades over time. It can be one of these values:

*   Linear is a straight falloff
*   Inverse is a more natural, non-linear curve 

You can control this using the `start`, `end`, and `rolloff` properties.

# AudioListener

The `AudioListener` component represents the `listener` and its position in the 3D realm. There is only one `listener`. It can either be constructed as the `listener` property of the `AudioEngine` component, or as a definable element:

```cpp
    AudioListener {
        engine: audioEngine
        position: Qt.vector3d(0, 0, 0)
    }
```

A `SoundInstance` is the component that a `Sound` uses to play the sample.

# SoundInstance

`SoundInstance` has a few properties that you can use to adjust the sound:

*   `direction`
*   `gain`
*   `pitch`
*   `position`

These properties take a `vector3d` value.

The `sound` property of the `SoundInstance` element takes a string that represents the name of a `Sound` component:

```cpp
    SoundInstance {
        id: plinkSound
        engine: audioEngine
        sound: "thunderengine"
        position: Qt.vector3d(leftRightValue, forwardBacktValue,
upDownValue)
        Component.onCompleted: plinkSound.play()
    }

```

Here, I start playing the sound when the component is completed.

Now, we just need some mechanism to move the sound position around. We can use the `Accelerometer` values if we have an accelerometer on the device. I'm just going to use the mouse. Remember that on a touchscreen, a `MouseArea` also includes touch input.

We must enable `hover` in order to track the mouse without clicking:

```cpp
    MouseArea {
        anchors.fill: parent
        hoverEnabled: true
        propagateComposedEvents: true
        onPositionChanged: {
            leftRightValue = -((window.width / 2) - mouse.x)
            forwardBacktValue = (window.height / 2) - mouse.y
        }
```

To propagate the mouse clicks to buttons or other items when using `MouseArea`, put the `MouseArea` at the top of the file, as Qt Quick will set the z order in the order of the components from the top of the file, down to the bottom. You could also set the `z` property of the buttons and set the `z` property of the `MouseArea` to the lowest value.

I previously declared three values in my `Window` component to use in the positioning of the audio:

```cpp
property real leftRightValue: 0;
property real forwardBacktValue: 0;
property real upDownValue: 0;
```

Now, when you move the mouse around, the audio will appear to move around.

But there is no mouse on a phone. There is a touch point, but no scrolling. I could use an `Accelerometer` as it has the z axis, or use `PinchArea` to control the up and down position.

Let's look at a few other ways to deal with audio.

# Audio

The `Audio` element is probably the easiest way to play audio. It only takes a few lines. It would be good for playing sound effects.

The source code can be found in this book's Git repository under the `Chapter09-1` directory, in the `cp9` branch.

We will use the following `import` statement:

```cpp
import QtMultimedia 5.12
```

Here's a simple stanza that will play the `.mp3` file named `sample.mp3`:

```cpp
Audio {
    id: audioPlayer
    source: "sample.mp3"
}
```

The `source` property is where you declare which file to play. Now, you just have to call the `play()` method to have this `sample.wav` file play:

```cpp
Component.onCompleted: audioPlayer.play()
```

You can also set the `autoPlay` property to `true` instead of calling `play`, and that would play the file once the component is completed.

Setting the volume is as easy as declaring the `volume` property and setting a decimal value between 0 and 1—1 being full volume and 0 being muted:

```cpp
volume: .75
```

Getting the metadata or ID tags from the file is not obvious, as they only become available after the `metaDataChanged` signal gets emitted. This is only emitted by the `Audio` element's `metaData` object.

Sometimes, you might need to display a file's metadata, or the extra data that can be within the audio file's headers. The `Audio` component has a `metaData` property that can be used like this:

```cpp
metaData {
    onMetaDataChanged: {
        titleLabel.text = "Title: " + metaData.title
        artistLabel.text = "Artist: " + metaData.contributingArtist
        albumLabel.text = "Album: " + metaData.albumTitle
    }
}
```

If you need to access the microphone and record audio, you will need to dive into C++, so let's take a look at `QAudioRecorder`.

# QAudioRecorder

Recording audio is one of my passions. Recording audio, or more specifically using the microphone, may require user permissions on some platforms.

The recording of audio, called taping back in my day, can be implemented by using the `QAudioRecorder` class. Recording properties are controlled by the `QAudioEncoderSettings` class, from which you can control the codec that's used, the channel count, the bit rate, and the sample rate. You can either explicitly set the bit rate and sample rate, or use the more generic `setQuality` function.

The source code can be found in this book's Git repository under the `Chapter09-2` directory, in the `cp9` branch.

You might want to query the input devices and see which settings are available. To do that, you would query using `QAudioDeviceInfo`, iterating through `QAudioDeviceInfo::availableDevices(QAudio::AudioInput)`:

```cpp

void MainWindow::listAudioDevices() 
{ 
    for (const QAudioDeviceInfo &deviceInfo : 
         QAudioDeviceInfo::availableDevices(QAudio::AudioInput)) { 
        ui->textEdit->insertPlainText( 
                    QString("Device name: %1\n") 
                    .arg(deviceInfo.deviceName())); 

        ui->textEdit->insertPlainText( 
                    "    Supported Codecs: " 
                    + deviceInfo.supportedCodecs() 
                    .join(", ") + "\n"); 
        ui->textEdit->insertPlainText( 
                    QString("    Supported channel count: %1\n") 
                    .arg(stringifyIntList(deviceInfo.supportedChannelCounts()))); 
        ui->textEdit->insertPlainText( 
                    QString("    Supported bit depth b/s: %1\n") 
                    .arg(stringifyIntList(deviceInfo.supportedSampleSizes()))); 
        ui->textEdit->insertPlainText( 
                    QString("    Supported sample rates Hz: %1\n") 
                    .arg(stringifyIntList(deviceInfo.supportedSampleRates()))); 
    }    
} 

```

Qt Multimedia uses the term sample sizes for the more common term bit depth.

As you can see from my laptop, I have a few different audio input devices. The laptop's built-in audio chip got fried from an electrical spike, which is why it isn't seen here:

![](img/a5acaf2c-000a-4c42-9ccb-ada79d0bf846.png)

For iPhone, it is different. It has only one audio device, named `default`:

![](img/4ef5a2e6-b9df-4ad1-ab1a-60cc43d0dd76.png)

My Linux desktop reports a lot of audio input devices because of the ALSA Driver, which I won't include here.

We need to set up the recording encoder settings with the type of audio file we want to record. This includes the number of channels, the code, sample rate, and bit rate:

```cpp
QAudioEncoderSettings audioSettings;
audioSettings.setCodec("audio/pcm");
audioSettings.setChannelCount(2);
audioSettings.setBitRate(16);
audioSettings.setSampleRate(44100);
```

If you want to let the system decide on the various settings, it is quicker and takes less code to use the `setQuality` function, which can take one of the following values:

*   `QMultimedia::VeryLowQuality`
*   `QMultimedia::LowQuality`
*   `QMultimedia::NormalQuality`
*   `QMultimedia::HighQuality`
*   `QMultimedia::VeryHighQuality`

Let's choose `NormalQuality`, which will give the same results:

```cpp
audioSettings.setQuality(QMultimedia::NormalQuality);
```

The `QAudioRecorder` class is used to record the audio, so let's construct a `QAudioRecorder` and set the encoding settings:

```cpp
QAudioRecorder *audioRecorder = new QAudioRecorder(this);
audioRecorder->setEncodingSettings(audioSettings);
```

You can also specify which audio input to use, but first you will need to get a list of available audio input:

```cpp
QStringList inputs = audioRecorder->audioInputs();
```

If you don't want to bother about which audio device to use, you can specify it using the default device with the `defaultAudioInput()` function:

```cpp
   audioRecorder->setAudioInput(audioRecorder->defaultAudioInput());
```

We can save it to a file, or even a network location, as the `setOutputLocation` function takes a `QUrl`. We will just specify a local file to save it to:

```cpp
audioRecorder->setOutputLocation(QUrl::fromLocalFile("record1.wav"));
```

If the file is relative, like it is here, you can get the actual output location using `outputLocation()` once the recording has started.

Finally, we can start the recording process:

```cpp
audioRecorder->record();
```

There are also the `stop()` and `pause()` methods to control the recording operation.

Of course, you will want to connect to the error signal, because errors can and will happen from time to time. Again, note the use of the `QOverload` syntax that's used in error-reporting signals:

```cpp
connect(audioRecorder, QOverload<QMediaRecorder::Error>::of(&QMediaRecorder::error),
           [=](QMediaRecorder::Error error){ 
                ui->textEdit->insertPlainText("QAudioRecorder Error: " + audioRecorder->errorString()); 
               on_stopButton_clicked(); 
            }); 
```

So, now that we have recorded some audio, we might want to listen to it. This is where `QMediaPlayer` comes in.

# QMediaPlayer

`QMediaPlayer` is fairly straightforward. It can play both audio and video, but here we will only be playing audio. First, we need to set up the media to play by calling `setMedia`. 

We can use `QAudioRecorder` to get the output file and use it to play:

```cpp
player = new QMediaPlayer(this);
player->setMedia(audioRecorder->outputLocation());
```

We will have to monitor the current playing position, so we will connect the `positionChanged` signal to a progress bar:

```cpp
connect(player, &QMediaPlayer::positionChanged,
         this, &MainWindow::positionChanged);
```

Connect the error signal and its `QOverload` syntax:

```cpp
connect(player, QOverload<QMediaPlayer::Error>::of(&QMediaPlayer::error),
            [=](QMediaPlayer::Error error){ 
            ui->textEdit->insertPlainText("QMediaPlayer Error: " + player->errorString());
           on_stopButton_clicked();
   });
```

Then, just call `play()` on the `QMediaPlayer` object:

```cpp
player->play();
```

You can even set the playback volume:

```cpp
player->setVolume(75);
```

If you need access to the media data, let's say for getting the volume level of the data as it plays, you will want to use something other than `QMediaPlayer` to play your file.

# QAudioOutput

`QAudioOutput` provides a way to send audio to an audio output device:

```cpp
QAudioOutput *audio;
```

Using `QAudioOutput`, you will need to set up the exact format of your file. To get the format of your file, you could use `QMediaResource`.

Scratch that—`QMediaResource` is being depreciated in Qt 6.0, and does not do what the docs say it is supposed to do, and doesn't work like it should. We need to hardcode the data format, so we will use the basic good-quality stereo format. `QAudioFormat` is the way to do this:

```cpp
    QAudioFormat format;
    format.setSampleRate(44100);
    format.setChannelCount(2);
    format.setSampleSize(16);
    format.setCodec("audio/pcm");
    format.setByteOrder(QAudioFormat::LittleEndian);
    format.setSampleType(QAudioFormat::UnSignedInt);
```

We will iterate through the audio devices and check that `QAudioDeviceInfo` supports this format:

```cpp
    for (const QAudioDeviceInfo &deviceInfo : QAudioDeviceInfo::availableDevices(QAudio::AudioOutput)) {
        if (deviceInfo.isFormatSupported(format)) {
            audio = new QAudioOutput(deviceInfo, format, this);
            connect(audio, &QAudioOutput::stateChanged, [=] (QAudio::State
state) {
            qDebug() << Q_FUNC_INFO << "state" << state;
            if (state == QAudio::StoppedState) {
                if (audio->error() != QAudio::NoError) {
                    qDebug() << Q_FUNC_INFO << audio->error();
                }
            }
        });
 }
```

Here, I connected to the `stateChanged` signal and tested whether the state is `StoppedState`; we know there might be an error, so we check the `error()` of the `QAudioOutput` object. Otherwise, we can play the file:

```cpp
QFile sourceFile;
sourceFile.setFileName(file);
sourceFile.open(QIODevice::ReadOnly);
audio->start(&sourceFile);
```

We see now that Qt Multimedia has various ways of playing audio. Now, let's take a look at the camera and recording video.

# Image sensors – camera

First, we should establish whether the device has any cameras. This helps us determine specifics about the use of the camera and other camera specifications, such as the orientation or position on the device.

For this, we will use `QCameraInfo`.

# QCameraInfo

We can get a list of cameras using the `QCameraInfo::availableCameras()` function:

The source code can be found in this book's Git repository under the `Chapter09-4` directory, in the `cp9` branch.

```cpp
    QList<QCameraInfo> cameras = QCameraInfo::availableCameras();
    foreach (const QCameraInfo &cameraInfo, cameras)
        ui->textEdit->insertPlainText(cameraInfo.deviceName() + "\n");
```

On my Android device, I see two cameras, named `back` and `front`. You can also check for `front` and `back` cameras using `QCameraInfo::position()`, which will return one of the following:

*   `QCamera::UnspecifiedPosition`
*   `QCamera::BackFace`
*   `QCamera::FrontFace`

`FrontFace` means that the camera lens is on the same side as the screen. You can then use `QCameraInfo` to construct a `QCamera` object:

```cpp
QCamera *camera;
if (cameraInfo.position() == QCamera::BackFace) {
    camera = new QCamera(cameraInfo);
}
```

Now, check for the capture modes the camera supports, which can be one of the following:

*   `QCamera::CaptureViewfinder`
*   `QCamera::CaptureStillImage`
*   `QCamera::CaptureVideo`

Let's do a quick still image shot first. We need to tell the camera to use the `QCamera::CaptureStillImage` mode:

```cpp
camera->setCaptureMode(QCamera::CaptureStillImage);
```

The `statusChanged` signal is used to monitor the status, which can be one of the following values:

*   `QCamera::UnavailableStatus`
*   `QCamera::UnloadedStatus`
*   `QCamera::UnloadingStatus`
*   `QCamera::LoadingStatus`
*   `QCamera::LoadedStatus`
*   `QCamera::StandbyStatus`
*   `QCamera::StartingStatus`
*   `QCamera::StoppingStatus`
*   `QCamera::ActiveStatus`

Let's connect to the `statusChanged` signal so that we can see status changes:

```cpp
connect(camera, &QCamera::statusChanged, [=] (QCamera::Status status) {
    ui->textEdit->insertPlainText(QString("Status changed %1").arg(status) + "\n");
});
```

If you need to fiddle with any of the camera settings, you will have to `load()` it before you can get access to the `QCameraImageProcessing` object:

```cpp
camera->load();
QCameraImageProcessing *imageProcessor = camera->imageProcessing();

```

With the `QCameraImageProcessing` class, you can set configurations, such as brightness, contrast, saturation, and sharpening.

Before we call start on the camera, we need to set up a `QMediaRecorder` object for the camera. Since `QCamera` is inherited from `QMediaObject`, we can feed it to the `QMediaRecorder` object.

Qt Multimedia Widgets are not supported on Android.

I tried `QCamera` version 5.12 on both Mac and iOS, but it kept crashing when I tried to `start()` the camera. I was successful on Linux desktop. On Android, since multimedia widgets are not supported, the camera viewfinder widget did not work, but I could still capture images from the image sensor.

Maybe you'll have better luck with the QML side of things. QML APIs are usually optimized for easy use.

# Camera

Yes, the QML `Camera` is so much easier to implement. Really, there are only two components you need to take a photo: `Camera` and `VideoOutput`. 

`VideoOutput` is the element to use for the viewfinder. It is also used when you are recording video:

```cpp
    Camera {
        id: camera
        position: Camera.BackFace
        onCameraStateChanged: console.log(cameraState)
        imageCapture {
            onImageCaptured: {
                console.log("Image captured")
            }
        }
    }
```

The `position` property controls which camera to use, especially on a mobile device that may have a front-facing and rear-facing camera. Here, I am not only using the rear camera. You would use the `FrontFace` position to take a selfie.

`imageCaptured` pertains to the `CameraCapture` sub-element. We can handle the `onImageCaptured` signal to preview the image or to alert the user that a photo has been taken.

The other properties of the `Camera` object can be controlled by their corresponding components:

*   `focus : CameraFocus`
*   `flash : CameraFlash`
*   `exposure : CameraExposure`
*   `imageProcessing : CameraImageProcessing`
*   `imageCapture : CameraCapture`
*   `videoRecorder: CameraRecorder`

`CameraRecorder` is what you would use to controls saturation, brightness, color filters, contrast, and other settings.

`CameraExposure` controls things such as aperture, exposure compensation, and shutter speed.

`CameraFlash` can turn the flash on, off, or use auto mode. It can also set red-eye compensation and video (constant) mode.

We need a view finder to see what the heck we are trying to capture, so let's take a look at the `VideoOutput` element. 

# VideoOutput

`VideoOutput` is the component we use to view what the camera is sensing.

The source code can be found on the Git repository under the `Chapter09-5` directory, in the `cp9` branch.

To implement the `VideoOutput` component, you need to define the `source` property. Here, we are using the camera:

```cpp
    VideoOutput {
        id: viewfinder
        source: camera
        autoOrientation: true
}
```

The `autoOrientation` property is used to allow the `VideoOutput` component to compensate for the device orientation of the image sensor. Without this being true, the image might show up in the view finder with the wrong orientation and confuse the user, making it harder to take a good photo or video.

Let's make this `VideoOutput` clickable by adding a `MouseArea`, where I will use the `onClicked` and `onPressAndHold` signals to focus and actually capture an image:

```cpp
MouseArea {
    anchors.fill: parent
    onPressAndHold: {
        captureMode: captureSwitch.position === 0 ?Camera.CaptureStillImage : Camera.CaptureVideo
        camera.imageCapture.capture()
    }
    onClicked: {
        if (camera.lockStatus == Camera.Unlocked)
            camera.unlock();
            camera.searchAndLock();
    }
 }
```

I also added a `Switch` component from Qt Quick Controls to control whether the user wants a still photo or video recorded.

To focus the camera, call the `searchAndLock()` method, which starts focus, white balance, and exposure computations.

Let's add support for recording videos. We will add a `CameraRecorder` container to the `Camera` component:

```cpp
VideoRecorder {
    audioEncodingMode: CameraRecorder.ConstantBitrateEncoding;
    audioBitRate: 128000
    mediaContainer: "mp4"
}
```

We can set certain aspects for the video, such as bit rate, frame rate, number of audio channels, and what container to use.

We need to also change the way our `onPressAndHold` signal works to make sure we record video when the user has specified it, by the use of the switch:

```cpp
onPressAndHold: {
    captureMode: captureSwitch.position === 0 ? Camera.CaptureStillImage : Camera.CaptureVideo
    if (captureSwitch.position === 0)
        camera.imageCapture.capture()
    else
        camera.videoRecorder.record()
}
```

We need some way to stop recording, so let's modify the `onClicked` signal handler to stop the recording when it is in `RecordingState`:

```cpp
onClicked: {
    if (camera.videoRecorder.recorderState === CameraRecorder.RecordingState) {
        camera.videoRecorder.stop()
     } else {
         if (camera.lockStatus == Camera.Unlocked)
             camera.unlock();
         camera.searchAndLock();
     }
}
```

Now, we need to actually see the video we just recorded. Let's move on and look at how to play a video.

# Visual media – playing video

Playing a video with QML is much like playing audio using `MediaPlayer`, only using a `VideoOutput` instead of an `AudioOutput` component.

The source code can be found on the Git repository under the `Chapter09-6` directory, in the `cp9` branch.

We begin by implementing a `MediaPlayer` component:

```cpp
MediaPlayer {
    id: player
```

The property named `autoPlay` will control the automatic starting of the video once the component is completed. 

Here, the `source` property is set to the filename of our video:

```cpp

    autoPlay: true
    source: "hellowindow.m4v"
    onStatusChanged: console.log("Status " + status)
    onError: console.log("Error: " + errorString)
}
```

We then create a `VideoOutput` component, with the source being our `MediaPlayer`:

```cpp

VideoOutput {
    source: player
    anchors.fill : parent
 }

MouseArea {
    id: playArea
    anchors.fill: parent
    onPressed: player.play();
}
```

The `MouseArea`, which is the entire application, is used here to start playing the video when you click anywhere on the application.

With C++, you would use the `QMediaPlayer` class with a `QGraphicsVideoItem`, `QVideoWidget`, or something else.

Since `QMultimediaWidgets` have limited support on mobile devices, I will leave this as an exercise for the reader.

Qt Multimedia also supports FM, AM, and some other radios, providing your device has a radio in it as well. 

# Tuning it in – FM radio tuner

Some Android phones have an FM radio receiver. Mine does! It requires the wired headphones to be inserted to work as the antenna. 

We start by implementing a `Radio` component:

```cpp
Radio {
    id: radio
```

The `Radio` element has a `band` property that you can use to configure the radio's frequency band use. They are one of the following:

*   `Radio.AM` : 520 - 1610 kHz
*   `Radio.FM` : 87.5 - 108 MHz, Japan 76 - 90 MHz
*   `Radio.SW` : 1.711 to 30 MHz
*   `Radio.LW` : 148.5 to 283.5 kHz
*   `Radio.FM2` : Range not defined

```cpp

    band: Radio.FM
    Component.onCompleted {
        if (radio.availability == Radio.Available)
            console.log("Good to go!")
        else 
           console.log("Sad face. No radio found. :(")
    }
}
```

The `availability` property can return the following different values:

*   `Radio.Available`
*   `Radio.Busy`
*   `Radio.Unavailable`
*   `Radio.ResourceMissing`

The first thing the user will do with a radio is scan for stations, which can be accomplished by using the `searchAllStations` method, which takes one of the following values:

*   `Radio.SearchFast`
*   `Radio.SearchGetStationId`: Like `SearchFast`, it emits the `stationFound` signal

The `stationsFound` signal returns an `int` `frequency` and `stationId` string for each station that's tuned in. You could collect these in a model-based component, such as `ListView`, using a `ListModel`. The `ListView` would use the `ListModel` as its model.

You can cancel the scan by calling the `cancelScan()` method. The `scanUp()` and `scanDown()` methods are similar to `searchAllStations`, except it does not remember the stations it found. The `tuneUp` and `tuneDown` methods will tune the frequency up or down one step, according to the `frequencyStep`.

Here are some other interesting properties:

*   `antennaConnected`: True if an antenna is connected
*   `signalStrength`: Strength of the signal in %
*   `frequency`: Holds and sets the frequency that the radio is tuned to

# Summary

In this chapter, we discussed the different aspects of the big API of Qt Multimedia. You should now be able to position sound in a 3-dimensional way for 3D games. We learned how to record and play audio and video, and control and use the camera to take a selfie. We also touched on using QML to listen to radio stations.

In the next chapter, we will dig into using `QSqlDatabase` to access databases.