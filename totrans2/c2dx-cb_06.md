# Chapter 6. Playing Sounds

A game without sound will be boring and lifeless. Background music and sound effects that suit the visuals will lighten up the game. Initially, we used a very famous audio engine called `SimpleAudioEngine`, but Cocos2d-x version 3.3 has now come up with an all-new `AudioEngine`. In this chapter, we're going to talk about both `SimpleAudioEngine` and `AudioEngine`. The following topics will be covered in this chapter:

*   Playing background music
*   Playing a sound effect
*   Controlling volume, pitch, and balance
*   Pausing and resuming background music
*   Pausing and resuming sound effects
*   Playing background music and a sound effect by using AudioEngine
*   Playing movies

# Playing background music

By using `SimpleAudioEngine`, we can play background music very easily. `SimpleAudioEngine` is a shared singleton object that can be called from anywhere in your code. In `SimpleAudioEngine`, we can play only one background score.

## Getting ready

We have to include the header file of `SimpleAudioEngine` to use it. Therefore, you will need to add the following code:

[PRE0]

## How to do it...

The following code is used to play background music called `background.mp3`.

[PRE1]

## How it works...

`SimpleAudioEngine` has a namespace called `CocosDenshion`. For `SimpleAudioEngine`, you just have to get an instance by using the `getInstance` method. You can play the background music without preloading it, but this could result in a delay in playback. That's why you should preload the music before playing it. If you want the playback to be continuous, you need to set the true value as the second argument.

## There's more...

`SimpleAudioEngine` supports a number of formats, including MP3 and Core Audio format. It can play the following formats:

| Format | iOS (BGM) | iOS (SE) | Android (BGM) | Android (SE) |
| --- | --- | --- | --- | --- |
| IMA (.caf) | ○ | ○ | □ | □ |
| Vorbis (.ogg) | □ | □ | ○ | ○ |
| MP3 (.mp3) | ○ | ○ | ○ | ○ |
| WAVE (.wav) | ○ | ○ | △ | △ |

If you want to play a sound in a different format on iOS and Android, you can play it by using the following macro code:

[PRE2]

In this code, if the device is Android, it plays a `.ogg` file. If the device is iOS, it plays a `.caf` file.

# Playing a sound effect

By using `SimpleAudioEngine`, we can play sound effects; to play them, we need to perform only two steps, namely preload and play. Sound effects are not background music; note that we can play multiple sound effects but only one background score at the same time. In this recipe, we will explain how to play sound effects.

## Getting ready

As in the case of playing background music, you have to include a header file for `SimpleAudioEngine`.

[PRE3]

## How to do it...

Let's try to immediately play a sound effect. The audio format is changed depending on the operating system by using the macro that was introduced at the time of playing the background music. The code for playing sound effects is as follows:

[PRE4]

## How it works...

The overall flow is the same as that for playing background music. You need to preload a sound effect file before playing it. The sound effect file is smaller than the background music file. So, you can preload a lot of sound effects before playing them.

## There's more...

The number of sound effects that we can play at the same time on Android is less than that on iOS. So, we will now explain how to increase this number for Android. The maximum number of simultaneous playbacks is defined in `Cocos2dxSound.java`.

The path of `Cocos2dxSound.java` is `cocos2d/cocos/platform/android/java/src/org/cocos2dx/lib`. Then, in line 66, the maximum number of simultaneous playbacks is defined.

[PRE5]

If we changed this value to 10, we can play 10 sound effects at the same time.

# Controlling volume, pitch, and balance

You can control the volume, pitch, and balance for sound effects. The right blend of these three factors makes your game sound more fun.

## How to do it...

Let's try to immediately play a sound effect by controlling its volume, pitch, and balance. The following is the code snippet to do so:

[PRE6]

## How it works...

You can control the volume for sound effects by using the `setEffectsVolume` method. The maximum value for the volume is 1.0, and the minimum value is 0.0\. If you set the volume to 0.0, the sound effect is muted. The default value of the volume is 1.0.

You can play multiple sound effects at the same time, but you cannot set the volume for these effects individually. To change the master volume for sound effects, set a volume by using the `setEffectsVolume` method. If you want to change the volume individually, you should use a `gain` value; which we will explain later.

The second argument in the `playEffect` method is the flag for continuously playing the sound effects. For the third and the subsequent arguments, please check the following table:

| Arguments | Description | Minimum | Maximum |
| --- | --- | --- | --- |
| Third argument (`pitch`) | Playing speed | 0.0 | 2.0 |
| Fourth argument (`pan`) | Balance of left and right | -1.0 | 1.0 |
| Fifth argument (`gain`) | Distance from a sound source | 0.0 | 1.0 |

The `pitch` is the quality that allows us to classify a sound as relatively high or low. By using this `pitch`, we can control the playing speed in the third argument. If you set the `pitch` to less than 1.0, the sound effect is played slowly. If you set it to more than 1.0, the sound effect is played quickly. If you set it to 1.0, the sound effect plays at the original speed. The maximum value of the `pitch` is 2.0\. However, you can set the `pitch` to more than 2.0 in iOS. On the other hand, the maximum value of the `pitch` in Android is 2.0\. Therefore, we adopted the maximum value as the lower.

You can change the balance of the left and the right speakers by changing the `pan` in the fourth argument. If you set it to -1.0, you can hear it only from the left speaker. If you set it to 1.0, you can hear it from only the right speaker. The default value is 0.0; you can hear it at the same volume from both the left and the right speakers. Unfortunately, you will not be able to figure out much difference in the speaker of the device. If you use the headphones, you can hear this difference.

You can change the volume of each sound effect by changing the `gain` in the fifth argument. You can set the master volume by using the `setEffectVolume` method and the volume of each effect by changing the gain value. If you set it to 0.0, its volume is mute. If you set it to 1.0, its volume is the maximum. The final volume of the sound effects will be a combination of the gain value and the value specified in the `setEffectsVolume` method.

# Pausing and resuming background music

This recipe will help you better understand the concept of pausing and resuming background music.

## How to do it...

It is very easy to stop or pause the background music. You don't specify the argument by using these methods. The code for stopping the background music is as follows:

[PRE7]

Code for pausing:

[PRE8]

Code for resuming the paused background music:

[PRE9]

## How it works...

You can stop the background music that is playing by using the `stopBackgroundMusic` method. Alternatively, you can pause the background music by using the `pauseBackgroundMusic` method. Once you stop it, you can play it again by using the `playBackgroundMusic` method. Further, if you pause it, you can resume playing the music by using the `resumeBackgroundMusic` method.

## There's more...

You can determine whether the background music is playing by using the `isBackgroundMusicPlaying` method. The following code can be used for doing so:

[PRE10]

### Tip

However, you are required to be careful while using this method. This method always returns a true value that specifies the playing status in the iOS simulator. At line 201 of `audio/ios/CDAudioManager.m` in Cocos2d-x, if the device is the iOS simulator, `SimpleAudioEngine` sets the volume to zero and plays it continuously. That's why there is a problem in the iOS simulator. However, we tested the latest iOS simulator before commenting out this process and found that there was no problem. If you want to use this method, you should comment out this process.

# Pausing and resuming sound effects

You might want to stop sound effects too. Also, you may want to pause them and then resume them.

## How to do it...

It is very easy to stop or pause a sound effect. The following is the code for stopping it:

[PRE11]

The following is the code for pausing it:

[PRE12]

You can resume the paused code as follows:

[PRE13]

## How it works...

`SimpleAudioEngine` can play multiple sound effects. Therefore, you have to specify the sound effect if you want to stop or pause it individually. You can get the sound ID when you play the sound effect. You can stop, pause, or resume the specific sound effect by using this ID.

## There's more...

You can stop, pause, or resume all the playing sound effects. The code to do so is as follows:

[PRE14]

# Playing background music and a sound effect by using AudioEngine

`AudioEngine` is a new class from Cocos2d-x version 3.3\. `SimpleAudioEngine` cannot play multiple background scores, but `AudioEngine` can play them. Furthermore, `AudioEngine` can call a callback function when it finishes playing the background music. In addition, we can get the playtime by using the callback function. In this recipe, we will learn more about the brand new `AudioEngine`.

## Getting ready

We have to include the header file of `AudioEngine` to use it. Further, `AudioEngine` has a namespace called `experimental`. To include the header file, you will need to add the following code:

[PRE15]

## How to do it...

`AudioEngine` is much easier than `SimpleAudioEngine`. Its API is very simple. The following code can be used to play, stop, pause, and resume the background music.

[PRE16]

## How it works...

`AudioEngine` no longer needs the preload method. Further, `AudioEngine` does not distinguish between background music and sound effects. You can play both background music and sound effects by using the same method. When you play it, you can get a sound ID as the return value. You have to specify the sound ID when you change the volume, stop it, pause it, and so on.

## There's more...

If you want to unload audio files from the memory, you can `uncache` by using the `AudioEngine::uncache` method or the `AudioEngine::uncacheAll` method. In the case of the `uncache` method, you have to specify the path that you want to unload. In the case of the `uncacheAll` method, all audio data is unloaded from the memory. While unloading files, you have to stop the related music and sound effects.

# Playing movies

You might want to play a movie in your game in order to enrich the representation. Cocos2d-x provides a `VideoPlayer` class for this purpose. This class makes it easy to play a movie; however, it is still an `experimental` class. So, you have to be very careful while using it.

## Getting ready

You have to prepare something before using the `VideoPlayer` class.

1.  You have to add the movie file to the `Resources/res` folder. In this case, we add the video called `splash.mp4`.
2.  Next, you have to including a header file. The code to do so is as follows:

    [PRE17]

3.  Then, you have to add the following code to the `proj.android/jni/Android.mk` file for building an Android application.

    [PRE18]

4.  In Xcode, you have to add `MediaPlayer.framework` for iOS, as shown in the following image:

![Getting ready](img/B0561_06_01.jpg)

## How to do it...

Let's try to play the video in your game. Here, it is:

[PRE19]

## How it works...

Basically, the `VideoPlayer` class is the same as the other nodes. First, you create an instance, specify its location, and then add it on a layer. Next, you set the content size by using the `setContentSize` method. If you set a false value by using the `setKeepAspectRatioEnabled` method, the video player's size becomes equal to the content size that you specify by using the `setContentSize` method. In contrast, if you set a true value, the video player retains the aspect ratio for the movie.

You can get the event of the playing status by adding an event listener. `VideoPlayer::EventType` has four types of events, namely `PLAYING`, `PAUSED`, `STOPPED`, and `COMPLETED`.

Finally, you set the movie file by using the `setFileName` method and you can play it by using the `play` method.

### Tip

There are a lot of video formats. However, the video format that you can play on both iOS and Android is mp4\. That's why you should use the mp4 format to play videos in your games.