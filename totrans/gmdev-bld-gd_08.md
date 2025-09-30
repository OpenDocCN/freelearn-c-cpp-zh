# 8

# Adding Sound Assets

Sound is often the most neglected part of game projects. While creating visual assets may seem hard to do, a lot of us still tackle it because we get quick and reliable feedback, however, most people don’t even know where to start when it comes to producing sound assets. Luckily, there are royalty-free assets out there that you can use.

This chapter will not cover how to make sound assets but how to import them into your game. We will focus on some of the technical aspects of sound management in Godot. This involves learning about the different sound formats the engine supports. Picking the appropriate sound format is no different than ironing out a topology for a 3D model for animation. Choose wisely and, even better, know the benefits and limitations of each format.

Next, you will learn when and how some sound assets should be looped. We’ll investigate the import options for different sound types and mention format-specific differences. We’ll also discuss scenarios where it makes sense to have your sound assets looped.

Lastly, we’ll get to know different types of Godot nodes that are responsible for playing sound assets in your scenes. This way, you can pick the appropriate audio player node for your project. To finish off, we’ll play some sample sound assets to show the differences between these different nodes.

Needless to say, to make the best of this chapter, you may want to be in a quiet place where you can practice some of the topics, especially in the later sections of this chapter.

In this chapter, we will cover the following topics:

*   Learning about different sound formats
*   Deciding on looping or not
*   Playing audio in Godot

By the end of this chapter, you’ll know how to import sound assets, choose which file type is correct, configure their settings, and play them in your project automatically or when it’s needed.

# Technical requirements

Unlike the other chapters, instead of a `Finish` folder with individual assets, we’ll give you the finished Godot project with all the scenes and scripts set up. Nevertheless, we would like you to practice but focus solely on the topics presented in this chapter. Thus, we suggest you start with a clean slate, import the sound files from the `Start` folder, and follow along. Following tradition, the necessary resources can be found in this book’s GitHub repository: [https://github.com/PacktPublishing/Game-Development-with-Blender-and-Godot](https://github.com/PacktPublishing/Game-Development-with-Blender-and-Godot).

# Learning about different sound formats

Sound files come in different formats, just like graphics files can come in different formats including JPG, GIF, PNG, and others. The industry, and sometimes the consumers, define the fate of these formats. Let’s place the consumers in the right context here. Occasionally, the specifications laid out by the creator of a file format are not welcome by the people who are using this very format to produce the work. Then, the work is created but not accepted by the platforms that would disperse such content due to technical reasons. It’s almost like a tug of war where the inconvenience or the cost of maintaining a file type outweighs the benefits and the ease of use. At these times, we tend to hear about newer formats, hence there being a multitude of file formats out there.

Most of the time, this kind of technical layer is not visible to an end user, especially if they are only perusing the content, such as listening to music on Spotify or YouTube. However, since we are building a game, even though we are not too concerned about the production of such assets, we should still be knowledgeable on this topic since we’d like to pick the most appropriate file format for a certain scenario.

Distinguishing what sound means

This is a note on what we mean by sound. We’ll be using the word sound or audio, in this chapter and the rest of this book, to cover all possible scenarios, such as the feedback you get when you interact with UI elements, when a player character is notified by an in-game event, or ambient music.

The version of Godot, 3.4.4, that this book is covering currently supports three different audio file formats. Each has different advantages and limitations. Although converting these files into each other is possible, after we present their formal definitions, perhaps you’ll decide not to.

## Introducing WAV

Pronounced *wave*, WAV files have been around since the early 90s. It’s the short form of Wavefront Audio File Format, a file specification created by IBM and Microsoft. This is a popular format among music and audio professionals, despite being uncompressed since it retains the quality of a sound recording. Thanks to the improvements in file storage capacity and internet speed, the high file size doesn’t seem to be a big issue anymore.

On the limitation side, as far as technical aspects go, a WAV file can’t exceed 4 GB. However, this should not be a concern because that number is equivalent to almost 7 hours of audio. It is extremely unlikely there will be one audio file in any video game of that size.

So, why should you choose this format? Since it’s an uncompressed file type, the CPU that is also responsible for processing a sound file will have an easier time playing it. A likely scenario for using this file type is for sound effects. Usually, these effects are short-lived, such as the creaking of a door, the swing of a sword, and so on. The file size won’t matter that much because the duration will be short.

Conversely, this is not the best format for background music. Sure, there won’t be any need to decompress the file to be able to play it, but the file size will be significantly larger.

In summary, if you want a quick reaction and would rather have a sound file play as quickly as possible, such as effects, then this is the right format for you. After all, you wouldn’t want the CPU to be dealing with the decompression of an effect file while your game characters are busy with the next chain of events.

If you are willing to sacrifice a few hundred milliseconds to wait for a decompression, such as when not having the background music play instantly is a big deal, then you can opt for compressed file types. These come in two different flavors.

## Introducing OGG

We should start by clarifying this format since the name could be confusing if you come across some resources on the internet. Technically, OGG is a container file format that can hold file types such as audio, video, text, and metadata. Its developer and maintainer, *Xiph*, is also responsible for another audio file format known as **Free Lossless Audio Codec** (**FLAC**). So, according to OGG specifications, a FLAC could be part of an OGG file. Historically speaking, most OGG files out there have contained a different audio file format known as Vorbis. So, you may find some websites with Vorbis content that are essentially complying with the OGG format’s specifications.

Here is an example to simplify all these names and how they relate to each other. Consider OGG as a ZIP file that knows what to do with its content. An OGG file carrying a video and a subtitle file will trigger the necessary settings in a video player so that the player knows where to find the subtitles since they will be embedded in one file. Similarly, another OGG file with an audio and metadata file will command an audio player to display the album and track, record, and play the audio.

Since the format is not just one thing, but rather a set of files, it is often confusing to associate a specific need with one file extension. For example, the `.ogg` extension was used before 2007 as a multimedia holder as that was its original intention. Since then, Xiph suggests we use the `.ogg` extension for Vorbis audio files. Additionally, the company has created a new set of file extensions to simplify things:

*   `.oga` for audio-only files
*   `.ogv` for video
*   `.ogx` for multiplexed cases

Despite the naming conundrum, what you need to know is that the OGG audio format is compressed, so it’s a lossy file format. Lossy in our context means that we could attain almost the same sound quality by requiring less hard disk space. So, this is a good thing because this file format is a perfect fit for playing background music. Keep in mind that since the CPU has to decompress this file type, this is not the preferred format for playing quick sound effects.

Speaking of a lossy file format, our next candidate is another lossy file format that gained some notoriety in the early 2000s.

## Introducing MP3

When internet speed and disk storage were at a premium in the late 90s, MP3 filled an important gap in transferring audio content just when a big audience needed it at the turn of the millennium. Consumers flocked to websites to download copies of the tracks from their favorite bands. Sadly, so many of these websites did not bother to have a legal license to distribute such content, so this led to copyright infringements and, in the case of Napster, a lawsuit.

From a technical standpoint, MP3 files are somewhere in between WAV and OGG, compression-wise. So, you’ll get smaller file sizes in OGG for the same quality of sound. That being said, decompressing an MP3 file is faster than decompressing an OGG. Hence, this makes the MP3 format still useful, especially where CPUs are challenged to the maximum, such as in mobile devices.

Despite disk space getting cheaper and cheaper, from a business point of view, it still makes sense to prioritize WAV over MP3\. For example, some websites that offer royalty-free sound files provide the MP3 version but put the WAV version of a sound behind a paywall. Since an MP3 file has already lost some of the original data due to its compression algorithm, editing with this file over and over will yield more lossy results. So, having access to the original WAV file is always better if you want to make modifications to it. However, if you don’t need to, then you might be fine with an MP3 version.

## Wrapping up

In summary, WAV files are better for short sound effects whereas longer sound effects, especially theme music, would be handled better with MP3 files. At the time of writing, most sound libraries still don’t offer OGG commonly, despite being a good candidate. Nevertheless, if you have access to a lot of WAV files and you want to be efficient in file size, then you can convert them into OGG using online converters. Two examples are as follows:

*   [https://audio.online-convert.com/convert-to-ogg](https://audio.online-convert.com/convert-to-ogg)
*   [https://online-audio-converter.com/](https://online-audio-converter.com/)

In the case of music files, which are normally a few minutes long, if your original is in WAV format, then uploading and processing these files online may take a long time since the file sizes will easily be over 50 MB. Also, some of these online converters have file size limitations. To get around these limitations, here is a link to a website that compares some offline converters that you can employ in your efforts: [https://www.lifewire.com/free-audio-converter-software-programs-2622863](https://www.lifewire.com/free-audio-converter-software-programs-2622863).

Regardless of what file type you choose and whether it’s for a sound effect or music, there comes a point in your game development journey when you will have to decide if your sound asset should loop or not. In the next section, we’ll discuss the reasons why having the loop feature on or off is useful.

# Deciding on looping or not

A loop, in literal terms, is a continuous motion or structure in which if you pick a random spot, you could come back to it by traveling all the way through. In aural terms, this is similar, but we don’t start anywhere; we usually start playing a sound file, but the player restarts the track once it reaches the end.

This definition is classic, and not that insightful, so let’s do a better job by discussing it in various contexts inside Godot or any game projects. So, you can make informed decisions in your projects since it’s situation-specific. We’ll do this by presenting different use cases:

*   **Background music**: This is the most typical case where a music piece plays in the background while the game is running. The composer creates this kind of piece with the intention that once played back to back, there will be no abrupt end. The sound at the end of the file will seamlessly match the beginning. Sure, if you pay attention to the ups and downs in the rhythm, you will know where you are in the file, but so long as the loop setting is on, everything will sound smooth and blend in so that you can focus on your game experience.
*   **Machine gun**: Imagine that either the player or an enemy character is interacting with a machine gun in your game. Although short bursts are possible, due to the nature of machine guns, the gun might be fired continuously. So, instead of detecting if the sound file has reached the end and instructing the player to restart the file, you may want to play the file once if the said file’s loop feature is on. This way, the machine gun effect will play until a stop command is given.
*   **Doors**: This one is a bit of an edge case. Let’s assume we have visuals and other sound effects in our game that indicate that we’re in an outdoor scene on a windy day. Perhaps the door is in poor condition with rusted hinges, and one of the hinges is even leaning out a bit. The artist may have decided to have this door animated to match the wind’s effect on the door so that it oscillates between a closed and an open state. Here, it would make sense to have a looped sound file that contains most likely squeaks and creaks that are synchronized with the door’s animation.

However, if a door will be responding to a player character’s action such as it being opened or closed, then it doesn’t make sense to have the sound file in a loop. This is going to be a one-off event.

*   **User interface**: The sound you hear when you interact with a user interface falls under this category. These are usually not looped since they are event-based, similar to the one-off-door action from the previous use case. However, let’s present a case that may seem like looping is a good idea. Nevertheless, we’ll rule it out for a good reason.

Imagine that there is a UI component that’s helping the player set an amount. The interface has two buttons that will increment and decrement the amount the player is seeing. Placing a UI sound effect on either button is fine, and the sound will play only once, so long as the player keeps clicking. What if we would like to give the player a chance to press and hold the button down? After all, clicking a button ad nauseam to get to really high or low numbers may get tedious quickly. So, how should we treat the looping condition in this case?

Human perception is sensitive during events like this. Players are usually busy during gameplay, so they won’t perceive the delay while the CPU is busily decompressing a music file. However, we are usually very perceptive in detecting the discrepancies at the end of a holding event for a UI button. So, instead of treating repetitive UI events such as a machine gun, even though they might feel similar, designers opt to trigger the sound effect individually instead of looping it.

In this section, we presented different use cases where the use of looping, or lack thereof, is common. However, what you haven’t seen is how to turn the loop functionality on and off. We’ll show this by revisiting our old friend, the **Import** panel.

## Turning the looping on and off

So far, we have discussed what looping is and under which scenarios it may make sense to have it on or off, but we haven’t seen how we can flip its status. In this section, we’ll put sound files of each type in our project and study their settings in the **Import** panel.

We are going to use the `Loop_Someday_03.wav` file from the *Freesound* website, which was created by a user called *LittleRobotSoundFactory*. The sound was originally in WAV format, but we have converted it into OGG and MP3 versions as well. You can find all the versions in the `Start` folder and compare their file sizes.

Once you’ve added the files to your project, let’s learn how Godot recognizes these files. So, switch on the **Import** panel, and select either the OGG or MP3 version. Then, select the WAV version. The interface differences are shown in the following screenshot:

![Figure 8.1 – The MP3 and OGG versions have fewer import settings than the WAV version ](img/Figure_8.1_B17473.jpg)

Figure 8.1 – The MP3 and OGG versions have fewer import settings than the WAV version

As you can see, by default, the MP3 and OGG versions come with the loop setting on. Also, these versions don’t seem to have that many settings. On the other hand, the WAV version’s loop is off by default. Why is that?

If you remember what we introduced for different sound formats earlier in the *Learning about different sound formats* section, Godot took the liberty of looping the compressed versions since these will most likely be used for background music. On the contrary, if our example file was for a sound effect, we’d most likely use a WAV file with no loop, since it’d be a quick one-off thing with minimal CPU requirements.

Other WAV settings

Since we are currently working with the **Import** interface, let’s also point out that you can reduce the file size of your WAV files by turning on some of the options in the **Force** section. *Figure 8.1* shows this and some other settings, such as trimming and normalizing your files. The former of these will trim the silent part at the beginning and the end of files, which is sometimes automatically added when exporting WAV files. This is especially important if you want your sound effects to start right away without a delay.

So, turning the loop feature for any given sound file on and off is as easy as a click and you know how to do it. Perhaps it’s more important to decide whether a file should be looped or not. This is something you’ll have to answer along the way.

Regardless, you still need a Godot node to play your sounds at some point. In the next section, we’ll get to know the different audio players Godot uses, and attach our sound files to the appropriate player.

# Playing audio in Godot

Since Godot uses nodes for almost everything, it is no different for playing sounds. To play an audio file, there are nodes you can attach to your scene, and you can configure them according to whether it’s for a 2D or 3D game. We’ll focus on different audio players Godot uses in this section.

No matter what audio file type you choose, you will be able to play it with the nodes we’ll present in this section. The experience you’ll feel will be different, of course, based on the node type, but this is something you have to decide, depending on the type of game you are making. So, let’s look at the audio streamer nodes Godot uses so that you can pick the appropriate one. Your three choices are as follows:

*   **AudioStreamPlayer**: This node’s official definition is somewhat dry; it plays audio non-positionally. What this means is that you are not concerned with which direction the audio is coming from. For an FPS game, it’s essential to know in which direction the enemy is firing at you. This involves positional data. You don’t have any kind of positional information in this audio node. However, this is the right candidate for playing background music. Find more about it at [https://docs.godotengine.org/en/3.4/classes/class_audiostreamplayer.xhtml](https://docs.godotengine.org/en/3.4/classes/class_audiostreamplayer.xhtml).
*   **AudioStreamPlayer2D**: You guessed it – this node includes position information. So, the farther away the camera is from this node, the quieter the sound will be. This node is useful for 2D platformer games, for example. So, as soon as a game object enters the view, the stream will be picked up by the camera. Also, objects that are on the right-hand side of the camera will prioritize the right speakers and vice versa. More details are available at [https://docs.godotengine.org/en/3.4/classes/class_audiostreamplayer2d.xhtml](https://docs.godotengine.org/en/3.4/classes/class_audiostreamplayer2d.xhtml).
*   **AudioStreamPlayer3D**: Last but not least is the 3D version of an audio streamer. This conveys 3D positional information to a listener. Therefore, this is the kind of audio streamer node you’ll be using in 3D setups. Naturally, this type of streamer employs more advanced features, such as attenuation, which controls how the sound will dampen over a distance, and Doppler effects. Thus, it might be a good idea to examine its properties by visiting [https://docs.godotengine.org/en/3.4/classes/class_audiostreamplayer3d.xhtml](https://docs.godotengine.org/en/3.4/classes/class_audiostreamplayer3d.xhtml).

We could go over every property for each type of stream player, but we leave that task to you since picking the right streamer and configuring its settings is a form of art. We’ll use the proper streamer when we build our game later in this book and focus on the important settings in that context. In the meantime, you can read what each one is capable of by going to the aforementioned URLs from the official documentation.

That being said, we won’t leave this chapter just yet. Let’s play a few sounds to simulate some of the examples we’ve enumerated so far.

## Playing background music

Let’s practice some of the things we’ve covered in this chapter. We’ll start by playing a sound that’s a good candidate for background music. We’ll use the MP3 version of the `loop-someday-03` file we imported in the *Deciding on looping or not* section. To play this sound as background music, follow these steps:

1.  Create a new scene and save it as `Background-Music.tscn`.
2.  Add an **AudioStreamPlayer** node to your scene and turn on its **Autoplay** property in the **Inspector** panel.
3.  Drag and drop `loop-someday-03.mp3` from the **FileSystem** panel into the **Stream** property in the **Inspector** panel.
4.  Press *F6*.

This will launch your current scene and automatically play the MP3 file. Since the file’s loop setting is set to true, the 9-second-long music will play endlessly. You can now add this scene to other scenes where you want to have background music.

## Playing a sound effect on demand

For this effort, we’ll return to the machine gun example from the *Deciding on looping or not* section. The sound for the machine gun is also set to loop, but we wouldn’t want this to *autoplay* when a scene is launched. It’s most likely that your player character will enter or approach an area where enemy forces are pummeling you with machine gun fire. Let’s write some code to simulate this sort of triggering behavior:

1.  Create a new scene and save it as `Machine-Gun.tscn`.
2.  Add an **AudioStreamPlayer** node to your scene and attach a script to it with the following lines of code in it:

    ```cpp
    extends AudioStreamPlayer
    func _unhandled_key_input(event: InputEventKey) -> void:
        if event.is_pressed() and event.scancode == 
          KEY_SPACE:
            play()
        else:
            stop()
    ```

3.  Drag and drop `machine-gun.ogg` from the **FileSystem** panel into the **Stream** property in the **Inspector** panel.
4.  Press *F6*.

Since we want the stream to play on demand, we are wiring it to a condition to be true – that is, pressing the spacebar. Go ahead and press it once or twice; even hold it down for a brief period. You’ll hear the machine gun sound going on or off, thanks to the play and stop commands of the **AudioStreamPlayer** node.

The script we’ve implemented looks good enough, but it’s also a bit problematic. Maybe you’ve already noticed it. Try to hold down the spacebar for long enough, such as 3 or 4 seconds, and you’ll hear a jamming sound. This is because the script is firing too many play commands. So, after a while, the CPU will be instructed to play the same asset too many times. We can do better by replacing the script’s content with the following:

```cpp
extends AudioStreamPlayer
func _unhandled_key_input(event: InputEventKey) -> void:
    if event.is_pressed() and event.scancode == KEY_SPACE:
        stream_paused = false
    else:
        stream_paused = true
```

Here, we have replaced the lines that had the play and stop commands with a different kind of command. The new version will control whether the stream should be paused or not. For this script to work, we need to turn on two things in the **Inspector** panel:

*   **Autoplay**
*   **Stream Paused**

This new setup will play the stream automatically, similar to what happened in the *Playing background music* section, but then pause it right away. This seems counter-intuitive at first, but let’s analyze what the new script is doing. When the spacebar is pressed, we resume the stream, and since the stream was already playing, thanks to `else` case, the stream will be paused again. So, the new script will not send consecutive play and stop commands, and thus will not clog the CPU.

We’ll conclude by discussing two more flavors of the machine gun firing in light of what we have presented throughout this chapter.

## Increasing gameplay experience

Did you notice that we used the same type of audio stream node for both the background music and machine gun? In a way, we treated the machine gun as if it was background music. In other words, we were not too concerned about where the sound was coming from.

To deliver a more enjoyable gameplay experience, you could use the **AudioStreamPlayer2D** and **AudioStreamPlayer3D** nodes in 2D and 3D games, respectively. By tweaking the attenuation values of these nodes, which define how sound travels over distances, your players can hear the sound of the machine gun louder and louder as their characters get closer to the source. This would elevate the sense of danger, and it’s a cheap and nice way to deliver immersion.

# Summary

We started this chapter by presenting different types of files that Godot uses for playing sound. Knowing the differences among these formats, when you work with composers, you can emphasize in which format you want your sound files to be delivered. Chances are they might ask you about this, and they might even deliver in all three possible formats.

Next, we discussed a few cases where looping a sound file might be a good idea. To facilitate this, we investigated the options presented in the **Import** panel. However, the decision to loop or not is still something you’ll have to decide.

Finally, to put our theoretical knowledge to use, we created two scenes that could play the sample files. In the first case, we attached a sound file to an audio streamer and let it play automatically. For the second case, we wrote a very simple script that let you start and stop the sound to mimic an enemy character’s behavior, hence the sound effect it may make.

So far, we have been discovering some of the ingredients that are necessary for building games, such as importing assets – whether it’s models from the previous chapter or sound assets in this one. In the next chapter, we’ll dive right into building our point-and-click adventure game by designing our level.

# Further reading

If you are into creating music and sound effects, here is a short list of software you can start with:

*   LMMS: [https://lmms.io](https://lmms.io)
*   Waveform Free: [https://www.tracktion.com/products/waveform-free](https://www.tracktion.com/products/waveform-free)
*   Cakewalk: [https://www.bandlab.com/products/cakewalk](https://www.bandlab.com/products/cakewalk)

The aforementioned links will only cover the using a tool side of music production, so you may also need to learn the artistic side of it, for which there are courses on multiple online training platforms, such as Udemy.

By the way, if you see a sound file out there and it looks like it is free to download, it doesn’t mean you have the license to utilize the piece in your work. You may want to read the fine print if you don’t want to get a surprise call from a lawyer someday. Nevertheless, the following are a few websites that offer paid and free sound content:

*   [https://gamesounds.xyz](https://gamesounds.xyz)
*   [https://freesound.org](https://freesound.org)
*   [https://www.zapsplat.com](https://www.zapsplat.com)
*   [https://opengameart.org](https://opengameart.org)

# Part 3: Clara’s Fortune – An Adventure Game

In this final part of the book, you'll be creating a point-and-click adventure game. Since it would be too time-consuming to prepare all the game assets, you'll be provided with the necessary files.

In this part, we cover the following chapters:

*   [*Chapter 9*](B17473_09.xhtml#_idTextAnchor146)*, Designing the Level*
*   [*Chapter 10*](B17473_10.xhtml#_idTextAnchor165)*, Making Things Look Better with Lights and Shadows*
*   [*Chapter 11*](B17473_11.xhtml#_idTextAnchor186)*, Creating the User Interface*
*   [*Chapter 12*](B17473_12.xhtml#_idTextAnchor206)*, Interacting with the World through Camera and Character Controllers*
*   [*Chapter 13*](B17473_13.xhtml#_idTextAnchor230)*, Finishing with Sound and Animation*
*   [*Chapter 14*](B17473_14.xhtml#_idTextAnchor255)*, Conclusion*