# Chapter 12. Can You Hear Me Now? – Sound and Music

There's nothing quite like the enjoyment of being immersed in a virtual environment. From the movies we watch to the games we play, appeal to and usage of as many human senses as possible can either make or break the captivation that a form of media can hold. Creating a living and breathing atmosphere can rarely, if ever, be only down to visual effects. Throughout this chapter, we will briefly close our eyes and engage in the auditory side of this project by covering subjects such as:

*   Basics of sound and music in SFML
*   Placement of sounds and the listener in 3D space
*   Proper management and recycling of sound instances
*   Expansion of the entity component system to allow for sounds

We have a long way to go until our first sonic boom, so let's dive right in!

# Use of copyrighted resources

Before we jump into managing sounds, let's give credit where it is due. Throughout this chapter, we're going to use the following resources:

*   *Fantozzi's Footsteps (Grass/Sand & Stone)* by *Fantozzi* under the CC0 license (public domain): [http://opengameart.org/content/fantozzis-footsteps-grasssand-stone](http://opengameart.org/content/fantozzis-footsteps-grasssand-stone)
*   *Electrix* (NES Version) by *Snabisch* under the CC-BY 3.0 license: [http://opengameart.org/content/electrix-nes-version](http://opengameart.org/content/electrix-nes-version)
*   *Town Theme RPG* by *cynicmusic* under the CC-BY 3.0 license: [http://opengameart.org/content/town-theme-rpg](http://opengameart.org/content/town-theme-rpg)

# Preparing the project for sound

In order to successfully compile a project that uses SFML audio, we need to make sure these additional dependency `.lib` files are included:

*   `sfml-audio-s.lib`
*   `openal32.lib`
*   `flac.lib`
*   `ogg.lib`
*   `vorbis.lib`
*   `vorbisenc.lib`
*   `vorbisfile.lib`

Additionally, the executable file must always be accompanied by the `openal32.dll` file, which comes with SFML and can be found inside the `bin` folder of the library.

# Basics of SFML sound

Anything audio related falls into one of two categories within SFML: `sf::Sound` that represents short sound effects, or `sf::Music` that is used to play longer audio tracks. It's prudent that we understand how these two classes are used before continuing further. Let's talk about each one individually.

## Playing sounds

The `sf::Sound` class is extremely lightweight and should only ever be used to play short sound effects that don't take up a lot of memory. The way it stores and utilizes actual audio files is by using a `sf::SoundBuffer` instance. It is analogous to `sf::Sprite` and the way it uses an instance of `sf::Texture` for drawing. The `sf::SoundBuffer` is used to hold audio data in memory, which the `sf::Sound` class then reads and plays from. It can be used as follows:

[PRE0]

As you can see, a sound buffer can be attached to an instance of `sf::Sound` by either passing it to the sound's constructor or by using the `setBuffer` method of a sound instance.

### Tip

As long as the sound is expected to be playing, the `sf::SoundBuffer` instance *shouldn't* be destroyed!

After the sound buffer loads the sound file and is attached to an instance of `sf::Sound`, it can be played by invoking the `play()` method:

[PRE1]

It can also be paused and stopped by using the appropriately named `pause()` and `stop()` methods:

[PRE2]

Obtaining the current status of a sound to determine if it's playing, paused, or stopped can be done like this:

[PRE3]

The status it returns is a simple enumeration of three values: `stopped`, `paused`, and `playing`.

Lastly, we can adjust the sound's volume, pitch, whether it loops or not, and how far the sound has progressed by using these methods respectively:

[PRE4]

Audio pitch is simply a numeric value that represents frequency of the sound. Values above 1 will result in the sound playing at a higher pitch, while anything below 1 has the opposite effect. If the pitch is changed, it also changes the sound's playing speed.

## Playing music

Any `sf::Music` instance supports all of the methods discussed previously, except `setBuffer`. As you already know, `sf::Sound` uses instances of `sf::SoundBuffer` that it reads from. This means that the entire sound file has to be loaded in memory for it to be played. With larger files, this quickly becomes inefficient, and that's the reason `sf::Music` exists. Instead of using buffer objects, it streams the data from the file itself as the music plays, only loading as much data as it needs for the time being.

Let's take a look at an example:

[PRE5]

Notice the name of the method `openFromFile`. In contrast, where sound buffers load files, `sf::Music` merely opens it and reads from it.

A very important thing to mention here is that `sf::Music` is a non-copyable class! This means that any sort of assignment by value will automatically result in an error:

[PRE6]

Passing a music instance to a function or a method by value would also produce the same results.

# Sound spatialization

Both `sf::Sound` and `sf::Music` also support spatial positioning. It takes advantage of left and right audio channels and makes it feel like the sound is actually playing around you. There is a catch, though. Every sound or music instance that is desired to be spatial has to only have a single channel. It is more commonly known as a monophonic or mono sound, as opposed to stereo that already decides how the speakers are used.

The way sounds are perceived in three-dimensional space is manipulated through a single, static class: `sf::Listener`. It's static because there can only ever be one listener per application. The main two aspects of this class we're interested in are the position and direction of the listener. Keep in mind that although we may be working on a 2D game, SFML sounds exist in 3D space. Let's take a look at an example:

[PRE7]

First, let's address the three-dimensional coordinates. In SFML, the default up vector is on the positive *Y* axis. Look at the following figure:

![Sound spatialization](img/B04284_12_01.jpg)

Each axis the character is on represents a direction vector in three dimensions

This arrangement of axes is known as a *right-handed Cartesian coordinate system* and is the standard for OpenGL, which is the basis of SFML. What this means is that what we've been calling the *Y* axis in two dimensions is really the *Z* axis in a three dimensional space. That's important to keep in mind if we want to have correct results when moving sound through space.

The listener direction is represented by something called a unit vector, also referred to as a normalized vector, which means it can only have a maximum magnitude of 1\. When setting the listener's direction, the vector provided is normalized again, so these two lines of code would produce equivalent results of a south-east direction:

[PRE8]

For our purposes, however, we're not going to need to use diagonal directions, as our main character, who will obviously be the sole listener, can only face four possible directions.

## Placing sounds in space

Much like how sprites are positioned in two-dimensional space, sounds can be positioned as well by using the method with the same name:

[PRE9]

Let's say that the listener is facing in the positive *X* direction *(1.f, 0.f, 0.f)*. The sound that we just placed at coordinates *(5.f, 0.f, 5.f)* would be five units ahead and five units to the right of our listener and would be heard through the right speaker. How loud would it have to be, though? That's where the minimum sound distance and attenuation come in:

[PRE10]

Sound minimum distance is the threshold at which the sound begins to lose volume and gets quieter. In the preceding example, if the listener is closer or exactly six units of distance away from the sound source, full volume of the sound will be heard. Otherwise, the sound begins to fade. How fast it fades is determined by the attenuation factor. Consider this figure:

![Placing sounds in space](img/B04284_12_02.jpg)

The circle with a radius of Min_Distance represents an area, where the sound can be heard at maximum volume. After the minimum distance is exceeded, the attenuation factor is applied to the volume.

Attenuation is simply a multiplicative factor. The higher it is, the faster sound fades over distance. Setting attenuation to 0 would result in a sound heard everywhere, while a value like 100 would mean that it is heard only when the listener is very close to it.

Remember that although we're not going to be taking advantage of it, music in SFML behaves under the same rules of spatialization as sound, as long as it only has one channel.

# Audio manager

Similar to what we did for textures and fonts, we're going to need a way to manage `sf::SoundBuffer` instances easily. Luckily, our `ResourceManager` class is there to make it extremely convenient, so let's create the `AudioManager.h` file and define the way sound buffers are set up:

[PRE11]

As you can tell already, the sound interface is pretty much exactly the same as that of textures or fonts. Similar to the previous resource managers, we also provide a file that paths are loaded from. In this case, it is the `audio.cfg` file:

[PRE12]

Once again, it is just like dealing with textures or fonts. So far, so good!

# Defining sound properties

Sound, much like any other medium, has a few different properties of interest that are up for tweaking. The effects we're going to be playing in our game don't just have varying sources, but also different volumes, pitch values, the distance a sound can cover, and a factor that represents how fast that sound fades. How we're going to store this information is defined in `SoundProps.h`:

[PRE13]

In addition to the qualities described earlier, it is also necessary to store the identifier of the audio file that a sound is going to be using. A typical sound file for our application would look something like `footstep.sound`:

[PRE14]

With this out of the way, we can actually jump right into managing the `sf::Sound` instances!

# Managing sounds

Due to the limitations on the number of sounds we can have in an application, it's best to have a centralized way of handling and recycling them. This is where the `SoundManager` class comes in. Let's begin aliasing a data type for sound IDs:

[PRE15]

A simple integer type is more than qualified for the job of keeping sounds identified.

Additionally, we'll want to store some information with the sound instance:

[PRE16]

In order to properly deallocate resources when it counts, we're going to want to store the string identifier of the audio file that the sound is using. Keeping track of whether the sound has been paused automatically or not is important for consistency. That's what the `m_manualPaused` Boolean flag is there for.

Lastly, before we delve deeper into the sound manager, looking at a few type definitions used here is essential:

[PRE17]

The `SoundProperties` type is just a map that associates the name of a sound to a structure that contains its properties. `SoundContainer` is another map that ties a `SoundID` to a pair that contains the `SoundInfo` structure, as well as the actual instance of the `sf::Sound` object. The `Sounds` data type is responsible for grouping these sound containers by `State`.

Further down the line, as sounds get recycled, they need to be moved to a different container of type `RecycledSounds`. It stores the sound ID and name alongside the `sf::Sound` instance.

The last type definition we're going to be dealing with here is a container for `sf::Music` instances. Just like sounds, they're grouped by states. One major difference here is the fact that we're only allowing one instance of `sf::Music` per state, which is stored together with a `SoundInfo` structure.

Now that we have everything we need, let's take a look at the sound manager header file:

[PRE18]

As mentioned previously, it's a good idea to keep the number of `sf::Sound` and `sf::Music` instances in your application down to a designated limit that never exceeds 256\. In this case, we're playing it pretty safe by using static data members for setting a limit of 150 sounds loaded in memory at the same time. In addition to that, we're also setting a limit to how many sound instances can be recycled before they're used again, which is 75\. These values can obviously be tweaked to your liking.

Let's talk about the private data members of this class before we get into implementation details. As expected, the sound and music containers are stored in this class under the names `m_audio` and `m_music`. Additionally, we're storing all of the sound properties in this class, alongside the recycled sound container. Because sound functionality is state based, the `m_currentState` data member is necessary for keeping tabs on what state the application is running in.

In order to assign sound IDs properly, keeping track of the last ID is a good idea, hence `m_lastID`. Also, since enforcing restrictions on how many instances of `sf::Sound` and `sf::Music` can be "alive" at the same time is of paramount importance; `m_numSounds` is used to keep track of every instance of these two classes. We're also going to need to check time passage in our application, which is what `m_elapsed` will be used for.

Finally, a pointer to the audio manager is kept around for resource management and retrieval.

## Implementing the sound manager

Let's begin, as always, by looking at the constructor and destructor of this class:

[PRE19]

A pointer to an `AudioManager` instance is obtained through the argument list of the constructor and initialized in the initializer list, alongside other data members and their default values. The destructor simply invokes another method called `Cleanup()`, which is responsible for the de-allocation of memory. It will be covered shortly.

We have already discussed the role that application states play in sound management. Now, let's take a look at actually defining the behavior of sound when states are changed:

[PRE20]

Upon the application state being altered, a `PauseAll` method is invoked with the argument of `m_currentState`. It's responsible for effectively silencing every sound that is currently playing. We don't want to be hearing fights and explosions of the in-game action while we're in the main menu. The `UnpauseAll` method is called next, with the identifier of the state being changed to being passed in as the argument. Obviously, if we're in the main menu and we're switching back to the game state, we want all of the action to resume and this includes all of the sound effects. The data member that holds the current state information is then altered.

The last few lines of code in this method are responsible for making sure that the music container has information about the new state. If nothing is found, some blank information is inserted into the `m_music` container in order to signify that the current state currently has no music playing.

Next, let's talk about what happens when a state is removed from the application:

[PRE21]

The sound container is first obtained for the state that is being removed. Every sound in that state is then iterated over and recycled via the `RecycleSound` method, which takes in the sound ID, pointer to the `sf::Sound` instance, and the sound name. Once that is done, all of the state information is erased from the `m_audio` container. Additionally, if an instance of `sf::Music` is found in that state, the memory for it is deallocated and the number of sounds currently existing in memory is decreased.

Good memory management is extremely important in an application and is one of the main reasons we're using manager classes instead of simply having resources scattered all over the place. The method responsible for cleaning up the mess in this case might look a little something like this:

[PRE22]

First, we iterate over the container of currently playing sounds and release the audio resources that are being used. The dynamic memory for the sound is then deleted safely instead of being recycled. The exact same process is repeated one more time for all of the sounds that exist in the `m_recycled` container. Finally, all of the music instances are also deleted. Once all containers are properly cleared, the number of sounds is set back to 0, along with the last sound ID.

Now that we've covered all of the "housekeeping" details, let's take a look at how we can make a system like this tick through its `Update` method:

[PRE23]

An important thing to keep in mind here is that we really don't need to run this chunk of code every single tick of the application. Instead, we keep track of time passing and check the `m_elapsed` data member each cycle to see if it's time to run our code yet. The `0.33f` value is arbitrary in this case and can be set to anything within a reasonable range.

If enough time has passed, we loop over every sound in the current state and check its status. If the sound has stopped, we can safely recycle it by invoking the `RecycleSound` method and then remove it from our primary sound container.

### Tip

When an element in an STL container is removed, all iterators of said container become invalid. If left unattended, this can lead to elements being skipped or out of bounds accesses. It can be addressed by setting the iterator to the return value of the `erase` method, as it returns a valid iterator to an element *after* the one that has been erased. It increments the iterator only if an element hasn't been erased during the current cycle of the loop.

In this system, music follows the exact same treatment and is removed if it's no longer playing.

Next, let's look at providing a way for the users of this class to play sounds:

[PRE24]

We begin by obtaining a pointer to the sound properties structure by using the `GetSoundProperties` method, which we will be covering later. If it returned a `nullptr` value, -1 is returned by the `Play` method to signify a loading error. Otherwise, we proceed by creating a sound ID instance that is going to be passed in *by reference* to the `CreateSound` method, along with the identifier of the audio sound buffer. If the sound was created successfully, it returns a pointer to the `sf::Sound` instance that is ready to be used.

The `SetUpSound` method is then invoked with pointers to the `sf::Sound` instance and properties being passed in as arguments, as well as two Boolean flags for whether the sound should loop and be relative to the listener. The latter two are passed in as arguments to the `Play` method we're currently implementing. The sound is then positioned in space and stored in the `m_audio` container, along with the `SoundInfo` structure that is set up just one line before and holds the audio identifier.

The final step is then calling the `play()` method of our sound instance and returning the ID of said sound for later manipulations.

As the header file suggests, there are two versions of the `Play` method. Let's cover the other one now:

[PRE25]

This version of the `Play` method only takes in a single argument of the sound ID and returns a Boolean flag. It's meant to start an already existing sound, which begins by it being located in the sound container. If the sound has been found, its `play` method is invoked and the `m_manualPaused` flag is set to `false`, showing that it is no longer paused.

Stopping a sound works in a very similar fashion:

[PRE26]

The only difference here is that the `stop` method is invoked instead, and the `m_manualPaused` flag is set to `true` to signify that it has been paused in a non-automatic fashion.

One more method that follows the exact same pattern is the `Pause` method:

[PRE27]

Now it's time to move on from sound and to music, specifically how it can be played:

[PRE28]

First, the music element for the current state is located. The path to the actual audio file is then obtained by using our newly added `GetPath` method and checked for being blank. If it isn't, we check whether an actual instance of `sf::Music` exists for the current state and create one if it doesn't. The `openFromFile` method of the `sf::Music` instance is then called in an `if` statement in order to check if it was successful or not. If it wasn't, the `sf::Music` instance is deleted and the number of sounds is decreased. Otherwise, the music instance is set to the volume and loop preferences provided as arguments and played. Note that we're setting every music instance to also be relative to the listener. While it is possible to make music positional, we have no need for it at this point.

Because we want the same functionality for music as we do for any given sound, we have a fairly similar line-up of methods for manipulating music as well:

[PRE29]

Let's get back to sound now. Since we're going to be utilizing its spatial qualities, it's a good idea to have a method that can be used for setting its position in space:

[PRE30]

This method simply locates the sound instance in its container and sets its position to the one provided as an argument.

What if we want to check if a sound is still playing? No problem! That's what the `IsPlaying` method is for:

[PRE31]

Due to the fact that sound status is a simple enumeration table, it can be forced into a Boolean value. Since we don't care about the "paused" state, returning the status as a Boolean works just fine.

Next, we have a way for obtaining the sound properties:

[PRE32]

Because sound properties aren't loaded during start-up, simply not finding the right information might simply mean that it was never loaded. If that's the case, the `LoadProperties` method is invoked. It returns a Boolean value that informs us of a failure, in which case a `nullptr` value is returned. Otherwise, the properties structure is searched for again and then returned at the end of this method.

As we're on the subject of loading properties, let's actually take a look at how they're loaded from the `.sound` file:

[PRE33]

Having loaded many files in the past, this should be nothing new. So, let's just breeze right through it. A temporary `SoundProps` instance called `props` is created on the stack with a default audio name that is blank. The file is then processed and checked line by line for relevant keywords. The information is then loaded directly into the temporary properties instance using the `>>` operator.

### Tip

For extra credit, the `if else` chain could be replaced with some sort of associative container of lambda functions, but let's keep the logic as it is for the sake of simplicity.

Once the file has all been read in, it is closed and the audio name of the properties instance is checked for not being a blank, as it should've been loaded during the process. If the name is, in fact, something other than a blank, the `SoundProps` instance is inserted into the property container and true is returned for success.

As we were covering changing states, a few methods for pausing and starting all sounds were introduced. Let's take a look at one of them now:

[PRE34]

The `PauseAll` method first obtains the container of all sounds for the provided state. It iterates over each one and checks if the sound is actually stopped or not. If it is, the sound is simply recycled and the element is erased. Otherwise, the sound's `pause` method is called. Music for the provided state is also paused, provided that it exists.

The `UnpauseAll` method is simpler, as it has no reason to recycle sounds:

[PRE35]

The catch here is that the sounds and music are only played again if they weren't manually paused by their respective `Pause` methods.

Now, let's implement arguably the most important piece of this class that is responsible for actual creation and recycling of the `sf::Sound` instances:

[PRE36]

A local variable named `sound` is first set up with the value of `nullptr`, and it will be manipulated throughout the rest of this method. The size of the recycled sound container is then checked, along with whether the number of maximum sounds overall or maximum cached sounds has been exceeded.

If the number of sounds is too high on either count and the recycled container isn't empty, we know we're going to be recycling an already existing sound. This process begins by first attempting to find a sound that already uses the same `sf::SoundBuffer` instance. In the case of such sound not existing, we simply pop the first element from the recycled container, store its ID in the variable `l_id` and release the resource that was used by the sound being recycled. The `l_id` argument takes a reference to a `SoundID` that it modifies, which serves as a way to let the outside code know the ID that has been assigned to the sound instance. The new resource that our sound is going to use is then reserved and our sound variable is set to point to the recycled sound instance, which is then set to use a new sound buffer. Our refurbished sound is removed from the recycled container. On the other hand, if a sound that uses the same `sf::SoundBuffer` instance was found, it doesn't need any additional setting up and can simply be returned after its ID is stored and it's erased from the `m_recycled` container.

If there were no recycled sounds available or we had extra space to spare, a new sound is created instead of using a recycled one. The ID of the sound is set to match that of `m_lastID`, which is then incremented (same as `m_numSounds`). After the sound's buffer is set up, it can safely be returned for further processing, such as in the `SetUpSound` method:

[PRE37]

The main idea of this method is simply reducing code wherever possible. It sets up the volume, pitch, minimum distance, attenuation, looping, and relativity of the sound all based on the arguments provided.

Let's wrap this class up with a relatively simple yet commonly used piece of code:

[PRE38]

This method is only responsible for pushing the information provided as arguments into the recycled container for later use.

# Adding support for sound

In order to make our entities emit sounds, some preparations have to be made. For now, we're only going to concern ourselves with simply adding the sound of footsteps whenever a character walks. Doing so requires a slight modification of the `EntityMessage` enumeration in `EntityMessages.h`:

[PRE39]

The highlighted bits are what we're going to be focusing on. `Frame_Change` is a new type of message that's been added in this chapter, and `Direction_Changed` will be used to manipulate the sound listener's direction. In order to detect when a frame changes during the animation process, however, we're going to need to make a few more adjustments to our code base.

## Animation system hooks

In order to have the ability to send out the `Frame_Change` message we've just created, our animation system is going to need a few minor additions, starting with `Anim_Base.h`:

[PRE40]

Here, we're adding a new data member and a method to check if the current frame of an animation has recently been changed. Let's actually integrate this code in `Anim_Base.cpp`:

[PRE41]

In the constructor, it's important to remember to set the newly added data member to a default value, which in this case is `false`. The actual `CheckMoved` method is a very basic chunk of code that returns the value of `m_hasMoved` but sets it to `false` at the same time in order to avoid false positives.

Now that we have an active flag that is going to be used to check for frame changes, all that's missing is simply setting it to `true` in the `SetFrame` method:

[PRE42]

Notice the return value is now a Boolean instead of void. This additional change makes it very easy to do error checking, which is very important for making our last alteration in `Anim_Directional.cpp`:

[PRE43]

The difference here is subtle but relevant. We essentially went from incrementing the current frame by hand by using `m_frameCurrent` to only using the `SetFrame` method.

## Entity component system expansion

With adjustments made previously, we can now put down our last piece of the puzzle in making this work by sending out the `Frame_Change` message in `S_SheetAnimation.cpp`:

[PRE44]

The `Update` method, as you might recall, already handles other types of messages that are related to entities attacking and dying, so this is already gift-wrapped for us. The `CheckMoved` method we added earlier comes in handy and aids us in checking for changes. If there has been a change, the current frame is obtained and stored in the message, which is shortly followed by a `Dispatch` call.

# The sound emitter component

Within the entity component system paradigm, every possible entity parameter or feature is represented as a component. Emitting sounds is definitely one of those features. In order for that to happen, we do have some setting up to do, starting with creating and implementing it in the `C_SoundEmitter.h` header. Before that, however, let's define the types of sounds an entity can have:

[PRE45]

As you can see, we're only going to be working with four types of sound, one of which is going to be implemented in this chapter. A `None` value is also set up in order to make error checking easier.

Every sound that an entity can emit will most likely have different frames it plays during, which calls for a new data structure that encapsulates such information:

[PRE46]

Since sounds are going to be tied to specific frames of animation, we need to define the maximum possible number of frames that can have sounds attached to them. The static constant named `Max_SoundFrames` is used for that purpose here.

The constructor of the `SoundParameters` structure initializes the entire array of frames to a value of -1\. This is going to allow us to check this information in a slightly more efficient way, as the check can be over whenever the first -1 value is encountered. In addition to an array of frame numbers, this structure also stores the name of the sound that is to be emitted.

Now, we can finally begin implementing the sound emitter component:

[PRE47]

First, another static constant is created in order to denote the number of entity sounds that are going to exist. The component itself only has two data members. The first one is a sound ID that will be used for emitting sounds that should not be played repeatedly and have to wait until the previous sound is finished. The second data member is an array of sound parameters for each possible type of entity sound.

Let's begin implementing the component, starting with its constructor:

[PRE48]

Apart from the typical invocation of the `C_Base` constructor with the component type passed in, the sound ID data member is initialized to -1 as well to signify that this component currently is not playing any sounds.

In order for the future sound system to know what sounds to play, we're going to provide a way sound information can be extracted from this component:

[PRE49]

By simply providing one of the enumerated values of `EntitySound` as an argument, outside classes can retrieve information about which sound to play given the circumstances.

Additionally, in order to know if a sound should be played or not, the sound system will need a way to tell if the current frame of animation should be emitting sound or not. This is where the `IsSoundFrame` method comes in:

[PRE50]

If the provided sound argument is larger than the highest supported entity sound ID, `false` is returned. Otherwise, all of the frames for the given sound are iterated over. If a -1 value is encountered, `false` is returned right away. However, if the frame provided as an argument matches a sound frame in the array, this method returns `true`.

Next, we're going to need a few helper methods to set and get certain information:

[PRE51]

Before we get to reading in this component's information from the entity file, let's take a gander at what it might look like. This snippet can be found inside `Player.entity`:

[PRE52]

After the component ID, we're going to be reading in the name of the sound effect to be played, followed by a set of frames delimited by commas. The name of the sound itself is separated from the frame information by a colon. Let's write this:

[PRE53]

After the delimiter information is set up, we iterate once for each possible entity sound and read in the contents of the next segment of the line into a string named `chunk`. If that string is actually empty, we break out of the loop as there's clearly no more information to be loaded. Otherwise, the chunk is split into two parts right at the colon delimiter: `sound` and `frames`. The entity sound is then stored inside the parameters structure.

Lastly, it's necessary to process the frame information, which is delimited by commas. Two local variables are set up to help us with this: `pos` that stores the position of the comma delimiter if one is found and `frameNum` that is used to make sure the `Max_SoundFrames` limit is honored. Inside the `while` loop, the frame delimiter is first located using the `find` method of the `std::string` class. If a delimiter was found, the frame is extracted from the string and converted to an integer, which is stored inside the variable `frame`. That entire segment, including the delimiter, is then erased from the string `frames` and the extracted information is stored inside the parameters structure. In a case where a delimiter wasn't found, however, the loop is stopped right after the frame information has been extracted.

# The sound listener component

In order to properly implement spatial sounds, there has to be a listener within our game world. That listener is, of course, the player of the game. Fortunately, there isn't a lot of information we need to process or store when creating a component for an audio listener:

[PRE54]

Yes, that's it! In its most essential form, this class simply represents a sign that its owner entity should be treated as a listener in the auditory world.

# Implementing the sound system

With both sound emitter and sound listener components out of the way, we have a green light to begin implementing the sound system that is going to bring all of this code to life. Let's get it started!

[PRE55]

Apart from the typical methods that a system is required to implement and a few custom ones, we also have two data members that point to instances of the `AudioManager` and `SoundManager` classes. Let's begin actually implementing the sound system:

[PRE56]

The constructor, predictably enough, sets up two possible versions of the requirement bitmask, both of which require the position component to be present. It then subscribes to the two message types we discussed previously.

Since we're going to need access to both the audio manager and sound manager, a method like this can definitely come in handy:

[PRE57]

Next, let's take a jab at implementing the `Update` method:

[PRE58]

Each entity in this system first has its position and elevation obtained and stored inside a few local variables. It also determines if the current entity is the sound listener or not and stores that information inside a Boolean variable.

If the current entity has a sound emitter component and its sound ID is not equal to -1, it's safe to deduce that the sound is currently still being played. If the current entity is not a sound listener, we attempt to update the sound's position and catch the result of that in an `if` statement. If the position update fails, the sound ID is set back to -1, since it means the sound is no longer active. If the entity is, in fact, a listener, we don't need to update the sound's position at all. Instead, we determine if the sound is still playing or not by calling the `IsPlaying` method.

Afterwards, it's necessary to update the position of the `sf::Listener` class if the current entity has the listener component. Note the use of the `MakeSoundPosition` method here, as well as in the previous chunk of code. It returns a `sf::Vector3f` based on the position and elevation of an entity. We're going to cover this method shortly.

Let's work on handling both of the message types we've discussed previously next:

[PRE59]

In case the entity's direction has changed and it is the sound listener, we obviously need to change the direction of the `sf::Listener` to match the one that is carried inside the message. On the other hand, if we receive a message about a frame changing, the `EmitSound` method is called with the entity ID, sound type, two Boolean flags indicating whether the sound should loop and whether it should be relative to the listener or not, and the current frame the animation is in all passed in as arguments. The sound relativity to the listener in the scene is simply decided by whether the current entity itself is a listener or not.

Positioning sounds in space is also a huge part of this whole system working correctly. Let's take a look at the `MakeSoundPosition` method:

[PRE60]

Due to the default up vector in SFML being the positive *Y* axis, the two dimensional coordinates of an entity position are passed in as X and Z arguments. Meanwhile, the *Y* argument is simply the entity's elevation multiplied by the `Tile_Size` value, found inside the `Map.h` header, which results in entity elevation simulating the height.

Last but definitely not least, we have a chunk of code that is responsible for entities emitting all their sounds that we need to take a look at:

[PRE61]

The first task is obviously checking if the sound system has an entity with the provided ID, and if the entity is a sound emitter. If it is, the sound emitter component is obtained and the sound ID it stores is checked for being equal to -1\. The code still proceeds, however, if an entity is already emitting another sound but the `l_useId` argument is set to `false`, which tells us that a sound should be emitted regardless. Next, the frame passed in as an argument is checked for either being equal to -1, which means the sound should be played regardless, or for it being one of the sound frames defined inside the sound emitter component.

Once we commit to playing the sound, the entity's position component is obtained and used to calculate the position of the sound. If it should be relative to the listener, the position is simply set to be at the absolute zero coordinate of all axes.

If we want to only keep a single instance of a particular sound, the `Play` method of the sound manager is invoked within the `SetSoundID` argument list of the sound emitter component to catch the returned ID. It only has two arguments passed in, as the other two Boolean flags hold the default values of `false`. Otherwise, if this particular sound should be played irrespective of whether the entity is already emitting another sound or not, the `Play` method of our sound manager is called by itself and the Boolean flag for sound being relative to the listener is passed in as the last argument.

# Integrating our code

In order to prevent sounds or music from playing at inappropriate times, our state manager must notify the sound manager of any state changes:

[PRE62]

Since the sound manager also cares about states being removed, let's tell it when that happens:

[PRE63]

The only thing we have left to do now is actually integrating everything we worked on into the rest of our code base, starting with `SharedContext.h`:

[PRE64]

Next, instantiating and managing these two new classes inside the shared context is of utmost importance. Let's start by modifying the `Game.h` header:

[PRE65]

As always, we keep these manager classes inside `Game` in order to manage their lifetime properly. For some of them, however, merely existing isn't enough. They require to be set up like this:

[PRE66]

After both classes are created, their addresses are passed to the shared context. One more important detail that's easy to overlook is actually setting up the sound system at this point. It needs to have access to both the audio and the sound manager.

Let's not forget to also update the sound manager properly during the flow of the entire application:

[PRE67]

With the creation of new components and systems comes the responsibility of making sure they can actually be created automatically, by adding the component types to the entity manager:

[PRE68]

Our sound system also needs to be created inside the system manager:

[PRE69]

Having all of that done, we can finally add some music to our game! Let's start by making sure we have an intro soundtrack by modifying `State_Intro.cpp`:

[PRE70]

Also, it would be nice to have some background music during actual game-play, so let's modify `State_Game.cpp` as follows:

[PRE71]

And voila! Just like that, we now have music and dynamic sound effects baked into our RPG!

# Summary

With possibilities ranging anywhere from simple ambiance to complex musical scores tugging at the heart strings of the player, our game world starts to develop a sense of character and presence. All of the hard work we put in towards making sure our project isn't mute adds up to yet another major leap in the direction of quality. However, as we begin to approach the end of this book with only two chapters remaining, the most challenging part is still yet to come.

In the next chapter, we will be exploring the vast world of networking and how it can help us turn our lonely, quiet RPG into a battle zone of multiple other players. See you there!