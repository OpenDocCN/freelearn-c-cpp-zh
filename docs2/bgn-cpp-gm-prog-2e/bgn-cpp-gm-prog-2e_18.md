# 第十八章：*第十七章*：声音空间化和 HUD

在本章中，我们将添加所有音效和 HUD。我们已经在之前的两个项目中这样做过，但这次我们将有所不同。我们将探讨声音**空间化**的概念以及 SFML 如何使这个原本复杂的概念变得简单易行。此外，我们还将构建一个 HUD 类来封装将信息绘制到屏幕上的代码。

我们将按照以下顺序完成这些任务。

+   什么是空间化？

+   SFML 如何处理空间化

+   构建 SoundManager 类

+   部署发射器

+   使用 SoundManager 类

+   构建一个`HUD`类

+   使用`HUD`类

# 什么是空间化？

**空间化**是将某物与它所包含的空间或其中的空间相关联的行为。在我们的日常生活中，自然世界中的所有事物默认都是空间化的。如果一辆摩托车从左到右呼啸而过，我们将会听到声音从一侧的微弱到另一侧的响亮。当它经过时，它会在另一只耳朵中变得更加突出，然后再逐渐消失在远处。如果我们某天早上醒来，发现世界不再空间化，那将会非常奇怪。

如果我们能让我们的视频游戏更接近现实世界，我们的玩家可以更加沉浸其中。如果玩家能在远处听到僵尸微弱的声音，而他们的非人类尖叫随着他们越来越近而变得越来越大，我们的僵尸游戏会更有趣。

很可能很明显，空间化的数学将是复杂的。我们如何根据玩家（声音的听者）到发出声音的对象（发射器）的距离和方向来计算特定扬声器中给定声音的响度？

幸运的是，SFML 为我们处理了所有复杂的过程。我们只需要熟悉一些技术术语，然后我们就可以开始使用 SFML 来空间化我们的音效。

## 发射器、衰减和听者

为了给 SFML 提供它完成工作所需的信息，我们需要了解一些信息。我们需要知道声音在我们的游戏世界中是从哪里发出的。这个声音的来源被称为**发射器**。在游戏中，发射器可以是僵尸、车辆，或者在我们当前的项目中，是一个火砖。我们已经一直在跟踪我们游戏中对象的位置，因此向 SFML 提供发射器的位置将非常直接。

我们需要关注的下一个因素是**衰减**。衰减是波衰减的速度。你可以简化这个陈述，并使其具体到声音，即衰减是声音减少音量的速度。这从技术上讲并不准确，但对于本章和我们的游戏来说，这是一个足够好的描述。

我们需要考虑的最后一个因素是**监听器**。当 SFML 空间化声音时，它是相对于什么进行空间化的；游戏的“耳朵”在哪里？在大多数游戏中，合乎逻辑的做法是使用玩家角色。在我们的游戏中，我们将使用托马斯（我们的玩家角色）。

# 使用 SFML 处理空间化

SFML 有几个函数允许我们处理发射器、衰减和监听器。让我们假设地看看它们，然后我们将编写一些代码来真正地将空间化声音添加到我们的项目中。

我们可以设置一个准备播放的声音效果，就像我们经常做的那样，如下所示：

```cpp
// Declare SoundBuffer in the usual way
SoundBuffer zombieBuffer;
// Declare a Sound object as-per-usual
Sound zombieSound;
// Load the sound from a file like we have done so often
zombieBuffer.loadFromFile("sound/zombie_growl.wav");
// Associate the Sound object with the Buffer
zombieSound.setBuffer(zombieBuffer);
```

我们可以使用以下代码中显示的 `setPosition` 函数来设置发射器的位置：

```cpp
// Set the horizontal and vertical positions of the emitter
// In this case the emitter is a zombie
// In the Zombie Arena project we could have used 
// getPosition().x and getPosition().y
// These values are arbitrary
float x = 500;
float y = 500;
zombieSound.setPosition(x, y, 0.0f);
```

如前一段代码的注释中所建议的，我们如何确切地获取发射器的坐标可能会取决于游戏类型。正如前一段代码所示，在僵尸竞技场项目中，这将会非常简单。在我们这个项目中设置位置时，我们将面临一些挑战。

我们可以设置衰减级别如下：

```cpp
zombieSound.setAttenuation(15);
```

实际的衰减级别可能有点模糊。我们希望玩家获得的效果可能与基于衰减的准确科学公式所使用的减少距离上的音量不同。获得正确的衰减级别通常是通过实验来实现的。衰减级别越高，音量降低到静音的速度就越快。

此外，我们可能还想设置一个围绕发射器的区域，其中音量不会衰减。如果我们认为这个特性在某个范围之外不合适，或者如果我们有多个声音源并且不想“过度”使用这个特性，我们可能会这样做。为此，我们可以使用如下所示的 `setMinimumDistance` 函数：

```cpp
zombieSound.setMinDistance(150);
```

使用上一行代码，衰减只有在监听器距离发射器 150 像素/单位时才会计算。

SFML 库中还有一些其他有用的函数，包括 `setLoop` 函数。当将 `true` 作为参数传入时，此函数会告诉 SFML 无限次地播放声音，如下面的代码所示：

```cpp
zombieSound.setLoop(true);
```

声音将继续播放，直到我们使用以下代码结束它：

```cpp
zombieSound.stop();
```

有时，我们想知道声音的状态（播放或停止）。我们可以通过以下代码中的 `getStatus` 函数来实现，如下所示：

```cpp
if (zombieSound.getStatus() == Sound::Status::Stopped)
{
    // The sound is NOT playing
    // Take whatever action here
}
if (zombieSound.getStatus() == Sound::Status::Playing)
{
    // The sound IS playing
    // Take whatever action here
}
```

使用 SFML 进行声音空间化时，我们需要覆盖的最后一个方面是监听器。监听器在哪里？我们可以使用以下代码设置监听器的位置：

```cpp
// Where is the listener? 
// How we get the values of x and y varies depending upon the game
// In the Zombie Arena game or the Thomas Was Late game
// We can use getPosition()
Listener::setPosition(m_Thomas.getPosition().x, 
    m_Thomas.getPosition().y, 0.0f);
```

之前的代码将使所有声音相对于该位置播放。这正是我们需要的，用于远处火砖或即将到来的僵尸的咆哮声，但对于像跳跃这样的常规声音效果，这是一个问题。我们可以开始处理玩家的位置发射器，但 SFML 为我们简化了这些事情。每次我们想要播放一个“正常”的声音时，我们只需像以下代码所示调用`setRelativeToListener`，然后以我们迄今为止相同的方式播放声音。以下是我们可能播放的“正常”非空间化跳跃声音效果的方式：

```cpp
jumpSound.setRelativeToListener(true);
jumpSound.play();
```

我们需要做的只是在我们播放任何空间化声音之前再次调用`Listener::setPosition`。

我们现在拥有丰富的 SFML 声音函数，我们准备好真正制作一些空间化噪音了。

# 构建 SoundManager 类

你可能还记得上一个项目中，所有的声音代码占据了相当多的代码行。现在，考虑到空间化，它还将变得更长。为了保持我们的代码可管理，我们将编写一个类来管理所有正在播放的声音效果。此外，为了帮助我们进行空间化，我们还将向`Engine`类添加一个函数，但我们将稍后在本章中讨论这一点。

## 编写 SoundManager.h

让我们开始编写和检查头文件。

右键点击`SoundManager.h`。最后，点击`SoundManager`类。

添加并检查以下代码：

```cpp
#pragma once
#include <SFML/Audio.hpp>
using namespace sf;
class SoundManager
{
    private:
        // The buffers
        SoundBuffer m_FireBuffer;
        SoundBuffer m_FallInFireBuffer;
        SoundBuffer m_FallInWaterBuffer;
        SoundBuffer m_JumpBuffer;
        SoundBuffer m_ReachGoalBuffer;
        // The Sounds
        Sound m_Fire1Sound;
        Sound m_Fire2Sound;
        Sound m_Fire3Sound;
        Sound m_FallInFireSound;
        Sound m_FallInWaterSound;
        Sound m_JumpSound;
        Sound m_ReachGoalSound;
        // Which sound should we use next, fire 1, 2 or 3
        int m_NextSound = 1;
    public:
        SoundManager();
        void playFire(Vector2f emitterLocation, 
            Vector2f listenerLocation);
        void playFallInFire();
        void playFallInWater();
        void playJump();
        void playReachGoal();
};
```

我们刚刚添加的代码中没有什么复杂的。有五个`SoundBuffer`对象和八个`Sound`对象。其中三个`Sound`对象将播放相同的`SoundBuffer`。这解释了为什么`Sound`/`SoundBuffer`对象的数量不同。我们这样做是为了能够同时播放多个具有不同空间化参数的咆哮声音效果。

注意`m_NextSound`变量，它将帮助我们跟踪下一次应该使用这些同时播放的声音中的哪一个。

有一个构造函数`SoundManager`，我们将设置所有声音效果，并且有五个函数将播放声音效果。其中四个函数简单地播放“正常”声音效果，它们的代码会更简单。

其中一个函数`playFire`将处理空间化声音效果，并且会稍微深入一些。注意`playFire`函数的参数。它接收一个`Vector2f`，这是发射器的位置，以及第二个`Vector2f`，这是听者的位置。

## 编写 SoundManager.cpp 文件

现在，我们可以编写函数定义。构造函数和`playFire`函数有大量的代码，所以我们将单独查看它们。其他函数都很短小精悍，所以我们将一次性处理它们。

右键点击`SoundManager.cpp`。最后，点击`SoundManager`类的`.cpp`文件。

### 编写构造函数

将以下代码添加到`SoundManager.cpp`的包含指令和构造函数中：

```cpp
#include "SoundManager.h"
#include <SFML/Audio.hpp>
using namespace sf;
SoundManager::SoundManager()
{
    // Load the sound in to the buffers
    m_FireBuffer.loadFromFile("sound/fire1.wav");
    m_FallInFireBuffer.loadFromFile("sound/fallinfire.wav");
    m_FallInWaterBuffer.loadFromFile("sound/fallinwater.wav");
    m_JumpBuffer.loadFromFile("sound/jump.wav");
    m_ReachGoalBuffer.loadFromFile("sound/reachgoal.wav");
    // Associate the sounds with the buffers
    m_Fire1Sound.setBuffer(m_FireBuffer);
    m_Fire2Sound.setBuffer(m_FireBuffer);
    m_Fire3Sound.setBuffer(m_FireBuffer);
    m_FallInFireSound.setBuffer(m_FallInFireBuffer);
    m_FallInWaterSound.setBuffer(m_FallInWaterBuffer);
    m_JumpSound.setBuffer(m_JumpBuffer);
    m_ReachGoalSound.setBuffer(m_ReachGoalBuffer);

    // When the player is 50 pixels away sound is full volume
    float minDistance = 150;
    // The sound reduces steadily as the player moves further away
    float attenuation = 15;
    // Set all the attenuation levels
    m_Fire1Sound.setAttenuation(attenuation);
    m_Fire2Sound.setAttenuation(attenuation);
    m_Fire3Sound.setAttenuation(attenuation);
    // Set all the minimum distance levels
    m_Fire1Sound.setMinDistance(minDistance);
    m_Fire2Sound.setMinDistance(minDistance);
    m_Fire3Sound.setMinDistance(minDistance);
    // Loop all the fire sounds
    // when they are played
    m_Fire1Sound.setLoop(true);
    m_Fire2Sound.setLoop(true);
    m_Fire3Sound.setLoop(true);
}
```

在前面的代码中，我们将五个声音文件加载到五个`SoundBuffer`对象中。接下来，我们将八个`Sound`对象与一个`SoundBuffer`对象关联起来。请注意，`m_Fire1Sound`、`m_Fire2Sound`和`m_Fire3Sound`都将从同一个`SoundBuffer`，即`m_FireBuffer`中播放。

接下来，我们设置了三个火焰声音的衰减和最小距离。

小贴士

`150`和`15`的值是通过实验得到的。一旦游戏开始运行，建议通过更改这些值来实验，看看（或者更确切地说，听听）它们之间的差异。

最后，对于构造函数，我们在每个与火焰相关的`Sound`对象上使用`setLoop`函数。现在，当我们调用`play`时，它们将连续播放。

### 编写`playFire`函数

添加`playFire`函数如下。然后，我们可以讨论它：

```cpp
void SoundManager::playFire(
    Vector2f emitterLocation, Vector2f listenerLocation)
{
    // Where is the listener? Thomas.
    Listener::setPosition(listenerLocation.x, 
        listenerLocation.y, 0.0f);
    switch(m_NextSound)
    {
    case 1:
        // Locate/move the source of the sound
        m_Fire1Sound.setPosition(emitterLocation.x, 
            emitterLocation.y, 0.0f);
        if (m_Fire1Sound.getStatus() == Sound::Status::Stopped)
        {
            // Play the sound, if its not already
            m_Fire1Sound.play();
        }
        break;
    case 2:
        // Do the same as previous for the second sound
        m_Fire2Sound.setPosition(emitterLocation.x, 
            emitterLocation.y, 0.0f);
        if (m_Fire2Sound.getStatus() == Sound::Status::Stopped)
        {
            m_Fire2Sound.play();
        }
        break;
    case 3:
        // Do the same as previous for the third sound
        m_Fire3Sound.setPosition(emitterLocation.x, 
            emitterLocation.y, 0.0f);
        if (m_Fire3Sound.getStatus() == Sound::Status::Stopped)
        {
            m_Fire3Sound.play();
        }
        break;
    }
    // Increment to the next fire sound
    m_NextSound++;
    // Go back to 1 when the third sound has been started
    if (m_NextSound > 3)
    {
        m_NextSound = 1;
    }
}
```

我们首先调用`Listener::setPosition`并根据传入的参数`Vector2f`设置听者的位置。

接下来，代码进入一个`switch`块，测试`m_NextSound`的值。每个`case`语句都做完全相同的事情，但针对`m_Fire1Sound`、`m_Fire2Sound`或`m_Fire3Sound`。

在每个`case`块中，我们使用`setPosition`函数和传入的参数设置发射器的位置。每个`case`块中的代码的下一部分检查声音是否当前已停止，如果是，则播放声音。很快，我们将看到如何得到传递给此函数的发射器和听者的位置。

`playFire`函数的最后部分增加`m_NextSound`，并确保它只能等于 1、2 或 3，这是`switch`块所要求的。

### 编写 SoundManager 的其他函数

添加以下四个简单函数：

```cpp
void SoundManager::playFallInFire()
{
    m_FallInFireSound.setRelativeToListener(true);
    m_FallInFireSound.play();
}
void SoundManager::playFallInWater()
{
    m_FallInWaterSound.setRelativeToListener(true);
    m_FallInWaterSound.play();
}
void SoundManager::playJump()
{
    m_JumpSound.setRelativeToListener(true);
    m_JumpSound.play();
}
void SoundManager::playReachGoal()
{
    m_ReachGoalSound.setRelativeToListener(true);
    m_ReachGoalSound.play();
}
```

`playFallInFire`、`playFallInWater`和`playReachGoal`函数只做两件事。首先，它们各自调用`setRelativeToListener`，以便声音效果不是空间化的，使声音效果“正常”，而不是方向性的，然后它们在适当的`Sound`对象上调用`play`。

这样，`SoundManager`类就完成了。现在，我们可以在`Engine`类中使用它。

# 将 SoundManager 添加到游戏引擎中

打开`Engine.h`文件，并添加一个新的`SoundManager`类实例，如下所示的高亮代码：

```cpp
#pragma once
#include <SFML/Graphics.hpp>
#include "TextureHolder.h"
#include "Thomas.h"
#include "Bob.h"
#include "LevelManager.h"
#include "SoundManager.h"
using namespace sf;
class Engine
{
private:
    // The texture holder
    TextureHolder th;
    // Thomas and his friend, Bob
    Thomas m_Thomas;
    Bob m_Bob;
    // A class to manage all the levels
    LevelManager m_LM;
    // Create a SoundManager
    SoundManager m_SM;
    const int TILE_SIZE = 50;
    const int VERTS_IN_QUAD = 4;
```

到目前为止，我们可以使用`m_SM`调用各种`play...`函数。不幸的是，为了管理发射器（火焰瓷砖）的位置，我们还需要做一些额外的工作。

# 填充声音发射器

打开`Engine.h`文件，并添加一个`populateEmitters`函数的新原型和一个新的 STL `vector`，包含`Vector2f`对象：

```cpp
    ...
    ...
    ...
    // Run will call all the private functions
    bool detectCollisions(PlayableCharacter& character);
    // Make a vector of the best places to emit sounds from
    void populateEmitters(vector <Vector2f>& vSoundEmitters,
        int** arrayLevel);
    // A vector of Vector2f for the fire emitter locations
    vector <Vector2f> m_FireEmitters;

public:
    ...
    ...
    ...
```

`populateEmitters`函数接受一个`Vector2f`对象的`vector`作为参数，以及一个指向`int`指针的指针（一个二维数组）。`vector`将保存每个发射器在关卡中的位置。数组是包含关卡布局的二维数组。

## 编写 populateEmitters 函数

`populateEmitters`函数的职责是遍历`arrayLevel`数组中的所有元素，并决定将发射器放置在哪里。它将结果存储在`m_FireEmitters`中。

右键点击`PopulateEmitters.cpp`。最后，点击`populateEmitters`。

将代码完整地添加进去。确保你在添加代码的同时仔细研究代码，这样我们就可以讨论它：

```cpp
#include "Engine.h"
using namespace sf;
using namespace std;
void Engine::populateEmitters(
    vector <Vector2f>& vSoundEmitters, 
   int** arrayLevel)
{
    // Make sure the vector is empty
    vSoundEmitters.empty();
    // Keep track of the previous emitter
    // so we don't make too many
    FloatRect previousEmitter;
    // Search for fire in the level
    for (int x = 0; x < (int)m_LM.getLevelSize().x; x++)
    {
        for (int y = 0; y < (int)m_LM.getLevelSize().y; y++)
        {
            if (arrayLevel[y][x] == 2)// fire is present
            {
                // Skip over any fire tiles too 
                // near a previous emitter
                if (!FloatRect(x * TILE_SIZE,
                    y * TILE_SIZE,
                    TILE_SIZE,
                    TILE_SIZE).intersects(previousEmitter))
                {
                    // Add the coordinates of this water block
                    vSoundEmitters.push_back(
                        Vector2f(x * TILE_SIZE, y * TILE_SIZE));
                    // Make a rectangle 6 blocks x 6 blocks,
                    // so we don't make any more emitters 
                    // too close to this one
                    previousEmitter.left = x * TILE_SIZE;
                    previousEmitter.top = y * TILE_SIZE;
                    previousEmitter.width = TILE_SIZE * 6;
                    previousEmitter.height = TILE_SIZE * 6;
                }
            }
        }
    }
    return;
}
```

一些代码可能乍一看很复杂。理解我们用来选择发射器位置的技巧会使它变得简单。在我们的关卡中，有大量的火砖块。例如，在一个关卡中，有超过 30 块火砖聚集在一起。代码确保在给定的矩形内只有一个发射器。这个矩形存储在`previousEmitter`中，大小为 300 像素乘 300 像素（`TILE_SIZE * 6`）。

代码设置了一个嵌套的`for`循环，遍历`arrayLevel`，寻找火砖。当找到火砖时，它会确保它不与`previousEmitter`相交。只有在确保不相交后，它才会使用`pushBack`函数向`vSoundEmitters`添加另一个发射器。之后，它还会更新`previousEmitter`以避免出现大量声音发射器的聚集。

让我们制造一些噪音。

# 播放声音

打开`LoadLevel.cpp`文件，并添加对新的`populateEmitters`函数的调用，如下面的代码所示：

```cpp
void Engine::loadLevel()
{
    m_Playing = false;
    // Delete the previously allocated memory
    for (int i = 0; i < m_LM.getLevelSize().y; ++i)
    {
        delete[] m_ArrayLevel[i];
    }
    delete[] m_ArrayLevel;
    // Load the next 2d array with the map for the level
    // And repopulate the vertex array as well
    m_ArrayLevel = m_LM.nextLevel(m_VALevel);
    // Prepare the sound emitters
    populateEmitters(m_FireEmitters, m_ArrayLevel);
    // How long is this new time limit
    m_TimeRemaining = m_LM.getTimeLimit();
    // Spawn Thomas and Bob
    m_Thomas.spawn(m_LM.getStartPosition(), GRAVITY);
    m_Bob.spawn(m_LM.getStartPosition(), GRAVITY);
    // Make sure this code isn't run again
    m_NewLevelRequired = false;
}
```

首先要添加的声音是跳跃声音。我们记得键盘处理代码位于`Bob`和`Thomas`类中的纯虚函数内，并且当成功发起跳跃时，`handleInput`函数返回`true`。

打开`Input.cpp`文件，并添加以下高亮代码行以在托马斯或鲍勃成功开始跳跃时播放跳跃声音：

```cpp
// Handle input specific to Thomas
if (m_Thomas.handleInput())
{
    // Play a jump sound
    m_SM.playJump();
}
// Handle input specific to Bob
if (m_Bob.handleInput())
{
    // Play a jump sound
    m_SM.playJump();
}
```

打开`Update.cpp`文件，并添加以下高亮代码行以在托马斯和鲍勃同时达到当前关卡目标时播放成功声音：

```cpp
// Detect collisions and see if characters have reached the goal tile
// The second part of the if condition is only executed
// when Thomas is touching the home tile
if (detectCollisions(m_Thomas) && detectCollisions(m_Bob))
{
    // New level required
    m_NewLevelRequired = true;
    // Play the reach goal sound
    m_SM.playReachGoal();
}
else
{
    // Run Bobs collision detection
    detectCollisions(m_Bob);
}
```

此外，在`Update.cpp`文件中，我们将添加代码来遍历`m_FireEmitters`向量，并决定何时调用`SoundManager`类的`playFire`函数。

仔细观察新高亮代码周围的小部分上下文。在正确的位置添加此代码是至关重要的：

```cpp
}// End if playing
// Check if a fire sound needs to be played
vector<Vector2f>::iterator it;
// Iterate through the vector of Vector2f objects
for (it = m_FireEmitters.begin(); it != m_FireEmitters.end(); it++)
{
    // Where is this emitter?
    // Store the location in pos
    float posX = (*it).x;
    float posY = (*it).y;
    // is the emitter near the player?
    // Make a 500 pixel rectangle around the emitter
    FloatRect localRect(posX - 250, posY - 250, 500, 500);
    // Is the player inside localRect?
    if (m_Thomas.getPosition().intersects(localRect))
    {
        // Play the sound and pass in the location as well
        m_SM.playFire(Vector2f(posX, posY), m_Thomas.getCenter());
    }
}

// Set the appropriate view around the appropriate character
```

上述代码有点类似于声音的碰撞检测。每当托马斯进入围绕火发射器的一个 500 像素乘 500 像素的矩形内时，就会调用`playFire`函数，并将发射器和托马斯的坐标传递给它。`playFire`函数完成剩余的工作并播放一个空间化的循环声音效果。

打开 `DetectCollisions.cpp` 文件，找到合适的位置，并添加以下突出显示的代码。这两行突出显示的代码会在任意一个角色掉入水或火焰方块时触发声音效果：

```cpp
// Has character been burnt or drowned?
// Use head as this allows him to sink a bit
if (m_ArrayLevel[y][x] == 2 || m_ArrayLevel[y][x] == 3)
{
    if (character.getHead().intersects(block))
    {
        character.spawn(m_LM.getStartPosition(), GRAVITY);
        // Which sound should be played?
        if (m_ArrayLevel[y][x] == 2)// Fire, ouch!
        {
            // Play a sound
            m_SM.playFallInFire();
        }
        else // Water
        {
            // Play a sound
            m_SM.playFallInWater();
        }
    }
}
```

现在玩游戏将允许你在靠近火焰方块时听到所有声音，包括酷炫的空间化效果。

# 实现 HUD 类

HUD 非常简单，与僵尸竞技场项目相比并没有什么不同。我们将要做的是将所有代码封装在一个新的 `HUD` 类中。如果我们将所有 `Font`、`Text` 和其他变量声明为这个新类的成员，我们就可以在构造函数中初始化它们，并为它们的值提供获取函数。这将使 `Engine` 类免于大量的声明和初始化。

## 编写 HUD.h

首先，我们将使用所有成员变量和函数声明来编写 `HUD.h` 文件。右键点击 `HUD.h`。最后，点击 `HUD` 类。

将以下代码添加到 `HUD.h` 文件中：

```cpp
#pragma once
#include <SFML/Graphics.hpp>
using namespace sf;
class Hud
{
private:
    Font m_Font;
    Text m_StartText;
    Text m_TimeText;
    Text m_LevelText;
public:
    Hud();
    Text getMessage();
    Text getLevel();
    Text getTime();
    void setLevel(String text);
    void setTime(String text);
};
```

在前面的代码中，我们添加了一个 `Font` 实例和三个 `Text` 实例。`Text` 对象将用于显示提示用户开始、剩余时间和当前关卡编号的消息。

公共函数更有趣。首先，是构造函数，大部分代码将在这里编写。构造函数将初始化 `Font` 和 `Text` 对象，并将它们相对于当前屏幕分辨率定位在屏幕上。

三个获取函数 `getMessage`、`getLevel` 和 `getTime` 将返回一个 `Text` 对象给调用代码，以便它可以将它们绘制到屏幕上。

`setLevel` 和 `setTime` 函数将分别用于更新显示在 `m_LevelText` 和 `m_TimeText` 中的文本。

现在，我们可以编写我们刚刚声明的所有函数的定义。

## 编写 HUD.cpp 文件

右键点击 `HUD.cpp`。最后，点击 `HUD` 类的 `.cpp` 文件。

添加包含指令和以下代码。然后，我们将讨论它：

```cpp
#include "Hud.h"
Hud::Hud()
{
    Vector2u resolution;
    resolution.x = VideoMode::getDesktopMode().width;
    resolution.y = VideoMode::getDesktopMode().height;
    // Load the font
    m_Font.loadFromFile("fonts/Roboto-Light.ttf");
    // when Paused
    m_StartText.setFont(m_Font);
    m_StartText.setCharacterSize(100);
    m_StartText.setFillColor(Color::White);
    m_StartText.setString("Press Enter when ready!");
    // Position the text
    FloatRect textRect = m_StartText.getLocalBounds();
    m_StartText.setOrigin(textRect.left +
        textRect.width / 2.0f,
        textRect.top +
        textRect.height / 2.0f);
    m_StartText.setPosition(
        resolution.x / 2.0f, resolution.y / 2.0f);
    // Time
    m_TimeText.setFont(m_Font);
    m_TimeText.setCharacterSize(75);
    m_TimeText.setFillColor(Color::White);
    m_TimeText.setPosition(resolution.x - 150, 0);
    m_TimeText.setString("------");
    // Level
    m_LevelText.setFont(m_Font);
    m_LevelText.setCharacterSize(75);
    m_LevelText.setFillColor(Color::White);
    m_LevelText.setPosition(25, 0);
    m_LevelText.setString("1");
}
```

首先，我们将水平和垂直分辨率存储在一个名为 `resolution` 的 `Vector2u` 中。接下来，我们从添加回的 `fonts` 目录中加载字体，这是在 *第十四章*，*抽象和代码管理 – 更好地使用面向对象编程* 中提到的。

接下来的四行代码设置了 `m_StartText` 的字体、颜色、大小和文本。在这段代码之后，它捕获了包裹 `m_StartText` 的矩形的尺寸，并执行计算以确定如何在屏幕上将其居中。如果你想要对这个代码部分的更详细解释，请参考 *第三章*，*C++ 字符串和 SFML 时间 – 玩家输入和 HUD*。

在构造函数的最后两个代码块中，设置了 `m_TimeText` 和 `m_LevelText` 的字体、文本大小、颜色、位置和实际文本。稍后我们将看到，这两个 `Text` 对象将通过两个设置函数进行更新，当需要时。

立即将以下获取和设置函数添加到我们刚刚添加的代码下方：

```cpp
Text Hud::getMessage()
{
    return m_StartText;
}
Text Hud::getLevel()
{
    return m_LevelText;
}
Text Hud::getTime()
{
    return m_TimeText;
}
void Hud::setLevel(String text)
{
    m_LevelText.setString(text);
}
void Hud::setTime(String text)
{
    m_TimeText.setString(text);
}
```

上一段代码中的前三个函数简单地返回适当的 `Text` 对象，即 `m_StartText`、`m_LevelText` 或 `m_TimeText`。我们将在将 HUD 绘制到屏幕上时不久使用这些函数。最后的两个函数，`setLevel` 和 `setTime`，使用 `setString` 函数来更新适当的 `Text` 对象，该对象将使用从 `Engine` 类的 `update` 函数传递的值，每 500 帧。

完成所有这些后，我们可以在我们的游戏引擎中使用 HUD 类。

# 使用 HUD 类

打开 `Engine.h` 文件，为我们的新类添加一个包含语句，声明一个新 `HUD` 类的实例，并声明并初始化两个新的成员变量，这两个变量将跟踪我们更新 HUD 的频率。正如我们在前面的项目中所学到的，我们不需要每帧都更新 HUD。

将以下突出显示的代码添加到 `Engine.h`：

```cpp
#pragma once
#include <SFML/Graphics.hpp>
#include "TextureHolder.h"
#include "Thomas.h"
#include "Bob.h"
#include "LevelManager.h"
#include "SoundManager.h"
#include "HUD.h"
using namespace sf;
class Engine
{
private:
    // The texture holder
    TextureHolder th;
    // Thomas and his friend, Bob
    Thomas m_Thomas;
    Bob m_Bob;
    // A class to manage all the levels
    LevelManager m_LM;
    // Create a SoundManager
    SoundManager m_SM;
    // The Hud
    Hud m_Hud;
    int m_FramesSinceLastHUDUpdate = 0;
    int m_TargetFramesPerHUDUpdate = 500;
    const int TILE_SIZE = 50;
```

接下来，我们需要向 `Engine` 类的 `update` 函数中添加一些代码。打开 `Update.cpp` 文件，并添加以下突出显示的代码以每 500 帧更新一次 HUD：

```cpp
    // Set the appropriate view around the appropriate character
    if (m_SplitScreen)
    {
        m_LeftView.setCenter(m_Thomas.getCenter());
        m_RightView.setCenter(m_Bob.getCenter());
    }
    else
    {
        // Centre full screen around appropriate character
        if (m_Character1)
        {
            m_MainView.setCenter(m_Thomas.getCenter());
        }
        else
        {
            m_MainView.setCenter(m_Bob.getCenter());
        }
    }
    // Time to update the HUD?
// Increment the number of frames since 
   // the last HUD calculation
    m_FramesSinceLastHUDUpdate++;
    // Update the HUD every m_TargetFramesPerHUDUpdate frames
    if (m_FramesSinceLastHUDUpdate > m_TargetFramesPerHUDUpdate)
    {
        // Update game HUD text
        stringstream ssTime;
        stringstream ssLevel;
        // Update the time text
        ssTime << (int)m_TimeRemaining;
        m_Hud.setTime(ssTime.str());
        // Update the level text
        ssLevel << "Level:" << m_LM.getCurrentLevel();
        m_Hud.setLevel(ssLevel.str());
        m_FramesSinceLastHUDUpdate = 0;
    }
}// End of update function
```

在上述代码中，`m_FramesSinceLastUpdate` 在每帧都会增加。当 `m_FramesSinceLastUpdate` 超过 `m_TargetFramesPerHUDUpdate` 时，执行进入 `if` 块。在 `if` 块内部，我们使用 `stringstream` 对象来更新我们的 `Text`，就像我们在前面的项目中做的那样。在这个项目中，我们使用 `HUD` 类，因此我们通过传递当前 `Text` 对象需要设置的值来调用 `setTime` 和 `setLevel` 函数。

`if` 块中的最后一步是将 `m_FramesSinceLastUpdate` 设置回零，以便它可以开始计算下一次更新。

最后，打开 `Draw.cpp` 文件，并添加以下突出显示的代码以每帧绘制 HUD：

```cpp
    else
    {
        // Split-screen view is active
        // First draw Thomas' side of the screen
        // Switch to background view
        m_Window.setView(m_BGLeftView);
        // Draw the background
        m_Window.draw(m_BackgroundSprite);
        // Switch to m_LeftView
        m_Window.setView(m_LeftView);
        // Draw the Level
        m_Window.draw(m_VALevel, &m_TextureTiles);

        // Draw thomas
        m_Window.draw(m_Bob.getSprite());
        // Draw thomas
        m_Window.draw(m_Thomas.getSprite());

        // Now draw Bob's side of the screen
        // Switch to background view
        m_Window.setView(m_BGRightView);
        // Draw the background
        m_Window.draw(m_BackgroundSprite);
        // Switch to m_RightView
        m_Window.setView(m_RightView);
        // Draw the Level
        m_Window.draw(m_VALevel, &m_TextureTiles);
        // Draw thomas
        m_Window.draw(m_Thomas.getSprite());
        // Draw bob
        m_Window.draw(m_Bob.getSprite());

    }
    // Draw the HUD
    // Switch to m_HudView
    m_Window.setView(m_HudView);
    m_Window.draw(m_Hud.getLevel());
    m_Window.draw(m_Hud.getTime());
    if (!m_Playing)
    {
        m_Window.draw(m_Hud.getMessage());
    }
    // Show everything we have just drawn
    m_Window.display();
}// End of draw
```

上述代码通过使用 HUD 类的获取函数来绘制 HUD。请注意，调用绘制提示玩家开始游戏的信息的代码仅在游戏当前未在播放时使用 `(!m_Playing)`。

运行游戏并玩几个关卡，以查看时间逐渐减少，关卡逐渐增加。当你再次回到关卡 1 时，请注意你比之前少了 10% 的时间。

# 摘要

在本章中，我们探讨了声音空间化。我们的 "Thomas Was Late" 游戏现在不仅完全可玩，我们还添加了方向性音效和一个简单但信息丰富的 HUD。我们也可以轻松添加新关卡。到目前为止，我们可以称之为完成了。

希望能增加一些亮点。在下一章中，我们将探讨两个游戏概念。首先，我们将研究粒子系统，这是如何处理爆炸或其他特殊效果的方法。为了实现这一点，我们需要学习更多关于 C++的知识。因此，多重继承这一主题将被引入。

之后，当我们学习 OpenGL 和可编程图形管线时，我们将为游戏增添最后的点缀。那时，我们将能够尝试接触**GLSL**语言，它允许我们编写直接在 GPU 上执行代码，从而创建一些特殊效果。
