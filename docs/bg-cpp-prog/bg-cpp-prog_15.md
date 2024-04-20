# 第十五章：声音空间化和 HUD

在本章中，我们将添加所有的音效和 HUD。我们在之前的两个项目中都做过这个，但这次我们会以稍微不同的方式来做。我们将探讨声音空间化的概念以及 SFML 如何使这个本来复杂的概念变得简单易行；此外，我们将构建一个 HUD 类来封装将信息绘制到屏幕上的代码。

我们将按照以下顺序完成这些任务：

+   什么是空间化？

+   SFML 如何处理空间化

+   构建一个`SoundManager`类

+   部署发射器

+   使用`SoundManager`类

+   构建一个`HUD`类

+   使用`HUD`类

# 什么是空间化？

空间化是使某物相对于其所在的空间或内部的行为。在我们的日常生活中，自然界中的一切默认都是空间化的。如果一辆摩托车从左到右呼啸而过，我们会听到声音从一侧变得微弱到响亮，当它经过时，它会在另一只耳朵中变得更加显著，然后再次消失在远处。如果有一天早上醒来，世界不再是空间化的，那将是异常奇怪的。

如果我们能让我们的视频游戏更像现实世界，我们的玩家就能更加沉浸其中。如果玩家能在远处微弱地听到僵尸的声音，而当它们靠近时，它们的不人道的哀嚎声会从一个方向或另一个方向变得更响亮，我们的僵尸游戏会更有趣。

很明显，空间化的数学将会很复杂。我们如何计算特定扬声器中的声音有多大声，基于声音来自的方向以及听者（声音的听者）到发出声音的物体（发射器）的距离？

幸运的是，SFML 为我们处理了所有复杂的事情。我们只需要熟悉一些技术术语，然后就可以开始使用 SFML 来对我们的音效进行空间化。

## 发射器、衰减和听众

为了让 SFML 能够正常工作，我们需要了解一些信息。我们需要知道声音在我们的游戏世界中来自哪里。这个声音的来源被称为**发射器**。在游戏中，发射器可以是僵尸、车辆，或者在我们当前的项目中，是一个火焰图块。我们已经在游戏中跟踪了物体的位置，所以给 SFML 发射器位置将会非常简单。

我们需要意识到的下一个因素是**衰减**。衰减是波动恶化的速率。你可以简化这个说法，并将其具体化为声音，即衰减是声音减小的速度。这在技术上并不准确，但对于本章的目的来说，这已经足够好了。

我们需要考虑的最后一个因素是**听众**。当 SFML 对声音进行空间化时，它是相对于什么进行空间化的？在大多数游戏中，合理的做法是使用玩家角色。在我们的游戏中，我们将使用 Thomas。

# SFML 如何处理空间化

SFML 有许多函数可以让我们处理发射器、衰减和听众。让我们先假设一下，然后我们将编写一些代码，真正为我们的项目添加空间化声音。

我们可以设置一个准备播放的音效，就像我们经常做的那样，如下所示：

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

我们可以使用`setPosition`函数来设置发射器的位置，如下面的代码所示：

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

如前面的代码注释中建议的，你如何获取发射器的坐标可能取决于游戏的类型。就像在 Zombie Arena 项目中所示的那样，这将是非常简单的。当我们在这个项目中设置位置时，我们将面临一些挑战。

我们可以使用以下代码设置衰减级别：

```cpp
zombieSound.setAttenuation(15); 

```

实际的衰减级别可能有些模糊。您希望玩家得到的效果可能与基于衰减的距离减小音量的准确科学公式不同。获得正确的衰减级别通常是通过实验来实现的。一般来说，衰减级别越高，声音级别降至静音的速度就越快。

此外，您可能希望在发射器周围设置一个音量完全不衰减的区域。如果该功能在一定范围之外不合适，或者您有大量的声源并且不想过度使用该功能，您可以这样做。为此，我们可以使用`setMinimumDistance`函数，如下所示：

```cpp
zombieSound.setMinDistance(150); 

```

通过上一行代码，衰减直到听众离发射器`150`像素/单位远才开始计算。

SFML 库中的一些其他有用函数包括`setLoop`函数。当传入 true 作为参数时，此函数将告诉 SFML 在循环播放声音时保持播放，如下面的代码所示：

```cpp
zombieSound.setLoop(true); 

```

声音将继续播放，直到我们使用以下代码结束它：

```cpp
zombieSound.stop(); 

```

不时地，我们会想要知道声音的状态（正在播放或已停止）。我们可以使用`getStatus`函数来实现这一点，如下面的代码所示：

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

在使用 SFML 进行声音空间化的最后一个方面我们需要涵盖的是听众在哪里？我们可以使用以下代码设置听众的位置：

```cpp
// Where is the listener?  
// How we get the values of x and y varies depending upon the game 
// In the Zombie Arena game or the Thomas Was Late game 
// We can use getPosition() 
Listener::setPosition(m_Thomas.getPosition().x,  
   m_Thomas.getPosition().y, 0.0f); 

```

上述代码将使所有声音相对于该位置播放。这正是我们需要的远处火瓦或迫近的僵尸的咆哮声，但对于像跳跃这样的常规音效来说，这是一个问题。我们可以开始处理一个发射器来定位玩家的位置，但 SFML 为我们简化了事情。每当我们想播放*正常*声音时，我们只需调用`setRelativeToListener`，如下面的代码所示，然后以与迄今为止完全相同的方式播放声音。以下是我们可能播放*正常*、非空间化跳跃音效的方式：

```cpp
jumpSound.setRelativeToListener(true); 
jumpSound.play(); 

```

我们只需要在播放任何空间化声音之前再次调用`Listener::setPosition`。

现在我们有了广泛的 SFML 声音函数，我们准备制作一些真正的空间化噪音。

# 构建 SoundManager 类

您可能还记得在上一个项目中，所有的声音代码占用了相当多的行数。现在考虑到空间化，它将变得更长。为了使我们的代码易于管理，我们将编写一个类来管理所有声音效果的播放。此外，为了帮助我们进行空间化，我们还将向 Engine 类添加一个函数，但我们将在后面的章节讨论。

## 编写 SoundManager.h

让我们开始编写和检查头文件。

在**解决方案资源管理器**中右键单击**标头文件**，然后选择**添加** | **新建项...**。在**添加新项**窗口中，通过左键单击**标头文件（** `.h` **）**，然后在**名称**字段中键入`SoundManager.h`。最后，单击**添加**按钮。现在我们准备为`SoundManager`类编写头文件。

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

在我们刚刚添加的代码中没有什么棘手的地方。有五个`SoundBuffer`对象和八个`Sound`对象。其中三个`Sound`对象将播放相同的`SoundBuffer`。这解释了不同数量的`Sound`/`SoundBuffer`对象的原因。我们这样做是为了能够同时播放多个咆哮声效，并具有不同的空间化参数。

请注意，有一个`m_NextSound`变量，将帮助我们跟踪这些潜在同时发生的声音中我们应该下一个使用哪一个。

有一个构造函数`SoundManager`，在那里我们将设置所有的音效，还有五个函数将播放音效。其中四个函数只是简单地播放*普通*音效，它们的代码将非常简单。

其中一个函数`playFire`将处理空间化的音效，并且会更加深入。注意`playFire`函数的参数。它接收一个`Vector2f`，这是发射器的位置，和第二个`Vector2f`，这是听众的位置。

## 编写 SoundManager.cpp 文件

现在我们可以编写函数定义了。构造函数和`playFire`函数有相当多的代码，所以我们将分别查看它们。其他函数都很简短，所以我们将一次处理它们。

在**解决方案资源管理器**中右键单击**源文件**，然后选择**添加** | **新建项...**。在**添加新项**窗口中，通过左键单击**C++文件（**`.cpp`**）**，然后在**名称**字段中键入`SoundManager.cpp`。最后，单击**添加**按钮。现在我们准备好为`SoundManager`类编写`.cpp`文件了。

### 编写构造函数

为`SoundManager.cpp`添加以下包含指令和构造函数的代码：

```cpp
#include "stdafx.h" 
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

在前面的代码中，我们将五个声音文件加载到五个`SoundBuffer`对象中。接下来，我们将八个`Sound`对象与其中一个`SoundBuffer`对象关联起来。请注意，`m_Fire1Sound`、`m_Fire2Sound`和`m_Fire3Sound`都将从同一个`SoundBuffer`，`m_FireBuffer`中播放。

接下来，我们设置了三个火焰声音的衰减和最小距离。

### 提示

通过实验得出了分别为`150`和`15`的值。一旦游戏运行起来，我鼓励你通过改变这些值来进行实验，看（或者说听）听到的差异。

最后，对于构造函数，我们在每个与火相关的`Sound`对象上使用了`setLoop`函数。现在当我们调用`play`时，它们将持续播放。

### 编写 playFire 函数

添加下面代码中显示的`playFire`函数，然后我们可以讨论它：

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

我们要做的第一件事是调用`Listener::setPosition`，并根据作为参数传入的`Vector2f`设置听众的位置。

接下来，代码根据`m_NextSound`的值进入了一个`switch`块。每个`case`语句都做了完全相同的事情，但是针对`m_Fire1Sound`、`m_Fire2Sound`或`m_Fire3Sound`。

在每个`case`块中，我们使用传入的参数调用`setPosition`函数来设置发射器的位置。每个`case`块中代码的下一部分检查音效当前是否已停止，如果是，则播放音效。我们很快就会看到如何得到传递到这个函数中的发射器和听众的位置。

`playFire`函数的最后部分增加了`m_NextSound`，并确保它只能等于 1、2 或 3，这是`switch`块所需的。

### 编写其余的 SoundManager 函数

添加这四个简单的函数：

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

`playFallInFire`、`playFallInWater`和`playReachGoal`函数只做两件事。首先，它们各自调用`setRelativeToListener`，所以音效不是空间化的，使音效变得*普通*，而不是定向的，然后它们调用适当的`Sound`对象上的`play`。

这就结束了`SoundManager`类。现在我们可以在`Engine`类中使用它。

# 将 SoundManager 添加到游戏引擎

打开`Engine.h`文件，并添加一个新的`SoundManager`类的实例，如下面突出显示的代码所示：

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

在这一点上，我们可以使用`m_SM`来调用各种`play...`函数。不幸的是，仍然有一些工作要做，以便管理发射器（火砖）的位置。

# 填充声音发射器

打开`Engine.h`文件，并为`populateEmitters`函数添加一个新的原型和一个新的 STL`vector`的`Vector2f`对象：

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

`populateEmitters`函数以`Vector2f`对象的`vector`和指向`int`的指针作为参数。这个`vector`将保存每个发射器在一个级别中的位置，而数组是我们的二维数组，它保存了一个级别的布局。

## 编写 populateEmitters 函数

`populateEmitters`函数的工作是扫描`arrayLevel`的所有元素，并决定在哪里放置发射器。它将其结果存储在`m_FireEmitters`中。

在“解决方案资源管理器”中右键单击“源文件”，然后选择“添加”|“新项目”。在“添加新项目”窗口中，选择（通过左键单击）“C++文件”（.cpp），然后在“名称”字段中输入`PopulateEmitters.cpp`。最后，单击“添加”按钮。现在我们可以编写新函数`populateEmitters`。

添加完整的代码；确保在编写代码时仔细研究代码，然后我们可以讨论它。

```cpp
#include "stdafx.h" 
#include "Engine.h" 

using namespace sf; 
using namespace std; 

void Engine::populateEmitters( 
   vector <Vector2f>& vSoundEmitters, int** arrayLevel) 
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
               // Add the coordiantes of this water block 
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

一些代码乍一看可能会很复杂。理解我们用来选择发射器位置的技术将使其变得更简单。在我们的级别中，通常会有大块的火砖。在我设计的一个级别中，有超过 30 个火砖。代码确保在给定的矩形内只有一个发射器。这个矩形存储在`previousEmitter`中，大小为 300 像素乘以 300 像素（TILE_SIZE * 6）。

该代码设置了一个嵌套的`for`循环，循环遍历`arrayLevel`以寻找火砖。当找到一个时，它确保它不与`previousEmitter`相交。只有在这种情况下，它才使用`pushBack`函数向`vSoundEmitters`添加另一个发射器。在这样做之后，它还更新`previousEmitter`以避免得到大量的声音发射器。

让我们制造一些噪音。

# 播放声音

打开`LoadLevel.cpp`文件，并在以下代码中添加对新的`populateEmitters`函数的调用：

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

要添加的第一个声音是跳跃声音。您可能记得键盘处理代码在`Bob`和`Thomas`类的纯虚函数中，而`handleInput`函数在成功启动跳跃时返回`true`。

打开`Input.cpp`文件，并添加高亮代码行，以在 Thomas 或 Bob 成功开始跳跃时播放跳跃声音。

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

打开`Update.cpp`文件，并添加高亮代码行，以在 Thomas 和 Bob 同时到达当前级别的目标时播放成功的声音。

```cpp
// Detect collisions and see if characters have reached the goal tile 
// The second part of the if condition is only executed 
// when thomas is touching the home tile 
if (detectCollisions(m_Thomas) && detectCollisions(m_Bob)) 
{ 
   // New level required 
   m_NewLevelRequired = true; 

   // Play the reach goal sound 
 m_SM.playReachGoal(); 

} 
else 
{ 
   // Run bobs collision detection 
   detectCollisions(m_Bob); 
} 

```

同样在`Update.cpp`文件中，我们将添加代码来循环遍历`m_FireEmitters`向量，并决定何时需要调用`SoundManager`类的`playFire`函数。

仔细观察新的高亮代码周围的一小部分上下文。在恰当的位置添加这段代码是至关重要的。

```cpp
}// End if playing 

// Check if a fire sound needs to be played
vector<Vector2f>::iterator it;

// Iterate through the vector of Vector2f objects
for (it = m_FireEmitters.begin();it != m_FireEmitters.end(); it++)
{
   // Where is this emitter?
   // Store the location in pos
   float posX = (*it).x;
   float posY = (*it).y;
   // is the emiter near the player?
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

前面的代码有点像声音的碰撞检测。每当 Thomas 停留在围绕火砖发射器的 500x500 像素矩形内时，`playFire`函数就会被调用，传入发射器和 Thomas 的坐标。然后`playFire`函数会完成其余的工作，并触发一个空间化的循环声音效果。

打开`DetectCollisions.cpp`文件，找到适当的位置，并按照以下所示添加高亮代码。两行高亮代码触发了当角色掉入水或火砖时播放声音效果。

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

玩游戏将允许您听到所有的声音，包括在靠近火砖时的很酷的空间化。

# HUD 类

HUD 是超级简单的，与书中的其他两个项目没有什么不同。我们要做的不同之处在于将所有代码封装在一个新的 HUD 类中。如果我们将所有的字体、文本和其他变量声明为这个新类的成员，然后在构造函数中初始化它们，并为它们提供 getter 函数，这将使得`Engine`类清除了大量的声明和初始化。

## 编写 HUD.h

首先，我们将编写`HUD.h`文件，其中包含所有成员变量和函数声明。在**解决方案资源管理器**中右键单击**头文件**，然后选择**添加** | **新建项...**。在**添加新项**窗口中，通过左键单击**头文件**（`.h`）并在**名称**字段中键入`HUD.h`。最后，单击**添加**按钮。现在我们准备为`HUD`类编写头文件。

将以下代码添加到`HUD.h`中：

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

在先前的代码中，我们添加了一个`Font`实例和三个`Text`实例。`Text`对象将用于显示提示用户开始、剩余时间和当前级别编号的消息。

公共函数更有趣。首先是构造函数，大部分代码将在其中。构造函数将初始化`Font`和`Text`对象，并将它们相对于当前屏幕分辨率定位在屏幕上。

三个 getter 函数，`getMessage`、`getLevel`和`getTime`将返回一个`Text`对象给调用代码，以便能够将它们绘制到屏幕上。

`setLevel`和`setTime`函数将用于更新显示在`m_LevelText`和`m_TimeText`中的文本。

现在我们可以编写刚刚概述的所有函数的定义。

## 编写 HUD.cpp 文件

在**解决方案资源管理器**中右键单击**源文件**，然后选择**添加** | **新建项...**。在**添加新项**窗口中，通过左键单击**C++文件（** `.cpp` **）**并在**名称**字段中键入`HUD.cpp`。最后，单击**添加**按钮。现在我们准备为`HUD`类编写`.cpp`文件。

添加包含指令和以下代码，然后我们将讨论它：

```cpp
#include "stdafx.h" 
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

首先，我们将水平和垂直分辨率存储在名为`resolution`的`Vector2u`中。接下来，我们从我们在第十二章中添加的`fonts`目录中加载字体。*抽象和代码管理 - 更好地利用面向对象编程*。

接下来的四行代码设置了`m_StartText`的字体、颜色、大小和文本。此后的代码块捕获了包裹`m_StartText`的矩形的大小，并进行计算以确定如何将其居中放置在屏幕上。如果您想对代码的这部分进行更详细的解释，请参考第三章：*C++字符串、SFML 时间 - 玩家输入和 HUD*。

构造函数中的最后两个代码块设置了`m_TimeText`和`m_LevelText`的字体、文本大小、颜色、位置和实际文本。然而，我们很快会看到，这两个`Text`对象将通过两个 setter 函数进行更新，每当需要时。

在我们刚刚添加的代码之后立即添加以下 getter 和 setter 函数：

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

先前代码中的前三个函数只是返回适当的`Text`对象，`m_StartText`、`m_LevelText`和`m_TimeText`。我们将很快使用这些函数，在屏幕上绘制 HUD 时。最后两个函数，`setLevel`和`setTime`，使用`setString`函数来更新适当的`Text`对象，该值将从`Engine`类的`update`函数中每 500 帧传入。

完成所有这些后，我们可以在游戏引擎中使用 HUD 类。

# 使用 HUD 类

打开`Engine.h`，添加一个包含我们新类的声明，声明一个新的`HUD`类的实例，并声明并初始化两个新的成员变量，用于跟踪我们更新 HUD 的频率。正如我们在之前的两个项目中学到的，我们不需要为每一帧都这样做。

将突出显示的代码添加到`Engine.h`中：

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

 // The Hud   Hud m_Hud;
   int m_FramesSinceLastHUDUpdate = 0;
   int m_TargetFramesPerHUDUpdate = 500; 

   const int TILE_SIZE = 50; 

```

接下来，我们需要在`Engine`类的`update`函数中添加一些代码。打开`Update.cpp`并添加突出显示的代码以每 500 帧更新 HUD：

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
   // Increment the number of frames since the last HUD calculation
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

在前面的代码中，`m_FramesSinceLastUpdate`在每帧递增。当`m_FramesSinceLastUpdate`超过`m_TargetFramesPerHUDUpdate`时，执行进入`if`块。在`if`块内部，我们使用`stringstream`对象来更新我们的`Text`，就像在之前的项目中所做的那样。然而，正如你可能期望的那样，在这个项目中我们使用了`HUD`类，所以我们调用`setTime`和`setLevel`函数，传入`Text`对象需要设置的当前值。

`if`块中的最后一步是将`m_FramesSinceLastUpdate`设置回零，这样它就可以开始计算下一个更新。

最后，打开`Draw.cpp`文件，并添加突出显示的代码来每帧绘制 HUD。

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

前面的代码使用 HUD 类的 getter 函数来绘制 HUD。请注意，只有在游戏当前未进行时`(!m_Playing)`才会调用绘制提示玩家开始的消息。

运行游戏并玩几个关卡，看时间倒计时和关卡增加。当你再次回到第一关时，注意你的时间比之前少了 10%。

# 总结

我们的《Thomas Was Late》游戏不仅可以完全玩得了，还有定向音效和简单但信息丰富的 HUD，而且我们还可以轻松添加新的关卡。在这一点上，我们可以说它已经完成了。

增加一些闪光效果会很好。在接下来的章节中，我们将探讨两个游戏概念。首先，我们将研究粒子系统，这是我们如何处理爆炸或其他特殊效果的方法。为了实现这一点，我们需要学习更多的 C++知识，看看我们如何彻底重新思考我们的游戏代码结构。

之后，当我们学习 OpenGL 和可编程图形管线时，我们将为游戏添加最后的点睛之笔。然后，我们将有机会涉足**GLSL**语言，这使我们能够编写直接在 GPU 上执行的代码，以创建一些特殊效果。
