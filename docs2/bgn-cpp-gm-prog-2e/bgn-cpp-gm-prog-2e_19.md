# *第十八章*：粒子系统和着色器

在本章中，我们将探讨粒子系统是什么，然后将其编码到我们的游戏中。我们将探讨 OpenGL 着色器的基础，并看看用另一种语言（**GLSL**）编写代码，该代码可以直接在图形卡上运行，可以产生可能无法实现的平滑图形效果。像往常一样，我们也将使用我们的新技能和知识来增强当前项目。

在本章中，我们将涵盖以下主题：

+   构建粒子系统

+   OpenGL 着色器和 GLSL

+   在《托马斯迟到》游戏中使用着色器

# 构建粒子系统

在我们开始编码之前，看看我们究竟要实现什么是非常有帮助的。

查看以下图表：

![图片](img/B14278_18_01.jpg)

之前的插图是粒子效果在普通背景上的截图。我们将在我们的游戏中使用这个效果。每当玩家死亡时，我们将生成这些效果之一。

我们实现这种效果的方式如下：

1.  首先，我们在一个选定的像素位置生成 1,000 个点（粒子），一个叠在另一个上面。

1.  游戏的每一帧都会将 1,000 个粒子以预定的但随机的速度和角度向外移动。

1.  重复第二步两秒钟，然后让粒子消失。

我们将使用 `VertexArray` 来绘制所有点，并使用 `Point` 的原始类型来在视觉上表示每个粒子。此外，我们将从 SFML 的 `Drawable` 类继承，这样我们的粒子系统就可以负责绘制自己。

## 编码粒子类

`Particle` 类将是一个简单的类，它只代表一千个粒子中的一个。让我们开始编码。

### 编码 `Particle.h`

右键点击 `Particle.h`。最后，点击 `Particle` 类。

将以下代码添加到 `Particle.h` 文件中：

```cpp
#pragma once
#include <SFML/Graphics.hpp>
using namespace sf;
class Particle
{
private:
    Vector2f m_Position;
    Vector2f m_Velocity;
public:
    Particle(Vector2f direction);
    void update(float dt);
    void setPosition(Vector2f position);
    Vector2f getPosition();
};
```

在前面的代码中，我们有两个 `Vector2f` 对象。一个将代表粒子的水平和垂直坐标，而另一个将代表水平和垂直速度。

重要提示

当你在多个方向上有变化率（速度）时，组合的值也定义了一个方向。这个 `Vector2f` 被称为 `m_Velocity`。

我们还有几个公共函数。首先是构造函数。它接受一个 `Vector2f` 并使用它来让系统知道这个粒子将有什么方向/速度。这意味着系统，而不是粒子本身，将选择速度。

接下来是 `update` 函数，它接受上一帧所花费的时间。我们将使用这个时间来精确地移动粒子。

最后两个函数，`setPosition` 和 `getPosition`，用于移动粒子的位置和分别找出其位置。

当我们编码这些函数时，它们都将完全有意义。

### 编码 `Particle.cpp` 文件

右键点击 `Particle.cpp`。最后，点击 `Particle` 类的 `.cpp` 文件。

将以下代码添加到 `Particle.cpp`：

```cpp
#include "Particle.h"
Particle::Particle(Vector2f direction)
{
    // Determine the direction

    m_Velocity.x = direction.x;
    m_Velocity.y = direction.y;
}
void Particle::update(float dtAsSeconds)
{
    // Move the particle
    m_Position += m_Velocity * dtAsSeconds;
}
void Particle::setPosition(Vector2f position)
{
    m_Position = position;
}
Vector2f Particle::getPosition()
{
    return m_Position;
}
```

所有这些函数都使用了我们之前见过的概念。构造函数使用传入的`Vector2f`对象设置`m_Velocity.x`和`m_Velocity.y`值。

`update`函数通过将`m_Velocity`乘以经过的时间(`dtAsSeconds`)来移动粒子的水平和垂直位置。注意，为了实现这一点，我们只需将两个`Vector2f`对象相加即可。没有必要分别对 x 和 y 成员进行计算。

`setPosition`函数，正如我们之前解释的，使用传入的值初始化`m_Position`对象。`getPosition`函数将`m_Position`返回给调用代码。

我们现在有一个完全功能的`Particle`类。接下来，我们将编写一个`ParticleSystem`类来生成和控制粒子。

## 编写 ParticleSystem 类

`ParticleSystem`类为我们处理大部分粒子效果的工作。我们将在这个`Engine`类中创建这个类的实例。在我们这样做之前，让我们再谈谈面向对象编程和 SFML 的`Drawable`类。

# 探索 SFML 的 Drawable 类和面向对象编程

`Drawable`类只有一个函数。它也没有变量。此外，它的唯一函数是纯虚函数。这意味着，如果我们从`Drawable`继承，我们必须实现它的唯一函数。提醒一下，从*第十四章*，*抽象和代码管理 – 更好地利用面向对象编程*，我们的类可以继承自`drawable`作为多态类型。更简单地说，我们可以用继承自`Drawable`的类做任何 SFML 允许我们对`Drawable`对象做的事情。唯一的要求是我们必须为纯虚函数`draw`提供一个定义。

一些继承自`Drawable`的类已经包括`Sprite`和`VertexArray`（以及其他）。每次我们使用`Sprite`或`VertexArray`时，我们都会将它们传递给`RenderWindow`类的`draw`函数。

我们之所以能够在这整本书中绘制出我们曾经绘制过的每一个对象，是因为它们都继承自`Drawable`。我们可以利用这一知识来发挥优势。

我们可以用我们喜欢的任何对象继承自`Drawable`，只要我们实现了纯虚函数`draw`。这也是一个简单直接的过程。考虑一个假设的`SpaceShip`类。继承自`Drawable`的`SpaceShip`类的头文件(`SpaceShip.h`)可能看起来像这样：

```cpp
class SpaceShip : public Drawable
{
private:
    Sprite m_Sprite;
    // More private members
public:
    virtual void draw(RenderTarget& target, 
        RenderStates states) const;
    // More public members
};
```

在前面的代码中，我们可以看到纯虚函数`draw`和一个`Sprite`实例。注意，在类外部无法访问私有的`Sprite` – 甚至没有`getSprite`函数！

`SpaceShip.cpp`文件可能看起来像这样：

```cpp
void SpaceShip::SpaceShip
{
    // Set up the spaceship
}
void SpaceShip::draw(RenderTarget& target, RenderStates states) const
{
    target.draw(m_Sprite, states);
}
// Any other functions
```

在前面的代码中，注意 `draw` 函数的简单实现。这些参数超出了本书的范围。只需记住，`target` 参数用于调用 `draw` 函数，并传入 `m_Sprite` 以及其他参数。

小贴士

虽然理解参数并非充分利用 `Drawable` 的必要条件，但在本书的语境中，你可能对此感到好奇。你可以在 SFML 网站上了解更多关于 SFML `Drawable` 的信息：[`www.sfml-dev.org/tutorials/2.5/graphics-vertex-array.php`](https://www.sfml-dev.org/tutorials/2.5/graphics-vertex-array.php)。

在主游戏循环中，我们现在可以将 `SpaceShip` 实例视为 `Sprite` 或任何继承自 `Drawable` 的其他类，如下所示：

```cpp
SpaceShip m_SpaceShip;
// create other objects here
// ...
// In the draw function
// Rub out the last frame
m_Window.clear(Color::Black);
// Draw the spaceship
m_Window.draw(m_SpaceShip);
// More drawing here
// ...
// Show everything we have just drawn
m_Window.display();
```

正是因为 `SpaceShip` 是 `Drawable`，我们才能将其视为 `Sprite` 或 `VertexArray`，并且因为我们重写了纯虚函数 `draw`，所以一切都能按预期工作。你将在本章中使用这种方法来绘制粒子系统。

在我们讨论面向对象编程（OOP）的主题时，让我们看看将绘图代码封装到游戏对象中的另一种方法，这是我们将在下一个项目中使用的方法。

## 继承自 `Drawable` 的替代方案

通过在类中实现自己的函数，我们也可以将所有绘图功能保留在要绘制的对象所在的类中，例如使用以下代码：

```cpp
void drawThisObject(RenderWindow window)
{
    window.draw(m_Sprite)
}
```

之前的代码假设 `m_Sprite` 代表当前我们正在绘制的类的视觉外观，正如在本项目和上一个项目中一样。假设包含 `drawThisObject` 函数的类的实例被称作 `playerHero`，并且进一步假设我们有一个名为 `m_Window` 的 `RenderWindow` 实例，我们就可以使用以下代码从主游戏循环中绘制对象：

```cpp
 playerHero.draw(m_Window);
```

在这个解决方案中，我们将 `RenderWindow` 的 `m_Window` 作为参数传递给 `drawThisObject` 函数。然后，`drawThisObject` 函数使用 `RenderWindow` 来绘制 `Sprite`，`m_Sprite`。

如果我们有一组更复杂的游戏对象，那么在每一帧将 `RenderWindow` 的引用传递给要绘制的对象，以便它能够自行绘制，这是一种很好的策略。

我们将在本书的最终项目中使用这种策略，我们将在下一章开始。让我们通过编码继承自 `Drawable` 的 `ParticleSystem` 类来完成粒子系统，这将继承自 `ParticleSystem` 类。

### 编码 `ParticleSystem.h`

右键点击 `ParticleSystem.h`。最后，点击 `ParticleSystem` 类。

将 `ParticleSystem` 类的代码添加到 `ParticleSystem.h` 中：

```cpp
#pragma once
#include <SFML/Graphics.hpp>
#include "Particle.h"
using namespace sf;
using namespace std;
class ParticleSystem : public Drawable
{
private:
    vector<Particle> m_Particles;
    VertexArray m_Vertices;
    float m_Duration;
    bool m_IsRunning = false;
public:
    virtual void draw(RenderTarget& target, 
      RenderStates states) const;

    void init(int count);
    void emitParticles(Vector2f position);
    void update(float elapsed);
    bool running();
};
```

我们一点一点地来。首先，注意我们正在继承自 SFML 的 `Drawable` 类。这将允许我们将 `ParticleSystem` 实例传递给 `m_Window.draw`，因为 `ParticleSystem` `Drawable`。而且，由于我们继承自 `Drawable`，我们可以使用与 `Drawable` 类内部使用的相同函数签名来重写 `draw` 函数。简而言之，当我们使用 `ParticleSystem` 类时，我们会看到以下代码。

```cpp
m_Window.draw(m_PS);
```

`m_PS` 对象是我们 `ParticleSystem` 类的一个实例，我们将直接将其传递给 `RenderWindow` 类的 `draw` 函数，就像我们为 `Sprite`、`VertexArray` 和 `RectangleShape` 实例所做的那样。所有这一切都是通过继承和多态的力量实现的。

提示

还不要添加 `m_Window.draw…` 代码；我们还有更多的工作要做。

有一个名为 `m_Particles` 的 `Particle` 类型的向量。这个向量将保存每个 `Particle` 实例。接下来，我们有一个名为 `m_Vertices` 的 `VertexArray`。这将用于以大量 `Point` 原语的形式绘制所有粒子。

`m_Duration`，`float` 变量表示每个效果将持续多长时间。我们将在构造函数中初始化它。

`m_IsRunning` 布尔变量将用于指示粒子系统是否正在使用中。

接下来，在公共部分，我们有一个纯虚函数 `draw`，我们很快就会实现它来处理当我们传递 `ParticleSystem` 实例到 `m_Window.draw` 时会发生什么。

`init` 函数将准备 `VertexArray` 和 `vector`。它还将使用速度和初始位置初始化所有 `Particle` 对象（由 `vector` 持有）。

`update` 函数将遍历 `vector` 中的每个 `Particle` 实例，并调用它们的单个 `update` 函数。

`running` 函数提供了访问 `m_IsRunning` 变量的权限，以便游戏引擎可以查询 `ParticleSystem` 是否正在使用中。

让我们编写函数定义，看看 `ParticleSystem` 内部发生了什么。

### 编写 `ParticleSystem.cpp` 文件

右键点击 `ParticleSystem.cpp`。最后，点击 `ParticleSystem` 类的 `.cpp` 文件。

我们将把这个文件分成五个部分，这样我们就可以更详细地进行编码和讨论。添加以下代码的第一部分：

```cpp
#include <SFML/Graphics.hpp>
#include "ParticleSystem.h"
using namespace sf;
using namespace std;
void ParticleSystem::init(int numParticles)
{
    m_Vertices.setPrimitiveType(Points);
    m_Vertices.resize(numParticles);
    // Create the particles
    for (int i = 0; i < numParticles; i++)
    {
        srand(time(0) + i);
        float angle = (rand() % 360) * 3.14f / 180.f;
        float speed = (rand() % 600) + 600.f;
        Vector2f direction;
        direction = Vector2f(cos(angle) * speed,
            sin(angle) * speed);
        m_Particles.push_back(Particle(direction));
    }
}
```

在必要的 `includes` 之后，我们有 `init` 函数的定义。我们使用 `Points` 作为参数调用 `setPrimitiveType`，以便 `m_VertexArray` 知道它将处理哪种类型的原语。我们使用 `numParticles` 调整 `m_Vertices` 的大小，这是在调用 `init` 函数时传递给它的。

`for` 循环为速度和角度创建随机值。然后使用三角函数将这些值转换为存储在 `Vector2f`，`direction` 中的向量。

提示

如果你想了解更多关于如何使用三角函数（`cos`和`sin`）将角度和速度转换为向量，你可以查看这篇文章系列：[`gamecodeschool.com/essentials/calculating-heading-in-2d-games-using-trigonometric-functions-part-1/`](http://gamecodeschool.com/essentials/calculating-heading-in-2d-games-using-trigonometric-functions-part-1/)。

在`for`循环（以及`init`函数）中最后发生的事情是将向量传递给`Particle`构造函数。新的`Particle`实例使用`push_back`函数存储在`m_Particles`中。因此，调用`init`并传入`1000`的值意味着我们在`m_Particles`中有 1,000 个`Particle`实例，它们具有随机速度，正等待着爆炸！

接下来，将`update`函数添加到`ParticleSysytem.cpp`中：

```cpp
void ParticleSystem::update(float dt)
{
    m_Duration -= dt;
    vector<Particle>::iterator i;
    int currentVertex = 0;
    for (i = m_Particles.begin(); i != m_Particles.end(); i++)
    {
        // Move the particle
        (*i).update(dt);
        // Update the vertex array
        m_Vertices[currentVertex++].position = i->getPosition();
    }
    if (m_Duration < 0)
    {
        m_IsRunning = false;
    }
}
```

`update`函数看起来比实际要简单。首先，`m_Duration`通过传入的时间`dt`减少。这样我们就可以知道两秒钟是否已经过去。声明了一个向量迭代器`i`用于`m_Particles`。

`for`循环遍历`m_Particles`中的每个`Particle`实例。对于每一个实例，它调用其`update`函数并传入`dt`。每个粒子将更新其位置。粒子更新后，使用粒子的`getPosition`函数更新`m_Vertices`中适当的顶点。每次`for`循环结束时，`currentVertex`增加，为下一个顶点做准备。

`for`循环完成后，代码检查`if(m_Duration < 0)`是否是时候关闭效果。如果已经过去了两秒钟，`m_IsRunning`被设置为`false`。

接下来，添加`emitParticles`函数：

```cpp
void ParticleSystem::emitParticles(Vector2f startPosition)
{
    m_IsRunning = true;
    m_Duration = 2;

    int currentVertex = 0;
    for (auto it = m_Particles.begin(); 
         it != m_Particles.end();
         it++)
    {
        m_Vertices[currentVertex++].color = Color::Yellow;
        it->setPosition(startPosition);
    }
}
```

这是我们将调用来启动粒子系统的函数。所以，不出所料，我们将`m_IsRunning`设置为`true`，将`m_Duration`设置为`2`。我们声明了一个迭代器`i`来遍历`m_Particles`中的所有`Particle`对象，然后在`for`循环中这样做。

在`for`循环内部，我们将顶点数组中的每个粒子设置为黄色，并将每个位置设置为传入的参数`startPosition`。记住，每个粒子从相同的位置开始，但它们被分配了不同的速度。

接下来，添加纯虚`draw`函数的定义：

```cpp
void ParticleSystem::
       draw(RenderTarget& target, 
       RenderStates states) const
{
    target.draw(m_Vertices, states);
}
```

在前面的代码中，我们简单地使用`target`调用`draw`，传入`m_Vertices`和`states`作为参数。记住，我们永远不会直接调用这个函数！不久，当我们声明`ParticleSystem`的实例时，我们将该实例传递给`RenderWindow draw`函数。我们刚刚编写的`draw`函数将从那里内部调用。

最后，添加`running`函数：

```cpp
bool ParticleSystem::running()
{
    return m_IsRunning;
}
```

`running`函数是一个简单的 getter 函数，它返回`m_IsRunning`的值。我们将在本章中看到它的用途，以便我们可以确定粒子系统的当前状态。

## 使用`ParticleSystem`对象

将我们的粒子

系统要工作的方式非常简单，尤其是在我们继承了`Drawable`之后。

### 将`ParticleSystem`对象添加到`Engine`类中

打开`Engine.h`文件，并添加一个`ParticleSystem`对象，如下所示突出显示的代码：

```cpp
#pragma once
#include <SFML/Graphics.hpp>
#include "TextureHolder.h"
#include "Thomas.h"
#include "Bob.h"
#include "LevelManager.h"
#include "SoundManager.h"
#include "HUD.h"
#include "ParticleSystem.h"
using namespace sf;
class Engine
{
private:
    // The texture holder
    TextureHolder th;
    // create a particle system
    ParticleSystem m_PS;
    // Thomas and his friend, Bob
    Thomas m_Thomas;
    Bob m_Bob;
```

现在，我们需要初始化系统。

### 初始化粒子系统

打开`Engine.cpp`文件，并在`Engine`构造函数的末尾添加以下简短的突出显示代码：

```cpp
Engine::Engine()
{
    // Get the screen resolution and create an SFML window and View
    Vector2f resolution;
    resolution.x = VideoMode::getDesktopMode().width;
    resolution.y = VideoMode::getDesktopMode().height;
    m_Window.create(VideoMode(resolution.x, resolution.y),
        "Thomas was late",
        Style::Fullscreen);
    // Initialize the full screen view
    m_MainView.setSize(resolution);
    m_HudView.reset(
        FloatRect(0, 0, resolution.x, resolution.y));
    // Initialize the split-screen Views
    m_LeftView.setViewport(
        FloatRect(0.001f, 0.001f, 0.498f, 0.998f));
    m_RightView.setViewport(
        FloatRect(0.5f, 0.001f, 0.499f, 0.998f));
    m_BGLeftView.setViewport(
        FloatRect(0.001f, 0.001f, 0.498f, 0.998f));
    m_BGRightView.setViewport(
        FloatRect(0.5f, 0.001f, 0.499f, 0.998f));
    // Can this graphics card use shaders?
    if (!sf::Shader::isAvailable())
    {
        // Time to get a new PC
        m_Window.close();
    }
    m_BackgroundTexture = TextureHolder::GetTexture(
        "graphics/background.png");
    // Associate the sprite with the texture
    m_BackgroundSprite.setTexture(m_BackgroundTexture);
    // Load the texture for the background vertex array
    m_TextureTiles = TextureHolder::GetTexture(
        "graphics/tiles_sheet.png");
    // Initialize the particle system
    m_PS.init(1000);
}// End Engine constructor
```

`VertexArray`和`Particle`实例的`vector`已经准备好行动了。

### 每帧更新粒子系统

打开`Update.cpp`文件，并添加以下突出显示的代码。它可以直接放在`update`函数的末尾：

```cpp
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
    // Update the particles
    if (m_PS.running())
    {
        m_PS.update(dtAsSeconds);
    }
}// End of update function
```

之前代码中所需的一切就是调用`update`。注意，它被包裹在一个检查中，以确保系统目前正在运行。如果它没有运行，就没有更新它的必要。

### 启动粒子系统

打开`DetectCollisions.cpp`文件，其中包含`detectCollisions`函数。我们在最初编写代码时在其中留下了注释。

从上下文中识别正确的位置，并添加以下突出显示的代码：

```cpp
// Is character colliding with a regular block
if (m_ArrayLevel[y][x] == 1)
{
    if (character.getRight().intersects(block))
    {
        character.stopRight(block.left);
    }
    else if (character.getLeft().intersects(block))
    {
        character.stopLeft(block.left);
    }
    if (character.getFeet().intersects(block))
    {
        character.stopFalling(block.top);
    }
    else if (character.getHead().intersects(block))
    {
        character.stopJump();
    }
}
// More collision detection here once 
// we have learned about particle effects
// Have the characters' feet touched fire or water?
// If so, start a particle effect
// Make sure this is the first time we have detected this
// by seeing if an effect is already running            
if (!m_PS.running()) {
    if (m_ArrayLevel[y][x] == 2 || m_ArrayLevel[y][x] == 3)
    {
        if (character.getFeet().intersects(block))
        {
            // position and start the particle system
            m_PS.emitParticles(character.getCenter());
        }
    }
}
// Has the character reached the goal?
if (m_ArrayLevel[y][x] == 4)
{
    // Character has reached the goal
    reachedGoal = true;
}
```

首先，代码检查粒子系统是否已经在运行。如果不是，它会检查当前正在检查的瓷砖是否是水或火瓷砖。如果是其中之一，它会检查角色的脚是否与之接触。当这些`if`语句中的每一个都为真时，通过调用`emitParticles`函数并传入角色的中心位置作为开始效果的坐标，粒子系统将通过调用`emitParticles`函数并传入角色的中心位置作为开始效果的坐标来启动。

### 绘制粒子系统

这是最精彩的部分。看看绘制`ParticleSystem`有多简单。我们在确认粒子系统正在运行后，直接将我们的实例传递给`m_Window.draw`函数。

打开`Draw.cpp`文件，并在所有必要的位置添加以下突出显示的代码：

```cpp
void Engine::draw()
{
    // Rub out the last frame
    m_Window.clear(Color::White);
    if (!m_SplitScreen)
    {
        // Switch to background view
        m_Window.setView(m_BGMainView);
        // Draw the background
        m_Window.draw(m_BackgroundSprite);
        // Switch to m_MainView
        m_Window.setView(m_MainView);        
        // Draw the Level
        m_Window.draw(m_VALevel, &m_TextureTiles);
        // Draw thomas
        m_Window.draw(m_Thomas.getSprite());
        // Draw bob
        m_Window.draw(m_Bob.getSprite());
        // Draw the particle system
        if (m_PS.running())
        {
            m_Window.draw(m_PS);
        }
    }
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

        // Draw bob
        m_Window.draw(m_Bob.getSprite());
        // Draw thomas
        m_Window.draw(m_Thomas.getSprite());
        // Draw the particle system
        if (m_PS.running())
        {
            m_Window.draw(m_PS);
        }

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
        // Draw the particle system
        if (m_PS.running())
        {
            m_Window.draw(m_PS);
        }

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
}
```

注意，我们必须在所有左、右和全屏代码块中绘制粒子系统。

运行游戏，并将其中一个角色的脚移到火瓷砖的边缘。注意粒子系统瞬间活跃起来：

![](img/B14278_18_02.jpg)

现在，是时候介绍一些新的内容了。

# OpenGL、着色器和 GLSL

**开放图形库**（**OpenGL**）是一个处理 2D 和 3D 图形的编程库。OpenGL 在所有主要的桌面操作系统上运行，还有一个版本可以在移动设备上运行，称为 OpenGL ES。

OpenGL 最初于 1992 年发布。经过二十多年的改进和优化。此外，显卡制造商设计他们的硬件以使其与 OpenGL 良好配合。提到这一点并不是为了历史课，而是为了解释尝试改进 OpenGL 并在桌面上的 2D（和 3D 游戏）中使用它将是一个徒劳的行为，尤其是如果我们希望我们的游戏在 Windows 以外的操作系统上运行，这是显而易见的选择。我们已经在使用 OpenGL，因为 SFML 使用了 OpenGL。着色器是运行在 GPU 上的程序。我们将在下一节中了解更多关于它们的信息。

## 可编程管道和着色器

通过 OpenGL，我们可以访问`RenderWindow`实例的`draw`函数。我们还可以编写在调用`draw`之后可以在 GPU 上运行的代码，以独立操纵每个像素。这是一个非常强大的功能。

在 GPU 上运行的额外代码被称为**着色器程序**。我们可以编写代码来操纵我们的图形的几何（位置）在**顶点着色器**中。我们也可以编写代码来单独操纵每个像素的外观。这被称为**片段着色器**。

虽然我们不会深入探讨着色器，但我们将使用**GL 着色器语言**（**GLSL**）编写一些着色器代码，并一窥它提供的可能性。

在 OpenGL 中，一切都是一个点、一条线或一个三角形。此外，我们还可以将颜色和纹理附加到这些基本几何形状上，我们还可以将这些元素组合起来，以制作出我们在现代游戏中看到的复杂图形。这些统称为`VertexArray`，以及`Sprite`和`Shape`类。

除了基本图形元素外，OpenGL 还使用矩阵。矩阵是一种进行算术运算的方法和结构。这种算术运算可以非常简单，例如移动（平移）坐标，也可以非常复杂，例如执行更高级的数学运算，例如将我们的游戏世界坐标转换为 GPU 可以使用的 OpenGL 屏幕坐标。幸运的是，SFML 在幕后为我们处理了这种复杂性。SFML 还允许我们直接处理 OpenGL。

小贴士

如果你想了解更多关于 OpenGL 的信息，你可以从这里开始：[`learnopengl.com/#!Introduction`](http://learnopengl.com/#!Introduction)。如果你想直接使用 OpenGL，同时使用 SFML，你可以阅读这篇文章来获取更多信息：[`www.sfml-dev.org/tutorials/2.5/window-opengl.php`](https://www.sfml-dev.org/tutorials/2.5/window-opengl.php)。

一个应用程序可以有多个着色器。然后我们可以将不同的着色器附加到不同的游戏对象上以创建所需的效果。在这个游戏中，我们只有一个顶点着色器和一个片段着色器。我们将将其应用于每一帧，以及背景。

然而，当你看到如何将着色器附加到 `draw` 调用时，很明显可以轻松地拥有更多着色器。

我们将遵循以下步骤：

1.  首先，我们需要为将在 GPU 上执行的着色器代码。

1.  然后，我们需要编译这段代码。

1.  最后，我们需要将着色器附加到我们游戏引擎的绘制函数中适当的 `draw` 函数调用。

GLSL 是一种语言，它也有自己的类型，以及这些类型的变量，可以声明和使用。此外，我们可以从我们的 C++ 代码中与着色器程序的变量进行交互。

正如我们将看到的，GLSL 有一些与 C++ 的语法相似之处。

## 编写片段着色器

这里是 `shaders` 文件夹中 `rippleShader.frag` 文件的代码。我们不需要编写这个，因为它包含在我们之前添加的资产中，在 *第十四章*，*抽象和代码管理 – 更好地利用面向对象编程*：

```cpp
// attributes from vertShader.vert
varying vec4 vColor;
varying vec2 vTexCoord;
// uniforms
uniform sampler2D uTexture;
uniform float uTime;
void main() {
    float coef = sin(gl_FragCoord.y * 0.1 + 1 * uTime);
    vTexCoord.y +=  coef * 0.03;
    gl_FragColor = vColor * texture2D(uTexture, vTexCoord);
}
```

前四行（不包括注释）是片段着色器将使用的变量，但它们不是普通变量。我们首先看到的是 `varying` 类型。这些变量在两个 `shaders` 之间都有作用域。接下来，我们有 `uniform` 变量。这些变量可以直接从我们的 C++ 代码中操作。我们很快就会看到如何做到这一点。

除了 `varying` 和 `uniform` 类型之外，每个变量还有一个更传统的类型，它定义了实际的数据，如下所示：

+   `vec4` 是一个包含四个值的向量。

+   `vec2` 是一个包含两个值的向量。

+   `sampler2d` 将保存一个纹理。

+   `float` 就像 C++ 中的 `float 数据类型`。

`main` 函数内部的代码被执行。如果我们仔细查看 `main` 中的代码，我们会看到正在使用的每个变量。这段代码的确切功能超出了本书的范围。然而，总的来说，纹理坐标（`vTexCoord`）和像素/片段的颜色（`glFragColor`）通过几个数学函数和操作进行操作。请记住，这将在我们游戏的每一帧中调用的 `draw` 函数涉及的每个像素上执行。此外，请注意 `uTime` 在每一帧中传递的值都不同。结果，正如我们很快将看到的，将会产生波纹效果。

## 编写顶点着色器

这里是 `vertShader.vert` 文件的代码。您不需要编写这个。它包含在我们之前添加的资产中，在 *第十四章*，*抽象和代码管理 – 更好地利用面向对象编程*：

```cpp
//varying "out" variables to be used in the fragment shader
varying vec4 vColor;
varying vec2 vTexCoord;

void main() {
    vColor = gl_Color;
    vTexCoord = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
}
```

首先，注意两个 `varying` 变量。这些正是我们在片段着色器中操作的变量。在 `main` 函数中，代码操作每个顶点的位置。代码的工作原理超出了本书的范围，但幕后有一些相当深入的数学运算。如果您对此感兴趣，那么进一步探索 GLSL 将会非常有趣。

现在我们有了两个着色器（一个片段和一个顶点），我们可以在游戏中使用它们。

## 将着色器添加到引擎类中

打开 `Engine.h` 文件。添加以下高亮显示的代码行，它将一个名为 `m_RippleShader` 的 SFML `Shader` 实例添加到 `Engine` 类中：

```cpp
// Three views for the background
View m_BGMainView;
View m_BGLeftView;
View m_BGRightView;
View m_HudView;
// Declare a sprite and a Texture for the background
Sprite m_BackgroundSprite;
Texture m_BackgroundTexture;
// Declare a shader for the background
Shader m_RippleShader;
// Is the game currently playing?
bool m_Playing = false;
// Is character 1 or 2 the current focus?
bool m_Character1 = true;
```

引擎对象及其所有功能现在都可以访问 `m_RippleShader`。请注意，一个 SFML `Shader` 对象将包含着色器代码文件。

## 加载着色器

添加以下代码，检查玩家的 GPU 是否可以处理着色器。如果不行，游戏将退出。

小贴士

如果你的 GPU 不能处理着色器，你需要有一台非常旧的 PC 才能使这个功能不起作用。如果你有一个不支持着色器的 GPU，请接受我的道歉。

接下来，我们将添加一个 `else` 子句，如果系统可以处理着色器，则加载着色器。打开 `Engine.cpp` 文件，并将以下代码添加到构造函数中：

```cpp
// Can this graphics card use shaders?
if (!sf::Shader::isAvailable())
{
    // Time to get a new PC
    // Or remove all the shader related code L
    m_Window.close();
}
else
{
    // Load two shaders (1 vertex, 1 fragment)
    m_RippleShader.loadFromFile("shaders/vertShader.vert",
        "shaders/rippleShader.frag");
}
m_BackgroundTexture = TextureHolder::GetTexture(
    "graphics/background.png");
```

我们几乎可以看到我们的涟漪效果在行动了。

## 更新和绘制着色器

打开 `Draw.cpp` 文件。正如我们在编写着色器时已经讨论过的，我们将直接从我们的 C++代码中每帧更新 `uTime` 变量。我们将使用 `setParameter` 函数这样做。

添加以下高亮显示的代码以更新着色器的 `uTime` 变量，并更改对 `m_BackgroundSprite` 的 `draw` 调用，在每个可能的绘制场景中：

```cpp
void Engine::draw()
{
    // Rub out the last frame
    m_Window.clear(Color::White);
    // Update the shader parameters
m_RippleShader.setUniform("uTime", 
      m_GameTimeTotal.asSeconds());
    if (!m_SplitScreen)
    {
        // Switch to background view
        m_Window.setView(m_BGMainView);
        // Draw the background
        //m_Window.draw(m_BackgroundSprite);
        // Draw the background, complete with shader effect
        m_Window.draw(m_BackgroundSprite, &m_RippleShader);
        // Switch to m_MainView
        m_Window.setView(m_MainView);
        // Draw the Level
        m_Window.draw(m_VALevel, &m_TextureTiles);
        // Draw thomas
        m_Window.draw(m_Thomas.getSprite());
        // Draw thomas
        m_Window.draw(m_Bob.getSprite());
        // Draw the particle system
        if (m_PS.running())
        {
            m_Window.draw(m_PS);
        }
    }
    else
    {
        // Split-screen view is active
        // First draw Thomas' side of the screen
        // Switch to background view
        m_Window.setView(m_BGLeftView);
        // Draw the background
        //m_Window.draw(m_BackgroundSprite);
        // Draw the background, complete with shader effect
        m_Window.draw(m_BackgroundSprite, &m_RippleShader);
        // Switch to m_LeftView
        m_Window.setView(m_LeftView);
        // Draw the Level
        m_Window.draw(m_VALevel, &m_TextureTiles);

        // Draw thomas
        m_Window.draw(m_Bob.getSprite());
        // Draw thomas
        m_Window.draw(m_Thomas.getSprite());
        // Draw the particle system
        if (m_PS.running())
        {
            m_Window.draw(m_PS);
        }

        // Now draw Bob's side of the screen
        // Switch to background view
        m_Window.setView(m_BGRightView);
        // Draw the background
        //m_Window.draw(m_BackgroundSprite);
        // Draw the background, complete with shader effect
        m_Window.draw(m_BackgroundSprite, &m_RippleShader);
        // Switch to m_RightView
        m_Window.setView(m_RightView);
        // Draw the Level
        m_Window.draw(m_VALevel, &m_TextureTiles);
        // Draw thomas
        m_Window.draw(m_Thomas.getSprite());
        // Draw bob
        m_Window.draw(m_Bob.getSprite());
        // Draw the particle system
        if (m_PS.running())
        {
            m_Window.draw(m_PS);
        }                
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
}
```

最好删除被注释掉的代码行。

运行游戏，你将得到一种神秘的熔岩岩石效果。尝试更改背景图像以获得乐趣：

![](img/B14278_18_03.jpg)

就这样！我们的第四个游戏完成了。

# 摘要

在本章中，我们探讨了粒子系统和着色器的概念。尽管我们可能已经看到了每个的最简单案例，但我们仍然设法创建了一个简单的爆炸和一个神秘的熔岩岩石效果。

在接下来的四章中，我们将探讨更多使用设计模式改进我们代码的方法，同时构建《太空侵略者》游戏。
