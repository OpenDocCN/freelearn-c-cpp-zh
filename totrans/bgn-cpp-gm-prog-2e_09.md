# *第八章*：SFML 视图 – 启动僵尸射击器游戏

在这个项目中，我们将更多地使用`View`类。这个多才多艺的类将使我们能够轻松地将游戏划分为不同方面的层。在僵尸射击器项目中，我们将有一个用于 HUD 的层和一个用于主游戏的层。这是必要的，因为随着玩家每次清除一波僵尸时游戏世界都会扩大，最终游戏世界将比屏幕大，需要滚动。使用`View`类将防止 HUD 的文本与背景一起滚动。在下一个项目中，我们将更进一步，使用 SFML 的`View`类创建一个合作分屏游戏，`View`类将完成大部分繁重的工作。

这就是我们将在本章中要做的事情：

+   规划并启动僵尸竞技场游戏

+   编码`Player`类

+   了解 SFML 的`View`类

+   构建僵尸竞技场游戏引擎

+   使用`Player`类

# 规划并启动僵尸竞技场游戏

到目前为止，如果你还没有看过，我建议你去看一下*Over 9000 Zombies* ([`store.steampowered.com/app/273500/`](http://store.steampowered.com/app/273500/)) 和 *Crimson Land* ([`store.steampowered.com/app/262830/`](http://store.steampowered.com/app/262830/)) 的视频。显然，我们的游戏不会像这两个例子那样深入或高级，但我们将拥有相同的基本功能集和游戏机制，如下所示：

+   一个显示详细信息如分数、最高分、弹夹中的子弹数量、剩余子弹数量、玩家生命值和剩余待杀僵尸数量的抬头显示（HUD）。

+   玩家将在疯狂地逃离僵尸的同时射击它们。

+   使用*WASD*键盘键在移动的同时，用鼠标瞄准枪支。

+   在每个关卡之间，玩家将选择一个“升级”，这将影响玩家为了获胜而需要玩游戏的方式。

+   玩家需要收集“拾取物”来恢复生命值和弹药。

+   每一波都会带来更多的僵尸和更大的竞技场，使其更具挑战性。

将有三种类型的僵尸可供击杀。它们将具有不同的属性，如外观、生命值和速度。我们将它们称为追逐者、膨胀者和爬行者。查看以下带有注释的游戏截图，以了解一些功能在实际操作中的表现以及构成游戏的组件和资产：

![图片](img/B14278_08_01.jpg)

下面是关于每个编号点的更多信息：

1.  分数和最高分。这些，连同 HUD 的其他部分，将绘制在一个单独的层上，称为视图，并由`View`类的实例表示。最高分将被保存并加载到文件中。

1.  一个纹理，将在竞技场周围建造墙壁。这个纹理包含在一个名为**精灵图集**的单个图形中，以及其他背景纹理（编号**3**、**5**和**6**）。

1.  来自精灵图的第一个泥地纹理。

1.  这是一个“弹药拾取”。当玩家获得这个时，他们将会获得更多的弹药。还有一个“健康拾取”，玩家将从中获得更多的生命。这些拾取可以在僵尸波之间由玩家选择升级。

1.  来自精灵图的草地纹理。

1.  来自精灵图的第二个泥地纹理。

1.  僵尸曾经所在的地方的血溅。

1.  HUD 的底部部分。从左到右，有一个代表弹药、弹夹中的子弹数量、备用子弹数量、生命条、当前僵尸波和当前波剩余僵尸数量的图标。

1.  玩家的角色。

1.  准星，玩家用鼠标瞄准。

1.  一个缓慢移动但强大的“浮肿僵尸”。

1.  一个稍微快一点的移动但较弱的“爬行僵尸”。还有一个非常快且弱的“追逐僵尸”。不幸的是，在他们都被杀死之前，我无法在截图中获得一个。

因此，我们有很多事情要做，还有很多新的 C++技能要学习。让我们从创建一个新项目开始。

## 创建新项目

由于创建项目是一个相对复杂的过程，我将再次详细说明所有步骤。对于更多细节和图片，请参阅*第一章*，*C++、SFML、Visual Studio 和开始第一个游戏*中的*设置 Timber 项目*部分。

由于设置项目是一个繁琐的过程，我们将一步一步地进行，就像我们在 Timber 项目中做的那样。我不会展示与 Timber 项目相同的图片，但过程是相同的，所以如果你想提醒各种项目属性的位置，请翻回*第一章*，*C++、SFML、Visual Studio 和开始第一个游戏*。让我们看看以下步骤：

1.  启动 Visual Studio 并点击**创建新项目**按钮。如果你有其他项目打开，你可以选择**文件** | **新建项目**。

1.  在下一个显示的窗口中，选择**控制台应用程序**并点击**下一步**按钮。然后你会看到**配置你的新项目**窗口。

1.  在**项目** **名称**字段中的`Zombie Arena`。

1.  在`VS Projects`文件夹中。

1.  选择**将解决方案和项目放在同一目录下**的选项。

1.  当你完成前面的步骤后，点击**创建**。

1.  现在，我们将配置项目以使用我们放在`SFML`文件夹中的 SFML 文件。从主菜单中选择**项目** | **僵尸竞技场属性…**。在这个阶段，你应该已经打开了**僵尸竞技场属性页**窗口。

1.  在**僵尸竞技场属性页**窗口中，执行以下步骤。从**配置：**下拉菜单中选择**所有配置**。

1.  现在，从左侧菜单中选择**C/C++**然后选择**常规**。

1.  接下来，定位到 `\SFML\include`。如果你将 `SFML` 文件夹位于你的 D 驱动器上，要输入的完整路径将是 `D:\SFML\include`。如果你将 SFML 安装在不同的驱动器上，请更改你的路径。

1.  点击 **应用** 以保存到目前为止的配置。

1.  现在，仍然在同一窗口中，执行以下下一步。从左侧菜单中选择 **链接器** 然后选择 **常规**。

1.  现在，找到 `SFML` 文件夹，然后是 `\SFML\lib`。所以，如果你将 `SFML` 文件夹位于你的 D 驱动器上，要输入的完整路径将是 `D:\SFML\lib`。如果你将 SFML 安装在不同的驱动器上，请更改你的路径。

1.  点击 **应用** 以保存到目前为止的配置。

1.  接下来，仍然在同一窗口中，执行以下步骤。将 **配置** 下拉菜单切换到 **调试**，因为我们将在调试模式下运行和测试 Pong。

1.  选择 **链接器** 然后选择 **输入**。

1.  找到 `sfml-graphics-d.lib;sfml-window-d.lib;sfml-system-d.lib;sfml-network-d.lib;sfml-audio-d.lib;`。请格外小心地将光标放在编辑框当前内容的起始位置，以免覆盖任何已存在的文本。

1.  点击 **确定**。

1.  点击 **应用** 然后点击 **确定**。

现在，你已经配置了项目属性，你几乎准备就绪了。接下来，我们需要按照以下步骤将 SFML `.dll` 文件复制到主项目目录中：

1.  我的主要项目目录是 `D:\VS Projects\Zombie Arena`。这个文件夹是在之前的步骤中由 Visual Studio 创建的。如果你将你的 `Projects` 文件夹放在其他地方，那么在你的目录中执行此步骤。我们需要复制到项目文件夹中的文件位于你的 `SFML\bin` 文件夹中。为这两个位置打开一个窗口，并突出显示所有的 `.dll` 文件。

1.  现在，将高亮显示的文件复制到项目中。

项目现在已经设置好并准备就绪。接下来，我们将探索并添加项目资源。

## 项目资源

与之前游戏相比，这个项目中的资源更多样化和丰富。资源包括以下内容：

+   屏幕上文本所需的字体

+   不同动作的音效，如射击、装弹或被僵尸击中

+   角色图形、僵尸图形以及各种背景纹理的精灵图

游戏所需的所有图形和音效都包含在下载包中。它们分别位于 `第八章/graphics` 和 `第八章/sound` 文件夹中。

所需的字体尚未提供。这是为了避免任何关于许可的歧义。这不会造成问题，因为将提供下载字体以及如何和在哪里选择字体的链接。

## 探索资源

图形资源构成了我们僵尸竞技场游戏的场景部分。看看以下图形资源；你应该能清楚地知道游戏中的资源将如何使用：

![](img/B14278_08_03.jpg)

然而，可能不那么明显的是 `background_sheet.png` 文件，它包含四幅不同的图像。这是我们之前提到的精灵图集。我们将在 *第九章*，*C++ 参考，精灵图集和顶点数组* 中看到如何使用精灵图集来节省内存并提高游戏速度。

所有声音文件都采用 `.wav` 格式。这些文件包含在触发某些事件时将播放的声音效果。具体如下：

+   `hit.wav`：僵尸与玩家接触时播放的声音。

+   `pickup.wav`：当玩家碰撞或踩到（收集）健康提升（拾取）时播放的声音。

+   `powerup.wav`：当玩家在每一波僵尸之间选择一个属性来增强他们的力量（升级）时播放的声音。

+   `reload.wav`：一个令人满意的点击声，让玩家知道他们已经装上了新的弹药。

+   `reload_failed.wav`：一个不那么令人满意的音效，表示未能装上新子弹。

+   `shoot.wav`：射击声音。

+   `splat.wav`：僵尸被子弹击中的声音。

一旦您决定使用哪些资产，就是时候将它们添加到项目中。

## 将资产添加到项目中

以下说明将假设您正在使用书中提供的下载包中的所有资产。如果您使用自己的资产，只需用您自己的相应声音或图形文件替换，使用相同的文件名。让我们看看步骤：

1.  浏览到 `D:\VS Projects\ZombieArena`。

1.  在此文件夹内创建三个新文件夹，分别命名为 `graphics`、`sound` 和 `fonts`。

1.  从下载包中，将 `Chapter 8/graphics` 的全部内容复制到 `D:\VS Projects\ZombieArena\graphics` 文件夹。

1.  从下载包中，将 `Chapter 6/sound` 的全部内容复制到 `D:\VS Projects\ZombieArena\sound` 文件夹。

1.  现在，在您的网络浏览器中访问 [`www.1001freefonts.com/zombie_control.font`](http://www.1001freefonts.com/zombie_control.font) 并下载 **Zombie Control** 字体。

1.  解压下载内容，并将 `zombiecontrol.ttf` 文件添加到 `D:\VS Projects\ZombieArena\fonts` 文件夹。

现在，是时候考虑面向对象编程如何帮助我们完成这个项目了，然后我们可以开始编写僵尸竞技场的代码。

# 面向对象编程和僵尸竞技场项目

我们面临的首要问题是当前项目的复杂性。让我们考虑只有一个僵尸的情况；以下是使其在游戏中运行所需的内容：

+   它的水平和垂直位置

+   它的大小

+   它面对的方向

+   每种僵尸类型不同的纹理

+   一个精灵

+   每种僵尸类型不同的速度

+   每种僵尸类型不同的健康值

+   跟踪每种僵尸的类型

+   碰撞检测数据

+   它的智能（追逐玩家），对于每种僵尸类型略有不同

+   一个指示僵尸是活着还是死了的标志

这可能意味着对于一个僵尸就需要十几个变量，而管理一群僵尸则需要每个变量的整个数组。但是，对于机枪的所有子弹、拾取物品以及不同等级的提升呢？简单的 Timber!!!和 Pong 游戏也开始变得难以管理，很容易推测这个更复杂的射击游戏将会更加难以控制！

幸运的是，我们将把在前两个章节中学到的所有面向对象编程技能付诸实践，并学习一些新的 C++技术。

我们将从这个项目开始编写代表玩家的类。

# 构建玩家类——第一个类

让我们思考一下`Player`类需要做什么，以及我们对其的要求。这个类需要知道它能以多快的速度移动，它在游戏世界中的当前位置，以及它有多少健康值。由于`Player`类在玩家眼中被表示为一个二维图形角色，这个类将需要一个`Sprite`对象和一个`Texture`对象。

此外，尽管现在可能不明显，但我们的`Player`类也将从了解游戏运行的整体环境的一些细节中受益。这些细节包括屏幕分辨率、组成竞技场的瓦片大小以及当前竞技场的整体大小。

由于`Player`类将负责在每一帧中更新自己（就像蝙蝠和球一样），它需要知道玩家在任何给定时刻的意图。例如，玩家当前是否按下了键盘方向键？或者玩家当前是否按下了多个键盘方向键？布尔变量用于确定*W*、*A*、*S*和*D*键的状态，并将是必不可少的。

很明显，我们将在新类中需要相当多的变量。在学到了所有关于面向对象编程的知识后，我们当然会把这些变量都设置为私有。这意味着在适当的地方，我们必须提供从`main`函数访问的权限。

我们将使用大量的 getter 函数以及一些设置对象状态的函数。这些函数数量相当多。这个类中有 21 个函数。一开始，这可能会显得有些令人畏惧，但我们将逐一过目，并会发现它们中的大多数只是设置或获取一个私有变量。

其中只有几个深入的功能：`update`，它将从`main`函数中每帧被调用一次，以及`spawn`，它将处理每次玩家被创建时初始化一些私有变量。然而，正如我们将看到的，它们并没有什么复杂的地方，并且它们都将被详细描述。

进行编码的最佳方式是编写头文件。这将给我们机会看到所有的私有变量并检查所有的函数签名。

小贴士

请密切注意返回值和参数类型，因为这会使理解函数定义中的代码变得容易得多。

## 编写玩家类头文件

首先右键单击 `Player.h`。最后，点击 **添加** 按钮。我们现在可以开始编写我们第一个类的头文件了。

通过添加声明，包括开闭花括号，然后加上分号来开始编写 `Player` 类：

```cpp
#pragma once
#include <SFML/Graphics.hpp>
using namespace sf;
class Player
{
};
```

现在，让我们将所有我们的私有成员变量添加到文件中。根据我们之前讨论的内容，看看你是否能弄清楚每个变量将做什么。我们稍后会逐一介绍：

```cpp
class Player
{
private:
    const float START_SPEED = 200;
    const float START_HEALTH = 100;
    // Where is the player
    Vector2f m_Position;
    // Of course, we will need a sprite
    Sprite m_Sprite;
    // And a texture
    // !!Watch this space – Interesting changes here soon!!
    Texture m_Texture;
    // What is the screen resolution
    Vector2f m_Resolution;
    // What size is the current arena
    IntRect m_Arena;
    // How big is each tile of the arena
    int m_TileSize;
    // Which direction(s) is the player currently moving in
    bool m_UpPressed;
    bool m_DownPressed;
    bool m_LeftPressed;
    bool m_RightPressed;
    // How much health has the player got?
    int m_Health;
    // What is the maximum health the player can have
    int m_MaxHealth;
    // When was the player last hit
    Time m_LastHit;
    // Speed in pixels per second
    float m_Speed;
// All our public functions will come next
};
```

之前的代码声明了所有我们的成员变量。其中一些是常规变量，而另一些是对象。请注意，它们都在类的 `private:` 部分下，因此不能从类外部直接访问。

此外，请注意我们正在使用命名约定，即给所有非常量变量的名称前缀为 `m_`。这个 `m_` 前缀将在编写函数定义时提醒我们，它们是成员变量，与我们在某些函数中创建的局部变量不同，也与函数参数不同。

所使用的所有变量都非常直接，例如 `m_Position`、`m_Texture` 和 `m_Sprite`，分别代表玩家的当前位置、纹理和精灵。除此之外，每个变量（或变量组）都有注释，以便清楚地说明其用法。

然而，为什么它们是必需的，以及它们将用于什么上下文，可能并不那么明显。例如，`m_LastHit` 是 `Time` 类型的对象，用于记录玩家最后一次被僵尸击中的时间。我们可能需要这个信息的 `why` 并不明显，但我们会很快讨论这个问题。

当我们将游戏的其余部分拼凑起来时，每个变量的上下文将变得更加清晰。现在的重要事情是熟悉名称和数据类型，以便轻松地跟随整个项目的其余部分。

小贴士

你不需要记住变量名称和类型，因为当它们被使用时我们会讨论所有代码。然而，你需要花时间仔细查看它们，并更多地熟悉它们。此外，随着我们的进展，如果任何内容似乎不清楚，参考这个头文件可能是有价值的。

现在，我们可以添加一个完整的函数列表。添加以下突出显示的代码，看看你是否能弄清楚它都做了什么。请密切注意返回类型、参数和每个函数的名称。这是理解我们将在这个项目的其余部分编写的代码的关键。它们告诉我们关于每个函数的什么？添加以下突出显示的代码，然后我们将检查它：

```cpp
// All our public functions will come next
public:
    Player();
    void spawn(IntRect arena, Vector2f resolution, int tileSize);
    // Call this at the end of every game
    void resetPlayerStats();

    // Handle the player getting hit by a zombie
    bool hit(Time timeHit);
    // How long ago was the player last hit
    Time getLastHitTime();
    // Where is the player
    FloatRect getPosition();
    // Where is the center of the player
    Vector2f getCenter();
    // What angle is the player facing
    float getRotation();
    // Send a copy of the sprite to the main function
    Sprite getSprite();
    // The next four functions move the player
    void moveLeft();
    void moveRight();
    void moveUp();
    void moveDown();
    // Stop the player moving in a specific direction
    void stopLeft();
    void stopRight();
    void stopUp();
    void stopDown();
    // We will call this function once every frame
    void update(float elapsedTime, Vector2i mousePosition);
    // Give the player a speed boost
    void upgradeSpeed();
    // Give the player some health
    void upgradeHealth();
    // Increase the maximum amount of health the player can have
    void increaseHealthLevel(int amount);
    // How much health has the player currently got?
    int getHealth();
};
```

首先，请注意，所有这些函数都是公开的。这意味着我们可以使用`main`函数中的类实例调用所有这些函数，代码如下：

```cpp
player.getSprite();
```

假设`player`是`Player`类的完整配置实例，之前的代码将返回`m_Sprite`的副本。将此代码放入实际上下文中，我们可以在`main`函数中编写如下代码：

```cpp
window.draw(player.getSprite());
```

之前的代码会在正确的位置绘制玩家图形，就像在`main`函数中直接声明精灵一样。这就是我们在 Pong 项目中使用`Bat`类所做的那样。

在我们将要实现（即编写相应的`.cpp`文件中的定义）这些函数之前，让我们逐一仔细看看它们：

+   `void spawn(IntRect arena, Vector2f resolution, int tileSize)`: 这个函数做它名字暗示的事情。它将准备对象以便使用，包括将其放置在起始位置（即生成）。请注意，它不返回任何数据，但它有三个参数。它接收一个名为`arena`的`IntRect`实例，这将表示当前级别的尺寸和位置；一个包含屏幕分辨率的`Vector2f`实例；以及一个整数，它将包含背景瓷砖的大小。

+   `void resetPlayerStats`: 一旦我们赋予玩家在波次之间升级的能力，我们将在新游戏开始时需要能够取消/重置这些能力。

+   `Time getLastHitTime()`: 这个函数只做一件事——它返回玩家最后一次被僵尸击中的时间。当检测碰撞时，我们将使用这个函数，并且它将确保玩家不会因为与僵尸接触而频繁受到惩罚。

+   `FloatRect getPosition()`: 这个函数返回一个`FloatRect`实例，描述了包含玩家图形的矩形的水平和垂直浮点坐标。这对于碰撞检测也很有用。

+   `Vector2f getCenter()`: 这与`getPosition`略有不同，因为它是一个`Vector2f`类型，只包含玩家图形中心的*x*和*y*位置。

+   `float getRotation()`: `main`函数中的代码有时需要知道，以度为单位，玩家当前面向的方向。3 点钟是 0 度，顺时针增加。

+   `Sprite getSprite()`: 正如我们之前讨论的，这个函数返回代表玩家的精灵的副本。

+   `void moveLeft()`, `..Right()`, `..Up()`, `..Down()`: 这四个函数没有返回类型或参数。它们将从`main`函数中调用，然后`Player`类将能够在按下一个或多个*WASD*键时采取行动。

+   `void stopLeft()`, `..Right()`, `..Up()`, `..Down()`: 这四个函数没有返回类型或参数。它们将从`main`函数中调用，然后`Player`类将能够在释放一个或多个*WASD*键时采取行动。

+   `void update(float elapsedTime, Vector2i mousePosition)`: 这将是整个类中唯一的长函数。它将每帧从 `main` 中调用一次。它将执行所有必要的操作，以确保 `player` 对象的数据被更新，以便进行碰撞检测和绘制。注意，它不返回任何数据，但接收自上一帧以来经过的时间量，以及一个 `Vector2i` 实例，它将包含鼠标指针/十字准线的水平和垂直屏幕位置。

    重要提示

    注意，这些是整数屏幕坐标，与浮点世界坐标不同。

+   `void upgradeSpeed()`: 一个可以在升级屏幕上调用，当玩家选择让玩家跑得更快时的函数。

+   `void upgradeHealth()`: 另一个可以在升级屏幕上调用，当玩家选择让玩家更强（即拥有更多健康）时的函数。

+   `void increaseHealthLevel(int amount)`: 与之前的函数相比，这个函数会增加玩家拥有的健康量，直到达到当前设定的最大值。这个函数将在玩家拾取健康物品时使用。

+   `int getHealth()`: 由于健康水平是如此动态，我们需要能够确定玩家在任何给定时刻的健康量。这个函数返回一个 `int`，它包含这个值。

就像变量一样，现在应该很清楚每个函数的作用。同时，使用这些函数的 *原因* 和精确的上下文也只有在项目进展过程中才会逐渐显现。

小贴士

你不需要记住函数名、返回类型或参数，因为当它们被使用时我们会讨论代码。然而，你需要花时间仔细查看它们，结合之前的解释，并熟悉它们。此外，随着项目的进行，如果任何内容似乎不清楚，参考这个头文件可能会有所帮助。

现在，我们可以继续到函数的核心部分：定义。

## 编写 Player 类函数定义

最后，我们可以开始编写执行我们类工作的代码。

在 `Player.cpp` 上 *右键点击*。最后，点击 **添加** 按钮。

小贴士

从现在起，我将简单地要求你创建一个新的类或头文件。所以，记住前面的步骤，或者如果需要提醒，请参考这里。

我们现在可以开始为这个项目中第一个类的 `.cpp` 文件编写代码了。

下面是必要的包含指令，接着是构造函数的定义。记住，构造函数将在我们首次实例化 `Player` 类型的对象时被调用。将以下代码添加到 `Player.cpp` 文件中，然后我们可以更仔细地查看它：

```cpp
#include "player.h"
Player::Player()
{
    m_Speed = START_SPEED;
    m_Health = START_HEALTH;
    m_MaxHealth = START_HEALTH;
    // Associate a texture with the sprite
    // !!Watch this space!!
    m_Texture.loadFromFile("graphics/player.png");
    m_Sprite.setTexture(m_Texture);
    // Set the origin of the sprite to the center, 
    // for smooth rotation
    m_Sprite.setOrigin(25, 25);
}
```

在构造函数中，它当然与类名相同且没有返回类型，我们编写代码来开始设置 `Player` 对象，使其准备好使用。

要清晰明了；此代码将在我们从`main`函数中编写以下代码时运行：

```cpp
Player player;
```

不要立即添加上一行的代码。

在构造函数中，我们只是从相关常量初始化`m_Speed`、`m_Health`和`m_MaxHealth`。然后，我们将玩家图形加载到`m_Texture`中，将`m_Texture`与`m_Sprite`关联，并将`m_Sprite`的原点设置为中心，`(25, 25)`。

小贴士

注意到这个神秘的注释`// !!Watch this space!!`，它表明我们将返回到加载我们的纹理以及与之相关的一些重要问题。一旦我们发现问题并学习更多 C++，我们最终将改变我们处理这个纹理的方式。我们将在*第十章**，指针、标准模板库和纹理管理*中这样做。

接下来，我们将编写`spawn`函数。我们只会创建一个`Player`类的实例。然而，我们需要在每一波中将其生成到当前关卡中。这就是`spawn`函数为我们处理的事情。将以下代码添加到`Player.cpp`文件中，并确保检查细节并阅读注释：

```cpp
void Player::spawn(IntRect arena, 
        Vector2f resolution, 
        int tileSize)
{
    // Place the player in the middle of the arena
    m_Position.x = arena.width / 2;
    m_Position.y = arena.height / 2;
    // Copy the details of the arena 
    // to the player's m_Arena
    m_Arena.left = arena.left;
    m_Arena.width = arena.width;
    m_Arena.top = arena.top;
    m_Arena.height = arena.height;
    // Remember how big the tiles are in this arena
    m_TileSize = tileSize;
    // Store the resolution for future use
    m_Resolution.x = resolution.x;
    m_Resolution.y = resolution.y;
}
```

前面的代码首先将`m_Position.x`和`m_Position.y`的值初始化为传入的`arena`高度和宽度的一半。这会将玩家移动到关卡的中心，无论其大小如何。

接下来，我们将传入的`arena`的所有坐标和尺寸复制到相同类型的成员对象`m_Arena`中。当前竞技场的尺寸和坐标被频繁使用，因此这样做是有意义的。现在我们可以使用`m_Arena`来执行诸如确保玩家不能穿过墙壁等任务。此外，我们将传入的`tileSize`实例复制到成员变量`m_TileSize`中，出于相同的目的。我们将在`update`函数中看到`m_Arena`和`m_TileSize`的实际应用。

前面的代码的最后两行将屏幕分辨率从`Vector2f`的`resolution`（`spawn`的参数）复制到`m_Resolution`（`Player`的成员变量）。现在我们可以在`Player`类内部访问这些值。

现在，添加`resetPlayerStats`函数的非常直接的代码：

```cpp
void Player::resetPlayerStats()
{
    m_Speed = START_SPEED;
    m_Health = START_HEALTH;
    m_MaxHealth = START_HEALTH;
}
```

当玩家死亡时，我们将使用此代码来重置他们可能使用的任何升级。

我们不会在接近完成项目之前编写调用`resetPlayerStats`函数的代码，但它已经准备好了，以备我们使用。

在代码的下一部分，我们将添加两个额外的函数。它们将处理玩家被僵尸击中的情况。我们将能够调用`player.hit()`并传入当前游戏时间。我们还可以通过调用`player.getLastHitTime()`来查询玩家最后一次被击中的时间。这些函数的确切用途将在我们有僵尸时变得明显。

将两个新的定义添加到`Player.cpp`文件中，然后更仔细地检查 C++代码：

```cpp
Time Player::getLastHitTime()
{
    return m_LastHit;
}
bool Player::hit(Time timeHit)
{
    if (timeHit.asMilliseconds() 
        - m_LastHit.asMilliseconds() > 200)
    {
        m_LastHit = timeHit;
        m_Health -= 10;
        return true;
    }
    else
    {
        return false;
    }
}
```

`getLastHitTime()`的代码非常直接；它将返回存储在`m_LastHit`中的任何值。

`hit`函数稍微复杂一些，并且更加微妙。首先，`if`语句检查传入的参数时间是否比存储在`m_LastHit`中的时间晚 200 毫秒。如果是这样，`m_LastHit`将更新为传入的时间，`m_Health`的当前值将扣除 10 点。`if`语句中的最后一行代码是`return true`。注意，`else`子句只是简单地返回`false`给调用代码。

这个函数的整体效果是，玩家的健康点数每秒最多只能扣除五次。记住，我们的游戏循环可能每秒运行数千次迭代。在这种情况下，如果没有这个函数提供的限制，僵尸只需要与玩家接触一秒钟，就会扣除数万健康点数。`hit`函数控制并限制这种现象。它还通过返回`true`或`false`来让调用代码知道是否已注册新的打击（或没有）。

这段代码暗示我们将在`main`函数中检测僵尸与玩家之间的碰撞。然后我们将调用`player.hit()`来确定是否扣除任何健康点数。

接下来，对于`Player`类，我们将实现一系列的 getter 函数。这些函数允许我们保持数据在`Player`类中整洁地封装，同时使它们的值对`main`函数可用。

在之前的代码块之后添加以下代码：

```cpp
FloatRect Player::getPosition()
{
    return m_Sprite.getGlobalBounds();
}
Vector2f Player::getCenter()
{
    return m_Position;
}
float Player::getRotation()
{
    return m_Sprite.getRotation();
}
Sprite Player::getSprite()
{
    return m_Sprite;
}
int Player::getHealth()
{
    return m_Health;
}
```

之前的代码非常直接。之前的五个函数中的每一个都返回我们成员变量中的一个值。仔细观察每一个，熟悉哪个函数返回哪个值。

下面的八个简短函数启用了键盘控制（我们将在`main`函数中使用），以便我们可以更改我们的`Player`类型对象的包含数据。将以下代码添加到`Player.cpp`文件中，然后我们将总结它是如何工作的：

```cpp
void Player::moveLeft()
{
    m_LeftPressed = true;
}
void Player::moveRight()
{
    m_RightPressed = true;
}
void Player::moveUp()
{
    m_UpPressed = true;
}
void Player::moveDown()
{
    m_DownPressed = true;
}
void Player::stopLeft()
{
    m_LeftPressed = false;
}
void Player::stopRight()
{
    m_RightPressed = false;
}
void Player::stopUp()
{
    m_UpPressed = false;
}
void Player::stopDown()
{
    m_DownPressed = false;
}
```

之前的代码包含四个函数（`moveLeft`、`moveRight`、`moveUp`和`moveDown`），这些函数将相关的布尔变量（`m_LeftPressed`、`m_RightPressed`、`m_UpPressed`和`m_DownPressed`）设置为`true`。另外四个函数（`stopLeft`、`stopRight`、`stopUp`和`stopDown`）执行相反的操作，并将相同的布尔变量设置为`false`。现在，`Player`类的实例可以知道哪些*WASD*键被按下，哪些没有被按下。

以下函数是完成所有繁重工作的函数。`update` 函数将在游戏循环的每一帧中调用一次。添加以下代码，然后我们将详细检查它。如果我们跟随着之前的八个函数，并且记得我们是如何为 Timber!!! 项目动画云和蜜蜂，以及为 Pong 项目动画蝙蝠和球的，我们可能会理解以下代码的大部分内容：

```cpp
void Player::update(float elapsedTime, Vector2i mousePosition)
{
    if (m_UpPressed)
    {
        m_Position.y -= m_Speed * elapsedTime;
    }
    if (m_DownPressed)
    {
        m_Position.y += m_Speed * elapsedTime;
    }
    if (m_RightPressed)
    {
        m_Position.x += m_Speed * elapsedTime;
    }
    if (m_LeftPressed)
    {
        m_Position.x -= m_Speed * elapsedTime;
    }
    m_Sprite.setPosition(m_Position);
    // Keep the player in the arena
    if (m_Position.x > m_Arena.width - m_TileSize)
    {
        m_Position.x = m_Arena.width - m_TileSize;
    }
    if (m_Position.x < m_Arena.left + m_TileSize)
    {
        m_Position.x = m_Arena.left + m_TileSize;
    }
    if (m_Position.y > m_Arena.height - m_TileSize)
    {
        m_Position.y = m_Arena.height - m_TileSize;
    }
    if (m_Position.y < m_Arena.top + m_TileSize)
    {
        m_Position.y = m_Arena.top + m_TileSize;
    }
    // Calculate the angle the player is facing
    float angle = (atan2(mousePosition.y - m_Resolution.y / 2,
        mousePosition.x - m_Resolution.x / 2)
        * 180) / 3.141;
    m_Sprite.setRotation(angle);
}
```

上述代码的前一部分移动玩家精灵。四个 `if` 语句检查哪些与移动相关的布尔变量（`m_LeftPressed`、`m_RightPressed`、`m_UpPressed` 或 `m_DownPressed`）为真，并相应地更改 `m_Position.x` 和 `m_Position.y`。同样，从之前的两个项目中使用的公式来计算移动量也被使用：

**位置（+ 或 -）速度 * 经过的时间**。

在这四个 `if` 语句之后，调用 `m_Sprite.setPosition` 并传入 `m_Position`。精灵现在已经调整得恰到好处，以适应那一帧。

接下来的四个 `if` 语句检查 `m_Position.x` 或 `m_Position.y` 是否超出了当前竞技场的任何边缘。记住，当前竞技场的范围存储在 `m_Arena` 的 `spawn` 函数中。让我们看看这四个 `if` 语句中的第一个，以便理解它们：

```cpp
if (m_Position.x > m_Arena.width - m_TileSize)
{
    m_Position.x = m_Arena.width - m_TileSize;
}
```

上述代码测试 `m_position.x` 是否大于 `m_Arena.width` 减去瓦片的大小（`m_TileSize`）。当我们创建背景图形时，这个计算将检测玩家是否越界到墙壁。

当 `if` 语句为真时，使用 `m_Arena.width - m_TileSize` 的计算来初始化 `m_Position.x`。这意味着玩家图形的中心永远不会超出右侧墙壁的左侧边缘。

接下来的三个 `if` 语句，紧随我们刚刚讨论的那个语句之后，做的是同样的事情，但针对其他三面墙壁。

上述代码的最后两行计算并设置玩家精灵旋转到的角度（即面向）。这一行代码可能看起来有点复杂，但它只是使用准星的位置（`mousePosition.x` 和 `mousePosition.y`）以及屏幕中心（`m_Resolution.x` 和 `m_Resolution.y`）在一个经过验证的三角函数中。

`atan` 如何使用这些坐标以及 Pi（3.141）相当复杂，这就是为什么它被封装在一个方便的函数中供我们使用。

重要提示

如果你想更详细地探索三角函数，可以在这里进行：[`www.cplusplus.com/reference/cmath/`](http://www.cplusplus.com/reference/cmath/)。

我们将为 `Player` 类添加的最后三个函数使玩家速度提高 20%，增加玩家生命值 20%，以及分别增加传入的生命值。

在 `Player.cpp` 文件的末尾添加以下代码，然后我们将更仔细地查看它：

```cpp
void Player::upgradeSpeed()
{
    // 20% speed upgrade
    m_Speed += (START_SPEED * .2);
}
void Player::upgradeHealth()
{
    // 20% max health upgrade
    m_MaxHealth += (START_HEALTH * .2);
}
void Player::increaseHealthLevel(int amount)
{
    m_Health += amount;
    // But not beyond the maximum
    if (m_Health > m_MaxHealth)
    {
        m_Health = m_MaxHealth;
    }
}
```

在前面的代码中，`upgradeSpeed()`和`upgradeHealth()`函数分别增加`m_Speed`和`m_MaxHealth`中存储的值。这些值通过将起始值乘以.2 并加到当前值上来增加 20%。这些函数将在玩家在关卡之间选择他们希望提高的角色的属性（即升级）时从`main`函数中调用。

`increaseHealthLevel()`函数从`main`中的`amount`参数接收一个`int`值。这个`int`值将由一个名为`Pickup`的类提供，我们将在*第十一章**碰撞检测、拾取和子弹*中编写。`m_Health`成员变量会增加传入的值。然而，对于玩家来说有一个限制。`if`语句检查`m_Health`是否超过了`m_MaxHealth`，如果是，则将其设置为`m_MaxHealth`。这意味着玩家不能简单地从拾取中获得无限的生命值。相反，他们必须在关卡之间仔细平衡他们选择的升级。

当然，我们的`Player`类在实例化并放入游戏循环中工作之前什么都不能做。在我们这样做之前，让我们看看游戏摄像机的概念。

# 使用 SFML View 控制游戏摄像机

在我看来，SFML `View`类是最整洁的类之一。在完成这本书之后，当我们不使用媒体/游戏库制作游戏时，我们真的会注意到`View`的缺失。

`View`类允许我们将游戏视为在一个具有其自身属性的世界中进行，我的意思是什么？当我们创建游戏时，我们通常试图创建一个虚拟世界。这个虚拟世界很少，如果不是永远，以像素为单位来衡量，而且很少，如果不是永远，这个世界的像素数将与玩家的显示器相同。我们需要一种方法来抽象我们正在构建的虚拟世界，使其可以是我们想要的任何大小或形状。

将 SFML `View`视为玩家通过其观看我们虚拟世界一部分的摄像机是另一种思考方式。大多数游戏将拥有多个世界摄像机/视图。

例如，考虑一个分屏游戏，其中两个玩家可以在同一时间在世界中的不同部分。

或者，考虑一个游戏，其中屏幕上的一个小区域代表整个游戏世界，但以非常高的级别/缩放，就像一个迷你地图。

即使我们的游戏比前两个例子简单得多，不需要分屏或迷你地图，我们可能仍然希望创建一个比正在玩的游戏屏幕更大的世界。当然，Zombie Arena 就是这样。

此外，如果我们不断移动游戏摄像机以显示虚拟世界的不同部分（通常是为了跟踪玩家），那么 HUD 会发生什么？如果我们绘制分数和其他屏幕上的 HUD 信息，然后滚动世界以跟随玩家，分数就会相对于摄像机移动。

SFML 的`View`类很容易实现所有这些功能，并用非常直接的代码解决了这个问题。诀窍是为每个相机创建一个`View`实例——可能是一个用于迷你地图的`View`实例，一个用于滚动游戏世界的`View`实例，然后是一个用于 HUD 的`View`实例。

`View`实例可以根据需要移动、调整大小和定位。因此，跟随游戏的`main`视图可以跟踪玩家，迷你地图视图可以保持在屏幕的一个固定、缩小的角落，而 HUD 可以覆盖整个屏幕且不会移动，尽管主`View`实例可以跟随玩家移动。

让我们看看使用几个`View`实例的一些代码。

小贴士

这段代码被用来介绍`View`类。不要将此代码添加到僵尸竞技场项目中。

创建并初始化几个`View`实例：

```cpp
// Create a view to fill a 1920 x 1080 monitor
View mainView(sf::FloatRect(0, 0, 1920, 1080));
// Create a view for the HUD
View hudView(sf::FloatRect(0, 0, 1920, 1080));
```

之前的代码创建了两个填充 1920 x 1080 监视器的`View`对象。现在，我们可以用`mainView`做一些魔法，同时完全不动`hudView`：

```cpp
// In the update part of the game
// There are lots of things you can do with a View
// Make the view centre around the player                
mainView.setCenter(player.getCenter());
// Rotate the view 45 degrees
mainView.rotate(45)
// Note that hudView is totally unaffected by the previous code
```

当我们操作`View`实例的属性时，我们这样做。当我们向视图中绘制精灵、文本或其他对象时，我们必须明确地将视图**设置为**当前窗口的视图：

```cpp
// Set the current view
window.setView(mainView);
```

现在，我们可以将我们想要绘制的一切都绘制到这个视图中：

```cpp
// Do all the drawing for this view
window.draw(playerSprite);
window.draw(otherGameObject);
// etc
```

玩家的坐标可能是什么都行；这无关紧要，因为`mainView`是围绕图形居中的。

现在，我们可以将 HUD 绘制到`hudView`中。注意，就像我们从后往前在层中绘制单个元素（背景、游戏对象、文本等）一样，我们也会从后往前绘制视图。因此，HUD 是在主游戏场景之后绘制的：

```cpp
// Switch to the hudView
window.setView(hudView);
// Do all the drawing for the HUD
window.draw(scoreText);
window.draw(healthBar);
// etc
```

最后，我们可以以通常的方式绘制/显示窗口及其当前帧的所有视图：

```cpp
window.display();
```

小贴士

如果你想要将你对 SFML `View`的理解进一步扩展到这个项目所必需的范围之外，包括如何实现分屏和迷你地图，那么网上最好的指南是官方 SFML 网站：[`www.sfml-dev.org/tutorials/2.5/graphics-view.php`](https://www.sfml-dev.org/tutorials/2.5/graphics-view.php)。

现在我们已经了解了`View`，我们可以开始编写僵尸竞技场`main`函数，并真正使用我们的第一个`View`实例。在*第十二章*，*分层视图和实现 HUD*，我们将介绍`View`的第二个实例用于 HUD，并将其叠加在主`View`实例之上。

# 启动僵尸竞技场游戏引擎

在这个游戏中，我们在`main`中需要一个稍微升级的游戏引擎。我们将有一个名为`state`的枚举，它将跟踪游戏当前的状态。然后，在`main`的整个过程中，我们可以将我们的代码部分包裹起来，以便在不同的状态下发生不同的事情。

当我们创建项目时，Visual Studio 为我们创建了一个名为`ZombieArena.cpp`的文件。这个文件将包含我们的`main`函数以及实例化和控制所有类的代码。

我们从现在熟悉的`main`函数和一些包含指令开始。注意添加了`Player`类的包含指令。

将以下代码添加到`ZombieArena.cpp`文件中：

```cpp
#include <SFML/Graphics.hpp>
#include "Player.h"
using namespace sf;
int main()
{
    return 0;
}
```

之前的代码中没有新内容，除了`#include "Player.h"`这一行意味着我们现在可以在代码中使用`Player`类。

让我们进一步完善我们的游戏引擎。以下代码做了很多事情。当你添加代码时，务必阅读注释，以了解正在发生的事情。然后我们将更详细地讨论它。

在`main`函数的开始处添加以下高亮代码：

```cpp
int main()
{
    // The game will always be in one of four states
enum class State { PAUSED, LEVELING_UP, 
            GAME_OVER, PLAYING };

    // Start with the GAME_OVER state
    State state = State::GAME_OVER;
// Get the screen resolution and 
    // create an SFML window
    Vector2f resolution;
resolution.x = 
        VideoMode::getDesktopMode().width;
resolution.y = 
        VideoMode::getDesktopMode().height;
    RenderWindow window(
VideoMode(resolution.x, resolution.y), 
        "Zombie Arena", Style::Fullscreen);
    // Create a an SFML View for the main action
View mainView(sf::FloatRect(0, 0, 
            resolution.x, resolution.y));
    // Here is our clock for timing everything
    Clock clock;
    // How long has the PLAYING state been active
    Time gameTimeTotal;
// Where is the mouse in 
    // relation to world coordinates
    Vector2f mouseWorldPosition;
// Where is the mouse in 
    // relation to screen coordinates
    Vector2i mouseScreenPosition;
    // Create an instance of the Player class
    Player player;
    // The boundaries of the arena
    IntRect arena;
    // The main game loop
    while (window.isOpen())
    {

    }
    return 0;
}
```

让我们逐个检查我们输入的所有代码的每个部分。在`main`函数内部，我们有以下代码：

```cpp
// The game will always be in one of four states
enum class State { PAUSED, LEVELING_UP, GAME_OVER, PLAYING };
// Start with the GAME_OVER state
State state = State::GAME_OVER;
```

之前的代码创建了一个名为`State`的新枚举类。然后，代码创建了一个名为`state`的`State`类实例。现在，`state`枚举可以是以下四个值之一，如声明中定义的那样。这些值是`PAUSED`、`LEVELING_UP`、`GAME_OVER`和`PLAYING`。这四个值正是我们跟踪和响应游戏在任何给定时间可能处于的不同状态所需要的。请注意，`state`一次不可能持有多个值。

紧接着，我们添加了以下代码：

```cpp
// Get the screen resolution and create an SFML window
Vector2f resolution;
resolution.x = VideoMode::getDesktopMode().width;
resolution.y = VideoMode::getDesktopMode().height;
RenderWindow window(VideoMode(resolution.x, resolution.y), 
    "Zombie Arena", Style::Fullscreen);
```

之前的代码声明了一个名为`resolution`的`Vector2f`实例。我们通过调用`VideoMode::getDesktopMode`函数来初始化`resolution`的两个成员变量（`x`和`y`），用于`width`和`height`。现在，`resolution`对象持有游戏运行在的监视器的分辨率。最后一行代码使用适当的分辨率创建了一个名为`window`的新`RenderWindow`实例。

以下代码创建了一个 SFML `View`对象。视图最初位于监视器像素的精确坐标。如果我们使用这个`View`在这个当前位置进行绘图，它将等同于在没有视图的窗口中绘图。然而，我们最终将开始移动这个视图，以聚焦于玩家需要看到的游戏世界的部分。然后，当我们开始使用第二个`View`实例（用于 HUD 并保持固定）时，我们将看到这个`View`实例如何跟踪动作，而另一个保持静态以显示 HUD：

```cpp
// Create a an SFML View for the main action
View mainView(sf::FloatRect(0, 0, resolution.x, resolution.y));
```

接下来，我们创建了一个`Clock`实例来进行计时，并创建了一个名为`gameTimeTotal`的`Time`对象，它将记录已经过去的时间。随着项目的进展，我们还将引入更多的变量和对象来处理计时：

```cpp
// Here is our clock for timing everything
Clock clock;
// How long has the PLAYING state been active
Time gameTimeTotal;
```

以下代码声明了两个向量：一个包含两个`float`变量，称为`mouseWorldPosition`，另一个包含两个整数，称为`mouseScreenPosition`。鼠标指针有点特殊，因为它存在于两个不同的坐标空间中。如果我们愿意，可以将其视为平行宇宙。首先，当玩家在世界中移动时，我们需要跟踪准星在那个世界中的位置。这些将是浮点坐标，并将存储在`mouseWorldCoordinates`中。当然，显示器本身的实际像素坐标永远不会改变。它们始终是 0,0 到水平分辨率-1，垂直分辨率-1。我们将使用存储在`mouseScreenPosition`中的整数来跟踪相对于此坐标空间的鼠标指针位置：

```cpp
// Where is the mouse in relation to world coordinates
Vector2f mouseWorldPosition;
// Where is the mouse in relation to screen coordinates
Vector2i mouseScreenPosition;
```

最后，我们开始使用我们的`Player`类。这一行代码将导致构造函数（`Player::Player`）执行。如果您想刷新对这个函数的记忆，请参考`Player.cpp`：

```cpp
// Create an instance of the Player class
Player player;
```

这个`IntRect`对象将包含起始水平和垂直坐标，以及宽度和高度。一旦初始化，我们将能够通过代码如`arena.left`、`arena.top`、`arena.width`和`arena.height`来访问当前竞技场的尺寸和位置详情：

```cpp
// The boundaries of the arena
IntRect arena;
```

我们之前添加的代码的最后部分当然是我们的游戏循环：

```cpp
// The main game loop
while (window.isOpen())
{
}
```

我们可能已经注意到代码变得相当长。我们将在下一节讨论这个不便之处。

# 管理代码文件

使用类和函数进行抽象的一个优点是，我们的代码文件长度（行数）可以减少。尽管我们将为这个项目使用十几个代码文件，但`ZombieArena.cpp`中的代码长度在项目结束时仍会变得有点难以管理。在最终项目 Space Invaders++中，我们将探讨更多抽象和管理代码的方法。

目前，使用这个技巧来保持事情的可管理性。注意，在 Visual Studio 代码编辑器的左侧，有几个**+**和**-**符号，其中一个在本图中显示：

![](img/B14278_08_04.jpg)

每个代码块（例如`if`、`while`、`for`等）将有一个对应的标记。您可以通过点击**+**和**-**符号来展开和折叠这些块。我建议将所有当前未讨论的代码块都折叠起来。这将使事情更加清晰。

此外，我们可以创建自己的可折叠块。我建议将主游戏循环开始之前的所有代码制作成一个可折叠块。要做到这一点，请突出显示代码，然后*右键单击*并选择**大纲**|**隐藏选择**，如图所示：

![](img/B14278_08_05.jpg)

现在，您可以点击 **-** 和 **+** 符号来展开和折叠块。每次我们在主游戏循环之前添加代码（这将会很频繁），您都可以展开代码，添加新行，然后再将其折叠。以下截图显示了代码折叠时的样子：

![图片](img/B14278_08_06.jpg)

这比之前要容易管理得多。现在，我们可以开始编写主游戏循环。

# 开始编写主游戏循环

如您所见，前面代码的最后部分是游戏循环（`while (window.isOpen()){}`）。我们现在将关注这个部分。具体来说，我们将编写游戏循环的输入处理部分。

我们将要添加的代码相当长。尽管如此，它并没有什么复杂的地方，我们稍后将会详细检查它。

将以下高亮显示的代码添加到游戏循环中：

```cpp
// The main game loop
while (window.isOpen())
{
    /*
    ************
    Handle input
    ************
    */
    // Handle events by polling
    Event event;
    while (window.pollEvent(event))
    {
        if (event.type == Event::KeyPressed)
        {                                    
            // Pause a game while playing
            if (event.key.code == Keyboard::Return &&
                state == State::PLAYING)
            {
                state = State::PAUSED;
            }
            // Restart while paused
            else if (event.key.code == Keyboard::Return &&
                state == State::PAUSED)
            {
                state = State::PLAYING;
                // Reset the clock so there isn't a frame jump
                clock.restart();
            }
            // Start a new game while in GAME_OVER state
            else if (event.key.code == Keyboard::Return &&
                state == State::GAME_OVER)
            {
                state = State::LEVELING_UP;
            }
            if (state == State::PLAYING)
            {
            }
        }
    }// End event polling
}// End game loop
```

在前面的代码中，我们实例化了一个 `Event` 类型的对象。我们将像在之前的项目中一样使用 `event` 来轮询系统事件。为此，我们将上一个代码块中的其余代码包裹在一个带有 `window.pollEvent(event)` 条件的 `while` 循环中。这将保持循环，直到没有更多事件需要处理。

在这个 `while` 循环内部，我们处理我们感兴趣的的事件。首先，我们测试 `Event::KeyPressed` 事件。如果游戏处于 `PLAYING` 状态时按下了 `Return` 键，那么我们将 `state` 切换到 `PAUSED`。

如果在游戏处于 `PAUSED` 状态时按下了 `Return` 键，那么我们将 `state` 切换到 `PLAYING` 并重新启动 `clock` 对象。我们在从 `PAUSED` 切换到 `PLAYING` 后重新启动 `clock` 的原因是，当游戏暂停时，经过的时间仍然会累积。如果我们不重新启动时钟，所有对象都会更新它们的位置，就像帧刚刚花费了很长时间一样。随着我们在文件中完善其余的代码，这一点将变得更加明显。

然后，我们有一个 `else if` 块来测试在游戏处于 `GAME_OVER` 状态时是否按下了 `Return` 键。如果是的话，那么 `state` 将被更改为 `LEVELING_UP`。

重要提示

注意，`GAME_OVER` 状态是显示主页面的状态。因此，`GAME_OVER` 状态是在玩家刚刚死亡以及玩家第一次运行游戏后的状态。玩家在每一局游戏中首先要做的事情就是选择一个属性来提升（即升级）。

在前面的代码中，有一个最终的 `if` 条件来测试状态是否等于 `PLAYING`。这个 `if` 块是空的，我们将在整个项目中向其中添加代码。

小贴士

我们将在整个项目过程中向这个文件的许多不同部分添加代码。因此，花时间了解我们的游戏可能处于的不同状态以及我们如何处理这些状态是非常有价值的。在适当的时候折叠和展开不同的 `if`、`else` 和 `while` 块也将非常有好处。

花些时间彻底熟悉我们刚刚编写的`while`、`if`和`else if`块。我们将会经常引用它们。

接下来，在之前的代码之后，仍然在游戏循环内，仍然在处理输入，添加以下突出显示的代码。注意现有的代码（未突出显示），它显示了新（突出显示）代码的确切位置：

```cpp
    }// End event polling
    // Handle the player quitting
    if (Keyboard::isKeyPressed(Keyboard::Escape))
    {
        window.close();
    }
    // Handle WASD while playing
    if (state == State::PLAYING)
    {
        // Handle the pressing and releasing of the WASD keys
        if (Keyboard::isKeyPressed(Keyboard::W))
        {
            player.moveUp();
        }
        else
        {
            player.stopUp();
        }
        if (Keyboard::isKeyPressed(Keyboard::S))
        {
            player.moveDown();
        }
        else
        {
            player.stopDown();
        }
        if (Keyboard::isKeyPressed(Keyboard::A))
        {
            player.moveLeft();
        }
        else
        {
            player.stopLeft();
        }
        if (Keyboard::isKeyPressed(Keyboard::D))
        {
            player.moveRight();
        }
        else
        {
            player.stopRight();
        }
    }// End WASD while playing
}// End game loop
```

在前面的代码中，我们首先测试玩家是否按下了*Escape*键。如果按下，游戏窗口将被关闭。

接下来，在一个大的`if(state == State::PLAYING)`块内，我们依次检查每个*WASD*键。如果按键被按下，我们调用相应的`player.move...`函数。如果没有，我们调用相关的`player.stop...`函数。

此代码确保在每个帧中，玩家对象都会根据按下的*WASD*键和未按下的键进行更新。`player.move...`和`player.stop...`函数将信息存储在成员布尔变量中（`m_LeftPressed`、`m_RightPressed`、`m_UpPressed`和`m_DownPressed`）。`Player`类然后在每个帧的`player.update`函数中响应这些布尔值，我们将在游戏循环的更新部分调用它。

现在，我们可以处理键盘输入，允许玩家在每场游戏的开始和每波之间升级。添加并学习以下突出显示的代码，然后我们将讨论它：

```cpp
    }// End WASD while playing
    // Handle the LEVELING up state
    if (state == State::LEVELING_UP)
    {
        // Handle the player LEVELING up
        if (event.key.code == Keyboard::Num1)
        {
            state = State::PLAYING;
        }
        if (event.key.code == Keyboard::Num2)
        {
            state = State::PLAYING;
        }
        if (event.key.code == Keyboard::Num3)
        {
            state = State::PLAYING;
        }
        if (event.key.code == Keyboard::Num4)
        {
            state = State::PLAYING;
        }
        if (event.key.code == Keyboard::Num5)
        {
            state = State::PLAYING;
        }
        if (event.key.code == Keyboard::Num6)
        {
            state = State::PLAYING;
        }

        if (state == State::PLAYING)
        {            
            // Prepare the level
            // We will modify the next two lines later
            arena.width = 500;
            arena.height = 500;
            arena.left = 0;
            arena.top = 0;
            // We will modify this line of code later
            int tileSize = 50;
            // Spawn the player in the middle of the arena
            player.spawn(arena, resolution, tileSize);

            // Reset the clock so there isn't a frame jump
            clock.restart();
        }
    }// End LEVELING up

}// End game loop
```

在前面的代码中，它全部包含在一个测试中，以查看当前`state`的值是否等于`LEVELING_UP`，我们处理键盘键*1*、*2*、*3*、*4*、*5*和*6*。在每一个`if`块中，我们只是将`state`设置为`State::PLAYING`。我们将在*第十三章**，声音效果、文件 I/O 和完成游戏*中稍后添加一些代码来处理每个升级选项。

此代码执行以下操作：

1.  如果`state`等于`LEVELING_UP`，等待按下*1*、*2*、*3*、*4*、*5*或*6*键。

1.  按下时，将`state`更改为`PLAYING`。

1.  当状态改变时，仍然在`if (state == State::LEVELING_UP)`块内，嵌套的`if(state == State::PLAYING)`块将会执行。

1.  在此块中，我们设置`arena`的位置和大小，将`tileSize`设置为`50`，将所有信息传递给`player.spawn`，并调用`clock.restart`。

现在，我们有一个实际生成的玩家对象，它了解其环境并能对按键做出响应。我们现在可以在循环的每次传递中更新场景。

一定要将游戏循环中输入处理部分的代码整洁地折叠起来，因为我们现在已经完成了这部分。以下代码是游戏循环的更新部分。添加并学习以下突出显示的代码，然后我们可以讨论它：

```cpp
    }// End LEVELING up
    /*
    ****************
    UPDATE THE FRAME
    ****************
    */
    if (state == State::PLAYING)
    {
        // Update the delta time
        Time dt = clock.restart();

        // Update the total game time
        gameTimeTotal += dt;

        // Make a decimal fraction of 1 from the delta time
        float dtAsSeconds = dt.asSeconds();
        // Where is the mouse pointer
        mouseScreenPosition = Mouse::getPosition();
        // Convert mouse position to world coordinates of mainView
        mouseWorldPosition = window.mapPixelToCoords(
            Mouse::getPosition(), mainView);
        // Update the player
        player.update(dtAsSeconds, Mouse::getPosition());
        // Make a note of the players new position
        Vector2f playerPosition(player.getCenter());

        // Make the view centre around the player                
        mainView.setCenter(player.getCenter());
    }// End updating the scene

}// End game loop
```

首先，请注意，上一段代码被包裹在一个测试中，以确保游戏处于`PLAYING`状态。我们不希望在这段代码在游戏暂停、结束或玩家选择升级时运行。

首先，我们重新启动时钟并将上一帧所花费的时间存储在`dt`变量中：

```cpp
// Update the delta time
Time dt = clock.restart();
```

接下来，我们将上一帧所花费的时间添加到游戏运行的总累积时间`gameTimeTotal`中：

```cpp
// Update the total game time
gameTimeTotal += dt;
```

现在，我们使用`dt.AsSeconds`函数返回的值初始化一个名为`dtAsSeconds`的`float`变量。对于大多数帧，这将是一个分数。这对于传递给`player.update`函数以计算移动玩家精灵的量是完美的。

现在，我们可以使用`MOUSE::getPosition`函数初始化`mouseScreenPosition`。

重要提示

你可能想知道获取鼠标位置略微不寻常的语法。这被称为**静态函数**。如果我们使用`static`关键字在类中定义一个函数，我们可以使用类名调用该函数，而不需要类的实例。C++面向对象编程有很多这样的怪癖和规则。随着我们的进展，我们将看到更多。

然后，我们使用`window`上的 SFML `mapPixelToCoords`函数初始化`mouseWorldPosition`。我们在本章前面讨论了该函数。

到目前为止，我们现在能够调用`player.update`并传入`dtAsSeconds`和鼠标的位置，正如所需的那样。

我们将玩家的新中心存储在一个名为`playerPosition`的`Vector2f`实例中。目前，这个变量尚未使用，但我们在项目后期将会有所用途。

然后，我们可以使用`mainView.setCenter(player.getCenter())`将视图中心定位在玩家最新位置的中央。

我们现在能够将玩家绘制到屏幕上。添加以下突出显示的代码，将主游戏循环的绘制部分拆分为不同的状态：

```cpp
        }// End updating the scene
        /*
        **************
        Draw the scene
        **************
        */
        if (state == State::PLAYING)
        {
            window.clear();
            // set the mainView to be displayed in the window
            // And draw everything related to it
            window.setView(mainView);
            // Draw the player
            window.draw(player.getSprite());
        }
        if (state == State::LEVELING_UP)
        {
        }
        if (state == State::PAUSED)
        {
        }
        if (state == State::GAME_OVER)
        {
        }
        window.display();
    }// End game loop
    return 0;
}
```

在上一段代码的`if(state == State::PLAYING)`部分中，我们清除屏幕，将窗口的视图设置为`mainView`，然后使用`window.draw(player.getSprite())`绘制玩家精灵。

在处理完所有不同的状态后，代码以通常的方式使用`window.display();`显示场景。

你可以运行游戏，并看到我们的玩家角色在鼠标移动时旋转。

小贴士

当你运行游戏时，你需要按*Enter*键开始游戏，然后从*1*到*6*选择一个数字来模拟选择升级选项。然后，游戏将开始。

你还可以在（空白的）500 x 500 像素的竞技场内移动玩家。你可以看到屏幕中央的孤独玩家，如图所示：

![图片](img/B14278_08_07.jpg)

然而，你无法感受到任何移动的感觉，因为我们还没有实现背景。我们将在下一章这样做。

# 摘要

呼！这一章内容很长。我们在本章做了很多工作：我们为 Zombie Arena 项目构建了第一个类`Player`，并在游戏循环中使用了它。我们还学习了并使用了`View`类的一个实例，尽管我们还没有探索这给我们带来的好处。

在下一章中，我们将通过探索精灵图集（sprite sheets）来构建我们的竞技场背景。我们还将学习关于 C++ **引用**的知识，这些引用允许我们在变量超出作用域（即在另一个函数中）时对其进行操作。

# 常见问题解答

Q) 我注意到我们为`Player`类编写了很多我们没有使用的函数。为什么会有这种情况？

A) 我们不是反复回到`Player`类，而是将整个项目所需的所有代码都添加进来了。到*第十三章*“音效、文件输入/输出和完成游戏”结束时，我们将充分利用所有这些功能。
