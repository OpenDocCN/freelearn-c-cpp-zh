# 第十三章。高级 OOP-继承和多态

在本章中，我们将通过学习稍微更高级的**继承**和**多态**概念来进一步扩展我们对 OOP 的知识。然后，我们将能够使用这些新知识来实现我们游戏的明星角色，Thomas 和 Bob。在本章中，我们将更详细地介绍以下内容：

+   如何使用继承扩展和修改类？

+   通过多态将一个类的对象视为多种类型的类

+   抽象类以及设计永远不会实例化的类实际上可以很有用

+   构建一个抽象的`PlayableCharacter`类

+   使用`Thomas`和`Bob`类来实现继承

+   将 Thomas 和 Bob 添加到游戏项目中

# 继承

我们已经看到了如何通过实例化/创建对象来使用 SFML 库的类的其他人的辛勤工作。但是，这整个面向对象的东西甚至比那更深入。

如果有一个类中有大量有用的功能，但不完全符合我们的要求怎么办？在这种情况下，我们可以**继承**自其他类。就像它听起来的那样，**继承**意味着我们可以利用其他人的类的所有特性和好处，包括封装，同时进一步完善或扩展代码，使其特别适合我们的情况。在这个项目中，我们将继承并扩展一些 SFML 类。我们也会用我们自己的类来做同样的事情。

让我们看一些使用继承的代码，

## 扩展一个类

考虑到所有这些，让我们看一个示例类，并看看我们如何扩展它，只是为了看看语法并作为第一步。

首先，我们定义一个要继承的类。这与我们创建任何其他类没有什么不同。看一下这个假设的`Soldier`类声明：

```cpp
class Soldier 
{ 
   private: 
      // How much damage can the soldier take 
      int m_Health; 
      int m_Armour; 
      int m_Range; 
      int m_ShotPower; 

   Public: 
      void setHealth(int h); 
      void setArmour(int a);   
      void setRange(int r); 
      void setShotPower(int p); 
}; 

```

在前面的代码中，我们定义了一个`Soldier`类。它有四个私有变量，`m_Health`，`m_Armour`，`m_Range`和`m_ShotPower`。它有四个公共函数`setHealth`，`setArmour`，`setRange`和`setShotPower`。我们不需要看到函数的定义，它们只是简单地初始化它们的名字明显的适当变量。

我们还可以想象，一个完全实现的`Soldier`类会比这更加深入。它可能有函数，比如`shoot`，`goProne`等。如果我们在一个 SFML 项目中实现了`Soldier`类，它可能会有一个`Sprite`对象，以及一个`update`和一个`getPostion`函数。

这里呈现的简单场景适合学习继承。现在让我们看看一些新东西，实际上是从`Soldier`类继承。看看这段代码，特别是突出显示的部分：

```cpp
class Sniper : public Soldier 
{ 
public: 
   // A constructor specific to Sniper 
   Sniper::Sniper(); 
}; 

```

通过将`: public Soldier`代码添加到`Sniper`类声明中，`Sniper`继承自`Soldier`。但这到底意味着什么？`Sniper`是一个`Soldier`。它拥有`Soldier`的所有变量和函数。然而，继承不仅仅是这样。

还要注意，在前面的代码中，我们声明了一个`Sniper`构造函数。这个构造函数是`Sniper`独有的。我们不仅继承了`Soldier`，还**扩展了**`Soldier`。`Soldier`类的所有功能（定义）都由`Soldier`类处理，但`Sniper`构造函数的定义必须由`Sniper`类处理。

这是假设的`Sniper`构造函数定义可能是这样的：

```cpp
// In Sniper.cpp 
Sniper::Sniper() 
{ 
   setHealth(10); 
   setArmour(10);  
   setRange(1000); 
   setShotPower(100); 
} 

```

我们可以继续编写一堆其他类，这些类是`Soldier`类的扩展，也许是`Commando`和`Infantryman`。每个类都有完全相同的变量和函数，但每个类也可以有一个独特的构造函数，用于初始化适合`Soldier`类型的变量。`Commando`可能有非常高的`m_Health`和`m_ShotPower`，但是`m_Range`非常小。`Infantryman`可能介于`Commando`和`Sniper`之间，每个变量的值都是中等水平。

### 提示

好像面向对象编程已经足够有用了，现在我们可以模拟现实世界的对象，包括它们的层次结构。我们通过子类化、扩展和继承其他类来实现这一点。

我们可能想要学习的术语是从中扩展的类是**超类**，从超类继承的类是**子类**。我们也可以说**父**类和**子**类。

### 提示

关于继承，您可能会问这样一个问题：为什么？原因是这样的：我们可以编写一次通用代码；在父类中，我们可以更新该通用代码，所有继承自它的类也会被更新。此外，子类只能使用公共和**受保护**实例变量和函数。因此，如果设计得当，这也进一步增强了封装的目标。

你说受保护？是的。有一个称为**受保护**的类变量和函数的访问限定符。您可以将受保护的变量视为介于公共和私有之间。以下是访问限定符的快速摘要，以及有关受保护限定符的更多详细信息：

+   `公共`变量和函数可以被任何人访问和使用。

+   `私有`变量和函数只能被类的内部代码访问/使用。这对封装很有用，当我们需要访问/更改私有变量时，我们可以提供公共的`getter`和`setter`函数（如`getSprite`等）。如果我们扩展了一个具有`私有`变量和函数的类，那么子类*不能*直接访问其父类的私有数据。

+   `受保护`变量和函数几乎与私有变量和函数相同。它们不能被类的实例直接访问/使用。但是，它们*可以*被扩展它们所声明的类的任何类直接使用。因此，它们就像是私有的，只不过对子类是可见的。

要充分理解受保护的变量和函数以及它们如何有用，让我们先看看另一个主题，然后我们可以看到它们的作用。

# 多态

**多态**允许我们编写的代码不那么依赖于我们试图操作的类型。这可以使我们的代码更清晰和更高效。多态意味着不同的形式。如果我们编写的对象可以是多种类型的东西，那么我们就可以利用这一点。

### 注意

多态对我们意味着什么？简而言之，多态就是：任何子类都可以作为使用超类的代码的一部分。这意味着我们可以编写更简单、更易于理解的代码，也更容易修改或更改。此外，我们可以为超类编写代码，并依赖于这样一个事实：在一定的参数范围内，无论它被子类化多少次，代码仍然可以正常工作。

让我们讨论一个例子。

假设我们想利用多态来帮助编写一个动物园管理游戏，我们需要喂养和照顾动物的需求。我们可能会想要有一个名为`feed`的函数。我们可能还想将要喂养的动物的实例传递给`feed`函数。

当然，动物园有很多种类的动物——`狮子`、`大象`和`三趾树懒`。有了我们对 C++继承的新知识，编写一个`Animal`类并让所有不同类型的动物从中继承就会有意义。

如果我们想编写一个函数（`feed`），我们可以将狮子、大象和三趾树懒作为参数传递进去，似乎需要为每种类型的`Animal`编写一个`feed`函数。但是，我们可以编写多态函数，具有多态返回类型和参数。看一下这个假设的`feed`函数的定义：

```cpp
void feed(Animal& a) 
{ 
   a.decreaseHunger(); 
} 

```

前面的函数将`Animal`引用作为参数，这意味着可以将任何从扩展`Animal`的类构建的对象传递给它。

因此，今天你甚至可以编写代码，然后在一周、一个月或一年后创建另一个子类，相同的函数和数据结构仍然可以工作。此外，我们可以对子类强制执行一组规则，规定它们可以做什么，不能做什么，以及如何做。因此，一个阶段的良好设计可以影响其他阶段。

但我们真的会想要实例化一个真正的动物吗？

# 抽象类 - 虚拟和纯虚拟函数

**抽象类**是一个不能被实例化的类，因此不能被制作成对象。

### 提示

在这里我们可能想学习的一些术语是**具体**类。**具体**类是任何不是抽象的类。换句话说，到目前为止我们编写的所有类都是具体类，可以实例化为可用的对象。

那么，这段代码永远不会被使用了吗？但这就像付钱给一个建筑师设计你的房子，然后永远不建造它！

如果我们或一个类的设计者想要强制其用户在使用他们的类之前继承它，他们可以将一个类**抽象化**。然后，我们就不能从中创建一个对象；因此，我们必须首先扩展它，然后从子类创建一个对象。

为此，我们可以创建一个**纯虚拟**函数并不提供任何定义。然后，任何扩展它的类都必须**覆盖**（重新编写）该函数。

让我们看一个例子；这会有所帮助。我们通过添加一个纯虚拟函数使一个类变成抽象类，比如这个只能执行通用动作`makeNoise`的抽象`Animal`类：

```cpp
Class Animal 
   private: 
      // Private stuff here 

   public: 

      void virtual makeNoise() = 0; 

      // More public stuff here 
}; 

```

如你所见，我们在函数声明之前添加了 C++关键字`virtual`，之后添加了`= 0`。现在，任何扩展/继承自`Animal`的类都必须覆盖`makeNoise`函数。这是有道理的，因为不同类型的动物发出的声音非常不同。也许我们可以假设任何扩展`Animal`类的人都足够聪明，能够注意到`Animal`类不能发出声音，他们需要处理它，但如果他们没有注意到呢？关键是通过创建一个纯虚拟函数，我们保证他们会注意到，因为他们必须注意到。

抽象类也很有用，因为有时我们需要一个可以用作多态类型的类，但需要保证它永远不能用作对象。例如，`Animal`单独使用并没有太多意义。我们不谈论动物；我们谈论动物的类型。我们不会说，“哦，看那只可爱的、蓬松的、白色的动物！”或者，“昨天我们去宠物店买了一只动物和一个动物床”。这太抽象了。

因此，抽象类有点像一个**模板**，可以被任何扩展它的类使用（继承自它）。如果我们正在构建一个*工业帝国*类型的游戏，玩家管理企业和员工，我们可能需要一个`Worker`类，并将其扩展为`Miner`、`Steelworker`、`OfficeWorker`，当然还有`Programmer`。但是一个普通的`Worker`到底是做什么的呢？我们为什么要实例化一个？

答案是我们不想实例化一个，但我们可能想将其用作多态类型，以便在函数之间传递多个`Worker`子类，并且有可以容纳所有类型的工人的数据结构。

所有纯虚拟函数必须被扩展父类的任何类覆盖，该父类包含纯虚拟函数。这意味着抽象类可以提供一些在所有子类中都可用的公共功能。例如，`Worker`类可能有`m_AnnualSalary`、`m_Productivity`和`m_Age`成员变量。它可能还有`getPayCheck`函数，这不是纯虚拟的，并且在所有子类中都是相同的，但它可能有一个`doWork`函数，这是纯虚拟的，必须被覆盖，因为所有不同类型的`Worker`都会以非常不同的方式`doWork`。

### 注意

顺便说一句，**virtual**与纯虚函数相反，是一个**可选重写**的函数。你声明一个虚函数的方式与声明纯虚函数的方式相同，但是最后不加上`= 0`。在当前的游戏项目中，我们将使用一个纯虚函数。

如果对虚拟、纯虚拟或抽象的任何内容不清楚，使用它可能是理解它的最佳方式。

# 构建 PlayableCharacter 类

现在我们已经了解了继承、多态和纯虚函数的基础知识，我们将把它们应用起来。我们将构建一个`PlayableCharacter`类，它将拥有我们游戏中任何角色大部分功能所需的功能。它将有一个纯虚函数，`handleInput`。`handleInput`函数在子类中需要有很大的不同，所以这是有道理的。

由于`PlayableCharacter`将有一个纯虚函数，它将是一个抽象类，不可能有它的对象。然后我们将构建`Thomas`和`Bob`类，它们将继承自`PlayableCharacter`，实现纯虚函数的定义，并允许我们在游戏中实例化`Bob`和`Thomas`对象。

## 编写 PlayableCharacter.h

通常，在创建一个类时，我们将从包含成员变量和函数声明的头文件开始。新的是，在这个类中，我们将声明一些**protected**成员变量。请记住，受保护的变量可以被继承自具有受保护变量的类的类使用，就好像它们是`Public`一样。

在**解决方案资源管理器**中右键单击**头文件**，然后选择**添加** | **新建项...**。在**添加新项**窗口中，通过左键单击**头文件（** `.h` **）**突出显示，然后在**名称**字段中键入`PlayableCharacter.h`。最后，单击**添加**按钮。我们现在准备为`PlayableCharacter`类编写头文件。

我们将在三个部分中添加和讨论`PlayableCharacter.h`文件的内容。首先是**protected**部分，然后是**private**，最后是**public**。

在`PlayableCharacter.h`文件旁边添加下面显示的代码：

```cpp
#pragma once 
#include <SFML/Graphics.hpp> 

using namespace sf; 

class PlayableCharacter 
{ 
protected: 
   // Of course we will need a sprite 
   Sprite m_Sprite; 

   // How long does a jump last 
   float m_JumpDuration; 

   // Is character currently jumping or falling 
   bool m_IsJumping; 
   bool m_IsFalling; 

   // Which directions is the character currently moving in 
   bool m_LeftPressed; 
   bool m_RightPressed; 

   // How long has this jump lasted so far 
   float m_TimeThisJump; 

   // Has the player just initialted a jump 
   bool m_JustJumped = false; 

   // Private variables and functions come next 

```

我们刚刚编写的代码中要注意的第一件事是所有变量都是`protected`的。这意味着当我们扩展类时，我们刚刚编写的所有变量将对扩展它的类可访问。我们将用`Thomas`和`Bob`类扩展这个类。

除了`protected`访问规范之外，先前的代码没有什么新的或复杂的。然而，值得注意的是一些细节。然后随着我们的进展，理解类的工作原理将变得容易。因此，让我们逐个运行这些`protected`变量。

我们有一个相对可预测的`Sprite`，`m_Sprite`。我们有一个名为`m_JumpDuration`的浮点数，它将保存代表角色能够跳跃的时间值。数值越大，角色就能够跳得越远/高。

接下来，我们有一个布尔值，`m_IsJumping`，当角色跳跃时为`true`，否则为`false`。这将有助于确保角色在空中时无法跳跃。

`m_IsFalling`变量与`m_IsJumping`有类似的用途。它将有助于知道角色何时下落。

接下来，我们有两个布尔值，如果角色的左或右键盘按钮当前被按下，则为`true`。这取决于角色（*A*和*D*为 Thomas，左右箭头键为 Bob）。我们将在`Thomas`和`Bob`类中看到如何响应这些布尔值。

`m_TimeThisJump`浮点变量在每一帧`m_IsJumping`为`true`时更新。然后我们就可以知道`m_JumpDuration`何时被达到。

最后一个`protected`变量是布尔值`m_JustJumped`。如果在当前帧中启动了跳跃，它将为`true`。这对于知道何时播放跳跃音效将很有用。

接下来，将以下`private`变量添加到`PlayableCharacter.h`文件中：

```cpp
private: 
   // What is the gravity 
   float m_Gravity; 

   // How fast is the character 
   float m_Speed = 400; 

   // Where is the player 
   Vector2f m_Position; 

   // Where are the characters various body parts? 
   FloatRect m_Feet; 
   FloatRect m_Head; 
   FloatRect m_Right; 
   FloatRect m_Left; 

   // And a texture 
   Texture m_Texture; 

   // All our public functions will come next 

```

在之前的代码中，我们有一些有趣的`private`变量。请记住，这些变量只能被`PlayableCharacter`类中的代码直接访问。`Thomas`和`Bob`类将无法直接访问它们。

`m_Gravity`变量将保存角色下落的每秒像素数。`m_Speed`变量将保存角色每秒可以向左或向右移动的像素数。

`Vector2f`，`m_Position`变量是角色在世界中（而不是屏幕上）的位置，即角色的中心位置。

接下来的四个`FloatRect`对象很重要。在*Zombie Arena*游戏中进行碰撞检测时，我们只是检查两个`FloatRect`对象是否相交。每个`FloatRect`对象代表整个角色、拾取物或子弹。对于非矩形形状的对象（僵尸和玩家），这有点不准确。

在这个游戏中，我们需要更加精确。`m_Feet`，`m_Head`，`m_Right`和`m_Left` `FloatRect`对象将保存角色身体不同部位的坐标。这些坐标将在每一帧中更新。

通过这些坐标，我们将能够准确地判断角色何时落在平台上，跳跃时是否碰到头部，或者与侧面的瓷砖擦肩而过。

最后，我们有`Texture`。`Texture`是`private`的，因为它不会被`Thomas`或`Bob`类直接使用，但正如我们所看到的，`Sprite`是`protected`的，因为它被直接使用。

现在将所有`public`函数添加到`PlayableCharacter.h`文件中，然后我们将讨论它们：

```cpp
public: 

   void spawn(Vector2f startPosition, float gravity); 

   // This is a pure virtual function 
   bool virtual handleInput() = 0; 
   // This class is now abstract and cannot be instanciated 

   // Where is the player 
   FloatRect getPosition(); 

   // A rectangle representing the position  
   // of different parts of the sprite 
   FloatRect getFeet(); 
   FloatRect getHead(); 
   FloatRect getRight(); 
   FloatRect getLeft(); 

   // Send a copy of the sprite to main 
   Sprite getSprite(); 

   // Make the character stand firm 
   void stopFalling(float position); 
   void stopRight(float position); 
   void stopLeft(float position); 
   void stopJump(); 

   // Where is the center of the character 
   Vector2f getCenter(); 

   // We will call this function once every frame 
   void update(float elapsedTime); 

};// End of the class 

```

让我们谈谈我们刚刚添加的每个函数声明。这将使编写它们的定义更容易跟踪。

+   `spawn`函数接收一个名为`startPosition`的`Vector2f`和一个名为`gravity`的`float`。顾名思义，`startPosition`将是角色在关卡中开始的坐标，`gravity`将是角色下落的每秒像素数。

+   `bool virtual handleInput() = 0`当然是我们的纯虚函数。由于`PlayableCharacter`有这个函数，任何扩展它的类，如果我们想要实例化它，必须为这个函数提供定义。因此，当我们在一分钟内为`PlayableCharacter`编写所有函数定义时，我们将不为`handleInput`提供定义。当然，`Thomas`和`Bob`类中也需要有定义。

+   `getPosition`函数返回一个代表整个角色位置的`FloatRect`。

+   `getFeet()`函数，以及`getHead`，`getRight`和`getLeft`，每个都返回一个代表角色身体特定部位位置的`FloatRect`。这正是我们需要进行详细的碰撞检测。

+   `getSprite`函数像往常一样，将`m_Sprite`的副本返回给调用代码。

+   `stopFalling`，`stopRight`，`stopLeft`和`stopJump`函数接收一个`float`值，函数将使用它来重新定位角色并阻止它在实心瓷砖上行走或跳跃。

+   `getCenter`函数将一个`Vector2f`返回给调用代码，让它准确地知道角色的中心在哪里。这个值当然保存在`m_Position`中。我们将在后面看到，它被`Engine`类用来围绕适当的角色中心适当地居中适当的`View`。

+   我们之前多次见过的`update`函数和往常一样，它接受一个`float`参数，表示当前帧所花费的秒数的一部分。然而，这个`update`函数需要做的工作比以前的`update`函数（来自其他项目）更多。它需要处理跳跃，以及更新代表头部、脚部、左侧和右侧的`FloatRect`对象。

现在我们可以为所有函数编写定义，当然，除了`handleInput`。

## 编写 PlayableCharacter.cpp

在**解决方案资源管理器**中右键单击**源文件**，然后选择**添加** | **新建项...**。在**添加新项**窗口中，通过左键单击**C++文件（**`.cpp`**）**，然后在**名称**字段中键入`PlayableCharacter.cpp`。最后，单击**添加**按钮。现在我们准备为`PlayableCharacter`类编写`.cpp`文件。

我们将把代码和讨论分成几个部分。首先，添加包含指令和`spawn`函数的定义：

```cpp
#include "stdafx.h" 
#include "PlayableCharacter.h" 

void PlayableCharacter::spawn(Vector2f startPosition, float gravity) 
{ 
   // Place the player at the starting point 
   m_Position.x = startPosition.x; 
   m_Position.y = startPosition.y; 

   // Initialize the gravity 
   m_Gravity = gravity; 

   // Move the sprite in to position 
   m_Sprite.setPosition(m_Position); 

} 

```

`spawn`函数使用传入的位置初始化`m_Position`，并初始化`m_Gravity`。代码的最后一行将`m_Sprite`移动到其起始位置。

接下来，在前面的代码之后立即添加`update`函数的定义：

```cpp
void PlayableCharacter::update(float elapsedTime) 
{ 

   if (m_RightPressed) 
   { 
      m_Position.x += m_Speed * elapsedTime; 
   } 

   if (m_LeftPressed) 
   { 
      m_Position.x -= m_Speed * elapsedTime; 
   } 

   // Handle Jumping 
   if (m_IsJumping) 
   { 
      // Update how long the jump has been going 
      m_TimeThisJump += elapsedTime; 

      // Is the jump going upwards 
      if (m_TimeThisJump < m_JumpDuration) 
      { 
         // Move up at twice gravity 
         m_Position.y -= m_Gravity * 2 * elapsedTime; 
      } 
      else 
      { 
         m_IsJumping = false; 
         m_IsFalling = true; 
      } 

   } 

   // Apply gravity 
   if (m_IsFalling) 
   { 
      m_Position.y += m_Gravity * elapsedTime; 
   } 

   // Update the rect for all body parts 
   FloatRect r = getPosition(); 

   // Feet 
   m_Feet.left = r.left + 3; 
   m_Feet.top = r.top + r.height - 1; 
   m_Feet.width = r.width - 6; 
   m_Feet.height = 1; 

   // Head 
   m_Head.left = r.left; 
   m_Head.top = r.top + (r.height * .3); 
   m_Head.width = r.width; 
   m_Head.height = 1; 

   // Right 
   m_Right.left = r.left + r.width - 2; 
   m_Right.top = r.top + r.height * .35; 
   m_Right.width = 1; 
   m_Right.height = r.height * .3; 

   // Left 
   m_Left.left = r.left; 
   m_Left.top = r.top + r.height * .5; 
   m_Left.width = 1; 
   m_Left.height = r.height * .3; 

   // Move the sprite into position 
   m_Sprite.setPosition(m_Position); 

} 

```

代码的前两部分检查`m_RightPressed`或`m_LeftPressed`是否为`true`。如果其中任何一个是，`m_Position`将使用与上一个项目相同的公式（经过的时间乘以速度）进行更改。

接下来，我们看看角色当前是否正在执行跳跃。我们从`if(m_IsJumping)`知道这一点。如果这个`if`语句为`true`，代码将执行以下步骤：

1.  用`elapsedTime`更新`m_TimeThisJump`。

1.  检查`m_TimeThisJump`是否仍然小于`m_JumpDuration`。如果是，则通过重力乘以经过的时间两倍来改变`m_Position`的 y 坐标。

1.  在`else`子句中，当`m_TimeThisJump`不低于`m_JumpDuration`时，`m_Falling`被设置为`true`。这样做的效果将在下面看到。此外，`m_Jumping`被设置为`false`。这样做是为了防止我们刚刚讨论的代码执行，因为`if(m_IsJumping)`现在为 false。

`if(m_IsFalling)`块每帧将`m_Position`向下移动。它使用`m_Gravity`的当前值和经过的时间进行移动。

以下代码（几乎所有剩余的代码）相对于精灵的当前位置更新角色的身体部位。看一下下面的图表，看看代码如何计算角色的虚拟头部、脚部、左侧和右侧的位置：

![编写 PlayableCharacter.cpp](img/image_13_001.jpg)

代码的最后一行使用`setPosition`函数将精灵移动到`update`函数的所有可能性之后的正确位置。

现在立即在上一个代码之后添加`getPosition`、`getCenter`、`getFeet`、`getHead`、`getLeft`、`getRight`和`getSprite`函数的定义：

```cpp
FloatRect PlayableCharacter::getPosition() 
{ 
   return m_Sprite.getGlobalBounds(); 
} 

Vector2f PlayableCharacter::getCenter() 
{ 
   return Vector2f( 
      m_Position.x + m_Sprite.getGlobalBounds().width / 2, 
      m_Position.y + m_Sprite.getGlobalBounds().height / 2 
      ); 
} 

FloatRect PlayableCharacter::getFeet() 
{ 
   return m_Feet; 
} 

FloatRect PlayableCharacter::getHead() 
{ 
   return m_Head; 
} 

FloatRect PlayableCharacter::getLeft() 
{ 
   return m_Left; 
} 

FloatRect PlayableCharacter::getRight() 
{ 
   return m_Right; 
} 

Sprite PlayableCharacter::getSprite() 
{ 
   return m_Sprite; 
} 

```

`getPosition`函数返回包装整个精灵的`FloatRect`，`getCenter`返回一个包含精灵中心的`Vector2f`。请注意，我们将精灵的高度和宽度除以二，以便动态地得到这个结果。这是因为 Thomas 和 Bob 的身高不同。

`getFeet`、`getHead`、`getLeft`和`getRight`函数返回表示角色各个身体部位的`FloatRect`对象，我们在`update`函数中每帧更新它们。我们将在下一章中编写使用这些函数的**碰撞检测代码**。

`getSprite`函数像往常一样返回`m_Sprite`的副本。

最后，对于`PlayableCharacter`类，立即在上一个代码之后添加`stopFalling`、`stopRight`、`stopLeft`和`stopJump`函数的定义：

```cpp
void PlayableCharacter::stopFalling(float position) 
{ 
   m_Position.y = position - getPosition().height; 
   m_Sprite.setPosition(m_Position); 
   m_IsFalling = false; 
} 

void PlayableCharacter::stopRight(float position) 
{ 

   m_Position.x = position - m_Sprite.getGlobalBounds().width; 
   m_Sprite.setPosition(m_Position); 
} 

void PlayableCharacter::stopLeft(float position) 
{ 
   m_Position.x = position + m_Sprite.getGlobalBounds().width; 
   m_Sprite.setPosition(m_Position); 
} 

void PlayableCharacter::stopJump() 
{ 
   // Stop a jump early  
   m_IsJumping = false; 
   m_IsFalling = true; 
} 

```

每个前面的函数都接收一个值作为参数，用于重新定位精灵的顶部、底部、左侧或右侧。这些值是什么以及如何获得它们将在下一章中看到。每个前面的函数也重新定位精灵。

最后一个函数是`stopJump`函数，它也将在碰撞检测中使用。它设置了`m_IsJumping`和`m_IsFalling`的必要值来结束跳跃。

# 构建 Thomas 和 Bob 类

现在我们真正要使用继承了。我们将为 Thomas 建立一个类，为 Bob 建立一个类。它们都将继承我们刚刚编写的`PlayableCharacter`类。然后它们将拥有`PlayableCharacter`类的所有功能，包括直接访问其`protected`变量。我们还将添加纯虚函数`handleInput`的定义。您会注意到，`Thomas`和`Bob`的`handleInput`函数将是不同的。

## 编写 Thomas.h

在**解决方案资源管理器**中右键单击**头文件**，然后选择**添加** | **新建项...**。在**添加新项**窗口中，通过左键单击**头文件**（`.h`）并在**名称**字段中键入`Thomas.h`。最后，单击**添加**按钮。现在我们准备好为`Thomas`类编写头文件了。

现在将此代码添加到`Thomas.h`类中：

```cpp
#pragma once 
#include "PlayableCharacter.h" 

class Thomas : public PlayableCharacter 
{ 
public: 
   // A constructor specific to Thomas 
   Thomas::Thomas(); 

   // The overridden input handler for Thomas 
   bool virtual handleInput(); 

}; 

```

上面的代码非常简短而简洁。我们可以看到我们有一个构造函数，我们将要实现纯虚的`handleInput`函数，所以现在让我们来做吧。

## 编写 Thomas.cpp

在**解决方案资源管理器**中右键单击**源文件**，然后选择**添加** | **新建项...**。在**添加新项**窗口中，通过左键单击**C++文件**（`.cpp`）并在**名称**字段中键入`Thomas.cpp`。最后，单击**添加**按钮。现在我们准备好为`Thomas`类编写`.cpp`文件了。

将`Thomas`构造函数添加到`Thomas.cpp`文件中，如下面的片段所示：

```cpp
#include "stdafx.h" 
#include "Thomas.h" 
#include "TextureHolder.h" 

Thomas::Thomas() 
{ 
   // Associate a texture with the sprite 
   m_Sprite = Sprite(TextureHolder::GetTexture( 
      "graphics/thomas.png")); 

   m_JumpDuration = .45; 
} 

```

我们只需要加载`thomas.png`图形并将跳跃持续时间（`m_JumpDuration`）设置为`.45`（几乎半秒）。

添加`handleInput`函数的定义，如下面的片段所示：

```cpp

// A virtual function 
bool Thomas::handleInput() 
{ 
   m_JustJumped = false; 

   if (Keyboard::isKeyPressed(Keyboard::W)) 
   { 

      // Start a jump if not already jumping 
      // but only if standing on a block (not falling) 
      if (!m_IsJumping && !m_IsFalling) 
      { 
         m_IsJumping = true; 
         m_TimeThisJump = 0; 
         m_JustJumped = true; 
      } 
   } 
   else 
   { 
      m_IsJumping = false; 
      m_IsFalling = true; 

   } 
   if (Keyboard::isKeyPressed(Keyboard::A)) 
   { 
      m_LeftPressed = true; 
   } 
   else 
   { 
      m_LeftPressed = false; 
   } 

   if (Keyboard::isKeyPressed(Keyboard::D)) 
   { 
      m_RightPressed = true; 
   } 
   else 
   { 
      m_RightPressed = false; 
   } 

   return m_JustJumped; 
} 

```

这段代码应该看起来很熟悉。我们使用 SFML 的`isKeyPressed`函数来查看*W*、*A*或*D*键是否被按下。

当按下*W*键时，玩家正在尝试跳跃。然后代码使用`if(!m_IsJumping && !m_IsFalling)`代码，检查角色是否已经在跳跃，而且也没有在下落。当这些测试都为真时，`m_IsJumping`被设置为`true`，`m_TimeThisJump`被设置为零，`m_JustJumped`被设置为 true。

当前两个测试不为`true`时，执行`else`子句，并将`m_Jumping`设置为`false`，将`m_IsFalling`设置为 true。

按下*A*和*D*键的处理就是简单地将`m_LeftPressed`和/或`m_RightPressed`设置为`true`或`false`。`update`函数现在将能够处理移动角色。

函数中的最后一行代码返回`m_JustJumped`的值。这将让调用代码知道是否需要播放跳跃音效。

我们现在将编写`Bob`类，尽管这几乎与`Thomas`类相同，但它具有不同的跳跃能力，不同的`Texture`，并且在键盘上使用不同的键。

## 编写 Bob.h

`Bob`类的结构与`Thomas`类相同。它继承自`PlayableCharacter`，有一个构造函数，并提供`handleInput`函数的定义。与`Thomas`相比的区别是，我们以不同的方式初始化了一些 Bob 的成员变量，并且我们也以不同的方式处理输入（在`handleInput`函数中）。让我们编写这个类并看看细节。

在**解决方案资源管理器**中右键单击**头文件**，然后选择**添加** | **新建项...**。在**添加新项**窗口中，通过左键单击**头文件**（`.h`）并在**名称**字段中键入`Bob.h`。最后，单击**添加**按钮。现在我们准备好为`Bob`类编写头文件了。

将以下代码添加到`Bob.h`文件中：

```cpp
#pragma once 
#include "PlayableCharacter.h" 

class Bob : public PlayableCharacter 
{ 
public: 
   // A constructor specific to Bob 
   Bob::Bob(); 

   // The overriden input handler for Bob 
   bool virtual handleInput(); 

}; 

```

上面的代码与`Thomas.h`文件相同，除了类名和构造函数名。

## 编写 Bob.cpp

在**解决方案资源管理器**中右键单击**源文件**，然后选择**添加** | **新建项...**。在**添加新项**窗口中，通过左键单击**C++文件（**`.cpp`**）**突出显示，然后在**名称**字段中键入`Thomas.cpp`。最后，单击**添加**按钮。我们现在准备为`Bob`类编写`.cpp`文件。

将`Bob`构造函数的代码添加到`Bob.cpp`文件中。注意纹理不同（`bob.png`），并且`m_JumpDuration`初始化为一个显着较小的值。Bob 现在是他自己独特的自己：

```cpp
#include "stdafx.h" 
#include "Bob.h" 
#include "TextureHolder.h" 

Bob::Bob() 
{ 
   // Associate a texture with the sprite 
   m_Sprite = Sprite(TextureHolder::GetTexture( 
      "graphics/bob.png")); 

   m_JumpDuration = .25; 
} 

```

在`Bob`构造函数之后立即添加`handleInput`代码：

```cpp
bool Bob::handleInput() 
{ 
   m_JustJumped = false; 

   if (Keyboard::isKeyPressed(Keyboard::Up)) 
   { 

      // Start a jump if not already jumping 
      // but only if standing on a block (not falling) 
      if (!m_IsJumping && !m_IsFalling) 
      { 
         m_IsJumping = true; 
         m_TimeThisJump = 0; 
         m_JustJumped = true; 
      } 

   } 
   else 
   { 
      m_IsJumping = false; 
      m_IsFalling = true; 

   } 
   if (Keyboard::isKeyPressed(Keyboard::Left)) 
   { 
      m_LeftPressed = true; 

   } 
   else 
   { 
      m_LeftPressed = false; 
   } 

   if (Keyboard::isKeyPressed(Keyboard::Right)) 
   { 

      m_RightPressed = true;; 

   } 
   else 
   { 
      m_RightPressed = false; 
   } 

   return m_JustJumped; 
} 

```

注意，代码几乎与`Thomas`类的`handleInput`函数中的代码相同。唯一的区别是我们对不同的键（**左**箭头键，**右**箭头键和**上**箭头键用于跳跃）做出响应。

现在我们有一个`PlayableCharacter`类，它已经被`Bob`和`Thomas`扩展，我们可以在游戏中添加一个`Bob`和一个`Thomas`实例。

# 更新游戏引擎以使用 Thomas 和 Bob

为了能够运行游戏并看到我们的新角色，我们必须声明它们的实例，调用它们的`spawn`函数，每帧更新它们，并每帧绘制它们。现在让我们来做这个。

## 更新 Engine.h 以添加 Bob 和 Thomas 的实例

打开`Engine.h`文件并添加下面突出显示的代码行，如下所示：

```cpp
#pragma once 
#include <SFML/Graphics.hpp> 
#include "TextureHolder.h" 
#include "Thomas.h"
#include "Bob.h" 

using namespace sf; 

class Engine 
{ 
private: 
   // The texture holder 
   TextureHolder th; 

 // Thomas and his friend, Bob
   Thomas m_Thomas;
   Bob m_Bob; 

   const int TILE_SIZE = 50; 
   const int VERTS_IN_QUAD = 4; 
   ... 
   ... 

```

现在我们有了`Thomas`和`Bob`的实例，它们都是从`PlayableCharacter`派生出来的。

## 更新输入函数以控制 Thomas 和 Bob

现在我们将添加控制这两个角色的能力。这段代码将放在代码的输入部分。当然，对于这个项目，我们有一个专门的`input`函数。打开`Input.cpp`并添加这段突出显示的代码：

```cpp
void Engine::input() 
{ 
   Event event; 
   while (m_Window.pollEvent(event)) 
   { 
      if (event.type == Event::KeyPressed) 
      { 
         // Handle the player quitting 
         if (Keyboard::isKeyPressed(Keyboard::Escape)) 
         { 
            m_Window.close(); 
         } 

         // Handle the player starting the game 
         if (Keyboard::isKeyPressed(Keyboard::Return)) 
         { 
            m_Playing = true; 
         } 

         // Switch between Thomas and Bob 
         if (Keyboard::isKeyPressed(Keyboard::Q)) 
         { 
            m_Character1 = !m_Character1; 
         } 

         // Switch between full and split-screen 
         if (Keyboard::isKeyPressed(Keyboard::E)) 
         { 
            m_SplitScreen = !m_SplitScreen; 
         } 
      } 
   } 

 // Handle input specific to Thomas
   if(m_Thomas.handleInput())
   {
     // Play a jump sound
   }

   // Handle input specific to Bob
   if(m_Bob.handleInput())
   {
     // Play a jump sound
   } 
} 

```

请注意，以前的代码是多么简单，因为所有功能都包含在`Thomas`和`Bob`类中。所有代码只需为`Thomas`和`Bob`类添加一个包含指令。然后，在`input`函数中，代码只需在`m_Thomas`和`m_Bob`上调用纯虚拟的`handleInput`函数。我们将每个调用包装在`if`语句中的原因是因为它们基于刚刚成功启动的新跳跃返回`true`或`false`。我们将在第十五章中处理播放跳跃音效，*声音空间化和 HUD*。

## 更新更新函数以生成和更新 PlayableCharacter 实例

这被分成两部分。首先，我们需要在新级别开始时生成 Bob 和 Thomas，其次，我们需要每帧更新（通过调用它们的`update`函数）。

### 生成 Thomas 和 Bob

随着项目的进展，我们需要在几个不同的地方调用我们的`Thomas`和`Bob`对象的生成函数。最明显的是，当新级别开始时，我们需要生成这两个角色。在接下来的章节中，随着我们需要在级别开始时执行的任务数量增加，我们将编写一个`loadLevel`函数。现在，让我们在`update`函数中调用`m_Thomas`和`m_Bob`的`spawn`函数，如下所示的突出显示的代码。添加这段代码，但请记住，这段代码最终将被删除并替换：

```cpp
void Engine::update(float dtAsSeconds) 
{ 
 if (m_NewLevelRequired)
   {
     // These calls to spawn will be moved to a new
     // loadLevel() function soon
     // Spawn Thomas and Bob
     m_Thomas.spawn(Vector2f(0,0), GRAVITY);
     m_Bob.spawn(Vector2f(100, 0), GRAVITY); 

     // Make sure spawn is called only once
     m_TimeRemaining = 10;
     m_NewLevelRequired = false;
   } 

   if (m_Playing) 
   { 
      // Count down the time the player has left 
      m_TimeRemaining -= dtAsSeconds; 

      // Have Thomas and Bob run out of time? 
      if (m_TimeRemaining <= 0) 
      { 
         m_NewLevelRequired = true; 
      } 

   }// End if playing 

} 

```

先前的代码只是调用`spawn`并传入游戏世界中的位置以及重力。该代码包裹在一个`if`语句中，检查是否需要新的级别。实际的生成代码将被移动到一个专门的`loadLevel`函数中，但`if`条件将成为完成项目的一部分。此外，`m_TimeRemaining`被设置为一个相当任意的 10 秒。

### 每帧更新 Thomas 和 Bob

接下来，我们将更新 Thomas 和 Bob。我们只需要调用它们的`update`函数并传入本帧所花费的时间。

添加下面突出显示的代码：

```cpp
void Engine::update(float dtAsSeconds) 
{ 
   if (m_NewLevelRequired) 
   { 
      // These calls to spawn will be moved to a new 
      // LoadLevel function soon 
      // Spawn Thomas and Bob 
      m_Thomas.spawn(Vector2f(0,0), GRAVITY); 
      m_Bob.spawn(Vector2f(100, 0), GRAVITY); 

      // Make sure spawn is called only once 
      m_NewLevelRequired = false; 
   } 

   if (m_Playing) 
   { 
 // Update Thomas
      m_Thomas.update(dtAsSeconds);

      // Update Bob
      m_Bob.update(dtAsSeconds); 

      // Count down the time the player has left 
      m_TimeRemaining -= dtAsSeconds; 

      // Have Thomas and Bob run out of time? 
      if (m_TimeRemaining <= 0) 
      { 
         m_NewLevelRequired = true; 
      } 

   }// End if playing 

} 

```

现在角色可以移动了，我们需要更新适当的`View`对象，使其围绕角色居中并使其成为关注的中心。当然，直到我们的游戏世界中有一些物体，才能实现实际移动的感觉。

添加下面片段中显示的突出代码：

```cpp
void Engine::update(float dtAsSeconds) 
{ 
   if (m_NewLevelRequired) 
   { 
      // These calls to spawn will be moved to a new 
      // LoadLevel function soon 
      // Spawn Thomas and Bob 
      m_Thomas.spawn(Vector2f(0,0), GRAVITY); 
      m_Bob.spawn(Vector2f(100, 0), GRAVITY); 

      // Make sure spawn is called only once 
      m_NewLevelRequired = false; 
   } 

   if (m_Playing) 
   { 
      // Update Thomas 
      m_Thomas.update(dtAsSeconds); 

      // Update Bob 
      m_Bob.update(dtAsSeconds); 

      // Count down the time the player has left 
      m_TimeRemaining -= dtAsSeconds; 

      // Have Thomas and Bob run out of time? 
      if (m_TimeRemaining <= 0) 
      { 
         m_NewLevelRequired = true; 
      } 

   }// End if playing 

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
} 

```

先前的代码处理了两种可能的情况。首先，`if(mSplitScreen)`条件将左侧视图定位在`m_Thomas`周围，右侧视图定位在`m_Bob`周围。当游戏处于全屏模式时执行的`else`子句测试`m_Character1`是否为`true`。如果是，则全屏视图（`m_MainView`）围绕 Thomas 居中，否则围绕 Bob 居中。您可能还记得玩家可以使用*E*键在分屏模式和全屏模式之间切换，使用*Q*键在全屏模式下切换 Bob 和 Thomas。我们在`Engine`类的`input`函数中编写了这些内容，回到第十二章。

## 绘制 Bob 和 Thomas

确保`Draw.cpp`文件已打开，并添加下面片段中显示的突出代码：

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

 // Draw thomas
     m_Window.draw(m_Thomas.getSprite());

     // Draw bob
     m_Window.draw(m_Bob.getSprite()); 
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

 // Draw bob
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

 // Draw thomas
     m_Window.draw(m_Thomas.getSprite());

     // Draw bob
     m_Window.draw(m_Bob.getSprite()); 

   } 

   // Draw the HUD 
   // Switch to m_HudView 
   m_Window.setView(m_HudView); 

   // Show everything we have just drawn 
   m_Window.display(); 
} 

```

请注意，我们在全屏、左侧和右侧都绘制了 Thomas 和 Bob。还要注意在分屏模式下绘制角色的微妙差异。在绘制屏幕的左侧时，我们改变了角色的绘制顺序，并在 Bob 之后绘制 Thomas。因此，Thomas 将始终位于左侧的顶部，Bob 位于右侧。这是因为左侧是为控制 Thomas 的玩家而设计的，右侧是为控制 Bob 的玩家而设计的。

您可以运行游戏，看到 Thomas 和 Bob 位于屏幕中央：

![绘制 Bob 和 Thomas](img/image_13_002.jpg)

如果按下*Q*键从 Thomas 切换到 Bob，您将看到`View`进行了轻微调整。如果移动任何一个角色向左或向右（Thomas 使用*A*和*D*，Bob 使用箭头键），您将看到它们相对移动。

尝试按下*E*键在全屏和分屏模式之间切换。然后尝试再次移动两个角色以查看效果。在下面的截图中，您可以看到 Thomas 始终位于左侧窗口的中心，Bob 始终位于右侧窗口的中心：

![绘制 Bob 和 Thomas](img/image_13_003.jpg)

如果您让游戏运行足够长的时间，角色将每十秒重新生成在它们的原始位置。这是我们在完成游戏时需要的功能的开端。这种行为是由`m_TimeRemaining`变为负值，然后将`m_NewLevelRequired`变量设置为`true`引起的。

还要注意，直到我们绘制了关卡的细节，我们才能看到移动的完整效果。实际上，虽然看不到，但两个角色都在以每秒 300 像素的速度持续下落。由于摄像机每帧都围绕它们居中，并且游戏世界中没有其他物体，我们看不到这种向下运动。

如果您想自己演示这一点，只需按照以下代码中所示更改对`m_Bob.spawn`的调用：

```cpp
m_Bob.spawn(Vector2f(0,0), 0); 

```

现在 Bob 没有重力效果，Thomas 会明显远离他。如下截图所示：

![绘制 Bob 和 Thomas](img/image_13_004.jpg)

在接下来的章节中，我们将添加一些可玩的关卡进行交互。

# 常见问题解答

Q）我们学习了多态性，但到目前为止，我没有注意到游戏代码中有任何多态性。

A）在接下来的章节中，当我们编写一个以`PlayableCharacter`作为参数的函数时，我们将看到多态性的实际应用。我们将看到如何可以将 Bob 或 Thomas 传递给这个新函数，并且无论使用哪个，它都能正常工作。

# 摘要

在本章中，我们学习了一些新的 C++概念。首先，继承允许我们扩展一个类并获得其所有功能。我们还学到，我们可以将变量声明为受保护的，这将使子类可以访问它们，但它们仍将被封装（隐藏）在所有其他代码之外。我们还使用了纯虚函数，这使得一个类成为抽象类，意味着该类不能被实例化，因此必须从中继承/扩展。我们还介绍了多态的概念，但需要等到下一章才能在我们的游戏中使用它。

接下来，我们将为游戏添加一些重要功能。在接下来的一章中，Thomas 和 Bob 将会行走、跳跃和下落。他们甚至可以跳在彼此的头上，以及探索从文本文件加载的一些关卡设计。
