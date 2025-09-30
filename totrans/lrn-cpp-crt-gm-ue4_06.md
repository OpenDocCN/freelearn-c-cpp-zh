# 第六章：对象、类和继承

在上一章中，我们讨论了函数作为一种将相关代码行捆绑在一起的方式。我们讨论了函数如何抽象实现细节，以及 `sqrt()` 函数不需要你了解其内部工作原理即可使用它来找到根。这是一件好事，主要是因为它节省了程序员的精力和时间，同时使寻找平方根的实际工作变得更容易。当我们讨论对象时，这个 *抽象* 原理将再次出现。

简而言之，对象将方法和相关数据绑定到一个单一的结构中。这个结构被称为 *类*。使用对象的主要思想是为游戏中的每个事物创建一个代码表示。代码中表示的每个对象都将有数据和相关的函数来操作这些数据。因此，你会有一个 *对象* 来表示玩家实例以及相关的函数，如 `jump()`、`shoot()` 和 `pickupItem()`。你也会有一个对象来表示每个怪物实例以及相关的函数，如 `growl()`、`attack()` 和可能还有 `follow()`。

然而，对象是变量的一种类型，并且只要你在那里保持它们，对象就会留在内存中。当你创建代表游戏中的事物的对象实例时，你将创建一个对象实例，当你代表游戏中的事物死亡时，你将销毁对象实例。

对象可以用来表示游戏中的事物，但它们也可以用来表示任何其他类型的事物。例如，你可以将一个图像存储为对象。数据字段将包括图像的宽度、高度以及其中的像素集合。C++ 字符串也是对象。

### 小贴士

本章包含许多可能一开始难以理解的关键词，包括 `virtual` 和 `abstract`。

不要让本章更难的部分让你感到困惑。我为了完整性而包括了众多高级概念的描述。然而，请记住，你不需要完全理解本章中的所有内容就能在 UE4 中编写有效的 C++ 代码。理解它是有帮助的，但如果某些内容让你感到困惑，不要陷入困境。阅读它，然后继续前进。可能的情况是，你一开始可能不会理解，但记住在编码时对相关概念的参考。然后，当你再次打开这本书时，“哇！”它就会变得有意义。

# 结构体对象

在 C++ 中，对象基本上是由更简单的类型组成的复合类型。C++ 中最基本的对象是 `struct`。我们使用 `struct` 关键字将许多较小的变量粘合在一起形成一个大的变量。如果你还记得，我们在 第二章 中简要介绍了 `struct`，*变量和内存*。让我们回顾一下这个简单的例子：

```cpp
struct Player
{
  string name;
  int hp;
};
```

这是`Player`对象的结构定义。玩家有一个`string`类型的`name`和一个表示`hp`值的整数。

如果你还记得第二章中的内容，即*变量和内存*，我们创建`Player`对象实例的方式是这样的：

```cpp
Player me;    // create an instance of Player, called me
```

从这里，我们可以这样访问`me`对象的字段：

```cpp
me.name = "Tom";
me.hp = 100;
```

## 成员函数

现在，这里是激动人心的部分。我们可以通过在`struct Player`定义内部编写这些函数来将成员函数附加到`struct`定义上。

```cpp
struct Player
{
  string name;
  int hp;
  // A member function that reduces player hp by some amount
  void damage( int amount )	
  {
    hp -= amount;
  }
  void recover( int amount )
  {
    hp += amount;
}
};
```

成员函数就是一个在`struct`或`class`定义内部声明的 C++函数。这不是一个好主意吗？

这里有一个有点奇怪的想法，所以我就直接说出来吧。`struct Player`的变量对所有`struct Player`内部的函数都是可访问的。在`struct Player`的每个成员函数内部，我们实际上可以像访问局部变量一样访问`name`和`hp`变量。换句话说，`struct Player`的`name`和`hp`变量在`struct Player`的所有成员函数之间是共享的。

### `this`关键字

在一些 C++代码（在后面的章节中），你会看到更多对`this`关键字的引用。`this`关键字是一个指向当前对象的指针。例如，在`Player::damage()`函数内部，我们可以明确写出对`this`的引用：

```cpp
void damage( int amount )
{
  this->hp -= amount;
}
```

`this`关键字仅在成员函数内部有意义。我们可以在成员函数中明确包含`this`关键字的用法，但如果没有写`this`，则默认我们谈论的是当前对象的`hp`。

## 字符串是对象吗？

是的！每次你使用字符串变量时，你都是在使用一个对象。让我们尝试一下`string`类的成员函数。

```cpp
#include <iostream>
#include <string>
using namespace std;
int main()
{
  string s = "strings are objects";
  s.append( "!!" ); // add on "!!" to end of the string!
  cout << s << endl;
}
```

我们在这里所做的是使用`append()`成员函数在字符串的末尾添加两个额外的字符（`!!`）。成员函数始终应用于调用成员函数的对象（点左侧的对象）。

### 小贴士

要查看对象上可用的成员和成员函数列表，请在 Visual Studio 中输入对象的变量名，然后输入一个点（`.`），然后按*Ctrl*和空格键。成员列表将弹出。

![字符串是对象吗？](img/00060.jpeg)

按下*Ctrl*和空格键将显示成员列表。

## 调用成员函数

成员函数可以使用以下语法调用：

```cpp
objectName.memberFunction();
```

调用成员函数的对象位于点的左侧。要调用的成员函数位于点的右侧。成员函数调用始终后跟圆括号`()`，即使没有传递参数给括号。

因此，在程序中怪物攻击的部分，我们可以这样减少玩家的`hp`值：

```cpp
player.damage( 15 );  // player takes 15 damage
```

这不是比以下内容更易读吗？

```cpp
player.hp -= 15;      // player takes 15 damage
```

### 小贴士

当成员函数和对象被有效使用时，你的代码将读起来更像散文或诗歌，而不是一堆操作符的组合。

除了美观和可读性之外，编写成员函数的目的是什么？现在，在`Player`对象外部，我们只用一行代码就可以做更多的事情，而不仅仅是减少`hp`成员`15`。我们还可以在减少玩家的`hp`时做其他事情，比如考虑玩家的护甲，检查玩家是否无敌，或者当玩家受伤时产生其他效果。当玩家受伤时应该由`damage()`函数来抽象处理。

现在想想如果玩家有一个护甲等级。让我们给`struct Player`添加一个护甲等级字段：

```cpp
struct Player
{
  string name;
  int hp;
  int armorClass;
};
```

我们需要通过玩家的护甲等级来减少玩家收到的伤害。所以现在我们可以输入一个公式来减少`hp`。我们可以通过直接访问`player`对象的数据字段来实现非面向对象的方式：

```cpp
player.hp -= 15 – player.armorClass; // non OOP
```

否则，我们可以通过编写一个成员函数来改变`player`对象的数据成员，以实现面向对象的方式。在`Player`对象内部，我们可以编写一个成员函数`damage()`：

```cpp
struct Player
{
  string name;
  int hp;
  int armorClass; 
void damage( int dmgAmount )	
  {
    hp -= dmgAmount - armorClass;
  }
};
```

### 练习

1.  在前面的代码中，玩家的`damage`函数中有一个微小的错误。你能找到并修复它吗？提示：如果造成的伤害小于玩家的`armorClass`，会发生什么？

1.  只有一个护甲等级的数字并不能提供足够的关于护甲的信息！护甲的名字是什么？它看起来像什么？为玩家的护甲设计一个`struct`函数，包含名称、护甲等级和耐久性评分字段。

### 解决方案

解决方案在下一节中列出的`struct`玩家代码中，*私有和封装*。

使用以下代码如何？

```cpp
struct Armor
{
  string name;
  int armorClass;
  double durability;
};
```

`struct Player`中将会放置一个`Armor`实例：

```cpp
struct Player
{
  string name;
  int hp;
  Armor armor; // Player has-an Armor
};
```

这意味着玩家有护甲。请记住这一点——我们将在以后探讨`has-a`与`is-a`关系。

## 私有和封装

因此，我们现在已经定义了一些成员函数，其目的是修改和维护我们的`Player`对象的数据成员，但有些人提出了一个论点。

论点如下：

+   一个对象的数据成员应该只通过其成员函数访问，永远不要直接访问。

这意味着你不应该直接从对象外部访问对象的数据成员，换句话说，直接修改玩家的`hp`：

```cpp
player.hp -= 15 – player.armorClass; // bad: direct member access
```

这应该被禁止，并且应该强制类用户使用适当的成员函数来更改数据成员的值：

```cpp
player.damage( 15 );	// right: access thru member function
```

这个原则被称为*封装*。封装的概念是每个对象都应该只通过其成员函数进行交互。封装表示原始数据成员不应直接访问。

封装背后的原因如下：

+   **为了使类自包含**：封装背后的主要思想是，当对象以这种方式编程时，它们工作得最好，即它们可以管理和维护自己的内部状态变量，而无需类外部的代码检查该类的私有数据。当对象以这种方式编码时，会使对象更容易处理，也就是说，更容易阅读和维护。要使玩家对象跳跃，你只需调用`player.jump()`；让玩家对象管理其`y-height`位置的状态变化（使玩家跳跃！）当对象的内部成员没有暴露时，与该对象的交互会更加容易和高效。仅与对象的公共成员函数交互；让对象管理其内部状态（我们将在稍后解释`private`和`public`关键字）。 

+   **为了避免破坏代码**：当类外部的代码仅与该类的公共成员函数交互（类的公共接口）时，对象的内部状态管理可以自由更改，而不会破坏任何调用代码。这样，如果对象的内部数据成员因任何原因而更改，只要成员函数保持不变，所有使用该对象的代码仍然有效。

那么，我们如何防止程序员犯错误并直接访问数据成员呢？C++引入了*访问修饰符*的概念，以防止访问对象的内部数据。

下面是如何使用访问修饰符来禁止从`struct Player`外部访问`struct Player`的某些部分。

你首先需要决定你希望从类外部访问的`struct`定义的哪些部分。这些部分将被标记为`public`。所有其他将不会从`struct`外部访问的区域将被标记为`private`，如下所示：

```cpp
struct Player
{
private:        // begins private section.. cannot be accessed 
                // outside the class until
  string name;
  int hp; 
  int armorClass;
public:         //  until HERE. This begins the public section
  // This member function is accessible outside the struct
  // because it is in the section marked public:
  void damage( int amount )
  {
    int reduction = amount – armorClass;
    if( reduction < 0 ) // make sure non-negative!
      reduction = 0;
    hp -= reduction;
  }
};
```

## 有些人喜欢将其设置为`public`

有些人会毫不犹豫地使用`public`数据成员，并且不封装他们的对象。尽管这被视为不良的面向对象编程实践，但这仍然是一个个人喜好问题。

然而，UE4 中的类有时确实会使用`public`成员。这是一个判断问题；数据成员应该是`public`还是`private`完全取决于程序员。

随着经验的积累，你会发现，当你将应该为`private`的数据成员设置为`public`时，有时你会陷入需要大量重构的情况。

# 类与结构体

你可能已经看到了另一种声明对象的方法，使用`class`关键字而不是`struct`，如下面的代码所示：

```cpp
class Player // we used class here instead of struct!
{
  string name;
  //
};
```

C++中的`class`和`struct`关键字几乎相同。`class`和`struct`之间只有一个区别，那就是在`struct`关键字内部的数据成员默认会被声明为`public`，而在`class`关键字中，类内部的数据成员默认会被声明为`private`。（这就是为什么我使用`struct`来引入对象；我不想在`class`的第一行无解释地放置`public`。） 

通常，对于简单类型，不使用封装，没有很多成员函数，并且必须与 C 语言向后兼容的情况，我们更倾向于使用`struct`。类几乎在其他所有地方都被使用。

从现在起，让我们使用`class`关键字而不是`struct`。

# 获取器和设置器

你可能已经注意到，一旦我们将`private`添加到`Player`类定义中，我们就不能再从`Player`类外部读取或写入玩家的名称。

如果我们尝试使用以下代码读取名称：

```cpp
Player me;
cout << me.name << endl;
```

或者按照以下方式写入名称：

```cpp
me.name = "William";
```

使用带有`private`成员的`struct Player`定义，我们将得到以下错误：

```cpp
main.cpp(24) : error C2248: 'Player::name' : cannot access private member declared in class 'Player'

```

这正是我们当我们将`name`字段标记为`private`时所期望的。我们使其在`Player`类外部完全不可访问。

## 获取器

获取器（也称为访问器函数）用于将内部数据成员的副本传递给调用者。为了读取玩家的名称，我们需要在`Player`类中添加一个特定的成员函数来检索该`private`数据成员的副本：

```cpp
class Player
{
private:
  string name;  // inaccessible outside this class!
                //  rest of class as before
public:
  // A getter function retrieves a copy of a variable for you
  string getName()
{
  return name;
}
};
```

因此，现在可以读取玩家的名称信息。我们可以通过以下代码语句来实现：

```cpp
cout << player.getName() << endl;
```

获取器用于检索那些从类外部无法访问的`private`成员。

### 提示

**实际技巧–const 关键字**

在类内部，你可以在成员函数声明中添加`const`关键字。`const`关键字的作用是向编译器承诺，运行此函数后对象的内部状态不会改变。添加`const`关键字的样子如下：

```cpp
string getName() const
{
  return name;
}
```

在标记为`const`的成员函数内部不能对数据成员进行赋值。由于对象的内部状态在运行`const`函数后保证不会改变，编译器可以在函数调用`const`成员函数时进行一些优化。

## 设置器

设置器（也称为修改器函数或突变函数）是一个成员函数，其唯一目的是更改类内部内部变量的值，如下面的代码所示：

```cpp
class Player
{
private:
  string name;  // inaccessible outside this class!
                //  rest of class as before
public:
  // A getter function retrieves a copy of a variable for you
  string getName()
{
  return name;
}
void setName( string newName )
{
  name = newName;
}
};
```

因此，我们仍然可以通过设置器函数从类外部更改`class`的`private`函数，但只能通过这种方式。

## 但获取/设置操作的意义何在？

所以，当新手程序员第一次遇到对 `private` 成员进行获取/设置操作时，他们首先想到的问题是不是获取/设置自我矛盾？我的意思是，当我们打算以另一种方式再次暴露相同的数据时，隐藏对数据成员的访问有什么意义？这就像说，“你不能有任何巧克力，因为它们是私有的，除非你说请 `getMeTheChocolate()`。然后，你就可以有巧克力了。”

一些经验丰富的程序员甚至将获取/设置函数缩短为一行，如下所示：

```cpp
string getName(){ return name; }
void setName( string newName ){ name = newName; }
```

让我们来回答这个问题。一个获取/设置对不是通过完全暴露数据来破坏封装性吗？

答案有两个方面。首先，获取成员函数通常只返回被访问的数据成员的副本。这意味着原始数据成员的值仍然受到保护，并且不能通过 `get()` 操作进行修改。

`Set()`（变更器方法）操作有点反直觉。如果设置器是一个 `passthru` 操作，例如 `void setName( string newName ) { name=newName; }`，那么拥有设置器可能看起来没有意义。使用变更器方法而不是直接覆盖变量有什么优势？

使用变更器方法的论点是，在变量赋值之前编写额外的代码来保护变量免受错误值的侵害。比如说，我们有一个 `hp` 数据成员的设置器，它看起来可能像这样：

```cpp
void setHp( int newHp )
{
  // guard the hp variable from taking on negative values
  if( newHp < 0 )
  {
    cout << "Error, player hp cannot be less than 0" << endl;
    newHp = 0;
  }
  hp = newHp;
}
```

变更器方法旨在防止内部 `hp` 数据成员取负值。你可能认为变更器方法有点事后诸葛亮。责任应该由调用代码在调用 `setHp( -2 )` 之前检查它设置的值，而不是只在变更器方法中捕获这个问题吗？你不能使用一个 `public` 成员变量，并将确保变量不取无效值的责任放在调用代码中，而不是在设置器中吗？你可以。

然而，这正是使用变更器方法的核心原因。变更器方法背后的想法是，调用代码可以将任何它想要的值传递给 `setHp` 函数（例如，`setHp( -2 )`），而不必担心它传递给函数的值是否有效。然后 `setHp` 函数负责确保该值对 `hp` 变量有效。

一些程序员认为直接变更函数，如 `getHp()`/`setHp()`，是一种代码恶臭。一般来说，代码恶臭是一种不良的编程实践，人们并没有明显注意到，除了有一种感觉，觉得某些事情做得不够优化。他们认为可以编写高级成员函数来代替变更器。例如，我们不应该有 `setHp()` 成员函数，而应该有 `public` 成员函数，如 `heal()` 和 `damage()`。关于这个话题的文章可在 [`c2.com/cgi/wiki?AccessorsAreEvil`](http://c2.com/cgi/wiki?AccessorsAreEvil) 查阅。

# 构造函数和析构函数

在你的 C++代码中，构造函数是一个简单的函数，当 C++对象首次创建时运行一次。析构函数在 C++对象被销毁时运行一次。比如说我们有以下程序：

```cpp
#include <iostream>
#include <string>
using namespace std;
class Player
{
private:
  string name;  // inaccessible outside this class!
public:
  string getName(){ return name; }
// The constructor!
  Player()
  {
    cout << "Player object constructed" << endl;
    name = "Diplo";
  }
  // ~Destructor (~ is not a typo!)
  ~Player()
  {
    cout << "Player object destroyed" << endl;
  }
};

int main()
  {
    Player player;
    cout << "Player named '" << player.getName() << "'" << endl;
  }
  // player object destroyed here
```

因此，我们在这里创建了一个`Player`对象。这段代码的输出将如下所示：

```cpp
Player object constructed
Player named 'Diplo'
Player object destroyed
```

对象构造过程中发生的第一件事是构造函数实际上运行了。这会打印出`Player object constructed`这一行。随后，会打印出带有玩家名字的行：`Player named 'Diplo'`。为什么玩家名字叫*Diplo*？因为这是在`Player()`构造函数中分配的名字。

最后，在程序结束时，调用玩家析构函数，我们看到`Player object destroyed`。当玩家对象在`main()`函数的末尾（在`main`的`}`处）超出作用域时，玩家对象被销毁。

那么，构造函数和析构函数有什么好处？它们看起来就是用来：设置和销毁对象。构造函数可以用来初始化数据字段，析构函数可以用来删除任何动态分配的资源（我们还没有涉及动态分配的资源，所以现在不用担心这个最后一点）。

# 类继承

当你想基于某个现有的代码类创建一个新的、功能更强大的代码类时，你会使用继承。继承是一个复杂的话题。让我们从*派生类*（或子类）的概念开始。

## 派生类

考虑继承最自然的方式是通过类比动物王国。以下截图显示了生物的分类：

![派生类](img/00061.jpeg)

这个图表示的意思是**狗**、**猫**、**马**、**人**都是**哺乳动物**。这意味着狗、猫、马和人都有一些共同的特征，例如有共同的器官（大脑有新皮层、肺、肝脏和女性的子宫），但在其他方面完全不同。它们走路的方式不同。它们说话的方式也不同。

如果你正在编写生物的代码，你只需要编写一次共同的功能。然后，你会为狗、猫、马和人这些类分别实现不同部分的代码。

上述图示的一个具体例子如下：

```cpp
#include <iostream>
using namespace std;
class Mammal
{
protected:
  // protected variables are like privates: they are
  // accessible in this class but not outside the class.
  // the difference between protected and private is
  // protected means accessible in derived subclasses also
int hp;
  double speed;

public:
  // Mammal constructor – runs FIRST before derived class ctors!
Mammal()
{
  hp = 100;
  speed = 1.0;
  cout << "A mammal is created!" << endl;
}
~Mammal()
{
  cout << "A mammal has fallen!" << endl;
}
// Common function to all Mammals and derivatives
  void breathe()
  {
    cout << "Breathe in.. breathe out" << endl;
  }
  virtual void talk()
  {
    cout << "Mammal talk.. override this function!" << endl;
  }
  // pure virtual function, (explained below)
  virtual void walk() = 0;
};

// This next line says "class Dog inherits from class Mammal"
class Dog : public Mammal // : is used for inheritance
{
public:
  Dog()
  {
cout << "A dog is born!" << endl;
}
~Dog()
{
  cout << "The dog died" << endl;
}
  virtual void talk() override
  {
    cout << "Woof!" << endl; // dogs only say woof!
  }
  // implements walking for a dog
  virtual void walk() override
  {
    cout << "Left front paw & back right paw, right front paw &  back left paw.. at the speed of " << speed << endl;
  }
};

class Cat : public Mammal
{
public:
  Cat()
  {
    cout << "A cat is born" << endl;
  }
  ~Cat()
  {
    cout << "The cat has died" << endl;
  }
virtual void talk() override
  {
    cout << "Meow!" << endl;
  }
// implements walking for a cat.. same as dog!
  virtual void walk() override
  {
    cout << "Left front paw & back right paw, right front paw &  back left paw.. at the speed of " << speed << endl;
  }
};

class Human : public Mammal
{
// Data member unique to Human (not found in other Mammals)
  bool civilized;
public:
  Human()
  {
    cout << "A new human is born" << endl;
    speed = 2.0; // change speed. Since derived class ctor
    // (ctor is short for constructor!) runs after base 
    // class ctor, initialization sticks initialize member 
    // variables specific to this class
    civilized = true;
  }
  ~Human()
  {
    cout << "The human has died" << endl;
  }
  virtual void talk() override
  {
    cout << "I'm good looking for a .. human" << endl;
  }
// implements walking for a human..
  virtual void walk() override
  {
    cout << "Left, right, left, right at the speed of " << speed  << endl;
  }
  // member function unique to human derivative
  void attack( Human & other )
  {
    // Human refuses to attack if civilized
    if( civilized )
      cout << "Why would a human attack another? Je refuse" <<  endl;
    else
      cout << "A human attacks another!" << endl;
  }
};

int main()
{
  Human human;
  human.breathe(); // breathe using Mammal base class  functionality
  human.talk();
  human.walk();

  Cat cat;
  cat.breathe(); // breathe using Mammal base class functionality
  cat.talk();
  cat.walk();

  Dog dog;
  dog.breathe();
  dog.talk();
  dog.walk();
}
```

所有的`Dog`、`Cat`和`Human`都从`class Mammal`继承。这意味着狗、猫和人是哺乳动物，还有更多。

### 继承的语法

继承的语法相当简单。让我们以`Human`类定义为例。以下截图是一个典型的继承语句：

![继承的语法](img/00062.jpeg)

冒号（**:**）左边的类是新派生类，冒号右边的类是基类。

### 继承有什么作用？

继承的目的是让派生类承担基类的所有特性（数据成员、成员函数），然后在此基础上扩展更多的功能。例如，所有哺乳动物都有一个`breathe()`函数。通过从`Mammal`类继承，`Dog`、`Cat`和`Human`类都自动获得了`breathe()`的能力。

继承减少了代码的重复，因为我们不需要为`Dog`、`Cat`和`Human`重新实现常见功能（如`.breathe()`）。相反，这些派生类都享受了在`class Mammal`中定义的`breathe()`函数的重用。

然而，只有`Human`类有`attack()`成员函数。这意味着在我们的代码中，只有`Human`类会攻击。除非你在`class Cat`（或`class Mammal`）内部编写一个成员函数`attack()`，否则`cat.attack()`函数将引入编译器错误。

## is-a 关系

继承通常被说成是`is-a`关系。当一个`Human`类从`Mammal`类继承时，我们说人类*是*哺乳动物。

![is-a 关系](img/00063.jpeg)

人类继承了哺乳动物的所有特征

例如，一个`Human`对象在其内部包含一个`Mammal`函数，如下所示：

```cpp
class Human
{
  Mammal mammal;
};
```

在这个例子中，我们可以说人类*有一个*`Mammal`在其某个地方（如果人类怀孕或以某种方式携带哺乳动物，这将是合理的）。

![is-a 关系](img/00064.jpeg)

这个人类类实例中附有一种哺乳动物

记住我们之前在`Player`内部给了它一个`Armor`对象。对于`Player`对象来说，从`Armor`类继承是没有意义的，因为这说不通*玩家是装甲*。在代码设计中决定一个类是否从另一个类继承时（例如，Human 类从 Mammal 类继承），你必须始终能够舒适地说出类似于 Human 类*是*Mammal 的话。如果*是*这个陈述听起来不对，那么很可能是继承对于这对对象的关系是错误的。

在前面的例子中，我们在这里引入了一些新的 C++关键字。第一个是`protected`。

## protected 变量

一个`protected`成员变量与一个`public`或`private`变量不同。所有这三个类别的变量都可以在定义它们的类内部访问。它们之间的区别在于对类外部的可访问性。一个`public`变量可以在类内部和类外部访问。一个`private`变量可以在类内部访问，但不能在类外部访问。一个`protected`变量可以在类内部和派生子类内部访问，但不能在类外部访问。因此，`class Mammal`中的`hp`和`speed`成员将在派生类 Dog、Cat、Horse 和 Human 中可访问，但不在这些类外部（例如在`main()`中）。

## 虚函数

虚函数是一个成员函数，其实现可以在派生类中重写。在这个例子中，`talk()` 成员函数（在 `class Mammal` 中定义）被标记为 `virtual`。这意味着派生类可能会也可能不会选择实现自己的 `talk()` 函数版本。

## 纯虚函数（以及抽象类）

纯虚函数是要求在派生类中重写的函数。`class Mammal` 中的 `walk()` 函数是纯虚的；它被声明如下：

```cpp
virtual void walk() = 0;
```

之前代码末尾的 `= 0` 部分使得函数成为纯虚函数。

`class Mammal` 中的 `walk()` 函数是纯虚函数，这使得 `Mammal` 类成为抽象类。在 C++ 中，抽象类是至少有一个纯虚函数的任何类。

如果一个类包含一个纯虚函数并且是抽象的，那么这个类不能直接实例化。也就是说，你现在不能创建一个 `Mammal` 对象，因为 `walk()` 是一个纯虚函数。如果你尝试执行以下代码，你会得到一个错误：

```cpp
int main()
{
  Mammal mammal;
}
```

如果你尝试创建一个 `Mammal` 对象，你会得到以下错误：

```cpp
error C2259: 'Mammal' : cannot instantiate abstract class
```

然而，你可以创建 `class Mammal` 的派生类的实例，只要这些派生类实现了所有的纯虚成员函数。

# 多重继承

并非所有听起来很好的多重继承都是如此。多重继承是指一个派生类从多个基类继承。通常，如果我们继承的多个基类完全无关，这个过程会顺利无误。

例如，我们可以有一个 `Window` 类，它从 `SoundManager` 和 `GraphicsManager` 基类继承。如果 `SoundManager` 提供一个成员函数 `playSound()`，而 `GraphicsManager` 提供一个成员函数 `drawSprite()`，那么 `Window` 类将能够无障碍地使用这些额外的功能。

![多重继承](img/00065.jpeg)

游戏窗口从声音管理和图形管理器继承意味着游戏窗口将拥有这两组功能

然而，多重继承可能会有负面影响。比如说，我们想要创建一个从 `Donkey` 和 `Horse` 类派生的 `Mule` 类。然而，`Donkey` 和 `Horse` 类都从基类 `Mammal` 继承。我们立刻遇到了问题！如果我们调用 `mule.talk()`，但 `mule` 没有重写 `talk()` 函数，应该调用哪个成员函数，`Horse` 的还是 `Donkey` 的？这是模糊的。

## 私有继承

C++中较少讨论的特性是 `私有` 继承。每当一个类以公有方式从另一个类继承时，它对其所属的父类中的所有代码都是已知的。例如：

```cpp
class Cat : public Mammal
```

这意味着所有代码都将知道 `Cat` 是 `Mammal` 的一个对象，并且可以使用基类 `Mammal*` 指针指向 `Cat*` 实例。例如，以下代码将是有效的：

```cpp
Cat cat;
Mammal* mammalPtr = &cat; // Point to the Cat as if it were a 
                          // Mammal
```

如果`Cat`从`Mammal`公开继承，前面的代码是好的。私有继承是外部`Cat`类不允许知道父类的地方：

```cpp
class Cat : private Mammal
```

在这里，外部调用代码不会“知道”`Cat`类是从`Mammal`类派生的。当继承为`private`时，编译器不允许将`Cat`实例强制转换为`Mammal`基类。当你需要隐藏某个类从某个父类派生的事实时，请使用`private`继承。

然而，在实际应用中，私有继承很少使用。大多数类只是使用 `public` 继承。如果你想了解更多关于私有继承的信息，请参阅[`stackoverflow.com/questions/406081/why-should-i-avoid-multiple-inheritance-in-c`](http://stackoverflow.com/questions/406081/why-should-i-avoid-multiple-inheritance-in-c)。

# 将你的类放入头文件中

到目前为止，我们的类只是粘贴在 `main()` 之前。如果你继续这样编程，你的代码将全部在一个文件中，看起来像一大团混乱。

因此，将你的类组织到单独的文件中是一种良好的编程实践。当项目中有多个类时，这使得单独编辑每个类的代码变得容易得多。

从之前的 `class Mammal` 和其派生类开始。我们将把这个例子适当地组织到单独的文件中。让我们分步骤来做：

1.  在你的 C++ 项目中创建一个名为 `Mammal.h` 的新文件。将整个 `Mammal` 类复制并粘贴到该文件中。注意，由于 `Mammal` 类使用了 `cout`，我们也在该文件中写了一个 `#include <iostream>` 语句。

1.  在你的 `Source.cpp` 文件顶部写一个 " `#include` `Mammal.h`" 语句。

如下截图所示，这是它的一个示例：

![将你的类放入头文件中](img/00066.jpeg)

当代码编译时，这里发生的情况是将整个 `Mammal` 类复制并粘贴（`#include`）到包含 `main()` 函数的 `Source.cpp` 文件中，其余的类都从 `Mammal` 派生。由于 `#include` 是一个复制粘贴函数，代码将完全按照之前的方式运行；唯一的区别是它将组织得更好，更容易查看。在这一步编译和运行你的代码，以确保它仍然可以工作。

### 小贴士

确保你的代码经常编译和运行，尤其是在重构时。当你不知道规则时，你肯定会犯很多错误。这就是为什么你应该只在小步骤中重构。重构是我们现在正在进行的活动的名称——我们正在重新组织源代码，使其对我们代码库的其他读者来说更有意义。重构通常不涉及太多重写。

下一步你需要做的是将 Dog、Cat 和 Human 类隔离到它们自己的文件中。为此，创建 `Dog.h`、`Cat.h` 和 `Human.h` 文件并将它们添加到你的项目中。

让我们从 Dog 类开始，如下截图所示：

![将你的类放入标题](img/00067.jpeg)

如果你使用这个设置并尝试编译和运行你的项目，你将看到如下所示的 **'Mammal' : 'class' 类型重新定义** 错误，如下面的截图所示：

![将你的类放入标题](img/00068.jpeg)

这个错误意味着 `Mammal.h` 在你的项目中已被包含两次，一次在 `Source.cpp` 中，然后又在 `Dog.h` 中。这意味着实际上有两个版本的 `Mammal` 类被添加到编译代码中，C++ 不确定使用哪个版本。

有几种方法可以解决这个问题，但最简单（也是虚幻引擎使用的方法）是使用 `#pragma once` 宏，如下面的截图所示：

![将你的类放入标题](img/00069.jpeg)

我们在每个头文件顶部写上 `#pragma once`。这样，当第二次包含 `Mammal.h` 时，编译器不会再次复制粘贴其内容，因为它之前已经包含过了，其内容实际上已经在编译文件组中。

对 `Cat.h` 和 `Human.h` 也做同样的事情，然后将它们都包含到你的 `Source.cpp` 文件中，你的 `main()` 函数就在那里。

![将你的类放入标题](img/00070.jpeg)

包含所有类的图

现在我们已经将所有类包含到你的项目中，代码应该可以编译并运行。

## .h 和 .cpp

下一个组织级别是将类声明留在头文件（`.h`）中，并将实际函数实现体放入一些新的 `.cpp` 文件中。同时，将现有成员留在 `class Mammal` 声明中。

对于每个类，执行以下操作：

1.  删除所有函数体（`{` 和 `}` 之间的代码）并替换为仅有一个分号。对于 `Mammal` 类，它看起来如下所示：

    ```cpp
    // Mammal.h
    #pragma once
    class Mammal
    {
    protected:
      int hp;
      double speed;

    public:
      Mammal();
      ~Mammal();
      void breathe();
      virtual void talk();
      // pure virtual function, 
      virtual void walk() = 0;
    };
    ```

1.  创建一个名为 `Mammal.cpp` 的新 `.cpp` 文件。然后只需将成员函数体放入此文件中：

    ```cpp
    // Mammal.cpp
    #include <iostream>
    using namespace std;

    #include "Mammal.h"
    Mammal::Mammal() // Notice use of :: (scope resolution operator)
    {
      hp = 100;
      speed = 1.0;
      cout << "A mammal is created!" << endl;
    }
    Mammal::~Mammal()
    {
      cout << "A mammal has fallen!" << endl;
    }
    void Mammal::breathe()
    {
      cout << "Breathe in.. breathe out" << endl;
    }
    void Mammal::talk()
    {
      cout << "Mammal talk.. override this function!" << endl;
    }
    ```

重要的是要注意在声明成员函数体时使用类名和作用域解析运算符（双冒号）。我们用 `Mammal::` 前缀所有属于 `Mammal` 类的成员函数。

注意，纯虚函数没有函数体；它不应该有！纯虚函数只是在基类中声明（并初始化为 0），然后在派生类中实现。

## 练习

将上述不同的生物类完全分离成类头文件 (.h) 和类定义文件 (.cpp)

# 摘要

你在 C++ 中学习了关于对象的知识；它们是将数据成员和成员函数结合在一起形成代码包的代码片段，称为 `class` 或 `struct`。面向对象编程意味着你的代码将充满各种事物，而不仅仅是 `int`、`float` 和 `char` 变量。你将有一个代表 `Barrel` 的变量，另一个代表 `Player` 的变量，等等，也就是说，一个代表你游戏中每个实体的变量。你将通过使用继承来重用代码；如果你不得不编写 `Cat` 和 `Dog` 的实现，你可以在基类 `Mammal` 中编写共同的功能。我们还讨论了封装以及如何编程对象以便它们保持自己的内部状态，这样做既容易又高效。
