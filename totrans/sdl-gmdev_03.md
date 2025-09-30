# 第三章. 与游戏对象协作

所有游戏都有对象，例如，玩家、敌人、**非玩家角色**（NPC）、陷阱、子弹和门。跟踪所有这些对象以及它们如何相互作用是一项庞大的任务，我们希望尽可能简化这项任务。如果我们没有坚实的实现，我们的游戏可能会变得难以控制且难以更新。那么我们如何使我们的任务更容易呢？我们可以从真正尝试利用面向对象编程（OOP）的强大功能开始。在本章中，我们将涵盖以下内容：

+   使用继承

+   实现多态

+   使用抽象基类

+   有效的继承设计

# 使用继承

我们将要探讨的第一个面向对象编程（OOP）的强大特性是继承。这个特性在开发可重用的框架时能给我们带来巨大的帮助。通过使用继承，我们可以在相似类之间共享通用功能，并从现有类型中创建子类型。我们不会深入探讨继承本身，而是开始思考如何将其应用到我们的框架中。

如前所述，所有游戏都有各种类型的对象。在大多数情况下，这些对象将拥有大量相同的数据和需要大量相同的基本功能。让我们看看一些这种常见功能性的例子：

+   几乎我们所有的对象都将被绘制到屏幕上，因此需要`draw`函数

+   如果我们的对象需要被绘制，它们将需要一个绘制位置，即 x 和 y 位置变量

+   我们不总是需要静态对象，所以我们需要一个`update`函数

+   对象将负责清理自己的事务；处理这个问题的函数将非常重要

这是我们第一个游戏对象类的一个很好的起点，所以让我们继续并创建它。向项目中添加一个新的类名为`GameObject`，然后我们可以开始：

```cpp
class GameObject
{
public:

  void draw() { std::cout << "draw game object"; }
  void update() { std::cout << "update game object"; }
  void clean() { std::cout << "clean game object"; }

protected:

  int m_x;
  int m_y;
};
```

### 注意

公共（public）、受保护（protected）和私有（private）关键字非常重要。公共函数和数据可以从任何地方访问。受保护状态仅允许从其派生的类访问。私有成员仅对该类可用，甚至其派生类也无法访问。

因此，我们有了第一个游戏对象类。现在让我们从它继承并创建一个名为`Player`的类：

```cpp
class Player : public GameObject // inherit from GameObject
{
public:

  void draw()
  {
    GameObject::draw();
    std::cout << "draw player";
  }
  void update()
  {
    std::cout << "update player";
    m_x = 10;
    m_y = 20;
  }
  void clean()
  {
    GameObject::clean();
    std::cout << "clean player";
  }
};
```

我们已经实现的能力是重用我们在`GameObject`中原本拥有的代码和数据，并将其应用到我们新的`Player`类中。正如你所看到的，派生类可以覆盖父类的功能：

```cpp
void update()
{
  std::cout << "update player";
  m_x = 10;
  m_y = 20;
}
```

或者它甚至可以使用父类的功能，同时在其之上拥有自己的附加功能：

```cpp
void draw()
{
  GameObject::draw();
  std::cout << "draw player";
}
```

在这里，我们调用`GameObject`中的`draw`函数，然后定义一些玩家特定的功能。

### 注意

`::`运算符被称为作用域解析运算符，它用于标识某些数据或函数的具体位置。

好的，到目前为止，我们的类没有做太多，所以让我们添加一些 SDL 功能。我们将在`GameObject`类中添加一些绘图代码，然后在`Player`类中重用它。首先，我们将更新`GameObject`头文件，添加一些新的值和函数，以便我们可以使用现有的 SDL 代码：

```cpp
class GameObject
{
public:

  void load(int x, int y, int width, int height, std::string 
  textureID);
  void draw(SDL_Renderer* pRenderer);
  void update();
  void clean();

protected:

  std::string m_textureID;

  int m_currentFrame;
  int m_currentRow;

  int m_x;
  int m_y;

  int m_width;
  int m_height;
};
```

现在我们有一些新的成员变量，它们将在新的`load`函数中设置。我们还在`draw`函数中传递了我们要使用的`SDL_Renderer`对象。让我们在一个实现文件中定义这些函数并创建`GameObject.cpp`：

首先定义我们的新`load`函数：

```cpp
void GameObject::load(int x, int y, int width, int height, std::string textureID)
{
  m_x = x;
  m_y = y;
  m_width = width;
  m_height = height;
  m_textureID = textureID;

  m_currentRow = 1;
  m_currentFrame = 1;
}
```

这里我们设置了在头文件中声明的所有值。现在我们可以创建我们的`draw`函数，它将使用这些值：

```cpp
void GameObject::draw(SDL_Renderer* pRenderer)
{
  TextureManager::Instance()->drawFrame(m_textureID, m_x, m_y, 
  m_width, m_height, m_currentRow, m_currentFrame, pRenderer);
}
```

我们使用`m_textureID`从`TextureManager`获取我们想要的纹理，并根据我们设置的值绘制它。最后，我们可以在`update`函数中添加一些内容，这些内容可以在`Player`类中重写：

```cpp
void GameObject::update()
{
  m_x += 1;
}
```

我们的`GameObject`类现在已经完成。我们现在可以修改`Player`头文件以反映我们的更改：

```cpp
#include "GameObject.h"

class Player : public GameObject
{
public:

  void load(int x, int y, int width, int height, std::string 
  textureID);
  void draw(SDL_Renderer* pRenderer);
  void update();
  void clean();
};
```

我们现在可以继续在实现文件中定义这些函数。创建`Player.cpp`，我们将遍历这些函数。首先，我们将从`load`函数开始：

```cpp
void Player::load(int x, int y, int width, int height, string textureID)
{
  GameObject::load(x, y, width, height, textureID);
}
```

这里我们可以使用我们的`GameObject::load`函数。同样也适用于我们的`draw`函数：

```cpp
void Player::draw(SDL_Renderer* pRenderer)
{
  GameObject::draw(pRenderer);
}
```

让我们用不同的方式重写`update`函数；让我们让这个对象动画化并朝相反方向移动：

```cpp
void Player::update()
{
  m_x -= 1;
}
```

现在我们已经准备好了；我们可以在`Game`头文件中创建这些对象：

```cpp
GameObject m_go;
Player m_player;
```

然后在`init`函数中加载它们：

```cpp
m_go.load(100, 100, 128, 82, "animate");
m_player.load(300, 300, 128, 82, "animate");
```

然后需要将它们添加到`render`和`update`函数中：

```cpp
void Game::render()
{

  SDL_RenderClear(m_pRenderer); // clear to the draw colour

  m_go.draw(m_pRenderer);
  m_player.draw(m_pRenderer);

  SDL_RenderPresent(m_pRenderer); // draw to the screen

}

void Game::update()
{
  m_go.update();
  m_player.update();
}
```

为了使程序正确运行，我们还需要添加一个东西。我们需要稍微限制帧率；如果不这样做，那么我们的对象会移动得太快。我们将在后面的章节中详细介绍这一点，但现在我们可以在主循环中添加一个延迟。所以，回到`main.cpp`，我们可以添加这一行：

```cpp
while(g_game->running())
{
  g_game->handleEvents();
  g_game->update();
  g_game->render();

  SDL_Delay(10); // add the delay
}
```

现在构建并运行以查看我们的两个独立对象：

![使用继承](img/6821OT_03_01.jpg)

我们的`Player`类编写起来非常简单，因为我们已经在`GameObject`类中编写了一些代码，以及所需的变量。然而，你可能已经注意到，我们在`Game`类中的很多地方都复制了代码。创建和添加新对象到游戏需要很多步骤。这并不理想，因为很容易遗漏一个步骤，而且当游戏对象超过两个或三个不同对象时，管理和维护会变得极其困难。

我们真正想要的是让`Game`类不需要关心不同类型；然后我们可以一次性遍历所有游戏对象，并为它们的每个函数分别使用循环。

# 实现多态

这引出了我们的下一个面向对象编程特性，多态。多态允许我们通过其父类或基类的指针来引用对象。一开始这可能看起来并不强大，但这将允许我们做到的是，我们的`Game`类只需要存储一个指向一种类型的指针列表，任何派生类型也可以添加到这个列表中。

让我们以`GameObject`和`Player`类为例，加上一个派生类`Enemy`。在我们的`Game`类中，我们有一个`GameObject*`数组：

```cpp
std::vector<GameObject*> m_gameObjects;
```

然后我们声明四个新对象，它们都是`GameObject*`：

```cpp
GameObject* m_player;
GameObject* m_enemy1;
GameObject* m_enemy2;
GameObject* m_enemy3;
```

在我们的`Game::init`函数中，我们可以创建对象的实例，使用它们的单独类型：

```cpp
m_player = new Player();
m_enemy1 = new Enemy();
m_enemy2 = new Enemy();
m_enemy3 = new Enemy();
```

现在它们可以被推入`GameObject*`数组中：

```cpp
m_gameObjects.push_back(m_player);
m_gameObjects.push_back(m_enemy1);
m_gameObjects.push_back(m_enemy2);
m_gameObjects.push_back(m_enemy3);
```

`Game::draw`函数现在可能看起来像这样：

```cpp
void Game::draw()
{
  for(std::vector<GameObject*>::size_type i = 0; i != 
  m_gameObjects.size(); i++) 
  {
    m_gameObjects[i]->draw(m_pRenderer);
  }
}
```

注意，我们正在遍历所有对象并调用`draw`函数。循环并不关心我们的某些对象实际上是`Player`或`Enemy`；它以相同的方式处理它们。我们通过它们基类的指针访问它们。因此，要添加新类型，它只需从`GameObject`派生即可，`Game`类可以处理它。

+   因此，让我们在我们的框架中真正实现这一点。首先，我们需要一个基类；我们将坚持使用`GameObject`。我们将不得不对这个类做一些修改，以便我们可以将其用作基类：

    ```cpp
    class GameObject
    {
    public:

      virtual void load(int x, int y, int width, int height, 
      std::string textureID);
      virtual void draw(SDL_Renderer* pRenderer);
      virtual void update();
      virtual void clean();

    protected:

      std::string m_textureID;

      int m_currentFrame;
      int m_currentRow;

      int m_x;
      int m_y;

      int m_width;
      int m_height;
    };
    ```

注意，我们现在已经用虚拟关键字前缀了我们的函数。虚拟关键字意味着当通过指针调用此函数时，它使用对象本身的类型定义，而不是其指针的类型：

```cpp
void Game::draw()
{
  for(std::vector<GameObject*>::size_type i = 0; i != 
  m_gameObjects.size(); i++) 
  {
    m_gameObjects[i]->draw(m_pRenderer);  
  }
}
```

换句话说，这个函数总是会调用`GameObject`中包含的`draw`函数，无论是`Player`还是`Enemy`。我们永远不会得到我们想要的覆盖行为。虚拟关键字将确保调用`Player`和`Enemy`的`draw`函数。

现在我们有一个基类，所以让我们在我们的`Game`类中实际尝试一下。我们首先在`Game`头文件中声明对象：

```cpp
GameObject* m_go;
GameObject* m_player;
```

现在声明与我们的`GameObject*`数组一起：

```cpp
std::vector<GameObject*> m_gameObjects;
```

现在在`init`函数中创建和加载对象，然后将它们推入数组中：

```cpp
m_go = new GameObject();
m_player = new Player();

m_go->load(100, 100, 128, 82, "animate");
m_player->load(300, 300, 128, 82, "animate");

m_gameObjects.push_back(m_go);
m_gameObjects.push_back(m_player);
```

到目前为止，一切顺利；我们现在可以创建一个循环来绘制我们的对象，另一个循环来更新它们。现在让我们看看`render`和`update`函数：

```cpp
void Game::render()
{

  SDL_RenderClear(m_pRenderer); // clear to the draw colour

  // loop through our objects and draw them
  for(std::vector<GameObject*>::size_type i = 0; i != 
  m_gameObjects.size(); i++)
  {
    m_gameObjects[i]->draw(m_pRenderer);
  }

  SDL_RenderPresent(m_pRenderer); // draw to the screen

}

void Game::update()
{
  // loop through and update our objects
  for(std::vector<GameObject*>::size_type i = 0; i != 
  m_gameObjects.size(); i++)
  {
    m_gameObjects[i]->update();
  }
}
```

如您所见，这要整洁得多，也更容易管理。让我们再从`GameObject`派生一个类，以便我们更深入地理解这个概念。创建一个名为`Enemy`的新类：

```cpp
class Enemy : public GameObject
{
public:

  void load(int x, int y, int width, int height, std::string 
  textureID);
  void draw(SDL_Renderer* pRenderer);
  void update();
  void clean();
};
```

我们将定义这个类的函数与`Player`相同，只有`update`函数是一个例外：

```cpp
void Enemy::update()
{
  m_y += 1;
  m_x += 1;
  m_currentFrame = int(((SDL_GetTicks() / 100) % 6));
}
```

现在让我们将其添加到游戏中。首先，我们这样声明：

```cpp
GameObject* m_enemy;
```

然后创建、加载并将它们添加到数组中：

```cpp
m_enemy = new Enemy();
m_enemy->load(0, 0, 128, 82, "animate");
m_gameObjects.push_back(m_enemy);
```

我们刚刚添加了一个新类型，而且非常快速简单。运行游戏，看看我们的三个对象，每个对象都有它们自己的不同行为。

![实现多态](img/6821OT_03_02.jpg)

我们在这里已经涵盖了大量的内容，并有一个处理游戏对象的非常不错的系统，但我们仍然有一个问题。没有任何东西阻止我们派生一个没有我们在这里使用的`update`或`draw`函数的类，甚至可以声明一个不同的函数并将`update`代码放在那里。作为开发者，我们不太可能犯这样的错误，但其他人使用框架时可能会。我们希望的是能够强制我们的派生类实现我们决定的一个函数，创建一个我们希望所有游戏对象都遵循的蓝图。我们可以通过使用抽象基类来实现这一点。

# 使用抽象基类

如果我们要正确实现我们的设计，那么我们必须确保所有派生类都有我们希望通过基类指针访问的每个函数的声明和定义。我们可以通过将`GameObject`设为抽象基类来确保这一点。抽象基类本身不能被初始化；它的目的是规定派生类的设计。这使我们能够重用，因为我们知道从`GameObject`派生的任何对象都将立即在游戏的整体方案中工作。

抽象基类是一个包含至少一个纯虚函数的类。纯虚函数是一个没有定义且必须在任何派生类中实现的函数。我们可以通过在函数后添加`=0`来使其成为纯虚函数。

# 我们是否应该总是使用继承？

继承和多态都非常有用，并且真正展示了面向对象编程的强大之处。然而，在某些情况下，继承可能会造成比解决的问题更多的问题，因此，在决定是否使用它时，我们应该牢记一些经验法则。

## 是否可以用更简单的解决方案达到同样的效果？

假设我们想要创建一个更强大的`Enemy`对象；它将具有与普通`Enemy`对象相同的行为，但拥有更多的生命值。一个可能的解决方案是从`Enemy`派生一个新的类`PowerEnemy`并给它双倍的生命值。在这个解决方案中，新类看起来会非常稀疏；它将使用`Enemy`的功能，但有一个不同的值。一个更简单的解决方案是提供一种方法来设置`Enemy`类的生命值，无论是通过访问器还是构造函数。在这种情况下，继承根本不是必需的。

## 派生类应该模拟“是”关系

在派生一个类时，让它模拟“是一个”关系是一个好主意。这意味着派生类也应该与父类具有相同的类型。例如，从 `Player` 类派生一个 `Player2` 类是符合模型的，因为 `Player2` “是一个” `Player`。但是，假设我们有一个 `Jetpack` 类，并且从该类派生 `Player` 类以使其能够访问 `Jetpack` 类的所有功能。这不会模拟“是一个”关系，因为 `Player` 类不是 `Jetpack` 类。更合理的是说 `Player` 类有一个 `Jetpack` 类，因此 `Player` 类应该有一个类型为 `Jetpack` 的成员变量，没有继承；这被称为包含。

## 可能的性能惩罚

在 PC 和 Mac 等平台上，使用继承和虚函数的性能惩罚是可以忽略不计的。然而，如果你正在为功能较弱的设备，如手持式游戏机、手机或嵌入式系统开发，这将是你需要考虑的事情。如果你的核心循环每秒多次调用虚函数，性能惩罚可能会累积。

# 将所有内容组合在一起

我们现在可以将所有这些知识结合起来，尽可能地将它们应用到我们的框架中，同时考虑到可重用性。我们有很多工作要做，所以让我们从我们的抽象基类 `GameObject` 开始。我们将移除所有与 SDL 相关的内容，以便在需要时可以在其他 SDL 项目中重用这个类。以下是我们的简化版 `GameObject` 抽象基类：

```cpp
class GameObject
{
public:

  virtual void draw()=0;
  virtual void update()=0;
  virtual void clean()=0;

protected:

  GameObject(const LoaderParams* pParams) {}
  virtual ~GameObject() {}
};
```

已经创建了纯虚函数，迫使任何派生类也必须声明和实现它们。现在也没有了 `load` 函数；这样做的原因是我们不希望为每个新项目都创建一个新的 `load` 函数。我们可以相当肯定，在加载不同游戏的对象时，我们将需要不同的值。我们将采取的方法是创建一个新的类 `LoaderParams` 并将其传递到对象的构造函数中。

`LoaderParams` 是一个简单的类，它接受构造函数中的值并将它们设置为成员变量，然后可以访问这些变量来设置对象的初始值。虽然这看起来我们只是将参数从 `load` 函数移动到其他地方，但创建一个新的 `LoaderParams` 类比追踪和修改所有对象的 `load` 函数要容易得多。

因此，这是我们的 `LoaderParams` 类：

```cpp
class LoaderParams
{
public:

  LoaderParams(int x, int y, int width, int height, std::string 
  textureID) : m_x(x), m_y(y), m_width(width), m_height(height), 
  m_textureID(textureID)
  {

  }

  int getX() const { return m_x; }
  int getY() const { return m_y; }
  int getWidth() const { return m_width; }
  int getHeight() const { return m_height; }
  std::string getTextureID() const { return m_textureID; }

private:

  int m_x;
  int m_y;

  int m_width;
  int m_height;

  std::string m_textureID;
};
```

这个类在创建对象时持有我们需要的任何值，其方式与我们的 `load` 函数曾经做的一样。

我们还从 `draw` 函数中移除了 `SDL_Renderer` 参数。我们将使 `Game` 类成为一个单例，例如 `TextureManager`。因此，我们可以将以下内容添加到我们的 `Game` 类中：

```cpp
// create the public instance function
static Game* Instance()
{
  if(s_pInstance == 0)
  {
    s_pInstance = new Game();
    return s_pInstance;
  }

  return s_pInstance;
}
// make the constructor private
private:

  Game();
// create the s_pInstance member variable
  static Game* s_pInstance;

// create the typedef
  typedef Game TheGame;
```

在 `Game.cpp` 文件中，我们必须定义我们的静态实例：

```cpp
Game* Game::s_pInstance = 0;
```

让我们在头文件中创建一个函数，该函数将返回我们的 `SDL_Renderer` 对象：

```cpp
SDL_Renderer* getRenderer() const { return m_pRenderer; }
```

现在，由于`Game`是一个单例，我们将在`main.cpp`文件中以不同的方式使用它：

```cpp
int main(int argc, char* argv[])
{
  std::cout << "game init attempt...\n";
  if(TheGame::Instance()->init("Chapter 1", 100, 100, 640, 480, 
  false))
  {
    std::cout << "game init success!\n";
    while(TheGame::Instance()->running())
    {
      TheGame::Instance()->handleEvents();
      TheGame::Instance()->update();
      TheGame::Instance()->render();

      SDL_Delay(10);
    }
  }
  else
  {
    std::cout << "game init failure - " << SDL_GetError() << "\n";
    return -1;
  }

  std::cout << "game closing...\n";
  TheGame::Instance()->clean();

  return 0;
}
```

现在我们想要从`Game`访问`m_pRenderer`值时，我们可以使用`getRenderer`函数。由于`GameObject`基本上是空的，我们如何实现我们最初所期望的代码共享？我们将从一个新的通用类`GameObject`派生出一个新类，并将其命名为`SDLGameObject`：

```cpp
class SDLGameObject : public GameObject
{
public:

  SDLGameObject(const LoaderParams* pParams);

  virtual void draw();
  virtual void update();
  virtual void clean();

protected:

  int m_x;
  int m_y;

  int m_width;
  int m_height;

  int m_currentRow;
  int m_currentFrame;

  std::string m_textureID;
};
```

使用这个类，我们可以创建可重用的 SDL 代码。首先，我们可以使用我们新的`LoaderParams`类来设置我们的成员变量：

```cpp
SDLGameObject::SDLGameObject(const LoaderParams* pParams) : 
GameObject(pParams)
{
  m_x = pParams->getX();
  m_y = pParams->getY();
  m_width = pParams->getWidth();
  m_height = pParams->getHeight();
  m_textureID = pParams->getTextureID();

  m_currentRow = 1;
  m_currentFrame = 1;
}
```

我们还可以使用之前相同的`draw`函数，利用我们的单例`Game`类来获取我们想要的渲染器：

```cpp
void SDLGameObject::draw()
{
  TextureManager::Instance()->drawFrame(m_textureID, m_x, m_y, 
  m_width, m_height, m_currentRow, m_currentFrame, 
  TheGame::Instance()->getRenderer());
}
```

`Player`和`Enemy`现在可以继承自`SDLGameObject`：

```cpp
class Player : public SDLGameObject
{
public:

  Player(const LoaderParams* pParams);

  virtual void draw();
  virtual void update();
  virtual void clean();
};
// Enemy class
class Enemy : public SDLGameObject
{
public:

  Enemy(const LoaderParams* pParams);

  virtual void draw();
  virtual void update();
  virtual void clean();
};
```

`Player`类的定义可以如下所示（`Enemy`类非常相似）：

```cpp
Player::Player(const LoaderParams* pParams) : 
SDLGameObject(pParams)
{

}

void Player::draw()
{
  SDLGameObject::draw(); // we now use SDLGameObject
}

void Player::update()
{
  m_x -= 1;
  m_currentFrame = int(((SDL_GetTicks() / 100) % 6));
}

void Player::clean()
{
}
```

现在一切准备就绪，我们可以继续创建`Game`类中的对象，并观察一切的实际运行情况。这次我们不会将对象添加到头文件中；我们将使用一个快捷方式，在`init`函数中一行内构建我们的对象：

```cpp
m_gameObjects.push_back(new Player(new LoaderParams(100, 100, 128, 82, "animate")));

m_gameObjects.push_back(new Enemy(new LoaderParams(300, 300, 128, 82, "animate")));
```

构建项目。我们现在已经准备好了一切，可以轻松地重用我们的`Game`和`GameObject`类。

# 摘要

在本章中，我们涵盖了大量的复杂主题，这些概念和想法需要一些时间才能深入人心。我们介绍了如何轻松创建类，而无需重写大量类似的功能，以及继承的使用方法，它使我们能够在类似类之间共享代码。我们探讨了多态性以及它如何使对象管理变得更加简洁和可重用，而抽象基类则通过创建我们希望所有对象遵循的蓝图，将我们的继承知识提升到一个新的层次。最后，我们将所有新的知识融入到我们的框架中。
