# 第六章。数据驱动设计

在上一章中，通过添加创建和处理游戏状态的能力，我们的框架已经开始成形。在本章中，我们将探讨一种新的创建状态和对象的方法，即通过移除在编译时硬编码对象创建的需求。为此，我们将解析一个外部文件，在我们的例子中是一个 XML 文件，该文件列出了我们状态所需的所有对象。这将使我们的状态变得通用，因为它们可以通过加载不同的 XML 文件而完全不同。以 `PlayState` 为例，在创建新关卡时，我们需要创建一个新的状态，包含不同的对象，并设置我们想要在该关卡中使用的对象。如果我们能够从外部文件加载对象，我们就可以重用相同的 `PlayState`，并根据我们想要的当前关卡简单地加载正确的文件。保持类通用并加载外部数据以确定其状态被称为 **数据驱动设计**。

在本章中，我们将介绍：

+   使用 **TinyXML** 库加载 XML 文件

+   创建 **分布式工厂**

+   使用工厂和 XML 文件动态加载对象

+   从 XML 文件解析状态

+   将一切整合到框架中

# 加载 XML 文件

我选择使用 XML 文件，因为它们非常容易解析。我们不会编写自己的 XML 解析器，而是将使用一个名为 TinyXML 的开源库。TinyXML 是由 *Lee Thomason* 编写的，并且可以在 [`sourceforge.net/projects/tinyxml/`](http://sourceforge.net/projects/tinyxml/) 下以 zlib 许可证获得。

下载后，我们唯一需要做的设置就是将几个文件包含到我们的项目中：

+   `tinyxmlerror.cpp`

+   `tinyxmlparser.cpp`

+   `tinystr.cpp`

+   `tinystr.h`

+   `tinyxml.cpp`

+   `tinyxml.h`

在 `tinyxml.h` 的顶部添加以下代码行：

```cpp
#define TIXML_USE_STL
```

通过这样做，我们确保正在使用 TinyXML 函数的 STL 版本。现在我们可以简要地介绍一下 XML 文件的构成。实际上它相当简单，我们只提供一个简要概述，以帮助您了解我们将如何使用它。

## 基本 XML 结构

这里是一个基本的 XML 文件：

```cpp
<?xml version="1.0" ?>
<ROOT>
    <ELEMENT>
    </ELEMENT>
</ROOT>
```

文件的第一行定义了 XML 文件的格式。第二行是我们的 `Root` 元素；其他所有内容都是这个元素的子元素。第三行是根元素的第一个子元素。现在让我们看看一个稍微复杂一点的 XML 文件：

```cpp
<?xml version="1.0" ?>
<ROOT>
    <ELEMENTS>
        <ELEMENT>Hello,</ELEMENT>
        <ELEMENT> World!</ELEMENT>
    </ELEMENTS>
</ROOT>
```

如您所见，我们现在已经向第一个子元素添加了子元素。您可以嵌套任意多的子元素。但是如果没有良好的结构，您的 XML 文件可能非常难以阅读。如果我们解析上述文件，我们将采取以下步骤：

1.  加载 XML 文件。

1.  获取根元素，`<ROOT>`。

1.  获取根元素的第一个子元素，`<ELEMENTS>`。

1.  对于 `<ELEMENTS>` 的每个子元素 `<ELEMENT>`，获取其内容。

1.  关闭文件。

另一个有用的 XML 功能是使用属性。以下是一个示例：

```cpp
<ROOT>
    <ELEMENTS>
        <ELEMENT text="Hello,"/>
        <ELEMENT text=" World!"/>
    </ELEMENTS>
</ROOT>
```

我们现在已经将想要存储的文本存储在一个名为`text`的属性中。当这个文件被解析时，我们会获取每个元素的`text`属性并将其存储，而不是存储`<ELEMENT></ELEMENT>`标签之间的内容。这对我们来说特别有用，因为我们可以使用属性来存储我们对象的大量不同值。所以让我们看看一些更接近我们将在游戏中使用的内容：

```cpp
<?xml version="1.0" ?>
<STATES>

<!--The Menu State-->
<MENU>
<TEXTURES>
  <texture filename="button.png" ID="playbutton"/>
  <texture filename="exit.png" ID="exitbutton"/>
</TEXTURES>

<OBJECTS>
  <object type="MenuButton" x="100" y="100" width="400" 
  height="100" textureID="playbutton"/>
  <object type="MenuButton" x="100" y="300" width="400" 
  height="100" textureID="exitbutton"/>
</OBJECTS>
</MENU>

<!--The Play State-->
<PLAY>
</PLAY>

<!-- The Game Over State -->
<GAMEOVER>
</GAMEOVER>
</STATES>
```

这稍微复杂一些。我们为每个状态定义一个元素，在这个元素中，我们有具有各种属性的物体和纹理。这些属性可以加载到状态中。

通过对 XML 的这些知识，你可以轻松地创建自己的文件结构，如果本书中涵盖的内容不符合你的需求。

# 实现对象工厂

我们现在拥有一些 XML 知识，但在我们继续前进之前，我们将看看对象工厂。对象工厂是一个负责创建我们对象的类。本质上，我们告诉工厂我们想要它创建的对象，然后它就会创建该对象的新实例并返回它。我们可以从查看一个基本的实现开始：

```cpp
GameObject* GameObjectFactory::createGameObject(ID id)
{
  switch(id)
  {
    case "PLAYER":
      return new Player();
    break;

    case "ENEMY":
      return new Enemy();
    break;

    // lots more object types 
  }
}
```

这个函数非常简单。我们传入一个对象的 ID，工厂使用一个大的 switch 语句来查找并返回正确的对象。这不是一个糟糕的解决方案，但也不是一个特别好的解决方案，因为工厂需要知道它需要创建的每个类型，并且维护许多不同对象的 switch 语句将会非常繁琐。就像我们在第三章中介绍遍历游戏对象时一样，*与游戏对象一起工作*，我们希望这个工厂不关心我们要求的是什么类型。它不需要知道我们想要它创建的所有具体类型。幸运的是，这确实是我们能够实现的事情。

## 使用分布式工厂

通过使用分布式工厂，我们可以创建一个通用的对象工厂，它可以创建我们的任何类型。分布式工厂允许我们动态地维护我们想要工厂创建的对象类型，而不是将它们硬编码到一个函数中（就像前面的简单示例中那样）。我们将采取的方法是让工厂包含一个`std::map`，它将一个字符串（我们对象的类型）映射到一个名为`Creator`的小类。`Creator`的唯一目的是创建特定的对象。我们将使用一个函数将新类型注册到工厂中，该函数接受一个字符串（ID）和一个`Creator`类，并将它们添加到工厂的映射中。我们将从所有`Creator`类型的基类开始。创建`GameObjectFactory.h`并在文件顶部声明这个类。

```cpp
#include <string>
#include <map>
#include "GameObject.h"

class BaseCreator
{
  public:

  virtual GameObject* createGameObject() const = 0;
  virtual ~BaseCreator() {}
};
```

我们现在可以继续创建其余的工厂，然后逐个分析它。

```cpp
class GameObjectFactory
{
  public:

  bool registerType(std::string typeID, BaseCreator* pCreator)
  {
    std::map<std::string, BaseCreator*>::iterator it = 
    m_creators.find(typeID);

    // if the type is already registered, do nothing
    if(it != m_creators.end())
    {
      delete pCreator;
      return false;
    }

    m_creators[typeID] = pCreator;

    return true;
  }

  GameObject* create(std::string typeID)
  {
    std::map<std::string, BaseCreator*>::iterator it = 
    m_creators.find(typeID);

    if(it == m_creators.end())
    {
      std::cout << "could not find type: " << typeID << "\n";
      return NULL;
    }

    BaseCreator* pCreator = (*it).second;
    return pCreator->createGameObject();
  }

  private:

  std::map<std::string, BaseCreator*> m_creators;

};
```

这是一个相当小的类，但实际上非常强大。我们将分别介绍每个部分，从`std::map m_creators`开始。

```cpp
std::map<std::string, BaseCreator*> m_creators;
```

这张地图包含了我们工厂的重要元素，类的功能本质上要么添加要么从这张地图中移除。当我们查看 `registerType` 函数时，这一点变得很明显：

```cpp
bool registerType(std::string typeID, BaseCreator* pCreator)
```

这个函数接受我们想要与对象类型关联的 ID（作为字符串），以及该类的创建者对象。然后函数尝试使用 `std::mapfind` 函数查找类型：

```cpp
std::map<std::string, BaseCreator*>::iterator it = m_creators.find(typeID);
```

如果找到类型，则它已经注册。然后函数删除传入的指针并返回 `false`：

```cpp
if(it != m_creators.end())
{
  delete pCreator;
  return false;
}
```

如果类型尚未注册，则可以将其分配给地图，然后返回 `true`：

```cpp
m_creators[typeID] = pCreator;
return true;
}
```

如您所见，`registerType` 函数实际上非常简单；它只是将类型添加到地图的一种方式。`create` 函数非常相似：

```cpp
GameObject* create(std::string typeID)
{
  std::map<std::string, BaseCreator*>::iterator it = 
  m_creators.find(typeID);

  if(it == m_creators.end())
  {
    std::cout << "could not find type: " << typeID << "\n";
    return 0;
  }

  BaseCreator* pCreator = (*it).second;
  return pCreator->createGameObject();
}
```

这个函数以与 `registerType` 相同的方式查找类型，但这次它检查类型是否未找到（而不是找到）。如果类型未找到，我们返回 `0`，如果类型找到，则我们使用该类型的 `Creator` 对象返回一个新的实例，作为 `GameObject` 指针。

值得注意的是，`GameObjectFactory` 类可能应该是一个单例。我们不会介绍如何将其变为单例，因为这在之前的章节中已经介绍过了。尝试自己实现它或查看源代码下载中的实现方式。

# 将工厂集成到框架中

现在我们已经设置了工厂，我们可以开始修改我们的 `GameObject` 类以使用它。我们的第一步是确保我们为每个对象都有一个 `Creator` 类。这里有一个 `Player` 的示例：

```cpp
class PlayerCreator : public BaseCreator
{
  GameObject* createGameObject() const
  {
    return new Player();
  }
};
```

这可以添加到 `Player.h` 文件的底部。我们想要工厂创建的任何对象都必须有自己的 `Creator` 实现。我们必须做的另一个补充是将 `LoaderParams` 从构造函数移动到它们自己的函数 `load` 中。这阻止了我们需要将 `LoaderParams` 对象传递给工厂本身。我们将 `load` 函数放入 `GameObject` 基类中，因为我们希望每个对象都有一个。

```cpp
class GameObject
{
  public:

  virtual void draw()=0;
  virtual void update()=0;
  virtual void clean()=0;

  // new load function 
  virtual void load(const LoaderParams* pParams)=0;

  protected:

  GameObject() {}
  virtual ~GameObject() {}
};
```

我们的所有派生类现在都需要实现这个 `load` 函数。`SDLGameObject` 类现在看起来像这样：

```cpp
SDLGameObject::SDLGameObject() : GameObject()
{
}

voidSDLGameObject::load(const LoaderParams *pParams)
{
  m_position = Vector2D(pParams->getX(),pParams->getY());
  m_velocity = Vector2D(0,0);
  m_acceleration = Vector2D(0,0);
  m_width = pParams->getWidth();
  m_height = pParams->getHeight();
  m_textureID = pParams->getTextureID();
  m_currentRow = 1;
  m_currentFrame = 1;
  m_numFrames = pParams->getNumFrames();
}
```

我们从 `SDLGameObject` 派生的对象也可以使用这个 `load` 函数；例如，这里有一个 `Player::load` 函数：

```cpp
Player::Player() : SDLGameObject()
{

}

void Player::load(const LoaderParams *pParams)
{
  SDLGameObject::load(pParams);
}
```

这可能看起来有点没有意义，但实际上它节省了我们不需要在各个地方传递 `LoaderParams`。如果没有它，我们就需要通过工厂的 `create` 函数传递 `LoaderParams`，然后它再传递给 `Creator` 对象。我们通过有一个专门处理解析我们加载值的函数来消除了这种需求。一旦我们开始从文件中解析我们的状态，这将会更有意义。

我们还有一个需要纠正的问题；我们有两个类在它们的构造函数中具有额外的参数（`MenuButton` 和 `AnimatedGraphic`）。这两个类都接受一个额外的参数以及 `LoaderParams`。为了解决这个问题，我们将这些值添加到 `LoaderParams` 并赋予它们默认值。

```cpp
LoaderParams(int x, int y, int width, int height, std::string textureID, int numFrames, int callbackID = 0, int animSpeed = 0) :
m_x(x),
m_y(y),
m_width(width),
m_height(height),
m_textureID(textureID),
m_numFrames(numFrames),
m_callbackID(callbackID),
m_animSpeed(animSpeed)
{

}
```

换句话说，如果没有传递参数，则将使用默认值（两种情况下都是 0）。与 `MenuButton` 传递函数指针的方式不同，我们正在使用 `callbackID` 来决定在状态内使用哪个回调函数。现在我们可以开始使用我们的工厂并从 XML 文件解析状态。

# 从 XML 文件解析状态

我们将要解析的文件如下（源代码下载中的 `test.xml`）：

```cpp
<?xml version="1.0" ?>
<STATES>
<MENU>
<TEXTURES>
  <texture filename="assets/button.png" ID="playbutton"/>
  <texture filename="assets/exit.png" ID="exitbutton"/>
</TEXTURES>

<OBJECTS>
  <object type="MenuButton" x="100" y="100" width="400" 
  height="100" textureID="playbutton" numFrames="0" 
  callbackID="1"/>
  <object type="MenuButton" x="100" y="300" width="400" 
  height="100" textureID="exitbutton" numFrames="0" 
  callbackID="2"/>
</OBJECTS>
</MENU>
<PLAY>
</PLAY>

<GAMEOVER>
</GAMEOVER>
</STATES>
```

我们将创建一个新的类来为我们解析状态，称为 `StateParser`。`StateParser` 类没有数据成员，它应该在状态的 `onEnter` 函数中使用一次，然后当它超出作用域时被丢弃。创建一个 `StateParser.h` 文件并添加以下代码：

```cpp
#include <iostream>
#include <vector>
#include "tinyxml.h"

class GameObject;

class StateParser
{
  public:

  bool parseState(const char* stateFile, std::string stateID, 
  std::vector<GameObject*> *pObjects);

  private:

  void parseObjects(TiXmlElement* pStateRoot, 
  std::vector<GameObject*> *pObjects);
  void parseTextures(TiXmlElement* pStateRoot, 
  std::vector<std::string> *pTextureIDs);

};
```

我们这里有三个函数，一个公共的，两个私有的。`parseState` 函数接受一个 XML 文件的文件名作为参数，以及当前的 `stateID` 值和一个指向 `std::vector` 的 `GameObject*` 指针，该指针对应于该状态。`StateParser.cpp` 文件将定义此函数：

```cpp
bool StateParser::parseState(const char *stateFile, string 
stateID, vector<GameObject *> *pObjects, std::vector<std::string> 
*pTextureIDs)
{
  // create the XML document
  TiXmlDocument xmlDoc;

  // load the state file
  if(!xmlDoc.LoadFile(stateFile))
  {
    cerr << xmlDoc.ErrorDesc() << "\n";
    return false;
  }

  // get the root element
  TiXmlElement* pRoot = xmlDoc.RootElement();

  // pre declare the states root node
  TiXmlElement* pStateRoot = 0;
  // get this states root node and assign it to pStateRoot
  for(TiXmlElement* e = pRoot->FirstChildElement(); e != NULL; e = 
  e->NextSiblingElement())
  {
    if(e->Value() == stateID)
    {
      pStateRoot = e;
    }
  }

  // pre declare the texture root
  TiXmlElement* pTextureRoot = 0;

  // get the root of the texture elements
  for(TiXmlElement* e = pStateRoot->FirstChildElement(); e != 
  NULL; e = e->NextSiblingElement())
  {
    if(e->Value() == string("TEXTURES"))
    {
      pTextureRoot = e;
    }
  }

  // now parse the textures
  parseTextures(pTextureRoot, pTextureIDs);

  // pre declare the object root node
  TiXmlElement* pObjectRoot = 0;

  // get the root node and assign it to pObjectRoot
  for(TiXmlElement* e = pStateRoot->FirstChildElement(); e != 
  NULL; e = e->NextSiblingElement())
  {
    if(e->Value() == string("OBJECTS"))
    {
      pObjectRoot = e;
    }
  }

  // now parse the objects
  parseObjects(pObjectRoot, pObjects);

  return true;
}
```

这个函数中有很多代码，所以值得深入探讨。我们将注意 XML 文件中相应的部分，以及我们使用的代码，以获取它。函数的第一部分尝试加载传递给函数的 XML 文件：

```cpp
// create the XML document
TiXmlDocument xmlDoc;

// load the state file
if(!xmlDoc.LoadFile(stateFile))
{
  cerr << xmlDoc.ErrorDesc() << "\n";
  return false;
}
```

如果 XML 加载失败，它会显示一个错误来告诉你发生了什么。接下来，我们必须获取 XML 文件的根节点：

```cpp
// get the root element
TiXmlElement* pRoot = xmlDoc.RootElement(); // <STATES>
```

文件中的其余节点都是这个根节点的子节点。现在我们必须获取我们正在解析的状态的根节点；比如说，我们正在寻找 `MENU`：

```cpp
// declare the states root node
TiXmlElement* pStateRoot = 0;
// get this states root node and assign it to pStateRoot
for(TiXmlElement* e = pRoot->FirstChildElement(); e != NULL; e = e->NextSiblingElement())
{
  if(e->Value() == stateID)
  {
    pStateRoot = e;
  }
}
```

这段代码遍历根节点的每个直接子节点，并检查其名称是否与 `stateID` 相同。一旦找到正确的节点，它就将其分配给 `pStateRoot`。现在我们有了我们想要解析的状态的根节点。

```cpp
<MENU> // the states root node
```

现在我们有了状态根节点的指针，我们可以开始从中获取值。首先，我们想要从文件中加载纹理，所以我们使用之前找到的 `pStateRoot` 对象的子节点查找 `<TEXTURE>` 节点：

```cpp
// pre declare the texture root
TiXmlElement* pTextureRoot = 0;

// get the root of the texture elements
for(TiXmlElement* e = pStateRoot->FirstChildElement(); e != NULL;
e = e->NextSiblingElement())
{
  if(e->Value() == string("TEXTURES"))
  {
    pTextureRoot = e;
  }
}
```

一旦找到 `<TEXTURE>` 节点，我们可以将其传递给私有的 `parseTextures` 函数（我们稍后会介绍它）。

```cpp
parseTextures(pTextureRoot, std::vector<std::string> *pTextureIDs);
```

然后该函数继续搜索 `<OBJECT>` 节点，一旦找到，它就将其传递给私有的 `parseObjects` 函数。我们还传递了 `pObjects` 参数：

```cpp
  // pre declare the object root node
  TiXmlElement* pObjectRoot = 0;

  // get the root node and assign it to pObjectRoot
  for(TiXmlElement* e = pStateRoot->FirstChildElement(); e != NULL; e = e->NextSiblingElement())
  {
    if(e->Value() == string("OBJECTS"))
    {
      pObjectRoot = e;
    }
  }
  parseObjects(pObjectRoot, pObjects);
  return true;
}
```

到目前为止，我们的状态已经解析完毕。现在我们可以介绍两个私有函数，从 `parseTextures` 开始。

```cpp
void StateParser::parseTextures(TiXmlElement* pStateRoot, std::vector<std::string> *pTextureIDs)
{
  for(TiXmlElement* e = pStateRoot->FirstChildElement(); e != 
  NULL; e = e->NextSiblingElement())
  {
    string filenameAttribute = e->Attribute("filename");
    string idAttribute = e->Attribute("ID");
    pTextureIDs->push_back(idAttribute); // push into list

    TheTextureManager::Instance()->load(filenameAttribute, 
    idAttribute, TheGame::Instance()->getRenderer());
  }
}
```

此函数从 XML 此部分的每个纹理值中获取 `filename` 和 `ID` 属性：

```cpp
<TEXTURES>
  <texture filename="button.png" ID="playbutton"/>
  <texture filename="exit.png" ID="exitbutton"/>
</TEXTURES>
```

然后将它们添加到 `TextureManager`。

```cpp
TheTextureManager::Instance()->load(filenameAttribute, idAttribute, TheGame::Instance()->getRenderer());
```

`parseObjects`函数相当复杂。它使用我们的`GameObjectFactory`函数创建对象，并从 XML 文件的这部分读取：

```cpp
<OBJECTS>
  <object type="MenuButton" x="100" y="100" width="400" 
  height="100" textureID="playbutton" numFrames="0" 
  callbackID="1"/>
  <object type="MenuButton" x="100" y="300" width="400" 
  height="100" textureID="exitbutton" numFrames="0" 
  callbackID="2"/>
</OBJECTS>
```

`parseObjects`函数的定义如下：

```cpp
void StateParser::parseObjects(TiXmlElement *pStateRoot, 
std::vector<GameObject *> *pObjects)
{
  for(TiXmlElement* e = pStateRoot->FirstChildElement(); e != 
  NULL; e = e->NextSiblingElement())
  {
    int x, y, width, height, numFrames, callbackID, animSpeed;
    string textureID;

    e->Attribute("x", &x);
    e->Attribute("y", &y);
    e->Attribute("width",&width);
    e->Attribute("height", &height);
    e->Attribute("numFrames", &numFrames);
    e->Attribute("callbackID", &callbackID);
    e->Attribute("animSpeed", &animSpeed);

    textureID = e->Attribute("textureID");

    GameObject* pGameObject = TheGameObjectFactory::Instance()
    ->create(e->Attribute("type"));
    pGameObject->load(new LoaderParams
    (x,y,width,height,textureID,numFrames,callbackID, animSpeed));
    pObjects->push_back(pGameObject);
  }
}
```

首先，我们从当前节点获取所需的任何值。由于 XML 文件是纯文本，我们无法直接从文件中获取整数或浮点数。TinyXML 有函数可以让你传入想要设置的值和属性名称。例如：

```cpp
e->Attribute("x", &x);
```

这将变量`x`设置为属性`"x"`中包含的值。接下来是使用工厂创建一个`GameObject` ***** 类：

```cpp
GameObject* pGameObject = TheGameObjectFactory::Instance()->create(e->Attribute("type"));
```

我们传递`type`属性的值，并使用它从工厂创建正确的对象。之后，我们必须使用`GameObject`的`load`函数来设置从 XML 文件加载的值。

```cpp
pGameObject->load(new LoaderParams(x,y,width,height,textureID,numFrames,callbackID));
```

最后，我们将`pGameObject`推入`pObjects`数组，这实际上是指向当前状态对象向量的指针。

```cpp
pObjects->push_back(pGameObject);
```

# 从 XML 文件加载菜单状态

我们现在已经有了大部分状态加载代码，并可以在`MenuState`类中使用它。首先，我们必须做一些前期工作，并设置一种新的方法来分配回调到我们的`MenuButton`对象，因为这不是我们可以从 XML 文件传递进来的东西。我们将采取的方法是给任何想要使用回调的对象在 XML 文件中赋予一个名为`callbackID`的属性。其他对象不需要这个值，`LoaderParams`将使用默认值`0`。`MenuButton`类将使用这个值，并从它的`LoaderParams`中拉取它，如下所示：

```cpp
void MenuButton::load(const LoaderParams *pParams)
{
  SDLGameObject::load(pParams);
  m_callbackID = pParams->getCallbackID();
  m_currentFrame = MOUSE_OUT;
}
```

`MenuButton`类还需要两个其他函数，一个用于设置回调函数，另一个用于返回其回调 ID：

```cpp
void setCallback(void(*callback)()) { m_callback = callback;}
int getCallbackID() { return m_callbackID; }
```

接下来，我们必须创建一个设置回调函数的函数。任何使用具有回调的对象的状态都需要这个函数的实现。最可能具有回调的状态是菜单状态，因此我们将我们的`MenuState`类重命名为`MainMenuState`，并将`MenuState`作为一个从`GameState`扩展的抽象类。该类将声明一个为任何需要它的项目设置回调的函数，并且它还将有一个`Callback`对象的向量作为成员；这将在每个状态的`setCallbacks`函数中使用。

```cpp
class MenuState : public GameState
{
  protected:

  typedef void(*Callback)();
  virtual void setCallbacks(const std::vector<Callback>& callbacks) 
  = 0;

  std::vector<Callback> m_callbacks;
};
```

`MainMenuState`类（之前称为`MenuState`）现在将从这个`MenuState`类派生。

```cpp
#include "MenuState.h"
#include "GameObject.h"

class MainMenuState : public MenuState
{
  public:

  virtual void update();
  virtual void render();

  virtual bool onEnter(); 
  virtual bool onExit(); 

  virtual std::string getStateID() const { return s_menuID; }

  private:

  virtual void setCallbacks(const std::vector<Callback>& 
  callbacks);

  // call back functions for menu items
  static void s_menuToPlay();
  static void s_exitFromMenu();

  static const std::string s_menuID;

  std::vector<GameObject*> m_gameObjects;
};
```

由于`MainMenuState`现在从`MenuState`派生，它当然必须声明和定义`setCallbacks`函数。我们现在可以使用我们的状态解析来加载`MainMenuState`类。我们的`onEnter`函数现在将看起来像这样：

```cpp
bool MainMenuState::onEnter()
{
  // parse the state
  StateParser stateParser;
  stateParser.parseState("test.xml", s_menuID, &m_gameObjects, 
  &m_textureIDList);

  m_callbacks.push_back(0); //pushback 0 callbackID start from 1
  m_callbacks.push_back(s_menuToPlay);
  m_callbacks.push_back(s_exitFromMenu);

  // set the callbacks for menu items
  setCallbacks(m_callbacks);

  std::cout << "entering MenuState\n";
  return true;
}
```

我们创建了一个状态解析器，然后使用它来解析当前状态。我们将任何回调函数推入从`MenuState`继承的`m_callbacks`数组中。现在我们需要定义`setCallbacks`函数：

```cpp
void MainMenuState::setCallbacks(const std::vector<Callback>& 
callbacks)
{
  // go through the game objects
  for(int i = 0; i < m_gameObjects.size(); i++)
  {
    // if they are of type MenuButton then assign a callback 
    based on the id passed in from the file
    if(dynamic_cast<MenuButton*>(m_gameObjects[i]))
    {
      MenuButton* pButton = 
      dynamic_cast<MenuButton*>(m_gameObjects[i]);
      pButton->setCallback(callbacks[pButton->getCallbackID()]);
    }
  }
}
```

我们使用 `dynamic_cast` 来检查对象是否是 `MenuButton` 类型；如果是，我们就进行实际的转换，然后使用对象的 `callbackID` 作为 `callbacks` 向量的索引，并分配正确的函数。虽然这种方法分配回调可能看起来不是很灵活，并且可能实现得更好，但它确实有一个优点；它允许我们将回调保留在它们需要被调用的状态中。这意味着我们不需要一个包含所有回调的大头文件。

我们需要做的最后一个更改是向每个状态添加一个纹理 ID 列表，这样我们就可以清除为该状态加载的所有纹理。打开 `GameState.h` 文件，我们将添加一个 `protected` 变量。

```cpp
protected:
std::vector<std::string> m_textureIDList;
```

我们将在 `onEnter` 中的状态解析器中传递这个类型，然后我们可以在每个状态中的 `onExit` 函数中清除任何已使用的纹理，如下所示：

```cpp
// clear the texture manager
for(int i = 0; i < m_textureIDList.size(); i++)
{
  TheTextureManager::Instance()->
  clearFromTextureMap(m_textureIDList[i]);
}
```

在我们开始运行游戏之前，我们需要将我们的 `MenuButton` 类型注册到 `GameObjectFactory` 中。打开 `Game.cpp` 文件，在 `Game::init` 函数中我们可以注册该类型。

```cpp
TheGameObjectFactory::Instance()->registerType("MenuButton", new MenuButtonCreator());
```

现在，我们可以运行游戏并看到我们的完全数据驱动的 `MainMenuState`。

# 从 XML 文件加载其他状态

我们的 `MainMenuState` 类现在从 XML 文件加载。我们需要让我们的其他状态也这样做。我们将只涵盖已更改的代码，所以假设在阅读本节时其他内容保持不变。

## 加载游戏状态

我们将从 `PlayState.cpp` 和它的 `onEnter` 函数开始。

```cpp
bool PlayState::onEnter()
{
  // parse the state
  StateParser stateParser;
  stateParser.parseState("test.xml", s_playID, &m_gameObjects, 
  &m_textureIDList);

  std::cout << "entering PlayState\n";
  return true;
}
```

我们还必须在 `onExit` 函数中添加我们之前在 `MainMenuState` 中有的新纹理清除代码。

```cpp
// clear the texture manager
for(int i = 0; i < m_textureIDList.size(); i++)
{
  TheTextureManager::Instance()->
  clearFromTextureMap(m_textureIDList[i]);
}
```

这些是我们在这里需要做的唯一更改，但我们还必须更新我们的 XML 文件，以便在 `PlayState` 中加载一些内容。

```cpp
<PLAY>
<TEXTURES>
  <texture filename="helicopter.png" ID="helicopter"/>
  <texture filename="helicopter2.png" ID="helicopter2"/>
</TEXTURES>

<OBJECTS>
  <object type="Player" x="500" y="100" width="128" height="55" 
  textureID="helicopter" numFrames="4"/>
  <object type="Enemy" x="100" y="100" width="128" height="55" 
  textureID="helicopter2" numFrames="4"/>
</OBJECTS>
</PLAY>
```

我们的 `Enemy` 对象现在需要在它的加载函数中设置其初始速度，而不是在构造函数中，否则 `load` 函数将覆盖它。

```cpp
void Enemy::load(const LoaderParams *pParams)
{
  SDLGameObject::load(pParams);
  m_velocity.setY(2);
}
```

最后，我们必须将这些对象注册到工厂中。我们可以在 `Game::init` 函数中这样做，就像 `MenuButton` 对象一样。

```cpp
TheGameObjectFactory::Instance()->registerType("Player", new PlayerCreator());
TheGameObjectFactory::Instance()->registerType("Enemy", new EnemyCreator());
```

## 加载暂停状态

我们的 `PauseState` 类现在必须从 `MenuState` 继承，因为我们希望它包含回调函数。我们必须更新 `PauseState.h` 文件，首先从 `MenuState` 继承。

```cpp
class PauseState : public MenuState
```

我们还必须声明 `setCallbacks` 函数。

```cpp
virtual void setCallbacks(const std::vector<Callback>& callbacks);
```

现在，我们必须更新 `PauseState.cpp` 文件，从 `onEnter` 函数开始。

```cpp
bool PauseState::onEnter()
{
  StateParser stateParser;
  stateParser.parseState("test.xml", s_pauseID, &m_gameObjects, 
  &m_textureIDList);

  m_callbacks.push_back(0);
  m_callbacks.push_back(s_pauseToMain);
  m_callbacks.push_back(s_resumePlay);

  setCallbacks(m_callbacks);

  std::cout << "entering PauseState\n";
  return true;
}
```

`setCallbacks` 函数与 `MainMenuState` 完全相同。

```cpp
void PauseState::setCallbacks(const std::vector<Callback>& 
callbacks)
{
  // go through the game objects
  for(int i = 0; i < m_gameObjects.size(); i++)
  {
    // if they are of type MenuButton then assign a callback based 
    on the id passed in from the file
    if(dynamic_cast<MenuButton*>(m_gameObjects[i]))
    {
      MenuButton* pButton = 
      dynamic_cast<MenuButton*>(m_gameObjects[i]);
      pButton->setCallback(callbacks[pButton->getCallbackID()]);
    }
  }
}
```

最后，我们必须将纹理清除代码添加到 `onExit` 中。

```cpp
// clear the texture manager
for(int i = 0; i < m_textureIDList.size(); i++)
{
  TheTextureManager::Instance()->
  clearFromTextureMap(m_textureIDList[i]);
}
```

然后更新我们的 XML 文件以包含此状态。

```cpp
<PAUSE>
<TEXTURES>
  <texture filename="resume.png" ID="resumebutton"/>
  <texture filename="main.png" ID="mainbutton"/>
</TEXTURES>

<OBJECTS>
  <object type="MenuButton" x="200" y="100" width="200" 
  height="80" textureID="mainbutton" numFrames="0" 
  callbackID="1"/>
  <object type="MenuButton" x="200" y="300" width="200" 
  height="80" textureID="resumebutton" numFrames="0" 
  callbackID="2"/>
</OBJECTS>
</PAUSE>
```

## 加载游戏结束状态

我们最后一个状态是 `GameOverState`。这与其他状态非常相似，我们只涵盖已更改的部分。由于我们希望 `GameOverState` 处理回调，它现在将从 `MenuState` 继承。

```cpp
class GameOverState : public MenuState
```

然后，我们将声明 `setCallbacks` 函数。

```cpp
virtual void setCallbacks(const std::vector<Callback>& callbacks);
```

`onEnter` 函数现在应该看起来非常熟悉了。

```cpp
bool GameOverState::onEnter()
{
  // parse the state
  StateParser stateParser;
  stateParser.parseState("test.xml", s_gameOverID, &m_gameObjects, 
  &m_textureIDList);
  m_callbacks.push_back(0);
  m_callbacks.push_back(s_gameOverToMain);
  m_callbacks.push_back(s_restartPlay);

  // set the callbacks for menu items
  setCallbacks(m_callbacks);

  std::cout << "entering PauseState\n";
  return true;
}
```

纹理清除方法与之前的状态相同，所以我们将其留给你自己实现。实际上，`onExit`在状态之间看起来非常相似，因此为`GameState`创建一个通用实现是一个好主意；我们再次将其留给你。

你可能已经注意到了`onEnter`函数之间的相似性。有一个默认的`onEnter`实现会很好，但不幸的是，由于需要指定不同的回调函数，我们的回调实现将不允许这样做，这也是其主要缺陷之一。

我们的`AnimatedGraphic`类现在需要在它的`load`函数中从`LoaderParams`获取`animSpeed`值。

```cpp
void AnimatedGraphic::load(const LoaderParams *pParams)
{
  SDLGameObject::load(pParams);
  m_animSpeed = pParams->getAnimSpeed();
}
```

我们还必须将此类型注册到`GameObjectFactory`。

```cpp
TheGameObjectFactory::Instance()->registerType("AnimatedGraphic", new AnimatedGraphicCreator());
```

最后，我们可以更新 XML 文件以包括此状态：

```cpp
<GAMEOVER>
<TEXTURES>
  <texture filename="gameover.png" ID="gameovertext"/>
  <texture filename="main.png" ID="mainbutton"/>
  <texture filename="restart.png" ID="restartbutton"/>
</TEXTURES>

<OBJECTS>
  <object type="AnimatedGraphic" x="200" y="100" width="190" 
  height="30" textureID="gameovertext" numFrames="2" 
  animSpeed="2"/>
  <object type="MenuButton" x="200" y="200" width="200" 
  height="80" textureID="mainbutton" numFrames="0" 
  callbackID="1"/>
  <object type="MenuButton" x="200" y="300" width="200" 
  height="80" textureID="restartbutton" numFrames="0" 
  callbackID="2"/>
</OBJECTS>
</GAMEOVER>
```

现在，我们所有的状态都是从 XML 文件中加载的，其中最大的好处之一是当你更改值时，你不需要重新编译游戏。你可以更改 XML 文件以移动位置或为对象使用不同的纹理；如果 XML 已保存，则只需再次运行游戏，它将使用新值。这对我们来说节省了大量时间，并使我们能够完全控制状态，而无需重新编译我们的游戏。

# 摘要

从外部文件加载数据是编程游戏中的一个极其有用的工具。这一章节使我们的游戏能够实现这一点，并将其应用于我们现有的所有状态。我们还介绍了如何使用工厂在运行时动态创建对象。下一章将涵盖更多数据驱动的设计以及瓦片地图，这样我们就可以真正地将游戏解耦，并允许它使用外部源而不是硬编码的值。
