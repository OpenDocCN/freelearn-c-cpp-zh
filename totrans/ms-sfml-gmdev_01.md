# 第一章. 内部结构 - 设置后端

# 简介

任何软件的核心是什么？在构建一个完整规模的项目过程中，这个问题的答案会逐渐显现出来，这本身可能是一项艰巨的任务，尤其是从零开始。是后端的设计和能力，要么通过利用其力量全力推动游戏前进，要么因为未实现的能力而使游戏陷入默默无闻。在这里，我们将讨论保持任何项目持续运行和站立的基础。

在本章中，我们将涵盖以下主题：

+   Windows 和 Linux 操作系统的实用函数和文件系统特定信息

+   实体组件系统模式的基本原理

+   窗口、事件和资源管理技术

+   创建和维护应用程序状态

+   图形用户界面基础

+   2D RPG 游戏项目的必备要素

有很多内容要介绍，所以我们不要浪费时间！

# 速度和源代码示例

我们将要讨论的所有系统都可以有整本书来专门介绍。由于时间和纸张都有限，我们只会简要回顾它们的基本原理，这足以让我们对这里提供的信息感到舒适。

### 注意

请记住，尽管我们在这章中不会深入细节，但本书附带的代码是一个很好的资源，可以查阅和实验以获得更多细节和熟悉度。强烈建议在阅读本章时回顾它，以便全面掌握。

# 常用实用函数

让我们从查看一个常见的函数开始，这个函数将用于确定我们的可执行文件所在的目录的完整绝对路径。不幸的是，在所有平台上都没有统一的方法来做这件事，所以我们将不得不为每个平台实现这个实用函数的版本，从 Windows 开始：

```cpp
#ifdef RUNNING_WINDOWS 
#define WIN32_LEAN_AND_MEAN 
#include <windows.h> 
#include <Shlwapi.h> 

```

首先，我们检查`RUNNING_WINDOWS`宏是否定义。这是一种基本技术，可以用来让代码库的其余部分知道它正在运行哪个操作系统。接下来，我们定义了另一个特定的定义，针对我们包含的 Windows 头文件。这大大减少了在过程中包含的其他头文件的数量。

在包含了 Windows OS 的所有必要头文件后，让我们看看实际函数是如何实现的：

```cpp
inline std::string GetWorkingDirectory() 
{ 
   HMODULE hModule = GetModuleHandle(nullptr); 
   if (!hModule) { return ""; } 
   char path[256]; 
   GetModuleFileName(hModule,path,sizeof(path)); 
   PathRemoveFileSpec(path); 
   strcat_s(path,""); 
   return std::string(path); 
} 

```

首先，我们获取由我们的可执行文件创建的进程句柄。在构建并填充了临时路径缓冲区、路径字符串、名称和扩展名之后，我们移除可执行文件的名字和扩展名。然后，我们在路径末尾添加一个尾随斜杠，并将其作为`std::string`返回。

如果有一种方法可以获取指定目录内文件列表，那将非常有用：

```cpp
inline std::vector<std::string> GetFileList( 
   const std::string& l_directory, 
   const std::string& l_search = "*.*") 
{ 
   std::vector<std::string> files; 
   if(l_search.empty()) { return files; } 
   std::string path = l_directory + l_search; 
   WIN32_FIND_DATA data; 
   HANDLE found = FindFirstFile(path.c_str(), &data); 
   if (found == INVALID_HANDLE_VALUE) { return files; } 
   do{ 
       if (!(data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) 
       { 
          files.emplace_back(data.cFileName); 
       } 
     }while (FindNextFile(found, &data)); 
   FindClose(found); 
   return files; 
} 

```

就像目录函数一样，这也是 Windows 特有的。它返回一个表示文件名和扩展名的字符串向量。一旦构建完成，就会拼接一个路径字符串。`l_search`参数提供了一个默认值，以防未指定。默认情况下列出所有文件。

在创建一个将保存我们的搜索数据的结构之后，我们将其传递给另一个 Windows 特定函数，该函数将找到目录中的第一个文件。其余的工作在`do-while`循环中完成，该循环检查找到的项目实际上是否不是目录。然后适当的项被推入一个向量，稍后返回。

## Linux 版本

如前所述，这两个先前的函数仅在 Windows 上有效。为了添加对基于 Linux 的操作系统运行系统的支持，我们需要以不同的方式实现它们。让我们首先包括适当的头文件：

```cpp
#elif defined RUNNING_LINUX 
#include <unistd.h> 
#include <dirent.h> 

```

幸运的是，Linux 确实提供了一个单次调用的解决方案来找到我们的可执行文件的确切位置：

```cpp
inline std::string GetWorkingDirectory() 
{ 
   char cwd[1024]; 
   if(!getcwd(cwd, sizeof(cwd))){ return ""; } 
   return std::string(cwd) + std::string("/"); 
} 

```

注意，我们仍然在末尾添加一个尾随斜杠。

获取特定目录的文件列表这次稍微复杂一些：

```cpp
inline std::vector<std::string> GetFileList( 
   const std::string& l_directory, 
   const std::string& l_search = "*.*") 
{ 
   std::vector<std::string> files; 

   DIR *dpdf; 
   dpdf = opendir(l_directory.c_str()); 
   if (!dpdf) { return files; } 
   if(l_search.empty()) { return files; } 
   std::string search = l_search; 
   if (search[0] == '*') { search.erase(search.begin()); } 
   if (search[search.length() - 1] == '*') { search.pop_back(); } 
  struct dirent *epdf; 
  while (epdf = readdir(dpdf)) { 
    std::string name = epdf->d_name; 
    if (epdf->d_type == DT_DIR) { continue; } 
    if (l_search != "*.*") { 
      if (name.length() < search.length()) { continue; } 
      if (search[0] == '.') { 
        if (name.compare(name.length() - search.length(), 
          search.length(), search) != 0) 
        { continue; } 
      } else if (name.find(search) == std::string::npos) { 
        continue; 
      } 
    } 
    files.emplace_back(name); 
  } 
  closedir(dpdf); 
  return files; 
} 

```

我们以之前相同的方式开始，通过创建一个字符串向量。然后通过`opendir()`函数获取目录流指针。如果它不是`NULL`，我们就开始修改搜索字符串。与更花哨的 Windows 替代方案不同，我们不能仅仅将一个搜索字符串传递给一个函数，然后让操作系统完成所有的匹配。在这种情况下，它更接近于匹配返回的文件名中的特定搜索字符串，因此需要剪除表示任何含义的星号符号。

接下来，我们利用`readdir()`函数在一个`while`循环中，该循环将逐个返回目录条目结构的指针。我们还想排除文件列表中的任何目录，因此检查条目的类型是否不等于`DT_DIR`。

最后，开始字符串匹配。假设我们不是在寻找任何具有任何扩展名的文件（用`"*.*"`表示），将首先根据长度比较条目的名称与搜索字符串。如果我们要搜索的字符串长度比文件名本身长，那么可以安全地假设我们没有匹配。否则，将再次分析搜索字符串以确定文件名对于正匹配是否重要。如果第一个字符是点，则表示它不重要，因此将文件名与搜索字符串相同长度的末尾部分与搜索字符串本身进行比较。然而，如果名称很重要，我们只需在文件名中搜索搜索字符串。

一旦程序完成，目录被关闭，表示文件的字符串向量被返回。

## 其他一些辅助函数

有时候，在读取文本文件时，能够获取一个包含空格的字符串，同时仍然保持空白分隔符，这很方便。在这种情况下，我们可以使用引号以及这个特殊函数，它帮助我们从一个空白分隔的文件中读取整个引号部分：

```cpp
inline void ReadQuotedString(std::stringstream& l_stream, 
  std::string& l_string) 
{ 
  l_stream >> l_string; 
  if (l_string.at(0) == '"'){ 
    while (l_string.at(l_string.length() - 1) != '"' || 
      !l_stream.eof()) 
    { 
      std::string str; 
      l_stream >> str; 
      l_string.append(" " + str); 
    } 
  } 
  l_string.erase(std::remove( 
    l_string.begin(), l_string.end(), '"'), l_string.end()); 
} 

```

流的第一个部分被输入到参数字符串中。如果它确实以双引号开头，则会启动一个`while`循环，将字符串附加到其中，直到它以另一个双引号结束，或者直到流到达末尾。最后，从字符串中删除所有双引号，得到最终结果。

插值是程序员工具箱中的另一个有用工具。想象一下，在两个不同时间点有两个不同的值，然后想要预测在这两个时间框架之间的某个点值会是什么。这个简单的计算使得这成为可能：

```cpp
template<class T> 
inline T Interpolate(float tBegin, float tEnd, 
   const T& begin_val, const T& end_val, float tX) 
{ 
   return static_cast<T>(( 
      ((end_val - begin_val) / (tEnd - tBegin)) * 
      (tX - tBegin)) + begin_val); 
} 

```

接下来，让我们看看几个可以帮助我们更好地居中`sf::Text`实例的函数：

```cpp
inline float GetSFMLTextMaxHeight(const sf::Text& l_text) { 
  auto charSize = l_text.getCharacterSize(); 
  auto font = l_text.getFont(); 
  auto string = l_text.getString().toAnsiString(); 
  bool bold = (l_text.getStyle() & sf::Text::Bold); 
  float max = 0.f; 
  for (size_t i = 0; i < string.length(); ++i) { 
    sf::Uint32 character = string[i]; 
    auto glyph = font->getGlyph(character, charSize, bold); 
    auto height = glyph.bounds.height; 
    if (height <= max) { continue; } 
    max = height; 
  } 
  return max; 
} 

inline void CenterSFMLText(sf::Text& l_text) { 
  sf::FloatRect rect = l_text.getLocalBounds(); 
  auto maxHeight = Utils::GetSFMLTextMaxHeight(l_text); 
  l_text.setOrigin( 
    rect.left + (rect.width * 0.5f), 
    rect.top + ((maxHeight >= rect.height ? 
      maxHeight * 0.5f : rect.height * 0.5f))); 
} 

```

使用 SFML 文本有时可能很棘手，尤其是在居中非常重要的时候。一些字符，根据字体和其他不同属性，实际上可以超过包围`sf::Text`实例的边界框的高度。为了解决这个问题，第一个函数遍历特定文本实例的每个字符，并获取表示它的字体字形。然后检查其高度并跟踪，以便确定整个文本的最大高度并返回。

第二个函数可以用来设置`sf::Text`实例的绝对中心作为其原点，以便实现完美的结果。在获取其局部边界框并计算最大高度后，这些信息被用来将我们的文本的原始点移动到其中心。

## 生成随机数

大多数游戏都依赖于一定程度上的随机性。虽然简单地使用`rand()`的经典方法可能很有吸引力，但它只能带你走这么远。生成随机负数或浮点数至少不是那么直接，而且它的范围非常糟糕。幸运的是，C++的新版本提供了统一分布和随机数发生器的形式作为答案：

```cpp
#include <random> 
#include <SFML/System/Mutex.hpp> 
#include <SFML/System/Lock.hpp> 

class RandomGenerator { 
public: 
  RandomGenerator() : m_engine(m_device()){} 
  ... 
  float operator()(float l_min, float l_max) { 
    return Generate(l_min, l_max); 
  } 
  int operator()(int l_min, int l_max) { 
    return Generate(l_min, l_max); 
  } 
private: 
  std::random_device m_device; 
  std::mt19937 m_engine; 
  std::uniform_int_distribution<int> m_intDistribution; 
  std::uniform_real_distribution<float> m_floatDistribution; 
  sf::Mutex m_mutex; 
}; 

```

首先，注意`include`语句。`random`库为我们提供了生成数字所需的一切。除此之外，我们还将使用 SFML 的互斥锁和锁，以防止我们的代码被多个独立的线程访问时出现混乱。

`std::random_device`类是一个随机数生成器，用于初始化引擎，该引擎将用于进一步的生成。引擎本身基于*Marsenne Twister*算法，并产生高质量的随机**无符号整数**，这些整数可以通过一个**均匀分布**对象进行过滤，以获得特定范围内的数字。理想情况下，由于构建和销毁这些对象相当昂贵，我们希望保留这个类的单个副本。正因为如此，我们在同一个类中将整数和浮点数分布放在一起。

为了方便起见，圆括号运算符被重载以接受整数和浮点数类型的数字范围。它们调用`Generate`方法，该方法也被重载以处理这两种数据类型：

```cpp
int Generate(int l_min, int l_max) { 
  sf::Lock lock(m_mutex); 
  if (l_min > l_max) { std::swap(l_min, l_max); } 
  if (l_min != m_intDistribution.min() || 
    l_max != m_intDistribution.max()) 
  { 
    m_intDistribution = 
      std::uniform_int_distribution<int>(l_min, l_max); 
  } 
  return m_intDistribution(m_engine); 
} 

float Generate(float l_min, float l_max) { 
  sf::Lock lock(m_mutex); 
  if (l_min > l_max) { std::swap(l_min, l_max); } 
  if (l_min != m_floatDistribution.min() || 
    l_max != m_floatDistribution.max()) 
  { 
    m_floatDistribution = 
      std::uniform_real_distribution<float>(l_min, l_max); 
  } 
  return m_floatDistribution(m_engine); 
} 

```

在生成开始之前，我们必须建立一个锁以确保线程安全。因为`l_min`和`l_max`值的顺序很重要，我们必须检查提供的值是否没有反转，如果是，则进行交换。此外，如果需要使用不同的范围，必须重建均匀分布对象，因此也设置了相应的检查。最后，在经历了所有这些麻烦之后，我们准备通过利用分布的圆括号运算符来返回随机数，并将引擎实例传递给它。

# 服务定位器模式

通常，我们的一个或多个类将需要访问我们的代码库的另一个部分。通常，这不是一个大问题。你只需要传递一个或两个指针，或者可能将它们存储为需要类的数据成员。然而，随着代码量的增加，类之间的关系变得越来越复杂。依赖性可能会增加到某个程度，以至于一个特定的类将具有比实际方法更多的参数/设置器。为了方便起见，有时传递单个指针/引用而不是十个更好。这就是**服务定位器**模式发挥作用的地方：

```cpp
class Window; 
class EventManager; 
class TextureManager; 
class FontManager; 
... 
struct SharedContext{ 
  SharedContext(): 
    m_wind(nullptr), 
    m_eventManager(nullptr), 
    m_textureManager(nullptr), 
    m_fontManager(nullptr), 
    ... 
  {} 

  Window* m_wind; 
  EventManager* m_eventManager; 
  TextureManager* m_textureManager; 
  FontManager* m_fontManager; 
  ... 
}; 

```

如您所见，它只是一个包含多个指向我们项目核心类指针的`struct`。所有这些类都是提前声明的，以避免不必要的`include`语句，从而减少编译过程的膨胀。

# 实体组件系统核心

让我们来看看我们的游戏实体将如何表示的本质。为了实现最高的可维护性和代码模块化，最好使用组合。实体组件系统正是如此。为了保持简洁，我们不会深入探讨实现细节。这只是一个为了熟悉后续将使用的代码的快速概述。

ECS 模式由三个基石组成，使其成为可能：实体、组件和系统。理想情况下，实体只是一个标识符，就像一个整数一样简单。组件是包含几乎没有逻辑的数据容器。会有多种类型的组件，如位置、可移动、可绘制等，它们本身并没有太多意义，但组合起来将形成复杂的实体。这种组合将使在任何给定时间保存任何实体的状态变得极其容易。

实现组件的方式有很多。其中一种就是简单地拥有一个基类组件，并从中继承：

```cpp
class C_Base{ 
public: 
  C_Base(const Component& l_type): m_type(l_type){} 
  virtual ~C_Base(){} 

  Component GetType() const { return m_type; } 

  friend std::stringstream& operator >>( 
    std::stringstream& l_stream, C_Base& b) 
    { 
      b.ReadIn(l_stream); 
      return l_stream; 
    } 

  virtual void ReadIn(std::stringstream& l_stream) = 0; 
protected: 
  Component m_type; 
}; 

```

`Component`类型只是一个*枚举类*，列出了我们可以在项目中拥有的不同类型的组件。除此之外，这个基类还提供了一种从字符串流中填充组件数据的方法，以便在读取文件时更容易地加载它们。

为了正确管理属于实体的组件集合，我们需要某种类型的管理类：

```cpp
class EntityManager{ 
public: 
  EntityManager(SystemManager* l_sysMgr, 
    TextureManager* l_textureMgr); 
  ~EntityManager(); 

  int AddEntity(const Bitmask& l_mask); 
  int AddEntity(const std::string& l_entityFile); 
  bool RemoveEntity(const EntityId& l_id); 

  bool AddComponent(const EntityId& l_entity, 
    const Component& l_component); 

  template<class T> 
  void AddComponentType(const Component& l_id) { ... } 

  template<class T> 
  T* GetComponent(const EntityId& l_entity, 
    const Component& l_component){ ... } 

  bool RemoveComponent(const EntityId& l_entity, 
    const Component& l_component); 
  bool HasComponent(const EntityId& l_entity, 
    const Component& l_component) const; 
  void Purge(); 
private: 
  ... 
}; 

```

如您所见，这是一种相当基本的处理我们称之为实体的数据集的方法。`EntityId`数据类型只是一个**无符号整数**的类型定义。组件的创建是通过利用工厂模式、lambda 表达式和模板来实现的。这个类还负责从可能看起来像这样的文件中加载实体：

```cpp
Name Player 
Attributes 255 
|Component|ID|Individual attributes| 
Component 0 0 0 1 
Component 1 Player 
Component 2 0 
Component 3 128.0 1024.0 1024.0 1 
Component 4 
Component 5 20.0 20.0 0.0 0.0 2 
Component 6 footstep:1,4 
Component 7 

```

`Attributes`字段是一个位掩码，其值用于确定实体具有哪些组件类型。实际的组件数据也存储在这个文件中，并且通过组件基类的`ReadIn`方法进行加载。

ECS 设计中最后一部分是系统。这是所有逻辑发生的地方。就像组件一样，可以有负责碰撞、渲染、移动等多种类型的系统。每个系统都必须继承自系统的基类并实现所有纯虚方法：

```cpp
class S_Base : public Observer{ 
public: 
  S_Base(const System& l_id, SystemManager* l_systemMgr); 
  virtual ~S_Base(); 

  bool AddEntity(const EntityId& l_entity); 
  bool HasEntity(const EntityId& l_entity) const; 
  bool RemoveEntity(const EntityId& l_entity); 

  System GetId() const; 

  bool FitsRequirements(const Bitmask& l_bits) const; 
  void Purge(); 

  virtual void Update(float l_dT) = 0; 
  virtual void HandleEvent(const EntityId& l_entity, 
    const EntityEvent& l_event) = 0; 
protected: 
  ... 
}; 

```

系统具有它们使用的组件签名，以及满足这些签名要求的实体列表。当一个实体通过添加或删除组件被修改时，每个系统都会运行检查，以便将其添加到或从自身中删除。注意从`Observer`类继承。这是另一种有助于实体和系统之间通信的图案。

一个`Observer`类本身只是一个接口，包含一个必须由所有派生类实现的纯虚方法：

```cpp
class Observer{ 
public: 
  virtual ~Observer(){} 
  virtual void Notify(const Message& l_message) = 0; 
}; 

```

它利用发送给特定目标所有观察者的消息。这个类的派生对象如何响应消息完全取决于它本身。

形状和大小各异的系统需要像实体一样进行管理。为此，我们还有一个管理类：

```cpp
class SystemManager{ 
public: 
  ... 
  template<class T> 
  void AddSystem(const System& l_system) { ... } 

  template<class T> 
  T* GetSystem(const System& l_system){ ... } 
  void AddEvent(const EntityId& l_entity, const EventID& l_event); 

  void Update(float l_dT); 
  void HandleEvents(); 
  void Draw(Window* l_wind, unsigned int l_elevation); 

  void EntityModified(const EntityId& l_entity, 
    const Bitmask& l_bits); 
  void RemoveEntity(const EntityId& l_entity); 

  void PurgeEntities(); 
  void PurgeSystems(); 
private: 
  ... 
  MessageHandler m_messages; 
}; 

```

这同样利用了工厂模式，通过使用模板和 lambda 来注册不同类型的类，以便稍后可以通过使用`System`数据类型（它是一个`enum class`）来构建它们。开始看到模式了吗？

系统管理器拥有一个类型为`MessageHandler`的数据成员。这是观察者模式的一部分。让我们看看它做了什么：

```cpp
class MessageHandler{ 
public: 
  bool Subscribe(const EntityMessage& l_type, 
    Observer* l_observer){ ... } 
  bool Unsubscribe(const EntityMessage& l_type, 
    Observer* l_observer){ ... } 
  void Dispatch(const Message& l_msg){ ... } 
private: 
  Subscribtions m_communicators; 
}; 

```

消息处理器只是`Communicator`对象的集合，如下所示：

```cpp
using Subscribtions = 
  std::unordered_map<EntityMessage,Communicator>; 

```

每种可能的`EntityMessage`类型（它只是一个`enum class`）都与一个负责向所有观察者发送消息的通信器相关联。观察者可以订阅或取消订阅特定消息类型。如果他们订阅了该类型，当调用`Dispatch`方法时，他们将接收到该消息。

`Communicator` 类本身相当简单：

```cpp
class Communicator{ 
public: 
  virtual ~Communicator(){ m_observers.clear(); } 
  bool AddObserver(Observer* l_observer){ ... } 
  bool RemoveObserver(Observer* l_observer){ ... } 
  bool HasObserver(const Observer* l_observer) const { ... } 
  void Broadcast(const Message& l_msg){ ... } 
private: 
  ObserverContainer m_observers; 
}; 

```

如您所知，它支持添加和删除观察者，并提供了一种向所有观察者广播消息的方法。观察者的实际容器只是一个指针的向量：

```cpp
// Not memory-owning pointers. 
using ObserverContainer = std::vector<Observer*>; 

```

# 资源管理

在较大的项目中，另一个至关重要的部分是有效管理资源的方法。由于我们将拥有几种不同类型的资源，例如纹理、字体和声音，因此为所有这些资源分别拥有单独的管理器是有意义的。是时候有一个基类了：

```cpp
template<typename Derived, typename T> 
class ResourceManager{ 
public: 
  ResourceManager(const std::string& l_pathsFile){ 
    LoadPaths(l_pathsFile); 
  } 
  virtual ~ResourceManager(){ ... } 
  T* GetResource(const std::string& l_id){ ... } 
  std::string GetPath(const std::string& l_id){ ... } 
  bool RequireResource(const std::string& l_id){ ... } 
  bool ReleaseResource(const std::string& l_id){ ... } 
  void PurgeResources(){ ... } 
protected: 
  bool Load(T* l_resource, const std::string& l_path) { 
    return static_cast<Derived*>(this)->Load(l_resource, l_path); 
  } 
private: 
  ... 
}; 

```

这种特定资源管理系统的理念是某些代码段需要并随后释放特定的资源标识符。第一次需要资源时，它将被加载到内存中并保留在那里。之后每次需要时，将简单地增加一个与之存储的整数。这个整数表示依赖于该资源加载的代码实例的数量。一旦它们完成使用资源，它就开始释放，每次都会减少计数器。当它达到零时，资源将从内存中删除。

公平地说，我们的资源管理器基类在创建资源实例之后使用**奇特重复的模板模式**来设置资源实例。由于管理器类实际上不需要在任何地方存储在一起，因此静态多态比使用虚方法更有意义。由于纹理、字体和声音可能以不同的方式加载，每个后续管理器都必须实现自己的`Load`方法版本，如下所示：

```cpp
class TextureManager : public ResourceManager<TextureManager, 
  sf::Texture> 
{ 
public: 
  TextureManager() : ResourceManager("textures.cfg"){} 

  bool Load(sf::Texture* l_resource, const std::string& l_path){ 
    return l_resource->loadFromFile( 
      Utils::GetWorkingDirectory() + l_path); 
  } 
}; 

```

每个单独的管理器都有自己的文件，列出资源名称和它们路径之间的关系。对于纹理，它可以看起来像这样：

```cpp
Intro media/Textures/intro.png 
PlayerSprite media/Textures/PlayerSheet.png 
... 

```

它通过将每个资源与一个名称相关联，简单地避免了传递路径和文件名，从而避免了这种需求。

# 窗口系统

当处理打开的窗口时，幕后有很多事情在进行。从窗口尺寸和标题到跟踪和处理特殊事件，所有这些都在一个指定的窗口类中集中处理：

```cpp
class Window{
public:
  Window(const std::string& l_title = "Window",
    const sf::Vector2u& l_size = {640,480},
    bool l_useShaders = true);
  ~Window();

  void BeginDraw();
  void EndDraw();

  void Update();

  bool IsDone() const;
  bool IsFullscreen() const;
  bool IsFocused() const;

  void ToggleFullscreen(EventDetails* l_details);
  void Close(EventDetails* l_details = nullptr);

  sf::RenderWindow* GetRenderWindow();
  Renderer* GetRenderer();
  EventManager* GetEventManager();
  sf::Vector2u GetWindowSize();
  sf::FloatRect GetViewSpace();
private:
  ...
};

```

注意两个突出显示的方法。它们将在我们即将讨论的事件管理器中用作回调。同时注意对象类型`Renderer`的返回方法。它是一个实用类，它简单地在一个`RenderWindow`上调用`.draw`方法，从而将其本地化并使其使用着色器变得更加容易。关于这一点，将在第六章中详细介绍，*添加一些收尾工作 – 使用着色器*。

# 应用程序状态

更复杂的应用程序的一个重要方面是跟踪和管理其状态。无论是玩家正在游戏中深入，还是简单地浏览主菜单，我们都希望它能够无缝处理，更重要的是，它应该是自包含的。我们可以通过首先定义我们将要处理的不同状态类型来开始这个过程：

```cpp
enum class StateType { Intro = 1, MainMenu, Game, Loading }; 

```

为了实现无缝集成，我们希望每个状态都能以可预测的方式表现。这意味着状态必须遵循我们提供的接口：

```cpp
class BaseState{ 
friend class StateManager; 
public: 
  BaseState(StateManager* l_stateManager)  
    :m_stateMgr(l_stateManager), m_transparent(false), 
    m_transcendent(false){} 
  virtual ~BaseState(){} 

  virtual void OnCreate() = 0; 
  virtual void OnDestroy() = 0; 

  virtual void Activate() = 0; 
  virtual void Deactivate() = 0; 

  virtual void Update(const sf::Time& l_time) = 0; 
  virtual void Draw() = 0; 
  ... 
  sf::View& GetView(){ return m_view; } 
  StateManager* GetStateManager(){ return m_stateMgr; } 
protected: 
  StateManager* m_stateMgr; 
  bool m_transparent; 
  bool m_transcendent; 
  sf::View m_view; 
}; 

```

游戏中的每个状态都将拥有自己的视图，它可以进行修改。除此之外，它还提供了钩子来实现各种不同场景的逻辑，例如状态的创建、销毁、激活、去激活、更新和渲染。最后，它通过提供`m_transparent`和`m_transcendent`标志，使得在更新和渲染过程中能够与其他状态混合。

管理这些状态相当直接：

```cpp
class StateManager{ 
public: 
  StateManager(SharedContext* l_shared); 
  ~StateManager(); 
  void Update(const sf::Time& l_time); 
  void Draw(); 
  void ProcessRequests(); 
  SharedContext* GetContext(); 
  bool HasState(const StateType& l_type) const; 
  StateType GetNextToLast() const; 
  void SwitchTo(const StateType& l_type); 
  void Remove(const StateType& l_type); 
  template<class T> 
  T* GetState(const StateType& l_type){ ... } 
  template<class T> 
  void RegisterState(const StateType& l_type) { ... } 
  void AddDependent(StateDependent* l_dependent); 
  void RemoveDependent(StateDependent* l_dependent); 
private: 
  ... 
  State_Loading* m_loading; 
  StateDependents m_dependents; 
}; 

```

`StateManager`类是项目中少数几个使用共享上下文的类之一，因为状态本身可能需要访问代码库的任何部分。它还使用工厂模式在运行时动态创建任何与状态类型绑定的状态。

为了保持简单，我们将把加载状态视为一个特殊情况，并始终只允许一个实例存在。加载可能发生在任何状态的转换过程中，因此这样做是有意义的。

关于状态管理值得注意的最后一件事是它维护一个状态依赖项列表。它只是一个从该接口继承的类的 STL 容器：

```cpp
class StateDependent { 
public: 
  StateDependent() : m_currentState((StateType)0){} 
  virtual ~StateDependent(){} 
  virtual void CreateState(const StateType& l_state){} 
  virtual void ChangeState(const StateType& l_state) = 0; 
  virtual void RemoveState(const StateType& l_state) = 0; 
protected: 
  void SetState(const StateType& l_state){m_currentState=l_state;} 
  StateType m_currentState; 
}; 

```

由于处理声音、GUI 元素或实体管理等事物的类需要支持不同的状态，因此它们还必须定义在状态创建、更改或删除时内部发生的事情，以便正确地分配/释放资源，停止更新不在同一状态中的数据，等等。

## 加载状态

那么，我们究竟将如何实现这个加载状态呢？嗯，为了灵活性和通过渲染花哨的加载条来轻松跟踪进度，线程将证明是无价的。需要加载到内存中的数据可以在单独的线程中加载，而加载状态本身则继续更新和渲染，以显示确实有事情在进行。仅仅知道应用程序没有挂起，就应该会让人感到温暖和舒适。

首先，让我们通过提供一个任何线程工作者都可以使用的接口来实现这个系统的基本功能：

```cpp
class Worker { 
public: 
  Worker() : m_thread(&Worker::Work, this), m_done(false), 
    m_started(false) {} 
  void Begin() { 
    if(m_done || m_started) { return; } 
    m_started = true; 
    m_thread.launch(); 
  } 
  bool IsDone() const { return m_done; } 
  bool HasStarted() const { return m_started; } 
protected: 
  void Done() { m_done = true; } 
  virtual void Work() = 0; 
  sf::Thread m_thread; 
  bool m_done; 
  bool m_started; 
}; 

```

它有自己的线程，该线程绑定到名为`Work`的纯虚方法。每当调用`Begin()`方法时，线程就会被启动。为了保护数据不被多个线程同时访问，在敏感调用期间使用`sf::Mutex`类创建一个锁。这个非常基础的类中的其他一切只是为了向外界提供有关工作者状态的信息。

## 文件加载器

现在线程的问题已经解决，我们可以专注于实际加载一些文件了。这个方法将专注于处理文本文件。然而，使用二进制格式应该以几乎相同的方式工作，只是没有文本处理。

让我们看看任何可想到的文件加载类的基类：

```cpp
using LoaderPaths = std::vector<std::pair<std::string, size_t>>; 

class FileLoader : public Worker { 
public: 
  FileLoader(); 
  void AddFile(const std::string& l_file);
  virtual void SaveToFile(const std::string& l_file);

  size_t GetTotalLines() const; 
  size_t GetCurrentLine() const; 
protected: 
  virtual bool ProcessLine(std::stringstream& l_stream) = 0; 
  virtual void ResetForNextFile(); 
  void Work(); 
  void CountFileLines(); 

  LoaderPaths m_files; 
  size_t m_totalLines; 
  size_t m_currentLine; 
}; 

```

有一个明显的可能性，即在某些时候可能需要加载两个或更多文件。`FileLoader`类跟踪所有添加到其中的路径，以及代表该文件中行数的数字。这对于确定加载过程中所取得的进度非常有用。除了每个单独文件的行数外，还跟踪总行数。

这个类提供了一个单一的纯虚方法，称为`ProcessLine`。这将允许派生类定义文件的确切加载和处理方式。

首先，让我们先处理一些基本的事情：

```cpp
FileLoader::FileLoader() : m_totalLines(0), m_currentLine(0) {}
void FileLoader::AddFile(const std::string& l_file) {
  m_files.emplace_back(l_file, 0);
}
size_t FileLoader::GetTotalLines()const {
  sf::Lock lock(m_mutex);
  return m_totalLines;
}
size_t FileLoader::GetCurrentLine()const {
  sf::Lock lock(m_mutex);
  return m_currentLine;
}
void FileLoader::SaveToFile(const std::string& l_file) {}
void FileLoader::ResetForNextFile(){}
```

`ResetForNextFile()`虚拟方法不是必须实现的，但可以用来清除在文件加载期间需要存在的某些内部数据的状态。由于实现此类的文件加载器将只能够在单个方法中一次处理一行，任何通常在该方法内部作为局部变量存储的临时数据都需要放在其他地方。这就是为什么我们必须确保实际上有一种方式可以知道我们何时完成了一个文件的加载并开始加载另一个文件，以及在必要时执行某种操作。

### 注意

注意上面两个获取方法中的互斥锁。它们的存在是为了确保那些变量不会被同时写入和读取。

现在，让我们来看看将在不同线程中执行的一段代码：

```cpp
void FileLoader::Work() { 
  CountFileLines(); 
  if (!m_totalLines) { Done(); return; } 
  for (auto& path : m_files) { 
    ResetForNextFile(); 
    std::ifstream file(path.first); 
    std::string line; 
    std::string name; 
    auto linesLeft = path.second; 
    while (std::getline(file, line)) { 
      { 
        sf::Lock lock(m_mutex); 
        ++m_currentLine; 
        --linesLeft; 
      } 
      if (line[0] == '|') { continue; } 
      std::stringstream keystream(line); 
      if (!ProcessLine(keystream)) { 
        std::cout << 
          "File loader terminated due to an internal error." 
          << std::endl; 
        { 
          sf::Lock lock(m_mutex); 
          m_currentLine += linesLeft; 
        } 
        break; 
      } 
    } 
    file.close(); 
  } 
  Done(); 
} 

```

首先调用一个私有方法来计算即将加载的任何文件中的所有行数。如果由于任何原因，总行数为零，继续下去没有意义，因此在返回之前调用`Worker::Done()`方法。这段代码非常容易忘记，但对于使此功能正常工作至关重要。它所做的只是将`Worker`基类的`m_done`标志设置为`true`，这样外部代码就知道处理已经完成。由于目前没有检查 SFML 线程是否真正完成的方法，这几乎是唯一的选择。

我们开始循环遍历需要加载的不同文件，并在开始工作之前调用重置方法。注意，在尝试打开文件时没有进行检查。这将在介绍下一个方法时进行解释。

在读取文件的每一行时，确保更新所有行数信息非常重要。为了防止两个线程同时访问正在修改的行数，为当前线程建立了一个临时锁。此外，以管道符号开头的行被排除在外，因为这是我们标准的注释说明。

最后，为当前行构造了一个`stringstream`对象，并将其传递给`ProcessLine()`方法。为了加分，它返回一个布尔值，可以表示错误并停止当前文件的处理。如果发生这种情况，该特定文件中的剩余行将添加到总行数中，并且循环被中断。

最后一块拼图是这段代码，负责验证文件的有效性并确定我们面前的工作量：

```cpp
void FileLoader::CountFileLines() {
  m_totalLines = 0;
  m_currentLine = 0;
  for (auto path = m_files.begin(); path != m_files.end();) {
    if (path->first.empty()) { m_files.erase(path); continue; }
    std::ifstream file(path->first);
    if (!file.is_open()) {
      std::cerr << “Failed to load file: “ << path->first
        << std::endl;
      m_files.erase(path);
      continue;
    }
    file.unsetf(std::ios_base::skipws);
    {
      sf::Lock lock(m_mutex);
      path->second = static_cast<size_t>(std::count(
        std::istreambuf_iterator<char>(file),
        std::istreambuf_iterator<char>(), ‘\n’));
      m_totalLines += path->second;
    }
    ++path;
    file.close();
  }
}
```

在设置行数的初始零值后，遍历所有添加的路径并进行检查。我们首先删除任何空路径。然后尝试打开每个路径，如果操作失败则删除。最后，为了获得准确的结果，文件输入流被设置为忽略空行。在建立锁之后，使用`std::count`来计算文件中的行数。然后将这个数字添加到总行数中，路径迭代器前进，文件被正确关闭。

由于此方法消除了不存在或无法打开的文件，因此没有必要在其他任何地方再次检查这些文件。

## 实现加载状态

现在所有东西都已经就绪，我们可以成功实现加载状态：

```cpp
using LoaderContainer = std::vector<FileLoader*>; 

class State_Loading : public BaseState { 
public: 
  ... 
  void AddLoader(FileLoader* l_loader); 
  bool HasWork() const; 
  void SetManualContinue(bool l_continue); 
  void Proceed(EventDetails* l_details); 
private: 
  void UpdateText(const std::string& l_text, float l_percentage); 
  float CalculatePercentage(); 
  LoaderContainer m_loaders; 
  sf::Text m_text; 
  sf::RectangleShape m_rect; 
  unsigned short m_percentage; 
  size_t m_originalWork; 
  bool m_manualContinue; 
}; 

```

状态本身将保留一个指向不同文件加载类指针的向量，这些类分别有自己的文件列表。它还提供了一种将这些对象添加到其中的方法。此外，请注意`Proceed()`方法。这是在即将介绍的的事件管理器中将要使用的一个回调。

对于视觉部分，我们将使用图形的最基本要素：一些文本表示进度百分比，以及一个表示加载条的矩形形状。

让我们看看这个类构建后将要进行的所有设置：

```cpp
void State_Loading::OnCreate() { 
  auto context = m_stateMgr->GetContext(); 
  context->m_fontManager->RequireResource("Main"); 
  m_text.setFont(*context->m_fontManager->GetResource("Main")); 
  m_text.setCharacterSize(14); 
  m_text.setStyle(sf::Text::Bold); 

  sf::Vector2u windowSize = m_stateMgr->GetContext()-> 
    m_wind->GetRenderWindow()->getSize(); 

  m_rect.setFillColor(sf::Color(0, 150, 0, 255)); 
  m_rect.setSize(sf::Vector2f(0.f, 32.f)); 
  m_rect.setOrigin(0.f, 16.f); 
  m_rect.setPosition(0.f, windowSize.y / 2.f); 

  EventManager* evMgr = m_stateMgr->GetContext()->m_eventManager; 
  evMgr->AddCallback(StateType::Loading, "Key_Space", 
    &State_Loading::Proceed, this); 
} 

```

首先，通过共享上下文获取字体管理器。需要并使用名为`"Main"`的字体来设置文本实例。在所有视觉元素设置完毕后，使用事件管理器为加载状态注册回调。这将在稍后介绍，但通过查看参数可以很容易地推断出正在发生的事情。每当按下空格键时，`State_Loading`类的`Proceed`方法将被调用。实际的类实例作为最后一个参数传递。

请记住，按照设计，我们需要的资源也必须被释放。对于加载状态来说，一个完美的释放位置就是它被销毁的时候：

```cpp
void State_Loading::OnDestroy() { 
  auto context = m_stateMgr->GetContext(); 
  EventManager* evMgr = context->m_eventManager; 
  evMgr->RemoveCallback(StateType::Loading, "Key_Space"); 
  context->m_fontManager->ReleaseResource("Main"); 
} 

```

除了释放字体外，空格键的回调也被移除。

接下来，让我们实际编写一些代码，将各个部分组合成一个完整、功能性的整体：

```cpp
void State_Loading::Update(const sf::Time& l_time) 
  if (m_loaders.empty()) {
    if (!m_manualContinue) { Proceed(nullptr); }
    return;
  }
  auto windowSize = m_stateMgr->GetContext()->
    m_wind->GetRenderWindow()->getSize();
  if (m_loaders.back()->IsDone()) {
    m_loaders.back()->OnRemove();
    m_loaders.pop_back();
    if (m_loaders.empty()) {
      m_rect.setSize(sf::Vector2f(
        static_cast<float>(windowSize.x), 16.f));
      UpdateText(".Press space to continue.", 100.f);
      return;
    }
  }
  if (!m_loaders.back()->HasStarted()) {
    m_loaders.back()->Begin();
  }

  auto percentage = CalculatePercentage();
  UpdateText("", percentage);
  m_rect.setSize(sf::Vector2f(
    (windowSize.x / 100) * percentage, 16.f));
}
```

第一个检查用于确定是否由于完成而将所有文件加载器从向量中移除。`m_manualContinue`标志用于让加载状态知道它是否应该等待空格键被按下，或者如果应该自动消失。然而，如果我们仍然有一些加载器在向量中，我们将检查顶部加载器是否已完成其工作。如果是这样，加载器将被弹出，并再次检查向量是否为空，这将需要我们更新加载文本以表示完成。

为了使这个过程完全自动化，我们需要确保在顶文件加载器被移除后，下一个加载器开始工作，这就是下面检查的作用所在。最后，计算进度百分比，并在调整加载条大小以视觉辅助我们之前，更新加载文本以表示该值。

对于此状态，绘图将非常简单：

```cpp
void State_Loading::Draw() { 
  sf::RenderWindow* wind = m_stateMgr->GetContext()-> 
    m_wind->GetRenderWindow(); 
  wind->draw(m_rect); 
  wind->draw(m_text); 
} 

```

首先通过共享上下文获取渲染窗口，然后使用它来绘制代表加载条的文本和矩形形状。

`Proceed`回调方法同样简单：

```cpp
void State_Loading::Proceed(EventDetails* l_details){ 
  if (!m_loaders.empty()) { return; } 
  m_stateMgr->SwitchTo(m_stateMgr->GetNextToLast()); 
} 

```

它必须先进行检查，以确保在所有工作完成之前不切换状态。如果不是这种情况，则使用状态管理器切换到在加载开始之前创建的状态。

其他所有加载状态逻辑基本上由每个方法的单行代码组成：

```cpp
void State_Loading::AddLoader(FileLoader* l_loader) {
 m_loaders.emplace_back(l_loader);
  l_loader->OnAdd();
}
bool State_Loading::HasWork() const { return !m_loaders.empty(); }
void State_Loading::SetManualContinue(bool l_continue) {
  m_manualContinue = l_continue;
}
void State_Loading::Activate(){m_originalWork = m_loaders.size();}
```

虽然这看起来相当简单，但`Activate()`方法扮演着相当重要的角色。由于在这里将加载状态视为一个特殊情况，必须记住一件事：它**永远不会**在应用程序关闭之前被移除。这意味着每次我们再次使用它时，都必须重置一些东西。在这种情况下，是`m_originalWork`数据成员，它只是所有加载类数量的计数。这个数字用于准确计算进度百分比，而重置它的最佳位置是在每次状态再次激活时被调用的方法内部。

# 管理应用程序事件

事件管理是我们提供流畅控制体验的基石之一。任何按键、窗口变化，甚至是稍后我们将要介绍的 GUI 系统创建的定制事件都将由这个系统处理和解决。为了有效地统一来自不同来源的事件信息，我们首先必须通过正确枚举它们来统一它们的类型：

```cpp
enum class EventType { 
  KeyDown = sf::Event::KeyPressed, 
  KeyUp = sf::Event::KeyReleased, 
  MButtonDown = sf::Event::MouseButtonPressed, 
  MButtonUp = sf::Event::MouseButtonReleased, 
  MouseWheel = sf::Event::MouseWheelMoved, 
  WindowResized = sf::Event::Resized, 
  GainedFocus = sf::Event::GainedFocus, 
  LostFocus = sf::Event::LostFocus, 
  MouseEntered = sf::Event::MouseEntered, 
  MouseLeft = sf::Event::MouseLeft, 
  Closed = sf::Event::Closed, 
  TextEntered = sf::Event::TextEntered, 
  Keyboard = sf::Event::Count + 1, Mouse, Joystick, 
  GUI_Click, GUI_Release, GUI_Hover, GUI_Leave 
}; 

enum class EventInfoType { Normal, GUI }; 

```

SFML 事件排在首位，因为它们是唯一遵循严格枚举方案的。然后是实时 SFML 输入类型和四个 GUI 事件。我们还枚举了事件信息类型，这些类型将用于此结构中：

```cpp
struct EventInfo { 
  EventInfo() : m_type(EventInfoType::Normal), m_code(0) {} 
  EventInfo(int l_event) : m_type(EventInfoType::Normal), 
    m_code(l_event) {} 
  EventInfo(const GUI_Event& l_guiEvent): 
    m_type(EventInfoType::GUI), m_gui(l_guiEvent) {} 
  EventInfo(const EventInfoType& l_type) { 
    if (m_type == EventInfoType::GUI) { DestroyGUIStrings(); } 
    m_type = l_type; 
    if (m_type == EventInfoType::GUI){ CreateGUIStrings("", ""); } 
  } 

  EventInfo(const EventInfo& l_rhs) { Move(l_rhs); } 

  EventInfo& operator=(const EventInfo& l_rhs) { 
    if (&l_rhs != this) { Move(l_rhs); } 
    return *this; 
  } 

  ~EventInfo() { 
    if (m_type == EventInfoType::GUI) { DestroyGUIStrings(); } 
  } 
  union { 
    int m_code; 
    GUI_Event m_gui; 
  }; 

  EventInfoType m_type; 
private: 
  void Move(const EventInfo& l_rhs) { 
    if (m_type == EventInfoType::GUI) { DestroyGUIStrings(); } 
    m_type = l_rhs.m_type; 
    if (m_type == EventInfoType::Normal){ m_code = l_rhs.m_code; } 
    else { 
      CreateGUIStrings(l_rhs.m_gui.m_interface, 
        l_rhs.m_gui.m_element); 
      m_gui = l_rhs.m_gui; 
    } 
  } 

  void DestroyGUIStrings() { 
    m_gui.m_interface.~basic_string(); 
    m_gui.m_element.~basic_string(); 
  } 

  void CreateGUIStrings(const std::string& l_interface, 
    const std::string& l_element) 
  { 
    new (&m_gui.m_interface) std::string(l_interface); 
    new (&m_gui.m_element) std::string(l_element); 
  } 
}; 

```

因为我们不仅关心发生的事件类型，还需要有一种良好的方式来存储与之相关的附加数据。C++11 的无限制联合是这一点的完美候选人。唯一的缺点是现在我们必须手动管理联合内部的数据，这包括数据分配和直接调用析构函数。

当事件回调被调用时，向它们提供实际的事件信息是个好主意。因为可以为特定的回调构造更复杂的要求，所以我们这次不能使用联合。任何可能相关的信息都需要被存储，这正是这里所做的事情：

```cpp
struct EventDetails { 
  EventDetails(const std::string& l_bindName): m_name(l_bindName){ 
    Clear(); 
  } 

  std::string m_name; 
  sf::Vector2i m_size; 
  sf::Uint32 m_textEntered; 
  sf::Vector2i m_mouse; 
  int m_mouseWheelDelta; 
  int m_keyCode; // Single key code. 

  std::string m_guiInterface; 
  std::string m_guiElement; 
  GUI_EventType m_guiEvent; 

  void Clear() { ... } 
}; 

```

这个结构填充了在事件处理过程中可用的所有信息，然后作为参数传递给被调用的回调。它还提供了一个`Clear()`方法，因为它的创建不仅限于回调期间，而是存在于绑定结构内部：

```cpp
using Events = std::vector<std::pair<EventType, EventInfo>>; 

struct Binding { 
  Binding(const std::string& l_name) : m_name(l_name), 
    m_details(l_name), c(0) {} 
  void BindEvent(EventType l_type, EventInfo l_info = EventInfo()) 
  { ... } 

  Events m_events; 
  std::string m_name; 
  int c; // Count of events that are "happening". 

  EventDetails m_details; 
}; 

```

绑定实际上允许事件被分组在一起，以形成更复杂的要求。从多个键需要同时按下以执行操作的角度来考虑，例如*Ctrl* + *C*复制文本。这种情况的绑定将等待两个事件：*Ctrl*键和*C*键。

## 事件管理器接口

在涵盖了所有关键部分之后，剩下的就是正确管理一切。让我们从一些类型定义开始：

```cpp
using Bindings = std::unordered_map<std::string, 
  std::unique_ptr<Binding>>; 
using CallbackContainer = std::unordered_map<std::string, 
  std::function<void(EventDetails*)>>; 
enum class StateType; 
using Callbacks = std::unordered_map<StateType, 
  CallbackContainer>; 

```

所有绑定都附加到特定的名称上，当应用程序启动时从`keys.cfg`文件中加载。它遵循一个基本格式，如下所示：

```cpp
Window_close 0:0 
Fullscreen_toggle 5:89 
Intro_Continue 5:57 
Mouse_Left 9:0 

```

当然，这些都是非常基础的例子。更复杂的绑定会有多个通过空格分隔的事件。

回调也存储在一个*无序映射*中，以及与它们监视的绑定的名称相关联。然后根据状态对实际的回调容器进行分组，以避免在按下类似键时调用多个函数/方法。正如你可以想象的那样，事件管理器将继承自`StateDependent`类，正是出于这个原因：

```cpp
class EventManager : public StateDependent{ 
public: 
  ... 
  bool AddBinding(std::unique_ptr<Binding> l_binding); 
  bool RemoveBinding(std::string l_name); 
  void ChangeState(const StateType& l_state); 
  void RemoveState(const StateType& l_state); 
  void SetFocus(bool l_focus); 

  template<class T> 
  bool AddCallback(const StateType& l_state, 
    const std::string& l_name,  
    void(T::*l_func)(EventDetails*), T* l_instance) 
  { ... } 

  template<class T> 
  bool AddCallback(const std::string& l_name, 
    void(T::*l_func)(EventDetails*), T* l_instance) 
  { ... } 

  bool RemoveCallback(const StateType& l_state, 
    const std::string& l_name){ ... } 
  void HandleEvent(sf::Event& l_event); 
  void HandleEvent(GUI_Event& l_event); 
  void Update(); 
  sf::Vector2i GetMousePos(sf::RenderWindow* l_wind = nullptr) 
    const { ... } 
private: 
  ... 
  Bindings m_bindings; 
  Callbacks m_callbacks; 
}; 

```

再次强调，这相当简单。由于这是一个状态相关的类，它需要实现`ChangeState()`和`RemoveState()`方法。它还跟踪窗口焦点何时获得/丢失，以避免轮询最小化/未聚焦窗口的事件。提供了两种版本的`AddCallback`：一个用于指定状态，另一个用于当前状态。还有为支持的所有事件类型提供的单独的`HandleEvent()`方法。到目前为止，我们只有两种：SFML 事件和 GUI 事件。后者将在下一节中使用。

# 图形用户界面的使用

在一个计算机基本上是每个家庭必需品的时代，以一种友好的方式与应用程序交互是必不可少的。GUI（图形用户界面）的整个主题本身就可以填满多本书，所以为了保持简单，我们只将触及我们必须要处理的部分：

```cpp
class GUI_Manager : public StateDependent{ 
  friend class GUI_Interface; 
public: 
  ... 
  bool AddInterface(const StateType& l_state, 
    const std::string& l_name); 
  bool AddInterface(const std::string& l_name); 
  GUI_Interface* GetInterface(const StateType& l_state, 
    const std::string& l_name); 
  GUI_Interface* GetInterface(const std::string& l_name); 
  bool RemoveInterface(const StateType& l_state, 
    const std::string& l_name); 
  bool RemoveInterface(const std::string& l_name); 
  bool LoadInterface(const StateType& l_state, 
    const std::string& l_interface, const std::string& l_name); 
  bool LoadInterface(const std::string& l_interface, 
    const std::string& l_name); 
  void ChangeState(const StateType& l_state); 
  void RemoveState(const StateType& l_state); 
  SharedContext* GetContext() const; 
  void DefocusAllInterfaces(); 
  void HandleClick(EventDetails* l_details); 
  void HandleRelease(EventDetails* l_details); 
  void HandleTextEntered(EventDetails* l_details); 
  void AddEvent(GUI_Event l_event); 
  bool PollEvent(GUI_Event& l_event); 
  void Update(float l_dT); 
  void Render(sf::RenderWindow* l_wind); 
  template<class T> 
  void RegisterElement(const GUI_ElementType& l_id){ ... } 
private: 
  ... 
}; 

```

接口管理，不出所料，也依赖于应用程序状态。接口本身也被分配了名称，这就是它们被加载和存储的方式。鼠标输入以及文本输入事件都被用于使 GUI 系统工作，这就是为什么这个类实际上使用了事件管理器，并与之注册了三个回调。与其他我们讨论过的类一样，它也使用工厂方法，以便能够动态创建填充我们接口的不同类型的元素。

接口被描述为元素组，如下所示：

```cpp
Interface MainMenu MainMenu.style 0 0 Immovable NoTitle "Main menu" 
Element Label Title 100 0 MainMenuTitle.style "Main menu:" 
Element Label Play 0 32 MainMenuLabel.style "PLAY" 
Element Label Credits 0 68 MainMenuLabel.style "CREDITS" 
Element Label Quit 0 104 MainMenuLabel.style "EXIT" 

```

每个元素也支持它可能处于的三种不同状态的风格：中性、悬停和点击。一个单独的样式文件描述了元素在这些所有条件下的外观：

```cpp
State Neutral 
Size 300 32 
BgColor 255 0 0 255 
TextColor 255 255 255 255 
TextSize 14 
Font Main 
TextPadding 150 16 
TextOriginCenter 
/State 

State Hover 
BgColor 255 100 0 255 
/State 

State Clicked 
BgColor 255 150 0 255 
/State 

```

`Neutral`样式是其他两种样式的基础，这就是为什么它们只定义与它不同的属性。使用这种模型，可以构建和定制具有高度复杂性的接口，几乎可以完成任何事情。

# 表示 2D 地图

地图是拥有一个复杂游戏的关键部分之一。就我们的目的而言，我们将表示支持不同层级的 2D 地图，以模拟 3D 深度：

```cpp
class Map : public FileLoader{ 
public: 
  ... 
  Tile* GetTile(unsigned int l_x, unsigned int l_y, 
    unsigned int l_layer); 
  TileInfo* GetDefaultTile(); 
  TileSet* GetTileSet(); 
  unsigned int GetTileSize()const; 
  sf::Vector2u GetMapSize()const; 
  sf::Vector2f GetPlayerStart()const; 
  int GetPlayerId()const; 
  void PurgeMap(); 
  void AddLoadee(MapLoadee* l_loadee); 
  void RemoveLoadee(MapLoadee* l_loadee); 
  void Update(float l_dT); 
  void Draw(unsigned int l_layer); 
protected: 
  bool ProcessLine(std::stringstream& l_stream); 
  ... 
}; 

```

如您所见，这个类实际上是从我们之前提到的`FileLoader`继承的。它还支持一种称为`MapLoadee*`的功能，这些只是将某些数据存储在地图文件中的类，并在加载过程中遇到此类数据时需要通知。它只是一个它们必须实现的接口：

```cpp
class MapLoadee { 
public: 
  virtual void ReadMapLine(const std::string& l_type, 
    std::stringstream& l_stream) = 0; 
}; 

```

地图文件本身相当简单：

```cpp
SIZE 64 64 
DEFAULT_FRICTION 1.0 1.0 
|ENTITY|Name|x|y|elevation| 
ENTITY Player 715 360 1 
ENTITY Skeleton 256.0 768.0 1 
|TILE|ID|x|y|layer|solid| 
TILE 0 0 0 0 0 
TILE 0 0 1 0 0 
TILE 0 0 2 0 0 
... 

```

这里一个好的`MapLoadee`候选者是一个处理生成实体的类。两个实体行将直接由它处理，这为不应真正重叠的代码创建了一个很好的分离级别。

# 精灵系统

由于我们正在开发 2D 游戏，图形最有可能采用的方法是精灵图集。统一处理精灵图集裁剪和动画的方式不仅有助于最小化代码，还能创建一个简单、整洁且易于交互的接口。让我们看看如何实现这一点：

```cpp
class SpriteSheet{ 
public: 
  ... 
  void CropSprite(const sf::IntRect& l_rect); 
  const sf::Vector2u& GetSpriteSize()const; 
  const sf::Vector2f& GetSpritePosition()const; 
  void SetSpriteSize(const sf::Vector2u& l_size); 
  void SetSpritePosition(const sf::Vector2f& l_pos); 
  void SetDirection(const Direction& l_dir); 
  Direction GetDirection() const; 
  void SetSheetPadding(const sf::Vector2f& l_padding); 
  void SetSpriteSpacing(const sf::Vector2f& l_spacing); 
  const sf::Vector2f& GetSheetPadding()const; 
  const sf::Vector2f& GetSpriteSpacing()const; 
  bool LoadSheet(const std::string& l_file); 
  void ReleaseSheet(); 
  Anim_Base* GetCurrentAnim(); 
  bool SetAnimation(const std::string& l_name, 
    bool l_play = false, bool l_loop = false); 
  void Update(float l_dT); 
  void Draw(sf::RenderWindow* l_wnd); 
private: 
  ... 
  Animations m_animations; 
}; 

```

`SpriteSheet`类本身并不复杂。它提供了将图集裁剪到特定矩形、更改存储方向、定义不同的属性（如间距、填充等）以及操作动画数据的方法。

动画通过名称存储在这个类中：

```cpp
using Animations = std::unordered_map<std::string, 
  std::unique_ptr<Anim_Base>>; 

```

动画类的接口看起来是这样的：

```cpp
class Anim_Base{ 
  friend class SpriteSheet; 
public: 
  ... 
  void SetSpriteSheet(SpriteSheet* l_sheet); 
  bool SetFrame(Frame l_frame); 
  void SetStartFrame(Frame l_frame); 
  void SetEndFrame(Frame l_frame); 
  void SetFrameRow(unsigned int l_row); 
  void SetActionStart(Frame l_frame); 
  void SetActionEnd(Frame l_frame); 
  void SetFrameTime(float l_time); 
  void SetLooping(bool l_loop); 
  void SetName(const std::string& l_name); 
  SpriteSheet* GetSpriteSheet(); 
  Frame GetFrame() const; 
  Frame GetStartFrame() const; 
  Frame GetEndFrame() const; 
  unsigned int GetFrameRow() const; 
  int GetActionStart() const; 
  int GetActionEnd() const; 
  float GetFrameTime() const; 
  float GetElapsedTime() const; 
  bool IsLooping() const; 
  bool IsPlaying() const; 
  bool IsInAction() const; 
  bool CheckMoved(); 
  std::string GetName() const; 
  void Play(); 
  void Pause(); 
  void Stop(); 
  void Reset(); 
  virtual void Update(float l_dT); 
  friend std::stringstream& operator >>( 
    std::stringstream&l_stream, Anim_Base& a){ ... } 
protected: 
  virtual void FrameStep() = 0; 
  virtual void CropSprite() = 0; 
  virtual void ReadIn(std::stringstream& l_stream) = 0; 
  ... 
}; 

```

首先，`Frame`数据类型只是一个整型的类型定义。这个类跟踪所有必要的动画数据，并提供了一种设置特定帧范围（也称为动作）的方法，这可以用于实体仅在攻击动画位于该特定动作范围内时攻击某物。

这个类明显的一点是它不代表任何单一类型的动画，而是代表每种类型动画的共同元素。这就是为什么提供了三个不同的纯虚方法，以便不同类型的动画可以定义如何处理帧步进、定义特定方法、裁剪位置以及从文件中加载动画的确切过程。这有助于我们区分方向动画，其中每一行代表一个朝向不同方向的字符，以及简单的、按线性顺序跟随的帧序列动画。

# 音响系统

最后，但绝对不是最不重要的，音响系统值得简要概述。在这个阶段，了解到声音同样依赖于应用程序状态，可能对任何人来说都不会感到惊讶，这就是为什么我们再次从`StateDependent`继承：

```cpp
class SoundManager : public StateDependent{ 
public: 
  SoundManager(AudioManager* l_audioMgr); 
  ~SoundManager(); 

  void ChangeState(const StateType& l_state); 
  void RemoveState(const StateType& l_state); 

  void Update(float l_dT); 

  SoundID Play(const std::string& l_sound, 
    const sf::Vector3f& l_position, 
    bool l_loop = false, bool l_relative = false); 
  bool Play(const SoundID& l_id); 
  bool Stop(const SoundID& l_id); 
  bool Pause(const SoundID& l_id); 

  bool PlayMusic(const std::string& l_musicId, 
    float l_volume = 100.f, bool l_loop = false); 
  bool PlayMusic(const StateType& l_state); 
  bool StopMusic(const StateType& l_state); 
  bool PauseMusic(const StateType& l_state); 

  bool SetPosition(const SoundID& l_id, 
    const sf::Vector3f& l_pos); 
  bool IsPlaying(const SoundID& l_id) const; 
  SoundProps* GetSoundProperties(const std::string& l_soundName); 

  static const int Max_Sounds = 150; 
  static const int Sound_Cache = 75; 
private: 
  ... 
  AudioManager* m_audioManager; 
}; 

```

`AudioManager` 类负责管理音频资源，就像纹理和字体在其他地方被管理一样。这里的一个较大差异是，我们实际上可以在 3D 空间中播放声音，因此当需要表示位置时，会使用 `sf::Vector3f` 结构。声音也可以按特定名称分组，但这个系统有一个小小的转折。SFML 只能同时处理大约 255 个不同的声音，这包括`sf::Music`实例。正因为如此，我们必须实现一个回收系统，该系统利用被丢弃的声音实例，以及一次允许的最大声音数量的静态限制。

每个加载和播放的不同声音都有特定的设置属性可以调整。它们由以下数据结构表示：

```cpp
struct SoundProps{ 
  SoundProps(const std::string& l_name): m_audioName(l_name), 
    m_volume(100), m_pitch(1.f), m_minDistance(10.f), 
    m_attenuation(10.f){} 
  std::string m_audioName; 
  float m_volume; 
  float m_pitch; 
  float m_minDistance; 
  float m_attenuation; 
}; 

```

`audioName` 只是加载到内存中的音频资源的标识符。声音的音量显然可以调整，以及其音调。最后两个属性稍微复杂一些。在空间中的某个点，声音会开始变得越来越小，因为我们开始远离它。最小距离属性描述了从声音源的单位距离，在此距离之后，声音开始失去音量。达到该点后音量损失的速度由衰减因子描述。

# 摘要

这里的信息量相当大。在约四十页的篇幅内，我们成功总结了整个代码库的大部分内容，这些内容足以让任何基础到中级复杂度的游戏运行。请记住，尽管这里涵盖了众多主题，但所有信息都相当精炼。请随意查阅我们提供的代码文件，直到您感到舒适地继续实际构建游戏，这正是下一章将要介绍的内容。我们那里见！
