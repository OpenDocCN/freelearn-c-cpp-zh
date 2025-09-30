# 第七章. 重拾火焰 – 常见游戏设计元素

视频游戏每天都在变得越来越复杂。似乎创新的想法正在兴起，尤其是在独立游戏（如 *Minecraft* 和 *Super Meat Boy*）越来越受欢迎的背景下。尽管游戏想法本身变得越来越抽象，至少在外观上，支撑着美丽皮肤并帮助其保持形状的刚性骨架仍然是游戏开发者眼中最低的共同点。即使游戏的焦点围绕着两只在空闲时间吸食仙女粉末并帮助德古拉制作松饼，以免海王星爆炸的独角兽，这个概念能否实现将极大地取决于游戏背后的底层逻辑。如果没有实体，就没有独角兽。如果实体只是在黑色屏幕上弹跳，游戏就不会吸引人。这些都是任何项目都必须能够依赖的常见游戏设计元素，否则它注定会失败。

在本章中，我们将涵盖以下内容：

+   设计和实现游戏地图类

+   通过创建和管理实体来填充地图

+   检查和处理碰撞

+   将所有代码合并成一个完整游戏

# 游戏地图

玩家实际探索的环境和周围环境与游戏的其他部分一样重要。如果没有世界存在，玩家就只能在一个空白的屏幕颜色中空转。设计一个良好的界面来展示游戏的不同部分，从关卡背景到玩家必须面对的众多危险，可能会很棘手。现在，让我们为这个坚实的基础打下基础，从定义我们的地图格式开始，同时展望我们想要实现的目标：

![游戏地图](img/B04284_07_01.jpg)

首先，我们想要指定一个纹理句柄作为背景。然后，我们想要明确定义地图大小并设置重力，这决定了实体落地的速度。此外，我们还需要存储默认的摩擦力，这决定了平均地砖有多滑。最后，我们想要存储的是当当前地图结束时将加载的下一个地图的名称。以下是我们将要工作的其中一个地图的片段，`Map1.map`：

```cpp
|type|~id|x|y|
BACKGROUND Bg1
SIZE 63 32
GRAVITY 512
DEFAULT_FRICTION 0.8 0
NEXTMAP map2.map
|PLAYER 0 512
|ENEMY Rat 128 512
TILE 0 0 25
TILE 1 0 26 WARP
...
```

如您所见，除了定义所讨论的所有内容外，地图文件还存储了玩家位置，以及不同的敌人和它们的出生位置。其中最后但绝对不是最不重要的部分是地砖存储以及指示哪个地砖在被触摸时会将玩家“传送”到下一个阶段。

## 地砖是什么？

“瓦片”这个词经常被提及，但还没有被定义。简单来说，瓦片是构成世界众多部分之一。瓦片是创建游戏环境的块，无论是你站立的草地还是你掉落的刺。地图使用瓦片图集，这与精灵图集非常相似，因为它一次可以持有许多不同的精灵。主要区别在于如何从瓦片图集中获取这些精灵。在我们的案例中，将要用作瓦片图集的纹理如下所示：

![什么是瓦片？](img/B04284_07_02.jpg)

每个瓦片还具有独特的属性，我们希望从`Tiles.cfg`文件中加载这些属性：

```cpp
|id|name|friction x|friction y|deadly
0 Grass 0.8 0 0
1 Dirt 0.8 0 0
2 Stone 0.8 0 0
3 Brick 0.8 0 0
4 Brick_Red 0.8 0 0
5 Rock 0.8 0 0
6 Icy_Rock 0.6 0 0
7 Spikes 1.0 0 1
8 Ice 0.25 0 0
```

它非常简单，只包含瓦片 ID、名称、两个摩擦轴和一个表示瓦片是否致命的二元标志。

# 构建游戏世界

由于瓦片在我们的游戏设计中将扮演如此重要的角色，因此拥有一个独立的数据结构，其中所有瓦片信息都可以本地化，将非常有帮助。一个不错的起点是定义一些瓦片大小的常量，以及将要使用的瓦片图集的尺寸。在存储此类信息时，一个简单的枚举可以非常有帮助：

```cpp
enum Sheet{Tile_Size = 32, Sheet_Width = 256, Sheet_Height = 256};
```

在这里，我们使所有瓦片都宽 32 px，高 32 px，并且每个瓦片图集都宽 256 px，高 256 px。显然，这些常量可以更改，但这里的想法是在运行时保持它们相同。

为了使我们的代码更短，我们还可以从类型别名中受益，用于瓦片 ID：

```cpp
using TileID = unsigned int;
```

## 飞行员模式

显然，每个瓦片都必须有一个代表其类型的精灵。从图形上讲，为了绘制草瓦片，我们希望调整精灵以仅裁剪到瓦片图集中的草瓦片。然后，我们设置其在屏幕上的位置并绘制它。看起来很简单，但考虑以下情况：你有一个大小为 1000x1000 瓦片的地图，其中可能有 25%的地图大小是实际瓦片，而不是空气，这让你有总共 62,500 个瓦片需要绘制。现在想象一下，你为每个瓦片存储一个精灵。当然，精灵是轻量级对象，但这仍然是一种巨大的资源浪费。这就是飞行员模式发挥作用的地方。

存储大量冗余数据显然是浪费，为什么不只存储每种类型的一个实例，并在瓦片中简单地存储对类型的指针呢？简而言之，这就是飞行员模式。让我们通过实现瓦片信息结构来观察它的实际应用：

```cpp
struct TileInfo{
  TileInfo(SharedContext* l_context, 
    const std::string& l_texture = "", TileID l_id = 0)
    : m_context(l_context), m_id(0), m_deadly(false)
  {
    TextureManager* tmgr = l_context->m_textureManager;
    if (l_texture == ""){ m_id = l_id; return; }
    if (!tmgr->RequireResource(l_texture)){ return; }
    m_texture = l_texture;
    m_id = l_id;
    m_sprite.setTexture(*tmgr->GetResource(m_texture));
    sf::IntRect tileBoundaries(m_id %
     (Sheet::Sheet_Width / Sheet::Tile_Size) * Sheet::Tile_Size,
     m_id/(Sheet::Sheet_Height/Sheet::Tile_Size)*Sheet::Tile_Size,
     Sheet::Tile_Size,Sheet::Tile_Size);
    m_sprite.setTextureRect(tileBoundaries);
  }

  ~TileInfo(){
    if (m_texture == ""){ return; }
    m_context->m_textureManager->ReleaseResource(m_texture);
  }

  sf::Sprite m_sprite;

  TileID m_id;
  std::string m_name;
  sf::Vector2f m_friction;
  bool m_deadly;

  SharedContext* m_context;
  std::string m_texture;
};
```

这个 `struct` 实际上包含了关于每种瓦片类型所有非唯一信息。它存储了它所使用的纹理，以及将代表瓦片的精灵。正如你所见，在这个结构的构造函数中，我们将精灵设置为指向瓦片图纹理，然后根据其瓦片 ID 进行裁剪。这种裁剪与精灵图类中的裁剪略有不同，因为我们现在只有瓦片 ID 可用，并不知道精灵位于哪一行。使用一些基本的数学知识，我们可以首先通过将我们的图尺寸除以瓦片大小来计算出瓦片图有多少列和行。在这种情况下，一个 256x256 像素的精灵图，瓦片大小为 32x32 像素，每行和每列将有 8 个瓦片。通过使用取模运算符 `%` 可以获得瓦片 ID 在 *x* 轴上的坐标。在每行有 8 个瓦片的情况下，它将返回从 0 到 7 的值，基于 ID。确定 *y* 坐标是通过将 ID 除以每列的瓦片数来完成的。这给了我们瓦片精灵在瓦片图中的左上角坐标，所以我们通过传递 `Sheet::Tile_Size` 来完成裁剪。

`TileInfo` 析构函数仅释放用于瓦片图的纹理。在这个结构体中存储的其他值将在地图加载时初始化。现在让我们定义我们的瓦片结构：

```cpp
struct Tile{
    TileInfo* m_properties;
    bool m_warp; // Is the tile a warp.
    // Other flags unique to each tile.
};
```

这就是为什么享乐模式如此强大的原因。如果瓦片对象只存储每个瓦片唯一的信息，而不是瓦片类型，那么它们将非常轻量级。到目前为止，我们唯一感兴趣的是瓦片是否是传送门，这意味着当玩家站在上面时，它会加载下一级。

## 设计地图类

在处理完瓦片之后，我们可以继续处理更高级的结构，例如游戏地图。让我们首先创建一些合适的容器类型，这些容器将包含地图信息以及瓦片类型信息：

```cpp
using TileMap = std::unordered_map<TileID,Tile*>;
using TileSet = std::unordered_map<TileID,TileInfo*>;
```

`TileMap` 类型是一个 `unordered_map` 容器，它包含指向 `Tile` 对象的指针，这些对象通过无符号整数进行寻址。

### 注意

在已知瓦片数量预定的情形下，使用不会改变大小的容器（例如 `std::array` 或预分配的 `std::vector`）是明智的，以实现连续存储，从而实现更快的访问。

但等等！我们不是在二维空间中工作吗？如果坐标由两个数字表示，我们如何将瓦片映射到只有一个整数上呢？好吧，通过一点数学知识，完全可以将两个维度的索引表示为一个单一的数字。这将在稍后进行说明。

`TileSet` 数据类型代表所有不同类型瓦片的容器，这些瓦片与一个由无符号整数表示的瓦片 ID 相关联。这为我们编写地图头文件提供了所有需要的信息，这个头文件可能看起来像这样：

```cpp
class Map{
public:
  Map(SharedContext* l_context, BaseState* l_currentState);
  ~Map();
  Tile* GetTile(unsigned int l_x, unsigned int l_y);
  TileInfo* GetDefaultTile();
  float GetGravity()const;
  unsigned int GetTileSize()const;
  const sf::Vector2u& GetMapSize()const;
  const sf::Vector2f& GetPlayerStart()const;
  void LoadMap(const std::string& l_path);
  void LoadNext();
  void Update(float l_dT);
  void Draw();
private:
  // Method for converting 2D coordinates to 1D ints.
  unsigned int ConvertCoords(unsigned int l_x, unsigned int l_y);
  void LoadTiles(const std::string& l_path);
  void PurgeMap();
  void PurgeTileSet();

  TileSet m_tileSet;
  TileMap m_tileMap;
  sf::Sprite m_background;
  TileInfo m_defaultTile;
  sf::Vector2u m_maxMapSize;
  sf::Vector2f m_playerStart;
  unsigned int m_tileCount;
  unsigned int m_tileSetCount;
  float m_mapGravity;
  std::string m_nextMap;
  bool m_loadNextMap;
  std::string m_backgroundTexture;
  BaseState* m_currentState;
  SharedContext* m_context;
};
```

首先，我们定义所有可预测的方法，例如在特定坐标获取瓦片、从类中获取各种信息，以及当然，更新和绘制地图的方法。让我们继续实现这些方法，以便更深入地讨论它们：

```cpp
Map::Map(SharedContext* l_context, BaseState* l_currentState)
    :m_context(l_context), m_defaultTile(l_context), m_maxMapSize(32, 32), m_tileCount(0), m_tileSetCount(0),m_mapGravity(512.f), m_loadNextMap(false),m_currentState(l_currentState)
{
    m_context->m_gameMap = this;
    LoadTiles("tiles.cfg");
}
```

地图构造函数将其数据成员初始化为一些默认值，并调用一个私有方法以从`tiles.cfg`文件加载不同类型的瓦片。相当标准。足够可预测，这个类的析构函数也没有做任何特别的事情：

```cpp
Map::~Map(){
    PurgeMap();
    PurgeTileSet();
    m_context->m_gameMap = nullptr;
}
```

从地图中获取瓦片是通过首先将此方法提供的作为参数的 2D 坐标转换为单个数字，然后在无序映射中定位特定的瓦片：

```cpp
Tile* Map::GetTile(unsigned int l_x, unsigned int l_y){
  auto itr = m_tileMap.find(ConvertCoords(l_x,l_y));
  return(itr != m_tileMap.end() ? itr->second : nullptr);
}
```

坐标转换看起来是这样的：

```cpp
unsigned int Map::ConvertCoords(const unsigned int& l_x, const unsigned int& l_y)
{
    return (l_x * m_maxMapSize.x) + l_y; // Row-major.
}
```

为了使这个方法工作，我们必须定义地图的最大尺寸，否则它会产生错误的结果。

更新地图是另一个关键部分：

```cpp
void Map::Update(float l_dT){
  if(m_loadNextMap){
    PurgeMap();
    m_loadNextMap = false;
    if(m_nextMap != ""){
      LoadMap("media/maps/"+m_nextMap);
    } else {
      m_currentState->GetStateManager()->
        SwitchTo(StateType::GameOver);
    }
    m_nextMap = "";
  }
  sf::FloatRect viewSpace = m_context->m_wind->GetViewSpace();
  m_background.setPosition(viewSpace.left, viewSpace.top);
}
```

这里，它检查`m_loadNextMap`标志。如果它设置为`true`，则清除地图信息并加载下一个地图，如果设置了持有其句柄的数据成员。如果没有设置，则将应用程序状态设置为`GameOver`，这将在稍后创建。这将模拟玩家通关游戏。最后，我们获取窗口的视图空间并将地图背景的左上角设置为视图空间的左角，以便背景跟随相机。让我们在屏幕上绘制这些更改：

```cpp
void Map::Draw(){
  sf::RenderWindow* l_wind = m_context->m_wind->GetRenderWindow();
  l_wind->draw(m_background);
  sf::FloatRect viewSpace = m_context->m_wind->GetViewSpace();

  sf::Vector2i tileBegin(
    floor(viewSpace.left / Sheet::Tile_Size),
    floor(viewSpace.top / Sheet::Tile_Size));
  sf::Vector2i tileEnd(
    ceil((viewSpace.left + viewSpace.width) / Sheet::Tile_Size),
    ceil((viewSpace.top + viewSpace.height) / Sheet::Tile_Size));

  unsigned int count = 0;
  for(int x = tileBegin.x; x <= tileEnd.x; ++x){
    for(int y = tileBegin.y; y <= tileEnd.y; ++y){
      if(x < 0 || y < 0){ continue; }
      Tile* tile = GetTile(x,y);
      if (!tile){ continue; }
      sf::Sprite& sprite = tile->m_properties->m_sprite;
      sprite.setPosition(x * Sheet::Tile_Size,
        y * Sheet::Tile_Size);
      l_wind->draw(sprite);
      ++count;
    }
  }
}
```

通过共享上下文获取渲染窗口的指针，并在前两行中绘制背景。接下来的三行有一个简单的名字，称为剔除。这是一种任何优秀的游戏程序员都应该利用的技术，其中任何当前不在屏幕视图空间内的东西都应该不被绘制。再次考虑这种情况，你有一个 1000x1000 大小的巨大地图。尽管现代硬件现在可以非常快地绘制，但仍然没有必要浪费这些时钟周期，当它们可以用来执行更好的任务时，而不是将一些甚至不可见的东西带到屏幕上。如果你在游戏中没有进行任何剔除，它最终将开始遭受严重的性能打击。

从视图空间的左上角到右下角的瓦片坐标被输入到一个循环中。首先，它们被评估为正数。如果它们是负数，我们计算地图容器 1D 索引的方式会产生一些镜像伪影，如果你向上或向左走得太远，你将看到相同的地图反复出现。

通过传递循环中的*x*和*y*坐标，我们获得一个瓦片的指针。如果它是一个有效的瓦片，我们就从`TileInfo`结构的指针中获取其精灵。精灵的位置被设置为与瓦片的坐标匹配，并在屏幕上绘制精灵。

现在有一个方法可以擦除整个地图：

```cpp
void Map::PurgeMap(){
  m_tileCount = 0;
  for (auto &itr : m_tileMap){
    delete itr.second;
  }
  m_tileMap.clear();
  m_context->m_entityManager->Purge();

  if (m_backgroundTexture == ""){ return; }
  m_context->m_textureManager->ReleaseResource(m_backgroundTexture);
  m_backgroundTexture = "";
}
```

除了清除地图容器外，你还会注意到我们在调用实体管理器的`Purge`方法。现在先忽略那行。实体将在稍后讨论。我们也不应忘记在擦除地图时释放背景纹理。

清空不同瓦片类型的容器也是必要的部分：

```cpp
void Map::PurgeTileSet(){
  for (auto &itr : m_tileSet){
    delete itr.second;
  }
  m_tileSet.clear();
  m_tileSetCount = 0;
}
```

这部分很可能会在析构函数中被调用，但有一个单独的方法还是不错的。说到不同的瓦片类型，我们需要从文件中加载它们：

```cpp
void Map::LoadTiles(const std::string& l_path){
  std::ifstream file;
  file.open(Utils::GetWorkingDirectory() + l_path);
  if (!file.is_open()){
    std::cout << "! Failed loading tile set file: "<< l_path << std::endl;
    return;
  }
  std::string line;
  while(std::getline(file,line)){
    if (line[0] == '|'){ continue; }
    std::stringstream keystream(line);
    int tileId;
    keystream >> tileId;
    if (tileId < 0){ continue; }
    TileInfo* tile = new TileInfo(m_context,"TileSheet",tileId);
    keystream >> tile->m_name >> tile->m_friction.x >> tile->m_friction.y >> tile->m_deadly;
    if(!m_tileSet.emplace(tileId,tile).second){
      // Duplicate tile detected!
      std::cout << "! Duplicate tile type: "<< tile->m_name << std::endl;
      delete tile;
    }
  }
  file.close();
}
```

首先加载瓦片 ID，正如`tiles.cfg`格式所建议的。它被检查是否越界，如果不是，就会为瓦片类型分配动态内存，此时所有内部数据成员都被初始化为字符串流中的值。如果瓦片信息对象无法插入到瓦片集容器中，那么必须有重复条目，此时动态内存将被释放。

现在是地图的压轴大戏——加载方法。由于实际的文件加载代码基本上保持不变，让我们直接跳到读取地图文件的内容，从瓦片条目开始：

```cpp
if(type == "TILE"){
  int tileId = 0;
  keystream >> tileId;
  if (tileId < 0){ std::cout << "! Bad tile id: " << tileId << std::endl;
    continue;
  }
  auto itr = m_tileSet.find(tileId);
  if (itr == m_tileSet.end()){
    std::cout << "! Tile id(" << tileId<< ") was not found in tileset." << std::endl;
    continue;
  }
  sf::Vector2i tileCoords;
  keystream >> tileCoords.x >> tileCoords.y;
  if (tileCoords.x>m_maxMapSize.x || tileCoords.y>m_maxMapSize.y)
  {
    std::cout << "! Tile is out of range: " <<tileCoords.x << " " << tileCoords.y << std::endl;
    continue;
  }
  Tile* tile = new Tile();
  // Bind properties of a tile from a set.
  tile->m_properties = itr->second;
  if(!m_tileMap.emplace(ConvertCoords(
    tileCoords.x,tileCoords.y),tile).second)
  {
    // Duplicate tile detected!
    std::cout << "! Duplicate tile! : " << tileCoords.x << "" << tileCoords.y << std::endl;
    delete tile;
    tile = nullptr;
    continue;
  }
  std::string warp;
  keystream >> warp;
  tile->m_warp = false;
  if(warp == "WARP"){ tile->m_warp = true; }
} else if ...
```

`TILE`行的第一部分被加载进来，这是瓦片 ID。按照惯例，它被检查是否在正数和*0*的范围内。如果是，就会在瓦片集中查找该特定瓦片 ID 的瓦片信息。因为我们不希望地图周围有空白的瓦片，所以我们只有在找到特定 ID 的瓦片信息时才会继续进行。接下来，读取瓦片坐标并检查它们是否在地图大小的范围内。如果是，就会为瓦片分配内存，并将其瓦片信息数据成员设置为指向瓦片集中的那个。最后，我们尝试读取`TILE`行末尾的字符串并检查它是否说“WARP”。这是指接触特定瓦片应该加载下一级。

现在来说说地图的背景：

```cpp
} else if(type == "BACKGROUND"){
  if (m_backgroundTexture != ""){ continue; }
  keystream >> m_backgroundTexture;
  if (!m_context->m_textureManager->RequireResource(m_backgroundTexture))
  {
    m_backgroundTexture = "";
    continue;
  }
  sf::Texture* texture = m_context->m_textureManager->GetResource(m_backgroundTexture);
  m_background.setTexture(*texture);
  sf::Vector2f viewSize = m_currentState->GetView().getSize();
  sf::Vector2u textureSize = texture->getSize();
  sf::Vector2f scaleFactors;
  scaleFactors.x = viewSize.x / textureSize.x;
  scaleFactors.y = viewSize.y / textureSize.y;
  m_background.setScale(scaleFactors);
} else if ...
```

这部分相当直接。从`BACKGROUND`行加载一个纹理句柄。如果句柄有效，背景精灵就会与纹理绑定。但是有一个问题。假设我们窗口的视图比背景纹理大。这会导致背景周围出现空白区域，看起来非常糟糕。重复纹理可能会解决空白区域的问题，但我们将要处理的特定背景并不适合平铺，所以最好的解决方案是将精灵缩放到足够大，以完全适应视图空间，无论它更大还是更小。缩放因子的值可以通过将视图大小乘以纹理大小来获得。例如，如果我们有一个 800x600 像素大小的视图和一个 400x300 像素大小的纹理，两个轴的缩放因子都是 2，背景被放大到原来的两倍大小。

接下来是简单地从文件中读取一些数据成员的部分：

```cpp
} else if(type == "SIZE"){
    keystream >> m_maxMapSize.x >> m_maxMapSize.y;
} else if(type == "GRAVITY"){
    keystream >> m_mapGravity;
} else if(type == "DEFAULT_FRICTION"){
    keystream >> m_defaultTile->m_friction.x >> m_defaultTile->m_friction.y;
} else if(type == "NEXTMAP"){
    keystream >> m_nextMap;
}
```

让我们用一个小助手方法来结束这个类，这个方法将帮助我们跟踪下一个地图何时应该被加载：

```cpp
void Map::LoadNext(){ m_loadNextMap = true; }
```

这就完成了地图类（map class）的实现。现在世界已经存在，但没有人去占据它。真是荒谬！我们不要贬低我们的工作，让我们创建一些实体来探索我们创造的环境。

# 所有世界对象的父类

实体实际上只是游戏对象（game object）的另一种说法。它是一个抽象类，作为所有其派生类的父类，包括玩家、敌人和可能的项目，具体取决于你如何实现。让这些完全不同的概念共享相同的根源，允许程序员定义适用于所有这些的共同行为类型。此外，它还允许游戏引擎以相同的方式对它们进行操作，因为它们都共享相同的接口。例如，敌人可以被推动，玩家也可以。所有敌人、项目和玩家都必须受到重力的影响。这些不同类型之间的共同血统使我们能够卸载大量冗余代码，并专注于每个实体的独特方面，而不是一遍又一遍地重写相同的代码。

让我们先定义我们将要处理哪些实体类型：

```cpp
enum class EntityType{ Base, Enemy, Player };
```

基类实体类型只是一个抽象类，实际上并不会被实例化。这让我们有了敌人和玩家。现在，让我们设置实体可能拥有的所有可能状态：

```cpp
enum class EntityState{
    Idle, Walking, Jumping, Attacking, Hurt, Dying
};
```

你可能已经注意到，这些状态与玩家精灵图（sprite sheet）中的动画大致相符。所有角色实体都将以此方式建模。

## 创建基类实体

在使用继承构建实体的情况下，编写这样一个基本父类（parent class）相当常见。它必须提供任何给定游戏内实体应有的所有功能。

在完成所有设置之后，我们终于可以开始这样塑造它：

```cpp
class EntityManager;
class EntityBase{
friend class EntityManager;
public:
  EntityBase(EntityManager* l_entityMgr);
  virtual ~EntityBase();
  ... // Getters and setters.
  void Move(float l_x, float l_y);
  void AddVelocity(float l_x, float l_y);
  void Accelerate(float l_x, float l_y);
  void SetAcceleration(float l_x, float l_y);
  void ApplyFriction(float l_x, float l_y);
  virtual void Update(float l_dT);
  virtual void Draw(sf::RenderWindow* l_wind) = 0;
protected:
  // Methods.
  void UpdateAABB();
  void CheckCollisions();
  void ResolveCollisions();
  // Method for what THIS entity does TO the l_collider entity.
  virtual void OnEntityCollision(EntityBase* l_collider,bool l_attack) = 0;
  // Data members.
  std::string m_name;
  EntityType m_type;
  unsigned int m_id; // Entity id in the entity manager.
  sf::Vector2f m_position; // Current position.
  sf::Vector2f m_positionOld; // Position before entity moved.
  sf::Vector2f m_velocity; // Current velocity.
  sf::Vector2f m_maxVelocity; // Maximum velocity.
  sf::Vector2f m_speed; // Value of acceleration.
  sf::Vector2f m_acceleration; // Current acceleration.
  sf::Vector2f m_friction; // Default friction value.
  TileInfo* m_referenceTile; // Tile underneath entity.
  sf::Vector2f m_size; // Size of the collision box.
  sf::FloatRect m_AABB; // The bounding box for collisions.
  EntityState m_state; // Current entity state.
  // Flags for remembering axis collisions.
  bool m_collidingOnX;
  bool m_collidingOnY;

  Collisions m_collisions;
  EntityManager* m_entityManager;
};
```

一开始，我们就将尚未编写的`EntityManager`类设置为基类实体（base entities）的朋友类（friend class）。因为代码可能有点令人困惑，所以添加了一大堆注释来解释类的每个数据成员，所以我们不会过多地涉及这些内容，直到我们在类的实现过程中遇到它们。

实体的三个主要属性包括其位置、速度和加速度。实体的位置是自解释的。速度表示实体移动的速度。由于我们应用程序中的所有更新方法都接受以秒为单位的 delta 时间，所以速度将表示实体每秒移动的像素数。三个主要属性中的最后一个元素是加速度，它负责实体速度增加的速度。它也被定义为每秒添加到实体速度的像素数。这里的事件序列如下：

1.  实体被加速，并且它的加速度调整其速度。

1.  实体的位置是根据其速度重新计算的。

1.  实体的速度会受到摩擦系数的阻尼。

### 碰撞和边界框

在深入实现之前，让我们谈谈所有游戏中最常用的元素之一——碰撞。检测和解决碰撞是防止玩家穿过地图或屏幕外部的关键。它还决定了玩家在受到敌人触碰时是否会受伤。以一种间接的方式，我们使用了一种基本的碰撞检测方法来确定在地图类中应该渲染哪些瓦片。如何检测和解决碰撞呢？有很多方法可以做到这一点，但就我们的目的而言，最基本的边界框碰撞就足够了。还可以使用其他类型的碰撞，例如圆形，但这可能取决于正在构建的游戏类型，可能不是最有效或最合适的方法。

边界框，正如其名，是一个盒子或矩形，代表实体的实体部分。以下是一个边界框的好例子：

![碰撞和边界框](img/B04284_07_03.jpg)

它不会像那样可见，除非我们创建一个与边界框具有相同位置和大小的实际 `sf::RectangleShape` 并进行渲染，这是一种调试应用程序的有用方法。在我们的基础实体类中，名为 `m_AABB` 的边界框只是一个 `sf::FloatRect` 类型。名称 "AABB" 代表它持有的两个不同值对：位置和大小。边界框碰撞，也称为 AABB 碰撞，简单来说就是两个边界框相互相交的情况。SFML 中的矩形数据类型为我们提供了一个检查交集的方法：

```cpp
if(m_AABB.intersects(SomeRectangle){...}
```

冲突解决这个术语简单来说就是执行一系列动作来通知并移动发生碰撞的实体。例如，在碰撞到瓦片的情况下，冲突解决意味着将实体推回足够远，使其不再与瓦片相交。

### 小贴士

本项目的代码文件包含一个额外的类，允许进行调试信息渲染，以及已经设置好的所有这些信息。按下 *O* 键可以切换其可见性。

## 实现基础实体类

在处理完所有这些信息之后，我们终于可以回到实现基础实体类的工作上了。一如既往，还有什么地方比构造函数更好的起点呢？让我们来看看：

```cpp
EntityBase::EntityBase(EntityManager* l_entityMgr)
  :m_entityManager(l_entityMgr), m_name("BaseEntity"),
  m_type(EntityType::Base), m_referenceTile(nullptr),
  m_state(EntityState::Idle), m_id(0),
  m_collidingOnX(false), m_collidingOnY(false){}
```

它只是将所有数据成员初始化为默认值。请注意，在它设置为零的所有成员中，摩擦系数实际上被设置为 *x* 轴为 0.8。这是因为我们不想实体的默认行为与冰上的牛一样，坦白说。摩擦系数定义了实体速度中有多少会损失到环境中。如果现在不太理解，不用担心。我们很快就会更详细地介绍它。

这里包含了修改实体基类数据成员的所有方法：

```cpp
void EntityBase::SetPosition(const float& l_x, const float& l_y){
    m_position = sf::Vector2f(l_x,l_y);
    UpdateAABB();
}
void EntityBase::SetPosition(const sf::Vector2f& l_pos){
    m_position = l_pos;
    UpdateAABB();
}
void EntityBase::SetSize(const float& l_x, const float& l_y){
    m_size = sf::Vector2f(l_x,l_y);
    UpdateAABB();
}
void EntityBase::SetState(const EntityState& l_state){
    if(m_state == EntityState::Dying){ return; }
    m_state = l_state;
}
```

如您所见，修改实体位置或大小都会调用内部方法 `UpdateAABB`。简单来说，它负责更新边界框的位置。更多相关信息即将揭晓。

有一个有趣的事情要注意的是在 `SetState` 方法中。如果当前状态是 `Dying`，则不允许状态改变。这样做是为了防止游戏中的某些其他事件神奇地将实体从死亡状态中拉出来。

现在我们有一个更有趣的代码块，负责移动实体：

```cpp
void EntityBase::Move(float l_x, float l_y){
  m_positionOld = m_position;
  m_position += sf::Vector2f(l_x,l_y);
  sf::Vector2u mapSize = m_entityManager->GetContext()->m_gameMap->GetMapSize();
  if(m_position.x < 0){
    m_position.x = 0;
  } else if(m_position.x > (mapSize.x + 1) * Sheet::Tile_Size){
    m_position.x = (mapSize.x + 1) * Sheet::Tile_Size;
  }

  if(m_position.y < 0){
    m_position.y = 0;
  } else if(m_position.y > (mapSize.y + 1) * Sheet::Tile_Size){
    m_position.y = (mapSize.y + 1) * Sheet::Tile_Size;
    SetState(EntityState::Dying);
  }

  UpdateAABB();
}
```

首先，我们将当前位置复制到另一个数据成员：`m_positionOld`。保留这些信息总是好的，以防以后需要。然后，位置通过提供的偏移量进行调整。之后，获取地图的大小，以便检查当前位置是否超出地图范围。如果它在任一轴上，我们只需将其位置重置为边界外的边缘。如果实体在 *y* 轴上超出地图范围，其状态被设置为 `Dying`。在所有这些之后，边界框被更新以反映实体精灵位置的变化。

现在我们来添加和管理实体的速度：

```cpp
void EntityBase::AddVelocity(float l_x, float l_y){
  m_velocity += sf::Vector2f(l_x,l_y);
  if(abs(m_velocity.x) > m_maxVelocity.x){
    if(m_velocity.x < 0){ m_velocity.x = -m_maxVelocity.x; }
    else { m_velocity.x = m_maxVelocity.x; }
  }

  if(abs(m_velocity.y) > m_maxVelocity.y){
    if(m_velocity.y < 0){ m_velocity.y = -m_maxVelocity.y; }
    else { m_velocity.y = m_maxVelocity.y; }
  }
}
```

如您所见，这相当简单。将速度成员添加到一起，然后检查它是否超出允许的最大速度范围。在第一次检查中，我们使用绝对值，因为速度可以是正的也可以是负的，这表示实体移动的方向。如果速度超出范围，则将其重置为允许的最大值。

说到加速实体，可以说，就像向另一个向量中添加一个向量一样简单：

```cpp
void EntityBase::Accelerate(float l_x, float l_y){
    m_acceleration += sf::Vector2f(l_x,l_y);
}
```

应用摩擦并不比管理我们的速度更复杂：

```cpp
void EntityBase::ApplyFriction(float l_x, float l_y){
  if(m_velocity.x != 0){
    if(abs(m_velocity.x) - abs(l_x) < 0){ m_velocity.x = 0; }
    else {
      if(m_velocity.x < 0){ m_velocity.x += l_x; }
      else { m_velocity.x -= l_x; }
    }
  }

  if(m_velocity.y != 0){
    if (abs(m_velocity.y) - abs(l_y) < 0){ m_velocity.y = 0; }
    else {
      if(m_velocity.y < 0){ m_velocity.y += l_y; }
      else { m_velocity.y -= l_y; }
    }
  }
}
```

它需要检查该轴上速度和摩擦系数绝对值之差是否不小于零，以防止通过摩擦改变实体运动方向，这会显得很奇怪。如果是小于零，则将速度设置回零。如果不是，则检查速度的符号并应用适当方向的摩擦。

为了使实体不是背景的静态部分，它需要被更新：

```cpp
void EntityBase::Update(float l_dT){
  Map* map = m_entityManager->GetContext()->m_gameMap;
  float gravity = map->GetGravity();
  Accelerate(0,gravity);
  AddVelocity(m_acceleration.x * l_dT, m_acceleration.y * l_dT);
  SetAcceleration(0.0f, 0.0f);
  sf::Vector2f frictionValue;
  if(m_referenceTile){
    frictionValue = m_referenceTile->m_friction;
    if(m_referenceTile->m_deadly){ SetState(EntityState::Dying); }
  } else if(map->GetDefaultTile()){
    frictionValue = map->GetDefaultTile()->m_friction;
  } else {
    frictionValue = m_friction;
  }

  float friction_x = (m_speed.x * frictionValue.x) * l_dT;
  float friction_y = (m_speed.y * frictionValue.y) * l_dT;
  ApplyFriction(friction_x, friction_y);
  sf::Vector2f deltaPos = m_velocity * l_dT;
  Move(deltaPos.x, deltaPos.y);
  m_collidingOnX = false;
  m_collidingOnY = false;
  CheckCollisions();
  ResolveCollisions();
}
```

这里发生了很多事情。让我们一步一步来。首先，通过共享上下文获取游戏地图的一个实例。然后使用它来获取地图的重力，该重力是从地图文件中加载的。然后，实体的加速度通过 *y* 轴的重力增加。通过使用 `AddVelocity` 方法并传递加速度乘以时间增量，调整速度并将加速度设置回零。接下来，我们必须获取速度将被阻尼的摩擦系数。如果 `m_referenceTile` 数据成员没有被设置为 `nullptr`，则首先使用它，以从实体所站的瓦片中获取摩擦力。如果它被设置为 `nullptr`，则实体必须在空中，因此从地图中获取默认瓦片以获取从地图文件中加载的摩擦值。如果由于任何原因，这也没有设置，则默认为在 `EntityBase` 构造函数中设置的值。

在计算摩擦力之前，重要的是要明确，除了被设置为默认值之外，`m_speed` 数据成员在这个类中既没有设置也没有初始化。速度是实体移动时加速的量，它将在 `EntityBase` 的一个派生类中实现。

如果你从该类的构造函数中回忆起来，我们设置了默认摩擦为 0.8f。这不仅仅是一个非常小的值。我们正在使用摩擦作为一个因素，以确定实体速度应该损失多少。话虽如此，将速度乘以摩擦系数，然后乘以时间增量，我们得到的是在此帧中损失的速度，然后将其传递到 `ApplyFriction` 方法中以操纵速度。

最后，位置的变化，称为 `deltaPos`，是通过将速度乘以时间增量来计算的，并将其传递到 `Move` 方法中以调整实体在世界中的位置。两个轴上的碰撞标志被重置为 false，实体调用其自己的私有成员以首先获取然后解决碰撞。

让我们看看负责更新边界框的方法：

```cpp
void EntityBase::UpdateAABB(){
  m_AABB = sf::FloatRect(m_position.x - (m_size.x / 2),m_position.y - m_size.y, m_size.x, m_size.y);
}
```

由于边界框的起点被留在左上角，并且实体的位置被设置为 (width / 2, height)，如果我们想要精确的碰撞，那么考虑这一点是必要的。代表边界框的矩形被重置以匹配精灵的新位置。

## 实体与瓦片碰撞

在跳入碰撞检测和解决之前，让我们回顾一下 SFML 提供的用于检查两个矩形是否相交的方法：

```cpp
sf::FloatRect r1;
sf::FloatRect r2;
if(r1.intersects(r2)){ ... }
```

无论我们检查哪个矩形，如果它们相交，交点方法仍然会返回 true。然而，此方法确实接受一个可选的第二个参数，它是一个矩形类的引用，该引用将填充有关交点本身的信息。考虑以下插图：

![实体与瓦片碰撞](img/B04284_07_04.jpg)

我们有两个相交的矩形。对角条纹区域表示交集的矩形，可以通过以下方式获得：

```cpp
...
sf::FloatRect intersection;
if(r1.intersects(r2,intersection)){ ... }
```

这对我们来说很重要，因为一个实体可能同时与多个瓦片发生碰撞。了解碰撞的深度也是解决碰撞的关键部分。考虑到这一点，让我们定义一个结构来临时存储碰撞信息，在它被解决之前：

```cpp
struct CollisionElement{
  CollisionElement(float l_area, TileInfo* l_info,const sf::FloatRect& l_bounds):m_area(l_area), m_tile(l_info), m_tileBounds(l_bounds){}
  float m_area;
  TileInfo* m_tile;
  sf::FloatRect m_tileBounds;
};

using Collisions = std::vector<CollisionElement>;
```

首先，我们创建一个结构，它包含一个表示碰撞面积的浮点数，一个包含实体碰撞的瓦片的边界信息的矩形，以及一个指向`TileInfo`实例的指针。你总是希望先解决最大的碰撞，这些信息将帮助我们做到这一点。这次，碰撞元素将存储在一个向量中。

接下来，我们需要一个函数来比较我们自定义容器中的两个元素以便对其进行排序，其蓝图在`EntityBase`类的头文件中看起来像这样：

```cpp
bool SortCollisions(const CollisionElement& l_1,const CollisionElement& l_2);
```

实现这个函数非常简单。向量容器简单地使用布尔检查来确定它正在比较的两个元素中哪一个更大。我们只需根据哪个元素更大返回 true 或 false。因为我们按面积大小对容器进行排序，所以比较是在第一对的第一元素之间进行的：

```cpp
bool SortCollisions(const CollisionElement& l_1,const CollisionElement& l_2)
{ return l_1.m_area > l_2.m_area; }
```

接下来是有趣的部分，检测碰撞：

```cpp
void EntityBase::CheckCollisions(){
  Map* gameMap = m_entityManager->GetContext()->m_gameMap;
  unsigned int tileSize = gameMap->GetTileSize();
  int fromX = floor(m_AABB.left / tileSize);
  int toX = floor((m_AABB.left + m_AABB.width) / tileSize);
  int fromY = floor(m_AABB.top / tileSize);
  int toY = floor((m_AABB.top + m_AABB.height) / tileSize);

  for(int x = fromX; x <= toX; ++x){
    for(int y = fromY; y <= toY; ++y){
      Tile* tile = gameMap->GetTile(x,y);
      if (!tile){ continue; }
      sf::FloatRect tileBounds(x * tileSize, y * tileSize,
        tileSize,tileSize);
      sf::FloatRect intersection;
      m_AABB.intersects(tileBounds,intersection);
      float area = intersection.width * intersection.height;

      CollisionElement e(area, tile->m_properties, tileBounds);
      m_collisions.emplace_back(e);
      if(tile->m_warp && m_type == EntityType::Player){
        gameMap->LoadNext();
      }
    }
  }
}
```

我们首先使用边界框的坐标和大小来获得它可能相交的瓦片的坐标。以下图像更好地说明了这一点：

![实体与瓦片碰撞](img/B04284_07_05.jpg)

然后将由四个整数表示的瓦片坐标范围输入到一个双重循环中，该循环检查是否有瓦片占据了我们所感兴趣的空间。如果`GetTile`方法返回一个瓦片，那么实体的边界框肯定与一个瓦片相交，因此创建一个表示瓦片边界框的浮点矩形。我们还准备另一个浮点矩形来保存交集的数据，并调用`intersects`方法以获取这些信息。交集的面积通过乘以其宽度和高度来计算，碰撞信息以及表示实体碰撞的瓦片类型的`TileInfo`对象的指针被推入碰撞容器中。

在结束这个方法之前，我们最后要检查实体正在碰撞的当前瓦片是否是扭曲瓦片，以及实体是否是玩家。如果这两个条件都满足，则加载下一个地图。

现在已经获得了一个实体的碰撞列表，下一步是解决它们：

```cpp
void EntityBase::ResolveCollisions(){
  if(!m_collisions.empty()){
    std::sort(m_collisions.begin(),m_collisions.end(), SortCollisions);
    Map* gameMap = m_entityManager->GetContext()->m_gameMap;
    unsigned int tileSize = gameMap->GetTileSize();
    for (auto &itr : m_collisions){
      if (!m_AABB.intersects(itr.m_tileBounds)){ continue; }
      float xDiff = (m_AABB.left + (m_AABB.width / 2)) -(itr.m_tileBounds.left + (itr.m_tileBounds.width / 2));
      float yDiff = (m_AABB.top + (m_AABB.height / 2)) -(itr.m_tileBounds.top + (itr.m_tileBounds.height / 2));
      float resolve = 0;
      if(abs(xDiff) > abs(yDiff)){
        if(xDiff > 0){
          resolve = (itr.m_tileBounds.left + tileSize) –m_AABB.left;
        } else {
          resolve = -((m_AABB.left + m_AABB.width) –itr.m_tileBounds.left);
        }
        Move(resolve, 0);
        m_velocity.x = 0;
        m_collidingOnX = true;
      } else {
        if(yDiff > 0){
          resolve = (itr.m_tileBounds.top + tileSize) –
            m_AABB.top;
        } else {
          resolve = - ((m_AABB.top + m_AABB.height) –itr.m_tileBounds.top);
        }
        Move(0,resolve);
        m_velocity.y = 0;
        if (m_collidingOnY){ continue; }
        m_referenceTile = itr.m_tile;
        m_collidingOnY = true;
      }
    }
    m_collisions.clear();
  }
  if(!m_collidingOnY){ m_referenceTile = nullptr; }
}
```

首先，我们检查容器中是否有任何碰撞。接下来是对所有元素进行排序。调用`std::sort`函数，并传入容器的开始和结束迭代器，以及将用于元素之间比较的函数名称。

代码接着遍历容器中存储的所有碰撞。这里在实体的边界框和瓦片的边界框之间还有一个交点检查。这样做是因为解决之前的碰撞可能会以某种方式移动实体，使其不再与容器中下一个瓦片发生碰撞。如果仍然存在碰撞，则计算实体边界框中心到瓦片边界框中心的距离。这些距离的第一个用途在下一条线中体现，其中它们的绝对值被比较。如果 x 轴上的距离大于 y 轴上的距离，则解决发生在 x 轴上。否则，它在 y 轴上解决。

距离计算的第二个目的是确定实体位于瓦片的哪一侧。如果距离为正，则实体位于瓦片的右侧，因此它向正 x 方向移动。否则，它向负 x 方向移动。`resolve`变量接受瓦片和实体之间的穿透量，这取决于轴和碰撞的侧面。

在两个轴的情况下，通过调用其实体的`Move`方法并传入穿透深度来移动实体。在模拟实体撞击固体时，停止该轴上的实体速度也很重要。最后，将特定轴上的碰撞标志设置为 true。

如果碰撞在 y 轴上解决，除了在 x 轴碰撞解决情况下采取的所有相同步骤外，我们还会检查是否设置了 y 轴碰撞标志。如果尚未设置，我们将`m_referenceTile`数据成员更改为指向实体正在与之碰撞的当前瓦片的瓦片类型，然后设置该标志为 true，以保持引用不变，直到下一次检查碰撞。这段小代码片段使任何实体根据其站立在哪个瓦片上而表现出不同的行为。例如，实体在冰瓦片上可以比在简单的草瓦片上滑动更多，如图所示：

![实体在瓦片上的碰撞](img/B04284_07_06.jpg)

如箭头所示，这些瓦片的摩擦系数不同，这意味着我们实际上是从直接下面的瓦片获取信息。

# 实体存储和管理

没有适当的管理，这些实体只是散布在内存中的随机类，没有任何规律。为了产生一种稳健的方式来创建实体之间的交互，它们需要由一个管理类来监护。在我们开始设计它之前，让我们定义一些数据类型来包含我们将要处理的信息：

```cpp
using EntityContainer = std::unordered_map<unsigned int,EntityBase*>;
using EntityFactory = std::unordered_map<EntityType, std::function<EntityBase*(void)>>;
using EnemyTypes = std::unordered_map<std::string,std::string>;
```

`EntityContainer`类型，正如其名所示，是一个实体容器。它再次由一个`unordered_map`提供支持，将实体实例与作为标识符的无符号整数关联起来。下一个类型是 lambda 函数的容器，它将实体类型与可以分配内存并返回从基实体类继承的类实例的代码链接起来，充当工厂。这种行为对我们来说并不陌生，所以让我们继续定义实体管理器类：

```cpp
class EntityManager{
public:
  EntityManager(SharedContext* l_context,unsigned int l_maxEntities);
  ~EntityManager();

  int Add(const EntityType& l_type,const std::string& l_name = "");
  EntityBase* Find(unsigned int l_id);
  EntityBase* Find(const std::string& l_name);
  void Remove(unsigned int l_id);

  void Update(float l_dT);
  void Draw();

  void Purge();

  SharedContext* GetContext();
private:
  template<class T>
  void RegisterEntity(const EntityType& l_type){
    m_entityFactory[l_type] = [this]() -> EntityBase*
    {
      return new T(this);
    };
  }

  void ProcessRemovals();
  void LoadEnemyTypes(const std::string& l_name);
  void EntityCollisionCheck();

  EntityContainer m_entities;
  EnemyTypes m_enemyTypes;
  EntityFactory m_entityFactory;
  SharedContext* m_context;
  unsigned int m_idCounter;
  unsigned int m_maxEntities;

  std::vector<unsigned int> m_entitiesToRemove;
};
```

除了将 lambda 函数插入实体工厂容器的私有模板方法之外，这个类看起来相对典型。我们有更新和绘制实体、添加、查找和删除它们以及清除所有数据的方法，就像我们通常做的那样。存在名为`ProcessRemovals`的私有方法表明我们正在使用延迟删除实体，就像我们在状态管理器类中所做的那样。让我们通过实现它来更详细地了解这个类的运作方式。

## 实体管理器的实现

和往常一样，一个好的开始是构造函数：

```cpp
EntityManager::EntityManager(SharedContext* l_context,unsigned int l_maxEntities):m_context(l_context),m_maxEntities(l_maxEntities), m_idCounter(0)
{
    LoadEnemyTypes("EnemyList.list");
    RegisterEntity<Player>(EntityType::Player);
    RegisterEntity<Enemy>(EntityType::Enemy);
}
EntityManager::~EntityManager(){ Purge(); }
```

其中一些数据成员通过初始化列表进行初始化。`m_idCounter`变量将用于跟踪分配给实体的最高 ID。接下来，调用一个私有方法来加载敌人名称和它们的角色定义文件对，这将在稍后进行解释。

最后，注册了两种实体类型：玩家和敌人。我们还没有设置它们的类，但很快就会完成，所以现在我们可以先注册它们。

实体管理器的析构函数简单地调用`Purge`方法。

通过将实体类型及其名称传递给实体管理器的`Add`方法来向游戏中添加新实体：

```cpp
int EntityManager::Add(const EntityType& l_type,const std::string& l_name)
{
  auto itr = m_entityFactory.find(l_type);
  if (itr == m_entityFactory.end()){ return -1; }
  EntityBase* entity = itr->second();
  entity->m_id = m_idCounter;
  if (l_name != ""){ entity->m_name = l_name; }
  m_entities.emplace(m_idCounter,entity);
  if(l_type == EntityType::Enemy){
    auto itr = m_enemyTypes.find(l_name);
    if(itr != m_enemyTypes.end()){
      Enemy* enemy = (Enemy*)entity;
      enemy->Load(itr->second);
    }
  }

  ++m_idCounter;
  return m_idCounter - 1;
}
```

实体工厂容器会搜索提供的参数类型。如果该类型已注册，则会调用 lambda 函数来为实体分配动态内存，并通过指向`EntityBase`类的指针变量`entity`捕获内存地址。然后，新创建的实体被插入到实体容器中，并使用`m_idCounter`数据成员设置其 ID。如果用户为实体名称提供了参数，它也会被设置。

接下来检查实体类型。如果是敌人，则会在敌人类型容器中搜索以找到角色定义文件的路径。如果找到，实体会被类型转换为敌人实例，并调用`Load`方法，将角色文件路径传递给它。

最后，ID 计数器递增，并返回刚刚使用的实体 ID 以表示成功。如果在任何点上方法失败，它将返回*-1*，表示失败。

如果你不能获取实体，拥有实体管理器是没有意义的。这就是`Find`方法的作用：

```cpp
EntityBase* EntityManager::Find(const std::string& l_name){
  for(auto &itr : m_entities){
    if(itr.second->GetName() == l_name){
      return itr.second;
    }
  }
  return nullptr;
}
```

我们的实体管理器提供了这个方法的两个版本。第一个版本接受一个实体名称，并在容器中搜索直到找到一个具有该名称的实体，此时它被返回。第二个版本根据数值标识符查找实体：

```cpp
EntityBase* EntityManager::Find(unsigned int l_id){
  auto itr = m_entities.find(l_id);
  if (itr == m_entities.end()){ return nullptr; }
  return itr->second;
}
```

由于我们将实体实例映射到数值，这更容易，因为我们只需调用我们的容器中的`Find`方法来找到我们正在寻找的元素。

现在我们来处理移除实体：

```cpp
void EntityManager::Remove(unsigned int l_id){
    m_entitiesToRemove.emplace_back(l_id);
}
```

这是一个公共方法，它接受一个实体 ID 并将其插入到容器中，该容器将用于稍后移除实体。

更新所有实体可以通过以下方式实现：

```cpp
void EntityManager::Update(float l_dT){
  for(auto &itr : m_entities){
    itr.second->Update(l_dT);
  }
  EntityCollisionCheck();
  ProcessRemovals();
}
```

管理器遍历其所有元素，并通过传递作为参数接收的 delta 时间调用它们各自的`Update`方法。在所有实体更新完毕后，将调用一个私有方法`EntityCollisionCheck`来检查和解决实体之间的碰撞。然后，我们处理由之前实现的`Remove`方法添加的实体移除。

让我们看看我们如何绘制所有这些实体：

```cpp
void EntityManager::Draw(){
  sf::RenderWindow* wnd = m_context->m_wind->GetRenderWindow();
  sf::FloatRect viewSpace = m_context->m_wind->GetViewSpace();

  for(auto &itr : m_entities){
    if (!viewSpace.intersects(itr.second->m_AABB)){ continue; }
    itr.second->Draw(wnd);
  }
}
```

在获取到渲染窗口的指针后，我们也得到了它的视图空间，以便出于效率原因剪裁实体。因为实体的视图空间和边界框都是矩形，我们可以简单地检查它们是否相交，以确定实体是否在视图空间内，如果是的话，它就会被绘制。

实体管理器需要有一种方式来分配其所有资源。这就是`Purge`方法发挥作用的地方：

```cpp
void EntityManager::Purge(){
  for (auto &itr : m_entities){
    delete itr.second;
  }
  m_entities.clear();
  m_idCounter = 0;
}
```

实体被迭代，它们的动态内存被释放——就像时钟一样规律。现在来处理需要被移除的实体：

```cpp
void EntityManager::ProcessRemovals(){
  while(m_entitiesToRemove.begin() != m_entitiesToRemove.end()){
    unsigned int id = m_entitiesToRemove.back();
    auto itr = m_entities.find(id);
    if(itr != m_entities.end()){
      std::cout << "Discarding entity: "<< itr->second->GetId() << std::endl;
      delete itr->second;
      m_entities.erase(itr);
    }
    m_entitiesToRemove.pop_back();
  }
}
```

当我们遍历包含需要移除的实体 ID 的容器时，会检查实体容器中是否存在每个添加的 ID。如果确实存在具有该 ID 的实体，其内存将被释放，并且元素将从实体容器中弹出。

现在是更有趣的部分——检测实体之间的碰撞：

```cpp
void EntityManager::EntityCollisionCheck(){
  if (m_entities.empty()){ return; }
  for(auto itr = m_entities.begin();
    std::next(itr) != m_entities.end(); ++itr)
  {
    for(auto itr2 = std::next(itr);
      itr2 != m_entities.end(); ++itr2)
    {
      if(itr->first == itr2->first){ continue; }

      // Regular AABB bounding box collision.
      if(itr->second->m_AABB.intersects(itr2->second->m_AABB)){
        itr->second->OnEntityCollision(itr2->second, false);
        itr2->second->OnEntityCollision(itr->second, false);
      }

      EntityType t1 = itr->second->GetType();
      EntityType t2 = itr2->second->GetType();
      if (t1 == EntityType::Player || t1 == EntityType::Enemy){
        Character* c1 = (Character*)itr->second;
        if (c1->m_attackAABB.intersects(itr2->second->m_AABB)){
          c1->OnEntityCollision(itr2->second, true);
        }
      }

      if (t2 == EntityType::Player || t2 == EntityType::Enemy){
        Character* c2 = (Character*)itr2->second;
        if (c2->m_attackAABB.intersects(itr->second->m_AABB)){
          c2->OnEntityCollision(itr->second, true);
        }
      }
    }
  }
}
```

首先，我们需要解决的是我们如何检查每个实体与每个其他实体的问题。当然，有更好的、更有效的方法来确定要检查哪些实体，而不仅仅是遍历所有实体，例如二叉空间划分。然而，鉴于我们项目的范围，那将是过度设计：

|   | *"过早优化是编程中所有邪恶（至少是大多数邪恶）的根源。"* |   |
| --- | --- | --- |
|   | --*唐纳德·克努特* |

话虽如此，我们将变得更聪明一些，而不仅仅是简单地迭代所有实体两次。因为检查实体 0 与实体 1 相同于检查实体 1 与 0，我们可以通过使用`std::next`来实现一个更高效的算法，它创建一个比提供的迭代器前移一个空间的迭代器，并在第二个循环中使用它。这创建了一个看起来像这样的检查模式：

![实现实体管理器](img/B04284_07_07.jpg)

这就是我们在游戏早期制作阶段需要的优化。

在遍历实体时，碰撞检查方法首先确保两个迭代器不共享相同的实体 ID，出于某种奇怪的原因。然后，它只是简单地检查我们感兴趣的两个实体的边界框之间的交集。如果有碰撞，则在两个实例中调用处理碰撞的方法，传递被碰撞的实体作为参数，以及作为第二个参数的`false`，以让实体知道这是一个简单的 AABB 碰撞。那是什么意思呢？嗯，一般来说，实体之间会有两种类型的碰撞：常规边界框碰撞和攻击碰撞。`EntityBase`类的子类，主要是`Character`实例，将必须保持另一个边界框以执行攻击，如图所示：

![实现实体管理器](img/B04284_07_08.jpg)

由于这并不复杂，我们可以继续实现实体管理器，直到我们不久后实现`Character`类。

由于只有`Character`类及其任何继承类将具有攻击边界框，因此有必要首先通过验证实体类型来检查我们是否正在处理一个`Character`实例。如果一个实体是`Enemy`或`Player`类型，则调用`Character`实例的`OnEntityCollision`方法，并传递与之碰撞的实体以及这次作为参数的布尔常量`true`，以指示攻击碰撞。

我们基本上已经完成了。让我们编写加载不同敌人类型的方法的代码，这些敌人类型可以解析像这样的文件：

```cpp
|Name|CharFile|
Rat Rat.char
```

这是一个相当简单的格式。让我们读取它：

```cpp
void EntityManager::LoadEnemyTypes(const std::string& l_name){
  std::ifstream file;
  ... // Opening the file.
  while(std::getline(file,line)){
    if (line[0] == '|'){ continue; }
    std::stringstream keystream(line);
    std::string name;
    std::string charFile;
    keystream >> name >> charFile;
    m_enemyTypes.emplace(name,charFile);
  }
  file.close();
}
```

这里没有什么是你以前没有见过的。两个字符串值被读取并存储在敌人类型容器中。这段简单的代码结束了我们对实体管理器类的兴趣。

# 使用实体构建角色

到目前为止，我们只有定义了一些抽象方法并提供操作它们的手段的实体，但没有可以在游戏世界中出现、渲染并四处走动的实体。同时，我们也不想在玩家或敌人类中重新实现所有这些功能，这意味着我们需要一个中间级别的抽象类：`Character`。这个类将提供所有需要在世界中移动并被渲染的实体之间共享的功能。让我们继续设计：

```cpp
class Character : public EntityBase{
friend class EntityManager;
public:
  Character(EntityManager* l_entityMgr);
  virtual ~Character();
  void Move(const Direction& l_dir);
  void Jump();
  void Attack();
  void GetHurt(const int& l_damage);
  void Load(const std::string& l_path);
  virtual void OnEntityCollision(
    EntityBase* l_collider, bool l_attack) = 0;
  virtual void Update(float l_dT);
  void Draw(sf::RenderWindow* l_wind);
protected:
  void UpdateAttackAABB();
  void Animate();
  SpriteSheet m_spriteSheet;
  float m_jumpVelocity;
  int m_hitpoints;
  sf::FloatRect m_attackAABB;
  sf::Vector2f m_attackAABBoffset;
};
```

首先，让我们谈谈公共方法。移动、跳跃、攻击和受到伤害是游戏中每个角色-实体常见的动作。角色还必须被加载，以便提供正确的图形和属性，这些属性在每种敌人类型和玩家之间是不同的。所有从它派生的类都必须实现它们自己的处理与其他实体碰撞的版本。此外，角色类的`Update`方法被设置为虚拟的，这允许任何从该类继承的类定义自己的更新方法或扩展现有的方法。

所有角色都将使用我们之前设计的精灵图集类来支持动画。

## 实现角色类

你现在应该知道了。这是构造函数：

```cpp
Character::Character(EntityManager* l_entityMgr)
  :EntityBase(l_entityMgr), 
  m_spriteSheet(m_entityManager->GetContext()->m_textureManager),
  m_jumpVelocity(250), m_hitpoints(5)
{ m_name = "Character"; }
```

精灵图集是通过在构造函数中传递指向纹理管理器的指针来创建和设置的。我们还有一个名为`m_jumpVelocity`的数据成员，它指定了玩家可以跳多远。最后，我们给`m_hitpoints`变量设置了一个任意值，它代表了实体在被击中多少次后才会死亡。

让我们继续到`Move`方法：

```cpp
void Character::Move(const Direction& l_dir){
  if (GetState() == EntityState::Dying){ return; }
  m_spriteSheet.SetDirection(l_dir);
  if (l_dir == Direction::Left){ Accelerate(-m_speed.x, 0); }
  else { Accelerate(m_speed.x, 0); }
  if (GetState() == EntityState::Idle){
    SetState(EntityState::Walking);
  }
}
```

无论实体的方向如何，都会检查实体的状态，以确保实体没有死亡。如果没有死亡，就会设置精灵图集的方向，并且角色开始在相关轴上加速。最后，如果实体目前处于空闲状态，它会被设置为行走状态，以便播放行走动画：

```cpp
void Character::Jump(){
  if (GetState() == EntityState::Dying || GetState() == EntityState::Jumping || GetState() == EntityState::Hurt)
  {
    return;
  }
  SetState(EntityState::Jumping);
  AddVelocity(0, -m_jumpVelocity);
}
```

一个角色只有在没有死亡、受到伤害或正在跳跃的情况下才能跳跃。当这些条件满足，并且角色被指示跳跃时，其状态被设置为`Jumping`，并在 y 轴上获得负速度，使其对抗重力并向上移动。速度必须足够高，才能打破该级别的重力。

攻击相当直接。因为实体管理器已经为我们做了碰撞检测，所以剩下的只是设置状态，如果实体没有死亡、跳跃、受到伤害或正在攻击的话：

```cpp
void Character::Attack(){
  if (GetState() == EntityState::Dying ||
    GetState() == EntityState::Jumping ||
    GetState() == EntityState::Hurt ||
    GetState() == EntityState::Attacking)
  {
    return;
  }
  SetState(EntityState::Attacking);
}
```

为了赋予我们的实体生命，它们需要有受伤的方式：

```cpp
void Character::GetHurt(const int& l_damage){
  if (GetState() == EntityState::Dying ||
    GetState() == EntityState::Hurt)
  {
    return;
  }
  m_hitpoints = (m_hitpoints - l_damage > 0 ?
    m_hitpoints - l_damage : 0);
  if (m_hitpoints){ SetState(EntityState::Hurt); }
  else { SetState(EntityState::Dying); }
}
```

如果角色尚未受到伤害或死亡，此方法会对角色造成伤害。伤害值要么从生命值中减去，要么将生命值变量设置为*0*，以防止其达到负值。如果减去伤害后实体仍有生命，其状态将被设置为`HURT`，以便播放正确的动画。否则，程序员将实体判处死刑。

如前所述，我们希望能够从像这样的文件中加载我们的角色（`Player.char`）：

```cpp
Name Player
Spritesheet Player.sheet
Hitpoints 5
BoundingBox 20 26
DamageBox -5 0 26 26
Speed 1024 128
JumpVelocity 250
MaxVelocity 200 1024
```

它包含了构成角色的所有基本组成部分，如精灵表句柄以及在前几节中讨论的所有其他信息。此类文件的加载方法与我们已实现的那些方法不会有太大差异：

```cpp
void Character::Load(const std::string& l_path){
  std::ifstream file;
  ...
  while(std::getline(file,line)){
    ...
    std::string type;
    keystream >> type;
    if(type == "Name"){
      keystream >> m_name;
    } else if(type == "Spritesheet"){
      std::string path;
      keystream >> path;
      m_spriteSheet.LoadSheet("media/SpriteSheets/" + path);
    } else if(type == "Hitpoints"){
      keystream >> m_hitpoints;
    } else if(type == "BoundingBox"){
      sf::Vector2f boundingSize;
      keystream >> boundingSize.x >> boundingSize.y;
      SetSize(boundingSize.x, boundingSize.y);
    } else if(type == "DamageBox"){
      keystream >> m_attackAABBoffset.x >> m_attackAABBoffset.y 
        >> m_attackAABB.width >> m_attackAABB.height;
    } else if(type == "Speed"){
      keystream >> m_speed.x >> m_speed.y;
    } else if(type == "JumpVelocity"){
      keystream >> m_jumpVelocity;
    } else if(type == "MaxVelocity"){
      keystream >> m_maxVelocity.x >> m_maxVelocity.y;
    } else {
      std::cout << "! Unknown type in character file: "
        << type << std::endl;
    }
  }
  file.close();
}
```

除了精灵表需要调用一个加载方法外，其余的只是从字符串流中加载数据成员。

就像基础实体及其边界框一样，角色必须有一种方法来更新其攻击区域的位置：

```cpp
void Character::UpdateAttackAABB(){
  m_attackAABB.left = 
    (m_spriteSheet.GetDirection() == Direction::Left ? 
    (m_AABB.left - m_attackAABB.width) - m_attackAABBoffset.x
    : (m_AABB.left + m_AABB.width) + m_attackAABBoffset.x);
  m_attackAABB.top = m_AABB.top + m_attackAABBoffset.y;
}
```

这里的一个细微差别是，攻击边界框使用的是实体边界框的位置，而不是其精灵位置。此外，其定位方式根据实体面对的方向而不同，因为边界框的位置代表其左上角。

现在是时候介绍将带来最大视觉差异的方法了：

```cpp
void Character::Animate(){
  EntityState state = GetState();

  if(state == EntityState::Walking && m_spriteSheet.
    GetCurrentAnim()->GetName() != "Walk")
  {
    m_spriteSheet.SetAnimation("Walk",true,true);
  } 
  else if(state == EntityState::Jumping && m_spriteSheet.
    GetCurrentAnim()->GetName() != "Jump")
  {
    m_spriteSheet.SetAnimation("Jump",true,false);
  }
  else if(state == EntityState::Attacking && m_spriteSheet.
    GetCurrentAnim()->GetName() != "Attack")
  {
    m_spriteSheet.SetAnimation("Attack",true,false);
  } else if(state == EntityState::Hurt && m_spriteSheet.
    GetCurrentAnim()->GetName() != "Hurt")
  {
    m_spriteSheet.SetAnimation("Hurt",true,false);
  }
  else if(state == EntityState::Dying && m_spriteSheet.
    GetCurrentAnim()->GetName() != "Death")
  {
    m_spriteSheet.SetAnimation("Death",true,false);
  }
  else if(state == EntityState::Idle && m_spriteSheet.
    GetCurrentAnim()->GetName() != "Idle")
  {
    m_spriteSheet.SetAnimation("Idle",true,true);
  }
}
```

它所做的只是简单地检查当前状态和当前动画。如果当前动画与当前状态不匹配，它将被设置为其他内容。注意`SetAnimation`方法中的第三个参数的使用，它是一个布尔常量，代表动画循环。某些动画不需要循环，如攻击或受伤动画。它们不循环并在达到最后一帧时停止，这为我们提供了一个钩子，可以根据特定动画的进度来操纵游戏中的发生的事情。以`Update`方法为例：

```cpp
void Character::Update(float l_dT){
  EntityBase::Update(l_dT);
  if(m_attackAABB.width != 0 && m_attackAABB.height != 0){
    UpdateAttackAABB();
  }
  if(GetState() != EntityState::Dying && GetState() !=
    EntityState::Attacking && GetState() != EntityState::Hurt)
  {
    if(abs(m_velocity.y) >= 0.001f){
      SetState(EntityState::Jumping);
    } else if(abs(m_velocity.x) >= 0.1f){
      SetState(EntityState::Walking);
    } else {
      SetState(EntityState::Idle);
    }
  } else if(GetState() == EntityState::Attacking ||
    GetState() == EntityState::Hurt)
  {
    if(!m_spriteSheet.GetCurrentAnim()->IsPlaying()){
      SetState(EntityState::Idle);
    }
  } else if(GetState() == EntityState::Dying){
    if(!m_spriteSheet.GetCurrentAnim()->IsPlaying()){
      m_entityManager->Remove(m_id);
    }
  }
  Animate();
  m_spriteSheet.Update(l_dT);
  m_spriteSheet.SetSpritePosition(m_position);
}
```

首先，我们调用实体基类的更新方法，因为角色的状态依赖于它。然后，我们检查攻击边界框的宽度和高度是否仍然为 0，这是它们的默认值。如果不是，这意味着攻击边界框已经设置好，可以更新。更新方法的其余部分基本上只是处理状态转换。如果实体没有死亡、攻击某物或受到伤害，其当前状态将由其速度决定。为了准确地描绘实体下落，我们必须使 y 轴上的速度优先于其他所有因素。如果实体没有垂直速度，则检查水平速度，如果速度高于指定的最小值，则将状态设置为`Walking`。使用小值而不是绝对零值可以解决动画有时会抖动的问题。

由于攻击和受到伤害的状态没有设置为循环，因此会检查精灵表动画，以查看它是否仍在播放。如果没有播放，状态会切换回空闲状态。最后，如果实体正在死亡并且死亡动画播放完毕，我们会调用实体管理器的 `Remove` 方法，以便从世界中移除这个实体。

`Animate` 方法在更新接近结束时被调用，以便反映可能发生的状态变化。此外，这也是精灵表更新并设置其位置以匹配实体位置的地方。

在所有这些代码之后，让我们以一个真正简单的东西结束——`Draw` 方法：

```cpp
void Character::Draw(sf::RenderWindow* l_wind){
    m_spriteSheet.Draw(l_wind);
}
```

由于我们的精灵表类负责绘制，我们只需要传递渲染窗口的指针到其 `Draw` 方法。

## 创建玩家

现在我们已经为在屏幕上可视化的实体创建了一个坚实的基础。让我们充分利用它，并从开始构建玩家类，从头文件开始：

```cpp
class Player : public Character{
public:
  Player(EntityManager* l_entityMgr);
  ~Player();

  void OnEntityCollision(EntityBase* l_collider, bool l_attack);
  void React(EventDetails* l_details);
};
```

这里事情变得简单起来。因为我们基本上将大部分通用功能外包给了基类，现在我们只剩下玩家特定的逻辑。注意 `React` 方法。根据其参数列表，很明显我们将将其用作处理玩家输入的回调。然而，在我们这样做之前，我们必须将此方法注册为：

```cpp
Player::Player(EntityManager* l_entityMgr)
  : Character(l_entityMgr)
{
  Load("Player.char");
  m_type = EntityType::Player;

  EventManager* events = m_entityManager->
    GetContext()->m_eventManager;
  events->AddCallback<Player>(StateType::Game,
    "Player_MoveLeft", &Player::React, this);
  events->AddCallback<Player>(StateType::Game,
    "Player_MoveRight", &Player::React, this);
  events->AddCallback<Player>(StateType::Game,
    "Player_Jump", &Player::React, this);
  events->AddCallback<Player>(StateType::Game,
    "Player_Attack", &Player::React, this);
}
```

我们在这里所做的只是调用 `Load` 方法来设置玩家的角色值，并向将用于处理键盘输入的同一 `React` 方法添加多个回调。实体的类型也被设置为 `Player`：

```cpp
Player::~Player(){
    EventManager* events = m_entityManager->GetContext()->m_eventManager;
    events->RemoveCallback(GAME,"Player_MoveLeft");
    events->RemoveCallback(GAME,"Player_MoveRight");
    events->RemoveCallback(GAME,"Player_Jump");
    events->RemoveCallback(GAME,"Player_Attack");
}
```

析构函数，不出所料，只是简单地移除了我们用来移动玩家的回调函数。

我们需要通过 `Character` 类实现的最后一个方法是负责实体之间的碰撞：

```cpp
void Player::OnEntityCollision(EntityBase* l_collider,
  bool l_attack)
{
  if (m_state == EntityState::Dying){ return; }
  if(l_attack){
    if (m_state != EntityState::Attacking){ return; }
    if (!m_spriteSheet.GetCurrentAnim()->IsInAction()){ return; }
    if (l_collider->GetType() != EntityType::Enemy &&
      l_collider->GetType() != EntityType::Player)
    {
      return;
    }
    Character* opponent = (Character*)l_collider;
    opponent->GetHurt(1);
    if(m_position.x > opponent->GetPosition().x){
      opponent->AddVelocity(-32,0);
    } else {
      opponent->AddVelocity(32,0);
    }
  } else {
    // Other behavior.
  }
}
```

这个方法，正如你从本章的实体管理部分所记得的，当某个实体与这个特定实体发生碰撞时会被调用。在碰撞的情况下，另一个发生碰撞的实体作为参数传递给这个方法，同时还有一个标志来确定实体是否与你的边界框或攻击区域发生碰撞。

首先，我们确保玩家实体没有死亡。然后，我们检查是否是攻击区域与另一个实体发生碰撞。如果是，并且玩家处于攻击状态，我们检查精灵表中的攻击动画是否目前正在“进行中”。如果当前帧在动作应该发生时的开始和结束帧的范围内，最后的检查是确定实体是玩家还是敌人。最后，如果是其中之一，对手会受到预先确定的伤害值，并根据其位置添加一些速度以产生击退效果。这就是最基本的游戏设计。

# 添加敌人

为了让我们的玩家不会在世界中孤独地行走且不受攻击，我们必须在游戏中添加敌人。再次，让我们从头文件开始：

```cpp
#pragma once
#include "Character.h"

class Enemy : public Character{
public:
    Enemy(EntityManager* l_entityMgr);
    ~Enemy();

    void OnEntityCollision(EntityBase* l_collider, bool l_attack);
    void Update(float l_dT);
private:
    sf::Vector2f m_destination;
    bool m_hasDestination;
};
```

这里的基本想法与玩家类中的相同。然而，这次敌人类需要指定它自己的`Update`方法版本。它还有两个私有数据成员，其中一个是一个目标向量。这是一个非常简单的尝试，为游戏添加基本的人工智能。它只会跟踪一个目标位置，而`Update`方法会不时随机化这个位置以模拟游荡的实体。让我们来实现它：

```cpp
Enemy::Enemy(EntityManager* l_entityMgr)
  :Character(l_entityMgr), m_hasDestination(false)
{
  m_type = EntityType::Enemy;
}
Enemy::~Enemy(){}
```

构造函数只是将一些数据成员初始化为其默认值，而析构函数仍然未使用。到目前为止，一切顺利！

```cpp
void Enemy::OnEntityCollision(EntityBase* l_collider,
  bool l_attack)
{
  if (m_state == EntityState::Dying){ return; }
  if (l_attack){ return; }
  if (l_collider->GetType() != EntityType::Player){ return; }
  Character* player = (Character*)l_collider;
  SetState(EntityState::Attacking);
  player->GetHurt(1);
  if(m_position.x > player->GetPosition().x){
    player->AddVelocity(-m_speed.x,0);
    m_spriteSheet.SetDirection(Direction::Left);
  } else {
    player->AddVelocity(m_speed.y,0);
    m_spriteSheet.SetDirection(Direction::Right);
  }
}
```

实体碰撞方法也非常相似，但这次我们确保只有在敌人的边界框与另一个实体碰撞时才采取行动，而不是它的攻击区域。此外，我们忽略每一次碰撞，除非它与玩家实体碰撞，在这种情况下，敌人的状态被设置为`Attacking`以显示攻击动画。它对玩家造成*1*点的伤害，并根据实体的位置将其击退一小点。精灵图的方向也根据敌人实体相对于其攻击的位置来设置。

现在，来更新我们的敌人：

```cpp
void Enemy::Update(float l_dT){
  Character::Update(l_dT);

  if (m_hasDestination){
    if (abs(m_destination.x - m_position.x) < 16){
      m_hasDestination = false;
      return;
    }
    if (m_destination.x - m_position.x > 0){
      Move(Direction::Right);
    } else { Move(Direction::Left); }
    if (m_collidingOnX){ m_hasDestination = false; }
    return;
  }
  int random = rand() % 1000 + 1;
  if (random != 1000){ return; }
  int newX = rand() % 65 + 0;
  if (rand() % 2){ newX = -newX; }
  m_destination.x = m_position.x + newX;
  if (m_destination.x < 0){ m_destination.x = 0; }
  m_hasDestination = true;
}
```

因为这依赖于`Character`类的功能，我们在做任何事情之前首先调用它的更新方法。然后，最基础的 AI 模拟首先检查实体是否有目标。如果没有，则在 1 到 1000 之间生成一个随机数。它有 1/1000 的机会将其目标位置设置为当前位置 128 像素内的任何地方。方向由另一个随机数生成决定，但这次要小得多。目标最终被设置并检查是否超出世界边界。

另一方面，如果实体确实有一个目标，则检查它与当前位置之间的距离。如果它大于 16，则根据目标点所在的方向调用适当的移动方法。我们还必须检查水平碰撞，因为敌人实体可能会被分配一个它无法跨越的瓦片之外的目标。如果发生这种情况，目标就会被简单地移除。

完成这些后，我们现在有了游荡的实体和可以在世界中移动的玩家！现在要真正将这些实体引入游戏，唯一剩下的事情就是加载它们。

# 从地图文件加载实体

如果您还记得本章中关于创建地图类的问题部分，我们还没有完全实现加载方法，因为我们当时还没有实体。既然这种情况已经不再存在，让我们来看看如何扩展它：

```cpp
} else if(type == "PLAYER"){
  if (playerId != -1){ continue; }
  // Set up the player position here.
  playerId = entityMgr->Add(EntityType::Player);
  if (playerId < 0){ continue; }
  float playerX = 0; float playerY = 0;
  keystream >> playerX >> playerY;
  entityMgr->Find(playerId)->SetPosition(playerX,playerY);
  m_playerStart = sf::Vector2f(playerX, playerY);
} else if(type == "ENEMY"){
  std::string enemyName;
  keystream >> enemyName;
  int enemyId = entityMgr->Add(EntityType::Enemy, enemyName);
  if (enemyId < 0){ continue; }
  float enemyX = 0; float enemyY = 0;
  keystream >> enemyX >> enemyY;
  entityMgr->Find(enemyId)->SetPosition(enemyX, enemyY);
} ...
```

如果地图遇到`PLAYER`行，它将尝试添加一个类型为`Player`的实体并获取其 ID。如果它大于或等于 0，则实体创建成功，这意味着我们可以从地图文件中读取其余的数据，这恰好是玩家位置。获取后，我们设置玩家的位置并确保在地图类本身中也跟踪起始位置。

所有上述内容对于`ENEMY`行也是正确的，只是它还加载了实体的名称，这是从文件中加载其角色信息所必需的。

现在游戏能够从地图文件中加载实体并将它们放入游戏世界，如下所示：

![从地图文件加载实体](img/B04284_07_09.jpg)

# 对代码库的最终修订

在本章的最后部分，我们将介绍为了实现这一点而做出的所有小改动和添加/修订，从共享上下文开始，现在它已经移动到了自己的头文件中。

## 共享上下文的变化

在我们定义的所有额外类中，其中一些需要可供代码库的其余部分访问。这就是现在共享上下文结构的样子：

```cpp
class Map;
struct SharedContext{
  SharedContext():
    m_wind(nullptr),
    m_eventManager(nullptr),
    m_textureManager(nullptr),
    m_entityManager(nullptr),
    m_gameMap(nullptr){}

  Window* m_wind;
  EventManager* m_eventManager;
  TextureManager* m_textureManager;
  EntityManager* m_entityManager;
  Map* m_gameMap;
  DebugOverlay m_debugOverlay;
};
```

其中的最后一个对象是我们之前在处理基础实体类时简要讨论的调试覆盖层，它通过为实体碰撞的瓦片、扭曲瓦片和尖刺瓦片提供覆盖图形，帮助我们通过实体边界框等的视觉表示来查看游戏中的情况。由于调试代码对于本章不是必需的，因此没有将其片段包含在这里，但它们存在于附带的代码中。

## 将所有部件组合在一起

接下来，我们需要将我们辛苦工作的代码实例放在正确的位置，首先是实体管理器类，它直接作为数据成员进入游戏类：

```cpp
class Game{
public:
    ...
private:
    ...
    EntityManager m_entityManager;
};
```

地图类实例保留在游戏状态类中：

```cpp
class State_Game : public BaseState{
public:
    ...
private:
    ...
    Map* m_gameMap;
};
```

主游戏状态还负责设置自己的视图并放大到足以使游戏看起来更有吸引力且不太可能导致眯眼，更不用说初始化和加载地图：

```cpp
void State_Game::OnCreate(){
    ...
    sf::Vector2u size = m_stateMgr->GetContext()->m_wind->GetWindowSize();
    m_view.setSize(size.x,size.y);
    m_view.setCenter(size.x/2,size.y/2);
    m_view.zoom(0.6f);
    m_stateMgr->GetContext()->m_wind->GetRenderWindow()->setView(m_view);

    m_gameMap = new Map(m_stateMgr->GetContext(), this);
    m_gameMap->LoadMap("media/Maps/map1.map");
}
```

由于地图是动态分配的，必须在游戏状态的`OnDestroy`方法中删除它：

```cpp
void State_Game::OnDestroy(){
    ...
    delete m_gameMap;
    m_gameMap = nullptr;
}
```

接下来是拼图中最后一块——游戏状态更新方法：

```cpp
void State_Game::Update(const sf::Time& l_time){
  SharedContext* context = m_stateMgr->GetContext();
  EntityBase* player = context->m_entityManager->Find("Player");
  if(!player){
    std::cout << "Respawning player..." << std::endl;
    context->m_entityManager->Add(EntityType::Player,"Player");
    player = context->m_entityManager->Find("Player");
    player->SetPosition(m_gameMap->GetPlayerStart());
  } else {
    m_view.setCenter(player->GetPosition());
    context->m_wind->GetRenderWindow()->setView(m_view);
  }

  sf::FloatRect viewSpace = context->m_wind->GetViewSpace();
  if(viewSpace.left <= 0){
    m_view.setCenter(viewSpace.width / 2,m_view.getCenter().y);
    context->m_wind->GetRenderWindow()->setView(m_view);
  } else if (viewSpace.left + viewSpace.width >
    (m_gameMap->GetMapSize().x + 1) * Sheet::Tile_Size)
  {
    m_view.setCenter(((m_gameMap->GetMapSize().x + 1) *
      Sheet::Tile_Size) - (viewSpace.width / 2),
      m_view.getCenter().y);
    context->m_wind->GetRenderWindow()->setView(m_view);
  }

  m_gameMap->Update(l_time.asSeconds());
  m_stateMgr->GetContext()->
    m_entityManager->Update(l_time.asSeconds());
}
```

首先，我们通过按名称搜索玩家来确定玩家是否仍然存活于游戏中。如果找不到玩家，他们肯定已经死亡，因此需要重生。创建一个新的玩家实体并将地图的起始坐标传递给其`SetPosition`方法。

现在是管理视图滚动方式的部分。如果玩家实体存在，我们将视图的中心设置为与玩家位置完全匹配，并使用共享上下文获取渲染窗口，该窗口将使用更新的视图。现在，我们遇到了屏幕离开地图边界的问题，这可以通过检查视图空间左上角来解决。如果它低于或等于零，我们将视图中心的 x 轴位置设置为将其左上角放置在屏幕边缘的位置，以防止无限地向左滚动。然而，如果视图在地图的相反方向外，视图中心的 x 坐标被设置为使其右侧也位于地图边界的最边缘。

最后，游戏地图以及实体管理器在这里进行了更新，因为我们不希望地图更新或实体移动，如果当前状态不同的话。

# 摘要

恭喜你完成了本书的一半！所有编写的代码、设计决策、考虑效率以及反复试验都把你带到了这个阶段。虽然我们构建的游戏相当基础，但其架构也非常稳健和可扩展，这可不是一件小事。尽管其中的一些事情可能并不完美，但你也已经遵循了先让它工作起来，然后再进行优化的黄金法则，现在你已经有了一些游戏设计模式，可以开始构建更复杂的应用程序，以及一个坚实的代码库来扩展和改进。

随着本章的结束，本书的第二个项目正式完成。我们解决了一些相当棘手的问题，编写了数千行代码，并将我们对游戏开发过程的理解扩展到了狭隘、幼稚的幻想阶段之外。但真正的冒险还在前方等着我们。我们可能不知道它最终会引导我们走向何方，但有一点是肯定的：现在绝不是停下脚步的时候。下一章见。
