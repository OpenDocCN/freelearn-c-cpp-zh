# 第九章. 一股清新的空气 – 实体组件系统继续

在上一章中，我们讨论了使用聚合而非简单继承的优点。虽然一开始可能不太直观，但由多个组件组成并由系统操作的实体无疑提高了代码的灵活性和可重用性，更不用说为未来的增长提供了一个更方便的环境。正如流行表达所说，“未来已经到来！”一座房子没有好的基础是无用的，就像一个好的基础如果没有在上面建造房子也是无用的。既然我们已经有了坚实的基础，那么接下来就是砌砖直到出现一个合适的结构。

在本章中，我们将：

+   实现基本移动

+   开发一个更新精灵图的系统

+   重新审视并实现实体状态

+   在实体组件系统范式下研究碰撞

# 添加实体移动

在实体组件系统范式下，特定身体的移动是通过作用在其上的所有力来量化的。这些力的集合可以表示为一个可移动组件：

```cpp
class C_Movable : public C_Base{
public:
    ...
private:
    sf::Vector2f m_velocity;
    float m_velocityMax;
    sf::Vector2f m_speed;
    sf::Vector2f m_acceleration;
    Direction m_direction;
};
```

此组件从本书的第二项目中移除了物理元素，即速度、速度和加速度属性。为了简化代码，这次将速度限制表示为一个单一的浮点数，因为我们不太可能需要根据其轴来不同地限制速度。

让我们看看可移动组件类的其余部分：

```cpp
C_Movable() : C_Base(Component::Movable),
    m_velocityMax(0.f), m_direction((Direction)0)
{}
```

此处的构造函数将数据成员初始化为一些默认值，这些值随后将由反序列化中的值替换：

```cpp
void ReadIn(std::stringstream& l_stream){
    l_stream >> m_velocityMax >> m_speed.x >> m_speed.y;

    unsigned int dir = 0;
    l_stream >> dir;
    m_direction = (Direction)dir;
}
```

为了方便地在一定范围内操纵速度，我们提供了`AddVelocity`方法：

```cpp
void AddVelocity(const sf::Vector2f& l_vec){
  m_velocity += l_vec;
  if(std::abs(m_velocity.x) > m_velocityMax){
    m_velocity.x = m_velocityMax *
      (m_velocity.x / std::abs(m_velocity.x));
  }

  if(std::abs(m_velocity.y) > m_velocityMax){
    m_velocity.y = m_velocityMax *
      (m_velocity.y / std::abs(m_velocity.y));
  }
}
```

在添加提供的速度参数后，检查最终结果是否高于每个轴上允许的最大值。如果是，则将速度限制在允许的最大值，并保留适当的符号。

```cpp
void ApplyFriction(const sf::Vector2f& l_vec){
  if(m_velocity.x != 0 && l_vec.x != 0){
    if(std::abs(m_velocity.x) - std::abs(l_vec.x) < 0){
      m_velocity.x = 0;
    } else {
      m_velocity.x += (m_velocity.x > 0 ? l_vec.x * -1 : l_vec.x);
    }
  }

  if(m_velocity.y != 0 && l_vec.y != 0){
    if(std::abs(m_velocity.y) - std::abs(l_vec.y) < 0){
      m_velocity.y = 0;
    } else {
      m_velocity.y += (m_velocity.y > 0 ? l_vec.y * -1 : l_vec.y);
    }
  }
}
```

将摩擦应用于当前速度也是受控的。为了避免摩擦使速度改变符号，它被检查是否不等于零，以及当前速度的绝对值与提供的摩擦之间的差异不会是负数。如果是，则将速度设置为零。否则，将摩擦值以适当的符号添加到当前速度。

为了使实体能够移动，它必须被加速。让我们提供一个方法：

```cpp
void Accelerate(const sf::Vector2f& l_vec){ 
    m_acceleration += l_vec;
}
void Accelerate(float l_x, float l_y){ 
    m_acceleration += sf::Vector2f(l_x,l_y);
}
```

为了方便起见，我们提供了相同的方法，重载以接受两种类型的参数：一个浮点向量和两个单独的浮点值。它所做的只是简单地将参数值添加到当前加速度。

最后，实体也可以根据提供的方向移动，而不是手动调用`Accelerate`方法：

```cpp
void Move(const Direction& l_dir){
  if(l_dir == Direction::Up){
    m_acceleration.y -= m_speed.y;
  } else if (l_dir == Direction::Down){
    m_acceleration.y += m_speed.y;
  } else if (l_dir == Direction::Left){
    m_acceleration.x -= m_speed.x;
  } else if (l_dir == Direction::Right){
    m_acceleration.x += m_speed.x;
  }
}
```

根据提供的方向参数，实体的速度被添加到加速度向量中。

## 移动系统

在设计好移动组件后，让我们尝试实现实际移动我们实体的系统：

```cpp
enum class Axis{ x, y };
class Map;
class S_Movement : public S_Base{
public:
  ...
  void SetMap(Map* l_gameMap);
private:
  void StopEntity(const EntityId& l_entity,
    const Axis& l_axis);
  void SetDirection(const EntityId& l_entity,
    const Direction& l_dir);
  const sf::Vector2f& GetTileFriction(unsigned int l_elevation, 
    unsigned int l_x, unsigned int l_y);
  void MovementStep(float l_dT, C_Movable* l_movable,
    C_Position* l_position);
  Map* m_gameMap;
};
```

首先，创建一个`Axis`枚举，以便简化此类私有辅助方法中的代码。然后，我们提前声明一个`Map`类，以便能够在头文件中使用它。这样，就有一个`Map`数据成员，以及一个公共方法，用于向移动系统提供一个`Map`实例。还需要一些私有辅助方法来使代码更易于阅读。让我们从设置构造函数开始：

```cpp
S_Movement::S_Movement(SystemManager* l_systemMgr) 
  : S_Base(System::Movement,l_systemMgr)
{
  Bitmask req;
  req.TurnOnBit((unsigned int)Component::Position);
  req.TurnOnBit((unsigned int)Component::Movable);
  m_requiredComponents.push_back(req);
  req.Clear();

  m_systemManager->GetMessageHandler()->
    Subscribe(EntityMessage::Is_Moving,this);

  m_gameMap = nullptr;
}
```

该系统的要求包括两个组件：位置和可移动。除此之外，该系统还订阅了`Is_Moving`消息类型，以便对其做出响应。

接下来，让我们更新我们的实体信息：

```cpp
void S_Movement::Update(float l_dT){
  if (!m_gameMap){ return; }
  EntityManager* entities = m_systemManager->GetEntityManager();
  for(auto &entity : m_entities){
    C_Position* position = entities->
      GetComponent<C_Position>(entity, Component::Position);
    C_Movable* movable = entities->
      GetComponent<C_Movable>(entity, Component::Movable);
    MovementStep(l_dT, movable, position);
    position->MoveBy(movable->GetVelocity() * l_dT);
  }
}
```

如该系统的要求所示，它将在位置组件和可移动组件上运行。对于属于此系统的每个实体，我们希望更新其物理属性并根据其速度和帧间经过的时间调整其位置，从而产生基于力的移动。

让我们看看移动步骤方法：

```cpp
void S_Movement::MovementStep(float l_dT, C_Movable* l_movable,
  C_Position* l_position)
{
  sf::Vector2f f_coefficient = 
    GetTileFriction(l_position->GetElevation(),
    floor(l_position->GetPosition().x / Sheet::Tile_Size),
    floor(l_position->GetPosition().y / Sheet::Tile_Size));

  sf::Vector2f friction(l_movable->GetSpeed().x * f_coefficient.x,
    l_movable->GetSpeed().y * f_coefficient.y);

  l_movable->AddVelocity(l_movable->GetAcceleration() * l_dT);
  l_movable->SetAcceleration(sf::Vector2f(0.0f, 0.0f));
  l_movable->ApplyFriction(friction * l_dT);

  float magnitude = sqrt(
    (l_movable->GetVelocity().x * l_movable->GetVelocity().x) +
    (l_movable->GetVelocity().y * l_movable->GetVelocity().y));

  if (magnitude <= l_movable->GetMaxVelocity()){ return; }
  float max_V = l_movable->GetMaxVelocity();
  l_movable->SetVelocity(sf::Vector2f(
    (l_movable->GetVelocity().x / magnitude) * max_V,
    (l_movable->GetVelocity().y / magnitude) * max_V));
}
```

首先获取实体站立的地砖的摩擦值。在根据加速度值更新速度后，立即将其应用于实体的可移动组件。

接下来，我们必须确保对角移动被正确处理。考虑以下插图：

![移动系统](img/B04284_09_01.jpg)

根据勾股定理，表示对角移动的直角三角形的斜边平方等于其两边的平方和。换句话说，斜边比两边的和要短。例如，向右下移动的角色看起来会比单方向移动得更快，除非我们根据速度向量的幅度（也称为我们插图中的三角形的斜边）限制它们的速度。一旦计算出幅度，就会检查它是否超过了实体可能的最大速度。如果超过了，它会被归一化并乘以最大速度的值，以强制对角移动变慢。

获取地砖摩擦力的方法如下：

```cpp
const sf::Vector2f& S_Movement::GetTileFriction(
  unsigned int l_elevation, unsigned int l_x, unsigned int l_y)
{
  Tile* t = nullptr;
  while (!t && l_elevation >= 0){
    t = m_gameMap->GetTile(l_x, l_y, l_elevation);
    --l_elevation;
  }

  return(t ? t->m_properties->m_friction :
    m_gameMap->GetDefaultTile()->m_friction);
}
```

在启动`while`循环之前设置一个地砖指针。它将不断尝试在提供的位置获取地砖，同时每次减少海拔。这意味着地砖摩擦力实际上是从玩家所在的最顶层地砖中获得的。如果没有找到地砖，则返回默认摩擦值。

如你现在可能猜到的，由于其重要性，移动系统需要响应相当多的事件：

```cpp
void S_Movement::HandleEvent(const EntityId& l_entity, 
  const EntityEvent& l_event)
{
  switch(l_event){
  case EntityEvent::Colliding_X:
    StopEntity(l_entity,Axis::x); break;
  case EntityEvent::Colliding_Y:
    StopEntity(l_entity, Axis::y); break;
  case EntityEvent::Moving_Left:
    SetDirection(l_entity, Direction::Left); break;
  case EntityEvent::Moving_Right:
    SetDirection(l_entity, Direction::Right); break;
  case EntityEvent::Moving_Up:
    {
      C_Movable* mov = m_systemManager->GetEntityManager()->
        GetComponent<C_Movable>(l_entity,Component::Movable);
      if(mov->GetVelocity().x == 0){
        SetDirection(l_entity, Direction::Up);
      }
    }
    break;
  case EntityEvent::Moving_Down:
    {
      C_Movable* mov = m_systemManager->GetEntityManager()->
        GetComponent<C_Movable>(l_entity,Component::Movable);
      if(mov->GetVelocity().x == 0){
        SetDirection(l_entity, Direction::Down);
      }
    }
    break;
  }
}
```

首先，它处理两个碰撞事件，通过调用私有的`StopEntity`方法来在指定轴上停止实体。接下来，我们有四个移动事件。在`Moving_Left`和`Moving_Right`的情况下，调用私有的`SetDirection`方法来更新实体的方向。然而，上下移动则略有不同。我们希望实体的方向只有在它没有*x*轴上的速度时才改变。否则，它最终会以一种相当滑稽的方式移动。

接下来是消息处理：

```cpp
void S_Movement::Notify(const Message& l_message){
  EntityManager* eMgr = m_systemManager->GetEntityManager();
  EntityMessage m = (EntityMessage)l_message.m_type;
  switch(m){
  case EntityMessage::Is_Moving:
    {
    if (!HasEntity(l_message.m_receiver)){ return; }
    C_Movable* movable = eMgr->GetComponent<C_Movable>
      (l_message.m_receiver, Component::Movable);
    if (movable->GetVelocity() != sf::Vector2f(0.0f, 0.0f))
    {
      return;
    }
    m_systemManager->AddEvent(l_message.m_receiver,
      (EventID)EntityEvent::Became_Idle);
    }
    break;
  }
}
```

在这里，我们只关心一种消息类型：`Is_Moving`。这是一个消息，当实体变得空闲时，会触发发送另一个消息。首先，检查系统是否包含相关的实体。然后获取其实体的可移动组件，检查其速度是否为零。既然是这样，就创建一个事件来表示实体变得空闲。

现在我们只剩下私有辅助方法。这些都是冗余逻辑，其存在避免了代码重复。我们将首先检查负责停止实体的第一个方法：

```cpp
void S_Movement::StopEntity(const EntityId& l_entity, 
  const Axis& l_axis)
{
  C_Movable* movable = m_systemManager->GetEntityManager()->
    GetComponent<C_Movable>(l_entity,Component::Movable);
  if(l_axis == Axis::x){
    movable->SetVelocity(sf::Vector2f(0.f, movable->GetVelocity().y));
  } else if(l_axis == Axis::y){
    movable->SetVelocity(sf::Vector2f(movable->GetVelocity().x, 0.f));
  }
}
```

在获得其可移动组件后，实体在其轴上将其速度设置为零，该轴作为此方法的参数提供。

```cpp
void S_Movement::SetDirection(const EntityId& l_entity, 
  const Direction& l_dir)
{
  C_Movable* movable = m_systemManager->GetEntityManager()->
    GetComponent<C_Movable>(l_entity,Component::Movable);
  movable->SetDirection(l_dir);

  Message msg((MessageType)EntityMessage::Direction_Changed);
  msg.m_receiver = l_entity;
  msg.m_int = (int)l_dir;
  m_systemManager->GetMessageHandler()->Dispatch(msg);
}
```

`SetDirection`方法更新可移动组件的方向。然后发送一条消息来通知所有其他系统这一变化。

最后，我们只剩下一个`Map`类的设置器方法：

```cpp
void S_Movement::SetMap(Map* l_gameMap){ m_gameMap = l_gameMap; }
```

为了使实体具有动态摩擦，移动系统必须能够访问`Map`类，因此它在游戏状态中设置：

```cpp
void State_Game::OnCreate(){
  ...
  m_stateMgr->GetContext()->m_systemManager->
    GetSystem<S_Movement>(SYSTEM_MOVEMENT)->SetMap(m_gameMap);
}
```

最后这段代码片段完成了移动系统的实现。我们的实体现在可以根据施加在它们身上的力进行移动。然而，有了移动支持实际上并不产生移动。这就是实体状态系统发挥作用的地方。

## 实现状态

移动，就像许多与实体相关的其他动作和事件一样，取决于它们当前状态是否令人满意。一个垂死的玩家不应该能够四处移动。应根据其实际状态播放相关的动画。强制执行这些规则需要实体具有状态组件：

```cpp
enum class EntityState{ Idle, Walking, Attacking, Hurt, Dying };
class C_State : public C_Base{
public:
  C_State(): C_Base(Component::State){}
  void ReadIn(std::stringstream& l_stream){
    unsigned int state = 0;
    l_stream >> state;
    m_state = (EntityState)state;
  }

  EntityState GetState(){ return m_state; }
  void SetState(const EntityState& l_state){
    m_state = l_state;
  }
private:
  EntityState m_state;
};
```

如您所知，这是一段非常简单的代码。它定义了自己的实体可能状态枚举。组件类本身仅提供设置器和获取器，以及所需的反序列化方法。其余的，像往常一样，留给系统自行处理。

### 状态系统

由于从现在开始的大多数系统头文件看起来几乎相同，因此将省略它们。话虽如此，让我们首先实现我们的状态系统的构造函数和析构函数：

```cpp
S_State::S_State(SystemManager* l_systemMgr)
  : S_Base(System::State,l_systemMgr)
{
  Bitmask req;
  req.TurnOnBit((unsigned int)Component::State);
  m_requiredComponents.push_back(req);

  m_systemManager->GetMessageHandler()->
    Subscribe(EntityMessage::Move,this);
  m_systemManager->GetMessageHandler()->
    Subscribe(EntityMessage::Switch_State,this);
}
```

这个系统所需的所有东西只是一个状态组件。它还订阅了两种消息类型：`Move` 和 `Switch_State`。后者是显而易见的，而 `Move` 消息是由游戏状态中的方法发送的，以移动玩家。因为移动完全依赖于实体状态，所以这是唯一处理这种类型消息并确定状态是否适合运动的系统。

接下来，让我们看看 `Update` 方法：

```cpp
void S_State::Update(float l_dT){
  EntityManager* entities = m_systemManager->GetEntityManager();
  for(auto &entity : m_entities){
    C_State* state = entities->
      GetComponent<C_State>(entity, Component::State);
    if(state->GetState() == EntityState::Walking){
      Message msg((MessageType)EntityMessage::Is_Moving);
      msg.m_receiver = entity;
      m_systemManager->GetMessageHandler()->Dispatch(msg);
    }
  }
}
```

这里发生的一切只是一个简单的检查实体当前的状态。如果它在运动中，就会分发一个 `Is_Moving` 消息。如果你还记得，这种类型的消息是由运动系统处理的，当实体变为空闲时，它会触发一个事件。这个事件由我们的状态系统处理：

```cpp
void S_State::HandleEvent(const EntityId& l_entity,
  const EntityEvent& l_event)
{
  switch(l_event){
  case EntityEvent::Became_Idle:
    ChangeState(l_entity,EntityState::Idle,false);
    break;
  }
}
```

它所做的只是调用一个私有方法 `ChangeState`，该方法将实体的当前状态更改为 `Idle`。这里的第三个参数只是一个标志，用于指示状态更改是否应该被强制执行。

我们在这里将要处理的最后一个公共方法是 `Notify`：

```cpp
void S_State::Notify(const Message& l_message){
  if (!HasEntity(l_message.m_receiver)){ return; }
  EntityMessage m = (EntityMessage)l_message.m_type;
  switch(m){
  case EntityMessage::Move:
    {
      C_State* state = m_systemManager->GetEntityManager()->
        GetComponent<C_State>(l_message.m_receiver,
        Component::State);
      if (state->GetState() == EntityState::Dying){ return; }
      EntityEvent e;
      if (l_message.m_int == (int)Direction::Up){
        e = EntityEvent::Moving_Up;
      } else if (l_message.m_int == (int)Direction::Down){
        e = EntityEvent::Moving_Down;
      } else if(l_message.m_int == (int)Direction::Left){
        e = EntityEvent::Moving_Left;
      } else if (l_message.m_int == (int)Direction::Right){
        e = EntityEvent::Moving_Right;
      }

      m_systemManager->AddEvent(l_message.m_receiver, (EventID)e);
      ChangeState(l_message.m_receiver,
        EntityState::Walking,false);
    }
    break;
  case EntityMessage::Switch_State: 
    ChangeState(l_message.m_receiver,
      (EntityState)l_message.m_int,false);
    break;
  }
}
```

`Move` 消息通过获取目标实体的状态来处理。如果实体没有死亡，就会根据消息包含的方向构建一个 `Moving_X` 事件。一旦事件被分发，实体的状态就会更改为 `Walking`。

`Switch_State` 消息只是通过调用这个私有方法来更改实体的当前状态，而不进行强制更改：

```cpp
void S_State::ChangeState(const EntityId& l_entity, 
  const EntityState& l_state, const bool& l_force)
{
  EntityManager* entities = m_systemManager->GetEntityManager();
  C_State* state = entities->
    GetComponent<C_State>(l_entity, Component::State);
  if (!l_force && state->GetState() == EntityState::Dying){
    return;
  }
  state->SetState(l_state);
  Message msg((MessageType)EntityMessage::State_Changed);
  msg.m_receiver = l_entity;
  msg.m_int = (int)l_state;
  m_systemManager->GetMessageHandler()->Dispatch(msg);
}
```

在获得状态后，检查 `l_force` 标志。如果它设置为 `false`，只有当实体当前不是 `DYING` 时，状态才会被更改。我们不希望任何东西随机地将实体从死亡中拉出来。如果 `l_force` 标志设置为 `true`，则无论是否更改状态。

现在我们可以根据实体的当前状态控制可能发生的事情。有了这个，实体现在就可以被控制了。

## 实体控制器

让一个独立的系统负责移动实体的想法不仅在于我们可以决定哪些实体可以被移动，而且还进一步分离了逻辑，并为未来的 A.I. 实现提供了钩子。让我们看看控制器组件：

```cpp
class C_Controller : public C_Base{
public:
    C_Controller() : C_Base(COMPONENT_CONTROLLER){}
    void ReadIn(std::stringstream& l_stream){}
};
```

是的，它只是一个空组件，它只是用作告诉控制系统它所属的实体可以被控制的一种方式。它可能需要存储一些额外的信息，但到目前为止，它只是一个“标志”。

实际的控制系统非常简单易实现。让我们从构造函数开始：

```cpp
S_Control::S_Control(SystemManager* l_systemMgr)
  :S_Base(System::Control,l_systemMgr)
{
  Bitmask req;
  req.TurnOnBit((unsigned int)Component::Position);
  req.TurnOnBit((unsigned int)Component::Movable);
  req.TurnOnBit((unsigned int)Component::Controller);
  m_requiredComponents.push_back(req);
  req.Clear();
}
```

它对位置、可移动和控制器组件提出了要求，以便能够移动实体，这正是这个系统的唯一目的。实际的移动由处理实体事件来处理，如下所示：

```cpp
void S_Control::HandleEvent(const EntityId& l_entity, 
  const EntityEvent& l_event)
{
  switch(l_event){
  case EntityEvent::Moving_Left:
    MoveEntity(l_entity,Direction::Left); break;
  case EntityEvent::Moving_Right:
    MoveEntity(l_entity, Direction::Right); break;
  case EntityEvent::Moving_Up:
    MoveEntity(l_entity, Direction::Up); break;
  case EntityEvent::Moving_Down:
    MoveEntity(l_entity, Direction::Down); break;
  }
}
```

所有四个事件都会调用同一个私有方法，该方法只是调用可移动组件的 `Move` 方法，并传入适当的方向：

```cpp
void S_Control::MoveEntity(const EntityId& l_entity, 
  const Direction& l_dir)
{
  C_Movable* mov = m_systemManager->GetEntityManager()->
    GetComponent<C_Movable>(l_entity, Component::Movable);
  mov->Move(l_dir);
}
```

在向我们的代码库添加了这个谦虚的补充之后，我们终于可以用键盘移动玩家了：

![实体控制器](img/B04284_09_02.jpg)

现在唯一的问题是实体看起来像是在冰上滑行，这是由于完全缺乏动画。为了解决这个问题，必须引入动画系统。

# 动画实体

如果您回忆起前面的章节，我们构建的 `SpriteSheet` 类已经对动画有很好的支持。在这个阶段没有必要添加这个功能，尤其是我们只处理基于精灵图的图形。这为我们节省了大量时间，并允许精灵图动画由一个单独的系统处理，无需额外的组件开销。

让我们开始实现精灵图动画系统，就像往常一样，先处理构造函数：

```cpp
S_SheetAnimation::S_SheetAnimation(SystemManager* l_systemMgr)
  : S_Base(System::SheetAnimation,l_systemMgr)
{
  Bitmask req;
  req.TurnOnBit((unsigned int)Component::SpriteSheet);
  req.TurnOnBit((unsigned int)Component::State);
  m_requiredComponents.push_back(req);

  m_systemManager->GetMessageHandler()->
    Subscribe(EntityMessage::State_Changed,this);
}
```

由于实体动画到目前为止完全是基于状态的，因此这个系统需要一个状态组件，除了精灵图组件之外。它还订阅了 `State_Changed` 消息类型，以便通过播放适当的动画来响应状态变化。更新所有实体是这个系统逻辑最多的区域，所以让我们看看 `Update` 方法：

```cpp
void S_SheetAnimation::Update(float l_dT){
  EntityManager* entities = m_systemManager->GetEntityManager();
  for(auto &entity : m_entities){
    C_SpriteSheet* sheet = entities->
      GetComponent<C_SpriteSheet>(entity, Component::SpriteSheet);
    C_State* state = entities->
      GetComponent<C_State>(entity, Component::State);

    sheet->GetSpriteSheet()->Update(l_dT);

    const std::string& animName = sheet->
      GetSpriteSheet()->GetCurrentAnim()->GetName();
    if(animName == "Attack"){
      if(!sheet->GetSpriteSheet()->GetCurrentAnim()->IsPlaying())
      {
        Message msg((MessageType)EntityMessage::Switch_State);
        msg.m_receiver = entity;
        msg.m_int = (int)EntityState::Idle;
        m_systemManager->GetMessageHandler()->Dispatch(msg);
      } else if(sheet->GetSpriteSheet()->GetCurrentAnim()->IsInAction())
      {
        Message msg((MessageType)EntityMessage::Attack_Action);
        msg.m_sender = entity;
        m_systemManager->GetMessageHandler()->Dispatch(msg);
      }
    } else if(animName == "Death" &&
      !sheet->GetSpriteSheet()->GetCurrentAnim()->IsPlaying())
    {
      Message msg((MessageType)EntityMessage::Dead);
      msg.m_receiver = entity;
      m_systemManager->GetMessageHandler()->Dispatch(msg);
    }
  }
}
```

首先，获取精灵图和状态组件。然后更新精灵图并检索当前动画的名称。如果攻击动画不再播放，则发送一个 `Switch_State` 类型的消息，以便将实体放回 `Idle` 状态。否则，检查动画是否当前处于精灵图文件中指定的 "action" 帧范围内。如果是，向当前实体发送一个 `Attack_Action` 消息，稍后不同的系统可以使用它来实现战斗。另一方面，如果死亡动画已经结束，则发送一个 `Dead` 消息。

接下来，让我们处理消息：

```cpp
void S_SheetAnimation::Notify(const Message& l_message){
  if(HasEntity(l_message.m_receiver)){
    EntityMessage m = (EntityMessage)l_message.m_type;
    switch(m){
    case EntityMessage::State_Changed:
      {
        EntityState s = (EntityState)l_message.m_int;
        switch(s){
        case EntityState::Idle:
          ChangeAnimation(l_message.m_receiver,"Idle",true,true);
          break;
        case EntityState::Walking:
          ChangeAnimation(l_message.m_receiver,"Walk",true,true);
          break;
        case EntityState::Attacking:
          ChangeAnimation(l_message.m_receiver,
            "Attack",true,false);
          break;
        case EntityState::Hurt: break;
        case EntityState::Dying:
          ChangeAnimation(l_message.m_receiver,
            "Death",true,false);
          break;
        }
      }
      break;
    }
  }
}
```

这个系统可能感兴趣的任何消息都与特定实体有关，所以首先进行这个检查。目前，我们只处理一种消息类型：`State_Changed`。每次状态改变时，我们都会改变实体的动画。唯一的可能例外是 `Hurt` 状态，稍后我们会处理它。

我们需要的最后一段代码是私有的 `ChangeAnimation` 方法：

```cpp
void S_SheetAnimation::ChangeAnimation(const EntityId& l_entity, 
  const std::string& l_anim, bool l_play, bool l_loop)
{
  C_SpriteSheet* sheet = m_systemManager->GetEntityManager()->
    GetComponent<C_SpriteSheet>(l_entity,Component::SpriteSheet);
  sheet->GetSpriteSheet()->SetAnimation(l_anim,l_play,l_loop);
}
```

获取实体的精灵图组件后，它简单地调用其 `SetAnimation` 方法来更改正在播放的当前动画。这段代码足够冗余，值得有一个单独的方法。

编译成功后，我们可以看到我们的实体现在已经开始动画了：

![动画实体](img/B04284_09_03.jpg)

# 处理碰撞

让实体相互碰撞，以及进入我们将要构建的所有茂密环境，是一种机制，没有这种机制，大多数游戏将无法运行。为了实现这一点，这些在屏幕上四处移动的动画图像必须有一个表示其固体的组件。边界框在过去为我们工作得很好，所以让我们坚持使用它们，并开始构建可碰撞体组件：

```cpp
enum class Origin{ Top_Left, Abs_Centre, Mid_Bottom };

class C_Collidable : public C_Base{
public:
    ...
private:
    sf::FloatRect m_AABB;
    sf::Vector2f m_offset;
    Origin m_origin;

    bool m_collidingOnX;
    bool m_collidingOnY;
};
```

每个可碰撞实体都必须有一个表示其实体固体部分的边界框。这正是`m_AABB`矩形发挥作用的地方。除此之外，边界框本身可以根据实体的类型偏移一定数量的像素，并且可以有不同的起点。最后，我们想要跟踪实体是否在给定的轴上发生碰撞，这需要使用`m_collidingOnX`和`m_collidingOnY`标志。

这个组件的构造函数可能看起来有点像这样：

```cpp
C_Collidable(): C_Base(Component::Collidable), 
  m_origin(Origin::Mid_Bottom), m_collidingOnX(false),
  m_collidingOnY(false)
{}
```

在将默认值初始化到其一些数据成员之后，这个组件，就像许多其他组件一样，需要有一种反序列化的方式：

```cpp
void ReadIn(std::stringstream& l_stream){
    unsigned int origin = 0;
    l_stream >> m_AABB.width >> m_AABB.height >> m_offset.x>> m_offset.y >> origin;
    m_origin = (Origin)origin;
}
```

这里有一些独特的设置器和获取器方法，我们将使用它们：

```cpp
void CollideOnX(){ m_collidingOnX = true; }
void CollideOnY(){ m_collidingOnY = true; }
void ResetCollisionFlags(){
    m_collidingOnX = false;
    m_collidingOnY = false;
}
void SetSize(const sf::Vector2f& l_vec){
    m_AABB.width      = l_vec.x;
    m_AABB.height     = l_vec.y;
}
```

最后，我们来到了这个组件的关键方法，`SetPosition`：

```cpp
void SetPosition(const sf::Vector2f& l_vec){
   switch(m_origin){
   case(Origin::Top_Left):
      m_AABB.left = l_vec.x + m_offset.x;
      m_AABB.top  = l_vec.y + m_offset.y;
      break;
   case(Origin::Abs_Centre):
      m_AABB.left = l_vec.x - (m_AABB.width / 2) + m_offset.x;
      m_AABB.top  = l_vec.y - (m_AABB.height / 2) + m_offset.y;
      break;
   case(Origin::Mid_Bottom):
      m_AABB.left = l_vec.x - (m_AABB.width / 2) + m_offset.x;
      m_AABB.top  = l_vec.y - m_AABB.height + m_offset.y;
      break;
   }
}
```

为了支持不同类型的起点，边界框矩形的定位必须不同。考虑以下插图：

![处理碰撞](img/B04284_09_04.jpg)

实际的边界框矩形的起点始终是左上角。为了正确定位它，我们使用其宽度和高度来补偿几种可能的起点类型之间的差异。

## 碰撞系统

实际的碰撞魔法只有在有了负责计算游戏中每个可碰撞体系统的系统之后才会开始。让我们首先看看在这个系统中将要使用的数据类型：

```cpp
struct CollisionElement{
  CollisionElement(float l_area, TileInfo* l_info,
    const sf::FloatRect& l_bounds):m_area(l_area),
    m_tile(l_info), m_tileBounds(l_bounds){}
  float m_area;
  TileInfo* m_tile;
  sf::FloatRect m_tileBounds;
};

using Collisions = std::vector<CollisionElement>;
```

为了进行适当的碰撞检测和响应，我们还需要一个能够存储碰撞信息的数据结构，这些信息可以稍后进行排序和处理。为此，我们将使用`CollisionElement`数据类型的向量。它是一个结构，由一个表示碰撞面积的浮点数、一个指向`TileInfo`实例的指针（该实例携带有关瓦片的所有信息）和一个简单的浮点矩形组成，该矩形包含地图瓦片的边界框信息。

为了检测实体和瓦片之间的碰撞，碰撞系统需要能够访问一个`Map`实例。了解所有这些后，让我们开始实现这个类！

### 实现碰撞系统

和往常一样，我们将在类的构造函数中设置组件要求：

```cpp
S_Collision::S_Collision(SystemManager* l_systemMgr) 
  :S_Base(System::Collision,l_systemMgr)
{
  Bitmask req;
  req.TurnOnBit((unsigned int)Component::Position);
  req.TurnOnBit((unsigned int)Component::Collidable);
  m_requiredComponents.push_back(req);
  req.Clear();

  m_gameMap = nullptr;
}
```

如您所见，该系统对实体施加了位置和可碰撞组件的要求。其`m_gameMap`数据成员也被初始化为`nullptr`，直到通过使用此方法进行设置：

```cpp
void S_Collision::SetMap(Map* l_map){ m_gameMap = l_map; }
```

接下来是那个非常常见的更新方法，它使一切行为如预期：

```cpp
void S_Collision::Update(float l_dT){
  if (!m_gameMap){ return; }
  EntityManager* entities = m_systemManager->GetEntityManager();
  for(auto &entity : m_entities){
    C_Position* position = entities->
      GetComponent<C_Position>(entity, Component::Position);
    C_Collidable* collidable = entities->
      GetComponent<C_Collidable>(entity, Component::Collidable);
    collidable->SetPosition(position->GetPosition());
    collidable->ResetCollisionFlags();
    CheckOutOfBounds(position, collidable);
    MapCollisions(entity, position, collidable);
  }
  EntityCollisions();
}
```

为了清晰起见，更新方法使用了另外两个辅助方法：`CheckOutOfBounds` 和 `MapCollisions`。在遍历所有可碰撞实体时，该系统获取其实体位置和可碰撞组件。后者使用实体的最新位置进行更新。同时，它的布尔碰撞标志也被重置。在所有实体都被更新后，私有的 `EntityCollisions` 方法被调用以处理实体与实体之间的交点测试。注意这个方法的开始部分。如果地图实例没有正确设置，它将立即返回。

首先，检查实体是否位于我们地图的边界之外：

```cpp
void S_Collision::CheckOutOfBounds(C_Position* l_pos,
  C_Collidable* l_col)
{
  unsigned int TileSize = m_gameMap->GetTileSize();

  if (l_pos->GetPosition().x < 0){
    l_pos->SetPosition(0.0f, l_pos->GetPosition().y);
    l_col->SetPosition(l_pos->GetPosition());
  } else if (l_pos->GetPosition().x >
    m_gameMap->GetMapSize().x * TileSize)
  {
    l_pos->SetPosition(m_gameMap->GetMapSize().x * TileSize,
      l_pos->GetPosition().y);
    l_col->SetPosition(l_pos->GetPosition());
  }

  if (l_pos->GetPosition().y < 0){
    l_pos->SetPosition(l_pos->GetPosition().x, 0.0f);
    l_col->SetPosition(l_pos->GetPosition());
  } else if (l_pos->GetPosition().y >
    m_gameMap->GetMapSize().y * TileSize)
  {
    l_pos->SetPosition(l_pos->GetPosition().x,
      m_gameMap->GetMapSize().y * TileSize);
    l_col->SetPosition(l_pos->GetPosition());
  }
}
```

如果实体意外地位于地图之外，其位置将被重置。

在这个阶段，我们开始运行实体与地砖的碰撞测试：

```cpp
void S_Collision::MapCollisions(const EntityId& l_entity,
  C_Position* l_pos, C_Collidable* l_col)
{
  unsigned int TileSize = m_gameMap->GetTileSize();
  Collisions c;

  sf::FloatRect EntityAABB = l_col->GetCollidable();
  int FromX = floor(EntityAABB.left / TileSize);
  int ToX = floor((EntityAABB.left + EntityAABB.width)/TileSize);
  int FromY = floor(EntityAABB.top / TileSize);
  int ToY = floor((EntityAABB.top + EntityAABB.height)/TileSize);
  ...
}
```

设置了一个名为 `c` 的碰撞信息向量。它将包含实体碰撞的所有重要信息，碰撞区域的尺寸以及它所碰撞的地砖的属性。然后从可碰撞组件中获取实体的边界框。根据该边界框计算出一个要检查的坐标范围，如下所示：

![实现碰撞系统](img/B04284_09_05.jpg)

这些坐标立即被使用，因为我们开始遍历计算出的地砖范围，检查碰撞：

```cpp
for (int x = FromX; x <= ToX; ++x){
  for (int y = FromY; y <= ToY; ++y){
    for (int l = 0; l < Sheet::Num_Layers; ++l){
      Tile* t = m_gameMap->GetTile(x, y, l);
      if (!t){ continue; }
      if (!t->m_solid){ continue; }
      sf::FloatRect TileAABB(x*TileSize, y*TileSize,TileSize, TileSize);
      sf::FloatRect Intersection;
      EntityAABB.intersects(TileAABB, Intersection);
      float S = Intersection.width * Intersection.height;
      c.emplace_back(S, t->m_properties, TileAABB);
      break;
    }
  }
}
```

一旦遇到一个固体地砖，就收集其边界框、地砖信息和交点区域细节，并将它们插入到向量 `c` 中。如果检测到固体地砖，则必须停止层循环，否则碰撞检测可能无法正常工作。

在找到计算范围内实体碰撞的所有固体之后，它们都必须进行排序：

```cpp
if (c.empty()){ return; }
std::sort(c.begin(), c.end(),
  [](CollisionElement& l_1, CollisionElement& l_2){
    return l_1.m_area > l_2.m_area;
});
```

排序后，我们最终可以开始解决碰撞：

```cpp
for (auto &col : c){
  EntityAABB = l_col->GetCollidable();
  if (!EntityAABB.intersects(col.m_tileBounds)){ continue; }
  float xDiff = (EntityAABB.left + (EntityAABB.width / 2)) -
    (col.m_tileBounds.left + (col.m_tileBounds.width / 2));
  float yDiff = (EntityAABB.top + (EntityAABB.height / 2)) -
    (col.m_tileBounds.top + (col.m_tileBounds.height / 2));
  float resolve = 0;
  if (std::abs(xDiff) > std::abs(yDiff)){
    if (xDiff > 0){
      resolve = (col.m_tileBounds.left + TileSize) -
        EntityAABB.left;
    } else {
      resolve = -((EntityAABB.left + EntityAABB.width) -
        col.m_tileBounds.left);
    }
    l_pos->MoveBy(resolve, 0);
    l_col->SetPosition(l_pos->GetPosition());
    m_systemManager->AddEvent(l_entity,
      (EventID)EntityEvent::Colliding_X);
    l_col->CollideOnX();
  } else {
    if (yDiff > 0){
      resolve = (col.m_tileBounds.top + TileSize) -
        EntityAABB.top;
    } else {
      resolve = -((EntityAABB.top + EntityAABB.height) -
        col.m_tileBounds.top);
    }
    l_pos->MoveBy(0, resolve);
    l_col->SetPosition(l_pos->GetPosition());
    m_systemManager->AddEvent(l_entity,
      (EventID)EntityEvent::Colliding_Y);
    l_col->CollideOnY();
  }
}
```

由于解决一个碰撞可能会解决另一个碰撞，因此在承诺解决碰撞之前，必须检查实体的边界框是否存在交点。实际的解决方法与第七章中描述的几乎相同，即*重新发现火焰 – 常见游戏设计元素*。

一旦计算了解决细节，位置组件就会根据它移动。可碰撞组件也必须在这里更新，否则它可能会被多次解决并错误地移动。最后需要关注的是向实体的事件队列中添加碰撞事件，并在可碰撞组件中调用 `CollideOnX` 或 `CollideOnY` 方法来更新其标志。

接下来是实体与实体之间的碰撞：

```cpp
void S_Collision::EntityCollisions(){
  EntityManager* entities = m_systemManager->GetEntityManager();
  for(auto itr = m_entities.begin();
    itr != m_entities.end(); ++itr)
  {
    for(auto itr2 = std::next(itr);
      itr2 != m_entities.end(); ++itr2){
      C_Collidable* collidable1 = entities->
        GetComponent<C_Collidable>(*itr, Component::Collidable);
      C_Collidable* collidable2 = entities->
        GetComponent<C_Collidable>(*itr2, Component::Collidable);
      if(collidable1->GetCollidable().intersects(
        collidable2->GetCollidable()))
      {
        // Entity-on-entity collision!
      }
    }
  }
}
```

此方法通过使用 SFML 矩形类提供的`intersects`方法，将所有实体与其余所有实体的边界框进行碰撞检查。目前，我们不必担心对这些类型的碰撞做出响应，然而，我们将在未来的章节中使用这个功能。

最后，就像其移动对应物一样，碰撞系统需要一个指向`Map`类的指针，所以让我们在游戏状态的`OnCreate`方法中给它一个：

```cpp
void State_Game::OnCreate(){
  ...
  m_stateMgr->GetContext()->m_systemManager->
    GetSystem<S_Collision>(SYSTEM_COLLISION)->SetMap(m_gameMap);
  ...
}
```

以下代码片段为碰撞系统提供了所有所需的权力，以防止实体穿过固体瓷砖，如下所示：

![实现碰撞系统](img/B04284_09_06.jpg)

# 摘要

在完成本章内容后，我们成功摆脱了基于继承的实体设计，并通过一种更加模块化的方法强化了我们的代码库，从而避免了组合留下的许多陷阱。链条的强度仅取决于其最薄弱的环节，而现在我们可以放心，实体部分将稳固。

在接下来的两章中，我们将讨论如何通过添加 GUI 系统以及添加一些不同类型的元素，管理它们的事件，并为它们提供图形自定义的空间，来使游戏更加互动和用户友好。那里见！
