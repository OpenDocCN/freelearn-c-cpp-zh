# 第六章. 使用多线程提升代码性能

在本章中，我们将学习以下技能：

+   如何并行运行程序中的多个部分

+   如何保护内存访问以避免数据竞争

+   如何将这些功能集成到 Gravitris 中

在本章结束时，你将能够通过以智能的方式暂停你的代码来利用计算机 CPU 提供的所有功能。但首先，让我们描述一下理论。

# 什么是多线程？

在计算机科学中，一个软件可以被看作是一个有起始点和退出点的流。每个软件都以 C/C++中的`main()`函数开始其生命周期。这是你程序的入口点。直到这一点，你可以做任何你想做的事情；包括创建新的例程流，克隆整个软件，并启动另一个程序。所有这些示例的共同点是都创建了一个新的流，并且它们有自己独立的生命周期，但它们并不等价。

## fork()函数

这种功能相当简单。调用`fork()`将复制你的整个运行进程到一个新的进程中。新创建的进程与其父进程完全分离（新的 PID，新的内存区域作为其父进程的精确副本），并在`fork()`调用后立即开始。`fork()`函数的返回值是两次执行之间的唯一区别。

以下是一个`fork()`函数的示例：

```cpp
int main()
{
  int pid = fork();
  if(pid == -1)
    std::cerr<<"Error when calling fork()"<<std::endl;
  else if (pid == 0)
    std::cout<<"I'm the child process"<<std::endl;
  else
    std::cout<<"I'm the parent process"<<std::endl;
  return 0;
}
```

如你所见，使用它非常简单，但也有一些使用上的限制。其中最重要的一个与内存共享有关。因为每个进程都有自己的内存区域，所以你无法在它们之间共享一些变量。一个解决方案是使用文件作为套接字、管道等。此外，如果父进程死亡，子进程仍将继续其自己的生命周期，而不会关注其父进程。

因此，这个解决方案只有在你不希望在不同的执行之间共享任何内容，甚至包括它们的状态时才有兴趣。

## exec()族函数

`exec()`族函数（`execl()`, `execlp()`, `execle()`, `execv()`, `execvp()`, `execvpe()`）将用另一个程序替换整个运行程序。当与`fork()`函数结合使用时，这些函数变得非常强大。以下是一个这些函数的示例：

```cpp
int main()
{
  int pid = fork();
  if(pid == -1)
  =  std::cerr<<"Error when calling fork()"<<std::endl;
  else if (pid == 0) {
    std::cout<<"I'm the child process"<<std::endl;
  }
  else {
    std::cout<<"I'm the parent process"<<std::endl;
    execlp("Gravitris", "Gravitris", "arg 1", "arg 2",NULL);
    std::cout<<"This message will never be print, except if execl() fail"<<std::endl;
  }
  return 0;
}
```

这个简短的小代码片段将创建两个不同的进程，如前所述。然后，子进程将被 Gravitris 的一个实例替换。由于`exec()`族函数中的任何调用都会用一个新的流替换整个运行流，所以`exec`调用下的所有代码都不会执行，除非发生错误。

# 线程功能

现在，我们将讨论线程。线程的功能与`fork`功能非常相似，但有一些重要的区别。一个线程将为你的运行进程创建一个新的流。它的起点是一个作为参数指定的函数。线程也将与其父进程在相同的环境中执行。主要影响是内存是相同的，但这不是唯一的一个。如果父进程死亡，所有它的线程也会死亡。

如果你不知道如何处理这些问题，这两个点可能会成为一个问题。让我们以并发内存访问为例。

假设你在程序中有一个名为`var`的全局变量。然后主进程将创建一个线程。这个线程将写入`var`，同时主进程也可以写入它。这将导致未定义的行为。有几种不同的解决方案可以避免这种行为，其中常见的一种是使用互斥锁来锁定对这个变量的访问。

简单来说，互斥锁是一个令牌。我们可以尝试获取（锁定）它或释放它（解锁）。如果有多个进程同时想要锁定它，第一个进程将有效地锁定它，第二个进程将等待第一个进程调用互斥锁的解锁函数。总结一下，如果你想通过多个线程访问共享变量，你必须为它创建一个互斥锁。然后，每次你想访问它时，锁定互斥锁，访问变量，最后解锁互斥锁。使用这种解决方案，你可以确保不会发生任何数据损坏。

第二个问题涉及到你的线程执行结束与主进程同步的问题。实际上，这个问题有一个简单的解决方案。在主流的末尾，你需要等待所有正在运行的线程结束。只要还有线程存活，流就会被阻塞，因此不会死亡。

下面是一个使用线程功能的示例：

```cpp
#include <SFML/System.hpp>
static sf::Mutex mutex;
static int i = 0;

void f()
{
  sf::Lock guard(mutex);
  std::cout<<"Hello world"<<std::endl;
  std::cout<<"The value of i is "<<(++i)<<" from f()"<<std::endl;
}

int main()
{
  sf::Thread thread(f);
  thread.launch();
  mutex.lock();
  std::cout<<"The value of i is "<<(++i)<<" from main"<<std::endl;
  mutex.unlock();
  thread.wait();
  return 0;
}
```

既然理论已经解释了，让我们解释一下使用多线程的动机是什么。

## 为什么我们需要使用线程功能？

现在，一般的计算机都有一个能够同时处理多个线程的 CPU。大多数情况下，CPU 中有 4-12 个计算单元。这些单元中的每一个都能够独立于其他单元完成任务。

让我们假设你的 CPU 只有四个计算单元。

如果你以我们之前的游戏为例，所有的工作都是在单个线程中完成的。所以只有四个核心中的一个被使用。这是很遗憾的，因为所有的工作都是由一个组件完成的，而其他的组件则没有被使用。我们可以通过将代码分成几个部分来改进这一点。每个部分将在不同的线程中执行，工作将在它们之间共享。然后，不同的线程将在不同的核心上执行（在我们的例子中最多四个）。所以现在工作是在并行中完成的。

创建多个线程可以让您利用计算机提供的所有功能，让您有更多时间专注于某些功能，如人工智能。

另一种用法是当您使用一些阻塞函数时，例如等待网络消息、播放音乐等。这里的问题是运行中的进程将等待某事，无法继续执行。为了处理这个问题，您可以简单地创建一个线程并将任务委托给它。这正是 `sf::Music` 的工作方式。有一个内部线程用于播放音乐。这也是为什么我们在播放声音或音乐时游戏不会冻结的原因。每次为这个任务创建线程时，它对用户来说都是透明的。现在理论已经解释清楚，让我们将其应用于实践。

## 使用线程

在第四章中，我们介绍了物理到我们的游戏中。为了这个功能，我们创建了两个游戏循环：一个用于逻辑，另一个用于物理。到目前为止，物理循环和其他循环的执行是在同一个进程中进行的。现在，是时候将它们的执行分离到不同的线程中去了。

我们需要创建一个线程，并使用 `Mutex` 类来保护我们的变量。有两种选择：

+   使用标准库中的对象

+   使用 SFML 库中的对象

这里是一个总结所需功能和从标准 C++ 库到 SFML 转换的表格。

`thread` 类：

| 库 | 头文件 | 类 | 启动 | 等待 |
| --- | --- | --- | --- | --- |
| C++ | `<thread>` | `std::thread` | 构造后直接 | `::join()` |
| SFML | `<SFML/System.hpp>` | `sf::Thread` | `::launch()` | `::wait()` |

`mutex` 类：

| 库 | 头文件 | 类 | 锁定 | 解锁 |
| --- | --- | --- | --- | --- |
| C++ | `<mutex>` | `std::mutex` | `::lock()` | `::unlock()` |
| SFML | `<SFML/System.hpp>` | `sf::Mutex` | `::lock()` | `::unlock()` |

还有一个可以使用的第三种类。它会在构造时自动调用 `mutex::lock()`，并在析构时调用 `mutex::unlock()`，遵循 RAII 习惯。这个类被称为锁或保护器。它的使用很简单，用 mutex 作为参数来构造它，它将自动锁定/解锁。以下表格解释了这个类的详细信息：

| 库 | 头文件 | 类 | 构造函数 |
| --- | --- | --- | --- |
| C++ | `<mutex>` | `std::lock_guard` | `std::lock_guard(std::mutex&)` |
| SFML | `<SFML/System.hpp>` | `sf::Lock` | `sf::Lock(sf::Mutex&)` |

如您所见，这两个库提供了相同的功能。`thread` 类的 API 有一些变化，但并不重要。

在这本书中，我将使用 SFML 库。选择这个库没有真正的理由，只是因为它让我能够向您展示更多 SFML 的可能性。

现在已经介绍了这个类，让我们回到之前的例子，并按照以下方式应用我们的新技能：

```cpp
#include <SFML/System.hpp>
static sf::Mutex mutex;
static int i = 0;

void f()
{
  sf::Lock guard(mutex);
  std::cout<<"Hello world"<<std::endl;
  std::cout<<"The value of i is "<<(++i)<<" from f()"<<std::endl;
}

int main()
{
  sf::Thread thread(f);
  thread.launch();
  mutex.lock();
  std::cout<<"The value of i is "<<(++i)<<" from main"<<std::endl;
  mutex.unlock();
  thread.wait();
  return 0;
}
```

在这个简单的例子中，有几个部分。第一部分初始化全局变量。然后，我们创建一个名为`f()`的函数，它打印**"Hello world"**，然后打印另一条消息。在`main()`函数中，我们创建一个与`f()`函数关联的线程，启动它，并打印`i`的值。每次，我们使用互斥锁（使用了两种不同的方法）来保护对共享变量`i`的访问。

来自`f()`函数的打印消息是不可预测的。它可能是**"来自 f()的 i 的值是 1"**或**"来自 f()的 i 的值是 2"**。我们无法确定`f()`或`main()`哪个先打印，因此不知道将打印的值。我们唯一确定的是，没有对`i`的并发访问，并且线程将在`main()`函数之前结束，这要归功于`thread.wait()`调用。

现在我们已经解释并展示了所需的类，让我们修改我们的游戏以使用它们。

# 将多线程添加到我们的游戏中

现在，我们将修改我们的 Gravitris 以使物理计算从程序的其他部分中瘫痪。我们只需要更改两个文件：`Game.hpp`和`Game.cpp`。

在头文件中，我们不仅需要添加所需的头文件，还需要更改`update_physics()`函数的原型，并最终给类添加一些属性。所以以下是需要遵循的不同步骤：

1.  添加`#include <SFML/System.hpp>`，这将允许我们访问所有需要的类。

1.  然后，更改以下代码片段：

    ```cpp
    void updatePhysics(const sf::Time& deltaTime,const sf::Time& timePerFrame);
    ```

    to:

    ```cpp
    void updatePhysics();
    ```

    原因在于一个线程无法向其包装的函数传递任何参数，因此我们将使用另一种解决方案：成员变量。

1.  将以下变量添加到`Game`类中作为私有变量：

    ```cpp
    sf::Thread _physicsThread;
    sf::Mutex _mutex;
    bool _isRunning;
    int _physicsFramePerSeconds;
    ```

    所有这些变量都将由物理线程使用，`_mutex`变量将确保不会对这些变量之一进行并发访问。出于相同的原因，我们还需要保护对`_world`变量的访问。

1.  现在头文件包含了所有要求，让我们转向实现部分。

1.  首先，我们不仅需要更新构造函数以初始化`_physicsThread`和`_isRunning`变量，还需要保护对`_world`的访问。

    ```cpp
    Game::Game(int X, int Y,int word_x,int word_y) : ActionTarget(Configuration::player_inputs), _window(sf::VideoMode(X,Y),"06_Multithreading"), _current_piece(nullptr), _world(word_x,word_y), _mainMenu(_window),_configurationMenu(_window), _pauseMenu(_window), _status(Status::StatusMainMenu), _physicsThread(&Game::update_physics,this), _isRunning(true)
    {
      bind(Configuration::PlayerInputs::HardDrop,this{
          sf::Lock lock(_mutex);
          _current_piece = _world.newPiece();
          timeSinceLastFall = sf::Time::Zero;
      });
    }
    ```

1.  在构造函数中，我们不仅需要初始化新的成员变量，还需要保护在其中一个回调中使用的`_world`变量。这个锁非常重要，以确保在执行过程中不会随机发生数据竞争。

1.  现在构造函数已经更新，我们需要更改`run()`函数。目标是运行物理线程。需要做的更改不多。请自己看看：

    ```cpp
    void Game::run(int minimum_frame_per_seconds, int physics_frame_per_seconds)
    {
      sf::Clock clock;
      const sf::Time timePerFrame = sf::seconds(1.f/minimum_frame_per_seconds);
      const sf::Time timePerFramePhysics = sf::seconds(1.f/physics_frame_per_seconds);
      _physics_frame_per_seconds = physics_frame_per_seconds;
      _physicsThread.launch();

      while (_window.isOpen())
      {
        sf::Time time = clock.restart();
        processEvents();
        if(_status == StatusGame and not _stats.isGameOver()){
          updatePhysics(time,timePerFramePhysics);
          update(time,timePerFrame);
        }
        render();
      }
      _isRunning = false;
      _physicsThread.wait();
    }
    ```

1.  现在主游戏循环已经更新，我们需要在`update()`方法中进行一个小改动以保护成员`_world`变量。

    ```cpp
    void Game::update(const sf::Time& deltaTime,const sf::Time& timePerFrame)
    {
      static sf::Time timeSinceLastUpdate = sf::Time::Zero;
      timeSinceLastUpdate+=deltaTime;
      timeSinceLastFall+=deltaTime;
      if(timeSinceLastUpdate > timePerFrame)
      {
        sf::Lock lock(_mutex);
        if(_current_piece != nullptr)
        {
          _currentPiece->rotate(_rotateDirection*3000);
          _currentPiece->moveX(_moveDirection*5000);
          bool new_piece;
          {
            int old_level =_stats.getLevel();
            _stats.addLines(_world.clearLines(new_piece,*_currentPiece));
            if(_stats.getLevel() != old_level)
            _world.add(Configuration::Sounds::LevelUp);
          }
          if(new_piece or timeSinceLastFall.asSeconds() > std::max(1.0,10-_stats.getLevel()*0.2))
          {
            _current_piece = _world.newPiece();
            timeSinceLastFall = sf::Time::Zero;
          }
        }
        _world.update(timePerFrame);
        _stats.setGameOver(_world.isGameOver());
        timeSinceLastUpdate = sf::Time::Zero;
      }
      _rotateDirection=0;
      _moveDirection=0;
    }
    ```

1.  如您所见，只有一处修改。我们只需要保护 `_world` 变量的访问，仅此而已。现在，我们需要修改 `updatePhysics()` 函数。这个函数将会像以下代码片段所示进行大量修改：

    ```cpp
    void Game::updatePhysics(const sf::Time& deltaTime,const sf::Time& timePerFrame)
    void Game::updatePhysics()
    {
      sf::Clock clock;
      const sf::Time timePerFrame = sf::seconds(1.f/_physics_frame_per_seconds);
      static sf::Time timeSinceLastUpdate = sf::Time::Zero;

      while (_isRunning)
      {
        sf::Lock lock(_mutex);
        timeSinceLastUpdate+=deltaTime;
        timeSinceLastUpdate+= clock.restart();
        _world.updateGravity(_stats.getLevel());

        while (timeSinceLastUpdate > timePerFrame)
        {
          if(_status == StatusGame and not _stats.isGameOver())
          _world.update_physics(timePerFrame);
          timeSinceLastUpdate -= timePerFrame;
        }
      }
    }
    ```

    我们需要更改这个函数的签名，因为我们无法通过线程传递给它一些参数。因此，我们为这个函数添加了一个内部时钟，以及它自己的循环。函数的其余部分遵循在 `update()` 方法中开发的逻辑。当然，我们也使用互斥锁来保护所有使用的变量的访问。现在，物理计算可以独立于游戏的其他部分进行更新。

1.  现在其他使用 `_world` 的函数，如 `initGame()` 和 `render()`，需要做的小改动很少。每次，我们都需要使用互斥锁来锁定这个变量的访问。

1.  关于 `initGame()` 函数的修改如下：

    ```cpp
    void Game::initGame()
    {
      sf::Lock lock(_mutex);
      timeSinceLastFall = sf::Time::Zero;
      _stats.reset();
      _world.reset();
      _current_piece = _world.newPiece();
    }
    ```

1.  现在看看更新后的 `render()` 函数：

    ```cpp
    void Game::render()
    {
      _window.clear();
      switch(_status)
      {
        case StatusMainMenu:
        {
          _window.draw(_mainMenu);
        }break;
        case StatusGame :
        {
          if(not _stats.isGameOver())
          {
            sf::Lock lock(_mutex);
            _window.draw(_world);
          }
          _window.draw(_stats);
        }break;
        case StatusConfiguration:
        {
          _sfg_desktop.Update(0.0);
          _sfgui.Display(_window);
          _window.draw(_configurationMenu);
        }break;
        case StatusPaused :
        {
          if(not _stats.isGameOver())
          {
            sf::Lock lock(_mutex);
            _window.draw(_world);
          }
          _window.draw(_pauseMenu);
        }break;
        default : break;
      }
      _window.display();
    }
    ```

1.  如您所见，所做的更改非常简约，但这是为了避免任何竞态条件。

现在代码中的所有更改都已完成，您应该能够编译项目并测试它。图形结果将保持不变，但 CPU 不同核心的使用方式已经改变。现在，项目使用两个线程而不是一个。第一个线程用于物理计算，另一个线程用于游戏的其他部分。

# 摘要

在本章中，我们介绍了多线程的使用，并将其应用于现有的 Gravitris 项目中。我们学习了这样做的原因，不同的可能用途，以及共享变量的保护。

在我们的实际游戏中，多线程可能有些过度，但在更大型的游戏中，例如有数百玩家、网络和实时策略的情况下，它就变成了**必须的**。

在下一章中，我们将构建一个全新的游戏，并介绍新的内容，如等距视图、组件系统、路径查找等。
