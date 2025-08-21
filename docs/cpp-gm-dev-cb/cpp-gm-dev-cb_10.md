# 第十章：游戏开发中的多线程

在本章中，将涵盖以下配方：

+   游戏中的并发性-创建线程

+   加入和分离线程

+   向线程传递参数

+   避免死锁

+   数据竞争和互斥

+   编写线程安全的类

# 介绍

要理解多线程，让我们首先了解线程的含义。线程是并发执行的单位。它有自己的调用堆栈，用于调用的方法，它们的参数和局部变量。每个应用程序在启动时至少有一个正在运行的线程，即主线程。当我们谈论多线程时，意味着一个进程有许多独立和并发运行的线程，但具有共享内存。通常，多线程与多处理混淆。多处理器有多个运行的进程，每个进程都有自己的线程。

尽管多线程应用程序可能编写起来复杂，但它们是轻量级的。然而，多线程架构不适合分布式应用程序。在游戏中，我们可能有一个或多个线程在运行。关键问题是何时以及为什么应该使用多线程。虽然这是相当主观的，但如果您希望多个任务同时发生，您将使用多线程。因此，如果您不希望游戏中的物理代码或音频代码等待主循环完成处理，您将对物理和音频循环进行多线程处理。

# 游戏中的并发性-创建线程

编写多线程代码的第一步是生成一个线程。在这一点上，我们必须注意应用程序已经运行了一个活动线程，即主线程。因此，当我们生成一个线程时，应用程序中将有两个活动线程。

## 准备工作

要完成这个配方，您需要一台运行 Windows 和 Visual Studio 的计算机。不需要其他先决条件。

## 如何做...

在这个配方中，我们将看到生成线程有多么容易。添加一个名为`Source.cpp`的源文件，并将以下代码添加到其中：

```cpp
int ThreadOne()
{
  std::cout << "I am thread 1" << std::endl;
  return 0;
}

int main()
{
  std::thread T1(ThreadOne);

  if (T1.joinable()) // Check if can be joined to the main thread
    T1.join();     // Main thread waits for this to finish

  _getch();
  return 0;
}
```

## 它是如何工作的...

第一步是包含头文件`thread.h`。这使我们可以访问所有内置库，以便创建我们的多线程应用程序所需的所有库。下一步是创建我们需要线程的任务或函数。在这个例子中，我们创建了一个名为`ThreadOne`的函数。这个函数代表我们可以用来多线程的任何函数。这可以是物理函数，音频函数，或者我们可能需要的任何函数。为简单起见，我们使用了一个打印消息的函数。下一步是生成一个线程。我们只需要编写关键字`thread`，为线程分配一个名称（`T1`），然后编写我们想要线程的函数/任务。在这种情况下，它是`ThreadOne`。

这会生成一个线程，并且不会独立于主线程执行。

# 加入和分离线程

线程生成后，它作为一个新任务开始执行，与主线程分开。然而，可能存在一些情况，我们希望任务重新加入主线程。这是可能的。我们可能还希望线程始终与主线程保持分离。这也是可能的。然而，在连接到主线程和分离时，我们必须采取一些预防措施。

## 准备工作

您需要一台运行 Windows 和 Visual Studio 的工作计算机。

## 如何做...

在这个配方中，我们将看到加入和分离线程有多么容易。添加一个名为`Source.cpp`的源文件。将以下代码添加到其中：

```cpp
int ThreadOne()
{
  std::cout << "I am thread 1" << std::endl;
  return 0;
}

int ThreadTwo()
{
  std::cout << "I am thread 2" << std::endl;
  return 0;
}

int main()
{
  std::thread T1(ThreadOne);
  std::thread T2(ThreadTwo);

  if (T1.joinable()) // Check if can be joined to the main thread
    T1.join();     // Main thread waits for this to finish

  T2.detach();    //Detached from main thread

  _getch();
  return 0;
}
```

## 它是如何工作的...

在上面的例子中，首先生成了两个线程。这两个线程是`T1`和`T2`。当线程被生成时，它们会独立并发地运行。然而，当需要将任何线程重新加入到主线程时，我们也可以这样做。首先，我们需要检查线程是否可以加入到主线程。我们可以通过 joinable 函数来实现这一点。如果函数返回`true`，则线程可以加入到主线程。我们可以使用`join`函数加入到主线程。如果我们直接加入，而没有首先检查线程是否可以加入到主线程，可能会导致主线程无法接受该线程而出现问题。线程加入到主线程后，主线程会等待该线程完成。

如果我们想要将线程从主线程分离，我们可以使用`detach`函数。然而，在我们将其从主线程分离后，它将永远分离。

# 向线程传递参数

就像在函数中一样，我们可能还想将参数和参数传递给线程。由于线程只是任务，而任务只是一系列函数的集合，因此有必要了解如何向线程发送参数。如果我们可以在运行时向线程发送参数，那么线程可以动态执行所有操作。在大多数情况下，我们会将物理、人工智能或音频部分线程化。所有这些部分都需要接受参数的函数。

## 准备工作

你需要一台 Windows 机器和一个安装好的 Visual Studio 副本。不需要其他先决条件。

## 如何做…

在这个食谱中，我们将发现为我们的游戏添加启发式函数进行路径规划有多么容易。添加一个名为`Source.cpp`的源文件。将以下代码添加到其中：

```cpp
class Wrapper
{
public:
  void operator()(std::string& msg)
  {
    msg = " I am from T1";
    std::cout << "T1 thread initiated" << msg << std::endl;

  }
};

int main()
{
  std::string s = "This is a message";
  std::cout << std::this_thread::get_id() << std::endl;

  std::thread T1((Wrapper()), std::move(s));
  std::cout << T1.get_id() << std::endl;

  std::thread T2 = std::move(T1);
  T2.join();

  _getch();

}
```

## 工作原理…

传递参数的最佳方法是编写一个`Wrapper`类并重载`()`运算符。在我们重载`()`运算符之后，我们可以向线程发送参数。为此，我们创建一个字符串并将字符串存储在一个变量中。然后我们需要像往常一样生成一个线程；然而，我们不仅仅传递函数名，而是传递类名和字符串。在线程中，我们需要通过引用传递参数，因此我们可以使用`ref`函数。然而，更好的方法是使用`move`函数，其中我们注意内存位置本身并将其传递给参数。`operator`函数接受字符串并打印消息。

如果我们想创建一个新线程并使其与第一个线程相同，我们可以再次使用`move`函数来实现这一点。除此之外，我们还可以使用`get_id`函数来获取线程的 ID。

# 避免死锁

当两个或更多任务想要使用相同的资源时，就会出现竞争条件。在一个任务完成使用资源之前，另一个任务无法访问它。这被称为**死锁**，我们必须尽一切努力避免死锁。例如，资源`Collision`和资源`Audio`被进程`Locomotion`和进程`Bullet`使用：

+   `Locomotion`开始使用`Collision`

+   `Locomotion`和`Bullet`尝试开始使用`Audio`

+   `Bullet`“赢得”并首先获得`Audio`

+   现在`Bullet`需要使用`Collision`

+   `Collision`被`Locomotion`锁定，它正在等待`Bullet`

## 准备工作

对于这个食谱，你需要一台 Windows 机器和一个安装好的 Visual Studio 副本。

## 如何做…

在这个食谱中，我们将发现避免死锁有多么容易：

```cpp
#include <thread>
#include <string>
#include <iostream>

using namespace std;

void Physics()
{
  for (int i = 0; i > -100; i--)
    cout << "From Thread 1: " << i << endl;

}

int main()
{
  std::thread t1(Physics);
  for (int i = 0; i < 100; i++)
    cout << "From main: " << i << endl;

  t1.join();

  int a;
  cin >> a;
  return 0;
}
```

## 工作原理…

在上面的例子中，我们生成了一个名为`t1`的线程，它开始一个函数以从 0 到-100 打印数字，递减 1。还有一个主线程，它开始从 0 到 100 打印数字，递增 1。同样，我们选择了这些函数是为了简单理解。这些可以很容易地被*A*算法和搜索算法或其他任何我们想要的东西所替代。

如果我们看控制台输出，我们会注意到它非常混乱。原因是`cout`对象被主线程和`t1`同时使用。因此，发生了数据竞争的情况。每次谁赢得了竞争，谁就会显示数字。我们必须尽一切努力避免这种编程结构。很多时候，它会导致死锁和中断。

# 数据竞争和互斥锁

数据竞争条件在多线程应用程序中非常常见，但我们必须避免这种情况，以防止死锁发生。**互斥锁**帮助我们克服死锁。互斥锁是一个程序对象，允许多个程序线程共享相同的资源，比如文件访问，但不是同时。当程序启动时，会创建一个带有唯一名称的互斥锁。

## 准备工作

对于这个食谱，你需要一台 Windows 机器和安装了 Visual Studio 的版本。

## 如何做…

在这个食谱中，我们将看到理解数据竞争和互斥锁是多么容易。添加一个名为`Source.cpp`的源文件，并将以下代码添加到其中：

```cpp
#include <thread>
#include <string>
#include <mutex>
#include <iostream>

using namespace std;

std::mutex MU;

void Locomotion(string msg, int id)
{
  std::lock_guard<std::mutex> guard(MU); //RAII
  //MU.lock();
  cout << msg << id << endl;
  //MU.unlock();
}
void InterfaceFunction()
{
  for (int i = 0; i > -100; i--)
    Locomotion(string("From Thread 1: "), i);

}

int main()
{
  std::thread FirstThread(InterfaceFunction);
  for (int i = 0; i < 100; i++)
    Locomotion(string("From Main: "), i);

  FirstThread.join();

  int a;
  cin >> a;
  return 0;
}
```

## 它是如何工作的…

在这个例子中，主线程和`t1`都想显示一些数字。然而，由于它们都想使用`cout`对象，这就产生了数据竞争的情况。为了避免这种情况，一种方法是使用互斥锁。因此，在执行`print`语句之前，我们有`mutex.lock`，在`print`语句之后，我们有`mutex.unlock`。这样可以工作，并防止数据竞争条件，因为互斥锁将允许一个线程使用资源，并使另一个线程等待它。然而，这个程序还不是线程安全的。这是因为如果`cout`语句抛出错误或异常，互斥锁将永远不会被解锁，其他线程将始终处于`等待`状态。

为了避免这种情况，我们将使用 C++的**资源获取即初始化技术**（**RAII**）。我们在函数中添加一个内置的锁保护。这段代码是异常安全的，因为 C++保证所有堆栈对象在封闭范围结束时被销毁，即所谓的**堆栈展开**。当从函数返回时，锁和文件对象的析构函数都将被调用，无论是否抛出了异常。因此，如果发生异常，它不会阻止其他线程永远等待。尽管这样做，这个应用程序仍然不是线程安全的。这是因为`cout`对象是一个全局对象，因此程序的其他部分也可以访问它。因此，我们需要进一步封装它。我们稍后会看到这一点。

# 编写一个线程安全的类

在处理多个线程时，编写一个线程安全的类变得非常重要。如果我们不编写线程安全的类，可能会出现许多复杂情况，比如死锁。我们还必须记住，当我们编写线程安全的类时，就不会有数据竞争和互斥锁的潜在危险。

## 准备工作

对于这个食谱，你需要一台 Windows 机器和安装了 Visual Studio 的版本。

## 如何做…

在这个食谱中，我们将看到在 C++中编写一个线程安全的类是多么容易。添加一个名为`Source.cpp`的源文件，并将以下代码添加到其中：

```cpp
#include <thread>
#include <string>
#include <mutex>
#include <iostream>
#include <fstream>

using namespace std;

class DebugLogger
{
  std::mutex MU;
  ofstream f;
public:
  DebugLogger()
  {
    f.open("log.txt");
  }
  void ResourceSharingFunction(string id, int value)
  {
    std::lock_guard<std::mutex> guard(MU); //RAII
    f << "From" << id << ":" << value << endl;
  }

};

void InterfaceFunction(DebugLogger& log)
{
  for (int i = 0; i > -100; i--)
    log.ResourceSharingFunction(string("Thread 1: "), i);

}

int main()
{
  DebugLogger log;
  std::thread FirstThread(InterfaceFunction,std::ref(log));
  for (int i = 0; i < 100; i++)
    log.ResourceSharingFunction(string("Main: "), i);

  FirstThread.join();

  int a;
  cin >> a;
  return 0;
}
```

## 它是如何工作的…

在上一个食谱中，我们看到尽管编写了互斥锁和锁，我们的代码仍然不是线程安全的。这是因为我们使用了一个全局对象`cout`，它也可以从代码的其他部分访问，因此不是线程安全的。因此，我们通过添加一层抽象来避免这样做，并将结果输出到日志文件中。

我们已经创建了一个名为`Logfile`的类。在这个类里，我们创建了一个锁保护和一个互斥锁。除此之外，我们还创建了一个名为`f`的流对象。使用这个对象，我们将内容输出到一个文本文件中。需要访问这个功能的线程将需要创建一个`LogFile`对象，然后适当地使用这个函数。我们在 RAII 系统中使用了锁保护。由于这种抽象层，外部无法使用这个功能，因此是非常安全的。

然而，即使在这个程序中，我们也需要采取一定的预防措施。我们应该采取的第一项预防措施是不要从任何函数中返回`f`。此外，我们必须小心，`f`不应该直接从任何其他类或外部函数中获取。如果我们做了上述任何一项，资源`f`将再次可用于程序的外部部分，将不受保护，因此将不再是线程安全的。
