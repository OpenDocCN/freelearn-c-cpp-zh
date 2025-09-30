# 第九章。用多线程保持你的理智

在前面的章节中，我们设法总是编写不依赖线程的代码。现在是时候面对这个怪物，真正理解在 Qt 中线程是如何工作的了。在本章中，你将开发一个显示曼德布罗特分形的多线程应用程序。这是一个计算密集型过程，将让你的 CPU 核心流泪。

在示例项目中，用户可以看到曼德布罗特分形，放大图片，并四处移动以发现分形的神奇之处。

本章涵盖了以下主题：

+   深入理解 `QThread` 框架

+   Qt 中所有可用的线程技术概述

+   使用 `QThreadPool` 类调度任务并汇总结果

+   如何同步线程并最小化共享状态

+   低级绘图以优化性能

+   常见的线程陷阱和挑战

# 发现 QThread

Qt 提供了一个复杂的线程系统。我们假设你已经了解了线程基础知识及其相关的问题（死锁、线程同步、资源共享等），我们将重点介绍 Qt 如何实现它。

`QThread` 是 Qt 线程系统的核心类。一个 `QThread` 实例管理程序中的一个执行线程。

你可以继承 `QThread` 来重写 `run()` 函数，该函数将在 `QThread` 框架中执行。以下是创建和启动 `QThread` 的方法：

```cpp
QThread thread; 
thread.start(); 

```

调用 `start()` 函数将自动调用线程的 `run()` 函数并发出 `started()` 信号。只有在这一点上，新的执行线程才会被创建。当 `run()` 完成时，`thread` 对象将发出 `finished()` 信号。

这将我们带到了 `QThread` 的一个基本方面：它与信号/槽机制无缝工作。Qt 是一个事件驱动的框架，其中主事件循环（或 GUI 循环）处理事件（用户输入、图形等）以刷新 UI。

每个 `QThread` 都有自己的事件循环，可以处理主循环之外的事件。如果不重写，`run()` 将调用 `QThread::exec()` 函数，该函数启动 `thread` 对象的事件循环。你也可以重写 `QThread` 并调用自己的 `exec()`，如下所示：

```cpp
class Thread : public QThread 
{ 
Q_OBJECT 
protected: 
    void run()  
    { 
      Object* myObject = new Object(); 
        connect(myObject, &Object::started,  
                this, &Thread::doWork); 
        exec(); 
    } 

private slots: 
    void doWork(); 
}; 

```

只有在调用 `exec()` 后，`started()` 信号才会由 `Thread` 事件循环处理。它将阻塞并等待直到调用 `QThread::exit()`。

一个需要注意的关键点是，线程事件循环为该线程中所有存活的 `QObjects` 提供事件。这包括在该线程中创建的所有对象或移动到该线程的对象。这被称为对象线程亲和力。让我们看一个例子：

```cpp
class Thread : public QThread 
{ 
    Thread() : 
        mObject(new QObject()) 
    { 
    } 
private : 
    QObject* myObject; 
}; 

// Somewhere in MainWindow 
Thread thread; 
thread.start(); 

```

在这个片段中，`myObject` 在 `Thread` 类的构造函数中构建，该构造函数在 `MainWindow` 中创建。此时，`thread` 正在 GUI 线程中运行。因此，`myObject` 也生活在 GUI 线程中。

### 注意

在创建 `QCoreApplication` 对象之前创建的对象没有线程亲和性。因此，不会向其派遣任何事件。

能够在我们的 `QThread` 中处理信号和槽是非常棒的，但我们如何控制跨多个线程的信号呢？一个经典的例子是，一个长时间运行的过程在一个单独的线程中执行，必须通知 UI 更新某些状态：

```cpp
class Thread : public QThread 
{ 
    Q_OBJECT 
    void run() { 
        // long running operation 
        emit result("I <3 threads"); 
    } 
signals: 
    void result(QString data); 
}; 

// Somewhere in MainWindow 
Thread* thread = new Thread(this); 
connect(thread, &Thread::result, this, &MainWindow::handleResult); 
connect(thread, &Thread::finished, thread, &QObject::deleteLater); 
thread->start(); 

```

直觉上，我们假设第一个 `connect` 会将信号发送到多个线程（以便在 `MainWindow::handleResult` 中可用结果），而第二个 `connect` 应该只在工作线程的事件循环上工作。

幸运的是，这是由于 `connect()` 函数签名中的默认参数：连接类型。让我们看看完整的签名：

```cpp
QObject::connect( 
    const QObject *sender, const char *signal,  
    const QObject *receiver, const char *method,  
    Qt::ConnectionType type = Qt::AutoConnection) 

```

`type` 关键字默认取 `Qt::AutoConnection`。让我们回顾一下 `Qt::ConnectionType` 枚举的可能值，如官方 Qt 文档所述：

+   `Qt::AutoConnection`：如果接收器位于发出信号的线程中，则使用 `Qt::DirectConnection`。否则，使用 `Qt::QueuedConnection`。连接类型在信号发出时确定。

+   `Qt::DirectConnection`：当信号发出时，将立即调用此槽。槽在发出信号的线程中执行。

+   `Qt::QueuedConnection`：当控制权返回接收器线程的事件循环时，将调用此槽。槽在接收器线程中执行。

+   `Qt::BlockingQueuedConnection`：这与 `Qt::QueuedConnection` 相同，只不过信号线程会阻塞，直到槽返回。如果接收器位于信号线程中，则不得使用此连接，否则应用程序将发生死锁。

+   `Qt::UniqueConnection`：这是一个可以与之前任何一种连接类型组合的标志，使用位或运算。当设置 `Qt::UniqueConnection` 时，如果连接已经存在（即，如果相同的信号已经连接到相同的槽，针对同一对对象），`QObject::connect()` 将失败。

当使用 `Qt::AutoConnection` 时，最终的 `ConnectionType` 仅在信号实际发出时才被解决。如果你再次看我们的例子，第一个 `connect()`：

```cpp
connect(thread, &Thread::result,  
        this, &MainWindow::handleResult); 

```

当 `result()` 发出时，Qt 会查看 `handleResult()` 线程亲和性，这与 `result()` 信号的线程亲和性不同。`thread` 对象位于 `MainWindow` 中（记住它是在 `MainWindow` 中创建的），但 `result()` 信号是在 `run()` 函数中发出的，该函数在不同的执行线程中运行。因此，将使用 `Qt::QueuedConnection` 槽。

我们现在可以看看第二个 `connect()`：

```cpp
connect(thread, &Thread::finished, thread, &QObject::deleteLater); 

```

在这里，`deleteLater()` 和 `finished()` 都位于同一个线程中；因此，将使用 `Qt::DirectConnection` 槽。

你必须明白，Qt 并不关心发出信号的线程亲和性，它只关注信号的“执行上下文”。

带着这些知识，我们可以再次审视我们的第一个 `QThread` 类示例，以全面理解这个系统：

```cpp
class Thread : public QThread 
{ 
Q_OBJECT 
protected: 
    void run()  
    { 
        Object* myObject = new Object(); 
        connect(myObject, &Object::started,  
                this, &Thread::doWork); 
        exec(); 
    } 

private slots: 
    void doWork(); 
}; 

```

当 `Object::started()` 函数被发出时，将使用一个 `Qt::QueuedConnection` 插槽。这就是你的大脑冻结的地方。`Thread::doWork()` 函数位于 `Object::started()` 所在的另一个线程中，该线程是在 `run()` 中创建的。如果线程是在 UI 线程中实例化的，那么这就是 `doWork()` 应该属于的地方。

这个系统功能强大，但也很复杂。为了使事情更简单，Qt 倾向于使用工作模型。它将线程管道与实际处理分离。以下是一个例子：

```cpp
class Worker : public QObject 
{ 
    Q_OBJECT 
public slots: 
    void doWork()  
    { 
        emit result("workers are the best"); 
    } 

signals: 
    void result(QString data); 
}; 

// Somewhere in MainWindow 
QThread* thread = new Thread(this); 
Worker* worker = new Worker(); 
worker->moveToThread(thread); 

connect(thread, &QThread::finished,  
        worker, &QObject::deleteLater); 
connect(this, &MainWindow::startWork,  
        worker, &Worker::doWork); 
connect(worker, &Worker::resultReady,  
        this, handleResult); 

thread->start(); 

// later on, to stop the thread 
thread->quit(); 
thread->wait(); 

```

我们首先创建一个具有以下内容的 `Worker` 类：

+   一个将包含我们旧的 `QThread::run()` 内容的 `doWork()` 插槽

+   一个将发出结果的 `result()` 信号

在 `MainWindow` 类中，我们创建了一个简单的 `thread` 对象和一个 `Worker` 实例。`worker->moveToThread(thread)` 是魔法发生的地方。它改变了 `worker` 对象的亲和力。现在 `worker` 位于 `thread` 对象中。

你只能从当前线程向另一个线程推送对象。相反，你不能从另一个线程中拉取一个对象。如果你不在你的线程中，你不能改变对象的线程亲和力。一旦执行了 `thread->start()`，除非我们从这个新线程中执行，否则我们不能调用 `worker->moveToThread(this)`。

之后，我们做了三个 `connect()`：

1.  我们通过在线程完成时回收它来处理 `worker` 生命周期。这个信号将使用 `Qt::DirectConnection`。

1.  在可能的 UI 事件上启动 `Worker::doWork()`。这个信号将使用 `Qt::QueuedConnection`。

1.  我们使用 `handleResult()` 在 UI 线程中处理结果数据。这个信号将使用 `Qt::QueuedConnection`。

总结来说，`QThread` 可以被继承或与 `worker` 类一起使用。通常，更倾向于使用工作方法，因为它更清晰地分离了线程亲和力管道和实际并行执行的操作。

# 飞越 Qt 多线程技术

基于 `QThread`，Qt 中提供了几种线程技术。首先，为了同步线程，通常的方法是使用互斥锁（mutex）来为给定资源提供互斥。Qt 通过 `QMutex` 类提供它。其用法很简单：

```cpp
QMutex mutex; 
int number = 1; 

mutex.lock(); 
number *= 2; 
mutex.unlock(); 

```

从 `mutex.lock()` 指令开始，任何尝试锁定 `mutex` 的其他线程都将等待直到 `mutex.unlock()` 被调用。

在复杂代码中，锁定/解锁机制容易出错。你可能会忘记在特定的退出条件下解锁互斥锁，从而导致死锁。为了简化这种情况，Qt 提供了一个 `QMutexLocker`，应该在需要锁定 `QMutex` 的地方使用：

```cpp
QMutex mutex; 
QMutexLocker locker(&mutex); 

int number = 1; 
number *= 2; 
if (overlyComplicatedCondition) { 
    return; 
} else if (notSoSimple) { 
    return; 
} 

```

当创建`locker`对象时，`mutex`会被锁定，并在`locker`对象被销毁时解锁；例如，当它超出作用域时。对于每个我们提到的出现`return`语句的条件，情况都是如此。这使得代码更简单、更易读。

你可能需要频繁地创建和销毁线程，因为手动管理`QThread`实例可能会变得繁琐。为此，你可以使用`QThreadPool`类，它管理着一组可重用的`QThread`。

要在由`QThreadPool`类管理的线程中执行代码，你将使用一个与我们之前覆盖的工人非常相似的模式。主要区别在于处理类必须扩展`QRunnable`类。以下是它的样子：

```cpp
class Job : public QRunnable 
{ 
    void run() 
    { 
        // long running operation 
    } 
} 

Job* job = new Job(); 
QThreadPool::globalInstance()->start(job); 

```

只需重写`run()`函数，并让`QThreadPool`在单独的线程中执行你的任务。`QThreadPool::globalInstance()`是一个静态辅助函数，它为你提供了访问应用程序全局实例的权限。如果你需要更精细地控制`QThreadPool`的生命周期，你可以创建自己的`QThreadPool`。

注意，`QThreadPool::start()`函数会接管`job`的所有权，并在`run()`完成后自动删除它。小心，这不会像`QObject::moveToThread()`对工作者那样改变线程亲和力！`QRunnable`类不能重用，它必须是一个全新的实例。

如果你启动了多个任务，`QThreadPool`会根据你的 CPU 核心数自动分配理想数量的线程。`QThreadPool`类可以启动的最大线程数可以通过`QThreadPool::maxThreadCount()`获取。

### 小贴士

如果你需要手动管理线程，但希望基于你的 CPU 核心数，你可以使用方便的静态函数`QThreadPool::idealThreadCount()`。

使用 Qt 并发框架，还有另一种多线程开发的方法。它是一个高级 API，避免了使用互斥锁/锁定/等待条件，并促进了处理在 CPU 核心之间的分布。

Qt 并发依赖于`QFuture`类来执行函数，并期望稍后得到结果：

```cpp
void longRunningFunction(); 
QFuture<void> future = QtConcurrent::run(longRunningFunction); 

```

`longRunningFunction()`函数将在从默认的`QThreadPool`类获得的单独线程中执行。

要将参数传递给`QFuture`类并检索操作的结果，请使用以下代码：

```cpp
QImage processGrayscale(QImage& image); 
QImage lenna; 

QFuture<QImage> future = QtConcurrent::run(processGrayscale, 
    lenna); 

QImage grayscaleLenna = future.result(); 

```

在这里，我们将`lenna`作为参数传递给`processGrayscale()`函数。因为我们想要一个`QImage`作为结果，所以我们使用模板类型声明`QFuture`类，为`QImage`。之后，`future.result()`会阻塞当前线程，等待操作完成以返回最终的`QImage`。

为了避免阻塞，`QFutureWatcher`来帮忙：

```cpp
QFutureWatcher<QImage> watcher; 
connect(&watcher, &QFutureWatcher::finished,  
        this, &QObject::handleGrayscale); 

QImage processGrayscale(QImage& image); 
QImage lenna; 
QFuture<QImage> future = QtConcurrent::run(processImage, lenna); 
watcher.setFuture(future); 

```

我们首先声明一个`QFutureWatcher`类，其模板参数与`QFuture`使用的参数相匹配。然后只需将`QFutureWatcher::finished`信号连接到当操作完成后要调用的槽。

最后一步是告诉`watcher`对象使用`watcher.setFuture(future)`来监视未来的对象。这个语句看起来几乎像是来自科幻电影。

Qt 并发还提供了`MapReduce`和`FilterReduce`的实现。`MapReduce`是一种编程模型，基本上做两件事：

+   在 CPU 的多个核心之间映射或分配数据集的处理

+   将结果减少或聚合以提供给调用者

这种技术最初由谷歌推广，以便能够在 CPU 集群中处理大量数据集。

这里是一个简单的映射操作的例子：

```cpp
QList images = ...; 

QImage processGrayscale(QImage& image); 
QFuture<void> future = QtConcurrent::mapped( 
                                     images, processGrayscale); 

```

我们不是使用`QtConcurrent::run()`，而是使用映射函数，该函数每次都接受一个列表和应用于列表中每个元素的不同线程中的函数。`images`列表就地修改，因此不需要使用模板类型声明`QFuture`。

可以通过使用`QtConcurrent::blockingMapped()`而不是`QtConcurrent::mapped()`来使操作阻塞。

最后，一个`MapReduce`操作看起来是这样的：

```cpp
QList images = ...; 

QImage processGrayscale(QImage& image); 
void combineImage(QImage& finalImage, const QImage& inputImage); 

QFuture<void> future = QtConcurrent::mappedReduced( 
                                            images,  
                                            processGrayscale,  
                                            combineImage); 

```

在这里，我们添加了一个`combineImage()`函数，它将在`processGrayscale()`映射函数返回的每个结果上被调用。它将中间数据`inputImage`合并到`finalImage`中。这个函数在每个线程中只被调用一次，因此不需要使用互斥锁来锁定结果变量。

`FilterReduce`遵循完全相同的模式；过滤器函数只是允许您过滤输入列表而不是转换它。

# 构建曼德布罗特项目架构

本章的示例项目是计算曼德布罗特分形的多线程计算。用户将看到分形，并能够在该窗口中平移和缩放。

在深入代码之前，我们必须对分形有一个广泛的理解，以及我们如何实现其计算。

曼德布罗特分形是一个使用复数（a + bi）的数值集。每个像素都与通过迭代计算出的一个值相关联。如果这个迭代值发散到无穷大，那么这个像素就超出了曼德布罗特集。如果不发散，那么这个像素就在曼德布罗特集内。曼德布罗特分形的视觉表示如下：

![构建曼德布罗特项目架构](img/image00428.jpeg)

这张图像中的每一个黑色像素都倾向于发散到无限大的值，而白色像素则被限制在有限值内。白色像素属于曼德布罗特集。

从多线程的角度来看，使其变得有趣的是，为了确定像素是否属于曼德布罗特集，我们必须迭代一个公式来假设它的发散与否。我们进行的迭代越多，我们声称“是的，这个像素在曼德布罗特集中，它是一个白色像素”就越安全。

更有趣的是，我们可以取图形图中的任何值，并始终应用 Mandelbrot 公式来推断像素应该是黑色还是白色。因此，你可以在分形图形内部无限缩放。只有两个主要限制：

+   你的 CPU 性能阻碍了图片生成速度。

+   你的 CPU 架构的浮点数精度限制了缩放。如果你继续缩放，你会得到视觉伪影，因为缩放因子只能处理 15 到 17 位有效数字。

应用程序的架构必须精心设计。因为我们正在使用线程，所以很容易导致死锁、线程饥饿，甚至更糟糕的是，冻结 UI。

我们真的想最大化 CPU 的使用。为此，我们将尽可能在每个核心上执行尽可能多的线程。每个线程将负责计算 Mandelbrot 集的一部分，然后再返回其结果。

应用程序的架构如下：

![Mandelbrot 项目架构](img/image00429.jpeg)

应用程序被分为三个部分：

+   `MandelbrotWidget`：这个请求显示图片。它处理绘图和用户交互。此对象位于 UI 线程中。

+   `MandelbrotCalculator`：这个处理图片请求并在发送回 `MandelbrotWidget` 之前聚合结果 `JobResults` 的对象。此对象在其自己的线程中运行。

+   `Job`：这个在将结果传回 `MandelbrotCalculator` 之前计算最终图片的一部分。每个任务都位于自己的线程中。

`MandelbrotCalculator` 线程将使用 `QThreadPool` 类在其自己的线程中调度任务。这将根据你的 CPU 核心完美缩放。每个任务将在将结果通过 `JobResult` 对象发送回 `MandelbrotCalculator` 之前计算最终图片的一行。

`MandelbrotCalculator` 线程实际上是计算的总指挥。考虑一个用户在计算完成之前放大图片的情况； `MandelbrotWidget` 将请求新的图片给 `MandelbrotCalculator`，而 `MandelbrotCalculator` 必须在调度新任务之前取消所有当前任务。

我们将为此项目添加最后一个约束：它必须是互斥锁免费的。互斥锁是非常方便的工具，但它们迫使线程相互等待，并且容易出错。为此，我们将依赖 Qt 提供的多个概念和技术：多线程信号/槽，隐式共享等。

通过最小化线程间的共享状态，我们将能够让它们尽可能快地执行。这就是我们在这里的原因，对吧？燃烧一些 CPU 核心？

现在宏观图景已经更加清晰，我们可以开始实施。创建一个名为 `ch09-mandelbrot-threadpool` 的新 **Qt Widget Application** 项目。记得将 `CONFIG += c++14` 添加到 `.pro` 文件中。

# 使用 QRunnable 定义 Job 类

让我们深入到项目的核心。为了加快 Mandelbrot 图片的生成，我们将整个计算分成多个工作。一个 `Job` 是一个任务请求。根据您的 CPU 架构，将同时执行多个工作。`Job` 类生成包含结果值的 `JobResult` 函数。在我们的项目中，`Job` 类为完整图片的一行生成值。例如，800 x 600 的图像分辨率需要 600 个工作，每个工作生成 800 个值。

请创建一个名为 `JobResult.h` 的 C++ 头文件：

```cpp
#include <QSize> 
#include <QVector> 
#include <QPointF> 

struct JobResult 
{ 
    JobResult(int valueCount = 1) : 
        areaSize(0, 0), 
        pixelPositionY(0), 
        moveOffset(0, 0), 
        scaleFactor(0.0), 
        values(valueCount) 
    { 
    } 

    QSize areaSize; 
    int pixelPositionY; 
    QPointF moveOffset; 
    double scaleFactor; 

    QVector<int> values; 
}; 

```

这个结构包含两个部分：

+   输入数据（`areaSize`，`pixelPositionY`，...）

+   由 `Job` 类生成的结果 `values`

我们现在可以创建 `Job` 类本身。使用以下 `Job.h` 片段创建一个 C++ 类 `Job`：

```cpp
#include <QObject> 
#include <QRunnable> 

#include "JobResult.h" 

class Job : public QObject, public QRunnable 
{ 
    Q_OBJECT 
public: 
    Job(QObject *parent = 0); 
    void run() override; 
}; 

```

这个 `Job` 类是一个 `QRunnable`，因此我们可以重写 `run()` 来实现 Mandelbrot 图片算法。如您所见，`Job` 也继承自 `QObject`，这允许我们使用 Qt 的信号/槽功能。算法需要一些输入数据。更新您的 `Job.h` 如下：

```cpp
#include <QObject> 
#include <QRunnable> 
#include <QPointF> 
#include <QSize> 
#include <QAtomicInteger> 

class Job : public QObject, public QRunnable 
{ 
    Q_OBJECT 
public: 
    Job(QObject *parent = 0); 
    void run() override; 

    void setPixelPositionY(int value); 
    void setMoveOffset(const QPointF& value); 
    void setScaleFactor(double value); 
    void setAreaSize(const QSize& value); 
    void setIterationMax(int value); 

private: 
    int mPixelPositionY; 
    QPointF mMoveOffset; 
    double mScaleFactor; 
    QSize mAreaSize; 
    int mIterationMax; 
}; 

```

让我们讨论这些变量：

+   `mPixelPositionY` 变量是图片高度索引。因为每个 `Job` 只为一条图片线生成数据，我们需要这个信息。

+   `mMoveOffset` 变量是 Mandelbrot 原点偏移。用户可以平移图片，因此原点不总是 (0, 0)。

+   `mScaleFactor` 变量是 Mandelbrot 缩放值。用户也可以放大图片。

+   `mAreaSize` 变量是最终图片的像素大小。

+   `mIterationMax` 变量是允许用于确定一个像素的 Mandelbrot 结果的迭代次数。

我们现在可以向 `Job.h` 添加一个信号，`jobCompleted()`，以及中止功能：

```cpp
#include <QObject> 
#include <QRunnable> 
#include <QPointF> 
#include <QSize> 
#include <QAtomicInteger> 

#include "JobResult.h" 

class Job : public QObject, public QRunnable 
{ 
    Q_OBJECT 
public: 
    ... 
signals: 
    void jobCompleted(JobResult jobResult); 

public slots: 
    void abort(); 

private: 
    QAtomicInteger<bool> mAbort; 
    ... 
}; 

```

当算法结束时，将发出 `jobCompleted()` 信号。`jobResult` 参数包含结果值。`abort()` 槽将允许我们停止工作，更新 `mIsAbort` 标志值。请注意，`mAbort` 不是一个经典的 `bool`，而是一个 `QAtomicInteger<bool>`。这种 Qt 跨平台类型允许我们在不中断的情况下执行原子操作。您可以使用互斥锁或其他同步机制来完成工作，但使用原子变量是安全地从不同线程更新和访问变量的快速方法。

是时候切换到使用 `Job.cpp` 的实现部分了。以下是 `Job` 类的构造函数：

```cpp
#include "Job.h" 

Job::Job(QObject* parent) : 
    QObject(parent), 
    mAbort(false), 
    mPixelPositionY(0), 
    mMoveOffset(0.0, 0.0), 
    mScaleFactor(0.0), 
    mAreaSize(0, 0), 
    mIterationMax(1) 
{ 
} 

```

这是一个经典的初始化；不要忘记调用 `QObject` 构造函数。

我们现在可以实现 `run()` 函数：

```cpp
void Job::run() 
{ 
    JobResult jobResult(mAreaSize.width()); 
    jobResult.areaSize = mAreaSize; 
    jobResult.pixelPositionY = mPixelPositionY; 
    jobResult.moveOffset = mMoveOffset; 
    jobResult.scaleFactor = mScaleFactor; 
    ... 
} 

```

在这个第一部分，我们初始化一个 `JobResult` 变量。区域大小的宽度用于将 `JobResult::values` 构造为具有正确初始大小的 `QVector`。其他输入数据从 `Job` 复制到 `JobResult`，以便 `JobResult` 的接收者可以使用上下文输入数据获取结果。

然后，我们可以使用 Mandelbrot 算法更新 `run()` 函数：

```cpp
void Job::run() 
{ 
   ... 
    double imageHalfWidth = mAreaSize.width() / 2.0; 
    double imageHalfHeight = mAreaSize.height() / 2.0; 
    for (int imageX = 0; imageX < mAreaSize.width(); ++imageX) { 
        int iteration = 0; 
        double x0 = (imageX - imageHalfWidth)  
                  * mScaleFactor + mMoveOffset.x(); 
        double y0 = (mPixelPositionY - imageHalfHeight)  
                  * mScaleFactor - mMoveOffset.y(); 
        double x = 0.0; 
        double y = 0.0; 
        do { 
            if (mAbort.load()) { 
                return; 
            } 

            double nextX = (x * x) - (y * y) + x0; 
            y = 2.0 * x * y + y0; 
            x = nextX; 
            iteration++; 

        } while(iteration < mIterationMax 
                && (x * x) + (y * y) < 4.0); 

        jobResult.values[imageX] = iteration; 
    } 

    emit jobCompleted(jobResult); 
} 

```

曼德布罗特算法本身超出了本书的范围。但您必须理解这个`run()`函数的主要目的。让我们分解一下：

+   for 循环遍历一行中所有像素的`x`位置

+   像素位置被转换为复平面坐标

+   如果尝试次数超过最大授权迭代次数，算法将以`iteration`到`mIterationMax`的值结束

+   如果曼德布罗特检查条件为真，算法将以`iteration < mIterationMax`结束

+   在任何情况下，对于每个像素，迭代次数都存储在`JobResult`的`values`中

+   最后，使用此算法的结果值发出`jobCompleted()`信号

+   我们使用`mAbort.load()`执行原子的读取；注意，如果返回值是`true`，则算法被终止，并且不会发出任何内容

最后一个函数是`abort()`槽：

```cpp
void Job::abort() 
{ 
    mAbort.store(true); 
} 

```

此方法执行原子的值写入，`true`。原子机制确保我们可以从多个线程调用`abort()`而不会干扰`run()`函数中的`mAbort`读取。

在我们的情况下，`run()`函数存在于受`QThreadPool`影响的线程中（我们很快会介绍它），而`abort()`槽将在`MandelbrotCalculator`线程上下文中被调用。

您可能想使用`QMutex`来确保对`mAbort`的操作。但是，请记住，如果频繁地进行锁定和解锁，锁定和解锁互斥锁可能会变得代价高昂。在这里使用`QAtomicInteger`类只提供了优势：对`mAbort`的访问是线程安全的，我们避免了昂贵的锁定。

`Job`实现的末尾只包含设置函数。如果您有任何疑问，请参阅完整的源代码。

# 在 MandelbrotCalculator 中使用 QThreadPool

现在当我们的`Job`类准备好使用时，我们需要创建一个类来管理作业。请创建一个新的类，`MandelbrotCalculator`。让我们看看在文件`MandelbrotCalculator.h`中我们需要什么：

```cpp
#include <QObject> 
#include <QSize> 
#include <QPointF> 
#include <QElapsedTimer> 
#include <QList> 

#include "JobResult.h" 

class Job; 

class MandelbrotCalculator : public QObject 
{ 
    Q_OBJECT 
public: 
    explicit MandelbrotCalculator(QObject *parent = 0); 
    void init(QSize imageSize); 

private: 
    QPointF mMoveOffset; 
    double mScaleFactor; 
    QSize mAreaSize; 
    int mIterationMax; 
    int mReceivedJobResults; 
    QList<JobResult> mJobResults; 
    QElapsedTimer mTimer; 
}; 

```

我们已经在上一节中讨论了`mMoveOffset`、`mScaleFactor`、`mAreaSize`和`mIterationMax`。我们还有一些新的变量：

+   `mReceivedJobResults`变量是接收到的`JobResult`的数量，这是由作业发送的

+   `mJobResults`变量是一个包含接收到的`JobResult`的列表

+   `mTimer`变量计算运行请求图片所需的所有作业的经过时间

现在您对所有的成员变量有了更好的了解，我们可以添加信号、槽和私有方法。更新您的`MandelbrotCalculator.h`文件：

```cpp
... 
class MandelbrotCalculator : public QObject 
{ 
    Q_OBJECT 
public: 
    explicit MandelbrotCalculator(QObject *parent = 0); 
    void init(QSize imageSize); 

signals: 
    void pictureLinesGenerated(QList<JobResult> jobResults); 
    void abortAllJobs(); 

public slots: 
    void generatePicture(QSize areaSize, QPointF moveOffset, 
                         double scaleFactor, int iterationMax); 
    void process(JobResult jobResult); 

private: 
    Job* createJob(int pixelPositionY); 
    void clearJobs(); 

private: 
    ... 
}; 

```

这里是这些角色的作用：

+   `generatePicture()`：这个槽由调用者用来请求新的曼德布罗特图片。这个函数准备并启动作业。

+   `process()`：这个槽处理作业生成的结果。

+   `pictureLinesGenerated()`：这个信号会定期触发以分发结果。

+   `abortAllJobs()`：这个信号用于终止所有活动作业。

+   `createJob()`：这是一个辅助函数，用于创建和配置一个新的作业。

+   `clearJobs()`：这个槽移除队列中的作业并中止正在进行的作业。

头文件已完成，我们现在可以执行实现。以下是 `MandelbrotCalculator.cpp` 实现的开始：

```cpp
#include <QDebug> 
#include <QThreadPool> 

#include "Job.h" 

const int JOB_RESULT_THRESHOLD = 10; 

MandelbrotCalculator::MandelbrotCalculator(QObject *parent) 
    : QObject(parent), 
      mMoveOffset(0.0, 0.0), 
      mScaleFactor(0.005), 
      mAreaSize(0, 0), 
      mIterationMax(10), 
      mReceivedJobResults(0), 
      mJobResults(), 
      mTimer() 
{ 
} 

```

和往常一样，我们使用默认值初始化列表来设置我们的成员变量。`JOB_RESULT_THRESHOLD` 的作用将在稍后介绍。以下是 `generatePicture()` 槽：

```cpp
void MandelbrotCalculator::generatePicture(QSize areaSize, QPointF moveOffset, double scaleFactor, int iterationMax) 
{ 
    if (areaSize.isEmpty()) { 
        return; 
    } 

    mTimer.start(); 
    clearJobs(); 

    mAreaSize = areaSize; 
    mMoveOffset = moveOffset; 
    mScaleFactor = scaleFactor; 
    mIterationMax = iterationMax; 

    for(int pixelPositionY = 0; 
        pixelPositionY < mAreaSize.height(); pixelPositionY++) { 
        QThreadPool::globalInstance()-> 
            start(createJob(pixelPositionY)); 
    } 
} 

```

如果 `areaSize` 维度为 0x0，我们就没有什么要做的。如果请求有效，我们可以启动 `mTimer` 来跟踪整个生成持续时间。每次生成新图片时，首先通过调用 `clearJobs()` 取消现有作业。然后我们设置成员变量为提供的那些。最后，为每条垂直图片线创建一个新的 `Job` 类。将很快介绍返回 `Job*` 值的 `createJob()` 函数。

`QThreadPool::globalInstance()` 是一个静态函数，它根据我们 CPU 的核心数提供最优的全局线程池。即使我们为所有 `Job` 类调用 `start()`，也只有一个会立即启动。其他则被添加到池队列中，等待可用的线程。

现在我们来看看如何使用 `createJob()` 函数创建一个 `Job` 类：

```cpp
Job* MandelbrotCalculator::createJob(int pixelPositionY) 
{ 
    Job* job = new Job(); 

    job->setPixelPositionY(pixelPositionY); 
    job->setMoveOffset(mMoveOffset); 
    job->setScaleFactor(mScaleFactor); 
    job->setAreaSize(mAreaSize); 
    job->setIterationMax(mIterationMax); 

    connect(this, &MandelbrotCalculator::abortAllJobs, 
            job, &Job::abort); 

    connect(job, &Job::jobCompleted, 
            this, &MandelbrotCalculator::process); 

    return job; 
} 

```

如你所见，作业是在堆上分配的。这个操作在 `MandelbrotCalculator` 线程中会花费一些时间。但结果是值得的；开销被多线程系统所补偿。注意，当我们调用 `QThreadPool::start()` 时，线程池会接管 `job` 的所有权。因此，当 `Job::run()` 结束时，它将被线程池删除。我们设置了由 Mandelbrot 算法所需的 `Job` 类的输入数据。

然后执行两个连接：

+   发出我们的 `abortAllJobs()` 信号将调用所有作业的 `abort()` 槽

+   我们的 `process()` 槽在每次 `Job` 完成其任务时执行

最后，将 `Job` 指针返回给调用者，在我们的例子中，是 `generatePicture()` 槽。

最后一个辅助函数是 `clearJobs()`。将其添加到你的 `MandelbrotCalculator.cpp`：

```cpp
void MandelbrotCalculator::clearJobs() 
{ 
    mReceivedJobResults = 0; 
    emit abortAllJobs(); 
    QThreadPool::globalInstance()->clear(); 
} 

```

重置接收到的作业结果计数器。我们发出信号以中止所有正在进行的作业。最后，我们移除线程池中等待可用线程的队列中的作业。

这个类的最后一个函数是 `process()`，可能是最重要的函数。用以下代码片段更新你的代码：

```cpp
void MandelbrotCalculator::process(JobResult jobResult) 
{ 
    if (jobResult.areaSize != mAreaSize || 
            jobResult.moveOffset != mMoveOffset || 
            jobResult.scaleFactor != mScaleFactor) { 
        return; 
    } 

    mReceivedJobResults++; 
    mJobResults.append(jobResult); 

    if (mJobResults.size() >= JOB_RESULT_THRESHOLD || 
            mReceivedJobResults == mAreaSize.height()) { 
        emit pictureLinesGenerated(mJobResults); 
        mJobResults.clear(); 
    } 

    if (mReceivedJobResults == mAreaSize.height()) { 
        qDebug() << "Generated in " << mTimer.elapsed() << " ms"; 
    } 
} 

```

这个槽将在每次作业完成其任务时被调用。首先需要检查的是当前的 `JobResult` 是否仍然与当前输入数据有效。当请求新的图片时，我们清除作业队列并中止正在进行的作业。然而，如果旧的 `JobResult` 仍然发送到这个 `process()` 槽，我们必须忽略它。

之后，我们可以增加 `mReceivedJobResults` 计数器并将此 `JobResult` 添加到我们的成员队列 `mJobResults` 中。计算器等待获取 `JOB_RESULT_THRESHOLD`（即 10）个结果，然后通过发出 `pictureLinesGenerated()` 信号来分发它们。您可以小心地尝试调整此值：

+   较低的值，例如 1，将在计算器获取每行数据后立即将每行数据发送到小部件。但是，小部件处理每行数据会比计算器慢。此外，您将淹没小部件事件循环。

+   较高的值可以缓解小部件事件循环。但用户在看到动作之前需要等待更长的时间。连续的局部帧更新可以提供更好的用户体验。

注意，当事件被分发时，包含作业结果的 `QList` 类是通过复制发送的。但是 Qt 对 `QList` 执行隐式共享，所以我们只发送浅拷贝而不是昂贵的深拷贝。然后我们清除计算器的当前 `QList`。

最后，如果处理过的 `JobResult` 是区域中的最后一个，我们将显示一个调试消息，其中包含用户调用 `generatePicture()` 以来经过的时间。

### 小贴士

**Qt 小贴士**

您可以使用 `setMaxThreadCount(x)` 设置 `QThreadPool` 类使用的线程数，其中 `x` 是线程数。

# 使用 MandelbrotWidget 显示分形

这里我们完成了，曼德布罗特算法已完成，多线程系统已准备好在所有 CPU 核心上计算复杂的分形。我们现在可以创建一个将所有 `JobResult` 转换为显示漂亮图片的小部件。创建一个新的 C++ 类 `MandelbrotWidget`。对于这个小部件，我们将自己处理绘图。因此，我们不需要任何 `.ui` Qt Designer 表单文件。让我们从 `MandelbrotWidget.h` 文件开始：

```cpp
#include <memory> 

#include <QWidget> 
#include <QPoint> 
#include <QThread> 
#include <QList> 

#include "MandelbrotCalculator.h" 

class QResizeEvent; 

class MandelbrotWidget : public QWidget 
{ 
    Q_OBJECT 

public: 
    explicit MandelbrotWidget(QWidget *parent = 0); 
    ~MandelbrotWidget(); 

private: 
    MandelbrotCalculator mMandelbrotCalculator; 
    QThread mThreadCalculator; 
    double mScaleFactor; 
    QPoint mLastMouseMovePosition; 
    QPointF mMoveOffset; 
    QSize mAreaSize; 
    int mIterationMax; 
    std::unique_ptr<QImage> mImage; 
}; 

```

您应该能识别一些已知的变量名，例如 `mScaleFactor`、`mMoveOffset`、`mAreaSize` 或 `mIterationMax`。我们已经在 `JobResult` 和 `Job` 实现中介绍了它们。以下是真正的新变量：

+   `mMandelbrotCalculator` 变量是我们多线程的 `Job` 管理器。小部件会向其发送请求并等待结果。

+   `mThreadCalculator` 变量允许曼德布罗特计算器在其自己的线程中运行。

+   `mLastMouseMovePosition` 变量被小部件用于处理用户事件，以实现平移功能。

+   `mImage` 变量是小部件当前显示的图片。它是一个 `unique_ptr` 指针，因此 `MandelbrotWidget` 是 `mImage` 的所有者。

我们现在可以添加函数。更新您的代码如下：

```cpp
class MandelbrotWidget : public QWidget 
{ 
... 
public slots: 
    void processJobResults(QList<JobResult> jobResults); 

signals: 
    void requestPicture(QSize areaSize, QPointF moveOffset, double scaleFactor, int iterationMax); 

protected: 
    void paintEvent(QPaintEvent*) override; 
    void resizeEvent(QResizeEvent* event) override; 
    void wheelEvent(QWheelEvent* event) override; 
    void mousePressEvent(QMouseEvent* event) override; 
    void mouseMoveEvent(QMouseEvent* event) override; 

private: 
    QRgb generateColorFromIteration(int iteration); 

private: 
    ... 
}; 

```

在我们深入实现之前，让我们谈谈这些函数：

+   `processJobResults()` 函数将处理由 `MandelbrotCalculator` 分发的 `JobResult` 列表。

+   每当用户更改输入数据（偏移量、缩放或区域大小）时，都会发出 `requestPicture()` 信号。

+   `paintEvent()` 函数使用当前的 `mImage` 绘制小部件。

+   当用户调整窗口大小时，`resizeEvent()` 函数会调整曼德布罗特区域的大小。

+   `wheelEvent()`函数处理用户的鼠标滚轮事件以应用缩放因子。

+   `mousePressEvent()`函数和`mouseMoveEvent()`函数检索用户的鼠标事件以移动 Mandelbrot 图片。

+   `generateColorFromIteration()`是一个辅助函数，用于将 Mandelbrot 图片着色。将像素的迭代值转换为颜色值。

我们现在可以实现`MandelbrotWidget`类。以下是`MandelbrotWidget.cpp`文件的开始部分：

```cpp
#include "MandelbrotWidget.h" 

#include <QResizeEvent> 
#include <QImage> 
#include <QPainter> 
#include <QtMath> 

const int ITERATION_MAX = 4000; 
const double DEFAULT_SCALE = 0.005; 
const double DEFAULT_OFFSET_X = -0.74364390249094747; 
const double DEFAULT_OFFSET_Y = 0.13182589977450967; 

MandelbrotWidget::MandelbrotWidget(QWidget *parent) : 
    QWidget(parent), 
    mMandelbrotCalculator(), 
    mThreadCalculator(this), 
    mScaleFactor(DEFAULT_SCALE), 
    mLastMouseMovePosition(), 
    mMoveOffset(DEFAULT_OFFSET_X, DEFAULT_OFFSET_Y), 
    mAreaSize(), 
    mIterationMax(ITERATION_MAX) 
{ 
    mMandelbrotCalculator.moveToThread(&mThreadCalculator); 

    connect(this, &MandelbrotWidget::requestPicture, 
        &mMandelbrotCalculator, 
        &MandelbrotCalculator::generatePicture); 

    connect(&mMandelbrotCalculator, 
        &MandelbrotCalculator::pictureLinesGenerated, 
        this, &MandelbrotWidget::processJobResults); 

    mThreadCalculator.start(); 
} 

```

在代码片段的顶部，我们设置了一些默认的常量值。如果您希望在启动应用程序时看到不同的视图，可以随意调整这些值。构造函数首先执行的操作是改变`mMandelbrotCalculator`类的线程亲和性。这样，计算器执行的处理（创建和启动任务、汇总任务结果以及清除任务）不会干扰 UI 线程。然后，我们与`MandelbrotCalculator`的信号和槽进行连接。由于小部件和计算器有不同的线程亲和性，连接将自动成为`Qt::QueuedConnection`槽。最后，我们可以启动`mThreadCalculator`的线程。现在我们可以添加析构函数：

```cpp
MandelbrotWidget::~MandelbrotWidget() 
{  
    mThreadCalculator.quit(); 
    mThreadCalculator.wait(1000); 
    if (!mThreadCalculator.isFinished()) { 
        mThreadCalculator.terminate(); 
    } 
} 

```

我们需要请求计算器线程退出。当计算器线程的事件循环处理我们的请求时，线程将返回代码 0。我们等待 1,000 毫秒以等待线程结束。我们可以继续实现所有请求新图片的情况。以下是`resizeEvent()`槽的实现：

```cpp
void MandelbrotWidget::resizeEvent(QResizeEvent* event) 
{ 
    mAreaSize = event->size(); 

    mImage = std::make_unique<QImage>(mAreaSize, 
        QImage::Format_RGB32); 
    mImage->fill(Qt::black); 

    emit requestPicture(mAreaSize, mMoveOffset, mScaleFactor, 
        mIterationMax); 
} 

```

我们使用新的小部件大小更新`mAreaSize`。然后，创建一个新的具有正确尺寸的黑色`QImage`。最后，我们请求`MandelbrotCalculator`进行图片计算。让我们看看如何处理鼠标滚轮：

```cpp
void MandelbrotWidget::wheelEvent(QWheelEvent* event) 
{ 
    int delta = event->delta(); 
    mScaleFactor *= qPow(0.75, delta / 120.0); 
    emit requestPicture(mAreaSize, mMoveOffset, mScaleFactor, 
        mIterationMax); 
} 

```

可以从`QWheelEvent::delta()`中检索鼠标滚轮值。我们使用幂函数在`mScaleFactor`上应用一个连贯的值，并请求一张更新后的图片。现在我们可以实现平移功能：

```cpp
void MandelbrotWidget::mousePressEvent(QMouseEvent* event) 
{ 
    if (event->buttons() & Qt::LeftButton) { 
        mLastMouseMovePosition = event->pos(); 
    } 
} 

```

第一个函数存储用户开始移动手势时的鼠标位置。然后下一个函数将使用`mLastMouseMovePosition`来创建一个偏移量：

```cpp
void MandelbrotWidget::mouseMoveEvent(QMouseEvent* event) 
{ 
    if (event->buttons() & Qt::LeftButton) { 
        QPointF offset = event->pos() - mLastMouseMovePosition; 
        mLastMouseMovePosition = event->pos(); 
        offset.setY(-offset.y()); 
        mMoveOffset += offset * mScaleFactor; 
        emit requestPicture(mAreaSize, mMoveOffset, mScaleFactor, 
            mIterationMax); 
    } 
} 

```

新旧鼠标位置之间的差异给我们提供了平移偏移量。请注意，我们反转了 y 轴的值，因为鼠标事件是在一个左上参照系中，而 Mandelbrot 算法依赖于一个左下参照系。最后，我们使用更新后的输入值请求一张图片。我们已经涵盖了所有发出`requestPicture()`信号的用戶事件。现在让我们看看我们如何处理由`MandelbrotCalculator`分发的`JobResult`：

```cpp
void MandelbrotWidget::processJobResults(QList<JobResult> jobResults) 
{ 
    int yMin = height(); 
    int yMax = 0; 

    for(JobResult& jobResult : jobResults) { 

        if (mImage->size() != jobResult.areaSize) { 
            continue; 
        } 

        int y = jobResult.pixelPositionY; 
        QRgb* scanLine =  
            reinterpret_cast<QRgb*>(mImage->scanLine(y)); 

        for (int x = 0; x < mAreaSize.width(); ++x) { 
            scanLine[x] = 
                generateColorFromIteration(jobResult.values[x]); 
        } 

        if (y < yMin) { 
            yMin = y; 
        } 

        if (y > yMax) { 
            yMax = y; 
        } 
    } 

    repaint(0, yMin, 
            width(), yMax); 
} 

```

计算器发送给我们一个 `QList` 的 `JobResult`。对于每一个，我们需要检查相关区域的大小是否仍然有效。我们直接更新 `mImage` 的像素颜色。`scanLine()` 函数返回像素数据的指针。这是一种快速更新 `QImage` 像素颜色的方法。`JobResult` 函数包含迭代次数，我们的辅助函数 `generateColorFromIteration()` 根据迭代值返回一个 RGB 值。不需要完全重绘小部件，因为我们只更新 `QImage` 的几行。因此，我们只重绘更新区域。

这里是如何将迭代值转换为 RGB 值的：

```cpp
QRgb MandelbrotWidget::generateColorFromIteration(int iteration) 
{ 
    if (iteration == mIterationMax) { 
        return qRgb(50, 50, 255); 
    } 

    return qRgb(0, 0, (255.0 * iteration / mIterationMax)); 
} 

```

为曼德布罗集上色本身就是一种艺术。在这里，我们在蓝色通道上实现了一种简单的线性插值。一个漂亮的曼德布罗集图片取决于每个像素的最大迭代次数及其着色技术。你可以随意增强它！

到这里了，最后一个但同样重要的函数，`paintEvent()`：

```cpp
void MandelbrotWidget::paintEvent(QPaintEvent* event) 
{ 
    QPainter painter(this); 
    painter.save(); 

    QRect imageRect = event->region().boundingRect(); 
    painter.drawImage(imageRect, *mImage, imageRect); 

    painter.setPen(Qt::white); 

    painter.drawText(10, 20, QString("Size: %1 x %2") 
        .arg(mImage->width()) 
        .arg(mImage->height())); 

    painter.drawText(10, 35, QString("Offset: %1 x %2") 
        .arg(mMoveOffset.x()) 
        .arg(mMoveOffset.y())); 

    painter.drawText(10, 50, QString("Scale: %1") 
        .arg(mScaleFactor)); 

    painter.drawText(10, 65, QString("Max iteration: %1") 
        .arg(ITERATION_MAX)); 

    painter.restore(); 
} 

```

我们必须重写这个函数，因为我们自己处理小部件的绘制。首先要做的是绘制图像的更新区域。`QPaintEvent` 对象包含需要更新的区域。`QPainter` 类使绘制变得简单。最后，我们用白色绘制一些当前输入数据的文本信息。你现在可以逐行查看完整的图片显示概览。让我们总结一下这个功能的流程：

1.  每个 `Job::run()` 生成一个 `JobResult` 对象。

1.  `MandelbrotCalculator::process()` 信号聚合 `JobResult` 对象并将它们按组（默认为 10 组）分发。

1.  `MandelbrotWidget::processJobResults()` 信号只更新图片的相关行，并请求小部件的部分重绘。

1.  `MandelbrotWidget::paintEvent()` 信号只重新绘制带有新值的图片。

这个功能会产生一点开销，但用户体验更平滑。确实，应用程序对用户事件反应迅速：前几行几乎立即更新。用户不需要等待整个图片生成才能看到变化。

小部件已经准备好了；不要忘记将其添加到 `MainWindow`。现在提升自定义小部件应该对你来说是个简单的任务。如果你有任何疑问，请查看第四章，“征服桌面 UI”，或本章的完整源代码。你现在应该能够显示并导航到你的多线程曼德布罗集了！

如果你启动应用程序，你应该会看到类似这样的内容：

![使用 MandelbrotWidget 显示分形](img/image00430.jpeg)

现在尝试放大并平移到曼德布罗集。你应该会找到一些有趣的地方，就像这样：

![使用 MandelbrotWidget 显示分形](img/image00431.jpeg)

# 概述

您已经了解了`QThread`类的工作原理，并学习了如何高效地使用 Qt 提供的工具来创建强大的多线程应用程序。您的 Mandelbrot 应用程序能够利用 CPU 的所有核心快速计算图片。

创建一个多线程应用程序存在许多陷阱（死锁、事件循环泛滥、孤儿线程、开销等）。应用程序架构非常重要。如果您能够隔离您想要并行化的重代码，一切应该都会顺利。然而，用户体验是最重要的；如果您的应用程序能够给用户带来更平滑的感觉，有时您可能不得不接受一点开销。

在下一章中，我们将探讨几种在应用程序之间实现进程间通信（IPC）的方法。项目示例将使用 TCP/IP 套接字系统增强您当前的 Mandelbrot 应用程序。因此，Mandelbrot 生成器将能够在多台计算机的多个 CPU 核心上计算图片！
