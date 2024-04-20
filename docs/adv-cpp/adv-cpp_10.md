# 附录

## 关于

本节旨在帮助学生执行本书中的活动。它包括详细的步骤，学生需要执行这些步骤以实现活动的目标。

## 第一章 - 可移植 C++软件的解剖

### 活动 1：向项目添加新的源文件-头文件对

在这个活动中，我们将创建一个包含名为`sum`的新函数的新源文件-头文件对。它接受两个参数并返回它们的和。这个文件对将被添加到现有项目中。按照以下步骤来实现这个活动：

1.  首先，打开 Eclipse IDE，并打开我们在*练习 3*中创建的现有项目，*向 CMake 和 Eclipse CDT 添加新源文件*。分别右键单击`.cpp`和`.h`文件，或使用新类向导，然后删除类代码。使用新类向导很方便，因为它还会创建有用的样板代码。

1.  选择`SumFunc`，然后点击**完成**按钮。

1.  接下来，编辑`SumFunc.h`文件，使其看起来像以下代码：

```cpp
#ifndef SRC_SUMFUNC_H_
#define SRC_SUMFUNC_H_
int sum(int a, int b);
#endif /* SRC_SUMFUNC_H_ */
```

请注意，我们实际上将删除类并提供一个单一函数。我们本可以分别创建这两个文件。但是，`add class`函数会同时创建它们并添加一些我们将利用的样板代码。在这里，我们的文件以`include`保护开始和结束，这是一种常见的策略，用于防止双重包含问题。我们有我们函数的前向声明，这样其他文件在包含这个头文件后就可以调用这个函数。

1.  接下来，编辑`SumFunc.cpp`文件，如下所示：

```cpp
#include "SumFunc.h"
#include <iostream>
int sum(int a, int b) {
  return a + b;
}
```

在这个文件中，我们包括头文件并提供我们函数的主体，它会添加并返回给定的两个整数。

1.  编辑`CMakeFiles.txt`文件，使其`add_executable`部分反映以下代码：

```cpp
add_executable(CxxTemplate
  src/CxxTemplate.cpp  
  src/ANewClass.cpp
  src/SumFunc.cpp
)
```

在这里，我们将`src/SumFunc.cpp`文件添加到可执行源文件列表中，以便将其链接到可执行文件中。

1.  在`CxxTemplate.cpp`中进行以下更改：

```cpp
#include "CxxTemplate.h"
#include "ANewClass.h"
#include "SumFunc.h" //add this line
...
CxxApplication::CxxApplication( int argc, char *argv[] ) {
  std::cout << "Hello CMake." << std::endl;
  ANewClass anew;
  anew.run();
  std::cout << sum(3, 4) << std::endl; // add this line
}
```

#### 注意

这个文件的完整代码可以在这里找到：[`github.com/TrainingByPackt/Advanced-CPlusPlus/blob/master/Lesson1/Activity01/src/CxxTemplate.cpp`](https://github.com/TrainingByPackt/Advanced-CPlusPlus/blob/master/Lesson1/Activity01/src/CxxTemplate.cpp)。

在这里，我们添加了一行，其中我们调用`sum`函数，传入`3`和`4`，并将结果打印到控制台。

1.  构建和运行项目（**项目** | **构建全部** | **运行** | **运行**）。您看到的输出应该如下所示：

![图 1.57：输出](img/C14583_01_57.jpg)

###### 图 1.57：输出

通过这个活动，您练习了向项目添加新的源文件-头文件对。这些文件对在 C++开发中是非常常见的模式。它们可以承载全局函数，比如我们在这个活动中所做的那样。更常见的是，它们承载类及其定义。在开发过程中，您将向应用程序添加更多的源文件-头文件对。因此，习惯于添加它们并不拖延是很重要的，否则会导致难以维护和测试的大型单片文件。

### 活动 2：添加新类及其测试

在这个活动中，我们将添加一个模拟`1D`线性运动的新类。该类将具有`position`和`velocity`的双字段。它还将有一个`advanceTimeBy()`方法，该方法接收一个双`dt`参数，根据`velocity`的值修改`position`。对于双值，请使用`EXPECT_DOUBLE_EQ`而不是`EXPECT_EQ`。在这个活动中，我们将向项目添加一个新类及其测试。按照以下步骤执行这个活动：

1.  打开我们现有的项目的 Eclipse IDE。要创建一个新类，右键单击`LinearMotion1D`，然后创建类。

1.  打开我们在上一步中创建的`LinearMotion1D.h`文件。将`position`和`velocity`的`double`字段添加到其中。还要添加对`advanceTimeBy`方法的前向引用，该方法以`double dt`变量作为参数。构造函数和析构函数已经在类中。以下是在`LinearMotion1D.h`中进行这些更改的最终结果：

```cpp
#ifndef SRC_LINEARMOTION1D_H_
#define SRC_LINEARMOTION1D_H_
class LinearMotion1D {
public:
  double position;
  double velocity;
  void advanceTimeBy(double dt);
  LinearMotion1D();
  virtual ~LinearMotion1D();
};
#endif /* SRC_LINEARMOTION1D_H_ */
```

1.  现在打开`LinearMotion1D.cpp`，并为`advanceTimeBy`方法添加实现。我们的`velocity`是类中的一个字段，时间差是这个方法的一个参数。位置的变化等于速度乘以时间变化，所以我们计算结果并将其添加到位置变量中。我们还使用现有的构造函数代码将`position`和`velocity`初始化为 0。以下是在`LinearMotion1D.cpp`中进行这些更改的最终结果：

```cpp
#include "LinearMotion1D.h"
void LinearMotion1D::advanceTimeBy(double dt) {
  position += velocity * dt;
}
LinearMotion1D::LinearMotion1D() {
  position = 0;
  velocity = 0;
}
LinearMotion1D::~LinearMotion1D() {
}
```

1.  为这个类创建一个测试。右键单击`LinearMotion1DTest.cpp`，并创建它。

1.  现在打开`LinearMotion1DTest.cpp`。为两个不同方向的运动创建两个测试，左和右。对于每一个，创建一个`LinearMotion1D`对象，初始化其位置和速度，并调用`advanceTimeBy`来实际进行运动。然后，检查它是否移动到我们期望的相同位置。以下是在`LinearMotion1DTest.cpp`中进行这些更改的最终结果：

```cpp
#include "gtest/gtest.h"
#include "../src/LinearMotion1D.h"
namespace {
class LinearMotion1DTest: public ::testing::Test {};
TEST_F(LinearMotion1DTest, CanMoveRight) {
  LinearMotion1D l;
  l.position = 10;
  l.velocity = 2;
  l.advanceTimeBy(3);
  EXPECT_DOUBLE_EQ(16, l.position);
}
TEST_F(LinearMotion1DTest, CanMoveLeft) {
  LinearMotion1D l;
  l.position = 10;
  l.velocity = -2;
  l.advanceTimeBy(3);
  EXPECT_DOUBLE_EQ(4, l.position);
}
}
```

1.  现在修改我们的 CMake 配置文件，以便这些生成的源文件也被使用。对于`LinearMotion1D`类，将其`.cpp`文件添加为可执行文件，以便它与其他源文件一起编译和链接。以下是`CMakeLists.txt`中`add_executable`部分的变化：

```cpp
add_executable(CxxTemplate
  src/CxxTemplate.cpp  
  src/ANewClass.cpp
  src/SumFunc.cpp
  src/LinearMotion1D.cpp # added
)
```

1.  对于我们刚刚创建的测试，编辑`LinearMotion1DTest.cpp`，以及它使用的类的源文件`LinearMotion1D.cpp`。由于它们位于不同的目录中，以`../src/LinearMotion1D.cpp`的方式访问它们。以下是`tests/CMakeLists.txt`中`add_executable`部分的变化：

```cpp
add_executable(tests 
  CanTest.cpp 
  SumFuncTest.cpp 
  ../src/SumFunc.cpp
  LinearMotion1DTest.cpp # added
  ../src/LinearMotion1D.cpp # added
)
```

1.  构建项目并运行测试。我们将看到所有测试都成功：

![图 1.58：所有测试都成功](img/C14583_01_58.jpg)

###### 图 1.58：所有测试都成功

通过这个活动，您完成了向项目添加新类及其测试的任务。您创建了一个模拟一维运动的类，并编写了单元测试以确保其正常工作。

### 活动 3：使代码更易读

在这个活动中，您将练习提高给定代码的质量。按照以下步骤执行此活动：

1.  打开 Eclipse CDT，并在 Eclipse 中的源文件-头文件对中创建一个类。要做到这一点，请在**项目资源管理器**中右键单击**src**文件夹。从弹出菜单中选择**新建** | **类**。

1.  将`SpeedCalculator`作为头文件名，并单击**完成**。它将创建两个文件：**SpeedCalculator.h**和**SpeedCalculator.cpp**。我们提供了上述两个文件的代码。添加为每个文件提供的代码。

1.  现在我们需要将这个类添加到 CMake 项目中。打开项目根目录（**src**文件夹之外）中的**CMakeLists.txt**文件，并对文件进行以下更改：

```cpp
  src/LinearMotion1D.cpp
  src/SpeedCalculator.cpp # add this line
)
```

1.  现在选择**文件** | **全部保存**以保存所有文件，并通过选择**项目** | **全部构建**来构建项目。确保没有错误。

1.  在我们的`main()`函数中创建`SpeedCalculator`类的实例，并调用其`run()`方法。通过添加以下代码打开`main`函数：

```cpp
#include "SpeedCalculator.h"
int main( int argc, char *argv[] ) {
  cxxt::CxxApplication app( argc, argv );
  // add these three lines
  SpeedCalculator speedCalculator;
  speedCalculator.initializeData(10);
  speedCalculator.calculateAndPrintSpeedData();
  return 0;
}
```

1.  要修复样式，只需使用**源代码** | **格式化**，并选择格式化整个文件。幸运的是，变量名没有任何问题。

1.  简化代码以使其更易理解。`calculateAndPrintSpeedData`中的循环同时执行了几件事。它计算速度，找到了最小和最大值，检查我们是否越过了阈值，并存储了速度。如果速度是一个瞬态值，将其拆分意味着将其存储在某个地方以再次循环。但是，由于我们无论如何都将其存储在速度数组中，我们可以在其上再循环一次以提高代码的清晰度。以下是循环的更新版本：

```cpp
for (int i = 0; i < numEntries; ++i) {
  double dt = timesInSeconds[i + 1] - timesInSeconds[i];
  assert(dt > 0);
  double speed = (positions[i + 1] - positions[i]) / dt;
  speeds[i] = speed;
}
for (int i = 0; i < numEntries; ++i) {
  double speed = speeds[i];
  if (maxSpeed < speed) {
    maxSpeed = speed;
  }
  if (minSpeed > speed) {
    minSpeed = speed;
  }
}
for (int i = 0; i < numEntries; ++i) {
  double speed = speeds[i];
  double dt = timesInSeconds[i + 1] - timesInSeconds[i];
  if (speed > speedLimit) {
    limitCrossDuration += dt;
  }
}
```

这在某种程度上是品味的问题，但是使大`for`循环更轻松有助于提高可读性。此外，它分离了任务并消除了它们在循环迭代期间相互影响的可能性。第一个循环创建并保存速度值。第二个循环找到最小和最大速度值。第三个循环确定超速限的时间。请注意，这是一个稍微不那么高效的实现；但是，它清楚地分离了采取的行动，我们不必在循环的长迭代中精神分离离散的行动。

1.  运行前述代码并观察运行时的问题。虽然代码现在在风格上更好，但它存在几个错误，其中一些将创建运行时错误。首先，当我们运行应用程序时，在 Eclipse 中看到以下输出：![图 1.59：Eclipse CDT 中的程序输出](img/C14583_01_59.jpg)

###### 图 1.59：Eclipse CDT 中的程序输出

注意`0`，这意味着我们的代码出了问题。

1.  在控制台手动执行程序。这是我们得到的输出：![图 1.60：带有错误的终端程序输出](img/C14583_01_60.jpg)

###### 图 1.60：带有错误的终端程序输出

不幸的是，我们在 Eclipse 中没有得到分段错误输出，因此您必须在 Eclipse 控制台视图中检查退出值。为了找到问题，我们将在下一步中使用调试器。

1.  在 Eclipse 中按下调试工具栏按钮以启动调试模式下的应用程序。按下继续按钮以继续执行。它将在`SpeedCalculator.cpp`的第 40 行停止，就在错误即将发生时。如果您将鼠标悬停在`speeds`上，您会意识到它是一个无效的内存引用：![图 1.61：无效的内存引用](img/C14583_01_61.jpg)

###### 图 1.61：无效的内存引用

1.  经过进一步检查，我们意识到我们从未将`speeds`指针初始化为任何值。在我们的速度计算器函数中为它分配内存：

```cpp
void SpeedCalculator::calculateAndPrintSpeedData() {
  speeds = new double[numEntries]; // add this line
  double maxSpeed = 0;
```

1.  再次运行。我们得到以下输出：

```cpp
Hello CMake.
Hello from ANewClass.
7
CxxTemplate: SpeedCalculator.cpp:38: void SpeedCalculator::calculateAndPrintSpeedData(): Assertion `dt > 0' failed.
```

请注意，这是一个断言，代码必须确保计算出的`dt`始终大于零。这是我们确信的事情，我们希望它在开发过程中帮助我们捕捉错误。断言语句在生产构建中被忽略，因此您可以在代码中自由地放置它们作为开发过程中捕捉错误的保障。特别是由于 C++缺乏与高级语言相比的许多安全检查，将`assert`语句放置在潜在不安全的代码中有助于捕捉错误。

1.  让我们调查一下为什么我们的`dt`最终没有大于零。为此，我们再次启动调试器。它停在了一个奇怪的地方：![图 1.62：调试器停在没有源代码的库](img/C14583_01_62.jpg)

###### 图 1.62：调试器停在没有源代码的库

1.  实际错误是在库的深处引发的。但是，我们自己的函数仍然在堆栈上，我们可以调查它们在那个时候的状态。单击`dt`变为`i`是`timesInSeconds[10]`，这是数组的不存在的第十一个元素。进一步思考，我们意识到当我们有 10 个位置时，我们只能有 9 个位置对的减法，因此有 9 个速度。这是一个非常常见且难以捕捉的错误，因为 C++不强制您留在数组内。

1.  重新设计我们的整个代码以解决这个问题：

```cpp
void SpeedCalculator::calculateAndPrintSpeedData() {
  speeds = new double[numEntries - 1];
  double maxSpeed = 0;
...
  for (int i = 0; i < numEntries - 1; ++i) {
    double dt = timesInSeconds[i + 1] - timesInSeconds[i];
...
  for (int i = 0; i < numEntries - 1; ++i) {
    double speed = speeds[i];
....
  for (int i = 0; i < numEntries - 1; ++i) {
    double speed = speeds[i];
```

最后，我们的代码似乎可以在没有任何错误的情况下运行，如下面的输出所示：

![图 1.65：程序输出](img/C14583_01_65.jpg)

###### 图 1.65：程序输出

1.  但是，这里有一个奇怪的地方：`0`，无论你运行多少次。为了调查，让我们在以下行放一个断点：![图 1.66：设置断点](img/C14583_01_66.jpg)

###### 图 1.66：设置断点

1.  当我们调试代码时，我们看到它从未停在这里。这显然是错误的。经过进一步调查，我们意识到`minSpeed`最初是 0，而且每个速度值都大于它。我们应该将其初始化为非常大的值，或者我们需要将第一个元素作为最小值。在这里，我们选择第二种方法：

```cpp
for (int i = 0; i < numEntries - 1; ++i) {
  double speed = speeds[i];
  if (i == 0 || maxSpeed < speed) { // changed
    maxSpeed = speed;
  }
  if (i == 0 || minSpeed > speed) { // changed
    minSpeed = speed;
  }
}
```

虽然`maxSpeed`不需要这样做，但保持一致是好的。现在当我们运行代码时，我们看到我们不再得到`0`作为我们的最小速度：

![图 1.67：程序输出](img/C14583_01_67.jpg)

###### 图 1.67：程序输出

1.  我们的代码似乎运行正常。但是，我们又犯了另一个错误。当我们调试代码时，我们发现我们的第一个元素不是零：![图 1.68：变量的值](img/C14583_01_68.jpg)

###### 图 1.68：变量的值

1.  指针解引用了数组中的第一个元素。我们在这里将元素初始化为零，但它们似乎不是零。这是更新后的代码：

```cpp
  // add these two lines:
  timesInSeconds[0] = 0.0;
  positions[0] = 0.0;
  for (int i = 0; i < numEntries; ++i) {
    positions[i] = positions[i - 1] + (rand() % 500);
    timesInSeconds[i] = timesInSeconds[i - 1] + ((rand() % 10) + 1);
  }
```

当我们调查时，我们意识到我们从零开始循环并覆盖了第一个项目。此外，我们尝试访问`positions[0 - 1]`，这是一个错误，也是 C++不强制执行数组边界的另一个例子。当我们让循环从 1 开始时，所有这些问题都消失了：

```cpp
  timesInSeconds[0] = 0.0;
  positions[0] = 0.0;
  for (int i = 1; i < numEntries; ++i) {
    positions[i] = positions[i - 1] + (rand() % 500);
    timesInSeconds[i] = timesInSeconds[i - 1] + ((rand() % 10) + 1);
  }
```

这是使用更新后的代码生成的输出：

![图 1.69：程序输出](img/C14583_01_69.jpg)

###### 图 1.69：程序输出

仅仅通过查看这段代码，我们无法看出区别。这些都是随机值，看起来与以前没有太大不同。这样的错误很难找到，并且可能导致随机行为，使我们难以跟踪错误。您可以避免此类错误的方法包括在解引用指针时特别小心，特别是在循环中；将代码分离为函数并为其编写单元测试；并且在强制执行编译器或运行时不支持的事物时大量使用`assert`语句。

## 第 2A 章 - 不允许鸭子 - 类型和推断

### 活动 1：图形处理

在这个活动中，我们将实现两个类（`Point3d`和`Matrix3d`），以及乘法运算符，以便我们可以转换、缩放和旋转点。我们还将实现一些帮助方法，用于创建所需的转换矩阵。按照以下步骤实现此活动：

1.  从`CMake Build（便携式）`中加载准备好的项目。构建和配置启动器并运行单元测试（失败）。建议用于测试运行程序的名称为`L2AA1graphicstests`。

#### CMake 配置

按照*练习 1*的*步骤 9*，*声明变量和探索大小*，将项目配置为 CMake 项目。

1.  添加一个`Point3d`类的测试，以验证默认构造函数创建一个`原点[0, 0, 0, 1]`。

1.  打开**point3dTests.cpp**文件并在顶部添加以下行。

1.  用以下测试替换失败的现有测试：

```cpp
TEST_F(Point3dTest, DefaultConstructorIsOrigin)
{
    Point3d pt;
    float expected[4] = {0,0,0,1};
    for(size_t i=0 ; i < 4 ; i++)
    {
        ASSERT_NEAR(expected[i], pt(i), Epsilon) << "cell [" << i << "]";
    }
}
```

这个测试要求我们编写一个访问操作符。

1.  用以下代码替换**point3d.hpp**文件中的当前类定义：

```cpp
include <cstddef>
class Point3d
{
public:
    static constexpr size_t NumberRows{4};
    float operator()(const int index) const
    {
        return m_data[index];
    }
private:
    float m_data[NumberRows];
};
```

现在测试可以构建和运行，但是失败了。

1.  在`Point3d`声明中添加默认构造函数的声明：

```cpp
Point3d();
```

1.  将实现添加到**point3d.cpp**文件中：

```cpp
Point3d::Point3d()
{
    for(auto& item : m_data)
    {
        item = 0;
    }
    m_data[NumberRows-1] = 1;
}
```

现在测试可以构建、运行并通过。

1.  添加下一个测试：

```cpp
TEST_F(Point3dTest, InitListConstructor3)
{
    Point3d pt {5.2, 3.5, 6.7};
    float expected[4] = {5.2,3.5,6.7,1};
    for(size_t i=0 ; i < 4 ; i++)
    {
        ASSERT_NEAR(expected[i], pt(i), Epsilon) << "cell [" << i << "]";
    }
}
```

这个测试无法编译。因此，我们需要实现另一个构造函数 - 接受`std::initializer_list<>`作为参数的构造函数。

1.  将以下包含添加到头文件中：

```cpp
#include <initializer_list>
```

1.  在头文件中的`Point3d`类中添加以下构造函数声明：

```cpp
Point3d(std::initializer_list<float> list);
```

1.  将以下代码添加到实现文件中。这段代码忽略了错误处理，这将在*第 3 课*，*Can 和 Should 之间的距离-对象、指针和继承*中添加：

```cpp
Point3d::Point3d(std::initializer_list<float> list)
{
    m_data[NumberRows-1] = 1;
    int i{0};
    for(auto it1 = list.begin(); 
        i<NumberRows && it1 != list.end();
        ++it1, ++i)
    {
        m_data[i] = *it1;
    }
}
```

现在测试应该构建、运行并通过。

1.  添加以下测试：

```cpp
TEST_F(Point3dTest, InitListConstructor4)
{
    Point3d pt {5.2, 3.5, 6.7, 2.0};
    float expected[4] = {5.2,3.5,6.7,2.0};
    for(size_t i=0 ; i < 4 ; i++)
    {
        ASSERT_NEAR(expected[i], pt(i), Epsilon) << "cell [" << i << "]";
    }
}
```

测试应该仍然构建、运行并通过。

1.  现在是时候通过将验证循环移动到`Point3dTest`类中的模板函数来重构测试用例了。在这个类中添加以下模板：

```cpp
template<size_t size>
void VerifyPoint(Point3d& pt, float (&expected)[size])
{
    for(size_t i=0 ; i< size ; i++)
    {
        ASSERT_NEAR(expected[i], pt(i), Epsilon) << "cell [" << i << "]";
    }
}
```

1.  这意味着最后一个测试现在可以重写如下：

```cpp
TEST_F(Point3dTest, InitListConstructor4)
{
    Point3d pt {5.2, 3.5, 6.7, 2.0};
    float expected[4] = {5.2,3.5,6.7,2.0};
    VerifyPoint(pt, expected);
}
```

与生产代码一样，保持测试的可读性同样重要。

1.  接下来，通过以下测试添加相等和不相等运算符的支持：

```cpp
TEST_F(Point3dTest, EqualityOperatorEqual)
{
    Point3d pt1 {1,3,5};
    Point3d pt2 {1,3,5};
    ASSERT_EQ(pt1, pt2);
}
TEST_F(Point3dTest, EqualityOperatorNotEqual)
{
    Point3d pt1 {1,2,3};
    Point3d pt2 {1,2,4};
    ASSERT_NE(pt1, pt2);
}
```

1.  为了实现这些，添加以下声明/定义到头文件中：

```cpp
bool operator==(const Point3d& rhs) const;
bool operator!=(const Point3d& rhs) const
{
    return !operator==(rhs);
}
```

1.  现在，在.cpp 文件中添加相等性的实现：

```cpp
bool Point3d::operator==(const Point3d& rhs) const
{
    for(int i=0 ; i<NumberRows ; i++)
    {
        if (m_data[i] != rhs.m_data[i])
        {
            return false;
        }
    }
    return true;
}
```

1.  当我们首次添加`Point3d`时，我们实现了一个常量访问器。添加以下测试，我们需要一个非常量访问器，以便我们可以将其分配给成员：

```cpp
TEST_F(Point3dTest, AccessOperator)
{
    Point3d pt1;
    Point3d pt2 {1,3,5};
    pt1(0) = 1;
    pt1(1) = 3;
    pt1(2) = 5;
    ASSERT_EQ(pt1, pt2);
}
```

1.  为了使这个测试能够构建，添加以下访问器到头文件中：

```cpp
float& operator()(const int index)
{
    return m_data[index];
}
```

注意它返回一个引用。因此，我们可以将其分配给一个成员值。

1.  为了完成`Point3d`，在类声明中添加默认复制构造函数和复制赋值：

```cpp
Point3d(const Point3d&) = default;
Point3d& operator=(const Point3d&) = default;
```

1.  现在，添加`Matrix3d`类。首先，在当前项目的顶层文件夹中创建两个空文件，`matrix3d.hpp`和`matrix3d.cpp`，然后在 tests 文件夹中添加一个名为`matrix3dTests.cpp`的空文件。

1.  打开顶层文件夹中的 CmakeLists.txt 文件，并将**matrix3d.cpp**添加到以下行：

```cpp
add_executable(graphics point3d.cpp main.cpp matrix3d.cpp)
```

1.  打开`../matrix3d.cpp`到`SRC_FILES`的定义，并添加`TEST_FILES`：

```cpp
SET(SRC_FILES 
    ../matrix3d.cpp
    ../point3d.cpp)
SET(TEST_FILES 
    matrix3dTests.cpp
    point3dTests.cpp)
```

如果你正确地进行了这些更改，现有的`point3d`测试应该仍然能够构建、运行和通过。

1.  在`matrix3dTests.cpp`中添加以下测试管道：

```cpp
#include "gtest/gtest.h"
#include "../matrix3d.hpp"
class Matrix3dTest : public ::testing::Test
{
public:
};
TEST_F(Matrix3dTest, DummyTest)
{
    ASSERT_TRUE(false);
}
```

1.  构建并运行测试。我们刚刚添加的测试应该失败。

1.  在`Matrix3d`类中用以下测试替换 DummyTest。我们现在将在**matrix3d.hpp**中进行此操作。

1.  在**matrix3d.hpp**中添加以下定义：

```cpp
class Matrix3d
{
public:
    float operator()(const int row, const int column) const
    {
        return m_data[row][column];
    }
private:
    float m_data[4][4];
};
```

现在测试将构建，但仍然失败，因为我们还没有创建一个创建单位矩阵的默认构造函数。

1.  在`Matrix3d`的公共部分的头文件中添加默认构造函数的声明：

```cpp
Matrix3d();
```

1.  将此定义添加到**matrix3d.cpp**中：

```cpp
#include "matrix3d.hpp"
Matrix3d::Matrix3d()
{
    for (int i{0} ; i< 4 ; i++)
        for (int j{0} ; j< 4 ; j++)
            m_data[i][j] = (i==j);
}
```

现在测试已经构建并通过。

1.  稍微重构代码以使其更易读。修改头文件如下：

```cpp
#include <cstddef>   // Required for size_t definition
class Matrix3d
{
public:
    static constexpr size_t NumberRows{4};
    static constexpr size_t NumberColumns{4};
    Matrix3d();
    float operator()(const int row, const int column) const
    {
    return m_data[row][column];
    }
private:
    float m_data[NumberRows][NumberColumns];
};
```

1.  更新**matrix3d.cpp**文件以使用常量：

```cpp
Matrix3d::Matrix3d()
{
    for (int i{0} ; i< NumberRows ; i++)
        for (int j{0} ; j< NumberColumns ; j++)
            m_data[i][j] = (i==j);
}
```

1.  重新构建测试并确保它们仍然通过。

1.  现在，我们需要添加初始化程序列表构造函数。为此，添加以下测试：

```cpp
TEST_F(Matrix3dTest, InitListConstructor)
{
    Matrix3d mat{ {1,2,3,4}, {5,6,7,8},{9,10,11,12}, {13,14,15,16}};
    int expected{1};
    for( int row{0} ; row<4 ; row++)
        for( int col{0} ; col<4 ; col++, expected++)
        {
            ASSERT_FLOAT_EQ(expected, mat(row,col)) << "cell[" << row << "][" << col << "]";
        }
}
```

1.  为初始化程序列表支持添加包含文件并在**matrix3d.hpp**中声明构造函数：

```cpp
#include <initializer_list>
class Matrix3d
{
public:
    Matrix3d(std::initializer_list<std::initializer_list<float>> list);
```

1.  最后，在.cpp 文件中添加构造函数的实现：

```cpp
Matrix3d::Matrix3d(std::initializer_list<std::initializer_list<float>> list)
{
    int i{0};
    for(auto it1 = list.begin(); i<NumberRows ; ++it1, ++i)
    {
        int j{0};
        for(auto it2 = it1->begin(); j<NumberColumns ; ++it2, ++j)
            m_data[i][j] = *it2;
    }
}
```

1.  为了改善我们测试的可读性，在测试框架中添加一个辅助方法。在`Matrix3dTest`类中声明以下内容：

```cpp
static constexpr float Epsilon{1e-12};
void VerifyMatrixResult(Matrix3d& expected, Matrix3d& actual);
```

1.  添加辅助方法的定义：

```cpp
void Matrix3dTest::VerifyMatrixResult(Matrix3d& expected, Matrix3d& actual)
{
    for( int row{0} ; row<4 ; row++)
        for( int col{0} ; col<4 ; col++)
        {
        ASSERT_NEAR(expected(row,col), actual(row,col), Epsilon) 
<< "cell[" << row << "][" << col << "]";
        }
}
```

1.  编写一个测试，将两个矩阵相乘并得到一个新的矩阵（预期将手动计算）：

```cpp
TEST_F(Matrix3dTest, MultiplyTwoMatricesGiveExpectedResult)
{
    Matrix3d mat1{ {5,6,7,8}, {9,10,11,12}, {13,14,15,16}, {17,18,19,20}};
    Matrix3d mat2{ {1,2,3,4}, {5,6,7,8},    {9,10,11,12},  {13,14,15,16}};
    Matrix3d expected{ {202,228,254,280},
                       {314,356,398,440},
                       {426,484,542,600},
                       {538,612,686,760}};
    Matrix3d result = mat1 * mat2;
    VerifyMatrixResult(expected, result);
}
```

1.  在头文件中定义`operator*=`：

```cpp
Matrix3d& operator*=(const Matrix3d& rhs);
```

然后，在类声明之外实现`operator*`的内联版本：

```cpp
inline Matrix3d operator*(const Matrix3d& lhs, const Matrix3d& rhs)
{
    Matrix3d temp(lhs);
    temp *= rhs;
    return temp;
}
```

1.  以及在**matrix3d.cpp**文件中的实现：

```cpp
Matrix3d& Matrix3d::operator*=(const Matrix3d& rhs)
{
    Matrix3d temp;
    for(int i=0 ; i<NumberRows ; i++)
        for(int j=0 ; j<NumberColumns ; j++)
        {
            temp.m_data[i][j] = 0;
            for (int k=0 ; k<NumberRows ; k++)
                temp.m_data[i][j] += m_data[i][k] * rhs.m_data[k][j];
        }
    *this = temp;
    return *this;
}
```

1.  构建并运行测试-再次，它们应该通过。

1.  通过在`Matrix3dTest`类中声明第二个辅助函数来引入测试类的辅助函数：

```cpp
void VerifyMatrixIsIdentity(Matrix3d& mat);
```

然后，声明它以便我们可以使用它：

```cpp
void Matrix3dTest::VerifyMatrixIsIdentity(Matrix3d& mat)
{
for( int row{0} ; row<4 ; row++)
    for( int col{0} ; col<4 ; col++)
    {
        int expected = (row==col) ? 1 : 0;
        ASSERT_FLOAT_EQ(expected, mat(row,col)) 
                             << "cell[" << row << "][" << col << "]";
    }
}
```

1.  更新一个测试以使用它：

```cpp
TEST_F(Matrix3dTest, DefaultConstructorIsIdentity)
{
    Matrix3d mat;
    VerifyMatrixIsIdentity(mat);
}
```

1.  编写一个健全性检查测试：

```cpp
TEST_F(Matrix3dTest, IdentityTimesIdentityIsIdentity)
{
    Matrix3d mat;
    Matrix3d result = mat * mat;
    VerifyMatrixIsIdentity(result);
}
```

1.  构建并运行测试-它们应该仍然通过。

1.  现在，我们需要能够将点和矩阵相乘。添加以下测试：

```cpp
TEST_F(Matrix3dTest, MultiplyMatrixWithPoint)
{
    Matrix3d mat { {1,2,3,4}, {5,6,7,8},    {9,10,11,12},  {13,14,15,16}};
    Point3d pt {15, 25, 35, 45};
    Point3d expected{350, 830, 1310, 1790};
    Point3d pt2 = mat * pt;
    ASSERT_EQ(expected, pt2);
}
```

1.  在`Matrix3d`类声明中：

```cpp
Point3d operator*(const Matrix3d& lhs, const Point3d& rhs);
```

1.  在**matrix3d.cpp**文件中添加运算符的定义：

```cpp
Point3d operator*(const Matrix3d& lhs, const Point3d& rhs)
{
    Point3d pt;
    for(int row{0} ; row<Matrix3d::NumberRows ; row++)
    {
        float sum{0};
        for(int col{0} ; col<Matrix3d::NumberColumns ; col++)
        {
            sum += lhs(row, col) * rhs(col);
        }
        pt(row) = sum;
    }
    return pt;
}
```

1.  构建并运行测试。它们应该再次全部通过。

1.  在**matrix3dtests.cpp**的顶部，添加包含文件：

```cpp
#include <cmath>
```

1.  开始添加转换矩阵工厂方法。使用以下测试，我们将开发各种工厂方法（测试应逐个添加）：

```cpp
TEST_F(Matrix3dTest, CreateTranslateIsCorrect)
{
    Matrix3d mat = createTranslationMatrix(-0.5, 2.5, 10.0);
    Matrix3d expected {{1.0, 0.0, 0.0, -0.5},
                       {0.0, 1.0, 0.0, 2.5},
                       {0.0, 0.0, 1.0, 10.0},
                       {0.0, 0.0, 0.0, 1.0}
    };
    VerifyMatrixResult(expected, mat);
}
TEST_F(Matrix3dTest, CreateScaleIsCorrect)
{
    Matrix3d mat = createScaleMatrix(3.0, 2.5, 11.0);
    Matrix3d expected {{3.0, 0.0,  0.0, 0.0},
                       {0.0, 2.5,  0.0, 0.0},
                       {0.0, 0.0, 11.0, 0.0},
                       {0.0, 0.0,  0.0, 1.0}
    };	
    VerifyMatrixResult(expected, mat);
}
TEST_F(Matrix3dTest, CreateRotateX90IsCorrect)
{
    Matrix3d mat = createRotationMatrixAboutX(90.0F);
    Matrix3d expected {{1.0, 0.0,  0.0, 0.0},
                       {0.0, 0.0, -1.0, 0.0},
                       {0.0, 1.0,  0.0, 0.0},
                       {0.0, 0.0,  0.0, 1.0}
    };
    VerifyMatrixResult(expected, mat);
}
TEST_F(Matrix3dTest, CreateRotateX60IsCorrect)
{
    Matrix3d mat = createRotationMatrixAboutX(60.0F);
    float sqrt3_2 = static_cast<float>(std::sqrt(3.0)/2.0);
    Matrix3d expected {{1.0, 0.0,     0.0,     0.0},
                       {0.0, 0.5,    -sqrt3_2, 0.0},
                       {0.0, sqrt3_2,  0.5,    0.0},
                       {0.0, 0.0,     0.0,     1.0}
    };
    VerifyMatrixResult(expected, mat);
}
TEST_F(Matrix3dTest, CreateRotateY90IsCorrect)
{
    Matrix3d mat = createRotationMatrixAboutY(90.0F);
    Matrix3d expected {{0.0, 0.0,  1.0, 0.0},
                       {0.0, 1.0,  0.0, 0.0},
                       {-1.0, 0.0, 0.0, 0.0},
                       {0.0, 0.0,  0.0, 1.0}
    };
    VerifyMatrixResult(expected, mat);
}
TEST_F(Matrix3dTest, CreateRotateY60IsCorrect)
{
    Matrix3d mat = createRotationMatrixAboutY(60.0F);
    float sqrt3_2 = static_cast<float>(std::sqrt(3.0)/2.0);
    Matrix3d expected {{0.5,      0.0,   sqrt3_2,  0.0},
                       {0.0,      1.0,    0.0,     0.0},
                       {-sqrt3_2, 0.0,    0.5,     0.0},
                       {0.0,      0.0,    0.0,     1.0}
    };
    VerifyMatrixResult(expected, mat);
}
TEST_F(Matrix3dTest, CreateRotateZ90IsCorrect)
{
    Matrix3d mat = createRotationMatrixAboutZ(90.0F);
    Matrix3d expected {{0.0, -1.0,  0.0, 0.0},
                       {1.0, 0.0,  0.0, 0.0},
                       {0.0, 0.0,  1.0, 0.0},
                       {0.0, 0.0,  0.0, 1.0}
    };
    VerifyMatrixResult(expected, mat);
}
TEST_F(Matrix3dTest, CreateRotateZ60IsCorrect)
{
    Matrix3d mat = createRotationMatrixAboutZ(60.0F);
    float sqrt3_2 = static_cast<float>(std::sqrt(3.0)/2.0);
    Matrix3d expected {{0.5,     -sqrt3_2,   0.0,  0.0},
                       {sqrt3_2,      0.5,   0.0,  0.0},
                       {0.0,          0.0,   1.0,  0.0},
                       {0.0,          0.0,   0.0,  1.0}
    };
    VerifyMatrixResult(expected, mat);
}
```

1.  将以下声明添加到 matrix3d 头文件中：

```cpp
Matrix3d createTranslationMatrix(float dx, float dy, float dz);
Matrix3d createScaleMatrix(float sx, float sy, float sz);
Matrix3d createRotationMatrixAboutX(float degrees);
Matrix3d createRotationMatrixAboutY(float degrees);
Matrix3d createRotationMatrixAboutZ(float degrees);
```

1.  在 matrix3d 实现文件的顶部添加`#include <cmath>`。

1.  最后，将以下实现添加到`matrix3d`实现文件中：

```cpp
Matrix3d createTranslationMatrix(float dx, float dy, float dz)
{
    Matrix3d matrix;
    matrix(0, 3) = dx;
    matrix(1, 3) = dy;
    matrix(2, 3) = dz;
    return matrix;
}
Matrix3d createScaleMatrix(float sx, float sy, float sz)
{
    Matrix3d matrix;
    matrix(0, 0) = sx;
    matrix(1, 1) = sy;
    matrix(2, 2) = sz;
    return matrix;
}
Matrix3d createRotationMatrixAboutX(float degrees)
{
    Matrix3d matrix;
    double pi{4.0F*atan(1.0F)};
    double radians = degrees / 180.0 * pi;
    float cos_theta = static_cast<float>(cos(radians));
    float sin_theta = static_cast<float>(sin(radians));
    matrix(1, 1) =  cos_theta;
    matrix(2, 2) =  cos_theta;
    matrix(1, 2) = -sin_theta;
    matrix(2, 1) =  sin_theta;
    return matrix;
}
Matrix3d createRotationMatrixAboutY(float degrees)
{
    Matrix3d matrix;
    double pi{4.0F*atan(1.0F)};
    double radians = degrees / 180.0 * pi;
    float cos_theta = static_cast<float>(cos(radians));
    float sin_theta = static_cast<float>(sin(radians));
    matrix(0, 0) =  cos_theta;
    matrix(2, 2) =  cos_theta;
    matrix(0, 2) =  sin_theta;
    matrix(2, 0) = -sin_theta;
    return matrix;
}
Matrix3d createRotationMatrixAboutZ(float degrees)
{
    Matrix3d matrix;
    double pi{4.0F*atan(1.0F)};
    double radians = degrees / 180.0 * pi;
    float cos_theta = static_cast<float>(cos(radians));
    float sin_theta = static_cast<float>(sin(radians));
    matrix(0, 0) =  cos_theta;
    matrix(1, 1) =  cos_theta;
    matrix(0, 1) = -sin_theta;
    matrix(1, 0) =  sin_theta;
    return matrix;
}
```

1.  为了使其编译并通过测试，我们需要在`matrix3d`的声明中添加一个访问器：

```cpp
float& operator()(const int row, const int column)
{
    return m_data[row][column];
}
```

1.  再次构建并运行所有测试，以显示它们都通过了。

1.  在`point3d.hpp`中，添加`<ostream>`的包含，并在 Point3d 类末尾添加以下友元声明：

```cpp
friend std::ostream& operator<<(std::ostream& , const Point3d& );
```

1.  在类之后编写操作符的内联实现：

```cpp
inline std::ostream&
operator<<(std::ostream& os, const Point3d& pt)
{
    const char* sep = "[ ";
    for(auto value : pt.m_data)
    {
        os << sep  << value;
        sep = ", ";
    }
    os << " ]";
    return os;
}
```

1.  打开**main.cpp**文件，并从以下行中删除注释分隔符，//：

```cpp
//#define ACTIVITY1
```

1.  构建并运行名为`graphics`的应用程序 - 您需要创建一个新的运行配置。如果您的`Point3d`和`Matrix3d`的实现正确，那么程序将显示以下输出：

![](img/C14583_02A_53.jpg)

###### 图 2A.53：成功运行活动程序

在这个活动中，我们实现了两个类，这两个类是实现 3D 图形渲染所需的所有操作的基础。我们使用运算符重载来实现这一点，以便 Matrix3d 和 Point3d 可以像本机类型一样使用。如果我们希望操作整个对象，这可以很容易地扩展到处理点的向量。

## 第 2B 章 - 不允许鸭子 - 模板和推断

### 活动 1：开发通用的“contains”模板函数

在这个活动中，我们将实现几个辅助类，用于检测`std::string`类情况和`std::set`情况，然后使用它们来调整包含函数以适应特定容器。按照以下步骤实现此活动：

1.  从`L2BA1tests`加载准备好的项目。

1.  打开**containsTests.cpp**文件，并用以下内容替换现有测试：

```cpp
TEST_F(containsTest, DetectNpos)
{
    ASSERT_TRUE(has_npos_v<std::string>);
    ASSERT_FALSE(has_npos_v<std::set<int>>);
    ASSERT_FALSE(has_npos_v<std::vector<int>>);
}
```

这个测试要求我们编写一组辅助模板，以检测容器类是否支持名为 npos 的静态成员变量。

1.  将以下代码添加到**contains.hpp**文件中：

```cpp
template <class T>
auto test_npos(int) -> decltype((void)T::npos, std::true_type{});
template <class T>
auto test_npos(long) -> std::false_type;
template <class T>
struct has_npos : decltype(test_npos<T>(0)) {};
template< class T >
inline constexpr bool has_npos_v = has_npos<T>::value;
```

现在测试运行并通过。

1.  将以下测试添加到接受一个参数的`find()`方法中。

1.  将以下代码添加到**contains.hpp**文件中：

```cpp
template <class T, class A0>
auto test_find(int) -> 
       decltype(void(std::declval<T>().find(std::declval<A0>())), 
                                                        std::true_type{});
template <class T, class A0>
auto test_find(long) -> std::false_type;
template <class T, class A0>
struct has_find : decltype(test_find<T,A0>(0)) {};
template< class T, class A0 >
inline constexpr bool has_find_v = has_find<T, A0>::value;
```

现在测试运行并通过。

1.  添加通用容器的实现；在这种情况下，是向量。在**containsTest.cpp**文件中编写以下测试：

```cpp
TEST_F(containsTest, VectorContains)
{
    std::vector<int> container {1,2,3,4,5};
    ASSERT_TRUE(contains(container, 5));
    ASSERT_FALSE(contains(container, 15));
}
```

1.  将`contains`的基本实现添加到**contains.hpp**文件中：

```cpp
template<class C, class T>
auto contains(const C& c, const T& key) -> decltype(std::end(c), true)
{
        return std::end(c) != std::find(begin(c), end(c), key);
}
```

现在测试运行并通过。

1.  下一步是为`set`特殊情况添加测试到**containsTest.cpp**：

```cpp
TEST_F(containsTest, SetContains)
{
    std::set<int> container {1,2,3,4,5};
    ASSERT_TRUE(contains(container, 5));
    ASSERT_FALSE(contains(container, 15));
}
```

1.  更新`contains`的实现以测试内置的`set::find()`方法：

```cpp
template<class C, class T>
auto contains(const C& c, const T& key) -> decltype(std::end(c), true)
{
    if constexpr(has_find_v<C, T>)
    {
        return std::end(c) != c.find(key);
    }
    else
    {
        return std::end(c) != std::find(begin(c), end(c), key);
    }
}
```

现在测试运行并通过。

1.  将`string`特殊情况的测试添加到**containsTest.cpp**文件中：

```cpp
TEST_F(containsTest, StringContains)
{
    std::string container{"This is the message"};
    ASSERT_TRUE(contains(container, "the"));
    ASSERT_TRUE(contains(container, 'm'));
    ASSERT_FALSE(contains(container, "massage"));
    ASSERT_FALSE(contains(container, 'z'));
}
```

1.  添加以下`contains`的实现以测试`npos`的存在并调整`find()`方法的使用：

```cpp
template<class C, class T>
auto contains(const C& c, const T& key) -> decltype(std::end(c), true)
{
    if constexpr(has_npos_v<C>)
    {
        return C::npos != c.find(key);
    }
    else
    if constexpr(has_find_v<C, T>)
    {
        return std::end(c) != c.find(key);
    }
    else
    {
        return std::end(c) != std::find(begin(c), end(c), key);
    }
}
```

现在测试运行并通过。

1.  构建并运行名为`contains`的应用程序。创建一个新的运行配置。如果您的 contains 模板实现正确，那么程序将显示以下输出：

![图 2B.36：包含成功实现的输出](img/C14583_02B_36.jpg)

###### 图 2B.36：包含成功实现的输出

在这个活动中，我们使用各种模板技术与 SFINAE 结合使用，根据包含类的能力选择`contains()`函数的适当实现。我们可以使用通用模板函数和一些专门的模板来实现相同的结果，但我们选择了不太常见的路径，并展示了我们新发现的模板技能。

## 第三章 - 能与应该之间的距离 - 对象，指针和继承

### 活动 1：使用 RAII 和 Move 实现图形处理

在这个活动中，我们将开发我们之前的`Matrix3d`和`Point3d`类，以使用`unique_ptr<>`来管理与实现这些图形类所需的数据结构相关联的内存。让我们开始吧：

1.  从**Lesson3/Activity01**文件夹加载准备好的项目，并为项目配置当前构建器为**CMake Build (Portable)**。构建和配置启动器并运行单元测试。我们建议为测试运行器使用的名称是**L3A1graphicstests**。

1.  打开`acpp::gfx`，这是 C++17 的一个新特性。以前，它需要显式使用`namespace`关键字两次。另外，请注意，为了提供帮助，您友好的邻里 IDE 可能会在您放置命名空间声明的那一行后面立即插入闭括号。

1.  对**matrix3d.hpp**、**matrix3d.cpp**和**point3d.cpp**执行相同的处理-确保包含文件不包含在命名空间的范围内。

1.  在各自的文件（**main.cpp**、**matrix3dTests.cpp**和**point3dTests.cpp**）中，在完成#include 指令后，插入以下行：

```cpp
using namespace acpp::gfx;
```

1.  现在，运行所有测试。所有**18**个现有测试应该再次通过。我们已经成功地将我们的类放入了一个命名空间。

1.  现在我们将转而将`Matrix3d`类转换为使用堆分配的内存。在`#include <memory>`行中，以便我们可以访问`unique_ptr<>`模板。

1.  接下来，更改声明`m_data`的类型：

```cpp
std::unique_ptr<float[]> m_data;
```

1.  从这一点开始，我们将使用编译器及其错误来提示我们需要修复的问题。尝试构建测试现在会显示我们在头文件中有以下两个方法存在问题。

```cpp
float operator()(const int row, const int column) const
{
    return m_data[row][column];
}
float& operator()(const int row, const int column)
{
    return m_data[row][column];
} 
```

问题在于`unique_ptr`保存了一个指向单维数组而不是二维数组的指针。因此，我们需要将行和列转换为一个单一的索引。

1.  添加一个名为`get_index()`的新方法，以从行和列获取一维索引，并更新前面的函数以使用它：

```cpp
float operator()(const int row, const int column) const
{
    return m_data[get_index(row,column)];
}
float& operator()(const int row, const int column)
{
    return m_data[get_index(row,column)];
}
private:
size_t get_index(const int row, const int column) const
{
    return row * NumberColumns + column;
}
```

1.  重新编译后，编译器给出的下一个错误是关于以下内联函数：

```cpp
inline Matrix3d operator*(const Matrix3d& lhs, const Matrix3d& rhs)
{
    Matrix3d temp(lhs);   // <=== compiler error – ill formed copy constructor
    temp *= rhs;
    return temp;
}
```

1.  以前，默认的复制构造函数对我们的目的已经足够了，它只是对数组的所有元素进行了浅复制，这是正确的。现在我们需要复制的数据有了间接引用，因此我们需要实现一个深复制构造函数和复制赋值。我们还需要处理现有的构造函数。现在，只需将构造函数声明添加到类中（与其他构造函数相邻）：

```cpp
Matrix3d(const Matrix3d& rhs);
Matrix3d& operator=(const Matrix3d& rhs);
```

尝试构建测试现在将显示我们已解决头文件中的所有问题，并且可以继续进行实现文件。

1.  修改两个构造函数以初始化`unique_ptr`如下：

```cpp
Matrix3d::Matrix3d() : m_data{new float[NumberRows*NumberColumns]}
{
    for (int i{0} ; i< NumberRows ; i++)
        for (int j{0} ; j< NumberColumns ; j++)
            m_data[i][j] = (i==j);
}
Matrix3d::Matrix3d(std::initializer_list<std::initializer_list<float>> list)
    : m_data{new float[NumberRows*NumberColumns]}
{
    int i{0};
    for(auto it1 = list.begin(); i<NumberRows ; ++it1, ++i)
    {
        int j{0};
        for(auto it2 = it1->begin(); j<NumberColumns ; ++it2, ++j)
            m_data[i][j] = *it2;
    }
}
```

1.  现在我们需要解决单维数组查找的问题。我们需要将`m_data[i][j]`类型的语句更改为`m_data[get_index(i,j)]`。将默认构造函数更改为以下内容：

```cpp
Matrix3d::Matrix3d() : m_data{new float[NumberRows*NumberColumns]}
{
    for (int i{0} ; i< NumberRows ; i++)
        for (int j{0} ; j< NumberColumns ; j++)
            m_data[get_index(i, j)] = (i==j);          // <= change here
}
```

1.  更改初始化列表构造函数如下：

```cpp
Matrix3d::Matrix3d(std::initializer_list<std::initializer_list<float>> list)
      : m_data{new float[NumberRows*NumberColumns]}
{
    int i{0};
    for(auto it1 = list.begin(); i<NumberRows ; ++it1, ++i)
    {
        int j{0};
        for(auto it2 = it1->begin(); j<NumberColumns ; ++it2, ++j)
            m_data[get_index(i, j)] = *it2;         // <= change here
    }
}
```

1.  更改乘法运算符，注意索引：

```cpp
Matrix3d& Matrix3d::operator*=(const Matrix3d& rhs)
{
    Matrix3d temp;
    for(int i=0 ; i<NumberRows ; i++)
        for(int j=0 ; j<NumberColumns ; j++)
        {
            temp.m_data[get_index(i, j)] = 0;        // <= change here
            for (int k=0 ; k<NumberRows ; k++)
                temp.m_data[get_index(i, j)] += m_data[get_index(i, k)] 
                                          * rhs.m_data[get_index(k, j)];
                                                     // <= change here
        }
    *this = temp;
    return *this;
}
```

1.  通过这些更改，我们已经修复了所有的编译错误，但现在我们有一个链接器错误要处理-我们只在第 11 步中声明了复制构造函数。

1.  在**matrix3d.cpp**文件中添加以下定义：

```cpp
Matrix3d::Matrix3d(const Matrix3d& rhs) : 
    m_data{new float[NumberRows*NumberColumns]}
{
    *this = rhs;
}
Matrix3d& Matrix3d::operator=(const Matrix3d& rhs)
{
    for(int i=0 ; i< NumberRows*NumberColumns ; i++)
        m_data[i] = rhs.m_data[i];
    return *this;
}
```

1.  现在测试将会构建，并且所有测试都会通过。下一步是强制移动构造函数。在**matrix3d.cpp**中找到`createTranslationMatrix()`方法，并将返回语句更改如下：

```cpp
return std::move(matrix);
```

1.  在`move`构造函数中。

```cpp
Matrix3d(Matrix3d&& rhs);
```

1.  重新构建测试。现在，我们得到了一个与移动构造函数不存在相关的错误。

1.  将构造函数的实现添加到**matrix3d.cpp**中，并重新构建测试。

```cpp
Matrix3d::Matrix3d(Matrix3d&& rhs)
{
    //std::cerr << "Matrix3d::Matrix3d(Matrix3d&& rhs)\n";
    std::swap(m_data, rhs.m_data);
}
```

1.  重新构建并运行测试。它们都会再次通过。

1.  为了确认移动构造函数是否被调用，将`#include <iostream>`添加到`cerr`中。检查后，再将该行注释掉。

#### 注意

关于移动构造函数的一个快速说明-我们没有像其他构造函数那样显式初始化`m_data`。这意味着它将被初始化为空，然后与传入的参数交换，这是一个临时的，所以它可以不保存数组在事务之后-它删除了一次内存的分配和释放。

1.  现在让我们转换`Point3d`类，以便它可以使用堆分配的内存。在`#include <memory>`行中添加，以便我们可以访问`unique_ptr<>`模板。

1.  接下来，更改`m_data`的声明类型如下：

```cpp
std::unique_ptr<float[]> m_data;
```

1.  编译器现在告诉我们，在`unique_ptr`的插入运算符（<<）中存在问题：用以下内容替换实现：

```cpp
inline std::ostream&
operator<<(std::ostream& os, const Point3d& pt)
{
    const char* sep = "[ ";
    for(int i{0} ; i < Point3d::NumberRows ; i++)
    {
        os << sep << pt.m_data[i];
        sep = ", ";
    }
    os << " ]";
    return os;
} 
```

1.  打开`unique_ptr`并更改初始化循环，因为`unique_ptr`不能使用范围 for：

```cpp
Point3d::Point3d() : m_data{new float[NumberRows]}
{
    for(int i{0} ; i < NumberRows-1 ; i++) {
        m_data[i] = 0;
    }
    m_data[NumberRows-1] = 1;
}
```

1.  通过初始化`unique_ptr`修改另一个构造函数：

```cpp
Point3d::Point3d(std::initializer_list<float> list)
            : m_data{new float[NumberRows]}
```

1.  现在所有的测试都运行并通过，就像以前一样。

1.  现在，如果我们运行原始应用程序**L3graphics**，那么输出将与原始输出相同，但是该实现使用 RAII 来分配和管理用于矩阵和点的内存。

![](img/C14583_03_52.jpg)

###### 图 3.52：成功转换为使用 RAII 后的活动 1 输出

## 活动 2：实现日期计算的类

在这个活动中，我们将实现两个类，`Date`和`Days`，这将使我们非常容易处理日期和它们之间的时间差异。让我们开始吧：

1.  从**Lesson3/Activity02**文件夹加载准备好的项目，并配置项目的当前构建器为**CMake Build (Portable)**。构建和配置启动器并运行单元测试。我们建议为测试运行器使用的名称是**L3A2datetests**。该项目有虚拟文件和一个失败的测试。

1.  打开`Date`类以允许访问存储的值：

```cpp
int Day()   const {return m_day;}
int Month() const {return m_month;}
int Year()  const {return m_year;}
```

1.  打开`DateTest`类：

```cpp
void VerifyDate(const Date& dt, int yearExp, int monthExp, int dayExp) const
{
    ASSERT_EQ(dayExp, dt.Day());
    ASSERT_EQ(monthExp, dt.Month());
    ASSERT_EQ(yearExp, dt.Year());
}
```

通常情况下，随着测试的发展，您会重构这个测试，但我们将它提前拉出来。

1.  用以下测试替换现有测试中的`ASSERT_FALSE()`：

```cpp
Date dt;
VerifyDate(dt, 1970, 1, 1);
```

1.  重建并运行测试-现在它们应该全部通过。

1.  添加以下测试：

```cpp
TEST_F(DateTest, Constructor1970Jan2)
{
    Date dt(2, 1, 1970);
    VerifyDate(dt, 1970, 1, 2);
}
```

1.  为了进行这个测试，我们需要向`Date`类添加以下两个构造函数：

```cpp
Date() = default;
Date(int day, int month, int year) :
        m_year{year}, m_month{month}, m_day{day}
{
}
```

1.  现在我们需要引入函数来转换`date_t`类型。在我们的命名空间内的**date.hpp**文件中添加以下别名：

```cpp
using date_t=int64_t;
```

1.  在`Date`类中，添加以下方法的声明：

```cpp
date_t ToDateT() const;
```

1.  然后，添加以下测试：

```cpp
TEST_F(DateTest, ToDateTDefaultIsZero)
{
    Date dt;
    ASSERT_EQ(0, dt.ToDateT());
}
```

1.  由于我们正在进行（`TDD`），我们添加方法的最小实现以通过测试。

```cpp
date_t Date::ToDateT() const
{
    return 0;
}
```

1.  现在，我们添加下一个测试：

```cpp
TEST_F(DateTest, ToDateT1970Jan2Is1)
{
    Date dt(2, 1, 1970);
    ASSERT_EQ(1, dt.ToDateT());
}
```

1.  我们继续添加一个测试，然后另一个，一直在不断完善`ToDateT()`中的算法，首先处理`1970`年的日期，然后是`1971 年 1 月 1 日`，然后是`1973`年的日期，这意味着我们跨越了一个闰年，依此类推。用于开发`ToDateT()`方法的完整测试集如下：

```cpp
TEST_F(DateTest, ToDateT1970Dec31Is364)
{
    Date dt(31, 12, 1970);
    ASSERT_EQ(364, dt.ToDateT());
}
TEST_F(DateTest, ToDateT1971Jan1Is365)
{
    Date dt(1, 1, 1971);
    ASSERT_EQ(365, dt.ToDateT());
}
TEST_F(DateTest, ToDateT1973Jan1Is1096)
{
    Date dt(1, 1, 1973);
    ASSERT_EQ(365*3+1, dt.ToDateT());
}
TEST_F(DateTest, ToDateT2019Aug28Is18136)
{
    Date dt(28, 8, 2019);
    ASSERT_EQ(18136, dt.ToDateT());
}
```

1.  为了通过所有这些测试，我们向`Date`类的声明中添加以下内容：

```cpp
public:
    static constexpr int EpochYear = 1970;
    static constexpr int DaysPerCommonYear = 365;
    static constexpr int YearsBetweenLeapYears = 4;
private:
    int GetDayOfYear(int day, int month, int year) const;
    bool IsLeapYear(int year) const;
    int CalcNumberLeapYearsFromEpoch(int year) const;
```

1.  **date.cpp**中`ToDateT()`的实现和支持方法如下：

```cpp
namespace {
int daysBeforeMonth[2][12] =
{
    { 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 204, 334}, // Common Year
    { 0, 31, 50, 91, 121, 152, 182, 213, 244, 274, 205, 335}  // Leap Year
};
}
namespace acpp::date
{
int Date::CalcNumberLeapYearsFromEpoch(int year) const
{
    return (year-1)/YearsBetweenLeapYears
                                   - (EpochYear-1)/YearsBetweenLeapYears;
}
int Date::GetDayOfYear(int day, int month, int year) const
{
    return daysBeforeMonth[IsLeapYear(year)][month-1] + day;
}
bool Date::IsLeapYear(int year) const
{
    return (year%4)==0;   // Not full story, but good enough to 2100
}
date_t Date::ToDateT() const
{
    date_t value = GetDayOfYear(m_day, m_month, m_year) - 1;
    value += (m_year-EpochYear) * DaysPerCommonYear;
    date_t numberLeapYears = CalcNumberLeapYearsFromEpoch(m_year);
    value += numberLeapYears;
    return value;
}
}
```

1.  现在`ToDateT()`正在工作，我们转向它的反向，即`FromDateT()`。同样，我们逐个构建测试，以开发一系列日期的算法。使用了以下测试：

```cpp
TEST_F(DateTest, FromDateT0Is1Jan1970)
{
    Date dt;
    dt.FromDateT(0);
    ASSERT_EQ(0, dt.ToDateT());
    VerifyDate(dt, 1970, 1, 1);
}
TEST_F(DateTest, FromDateT1Is2Jan1970)
{
    Date dt;
    dt.FromDateT(1);
    ASSERT_EQ(1, dt.ToDateT());
    VerifyDate(dt, 1970, 1, 2);
}
TEST_F(DateTest, FromDateT364Is31Dec1970)
{
    Date dt;
    dt.FromDateT(364);
    ASSERT_EQ(364, dt.ToDateT());
    VerifyDate(dt, 1970, 12, 31);
}
TEST_F(DateTest, FromDateT365Is1Jan1971)
{
    Date dt;
    dt.FromDateT(365);
    ASSERT_EQ(365, dt.ToDateT());
    VerifyDate(dt, 1971, 1, 1);
}
TEST_F(DateTest, FromDateT1096Is1Jan1973)
{
    Date dt;
    dt.FromDateT(1096);
    ASSERT_EQ(1096, dt.ToDateT());
    VerifyDate(dt, 1973, 1, 1);
}
TEST_F(DateTest, FromDateT18136Is28Aug2019)
{
    Date dt;
    dt.FromDateT(18136);
    ASSERT_EQ(18136, dt.ToDateT());
    VerifyDate(dt, 2019, 8, 28);
}
```

1.  在头文件中添加以下声明：

```cpp
public:
    void FromDateT(date_t date);
private:
    int CalcMonthDayOfYearIsIn(int dayOfYear, bool IsLeapYear) const;
```

1.  使用以下实现，因为之前的测试是逐个添加的：

```cpp
void Date::FromDateT(date_t date)
{
    int number_years = date / DaysPerCommonYear;
    date = date - number_years * DaysPerCommonYear;
    m_year = EpochYear + number_years;
    date_t numberLeapYears = CalcNumberLeapYearsFromEpoch(m_year);
    date -= numberLeapYears;
    m_month = CalcMonthDayOfYearIsIn(date, IsLeapYear(m_year));
    date -= daysBeforeMonth[IsLeapYear(m_year)][m_month-1];
    m_day = date + 1;
}
int Date::CalcMonthDayOfYearIsIn(int dayOfYear, bool isLeapYear) const
{
    for(int i = 1 ; i < 12; i++)
    {
    if ( daysBeforeMonth[isLeapYear][i] > dayOfYear)
            return i;
    }
    return 12;
}
```

1.  现在我们已经准备好支持例程，我们可以实现`Date`类的真正特性，即两个日期之间的差异，并通过添加一定数量的天来确定新日期。这两个操作都需要一个新类型（类）`Days`。

1.  将以下`Days`的实现添加到头文件（在`Date`之前）：

```cpp
class Days
{
public:
    Days() = default;
    Days(int days) : m_days{days}     {    }
    operator int() const
    {
        return m_days;
    }
private:
    int m_days{0};
};
```

1.  第一个运算符将是将`Days`添加到`Date`的加法。添加以下方法声明（在`Date`类的公共部分内）：

```cpp
Date& operator+=(const Days& day);
```

1.  然后，在头文件中（在`Date`类之外）添加内联实现：

```cpp
inline Date operator+(const Date& lhs, const Days& rhs )
{
    Date tmp(lhs);
    tmp += rhs;
    return tmp;
}
```

1.  编写以下测试来验证`sum`操作：

```cpp
TEST_F(DateTest, AddZeroDays)
{
    Date dt(28, 8, 2019);
    Days days;
    dt += days;
    VerifyDate(dt, 2019, 8, 28);
}
TEST_F(DateTest, AddFourDays)
{
    Date dt(28, 8, 2019);
    Days days(4);
    dt += days;
    VerifyDate(dt, 2019, 9, 1);
}
```

1.  `sum`操作的实际实现仅基于两个支持方法

```cpp
Date& Date::operator+=(const Days& day)
{
    FromDateT(ToDateT()+day);
    return *this;
}
```

1.  添加以下测试：

```cpp
TEST_F(DateTest, AddFourDaysAsInt)
{
    Date dt(28, 8, 2019);
    dt += 4;
    VerifyDate(dt, 2019, 9, 1);
}
```

1.  当我们运行测试时，它们都构建了，并且这个测试通过了。但这不是期望的结果。我们不希望它们能够将裸整数添加到我们的日期中。（将来的版本可能会添加月份和年份，那么添加整数意味着什么？）。为了使其失败并导致构建失败，我们将 Days 构造函数更改为`explicit`：

```cpp
explicit Days(int days) : m_days{days}     {    }
```

1.  现在构建失败了，所以我们需要通过将添加行转换为`Days`来修复测试，如下所示：

```cpp
dt += static_cast<Days>(4);
```

所有测试应该再次通过。

1.  我们想要的最终功能是两个日期之间的差异。以下是用于验证实现的测试：

```cpp
TEST_F(DateTest, DateDifferences27days)
{
    Date dt1(28, 8, 2019);
    Date dt2(1, 8, 2019);
    Days days = dt1 - dt2;
    ASSERT_EQ(27, (int)days);
}
TEST_F(DateTest, DateDifferences365days)
{
    Date dt1(28, 8, 2019);
    Date dt2(28, 8, 2018);
    Days days = dt1 - dt2;
    ASSERT_EQ(365, (int)days);
}
```

1.  在头文件中的`Date`类的公共部分中添加以下函数声明：

```cpp
Days operator-(const Date& rhs) const;
```

1.  在头文件中的 Date 类之后添加以下代码：

```cpp
inline Days Date::operator-(const Date& rhs) const
{
    return Days(ToDateT() - rhs.ToDateT());
}
```

因为我们使`Days`构造函数显式，所以必须在返回语句中调用它。在所有这些更改都就位后，所有测试应该都通过。

1.  将`L3A2date`配置为`datetools`二进制文件，并在编辑器中打开 main.cpp。从`ACTIVITY2`的定义中删除注释：

```cpp
#define ACTIVITY2
```

1.  构建然后运行示例应用程序。这将产生以下输出：

![图 3.53：成功的 Date 示例应用程序的输出](img/C14583_03_53.jpg)

###### 图 3.53：成功的 Date 示例应用程序的输出

我们已经实现了 Date 和 Days 类的所有要求，并通过单元测试交付了它们。单元测试使我们能够实现增量功能，以构建两个复杂算法`ToDateT`和`FromDateT`，它们构成了我们想要交付的功能的基础支持。

## 第四章 - 关注点分离 - 软件架构，函数，可变模板

### 活动 1：实现多播事件处理程序

1.  从**Lesson4/Activity01**文件夹加载准备好的项目，并将项目的当前构建器配置为 CMake Build（Portable）。构建项目，配置启动器并运行单元测试（其中一个虚拟测试失败）。建议为测试运行器使用*L4delegateTests*。

1.  在**delegateTests.cpp**中，用以下测试替换失败的虚拟测试：

```cpp
TEST_F(DelegateTest, BasicDelegate)
{
    Delegate delegate;
    ASSERT_NO_THROW(delegate.Notify(42));
}
```

1.  现在构建失败了，所以我们需要向`Delegate`添加一个新方法。由于这将演变为一个模板，我们将在头文件中进行所有这些开发。在**delegate.hpp**中，添加以下定义：

```cpp
class Delegate
{
public:
    Delegate() = default;
    void Notify(int value) const
    {
    }
};
```

现在测试运行并通过。

1.  在现有测试中添加以下行：

```cpp
ASSERT_NO_THROW(delegate(22));
```

1.  再次构建失败，所以我们更新`Delegate`的定义如下（我们可以让`Notify`调用`operator()`，但这样更容易阅读）：

```cpp
void operator()(int value)
{
    Notify(value);
}
```

测试再次运行并通过。

1.  在添加下一个测试之前，我们将添加一些基础设施来帮助我们开发测试。处理程序最容易的方法是让它们写入`std::cout`，为了能够验证它们是否被调用，我们需要捕获输出。为此，通过更改`DelegateTest`类将标准输出流重定向到不同的缓冲区：

```cpp
class DelegateTest : public ::testing::Test
{
public:
    void SetUp() override;
    void TearDown() override;
    std::stringstream m_buffer;
    // Save cout's buffer here
    std::streambuf *m_savedBuf{};
};
void DelegateTest::SetUp()
{
    // Save the cout buffer
    m_savedBuf = std::cout.rdbuf();
    // Redirect cout to our buffer
    std::cout.rdbuf(m_buffer.rdbuf());
}
void DelegateTest::TearDown()
{
    // Restore cout buffer to original
    std::cout.rdbuf(m_savedBuf);
}
```

1.  还要在文件顶部添加`<iostream>`、`<sstream>`和`<string>`的包含语句。

1.  在支持框架的基础上，添加以下测试：

```cpp
TEST_F(DelegateTest, SingleCallback)
{
    Delegate delegate;
    delegate += [] (int value) { std::cout << "value = " << value; };
    delegate.Notify(42);
    std::string result = m_buffer.str();
    ASSERT_STREQ("value = 42", result.c_str());
}
```

1.  为了使测试再次构建和运行，添加以下代码到**delegate.h**类中：

```cpp
Delegate& operator+=(const std::function<void(int)>& delegate)
{
    m_delegate = delegate;
    return *this;
}
```

随着以下代码：

```cpp
private:
    std::function<void(int)> m_delegate;
```

现在测试构建了，但我们的新测试失败了。

1.  更新`Notify()`方法为：

```cpp
void Notify(int value) const
{
    m_delegate(value);
}
```

1.  现在测试构建并且我们的新测试通过了，但原始测试现在失败了。调用委托时抛出了异常，所以在调用之前我们需要检查委托是否为空。编写以下代码来实现这一点：

```cpp
void Notify(int value) const
{
    if(m_delegate)
        m_delegate(value);
}
```

所有测试现在都运行并通过。

1.  我们现在需要为`Delegate`类添加多播支持。添加新的测试：

```cpp
TEST_F(DelegateTest, DualCallbacks)
{
    Delegate delegate;
    delegate += [] (int value) { std::cout << "1: = " << value << "\n"; };
    delegate += [] (int value) { std::cout << "2: = " << value << "\n"; };
    delegate.Notify(12);
    std::string result = m_buffer.str();
    ASSERT_STREQ("1: = 12\n2: = 12\n", result.c_str());
}
```

1.  当然，这个测试现在失败了，因为`operator+=()`只分配给成员变量。我们需要添加一个列表来存储我们的委托。我们选择 vector，这样我们可以按照添加的顺序调用委托。在**delegate.hpp**的顶部添加`#include <vector>`，并更新 Delegate 将**m_delegate**替换为**m_delegates**回调的 vector：

```cpp
class Delegate
{
public:
    Delegate() = default;
    Delegate& operator+=(const std::function<void(int)>& delegate)
    {
        m_delegates.push_back(delegate);
        return *this;
    }
    void Notify(int value) const
    {
        for(auto& delegate : m_delegates)
        {
            delegate(value);
        }
    }
    void operator()(int value)
    {
        Notify(value);
    }
private:
    std::vector<std::function<void(int)>> m_delegates;
};
```

所有测试现在再次运行并通过。

1.  我们现在已经实现了基本的多播`delegate`类。现在我们需要将其转换为基于模板的类。通过在三个测试中将所有`Delegate`的声明更改为`Delegate<int>`来更新现有的测试。

1.  现在通过在类之前添加`template<class Arg>`来更新 Delegate 类，将其转换为模板，并将四个`int`的出现替换为`Arg`：

```cpp
template<class Arg>
class Delegate
{
public:
    Delegate() = default;
    Delegate& operator+=(const std::function<void(Arg)>& delegate)
    {
        m_delegates.push_back(delegate);
        return *this;
    }
    void Notify(Arg value) const
    {
        for(auto& delegate : m_delegates)
        {
            delegate(value);
        }
    }
    void operator()(Arg value)
    {
        Notify(value);
    }
private:
    std::vector<std::function<void(Arg)>> m_delegates;
};
```

1.  所有测试现在都运行并通过，因此它仍然适用于处理程序的`int`参数。

1.  添加以下测试并重新运行测试以确认模板转换是正确的：

```cpp
TEST_F(DelegateTest, DualCallbacksString)
{
    Delegate<std::string&> delegate;
    delegate += [] (std::string value) { std::cout << "1: = " << value << "\n"; };
    delegate += [] (std::string value) { std::cout << "2: = " << value << "\n"; };
    std::string hi{"hi"};
    delegate.Notify(hi);
    std::string result = m_buffer.str();
    ASSERT_STREQ("1: = hi\n2: = hi\n", result.c_str());
}
```

1.  现在它作为一个接受一个参数的模板运行。我们需要将其转换为接受零个或多个参数的可变模板。使用上一个主题的信息，将模板更新为以下内容：

```cpp
template<typename... ArgTypes>
class Delegate
{
public:
    Delegate() = default;
    Delegate& operator+=(const std::function<void(ArgTypes...)>& delegate)
    {
        m_delegates.push_back(delegate);
        return *this;
    }
    void Notify(ArgTypes&&... args) const
    {
        for(auto& delegate : m_delegates)
        {
            delegate(std::forward<ArgTypes>(args)...);
        }
    }
    void operator()(ArgTypes&&... args)
    {
        Notify(std::forward<ArgTypes>(args)...);
    }
private:
    std::vector<std::function<void(ArgTypes...)>> m_delegates;
};
```

测试应该仍然运行并通过。

1.  添加两个更多的测试 - 零参数测试和多参数测试：

```cpp
TEST_F(DelegateTest, DualCallbacksNoArgs)
{
    Delegate delegate;
    delegate += [] () { std::cout << "CB1\n"; };
    delegate += [] () { std::cout << "CB2\n"; };
    delegate.Notify();
    std::string result = m_buffer.str();
    ASSERT_STREQ("CB1\nCB2\n", result.c_str());
}
TEST_F(DelegateTest, DualCallbacksStringAndInt)
{
    Delegate<std::string&, int> delegate;
    delegate += [] (std::string& value, int i) {
            std::cout << "1: = " << value << "," << i << "\n"; };
    delegate += [] (std::string& value, int i) {
        std::cout << "2: = " << value << "," << i << "\n"; };
    std::string hi{"hi"};
    delegate.Notify(hi, 52);
    std::string result = m_buffer.str();
    ASSERT_STREQ("1: = hi,52\n2: = hi,52\n", result.c_str());
}
```

所有测试都运行并通过，显示我们现在已经实现了期望的`Delegate`类。

1.  现在，将运行配置更改为执行程序`L4delegate`。在编辑器中打开**main.cpp**文件，并更改文件顶部的定义为以下内容，然后运行程序：

```cpp
#define ACTIVITY_STEP 27
```

我们得到以下输出：

![图 4.35：委托成功实现的输出](img/C14583_04_35.jpg)

###### 图 4.35：委托成功实现的输出

在这个活动中，我们首先实现了一个提供基本单一委托功能的类，然后添加了多播功能。有了这个实现，并且有了单元测试，我们很快就能够转换为一个带有一个参数的模板，然后转换为一个可变模板版本。根据您正在开发的功能，特定实现过渡到一般形式，然后再到更一般形式的方法是正确的。可变模板的开发并不总是显而易见的。

## 第五章 - 哲学家的晚餐 - 线程和并发

### 活动 1：创建模拟器来模拟艺术画廊的工作

艺术画廊工作模拟器是一个模拟访客和看门人行为的应用程序。访客数量有限，即画廊内同时只能容纳 50 人。访客不断前来画廊。看门人检查是否超过了访客限制。如果是，它会要求新的访客等待并将他们放在等待列表上。如果没有，它允许他们进入画廊。访客可以随时离开画廊。如果有人离开画廊，看门人会让等待列表中的人进入画廊。

按照以下步骤执行此活动：

1.  创建一个文件，其中包含我们项目所需的所有常量 - `Common.hpp`。

1.  添加包含保护和第一个变量`CountPeopleInside`，它表示访客限制为 50 人：

```cpp
#ifndef COMMON_HPP
#define COMMON_HPP
constexpr size_t CountPeopleInside = 5;
#endif // COMMON_HPP
```

1.  现在，创建`Person`类的头文件和源文件，即`Person.hpp`和`Person.cpp`。还要添加包含保护。定义`Person`类并删除复制构造函数和复制赋值运算符；我们只会使用用户定义的默认构造函数、移动构造函数和移动赋值运算符以及默认析构函数。添加一个名为`m_Id`的私有变量；我们将用它来记录。还要添加一个名为`m_NextId`的私有静态变量；它将用于生成唯一的 ID：

```cpp
#ifndef PERSON_HPP
#define PERSON_HPP
class Person
{
public:
    Person();
    Person& operator=(Person&);
    Person(Person&&);
    ~Person() = default;
    Person(const Person&) = delete;
    Person& operator=(const Person&) = delete;
private:
    int m_Id;
    static int m_NextId;
};
#endif // PERSON_HPP
```

1.  在源文件中，定义我们的静态变量`m_NextId`。然后，在构造函数中，使用`m_NextId`的值初始化`m_Id`变量。在构造函数中打印日志。实现移动复制构造函数和移动赋值运算符。现在，为我们的`Person`对象实现线程安全存储。创建所需的头文件和源文件，即`Persons.hpp`和`Persons.cpp`。还要添加包含保护。包括"`Person.hpp`"和`<mutex>`和`<vector>`头文件。定义具有用户定义默认构造函数和默认析构函数的`Persons`类。声明`add()`函数以添加`Person`和`get()`以获取`Person`并从列表中删除它。定义`size()`函数以获取`Person`元素的计数，以及`removePerson()`，它从存储中删除任何人。在私有部分中，声明互斥类型的变量`m_Mutex`，即`m_Persons`来存储 Persons 的向量：

```cpp
#ifndef PERSONS_HPP
#define PERSONS_HPP
#include "Person.hpp"
#include <mutex>
#include <vector>
class Persons
{
public:
    Persons();
    ~Persons() = default;
    void add(Person&& person);
    Person get();
    size_t size() const;
    void removePerson();
private:
    std::mutex m_Mutex;
    std::vector<Person> m_Persons;
};
#endif // PERSONS_HPP
```

1.  在源文件中，声明用户定义的构造函数，我们将向量的大小保留为 50 个元素（以避免在增长过程中重新调整大小）：

```cpp
Persons::Persons()
{
    m_Persons.reserve(CountPeopleInside);
}
```

1.  声明`add()`函数，它接受`Person`类型的 rvalue 参数，锁定互斥锁，并使用`std::move()`函数将`Person`添加到向量中：

```cpp
void Persons::add(Person&& person)
{
    std::lock_guard<std::mutex> m_lock(m_Mutex);
    m_Persons.emplace_back(std::move(person));
}
```

1.  声明`get()`函数，锁定互斥锁并返回最后一个元素，然后从向量中删除它。如果向量为空，它将抛出异常：

```cpp
Person Persons::get()
{
    std::lock_guard<std::mutex> m_lock(m_Mutex);
    if (m_Persons.empty())
    {
        throw "Empty Persons storage";
    }
    Person result = std::move(m_Persons.back());
    m_Persons.pop_back();
    return result;
}
```

1.  声明`size()`函数，返回向量的大小：

```cpp
size_t Persons::size() const
{
    return m_Persons.size();
}
```

1.  最后，声明`removePerson()`函数，该函数锁定互斥锁并从向量中删除最后一个项目：

```cpp
void Persons::removePerson()
{
    std::lock_guard<std::mutex> m_lock(m_Mutex);
    m_Persons.pop_back();
    std::cout << "Persons | removePerson | removed" << std::endl;
}
```

1.  现在，实现`PersonGenerator`类，负责创建和删除`Person`项。创建相应的头文件和源文件，即`PersonGenerator.hpp`和`PersonGenerator.cpp`。还要添加包含保护。包括"`Person.hpp`"，`<thread>`和`<condition_variable>`头文件。定义`PersonGenerator`类。在私有部分中，定义两个`std::thread`变量，即`m_CreateThread`和`m_RemoveThread`。在一个线程中，我们将创建新的`Person`对象，并在另一个线程中异步通知用户删除`Person`对象。定义对`Persons`类型的共享变量的引用，即`m_CreatedPersons`。我们将把每个新人放在其中。`m_CreatedPersons`将在多个线程之间共享。定义两个`std::condition_variable`的引用，即`m_CondVarAddPerson`和`m_CondVarRemovePerson`。它们将用于线程之间的通信。定义两个`std::mutex`变量的引用，即`m_AddLock`和`m_RemoveLock`。它们将用于接收对条件变量的访问。最后，在私有部分中，定义两个函数，它们将是我们线程的启动函数 - `runCreating()`和`runRemoving()`。接下来，定义两个将触发条件变量的函数，即`notifyCreated()`和`notifyRemoved()`。在公共部分中，定义一个构造函数，它将所有在私有部分中定义的引用作为参数。最后，定义一个析构函数。这将确保其他默认生成的函数被删除：

```cpp
#ifndef PERSON_GENERATOR_HPP
#define PERSON_GENERATOR_HPP
#include "Persons.hpp"
#include <condition_variable>
#include <thread>
class PersonGenerator
{
public:
    PersonGenerator(Persons& persons,
            std::condition_variable& add_person,
            std::condition_variable& remove_person,
            std::mutex& add_lock,
            std::mutex& remove_lock,
            bool& addNotified,
            bool& removeNotified);
    ~PersonGenerator();
    PersonGenerator(const PersonGenerator&) = delete;
    PersonGenerator(PersonGenerator&&) = delete;
    PersonGenerator& operator=(const PersonGenerator&) = delete;
    PersonGenerator& operator=(PersonGenerator&&) = delete;
private:
    void runCreating();
    void runRemoving();
    void notifyCreated();
    void notifyRemoved();
private:
    std::thread m_CreateThread;
    std::thread m_RemoveThread;
    Persons& m_CreatedPersons;
    // to notify about creating new person
    std::condition_variable& m_CondVarAddPerson;
    std::mutex& m_AddLock;
    bool& m_AddNotified;
    // to notify that person needs to be removed
    std::condition_variable& m_CondVarRemovePerson;
    std::mutex& m_RemoveLock;
    bool& m_RemoveNotified;
};
#endif // PERSON_GENERATOR_HPP
```

1.  现在，转到源文件。包括`<stdlib.h>`文件，以便我们可以访问`rand()`和`srand()`函数，这些函数用于生成随机数。包括`<time.h>`头文件，以便我们可以访问`time()`函数，以及`std::chrono`命名空间。它们用于处理时间。包括`<ratio>`文件，用于 typedefs，以便我们可以使用时间库：

```cpp
#include "PersonGenerator.hpp"
#include <iostream>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time, chrono */
#include <ratio>        /* std::milli */
```

1.  声明构造函数并在初始化程序列表中初始化除线程之外的所有参数。在构造函数体中使用适当的函数初始化线程：

```cpp
PersonGenerator::PersonGenerator(Persons& persons,
                    std::condition_variable& add_person,
                    std::condition_variable& remove_person,
                    std::mutex& add_lock,
                    std::mutex& remove_lock,
                    bool& addNotified,
                    bool& removeNotified)
    : m_CreatedPersons(persons)
    , m_CondVarAddPerson(add_person)
    , m_AddLock(add_lock)
    , m_AddNotified(addNotified)
    , m_CondVarRemovePerson(remove_person)
    , m_RemoveLock(remove_lock)
    , m_RemoveNotified(removeNotified)
{
    m_CreateThread = std::thread(&PersonGenerator::runCreating, this);
    m_RemoveThread = std::thread(&PersonGenerator::runRemoving, this);
}
```

1.  声明一个析构函数，并检查线程是否可连接。如果不可连接，则加入它们：

```cpp
PersonGenerator::~PersonGenerator()
{
    if (m_CreateThread.joinable())
    {
        m_CreateThread.join();
    }
    if (m_RemoveThread.joinable())
    {
        m_RemoveThread.join();
    }
}
```

1.  声明`runCreating()`函数，这是`m_CreateThread`线程的启动函数。在这个函数中，我们将在一个无限循环中生成一个从 1 到 10 的随机数，并使当前线程休眠这段时间。之后，创建一个 Person 值，将其添加到共享容器，并通知其他线程：

```cpp
void PersonGenerator::runCreating()
{
    using namespace std::chrono_literals;
    srand (time(NULL));
    while(true)
    {
        std::chrono::duration<int, std::milli> duration((rand() % 10 + 1)*1000);
        std::this_thread::sleep_for(duration);
        std::cout << "PersonGenerator | runCreating | new person:" << std::endl;
        m_CreatedPersons.add(std::move(Person()));
        notifyCreated();
    }
}
```

1.  声明`runRemoving()`函数，这是`m_RemoveThread`线程的启动函数。在这个函数中，我们将在一个无限循环中生成一个从 20 到 30 的随机数，并使当前线程休眠这段时间。之后，通知其他线程应该移除一些访问者：

```cpp
void PersonGenerator::runRemoving()
{
    using namespace std::chrono_literals;
    srand (time(NULL));
    while(true)
    {
        std::chrono::duration<int, std::milli> duration((rand() % 10 + 20)*1000);
        std::this_thread::sleep_for(duration);
        std::cout << "PersonGenerator | runRemoving | somebody has left the gallery:" << std::endl;
        notifyRemoved();
    }
}
```

1.  声明`notifyCreated()`和`notifyRemoved()`函数。在它们的主体中，锁定适当的互斥锁，将适当的布尔变量设置为 true，并在适当的条件变量上调用`notify_all()`函数：

```cpp
void PersonGenerator::notifyCreated()
{
    std::unique_lock<std::mutex> lock(m_AddLock);
    m_AddNotified = true;
    m_CondVarAddPerson.notify_all();
}
void PersonGenerator::notifyRemoved()
{
    std::unique_lock<std::mutex> lock(m_RemoveLock);
    m_RemoveNotified = true;
    m_CondVarRemovePerson.notify_all();
}
```

1.  最后，我们需要为我们的最后一个类 Watchman 创建文件，即`Watchman.hpp`和`Watchman.cpp`。像往常一样，添加包含保护。包括"`Persons.hpp`"、`<thread>`、`<mutex>`和`<condition_variable>`头文件。定义`Watchman`类。在私有部分，定义两个`std::thread`变量，即`m_ThreadAdd`和`m_ThreadRemove`。在一个线程中，我们将新的`Person`对象移动到适当的队列中，并在另一个线程中异步移除`Person`对象。定义对共享`Persons`变量的引用，即`m_CreatedPeople`、`m_PeopleInside`和`m_PeopleInQueue`。如果限制未超出，我们将从`m_CreatedPeople`列表中获取每个新人，并将其移动到`m_PeopleInside`列表中。否则，我们将把它们移动到`m_PeopleInQueue`列表中。它们将在多个线程之间共享。定义两个对`std::condition_variable`的引用，即`m_CondVarAddPerson`和`m_CondVarRemovePerson`。它们将用于线程之间的通信。定义两个对`std::mutex`变量的引用，即`m_AddMux`和`m_RemoveMux`。它们将用于接收对条件变量的访问。最后，在私有部分中，定义两个函数，它们将成为我们线程的启动函数——`runAdd()`和`runRemove()`。在公共部分中，定义一个构造函数，它将所有在私有部分中定义的引用作为参数。现在，定义一个析构函数。确保删除所有其他默认生成的函数：

```cpp
#ifndef WATCHMAN_HPP
#define WATCHMAN_HPP
#include <mutex>
#include <thread>
#include <condition_variable>
#include "Persons.hpp"
class Watchman
{
public:
    Watchman(std::condition_variable&,
            std::condition_variable&,
            std::mutex&,
            std::mutex&,
            bool&,
            bool&,
            Persons&,
            Persons&,
            Persons&);
    ~Watchman();
    Watchman(const Watchman&) = delete;
    Watchman(Watchman&&) = delete;
    Watchman& operator=(const Watchman&) = delete;
    Watchman& operator=(Watchman&&) = delete;
private:
    void runAdd();
    void runRemove();
private:
    std::thread m_ThreadAdd;
    std::thread m_ThreadRemove;
    std::condition_variable& m_CondVarRemovePerson;
    std::condition_variable& m_CondVarAddPerson;
    std::mutex& m_AddMux;
    std::mutex& m_RemoveMux;
    bool& m_AddNotified;
    bool& m_RemoveNotified;
    Persons& m_PeopleInside;
    Persons& m_PeopleInQueue;
    Persons& m_CreatedPeople;
};
#endif // WATCHMAN_HPP
```

1.  现在，转到源文件。包括"`Common.hpp`"头文件，以便我们可以访问`m_CountPeopleInside`变量和其他必要的头文件：

```cpp
#include "Watchman.hpp"
#include "Common.hpp"
#include <iostream>
```

1.  声明构造函数，并在初始化列表中初始化除线程之外的所有参数。在构造函数的主体中使用适当的函数初始化线程：

```cpp
Watchman::Watchman(std::condition_variable& addPerson,
            std::condition_variable& removePerson,
            std::mutex& addMux,
            std::mutex& removeMux,
            bool& addNotified,
            bool& removeNotified,
            Persons& peopleInside,
            Persons& peopleInQueue,
            Persons& createdPeople)
    : m_CondVarRemovePerson(removePerson)
    , m_CondVarAddPerson(addPerson)
    , m_AddMux(addMux)
    , m_RemoveMux(removeMux)
    , m_AddNotified(addNotified)
    , m_RemoveNotified(removeNotified)
    , m_PeopleInside(peopleInside)
    , m_PeopleInQueue(peopleInQueue)
    , m_CreatedPeople(createdPeople)
{
    m_ThreadAdd = std::thread(&Watchman::runAdd, this);
    m_ThreadRemove = std::thread(&Watchman::runRemove, this);
}
```

1.  声明一个析构函数，并检查线程是否可连接。如果不可连接，则加入它们：

```cpp
Watchman::~Watchman()
{
    if (m_ThreadAdd.joinable())
    {
        m_ThreadAdd.join();
    }
    if (m_ThreadRemove.joinable())
    {
        m_ThreadRemove.join();
    }
}
```

1.  声明`runAdd()`函数。在这里，我们创建一个无限循环。在循环中，我们正在等待条件变量。当条件变量通知时，我们从`m_CreatedPeople`列表中取出人员，并将其移动到适当的列表，即`m_PeopleInside`，或者如果超出限制，则移动到`m_PeopleInQueue`。然后，我们检查`m_PeopleInQueue`列表中是否有人，以及`m_PeopleInside`是否已满，如果是，则将它们移动到这个列表中：

```cpp
void Watchman::runAdd()
{
    while (true)
    {
        std::unique_lock<std::mutex> locker(m_AddMux);
        while(!m_AddNotified)
        {
            std::cerr << "Watchman | runAdd | false awakening" << std::endl;
            m_CondVarAddPerson.wait(locker);
        }
        std::cout << "Watchman | runAdd | new person came" << std::endl;
        m_AddNotified = false;
        while (m_CreatedPeople.size() > 0)
        {
            try
            {
                auto person = m_CreatedPeople.get();
                if (m_PeopleInside.size() < CountPeopleInside)
                {
                    std::cout << "Watchman | runAdd | welcome in our The Art Gallery" << std::endl;
                    m_PeopleInside.add(std::move(person));
                }
                else
                {
                    std::cout << "Watchman | runAdd | Sorry, we are full. Please wait" << std::endl;
                    m_PeopleInQueue.add(std::move(person));
                }
            }
            catch(const std::string& e)
            {
                std::cout << e << std::endl;
            }
        }
        std::cout << "Watchman | runAdd | check people in queue" << std::endl;
        if (m_PeopleInQueue.size() > 0)
        {
            while (m_PeopleInside.size() < CountPeopleInside)
            {
                try
                {
                    auto person = m_PeopleInQueue.get();
                    std::cout << "Watchman | runAdd | welcome in our The Art Gallery" << std::endl;
                    m_PeopleInside.add(std::move(person));
                }
                catch(const std::string& e)
                {
                    std::cout << e << std::endl;
                }
            }
        }
    }
}
```

1.  接下来，声明`runRemove()`函数，我们将从`m_PeopleInside`中移除访问者。同样，在无限循环中，我们正在等待`m_CondVarRemovePerson`条件变量。当它通知线程时，我们从访问者列表中移除人员。接下来，我们将检查`m_PeopleInQueue`列表中是否有人，以及是否未超出限制，如果是，则将它们添加到`m_PeopleInside`中：

```cpp
void Watchman::runRemove()
{
    while (true)
    {
        std::unique_lock<std::mutex> locker(m_RemoveMux);
        while(!m_RemoveNotified)
        {
            std::cerr << "Watchman | runRemove | false awakening" << std::endl;
            m_CondVarRemovePerson.wait(locker);
        }
        m_RemoveNotified = false;
        if (m_PeopleInside.size() > 0)
        {
            m_PeopleInside.removePerson();
            std::cout << "Watchman | runRemove | good buy" << std::endl;
        }
        else
        {
            std::cout << "Watchman | runRemove | there is nobody in The Art Gallery" << std::endl;
        }
        std::cout << "Watchman | runRemove | check people in queue" << std::endl;
        if (m_PeopleInQueue.size() > 0)
        {
            while (m_PeopleInside.size() < CountPeopleInside)
            {
                try
                {
                    auto person = m_PeopleInQueue.get();
                    std::cout << "Watchman | runRemove | welcome in our The Art Gallery" << std::endl;
                    m_PeopleInside.add(std::move(person));
                }
                catch(const std::string& e)
                {
                    std::cout << e << std::endl;
                }
            }
        }
    }
}
```

1.  最后，转到`main()`函数。首先，创建我们在`Watchman`和`PersonGenerator`类中使用的所有共享变量。接下来，创建`Watchman`和`PersonGenerator`变量，并将这些共享变量传递给构造函数。在主函数的末尾，读取字符以避免关闭应用程序：

```cpp
int main()
{
    {
        std::condition_variable g_CondVarRemovePerson;
        std::condition_variable g_CondVarAddPerson;
        std::mutex g_AddMux;
        std::mutex g_RemoveMux;
        bool g_AddNotified = false;;
        bool g_RemoveNotified = false;
        Persons g_PeopleInside;
        Persons g_PeopleInQueue;
        Persons g_CreatedPersons;
        PersonGenerator generator(g_CreatedPersons, g_CondVarAddPerson, g_CondVarRemovePerson,
                        g_AddMux, g_RemoveMux, g_AddNotified, g_RemoveNotified);
        Watchman watchman(g_CondVarAddPerson,
                g_CondVarRemovePerson,
                g_AddMux,
                g_RemoveMux,
                g_AddNotified,
                g_RemoveNotified,
                g_PeopleInside,
                g_PeopleInQueue,
                g_CreatedPersons);
    }
    char a;
    std::cin >> a;
    return 0;
}
```

1.  编译并运行应用程序。在终端中，您将看到来自不同线程的日志，说明创建和移动人员从一个列表到另一个列表。您的输出将类似于以下屏幕截图：

![图 5.27：应用程序执行的结果](img/C14583_05_27.jpg)

###### 图 5.27：应用程序执行的结果

正如您所看到的，所有线程之间都以非常简单和清晰的方式进行通信。我们通过使用互斥锁来保护我们的共享数据，以避免竞争条件。在这里，我们使用异常来警告空列表，并在线程函数中捕获它们，以便我们的线程自行处理异常。我们还在析构函数中检查线程是否可连接之前加入它。这使我们能够避免程序意外终止。因此，这个小项目展示了我们在处理线程时的技能。

## 第六章-流和 I/O

### 活动 1 艺术画廊模拟器的记录系统

线程安全的记录器允许我们同时将数据输出到终端。我们通过从`std::ostringstream`类继承并使用互斥锁进行同步来实现此记录器。我们将实现一个提供格式化输出接口的类，我们的记录器将使用它来扩展基本输出。我们定义了不同日志级别的宏定义，以提供易于使用和清晰的接口。按照以下步骤完成此活动：

1.  从 Lesson6\中打开项目。

1.  在**src/**目录中创建一个名为 logger 的新目录。您将获得以下层次结构：![图 6.25：项目的层次结构](img/C14583_06_25.jpg)

###### 图 6.25：项目的层次结构

1.  创建名为`LoggerUtils`的头文件和源文件。在`LoggerUtils.hpp`中，添加包括保护。包括`<string>`头文件以添加对字符串的支持。定义一个名为 logger 的命名空间，然后定义一个嵌套命名空间叫做`utils`。在`utils`命名空间中，声明`LoggerUtils`类。

1.  在公共部分，声明以下静态函数：`getDateTime`、`getThreadId`、`getLoggingLevel`、`getFileAndLine`、`getFuncName`、`getInFuncName`和`getOutFuncName`。您的类应如下所示：

```cpp
#ifndef LOGGERUTILS_HPP_
#define LOGGERUTILS_HPP_
#include <string>
namespace logger
{
namespace utils
{
class LoggerUtils
{
public:
     static std::string getDateTime();
     static std::string getThreadId();
     static std::string getLoggingLevel(const std::string& level);
     static std::string getFileAndLine(const std::string& file, const int& line);
     static std::string getFuncName(const std::string& func);
     static std::string getInFuncName(const std::string& func);
     static std::string getOutFuncName(const std::string& func);
};
} // namespace utils
} // namespace logger
#endif /* LOGGERUTILS_HPP_ */
```

1.  在`LoggerUtils.cpp`中，添加所需的包括："`LoggerUtils.hpp`"头文件，`<sstream>`用于`std::stringstream`支持，`<ctime>`用于日期和时间支持：

```cpp
#include "LoggerUtils.hpp"
#include <sstream>
#include <ctime>
#include <thread>
```

1.  进入`logger`和`utils`命名空间。编写所需的函数定义。在`getDateTime()`函数中，使用`localtime()`函数获取本地时间。使用`strftime()`函数将其格式化为字符串。使用`std::stringstream`将其转换为所需格式：

```cpp
std::string LoggerUtils::getDateTime()
{
     time_t rawtime;
     struct tm * timeinfo;
     char buffer[80];
     time (&rawtime);
     timeinfo = localtime(&rawtime);
     strftime(buffer,sizeof(buffer),"%d-%m-%YT%H:%M:%S",timeinfo);
     std::stringstream ss;
     ss << "[";
     ss << buffer;
     ss << "]";
     return ss.str();
}
```

1.  在`getThreadId()`函数中，获取当前线程 ID 并使用`std::stringstream`将其转换为所需格式：

```cpp
std::string LoggerUtils::getThreadId()
{
     std::stringstream ss;
     ss << "[";
     ss << std::this_thread::get_id();
     ss << "]";
     return ss.str();
}
```

1.  在`getLoggingLevel()`函数中，使用`std::stringstream`将给定的字符串转换为所需格式：

```cpp
std::string LoggerUtils::getLoggingLevel(const std::string& level)
{
     std::stringstream ss;
     ss << "[";
     ss << level;
     ss << "]";
     return ss.str();
}
```

1.  在`getFileAndLine()`函数中，使用`std::stringstream`将给定的文件和行转换为所需格式：

```cpp
std::string LoggerUtils::getFileAndLine(const std::string& file, const int& line)
{
     std::stringstream ss;
     ss << " ";
     ss << file;
     ss << ":";
     ss << line;
     ss << ":";
     return ss.str();
}
```

1.  在`getFuncName()`函数中，使用`std::stringstream`将函数名转换为所需格式：

```cpp
std::string LoggerUtils::getFuncName(const std::string& func)
{
     std::stringstream ss;
     ss << " --- ";
     ss << func;
     ss << "()";
     return ss.str();
}
```

1.  在`getInFuncName()`函数中，使用`std::stringstream`将函数名转换为所需格式。

```cpp
std::string LoggerUtils::getInFuncName(const std::string& func)
{
     std::stringstream ss;
     ss << " --> ";
     ss << func;
     ss << "()";
     return ss.str();
}
```

1.  在`getOutFuncName()`函数中，使用`std::stringstream`将函数名转换为所需格式：

```cpp
std::string LoggerUtils::getOutFuncName(const std::string& func)
{
     std::stringstream ss;
     ss << " <-- ";
     ss << func;
     ss << "()";
     return ss.str();
}
```

1.  创建一个名为`LoggerMacroses.hpp`的头文件。添加包含保护。为每个`LoggerUtils`函数创建宏定义：`DATETIME`用于`getDateTime()`函数，`THREAD_ID`用于`getThreadId()`函数，`LOG_LEVEL`用于`getLoggingLevel()`函数，`FILE_LINE`用于`getFileAndLine()`函数，`FUNC_NAME`用于`getFuncName()`函数，`FUNC_ENTRY_NAME`用于`getInFuncName()`函数，`FUNC_EXIT_NAME`用于`getOutFuncName()`函数。结果，头文件应如下所示：

```cpp
#ifndef LOGGERMACROSES_HPP_
#define LOGGERMACROSES_HPP_
#define DATETIME \
     logger::utils::LoggerUtils::getDateTime()
#define THREAD_ID \
     logger::utils::LoggerUtils::getThreadId()
#define LOG_LEVEL( level ) \
     logger::utils::LoggerUtils::getLoggingLevel(level)
#define FILE_LINE \
     logger::utils::LoggerUtils::getFileAndLine(__FILE__, __LINE__)
#define FUNC_NAME \
     logger::utils::LoggerUtils::getFuncName(__FUNCTION__)
#define FUNC_ENTRY_NAME \
     logger::utils::LoggerUtils::getInFuncName(__FUNCTION__)
#define FUNC_EXIT_NAME \
     logger::utils::LoggerUtils::getOutFuncName(__FUNCTION__)
#endif /* LOGGERMACROSES_HPP_ */
```

1.  创建一个名为`StreamLogger`的头文件和源文件。在`StreamLogger.hpp`中，添加所需的包含保护。包含`LoggerMacroses.hpp`和`LoggerUtils.hpp`头文件。然后，包含`<sstream>`头文件以支持`std::ostringstream`，包含`<thread>`头文件以支持`std::thread`，以及包含`<mutex>`头文件以支持`std::mutex`：

```cpp
#include "LoggerMacroses.hpp"
#include "LoggerUtils.hpp"
#include <sstream>
#include <thread>
#include <mutex>
```

1.  进入`namespace` logger。声明`StreamLogger`类，它继承自`std::ostringstream`类。这种继承允许我们使用重载的左移操作符<<进行记录。我们不设置输出设备，因此输出不会执行 - 只是存储在内部缓冲区中。在私有部分，声明一个名为`m_mux`的静态`std::mutex`变量。声明常量字符串，以便存储日志级别、文件和行以及函数名。在公共部分，声明一个以日志级别、文件和行以及函数名为参数的构造函数。声明一个类析构函数。类声明应如下所示：

```cpp
namespace logger
{
class StreamLogger : public std::ostringstream
{
public:
     StreamLogger(const std::string logLevel,
                  const std::string fileLine,
                  const std::string funcName);
     ~StreamLogger();
private:
     static std::mutex m_mux;
     const std::string m_logLevel;
     const std::string m_fileLine;
     const std::string m_funcName;
};
} // namespace logger
```

1.  在`StreamLogger.cpp`中，包含`StreamLogger.hpp`和`<iostream>`头文件以支持`std::cout`。进入`logger`命名空间。定义构造函数并在初始化列表中初始化所有成员。然后，定义析构函数并进入其作用域。锁定`m_mux`互斥体。如果内部缓冲区为空，则仅输出日期和时间、线程 ID、日志级别、文件和行以及函数名。结果，我们将得到以下格式的行：`[dateTtime][threadId][logLevel][file:line: ][name() --- ]`。如果内部缓冲区包含任何数据，则在末尾输出相同的字符串与缓冲区。结果，我们将得到以下格式的行：`[dateTtime][threadId][logLevel][file:line: ][name() --- ] | message`。完整的源文件应如下所示：

```cpp
#include "StreamLogger.hpp"
#include <iostream>
std::mutex logger::StreamLogger::m_mux;
namespace logger
{
StreamLogger::StreamLogger(const std::string logLevel,
                  const std::string fileLine,
                  const std::string funcName)
          : m_logLevel(logLevel)
          , m_fileLine(fileLine)
          , m_funcName(funcName)
{}
StreamLogger::~StreamLogger()
{
     std::lock_guard<std::mutex> lock(m_mux);
     if (this->str().empty())
     {
          std::cout << DATETIME << THREAD_ID << m_logLevel << m_fileLine << m_funcName << std::endl;
     }
     else
     {
          std::cout << DATETIME << THREAD_ID << m_logLevel << m_fileLine << m_funcName << " | " << this->str() << std::endl;
     }
}
}
```

1.  创建一个名为`Logger.hpp`的头文件并添加所需的包含保护。包含`StreamLogger.hpp`和`LoggerMacroses.hpp`头文件。接下来，为不同的日志级别创建宏定义：`LOG_TRACE()`、`LOG_DEBUG()`、`LOG_WARN()`、`LOG_TRACE()`、`LOG_INFO()`、`LOG_ERROR()`、`LOG_TRACE_ENTRY()`和`LOG_TRACE_EXIT()`。完整的头文件应如下所示：

```cpp
#ifndef LOGGER_HPP_
#define LOGGER_HPP_
#include "StreamLogger.hpp"
#include "LoggerMacroses.hpp"
#define LOG_TRACE() logger::StreamLogger{LOG_LEVEL("Trace"), FILE_LINE, FUNC_NAME}
#define LOG_DEBUG() logger::StreamLogger{LOG_LEVEL("Debug"), FILE_LINE, FUNC_NAME}
#define LOG_WARN() logger::StreamLogger{LOG_LEVEL("Warning"), FILE_LINE, FUNC_NAME}
#define LOG_TRACE() logger::StreamLogger{LOG_LEVEL("Trace"), FILE_LINE, FUNC_NAME}
#define LOG_INFO() logger::StreamLogger{LOG_LEVEL("Info"), FILE_LINE, FUNC_NAME}
#define LOG_ERROR() logger::StreamLogger{LOG_LEVEL("Error"), FILE_LINE, FUNC_NAME}
#define LOG_TRACE_ENTRY() logger::StreamLogger{LOG_LEVEL("Error"), FILE_LINE, FUNC_ENTRY_NAME}
#define LOG_TRACE_EXIT() logger::StreamLogger{LOG_LEVEL("Error"), FILE_LINE, FUNC_EXIT_NAME}
#endif /* LOGGER_HPP_ */
```

1.  用适当的宏定义调用替换所有`std::cout`调用。在`Watchman.cpp`源文件中包含`logger/Logger.hpp`头文件。在`runAdd()`函数中，用不同日志级别的宏定义替换所有`std::cout`的实例。`runAdd()`函数应如下所示：

```cpp
void Watchman::runAdd()
{
     while (true)
     {
          std::unique_lock<std::mutex> locker(m_AddMux);
          while(!m_AddNotified)
          {
               LOG_DEBUG() << "Spurious awakening";
               m_CondVarAddPerson.wait(locker);
          }
          LOG_INFO() << "New person came";
          m_AddNotified = false;
          while (m_CreatedPeople.size() > 0)
          {
               try
               {
                    auto person = m_CreatedPeople.get();
                    if (m_PeopleInside.size() < CountPeopleInside)
                    {
                         LOG_INFO() << "Welcome in the our Art Gallery";
                         m_PeopleInside.add(std::move(person));
                    }
                    else
                    {
                         LOG_INFO() << "Sorry, we are full. Please wait";
                         m_PeopleInQueue.add(std::move(person));
                    }
               }
               catch(const std::string& e)
               {
                    LOG_ERROR() << e;
               }
          }
          LOG_TRACE() << "Check people in queue";
          if (m_PeopleInQueue.size() > 0)
          {
               while (m_PeopleInside.size() < CountPeopleInside)
               {
                    try
                    {
                         auto person = m_PeopleInQueue.get();
                         LOG_INFO() << "Welcome in the our Art Gallery";
                         m_PeopleInside.add(std::move(person));
                    }
                    catch(const std::string& e)
                    {
                         LOG_ERROR() << e;
                    }
               }
          }
     }
}
```

1.  注意我们如何使用我们的新记录器。我们用括号调用宏定义，并使用左移操作符：

```cpp
LOG_ERROR() << e;
Or
LOG_INFO() << "Welcome in the our Art Gallery";
```

1.  对代码的其余部分进行相同的替换。

1.  构建并运行应用程序。在终端中，您将看到来自不同线程的不同日志级别的日志消息，并带有有用的信息。一段时间后，您将获得类似以下的输出：

![图 6.26：活动项目的执行结果](img/C14583_06_26.jpg)

###### 图 6.26：活动项目的执行结果

如您所见，阅读和理解日志非常容易。如果需要，您可以轻松地更改`StreamLogger`类以将日志写入文件系统中的文件。您可以添加任何其他您可能需要用于调试应用程序的信息，例如输出函数参数。您还可以轻松地重写自定义类型的左移操作符以输出调试信息。

在这个项目中，我们运用了本章学到的许多东西。我们为线程安全输出创建了一个额外的流，将输出格式化为所需的表示形式，使用`std::stringstream`来格式化数据，并使用宏定义方便地记录器使用。因此，这个项目展示了我们在处理并发 I/O 方面的技能。

## 第七章 - 每个人都会跌倒，重要的是如何重新站起来 - 测试和调试

### 活动 1：使用测试用例检查函数的准确性并理解测试驱动开发（TDD）

在这个活动中，我们将开发函数来解析**RecordFile.txt**和**CurrencyConversion.txt**文件，并编写测试用例来检查函数的准确性。按照以下步骤实施此活动：

1.  创建一个名为**parse.conf**的配置文件并编写配置。

1.  请注意，这里只有两个变量是感兴趣的，即`currencyFile`和`recordFile`。其余的是为其他环境变量准备的：

```cpp
CONFIGURATION_FILE
currencyFile = ./CurrencyConversion.txt
recordFile = ./RecordFile.txt
DatabaseServer = 192.123.41.112
UserId = sqluser
Password = sqluser 
RestApiServer = 101.21.231.11
LogFilePath = /var/project/logs
```

1.  创建一个名为`CommonHeader.h`的头文件，并声明所有实用函数，即`isAllNumbers()`，`isDigit()`，`parseLine()`，`checkFile()`，`parseConfig()`，`parseCurrencyParameters()`，`fillCurrencyMap()`，`parseRecordFile()`，`checkRecord()`，`displayCurrencyMap()`和`displayRecords()`。

```cpp
#ifndef __COMMON_HEADER__H
#define __COMMON_HEADER__H
#include<iostream>
#include<cstring>
#include<fstream>
#include<vector>
#include<string>
#include<map>
#include<sstream>
#include<iterator>
#include<algorithm>
#include<iomanip>
using namespace std;
// Forward declaration of global variables. 
extern string configFile;
extern string recordFile;
extern string currencyFile;
extern map<string, float> currencyMap;
struct record;
extern vector<record>      vecRecord;
//Structure to hold Record Data . 
struct record{
    int     customerId;
    string  firstName;
    string  lastName;
    int     orderId;
    int     productId;
    int     quantity;
    float   totalPriceRegional;
    string  currency;
    float   totalPriceUsd;

    record(vector<string> & in){
        customerId      = atoi(in[0].c_str());
        firstName       = in[1];
        lastName        = in[2];
        orderId         = atoi(in[3].c_str());
        productId       = atoi(in[4].c_str());
        quantity        = atoi(in[5].c_str());
        totalPriceRegional = static_cast<float>(atof(in[6].c_str()));
        currency        = in[7];
        totalPriceUsd   = static_cast<float>(atof(in[8].c_str()));
    }
};
// Declaration of Utility Functions.. 
string trim (string &);
bool isAllNumbers(const string &);
bool isDigit(const string &);
void parseLine(ifstream &, vector<string> &, char);
bool checkFile(ifstream &, string &, string, char, string &);
bool parseConfig();
bool parseCurrencyParameters( vector<string> &);
bool fillCurrencyMap();
bool parseRecordFile();
bool checkRecord(vector<string> &);
void displayCurrencyMap();
ostream& operator<<(ostream &, const record &);
void displayRecords();
#endif
```

1.  创建一个名为`trim()`函数的文件：

```cpp
#include<CommonHeader.h>
// Utility function to remove spaces and tabs from start of string and end of string.. 
string trim (string &str) { // remove space and tab from string.
    string res("");
    if ((str.find(' ') != string::npos) || (str.find(' ') != string::npos)){ // if space or tab found.. 
        size_t begin, end;
        if ((begin = str.find_first_not_of(" \t")) != string::npos){ // if string is not empty.. 
            end = str.find_last_not_of(" \t");
            if ( end >= begin )
                res = str.substr(begin, end - begin + 1);
        }
    }else{
        res = str; // No space or tab found.. 
    }
    str = res;
    return res;
}
```

1.  将以下代码写入以定义`isAllNumbers()`，`isDigit()`和`parseLine()`函数：

```cpp
// Utility function to check if string contains only digits ( 0-9) and only single '.' 
// eg . 1121.23 , .113, 121\. are valid, but 231.14.143 is not valid.
bool isAllNumbers(const string &str){ // make sure, it only contains digit and only single '.' if any 
    return ( all_of(str.begin(), str.end(), [](char c) { return ( isdigit(c) || (c == '.')); }) 
             && (count(str.begin(), str.end(), '.') <= 1) );
}
//Utility function to check if string contains only digits (0-9).. 
bool isDigit(const string &str){
    return ( all_of(str.begin(), str.end(), [](char c) { return isdigit(c); }));
}
// Utility function, where single line of file <infile> is parsed using delimiter. 
// And store the tokens in vector of string. 
void parseLine(ifstream &infile, vector<string> & vec, char delimiter){
    string line, token;
    getline(infile, line);
    istringstream ss(line);
    vec.clear();
    while(getline(ss, token, delimiter)) // break line using delimiter
        vec.push_back(token);  // store tokens in vector of string
}
```

1.  将以下代码写入以定义`parseCurrencyParameters()`和`checkRecord()`函数：

```cpp
// Utility function to check if vector string of 2 strings contain correct 
// currency and conversion ratio. currency should be 3 characters, conversion ratio
// should be in decimal number format. 
bool parseCurrencyParameters( vector<string> & vec){
    trim(vec[0]);  trim(vec[1]);
    return ( (!vec[0].empty()) && (vec[0].size() == 3) && (!vec[1].empty()) && (isAllNumbers(vec[1])) );
}
// Utility function, to check if vector of string has correct format for records parsed from Record File. 
// CustomerId, OrderId, ProductId, Quantity should be in integer format
// TotalPrice Regional and USD should be in decimal number format
// Currecny should be present in map. 
bool checkRecord(vector<string> &split){
    // Trim all string in vector
    for (auto &s : split)
        trim(s);

    if ( !(isDigit(split[0]) && isDigit(split[3]) && isDigit(split[4]) && isDigit(split[5])) ){
        cerr << "ERROR: Record with customer id:" << split[0] << " doesnt have right DIGIT parameter" << endl;
        return false;
    }
    if ( !(isAllNumbers(split[6]) && isAllNumbers(split[8])) ){
        cerr << "ERROR: Record with customer id:" << split[0] << " doesnt have right NUMBER parameter" << endl;
        return false;
    }
    if ( currencyMap.find(split[7]) == currencyMap.end() ){
        cerr << "ERROR: Record with customer id :" << split[0] << " has currency :" << split[7] << " not present in map" << endl;
        return false;
    }
    return true;
}
```

1.  将以下代码写入以定义`checkFile()`函数：

```cpp
// Function to test initial conditions of file.. 
// Check if file is present and has correct header information. 
bool checkFile(ifstream &inFile, string &fileName, string parameter, char delimiter, string &error){
    bool flag = true;
    inFile.open(fileName);
    if ( inFile.fail() ){
        error = "Failed opening " + fileName + " file, with error: " + strerror(errno);
        flag = false;
    }
    if (flag){
        vector<string> split;
        // Parse first line as header and make sure it contains parameter as first token. 
        parseLine(inFile, split, delimiter);
        if (split.empty()){
            error = fileName + " is empty";
            flag = false;
        } else if ( split[0].find(parameter) == string::npos ){
            error = "In " + fileName + " file, first line doesnt contain header ";
            flag = false;
        }
    }
    return flag;
}
```

1.  将以下代码写入以定义`parseConfig()`函数：

```cpp
// Function to parse Config file. Each line will have '<name> = <value> format
// Store CurrencyConversion file and Record File parameters correctly. 
bool parseConfig() {
    ifstream coffle;
    string error;
    if (!checkFile(confFile, configFile, "CONFIGURATION_FILE", '=', error)){
        cerr << "ERROR: " << error << endl;
        return false;
    }
    bool flag = true;
    vector<string> split;
    while (confFile.good()){
        parseLine(confFile, split, '=');
        if ( split.size() == 2 ){ 
            string name = trim(split[0]);
            string value = trim(split[1]);
            if ( name == "currencyFile" )
                currencyFile = value;
            else if ( name == "recordFile")
                recordFile = value;
        }
    }
    if ( currencyFile.empty() || recordFile.empty() ){
        cerr << "ERROR : currencyfile or recordfile not set correctly." << endl;
        flag = false;
    }
    return flag;
}
```

1.  将以下代码写入以定义`fillCurrencyMap()`函数：

```cpp
// Function to parse CurrencyConversion file and store values in Map.
bool fillCurrencyMap() {
    ifstream currFile;
    string error;
    if (!checkFile(currFile, currencyFile, "Currency", '|', error)){
        cerr << "ERROR: " << error << endl;
        return false;
    }
    bool flag = true;
    vector<string> split;
    while (currFile.good()){
        parseLine(currFile, split, '|');
        if (split.size() == 2){
            if (parseCurrencyParameters(split)){
                currencyMap[split[0]] = static_cast<float>(atof(split[1].c_str())); // make sure currency is valid.
            } else {
                cerr << "ERROR: Processing Currency Conversion file for Currency: "<< split[0] << endl;
                flag = false;
                break;
            }
        } else if (!split.empty()){
            cerr << "ERROR: Processing Currency Conversion , got incorrect parameters for Currency: " << split[0] << endl;
            flag = false;
            break;
        }
    }
    return flag;
}
```

1.  将以下代码写入以定义`parseRecordFile()`函数：

```cpp
// Function to parse Record File .. 
bool parseRecordFile(){
    ifstream recFile;
    string error;
    if (!checkFile(recFile, recordFile, "Customer Id", '|', error)){
        cerr << "ERROR: " << error << endl;
        return false;
    }
    bool flag = true;
    vector<string> split;
    while(recFile.good()){
        parseLine(recFile, split, '|');
        if (split.size() == 9){ 
            if (checkRecord(split)){
                vecRecord.push_back(split); //Construct struct record and save it in vector... 
            }else{
                cerr << "ERROR : Parsing Record, for Customer Id: " << split[0] << endl;
                flag = false;
                break;
            }
        } else if (!split.empty()){
            cerr << "ERROR: Processing Record, for Customer Id: " << split[0] << endl;
            flag = false;
            break;
        }
    }
    return flag;
}
```

1.  将以下代码写入以定义`displayCurrencyMap()`函数：

```cpp
void displayCurrencyMap(){

    cout << "Currency MAP :" << endl;
    for (auto p : currencyMap)
        cout << p.first <<"  :  " << p.second << endl;
    cout << endl;
}
ostream& operator<<(ostream& os, const record &rec){
    os << rec.customerId <<"|" << rec.firstName << "|" << rec.lastName << "|" 
       << rec.orderId << "|" << rec.productId << "|" << rec.quantity << "|" 
       << fixed << setprecision(2) << rec.totalPriceRegional << "|" << rec.currency << "|" 
       << fixed << setprecision(2) << rec.totalPriceUsd << endl;
    return os;
}
```

1.  将以下代码写入以定义`displayRecords()`函数：

```cpp
void displayRecords(){
    cout << " Displaying records with '|' delimiter" << endl;
    for (auto rec : vecRecord){
        cout << rec;
    }
    cout << endl;
}
```

1.  创建名为`parseConfig()`，`fillCurrencyMap()`和`parseRecordFile()`函数的文件：

```cpp
#include <CommonHeader.h>
// Global variables ... 
string configFile = "./parse.conf";
string recordFile;
string currencyFile;
map<string, float>  currencyMap;
vector<record>      vecRecord;
int main(){
    // Read Config file to set global configuration variables. 
    if (!parseConfig()){
        cerr << "Error parsing Config File " << endl;
        return false;
    }
    // Read Currency file and fill map
    if (!fillCurrencyMap()){
        cerr << "Error setting CurrencyConversion Map " << endl;
        return false;
    }
    if (!parseRecordFile()){
        cerr << "Error parsing Records File " << endl;
        return false;
    }
        displayCurrencyMap();
    displayRecords();
    return 0;
}
```

1.  打开编译器。编译并执行已生成的`Util.o`和`ParseFiles`文件：![图 7.25：生成的新文件](img/C14583_07_25.jpg)

###### 图 7.25：生成的新文件

1.  运行`ParseFiles`可执行文件后，我们将收到以下输出：![图 7.26：生成的新文件](img/C14583_07_26.jpg)

###### 图 7.26：生成的新文件

1.  创建一个名为`trim`函数的文件：

```cpp
#include<gtest/gtest.h>
#include"../CommonHeader.h"
using namespace std;
// Global variables ... 
string configFile = "./parse.conf";
string recordFile;
string currencyFile;
map<string, float>  currencyMap;
vector<record>      vecRecord;
void setDefault(){
    configFile = "./parse.conf";
    recordFile.clear();
    currencyFile.clear();
    currencyMap.clear();
    vecRecord.clear();
}
// Test Cases for trim function ... 
TEST(trim, empty){
    string str="    ";
    EXPECT_EQ(trim(str), string());
}
TEST(trim, start_space){
    string str = "   adas";
    EXPECT_EQ(trim(str), string("adas"));
}
TEST(trim, end_space){
    string str = "trip      ";
    EXPECT_EQ(trim(str), string("trip"));
}
TEST(trim, string_middle){
    string str = "  hdgf   ";
    EXPECT_EQ(trim(str), string("hdgf"));
}
TEST(trim, single_char_start){
    string str = "c  ";
    EXPECT_EQ(trim(str), string("c"));
}
TEST(trim, single_char_end){
    string str = "   c";
    EXPECT_EQ(trim(str), string("c"));
}
TEST(trim, single_char_middle){
    string str = "      c  ";
    EXPECT_EQ(trim(str), string("c"));
}
```

1.  为`isAllNumbers`函数编写以下测试用例：

```cpp
// Test Cases for isAllNumbers function.. 
TEST(isNumber, alphabets_present){
    string str = "11.qwe13";
    ASSERT_FALSE(isAllNumbers(str));
}
TEST(isNumber, special_character_present){
    string str = "34.^%3";
    ASSERT_FALSE(isAllNumbers(str));
}
TEST(isNumber, correct_number){
    string str = "54.765";
    ASSERT_TRUE(isAllNumbers(str));
}
TEST(isNumber, decimal_begin){
    string str = ".624";
    ASSERT_TRUE(isAllNumbers(str));
}
TEST(isNumber, decimal_end){
    string str = "53.";
    ASSERT_TRUE(isAllNumbers(str));
}
```

1.  为`isDigit`函数编写以下测试用例：

```cpp
// Test Cases for isDigit funtion... 
TEST(isDigit, alphabet_present){
    string str = "527A";
    ASSERT_FALSE(isDigit(str));
}
TEST(isDigit, decimal_present){
    string str = "21.55";
    ASSERT_FALSE(isDigit(str));
}
TEST(isDigit, correct_digit){
    string str = "9769";
    ASSERT_TRUE(isDigit(str));
}
```

1.  为`parseCurrencyParameters`函数编写以下测试用例：

```cpp
// Test Cases for parseCurrencyParameters function
TEST(CurrencyParameters, extra_currency_chararcters){
    vector<string> vec {"ASAA","34.22"};
    ASSERT_FALSE(parseCurrencyParameters(vec));
}
TEST(CurrencyParameters, correct_parameters){
    vector<string> vec {"INR","1.44"};
    ASSERT_TRUE(parseCurrencyParameters(vec));
}
```

1.  为`checkFile`函数编写以下测试用例：

```cpp
//Test Cases for checkFile function...
TEST(checkFile, no_file_present){
    string fileName = "./NoFile";
    ifstream infile; 
    string parameter("nothing");
    char delimit =';';
    string err;
    ASSERT_FALSE(checkFile(infile, fileName, parameter, delimit, err));
}
TEST(checkFile, empty_file){
    string fileName = "./emptyFile";
    ifstream infile; 
    string parameter("nothing");
    char delimit =';';
    string err;
    ASSERT_FALSE(checkFile(infile, fileName, parameter, delimit, err));
}
TEST(checkFile, no_header){
    string fileName = "./noHeaderFile";
    ifstream infile; 
    string parameter("header");
    char delimit ='|';
    string err;
    ASSERT_FALSE(checkFile(infile, fileName, parameter, delimit, err));
}
TEST(checkFile, incorrect_header){
    string fileName = "./correctHeaderFile";
    ifstream infile; 
    string parameter("header");
    char delimit ='|';
    string err;
    ASSERT_FALSE(checkFile(infile, fileName, parameter, delimit, err));
}
TEST(checkFile, correct_file){
    string fileName = "./correctHeaderFile";
    ifstream infile; 
    string parameter("Currency");
    char delimit ='|';
    string err;
    ASSERT_TRUE(checkFile(infile, fileName, parameter, delimit, err));
}
```

#### 注意

在前述函数中用作输入参数的**NoFile**，**emptyFile**，**noHeaderFile**和**correctHeaderFile**文件可以在此处找到：[`github.com/TrainingByPackt/Advanced-CPlusPlus/tree/master/Lesson7/Activity01`](https://github.com/TrainingByPackt/Advanced-CPlusPlus/tree/master/Lesson7/Activity01)。

1.  为`parseConfig`函数编写以下测试用例：

```cpp
//Test Cases for parseConfig function...
TEST(parseConfig, missing_currency_file){
    setDefault();
    configFile = "./parseMissingCurrency.conf";
    ASSERT_FALSE(parseConfig());
}
TEST(parseConfig, missing_record_file){
    setDefault();
    configFile = "./parseMissingRecord.conf";
    ASSERT_FALSE(parseConfig());
}
TEST(parseConfig, correct_config_file){
    setDefault();
    configFile = "./parse.conf";
    ASSERT_TRUE(parseConfig());
}
```

#### 注意

在前述函数中用作输入参数的**parseMissingCurrency.conf**，**parseMissingRecord.conf**和**parse.conf**文件可以在此处找到：[`github.com/TrainingByPackt/Advanced-CPlusPlus/tree/master/Lesson7/Activity01`](https://github.com/TrainingByPackt/Advanced-CPlusPlus/tree/master/Lesson7/Activity01)。

1.  为`fillCurrencyMap`函数编写以下测试用例：

```cpp
//Test Cases for fillCurrencyMap function...
TEST(fillCurrencyMap, wrong_delimiter){
    currencyFile = "./CurrencyWrongDelimiter.txt";
    ASSERT_FALSE(fillCurrencyMap());
}
TEST(fillCurrencyMap, extra_column){
    currencyFile = "./CurrencyExtraColumn.txt";
    ASSERT_FALSE(fillCurrencyMap());
}
TEST(fillCurrencyMap, correct_file){
    currencyFile = "./CurrencyConversion.txt";
    ASSERT_TRUE(fillCurrencyMap());
}
```

#### 注意

在前面的函数中用作输入参数的**CurrencyWrongDelimiter.txt**、**CurrencyExtraColumn.txt**和**CurrencyConversion.txt**文件可以在此处找到：[`github.com/TrainingByPackt/Advanced-CPlusPlus/tree/master/Lesson7/Activity01`](https://github.com/TrainingByPackt/Advanced-CPlusPlus/tree/master/Lesson7/Activity01)。

1.  为 parseRecordFile 函数编写以下测试用例：

```cpp
//Test Cases for parseRecordFile function...
TEST(parseRecordFile, wrong_delimiter){
    recordFile = "./RecordWrongDelimiter.txt";
    ASSERT_FALSE(parseRecordFile());
}
TEST(parseRecordFile, extra_column){
    recordFile = "./RecordExtraColumn.txt";
    ASSERT_FALSE(parseRecordFile());
}
TEST(parseRecordFile, correct_file){
    recordFile = "./RecordFile.txt";
    ASSERT_TRUE(parseRecordFile());
}
```

在前面的函数中用作输入参数的**RecordWrongDelimiter.txt**、**RecordExtraColumn.txt**和**RecordFile.txt**文件可以在此处找到：[`github.com/TrainingByPackt/Advanced-CPlusPlus/tree/master/Lesson7/Activity01`](https://github.com/TrainingByPackt/Advanced-CPlusPlus/tree/master/Lesson7/Activity01)。

1.  打开编译器。通过编写以下命令编译和执行`Util.cpp`和`ParseFileTestCases.cpp`文件：

```cpp
g++ -c -g -Wall ../Util.cpp -I../
g++ -c -g -Wall ParseFileTestCases.cpp 
g++ -g -Wall Util.o ParseFileTestCases.o -lgtest -lgtest_main -pthread -o ParseFileTestCases
```

以下是此的截图。您将看到所有命令都存储在`Test.make`脚本文件中。一旦执行，它将创建用于单元测试的二进制程序`ParseFileTestCases`。您还会注意到在 Project 中创建了一个名为`unitTesting`的目录。在此目录中，编写了所有与单元测试相关的代码，并创建了一个二进制文件。此外，还通过编译`Util.cpp`文件来创建项目的依赖库`Util.o`：

![](img/C14583_07_27.jpg)

###### 图 7.27：执行脚本文件中的所有命令

1.  键入以下命令以运行所有测试用例：

```cpp
./ParseFileTestCases
```

屏幕上的输出将显示总共 31 个测试运行，其中包括 8 个测试套件。它还将显示各个测试套件的统计信息，以及通过/失败的结果：

![图 7.28：所有测试都正常运行](img/C14583_07_28.jpg)

###### 图 7.28：所有测试都正常运行

以下是下一个测试的截图：

![图 7.29：所有测试都正常运行](img/C14583_07_29.jpg)

###### 图 7.29：所有测试都正常运行

最后，我们通过解析两个文件并使用我们的测试用例来检查我们开发的函数的准确性。这将确保我们的项目在与具有测试用例的不同函数/模块集成时能够正常运行。

## 第八章 - 需要速度 - 性能和优化

### 活动 1：优化拼写检查算法

在这个活动中，我们将开发一个简单的拼写检查演示，并尝试逐步加快速度。您可以使用骨架文件**Speller.cpp**作为起点。执行以下步骤来实现此活动：

1.  拼写检查的第一个实现（完整代码可以在`getMisspelt()`函数中找到：

```cpp
set<string> setDict(vecDict.begin(), vecDict.end());
```

1.  循环遍历文本单词，并使用`set::count()`方法检查不在字典中的单词。将拼写错误的单词添加到结果向量中：

```cpp
vector<int> ret;
for(int i = 0; i < vecText.size(); ++i)
{
  const string &s = vecText[i];
  if(!setDict.count(s))
  {
    ret.push_back(i);
  }
};
```

1.  打开终端。编译程序并按以下方式运行：

```cpp
$ g++ -O3 Speller1.cpp Timer.cpp
$ ./a.out
```

将生成以下输出：

![图 8.60：第 1 步解决方案的示例输出](img/C14583_08_60.jpg)

###### 图 8.60：第 1 步解决方案的示例输出

1.  打开程序的`unordered_set`头文件：

```cpp
#include <unordered_set>
```

1.  接下来，将用于字典的集合类型更改为`unordered_set`：

```cpp
unordered_set<string> setDict(vecDict.begin(), vecDict.end());
```

1.  打开终端。编译程序并按以下方式运行：

```cpp
$ g++ -O3 Speller2.cpp Timer.cpp
$ ./a.out
```

将生成以下输出：

![图 8.61：第 2 步解决方案的示例输出](img/C14583_08_61.jpg)

###### 图 8.61：第 2 步解决方案的示例输出

1.  对于第三个也是最终版本，即`BKDR`函数。添加以下代码来实现这一点：

```cpp
const size_t SIZE = 16777215;
template<size_t SEED> size_t hasher(const string &s)
{
  size_t h = 0;
  size_t len = s.size();
  for(size_t i = 0; i < len; i++)
  {
    h = h * SEED + s[i];
  }
  return h & SIZE;
}
```

在这里，我们使用了整数模板参数，以便我们可以使用相同的代码创建任意数量的不同哈希函数。请注意使用`16777215`常量，它等于`2²⁴ - 1`。这使我们可以使用快速的按位与运算符，而不是模运算符，以使哈希整数小于`SIZE`。如果要更改大小，请将其保持为 2 的幂减一。

1.  接下来，让我们在`getMisspelt()`中声明一个用于布隆过滤器的`vector<bool>`，并用字典中的单词填充它。使用三个哈希函数。BKDR 哈希可以使用值如`131`、`3131`、`31313`等进行种子化。添加以下代码来实现这一点：

```cpp
vector<bool> m_Bloom;
m_Bloom.resize(SIZE);
for(auto i = vecDict.begin(); i != vecDict.end(); ++i)
{
  m_Bloom[hasher<131>(*i)] = true;
  m_Bloom[hasher<3131>(*i)] = true;
  m_Bloom[hasher<31313>(*i)] = true;
}
```

1.  编写以下代码创建一个检查单词的循环：

```cpp
for(int i = 0; i < vecText.size(); ++i)
{
  const string &s = vecText[i];
  bool hasNoBloom = 
          !m_Bloom[hasher<131>(s)] 
      &&  !m_Bloom[hasher<3131>(s)]
      &&  !m_Bloom[hasher<31313>(s)];

  if(hasNoBloom)
  {
    ret.push_back(i);
  }
  else if(!setDict.count(s))
  {
    ret.push_back(i);
  }
}
```

首先检查布隆过滤器，如果它在字典中找到了这个单词，我们必须像之前一样进行验证。

1.  打开终端。编译并运行程序如下：

```cpp
$ g++ -O3 Speller3.cpp Timer.cpp
$ ./a.out
```

将生成以下输出：

![图 8.62：第 3 步解决方案的示例输出](img/C14583_08_62.jpg)

###### 图 8.62：第 3 步解决方案的示例输出

在前面的活动中，我们试图解决一个现实世界的问题并使其更加高效。让我们考虑一下三个步骤中每个实现的一些要点，如下所示：

+   对于第一个版本，使用`std::set`的最明显的解决方案是-但是，性能可能会较低，因为集合数据结构是基于二叉树的，查找元素的复杂度为`O(log N)`。

+   对于第二个版本，我们可以通过切换到使用哈希表作为底层数据结构的`std::unordered_set`来获得很大的性能提升。如果哈希函数很好，性能将接近`O(1)`。

+   基于**布隆过滤器**数据结构的第三个版本需要一些考虑。-布隆过滤器的主要性能优势在于它是一种紧凑的数据结构，实际上并不存储其中的实际元素，因此提供了非常好的缓存性能。

从实现的角度来看，以下准则适用：

+   `vector<bool>`可以用作后备存储，因为这是一种高效存储和检索位的方式。

+   布隆过滤器的假阳性百分比应该很小-超过 5%将不高效。

+   有许多字符串哈希算法-参考实现中使用了**BKDR**哈希算法。可以在这里找到带有实现的字符串哈希算法的综合列表：[`www.partow.net/programming/hashfunctions/index.html`](http://www.partow.net/programming/hashfunctions/index.html)。

+   所使用的哈希函数数量和布隆过滤器的大小对于获得性能优势非常关键。

+   在决定布隆过滤器应该使用什么参数时，应考虑数据集的性质-请考虑，在这个例子中，拼写错误的单词很少，大部分都在字典中。

鉴于我们收到的结果，有一些值得探讨的问题：

+   为什么布隆过滤器的性能改进如此微弱？

+   使用更大或更小容量的布隆过滤器会有什么影响？

+   当使用更少或更多的哈希函数时会发生什么？

+   在什么条件下，这个版本比**Speller2.cpp**中的版本要快得多？

以下是这些问题的答案：

+   为什么布隆过滤器的性能改进如此微弱？

`std::unordered_set` 在达到存储的值之前执行一次哈希操作，可能还有几次内存访问。我们使用的布隆过滤器执行三次哈希操作和三次内存访问。因此，从本质上讲，布隆过滤器所做的工作比哈希表更多。由于我们的字典中只有 31,870 个单词，布隆过滤器的缓存优势就丧失了。这是另一个传统数据结构分析与现实结果不符的案例，因为缓存的原因。

+   使用更大或更小容量的布隆过滤器会有什么影响？

当使用更大的容量时，哈希冲突的数量减少，假阳性也减少，但缓存行为变差。相反，当使用较小的容量时，哈希冲突和假阳性增加，但缓存行为改善。

+   当使用更少或更多的哈希函数时会发生什么？

使用的哈希函数越多，误判就越少，反之亦然。

+   在什么条件下，这个版本比 Speller2.cpp 中的版本快得多？

布隆过滤器在测试少量位的成本低于访问哈希表中的值的成本时效果最好。只有当布隆过滤器的位完全适合缓存而字典不适合时，这一点才成立。
