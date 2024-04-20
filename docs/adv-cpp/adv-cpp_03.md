# 2B.不允许鸭子-模板和推导

## 学习目标

通过本章结束时，您将能够：

+   使用继承和多态将自己的类发挥到更大的效果

+   实现别名以使您的代码更易于阅读

+   使用 SFINAE 和 constexpr 开发模板以简化您的代码

+   使用 STL 实现自己的解决方案，以利用通用编程

+   描述类型推导的上下文和基本规则

本章将向您展示如何通过继承，多态和模板来定义和扩展您的类型。

## 介绍

在上一章中，我们学习了如何通过单元测试开发自己的类型（类），并使它们表现得像内置类型。我们介绍了函数重载，三/五法则和零法则。

在本章中，我们将学习如何进一步扩展类型系统。我们将学习如何使用模板创建函数和类，并重新讨论函数重载，因为它受到模板的影响。我们将介绍一种新技术**SFINAE**，并使用它来控制我们模板中包含在生成代码中的部分。

## 继承，多态和接口

在我们的面向对象设计和 C++的旅程中，我们已经专注于抽象和数据封装。现在我们将把注意力转向**继承**和**多态**。什么是继承？什么是多态？我们为什么需要它？考虑以下三个对象：

![图 2B.1：车辆对象](img/C14583_02B_01.jpg)

###### 图 2B.1：车辆对象

在上图中，我们可以看到有三个非常不同的对象。它们有一些共同之处。它们都有轮子（不同数量），发动机（不同大小，功率或配置），启动发动机，驾驶，刹车，停止发动机等，我们可以使用这些来做一些事情。

因此，我们可以将它们抽象成一个称为车辆的东西，展示这些属性和一般行为。如果我们将其表达为 C++类，可能会看起来像下面这样：

```cpp
class Vehicle
{
public:
  Vehicle() = default;
  Vehicle(int numberWheels, int engineSize) : 
          m_numberOfWheels{numberWheels}, m_engineSizeCC{engineSize}
  {
  }
  bool StartEngine()
  {
    std::cout << "Vehicle::StartEngine " << m_engineSizeCC << " CC\n";
    return true;
  };
  void Drive()
  {
    std::cout << "Vehicle::Drive\n";
  };
  void ApplyBrakes()
  {
    std::cout << "Vehicle::ApplyBrakes to " << m_numberOfWheels << " wheels\n";
  };
  bool StopEngine()
  {
    std::cout << "Vehicle::StopEngine\n";
    return true;
  };
private:
  int m_numberOfWheels {4};
  int m_engineSizeCC{1000};
};
```

`Vehicle`类是`Motorcycle`，`Car`和`Truck`的更一般（或抽象）表达。我们现在可以通过重用 Vehicle 类中已有的内容来创建更专业化的类型。我们将通过继承来重用 Vehicle 的属性和方法。继承的语法如下：

```cpp
class DerivedClassName : access_modifier BaseClassName
{
  // Body of DerivedClass
};
```

我们之前遇到过`public`，`protected`和`private`等访问修饰符。它们控制我们如何访问基类的成员。Motorcycle 类将派生如下：

```cpp
class Motorcycle : public Vehicle
{
public:
  Motorcycle(int engineSize) : Vehicle(2, engineSize) {};
};
```

在这种情况下，Vehicle 类被称为**基类**或**超类**，而 Motorcycle 类被称为**派生类**或**子类**。从图形上看，我们可以表示为下面的样子，箭头从派生类指向基类：

![图 2B.2：车辆类层次结构](img/C14583_02B_02.jpg)

###### 图 2B.2：车辆类层次结构

但摩托车的驾驶方式与通用车辆不同。因此，我们需要修改`Motorcycle`类，使其行为不同。更新后的代码将如下所示：

```cpp
class Motorcycle : public Vehicle
{
public:
  Motorcycle(int engineSize) : Vehicle(2, engineSize) {};
  void Drive()
  {
    std::cout << "Motorcycle::Drive\n";
  };
};
```

如果我们考虑面向对象设计，这是关于以对象协作的方式对问题空间进行建模。这些对象通过消息相互通信。现在，我们有两个类以不同的方式响应相同的消息（`Drive()`方法）。发送消息的人不知道会发生什么，也不真的在乎，这就是多态的本质。

#### 注意

多态来自希腊词 poly 和 morph，其中`poly`表示许多，`morph`表示形式。因此，多态意味着`具有多种形式`。

我们现在可以使用这些类来尝试多态：

```cpp
#include <iostream>
int main()
{
  Vehicle vehicle;
  Motorcycle cycle{1500};
  Vehicle* myVehicle{&vehicle};
  myVehicle->StartEngine();
  myVehicle->Drive();
  myVehicle->ApplyBrakes();
  myVehicle->StopEngine();
  myVehicle = &cycle;
  myVehicle->StartEngine();
  myVehicle->Drive();
  myVehicle->ApplyBrakes();
  myVehicle->StopEngine();
  return 0;
}
```

如果我们编译并运行此程序，我们会得到以下输出：

![图 2B.3：车辆程序输出](img/C14583_02B_03.jpg)

###### 图 2B.3：车辆程序输出

在前面的屏幕截图中，在`Vehicle::StartEngine 1500 cc`之后的行都与`Motorcycle`有关。但是`Drive`行仍然显示`Vehicle::Drive`，而不是预期的`Motorcycle::Drive`。出了什么问题？问题在于我们没有告诉编译器`Vehicle`类中的`Drive`方法可以被派生类修改（或覆盖）。我们需要在代码中做出一些改变：

```cpp
virtual void Drive()
{
  std::cout << "Vehicle::Drive\n";
};
```

通过在成员函数声明之前添加`virtual`关键字，我们告诉编译器派生类可以（但不一定）覆盖或替换该函数。如果我们进行此更改，然后编译并运行程序，将得到以下输出：

![图 2B.4：带有虚方法的车辆程序输出](img/C14583_02B_04.jpg)

###### 图 2B.4：带有虚方法的车辆程序输出

现在，我们已经了解了继承和多态性。我们使用`Vehicle`类的指针来控制`Motorcycle`类。作为最佳实践的一部分，应该对代码进行另一个更改。我们还应该更改`Motorcyle`中`Drive`函数的声明如下：

```cpp
void Drive() override
{
  std::cout << "Motorcycle::Drive\n";
};
```

C++11 引入了`override`关键字，作为向编译器的提示，说明特定方法应具有与其父树中某个方法相同的函数原型。如果找不到，则编译器将报告错误。这是一个非常有用的功能，可以帮助您节省数小时的调试时间。如果编译器有办法报告错误，请使用它。缺陷检测得越早，修复就越容易。最后一个变化是，每当我们向类添加虚函数时，必须声明其析构函数为`virtual`：

```cpp
class Vehicle
{
public:
  // Constructors - hidden 
  virtual ~Vehicle() = default;  // Virtual Destructor
  // Other methods and data -- hidden
};
```

在将`Drive()`函数设为虚函数之前，我们已经看到了这一点。当通过指向 Vehicle 的指针调用析构函数时，需要知道调用哪个析构函数。因此，将其设为虚函数可以实现这一点。如果未能这样做，可能会导致资源泄漏或对象被切割。

### 继承和访问说明符

正如我们之前提到的，从超类继承一个子类的一般形式如下：

```cpp
class DerivedClassName : access_modifier BaseClassName
```

当我们从 Vehicle 类派生 Motorcycle 类时，我们使用以下代码：

```cpp
class Motorcycle : public Vehicle
```

访问修饰符是可选的，是我们之前遇到的`public`、`protected`和`private`之一。在下表中，您可以看到基类成员的可访问性。如果省略 access_modifier，则编译器会假定指定了 private。

![图 2B.5：派生类中基类成员的可访问性](img/C14583_02B_05.jpg)

###### 图 2B.5：派生类中基类成员的可访问性

### 抽象类和接口

到目前为止，我们谈论过的所有类都是**具体类** - 它们可以实例化为变量的类型。还有另一种类型的类 - **抽象类** - 它包含至少一个**纯虚成员函数**。纯虚函数是一个在类中没有定义（或实现）的虚函数。由于它没有实现，该类是畸形的（或抽象的），无法实例化。如果尝试创建抽象类型的变量，则编译器将生成错误。

要声明纯虚成员函数，将函数原型声明结束为`= 0`。要将`Drive()`作为 Vehicle 类中的纯虚函数声明，我们将其声明如下：

```cpp
virtual void Drive() = 0;
```

现在，为了能够将派生类用作变量类型（例如`Motorcycle`类），它必须定义`Drive()`函数的实现。

但是，您可以声明变量为抽象类的指针或引用。在任何一种情况下，它必须指向或引用从抽象类派生的某个非抽象类。

在 Java 中，有一个关键字接口，允许你定义一个全是纯虚函数的类。在 C++中，通过声明一个只声明公共纯虚函数（和虚析构函数）的类来实现相同的效果。通过这种方式，我们定义了一个接口。

#### 注意

在本章中解决任何实际问题之前，请下载本书的 GitHub 存储库（[`github.com/TrainingByPackt/Advanced-CPlusPlus`](https://github.com/TrainingByPackt/Advanced-CPlusPlus)）并在 Eclipse 中导入 Lesson 2B 文件夹，以便查看每个练习和活动的代码。

### 练习 1：使用多态实现游戏角色

在这个练习中，我们将演示继承、接口和多态。我们将从一个临时实现的角色扮演游戏开始，然后将其演变为更通用和可扩展的形式。让我们开始吧：

1.  打开 Eclipse，并使用**Lesson2B**示例文件夹中的文件创建一个名为**Lesson2B**的新项目。

1.  由于这是一个**基于 CMake 的项目**，将当前构建器更改为**Cmake Build (portable)**。

1.  转到**项目** | **构建所有**菜单以构建所有练习。默认情况下，屏幕底部的控制台将显示**CMake 控制台[Lesson2B]**。

1.  配置一个名为**L2BExercise1**的**新启动配置**，运行**Exercise1**二进制文件，然后点击**运行**以构建和运行**Exercise 1**。你将收到以下输出：![图 2B.6：练习 1 默认输出](img/C14583_02B_06.jpg)

###### 图 2B.6：练习 1 默认输出

1.  直接打开`speak()`和`act()`。对于一个小程序来说这是可以的。但是当游戏扩大到几十甚至上百个角色时，就会变得难以管理。因此，我们需要将所有角色抽象出来。在文件顶部添加以下接口声明：

```cpp
    class ICharacter
    {
    public:
        ~ICharacter() {
            std::cout << "Destroying Character\n";
        }
        virtual void speak() = 0;
        virtual void act() = 0;
    };
    ```

通常，析构函数将是空的，但在这里，它有日志来显示行为。

1.  从这个接口类派生`Wizard`、`Healer`和`Warrior`类，并在每个类的`speak()`和`act()`函数声明末尾添加`override`关键字：

```cpp
    class Wizard : public Icharacter { ...
    ```

1.  点击**运行**按钮重新构建和运行练习。现在我们将看到在派生类的析构函数之后也调用了基类的析构函数：![图 2B.7：修改后程序的输出](img/C14583_02B_07.jpg)

###### 图 2B.7：修改后程序的输出

1.  创建角色并在容器中管理它们，比如`vector`。在`main()`函数之前在文件中创建以下两个方法：

```cpp
    void createCharacters(std::vector<ICharacter*>& cast)
    {
        cast.push_back(new Wizard("Gandalf"));
        cast.push_back(new Healer("Glenda"));
        cast.push_back(new Warrior("Ben Grimm"));
    }
    void freeCharacters(std::vector<ICharacter*>& cast)
    {
        for(auto* character : cast)
        {
            delete character;
        }
        cast.clear();
    }
    ```

1.  用以下代码替换`main()`的内容：

```cpp
    int main(int argc, char**argv)
    {
        std::cout << "\n------ Exercise 1 ------\n";
        std::vector<ICharacter*> cast;
        createCharacters(cast);
        for(auto* character : cast)
        {
            character->speak();
        }
        for(auto* character : cast)
        {
            character->act();
        }
        freeCharacters(cast);
        std::cout << "Complete.\n";
        return 0;
    }
    ```

1.  点击**运行**按钮重新构建和运行练习。以下是生成的输出：![图 2B.8：多态版本的输出](img/C14583_02B_08.jpg)

###### 图 2B.8：多态版本的输出

从上面的截图中可以看出，“销毁巫师”等日志已经消失了。问题在于容器保存了指向基类的指针，并且不知道如何在每种情况下调用完整的析构函数。

1.  为了解决这个问题，只需将`ICharacter`的析构函数声明为虚函数：

```cpp
    virtual ~ICharacter() {
    ```

1.  点击**运行**按钮重新构建和运行练习。输出现在如下所示：

![图 2B.9：完整多态版本的输出](img/C14583_02B_09.jpg)

###### 图 2B.9：完整多态版本的输出

我们现在已经为我们的`ICharacter`角色实现了一个接口，并通过在容器中存储基类指针简单地调用`speak()`和`act()`方法进行了多态使用。

### 类、结构体和联合体再讨论

之前我们讨论过类和结构体的区别是默认访问修饰符 - 类的为私有，结构体的为公共。这个区别更进一步 - 如果基类没有指定任何内容，它将应用于基类：

```cpp
class DerivedC : Base  // inherits as if "class DerivedC : private Base" was used
{
};
struct DerivedS : Base // inherits as if "struct DerivedS : public Base" was used
{
};
```

应该注意的是，联合既不能是基类，也不能从基类派生。如果结构和类之间本质上没有区别，那么我们应该使用哪种类型？本质上，这是一种惯例。**结构**用于捆绑几个相关的元素，而**类**可以执行操作并具有责任。结构的一个例子如下：

```cpp
struct Point     // A point in 3D space
{
  double m_x;
  double m_y;
  double m_z;
};
```

在前面的代码中，我们可以看到它将三个坐标组合在一起，这样我们就可以推断出三维空间中的一个点。这个结构可以作为一个连贯的数据集传递给需要点的方法，而不是每个点的三个单独的参数。另一方面，类模拟了一个可以执行操作的对象。看看下面的例子：

```cpp
class Matrix
{
public:
  Matrix& operator*(const Matrix& rhs)
  {
     // nitty gritty of the multiplication
  }
private:
  // Declaration of the 2D array to store matrix.
};
```

经验法则是，如果至少有一个私有成员，则应使用类，因为这意味着实现的细节将在公共成员函数的后面。

## 可见性、生命周期和访问

我们已经讨论了创建自己的类型和声明变量和函数，主要关注简单函数和单个文件。现在我们将看看当有多个包含类和函数定义的源文件（翻译单元）时会发生什么。此外，我们将检查哪些变量和函数可以从源文件的其他部分可见，变量的生存周期有多长，并查看内部链接和外部链接之间的区别。在*第一章*，*可移植 C++软件的解剖学*中，我们看到了工具链是如何工作的，编译源文件并生成目标文件，链接器将其全部组合在一起形成可执行程序。

当编译器处理源文件时，它会生成一个包含转换后的 C++代码和足够信息的目标文件，以便链接器解析已编译源文件到另一个源文件的任何引用。在*第一章*，*可移植 C++软件的解剖学*中，`sum()`在**SumFunc.cpp**文件中定义。当编译器构建目标文件时，它创建以下段：

+   **代码段**（也称为文本）：这是 C++函数翻译成目标机器指令的结果。

+   **数据段**：这包含程序中声明的所有变量和数据结构，不是本地的或从堆栈分配的，并且已初始化。

+   **BSS 段**：这包含程序中声明的所有变量和数据结构，不是本地的或从堆栈分配的，并且未初始化（但将初始化为零）。

+   **导出符号数据库**：此对象文件中的变量和函数列表及其位置。

+   **引用符号数据库**：此对象文件需要从外部获取的变量和函数列表以及它们的使用位置。

#### 注意

BSS 用于命名未初始化的数据段，其名称历史上源自 Block Started by Symbol。

然后，链接器将所有代码段、数据段和**BSS**段收集在一起形成程序。它使用两个数据库（DB）中的信息将所有引用的符号解析为导出的符号列表，并修补代码段，使其能够正确运行。从图形上看，这可以表示如下：

![图 2B.10：目标文件和可执行文件的部分](img/C14583_02B_10.jpg)

###### 图 2B.10：目标文件和可执行文件的部分

为了后续讨论的目的，BSS 和数据段将简称为数据段（唯一的区别是 BSS 未初始化）。当程序执行时，它被加载到内存中，其内存看起来有点像可执行文件布局 - 它包含文本段、数据段、BSS 段以及主机系统分配的空闲内存，其中包含所谓的**堆栈**和**堆**。堆栈通常从内存顶部开始并向下增长，而堆从 BSS 结束的地方开始并向上增长，朝向堆栈：

![图 2B.11：CxxTemplate 运行时内存映射](img/C14583_02B_11.jpg)

###### 图 2B.11：CxxTemplate 运行时内存映射

变量或标识符可访问的程序部分称为**作用域**。作用域有两个广泛的类别：

+   `{}`). 变量可以在大括号内部访问。就像块可以嵌套一样，变量的作用域也可以嵌套。这通常包括局部变量和函数参数，这些通常存储在堆栈中。

+   **全局/文件作用域**：这适用于在普通函数或类之外声明的变量，以及普通函数。可以在文件中的任何地方访问变量，并且如果链接正确，可能还可以从其他文件（全局）访问。这些变量由链接器在数据段中分配内存。标识符被放入全局命名空间，这是默认命名空间。

### 命名空间

我们可以将命名空间看作是变量、函数和用户定义类型的名称字典。对于小型程序，使用全局命名空间是可以的，因为很少有可能创建多个具有相同名称并发生名称冲突的变量。随着程序变得更大，并且包含了更多的第三方库，名称冲突的机会增加。因此，库编写者将他们的代码放入一个命名空间（希望是唯一的）。这允许程序员控制对命名空间中标识符的访问。通过使用标准库，我们已经在使用 std 命名空间。命名空间的声明如下：

```cpp
namespace name_of_namespace {  // put declarations in here }
```

通常，name_of_namespace 很短，命名空间可以嵌套。

#### 注意

在 boost 库中可以看到命名空间的良好使用：[`www.boost.org/`](https://www.boost.org/)。

变量还有另一个属性，即**寿命**。有三种基本寿命；两种由编译器管理，一种由程序员选择：

+   **自动寿命**：局部变量在声明时创建，并在退出其所在的作用域时被销毁。这些由堆栈管理。

+   **永久寿命**：全局变量和静态局部变量。编译器在程序开始时（进入 main()函数之前）创建全局变量，并在首次访问静态局部变量时创建它们。在这两种情况下，变量在程序退出时被销毁。这些变量由链接器放置在数据段中。

+   `new`和`delete`）。这些变量的内存是从堆中分配的。

我们将考虑的变量的最终属性是**链接**。链接指示编译器和链接器在遇到具有相同名称（或标识符）的变量和函数时会执行什么操作。对于函数，实际上是所谓的重载名称 - 编译器使用函数的名称、返回类型和参数类型来生成重载名称。有三种类型的链接：

+   **无链接**：这意味着标识符只引用自身，并适用于局部变量和本地定义的用户类型（即在块内部）。

+   **内部链接**：这意味着可以在声明它的文件中的任何地方访问该标识符。这适用于静态全局变量、const 全局变量、静态函数以及文件中匿名命名空间中声明的任何变量或函数。匿名命名空间是一个没有指定名称的命名空间。

+   **外部链接**：这意味着在正确的前向声明的情况下，可以从所有文件中访问它。这包括普通函数、非静态全局变量、extern const 全局变量和用户定义类型。

虽然这些被称为链接，但只有最后一个实际上涉及链接器。其他两个是通过编译器排除导出标识符数据库中的信息来实现的。

## 模板-泛型编程

作为计算机科学家或编程爱好者，您可能在某个时候不得不编写一个（或多个）排序算法。在讨论算法时，您可能并不特别关心正在排序的数据类型，只是该类型的两个对象可以进行比较，并且该域是一个完全有序的集合（也就是说，如果一个对象与任何其他对象进行比较，您可以确定哪个排在前面）。不同的编程语言为这个问题提供了不同的解决方案：

+   `swap`函数。

+   `void 指针`。`size_t`大小定义了每个对象的大小，而`compare()`函数定义了如何比较这两个对象。

+   `std::sort()`是标准库中提供的一个函数，其中一个签名如下：

```cpp
    template< class RandomIt > void sort( RandomIt first, RandomIt last );
    ```

在这种情况下，类型的细节被捕获在名为`RandomIt`的迭代器类型中，并在编译时传递给方法。

在下一节中，我们将简要定义泛型编程，展示 C++如何通过模板实现它们，突出语言已经提供的内容，并讨论编译器如何推断类型，以便它们可以用于模板。

### 什么是泛型编程？

当您开发排序算法时，您可能最初只关注对普通数字的排序。但一旦建立了这一点，您就可以将其抽象为任何类型，只要该类型具有某些属性，例如完全有序集（即比较运算符<在我们正在排序的域中的所有元素之间都有意义）。因此，为了以泛型编程的方式表达算法，我们在算法中为需要由该算法操作的类型定义了一个占位符。

**泛型编程**是开发一种类型不可知的通用算法。通过传递类型作为参数，可以重用该算法。这样，算法被抽象化，并允许编译器根据类型进行优化。

换句话说，泛型编程是一种编程方法，其中算法是以参数化的类型定义的，当实例化算法时指定了参数。许多语言提供了不同名称的泛型编程支持。在 C++中，泛型编程是通过模板这种语言特性来支持的。

### 介绍 C++模板

模板是 C++对泛型编程的支持。把模板想象成一个饼干模具，我们给它的类型参数就像饼干面团（可以是巧克力布朗尼、姜饼或其他美味口味）。当我们使用饼干模具时，我们得到的饼干实例形式相同，但口味不同。因此，模板捕获了泛型函数或类的定义，当指定类型参数时，编译器会根据我们手动编码的类型来为我们编写类或函数。它有几个优点，例如：

+   您只需要开发一次类或算法，然后进行演化。

+   您可以将其应用于许多类型。

+   您可以将复杂细节隐藏在简单的接口后，编译器可以根据类型对生成的代码进行优化。

那么，我们如何编写一个模板呢？让我们从一个模板开始，它允许我们将值夹在从`lo`到`hi`的范围内，并且能够在`int`、`float`、`double`或任何其他内置类型上使用它：

```cpp
template <class T>
T clamp(T val, T lo, T hi)
{
  return (val < lo) ? lo : (hi < val) ? hi : val;
}
```

让我们来分解一下：

+   `template <class T>`声明接下来是一个模板，并使用一个类型，模板中有一个`T`的占位符。

+   `T`被替换。它声明函数 clamp 接受三个类型为`T`的参数，并返回类型为`T`的值。

+   `<`运算符，然后我们可以对三个值执行 clamp，使得`lo <= val <= hi`。这个算法对所有可以排序的类型都有效。

假设我们在以下程序中使用它：

```cpp
#include <iostream>
int main()
{
    std::cout << clamp(5, 3, 10) << "\n";
    std::cout << clamp(3, 5, 10) << "\n";
    std::cout << clamp(13, 3, 10) << "\n";
    std::cout << clamp(13.0, 3.0, 10.1) << "\n";
    std::cout << clamp<double>(13.0, 3, 10.2) << "\n";
    return 0;
}
```

我们将得到以下预期输出：

![图 2B.12：Clamp 程序输出](img/C14583_02B_12.jpg)

###### 图 2B.12：Clamp 程序输出

在最后一次调用 clamp 时，我们在`<`和`>`之间传递了 double 类型的模板。但是我们没有对其他四个调用遵循相同的方式。为什么？原来编译器随着年龄的增长变得越来越聪明。随着每个标准的发布，它们改进了所谓的**类型推导**。因为编译器能够推断类型，我们不需要告诉它使用什么类型。这是因为类的三个参数没有模板参数，它们具有相同的类型 - 前三个都是 int，而第四个是 double。但是我们必须告诉编译器使用最后一个的类型，因为它有两个 double 和一个 int 作为参数，这导致编译错误说找不到函数。但是然后，它给了我们关于为什么不能使用模板的信息。这种形式，你强制类型，被称为**显式模板参数规定**。

### C++预打包模板

C++标准由两个主要部分组成：

+   语言定义，即关键字、语法、词法定义、结构等。

+   标准库，即编译器供应商提供的所有预先编写的通用函数和类。这个库的一个子集是使用模板实现的，被称为**标准模板库**（**STL**）。

STL 起源于 Ada 语言中提供的泛型，该语言由 David Musser 和 Alexander Stepanov 开发。Stepanov 是泛型编程作为软件开发基础的坚定支持者。在 90 年代，他看到了用新语言 C++来影响主流开发的机会，并建议 ISO C++委员会应该将 STL 作为语言的一部分包含进去。其余的就是历史了。

STL 由四类预定义的通用算法和类组成：

+   **容器**：通用序列（vector，list，deque）和关联容器（set，multiset，map）

+   `begin()`和`end()`）。请注意，STL 中的一个基本设计选择是`end()`指向最后一项之后的位置 - 在数学上，即`begin()`，`end()`)。

+   **算法**：涵盖排序、搜索、集合操作等 100 多种不同算法。

+   `find_if()`.

我们之前实现的 clamp 函数模板是简单的，虽然它适用于支持小于运算符的任何类型，但它可能不太高效 - 如果类型具有较大的大小，可能会导致非常大的副本。自 C++17 以来，STL 包括一个`std::clamp()`函数，声明更像这样：

```cpp
#include <cassert>
template<class T, class Compare>
const T& clamp( const T& v, const T& lo, const T& hi, Compare comp )
{
    return assert( !comp(hi, lo) ),
        comp(v, lo) ? lo : comp(hi, v) ? hi : v;
}
template<class T>
const T& clamp( const T& v, const T& lo, const T& hi )
{
    return clamp( v, lo, hi, std::less<>() );
}
```

正如我们所看到的，它使用引用作为参数和返回值。将参数更改为使用引用减少了需要传递和返回的堆栈上的内容。还要注意，设计者们努力制作了模板的更通用版本，这样我们就不会依赖于类型存在的<运算符。然而，我们可以通过传递 comp 来定义排序。

从前面的例子中，我们已经看到，像函数一样，模板可以接受多个逗号分隔的参数。

## 类型别名 - typedef 和 using

如果您使用了`std::string`类，那么您一直在使用别名。有一些与字符串相关的模板类需要实现相同的功能。但是表示字符的类型是不同的。例如，对于`std::string`，表示是`char`，而`std::wstring`使用`wchar_t`。还有一些其他的用于`char16_t`和`char32_t`。任何功能上的变化都将通过特性或模板特化来管理。

在 C++11 之前，这将从`std::basic_string`基类中进行别名处理，如下所示：

```cpp
namespace std {
  typedef basic_string<char> string;
}
```

这做了两件主要的事情：

+   减少声明变量所需的输入量。这是一个简单的情况，但是当你声明一个指向字符串到对象的映射的唯一指针时，可能会变得非常长，你会犯错误：

```cpp
    typedef std::unique_ptr<std::map<std::string,myClass>> UptrMapStrToClass;
    ```

+   提高了可读性，因为现在你在概念上将其视为一个字符串，不需要担心细节。

但是 C++11 引入了一种更好的方式 - `别名声明` - 它利用了`using`关键字。前面的代码可以这样实现：

```cpp
namespace std {
  using string = basic_string<char>;
}
```

前面的例子很简单，别名，无论是 typedef 还是 using，都不太难理解。但是当别名涉及更复杂的表达式时，它们也可能有点难以理解 - 特别是函数指针。考虑以下代码：

```cpp
typedef int (*FunctionPointer)(const std::string&, const Point&); 
```

现在，考虑以下代码：

```cpp
using FunctionPointer = int (*)(const std::string&, const Point&);
```

C++11 中有一个新功能，即别名声明可以轻松地并入模板中 - 它们可以被模板化。`typedef`不能被模板化，虽然可以通过`typedef`实现相同的结果，但别名声明（`using`）是首选方法，因为它会导致更简单、更易于理解的模板代码。

### 练习 2：实现别名

在这个练习中，我们将使用 typedef 实现别名，并看看通过使用引用使代码变得更容易阅读和高效。按照以下步骤实现这个练习：

1.  在 Eclipse 中打开**Lesson2B**项目，然后在项目资源管理器中展开**Lesson2B**，然后展开**Exercise02**，双击**Exercise2.cpp**以在编辑器中打开此练习的文件。

1.  单击**启动配置**下拉菜单，然后选择**新启动配置...**。配置**L2BExercise2**以使用名称**Exercise2**运行。完成后，它将成为当前选择的启动配置。

1.  单击**运行**按钮。**Exercise 2**将运行并产生类似以下输出：![图 2B.13：练习 2 输出###### 图 2B.13：练习 2 输出 1.  在编辑器中，在`printVector()`函数的声明之前，添加以下行：```cpp    typedef std::vector<int> IntVector;    ```1.  现在，将文件中所有的`std::vector<int>`更改为`IntVector`。1.  单击**运行**按钮。输出应与以前相同。1.  在编辑器中，更改之前添加的行为以下内容：```cpp    using IntVector = std::vector<int>;    ```1.  单击**运行**按钮。输出应与以前相同。1.  在编辑器中，添加以下行：```cpp    using IntVectorIter = std::vector<int>::iterator;    ```1.  现在，将`IntVector::iterator`的一个出现更改为`IntVectorIter`。1.  单击**运行**按钮。输出应与以前相同。在这个练习中，typedef 和使用别名似乎没有太大区别。在任何一种情况下，使用一个命名良好的别名使得代码更容易阅读和理解。当涉及更复杂的别名时，`using`提供了一种更容易编写别名的方法。在 C++11 中引入，`using`现在是定义别名的首选方法。它还比`typedef`有其他优点，例如能够在模板内部使用它。## 模板 - 不仅仅是泛型编程模板还可以提供比泛型编程更多的功能（一种带有类型的模板）。在泛型编程的情况下，模板作为一个不能更改的蓝图运行，并为指定的类型或类型提供模板的编译版本。模板可以被编写以根据涉及的类型提供函数或算法的特化。这被称为**模板特化**，并不是我们先前使用的意义上的通用编程。只有当它使某些类型在给定上下文中表现得像我们期望它们在某个上下文中表现得一样时，它才能被称为通用编程。当用于所有类型的算法被修改时，它不能被称为通用编程。检查以下专业化代码的示例：```cpp#include <iostream>#include <type_traits>template <typename T, std::enable_if_t<sizeof(T) == 1, int> = 0>void print(T val){    printf("%c\n", val);}template <typename T, std::enable_if_t<sizeof(T) == sizeof(int), int> = 0>void print(T val){    printf("%d\n", val);}template <typename T, std::enable_if_t<sizeof(T) == sizeof(double), int> = 0>void print(T val){    printf("%f\n", val);}int main(int argc, char** argv){    print('c');    print(55);    print(32.1F);    print(77.3);}```它定义了一个模板，根据使用`std::enable_if_t<>`和`sizeof()`的模板的特化，调用`printf()`并使用不同的格式字符串。当我们运行它时，会生成以下输出：![图 2B.14：错误的打印模板程序输出](img/C14583_02B_14.jpg)

###### 图 2B.14：错误的打印模板程序输出

### 替换失败不是错误 - SFINAE

对于`32.1F`打印的值（`-1073741824`）与数字毫不相干。如果我们检查编译器为以下程序生成的代码，我们会发现它生成的代码就好像我们写了以下内容（以及更多）：

```cpp
template<typename int, int=0>
void print<int,0>(int val)
{
    printf("%d\n",val);
}
template<typename float, int=0>
void print<float,0>(float val)
{
    printf("%d\n", val);
}
```

为什么会生成这段代码？前面的模板使用了 C++编译器的一个特性，叫做`std::enable_if_t<>`，并访问了所谓的**类型特征**来帮助我们。首先，我们将用以下代码替换最后一个模板：

```cpp
#include <type_traits>
template <typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
void print(T val)
{
    printf("%f\n", val);
}
```

这需要一些解释。首先，我们考虑`std::enable_if_t`的定义，实际上是一个类型别名：

```cpp
template<bool B, class T = void>
struct enable_if {};
template<class T>
struct enable_if<true, T> { typedef T type; };
template< bool B, class T = void >
using enable_if_t = typename enable_if<B,T>::type;
```

`enable_if`的第一个模板将导致定义一个空的结构体（或类）。`enable_if`的第二个模板是对 true 的第一个模板参数的特化，将导致具有 typedef 定义的类。`enable_if_t`的定义是一个帮助模板，它消除了我们在使用它时需要在模板末尾输入`::type`的需要。那么，这是如何工作的呢？考虑以下代码：

```cpp
template <typename T, std::enable_if_t<condition, int> = 0>
void print(T val) { … }
```

如果在编译时评估的条件导致`enable_if_t`模板将导致一个看起来像这样的模板：

```cpp
template <typename T, int = 0>
void print(T val) { … }
```

这是有效的语法，函数被添加到符号表作为候选函数。如果在编译时计算的条件导致`enable_if_t`模板将导致一个看起来像这样的模板：

```cpp
template <typename T, = 0>
void print(T val) { … }
```

这是**格式错误的代码**，现在被丢弃了 - SFINAE 在起作用。

`std::is_floating_point_v<T>`是另一个访问`std::is_floating_point<T>`模板的`::value`成员的帮助类。它的名字说明了一切 - 如果 T 是浮点类型（float、double、long double），它将为 true；否则，它将为 false。如果我们进行这个改变，那么编译器（GCC）会生成以下错误：

![图 2B.15：修改后的打印模板程序的编译器错误](img/C14583_02B_15.jpg)

###### 图 2B.15：修改后的打印模板程序的编译器错误

现在的问题是，当类型是浮点数时，我们有两个可以满足的模板：

```cpp
template <typename T, std::enable_if_t<sizeof(T) == sizeof(int), int> = 0>
void print(T val)
{
    printf("%d\n", val);
}
template <typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
void print(T val)
{
    printf("%f\n", val);
}
```

事实证明，通常情况下`sizeof(float) == sizeof(int)`，所以我们需要做另一个改变。我们将用另一个类型特征`std::is_integral_v<>`替换第一个条件：

```cpp
template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
void print(T val)
{
    printf("%d\n", val);
}
```

如果我们进行这个改变，那么编译器（GCC）会生成以下错误：

![图 2B.16：修改后的打印模板程序的第二个编译器错误](img/C14583_02B_16.jpg)

###### 图 2B.16：修改后的打印模板程序的第二个编译器错误

我们解决了浮点数的歧义，但这里的问题是`std::is_integral_v(char)`返回 true，再次生成了具有相同原型的模板函数。原来传递给`std::enable_if_t<>`的条件遵循标准 C++逻辑表达式。因此，为了解决这个问题，我们将添加一个额外的条件来排除字符：

```cpp
template <typename T, std::enable_if_t<std::is_integral_v<T> && sizeof(T) != 1, int> = 0>
void print(T val)
{
    printf("%d\n", val);
}
```

如果我们现在编译程序，它完成编译并链接程序。如果我们运行它，它现在会产生以下（预期的）输出：

![图 2B.17：修正的打印模板程序输出](img/C14583_02B_17.jpg)

###### 图 2B.17：修正的打印模板程序输出

### 浮点表示

`32.099998`不应该是`32.1`吗？这是传递给函数的值。在计算机上执行浮点运算的问题在于，表示自动引入了误差。实数形成一个连续（无限）的域。如果你考虑实域中的数字 1 和 2，那么它们之间有无限多个实数。不幸的是，计算机对浮点数的表示量化了这些值，并且无法表示所有无限数量的数字。用于存储数字的位数越多，值在实域上的表示就越好。因此，long double 比 double 好，double 比 float 好。对于存储数据来说，真正取决于您的问题域。回到`32.099998`。计算机将单精度数存储为 2 的幂的和，然后将它们移位一个幂因子。整数通常很容易，因为它们可以很容易地表示为`2^n`的和（n>=0）。在这种情况下的小数部分，即 0.1，必须表示为`2^(-n) (n>0)`的和。我们添加更多的 2 的幂分数，以尝试使数字更接近目标，直到我们用完了单精度浮点数中的 24 位精度。

#### 注意

如果您想了解计算机如何存储浮点数，请研究定义它的 IEEE 754 标准。

### Constexpr if 表达式

C++17 引入了`constexpr if`表达式到语言中，大大简化了模板编写。我们可以将使用 SFINAE 的前面三个模板重写为一个更简单的模板：

```cpp
#include <iostream>
#include <type_traits>
template <typename T>
void print(T val)
{
   if constexpr(sizeof(T)==1) {
      printf("%c",val);
   }
   else if constexpr(std::is_integral_v<T>) {
      printf("%d",val);
   }
   else if constexpr(std::is_floating_point_v<T>) {
      printf("%f",val);
   }
   printf("\n");
}
int main(int argc, char** argv)
{
    print('c');
    print(55);
    print(32.1F);
    print(77.3);
}
```

对于对`print(55)`的调用，编译器生成的函数调用如下：

```cpp
template<>
void print<int>(int val)
{
    printf("%d",val);
    printf("\n");
}
```

if/else if 语句发生了什么？`constexpr if`表达式的作用是，编译器在上下文中确定条件的值，并将其转换为布尔值（true/false）。如果评估的值为 true，则 if 条件和 else 子句被丢弃，只留下 true 子句生成代码。同样，如果为 false，则留下 false 子句生成代码。换句话说，只有第一个 constexpr if 条件评估为 true 时，才会生成其子句的代码，其余的都会被丢弃。

### 非类型模板参数

到目前为止，我们只看到了作为模板参数的类型。还可以将整数值作为模板参数传递。这允许我们防止函数的数组衰减。例如，考虑一个计算`sum`的模板函数：

```cpp
template <class T>
T sum(T data[], int number)
{
  T total = 0;
  for(auto i=0U ; i<number ; i++)
  {
    total += data[i];
  }
  return total;
}
```

在这种情况下，我们需要在调用中传递数组的长度：

```cpp
float data[5] = {1.1, 2.2, 3.3, 4.4, 5.5};
auto total = sum(data, 5);
```

但是，如果我们只能调用以下内容会不会更好呢？

```cpp
auto total = sum(data);
```

我们可以通过对模板进行更改来实现，就像下面的代码一样：

```cpp
template <class T, std::size_t size>
T sum(T (&data)[size])
{
  T total = 0;
  for(auto i=0U ; i< size; i++)
  {
    total += data[i];
  }
  return total;
}
```

在这里，我们将数据更改为对模板传递的特定大小的数组的引用，因此编译器会自行解决。我们不再需要函数调用的第二个参数。这个简单的例子展示了如何直接传递和使用非类型参数。我们将在*模板类型推导*部分进一步探讨这个问题。

### 练习 3：实现 Stringify - 专用与 constexpr

在这个练习中，我们将利用 constexpr 实现一个 stringify 模板，以生成一个更易读和更简单的代码版本。按照以下步骤实现这个练习：

#### 注意

可以在[`isocpp.org/wiki/faq/templates#template-specialization-example`](https://isocpp.org/wiki/faq/templates#template-specialization-example)找到 stringify 专用模板。

1.  在 Eclipse 中打开**Lesson2B**项目，然后在**项目资源管理器**中展开**Lesson2B**，然后展开**Exercise03**，双击**Exercise3.cpp**以在编辑器中打开此练习的文件。

1.  单击**启动配置**下拉菜单，选择**新启动配置...**。配置**L2BExercise3**以使用名称**Exercise3**运行。

1.  单击**运行**按钮。**练习 3**将运行并产生以下输出：![图 2B.18：练习 3 特化模板输出](img/C14583_02B_18.jpg)

###### 图 2B.18：练习 3 特化模板输出

1.  在**Exercise3.cpp**中，将 stringify 模板的所有特化模板注释掉，同时保留原始的通用模板。

1.  单击**运行**按钮。输出将更改为将布尔型打印为数字，将双精度浮点数打印为仅有两位小数：![图 2B.19：练习 3 仅通用模板输出](img/C14583_02B_19.jpg)

###### 图 2B.19：练习 3 仅通用模板输出

1.  我们现在将再次为布尔类型“特化”模板。在其他`#includes`中添加`#include <type_traits>`指令，并修改模板，使其如下所示：

```cpp
    template<typename T> std::string stringify(const T& x)
    {
      std::ostringstream out;
      if constexpr (std::is_same_v<T, bool>)
      {
          out << std::boolalpha;
      }
      out << x;
      return out.str();
    }
    ```

1.  单击**运行**按钮。布尔型的 stringify 输出与以前一样：![图 2B.20：针对布尔型定制的 stringify](img/C14583_02B_20.jpg)

###### 图 2B.20：针对布尔型定制的 stringify

1.  我们现在将再次为浮点类型（`float`、`double`、`long double`）“特化”模板。修改模板，使其如下所示：

```cpp
    template<typename T> std::string stringify(const T& x)
    {
      std::ostringstream out;
      if constexpr (std::is_same_v<T, bool>)
      {
          out << std::boolalpha;
      }
      else if constexpr (std::is_floating_point_v<T>)
      {
          const int sigdigits = std::numeric_limits<T>::digits10;
          out << std::setprecision(sigdigits);
      }
      out << x;
      return out.str();
    }
    ```

1.  单击**运行**按钮。输出恢复为原始状态：![图 2B.21：constexpr if 版本模板输出](img/C14583_02B_21.jpg)

###### 图 2B.21：constexpr if 版本模板输出

1.  如果您将多个模板的原始版本与最终版本进行比较，您会发现最终版本更像是一个普通函数，更易于阅读和维护。

在这个练习中，我们学习了在 C++17 中使用新的 constexpr if 结构时，模板可以变得更简单和紧凑。

### 函数重载再探讨

当我们首次讨论函数重载时，我们只考虑了函数名称来自我们手动编写的函数列表的情况。现在，我们需要更新这一点。我们还可以编写可以具有相同名称的模板函数。就像以前一样，当编译器遇到`print(55)`这一行时，它需要确定调用先前定义的函数中的哪一个。因此，它执行以下过程（大大简化）：

![图 2B.22：模板的函数重载解析（简化版）](img/C14583_02B_22.jpg)

###### 图 2B.22：模板的函数重载解析（简化版）

### 模板类型推断

当我们首次介绍模板时，我们涉及了模板类型推断。现在，我们将进一步探讨这一点。我们将从考虑函数模板的一般声明开始：

```cpp
template<typename T>
void function(ParamType parameter);
```

此调用可能如下所示：

```cpp
function(expression);              // deduce T and ParamType from expression
```

当编译器到达这一行时，它现在必须推断与模板相关的两种类型—`T`和`ParamType`。由于 T 在 ParamType 中附加了限定符和其他属性（例如指针、引用、const 等），它们通常是不同的。这些类型是相关的，但推断的过程取决于所使用的`expression`的形式。

### 显示推断类型

在我们研究不同形式之前，如果我们能让编译器告诉我们它推断出的类型，那将非常有用。我们有几种选择，包括 IDE 编辑器显示类型、编译器生成错误和运行时支持（由于 C++标准的原因，这不一定有效）。我们将使用编译器错误来帮助我们探索一些类型推断。

我们可以通过声明一个没有定义的模板来实现类型显示器。任何尝试实例化模板都将导致编译器生成错误消息，因为没有定义，以及它正在尝试实例化的类型信息：

```cpp
template<typename T>
struct TypeDisplay;
```

让我们尝试编译以下程序：

```cpp
template<typename T>
class TypeDisplay;
int main()
{
    signed int x = 1;
    unsigned int y = 2;
    TypeDisplay<decltype(x)> x_type;
    TypeDisplay<decltype(y)> y_type;
    TypeDisplay<decltype(x+y)> x_y_type;
    return 0;
}
```

编译器输出以下错误：

![图 2B.23：显示推断类型的编译器错误](img/C14583_02B_23.jpg)

###### 图 2B.23：显示推断类型的编译器错误

请注意，在每种情况下，被命名的聚合包括被推断的类型 - 对于 x，它是一个 int，对于 y，是一个 unsigned int，对于 x+y，是一个 unsigned int。还要注意，TypeDisplay 模板需要其参数的类型，因此使用`decltype()`函数来获取编译器提供括号中表达式的类型。

还可以使用内置的`typeid(T).name()`运算符在运行时显示推断的类型，它返回一个 std::string，或者使用名为 type_index 的 boost 库。

#### 注意

有关更多信息，请访问以下链接：[`www.boost.org/doc/libs/1_70_0/doc/html/boost_typeindex.html`](https://www.boost.org/doc/libs/1_70_0/doc/html/boost_typeindex.html)。

由于类型推断规则，内置运算符将为您提供类型的指示，但会丢失引用（`&`和`&&`）和任何 constness 信息（const 或 volatile）。如果需要在运行时，考虑使用`boost::type_index`，它将为所有编译器产生相同的输出。

### 模板类型推断 - 详细信息

让我们回到通用模板：

```cpp
template<typename T>
void function(ParamType parameter);
```

假设调用看起来像这样：

```cpp
function(expression);             // deduce T and ParamType from expression
```

类型推断取决于 ParamType 的形式：

+   **ParamType 是值（T）**：按值传递函数调用

+   **ParamType 是引用或指针（T&或 T*）**：按引用传递函数调用

+   **ParamType 是右值引用（T&&）**：按引用传递函数调用或其他内容

**情况 1：ParamType 是按值传递（T）**

```cpp
template<typename T>
void function(T parameter);
```

作为按值传递的调用，这意味着参数将是传入内容的副本。因为这是对象的新实例，所以以下规则适用于表达式：

+   如果表达式的类型是引用，则忽略引用部分。

+   如果在步骤 1 之后，剩下的类型是 const 和/或 volatile，则也忽略它们。

剩下的是 T。让我们尝试编译以下文件代码：

```cpp
template<typename T>
class TypeDisplay;
template<typename T>
void function(T parameter)
{
    TypeDisplay<T> type;
}
void types()
{
    int x = 42;
    function(x);
}
```

编译器产生以下错误：

![图 2B.24：显示按类型推断类型的编译器错误](img/C14583_02B_24.jpg)

###### 图 2B.24：显示按类型推断类型的编译器错误

因此，类型被推断为`int`。同样，如果我们声明以下内容，我们将得到完全相同的错误：

```cpp
const int x = 42;
function(x);
```

如果我们声明这个版本，同样的情况会发生：

```cpp
int x = 42;
const int& rx = x;
function(rx);
```

在所有三种情况下，根据先前规定的规则，推断的类型都是`int`。

**情况 2：ParamType 是按引用传递（T&）**

作为按引用传递的调用，这意味着参数将能够访问对象的原始存储位置。因此，生成的函数必须遵守我们之前忽略的 constness 和 volatileness。类型推断适用以下规则：

+   如果表达式的类型是引用，则忽略引用部分。

+   模式匹配表达式类型的剩余部分与 ParamType 以确定 T。

让我们尝试编译以下文件：

```cpp
template<typename T>
class TypeDisplay;
template<typename T>
void function(T& parameter)
{
    TypeDisplay<T> type;
}
void types()
{
    int x = 42;
    function(x);
}
```

编译器将生成以下错误：

![图 2B.25：显示按引用传递推断类型的编译器错误](img/C14583_02B_25.jpg)

###### 图 2B.25：显示按引用传递推断类型的编译器错误

从这里，我们可以看到编译器将 T 作为`int`，从 ParamType 作为`int&`。将 x 更改为 const int 不会有任何意外，因为 T 被推断为`const int`，从 ParamType 作为`const int&`：

![图 2B.26：显示按 const 引用传递推断类型的编译器错误](img/C14583_02B_26.jpg)

###### 图 2B.26：传递 const 引用时显示推断类型的编译器错误

同样，像之前一样引入 rx 作为对 const int 的引用，不会有令人惊讶的地方，因为 T 从 ParamType 作为`const int&`推断为`const int`：

```cpp
void types()
{
    const int x = 42;
    const int& rx = x;
    function(rx);
}
```

![图 2B.27：传递 const 引用时显示推断类型的编译器错误](img/C14583_02B_27.jpg)

###### 图 2B.27：传递 const 引用时显示推断类型的编译器错误

如果我们改变声明以包括一个 const，那么编译器在从模板生成函数时将遵守 constness：

```cpp
template<typename T>
void function(const T& parameter)
{
    TypeDisplay<T> type;
}
```

这次，编译器报告如下

+   `int x`：T 是 int（因为 constness 将被尊重），而参数的类型是`const int&`。

+   `const int x`：T 是 int（const 在模式中，留下 int），而参数的类型是`const int&`。

+   `const int& rx`：T 是 int（引用被忽略，const 在模式中，留下 int），而参数的类型是`const int&`。

如果我们尝试编译以下内容，我们期望会发生什么？通常，数组会衰减为指针：

```cpp
int ary[15];
function(ary);
```

编译器错误如下：

![图 2B.28：传递数组参数时显示推断类型的编译器错误传递引用时](img/C14583_02B_28.jpg)

###### 图 2B.28：传递引用时显示数组参数的推断类型的编译器错误

这次，数组被捕获为引用，并且大小也被包括在内。因此，如果 ary 声明为`ary[10]`，那么将得到一个完全不同的函数。让我们将模板恢复到以下内容：

```cpp
template<typename T>
void function(T parameter)
{
    TypeDisplay<T> type;
}
```

如果我们尝试编译数组调用，那么错误报告如下：

![图 2B.29：传递数组参数时显示推断类型的编译器错误传递值时](img/C14583_02B_29.jpg)

###### 图 2B.29：传递值时显示数组参数的推断类型的编译器错误

我们可以看到，在这种情况下，数组已经衰减为传递数组给函数时的通常行为。当我们谈论*非类型模板参数*时，我们看到了这种行为。

**情况 3：ParamType 是右值引用（T&&）**

T&&被称为右值引用，而 T&被称为左值引用。C++不仅通过类型来表征表达式，还通过一种称为**值类别**的属性来表征。这些类别控制编译器中表达式的评估，包括创建、复制和移动临时对象的规则。C++17 标准中定义了五种表达式值类别，它们具有以下关系：

![图 2B.30：C++值类别](img/C14583_02B_30.jpg)

###### 图 2B.30：C++值类别

每个的定义如下：

+   决定对象身份的表达式是`glvalue`。

+   评估初始化对象或操作数的表达式是`prvalue`。例如，文字（除了字符串文字）如 3.1415，true 或 nullptr，this 指针，后增量和后减量表达式。

+   具有资源并且可以被重用（因为它的生命周期即将结束）的 glvalue 对象是`xvalue`。例如，返回类型为对象的右值引用的函数调用，如`std::move()`。

+   不是 xvalue 的 glvalue 是`lvalue`。例如，变量的名称，函数或数据成员的名称，或字符串文字。

+   prvalue 或 xvalue 是一个`rvalue`。

不要紧，如果你不完全理解这些，因为接下来的解释需要你知道什么是左值，以及什么不是左值：

```cpp
template<typename T>
void function(T&& parameter)
{
    TypeDisplay<T> type;
}
```

这种 ParamType 形式的类型推断规则如下：

+   如果表达式是左值引用，那么 T 和 ParamType 都被推断为左值引用。这是唯一一种类型被推断为引用的情况。

+   如果表达式是一个右值引用，那么适用于情况 2 的规则。

### SFINAE 表达式和尾返回类型

C++11 引入了一个名为`尾返回类型`的功能，为模板提供了一种通用返回类型的机制。一个简单的例子如下：

```cpp
template<class T>
auto mul(T a, T b) -> decltype(a * b) 
{
    return a * b;
}
```

这里，`auto`用于指示定义尾返回类型。尾返回类型以`->`指针开始，在这种情况下，返回类型是通过将`a`和`b`相乘返回的类型。编译器将处理 decltype 的内容，如果它格式不正确，它将从函数名的查找中删除定义，与往常一样。这种能力打开了许多可能性，因为逗号运算符“`,`”可以在`decltype`内部使用来检查某些属性。

如果我们想测试一个类是否实现了一个方法或包含一个类型，那么我们可以将其放在 decltype 内部，将其转换为 void（以防逗号运算符已被重载），然后在逗号运算符的末尾定义一个真实返回类型的对象。下面的程序示例中展示了这种方法：

```cpp
#include <iostream>
#include <algorithm>
#include <utility>
#include <vector>
#include <set>
template<class C, class T>
auto contains(const C& c, const T& x) 
             -> decltype((void)(std::declval<C>().find(std::declval<T>())), true)
{
    return end(c) != c.find(x);
}
int main(int argc, char**argv)
{
    std::cout << "\n\n------ SFINAE Exercise ------\n";
    std::set<int> mySet {1,2,3,4,5};
    std::cout << std::boolalpha;
    std::cout << "Set contains 5: " << contains(mySet,5) << "\n";
    std::cout << "Set contains 15: " << contains(mySet,15) << "\n";
    std::cout << "Complete.\n";
    return 0;
}
```

当编译并执行此程序时，我们将获得以下输出：

![图 2B.31：SFINAE 表达式的输出](img/C14583_02B_31.jpg)

###### 图 2B.31：SFINAE 表达式的输出

返回类型由以下代码给出：

```cpp
decltype( (void)(std::declval<C>().find(std::declval<T>())), true)
```

让我们来分解一下：

+   `decltype`的操作数是一个逗号分隔的表达式列表。这意味着编译器将构造但不评估表达式，并使用最右边的值的类型来确定函数的返回类型。

+   `std::declval<T>()`允许我们将 T 类型转换为引用类型，然后可以使用它来访问成员函数，而无需实际构造对象。

+   与所有基于 SFINAE 的操作一样，如果逗号分隔列表中的任何表达式无效，则函数将被丢弃。如果它们都有效，则将其添加到查找函数列表中。

+   将 void 转换是为了防止用户重载逗号运算符可能引发的任何问题。

+   基本上，这是在测试`C`类是否有一个名为`find()`的成员函数，该函数以`class T`、`class T&`或`const class T&`作为参数。

这种方法适用于`std::set`，它具有一个接受一个参数的`find()`方法，但对于其他容器来说会失败，因为它们没有`find()`成员方法。

如果我们只处理一种类型，这种方法效果很好。但是，如果我们有一个需要根据类型生成不同实现的函数，就像我们以前看到的那样，`if constexpr`方法更清晰，通常更容易理解。要使用`if constexpr`方法，我们需要生成在编译时评估为`true`或`false`的模板。标准库提供了这方面的辅助类：`std::true_type`和`std::false_type`。这两个结构都有一个名为 value 的静态常量成员，分别设置为`true`和`false`。使用 SFINAE 和模板重载，我们可以创建新的检测类，这些类从这些类中派生，以给出我们想要的结果：

```cpp
template <class T, class A0>
auto test_find(long) -> std::false_type;
template <class T, class A0>
auto test_find(int) 
-> decltype(void(std::declval<T>().find(std::declval<A0>())), std::true_type{});
template <class T, class A0>
struct has_find : decltype(test_find<T,A0>(0)) {};
```

`test_find`的第一个模板创建了将返回类型设置为`std::false_type`的默认行为。注意它的参数类型是`long`。

`test_find`的第二个模板创建了一个专门测试具有名为`find()`的成员函数并具有`std::true_type`返回类型的类的特化。注意它的参数类型是`int`。

`has_find<T,A0>`模板通过从`test_find()`函数的返回类型派生自身来工作。如果 T 类没有`find()`方法，则只会生成`std::false_type`版本的`test_find()`，因此`has_find<T,A0>::value`值将为 false，并且可以在`if constexpr()`中使用。

有趣的部分是，如果 T 类具有`find()`方法，则两个`test_find()`方法都会生成。但是专门的版本使用`int`类型的参数，而默认版本使用`long`类型的参数。当我们使用零（0）“调用”函数时，它将匹配专门的版本并使用它。参数的差异很重要，因为您不能有两个具有相同参数类型但仅返回类型不同的函数。如果要检查此行为，请将参数从 0 更改为 0L 以强制使用长版本。

## 类模板

到目前为止，我们只处理了函数模板。但是模板也可以用于为类提供蓝图。模板类声明的一般结构如下：

```cpp
template<class T>
class MyClass {
   // variables and methods that use T.
};
```

而模板函数允许我们生成通用算法，模板类允许我们生成通用数据类型及其相关行为。

当我们介绍标准模板库时，我们强调它包括容器的模板-`vector`，`deque`，`stack`等。这些模板允许我们存储和管理任何我们想要的数据类型，但仍然表现得像我们期望的那样。

### 练习 4：编写类模板

在计算科学中，最常用的两种数据结构是堆栈和队列。目前，STL 中已经有了它们的实现。但是为了尝试使用模板类，我们将编写一个可以用于任何类型的堆栈模板类。让我们开始吧：

1.  在 Eclipse 中打开**Lesson2B**项目，然后在**Project Explorer**中展开**Lesson2B**，然后展开**Exercise04**，双击**Exercise4.cpp**以在编辑器中打开此练习的文件。

1.  配置一个新的**Launch Configuration**，**L2BExercise4**，以运行名称为**Exercise4**的配置。

1.  还要配置一个新的 C/C++单元运行配置，**L2BEx4Tests**，以运行**L2BEx4tests**。设置**Google Tests Runner**。

1.  单击**运行**选项以运行测试，这是我们第一次运行：![图 2B.32：堆栈的初始单元测试](img/C14583_02B_32.jpg)

###### 图 2B.32：堆栈的初始单元测试

1.  打开`#pragma once`），告诉编译器如果再次遇到此文件要被#include，它就不需要了。虽然不严格属于标准的一部分，但几乎所有现代 C++编译器都支持它。最后，请注意，为了本练习的目的，我们选择将项目存储在 STL 向量中。

1.  在编辑器中，在`Stack`类的`public`部分中添加以下声明：

```cpp
    bool empty() const
    {
      return m_stack.empty();
    }
    ```

1.  在文件顶部，将**EXERCISE4_STEP**更改为值**10**。单击**运行**按钮。练习 4 的测试应该运行并失败：![图 2B.33：跳转到失败的测试](img/C14583_02B_33.jpg)

###### 图 2B.33：跳转到失败的测试

1.  单击失败测试的名称，即`empty()`报告为 false。

1.  将`ASSERT_FALSE`更改为`ASSERT_TRUE`并重新运行测试。这一次，它通过了，因为它正在测试正确的事情。

1.  我们接下来要做的是添加一些类型别名，以便在接下来的几个方法中使用。在编辑器中，在`empty()`方法的上面添加以下行：

```cpp
    using value_type = T;
    using reference = value_type&;
    using const_reference = const value_type&;
    using size_type = std::size_t;
    ```

1.  单击**运行**按钮重新运行测试。它们应该通过。在进行测试驱动开发时，口头禅是编写一个小测试并看到它失败，然后编写足够的代码使其通过。在这种情况下，我们实际上测试了我们是否正确获取了别名的定义，因为编译失败是一种测试失败的形式。我们现在准备添加 push 函数。

1.  在编辑器中，通过在**empty()**方法的下面添加以下代码来更改**Stack.hpp**：

```cpp
    void push(const value_type& value)
    {
        m_stack.push_back(value);
    }
    ```

1.  在文件顶部，将`EXERCISE4_STEP`更改为值`15`。单击**PushOntoStackNotEmpty**，在**StackTests.cpp**中证明了 push 对使堆栈不再为空做了一些事情。我们需要添加更多方法来确保它已经完成了预期的工作。

1.  在编辑器中，更改`push()`方法并将`EXERCISE4_STEP`更改为值`16`：

```cpp
    size_type size() const
    {
        return m_stack.size();
    }
    ```

1.  单击**运行**按钮运行测试。现在应该有三个通过的测试。

1.  在编辑器中，更改`push()`方法并将`EXERCISE4_STEP`更改为`18`的值：

```cpp
    void pop()
    {
        m_stack.pop_back();
    }
    ```

1.  单击**运行**按钮运行测试。现在应该有四个通过的测试。

1.  在编辑器中，更改`pop()`方法并将`EXERCISE4_STEP`更改为`20`的值：

```cpp
    reference top()
    {
        m_stack.back();
    }
    const_reference top() const
    {
        m_stack.back();
    }
    ```

1.  单击**运行**按钮运行测试。现在有五个通过的测试，我们已经实现了一个堆栈。

1.  从启动配置下拉菜单中，选择**L2BExercise4**，然后单击**运行**按钮。练习 4 将运行并产生类似以下输出：

![图 2B.34：练习 4 输出](img/C14583_02B_34.jpg)

###### 图 2B.34：练习 4 输出

检查现在在`std::stack`模板中的代码，它带有两个参数，第二个参数定义要使用的容器 - vector 可以是第一个。检查**StackTests.cpp**中的测试。测试应该被命名以指示它们的测试目标，并且它们应该专注于做到这一点。

### 活动 1：开发一个通用的“contains”模板函数

编程语言 Python 有一个称为“in”的成员运算符，可以用于任何序列，即列表、序列、集合、字符串等。尽管 C++有 100 多种算法，但它没有相应的方法来实现相同的功能。C++ 20 在`std::set`上引入了`contains()`方法，但这对我们来说还不够。我们需要创建一个`contains()`模板函数，它可以与`std::set`、`std::string`、`std::vector`和任何提供迭代器的其他容器一起使用。这是通过能够在其上调用 end()来确定的。我们的目标是获得最佳性能，因此我们将在任何具有`find()`成员方法的容器上调用它（这将是最有效的），否则将退回到在容器上使用`std::end()`。我们还需要将`std::string()`区别对待，因为它的`find()`方法返回一个特殊值。

我们可以使用通用模板和两个特化来实现这一点，但是这个活动正在使用 SFINAE 和 if constexpr 的技术来使其工作。此外，这个模板必须只能在支持`end(C)`的类上工作。按照以下步骤实现这个活动：

1.  从**Lesson2B/Activity01**文件夹加载准备好的项目。

1.  定义辅助模板函数和类来检测 std:string 情况，使用`npos`成员。

1.  定义辅助模板函数和类，以检测类是否具有`find()`方法。

1.  定义包含模板函数，使用 constexpr 来在三种实现中选择一种 - 字符串情况、具有 find 方法的情况或一般情况。

在实现了上述步骤之后，预期输出应如下所示：

![图 2B.35：包含成功实现的输出](img/C14583_02B_35.jpg)

###### 图 2B.35：包含成功实现的输出

#### 注意

此活动的解决方案可在第 653 页找到。

## 总结

在本章中，我们学习了接口、继承和多态，这扩展了我们对类型的操作技能。我们首次尝试了 C++模板的泛型编程，并接触了语言从 C++标准库（包括 STL）中免费提供给我们的内容。我们探索了 C++的一个功能，即模板类型推断，它在使用模板时使我们的生活更加轻松。然后我们进一步学习了如何使用 SFINAE 和 if constexpr 控制编译器包含的模板部分。这些构成了我们进入 C++之旅的基石。在下一章中，我们将重新讨论堆栈和堆，并了解异常是什么，发生了什么，以及何时发生。我们还将学习如何在异常发生时保护我们的程序免受资源损失。
