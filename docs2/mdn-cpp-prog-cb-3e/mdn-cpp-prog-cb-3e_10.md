

# 实现模式和惯用语

设计模式是一般可重用的解决方案，可以应用于软件开发中出现的常见问题。惯用法是模式、算法或结构代码的一种或多种编程语言的方式。关于设计模式已经编写了大量的书籍。本章的目的不是重复它们，而是展示如何实现几个有用的模式和惯用法，重点关注可读性、性能和健壮性，从现代 C++的角度出发。

本章节包含的食谱如下：

+   避免在工厂模式中重复使用`if-else`语句

+   实现 pimpl 惯用法

+   实现命名参数习语

+   使用非虚拟接口惯用语分离接口和实现

+   使用律师-客户习语处理友谊

+   奇异重复模板模式下的静态多态

+   使用混入（mixins）向类添加功能

+   使用类型擦除惯用语泛型处理无关类型

+   实现线程安全的单例

本章的第一个菜谱介绍了一种避免重复`if-else`语句的简单机制。让我们来探究这个机制是如何工作的。

# 避免在工厂模式中重复使用 if-else 语句

通常情况下，我们会陷入编写重复的`if...else`语句（或等效的`switch`语句），这些语句执行类似的事情，通常变化很小，而且常常是通过复制粘贴并做些小改动来完成的。随着可选条件的数量增加，代码既难以阅读也难以维护。重复的`if...else`语句可以用各种技术来替换，例如多态。在这个菜谱中，我们将看到如何使用函数映射来避免在工厂模式（工厂是一个用于创建其他对象的函数或对象）中使用`if...else`语句。

## 准备就绪

在这个菜谱中，我们将考虑以下问题：构建一个能够处理各种格式图像文件的系统，例如位图、PNG、JPG 等等。显然，这些细节超出了本菜谱的范围；我们关注的部分是创建处理各种图像格式的对象。为此，我们将考虑以下类的层次结构：

```cpp
class Image {};
class BitmapImage : public Image {};
class PngImage    : public Image {};
class JpgImage    : public Image {}; 
```

另一方面，我们将定义一个用于工厂类的接口，该接口可以创建上述类的实例，以及使用`if...else`语句的典型实现：

```cpp
struct IImageFactory
{
  virtual std::unique_ptr<Image> Create(std::string_view type) = 0;
};
struct ImageFactory : public IImageFactory
{
  std::unique_ptr<Image> 
 Create(std::string_view type) override
 {
    if (type == "bmp")
      return std::make_unique<BitmapImage>();
    else if (type == "png")
      return std::make_unique<PngImage>();
    else if (type == "jpg")
      return std::make_unique<JpgImage>();
    return nullptr;
  }
}; 
```

本食谱的目标是查看如何重构此实现以避免重复的`if...else`语句。

## 如何做到这一点...

执行以下步骤以重构前面展示的工厂，避免使用`if...else`语句：

1.  实现工厂接口：

    ```cpp
    struct ImageFactory : public IImageFactory
    {
      std::unique_ptr<Image> Create(std::string_view type) override
     { 
        // continued with 2\. and 3.
      }
    }; 
    ```

1.  定义一个映射，其中键是要创建的对象类型，值是创建对象的函数：

    ```cpp
    static std::map<
      std::string,
      std::function<std::unique_ptr<Image>()>> mapping
    {
      { "bmp", []() {return std::make_unique<BitmapImage>(); } },
      { "png", []() {return std::make_unique<PngImage>(); } },
      { "jpg", []() {return std::make_unique<JpgImage>(); } }
    }; 
    ```

1.  要创建一个对象，请在映射中查找对象类型，如果找到，则使用关联的函数来创建该类型的新实例：

    ```cpp
    auto it = mapping.find(type.data());
    if (it != mapping.end())
      return it->second();
    return nullptr; 
    ```

## 它是如何工作的...

第一实现中的重复`if...else`语句非常相似——它们检查`type`参数的值并创建适当的`Image`类的实例。如果检查的参数是整型（例如枚举类型），`if...else`语句的序列也可以写成`switch`语句的形式。这段代码可以这样使用：

```cpp
auto factory = ImageFactory{};
auto image = factory.Create("png"); 
```

无论实现是使用`if...else`语句还是`switch`，重构以避免重复检查相对简单。在重构的代码中，我们使用了一个键类型为`std::string`的映射，表示类型，即图像格式的名称。值是一个`std::function<std::unique_ptr<Image>()>`。这是一个用于无参数且返回`std::unique_ptr<Image>`（派生类的`unique_ptr`隐式转换为基类的`unique_ptr`）的函数包装器。

现在我们有了创建对象的函数映射，工厂的实际实现就简单多了；在映射中检查要创建的对象的类型，如果存在，则使用映射中关联的值作为创建对象的实际函数，如果映射中不存在该对象类型，则返回`nullptr`。

这种重构对客户端代码来说是透明的，因为客户端使用工厂的方式没有变化。另一方面，这种方法确实需要更多的内存来处理静态映射，对于某些应用程序类别，如物联网（IoT），这可能是一个重要的方面。这里提供的示例相对简单，因为目的是演示这个概念。在实际代码中，可能需要以不同的方式创建对象，例如使用不同数量的参数和不同类型的参数。然而，这并不特定于重构的实现，使用`if...else`/`switch`语句的解决方案也需要考虑这一点。因此，在实践中，使用`if...else`语句解决问题的解决方案也应该适用于映射。

## 还有更多...

在先前的实现中，映射是一个属于虚拟函数的局部静态变量，但它也可以是类的成员，甚至是一个全局变量。在下面的实现中，映射被定义为类的静态成员。对象不是基于格式名称创建的，而是基于类型信息创建的，这是由`typeid`运算符返回的：

```cpp
struct IImageFactoryByType
{
  virtual std::unique_ptr<Image> Create(
    std::type_info const & type) = 0;
};
struct ImageFactoryByType : public IImageFactoryByType
{
  std::unique_ptr<Image> Create(std::type_info const & type) 
 override
 {
    auto it = mapping.find(&type);
    if (it != mapping.end())
      return it->second();
    return nullptr;
  }
private:
  static std::map<
    std::type_info const *,
    std::function<std::unique_ptr<Image>()>> mapping;
};
std::map<
  std::type_info const *,
  std::function<std::unique_ptr<Image>()>> ImageFactoryByType::mapping
{
  {&typeid(BitmapImage),[](){
      return std::make_unique<BitmapImage>();}},
  {&typeid(PngImage),   [](){
      return std::make_unique<PngImage>();}},
  {&typeid(JpgImage),   [](){
      return std::make_unique<JpgImage>();}}
}; 
```

在这种情况下，客户端代码略有不同，因为我们不是传递一个表示要创建的类型名称，例如 PNG，而是传递`typeid`运算符返回的值，例如`typeid(PngImage)`：

```cpp
auto factory = ImageFactoryByType{};
auto movie = factory.Create(typeid(PngImage)); 
```

这种替代方案可以说是更健壮的，因为映射键不是字符串，这可能会更容易出错。本食谱提出了一种模式作为解决常见问题的方案，而不是实际的实现。正如大多数模式的情况一样，它们有不同的实现方式，取决于你选择最适合每个上下文的那一种。

## 参见

+   *实现 pimpl 习语*，学习一种能够将实现细节与接口分离的技术

+   *第九章，使用 unique_ptr 唯一拥有内存资源*，了解`std::unique_ptr`类，它代表一个智能指针，它拥有并管理在堆上分配的另一个对象或对象数组

# 实现 pimpl 习语

**pimpl**代表**指向实现**（也称为**查理猫习语**或**编译器防火墙习语**）是一种不透明的指针技术，它能够将实现细节与接口分离。这种技术的优点是它允许在不修改接口的情况下更改实现，因此避免了需要重新编译使用该接口的代码。这使得使用 pimpl 习语的库在实现细节更改时，其 ABIs 与旧版本向后兼容。在本食谱中，我们将看到如何使用现代 C++特性实现 pimpl 习语。

**ABI**这个术语代表**应用程序二进制接口**，指的是两个二进制模块之间的接口。通常，其中一个模块是库或操作系统，另一个是用户执行的程序。

## 准备工作

读者应熟悉智能指针和`std::string_view`，这两者都在本书的前几章中讨论过。

为了以实际的方式演示 pimpl 习语，我们将考虑以下类，然后我们将根据 pimpl 模式对其进行重构：

```cpp
class control
{
  std::string text;
  int width = 0;
  int height = 0;
  bool visible = true;
  void draw()
 {
    std::cout 
      << "control " << '\n'
      << " visible: " << std::boolalpha << visible << 
         std::noboolalpha << '\n'
      << " size: " << width << ", " << height << '\n'
      << " text: " << text << '\n';
  }
public:
  void set_text(std::string_view t)
 {
    text = t.data();
    draw();
  }
  void resize(int const w, int const h)
 {
    width = w;
    height = h;
    draw();
  }
  void show() 
 { 
    visible = true; 
    draw();
  }
  void hide() 
 { 
    visible = false; 
    draw();
  }
}; 
```

这个类表示具有文本、大小和可见性等属性的控件。每次这些属性发生变化时，控件都会重新绘制。在这个模拟实现中，绘制意味着将属性的值打印到控制台。

## 如何做...

按照以下步骤实现 pimpl 习语，以下以重构前面展示的`control`类为例：

1.  将所有私有成员，包括数据和函数，放入一个单独的类中。我们将这个类称为**pimpl 类**，而原始类称为**公共类**。

1.  在公共类的头文件中，对 pimpl 类进行前置声明：

    ```cpp
    // in control.h
    class control_pimpl; 
    ```

1.  在公共类定义中，使用`unique_ptr`声明对 pimpl 类的指针。这应该是类的唯一私有数据成员：

    ```cpp
    class control
    {
      std::unique_ptr<control_pimpl, void(*)(control_pimpl*)> pimpl;
      public:
        control();
        void set_text(std::string_view text);
        void resize(int const w, int const h);
        void show();
        void hide();
    }; 
    ```

1.  将 pimpl 类定义放在公共类的源文件中。pimpl 类反映了公共类的公共接口：

    ```cpp
    // in control.cpp
    class control_pimpl
    {
      std::string text;
      int width = 0;
      int height = 0;
      bool visible = true;
      void draw()
     {
        std::cout
          << "control " << '\n'
          << " visible: " << std::boolalpha << visible 
          << std::noboolalpha << '\n'
          << " size: " << width << ", " << height << '\n'
          << " text: " << text << '\n';
      }
    public:
      void set_text(std::string_view t)
     {
        text = t.data();
        draw();
      }
      void resize(int const w, int const h)
     {
        width = w;
        height = h;
        draw();
      }
      void show()
     {
        visible = true;
        draw();
      }
      void hide()
     {
        visible = false;
        draw();
      }
    }; 
    ```

1.  pimpl 类在公共类的构造函数中被实例化：

    ```cpp
    control::control() :
      pimpl(new control_pimpl(),
            [](control_pimpl* pimpl) {delete pimpl; })
    {} 
    ```

1.  公共类成员函数调用 pimpl 类的相应成员函数：

    ```cpp
    void control::set_text(std::string_view text)
    {
      pimpl->set_text(text);
    }
    void control::resize(int const w, int const h)
    {
      pimpl->resize(w, h);
    }
    void control::show()
    {
      pimpl->show();
    }
    void control::hide()
    {
      pimpl->hide();
    } 
    ```

## 它是如何工作的...

pimpl 习语允许隐藏类内部实现，从而为库或模块的客户端提供以下好处：

+   对于其客户端可见的类，提供一个干净的接口。

+   内部实现的变化不会影响公共接口，这使库的新版本（当公共接口保持不变时）具有二进制向后兼容性。

+   当内部实现发生变化时，使用这种习语的类的客户端无需重新编译。这导致构建时间更短。

+   头文件不需要包含私有实现中使用的类型和函数的头文件。这同样导致构建时间更短。

提到的上述好处并非免费获得；也存在一些需要提到的缺点：

+   需要编写和维护的代码更多。

+   代码的可读性可能较低，因为存在一定程度的间接引用，并且所有实现细节都需要在其他文件中查找。在本例中，pimpl 类定义在公共类的源文件中提供，但在实践中，它可能位于单独的文件中。

+   由于从公共类到 pimpl 类的间接引用级别，存在轻微的运行时开销，但在实践中，这很少是显著的。

+   这种方法不适用于私有和受保护的成员，因为这些成员必须对派生类可用。

+   这种方法不适用于私有虚拟函数，这些函数必须出现在类中，要么是因为它们覆盖了基类的函数，要么是因为它们必须对派生类中的覆盖可用。

作为经验法则，在实现 pimpl 习语时，始终将所有私有成员数据和函数（除了虚拟函数外）放在 pimpl 类中，并将受保护的成员数据和函数以及所有私有虚拟函数留在公共类中。

在本例中，`control_pimpl`类基本上与原始的`control`类相同。在实践中，当类更大，具有虚拟函数和受保护的成员以及函数和数据时，pimpl 类不是类未进行 pimpl 化时的完整等价物。此外，在实践中，pimpl 类可能需要一个指向公共类的指针，以便调用未移动到 pimpl 类的成员。

关于重构后的`control`类的实现，`control_pimpl`对象的指针由`unique_ptr`管理。在声明此指针时，我们使用了自定义的删除器：

```cpp
std::unique_ptr<control_pimpl, void(*)(control_pimpl*)> pimpl; 
```

原因在于`control`类在`control_pimpl`类型仍然不完整（即在头文件中）的地方被编译器隐式定义了析构函数。这会导致`unique_ptr`出错，因为`unique_ptr`不能删除一个不完整类型。这个问题可以通过两种方式解决：

+   为`control`类提供一个用户定义的析构函数，该析构函数在`control_pimpl`类的完整定义可用后显式实现（即使声明为`default`）。

+   为`unique_ptr`提供一个自定义的删除器，就像在这个例子中所做的那样。

## 还有更多...

原始的`control`类既可复制也可移动：

```cpp
control c;
c.resize(100, 20);
c.set_text("sample");
c.hide();
control c2 = c;             // copy
c2.show();
control c3 = std::move(c2); // move
c3.hide(); 
```

重新设计的`control`类仅可移动，不可复制。以下代码展示了实现既可复制也可移动的`control`类的示例：

```cpp
class control_copyable
{
  std::unique_ptr<control_pimpl, void(*)(control_pimpl*)> pimpl;
public:
  control_copyable();
  control_copyable(control_copyable && op) noexcept;
  control_copyable& operator=(control_copyable && op) noexcept;
  control_copyable(const control_copyable& op);
  control_copyable& operator=(const control_copyable& op);
  void set_text(std::string_view text);
  void resize(int const w, int const h);
  void show();
  void hide();
};
control_copyable::control_copyable() :
  pimpl(new control_pimpl(),
        [](control_pimpl* pimpl) {delete pimpl; })
{}
control_copyable::control_copyable(control_copyable &&) 
   noexcept = default;
control_copyable& control_copyable::operator=(control_copyable &&) 
   noexcept = default;
control_copyable::control_copyable(const control_copyable& op)
   : pimpl(new control_pimpl(*op.pimpl),
           [](control_pimpl* pimpl) {delete pimpl; })
{}
control_copyable& control_copyable::operator=(
   const control_copyable& op) 
{
  if (this != &op) 
  {
    pimpl = std::unique_ptr<control_pimpl,void(*)(control_pimpl*)>(
               new control_pimpl(*op.pimpl),
               [](control_pimpl* pimpl) {delete pimpl; });
  }
  return *this;
}
// the other member functions 
```

`control_copyable`类既可复制也可移动，但为了使其如此，我们提供了复制构造函数和复制赋值运算符，以及移动构造函数和移动赋值运算符。后两者可以省略，但前两者被显式实现，以便从被复制的对象中创建一个新的`control_pimpl`对象。

## 参见

+   *第九章*，*使用`unique_ptr`唯一拥有内存资源*，了解`std::unique_ptr`类，它表示一个智能指针，它拥有并管理在堆上分配的另一个对象或对象数组

# 实现命名参数惯例

C++只支持位置参数，这意味着参数是根据参数的位置传递给函数的。其他语言也支持命名参数——即在调用时指定参数名称并调用参数。这对于具有默认值的参数特别有用。一个函数可能有具有默认值的参数，尽管它们总是出现在所有非默认参数之后。

然而，如果您只想为一些默认参数提供值，没有提供在函数参数列表中位于它们之前的参数的参数，就无法做到这一点。

一种称为**命名参数惯例**的技术提供了一种模拟命名参数并帮助解决这个问题的方法。我们将在本食谱中探讨这项技术。

## 准备工作

为了说明命名参数惯例，我们将使用以下代码片段中的`control`类：

```cpp
class control
{
  int id_;
  std::string text_;
  int width_;
  int height_;
  bool visible_;
public:
  control(
    int const id,
    std::string_view text = "",
    int const width = 0,
    int const height = 0,
    bool const visible = false):
      id_(id), text_(text), 
      width_(width), height_(height), 
      visible_(visible)
  {}
}; 
```

`control`类表示一个视觉控件，如按钮或输入，具有数值标识符、文本、大小和可见性等属性。这些属性被提供给构造函数，除了 ID 之外，所有其他属性都有默认值。实际上，此类会有更多属性，如文本画笔、背景画笔、边框样式、字体大小、字体家族等。

## 如何做到这一点...

要为函数实现命名参数惯例（通常具有许多默认参数），请执行以下操作：

1.  创建一个类来封装函数的参数：

    ```cpp
    class control_properties
    {
      int id_;
      std::string text_;
      int width_ = 0;
      int height_ = 0;
      bool visible_ = false;
    }; 
    ```

1.  需要访问这些属性的类或函数可以声明为`friend`以避免编写 getter：

    ```cpp
    friend class control; 
    ```

1.  原始函数的每个没有默认值的定位参数都应该成为没有默认值的定位参数，在类的构造函数中：

    ```cpp
    public:
      control_properties(int const id) :id_(id)
      {} 
    ```

1.  对于原始函数的每个具有默认值的定位参数，应该有一个具有相同名称的函数，该函数在内部设置值并返回对类的引用：

    ```cpp
    public:
      control_properties& text(std::string_view t) 
     { text_ = t.data(); return *this; }
      control_properties& width(int const w) 
     { width_ = w; return *this; }
      control_properties& height(int const h) 
     { height_ = h; return *this; }
      control_properties& visible(bool const v) 
     { visible_ = v; return *this; } 
    ```

1.  原始函数应该被修改，或者提供一个重载，以接受来自新类的新参数，从该类中读取属性值：

    ```cpp
    control(control_properties const & cp):
      id_(cp.id_), 
      text_(cp.text_),
      width_(cp.width_), 
      height_(cp.height_),
      visible_(cp.visible_)
    {} 
    ```

如果我们将所有这些放在一起，结果如下：

```cpp
class control;
class control_properties
{
  int id_;
  std::string text_;
  int width_ = 0;
  int height_ = 0;
  bool visible_ = false;
  friend class control;
public:
  control_properties(int const id) :id_(id)
  {}
  control_properties& text(std::string_view t) 
 { text_ = t.data(); return *this; }
  control_properties& width(int const w) 
 { width_ = w; return *this; }
  control_properties& height(int const h) 
 { height_ = h; return *this; }
  control_properties& visible(bool const v) 
 { visible_ = v; return *this; }
};
class control
{
  int         id_;
  std::string text_;
  int         width_;
  int         height_;
  bool        visible_;
public:
  control(control_properties const & cp):
    id_(cp.id_), 
    text_(cp.text_),
    width_(cp.width_), 
    height_(cp.height_),
    visible_(cp.visible_)
  {}
}; 
```

## 它是如何工作的...

初始的`control`类有一个带有许多参数的构造函数。在实际代码中，你可以找到类似这样的例子，其中参数的数量要高得多。一个可能的解决方案，通常在实践中找到，是将常见的布尔类型属性分组在位标志中，可以作为一个单独的整型参数一起传递（一个例子可以是控制器的边框样式，它定义了边框应该可见的位置：顶部、底部、左侧、右侧，或这些四个位置的任意组合）。使用初始实现创建`control`对象的方式如下：

```cpp
control c(1044, "sample", 100, 20, true); 
```

命名参数习语的优势在于，它允许你使用名称指定你想要的参数值，顺序不限，这比固定的定位顺序更加直观。

虽然没有单一的策略来实现习语，但本食谱中的示例相当典型。`control`类的属性，作为构造函数中的参数提供，已被放入一个单独的类中，称为`control_properties`，该类将`control`类声明为友元类，以允许它访问其私有数据成员而不提供 getter。这有一个副作用，即限制了`control_properties`在`control`类之外的用途。`control`类构造函数的非可选参数也是`control_properties`构造函数的非可选参数。对于所有其他具有默认值的参数，`control_properties`类定义了一个具有相关名称的函数，该函数简单地设置数据成员为提供的参数，然后返回对`control_properties`的引用。这使得客户端可以以任何顺序链式调用这些函数。

控制类构造函数已被替换为一个新的构造函数，它只有一个参数，即对`control_properties`对象的常量引用，其数据成员被复制到`control`对象的数据成员中。

以这种方式实现命名参数习语的`control`对象创建，如下代码片段所示：

```cpp
control c(control_properties(1044)
          .visible(true)
          .height(20)
          .width(100)); 
```

## 参见

+   *使用非虚接口惯用法来分离接口和实现*，探索一个通过使（公共）接口非虚和虚函数私有来促进接口和实现关注点分离的惯用法

+   *使用律师-客户惯用法处理友谊*，了解一个简单的机制来限制朋友对类中指定、私有成员的访问

# 使用非虚接口惯用法来分离接口和实现

虚函数通过允许派生类修改从基类继承的实现，为类提供了特殊化的点。当一个派生类对象通过基类指针或引用来处理时，对重写的虚函数的调用最终会调用派生类中的重写实现。另一方面，定制是实现细节，良好的设计将接口与实现分离。

Herb Sutter 在 *C/C++ Users Journal* 关于虚函数的文章中提出的 **非虚接口惯用法**，通过使（公共）接口非虚和虚函数私有，促进了接口和实现的关注点分离。

公共虚接口阻止类在其接口上强制执行前条件和后条件。期望基类实例的用户不能保证公共虚方法会提供预期的行为，因为它可以在派生类中被重写。这个惯用法有助于强制执行接口的承诺合同。

## 准备工作

读者应该熟悉与虚函数相关的方面，例如定义和重写虚函数、抽象类和纯指定符。

## 如何做到这一点...

实现这个惯用法需要遵循几个简单的设计准则，这些准则由 Herb Sutter 在 *C/C++ Users Journal*，19(9)，2001 年 9 月提出：

1.  将（公共）接口设为非虚。

1.  将虚函数设为私有。

1.  只有当基类实现必须从派生类中调用时，才将虚函数设为保护。

1.  将基类析构函数设为公共和虚的或保护和非虚的。

以下是一个简单的控件层次结构的示例，遵循所有这四个准则：

```cpp
class control
{
private:
  virtual void paint() = 0;
protected:
  virtual void erase_background() 
 {
    std::cout << "erasing control background..." << '\n';
  }
public:
  void draw()
 {
    erase_background();
    paint();
  }
  virtual ~control() {}
};
class button : public control
{
private:
  virtual void paint() override
 {
    std::cout << "painting button..." << '\n';
  }
protected:
  virtual void erase_background() override
 {
    control::erase_background();
    std::cout << "erasing button background..." << '\n';
  }
};
class checkbox : public button
{
private:
  virtual void paint() override
 {
    std::cout << "painting checkbox..." << '\n';
  }
protected:
  virtual void erase_background() override
 {
    button::erase_background();
    std::cout << "erasing checkbox background..." << '\n';
  }
}; 
```

## 它是如何工作的...

NVI 惯用法使用 **模板方法** 设计模式，允许派生类定制基类功能（即算法）的部分（即步骤）。这是通过将整体算法拆分为更小的部分来实现的，每个部分都由一个虚函数实现。基类可以提供或不需要默认实现，派生类可以覆盖它们，同时保持算法的整体结构和意义。

NVI 习语的核心理念是虚拟函数不应该公开；它们应该是私有或受保护的，以防基类实现可以从派生类中调用。类的接口，其客户端可以访问的公共部分，应该仅由非虚拟函数组成。这提供了几个优点：

+   它将接口与不再暴露给客户端的实现细节分离。

+   它使得在不改变公共接口且不需要修改客户端代码的情况下更改实现细节成为可能，因此使基类更加健壮。

+   它允许一个类对其接口拥有完全控制权。如果公共接口包含虚拟方法，派生类可以改变承诺的功能，因此，类不能确保其前置条件和后置条件。当没有虚拟方法（除了析构函数）可供其客户端访问时，类可以在其接口上强制执行前置条件和后置条件。

对于这个习语，需要特别提及类的析构函数。通常强调基类析构函数应该是虚拟的，这样对象就可以通过基类指针或引用进行多态删除。当析构函数不是虚拟的时，进行多态删除对象会导致未定义的行为。然而，并非所有基类都旨在进行多态删除。对于这些特定情况，基类析构函数不应该虚拟，但也应该不是公共的，而是受保护的。

上一节中的例子定义了一个表示视觉控件的类层次结构：

+   `control` 是基类，但存在派生类，如 `button` 和 `checkbox`，它们是按钮类型，因此从这个类派生出来。

+   `control` 类定义的唯一功能是绘制控件。`draw()` 方法是非虚拟的，但它调用了两个虚拟方法，`erase_background()` 和 `paint()`，以实现绘制控件的两个阶段。

+   `erase_background()` 是一个受保护的虚拟方法，因为派生类需要在它们自己的实现中调用它。

+   `paint()` 是一个私有的纯虚拟方法。派生类必须实现它，但不应该调用基类实现。

+   控件类的析构函数是公共的且虚拟的，因为预期对象将通过多态删除。

下面展示了使用这些类的示例。这些类的实例由基类智能指针管理：

```cpp
std::vector<std::unique_ptr<control>> controls;
controls.emplace_back(std::make_unique<button>());
controls.emplace_back(std::make_unique<checkbox>());
for (auto& c : controls)
  c->draw(); 
```

该程序的输出如下：

```cpp
erasing control background...
erasing button background...
painting button...
erasing control background...
erasing button background...
erasing checkbox background...
painting checkbox...
destroying button...
destroying control...
destroying checkbox...
destroying button...
destroying control... 
```

NVI 习语在公共函数调用实际实现的非公共虚拟函数时引入了一层间接性。在先前的例子中，`draw()` 方法调用了几个其他函数，但在许多情况下，可能只需要一个调用：

```cpp
class control
{
protected:
  virtual void initialize_impl()
 {
    std::cout << "initializing control..." << '\n';
  }
public:
  void initialize()
 {
    initialize_impl();
  }
};
class button : public control
{
protected:
  virtual void initialize_impl()
 {
    control::initialize_impl();
    std::cout << "initializing button..." << '\n';
  }
}; 
```

在这个例子中，类`control`有一个额外的名为`initialize()`的方法（为了保持简单，没有显示类的前置内容），它调用一个单独的非公共虚拟方法`initialize_impl()`，该方法在每个派生类中实现不同。这不会产生太多开销——如果有的话——因为像这样的简单函数很可能被编译器内联。

## 参见

+   *第一章*，*使用 override 和 final 指定虚拟方法*，了解如何指定一个虚拟函数覆盖另一个虚拟函数，以及如何指定在派生类中不能覆盖虚拟函数

# 使用律师-客户习语处理友元关系

使用友元声明授予函数和类对类非公共部分的访问权限通常被视为设计不佳的标志，因为友元关系破坏了封装性，并使类和函数之间产生了联系。无论友元是类还是函数，它们都可以访问类的所有私有成员，尽管它们可能只需要访问其中的一部分。

**律师-客户习语**提供了一个简单的机制，以限制友元对类中仅指定的私有成员的访问。

## 准备工作

为了演示如何实现这个习语，我们将考虑以下类：`Client`，它具有一些私有成员数据和函数（在这里，公共接口并不重要），以及`Friend`，它应该只访问私有细节的一部分，例如`data1`和`action1()`，但它可以访问一切：

```cpp
class Client
{
  int data_1;
  int data_2;
  void action1() {}
  void action2() {}
  friend class Friend;
public:
  // public interface
};
class Friend
{
public:
  void access_client_data(Client& c)
 {
    c.action1();
    c.action2();
    auto d1 = c.data_1;
    auto d2 = c.data_1;
  }
}; 
```

要理解这个习语，你必须熟悉 C++语言中如何声明友元关系以及它是如何工作的。

## 如何做...

采取以下步骤以限制友元对您需要访问的类中私有成员的访问：

1.  在`Client`类中，该类将其所有私有成员对友元类提供访问权限，将友元关系声明给一个中间类，称为`Attorney`类：

    ```cpp
    class Client
    {
      int data_1;
      int data_2;
      void action1() {}
      void action2() {}
      friend class Attorney;
    public:
      // public interface
    }; 
    ```

1.  创建一个只包含私有（内联）函数的类，这些函数访问客户端的私有成员。这个中间类允许实际的友元访问其私有成员：

    ```cpp
    class Attorney
    {
      static inline void run_action1(Client& c)
     {
        c.action1();
      }
      static inline int get_data1(Client& c)
     {
        return c.data_1;
      }
      friend class Friend;
    }; 
    ```

1.  在`Friend`类中，通过`Attorney`类间接访问`Client`类的私有成员：

    ```cpp
    class Friend
    {
    public:
      void access_client_data(Client& c)
     {
        Attorney::run_action1(c);
        auto d1 = Attorney::get_data1(c);
      }
    }; 
    ```

## 它是如何工作的...

律师-客户惯用法通过引入中间人律师来限制对客户端私有成员的访问。客户端类不是直接向使用其内部状态的人提供友元关系，而是向律师提供友元关系，律师反过来提供对客户端受限的私有数据或函数的访问。它是通过定义私有静态函数来实现的。通常，这些也是内联函数，这避免了律师类引入的间接级别导致的任何运行时开销。客户端的友元通过实际使用律师的私有成员来访问其私有成员。这种惯用法被称为**律师-客户**，因为它与律师-客户关系的方式相似，律师知道客户的所有秘密，但只向其他方透露其中的一部分。

在实践中，如果不同的友元类或函数必须访问客户端类的不同私有成员，可能需要为客户端类创建多个律师。

另一方面，友元关系是不可继承的，这意味着一个与类`B`为友元的类或函数不会与从`B`派生出的类`D`为友元。然而，`D`中重写的虚拟函数仍然可以通过指向或引用`B`的指针或引用从友元类以多态方式访问。以下是一个示例，其中从`F`调用`run()`方法会打印出`base`和`derived`：

```cpp
class B
{
  virtual void execute() { std::cout << "base" << '\n'; }
  friend class BAttorney;
};
class D : public B
{
  virtual void execute() override 
 { std::cout << "derived" << '\n'; }
};
class BAttorney
{
  static inline void execute(B& b)
 {
    b.execute();
  }
  friend class F;
};
class F
{
public:
  void run()
 {
    B b;
    BAttorney::execute(b); // prints 'base'
    D d;
    BAttorney::execute(d); // prints 'derived'
  }
};
F;
f.run(); 
```

使用设计模式总会有权衡，这个也不例外。在某些情况下，使用此模式可能会导致在开发、测试和维护方面产生过多的开销。然而，对于某些类型的应用程序，例如可扩展框架，该模式可能非常有价值。

## 参见

+   *实现 pimpl 惯用法*，学习一种能够将实现细节与接口分离的技术

# 使用奇特重复的模板模式实现静态多态

多态为我们提供了具有相同接口的多种形式的能力。虚拟函数允许派生类覆盖基类中的实现。它们是多态形式中最常见的元素，称为**运行时多态**，因为从类层次结构中调用特定虚拟函数的决定是在运行时发生的。它也称为**后期绑定**，因为函数调用与函数调用的绑定是在程序执行期间较晚发生的。与此相反的是称为**早期绑定**、**静态多态**或**编译时多态**，因为它在编译时通过函数和运算符重载发生。

另一方面，一种称为**奇特重复的模板模式**（或**CRTP**）的技术允许通过从基类模板派生类来在编译时模拟基于虚函数的运行时多态。这种技术在某些库中得到了广泛的应用，包括微软的**活动模板库**（**ATL**）和**Windows 模板库**（**WTL**）。在这个配方中，我们将探索 CRTP，了解如何实现它以及它是如何工作的。

## 准备工作

为了演示 CRTP 的工作原理，我们将回顾我们在*使用非虚拟接口习惯用法分离接口和实现*配方中实现的控制类层次结构的示例。我们将定义一组具有诸如绘制控件等功能的控制类，在我们的示例中，这是一个分为两个阶段进行的操作：擦除背景然后绘制控件。为了简单起见，在我们的实现中，这些将只打印文本到控制台的操作。

## 如何做...

为了实现奇特重复的模板模式以实现静态多态，请执行以下操作：

1.  提供一个类模板，它将代表其他应在编译时进行多态处理的类的基类。多态函数从此类调用：

    ```cpp
    template <class T>
    class control
    {
    public:
      void draw()
     {
        static_cast<T*>(this)->erase_background();
        static_cast<T*>(this)->paint();
      }
    }; 
    ```

1.  派生类使用类模板作为它们的基类；派生类也是基类的模板参数。派生类实现了从基类调用的函数：

    ```cpp
    class button : public control<button>
    {
    public:
      void erase_background()
     {
        std::cout << "erasing button background..." << '\n';
      }
      void paint()
     {
        std::cout << "painting button..." << '\n';
      }
    };
    class checkbox : public control<checkbox>
    {
    public:
      void erase_background()
     {
        std::cout << "erasing checkbox background..." 
                  << '\n';
      }
      void paint()
     {
        std::cout << "painting checkbox..." << '\n';
      }
    }; 
    ```

1.  函数模板可以通过基类模板的指针或引用来多态地处理派生类：

    ```cpp
    template <class T>
    void draw_control(control<T>& c)
    {
      c.draw();
    }
    button b;
    draw_control(b);
    checkbox c;
    draw_control(c); 
    ```

## 它是如何工作的...

虚函数可能会引起性能问题，尤其是在它们很小并且在循环中多次调用时。现代硬件使得这些情况中的大多数变得相当无关紧要，但仍然有一些应用类别，性能至关重要，任何性能提升都很重要。奇特重复的模板模式允许使用元编程在编译时模拟虚函数调用，这最终转化为函数重载。

这种模式乍一看可能相当奇怪，但它完全合法。想法是从一个基类派生一个类，该基类是一个模板类，然后传递派生类本身作为基类的类型模板参数。基类随后调用派生类函数。在我们的示例中，`control<button>::draw()`在`button`类对编译器已知之前声明。然而，`control`类是一个类模板，这意味着它仅在编译器遇到使用它的代码时实例化。在那个时刻，在这个例子中，`button`类已经定义并且对编译器已知，因此可以调用`button::erase_background()`和`button::paint()`。

要调用派生类的函数，我们首先需要获得派生类的指针。这通过`static_cast`转换完成，如`static_cast<T*>(this)->erase_background()`所示。如果需要多次这样做，可以通过提供一个执行此操作的私有函数来简化代码：

```cpp
template <class T>
class control
{
  T* derived() { return static_cast<T*>(this); }
public:
  void draw()
 {
    derived()->erase_background();
    derived()->paint();
  }
}; 
```

在使用 CRTP 时，有一些陷阱你必须注意：

+   所有从基类模板调用的派生类中的函数都必须是公共的；否则，基类特化必须声明为派生类的友元：

    ```cpp
    class button : public control<button>
    {
    private:
      friend class control<button>;
      void erase_background()
     {
        std::cout << "erasing button background..." << '\n';
      }
      void paint()
     {
        std::cout << "painting button..." << '\n';
      }
    }; 
    ```

+   在同质容器，例如`vector`或`list`中，无法存储 CRTP 类型的对象，因为每个基类都是一个独特的类型（例如`control<button>`和`control<checkbox>`）。如果这确实是必要的，那么可以使用一种变通方法来实现它。这将在下一节中进行讨论和示例。

+   当使用这种技术时，程序的大小可能会增加，因为模板的实例化方式。

## 还有更多...

当需要将实现 CRTP 类型的对象同质地存储在容器中时，必须使用一个额外的惯用用法。基类模板本身必须从另一个具有纯虚拟函数（以及虚拟公共析构函数）的类派生。为了在`control`类上说明这一点，需要以下更改：

```cpp
class controlbase
{
public:
  virtual void draw() = 0;
  virtual ~controlbase() {}
};
template <class T>
class control : public controlbase
{
public:
  virtual void draw() override
 {
    static_cast<T*>(this)->erase_background();
    static_cast<T*>(this)->paint();
  }
}; 
```

不需要对派生类，如`button`和`checkbox`，进行任何更改。然后，我们可以在容器中存储抽象类的指针，例如`std::vector`，如下所示：

```cpp
void draw_controls(std::vector<std::unique_ptr<controlbase>>& v)
{
  for (auto & c : v)
  {
    c->draw();
  }
}
std::vector<std::unique_ptr<controlbase>> v;
v.emplace_back(std::make_unique<button>());
v.emplace_back(std::make_unique<checkbox>());
draw_controls(v); 
```

## 参见

+   实现 pimpl 惯用用法，学习一种使实现细节与接口分离的技术

+   使用非虚拟接口惯用用法来分离接口和实现，以探索一种惯用用法，通过使（公共）接口非虚拟和虚拟函数私有，来促进接口和实现的关注点分离

# 向混入类添加功能

在前面的菜谱中，我们了解了一个称为“好奇地反复出现的模板模式”或简称 CRTP 的模式，以及它是如何被用来向类添加共同功能的。这并不是它的唯一用途；其他用例包括限制类型的实例化次数和实现组合模式。与这个模式相关，还有一个称为**混合**的模式。混合是设计用来向其他现有类添加功能的小类。你可能可以找到关于这个模式的文章，声称它是使用 CRTP 实现的。这是不正确的。确实，CRTP 和混合是相似的模式，两者都用于向类添加功能，但它们的结构并不相同。在 CRTP 中，基类向从它派生的类添加功能。混合类向它派生的类添加功能。因此，从某种意义上说，它是一个颠倒的 CRTP。在这个菜谱中，你将学习如何使用混合向类添加共同功能。为此，我们将检查绘制控件（如按钮和复选框）的相同示例。这将允许与 CRTP 进行良好的比较，这将帮助你更好地理解两者之间的差异（和相似之处）。

## 如何做到这一点…

要实现混合模式以向现有类添加共同功能，请按照以下步骤操作（在以下示例中，所涉及的共同功能是绘制控件的背景和内容）：

1.  考虑（可能无关的）表现出共同功能（的）类：

    ```cpp
    class button
    {
    public:
       void erase_background()
     {
          std::cout << "erasing button background..." << '\n';
       }
       void paint()
     {
          std::cout << "painting button..." << '\n';
       }
    };
    class checkbox
    {
    public:
       void erase_background()
     {
          std::cout << "erasing checkbox background..." << '\n';
       }
       void paint()
     {
          std::cout << "painting checkbox..." << '\n';
       }
    }; 
    ```

1.  创建一个从其类型模板参数派生的类模板。这个混合类定义了一些新的功能，这些功能是通过基类中现有的功能实现的：

    ```cpp
    template <typename T>
    class control : public T
    {
    public:
       void draw()
     {
          T::erase_background();
          T::paint();
       }
    }; 
    ```

1.  实例化和使用混合类的对象以利用添加的功能：

    ```cpp
    control<button> b;
    b.draw();
    control<checkbox> c;
    c.draw(); 
    ```

## 它是如何工作的…

混合是一个允许我们向现有类添加新功能的概念。在许多编程语言中，这种模式有不同的实现方式。在 C++中，混合是一个小的类，它向现有的类添加功能（而不需要对现有类进行任何修改）。为此，你需要：

+   将混合类做成模板。在我们的例子中，这是`control`类。如果只有一个类型需要扩展，那么不需要使用模板，因为没有代码重复。然而，在实践中，这通常是为了向多个类似类添加共同功能。

+   从其类型模板参数派生它，该参数应该实例化为要扩展的类型。通过重用类型模板参数类的功能来实现添加的功能。在我们的例子中，新的功能是`draw()`，它使用了`T::erase_background()`和`T::paint()`。

由于混入类是一个模板，它不能被多态处理。例如，也许您想要一个能够绘制按钮、复选框以及其他可绘制控件的函数。这个函数可以看起来如下：

```cpp
void draw_all(std::vector<???*> const & controls)
{
   for (auto& c : controls)
   {
      c->draw();
   }
} 
```

但在这个片段中`???`代表什么？我们需要一个非模板基类才能使其以多态方式工作。这样的基类可以看起来如下：

```cpp
class control_base
{
public:
   virtual ~control_base() {}
   virtual void draw() = 0;
}; 
```

混入类（`control`）还需要从该基类（`control_base`）派生，并且`draw()`函数成为一个被重写的虚函数：

```cpp
template <typename T>
class control : public control_base, public T 
{
public:
   void draw() override
 {
      T::erase_background();
      T::paint();
   }
}; 
```

这允许我们以多态方式处理控件对象，如下面的示例所示：

```cpp
void draw_all(std::vector<control_base*> const & controls)
{
   for (auto& c : controls)
   {
      c->draw();
   }
}
int main()
{
   std::vector<control_base*> controls;
   control<button> b;
   control<checkbox> c;
   draw_all({&b, &c});
} 
```

如您从本食谱和上一个食谱中可以看到，混入和 CRTP 都用于添加功能到类的相同目的。此外，它们看起来很相似，尽管实际的模式结构是不同的。

## 参见

+   *使用好奇地重复出现的模板模式进行静态多态*，要了解 CRTP，它允许通过从基类模板派生类来在编译时模拟运行时多态

# 使用类型擦除惯用语泛型处理无关类型

多态（特别是 C++中的运行时多态）允许我们以通用方式处理类的层次结构。然而，有些情况下我们想要做的是相同的，但与不继承自公共基类的类。这可能发生在我们不拥有代码或由于各种原因无法更改代码以创建层次结构时。这个过程是利用具有某些特定成员（函数或变量）的不相关类型来完成给定任务（并且只使用那些公共成员）的过程，称为**鸭子类型**。解决这个问题的一个简单方法是为我们想要以通用方式处理的每个类构建一个包装类层次结构。这有缺点，因为有很多样板代码，并且每次需要以相同方式处理新类时，都必须创建一个新的包装器。这种方法的替代方法是称为**类型擦除**的惯用语。这个术语指的是擦除了有关具体类型的信息，允许以通用方式处理不同甚至不相关的类型。在本食谱中，我们将学习这个惯用语是如何工作的。

## 准备工作

为了展示类型擦除惯用语，我们将使用以下两个类，分别代表按钮和复选框控件：

```cpp
class button
{
public:
   void erase_background()
 {
      std::cout << "erasing button background..." << '\n';
   }
   void paint()
 {
      std::cout << "painting button..." << '\n';
   }
};
class checkbox
{
public:
   void erase_background()
 {
      std::cout << "erasing checkbox background..." << '\n';
   }
   void paint()
 {
      std::cout << "painting checkbox..." << '\n';
   }
}; 
```

这些是我们之前在各种形式中看到过的相同类。它们都有`erase_background()`和`paint()`成员函数，但没有一个共同的基类；因此，它们不是属于允许我们以多态方式处理它们的层次结构的一部分。

## 如何做到这一点...

要实现类型擦除惯用语，您需要遵循以下步骤：

+   定义一个将提供擦除类型信息机制的类。对于本食谱中展示的与控件相关的示例，我们将简单地称其为`control`：

    ```cpp
    struct control
    {
    }; 
    ```

+   创建一个内部类（`control`类的内部类），该类定义了需要通用处理的类型所共有的接口。这个接口被称为**概念**；因此，我们将称这个类为`control_concept`：

    ```cpp
    struct control_concept
    {
       virtual ~control_concept() = default;
       virtual void draw() = 0;
    }; 
    ```

+   创建另一个内部类（`control` 类的内部类），它从概念类派生。然而，这将是一个类模板，其类型模板参数代表一个需要通用处理的类型。在我们的例子中，它将被替换为 `button` 和 `checkbox`。这种实现称为 **模型**，因此我们将这个类模板称为 `control_model`：

    ```cpp
    template <typename T>
    struct control_model : public control_concept
    {
       control_model(T & unit) : t(unit) {}
       void draw() override
     {
          t.erase_background();
          t.paint();
       }
    private:
       T& t;
    }; 
    ```

+   向`control`类添加一个数据成员，表示指向该概念实例的指针。在这个菜谱中，我们将使用智能指针来完成这个目的：

    ```cpp
    private:
       std::shared_ptr<control_concept> ctrl; 
    ```

+   定义`control`类的构造函数。这必须是一个函数模板，并且它必须将概念指针设置为模型的一个实例：

    ```cpp
    template <typename T>
    control(T&& obj) : 
       ctrl(std::make_shared<control_model<T>>(std::forward<T>(obj)))
    {
    } 
    ```

+   定义`control`类客户端能够调用的公共接口。在我们的示例中，这是一个用于绘制控制的函数。我们将称之为`draw()`（尽管它不必与概念中的虚拟方法同名）：

    ```cpp
    void draw()
    {
       ctrl->draw();
    } 
    ```

```cpp
struct control
{
   template <typename T>
   control(T&& obj) : 
      ctrl(std::make_shared<control_model<T>>(std::forward<T>(obj)))
   {
   }
   void draw()
 {
      ctrl->draw();
   }
   struct control_concept
   {
      virtual ~control_concept() = default;
      virtual void draw() = 0;
   };
   template <typename T>
   struct control_model : public control_concept
   {
      control_model(T& unit) : t(unit) {}
      void draw() override
 {
         t.erase_background();
         t.paint();
      }
   private:
      T& t;
   };
private:
   std::shared_ptr<control_concept> ctrl;
}; 
```

我们可以使用这个包装类来多态地处理按钮和复选框（以及类似的其它类），例如在以下代码片段中：

```cpp
void draw(std::vector<control>& controls)
{
   for (auto& c : controls)
   {
      c.draw();
   }
}
int main()
{
   checkbox cb;
   button btn;
   std::vector<control> v{control(cb), control(btn)};
   draw(v);
} 
```

## 它是如何工作的…

最基本的类型擦除形式（也许可以说是最终形式）是使用`void`指针。尽管这为在 C 语言中实现该惯用表达式提供了机制，但在 C++中应避免使用，因为它不保证类型安全。它需要从指向类型的指针转换为指向`void`的指针，然后再反过来，这很容易出错，如下面的示例所示：

```cpp
void draw_button(void* ptr)
{
   button* b = static_cast<button*>(ptr);
   if (b)
   {
      b->erase_background();
      b->paint();
   }
}
int main()
{
   button btn;
   draw_button(&btn);
   checkbox cb;
   draw_button(&cb); // runtime error
} 
draw_button() is a function that knows how to draw a button. But we can pass a pointer to anything – there will be no compile-time error or warning. However, the program will likely crash at runtime.
```

在 C++中，解决这个问题的方法是定义一个处理单个类的包装器层次结构。为此，我们可以从一个定义包装器类接口的基类开始。在我们的情况下，我们感兴趣的是绘制一个控件，因此唯一的虚方法是名为`draw()`的方法。

我们将把这个类称为`control_concept`。其定义如下：

```cpp
struct control_concept
{
   virtual ~control_concept() = default;
   virtual void draw() = 0;
}; 
```

下一步是针对可以绘制的每种控制类型推导出相应的实现（使用两个 `erase_background()` 和 `paint()` 函数）。`button` 和 `checkbox` 的包装器如下：

```cpp
struct button_wrapper : control_concept
{
   button_wrapper(button& b):btn(b)
   {}
   void draw() override
 {
      btn.erase_background();
      btn.paint();
   }
private:
   button& btn;
};
struct checkbox_wrapper : control_concept
{
   checkbox_wrapper(checkbox& cb) :cbox(cb)
   {}
   void draw() override
 {
      cbox.erase_background();
      cbox.paint();
   }
private:
   checkbox& cbox;
}; 
```

有这样的包装器层次结构，我们可以编写一个函数，通过使用指向`control_concept`（包装器层次结构的基类）的指针，以多态方式绘制控件：

```cpp
void draw(std::vector<control_concept*> const & controls)
{
   for (auto& c : controls)
      c->draw();
}
int main()
{
   checkbox cb;
   button btn;
   checkbox_wrapper cbw(cb);
   button_wrapper btnw(btn);
   std::vector<control_concept*> v{ &cbw, &btnw };
   draw(v);
} 
```

虽然这样做是可行的，`button_wrapper` 和 `control_wrapper` 几乎完全相同。因此，它们是模板化的良好候选者。下面展示了一个封装了这两个类中看到的功能的类模板：

```cpp
template <typename T>
struct control_wrapper : control_concept
{
   control_wrapper(T& b) : ctrl(b)
   {}
   void draw() override
 {
      ctrl.erase_background();
      ctrl.paint();
   }
private:
   T& ctrl;
}; 
```

客户端代码只需进行微小修改：将`button_wrapper`和`checkbox_wrapper`替换为`control_wrapper<button>`和`control_wrapper<checkbox>`，如下所示片段：

```cpp
int main()
{
   checkbox cb;
   button btn;
   control_wrapper<checkbox> cbw(cb);
   control_wrapper<button> btnw(btn);
   std::vector<control_concept*> v{ &cbw, &btnw };
   draw(v);
} 
```

我们也可以将处理这些控制类型的`draw()`自由函数移动到`control`类内部。得到的实现如下：

```cpp
struct control_collection
{
   template <typename T>
   void add_control(T&& obj) 
 {      
      ctrls.push_back(
         std::make_shared<control_model<T>>(std::forward<T>(obj)));
   }
   void draw()
 {
      for (auto& c : ctrls)
      {
         c->draw();
      }
   }
   struct control_concept
   {
      virtual ~control_concept() = default;
      virtual void draw() = 0;
   };
   template <typename T>
   struct control_model : public control_concept
   {
      control_model(T& unit) : t(unit) {}
      void draw() override
 {
         t.erase_background();
         t.paint();
      }
   private:
      T& t;
   };
private:
   std::vector<std::shared_ptr<control_concept>> ctrls;
}; 
```

这需要对客户端代码进行一些小的修改（见 *如何操作…* 部分），它将类似于以下代码片段：

```cpp
int main()
{
   checkbox cb;
   button btn;
   control_collection cc;

   cc.add_control(cb);
   cc.add_control(btn);
   cc.draw();
} 
```

尽管我们在本食谱中看到了一个简单的例子，但这个习语在现实世界的场景中也被使用，包括 C++标准库，其中它被用于实现：

+   `std::function`，这是一个多态函数包装器，允许我们存储、复制和调用可调用项：函数、函数对象、成员函数指针、成员数据指针、lambda 表达式和绑定表达式。

+   `std::any`，这是一个表示任何可复制构造类型值的容器类型。

## 参见

+   *静态多态与古怪重复出现的模板模式*，了解 CRTP，它允许通过从使用派生类参数化的基类模板派生类来在编译时模拟运行时多态

+   *通过混入（mixins）向类添加功能，* 了解如何在不更改现有类的情况下向其添加通用功能

+   *第六章*，*使用 std::any 存储任何值*，学习如何使用 C++17 的 `std::any` 类，它代表了一个任何类型单值的类型安全容器

# 实现线程安全的单例

单例模式可能是最广为人知的设计模式之一。它限制了类中单个对象的实例化，这在某些情况下是必要的，尽管很多时候单例的使用更像是一种可以避免的反模式，可以通过其他设计选择来替代。

由于单例意味着一个类的单个实例对整个程序都是可用的，因此这种独特的实例可能可以从不同的线程中访问。因此，当你实现单例时，你也应该使其线程安全。

在 C++11 之前，做这件事并不容易，双重检查锁定技术是典型的解决方案。然而，Scott Meyers 和 Andrei Alexandrescu 在一篇名为《C++与双重检查锁定之危险》的论文中表明，使用这种模式并不能保证在可移植的 C++中实现线程安全的单例。幸运的是，这种情况在 C++11 中得到了改变，这个配方展示了如何在现代 C++中编写线程安全的单例。

## 准备就绪

对于这个食谱，你需要了解静态存储持续时间、内部链接以及删除和默认函数是如何工作的。如果你还没有阅读过，并且不熟悉该模式，你应该首先阅读之前的食谱 *使用奇特重复模板模式的静态多态性*，因为我们将在本食谱中稍后使用它。

## 如何做到这一点...

要实现线程安全的单例，你应该做以下事情：

1.  定义`Singleton`类：

    ```cpp
    class Singleton
    {
    }; 
    ```

1.  将默认构造函数设置为私有：

    ```cpp
    private:
      Singleton() = default; 
    ```

1.  将复制构造函数和复制赋值运算符分别设置为`public`和`delete`：

    ```cpp
    public:
      Singleton(Singleton const &) = delete;
      Singleton& operator=(Singleton const&) = delete; 
    ```

1.  创建并返回单个实例的函数应该是静态的，并且应该返回对类类型的引用。它应该声明一个类类型的静态对象，并返回对其的引用：

    ```cpp
    public:
      static Singleton& instance()
     {
        static Singleton single;
        return single;
      } 
    ```

## 它是如何工作的...

由于单例对象不应该由用户直接创建，所有构造函数要么是私有的，要么是公共的并且`deleted`。默认构造函数是私有的且未被删除，因为类代码中必须实际创建类的实例。在这个实现中，有一个名为`instance()`的静态函数，它返回类的单个实例。

尽管大多数实现返回一个指针，但实际上返回一个引用更有意义，因为在这个函数返回 null 指针（没有对象）的情况下是没有情况的。

`instance()`方法的实现可能看起来很简单且不是线程安全的，尤其是如果你熟悉**双重检查锁定模式**（**DCLP**）。在 C++11 中，这实际上不再是必要的，因为对象具有静态存储持续时间初始化的关键细节。初始化只发生一次，即使多个线程同时尝试初始化相同的静态对象也是如此。DCLP 的责任已经从用户转移到编译器，尽管编译器可能使用另一种技术来保证结果。

来自 C++标准文档版本 N4917 的第 8.8.3 段落的以下引文定义了静态对象初始化的规则（高亮显示的部分与并发初始化相关）： 

> 块变量的动态初始化具有静态存储持续时间（6.7.5.2）或线程存储持续时间（6.7.5.3）是在控制首次通过其声明时执行的；这样的变量在其初始化完成后被认为是初始化过的。如果初始化通过抛出异常退出，则初始化未完成，因此它将在下一次控制进入声明时再次尝试。**如果在变量初始化的同时控制并发进入声明，则并发执行应等待初始化完成。**
> 
> [注 2：符合规范的实现不能在初始化器的执行过程中引入任何死锁。死锁可能仍然由程序逻辑引起；实现只需避免由于自己的同步操作引起的死锁。—结束注]
> 
> 如果在变量初始化过程中控制递归地重新进入声明，则行为是未定义的。

静态局部对象具有静态存储持续时间，但它仅在首次使用时（在第一次调用 `instance()` 方法时）实例化。程序退出时对象将被释放。作为旁注，返回指针而不是引用的唯一可能优势是在程序退出之前某个时刻删除此单个实例，然后可能重新创建它。这再次没有太多意义，因为它与类单例、全局实例的概念相冲突，该实例可以从程序的任何地方访问。

## 更多内容...

在较大的代码库中，可能存在需要多个单例类型的情况。为了避免多次编写相同的模式，可以以通用方式实现它。为此，我们需要使用本章前面看到的 **好奇重复模板模式**（或 **CRTP**）。实际的单例作为类模板实现。`instance()` 方法创建并返回一个类型为模板参数的对象，这将是一个派生类：

```cpp
template <class T>
class SingletonBase
{
protected:
  SingletonBase() {}
public:
  SingletonBase(SingletonBase const &) = delete;
  SingletonBase& operator=(SingletonBase const&) = delete;
  static T& instance()
 {
    static T single;
    return single;
  }
};
class Single : public SingletonBase<Single>
{
  Single() {}
  friend class SingletonBase<Single>;
public:
  void demo() { std::cout << "demo" << '\n'; }
}; 
```

上一个部分中的 `Singleton` 类已变为 `SingletonBase` 类模板。默认构造函数不再是私有的，而是受保护的，因为它必须可以从派生类访问。在这个例子中，需要实例化单个对象的类被称为 `Single`。它的构造函数必须是私有的，但默认构造函数也必须对基类模板可用；因此，`SingletonBase<Single>` 是 `Single` 类的朋友。

## 参见

+   *使用好奇重复模板模式实现静态多态性*，了解 CRTP，它允许通过从使用派生类参数化的基类模板派生类来在编译时模拟运行时多态

+   *第三章*，*已弃用和已删除的函数*，了解在特殊成员函数上使用默认指定符以及如何使用 delete 指定符定义已删除的函数

# 在 Discord 上了解更多

加入我们社区的 Discord 空间，与作者和其他读者进行讨论：

`discord.gg/7xRaTCeEhx`

![二维码](img/QR_Code2659294082093549796.png)
