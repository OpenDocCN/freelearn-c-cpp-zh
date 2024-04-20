# 第四章：面向对象编程的细节

设计、实现和维护软件项目的难度取决于项目的复杂性。一个简单的计算器可以使用过程化方法（即过程式编程范式）编写，而使用相同方法实现银行账户管理系统将会太复杂。

C++支持**面向对象编程（OOP）**，这是一种建立在将实体分解为存在于紧密互联网中的对象的范式。想象一下现实世界中的一个简单场景，当你拿遥控器换电视频道时。至少有三个不同的对象参与了这个动作：遥控器，电视，还有最重要的，你。为了用编程语言表达现实世界的对象及其关系，我们并不强制使用类，类继承，抽象类，接口，虚函数等。提到的特性和概念使得设计和编码过程变得更加容易，因为它们允许我们以一种优雅的方式表达和分享想法，但它们并不是强制性的。正如 C++的创造者 Bjarne Stroustrup 所说，“并非每个程序都应该是面向对象的。”为了理解面向对象编程范式的高级概念和特性，我们将尝试看看幕后发生了什么。在本书中，我们将深入探讨面向对象程序的设计。理解对象及其关系的本质，然后使用它们来设计面向对象的程序，是本书的目标之一。

在本章中，我们将详细了解以下主题：

+   面向对象编程简介

+   C++对象模型

+   类关系，包括继承

+   多态

+   有用的设计模式

# 技术要求

在本章中，我们将使用带有`-std=c++2a`选项的 g++编译器来编译示例。

您可以在[`github.com/PacktPublishing/Expert-CPP`](https://github.com/PacktPublishing/Expert-CPP)找到本章的源文件。

# 理解对象

大多数时候，我们操作的是以某个名称分组的数据集合，从而形成了**抽象**。例如`is_military`，`speed`和`seats`等变量如果单独看并没有太多意义。将它们组合在`spaceship`这个名称下，改变了我们感知变量中存储的数据的方式。现在我们将许多变量打包成一个单一对象。为此，我们使用抽象；也就是说，我们从观察者的角度收集现实世界对象的各个属性。抽象是程序员工具链中的关键工具，因为它允许他们处理复杂性。C 语言引入了`struct`作为一种聚合数据的方式，如下面的代码所示：

```cpp
struct spaceship {
  bool is_military;
  int speed;
  int seats;
};
```

对于面向对象编程来说，对数据进行分组是有必要的。每组数据都被称为一个对象。

# 对象的低级细节

C++尽其所能支持与 C 语言的兼容性。虽然 C 结构体只是一种允许我们聚合数据的工具，但 C++使它们等同于类，允许它们拥有构造函数、虚函数、继承其他结构体等。`struct`和`class`之间唯一的区别是默认的可见性修饰符：结构体是`public`，类是`private`。通常使用结构体和类没有太大区别。面向对象编程需要的不仅仅是数据聚合。为了充分理解面向对象编程，让我们看看如果我们只有简单的结构体提供数据聚合而没有其他东西，我们如何将面向对象编程范式融入其中。

像亚马逊或阿里巴巴这样的电子商务市场的中心实体是`Product`，我们以以下方式表示它：

```cpp
struct Product {
  std::string name;
  double price;
  int rating;
  bool available;
};
```

如果需要，我们将向`Product`添加更多成员。`Product`类型的对象的内存布局可以像这样：

![](img/07ae95a7-5342-466b-84d1-60c5ffccb2a6.png)

声明`Product`对象在内存中占用`sizeof(Product)`的空间，而声明对象的指针或引用占用存储地址的空间（通常为 4 或 8 个字节）。请参阅以下代码块：

```cpp
Product book;
Product tshirt;
Product* ptr = &book;
Product& ref = tshirt;
```

我们可以将上述代码描述如下：

![](img/67e1849e-5135-4c27-ad35-ab9ef1638cfa.png)

让我们从`Product`对象在内存中占用的空间开始。我们可以通过总结其成员变量的大小来计算`Product`对象的大小。`boolean`变量的大小为 1 个字节。在 C++标准中没有明确规定`double`或`int`的确切大小。在 64 位机器上，`double`变量通常占用 8 个字节，`int`变量占用 4 个字节。

`std::string`的实现在标准中没有指定，因此其大小取决于库的实现。`string`存储指向字符数组的指针，但也可能存储分配的字符数，以便在调用`size()`时高效返回。`std::string`的一些实现占用 8、24 或 32 个字节的内存，但我们将在示例中坚持使用 24 个字节。总结一下，`Product`的大小如下：

```cpp
24 (std::string) + 8 (double) + 4 (int) + 1 (bool) = 37 bytes.
```

打印`Product`的大小会输出不同的值：

```cpp
std::cout << sizeof(Product);
```

它输出`40`而不是计算出的 37 个字节。冗余字节背后的原因是结构的填充，这是编译器为了优化对对象的各个成员的访问而实践的一种技术。**中央处理单元（CPU）**以固定大小的字读取内存。字的大小由 CPU 定义（通常为 32 位或 64 位）。如果数据从与字对齐的地址开始，CPU 可以一次访问数据。例如，`Product`的`boolean`数据成员需要 1 个字节的内存，可以直接放在评级成员后面。事实证明，编译器对数据进行了对齐以加快访问速度。假设字大小为 4 个字节。这意味着如果变量从可被 4 整除的地址开始，CPU 将无需冗余步骤即可访问变量。编译器会提前用额外的字节来对齐结构的成员到字边界地址。

# 对象的高级细节

我们将对象视为代表抽象结果的实体。我们已经提到了观察者的角色，即根据问题域定义对象的程序员。程序员定义这个过程代表了抽象的过程。让我们以电子商务市场及其产品为例。两个不同的程序员团队可能对同一产品有不同的看法。实现网站的团队关心对象的属性，这些属性对网站访问者：购买者至关重要。我们在`Product`结构中显示的属性主要是为网站访问者而设，比如销售价格、产品评级等。实现网站的程序员接触问题域，并验证定义`Product`对象所必需的属性。

负责实现帮助管理仓库中产品的在线工具的团队关心对象的属性，这些属性在产品放置、质量控制和装运方面至关重要。这个团队实际上不应该关心产品的**评级**甚至**价格**。这个团队主要关心产品的**重量**、**尺寸**和**状态**。以下插图显示了感兴趣的属性：

![](img/d45f90f5-f11f-41e0-8392-1f11fd254d95.png)

程序员在开始项目时应该做的第一件事是分析问题并收集需求。换句话说，他们应该熟悉*问题域*并定义*项目需求*。分析的过程导致定义对象及其类型，比如我们之前讨论的`Product`。为了从分析中得到正确的结果，我们应该以对象的方式思考，而通过以对象的方式思考，我们指的是考虑对象的三个主要属性：**状态**、**行为**和**身份**。

# 状态

每个对象都有一个状态，可能与其他对象的状态相同也可能不同。我们已经介绍了`Product`结构，它代表了一个物理（或数字）产品的抽象。`product`对象的所有成员共同代表了对象的状态。例如，`Product`包含诸如`available`之类的成员，它是一个布尔值；如果产品有库存，则等于`true`。成员变量的值定义了对象的状态。如果给对象成员分配新值，它的状态将会改变：

```cpp
Product cpp_book; // declaring the object
...
// changing the state of the object cpp_book
cpp_book.available = true;
cpp_book.rating = 5;
```

对象的状态是其所有属性和值的组合。

# 身份

身份是区分一个对象与另一个对象的特征。即使我们试图声明两个在物理上无法区分的对象，它们仍然会有不同的变量名称，也就是不同的身份：

```cpp
Product book1;
book1.rating = 4;
book1.name = "Book";
Product book2;
book2.rating = 4;
book2.name = "Book";
```

前面例子中的对象具有相同的状态，但它们的名称不同，即`book1`和`book2`。假设我们有能力以某种方式创建具有相同名称的对象，就像下面的代码所示：

```cpp
Product prod;
Product prod; // won't compile, but still "what if?"
```

如果是这样的话，它们在内存中仍然会有不同的地址：

![](img/a53517c2-6eb7-46a1-80ba-806d57324296.png)

身份是对象的基本属性，也是我们无法创建*空*对象的原因之一，比如下面的情况：

```cpp
struct Empty {};

int main() {
 Empty e;
  std::cout << sizeof(e);
}
```

前面的代码不会像预期的那样输出`0`。空对象的大小在标准中没有指定；编译器开发人员倾向于为这样的对象分配 1 个字节，尽管您可能也会遇到 4 或 8。两个或更多个`Empty`的实例在内存中应该有不同的地址，因此编译器必须确保对象至少占用 1 个字节的内存。

# 行为

在之前的例子中，我们将`5`和`4`分配给了`rating`成员变量。通过给对象分配无效的值，我们可以很容易地使事情出乎意料地出错，就像这样：

```cpp
cpp_book.rating = -12;
```

`-12`在产品评级方面是无效的，如果允许的话会使用户感到困惑。我们可以通过提供**setter**函数来控制对对象所做更改的行为：

```cpp
void set_rating(Product* p, int r) {
  if (r >= 1 && r <= 5) {
 p->rating = r;
 }
  // otherwise ignore
}
...
set_rating(&cpp_book, -12); // won't change the state
```

对象对来自其他对象的请求作出反应。请求是通过函数调用执行的，否则称为**消息**：一个对象向另一个对象传递消息。在前面的例子中，将相应的`set_rating`消息传递给`cpp_book`对象的对象代表我们调用`set_rating()`函数的对象。在这种情况下，我们假设从`main()`中调用函数，实际上`main()`并不代表任何对象。我们可以说它是全局对象，操作`main()`函数的对象，尽管在 C++中并没有这样的实体。

我们在概念上区分对象，而不是在物理上。这是以对象思考的主要观点。面向对象编程的一些概念的物理实现并不是标准化的，所以我们可以将`Product`结构命名为类，并声称`cpp_book`是`Product`的**实例**，并且它有一个名为`set_rating()`的成员函数。C++的实现几乎是一样的：它提供了语法上方便的结构（类、可见性修饰符、继承等），并将它们转换为简单的结构，例如前面例子中的`set_rating()`全局函数。现在，让我们深入了解 C++对象模型的细节。

# 模拟类

结构体允许我们将变量分组，命名它们，并创建对象。类的概念是在对象中包含相应的操作，将适用于该特定数据的数据和操作分组在一起。例如，对于`Product`类型的对象，直接在对象上调用`set_rating()`函数将是很自然的，而不是使用一个单独的接受`Product`对象指针并修改它的全局函数。然而，由于我们同意以 C 方式使用结构体，我们无法负担得起拥有成员函数。为了使用 C 结构体模拟类，我们必须声明与`Product`对象一起工作的函数作为全局函数，如下面的代码所示：

```cpp
struct Product {
  std::string name;
  double price;
  int rating;
  bool available;
};

void initialize(Product* p) {
  p->price = 0.0;
  p->rating = 0;
  p->available = false;
}

void set_name(Product* p, const std::string& name) {
  p->name = name;
}

std::string get_name(Product* p) {
  return p->name;
}

void set_price(Product* p, double price) {
  if (price < 0 || price > 9999.42) return;
  p->price = price;
}

double get_price(Product* p) {
  return p->price;
}

// code omitted for brevity
```

要将结构体用作类，我们应该按正确的顺序手动调用函数。例如，要使用具有正确初始化默认值的对象，我们必须首先调用`initialize()`函数：

```cpp
int main() {
  Product cpp_book;
 initialize(&cpp_book);
  set_name(&cpp_book, "Mastering C++ Programming");
  std::cout << "Book title is: " << get_name(&cpp_book);
  // ...
}
```

这似乎是可行的，但如果添加新类型，前面的代码将很快变成一个无组织的混乱。例如，考虑跟踪产品的`Warehouse`结构体： 

```cpp
struct Warehouse {
  Product* products;
  int capacity;
  int size;
};

void initialize_warehouse(Warehouse* w) {
  w->capacity = 1000;
  w->size = 0;
  w->products = new Product[w->capacity];
  for (int ix = 0; ix < w->capacity; ++ix) {
    initialize(&w->products[ix]); // initialize each Product object
  }
}

void set_size(int size) { ... }
// code omitted for brevity
```

首先明显的问题是函数的命名。我们不得不将`Warehouse`的初始化函数命名为`initialize_warehouse`，以避免与已声明的`Product`的`initialize()`函数发生冲突。我们可能会考虑重命名`Product`类型的函数，以避免将来可能的冲突。接下来是函数的混乱。现在，我们有一堆全局函数，随着我们添加新类型，这些函数的数量将增加。如果我们添加一些类型的层次结构，它将变得更加难以管理。

尽管编译器倾向于将类翻译为具有全局函数的结构体，正如我们之前展示的那样，但 C++和其他高级编程语言解决了这些问题以及其他未提及的问题，引入了将它们组织成层次结构的平滑机制。从概念上讲，关键字（`class`，`public`或`private`）和机制（继承和多态）是为了方便开发人员组织他们的代码，但不会使编译器的生活变得更容易。

# 使用类进行工作

在处理对象时，类使事情变得更容易。它们在面向对象编程中做了最简单必要的事情：将数据与操作数据的函数结合在一起。让我们使用类及其强大的特性重写`Product`结构体的示例：

```cpp
class Product {
public:
  Product() = default; // default constructor
  Product(const Product&); // copy constructor
  Product(Product&&); // move constructor

  Product& operator=(const Product&) = default;
  Product& operator=(Product&&) = default;
  // destructor is not declared, should be generated by the compiler
public:
  void set_name(const std::string&);
  std::string name() const;
  void set_availability(bool);
  bool available() const;
  // code omitted for brevity

private:
  std::string name_;
  double price_;
  int rating_;
  bool available_;
};

std::ostream& operator<<(std::ostream&, const Product&);
std::istream& operator>>(std::istream&, Product&);
```

类声明似乎更有组织性，尽管它公开的函数比我们用来定义类似结构体的函数更多。这是我们应该如何说明这个类的方式：

![](img/c8a27344-1681-4f83-97ea-90d8f78dda1a.png)

前面的图像有些特殊。正如你所看到的，它有组织良好的部分，在函数名称之前有标志等。这种类型的图表被称为**统一建模语言（UML）**类图。UML 是一种标准化说明类及其关系的方式。第一部分是类的名称（粗体），接下来是成员变量部分，然后是成员函数部分。函数名称前的`+`（加号）表示该函数是公共的。成员变量通常是私有的，但如果需要强调这一点，可以使用`-`（减号）。我们可以通过简单地说明类来省略所有细节，如下面的 UML 图所示：

![](img/24e54100-c4fb-4c79-af94-8085e241f878.png)

我们将在本书中使用 UML 图表，并根据需要引入新类型的图表。在处理初始化、复制、移动、默认和删除函数以及运算符重载之前，让我们先澄清一些事情。

# 从编译器的角度看待类

首先，无论与之前的类相比，类似怪物的类看起来多么庞大，编译器都会将其转换为以下代码（我们稍微修改了它以简化）：

```cpp
struct Product {
  std::string name_;
  bool available_;
  double price_;
  int rating_;
};

// we forced the compiler to generate the default constructor
void Product_constructor(Product&); 
void Product_copy_constructor(Product& this, const Product&);
void Product_move_constructor(Product& this, Product&&);
// default implementation
Product& operator=(Product& this, const Product&); 
// default implementation
Product& operator=(Product& this, Product&&); 

void Product_set_name(const std::string&);
// takes const because the method was declared as const
std::string Product_name(const Product& this); 
void Product_set_availability(Product& this, bool b);
bool Product_availability(const Product& this);

std::ostream& operator<<(std::ostream&, const Product&);
std::istream& operator>>(std::istream&, Product&);
```

基本上，编译器生成了与我们之前介绍的相同的代码，以模仿使用简单结构体来实现类行为的方式。尽管编译器在实现 C++对象模型的技术和方法上有所不同，但前面的例子是编译器开发人员实践的流行方法之一。它在访问对象成员（包括成员函数）的空间和时间效率之间取得了平衡。

接下来，我们应该考虑编译器通过增加和修改来编辑我们的代码。下面的代码声明了全局`create_apple()`函数，它创建并返回一个具有特定苹果值的`Product`对象。它还在`main()`函数中声明了一个书对象：

```cpp
Product create_apple() {
 Product apple;
  apple.set_name("Red apple");
  apple.set_price("0.2");
  apple.set_rating(5);
  apple.set_available(true);
  return apple;
}

int main() {
 Product red_apple = create_apple();
 Product book;  Product* ptr = &book;
  ptr->set_name("Alice in Wonderland");
  ptr->set_price(6.80);
  std::cout << "I'm reading " << book.name() 
            << " and I bought an apple for " << red_apple.price()
            << std::endl;
}
```

我们已经知道编译器修改类以将其转换为结构体，并将成员函数移动到全局范围，每个成员函数都以类的引用（或指针）作为其第一个参数。为了支持客户端代码中的这些修改，它还应该修改对所有对象的访问。

客户端代码是声明或使用已声明的类对象的一行或多行代码。

以下是我们假设编译器修改了前面代码的方式（我们使用了“假设”这个词，因为我们试图引入一个编译器抽象而不是特定于编译器的方法）：

```cpp
void create_apple(Product& apple) {
  Product_set_name(apple, "Red apple");
  Product_set_price(apple, 0.2);
  Product_set_rating(apple, 5);
  Product_set_available(apple, true);
  return;
}

int main() {
  Product red_apple;
 Product_constructor(red_apple);
 create_apple(red_apple);
  Product book;
 Product* ptr;
 Product_constructor(book);
 Product_set_name(*ptr, "Alice in Wonderland");
 Product_set_price(*ptr, 6.80);
  std::ostream os = operator<<(std::cout, "I'm reading ");
  os = operator<<(os, Product_name(book));
  os = operator<<(os, " and I bought an apple for ");
  os = operator<<(os, Product_price(red_apple));
  operator<<(os, std::endl);
  // destructor calls are skipped because the compiler 
  // will remove them as empty functions to optimize the code
  // Product_destructor(book);
  // Product_destructor(red_apple);
}
```

编译器还优化了对`create_apple()`函数的调用，以避免临时对象的创建。我们将在本章后面讨论编译器生成的隐式临时对象。

# 初始化和销毁

正如之前所示，对象的创建是一个两步过程：内存分配和初始化。内存分配是对象声明的结果。C++不关心变量的初始化；它分配内存（无论是自动还是手动）就完成了。实际的初始化应该由程序员完成，这就是我们首先需要构造函数的原因。

析构函数也是同样的逻辑。如果我们跳过默认构造函数或析构函数的声明，编译器应该会隐式生成它们，如果它们是空的话也会移除它们（以消除对空函数的冗余调用）。如果声明了带参数的构造函数，包括拷贝构造函数，编译器就不会生成默认构造函数。我们可以强制编译器隐式生成默认构造函数：

```cpp
class Product {
public:
 Product() = default;
  // ...
};
```

我们还可以通过使用`delete`修饰符来强制不生成编译器，如下所示：

```cpp
class Product {
public:
 Product() = delete;
  // ...
};
```

这将禁止默认初始化对象的声明，也就是说，`Product p`; 不会编译。

析构函数的调用顺序与对象声明的顺序相反，因为自动内存分配由堆栈管理，而堆栈是遵循**后进先出（LIFO）**规则的数据结构适配器。

对象初始化发生在对象创建时。销毁通常发生在对象不再可访问时。当对象在堆上分配时，后者可能会有些棘手。看一下下面的代码；它在不同的作用域和内存段中声明了四个`Product`对象：

```cpp
static Product global_prod; // #1

Product* foo() {
  Product* heap_prod = new Product(); // #4
  heap_prod->name = "Sample";
  return heap_prod;
}

int main() {
 Product stack_prod; // #2
  if (true) {
    Product tmp; // #3
    tmp.rating = 3;
  }
  stack_prod.price = 4.2;
  foo();
}
```

`global_prod`具有静态存储期，并且放置在程序的全局/静态部分；它在调用`main()`之前被初始化。当`main()`开始时，`stack_prod`被分配在堆栈上，并且在`main()`结束时被销毁（函数的闭合大括号被视为其结束）。虽然条件表达式看起来很奇怪和太人为，但这是一种表达块作用域的好方法。

`tmp`对象也将分配在堆栈上，但其存储持续时间限制在其声明的范围内：当执行离开`if`块时，它将被自动销毁。这就是为什么堆栈上的变量具有*自动存储持续时间*。最后，当调用`foo()`函数时，它声明了`heap_prod`指针，该指针指向在堆上分配的`Product`对象的地址。

上述代码包含内存泄漏，因为`heap_prod`指针（它本身具有自动存储持续时间）将在执行到达`foo()`末尾时被销毁，而在堆上分配的对象不会受到影响。不要混淆指针和它指向的实际对象：指针只包含对象的值，但它并不代表对象。

不要忘记释放在堆上动态分配的内存，可以通过手动调用删除运算符或使用智能指针来实现。智能指针将在第五章中讨论，*内存管理和智能指针*。

当函数结束时，分配在堆栈上的参数和局部变量的内存将被释放，但`global_prod`将在程序结束时被销毁，也就是在`main()`函数结束后。当对象即将被销毁时，析构函数将被调用。

# 复制对象

有两种复制方式：对象的*深*复制和*浅*复制。语言允许我们使用**复制构造函数**和**赋值运算符**来管理对象的复制初始化和赋值。这对程序员来说是一个必要的特性，因为我们可以控制复制的语义。看下面的例子：

```cpp
Product p1;
Product p2;
p2.set_price(4.2);
p1 = p2; // p1 now has the same price
Product p3 = p2; // p3 has the same price
```

`p1 = p2;`这一行是对赋值运算符的调用，而最后一行是对复制构造函数的调用。等号不应该让你困惑，无论是赋值还是复制构造函数调用。每当看到声明后面跟着一个赋值时，都可以将其视为复制构造。新的初始化程序语法(`Product p3{p2};`)也是如此。

编译器将生成以下代码：

```cpp
Product p1;
Product p2;
Product_set_price(p2, 4.2);
operator=(p1, p2);
Product p3;
Product_copy_constructor(p3, p2);
```

复制构造函数（和赋值运算符）的默认实现执行对象的成员逐个复制，如下图所示：

![](img/4a801266-0db0-4ca7-81e1-afa9b2da53a0.png)

如果成员逐个复制产生无效副本，则需要自定义实现。例如，考虑以下`Warehouse`对象的复制：

```cpp
class Warehouse {
public:
  Warehouse() 
    : size_{0}, capacity_{1000}, products_{nullptr}
  {
    products_ = new Products[capacity_];
  }

  ~Warehouse() {
    delete [] products_;
  }

public:
  void add_product(const Product& p) {
    if (size_ == capacity_) { /* resize */ }
    products_[size_++] = p;
  }
  // other functions omitted for brevity

private:
  int size_;
  int capacity_;
  Product* products_;
};

int main() {
  Warehouse w1;
  Product book;
  Product apple;
  // ...assign values to products (omitted for brevity)
  w1.add_product(book);
  Warehouse w2 = w1; // copy
  w2.add_product(apple);
  // something somewhere went wrong...
}
```

上述代码声明了两个`Warehouse`对象，然后向仓库添加了两种不同的产品。虽然这个例子有些不自然，但它展示了默认复制实现的危险。以下插图展示了代码中出现的问题：

![](img/9cb93f9f-29a1-4e3d-a0df-d9b226e141d9.png)

将**w1**赋给**w2**会导致以下结构：

![](img/57f8bc17-d88a-4fd8-b03b-8263a6b20b2d.png)

默认实现只是将`w1`的每个成员复制到`w2`。复制后，`w1`和`w2`的`products_`成员都指向堆上的相同位置。当我们向`w2`添加新产品时，`w1`指向的数组会受到影响。这是一个逻辑错误，可能导致程序中的未定义行为。我们需要进行*深*复制而不是*浅*复制；也就是说，我们需要实际创建一个包含 w1 数组副本的新产品数组。

自定义实现复制构造函数和赋值运算符解决了*浅*复制的问题：

```cpp
class Warehouse {
public:
  // ...
  Warehouse(const Warehouse& rhs) {
 size_ = rhs.size_;
 capacity_ = rhs.capacity_;
 products_ = new Product[capacity_];
 for (int ix = 0; ix < size_; ++ix) {
 products_[ix] = rhs.products_[ix];
 }
 }
  // code omitted for brevity
};  
```

复制构造函数的自定义实现创建一个新数组。然后，它逐个复制源对象的数组元素，从而消除了`product_`指针指向错误的内存地址。换句话说，我们通过创建一个新数组实现了`Warehouse`对象的深复制。

# 移动对象

临时对象在代码中随处可见。大多数情况下，它们是必需的，以使代码按预期工作。例如，当我们将两个对象相加时，会创建一个临时对象来保存`operator+`的返回值：

```cpp
Warehouse small;
Warehouse mid;
// ... some data inserted into the small and mid objects
Warehouse large{small + mid}; // operator+(small, mid)
```

让我们来看看`Warehouse`对象的全局`operator+()`的实现：

```cpp
// considering declared as friend in the Warehouse class
Warehouse operator+(const Warehouse& a, const Warehouse& b) {
  Warehouse sum; // temporary
  sum.size_ = a.size_ + b.size_;
  sum.capacity_ = a.capacity_ + b.capacity_;
  sum.products_ = new Product[sum.capacity_];
  for (int ix = 0; ix < a.size_; ++ix) { sum.products_[ix] = a.products_[ix]; }
  for (int ix = 0; ix < b.size_; ++ix) { sum.products_[a.size_ + ix] = b.products_[ix]; }
  return sum;
}
```

前面的实现声明了一个临时对象，并在填充必要数据后返回它。在前面的示例中，调用可以被翻译成以下内容：

```cpp
Warehouse small;
Warehouse mid;
// ... some data inserted into the small and mid objects
Warehouse tmp{operator+(small, mid)};
Warehouse large;
Warehouse_copy_constructor(large, tmp);
__destroy_temporary(tmp);
```

*移动语义*，它在 C++11 中引入，允许我们通过*移动*返回值到`Warehouse`对象中来跳过临时创建。为此，我们应该为`Warehouse`声明一个**移动构造函数**，它可以*区分*临时对象并有效地处理它们：

```cpp
class Warehouse {
public:
  Warehouse(); // default constructor
  Warehouse(const Warehouse&); // copy constructor
  Warehouse(Warehouse&&); // move constructor
  // code omitted for brevity
};
```

移动构造函数的参数是**rvalue 引用**（**&&**）。

# Lvalue 引用

在理解为什么首先引入 rvalue 引用之前，让我们澄清一下关于`lvalues`、`references`和`lvalue-references`的事情。当一个变量是 lvalue 时，它可以被寻址，可以被指向，并且具有作用域存储期：

```cpp
double pi{3.14}; // lvalue
int x{42}; // lvalue
int y{x}; // lvalue
int& ref{x}; // lvalue-reference
```

`ref`是一个`lvalue`引用，相当于可以被视为`const`指针的变量：

```cpp
int * const ref = &x;
```

除了通过引用修改对象的能力，我们还通过引用将重型对象传递给函数，以便优化和避免冗余对象的复制。例如，`Warehouse`的`operator+`接受两个对象的*引用*，因此它复制对象的地址而不是完整对象。

`Lvalue`引用在函数调用方面优化了代码，但是为了优化临时对象，我们应该转向 rvalue 引用。

# Rvalue 引用

我们不能将`lvalue`引用绑定到临时对象。以下代码将无法编译：

```cpp
int get_it() {
  int it{42};
  return it;
}
...
int& impossible{get_it()}; // compile error
```

我们需要声明一个`rvalue`引用，以便能够绑定到临时对象（包括文字值）：

```cpp
int&& possible{get_it()};
```

`Rvalue`引用允许我们尽可能地跳过临时对象的生成。例如，以 rvalue 引用接受结果的函数通过消除临时对象而运行得更快：

```cpp
void do_something(int&& val) {
  // do something with the val
}
// the return value of the get_it is moved to do_something rather than copied
do_something(get_it()); 
```

为了想象移动的效果，想象一下前面的代码将被翻译成以下内容（只是为了完全理解移动）：

```cpp
int val;
void get_it() {
  val = 42;
}
void do_something() {
  // do something with the val
}
do_something();
```

在引入移动之前，前面的代码看起来像这样（带有一些编译器优化）：

```cpp
int tmp;
void get_it() {
  tmp = 42;
}
void do_something(int val) {
  // do something with the val
}
do_something(tmp);
```

移动构造函数和移动操作符`=()`一起，当输入参数表示一个`rvalue`时，具有复制而不实际执行复制操作的效果。这就是为什么我们应该在类中实现这些新函数：这样我们就可以在任何有意义的地方优化代码。移动构造函数可以获取源对象而不是复制它，如下所示：

```cpp
class Warehouse {
public:
  // constructors omitted for brevity
  Warehouse(Warehouse&& src)
 : size_{src.size_}, 
 capacity_{src.capacity_},
 products_{src.products_}
 {
 src.size_ = 0;
 src.capacity_ = 0;
 src.products_ = nullptr;
 }
};
```

我们不是创建一个`capacity_`大小的新数组，然后复制`products_`数组的每个元素，而是直接获取了数组的指针。我们知道`src`对象是一个 rvalue，并且它很快就会被销毁，这意味着析构函数将被调用，并且析构函数将删除分配的数组。现在，我们指向新创建的`Warehouse`对象的分配数组，这就是为什么我们不能让析构函数删除源数组。因此，我们将`nullptr`赋给它，以确保析构函数不会错过分配的对象。因此，由于移动构造函数，以下代码将被优化：

```cpp
Warehouse large = small + mid;
```

`+`操作符的结果将被移动而不是复制。看一下下面的图表：

![](img/d7d0904d-6549-4c3c-aadd-6bb1785dfa17.png)

前面的图表演示了临时对象如何被移动到大对象中。

# 运算符重载的注意事项

C++为自定义类型提供了强大的运算符重载机制。使用`+`运算符计算两个对象的和要比调用成员函数好得多。调用成员函数还涉及在调用之前记住它的名称。它可能是`add`，`calculateSum`，`calculate_sum`或其他名称。运算符重载允许在类设计中采用一致的方法。另一方面，运算符重载会增加代码中不必要的冗长。以下代码片段表示对`Money`类进行了比较运算符的重载，以及加法和减法：

```cpp
constexpr bool operator<(const Money& a, const Money& b) { 
  return a.value_ < b.value_; 
}
constexpr bool operator==(const Money& a, const Money& b) { 
  return a.value_ == b.value_; 
}
constexpr bool operator<=(const Money& a, const Money& b) { 
  return a.value_ <= b.value_; 
}
constexpr bool operator!=(const Money& a, const Money& b) { 
  return !(a == b); 
}
constexpr bool operator>(const Money& a, const Money& b) { 
  return !(a <= b); 
}
constexpr bool operator>=(const Money& a, const Money& b) { 
  return !(a < b); 
}
constexpr Money operator+(const Money& a, const Money& b) { 
  return Money{a.value_ + b.value_}; 
}
constexpr Money operator-(const Money& a, const Money& b) { 
  return Money{a.value_ - b.value_}; 
}
```

正如你所看到的，前面大部分函数直接访问了`Money`实例的值成员。为了使其工作，我们应该将它们声明为`Money`的友元。`Money`将如下所示：

```cpp
class Money
{
public:
  Money() {}
  explicit Money(double v) : value_{v} {}
  // construction/destruction functions omitted for brevity

public:
  friend constexpr bool operator<(const Money&, const Money&);
 friend constexpr bool operator==(const Money&, const Money&);
 friend constexpr bool operator<=(const Money&, const Money&);
 friend constexpr bool operator!=(const Money&, const Money&);
 friend constexpr bool operator>(const Money&, const Money&);
 friend constexpr bool operator>=(const Money&, const Money&);
 friend constexpr bool operator+(const Money&, const Money&);
 friend constexpr bool operator-(const Money&, const Money&);

private:
  double value_;
}; 
```

这个类看起来很庞大。C++20 引入了太空船操作符，它允许我们跳过比较运算符的定义。`operator<=>()`，也被称为三路比较运算符，请求编译器生成关系运算符。对于`Money`类，我们可以使用默认的`operator<=>()`，如下所示：

```cpp
class Money
{
  // code omitted for brevity
 friend auto operator<=>(const Money&, const Money&) = default;
};
```

编译器将生成`==`，`!=`，`<`，`>`，`<=`，`>=`运算符。`太空船`运算符减少了运算符的冗余定义，并提供了一种为所有生成的运算符实现通用行为的方法。在为`太空船`运算符实现自定义行为时，我们应该注意运算符的返回值类型。它可以是以下之一：

+   `std::strong_ordering`

+   `std::weak_ordering`

+   `std::partial_ordering`

+   `std::strong_equality`

+   `std::weak_equality`

它们都在`<compare>`头文件中定义。编译器根据三路运算符的返回类型生成运算符。

# 封装和公共接口

**封装**是面向对象编程中的一个关键概念。它允许我们隐藏对象的实现细节，使其对客户端代码不可见。以计算机键盘为例；它有用于字母、数字和符号的按键，每个按键在按下时都会起作用。它的使用简单直观，并隐藏了许多只有熟悉电子设备的人才能处理的低级细节。想象一下一个没有按键的键盘——一个只有裸板和未标记引脚的键盘。你将不得不猜测要按下哪个键才能实现所需的按键组合或文本输入。现在，想象一个没有引脚的键盘——你必须向相应的插座发送正确的信号才能获得特定符号的按键*按下*事件。用户可能会因为缺少标签而感到困惑，他们也可能会错误地按下或向无效的插座发送信号。我们所知道的键盘通过封装实现了这一点——程序员也通过封装对象来确保用户不会因为冗余成员而负担过重，以及确保用户不会以错误的方式使用对象。

在类中，可见性修饰符通过允许我们定义任何成员的可访问级别来实现这一目的。`private`修饰符禁止客户端代码使用`private`成员，这使我们能够通过提供相应的成员函数来控制`private`成员的修改。一个`mutator`函数，对许多人来说是一个设置函数，会在测试该特定类的值是否符合指定规则后修改`private`成员的值。以下代码中可以看到这一点的例子：

```cpp
class Warehouse {
public:
  // rather naive implementation
  void set_size(int sz) {
 if (sz < 1) throw std::invalid_argument("Invalid size");
 size_ = sz;
 }
  // code omitted for brevity
private:
  int size_;
};
```

通过`mutator`函数修改数据成员允许我们控制其值。实际数据成员是私有的，这使得它无法从客户端代码访问，而类本身提供了公共函数来更新或读取其私有成员的内容。这些函数以及构造函数通常被称为类的*公共接口*。程序员们努力使类的公共接口用户友好。

看一下下面的类，它表示一个二次方程求解器：一个形式为`ax² + bx + c = 0`的方程。找到判别式并根据判别式（D）的值计算`x`的值是解决方案之一。以下类提供了五个函数，分别用于设置`a`、`b`和`c`的值，找到判别式，解决并返回`x`的值：

```cpp
class QuadraticSolver {
public:
  QuadraticSolver() = default;
  void set_a(double a);
 void set_b(double b);
 void set_c(double c);
 void find_discriminant();
 double solve(); // solve and return the x
private:
  double a_;
  double b_;
  double c_;
  double discriminant_;
};
```

公共接口包括前面提到的四个函数和默认构造函数。要解决方程*2x² + 5x - 8 = 0*，我们应该这样使用`QuadraticSolver`：

```cpp
QuadraticSolver solver;
solver.set_a(2);
solver.set_b(5);
solver.set_c(-8);
solver.find_discriminant();
std::cout << "x is: " << solver.solve() << std::endl;
```

类的公共接口应该被明智地设计；前面的例子显示了糟糕设计的迹象。用户必须知道协议，也就是确切的调用函数的顺序。如果用户忽略了对`find_discriminant()`的调用，结果将是未定义或无效的。公共接口强迫用户学习协议，并按正确的顺序调用函数，即设置`a`、`b`和`c`的值，然后调用`find_discriminant()`函数，最后调用`solve()`函数以获得`x`的期望值。一个好的设计应该提供一个直观易用的公共接口。我们可以重写`QuadraticSolver`，使其只有一个函数，接受所有必要的输入值，计算判别式本身，并返回解决方案：

```cpp
class QuadtraticSolver {
public:
  QuadraticSolver() = default;
 double solve(double a, double b, double c);
};
```

前面的设计比之前的更直观。以下代码演示了如何使用`QuadraticSolver`来找到方程*2x2 + 5x - 8 = 0*的解：

```cpp
QuadraticSolver solver;
std::cout << solver.solve(2, 5, -8) << std::endl;
```

在这里需要考虑的最后一件事是，二次方程可以有多种解法。我们介绍的方法是通过找到判别式来解决的。我们应该考虑，将来我们可能会为这个类添加更多的实现方法。改变函数的名称可能会增加公共接口的可读性，并确保对类的未来更新。我们还应该注意，在前面的例子中，`solve()`函数接受`a`、`b`和`c`作为参数，我们不需要在类中存储它们，因为解决方案是直接在函数中计算的。

显然，声明一个`QuadraticSolver`的对象只是为了能够访问`solve()`函数似乎是一个多余的步骤。类的最终设计将如下所示：

```cpp
class QuadraticSolver {
public:
  QuadraticSolver() = delete;

  static double solve_by_discriminant(double a, double b, double c);
  // other solution methods' implementations can be prefixed by "solve_by_"
};
```

我们将`solve()`函数重命名为`solve_by_discriminant()`，这也暴露了解决方案的底层方法。我们还将函数设为*static*，这样用户就可以在不声明类的实例的情况下使用它。然而，我们还将默认构造函数标记为*deleted*，这再次强制用户不要声明对象：

```cpp
std::cout << QuadraticSolver::solve_by_discriminant(2, 5, -8) << std::endl;
```

客户端代码现在使用该类的工作量更少。

# C++中的结构体

在 C++中，结构体和类几乎是相同的。它们具有类的所有特性，你可以从结构体继承一个类，反之亦然。`class`和`struct`之间唯一的区别是默认可见性。对于结构体，默认可见性修饰符是公共的。它也与继承有关。例如，当你从另一个类继承一个类而不使用修饰符时，它会私有继承。以下类私有地继承自`Base`：

```cpp
class Base
{
public:
  void foo() {}
};

class Derived : Base
{
  // can access foo() while clients of Derived can't
};
```

按照相同的逻辑，以下结构体公开继承`Base`：

```cpp
struct Base
{
  // no need to specify the public section
  void foo() {}
};

struct Derived : Base
{
  // both Derived and clients of Derived can access foo()
};
```

与继承自结构体的类相关。例如，如果没有直接指定，`Derived`类会私有地继承`Base`：

```cpp
struct Base
{
  void foo() {}
};

// Derived inherits Base privately
class Derived: Base
{
  // clients of Derived can't access foo()
};
```

在 C++中，结构体和类是可以互换的，但大多数程序员更喜欢使用结构体来表示简单类型。C++标准对简单类型给出了更好的定义，并称它们为**聚合**。如果一个类（结构体）符合以下规则，则它是一个聚合：

+   没有私有或受保护的非静态数据成员

+   没有用户声明或继承的构造函数

+   没有虚拟、私有或受保护的基类

+   没有虚成员函数

在完成本章后，大多数规则会更加清晰。以下结构是一个聚合的例子：

```cpp
struct Person
{
  std::string name;
  int age;
  std::string profession;
};
```

在深入研究继承和虚函数之前，让我们看看聚合在初始化时带来了什么好处。我们可以以以下方式初始化`Person`对象：

```cpp
Person john{"John Smith", 22, "programmer"};
```

C++20 提供了更多初始化聚合的新方法：

```cpp
Person mary{.name = "Mary Moss", .age{22}, .profession{"writer"}};
```

注意我们如何通过指示符混合初始化成员。

结构化绑定允许我们声明绑定到聚合成员的变量，如下面的代码所示：

```cpp
const auto [p_name, p_age, p_profession] = mary;
std::cout << "Profession is: " << p_profession << std::endl;
```

结构化绑定也适用于数组。

# 类关系

对象间通信是面向对象系统的核心。关系是对象之间的逻辑链接。我们如何区分或建立类对象之间的适当关系，定义了系统设计的性能和质量。考虑`Product`和`Warehouse`类；它们处于一种称为聚合的关系，因为`Warehouse`包含`Products`，也就是说，`Warehouse`聚合了`Products`：

![](img/76b956c9-22ba-4ee7-af40-f4edc84ae8ca.png)

在纯面向对象编程中有几种关系，比如关联、聚合、组合、实例化、泛化等。

# 聚合和组合

我们在`Warehouse`类的例子中遇到了聚合。`Warehouse`类存储了一个产品数组。更一般的说，它可以被称为*关联*，但为了强调确切的包含性，我们使用*聚合*或*组合*这个术语。在聚合的情况下，包含其他类的类可以在没有聚合的情况下实例化。这意味着我们可以创建和使用`Warehouse`对象，而不一定要创建`Warehouse`中包含的`Product`对象。

聚合的另一个例子是`Car`和`Person`。`Car`可以包含一个`Person`对象（作为驾驶员或乘客），因为它们彼此相关，但包含性不强。我们可以创建一个没有`Driver`的`Car`对象，如下所示：

```cpp
class Person; // forward declaration
class Engine { /* code omitted for brevity */ };
class Car {
public:
  Car();
  // ...
private:
  Person* driver_; // aggregation
  std::vector<Person*> passengers_; // aggregation
  Engine engine_; // composition
  // ...
}; 
```

强大的包含性由**组合**来表达。以`Car`为例，需要一个`Engine`类的对象才能组成一个完整的`Car`对象。在这种物理表示中，当创建一个`Car`时，`Engine`成员会自动创建。

以下是聚合和组合的 UML 表示：

![](img/7e185cb8-02eb-4899-9520-8ca23b59015c.png)

在设计类时，我们必须决定它们的关系。定义两个类之间的组合关系的最佳方法是*有一个*关系测试。`Car` 有一个 `Engine`，因为汽车有发动机。每当你不能确定关系是否应该以组合的方式表达时，问一下*有一个*的问题。聚合和组合有些相似；它们只是描述了连接的强度。对于聚合，适当的问题应该是*可以有一个*；例如，一个`Car`可以有一个驾驶员（类型为`Person`）；也就是说，包含性是弱的。

# 继承

**继承**是一种允许我们重用类的编程概念。编程语言提供了不同的继承实现，但总的规则始终是：类关系应该回答*是一个*的问题。例如，`Car`是一个`Vehicle`，这使我们可以从`Vehicle`继承`Car`：

```cpp
class Vehicle {
public:
  void move();
};

class Car : public Vehicle {
public:
  Car();
  // ...
};
```

`Car`现在有了从`Vehicle`继承而来的`move()`成员函数。继承本身代表了一种泛化/特化的关系，其中父类（`Vehicle`）是泛化，子类（`Car`）是特化。

父类可以被称为基类或超类，而子类可以被称为派生类或子类。

只有在绝对必要的情况下才应考虑使用继承。正如我们之前提到的，类应该满足*是一个*的关系，有时这有点棘手。考虑`Square`和`Rectangle`类。以下代码以可能的最简形式声明了`Rectangle`类：

```cpp
class Rectangle {
public:
  // argument checks omitted for brevity
  void set_width(int w) { width_ = w; }
  void set_height(int h) { height_ = h; }
  int area() const { return width_ * height_; }
private:
  int width_;
  int height_;
};
```

`Square` *是一个* `Rectangle`，所以我们可以很容易地从`Rectangle`继承它：

```cpp
class Square : public Rectangle {
public:
  void set_side(int side) {
 set_width(side);
 set_height(side);
  }

 int area() { 
    area_ = Rectangle::area();
    return area_; 
  }
private:
 int area_;
};
```

`Square`通过添加一个新的数据成员`area_`并覆盖`area()`成员函数的实现来扩展`Rectangle`。在实践中，`area_`及其计算方式是多余的；我们这样做是为了演示一个糟糕的类设计，并使`Square`在一定程度上扩展其父类。很快，我们将得出结论，即在这种情况下，继承是一个糟糕的设计选择。`Square`是一个`Rectangle`，所以应该在`Rectangle`使用的任何地方使用`Rectangle`，如下所示：

```cpp
void make_big_rectangle(Rectangle& ref) {
  ref->set_width(870);
  ref->set_height(940);
}

int main() {
  Rectangle rect;
  make_big_rectangle(rect);
  Square sq;
  // Square is a Rectangle
  make_big_rectangle(sq);
}
```

`make_big_rectangle()`函数接受`Rectangle`的引用，而`Square`继承了它，所以将`Square`对象发送到`make_big_rectangle()`函数是完全可以的；`Square` *是一个* `Rectangle`。这种成功用其子类型替换类型的示例被称为**Liskov 替换原则**。让我们找出为什么这种替换在实践中有效，然后决定我们是否通过从`Rectangle`继承`Square`而犯了设计错误（是的，我们犯了）。

# 从编译器的角度来看继承

我们可以这样描述我们之前声明的`Rectangle`类：

![](img/d6008180-4ca7-4fb1-9983-255e24f7972d.png)

当我们在`main()`函数中声明`rect`对象时，函数的本地对象所需的空间被分配在堆栈中。当调用`make_big_rectangle()`函数时，遵循相同的逻辑。它没有本地参数；相反，它有一个`Rectangle&`类型的参数，其行为类似于指针：它占用存储内存地址所需的内存空间（在 32 位和 64 位系统中分别为 4 或 8 字节）。`rect`对象通过引用传递给`make_big_rectangle()`，这意味着`ref`参数指的是`main()`中的本地对象：

![](img/e4df863f-4b4a-4a28-ae69-dffaab6c6b0a.png)

以下是`Square`类的示例：

![](img/6d479be2-8fce-47b9-937f-062bbb09e7af.png)

如前图所示，`Square`对象包含`Rectangle`的**子对象**；它部分代表了`Rectangle`。在这个特定的例子中，`Square`类没有用新的数据成员扩展矩形。

`Square`对象被传递给`make_big_rectangle()`，尽管后者需要一个`Rectangle&`类型的参数。我们知道在访问底层对象时需要指针（引用）的类型。类型定义了应该从指针指向的起始地址读取多少字节。在这种情况下，`ref`存储了在`main()`中声明的本地`rect`对象的起始地址的副本。当`make_big_rectangle()`通过`ref`访问成员函数时，实际上调用的是以`Rectangle`引用作为第一个参数的全局函数。该函数被转换为以下形式（再次，为了简单起见，我们稍作修改）：

```cpp
void make_big_rectangle(Rectangle * const ref) {
  Rectangle_set_width(*ref, 870);
  Rectangle_set_height(*ref, 940);
}
```

解引用`ref`意味着从`ref`指向的内存位置开始读取`sizeof(Rectangle)`字节。当我们将`Square`对象传递给`make_big_rectangle()`时，我们将`sq`（`Square`对象）的起始地址分配给`ref`。这将正常工作，因为`Square`对象实际上包含一个`Rectangle`子对象。当`make_big_rectangle()`函数解引用`ref`时，它只能访问对象的`sizeof(Rectangle)`字节，并且看不到实际`Square`对象的附加字节。以下图示了`ref`指向的子对象的部分：

![](img/e80a6dd3-39ba-4f60-9d03-d5eb016642aa.png)

从`Rectangle`继承`Square`几乎与声明两个结构体相同，其中一个（子类）包含另一个（父类）：

```cpp
struct Rectangle {
 int width_;
 int height_;
};

void Rectangle_set_width(Rectangle& this, int w) {
  this.width_ = w;
}

void Rectangle_set_height(Rectangle& this, int h) {
  this.height_ = h;
}

int Rectangle_area(const Rectangle& this) {
  return this.width_ * this.height_;
}

struct Square {
 Rectangle _parent_subobject_;
 int area_; 
};

void Square_set_side(Square& this, int side) {
  // Rectangle_set_width(static_cast<Rectangle&>(this), side);
 Rectangle_set_width(this._parent_subobject_, side);
  // Rectangle_set_height(static_cast<Rectangle&>(this), side);
 Rectangle_set_height(this._parent_subobject_, side);
}

int Square_area(Square& this) {
  // this.area_ = Rectangle_area(static_cast<Rectangle&>(this));
 this.area_ = Rectangle_area(this._parent_subobject_); 
  return this.area_;
}
```

上述代码演示了编译器支持继承的方式。看一下`Square_set_side`和`Square_area`函数的注释代码。我们实际上并不坚持这种实现，但它表达了编译器处理面向对象编程代码的完整思想。

# 组合与继承

C++语言为我们提供了方便和面向对象的语法，以便我们可以表达继承关系，但编译器处理它的方式更像是组合而不是继承。实际上，在适用的地方使用组合而不是继承甚至更好。`Square`类及其与`Rectangle`的关系被认为是一个糟糕的设计选择。其中一个原因是子类型替换原则，它允许我们以错误的方式使用`Square`：将其传递给一个将其作为`Rectangle`而不是`Square`修改的函数。这告诉我们*是一个*关系并不正确，因为`Square`毕竟不是`Rectangle`。它是`Rectangle`的一种适应，而不是`Rectangle`本身，这意味着它实际上并不代表`Rectangle`；它使用`Rectangle`来为类用户提供有限的功能。

`Square`的用户不应该知道它可以被用作`Rectangle`；否则，在某个时候，他们会向`Square`实例发送无效或不支持的消息。无效消息的例子是调用`set_width`或`set_height`函数。`Square`实际上不应该支持两个不同的成员函数来分别修改它的边，但它不能隐藏这一点，因为它宣布它是从`Rectangle`继承而来的：

```cpp
class Square : public Rectangle {
  // code omitted for brevity
};
```

如果我们将修饰符从 public 改为 private 会怎么样？嗯，C++支持公有和私有继承类型。它还支持受保护的继承。当从类私有继承时，子类打算使用父类并且可以访问其公共接口。然而，客户端代码并不知道它正在处理一个派生类。此外，从父类继承的公共接口对于子类的用户来说变成了私有的。似乎`Square`将继承转化为组合：

```cpp
class Square : private Rectangle {
public:
  void set_side(int side) {
    // Rectangle's public interface is accessible to the Square
    set_width(side);
 set_height(side);
  }
  int area() {
    area_ = Rectangle::area();
    return area_;
  }
private:
  int area_;
};
```

客户端代码无法访问从`Rectangle`继承的成员：

```cpp
Square sq;
sq.set_width(14); // compile error, the Square has no such public member
make_big_rectangle(sq); // compile error, can't cast Square to Rectangle
```

通过在`Square`的私有部分声明一个`Rectangle`成员也可以实现相同的效果：

```cpp
class Square {
public: 
  void set_side(int side) {
 rectangle_.set_width(side);
 rectangle_.set_height(side);
  }
  int area() {
 area_ = rectangle_.area();
    return area_;
  }
private:
 Rectangle rectangle_;
  int area_;
};
```

你应该仔细分析使用场景，并完全回答*是一个*问题，以便毫无疑问地使用继承。每当你在组合和继承之间做出选择时，选择组合。

在私有继承时，我们可以省略修饰符。类的默认访问修饰符是 private，所以`class Square : private Rectangle {};`和`class Square : Rectangle {};`是一样的。相反，结构体的默认修饰符是 public。

# 受保护的继承

最后，我们有**protected**访问修饰符。它指定了类成员在类体中使用时的访问级别。受保护成员对于类的用户来说是私有的，但对于派生类来说是公共的。如果该修饰符用于指定继承的类型，它对于派生类的用户的行为类似于私有继承。私有继承隐藏了基类的公共接口，而受保护继承使其对派生类的后代可访问。

很难想象一个需要受保护继承的场景，但你应该将其视为一个可能在意料之外的明显设计中有用的工具。假设我们需要设计一个栈数据结构适配器。栈通常是基于向量（一维数组）、链表或双端队列实现的。

栈符合 LIFO 规则，即最后插入栈的元素将首先被访问。同样，首先插入栈的元素将最后被访问。我们将在第六章中更详细地讨论数据结构和 STL 中的数据结构适配器和算法

栈本身并不代表一个数据结构；它*位于*数据结构的顶部，并通过限制、修改或扩展其功能来适应其使用。以下是表示整数一维数组的`Vector`类的简单声明：

```cpp
class Vector {
public:
  Vector();
  Vector(const Vector&);
  Vector(Vector&&);
  Vector& operator=(const Vector&);
  Vector& operator=(Vector&&);
  ~Vector();

public:
  void push_back(int value);
  void insert(int index, int value);
  void remove(int index);
  int operator[](int index);
  int size() const;
  int capacity() const;

private:
  int size_;
  int capacity_;
  int* array_;
};
```

前面的`Vector`不是一个具有随机访问迭代器支持的 STL 兼容容器；它只包含动态增长数组的最低限度。可以这样声明和使用它：

```cpp
Vector v;
v.push_back(4);
v.push_back(5);
v[1] = 2;
```

`Vector`类提供`operator[]`，允许我们随机访问其中的任何项，而`Stack`禁止随机访问。`Stack`提供`push`和`pop`操作，以便我们可以插入值到其底层数据结构中，并分别获取该值：

```cpp
class Stack : private Vector {
public:
  // constructors, assignment operators and the destructor are omitted for brevity
 void push(int value) {
 push_back(value);
 }
 int pop() {
 int value{this[size() - 1]};
 remove(size() - 1);
 return value;
 }
};
```

`Stack`可以以以下方式使用：

```cpp
Stack s;
s.push(5);
s.push(6);
s.push(3);
std::cout << s.pop(); // outputs 3
std::cout << s.pop(); // outputs 6
s[2] = 42; // compile error, the Stack has no publicly available operator[] defined
```

栈*适配*`Vector`并提供两个成员函数，以便我们可以访问它。私有继承允许我们使用`Vector`的全部功能，并且隐藏继承信息，不让`Stack`的用户知道。如果我们想要继承`Stack`来创建其高级版本怎么办？假设`AdvancedStack`类提供了`min()`函数，以常数时间返回栈中包含的最小值。

私有继承禁止`AdvancedStack`使用`Vector`的公共接口，因此我们需要一种方法来允许`Stack`的子类使用其基类，但是隐藏基类的存在。受保护的继承可以实现这一目标，如下所示：

```cpp
class Stack : protected Vector {
  // code omitted for brevity
};

class AdvancedStack : public Stack {
  // can use the Vector
};
```

通过从`Vector`继承`Stack`，我们允许`Stack`的子类使用`Vector`的公共接口。但是`Stack`和`AdvancedStack`的用户将无法将它们视为`Vector`。

# 多态

**多态**是面向对象编程中的另一个关键概念。它允许子类对从基类派生的函数进行自己的实现。假设我们有`Musician`类，它有`play()`成员函数：

```cpp
class Musician {
public:
  void play() { std::cout << "Play an instrument"; }
};
```

现在，让我们声明`Guitarist`类，它有`play_guitar()`函数：

```cpp
class Guitarist {
public:
  void play_guitar() { std::cout << "Play a guitar"; }
};
```

这是继承的明显案例，因为`Guitarist`明显表明它*是一个*`Musician`。`Guitarist`自然不应该通过添加新函数（如`play_guitar()`）来扩展`Musician`；相反，它应该提供其自己从`Musician`派生的`play()`函数的实现。为了实现这一点，我们使用**虚函数**：

```cpp
class Musician {
public:
  virtual void play() { std::cout << "Play an instrument"; }
};

class Guitarist : public Musician {
public:
  void play() override { std::cout << "Play a guitar"; }
};
```

现在，显然`Guitarist`类提供了`play()`函数的自己的实现，客户端代码可以通过使用指向基类的指针来访问它：

```cpp
Musician armstrong;
Guitarist steve;
Musician* m = &armstrong;
m->play();
m = &steve;
m->play();
```

前面的例子展示了多态的实际应用。虚函数的使用虽然很自然，但实际上除非我们正确使用它，否则并没有太多意义。首先，`Musician`的`play()`函数根本不应该有任何实现。原因很简单：音乐家应该能够在具体的乐器上演奏，因为他们不能同时演奏多个乐器。为了摆脱实现，我们通过将`0`赋值给它将函数设置为**纯虚函数**：

```cpp
class Musician {
public:
 virtual void play() = 0;
};
```

当客户端代码尝试声明`Musician`的实例时，会导致编译错误。当然，这必须导致编译错误，因为不应该能够创建具有*未定义*函数的对象。`Musician`只有一个目的：它必须只能被其他类继承。存在供继承的类称为**抽象类**。实际上，`Musician`被称为**接口**而不是抽象类。抽象类是半接口半类，可以具有有和无实现的函数。

回到我们的例子，让我们添加`Pianist`类，它也实现了`Musician`接口：

```cpp
class Pianist : public Musician {
public: 
 void play() override { std::cout << "Play a piano"; }
};
```

为了表达多态性的全部功能，假设我们在某处声明了一个函数，返回吉他手或钢琴家的集合：

```cpp
std::vector<Musician*> get_musicians();
```

从客户端代码的角度来看，很难解析`get_musicians()`函数的返回值，并找出对象的实际子类型是什么。它可能是`吉他手`或`钢琴家`，甚至是纯粹的`音乐家`。关键是客户端不应该真正关心对象的实际类型，因为它知道集合包含音乐家，而`音乐家`对象具有`play()`函数。因此，为了让它们发挥作用，客户端只需遍历集合，并让每个音乐家演奏其乐器（每个对象调用其实现）：

```cpp
auto all_musicians = get_musicians();
for (const auto& m: all_musicians) {
 m->play();
}
```

前面的代码表达了多态性的全部功能。现在，让我们了解语言如何在低级别上支持多态性。

# 底层的虚函数

虽然多态性不限于虚函数，但我们将更详细地讨论它们，因为动态多态性是 C++中最流行的多态性形式。而且，更好地理解一个概念或技术的最佳方法是自己实现它。无论我们在类中声明虚成员函数还是它具有具有虚函数的基类，编译器都会用额外的指针增强类。指针指向的表通常被称为虚函数表，或者简称为*虚表*。我们还将指针称为*虚表指针*。

假设我们正在为银行客户账户管理实现一个类子系统。假设银行要求我们根据账户类型实现取款。例如，储蓄账户允许每年取款一次，而支票账户允许客户随时取款。不涉及`Account`类的任何不必要的细节，让我们声明最少的内容，以便理解虚拟成员函数。让我们看一下`Account`类的定义：

```cpp
class Account
{
public:
 virtual void cash_out() {
 // the default implementation for cashing out 
 }  virtual ~Account() {}
private:
  double balance_;
};
```

编译器将`Account`类转换为一个具有指向虚函数表的指针的结构。以下代码表示伪代码，解释了在类中声明虚函数时发生的情况。与往常一样，请注意，我们提供的是一般性的解释，而不是特定于编译器的实现（名称修饰也是以通用形式进行的；例如，我们将`cash_out`重命名为`Account_cash_out`）：

```cpp
struct Account
{
 VTable* __vptr;
  double balance_;
};

void Account_constructor(Account* this) {
 this->__vptr = &Account_VTable;
}

void Account_cash_out(Account* this) {
  // the default implementation for cashing out
}

void Account_destructor(Account* this) {}
```

仔细看前面的伪代码。`Account`结构的第一个成员是`__vptr`。由于先前声明的`Account`类有两个虚函数，我们可以将虚表想象为一个数组，其中有两个指向虚成员函数的指针。请参阅以下表示：

```cpp
VTable Account_VTable[] = {
 &Account_cash_out,
 &Account_destructor
};
```

有了我们之前的假设，让我们找出当我们在对象上调用虚函数时编译器将生成什么代码：

```cpp
// consider the get_account() function as already implemented and returning an Account*
Account* ptr = get_account();
ptr->cash_out();
```

以下是我们可以想象编译器为前面的代码生成的代码：

```cpp
Account* ptr = get_account();
ptr->__vptr[0]();
```

虚函数在层次结构中使用时显示其功能。`SavingsAccount`从`Account`类继承如下：

```cpp
class SavingsAccount : public Account
{
public:
 void cash_out() override {
 // an implementation specific to SavingsAccount
 }
  virtual ~SavingsAccount() {}
};
```

当我们通过指针（或引用）调用`cash_out()`时，虚函数是根据指针指向的目标对象调用的。例如，假设`get_savings_account()`将`SavingsAccount`作为`Account*`返回。以下代码将调用`SavingsAccount`的`cash_out()`实现：

```cpp
Account* p = get_savings_account();
p->cash_out(); // calls SavingsAccount version of the cash_out
```

这是编译器为`SavingsClass`生成的内容：

```cpp
struct SavingsAccount
{
  Account _parent_subobject_;
  VTable* __vptr;
};

VTable* SavingsAccount_VTable[] = {
  &SavingsAccount_cash_out,
  &SavingsAccount_destructor,
};

void SavingsAccount_constructor(SavingsAccount* this) {
  this->__vptr = &SavingsAccount_VTable;
}

void SavingsAccount_cash_out(SavingsAccount* this) {
  // an implementation specific to SavingsAccount
}

void SavingsAccount_destructor(SavingsAccount* this) {}
```

所以，我们有两个不同的虚拟函数表。当我们创建一个`Account`类型的对象时，它的`__vptr`指向`Account_VTable`，而`SavingsAccount`类型的对象的`__vptr`指向`SavingsAccount_VTable`。让我们看一下以下代码：

```cpp
p->cash_out();
```

前面的代码转换成了这样：

```cpp
p->__vptr[0]();
```

现在很明显，`__vptr[0]`解析为正确的函数，因为它是通过`p`指针读取的。

如果`SavingsAccount`没有覆盖`cash_out()`函数会怎么样？在这种情况下，编译器会将基类实现的地址放在与`SavingsAccount_VTable`相同的位置，如下所示：

```cpp
VTable* SavingsAccount_VTable[] = {
  // the slot contains the base class version 
  // if the derived class doesn't have an implementation
 &Account_cash_out,
  &SavingsAccount_destructor
};
```

编译器以不同的方式实现和管理虚拟函数的表示。一些实现甚至使用不同的模型，而不是我们之前介绍的模型。我们采用了一种流行的方法，并以通用的方式表示，以简化起见。现在，我们将看看在包含动态多态性的代码底层发生了什么。

# 设计模式

设计模式是程序员最具表现力的工具之一。它们使我们能够以一种优雅和经过充分测试的方式解决设计问题。当您努力提供最佳的类设计和它们的关系时，一个众所周知的设计模式可能会挽救局面。

设计模式的最简单示例是**单例**。它为我们提供了一种声明和使用类的唯一实例的方法。例如，假设电子商务平台只有一个`Warehouse`。要访问`Warehouse`类，项目可能需要在许多源文件中包含并使用它。为了保持同步，我们应该将`Warehouse`设置为单例：

```cpp
class Warehouse {
public:
  static create_instance() {
 if (instance_ == nullptr) {
 instance_ = new Warehouse();
 }
 return instance_;
 }

 static remove_instance() {
 delete instance_;
 instance_ = nullptr;
 }

private:
  Warehouse() = default;

private:
  static Warehouse* instance_ = nullptr;
};
```

我们声明了一个静态的`Warehouse`对象和两个静态函数来创建和销毁相应的实例。私有构造函数导致每次用户尝试以旧的方式声明`Warehouse`对象时都会产生编译错误。为了能够使用`Warehouse`，客户端代码必须调用`create_instance()`函数。

```cpp
Warehouse* w = Warehouse::create_instance();
Product book;
w->add_product(book);
Warehouse::remove_instance();
```

`Warehouse`的单例实现并不完整，只是一个引入设计模式的示例。我们将在本书中介绍更多的设计模式。

# 总结

在本章中，我们讨论了面向对象编程的基本概念。我们涉及了类的低级细节和 C++对象模型的编译器实现。知道如何设计和实现类，而实际上没有类，有助于正确使用类。

我们还讨论了继承的必要性，并尝试在可能适用的地方使用组合而不是继承。C++支持三种类型的继承：公有、私有和保护。所有这些类型都在特定的类设计中有它们的应用。最后，我们通过一个大大增加客户端代码便利性的例子理解了多态性的用途和力量。

在下一章中，我们将学习更多关于模板和模板元编程的知识，这将成为我们深入研究名为概念的新 C++20 特性的基础。

# 问题

1.  对象的三个属性是什么？

1.  将对象移动而不是复制它们有什么优势？

1.  C++中结构体和类有什么区别？

1.  聚合和组合关系之间有什么区别？

1.  私有继承和保护继承有什么区别？

1.  如果我们在类中定义了虚函数，类的大小会受到影响吗？

1.  使用单例设计模式有什么意义？

# 进一步阅读

更多信息，请参考：

+   Grady Booch，《面向对象的分析与设计》（[`www.amazon.com/Object-Oriented-Analysis-Design-Applications-3rd/dp/020189551X/`](https://www.amazon.com/Object-Oriented-Analysis-Design-Applications-3rd/dp/020189551X/)）

+   Stanley Lippman，《C++对象模型内部》（[`www.amazon.com/Inside-Object-Model-Stanley-Lippman/dp/0201834545/`](https://www.amazon.com/Inside-Object-Model-Stanley-Lippman/dp/0201834545/)）
