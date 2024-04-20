# 2.构建高质量的面向对象代码

概述

在本章中，您将学习如何使用面向对象编程（OOP）简化复杂的逻辑。您将首先创建类和对象，然后探索面向对象编程的四大支柱。然后，您将了解一些最佳编码实践，即 SOLID 原则，并了解如何使用 C# 10 功能编写受这些原则指导的有效代码。通过本章结束时，您将能够使用 C#进行面向对象设计编写清晰的代码。

# 介绍

人们如何编写经过多年仍然易于维护的软件？围绕现实世界概念建模软件的最佳方法是什么？这两个问题的答案都是面向对象编程（OOP）。面向对象编程是专业编程中广泛使用的范式，尤其在企业环境中特别有用。

面向对象编程可以被认为是连接现实世界概念和源代码的桥梁。例如，猫具有一些定义属性，如年龄、毛色、眼睛颜色和名字。天气可以用温度和湿度等因素来描述。这些都是人类随着时间识别和定义的现实世界概念。在面向对象编程中，类是帮助定义程序逻辑的东西。当为这些类的属性分配具体值时，结果就是一个对象。例如，使用面向对象编程，您可以定义一个表示房子中的房间的类，然后为其属性（颜色和面积）分配值，以创建该类的对象。

在*第一章*“你好 C#”中，你学会了如何使用 C#编写基本程序。在本章中，您将看到如何通过实现面向对象编程概念和充分利用 C#来设计您的代码。

# 类和对象

类似于描述概念的蓝图。另一方面，对象是应用此蓝图后获得的结果。例如，`weather`可以是一个类，`25 度`和`无云`可以指代这个类的一个对象。同样，您可以有一个名为`Dog`的类，而四岁的`Spaniel`可以代表`Dog`类的一个对象。

在 C#中声明一个类很简单。它以`class`关键字开头，后跟类名和一对花括号。要定义一个名为`Dog`的类，您可以编写以下代码：

```cpp
class Dog
{
}
```

现在，这个类只是一个空的骨架。但是，仍然可以使用`new`关键字来创建对象，如下所示：

```cpp
Dog dog = new Dog();
```

这将创建一个名为`dog`的对象。目前，该对象是一个空壳，因为它缺少属性。在接下来的部分中，您将看到如何为类定义属性，但首先，您将探索构造函数。

# 构造函数

在 C#中，构造函数是用于创建新对象的函数。您还可以使用它们来设置对象的初始值。与任何函数一样，构造函数有一个名称，接受参数，并且可以重载。一个类必须至少有一个构造函数，但如果需要，它可以有多个具有不同参数的构造函数。即使您没有显式定义一个构造函数，类仍将具有默认构造函数-一个不接受任何参数或执行任何操作，而只是为新创建的对象及其字段分配内存的构造函数。

考虑以下代码片段，其中声明了`Dog`类的构造函数：

```cpp
// Within a class named Dog
public class Dog
{
  // Constructor
  public Dog()
  {
    Console.WriteLine("A Dog object has been created");
  }
}
```

注意

您可以在[`packt.link/H2lUF`](https://packt.link/H2lUF)找到此示例中使用的代码。您可以在[`packt.link/4WoSX`](https://packt.link/4WoSX)找到代码的用法。

如果一个方法与类同名且没有提供 `return` 类型，则它是一个构造函数。在这里，代码片段在一个名为 `Dog` 的类中。因此，构造函数在指定的代码行内。请注意，通过显式定义此构造函数，您隐藏了默认构造函数。如果有一个或多个这样的自定义构造函数，您将不再能够使用默认构造函数。一旦调用新的构造函数，您应该在控制台中看到打印出此消息："已创建一个 Dog 对象"。

## 字段和类成员

您已经知道什么是变量：它有一个类型、一个名称和一个值，就像您在*第一章* *Hello C#*中看到的那样。变量也可以存在于类范围内，这样的变量称为字段。声明字段与声明局部变量一样简单。唯一的区别是在开头添加一个关键字，即访问修饰符。例如，您可以在 `Dog` 类中声明一个具有公共访问修饰符的字段，如下所示：

```cpp
public string Name = "unnamed";
```

这行代码说明了 `Name` 字段，它是一个值为 `"unnamed"` 的字符串，可以公开访问。除了 `public`，C# 中的另外两个主要访问修饰符是 `private` 和 `protected`，您将在后面详细了解它们。

注意

您可以在[`docs.microsoft.com/en-us/dotnet/csharp/language-reference/keywords/access-modifiers`](https://docs.microsoft.com/en-us/dotnet/csharp/language-reference/keywords/access-modifiers)找到有关访问修饰符的更多信息。

类所持有的一切都称为类成员。类成员可以从类的外部访问；但是，这种访问需要使用 `public` 访问修饰符明确授予。默认情况下，所有成员都具有 `private` 访问修饰符。

您可以通过写对象名称后跟一个点(`.`)和成员名称来访问类成员。例如，考虑以下代码片段，其中创建了 `Dog` 类的两个对象：

```cpp
Dog sparky = new Dog();
Dog ricky = new Dog();
```

在这里，您可以声明两个独立的变量，`sparky` 和 `ricky`。但是，您还没有明确地将这些名称分配给对象；请注意，这些只是变量名称。要将名称分配给对象，您可以使用点表示法编写以下代码：

```cpp
sparky.Name = "Sparky";
ricky.Name = "Ricky";
```

您现在可以通过练习来亲身体验创建类和对象。

## 练习 2.01：创建类和对象

假设有两本书，都是由名为 `New Writer` 的作者写的。第一本书名为 `First Book`，由 `Publisher 1` 出版。这本书没有可用的描述。类似地，第二本书名为 `Second Book`，由 `Publisher 2` 出版。它有一个简单地说："有趣的阅读"的描述。

在这个练习中，您将在代码中对这些书进行建模。以下步骤将帮助您完成这个练习。

1.  创建一个名为 `Book` 的类。为 `Title`、`Author`、`Publisher`、`Description` 和页数添加字段。您必须从类的外部打印这些信息，因此请确保每个字段都是 `public` 的：

```cpp
    public class Book
    {
        public string Title;
        public string Author;
        public string Publisher;
        public int Pages;
        public string Description;
    }
```

1.  创建一个名为 `Solution` 的类，其中包含 `Main` 方法。正如您在*第一章* *Hello C#*中看到的那样，这个带有 `Main` 方法的类是您应用程序的起点：

```cpp
    public static class Solution
    {
        public static void Main()
        {
        }
    }
```

1.  在 `Main` 方法内，为第一本书创建一个对象，并设置字段的值，如下所示：

```cpp
Book book1 = new Book();
book1.Author = "New Writer";
book1.Title = "First Book";
book1.Publisher = "Publisher 1";
```

在这里，创建了一个名为 `book1` 的新对象。通过写点(`.`)后跟字段名称，为不同的字段分配值。第一本书没有描述，因此您可以省略字段 `book1.Description`。

1.  重复此步骤以创建第二本书。对于这本书，您还需要为 `Description` 字段设置一个值：

```cpp
Book book2 = new Book();
book2.Author = "New Writer";
book2.Title = "Second Book";
book2.Publisher = "Publisher 2";
book2.Description = "Interesting read";
```

在实践中，您很少会看到具有公共访问修饰符的字段。数据很容易变异，您可能不希望在初始化后让程序对外部更改开放。

1.  在`Solution`类中，创建一个名为`Print`的方法，该方法以`Book`对象作为参数，并打印所有字段及其值。使用字符串插值将书籍信息连接起来，并使用`Console.WriteLine()`将其打印到控制台，如下所示：

```cpp
private static void Print(Book book)
{
    Console.WriteLine($"Author: {book.Author}, " +
                      $"Title: {book.Title}, " +
                      $"Publisher: {book.Publisher}, " +
                      $"Description: {book.Description}.");
}
```

1.  在`Main`方法中，调用`book1`和`book2`的`Print`方法：

```cpp
Print(book1);
Print(book2);
```

运行此代码后，您将在控制台上看到以下输出：

```cpp
Author: New Writer, Title: First Book, Publisher: Publisher 1, Description: .
Author: New Writer, Title: Second Book, Publisher: Publisher 2, Description: Interesting read.
```

注意

您可以在此练习中使用的代码[`packt.link/MGT9b`](https://packt.link/MGT9b)。

在这个练习中，您看到了如何在简单程序中使用字段和类成员。现在继续了解引用类型。

# 引用类型

假设您有一个对象，该对象尚未创建，只是声明如下：

```cpp
Dog speedy;
```

如果您尝试访问其`Name`值会发生什么？调用`speedy.Name`将抛出`NullReferenceException`异常，因为`speedy`尚未初始化。对象是引用类型，它们的默认值是 null，直到初始化。您已经使用过值类型，比如`int`、`float`和`decimal`。现在您需要了解值类型和引用类型之间的两个主要区别。

首先，值类型在堆栈上分配内存，而引用类型在堆上分配内存。堆栈是内存中的临时位置。顾名思义，在堆栈中，内存块被堆叠在彼此之上。当您调用一个函数时，所有局部函数变量都将最终位于堆栈的一个单一块上。如果您调用一个嵌套函数，该函数的局部变量将分配在堆栈的另一个块上。

在下图中，您可以看到代码的哪些部分在执行过程中将在堆栈中分配内存，哪些部分将在堆中分配内存。方法调用（1、8、10）和局部变量（2、4）将存储在堆栈中。对象（3、5）及其成员（6）将存储在堆中。堆栈使用 Push 方法来分配数据，并使用 Pop 来释放它。当内存被分配时，它位于堆栈的顶部。当它被释放时，它也从顶部移除。一旦离开方法的范围（8、10、11），就会从堆栈中释放内存。堆要随机得多，垃圾收集器（GC）会自动（不像其他一些语言，需要自己做）释放内存。

注意

GC 本身就是一个庞大的主题。如果您想了解更多，请参阅微软官方文档[`docs.microsoft.com/en-us/dotnet/standard/garbage-collection/fundamentals`](https://docs.microsoft.com/en-us/dotnet/standard/garbage-collection/fundamentals)。

![图 2.1：堆栈和堆比较](img/B16835_02_01.jpg)

图 2.1：堆栈和堆比较

注意

如果您进行太多的嵌套调用，您将遇到`StackoverflowException`异常，因为堆栈内存不足。释放堆栈上的内存只是退出函数的问题。

第二个区别是，当值类型传递给方法时，它们的值被复制，而对于引用类型，只有引用被复制。这意味着引用类型对象的状态在方法内是可修改的，不像值类型，因为引用只是对象的地址。

考虑以下代码片段。这里，一个名为`SetTo5`的函数将数字的值设置为`5`：

```cpp
private static void SetTo5(int number)
{
        number = 5;
}
```

现在，考虑以下代码：

```cpp
int a = 2;
// a is 2
Console.WriteLine(a);
SetTo5(a);
// a is still 2
Console.WriteLine(a);
```

这应该导致以下输出：

```cpp
2
2 
```

如果您运行此代码，您会发现`a`的打印值仍然是`2`而不是`5`。这是因为`a`是一个传递值`2`的值类型，因此它的值被复制。在函数内部，您永远不会使用原始值；总是会进行复制。

那引用类型呢？假设您在`Dog`类中添加一个名为`Owner`的字段：

```cpp
public class Dog
{    public string Owner;
}
```

创建一个名为`ResetOwner`的函数，将对象的`Owner`字段的值设置为`None`：

```cpp
private static void ResetOwner(Dog dog)
{
    dog.Owner = "None";
}
```

现在，假设执行以下代码：

```cpp
Dog dog = new Dog("speedy");
Console.WriteLine(dog.Owner);
ResetOwner(dog);
// Owner is "None"- changes remain
Console.WriteLine(dog.Owner);
```

这应该导致以下输出：

```cpp
speedy
None 
```

注意

你可以在[`packt.link/gj164`](https://packt.link/gj164)找到本示例使用的代码。

如果你尝试运行这段代码片段，你会先看到一行上的名字`speedy`，然后在另一行上打印出`None`。这将改变狗的名字，并且这些改变将保留在函数外部。这是因为 Dog 是一个类，类是一个引用类型。当传递给一个函数时，会创建一个引用的副本。然而，引用的副本指向整个对象，因此所做的更改也会保留在外部。

听到你传递引用的副本可能会让人感到困惑。你怎么能确定你正在使用一个副本呢？为了了解这一点，考虑以下函数：

```cpp
private static void Recreate(Dog dog)
{
    dog = new Dog("Recreated");
}
```

在这里，创建一个新对象会创建一个新的引用。如果你改变引用类型的值，你实际上是在使用一个完全不同的对象。它可能看起来一样，但存储在内存中的位置完全不同。为传递的参数创建一个对象不会影响对象外部的任何东西。虽然这可能听起来有用，但通常应该避免这样做，因为它会使代码难以理解。

## 属性

`Dog`类有一个缺陷。从逻辑上讲，你不希望一只狗的名字在分配后被改变。然而，目前还没有任何东西可以阻止它被改变。从你可以对对象做什么的角度来考虑这个对象。你可以设置一只狗的名字（`sparky.Name = "Sparky"`），或者通过调用`sparky.Name`来获取它。然而，你想要的是一个只读的名字，只能设置一次。

大多数语言通过 setter 和 getter 方法来处理这个问题。如果给一个字段添加`public`修饰符，这意味着它既可以被检索（读取）又可以被修改（写入）。不可能只允许其中一个操作。然而，通过 setter 和 getter，你可以限制读和写访问。在面向对象编程中，限制对象的操作是确保数据完整性的关键。在 C#中，你可以使用属性来代替 setter 和 getter 方法。

在面向对象编程语言（例如 Java）中，要设置或获取名字的值，你会写类似于这样的代码：

```cpp
public string GetName()
{
    return Name;
}
public string SetName (string name)
{
    Name = name;
}
```

在 C#中，它就是这么简单的：

```cpp
public string Name {get; set;}
```

这是一个属性，实际上就是一个读起来像字段的方法。属性有两种类型：获取器和设置器。你可以用它们执行读和写操作。从前面的代码中，如果你移除`get`，它将变成只写，如果你移除`set`，它将变成只读。

在内部，属性包括一个带有后备字段的 setter 和 getter 方法。后备字段只是一个存储值的私有字段，getter 和 setter 方法与该值一起工作。你也可以编写自定义的 getter 和 setter，如下所示：

```cpp
private string _owner;
public string Owner
{
    get
    {
        return _owner;
    }
    set
    {
        _owner = value;
    }
}
```

在前面的片段中，`Owner`属性展示了`Dog`类的默认 getter 和 setter 方法的样子。

就像其他成员一样，属性的各个部分（getter 或 setter）可以有自己的访问修饰符，如下所示：

```cpp
public string Name {get; private set;}
```

在这种情况下，getter 是`public`，setter 是`private`。属性的所有部分（getter、setter 或两者，如定义的那样）都从属性（在这种情况下是`Name`）中获取访问修饰符，除非另有明确规定（如`private` set 的情况）。如果不需要设置名字，可以摆脱 setter。如果需要默认值，可以编写以下代码：

```cpp
public string Name {get;} = "unnamed";
```

这段代码意味着`Name`字段是只读的。你只能通过构造函数设置名字。请注意，这与`private` set 不同，因为后者意味着你仍然可以在`Dog`类内部更改名字。如果没有提供 setter（就像这里的情况一样），你只能在一个地方设置值，那就是构造函数。

当你创建一个只读属性时，内部会发生什么？编译器生成以下代码：

```cpp
private readonly string _name;
public string get_Name()
{
    return _name;
}
```

这表明 getter 和 setter 属性只是带有后备字段的方法。重要的是要注意，如果您有一个名为`Name`的属性，那么`set_Name()`和`get_Name()`方法将被保留，因为这是编译器在内部生成的。

在上一个片段中，您可能已经注意到了一个新的关键字`readonly`。它表示字段的值只能在声明时或在构造函数中初始化一次。

有时，使用属性返回后备字段可能看起来有些多余。例如，考虑下一个片段：

```cpp
private string _name;

public string Name
{
    get
    {
        return "Dog's name is " + _name;
    }
}
```

这段代码片段是一个自定义属性。当 getter 或 setter 不仅仅是基本返回时，您可以以这种方式编写属性，以向其添加自定义逻辑。这个属性，在不影响狗的原始名称的情况下，将在返回名称之前添加`Dog's name is`。您可以使用表达式主体属性语法使其更加简洁，如下所示：

```cpp
public string Name => "Dog's name is " + _name;
```

这段代码与上一段代码做了同样的事情；`=>`运算符表示它是一个只读属性，并且您返回的值是`=>`运算符右侧指定的值。

如果没有 setter，您如何设置初始值？答案是构造函数。在面向对象编程中，构造函数有一个目的——那就是设置字段的初始值。使用构造函数非常适合防止以无效状态创建对象。

要向`Dog`类添加一些验证，您可以编写以下代码：

```cpp
public Dog(string name)
{
  if(string.IsNullOrWhitespace(name))
  {
    throw new ArgumentNullException("name")
  }
  Name = name;
}
```

您刚刚编写的代码将阻止在创建`Dog`实例时传递空名称。

值得一提的是，在类内部，您可以访问将要创建的对象本身。这可能听起来有点混乱，但通过这个例子应该会有所启发：

```cpp
private readonly string name;
public Dog(string name)
{
  this.name = name;
}
```

`this`关键字通常用于清除类成员和参数之间的区别。`this`指的是刚刚创建的对象，因此，`this.name`指的是该对象的名称，而`name`指的是传递的参数。

现在，创建`Dog`类的对象，并设置名称的初始值，可以简化如下：

```cpp
Dog ricky = new Dog("Ricky");
Dog sparky = new Dog("Sparky");
```

您仍然有一个私有的 setter，这意味着您拥有的属性并非完全只读。您仍然可以在类本身内部更改名称的值。然而，修复这个问题非常容易；您只需删除 setter，它就会变成真正的只读。

注意

您可以在[`packt.link/hjHRV`](http://packt.link/hjHRV)找到本示例使用的代码。

## 对象初始化

通常，一个类有读写属性。通常情况下，不是通过构造函数设置属性值，而是在创建对象后分配。然而，在 C#中有一种更好的方法——对象初始化。这是您创建一个新对象并立即设置可变（读写）字段值的地方。如果您必须创建一个`Dog`类的新对象，并为该对象的`Owner`设置为`Tobias`，您可以添加以下代码：

```cpp
Dog dog = new Dog("Ricky");
dog.Owner = "Tobias";
```

可以通过对象初始化来完成这个操作，如下所示：

```cpp
Dog dog = new Dog("Ricky")
{
  Owner = "Tobias"
};
```

当它们不是构造函数的一部分时，设置初始属性通常更加简洁。对数组和其他集合类型也是如此。假设您有两个`Dog`类的对象，如下所示：

```cpp
Dog ricky = new Dog("Ricky");
Dog sparky = new Dog("Sparky");
```

在这种情况下，创建数组的一种方式如下：

```cpp
Dog[] dogs = new Dog[2];
dogs[0] = ricky;
dogs[1] = sparky;
```

然而，您可以简单地添加以下代码，这更加简洁：

```cpp
Dog[] dogs = {ricky, sparky};
```

在 C# 10 中，您可以简化对象初始化，而无需提供类型，如果可以从声明中推断出类型，如下面的代码所示：

```cpp
Dog dog = new("Dog");
```

## 比较函数和方法

到目前为止，您可能经常看到术语——函数和方法——几乎可以互换使用。现在继续深入了解函数和方法。函数是一段可以使用其名称和一些输入调用的代码块。方法是存在于类中的函数。

然而，在 C#中，您不能在类外部有函数。因此，在 C#中，每个函数都是一个方法。许多语言，特别是非面向对象的语言，只有一些可以称为方法的函数（例如 JavaScript）。

类的行为是使用方法定义的。您已经为`Dog`类定义了一些行为，即获取其名称。要完成此类的行为实现，您可以实现一些现实世界的类比，例如坐下和吠叫。这两种方法都将从外部调用：

```cpp
public void Sit()
{
    // Implementation of how a dog sits
}
public void Bark()
{
    // Implementation of how a dog barks 
}
```

您可以这样调用这两种方法：

```cpp
Ricky.Sit();
Sparky.Bark();
```

在大多数情况下，最好避免公开数据，因此您应该只公开函数。在这里，您可能会想知道，属性呢？属性只是获取器和设置器函数；它们处理数据，但并不是数据本身。您应该避免直接公开数据，原因与您锁门或将手机放在套子中的原因相同。如果数据是公开的，每个人都可以无限制地访问它。

此外，当程序要求数据保持不变时，数据不应更改。方法是一种确保对象不以无效方式使用的机制，如果是，它会被很好地处理。

如果您需要在整个应用程序中一致地验证字段，那么属性，即获取器和设置器方法，可以帮助实现这一点。您可以限制对数据的操作，并向其添加验证逻辑。属性帮助您完全控制如何获取和设置数据。属性很方便，但重要的是要谨慎使用它们。如果要做一些复杂的事情，需要额外的计算，最好使用方法。

例如，假设您有一个库存类，由物品组成，每个物品都有一些重量。在这里，可能有一个属性返回最重的物品。如果您选择通过属性（称为`MaxWeight`）这样做，您可能会得到意想不到的结果；获取最重的物品将需要遍历所有物品的集合，并按重量找到最大值。这个过程不像你期望的那样快。事实上，在某些情况下，它甚至可能会抛出错误。属性应该有简单的逻辑，否则与它们一起工作可能会产生意想不到的结果。因此，当需要计算密集型属性时，考虑将它们重构为方法。在这种情况下，您将`MaxWeight`属性重构为`GetMaxWeight`方法。

应避免使用属性返回复杂计算的结果，因为调用属性可能很昂贵。获取或设置字段的值应该是直接的。如果变得昂贵，它就不应该再被视为属性。

## 一个有效的类

`Dog`类模拟了一个`dog`对象；因此，它可以被称为模型。一些开发人员更喜欢在数据和逻辑之间有严格的分离。其他人则尽可能多地将逻辑放入模型中，只要它是自包含的。这里没有对与错的方法。这一切都取决于您所处理的上下文。

注意

这个讨论超出了本章的范围，但如果您想了解更多，可以参考[`martinfowler.com/bliki/DomainDrivenDesign.html`](https://martinfowler.com/bliki/DomainDrivenDesign.html)上关于领域驱动设计（DDD）的讨论。

很难准确描述一个有效的类是什么样的。但是，在决定方法更适合于 A 类还是 B 类时，尝试问自己以下问题：

+   不是程序员的人会知道您在谈论类吗？它是对现实世界概念的逻辑表示吗？

+   类有多少个原因需要更改？只有一个还是有更多原因？

+   私有数据是否与公共行为紧密相关？

+   类有多经常更改？

+   代码容易出错吗？

+   类自己是否做了什么？

高内聚性是一个用来描述一个类的术语，它的所有成员不仅在语义上，而且在逻辑上都是强相关的。相比之下，低内聚性的类具有松散相关的方法和字段，这些方法和字段可能有更好的位置。这样的类是低效的，因为它因多种原因而改变，你不能期望在其中查找任何东西，因为它根本没有强烈的逻辑意义。

例如，`Computer`类的一部分可能如下所示：

```cpp
class Computer
{
    private readonly Key[] keys;
}
```

然而，`Computer`和`keys`并不是在同一级别相关的。可能有另一个类更适合`Key`类，那就是`Keyboard`：

```cpp
class Computer
{
    private readonly Keyboard keyboard;
}
class Keyboard
{
    private readonly Key[] keys;
}
```

注意

你可以在[`packt.link/FFcDa`](https://packt.link/FFcDa)找到此示例使用的代码。

键盘与键直接相关，就像它与计算机直接相关一样。在这里，`Keyboard`和`Computer`类都具有高内聚性，因为依赖关系有一个稳定的逻辑位置。现在，你可以通过练习更多地了解它。

## 练习 2.02：比较不同形状占用的面积

你有两个后院的区域，一个有圆形瓷砖，另一个有矩形瓷砖。你想要拆除后院的一个区域，但你不确定应该拆除哪一个。显然，你希望尽可能少地弄脏，因此决定选择占用面积最小的区域。

给定两个数组，一个是不同尺寸的矩形瓷砖，另一个是不同尺寸的圆形瓷砖，你需要找出哪个区域要拆除。这个练习旨在输出占用面积较小的区域的名称，即`rectangular`或`circular`。

执行以下步骤来完成这个过程：

1.  创建一个`Rectangle`类如下。它应该有`width`、`height`和`area`字段：

```cpp
public class Rectangle
{
    private readonly double _width;
    private readonly double _height;
    public double Area
    {
        get
        {
            return _width * _height;
        }
    } 

    public Rectangle(double width, double height)
    {
        _width = width;
        _height = height;
    }
}
```

在这里，使用`readonly`关键字使`_width`和`_height`成为不可变的。所选的类型是`double`，因为你将执行`math`操作。唯一公开的属性是`Area`。它将返回一个简单的计算：宽度和高度的乘积。`Rectangle`是不可变的，因此它只需要通过构造函数传递一次，之后保持不变。

1.  同样，创建一个`Circle`类如下：

```cpp
public class Circle
{
    private readonly double _radius;

    public Circle(double radius)
    {
        _radius = radius;
    }

    public double Area
    {
        get { return Math.PI * _radius * _radius; }
    }
}
```

`Circle`类与`Rectangle`类类似，只是它有`radius`而不是宽度和高度，并且`Area`的计算使用了不同的公式。使用了常量`PI`，可以从`Math`命名空间中访问。

1.  创建一个名为`Solve`的骨架方法的`Solution`类：

```cpp
public static class Solution
{
    public const string Equal = "equal";
    public const string Rectangular = "rectangular";
    public const string Circular = "circular";
    public static string Solve(Rectangle[] rectangularSection, Circle[] circularSection)
    {
        var totalAreaOfRectangles = CalculateTotalAreaOfRectangles(rectangularSection);
        var totalAreaOfCircles = CalculateTotalAreaOfCircles(circularSection);
        return GetBigger(totalAreaOfRectangles, totalAreaOfCircles);
    }
}
```

在这里，`Solution`类演示了代码的工作原理。目前，有三个基于要求的常量（哪个区域更大？矩形还是圆形，或者它们相等？）。此外，流程将是先计算矩形的总面积，然后是圆形的总面积，最后返回更大的那个。

在实现解决方案之前，你必须首先创建用于计算矩形部分的总面积、计算圆形部分的总面积和比较两者的辅助方法。你将在接下来的几个步骤中完成这些工作。

1.  在`Solution`类中，添加一个方法来计算矩形部分的总面积：

```cpp
private static double CalculateTotalAreaOfRectangles(Rectangle[] rectangularSection)
{
    double totalAreaOfRectangles = 0;
    foreach (var rectangle in rectangularSection)
    {
        totalAreaOfRectangles += rectangle.Area;
    }

    return totalAreaOfRectangles;
}
```

该方法遍历所有矩形，获取每个矩形的面积，并将其添加到总和中。

1.  同样，添加一个方法来计算圆形部分的总面积：

```cpp
private static double CalculateTotalAreaOfCircles(Circle[] circularSection)
{
    double totalAreaOfCircles = 0;
    foreach (var circle in circularSection)
    {
        totalAreaOfCircles += circle.Area;
    }

    return totalAreaOfCircles;
}
```

1.  接下来，添加一个获取更大面积的方法，如下所示：

```cpp
private static string GetBigger(double totalAreaOfRectangles, double totalAreaOfCircles)
{
    const double margin = 0.01;
    bool areAlmostEqual = Math.Abs(totalAreaOfRectangles - totalAreaOfCircles) <= margin;
    if (areAlmostEqual)
    {
        return Equal;
    }
    else if (totalAreaOfRectangles > totalAreaOfCircles)
    {
        return Rectangular;
    }
    else
    {
        return Circular;
    }
}
```

这段代码包含了最有趣的部分。在大多数语言中，带有小数点的数字是不准确的。实际上，在大多数情况下，如果 a 和 b 是浮点数或双精度浮点数，它们可能永远不会相等。因此，在比较这样的数字时，你必须考虑精度。

在这段代码中，你定义了边距，以便在数字被认为相等时有一个可接受的比较精度范围（例如，0.001 和 0.0011 在这种情况下将是相等的，因为边距是 0.01）。之后，你可以进行常规比较，并返回具有最大面积的部分的值。

1.  现在，创建`Main`方法，如下所示：

```cpp
public static void Main()
{ 
    string compare1 = Solve(new Rectangle[0], new Circle[0]);
    string compare2 = Solve(new[] { new Rectangle(1, 5)}, new Circle[0]);
    string compare3 = Solve(new Rectangle[0], new[] { new Circle(1) });
    string compare4 = Solve(new []
    {
        new Rectangle(5.0, 2.1), 
        new Rectangle(3, 3), 
    }, new[]
    {
        new Circle(1),
        new Circle(10), 
    });

    Console.WriteLine($"compare1 is {compare1}, " +
                      $"compare2 is {compare2}, " +
                      $"compare3 is {compare3}, " +
                      $"compare4 is {compare4}.");
}
```

在这里，创建了四组形状进行比较。`compare1`有两个空的部分，意味着它们应该是相等的。`compare2`有一个矩形和没有圆，所以矩形更大。`compare3`有一个圆和没有矩形，所以圆更大。最后，`compare4`既有矩形又有圆，但圆的总面积更大。你在`Console.WriteLine`中使用了字符串插值来打印结果。

1.  运行代码。你应该看到以下内容被打印到控制台上：

```cpp
compare1 is equal, compare2 is rectangular, compare3 is circular, compare4 is circular.
```

注意

你可以在[`packt.link/tfDCw`](https://packt.link/tfDCw)找到此练习中使用的代码。

如果没有对象会怎样？在这种情况下，部分将由什么组成？对于一个圆来说，可能只需传递半径，但对于矩形，你需要传递另一个共线数组，其中包括宽度和高度。

面向对象的代码非常适合将类似的数据和逻辑分组在一个外壳下，也就是一个类，并传递这些类对象。通过与类的简单交互，你可以简化复杂的逻辑。

现在，你将了解面向对象编程的四大支柱。

# 面向对象编程的四大支柱

高效的代码应该易于理解和维护，而面向对象编程致力于实现这种简单性。面向对象设计的整个概念基于四个主要原则，也被称为面向对象编程的四大支柱。

## 封装

面向对象编程的第一个支柱是封装。它定义了数据和行为之间的关系，放置在同一个外壳中，也就是一个类。它指的是只公开必要的内容，隐藏其他所有内容。当你考虑封装时，考虑一下对于你的代码来说安全性的重要性：如果你泄露了密码、返回了机密数据或者公开了 API 密钥会怎么样？鲁莽行事往往会导致难以修复的损害。

安全性不仅仅限于防止恶意意图，还包括防止手动错误。人们往往会犯错误。事实上，可供选择的选项越多，他们犯错的可能性就越大。封装有助于解决这个问题，因为你可以简单地限制将使用代码的人可用的选项数量。

你应该默认阻止所有访问，只有在必要时才授予显式访问权限。例如，考虑一个简化的`LoginService`类：

```cpp
public class LoginService
{
    // Could be a dictionary, but we will use a simplified example.
    private string[] _usernames;
    private string[] _passwords;

    public bool Login(string username, string password)
    {
        // Do a password lookup based on username
        bool isLoggedIn = true;
        return isLoggedIn;
    }
}
```

这个类有两个`private`字段：`_usernames`和`_passwords`。这里需要注意的关键点是，密码和用户名都不对外公开，但你仍然可以通过`Login`方法公开足够的逻辑来实现所需的功能。

注意

你可以在[`packt.link/6SO7a`](https://packt.link/6SO7a)找到此示例中使用的代码。

## 继承

一个警察可以逮捕某人，邮递员递送邮件，老师教授一个或多个科目。他们每个人都执行着完全不同的职责，但他们有什么共同之处呢？在现实世界的背景下，他们都是人类。他们都有姓名、年龄、身高和体重。如果你要对每个人建模，你需要创建三个类。这些类中的每一个看起来都是一样的，除了每个类有一个独特的方法。你如何在代码中表达他们都是人类呢？

解决这个问题的关键是继承。它允许你从父类中获取所有属性并将它们传递给子类。继承还定义了一种 is-a 关系。警察、邮递员和老师都是人类，所以你可以使用继承。现在你要把这些写成代码。

1.  创建一个“人”类，其中包含“姓名”、“年龄”、“体重”和“身高”字段：

```cpp
public class Human
{
    public string Name { get; }
    public int Age { get; }
    public float Weight { get; }
    public float Height { get; }

    public Human(string name, int age, float weight, float height)
    {
        Name = name;
        Age = age;
        Weight = weight;
        Height = height;
    }
}
```

1.  邮递员是一个人。因此，“邮递员”类应该拥有“人”类拥有的一切，但除此之外，它还应该具有能够投递邮件的附加功能。编写代码如下：

```cpp
public class Mailman : Human
{
    public Mailman(string name, int age, float weight, float height) : base(name, age, weight, height)
    {
    }

    public void DeliverMail(Mail mail)
    {
       // Delivering Mail...
    }
}
```

现在，仔细看看“邮递员”类。编写`class Mailman : Human`意味着“邮递员”继承自“人”。这意味着“邮递员”继承了“人”的所有属性和方法。你还可以看到一个新关键字，`base`。这个关键字用于告诉在创建“邮递员”时将使用哪个父构造函数；在这种情况下是“人”。

1.  接下来，创建一个名为`Mail`的类来表示邮件，其中包含一个字段，用于将消息传递到地址：

```cpp
public class Mail
{
   public string Message { get; }
   public string Address { get; }

   public Mail(string message, string address)
   {
       Message = message;
       Address = address;
   }
}
```

创建“邮递员”对象与创建不使用继承的类的对象没有任何不同。

1.  创建“邮递员”和“邮件”变量，并告诉“邮递员”投递邮件如下：

```cpp
var mailman = new Mailman("Thomas", 29, 78.5f, 190.11f);
var mail = new Mail("Hello", "Somewhere far far way");
mailman.DeliverMail(mail);
```

注意

你可以在[`packt.link/w1bbf`](https://packt.link/w1bbf)找到此示例使用的代码。

在前面的代码片段中，你创建了“邮递员”和“邮件”变量。然后，你告诉“邮递员”投递“邮件”。

通常，在定义子构造函数时必须提供基础构造函数。唯一的例外是当父类有一个无参数的构造函数时。如果基础构造函数不带参数，则使用基础构造函数的子构造函数将是多余的，因此可以忽略。例如，考虑以下代码片段：

```cpp
Public class A
{
}
Public class B : A
{
}
```

`A`没有自定义构造函数，因此实现`B`也不需要自定义构造函数。

在 C#中，只能继承一个类；但是，可以进行多级深度继承。例如，你可以为“邮递员”命名为“区域邮递员”的子类，该子类将负责一个地区。通过这种方式，你可以更深入地进行继承，为“区域邮递员”创建另一个子类，称为“区域结算邮递员”，然后是“欧洲区域结算邮递员”，依此类推。

在使用继承时，重要的是要知道即使一切都被继承，也不是一切都可见。就像以前一样，`public`成员只能从父类中访问。但是，在 C#中，有一个特殊的修饰符，名为`protected`，它的作用类似于`private`修饰符。它允许子类访问`protected`成员（就像`public`成员一样），但阻止它们从类的外部访问（就像`private`一样）。

几十年前，继承曾经是许多问题的答案和代码重用的关键。然而，随着时间的推移，人们发现使用继承是有代价的，即耦合。当应用继承时，你将子类与父类耦合在一起。深度继承将类的范围一直堆叠到子类。继承越深，范围就越深。应该避免深度继承（两个或更多级深度），原因与避免全局变量相同——很难知道来自何处，很难控制状态变化。这反过来使得代码难以维护。

没有人想要编写重复的代码，但是替代方案是什么？答案是组合。就像计算机由不同的部分组成一样，代码也应该由不同的部分组成。例如，想象一下你正在开发一个 2D 游戏，它有一个`Tile`对象。一些瓷砖包含陷阱，一些瓷砖会移动。使用继承，你可以这样编写代码：

```cpp
class Tile
{
}
class MovingTile : Tile
{
    public void Move() {}
}
class TrapTile : Tile
{
    public void Damage() {}
}
//class MovingTrapTile : ?
```

这种方法在面对更复杂的要求时效果很好。如果有一种瓷砖既可以是陷阱又可以移动怎么办？您应该从移动瓷砖继承并在那里重写`TrapTile`的功能吗？您可以同时继承吗？正如您所见，您不能一次继承多个类，因此，如果您要使用继承来实现这一点，您将被迫复杂化情况，并重写一些代码。相反，您可以考虑不同瓷砖包含的内容。`TrapTile`有一个陷阱。`MovingTile`有一个电机。

两者都代表瓷砖，但它们各自具有的额外功能应来自不同的组件，而不是子类。如果您想要将其作为基于组合的方法，您需要进行相当大的重构。

要解决这个问题，保持`Tile`类不变：

```cpp
class Tile
{
}
```

现在，添加两个组件——`Motor`和`Trap`类。这些组件作为逻辑提供者。目前，它们什么也不做：

```cpp
class Motor
{
    public void Move() { }
}
class Trap
{
    public void Damage() { }
}
```

注意

您可以在[`packt.link/espfn`](https://packt.link/espfn)找到此示例使用的代码。

接下来，您定义一个`MovingTile`类，它有一个名为`_motor`的单个组件。在组合中，组件很少动态变化。您不应该暴露类的内部，因此应用`private readonly`修饰符。组件本身可以有一个子类或更改，因此不应该从构造函数中创建。相反，它应该作为参数传递（请参阅突出显示的代码）：

```cpp
class MovingTile : Tile
{
    private readonly Motor _motor;

    public MovingTile(Motor motor)
    {
        _motor = motor;
    } 

    public void Move()
    {
        _motor.Move();
    }
}
```

请注意，`Move`方法现在调用了`_motor.Move()`。这就是组合的本质；持有组合的类通常本身不做任何事情。它只是将逻辑的调用委托给它的组件。实际上，即使这只是一个示例类，一个真正的游戏类看起来也会非常类似。

您将为`TrapTile`做同样的事情，只是它将包含一个`Trap`组件，而不是`Motor`：

```cpp
class TrapTile : Tile
{
    private readonly Trap _trap;

    public TrapTile(Trap trap)
    {
        _trap = trap;
    }

    public void Damage()
    {
        _trap.Damage();
    }
}
```

最后，是时候创建`MovingTrapTile`类了。它有两个组件，分别为`Move`和`Damage`方法提供逻辑。同样，这两个方法作为参数传递给构造函数：

```cpp
class MovingTrapTile : Tile
{
    private readonly Motor _motor;
    private readonly Trap _trap;

    public MovingTrapTile(Motor motor, Trap trap)
    {
        _motor = motor;
        _trap = trap;
    }
    public void Move()
    {
        _motor.Move();
    }
    public void Damage()
    {
        _trap.Damage();
    }
}
```

注意

您可以在[`packt.link/SX4qG`](https://packt.link/SX4qG)找到此示例使用的代码。

这个类似乎重复了另一个类的一些代码，但重复是微不足道的，而好处是非常值得的。毕竟，最大的逻辑块来自组件本身，重复的字段或调用并不重要。

您可能已经注意到，尽管没有将`Tile`作为其他类的组件提取出来，但您仍然继承了它。这是因为`Tile`是所有继承它的类的本质。无论瓷砖是什么类型，它仍然是一种瓷砖。继承是面向对象编程的第二支柱。它是强大且有用的。然而，要正确地使用继承可能很困难，因为为了可维护性，它确实需要非常清晰和合乎逻辑。在选择是否应该使用继承时，请考虑以下因素：

+   不深（理想情况下是单层）。

+   逻辑的（是一个关系，就像您在瓷砖示例中看到的）。

+   稳定且极不可能在未来类之间的关系发生变化；不会经常修改。

+   纯添加（子类不应使用父类成员，除了构造函数）。

如果这些规则中的任何一个被打破，建议使用组合而不是继承。

## 多态性

面向对象编程的第三支柱是多态性。要理解这一支柱，有必要看一下这个词的含义。`Thomas`。`Thomas`既是一个人，也是一个邮递员。`Mailman`是`Thomas`的专用形式，`Human`是`Thomas`的通用形式。然而，您可以通过这两种形式与`Thomas`进行交互。

如果您不知道每个人的工作，可以使用一个`abstract`类。

`abstract`类是不完整类的同义词。这意味着它不能被初始化。这也意味着如果您用`abstract`关键字标记它们，它的一些方法可能没有实现。您可以为`Human`类实现如下：

```cpp
public abstract class Human
{
    public string Name { get; }

    protected Human(string name)
    {
        Name = name;
    }

    public abstract void Work();
}
```

在这里，您创建了一个抽象（不完整的）`Human`类。与之前的唯一区别是，您将`abstract`关键字应用于类，并添加了一个新的`abstract`方法`public abstract void Work()`。您还将构造函数更改为受保护的，以便只能从子类访问。这是因为如果您不能创建一个`abstract`类，那么将它设为`public`就不再有意义；您不能调用`public`构造函数。逻辑上讲，这意味着`Human`类本身没有意义，只有在其他地方实现了`Work`方法后（也就是在子类中）才有意义。

现在，您将更新`Mailman`类。它并没有太多变化；只是增加了一个额外的方法，即`Work()`。要为抽象方法提供实现，必须使用`override`关键字。一般来说，这个关键字用于在子类中更改现有方法的实现。稍后您将详细探讨这一点：

```cpp
public override void Work()
{
    Console.WriteLine("A mailman is delivering mails.");
}
```

如果您为这个类创建一个新对象并调用`Work`方法，它将在控制台上打印`"A mailman is delivering mails."`。为了全面了解多态，现在您将创建另一个类`Teacher`：

```cpp
public class Teacher : Human
{
    public Teacher(string name, int age, float weight, float height) : base(name, age, weight, height)
    {
    }

    public override void Work()
    {
        Console.WriteLine("A teacher is teaching.");
    }
}
```

这个类几乎与`Mailman`相同；但是提供了`Work`方法的不同实现。因此，您有两个类以两种不同的方式执行相同的操作。调用同名方法，但获得不同行为的行为称为多态。

您已经了解了方法重载（不要与重写混淆），这是指具有相同名称但不同输入的方法。这称为静态多态，它发生在编译时。以下是一个例子：

```cpp
public class Person
{
    public void Say()
    {
        Console.WriteLine("Hello");
    }

    public void Say(string words)
    {
        Console.WriteLine(words);
    }
}
```

`Person`类有两个同名方法 Say。一个不带参数，另一个带一个字符串作为参数。根据传递的参数，将调用方法的不同实现。如果什么都不传，将打印`"Hello"`。否则，将打印您传递的单词。

在面向对象编程的上下文中，多态被称为动态多态，它发生在运行时。在本章的其余部分，多态应该被解释为动态多态。

### 多态的好处是什么？

老师是一个人，老师的工作方式是教书。这与邮递员不同，但老师也有姓名、年龄、体重和身高，就像邮递员一样。多态允许您以相同的方式与两者交互，而不考虑它们的专业形式。最好的方法是将两者存储在`humans`值数组中并让它们工作：

```cpp
Mailman mailman = new Mailman("Thomas", 29, 78.5f, 190.11f);
Teacher teacher = new Teacher("Gareth", 35, 100.5f, 186.49f);
// Specialized types can be stored as their generalized forms.
Human[] humans = {mailman, teacher};
// Interacting with different human types
// as if they were the same type- polymorphism.
foreach (var human in humans)
{
    human.Work();
}
```

此代码将在控制台中打印以下内容：

```cpp
A mailman is delivering mails.
A teacher is teaching.
```

注意

您可以在[`packt.link/ovqru`](https://packt.link/ovqru)找到此示例使用的代码。

这段代码展示了多态。您将`Mailman`和`Teacher`都视为`Human`，并为两者都实现了`Work`方法。结果是每种情况下的不同行为。这里需要注意的重要一点是，您不必关心实现`Human`的具体细节来实现`Work`。

如果没有多态，您将需要编写基于对象的确切类型的`if`语句来找到它应该使用的行为：

```cpp
foreach (var human in humans)
{
    Type humanType = human.GetType();
    if (humanType == typeof(Mailman))
    {
        Console.WriteLine("Mailman is working...");
    }
    else
    {
        Console.WriteLine("Teaching");
    }
}
```

如您所见，这更加复杂且难以理解。当您遇到许多`if`语句的情况时，请记住这个例子。多态可以通过将每个分支的代码移动到子类中并简化交互来消除所有分支代码的负担。

如果您想要打印有关一个人的一些信息，可以考虑以下代码：

```cpp
Human[] humans = {mailman, teacher};
foreach (var human in humans)
{
    Console.WriteLine(human);
}
```

运行此代码将导致对象类型名称被打印到控制台：

```cpp
Chapter02.Examples.Professions.Mailman
Chapter02.Examples.Professions.Teacher
```

在 C#中，一切都源自`System.Object`类，因此 C#中的每种类型都有一个名为`ToString()`的方法。每种类型都有其自己的此方法的实现，这是多态性的另一个例子，在 C#中广泛使用。

注意

`ToString()`与`Work()`不同之处在于它提供了一个默认实现。您可以使用`virtual`关键字来实现这一点，这将在本章后面详细介绍。从子类的角度来看，使用`virtual`或`abstract`关键字是相同的。如果要更改或提供行为，您将覆盖该方法。

在以下片段中，`Human`对象被赋予了`ToString()`方法的自定义实现：

```cpp
public override string ToString()
{
    return $"{nameof(Name)}: {Name}," +
           $"{nameof(Age)}: {Age}," +
           $"{nameof(Weight)}: {Weight}," +
           $"{nameof(Height)}: {Height}";
}
```

尝试在同一个 foreach 循环中打印有关人类的信息将导致以下输出：

```cpp
Name: Thomas,Age: 29,Weight: 78.5,Height: 190.11
Name: Gareth,Age: 35,Weight: 100.5,Height: 186.49
```

注意

您可以在[`packt.link/EGDkC`](https://packt.link/EGDkC)找到此示例使用的代码。

多态性是在处理缺少类型信息时使用不同底层行为的最佳方法之一。

## 抽象

面向对象编程的最后一个支柱是抽象。有人说面向对象编程只有三个支柱，因为抽象并没有真正引入太多新内容。抽象鼓励您隐藏实现细节并简化对象之间的交互。每当您只需要通用形式的功能时，您不应该依赖于其实现。

抽象可以通过人们如何与计算机交互的示例来说明。当您打开计算机时，内部电路会发生什么？大多数人可能不知道，这没关系。如果您只需要使用某些功能，您不需要了解内部工作原理。您只需要知道您可以通过按按钮打开和关闭计算机，所有复杂的细节都被隐藏起来。抽象对其他三个支柱几乎没有增加新内容，因为它反映了它们的每一个。**抽象类似于封装**，因为它隐藏了不必要的细节以简化交互。它也类似于多态性，因为它可以与对象交互而不知道它们的确切类型。最后，继承只是创建抽象的一种方式。

在创建函数时，不需要提供实现类型传递的不必要细节。以下示例说明了这个问题。您需要创建一个进度条。它应该跟踪当前进度，并应该增加进度直到某个点。您可以创建一个带有设置器和获取器的基本类，如下所示：

```cpp
public class ProgressBar
{
    public float Current { get; set; }
    public float Max { get; }

    public ProgressBar(float current, float max)
    {
        Max = max;
        Current = current;
    }
}
```

以下代码演示了如何初始化一个从`0`进度开始并增加到`100`的进度条。代码的其余部分说明了当您想要将新进度设置为 120 时会发生什么。进度不能超过`Max`，因此，如果超过`bar.Max`，它应该保持在`bar.Max`。否则，您可以使用您设置的值更新新的进度。最后，您需要检查进度是否完成（达到`Max`值）。为此，您将比较增量与允许的误差容限（`0.0001`）。如果进度条接近容限，进度条就完成了。因此，更新进度可能如下所示：

```cpp
var bar = new ProgressBar(0, 100);
var newProgress = 120;
if (newProgress > bar.Max)
{
    bar.Current = bar.Max;
}
else
{
    bar.Current = newProgress;
}

const double tolerance = 0.0001;
var isComplete = Math.Abs(bar.Max - bar.Current) < tolerance;
```

这段代码做了被要求的事情，但对于一个函数来说需要很多细节。想象一下，如果你需要在其他代码中使用它，你需要再次执行相同的检查。换句话说，实现起来很容易，但消耗起来很复杂。类本身很少。一个很明显的指标是你一直在调用对象，而不是在类内部做一些事情。公开地说，通过忘记检查进度的`Max`值并将其设置为较高或负值，很容易破坏对象状态。你写的代码具有低内聚性，因为要改变`ProgressBar`，你需要在类的外部而不是类内部进行。你需要创建一个更好的抽象。

考虑以下片段：

```cpp
public class ProgressBar
{
    private const float Tolerance = 0.001f;

    private float _current;
    public float Current
    {
        get => _current;
        set
        {
            if (value >= Max)
            {
                _current = Max;
            }
            else if (value < 0)
            {
                _current = 0;
            }
            else
            {
                _current = value;
            }
        }
    }
```

通过这段代码，你隐藏了繁琐的细节。当涉及到更新进度和定义容差时，这取决于`ProgressBar`类来决定。在重构后的代码中，你有一个属性`Current`，有一个后备字段`_current`来存储进度。属性的 setter 检查进度是否超过最大值，如果是，它将不允许将`_current`的值设置为更高的值，`=`。它也不能是负数，因为在这种情况下，值将被调整为`0`。最后，如果它既不是负数也不超过最大值，那么你可以将`_current`设置为你传递的任何值。

显然，这段代码使与`ProgressBar`类的交互变得简单得多：

```cpp
var bar = new ProgressBar(0, 100);
bar.Current = 120;
bool isComplete = bar.IsComplete;
```

你不能破坏任何东西；你没有任何额外的选择，你所能做的一切都是通过最小化的方法定义的。当你被要求实现一个功能时，不建议做比要求更多的事情。尽量做到最小化和简单化，因为这是有效代码的关键。

请记住，良好抽象的代码对读者充满同理心。仅仅因为今天很容易实现一个类或一个函数，你不应该忘记明天。需求会改变，实现会改变，但结构应该保持稳定，否则你的代码很容易出错。

注意

您可以在[`packt.link/U126i`](https://packt.link/U126i)找到此示例使用的代码。在 GitHub 上提供的代码分为两个对比示例——`ProgressBarGood`和`ProgressBarBad`。这两个代码都是简单的`ProgressBar`，但它们被命名为不同的名称以避免歧义。

## 接口

之前提到过，继承不是设计代码的正确方式。然而，你希望有一个高效的抽象，同时支持多态性，以及尽可能少的耦合。如果你想要有机器人或蚂蚁工人怎么办？它们没有名字。身高和体重等信息是无关紧要的。从`Human`类继承就没有多大意义了。使用接口可以解决这个难题。

按照惯例，在 C#中，接口的命名以字母`I`开头，后面跟着它们的实际名称。接口是一个合同，规定了一个类能做什么。它没有任何实现。它只为实现它的每个类定义行为。现在，你将使用接口重构人类的示例。

`Human`类的对象能做什么？它可以工作。谁或什么能工作？工作者。现在，考虑以下片段：

```cpp
public interface IWorker
{
    void Work();
}
```

注意

接口`Work`方法的访问修饰符与接口相同，即`public`。

蚂蚁不是人类，但它也可以工作。通过接口，将蚂蚁抽象为工作者是很简单的：

```cpp
public class Ant : IWorker
{
    public void Work()
    {
        Console.WriteLine("Ant is working hard.");
    }
}
```

同样，机器人不是人类，但它也可以工作：

```cpp
public class Robot : IWorker
{
    public void Work()
    {
        Console.WriteLine("Beep boop- I am working.");
    }
}
```

如果你参考`Human`类，你可以将其定义为`public abstract class Human : IWorker`。这可以理解为：`Human`类实现了`IWorker`接口。

在下一个片段中，`Mailman`继承了`Human`类，该类实现了`IWorker`接口：

```cpp
public class Mailman : Human
{
    public Mailman(string name, int age, float weight, float height) : base(name, age, weight, height)
    {
    }

    public void DeliverMail(Mail mail)
    {
        // Delivering Mail...
    }

    public override void Work()
    {
        Console.WriteLine("Mailman is working...");
    }
}
```

如果一个子类继承了一个实现了一些接口的父类，那么子类也将能够默认实现相同的接口。然而，`Human`是一个抽象类，你必须为`abstract void Work`方法提供实现。

如果有人问人类、蚂蚁和机器人有什么共同之处，你可以说他们都能工作。你可以模拟这种情况，如下所示：

```cpp
IWorker human = new Mailman("Thomas", 29, 78.5f, 190.11f);
IWorker ant = new Ant();
IWorker robot = new Robot();

IWorker[] workers = {human, ant, robot};
foreach (var worker in workers)
{
    worker.Work();
}
```

这将在控制台上打印如下内容：

```cpp
Mailman is working...
Ant is working hard.
Beep boop- I am working.
```

注意

你可以在[`packt.link/FE2ag`](https://packt.link/FE2ag)找到示例中使用的代码。

C#不支持多重继承。然而，可以实现多个接口。实现多个接口不算是多重继承。例如，要实现一个`Drone`类，你可以添加一个`IFlyer`接口：

```cpp
public interface IFlyer
{
    void Fly();
}
```

`无人机`是一种可以执行一些工作的飞行物体；因此，它可以表示如下：

```cpp
public class Drone : IFlyer, IWorker
{
    public void Fly()
    {
        Console.WriteLine("Flying");
    }

    public void Work()
    {
        Console.WriteLine("Working");
    }
}
```

用逗号分隔列出多个接口意味着类实现了每一个接口。你可以组合任意数量的接口，但尽量不要过度。有时，两个接口的组合构成一个逻辑抽象。如果每个无人机都能飞行并且能做一些工作，那么你可以在代码中写出来，如下所示：

```cpp
public interface IDrone : IWorker, IFlyer
{
}
```

`Drone`类变得简化为`public class Drone : IDrone`。

还可以将接口与基类混合使用（但不超过一个基类）。如果你想表示一只会飞的蚂蚁，你可以编写以下代码：

```cpp
public class FlyingAnt : Ant, IFlyer
{
    public void Fly()
    {
        Console.WriteLine("Flying");
    }
}
```

接口无疑是最好的抽象，因为依赖于它不会强迫你依赖于任何实现细节。所需的只是已经定义的逻辑概念。实现容易改变，但类之间关系的逻辑不会改变。

如果一个接口定义了一个类可以做什么，那么也可以定义一个用于共同数据的契约吗？当然可以。一个接口包含行为，因此它也可以包含属性，因为它们定义了 setter 和 getter 行为。例如，你应该能够追踪无人机，为此，它应该是可识别的，也就是说，它需要有一个 ID。这可以编码如下：

```cpp
public interface IIdentifiable
{
    long Id { get; }
}
public interface IDrone : IWorker, IFlyer 
{
}
```

在现代软件开发中，程序员每天都会使用一些复杂的低级细节。然而，他们经常在不知情的情况下这样做。如果你想创建一个易于理解的代码库，其中包含大量逻辑和易于理解的代码，你应该遵循这些抽象原则：

+   保持简单和小。

+   不要依赖于细节。

+   隐藏复杂性。

+   只暴露必要的内容。

通过这个练习，你将了解面向对象编程的功能。

## 练习 2.03：在后院铺地板

一个建筑师正在用马赛克铺地，他需要覆盖 x 平方米的区域。你有一些剩下的瓷砖，要么是矩形的，要么是圆形的。在这个练习中，你需要找出，如果你打碎瓷砖来完全填满它们所占据的区域，瓷砖是否可以完全填满马赛克。

你将编写一个程序，如果马赛克可以用瓷砖覆盖，则打印`true`，如果不能，则打印`false`。执行以下步骤来完成：

1.  创建一个名为`IShape`的接口，带有一个`Area`属性：

```cpp
public interface IShape
{
    double Area { get; }
}
```

这是一个只读属性。请注意，属性是一个方法，所以在接口中拥有它是可以的。

1.  创建一个名为`Rectangle`的类，带有宽度和高度以及一个用于计算面积的方法，名为`Area`。为此实现一个`IShape`接口，如下所示的代码所示：

```cpp
Rectangle.cs
public class Rectangle : IShape
{
    private readonly double _width;
    private readonly double _height;

    public double Area
    {
        get
        {
            return _width * _height;
        }
    } 

    public Rectangle(double width, double height)
    {
```

```cpp
You can find the complete code here: https://packt.link/zSquP.
```

唯一需要做的就是计算面积。因此，只有`Area`属性是`public`。你的接口需要实现一个 getter `Area`属性，通过将`width`和`height`相乘来实现。

1.  创建一个带有`半径`和`Area`计算的`Circle`类，它还实现了`IShape`接口：

```cpp
public class Circle : IShape
{
    Private readonly double _radius;

    public Circle(double radius)
    {
        _radius = radius;
    }

    public double Area
    {
        get { return Math.PI * _radius * _radius; }
    }
}
```

1.  创建一个名为`IsEnough`的方法的骨架`Solution`类，如下所示：

```cpp
public static class Solution
{
        public static bool IsEnough(double mosaicArea, IShape[] tiles)
        {
   }
}
```

类和方法都只是实现的占位符。该类是`static`，因为它将用作演示，不需要具有状态。`IsEnough`方法接受所需的`mosaicArea`、一组瓷砖对象，并返回瓷砖占据的总面积是否足以覆盖马赛克。

1.  在`IsEnough`方法内部，使用`for`循环来计算`totalArea`。然后，返回总面积是否覆盖了马赛克区域：

```cpp
            double totalArea = 0;
            foreach (var tile in tiles)
            {
                totalArea += tile.Area;
            }
            const double tolerance = 0.0001;
            return totalArea - mosaicArea >= -tolerance;
       }
```

1.  在`Solution`类内部创建一个演示。添加几组不同形状，如下所示：

```cpp
public static void Main()
{
    var isEnough1 = IsEnough(0, new IShape[0]);
    var isEnough2 = IsEnough(1, new[] { new Rectangle(1, 1) });
    var isEnough3 = IsEnough(100, new IShape[] { new Circle(5) });
    var isEnough4 = IsEnough(5, new IShape[]
    {
        new Rectangle(1, 1), new Circle(1), new Rectangle(1.4,1)
    });

    Console.WriteLine($"IsEnough1 = {isEnough1}, " +
                      $"IsEnough2 = {isEnough2}, " +
                      $"IsEnough3 = {isEnough3}, " +
                      $"IsEnough4 = {isEnough4}.");
}
```

在这里，您使用了四个例子。当要覆盖的面积为`0`时，无论您传递什么形状，都足够了。当要覆盖的面积为`1`时，面积为`1x1`的矩形刚好足够。当面积为`100`时，半径为`5`的圆不够。最后，对于第四个例子，三个形状占据的面积相加，即面积为`1x1`的矩形、半径为`1`的圆和面积为`1.4x1`的第二个矩形。总面积为`5`，小于这三个形状的组合面积。

1.  运行演示。您应该在屏幕上看到以下输出：

```cpp
IsEnough1 = True, IsEnough2 = True, IsEnough3 = False, IsEnough4 = False.
```

注意

您可以在[`packt.link/EODE6`](https://packt.link/EODE6)找到用于此练习的代码。

这个练习与*练习 2.02*非常相似。然而，尽管任务更复杂，但代码比上一个任务少。通过使用面向对象编程的支柱，您能够为复杂问题创建简单的解决方案。您能够创建依赖于抽象的函数，而不是为不同类型创建重载。因此，面向对象编程是一个强大的工具，这只是冰山一角。

每个人都可以编写能够工作的代码，但编写能够持续数十年并且易于理解的代码是困难的。因此，了解面向对象编程中的最佳实践是至关重要的。

# 面向对象编程中的 SOLID 原则

SOLID 原则是面向对象编程的最佳实践。SOLID 是五个原则的首字母缩写，即单一职责、开闭原则、里氏替换、接口隔离和依赖反转。您将不会详细探讨每一个原则。

## 单一职责原则

函数、类、项目和整个系统随着时间的推移而发生变化。每一次变化都有可能是破坏性的，因此您应该限制同时发生太多变化的风险。换句话说，代码块的一部分应该只有一个改变的原因。

对于一个函数来说，这意味着它应该只做一件事，并且没有副作用。实际上，这意味着一个函数应该要么改变，要么获取某些东西，但不能两者兼有。这也意味着负责高级事务的函数不应该与执行低级事务的函数混合在一起。低级是指实现与硬件的交互和使用原语。高级侧重于软件构建块或服务的组合。在谈论高级和低级函数时，通常称之为依赖链。如果函数 A 调用函数 B，那么 A 被认为比 B 更高级。一个函数不应该实现多个事情；它应该调用实现单一事情的其他函数。对于这一点的一般指导原则是，如果您认为可以将代码拆分成不同的函数，那么在大多数情况下，您应该这样做。

对于类来说，这意味着您应该使它们小而相互隔离。一个高效的类的例子是`File`类，它可以读取和写入。如果它同时实现了读取和写入，它将因为两个原因（读取和写入）而发生变化。

```cpp
public class File
{
    public string Read(string filePath)
    {
        // implementation how to read file contents
        // complex logic
        return "";
    }

    public void Write(string filePath, string content)
    {
        // implementation how to append content to an existing file
        // complex logic
    }
}
```

因此，为了符合这一原则，您可以将读取代码拆分为一个名为`Reader`的类，将写入代码拆分为一个名为`Writer`的类，如下所示：

```cpp
public class Reader
{
    public string Read(string filePath)
    {
        // implementation how to read file contents
        // complex logic
        return "";
    }
}
public class Writer
{
    public void Write(string filePath, string content)
    {
        // implementation how to append content to an existing file
        // complex logic
    }
}
```

现在，`File`类不再实现读取和写入，而是简单地由读取器和写入器组成：

```cpp
public class File
{
    private readonly Reader _reader;
    private readonly Writer _writer;

    public File()
    {
        _reader = new Reader();
        _writer = new Writer();
    }  

    public string Read(string filePath) => _reader.Read(filePath);
    public void Write(string filePath, string content) => _writer.Write(filePath, content);
}
```

注意

您可以在[`packt.link/PBppV`](https://packt.link/PBppV)找到此示例使用的代码。

这可能会令人困惑，因为类的本质实质上保持不变。然而，现在它只是消耗一个组件，而不是负责实现它。一个高级别的类（`File`）只是为了解如何使用低级别的类（`Reader`、`Writer`）而添加了上下文。

对于一个模块（库），这意味着你应该努力不引入依赖关系，这些依赖关系可能超出消费者的需求。例如，如果你正在使用一个用于记录日志的库，它不应该带有某些特定于第三方记录提供者的实现。

对于一个子系统来说，这意味着不同的系统应尽可能地隔离。如果两个（较低级别）系统需要通信，它们可以直接相互调用。一个考虑（不是强制性的）是有一个第三个系统（较高级别）用于协调。系统还应该通过边界（比如指定通信参数的合同）进行分离，隐藏所有细节。如果一个子系统是一个大型库集合，它应该有一个接口来暴露它可以做什么。如果一个子系统是一个网络服务，它应该是一个端点集合。在任何情况下，子系统的合同应该只提供客户可能需要的方法。

有时，原则被过分强调，类被分割得太多，以至于进行更改需要改变多个地方。它确实符合原则，因为一个类只有一个改变的原因，但在这种情况下，多个类将因同样的原因而改变。例如，假设你有两个类：`Merchandise`和`TaxCalculator`。`Merchandise`类有`Name`、`Price`和`Vat`字段：

```cpp
public class Merchandise
{
    public string Name { get; set; }
    public decimal Price { get; set; }
    // VAT on top in %
    public decimal Vat { get; set; }
}
```

接下来，您将创建`TaxCalculator`类。`vat`是以百分比计量的，所以实际要支付的价格将是原始价格加上`vat`：

```cpp
public static class TaxCalculator
{
    public static decimal CalculateNextPrice(decimal price, decimal vat)
    {
        return price * (1 + vat / 100);
    }
}
```

如果计算价格的功能移动到`Merchandise`类，会发生什么变化？您仍然可以执行所需的操作。这里有两个关键点：

+   操作本身很简单。

+   此外，税收计算器需要的一切都来自`Merchandise`类。

如果一个类可以自己实现逻辑，只要它是自包含的（不涉及额外的组件），通常应该这样做。因此，代码的适当版本如下：

```cpp
public class Merchandise
{
    public string Name { get; set; }
    public decimal Price { get; set; }
    // VAT on top in %
    public decimal Vat { get; set; }
    public decimal NetPrice => Price * (1 + Vat / 100);
}
```

这段代码将`NetPrice`计算移到`Merchandise`类中，并删除了`TaxCalculator`类。

注意

单一责任原则（SRP）可以用几个词来概括：**拆分它**。您可以在[`packt.link/lWxNO`](https://packt.link/lWxNO)找到此示例使用的代码。

## 开闭原则

如前所述，代码中的每一次更改都可能是破坏性的。为了避免这种情况，通常最好的方法不是更改现有的代码，而是编写新的代码。每个软件实体都应该有一个扩展点，通过这个扩展点应该引入更改。然而，在完成这些更改之后，不应该再干预软件实体。开闭原则（OCP）很难实现，需要大量的实践，但好处（最小数量的破坏性更改）是非常值得的。

如果一个多步算法不改变，但它的各个步骤可以改变，你应该将它拆分成几个函数。对于一个单独的步骤的更改将不再影响整个算法，而只是影响到那一步。这种减少单个类或函数改变原因的做法正是 OCP 的全部内容。

注意

您可以在[`social.technet.microsoft.com/wiki/contents/articles/18062.open-closed-principle-ocp.aspx`](https://social.technet.microsoft.com/wiki/contents/articles/18062.open-closed-principle-ocp.aspx)找到有关 OCP 的更多信息。

另一个你可能想要实现这个原则的例子是一个函数在代码中使用特定值的组合。这被称为硬编码，通常被认为是一种低效的做法。为了使它适用于新的值，你可能会想要创建一个新的函数，但是通过简单地移除硬编码部分并通过函数参数公开它，你可以使它变得可扩展。然而，当你有已知是固定且不会改变的变量时，硬编码是可以的，但它们应该被标记为常量。

以前，你创建了一个带有两个依赖项的文件类——`Reader`和`Writer`。这些依赖是硬编码的，并且没有扩展点。修复这个问题将涉及两件事。首先，为`Reader`和`Writer`类的方法添加虚拟修饰符：

```cpp
public virtual string Read(string filePath)
public virtual void Write(string filePath, string content)
```

然后，改变`File`类的构造函数，使其接受`Reader`和`Writer`的实例，而不是硬编码依赖关系：

```cpp
public File(Reader reader, Writer writer)
{
    _reader = reader;
    _writer = writer;
}
```

这段代码使你能够覆盖现有的读取器和写入器行为，并用你想要的任何行为替换它，也就是说，`File`类扩展点。

OCP 可以用几个词来概括：**不要改变它，扩展它**。

## 里斯科夫替换

里斯科夫替换原则（LSP）是最直接的原则之一。它简单地意味着子类应该支持父类的所有公共行为。如果你有两个类，`Car`和`CarWreck`，其中一个继承另一个，那么你就违反了这个原则：

```cpp
class Car
{
    public object Body { get; set; }
    public virtual void Move()
    {
        // Moving
    }
}
class CarWreck : Car
{
    public override void Move()
    {
        throw new NotSupportedException("A broken car cannot start.");
    }
}
```

注意

你可以在[`packt.link/6nD76`](https://packt.link/6nD76)找到这个例子的代码。

`Car`和`CarWreck`都有一个`Body`对象。`Car`可以移动，但`CarWreck`呢？它只能停在一个地方。`Move`方法是虚拟的，因为`CarWreck`打算覆盖它以标记它为不支持。如果一个子类不能再支持父类能做的事情，那么它就不应该再继承那个父类。在这种情况下，车祸不是一辆车，它只是一堆废墟。

你如何符合这个原则？你所要做的就是移除继承关系并复制必要的行为和结构。在这种情况下，`CarWreck`仍然有一个`Body`对象，但`Move`方法是不必要的。

```cpp
class CarWreck
{
    public object Body { get; set; }
}
```

代码变化经常发生，有时你可能会无意中使用错误的方法来实现你的目标。有时，你会以一种你认为是灵活的方式耦合代码，结果变成了一个复杂的混乱。不要使用继承作为代码重用的一种方式。保持事物的小而组合它们（再次）而不是试图覆盖现有的行为。在事物可以重用之前，它们应该是可用的。设计简单，你将会得到灵活性。

LSP 可以用几个词来概括：**不要假装**。

注意

你可以在[`www.microsoftpressstore.com/articles/article.aspx?p=2255313`](https://www.microsoftpressstore.com/articles/article.aspx?p=2255313)找到有关 LSP 的更多信息。

## 接口隔离

接口隔离原则是 OCP 的一个特例，但只适用于将公开的合同。记住，你所做的每一个改变都有可能是一个破坏性的改变，这在对合同进行更改时尤其重要。破坏性的改变是低效的，因为它们通常需要多人努力来适应改变。

例如，假设你有一个接口，`IMovableDamageable`：

```cpp
interface IMovableDamageable
{
    void Move(Location location);
    float Hp{get;set;}
}
```

一个单一的接口应该代表一个单一的概念。然而，在这种情况下，它做了两件事：移动和管理`Hp`（生命值）。一个接口本身有两个方法并不是有问题的。然而，在实现只需要接口的一部分的情况下，你被迫创建一个变通方法。

例如，分数文本是不可摧毁的，但你希望它能够被动画化并在场景中移动：

```cpp
class ScoreText : IMovableDamageable
{
    public float Hp 
    { 
        get => throw new NotSupportedException(); 
        set => throw new NotSupportedException(); 
    }

    public void Move(Location location)
    {
        Console.WriteLine($"Moving to {location}");
    }
}

public class Location
{
}
```

注意

这里的重点不是打印位置；只是为了举例说明它的使用。是否打印取决于位置的实现。

再举一个例子，您可能有一个不会移动但可以被摧毁的房子：

```cpp
class House : IMovableDamageable
{
    public float Hp { get; set; }

    public void Move(Location location)
    {
        throw new NotSupportedException();
    }
}
```

在这两种情况下，您通过抛出`NotSupportedException`来解决了问题。但是，不应该给另一个程序员调用从一开始就不起作用的代码的选项。为了解决表示太多概念的问题，您应该将`IMoveableDamageable`接口拆分为`IMoveable`和`IDamageable`：

```cpp
interface IMoveable
{
    void Move(Location location);
}
interface IDamageable
{
    float Hp{get;set;}
}
```

现在，实现可以摆脱不必要的部分：

```cpp
class House : IDamageable
{
    public float Hp { get; set; }
}

class ScoreText : IMovable
{
    public void Move(Location location)
    {
        Console.WriteLine($"Moving to {location}");
    }
}
```

在上述代码中，`Console.WriteLine`将显示命名空间名称和类名。

注意

接口隔离可以总结为**不要强制执行**。您可以在[`packt.link/32mwP`](https://packt.link/32mwP)找到此示例的代码。

## 依赖倒置

大型软件系统可能由数百万个类组成。每个类都是一个小的依赖项，如果不加管理，复杂性可能堆积成无法维护的东西。如果一个低级组件出现故障，就会产生连锁反应，破坏整个依赖链。依赖倒置原则指出，您应该避免对底层类的硬依赖。

依赖注入是实现依赖倒置的行业标准方式。不要混淆这两者；一个是原则，另一个是指这个原则的实现。

请注意，您也可以实现依赖倒置而不使用依赖注入。例如，在声明字段时，不要写类似`private readonly List<int> _numbers = new List<int>();`这样的代码，而是更倾向于写`private readonly IList<int> = _numbers`，这样可以将依赖转移到抽象（`IList`）而不是具体实现（`List`）。

什么是依赖注入？这是将实现传递并将其设置为抽象槽的行为。有三种实现方式：

+   构造函数注入是通过构造函数参数公开一个抽象，并在创建对象时传递一个实现，然后将其分配给一个字段来实现的。当您想要在同一个对象中一致使用相同的依赖项时（但不一定是同一个类）时，请使用它。

+   方法注入是通过方法参数公开一个抽象，然后在调用该方法时传递一个实现来完成的。当一个方法的依赖可能变化，并且您不打算在该对象的整个生命周期内存储依赖时，请使用它。

+   属性注入是通过公共属性公开一个抽象，然后将该属性分配（或不分配）给某个确切的实现来实现的。属性注入是一种罕见的注入依赖的方式，因为它暗示依赖甚至可能是空的或临时的，并且有许多可能导致它破坏的方式。

给定两种类型，`interface IBartender { }`和`class Bar : Bartender { }`，您可以为名为`Bar`的类说明依赖注入的三种方式。

首先，为构造函数注入准备`Bar`类：

```cpp
class Bar
{
    private readonly IBartender _bartender;

    public Bar(IBartender bartender)
    {
        _bartender = bartender;
    }
}
```

构造函数注入如下完成：

```cpp
var bar = new Bar(new Bartender());
```

这种依赖注入是一种主导的继承方式，因为它通过不可变性来强制稳定性。例如，有些酒吧只有一个调酒师。

方法注入看起来像这样：

```cpp
class Bar
{
    public void ServeDrinks(IBartender bartender)
    {
        // serve drinks using bartender
    }
}
```

注入本身如下：

```cpp
var bar = new Bar();
bar.ServeDrinks(new Bartender());
```

通常，这种依赖注入被称为接口注入，因为该方法通常在接口下进行。接口本身是一个很好的想法，但这并不改变这种依赖注入背后的思想。当您立即使用您设置的依赖项时，或者当您有一种动态设置新依赖项的复杂方式时，请使用方法注入。例如，为了为饮料服务使用不同的调酒师是有意义的。

最后，属性注入可以这样完成：

```cpp
class Bar
{
    public IBartender Bartender { get; set; }
}
```

调酒师现在是这样注入的：

```cpp
var bar = new Bar();
bar.Bartender = new Bartender();
```

例如，酒吧可能会有调酒师轮班，但一次只有一个调酒师。

注意

您可以在[`packt.link/JcmAT`](https://packt.link/JcmAT)找到此示例中使用的代码。

其他语言中的属性注入可能有不同的名称：setter 注入。在实践中，组件并不经常更改，因此这种依赖注入是最罕见的。

对于`File`类，这意味着您应该暴露抽象（接口）而不是暴露类（实现）。这意味着您的`Reader`和`Writer`类应该实现某些契约：

```cpp
public class Reader : IReader
public class Writer: IWriter
```

您的文件类应该暴露阅读器和写入器抽象，而不是实现，如下所示：

```cpp
private readonly IReader _reader;
private readonly IWriter _writer;

public File(IReader reader, IWriter writer)
{
    _reader = reader;
    _writer = writer;
}
```

这允许您选择要注入的`IReader`和`IWriter`的类型。不同的阅读器可能读取不同的文件格式，或者不同的写入器可能以不同的方式输出。您有选择的余地。

依赖注入是一个强大的工具，经常被使用，特别是在企业环境中。它允许您通过在接口之间放置一个接口并具有实现-抽象-实现的 1:1 依赖关系来简化复杂系统。

编写有效的不易破坏的代码可能是矛盾的。这就像从商店购买工具一样；您无法确定它会持续多长时间，或者它会工作得有多好。代码，就像那些工具一样，现在可能有效，但在不久的将来可能会出现问题，只有在它出现问题时才会知道它不起作用。

观察和等待，看代码如何演变，是确保您编写了有效代码的唯一方法。在小型个人项目中，您甚至可能察觉不到任何变化，除非将项目公开或者涉及其他人。对大多数人来说，SOLID 原则通常听起来像是过时的原则，就像过度工程一样。但实际上，它们是经受住时间考验的一套最佳实践，由在企业环境中经验丰富的顶尖专业人士制定。一开始就写出完美的 SOLID 代码是不可能的。实际上，在某些情况下，这甚至不是必要的（例如，如果项目很小并且预计寿命很短）。作为一个想要生产高质量软件并且想要成为专业人士的人，您应该尽早练习它。

# C#如何帮助面向对象设计

到目前为止，您学到的原则并不是特定于语言的。现在是时候学习如何使用 C#进行面向对象编程了。C#是一种很棒的语言，因为它充满了一些非常有用的功能。它不仅是最具生产力的语言之一，而且还允许您编写美观且不易破坏的代码。借助丰富的关键字和语言特性，您可以完全按照自己的意愿对类进行建模，使意图清晰明了。本节将深入探讨帮助面向对象设计的 C#功能。

## 静态

到目前为止，在本书中，您主要与`static`代码进行交互。这指的是不需要新类和对象的代码，并且可以立即调用。在 C#中，静态修饰符可以应用于五种不同的场景——方法、字段、类、构造函数和`using`语句。

静态方法和字段是`static`关键字的最简单应用：

```cpp
public class DogsGenerator
{
    public static int Counter { get; private set; }
    static DogsGenerator()
    {
        // Counter will be 0 anyways if not explicitly provided,
        // this just illustrates the use of a static constructor.
        Counter = 0;
    }
    public static Dog GenerateDog()
    {
        Counter++;
        return new Dog("Dog" + Counter);
    }
}
```

注意

您可以在[`packt.link/748m3`](https://packt.link/748m3)找到此示例中使用的代码。

在这里，您创建了一个名为`DogsGenerator`的类。`静态类`不能手动初始化（使用`new`关键字）。在内部，它被初始化，但只有一次。调用`GenerateDog`方法将返回一个带有名称旁边计数器的新`Dog`对象，例如`Dog1`，`Dog2`和`Dog3`。像这样写一个计数器允许您从任何地方递增它，因为它是`public static`并且有一个 setter。这可以通过直接从类中访问成员来完成：`DogsGenerator.Counter++`将计数器递增`1`。

再次注意，这不需要通过对象调用，因为`static class`实例对整个应用程序是相同的。然而，`DogsGenerator`并不是`static class`的最佳示例。这是因为您刚刚创建了一个全局状态。许多人会说`static`是低效的，应该避免使用，因为它可能由于被无法控制地修改和访问而产生不可预测的结果。

公共可变状态意味着应用程序中的任何地方都可以发生更改。除了难以理解之外，这样的代码在具有多个线程的应用程序环境中也容易出现故障（即它不是线程安全的）。

注意

您将在*第五章*“并发：多线程并行和异步代码”中详细了解线程。

您可以通过使全局状态公开不可变来减少其影响。这样做的好处是现在您有了控制权。与允许计数器在程序的任何地方增加不同，您将仅在`DogsGenerator`内部更改它。对于`counter`属性，实现它就像将 setter 属性设为`private`那样简单。

然而，`static`关键字还有一个有价值的用例，那就是辅助函数。这样的函数接受输入并返回输出，而不在内部修改任何状态。此外，包含这些函数的类是`static`并且没有状态。`static`关键字的另一个良好应用是创建不可变常量。它们使用不同的关键字（`const`）进行定义。`PI`和`E`，静态辅助方法如`Sqrt`和`Abs`等。

`DogsGenerator`类没有适用于对象的成员。如果所有类成员都是`static`，那么类也应该是`static`。因此，您应该将类更改为`public static class DateGenerator`。然而，请注意，依赖`static`与依赖具体实现是一样的。虽然它们易于使用和直接，但是静态依赖很难摆脱，应该仅用于简单的代码，或者您确定不会更改并且在其实现细节中至关重要的代码。因此，`Math`类也是一个`static class`；它具有所有算术计算的基础。

`static`的最后一个应用是`using static`。在`using`语句之前应用`static`关键字会导致所有方法和字段可以直接访问，而无需调用`class`。例如，考虑以下代码：

```cpp
using static Math;
public static class Demo
{
    public static void Run()
    {
   //No need Math.PI
        Console.WriteLine(PI);
    } 
}
```

这是 C#中的静态导入功能。通过使用`static Math`，所有静态成员都可以直接访问。

## Sealed

之前，您提到继承应该谨慎处理，因为复杂性可能会迅速失控。在阅读和编写代码时，您可以仔细考虑复杂性，但是您能通过设计来预防复杂性吗？C#有一个用于阻止继承的关键字叫做`sealed`。如果逻辑上不合理继承一个类，那么您应该用`sealed`关键字标记它。与安全相关的类也应该是 sealed，因为保持它们简单和不可重写是至关重要的。此外，如果性能很重要，那么继承类中的方法比直接在 sealed 类中慢。这是由于方法查找的工作方式。

## 部分

在.NET 中，使用`WinForms`制作桌面应用程序非常流行。`WinForms`的工作方式是您可以通过设计帮助设计应用程序的外观。在内部，它会生成 UI 代码，您只需双击一个组件，它就会生成事件处理程序代码。这就是部分类的用处。所有无聊的自动生成的代码将在一个类中，而您编写的代码将在另一个类中。需要注意的关键点是，这两个类将具有相同的名称，但位于不同的文件中。

你可以拥有尽可能多的部分类。然而，推荐的部分类数量不超过两个。编译器会将它们视为一个大类，但对用户来说，它们会看起来像是两个独立的类。生成代码会生成新的类文件，这将覆盖你编写的代码。在处理自动生成的代码时使用`partial`。初学者最大的错误是使用`partial`来管理复杂的大类。如果你的类很复杂，最好将其拆分为较小的类，而不仅仅是不同的文件。

`partial`还有一个用例。想象一下，你在一个类中有一部分代码，这部分代码只在另一个程序集中需要，但在它最初定义的程序集中是不必要的。你可以在不同的程序集中拥有相同的类并标记为`partial`。这样，不需要的类的一部分将只在需要的地方使用，并在不应该看到的地方隐藏起来。

## 虚拟

抽象方法可以被重写；然而，它们不能被实现。如果你想要一个具有默认行为并且将来可以被重写的方法，你可以使用`virtual`关键字，如下例所示：

```cpp
public class Human
{
    public virtual void SayHi()
    {
        Console.WriteLine("Hello!");
    }
}
```

在这里，`Human`类有`SayHi`方法。这个方法前缀有虚拟关键字，这意味着它可以在子类中改变行为，例如：

```cpp
public class Frenchman : Human
{
    public override void SayHi()
    {
        Console.WriteLine("Bonjour!");
    }
}
```

注意

你可以在[`packt.link/ZpHhI`](https://packt.link/ZpHhI)找到本例中使用的代码。

`Frenchman`类继承了`Human`类并重写了`SayHi`方法。从`Frenchman`对象调用`SayHi`将打印`Bonjour`。

关于 C#的一件事是它的行为很难被重写。在声明方法时，你需要明确告诉编译器该方法可以被重写。只有`virtual`方法可以被重写。接口方法是虚拟的（因为它们后来会得到行为），然而，你不能从子类中重写接口方法。你只能在父类中实现接口。

抽象方法是最后一种虚拟方法，与`virtual`最相似，因为它可以在子类和孙子类中被重写多次。

为了避免脆弱、易变、可重写的行为，最好的虚拟方法是来自接口的方法。`abstract`和`virtual`关键字使得可以在子类中改变类的行为并重写它，如果不加控制地使用，这可能会成为一个大问题。重写行为经常导致不一致和意外的结果，因此在使用`virtual`关键字之前应该小心。

## 内部

`public`、`private`和`protected`是三个已经提到的访问修饰符。许多初学者认为默认的类修饰符是`private`。然而，`private`意味着它不能从类外部调用，在命名空间的上下文中，这并没有太多意义。类的默认访问修饰符是`internal`。这意味着该类只在它所定义的命名空间内可见。`internal`修饰符非常适合在同一个程序集中重用类，同时将它们隐藏在外部。

# 条件运算符

空引用异常可能是编程中最常见的错误。例如，参考以下代码：

```cpp
int[] numbers = null;
numbers.length;
```

这段代码会抛出`NullReferenceException`，因为你正在与一个空值的变量交互。空数组的长度是多少？这个问题没有正确的答案，所以这里会抛出异常。

防止这种错误的最佳方法是完全避免使用空值。然而，有时是不可避免的。在这些情况下，有一种叫做防御性编程的技术。在使用可能为`null`的值之前，确保它不是`null`。

现在回想一下 `Dog` 类的例子。如果你创建一个新对象，`Owner` 的值可能为 `null`。如果你要确定所有者的名字是否以字母 `A` 开头，你需要首先检查 `Owner` 的值是否为 `null`，如下所示：

```cpp
if (dog.Owner != null)
{
    bool ownerNameStartsWithA = dog.Owner.StartsWith('A');
}
```

然而，在 C# 中，使用空值条件，这段代码变得和以下一样简单：

```cpp
dog.Owner?.StartsWith('A');
```

空值条件运算符（`?`）是 C# 中条件运算符的一个例子。它是一个隐式运行 `if` 语句的运算符（特定的 `if` 语句是基于该运算符），并且要么返回某些东西，要么继续工作。`Owner?.StartsWith('A')` 部分如果条件满足则返回 `true`，如果条件不满足或对象为 `null`，则返回 `false`。

C# 中还有更多的条件运算符，你会在学习中了解到。

## 三元运算符

几乎没有一种语言不包含 `if` 语句。最常见的 `if` 语句之一是 `if-else`。例如，如果 `Dog` 类的实例的 `Owner` 的值为 `null`，你可以简单地描述实例为 `{Name}`。否则，你可以更好地描述它为 `{Name}, dog of {Owner}`，如下所示：

```cpp
if (dog1.Owner == null)
{
    description = dog1.Name;
}
else
{
    description = $"{dog1.Name}, dog of {dog1.Owner}";
}
```

与许多其他语言一样，C# 通过使用三元运算符简化了这个过程。

```cpp
description = dog1.Owner == null
    ? dog1.Name
    : $"{dog1.Name}, dog of {dog1.Owner}";
```

在左侧，你有一个条件（true 或 false），后面跟着一个问号（`?`），如果条件为真，则返回右侧的值，后面跟着一个冒号（`:`），如果条件为假，则返回左侧的值。`$` 是一个字符串插值文字，它允许你写 `$"{dog1.Name}, dog of {dog1.Owner}"` 而不是 `dog1.Name + "dog of" + dog1.Owner`。在连接文本时应该使用它。

现在假设有两只狗。你希望第一只狗加入第二只狗（也就是说，被第二只狗的主人拥有），但这只有在第二只狗有主人的情况下才能发生。通常，你会使用以下代码：

```cpp
if (dog1.Owner != null)
{
    dog2.Owner = dog1.Owner;
}
```

但是在 C# 中，你可以使用以下代码：

```cpp
dog1.Owner = dog1.Owner ?? dog2.Owner;
```

在这里，你应用了空值合并运算符（`??`），如果它是 `null`，则返回右侧的值，如果不是 `null`，则返回左侧的值。但是，你可以进一步简化这个过程：

```cpp
dog1.Owner ??= dog2.Owner;
```

这意味着如果你要分配的值（在左侧）是 `null`，那么输出将是右侧的值。

空值合并运算符的最后一个用例是输入验证。假设有两个类，`ComponentA` 和 `ComponentB`，并且 `ComponentB` 必须包含 `ComponentA` 的一个初始化实例。你可以写以下代码：

```cpp
public ComponentB(ComponentA componentA)
{
    if (componentA == null)
    {
        throw new ArgumentException(nameof(componentA));
    }
    else
    {
        _componentA = componentA;
    }
}
```

然而，你可以简单地写成以下形式：

```cpp
_componentA = componentA ?? throw new ArgumentNullException(nameof(componentA));
```

这可以理解为如果没有 `componentA`，那么必须抛出异常。

注意

你可以在 [`packt.link/yHYbh`](https://packt.link/yHYbh) 找到此示例中使用的代码。

在大多数情况下，空值运算符应该替换标准的 `if null-else` 语句。但是，要小心使用三元运算符的方式，并将其限制在简单的 `if-else` 语句中，因为代码可能会变得非常难读。

## 重载运算符

C# 中有多少东西可以被抽象化，这是很迷人的。比较原始数字，相乘或相除都很容易，但是当涉及到对象时，情况就不那么简单了。一个人加上另一个人是什么？一个袋子的苹果乘以另一个袋子的苹果是什么？很难说，但在某些领域的情况下，这是完全有意义的。

考虑一个稍微好一点的例子。假设你正在比较银行账户。找出哪个银行账户里的钱更多是一个常见的用例。通常，要比较两个账户，你需要访问它们的成员，但是 C# 允许你重载比较运算符，以便你可以比较对象。例如，假设你有一个像这样的 `BankAccount` 类：

```cpp
public class BankAccount
{
    private decimal _balance;

    public BankAccount(decimal balance)
    {
        _balance = balance;
    }
}
```

在这里，余额金额是`private`。你不关心`balance`的确切值；你只想比较一个与另一个。你可以实现一个`CompareTo`方法，但相反，你将实现一个比较运算符。在`BankAccount`类中，你将添加以下代码：

```cpp
public static bool operator >(BankAccount account1, BankAccount account2)
    => account1?._balance > account2?._balance;
```

上述代码称为运算符重载。通过自定义运算符重载，你可以在余额更大时返回 true，否则返回 false。在 C#中，运算符是`public static`，后面跟着返回类型。之后，你有`operator`关键字，后面跟着被重载的实际运算符。输入取决于被重载的运算符。在这种情况下，你传递了两个银行账户。

如果你尝试按原样编译代码，你会得到一个错误，说有东西丢失了。比较运算符有一个相反的方法是有意义的。现在，添加小于运算符重载如下：

```cpp
public static bool operator <(BankAccount account1, BankAccount account2)
    => account1?._balance < account2?._balance;
```

现在代码已经编译。最后，有一个相等比较是有意义的。记住，你需要添加一对，相等和不相等：

```cpp
public static bool operator ==(BankAccount account1, BankAccount account2)
    => account1?._balance == account2?._balance; 
public static bool operator !=(BankAccount account1, BankAccount account2)
    => !(account1 == account2);
```

接下来，你将创建要比较的银行账户。请注意，所有数字都有一个附加的`m`，因为这个后缀使这些数字成为`decimal`。默认情况下，带有小数的数字是`double`，所以你需要在末尾添加`m`使它们成为`decimal`：

```cpp
var account1 = new BankAccount(-1.01m);
var account2 = new BankAccount(1.01m);
var account3 = new BankAccount(1001.99m);
var account4 = new BankAccount(1001.99m);
```

现在比较两个银行账户变得如此简单：

```cpp
Console.WriteLine(account1 == account2);
Console.WriteLine(account1 != account2);
Console.WriteLine(account2 > account1);
Console.WriteLine(account1 < account2);
Console.WriteLine(account3 == account4);
Console.WriteLine(account3 != account4);
```

运行代码会导致以下内容被打印到控制台：

```cpp
False
True
True
True
True
False
```

注意

你可以在[`packt.link/5DioJ`](https://packt.link/5DioJ)找到此示例中使用的代码。

许多（但不是所有）运算符可以被重载，但仅仅因为你可以这样做并不意味着你应该这样做。在某些情况下，重载运算符是有意义的，但在其他情况下，可能是违反直觉的。再次强调，记住不要滥用 C#的特性，只有在**逻辑**上有意义，使代码更易于阅读、学习和维护时才使用它们。

## 可空原始类型

你是否曾经想过当原始值是未知的时候该怎么办？例如，假设一组产品已经宣布。它们的名称、描述和一些其他参数是已知的，但价格只在发布前才公布。你应该使用什么类型来存储价格值？

可空原始类型是可能具有一些值或没有值的原始类型。在 C#中，要声明这样的类型，你必须在原始类型后添加`?`，如下面的代码所示：

```cpp
int? a = null;
```

在这里，你声明了一个可能有值也可能没有值的字段。具体来说，这意味着 a 可能是未知的。不要将其与默认值混淆，因为默认情况下，`int`类型的值是`0`。

你可以很简单地给可空字段赋值，如下所示：

```cpp
a = 1;
```

然后，你可以按照以下方式编写代码来检索其值：

```cpp
int b = a.Value;
```

## 泛型

有时，你会遇到这样的情况，你用不同的类型做完全相同的事情，唯一的区别是类型。例如，如果你需要创建一个打印`int`值的方法，你可以写以下代码：

```cpp
public static void Print(int element)
{
    Console.WriteLine(element);
}
If you need to print a float, you could add another overload:
public static void Print(float element)
{
    Console.WriteLine(element);
}
```

同样，如果你需要打印一个字符串，你可以添加另一个重载：

```cpp
public static void Print(string element)
{
    Console.WriteLine(element);
}
```

你做了三次相同的事情。当然，一定有办法减少代码重复。记住，在 C#中，所有类型都派生自`object`类型，它有`ToString()`方法，所以你可以执行以下命令：

```cpp
public static void Print(object element)
{
    Console.WriteLine(element);
}
```

尽管最后的实现包含的代码最少，但实际上效率最低。对象是引用类型，而原始类型是值类型。当你将一个原始类型赋值给一个对象时，你也创建了一个新的引用。这就是所谓的装箱。它并不是免费的，因为你将对象从`堆栈`移动到`堆`。程序员应该意识到这一事实，并尽可能避免它。

在本章的前面，你遇到了多态性——一种使用相同类型进行不同操作的方式。你也可以使用不同类型做同样的事情，泛型是让你能够做到这一点的关键。在`Print`示例中，你需要一个泛型方法：

```cpp
public static void Print<T>(T element)
{
    Console.WriteLine(element);
}
```

使用菱形括号（`<>`），你可以指定一个类型`T`，这个函数可以使用`<T>`表示它可以与任何类型一起工作。

现在，假设你想打印数组的所有元素。简单地将一个集合传递给`WriteLine`语句将导致打印一个引用，而不是所有的元素。通常情况下，你会创建一个打印所有传递元素的方法。有了泛型的强大功能，你可以有一个方法来打印任何类型的数组：

```cpp
public static void Print<T>(T[] elements)
{
    foreach (var element in elements)
    {
        Console.WriteLine(element);
    }
}
```

请注意，泛型版本不像使用`object`类型那样高效，因为你仍然会使用以`object`作为参数的`WriteLine`重载。当传递一个泛型时，你无法确定它是否需要调用一个带有`int`、`float`或`String`的重载，或者是否一开始就有一个确切的重载。如果没有一个接受对象参数的`WriteLine`重载，你将无法调用`Print`方法。因此，最有效的代码实际上是具有三个重载的代码。不过这并不是非常重要，因为这只是一个非常特定的场景，无论如何都会发生装箱。然而，在许多其他情况下，你不仅可以使代码简洁，而且还可以使其高效。

有时，选择泛型或多态函数的答案隐藏在微小的细节中。如果你需要实现一个比较两个元素并在第一个元素更大时返回`true`的方法，你可以在 C#中使用`IComparable`接口来实现：

```cpp
public static bool IsFirstBigger1(IComparable first, IComparable second)
{
    return first.CompareTo(second) > 0;
}
```

这个的泛型版本将如下所示：

```cpp
public static bool IsFirstBigger2<T>(T first, T second)
    where T : IComparable
{
    return first.CompareTo(second) > 0;
}
```

这里的新内容是`where T : IComparable`。这是一个泛型约束。默认情况下，你可以将任何类型传递给泛型类或方法。约束仍然允许传递不同的类型，但它们显著减少了可能的选项。泛型约束只允许符合约束的类型作为泛型类型传递。在这种情况下，你只允许实现`IComparable`接口的类型。约束可能看起来像是对类型的限制；然而，它们暴露了在泛型方法中可以使用的受约束类型的行为。有了约束，你可以使用这些类型的特性，因此它非常有用。在这种情况下，你确实限制了可以使用的类型，但同时，无论你传递什么类型到泛型方法中，它都是可比较的。

如果你需要返回第一个元素本身，而不是返回第一个元素是否更大，你可以编写一个非泛型方法如下：

```cpp
public static IComparable Max1(IComparable first, IComparable second)
{
    return first.CompareTo(second) > 0
        ? first
        : second;
}
```

泛型版本如下所示：

```cpp
public static T Max2<T>(T first, T second)
    where T : IComparable
{
    return first.CompareTo(second) > 0
        ? first
        : second;
}
```

此外，值得比较的是，你将如何使用每个版本获得有意义的输出。使用非泛型方法，代码将如下所示：

```cpp
int max1 = (int)Comparator.Max1(3, -4);
```

有了泛型版本，代码会像这样：

```cpp
int max2 = Comparator.Max2(3, -4);
```

注意

你可以在[`packt.link/sIdOp`](https://packt.link/sIdOp)找到本示例中使用的代码。

在这种情况下，胜者是显而易见的。在非泛型版本中，你必须进行强制转换。在代码中进行强制转换是不受欢迎的，因为如果你出现错误，你将在运行时得到错误，事情可能会发生变化，强制转换将失败。强制转换也是额外的操作，而泛型版本更加流畅，因为它不需要强制转换。当你想要直接使用类型而不是通过它们的抽象时，请使用泛型。从函数中返回一个确切的（非多态）类型是它的最佳用例之一。

C#泛型将在*第四章*，*数据结构和 LINQ*中详细介绍。

## 枚举

`enum`类型表示一组已知的值。由于它是一种类型，您可以将其传递给方法而不是传递原始值。`enum`包含所有可能的值，因此不可能有一个值它不包含。以下代码片段显示了一个简单的示例：

```cpp
public enum Gender
{
    Male,
    Female,
    Other
}
```

注意

您可以在[`packt.link/gP9Li`](https://packt.link/gP9Li)找到此示例使用的代码。

现在，您可以通过编写`Gender.Other`来获取可能的性别值，就好像它在一个`static class`中一样。枚举可以很容易地通过强制转换转换为整数—`(int)Gender.Male`将返回`0`，`(int)Gender.Female`将返回`1`，依此类推。这是因为`enum`默认从`0`开始编号。

枚举没有任何行为，它们被称为常量容器。当您想要使用常量并防止通过设计传递无效值时，应该使用它们。

## 扩展方法

几乎总是，您将会处理不属于您的代码的一部分。有时，这可能会造成不便，因为您无法访问更改它。是否可能以某种方式扩展现有类型以获得所需的功能？是否可能在不继承或创建新组件类的情况下做到这一点？

通过扩展方法，您可以轻松实现这一点。它们允许您在完整类型上添加方法，并且可以像本地方法一样调用它们。

如果您想要使用`Print`方法将`string`打印到控制台，但是从`string`本身调用它呢？`String`没有这样的方法，但是您可以使用扩展方法添加它：

```cpp
public static class StringExtensions
{
    public static void Print(this string text)
    {
        Console.WriteLine(text);
    }
}
```

这样可以编写以下代码：

```cpp
"Hey".Print();
```

这将在控制台上打印`Hey`如下：

```cpp
Hey
```

注意

您可以在[`packt.link/JC5cj`](https://packt.link/JC5cj)找到此示例使用的代码。

扩展方法是`static`的，必须放在一个`static class`中。如果您查看方法的语义，您会注意到使用了`this`关键字。`this`关键字应该是扩展方法中的第一个参数。之后，函数继续正常进行，您可以使用带有`this`关键字的参数，就好像它只是另一个参数一样。

使用扩展方法可以向现有类型添加（扩展，但不是与继承发生的相同扩展）新的行为，即使该类型在其他情况下不支持具有方法。通过扩展方法，甚至可以向`enum`类型添加方法，否则是不可能的。

# 结构

类是引用类型，但并非所有对象都是引用类型（保存在堆上）。有些对象可以在堆栈上创建，这样的对象是使用结构创建的。

结构的定义类似于类，但用于稍有不同的事情。现在，创建一个名为`Point`的`struct`：

```cpp
public struct Point
{
    public readonly int X;
    public readonly int Y;

    public Point(int x, int y)
    {
        X = x;
        Y = y;
    }
}
```

这里唯一的真正区别是`struct`关键字，它表示这个对象将被保存在堆栈上。此外，您可能已经注意到没有使用属性。有很多人会用`x`和`y`代替`Point`。这并不是什么大不了的事，但是你会用两个变量来代替一个变量。这种使用原始类型的方式被称为原始类型偏执。您应该遵循面向对象编程的原则，使用抽象、封装良好的数据以及行为来保持高内聚。在选择变量放置的位置时，问问自己这个问题：`x`能独立于`y`改变吗？你会修改一个点吗？点本身是一个完整的值吗？所有这些问题的答案都是**是**，因此将其放入数据结构中是有意义的。但为什么选择结构而不是类呢？

结构体很快，因为它们在堆上没有任何分配。它们也很快，因为它们是按值传递的（因此，访问是直接的，而不是通过引用）。按值传递值，因此即使您可以修改结构体，更改也不会在方法外保留。当某物只是一个简单的、小的复合值时，您应该使用结构体。最后，使用结构体，您可以获得值相等。

`struct`的另一个有效示例是`DateTime`。`DateTime`只是一个时间单位，包含一些信息。它也不会单独改变，并支持`AddDays`、`TryParse`和`Now`等方法。即使它有几个不同的数据片段，它们可以被视为一个单位，因为它们与日期和时间相关。

大多数`structs`应该是不可变的，因为它们是通过值的副本传递的，所以在方法内部更改某些东西不会保留这些更改。您可以向`struct`添加一个`readonly`关键字，使其所有字段都是`readonly`：

```cpp
public readonly struct Point
{
    public int X { get; }
    public int Y { get; }

    public Point(int x, int y)
    {
        X = x;
        Y = y;
    }
}
```

一个`readonly` `struct`可以有一个`readonly`字段或 getter 属性。这对于您代码库的未来维护者非常有用，因为它可以防止他们做您没有设计的事情（不可变性）。结构体只是一小组数据位，但它们也可以有行为。有一个方法来计算两点之间的距离是有意义的：

```cpp
public static double DistanceBetween(Point p1, Point p2)
{
    return Math.Sqrt((p1.X - p2.X) * (p1.X - p2.X) + (p1.Y - p2.Y) * (p1.Y - p2.Y));
}
```

前面的代码中有一点数学知识——即两点之间的距离是点 x 和 y 的平方差相加的平方根。

计算这个点和其他点之间的距离也是有意义的。您不需要改变任何东西，因为您可以重用现有的代码，传递正确的参数：

```cpp
public double DistanceTo(Point p)
{
    return DistanceBetween(this, p);
}
```

如果您想要测量两点之间的距离，可以这样创建它们：

```cpp
var p1 = new Point(3,1);
var p2 = new Point(3,4);
```

并使用成员函数来计算距离：

```cpp
var distance1 = p1.DistanceTo(p2);
```

或一个静态函数：

```cpp
var distance2 = Point.DistanceBetween(p1, p2);
```

每个版本的结果将如下：

```cpp
– 3.
```

注意

您可以在[`packt.link/PtQzz`](https://packt.link/PtQzz)找到此示例使用的代码。

当您考虑一个结构体时，把它看作只是一组原始类型。需要记住的关键点是结构体中的所有数据成员（属性或字段）在对象初始化期间必须被赋值。出于同样的原因，局部变量在没有初始设置值的情况下不能使用。结构体不支持继承；然而，它们支持实现接口。

结构体实际上是一个简单业务逻辑的好方法。结构体应该保持简单，不应该包含其他对象引用；它们应该只包含原始值。然而，一个类可以持有它需要的许多结构体对象。使用结构体是一种逃避过度使用原始类型并自然地使用简单逻辑的好方法，它们属于数据的一个小组——即`struct`。

## 记录

记录是引用类型（不像`struct`，更像类）。然而，它具有按值比较的方法（使用`equals`方法和运算符）。此外，记录有一个不同的`ToString()`的默认实现，不再打印类型，而是打印所有属性。在许多情况下，这正是所需要的，因此它非常有帮助。最后，记录周围有很多语法糖，您即将见证。

您已经知道如何在 C#中创建自定义类型。不同自定义类型之间唯一的区别是使用的关键字。对于记录类型，这样一个关键字是`record`。例如，您现在将创建一个电影记录。它有一个`Title`、`Director`、`Producer`、`Description`和一个`ReleaseDate`：

```cpp
public record MovieRecordV1
{
    public string Title { get; }
    public string Director { get; }
    public string Producer { get; } 
    public string Description { get; set; }
    public DateTime ReleaseDate { get; }

    public MovieRecordV1(string title, string director, string producer, DateTime releaseDate)
    {
        Title = title;
        Director = director;
        Producer = producer;
        ReleaseDate = releaseDate;
    }
}
```

到目前为止，您应该会觉得这非常熟悉，因为唯一的区别是关键字。尽管有这样一个细微的差别，您已经获得了重大的好处。

注意

在本章中拥有`MovieRecordV1`类的意图，与 GitHub 代码中的`MovieClass`相对应，是为了拥有一个类似于类的类型，然后重构突出显示记录如何帮助。

创建两部相同的电影：

```cpp
private static void DemoRecord()
{
    var movie1 = new MovieRecordV1(
        "Star Wars: Episode I – The Phantom Menace",
        "George Lucas",
        "Rick McCallum",
        new DateTime(1999, 5, 15));

    var movie2 = new MovieRecordV1(
        "Star Wars: Episode I – The Phantom Menace",
        "George Lucas",
        "Rick McCallum",
        new DateTime(1999, 5, 15));
}
```

到目前为止，一切都一样。尝试将电影打印到控制台：

```cpp
    Console.WriteLine(movie1);
```

输出将如下所示：

```cpp
MovieRecordV1 { Title = Star Wars: Episode I - The Phantom Menace, Director = George Lucas, Producer
= Rick McCallum, Description = , ReleaseDate = 5/15/1999 12:00:00 AM }
```

注意

您可以在[`packt.link/xylkW`](https://packt.link/xylkW)找到此示例中使用的代码。

如果您尝试对类或`struct`对象执行相同操作，您将只获得一个类型打印。但是，对于记录，默认行为是打印其所有属性及其值。

这不是记录的唯一好处。再次，记录具有值相等语义。比较两个电影记录将通过它们的属性值进行比较：

```cpp
    Console.WriteLine(movie1.Equals(movie2));
    Console.WriteLine(movie1 == movie2);
```

这将打印`true true`。

使用相同数量的代码，您已经成功地通过简单地将数据结构更改为记录来获得最大的功能。记录提供了`Equals()`，`GetHashCode()`覆盖，`==和!=覆盖`，甚至`ToString`覆盖，它打印记录本身（所有成员及其值）。记录的好处并不止于此，因为使用它们，您可以减少大量样板代码。充分利用记录并重写您的电影记录：

```cpp
public record MovieRecord(string Title, string Director, string Producer, string Description, DateTime ReleaseDate);
```

这是一个位置记录，这意味着您传递的所有参数将最终出现在正确的只读数据成员中，就好像它是一个专用构造函数。如果您再次运行演示，您会注意到它不再编译。这个声明的主要区别是，现在不再可能更改描述。使可变属性并不困难，您只需要明确说明：

```cpp
public record MovieRecord(string Title, string Director, string Producer, DateTime ReleaseDate)
{
    public string Description { get; set; }
}
```

您从讨论不可变性开始了这一段，但为什么主要关注记录？记录的好处实际上是不可变性。使用`with`表达式，您可以创建一个记录对象的副本，其中有零个或更多个属性被修改。所以，假设您将这个添加到您的演示中：

```cpp
var movie3 = movie2 with { Description = "Records can do that?" };
movie2.Description = "Changing original";
Console.WriteLine(movie3);
```

代码将导致这种情况：

```cpp
MovieRecord { Title = Star Wars: Episode I - The Phantom Menace, Director = George Lucas, Producer
= Rick McCallum, ReleaseDate = 5/15/1999 12:00:00 AM, Description = Records can do that? }
```

正如您所看到的，此代码复制了一个仅更改了一个属性的对象。在记录之前，您需要大量的代码来确保所有成员都被复制，然后才能设置一个值。请记住，这会创建一个浅表复制。浅复制是一个所有引用都被复制的对象。深复制是一个所有引用类型对象都被重新创建的对象。不幸的是，没有办法覆盖这种行为。记录不能继承类，但可以继承其他记录。它们也可以实现接口。

除了作为引用类型之外，记录更像是结构，因为它们具有值相等性和围绕不可变性的语法糖。它们不应该用作结构的替代品，因为结构仍然更适合于具有简单逻辑的小型和简单对象。当您想要不可变对象用于数据时，请使用记录，这些数据可能包含其他复杂对象（如果嵌套对象可能具有更改状态的状态，浅复制可能会导致意外行为）。

## 仅初始化设置器

随着记录的引入，之前的版本 C# 9 还引入了`init`-only setter 属性。使用`init`而不是`set`可以为属性启用对象初始化：

```cpp
public class House
{
    public string Address { get; init; }
    public string Owner { get; init; }
    public DateTime? Built { get; init; }
}
```

这使您能够创建具有未知属性的房屋：

```cpp
var house2 = new House();
```

或者分配它们：

```cpp
var house1 = new House
{
    Address = "Kings street 4",
    Owner = "King",
    Built = DateTime.Now
};
```

当您想要只读数据时，使用`init`-only 设置器特别有用，这些数据可以是已知的或未知的，但不是以一致的方式。

注意

您可以在[`packt.link/89J99`](https://packt.link/89J99)找到此示例中使用的代码。

## ValueTuple 和解构

您已经知道函数只能返回一件事。在某些情况下，您可以使用`out`关键字返回第二件事。例如，将字符串转换为数字通常是这样做的：

```cpp
var text = "123";
var isNumber = int.TryParse(text, out var number);
```

`TryParse`返回解析的数字以及文本是否为数字。

然而，C#有一种更好的方法来返回多个值。您可以使用一个名为`ValueTuple`的数据结构来实现这一点。它是一个通用的`struct`，包含了一个到六个公共可变字段，字段的类型可以是任意指定的。它只是一个用来保存不相关值的容器。例如，如果你有一个`dog`，一个`human`和一个`Bool`，你可以把这三个值存储在一个`ValueTuple`结构中：

```cpp
var values1 = new ValueTuple<Dog, Human, bool>(dog, human, isDogKnown);
```

然后你可以通过`values1.Item1`访问每一个值，比如`dog`，通过`values1.Item2`访问`human`，通过`values.Item3`访问`isDogKnown`。创建`ValueTuple`结构的另一种方法是使用括号。这与之前的方法完全相同，只是使用了括号的语法：

```cpp
var values2 = (dog, human, isDogKnown);
```

以下语法非常有用，因为你可以声明一个几乎返回多个值的函数：

```cpp
public (Dog, Human, bool) GetDogHumanAndBool()
{
    var dog = new Dog("Sparky");
    var human = new Human("Thomas");
    bool isDogKnown = false;

    return (dog, human, isDogKnown);
}
```

注意

你可以在[`packt.link/OTFpm`](https://packt.link/OTFpm)找到本例中使用的代码。

你也可以使用另一个 C#特性，叫做解构，来做相反的操作。它可以获取对象的数据成员，并允许你将它们分开成单独的变量。元组类型的问题在于它没有一个强有力的名称。如前所述，每个字段都将被称为`ItemX`，其中`X`是返回的项目的顺序。在处理所有这些时，`GetDogHumanAndBool`需要将结果分配给三个不同的变量：

```cpp
var dogHumanAndBool = GetDogHumanAndBool();
var dog = dogHumanAndBool.Item1;
var human = dogHumanAndBool.Item2;
var boo = dogHumanAndBool.Item3;
```

你可以简化这个过程，而是使用解构——直接将对象属性分配给不同的变量：

```cpp
var (dog, human, boo) = GetDogHumanAndBool(); 
```

使用解构，你可以使这个过程更加可读和简洁。当你有多个不相关的变量，并且想要从一个函数中返回它们时，可以使用`ValueTuple`。你不必总是使用`out`关键字来解决问题，也不必通过创建一个新的类来增加开销。你可以通过简单地返回然后解构`ValueTuple`结构来解决这个问题。

通过以下练习，你可以亲身体验使用 SOLID 原则逐步编写代码。

## 练习 2.04：创建一个可组合的温度单位转换器

温度可以用不同的单位来测量：摄氏度、开尔文和华氏度。将来可能会添加更多的单位。但是，单位不需要由用户动态添加；应用程序要么支持它，要么不支持。你需要制作一个应用程序，将温度从任何单位转换为另一个单位。

需要注意的是，转换到和从这些单位的转换将是完全不同的事情。因此，你将需要为每个转换器编写两种方法。作为标准单位，你将使用摄氏度。因此，每个转换器都应该有一个从摄氏度到其他单位的转换方法，这使得它成为程序中最简单的单位。当你需要将非摄氏度转换为摄氏度时，你将需要涉及两个转换器——一个用来将输入适应标准单位（C），然后另一个用来将 C 转换为你想要的任何单位。这个练习将帮助你使用本章学到的 SOLID 原则和 C#特性来开发一个应用程序，比如`record`和`enum`。

按照以下步骤执行：

1.  创建一个`TemperatureUnit`，它使用`enum`类型来定义常量，即一组已知的值。你不需要动态添加它：

```cpp
public enum TemperatureUnit
{
    C,
    F,
    K
}
```

在这个例子中，你将使用三种温度单位，分别是`C`、`K`和`F`。

1.  温度应该被看作一个由两个属性组成的简单对象：`Unit`和`Degrees`。你可以使用`record`或`struct`，因为它是一个非常简单的带有数据的对象。在这里，最好的选择是选择`struct`（因为对象的大小），但为了练习，你将使用一个`record`：

```cpp
public record Temperature(double Degrees, TemperatureUnit Unit);
```

1.  接下来，添加一个合同，定义你从一个特定的温度转换器中想要得到什么：

```cpp
public interface ITemperatureConverter
{
    public TemperatureUnit Unit { get; }
    public Temperature ToC(Temperature temperature);
    public Temperature FromC(Temperature temperature);
}
```

你定义了一个接口，其中包含三个方法——`Unit`属性用于标识转换器所针对的温度，`ToC`和`FromC`用于从标准单位转换到和从标准单位转换。

1.  现在您有了一个转换器，添加可组合的转换器，它具有一组转换器：

```cpp
public class ComposableTemperatureConverter
{
    private readonly ITemperatureConverter[] _converters;
```

1.  拥有重复的温度单位转换器是没有意义的。因此，当检测到重复转换器时，添加一个将被抛出的错误。而且，没有任何转换器也是没有意义的。因此，应该有一些代码来验证`null`或空转换器：

```cpp
public class InvalidTemperatureConverterException : Exception
{
    public InvalidTemperatureConverterException(TemperatureUnit unit) : base($"Duplicate converter for {unit}.")
    {
    }

    public InvalidTemperatureConverterException(string message) : base(message)
    {
    }
}
```

在创建自定义异常时，应尽可能提供有关错误上下文的尽可能多的信息。在这种情况下，传递未找到转换器的`unit`。

1.  添加一个需要非空转换器的方法：

```cpp
private static void RequireNotEmpty(ITemperatureConverter[] converters)
{
    if (converters?.Length > 0 == false)
    {
        throw new InvalidTemperatureConverterException("At least one temperature conversion must be supported");
    }
}
```

传递一个空转换器数组会抛出`InvalidTemperatureConverterException`异常。

1.  添加一个需要非重复转换器的方法：

```cpp
private static void RequireNoDuplicate(ITemperatureConverter[] converters)
{
    for (var index1 = 0; index1 < converters.Length - 1; index1++)
    {
        var first = converters[index1];
        for (int index2 = index1 + 1; index2 < converters.Length; index2++)
        {
            var second = converters[index2];
            if (first.Unit == second.Unit)
            {
                throw new InvalidTemperatureConverterException(first.Unit);
            }
        }
    }
}
```

这个方法遍历每个转换器，并检查在其他索引处是否重复转换器（通过重复`TemperatureUnit`）。如果找到重复的单位，它将抛出异常。如果没有，它将正常终止。

1.  现在将所有内容组合在一个构造函数中：

```cpp
public ComposableTemperatureConverter(ITemperatureConverter[] converters)
{
    RequireNotEmpty(converters);
    RequireNoDuplicate(converters);
    _converters = converters;
}
```

在创建转换器时，验证不为空且不重复的转换器，然后设置它们。

1.  接下来，在可组合的转换器内创建一个`private`辅助方法来帮助您找到所需的转换器`FindConverter`：

```cpp
private ITemperatureConverter FindConverter(TemperatureUnit unit)
{
    foreach (var converter in _converters)
    {
        if (converter.Unit == unit)
        {
            return converter;
        }
    }

    throw new InvalidTemperatureConversionException(unit);
}
```

该方法返回所需单位的转换器，如果找不到转换器，则抛出异常。

1.  为了简化您搜索和从任何单位转换为摄氏度的过程，添加一个`ToCelsius`方法：

```cpp
private Temperature ToCelsius(Temperature temperatureFrom)
{
    var converterFrom = FindConverter(temperatureFrom.Unit);
    return converterFrom.ToC(temperatureFrom);
}
```

在这里，您找到所需的转换器并将`Temperature`转换为 Celsius。

1.  对于从摄氏度转换为任何其他单位的转换，也是同样的操作：

```cpp
private Temperature CelsiusToOther(Temperature celsius, TemperatureUnit unitTo)
{
    var converterTo = FindConverter(unitTo);
    return converterTo.FromC(celsius);
}
```

1.  通过实现这个算法，将温度标准化（转换为摄氏度），然后转换为任何其他温度，将所有内容包装起来：

```cpp
public Temperature Convert(Temperature temperatureFrom, TemperatureUnit unitTo)
{
    var celsius = ToCelsius(temperatureFrom);
    return CelsiusToOther(celsius, unitTo);
}
```

1.  添加一些转换器。从 Kelvin 转换器`KelvinConverter`开始：

```cpp
public class KelvinConverter : ITemperatureConverter
{
    public const double AbsoluteZero = -273.15;

    public TemperatureUnit Unit => TemperatureUnit.K;

    public Temperature ToC(Temperature temperature)
    {
        return new(temperature.Degrees + AbsoluteZero, TemperatureUnit.C);
    }

    public Temperature FromC(Temperature temperature)
    {
        return new(temperature.Degrees - AbsoluteZero, Unit);
    }
}
```

这个方法的实现和所有其他转换器的实现都很简单。您只需要实现将正确单位转换为或从摄氏度的公式。Kelvin 有一个有用的常数，绝对零度，所以您使用了一个命名常量而不是一个魔术数字`–273.15`。另外，值得记住的是温度不是一个原始类型。它既是一个度数值又是一个单位。因此，在转换时，您需要同时传递两者。`ToC`将始终以`TemperatureUnit.C`作为单位，而`FromC`将采用转换器被识别为的任何单位，即`TemperatureUnit.K`。

1.  现在添加一个 Fahrenheit 转换器`FahrenheitConverter`：

```cpp
public class FahrenheitConverter : ITemperatureConverter
{
    public TemperatureUnit Unit => TemperatureUnit.F;

    public Temperature ToC(Temperature temperature)
    {
        return new(5.0/9 * (temperature.Degrees - 32), TemperatureUnit.C);
    }

    public Temperature FromC(Temperature temperature)
    {
        return new(9.0 / 5 * temperature.Degrees + 32, Unit);
    }
}
```

Fahrenheit 在结构上是相同的；唯一的区别是公式和单位值。

1.  添加一个`CelsiusConverter`，它将接受一个温度值并返回相同的值，如下所示：

```cpp
    public class CelsiusConverter : ITemperatureConverter
    {
        public TemperatureUnit Unit => TemperatureUnit.C;

        public Temperature ToC(Temperature temperature)
        {
            return temperature;
        }

        public Temperature FromC(Temperature temperature)
        {
            return temperature;
        }
    }
```

`CelsiusConverter`是最简单的。它什么也不做；它只是返回相同的温度。转换器将温度转换为标准温度——摄氏度转换为摄氏度始终是摄氏度。为什么你需要这样一个类呢？没有它，您需要稍微改变流程，添加`if`语句来忽略摄氏度的温度。但是通过这种实现，您可以将其合并到相同的流程中，并且可以在相同的抽象`ITemperatureConverter`的帮助下以相同的方式使用它。

1.  最后，创建一个演示：

```cpp
Solution.cs
public static class Solution
{
    public static void Main()
    {
        ITemperatureConverter[] converters = {new FahrenheitConverter(), new KelvinConverter(), new CelsiusConverter()};
        var composableConverter = new ComposableTemperatureConverter(converters);

        var celsius = new Temperature(20.00001, TemperatureUnit.C);

        var celsius1 = composableConverter.Convert(celsius, TemperatureUnit.C);
        var fahrenheit = composableConverter.Convert(celsius1, TemperatureUnit.F);
        var kelvin = composableConverter.Convert(fahrenheit, TemperatureUnit.K);
        var celsiusBack = composableConverter.Convert(kelvin, TemperatureUnit.C);
        Console.WriteLine($"{celsius} = {fahrenheit}");
```

```cpp
You can find the complete code here: https://packt.link/ruBph.
```

在这个例子中，您已经创建了所有的转换器并将它们传递给名为`composableConverter`的转换器容器。然后，您创建了一个摄氏度温度，并用它来执行从其他温度到摄氏度的转换。

1.  运行代码，您将得到以下结果：

```cpp
Temperature { Degrees = 20.00001, Unit = C } = Temperature { Degrees = 68.000018, Unit = F }
Temperature { Degrees = 68.000018, Unit = F } = Temperature { Degrees = -253.14998999999997, Unit = K }
Temperature { Degrees = -253.14998999999997, Unit = K } = Temperature { Degrees = 20.000010000000003, Unit = C }
```

注意

您可以在[`packt.link/dDRU6`](https://packt.link/dDRU6)找到用于此练习的代码。

一个软件开发人员理想情况下应该以这样的方式设计代码，使得现在或将来进行更改需要相同的时间。使用 SOLID 原则，你可以逐步编写代码并最小化破坏性更改的风险，因为你永远不会改变现有的代码；你只是添加新的代码。随着系统的增长，复杂性增加，学习事物如何工作可能会很困难。通过明确定义的契约，SOLID 使你能够拥有易于阅读和易于维护的代码，因为每个部分都是简单明了的，并且它们彼此之间是隔离的。

现在，你将通过一个活动来测试你创建类和重载运算符的知识。

## 活动 2.01：合并两个圆

在这个活动中，你将创建类和重载运算符来解决以下数学问题：一部分比萨面团可以用来制作两个半径为三厘米的圆形比萨小块。使用相同量的面团制作一个单独的比萨小块的半径是多少？你可以假设所有的比萨小块厚度都是一样的。以下步骤将帮助你完成这个活动：

1.  创建一个带有半径的`Circle`结构。它应该是一个`struct`，因为它是一个简单的数据对象，具有一点点逻辑，计算面积。

1.  添加一个属性来获取圆的面积（尝试使用表达式主体成员）。记住，圆的面积公式是`pi*r*r`。要使用`PI`常量，你需要导入`Math`包。

1.  将两个圆的面积相加。最自然的方法是使用加号（`+`）运算符的重载。实现一个接受两个圆并返回一个新圆的`+`运算符重载。新圆的面积是两个旧圆的面积之和。然而，不要通过传递面积来创建一个新圆。你需要一个半径。你可以通过将新面积除以`PI`然后取结果的平方根来计算这个半径。

1.  现在创建一个`Solution`类，它接受两个圆并返回一个结果——新圆的半径。

1.  在`main`方法中，创建两个半径为`3`厘米的圆，并定义一个新的圆，它等于两个其他圆的面积之和。打印结果。

1.  运行`main`方法，结果应该如下：

```cpp
Adding circles of radius of 3 and 3 results in a new circle with a radius 4.242640687119285
```

从最终输出中可以看出，新的圆的半径将是`4.24`（四舍五入到小数点后两位）。

注

这个活动的解决方案可以在[`packt.link/qclbF`](https://packt.link/qclbF)找到。

这个活动旨在测试你创建类和重载运算符的知识。通常不会使用运算符来解决这种问题，但在这种情况下，效果很好。

# 总结

在本章中，你学习了面向对象编程以及它如何帮助将复杂问题抽象成简单概念。C#有几个有用的特性，大约每一到两年就会发布一个新的语言版本。本章提到的特性只是 C#在提高生产力方面的一些方式。你已经看到了，通过设计，它允许更好、更清晰的代码，更不容易出错。C#在提高生产力方面是最好的语言之一。使用 C#，你可以编写有效的代码，并且快速，因为很多样板代码都已经为你做好了。

最后，你学会了 SOLID 原则并在一个应用中使用它们。SOLID 不是你可以立即阅读和学习的东西；在你掌握并开始一致应用之前，需要练习、与同行讨论和大量的试错。然而，好处是值得的。在现代软件开发中，生产快速、最佳代码不再是头等大事。如今，重点是生产力（你开发的速度）和性能（你的程序运行的速度）的平衡。C#是最高效的语言之一，无论是性能还是生产力方面。

在下一章中，您将学习什么是函数式编程，以及如何使用 lambda 和函数构造，比如委托。
