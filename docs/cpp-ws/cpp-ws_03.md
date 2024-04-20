# 3. 委托、事件和 Lambda

概述

在这一章中，您将学习如何定义和调用委托，并探索它们在.NET 生态系统中的广泛使用。有了这些知识，您将继续使用内置的`Action`和`Func`委托，以发现它们的使用如何减少不必要的样板代码。然后，您将看到如何利用多播委托向多个方发送消息，以及如何将事件纳入事件驱动的代码中。在这个过程中，您将发现一些常见的陷阱要避免，以及一些最佳实践，以防止一个优秀的应用程序变成一个不可靠的应用程序。

本章将揭开 lambda 语法风格的神秘面纱，并展示如何有效地使用它。在本章结束时，您将能够舒适地使用 lambda 语法来创建简洁、易于理解和维护的代码。

# 介绍

在上一章中，您学习了面向对象编程（OOP）的一些关键方面。在本章中，您将通过查看 C#中专门用于使类进行交互的常见模式来进一步学习。

您是否发现自己正在处理必须监听某些信号并对其进行操作的代码，但在运行时无法确定这些操作应该是什么？也许您有一段代码块需要重用或传递给其他方法，在它们准备好调用时。或者，您可能想要过滤对象列表，但需要根据用户偏好的组合来确定如何进行过滤。使用接口可以实现这些，但通常更有效的方法是创建代码块，然后以类型安全的方式将其传递给其他类。这些代码块被称为委托，并构成许多.NET 库的基础，允许将方法或代码片段作为参数传递。

委托的自然扩展是事件，它使得在软件中提供一种可选行为成为可能。例如，您可能有一个组件可以广播实时新闻和股票价格，但除非您提供一种选择这些服务的方式，否则您可能会限制这种组件的可用性。

用户界面（UI）应用程序通常会提供各种用户操作的通知，例如按键、滑动屏幕或点击鼠标按钮；这些通知遵循 C#中的标准模式，在本章中将对此进行全面讨论。在这种情况下，检测此类操作的 UI 元素被称为发布者，而对这些消息进行操作的代码被称为订阅者。当它们聚集在一起时，它们形成了一个称为发布者-订阅者或 pub-sub 模式的事件驱动设计。您将看到这可以在所有类型的 C#中使用。请记住，它的使用不仅仅局限于 UI 应用程序的领域。

最后，您将学习关于 lambda 语句和 lambda 表达式，统称为 lambda。这些具有不寻常的语法，最初可能需要一段时间才能适应。与在类中散布大量方法和函数不同，lambda 允许使用更小的代码块，这些代码块通常是自包含的，并且位于代码中使用它们的地方附近，从而提供了一种更容易遵循和维护代码的方式。您将在本章的后半部分详细了解 lambda。首先，您将学习关于委托。

# 委托

.NET 委托类似于其他语言中的函数指针，比如 C++；换句话说，它就像是一个在运行时调用的方法的指针。实质上，它是一段代码的占位符，可以是一个简单的语句，也可以是一个完整的多行代码块，包括复杂的执行分支，您可以要求其他代码在某个时间点执行。委托一词暗示了某种**代表**，这正是这个占位符概念所涉及的。

委托允许对象之间的耦合最小化，并且代码量大大减少。无需创建从特定类或接口派生的类。通过使用委托，您正在定义兼容方法应该是什么样子，无论它是在类或结构、静态或基于实例。参数和返回类型定义了这种调用兼容性。

此外，委托可以以回调方式使用，允许将多个方法连接到单个发布源。它们通常需要更少的代码，并提供比使用基于接口的设计更多的功能。

以下示例显示了委托可以有多么有效。假设您有一个按姓氏搜索用户的类。它可能看起来像这样：

```cpp
public User FindBySurname(string name)
{
    foreach(var user in _users)
       if (user.Surname == name)
          return user;
    return null;
}
```

然后，您需要扩展此功能以包括对用户登录名的搜索：

```cpp
public User FindByLoginName(string name)
{
    foreach(var user in _users)
       if (user.LoginName == name)
          return user;
    return null;
}
```

再次，您决定添加另一个搜索，这次是按位置搜索：

```cpp
public User FindByLocation(string name)
{
    foreach(var user in _users)
       if (user.Location == name)
          return user;
    return null;
}
```

您可以使用以下代码开始搜索：

```cpp
public void DoSearch()
{
  var user1 = FindBySurname("Wright");
  var user2 = FindByLoginName("JamesR");
  var user3 = FindByLocation("Scotland"); 
}
```

您能看到每次发生的模式吗？您重复了遍历用户列表的相同代码，应用布尔条件（也称为谓词）以找到第一个匹配的用户。

唯一不同的是谓词决定是否找到了匹配项。这是委托在基本级别上使用的常见情况之一。`predicate`可以被替换为一个委托，充当占位符，在需要时进行评估。

将此代码转换为委托样式，您定义了一个名为`FindUser`的委托（可以跳过此步骤，因为.NET 包含一个可以重用的委托定义；稍后您将了解到这一点）。

只需要一个名为`Find`的单一辅助方法，它接受一个`FindUser`委托实例。Find 知道如何循环遍历用户，调用传入用户的委托，返回 true 或 false 以进行匹配：

```cpp
private delegate bool FindUser(User user);
private User Find(FindUser predicate)
{
  foreach (var user in _users)
    if (predicate(user))
      return user;
  return null;
}
public void DoSearch()
{
  var user4 = Find(user => user.Surname == "Wright");
  var user5 = Find(user => user.LoginName == "JamesR");
  var user6 = Find(user => user.Location == "Scotland");
}
```

正如您所看到的，代码现在被保持在一起，更加简洁。无需剪切和粘贴循环遍历用户的代码，因为所有这些都在一个地方完成。对于每种类型的搜索，您只需一次定义一个委托并将其传递给`Find`。要添加新类型的搜索，您只需要在一条语句中定义它，而不是复制至少八行重复循环功能的代码。

Lambda 语法是一种用于定义方法体的基本样式，但其奇怪的语法可能首先会成为障碍。乍一看，Lambda 表达式的`=>`样式可能看起来很奇怪，但它们确实提供了一种更清晰的指定目标方法的方式。定义 Lambda 的行为类似于定义方法；您基本上省略了方法名，并使用`=>`来前缀一段代码块。

现在，您将看另一个示例，这次使用接口。假设您正在开发一个图形引擎，并且需要在用户旋转或缩放时每次计算图像在屏幕上的位置。请注意，此示例跳过了任何复杂的数学计算。

考虑到您需要使用具有名为`Move`的单个方法的`ITransform`接口来转换`Point`类，如下面的代码片段所示：

```cpp
public class Point
{
  public double X { get; set; } 
  public double Y { get; set; }
}
public interface ITransform
{
  Point Move(double height, double width);
}
```

当用户旋转对象时，您需要使用`RotateTransform`，而对于缩放操作，您将使用`ZoomTransform`，如下所示。两者都基于`ITransform`接口：

```cpp
public class RotateTransform : ITransform
{
    public Point Move(double height, double width)
    {
        // do stuff
        return new Point();
    }
}
public class ZoomTransform : ITransform
{
    public Point Move(double height, double width)
    {
        // do stuff
        return new Point();
    }
}
```

因此，鉴于这两个类，可以通过创建一个新的`Transform`实例来转换一个点，该实例被传递给一个名为`Calculate`的方法，如下面的代码所示。`Calculate`调用相应的`Move`方法，并对点进行一些额外的未指定的工作，然后将点返回给调用者：

```cpp
public class Transformer
{
    public void Transform()
    {
        var rotatePoint = Calculate(new RotateTransform(), 100, 20);
        var zoomPoint = Calculate(new ZoomTransform(), 5, 5);
    }
    private Point Calculate(ITransform transformer, double height, double width)
    {
        var point = transformer.Move(height, width);
        //do stuff to point
        return point;
    }
}
```

这是一个标准的基于类和接口的设计，但您可以看到，您已经付出了很多努力，只是从`Move`方法中获得了一个单一的数值。将计算分解成易于遵循的实现是一个值得的想法。毕竟，如果在一个方法中实现了多个 if-then 分支，可能会导致未来的维护问题。

通过重新实现基于委托的设计，您仍然可以拥有可维护的代码，但要处理的代码量要少得多。您可以有一个`TransformPoint`委托和一个新的`Calculate`函数，该函数可以传递一个`TransformPoint`委托。

您可以通过在其名称周围添加括号并传递任何参数来调用委托。这类似于调用标准的类级函数或方法。稍后您将更详细地介绍这种调用；现在，请考虑以下片段：

```cpp
    private delegate Point TransformPoint(double height, double width);
    private Point Calculate(TransformPoint transformer, double height, double width)
    {
        var point = transformer(height, width);
        //do stuff to point
        return point;
    }
```

您仍然需要实际的目标`Rotate`和`Zoom`方法，但您不需要创建不必要的类来执行这些操作。您可以添加以下代码：

```cpp
    private Point Rotate(double height, double width)
    {
        return new Point();
    }
    private Point Zoom(double height, double width)
    {
        return new Point();
    }
```

现在，调用方法委托就像下面这样简单：

```cpp
    public void Transform()
    {
         var rotatePoint1 = Calculate(Rotate, 100, 20);
         var zoomPoint1 = Calculate(Zoom, 5, 5);
    }
```

注意，使用委托的方式有助于消除大量不必要的代码。

注意

您可以在[`packt.link/AcwZA`](https://packt.link/AcwZA)找到此示例使用的代码。

除了调用单个占位符方法之外，委托还包含额外的管道，使其能够以**多播**的方式使用，即一种将多个目标方法链接在一起的方式，每个方法依次被调用。这通常被称为调用列表或委托链，并由充当发布源的代码发起。

这个多播概念应用的一个简单例子可以在 UI 中看到。想象一下，您有一个显示国家地图的应用程序。当用户在地图上移动鼠标时，您可能希望执行各种操作，例如以下操作：

+   在鼠标悬停在建筑物上时，将鼠标指针更改为不同的形状。

+   显示一个工具提示，计算真实世界的经度和纬度坐标。

+   在状态栏中显示一个消息，计算鼠标悬停区域的人口。

为了实现这一点，您需要一种方法来检测用户在屏幕上移动鼠标的方式。这通常被称为发布者。在这个例子中，它的唯一目的是检测鼠标移动并将其发布给任何正在监听的人。

为了执行三个必需的 UI 操作，您可以创建一个类，该类具有一个对象列表，当鼠标位置发生变化时通知这些对象，使每个对象能够独立于其他对象执行其所需的任何活动。这些对象中的每一个被称为订阅者。

当您的发布者检测到鼠标移动时，您可以按照以下伪代码进行操作：

```cpp
MouseEventArgs args = new MouseEventArgs(100,200)
foreach(subscription in subscriptionList)
{
   subscription.OnMouseMoved(args)
} 
```

这假设`subscriptionList`是一个对象列表，可能基于具有`OnMouseMoved`方法的接口。您需要添加代码，使感兴趣的各方能够订阅和取消订阅`OnMouseMoved`通知。如果以前订阅的代码没有取消订阅的方法，并且在不再需要调用它时被重复调用，那将是一个不幸的设计。

在前面的代码中，发布者和订阅者之间存在相当多的耦合，并且您又开始使用接口进行类型安全的实现。如果您需要监听按键按下和松开，会怎么样？如果您不得不反复复制这样相似的代码，很快就会变得相当沮丧。

幸运的是，委托类型包含所有这些内置行为。您可以交替使用单个或多个目标方法；您只需要调用委托，委托将为您处理其余的工作。

不久之后，您将深入研究多播委托，但首先，您将探索单目标方法场景。

## 定义自定义委托

委托的定义方式与标准方法的定义方式类似。编译器不关心目标方法体中的代码，只关心它在某个时间点上可以安全地被调用。

使用`delegate`关键字来定义委托，格式如下：

```cpp
public delegate void MessageReceivedHandler(string message, int size);
```

以下列表描述了此语法的每个组件：

+   范围：访问修饰符，如`public`、`private`或`protected`，用于定义委托的范围。如果不包括修饰符，编译器将默认将其标记为私有，但最好明确显示代码的意图。

+   `delegate`关键字。

+   返回类型：如果没有返回类型，将使用`void`。

+   委托名称：这可以是任何您喜欢的东西，但名称必须在命名空间内是唯一的。许多命名约定（包括微软的）建议在委托的名称中添加`Handler`或`EventHandler`。

+   如果需要，参数。

注意

委托可以嵌套在类或命名空间中；它们也可以在全局命名空间中定义，尽管这种做法是不鼓励的。在 C#中定义类时，通常习惯于在父命名空间中定义它们，通常基于以公司名称开头的分层约定，然后是产品名称，最后是功能。这有助于为类型提供更独特的标识。

通过在没有命名空间的情况下定义委托，很有可能会与另一个具有相同名称的委托发生冲突，如果它也在没有命名空间保护的库中定义。这可能会导致编译器对你所指的委托感到困惑。

在较早版本的.NET 中，定义自定义委托是常见做法。这样的代码已经被各种内置的.NET 委托所取代，您很快将看到。现在，您将简要介绍定义自定义委托的基础知识。如果您维护任何旧的 C#代码，了解这一点是值得的。

在下一个练习中，您将创建一个自定义委托，该委托传递一个`DateTime`参数并返回一个布尔值以指示有效性。

## 练习 3.01：定义和调用自定义委托

假设您有一个允许用户订购产品的应用程序。在填写订单详细信息时，客户可以指定订单日期和交货日期，这两者在接受订单之前必须经过验证。您需要一种灵活的方式来验证这些日期。对于一些客户，您可以允许周末交货日期，而对于其他客户，必须至少提前七天。您还可以允许某些客户对订单进行回溯。

您知道委托提供了一种在运行时变化实现的方式，因此这是继续的最佳方式。您不希望使用多个接口，或者更糟糕的是，一堆复杂的`if-then`语句来实现这一点。

根据客户的配置文件，您可以创建一个名为`Order`的类，该类可以传递不同的日期验证规则。这些规则可以通过`Validate`方法进行验证：

执行以下步骤：

1.  创建一个名为`Chapter03`的新文件夹。

1.  切换到`Chapter03`文件夹并使用 CLI `dotnet`命令创建一个名为`Exercise01`的新控制台应用程序，如下所示：

```cpp
source\Chapter03>dotnet new console -o Exercise01
```

您将看到以下输出：

```cpp
The template "Console Application" was created successfully.
Processing post-creation actions...
Running 'dotnet restore' on Exercise01\Exercise01.csproj...
  Determining projects to restore...
  Restored source\Chapter03\Exercise01\Exercise01.csproj (in 191 ms).
Restore succeeded.
```

1.  打开`Chapter03\Exercise01.csproj`并用以下设置替换内容：

```cpp
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net6.0</TargetFramework>
  </PropertyGroup>
</Project>
```

1.  打开`Exercise01\Program.cs`并清空内容。

1.  早些时候提到了使用命名空间来防止与其他库中的对象发生冲突的偏好，因此为了保持事物的隔离，使用`Chapter03.Exercise01`作为命名空间。

为了实现您的日期验证规则，您将定义一个委托，该委托接受一个`DateTime`参数并返回一个布尔值。您将其命名为`DateValidationHandler`：

```cpp
using System;
namespace Chapter03.Exercise01 
{
    public delegate bool DateValidationHandler(DateTime dateTime);
}
```

1.  接下来，您将创建一个名为`Order`的类，其中包含订单的详细信息，并可以传递给两个日期验证委托：

```cpp
   public class Order
    {
        private readonly DateValidationHandler _orderDateValidator;
        private readonly DateValidationHandler _deliveryDateValidator;
```

请注意，您已声明了两个只读的类级别`DateValidationHandler`实例，一个用于验证订单日期，另一个用于验证交货日期。 此设计假定日期验证规则不会为此`Order`实例更改。

1.  现在对于构造函数，您传递了两个委托：

```cpp
       public Order(DateValidationHandler orderDateValidator,
            DateValidationHandler deliveryDateValidator)
        {
            _orderDateValidator = orderDateValidator;
            _deliveryDateValidator = deliveryDateValidator;
        }  
```

在此设计中，通常由不同的类负责根据所选客户的配置文件决定使用哪些委托。

1.  您需要添加要验证的两个日期属性。 这些日期可以使用监听按键并直接将用户编辑应用于此类的 UI 来设置：

```cpp
        public DateTime OrderDate { get; set; }
        public DateTime DeliveryDate { get; set; }
```

1.  现在添加一个`IsValid`方法，将`OrderDate`传递给`orderDateValidator`委托，并将`DeliveryDate`传递给`deliveryDateValidator`委托：

```cpp
        public bool IsValid() => 
            _orderDateValidator(OrderDate) &&
            _deliveryDateValidator(DeliveryDate);
    }
```

如果两者都有效，则此调用将返回`true`。 关键在于`Order`不需要了解单个客户的日期验证规则的具体实现，因此您可以轻松地在程序的其他位置重用`Order`。 要调用委托，只需将任何参数括在括号中，本例中将正确的日期属性传递给每个委托实例：

1.  要创建一个控制台应用程序来测试这一点，请添加一个名为`Program`的`static`类：

```cpp
    public static class Program
    {
```

1.  您希望创建两个函数，用于验证传递给它们的日期是否有效。 这些函数将成为您的委托目标方法的基础：

```cpp
        private static bool IsWeekendDate(DateTime date)
        {
            Console.WriteLine("Called IsWeekendDate");
            return date.DayOfWeek == DayOfWeek.Saturday ||
                   date.DayOfWeek == DayOfWeek.Sunday;
        }
        private static bool IsPastDate(DateTime date)
        {
            Console.WriteLine("Called IsPastDate");
            return date < DateTime.Today;
        }
```

请注意，两者都具有`DateValidationHandler`委托所期望的确切签名。 他们都不知道他们正在验证的日期的性质，因为这不是他们关心的事情。 他们都标记为`static`，因为它们不与此类中的任何变量或属性交互。

1.  现在是`Main`入口点。 在这里，您创建了两个`DateValidationHandler`委托实例，将`IsPastDate`传递给一个委托，将`IsWeekendDate`传递给第二个委托。 这些是在调用每个委托时将被调用的目标方法：

```cpp
        public static void Main()
        {
           var orderValidator = new DateValidationHandler(IsPastDate);
           var deliverValidator = new DateValidationHandler(IsWeekendDate);
```

1.  现在，您可以创建一个`Order`实例，传递委托并设置订单和交货日期：

```cpp
          var order = new Order(orderValidator, deliverValidator)
            {
                OrderDate = DateTime.Today.AddDays(-10), 
                DeliveryDate = new DateTime(2020, 12, 31)
            };
```

有多种方法可以创建委托。 在这里，您首先将它们分配给变量，以使代码更清晰（稍后将介绍不同的样式）。

1.  现在只需在控制台中显示日期并调用`IsValid`，然后`IsValid`将依次调用您的每个委托方法。 请注意，使用自定义日期格式使日期更易读：

```cpp
          Console.WriteLine($"Ordered: {order.OrderDate:dd-MMM-yy}");
          Console.WriteLine($"Delivered: {order.DeliveryDate:dd-MMM-yy }");
          Console.WriteLine($"IsValid: {order.IsValid()}");
        }
    }
}
```

1.  运行控制台应用程序会产生以下输出：

```cpp
Ordered: 07-May-22
Delivered: 31-Dec-20
Called IsPastDate
Called IsWeekendDate
IsValid: False
```

此顺序**无效**，因为交货日期是星期四，而不是周末，正如您所要求的：

您已经学会了如何定义自定义委托，并创建了两个实例，这些实例使用小的辅助函数来验证日期。 这使您了解了委托可以有多灵活的想法。

注意

您可以在[`packt.link/cmL0s`](https://packt.link/cmL0s)找到此练习使用的代码。

## 内置的 Action 和 Func 委托

当您定义委托时，您正在描述其签名，即返回类型和输入参数列表。 也就是说，考虑这两个委托：

```cpp
public delegate string DoStuff(string name, int age);
public delegate string DoMoreStuff(string name, int age);
```

它们都具有相同的签名，但仅通过名称不同，这就是为什么您可以声明每个实例并在调用时**都**指向**相同**的目标方法：

```cpp
public static void Main()
{
    DoStuff stuff = new DoStuff(MyMethod);
    DoMoreStuff moreStuff = new DoMoreStuff(MyMethod);
    Console.WriteLine($"Stuff: {stuff("Louis", 2)}");
    Console.WriteLine($"MoreStuff: {moreStuff("Louis", 2)}");
}
private static string MyMethod(string name, int age)
{
    return $"{name}@{age}";
}
```

运行控制台应用程序会产生两次调用相同的结果：

```cpp
Stuff: Louis@2
MoreStuff: Louis@2
```

注意

您可以在[`packt.link/r6B8n`](https://packt.link/r6B8n)找到此示例使用的代码。

如果您可以不定义`DoStuff`和`DoMoreStuff`委托并使用具有完全相同签名的更通用的委托，那将是很好的。 毕竟，在前面的片段中，如果您创建了`DoStuff`或`DoMoreStuff`委托，都不重要，因为两者都调用相同的目标方法。

.NET 实际上提供了各种内置委托，您可以直接使用这些委托，而无需自己定义这些委托。 这些是`Action`和`Func`委托。

`Action`和`Func`委托有许多可能的组合，每个组合都允许越来越多的参数。你可以指定从零到 16 个不同的参数类型。由于有这么多的组合可用，你极有可能永远不需要定义自己的委托类型。

值得注意的是，`Action`和`Func`委托是在.NET 的较新版本中添加的，因此自定义委托的使用往往可以在较旧的遗留代码中找到。没有必要自己创建新的委托。

在下面的片段中，`MyMethod`使用了三参数的`Func`变体进行调用；你很快就会涵盖到那个看起来有点奇怪的`<string, int, string>`语法：

```cpp
Func<string, int, string> funcStuff = MyMethod;
Console.WriteLine($"FuncStuff: {funcStuff("Louis", 2)}");
```

这产生了与前两个调用相同的返回值：

```cpp
FuncStuff: Louis@2
```

在继续探索`Action`和`Func`委托之前，探索`Action<string, int, string>`语法会很有用。这种语法允许使用类型参数来定义类和方法。这些被称为泛型，并充当特定类型的占位符。在*第四章*，*数据结构和 LINQ*中，你将更详细地介绍泛型，但在这里用`Action`和`Func`委托总结它们的用法是值得的。

`Action`委托的非泛型版本在.NET 中预定义如下：

```cpp
public delegate void Action()
```

正如你从之前对委托的了解，这是一个不带任何参数且没有返回类型的委托；这是可用的最简单的委托类型。

与.NET 中预定义的一个泛型`Action`委托相对比：

```cpp
public delegate void Action<T>(T obj)
```

你可以看到这包括一个`<T>`和`T`参数部分，这意味着它接受一个被限制为字符串的`Action`，它接受一个字符串参数并且不返回值，如下所示：

```cpp
Action<string> actionA;
```

再来一个被限制为`int`的版本？这也没有返回类型，接受一个`int`参数：

```cpp
Action<int> actionB;
```

你能看到这里的模式吗？实质上，你指定的类型可以用来在编译时声明一个类型。如果你想要两个参数，或三个，或四个…或 16 个呢？简单。有`Action`和`Func`泛型类型，可以接受**16**种不同的参数类型。你很少会写需要超过 16 个参数的代码。

这个两参数的`Action`接受`int`和`string`作为参数：

```cpp
Action<int, string> actionC;
```

你可以把它转过来。这里是另一个两参数的`Action`，但是这个接受一个`string`参数，然后是一个`int`参数：

```cpp
Action<string, int> actionD;
```

这些涵盖了大多数参数组合，所以你可以看到很少需要创建自己的委托类型。

委托返回值也适用相同的规则；这就是`Func`类型被使用的地方。通用的`Func`类型以单个值类型参数开始：

```cpp
public delegate T Func<T>()
```

在下面的例子中，`funcE`是一个返回布尔值且不带参数的委托：

```cpp
Func<bool> funcE;
```

你能猜到这个相当长的`Func`声明的返回类型是什么吗？

```cpp
Func<bool, int, int, DateTime, string> funcF;
```

这给出了一个返回`string`的委托。换句话说，在`Func`中的最后一个参数类型定义了返回类型。注意`funcF`接受四个参数：`bool`、`int`、`int`和`DateTime`。

总之，泛型是定义类型的一种很好的方式。它们通过允许类型参数充当占位符来节省了大量重复的代码。

## 分配委托

你在*练习 3.01*中介绍了创建自定义委托以及如何分配和调用委托的简要方法。然后你看了使用首选的`Action`和`Func`等价物，但是你还有哪些其他选项来分配形成委托的方法（或方法）？有其他方式来调用委托吗？

委托可以被分配给一个变量，就像你可能会分配一个类实例一样。你也可以传递新实例或静态实例，而不必使用变量来这样做。一旦分配，你可以调用委托或将引用传递给其他类，以便它们可以调用它，这在框架 API 中经常这样做。

现在您将查看一个`Func`委托，它接受一个`DateTime`参数并返回一个`bool`值来指示有效性。您将使用一个包含两个帮助方法的`static`类，这些方法形成了实际的目标：

```cpp
public static class DateValidators
{
    public static bool IsWeekend(DateTime dateTime)
        => dateTime.DayOfWeek == DayOfWeek.Saturday ||
           dateTime.DayOfWeek == DayOfWeek.Sunday;
    public static bool IsFuture(DateTime dateTime) 
      => dateTime.Date > DateTime.Today;
}
```

注意

您可以在[`packt.link/mwmxh`](https://packt.link/mwmxh)找到此示例的代码。

请注意，`DateValidators`类标记为`static`。您可能听说过短语**静态是低效的**。换句话说，创建具有许多静态类的应用程序是一种薄弱的做法。静态类在首次被运行代码访问时实例化，并且会一直保留在内存中，直到应用程序关闭。这使得难以控制它们的生命周期。将小型实用程序类定义为静态类不是问题，前提是它们确实保持无状态。无状态意味着它们不设置任何局部变量。设置局部状态的静态类非常难以进行单元测试；您永远无法确定设置的变量是来自一个测试还是另一个测试。

在前面的片段中，如果`DateTime`参数的`Date`属性晚于当前日期，则`IsFuture`返回`true`。您正在使用静态的`DateTime.Today`属性来检索当前系统日期。使用表达式主体语法定义了`IsWeekend`，如果`DateTime`参数的星期几是星期六或星期日，则将返回`true`。

您可以像分配常规变量一样分配委托（记住您要做`futureValidator`和`weekendValidator`）。每个构造函数分别传递实际的目标方法，即`IsFuture`或`IsWeekend`实例：

```cpp
var futureValidator = new Func<DateTime, bool>(DateValidators.IsFuture);
var weekendValidator = new Func<DateTime, bool>(DateValidators.IsWeekend);
```

请注意，使用`var`关键字分配委托而不包装在`Func`前缀中是无效的：

```cpp
var futureValidator = DateValidation.IsFuture;
```

这将导致以下编译器错误：

```cpp
Cannot assign method group to an implicitly - typed variable
```

掌握了委托的这些知识后，继续了解如何调用委托。

## 调用委托

有几种调用委托的方法。例如，考虑以下定义：

```cpp
var futureValidator = new Func<DateTime, bool>(DateValidators.IsFuture);
```

要调用`futureValidator`，您必须传入一个`DateTime`值，并且它将返回一个`bool`值，可以使用以下任何一种样式：

+   使用空合并运算符调用：

```cpp
var isFuture1 = futureValidator?.Invoke(new DateTime(2000, 12, 31));
```

这是首选且最安全的方法；在调用`Invoke`之前，您应该始终检查空值。如果委托有可能不指向内存中的对象，则在访问方法和属性之前必须执行空引用检查。不这样做将导致抛出`NullReferenceException`。这是运行时警告您对象没有指向任何内容的方式。

通过使用空合并运算符，编译器将为您添加空检查。在代码中，您明确声明了`futureValidator`，因此它在这里不可能为空。但是，如果您从另一个方法中传递了`futureValidator`会怎么样？您如何确保调用者已正确分配了引用？

委托有额外的规则，使它们在被调用时可能会抛出`NullReferenceException`。在前面的例子中，`futureValidator`有一个单一的目标，但正如您将在后面看到的，会抛出`NullReferenceException`。

+   直接调用

这与以前的方法相同，但没有空检查的安全性。出于同样的原因，这是不推荐的；也就是说，委托可能会抛出`NullReferenceException`：

```cpp
var isFuture1 = futureValidator.Invoke(new DateTime(2000, 12, 31));
```

+   没有`Invoke`前缀

这看起来更简洁，因为您只需调用委托，而无需`Invoke`前缀。同样，由于可能存在空引用，这是不推荐的：

```cpp
var isFuture2 = futureValidator(new DateTime(2050, 1, 20));
```

通过练习将委托分配和安全调用结合在一起。

## 练习 3.02：分配和调用委托

在这个练习中，你将编写一个控制台应用程序，展示如何使用`Func`委托来提取数值。你将创建一个`Car`类，它有`Distance`和`JourneyTime`属性。你将提示用户输入昨天和今天的行驶距离，将这些信息传递给一个`Comparison`类，告诉它如何提取值并计算它们的差异。

执行以下步骤来完成： 

1.  切换到`Chapter03`文件夹，并使用 CLI `dotnet`命令创建一个名为`Exercise02`的新控制台应用程序：

```cpp
source\Chapter03>dotnet new console -o Exercise02
```

1.  打开`Chapter03\Exercise02.csproj`，并用以下设置替换整个文件：

```cpp
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net6.0</TargetFramework>
  </PropertyGroup>
</Project>
```

1.  打开`Exercise02\Program.cs`并清空内容。

1.  首先添加一个名为`Car`的记录。包括`System.Globalization`命名空间以进行字符串解析。使用`Chapter03.Exercise02`命名空间将代码与其他练习分开。

1.  添加两个属性，`Distance`和`JourneyTime`。它们将具有`init`-only 属性，因此你将使用`init`关键字：

```cpp
using System;
using System.Globalization;
namespace Chapter03.Exercise02
{
    public record Car
    {
        public double Distance { get; init; }
        public double JourneyTime { get; init; }
    }
```

1.  接下来，创建一个名为`Comparison`的类，它被传递一个`Func`委托来使用。`Comparison`类将使用委托来提取`Distance`或`JourneyTime`属性，并计算两个`Car`实例的差异。通过使用委托的灵活性，`Comparison`将不知道它是提取`Distance`还是`JourneyTime`，只知道它使用一个`double`来计算差异。这表明你可以在将来需要计算其他`Car`属性时重用这个类：

```cpp
    public class Comparison
    {
        private readonly Func<Car, double> _valueSelector;
        public Comparison(Func<Car, double> valueSelector)
        {
            _valueSelector = valueSelector;
        } 
```

1.  添加三个属性，形成计算结果，如下：

```cpp
        public double Yesterday { get; private set; }
        public double Today { get; private set; }
        public double Difference { get; private set; }
```

1.  现在进行计算，传入两个`Car`实例，一个是昨天的汽车行程`yesterdayCar`，另一个是今天的`todayCar`：

```cpp
        public void Compare(Car yesterdayCar, Car todayCar)
        {
```

1.  要计算`Yesterday`的值，调用`valueSelector` `Func`委托，传入`yesterdayCar`实例。再次记住，`Comparison`类并不知道它是提取`Distance`还是`JourneyTime`；它只需要知道当`delegate`被`Car`参数调用时，它将得到一个`double`数字：

```cpp
            Yesterday = _valueSelector(yesterdayCar);
```

1.  同样的方法提取`Today`的值，使用相同的`Func`委托，但传入`todayCar`实例： 

```cpp
            Today = _valueSelector(todayCar);
```

1.  现在只是计算两个提取的数字之间的差异；你不需要使用`Func`委托来做到这一点：

```cpp
            Difference = Yesterday - Today;
        }
     }
```

1.  因此，你有一个知道如何调用`Func`委托来提取特定`Car`属性的类，当它被告知如何操作时。现在，你需要一个类来封装`Comparison`实例。为此，添加一个名为`JourneyComparer`的类：

```cpp
    public class JourneyComparer
    {
        public JourneyComparer()
        {
```

1.  对于汽车行程，你需要计算`Yesterday`和`Today`的`Distance`属性之间的差异。为此，创建一个`Comparison`类，告诉它如何从`Car`实例中提取值。你可能会使用相同的名称来为这个`Comparison`类，因为你将提取汽车的`Distance`。记住，`Comparison`构造函数需要一个`Func`委托，它被传递一个`Car`实例并返回一个`double`值。你将很快添加`GetCarDistance()`；这将最终通过传递昨天和今天的行程的`Car`实例来调用：

```cpp
          Distance = new Comparison(GetCarDistance);
```

1.  按照前面步骤中描述的过程重复这个过程，用于`JourneyTime` `Comparison`；这个应该被告知使用`GetCarJourneyTime()`如下：

```cpp
          JourneyTime = new Comparison(GetCarJourneyTime);
```

1.  最后，添加另一个名为`AverageSpeed`的`Comparison`属性，如下。你很快会看到`GetCarAverageSpeed()`是另一个函数：

```cpp
           AverageSpeed = new Comparison(GetCarAverageSpeed);
```

1.  现在对于`GetCarDistance`和`GetCarJourneyTime`本地函数，它们被传递一个`Car`实例，并根据需要返回`Distance`或`JourneyTime`：

```cpp
           static double GetCarDistance(Car car) => car.Distance; 
           static double GetCarJourneyTime(Car car) => car.JourneyTime;
```

1.  `GetCarAverageSpeed`，顾名思义，返回平均速度。在这里，你已经表明`Func`委托只需要一个兼容的函数；只要返回值是`double`，它返回的内容并不重要。`Comparison`类在调用`Func`委托时并不需要知道它返回的是这样的一个计算值：

```cpp
          static double GetCarAverageSpeed(Car car)             => car.Distance / car.JourneyTime;
       }
```

1.  三个`Comparison`属性应该定义如下：

```cpp
        public Comparison Distance { get; }
        public Comparison JourneyTime { get; }
        public Comparison AverageSpeed { get; }
```

1.  现在是主要的`Compare`方法。这将传递两个`Car`实例，一个用于`昨天`，一个用于`今天`，然后简单地调用三个`Comparison`项上的`Compare`，传入两个`Car`实例：

```cpp
        public void Compare(Car yesterday, Car today)
        {
            Distance.Compare(yesterday, today);
            JourneyTime.Compare(yesterday, today);
            AverageSpeed.Compare(yesterday, today);
        }
    }
```

1.  您需要一个控制台应用程序来输入每天的行驶里程，因此添加一个名为`Program`的类，并具有静态`Main`入口点：

```cpp
    public class Program
    {
        public static void Main()
        {
```

1.  您可以随机分配旅行时间以保存一些输入，因此添加一个新的`Random`实例和一个`do-while`循环的开始，如下所示：

```cpp
            var random = new Random();
            string input;
            do
            {
```

1.  阅读昨天的距离，如下所示：

```cpp
                Console.Write("Yesterday's distance: ");
                input = Console.ReadLine();
                double.TryParse(input, NumberStyles.Any,                    CultureInfo.CurrentCulture, out var distanceYesterday);
```

1.  您可以使用距离创建昨天的`Car`，并使用随机的`JourneyTime`，如下所示：

```cpp
                var carYesterday = new Car
                {
                    Distance = distanceYesterday,
                    JourneyTime = random.NextDouble() * 10D
                };
```

1.  对于今天的距离也是如此：

```cpp
                Console.Write("    Today's distance: ");
                input = Console.ReadLine();
                double.TryParse(input, NumberStyles.Any,                    CultureInfo.CurrentCulture, out var distanceToday);
                var carToday = new Car
                {
                    Distance = distanceToday,
                    JourneyTime = random.NextDouble() * 10D
                };
```

1.  现在，您有两个填充了昨天和今天的`Car`实例，您可以创建`JourneyComparer`实例并调用`Compare`。然后，这将在三个`Comparison`实例上调用`Compare`：

```cpp
                var comparer = new JourneyComparer();
                comparer.Compare(carYesterday, carToday);
```

1.  现在，将结果写入控制台：

```cpp
                Console.WriteLine();
                Console.WriteLine("Journey Details   Distance\tTime\tAvg Speed");
                Console.WriteLine("-------------------------------------------------");
```

1.  写出昨天的结果：

```cpp
                Console.Write($"Yesterday         {comparer.Distance.Yesterday:N0}   \t");
                Console.WriteLine($"{comparer.JourneyTime.Yesterday:N0}\t {comparer.AverageSpeed.Yesterday:N0}");
```

1.  写出今天的结果：

```cpp
                Console.Write($"Today             {comparer.Distance.Today:N0}     \t");                 Console.WriteLine($"{comparer.JourneyTime.Today:N0}\t {comparer.AverageSpeed.Today:N0}");
```

1.  最后，使用`Difference`属性写入摘要值：

```cpp
                Console.WriteLine("=================================================");
                Console.Write($"Difference             {comparer.Distance.Difference:N0}     \t");                Console.WriteLine($"{comparer.JourneyTime.Difference:N0} \t{comparer.AverageSpeed.Difference:N0}");
               Console.WriteLine("=================================================");
```

1.  完成`do-while`循环，如果用户输入空字符串，则退出：

```cpp
            } 
            while (!string.IsNullOrEmpty(input));
        }
    }
}
```

运行控制台并输入`1000`和`900`的距离会产生以下结果：

```cpp
Yesterday's distance: 1000
    Today's distance: 900
Journey Details   Distance      Time    Avg Speed
-------------------------------------------------
Yesterday         1,000         8       132
Today             900           4       242
=================================================
Difference        100           4       -109
```

该程序将在循环中运行，直到您输入空白值。您会注意到不同的输出，因为`JourneyTime`是使用`Random`类的实例返回的随机值设置的。

注意

您可以在[`packt.link/EJTtS`](https://packt.link/EJTtS)找到此练习使用的代码。

在这个练习中，您已经看到了如何使用`Func<Car, double>`委托来创建通用代码，而无需创建额外的接口或类。

现在是时候看一下委托的第二个重要方面，即它们能够将多个目标方法链接在一起。

## 多播委托

到目前为止，您已经调用了具有单个分配方法的委托，通常以函数调用的形式。委托提供了将一系列方法组合在一起并使用`+=`运算符进行单次调用的能力，可以将任意数量的附加目标方法添加到目标列表中。每次调用委托时，每个目标方法都会被调用。但是，如果您决定要删除目标方法怎么办？这就是`-=`运算符的用法。

在以下代码片段中，您有一个名为`logger`的`Action<string>`委托。它以单个目标方法`LogToConsole`开始。如果您调用此委托并传入一个字符串，那么`LogToConsole`方法将被调用一次：

```cpp
Action<string> logger = LogToConsole;
logger("1\. Calculating bill");  
```

如果您观察调用堆栈，您将观察到这些调用：

```cpp
logger("1\. Calculating bill")
--> LogToConsole("1\. Calculating bill")
```

要添加新的目标方法，您可以使用`+=`运算符。以下语句将`LogToFile`添加到`logger`委托的调用列表中：

```cpp
logger += LogToFile;
```

现在，每次调用`logger`时，都会调用`LogToConsole`和`LogToFile`。现在再次调用`logger`：

```cpp
logger("2\. Saving order"); 
```

调用堆栈如下所示：

```cpp
logger("2\. Saving order")
--> LogToConsole("2\. Saving order")
--> LogToFile("2\. Saving order")
```

再次假设您使用`+=`添加第三个目标方法，称为`LogToDataBase`，如下所示：

```cpp
logger += LogToDataBase
```

现在再次调用它：

```cpp
logger("3\. Closing order"); 
```

调用堆栈如下所示：

```cpp
logger("3\. Closing order")
--> LogToConsole("3\. Closing order")
--> LogToFile("3\. Closing order")
--> LogToDataBase("3\. Closing order")
```

但是，请考虑您可能不再想在目标方法列表中包括`LogToFile`。在这种情况下，只需使用`-=`运算符将其删除，如下所示：

```cpp
logger -= LogToFile
```

您可以按以下方式再次调用委托：

```cpp
logger("4\. Closing customer"); 
```

现在，调用堆栈如下所示：

```cpp
logger("4\. Closing customer")
--> LogToConsole("4\. Closing customer")
--> LogToDataBase("4\. Closing customer")
```

如图所示，此代码仅导致`LogToConsole`和`LogToDataBase`。

通过这种方式使用委托，您可以根据运行时的某些条件决定调用哪些目标方法。这使您可以将配置的委托传递到其他方法中，以在需要时调用。

您已经看到可以使用`Console.WriteLine`将消息写入控制台窗口。要创建一个记录到文件的方法（如前面示例中的`LogToFile`），您需要使用`System.IO`命名空间中的`File`类。 `File`有许多静态方法可用于读取和写入文件。在这里不会详细介绍`File`，但值得一提的是`File.AppendAllText`方法，它可用于创建或替换包含字符串值的文本文件，`File.Exists`用于检查文件是否存在，以及`File.Delete`用于删除文件。

现在是练习所学知识的时间。

## 练习 3.03：调用多播委托

在这个练习中，您将使用多播委托创建一个现金机，在用户输入他们的 PIN 并要求查看余额时记录详细信息。为此，您将创建一个`CashMachine`类，该类调用配置的**日志**委托，您可以将其用作控制器类，以决定消息是发送到文件还是控制台。

您将使用`Action<string>`委托，因为您不需要返回任何值。使用`+=`，您可以控制在调用`CashMachine`时调用哪些目标方法。

执行以下步骤来这样做：

1.  切换到`Chapter03`文件夹，并使用 CLI`dotnet`命令创建一个名为`Exercise03`的新控制台应用程序：

```cpp
source\Chapter03>dotnet new console -o Exercise03
```

1.  打开`Chapter03\Exercise03.csproj`并用以下设置替换整个文件：

```cpp
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net6.0</TargetFramework>
  </PropertyGroup>
</Project>
```

1.  打开`Exercise03\Program.cs`并清除内容。

1.  添加一个名为`CashMachine`的新类。

1.  使用`Chapter03.Exercise03`命名空间：

```cpp
using System;
using System.IO;
namespace Chapter03.Exercise03
{
    public class CashMachine
    {
        private readonly Action<string> _logger;
        public CashMachine(Action<string> logger)
        {
            _logger = logger;
        } 
```

`CashMachine`构造函数接收`Action<string>`委托，您可以将其分配给名为`_logger`的`readonly`类变量。

1.  添加一个`Log`辅助函数，检查`_logger`委托在调用之前是否为 null：

```cpp
        private void Log(string message)
            => _logger?.Invoke(message);
```

1.  当调用`VerifyPin`和`ShowBalance`方法时，应记录一条带有一些详细信息的消息。按照以下方式创建这些方法：

```cpp
        public void VerifyPin(string pin) 
            => Log($"VerifyPin called: PIN={pin}");
        public void ShowBalance() 
            => Log("ShowBalance called: Balance=999");
    }
```

1.  现在，添加一个配置`logger`委托的控制台应用程序，您可以将其传递给`CashMachine`对象。请注意，这是一种常见的用法：一个负责决定其他类如何记录消息的类。使用常量`OutputFile`作为文件记录使用的文件名，如下所示：

```cpp
    public static class Program
    {
        private const string OutputFile = "activity.txt";
        public static void Main()
        {
```

1.  每次程序运行时，它应该从`File.Delete`开始删除输出文件：

```cpp
            if (File.Exists(OutputFile))
            {
                File.Delete(OutputFile);
            }
```

1.  创建一个委托实例`logger`，该实例以单个目标方法`LogToConsole`开始：

```cpp
            Action<string> logger = LogToConsole;
```

1.  使用`+=`运算符，添加`LogToFile`作为第二个目标方法，每当`CashMachine`调用委托时也会被调用：

```cpp
            logger += LogToFile;
```

1.  您将很快实现这两个目标日志方法；现在，创建一个`cashMachine`实例，并准备调用其方法，如下所示：

```cpp
            var cashMachine = new CashMachine(logger);
```

1.  提示输入`pin`并将其传递给`VerifyPin`方法：

```cpp
            Console.Write("Enter your PIN:");
            var pin = Console.ReadLine();
            if (string.IsNullOrEmpty(pin))
            {
                Console.WriteLine("No PIN entered");
                return;
            }
            cashMachine.VerifyPin(pin);
            Console.WriteLine();
```

如果输入空值，则会进行检查并显示警告。然后使用`return`语句关闭程序。

1.  在调用`ShowBalance`方法之前，等待按下`Enter`键：

```cpp
            Console.Write("Press Enter to show balance");
            Console.ReadLine();
            cashMachine.ShowBalance();
            Console.Write("Press Enter to quit");
            Console.ReadLine();
```

1.  现在是记录方法的时间。它们必须与您的`Action<string>`委托兼容。一个将消息写入控制台，另一个将其附加到文本文件。按照以下方式添加这两个静态方法：

```cpp
            static void LogToConsole(string message)
                => Console.WriteLine(message);
            static void LogToFile(string message)
                => File.AppendAllText(OutputFile, message);
        }
     }
}
```

1.  运行控制台应用程序，您会看到`VerifyPin`和`ShowBalance`调用被写入控制台：

```cpp
Enter your PIN:12345
VerifyPin called: PIN=12345
Press Enter to show balance
ShowBalance called: Balance=999
```

1.  对于每个`logger`委托调用，`LogToFile`方法也将被调用，因此在打开`activity.txt`时，您应该看到以下行：

```cpp
VerifyPin called: PIN=12345ShowBalance called: Balance=999
```

注意

您可以在[`packt.link/h9vic`](https://packt.link/h9vic)找到用于此练习的代码。

重要的是要记住委托是不可变的，因此每次使用`+=`或`-=`运算符时，都会创建一个**新的**委托实例。这意味着如果在将委托传递给目标类后更改委托，则不会看到从该目标类内部调用的方法发生任何更改。

您可以在以下示例中看到这一点：

```cpp
MulticastDelegatesAddRemoveExample.cs
using System;
namespace Chapter03Examples
{
    class MulticastDelegatesAddRemoveExample
    {
        public static void Main()
        {
            Action<string> logger = LogToConsole;
            Console.WriteLine($"Logger1 #={logger.GetHashCode()}");
            logger += LogToConsole;
            Console.WriteLine($"Logger2 #={logger.GetHashCode()}");
            logger += LogToConsole;
            Console.WriteLine($"Logger3 #={logger.GetHashCode()}");
You can find the complete code here: https://packt.link/vqZMF.
```

C# 中的所有对象都有一个返回唯一 ID 的 `GetHashCode()` 函数。运行代码会产生这个输出：

```cpp
Logger1 #=46104728
Logger2 #=1567560752
Logger3 #=236001992
```

您可以看到 `+=` 调用。这表明对象引用每次都在改变。

现在看另一个示例，使用 `Action<string>` 委托。在这里，您将使用 `+=` 运算符添加目标方法，然后使用 `-=` 删除目标方法：

```cpp
MulticastDelegatesExample.cs
using System;
namespace Chapter03Examples
{
    class MulticastDelegatesExample
    {
        public static void Main()
        {
            Action<string> logger = LogToConsole;
            logger += LogToConsole;
            logger("Console x 2");

            logger -= LogToConsole;
            logger("Console x 1");
            logger -= LogToConsole;
You can find the complete code here: https://packt.link/Xe0Ct.
```

您首先使用一个目标方法 `LogToConsole`，然后第二次添加相同的目标方法。使用 `logger("Console x 2")` 调用 logger 委托会导致 `LogToConsole` 被调用两次。

然后使用 `-=` 两次删除 `LogToConsole`，这样就有了两个目标，现在一个也没有了。运行代码会产生以下输出：

```cpp
Console x 2
Console x 2
Console x 1
```

然而，与其正确运行 `logger("logger is now null")`，你最终会遇到一个未处理的异常被抛出，如下所示：

```cpp
System.NullReferenceException
  HResult=0x80004003
  Message=Object reference not set to an instance of an object.
  Source=Examples
  StackTrace:
   at Chapter03Examples.MulticastDelegatesExample.Main() in Chapter03\MulticastDelegatesExample.cs:line 16
```

通过移除最后一个目标方法，`-=` 运算符返回了一个空引用，然后将其分配给了 logger。正如您所看到的，重要的是在尝试调用之前始终检查委托是否为空。

### 使用 Func 委托进行多播

到目前为止，您已经在 `Action` 委托中使用了 `Action<string>` 委托。

您已经看到当从调用的委托中需要返回值时，使用 `Func` 委托。C# 编译器在多播委托中使用 `Func` 委托也是完全合法的。

考虑以下示例，其中您有一个 `Func<string, string>` 委托。这个委托支持传入一个字符串并返回一个格式化的字符串。当您需要通过删除 `@` 符号和点符号来格式化电子邮件地址时，可以使用这个委托：

```cpp
using System;
namespace Chapter03Examples
{
    class FuncExample
    {
        public static void Main()
        {
```

您首先将 `RemoveDots` 字符串函数分配给 `emailFormatter`，然后使用 `Address` 常量调用它：

```cpp
            Func<string, string> emailFormatter = RemoveDots;
            const string Address = "admin@google.com";
            var first = emailFormatter(Address);
            Console.WriteLine($"First={first}");
```

然后添加第二个目标 `RemoveAtSign`，并第二次调用 `emailFormatter`：

```cpp
            emailFormatter += RemoveAtSign;
            var second = emailFormatter(Address);
            Console.WriteLine($"Second={second}");
            Console.ReadLine();
            static string RemoveAtSign(string address)
                => address.Replace("@", "");
            static string RemoveDots(string address)
                => address.Replace(".", "");
        }
    }
} 
```

运行代码会产生这个输出：

```cpp
First=admin@googlecom
Second=admingoogle.com
```

第一次调用返回 `admin@googlecom` 字符串。添加到目标列表的 `RemoveAtSign` 返回一个只删除 `@` 符号的值。

注意

您可以在 [`packt.link/fshse`](https://packt.link/fshse) 找到此示例使用的代码。

`Func1` 和 `Func2` 都被调用，但只有 `Func2` 的值被返回给 `ResultA` 和 `ResultB` 变量，尽管传入了正确的参数。当以这种方式使用多播的 `Func<>` 委托时，所有目标 `Func` 实例都会被调用，但返回值将是链中最后一个 `Func<>` 的返回值。`Func<>` 更适合于单个方法的场景，尽管编译器仍然允许您将其用作多播委托，而不会出现任何编译错误或警告。

### 当事情出错时会发生什么？

当调用委托时，调用列表中的所有方法都会被调用。对于单个名称委托，这将是一个目标方法。如果多播委托中的一个目标抛出异常会发生什么呢？

考虑以下代码。当调用 `logger` 委托时，通过传入 `try log this`，您可能期望按照它们被添加的顺序调用方法：`LogToConsole`，`LogToError`，最后是 `LogToDebug`：

```cpp
MulticastWithErrorsExample.cs
using System;
using System.Diagnostics;
namespace Chapter03Examples
{
    class MulticastWithErrorsExample
    {
            public static void Main()
            {
                Action<string> logger = LogToConsole;
                logger += LogToError;
                logger += LogToDebug;
                try
                {
                    logger("try log this");
You can find the complete code here: https://packt.link/Ti3Nh.
```

如果任何目标方法抛出异常，比如您在 `LogToError` 中看到的异常，那么剩下的目标就**不会**被调用。

运行代码会产生以下输出：

```cpp
Console: try log this
Caught oops!
All done
```

您将看到这个输出，因为 `LogToDebug` 方法根本没有被调用。考虑一个 UI，其中有多个目标监听鼠标按钮点击。第一个方法在按下按钮时触发并禁用按钮以防止双击，第二个方法更改按钮的图像以指示成功，第三个方法启用按钮。

如果第二种方法失败，那么第三种方法将不会被调用，按钮可能会保持禁用状态，并且分配了一个不正确的图像，从而使用户感到困惑。

为了确保无论如何都运行所有目标方法，你可以枚举调用列表并手动调用每个方法。查看一下.NET 的`MulticastDelegate`类型。你会发现有一个函数`GetInvocationList`，它返回一个委托对象的数组。这个数组包含已添加的目标方法：

```cpp
public abstract class MulticastDelegate : Delegate {
  public sealed override Delegate[] GetInvocationList();
}
```

然后，你可以循环遍历这些目标方法，并在`try`/`catch`块中执行每个方法。现在通过这个练习来实践你所学到的知识。

## 练习 3.04：确保在多播委托中调用所有目标方法

在本章中，你一直在使用`Action<string>`委托来执行各种日志记录操作。在这个练习中，你有一个日志委托的目标方法列表，你希望确保“所有”目标方法都被调用，即使之前的方法失败了。你可能会遇到偶尔由于网络问题而导致向数据库或文件系统记录日志失败的情况。在这种情况下，你希望其他日志操作至少有机会执行它们的日志记录活动。

执行以下步骤来实现：

1.  切换到`Chapter03`文件夹并使用 CLI `dotnet`命令创建一个名为`Exercise04`的新控制台应用程序：

```cpp
source\Chapter03>dotnet new console -o Exercise04
```

1.  打开`Chapter03\Exercise04.csproj`并用以下设置替换整个文件：

```cpp
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net6.0</TargetFramework>
  </PropertyGroup>
</Project>
```

1.  打开`Exercise04\Program.cs`并清空内容。

1.  现在为你的控制台应用程序添加一个静态`Program`类，包括`System`和额外的`System.IO`，因为你想要创建一个文件：

```cpp
using System;
using System.IO;
namespace Chapter03.Exercise04
{
    public static class Program
    {
```

1.  使用`const`来命名日志文件。当程序执行时，将创建此文件：

```cpp
        private const string OutputFile = "Exercise04.txt";
```

1.  现在你必须定义应用程序的`Main`入口点。在这里，如果输出文件已经存在，你可以删除它。最好从一个空文件开始，否则每次运行应用程序时，日志文件都会不断增长：

```cpp
        public static void Main()
        {
            if (File.Exists(OutputFile))
            {
                File.Delete(OutputFile);
            }
```

1.  你将从`logger`只有一个目标方法`LogToConsole`开始，稍后将添加：

```cpp
            Action<string> logger = LogToConsole;
```

1.  你使用`InvokeAll`方法来调用委托，传入`"First call"`作为参数。这不会失败，因为`logger`有一个有效的方法，你很快也会添加`InvokeAll`：

```cpp
            InvokeAll(logger, "First call"); 
```

1.  这个练习的目的是要有一个多播委托，所以添加一些额外的目标方法：

```cpp
            logger += LogToConsole;
            logger += LogToDatabase;
            logger += LogToFile; 
```

1.  尝试使用`InvokeAll`进行第二次调用，如下所示：

```cpp
            InvokeAll(logger, "Second call"); 
            Console.ReadLine();
```

1.  现在针对添加到委托中的目标方法，添加以下代码：

```cpp
            static void LogToConsole(string message)
                => Console.WriteLine($"LogToConsole: {message}");
            static void LogToDatabase(string message)
                => throw new ApplicationException("bad thing happened!");
            static void LogToFile(string message)
                => File.AppendAllText(OutputFile, message);

```

1.  现在你可以实现`InvokeAll`方法：

```cpp
            static void InvokeAll(Action<string> logger, string arg)
            {
                if (logger == null)
                     return;
```

它传递了一个与`logger`委托类型匹配的`Action<string>`委托，以及在调用每个目标方法时使用的`arg`字符串。不过，在此之前，重要的是要检查`logger`是否已经为 null，对于 null 委托，你无法做任何操作。

1.  使用委托的`GetInvocationList()`方法来获取所有目标方法的列表：

```cpp
                var delegateList = logger.GetInvocationList();
                Console.WriteLine($"Found {delegateList.Length} items in {logger}"); 
```

1.  现在，按照以下方式循环遍历列表中的每个项目：

```cpp
                foreach (var del in delegateList)
                {
```

1.  在每个循环元素中用`try`/`catch`包装，将`del`转换为`Action<string>`：

```cpp
                   try
                   {
                     var action = del as Action<string>; 
```

`GetInvocationList`返回每个项目作为基本委托类型，而不考虑它们的实际类型。

1.  如果它是正确的类型且**不是**null，那么尝试调用是安全的：

```cpp
                      if (del is Action<string> action)
                      {
                          Console.WriteLine($"Invoking '{action.Method.Name}' with '{arg}'");
                          action(arg);
                      }
                      else
                      {
                          Console.WriteLine("Skipped null");
                      } 
```

你已经添加了一些额外的细节，以显示委托的`Method.Name`属性即将被调用的内容。

1.  最后使用一个`catch`块，如果捕获到错误，则记录错误消息：

```cpp
                  }
                  catch (Exception e)
                  {
                      Console.WriteLine($"Error: {e.Message}");
                  }
                }
            }
        }
    }
}
```

1.  运行代码，创建一个名为`Exercise04.txt`的文件，其中包含以下结果：

```cpp
Found 1 items in System.Action`1[System.String]
Invoking '<Main>g__LogToConsole|1_0' with 'First call'
LogToConsole: First call
Found 4 items in System.Action`1[System.String]
Invoking '<Main>g__LogToConsole|1_0' with 'Second call'
LogToConsole: Second call
Invoking '<Main>g__LogToConsole|1_0' with 'Second call'
LogToConsole: Second call
Invoking '<Main>g__LogToDatabase|1_1' with 'Second call'
Error: bad thing happened!
Invoking '<Main>g__LogToFile|1_2' with 'Second call'
```

你会发现它捕获了`LogToDatabase`抛出的错误，但仍然允许调用`LogToFile`。

注意

你可以在[`packt.link/Dp5H4`](https://packt.link/Dp5H4)找到用于这个练习的代码。

现在很重要的是要扩展使用事件的多播概念。

## 事件

在之前的章节中，您已经创建了委托，并直接在同一方法中调用它们，或者将它们传递给另一个方法，在需要时进行调用。通过这种方式使用委托，您可以简单地让代码在感兴趣的事情发生时得到通知。到目前为止，这还不是一个主要问题，但您可能已经注意到，似乎没有办法阻止具有委托访问权限的对象直接调用它。

考虑以下情景：您创建了一个应用程序，允许其他程序通过将它们的目标方法添加到您提供的委托中来注册通知，当新的电子邮件到达时。如果一个程序，无论是出于错误还是出于恶意原因，决定自己调用您的委托会怎么样？这很容易就会压倒调用列表中的所有目标方法。这样的监听程序绝不能被允许以这种方式调用委托——毕竟，它们应该是被动的监听者。

您可以添加额外的方法，允许监听者将他们的目标方法添加或从调用列表中移除，并保护委托免受直接访问，但是如果在应用程序中有数百个这样的委托会怎么样？这需要大量的代码来编写。

`event`关键字指示 C#编译器添加额外的代码，以确保委托**只能**由声明它的类或结构调用。外部代码可以添加或移除目标方法，但不能调用委托。试图这样做会导致编译器错误。

这种模式通常被称为发布-订阅模式。引发事件的对象称为事件发送者或**发布者**；接收事件的对象称为事件处理程序或**订阅者**。

## 定义事件

`event`关键字用于定义事件及其关联的委托。它的定义看起来类似于委托的定义方式，但与委托不同的是，您不能使用全局命名空间来定义事件。

```cpp
public event EventHandler MouseDoubleClicked
```

事件有四个元素：

+   范围：访问修饰符，如`public`、`private`或`protected`，用于定义范围。

+   `event`关键字。

+   委托类型：关联的委托，在这个例子中是`EventHandler`。

+   事件名称：这可以是您喜欢的任何名称，例如`MouseDoubleClicked`。但是，名称必须在命名空间内是唯一的。

事件通常与内置的.NET 委托`EventHandler`或其泛型`EventHandler<>`版本相关联。很少会为事件创建自定义委托，但您可能会在旧的遗留代码中找到这种情况，这些代码是在`Action`和泛型`Action<T>`委托之前创建的。

`EventHandler`委托在早期的.NET 版本中是可用的。它具有以下签名，接受一个发送者`object`和一个`EventArgs`参数：

```cpp
public delegate void EventHandler(object sender, EventArgs e); 
```

最近的基于泛型的`EventHandler<T>`委托看起来类似；它也接受一个发送者`object`和由类型`T`定义的参数。

```cpp
public delegate void EventHandler<T>(object sender, T e); 
```

`sender`参数被定义为`object`，允许任何类型的对象被发送给订阅者，以便它们识别事件的发送者。这在您需要一个集中的方法来处理各种类型的对象而不是特定实例的情况下非常有用。

例如，在 UI 应用程序中，您可能有一个订阅者监听 OK 按钮的点击，另一个订阅者监听**取消**按钮的点击——每个按钮可以由两个不同的方法处理。在使用多个复选框来切换选项的情况下，您可以使用一个单一的目标方法，只需告诉它复选框是发送者，并相应地切换设置。这允许您重用相同的复选框处理程序，而不是为屏幕上的每个复选框创建一个方法。

在调用`EventHandler`委托时，不是强制包含发送者的详细信息。通常，您可能不希望向外界透露代码的内部工作方式；在这种情况下，将一个空引用传递给委托是常见做法。

这两个委托中的第二个参数可以用于提供有关事件的额外上下文信息（例如，是按下了左键还是右键？）。传统上，这些额外信息是使用从`EventArgs`派生的类进行封装的，但在较新的.NET 版本中，这种约定已经放宽。

有两个标准的.NET 委托可以用于事件定义？

+   `EventHandler`：当没有额外信息描述事件时可以使用此委托。例如，复选框点击事件可能不需要任何额外信息，只是被点击了。在这种情况下，将 null 或`EventArgs.Empty`作为第二个参数传递是完全有效的。这个委托通常可以在使用从`EventArgs`派生的类来进一步描述事件的旧应用程序中找到。是鼠标的双击触发了这个事件吗？在这种情况下，可能已经向`EventArgs`派生类添加了一个`Clicks`属性来提供这样的额外细节。

+   `EventHandler<T>`：自从 C#中引入泛型以来，这已经成为更频繁使用的事件委托，简单地因为使用泛型需要创建更少的类。

有趣的是，无论您给事件赋予什么作用域（例如`public`），C#编译器都会在内部创建一个同名的私有成员。这是事件的关键概念：只有定义事件的类可以**调用它**。消费者可以自由添加或删除他们的兴趣，但他们**不能**自己调用它。

当定义事件时，其中定义它的发布者类可以在需要时简单地调用它，就像调用委托一样。在早期的示例中，总是强调在调用之前始终检查委托是否为 null。与事件一样，应采用相同的方法，因为您无法控制订阅者何时以及如何添加或删除他们的目标方法。

当发布者类最初创建时，所有事件的初始值都为 null。当任何订阅者添加目标方法时，这将更改为非 null。相反，一旦订阅者删除目标方法，如果调用列表中没有方法了，事件将恢复为 null，所有这些都由运行时处理。这是您在早期使用委托时看到的标准行为。

您可以通过在事件定义的末尾添加一个空委托来防止事件永远变为 null：

```cpp
public event EventHandler<MouseEventArgs> MouseDoubleClicked = delegate {};
```

您不是使用默认的 null 值，而是添加自己的默认委托实例——一个什么也不做的实例。因此在`{}`符号之间留空。

在发布者类中使用事件时通常遵循一种常见模式，特别是在可能进一步被子类化的类中。现在，您将通过一个简单的示例来看到这一点：

1.  定义一个名为`MouseClickedEventArgs`的类，其中包含有关事件的其他信息，例如检测到的鼠标点击次数：

```cpp
using System;
namespace Chapter03Examples
{
    public class MouseClickedEventArgs 
    {
        public MouseClickedEventArgs(int clicks)
        {
            Clicks = clicks;
        }
        public int Clicks { get; }
    }
```

观察`MouseClickPublisher`类，它使用泛型`EventHandler<>`委托定义了一个`MouseClicked`事件。

1.  现在添加`delegate { };`块以防止`MouseClicked`最初为 null：

```cpp
    public class MouseClickPublisher
    {
     public event EventHandler<MouseClickedEventArgs> MouseClicked = delegate { };
```

1.  添加一个`OnMouseClicked`虚方法，让任何进一步子类化的`MouseClickPublisher`类有机会抑制或更改事件通知，如下所示：

```cpp
        protected virtual void OnMouseClicked( MouseClickedEventArgs e)
        {
            var evt = MouseClicked;
            evt?.Invoke(this, e);
        }
```

1.  现在您需要一个跟踪鼠标点击的方法。在这个例子中，您实际上不会展示如何检测鼠标点击，但您将调用`OnMouseClicked`，传入`2`以指示双击。

1.  请注意，您没有直接调用 `MouseClicked` 事件；您总是通过中间方法 `OnMouseClicked` 进行。这为其他 `MouseClickPublisher` 的实现提供了一种覆盖事件通知的方式，如果需要的话：

```cpp
        private void TrackMouseClicks()
        {
            OnMouseClicked(new MouseClickedEventArgs(2));
        }
    } 
```

1.  现在添加一个基于 `MouseClickPublisher` 的新类型的发布者：

```cpp
    public class MouseSingleClickPublisher : MouseClickPublisher
    {
        protected override void OnMouseClicked(MouseClickedEventArgs e)
        {
            if (e.Clicks == 1)
            {
                OnMouseClicked(e);
            }
        }
    }
} 
```

这个 `MouseSingleClickPublisher` 覆盖了 `OnMouseClicked` 方法，并且只有在检测到单击时才调用基本的 `OnMouseClicked`。通过实现这种类型的模式，您可以允许不同类型的发布者以定制的方式控制事件是否传递给订阅者。

注意

您可以在 [`packt.link/J1EiB`](https://packt.link/J1EiB) 找到此示例的代码。

您现在可以通过以下练习来练习所学到的知识。

## 练习 3.05：发布和订阅事件

在这个练习中，您将创建一个闹钟作为发布者的示例。闹钟将模拟 `Ticked` 事件。您还将添加一个 `WakeUp` 事件，当当前时间匹配闹钟时间时发布。在 .NET 中，`DateTime` 用于表示时间点，因此您将用它来表示当前时间和闹钟时间属性。您将使用 `DateTime.Subtract` 来获取当前时间和闹钟时间之间的差异，并在到期时发布 `WakeUp` 事件。

执行以下步骤：

1.  切换到 `Chapter03` 文件夹，并使用 CLI `dotnet` 命令创建一个名为 `Exercise05` 的新控制台应用程序：

```cpp
dotnet new console -o Exercise05
```

1.  打开 `Chapter03\Exercise05.csproj` 并用以下设置替换整个文件：

```cpp
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net6.0</TargetFramework>
  </PropertyGroup>
</Project>
```

1.  打开 `Exercise05\Program.cs` 并清空内容。

1.  添加一个名为 `AlarmClock` 的新类。在这里，您需要使用 `DateTime` 类，因此包括 `System` 命名空间：

```cpp
using System;
namespace Chapter03.Exercise05
{
    public class AlarmClock
    {
```

您将为订阅者提供两个事件来监听——`WakeUp`，基于非泛型的 `EventHandler` 委托（因为您不会在此事件中传递任何额外信息），以及使用泛型 `EventHandler` 委托和 `DateTime` 参数类型的 `Ticked`。

1.  您将使用此方法将当前时间传递给控制台显示。请注意，两者都具有初始的 `delegate {};` 安全机制：

```cpp
        public event EventHandler WakeUp = delegate {};
        public event EventHandler<DateTime> Ticked = delegate {};
```

1.  包括一个 `OnWakeUp` 覆盖作为示例，但不要对 `Ticked` 做同样的操作；这是为了展示不同的调用方法：

```cpp
        protected void OnWakeUp()
        {
            WakeUp.Invoke(this, EventArgs.Empty);
        }
```

1.  现在添加两个 `DateTime` 属性，闹钟和时钟时间，如下所示：

```cpp
        public DateTime AlarmTime { get; set; }
        public DateTime ClockTime { get; set; }
```

1.  `Start` 方法用于启动时钟。您使用一个简单的循环模拟每分钟一次的时钟滴答声，持续 `24 小时`，如下所示：

```cpp
        public void Start()
        {
            // Run for 24 hours
            const int MinutesInADay = 60 * 24;
```

1.  对于每个模拟的分钟，使用 `DateTime.AddMinute` 增加时钟并发布 `Ticked` 事件，传入 `this`（`AlarmClock` 发送方实例）和时钟时间：

```cpp
            for (var i = 0; i < MinutesInADay; i++)
            {
                ClockTime = ClockTime.AddMinutes(1);
                Ticked.Invoke(this, ClockTime);
```

`ClockTime.Subtract` 用于计算点击和闹钟时间之间的差异。

1.  将 `timeRemaining` 值传递给本地函数 `IsTimeToWakeUp`，调用 `OnWakeUp` 方法，并在该时间到达时退出循环：

```cpp
              var timeRemaining = ClockTime                 .Subtract(AlarmTime)                .TotalMinutes;
               if (IsTimeToWakeUp(timeRemaining))
                {
                    OnWakeUp();
                    break;
                }
            }
```

1.  使用关系模式 `IsTimeToWakeUp` 来查看是否剩余不到一分钟。添加以下代码：

```cpp
            static bool IsTimeToWakeUp(double timeRemaining) 
                => timeRemaining is (>= -1.0 and <= 1.0);
        }
    }   
```

1.  现在添加一个控制台应用程序，通过从静态 void `Main` 入口点开始订阅闹钟及其两个事件：

```cpp
         public static class Program
    {
        public static void Main()
        {
```

1.  创建 `AlarmClock` 实例，并使用 `+=` 运算符订阅 `Ticked` 事件和 `WakeUp` 事件。您将很快定义 `ClockTicked` 和 `ClockWakeUp`。现在，只需添加以下代码：

```cpp
            var clock = new AlarmClock();
            clock.Ticked += ClockTicked;
            clock.WakeUp += ClockWakeUp; 
```

1.  设置时钟的当前时间，使用 `DateTime.AddMinutes` 来将 `120` 分钟添加到闹钟时间，然后启动时钟，如下所示：

```cpp
            clock.ClockTime = DateTime.Now;
            clock.AlarmTime = DateTime.Now.AddMinutes(120);
            Console.WriteLine($"ClockTime={clock.ClockTime:t}");
            Console.WriteLine($"AlarmTime={clock.AlarmTime:t}");
            clock.Start(); 
```

1.  通过提示按下 `Enter` 键来完成 `Main`：

```cpp
            Console.WriteLine("Press ENTER");
            Console.ReadLine();

```

1.  现在您可以添加事件订阅者本地方法：

```cpp
            static void ClockWakeUp(object sender, EventArgs e)
            {
               Console.WriteLine();
               Console.WriteLine("Wake up");
            }
```

`ClockWakeUp` 传递了发送方和 `EventArgs` 参数。您两者都没有使用，但它们是 `EventHandler` 委托所需的。当调用此订阅者的方法时，您将在控制台中写入 `"Wake up"`。

1.  `ClockTicked`按照`EventHandler<DateTime>`委托所需的方式传递`DateTime`参数。在这里，您传递当前时间，因此使用`:t`将其以短格式显示在控制台上：

```cpp
             static void ClockTicked(object sender, DateTime e)
                => Console.Write($"{e:t}...");
        }
    }
} 
```

1.  运行应用程序会产生以下输出：

```cpp
ClockTime=14:59
AlarmTime=16:59
15:00...15:01...15:02...15:03...15:04...15:05...15:06...15:07...15:08...15:09...15:10...15:11...15:12...15:13...15:14...15:15...15:16...15:17...15:18...15:19...15:20...15:21...15:22...15:23...15:24...15:25...15:26...15:27...15:28...15:29...15:30...15:31...15:32...15:33...15:34...15:35...15:36...15:37...15:38...15:39...15:40...15:41...15:42...15:43...15:44...15:45...15:46...15:47...15:48...15:49...15:50...15:51...15:52...15:53...15:54...15:55...15:56...15:57...15:58...15:59...16:00...16:01...16:02...16:03...16:04...16:05...16:06...16:07...16:08...16:09...16:10...16:11...16:12...16:13...16:14...16:15...16:16...16:17...16:18...16:19...16:20...16:21...16:22...16:23...16:24...16:25...16:26...16:27...16:28...16:29...16:30...16:31...16:32...16:33...16:34...16:35...16:36...16:37...16:38...16:39...16:40...16:41...16:42...16:43...16:44...16:45...16:46...16:47...16:48...16:49...16:50...16:51...16:52...16:53...16:54...16:55...16:56...16:57...16:58...16:59...
Wake up
Press ENTER
```

在这个例子中，您可以看到闹钟模拟每分钟发出一次滴答声并发布`Ticked`事件。

注意

您可以在[`packt.link/GPkYQ`](https://packt.link/GPkYQ)找到用于此练习的代码。

现在是时候理解事件和委托之间的区别了。

# 事件还是委托？

乍一看，事件和委托看起来非常相似：

+   事件是委托的扩展形式。

+   两者都提供**后期绑定**语义，因此不是在编译时精确知道要调用的方法，而是在运行时知道时，可以推迟一系列目标方法。

+   两者都是`Invoke()`，或者更简单地说，是`()`后缀的快捷方式，在这样做之前最好进行空值检查。

关键考虑因素如下：

+   可选性：事件提供一种可选的方法；调用者可以决定是否选择事件。如果您的组件可以在不需要任何订阅者方法的情况下完成其任务，那么最好使用基于事件的方法。

+   返回类型：您需要处理返回类型吗？与事件相关的委托始终是 void。

+   生命周期：事件订阅者通常比发布者的生命周期更短，即使没有活跃的订阅者，发布者仍然会继续检测新消息。

## 静态事件可能会导致内存泄漏

在结束查看事件之前，使用事件时要**小心**。

每当将订阅者的目标方法添加到发布者的事件中时，发布者类将存储对目标方法的引用。当您完成使用订阅者实例并且它仍然附加到`static`发布者时，可能会导致订阅者使用的内存不会被清除。

这些通常被称为孤立、幻影或幽灵事件。为了防止这种情况发生，始终尝试将每个`+=`调用与相应的`-=`运算符配对。

注意

响应式扩展（Rx）([`github.com/dotnet/reactive`](https://github.com/dotnet/reactive))是一个很棒的库，可以利用和驯服基于事件和异步编程，使用 LINQ 风格的操作符。Rx 提供了一种时移的方法，例如，用几行代码将一个非常喧闹的事件缓冲成可管理的流。而且，Rx 流非常容易进行单元测试，可以有效地控制时间。

现在阅读有关 lambda 表达式的有趣主题。

# Lambda 表达式

在前面的部分中，您主要使用了类级别的方法作为委托和事件的目标，例如`ClockTicked`和`ClockWakeUp`方法，这些方法也在*Exercise 3.05*中使用过：

```cpp
var clock = new AlarmClock();
clock.Ticked += ClockTicked;
clock.WakeUp += ClockWakeUp;
static void ClockTicked(object sender, DateTime e)
  => Console.Write($"{e:t}...");

static void ClockWakeUp(object sender, EventArgs e)
{
    Console.WriteLine();
    Console.WriteLine("Wake up");
}
```

`ClockWakeUp`和`ClockTicked`方法易于理解和逐步执行。然而，通过将它们转换为 lambda 表达式语法，您可以获得更简洁的语法，并且更接近它们在代码中的位置。

现在将`Ticked`和`WakeUp`事件转换为使用两个不同的 lambda 表达式：

```cpp
clock.Ticked += (sender, e) =>
{
    Console.Write($"{e:t}..."); 
};  
clock.WakeUp += (sender, e) =>
{
    Console.WriteLine();
    Console.WriteLine("Wake up");
}; 
```

您已经使用了相同的`+=`运算符，但是不是方法名称，而是看到了`(sender, e) =>`和相同的代码块，就像在`ClockTicked`和`ClockWakeUp`中看到的那样。

在定义 lambda 表达式时，您可以在括号`()`内传递任何参数，然后是`=>`（这经常被读作**转到**），然后是您的表达式/语句块：

```cpp
(parameters) => expression-or-block
```

代码块可以尽您需要的那样复杂，并且如果是基于`Func`的委托，可以返回一个值。

编译器通常可以推断出每个参数的类型，因此您甚至不需要指定它们的类型。此外，如果只有一个参数且编译器可以推断出其类型，您可以省略括号。

无论何时需要使用委托（记住`Action`、`Action<T>`和`Func<T>`是内置的委托示例），而不是创建类或本地方法或函数，您都应该考虑使用 lambda 表达式。主要原因是这通常会导致更少的代码，并且该代码放置在使用它的位置附近。

现在考虑 Lambda 的另一个例子。给定一个电影列表，您可以使用`List<string>`类来存储这些基于字符串的名称，如下所示：

```cpp
using System;
using System.Collections.Generic;
namespace Chapter03Examples
{
    class LambdaExample
    {
        public static void Main()
        {
            var names = new List<string>
            {
                "The A-Team",
                "Blade Runner",
                "There's Something About Mary",
                "Batman Begins",
                "The Crow"
            };
```

您可以使用`List.Sort`方法按字母顺序对名称进行排序（最终输出将在本示例结束时显示）：

```cpp
            names.Sort();
            Console.WriteLine("Sorted names:");
            foreach (var name in names)
            {
                Console.WriteLine(name);
            }
            Console.WriteLine();
```

如果您需要更多控制如何进行排序，`List`类还有另一个接受此形式委托的`Sort`方法：`delegate int Comparison<T>(T x, T y)`。这个委托传递了相同类型的两个参数（`x`和`y`），并返回一个`int`值。`int`值可以用来定义列表中项目的排序顺序，而无需担心`Sort`方法的内部工作。

作为另一种选择，您可以对名称进行排序，从电影标题开头排除“the”。这通常被用作列出名称的另一种方式。您可以通过传递一个 lambda 表达式来实现这一点，使用`( )`语法来包装两个字符串`x, y`，当`Sort()`调用您的 lambda 时，这两个字符串将被传递。

如果`x`或`y`以您的噪声词“the”开头，那么您可以使用`string.Substring`函数跳过前四个字符。然后使用`String.Compare`返回一个比较生成的字符串值的数值，如下所示：

```cpp
            const string Noise = "The ";
            names.Sort( (x, y) =>
            {
                if (x.StartsWith(Noise))
                {
                    x = x.Substring(Noise.Length);
                }
                if (y.StartsWith(Noise))
                {
                    y = x.Substring(Noise.Length);
                }
                return string.Compare(x , y);
            });
```

然后将排序后的结果写入控制台：

```cpp
            Console.WriteLine($"Sorted excluding leading '{Noise}':");
            foreach (var name in names)
            {
                Console.WriteLine(name);
            }
            Console.ReadLine();
         }
     }
} 
```

运行示例代码会产生以下输出：

```cpp
Sorted names:
Batman Begins
Blade Runner
The A-Team
The Crow
There's Something About Mary
Sorted excluding leading 'The ':
The A-Team
Batman Begins
Blade Runner
The Crow
There's Something About Mary 
```

您可以看到第二组名称是按照忽略“the”进行排序的。

注意

您可以在[`packt.link/B3NmQ`](http://packt.link/B3NmQ)找到此示例中使用的代码。

要看到这些 lambda 语句付诸实践，请尝试以下练习。

## 练习 3.06：使用语句 Lambda 来反转句子中的单词

在这个练习中，您将创建一个实用程序类，它会拆分句子中的单词，并返回单词顺序相反的句子。

执行以下步骤来实现：

1.  切换到`Chapter03`文件夹并使用 CLI`dotnet`命令创建一个名为`Exercise06`的新控制台应用程序：

```cpp
source\Chapter03>dotnet new console -o Exercise06
```

1.  打开`Chapter03\Exercise06.csproj`并用以下设置替换整个文件：

```cpp
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net6.0</TargetFramework>
  </PropertyGroup>
</Project>
```

1.  打开`Exercise02\Program.cs`并清空内容。

1.  添加一个名为`WordUtilities`的新类，其中包含一个名为`ReverseWords`的字符串函数。您需要包括`System.Linq`命名空间来帮助进行字符串操作：

```cpp
using System;
using System.Linq;
namespace Chapter03.Exercise06
{
    public static class WordUtilities
    {
        public static string ReverseWords(string sentence)
        {
```

1.  定义一个名为`swapWords`的`Func<string, string>`委托，它接受一个字符串输入并返回一个字符串值：

```cpp
          Func<string, string> swapWords = 
```

1.  您将接受一个名为`phrase`的字符串输入参数：

```cpp
            phrase =>
```

1.  现在是 lambda 语句体。使用`string.Split`函数将`phrase`字符串按空格拆分为字符串数组：

```cpp
                  {
                    const char Delimit = ' ';
                    var words = phrase
                        .Split(Delimit)
                        .Reverse();
                    return string.Join(Delimit, words);
                };
```

`String.Reverse`反转数组中字符串的顺序，最后使用`string.Join`将反转的单词字符串数组连接成一个字符串。

1.  您已经定义了所需的`Func`，现在通过传递句子参数来调用它，并将其作为结果返回：

```cpp
            return swapWords(sentence);
         }
    }
```

1.  现在创建一个控制台应用程序，提示输入一个句子，将其传递给`WordUtilities.ReverseWords`，并将结果写入控制台：

```cpp
    public static class Program
    {
        public static void Main()
        {
            do
            {
                Console.Write("Enter a sentence:");
                var input = Console.ReadLine();
                if (string.IsNullOrEmpty(input))
                {
                    break;
                }
                var result = WordUtilities.ReverseWords(input);
                Console.WriteLine($"Reversed: {result}")
```

运行控制台应用程序会产生类似于以下内容的结果输出：

```cpp
Enter a sentence:welcome to c#
Reversed: c# to welcome
Enter a sentence:visual studio by microsoft
Reversed: microsoft by studio visual
```

注意

您可以在[`packt.link/z12sR`](https://packt.link/z12sR)找到此练习中使用的代码。

您将通过一些不太明显的问题来结束对 lambda 的探讨，这些问题在运行和调试时可能出乎您的意料。

## 捕获和闭包

Lambda 表达式可以**捕获**方法内的任何变量或参数。捕获一词用于描述 lambda 表达式捕获或访问父方法中的任何变量或参数的方式。

为了更好地理解这一点，请考虑以下示例。在这里，您将创建一个名为`joiner`的`Func<int, string>`，它使用`Enumerable.Repeat`方法将单词连接在一起。`word`变量（称为`Outer Variables`）被捕获在`joiner`表达式的主体内部：

```cpp
var word = "hello";
Func<int, string> joiner = 
    input =>
    {
        return string.Join(",", Enumerable.Repeat(word, input));
    };  
Console.WriteLine($"Outer Variables: {joiner(2)}"); 
```

运行上面的例子会产生以下输出：

```cpp
Outer Variables: hello,hello
```

您通过传递`2`作为参数来调用`joiner`委托。在那一刻，外部`word`变量的值为`"hello"`，这个值被重复两次。

这证实了从父方法中捕获的变量在调用`Func`时被评估。现在将`word`的值从`hello`更改为`goodbye`，再次调用`joiner`，并将`3`作为参数传递：

```cpp
word = "goodbye";
Console.WriteLine($"Outer Variables Part2: {joiner(3)}");
```

运行此示例会产生以下输出：

```cpp
Outer Variables Part2: goodbye,goodbye,goodbye
```

值得记住的是，无论您在代码中的何处定义了`joiner`，都不重要。在声明`joiner`之前或之后，您可以将`word`的值更改为任意数量的字符串。

进一步地，如果您在 lambda 内部定义了与外部同名的变量，它将被作用域化为`word`，这对同名的外部变量没有影响：

```cpp
Func<int, string> joinerLocal =
    input =>
    {
        var word = "local";
        return string.Join(",", Enumerable.Repeat(word, input));
    };
Console.WriteLine($"JoinerLocal: {joinerLocal(2)}");
Console.WriteLine($"JoinerLocal: word={word}");   
```

上面的示例导致以下输出。请注意外部变量`word`如何保持不变，仍为`goodbye`：

```cpp
JoinerLocal: local,local
JoinerLocal: word=goodbye
```

最后，您将了解闭包的概念，这是 C#语言中微妙的部分，经常导致意想不到的结果。

在下面的示例中，您有一个名为`actions`的变量，其中包含`Action`委托的`List`。您使用基本的`for`循环将五个单独的`Action`实例添加到列表中。每个`Action`的 lambda 表达式只是将`for`循环中的`i`的值写入控制台。最后，代码只是运行`actions`列表中的每个`Action`并调用每个动作：

```cpp
var actions = new List<Action>();
for (var i = 0; i < 5; i++)
{
    actions.Add( () => Console.WriteLine($"MyAction: i={i}")) ;
}
foreach (var action in actions)
{
    action();
}
```

运行示例会产生以下输出：

```cpp
MyAction: i=5
MyAction: i=5
MyAction: i=5
MyAction: i=5
MyAction: i=5
```

`MyAction: i`之所以没有从`0`开始，是因为当从`Action`委托内部访问`i`的值时，只有在调用`Action`时才会计算。在每次调用委托时，外部循环已经重复了五次。

注意

您可以在[`packt.link/vfOPx`](https://packt.link/vfOPx)找到此示例的代码。

这类似于您观察到的捕获概念，其中外部变量，此处为`i`，只有在调用时才会被计算。您在`for`循环中使用`i`来将每个`Action`添加到列表中，但在调用每个动作时，`i`已经具有最终值`5`。

这通常会导致意外的行为，特别是如果您假设在每个动作的循环变量中使用了`i`。为了确保在每个 lambda 表达式中使用递增值`i`，您需要引入一个`for`循环，它会复制迭代器变量。

在下面的代码片段中，您已经添加了`closurei`变量。它看起来非常微妙，但现在您有了一个更局部作用域的变量，您可以从 lambda 表达式内部访问它，而不是迭代器`i`：

```cpp
var actionsSafe = new List<Action>();
for (var i = 0; i < 5; i++)
{
    var closurei = i;
    actionsSafe.Add(() => Console.WriteLine($"MyAction: closurei={closurei}"));
}
foreach (var action in actionsSafe)
{
    action();
}
```

运行示例会产生以下输出。您可以看到在每次调用`Action`时都使用递增值，而不是您之前看到的`5`的值：

```cpp
MyAction: closurei=0
MyAction: closurei=1
MyAction: closurei=2
MyAction: closurei=3
MyAction: closurei=4
```

您已经涵盖了委托和事件在事件驱动应用程序中的关键方面。您通过使用 lambda 提供的简洁编码风格来扩展了这一点，以便在发生感兴趣的事件时得到通知。

现在，您将把这些想法结合到一个活动中，在这个活动中，您将使用一些内置的.NET 类及其自己的事件。您需要调整这些事件以适应您自己的格式，并发布，以便可以由控制台应用程序订阅。

现在是时候通过以下活动来练习您所学到的所有知识了。

## 活动 3.01：创建 Web 文件下载器

您计划调查美国风暴事件中的模式。为此，您需要从在线来源下载风暴事件数据集，以供以后分析。美国国家海洋和大气管理局是这样的数据来源之一，可以从[`www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles`](https://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles)访问。

您的任务是创建一个.NET Core 控制台应用程序，允许输入网址，其内容将被下载到本地磁盘。为了尽可能用户友好，应用程序需要使用事件来信号当输入无效地址时，下载的进度以及下载完成时。

理想情况下，您应该尝试隐藏用于下载文件的内部实现，而是倾向于调整您使用的任何事件，使其适用于您的调用者可以订阅的事件。这种调整形式通常用于通过隐藏内部细节来使代码更易于维护。

为此，C#中的`WebClient`类可以用于下载请求。与.NET 的许多部分一样，这个类返回实现`IDisposable`接口的对象。这是一个标准接口，它表示您正在使用的对象应该被包装在`using`语句中，以确保在您使用完对象后，任何资源或内存都被清除。`using`采用以下格式：

```cpp
using (IDisposable) { statement_block }
```

最后，`WebClient.DownloadFileAsync`方法在后台下载文件。理想情况下，您应该使用一种机制，允许代码的一部分`System.Threading.ManualResetEventSlim`是一个具有`Set`和`Wait`方法的类，可以帮助进行此类信号传递。

对于这个活动，您需要执行以下步骤：

1.  添加一个进度更改`EventArgs`类（一个示例名称可以是`DownloadProgressChangedEventArgs`），在发布进度事件时可以使用。这应该有`ProgressPercentage`和`BytesReceived`属性。

1.  应该使用`System.Net`中的`WebClient`类来下载所请求的网络文件。您应该创建一个适配器类（建议的名称是`WebClientAdapter`），它可以隐藏您对`WebClient`的内部使用，使其对您的调用者不可见。

1.  您的适配器类应该提供三个事件——`DownloadCompleted`，`DownloadProgressChanged`和`InvalidUrlRequested`——供调用者订阅。

1.  适配器类将需要一个`DownloadFile`方法，该方法调用`WebClient`类的`DownloadFileAsync`方法来启动下载请求。这需要将基于字符串的网址转换为统一资源标识符（URI）类。`Uri.TryCreate()`方法可以从通过控制台输入的字符串创建绝对地址。如果调用`Uri.TryCreate`失败，您应该发布`InvalidUrlRequested`事件以指示此失败。

1.  `WebClient`有两个事件——`DownloadFileCompleted`和`DownloadProgressChanged`。您应该订阅这两个事件，并使用自己类似的事件重新发布它们。

1.  创建一个控制台应用程序，使用`WebClientAdapter`的一个实例（如*步骤 2*中创建的），并订阅这三个事件。

1.  通过订阅`DownloadCompleted`事件，您应该在控制台中指示成功。

1.  通过订阅`DownloadProgressChanged`，您应该向控制台报告进度消息，显示`ProgressPercentage`和`BytesReceived`的值。

1.  通过订阅`InvalidUrlRequested`事件，您应该使用不同的控制台背景颜色在控制台上显示警告。

1.  使用一个`do`循环，允许用户重复输入网址。直到用户输入空白地址退出为止，可以将该地址和临时目标文件路径传递给`WebClientAdapter.DownloadFile()`。

1.  一旦您使用各种下载请求运行控制台应用程序，您应该看到类似以下的输出：

```cpp
Enter a URL:
https://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles/StormEvents_details-ftp_v1.0_d1950_c20170120.csv.gz
Downloading https://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles/StormEvents_details-ftp_v1.0_d1950_c20170120.csv.gz...
Downloading...73% complete (7,758 bytes)
Downloading...77% complete (8,192 bytes)
Downloading...100% complete (10,597 bytes)
Downloaded to C:\Temp\StormEvents_details-ftp_v1.0_d1950_c20170120.csv.gz
Enter a URL:
https://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles/StormEvents_details-ftp_v1.0_d1954_c20160223.csv.gz
Downloading https://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles/StormEvents_details-ftp_v1.0_d1954_c20160223.csv.gz...
Downloading...29% complete (7,758 bytes)
Downloading...31% complete (8,192 bytes)
Downloading...54% complete (14,238 bytes)
Downloading...62% complete (16,384 bytes)
Downloading...84% complete (22,238 bytes)
Downloading...93% complete (24,576 bytes)
Downloading...100% complete (26,220 bytes)
Downloaded to C:\Temp\StormEvents_details-ftp_v1.0_d1954_c20160223.csv.gz
```

通过完成这个活动，您已经学会了如何订阅现有的.NET 基于事件的发布者类（`WebClient`）的事件，并在重新发布到适配器类（`WebClientAdapter`）中进行自己的规范调整，最终由控制台应用程序订阅。

注意

可以在[`packt.link/qclbF`](https://packt.link/qclbF)找到此活动的解决方案。

# 总结

在本章中，您深入了解了委托。您创建了自定义委托，并看到它们如何被其现代对应物，内置的`Action`和`Func`委托所取代。通过使用空引用检查，您发现了调用委托的安全方式，以及如何将多个方法链接在一起形成多播委托。您进一步扩展了委托，将其与`event`关键字一起使用，以限制调用，并遵循在定义和调用事件时的首选模式。最后，您了解了简洁的 lambda 表达式风格，并看到通过识别捕获和闭包的使用可以避免错误。

在下一章中，您将学习 LINQ 和数据结构，这是 C#语言的基本部分。
