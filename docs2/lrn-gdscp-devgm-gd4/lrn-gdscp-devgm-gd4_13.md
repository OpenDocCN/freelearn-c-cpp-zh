# 13

# 面向对象编程的继续和高级主题

在这本书的整个过程中，我们学习了大量的编程知识，从基本的变量、控制流和类到 GDScript 特有的内容，例如访问场景树中的节点和特定的注释。然而，不要误解——还有更多知识可以帮助我们更容易、更快地解决问题。

经过多年的学习和专业应用我的编程技能，我可以自信地说，计算机科学是一个深奥且值得继续学习的领域。此外，每隔几年就会有一种新技术崭露头角，等待我们去研究。

在本章中，我们将探讨一系列更高级的技术和概念，这将使你的编程技能达到新的高度！

在本章中，我们将涵盖以下主要主题：

+   `super`关键字

+   静态变量和函数

+   枚举

+   Lambda 函数

+   通过值或引用传递参数

+   `@tool`注释

# 技术要求

与每一章一样，你可以在 GitHub 仓库的该章节子文件夹中找到最终代码：[`github.com/PacktPublishing/Learning-GDScript-by-Developing-a-Game-with-Godot-4/tree/main/chapter13`](https://github.com/PacktPublishing/Learning-GDScript-by-Developing-a-Game-with-Godot-4/tree/main/chapter13)。

# `super`关键字

在*第四章*中，我们学习了继承以及如何在继承类中覆盖基类中的函数。这种覆盖用全新的主体替换了函数，并丢弃了基类的原始实现。然而，有时我们仍然希望执行在父类中定义的原始逻辑。

为了实现这一点，我们可以使用`super`关键字。这个关键字让我们直接访问当前类基于的父类的所有函数。考虑以下示例，我们想在游戏中拥有不同种类的箭来射击敌人：

```cpp
class BaseArrow:
   func describe_damage():
      print("Pierces a person")
class FireArrow extends BaseArrow:
   func describe_damage():
      super()
      print("And sets them ablaze")
```

在这里，我们定义了一个`BaseArrow`类，它是所有种类箭的基础。它有一个函数`describe_damage()`，该函数只是通过将`Pierces a person`打印到控制台来描述箭造成的伤害。

当我们覆盖`FireArrow`类的`describe_damage()`函数时，我们首先将`super()`作为一个函数调用。这将执行`BaseArrow`类的原始`describe_damage()`函数，然后再执行其余部分。

让我们执行一些使用这些类的代码：

```cpp
var fire_arrow: FireArrow = FireArrow.new()
fire_arrow.describe_damage()
```

结果将如下所示：

```cpp
Pierces a person
And sets them ablaze
```

你可以看到，使用`super()`关键字执行了基类中的`describe_damage()`函数，以及`FireArrow`类其余的实现。

`super`关键字提供了访问我们继承的底层类的方法；无论我们是否覆盖了该函数，它总是会返回原始的函数。让我们继续探讨另一个新关键字——`static`。

# 静态变量和函数

我们接下来要查看的下一个关键字是`static`。我们可以通过在这个关键字前放置它来声明一个变量或函数为静态：

```cpp
class Enemy:
   static var damage: float = 10.0
   static func do_battle_cry():
      print("Aaaaaargh!")
```

静态变量和函数是在类本身上声明的。这意味着它们可以在不创建类实例的情况下访问：

```cpp
print(Enemy.damage)
Enemy.do_battle_cry ()
```

静态变量是为了包含与整个对象类绑定在一起的信息。但要注意 – 以下是静态变量和函数的两个大陷阱：

+   在 GDScript 中，静态变量可以被赋予新的值，你可以在游戏执行过程中更改它们。理想情况下，你不想这样做，因为这可能会以难以调试的方式影响你的程序。

+   从一个静态函数中，你可以调用其他函数并使用类的成员变量，但前提是它们也被定义为静态。因为静态函数是在类本身上定义的，它们没有初始化对象的全部上下文。静态函数需要非常自包含。

总的来说，在 GDScript 中，你不会经常看到静态变量和函数，但它是一个在许多面向对象编程语言中，如 C++或 Java，广为人知的概念。接下来，让我们看看枚举。

# 枚举

**枚举**，简称**枚举**，是一种变量类型，它定义了一组需要组合在一起的常量。与我们要存储特定值的普通常量不同，枚举会自动为常量分配值。

在**第二章**和**第五章**中，我们看到了拥有良好命名的变量非常重要。这样，我们总能知道它们将包含什么。实际上，我们也可以为变量的值使用命名值。使用命名值，我们可以将一个可读性强的名称与某个值关联起来，使代码更易于阅读。它还从代码中移除了魔法数字。看看这个枚举：

```cpp
enum DAMAGE_TYPES {
   NONE,
   FIRE,
   ICE
}
```

这里，我们创建了一个名为`DAMAGE_TYPES`的枚举，它定义了三个命名值 – `NONE`、`FIRE`和`ICE`。你可以这样访问这些值：

```cpp
DAMAGE_TYPES.FIRE
```

让我们尝试将它们打印出来：

```cpp
print(DAMAGE_TYPES.NONE)
print(DAMAGE_TYPES.FIRE)
print(DAMAGE_TYPES.ICE)
```

你会看到它打印出以下内容：

```cpp
0
1
2
```

这是因为枚举内的每个名称都与一个整数值相关联。然而，我们不再使用这些粗略的整数，现在我们可以使用易于阅读的名称。第一个命名值与`0`相关联，每个随后的值递增 1。

枚举也可以用来类型提示变量；这样，我们知道变量需要被分配来自某个类型的枚举值：

```cpp
var damage_type: DAMAGE_TYPES = DAMAGE_TYPES.FIRE
match damage_type:
   DAMAGE_TYPES.NONE:
      print("Nothing special happens")
   DAMAGE_TYPES.FIRE:
      print("You catch fire! ")
   DAMAGE_TYPES.ICE:
      print("You freeze!")
```

在这个例子中，我们将`damage_type`变量类型提示为`DAMAGE_TYPES`。然后，例如，我们可以匹配这个变量并确定要执行的操作。

枚举与字符串的比较

现在，你可能会想，“*为什么我们不用字符串来读取值呢？*”简单来说，这是因为字符串与整数（枚举值的底层数据类型）相比，处理起来更慢且占用更多内存。另一个原因是易用性。枚举有一组有限的值，即我们定义的值，而字符串可以有任意数量的字符。因此，当我们使用枚举时，我们可以确信我们只处理我们知道的值。

我们还可以从完全不同的类中访问在一个类中定义的枚举；就像静态变量和函数一样，它们可以直接从类类型访问：

```cpp
class Arrow:
   Enum DAMAGE_TYPES {
      NONE,
      FIRE
   }
func _ready():
   var damage_type: DAMAGE_TYPES enum from within the Arrow class. Later, we can access this enum by using Arrow.DAMAGE_TYPES directly.
			In this section, we looked at enums, named values that help us by providing human-readable labels. Next, we’ll take a look at lambda functions.
			Lambda functions
			So far, every function we have written belonged to a class or file, which could be treated as a class, but there is actually a way to define functions separately from any class definition. These kinds of functions are called **lambda functions**.
			Creating a lambda function
			Let’s take a look at a lambda function:

```

var print_hello: Callable = func(): print("Hello")

```cpp

			You can see that we’ve defined a function, just as we normally do, but this time without a function name. Instead, we assigned the function to a variable. This variable now contains the function in the form of the `Callable` object type. We can call a `Callable` object later on, like this:

```

print_hello.call()

```cpp

			This will run the function that we defined and, thus, print out `Hello` to the console.
			Lambda functions, just like normal functions, can take arguments too:

```

var print_largest: Callable = func(number_a: float, number_b: float):

if number_a > number_b:

print(number_a)

else:

print(number_b)

```cpp

			In this example, you can also see that lambda functions can contain multiple lines of code in the form of a code block, where each line has the same level of indentation.
			Where to use lambda functions
			So, where would we use lambda functions? Well, they are very useful in scenarios where you need a relatively small function but don’t want to have it as a permanent residence in the class.
			One great application of lambda functions is connecting signals. If we have a button for example, then we can connect to its pressed signal using a lambda function, as follows:

```

button.connect("pressed", pressed signal, 我们的自定义 lambda 函数被调用并打印出 Button pressed!。

            它的另一个用例是`filter()`或`sort_custome()`函数，这些函数可以使用 lambda 函数来过滤或排序数组中的元素：

```cpp
[0, 1, 2, 3, 4].filter(func(number: int): return number % 2 == 0)
[0, 3, 2, 4, 1].sort_custome(func(number_a: int, number_b: int): return number_a < number_b)
```

            每个数组都有一个`filter()`和`sort_custome()`函数，这些函数接受`Callable`作为参数。`filter()`函数将过滤掉数组中任何返回`false`的元素，从而得到一个只包含返回`true`的元素的数组。在上面的例子中，这导致了一个只包含偶数的数组。

            `sort_custome()`函数使用我们提供给它的`Callable`对数组内的元素进行排序。lambda 函数应该接受两个元素，如果第一个元素应该在第二个元素之前排序，则返回`true`；否则，应返回`false`。这样，我们可以定义自己的规则来排序数组的元素。

            使用我们的 lambda 函数运行`filter()`和`sort_custome()`后的结果数组如下：

```cpp
[0, 2, 4]
[0, 1, 2, 3, 4]
```

            更多信息

            更多关于 lambda 函数的信息，请参阅官方文档：[`docs.godotengine.org/en/stable/classes/class_callable.html`](https://docs.godotengine.org/en/stable/classes/class_callable.html)。

            现在我们已经知道了 lambda 函数是什么，让我们看看我们可以以不同的方式向函数传递值。

            按值或按引用传递参数

            当向函数传递参数时，实际上有两种不同的方式可以使这些参数到达函数体中——按值或按引用。作为程序员，我们并不选择使用哪一种；GDScript 会根据我们提供给函数的值的类型来做出这个决定。让我们更深入地了解一下两种传递值的方法，适用于每种数据类型，以及为什么了解这些差异很重要。

            按值传递

            按值传递意味着 GDScript 将值的精确副本发送到函数中。这种方法非常简单且可预测，因为我们得到了在函数中调用的新变量。然而，由于复制数据需要时间，对于大数据类型来说可能会相当慢。

            按值传递的数据类型包括任何简单的内置数据类型，如整数、浮点数和布尔值。还有一些稍微复杂一些的类，如字符串、`Vector2`和`Colors`，也是按值传递的。这个列表并不全面。一般规则是包含任何不是数组、不是字典且不是从`Object`类继承的任何东西。

            让我们看看按值传递在实际中是什么样子：

```cpp
func _ready():
   var number: int = 5
   print("Number before the function: ", number)
   function_taking_integers(number)
   print("Number after the function: ", number)
   var string: String = "Hello there!"
   print("String before the function: ", string)
   function_taking_strings(string)
   print("String after the function: ", string)
func function_taking_integers(number: int):
   number += 10
   print("Number during the function: ", number)
func function_taking_strings(string: String):
   string[0] = "W"
   print("String during the function: ", string)
```

            在这里，你可以看到我们有两个函数，分别接受一个整数和一个字符串，并在其执行过程中修改参数的值。我们还打印出整数和字符串的值，在函数执行之前、期间和之后，以查看原始变量（来自`_ready()`函数）是否被更改。运行此代码将打印出以下内容：

```cpp
Number before the function: 5
Number during the function: 15
Number after the function: 5
String before the function: Hello there!
String during the function: Wello there!
String after the function: Hello there!
```

            我们可以看到，尽管在函数执行过程中值以某种方式被更改了，但原始值并没有改变。这就是按值传递的乐趣；我们不需要担心副作用。

            函数的副作用

            在程序员的语言中，副作用意味着函数以不是直接明显的方式改变程序的状态，改变其作用域之外的变量。你应尽可能避免这种情况，以便更容易理解函数的作用。

            这就是按值传递的工作方式——只是数据的直接复制。现在，我们将看到对比的概念——按引用传递。

            通过引用传递

            向函数传递值的另一种方式是通过引用。这意味着 GDScript 不会复制整个值，而是发送一个指向值的引用。这个引用指向实际值存储的位置，并可用于访问和更改它。

            这种传递参数的模式用于数组、字典以及从`Object`类继承的任何类，这包括所有类型的节点。它本质上用于传递更大的数据类型，因为复制它们的完整值会花费太多时间并减慢游戏的执行速度。

            下面是一个按引用传递的例子：

```cpp
func _ready():
   var dictionary: Dictionary = { "value": 5 }
   print("Dictionary before the function: ", dictionary)
   function_taking_dictionary(dictionary)
   print("Dictionary after the function: ", dictionary)
func function_taking_dictionary(dictionary: Dictionary):
   dictionary["a_value"] = "has changed"
   print("Dictionary during the function: ", dictionary)
```

            再次，我们使用之前相同的设置，在每一步打印出我们的字典。我们运行以下代码：

```cpp
Dictionary before the function: { "value": 5 }
Dictionary during the function: { "value": 5, "a_value": "has changed" }
Dictionary after the function: { "value": 5, "a_value": "has changed" }
```

            如预期的那样，我们可以看到在函数运行后，`_ready()`函数中的原始字典也被更改了！这是副作用在起作用。

            通常，一个好的做法是永远不要更改传入函数的值和变量，并且始终复制它们或直接使用它们来计算另一个变量的值。如果有疑问，最好测试一个值是按值传递还是按引用传递；这样，你永远不会遇到意外的错误。

            复制数组或字典

            如果你真的想复制一个数组或字典，那么你可以使用定义在这些数据类型上的 `duplicate()` 函数。这个函数将返回一个数组或字典的副本，你可以安全地对其进行修改。

            查看文档以获取更多详细信息：[`docs.godotengine.org/fr/4.x/classes/class_array.html#class-array-method-duplicate`](https://docs.godotengine.org/fr/4.x/classes/class_array.html#class-array-method-duplicate)。

            现在，我们将转换方向，看看我们如何从编辑器内部创建编辑器工具。

            @tool 注解

            除了在游戏执行期间使用 GDScript 运行代码外，我们实际上还可以在编辑器本身中运行代码。在编辑器中运行代码赋予我们可视化事物的能力，例如角色的跳跃高度，或自动化我们的工作流程。通过这样做，我们扩展了 Godot 编辑器以满足我们自己的特定需求。在编辑器中运行 GDScript 代码有多种方式，从运行单独的脚本到编写整个插件，但最简单的方法是使用 `@tool` 注解。

            `@tool` 注解是一种可以添加到任何脚本顶部的注解。它的作用是，具有该脚本的节点将在编辑器中运行它们的脚本，就像它们在游戏中实例化一样。这意味着它们的所有代码都是在编辑器中运行的。

            当我们编辑场景并希望在编辑器中预览事物时，例如玩家的健康状态，或使用代码创建新节点，这非常有用。

            了解这一点后，我们可以通过在顶部添加 `@tool` 注解来调整我们的玩家脚本，以更新编辑器中的健康标签：

```cpp
@tool
class_name Player extends CharacterBody2D
const MAX_HEALTH: int = 10
@onready var _health_label: Label = $Health
@export var health: int = 10:
   set(new_value):
      health = new_value
      update_health_label()
func _ready():
   update_health_label()
func update_health_label():
   if not is_instance_valid(_health_label):
      return
   _health_label.text = str(health) + "/" + str(MAX_HEALTH)
```

            这个示例是更新编辑器中健康标签所需的最小代码量。然而，你只需在现有的玩家脚本顶部添加 `@tool` 注解，它就会发挥作用。现在，每次你在编辑器中更改玩家的健康状态时，健康标签都会自动反映这一变化。

            @tool 的风险

            `@tool` 注解非常强大，但并非没有风险。如果不小心，它可能会永久删除场景中的事物，并轻松更改节点的值，所以请谨慎使用。

            然而，有时你希望在游戏中使用一个节点，并在编辑器中运行一些代码。当我们这样做时，我们需要一种方法来区分代码是在游戏还是编辑器中运行的。这可以通过使用 `Engine.is_editor_hint()` 函数来实现。这个全局 `Engine` 对象上的函数，如果我们从编辑器中运行代码，则返回 `true`；如果从游戏中运行，则返回 `false`：

```cpp
if Engine.is_editor_hint():
   # Code to execute in editor.
if not Engine.is_editor_hint():
   # Code to execute in game.
```

            这个代码示例向我们展示了在编辑器或游戏中运行代码之间的区别是多么容易区分。

            更多信息

            想了解更多关于 `@tool` 注释和在编辑器中运行代码的信息？请查看官方文档：[`docs.godotengine.org/en/stable/tutorials/plugins/running_code_in_the_editor.html`](https://docs.godotengine.org/en/stable/tutorials/plugins/running_code_in_the_editor.html)。

            使用 `@tool` 注释明智地，我们可以使我们的工作流程更简单、更快。可能性是无限的；你甚至可以从这些脚本之一中访问和更改 Godot 编辑器几乎每一个方面，但这超出了本书的范围。

            摘要

            本章深入探讨了使用 GDScript 编程的一些更高级的主题。我们通过 `super` 和 `static` 关键字以及按值或按引用传递的区别，扩展了我们对面向对象编程的知识。然后，我们看到了 GDScript 编程语言的更多功能，例如枚举和 lambda 函数。我们以在 Godot 编辑器本身中运行代码的方式结束了本章，使用的是 `@tool` 注释。

            测验时间

                +   假设我们有一个名为 `Character` 的类，它有一个名为 `move()` 的函数。现在，我们创建一个 `Player` 类，它继承自这个 `Character` 类并重写这个 `move()` 函数。但是，我们不想完全重写它，而是想扩展 `Character` 类的 `move()` 函数的原始功能。我们可以在 `Player` 类中调用 `Character` 类的原始 `move()` 函数的哪个关键字？

                +   标记为 `static` 的函数能否调用未标记为 `static` 的函数？

                +   以下代码片段将打印出什么？

    ```cpp
    enum COLLECTIBLE_TYPE {
       HEALTH,
       UPGRADE,
       DAMAGE,
    }
    print(COLLECTIBLE_TYPES.DAMAGE)
    ```

                    +   容器类型，如数组和字典，是唯一按引用传递的类型吗？

                +   如果我们想在编辑器中运行脚本，我们在脚本顶部使用什么注释？

```cpp

```
