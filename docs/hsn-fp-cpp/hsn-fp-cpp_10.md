# 使用类提高内聚性

我们之前讨论过如何使用函数和函数操作来组织我们的代码。然而，我们不能忽视过去几十年软件设计的主流范式——面向对象编程（OOP）。面向对象编程能够与函数式编程配合吗？它们之间是否存在任何兼容性，还是完全不相关？

事实证明，我们可以很容易地在类和函数之间进行转换。我通过我的朋友和导师 J.B. Rainsberger 学到，类只不过是一组部分应用的、内聚的纯函数。换句话说，我们可以使用类作为一个方便的位置，将内聚的函数组合在一起。但是，为了做到这一点，我们需要理解高内聚原则以及如何将函数转换为类，反之亦然。

本章将涵盖以下主题：

+   理解函数式编程和面向对象编程之间的联系

+   理解类如何等同于一组内聚的、部分应用的纯函数

+   理解高内聚性的必要性

+   如何将纯函数分组到类中

+   如何将一个类分解为纯函数

# 技术要求

您将需要一个支持 C++ 17 的编译器。我使用的是 GCC 7.3.0。

代码可以在 GitHub 的[https:/​/​github.​com/​PacktPublishing/​Hands-​On-​Functional-Programming-​with-​Cpp](https://github.%E2%80%8Bcom/PacktPublishing/Hands-On-Functional-Programming-with-Cpp)的`Chapter08`文件夹中找到。它包括并使用了`doctest`，这是一个单头开源单元测试库。您可以在其 GitHub 存储库中找到它，网址为[https:/​/github.​com/​onqtam/​doctest](https://github.%E2%80%8Bcom/onqtam/doctest)。

# 使用类提高内聚性

作为一名年轻的软件工程学生，我花了大量时间阅读面向对象编程的相关内容。我试图理解面向对象编程的工作原理，以及为什么它对现代软件开发如此重要。那时，大多数书籍都提到面向对象编程是将代码组织成具有封装、继承和多态三个重要属性的类。

近 20 年后，我意识到这种面向对象编程的观点相当有限。面向对象编程主要是在施乐帕克（Xerox PARC）开发的，这个实验室以产生大量高质量的想法而闻名，比如图形用户界面、点和点击、鼠标和电子表格等。艾伦·凯（Alan Kay）是面向对象编程的创始人之一，他在面对支持新的图形用户界面范式的大型代码库组织问题时，借鉴了自己作为生物学专业的知识。他提出了对象和类的概念，但多年后他表示，这种代码组织风格的主要思想是消息传递。他对对象的看法是，它们应该以与细胞类似的方式进行通信，在代码中模拟它们的化学信息传递。这就是为什么从他的观点来看，面向对象编程语言中的方法调用应该是一个从一个细胞或对象传递到另一个细胞或对象的消息。

一旦我们忘记了封装、继承和多态的概念，更加重视对象而不是类，函数式编程范式和面向对象编程之间的摩擦就消失了。让我们看看这种面向对象编程的基本观点会带我们去哪里。

# 从功能角度看待类

有多种方式来看待类。在知识管理方面，我将*类*概念化为分类——它是一种将具有相似属性的实例（或对象）分组的方式。如果我们以这种方式思考类，那么继承就是一种自然的属性——有一些对象类具有相似的属性，但它们在各种方面也有所不同；说它们继承自彼此是一种快速解释的方式。

然而，这种类的概念适用于我们的知识是准完全的领域。在软件开发领域，我们经常在应用领域的知识有限的情况下工作，而且领域随着时间的推移而不断扩展。因此，我们需要专注于代码结构，这些结构在概念之间有着薄弱的联系，使我们能够在了解领域的更多内容时进行更改或替换。那么，我们应该怎么处理类呢？

即使没有强大的关系，类在软件设计中也是一个强大的构造。它们提供了一种整洁的方法来分组方法，并将方法与数据结合在一起。与函数相比，它们可以帮助我们更好地导航更大的领域，因为我们最终可能会有成千上万个函数（如果不是更多）。那么，我们如何在函数式编程中使用类呢？

首先，正如你可能从我们之前的例子中注意到的那样，函数式编程将复杂性放在数据结构中。类通常是定义我们需要的数据结构的一种整洁方式，特别是在像 C++这样的语言中，它允许我们重写常见的运算符。常见的例子包括虚数、可测单位（温度、长度、速度等）和货币数据结构。每个例子都需要将数据与特定的运算符和转换进行分组。

其次，我们编写的不可变函数往往自然地分组成逻辑分类。在我们的井字棋示例中，我们有许多函数与我们称之为**line**的数据结构一起工作；我们的自然倾向是将这些函数分组在一起。虽然没有什么能阻止我们将它们分组在头文件中，但类提供了一个自然的地方来组合函数，以便以后能够找到它们。这导致了另一种类型的类——一个初始化一次的不可变对象，其每个操作都返回一个值，而不是改变其状态。

让我们更详细地看一下面向对象设计和函数结构之间的等价关系。

# 面向对象设计和函数式的等价关系

如果我们回到我们的井字棋结果解决方案，你会注意到有许多函数将`board`作为参数接收：

```cpp
auto allLines = [](const auto& board) {
...
};

auto allColumns = [](const auto& board) {
...
};

auto mainDiagonal = [](const auto& board){
...
};

auto secondaryDiagonal = [](const auto& board){
 ...
};

auto allDiagonals = [](const auto& board) -> Lines {
...
};

auto allLinesColumnsAndDiagonals = [](const auto& board) {
 ...
};
```

例如，我们可以定义一个棋盘如下：

```cpp
    Board board {
        {'X', 'X', 'X'},
        {' ', 'O', ' '},
        {' ', ' ', 'O'}
    };
```

然后，当我们将其传递给函数时，就好像我们将棋盘绑定到函数的参数上。现在，让我们为我们的`allLinesColumnsAndDiagonals` lambda 做同样的事情：

```cpp
auto bindAllToBoard = [](const auto& board){
    return map<string, function<Lines  ()>>{
        {"allLinesColumnsAndDiagonals",   
            bind(allLinesColumnsAndDiagonals, board)},
    };
};
```

前面的 lambda 和我们在早期章节中看到的许多其他例子都调用了其他 lambda，但它们没有捕获它们。例如，`bindAllToBoard` lambda 如何知道`allLinesColumnsAndDiagonal` lambda？这能够工作的唯一原因是因为 lambda 在全局范围内。此外，使用我的编译器，当尝试捕获`allLinesColumnsAndDiagonals`时，我会得到以下错误消息：`<lambda>` *cannot be captured because it does not have automatic storage duration*，因此如果我尝试捕获我使用的 lambda，它实际上不会编译。

我希望我即将说的是不言自明的，但我还是要说一下——对于生产代码，避免在全局范围内使用 lambda（以及其他任何东西）是一个好习惯。这也会迫使你捕获变量，这是一件好事，因为它会使依赖关系变得明确。

现在，让我们看看我们如何调用它：

```cpp
TEST_CASE("all lines, columns and diagonals with class-like structure"){
    Board board{
        {'X', 'X', 'X'},
        {' ', 'O', ' '},
        {' ', ' ', 'O'}
    };

    Lines expected{
        {'X', 'X', 'X'},
        {' ', 'O', ' '},
        {' ', ' ', 'O'},
        {'X', ' ', ' '},
        {'X', 'O', ' '},
        {'X', ' ', 'O'},
        {'X', 'O', 'O'},
        {'X', 'O', ' '}
    };

    auto boardObject = bindAllToBoard(board);
    auto all = boardObject["allLinesColumnsAndDiagonals"]();
    CHECK_EQ(expected, all);
}
```

这让你想起了什么吗？让我们看看我们如何在类中编写这个。我现在将其命名为`BoardResult`，因为我想不出更好的名字：

```cpp
class BoardResult{
    private:
        const vector<Line> board;

    public:
        BoardResult(const vector<Line>& board) : board(board){
        };

         Lines allLinesColumnsAndDiagonals() const {
             return concatenate3(allLines(board), allColumns(board),  
                 allDiagonals(board));
        }
};

TEST_CASE("all lines, columns and diagonals"){
 BoardResult boardResult{{
 {'X', 'X', 'X'},
 {' ', 'O', ' '},
 {' ', ' ', 'O'}
 }};

 Lines expected {
 {'X', 'X', 'X'},
 {' ', 'O', ' '},
 {' ', ' ', 'O'},
 {'X', ' ', ' '},
 {'X', 'O', ' '},
 {'X', ' ', 'O'},
 {'X', 'O', 'O'},
 {'X', 'O', ' '}
 };

 auto all = boardResult.allLinesColumnsAndDiagonals();
 CHECK_EQ(expected, all);
}
```

让我们回顾一下我们做了什么：

+   我们看到更多的函数将`board`作为参数。

+   我们决定使用一个单独的函数将`board`参数绑定到一个值，从而获得一个字符串表示函数名和与该值绑定的 lambda 之间的映射。

+   要调用它，我们需要先调用初始化函数，然后才能调用部分应用的 lambda。

+   *这看起来非常类似于一个类*——使用构造函数传递类方法之间共享的值，然后调用方法而不传递参数。

因此，*一个类只是一组部分应用的 lambda*。但我们如何将它们分组呢？

# 高内聚原则

在我们之前的例子中，我们根据它们都需要相同的参数`board`将函数分组在一起。我发现这是一个很好的经验法则。然而，我们可能会遇到更复杂的情况。

为了理解为什么，让我们看另一组函数（为了讨论的目的，实现已被忽略）：

```cpp
using Coordinate = pair<int, int>;

auto accessAtCoordinates = [](const auto& board, const Coordinate& coordinate)
auto mainDiagonalCoordinates = [](const auto& board)
auto secondaryDiagonalCoordinates = [](const auto& board)
auto columnCoordinates = [](const auto& board, const auto& columnIndex)
auto lineCoordinates = [](const auto& board, const auto& lineIndex)
auto projectCoordinates = [](const auto& board, const auto& coordinates)
```

这些函数应该是之前定义的`BoardResult`类的一部分吗？还是应该是另一个类`Coordinate`的一部分？或者我们应该将它们拆分，其中一些归入`BoardResult`类，另一些归入`Coordinate`类？

我们以前的方法并不适用于所有的功能。如果我们仅仅看它们的参数，所有之前的函数都需要`board`。然而，其中一些还需要`coordinate / coordinates`作为参数。`projectCoordinates`应该是`BoardResult`类的一部分，还是`Coordinate`类的一部分？

更重要的是，我们可以遵循什么基本原则将这些功能分组到类中呢？

由于代码的静态结构没有明确的答案，我们需要考虑代码的演变。我们需要问的问题是：

+   我们期望哪些函数一起改变？我们期望哪些函数分开改变？

+   这种推理方式引导我们到高内聚原则。但是，让我们先解开它。我们所说的内聚是什么意思？

作为一名工程师和科学迷，我在物理世界中遇到了内聚。例如，当我们谈论水时，构成液体的分子倾向于粘在一起。我也遇到了内聚作为一种社会力量。作为一个与试图采用现代软件开发实践的客户合作的变革者，我经常不得不处理群体凝聚力——人们围绕一种观点聚集在一起的倾向。

当我们谈论函数的内聚性时，没有物理力量将它们推在一起，它们绝对不会固守观点。那么，我们在谈论什么呢？我们在谈论一种神经力量，可以这么说。

人脑有着发现模式和将相关物品分组到类别中的巨大能力，再加上一种神奇的快速导航方式。将函数绑在一起的力量在我们的大脑中——它是从看似无关的功能组合中出现的统一目的的发现。

高内聚性很有用，因为它使我们能够理解和导航一些大概念（如棋盘、线和标记），而不是数十甚至数百个小函数。此外，当（而不是如果）我们需要添加新的行为或更改现有行为时，高内聚性将使我们能够快速找到新行为的位置，并且以最小的更改添加它到网络的其余部分。

内聚是软件设计的一个度量标准，由拉里·康斯坦丁在 20 世纪 60 年代作为他的*结构化设计*方法的一部分引入。通过经验，我们注意到高内聚性与低变更成本相关。

让我们看看如何应用这个原则来将我们的函数分组到类中。

# 将内聚的函数分组到类中

正如之前讨论的，我们可以从一个类的统一目的或概念的角度来看内聚。然而，我通常发现更彻底的方法是根据代码的演变来决定函数组，以及未来可能发生的变化以及它可能触发的其他变化。

你可能不会指望从我们的井字棋结果问题中学到很多东西。它相当简单，看起来相当容易控制。然而，网上的快速搜索会带我们找到一些井字棋的变体，包括以下内容：

+   *m x n*棋盘，赢家由一排中的*k*个项目决定。一个有趣的变体是五子棋，在*15 x 15*的棋盘上进行，赢家必须连成 5 个。

+   一个 3D 版本。

+   使用数字作为标记，并以数字的总和作为获胜条件。

+   使用单词作为标记，获胜者必须在一行中放置 3 个带有 1 个共同字母的单词。

+   使用*3 x 3*的 9 个棋盘进行游戏，获胜者必须连续获胜 3 个棋盘。

这些甚至不是最奇怪的变体，如果你感兴趣，可以查看维基百科上关于这个主题的文章[`en.wikipedia.org/wiki/Tic-tac-toe_variants`](https://en.wikipedia.org/wiki/Tic-tac-toe_variants)。

那么，在我们的实现中可能会发生什么变化呢？以下是一些建议：

+   棋盘大小

+   玩家数量

+   标记

+   获胜规则（仍然是一行，但条件不同）

+   棋盘拓扑——矩形、六边形、三角形或 3D 而不是正方形

幸运的是，如果我们只是改变了棋盘的大小，我们的代码实际上不会有太大变化。事实上，我们可以传入一个更大的棋盘，一切仍然可以正常工作。改变玩家数量只需要做很小的改动；我们假设他们有不同的标记，我们只需要将`tokenWins`函数绑定到不同的标记值上。

那么获胜规则呢？我们假设规则仍然考虑了行、列和对角线，因为这是井字游戏的基本要求，所有变体都使用它们。然而，我们可能不考虑完整的行、列或对角线；例如，在五子棋中，我们需要在大小为 15 的行、列或对角线上寻找 5 个标记。从我们的代码来看，这只是选择其他坐标组的问题；我们不再需要寻找被标记`X`填满的完整行，而是需要选择所有可能的五连坐标集。这意味着我们的与坐标相关的函数需要改变——`lineCoordinates`、`mainDiagonalCoordinates`、`columnCoordinates`和`secondaryDiagonalCoordinates`。它们将返回一个五连坐标的向量，这将导致`allLines`、`allColumns`和`allDiagonals`的变化，以及我们连接它们的方式。

如果标记是一个单词，获胜条件是找到单词之间的共同字母呢？好吧，坐标是一样的，我们获取行、列和对角线的方式也是一样的。唯一的变化在于`fill`条件，所以这相对容易改变。

这引出了最后一个可能的变化——棋盘拓扑。改变棋盘拓扑将需要改变棋盘数据结构，以及所有的坐标和相应的函数。但是这是否需要改变行、列和对角线的规则呢？如果我们切换到 3D，那么我们将有更多的行、更多的列，以及一个不同的对角线寻址方式——所有坐标的变化。矩形棋盘本身并没有对角线；我们需要使用部分对角线，比如在五子棋的情况下。至于六边形或三角形的棋盘，目前还没有明确的变体，所以我们可以暂时忽略它们。

这告诉我们，如果我们想要为变化做好准备，我们的函数应该围绕以下几个方面进行分组：

+   规则（也称为**填充条件**）

+   坐标和投影——并为多组行、列和对角线准备代码

+   基本的棋盘结构允许基于坐标进行访问

这就解决了问题——我们需要将坐标与棋盘本身分开。虽然坐标数据类型将与棋盘数据类型同时改变，但由于游戏规则的原因，提供行、列和对角线坐标的函数可能会发生变化。因此，我们需要将棋盘与其拓扑分开。

在**面向对象设计**（**OOD**）方面，我们需要在至少三个内聚的类之间分离程序的责任——`Rules`，`Topology`和`Board`。`Rules`类包含游戏规则——基本上是我们如何计算获胜条件，当我们知道是平局时，或者游戏何时结束。`Topology`类涉及坐标和棋盘的结构。`Board`类应该是我们传递给算法的结构。

那么，我们应该如何组织我们的函数？让我们列个清单：

+   **规则**：`xWins`，`oWins`，`tokenWins`，`draw`和`inProgress`

+   **Topology**：`lineCoordinates`，`columnCoordinates`，`mainDiagonalCoordinates`和`secondaryDiagonalCoordinates`

+   **Board**：`accessAtCoordinates`和`allLinesColumnsAndDiagonals`

+   **未决**：`allLines`，`allColumns`，`allDiagonals`，`mainDiagonal`和`secondaryDiagonal`

总是有一系列函数可以成为更多结构的一部分。在我们的情况下，`allLines`应该是`Topology`类还是`Board`类的一部分？我可以为两者找到同样好的论点。因此，解决方案留给编写代码的程序员的直觉。

然而，这显示了你可以用来将这些函数分组到类中的方法——考虑可能发生的变化，并根据哪些函数将一起变化来分组它们。

然而，对于练习这种方法有一个警告——避免陷入过度分析的陷阱。代码相对容易更改；当你对可能发生变化的事情知之甚少时，让它工作并等待直到同一代码区域出现新的需求。然后，你会对函数之间的关系有更好的理解。这种分析不应该花费你超过 15 分钟；任何额外的时间很可能是过度工程。

# 将一个类分割成纯函数

我们已经学会了如何将函数分组到一个类中。但是我们如何将代码从一个类转换为纯函数？事实证明，这是相当简单的——我们只需要使函数成为纯函数，将它们移出类，然后添加一个初始化器，将它们绑定到它们需要的数据上。

让我们举另一个例子，一个执行两个整数操作数的数学运算的类：

```cpp
class Calculator{
    private:
        int first;
        int second;

    public:
        Calculator(int first, int second): first(first), second(second){}

        int add() const {
            return first + second;
        }

        int multiply() const {
            return first * second;
        }

        int mod() const {
            return first % second;
        }

};

TEST_CASE("Adds"){
    Calculator calculator(1, 2);

    int result = calculator.add();

    CHECK_EQ(result, 3);
}

TEST_CASE("Multiplies"){
    Calculator calculator(3, 2);

    int result = calculator.multiply();

    CHECK_EQ(result, 6);
}

TEST_CASE("Modulo"){
    Calculator calculator(3, 2);

    int result = calculator.mod();

    CHECK_EQ(result, 1);
}
```

为了使它更有趣，让我们添加另一个函数，用于反转第一个参数：

```cpp
class Calculator{
...
    int negateInt() const {
        return -first;
    }
...
}

TEST_CASE("Revert"){
    Calculator calculator(3, 2);

    int result = calculator.negateInt();

    CHECK_EQ(result, -3);
}
```

我们如何将这个类分割成函数？幸运的是，这些函数已经是纯函数。很明显，我们可以将函数提取为 lambda：

```cpp
auto add = [](const auto first, const auto second){
    return first + second;
};

auto multiply = [](const auto first, const auto second){
    return first * second;
};

auto mod = [](const auto first, const auto second){
    return first % second;
};

auto negateInt = [](const auto value){
    return -value;
};
```

如果你真的需要，让我们添加初始化器：

```cpp
auto initialize = [] (const auto first, const auto second) -> map<string, function<int()>>{
    return  {
        {"add", bind(add, first, second)},
        {"multiply", bind(multiply, first, second)},
        {"mod", bind(mod, first, second)},
        {"revert", bind(revert, first)}
    };
};
```

然后，可以进行检查以确定一切是否正常工作：

```cpp
TEST_CASE("Adds"){
    auto calculator = initialize(1, 2);

    int result = calculator["add"]();

    CHECK_EQ(result, 3);
}

TEST_CASE("Multiplies"){
    auto calculator = initialize(3, 2);

    int result = calculator["multiply"]();

    CHECK_EQ(result, 6);
}

TEST_CASE("Modulo"){
    auto calculator = initialize(3, 2);

    int result = calculator["mod"]();

    CHECK_EQ(result, 1);
}

TEST_CASE("Revert"){
    auto calculator = initialize(3, 2);

    int result = calculator["revert"]();

    CHECK_EQ(result, -3);
}

```

这让我们只剩下一个未决问题——如何将不纯的函数转变为纯函数？我们将在第十二章中详细讨论这个问题，*重构为纯函数*。现在，让我们记住本章的重要结论——*一个类只不过是一组内聚的、部分应用的函数*。

# 总结

在本章中，我们有一个非常有趣的旅程！我们成功地以一种非常优雅的方式将两种看似不相关的设计风格——面向对象编程和函数式编程联系起来。纯函数可以根据内聚性原则分组到类中。我们只需要发挥想象力，想象一下函数可能发生变化的情景，并决定哪些函数应该分组在一起。反过来，我们总是可以通过使它们成为纯函数并反转部分应用，将函数从一个类移动到多个 lambda 中。

面向对象设计和函数式编程之间没有摩擦；它们只是实现功能的代码的两种不同结构方式。

我们使用函数进行软件设计的旅程还没有结束。在下一章中，我们将讨论如何使用**测试驱动开发**（**TDD**）设计函数。
