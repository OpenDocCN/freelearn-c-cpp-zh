# 第八章：从输入数据到输出数据的函数思维

在我迈向理解函数式编程的旅程中，我遇到了一个困难的障碍——我的思维是在完全不同的编程风格中训练的。我们称之为命令式面向对象编程。那么，我如何将我的思维模式从对象思考转变为函数思考？我如何以一种良好的方式将这两者结合起来？

我首先研究了函数式编程资源。不幸的是，其中大多数都集中在数学和概念的内在美上，这对于那些已经能够以这些术语思考的人来说是很好的。但是，如果你只是想学习它们呢？难道只能通过数学理论来学习吗？虽然我喜欢数学，但我已经生疏了，我宁愿找到更实际的方法。

我已经接触过各种编写代码的方式，比如 Coderetreats、Coding Dojos，或者与来自欧洲各地的程序员进行配对编程。我逐渐意识到，解决这个问题的一个简单方法是专注于输入和输出，而不是专注于它们之间的模型。这是学习以函数思考的一个更具体和实际的方法，接下来我们将探讨这个问题。

本章将涵盖以下主题：

+   函数思维的基础。

+   重新学习如何识别功能的输入和输出数据，并利用类型推断

+   将数据转换定义为纯函数

+   如何使用典型的数据转换，比如 map、reduce、filter 等

+   如何使用函数思维解决问题

+   为围绕函数设计的代码设计错误管理

# 技术要求

您将需要一个支持 C++ 17 的编译器。我使用的是 GCC 7.3.0。

代码可以在 GitHub 上找到[`github.com/PacktPublishing/Hands-On-Functional-Programming-with-Cpp`](https://github.com/PacktPublishing/Hands-On-Functional-Programming-with-Cpp)，在`Chapter06`文件夹中。它包括并使用了`doctest`，这是一个单头开源单元测试库。您可以在其 GitHub 存储库上找到它[`github.com/onqtam/doctest`](https://github.com/onqtam/doctest)。

# 通过函数从输入数据到输出数据

我的计算机编程教育和作为程序员的重点大多是编写代码，而不是深入理解输入和输出数据。当我学习测试驱动开发（TDD）时，这种重点发生了变化，因为这种实践迫使程序员从输入和输出开始。通过应用一种称为“TDD As If You Meant It”的极端形式，我对程序的核心定义有了新的认识——接受输入数据并返回输出数据。

然而，这并不容易。我的训练使我重新思考构成程序的事物。但后来，我意识到这些事物只是纯函数。毕竟，任何程序都可以按照以下方式编写：

+   一组纯函数，如前所定义

+   一组与输入/输出（I/O）交互的函数

如果我们将程序简化到最小，并将所有 I/O 分开，找出其余程序的 I/O，并为我们能够的一切编写纯函数，我们刚刚迈出了以函数思考的第一步。

接下来的问题是——这些函数应该是什么？在本章中，我们将探讨最简单的使用函数进行设计的方法：

1.  从输入数据开始。

1.  定义输出数据。

1.  逐步定义一系列转换（纯函数），将输入数据转换为输出数据。

让我们看一些对比两种编写程序的方法的例子。

# 命令式与函数式风格的工作示例

为了展示不同的方法之间的差异，我们需要使用一个问题。我喜欢使用从游戏中衍生出的问题来练习新的编程技术。一方面，这是一个我不经常接触的有趣领域。另一方面，游戏提供了许多常见的商业应用所没有的挑战，从而使我们能够探索新的想法。

在接下来的部分中，我们将看一个问题，让人们学会如何开始以函数的方式思考——**井字棋结果**问题。

# 井字棋结果

井字棋结果问题有以下要求——给定一个可能为空的井字棋棋盘或已经有了棋子的棋盘，打印出游戏的结果，如果游戏已经结束，或者打印出仍在进行中的游戏。

看起来问题似乎相当简单，但它将向我们展示功能和命令式**面向对象**（**OO**）方法之间的根本区别。

如果我们从面向对象的角度来解决问题，我们已经在考虑一些要定义的对象——一个游戏，一个玩家，一个棋盘，也许一些代表`X`和`O`的表示（我称之为标记），等等。然后，我们可能会考虑如何连接这些对象——一个游戏有两个玩家和一个棋盘，棋盘上有标记或空格等等。正如你所看到的，这涉及到很多表示。然后，我们需要在某个地方实现一个`computeResult`方法，返回`GameState`，要么是`XWon`，`OWon`，`draw`，要么是`InProgress`。乍一看，`computeResult`似乎适合于`Game`类。该方法可能需要在`Board`内部循环，使用一些条件语句，并返回相应的`GameState`。

我们将使用一些严格的步骤来帮助我们以不同的方式思考代码结构，而不是使用面向对象的方法：

1.  清晰地定义输入；给出例子。

1.  清晰地定义输出；给出例子。

1.  识别一系列功能转换，你可以将其应用于输入数据，将其转换为输出数据。

在我们继续之前，请注意，这种心态的改变需要一些知识和实践。我们将研究最常见的转换，为您提供一个良好的开始，但您需要尝试这种方法。

# 输入和输出。

我们作为程序员学到的第一课是任何程序都有输入和输出。然后我们继续把我们的职业生涯的其余部分放在输入和输出之间发生的事情上，即代码本身。

尽管如此，输入和输出值得程序员更多的关注，因为它们定义了我们软件的要求。我们知道，软件中最大的浪费是实现了完美的功能，但却没有完成它应该完成的任务。

我注意到程序员很难重新开始思考输入和输出。对于给定功能的输入和输出应该是什么的看似简单的问题经常让他们感到困惑和困惑。所以，让我们详细看看我们问题的输入和输出数据。

在这一点上，我们将做一些意想不到的事情。我从业务分析师那里学到了一个很棒的技巧——在分析一个功能时最好从输出开始，因为输出往往比输入数据更小更清晰。所以，让我们这样做。

# 输出数据是什么？

我们期望什么样的输出？鉴于棋盘上可以有任何东西，或者根本没有东西，我们正在考虑以下可能性：

+   *游戏未开始*

+   *游戏正在进行中*

+   `X`赢了

+   `O`赢了

+   平局

看，输出很简单！现在，我们可以看到输入数据与这些可能性之间的关系。

# 输入数据是什么？

在这种情况下，输入数据在问题陈述中——我们的输入是一个有棋子的棋盘。但让我们看一些例子。最简单的例子是一个空棋盘：

```cpp
_ _ _ 
_ _ _ 
_ _ _
```

为了清晰起见，我们使用`_`来表示棋盘上的空格。

当然，空白的棋盘对应于“游戏未开始”的输出。

这足够简单了。现在，让我们看一个上面有几步的例子：

```cpp
X _ _    
O _ _ 
_ _ _
```

`X`和`O`都已经走了他们的步子，但游戏仍在进行中。我们可以提供许多*进行中的游戏*的例子：

```cpp
X X _ 
O _ _ 
_ _ _
```

这是另一个例子：

```cpp
X X O 
O _ _ 
_ _ _
```

有一些例子在井字棋游戏中永远不会发生，比如这个：

```cpp
X X _ 
O X _ 
X _ _
```

在这种情况下，`X`已经走了四步，而`O`只走了一步，这是井字棋规则不允许的。我们现在将忽略这种情况，只返回一个*进行中的游戏*。不过，一旦我们完成了代码的其余部分，你可以自己实现这个算法。

让我们看一个`X`赢得的游戏：

```cpp
X X X 
O O _ 
_ _ _
```

`X`赢了，因为第一行被填满了。`X`还有其他赢的方式吗？是的，在一列上：

```cpp
X _ _ 
X O O 
X _ _
```

它也可以在主对角线上获胜：

```cpp
X O _ 
O X _ 
_ _ X
```

这是`X`在次对角线上的胜利：

```cpp
_ O X 
O X _ 
X _ _
```

同样地，我们有`O`通过填充一条线获胜的例子：

```cpp
X X _ 
O O O 
X _ _
```

这是通过填充一列获胜的情况：

```cpp
X O _ 
X O X 
_ O _
```

这是`O`在主对角线上的胜利：

```cpp
O X _ 
_ O X 
X _ O
```

这是通过次对角线获胜的情况：

```cpp
X X O 
_ O X 
O _ _
```

那么，怎么样才能结束成为平局呢？很简单——所有的方格都被填满了，但没有赢家：

```cpp
X X O 
O X X 
X O O
```

我们已经看过了所有可能的输出的例子。现在是时候看看数据转换了。

# 数据转换

我们如何将输入转换为输出？为了做到这一点，我们将不得不选择一个可能的输出来先解决。现在最容易的是`X`获胜的情况。那么，`X`怎么赢？

根据游戏规则，如果棋盘上的一条线、一列或一条对角线被`X`填满，`X`就赢了。让我们写下所有可能的情况。如果发生以下任何一种情况，`X`就赢了：

+   任何一条线都被`X`填满了，或者

+   任何一列都被`X`填满，或者

+   主对角线被`X`填满，或者

+   次对角线被`X`填满了。

为了实现这一点，我们需要一些东西：

+   从棋盘上得到所有的线。

+   从棋盘上得到所有的列。

+   从棋盘上得到主对角线和次对角线。

+   如果它们中的任何一个被`X`填满了，`X`就赢了！

我们可以用另一种方式来写这个：

```cpp
board -> collection(all lines, all columns, all diagonals) -> any(collection, filledWithX) -> X won
```

`filledWithX`是什么意思？让我们举个例子；我们正在寻找这样的线：

```cpp
X X X
```

我们不是在寻找`X O X`或`X _ X`这样的线。

听起来我们正在检查一条线、一列或一条对角线上的所有标记是否都是`'X'`。让我们将这个检查视为一个转换：

```cpp
line | column | diagonal -> all tokens equal X -> line | column | diagonal filled with X
```

因此，我们的转换集合变成了这样：

```cpp
board -> collection(all lines, all columns, all diagonals) -> if any(collection, filledWithX) -> X won 

filledWithX(line|column|diagonal L) = all(token on L equals 'X')
```

还有一个问题——我们如何得到线、列和对角线？我们可以分别看待这个问题，就像我们看待大问题一样。我们的输入肯定是棋盘。我们的输出是由第一行、第二行和第三行、第一列、第二列和第三列、主对角线和次对角线组成的列表。

下一个问题是，什么定义了一条线？嗯，我们知道如何得到第一条线——我们使用`[0, 0]`，`[0, 1]`和`[0, 2]`坐标。第二条线有`[1, 0]`，`[1, 1]`和`[1, 2]`坐标。列呢？嗯，第一列有`[1, 0]`，`[1, 1]`和`[2, 1]`坐标。而且，正如我们将看到的，对角线也是由特定的坐标集定义的。

那么，我们学到了什么？我们学到了为了得到线、列和对角线，我们需要以下的转换：

```cpp
board -> collection of coordinates for lines, columns, diagonals -> apply coordinates to the board -> obtain list of elements for lines, columns, and diagonals
```

这就结束了我们的分析。现在是时候转向实现了。所有之前的转换都可以通过使用函数式构造来用代码表达。事实上，一些转换是如此常见，以至于它们已经在标准库中实现了。让我们看看我们如何可以使用它们！

# 使用`all_of`来判断是否被`X`填满

我们将要看的第一个转换是`all_of`。给定一个集合和一个返回布尔值的函数（也称为**逻辑谓词**），`all_of`将谓词应用于集合的每个元素，并返回结果的逻辑与。让我们看一些例子：

```cpp
auto trueForAll = [](auto x) { return true; };
auto falseForAll = [](auto x) { return false; };
auto equalsChara = [](auto x){ return x == 'a';};
auto notChard = [](auto x){ return x != 'd';};

TEST_CASE("all_of"){
    vector<char> abc{'a', 'b', 'c'};

    CHECK(all_of(abc.begin(), abc.end(), trueForAll));
    CHECK(!all_of(abc.begin(), abc.end(), falseForAll));
    CHECK(!all_of(abc.begin(), abc.end(), equalsChara));
    CHECK(all_of(abc.begin(), abc.end(), notChard));
}
```

`all_of`函数接受两个定义范围开始和结束的迭代器和一个谓词作为参数。当你想将转换应用于集合的子集时，迭代器是有用的。由于我通常在整个集合上使用它，我发现反复写`collection.begin()`和`collection.end()`很烦人。因此，我实现了自己简化的`all_of_collection`版本，它接受整个集合并处理其余部分：

```cpp
auto all_of_collection = [](const auto& collection, auto lambda){
    return all_of(collection.begin(), collection.end(), lambda);
};

TEST_CASE("all_of_collection"){
    vector<char> abc{'a', 'b', 'c'};

    CHECK(all_of_collection(abc, trueForAll));
    CHECK(!all_of_collection(abc, falseForAll));
    CHECK(!all_of_collection(abc, equalsChara));
    CHECK(all_of_collection(abc, notChard));
}
```

知道这个转换后，编写我们的`lineFilledWithX`函数很容易-我们将标记的集合转换为指定标记是否为`X`的布尔值的集合：

```cpp
auto lineFilledWithX = [](const auto& line){
    return all_of_collection(line, [](const auto& token){ return token == 'X';});
};

TEST_CASE("Line filled with X"){
    vector<char> line{'X', 'X', 'X'};

    CHECK(lineFilledWithX(line));
}
```

就是这样！我们可以确定我们的线是否填满了`X`。

在我们继续之前，让我们做一些简单的调整。首先，通过为我们的`vector<char>`类型命名来使代码更清晰：

```cpp
using Line = vector<char>;
```

然后，让我们检查代码是否对负面情况也能正常工作。如果`Line`没有填满`X`标记，`lineFilledWithX`应该返回`false`：

```cpp
TEST_CASE("Line not filled with X"){
    CHECK(!lineFilledWithX(Line{'X', 'O', 'X'}));
    CHECK(!lineFilledWithX(Line{'X', ' ', 'X'}));
}
```

最后，一个敏锐的读者会注意到我们需要相同的函数来满足`O`获胜的条件。我们现在知道如何做到这一点-记住参数绑定的力量。我们只需要提取一个`lineFilledWith`函数，并通过将`tokenToCheck`参数绑定到`X`和`O`标记值，分别获得`lineFilledWithX`和`lineFilledWithO`函数：

```cpp
auto lineFilledWith = [](const auto line, const auto tokenToCheck){
    return all_of_collection(line, &tokenToCheck{  
        return token == tokenToCheck;});
};

auto lineFilledWithX = bind(lineFilledWith, _1, 'X'); 
auto lineFilledWithO = bind(lineFilledWith, _1, 'O');
```

让我们回顾一下-我们有一个`Line`数据结构，我们有一个可以检查该行是否填满`X`或`O`的函数。我们使用`all_of`函数来为我们做繁重的工作；我们只需要定义我们的井字棋线的逻辑。

是时候继续前进了。我们需要将我们的棋盘转换为线的集合，由三条线、三列和两条对角线组成。为此，我们需要使用另一个函数式转换`map`，它在 STL 中实现为`transform`函数。

# 使用 map/transform

现在我们需要编写一个将棋盘转换为线、列和对角线列表的函数；因此，我们可以使用一个将集合转换为另一个集合的转换。这种转换通常在函数式编程中称为`map`，在 STL 中实现为`transform`。为了理解它，我们将使用一个简单的例子；给定一个字符向量，让我们用`'a'`替换每个字符：

```cpp
TEST_CASE("transform"){
    vector<char> abc{'a', 'b', 'c'};

// Not the best version, see below
vector<char> aaa(3);
transform(abc.begin(), abc.end(), aaa.begin(), [](auto element){return 
    'a';});
CHECK_EQ(vector<char>{'a', 'a', 'a'}, aaa);
}
```

虽然它有效，但前面的代码示例是天真的，因为它用稍后被覆盖的值初始化了`aaa`向量。我们可以通过首先在`aaa`向量中保留`3`个元素，然后使用`back_inserter`来避免这个问题，这样`transform`就会自动在`aaa`向量上调用`push_back`：

```cpp
TEST_CASE("transform-fixed") { 
    const auto abc = vector{'a', 'b', 'c'}; 
    vector<char> aaa; 
    aaa.reserve(abc.size()); 
    transform(abc.begin(), abc.end(), back_inserter(aaa), 
            [](const char elem) { return 'a'; }
    ); 
    CHECK_EQ(vector{'a', 'a', 'a'}, aaa); 
}
```

如你所见，`transform`基于迭代器，就像`all_of`一样。到目前为止，你可能已经注意到我喜欢保持事情简单，专注于我们要完成的任务。没有必要一直写这些；相反，我们可以实现我们自己的简化版本，它可以在整个集合上工作，并处理围绕此函数的所有仪式。

# 简化转换

让我们尝试以最简单的方式实现`transform_all`函数：

```cpp
auto transform_all = [](auto const source, auto lambda){
    auto destination; // Compilation error: the type is not defined
    ...
}
```

不幸的是，当我们尝试以这种方式实现它时，我们需要一个目标集合的类型。这样做的自然方式是使用 C++模板并传递`Destination`类型参数：

```cpp
template<typename Destination>
auto transformAll = [](auto const source,  auto lambda){
    Destination result;
    result.reserve(source.size());
    transform(source.begin(), source.end(), back_inserter(result), 
        lambda);
    return result;
};

```

这对于任何具有`push_back`函数的集合都有效。一个很好的副作用是，我们可以用它来连接`string`中的结果字符：

```cpp
auto turnAllToa = [](auto x) { return 'a';};

TEST_CASE("transform all"){
    vector abc{'a', 'b', 'c'};

    CHECK_EQ(vector<char>({'a', 'a', 'a'}), transform_all<vector<char>>
        (abc, turnAllToa));
    CHECK_EQ("aaa", transform_all<string>(abc,turnAllToa));
}
```

使用`transform_all`与`string`允许我们做一些事情，比如将小写字符转换为大写字符：

```cpp
auto makeCaps = [](auto x) { return toupper(x);};

TEST_CASE("transform all"){
    vector<char> abc = {'a', 'b', 'c'};

    CHECK_EQ("ABC", transform_all<string>(abc, makeCaps));
}
```

但这还不是全部-输出类型不一定要与输入相同：

```cpp
auto toNumber = [](auto x) { return (int)x - 'a' + 1;};

TEST_CASE("transform all"){
    vector<char> abc = {'a', 'b', 'c'};
    vector<int> expected = {1, 2, 3};

    CHECK_EQ(expected, transform_all<vector<int>>(abc, toNumber));
}
```

因此，`transform`函数在我们需要将一个集合转换为另一个集合时非常有用，无论是相同类型还是不同类型。在`back_inserter`的支持下，它还可以用于`string`输出，从而实现对任何类型集合的字符串表示的实现。

我们现在知道如何使用 transform 了。所以，让我们回到我们的问题。

# 我们的坐标

我们的转换从计算坐标开始。因此，让我们首先定义它们。STL `pair`类型是坐标的简单表示：

```cpp
using Coordinate = pair<int, int>;
```

# 从板和坐标获取一条线

假设我们已经为一条线、一列或一条对角线构建了坐标列表，我们需要将令牌的集合转换为`Line`参数。这很容易通过我们的`transformAll`函数完成：

```cpp
auto accessAtCoordinates = [](const auto& board, const Coordinate&  
    coordinate){
        return board[coordinate.first][coordinate.second];
};

auto projectCoordinates = [](const auto& board, const auto&  
    coordinates){
        auto boardElementFromCoordinates = bind(accessAtCoordinates,  
        board, _1);
        return transform_all<Line>(coordinates,  
            boardElementFromCoordinates);
};
```

`projectCoordinates` lambda 接受板和坐标列表，并返回与这些坐标对应的板元素列表。我们在坐标列表上使用`transformAll`，并使用一个接受两个参数的转换——`board`参数和`coordinate`参数。然而，`transformAll`需要一个带有单个参数的 lambda，即`Coordinate`值。因此，我们必须要么捕获板的值，要么使用部分应用。

现在我们只需要构建我们的线、列和对角线的坐标列表了！

# 从板上得到一条线

我们可以通过使用前一个函数`projectCoordinates`轻松地从板上得到一条线：

```cpp
auto line = [](auto board, int lineIndex){
   return projectCoordinates(board, lineCoordinates(board, lineIndex));
};
```

`line` lambda 接受`board`和`lineIndex`，构建线坐标列表，并使用`projectCoordinates`返回线。

那么，我们如何构建线坐标？嗯，由于我们有`lineIndex`和`Coordinate`作为一对，我们需要在`(lineIndex, 0)`、`(lineIndex, 1)`和`(lineIndex, 2)`上调用`make_pair`。这看起来也像是一个`transform`调用；输入是一个`{0, 1, 2}`集合，转换是`make_pair(lineIndex, index)`。让我们写一下：

```cpp
auto lineCoordinates = [](const auto board, auto lineIndex){
    vector<int> range{0, 1, 2};
    return transformAll<vector<Coordinate>>(range, lineIndex{return make_pair(lineIndex, index);});
};
```

# 范围

但是`{0, 1, 2}`是什么？在其他编程语言中，我们可以使用范围的概念；例如，在 Groovy 中，我们可以编写以下内容：

```cpp
def range = [0..board.size()]
```

范围非常有用，并且已经在 C++ 20 标准中被采用。我们将在第十四章中讨论它们，*使用 Ranges 库进行惰性求值*。在那之前，我们将编写我们自己的`toRange`函数：

```cpp
auto toRange = [](auto const collection){
    vector<int> range(collection.size());
    iota(begin(range), end(range), 0);
    return range;
};
```

`toRange`接受一个集合作为输入，并从`0`到`collection.size()`创建`range`。因此，让我们在我们的代码中使用它：

```cpp
using Board = vector<Line>;
using Line = vector<char>;

auto lineCoordinates = [](const auto board, auto lineIndex){
    auto range = toRange(board);
    return transform_all<vector<Coordinate>>(range, lineIndex{return make_pair(lineIndex, index);});
};

TEST_CASE("lines"){
    Board board {
        {'X', 'X', 'X'},
        {' ', 'O', ' '},
        {' ', ' ', 'O'}
    };

    Line expectedLine0 = {'X', 'X', 'X'};
    CHECK_EQ(expectedLine0, line(board, 0));
    Line expectedLine1 = {' ', 'O', ' '};
    CHECK_EQ(expectedLine1, line(board, 1));
    Line expectedLine2 = {' ', ' ', 'O'};
    CHECK_EQ(expectedLine2, line(board, 2));
}
```

我们已经把所有元素都放在了正确的位置，所以现在是时候看看列了。

# 获取列

获取列的代码与获取线的代码非常相似，只是我们保留`columnIndex`而不是`lineIndex`。我们只需要将其作为参数传递：

```cpp
auto columnCoordinates = [](const auto& board, const auto columnIndex){
    auto range = toRange(board);
    return transformAll<vector<Coordinate>>(range, columnIndex{return make_pair(index, columnIndex);});
};

auto column = [](auto board, auto columnIndex){
    return projectCoordinates(board, columnCoordinates(board,  
        columnIndex));
};

TEST_CASE("all columns"){
    Board board{
        {'X', 'X', 'X'},
        {' ', 'O', ' '},
        {' ', ' ', 'O'}
    };

    Line expectedColumn0{'X', ' ', ' '};
    CHECK_EQ(expectedColumn0, column(board, 0));
    Line expectedColumn1{'X', 'O', ' '};
    CHECK_EQ(expectedColumn1, column(board, 1));
    Line expectedColumn2{'X', ' ', 'O'};
    CHECK_EQ(expectedColumn2, column(board, 2));
}
```

这不是很酷吗？通过几个函数和标准的函数变换，我们可以在我们的代码中构建复杂的行为。现在对角线变得轻而易举了。

# 获取对角线

主对角线由相等的行和列坐标定义。使用与之前相同的机制读取它非常容易；我们构建相等索引的对，并将它们传递给`projectCoordinates`函数：

```cpp
auto mainDiagonalCoordinates = [](const auto board){
    auto range = toRange(board);
    return transformAll<vector<Coordinate>>(range, [](auto index) 
       {return make_pair(index, index);});
};
auto mainDiagonal = [](const auto board){
    return projectCoordinates(board, mainDiagonalCoordinates(board));
};

TEST_CASE("main diagonal"){
    Board board{
        {'X', 'X', 'X'},
        {' ', 'O', ' '},
        {' ', ' ', 'O'}
    };

    Line expectedDiagonal = {'X', 'O', 'O'};

    CHECK_EQ(expectedDiagonal, mainDiagonal(board));
}
```

那么对于次对角线呢？嗯，坐标的总和总是等于`board`参数的大小。在 C++中，我们还需要考虑基于 0 的索引，因此在构建坐标列表时，我们需要通过`1`进行适当的调整：

```cpp
auto secondaryDiagonalCoordinates = [](const auto board){
    auto range = toRange(board);
    return transformAll<vector<Coordinate>>(range, board 
        {return make_pair(index, board.size() - index - 1);});
};

auto secondaryDiagonal = [](const auto board){
    return projectCoordinates(board, 
        secondaryDiagonalCoordinates(board));
};

TEST_CASE("secondary diagonal"){
    Board board{
        {'X', 'X', 'X'},
        {' ', 'O', ' '},
        {' ', ' ', 'O'}
    };

    Line expectedDiagonal{'X', 'O', ' '};

    CHECK_EQ(expectedDiagonal, secondaryDiagonal(board));
}
```

# 获取所有线、所有列和所有对角线

说到这一点，我们现在可以构建所有线、列和对角线的集合了。有多种方法可以做到这一点；因为我要写一个以函数式风格编写的通用解决方案，我将再次使用`transform`。我们需要将`(0..board.size())`范围转换为相应的线列表和列列表。然后，我们需要返回一个包含主对角线和次对角线的集合：

```cpp
typedef vector<Line> Lines;

auto allLines = [](auto board) {
    auto range = toRange(board);
    return transform_all<Lines>(range, board { return 
        line(board, index);});
};

auto allColumns = [](auto board) {
    auto range = toRange(board);
    return transform_all<Lines>(range, board { return 
        column(board, index);});
};

auto allDiagonals = [](auto board) -> Lines {
    return {mainDiagonal(board), secondaryDiagonal(board)};
};
```

我们只需要一件事情——一种连接这三个集合的方法。由于向量没有实现这个功能，推荐的解决方案是使用`insert`和`move_iterator`，从而将第二个集合的项目移动到第一个集合的末尾：

```cpp
auto concatenate = [](auto first, const auto second){
    auto result(first);
    result.insert(result.end(), make_move_iterator(second.begin()), 
        make_move_iterator(second.end()));
    return result;
};

```

然后，我们只需将这三个集合合并为两个步骤：

```cpp
auto concatenate3 = [](auto first, auto const second, auto const third){
    return concatenate(concatenate(first, second), third);
};
```

现在我们可以从棋盘中获取所有行、列和对角线的完整列表，就像你在下面的测试中看到的那样：

```cpp
auto allLinesColumnsAndDiagonals = [](const auto board) {
    return concatenate3(allLines(board), allColumns(board),  
        allDiagonals(board));
};

TEST_CASE("all lines, columns and diagonals"){
    Board board {
        {'X', 'X', 'X'},
        {' ', 'O', ' '},
        {' ', ' ', 'O'}
    };

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

    auto all = allLinesColumnsAndDiagonals(board);
    CHECK_EQ(expected, all);
}
```

在找出`X`是否获胜的最后一步中只剩下一个任务。我们有所有行、列和对角线的列表。我们知道如何检查一行是否被`X`填满。我们只需要检查列表中的任何一行是否被`X`填满。

# 使用 any_of 来检查 X 是否获胜

类似于`all_of`，另一个函数构造帮助我们在集合上应用的谓词之间表达 OR 条件。在 STL 中，这个构造是在`any_of`函数中实现的。让我们看看它的作用：

```cpp
TEST_CASE("any_of"){
    vector<char> abc = {'a', 'b', 'c'};

    CHECK(any_of(abc.begin(), abc.end(), trueForAll));
    CHECK(!any_of(abc.begin(), abc.end(), falseForAll));
    CHECK(any_of(abc.begin(), abc.end(), equalsChara));
    CHECK(any_of(abc.begin(), abc.end(), notChard));
}
```

像我们在本章中看到的其他高级函数一样，它使用迭代器作为集合的开始和结束。像往常一样，我喜欢保持简单；因为我通常在完整集合上使用`any_of`，我喜欢实现我的辅助函数：

```cpp
auto any_of_collection = [](const auto& collection, const auto& fn){
 return any_of(collection.begin(), collection.end(), fn);
};

TEST_CASE("any_of_collection"){
    vector<char> abc = {'a', 'b', 'c'};

    CHECK(any_of_collection(abc, trueForAll));
    CHECK(!any_of_collection(abc, falseForAll));
    CHECK(any_of_collection(abc, equalsChara));
    CHECK(any_of_collection(abc, notChard));
}
```

我们只需要在我们的列表上使用它来检查`X`是否是赢家：

```cpp
auto xWins = [](const auto& board){
    return any_of_collection(allLinesColumnsAndDiagonals(board), 
        lineFilledWithX);
};

TEST_CASE("X wins"){
    Board board{
        {'X', 'X', 'X'},
        {' ', 'O', ' '},
        {' ', ' ', 'O'}
    };

    CHECK(xWins(board));
}
```

这就结束了我们对`X`获胜条件的解决方案。在我们继续之前，能够在控制台上显示棋盘将是很好的。现在是使用`map`/`transform`的近亲——`reduce`的时候了，或者在 STL 中被称为`accumulate`。

# 使用 reduce/accumulate 来显示棋盘

我们想在控制台上显示棋盘。通常，我们会使用可变函数，比如`cout`来做到这一点；然而，记住我们讨论过，虽然我们需要保持程序的某些部分可变，比如调用`cout`的部分，但我们应该将它们限制在最小范围内。那么，替代方案是什么呢？嗯，我们需要再次考虑输入和输出——我们想要编写一个以`board`作为输入并返回`string`表示的函数，我们可以通过使用可变函数，比如`cout`来显示它。让我们以测试的形式写出我们想要的：

```cpp
TEST_CASE("board to string"){
    Board board{
        {'X', 'X', 'X'},
        {' ', 'O', ' '},
        {' ', ' ', 'O'}
    };
    string expected = "XXX\n O \n  O\n";

    CHECK_EQ(expected, boardToString(board));
}
```

为了获得这个结果，我们首先需要将`board`中的每一行转换为它的`string`表示。我们的行是`vector<char>`，我们需要将它转换为`string`；虽然有很多方法可以做到这一点，但请允许我使用带有`string`输出的`transformAll`函数：

```cpp
auto lineToString = [](const auto& line){
    return transformAll<string>(line, [](const auto token) -> char { 
        return token;});
};

TEST_CASE("line to string"){
    Line line {
        ' ', 'X', 'O'
    };

    CHECK_EQ(" XO", lineToString(line));
}
```

有了这个函数，我们可以轻松地将一个棋盘转换为`vector<string>`：

```cpp
auto boardToLinesString = [](const auto board){
    return transformAll<vector<string>>(board, lineToString);
};

TEST_CASE("board to lines string"){
    Board board{
        {'X', 'X', 'X'},
        {' ', 'O', ' '},
        {' ', ' ', 'O'}
    };
    vector<string> expected{
        "XXX",
        " O ",
        "  O"
    };

    CHECK_EQ(expected, boardToLinesString(board));
}
```

最后一步是用`\n`将这些字符串组合起来。我们经常需要以各种方式组合集合的元素；这就是`reduce`发挥作用的地方。在函数式编程中，`reduce`是一个接受集合、初始值（例如，空的`strings`）和累积函数的操作。该函数接受两个参数，对它们执行操作，并返回一个新值。

让我们看几个例子。首先是添加一个数字向量的经典例子：

```cpp
TEST_CASE("accumulate"){
    vector<int> values = {1, 12, 23, 45};

    auto add = [](int first, int second){return first + second;};
    int result = accumulate(values.begin(), values.end(), 0, add);
    CHECK_EQ(1 + 12 + 23 + 45, result);
}
```

以下向我们展示了如果需要添加具有初始值的向量应该怎么做：

```cpp
    int resultWithInit100 = accumulate(values.begin(), values.end(),  
        100, add);
    CHECK_EQ(1oo + 1 + 12 + 23 + 45, resultWithInit100);
```

同样，我们可以连接`strings`：

```cpp
    vector<string> strings {"Alex", "is", "here"};
    auto concatenate = [](const string& first, const string& second) ->  
        string{
        return first + second;
    };
    string concatenated = accumulate(strings.begin(), strings.end(),  
        string(), concatenate);
    CHECK_EQ("Alexishere", concatenated);
```

或者，我们可以添加一个前缀：

```cpp
    string concatenatedWithPrefix = accumulate(strings.begin(),  
        strings.end(), string("Pre_"), concatenate);
    CHECK_EQ("Pre_Alexishere", concatenatedWithPrefix);
```

像我们在整个集合上使用默认值作为初始值的简化实现一样，我更喜欢使用`decltype`魔术来实现它：

```cpp
auto accumulateAll = [](auto source, auto lambda){
    return accumulate(source.begin(), source.end(), typename  
        decltype(source)::value_type(), lambda);
};
```

这只留下了我们的最后一个任务——编写一个连接`string`行的实现，使用换行符：

```cpp
auto boardToString = [](const auto board){
    auto linesAsString = boardToLinesString(board);
    return accumulateAll(linesAsString, 
        [](string current, string lineAsString) { return current + lineAsString + "\n"; }
    );
};
TEST_CASE("board to string"){
    Board board{
        {'X', 'X', 'X'},
        {' ', 'O', ' '},
        {' ', ' ', 'O'}
    };
    string expected = "XXX\n O \n  O\n";

    CHECK_EQ(expected, boardToString(board));
}
```

现在我们可以使用`cout << boardToString`来显示我们的棋盘。再次，我们使用了一些函数变换和非常少的自定义代码来将一切整合在一起。这非常好。

`map`/`reduce`组合，或者在 STL 中被称为`transform`/`accumulate`，是功能性编程中非常强大且非常常见的。我们经常需要从一个集合开始，多次将其转换为另一个集合，然后再组合集合的元素。这是一个如此强大的概念，以至于它是大数据分析的核心，使用诸如 Apache Hadoop 之类的工具，尽管在机器级别上进行了扩展。这表明，通过掌握这些转换，您可能最终会在意想不到的情况下应用它们，使自己成为一个不可或缺的问题解决者。很酷，不是吗？

# 使用`find_if`来显示特定的赢的细节

我们现在很高兴，因为我们已经解决了`X`的井字游戏结果问题。然而，正如总是一样，需求会发生变化；我们现在不仅需要说`X`是否赢了，还需要说赢了在哪里——在哪一行、或列、或对角线。

幸运的是，我们已经有了大部分元素。由于它们都是非常小的函数，我们只需要以一种有助于我们的方式重新组合它们。让我们再次从数据的角度思考——我们的输入数据现在是一组行、列和对角线；我们的结果应该是类似于`X`赢*在第一行*的信息。我们只需要增强我们的数据结构，以包含有关每行的信息；让我们使用`map`：

```cpp
    map<string, Line> linesWithDescription{
        {"first line", line(board, 0)},
        {"second line", line(board, 1)},
        {"last line", line(board, 2)},
        {"first column", column(board, 0)},
        {"second column", column(board, 1)},
        {"last column", column(board, 2)},
        {"main diagonal", mainDiagonal(board)},
        {"secondary diagonal", secondaryDiagonal(board)},
    };
```

我们知道如何找出`X`是如何赢的——通过我们的`lineFilledWithX`谓词函数。现在，我们只需要在地图中搜索符合`lineFilledWithX`谓词的行，并返回相应的消息。

这是功能性编程中的一个常见操作。在 STL 中，它是用`find_if`函数实现的。让我们看看它的运行情况：

```cpp
auto equals1 = [](auto value){ return value == 1; };
auto greaterThan11 = [](auto value) { return value > 11; };
auto greaterThan50 = [](auto value) { return value > 50; };

TEST_CASE("find if"){
    vector<int> values{1, 12, 23, 45};

    auto result1 = find_if(values.begin(), values.end(), equals1);
    CHECK_EQ(*result1, 1);

    auto result12 = find_if(values.begin(), values.end(), 
        greaterThan11);
    CHECK_EQ(*result12, 12);

    auto resultNotFound = find_if(values.begin(), values.end(), 
        greaterThan50);
    CHECK_EQ(resultNotFound, values.end());
}
```

`find_if`根据谓词在集合中查找并返回结果的指针，如果找不到任何内容，则返回指向`end()`迭代器的指针。

像往常一样，让我们实现一个允许在整个集合中搜索的包装器。我们需要以某种方式表示`not found`的值；幸运的是，我们可以使用 STL 中的可选类型：

```cpp
auto findInCollection = [](const auto& collection, auto fn){
    auto result = find_if(collection.begin(), collection.end(), fn);
    return (result == collection.end()) ? nullopt : optional(*result);
};

TEST_CASE("find in collection"){
    vector<int> values {1, 12, 23, 45};

    auto result1 = findInCollection(values, equals1);
    CHECK_EQ(result1, 1);

    auto result12 = findInCollection(values, greaterThan11);
    CHECK_EQ(result12, 12);

    auto resultNotFound = findInCollection(values, greaterThan50);
    CHECK(!resultNotFound.has_value());
}
```

现在，我们可以轻松实现新的要求。我们可以使用我们新实现的`findInCollection`函数找到被`X`填满的行，并返回相应的描述。因此，我们可以告诉用户`X`是如何赢的——是在一行、一列还是对角线上：

```cpp
auto howDidXWin = [](const auto& board){
    map<string, Line> linesWithDescription = {
        {"first line", line(board, 0)},
        {"second line", line(board, 1)},
        {"last line", line(board, 2)},
        {"first column", column(board, 0)},
        {"second column", column(board, 1)},
        {"last column", column(board, 2)},
        {"main diagonal", mainDiagonal(board)},
        {"secondary diagonal", secondaryDiagonal(board)},
    };
    auto found = findInCollection(linesWithDescription,[](auto value) 
        {return lineFilledWithX(value.second);}); 
    return found.has_value() ? found->first : "X did not win";
};
```

当然，我们应该从棋盘生成地图，而不是硬编码。我将把这个练习留给读者；只需再次使用我们最喜欢的`transform`函数即可。

# 完成我们的解决方案

虽然我们已经为`X`赢实现了解决方案，但现在我们需要研究其他可能的输出。让我们先来看最简单的一个——`O`赢。

# 检查`O`是否赢了

检查`O`是否赢很容易——我们只需要在我们的函数中做一个小改变。我们需要一个新函数`oWins`，它检查任何一行、一列或对角线是否被`O`填满：

```cpp
auto oWins = [](auto const board){
    return any_of_collection(allLinesColumnsAndDiagonals(board),  
        lineFilledWithO);
};
TEST_CASE("O wins"){
    Board board = {
        {'X', 'O', 'X'},
        {' ', 'O', ' '},
        {' ', 'O', 'X'}
    };

    CHECK(oWins(board));
}
```

我们使用与`xWins`相同的实现，只是在作为参数传递的 lambda 中稍作修改。

# 使用`none_of`检查平局

那么`平局`呢？嗯，当`board`参数已满且既没有`X`也没有`O`赢时，就会出现平局：

```cpp
auto draw = [](const auto& board){
    return full(board) && !xWins(board) && !oWins(board); 
};

TEST_CASE("draw"){
    Board board {
        {'X', 'O', 'X'},
        {'O', 'O', 'X'},
        {'X', 'X', 'O'}
    };

    CHECK(draw(board));
}
```

满棋盘意味着每一行都已满：

```cpp
auto full = [](const auto& board){
    return all_of_collection(board, fullLine);
};
```

那么我们如何知道一行是否已满？嗯，我们知道如果行中的任何一个标记都不是空（`' '`）标记，那么该行就是满的。正如您现在可能期望的那样，STL 中有一个名为`none_of`的函数，可以为我们检查这一点：

```cpp
auto noneOf = [](const auto& collection, auto fn){
    return none_of(collection.begin(), collection.end(), fn);
};

auto isEmpty = [](const auto token){return token == ' ';};
auto fullLine = [](const auto& line){
    return noneOf(line, isEmpty);
};
```

# 检查游戏是否正在进行中

最后一种情况是游戏仍在进行中。最简单的方法就是检查游戏是否没有赢，且棋盘还没有满：

```cpp
auto inProgress = [](const auto& board){
    return !full(board) && !xWins(board) && !oWins(board); 
};
TEST_CASE("in progress"){
    Board board {
        {'X', 'O', 'X'},
        {'O', ' ', 'X'},
        {'X', 'X', 'O'}
    };

    CHECK(inProgress(board));
}
```

恭喜，我们做到了！我们使用了许多功能转换来实现了井字游戏结果问题；还有我们自己的一些 lambda。但更重要的是，我们学会了如何开始像一个功能性程序员一样思考——清晰地定义输入数据，清晰地定义输出数据，并找出可以将输入数据转换为所需输出数据的转换。

# 使用可选类型进行错误管理

到目前为止，我们已经用函数式风格编写了一个小程序。但是错误情况怎么处理呢？

显然，我们仍然可以使用 C++机制——返回值或异常。但是函数式编程还可以看作另一种方式——将错误视为数据。

我们在实现`find_if`包装器时已经看到了这种技术的一个例子：

```cpp
auto findInCollection = [](const auto& collection, auto fn){
    auto result = find_if(collection.begin(), collection.end(), fn);
    return (result == collection.end()) ? nullopt : optional(*result);
};
```

我们使用了`optional`类型，而不是抛出异常或返回`collection.end()`，这是一个本地值。如其名称所示，optional 类型表示一个可能有值，也可能没有值的变量。可选值可以被初始化，可以使用底层类型支持的值，也可以使用`nullopt`——一个默认的非值，可以这么说。

当在我们的代码中遇到可选值时，我们需要考虑它，就像我们在检查`X`赢得函数中所做的那样：

```cpp
return found.has_value() ? found->first : "X did not win";
```

因此，“未找到”条件不是错误；相反，它是我们代码和数据的正常部分。事实上，处理这种情况的另一种方法是增强`findInCollection`，在未找到时返回指定的值：

```cpp
auto findInCollectionWithDefault = [](auto collection, auto 
    defaultResult, auto lambda){
        auto result = findInCollection(collection, lambda);
        return result.has_value() ? (*result) : defaultResult;
}; 
```

现在我们可以使用`findInCollectionWithDefault`来在`X`没有赢得情况下调用`howDidXWin`时获得一个`X 没有赢`的消息：

```cpp
auto howDidXWin = [](auto const board){
    map<string, Line> linesWithDescription = {
        {"first line", line(board, 0)},
        {"second line", line(board, 1)},
        {"last line", line(board, 2)},
        {"first column", column(board, 0)},
        {"second column", column(board, 1)},
        {"last column", column(board, 2)},
        {"main diagonal", mainDiagonal(board)},
        {"secondary diagonal", secondaryDiagonal(board)},
        {"diagonal", secondaryDiagonal(board)},
    };
    auto xDidNotWin = make_pair("X did not win", Line());
    auto xWon = [](auto value){
        return lineFilledWithX(value.second);
    };

    return findInCollectionWithDefault(linesWithDescription, xDidNotWin, xWon).first; 
};

TEST_CASE("X did not win"){
    Board board {
        {'X', 'X', ' '},
        {' ', 'O', ' '},
        {' ', ' ', 'O'}
    };

    CHECK_EQ("X did not win", howDidXWin(board));
}
```

我最好的建议是这样——对所有异常情况使用异常，并将其他所有情况作为数据结构的一部分。使用可选类型，或者带有默认值的转换。你会惊讶于错误管理变得多么容易和自然。

# 总结

在本章中，我们涵盖了很多内容！我们经历了一次发现之旅——我们首先列出了问题的输出和相应的输入，对它们进行了分解，并找出了如何将输入转换为所需的输出。我们看到了当需要新功能时，小函数和函数操作如何给我们带来灵活性。我们看到了如何使用`any`、`all`、`none`、`find_if`、`map`/`transform`和`reduce`/`accumulate`，以及如何使用可选类型或默认值来支持代码中的所有可能情况。

现在我们已经了解了如何以函数式风格编写代码，是时候在下一章中看看这种方法如何与面向对象编程结合了。
