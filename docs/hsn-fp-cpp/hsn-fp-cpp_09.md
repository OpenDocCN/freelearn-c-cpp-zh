# 第七章：使用函数操作消除重复

软件设计中的一个关键原则是减少代码重复。函数式构造通过柯里化和函数组合提供了额外的机会来减少代码重复。

本章将涵盖以下主题：

+   如何以及为什么避免重复代码

+   如何识别代码相似性

+   使用柯里化来消除某些类型的代码相似性

+   使用组合来消除某些类型的代码相似性

+   使用 lambda 表达式或组合来消除某些类型的代码相似性

# 技术要求

你需要一个支持 C++ 17 的编译器。我使用的是 GCC 7.3.0。

该代码可以在 GitHub 上找到，网址为[`github.com/PacktPublishing/Hands-On-Functional-Programming-with-Cpp`](https://github.com/PacktPublishing/Hands-On-Functional-Programming-with-Cpp)，在`Chapter07`文件夹中。它包括并使用了`doctest`，这是一个单头文件的开源单元测试库。你可以在它的 GitHub 仓库上找到它，网址为[`github.com/onqtam/doctest`](https://github.com/onqtam/doctest)。

# 使用函数操作来消除重复

长时间维护代码时，只需要在一个地方更改代码，以及可以重新组合现有的代码片段，会更加容易。朝着这个理想的最有效方法之一是识别并消除代码中的重复。函数式编程的操作——部分应用、柯里化和函数组合——提供了许多机会，使代码更清晰，重复更少。

但首先，让我们了解重复是什么，以及为什么我们需要减少它。首先，我们将看看**不要重复自己**（DRY）原则，然后看看重复和代码相似性之间的关系。最后，我们将看看如何消除代码相似性。

# DRY 原则

软件开发中核心书籍的数量出乎意料地少。当然，有很多关于细节和帮助人们更好地理解这些想法的书籍，但是关于核心思想的书籍却非常少而且陈旧。能够列入核心书籍名单对作者来说是一种荣誉，也是该主题极其重要的一个暗示。许多程序员会把《程序员修炼之道》（Andrew Hunt 和 David Thomas 合著，1999 年出版）列入这样的书单。这本书详细介绍了一个原则，对于长期从事大型代码库工作的人来说非常有意义——DRY 原则。

在核心，DRY 原则是基于代码是存储知识的理解。每个函数和每个数据成员都代表了对问题的知识。理想情况下，我们希望避免在系统中重复存储知识。换句话说，无论你在找什么，它都应该只存在于一个地方。不幸的是，大多数代码库都是**WET**（写两遍、我们喜欢打字或浪费每个人的时间的缩写），而不是 DRY。

然而，消除重复的想法是很久以前就有的。肯特·贝克在 1990 年代曾提到过，作为**极限编程**（XP）实践的一部分。肯特·贝克描述了简单设计的四个要素，这是一种获得或改进软件设计的思维工具。

简单的设计意味着它做了以下事情：

+   通过了测试

+   揭示意图

+   减少重复

+   元素更少

我从 J.B. Rainsberger 那里学到了这些规则，他也致力于简化这些规则。他教会我，在大多数情况下，专注于三件事就足够了——测试代码、改进命名和减少重复。

但这并不是唯一提到消除重复的地方。这个原则以各种方式出现在 Unix 设计哲学中，在领域驱动设计（DDD）技术中，作为测试驱动开发（TDD）实践的帮助，以及许多其他方面。可以说这是一个良好软件设计的普遍原则，每当我们谈论模块内部代码的结构时，使用它是有意义的。

# 重复和相似

在我迈向学习良好软件设计的旅程中，我意识到术语“重复”对于表达我们试图实现的哲学非常有用，但很难理解如何将其付诸实践。我找到了一个更好的名字，用于描述我在尝试改进设计时寻找的东西——我寻找“代码相似之处”。一旦我找到相似之处，我会问它们是否显示了更深层次的重复，还是它们只是偶然事件。

我也及时注意到，我寻找了一些特定类型的相似之处。以下是一些例子：

+   相似的名称，无论是函数、参数、方法、变量、常量、类、模块、命名空间等的全名或嵌入在更长的名称中

+   相似的参数列表

+   相似的函数调用

+   不同的代码试图实现类似的结果

总的来说，我遵循这两个步骤：

1.  首先，注意相似之处。

1.  其次，决定是否移除相似之处。

当不确定相似之处是否对设计有更深层次的影响时，最好保留它。一旦你看到它们出现了三次，最好开始消除相似之处；这样，你就知道它违反了 DRY 原则，而不仅仅是一个偶然事件。

接下来，我们将看一下通过函数操作可以消除的几种相似之处。

# 通过部分应用解决参数相似之处

在我们之前的章节中，你已经看到了在一个参数的值相同时多次调用函数的情况。例如，在我们的井字游戏结果问题中的代码中，我们有一个函数负责检查一行是否被一个标记填满：

```cpp
auto lineFilledWith = [](const auto& line, const auto tokenToCheck){
    return all_of_collection(line, &tokenToCheck{   
        return token == tokenToCheck;});
};
```

由于井字游戏使用两个标记，`X`和`O`，很明显我们会重复调用这个函数，其中`tokenToCheck`要么是`X`要么是`O`。消除这种相似之处的常见方法是实现两个新函数，`lineFilledWithX`和`lineFilledWithO`：

```cpp
auto lineFilledWithX = [](const auto& line){
    return lineFilledWith(line, 'X');
};
```

这是一个可行的解决方案，但它仍然需要我们编写一个单独的函数和三行代码。正如我们所见，我们在函数式编程中还有另一个选择；我们可以简单地使用部分应用来获得相同的结果：

```cpp
auto lineFilledWithX = bind(lineFilledWith, _1, 'X'); 
auto lineFilledWithO = bind(lineFilledWith, _1, 'O');
```

我更喜欢在可能的情况下使用部分应用，因为这种代码只是管道，我需要编写的管道越少越好。然而，在团队中使用部分应用时需要小心。每个团队成员都应该熟悉部分应用，并且熟练理解这种类型的代码。否则，部分应用的使用只会使开发团队更难理解代码。

# 用函数组合替换另一个函数输出的调用函数相似之处

你可能已经注意到了过去在下面的代码中显示的模式：

```cpp
int processA(){
    a  = f1(....)
    b = f2(a, ...)
    c = f3(b, ...)
}
```

通常，如果你足够努力地寻找，你会发现在你的代码库中有另一个做类似事情的函数：

```cpp
int processB(){
    a  = f1Prime(....)
    b = f2(a, ...)
    c = f3(b, ...)
}
```

由于应用程序随着时间的推移变得越来越复杂，这种相似之处似乎有更深层次的原因。我们经常从实现一个通过多个步骤的简单流程开始。然后，我们实现同一流程的变体，其中一些步骤重复，而其他步骤则发生变化。有时，流程的变体涉及改变步骤的顺序，或者调整一些步骤。

在我们的实现中，这些步骤转化为以各种方式组合在其他函数中的函数。但是，如果我们使用上一步的输出并将其输入到下一步，我们就会发现代码中的相似之处，而不取决于每个步骤的具体操作。

为了消除这种相似之处，传统上我们会提取代码的相似部分并将结果传递，如下所示：

```cpp
int processA(){
    a  = f1(....)
    return doSomething(a)
}

int processB(){
    a = f1Prime(....)
    return doSomething(a)
}

int doSomething(auto a){
    b = f2(a, ...)
    return f3(b, ...)
}
```

然而，当提取函数时，代码通常变得更难理解和更难更改，如前面的代码所示。提取函数的共同部分并没有考虑到代码实际上是一个链式调用。

为了使这一点显而易见，我倾向于将代码模式重新格式化为单个语句，如下所示：

```cpp
processA = f3(f2(f1(....), ...), ...)
processB = f3(f2(f1Prime(....), ...), ...)
```

虽然不是每个人都喜欢这种格式，但两个调用之间的相似性和差异更加清晰。很明显，我们可以使用函数组合来解决问题——我们只需要将`f3`与`f2`组合，并将结果与`f1`或`f1Prime`组合，就可以得到我们想要的结果：

```cpp
C = f3 ∘ f2
processA = C ∘ f1
processB  = C ∘ f1Prime
```

这是一个非常强大的机制！我们可以通过函数组合创建无数的链式调用组合，只需几行代码。我们可以用几个组合语句替换隐藏的管道，这些管道伪装成函数中语句的顺序，表达我们代码的真实本质。

然而，正如我们在第四章中所看到的，*函数组合的概念*，在 C++中这并不一定是一项容易的任务，因为我们需要编写适用于我们特定情况的`compose`函数。在 C++提供更好的函数组合支持之前，我们被迫将这种机制保持在最低限度，并且只在相似性不仅明显，而且我们预计它会随着时间的推移而增加时才使用它。

# 使用更高级函数消除结构相似性

到目前为止，我们的讨论中一直存在一个模式——函数式编程帮助我们从代码中消除管道，并表达代码的真实结构。命令式编程使用语句序列作为基本结构；函数式编程减少了序列，并专注于函数的有趣运行。

当我们讨论结构相似性时，这一点最为明显。结构相似性是指代码结构重复的情况，尽管不一定是通过调用相同的函数或使用相同的参数。为了看到它的作用，让我们从我们的井字棋代码中一个非常有趣的相似之处开始。这是我们在第六章中编写的代码，*从数据到函数的思考*：

```cpp
auto lineFilledWith = [](const auto& line, const auto& tokenToCheck){
    return allOfCollection(line, &tokenToCheck{  
        return token == tokenToCheck;});
};

auto lineFilledWithX = bind(lineFilledWith, _1, 'X'); 
auto lineFilledWithO = bind(lineFilledWith, _1, 'O');

auto xWins = [](const auto& board){
    return any_of_collection(allLinesColumnsAndDiagonals(board), 
        lineFilledWithX);
};

auto oWins = [](const auto& board){
    return any_of_collection(allLinesColumnsAndDiagonals(board), 
        lineFilledWithO);
};

```

`xWins`和`oWins`函数看起来非常相似，因为它们都将相同的函数作为第一个参数调用，并且将`lineFilledWith`函数的变体作为它们的第二个参数。让我们消除它们的相似之处。首先，让我们移除`lineFilledWithX`和`lineFilledWithO`，并用它们的`lineFilledWith`等效替换：

```cpp
auto xWins = [](const auto& board){
    return any_of_collection(allLinesColumnsAndDiagonals(board), []  
        (const auto& line) { return lineFilledWith(line, 'X');});
};

auto oWins = [](const auto& board){
    return any_of_collection(allLinesColumnsAndDiagonals(board), []
        (const auto& line) { return lineFilledWith(line, 'O');});
};
```

现在相似之处显而易见，我们可以轻松提取一个通用函数：

```cpp
auto tokenWins = [](const auto& board, const auto& token){
    return any_of_collection(allLinesColumnsAndDiagonals(board),  
        token { return lineFilledWith(line, token);});
};
auto xWins = [](auto const board){
    return tokenWins(board, 'X');
};

auto oWins = [](auto const board){
    return tokenWins(board, 'O');
}
```

我们还注意到`xWins`和`oWins`只是`tokenWins`的偏函数应用，所以让我们明确这一点：

```cpp
auto xWins = bind(tokenWins, _1, 'X');
auto oWins = bind(tokenWins, _1, 'O');
```

现在，让我们专注于`tokenWins`：

```cpp
auto tokenWins = [](const auto& board, const auto& token){
    return any_of_collection(allLinesColumnsAndDiagonals(board),  
        token { return lineFilledWith(line, token);});
};
```

首先，我们注意到我们传递给`any_of_collection`的 lambda 是一个带有固定令牌参数的偏函数应用，所以让我们替换它：

```cpp
auto tokenWins = [](const auto& board, const auto& token){
    return any_of_collection(
            allLinesColumnsAndDiagonals(board), 
            bind(lineFilledWith, _1, token)
    );
};
```

这是一个非常小的函数，由于我们的偏函数应用，它具有很强的功能。然而，我们已经可以提取一个更高级的函数，它可以让我们创建更相似的函数而不需要编写任何代码。我还不知道该如何命名它，所以我暂时称它为`foo`：

```cpp
template <typename F, typename G, typename H>
auto foo(F f, G g, H h){
    return ={
    return f(g(first), 
    bind(h, _1, second));
    };
}
auto tokenWins = compose(any_of_collection, allLinesColumnsAndDiagonals, lineFilledWith);
```

我们的`foo`函数展示了代码的结构，但它相当难以阅读，所以让我们更好地命名事物：

```cpp
template <typename CollectionBooleanOperation, typename CollectionProvider, typename Predicate>
auto booleanOperationOnProvidedCollection(CollectionBooleanOperation collectionBooleanOperation, CollectionProvider collectionProvider, Predicate predicate){
    return ={
      return collectionBooleanOperation(collectionProvider(collectionProviderSeed), 
              bind(predicate, _1, predicateFirstParameter));
  };
}
auto tokenWins = booleanOperationOnProvidedCollection(any_of_collection, allLinesColumnsAndDiagonals, lineFilledWith);
```

我们引入了更高级的抽象层次，这可能会使代码更难理解。另一方面，我们使得能够在一行代码中创建`f(g(first), bind(h, _1, second))`形式的函数成为可能。

代码变得更好了吗？这取决于上下文、你的判断以及你和同事对高级函数的熟悉程度。然而，请记住——抽象虽然非常强大，但是也是有代价的。抽象更难理解，但如果你能够用抽象进行交流，你可以以非常强大的方式组合它们。使用这些高级函数就像从头开始构建一种语言——它使你能够在不同的层次上进行交流，但也为其他人设置了障碍。谨慎使用抽象！

# 使用高级函数消除隐藏的循环

结构重复的一个特殊例子经常在代码中遇到，我称之为**隐藏的循环**。隐藏的循环的概念是我们在一个序列中多次使用相同的代码结构。然而，其中的技巧在于被调用的函数或参数并不一定相同；因为函数式编程的基本思想是函数也是数据，我们可以将这些结构视为对可能也存储我们调用的函数的数据结构的循环。

我通常在一系列`if`语句中看到这种模式。事实上，我在使用井字棋结果问题进行实践会话时开始看到它们。在**面向对象编程**（**OOP**）或命令式语言中，问题的通常解决方案大致如下所示：

```cpp
enum Result {
    XWins,
    OWins,
    GameNotOverYet,
    Draw
};

Result winner(const Board& board){ 
    if(board.anyLineFilledWith(Token::X) ||    
        board.anyColumnFilledWith(Token::X) || 
        board.anyDiagonalFilledWith(Token::X)) 
    return XWins; 

    if(board.anyLineFilledWith(Token::O) ||  
        board.anyColumnFilledWith(Token::O) ||  
        board.anyDiagonalFilledWith(Token::O)) 
    return OWins; 

    if(board.notFilledYet()) 
    return GameNotOverYet; 

return Draw; 
}
```

在前面的示例中，`enum`标记包含三个值：

```cpp
enum Token {
    X,
    O,
    Blank
};

```

`Board`类大致如下：

```cpp
using Line = vector<Token>;

class Board{
    private: 
        const vector<Line> _board;

    public: 
        Board() : _board{Line(3, Token::Blank), Line(3, Token::Blank),  
            Line(3, Token::Blank)}{}
        Board(const vector<Line>& initial) : _board{initial}{}
...
}
```

`anyLineFilledWith`、`anyColumnFilledWith`、`anyDiagonalFilledWith`和`notFilledYet`的实现非常相似；假设一个 3 x 3 的棋盘，`anyLineFilledWith`的非常简单的实现如下：

```cpp
        bool anyLineFilledWith(const Token& token) const{
            for(int i = 0; i < 3; ++i){
                if(_board[i][0] == token && _board[i][1] == token &&  
                    _board[i][2] == token){
                    return true;
                }
            }
            return false;
        };
```

然而，我们对底层实现不太感兴趣，更感兴趣的是前面的 winner 函数中的相似之处。首先，`if`语句中的条件重复了，但更有趣的是，有一个重复的结构如下：

```cpp
if(condition) return value;
```

如果你看到一个使用数据而不是不同函数的结构，你会立刻注意到这是一个隐藏的循环。当涉及到函数调用时，我们并没有注意到这种重复，因为我们没有接受将函数视为数据的训练。但这确实就是它们的本质。

在我们消除相似之前，让我们简化条件。我将通过部分函数应用使所有条件成为无参数函数：

```cpp
auto tokenWins = [](const auto board, const auto& token){
    return board.anyLineFilledWith(token) ||   
board.anyColumnFilledWith(token) || board.anyDiagonalFilledWith(token);
};

auto xWins = bind(tokenWins, _1, Token::X);
auto oWins = bind(tokenWins, _1, Token::O);

auto gameNotOverYet = [](auto board){
    return board.notFilledYet();
};

Result winner(const Board& board){ 
    auto gameNotOverYetOnBoard = bind(gameNotOverYet, board);
    auto xWinsOnBoard = bind(xWins, board);
    auto oWinsOnBoard = bind(oWins, board);

    if(xWins()) 
        return XWins; 

    if(oWins())
        return OWins; 

    if(gameNotOverYetOnBoard()) 
        return GameNotOverYet; 

    return Draw; 
}
```

我们的下一步是消除四种不同条件之间的差异，并用循环替换相似之处。我们只需要有一对*(lambda, result)*的列表，并使用`find_if`这样的高级函数来为我们执行循环：

```cpp
auto True = [](){
    return true;
};

Result winner(Board board){
    auto gameNotOverYetOnBoard = bind(gameNotOverYet, board);
    auto xWinsOnBoard = bind(xWins, board);
    auto oWinsOnBoard = bind(oWins, board);

    vector<pair<function<bool()>, Result>> rules = {
        {xWins, XWins},
        {oWins, OWins},
        {gameNotOverYetOnBoard, GameNotOverYet},
        {True, Draw}
    };

    auto theRule = find_if(rules.begin(), rules.end(), [](auto pair){
            return pair.first();
            });
    // theRule will always be found, the {True, Draw} by default.
    return theRule->second;
}
```

最后一块拼图是确保我们的代码在没有其他情况适用时返回`Draw`。由于`find_if`返回符合规则的第一个元素，我们只需要在最后放上`Draw`，并与一个总是返回`true`的函数关联。我将这个函数恰如其分地命名为`True`。

这段代码对我们有什么作用呢？首先，我们可以轻松地添加新的条件和结果对，例如，如果我们曾经收到要在多个维度或更多玩家的情况下实现井字棋变体的请求。其次，代码更短。第三，通过一些改变，我们得到了一个简单但相当通用的规则引擎：

```cpp
auto True = [](){
    return true;
};

using Rule = pair<function<bool()>, Result>;

auto condition = [](auto rule){
    return rule.first();
};

auto result = [](auto rule){
    return rule.second;
};

// assumes that a rule is always found
auto findTheRule = [](const auto& rules){
    return *find_if(rules.begin(), rules.end(), [](auto rule){
 return condition(rule);
 });
};

auto resultForFirstRuleThatApplies = [](auto rules){
    return result(findTheRule(rules));
};

Result winner(Board board){
    auto gameNotOverYetOnBoard = bind(gameNotOverYet, board);
    vector<Rule> rules {
        {xWins, XWins},
        {oWins, OWins},
        {gameNotOverYetOnBoard, GameNotOverYet},
        {True, Draw}
    };

    return resultForFirstRuleThatApplies(rules);
}
```

在前面示例中唯一特殊的代码是规则列表。其他所有内容都是相当通用的，可以在多个问题上重复使用。

和往常一样，提升抽象级别是需要付出代价的。我们花时间尽可能清晰地命名事物，我相信这段代码非常容易阅读。然而，对许多人来说可能并不熟悉。

另一个可能的问题是内存使用。尽管初始版本的代码重复了相同的代码结构，但它不需要为函数和结果对的列表分配内存；然而，重要的是要测量这些东西，因为即使初始代码也需要一些额外指令的处理内存。

这个例子向我们展示了如何通过一个非常简单的代码示例将重复的结构转换为循环。这只是皮毛；这种模式是如此普遍，我相信一旦你开始寻找，你会在你的代码中注意到它。

# 摘要

在本章中，我们看了不同类型的代码相似之处，以及如何通过各种函数式编程技术来减少它们。从可以用部分应用替换的重复参数，到可以转换为函数组合的链式调用，一直到可以通过更高级别的函数移除的结构相似之处，你现在已经有能力注意并减少任何代码库中的相似之处了。

正如你已经注意到的，我们开始讨论代码结构和软件设计。这将我们引向设计的另一个核心原则——高内聚和低耦合。我们如何使用函数来增加内聚？原来这正是类非常有用的地方，这也是我们将在下一章讨论的内容。
