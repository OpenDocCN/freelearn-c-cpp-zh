# 第九章：函数式编程的测试驱动开发

**测试驱动开发**（**TDD**）是一种设计软件的非常有用的方法。该方法如下——我们首先编写一个失败的单一测试，然后实现最少的代码使测试通过，最后进行重构。我们在短时间内进行小循环来完成这个过程。

我们将看看纯函数如何简化测试，并提供一个应用 TDD 的函数示例。纯函数允许我们编写简单的测试，因为它们始终为相同的输入参数返回相同的值；因此，它们相当于大数据表。因此，我们可以编写模拟输入和预期输出的数据表的测试。

本章将涵盖以下主题：

+   如何使用数据驱动测试利用纯函数的优势

+   了解 TDD 周期的基础

+   如何使用 TDD 设计纯函数

# 技术要求

您将需要一个支持**C++ 17**的编译器。我使用了**GCC 7.3.0**。

代码可以在 GitHub 上找到，网址为[https:/​/​github.​com/​PacktPublishing/​Hands-​On-​Functional-Programming-​with-​Cpp](https://github.%E2%80%8Bcom/PacktPublishing/Hands-On-Functional-Programming-with-Cpp)，在`Chapter09`文件夹中。它包括并使用`doctest`，这是一个单头开源单元测试库。您可以在其 GitHub 存储库上找到它，网址为[https:/​/github.​com/​onqtam/​doctest](https://github.%E2%80%8Bcom/onqtam/doctest)。

# 函数式编程的 TDD

20 世纪 50 年代的编程与今天的编程非常不同。我们现在所知道的程序员的工作被分为三个角色。程序员会编写要实现的算法。然后，专门的打字员会使用特殊的机器将其输入到穿孔卡片中。然后，程序员必须手动验证穿孔卡片是否正确——尽管有数百张。一旦确认穿孔卡片正确，程序员会将它们交给大型机操作员。由于当时唯一存在的计算机非常庞大且价格昂贵，因此在计算机上花费的时间必须受到保护。大型机操作员负责计算机，确保最重要的任务优先进行，因此新程序可能需要等待几天才能运行。运行后，程序将打印完整的堆栈跟踪。如果出现错误，程序员必须查看一张充满奇怪符号的非常长的纸，并找出可能出错的地方。这个过程缓慢、容易出错且不可预测。

然而，一些工程师提出了一个想法。如果他们不是从失败的程序中获得复杂的输出，而是得到明确指出问题的信息会怎样？他们决定开始编写额外的代码，检查生产代码并生成通过或失败的输出。他们不是运行程序，或者在运行程序的同时，他们会运行单元测试。

一旦程序员拥有了更短的反馈循环，如终端的发明，后来是个人电脑和强大的调试器，单元测试的实践就被遗忘了。然而，它从未完全消失，突然以不同的形式回来了。

直到 20 世纪 90 年代，单元测试才意外地重新出现。包括 Kent Beck、Ward Cunningham 和 Ron Jeffries 在内的一群程序员尝试将开发实践推向极端。他们的努力的结果被称为**极限编程**（**XP**）。其中一种实践就是单元测试，结果非常有趣。

常见的单元测试实践是在编写代码后写一些测试，作为测试期间的一部分。这些测试通常由测试人员编写——与实现功能的程序员不同的一个组。

然而，最初的 XPers 尝试了一种不同的单元测试方式。如果我们在编写代码的同时编写测试呢？更有趣的是，如果我们在*实现之前*编写测试呢？这导致了两种不同的技术——**测试驱动编程**（**TFP**），它包括首先编写一些测试，然后编写一些代码使测试通过，以及我们将在更详细地讨论的 TDD。

当我第一次听说这些技术时，我既感到困惑又着迷。你怎么能为不存在的东西编写测试呢？这有什么好处呢？幸运的是，在 J.B. Rainsberger 的支持下，我很快意识到了 TFP/TDD 的力量。我们的客户和利益相关者希望尽快在软件中获得可用的功能。然而，往往他们无法解释他们想要的功能。从测试开始意味着你完全理解了要实现什么，并且会引发有用和有趣的对话，澄清需求。一旦需求明确，我们就可以专注于实现。此外，在 TDD 中，我们尽快清理代码，以免随着时间的推移造成混乱。这真的是一种非常强大的技术！

但让我们从头开始。我们如何编写单元测试呢？更重要的是，对于我们的目的来说，为纯函数编写单元测试更容易吗？

# 纯函数的单元测试

让我们首先看一下单元测试是什么样子的。在本书中，我已经使用了一段时间，我相信你能理解这段代码。但是现在是时候看一个特定的例子了：

```cpp
TEST_CASE("Greater Than"){
    int first = 3;
    int second = 2;

    bool result = greater<int>()(first, second);

    CHECK(result);
}
```

我们首先使用特定值初始化两个变量（单元测试的*安排*部分）。然后我们调用生产代码（单元测试的*行动*部分）。最后，我们检查结果是否符合我们的预期（单元测试的*断言*部分）。我们正在使用的名为`doctest`的库提供了允许我们编写单元测试的宏的实现。虽然 C++存在更多的单元测试库，包括 GTest 和`Boost::unit_test`等，但它们提供给程序员的功能相当相似。

在谈论单元测试时，更重要的是找出使其有用的特征。前面的测试是小型、专注、快速的，只能因为一个原因而失败。所有这些特征使测试有用，因为它易于编写、易于维护、清晰明了，并且在引入错误时提供有用和快速的反馈。

在技术方面，前面的测试是基于示例的，因为它使用一个非常具体的示例来检查代码的特定行为。我们将在第十一章中看到一种名为**基于属性的测试**的不同单元测试方法，*基于属性的测试*。由于这是基于示例的测试，一个有趣的问题出现了：如果我们想测试`greaterThan`函数，还有哪些其他示例会很有趣呢？

好吧，我们想要查看函数的所有可能行为。那么，它可能的输出是什么？以下是一个列表：

+   如果第一个值大于第二个值，则为 True

+   如果第一个值小于第二个值，则为 False

然而，这还不够。让我们添加边缘情况：

+   如果第一个值等于第二个值，则为 False

还有，不要忘记可能的错误。传入值的域是什么？可以传入负值吗？浮点数值？复数？这是与该函数的利益相关者进行有趣对话。

现在让我们假设最简单的情况——该函数将仅接受有效的整数。这意味着我们需要另外两个单元测试来检查第一个参数小于第二个参数的情况以及两者相等的情况：

```cpp
TEST_CASE("Not Greater Than when first is less than second"){
    int first = 2;
    int second = 3;

    bool result = greater<int>()(first, second);

    CHECK_FALSE(result);
}

TEST_CASE("Not Greater Than when first equals second"){
    int first = 2;

    bool result = greater<int>()(first, first);

    CHECK_FALSE(result);
}
```

在第七章中，*使用功能操作去除重复*，我们讨论了代码相似性以及如何去除它。在这里，我们有一个测试之间的相似性。去除它的一种方法是编写所谓的**数据驱动测试**（**DDT**）。在 DDT 中，我们编写一组输入和期望的输出，并在每行数据上重复测试。不同的测试框架提供了不同的编写这些测试的方式；目前，`doctest`对 DDT 的支持有限，但我们仍然可以按照以下方式编写它们：

```cpp
TEST_CASE("Greater than") {
    struct Data {
        int first;
        int second;
        bool expected;
 } data;

    SUBCASE("2 is greater than 1") { data.first = 2; data.second = 1; 
        data.expected = true; }
    SUBCASE("2 is not greater than 2") { data.first = 2; data.second = 
         2; data.expected = false; }
    SUBCASE("2 is not greater than 3") { data.first = 2; data.second = 
         3; data.expected = false; }

    CAPTURE(data);

    CHECK_EQ(greaterThan(data.first, data.second), data.expected);
}
```

如果我们忽略管道代码（`struct Data`定义和对`CAPTURE`宏的调用），这显示了一种非常方便的编写测试的方式——特别是对于纯函数。鉴于纯函数根据定义在接收相同输入时返回相同输出，用一组输入/输出进行测试是很自然的。

DDT 的另一个便利之处在于，我们可以通过向列表添加新行来轻松添加新的测试。这在使用纯函数进行 TDD 时特别有帮助。

# TDD 循环

TDD 是一个常见的开发循环，通常如下所示：

+   **红色**：编写一个失败的测试。

+   **绿色**：通过对生产代码进行尽可能小的更改来使测试通过。

+   **重构**：重新组织代码以包含新引入的行为。

然而，TDD 的实践者（比如我自己）会急于提到 TDD 循环始于另一步骤——思考。更准确地说，在编写第一个测试之前，让我们理解我们要实现的内容，并找到现有代码中添加行为的好位置。

这个循环看起来简单得令人误解。然而，初学者经常在第一个测试应该是什么以及之后的测试应该是什么方面挣扎，同时编写过于复杂的代码。**重构**本身就是一门艺术，需要对代码异味、设计原则和设计模式有所了解。总的来说，最大的错误是过于考虑你想要获得的代码结构，并编写导致那种结构的测试。

相反，TDD 需要一种心态的改变。我们从行为开始，在小步骤中完善适合该行为的代码结构。一个好的实践者会有小于 15 分钟的步骤。但这并不是 TDD 的唯一惊喜。

TDD 最大的惊喜是，它可以通过允许您探索同一问题的各种解决方案来教您软件设计。您愿意探索的解决方案越多，您在设计代码方面就会变得越好。当以适当的好奇心进行实践时，TDD 是一个持续的学习经验。

我希望我引起了你对 TDD 的好奇心。关于这个主题还有很多要学习的，但是对于我们的目标来说，尝试一个例子就足够了。而且，由于我们正在谈论函数式编程，我们将使用 TDD 来设计一个纯函数。

# 例子——使用 TDD 设计一个纯函数

再次，我们需要一个问题来展示 TDD 的实际应用。由于我喜欢使用游戏来练习开发实践，我查看了 Coding Dojo Katas（[`codingdojo.org/kata/PokerHands/`](http://codingdojo.org/kata/)）的列表，并选择了扑克牌问题来进行练习。

# 扑克牌问题

问题的描述如下——给定两个或多个扑克牌手，我们需要比较它们并返回排名较高的手以及它赢得的原因。

每手有五张牌，这些牌是从一副普通的 52 张牌的牌组中挑选出来的。牌组由四种花色组成——梅花、方块、红桃和黑桃。每种花色从`2`开始，以 A 结束，表示如下——`2`、`3`、`4`、`5`、`6`、`7`、`8`、`9`、`T`、`J`、`Q`、`K`、`A`（`T`表示 10）。

扑克牌手中的牌将形成不同的组合。手的价值由这些组合决定，按以下降序排列：

+   **同花顺**：五张相同花色的牌，连续的值。例如，`2♠`，`3♠`，`4♠`，`5♠`和`6♠`。起始值越高，同花顺的价值就越高。

+   **四条**：四张相同牌值的牌。最高的是四张 A——`A♣`，`A♠`，`A♦`和`A♥`。

+   **葫芦**：三张相同牌值的牌，另外两张牌也是相同的牌值（但不同）。最高的是——`A♣`，`A♠`，`A♦`，`K♥`和`K♠`。

+   **同花**：五张相同花色的牌。例如——`2♠`，`3♠`，`5♠`，`6♠`和`9♠`。

+   **顺子**：五张连续值的牌。例如——`2♣`，`3♠`，`4♥`，`5♣`和`6♦`。

+   **三条**：三张相同牌值的牌。例如——`2♣`，`2♠`和`2♥`。

+   **两对**：见对子。例如——`2♣`，`2♠`，`3♥`和`3♣`。

+   **对子**：两张相同牌值的牌。例如——`2♣`和`2♠`。

+   **高牌**：当没有其他组合时，比较每手中最高的牌，最高的获胜。如果最高的牌具有相同的值，则比较下一个最高的牌，以此类推。

# 要求

我们的目标是实现一个程序，比较两个或更多个扑克牌手，并返回赢家和原因。例如，让我们使用以下输入：

+   **玩家 1**：`*2♥ 4♦ 7♣ 9♠ K♦*`

+   **玩家 2**：`*2♠ 4♥ 8♣ 9♠ A♥*`

对于这个输入，我们应该得到以下输出：

+   *玩家 2 以他们的高牌——一张 A 赢得比赛*

# 步骤 1 - 思考

让我们更详细地看一下问题。更准确地说，我们试图将问题分解为更小的部分，而不要过多考虑实现。我发现查看可能的输入和输出示例，并从一个简化的问题开始，可以让我尽快实现一些有效的东西，同时保持问题的本质。

很明显，我们有很多组合要测试。那么，什么是限制我们测试用例的问题的有用简化呢？

一个明显的方法是从手中的牌较少开始。我们可以从一张牌开始，而不是五张牌。这将限制我们的规则为高牌。下一步是有两张牌，这引入了*对子>高牌*，*更高的对子>更低的对子*，依此类推。

另一种方法是从五张牌开始，但限制规则。从高牌开始，然后实现一对，然后两对，依此类推；或者，从同花顺一直到对子和高牌。

TDD 的有趣之处在于，这些方法中的任何一个都将以相同的方式产生结果，尽管通常使用不同的代码结构。TDD 的一个优势是通过改变测试的顺序来帮助您访问相同问题的多种设计。

不用说，我以前做过这个问题，但我总是从手中的一张牌开始。让我们有些乐趣，尝试一种不同的方式，好吗？我选择用五张牌开始，从同花顺开始。为了保持简单，我现在只支持两个玩家，而且由于我喜欢给他们起名字，我会用 Alice 和 Bob。

# 例子

对于这种情况，有一些有趣的例子是什么？让我们先考虑可能的输出：

+   Alice 以同花顺获胜。

+   Bob 以同花顺获胜。

+   Alice 和 Bob 有同样好的同花顺。

+   未决（即尚未实施）。

现在，让我们写一些这些输出的输入示例：

```cpp
Case 1: Alice wins

Inputs:
 Alice: 2♠, 3♠, 4♠, 5♠, 6♠
 Bob: 2♣, 4♦, 7♥, 9♠, A♥

Output:
 Alice wins with straight flush

Case 2: Bob wins

Inputs:
    Alice: 2♠, 3♠, 4♠, 5♠, 9♠
    Bob: 2♣, 3♣, 4♣, 5♣, 6♣

Output:
    Bob wins with straight flush

Case 3: Alice wins with a higher straight flush

Inputs:
    Alice: 3♠, 4♠, 5♠, 6♠, 7♠
    Bob: 2♣, 3♣, 4♣, 5♣, 6♣

Output:
    Alice wins with straight flush

Case 4: Draw

Inputs:
    Alice: 3♠, 4♠, 5♠, 6♠, 7♠
    Bob: 3♣, 4♣, 5♣, 6♣, 7♣

Output:
    Draw (equal straight flushes)

Case 5: Undecided

Inputs:
    Alice: 3♠, 3♣, 5♠, 6♠, 7♠
    Bob: 3♣, 4♣, 6♣, 6♥, 7♣

Output:
    Not implemented yet.

```

有了这些例子，我们准备开始编写我们的第一个测试！

# 第一个测试

根据我们之前的分析，我们的第一个测试如下：

```cpp
Case 1: Alice wins

Inputs:
 Alice: 2♠, 3♠, 4♠, 5♠, 6♠
 Bob: 2♣, 4♦, 7♥, 9♠, A♥

Output:
 Alice wins with straight flush
```

让我们写吧！我们期望这个测试失败，所以在这一点上我们可以做任何我们想做的事情。我们需要用前面的卡片初始化两只手。现在，我们将使用`vector<string>`来表示每只手。然后，我们将调用一个函数（目前还不存在）来比较这两只手，我们想象这个函数将在某个时候实现。最后，我们将检查结果是否与之前定义的预期输出消息相匹配：

```cpp
TEST_CASE("Alice wins with straight flush"){
    vector<string> aliceHand{"2♠", "3♠", "4♠", "5♠", "6♠"};
    vector<string> bobHand{"2♣", "4♦", "7♥", "9♠", "A♥"};

    auto result = comparePokerHands(aliceHand, bobHand);

    CHECK_EQ("Alice wins with straight flush", result);
}
```

现在，这个测试无法编译，因为我们甚至还没有创建`comparePokerHands`函数。是时候继续前进了。

# 使第一个测试通过

让我们先写这个函数。这个函数需要返回一些东西，所以我们暂时只返回空字符串：

```cpp
auto comparePokerHands = [](const auto& aliceHand, const auto& bobHand){
    return "";
};
```

使测试通过的最简单实现是什么？这是 TDD 变得更加奇怪的地方。使测试通过的最简单实现是将预期结果作为硬编码值返回：

```cpp
auto comparePokerHands = [](const auto& aliceHand, const auto& bobHand){
    return "Alice wins with straight flush";
};
```

此时，我的编译器抱怨了，因为我打开了所有警告，并且将所有警告报告为错误。编译器注意到我们没有使用这两个参数并抱怨。这是一个合理的抱怨，但我计划很快开始使用这些参数。C++语言给了我们一个简单的解决方案——只需删除或注释掉参数名，如下面的代码所示：

```cpp
auto comparePokerHands = [](const auto& /*aliceHand*/, const auto&  
    /*bobHand*/){
        return "Alice wins with straight flush";
};
```

我们运行测试，我们的第一个测试通过了！太棒了，有东西可以用了！

# 重构

有什么需要重构的吗？嗯，我们有两个被注释掉的参数名，我通常会把它们删除掉，因为注释掉的代码只会增加混乱。但是，我决定暂时保留它们，因为我知道我们很快会用到它们。

我们还有一个重复的地方——在测试和实现中都出现了相同的“Alice 以顺子获胜”的字符串。值得把它提取为一个常量或者公共变量吗？如果这是我们的实现的最终结果，那当然可以。但我知道这个字符串实际上是由多个部分组成的——获胜玩家的名字，以及根据哪种手牌获胜的规则。我想暂时保持它原样。

因此，没有什么需要重构的。让我们继续吧！

# 再次思考

当前的实现感觉令人失望。只是返回一个硬编码的值并不能解决太多问题。或者呢？

这是学习 TDD 时需要的心态转变。我知道这一点，因为我经历过。我习惯于看最终结果，将这个解决方案与我试图实现的目标进行比较，感觉令人失望。然而，有一种不同的看待方式——我们有一个可以工作的东西，而且我们有最简单的实现。还有很长的路要走，但我们已经可以向利益相关者展示一些东西。而且，正如我们将看到的，我们总是在坚实的基础上构建，因为我们编写的代码是经过充分测试的。这两件事是非常令人振奋的；我只希望你在尝试 TDD 时也能有同样的感受。

但是，接下来我们该怎么办呢？我们有几个选择。

首先，我们可以写另一个测试，其中 Alice 以顺子获胜。然而，这不会改变实现中的任何东西，测试会立即通过。虽然这似乎违反了 TDD 循环，但为了我们的安心，增加更多的测试并没有错。绝对是一个有效的选择。

其次，我们可以转移到下一个测试，其中 Bob 以顺子获胜。这肯定会改变一些东西。

这两个选项都不错，你可以选择其中任何一个。但由于我们想要看到 DDT 的实践，让我们先写更多的测试。

# 更多的测试

将我们的测试转换成 DDT 并添加更多的案例非常容易。我们只需改变 Alice 手牌的值，而保持 Bob 的手牌不变。结果如下：

```cpp
TEST_CASE("Alice wins with straight flush"){
    vector<string> aliceHand;
    const vector<string> bobHand {"2♣", "4♦", "7♥", "9♠", "A♥"};

    SUBCASE("2 based straight flush"){
        aliceHand = {"2♠", "3♠", "4♠", "5♠", "6♠"};
    };
    SUBCASE("3 based straight flush"){
        aliceHand = {"3♠", "4♠", "5♠", "6♠", "7♠"};
    };
    SUBCASE("4 based straight flush"){
        aliceHand = {"4♠", "5♠", "6♠", "7♠", "8♠"};
    };
    SUBCASE("10 based straight flush"){
        aliceHand = {"T♠", "J♠", "Q♠", "K♠", "A♠"};
    };

    CAPTURE(aliceHand);

    auto result = comparePokerHands(aliceHand, bobHand);

    CHECK_EQ("Alice wins with straight flush", result);
}
```

再次，所有这些测试都通过了。是时候继续进行我们的下一个测试了。

# 第二个测试

我们描述的第二个测试是 Bob 以顺子获胜：

```cpp
Case: Bob wins

Inputs:
 Alice: 2♠, 3♠, 4♠, 5♠, 9♠
 Bob: 2♣, 3♣, 4♣, 5♣, 6♣

Output:
 Bob wins with straight flush
```

让我们写吧！这一次，让我们从一开始就使用数据驱动的格式：

```cpp
TEST_CASE("Bob wins with straight flush"){
    const vector<string> aliceHand{"2♠", "3♠", "4♠", "5♠", "9♠"};
    vector<string> bobHand;

    SUBCASE("2 based straight flush"){
        bobHand = {"2♣", "3♣", "4♣", "5♣", "6♣"};
    };

    CAPTURE(bobHand);

    auto result = comparePokerHands(aliceHand, bobHand);

    CHECK_EQ("Bob wins with straight flush", result);
}
```

当我们运行这个测试时，它失败了，原因很简单——我们有一个硬编码的实现，说 Alice 获胜。现在怎么办？

# 使测试通过

再次，我们需要找到使这个测试通过的最简单方法。即使我们可能不喜欢这个实现，下一步是清理混乱。那么，最简单的实现是什么呢？

显然，我们需要在我们的实现中引入一个条件语句。问题是，我们应该检查什么？

再次，我们有几个选择。一个选择是再次伪装，使用与我们期望获胜的确切手牌进行比较：

```cpp
auto comparePokerHands = [](const vector<string>& /*aliceHand*/, const vector<string>& bobHand){
    const vector<string> winningBobHand {"2♣", "3♣", "4♣", "5♣", "6♣"};
    if(bobHand == winningBobHand){
        return "Bob wins with straight flush";
    }
    return "Alice wins with straight flush";
};
```

为了使其编译，我们还必须使`vector<string>` hands 的类型出现在各处。一旦这些更改完成，测试就通过了。

我们的第二个选择是开始实现实际的同花顺检查。然而，这本身就是一个小问题，要做好需要更多的测试。

我现在会选择第一种选项，重构，然后开始更深入地研究检查同花顺的实现。

# 重构

有什么需要重构的吗？我们仍然有字符串的重复。此外，我们在包含 Bob 的手的向量中添加了重复。但我们期望这两者很快都会消失。

然而，还有一件事让我感到不安——`vector<string>` 出现在各处。让我们通过为`vector<string>`类型命名为`Hand`来消除这种重复：

```cpp
using Hand = vector<string>;

auto comparePokerHands = [](const Hand& /*aliceHand*/, const Hand& bobHand){
    Hand winningBobHand {"2♣", "3♣", "4♣", "5♣", "6♣"};
    if(bobHand == winningBobHand){
        return "Bob wins with straight flush";
    }
    return "Alice wins with straight flush";
};

TEST_CASE("Bob wins with straight flush"){
    Hand aliceHand{"2♠", "3♠", "4♠", "5♠", "9♠"};
    Hand bobHand;

    SUBCASE("2 based straight flush"){
        bobHand = {"2♣", "3♣", "4♣", "5♣", "6♣"};
    };

    CAPTURE(bobHand);

    auto result = comparePokerHands(aliceHand, bobHand);

    CHECK_EQ("Bob wins with straight flush", result);
}
```

# 思考

再次思考。我们已经用硬编码的值实现了两种情况。对于 Alice 以同花顺获胜并不是一个大问题，但如果我们为 Bob 添加另一组不同的牌测试用例，这就是一个问题。我们可以进行更多的测试，但不可避免地，我们需要实际检查同花顺。我认为现在是一个很好的时机。

那么，什么是同花顺？它是一组有相同花色和连续值的五张牌。我们需要一个函数，它可以接受一组五张牌，并在是同花顺时返回`true`，否则返回`false`。让我们写下一些例子：

+   输入：`2♣ 3♣ 4♣ 5♣ 6♣` => 输出：`true`

+   输入：`2♠ 3♠ 4♠ 5♠ 6♠` => 输出：`true`

+   输入：`T♠ J♠ Q♠ K♠ A♠` => 输出：`true`

+   输入：`2♣ 3♣ 4♣ 5♣ 7♣` => 输出：`false`

+   输入：`2♣ 3♣ 4♣ 5♣ 6♠` => 输出：`false`

+   输入：`2♣ 3♣ 4♣ 5♣` => 输出：`false`（只有四张牌，需要正好五张）

+   输入：`[空向量]` => 输出：`false`（没有牌，需要正好五张）

+   输入：`2♣ 3♣ 4♣ 5♣ 6♣ 7♣` => 输出：`false`（六张牌，需要正好五张）

你会注意到我们也考虑了边缘情况和奇怪的情况。我们有足够的信息可以继续，所以让我们写下下一个测试。

# 下一个测试-简单的同花顺

我更喜欢从正面案例开始，因为它们往往会更推进实现。让我们看最简单的一个：

+   输入：`2♣ 3♣ 4♣ 5♣ 6♣` => 输出：`true`

测试如下：

```cpp
TEST_CASE("Hand is straight flush"){
    Hand hand;

    SUBCASE("2 based straight flush"){
        hand = {"2♣", "3♣", "4♣", "5♣", "6♣"};
    };

    CAPTURE(hand);

    CHECK(isStraightFlush(hand));
}
```

再次，测试无法编译，因为我们没有实现`isStraightFlush`函数。但测试是正确的，它失败了，所以是时候继续了。

# 使测试通过

再次，第一步是编写函数的主体并返回预期的硬编码值：

```cpp
auto isStraightFlush = [](const Hand&){
    return true;
};
```

我们运行了测试，它们通过了，所以现在我们完成了！

# 继续前进

嗯，你可以看到这是怎么回事。我们可以为正确的同花顺添加一些更多的输入，但它们不会改变实现。第一个将迫使我们推进实现的测试是我们的第一个不是同花顺的一组牌的例子。

对于本章的目标，我将快进。但我强烈建议你自己经历所有的小步骤，并将你的结果与我的进行比较。学习 TDD 的唯一方法是自己练习并反思自己的方法。

# 实现 isStraightFlush

让我们再次看看我们要达到的目标——同花顺，它由正好五张具有相同花色和连续值的牌定义。我们只需要在代码中表达这三个条件：

```cpp
auto isStraightFlush = [](const Hand& hand){
    return has5Cards(hand) && 
        isSameSuit(allSuits(hand)) && 
        areValuesConsecutive(allValuesInOrder(hand));
};
```

实现得到了一些不同的 lambda 的帮助。首先，为了检查组合的长度，我们使用`has5Cards`：

```cpp
auto has5Cards = [](const Hand& hand){
    return hand.size() == 5;
};
```

然后，为了检查它是否有相同的花色，我们使用`allSuits`来提取手中的花色，`isSuitEqual`来比较两个花色，`isSameSuit`来检查手中的所有花色是否相同：

```cpp
using Card = string;
auto suitOf = [](const Card& card){
    return card.substr(1);
};

auto allSuits = [](Hand hand){
    return transformAll<vector<string>>(hand, suitOf);
};

auto isSameSuit = [](const vector<string>& allSuits){
    return std::equal(allSuits.begin() + 1, allSuits.end(),  
        allSuits.begin());
};
```

最后，为了验证这些值是连续的，我们使用`valueOf`从一张牌中提取值，使用`allValuesInOrder`获取一手牌中的所有值并排序，使用`toRange`从一个初始值开始创建一系列连续的值，使用`areValuesConsecutive`检查一手牌中的值是否连续：

```cpp
auto valueOf = [](const Card& card){
    return charsToCardValues.at(card.front());
};

auto allValuesInOrder = [](const Hand& hand){
    auto theValues = transformAll<vector<int>>(hand, valueOf);
    sort(theValues.begin(), theValues.end());
    return theValues;
};

auto toRange = [](const auto& collection, const int startValue){
    vector<int> range(collection.size());
    iota(begin(range), end(range), startValue);
    return range;
};

auto areValuesConsecutive = [](const vector<int>& allValuesInOrder){
    vector<int> consecutiveValues = toRange(allValuesInOrder, 
        allValuesInOrder.front());

    return consecutiveValues == allValuesInOrder;
};
```

最后一块拼图是一个从`char`到`int`的映射，帮助我们将所有的牌值，包括`T`、`J`、`Q`、`K`和`A`，转换成数字：

```cpp
const std::map<char, int> charsToCardValues = {
    {'1', 1},
    {'2', 2},
    {'3', 3},
    {'4', 4},
    {'5', 5},
    {'6', 6},
    {'7', 7},
    {'8', 8},
    {'9', 9},
    {'T', 10},
    {'J', 11},
    {'Q', 12},
    {'K', 13},
    {'A', 14},
};
```

让我们也看一下我们的测试（显然都通过了）。首先是有效的顺子同花的测试；我们将检查以`2`、`3`、`4`和`10`开头的顺子同花，以及它们在数据区间上的变化：

```cpp
TEST_CASE("Hand is straight flush"){
    Hand hand;

    SUBCASE("2 based straight flush"){
        hand = {"2♣", "3♣", "4♣", "5♣", "6♣"};
    };

    SUBCASE("3 based straight flush"){
        hand = {"3♣", "4♣", "5♣", "6♣", "7♣"};
    };

    SUBCASE("4 based straight flush"){
        hand = {"4♣", "5♣", "6♣", "7♣", "8♣"};
    };

    SUBCASE("4 based straight flush on hearts"){
        hand = {"4♥", "5♥", "6♥", "7♥", "8♥"};
    };

    SUBCASE("10 based straight flush on hearts"){
        hand = {"T♥", "J♥", "Q♥", "K♥", "A♥"};
    };

    CAPTURE(hand);

    CHECK(isStraightFlush(hand));
}
```

最后，对于一组不是有效顺子同花的牌的测试。我们将使用几乎是顺子同花的手牌作为输入，除了花色不同、牌数不够或者牌数太多之外：

```cpp
TEST_CASE("Hand is not straight flush"){
    Hand hand;

    SUBCASE("Would be straight flush except for one card from another 
        suit"){
            hand = {"2♣", "3♣", "4♣", "5♣", "6♠"};
    };

    SUBCASE("Would be straight flush except not enough cards"){
        hand = {"2♣", "3♣", "4♣", "5♣"};
    };

    SUBCASE("Would be straight flush except too many cards"){
        hand = {"2♣", "3♣", "4♣", "5♣", "6♠", "7♠"};
    };

    SUBCASE("Empty hand"){
        hand = {};
    };

    CAPTURE(hand);

    CHECK(!isStraightFlush(hand));
}
```

现在是时候回到我们的主要问题了——比较扑克牌的手。

# 将检查顺子同花的代码重新插入到 comparePokerHands 中

尽管我们迄今为止实现了所有这些，但我们的`comparePokerHands`的实现仍然是硬编码的。让我们回顾一下它当前的状态：

```cpp
auto comparePokerHands = [](const Hand& /*aliceHand*/, const Hand& bobHand){
    const Hand winningBobHand {"2♣", "3♣", "4♣", "5♣", "6♣"};
    if(bobHand == winningBobHand){
        return "Bob wins with straight flush";
    }
    return "Alice wins with straight flush";
};
```

但是，现在我们有了检查顺子同花的方法！所以，让我们把我们的实现插入进去：

```cpp
auto comparePokerHands = [](Hand /*aliceHand*/, Hand bobHand){
    if(isStraightFlush(bobHand)) {
        return "Bob wins with straight flush";
    }
    return "Alice wins with straight flush";
};
```

所有的测试都通过了，所以我们快要完成了。是时候为我们的`Bob 赢得顺子同花`情况添加一些额外的测试，以确保我们没有遗漏。我们将保持 Alice 的相同手牌，一个几乎是顺子同花的手牌，然后改变 Bob 的手牌，从以`2`、`3`和`10`开头的顺子同花：

```cpp
TEST_CASE("Bob wins with straight flush"){
    Hand aliceHand{"2♠", "3♠", "4♠", "5♠", "9♠"};
    Hand bobHand;

    SUBCASE("2 based straight flush"){
        bobHand = {"2♣", "3♣", "4♣", "5♣", "6♣"};
    };

    SUBCASE("3 based straight flush"){
        bobHand = {"3♣", "4♣", "5♣", "6♣", "7♣"};
    };

    SUBCASE("10 based straight flush"){
        bobHand = {"T♣", "J♣", "Q♣", "K♣", "A♣"};
    };

    CAPTURE(bobHand);

    auto result = comparePokerHands(aliceHand, bobHand);

    CHECK_EQ("Bob wins with straight flush", result);
}
```

所有之前的测试都通过了。所以，我们已经完成了两种情况——当 Alice 或 Bob 有顺子同花而对手没有时。是时候转移到下一个情况了。

# 比较两个顺子同花

正如我们在本节开头讨论的那样，当 Alice 和 Bob 都有顺子同花时还有另一种情况，但是 Alice 用更高的顺子同花赢了：

```cpp
Case: Alice wins with a higher straight flush

Inputs:
 Alice: 3♠, 4♠, 5♠, 6♠, 7♠
 Bob: 2♣, 3♣, 4♣, 5♣, 6♣

Output:
 Alice wins with straight flush
```

让我们写下测试并运行它：

```cpp
TEST_CASE("Alice and Bob have straight flushes but Alice wins with higher straight flush"){
    Hand aliceHand;
    Hand bobHand{"2♣", "3♣", "4♣", "5♣", "6♣"};

    SUBCASE("3 based straight flush"){
        aliceHand = {"3♠", "4♠", "5♠", "6♠", "7♠"};
    };

    CAPTURE(aliceHand);

    auto result = comparePokerHands(aliceHand, bobHand);

    CHECK_EQ("Alice wins with straight flush", result);
}
```

测试失败了，因为我们的`comparePokerHands`函数返回 Bob 赢了，而不是 Alice。让我们用最简单的实现来修复这个问题：

```cpp
auto comparePokerHands = [](const Hand& aliceHand, const Hand& bobHand){
    if(isStraightFlush(bobHand) && isStraightFlush(aliceHand)){
         return "Alice wins with straight flush";
    }

    if(isStraightFlush(bobHand)) {
        return "Bob wins with straight flush";
    }

    return "Alice wins with straight flush";
};
```

我们的实现决定了如果 Alice 和 Bob 都有顺子同花，那么 Alice 总是赢。这显然不是我们想要的，但测试通过了。那么我们可以写什么测试来推动实现向前发展呢？

# 思考

事实证明，我们在之前的分析中漏掉了一个情况。我们看了当 Alice 和 Bob 都有顺子同花并且 Alice 赢的情况；但是如果 Bob 有更高的顺子同花呢？让我们写一个例子：

```cpp
Case: Bob wins with a higher straight flush

Inputs:
 Alice: 3♠, 4♠, 5♠, 6♠, 7♠
 Bob: 4♣, 5♣, 6♣, 7♣, 8♣

Output:
 Bob wins with straight flush
```

是时候写另一个失败的测试了。

# 比较两个顺子同花（续）

现在写这个测试已经相当明显了：

```cpp
TEST_CASE("Alice and Bob have straight flushes but Bob wins with higher 
    straight flush"){
        Hand aliceHand = {"3♠", "4♠", "5♠", "6♠", "7♠"};
        Hand bobHand;

        SUBCASE("3 based straight flush"){
            bobHand = {"4♣", "5♣", "6♣", "7♣", "8♣"};
    };

    CAPTURE(bobHand);

    auto result = comparePokerHands(aliceHand, bobHand);

    CHECK_EQ("Bob wins with straight flush", result);
}
```

测试再次失败了，因为我们的实现假设当 Alice 和 Bob 都有顺子同花时，Alice 总是赢。也许是时候检查哪个是它们中最高的顺子同花了。

为此，我们需要再次写下一些情况并进行 TDD 循环。我将再次快进到实现。我们最终得到了以下的辅助函数，用于比较两个顺子同花。如果第一手牌有更高的顺子同花，则返回`1`，如果两者相等，则返回`0`，如果第二手牌有更高的顺子同花，则返回`-1`：

```cpp
auto compareStraightFlushes = [](const Hand& first, const Hand& second){
    int firstHandValue = allValuesInOrder(first).front();
    int secondHandValue = allValuesInOrder(second).front();
    if(firstHandValue > secondHandValue) return 1;
    if(secondHandValue > firstHandValue) return -1;
    return 0;
};
```

通过改变我们的实现，我们可以让测试通过：

```cpp
auto comparePokerHands = [](const Hand& aliceHand, const Hand& bobHand){
    if(isStraightFlush(bobHand) && isStraightFlush(aliceHand)){
        int whichIsHigher = compareStraightFlushes(aliceHand, bobHand);
        if(whichIsHigher == 1) return "Alice wins with straight flush";
        if(whichIsHigher == -1) return "Bob wins with straight flush";
    }

    if(isStraightFlush(bobHand)) {
        return "Bob wins with straight flush";
    }

    return "Alice wins with straight flush";
};
```

这让我们留下了最后一种情况——平局。测试再次非常明确：

```cpp
TEST_CASE("Draw due to equal straight flushes"){
    Hand aliceHand;
    Hand bobHand;

    SUBCASE("3 based straight flush"){
        aliceHand = {"3♠", "4♠", "5♠", "6♠", "7♠"};
    };

    CAPTURE(aliceHand);
    bobHand = aliceHand;

    auto result = comparePokerHands(aliceHand, bobHand);

    CHECK_EQ("Draw", result);
}
```

而且实现的改变非常直接：

```cpp
auto comparePokerHands = [](Hand aliceHand, Hand bobHand){
    if(isStraightFlush(bobHand) && isStraightFlush(aliceHand)){
        int whichIsHigher = compareStraightFlushes(aliceHand, bobHand);
        if(whichIsHigher == 1) return "Alice wins with straight flush";
        if(whichIsHigher == -1) return "Bob wins with straight flush";
        return "Draw";
    }

    if(isStraightFlush(bobHand)) {
        return "Bob wins with straight flush";
    }

    return "Alice wins with straight flush";
};
```

这不是最漂亮的函数，但它通过了我们所有的顺子同花比较测试。我们肯定可以将它重构为更小的函数，但我会在这里停下来，因为我们已经达到了我们的目标——使用 TDD 和 DDT 设计了不止一个纯函数。

# 总结

在本章中，你学会了如何编写单元测试，如何编写数据驱动测试，以及如何将数据驱动测试与 TDD 结合起来设计纯函数。

TDD 是有效软件开发的核心实践之一。虽然有时可能看起来奇怪和违反直觉，但它有一个强大的优势——每隔几分钟，你都有一个可以演示的工作内容。通过测试通过不仅是一个演示点，而且也是一个保存点。如果在尝试重构或实现下一个测试时发生任何错误，你总是可以回到上一个保存点。我发现这种实践在 C++中更有价值，因为有很多事情可能会出错。事实上，我自第三章 *深入了解 Lambda*以来，都是采用 TDD 方法编写的所有代码。这非常有帮助，因为我知道我的代码是有效的——在没有这种方法的情况下编写技术书籍时，这是相当困难的。我强烈建议你更深入地了解 TDD 并亲自实践；这是你成为专家的唯一途径。

函数式编程与 TDD 完美契合。当将其与命令式面向对象的代码一起使用时，我们经常需要考虑到变异，这使事情变得更加困难。通过纯函数和数据驱动的测试，添加更多的测试实践变得尽可能简单，并允许我们专注于实现。在函数操作的支持下，在许多情况下使测试通过变得更容易。我个人发现这种组合非常有益；我希望你也会觉得同样有用。

现在是时候向前迈进，重新审视软件设计的另一个部分——设计模式。它们在函数式编程中会发生变化吗？（剧透警告——实际上它们变得简单得多。）这是我们将在下一章讨论的内容。
