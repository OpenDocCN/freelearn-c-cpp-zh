

# 第七章：测试设置和拆卸

你是否曾经在一个项目中工作过，需要首先准备你的工作区域？一旦准备好，你就可以开始工作了。然后，过了一段时间，你需要清理你的区域。也许你会用这个区域做其他事情，不能只是让你的项目闲置在那里，否则会妨碍你。

有时候，测试可能就像那样。它们可能不占用表空间，但有时它们在运行之前可能需要设置环境或准备一些其他结果。也许一个测试确保某些数据可以被删除。数据首先存在是有意义的。测试是否应该负责创建它试图删除的数据？最好是将数据创建封装在其自己的函数中。但如果你需要测试几种不同的删除数据方式呢？每个测试是否都应该创建数据？它们可以调用相同的设置函数。

如果多个测试需要执行类似的前期准备和后期清理工作，不仅将相同的代码写入每个测试是冗余的，而且还会隐藏测试的真实目的。

本章将允许测试运行准备和清理代码，以便它们可以专注于需要测试的内容。准备工作称为**设置**。清理工作称为**拆卸**。

我们遵循 TDD（测试驱动开发）方法，这意味着我们将从一些简单的测试开始，让它们工作，然后增强它们以实现更多功能。

初始时，我们将让测试运行设置代码，然后在结束时进行拆卸。多个测试可以使用相同的设置和拆卸，但设置和拆卸将针对每个测试单独运行。

一旦这个功能工作，我们将增强设计，让一组测试在测试组之前和之后只运行一次共享的设置和拆卸代码。

在本章中，我们将涵盖以下主要主题：

+   支持测试设置和拆卸

+   为多个测试增强测试设置和拆卸

+   处理设置和拆卸过程中的错误

到本章结束时，测试将能够拥有单独的设置和拆卸代码，以及封装测试组的设置和拆卸代码。

# 技术要求

本章中的所有代码都使用标准 C++，它基于任何现代 C++ 20 或更高版本的编译器和标准库。代码基于前几章并继续发展。

你可以在以下 GitHub 仓库中找到本章的所有代码：

[`github.com/PacktPublishing/Test-Driven-Development-with-CPP`](https://github.com/PacktPublishing/Test-Driven-Development-with-CPP)

# 支持测试设置和拆卸

为了支持测试设置和拆卸，我们只需要安排在测试开始前运行一些代码，并在测试完成后运行一些代码。对于设置，我们可能只需在测试开始附近调用一个函数。设置实际上不必在测试之前运行，只要它在测试需要设置结果之前运行即可。我的意思是，单元测试库实际上不需要在测试开始前运行设置。只要测试本身在测试开始时运行设置，我们就能得到相同的结果。这将是最简单的解决方案。但这并不是一个全新的解决方案。测试已经可以调用其他函数。

我看到简单地声明一个独立函数并在测试开始时调用它的最大问题是意图可能会丢失。我的意思是，测试作者必须确保在测试中调用的函数明确定义为设置函数。因为函数可以有任意的名称，除非有一个好名字，仅仅调用一个函数是不够的，无法识别出设置的意图。

关于拆卸呢？这也可以是一个简单的函数调用吗？因为拆卸代码应该在测试结束时始终运行，测试作者必须确保即使在抛出异常的情况下，拆卸也会运行。

由于这些原因，测试库应该提供一些帮助来进行设置和拆卸。我们需要决定的是帮助的程度以及这种帮助将如何体现。我们的目标是保持测试简单，并确保处理所有边缘情况。

按照在*第三章**中首次解释的 TDD 方法，即《TDD 过程》，我们应该做以下事情：

+   首先，思考理想解决方案应该是什么。

+   编写一些使用该解决方案的测试，以确保它将满足我们的期望。

+   构建项目并修复构建错误，无需担心测试是否通过。

+   实现一个基本的解决方案，并通过测试。

+   提高解决方案并改进测试。

帮助设置和拆卸的一个选项可能是向`TEST`和`TEST_EX`宏添加新参数。这将使设置和拆卸部分成为测试声明的一部分。但这是必要的吗？如果可能，我们应该避免依赖于这些宏。如果可以避免，它们已经足够复杂，无需添加更多功能。修改宏通常不需要用于测试设置和拆卸。

另一个可能的解决方案是在`TestBase`类中创建一个方法，就像我们在*第三章**《TDD 过程》*中设置预期失败原因时做的那样。这会起作用吗？为了回答这个问题，让我们思考设置和拆卸代码应该做什么。

设置应该为测试做好准备。这很可能意味着测试将需要引用设置代码准备的数据或资源，例如文件。如果测试得不到可以使用的东西，可能看起来设置并不重要，但谁知道呢？也许设置做了与测试代码无关但未注意到的相关操作。我主要想说的是，设置代码几乎可以执行任何事情。它可能需要自己的参数来定制。或者，它可能能够在没有任何输入的情况下运行。它可能生成测试直接使用的内容。或者，它可能在幕后以对测试有用的方式工作，但对测试代码来说是未知的。

此外，拆解可能需要参考之前设置的内容，以便能够撤销。或者，拆解可能只是清理一切，而不关心它们来自何处。

在`TestBase`中调用方法以注册和运行设置和拆解似乎可能会使与测试代码的交互变得更加复杂，因为我们需要一种方式来共享设置结果。我们真正想要的只是运行设置，获取设置提供的内容，然后在测试结束时运行拆解。有一种简单的方法可以实现这一点，允许设置、拆解和其余测试代码之间所需的任何交互。

让我们从在`tests`文件夹中创建一个新的`.cpp`文件开始，命名为`Setup.cpp`。项目结构将如下所示：

```cpp
MereTDD project root folder
    Test.h
    tests folder
        main.cpp
        Confirm.cpp
        Creation.cpp
        Setup.cpp
```

这里有一个在`Setup.cpp`中的测试，我们可以用它开始：

```cpp
TEST_EX("Test will run setup and teardown code", int)
{
    int id = createTestEntry();
    // If this was a project test, it might be called
    // "Updating empty name throws". And the type thrown
    // would not be an int.
    updateTestEntryName(id, "");
    deleteTestEntry(id);
}
```

测试使用三个函数：`createTestEntry`、`updateTestEntryName`和`deleteTestEntry`。注释解释了测试可能被调用的名称以及如果这是一个实际项目的测试而不是测试库的测试，它将执行的操作。测试的思路是调用`createTestEntry`来设置一些数据，尝试使用空字符串更新名称以确保不允许这样做，然后调用`deleteTestEntry`来拆解测试开始时创建的数据。你可以看到设置提供了一个名为`id`的标识符，这是测试和拆解所需的。

测试期望`updateTestEntryName`的调用因名称为空而失败，这将导致抛出异常。我们在这里只是抛出一个整数，但在实际项目中，异常类型通常是其他类型。异常将导致跳过拆解调用`deleteTestEntry`。

此外，如果需要，测试可以使用确认来验证其结果。如果确认失败，也会抛出异常。我们需要确保在所有情况下都运行拆解代码。目前，它总是会跳过，因为整个测试的目的就是期望从`updateTestEntryName`抛出异常。但其他测试如果确认失败，可能仍然会跳过拆解。

即使解决了`deleteTestEntry`没有被调用的这个问题，测试仍然不够清晰。真正要测试的是什么？在这个测试中，唯一应该突出显示为测试意图的是对`updateTestEntryName`的调用。对`createTestEntry`和`deleteTestEntry`的调用只是隐藏了测试的真实目的。如果我们添加一个`try/catch`块来确保`deleteTestEntry`被调用，那么真实目的只会被进一步隐藏。

测试中的三个函数是那种在项目中可能会找到的函数类型。我们没有单独的项目，所以它们可以放在`Setup.cpp`中，因为它们是我们目的的辅助函数。它们看起来是这样的：

```cpp
#include "../Test.h"
#include <string_view>
int createTestEntry ()
{
    // If this was real code, it might open a
    // connection to a database, insert a row
    // of data, and return the row identifier.
    return 100;
}
void updateTestEntryName (int /*id*/, std::string_view name)
{
    if (name.empty())
    {
        throw 1;
    }
    // Real code would proceed to update the
    // data with the new name.
}
void deleteTestEntry (int /*id*/)
{
    // Real code would use the id to delete
    // the temporary row of data.
}
```

`id`参数名称被注释掉了，因为辅助函数没有使用它们。

我们可以在类的构造函数和析构函数中包装`createTestEntry`和`deleteTestEntry`的调用。这有助于简化测试并确保调用拆解代码。新的测试看起来像这样：

```cpp
TEST_EX("Test will run setup and teardown code", int)
{
    TempEntry entry;
    // If this was a project test, it might be called
    // "Updating empty name throws". And the type thrown
    // would not be an int.
    updateTestEntryName(entry.id(), "");
}
```

`TempEntry`类包含设置和拆解调用以及测试和拆解所需的标识符。它可以在三个辅助方法之后直接放入`Setup.cpp`中：

```cpp
class TempEntry
{
public:
    TempEntry ()
    {
        mId = createTestEntry();
    }
    ~TempEntry ()
    {
        deleteTestEntry(mId);
    }
    int id ()
    {
        return mId;
    }
private:
    int mId;
};
```

编写这样的类是确保实例超出作用域时代码被执行的绝佳方式，我们可以用它来确保在测试结束时总是运行拆解代码。它很简单，可以维护自己的状态，例如标识符。此外，它只需要在测试开始时创建一个实例的单行代码，这样就不会分散测试试图做的事情。

当库代码不能满足特定需求时，你可以随时采取这种做法。但是，测试库是否有方法可以使这变得更好？

我见过一些类，允许你将 lambda 或函数传递给构造函数，执行类似操作。构造函数会立即调用第一个函数，并在实例被销毁时调用第二个函数。这就像`TempEntry`类所做的那样，除了一个细节。`TempEntry`还管理拆解代码所需的身份。我想到的所有 lambda 解决方案都没有像专门为此目的编写的类（如`TempEntry`）那样干净。但也许我们还可以再改进一点。

`TempEntry`的问题在于，它不清楚什么是设置和什么是拆解。在测试中，第一条创建`TempEntry`类的语句与设置和拆解有什么关系也不清楚。当然，稍微研究一下会让你意识到设置在构造函数中，拆解在析构函数中。如果我们可以有名为`setup`和`teardown`的方法，并且测试本身清楚地标识了正在运行的设置和拆解代码的使用，那就太好了。

一个想到的解决方案可能是调用虚拟`setup`和`teardown`方法的基本类。但我们不能使用正常的继承，因为我们需要在构造函数和析构函数中调用它们。相反，我们可以使用一个称为*基于策略的设计*的设计模式。

一个*策略类*实现了一个或多个派生类将使用的方法。派生类使用的方法被称为*策略*。这就像继承的反向。我们将通过如下修改将`TempEntry`类转变为一个实现`setup`和`teardown`方法的策略类：

```cpp
class TempEntry
{
public:
    void setup ()
    {
        mId = createTestEntry();
    }
    void teardown ()
    {
        deleteTestEntry(mId);
    }
    int id ()
    {
        return mId;
    }
private:
    int mId;
};
```

唯一真正的变化是将构造函数转变为`setup`方法，将析构函数转变为`teardown`方法。这仅仅是因为我们之前使用这些方法来做这项工作。现在我们有一个清晰且易于理解的类。但我们如何使用它呢？我们不再有在类构造时自动运行的设置代码和在析构时运行的清理代码。我们需要创建另一个类，这个类可以放在`Test.h`中，因为它将用于所有测试的设置和清理需求。在`Test.h`中`MereTDD`命名空间内添加如下模板类：

```cpp
template <typename T>
class SetupAndTeardown : public T
{
public:
    SetupAndTeardown ()
    {
        T::setup();
    }
    ~SetupAndTeardown ()
    {
        T::teardown();
    }
};
```

`SetupAndTeardown`类是我们将`setup`和`teardown`的调用与构造函数和析构函数重新连接的地方。只要那个类实现了两个`setup`和`teardown`方法，你可以使用任何你想要的类作为策略。还有一个很好的好处是，由于公有继承，你可以访问策略类中定义的其他方法。我们使用这一点仍然能够调用`id`方法。基于策略的设计让你能够扩展接口到你需要的样子，只要实现了策略。在这个例子中，策略只是两个`setup`和`teardown`方法。

使用基于策略的设计，特别是关于继承的问题，还有一个其他方面，那就是这种模式与面向对象设计的“is-a”关系相悖。如果我们以正常的方式使用公有继承，那么我们可以说`SetupAndTeardown`是`TempEntry`类。在这种情况下，这显然没有意义。没关系，因为我们不会使用这种模式来创建可以互相替换的实例。我们使用公有继承只是为了能够在策略类内部调用诸如`id`这样的方法。

现在我们已经拥有了所有这些，测试看起来会是什么样子？现在测试可以使用`SetupAndTeardown`类，如下所示：

```cpp
TEST_EX("Test will run setup and teardown code", int)
{
    MereTDD::SetupAndTeardown<TempEntry> entry;
    // If this was a project test, it might be called
    // "Updating empty name throws". And the type thrown
    // would not be an int.
    updateTestEntryName(entry.id(), "");
}
```

这是因为以下列出的原因而有一个很大的改进：

+   在测试的开始阶段就很清楚，测试代码中附带了设置代码和清理代码。

+   清理代码将在测试结束时运行，我们不需要通过 try/catch 块来复杂化测试代码。

+   我们不需要将`setup`和`teardown`的调用与测试的其他部分混合。

+   我们可以通过我们编写的诸如`id`方法这样的方法与设置结果进行交互。

无论何时需要在测试中编写设置和/或拆除代码，你只需要编写一个实现`setup`和`teardown`方法的类。如果其中一个方法不需要执行任何操作，则可以留空该方法。然而，两个方法都必须存在，因为它们是策略。实现策略方法就是创建策略类。然后，添加一个使用策略类作为模板参数的`MereTDD:SetupAndTeardown`实例。测试应该在测试开始时声明`SetupAndTeardown`实例，以便从这种设计中获得最大好处。

虽然我们可以像这样声明在每个测试开始和结束时运行的设置和拆除代码，但我们需要一个不同的解决方案来共享设置和拆除代码，以便设置在测试组之前运行，而拆除代码在测试组完成后运行。下一节将增强设置和拆除功能以满足这一扩展需求。

# 多个测试的测试设置和拆除增强

现在我们有了为测试设置事物并在测试后清理的能力，我们可以在设置中准备测试所需的临时数据，以便运行测试，然后在测试运行后拆除设置中的临时数据。如果有许多不同的测试使用此类数据，它们可以各自创建类似的数据。

但如果我们需要为整个测试组设置某些东西，然后在所有测试完成后将其拆除呢？我指的是在多个测试中保持不变的东西。对于临时数据，可能我们需要准备一个地方来存储数据。如果数据存储在数据库中，那么现在是打开数据库并确保必要的表已经准备好以存储每个测试将创建的数据的好时机。甚至数据库连接本身也可以保持打开状态，供测试使用。一旦所有数据测试完成，那么拆除代码就可以关闭数据库。

这种场景适用于许多不同的情况。如果你正在测试与硬盘上的文件相关的东西，那么你可能想确保适当的目录已经准备好，以便创建文件。这些目录可以在任何文件测试开始之前设置，而测试只需要担心创建它们将要测试的文件。

如果你正在测试一个网络服务，那么在测试开始前确保测试有一个有效的和经过身份验证的登录可能是有意义的。可能没有必要让每个测试每次都重复登录步骤。除非，当然，这是测试的目的。

这里的主要思想是，虽然让一些代码在每个测试中作为设置和拆除运行是好的，但也可以让不同的设置和拆除代码只为测试组运行一次。这正是本节将要探讨的内容。

我们将称由共同的设置和拆解代码关联的一组测试为*测试套件*。测试不必属于测试套件，但我们将创建一个内部和隐藏的测试套件来分组所有没有特定套件的单独测试。

我们能够在单个测试中完全添加设置和拆解代码，因为测试中的设置和拆解代码就像调用几个函数一样。然而，为了支持测试套件的设置和拆解，我们可能需要在测试之外做一些工作。我们需要确保设置代码在相关测试运行之前运行。然后，在所有相关测试完成后运行拆解代码。

包含并运行所有测试的测试项目应该能够支持多个测试套件。这意味着测试需要一种方式来识别它所属的测试套件。此外，我们还需要一种方式来声明测试套件的设置和拆解代码。

这个想法是这样的：我们将声明并编写一些代码作为测试套件的设置。或者，如果我们只需要拆解代码，也许我们可以让设置代码是可选的。然后，我们将声明并编写一些拆解代码。拆解代码也应该可选。无论是设置、拆解，还是两者都要定义，才能有一个有效的测试套件。每个测试都需要一种方式来识别它所属的测试套件。当运行项目中的所有测试时，我们需要按照正确的顺序运行它们，以便测试套件设置首先运行，然后是测试套件中的所有测试，最后是测试套件的拆解。

我们将如何识别测试套件？测试库会自动为每个测试生成唯一的名称，并且这些名称对测试作者来说是隐藏的。我们也可以为测试套件使用名称，但让测试作者指定每个测试套件的名称。这似乎是可理解的，并且应该足够灵活以处理任何情况。我们将让测试作者为每个测试套件提供一个简单的字符串名称。

当处理名称时，一个经常出现的边缘情况是处理重复名称的问题。我们需要做出决定。我们可以检测重复名称并停止测试以显示错误，或者我们可以堆叠设置和拆解，以便它们都运行。

我们是否在单个测试的设置和拆解中遇到了这个问题？实际上并没有，因为设置和拆解没有命名。但是，如果一个测试声明了多个`SetupAndTeardown`实例会发生什么？我们实际上在前一节中没有考虑这种可能性。在一个测试中，它可能看起来像这样：

```cpp
TEST("Test will run multiple setup and teardown code")
{
    MereTDD::SetupAndTeardown<TempEntry> entry1;
    MereTDD::SetupAndTeardown<TempEntry> entry2;
    // If this was a project test, it might need
    // more than one temporary entry. The TempEntry
    // policy could either create multiple data records
    // or it is easier to just have multiple instances
    // that each create a single data entry.
    updateTestEntryName(entry1.id(), "abc");
    updateTestEntryName(entry2.id(), "def");
}
```

拥有多个设置和清理实例的能力是很有趣的，这应该有助于简化并允许你重用设置和清理代码。而不是创建执行许多操作的特定设置和清理策略类，这将允许它们堆叠起来，以便更加专注。也许一个测试只需要在最后设置和清理单个数据集，而另一个测试则需要两个。而不是创建两个不同的策略类，这种能力将允许第一个测试声明一个单独的 `SetupAndTeardown` 实例，而第二个测试通过声明两个来重用相同的策略类。

既然我们现在允许单个测试设置和清理代码的组合，为什么不允许测试套件的设置和清理代码的组合呢？这似乎是合理的，甚至可能简化测试库代码。这是如何实现的呢？

好吧，既然我们已经了解了这种能力，我们就可以为此进行规划，并可能避免编写检测和抛出错误的代码。如果我们注意到两个或更多具有相同名称的测试套件设置定义，我们可以将它们添加到集合中，而不是将这种情况视为一个特殊错误情况。

如果我们确实有多个具有相同名称的设置和清理定义，我们就不应该依赖于它们之间的任何特定顺序。它们可以像测试一样被分割到不同的 `.cpp` 文件中。这将简化代码，因为我们可以在找到它们时将它们添加到集合中，而不用担心特定的顺序。

下一步要考虑的是如何定义测试套件的设置和清理代码。它们可能不能是简单的函数，因为它们需要与测试库进行注册。注册是必要的，这样当测试提供一个套件名称时，我们将知道这个名称的含义。注册看起来与测试进行自我注册的方式非常相似。我们应该能够为套件名称添加一个额外的字符串。此外，即使测试不是特定测试套件的一部分，它们也需要这个新的套件名称。我们将使用空套件名称为想要在测试套件之外运行的测试。

注册时需要让测试自己使用套件名称进行注册，即使该套件名称的设置和清理代码尚未注册。这是因为测试可以定义在多个 `.cpp` 文件中，我们无法知道初始化代码将按什么顺序注册测试和测试套件的设置和清理代码。

还有一个更重要的要求。我们有一种方式可以与单个测试设置和清理代码中的设置结果进行交互。在测试套件设置和清理中，我们也将需要这种能力。假设测试套件设置需要打开一个数据库连接，该连接将被套件中的所有测试使用。测试需要某种方式来了解这个连接。此外，如果测试套件清理想要关闭连接，它也需要了解这个连接。也许测试套件设置还需要创建一个数据库表。测试将需要该表的名字以便使用它。

让我们在`Setup.cpp`中创建几个辅助函数，以模拟创建和删除表的操作。它们应该看起来像这样：

```cpp
#include <string>
#include <string_view>
std::string createTestTable ()
{
    // If this was real code, it might open a
    // connection to a database, create a temp
    // table with a random name, and return the
    // table name.
    return "test_data_01";
}
void dropTestTable (std::string_view /*name*/)
{
    // Real code would use the name to drop
    // the table.
}
```

然后，在`Setup.cpp`文件中，我们可以使我们的第一个测试套件设置和清理看起来像这样：

```cpp
class TempTable
{
public:
    void setup ()
    {
        mName = createTestTable();
    }
    void teardown ()
    {
        dropTestTable(mName);
    }
    std::string tableName ()
    {
        return mName;
    }
private:
    std::string mName;
};
```

这看起来非常像上一节中用来定义`setup`和`teardown`方法以及提供访问由设置代码提供的任何额外方法或数据的策略类。这是因为这也将是一个策略类。我们不妨让策略保持一致，无论设置和清理代码是用于单个测试还是整个测试套件。

当我们声明一个测试只有设置和清理代码时，我们声明了一个使用策略类的特化`MereTDD::SetupAndTeardown`的实例。这足以立即运行设置代码并确保在测试结束时运行清理代码。但为了获取其他信息，给`SetupAndTeardown`实例一个名字是很重要的。设置和清理代码完全定义并通过本地命名实例可访问。

然而，在测试套件设置和清理中，我们需要将策略类的实例放入容器中。容器希望它内部的所有内容都是单一类型。设置和清理实例不能再是测试中的简单本地命名变量。然而，我们仍然需要一个命名类型，因为这是测试套件中的测试访问设置代码提供的资源的方式。

我们需要弄清楚两件事。第一是创建测试套件设置和清理代码实例的位置。第二是如何协调容器需要所有内容都是单一类型的需求与测试能够引用特定类型的命名实例的需求，这些实例可能因策略类而异。

第一个问题最容易解决，因为我们需要考虑生命周期和可访问性。测试套件的设置和销毁实例需要在测试套件内的多个测试中存在并有效。它们不能作为单个测试中的局部变量存在。它们需要在一个将保持对多个测试有效的地方。它们可以是`main`内部的局部实例——这将解决生命周期问题。但这样它们就只能被`main`访问。测试套件的设置和销毁实例需要是全局的。只有这样，它们才能在整个测试应用程序期间存在，并且可以被多个测试访问。

对于第二个问题，我们首先将声明一个接口，该接口将用于存储所有测试套件的设置和销毁实例。测试库在需要运行设置和销毁代码时也将使用此相同的接口。测试库需要将所有内容视为相同，因为它对特定的策略类一无所知。

我们稍后会回到所有这些。在我们走得太远之前，我们需要考虑我们的预期使用方式。我们仍然遵循 TDD 方法，虽然考虑所有需求和可能的情况是好的，但我们已经足够深入，可以有一个关于测试套件设置和销毁使用的良好想法。我们甚至已经有了准备好的策略类和定义。将以下内容添加到`Setup.cpp`中，作为我们将要实现的预期使用：

```cpp
MereTDD::TestSuiteSetupAndTeardown<TempTable>
gTable1("Test suite setup/teardown 1", "Suite 1");
MereTDD::TestSuiteSetupAndTeardown<TempTable>
gTable2("Test suite setup/teardown 2", "Suite 1");
TEST_SUITE("Test part 1 of suite", "Suite 1")
{
    // If this was a project test, it could use
    // the table names from gTable1 and gTable2.
    CONFIRM("test_data_01", gTable1.tableName());
    CONFIRM("test_data_01", gTable2.tableName());
}
TEST_SUITE_EX("Test part 2 of suite", "Suite 1", int)
{
    // If this was a project test, it could use
    // the table names from gTable1 and gTable2.
    throw 1;
}
```

有几点需要通过前面的代码进行解释。你可以看到它声明了两个`MereTDD::TestSuiteSetupAndTeardown`的实例，每个实例都专门使用`TempTable`策略类。这些是具有特定类型的全局变量，因此测试将能够看到它们并使用策略类中的方法。如果你想的话，可以为每个实例使用不同的策略类。或者，如果你使用相同的策略类，那么通常应该有一些差异。否则，为什么有两个实例？对于创建临时表，正如这个示例所示，每个表可能都有一个唯一的随机名称，并且能够使用相同的策略类。

构造函数需要两个字符串。第一个是设置和销毁代码的名称。我们将测试套件的设置和销毁代码视为一个测试本身。我们将测试套件的设置和销毁通过或失败的结果包含在测试应用程序摘要中，并用构造函数提供的名称来标识它。第二个字符串是测试套件的名称。这可以是任何内容，但不能是空字符串。我们将空测试套件名称视为不属于任何测试套件的测试的特殊值。

在这个示例中，`TestSuiteSetupAndTeardown`的两个实例使用相同的套件名称。这是可以接受的，也是支持的，因为我们之前决定。任何有多个具有相同名称的测试套件设置和销毁实例时，它们都将运行在测试套件开始之前。

为什么使用新的`TestSuiteSetupAndTeardown`测试库类而不是重用现有的`SetupAndTeardown`类将在稍后变得清晰。它需要合并一个通用接口与策略类。新的类也清楚地表明，这个设置和清理是为测试套件而设的。

然后是测试。我们需要一个新的宏`TEST_SUITE`，以便可以指定测试套件名称。除了测试套件名称外，该宏的行为几乎与现有的`TEST`宏相同。我们还需要一个新的宏来表示属于测试套件且期望异常的测试。我们将称其为`TEST_SUITE_EX`；它的行为类似于`TEST_EX`，但增加了测试套件名称。

在`Test.h`中需要进行许多更改以支持测试套件。大多数更改都与测试的注册和运行方式相关。我们有一个名为`TestBase`的测试基类，它通过将`TestBase`指针推送到向量中来执行注册。由于我们还需要注册测试套件的设置和清理代码，并按测试套件分组运行测试，因此我们需要对此进行更改。我们将保持`TestBase`作为所有测试的基类。但现在它也将成为测试套件的基类。

测试集合需要更改为映射，以便可以通过测试套件名称访问测试。没有测试套件的测试仍然有一个套件名称。它将只是空的。此外，我们还需要通过套件名称查找测试套件的设置和清理代码。我们需要两个集合：一个映射用于测试，一个映射用于测试套件的设置和清理代码。由于我们需要将现有的注册代码从`TestBase`中重构出来，我们将创建一个名为`Test`的类，用于测试，以及一个名为`TestSuite`的类，用于测试套件的设置和清理代码。`Test`和`TestSuite`类都将从`TestBase`派生。

将使用现有的`getTests`函数来访问映射，该函数将被修改为使用映射并添加一个新的`getTestSuites`函数。首先，在`Test.h`的顶部包含一个映射：

```cpp
#include <map>
#include <ostream>
#include <string_view>
#include <vector>
```

然后，在更下面，将前向声明`TestBase`类和实现`getTests`函数的部分修改如下：

```cpp
class Test;
class TestSuite;
inline std::map<std::string, std::vector<Test *>> & getTests ()
{
    static std::map<std::string, std::vector<Test *>> tests;
    return tests;
}
inline std::map<std::string, std::vector<TestSuite *>> & getTestSuites ()
{
    static std::map<std::string,            std::vector<TestSuite *>> suites;
    return suites;
}
```

每个映射的键将是测试套件名称的字符串。值将是`Test`或`TestSuite`指针的向量。当我们注册测试或测试套件设置和清理代码时，我们将通过测试套件名称进行注册。对于任何测试套件名称的第一个注册，需要设置一个空向量。一旦向量已经设置，测试就可以像之前一样推送到向量的末尾。测试套件的设置和清理代码也将执行相同操作。为了使这个过程更容易，我们将在`Test.h`中`getTestSuites`函数之后创建几个辅助方法：

```cpp
inline void addTest (std::string_view suiteName, Test * test)
{
    std::string name(suiteName);
    if (not getTests().contains(name))
    {
        getTests().try_emplace(name, std::vector<Test *>());
    }
    getTests()[name].push_back(test);
}
inline void addTestSuite (std::string_view suiteName, TestSuite * suite)
{
    std::string name(suiteName);
    if (not getTestSuites().contains(name))
    {
        getTestSuites().try_emplace(name,            std::vector<TestSuite *>());
    }
    getTestSuites()[name].push_back(suite);
}
```

接下来是重构后的 `TestBase` 类，它已经被修改以添加测试套件名称，停止进行测试注册，移除预期失败原因，并移除运行代码。现在 `TestBase` 类将只包含测试和测试套件设置和清理代码之间的公共数据。修改后的类如下：

```cpp
class TestBase
{
public:
    TestBase (std::string_view name, std::string_view suiteName)
    : mName(name),
      mSuiteName(suiteName),
      mPassed(true),
      mConfirmLocation(-1)
    { }
    virtual ~TestBase () = default;
    std::string_view name () const
    {
        return mName;
    }
    std::string_view suiteName () const
    {
        return mSuiteName;
    }
    bool passed () const
    {
        return mPassed;
    }
    std::string_view reason () const
    {
        return mReason;
    }
    int confirmLocation () const
    {
        return mConfirmLocation;
    }
    void setFailed (std::string_view reason,          int confirmLocation = -1)
    {
        mPassed = false;
        mReason = reason;
        mConfirmLocation = confirmLocation;
    }
private:
    std::string mName;
    std::string mSuiteName;
    bool mPassed;
    std::string mReason;
    int mConfirmLocation;
};
```

从之前的 `TestBase` 类中提取的功能现在进入了一个新的派生类，称为 `Test`，看起来是这样的：

```cpp
class Test : public TestBase
{
public:
    Test (std::string_view name, std::string_view suiteName)
    : TestBase(name, suiteName)
    {
        addTest(suiteName, this);
    }
    virtual void runEx ()
    {
        run();
    }
    virtual void run () = 0;
    std::string_view expectedReason () const
    {
        return mExpectedReason;
    }
    void setExpectedFailureReason (std::string_view reason)
    {
        mExpectedReason = reason;
    }
private:
    std::string mExpectedReason;
};
```

`Test` 类更短，因为现在很多基本信息都存储在 `TestBase` 类中。此外，我们曾经有一个 `TestExBase` 类，需要稍作修改。现在它将被称为 `TestEx`，看起来是这样的：

```cpp
template <typename ExceptionT>
class TestEx : public Test
{
public:
    TestEx (std::string_view name,
        std::string_view suiteName,
        std::string_view exceptionName)
    : Test(name, suiteName), mExceptionName(exceptionName)
    { }
    void runEx () override
    {
        try
        {
            run();
        }
        catch (ExceptionT const &)
        {
            return;
        }
        throw MissingException(mExceptionName);
    }
private:
    std::string mExceptionName;
};
```

`TestEx` 类真正改变的是名称和基类名称。

现在，我们可以进入新的 `TestSuite` 类。这将是一个将被存储在映射中并作为测试库运行设置和清理代码的公共接口的通用接口。

这个类看起来是这样的：

```cpp
class TestSuite : public TestBase
{
public:
    TestSuite (
        std::string_view name,
        std::string_view suiteName)
    : TestBase(name, suiteName)
    {
        addTestSuite(suiteName, this);
    }
    virtual void suiteSetup () = 0;
    virtual void suiteTeardown () = 0;
};
```

`TestSuite` 类没有像 `Test` 类那样的 `runEx` 方法。测试套件的存在是为了分组测试并为测试提供一个使用环境，因此编写预期会抛出异常的设置代码是没有意义的。测试套件的存在不是为了测试任何东西。它存在是为了准备一个或多个将使用 `suiteSetup` 方法准备好的资源的测试。同样，清理代码也不打算测试任何东西。`suiteTeardown` 代码只是用来清理设置的内容。如果在测试套件设置和清理过程中发生任何异常，我们希望知道它们。

此外，`TestSuite` 类没有像 `Test` 类那样的 `run` 方法，因为我们需要明确区分设置和清理。没有单一的代码块可以运行。现在有两个独立的代码块需要运行，一个在设置时运行，一个在清理时运行。因此，虽然 `Test` 类的设计是用来运行某些内容的，但 `TestSuite` 类的设计是用来准备一组测试，通过设置来准备环境，然后在测试后通过清理来清理环境。

你可以看到，`TestSuite` 构造函数通过调用 `addTestSuite` 来注册测试套件的设置和清理代码。

我们有一个名为 `runTests` 的函数，它目前遍历所有测试并运行它们。如果我们把运行单个测试的代码放在一个新的函数中，我们可以简化遍历所有测试并显示总结的代码。这将是重要的，因为在新设计中我们需要运行更多的测试。我们还需要运行测试套件的设置和清理代码。

这里有一个辅助函数来运行单个测试：

```cpp
inline void runTest (std::ostream & output, Test * test,
    int & numPassed, int & numFailed, int & numMissedFailed)
{
    output << "------- Test: "
        << test->name()
        << std::endl;
    try
    {
        test->runEx();
    }
    catch (ConfirmException const & ex)
    {
        test->setFailed(ex.reason(), ex.line());
    }
    catch (MissingException const & ex)
    {
        std::string message = "Expected exception type ";
        message += ex.exType();
        message += " was not thrown.";
        test->setFailed(message);
    }
    catch (...)
    {
        test->setFailed("Unexpected exception thrown.");
    }
    if (test->passed())
    {
        if (not test->expectedReason().empty())
        {
            // This test passed but it was supposed
            // to have failed.
            ++numMissedFailed;
            output << "Missed expected failure\n"
                << "Test passed but was expected to fail."
                << std::endl;
        }
        else
        {
            ++numPassed;
            output << "Passed"
                << std::endl;
        }
    }
    else if (not test->expectedReason().empty() &&
        test->expectedReason() == test->reason())
    {
        ++numPassed;
        output << "Expected failure\n"
            << test->reason()
            << std::endl;
    }
    else
    {
        ++numFailed;
        if (test->confirmLocation() != -1)
        {
            output << "Failed confirm on line "
                << test->confirmLocation() << "\n";
        }
        else
        {
            output << "Failed\n";
        }
        output << test->reason()
            << std::endl;
    }
}
```

上述代码几乎与`runTests`中的代码相同。对测试名称显示的开始输出进行了一些细微的更改，这是为了帮助区分测试与设置和清理代码。辅助函数还接收记录计数器的引用。

我们可以创建另一个辅助函数来运行设置和清理代码。此函数将执行几乎相同的设置和清理步骤。主要区别在于调用`TestSuite`指针的方法，即`suiteSetup`或`suiteTeardown`。辅助函数看起来如下：

```cpp
inline bool runSuite (std::ostream & output,
    bool setup, std::string const & name,
    int & numPassed, int & numFailed)
{
    for (auto & suite: getTestSuites()[name])
    {
        if (setup)
        {
            output << "------- Setup: ";
        }
        else
        {
            output << "------- Teardown: ";
        }
        output << suite->name()
            << std::endl;
        try
        {
            if (setup)
            {
                suite->suiteSetup();
            }
            else
            {
                suite->suiteTeardown();
            }
        }
        catch (ConfirmException const & ex)
        {
            suite->setFailed(ex.reason(), ex.line());
        }
        catch (...)
        {
            suite->setFailed("Unexpected exception thrown.");
        }
        if (suite->passed())
        {
            ++numPassed;
            output << "Passed"
                << std::endl;
        }
        else
        {
            ++numFailed;
            if (suite->confirmLocation() != -1)
            {
                output << "Failed confirm on line "
                    << suite->confirmLocation() << "\n";
            }
            else
            {
                output << "Failed\n";
            }
            output << suite->reason()
                << std::endl;
            return false;
        }
    }
    return true;
}
```

此函数比运行测试的辅助函数稍微简单一些。那是因为我们不需要担心遗漏的异常或预期的失败。它几乎做了同样的事情。它尝试运行设置或清理，捕获异常，并更新通过或失败的计数。

我们可以在`runTests`函数内部使用两个辅助函数`runTest`和`runSuite`，该函数需要按照以下方式修改：

```cpp
inline int runTests (std::ostream & output)
{
    output << "Running "
        << getTests().size()
        << " test suites\n";
    int numPassed = 0;
    int numMissedFailed = 0;
    int numFailed = 0;
    for (auto const & [key, value]: getTests())
    {
        std::string suiteDisplayName = "Suite: ";
        if (key.empty())
        {
            suiteDisplayName += "Single Tests";
        }
        else
        {
            suiteDisplayName += key;
        }
        output << "--------------- "
            << suiteDisplayName
            << std::endl;
        if (not key.empty())
        {
            if (not getTestSuites().contains(key))
            {
                output << "Test suite is not found."
                    << " Exiting test application."
                    << std::endl;
                return ++numFailed;
            }
            if (not runSuite(output, true, key,
                numPassed, numFailed))
            {
                output << "Test suite setup failed."
                    << " Skipping tests in suite."
                    << std::endl;
                continue;
            }
        }
        for (auto * test: value)
        {
            runTest(output, test,
                numPassed, numFailed, numMissedFailed);
        }
        if (not key.empty())
        {
            if (not runSuite(output, false, key,
                numPassed, numFailed))
            {
                output << "Test suite teardown failed."
                    << std::endl;
            }
        }
    }
    output << "-----------------------------------\n";
    output << "Tests passed: " << numPassed
        << "\nTests failed: " << numFailed;
    if (numMissedFailed != 0)
    {
        output << "\nTests failures missed: "                << numMissedFailed;
    }
    output << std::endl;
    return numFailed;
}
```

显示的初始语句显示了正在运行的测试套件数量。为什么代码查看测试的大小而不是测试套件的大小？嗯，那是因为测试包括了所有内容，包括有测试套件的测试以及在没有测试套件的测试，这些测试在名为`Single Tests`的虚构套件下运行。

此函数中的主要循环检查测试映射中的每个项目。之前，这些是测试的指针。现在每个条目都是一个测试套件名称和测试指针的向量。这使得我们可以遍历每个测试所属的测试套件已经分组的测试。空测试套件名称代表没有测试套件的单个测试。

如果我们找到一个非空测试套件，那么我们需要确保至少有一个条目与测试套件的名称匹配。如果没有，那么这是测试项目中的错误，并且不会运行进一步的测试。

如果测试项目注册了一个带有套件名称的测试，那么它还必须为该套件注册设置和清理代码。假设我们已经有该套件的设置和清理代码，每个注册的设置都会运行并检查是否有错误。如果设置测试套件时出现错误，则只会跳过该套件中的测试。

一旦运行完所有测试套件的设置代码，那么就会为该套件运行测试。

在运行完套件的所有测试之后，然后运行所有测试套件的清理代码。

要启用所有这些功能，还有两个部分。第一部分是`TestSuiteSetupAndTeardown`类，它位于`Test.h`中，紧接现有的`SetupAndTeardown`类之后。它看起来如下：

```cpp
template <typename T>
class TestSuiteSetupAndTeardown :
    public T,
    public TestSuite
{
public:
    TestSuiteSetupAndTeardown (
        std::string_view name,
        std::string_view suite)
    : TestSuite(name, suite)
    { }
    void suiteSetup () override
    {
        T::setup();
    }
    void suiteTeardown () override
    {
        T::teardown();
    }
};
```

这是一个在测试`.cpp`文件中使用，用于声明具有特定策略类的测试套件设置和清理实例的类。这个类使用多重继承来连接策略类和常见的`TestSuite`接口类。当`runSuite`函数通过指向`TestSuite`的指针调用`suiteSetup`或`suiteTeardown`时，这些虚拟方法最终会调用这个类中的重写方法。每个方法只是调用策略类中的`setup`或`teardown`方法来完成实际工作。

需要解释的最后一种更改是宏。我们需要两个额外的宏来声明一个属于测试套件但没有预期异常和有预期异常的测试。这些宏被称为`TEST_SUITE`和`TEST_SUITE_EX`。由于`TestBase`类的重构，现有的`TEST`和`TEST_EX`宏需要做些小的修改。现有的宏需要更新，以使用新的`Test`和`TestEx`类而不是`TestBase`和`TestExBase`。此外，现有的宏现在需要传递一个空字符串作为测试套件名称。我将在这里展示新的宏，因为它们非常相似，除了测试套件名称的不同。`TEST_SUITE`宏看起来是这样的：

```cpp
#define TEST_SUITE( testName, suiteName ) \
namespace { \
class MERETDD_CLASS : public MereTDD::Test \
{ \
public: \
    MERETDD_CLASS (std::string_view name, \
      std::string_view suite) \
    : Test(name, suite) \
    { } \
    void run () override; \
}; \
} /* end of unnamed namespace */ \
MERETDD_CLASS MERETDD_INSTANCE(testName, suiteName); \
void MERETDD_CLASS::run ()
```

现在的宏接受一个`suiteName`参数，该参数作为套件名称传递给实例。而`TEST_SUITE_EX`宏看起来是这样的：

```cpp
#define TEST_SUITE_EX( testName, suiteName, exceptionType ) \
namespace { \
class MERETDD_CLASS : public MereTDD::TestEx<exceptionType> \
{ \
public: \
    MERETDD_CLASS (std::string_view name, \
        std::string_view suite, \
        std::string_view exceptionName) \
    : TestEx(name, suite, exceptionName) \
    { } \
    void run () override; \
}; \
} /* end of unnamed namespace */ \
MERETDD_CLASS MERETDD_INSTANCE(testName, suiteName, #exceptionType); \
void MERETDD_CLASS::run ()
```

新的套件宏与修改后的非套件宏非常相似，所以我尝试将非套件宏改为使用空套件名称调用套件宏。但我无法找出如何将空字符串传递给另一个宏。这些宏很短，所以我保留了它们相似的代码。

这就是启用测试套件所需的所有更改。这些更改后，总结输出看起来略有不同。构建和运行测试项目会产生以下输出。因为它现在有 30 个测试，所以输出有点长。因此，我不会显示整个输出。第一部分看起来是这样的：

```cpp
Running 2 test suites
--------------- Suite: Single Tests
------- Test: Test will run setup and teardown code
Passed
------- Test: Test will run multiple setup and teardown code
Passed
------- Test: Test can be created
Passed
------- Test: Test that throws unexpectedly can be created
Expected failure
Unexpected exception thrown.
```

在这里，你可以看到有两个测试套件。一个是名为`Suite 1`的套件，包含两个测试以及套件设置和清理，另一个是无名的，包含所有不属于测试套件的其它测试。输出的一部分恰好是单个测试。总结输出的其余部分显示了测试套件，看起来是这样的：

```cpp
--------------- Suite: Suite 1
------- Setup: Test suite setup/teardown 1
Passed
------- Setup: Test suite setup/teardown 2
Passed
------- Test: Test part 1 of suite
Passed
------- Test: Test part 2 of suite
Passed
------- Teardown: Test suite setup/teardown 1
Passed
------- Teardown: Test suite setup/teardown 2
Passed
-----------------------------------
Tests passed: 30
Tests failed: 0
Tests failures missed: 1
```

每个测试套件都在总结输出中以套件名称开头，后面跟着该套件中的所有测试。对于实际的套件，你可以看到围绕所有测试的设置和清理。每个设置和清理都像测试一样运行。

最后，它显示的通过和失败计数与之前一样。

在本节中，我简要地解释了一些设置和清理代码的错误处理，但还需要更多。本节的主要目的是让设置和清理代码为测试套件工作。其中一部分需要一些错误处理，比如当测试声明它属于一个不存在的套件时应该怎么做。下一节将更深入地探讨这个问题。

# 处理设置和清理过程中的错误

错误可以在代码的任何地方找到，包括设置和清理代码中。那么，应该如何处理这些错误呢？在本节中，您将看到处理设置和清理代码中的错误没有唯一的方法。更重要的是，您应该意识到后果，以便您可以编写更好的测试。

让我们从开始的地方开始。我们已经避开了与多个设置和清理声明相关的一类问题。我们决定简单地允许这些声明而不是试图阻止它们。因此，一个测试可以有任意多的设置和清理声明。此外，测试套件也可以声明任意多的设置和清理实例。

然而，尽管允许了多个实例，但这并不意味着不会有问题。创建测试数据条目的代码就是一个很好的例子。我考虑在代码中修复这个问题，但留了下来，以便在这里解释问题：

```cpp
int createTestEntry ()
{
    // If this was real code, it might open a
    // connection to a database, insert a row
    // of data, and return the row identifier.
    return 100;
}
```

问题在先前的注释中有所暗示。它提到真正的代码会返回行标识符。由于这是一个与实际数据库没有关联的测试辅助函数，它只是简单地返回一个常量值 100。

您希望避免设置代码执行任何可能与其他设置代码冲突的操作。数据库中的行标识符不会冲突，因为每次插入数据时数据库都会返回不同的 ID。但是，其他在数据中填充的字段呢？例如，您可能在表中设置了约束，其中名称必须是唯一的。如果您在一个设置中创建了一个固定的测试名称，那么您将无法在另一个设置中使用相同的名称。

即使您在不同的设置块中有不同的固定名称，它们不会引起冲突，但如果测试数据没有得到适当的清理，您仍然会遇到问题。您可能会发现第一次运行测试时一切正常，然后之后失败，因为固定的名称已经在数据库中存在。

我建议您随机化您的测试数据。以下是一个创建测试表的另一个示例：

```cpp
std::string createTestTable ()
{
    // If this was real code, it might open a
    // connection to a database, create a temp
    // table with a random name, and return the
    // table name.
    return "test_data_01";
}
```

先前代码中的注释也提到了创建一个随机名称。使用固定的前缀是可以的，但考虑将末尾的数字设置为随机而不是固定的。这不会完全解决数据冲突的问题。随机数字可能会变成相同的。但是，与良好的测试数据清理一起做，应该有助于消除大多数冲突设置的情况。

另一个问题已经在测试库代码中得到了处理。那就是当测试声明它属于某个测试套件，而该测试套件没有定义任何设置和拆卸代码时应该怎么做。

这在测试应用程序本身中被视为一个致命错误。一旦找不到所需的测试套件设置和拆卸注册，测试应用程序将退出并且不再运行任何更多测试。

修复很简单。确保为所有测试使用的测试套件始终定义测试套件设置和拆卸代码。即使注册的测试套件设置和拆卸代码从未被任何测试使用，这也是可以的。但是，一旦测试声明它属于某个测试套件，那么该套件就变得是必需的。

现在，让我们谈谈设置和拆卸代码中的异常。这包括确认，因为失败的`CONFIRM`宏会导致抛出异常。将确认添加到如下设置代码中是可以的：

```cpp
class TempEntry
{
public:
    void setup ()
    {
        mId = createTestEntry();
        CONFIRM(10, mId);
    }
```

目前，这会导致设置失败，因为身份被固定为始终是 100 的值。而确认尝试确保该值为 10。由于测试设置代码被调用时就像是一个常规函数调用，这次失败的确认结果将与测试本身中任何其他失败的确认相同。测试将失败，总结将显示失败发生的位置和原因。总结看起来像这样：

```cpp
------- Test: Test will run multiple setup and teardown code
Failed confirm on line 51
    Expected: 10
    Actual  : 100
```

然而，将确认放入拆卸代码是不推荐的。从拆卸代码中抛出异常也是不推荐的——特别是对于测试拆卸代码，因为测试拆卸代码是在析构函数内部运行的。所以，将确认移动到如下拆卸代码中不会以相同的方式工作：

```cpp
class TempEntry
{
public:
    void setup ()
    {
        mId = createTestEntry();
    }
    void teardown ()
    {
        deleteTestEntry(mId);
        CONFIRM(10, mId);
    }
```

这将导致在使用`TempEntry`策略类`SetupAndTeardown`类析构时抛出异常。整个测试应用程序将像这样终止：

```cpp
Running 2 test suites
--------------- Suite: Single Tests
------- Test: Test will run setup and teardown code
terminate called after throwing an instance of 'MereTDD::ActualConfirmException'
/tmp/codelite-exec.sh: line 3: 38155 Abort trap: 6           ${command}
```

在测试套件拆卸代码中，问题并不那么严重，因为该拆卸代码是在套件中的所有测试完成后由测试库运行的。它不是作为类析构函数的一部分运行的。仍然建议在拆卸代码中不要抛出任何异常。

将你的拆卸代码视为清理设置和测试留下的混乱的机会。通常，它不应该包含任何需要测试的内容。

测试套件设置代码与测试设置代码略有不同。虽然测试设置代码中的异常会导致测试停止运行并失败，但在测试套件设置中抛出的异常会导致该套件中的所有测试被跳过。将此确认添加到测试套件设置将触发异常：

```cpp
class TempTable
{
public:
    void setup ()
    {
        mName = createTestTable();
        CONFIRM("test_data_02", mName);
    }
```

并且输出总结显示整个测试套件像这样被干扰：

```cpp
--------------- Suite: Suite 1
------- Setup: Test suite setup/teardown 1
Failed confirm on line 73
    Expected: test_data_02
    Actual  : test_data_01
Test suite setup failed. Skipping tests in suite.
```

前面的消息表示测试套件将被跳过。

在测试库中为测试设置和清理以及测试套件设置和清理所进行的所有错误处理在很大程度上都没有经过测试。我的意思是，我们为测试库添加了一个额外的功能来支持任何预期的失败。我没有为设置和清理代码中的预期失败做同样的事情。我觉得处理设置和清理代码中预期失败所需的额外复杂性不值得其带来的好处。

我们使用 TDD（测试驱动开发）来指导软件的设计并提高软件的质量。但 TDD 并不能完全消除对某些边缘条件的手动测试需求，这些条件在自动化测试中太难测试，或者根本不可行。

那么，是否会有一个测试来确保当所需的测试套件未注册时，测试库确实会终止？不会。这似乎是最好通过手动测试来处理的那种测试。你可能会遇到类似的情况，你将不得不决定编写测试所需的努力程度，以及这种努力是否值得成本。

# 摘要

本章完成了单元测试库中所需的最小功能。我们还没有完成测试库的开发，但现在它已经具有足够的功能，可以用于其他项目。

你已经了解了添加设置和清理代码所涉及的问题以及提供的优势。主要优势是，测试现在可以专注于需要测试的重要部分。当不再有杂乱无章的测试代码和自动处理的清理时，测试更容易理解。

设置和清理有两种类型。一种是局部于测试的；它可以在其他测试中重用，但局部意味着设置在测试开始时运行，清理在测试结束时发生。另一个共享相同设置和清理的测试将重复在该其他测试中的设置和清理。

另一种类型的设置和清理实际上是由多个测试共享的。这是测试套件的设置和清理；它的设置在套件中的任何测试开始之前运行，它的清理在套件中的所有测试完成后运行。

对于局部测试，我们能够相当容易地将它们集成到测试中，对测试库的影响不大。我们使用基于策略的设计来简化设置和清理代码的编写。而且，该设计允许测试代码访问设置中准备好的资源。

测试套件的设置和清理更为复杂，需要从测试库中获得广泛的支持。我们不得不改变测试注册和运行的方式。但与此同时，我们简化了代码，并使其更好。测试套件的设置和清理设计采用了与局部设置和清理相同的策略，这使得整个设计保持一致。

你还学到了一些关于如何处理设置和清理代码中错误的小技巧。

下一章将继续为您提供如何编写更好测试的指导和建议。
