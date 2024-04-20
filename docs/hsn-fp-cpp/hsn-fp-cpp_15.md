# 第十二章：重构到纯函数并通过纯函数

程序员经常遇到他们害怕改变的代码。通过提取纯函数，使用柯里化和组合，并利用编译器，你可以以更安全的方式重构现有代码。我们将看一个通过纯函数重构的例子，然后我们将看一些设计模式，以及它们在函数式编程中的实现，以及如何在重构中使用它们。

本章将涵盖以下主题：

+   如何思考遗留代码

+   如何使用编译器和纯函数来识别和分离依赖关系

+   如何从任何代码中提取 lambda

+   如何使用柯里化和组合消除 lambda 之间的重复，并将它们分组到类中

+   如何使用函数实现一些设计模式（策略、命令和依赖注入）

+   如何使用基于函数的设计模式来重构

# 技术要求

你将需要一个支持 C++ 17 的编译器。我使用的是 GCC 7.4.0c。

代码在 GitHub 上的`Chapter12`文件夹中。它包括并使用`doctest`，这是一个单头文件的开源单元测试库。你可以在它的 GitHub 仓库上找到它[https:/​/github.​com/​onqtam/​doctest](https://github.%E2%80%8Bcom/onqtam/doctest)。

# 重构到纯函数并通过纯函数

**重构**是软件开发的一个重要而持续的部分。主要原因是需求的持续变化，由我们构建的应用程序周围世界的变化所驱动。我们的客户不断了解产品所在的生态系统，并需要我们将这些产品适应他们发现的新现实。因此，我们的代码，即使结构完美，几乎总是落后于我们当前对所解决问题的理解。

完美地构建我们的代码也不容易。程序员是人，所以我们会犯错，失去焦点，有时找不到最佳解决方案。处理这种复杂情况的唯一方法是使用无情的重构；也就是说，在让事情运转后，我们改进代码结构，直到在我们拥有的约束下代码达到最佳状态。

只要我们很早就重构并编写测试，那就很容易说和做。但是如果我们继承了一个没有测试的代码库呢？那我们该怎么办？我们将讨论这个问题，以及后面将使用纯函数来重构遗留代码的一个有前途的想法。

首先，让我们定义我们的术语。什么是重构？

# 什么是重构？

重构是行业中普遍使用的术语之一，但并不被很好理解。不幸的是，这个术语经常被用来证明大的重设计。考虑以下关于给定项目的常见故事：

+   项目开始时，功能以快速的速度添加。

+   很快（几个月、一年，甚至几周），速度下降了，但需求是一样的。

+   多年后，添加新功能变得如此困难，以至于客户感到恼火并向团队施加压力。

+   最终，决定重写或改变代码的整体结构，希望能加快速度。

+   六个月后，重写或重设计（通常）失败，管理层面临着一个不可能的情况——我们应该尝试重设计、重新启动项目，还是做其他事情？

这个循环的**大重设计**阶段通常错误地被称为重构，但这并不是重构的含义。

相反，要理解重构的真正含义，让我们从思考对代码库可以做出的改变开始。我们通常可以将这些改变分类如下：

+   实施新要求

+   修复一个错误

+   以各种方式重新组织代码——重构、重工程、重设计和/或重架构

我们可以将这些更改大致分类为两大类，如下：

+   影响代码行为的更改

+   不影响代码行为的更改

当我们谈论行为时，我们谈论输入和输出，比如“当我在**用户界面**（UI）表单中输入这些值并单击此按钮时，然后我看到这个输出并保存这些东西”。我们通常不包括性能、可伸缩性或安全性等跨功能关注点在行为中。

有了这些明确的术语，我们可以定义重构——简单地对不影响程序外部行为的代码结构进行更改。大型重设计或重写很少符合这个定义，因为通常进行大型重设计的团队并不证明结果与原始代码具有相同的行为（包括已知的错误，因为有人可能依赖它们）。

对程序进行任何修改其行为的更改都不是重构。这包括修复错误或添加功能。然而，我们可以将这些更改分为两个阶段——首先重构以*为更改腾出空间*，然后进行行为更改。

这个定义引发了一些问题，如下：

+   我们如何证明我们没有改变行为？我们知道的唯一方法是：自动回归测试。如果我们有一套我们信任且足够快速的自动化测试，我们可以轻松地进行更改而不改变任何测试，并查看它们是否通过。

+   重构有多小？更改越大，证明没有受到影响就越困难，因为程序员是人类，会犯错误。我们更喜欢在重构中采取非常小的步骤。以下是一些保持行为的小代码更改的示例：重命名、向函数添加参数、更改函数的参数顺序以及将一组语句提取到函数中等。每个小更改都可以轻松进行，并运行测试以证明没有发生行为更改。每当我们需要进行更大的重构时，我们只需进行一系列这些小更改。

+   当我们没有测试时，我们如何证明我们没有改变代码的行为？这就是我们需要谈论遗留代码和遗留代码困境的时候。

# 遗留代码困境

编程可能是唯一一个“遗留”一词具有负面含义的领域。在任何其他情况下，“遗留”都意味着某人留下的东西，通常是某人引以为傲的东西。在编程中，遗留代码指的是我们继承的独占代码，维护起来很痛苦。

程序员经常认为遗留代码是不可避免的，对此无能为力。然而，我们可以做很多事情。首先是澄清我们所说的遗留代码是什么意思。迈克尔·菲瑟斯在他的遗留代码书中将其定义为没有测试的代码。然而，我更倾向于使用更一般的定义：*你害怕改变的代码*。你害怕改变的代码会减慢你的速度，减少你的选择，并使任何新的开发成为一场磨难。但这绝不是不可避免的：我们可以改变它，我们将看到如何做到这一点。

我们可以做的第二件事是了解遗留代码的困境。为了不那么害怕改变，我们需要对其进行重构，但为了重构代码，我们需要编写测试。要编写测试，我们需要调整代码使其可测试；这看起来像一个循环——为了改变代码，我们需要改变代码！如果我们一开始就害怕改变代码，我们该怎么办？

幸运的是，这个困境有一个解决办法。如果我们能够对代码进行安全的更改——这些更改几乎没有错误的机会，并且允许我们测试代码——那么我们就可以慢慢但肯定地改进代码。这些更改确实是重构，但它们甚至比重构步骤更小、更安全。它们的主要目标是打破代码中设计元素之间的依赖关系，使我们能够编写测试，以便在之后继续重构。

由于我们的重点是使用纯函数和函数构造来重构代码，我们不会查看完整的技术列表。我可以给出一个简单的例子，称为**提取和覆盖**。假设您需要为一个非常大的函数编写测试。如果我们只能为函数的一小部分编写测试，那将是理想的。我们可以通过将要测试的代码提取到另一个函数中来实现这一点。然而，新函数依赖于旧代码，因此我们将很难弄清所有的依赖关系。为了解决这个问题，我们可以创建一个派生类，用虚拟函数覆盖我们函数的所有依赖关系。在单元测试中，这称为*部分模拟*。这使我们能够用测试覆盖我们提取函数的所有代码，同时假设类的所有其他部分都按预期工作。一旦我们用测试覆盖了它，我们就可以开始重构；在这个练习结束时，我们经常会提取一个完全由模拟或存根的新类。

这些技术是在我们的语言中广泛支持函数式编程之前编写的。现在我们可以利用纯函数来安全地重构我们编写的代码。但是，为了做到这一点，我们需要了解依赖关系如何影响我们测试和更改代码的能力。

# 依赖和变更

我们的用户和客户希望项目成功的时间越长，就能获得越多的功能。然而，我们经常无法交付，因为随着时间的推移，代码往往变得越来越僵化。随着时间的推移，添加新功能变得越来越慢，而且在添加功能时会出现新的错误。

这引出了一个十分重要的问题——是什么使代码难以更改？我们如何编写能够保持变更速度甚至增加变更速度的代码？

这是一个复杂的问题，有许多方面和各种解决方案。其中一个在行业中基本上是一致的——依赖关系往往会减慢开发速度。具有较少依赖关系的代码结构通常更容易更改，从而更容易添加功能。

我们可以从许多层面来看依赖关系。在更高的层面上，我们可以谈论依赖于其他可执行文件的可执行文件；例如，直接调用另一个网络服务的网络服务。通过使用基于事件的系统而不是直接调用，可以减少这个层面上的依赖关系。在更低的层面上，我们可以谈论对库或操作系统例程的依赖；例如，一个网络服务依赖于特定文件夹或特定库版本的存在。

虽然其他所有层面都很有趣，但对于我们的目标，我们将专注于类/函数级别，特别是类和函数如何相互依赖。由于在任何非平凡的代码库中都不可能避免依赖关系，因此我们将专注于依赖关系的强度。

我们将以我编写的一小段代码作为示例，该代码根据员工列表和角色、资历、组织连续性和奖金水平等参数计算工资。它从 CSV 文件中读取员工列表，根据一些规则计算工资，并打印计算出的工资列表。代码的第一个版本是天真地编写的，只使用`main`函数，并将所有内容放在同一个文件中，如下面的代码示例所示。

```cpp
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>

using namespace std;

int main(){
    string id;
    string employee_id;
    string first_name;
    string last_name;
    string seniority_level;
    string position;
    string years_worked_continuously;
    string special_bonus_level;

    ifstream employeesFile("./Employees.csv");
    while (getline(employeesFile, id, ',')) {
        getline(employeesFile, employee_id, ',') ;
        getline(employeesFile, first_name, ',') ;
        getline(employeesFile, last_name, ',') ;
        getline(employeesFile, seniority_level, ',') ;
        getline(employeesFile, position, ',') ;
        getline(employeesFile, years_worked_continuously, ',') ;
        getline(employeesFile, special_bonus_level);
        if(id == "id") continue;

        int baseSalary;
        if(position == "Tester") baseSalary= 1500;
        if(position == "Analyst") baseSalary = 1600;
        if(position == "Developer") baseSalary = 2000;
        if(position == "Team Leader") baseSalary = 3000;
        if(position == "Manager") baseSalary = 4000;

        double factor;
        if(seniority_level == "Entry") factor = 1;
        if(seniority_level == "Junior") factor = 1.2;
        if(seniority_level == "Senior") factor = 1.5;

        double continuityFactor;
        int continuity = stoi(years_worked_continuously);
        if(continuity < 3) continuityFactor = 1;
        if(continuity >= 3 && continuity < 5) continuityFactor = 1.2;
        if(continuity >= 5 && continuity < 10) continuityFactor = 1.5;
        if(continuity >=10 && continuity <= 20) continuityFactor = 1.7;
        if(continuity > 20) continuityFactor = 2;

        int specialBonusLevel = stoi(special_bonus_level);
        double specialBonusFactor = specialBonusLevel * 0.03;

        double currentSalary = baseSalary * factor * continuityFactor;
        double salary = currentSalary + specialBonusFactor * 
            currentSalary;

        int roundedSalary = ceil(salary);

        cout  << seniority_level << position << " " << first_name << " 
            " << last_name << " (" << years_worked_continuously << 
            "yrs)" <<  ", " << employee_id << ", has salary (bonus                 
            level  " << special_bonus_level << ") " << roundedSalary << 
            endl;
    }
}
```

输入文件是使用专门的工具生成的随机值，看起来像这样：

```cpp
id,employee_id,First_name,Last_name,Seniority_level,Position,Years_worked_continuously,Special_bonus_level
1,51ef10eb-8c3b-4129-b844-542afaba7eeb,Carmine,De Vuyst,Junior,Manager,4,3
2,171338c8-2377-4c70-bb66-9ad669319831,Gasper,Feast,Entry,Team Leader,10,5
3,807e1bc7-00db-494b-8f92-44acf141908b,Lin,Sunley,Medium,Manager,23,3
4,c9f18741-cd6c-4dee-a243-00c1f55fde3e,Leeland,Geraghty,Medium,Team Leader,7,4
5,5722a380-f869-400d-9a6a-918beb4acbe0,Wash,Van der Kruys,Junior,Developer,7,1
6,f26e94c5-1ced-467b-ac83-a94544735e27,Marjie,True,Senior,Tester,28,1

```

当我们运行程序时，为每个员工计算了`salary`，输出如下所示：

```cpp
JuniorManager Carmine De Vuyst (4yrs), 51ef10eb-8c3b-4129-b844-542afaba7eeb, has salary (bonus level  3) 6279
EntryTeam Leader Gasper Feast (10yrs), 171338c8-2377-4c70-bb66-9ad669319831, has salary (bonus level  5) 5865
MediumManager Lin Sunley (23yrs), 807e1bc7-00db-494b-8f92-44acf141908b, has salary (bonus level  3) 8720
MediumTeam Leader Leeland Geraghty (7yrs), c9f18741-cd6c-4dee-a243-00c1f55fde3e, has salary (bonus level  4) 5040
JuniorDeveloper Wash Van der Kruys (7yrs), 5722a380-f869-400d-9a6a-918beb4acbe0, has salary (bonus level  1) 3708
SeniorTester Marjie True (28yrs), f26e94c5-1ced-467b-ac83-a94544735e27, has salary (bonus level  1) 4635
EntryAnalyst Muriel Dorken (10yrs), f4934e00-9c01-45f9-bddc-2366e6ea070e, has salary (bonus level  8) 3373
SeniorTester Harrison Mawditt (17yrs), 66da352a-100c-4209-a13e-00ec12aa167e, has salary (bonus level  10) 4973
```

那么，这段代码有依赖关系吗？有，并且它们就在眼前。

查找依赖关系的一种方法是查找构造函数调用或全局变量。在我们的例子中，我们有一个对`ifstream`的构造函数调用，以及一个对`cout`的使用，如下例所示：

```cpp
ifstream employeesFile("./Employees.csv")
cout  << seniority_level << position << " " << first_name << " " << 
    last_name << " (" << years_worked_continuously << "yrs)" <<  ", " 
    << employee_id << ", has salary (bonus level  " << 
    special_bonus_level << ") " << roundedSalary << endl;
```

识别依赖的另一种方法是进行一种想象练习。想象一下什么要求可能会导致代码的变化。有几种情况。如果我们决定切换到员工数据库，我们将需要改变读取数据的方式。如果我们想要输出到文件，我们将需要改变打印工资的代码行。如果计算工资的规则发生变化，我们将需要更改计算`salary`的代码行。

这两种方法都得出了相同的结论；我们对文件系统和标准输出有依赖。让我们专注于标准输出，并提出一个问题；我们如何改变代码，以便将工资输出到标准输出和文件中？答案非常简单，由于**标准模板库**（**STL**）流的多态性，只需提取一个接收输出流并写入数据的函数。让我们看看这样一个函数会是什么样子；为了简单起见，我们还引入了一个名为`Employee`的结构，其中包含我们需要的所有字段，如下例所示：

```cpp
void printEmployee(const Employee& employee, ostream& stream, int 
    roundedSalary){
        stream << employee.seniority_level << employee.position << 
        " " << employee.first_name << " " << employee.last_name << 
        " (" << employee.years_worked_continuously << "yrs)" <<  ",             
        " << employee.employee_id << ", has salary (bonus level  " << 
        employee.special_bonus_level << ") " << roundedSalary << endl;
    }
```

这个函数不再依赖于标准输出。在依赖方面，我们可以说*我们打破了依赖关系*，即员工打印和标准输出之间的依赖关系。我们是如何做到的呢？嗯，我们将`cout`流作为函数的参数从调用者传递进来：

```cpp
        printEmployee(employee, cout, roundedSalary);
```

这个看似微小的改变使函数成为多态的。`printEmployee`的调用者现在控制函数的输出，而不需要改变函数内部的任何东西。

此外，我们现在可以为`printEmployee`函数编写测试，而不必触及文件系统。这很重要，因为文件系统访问速度慢，而且由于诸如磁盘空间不足或损坏部分等原因，在测试正常路径时可能会出现错误。我们如何编写这样的测试呢？嗯，我们只需要使用内存流调用该函数，然后将写入内存流的输出与我们期望的输出进行比较。

因此，打破这种依赖关系会极大地改善我们代码的可更改性和可测试性。这种机制非常有用且广泛，因此它得到了一个名字——**依赖注入**（**DI**）。在我们的情况下，`printEmployee`函数的调用者（`main`函数、`test`函数或另一个未来的调用者）将依赖注入到我们的函数中，从而控制其行为。

关于 DI 有一点很重要——它是一种设计模式，而不是一个库。许多现代库和 MVC 框架都支持 DI，但您不需要任何外部内容来注入依赖关系。您只需要将依赖项传递给构造函数、属性或函数参数，然后就可以了。

我们学会了如何识别依赖关系以及如何使用 DI 来打破它们。现在是时候看看我们如何利用纯函数来重构这段代码了。

# 纯函数和程序的结构

几年前，我学到了关于计算机程序的一个基本定律，这导致我研究如何在重构中使用纯函数：

*任何计算机程序都可以由两种类型的类/函数构建——一些进行 I/O，一些是纯函数。*

在之后寻找类似想法时，我发现 Gary Bernhardt 对这些结构的简洁命名：*functional core, imperative shell*（[`www.destroyallsoftware.com/screencasts/catalog/functional-core-imperative-shell`](https://www.destroyallsoftware.com/screencasts/catalog/functional-core-imperative-shell)）。

无论你如何称呼它，这个想法对重构的影响都是根本的。如果任何程序都可以被写成两种不同类型的类/函数，一些是不可变的，一些是 I/O，那么我们可以利用这个属性来重构遗留代码。高层次的过程看起来会像这样：

+   提取纯函数（我们将看到这些步骤识别依赖关系）。

+   测试和重构它们。

+   根据高内聚原则将它们重新分组为类。

我想在这个定律中添加一个公理。我相信我们可以在代码的任何级别应用这个定律，无论是函数、类、代码行组、类组还是整个模块，除了那些纯 I/O 的代码行。换句话说，这个定律是分形的；它适用于代码的任何级别，除了最基本的代码行。

这个公理的重要性是巨大的。它告诉我们的是，我们可以在代码的任何级别应用之前描述的相同方法，除了最基本的。换句话说，我们从哪里开始应用这个方法并不重要，因为它在任何地方都会起作用。

在接下来的几节中，我们将探讨该方法的每个步骤。首先，让我们提取一些纯函数。

# 使用编译器和纯函数来识别依赖关系。

尝试更改我们不理解且没有测试的代码可能会感到冒险。任何错误都可能导致丑陋的错误，任何更改都可能导致错误。

幸运的是，编译器和纯函数可以帮助揭示依赖关系。记住纯函数是什么——对于相同的输入返回相同输出的函数。这意味着，根据定义，纯函数的所有依赖关系都是可见的，通过参数、全局变量或变量捕获传递。

这引导我们以一种简单的方式来识别代码中的依赖关系：选择几行代码，将它们提取到一个函数中，使其成为纯函数，然后让编译器告诉你依赖关系是什么。此外，这些依赖关系将需要被注入，从而使我们得到一个可测试的函数。

让我们看几个例子。一个简单的开始是下面几行代码，根据公司员工的职位计算基本工资：

```cpp
        int baseSalary;
        if(position == "Tester") baseSalary = 1500;
        if(position == "Analyst") baseSalary = 1600;
        if(position == "Developer") baseSalary = 2000;
        if(position == "Team Leader") baseSalary = 3000;
        if(position == "Manager") baseSalary = 4000;
```

让我们将其提取为一个纯函数。现在名称并不重要，所以我们暂时称之为`doesSomething`，然后我将代码行复制粘贴到新函数中，而不是从旧函数中删除它们，如下例所示：

```cpp
auto doesSomething = [](){
        int baseSalary;
        if(position == "Tester") baseSalary = 1500;
        if(position == "Analyst") baseSalary = 1600;
        if(position == "Developer") baseSalary = 2000;
        if(position == "Team Leader") baseSalary = 3000;
        if(position == "Manager") baseSalary = 4000;
};
```

我的编译器立即抱怨说位置未定义，所以它帮我找出了依赖关系。让我们将其添加为一个参数，如下面的示例所示：

```cpp
auto doesSomething = [](const string& position){
        int baseSalary;
        if(position == "Tester") baseSalary = 1500;
        if(position == "Analyst") baseSalary = 1600;
        if(position == "Developer") baseSalary = 2000;
        if(position == "Team Leader") baseSalary = 3000;
        if(position == "Manager") baseSalary = 4000;
};
```

这个函数缺少一些东西；纯函数总是返回值，但这个函数没有。让我们添加`return`语句，如下面的代码示例所示：

```cpp
auto doesSomething = [](const string& position){
        int baseSalary;
        if(position == "Tester") baseSalary = 1500;
        if(position == "Analyst") baseSalary = 1600;
        if(position == "Developer") baseSalary = 2000;
        if(position == "Team Leader") baseSalary = 3000;
        if(position == "Manager") baseSalary = 4000;
        return baseSalary;
};
```

现在这个函数足够简单，可以独立测试了。但首先，我们需要将其提取到一个单独的`.h`文件中，并给它一个合适的名称。`baseSalaryForPosition`听起来不错；让我们在下面的代码中看看它的测试：

```cpp
TEST_CASE("Base salary"){
    CHECK_EQ(1500, baseSalaryForPosition("Tester"));
    CHECK_EQ(1600, baseSalaryForPosition("Analyst"));
    CHECK_EQ(2000, baseSalaryForPosition("Developer"));
    CHECK_EQ(3000, baseSalaryForPosition("Team Leader"));
    CHECK_EQ(4000, baseSalaryForPosition("Manager"));
    CHECK_EQ(0, baseSalaryForPosition("asdfasdfs"));
}
```

编写这些测试相当简单。它们也重复了许多来自函数的东西，包括位置字符串和薪水值。有更好的方法来组织代码，但这是预期的遗留代码。现在，我们很高兴我们用测试覆盖了初始代码的一部分。我们还可以向领域专家展示这些测试，并检查它们是否正确，但让我们继续进行重构。我们需要从`main()`开始调用新函数，如下所示：

```cpp
    while (getline(employeesFile, id, ',')) {
        getline(employeesFile, employee_id, ',') ;
        getline(employeesFile, first_name, ',') ;
        getline(employeesFile, last_name, ',') ;
        getline(employeesFile, seniority_level, ',') ;
        getline(employeesFile, position, ',') ;
        getline(employeesFile, years_worked_continuously, ',') ;
        getline(employeesFile, special_bonus_level);
        if(id == "id") continue;

 int baseSalary = baseSalaryForPosition(position);
        double factor;
        if(seniority_level == "Entry") factor = 1;
        if(seniority_level == "Junior") factor = 1.2;
        if(seniority_level == "Senior") factor = 1.5;
        ...
}

```

虽然这是一个简单的案例，但它展示了基本的过程，如下所示：

+   选择几行代码。

+   将它们提取到一个函数中。

+   使函数成为纯函数。

+   注入所有依赖。

+   为新的纯函数编写测试。

+   验证行为。

+   重复，直到整个代码都被测试覆盖。

如果您遵循这个过程，引入错误的风险将变得极小。根据我的经验，您需要最小心的是使函数成为纯函数。记住——如果它在一个类中，将其设为带有`const`参数的静态函数，但如果它在类外部，将所有参数作为`const`传递，并将其设为 lambda。

如果我们重复这个过程几次，我们最终会得到更多的纯函数。首先，`factorForSeniority`根据资历级别计算因子，如下例所示：

```cpp
auto factorForSeniority = [](const string& seniority_level){
    double factor;
    if(seniority_level == "Entry") factor = 1;
    if(seniority_level == "Junior") factor = 1.2;
    if(seniority_level == "Senior") factor = 1.5;
    return factor;
};
```

然后，`factorForContinuity`根据——你猜对了——连续性计算因子：

```cpp
auto factorForContinuity = [](const string& years_worked_continuously){
    double continuityFactor;
    int continuity = stoi(years_worked_continuously);
    if(continuity < 3) continuityFactor = 1;
    if(continuity >= 3 && continuity < 5) continuityFactor = 1.2;
    if(continuity >= 5 && continuity < 10) continuityFactor = 1.5;
    if(continuity >=10 && continuity <= 20) continuityFactor = 1.7;
    if(continuity > 20) continuityFactor = 2;
    return continuityFactor;
};

```

最后，`bonusLevel`函数读取奖金级别：

```cpp
auto bonusLevel = [](const string& special_bonus_level){
    return stoi(special_bonus_level);
};
```

这些函数中的每一个都可以很容易地通过基于示例的、数据驱动的或基于属性的测试进行测试。提取了所有这些函数后，我们的主要方法看起来像以下示例（为简洁起见，省略了几行）：

```cpp
int main(){
...
    ifstream employeesFile("./Employees.csv");
    while (getline(employeesFile, id, ',')) {
        getline(employeesFile, employee_id, ',') ;
...
        getline(employeesFile, special_bonus_level);
        if(id == "id") continue;

 int baseSalary = baseSalaryForPosition(position);
 double factor = factorForSeniority(seniority_level);

 double continuityFactor = 
            factorForContinuity(years_worked_continuously);

 int specialBonusLevel =  bonusLevel(special_bonus_level);
        double specialBonusFactor = specialBonusLevel * 0.03;

        double currentSalary = baseSalary * factor * continuityFactor;
        double salary = currentSalary + specialBonusFactor * 
            currentSalary;

        int roundedSalary = ceil(salary);

        cout  << seniority_level << position << " " << first_name << "           
          " << last_name << " (" << years_worked_continuously << "yrs)"     
          <<  ", " << employee_id << ", has salary (bonus level  " << 
          special_bonus_level << ") " << roundedSalary << endl;
    }
```

这样会更清晰，而且测试覆盖更好。然而，lambda 还可以用于更多的操作；让我们看看我们如何做到这一点。

# 从遗留代码到 lambda

除了纯度，lambda 还为我们提供了许多可以使用的操作：函数组合、部分应用、柯里化和高级函数。在重构遗留代码时，我们可以利用这些操作。

展示这一点最简单的方法是从`main`方法中提取整个`salary`计算。以下是计算`salary`的代码行：

```cpp
...        
        int baseSalary = baseSalaryForPosition(position);
        double factor = factorForSeniority(seniority_level);

        double continuityFactor = 
            factorForContinuity(years_worked_continuously);

        int specialBonusLevel =  bonusLevel(special_bonus_level);
        double specialBonusFactor = specialBonusLevel * 0.03;

        double currentSalary = baseSalary * factor * continuityFactor;
        double salary = currentSalary + specialBonusFactor * 
            currentSalary;

        int roundedSalary = ceil(salary);
...
```

我们可以以两种方式提取这个纯函数——一种是将需要的每个值作为参数传递，结果如下所示：

```cpp
auto computeSalary = [](const string& position, const string seniority_level, const string& years_worked_continuously, const string& special_bonus_level){
    int baseSalary = baseSalaryForPosition(position);
    double factor = factorForSeniority(seniority_level);

    double continuityFactor = 
        factorForContinuity(years_worked_continuously);

    int specialBonusLevel =  bonusLevel(special_bonus_level);
    double specialBonusFactor = specialBonusLevel * 0.03;

    double currentSalary = baseSalary * factor * continuityFactor;
    double salary = currentSalary + specialBonusFactor * currentSalary;

    int roundedSalary = ceil(salary);
    return roundedSalary;
};
```

第二个选项更有趣。与其传递变量，不如我们传递函数并事先将它们绑定到所需的变量？

这是一个有趣的想法。结果是一个接收多个函数作为参数的函数，每个函数都没有任何参数：

```cpp
auto computeSalary = [](auto baseSalaryForPosition, auto factorForSeniority, auto factorForContinuity, auto bonusLevel){
    int baseSalary = baseSalaryForPosition();
    double factor = factorForSeniority();
    double continuityFactor = factorForContinuity();
    int specialBonusLevel =  bonusLevel();

    double specialBonusFactor = specialBonusLevel * 0.03;

    double currentSalary = baseSalary * factor * continuityFactor;
    double salary = currentSalary + specialBonusFactor * currentSalary;

    int roundedSalary = ceil(salary);
    return roundedSalary;
};
```

`main`方法需要首先绑定这些函数，然后将它们注入到我们的方法中，如下所示：

```cpp
        auto roundedSalary = computeSalary(
                bind(baseSalaryForPosition, position), 
                bind(factorForSeniority, seniority_level),
        bind(factorForContinuity, years_worked_continuously),
        bind(bonusLevel, special_bonus_level));

        cout  << seniority_level << position << " " << first_name << " 
          " << last_name << " (" << years_worked_continuously << "yrs)"           
          <<  ", " << employee_id << ", has salary (bonus level  " <<              
          special_bonus_level << ") " << roundedSalary << endl;
```

为什么这种方法很有趣？好吧，让我们从软件设计的角度来看看。我们创建了小的纯函数，每个函数都有明确的责任。然后，我们将它们绑定到特定的值。之后，我们将它们作为参数传递给另一个 lambda，该 lambda 使用它们来计算我们需要的结果。

在**面向对象编程**（**OOP**）风格中，这意味着什么？好吧，函数将成为类的一部分。将函数绑定到值相当于调用类的构造函数。将对象传递给另一个函数称为 DI。

等一下！实际上我们正在分离责任并注入依赖项，只是使用纯函数而不是对象！因为我们使用纯函数，依赖关系由编译器明确表示。因此，我们有一种重构代码的方法，几乎没有错误的可能性，因为我们经常使用编译器。这是一个非常有用的重构过程。

我不得不承认，结果并不如我所希望的那样好。让我们重构我们的 lambda。

# 重构 lambda

我对我们提取出来的`computeSalary` lambda 的样子并不满意。由于接收了许多参数和多个责任，它相当复杂。让我们仔细看看它，看看我们如何可以改进它：

```cpp
auto computeSalary = [](auto baseSalaryForPosition, auto 
    factorForSeniority, auto factorForContinuity, auto bonusLevel){
        int baseSalary = baseSalaryForPosition();
        double factor = factorForSeniority();
        double continuityFactor = factorForContinuity();
        int specialBonusLevel =  bonusLevel();

        double specialBonusFactor = specialBonusLevel * 0.03;

        double currentSalary = baseSalary * factor * continuityFactor;
        double salary = currentSalary + specialBonusFactor * 
            currentSalary;

        int roundedSalary = ceil(salary);
         return roundedSalary;
};
```

所有迹象似乎表明这个函数有多个责任。如果我们从中提取更多的函数会怎样呢？让我们从`specialBonusFactor`计算开始：

```cpp
auto specialBonusFactor = [](auto bonusLevel){
    return bonusLevel() * 0.03;
};
auto computeSalary = [](auto baseSalaryForPosition, auto     
factorForSeniority, auto factorForContinuity, auto bonusLevel){
    int baseSalary = baseSalaryForPosition();
    double factor = factorForSeniority();
    double continuityFactor = factorForContinuity();

    double currentSalary = baseSalary * factor * continuityFactor;
    double salary = currentSalary + specialBonusFactor() * 
        currentSalary;

    int roundedSalary = ceil(salary);
    return roundedSalary;
};
```

现在我们可以注入`specialBonusFactor`。但是，请注意，`specialBonusFactor`是唯一需要`bonusLevel`的 lambda。这意味着我们可以将`bonusLevel` lambda 部分应用于`specialBonusFactor` lambda，如下例所示：

```cpp
int main(){
        ...
  auto bonusFactor = bind(specialBonusFactor, [&](){ return 
    bonusLevel(special_bonus_level); } );
  auto roundedSalary = computeSalary(
      bind(baseSalaryForPosition, position), 
      bind(factorForSeniority, seniority_level),
      bind(factorForContinuity, years_worked_continuously),
      bonusFactor
     );
 ...
}

auto computeSalary = [](auto baseSalaryForPosition, auto factorForSeniority, auto factorForContinuity, auto bonusFactor){
    int baseSalary = baseSalaryForPosition();
    double factor = factorForSeniority();
    double continuityFactor = factorForContinuity();

    double currentSalary = baseSalary * factor * continuityFactor;
    double salary = currentSalary + bonusFactor() * currentSalary;

    int roundedSalary = ceil(salary);
    return roundedSalary;
};
```

我们的`computeSalary` lambda 现在更小了。我们甚至可以通过内联临时变量使它更小：

```cpp
auto computeSalary = [](auto baseSalaryForPosition, auto 
    factorForSeniority, auto factorForContinuity, auto bonusFactor){
        double currentSalary = baseSalaryForPosition() * 
            factorForSeniority() * factorForContinuity();
    double salary = currentSalary + bonusFactor() * currentSalary;
    return ceil(salary);
};
```

这很不错！然而，我想让它更接近一个数学公式。首先，让我们重写计算`salary`的那一行（在代码中用粗体标出）：

```cpp
auto computeSalary = [](auto baseSalaryForPosition, auto 
    factorForSeniority, auto factorForContinuity, auto bonusFactor){
        double currentSalary = baseSalaryForPosition() * 
            factorForSeniority() * factorForContinuity();
 double salary = (1 + bonusFactor()) * currentSalary;
    return ceil(salary);
};
```

然后，让我们用函数替换变量。然后我们得到以下代码示例：

```cpp
auto computeSalary = [](auto baseSalaryForPosition, auto 
    factorForSeniority, auto factorForContinuity, auto bonusFactor){
        return ceil (
                (1 + bonusFactor()) * baseSalaryForPosition() *                             
                    factorForSeniority() * factorForContinuity()
    );
};
```

因此，我们有一个 lambda 函数，它接收多个 lambda 函数并使用它们来计算一个值。我们仍然可以对其他函数进行改进，但我们已经达到了一个有趣的点。

那么我们接下来该怎么办呢？我们已经注入了依赖关系，代码更加模块化，更容易更改，也更容易测试。我们可以从测试中注入 lambda 函数，返回我们想要的值，这实际上是单元测试中的一个 stub。虽然我们没有改进整个代码，但我们通过提取纯函数和使用函数操作来分离依赖关系和责任。如果我们愿意，我们可以把代码留在这样。或者，我们可以迈出另一步，将函数重新分组成类。

# 从 lambda 到类

在这本书中，我们已经多次指出，一个类只不过是一组具有内聚性的部分应用纯函数。到目前为止，我们使用的技术已经创建了一堆部分应用的纯函数。现在将它们转换成类是一项简单的任务。

让我们看一个`baseSalaryForPosition`函数的简单例子：

```cpp
auto baseSalaryForPosition = [](const string& position){
    int baseSalary;
    if(position == "Tester") baseSalary = 1500;
    if(position == "Analyst") baseSalary = 1600;
    if(position == "Developer") baseSalary = 2000;
    if(position == "Team Leader") baseSalary = 3000;
    if(position == "Manager") baseSalary = 4000;
    return baseSalary;
};
```

我们在`main()`中使用它，就像下面的例子一样：

```cpp
        auto roundedSalary = computeSalary(
 bind(baseSalaryForPosition, position), 
                bind(factorForSeniority, seniority_level),
                bind(factorForContinuity, years_worked_continuously),
                bonusFactor
            );
```

要将其转换成类，我们只需要创建一个接收`position`参数的构造函数，然后将其改为类方法。让我们在下面的示例中看一下：

```cpp
class BaseSalaryForPosition{
    private:
        const string& position;

    public:
        BaseSalaryForPosition(const string& position) : 
            position(position){};

        int baseSalaryForPosition() const{
            int baseSalary;
            if(position == "Tester") baseSalary = 1500;
            if(position == "Analyst") baseSalary = 1600;
            if(position == "Developer") baseSalary = 2000;
            if(position == "Team Leader") baseSalary = 3000;
            if(position == "Manager") baseSalary = 4000;
            return baseSalary;
        }
};
```

我们可以简单地将部分应用函数传递给`computeSalary` lambda，如下面的代码所示：

```cpp
 auto bonusFactor = bind(specialBonusFactor, [&](){ return 
            bonusLevel(special_bonus_level); } );
            auto roundedSalary = computeSalary(
                theBaseSalaryForPosition,
                bind(factorForSeniority, seniority_level),
                bind(factorForContinuity, years_worked_continuously),
                bonusFactor
            );
```

为了使其工作，我们还需要像这里所示的改变我们的`computeSalary` lambda：

```cpp
auto computeSalary = [](const BaseSalaryForPosition& 
    baseSalaryForPosition, auto factorForSeniority, auto     
        factorForContinuity, auto bonusFactor){
            return ceil (
                (1 + bonusFactor()) * 
                    baseSalaryForPosition.baseSalaryForPosition() *                             
                        factorForSeniority() * factorForContinuity()
            );
};
```

现在，为了允许注入不同的实现，我们实际上需要从`BaseSalaryForPosition`类中提取一个接口，并将其作为接口注入，而不是作为一个类。这对于从测试中注入 double 值非常有用，比如 stub 或 mock。

从现在开始，你可以根据自己的需要将函数重新分组成类。我会把这留给读者作为一个练习，因为我相信我们已经展示了如何使用纯函数来重构代码，即使我们最终想要得到面向对象的代码。

# 重温重构方法

到目前为止，我们学到了什么？嗯，我们经历了一个结构化的重构过程，可以在代码的任何级别使用，减少错误的概率，并实现可更改性和测试性。这个过程基于两个基本思想——任何程序都可以被写成不可变函数和 I/O 函数的组合，或者作为一个函数核心在一个命令式外壳中。此外，我们已经表明这个属性是分形的——我们可以将它应用到任何代码级别，从几行到整个模块。

由于不可变函数可以成为我们程序的核心，我们可以逐渐提取它们。我们写下新的函数名称，复制并粘贴函数体，并使用编译器将任何依赖项作为参数传递。当代码编译完成时，如果我们小心而缓慢地进行更改，我们可以相当确信代码仍然正常工作。这种提取揭示了我们函数的依赖关系，从而使我们能够做出设计决策。

接下来，我们将提取更多的函数，这些函数接收其他部分应用的纯函数作为参数。这导致了依赖关系和实际的破坏性依赖关系之间的明显区别。

最后，由于部分应用函数等同于类，我们可以根据内聚性轻松地封装一个或多个函数。这个过程无论我们是从类还是函数开始，都可以工作，而且无论我们最终想要以函数或类结束都没有关系。然而，它允许我们使用函数构造来打破依赖关系，并在我们的代码中分离责任。

由于我们正在改进设计，现在是时候看看设计模式如何应用于函数式编程以及如何向它们重构。我们将访问一些四人帮模式，以及我们已经在我们的代码中使用过的 DI。

# 设计模式

软件开发中的许多好东西都来自于那些注意到程序员工作方式并从中提取某些教训的人；换句话说，看待实际方法并提取共同和有用的教训，而不是推测解决方案。

所谓的四人帮（Erich Gamma，Richard Helm，Ralph Johnson 和 John Vlissides）在记录设计模式时采取了这种确切的方法，用精确的语言列出了一系列设计模式。在注意到更多程序员以类似的方式解决相同问题后，他们决定将这些模式写下来，并向编程世界介绍了在明确上下文中对特定问题的可重用解决方案的想法。

由于当时的设计范式是面向对象编程，他们出版的*设计模式*书籍展示了使用面向对象方法的这些解决方案。顺便说一句，有趣的是注意到他们在可能的情况下至少记录了两种类型的解决方案——一种基于继承，另一种基于对象组合。我花了很多时间研究设计模式书籍，我可以告诉你，这是一个非常有趣的软件设计课程。

我们将在下一节中探讨一些设计模式以及如何使用函数来实现它们。

# 策略模式，功能风格

策略模式可以简要描述为一种结构化代码的方式，它允许在运行时选择算法。面向对象编程的实现使用 DI，你可能已经熟悉 STL 中的面向对象和功能性设计。

让我们来看看 STL `sort`函数。其最复杂的形式需要一个函数对象，如下例所示：

```cpp
class Comparator{
    public: 
        bool operator() (int first, int second) { return (first < second);}
};

TEST_CASE("Strategy"){
    Comparator comparator;
    vector<int> values {23, 1, 42, 83, 52, 5, 72, 11};
    vector<int> expected {1, 5, 11, 23, 42, 52, 72, 83};

    sort(values.begin(), values.end(), comparator);

    CHECK_EQ(values, expected);
}
```

`sort`函数使用`comparator`对象来比较向量中的元素并对其进行排序。这是一种策略模式，因为我们可以用具有相同接口的任何东西来交换`comparator`；实际上，它只需要实现`operator()`函数。例如，我们可以想象一个用户在 UI 中选择比较函数并使用它对值列表进行排序；我们只需要在运行时创建正确的`comparator`实例并将其发送给`sort`函数。

你已经可以看到功能性解决方案的种子。事实上，`sort`函数允许一个更简单的版本，如下例所示：

```cpp
auto compare = [](auto first, auto second) { return first < second;};

TEST_CASE("Strategy"){
    vector<int> values {23, 1, 42, 83, 52, 5, 72, 11};
    vector<int> expected {1, 5, 11, 23, 42, 52, 72, 83};

    sort(values.begin(), values.end(), compare);

    CHECK_EQ(values, expected);
}
```

这一次，我们放弃了仪式感，直接开始实现我们需要的东西——一个可以插入`sort`的比较函数。不再有类，不再有运算符——策略只是一个函数。

让我们看看这在更复杂的情境中是如何工作的。我们将使用维基百科关于*策略模式*的页面上的问题，并使用功能性方法来编写它。

这里有个问题：我们需要为一家酒吧编写一个计费系统，可以在欢乐时光时应用折扣。这个问题适合使用策略模式，因为我们有两种计算账单最终价格的策略——一种返回全价，而另一种返回全账单的欢乐时光折扣（在我们的例子中使用 50%）。再次，解决方案就是简单地使用两个函数来实现这两种策略——`normalBilling`函数只返回它接收到的全价，而`happyHourBilling`函数返回它接收到的值的一半。让我们在下面的代码中看看这个解决方案（来自我的测试驱动开发（TDD）方法）：

```cpp
map<string, double> drinkPrices = {
    {"Westmalle Tripel", 15.50},
    {"Lagavulin 18y", 25.20},
};

auto happyHourBilling = [](auto price){
    return price / 2;
};

auto normalBilling = [](auto price){
    return price;
};

auto computeBill = [](auto drinks, auto billingStrategy){
    auto prices = transformAll<vector<double>>(drinks, [](auto drink){ 
    return drinkPrices[drink]; });
    auto sum = accumulateAll(prices, 0.0, std::plus<double>());
    return billingStrategy(sum);
};

TEST_CASE("Compute total bill from list of drinks, normal billing"){
   vector<string> drinks; 
   double expectedBill;

   SUBCASE("no drinks"){
       drinks = {};
       expectedBill = 0;
   };

   SUBCASE("one drink no discount"){
       drinks = {"Westmalle Tripel"};
       expectedBill = 15.50;
   };

   SUBCASE("one another drink no discount"){
       drinks = {"Lagavulin 18y"};
       expectedBill = 25.20;
   };

  double actualBill = computeBill(drinks, normalBilling);

   CHECK_EQ(expectedBill, actualBill);
}

TEST_CASE("Compute total bill from list of drinks, happy hour"){
   vector<string> drinks; 
   double expectedBill;

   SUBCASE("no drinks"){
       drinks = {};
       expectedBill = 0;
   };

   SUBCASE("one drink happy hour"){
       drinks = {"Lagavulin 18y"};
       expectedBill = 12.60;
   };

   double actualBill = computeBill(drinks, happyHourBilling);

   CHECK_EQ(expectedBill, actualBill);
}
```

我认为这表明，策略的最简单实现是一个函数。我个人喜欢这种模型为策略模式带来的简单性；编写最小的有用代码使事情正常运行是一种解放。

# 命令模式，函数式风格

命令模式是我在工作中广泛使用的一种模式。它与 MVC 网络框架完美契合，允许将控制器分离为多个功能片段，并同时允许与存储格式分离。它的意图是将请求与动作分离开来——这就是它如此多才多艺的原因，因为任何调用都可以被视为一个请求。

命令模式的一个简单用法示例是在支持多个控制器和更改键盘快捷键的游戏中。这些游戏不能直接将*W*键按下事件与移动角色向上的代码关联起来；相反，您将*W*键绑定到`MoveUpCommand`，从而将两者清晰地解耦。我们可以轻松地更改与命令关联的控制器事件或向上移动的代码，而不会干扰两者之间的关系。

当我们看命令在面向对象代码中是如何实现的时，函数式解决方案变得同样明显。`MoveUpCommand`类将如下例所示：

```cpp
class MoveUpCommand{
    public:
        MoveUpCommand(/*parameters*/){}
        void execute(){ /* implementation of the command */}
}
```

我说过这是显而易见的！我们实际上要做的是很容易用一个命名函数来完成，如下例所示：

```cpp
auto moveUpCommand = [](/*parameters*/{
/* implementation */
};
```

最简单的命令模式就是一个函数。谁会想到呢？

# 函数依赖注入

谈论广泛传播的设计模式时，不能不提及 DI。虽然没有在《四人组》的书中定义，但这种模式在现代代码中变得如此普遍，以至于许多程序员认为它是框架或库的一部分，而不是设计模式。

DI 模式的意图是将类或函数的依赖项的创建与其行为分离。为了理解它解决的问题，让我们看看这段代码：

```cpp
auto readFromFileAndAddTwoNumbers = [](){
    int first;
    int second;
    ifstream numbersFile("numbers.txt");
    numbersFile >> first;
    numbersFile >> second;
    numbersFile.close();
    return first + second;
};

TEST_CASE("Reads from file"){
    CHECK_EQ(30, readFromFileAndAddTwoNumbers());
}
```

如果您只需要从文件中读取两个数字并将它们相加，那么这是相当合理的代码。不幸的是，在现实世界中，我们的客户很可能需要更多的读取数字的来源，比如，如下所示，控制台：

```cpp
auto readFromConsoleAndAddTwoNumbers = [](){
    int first;
    int second;
    cout << "Input first number: ";
    cin >> first;
    cout << "Input second number: ";
    cin >> second;
    return first + second;
};

TEST_CASE("Reads from console"){
    CHECK_EQ(30, readFromConsoleAndAddTwoNumbers());
}
```

在继续之前，请注意，此函数的测试只有在您从控制台输入两个和为`30`的数字时才会通过。因为它们需要在每次运行时输入，所以测试用例在我们的代码示例中被注释了；请随意启用它并进行测试。

这两个函数看起来非常相似。为了解决这种相似之处，DI 可以帮助，如下例所示：

```cpp
auto readAndAddTwoNumbers = [](auto firstNumberReader, auto 
    secondNumberReader){
        int first = firstNumberReader();
        int second = secondNumberReader();
        return first + second;
};
```

现在我们可以实现使用文件的读取器：

```cpp

auto readFirstFromFile = [](){
    int number;
    ifstream numbersFile("numbers.txt");
    numbersFile >> number;
    numbersFile.close();
    return number;
};

auto readSecondFromFile = [](){
    int number;
    ifstream numbersFile("numbers.txt");
    numbersFile >> number;
    numbersFile >> number;
    numbersFile.close();
    return number;
};
```

我们还可以实现使用控制台的读取器：

```cpp

auto readFirstFromConsole = [](){
    int number;
    cout << "Input first number: ";
    cin >> number;
    return number;
};

auto readSecondFromConsole = [](){
    int number;
    cout << "Input second number: ";
    cin >> number;
    return number;
};
```

像往常一样，我们可以测试它们在各种组合中是否正确工作，如下所示：

```cpp
TEST_CASE("Reads using dependency injection and adds two numbers"){
    CHECK_EQ(30, readAndAddTwoNumbers(readFirstFromFile, 
        readSecondFromFile));
    CHECK_EQ(30, readAndAddTwoNumbers(readFirstFromConsole, 
        readSecondFromConsole));
    CHECK_EQ(30, readAndAddTwoNumbers(readFirstFromFile, 
        readSecondFromConsole));
}
```

我们通过 lambda 注入了读取数字的代码。请注意测试代码中使用此方法允许我们随心所欲地混合和匹配依赖项——最后一个检查从文件中读取第一个数字，而第二个数字从控制台中读取。

当然，我们通常在面向对象语言中实现 DI 的方式是使用接口和类。然而，正如我们所看到的，实现 DI 的最简单方式是使用函数。

# 纯函数式设计模式

到目前为止，我们已经看到了一些经典面向对象设计模式如何转变为函数变体。但我们能想象出源自函数式编程的设计模式吗？

嗯，我们实际上已经使用了其中一些。`map`/`reduce`（或 STL 中的`transform`/`accumulate`）就是一个例子。大多数高阶函数（如`filter`、`all_of`和`any_of`等）也是模式的例子。然而，我们甚至可以进一步探索一种常见但不透明的设计模式，它源自函数式编程。

理解它的最佳方法是从具体的问题开始。首先，我们将看看如何在不可变的上下文中保持状态。然后，我们将了解设计模式。最后，我们将在另一个上下文中看到它的应用。

# 保持状态

在函数式编程中如何保持状态？鉴于函数式编程背后的一个想法是不可变性，这似乎是一个奇怪的问题，因为不可变性似乎阻止了状态的改变。

然而，这种限制是一种幻觉。为了理解这一点，让我们想一想时间是如何流逝的。如果我戴上帽子，我就会从没戴帽子变成戴帽子。如果我能够一秒一秒地回顾过去，从我伸手拿帽子的那一刻到戴上它，我就能看到我的每一次动作是如何每秒向着这个目标前进的。但我无法改变任何过去的一秒。无论我们喜欢与否，过去是不可改变的（毕竟，也许我戴帽子看起来很傻，但我无法恢复它）。因此，自然使时间以这样的方式运行，过去是不可改变的，但我们可以改变状态。

我们如何在概念上对这进行建模？好吧，这样想一想——首先，我们有一个初始状态，亚历克斯没戴帽子，以及一个意图到达帽子并戴上的运动定义。在编程术语中，我们用一个函数来模拟运动。该函数接收手的位置和函数本身，并返回手的新位置加上函数。因此，通过模仿自然，我们得到了以下示例中的状态序列：

```cpp
Alex wants to put the hat on
Initial state: [InitialHandPosition, MovementFunction (HandPosition -> next HandPosition)]
State1 = [MovementFunction(InitialHandPosition), MovementFunction]
State2 = [MovementFunction(HandPosition at State1),MovementFunction]...
Staten = [MovementFunction(HandPosition at Staten-1), MovementFunction]
until Alex has hat on
```

通过反复应用`MovementFunction`，我们最终得到一系列状态。*每个状态都是不可变的，但我们可以存储状态*。

现在让我们看一个在 C++中的简单例子。我们可以使用的最简单的例子是一个自增索引。索引需要记住上次使用的值，并使用`increment`函数从索引返回下一个值。通常情况下，我们在尝试使用不可变代码实现这一点时会遇到麻烦，但我们可以用之前描述的方法做到吗？

让我们找出来。首先，我们需要用第一个值初始化自增索引——假设它是`1`。像往常一样，我想检查值是否初始化为我期望的值，如下所示：

```cpp
TEST_CASE("Id"){
    const auto autoIncrementIndex = initAutoIncrement(1);
    CHECK_EQ(1, value(autoIncrementIndex)); 
}
```

请注意，由于`autoIncrementIndex`不会改变，我们可以将其设为`const`。

我们如何实现`initAutoIncrement`？正如我们所说，我们需要初始化一个结构，其中包含当前值（在这种情况下为`1`）和增量函数。我将从这样的一对开始：

```cpp
auto initAutoIncrement = [](const int initialId){
    function<int(const int)> nextId = [](const int lastId){
        return lastId + 1;
    };

    return make_pair(initialId, nextId);
};
```

至于之前的`value`函数，它只是返回一对中的值；它是一对中的第一个元素，如下面的代码片段所示：

```cpp
auto value = [](const auto previous){
    return previous.first;
};
```

现在让我们计算一下我们的自增索引的下一个元素。我们初始化它，然后计算下一个值，并检查下一个值是否为`2`：

```cpp
TEST_CASE("Compute next auto increment index"){
    const auto autoIncrementIndex = initAutoIncrement(1);

    const auto nextAutoIncrementIndex = 
        computeNextAutoIncrement(autoIncrementIndex);

    CHECK_EQ(2, value(nextAutoIncrementIndex)); 
}
```

请再次注意，由于它们永远不会变化，所以两个`autoIncrementIndex`变量都是`const`。我们已经有了值函数，但`computeNextAutoIncrement`函数是什么样子的呢？好吧，它必须接受当前值和一对中的函数，将函数应用于当前值，并返回新值和函数之间的一对：

```cpp
auto computeNextAutoIncrement = [](pair<const int, function<int(const 
    int)>> current){
        const auto currentValue = value(current);
        const auto functionToApply = lambda(current);
        const int newValue = functionToApply(currentValue);
        return make_pair(newValue, functionToApply);
};
```

我们正在使用一个实用函数`lambda`，它返回一对中的 lambda：

```cpp
auto lambda = [](const auto previous){
    return previous.second;
};
```

这真的有效吗？让我们测试下一个值：

```cpp
TEST_CASE("Compute next auto increment index"){
    const auto autoIncrementIndex = initAutoIncrement(1);
    const auto nextAutoIncrementIndex = 
        computeNextAutoIncrement(autoIncrementIndex);
    CHECK_EQ(2, value(nextAutoIncrementIndex)); 

 const auto newAutoIncrementIndex = 
        computeNextAutoIncrement(nextAutoIncrementIndex);
 CHECK_EQ(3, value(newAutoIncrementIndex));
}
```

所有的测试都通过了，表明我们刚刚以不可变的方式存储了状态！

由于这个解决方案看起来非常简单，下一个问题是——我们能否将其概括化？让我们试试看。

首先，让我们用`struct`替换`pair`。结构需要有一个值和一个计算下一个值的函数作为数据成员。这将消除我们的`value()`和`lambda()`函数的需要：

```cpp
struct State{
    const int value;
    const function<int(const int)> computeNext;
};
```

`int`类型会重复出现，但为什么呢？状态可能比`int`更复杂，所以让我们把`struct`变成一个模板：

```cpp
template<typename ValueType>
struct State{
    const ValueType value;
    const function<ValueType(const ValueType)> computeNext;
};
```

有了这个，我们可以初始化一个自增索引并检查初始值：

```cpp
auto increment = [](const int current){
    return current + 1;
};

TEST_CASE("Initialize auto increment"){
    const auto autoIncrementIndex = State<int>{1, increment};

    CHECK_EQ(1, autoIncrementIndex.value); 
}
```

最后，我们需要一个计算下一个`State`的函数。该函数需要返回一个`State<ValueType>`，所以最好将其封装到`State`结构中。此外，它可以使用当前值，因此无需将值传递给它：

```cpp
template<typename ValueType>
struct State{
    const ValueType value;
    const function<ValueType(const ValueType)> computeNext;

 State<ValueType> nextState() const{
 return State<ValueType>{computeNext(value), computeNext};
 };
};

```

有了这个实现，我们现在可以检查我们的自动增量索引的下两个值：

```cpp
TEST_CASE("Compute next auto increment index"){
    const auto autoIncrementIndex = State<int>{1, increment};

    const auto nextAutoIncrementIndex = autoIncrementIndex.nextState();

    CHECK_EQ(2, nextAutoIncrementIndex.value); 

    const auto newAutoIncrementIndex = 
        nextAutoIncrementIndex.nextState();
    CHECK_EQ(3, newAutoIncrementIndex.value);
}
```

测试通过了，所以代码有效！现在让我们再玩一会儿。

假设我们正在实现一个简单的井字棋游戏。我们希望在移动后使用相同的模式来计算棋盘的下一个状态。

首先，我们需要一个可以容纳 TicTacToe 棋盘的结构。为简单起见，我将使用`vector<vector<Token>>`，其中`Token`是一个可以容纳`Blank`、`X`或`O`值的`enum`：

```cpp
enum Token {Blank, X, O};
typedef vector<vector<Token>> TicTacToeBoard;
```

然后，我们需要一个`Move`结构。`Move`结构需要包含移动的棋盘坐标和用于进行移动的标记：

```cpp
struct Move{
    const Token token;
    const int xCoord;
    const int yCoord;
};
```

我们还需要一个函数，它可以接受一个`TicTacToeBoard`，应用一个移动，并返回新的棋盘。为简单起见，我将使用本地变异来实现它，如下所示：

```cpp
auto makeMove = [](const TicTacToeBoard board, const Move move) -> 
    TicTacToeBoard {
        TicTacToeBoard nextBoard(board);
        nextBoard[move.xCoord][move.yCoord] = move.token;
         return nextBoard;
};
```

我们还需要一个空白的棋盘来初始化我们的`State`。让我们手工填充`Token::Blank`：

```cpp
const TicTacToeBoard EmptyBoard{
    {Token::Blank,Token::Blank, Token::Blank},
    {Token::Blank,Token::Blank, Token::Blank},
    {Token::Blank,Token::Blank, Token::Blank}
};
```

我们想要进行第一步移动。但是，我们的`makeMove`函数不符合`State`结构允许的签名；它需要一个额外的参数，`Move`。首先，我们可以将`Move`参数绑定到一个硬编码的值。假设`X`移动到左上角，坐标为*(0,0)*：

```cpp
TEST_CASE("TicTacToe compute next board after a move"){
    Move firstMove{Token::X, 0, 0};
    const function<TicTacToeBoard(const TicTacToeBoard)> makeFirstMove 
        = bind(makeMove, _1, firstMove);
    const auto emptyBoardState = State<TicTacToeBoard>{EmptyBoard, 
        makeFirstMove };
    CHECK_EQ(Token::Blank, emptyBoardState.value[0][0]); 

    const auto boardStateAfterFirstMove = emptyBoardState.nextState();
    CHECK_EQ(Token::X, boardStateAfterFirstMove.value[0][0]); 
}
```

如你所见，我们的`State`结构在这种情况下运行良好。但是，它有一个限制：它只允许一次移动。问题在于计算下一个阶段的函数不能更改。但是，如果我们将其作为参数传递给`nextState()`函数呢？我们最终得到了一个新的结构；让我们称之为`StateEvolved`。它保存一个值和一个`nextState()`函数，该函数接受计算下一个状态的函数，应用它，并返回下一个`StateEvolved`：

```cpp
template<typename ValueType>
struct StateEvolved{
    const ValueType value;
    StateEvolved<ValueType> nextState(function<ValueType(ValueType)> 
        computeNext) const{
            return StateEvolved<ValueType>{computeNext(value)};
    };
};
```

现在我们可以通过将`makeMove`函数与绑定到实际移动的`Move`参数一起传递给`nextState`来进行移动：

```cpp
TEST_CASE("TicTacToe compute next board after a move with 
    StateEvolved"){
    const auto emptyBoardState = StateEvolved<TicTacToeBoard>
        {EmptyBoard};
    CHECK_EQ(Token::Blank, emptyBoardState.value[0][0]); 
    auto xMove = bind(makeMove, _1, Move{Token::X, 0, 0});
    const auto boardStateAfterFirstMove = 
        emptyBoardState.nextState(xMove);
    CHECK_EQ(Token::X, boardStateAfterFirstMove.value[0][0]); 
}
```

我们现在可以进行第二步移动。假设`O`移动到坐标*(1,1)*的中心。让我们检查前后状态：

```cpp
    auto oMove = bind(makeMove, _1, Move{Token::O, 1, 1});
    const auto boardStateAfterSecondMove = 
        boardStateAfterFirstMove.nextState(oMove);
    CHECK_EQ(Token::Blank, boardStateAfterFirstMove.value[1][1]); 
    CHECK_EQ(Token::O, boardStateAfterSecondMove.value[1][1]); 
```

正如你所看到的，使用这种模式，我们可以以不可变的方式存储任何状态。

# 揭示

我们之前讨论的设计模式对函数式编程似乎非常有用，但你可能已经意识到我一直在避免命名它。

事实上，到目前为止我们讨论的模式是单子的一个例子，具体来说是`State`单子。我一直避免告诉你它的名字，因为单子在软件开发中是一个特别晦涩的话题。对于这本书，我观看了数小时的单子视频；我还阅读了博客文章和文章，但出于某种原因，它们都无法理解。由于单子是范畴论中的一个数学对象，我提到的一些资源采用数学方法，并使用定义和运算符来解释它们。其他资源尝试通过示例来解释，但它们是用具有对单子模式的本地支持的编程语言编写的。它们都不符合我们这本书的目标——对复杂概念的实际方法。

要更好地理解单子，我们需要看更多的例子。最简单的例子可能是`Maybe`单子。

# 也许

考虑尝试在 C++中计算以下表达式：

```cpp
2  + (3/0) * 5
```

可能会发生什么？通常会抛出异常，因为我们试图除以`0`。但是，有些情况下，我们希望看到一个值，比如`None`或`NaN`，或者某种消息。我们已经看到，我们可以使用`optional<int>`来存储可能是整数或值的数据；因此，我们可以实现一个返回`optional<int>`的除法函数，如下所示：

```cpp
    function<optional<int>(const int, const int)> divideEvenWith0 = []
      (const int first, const int second) -> optional<int>{
        return (second == 0) ? nullopt : make_optional(first / second);
    };
```

然而，当我们尝试在表达式中使用`divideEvenWith0`时，我们意识到我们还需要改变所有其他操作符。例如，我们可以实现一个`plusOptional`函数，当任一参数为`nullopt`时返回`nullopt`，否则返回值，如下例所示：

```cpp
    auto plusOptional = [](optional<int> first, optional<int> second) -
        > optional<int>{
            return (first == nullopt || second == nullopt) ? 
                nullopt :
            make_optional(first.value() + second.value());
    };
```

虽然它有效，但这需要编写更多的函数和大量的重复。但是，嘿，我们能写一个函数，它接受一个`function<int(int, int)>`并将其转换为`function<optional<int>(optional<int>, optional<int>)`吗？当然，让我们编写以下函数：

```cpp
    auto makeOptional = [](const function<int(int, int)> operation){
        return operation -> optional<int>{
            if(first == nullopt || second == nullopt) return nullopt;
            return make_optional(operation(first.value(), 
                second.value()));
        };
    };
```

这很好地运行了，如下所示通过了测试：

```cpp
    auto plusOptional = makeOptional(plus<int>());
    auto divideOptional = makeOptional(divides<int>());

    CHECK_EQ(optional{3}, plusOptional(optional{1}, optional{2}));
    CHECK_EQ(nullopt, plusOptional(nullopt, optional{2}));

    CHECK_EQ(optional{2}, divideOptional(optional{2}, optional{1}));
    CHECK_EQ(nullopt, divideOptional(nullopt, optional{1}));
```

然而，这并没有解决一个问题——当除以`0`时，我们仍然需要返回`nullopt`。因此，以下测试将失败如下：

```cpp
//    CHECK_EQ(nullopt, divideOptional(optional{2}, optional{0}));
//    cout << "Result of 2 / 0 = " << to_string(divideOptional
        (optional{2}, optional{0})) << endl;
```

我们可以通过使用我们自己的`divideEvenBy0`方法来解决这个问题，而不是使用标准的除法：

```cpp
    function<optional<int>(const int, const int)> divideEvenWith0 = []
      (const int first, const int second) -> optional<int>{
        return (second == 0) ? nullopt : make_optional(first / second);
    };

```

这次，测试通过了，如下所示：

```cpp
    auto divideOptional = makeOptional(divideEvenWith0);

    CHECK_EQ(nullopt, divideOptional(optional{2}, optional{0}));
    cout << "Result of 2 / 0 = " << to_string(divideOptional
        (optional{2}, optional{0})) << endl;
```

此外，运行测试后的显示如下：

```cpp
Result of 2 / 0 = None
```

我不得不说，摆脱除以`0`的暴政并得到一个结果有一种奇怪的满足感。也许这只是我。

无论如何，这引导我们来定义`Maybe`单子。它存储一个值和一个名为`apply`的函数。`apply`函数接受一个操作（`plus<int>()`，`minus<int>()`，`divideEvenWith0`，或`multiplies<int>()`），以及一个要应用操作的第二个值，并返回结果：

```cpp
template<typename ValueType>
struct Maybe{
    typedef function<optional<ValueType>(const ValueType, const 
        ValueType)> OperationType;
    const optional<ValueType> value;

    optional<ValueType> apply(const OperationType operation, const 
        optional<ValueType> second){
            if(value == nullopt || second == nullopt) return nullopt;
            return operation(value.value(), second.value());
    }
};
```

我们可以使用`Maybe`单子来进行计算如下：

```cpp
TEST_CASE("Compute with Maybe monad"){
    function<optional<int>(const int, const int)> divideEvenWith0 = []
      (const int first, const int second) -> optional<int>{
        return (second == 0) ? nullopt : make_optional(first / second);
    };

    CHECK_EQ(3, Maybe<int>{1}.apply(plus<int>(), 2));
    CHECK_EQ(nullopt, Maybe<int>{nullopt}.apply(plus<int>(), 2));
    CHECK_EQ(nullopt, Maybe<int>{1}.apply(plus<int>(), nullopt));

    CHECK_EQ(2, Maybe<int>{2}.apply(divideEvenWith0, 1));
    CHECK_EQ(nullopt, Maybe<int>{nullopt}.apply(divideEvenWith0, 1));
    CHECK_EQ(nullopt, Maybe<int>{2}.apply(divideEvenWith0, nullopt));
    CHECK_EQ(nullopt, Maybe<int>{2}.apply(divideEvenWith0, 0));
    cout << "Result of 2 / 0 = " << to_string(Maybe<int>
        {2}.apply(divideEvenWith0, 0)) << endl;
}
```

再次，我们可以计算表达式，即使有`nullopt`。

# 那么单子是什么？

**单子**是一种模拟计算的函数式设计模式。它来自数学；更确切地说，来自称为**范畴论**的领域。

什么是计算？基本计算是一个函数；但是，我们有兴趣为函数添加更多的行为。我们已经看到了维护状态和允许可选类型操作的两个例子，但是单子在软件设计中是相当普遍的。

单子基本上有一个值和一个高阶函数。为了理解它们的作用，让我们来比较以下代码中显示的`State`单子：

```cpp
template<typename ValueType>
struct StateEvolved{
    const ValueType value;

    StateEvolved<ValueType> nextState(function<ValueType(ValueType)> 
        computeNext) const{
            return StateEvolved<ValueType>{computeNext(value)};
    };
};
```

使用此处显示的`Maybe`单子：

```cpp
template<typename ValueType>
struct Maybe{
    typedef function<optional<ValueType>(const ValueType, const 
        ValueType)> OperationType;
    const optional<ValueType> value;

    optional<ValueType> apply(const OperationType operation, const 
        optional<ValueType> second) const {
            if(value == nullopt || second == nullopt) return nullopt;
            return operation(value.value(), second.value());
    }
};
```

它们都包含一个值。该值封装在单子结构中。它们都包含一个对该值进行计算的函数。`apply`/`nextState`（在文献中称为`bind`）函数本身接收一个封装计算的函数；但是，单子除了计算之外还做了一些其他事情。

单子还有更多的内容，不仅仅是这些简单的例子。但是，它们展示了如何封装某些计算以及如何消除某些类型的重复。

值得注意的是，C++中的`optional<>`类型实际上是受到了`Maybe`单子的启发，以及承诺，因此您可能已经在代码中使用了等待被发现的单子。

# 总结

在本章中，我们学到了很多关于改进设计的知识。我们了解到重构意味着重构代码而不改变程序的外部行为。我们看到为了确保行为的保留，我们需要采取非常小的步骤和测试。我们了解到遗留代码是我们害怕改变的代码，为了为其编写测试，我们需要首先更改代码，这导致了一个困境。我们还学到，幸运的是，我们可以对代码进行一些小的更改，这些更改保证了行为的保留，但打破了依赖关系，从而允许我们通过测试插入代码。然后我们看到，我们可以使用纯函数来识别和打破依赖关系，从而导致我们可以根据内聚性将它们重新组合成类。

最后，我们了解到我们可以在函数式编程中使用设计模式，并且看到了一些例子。即使您不使用函数式编程的其他内容，使用策略、命令或注入依赖等函数将使您的代码更容易进行最小干扰的更改。我们提到了一个非常抽象的设计模式，单子，以及我们如何使用`Maybe`单子和`State`单子。这两者都可以在我们的写作中帮助我们更少的代码实现更丰富的功能。

我们已经讨论了很多关于软件设计的内容。但是函数式编程是否适用于架构？这就是我们将在下一章中讨论的内容——事件溯源。
