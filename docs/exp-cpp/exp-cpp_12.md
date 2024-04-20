# 第十章：设计面向全球的应用程序

在生产就绪项目中使用编程语言是学习语言本身的一个全新步骤。有时，这本书中的简单示例可能会在实际项目中采用不同的方法或面临许多困难。当理论遇到实践时，你才会学会这门语言。C++也不例外。学习语法、解决一些书中的问题或理解书中的一些简单示例是不同的。在创建真实世界的应用程序时，我们面临着不同范围的挑战，有时书籍缺乏支持实际问题的理论。

在本章中，我们将尝试涵盖使用 C++进行实际编程的基础知识，这将帮助你更好地处理真实世界的应用程序。复杂的项目需要大量的思考和设计。有时，程序员不得不完全重写项目，并从头开始，只是因为他们在开发初期做出了糟糕的设计选择。本章试图尽最大努力阐明软件设计的过程。你将学习更好地为你的项目设计架构的步骤。

我们将在本章中涵盖以下主题：

+   了解项目开发生命周期

+   设计模式及其应用

+   领域驱动设计

+   以亚马逊克隆为例的真实项目设计

# 技术要求

本章中使用`-std=c++2a`选项的 g++编译器来编译示例。你可以在[`github.com/PacktPublishing/Expert-CPP`](https://github.com/PacktPublishing/Expert-CPP)找到本章中使用的源文件。

# 项目开发生命周期

每当你面对一个问题时，你应该仔细考虑需求分析的过程。项目开发中最大的错误之一是在没有对问题本身进行彻底分析的情况下开始编码。

想象一种情况，你被要求创建一个计算器，一个简单的工具，允许用户对数字进行算术计算。假设你神奇地按时完成了项目并发布了程序。现在，用户开始使用你的计算器，迟早会发现他们的计算结果不会超过整数的最大值。当他们抱怨这个问题时，你准备用坚实的编码支持论据来为自己（和你的作品）辩护，比如这是因为在计算中使用了`int`数据类型。对你和你的同行程序员来说，这是完全可以理解的，但最终用户却无法接受你的论点。他们想要一个可以对足够大的数字进行求和的工具，否则他们根本不会使用你的程序。你开始着手下一个版本的计算器，这一次，你使用长整型甚至自定义实现的大数。当你自豪地将程序交付给等待你掌声的用户时，你突然意识到同样的用户抱怨没有功能来找到数字的对数或指数。这似乎令人生畏，因为可能会有越来越多的功能请求和越来越多的抱怨。

尽管这个例子有点简单，但它完全覆盖了真实世界中通常发生的情况。即使你为你的程序实现了所有功能，并考虑着去度一个值得的长假，用户也会开始抱怨程序中的错误。事实证明，有几种情况下，你的计算器表现出乎意料的行为，给出了错误的结果。迟早，你会意识到在将程序发布给大众之前，需要进行适当的测试。

我们将涉及在处理真实世界项目时应考虑的主题。每当你开始一个新项目时，应考虑以下步骤：

1.  需求收集和分析

1.  规格书创建

1.  设计和测试规划

1.  编码

1.  测试和稳定性

1.  发布和维护

前面的步骤并非对每个项目都是硬性规定，尽管它可能被认为是每个软件开发团队应该完成以实现成功产品发布的最低要求。实际上，由于 IT 领域的每个人最缺乏的是时间，大多数步骤都被省略了。但是，强烈建议遵循前面的步骤，因为最终它将在长期节省更多时间。

# 需求收集和分析

这是创建稳定产品的最关键步骤。程序员未能按时完成任务或在代码中留下许多错误的最常见原因之一是对项目的完全理解不足。

领域知识是如此重要，以至于在任何情况下都不应该被忽略。您可能很幸运地开发与您非常了解的内容相关的项目。但是，您应该考虑到并非每个人都像您一样幸运（嗯，您也可能是那么不幸）。

想象一下，您正在开发一个自动化分析和报告某家公司股票交易的项目。现在想象一下，您对股票和股票交易一无所知。您不了解熊市或牛市，交易交易的限制等等。您如何才能成功完成这个项目？

即使您了解股票市场和交易，您可能不了解下一个重大项目领域。如果您被要求设计和实施（有或没有团队）控制您所在城市气象站的项目，您在开始项目时会首先做什么？

您绝对应该从需求收集和分析开始。这只是一个涉及与客户沟通并就项目提出许多问题的过程。如果您没有与任何客户打交道，而是在一家产品公司工作，项目经理应被视为客户。即使项目是您的想法，您是独自工作，您也应该将自己视为客户，并且，尽管这听起来很荒谬，但要问自己很多问题（关于项目）。

假设我们要征服电子商务，并希望发布一个最终能够击败市场上的大鳄的产品。受欢迎和成功的电子商务市场包括亚马逊，eBay，阿里巴巴等。我们应该将问题陈述为“编写我们自己的亚马逊克隆”。我们应该如何收集项目的需求？

首先，我们应该列出所有我们应该实现的功能，然后我们会进行优先排序。例如，对于亚马逊克隆项目，我们可能会列出以下功能清单：

+   创建产品。

+   列出产品。

+   购买产品。

+   编辑产品细节。

+   移除产品。

+   按名称，价格范围和重量搜索产品。

+   偶尔通过电子邮件提醒用户产品的可用性。

功能应尽可能详细地描述；这将为开发人员（在这种情况下是您）解决问题。例如，创建产品应该由项目管理员或任何用户完成。如果用户可以创建产品，那么可能会有限制。可能会有用户错误地在我们的系统中创建数百个产品，以增加他们唯一产品的可见性。

详细信息应在与客户的沟通中说明，讨论和最终确定。如果您独自承担项目并且是项目的客户，则沟通是在项目需求上“为自己思考”的过程。

在获取需求完成后，我们建议对每个功能进行优先排序，并将它们分类为以下类别之一：

+   必须有

+   应该有

+   最好有

经过更多思考并对前述功能进行分类后，我们可以列出以下清单：

+   创建产品[必须有]。

+   列出产品[必须有]。

+   购买产品[必须有]。

+   编辑产品细节[应该有]。

+   移除产品[必须有]。

+   按名称搜索产品[必须有]。

+   按价格范围搜索产品[应该有]。

+   按重量搜索产品[很好有]。

+   偶尔通过电子邮件提醒用户产品的可用性[很好有]。

分类将为您提供一个从哪里开始的基本想法。程序员是贪婪的人；他们想要为他们的产品实现每一个可能的功能。这是通向失败的确定途径。你应该从最基本的功能开始——这就是为什么我们有一些很好的功能。有些人开玩笑地坚持认为，应该将很好的功能重新命名为永远不会有的功能，因为在实践中，它们永远不会被实现。

# 规格创建

并不是每个人都喜欢创建规格。嗯，大多数程序员讨厌这一步，因为这不是编码，而是写作。

在收集项目需求之后，你应该创建一个包含描述你的项目的每个细节的文档。这种规格有许多名称和类型。它可能被称为**项目需求文档**（**PRD**），**功能规格**，**开发规格**等等。认真的程序员和团队会在需求分析的结果中产生一个 PRD。这些认真的人的下一步是创建功能规格以及开发规格等等。我们将所有文档组合在一个名为**规格创建**的单一步骤中。

是否需要之前提到的任何子文档，这取决于你和你的团队。甚至最好有一个产品的视觉表示，而不是一个文本文档。无论你的文档采取什么形式，它都应该仔细地代表你在需求收集步骤中所取得的成就。为了对此有一个基本的理解，让我们试着记录一些我们之前收集到的功能（我们将把我们的项目称为*平台）*

+   创建产品。平台的用户具有管理员特权可以创建产品。

+   平台必须允许创建具有定义特权的用户。在这一点上，应该有两种类型的用户，即普通用户和管理员用户。

+   使用平台的任何用户都必须能够看到可用产品的列表。

+   产品应该有图片、价格、名称、重量和描述。

+   购买产品时，用户提供他们的卡片详细信息以结账和产品装运的详细信息。

+   每个注册用户都应该提供一个送货地址、信用卡详细信息和一个电子邮件账户。

列表可能会很长，实际上应该很长，因为列表越长，开发人员就越了解项目。

# 设计和测试规划

尽管我们坚持认为需求收集步骤是软件开发中最关键的一步，但设计和测试规划也可以被认为是同样关键的一步。如果你曾经在没有先设计项目的情况下开始一个项目，你已经知道它是不可能的。尽管激励性的语录坚持认为没有什么是不可能的，程序员确信至少有一件事是不可能的，那就是在没有先设计项目的情况下成功完成一个项目。

设计的过程是最有趣的一步；它迫使我们思考、绘画、再次思考、清理一切，然后重新开始。在设计项目时，你应该从顶部开始。首先，列出所有在项目中以某种方式涉及的实体和过程。以亚马逊克隆为例，我们可以列出以下实体和过程：

+   用户

+   注册和授权

+   产品

+   交易

+   仓库（包含产品）

+   装运

这是一个高层设计——一个通过最终设计的起点。在这一章中，我们将主要集中在项目的设计上。

# 分解实体

在列出关键实体和流程之后，我们开始将它们分解为更详细的实体，稍后将转换为类。最好还是勾画一下项目的设计。只需绘制包含实体名称的矩形，并用箭头连接它们，如果它们有某种联系或是同一流程的一部分。如果有一个包含或由实体 A 开始的流程，并在实体 B 结束或导致实体 B，你可以从实体 A 开始一个箭头指向实体 B。图画得多好并不重要，这是更好地理解项目的必要步骤。例如，看看下面的图表：

![](img/747174ee-fe32-4e39-930e-9add32033617.png)

将实体和流程分解为类及其相互通信是一种需要耐心和一致性的微妙艺术。例如，让我们尝试为**User**实体添加细节。根据规范创建步骤中所述，注册用户应提供交货地址、电子邮件地址和信用卡详细信息。让我们绘制一个代表用户的类图：

![](img/62615d23-0f68-42b3-914a-380356d7ba16.png)

现在出现了一个有趣的问题：我们应该如何处理实体内包含的复杂类型？例如，用户的交货地址是一个复杂类型。它不能只是`string`，因为迟早我们可能需要按照用户的交货地址对用户进行排序，以进行最佳的发货。例如，如果用户的交货地址与包含所购产品的仓库的地址不在同一个国家，那么货运公司可能会花费我们（或用户）一大笔钱。这是一个很好的场景，因为它引入了一个新问题，并更新了我们对项目的理解。原来我们应该处理的情况是，当用户订购的产品分配给一个距离用户物理位置很远的特定仓库时。如果我们有很多仓库，我们应该选择离用户最近的一个，其中包含所需的产品。这些问题不能立即得到答案，但这是设计项目的高质量结果。否则，这些问题将在编码过程中出现，并且我们会陷入其中比我们预想的时间更长的困境中。在任何已知的宇宙中，项目的初始估计都无法满足其完成日期。

那么，如何在`User`类中存储用户地址呢？如下例所示，简单的`std::string`就可以：

```cpp
class User
{
public:
  // code omitted for brevity
private:
  std::string address_;
  // code omitted for brevity
};
```

地址在其组成部分方面是一个复杂的对象。地址可能包括国家名称、国家代码、城市名称和街道名称，甚至可能包含纬度和经度。如果需要找到用户最近的仓库，后者就非常有用。为程序员创建更多类型以使设计更直观也是完全可以的。例如，以下结构可能非常适合表示用户的地址：

```cpp
struct Address
{
  std::string country;
  std::string city;
  std::string street;
  float latitude{};
  float longitude{};
};
```

现在，存储用户地址变得更加简单：

```cpp
class User
{
  // code omitted for brevity
  Address address_;
}; 
```

我们稍后会在本章回到这个例子。

设计项目的过程可能需要回到几个步骤来重新阐明项目需求。在澄清设计步骤之后，我们可以继续将项目分解为更小的组件。创建交互图也是一个不错的选择。

像下面这样的交互图将描述一些操作，比如**用户**进行**购买**产品的交易：

![](img/0fc423b9-d3fb-4643-abb0-aadd1f55632b.png)

测试规划也可以被视为设计的一部分。它包括规划最终应用程序将如何进行测试。例如，之前的步骤包括一个地址的概念，结果发现，地址可以包含国家、城市等。一个合适的测试应该包括检查用户地址中的国家值是否可以成功设置。尽管测试规划通常不被认为是程序员的任务，但为您的项目做测试规划仍然是一种良好的实践。一个合适的测试计划会在设计项目时产生更多有用的信息。大多数输入数据处理和安全检查都是在测试规划中发现的。例如，在进行需求分析或编写功能规范时，可能不会考虑对用户名称或电子邮件地址设置严格限制。测试规划关心这样的情况，并迫使开发人员注意数据检查。然而，大多数程序员都急于达到项目开发的下一步，编码。

# 编码

正如之前所说，编码并不是项目开发的唯一部分。在编码之前，您应该通过利用规范中的所有需求来仔细设计您的项目。在项目开发的前几步彻底完成后，编码会变得更加容易和高效。

一些团队实践**测试驱动开发（TDD）**，这是生产更加稳定的项目发布的好方法。TDD 的主要概念是在项目实现之前编写测试。这对程序员来说是定义项目需求和在开发过程中出现的进一步问题的一个很好的方法。

假设我们正在为`User`类实现 setter。用户对象包含了之前讨论过的 email 字段，这意味着我们应该有一个`set_email()`方法，如下面的代码片段所示：

```cpp
class User
{
public:
  // code omitted for brevity
  void set_email(const std::string&);

private: 
  // code omitted for brevity
  std::string email_;
};
```

TDD 方法建议在实现`set_email()`方法之前编写一个测试函数。假设我们有以下测试函数：

```cpp
void test_set_email()
{
  std::string valid_email = "valid@email.com";
  std::string invalid_email = "112%$";
  User u;
  u.set_email(valid_email);
  u.set_email(invalid_email);
}
```

在上面的代码中，我们声明了两个`string`变量，其中一个包含了一个无效的电子邮件地址值。甚至在运行测试函数之前，我们就知道，在无效数据输入的情况下，`set_email()`方法应该以某种方式做出反应。常见的方法之一是抛出一个指示无效输入的异常。您也可以在`set_email`的实现中忽略无效输入，并返回一个指示操作成功的`boolean`值。错误处理应该在项目中保持一致，并得到所有团队成员的认可。假设我们选择抛出异常，因此，测试函数应该在将无效值传递给方法时期望一个异常。

然后，上述代码应该被重写如下：

```cpp
void test_set_email()
{
  std::string valid_email = "valid@email.com";
  std::string invalid_email = "112%$";

  User u;
  u.set_email(valid_email);
  if (u.get_email() == valid_email) {
    std::cout << "Success: valid email has been set successfully" << std::endl;
  } else {
    std::cout << "Fail: valid email has not been set" << std::endl;
  }

  try {
    u.set_email(invalid_email);
    std::cerr << "Fail: invalid email has not been rejected" << std::endl;
  } catch (std::exception& e) {
    std::cout << "Success: invalid email rejected" << std::endl;
  }
}
```

测试函数看起来已经完成。每当我们运行测试函数时，它会输出`set_email()`方法的当前状态。即使我们还没有实现`set_email()`函数，相应的测试函数也是实现细节的重要一步。我们现在基本上知道了这个函数应该如何对有效和无效的数据输入做出反应。我们可以添加更多种类的数据来确保`set_email()`方法在实现完成时得到充分测试。例如，我们可以用空字符串和长字符串来测试它。

这是`set_email()`方法的初始实现：

```cpp
#include <regex>
#include <stdexcept>

void User::set_email(const std::string& email)
{
  if (!std::regex_match(email, std::regex("(\\w+)(\\.|_)?(\\w*)@(\\w+)(\\.(\\w+))+")) {
    throw std::invalid_argument("Invalid email");
  }

  this->email_ = email;
}
```

在方法的初始实现之后，我们应该再次运行我们的测试函数，以确保实现符合定义的测试用例。

为项目编写测试被认为是一种良好的编码实践。有不同类型的测试，如单元测试、回归测试、冒烟测试等。开发人员应该为他们的项目支持单元测试覆盖率。

编码过程是项目开发生命周期中最混乱的步骤之一。很难估计一个类或其方法的实现需要多长时间，因为大部分问题和困难都是在编码过程中出现的。本章开头描述的项目开发生命周期的前几个步骤往往涵盖了大部分这些问题，并简化了编码过程。

# 测试和稳定

项目完成后，应进行适当的测试。通常，软件开发公司会有**质量保证**（**QA**）工程师，他们会细致地测试项目。

在测试阶段验证的问题会转化为相应的任务分配给程序员来修复。问题可能会影响项目的发布，也可能被归类为次要问题。

程序员的基本任务不是立即修复问题，而是找到问题的根本原因。为了简单起见，让我们看一下`generate_username()`函数，它使用随机数与电子邮件结合生成用户名：

```cpp
std::string generate_username(const std::string& email)
{
  int num = get_random_number();
  std::string local_part = email.substr(0, email.find('@'));
  return local_part + std::to_string(num);
}
```

`generate_username()`函数调用`get_random_number()`将返回的值与电子邮件地址的本地部分组合在一起。本地部分是电子邮件地址中`@`符号之前的部分。

QA 工程师报告说，与电子邮件的本地部分相关联的数字总是相同的。例如，对于电子邮件`john@gmail.com`，生成的用户名是`john42`，对于`amanda@yahoo.com`，是`amanda42`。因此，下次使用电子邮件`amanda@hotmail.com`尝试在系统中注册时，生成的用户名`amanda42`与已存在的用户名冲突。测试人员不了解项目的实现细节是完全可以的，因此他们将其报告为用户名生成功能中的问题。虽然你可能已经猜到真正的问题隐藏在`get_random_number()`函数中，但总会有情况出现，问题被修复而没有找到其根本原因。错误的方法修复问题可能会改变`generate_username()`函数的实现。`generate_random_number()`函数也可能在其他函数中使用，这将使调用`get_random_number()`的所有函数工作不正确。虽然这个例子很简单，但深入思考并找到问题的真正原因至关重要。这种方法将节省大量时间。

# 发布和维护

在修复所有关键和重大问题使项目变得相对稳定之后，可以发布项目。有时公司会在软件上加上**beta**标签，以防用户发现有 bug 时有借口。需要注意的是，很少有软件能够完美无缺地运行。发布后，会出现更多问题。因此，维护阶段就会到来，开发人员会在修复和发布更新时工作。

程序员有时开玩笑说，发布和维护是永远无法实现的步骤。然而，如果你花足够的时间设计项目，发布第一个版本就不会花费太多时间。正如我们在前一节中已经介绍的，设计从需求收集开始。之后，我们花时间定义实体，分解它们，将其分解为更小的组件，编码，测试，最后发布。作为开发人员，我们对设计和编码阶段更感兴趣。正如已经指出的，良好的设计选择对进一步的项目开发有很大的影响。现在让我们更仔细地看一下整个设计过程。

# 深入设计过程

如前所述，项目设计始于列出一般实体，如用户、产品和仓库，当设计电子商务平台时：

![](img/b8991f30-493e-4e8f-84d7-02f1b58ed92e.png)

然后我们将每个实体分解为更小的组件。为了使事情更清晰，将每个实体视为一个单独的类。将实体视为类时，在分解方面更有意义。例如，我们将`user`实体表示为一个类：

```cpp
class User
{
public:
  // constructors and assignment operators are omitted for code brevity
  void set_name(const std::string& name);
  std::string get_name() const;
  void set_email(const std::string&);
  std::string get_email() const;
  // more setters and getters are omitted for code brevity

private:
  std::string name_;
  std::string email_;
  Address address_;
  int age;
};
```

`User`类的类图如下：

![](img/bd7b14fe-6cee-47bd-be57-e54cd658200b.png)

然而，正如我们已经讨论过的那样，`User`类的地址字段可能被表示为一个单独的类型（`class`或`struct`，目前并不重要）。无论是数据聚合还是复杂类型，类图都会发生以下变化：

![](img/a62f3623-ed77-4cca-9d29-fffe8c9a0bfc.png)

这些实体之间的关系将在设计过程中变得清晰。例如，**Address**不是一个独立的实体，它是**User**的一部分，也就是说，如果没有实例化**User**对象，它就不能有一个实例。然而，由于我们可能希望指向可重用的代码，**Address**类型也可以用于仓库对象。也就是说，**User**和**Address**之间的关系是简单的聚合而不是组合。

在讨论支付选项时，我们可能会对**User**类型提出更多要求。平台的用户应该能够插入支付产品的选项。在决定如何在`User`类中表示支付选项之前，我们应该首先找出这些选项是什么。让我们保持简单，假设支付选项是包含信用卡号、持卡人姓名、到期日和卡的安全码的选项。这听起来像另一个数据聚合，所以让我们将所有这些内容收集到一个单独的结构体中，如下所示：

```cpp
struct PaymentOption
{
  std::string number;
  std::string holder_name;
  std::chrono::year_month expiration_date;
  int code;
};
```

请注意前面结构体中的`std::chrono::year_month`；它表示特定年份的特定月份，是在 C++20 中引入的。大多数支付卡只包含卡的到期月份和年份，因此这个`std::chrono::year_month`函数非常适合`PaymentOption`。

因此，在设计`User`类的过程中，我们提出了一个新类型`PaymentOption`。用户可以拥有多个支付选项，因此`User`和`PaymentOption`之间的关系是一对多的。现在让我们用这个新的聚合更新`User`类的类图（尽管在这种情况下我们使用组合）：

![](img/c85c2d3a-0d3f-4143-8879-b5cf71ef29dd.png)

`User`和`PaymentOption`之间的依赖关系在以下代码中表示：

```cpp
class User
{
public:
  // code omitted for brevity
  void add_payment_option(const PaymentOption& po) {
    payment_options_.push_back(op);
  }

  std::vector get_payment_options() const {
    return payment_options_;
  }
private:
  // code omitted for brevity
  std::vector<PaymentOption> payment_options_;
};
```

我们应该注意，即使用户可能设置了多个支付选项，我们也应该将其中一个标记为主要选项。这很棘手，因为我们可以将所有选项存储在一个向量中，但现在我们必须将其中一个设为主要选项。

我们可以使用一对或`tuple`（如果想要花哨一点）将向量中的选项与`boolean`值进行映射，指示它是否是主要选项。以下代码描述了之前引入的`User`类中元组的使用：

```cpp
class User
{
public:
  // code omitted for brevity
  void add_payment_option(const PaymentOption& po, bool is_primary) {
    payment_options_.push_back(std::make_tuple(po, is_primary));
  }

  std::vector<std::tuple<PaymentOption, boolean> > get_payment_options() const {
    return payment_options_;
  }
private:
  // code omitted for brevity
  std::vector<std::tuple<PaymentOption, boolean> > payment_options_;
};
```

我们可以通过以下方式利用类型别名简化代码：

```cpp
class User
{
public:
  // code omitted for brevity
  using PaymentOptionList = std::vector<std::tuple<PaymentOption, boolean> >;

  // add_payment_option is omitted for brevity
  PaymentOptionList get_payment_options() const {
    return payment_options_;
  }

private:
  // code omitted for brevity
  PaymentOptionList payment_options_;
};
```

以下是用户类如何检索用户的主要支付选项的方法：

```cpp
User john = get_current_user(); // consider the function is implemented and works
auto payment_options = john.get_payment_options();
for (const auto& option : payment_options) {
  auto [po, is_primary] = option;
  if (is_primary) {
    // use the po payment option
  }
}
```

在`for`循环中访问元组项时，我们使用了结构化绑定。然而，在学习了关于数据结构和算法的章节之后，您现在意识到搜索主要支付选项是一个线性操作。每次需要检索主要支付选项时循环遍历向量可能被认为是一种不好的做法。

您可能会更改底层数据结构以使事情运行更快。例如，`std::unordered_map`（即哈希表）听起来更好。但是，这并不会使事情变得更快，仅仅因为它可以在常数时间内访问其元素。在这种情况下，我们应该将`boolean`值映射到支付选项。对于除一个之外的所有选项，`boolean`值都是相同的假值。这将导致哈希表中的冲突，这将由将值链接在一起映射到相同哈希值的方式来处理。使用哈希表的唯一好处将是对主要支付选项进行常数时间访问。

最后，我们来到了将主要支付选项单独存储在类中的最简单的解决方案。以下是我们应该如何重写`User`类中处理支付选项的部分：

```cpp
class User
{
public:
  // code omitted for brevity
  using PaymentOptionList = std::vector<PaymentOption>;
  PaymentOption get_primary_payment_option() const {
    return primary_payment_option_;
  }

  PaymentOptionList get_payment_options() const {
    return payment_options_;
  }

  void add_payment_option(const PaymentOption& po, bool is_primary) {
    if (is_primary) {
      // moving current primary option to non-primaries
      add_payment_option(primary_payment_option_, false);
      primary_payment_option_ = po;
      return;
    }
    payment_options_.push_back(po);
  }

private:
  // code omitted for brevity
  PaymentOption primary_payment_option_;
  PaymentOptionList payment_options_;
};
```

到目前为止，我们已经带您了解了存储支付选项的方式的过程，只是为了展示设计伴随编码的过程。尽管我们为支付选项的单一情况创建了许多版本，但这并不是最终版本。在支付选项向量中处理重复值的情况总是存在。每当您将一个支付选项添加为主要选项，然后再添加另一个选项为主要选项时，先前的选项将移至非主要列表。如果我们改变主意并再次将旧的支付选项添加为主要选项，它将不会从非主要列表中移除。

因此，总是有机会深入思考并避免潜在问题。设计和编码是相辅相成的；然而，您不应忘记 TDD。在大多数情况下，在编码之前编写测试将帮助您发现许多用例。

# 使用 SOLID 原则

在项目设计中，您可以使用许多原则和设计方法。保持设计简单总是更好，但是有些原则在一般情况下几乎对所有项目都有用。例如，**SOLID**包括五个原则，其中的一个或全部可以对设计有用。

SOLID 代表以下原则：

+   单一职责

+   开闭原则

+   里氏替换

+   接口隔离

+   依赖反转

让我们通过示例讨论每个原则。

# 单一职责原则

单一职责原则简单地说明了一个对象，一个任务。尽量减少对象的功能和它们的关系复杂性。使每个对象只负责一个任务，即使将复杂对象分解为更小更简单的组件并不总是容易的。单一职责是一个上下文相关的概念。它不是指类中只有一个方法；而是使类或模块负责一个事情。例如，我们之前设计的`User`类只有一个职责：存储用户信息。然而，我们将支付选项添加到`User`类中，并强制它具有添加和删除支付选项的方法。我们还引入了主要支付选项，这涉及**User**方法中的额外逻辑。我们可以朝两个方向发展。

第一个建议将`User`类分解为两个单独的类。每个类将负责一个单一的功能。以下类图描述了这个想法：

![](img/711b0c46-dd8c-438a-904c-c1727528681d.png)

其中一个将仅存储用户的基本信息，下一个将存储用户的支付选项。我们分别命名它们为`UserInfo`和`UserPaymentOptions`。有些人可能会喜欢这种新设计，但我们会坚持旧的设计。原因在于，`User`类既包含用户信息又包含支付选项，后者也代表了一部分信息。我们设置和获取支付选项的方式与设置和获取用户的电子邮件的方式相同。因此，我们保持`User`类不变，因为它已经满足了单一职责原则。当我们在`User`类中添加付款功能时，这将破坏平衡。在这种情况下，`User`类将既存储用户信息又进行付款交易。这在单一职责原则方面是不可接受的，因此我们不会这样做。

单一职责原则也与函数相关。`add_payment_option()`方法有两个职责。如果函数的第二个（默认）参数为 true，则它会添加一个新的主要支付选项。否则，它会将新的支付选项添加到非主要选项列表中。最好为添加主要支付选项单独创建一个方法。这样，每个方法都将有单一职责。

# 开闭原则

开闭原则规定一个类应该对扩展开放，对修改关闭。这意味着每当你需要新的功能时，最好是扩展基本功能而不是修改它。例如，我们设计的电子商务应用程序中的`Product`类。以下是`Product`类的简单图表：

![](img/7a087189-9c02-4c59-857f-38a419d1299d.png)

每个`Product`对象都有三个属性：**名称**、**价格**和**重量**。现在，想象一下，在设计了`Product`类和整个电子商务平台之后，客户提出了一个新的需求。他们现在想购买数字产品，如电子书、电影和音频录音。一切都很好，除了产品的重量。现在可能会有两种产品类型——有形和数字——我们应该重新思考`Product`使用的逻辑。我们可以像这里的代码中所示那样在`Product`中加入一个新的功能：

```cpp
class Product
{
public:
  // code omitted for brevity
  bool is_digital() const {
    return weight_ == 0.0;
  }

  // code omitted for brevity
};
```

显然，我们修改了类——违反了开闭原则。该原则规定类应该对修改关闭。它应该对扩展开放。我们可以通过重新设计`Product`类并将其制作成所有产品的抽象基类来实现这一点。接下来，我们创建两个更多的类，它们继承`Product`基类：`PhysicalProduct`和`DigitalProduct`。下面的类图描述了新的设计：

![](img/39a14c81-c9c5-4dee-8924-e6cbccf9a257.png)

正如前面的图表所示，我们从`Product`类中删除了`weight_`属性。现在我们有了两个更多的类，`PhysicalProduct`有一个`weight_`属性，而`DigitalProduct`没有。相反，它有一个`file_path_`属性。这种方法满足了开闭原则，因为现在所有的类都可以扩展。我们使用继承来扩展类，而下面的原则与此密切相关。

# 里斯科夫替换原则

里斯科夫替换原则是关于正确继承类型的方式。简单来说，如果有一个函数接受某种类型的参数，那么同一个函数应该接受派生类型的参数。

里斯科夫替换原则是以图灵奖获得者、计算机科学博士芭芭拉·里斯科夫的名字命名的。

一旦你理解了继承和里氏替换原则，就很难忘记它。让我们继续开发`Product`类，并添加一个根据货币类型返回产品价格的新方法。我们可以将价格存储在相同的货币单位中，并提供一个将价格转换为指定货币的函数。以下是该方法的简单实现：

```cpp
enum class Currency { USD, EUR, GBP }; // the list goes further

class Product
{
public:
  // code omitted for brevity
  double convert_price(Currency c) {
    // convert to proper value
  }

  // code omitted for brevity
};
```

过了一段时间，公司决定为所有数字产品引入终身折扣。现在，每个数字产品都将享有 12%的折扣。在短时间内，我们在`DigitalProduct`类中添加了一个单独的函数，该函数通过应用折扣返回转换后的价格。以下是`DigitalProduct`中的实现：

```cpp
class DigitalProduct : public Product
{
public:
  // code omitted for brevity
  double convert_price_with_discount(Currency c) {
    // convert by applying a 12% discount
  } 
};
```

设计中的问题是显而易见的。在`DigitalProduct`实例上调用`convert_price()`将没有效果。更糟糕的是，客户端代码不应该调用它。相反，它应该调用`convert_price_with_discount()`，因为所有数字产品必须以 12%的折扣出售。设计违反了里氏替换原则。

我们不应该破坏类层次结构，而应该记住多态的美妙之处。一个更好的版本将如下所示：

```cpp
class Product
{
public:
  // code omitted for brevity
  virtual double convert_price(Currency c) {
    // default implementation
  }

  // code omitted for brevity
};

class DigitalProduct : public Product
{
public:
  // code omitted for brevity
  double convert_price(Currency c) override {
    // implementation applying a 12% discount
  }

  // code omitted for brevity
};
```

正如您所看到的，我们不再需要`convert_price_with_discount()`函数。而且里氏替换原则得到了遵守。然而，我们应该再次检查设计中的缺陷。让我们通过在基类中引入用于折扣计算的私有虚方法来改进它。以下是`Product`类的更新版本，其中包含一个名为`calculate_discount()`的私有虚成员函数：

```cpp
class Product
{
public:
  // code omitted for brevity
  virtual double convert_price(Currency c) {
    auto final_price = apply_discount();
    // convert the final_price based on the currency
  }

private:
 virtual double apply_discount() {
 return getPrice(); // no discount by default
 }

  // code omitted for brevity
};
```

`convert_price()`函数调用私有的`apply_discount()`函数，该函数返回原价。这里有一个技巧。我们在派生类中重写`apply_discount()`函数，就像下面的`DigitalProduct`实现中所示：

```cpp
class DigitalProduct : public Product
{
public:
  // code omitted for brevity

private:
  double apply_discount() override {
 return getPrice() * 0.12;
 }

  // code omitted for brevity
};
```

我们无法在类外部调用私有函数，但我们可以在派生类中重写它。前面的代码展示了重写私有虚函数的美妙之处。我们修改了实现，但接口保持不变。如果派生类不需要为折扣计算提供自定义功能，则不需要重写它。另一方面，`DigitalProduct`需要在转换之前对价格进行 12%的折扣。不需要修改基类的公共接口。

您应该考虑重新思考`Product`类的设计。直接在`getPrice()`中调用`apply_discount()`似乎更好，因此始终返回最新的有效价格。尽管在某些时候，您应该强迫自己停下来。

设计过程是有创意的，有时也是不感激的。由于意外的新需求，重写所有代码并不罕见。我们使用原则和方法来最小化在实现新功能后可能出现的破坏性变化。SOLID 的下一个原则是最佳实践之一，它将使您的设计更加灵活。

# 接口隔离原则

接口隔离原则建议将复杂的接口分成更简单的接口。这种隔离允许类避免实现它们不使用的接口。

在我们的电子商务应用中，我们应该实现产品发货、替换和过期功能。产品的发货是将产品项目移交给买家。在这一点上，我们不关心发货的细节。产品的替换考虑在向买家发货后替换损坏或丢失的产品。最后，产品的过期意味着处理在到期日期之前未销售的产品。

我们可以在前面介绍的`Product`类中实现所有功能。然而，最终我们会遇到一些产品类型，例如无法运输的产品（例如，很少有人会将房屋运送给买家）。可能有一些产品是不可替代的。例如，原始绘画即使丢失或损坏也无法替换。最后，数字产品永远不会过期。嗯，大多数情况下是这样。

我们不应该强制客户端代码实现它不需要的行为。在这里，客户端指的是实现行为的类。以下示例是违反接口隔离原则的不良实践：

```cpp
class IShippableReplaceableExpirable
{
public:
  virtual void ship() = 0;
  virtual void replace() = 0;
  virtual void expire() = 0;
};
```

现在，`Product`类实现了前面展示的接口。它必须为所有方法提供实现。接口隔离原则建议以下模型：

```cpp
class IShippable
{
public:
  virtual void ship() = 0;
};

class IReplaceable
{
public:
  virtual void replace() = 0;
};

class IExpirable
{
public:
  virtual void expire() = 0;
};
```

现在，`Product`类跳过了实现任何接口。它的派生类从特定类型派生（实现）。以下示例声明了几种产品类的类型，每种类型都支持前面介绍的有限数量的行为。请注意，为了代码简洁起见，我们省略了类的具体内容：

```cpp
class PhysicalProduct : public Product {};

// The book does not expire
class Book : public PhysicalProduct, public IShippable, public IReplaceable
{
};

// A house is not shipped, not replaced, but it can expire 
// if the landlord decided to put it on sell till a specified date
class House : public PhysicalProduct, public IExpirable
{
};

class DigitalProduct : public Product {};

// An audio book is not shippable and it cannot expire. 
// But we implement IReplaceable in case we send a wrong file to the user.
class AudioBook : public DigitalProduct, public IReplaceable
{
};
```

如果要将文件下载包装为货物，可以考虑为`AudioBook`实现`IShippable`。

# 依赖倒置原则

最后，依赖倒置原则规定对象不应该紧密耦合。它允许轻松切换到替代依赖。例如，当用户购买产品时，我们会发送购买收据。从技术上讲，有几种发送收据的方式，即打印并通过邮件发送，通过电子邮件发送，或在平台的用户账户页面上显示收据。对于后者，我们会通过电子邮件或应用程序向用户发送通知，告知收据已准备好查看。看一下以下用于打印收据的接口：

```cpp
class IReceiptSender
{
public:
  virtual void send_receipt() = 0;
};
```

假设我们已经在`Product`类中实现了`purchase()`方法，并在完成后发送了收据。以下代码部分处理了发送收据的过程：

```cpp
class Product
{
public:
  // code omitted for brevity
  void purchase(IReceiptSender* receipt_sender) {
    // purchase logic omitted
    // we send the receipt passing purchase information
 receipt_sender->send(/* purchase-information */);
  }
};
```

我们可以通过添加所需的收据打印选项来扩展应用程序。以下类实现了`IReceiptSender`接口：

```cpp
class MailReceiptSender : public IReceiptSender
{
public:
  // code omitted for brevity
  void send_receipt() override { /* ... */ }
};
```

另外两个类——`EmailReceiptSender`和`InAppReceiptSender`——都实现了`IReceiptSender`。因此，要使用特定的收据，我们只需通过`purchase()`方法将依赖注入到`Product`中，如下所示：

```cpp
IReceiptSender* rs = new EmailReceiptSender();
// consider the get_purchasable_product() is implemented somewhere in the code
auto product = get_purchasable_product();
product.purchase(rs);
```

我们可以进一步通过在`User`类中实现一个方法，返回具体用户所需的收据发送选项。这将使类之间的耦合更少。

在前面讨论的所有 SOLID 原则中，都是组合类的一种自然方式。遵循这些原则并不是强制性的，但如果遵循这些原则，将会改善你的设计。

# 使用领域驱动设计

领域是程序的主题领域。我们正在讨论和设计一个以电子商务为主题概念的电子商务平台，所有附属概念都是该领域的一部分。我们建议您在项目中考虑领域驱动设计。然而，该方法并不是程序设计的万能药。

设计项目时，考虑以下三层三层架构的三个层次是很方便的：

+   演示

+   业务逻辑

+   数据

三层架构适用于客户端-服务器软件，例如我们在本章中设计的软件。表示层向用户提供与产品、购买和货物相关的信息。它通过向客户端输出结果与其他层进行通信。这是客户直接访问的一层，例如，Web 浏览器。

业务逻辑关心应用功能。例如，用户浏览由表示层提供的产品，并决定购买其中的一个。请求的处理是业务层的任务。在领域驱动设计中，我们倾向于将领域级实体与其属性结合起来，以应对应用程序的复杂性。我们将用户视为`User`类的实例，产品视为`Product`类的实例，依此类推。用户购买产品被业务逻辑解释为`User`对象创建一个`Order`对象，而`Order`对象又与`Product`对象相关联。然后，`Order`对象与与购买产品相关的`Transaction`对象相关联。购买的相应结果通过表示层表示。

最后，数据层处理存储和检索数据。从用户认证到产品购买，每个步骤都从系统数据库（或数据库）中检索或记录。

将应用程序分成层可以处理其整体的复杂性。最好协调具有单一责任的对象。领域驱动设计区分实体和没有概念身份的对象。后者被称为值对象。例如，用户不区分每个唯一的交易；他们只关心交易所代表的信息。另一方面，用户对象以`User`类的形式具有概念身份（实体）。

使用其他对象（或不使用）对对象执行的操作称为服务。服务更像是一个不与特定对象绑定的操作。例如，通过`set_name()`方法设置用户的名称是一个不应被视为服务的操作。另一方面，用户购买产品是由服务封装的操作。

最后，领域驱动设计强烈地融合了**存储库**和**工厂**模式。存储库模式负责检索和存储领域对象的方法。工厂模式创建领域对象。使用这些模式允许我们在需要时交换替代实现。现在让我们在电子商务平台的背景下发现设计模式的力量。

# 利用设计模式

设计模式是软件设计中常见问题的架构解决方案。重要的是要注意，设计模式不是方法或算法。它们是提供组织类和它们之间关系的一种架构构造，以实现更好的代码可维护性的方式。即使以前没有使用过设计模式，你很可能已经自己发明了一个。许多问题在软件设计中往往会反复出现。例如，为现有库创建更好的接口是一种称为**facade**的设计模式形式。设计模式有名称，以便程序员在对话或文档中使用它们。与其他程序员使用 facade、factory 等进行闲聊应该是很自然的。

我们之前提到领域驱动设计融合了存储库和工厂模式。现在让我们来了解它们是什么，以及它们如何在我们的设计努力中发挥作用。

# 存储库模式

正如 Martin Fowler 最好地描述的那样，存储库模式“在领域和数据映射层之间使用类似集合的接口来访问领域对象”。

该模式提供了直接的数据操作方法，无需直接使用数据库驱动程序。添加、更新、删除或选择数据自然地适用于应用程序域。

其中一种方法是创建一个提供必要功能的通用存储库类。简单的接口如下所示：

```cpp
class Entity; // new base class

template <typename T, typename = std::enable_if_t<std::is_base_of_v<Entity, T>>>
class Repository
{
public:
 T get_by_id(int);
 void insert(const T&);
 void update(const T&);
 void remove(const T&);
 std::vector<T> get_all(std::function<bool(T)> condition);
};
```

我们在前面引入了一个名为`Entity`的新类。`Repository`类与实体一起工作，并确保每个实体都符合`Entity`的相同接口，它应用`std::enable_if`以及`std::is_base_of_v`到模板参数。

`std::is_base_of_v`是`std::is_base_of<>::value`的简写。此外，`std::enable_if_t`替换了`std::enable_if<>::type`。

`Entity`类的表示如下：

```cpp
class Entity
{
public:
  int get_id() const;
  void set_id(int);
private:
  int id_;
};
```

每个业务对象都是一个`Entity`，因此，前面讨论的类应该更新为从`Entity`继承。例如，`User`类的形式如下：

```cpp
class User : public Entity
{
// code omitted for brevity
};
```

因此，我们可以这样使用存储库：

```cpp
Repository<User> user_repo;
User fetched_user = user_repo.get_by_id(111);
```

前面介绍的存储库模式是对该主题的简单介绍，但是你可以使它更加强大。它类似于外观模式。虽然使用外观模式的重点不是访问数据库，但是最好用数据库访问来解释。外观模式包装了一个复杂的类或类，为客户端提供了一个简单的预定义接口，以便使用底层功能。

# 工厂模式

当程序员谈论工厂模式时，他们可能会混淆工厂方法和抽象工厂。这两者都是提供各种对象创建机制的创建模式。让我们讨论工厂方法。它提供了一个在基类中创建对象的接口，并允许派生类修改将被创建的对象。

现在是处理物流的时候了，工厂方法将在这方面帮助我们。当你开发一个提供产品发货的电子商务平台时，你应该考虑到并非所有用户都住在你的仓库所在的同一地区。因此，从仓库向买家发货时，你应该选择适当的运输类型。自行车、无人机、卡车等等。感兴趣的问题是设计一个灵活的物流管理系统。

不同的交通工具需要不同的实现。然而，它们都符合一个接口。以下是`Transport`接口及其派生的具体交通工具实现的类图：

![](img/9440ba69-a862-495f-b00d-4e2a4db2e746.png)

前面图表中的每个具体类都提供了特定的交付实现。

假设我们设计了以下`Logistics`基类，负责与物流相关的操作，包括选择适当的运输方式，如下所示：

![](img/69a43738-0ce0-4c88-8f17-b5a8d36fec0f.png)

前面应用的工厂方法允许灵活地添加新的运输类型以及新的物流方法。注意`createTransport()`方法返回一个`Transport`指针。派生类覆盖该方法，每个派生类返回`Transport`的子类，从而提供了特定的运输方式。这是可能的，因为子类返回了派生类型，否则在覆盖基类方法时无法返回不同的类型。

`Logistics`中的`createTransport()`如下所示：

```cpp
class Logistics 
{
public:
 Transport* getLogistics() = 0;
  // other functions are omitted for brevity
};
```

`Transport`类代表了`Drone`、`Truck`和`Ship`的基类。这意味着我们可以创建每个实例，并使用`Transport`指针引用它们，如下所示：

```cpp
Transport* ship_transport = new Ship();
```

这是工厂模式的基础，因为例如`RoadLogistics`覆盖了`getLogistics()`，如下所示：

```cpp
class RoadLogistics : public Logistics
{
public: 
  Truck* getLogistics() override {
 return new Truck();
 }
}
```

注意函数的返回类型，它是`Truck`而不是`Transport`。这是因为`Truck`继承自`Transport`。另外，看看对象的创建是如何与对象本身解耦的。创建新对象是通过工厂完成的，这与之前讨论的 SOLID 原则保持一致。

乍一看，利用设计模式似乎会给设计增加额外的复杂性。然而，当实践设计模式时，你应该培养对更好设计的真正感觉，因为它们允许项目整体具有灵活性和可扩展性。

# 总结

软件开发需要细致的规划和设计。我们在本章中学到，项目开发包括以下关键步骤：

+   需求收集和分析：这包括理解项目的领域，讨论和最终确定应该实现的功能。

+   规范创建：这包括记录需求和项目功能。

+   设计和测试规划：这指的是从更大的实体开始设计项目，然后将每个实体分解为一个单独的类，考虑到项目中的其他类。这一步还涉及规划项目的测试方式。

+   编码：这一步涉及编写代码，实现前面步骤中指定的项目。

+   测试和稳定性：这意味着根据预先计划的用例和场景检查项目，以发现问题并加以修复。

+   发布和维护：这是最后一步，将我们带到项目的发布和进一步的维护。

项目设计对程序员来说是一个复杂的任务。他们应该提前考虑，因为部分功能是在开发过程中引入的。

为了使设计灵活而健壮，我们已经讨论了导致更好架构的原则和模式。我们已经学习了设计软件项目及其复杂性的过程。

避免糟糕的设计决策的最佳方法之一是遵循已经设计好的模式和实践。在未来的项目中，你应该考虑使用 SOLID 原则以及经过验证的设计模式。

在下一章中，我们将设计一个策略游戏。我们将熟悉更多的设计模式，并看到它们在游戏开发中的应用。

# 问题

1.  TDD 的好处是什么？

1.  UML 中交互图的目的是什么？

1.  组合和聚合之间有什么区别？

1.  你会如何描述 Liskov 替换原则？

1.  假设你有一个`Animal`类和一个`Monkey`类。后者描述了一种特定的会在树上跳跃的动物。从`Animal`类继承`Monkey`类是否违反了开闭原则？

1.  在本章讨论的`Product`类及其子类上应用工厂方法。

# 进一步阅读

有关更多信息，请参阅：

+   *《面向对象的分析与设计与应用》* by Grady Booch，[`www.amazon.com/Object-Oriented-Analysis-Design-Applications-3rd/dp/020189551X/`](https://www.amazon.com/Object-Oriented-Analysis-Design-Applications-3rd/dp/020189551X/)

+   *《设计模式：可复用面向对象软件的元素》* by Erich Gamma 等人，[`www.amazon.com/Design-Patterns-Elements-Reusable-Object-Oriented/dp/0201633612/`](https://www.amazon.com/Design-Patterns-Elements-Reusable-Object-Oriented/dp/0201633612/)

+   *《代码大全：软件构建的实用手册》* by Steve McConnel，[`www.amazon.com/Code-Complete-Practical-Handbook-Construction/dp/0735619670/`](https://www.amazon.com/Code-Complete-Practical-Handbook-Construction/dp/0735619670/)

+   *《领域驱动设计：软件核心复杂性的应对》* by Eric Evans，[`www.amazon.com/Domain-Driven-Design-Tackling-Complexity-Software/dp/0321125215/`](https://www.amazon.com/Domain-Driven-Design-Tackling-Complexity-Software/dp/0321125215/)
