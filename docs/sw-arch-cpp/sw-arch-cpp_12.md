# 第九章：持续集成和持续部署

在之前的一章中，我们学习了关于不同构建系统和不同打包系统的知识，我们的应用程序可以使用。持续集成（CI）和持续部署（CD）允许我们利用构建和打包的知识来提高服务质量和我们正在开发的应用程序的健壮性。

CI 和 CD 都依赖于良好的测试覆盖率。CI 主要使用单元测试和集成测试，而 CD 更依赖于冒烟测试和端到端测试。您在《第八章》《编写可测试的代码》中了解了测试的不同方面。有了这些知识，您就可以构建 CI/CD 流水线了。

在本章中，我们将涵盖以下主题：

+   理解 CI

+   审查代码更改

+   探索测试驱动的自动化

+   将部署管理为代码

+   构建部署代码

+   构建 CD 流水线

+   使用不可变基础设施

# 技术要求

本章的示例代码可以在[`github.com/PacktPublishing/Software-Architecture-with-Cpp/tree/master/Chapter09`](https://github.com/PacktPublishing/Software-Architecture-with-Cpp/tree/master/Chapter09)找到。

要理解本章中解释的概念，您需要进行以下安装：

+   免费的 GitLab 账户

+   Ansible 版本 2.8+

+   Terraform 版本 0.12+

+   Packer 版本 1.4+

# 理解 CI

CI 是缩短集成周期的过程。在传统软件中，许多不同的功能可能是分开开发的，只有在发布之前才进行集成，而在 CI 项目中，集成可以每天发生多次。通常，开发人员进行的每个更改都会在提交到中央代码库时进行测试和集成。

由于测试发生在开发之后，反馈循环要快得多。这使得开发人员更容易修复错误（因为他们通常还记得做了什么改动）。与传统的在发布之前进行测试的方法相比，CI 节省了大量工作，并提高了软件的质量。

## 尽早发布，经常发布

您是否听说过“尽早发布，经常发布”的说法？这是一种强调短周期发布的软件开发理念。而短周期的发布循环则在规划、开发和验证之间提供了更短的反馈循环。当出现问题时，应该尽早出现，以便修复问题的成本相对较小。

这一理念是由埃里克·S·雷蒙德（也被称为 ESR）在他 1997 年的文章《大教堂与集市》中推广的。还有一本同名的书，其中包含了作者的这篇文章和其他文章。考虑到 ESR 在开源运动中的活动，"尽早发布，经常发布"的口号成为了开源项目运作方式的代名词。

几年后，同样的原则不仅仅适用于开源项目。随着对敏捷方法学（如 Scrum）日益增长的兴趣，“尽早发布，经常发布”的口号成为了以产品增量结束的开发冲刺的代名词。当然，这个增量是软件发布，但通常在冲刺期间会有许多其他发布。

如何实现这样的短周期发布循环？一个答案是尽可能依赖自动化。理想情况下，代码库的每次提交都应该以发布结束。这个发布是否面向客户是另一回事。重要的是，每次代码变更都可能导致可用的产品。

当然，为每个提交构建和发布到公共环境对于任何开发人员来说都是一项繁琐的工作。即使一切都是脚本化的，这也会给通常的琐事增加不必要的开销。这就是为什么您希望设置一个 CI 系统来自动化您和您的开发团队的发布。

## CI 的优点

CI 是将几个开发人员的工作至少每天集成在一起的概念。正如已经讨论过的，有时它可能意味着每天几次。进入存储库的每个提交都是单独集成和验证的。构建系统检查代码是否可以无错误地构建。打包系统可以创建一个准备保存为工件的软件包，甚至在使用 CD 时稍后部署。最后，自动化测试检查是否与更改相关的已知回归没有发生。现在让我们详细看看它的优点：

+   CI 允许快速解决问题。如果其中一个开发人员在行末忘记了一个分号，CI 系统上的编译器将立即捕捉到这个错误，这样错误的代码就不会传播给其他开发人员，从而阻碍他们的工作。当然，开发人员在提交代码之前应该构建更改并对其进行测试，但是在开发人员的机器上可能会忽略一些小错误，并且这些错误可能会进入共享存储库。

+   使用 CI 的另一个好处是，它可以防止常见的“在我的机器上可以运行”的借口。如果开发人员忘记提交必要的文件，CI 系统将无法构建更改，再次阻止它们进一步传播并对整个团队造成麻烦。一个开发人员环境的特殊配置也不再是问题。如果一个更改在两台机器上构建，即开发人员的计算机和 CI 系统，我们可以安全地假设它也应该在其他机器上构建。

## 门控机制

如果我们希望 CI 能够为我们带来价值，而不仅仅是为我们构建软件包，我们需要一个门控机制。这个门控机制将允许我们区分好的代码更改和坏的代码更改，从而使我们的应用程序免受使其无用的修改。为了实现这一点，我们需要一个全面的测试套件。这样的套件使我们能够自动识别何时更改有问题，并且我们能够迅速做到这一点。

对于单个组件，单元测试起到了门控机制的作用。CI 系统可以丢弃任何未通过单元测试的更改，或者任何未达到一定代码覆盖率阈值的更改。在构建单个组件时，CI 系统还可以使用集成测试来进一步确保更改是稳定的，不仅仅是它们自己，而且它们在一起的表现也是正常的。

## 使用 GitLab 实施流水线

在本章中，我们将使用流行的开源工具构建一个完整的 CI/CD 流水线，其中包括门控机制、自动部署，并展示基础设施自动化的概念。

第一个这样的工具是 GitLab。您可能听说过它作为一个 Git 托管解决方案，但实际上，它远不止于此。GitLab 有几个版本，即以下版本：

+   一种开源解决方案，您可以在自己的设施上托管

+   提供额外功能的自托管付费版本，超过开源社区版

+   最后，一个**软件即服务**（**SaaS**）托管在[`gitlab.com`](https://gitlab.com)下的托管服务

对于本书的要求，每个版本都具备所有必要的功能。因此，我们将专注于 SaaS 版本，因为这需要最少的准备工作。

尽管[`gitlab.com`](https://gitlab.com)主要针对开源项目，但如果您不想与整个世界分享您的工作，您也可以创建私有项目和存储库。这使我们能够在 GitLab 中创建一个新的私有项目，并用我们已经在第七章中演示的代码填充它，*构建和打包*。

许多现代 CI/CD 工具可以代替 GitLab CI/CD。例如 GitHub Actions、Travis CI、CircleCI 和 Jenkins。我们选择了 GitLab，因为它既可以作为 SaaS 形式使用，也可以在自己的设施上使用，因此应该适应许多不同的用例。

然后，我们将使用之前的构建系统在 GitLab 中创建一个简单的 CI 流水线。这些流水线在 YAML 文件中被描述为一系列步骤和元数据。一个构建所有要求的示例流水线，以及来自第七章的示例项目，*构建和打包*，将如下所示：

```cpp
# We want to cache the conan data and CMake build directory
cache:
  key: all
  paths:
    - .conan
    - build

# We're using conanio/gcc10 as the base image for all the subsequent commands
default:
  image: conanio/gcc10

stages:
  - prerequisites
  - build

before_script:
  - export CONAN_USER_HOME="$CI_PROJECT_DIR"

# Configure conan
prerequisites:
  stage: prerequisites
  script:
    - pip install conan==1.34.1
    - conan profile new default || true
    - conan profile update settings.compiler=gcc default
    - conan profile update settings.compiler.libcxx=libstdc++11 default
    - conan profile update settings.compiler.version=10 default
    - conan profile update settings.arch=x86_64 default
    - conan profile update settings.build_type=Release default
    - conan profile update settings.os=Linux default
    - conan remote add trompeloeil https://api.bintray.com/conan/trompeloeil/trompeloeil || true

# Build the project
build:
  stage: build
  script:
    - sudo apt-get update && sudo apt-get install -y docker.io
    - mkdir -p build
    - cd build
    - conan install ../ch08 --build=missing
    - cmake -DBUILD_TESTING=1 -DCMAKE_BUILD_TYPE=Release ../ch08/customer
    - cmake --build .
```

将上述文件保存为`.gitlab-ci.yml`，放在 Git 存储库的根目录中，将自动在 GitLab 中启用 CI，并在每次提交时运行流水线。

# 审查代码更改

代码审查可以在有 CI 系统和没有 CI 系统的情况下使用。它们的主要目的是对引入代码的每个更改进行双重检查，以确保其正确性，符合应用程序的架构，并遵循项目的指南和最佳实践。

当没有 CI 系统时，通常是审阅者的任务手动测试更改并验证其是否按预期工作。CI 减轻了这一负担，让软件开发人员专注于代码的逻辑结构。

## 自动化的门控机制

自动化测试只是门控机制的一个例子。当它们的质量足够高时，它们可以保证代码按照设计工作。但正确工作的代码和好的代码之间仍然存在差异。从本书到目前为止，您已经了解到，如果代码满足了几个价值观，那么它可以被认为是好的。功能上的正确性只是其中之一。

还有其他工具可以帮助实现代码基准的期望标准。其中一些在前几章中已经涵盖，所以我们不会详细介绍。请记住，在 CI/CD 流水线中使用代码检查器、代码格式化程序和静态分析是一个很好的做法。虽然静态分析可以作为一个门控机制，但你可以将代码检查和格式化应用到进入中央存储库的每个提交，以使其与代码库的其余部分保持一致。附录中会有更多关于代码检查器和格式化程序的内容。

理想情况下，这个机制只需要检查代码是否已经被格式化，因为在将代码推送到存储库之前，开发人员应该完成格式化步骤。当使用 Git 作为版本控制系统时，Git Hooks 机制可以防止在没有运行必要工具的情况下提交代码。

但自动化分析只能帮你解决一部分问题。你可以检查代码是否功能完整，是否没有已知的错误和漏洞，并且是否符合编码标准。这就是手动检查的作用。

## 代码审查-手动门控机制

对代码更改的手动检查通常被称为代码审查。代码审查的目的是识别问题，包括特定子系统的实现以及对应用程序整体架构的遵循。自动化性能测试可能会或可能不会发现给定功能的潜在问题。另一方面，人眼通常可以发现问题的次优解决方案。无论是错误的数据结构还是计算复杂度过高的算法，一个好的架构师应该能够找出问题所在。

但执行代码审查并不仅仅是架构师的角色。同行审查，也就是由作者的同行进行的代码审查，在开发过程中也有其作用。这样的审查之所以有价值，不仅因为它们允许同事发现彼此代码中的错误。更重要的方面是许多队友突然意识到其他人正在做什么。这样，当团队中有人缺席（无论是因为长时间会议、度假还是工作轮换），另一名团队成员可以替补缺席者。即使他们不是该主题的专家，每个成员至少知道有趣的代码位于何处，每个人都应该能够记住代码的最后更改。这意味着它们发生的时间、范围和内容。

随着更多人意识到应用程序内部的情况，他们更有可能发现一个组件最近的变化和一个新发现的错误之间的关联。即使团队中的每个人可能有不同的经验，但当每个人都非常了解代码时，他们可以共享资源。

因此，代码审查可以检查更改是否符合所需的架构，以及其实现是否正确。我们称这样的代码审查为架构审查或专家审查。

另一种类型的代码审查，同行审查，不仅有助于发现错误，还提高了团队对其他成员正在做什么的意识。如果需要，您还可以在处理与外部服务集成的更改时执行不同类型的专家审查。

由于每个接口都是潜在问题的源头，接近接口级别的更改应被视为特别危险。我们建议您将通常的同行审查与来自接口另一侧的专家的审查相结合。例如，如果您正在编写生产者的代码，请向消费者请求审查。这样，您可以确保不会错过一些您可能认为非常不太可能的重要用例，但另一方却经常使用。

## 代码审查的不同方法

您通常会进行异步代码审查。这意味着正在审查的更改的作者和审阅者之间的通信不是实时发生的。相反，每个参与者都可以在任何时间发表他们的评论和建议。一旦没有更多的评论，作者会重新修改原始更改，然后再次进行审查。这可能需要多轮，直到每个人都同意不需要进一步的更正为止。

当一个更改特别有争议并且异步代码审查需要太长时间时，进行同步代码审查是有益的。这意味着举行一次会议（面对面或远程），解决对未来方向的任何相反意见。这将在特定情况下发生，当一个更改与最初的决定之一相矛盾，因为在实施更改时获得了新的知识。

有一些专门针对代码审查的工具。更常见的是，您会希望使用内置到存储库服务器中的工具，其中包括以下服务：

+   GitHub

+   Bitbucket

+   GitLab

+   Gerrit

所有这些都提供 Git 托管和代码审查。其中一些甚至提供整个 CI/CD 流水线、问题管理、wiki 等等。

当您使用代码托管和代码审查的综合包时，默认工作流程是将更改推送为单独的分支，然后要求项目所有者合并更改，这个过程称为拉取请求（或合并请求）。尽管名字很花哨，但拉取请求或合并请求通知项目所有者，您有代码希望与主分支合并。这意味着审阅者应该审查您的更改，以确保一切都井井有条。

## 使用拉取请求（合并请求）进行代码审查

使用 GitLab 等系统创建拉取请求或合并请求非常容易。首先，当我们从命令行推送新分支到中央存储库时，我们可以观察到以下消息：

```cpp
remote:
remote: To create a merge request for fix-ci-cd, visit:
remote:   https://gitlab.com/hosacpp/continuous-integration/merge_requests/new?merge_request%5Bsource_branch%5D=fix-ci-cd
remote:                         
```

如果您之前已启用 CI（通过添加`.gitlab-ci.yml`文件），您还会看到新推送的分支已经经过了 CI 流程。这甚至发生在您打开合并请求之前，这意味着您可以在从 CI 获得每个自动检查都通过的信息之前推迟通知同事。

打开合并请求的两种主要方式如下：

+   通过按照推送消息中提到的链接

+   通过在 GitLab UI 中导航到合并请求并选择“创建合并请求”按钮或“新合并请求”按钮

当您提交合并请求并填写完所有相关字段时，您会看到 CI 流水线的状态也是可见的。如果流水线失败，将无法合并更改。

# 探索测试驱动的自动化

CI 主要侧重于集成部分。这意味着构建不同子系统的代码并确保它们可以一起工作。虽然测试不是严格要求实现此目的，但在没有测试的情况下运行 CI 似乎是一种浪费。没有自动化测试的 CI 使得更容易向代码引入微妙的错误，同时给人一种虚假的安全感。

这就是为什么 CI 经常与持续测试紧密结合的原因之一，我们将在下一节中介绍。

## 行为驱动开发

到目前为止，我们已经设立了一个可以称之为持续构建的流水线。我们对代码所做的每一次更改最终都会被编译，但我们不会进一步测试它。现在是时候引入持续测试的实践了。在低级别进行测试也将作为一个门控机制，自动拒绝所有不满足要求的更改。

您如何检查给定的更改是否满足要求？最好的方法是根据这些要求编写测试。其中一种方法是遵循**行为驱动开发**（**BDD**）。BDD 的概念是鼓励敏捷项目中不同参与者之间更深入的协作。

与传统方法不同，传统方法要么由开发人员编写测试，要么由 QA 团队编写测试，而 BDD 中，测试是由以下个人共同创建的：

+   开发人员

+   QA 工程师

+   业务代表。

指定 BDD 测试的最常见方式是使用 Cucumber 框架，该框架使用简单的英语短语来描述系统的任何部分的期望行为。这些句子遵循特定的模式，然后可以转换为可工作的代码，与所选的测试框架集成。

Cucumber 框架中有对 C++的官方支持，它基于 CMake、Boost、GTest 和 GMock。在以 cucumber 格式指定所需行为（使用称为 Gherkin 的领域特定语言）之后，我们还需要提供所谓的步骤定义。步骤定义是与 cucumber 规范中描述的操作相对应的实际代码。例如，考虑以下以 Gherkin 表达的行为：

```cpp
# language: en
Feature: Summing
In order to see how much we earn,
Sum must be able to add two numbers together

Scenario: Regular numbers
  Given I have entered 3 and 2 as parameters
  When I add them
  Then the result should be 5
```

我们可以将其保存为`sum.feature`文件。为了生成带有测试的有效 C++代码，我们将使用适当的步骤定义：

```cpp
#include <gtest/gtest.h>
#include <cucumber-cpp/autodetect.hpp>

#include <Sum.h>

using cucumber::ScenarioScope;

struct SumCtx {
  Sum sum;
  int a;
  int b;
  int result;
};

GIVEN("^I have entered (\\d+) and (\\d+) as parameters$", (const int a, const int b)) {
    ScenarioScope<SumCtx> context;

    context->a = a;
    context->b = b;
}

WHEN("^I add them") {
    ScenarioScope<SumCtx> context;

    context->result = context->sum.sum(context->a, context->b);
}

THEN("^the result should be (.*)$", (const int expected)) {
    ScenarioScope<SumCtx> context;

    EXPECT_EQ(expected, context->result);
}
```

在从头开始构建应用程序时，遵循 BDD 模式是一个好主意。本书旨在展示您可以在这样的绿地项目中使用的最佳实践。但这并不意味着您不能在现有项目中尝试我们的示例。在项目的生命周期中的任何时间都可以添加 CI 和 CD。由于尽可能经常运行测试总是一个好主意，因此几乎总是一个好主意仅出于持续测试目的使用 CI 系统。

如果你没有行为测试，你不需要担心。你可以稍后添加它们，目前只需专注于你已经有的那些测试。无论是单元测试还是端到端测试，任何有助于评估你的应用程序状态的东西都是一个很好的门控机制的候选者。

## 为 CI 编写测试

对于 CI 来说，最好专注于单元测试和集成测试。它们在可能的最低级别上工作，这意味着它们通常执行速度快，要求最小。理想情况下，所有单元测试应该是自包含的（没有像工作数据库这样的外部依赖）并且能够并行运行。这样，当问题出现在单元测试能够捕捉到的级别时，有问题的代码将在几秒钟内被标记出来。

有些人说单元测试只在解释性语言或动态类型语言中才有意义。论点是 C++已经通过类型系统和编译器检查内置了测试。虽然类型检查可以捕捉一些在动态类型语言中需要单独测试的错误，但这不应该成为不编写单元测试的借口。毕竟，单元测试的目的不是验证代码能够无问题地执行。我们编写单元测试是为了确保我们的代码不仅执行，而且还满足我们所有的业务需求。

作为一个极端的例子，看一下以下两个函数。它们都在语法上是正确的，并且使用了适当的类型。然而，仅仅通过看它们，你可能就能猜出哪一个是正确的，哪一个是错误的。单元测试有助于捕捉这种行为不当：

```cpp
int sum (int a, int b) {
 return a+b;
}
```

前面的函数返回提供的两个参数的总和。下一个函数只返回第一个参数的值：

```cpp
int sum (int a, int b) {
  return a;
}
```

即使类型匹配，编译器不会抱怨，这段代码也不能执行其任务。为了区分有用的代码和错误的代码，我们使用测试和断言。

## 持续测试

已经建立了一个简单的 CI 流水线，非常容易通过测试来扩展它。由于我们已经在构建和测试过程中使用 CMake 和 CTest，我们所需要做的就是在我们的流水线中添加另一个步骤来执行测试。这一步可能看起来像这样：

```cpp
# Run the unit tests with ctest
test:
  stage: test
  script:
    - cd build
    - ctest .
```

因此，整个流水线将如下所示：

```cpp
cache:
  key: all
  paths:
    - .conan
    - build

default:
  image: conanio/gcc9

stages:
  - prerequisites
  - build
 - test # We add another stage that tuns the tests

before_script:
  - export CONAN_USER_HOME="$CI_PROJECT_DIR"

prerequisites:
  stage: prerequisites
  script:
    - pip install conan==1.34.1
    - conan profile new default || true
    - conan profile update settings.compiler=gcc default
    - conan profile update settings.compiler.libcxx=libstdc++11 default
    - conan profile update settings.compiler.version=10 default
    - conan profile update settings.arch=x86_64 default
    - conan profile update settings.build_type=Release default
    - conan profile update settings.os=Linux default
    - conan remote add trompeloeil https://api.bintray.com/conan/trompeloeil/trompeloeil || true

build:
  stage: build
  script:
    - sudo apt-get update && sudo apt-get install -y docker.io
    - mkdir -p build
    - cd build
    - conan install ../ch08 --build=missing
    - cmake -DBUILD_TESTING=1 -DCMAKE_BUILD_TYPE=Release ../ch08/customer
    - cmake --build .

# Run the unit tests with ctest
test:
 stage: test
 script:
 - cd build
 - ctest .
```

这样，每个提交不仅会经历构建过程，还会经历测试。如果其中一个步骤失败，我们将收到通知，知道是哪一个步骤导致了失败，并且可以在仪表板上看到哪些步骤成功了。

# 管理部署作为代码

经过测试和批准的更改，现在是将它们部署到一个操作环境的时候了。

有许多工具可以帮助部署。我们决定提供 Ansible 的示例，因为这不需要在目标机器上进行任何设置，除了一个功能齐全的 Python 安装（大多数 UNIX 系统已经有了）。为什么选择 Ansible？它在配置管理领域非常流行，并且由一个值得信赖的开源公司（红帽）支持。

## 使用 Ansible

为什么不使用已经可用的东西，比如 Bourne shell 脚本或 PowerShell？对于简单的部署，shell 脚本可能是一个更好的方法。但是随着我们的部署过程变得更加复杂，使用 shell 的条件语句来处理每种可能的初始状态就变得更加困难。

处理初始状态之间的差异实际上是 Ansible 特别擅长的。与使用命令式形式（移动这个文件，编辑那个文件，运行特定命令）的传统 shell 脚本不同，Ansible playbook（它们被称为）使用声明式形式（确保文件在这个路径上可用，确保文件包含指定的行，确保程序正在运行，确保程序成功完成）。

这种声明性的方法也有助于实现幂等性。幂等性是函数的一个特性，意味着多次应用该函数将产生与单次应用完全相同的结果。如果 Ansible playbook 的第一次运行引入了对配置的一些更改，每次后续运行都将从所需状态开始。这可以防止 Ansible 执行任何额外的更改。

换句话说，当您调用 Ansible 时，它将首先评估您希望配置的所有机器的当前状态：

+   如果其中任何一个需要进行任何更改，Ansible 将只运行所需的任务以实现所需的状态。

+   如果没有必要修改特定的内容，Ansible 将不会触及它。只有当所需状态和实际状态不同时，您才会看到 Ansible 采取行动将实际状态收敛到 playbook 内容描述的所需状态。

## Ansible 如何与 CI/CD 流水线配合

Ansible 的幂等性使其成为 CI/CD 流水线中的一个很好的目标。毕竟，即使两次运行之间没有任何更改，多次运行相同的 Ansible playbook 也没有风险。如果您将 Ansible 用于部署代码，创建 CD 只是准备适当的验收测试（例如冒烟测试或端到端测试）的问题。

声明性方法可能需要改变您对部署的看法，但收益是非常值得的。除了运行 playbooks，您还可以使用 Ansible 在远程机器上执行一次性命令，但我们不会涵盖这种用例，因为它实际上对部署没有帮助。

您可以使用 Ansible 的`shell`模块执行与 shell 相同的操作。这是因为在 playbooks 中，您编写指定使用哪些模块及其各自参数的任务。其中一个模块就是前面提到的`shell`模块，它只是在远程机器上执行提供的参数。但是，使 Ansible 不仅方便而且跨平台（至少在涉及不同的 UNIX 发行版时）的是可以操作常见概念的模块的可用性，例如用户管理、软件包管理和类似实例。

## 使用组件创建部署代码

除了标准库中提供的常规模块外，还有第三方组件允许代码重用。您可以单独测试这些组件，这也使您的部署代码更加健壮。这些组件称为角色。它们包含一组任务，使机器适合承担特定角色，例如`webserver`、`db`或`docker`。虽然一些角色准备机器提供特定服务，其他角色可能更抽象，例如流行的`ansible-hardening`角色。这是由 OpenStack 团队创建的，它使使用该角色保护的机器更难被入侵。

当您开始理解 Ansible 使用的语言时，所有的 playbooks 都不再只是脚本。反过来，它们将成为部署过程的文档。您可以通过运行 Ansible 直接使用它们，或者您可以阅读描述的任务并手动执行所有操作，例如在离线机器上。

使用 Ansible 进行团队部署的一个风险是，一旦开始使用，您必须确保团队中的每个人都能够使用它并修改相关的任务。DevOps 是整个团队必须遵循的一种实践；它不能只部分实施。当应用程序的代码发生相当大的变化，需要在部署方面进行适当的更改时，负责应用程序更改的人也应提供部署代码的更改。当然，这是您的测试可以验证的内容，因此门控机制可以拒绝不完整的更改。

Ansible 的一个值得注意的方面是它可以在推送和拉取模型中运行：

+   推送模型是当您在自己的机器上或在 CI 系统中运行 Ansible 时。然后，Ansible 连接到目标机器，例如通过 SSH 连接，并在目标机器上执行必要的步骤。

+   在拉模型中，整个过程由目标机器发起。Ansible 的组件`ansible-pull`直接在目标机器上运行，并检查代码存储库以确定特定分支是否有任何更新。刷新本地 playbook 后，Ansible 像往常一样执行所有步骤。这一次，控制组件和实际执行都发生在同一台机器上。大多数情况下，您会希望定期运行`ansible-pull`，例如，从 cron 作业中运行。

# 构建部署代码

在其最简单的形式中，使用 Ansible 进行部署可能包括将单个二进制文件复制到目标机器，然后运行该二进制文件。我们可以使用以下 Ansible 代码来实现这一点：

```cpp
tasks:
  # Each Ansible task is written as a YAML object
  # This uses a copy module
  - name: Copy the binaries to the target machine
    copy:
      src: our_application
      dest: /opt/app/bin/our_application
  # This tasks invokes the shell module. The text after the `shell:` key
  # will run in a shell on target machine
  - name: start our application in detached mode
    shell: cd /opt/app/bin; nohup ./our_application </dev/null >/dev/null 2>&1 &
```

每个任务都以连字符开头。对于每个任务，您需要指定它使用的模块（例如`copy`模块或`shell`模块），以及它的参数（如果适用）。任务还可以有一个`name`参数，这样可以更容易地单独引用任务。

# 构建 CD 管道

我们已经达到了可以安全地使用本章学到的工具构建 CD 管道的地步。我们已经知道 CI 是如何运作的，以及它如何帮助拒绝不适合发布的更改。测试自动化部分介绍了使拒绝过程更加健壮的不同方法。拥有冒烟测试或端到端测试使我们能够超越 CI，并检查整个部署的服务是否满足要求。并且有了部署代码，我们不仅可以自动化部署过程，还可以在我们的测试开始失败时准备回滚。

## 持续部署和持续交付

出于有趣的巧合，CD 的缩写可以有两种不同的含义。持续交付和持续部署的概念非常相似，但它们有一些细微的差异。在整本书中，我们专注于持续部署的概念。这是一个自动化的过程，当一个人将更改推送到中央存储库时开始，并在更改成功部署到生产环境并通过所有测试时结束。因此，我们可以说这是一个端到端的过程，因为开发人员的工作可以在没有手动干预的情况下一直传递到客户那里（当然，要经过代码审查）。您可能听说过 GitOps 这个术语来描述这种方法。由于所有操作都是自动化的，将更改推送到 Git 中的指定分支会触发部署脚本。

持续交付并不会走得那么远。与 CD 一样，它具有能够发布最终产品并对其进行测试的管道，但最终产品永远不会自动交付给客户。它可以首先交付给 QA 或用于内部业务。理想情况下，交付的构件准备好在内部客户接受后立即部署到生产环境中。

## 构建一个示例 CD 管道

让我们再次将所有这些技能结合起来，以 GitLab CI 作为示例来构建我们的管道。在测试步骤之后，我们将添加另外两个步骤，一个用于创建包，另一个用于使用 Ansible 部署此包。

我们打包步骤所需的全部内容如下：

```cpp
# Package the application and publish the artifact
package:
  stage: package
  # Use cpack for packaging
  script:
    - cd build
    - cpack .
  # Save the deb package artifact
  artifacts:
    paths:
      - build/Customer*.deb
```

当我们添加包含构件定义的包步骤时，我们将能够从仪表板下载它们。

有了这个，我们可以将 Ansible 作为部署步骤的一部分来调用：

```cpp
# Deploy using Ansible
deploy:
  stage: deploy
  script:
    - cd build
    - ansible-playbook -i localhost, ansible.yml
```

最终的管道将如下所示：

```cpp
cache:
  key: all
  paths:
    - .conan
    - build

default:
  image: conanio/gcc9

stages:
  - prerequisites
  - build
  - test
 - package
 - deploy

before_script:
  - export CONAN_USER_HOME="$CI_PROJECT_DIR"

prerequisites:
  stage: prerequisites
  script:
    - pip install conan==1.34.1
    - conan profile new default || true
    - conan profile update settings.compiler=gcc default
    - conan profile update settings.compiler.libcxx=libstdc++11 default
    - conan profile update settings.compiler.version=10 default
    - conan profile update settings.arch=x86_64 default
    - conan profile update settings.build_type=Release default
    - conan profile update settings.os=Linux default
    - conan remote add trompeloeil https://api.bintray.com/conan/trompeloeil/trompeloeil || true

build:
  stage: build
  script:
    - sudo apt-get update && sudo apt-get install -y docker.io
    - mkdir -p build
    - cd build
    - conan install ../ch08 --build=missing
    - cmake -DBUILD_TESTING=1 -DCMAKE_BUILD_TYPE=Release ../ch08/customer
    - cmake --build .

test:
  stage: test
  script:
    - cd build
    - ctest .

# Package the application and publish the artifact
package:
 stage: package
 # Use cpack for packaging
 script:
 - cd build
 - cpack .
 # Save the deb package artifact
 artifacts:
 paths:
 - build/Customer*.deb

# Deploy using Ansible
deploy:
 stage: deploy
 script:
 - cd build
 - ansible-playbook -i localhost, ansible.yml
```

要查看整个示例，请转到原始来源的*技术要求*部分的存储库。

# 使用不可变基础设施

如果您对 CI/CD 流水线足够自信，您可以再走一步。您可以部署*系统*的构件，而不是应用程序的构件。有什么区别？我们将在以下部分了解到。

## 什么是不可变基础设施？

以前，我们关注的是如何使应用程序的代码可以部署到目标基础设施上。CI 系统创建软件包（如容器），然后 CD 流程部署这些软件包。每次流水线运行时，基础设施保持不变，但软件不同。

关键是，如果您使用云计算，您可以将基础设施视为任何其他构件。例如，您可以部署整个**虚拟机**（**VM**），作为 AWS EC2 实例的构件，而不是部署容器。您可以预先构建这样的 VM 镜像作为 CI 流程的另一个构件。这样，版本化的 VM 镜像以及部署它们所需的代码成为您的构件，而不是容器本身。

有两个工具，都由 HashiCorp 编写，处理这种情况。Packer 帮助以可重复的方式创建 VM 镜像，将所有指令存储为代码，通常以 JSON 文件的形式。Terraform 是一个基础设施即代码工具，这意味着它用于提供所有必要的基础设施资源。我们将使用 Packer 的输出作为 Terraform 的输入。这样，Terraform 将创建一个包含以下内容的整个系统：

+   实例组

+   负载均衡器

+   VPC

+   其他云元素，同时使用包含我们自己代码的 VM

这一部分的标题可能会让您感到困惑。为什么它被称为**不可变基础设施**，而我们明显是在提倡在每次提交后更改整个基础设施？如果您学过函数式语言，不可变性的概念可能对您更清晰。

可变对象是其状态可以改变的对象。在基础设施中，这很容易理解：您可以登录到虚拟机并下载更近期的代码。状态不再与您干预之前相同。

不可变对象是其状态我们无法改变的对象。这意味着我们无法登录到机器上并更改东西。一旦我们从镜像部署了虚拟机，它就会保持不变，直到我们销毁它。这听起来可能非常麻烦，但实际上，它解决了软件维护的一些问题。

## 不可变基础设施的好处

首先，不可变基础设施使配置漂移的概念过时。没有配置管理，因此也不会有漂移。升级也更安全，因为我们不会陷入一个半成品状态。这是既不是上一个版本也不是下一个版本，而是介于两者之间的状态。部署过程提供了二进制信息：机器要么被创建并运行，要么没有。没有其他方式。

为了使不可变基础设施在不影响正常运行时间的情况下工作，您还需要以下内容：

+   负载均衡

+   一定程度的冗余

毕竟，升级过程包括关闭整个实例。您不能依赖于这台机器的地址或任何特定于该机器的东西。相反，您需要至少有第二个机器来处理工作负载，同时用更近期的版本替换另一个机器。当您完成升级一个机器后，您可以重复相同的过程。这样，您将有两个升级的实例而不会丢失服务。这种策略被称为滚动升级。

从这个过程中，您可以意识到，当处理无状态服务时，不可变基础架构效果最佳。当您的服务具有某种持久性时，正确实施变得更加困难。在这种情况下，通常需要将持久性级别拆分为一个单独的对象，例如，包含所有应用程序数据的 NFS 卷。这些卷可以在实例组中的所有机器之间共享，并且每个新机器上线时都可以访问之前运行应用程序留下的共同状态。

## 使用 Packer 构建实例镜像

考虑到我们的示例应用程序已经是无状态的，我们可以继续在其上构建一个不可变的基础架构。由于 Packer 生成的工件是 VM 镜像，我们必须决定要使用的格式和构建器。

让我们专注于 Amazon Web Services 的示例，同时牢记类似的方法也适用于其他支持的提供者。一个简单的 Packer 模板可能如下所示：

```cpp
{
  "variables": {
    "aws_access_key": "",
    "aws_secret_key": ""
  },
  "builders": [{
    "type": "amazon-ebs",
    "access_key": "{{user `aws_access_key`}}",
    "secret_key": "{{user `aws_secret_key`}}",
    "region": "eu-central-1",
    "source_ami": "ami-0f1026b68319bad6c",
    "instance_type": "t2.micro",
    "ssh_username": "admin",
    "ami_name": "Project's Base Image {{timestamp}}"
  }],
  "provisioners": [{
    "type": "shell",
    "inline": [
      "sudo apt-get update",
      "sudo apt-get install -y nginx"
    ]
  }]
}
```

上述代码将使用 EBS 构建器为 Amazon Web Services 构建一个镜像。该镜像将驻留在`eu-central-1`地区，并将基于`ami-5900cc36`，这是一个 Debian Jessie 镜像。我们希望构建器是一个`t2.micro`实例（这是 AWS 中的 VM 大小）。为了准备我们的镜像，我们运行两个`apt-get`命令。

我们还可以重用先前定义的 Ansible 代码，而不是使用 Packer 来配置我们的应用程序，我们可以将 Ansible 替换为 provisioner。我们的代码将如下所示：

```cpp
{
  "variables": {
    "aws_access_key": "",
    "aws_secret_key": ""
  },
  "builders": [{
    "type": "amazon-ebs",
    "access_key": "{{user `aws_access_key`}}",
    "secret_key": "{{user `aws_secret_key`}}",
    "region": "eu-central-1",
    "source_ami": "ami-0f1026b68319bad6c",
    "instance_type": "t2.micro",
    "ssh_username": "admin",
    "ami_name": "Project's Base Image {{timestamp}}"
  }],
  "provisioners": [{
 "type": "ansible",
 "playbook_file": "./provision.yml",
 "user": "admin",
 "host_alias": "baseimage"
 }],
 "post-processors": [{
 "type": "manifest",
 "output": "manifest.json",
 "strip_path": true
 }]
}
```

更改在`provisioners`块中，还添加了一个新的块`post-processors`。这一次，我们不再使用 shell 命令，而是使用一个运行 Ansible 的不同的 provisioner。后处理器用于以机器可读的格式生成构建结果。一旦 Packer 完成构建所需的工件，它会返回其 ID，并将其保存在`manifest.json`中。对于 AWS 来说，这意味着一个 AMI ID，然后我们可以将其提供给 Terraform。

## 使用 Terraform 编排基础架构

使用 Packer 创建镜像是第一步。之后，我们希望部署该镜像以使用它。我们可以使用 Terraform 基于我们的 Packer 模板中的镜像构建一个 AWS EC2 实例。

示例 Terraform 代码如下所示：

```cpp
# Configure the AWS provider
provider "aws" {
  region = var.region
  version = "~> 2.7"
}

# Input variable pointing to an SSH key we want to associate with the 
# newly created machine
variable "public_key_path" {
  description = <<DESCRIPTION
Path to the SSH public key to be used for authentication.
Ensure this keypair is added to your local SSH agent so provisioners can
connect.
Example: ~/.ssh/terraform.pub
DESCRIPTION

  default = "~/.ssh/id_rsa.pub"
}

# Input variable with a name to attach to the SSH key
variable "aws_key_name" {
  description = "Desired name of AWS key pair"
  default = "terraformer"
}

# An ID from our previous Packer run that points to the custom base image
variable "packer_ami" {
}

variable "env" {
  default = "development"
}

variable "region" {
}

# Create a new AWS key pair cotaining the public key set as the input 
# variable
resource "aws_key_pair" "deployer" {
  key_name = var.aws_key_name

  public_key = file(var.public_key_path)
}

# Create a VM instance from the custom base image that uses the previously created key
# The VM size is t2.xlarge, it uses a persistent storage volume of 60GiB,
# and is tagged for easier filtering
resource "aws_instance" "project" {
  ami = var.packer_ami

  instance_type = "t2.xlarge"

  key_name = aws_key_pair.deployer.key_name

  root_block_device {
    volume_type = "gp2"
    volume_size = 60
  }

  tags = {
    Provider = "terraform"
    Env = var.env
    Name = "main-instance"
  }
}
```

这将创建一个密钥对和一个使用此密钥对的 EC2 实例。EC2 实例基于作为变量提供的 AMI。在调用 Terraform 时，我们将设置此变量指向 Packer 生成的镜像。

# 总结

到目前为止，您应该已经了解到，在项目开始阶段实施 CI 如何帮助您节省长期时间。尤其是与 CD 配对时，它还可以减少工作进展。在本章中，我们介绍了一些有用的工具，可以帮助您实施这两个过程。

我们已经展示了 GitLab CI 如何让我们在 YAML 文件中编写流水线。我们已经讨论了代码审查的重要性，并解释了各种形式的代码审查之间的区别。我们介绍了 Ansible，它有助于配置管理和部署代码的创建。最后，我们尝试了 Packer 和 Terraform，将我们的重点从创建应用程序转移到创建系统。

本章中的知识并不局限于 C++语言。您可以在使用任何技术编写的任何语言的项目中使用它。您应该牢记的重要事情是：所有应用程序都需要测试。编译器或静态分析器不足以验证您的软件。作为架构师，您还必须考虑的不仅是您的项目（应用程序本身），还有产品（您的应用程序将在其中运行的系统）。仅交付可工作的代码已不再足够。了解基础架构和部署过程至关重要，因为它们是现代系统的新构建模块。

下一章将专注于软件的安全性。我们将涵盖源代码本身、操作系统级别以及与外部服务和最终用户的可能交互。

# 问题

1.  CI 在开发过程中如何节省时间？

1.  您是否需要单独的工具来实施 CI 和 CD？

1.  在会议中进行代码审查有何意义？

1.  在 CI 期间，您可以使用哪些工具来评估代码的质量？

1.  谁参与指定 BDD 场景？

1.  在什么情况下会考虑使用不可变基础设施？在什么情况下会排除它？

1.  您如何描述 Ansible、Packer 和 Terraform 之间的区别？

# 进一步阅读

+   持续集成/持续部署/持续交付：

[`www.packtpub.com/virtualization-and-cloud/hands-continuous-integration-and-delivery`](https://www.packtpub.com/virtualization-and-cloud/hands-continuous-integration-and-delivery)

[`www.packtpub.com/virtualization-and-cloud/cloud-native-continuous-integration-and-delivery`](https://www.packtpub.com/virtualization-and-cloud/cloud-native-continuous-integration-and-delivery)

+   Ansible：

[`www.packtpub.com/virtualization-and-cloud/mastering-ansible-third-edition`](https://www.packtpub.com/virtualization-and-cloud/mastering-ansible-third-edition)

[`www.packtpub.com/application-development/hands-infrastructure-automation-ansible-video`](https://www.packtpub.com/application-development/hands-infrastructure-automation-ansible-video)

+   Terraform：

[`www.packtpub.com/networking-and-servers/getting-started-terraform-second-edition`](https://www.packtpub.com/networking-and-servers/getting-started-terraform-second-edition)

[`www.packtpub.com/big-data-and-business-intelligence/hands-infrastructure-automation-terraform-aws-video`](https://www.packtpub.com/big-data-and-business-intelligence/hands-infrastructure-automation-terraform-aws-video)

+   黄瓜：

[`www.packtpub.com/web-development/cucumber-cookbook`](https://www.packtpub.com/web-development/cucumber-cookbook)

+   GitLab：

[`www.packtpub.com/virtualization-and-cloud/gitlab-quick-start-guide`](https://www.packtpub.com/virtualization-and-cloud/gitlab-quick-start-guide)

[`www.packtpub.com/application-development/hands-auto-devops-gitlab-ci-video`](https://www.packtpub.com/application-development/hands-auto-devops-gitlab-ci-video)
