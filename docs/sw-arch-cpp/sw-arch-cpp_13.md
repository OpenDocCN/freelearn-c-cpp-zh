# 第十章：代码和部署中的安全性

在建立适当的测试之后，有必要进行安全审计，以确保我们的应用程序不会被用于恶意目的。本章描述了如何评估代码库的安全性，包括内部开发的软件和第三方模块。它还将展示如何在代码级别和操作系统级别改进现有软件。

您将学习如何在每个级别上设计重点放在安全性上的应用程序，从代码开始，通过依赖关系、架构和部署。

本章将涵盖以下主题：

+   检查代码安全性

+   检查依赖项是否安全

+   加固您的代码

+   加固您的环境

# 技术要求

本章中使用的一些示例需要具有以下最低版本的编译器：

+   GCC 10+

+   Clang 3.1+

本章中的代码已经放在 GitHub 上[`github.com/PacktPublishing/Software-Architecture-with-Cpp/tree/master/Chapter10`](https://github.com/PacktPublishing/Software-Architecture-with-Cpp/tree/master/Chapter10)。

# 检查代码安全性

在本章中，我们提供了有关如何检查您的代码、依赖项和环境是否存在潜在威胁的信息。但请记住，遵循本章中概述的每个步骤不一定会保护您免受所有可能的问题。我们的目标是向您展示一些可能的危险以及处理它们的方法。鉴于此，您应始终意识到系统的安全性，并使审计成为例行事件。

在互联网变得无处不在之前，软件作者并不太关心他们设计的安全性。毕竟，如果用户提供了格式不正确的数据，用户最多只能使自己的计算机崩溃。为了利用软件漏洞访问受保护的数据，攻击者必须获得物理访问权限到保存数据的机器。

即使是设计用于网络内部使用的软件，安全性也经常被忽视。以**超文本传输协议**（**HTTP**）为例。尽管它允许对某些资产进行密码保护，但所有数据都是以明文传输的。这意味着在同一网络上的每个人都可以窃听正在传输的数据。

今天，我们应该从设计的最初阶段就开始重视安全，并在软件开发、运营和维护的每个阶段都牢记安全性。我们每天生产的大部分软件都意味着以某种方式与其他现有系统连接。

通过省略安全措施，我们不仅使自己暴露于潜在的攻击、数据泄漏和最终诉讼的风险中，还使我们的合作伙伴暴露于潜在的攻击、数据泄漏和最终诉讼的风险中。请记住，未能保护个人数据可能会导致数百万美元的罚款。

## 注重安全的设计

我们如何为安全性设计架构？这样做的最佳方式是像潜在的攻击者一样思考。有许多方法可以打开一个盒子，但通常，您会寻找不同元素连接的裂缝。（在盒子的情况下，这可能是盒子的盖子和底部之间。）

在软件架构中，元素之间的连接称为接口。由于它们的主要作用是与外部世界进行交互，它们是整个系统中最容易受到攻击的部分。确保您的接口受到保护、直观和稳健将解决软件可能被破坏的最明显的方式。

### 使接口易于使用且难以滥用

为了设计接口既易于使用又难以滥用，考虑以下练习。想象一下你是接口的客户。您希望实现一个使用您的支付网关的电子商务商店，或者您希望实现一个连接本书中始终使用的示例系统的客户 API 的 VR 应用程序。

作为关于接口设计的一般规则，避免以下特征：

+   传递给函数/方法的参数太多

+   参数名称模糊

+   使用输出参数

+   参数取决于其他参数

为什么这些特征被认为是有问题的？

+   第一个特征不仅使参数的含义难以记忆，而且使参数的顺序也难以记忆。这可能导致使用错误，进而可能导致崩溃和安全问题。

+   第二个特征与第一个特征有类似的后果。通过使接口使用起来不太直观，您使用户更容易犯错误。

+   第三个特征是第二个特征的一个变体，但有一个额外的转折。用户不仅需要记住哪些参数是输入，哪些是输出，还需要记住如何处理输出。谁管理资源的创建和删除？这是如何实现的？背后的内存管理模型是什么？

使用现代 C++，返回包含所有必要数据的值比以往任何时候都更容易。通过对成对、元组和向量的使用，没有理由使用输出参数。此外，返回值有助于接受不修改对象状态的做法。这反过来又减少了与并发相关的问题。

+   最后一个特征引入了不必要的认知负荷，就像前面的例子一样，可能导致错误，最终导致失败。这样的代码也更难测试和维护，因为每次引入的更改都必须考虑到已经存在的所有可能的组合。未能正确处理任何组合都是对系统的潜在威胁。

接口的前述规则适用于接口的外部部分。您还应该通过验证输入、确保值正确和合理，并防止接口提供的服务被不必要地使用来对内部部分应用类似的措施。

### 启用自动资源管理

系统不稳定也可能是由于内存泄漏、数据竞争和死锁引起的。所有这些症状都是资源管理不善的表现。尽管资源管理是一个难题，但有一种机制可以帮助您减少问题的数量。这样的机制之一是自动资源管理。

在这种情况下，资源是通过操作系统获得访问权限的东西，您必须确保正确使用它。这可能意味着使用动态分配的内存、打开文件、套接字、进程或线程。当您获取它们和释放它们时，所有这些都需要采取特定的操作。其中一些在其生命周期内还需要特定的操作。在正确的时间释放这些资源失败会导致泄漏。由于资源通常是有限的，从长远来看，泄漏将导致无法创建新资源时出现意外行为。

资源管理在 C++中非常重要，因为与许多其他高级语言不同，C++中没有垃圾回收，软件开发人员负责资源的生命周期。了解这种生命周期有助于创建安全稳定的系统。

资源管理最常见的习惯用法是**资源获取即初始化**（**RAII**）。尽管它起源于 C++，但它也被用于其他语言，如 Vala 和 Rust。这种习惯用法使用对象的构造函数和析构函数来分配和释放资源。这样，我们可以保证在持有资源的对象超出范围时，资源将被正确释放。

在标准库中使用此习惯用法的一些示例是`std::unique_ptr`和`std::shared_ptr`智能指针类型。其他示例包括互斥锁-`std::lock_guard`、`std::unique_lock`和`std:shared_lock`-或文件-`std::ifstream`和`std::ofstream`。

**指南支持库**（**GSL**），我们将很快讨论，还实现了一项特别有用的自动资源管理指南。通过在我们的代码中使用`gsl::finally()`函数，我们创建了一个附有一些代码的`gsl::final_action()`对象。当对象的析构函数被调用时，这些代码将被执行。这意味着该代码将在成功从函数返回时执行，以及在发生异常期间进行堆栈展开时执行。

这种方法不应该经常使用，因为通常最好在设计类时考虑 RAII。但如果您正在与第三方模块进行接口，并且希望确保包装器的安全性，`finally()`可以帮助您实现这一点。

举个例子，假设我们有一个支付操作员，每个账户只允许一个并发登录。如果我们不想阻止用户进行未来的支付，我们应该在完成交易处理后立即注销。当一切按照我们的设计进行时，这并不是一个问题。但在发生异常时，我们也希望安全地释放资源。以下是我们可以使用`gsl::finally()`来实现的方式：

```cpp
TransactionStatus processTransaction(AccountName account, ServiceToken token,

Amount amount)

{

  payment::login(account, token);

  auto _ = gsl::finally([] { payment::logout(); });

  payment::process(amount); // We assume this can lead to exception


  return TransactionStatus::TransactionSuccessful;

}
```

无论在调用`payment::process()`期间发生了什么，我们至少可以保证在退出`processTransaction()`的范围时注销用户。

简而言之，使用 RAII 使您在类设计阶段更多地考虑资源管理，同时在您完全控制代码并且在您使用接口时不再那么清晰时，您不再那么考虑。

### 并发的缺点及如何处理

虽然并发可以提高性能和资源利用率，但也使您的代码更难设计和调试。这是因为，与单线程流程不同，操作的时间无法提前确定。在单线程代码中，您要么写入资源，要么从中读取，但您总是知道操作的顺序，因此可以预测对象的状态。

并发时，多个线程或进程可以同时从对象中读取或修改。如果修改不是原子的，我们可能会遇到常见更新问题的变体之一。考虑以下代码：

```cpp
TransactionStatus chargeTheAccount(AccountNumber acountNumber, Amount amount)

{

  Amount accountBalance = getAcountBalance(accountNumber);

  if (accountBalance > amount)

  {

    setAccountBalance(accountNumber, accountBalance - amount);

    return TransactionStatus::TransactionSuccessful;

  }

  return TransactionStatus::InsufficientFunds;

}
```

调用`chargeTheAccount`函数时，从非并发代码中，一切都会顺利进行。我们的程序将检查账户余额，并在可能的情况下进行扣款。然而，并发执行可能会导致负余额。这是因为两个线程可以依次调用`getAccountBalance()`，它将返回相同的金额，比如`20`。在执行完该调用后，两个线程都会检查当前余额是否高于可用金额。最后，在检查后，它们修改账户余额。假设两个交易金额都为`10`，每个线程都会将余额设置为 20-10=10。在*两个*操作之后，账户的余额为 10，尽管它应该是 0！

为了减轻类似问题，我们可以使用诸如互斥锁和临界区、CPU 提供的原子操作或并发安全数据结构等解决方案。

互斥锁、临界区和其他类似的并发设计模式可以防止多个线程修改（或读取）数据。尽管它们在设计并发应用程序时很有用，但与之相关的是一种权衡。它们有效地使您的代码的某些部分变成单线程。这是因为由互斥锁保护的代码只允许一个线程执行；其他所有线程都必须等待，直到互斥锁被释放。由于我们引入了等待，即使我们最初的目标是使代码更具性能，我们也可能使代码的性能下降。

原子操作意味着使用单个 CPU 指令来获得期望的效果。这个术语可以指任何将高级操作转换为单个 CPU 指令的操作。当单个指令实现的效果*超出*通常可能的范围时，它们特别有趣。例如，**比较和交换**（**CAS**）是一种指令，它将内存位置与给定值进行比较，并仅在比较成功时将该位置的内容修改为新值。自 C++11 以来，有一个`<std::atomic>`头文件可用，其中包含几种原子数据类型和操作。例如，CAS 被实现为一组`compare_and_exchange_*`函数。

最后，并发安全的数据结构（也称为并发数据结构）为数据结构提供了安全的抽象，否则这些数据结构将需要某种形式的同步。例如，Boost.Lockfree（[`www.boost.org/doc/libs/1_66_0/doc/html/lockfree.html`](https://www.boost.org/doc/libs/1_66_0/doc/html/lockfree.html)）库提供了用于多个生产者和多个消费者的并发队列和栈。libcds（[`github.com/khizmax/libcds`](https://github.com/khizmax/libcds)）还提供了有序列表、集合和映射，但截至撰写本书时，已经有几年没有更新了。

在设计并发处理时要牢记的有用规则如下：

+   首先考虑是否需要并发。

+   通过值传递数据，而不是通过指针或引用。这可以防止其他线程在读取数据时修改该值。

+   如果数据的大小使得按值共享变得不切实际，可以使用`shared_ptr`。这样，更容易避免资源泄漏。

## 安全编码、指南和 GSL

标准 C++基金会发布了一套指南，记录了构建 C++系统的最佳实践。这是一个在 GitHub 上发布的 Markdown 文档，网址为[`github.com/isocpp/CppCoreGuidelines`](https://github.com/isocpp/CppCoreGuidelines)。这是一个不断发展的文档，没有发布计划（不像 C++标准本身）。这些指南针对的是现代 C++，基本上意味着实现了至少 C++11 特性的代码库。

指南中提出的许多规则涵盖了我们在本章中介绍的主题。例如，有关接口设计、资源管理和并发的规则。指南的编辑是 Bjarne Stroustrup 和 Herb Sutter，他们都是 C++社区中受尊敬的成员。

我们不会详细描述这些指南。我们鼓励您自己阅读。本书受到其中许多规则的启发，并在我们的示例中遵循这些规则。

为了方便在各种代码库中使用这些规则，微软发布了**指南支持库**（**GSL**）作为一个开源项目，托管在[`github.com/microsoft/GSL`](https://github.com/microsoft/GSL)上。这是一个仅包含头文件的库，您可以将其包含在项目中以使用定义的类型。您可以包含整个 GSL，也可以选择性地仅使用您计划使用的一些类型。

该库的另一个有趣之处在于它使用 CMake 进行构建，Travis 进行持续集成，以及 Catch 进行单元测试。因此，它是我们在第七章、*构建和打包*，第八章、*可测试代码编写*和第九章、*持续集成和持续部署*中涵盖的主题的一个很好的例子。

## 防御性编码，验证一切

在前一章中，我们提到了防御性编程的方法。尽管这种方法并不严格属于安全功能，但它确实有助于创建健壮的接口。这样的接口反过来又增加了系统的整体安全性。

作为一个很好的启发式方法，您可以将所有外部数据视为不安全。我们所说的外部数据是通过某个接口（编程接口或用户界面）进入系统的每个输入。为了表示这一点，您可以在适当的类型前加上`Unsafe`前缀，如下所示：

```cpp
RegistrationResult registerUser(UnsafeUsername username, PasswordHash passwordHash)

{

  SafeUsername safeUsername = username.sanitize();

  try

  {

    std::unique_ptr<User> user = std::make_unique<User>(safeUsername, passwordHash);

    CommitResult result = user->commit();

    if (result == CommitResult::CommitSuccessful)

    {

      return RegistrationResult::RegistrationSuccessful;

    }

    else

    {

      return RegistrationResult::RegistrationUnsuccessful;

    }

  }

  catch (UserExistsException _)

  {

    return RegistrationResult::UserExists;

  }

}
```

如果您已经阅读了指南，您将知道通常应避免直接使用 C API。C API 中的一些函数可能以不安全的方式使用，并需要特别小心地防御性使用它们。最好使用 C++中相应的概念，以确保更好的类型安全性和保护（例如，防止缓冲区溢出）。

防御性编程的另一个方面是智能地重用现有代码。每次尝试实现某种技术时，请确保没有其他人在您之前实现过它。当您学习一种新的编程语言时，自己编写排序算法可能是一种有趣的挑战，但对于生产代码，最好使用标准库中提供的排序算法。对于密码哈希也是一样。毫无疑问，您可以找到一些聪明的方法来计算密码哈希并将其存储在数据库中，但通常更明智的做法是使用经过验证的`bcrypt`。请记住，智能的代码重用假设您以与您自己的代码一样的尽职调查检查和审计第三方解决方案。我们将在下一节“我的依赖项安全吗？”中深入探讨这个话题。

值得注意的是，防御性编程不应该变成偏执的编程。检查用户输入是明智的做法，而在初始化变量后立即断言初始化变量是否仍然等于原始值则有些过分。您希望控制数据和算法的完整性以及第三方解决方案的完整性。您不希望通过采用语言特性来验证编译器的正确性。

简而言之，从安全性和可读性的角度来看，使用 C++核心指南中提出的`Expects()`和`Ensures()`以及通过类型和转换区分不安全和安全数据是一个好主意。

## 最常见的漏洞

要检查您的代码是否安全防范最常见的漏洞，您应首先了解这些漏洞。毕竟，只有当您知道攻击是什么样子时，防御才有可能。**开放式网络应用安全项目**（**OWASP**）已经对最常见的漏洞进行了分类，并在[`www.owasp.org/index.php/Category:OWASP_Top_Ten_Project`](https://www.owasp.org/index.php/Category:OWASP_Top_Ten_Project)上发布了它们。在撰写本书时，这些漏洞如下：

+   注入：通常称为 SQL 注入。这不仅限于 SQL；当不受信任的数据直接传递给解释器（如 SQL 数据库、NoSQL 数据库、shell 或 eval 函数）时，就会出现这种漏洞。攻击者可能以这种方式访问应该受到保护的系统部分。

+   **破坏的身份验证**：如果身份验证实施不当，攻击者可能利用漏洞来获取秘密数据或冒充其他用户。

+   **敏感数据暴露**：缺乏加密和适当的访问权限可能导致敏感数据被公开。

+   **XML 外部实体**（**XXE**）：一些 XML 处理器可能会泄露服务器文件系统的内容或允许远程代码执行。

+   **破坏的访问控制**：当访问控制未正确执行时，攻击者可能会访问应受限制的文件或数据。

+   **安全配置错误**：使用不安全的默认值和不正确的配置是最常见的漏洞来源。

+   **跨站脚本攻击**（**XSS**）：包括并执行不受信任的外部数据，特别是使用 JavaScript，这允许控制用户的网络浏览器。

+   **不安全的反序列化**：一些有缺陷的解析器可能会成为拒绝服务攻击或远程代码执行的牺牲品。

+   **使用已知漏洞的组件**：现代应用程序中的许多代码都是第三方组件。这些组件应该定期进行审计和更新，因为单个依赖中已知的安全漏洞可能导致整个应用程序和数据被攻击。幸运的是，有一些工具可以帮助自动化这一过程。

+   **日志和监控不足**：如果你的系统受到攻击，而你的日志和监控不够彻底，攻击者可能会获得更深入的访问权限，而你却没有察觉。

我们不会详细介绍每个提到的漏洞。我们想要强调的是，通过将所有外部数据视为不安全，你可以首先通过删除所有不安全的内容来对其进行净化，然后再开始实际处理。

当涉及到日志和监控不足时，我们将在第十五章中详细介绍*云原生设计*。在那里，我们将介绍一些可能的可观察性方法，包括日志记录、监控和分布式跟踪。

# 检查依赖是否安全

计算机早期，所有程序都是单体结构，没有任何外部依赖。自操作系统诞生以来，任何非平凡的软件很少能摆脱依赖。这些依赖可以分为两种形式：外部依赖和内部依赖。

+   外部依赖是我们运行应用程序时应该存在的环境。例如，前面提到的操作系统、动态链接库和其他应用程序（如数据库）。

+   内部依赖是我们想要重用的模块，因此通常是静态库或仅包含头文件的库。

两种依赖都提供潜在的安全风险。随着每一行代码增加漏洞的风险，你拥有的组件越多，你的系统可能受到攻击的机会就越高。在接下来的章节中，我们将看到如何检查你的软件是否确实容易受到已知的漏洞攻击。

## 通用漏洞和暴露

检查软件中已知的安全问题的第一个地方是**通用漏洞和暴露**（**CVE**）列表，可在[`cve.mitre.org/`](https://cve.mitre.org/)上找到。该列表由几个被称为**CVE 编号机构**（**CNAs**）的机构不断更新。这些机构包括供应商和项目、漏洞研究人员、国家和行业 CERT 以及漏洞赏金计划。

该网站还提供了一个搜索引擎。通过这个，你可以使用几种方法了解漏洞：

+   你可以输入漏洞编号。这些编号以`CVE`为前缀，例如 CVE-2014-6271，臭名昭著的 ShellShock，或者 CVE-2017-5715，也被称为 Spectre。

+   你可以输入漏洞的通用名称，比如前面提到的 ShellShock 或 Spectre。

+   你可以输入你想审计的软件名称，比如 Bash 或 Boost。

对于每个搜索结果，你可以看到描述以及其他 bug 跟踪器和相关资源的参考列表。描述通常列出受漏洞影响的版本，因此你可以检查你计划使用的依赖是否已经修补。

## 自动化扫描器

有一些工具可以帮助您审计依赖项列表。其中一个工具是 OWASP Dependency-Check ([`www.owasp.org/index.php/OWASP_Dependency_Check`](https://www.owasp.org/index.php/OWASP_Dependency_Check))。尽管它只正式支持 Java 和.NET，但它对 Python、Ruby、Node.js 和 C++（与 CMake 或`autoconf`一起使用时）有实验性支持。除了作为独立工具使用外，它还可以与 Jenkins、SonarQube 和 CircleCI 等**持续集成/持续部署**（**CI/CD**）软件集成。

另一个允许检查已知漏洞的依赖项的工具是 Snyk。这是一个商业产品，有几个支持级别。与 OWASP Dependency-Check 相比，它还可以执行更多操作，因为 Snyk 还可以审计容器映像和许可合规性问题。它还提供了更多与第三方解决方案的集成。

## 自动化依赖项升级管理

监视依赖项的漏洞只是确保项目安全的第一步。之后，您需要采取行动并手动更新受损的依赖项。正如您可能已经预料到的那样，也有专门的自动化解决方案。其中之一是 Dependabot，它会扫描您的源代码存储库，并在有安全相关更新可用时发布拉取请求。在撰写本书时，Dependabot 尚不支持 C++。但是，它可以与您的应用程序可能使用的其他语言一起使用。除此之外，它还可以扫描 Docker 容器，查找基础映像中发现的漏洞。

自动化依赖项管理需要成熟的测试支持。在没有测试的情况下切换依赖项版本可能会导致不稳定和错误。防止与依赖项升级相关的问题的一种保护措施是使用包装器与第三方代码进行接口。这样的包装器可能有自己的一套测试，可以在升级期间立即告诉我们接口何时被破坏。

# 加固您的代码

通过使用现代 C++构造而不是较旧的 C 等效构造，可以减少自己代码中常见的安全漏洞数量。然而，即使更安全的抽象也可能存在漏洞。仅仅选择更安全的实现并认为自己已经尽了最大努力是不够的。大多数情况下，都有方法可以进一步加固您的代码。

但是什么是代码加固？根据定义，这是减少系统漏洞表面的过程。通常，这意味着关闭您不会使用的功能，并追求一个简单的系统而不是一个复杂的系统。这也可能意味着使用工具来增加已有功能的健壮性。

这些工具可能意味着在操作系统级别应用内核补丁、防火墙和**入侵检测系统**（**IDSes**）。在应用程序级别，这可能意味着使用各种缓冲区溢出和下溢保护机制，使用容器和**虚拟机**（**VMs**）进行特权分离和进程隔离，或者强制执行加密通信和存储。

在本节中，我们将重点介绍应用程序级别的一些示例，而下一节将重点介绍操作系统级别。

## 面向安全的内存分配器

如果您认真保护应用程序免受与堆相关的攻击，例如堆溢出、释放后使用或双重释放，您可能会考虑用面向安全的版本替换标准内存分配器。可能感兴趣的两个项目如下：

+   FreeGuard，可在[`github.com/UTSASRG/FreeGuard`](https://github.com/UTSASRG/FreeGuard)上找到，并在[`arxiv.org/abs/1709.02746`](https://arxiv.org/abs/1709.02746)的论文中描述

+   GrapheneOS 项目的`hardened_malloc`，可在[`github.com/GrapheneOS/hardened_malloc`](https://github.com/GrapheneOS/hardened_malloc)上找到

FreeGuard 于 2017 年发布，自那时以来除了零星的错误修复外，没有太多变化。另一方面，`hardened_malloc`正在积极开发。这两个分配器都旨在作为标准`malloc()`的替代品。您可以通过设置`LD_PRELOAD`环境变量或将库添加到`/etc/preload.so`配置文件中，而无需修改应用程序即可使用它们。虽然 FreeGuard 针对的是 64 位 x86 系统上的 Linux 与 Clang 编译器，`hardened_malloc`旨在更广泛的兼容性，尽管目前主要支持 Android 的 Bionic，`musl`和`glibc`。`hardened_malloc`也基于 OpenBSD 的`alloc`，而 OpenBSD 本身是一个以安全为重点的项目。

不要替换内存分配器，可以替换你用于更安全的集合。 SaferCPlusPlus（[`duneroadrunner.github.io/SaferCPlusPlus/`](https://duneroadrunner.github.io/SaferCPlusPlus/)）项目提供了`std::vector<>`，`std::array<>`和`std::string`的替代品，可以作为现有代码中的替代品。该项目还包括用于保护未初始化使用或符号不匹配的基本类型的替代品，并发数据类型的替代品，以及指针和引用的替代品。

## 自动化检查

有一些工具可以特别有助于确保正在构建的系统的安全。我们将在下一节中介绍它们。

### 编译器警告

虽然编译器警告本身不一定是一个工具，但可以使用和调整编译器警告，以实现更好的输出，从而使每个 C++开发人员都将使用的 C++编译器获得更好的输出。

由于编译器已经可以进行一些比标准要求更深入的检查，建议利用这种可能性。当使用诸如 GCC 或 Clang 之类的编译器时，推荐的设置包括`-Wall -Wextra`标志。这将生成更多的诊断，并在代码不遵循诊断时产生警告。如果您想要非常严格，还可以启用`-Werror`，这将把所有警告转换为错误，并阻止不能通过增强诊断的代码的编译。如果您想严格遵循标准，还有`-pedantic`和`-pedantic-errors`标志，将检查是否符合标准。

在使用 CMake 进行构建时，您可以使用以下函数在编译期间启用这些标志：

```cpp
add_library(customer ${SOURCES_GO_HERE})

target_include_directories(customer PUBLIC include)

target_compile_options(customer PRIVATE -Werror -Wall -Wextra)
```

这样，除非您修复编译器报告的所有警告（转换为错误），否则编译将失败。

您还可以在 OWASP（[`www.owasp.org/index.php/C-Based_Toolchain_Hardening`](https://www.owasp.org/index.php/C-Based_Toolchain_Hardening)）和 Red Hat（[`developers.redhat.com/blog/2018/03/21/compiler-and-linker-flags-gcc/`](https://developers.redhat.com/blog/2018/03/21/compiler-and-linker-flags-gcc/)）的文章中找到工具链加固的建议设置。

### 静态分析

一类可以帮助使您的代码更安全的工具是所谓的**静态应用安全测试**（**SAST**）工具。它们是专注于安全方面的静态分析工具的变体。

SAST 工具很好地集成到 CI/CD 管道中，因为它们只是读取您的源代码。输出通常也适用于 CI/CD，因为它突出显示了源代码中特定位置发现的问题。另一方面，静态分析可能会忽略许多类型的问题，这些问题无法自动发现，或者仅通过静态分析无法发现。这些工具也对与配置相关的问题视而不见，因为配置文件并未在源代码本身中表示。

C++ SAST 工具的示例包括以下开源解决方案：

+   Cppcheck（[`cppcheck.sourceforge.net/`](http://cppcheck.sourceforge.net/)）是一个通用的静态分析工具，专注于较少的误报。

+   Flawfinder（[`dwheeler.com/flawfinder/`](https://dwheeler.com/flawfinder/)），似乎没有得到积极维护

+   LGTM（[`lgtm.com/help/lgtm/about-lgtm`](https://lgtm.com/help/lgtm/about-lgtm)），支持多种不同的语言，并具有对拉取请求的自动化分析功能

+   SonarQube（[`www.sonarqube.org/`](https://www.sonarqube.org/)）具有出色的 CI/CD 集成和语言覆盖，并提供商业版本

还有商业解决方案可用：

+   Checkmarx CxSAST（[`www.checkmarx.com/products/static-application-security-testing/`](https://www.checkmarx.com/products/static-application-security-testing/)），承诺零配置和广泛的语言覆盖

+   CodeSonar（[`www.grammatech.com/products/codesonar`](https://www.grammatech.com/products/codesonar)），专注于深度分析和发现最多的缺陷

+   Klocwork（[`www.perforce.com/products/klocwork`](https://www.perforce.com/products/klocwork)），专注于准确性

+   Micro Focus Fortify（[`www.microfocus.com/en-us/products/static-code-analysis-sast/overview`](https://www.microfocus.com/en-us/products/static-code-analysis-sast/overview)），支持广泛的语言并集成了同一制造商的其他工具

+   Parasoft C/C++test（[`www.parasoft.com/products/ctest`](https://www.parasoft.com/products/ctest)），这是一个集成的静态和动态分析、单元测试、跟踪等解决方案

+   MathWorks 的 Polyspace Bug Finder（[`www.mathworks.com/products/polyspace-bug-finder.html`](https://www.mathworks.com/products/polyspace-bug-finder.html)），集成了 Simulink 模型

+   Veracode 静态分析（[`www.veracode.com/products/binary-static-analysis-sast`](https://www.veracode.com/products/binary-static-analysis-sast)），这是一个用于静态分析的 SaaS 解决方案

+   WhiteHat Sentinel Source（[`www.whitehatsec.com/platform/static-application-security-testing/`](https://www.whitehatsec.com/platform/static-application-security-testing/)），也专注于消除误报

### 动态分析

就像静态分析是在源代码上执行的一样，动态分析是在生成的二进制文件上执行的。名称中的“动态”指的是观察代码在处理实际数据时的行为。当专注于安全性时，这类工具也可以被称为**动态应用安全性测试**（**DAST**）。

它们相对于 SAST 工具的主要优势在于，它们可以发现许多从源代码分析角度看不到的流程。当然，这也带来了一个缺点，即您必须运行应用程序才能进行分析。而且我们知道，运行应用程序可能既耗时又耗内存。

DAST 工具通常专注于与 Web 相关的漏洞，如 XSS、SQL（和其他）注入或泄露敏感信息。我们将在下一小节中更多地关注一个更通用的动态分析工具 Valgrind。

#### Valgrind 和 Application Verifier

Valgrind 主要以内存泄漏调试工具而闻名。实际上，它是一个帮助构建与内存问题无关的动态分析工具的仪器框架。除了内存错误检测器外，该套工具目前还包括线程错误检测器、缓存和分支预测分析器以及堆分析器。它在类 Unix 操作系统（包括 Android）上支持各种平台。

基本上，Valgrind 充当虚拟机，首先将二进制文件转换为称为中间表示的简化形式。它不是在实际处理器上运行程序，而是在这个虚拟机下执行，以便分析和验证每个调用。

如果您在 Windows 上开发，可以使用**Application Verifier**（**AppVerifier**）代替 Valgrind。AppVerifier 可以帮助您检测稳定性和安全性问题。它可以监视运行中的应用程序和用户模式驱动程序，以查找内存问题，如泄漏和堆破坏，线程和锁定问题，句柄的无效使用等。

#### 消毒剂

消毒剂是基于代码的编译时仪器的动态测试工具。它们可以帮助提高系统的整体稳定性和安全性，避免未定义的行为。在[`github.com/google/sanitizers`](https://github.com/google/sanitizers)，您可以找到 LLVM（Clang 基于此）和 GCC 的实现。它们解决了内存访问、内存泄漏、数据竞争和死锁、未初始化内存使用以及未定义行为的问题。

**AddressSanitizer**（**ASan**）可保护您的代码免受与内存寻址相关的问题，如全局缓冲区溢出，释放后使用或返回后使用堆栈。尽管它是同类解决方案中最快的之一，但仍会使进程减速约两倍。最好在运行测试和进行开发时使用它，但在生产构建中关闭它。您可以通过向 Clang 添加`-fsanitize=address`标志来为您的构建启用它。

**AddressSanitizerLeakSanitizer**（**LSan**）与 ASan 集成以查找内存泄漏。它在 x86_64 Linux 和 x86_64 macOS 上默认启用。它需要设置一个环境变量，`ASAN_OPTIONS=detect_leaks=1`。LSan 在进程结束时执行泄漏检测。LSan 也可以作为一个独立库使用，而不需要 AddressSanitizer，但这种模式测试较少。

**ThreadSanitizer**（**TSan**），正如我们之前提到的，可以检测并发问题，如数据竞争和死锁。您可以使用`-fsanitize=thread`标志启用它到 Clang。

**MemorySanitizer**（**MSan**）专注于与对未初始化内存的访问相关的错误。它实现了我们在前一小节中介绍的 Valgrind 的一些功能。MSan 支持 64 位 x86、ARM、PowerPC 和 MIPS 平台。您可以通过向 Clang 添加`-fsanitize=memory -fPIE -pie`标志来启用它（这也会打开位置无关可执行文件，这是我们稍后将讨论的概念）。

**硬件辅助地址消毒剂**（**HWASAN**）类似于常规 ASan。主要区别在于尽可能使用硬件辅助。目前，此功能仅适用于 64 位 ARM 架构。

**UndefinedBehaviorSanitizer**（**UBSan**）寻找未定义行为的其他可能原因，如整数溢出、除以零或不正确的位移操作。您可以通过向 Clang 添加`-fsanitize=undefined`标志来启用它。

尽管消毒剂可以帮助您发现许多潜在问题，但它们只有在您对其进行测试时才有效。在使用消毒剂时，请记住保持测试的代码覆盖率高，否则您可能会产生一种虚假的安全感。

#### 模糊测试

作为 DAST 工具的一个子类，模糊测试检查应用程序在面对无效、意外、随机或恶意形成的数据时的行为。在针对跨越信任边界的接口（如最终用户文件上传表单或输入）时，此类检查尤其有用。

此类别中的一些有趣工具包括以下内容：

+   Peach Fuzzer：[`www.peach.tech/products/peach-fuzzer/`](https://www.peach.tech/products/peach-fuzzer/)

+   PortSwigger Burp：[`portswigger.net/burp`](https://portswigger.net/burp)

+   OWASP Zed Attack Proxy 项目：[`www.owasp.org/index.php/OWASP_Zed_Attack_Proxy_Project`](https://www.owasp.org/index.php/OWASP_Zed_Attack_Proxy_Project)

+   Google 的 ClusterFuzz：[`github.com/google/clusterfuzz`](https://github.com/google/clusterfuzz)（和 OSS-Fuzz：[`github.com/google/oss-fuzz`](https://github.com/google/oss-fuzz)）

## 进程隔离和沙箱

如果您想在自己的环境中运行未经验证的软件，您可能希望将其与系统的其余部分隔离开来。通过虚拟机、容器或 AWS Lambda 使用的 Firecracker（[`firecracker-microvm.github.io/`](https://firecracker-microvm.github.io/)）等微型虚拟机，可以对执行的代码进行沙盒化。

这样，一个应用程序的崩溃、泄漏和安全问题不会传播到整个系统，使其变得无用或者受到威胁。由于每个进程都有自己的沙盒，最坏的情况就是只丢失这一个服务。

对于 C 和 C++代码，还有一个由谷歌领导的开源项目**Sandboxed API**（**SAPI**；[`githu`](https://github.com/google/sandboxed-api)[b.com/google/sandboxed-api](https://github.com/google/sandboxed-api)[)，它允许构建沙盒不是为整个进程，而是为库。它被谷歌自己的 Chrome 和 Chromium 网页浏览器等使用。](https://github.com/google/sandboxed-api)

即使虚拟机和容器可以成为进程隔离策略的一部分，也不要将它们与微服务混淆，后者通常使用类似的构建模块。微服务是一种架构设计模式，它们并不自动等同于更好的安全性。

# 加固您的环境

即使您采取了必要的预防措施，确保您的依赖项和代码没有已知的漏洞，仍然存在一个可能会危及您的安全策略的领域。所有应用程序都需要一个执行环境，这可能意味着容器、虚拟机或操作系统。有时，这也可能意味着底层基础设施。

当运行应用程序的操作系统具有开放访问权限时，仅仅使应用程序达到最大程度的硬化是不够的。这样，攻击者可以从系统或基础设施级别直接获取未经授权的数据，而不是针对您的应用程序。

本节将重点介绍一些硬化技术，您可以在执行的最低级别应用这些技术。

## 静态与动态链接

链接是在编译后发生的过程，当您编写的代码与其各种依赖项（如标准库）结合在一起时。链接可以在构建时、加载时（操作系统执行二进制文件时）或运行时发生，如插件和其他动态依赖项的情况。最后两种用例只可能发生在动态链接中。

那么，动态链接和静态链接有什么区别呢？使用静态链接，所有依赖项的内容都会被复制到生成的二进制文件中。当程序加载时，操作系统将这个单一的二进制文件放入内存并执行它。静态链接是由称为链接器的程序在构建过程的最后一步执行的。

由于每个可执行文件都必须包含所有的依赖项，静态链接的程序往往体积较大。这也有其好处；因为执行所需的一切都已经在一个地方可用，所以执行速度可能会更快，并且加载程序到内存中所需的时间总是相同的。对依赖项的任何更改都需要重新编译和重新链接；没有办法升级一个依赖项而不改变生成的二进制文件。

在动态链接中，生成的二进制文件包含您编写的代码，但是依赖项的内容被替换为需要单独加载的实际库的引用。在加载时，动态加载器的任务是找到适当的库并将它们加载到内存中与您的二进制文件一起。当多个应用程序同时运行并且它们每个都使用类似的依赖项（例如 JSON 解析库或 JPEG 处理库）时，动态链接的二进制文件将导致较低的内存使用率。这是因为只有一个给定库的副本可以加载到内存中。相比之下，使用静态链接的二进制文件中相同的库会作为结果的一部分一遍又一遍地加载。当您需要升级其中一个依赖项时，您可以在不触及系统的任何其他组件的情况下进行。下次加载应用程序到内存时，它将自动引用新升级的组件。

静态和动态链接也具有安全性影响。更容易未经授权地访问动态链接的应用程序。这可以通过在常规库的位置替换受损的动态库或在每次新执行的进程中预加载某些库来实现。

当您将静态链接与容器结合使用时（在后面的章节中详细解释），您将获得小型、安全、沙箱化的执行环境。您甚至可以进一步使用这些容器与基于微内核的虚拟机，从而大大减少攻击面。

## 地址空间布局随机化

**地址空间布局随机化**（**ASLR**）是一种用于防止基于内存的攻击的技术。它通过用随机化的内存布局替换程序和数据的标准布局来工作。这意味着攻击者无法可靠地跳转到在没有 ASLR 的系统上本来存在的特定函数。

当与**不执行**（**NX**）位支持结合使用时，这种技术可以变得更加有效。NX 位标记内存中的某些页面，例如堆和栈，只包含不能执行的数据。大多数主流操作系统都已实现了 NX 位支持，并且可以在硬件支持时使用。

## DevSecOps

为了按可预测的方式交付软件增量，最好采用 DevOps 理念。简而言之，DevOps 意味着打破传统模式，鼓励业务、软件开发、软件运营、质量保证和客户之间的沟通。DevSecOps 是 DevOps 的一种形式，它还强调了在每个步骤中考虑安全性的必要性。

这意味着您正在构建的应用程序从一开始就具有内置的可观察性，利用 CI/CD 流水线，并定期扫描漏洞。DevSecOps 使开发人员在基础架构设计中发挥作用，并使运营专家在构成应用程序的软件包设计中发挥作用。由于每个增量代表一个可工作的系统（尽管不是完全功能的），因此安全审计定期进行，所需时间比正常情况下少。这导致更快速和更安全的发布，并允许更快地对安全事件做出反应。

# 总结

在本章中，我们讨论了安全系统的不同方面。由于安全性是一个复杂的主题，您不能仅从自己的应用程序的角度来处理它。现在所有的应用程序都在某种环境中运行，要么控制这个环境并根据您的要求塑造它，要么通过沙箱化和隔离代码来保护自己免受环境的影响。

阅读完本章后，您现在可以开始搜索依赖项和自己代码中的漏洞。您知道如何设计增强安全性的系统以及使用哪些工具来发现可能的缺陷。保持安全是一个持续的过程，但良好的设计可以减少未来的工作量。

下一章将讨论可扩展性以及在系统扩展时可能面临的各种挑战。

# 问题

1.  为什么安全在现代系统中很重要？

1.  并发的一些挑战是什么？

1.  C++核心指南是什么？

1.  安全编码和防御性编码有什么区别？

1.  您如何检查您的软件是否包含已知的漏洞？

1.  静态分析和动态分析有什么区别？

1.  静态链接和动态链接有什么区别？

1.  您如何使用编译器来解决安全问题？

1.  您如何在 CI 流程中实施安全意识？

# 进一步阅读

**一般的网络安全**：

+   [`www.packtpub.com/eu/networking-and-servers/hands-cybersecurity-architects`](https://www.packtpub.com/eu/networking-and-servers/hands-cybersecurity-architects)

+   [`www.packtpub.com/eu/networking-and-servers/information-security-handbook`](https://www.packtpub.com/eu/networking-and-servers/information-security-handbook)

+   [`www.owasp.org/index.php/Main_Page`](https://www.owasp.org/index.php/Main_Page)

+   [`www.packtpub.com/eu/networking-and-servers/practical-security-automation-and-testing`](https://www.packtpub.com/eu/networking-and-servers/practical-security-automation-and-testing)

**并发**：

+   [`www.packtpub.com/eu/application-development/concurrent-patterns-and-best-practices`](https://www.packtpub.com/eu/application-development/concurrent-patterns-and-best-practices)

+   [`www.packtpub.com/eu/application-development/mastering-c-multithreading`](https://www.packtpub.com/eu/application-development/mastering-c-multithreading)

**操作系统加固**：

+   [`www.packtpub.com/eu/networking-and-servers/mastering-linux-security-and-hardening`](https://www.packtpub.com/eu/networking-and-servers/mastering-linux-security-and-hardening)
