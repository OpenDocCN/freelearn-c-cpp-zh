# 前言

# 关于本书

C#是一种强大而多才多艺的面向对象编程（OOP）语言，可以打开各种职业道路。但是，与任何编程语言一样，学习 C#可能是具有挑战性的。由于有各种不同的资源可用，很难知道从哪里开始。

这就是*The C# Workshop*的用武之地。由行业专家撰写和审查，它提供了一个快节奏、支持性的学习体验，可以让您迅速编写 C#代码并构建应用程序。与其他侧重于干燥、技术性解释基础理论的软件开发书籍不同，这个研讨会剔除了噪音，使用引人入胜的例子来帮助您了解每个概念在现实世界中的应用。

在阅读本书时，您将解决模拟软件开发人员每天处理的问题的真实练习。这些小项目包括构建一个猜数字游戏，使用发布者-订阅者模型设计 Web 文件下载器，使用 Razor Pages 创建待办事项列表，使用 async/await 任务从斐波那契序列生成图像，以及开发一个温度单位转换应用程序，然后将其部署到生产服务器上。

通过本书，您将具备知识、技能和信心，可以推动您的职业发展，并应对 C#的雄心勃勃的项目。

## 受众

本书适用于有志成为 C#开发人员的人。建议您在开始之前具备基本的核心编程概念知识。虽然不是绝对必要的，但有其他编程语言的经验会有所帮助。

# 关于作者

**Jason Hales**自 2001 年 C#首次发布以来，一直在使用各种微软技术开发低延迟、实时应用程序。他是设计模式、面向对象原则和测试驱动实践的热心倡导者。当他不忙于编码时，他喜欢和妻子 Ann 以及他们在英国剑桥郡的三个女儿一起度过时间。

**Almantas Karpavicius**是一名领先的软件工程师，就职于信息技术公司 TransUnion。他已经是一名专业程序员超过五年。除了全职编程工作外，Almantas 在[Twitch.tv](http://Twitch.tv)上利用业余时间免费教授编程已经三年。他是 C#编程社区 C# Inn 的创始人，拥有 7000 多名成员，并创建了两个免费的 C#训练营，帮助数百人开始他们的职业生涯。他曾与编程名人进行采访，如 Jon Skeet、Robert C. Martin（Uncle Bob）、Mark Seemann，还曾是兼职的 Java 教师。Almantas 喜欢谈论软件设计、清晰的代码和架构。他还对敏捷（特别是 Scrum）感兴趣，是自动化测试的忠实粉丝，尤其是使用 BDD 进行的测试。他还拥有两年的微软 MVP 资格（[`packt.link/2qUJp`](https://packt.link/2qUJp)）。

**Mateus Viegas**在软件工程和架构领域工作了十多年，最近几年致力于领导和管理工作。他在技术上的主要兴趣是 C#、分布式系统和产品开发。他热爱户外活动，工作之余喜欢和家人一起探索大自然、拍照或者跑步。

# 关于章节

*第一章*，*你好，C#*，介绍了语言的基本概念，如变量、常量、循环和算术和逻辑运算符。

*第二章*，*构建高质量的面向对象代码*，介绍了面向对象编程的基础知识和其四大支柱，然后介绍了清晰编码的五大主要原则——SOLID。本章还涵盖了 C#语言的最新特性。

*第三章*，*委托、事件和 Lambda*，介绍了委托和事件，它们构成了对象之间通信的核心机制，以及 lambda 语法，它提供了一种清晰表达代码意图的方式。

*第四章*，*数据结构和 LINQ*，涵盖了用于存储多个值的常见集合类，以及专为在内存中查询集合而设计的集成语言 LINQ。

*第五章*，*并发：多线程并行和异步代码*，介绍了编写高性能代码的基本知识，以及如何避免常见的陷阱和错误。

*第六章*，*使用 SQL Server 的 Entity Framework*，介绍了使用 SQL 和 C#进行数据库设计和存储，并深入研究了使用 Entity Framework 进行对象关系映射。本章还教授了与数据库一起工作的常见设计模式。

注

对于那些有兴趣学习数据库基础知识以及如何使用 PostgreSQL 的人，本书的 GitHub 存储库中包含了一个参考章节。你可以在以下链接访问：[`packt.link/oLQsL`](https://packt.link/oLQsL)。

*第七章*，*使用 ASP.NET 创建现代 Web 应用程序*，介绍了如何编写简单的 ASP.NET 应用程序，以及如何使用服务器端渲染和单页应用程序等方法来创建 Web 应用程序。

*第八章*，*创建和使用 Web API 客户端*，介绍了 API 并教你如何从 ASP.NET 代码访问和使用 Web API。

*第九章*，*创建 API 服务*，继续讨论 API 的主题，并教你如何为消费创建 API 服务，以及如何保护它。本章还向你介绍了微服务的概念。

注

此外，还有两个额外的章节（*第十章*，*自动化测试*，和*第十一章*，*生产就绪的 C#：从开发到部署*），你可以在以下链接找到：[`packt.link/44j2X`](https://packt.link/44j2X) 和 [`packt.link/39qQA`](https://packt.link/39qQA)。

你也可以在[`packt.link/qclbF`](https://packt.link/qclbF)的在线工作坊中找到所有活动的解决方案。

本书有一些约定用于高效安排内容。请在下一节中了解相关内容。

## 约定

### 代码块

在书中，代码块设置如下：

```cpp
using System;
namespace Exercise1_01
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
        }
    }
}
```

在输入和执行一些代码会立即输出的情况下，显示如下：

```cpp
dotnet run
Hello World!
Good morning Mars!
```

### 强调

定义、新术语和重要词汇显示如下：

多线程是一种并发形式，其中**多个**线程用于执行操作。

### 技术术语

章节正文中的语言命令以以下方式表示：

在这里，最简单的`Task`构造函数传递了一个`Action` lambda 语句，这是实际要执行的目标代码。目标代码将消息`Inside taskA`写入控制台。

### 附加信息

基本信息以以下方式表示：

注

术语`Factory`经常在软件开发中用来表示帮助创建对象的方法。

### 截断

长代码片段被截断，相应的代码文件名称放在截断代码的顶部。整个代码的永久链接放在代码片段下方，如下所示：

```cpp
HashSetExamples.cs
using System;
using System.Collections.Generic;
namespace Chapter04.Examples
{
}
You can find the complete code here: http://packt.link/ZdNbS.
```

在你深入学习 C#语言的强大之前，你需要安装.NET 运行时和 C#开发和调试工具。

## 开始之前

你可以安装完整的 Visual Studio 集成开发环境（IDE），它提供了一个功能齐全的代码编辑器（这是一个昂贵的许可证），或者你可以安装 Visual Studio Code（VS Code），微软的轻量级跨平台编辑器。*C#工作坊*以 VS Code 编辑器为目标，因为它不需要许可证费用，并且可以在多个平台上无缝运行。

# 安装 VS Code

访问网址 https://code.visualstudio.com 并根据您选择的平台的安装说明在 Windows、macOS 或 Linux 上下载它。

注意

最好勾选“创建桌面图标”复选框以方便使用。

VS Code 是免费且开源的。它支持多种语言，需要为 C#语言进行配置。安装 VS Code 后，您需要添加`C# for Visual Studio Code`（由 OmniSharp 提供支持）扩展以支持 C#。这可以在 https://marketplace.visualstudio.com/items?itemName=ms-dotnettools.csharp 找到。要安装 C#扩展，请按照每个平台的说明进行操作：

1.  打开“扩展”选项卡，输入`C#`。

注意

如果您不想直接从网站安装 C#扩展，可以从 VS Code 本身安装。

1.  选择第一个选择，即`C# for Visual Studio Code (powered by OmniSharp)`。

1.  点击“安装”按钮。

1.  重新启动`VS Code`：

![图 0.1：安装 VS Code 的 C#扩展](img/B16385_Preface_0.1.jpg)

图 0.1：安装 VS Code 的 C#扩展

您将看到 C#扩展成功安装在 VS Code 上。您现在已经在系统上安装了 VS Code。

下一节将介绍如何在您在书的章节之间移动时使用 VS Code。

## 在 VS Code 中移动章节

要更改默认要构建的项目（无论是活动、练习还是演示），您需要指向这些练习文件：

+   `tasks.json` / `tasks.args`

+   `launch.json` / `configurations.program`

有两种不同的练习模式需要注意。一些练习有自己的项目。其他练习有不同的主方法。每个练习的单个项目的主方法可以按照以下方式进行配置（在此示例中，对于*第三章*，*委托、事件和 Lambda*，您正在配置*练习 02*为构建和启动点）：

`launch.json`

```cpp
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": ".NET Core Launch (console)",
            "type": "coreclr",
            "request": "launch",
            "preLaunchTask": "build",
            "program": "${workspaceFolder}/Exercises/ /Exercise02/bin/Debug/net6.0/Exercise02.exe",
            "args": [],
            "cwd": "${workspaceFolder}",
            "stopAtEntry": false,
            "console": "internalConsole"
        }

    ]
}
```

`tasks.json`

```cpp
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "build",
            "command": "dotnet",
            "type": "process",
            "args": [
                "build",
                "${workspaceFolder}/Chapter05.csproj",
                "/property:GenerateFullPaths=true",
                "/consoleloggerparameters:NoSummary"
            ],
            "problemMatcher": "$msCompile"
        },

    ]
}
```

每个练习（例如，“第五章 练习 02”）都可以按照以下方式进行配置：

`launch.json`

```cpp
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": ".NET Core Launch (console)",
            "type": "coreclr",
            "request": "launch",
            "preLaunchTask": "build",
            "program": "${workspaceFolder}/bin/Debug/net6.0/Chapter05.exe",
            "args": [],
            "cwd": "${workspaceFolder}",
            "stopAtEntry": false,
            "console": "internalConsole"
        }

    ]
}
```

`tasks.json`

```cpp
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "build",
            "command": "dotnet",
            "type": "process",
            "args": [
              "build",
              "${workspaceFolder}/Chapter05.csproj",
              "/property:GenerateFullPaths=true",
              "/consoleloggerparameters:NoSummary",
              "-p:StartupObject=Chapter05.Exercises.Exercise02.Program",
            ],
            "problemMatcher": "$msCompile"
        },

    ]
}
```

现在您已经了解了`launch.json`和`tasks.json`，可以继续下一节，详细介绍.NET 开发平台的安装。

## 安装.NET 开发平台

.NET 开发平台可以从[`dotnet.microsoft.com/download`](https://dotnet.microsoft.com/download)下载。Windows、macOS 和 Linux 上都有不同的变体。*C# Workshop*书籍使用.NET 6.0。

按照以下步骤在 Windows 上安装.NET 6.0 平台：

1.  选择`Windows`平台选项卡：

![图 0.2：.NET 6.0 下载窗口](img/B16385_Preface_0.2.jpg)

图 0.2：.NET 6.0 下载窗口

1.  点击“下载.NET SDK x64”选项。

注意

*图 0.2*中显示的屏幕可能会根据 Microsoft 的最新发布而更改。

1.  根据系统上安装的操作系统打开并完成安装。

1.  安装后重新启动计算机。

按照以下步骤在 macOS 上安装.NET 6.0 平台：

1.  选择`macOS`平台选项卡（*图 0.2*）。

1.  点击“下载.NET SDK x64”选项。

下载完成后，打开安装程序文件。您应该有一个类似*图 0.3*的屏幕：

![图 0.3：macOS 安装开始屏幕](img/B16385_Preface_0.3.jpg)

图 0.3：macOS 安装开始屏幕

1.  点击“继续”按钮。

以下屏幕将确认安装所需的空间量：

1.  点击“安装”按钮继续：

![图 0.4：显示安装所需磁盘空间的窗口](img/B16385_Preface_0.4.jpg)

图 0.4：显示安装所需磁盘空间的窗口

您将在下一个屏幕上看到一个移动的进度条：

![图 0.5：显示安装进度的窗口](img/B16385_Preface_0.5.jpg)

图 0.5：显示安装进度的窗口

安装完成后不久，您将看到一个成功的屏幕（*图 0.6*）：

![图 0.6：显示安装完成的窗口](img/B16385_Preface_0.6.jpg)

图 0.6：显示安装完成的窗口

1.  为了检查安装是否成功，请打开您的终端应用程序并键入：

```cpp
     dotnet –list-sdks 
    ```

这将检查您计算机上安装的.NET 版本。*图 0.7*显示了您安装的 SDK 的列表：

![图 0.7：在终端中检查安装的.NET SDK](img/B16385_Preface_0.7.jpg)

图 0.7：在终端中检查安装的.NET SDK

通过这些步骤，您可以在计算机上安装.NET 6.0 SDK 并检查已安装的版本。

注意

Linux 的.NET 6.0 安装步骤未包括在内，因为它们与 Windows 和 macOS 相似。

在继续之前，了解.NET 6.0 的功能很重要。

## .NET 6.0 在 Windows、macOS 和 Linux 中的功能

### Windows

+   .NET 6.0：这是 Windows 推荐的最新长期支持（LTS）版本。它可用于构建许多不同类型的应用程序。

+   .NET Framework 4.8：这是仅适用于 Windows 的版本，用于构建仅在 Windows 上运行的任何类型的应用程序。

### macOS

+   .NET 6.0：这是 macOS 推荐的 LTS 版本。它可用于构建许多不同类型的应用程序。选择与您的 Apple 计算机处理器兼容的版本——Intel 芯片为 x64，Apple 芯片为 ARM64。

### Linux

+   .NET 6.0：这是 Linux 推荐的 LTS 版本。它可用于构建许多不同类型的应用程序。

### Docker

+   .NET 图像：此开发平台可用于构建不同类型的应用程序。

+   .NET Core 图像：这为构建许多类型的应用程序提供了终身支持。

+   .NET 框架图像：这些是仅适用于 Windows 的.NET 版本，用于构建仅在 Windows 上运行的任何类型的应用程序。

在系统上安装了.NET 6.0 后，下一步是使用 CLI 配置项目。

# .NET 命令行界面（CLI）

安装了.NET 后，CLI 可用于创建和配置用于 VS Code 的项目。要启动.NET CLI，请在命令提示符下运行以下命令：

```cpp
dotnet
```

如果.NET 安装正确，您将在屏幕上看到以下消息：

```cpp
Usage: dotnet [options]
Usage: dotnet [path-to-application]
```

安装了 CLI 以配置 VS Code 项目后，您需要了解使用和扩展 SQL 语言的功能强大的开源对象关系数据库系统 PostgreSQL。

注意

您将首先按照 Windows 的说明安装 PostgreSQL，然后是 macOS，然后是 Linux。

## Windows 的 PostgreSQL 安装

在*第六章*中使用了 PostgreSQL，*使用 SQL Server 的 Entity Framework*。在继续进行该章之前，您必须按照以下步骤在系统上安装 PostgreSQL：

1.  转到[`www.enterprisedb.com/downloads/postgres-postgresql-downloads`](https://www.enterprisedb.com/downloads/postgres-postgresql-downloads)并下载 Windows 的最新版本安装程序：

![图 0.8：每个平台的最新 PostgreSQL 版本](img/B16385_Preface_0.8.jpg)

图 0.8：每个平台的最新 PostgreSQL 版本

注意

*图 0.8*中显示的屏幕可能会根据供应商的最新发布而更改。

1.  打开下载的交互式安装程序，然后单击“下一步”按钮。将显示“设置 PostgreSQL”屏幕：

![图 0.9：用于上传 PostgreSQL 的欢迎屏幕](img/B16385_Preface_0.9.jpg)

图 0.9：用于上传 PostgreSQL 的欢迎屏幕

1.  单击“下一步”按钮，转到下一个屏幕，要求提供安装目录详细信息：

![图 0.10：PostgreSQL 默认安装目录](img/B16385_Preface_0.10.jpg)

图 0.10：PostgreSQL 默认安装目录

1.  保持默认的“安装目录”不变，然后单击“下一步”按钮。

1.  从*图 0.11*的列表中选择以下内容：

+   `PostgreSQL 服务器`指的是数据库。

+   `pgAdmin 4`是数据库管理工具。

+   `Stack Builder`是 PostgreSQL 环境构建器（可选）。

+   `命令行工具`使用命令行与数据库一起工作。

![图 0.11：选择要继续的 PostgreSQL 组件](img/B16385_Preface_0.11.jpg)

图 0.11：选择要继续的 PostgreSQL 组件

1.  然后点击“下一步”按钮。

1.  在下一个屏幕上，“数据目录”屏幕要求您输入用于存储数据的目录。因此，输入数据目录名称：

![图 0.12：存储数据的目录](img/B16385_Preface_0.12.jpg)

图 0.12：存储数据的目录

1.  一旦输入了数据目录，点击“下一步”按钮继续。下一个屏幕要求您输入密码。

1.  输入新密码。

1.  在“重新输入密码”旁边重新输入数据库超级用户的密码：

![图 0.13：为数据库超级用户提供密码](img/B16385_Preface_0.13.jpg)

图 0.13：为数据库超级用户提供密码

1.  然后点击“下一步”按钮继续。

1.  下一个屏幕显示端口为`5432`。使用默认端口，即`5432`：

![图 0.14：选择端口](img/B16385_Preface_0.14.jpg)

图 0.14：选择端口

1.  点击“下一步”按钮。

1.  “高级选项”屏幕要求您输入数据库集群的区域设置。将其保留为“[默认区域设置]”：

![图 0.15：选择数据库集群的区域设置](img/B16385_Preface_0.15.jpg)

图 0.15：选择数据库集群的区域设置

1.  然后点击“下一步”按钮。

1.  当显示“预安装摘要”屏幕时，点击“下一步”按钮继续：

![图 0.16：设置窗口显示准备安装消息](img/B16385_Preface_0.16.jpg)

图 0.16：设置窗口显示准备安装消息

1.  继续选择“下一步”按钮（保持默认设置不变），直到安装过程开始。

1.  等待完成。完成后，将显示“完成 PostgreSQL 安装向导”屏幕。

1.  取消选中“退出时启动堆栈生成器”选项：

![图 0.17：安装完成，未选中堆栈生成器](img/B16385_Preface_0.17.jpg)

图 0.17：安装完成，未选中堆栈生成器

堆栈生成器用于下载和安装其他工具。默认安装包含所有练习和活动所需的所有工具。

1.  最后，点击“完成”按钮。

1.  现在从 Windows 打开`pgAdmin4`。

1.  在“设置主密码”窗口中为连接到 PostgreSQL 中的任何数据库输入主密码：

![图 0.18：设置连接到 PostgreSQL 服务器的主密码](img/B16385_Preface_0.18.jpg)

图 0.18：设置连接到 PostgreSQL 服务器的主密码

注意

最好输入一个你能轻松记住的密码，因为它将用于管理所有其他凭据。

1.  接下来点击“确定”按钮。

1.  在 pgadmin 窗口的左侧，通过单击旁边的箭头展开“服务器”。

1.  您将被要求输入您的 PostgreSQL 服务器密码。输入与*步骤 22*中输入的相同的密码。

1.  出于安全原因，请勿点击“保存密码”：

![图 0.19：为 PostgreSQL 服务器设置 postgres 用户密码](img/B16385_Preface_0.19.jpg)

图 0.19：为 PostgreSQL 服务器设置 postgres 用户密码

PostgreSQL 服务器密码是连接到 PostgreSQL 服务器并使用`postgres`用户时使用的密码。

1.  最后点击“确定”按钮。您将看到 pgAdmin 仪表板：

![图 0.20：pgAdmin 4 仪表板窗口](img/B16385_Preface_0.20.jpg)

图 0.20：pgAdmin 4 仪表板窗口

要探索 pgAdmin 仪表板，请转到*探索 pgAdmin 仪表板*部分。

## macOS 上的 PostgreSQL 安装

按照以下步骤在 macOS 上安装 PostgreSQL：

1.  访问 Postgres 应用的官方网站，在 mac 平台上下载并安装 PostgreSQL：[`www.enterprisedb.com/downloads/postgres-postgresql-downloads`](https://www.enterprisedb.com/downloads/postgres-postgresql-downloads)。

1.  在 macOS 上下载最新的 PostgreSQL：

注意

以下屏幕截图是在 macOS Monterey（版本 12.2）上的版本 14.4 拍摄的。

![图 0.21：PostgreSQL 的安装页面](img/B16385_Preface_0.21.jpg)

图 0.21：PostgreSQL 的安装页面

1.  下载 macOS 的安装程序文件后，双击“安装程序文件”以启动 PostgreSQL 设置向导：

![图 0.22：启动 PostgreSQL 设置向导](img/B16385_Preface_0.22.jpg)

图 0.22：启动 PostgreSQL 设置向导

1.  选择要安装 PostgreSQL 的位置：

![图 0.23：选择安装目录](img/B16385_Preface_0.23.jpg)

图 0.23：选择安装目录

1.  单击“下一步”按钮。

1.  在下一个屏幕上，选择以下组件进行安装：

+   PostgreSQL 服务器

+   `pgAdmin 4`

+   命令行工具

1.  取消`Stack Builder`组件的选择：

![图 0.24：选择要安装的组件](img/B16385_Preface_0.24.jpg)

图 0.24：选择要安装的组件

1.  选择完选项后，单击“下一步”按钮。

1.  指定 PostgreSQL 将存储数据的数据目录：

![图 0.25：指定数据目录](img/B16385_Preface_0.25.jpg)

图 0.25：指定数据目录

1.  单击“下一步”按钮。

1.  现在为 Postgres 数据库超级用户设置“密码”：

![图 0.26：设置密码](img/B16385_Preface_0.26.jpg)

图 0.26：设置密码

确保安全地记下密码，以便登录到 PostgreSQL 数据库。

1.  单击“下一步”按钮。

设置要运行 PostgreSQL 服务器的端口号。这里将默认端口号设置为`5432`：

![图 0.27：指定端口号](img/B16385_Preface_0.27.jpg)

图 0.27：指定端口号

1.  单击“下一步”按钮。

1.  选择要由 PostgreSQL 使用的区域设置。在这里，“[默认区域设置]”是为 macOS 选择的区域设置：

![图 0.28：选择区域设置](img/B16385_Preface_0.28.jpg)

图 0.28：选择区域设置

1.  单击“下一步”按钮。

1.  在下一个屏幕上，检查安装详细信息：

![图 0.29：预安装摘要页面](img/B16385_Preface_0.29.jpg)

图 0.29：预安装摘要页面

最后，单击“下一步”按钮开始在您的系统上安装 PostgreSQL 数据库服务器的安装过程：

![图 0.30：在开始安装过程之前准备安装页面](img/B16385_Preface_0.30.jpg)

图 0.30：在开始安装过程之前准备安装页面

1.  等待一会儿，直到安装过程完成：

![图 0.31：安装设置正在进行中](img/B16385_Preface_0.31.jpg)

图 0.31：安装设置正在进行中

1.  在提示时，单击“下一步”按钮。下一个屏幕会显示消息，即 PostgreSQL 已在您的系统上安装完成：

![图 0.32：显示设置完成的成功消息](img/B16385_Preface_0.32.jpg)

图 0.32：显示设置完成的成功消息

1.  安装完成后，单击“完成”按钮。

1.  现在在 PostgreSQL 服务器中加载数据库。

1.  双击`pgAdmin 4`图标，从启动台启动它。

1.  输入在安装过程中设置的 PostgreSQL 用户的密码。

1.  然后单击“确定”按钮。现在您将看到 pgAdmin 仪表板。

这样就完成了在 macOS 上安装 PostgreSQL。下一节将使您熟悉 PostgreSQL 界面。

### 探索 pgAdmin 仪表板

在 Windows 和 macOS 上安装 PostgreSQL 后，按照以下步骤更好地了解界面：

1.  从 Windows/macOS 打开`pgAdmin4`（如果您的系统上没有打开 pgAdmin）。

1.  在左侧单击“服务器”选项：

![图 0.33：单击“服务器”以创建数据库](img/B16385_Preface_0.33.jpg)

图 0.33：单击“服务器”以创建数据库

1.  右键单击`PostgreSQL 14`。

1.  然后单击“创建”选项。

1.  选择`数据库…`选项以创建新数据库：

![图 0.34：创建新数据库](img/B16385_Preface_0.34.jpg)

图 0.34：创建新数据库

这将打开一个创建-数据库窗口。

1.  输入数据库名称，如`TestDatabase`。

1.  选择数据库的所有者或将其保留为默认值。现在，只需将`Owner`设置为`postgres`：

![图 0.35：选择数据库的所有者](img/B16385_Preface_0.35.jpg)

图 0.35：选择数据库的所有者

1.  然后点击`保存`按钮。这将创建一个数据库。

1.  右键单击`数据库`，然后选择`刷新`按钮：

![图 0.36：右键单击数据库后点击刷新按钮](img/B16385_Preface_0.36.jpg)

图 0.36：右键单击数据库后点击刷新按钮

现在在仪表板中显示了名为`TestDatabase`的数据库：

![图 0.37：TestDatabase 准备就绪](img/B16385_Preface_0.37.jpg)

图 0.37：TestDatabase 准备就绪

现在您的数据库已准备好在 Windows 和 Mac 环境中使用。

## Ubuntu 上的 PostgreSQL 安装

在此示例中，您正在使用 Ubuntu 20.04 进行安装。执行以下步骤：

1.  要安装 PostgreSQL，请首先打开您的 Ubuntu 终端。

1.  请确保使用以下命令更新您的存储库：

```cpp
    $ sudo apt update
    ```

1.  使用以下命令安装 PostgreSQL 软件以及额外的包（推荐）：

```cpp
    $ sudo apt install postgresql postgresql-contrib
    ```

注意

要仅安装 PostgreSQL（不建议没有额外包），请使用命令`$ sudo apt install postgresql`，然后按`Enter`键。

此安装过程创建了一个名为`postgres`的用户账户，该账户具有默认的`Postgres`角色。

### 使用 postgres 角色访问 postgres 用户账户

有两种方法可以使用`postgres`用户账户启动 PostgreSQL CLI：

选项 1 如下：

1.  要以 postgres 用户身份登录，请使用以下命令：

```cpp
    $ sudo -i -u postgres
    ```

1.  使用以下命令访问 CLI：

```cpp
    $ psql
    ```

注意

有时，在执行上述命令时，可能会显示`psql`错误，如`无法连接到服务器：没有这样的文件或目录`。这是由于系统上的端口问题。由于此端口阻塞，PostgreSQL 应用程序可能无法工作。您可以稍后再次尝试该命令。

1.  要退出 CLI，请使用以下命令：

```cpp
    $ \q
    ```

选项 2 如下：

1.  要以 postgres 用户身份登录，请使用以下命令：

```cpp
    $ sudo -u postgres psql
    ```

1.  要退出 CLI，请使用以下命令：

```cpp
    $ \q
    ```

### 验证 postgres 用户账户作为 postgres 用户角色

1.  要验证用户账户，请登录并使用`conninfo`命令：

```cpp
    $ sudo -u postgres psql
    $ \conninfo
    $ \q
    ```

使用此命令，您可以确保以端口`5432`连接到`postgres`数据库，作为`postgres`用户。如果您不想使用默认用户`postgres`，可以为自己创建一个新用户。

### 访问新用户和数据库

1.  使用以下命令并按`Enter`键创建一个新用户：

```cpp
    $ sudo -u postgres createuser –interactive
    ```

上述命令将要求用户添加角色的名称及其类型。

1.  输入角色的名称，例如`testUser`。

1.  然后，在提示时输入`y`以设置新角色为超级用户：

```cpp
    Prompt:
    Enter the name of the role to add: testUser
    Shall the new role be a superuser? (y/n) y
    ```

这将创建一个名为`testUser`的新用户。

1.  使用以下命令创建名为`testdb`的新数据库：

```cpp
    $ sudo -u postgres createdb testdb
    ```

1.  使用以下命令登录到新创建的用户账户：

```cpp
    $ sudo -u testUser psql -d testdb
    ```

1.  使用以下命令检查连接详细信息：

```cpp
    $ \conninfo
    ```

1.  要退出 CLI，请使用以下命令：

```cpp
    $ \q
    ```

使用此命令，您可以确保以端口`5432`连接到`testdb`数据库，作为`testUser`用户。

通过这些步骤，您已经完成了 Ubuntu 上的 PostgreSQL 安装。

# 下载代码

从 GitHub 下载代码[`packt.link/sezEm`](https://packt.link/sezEm)。参考这些文件获取完整的代码。

本书中使用的高质量彩色图片可以在[`packt.link/5XYmX`](https://packt.link/5XYmX)找到。

如果您在安装过程中遇到任何问题或有任何问题，请发送电子邮件至`workshops@packt.com`。
