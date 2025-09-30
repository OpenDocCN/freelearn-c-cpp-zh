# 前言

当我在 2008 年对 StyleCop 的 4.2 版本产生兴趣时，该工具在互联网上受到了严重的批评。这些反应的根源在于以下几点：

+   该工具不是开源的

+   实施的规则是微软设定的任意规则，并不被某些人所喜欢

+   它们与.NET 运行时如何解释我们的代码没有任何关系

+   一些微软之前制作的工具与 StyleCop 制定的规则直接相矛盾

如果我们今天来看，那时的所有规则仍然存在，StyleCop 最终已被广泛接受。这种接受的部分原因无疑是微软在 4.3.3 版本中将 StyleCop 的源代码发布给社区，但这不是唯一的原因。

如果我们回顾我们在中等规模和大项目上开始开发的过程，我们首先做的事情之一就是确立基本原则，其中之一就是编码规范的定义。这些规则，即我们的代码应该如何看起来，是为了提高所有团队成员的代码可读性和可维护性而存在的。那里的选择相当随意，取决于制定它们的人（或开发团队）的背景和偏好。

然而，在项目开始后，需要花费大量时间和代码审查来遵循它们。

这就是 StyleCop 变得方便的地方。无论制定的规则是否符合微软的规则集，还是团队必须从头开始制定自己的规则，一旦参数化，该工具就可以根据命令审查项目的代码，甚至可以在持续集成中使用以强制执行之前定义的规则集。

# 本书涵盖的内容

*使用 Visual Studio 安装 StyleCop（简单）* 介绍了 StyleCop 的安装过程，并教授如何配置要在项目中执行的规定，以及如何从 Visual Studio 启动分析。

*理解 ReSharper 插件（简单）* 介绍了 ReSharper 的 StyleCop 插件。我们将看到其实时分析和如何轻松修复大多数 StyleCop 违规。

*使用 MSBuild 自动化 StyleCop（简单）* 介绍了如何使用 MSBuild 自动化我们的构建过程。我们将描述需要添加到 MSBuild 项目中的哪些行才能启用 StyleCop 对其的分析，以及如何在构建中断之前限制遇到的违规数量。

*使用命令行批处理自动化 StyleCop（简单）* 介绍了如何从命令行使用 StyleCop 分析您的项目。为此，我们将使用一个名为 StyleCopCmd 的工具，并准备它以能够启动 StyleCop 的最新版本。

*使用 NAnt 自动化 StyleCop（中级）* 介绍了如何使用 StyleCopCmd 通过 NAnt 自动化我们的过程。

*在 Jenkins/Hudson 中集成 StyleCop 分析结果（中级）* 介绍了如何为项目构建一个 StyleCop 分析作业并显示其错误。

*自定义文件头（简单）* 说明了如何自定义文件头以避免 StyleCop 违规，以及我们如何使用 Visual Studio 模板和代码片段来使我们的开发生活更轻松。

*创建自定义规则（中级）* 说明了如何为 StyleCop 引擎创建我们自己的自定义规则。我们还将看到如何向此规则添加参数。

*在您的工具中集成 StyleCop（高级）* 将向我们展示如何将 StyleCop 集成到您的工具中。作为一个例子，我们将创建一个用于 MonoDevelop/Xamarin Studio 的实时分析插件。

# 您需要为此书准备的东西

StyleCop 是一个 C# 代码分析器；它可以与 Visual Studio 或不与它一起使用。

本书涵盖了使用 StyleCop 与 Visual Studio 一起使用以及不使用 Visual Studio 的情况。为了跟随本书的不同章节，您需要安装以下软件：

+   Visual Studio 2008 专业版或更高版本

+   Jenkins

+   Xamarin Studio 或 MonoDevelop 4.0

# 本书面向对象

本书旨在为希望发现 StyleCop 的 .Net 开发者提供帮助。

# 惯例

在这本书中，您会发现许多不同风格的文本，用以区分不同类型的信息。以下是一些这些风格的示例及其含义的解释。

文本中的代码单词如下所示：“也可以包含 `Stylecop.targets` 文件。”

代码块如下设置：

```cpp
    <Configuration Condition=" '$(Configuration)' == '' ">
        Debug
    </Configuration>
    <Platform Condition=" '$(Platform)' == '' ">
        AnyCPU
    </Platform>
    <ProductVersion>8.0.50727</ProductVersion>
    <SchemaVersion>2.0</SchemaVersion>
```

当我们希望您注意代码块中的特定部分时，相关的行或项目将以粗体显示：

```cpp
<?xml version="1.0" encoding="utf-8" ?>
<Project DefaultTargets="StyleCop" >
 <UsingTask TaskName="StyleCopTask" AssemblyFile="$(MSBuildExtensionsPath)\..\StyleCop 4.7\StyleCop.dll" />
  <PropertyGroup>
    <!-- Set a default value of 1000000 as maximum Stylecop violations found -->
```

任何命令行输入或输出都如下所示：

```cpp
NAnt 0.92 (Build 0.92.4543.0; release; 09/06/2012)
Copyright (C) 2001-2012 Gerry Shaw
http://nant.sourceforge.net

```

**新术语** 和 **重要词汇** 以粗体显示。您在屏幕上看到的单词，例如在菜单或对话框中，在文本中如下所示：“点击 **下一步** 按钮将您带到下一屏幕”。

### 注意

警告或重要注意事项如下所示。

### 小贴士

小贴士和技巧看起来像这样。

# 读者反馈

我们读者的反馈总是受欢迎的。请告诉我们您对这本书的看法——您喜欢什么或可能不喜欢什么。读者反馈对我们开发您真正从中受益的标题非常重要。

要向我们发送一般反馈，请简单地将电子邮件发送到 `<feedback@packtpub.com>`，并在邮件主题中提及书籍标题。

如果您需要一本书并且希望我们出版，请通过 [www.packtpub.com](http://www.packtpub.com) 上的 **SUGGEST A TITLE** 表格或发送电子邮件到 `<suggest@packtpub.com>》给我们留言。

如果您在某个主题上具有专业知识，并且您对撰写或为书籍做出贡献感兴趣，请参阅我们关于 [www.packtpub.com/authors](http://www.packtpub.com/authors) 的作者指南。

# 客户支持

现在，您是 Packt 书籍的骄傲拥有者，我们有一些事情可以帮助您从您的购买中获得最大收益。

## 下载示例代码

您可以从[`www.PacktPub.com`](http://www.PacktPub.com)的账户下载您购买的所有 Packt 书籍的示例代码文件。如果您在其他地方购买了这本书，您可以访问[`www.PacktPub.com/support`](http://www.PacktPub.com/support)，并注册以直接将文件通过电子邮件发送给您。

## 勘误

尽管我们已经尽一切努力确保内容的准确性，但错误仍然可能发生。如果您在我们的书中发现错误——可能是文本或代码中的错误——如果您能向我们报告这一点，我们将不胜感激。通过这样做，您可以节省其他读者的挫败感，并帮助我们改进本书的后续版本。如果您发现任何勘误，请通过访问[`www.packtpub.com/support`](http://www.packtpub.com/support)，选择您的书籍，点击**勘误提交表单**链接，并输入您的勘误详情来报告它们。一旦您的勘误得到验证，您的提交将被接受，勘误将被上传到我们的网站，或添加到该标题的勘误部分下的现有勘误列表中。您可以通过从[`www.packtpub.com/support`](http://www.packtpub.com/support)选择您的标题来查看任何现有勘误。

## 盗版

互联网上版权材料的盗版是所有媒体中持续存在的问题。在 Packt，我们非常重视我们版权和许可证的保护。如果您在互联网上遇到任何我们作品的非法副本，无论形式如何，请立即提供位置地址或网站名称，以便我们可以寻求补救措施。

请通过发送链接到疑似盗版材料至`<copyright@packtpub.com>`与我们联系。

我们感谢您在保护我们的作者以及为我们提供有价值内容的能力方面的帮助。

## 询问

如果您在本书的任何方面遇到问题，可以通过`<questions@packtpub.com>`与我们联系，我们将尽力解决。
