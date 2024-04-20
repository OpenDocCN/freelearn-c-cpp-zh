# 前言

Qt 是一个成熟而强大的框架，可在多种平台上交付复杂的应用程序。它在嵌入式设备中被广泛使用，包括电视、卫星机顶盒、医疗设备、汽车仪表板等。它在 Linux 世界中也有丰富的历史，KDE 和 Sailfish OS 广泛使用它，许多应用程序也是使用 Qt 开发的。在过去几年中，它在移动领域也取得了巨大进展。然而，在 Microsoft Windows 和 Apple macOS X 世界中，C#/.NET 和 Objective-C/Cocoa 的主导地位意味着 Qt 经常被忽视。

本书旨在展示 Qt 框架的强大和灵活性，并展示如何编写应用程序一次并将其部署到多个操作系统的桌面。读者将从头开始构建一个完整的现实世界**业务线**（**LOB**）解决方案，包括独立的库、用户界面和单元测试项目。

我们将使用 QML 构建现代和响应式的用户界面，并将其连接到丰富的 C++类。我们将使用 QMake 控制项目配置和输出的每个方面，包括平台检测和条件表达式。我们将构建“自我意识”的数据实体，它们可以将自己序列化到 JSON 并从中反序列化。我们将在数据库中持久化这些数据实体，并学习如何查找和更新它们。我们将访问互联网并消费 RSS 源。最后，我们将生成一个安装包，以便将我们的应用部署到其他机器上。

这是一套涵盖大多数 LOB 应用程序核心要求的基本技术，将使读者能够从空白页面到已部署应用程序的进程。

# 本书的受众

本书面向寻找在 Microsoft Windows、Apple Mac OS X 和 Linux 桌面平台上创建现代和响应式应用程序的强大而灵活的框架的应用程序开发人员。虽然专注于桌面应用程序开发，但所讨论的技术在移动开发中也大多适用。

# 充分利用本书

读者应该熟悉 C++，但不需要先前了解 Qt 或 QML。在 Mac OS X 上，您需要安装 XCode 并至少启动一次。在 Windows 上，您可以选择安装 Visual Studio 以便使用 MSVC 编译器。

# 下载示例代码文件

您可以从[www.packtpub.com](http://www.packtpub.com)的帐户中下载本书的示例代码文件。如果您在其他地方购买了本书，您可以访问[www.packtpub.com/support](http://www.packtpub.com/support)并注册，以便直接通过电子邮件接收文件。

您可以按照以下步骤下载代码文件：

1.  在[www.packtpub.com](http://www.packtpub.com/support)登录或注册。

1.  选择“支持”选项卡。

1.  单击“代码下载和勘误”。

1.  在搜索框中输入书名，然后按照屏幕上的说明操作。

下载文件后，请确保使用最新版本的以下工具解压或提取文件夹：

+   Windows 需要 WinRAR/7-Zip

+   Mac 需要 Zipeg/iZip/UnRarX

+   Linux 需要 7-Zip/PeaZip

本书的代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Learn-Qt-5`](https://github.com/PacktPublishing/Learn-Qt-5)。我们还有其他书籍和视频的代码包可供下载，网址为**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**。请查看！

# 使用的约定

本书中使用了许多文本约定。

`CodeInText`：指示文本中的代码词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 句柄。这是一个例子：“在`cm-ui/ui/views`中创建`SplashView.qml`文件”。

代码块设置如下：

```cpp
<RCC>
    <qresource prefix="/views">
        <file alias="MasterView">views/MasterView.qml</file>
    </qresource>
    <qresource prefix="/">
        <file>views/SplashView.qml</file>
        <file>views/DashboardView.qml</file>
        <file>views/CreateClientView.qml</file>
        <file>views/EditClientView.qml</file>
        <file>views/FindClientView.qml</file>
    </qresource>
</RCC>
```

当我们希望引起您对代码块的特定部分的注意时，相关行或项目将以粗体显示：

```cpp
QT += sql network
```

任何命令行输入或输出都以以下方式书写：

```cpp
$ <Qt Installation Path> \Tools \QtInstallerFramework \3.0\ bin\ binarycreator.exe -c config\config.xml -p packages ClientManagementInstaller.exe
```

**粗体**：表示一个新术语，一个重要词，或者你在屏幕上看到的词。例如，菜单或对话框中的单词会以这种方式出现在文本中。这是一个例子：“用 Client Management 替换 Hello World 标题，并在 Window 的正文中插入一个 Text 组件”。

警告或重要说明会出现在这样的地方。

提示和技巧会出现在这样的地方。