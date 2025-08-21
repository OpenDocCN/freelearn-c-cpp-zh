# 前言

Windows 8 是微软最新的客户端操作系统。一方面，它延续了 Windows 7 的趋势，建立了一个稳定、强大和现代的操作系统。另一方面，它改变了许多从以前的 Windows 版本中学到的假设和习惯。

任务栏中的常见“开始”按钮已经消失，用户登录后不再首先看到桌面。一个新的开始屏幕等待着毫无准备的用户，上面充满了定期更改其内容的“动态磁贴”。经典的开始菜单已经不复存在；有趣的是，桌面可以在开始屏幕的磁贴中找到。

Windows 8 的新外观显然是针对平板设备的——最近几个月出现了许多型号。新的用户界面在基于触摸的设备上是有意义的，但传统的鼠标和键盘设置在笔记本电脑或台式机上仍然可以按预期工作。

随着这个新的 Windows，也带来了一个新的运行时，一种新类型的应用程序运行在上面——Windows Runtime。基于这个新的运行时，应用程序可以构建并上传到 Windows Store——一个存储经过认证的应用程序的库，将它们标识为安全和有用。事实上，普通用户只能通过 Windows Store 获取这些新应用程序——Windows Store 应用程序，而不是传统的安装方式，如安装程序或 MSI 文件。

经典应用程序，现在被称为桌面应用程序，仍然可以以通常的方式使用现有技术在本机空间（Win32、COM、ATL、MFC、WTL 等）或托管空间（WinForms、WPF、WCF、EF 等）中编写，并且在 Windows 8 上运行方式与在 Windows 7 上一样——也许更好，因为 Windows 内核的改进。

新的 Windows Store 应用程序只能在 Windows 8（及更高版本）操作系统上运行；它们需要基于**组件对象模型**（**COM**）技术的 Windows Runtime。这些应用在几个方面在视觉上看起来不同：它们总是全屏显示（除了特殊的“分屏视图”），没有边框，使用了新的 UI 设计方案，现在称为现代 UI，是面向触摸的，并具有一些其他不太明显的特性。

这本书主要讲述了新的 Windows Store 应用程序。从它们是什么开始，我们将逐步介绍 Windows Runtime 的各个方面，重点是使用 C++和新的扩展（C++/CX）来利用这个新的运行时，编写可以上传到商店并与运行 Windows 8 的任何人共享的应用程序。

# 本书内容

第一章 *介绍 Windows 8 应用程序*，从 Windows Store 应用程序的角度介绍了 Windows 8 操作系统，并讨论了围绕 Windows Store 应用程序和 Windows Runtime 的一些概念。

第二章 *Windows 8 商店应用的 COM 和 C++*，介绍了 C++ 11 的重要特性和新的语言扩展 C++/CX，它们允许更容易地访问 Windows Runtime 类型。本章还讨论了其他经典技术以及它们在 Windows Store 应用程序模型中的适用性（如果有的话）。

第三章 *使用 XAML 构建 UI*，展示了如何使用声明性的 XAML 语言和语义为 Windows Store 应用程序构建用户界面。详细解释了资源的概念，以及它们如何适用于 WinRT。

第四章 *布局、元素和控件*，讨论了控件的布局方式，以构建灵活的用户界面。讨论了 Windows Runtime 提供的许多元素，特别关注具有特定特征的控件组。

第五章，“数据绑定”，讨论了允许控件和数据之间无缝集成的最强大的 WinRT 功能之一。介绍了流行的**Model-View-ViewModel**（**MVVM**）模式，并提供了可能实现的示例。

第六章，“组件，模板和自定义元素”，展示了如何创建可供其他语言使用的可重用 WinRT 组件，而不仅仅是 C++。讨论了控件模板，允许完全更改控件的外观而不影响其行为。最后，本章演示了如何创建自定义控件，当需要一些现有行为但内置控件中不可用时。

第七章，“应用程序，磁贴，任务和通知”，探讨了 Windows Store 应用程序的一些特殊功能，如动态磁贴以及它们可以从本地和服务器更新的方式。讨论了后台任务，允许代码在应用程序不在前台时执行。本章还展示了如何利用设备锁屏，如何进行长时间数据传输以及播放背景音乐。

第八章，“合同和扩展”，展示了 Windows Store 应用程序如何通过实现 Windows 定义的合同和扩展与 Windows 更好地集成并与其他应用程序通信。

第九章，“打包和 Windows 商店”，介绍了将应用程序打包，测试和部署到 Windows 商店的过程，并详细说明了一些需要注意的事项，以便成功获得认证。

# 本书所需内容

要使用本书中的示例，您需要在运行 Windows 8（任何版本）上安装 Visual Studio 2012 或更高版本（包括 Express 版本）。

# 本书的受众

本书面向想要利用其现有技能创建 Windows Store 应用程序的 C++开发人员。不需要了解旧技术，如 Win32 或 MFC；熟悉 COM 是有益的，但不是必需的。

# 约定

在本书中，您将找到一些文本样式，用于区分不同类型的信息。以下是一些这些样式的示例，以及它们的含义解释。

文本中的代码字，数据库表名，文件夹名，文件名，文件扩展名，路径名，虚拟 URL，用户输入和 Twitter 句柄显示如下：“XAML 显示了一个`Page`根元素，带有几个属性和一个内部的`Grid`元素。”

代码块设置如下：

```cpp
<StackPanel Orientation="Horizontal" Margin="20" VerticalAlignment="Center">
    <TextBox Width="150" Margin="10" x:Name="_number1" FontSize="30" Text="0" TextAlignment="Right"/>
    <TextBlock Text="+" Margin="10" FontSize="30" VerticalAlignment="Center"/>
    <TextBox Width="150" Margin="10" x:Name="_number2" FontSize="30" Text="0" TextAlignment="Right"/>
    <TextBlock Text="=" Margin="10" FontSize="30" VerticalAlignment="Center"/>
    <TextBlock Text="?" Width="150" Margin="10" x:Name="_result" FontSize="30" VerticalAlignment="Center"/>
    <Button Content="Caclulate" Margin="10" FontSize="25" />
</StackPanel>
```

当我们希望引起您对代码块的特定部分的注意时，相关行或项目将以粗体显示：

```cpp
<Button Style="{StaticResource numericStyle}" Grid.Row="1" 
        Content="7" Click="OnNumericClick" />
<Button Style="{StaticResource numericStyle}" Grid.Row="1" 
        Grid.Column="1" Content="8" Click="OnNumericClick"/>
<Button Style="{StaticResource numericStyle}" Grid.Row="1" 
        Grid.Column="2" Content="9" Click="OnNumericClick"/>
```

**新术语**和**重要单词**以粗体显示。例如：“关于 Windows 8 的第一件事是新的**开始**屏幕。”

### 注意

警告或重要说明以此框显示。

### 提示

提示和技巧以此方式显示。
