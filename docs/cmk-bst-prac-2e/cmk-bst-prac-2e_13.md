# 第十一章：为 Apple 系统创建软件

在软件开发领域，为 Apple 平台——macOS、iOS、watchOS 和 tvOS——构建应用程序具有一套独特的要求和最佳实践。由于封闭的生态系统，Apple 平台有一些独特的构建特点，尤其是对于具有图形用户界面和更复杂库框架的应用程序。针对这些情况，Apple 使用了名为应用程序包（app bundle）和框架（framework）的特定格式。本章深入探讨了如何有效地使用 CMake 这一强大的跨平台构建系统，针对 Apple 生态系统进行开发。无论你是希望简化工作流程的资深开发者，还是渴望探索 Apple 特定开发的新人，本章将为你提供掌握这一过程所需的知识和工具。

在本章中，我们将涵盖以下主要内容：

+   使用 XCode 作为 CMake 生成器

+   创建应用程序包

+   创建 Apple 库框架

+   为 Apple Store 使用的软件进行签名

# 技术要求

与前面章节相同，示例在 macOS Sonoma 14.2 上使用 CMake 3.24 进行了测试，并可在以下编译器中运行：

+   Apple Clang 15 或更新版本

+   Xcode 15.4

有些示例要求你已为 Xcode 设置了代码签名，这需要一个已加入开发者计划的 Apple ID。

所有示例和源代码可以在本书的 GitHub 仓库中找到：[`github.com/PacktPublishing/CMake-Best-Practices---2nd-Edition`](https://github.com/PacktPublishing/CMake-Best-Practices---2nd-Edition)。

如果缺少任何软件，相应的示例将从构建中排除。

# 在 Apple 上使用 CMake 开发

在 macOS 上开发时，理解 Apple 的生态系统和开发期望非常重要。Apple 强烈鼓励开发者使用 Xcode 作为主要开发环境，CMake 可以为其生成构建指令。Xcode 是一个包含代码编辑器、调试器以及其他专为 macOS 和 iOS 开发设计的工具的综合工具套件。Apple 会频繁更新 Xcode，新增功能、修复 bug 以及提升性能。因此，开发者需要定期升级到最新版本的 Xcode，以确保兼容性并充分利用新功能。

以下命令为 Xcode 生成构建指令：

```cpp
cmake -S . -B ./build -G Xcode
```

在运行前述命令后，CMake 将在构建目录中生成一个以 `.xcodeproj` 后缀的 Xcode 项目文件。该项目可以直接在 Xcode 中打开，或者通过调用以下命令，使用 CMake 构建项目：

```cpp
cmake --build ./build
```

如果项目已由 CMake 更新，Xcode 将检测到更改并重新加载项目。

虽然 Xcode 是首选的 IDE，但并非唯一的选择。例如，**Visual Studio Code**（**VS Code**）是一个广受欢迎的替代方案，许多开发者使用它，因为它的多功能性和丰富的扩展生态系统。

虽然 Xcode 是推荐的 macOS 生成器，但也可以选择其他生成器，如 Ninja 或 Makefile。尽管它们缺乏一些 Apple 集成的特性，但它们轻量且也可以用于构建简单的应用程序和库。

CMake 中的 Xcode 生成器会创建可以直接在 Xcode 中打开和管理的项目文件。这确保了构建过程能够利用 Xcode 的功能，如优化的构建设置、资源管理以及与 macOS 特性的无缝集成。

Xcode 支持各种构建设置，这些设置可以通过 CMake 使用`XCODE_ATTRIBUTE_<SETTING>`属性进行配置。要获取所有可能设置的列表，可以参考 Apple 的开发者文档，或者运行以下构建目录调用：

```cpp
xcodebuild -showBuildSettings
```

Xcode 属性可以全局设置，也可以针对每个目标进行设置，如下例所示：

```cpp
set(CMAKE_XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY "Apple Development")
set(CMAKE_XCODE_ATTRIBUTE_DEVELOPMENT_TEAM "12345ABC")
add_executable(ch11_hello_world_apple src/main.mm)
set_target_properties(ch11_hello_world_apple PROPERTIES
    XCODE_ATTRIBUTE_CXX_LANGUAGE_STANDARD «c++17»
    XCODE_ATTRIBUTE_ENABLE_HARDENED_RUNTIME[variant=Release] "YES"
)
```

如示例所示，Xcode 属性也可以仅为特定的构建变体设置，通过在`[variant=ConfigName]`后缀中指定，其中配置是常见的 CMake 构建变体，如`Debug`和`Release`。如果使用不同于 Xcode 的生成器来构建项目，这些属性将无效。要创建简单的应用程序，选择 Xcode 生成器并设置适当的属性就足够了。然而，要为 Apple 构建和分发更复杂的软件，就必须深入研究应用程序包和 Apple 框架。

# Apple 应用程序包

Apple 应用程序包是 macOS 和 iOS 中用于打包和组织应用程序所需所有文件的目录结构。这些文件包括可执行二进制文件、资源文件（如图片和声音）、元数据（如应用程序的`Info.plist`文件）等。应用程序包使得分发和管理应用程序变得更加容易，因为它们将所有必要的组件封装在一个单独的目录中，用户可以像处理单个文件一样移动、复制或删除该目录。应用程序包在 Finder 中显示为单个文件，但实际上它们是具有特定结构的目录，如下所示：

```cpp
MyApp.app
└── Contents
    ├── MacOS
    │   └── MyApp (executable)
    ├── Resources
    │   ├── ...
    ├── Info.plist
    ├── Frameworks
    │   └── ...
    └── PlugIns
        └── ...
```

值得注意的是，这个结构被扁平化处理，适用于 iOS、tvOS 和 watchOS 目标平台，这由 CMake 本身处理。要将可执行文件标记为一个应用包，需要将`MACOSX_BUNDLE`关键字添加到目标中，如下所示：

```cpp
add_executable(myApp MACOSX_BUNDLE)
```

设置关键字会告诉 CMake 创建应用程序包的目录结构并生成一个`Info.plist`文件。`Info.plist`文件是应用程序包中的关键文件，因为它包含了该包的必要配置。CMake 会生成一个默认的`Info.plist`文件，在许多情况下它已经很好用了。然而，你可以通过指定模板文件的路径来使用自定义的`Info.plist`模板，方法是使用`MACOSX_BUNDLE_INFO_PLIST`属性，像这样在目标上指定：

```cpp
set_target_properties(hello_world PROPERTIES
    MACOSX_BUNDLE_INFO_PLIST ${CMAKE_SOURCE_DIR}/Info.plist.in
)
```

模板文件使用与`configure_file`相同的语法，如*第八章*所述，*使用 CMake 执行自定义任务*。一些标识符会从 CMake 自动设置。下面是一个自定义`Info.plist.in`模板的示例：

```cpp
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key>
    <string>@MACOSX_BUNDLE_IDENTIFIER@</string>
    <key>CFBundleName</key>
    <string>@MACOSX_BUNDLE_NAME@</string>
    <key>CFBundleVersion</key>
    <string>@MACOSX_BUNDLE_VERSION@</string>
    <key>CFBundleExecutable</key>
    <string>@MACOSX_BUNDLE_EXECUTABLE@</string>
</dict>
</plist>
```

这样，一个基本的应用程序包就可以定义并构建了。虽然这在 Xcode 生成器下效果最好，但创建文件结构和`Info.plist`文件在其他生成器如 Ninja 或 Makefile 中也能工作。然而，如果应用程序包含界面构建器文件或故事板，Xcode 是唯一的选择。接下来我们来看看如何包含资源文件。

## Apple 应用程序包的资源文件

Apple 提供了自己的 SDK 来构建应用程序，它支持故事板或界面构建器文件来定义用户界面。为了创建包，这些文件会被编译成 Apple 特有的资源格式，并放置在包中的适当位置。只有 Xcode 生成器支持自动处理这些文件，因此在这时，使用其他生成器几乎没有什么用处。

故事板和界面构建器源文件必须作为源文件包含在`add_executable()`或`target_sources()`命令中。为了确保它们被自动编译并复制到包内的正确位置，这些源文件应该设置`MACOSX_PACKAGE_LOCATION`文件属性。为了避免重新输入所有文件名，将它们放入变量中会很方便：

```cpp
set(resource_files
storyboards/Main.storyboard
storyboards/My_Other.storyboard
)
add_executable(ch11_app_bundle_storyboard MACOSX_BUNDLE src/main.mm)
target_sources(ch11_app_bundle_storyboard PRIVATE ${resource_files})
set_source_files_properties(${resource_files} PROPERTIES MACOSX_PACKAGE_LOCATION Resources)
```

到此为止，你应该已经设置好创建 Apple 应用程序包来生成可执行文件了。如果你想为 Apple 构建易于重新分发的复杂库，那么使用框架是最佳选择。

# Apple 框架

macOS 框架是用于简化 macOS 应用程序开发的可重用类、函数和资源的集合。它们提供了一种结构化的方式来访问系统功能，并与 macOS 无缝集成。

链接框架与链接普通库非常相似。首先，使用`find_package`或`find_library`找到它们，然后使用`target_link_libraries`，只是语法稍有不同。以下示例将`Cocoa`框架链接到`ch11_hello_world_apple`目标：

```cpp
target_link_libraries(ch11_hello_world_apple PRIVATE
"-framework Cocoa")
```

请注意，`–framework`关键字是必要的，而且框架的名称应该加引号。如果链接多个框架，它们都需要分别加引号。

既然我们可以使用框架，接下来让我们来看看如何创建自己的框架。框架与应用程序包非常相似，但有一些区别。特别是，在 macOS 上可以安装同一框架的多个版本。除了资源之外，框架还包含使用它所需的头文件和库。让我们来看看框架的文件结构，以便理解框架如何支持多个版本，并填充头文件：

```cpp
MyFramework.framework/
│
├── Versions/
│   ├── Current -> A/
│   ├── A/
│   │   ├── Headers
│   │   │    └── ...
│   │   ├── PrivateHeaders
│   │   │    └── ...
│   │   ├── Resources
│   │   │    ├── Info.plist
│   │   │    └── ...
│   │   ├── MyFramework
├── MyFramework -> Versions/Current/MyFramework
├── Resources -> Versions/Current/Resources
├── Headers -> Versions/Current/Headers
├── PrivateHeaders -> Versions/Current/PrivateHeaders
└── Info.plist -> Versions/Current/Resources/Info.plist
```

框架的顶层必须以 `.framework` 结尾，通常，除了 `Versions` 文件夹外，所有顶层文件都是指向 `Versions` 文件夹中文件或文件夹的符号链接。

`Versions` 文件夹包含不同版本的库。目录的名称可以是任意的，但通常，它们会被命名为 `A`、`B`、`C` 等，或者使用数字版本。

无论使用哪种约定，都需要有一个名为 `Current` 的符号链接，并且该链接应该指向最新版本。每个版本必须有一个名为 `Resources` 的子文件夹，该文件夹包含如在应用程序包部分所述的 `Info.plist` 文件。

CMake 支持通过设置目标的 `FRAMEWORK` 和 `FRAMEWORK_VERSION` 属性来创建 macOS 框架：

```cpp
add_library(ch11_framework_example SHARED)
set_target_properties(
    ch11_framework_example
    PROPERTIES FRAMEWORK TRUE
               FRAMEWORK_VERSION 1
               PUBLIC_HEADER include/hello.hpp
               PRIVATE_HEADER src/internal.hpp
)
```

由于框架通常包含头文件，因此可以使用 `PUBLIC_HEADER` 和 `PRIVATE_HEADER` 属性来指定它们，以便将它们复制到正确的位置。可以使用 `MACOSX_FRAMEWORK_INFO_PLIST` 属性为框架设置自定义的 `Info.plist` 文件。如果没有提供自定义的 `Info.plist` 文件，CMake 将生成一个默认文件，在大多数情况下这是足够的。

到目前为止，我们已经介绍了构建 macOS 软件的基础知识，但有一点是缺失的，那就是对代码进行签名，以便将其发布到 Mac 上。

# macOS 的代码签名

对于许多使用场景，创建未签名的应用或框架可能已经足够；然而，如果应用程序需要通过 macOS 的官方渠道进行分发，则必须进行签名。签名可以通过 Xcode 本身进行；但是，也可以使用 CMake 来进行签名。

要进行签名，需要三个信息：*包或框架的 ID*、*开发团队 ID* 和 *代码签名实体*。可以使用 `XCODE_ATTRIBUTE_DEVELOPMENT_TEAM` 和 `XCODE_ATTRIBUTE_CODE_SIGN_ENTITY` Xcode 属性来设置这些值。通常，这些设置是在项目级别进行，而不是针对单独的目标：

```cpp
set(CMAKE_XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY "Apple Development" CACHE STRING "")
set(CMAKE_XCODE_ATTRIBUTE_DEVELOPMENT_TEAM "12345ABC" CACHE STRING "")
```

签名身份表示证书提供者，通常可以保持为 `"Apple Development"`，这将使 Xcode 选择适当的签名身份。在 Xcode 11 之前，签名身份必须设置为 `"Mac Developer"`（用于 macOS）或 `"iPhone Developer"`（用于 iOS、tvOS 或 watchOS 应用程序）。团队 ID 是一个 10 位的代码，分配给 Apple 开发者帐户，可以在 Apple 的开发者门户网站上创建 ([`developer.apple.com`](https://developer.apple.com))。可以通过 Xcode 下载签名证书。

# 总结

苹果生态系统在处理上有些特殊，因为其封闭的设计与 Linux 甚至 Windows 相比有很大不同，但尤其是在移动市场，如果不想失去一个重要市场，通常无法避免为 Apple 构建应用。通过本章中的信息，如创建应用程序包和框架以及签署软件，你应该能够开始为 Apple 部署应用程序。虽然为 Apple 构建应用需要使用 Xcode，并且可能还需要拥有 Apple 硬件，但这并不是所有其他平台的情况。CMake 擅长于平台独立性并能够跨不同平台构建软件，这正是我们将在下一章讨论的内容。

# 问题

1.  哪种生成器最适合为 Apple 构建软件？

1.  如何设置 Xcode 属性？

1.  关于不同版本，应用程序包和框架之间有什么区别？

1.  签署应用程序包或框架需要什么？

# 答案

1.  尽管不同的生成器可以用于 Apple，推荐使用 Xcode。

1.  通过设置`CMAKE_XCODE_ATTRIBUTE_<ATTRIBUTE>`变量或`XCODE_ATTRIBUTE_<ATTRIBUTE>`属性来实现。

1.  每次只能安装一个版本的应用程序包，而可以同时安装多个版本的框架。

1.  要为 Apple 签署软件，你需要一个已注册开发者计划的 Apple ID 和 Xcode。
