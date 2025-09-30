# 5

# 保存和加载配置

欢迎来到*第五章*！在前一章中，我们向应用程序添加了一个单独的视图模式。在这个仅查看模式下，用户界面和选择功能被禁用。然后，我们添加了实例的简化版和最终的全版本撤销/重做。现在，实例的设置更改可以被撤销或重新应用。

在本章中，我们将添加将应用程序配置保存到文件的能力。首先，我们将探索不同的文件类型来存储数据。在考虑每种文件类型的优缺点并确定合适的文件格式后，我们将深入研究文件格式的结构。然后，我们将实现一个解析器类，使我们能够加载和保存配置。最后，我们将在应用程序启动时加载默认配置，使用户能够尝试使用应用程序。

对于任何更大的应用程序，能够保存创建或更改的数据的当前状态，停止应用程序，然后再次加载数据以继续工作至关重要。从挂起或崩溃的应用程序中恢复到最新保存版本也同样重要。你不希望因为应用程序无法正确地将数据保存到本地或远程存储而丢失数小时的工作。

在本章中，我们将涵盖以下主题：

+   文本或二进制文件格式——优缺点

+   选择一种文本格式来保存我们的数据

+   探索 YAML 文件的结构

+   添加 YAML 解析器

+   保存和加载配置文件

+   启动时加载默认配置文件

# 技术要求

本章的示例源代码位于`chapter05`文件夹中，在`01_opengl_load_sve`子文件夹中为 OpenGL，在`02_vulkan_load_save`中为 Vulkan。

在我们向应用程序添加保存和加载功能之前，让我们看看一些将数据保存到存储设备的方法。

# 文本或二进制文件格式——优缺点

每当你玩游戏、剪辑视频、编辑照片、编写文本或书籍时，你都会使用与所使用的软件集成的集成加载和保存功能。在安全位置保存游戏、在大量编辑后存储视频序列，或者在文本编辑器中不时按下*Ctrl* + *S*已经变得很正常了。

但是，你有没有想过保存和加载数据？你可能对以下功能有一些疑问，比如：

+   需要存储什么才能完全恢复你的最新状态？

+   应该使用什么格式来保存数据？

+   如果程序更新了会发生什么？我能否加载保存的数据？

+   我能否在没有原始程序的情况下读取或更改保存的文件？

对于你使用的所有程序，关于格式和数据量的决定在于它们的开发者。但是，对于我们的应用程序，我们必须决定哪些数据需要保存，以及如何将数据保存到存储中。

所有类型的数据格式都有其优缺点；以下是一个简要总结。

## 保存和加载二进制数据

在计算机时代的早期，存储空间和计算时间都很昂贵、宝贵且稀缺。为了在保存和恢复过程中最小化空间和时间，数据基本上只是内存转储。

要保存的数据存储在内部数据类型中，进入计算机的内存区域，然后逐字节地复制到软盘或硬盘上。

加载相同的数据与保存它一样简单快捷：从存储设备读取数据到计算机内存（再次，逐字节）并将其解释为保存时使用的相同内部数据类型。

让我们来看看二进制数据的优点和缺点：

+   **优点**：

    +   文件较小。

    +   通过仅复制数据即可保存和加载数据，这导致保存和加载操作的速度更快。

+   **缺点**：

    +   要在应用程序外部更改数据，需要特殊知识。

    +   损坏的资料可能会引起不可预测的副作用或崩溃。

    +   保存文件格式的更新可能很困难。需要使用**魔数**来找到实际版本，并将加载的数据映射到正确的内部数据类型。

由于不同的字节序，二进制数据可能无法在不同的架构之间移植。因此，除非绝对必要，否则通常建议避免使用二进制保存文件。我们仍然可以读取 30 年前在 MS-DOS 系统上创建的`CONFIG.SYS`和`AUTOEXEC.BAT`文件，但同一时期的电子表格计算器或文字处理器的二进制保存文件是无法使用的，至少在没有正确工具或进行逆向工程文件格式的艰苦工作的情况下是这样。在像图片或声音文件这样有良好文档和标准化的格式之外，保存二进制数据**将会**引起麻烦，因为你可能无法在不同的操作系统上打开二进制文件，甚至可能无法打开同一系统的较新版本。

由于 CPU 时间和存储空间不再受限，文本格式的优势现在明显超过了二进制保存的优势。

## 保存和加载文本数据

随着 CPU 性能、网络带宽和存储空间的不断增加，文本格式开始成为保存数据的首选。当你可以使用简单的文本编辑器创建或调整保存文件，或者可以通过将文本行打印到日志文件中查找错误时，编写代码会变得容易得多。

对于基于文本的保存文件，条件与二进制保存文件的条件不同：

+   文件较大，除非它们被压缩成.zip 文件或类似的格式。那时，还需要另一个转换步骤（打包/解包）。

+   每次加载数据和保存数据时，数据都必须从二进制表示形式转换为文本，然后再转换回来。

+   对于较大的更改或从头创建保存文件，可能需要特定于域的文件格式知识。但对于简单的值更改，我们只需要一个文本编辑器。

+   损坏的文件可以修复，或者可以直接从文本文件中删除损坏的数据元素。宁愿丢失一些数据，也不要全部丢失。

+   通过在文件中增加版本号，可以检测文件格式的更新，帮助应用程序使用正确的转换。

+   在具有不同架构或操作系统的计算机上加载相同的文件根本不是问题。文本表示法是相同的，并且由于从文本到二进制数据类型的转换，字节序或数据类型长度并不重要。

+   对于跨平台使用，仍然存在一些注意事项，例如 Windows 和 Linux 中的不同路径分隔符，或者系统区域设置中点号和逗号的不同解释。

+   如果需要将配置分割成多个文件，将所有文件打包到压缩文件中是最常见的方式。通过将所有配置文件添加到 `.zip` 或 `.tar.gz` 文件中，最终得到一个单独的文件，并且由于压缩节省了一些磁盘空间。

使用文本表示法来保存文件是可行的。通过定义简单的文件格式，甚至可以手动创建配置文件。

但是，在我们自己创建文件格式之前，让我们检查一些可用的文件格式。通过使用众所周知的文件格式，我们可以节省大量时间，因为我们不需要创建解析和写入文件的函数。

# 选择文本格式来保存我们的数据

在本节中，我们将探讨三种流行的配置文件格式：

+   **INI**

+   **JSON**

+   **YAML**

所有三种格式在某些领域都有应用，但可能不适合其他领域。

## INI 文件格式

存储配置数据的最古老格式之一是所谓的 INI 格式。这个名字来自文件扩展名 `.ini`，它是初始化的三个字母缩写。INI 文件主要用于配置程序。

在 INI 文件中，简单的键/值对存储并组织在可选的部分中。部分名称用方括号括起来，部分作用域从部分的开始到另一个部分的开始，或文件的末尾。

这里是一个数据库连接的示例部分和一些键/值对：

```cpp
[database-config]
type = mysql
host = mysql
port = 3306
user = app1
password = ASafePassWord1# 
```

通过使用特殊字符（如点`.`或反斜杠`\`）分隔部分和子部分，可以嵌套部分以创建某种层次结构。对于解析器识别这些部分划分至关重要；否则，每个部分都将独立处理，不考虑它们之间的层次关系。

缺乏重复键名使得存储具有重复键的分层或非平凡数据变得困难，例如模型文件或实例配置。

## JSON 文件格式

JSON，即 JavaScript 对象表示法，在 2000 年代初首次亮相。与 INI 文件一样，键/值对存储在 JSON 文件中。与 INI 文件部分类似的部分不存在；相反，JSON 文件允许创建复杂、树状的结构。此外，可以定义相同数据类型的数组。

JSON 文件的主要用途是电子数据交换，例如，在 Web 应用程序和后端服务器之间。文件格式的良好可读性只是副作用；JSON 文件主要是由应用程序读取和编写的。

这是一个 JSON 文件的示例，包含与 INI 文件示例相同的数据：

```cpp
{
  "database-config": {
    "type": "mysql",
    "host": "mysql",
    "port": 3306,
    "user": "app1",
    "password": "ASafePassWord1#"
  }
} 
```

很遗憾，由于大括号数量众多，第一次尝试正确编写文件格式是困难的。此外，不允许注释，因此只能在保存原始文件的副本并调整内容的情况下，*实时* 测试不同的选项。

## YAML 文件格式

YAML 的名字最初是一个缩写，代表 *Yet Another Markup Language*。在 2000 年代初，产品名称前缀的 *yet another* 被用作与计算机相关的幽默，表明原创想法的不断重复。但是，由于 YAML 不是一个像 HTML 或 XML 那样的标记语言，因此名字的含义被改为递归缩写 *YAML Ain’t Markup Language*。

YAML 和 JSON 密切相关。一个 JSON 文件可以被转换成 YAML 文件，反之亦然。这两种格式之间的主要区别是，YAML 中使用缩进来创建层次结构，而不是使用大括号。

YAML 的主要目标是使人类可读。YAML 格式广泛用于创建和维护结构化和层次化的配置文件（例如，在配置管理系统和云环境中）。

下面是一个 YAML 文件格式的示例，再次使用与 INI 和 JSON 相同的数据：

```cpp
database-config:
  type: mysql
  host: mysql
  port: 3306
  user: app1
  password: ASafePassWord1# 
```

由于 YAML 格式简单且强大，具备我们所需的所有功能，并且可以像 JSON 一样无需在缺失的大括号上卡壳进行读写，因此我们将使用 YAML 文件来存储我们应用程序的配置数据。

# 探索 YAML 文件的结构

让我们看看 YAML 文件的主要三个组成部分：

+   **节点**

+   **映射**

+   **序列**

让我们从节点开始。

## YAML 节点

YAML 文件的主要对象是一个所谓的节点。节点代表其下面的数据结构，可以是标量值、映射或序列。

```cpp
of a configuration file as an example:
```

```cpp
database-config:
  type: mysql 
```

`database-config` 和 `type` 都是 YAML 节点。虽然 `database-config` 节点包含一个包含键 `type` 和值 `mysql` 的映射，但 `type` 节点中仅包含标量值 `mysql`。

## YAML 映射

一个 YAML 映射包含零个或任意数量的键/值对。此外，节点和映射之间存在一个有趣的关联：键和值可能又是另一个节点，从而在文件中创建层次结构。

让我们扩展节点部分之前的配置片段：

```cpp
database-config:
  type: mysql
  host: mysql
  port: 3306 
```

如前所述，`database-config` 既是节点也是地图。名为 `database-config` 的地图的键是名称 `database-config`，值是另一个包含三个键值对的地图。

## YAML 序列

为了列举相似元素，使用 YAML 序列。序列可以看作是一个 C++ `std::vector`，其中所有元素必须是同一类型。像 C++ 向量一样，你遍历序列，逐个读取数据元素。

序列有两种不同的风格：*块风格*和*流风格*。

在块风格中，缩进的破折号（`-`）用作元素的指示符：

```cpp
colors:
  - red
  - green
  - blue 
```

相反，流风格使用方括号，元素之间用逗号分隔：

```cpp
colors: [red, green, blue] 
```

这两种风格表示相同的数据。这取决于个人喜好和可读性。

通过组合地图和序列，可以创建复杂的数据结构。

## 地图和序列的组合

通过混合地图和序列，可以实现表示数据的一种强大方式。例如，我们可以像这样存储所有模型实例的 `position` 和 `rotation`：

```cpp
instances:
  - position: [0, 0, 0]
    rotation: [0, 0, 0]
  - position: [1, 0, -3]
    rotation: [0, 90, 0]
  - position: [3, 0, 3]
    rotation: [0, -90, 0] 
```

在这里，我们使用了地图和序列的组合。

首先，我们创建一个由键 `position` 和 `rotation` 组成的地图；值是表示 `glm::vec3` 的流式序列数字。YAML 总是存储标量数字的最短可能表示形式。因此，只要值没有小数部分，就会使用整数值，即使对于 `float` 和 `double` 类型也是如此。然后，使用 `position` 和 `rotation` 的地图在块序列中创建模型实例的数组式表示。

要将实例的数据读入我们的应用程序，我们必须首先遍历模型实例序列，并且对于每个实例，我们可以提取位置和旋转值。

在对 YAML 文件格式的基本介绍之后，我们现在将为我们的应用程序实现一个 YAML 解析器和写入器类，以便保存和加载其配置。将配置存储在磁盘上就像保存一个文本文档一样——我们可以退出应用程序，稍后再继续在虚拟世界中工作。我们还可以使用保存的文件返回到虚拟世界的先前状态。

# 添加 YAML 解析器

与我们正在使用的其他工具（如 Open Asset Import Library、GLFW 或 ImGui）一样，我们将使用一个免费的开源解决方案：`yaml-cpp`。

通过集成 `yaml-cpp`，我们可以用最小的努力从 C++ 中读取和写入 YAML 文件。最大的步骤是确保我们的自定义数据类型为 `yaml-cpp` 所知。此外，我们必须考虑数据文件的正确结构。

让我们先探讨如何将 `yaml-cpp` 集成到我们的项目中。

## 获取 yaml-cpp

对于 Linux 系统，获取 `yaml-cpp` 很容易。与其他工具类似，大多数发行版已经包含了 `yaml-cpp` 库和头文件。例如，在 Ubuntu 22.04 或更高版本中，可以使用以下命令安装 `yaml-cpp` 和其开发文件：

```cpp
sudo apt install libyaml-cpp-dev 
```

如果你使用的是基于 Arch 的 Linux 发行版，可以使用以下命令安装 `yaml-cpp`：

```cpp
sudo pacman –S yaml-cpp 
```

对于 Windows，我们也同样幸运。`yaml-cpp` 使用 CMake，通过使用 CMake 的 `FetchContent` 命令，只需几行代码就可以将 `yaml-cpp` 添加到项目中。首先，我们在项目根目录下的 `CMakeLists.txt` 文件中添加 `FetchContent` 声明。我们使用 `yaml-cpp` 的 `0.8.0` 版本：

```cpp
 FetchContent_Declare(
    yaml-cpp
    GIT_REPOSITORY https://github.com/jbeder/yaml-cpp
    GIT_TAG 0.8.0
  ) 
```

确保我们在 `CMakeLists.txt` 文件的 `WIN32` 部分内部。在 Linux 上我们不需要下载库。

然后，我们触发 `yaml-cpp` 的下载并添加目录变量：

```cpp
 FetchContent_MakeAvailable(yaml-cpp)
  FetchContent_GetProperties(yaml-cpp)
  if(NOT yaml-cpp_POPULATED)
    FetchContent_Populate(yaml-cpp)
    add_subdirectory(${yaml-cpp_SOURCE_DIR}
      ${yaml-cpp_BINARY_DIR} EXCLUDE_FROM_ALL)
  endif() 
```

Windows 还需要一个脚本来检测下载的依赖项。检测脚本必须命名为 `Findyaml-cpp.cmake` 并放置在 `cmake` 文件夹中。

脚本的主要功能归结为这两个 CMake 函数：

```cpp
find_path(YAML-CPP_INCLUDE_DIR yaml-cpp/yaml.h
          PATHS ${YAML-CPP_DIR}/include/)
find_library(YAML-CPP_LIBRARY
             NAMES ${YAML-CPP_STATIC} yaml-cpp
             PATHS  ${YAML-CPP_DIR}/lib) 
```

感谢 CMake 的 `FetchContent`，`YAML-CPP_DIR` 变量被填充为下载的 `yaml-cpp` 代码的路径。因此，脚本只需检查头文件和库是否可以找到。

## 将 yaml-cpp 集成到 CMake 构建

对于 Linux 和 Windows，我们必须为编译器设置正确的包含路径，并将 `yaml-cpp` 库添加到链接库列表中。

要更新 `include` 路径，将 `YAML_CPP_INCLUDE_DIR` 变量添加到 `include_directories` 指令中：

```cpp
include_directories(... ${YAML_CPP_INCLUDE_DIR}) 
```

对于链接器，需要在 Windows 中将 `yaml-cpp::yaml-cpp` 添加到 `target_link_libraries`：

```cpp
target_link_libraries(... yaml-cpp::yaml-cpp) 
```

对于 Linux，只需要共享库的名称 `yaml-cpp`：

```cpp
target_link_libraries(... yaml-cpp) 
```

再次运行 CMake 后，`yaml-cpp` 将被下载并可供代码的其他部分使用。

## 添加解析器类

解析 YAML 文件以加载和创建写入内容将在一个名为 `YamlParser` 的新类中完成，该类位于 `tools` 目录中。我们可以在包含头文件后使用 `yaml-cpp`：

```cpp
#include <yaml-cpp/yaml.h> 
```

在加载或创建用于保存到磁盘的数据结构时，需要两个额外的私有成员来存储中间数据：

```cpp
 YAML::Node mYamlNode{};
    YAML::Emitter mYamlEmit{}; 
```

`YAML::Node` 将 YAML 文件的节点从磁盘转换为 C++ 数据结构，简化了对已加载数据的访问。`YAML::Emitter` 用于通过追加数据元素在内存中创建 YAML 文件，最终将结构化数据写入文件。

### 使用 yaml-cpp 的节点类型

通过使用节点名称作为 `YAML::Node` 类中存储的 C++ 映射的索引来访问 `yaml-cpp` 节点的结构化或标量数据：

```cpp
 mYamlNode = YAML::LoadFile(fileName);
  YAML::Node settingsNode = mYamlNode["settings"]; 
```

要检索简单键/值映射的标量值，存在特殊运算符 `as`：

```cpp
 int value = dataNode.as<int>(); 
```

由于 YAML 文件不了解存储在值中的数据类型，我们必须明确告诉 `yaml-cpp` 如何解释传入的数据。在这里，`yaml-cpp` 将尝试获取 `dataNode` 节点的值为 `int`。

在定义了转换模板之后，也可以直接将自定义数据类型如结构体读入相同类型的变量中：

```cpp
InstanceSetting instSet = instNode.as<InstanceSettings>(); 
```

我们将在**保存和加载配置文件**部分处理转换模板。

### 访问序列和映射

在 `yaml-cpp` 中，可以通过使用 `for` 循环遍历它们来读取序列，并通过索引访问元素：

```cpp
for (size_t i = 0; i < instNode.size(); ++i) {
  instSettings.emplace_back(
    instNode[i].as<InstanceSettings>());
} 
```

序列的元素以它们在 YAML 文件中出现的顺序提供。

对于映射，需要一个迭代风格的 `for` 循环：

```cpp
for(auto it = settingsNode.begin();
    it != settingsNode.end(); ++it) {
  if (it->first.as<std::string>() == "selected-model") {
    ....
  }
} 
```

映射元素的键可以通过 C++ 映射容器的 `first` 访问器读取。同样，我们必须告诉 `yaml-cpp` 映射键的数据类型。然后，可以使用 C++ 映射容器的 `second` 访问器检索值。

如果读取值失败，例如，因为类型错误或不存在这样的节点，将抛出异常。为了避免我们的程序被终止，我们必须处理所有抛出的异常。

### 处理由 yaml-cpp 抛出的异常

`yaml-cpp` 在出错时抛出异常，而不是返回错误代码。默认情况下，任何未处理的异常都会终止程序。在 C++ 中处理异常的方式与其他语言类似：

```cpp
 try {
    mYamlNode = YAML::LoadFile(fileName);
  } catch(...) {
    return false;
  } 
```

可能引发异常的调用将被包含在 `try` 块中，如果发生异常，则执行 `catch` 块。

我们可以简单地捕获所有异常，因为任何解析失败可能会导致配置文件为空或不完整。如果您想要更详细的异常处理，可以探索 `yaml-cpp` 的源代码。

借助 `yaml-cpp` 的基础知识，我们可以开始实现保存和加载 YAML 文件的代码。我们将从保存功能开始，因为在我们已经在磁盘上创建了一个文件之后，将数据元素重新加载到应用程序中将会比首先手工制作配置文件要容易得多。

# 保存和加载配置文件

构建我们的配置文件始于决定需要存储什么以及我们希望如何存储元素。通过重用我们的自定义数据类型，如 `InstanceSettings`，创建保存和加载文件的函数可以简化。现在我们不再需要逐个读取每个值，而是可以使用来自 `AssimpInstance` 类的 `getInstanceSettings()` 和 `setInstanceSettings()` 调用来直接在解析器和实例之间传递值。

我们将首先探索我们想要保存的内容，在添加将我们的自定义数据写入文件的代码之后，将添加一个用户界面对话框，允许以简单的方式将文件保存到磁盘。最后，我们将逐步通过将配置重新加载到应用程序中的过程。

## 决定在配置文件中存储什么

如**保存和加载文本数据**部分所述，添加版本号可以在应用程序的开发过程中非常有帮助。如果我们需要更改数据格式，即使只是轻微的更改，提高版本号可以帮助我们在读取文件时简化在旧格式和新格式之间的分支。

接下来，我们应该存储有关选择的信息。恢复选定的模型和实例后，我们可以从保存配置文件的地方继续。

此外，我们还应该存储相机信息。在本书后面处理更复杂的场景时，将相机恢复到默认位置和角度可能会让应用程序用户感到困惑。

作为最重要的部分，我们必须存储所有关于模型和所有屏幕上所需的信息，以便将应用程序恢复到保存配置时的相同状态。

对于模型，文件名和路径就足够了，因为我们使用文件名作为模型的名称。模型文件的路径将相对于应用程序的可执行文件保存，而不是绝对路径，至少在模型位于与可执行文件相同的分区上时是这样（仅限 Windows）。这两种方法都有其优缺点：

+   相对路径允许用户从系统上的任何位置检出代码，能够直接使用示例配置文件和示例模型。然而，将可执行文件移动到另一个目录或分区时，需要移动所有配置数据和模型，或者必须手动调整配置文件。

+   使用绝对路径可能便于在 PC 上的固定位置（例如，在用户的家目录中）存储新的配置。这样，应用程序可以从 PC 上的任何位置启动，并且仍然可以找到配置文件和模型。

为了恢复所有实例，我们需要存储在 `InstanceSetting`s 结构中的所有信息以及模型名称。为了简化通过模型名称恢复实例，我们将模型名称作为 `std::string` 添加到 `InstanceSettings` 结构中。在结构中拥有模型名称允许我们将从 YAML 解析器传递给渲染类的一个 `InstanceSettings` 值的 `std::vector`；我们不需要更复杂的数据结构。

让我们从创建自定义元素写入器重载开始实现。

## 重载发射器的输出运算符

`yaml-cpp` 的创建者增加了一个很好的方法，可以将复杂结构的输出内容添加到 `YAML::Emitter` 中。我们只需在 `tools` 文件夹中的 `YamlParser.cpp` 文件中重载 `operator<<` 即可：

```cpp
YAML::Emitter& operator<<(YAML::Emitter& out,
    const glm::vec3& vec) {
  out << YAML::Flow;
  out << YAML::BeginSeq;
  out << vec.x << vec.y << vec.z;
  out << YAML::EndSeq;
  return out;
} 
```

对于 `glm::vec3` 数据类型，我们添加一个流类型序列，然后向流中添加向量的三个元素。在最终的文件中，将出现一个默认的 YAML 序列，包含 `glm::vec3` 向量的值，如下例所示：

```cpp
[1, 2, 3] 
```

我们的 `InstanceSettings` 结构必须作为 YAML 映射的键/值对添加。映射的开始和结束在存储 `InstanceSettings` 的函数中设置：

```cpp
YAML::Emitter& operator<<(YAML::Emitter& out,
    const InstanceSettings& settings) {
  out << YAML::Key << "model-file";
  out << YAML::Value << settings.isModelFile;
  out << YAML::Key << "position";
  out << YAML::Value << settings.isWorldPosition;
  out << YAML::Key << "rotation";
  out << YAML::Value << settings.isWorldRotation
  ...
} 
```

生成的映射将被添加为 YAML 序列的值，将所有相关实例数据存储在配置文件中。

## 创建和写入配置文件

要创建配置文件，渲染器将实例化我们的 `YamlParser` 类的一个本地对象，并对其调用 `createConfigFile()`：

```cpp
YamlParser parser;
if (!parser.createConfigFile(mRenderData, mModelInstData)) {
    return false;
  } 
```

在 `createConfigFile()` 方法中，`YAML::Emitter` 将填充我们的数据结构。例如，我们将在文件顶部添加一条注释，并在第二行保存版本号：

```cpp
 mYamlEmit << YAML::Comment("Application viewer
    config file");
  mYamlEmit << YAML::BeginMap;
  mYamlEmit << YAML::Key << "version";
  mYamlEmit << YAML::Value << 1.0f;
  mYamlEmit << YAML::EndMap;
  ... 
```

在最终的 YAML 文件中，以下第一行将出现：

```cpp
# Application viewer config file
version: 1 
```

作为另一个示例，为了存储实例设置，我们创建一个名为 `instances` 的映射，并开始创建一个序列作为其值：

```cpp
 mYamlEmit << YAML::BeginMap;
  mYamlEmit << YAML::Key << "instances";
  mYamlEmit << YAML::Value;
  mYamlEmit << YAML::BeginSeq; 
```

然后，我们可以创建一个遍历实例的 `for` 循环，并使用实例的 `getInstanceSettings()` 调用来直接将实例设置存储到发射流中：

```cpp
for (const auto& instance : modInstData.miAssimpInstances) {
  mYamlEmit << YAML::BeginMap;
  mYamlEmit << instance->getInstanceSettings();
  mYamlEmit << YAML::EndMap;
} 
```

多亏了 `operator<<` 重载，循环内不需要复杂的处理。

作为最后一步，我们关闭实例设置的序列，并关闭 `instances` 键的映射：

```cpp
 mYamlEmit << YAML::EndSeq;
  mYamlEmit << YAML::EndMap 
```

最终的 YAML 文件将包含所有实例的序列，包括新添加的模型文件名：

```cpp
instances:
  - model-file: Woman.gltf
    position: [0, 0, 0]
    rotation: [0, 0, 0]
    scale: 1
    swap-axes: false
    anim-clip-number: 0
    anim-clip-speed: 1 
```

如果我们想在执行任何磁盘写入之前查看创建的配置文件内容，我们可以创建一个 C 字符串并通过日志类输出该字符串：

```cpp
 Logger::log(1, "%s\n", mYamlEmit.c_str()); 
```

将文件写入磁盘将使用 `std::ostream` 完成。为了简洁，以下列表中省略了流的错误处理，但将文件保存到磁盘实际上只需三行代码：

```cpp
bool YamlParser::writeYamlFile(std::string fileName) {
  std::ofstream fileToWrite(fileName);
  fileToWrite << mYamlEmit.c_str();
  fileToWrite.close();
  return true;
} 
```

首先，我们使用给定的文件名创建输出流。然后，我们将 `YAML::Emitter` 的 `std::string` 转换为 C 字符串，并将字符串写入输出流。通过关闭流，文件将被刷新到存储设备。

## 向用户界面添加文件对话框

为了允许用户将配置文件存储在任意位置并使用自定义名称，我们将在用户界面中添加一个文件对话框。我们已经在使用基于 ImGui 的文件对话框来加载模型文件，并且我们可以重用相同的对话框实例向用户展示一个 **保存文件** 对话框。

要创建一个允许用户选择文件名和位置的对话框，必须对名为 `config` 的 `IGFD::FileDialogConfig` 变量进行三项更改。

首先，通过选择现有文件，我们需要一个额外的对话框来确认文件覆盖。幸运的是，文件对话框已经内置了这样的确认对话框。我们只需添加标志 `ImGuiFileDialogFlags_ConfirmOverwrite`：

```cpp
 config.flags = ImGuiFileDialogFlags_Modal |
      ImGuiFileDialogFlags_ConfirmOverwrite; 
```

如果我们选择一个现有文件，将显示一个新对话框，询问用户是否要替换现有文件。

接下来，我们将展示默认路径和文件名用于配置：

```cpp
 const std::string defaultFileName = "config/conf.acfg";
    config.filePathName = defaultFileName.c_str(); 
```

在这里，我们使用 `config` 文件夹和一个名为 `config.acfg` 的文件向用户展示一个默认文件。文件对话框代码将自动进入 `config` 文件夹并填写文件名和扩展名。

作为最后一步，我们将 `.acfg` 作为对话框的唯一文件扩展名添加：

```cpp
 ImGuiFileDialog::Instance()->OpenDialog(
      "SaveConfigFile", "Save Configuration File",
      ".acfg", config); 
```

通过为配置文件使用新的扩展名，我们避免了麻烦，例如尝试加载不同的文件格式或覆盖系统上的其他文件。

文件对话框中的 **OK** 按钮获取选定的文件名，并调用负责将配置保存到磁盘的回调函数：

```cpp
 if (ImGuiFileDialog::Instance()->IsOk()) {
      std::string filePathName =
        ImGuiFileDialog::Instance()->GetFilePathName();
      saveSuccessful =
        modInstData.miSaveConfigCallbackFunction(
        filePathName);
    } 
```

我们将回调函数的结果存储在布尔变量 `saveSuccessful` 中。这样，我们可以检查任何错误，并在保存配置不成功的情况下向用户显示对话框。

为了通知用户保存错误，仅实现了一个简单的对话框，提示用户检查应用程序的输出消息以获取关于写入错误原因的详细信息。

如果你现在加载一些模型，创建实例或克隆，并保存配置，你可以检查创建的配置文件。所有来自 *决定在配置文件中存储什么* 部分的数据都应该包含在配置文件中。

将数据保存到磁盘只是工作的一半。为了从保存文件的位置继续工作，我们需要将配置文件重新加载到应用程序中。

## 重新加载配置文件并解析节点

为了支持在 YAML 文件中解析自定义数据类型，`yaml-cpp` 允许我们定义一个位于 YAML `namespace` 中的名为 `convert` 的 C++ 模板结构体。`convert` 结构体必须实现两个方法，即 `encode` 和 `decode`，这两个方法分别负责将 C++ 类型序列化为 YAML (`encode`) 以及从 YAML 反序列化回 C++ (`decode`)。通过使用这两个方法，`yaml-cpp` 允许在 C++ 类型与 YAML 条目之间实现无缝转换。`encode` 方法从一个原始或自定义数据类型创建一个新的 YAML 节点，而 `decode` 方法读取 YAML 节点数据并返回原始或自定义数据类型。

对于将 `glm::vec3` 元素写入 YAML 节点并将 YAML 节点读取回 `glm::vec3`，必须在头文件中实现以下模板代码：

```cpp
namespace YAML {
  template<>
  struct convert<glm::vec3> {
    static Node encode(const glm::vec3& rhs) {
      Node node;
      node.push_back(rhs.x);
      node.push_back(rhs.y);
      node.push_back(rhs.z);
      return node;
    } 
```

要从 `glm::vec3` 保存数据，我们创建一个新的 YAML 节点称为 `node`，并将 `glm::vec3` 的三个元素 `x`、`y` 和 `z` 添加到节点中。然后，节点被返回给 `encode()` 方法的调用者。

使用 `decode()` 方法将数据从节点读取到 `glm::vec3` 变量中：

```cpp
 static bool decode(const Node& node, glm::vec3& rhs) {
      if(!node.IsSequence() || node.size() != 3) {
        return false;
      } 
```

检查节点类型以获取正确类型和大小是可选的，但这是一个良好的风格，确保我们有正确的自定义数据类型数据，以防止运行时错误。跳过此检查并尝试解析错误的数据类型将导致异常，如果未处理，则终止整个程序。

然后，我们通过序列索引从节点中读取数据，并将 `glm::vec3` 的三个元素 `x`、`y` 和 `z` 设置为节点中的浮点值：

```cpp
 rhs.x = node[0].as<float>();
      rhs.y = node[1].as<float>();
      rhs.z = node[2].as<float>();
      return true;
    }
  }; 
```

在定义了 `encode()` 和 `decode()` 方法之后，我们可以通过正常赋值在 YAML `node` 和 `glm::vec3` 之间交换数据：

```cpp
glm::vec3 data;
node["rotation"] = data.isWorldRotation;
data.isWorldRotation = node["rotation"].as<glm::vec3>(); 
```

对于`InstanceSettings`结构，实现了相同的方法，帮助我们直接将实例设置读回到`InstanceSettings`类型的变量中。为了避免污染我们的解析器类头文件，已在`tools`文件夹中创建了一个新的头文件`YamlParserTypes.h`。`YamlParserTypes.h`头文件将被包含在`YamlParser`类的头文件中，以便新的转换可用。

一旦配置文件成功解析，所有设置、模型路径和实例设置都被提取出来。但在我们可以加载模型和创建新实例之前，我们必须首先清除当前模型和实例列表。

## 从保存的值中清理和重新创建场景

移除所有模型和实例是一个简单直接的过程。在渲染器中，我们必须执行以下步骤以获得一个新鲜的环境：

1.  将包含当前选定的实例和模型的`miSelectedInstance`和`miSelectedModel`设置为零。从本步骤以及*步骤 2 和 3*中引入的变量在*第一章*的*动态模型和实例管理*部分中介绍。然后，在索引零处，将创建新的空模型和空实例。

1.  删除`miAssimpInstances`向量和清除`miAssimpInstancesPerModel`映射。现在，所有模型都未使用。

1.  删除`miModelList`向量。由于所有实例都已删除，模型的共享指针将不再被引用，模型将被删除。

1.  添加一个新的空模型和一个空实例。空模型和空实例必须是模型列表和实例向量以及映射中的第一个元素。

1.  清除撤销和重做栈。在栈中，我们只使用了弱指针，因此这一步可以在任何时候进行。

1.  更新三角形计数。在所有模型和实例被移除后，三角形计数应为零。

清理所有模型和实例的整个流程已被添加到渲染类的新`removeAllModelsAndInstances()`方法中，简化了每次需要干净且新鲜环境时的使用。

现在，我们可以从磁盘加载模型文件，但不需要创建默认实例。在所有模型加载完毕后，我们从模型列表中的`InstanceSettings`中搜索模型，创建一个新的实例，并应用配置文件中的设置。

接下来，我们应该列举实例，因为实例索引号并未存储在`InstanceSettings`中。但由于在创建 YAML 发射器时对`miAssimpInstances`向量的线性读取以及解析 YAML 文件时对节点的相同线性读取，实例应保持其在保存时的相同索引。

最后，我们从解析器中恢复相机设置、选定的模型、实例以及选择高亮的状态。

到目前为止，配置应该已经完全加载，应用程序应包含与保存操作时相同的模型、实例和设置。

对于加载过程，还有一个问题：如果配置文件解析只部分失败，我们应该怎么办？可能是一个模型文件被重命名或删除，或者文件被截断或损坏，导致最后一个实例的设置不完整。

## 严格或宽松配置文件加载

一种克服解析错误的方法是在删除应用程序当前所有内容之前丢弃整个配置。这种严格的加载类型易于实现；任何类型的解析错误都会在解析时使配置文件无效。我们忽略加载请求，并向用户显示错误信息。

另一个选项是宽松解析。我们尽力加载有效的模型，并用默认值填充缺失的配置部分，同时也会告知用户配置文件的部分内容无法加载。

在这两种情况下，错误信息应该给出详细的提示，说明解析失败的位置。因此，异常处理可以扩展为确切知道出了什么问题，以及在哪里。对于宽松处理，应尽可能向用户展示受影响模型、实例或设置的相关附加信息。

应用程序的创建者决定哪种策略最适合。通常，应尝试尽可能恢复数据。只丢失创建工作的很小一部分比丢失所有数据要好。

## 导致文件损坏的常见错误

几个因素可能导致保存的配置文件损坏。以下列出了一些常见原因：

+   在写入文件时磁盘或分区已满至 100%：即使我们今天有大量的存储空间，这也可能发生，您只能保存部分数据。

+   权限问题：有时，您可能有创建文件的权限，但没有写入文件内容的权限。因此，您的文件看起来已经保存，但文件长度为零字节。

+   保存到远程位置时出现的连接错误：在写入较大文件时，您的连接可能会中断，导致文件只部分写入。

+   转换错误，例如通过电子邮件发送文件：邮件程序或邮件服务器可能会以错误的方式转换文件，导致部分损坏的文件，其中一些字符被替换。

+   不兼容的区域设置：保存文件的机器可能使用逗号作为小数分隔符，而您的计算机使用点作为小数分隔符。文件中的数字可能会被误解，甚至在解析失败时被设置为零。这个问题很难找到，并且很容易被忽视。

+   编程错误，如版本处理错误、转换错误或不完整的错误/异常处理：您可能无法保存所有数据，意外地将数据转换为错误的格式，或者错过解析文件中的某些数据。您应该尽可能多地测试代码的文件读取和写入功能，以找到此类错误。

+   您应该意识到，您的保存文件可能在您的机器上或从您那里到那里的路上损坏。因此，经常保存您的作品，使用版本控制系统如 Git 存储文件的不同版本，并定期备份所有配置文件。

现在我们有了保存和加载应用程序状态的代码，我们可以在应用程序启动时提供一个预定义的默认配置。

# 在启动时加载默认配置文件

为了帮助用户探索新的应用程序，除了广泛的教程外，还可以在第一次启动时或在任何启动时加载使用应用程序创建的内容的简单示例。通过调整可用的选项可以帮助我们了解应用程序的工作方式，以及可能进行的内容操作。

在启动时加载默认配置可以通过不同的方式实现。可以在编译时添加配置文件（*嵌入*到应用程序中），或者在一个可访问的文件夹中放置一个或多个示例文件，并在启动时加载示例文件。通常，应用程序有一个单独的配置设置，可以用来禁用自动加载示例文件。

例如，我们将在应用程序启动时从加载和保存对话框中加载配置文件`config/conf.acfg`。多亏了已经实现的 YAML 解析器和文件加载代码，对渲染类所做的更改只需几行代码即可完成。

首先，我们将默认配置文件定义为渲染类的一个新的`private`成员变量`mDefaultConfigFileName`：

```cpp
 const std::string mDefaultConfigFileName =
      "config/conf.acfg"; 
```

通常应避免硬编码文件路径或文件名，但对于第一个配置文件，我们最终陷入了一个鸡生蛋的问题。如果我们想将默认配置的名称存储在另一个配置文件中，而不是在代码中硬编码文件名，我们需要另一个硬编码的文件名。这种启动问题只能通过硬编码第一个值来解决。

然后，在渲染器的`init()`方法中，我们尝试加载默认配置文件：

```cpp
 if (!loadConfigFile(mDefaultConfigFileName)) {
    addNullModelAndInstance();
  } 
```

如果找不到文件或加载失败，我们只会创建空模型和空实例。由于所有其他值在第一次启动时都设置为默认值，所以我们最终得到的应用程序与完全没有默认配置的情况相同。

将加载和保存功能实现到应用程序中需要一些研究来确定合适的保存文件类型，并且还需要更多的工作来实现这些功能到现有代码中。所有更改和新功能都应该反映在应用程序的保存文件中，因此需要更多的工作来保持加载和保存代码与应用程序功能的更新同步。通过在配置文件中添加版本控制方案，我们甚至能够从应用程序的不同开发阶段加载配置。

# 摘要

在本章中，我们添加了将应用程序当前配置保存到文件并重新加载相同配置到应用程序中的功能。首先，我们评估了二进制和文本保存文件的优缺点，并检查了三种常见的文本文件类型，以找到适合我们保存文件的格式。接下来，我们探讨了选择的 YAML 文件格式并实现了保存和加载功能。最后，我们添加了一个默认文件，在应用程序启动时加载，以帮助用户处理应用程序的第一步。

在下一章中，我们将处理应用程序中的自定义相机。目前，我们只使用 *内部* 相机在虚拟世界中飞行。通过添加自定义相机类型，可以为虚拟世界提供更多的可视化选项。我们将添加一个第三人称风格的相机，类似于动作游戏中的跟随一个实例，以及一个固定相机，它跟随一个实例。此外，还将添加一个简单的相机管理器，相机的配置也将保存在配置文件中。

# 实践课程

您可以在代码中添加一些内容：

+   添加一个菜单项以创建一个新的空场景。

目前，我们只能加载和保存配置文件。移除所有模型和实例仍然需要手动完成。添加一个菜单项和代码，以便一次性移除所有模型和实例，为用户提供一种简单的方法从头开始。

+   如果更改了设置，添加一个标志和确认对话框。

如果更改了一个模型的设置，设置一个 `dirty` 标志以记住应用程序用户更改了加载的模型实例或保存的状态。然后，如果用户想要加载另一个配置文件，可以从一个空配置开始，或者退出应用程序并显示一个确认对话框以确保有机会保存当前设置。

+   在标题中添加一个 `dirty` 标记。

几个其他应用程序向用户显示了一些通知，说明自上次保存以来已进行了更改。应用程序窗口的标题会相应调整以显示我们是在编辑模式还是查看模式，因此向窗口标题添加一个星号 (*) 或一些像“未保存”这样的词应该很容易。

# 其他资源

这里是 `yaml-cpp` 的 GitHub 仓库：[`github.com/jbeder/yaml-cpp`](https://github.com/jbeder/yaml-cpp)。

# 加入我们的 Discord 社区

加入我们社区的 Discord 空间，与作者和其他读者进行讨论：[`packt.link/cppgameanimation`](https://packt.link/cppgameanimation)

![带有黑色方块的二维码，AI 生成的内容可能不正确。](img/QR_code_Discord.png)
