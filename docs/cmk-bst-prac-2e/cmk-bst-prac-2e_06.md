# 第四章：打包、部署和安装 CMake 项目

正确地打包软件往往被编写和构建软件的过程所掩盖，然而它通常是确保任何软件项目成功和持久性的一个重要因素。打包是开发者创作与最终用户体验之间的桥梁，涵盖了从分发到安装和维护的方方面面。打包得当的软件不仅简化了部署过程，还增强了用户满意度、可靠性，并且便于无缝更新和修复漏洞。

确保软件以与这些不同环境兼容的方式打包，对其可用性和可访问性至关重要。此外，用户的技术能力跨度广泛，从经验丰富的专业人士到新手不等。因此，打包必须迎合这一范围，为经验较少的用户提供直观的安装过程，同时为技术熟练的用户提供高级选项。此外，遵守安装标准对于用户的便利性和系统完整性都至关重要。通过遵循既定的打包规范，开发者可以减少在目标系统中堆积不必要的文件或冲突的依赖关系，从而促进系统的稳定性和整洁性。归根结底，软件打包是将原始代码转化为精致、可访问产品的关键最后一步，和开发过程本身一样至关重要。

CMake 内部有良好的支持和工具，使得安装和打包变得简单。这一点的好处在于，CMake 利用现有的项目代码来实现这些功能。因此，使项目可安装或打包项目不会带来沉重的维护成本。本章中，我们将学习如何利用 CMake 在安装和打包方面的现有能力，来支持部署工作。

本章将涵盖以下主题：

+   使 CMake 目标可安装

+   使用你的项目为他人提供配置信息

+   使用 CPack 创建可安装包

# 技术要求

在深入本章之前，你应该对 CMake 中的目标有一个良好的理解（在*第一章*《启动 CMake》和*第三章*《创建 CMake 项目》中简要介绍，详细内容见其中）。本章将基于这些知识进行扩展。

请从本书的 GitHub 仓库获取本章的示例，地址为 [`github.com/PacktPublishing/CMake-Best-Practices---2nd-Edition`](https://github.com/PacktPublishing/CMake-Best-Practices---2nd-Edition)。本章的示例内容位于 `chapter04/` 子文件夹中。

# 使 CMake 目标可安装

在 CMake 的上下文中，*安装*和*打包*软件是两个不同的概念。安装软件涉及将可执行文件、库和其他必要资源复制到预定的位置。而打包则是将所有必要的文件和依赖项捆绑成一个可分发格式（例如 tarball、ZIP 压缩包或安装程序包），以便于在其他系统上进行分发和安装。CMake 的打包机制是先将项目安装到临时位置，然后将安装的文件打包成适当的格式。

支持项目部署的最原始方式是将其设置为*可安装*。相反，最终用户仍然必须获取项目的源代码并从头开始构建它。一个可安装的项目会有额外的构建系统代码，用于在系统上安装运行时或开发工件。如果有适当的安装指令，构建系统将在这里执行安装操作。由于我们使用 CMake 生成构建系统文件，因此 CMake 必须生成相关的安装代码。在这一节中，我们将深入探讨如何指导 CMake 为 CMake 目标生成此类代码的基础知识。

## `install()`命令

`install(...)`命令是一个内置的 CMake 命令，允许您生成安装目标、文件、目录等的构建系统指令。CMake 不会生成安装指令，除非明确告诉它这么做。因此，什么内容被安装始终在您的控制之下。让我们来看一下它的基本用法。

### 安装 CMake 目标

要使 CMake 目标可安装，必须指定`TARGETS`参数，并提供至少一个参数。该用法的命令签名如下：

```cpp
install(TARGETS <target>... [...])
```

`TARGETS`参数表示`install`将接受一组 CMake 目标，生成安装代码。在这种形式下，只有目标的输出工件会被安装。目标的最常见输出工件定义如下：

+   `ARCHIVE`（静态库、DLL 导入库和链接器导入文件）：

    +   除了在 macOS 中标记为`FRAMEWORK`的目标

+   `LIBRARY`（共享库）：

    +   除了在 macOS 中标记为`FRAMEWORK`的目标

    +   除了 DLL（在 Windows 中）

+   `RUNTIME`（可执行文件和 DLL）：

    +   除了在 macOS 中标记为`MACOSX_BUNDLE`的目标

在将目标设置为可安装后，CMake 会生成必要的安装代码，以便安装为该目标生成的输出工件。为了说明这一点，让我们一起将一个基本的可执行目标设置为可安装。要查看`install(...)`命令的实际操作，我们可以查看位于`chapter04/ex01_executable`文件夹中的`Chapter 4`，`example 1`的`CMakeLists.txt`文件：

```cpp
add_executable(ch4_ex01_executable)
target_sources(ch4_ex01_executable src/main.cpp)
target_compile_features(ch4_ex01_executable PRIVATE cxx_std_11)
install(TARGETS ch4_ex01_executable)
```

在前面的代码中，定义了一个名为`ch4_ex01_executable`的可执行目标，并在接下来的两行中填充了它的属性。最后一行`install(...)`是我们感兴趣的部分，它告诉 CMake 为`ch4_ex01_executable`创建所需的安装代码。

为了检查`ch4_ex01_executable`是否可以被安装，让我们在`chapter 4`的根文件夹中通过 CLI 构建并安装该项目：

```cpp
cmake -S . -B ./build -DCMAKE_BUILD_TYPE="Release"
cmake --build ./build
cmake --install ./build --prefix /tmp/install-test
```

注意

与其为`cmake --install`指定`--prefix`参数，你也可以使用`CMAKE_INSTALL_PREFIX`变量来提供非默认的`install`前缀。

在使用 CMake 与多配置生成器（如 Ninja 多配置和 Visual Studio）时，请为`cmake --build`和`cmake --install`命令指定`--config`参数：

```cpp
# For multi-config generators:
cmake --build ./build --config Release
cmake --install ./build --prefix /tmp/install-test --config Debug
```

让我们检查一下`cmake --install`命令的作用：

```cpp
-- Install configuration: "Release"
-- Installing: /tmp/install-test/lib/libch2.framework.component1.a
-- Installing: /tmp/install-test/lib/libch2.framework.component2.so
-- Installing: /tmp/install-test/bin/ch2.driver_application
-- Set runtime path of "/tmp/install-test/bin/
    ch2.driver_application" to ""
-- Installing: /tmp/install-test/bin/ch4_ex01_executable
```

在前面输出的最后一行中，我们可以看到`ch4_ex01_executable`目标的*输出工件*——也就是说，`ch4_ex01_executable`二进制文件已经被安装。由于这是`ch4_ex01_executable`目标的唯一输出工件，我们可以得出结论，目标确实已经变得可以安装了。

请注意，`ch4_ex01_executable`并没有直接安装到`/tmp/install-test`（前缀）目录中。相反，`install`命令将它放入了`bin/`子目录。这是因为 CMake 智能地判断了应该将什么类型的工件放到哪里。在传统的 UNIX 系统中，二进制文件通常放在`/usr/bin`，而库文件放在`/usr/lib`。CMake 知道`add_executable()`命令会生成一个可执行的二进制工件，并将其放入`/bin`子目录。这些目录是 CMake 默认提供的，具体取决于目标类型。提供默认安装路径信息的 CMake 模块被称为`GNUInstallDirs`模块。`GNUInstallDirs`模块在被包含时定义了各种`CMAKE_INSTALL_`路径。下表显示了各个目标的默认安装目录：

| **目标类型** | **GNUInstallDirs 变量** | **内置默认值** |
| --- | --- | --- |
| RUNTIME | ${CMAKE_INSTALL_BINDIR} | bin |
| LIBRARY | ${CMAKE_INSTALL_LIBDIR} | lib |
| ARCHIVE | ${CMAKE_INSTALL_LIBDIR} | lib |
| PRIVATE_HEADER | ${CMAKE_INSTALL_INCLUDEDIR} | include |
| PUBLIC_HEADER | ${CMAKE_INSTALL_INCLUDEDIR} | include |

为了覆盖内置的默认值，`install(...)`命令中需要一个额外的`<TARGET_TYPE> DESTINATION`参数。为了说明这一点，假设我们要将默认的`RUNTIME`安装目录更改为`qbin`，而不是`bin`。这样做只需要对原始的`install(...)`命令做一个小的修改：

```cpp
# …
install(TARGETS ch4_ex01_executable
        RUNTIME DESTINATION qbin
)
```

做出此更改后，我们可以重新运行 `configure`、`build` 和 `install` 命令。我们可以通过检查 `cmake --install` 命令的输出确认 `RUNTIME` 目标已经更改。与第一次不同，我们可以观察到 `ch4_ex01_executable` 二进制文件被放入 `qbin` 而不是默认的 (`bin`) 目录：

```cpp
# ...
-- Installing: /tmp/install-test/qbin/ch4_ex01_executable
```

现在，让我们看另一个示例。这次我们将安装一个 `STATIC` 库。让我们看看 *第四章* 中的 `CMakeLists.txt` 文件，*示例 2*，它位于 `chapter04/ex02_static` 文件夹中。由于篇幅原因，注释和 `project(...)` 命令已被省略。让我们开始检查文件：

```cpp
add_library(ch4_ex02_static STATIC)
target_sources(ch4_ex02_static PRIVATE src/lib.cpp)
target_include_directories(ch4_ex02_static PUBLIC include)
target_compile_features(ch4_ex02_static PRIVATE cxx_std_11)
include(GNUInstallDirs)
install(TARGETS ch4_ex02_static)
install (
     DIRECTORY include/
     DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)
```

如你所见，它与我们之前的示例稍有不同。首先，新增了一个带有 `DIRECTORY` 参数的 `install(...)` 命令。这是为了使静态库的头文件可以被安装。原因是 CMake 不会安装任何不是 *输出产物* 的文件，而 `STATIC` 库目标只会生成一个二进制文件作为 *输出产物*。头文件不是 *输出产物*，应单独安装。

注意

`DIRECTORY` 参数中的尾随斜杠会导致 CMake 复制文件夹内容，而不是按名称复制文件夹。CMake 处理尾随斜杠的方式与 Linux 的 `rsync` 命令相同。

### 安装文件和目录

如我们在前一节中看到的，我们打算安装的内容并不总是目标的 *输出产物*。它们可能是目标的运行时依赖项，例如图像、资源、配置文件、脚本和资源文件。CMake 提供了 `install(FILES...)` 和 `install(DIRECTORY...)` 命令，用于安装任何特定的文件或目录。让我们从安装文件开始。

#### 安装文件

`install(FILES...)` 命令接受一个或多个文件作为参数。它还需要一个额外的 `TYPE` 或 `DESTINATION` 参数。这两个参数用于确定指定文件的目标目录。`TYPE` 参数用于指示哪些文件将使用该文件类型的默认路径作为安装目录。通过设置相关的 `GNUInstallDirs` 变量可以覆盖默认值。以下表格显示了有效的 `TYPE` 值及其目录映射：

| **类型** | **GNUInstallDirs 变量** | **内置默认值** |
| --- | --- | --- |
| BIN | ${CMAKE_INSTALL_BINDIR} | bin |
| SBIN | ${CMAKE_INSTALL_SBINDIR} | sbin |
| LIB | ${CMAKE_INSTALL_LIBDIR} | lib |
| INCLUDE | ${CMAKE_INSTALL_INCLUDEDIR} | include |
| SYSCONF | ${CMAKE_INSTALL_SYSCONFDIR} | etc |
| SHAREDSTATE | ${CMAKE_INSTALL_SHARESTATEDIR} | com |
| LOCALSTATE | ${CMAKE_INSTALL_LOCALSTATEDIR} | var |
| RUNSTATE | ${CMAKE_INSTALL_RUNSTATEDIR} | <LOCALSTATE dir>/run |
| DATA | ${CMAKE_INSTALL_DATADIR} | <DATAROOT dir> |
| INFO | ${CMAKE_INSTALL_INFODIR} | <DATAROOT dir>/info |
| LOCALE | ${CMAKE_INSTALL_LOCALEDIR} | <DATAROOT dir>/locale |
| MAN | ${CMAKE_INSTALL_MANDIR} | <DATAROOT dir>/man |
| DOC | ${CMAKE_INSTALL_DOCDIR} | <DATAROOT dir>/doc |

如果你不想使用 `TYPE` 参数，可以改用 `DESTINATION` 参数。它允许你为 `install(...)` 命令中指定的文件提供自定义的目标位置。

`install(FILES...)` 的一种替代形式是 `install(PROGRAMS...)`，它与 `install(FILES...)` 相同，区别在于它还为已安装的文件设置了 `OWNER_EXECUTE`、`GROUP_EXECUTE` 和 `WORLD_EXECUTE` 权限。对于必须由最终用户执行的二进制文件或脚本文件来说，这样做是有意义的。

要理解 `install(FILES|PROGRAMS...)`，让我们看一个例子。我们将要查看的例子是 *第四章**，示例 3*（`chapter04/ex03_file`）。它实际上包含了三个文件：`chapter04_greeter_content`、`chapter04_greeter.py` 和 `CMakeLists.txt`。首先，让我们看看它的 `CMakeLists.txt` 文件：

```cpp
install(FILES "${CMAKE_CURRENT_LIST_DIR}/chapter04_greeter_content"
  DESTINATION "${CMAKE_INSTALL_BINDIR}")
install(PROGRAMS "${CMAKE_CURRENT_LIST_DIR}/chapter04_greeter.py"
  DESTINATION "${CMAKE_INSTALL_BINDIR}" RENAME chapter04_greeter)
```

让我们消化一下我们所看到的内容；在第一个 `install(...)` 命令中，我们告诉 CMake 将 `chapter04_greeter_content` 文件安装到当前 `CMakeLists.txt` 目录（`chapter04/ex03_file`）的系统默认 `BIN` 目录中。在第二个 `install(...)` 命令中，我们告诉 CMake 将 `chapter04_greeter.py` 文件安装到默认的 `BIN` 目录，并且文件名为 `chapter04_greeter`。

注意

`RENAME` 参数仅在单文件 `install(...)` 调用时有效。

使用这些 `install(...)` 指令，CMake 应该会将 `chapter04_greeter.py` 和 `chapter04_greeter_content` 文件安装到 `${CMAKE_INSTALL_PREFIX}/bin` 目录。让我们通过 CLI 构建并安装项目：

```cpp
cmake -S . -B ./build
cmake --build ./build
cmake --install ./build --prefix /tmp/install-test
```

让我们看看 `cmake --install` 命令做了什么：

```cpp
/* … */
-- Installing: /tmp/install-test/bin/chapter04_greeter_content
-- Installing: /tmp/install-test/bin/chapter04_greeter
```

上面的输出确认了 CMake 为 `chapter04_greeter_content` 和 `chapter04_greeter.py` 文件生成了所需的安装代码。最后，让我们检查一下 `chapter04_greeter` 文件是否能够执行，因为我们使用了 `PROGRAMS` 参数来安装它：

```cpp
15:01 $ /tmp/install-test/bin/chapter04_greeter
['Hello from installed file!']
```

这样，我们就完成了 `install(FILES|PROGRAMS...)` 部分的内容。接下来，让我们继续安装目录。

#### 安装目录

`install(DIRECTORY...)` 命令对于安装目录非常有用。目录的结构将会被原样复制到目标位置。目录可以作为整体安装，也可以选择性地安装。让我们先从最基本的目录安装示例开始：

```cpp
install(DIRECTORY dir1 dir2 dir3 TYPE LOCALSTATE)
```

上面的例子将会把 `dir1` 和 `dir2` 目录安装到 `${CMAKE_INSTALL_PREFIX}/var` 目录中，并且连同它们的所有子文件夹和文件一起原样安装。有时候，安装文件夹的全部内容并不理想。幸运的是，CMake 允许 `install` 命令根据通配符模式和正则表达式来包含或排除目录内容。让我们这次选择性地安装 `dir1`、`dir2` 和 `dir3`：

```cpp
include(GNUInstallDirs)
install(DIRECTORY dir1 DESTINATION ${CMAKE_INSTALL_LOCALSTATEDIR}
  FILES_MATCHING PATTERN "*.x")
install(DIRECTORY dir2 DESTINATION ${CMAKE_INSTALL_LOCALSTATEDIR}
  FILES_MATCHING PATTERN "*.hpp" EXCLUDE PATTERN "*")
install(DIRECTORY dir3 DESTINATION ${CMAKE_INSTALL_LOCALSTATEDIR}
  PATTERN "bin" EXCLUDE)
```

在前面的示例中，我们使用了`FILES_MATCHING`参数来定义文件选择的标准。`FILES_MATCHING`后面可以跟`PATTERN`或`REGEX`参数。`PATTERN`允许您定义一个通配符模式，而`REGEX`允许您定义一个正则表达式。默认情况下，这些表达式用于包含文件。如果要排除符合标准的文件，可以在模式后添加`EXCLUDE`参数。请注意，这些过滤器不会应用于子目录名称，因为`FILES_MATCHING`参数的限制。我们还在最后一个`install(...)`命令中使用了`PATTERN`而没有加上`FILES_MATCHING`，这使得我们可以过滤子目录而非文件。这一次，只有`dir1`中扩展名为`.x`的文件、`dir2`中没有`.hpp`扩展名的文件以及`dir3`中除`bin`文件夹外的所有内容将被安装。这个示例可以在`chapter04/ex04_directory`文件夹中的*Chapter 4**，示例 4*中找到。让我们编译并安装它，看看它是否执行了正确的操作：

```cpp
cmake -S . -B ./build
cmake -- build ./build
cmake -- install ./build –prefix /tmp/install-test
```

`cmake --install`的输出应该如下所示：

```cpp
-- Installing: /tmp/install-test/var/dir1
-- Installing: /tmp/install-test/var/dir1/subdir
-- Installing: /tmp/install-test/var/dir1/subdir/asset5.x
-- Installing: /tmp/install-test/var/dir1/asset1.x
-- Installing: /tmp/install-test/var/dir2
-- Installing: /tmp/install-test/var/dir2/chapter04_hello.dat
-- Installing: /tmp/install-test/var/dir3
-- Installing: /tmp/install-test/var/dir3/asset4
```

注意

`FILES_MATCHING`不能在`PATTERN`或`REGEX`之后使用，但可以反过来使用。

在输出中，我们可以看到只有扩展名为`.x`的文件被从`dir1`中选取。这是因为在第一个`install(...)`命令中使用了`FILES_MATCHING PATTERN "*.x"`参数，导致`asset2`文件没有被安装。同时，注意到`dir2/chapter04_hello.dat`文件被安装，而`dir2/chapter04_hello.hpp`文件被跳过。这是因为第二个`install(…)`命令中的`FILES_MATCHING PATTERN "*.hpp" EXCLUDE PATTERN "*"`参数所致。最后，我们看到`dir3/asset4`文件被安装，而`dir3/bin`目录被完全跳过，因为在最后一个`install(...)`命令中指定了`PATTERN "bin" EXCLUDE`参数。

使用`install(DIRECTORY...)`时，我们已经涵盖了`install(...)`命令的基础知识。接下来，让我们继续了解`install(…)`命令的其他常见参数。

### `install()`命令的其他常见参数

如我们所见，`install()`命令的第一个参数指示要安装的内容。还有一些额外的参数可以让我们定制安装过程。让我们一起查看一些常见的参数。

#### DESTINATION 参数

该参数允许你为 `install(...)` 命令中指定的文件指定目标目录。目录路径可以是相对路径或绝对路径。相对路径将相对于 `CMAKE_INSTALL_PREFIX` 变量。建议使用相对路径以使安装*可重定位*。此外，为了打包，使用相对路径也很重要，因为 `cpack` 要求安装路径必须是相对的。最好使用以相关的 `GNUInstallDirs` 变量开头的路径，这样包维护者可以根据需要覆盖安装目标位置。`DESTINATION` 参数可以与 `TARGETS`、`FILES`、`IMPORTED_RUNTIME_ARTIFACTS`、`EXPORT` 和 `DIRECTORY` 安装类型一起使用。

#### PERMISSIONS 参数

该参数允许你在支持的平台上更改已安装文件的权限。可用的权限有 `OWNER_READ`、`OWNER_WRITE`、`OWNER_EXECUTE`、`GROUP_READ`、`GROUP_WRITE`、`GROUP_EXECUTE`、`WORLD_READ`、`WORLD_WRITE`、`WORLD_EXECUTE`、`SETUID` 和 `SETGID`。`PERMISSIONS` 参数可以与 `TARGETS`、`FILES`、`IMPORTED_RUNTIME_ARTIFACTS`、`EXPORT` 和 `DIRECTORY` 安装类型一起使用。

#### CONFIGURATIONS 参数

这允许你在指定特定构建配置时限制应用的参数集。

#### OPTIONAL 参数

该参数使得文件的安装变为可选，这样当文件不存在时，安装不会失败。`OPTIONAL` 参数可以与 `TARGETS`、`FILES`、`IMPORTED_RUNTIME_ARTIFACTS` 和 `DIRECTORY` 安装类型一起使用。

在本节中，我们学习了如何使目标、文件和目录可安装。在下一节中，我们将学习如何生成配置信息，以便可以直接将 CMake 项目导入到另一个 CMake 项目中。

# 为他人提供项目的配置信息

在上一节中，我们学习了如何使我们的项目可安装，以便他人可以通过安装它到他们的系统中来使用我们的项目。但有时候，仅仅交付制品并不足够。例如，如果你交付的是一个库，它必须也能方便地导入到另一个项目中——尤其是 CMake 项目中。在本节中，我们将学习如何让其他 CMake 项目更容易导入你的项目。

如果被导入的项目具有适当的配置文件，则有一些便捷的方法可以导入库。一个突出的方式是利用 `find_package()` 方法（我们将在*第五章*中讲解，*集成第三方库* 和 *依赖管理*）。如果你的消费者在工作流程中使用 CMake，他们会很高兴能够直接写 `find_package(your_project_name)`，并开始使用你的代码。在本节中，我们将学习如何生成所需的配置文件，以使 `find_package()` 能在你的项目中正常工作。

CMake 推荐的依赖管理方式是通过包（packages）。包用于传递 CMake 基于构建系统的依赖信息。包可以是 `Config-file` 包、`Find-module` 包或 `pkg-config` 包的形式。所有这些包类型都可以通过 `find_package()` 查找并使用。为了提高效率并遵循最佳实践，本节将仅关注 `Config-file` 包。其他方法，如 `find-modules` 和 `pkg-config` 包，通常被视为过时的变通方法，主要在没有配置文件的情况下使用，通常不推荐使用。让我们深入了解 `Config-file` 包，理解它们的优点和实现方式。

## 进入 CMake 包的世界 —— `Config-file` 包

`Config-file` 包基于包含包内容信息的配置文件。这些信息指示包的内容位置，因此 CMake 会读取此文件并使用该包。因此，仅发现包的配置文件就足够使用该包了。

配置文件有两种类型 —— 包配置文件和可选的包版本文件。两个文件都必须遵循特定的命名约定。包配置文件可以命名为 `<ProjectName>Config.cmake` 或 `<projectname>-config.cmake`，具体取决于个人偏好。在 `find_package(ProjectName)`/`find_package(projectname)` 调用时，CMake 会自动识别这两种命名方式。包配置文件的内容大致如下：

```cpp
set(Foo_INCLUDE_DIRS ${PREFIX}/include/foo-1.2)
set(Foo_LIBRARIES ${PREFIX}/lib/foo-1.2/libfoo.a)
```

在这里，`${PREFIX}` 是项目的安装前缀。它是一个变量，因为安装前缀可以根据系统类型进行更改，也可以由用户更改。

和包配置文件一样，包版本文件也可以命名为 `<ProjectName>ConfigVersion.cmake` 或 `<projectname>-config-version.cmake`。CMake 期望在 `find_package(...)` 搜索路径中找到包配置文件和包版本文件。你可以在 CMake 的帮助下创建这些文件。`find_package(...)` 在查找包时会检查多个位置，其中之一就是 `<CMAKE_PREFIX_PATH>/cmake` 目录。在我们的例子中，我们将把 `config-file` 包配置文件放到这个文件夹中。

为了创建 `config-file` 包，我们需要了解一些额外的内容，例如 `CmakePackageConfigHelpers` 模块。为了了解这些内容，让我们开始深入探讨一个实际的例子。我们将跟随 *第四章**，示例 5* 来学习如何构建一个 CMake 项目，将其组织成 `chapter04/ex05_config_file_package` 文件夹。首先，让我们检查 `chapter04/ex05_config_file_package` 目录中的 `CMakeLists.txt` 文件（注释和项目命令已省略以节省空间；另外，请注意，所有与主题无关的行将不被提及）：

```cpp
include(GNUInstallDirs)
set(ch4_ex05_lib_INSTALL_CMAKEDIR cmake CACHE PATH "Installation
  directory for config-file package cmake files")Is
```

`CMakeLists.txt` 文件与 `chapter04/ex02_static` 非常相似。这是因为它是同一个示例，只是它支持 `config-file` 包。第一行 `include(GNUInstallDirs)` 用于包含 `GNUInstallDirs` 模块。这个模块提供了 `CMAKE_INSTALL_INCLUDEDIR` 变量，稍后会用到。`set(ch4_ex05_lib_INSTALL_CMAKEDIR...)` 是一个用户定义的变量，用于设置 `config-file` 打包配置文件的目标安装目录。它是一个相对路径，应在 `install(…)` 指令中使用，因此它隐式地是相对于 `CMAKE_INSTALL_PREFIX` 的：

```cpp
target_include_directories(ch4_ex05_lib PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_compile_features(ch4_ex05_lib PUBLIC cxx_std_11)
```

`target_include_directories(...)` 调用与通常的调用非常不同。它使用了 `generator expressions` 来区分构建时的 `include` 目录和安装时的 `include` 目录，因为构建时的 `include` 路径在目标被导入到另一个项目时将不存在。以下一组命令将使目标可安装：

```cpp
install(TARGETS ch4_ex05_lib
        EXPORT ch4_ex05_lib_export
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install (
      DIRECTORY ${PROJECT_SOURCE_DIR}/include/
      DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
```

`install(TARGETS...)` 与常规调用稍有不同。它包含了一个额外的 `EXPORT` 参数。这个 `EXPORT` 参数用于从给定的 `install(…)` 目标创建一个导出名称。然后可以使用这个导出名称来导出这些目标。通过 `INCLUDES DESTINATION` 参数指定的路径将用于填充导出目标的 `INTERFACE_INCLUDE_DIRECTORIES` 属性，并会自动加上安装前缀路径。在这里，`install(DIRECTORY...)` 命令用于安装目标的头文件，这些文件位于 `${PROJECT_SOURCE_DIR}/include/`，并安装到 `${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}` 目录下。`${CMAKE_INSTALL_INCLUDEDIR}` 变量用于为用户提供覆盖此安装的 `include` 目录的能力。现在，让我们从之前示例中创建一个导出文件：

```cpp
install(EXPORT ch4_ex05_lib_export
        FILE ch4_ex05_lib-config.cmake
        NAMESPACE ch4_ex05_lib::
        DESTINATION ${ch4_ex05_lib_INSTALL_CMAKEDIR}
)
```

`install(EXPORT...)` 可能是这个文件中最重要的代码部分。它执行实际的目标导出。它生成一个包含所有导出目标的 CMake 文件，并使用给定的导出名称。`EXPORT` 参数接受一个现有的导出名称来执行导出。它引用了我们之前通过 `install(TARGETS...)` 调用创建的 `ch4_ex05_lib_export` 导出名称。`FILE` 参数用于确定导出的文件名，并设置为 `ch4_ex05_lib-config.cmake`。`NAMESPACE` 参数用于给所有导出的目标添加前缀命名空间。这使得你可以将所有导出的目标放在一个公共的命名空间下，避免与其他有相似目标名称的包发生冲突。最后，`DESTINATION` 参数确定了生成的导出文件的安装路径。它设置为 `${ch4_ex05_lib_INSTALL_CMAKEDIR}`，以便 `find_package()` 可以找到它。

注意

由于我们除了导出的目标之外不提供任何额外内容，因此导出文件的名称是`ch4_ex05_lib-config.cmake`。这是此包所需的包配置文件名称。我们这样做是因为示例项目不需要先满足任何额外的依赖关系，可以直接按原样导入。如果需要任何额外的操作，建议先创建一个中间包配置文件，以满足这些依赖关系，然后再包含导出的文件。

使用`install(EXPORT...)`命令，我们获得了`ch4_ex05_lib-config.cmake`文件。这意味着我们的目标可以通过`find_package(..)`来使用。为了实现对`find_package(…)`的完全支持，还需要执行一个额外步骤，即获取`ch4_ex05_lib-config-version.cmake`文件：

```cpp
/*…*/
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "ch4_ex05_lib-config-version.cmake"
  # Package compatibility strategy. SameMajorVersion is essentially
    `semantic versioning`.
  COMPATIBILITY SameMajorVersion
)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/ch4_ex05_lib-config-version.cmake"
  DESTINATION «${ch4_ex05_lib_INSTALL_CMAKEDIR}»
)
/* end of the file */
```

在最后几行中，您可以找到生成并安装`ch4_ex05_lib-config-version.cmake`文件所需的代码。通过`include(CMakePackageConfigHelpers)`这一行，导入了`CMakePackageConfigHelpers`模块。该模块提供了`write_basic_package_version_file(…)`函数。`write_basic_package_version_file(…)`函数用于根据给定的参数自动生成包版本文件。第一个位置参数是输出文件的文件名。`VERSION`参数用于指定我们正在生成的包的版本，格式为`major.minor.patch`。我们选择不显式指定版本，以允许`write_basic_package_version_file`自动从项目版本中获取。`COMPATIBILITY`参数允许根据版本值指定兼容性策略。`SameMajorVersion`表示该包与任何具有相同主版本号的版本兼容。其他可能的值包括`AnyNewerVersion`、`SameMinorVersion`和`ExactVersion`。

现在，让我们测试一下这个是否有效。为了测试包配置，我们必须以常规方式安装项目：

```cpp
cmake -S . -B ./build
cmake --build ./build
cmake --install ./build --prefix /tmp/install-test
```

`cmake --install`命令的输出应如下所示：

```cpp
-- Installing: /tmp/install-test/cmake/ch4_ex05_lib-config.cmake
-- Installing: /tmp/install-test/cmake/ch4_ex05_lib-config-
  noconfig.cmake
-- Installing: /tmp/install-test/cmake/ch4_ex05_lib-config-
  version.cmake
```

在这里，我们可以看到我们的包配置文件已成功安装到`/tmp/install-test/cmake`目录中。检查这些文件的内容作为练习留给您自己。所以，现在我们手头有一个可消费的包。让我们换个角度，尝试消费我们新创建的包。为此，我们将查看`chapter04/ex05_consumer`示例。让我们一起检查`CMakeLists.txt`文件：

```cpp
if(NOT PROJECT_IS_TOP_LEVEL)
  message(FATAL_ERROR "The chapter-4, ex05_consumer project is
    intended to be a standalone, top-level project. Do not include
      this directory.")
endif()
find_package(ch4_ex05_lib 1 CONFIG REQUIRED)
add_executable(ch4_ex05_consumer src/main.cpp)
target_compile_features(ch4_ex05_consumer PRIVATE cxx_std_11)
target_link_libraries(ch4_ex05_consumer ch4_ex05_lib::ch4_ex05_lib)
```

在前几行中，我们可以看到关于该项目是否是顶级项目的验证。由于这个示例旨在作为外部应用程序，它不应成为根示例项目的一部分。因此，我们可以保证使用由软件包导出的目标，而不是根项目的目标。根项目也不包括`ex05_consumer`文件夹。接下来，有一个`find_package(…)`调用，其中`ch4_ex05_lib`作为软件包名称给出。还明确要求该软件包的主版本为 1；`find_package(…)`只能考虑`CONFIG`软件包，并且此`find_package(…)`调用中指定的软件包是必需的。在接下来的几行中，定义了一个常规可执行文件`ch4_ex05_consumer`，它在`ch4_ex05_lib`命名空间下链接到`ch4_ex05_lib`（`ch4_ex05_lib::ch4_ex05_lib`）。`ch4_ex05_lib::ch4_ex05_lib`就是我们在软件包中定义的实际目标。让我们来看一下源文件`src/main.cpp`：

```cpp
#include <chapter04/ex05/lib.hpp>
int main(void){
    chapter04::ex05::greeter g;
    g.greet();
}
```

这是一个简单的应用程序，它包括`chapter04/ex05/lib.hpp`，创建一个`greeter`类的实例，并调用`greet()`函数。让我们尝试编译并运行该应用程序：

```cpp
cd chapter04/ex05_consumer
cmake -S . -B build/ -DCMAKE_PREFIX_PATH:STRING=/tmp/install-test
cmake --build build/
./build/ch4_ex05_consumer
```

由于我们已经使用自定义前缀（`/tmp/install-test`）安装了软件包，我们可以通过设置`CMAKE_PREFIX_PATH`变量来指示这一点。这将使得`find_package(…)`在`/tmp/install-test`中也查找软件包。对于默认前缀安装，此参数设置是不可选的。如果一切顺利，我们应该看到臭名昭著的`Hello, world!`消息：

```cpp
 ./build/ch4_ex05_consumer
Hello, world!
```

在这里，我们的消费者可以使用我们的小**欢迎程序**，每个人都很高兴。现在，让我们通过学习如何使用**CPack**打包来结束这一部分。

# 使用 CPack 创建可安装软件包

到目前为止，我们已经看到了 CMake 如何构建软件项目。尽管 CMake 是这场演出的主角，但它也有一些强大的朋友。现在是时候向你介绍 CPack——CMake 的打包工具了。它默认与 CMake 一起安装。它允许你利用现有的 CMake 代码生成特定平台的安装包。CPack 的概念类似于 CMake。它基于生成器，这些生成器生成的是软件包而非构建系统文件。下表展示了截至版本 3.21.3 的可用 CPack 生成器类型：

| **生成器名称** | **描述** |
| --- | --- |
| 7Z | 7-zip 压缩档案 |
| DEB | Debian 软件包 |
| External | CPack 外部软件包 |
| IFW | Qt 安装程序框架 |
| NSIS | Null Soft 安装程序 |
| NSIS64 | Null Soft 安装程序（64 位） |
| NuGet | NuGet 软件包 |
| RPM | RPM 软件包 |
| STGZ | 自解压 TAR gzip 压缩档案 |
| TBZ2 | Tar BZip2 压缩档案 |
| TGZ | Tar GZip 压缩档案 |
| TXZ | Tar XZ 压缩档案 |
| TZ | Tar 压缩档案 |
| TZST | Tar Zstandard 压缩档案 |
| ZIP | Zip 压缩档案 |

CPack 使用 CMake 的安装机制来填充包的内容。CPack 使用位于`CPackConfig.cmake`和`CPackSourceConfig.cmake`文件中的配置详情来生成包。这些文件可以手动填充，也可以通过 CMake 配合 CPack 模块自动生成。对于一个已有的 CMake 项目，使用 CPack 非常简单，只需要包含 CPack 模块，前提是项目已经有正确的`install(…)`命令。包含 CPack 模块会使 CMake 生成`CPackConfig.cmake`和`CPackSourceConfig.cmake`文件，这些文件是打包项目所需的 CPack 配置。此外，还会生成一个额外的`package`目标，用于构建步骤。这个步骤会构建项目并运行 CPack，从而开始打包。当 CPack 配置文件已经正确填充时，无论是通过 CMake 还是用户，CPack 都可以使用。CPack 模块允许你自定义打包过程。可以设置大量的 CPack 变量，这些变量分为两类——通用变量和生成器特定变量。通用变量影响所有包生成器，而生成器特定变量仅影响特定类型的生成器。我们将检查最基本和最显著的变量，主要处理通用变量。以下表格展示了我们将在示例中使用的最常见的 CPack 变量：

| **变量名** | **描述** | **默认值** |
| --- | --- | --- |
| CPACK_PACKAGE_NAME | 包名 | 项目名 |
| CPACK_PACKAGE_VENDOR | 包的供应商名称 | “Humanity” |
| CPACK_PACKAGE_VERSION_MAJOR | 包的主版本 | 项目的主版本 |
| CPACK_PACKAGE_VERSION_MINOR | 包的次版本 | 项目的次版本 |
| CPACK_PACKAGE_VERSION_PATCH | 包的补丁版本 | 项目的补丁版本 |
| CPACK_GENERATOR | 使用的 CPack 生成器列表 | 无 |
| CPACK_THREADS | 支持并行时使用的线程数 | 1 |

必须在包含 CPack 模块之前修改变量，否则将使用默认值。让我们通过一个例子来深入了解 CPack 的实际操作。我们将跟随*第四章*，*示例 6*（`chapter04/ex06_pack`）进行。这个示例是一个独立的项目，不是根项目的一部分。它是一个常规项目，包含名为`executable`和`library`的两个子目录。`executable`目录的`CMakeLists.txt`文件如下所示：

```cpp
add_executable(ch4_ex06_executable src/main.cpp)
target_compile_features(ch4_ex06_executable PRIVATE cxx_std_11)
target_link_libraries(ch4_ex06_executable PRIVATE ch4_ex06_library)
install(TARGETS ch4_ex06_executable)
```

`library`目录的`CMakeLists.txt`文件如下所示：

```cpp
add_library(ch4_ex06_library STATIC src/lib.cpp)
target_compile_features(ch4_ex06_library PRIVATE cxx_std_11)
target_include_directories(ch4_ex06_library PUBLIC include)
set_target_properties(ch4_ex06_library PROPERTIES PUBLIC_HEADER
  include/chapter04/ex06/lib.hpp)
include(GNUInstallDirs) # Defines the ${CMAKE_INSTALL_INCLUDEDIR}
  variable.
install(TARGETS ch4_ex06_library)
install (
    DIRECTORY ${PROJECT_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
```

这些文件夹的`CMakeLists.txt`文件并没有什么特别之处。它们包含常规的可安装 CMake 目标，并且没有声明关于 CPack 的任何内容。让我们也看一下顶级`CMakeLists.txt`文件：

```cpp
cmake_minimum_required(VERSION 3.21)
project(
  ch4_ex06_pack
  VERSION 1.0
  DESCRIPTION "Chapter 4 Example 06, Packaging with CPack"
  LANGUAGES CXX)
if(NOT PROJECT_IS_TOP_LEVEL)
  message(FATAL_ERROR "The chapter04, ex06_pack project is intended
    to be a standalone, top-level project. Do not include this
      directory.")
endif()
add_subdirectory(executable)
add_subdirectory(library)
set(CPACK_PACKAGE_VENDOR "CBP Authors")
set(CPACK_GENERATOR "DEB;RPM;TBZ2")
set(CPACK_THREADS 0)
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "CBP Authors")
include(CPack)
```

顶层的 `CMakeLists.txt` 文件几乎是一个常规的顶层 `CMakeLists.txt` 文件，唯一不同的是最后四行。它设置了三个与 CPack 相关的变量，并引入了 CPack 模块。这四行足以提供基本的 CPack 支持。`CPACK_PACKAGE_NAME` 和 `CPACK_PACKAGE_VERSION_*` 变量没有被设置，让 CPack 从顶层项目的名称和版本参数中推导出来。让我们配置一下项目，看看是否有效：

```cpp
cd chapter04/ex06_pack
cmake –S . -B build/
```

配置项目后，`CpackConfig.cmake` 和 `CpackConfigSource.cmake` 文件应该由 CPack 模块生成，并存放在 `build/CPack*` 目录下。我们来检查一下它们是否存在：

```cpp
$ ls build/CPack*
build/CPackConfig.cmake  build/CPackSourceConfig.cmake
```

在这里，我们可以看到 CPack 配置文件已自动生成。让我们构建一下，并尝试使用 CPack 打包项目：

```cpp
cmake --build build/
cpack --config build/CPackConfig.cmake -B build/
```

`--config` 参数是 CPack 命令的主要输入。`-B` 参数覆盖了 CPack 默认的包目录，指定了它将写入工件的路径。我们来看看 CPack 的输出：

```cpp
CPack: Create package using DEB
/*…*/
CPack: - package: /home/user/workspace/personal/CMake-Best-Practices/chapter04/ex06_pack/build/ch4_ex06_pack-1.0-Linux.deb
generated.
CPack: Create package using RPM
/*…*/
CPack: - package: /home/user/workspace/personal/CMake-Best-Practices/chapter04/ex06_pack/build/ch4_ex06_pack-1.0-Linux.rpm
generated.
CPack: Create package using TBZ2
/*…*/
CPack: - package: /home/user/workspace/personal/CMake-Best-Practices/chapter04/ex06_pack/build/ch4_ex06_pack-1.0-Linux.tar.bz2
generated.
```

在这里，我们可以看到 CPack 使用了 `DEB`、`RPM` 和 `TBZ2` 生成器分别生成了 `ch4_ex06_pack-1.0-Linux.deb`、`ch4_ex06_pack-1.0-Linux.rpm` 和 `ch4_ex06_pack-1.0-Linux.tar.bz2` 包。我们来尝试在 Debian 环境中安装生成的 Debian 包：

```cpp
sudo dpkg -i build/ch4_ex06_pack-1.0-Linux.deb
```

如果打包正确，我们应该能够在命令行中直接调用 `ch4_ex06_executable`：

```cpp
13:38 $ ch4_ex06_executable
Hello, world!
```

成功了！作为练习，试着安装 `RPM` 和 `tar.bz2` 包。处理包文件通常有两种方式。一种是创建小型包，依赖其他包来安装所需的依赖项；另一种方式是创建包含所有必要库的独立安装包，以便独立运行。通常，Linux 发行版自带包管理器来处理这些依赖项，而 Windows 和 macOS 默认依赖独立的安装程序。虽然近年来，Windows 上的 Chocolatey 和 macOS 上的 Homebrew 已成为支持依赖包的流行包管理器，但 CPack 目前（尚未）支持它们。到目前为止，我们只看过需要用户自行安装所有依赖项的简单包。接下来，我们来看一下如何为 Windows 构建一个便于分发的独立包。

## 为 Windows 创建独立安装程序

由于 Windows 并没有自带标准的包管理器，软件的安装程序通常需要包含所有必要的库。一种做法是将预制的安装程序打包成 NSIS 或 WIX 安装包，但这并非总是可行的，所以我们来看一下如何查找依赖文件。为此，CMake 提供了 `install` 命令的可选 `RUNTIME_DEPENDENCIES` 标志和 `InstallRequiredSystemLibraries` 模块，用于查找打包所需的依赖项。

它们的使用方式如下：

```cpp
if(WIN32)
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_INSTALL_DEBUG_LIBRARIES TRUE)
  endif()
  include(InstallRequiredSystemLibraries)
endif()
```

在前面的示例中，包含了 `InstallRequiredSystemLibraries` 模块。该模块是针对 Windows 进行定制的。包含该模块会创建安装编译器提供的库的指令，例如 MSVC 提供的 Visual Studio C++ 运行时库。通过将 `CMAKE_INSTALL_DEBUG_LIBRARIES` 变量设置为 true（如前面示例中所做），可以配置为包括库的调试版本。还有更多选项可以指示 CMake 安装额外的库，例如 Windows MFC 库、OpenMP 或用于在 Windows XP 或更早版本的 Windows 上进行应用本地部署的 Microsoft Universal CRT 库。

模块的完整文档可以在这里找到：[`cmake.org/cmake/help/latest/module/InstallRequiredSystemLibraries.html`](https://cmake.org/cmake/help/latest/module/InstallRequiredSystemLibraries.html)

包括编译器提供的库是一回事，但通常软件项目还会依赖其他库。如果这些库需要与项目一起打包，可以通过 `install()` 命令的 `RUNTIME_DEPENDENCIES` 选项来包含它们，如下所示：

```cpp
# this includes the runtime directories of the executable and the library
install(TARGETS ch4_ex07_executable
        RUNTIME_DEPENDENCIES
        PRE_EXCLUDE_REGEXES "api-ms-.*" "ext-ms-.*"
        POST_EXCLUDE_REGEXES ".*system32/.*\\.dll"
        )
```

这将尝试找出目标指定依赖的任何共享库。由于 Windows 处理 DLL 解析的方式，这很可能会找到比实际需要更多的库。具体来说，它很可能会找到以*api-ms*或*ext-ms*开头的库，这些库是为了兼容性原因存在的，并且并不需要。可以通过 `PRE_EXCLUDE_REGEXES` 选项将这些库过滤掉，该选项会在包含库之前进行过滤。任何与这些正则表达式匹配的文件路径都将在确定运行时依赖时被排除在考虑范围之外。或者，也可以使用 `POST_EXCLUDE_REGEXES` 选项，在找到文件之后对其进行过滤。如果你想排除来自某个特定位置的文件，这个选项很有用。在前面的示例中，它被用来排除来自 32 位 `system32` 文件夹的 DLL 文件。

在本节中，我们学习了如何使用 CPack 打包我们的项目。这不是一本详尽的指南。有关完整指南，官方的 CPack 文档提供了大量的信息。至此，我们成功完成了本章内容。

# 总结

在本章中，我们学习了如何使目标可安装的基础知识，以及如何为开发和消费者环境打包项目。部署是专业软件项目中的一个重要方面，借助我们在本章中覆盖的内容，你可以轻松处理这些部署需求。

在下一章中，我们将学习如何将第三方库集成到 CMake 项目中。

# 问题

请回答以下问题，测试你对本章内容的理解：

1.  我们如何指示 CMake 使 CMake 目标可安装？

1.  通过 `install(TARGETS)` 命令安装时，哪些文件会被安装？

1.  对于库目标，`install(TARGETS)` 命令是否会安装头文件？为什么？如果没有，如何安装头文件？

1.  `GNUInstallDirs` CMake 模块提供了什么？

1.  如何选择性地将一个目录的内容安装到目标目录中？

1.  为什么在指定安装目标目录时应该使用相对路径？

1.  `config-file` 包所需的基本文件是什么？

1.  导出一个目标是什么意思？

1.  如何使 CMake 项目能够通过 CPack 打包？

# 答案

以下是上述问题的答案：

1.  这可以通过 `install(TARGETS <target_name>)` 命令实现。

1.  指定目标的输出工件。

1.  不会，因为头文件不被视为目标的输出工件。它们必须通过 `install(DIRECTORY)` 命令单独安装。

1.  `GNUInstallDirs` CMake 模块提供了系统特定的默认安装路径，例如 `bin`、`lib` 和 `include`。

1.  通过 `install(DIRECTORY)` 命令的 `PATTERN` 和 `FILES_MATCHING` 参数的帮助。

1.  为了使安装可迁移，用户可以通过指定安装前缀来更改安装目录。

1.  `<package-name>-config.cmake` 或 `<package-name>Config.cmake` 文件，另可选择包含 `<package-name>-config-version.cmake` 或 `<package-name>ConfigVersion.cmake` 文件。

1.  导出一个目标意味着创建所需的 CMake 代码，以便将其导入到另一个 CMake 项目中。

1.  通过包含 CPack 模块可以实现。
