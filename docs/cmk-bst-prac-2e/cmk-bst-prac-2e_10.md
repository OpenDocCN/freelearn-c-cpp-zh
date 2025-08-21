# 第八章：使用 CMake 执行自定义任务

构建和发布软件可能是一个复杂的任务，任何工具都无法完成构建和发布项目所需的所有不同任务。在某些时候，您可能需要执行一个编译器或 CMake 功能没有涵盖的任务。常见任务包括归档构建成果、创建哈希以验证下载，或生成或自定义构建的输入文件。还有许多其他依赖于特定软件构建环境的专门任务。

在本章中，我们将学习如何将自定义任务包含到 CMake 项目中，以及如何创建自定义构建目标和自定义命令。我们将讨论如何创建和管理目标之间的依赖关系，以及如何将它们包含或排除在标准构建之外。

在项目的构建步骤中包含这样的外部程序可以帮助确保代码的一致性，即使有很多人参与其中。由于 CMake 构建非常容易自动化，使用 CMake 调用必要的命令使得将这些工具应用到不同的机器或 CI 环境变得简单。

在本章中，我们将学习如何定义自定义任务，以及如何控制它们的执行时机。特别地，我们将专注于管理自定义任务和常规目标之间的依赖关系。由于 CMake 通常用于在多个平台上提供构建信息，您还将学习如何定义通用任务，以便它们能在任何运行 CMake 的地方执行。

本章将涵盖以下主要内容：

+   在 CMake 中使用外部程序

+   在构建时执行自定义任务

+   在配置时执行自定义任务

+   复制和修改文件

+   使用 CMake 执行平台独立的命令

那么，让我们开始吧！

# 技术要求

与前几章一样，本章中的示例已在 CMake 3.21 上进行过测试，并且可以在以下编译器上运行：

+   GCC 9 或更高版本

+   Clang 12 或更高版本

+   MSVC 19 或更高版本

本章的所有示例和源代码可以在本书的 GitHub 仓库中找到，地址是 [`github.com/PacktPublishing/CMake-Best-Practices---2nd-Edition`](https://github.com/PacktPublishing/CMake-Best-Practices---2nd-Edition)。如果缺少任何软件，相应的示例将从构建中排除。

# 在 CMake 中使用外部程序

CMake 功能非常广泛，因此它可以覆盖许多构建软件时的任务。然而，也有一些情况，开发者需要执行一些 CMake 功能没有涵盖的任务。常见的例子包括运行特殊工具，对目标的文件进行预处理或后处理，使用源代码生成器为编译器生成输入，以及压缩和归档不由 CPack 处理的构建成果。必须在构建步骤中完成的此类特殊任务的列表可能是几乎无尽的。CMake 支持三种执行自定义任务的方式：

+   通过定义一个使用 `add_custom_target` 执行命令的目标

+   通过使用`add_custom_command`将自定义命令附加到现有目标，或者通过使目标依赖于由自定义命令生成的文件

+   通过使用`execute_process`函数，在配置步骤中执行命令

如果可能，应在构建步骤中调用外部程序，因为配置步骤用户控制性较低，应尽可能快速地运行。

让我们学习如何定义在构建时运行的任务。

# 在构建时执行自定义任务

添加自定义任务的最通用方法是通过创建一个自定义目标，该目标以命令序列执行外部任务。自定义目标像任何其他库或可执行目标一样处理，不同之处在于它们不调用编译器和链接器，而是执行用户定义的操作。自定义目标使用`add_custom_target`命令定义：

```cpp
add_custom_target(Name [ALL] [command1 [args1...]]
                  [COMMAND command2 [args2...] ...]
                  [DEPENDS depend depend depend ... ]
                  [BYPRODUCTS [files...]]
                  [WORKING_DIRECTORY dir]
                  [COMMENT comment]
                  [JOB_POOL job_pool]
                  [VERBATIM] [USES_TERMINAL]
                  [COMMAND_EXPAND_LISTS]
                  [SOURCES src1 [src2...]])
```

`add_custom_target`命令的核心是通过`COMMAND`选项传递的命令列表。虽然第一个命令可以在没有此选项的情况下传递，但最好在任何`add_custom_target`调用中始终添加`COMMAND`选项。默认情况下，只有在明确请求时，自定义目标才会执行，除非指定了`ALL`选项。自定义目标始终被认为是过时的，因此指定的命令会始终运行，无论它们是否重复产生相同的结果。通过`DEPENDS`关键字，可以使自定义目标依赖于通过`add_custom_command`函数定义的自定义命令的文件和输出，或依赖于其他目标。若要使自定义目标依赖于另一个目标，请使用`add_dependencies`函数。反过来也适用——任何目标都可以依赖于自定义目标。如果自定义目标创建了文件，可以在`BYPRODUCTS`选项下列出这些文件。列在其中的任何文件都会标记为`GENERATED`属性，CMake 会用这个属性来判断构建是否过时，并找出需要清理的文件。然而，使用`add_custom_command`创建文件的任务可能更适合，如本节后续所述。

默认情况下，这些命令在当前二进制目录中执行，该目录存储在`CMAKE_CURRENT_BINARY_DIRECTORY`缓存变量中。如有必要，可以通过`WORKING_DIRECTORY`选项更改此目录。此选项可以是绝对路径，也可以是相对路径，若为相对路径，则相对于当前二进制目录。

`COMMENT`选项用于指定在命令运行之前打印的消息，这在命令默默运行时非常有用。不幸的是，并非所有生成器都显示这些消息，因此将其用于显示关键信息可能不太可靠。

`VERBATIM` 标志会将所有命令直接传递给平台，而不经过底层 shell 的转义或变量替换。CMake 本身仍会替换传递给命令或参数的变量。当转义可能成为问题时，建议传递 `VERBATIM` 标志。编写自定义任务时，使其与底层平台独立也是一种良好的实践。在本章稍后，在 *使用 CMake 创建平台独立的* *命令* 部分，你可以找到更多有关如何创建平台独立命令的技巧。

`USES_TERMINAL` 选项指示 CMake 如果可能的话让命令访问终端。如果使用的是 Ninja 生成器，这意味着它将在 `terminal` 作业池中运行。该池中的所有命令是串行执行的。

`JOB_POOL` 选项可在使用 Ninja 生成时控制作业的并发性。它很少使用，并且不能与 `USES_TERMINAL` 标志一起使用。你很少需要干预 Ninja 的作业池，且处理起来并不简单。如果你想了解更多信息，可以参考 CMake 官方文档中的 `JOB_POOLS` 属性部分。

`SOURCES` 属性接受与自定义目标关联的源文件列表。该属性不会影响源文件，但可以帮助在某些 IDE 中显示文件。如果一个命令依赖于例如与项目一起交付的脚本等文件，这些文件应该在这里添加。

`COMMAND_EXPAND_LISTS` 选项告诉 CMake 在将列表传递给命令之前展开它们。这在某些情况下是必要的，因为在 CMake 中，列表只是由分号分隔的字符串，这可能导致语法错误。当传递 `COMMAND_EXPAND_LISTS` 选项时，分号会根据平台被替换为合适的空白字符。展开操作包括使用 `$<JOIN:` 生成器表达式生成的列表。

以下是一个示例，展示了一个使用名为 *CreateHash* 的外部程序来为另一个目标的输出创建哈希值的自定义目标：

```cpp
add_executable(SomeExe)
add_custom_target(CreateHash ALL
                  COMMAND Somehasher $<TARGET_FILE:SomeExe>
)
```

本例创建了一个名为 *CreateHash* 的自定义目标，它调用外部的 *SomeHasher* 程序，并将 *SomeExe* 目标的二进制文件作为参数。请注意，二进制文件是通过 `$<TARGET_FILE:SomeExe>` 生成器表达式获取的。这有两个目的——它消除了用户需要跟踪目标二进制文件名的需求，并且在两个目标之间建立了一个隐式依赖关系。CMake 会识别这些隐式依赖并按正确的顺序执行目标。如果生成所需文件的目标尚未构建，CMake 将自动构建它。你还可以使用 `$<TARGET_FILE:` 生成器来直接执行由另一个目标创建的可执行文件。以下生成器表达式会在目标之间引发隐式依赖：

+   `$<TARGET_FILE:target>`：这包含了目标的主二进制文件的完整路径，如`.exe`、`.so`或`.dll`。

+   `$<TARGET_LINKER_FILE: target>`：这包含了用于与目标进行链接的文件的完整路径。通常是库文件本身，但在 Windows 上，`.lib`文件会与 DLL 相关联。

+   `$<TARGET_SONAME_FILE: target>`：这包含了库文件及其完整名称，包括由`SOVERSION`属性设置的任何数字，如`.so.3`。

+   `$<TARGET_PDB_FILE: target>`：这包含了用于调试的生成的程序数据库文件的完整路径。

创建自定义目标是一种在构建时执行外部任务的方法。另一种方法是定义自定义命令。自定义命令可以用来将自定义任务添加到现有目标中，包括自定义目标。

## 将自定义任务添加到现有目标

有时，在构建目标时，你可能需要执行一个额外的外部任务。在 CMake 中，你可以使用`add_custom_command`来实现这一点，它有两种签名。一种用于将命令钩入现有的目标，另一种用于生成文件。我们将在本节后续部分讲解这一点。将命令添加到现有目标的签名如下所示：

```cpp
add_custom_command(TARGET <target>
                   PRE_BUILD | PRE_LINK | POST_BUILD
                   COMMAND command1 [ARGS] [args1...]
                   [COMMAND command2 [ARGS] [args2...] ...]
                   [BYPRODUCTS [files...]]
                   [WORKING_DIRECTORY dir]
                   [COMMENT comment]
                   [VERBATIM] [USES_TERMINAL]
                   [COMMAND_EXPAND_LISTS])
```

大多数选项的工作方式与之前提到的`add_custom_target`类似。`TARGET`属性可以是当前目录中定义的任何目标，这是该命令的一个限制，尽管这很少成为问题。命令可以在以下时间钩入构建过程：

+   `PRE_BUILD`：在 Visual Studio 中，此命令会在任何其他构建步骤之前执行。当你使用其他生成器时，它将在`PRE_LINK`命令之前执行。

+   `PRE_LINK`：此命令将在源代码编译完成后执行，但在可执行文件或归档工具链接到静态库之前执行。

+   `POST_BUILD`：在所有其他构建规则执行完毕后运行此命令。

执行自定义步骤的最常见方法是使用`POST_BUILD`；另外两种选项很少使用，可能是因为支持有限，或者因为它们既不能影响链接，也不能影响构建。

将自定义命令添加到现有目标相对简单。以下代码在每次编译后添加一个命令，用于生成并存储已构建文件的哈希值：

```cpp
add_executable(MyExecutable)
add_custom_command(TARGET MyExecutable
   POST_BUILD
  COMMAND hasher $<TARGET_FILE:ch8_custom_command_example>
    ${CMAKE_CURRENT_BINARY_DIR}/MyExecutable.sha256
COMMENT "Creating hash for MyExecutable"
)
```

在这个例子中，使用一个名为`hasher`的自定义可执行文件来生成`MyExecutable`目标的输出文件的哈希值。

通常，在构建之前你可能需要执行某些操作，以更改文件或生成额外的信息。对于这种情况，第二种签名通常是更好的选择。让我们仔细看看。

## 生成文件与自定义任务

通常，我们希望自定义任务能生成特定的输出文件。这可以通过定义自定义目标并设置目标之间的必要依赖关系来完成，或者通过挂钩构建步骤来实现，正如前面所述。不幸的是，`PRE_BUILD` 钩子并不可靠，因为只有 Visual Studio 生成器能正确支持它。因此，一个更好的方法是创建一个自定义命令来生成文件，通过使用 `add_custom_command` 函数的第二种签名：

```cpp
add_custom_command(OUTPUT output1 [output2 ...]
                   COMMAND command1 [ARGS] [args1...]
                   [COMMAND command2 [ARGS] [args2...] ...]
                   [MAIN_DEPENDENCY depend]
                   [DEPENDS [depends...]]
                   [BYPRODUCTS [files...]]
                   [IMPLICIT_DEPENDS <lang1> depend1
                                    [<lang2> depend2] ...]
                   [WORKING_DIRECTORY dir]
                   [COMMENT comment]
                   [DEPFILE depfile]
                   [JOB_POOL job_pool]
                   [VERBATIM] [APPEND] [USES_TERMINAL]
                   [COMMAND_EXPAND_LISTS])
```

这种签名的 `add_custom_command` 定义了一个生成 `OUTPUT` 中指定文件的命令。该命令的大多数选项与 `add_custom_target` 和挂钩自定义任务到构建步骤的签名非常相似。`DEPENDS` 选项可以用来手动指定文件或目标的依赖关系。需要注意的是，与此相比，自定义目标的 `DEPENDS` 选项只能指向文件。如果任何依赖关系在构建或 CMake 更新时发生变化，自定义命令将再次运行。`MAIN_DEPENDENCY` 选项密切相关，指定命令的主要输入文件。它的作用类似于 `DEPENDS` 选项，只是它只接受一个文件。`MAIN_DEPENDENCY` 主要用于告诉 Visual Studio 添加自定义命令的位置。

注意

如果源文件列为 `MAIN_DEPENDENCY`，则自定义命令会替代正常的文件编译，这可能导致链接错误。

另外两个与依赖相关的选项，`IMPLICIT_DEPENDS` 和 `DEPFILE`，很少使用，因为它们的支持仅限于 Makefile 生成器。`IMPLICIT_DEPENDS` 告诉 CMake 使用 C 或 C++ 扫描器来检测列出文件的任何编译时依赖关系，并基于此创建依赖关系。另一个选项 `DEPFILE` 可以用来指向 `.d` 依赖文件，该文件由 Makefile 项目生成。`.d` 文件最初来自 GNU Make 项目，虽然它们非常强大，但也比较复杂，大多数项目不应手动管理这些文件。以下示例展示了如何使用自定义命令，在常规目标运行之前，根据用于输入的另一个文件生成源文件：

```cpp
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/main.cpp
COMMAND sourceFileGenerator ${CMAKE_CURRENT_SOURCE_DIR}/message.txt
  ${CMAKE_CURRENT_BINARY_DIR}/main.cpp
COMMENT "Creating main.cpp frommessage.txt"
DEPENDS message.txt
VERBATIM
)
add_executable(
ch8_create_source_file_example
${CMAKE_CURRENT_BINARY_DIR}/main.cpp
)
```

在这个示例中发生了几件事。首先，自定义命令将当前二进制目录中的 `main.cpp` 文件定义为 `OUTPUT` 文件。然后，定义了生成该文件的命令——这里使用了一个名为 `sourceFileGenerator` 的假设程序——它将消息文件转换为 `.cpp` 文件。`DEPENDS` 部分指出，每次 `message.txt` 文件发生变化时，都应重新运行该命令。

后续创建了可执行文件的目标。由于可执行文件引用了在自定义命令的`OUTPUT`部分指定的`main.cpp`文件，CMake 会隐式添加命令和目标之间的必要依赖关系。以这种方式使用自定义命令比使用`PRE_BUILD`指令更可靠且具有更好的移植性，因为它适用于所有生成器。

有时，为了创建所需的输出，可能需要多个命令。如果存在一个生成相同输出的先前命令，可以通过使用`APPEND`选项将命令链接起来。使用`APPEND`的自定义命令只能定义额外的`COMMAND`和`DEPENDS`选项；其他选项会被忽略。如果两个命令生成相同的输出文件，除非指定`APPEND`，CMake 会打印出错误。这个功能主要用于当一个命令是可选执行时。考虑以下示例：

```cpp
add_custom_command(OUTPUT archive.tar.gz
COMMAND cmake -E tar czf ${CMAKE_CURRENT_BINARY_DIR}/archive.tar.gz
  $<TARGET_FILE:MyTarget>
COMMENT "Creating Archive for MyTarget"
VERBATIM
)
add_custom_command(OUTPUT archive.tar.gz
COMMAND cmake -E tar czf ${CMAKE_CURRENT_BINARY_DIR}/archive.tar.gz
  ${CMAKE_CURRENT_SOURCE_DIR}/SomeFile.txt
APPEND
)
```

在这个示例中，目标`MyTarget`的输出文件已经被添加到一个`tar.gz`归档中；之后，另一个文件被添加到相同的归档中。注意，第一个命令自动依赖于`MyTarget`，因为它使用了在命令中创建的二进制文件。然而，它不会通过构建自动执行。第二个自定义命令列出了与第一个命令相同的输出文件，但将压缩文件作为第二个输出添加。通过指定`APPEND`，第二个命令会在每次执行第一个命令时自动执行。如果缺少`APPEND`关键字，CMake 会打印出类似如下的错误：

```cpp
CMake Error at CMakeLists.txt:30 (add_custom_command):
  Attempt to add a custom rule to output
     /create_hash_example/build/hash_example.md5.rule
   which already has a custom rule.
```

如前所述，本示例中的自定义命令隐式依赖于`MyTarget`，但它们不会自动执行。为了执行这些命令，推荐的做法是创建一个依赖于输出文件的自定义目标，可以像这样生成：

```cpp
add_custom_target(create_archive ALL DEPENDS
    ${CMAKE_CURRENT_BINARY_DIR}/archive.tar.gz
)
```

在这里，创建了一个名为`create_archive`的自定义目标，该目标作为`All`构建的一部分执行。由于它依赖于自定义命令的输出，因此构建该目标会调用自定义命令。自定义命令反过来依赖于`MyTarget`，因此如果`MyTarget`尚未是最新的，构建`create_archive`也会触发`MyTarget`的构建。

`add_custom_command`和`add_custom_target`自定义任务都会在 CMake 的构建步骤中执行。如果需要，也可以在配置时添加任务。我们将在下一节中讨论这个问题。

# 在配置时执行自定义任务

要在配置时执行自定义任务，可以使用`execute_process`函数。常见的需求是，如果构建在开始之前需要额外的信息，或者需要更新文件以便重新运行 CMake。另一个常见的情况是，当`CMakeLists.txt`文件或其他输入文件在配置步骤中生成时，尽管这也可以通过专用的`configure_file`命令实现，正如本章稍后所展示的那样。

`execute_process` 函数的工作方式与我们之前看到的 `add_custom_target` 和 `add_custom_command` 函数非常相似。然而，有一个区别是，`execute_process` 可以将输出捕获到变量或文件中的 `stdout` 和 `stderr`。`execute_process` 的函数签名如下：

```cpp
execute_process(COMMAND <cmd1> [<arguments>]
                [COMMAND <cmd2> [<arguments>]]...
                [WORKING_DIRECTORY <directory>]
                [TIMEOUT <seconds>]
                [RESULT_VARIABLE <variable>]
                [RESULTS_VARIABLE <variable>]
                [OUTPUT_VARIABLE <variable>]
                [ERROR_VARIABLE <variable>]
                [INPUT_FILE <file>]
                [OUTPUT_FILE <file>]
                [ERROR_FILE <file>]
                [OUTPUT_QUIET]
                [ERROR_QUIET]
                [COMMAND_ECHO <where>]
                [OUTPUT_STRIP_TRAILING_WHITESPACE]
                [ERROR_STRIP_TRAILING_WHITESPACE]
                [ENCODING <name>]
                [ECHO_OUTPUT_VARIABLE]
                [ECHO_ERROR_VARIABLE]
                [COMMAND_ERROR_IS_FATAL <ANY|LAST>])
```

`execute_process` 函数接受一系列要在 `WORKING_DIRECTORY` 中执行的 `COMMAND` 属性。最后执行命令的返回代码可以存储在使用 `RESULT_VARIABLE` 定义的变量中。或者，可以将以分号分隔的变量列表传递给 `RESULTS_VARIABLE`。如果使用 `list` 版本，命令会按照定义的变量顺序存储命令的返回码。如果定义的变量少于命令，任何多余的返回码将被忽略。如果定义了 `TIMEOUT` 且任何子进程未能返回，结果变量将包含 `timeout`。从 CMake 3.19 版本开始，提供了方便的 `COMMAND_ERROR_IS_FATAL` 选项，它告诉 CMake 如果任何（或仅最后一个）进程失败，则中止执行。这比在执行后获取所有返回码并逐个检查要方便得多。在以下示例中，如果任何命令返回非零值，CMake 的配置步骤将失败并报错：

```cpp
execute_process(
   COMMAND SomeExecutable
   COMMAND AnotherExecutable
   COMMAND_ERROR_IS_FATAL_ANY
)
```

任何输出到 `stdout` 或 `stderr` 的内容可以分别通过 `OUTPUT_VARIABLE` 或 `ERROR_VARIABLE` 变量进行捕获。作为替代方法，它们可以通过使用 `OUTPUT_FILE` 或 `ERROR_FILE` 重定向到文件，或者通过传递 `OUTPUT_QUIET` 或 `ERROR_QUIET` 完全忽略。不能同时将输出捕获到变量和文件中，这会导致其中一个为空。保留哪个输出，丢弃哪个，取决于平台。如果没有其他设置，`OUTPUT_*` 选项表示输出将发送到 CMake 进程本身。

如果输出被捕获到变量中但仍然可以显示，可以添加 `ECHO_<STREAM>_VARIABLE`。也可以通过传递 `STDOUT`、`STDERR` 或 `NONE` 给 `COMMAND_ECHO` 选项来让 CMake 输出命令本身。然而，如果输出被捕获到文件中，这将没有任何效果。如果为 `stdout` 和 `stderr` 指定相同的变量或文件，结果将被合并。如果需要，可以通过传递文件给 `INPUT_FILE` 选项来控制第一个命令的输入流。

输出到变量的行为可以通过使用 `<STREAM>_STRIP_TRAILING_WHITESPACE` 选项进行有限控制，该选项会去除输出末尾的空白字符。当输出被重定向到文件时，此选项无效。在 Windows 上，可以使用 `ENCODING` 选项来控制输出编码。它支持以下几种值：

+   `NONE`：不进行重新编码。这将保持 CMake 内部的编码格式，即 UTF-8。

+   `AUTO`：使用当前控制台的编码。如果不可用，则使用 ANSI 编码。

+   `ANSI`：使用 ANSI 代码页进行编码。

+   `OEM`：使用平台定义的代码页。

+   `UTF8` 或 `UTF-8`：强制使用 UTF-8 编码。

使用 `execute_process` 的常见原因之一是收集构建所需的信息，然后将其传递给项目。考虑一个示例，我们想要将 git 修订版编译到可执行文件中，通过将其作为预处理器定义传递。这样做的缺点是，为了执行自定义任务，必须调用 CMake，而不仅仅是构建系统。因此，使用带有 `OUTPUT` 参数的 `add_custom_command` 可能是更实际的解决方案，但为了说明目的，这个示例应该已经足够。以下是一个示例，其中在配置时读取 git 哈希并作为编译定义传递给目标：

```cpp
find_package(Git REQUIRED)
execute_process(COMMAND ${GIT_EXECUTABLE} "rev-parse" "--short"
  "HEAD"
OUTPUT_VARIABLE GIT_REVISION
OUTPUT_STRIP_TRAILING_WHITESPACE
COMMAND_ERROR_IS_FATAL ANY
WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
add_executable(SomeExe src/main.cpp)
target_compile_definitions(SomeExe PRIVATE VERSION=
  \"${GIT_REVISION}\")
```

在这个示例中，传递给 `execute_process` 的 `git` 命令是在包含当前正在执行的 `CMakeLists.txt` 文件的目录中执行的。生成的哈希值存储在 `GIT_REVISION` 变量中，如果命令由于任何原因失败，配置过程将会停止并报错。

通过使用预处理器定义将 `execute_process` 的信息传递给编译器的做法远非最佳。更好的解决方案是，如果我们能够生成一个包含这些信息的头文件，并将其包含进来。CMake 还有一个名为 `configure_file` 的功能可以用来实现这一目的，正如我们将在下一节中看到的那样。

# 复制和修改文件

在构建软件时，一个相对常见的任务是必须在构建前将某些文件复制到特定位置。大多数文件操作可以在配置时通过 `file()` 命令来完成。例如，复制文件可以通过以下方式调用：

```cpp
file(COPY_FILE old_file new_file)
```

有几种文件操作可用，例如 `file(REMOVE)` 和 `file(REMOVE_RECURSE)` 用于删除文件或目录树，`file(RENAME)` 用于移动文件，`file(CHMOD)` 用于更改支持该操作的系统上的权限。`file` 命令的完整文档请参见：[`cmake.org/cmake/help/latest/command/file.html`](https://cmake.org/cmake/help/latest/command/file.html)。

但如果我们想要同时复制和修改一个文件该怎么办呢？在*配置时执行自定义任务*一节中，我们看到了一个示例，其中获取了 git 修订版本并作为预处理器定义传递给编译器。更好的做法是生成一个包含必要信息的头文件。虽然直接回显代码片段并将其写入文件是可行的，但这样做是危险的，因为它可能会导致平台特定的代码。CMake 的解决方案是 `configure_file` 命令，它可以将文件从一个位置复制到另一个位置并在此过程中修改其内容。`configure_file` 的函数签名如下：

```cpp
configure_file(<input> <output>
               [NO_SOURCE_PERMISSIONS | USE_SOURCE_PERMISSIONS |
                FILE_PERMISSIONS <permissions>...]
               [COPYONLY] [ESCAPE_QUOTES] [@ONLY]
               [NEWLINE_STYLE [UNIX|DOS|WIN32|LF|CRLF] ])
```

`configure_file`函数会将`<input>`文件复制到`<output>`文件。如果需要，输出文件的路径将被创建，路径可以是相对路径或绝对路径。如果使用相对路径，输入文件将从当前源目录中查找，但输出文件的路径将相对于当前构建目录。如果无法写入输出文件，命令将失败，配置将被停止。默认情况下，输出文件与目标文件具有相同的权限，尽管如果当前用户与输入文件所属的用户不同，所有权可能会发生变化。如果添加`NO_SOURCE_PERMISSION`，则不会传递权限，输出文件将获得默认的`rw-r--r--`权限。或者，可以通过`FILE_PERMISSIONS`选项手动指定权限，该选项需要一个三位数字作为参数。`USE_SOURCE_PERMISSION`已经是默认值，该选项仅用于更明确地表达意图。

如前所述，`configure_file`在复制到输出路径时也会替换输入文件的部分内容，除非传递了`COPYONLY`。默认情况下，`configure_file`会将所有引用的变量`${SOME_VARIABLE}`或`@SOME_VARIABLE@`替换为相同名称的变量的值。如果在`CMakeLists.txt`中定义了变量，当调用`configure_file`时，相应的值会写入输出文件。如果未指定变量，输出文件中的相应位置将包含空字符串。考虑一个包含以下信息的`hello.txt.in`文件：

```cpp
Hello ${GUEST} from @GREETER@
```

在`CMakeLists.txt`文件中，`configure_file`函数用于配置`hello.txt.in`文件：

```cpp
set(GUEST "World")
set(GREETER "The Universe")
configure_file(hello.txt.in hello.txt)
```

在这个示例中，生成的`hello.txt`文件将包含`Hello World from The Universe`。如果将`@ONLY`选项传递给`configure_file`，只有`@GREETER@`会被替换，生成的内容将是`Hello ${GUEST} from The Universe`。使用`@ONLY`在你转换可能包含大括号括起来的变量的 CMake 文件时非常有用，这些变量不应该被替换。`ESCAPE_QUOTES`会在目标文件中用反斜杠转义任何引号。默认情况下，`configure_file`会转换换行符，以便目标文件与当前平台匹配。默认行为可以通过设置`NEWLINE_STYLE`来改变。`UNIX`或`LF`将使用`\n`作为换行符，而`DOS`、`WIN32`和`CRLF`将使用`\r\n`。同时设置`NEWLINE_STYLE`和`COPYONLY`选项将导致错误。请注意，设置`COPYONLY`不会影响换行符样式。

让我们回到我们希望将 git 修订版编译到可执行文件中的示例。在这里，我们将编写一个头文件作为输入。它可能包含如下内容：

```cpp
#define CMAKE_BEST_PRACTICES_VERSION "@GIT_REVISION@"
The CMakeLists.txt could look something like this:
execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
    OUTPUT_VARIABLE GIT_REVISION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
configure_file(version.h.in ${CMAKE_CURRENT_SOURCE_DIR}/src
  /version.h @ONLY)
```

如前一节中的示例所示，版本信息是作为编译定义传递的，git 修订版首先通过`execute_process`获取。随后，文件通过`configure_file`进行复制，`@GIT_REVISION@`被替换为当前提交的短哈希值。

当你使用预处理器定义时，`configure_file`会将所有形如`#cmakedefine VAR ...`的行替换为`#define VAR`或`/* undef VAR */`，具体取决于`VAR`是否包含 CMake 解释为`true`或`false`的值。

假设有一个名为`version.in.h`的文件，其中包含以下两行：

```cpp
#cmakedefine GIT_VERSION_ENABLE
#cmakedefine GIT_VERSION "@GIT_REVISION@"
```

附带的`CMakeLists.txt`文件可能如下所示：

```cpp
option(GIT_VERSION_ENABLE "Define revision in a header file" ON)
if(GIT_VERSION_ENABLE)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
    OUTPUT_VARIABLE GIT_REVISION
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
endif()
configure_file(version.h.in ${CMAKE_CURRENT_SOURCE_DIR}/src/version.h @ONLY)
```

一旦配置已运行，如果`GIT_REVISION_ENABLE`被启用，生成的文件将包含以下输出：

```cpp
#define GIT_VERSION_ENABLE
#define CMAKE_BEST_PRACTICES_VERSION "c030d83"
```

如果`GIT_REVISION_ENABLE`被禁用，生成的文件将包含以下输出：

```cpp
/* #undef GIT_VERSION_ENABLE */
/* #undef GIT_REVISION */
```

总而言之，`configure_file`命令非常有用，可以为构建准备输入。除了生成源文件外，它常用于生成 CMake 文件，这些文件随后会被包含在`CMakeLists.txt`文件中。其优点之一是它允许你独立于平台复制和修改文件，这在跨平台工作时是一个重要的优势。由于`configure_file`和`execute_process`常常一起使用，因此确保执行的命令也是平台无关的。

在下一节中，你将学习如何使用 CMake 来定义平台无关的命令和脚本。

# 使用 CMake 进行平台无关命令

CMake 成功的一个关键因素是它允许你在多种平台上构建相同的软件。相反，这意味着`CMakeLists.txt`必须以不假设某个平台或编译器必须使用的方式编写。这可能会很具挑战性，特别是当你在处理自定义任务时。在这种情况下，`cmake`命令行工具提供的`-E`标志非常有帮助，它可以用于执行常见任务，例如文件操作和创建哈希。大多数`cmake -E`命令用于与文件相关的操作，如创建、复制、重命名和删除文件，以及创建目录。在支持文件系统链接的系统上，CMake 还可以在文件之间创建符号链接或硬链接。自 CMake 版本 3.21 以来，大多数操作也可以通过使用`file()`命令来实现，但并非所有操作都可以。值得注意的是，创建哈希值时，可以使用`cmake –``E <algorithm>`以平台无关的方式进行。

此外，CMake 可以使用`tar`命令创建文件归档，并使用`cat`命令连接文本文件。它还可以用于为文件创建各种哈希值。

还有一些操作可以提供关于当前系统信息的信息。`capabilities`操作将打印出 CMake 的能力，例如了解支持的生成器和当前正在运行的 CMake 版本。`environment`命令将打印出已设置的环境变量列表。

可以通过运行`cmake -E`而不带任何其他参数来获取命令行选项的完整参考。CMake 的在线文档可以在[`cmake.org/cmake/help/latest/manual/cmake.1.html#run-a-command-line-tool`](https://cmake.org/cmake/help/latest/manual/cmake.1.html#run-a-command-line-tool)找到。

平台无关的文件操作

每当需要通过自定义任务执行文件操作时，请使用`cmake –``E`。

使用`cmake -E`，在大多数情况下可以做得相当远。但是，有时需要执行更复杂的操作。为此，CMake 可以在脚本模式下运行，执行 CMake 文件。

## 执行 CMake 文件作为脚本

CMake 的脚本模式在创建跨平台脚本时非常强大。这是因为它允许您创建完全与平台无关的脚本。通过调用`cmake -P <script>.cmake`，执行指定的 CMake 文件。脚本文件可能不包含定义构建目标的任何命令。可以使用`-D`标志将参数作为变量传递，但必须在`-P`选项之前执行此操作。或者，参数仅可以在脚本名称之后追加，以便可以使用`CMAKE_ARGV[n]`变量检索它们。参数的数量存储在`CMAKE_ARGC`变量中。以下脚本演示了如何使用位置参数生成文件的哈希并将其存储在另一个文件中：

```cpp
cmake_minimum_required(VERSION 3.21)
if(CMAKE_ARGC LESS 5)
    message(FATAL_ERROR "Usage: cmake -P CreateSha256.cmake
      file_to_hash target_file")
endif()
set(FILE_TO_HASH ${CMAKE_ARGV3})
set(TARGET_FILE ${CMAKE_ARGV4})
# Read the source file and generate the hash for it
file(SHA256 "${FILE_TO_HASH}" GENERATED_HASH)
# write the hash to a new file
file(WRITE "${TARGET_FILE}" "${GENERATED_HASH}")
```

可以使用`cmake -P CreateSha256.cmake <input file> <output_file>`来调用此脚本。请注意，前三个参数被`cmake`，`-P`和脚本名称（`CreateSha256.cmake`）占用。虽然不是严格要求，但脚本文件应始终在开头包含`cmake_minimum_required`语句。定义脚本的另一种方式，而不使用位置参数，如下所示：

```cpp
cmake_minimum_required(VERSION 3.21)
if(NOT FILE_TO_HASH OR NOT TARGET_FILE)
   message(FATAL_ERROR "Usage: cmake –DFILE_TO_HASH=<intput_file> \
-DTARGET_FILE=<target file> -P CreateSha256.cmake")
endif()
# Read the source file and generate the hash for it
file(SHA256 "${FILE_TO_HASH}" GENERATED_HASH)
# write the hash to a new file
file(WRITE "${TARGET_FILE}" "${GENERATED_HASH}")
```

在这种情况下，脚本必须通过显式传递变量来调用，如下所示：

```cpp
cmake –DFILE_TO_HASH=<input>
      -DTARGET_FILE=<target> -P CreateSha256.cmake
```

这两种方法也可以结合使用。一个常见的模式是将所有简单的强制参数作为位置参数来期望，并将任何可选或更复杂的参数作为定义的变量。将脚本模式与`add_custom_command`、`add_custom_target`或`execute_process`结合使用是创建跨平台无关的构建指令的好方法。从前面章节生成哈希的示例可能如下所示：

```cpp
add_custom_target(Create_hash_target ALL
COMMAND cmake -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/
  CreateSha256.cmake $<TARGET_FILE:SomeTarget>
   ${CMAKE_CURRENT_BINARY_DIR}/hash_example.sha256
)
add_custom_command(TARGET SomeTarget
POST_BUILD
COMMAND cmake -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake
  /CreateSha256.cmake $<TARGET_FILE:SomeTarget>
    ${CMAKE_CURRENT_BINARY_DIR}/hash_example.sha256
)
```

将 CMake 的脚本模式与在项目的配置或构建阶段执行自定义命令的各种方式结合使用，为您在定义构建过程时提供了很大的自由，甚至适用于不同的平台。然而，需注意的是，向构建过程中添加过多的逻辑可能会使其维护变得比预期更加困难。每当您需要编写脚本或向`CMakeLists.txt`文件中添加自定义命令时，最好先休息一下，考虑一下这一步是否属于构建过程，还是应该留给用户在设置开发环境时处理。

# 总结

在本章中，您学习了如何通过执行外部任务和程序来定制构建。我们介绍了如何将自定义构建操作作为目标添加，如何将它们添加到现有目标中，以及如何在配置步骤期间执行它们。我们探讨了如何通过命令生成文件，以及如何使用`configure_file`命令复制和修改文件。最后，我们学习了如何使用 CMake 命令行工具以平台无关的方式执行任务。

定制 CMake 构建的能力是一个非常强大的资产，但它也往往会使构建变得更加脆弱，因为当执行任何自定义任务时，构建的复杂性通常会增加。虽然有时不可避免，但依赖于除编译器和链接器之外的外部程序的安装可能意味着某些软件无法在未安装或不可用这些程序的平台上构建。这意味着，必须特别小心，确保自定义任务在可能的情况下不会假设使用 CMake 的系统有什么特定的配置。最后，执行自定义任务可能会对构建系统带来性能负担，尤其是当它们在每次构建时进行大量工作时。

然而，如果您小心处理自定义构建步骤，它们是增加构建凝聚力的一个很好的方式，因为许多与构建相关的任务可以在构建定义的位置进行定义。这可以使自动化任务（如创建构建产物的哈希值或将所有文档打包成一个公共档案）变得更加容易。

在下一章中，您将学习如何使构建环境在不同系统之间具有可移植性。您将学习如何使用预设来定义配置 CMake 项目的常见方式，如何将您的构建环境打包到容器中，以及如何使用`sysroots`来定义工具链和库，以便它们在不同系统之间具有可移植性。

# 问题

请回答以下问题，测试您对本章内容的掌握情况：

1.  `add_custom_command`和`execute_process`之间的主要区别是什么？

1.  `add_custom_command`的两种签名分别用于什么？

1.  `add_custom_command`的`PRE_BUILD`、`PRE_LINK`和`POST_BUILD`选项有什么问题？

1.  有哪两种方式可以定义变量，以便它们可以通过`configure_file`进行替换？

1.  如何控制`configure_file`的替换行为？

1.  CMake 命令行工具执行任务的两个标志是什么？

# 答案

以下是本章节问题的答案：

1.  使用`add_custom_command`添加的命令在构建时执行，而使用`execute_process`添加的命令在配置时执行。

1.  一个签名用于创建自定义构建步骤，而另一个用于生成文件。

1.  只有`POST_BUILD`在所有生成器中都可靠地得到支持。

1.  变量可以定义为`${VAR}`或`@VAR@`。

1.  变量替换可以通过传递`@ONLY`来控制，这样只替换定义为`@VAR@`的变量，或者通过指定`COPYONLY`选项来控制，这样完全不执行任何替换。

1.  使用`cmake -E`可以直接执行常见任务。使用`cmake -P`，`.cmake`文件可以作为脚本执行。
