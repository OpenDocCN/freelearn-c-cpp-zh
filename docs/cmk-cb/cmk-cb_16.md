# 第十六章：将项目移植到 CMake

在本书的最后一章中，我们将结合前面章节中讨论的多个不同的构建块，并将其应用于一个实际项目。我们的目标将是逐步展示如何将一个非平凡的项目移植到 CMake，并讨论这样的过程中的步骤。我们将为移植您自己的项目或为遗留代码添加 CMake 支持提供建议，无论是来自 Autotools，来自手工编写的配置脚本和 Makefile，还是来自 Visual Studio 项目文件。

为了有一个具体和现实的示例，我们将使用流行的编辑器 Vim（[`www.vim.org`](https://www.vim.org)）背后的源代码，并尝试将配置和编译从 Autotools 移植到 CMake。

为了保持讨论和示例的相对简单性，我们将不尝试为整个 Vim 代码提供完整的 CMake 移植，包括所有选项。相反，我们将挑选并讨论最重要的方面，并且只构建一个核心版本的 Vim，不支持图形用户界面（GUI）。尽管如此，我们将得到一个使用 CMake 和本书中介绍的其他工具配置、构建和测试的 Vim 工作版本。

本章将涵盖以下主题：

+   移植项目时的初始步骤

+   生成文件和编写平台检查

+   检测所需的依赖项并进行链接

+   重现编译器标志

+   移植测试

+   移植安装目标

+   将项目转换为 CMake 时常见的陷阱

# 从哪里开始

我们将首先展示在哪里可以在线找到我们的示例，然后逐步讨论移植示例。

# 重现移植示例

我们将从 Vim 源代码仓库的`v8.1.0290`发布标签（[`github.com/vim/vim`](https://github.com/vim/vim)）开始，并基于 Git 提交哈希`b476cb7`进行工作。以下步骤可以通过克隆 Vim 的源代码仓库并检出该特定版本的代码来重现：

```cpp
$ git clone --single-branch -b v8.1.0290 https://github.com/vim/vim.git
```

或者，我们的解决方案可以在[`github.com/dev-cafe/vim`](https://github.com/dev-cafe/vim)的`cmake-support`分支上找到，并使用以下命令克隆到您的计算机上：

```cpp
$ git clone --single-branch -b cmake-support https://github.com/dev-cafe/vim
```

在本示例中，我们将模拟在 CMake 中使用 GNU 编译器集合构建的`./configure --enable-gui=no`配置。

为了与我们的解决方案进行比较，并获得额外的灵感，我们鼓励读者也研究 Neovim 项目（[`github.com/neovim/neovim`](https://github.com/neovim/neovim)），这是一个传统的 Vi 编辑器的分支，并提供了一个 CMake 构建系统。

# 创建顶层 CMakeLists.txt

作为开始，我们在源代码仓库的根目录中创建一个顶级的`CMakeLists.txt`，在其中设置最小 CMake 版本、项目名称和支持的语言，在本例中为 C：

```cpp
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(vim LANGUAGES C)
```

在添加任何目标或源文件之前，我们可以设置默认的构建类型。在这种情况下，我们默认使用`Release`配置，这将启用某些编译器优化：

```cpp
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
```

我们还使用便携式安装目录变量，如 GNU 软件所定义：

```cpp
include(GNUInstallDirs)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY
  ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY
  ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
  ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})
```

作为健全性检查，我们可以尝试配置和构建项目，但到目前为止还没有目标，因此构建步骤的输出将为空：

```cpp
$ mkdir -p build
$ cd build
$ cmake ..
$ cmake --build .
```

我们很快将开始添加目标，以使构建更加充实。

# 如何同时允许传统配置和 CMake 配置

CMake 的一个非常好的特性是，我们可以在源代码目录之外构建，构建目录可以是任何目录，而不必是项目目录的子目录。这意味着我们可以在不干扰先前/当前配置和构建机制的情况下将项目迁移到 CMake。对于非平凡项目的迁移，CMake 文件可以与其他构建框架共存，以允许逐步迁移，无论是选项、功能和可移植性方面，还是允许开发人员社区适应新框架。为了允许传统和 CMake 配置在一段时间内共存，一个典型的策略是将所有 CMake 代码收集在`CMakeLists.txt`文件中，并将所有辅助 CMake 源文件放在`cmake`子目录下。在我们的示例中，我们不会引入`cmake`子目录，而是将辅助文件更靠近需要它们的目标和源文件，但我们会注意保持几乎所有用于传统 Autotools 构建的文件不变，只有一个例外：我们将对自动生成的文件进行少量修改，以便将它们放置在构建目录下，而不是源代码树中。

# 记录传统构建过程的记录

在我们向配置中添加任何目标之前，通常首先记录传统构建过程的内容，并将配置和构建步骤的输出保存到日志文件中，这通常很有用。对于我们的 Vim 示例，可以使用以下方法完成：

```cpp
$ ./configure --enable-gui=no

... lot of output ...

$ make > build.log
```

在我们的情况下（`build.log`的完整内容未在此处显示），我们能够验证哪些源文件被编译以及使用了哪些编译标志（`-I. -Iproto`）

`-DHAVE_CONFIG_H -g -O2 -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=1`。从日志文件中，我们可以推断出以下内容：

+   所有对象都被链接成一个单一的二进制文件

+   不生成库文件

+   可执行目标链接了以下库：`-lSM -lICE -lXpm -lXt -lX11 -lXdmcp -lSM -lICE -lm -ltinfo -lelf -lnsl -lacl -lattr -lgpm -ldl`

# 调试迁移过程

在逐步将目标和命令迁移到 CMake 侧时，使用`message`命令打印变量值将非常有用：

```cpp
message(STATUS "for debugging printing the value of ${some_variable}")
```

通过添加选项、目标、源文件和依赖项，同时使用`message`进行调试，我们将逐步构建起一个可用的构建系统。

# 实现选项

找出传统配置向用户提供的选项（例如，通过运行`./configure --help`）。Vim 项目提供了一个非常长的选项和标志列表，为了在本章中保持讨论的简单性，我们只会在 CMake 侧实现四个选项：

```cpp
--disable-netbeans    Disable NetBeans integration support.
--disable-channel     Disable process communication support.
--enable-terminal     Enable terminal emulation support.
--with-features=TYPE  tiny, small, normal, big or huge (default: huge)
```

我们还将忽略任何 GUI 支持，并模拟`--enable-gui=no`，因为这会使示例复杂化，而对学习成果没有显著增加。

我们将在`CMakeLists.txt`中放置以下选项和默认值：

```cpp
option(ENABLE_NETBEANS "Enable netbeans" ON)
option(ENABLE_CHANNEL "Enable channel" ON)
option(ENABLE_TERMINAL "Enable terminal" ON)
```

我们将使用一个变量`FEATURES`来模拟`--with-features`标志，该变量可以通过`cmake -D FEATURES=value`来定义。我们确保如果`FEATURES`未设置，它默认为"huge"：

```cpp
if(NOT FEATURES)
  set(FEATURES "huge" CACHE STRING
    "FEATURES chosen by the user at CMake configure time")
endif()
```

我们还要确保用户为`FEATURES`提供有效的值：

```cpp
list(APPEND _available_features "tiny" "small" "normal" "big" "huge")
if(NOT FEATURES IN_LIST _available_features)
  message(FATAL_ERROR "Unknown features: \"${FEATURES}\". Allowed values are: ${_available_features}.")
endif()
set_property(CACHE FEATURES PROPERTY STRINGS ${_available_features})
```

最后一行`set_property(CACHE FEATURES PROPERTY STRINGS ${_available_features})`有一个很好的效果，即在使用`cmake-gui`配置项目时，用户会看到一个用于`FEATURES`的选择字段，列出了我们已定义的所有可用功能（另请参见[`blog.kitware.com/constraining-values-with-comboboxes-in-cmake-cmake-gui/`](https://blog.kitware.com/constraining-values-with-comboboxes-in-cmake-cmake-gui/)）。

这些选项可以放在顶层的`CMakeLists.txt`中（正如我们在这里所做的），或者可以定义在查询`ENABLE_NETBEANS`、`ENABLE_CHANNEL`、`ENABLE_TERMINAL`和`FEATURES`的目标附近。前一种策略的优势在于选项集中在一个地方，不需要遍历`CMakeLists.txt`文件树来查找选项的定义。由于我们还没有定义任何目标，我们可以从将选项保存在一个中心文件开始，但稍后我们可能会将选项定义移到更接近目标的位置，以限制范围并得到更可重用的 CMake 构建块。

# 从可执行文件和非常少的目标开始，稍后限制范围

让我们添加一些源文件。在 Vim 示例中，源文件位于`src`目录下，为了保持主`CMakeLists.txt`的可读性和可维护性，我们将创建一个新文件`src/CMakeLists.txt`，并通过在主`CMakeLists.txt`中添加以下内容来在它自己的目录范围内处理该文件：

```cpp
add_subdirectory(src)
```

在`src/CMakeLists.txt`内部，我们可以开始定义可执行目标并列出从`build.log`中提取的所有源文件：

```cpp
add_executable(vim
  arabic.c beval.c buffer.c blowfish.c crypt.c crypt_zip.c dict.c diff.c digraph.c edit.c eval.c evalfunc.c ex_cmds.c ex_cmds2.c ex_docmd.c ex_eval.c ex_getln.c farsi.c fileio.c fold.c getchar.c hardcopy.c hashtab.c if_cscope.c if_xcmdsrv.c list.c mark.c memline.c menu.c misc1.c misc2.c move.c mbyte.c normal.c ops.c option.c os_unix.c auto/pathdef.c popupmnu.c pty.c quickfix.c regexp.c screen.c search.c sha256.c spell.c spellfile.c syntax.c tag.c term.c terminal.c ui.c undo.c userfunc.c window.c libvterm/src/encoding.c libvterm/src/keyboard.c libvterm/src/mouse.c libvterm/src/parser.c libvterm/src/pen.c libvterm/src/screen.c libvterm/src/state.c libvterm/src/unicode.c libvterm/src/vterm.c netbeans.c channel.c charset.c json.c main.c memfile.c message.c version.c
  )
```

这是一个开始。在这种情况下，代码甚至不会配置，因为源文件列表包含生成的文件。在我们讨论生成的文件和链接依赖之前，我们将把这个长列表分成几个部分，以限制目标依赖的范围，并使项目更易于管理。如果我们将它们分组到目标中，我们还将使 CMake 更容易扫描源文件依赖关系，并避免出现非常长的链接行。

对于 Vim 示例，我们可以从 `src/Makefile` 和 `src/configure.ac` 中获得关于源文件分组的更多见解。从这些文件中，我们可以推断出大多数源文件是基本的和必需的。有些源文件是可选的（`netbeans.c` 应该只在 `ENABLE_NETBEANS` 为 `ON` 时构建，`channel.c` 应该只在 `ENABLE_CHANNEL` 为 `ON` 时构建）。此外，我们可能可以将所有源文件归类在 `src/libvterm/` 下，并使用 `ENABLE_TERMINAL` 使它们的编译成为可选。

通过这种方式，我们将 CMake 结构重新组织为以下树形结构：

```cpp
.
├── CMakeLists.txt
└── src
    ├── CMakeLists.txt
    └── libvterm
        └── CMakeLists.txt
```

顶级文件添加了 `src/CMakeLists.txt` 并包含 `add_subdirectory(src)`。`src/CMakeLists.txt` 文件现在包含三个目标（一个可执行文件和两个库），每个目标都带有编译定义和包含目录。我们首先定义可执行文件：

```cpp
add_executable(vim
  main.c
  )

target_compile_definitions(vim
  PRIVATE
    "HAVE_CONFIG_H"
  )
```

然后，我们定义所需的源文件：

```cpp
add_library(basic_sources "")

target_sources(basic_sources
  PRIVATE
    arabic.c beval.c blowfish.c buffer.c charset.c
    crypt.c crypt_zip.c dict.c diff.c digraph.c
    edit.c eval.c evalfunc.c ex_cmds.c ex_cmds2.c
    ex_docmd.c ex_eval.c ex_getln.c farsi.c fileio.c
    fold.c getchar.c hardcopy.c hashtab.c if_cscope.c
    if_xcmdsrv.c json.c list.c main.c mark.c
    memfile.c memline.c menu.c message.c misc1.c
    misc2.c move.c mbyte.c normal.c ops.c
    option.c os_unix.c auto/pathdef.c popupmnu.c pty.c
    quickfix.c regexp.c screen.c search.c sha256.c
    spell.c spellfile.c syntax.c tag.c term.c
    terminal.c ui.c undo.c userfunc.c version.c
    window.c
  )

target_include_directories(basic_sources
  PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/proto
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
  )

target_compile_definitions(basic_sources
  PRIVATE
    "HAVE_CONFIG_H"
  )

target_link_libraries(vim
  PUBLIC
    basic_sources
  )
```

然后，我们定义可选的源文件：

```cpp
add_library(extra_sources "")

if(ENABLE_NETBEANS)
  target_sources(extra_sources
    PRIVATE
      netbeans.c
    )
endif()

if(ENABLE_CHANNEL)
  target_sources(extra_sources
    PRIVATE
      channel.c
    )
endif()

target_include_directories(extra_sources
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/proto
    ${CMAKE_CURRENT_BINARY_DIR}
  )

target_compile_definitions(extra_sources
  PRIVATE
    "HAVE_CONFIG_H"
  )

target_link_libraries(vim
  PUBLIC
    extra_sources
  )
```

该文件还选择性地处理并链接 `src/libvterm/`，使用以下代码：

```cpp
if(ENABLE_TERMINAL)
  add_subdirectory(libvterm)

  target_link_libraries(vim
    PUBLIC
      libvterm
    )
endif()
```

相应的 `src/libvterm/CMakeLists.txt` 包含以下内容：

```cpp
add_library(libvterm "")

target_sources(libvterm
  PRIVATE
    src/encoding.c
    src/keyboard.c
    src/mouse.c
    src/parser.c
    src/pen.c
    src/screen.c
    src/state.c
    src/unicode.c
    src/vterm.c
  )

target_include_directories(libvterm
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/include
  )

target_compile_definitions(libvterm
  PRIVATE
    "HAVE_CONFIG_H"
    "INLINE="
    "VSNPRINTF=vim_vsnprintf"
    "IS_COMBINING_FUNCTION=utf_iscomposing_uint"
    "WCWIDTH_FUNCTION=utf_uint2cells"
  )
```

我们已经从记录的 `build.log` 中提取了编译定义。树形结构的优点是目标定义靠近源文件所在的位置。如果我们决定重构代码并重命名或移动目录，描述目标的 CMake 文件有机会随源文件一起移动。

我们的示例代码甚至还没有配置（除非在成功的 Autotools 构建之后尝试）：

```cpp
$ mkdir -p build
$ cd build
$ cmake ..

-- The C compiler identification is GNU 8.2.0
-- Check for working C compiler: /usr/bin/cc
-- Check for working C compiler: /usr/bin/cc -- works
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Detecting C compile features
-- Detecting C compile features - done
-- Configuring done
CMake Error at src/CMakeLists.txt:12 (add_library):
  Cannot find source file:

    auto/pathdef.c

  Tried extensions .c .C .c++ .cc .cpp .cxx .cu .m .M .mm .h .hh .h++ .hm
  .hpp .hxx .in .txx
```

我们需要生成 `auto/pathdef.c`（以及其他文件），我们将在下一节中考虑这一点。

# 生成文件和编写平台检查

事实证明，对于 Vim 代码示例，我们需要在配置时生成三个文件：`src/auto/pathdef.c`、`src/auto/config.h` 和 `src/auto/osdef.h`：

+   `pathdef.c` 记录安装路径、编译和链接标志、编译代码的用户以及主机名

+   `config.h` 包含特定于系统环境的编译定义

+   `osdef.h` 是一个包含由 `src/osdef.sh` 生成的编译定义的文件。

这种情况相当常见。我们需要根据 CMake 变量配置一个文件，执行一系列平台检查以生成 `config.h`，并在配置时执行一个脚本。特别是，平台检查对于追求可移植性的项目来说非常常见，以适应操作系统之间的微妙差异。

在原始布局中，文件在 `src` 文件夹下生成。我们不喜欢这种方法，在我们的示例 CMake 移植中将采取不同的做法：这些文件将在构建目录中生成。这样做的原因是，生成的文件通常依赖于所选的选项、编译器或构建类型，我们希望保持能够配置多个具有相同源代码的构建的能力。为了在构建目录中启用生成，我们将不得不对之前列出的文件之一的生成脚本进行最小程度的更改。

# 如何组织文件

我们将收集生成这些文件的函数在`src/autogenerate.cmake`中，包含此模块，并在定义可执行目标之前在`src/CMakeLists.txt`中调用这些函数：

```cpp
# generate config.h, pathdef.c, and osdef.h
include(autogenerate.cmake)
generate_config_h()
generate_pathdef_c()
generate_osdef_h()

add_executable(vim
  main.c
  )

# ...
```

包含的`src/autogenerate.cmake`包含其他包含功能，我们将需要这些功能来探测头文件，函数和库，以及三个函数：

```cpp
include(CheckTypeSize)
include(CheckFunctionExists)
include(CheckIncludeFiles)
include(CheckLibraryExists)
include(CheckCSourceCompiles)

function(generate_config_h)
  # ... to be written
endfunction()

function(generate_pathdef_c)
  # ... to be written
endfunction()

function(generate_osdef_h)
  # ... to be written
endfunction()
```

我们选择使用函数生成文件，而不是宏或“裸”CMake 代码。正如我们在前几章中讨论的那样，这避免了许多陷阱：

+   它使我们能够避免文件被多次生成，以防我们不小心多次包含该模块。如第五章中的*重新定义函数和宏*所述，在第七章，*项目结构*中，我们可以使用包含保护来防止不小心多次运行代码。

+   它确保完全控制函数内部定义的变量的作用域。这避免了这些定义泄漏并污染主作用域。

# 根据系统环境配置预处理器定义

`config.h`文件是从`src/config.h.in`生成的，其中包含根据系统能力配置的预处理器标志：

```cpp
/* Define if we have EBCDIC code */
#undef EBCDIC

/* Define unless no X support found */
#undef HAVE_X11

/* Define when terminfo support found */
#undef TERMINFO

/* Define when termcap.h contains ospeed */
#undef HAVE_OSPEED

/* ... */
```

从`src/config.h`生成的示例可以像这个示例一样开始（定义可能因环境而异）：

```cpp
/* Define if we have EBCDIC code */
/* #undef EBCDIC */

/* Define unless no X support found */
#define HAVE_X11 1

/* Define when terminfo support found */
#define TERMINFO 1

/* Define when termcap.h contains ospeed */
/* #undef HAVE_OSPEED */

/* ... */
```

平台检查的一个很好的资源是这个页面：[`www.vtk.org/Wiki/CMake:How_To_Write_Platform_Checks`](https://www.vtk.org/Wiki/CMake:How_To_Write_Platform_Checks)。

在`src/configure.ac`中，我们可以检查需要执行哪些平台检查以设置相应的预处理器定义。

我们将使用`#cmakedefine`（[`cmake.org/cmake/help/v3.5/command/configure_file.html?highlight=cmakedefine`](https://cmake.org/cmake/help/v3.5/command/configure_file.html?highlight=cmakedefine)），并确保我们不会破坏现有的 Autotools 构建，我们将复制`config.h.in`到`config.h.cmake.in`，并将所有`#undef SOME_DEFINITION`更改为`#cmakedefine SOME_DEFINITION @SOME_DEFINITION@`。

在`generate_config_h`函数中，我们首先定义一些变量：

```cpp
set(TERMINFO 1)
set(UNIX 1)

# this is hardcoded to keep the discussion in the book chapter
# which describes the migration to CMake simpler
set(TIME_WITH_SYS_TIME 1)
set(RETSIGTYPE void)
set(SIGRETURN return)

find_package(X11)
set(HAVE_X11 ${X11_FOUND})
```

然后，我们执行一些类型大小检查：

```cpp
check_type_size("int" VIM_SIZEOF_INT)
check_type_size("long" VIM_SIZEOF_LONG)
check_type_size("time_t" SIZEOF_TIME_T)
check_type_size("off_t" SIZEOF_OFF_T)
```

然后，我们遍历函数并检查系统是否能够解析它们：

```cpp
foreach(
  _function IN ITEMS
  fchdir fchown fchmod fsync getcwd getpseudotty
  getpwent getpwnam getpwuid getrlimit gettimeofday getwd lstat
  memset mkdtemp nanosleep opendir putenv qsort readlink select setenv
  getpgid setpgid setsid sigaltstack sigstack sigset sigsetjmp sigaction
  sigprocmask sigvec strcasecmp strerror strftime stricmp strncasecmp
  strnicmp strpbrk strtol towlower towupper iswupper
  usleep utime utimes mblen ftruncate
  )

  string(TOUPPER "${_function}" _function_uppercase)
  check_function_exists(${_function} HAVE_${_function_uppercase})
endforeach()
```

我们验证特定的库是否包含特定的函数：

```cpp
check_library_exists(tinfo tgetent "" HAVE_TGETENT)

if(NOT HAVE_TGETENT)
  message(FATAL_ERROR "Could not find the tgetent() function. You need to install a terminal library; for example ncurses.")
endif()
```

然后，我们遍历头文件并检查它们是否可用：

```cpp
foreach(
  _header IN ITEMS
  setjmp.h dirent.h
  stdint.h stdlib.h string.h
  sys/select.h sys/utsname.h termcap.h fcntl.h
  sgtty.h sys/ioctl.h sys/time.h sys/types.h
  termio.h iconv.h inttypes.h langinfo.h math.h
  unistd.h stropts.h errno.h sys/resource.h
  sys/systeminfo.h locale.h sys/stream.h termios.h
  libc.h sys/statfs.h poll.h sys/poll.h pwd.h
  utime.h sys/param.h libintl.h libgen.h
  util/debug.h util/msg18n.h frame.h sys/acl.h
  sys/access.h sys/sysinfo.h wchar.h wctype.h
  )

  string(TOUPPER "${_header}" _header_uppercase)
  string(REPLACE "/" "_" _header_normalized "${_header_uppercase}")
  string(REPLACE "." "_" _header_normalized "${_header_normalized}")
  check_include_files(${_header} HAVE_${_header_normalized})
endforeach()
```

然后，我们将 CMake 选项从主`CMakeLists.txt`转换为预处理器定义：

```cpp
string(TOUPPER "${FEATURES}" _features_upper)
set(FEAT_${_features_upper} 1)

set(FEAT_NETBEANS_INTG ${ENABLE_NETBEANS})
set(FEAT_JOB_CHANNEL ${ENABLE_CHANNEL})
set(FEAT_TERMINAL ${ENABLE_TERMINAL})
```

最后，我们检查是否能够编译特定的代码片段：

```cpp
check_c_source_compiles(
  "
  #include <sys/types.h>
  #include <sys/stat.h>
  int
  main ()
  {
          struct stat st;
          int n;

          stat(\"/\", &st);
          n = (int)st.st_blksize;
    ;
    return 0;
  }
  "
  HAVE_ST_BLKSIZE
  )
```

然后使用定义的变量来配置`src/config.h.cmake.in`到`config.h`，这完成了`generate_config_h`函数：

```cpp
configure_file(
  ${CMAKE_CURRENT_LIST_DIR}/config.h.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/auto/config.h
  @ONLY
  )
```

# 使用路径和编译器标志配置文件

我们生成`pathdef.c`从`src/pathdef.c.in`：

```cpp
#include "vim.h"

char_u *default_vim_dir = (char_u *)"@_default_vim_dir@";
char_u *default_vimruntime_dir = (char_u *)"@_default_vimruntime_dir@";
char_u *all_cflags = (char_u *)"@_all_cflags@";
char_u *all_lflags = (char_u *)"@_all_lflags@";
char_u *compiled_user = (char_u *)"@_compiled_user@";
char_u *compiled_sys = (char_u *)"@_compiled_sys@";
```

`generate_pathdef_c`函数配置`src/pathdef.c.in`，但我们省略了链接标志以简化：

```cpp
function(generate_pathdef_c)
  set(_default_vim_dir ${CMAKE_INSTALL_PREFIX})
  set(_default_vimruntime_dir ${_default_vim_dir})

  set(_all_cflags "${CMAKE_C_COMPILER} ${CMAKE_C_FLAGS}")
  if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(_all_cflags "${_all_cflags} ${CMAKE_C_FLAGS_RELEASE}")
  else()
    set(_all_cflags "${_all_cflags} ${CMAKE_C_FLAGS_DEBUG}")
  endif()

  # it would require a bit more work and execute commands at build time
  # to get the link line into the binary
  set(_all_lflags "undefined")

  if(WIN32)
    set(_compiled_user $ENV{USERNAME})
  else()
    set(_compiled_user $ENV{USER})
  endif()

  cmake_host_system_information(RESULT _compiled_sys QUERY HOSTNAME)

  configure_file(
    ${CMAKE_CURRENT_LIST_DIR}/pathdef.c.in
    ${CMAKE_CURRENT_BINARY_DIR}/auto/pathdef.c
    @ONLY
    )
endfunction()
```

# 在配置时执行 shell 脚本

最后，我们使用以下函数生成`osdef.h`：

```cpp
function(generate_osdef_h)
  find_program(BASH_EXECUTABLE bash)

  execute_process(
    COMMAND
      ${BASH_EXECUTABLE} osdef.sh ${CMAKE_CURRENT_BINARY_DIR}
    WORKING_DIRECTORY
      ${CMAKE_CURRENT_LIST_DIR}
    )
endfunction()
```

为了在 `${CMAKE_CURRENT_BINARY_DIR}/src/auto` 而不是 `src/auto` 中生成 `osdef.h`，我们不得不修改 `osdef.sh` 以接受 `${CMAKE_CURRENT_BINARY_DIR}` 作为命令行参数。

在`osdef.sh`内部，我们检查是否给出了这个参数：

```cpp
if [ $# -eq 0 ]
  then
    # there are no arguments
    # assume the target directory is current directory
    target_directory=$PWD
  else
    # target directory is provided as argument
    target_directory=$1
fi
```

然后，我们生成 `${target_directory}/auto/osdef.h`。为此，我们还需要调整`osdef.sh`内部的下述编译行：

```cpp
$CC -I. -I$srcdir -I${target_directory} -E osdef0.c >osdef0.cc
```

# 检测所需依赖项和链接

现在我们已经将所有生成的文件放置到位，让我们重新尝试构建。我们应该能够配置和编译源代码，但我们无法链接：

```cpp
$ mkdir -p build
$ cd build
$ cmake ..
$ cmake --build .

...
Scanning dependencies of target vim
[ 98%] Building C object src/CMakeFiles/vim.dir/main.c.o
[100%] Linking C executable ../bin/vim
../lib64/libbasic_sources.a(term.c.o): In function `set_shellsize.part.12':
term.c:(.text+0x2bd): undefined reference to `tputs'
../lib64/libbasic_sources.a(term.c.o): In function `getlinecol':
term.c:(.text+0x902): undefined reference to `tgetent'
term.c:(.text+0x915): undefined reference to `tgetent'
term.c:(.text+0x935): undefined reference to `tgetnum'
term.c:(.text+0x948): undefined reference to `tgetnum'

... many other undefined references ...
```

同样，我们可以从 Autotools 编译的日志文件中，特别是链接行中获得灵感，通过在`src/CMakeLists.txt`中添加以下代码来解决缺失的依赖：

```cpp
# find X11 and link to it
find_package(X11 REQUIRED)
if(X11_FOUND)
  target_link_libraries(vim
    PUBLIC
      ${X11_LIBRARIES}
    )
endif()

# a couple of more system libraries that the code requires
foreach(_library IN ITEMS Xt SM m tinfo acl gpm dl)
  find_library(_${_library}_found ${_library} REQUIRED)
  if(_${_library}_found)
    target_link_libraries(vim
      PUBLIC
        ${_library}
      )
  endif()
endforeach()
```

注意我们是如何一次向目标添加一个库依赖，而不必构建和携带一个变量中的库列表，这会导致更脆弱的 CMake 代码，因为变量在过程中可能会被破坏，尤其是在大型项目中。

通过这个更改，代码编译并链接：

```cpp
$ cmake --build .

...
Scanning dependencies of target vim
[ 98%] Building C object src/CMakeFiles/vim.dir/main.c.o
[100%] Linking C executable ../bin/vim
[100%] Built target vim
```

我们现在可以尝试执行编译后的二进制文件，并用我们新编译的 Vim 版本编辑一些文件！

# 重现编译器标志

现在让我们尝试调整编译器标志以反映参考构建。

# 定义编译器标志

到目前为止，我们还没有定义任何自定义编译器标志，但从参考 Autotools 构建中，我们记得代码是用`-g -U_FORTIFY_SOURCE`编译的

`-D_FORTIFY_SOURCE=1 -O2` 使用 GNU C 编译器。

我们的第一个方法可能是定义以下内容：

```cpp
if(CMAKE_C_COMPILER_ID MATCHES GNU)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=1 -O2")
endif()
```

而且，我们会将这段代码放在`src/CMakeLists.txt`的顶部，就在生成源文件之前（因为`pathdef.c`使用了`${CMAKE_C_FLAGS}`）：

```cpp
# <- we will define flags right here

include(autogenerate.cmake)
generate_config_h()
generate_pathdef_c()
generate_osdef_h()
```

对编译器标志定义的一个小改进是将`-O2`定义为`Release`配置标志，并为`Debug`配置关闭优化：

```cpp
if(CMAKE_C_COMPILER_ID MATCHES GNU)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -U_FORTIFY_SOURCE 
-D_FORTIFY_SOURCE=1")
  set(CMAKE_C_FLAGS_RELEASE "-O2")
  set(CMAKE_C_FLAGS_DEBUG "-O0")
endif()
```

请使用`make VERBOSE=1`验证构建是否使用了预期的标志。

# 编译器标志的范围

在这个特定的示例项目中，所有源文件使用相同的编译标志。对于其他项目，我们可能更倾向于不全局定义编译标志，而是使用`target_compile_options`为每个目标单独定义标志。这样做的好处是更灵活和更局部的范围。在我们这里的例子中，代价可能是不必要的代码重复。

# 移植测试

现在让我们讨论如何将测试从参考构建移植到我们的 CMake 构建。

# 开始

如果正在移植的项目包含测试目标或任何形式的自动化测试或测试脚本，第一步将再次是运行传统的测试步骤并记录使用的命令。对于 Vim 项目，起点是`src/testdir/Makefile`。在 CMake 侧定义测试可能是有意义的，接近`src/testdir/Makefile`和测试脚本，我们将选择在`src/testdir/CMakeLists.txt`中定义测试。为了处理这样的文件，我们必须在其`src/CMakeLists.txt`中引用它：

```cpp
add_subdirectory(testdir)
```

我们还应该在顶层`CMakeLists.txt`中启用测试目标，就在处理`src/CMakeLists.txt`之前：

```cpp
# enable the test target
enable_testing()

# process src/CMakeLists.txt in its own scope
add_subdirectory(src)
```

到目前为止，在我们向`src/testdir/CMakeLists.txt`填充`add_test`指令之前，测试目标还是空的。`add_test`中最少需要指定的是测试名称和一个运行命令。该命令可以是任何语言编写的任何脚本。对于 CMake 来说，关键的是如果测试成功，脚本返回零，如果测试失败，则返回非零。更多详情，我们请读者参考第四章，*创建和运行测试*。对于 Vim 的情况，我们需要更多来适应多步骤测试，我们将在下一节讨论。

# 实现多步骤测试

在`src/testdir/Makefile`中的目标表明 Vim 代码以多步骤测试运行：首先，`vim`可执行文件处理一个脚本并生成一个输出文件，然后在第二步中，输出文件与参考文件进行比较，如果这些文件没有差异，则测试成功。临时文件随后在第三步中被删除。这可能无法以可移植的方式适应单个`add_test`命令，因为`add_test`只能执行一个命令。一个解决方案是将测试步骤定义在一个 Python 脚本中，并用一些参数执行该 Python 脚本。我们将在这里介绍的另一种替代方案也是跨平台的，即将测试步骤定义在一个单独的 CMake 脚本中，并从`add_test`执行该脚本。我们将在`src/testdir/test.cmake`中定义测试步骤：

```cpp
function(execute_test _vim_executable _working_dir _test_script)
  # generates test.out
  execute_process(
    COMMAND ${_vim_executable} -f -u unix.vim -U NONE --noplugin --not-a-term -s dotest.in ${_test_script}.in
    WORKING_DIRECTORY ${_working_dir}
    )

  # compares test*.ok and test.out
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E compare_files ${_test_script}.ok test.out
    WORKING_DIRECTORY ${_working_dir}
    RESULT_VARIABLE files_differ
    OUTPUT_QUIET
    ERROR_QUIET
    )

  # removes leftovers
  file(REMOVE ${_working_dir}/Xdotest)

  # we let the test fail if the files differ
  if(files_differ)
    message(SEND_ERROR "test ${_test_script} failed")
  endif()
endfunction()

execute_test(${VIM_EXECUTABLE} ${WORKING_DIR} ${TEST_SCRIPT})
```

再次，我们选择函数而非宏来确保变量不会逃逸函数作用域。我们将处理这个脚本，该脚本将调用`execute_test`函数。然而，我们必须确保从外部定义了`${VIM_EXECUTABLE}`、`${WORKING_DIR}`和`${TEST_SCRIPT}`。这些在`src/testdir/CMakeLists.txt`中定义：

```cpp
add_test(
  NAME
    test1
  COMMAND
    ${CMAKE_COMMAND} -D VIM_EXECUTABLE=$<TARGET_FILE:vim>
                     -D WORKING_DIR=${CMAKE_CURRENT_LIST_DIR}
                     -D TEST_SCRIPT=test1
                     -P ${CMAKE_CURRENT_LIST_DIR}/test.cmake
  WORKING_DIRECTORY
    ${PROJECT_BINARY_DIR}
  )
```

Vim 项目有许多测试，但在本例中，我们只移植了一个（test1）作为概念验证。

# 测试建议

我们至少可以给出两个关于移植测试的建议。首先，确保测试不会总是报告成功，如果代码被破坏或参考数据被更改，请验证测试是否失败。其次，为测试添加`COST`估计，以便在并行运行时，较长的测试首先启动，以最小化总测试时间（参见第四章，*创建和运行测试*，第 8 个配方，*并行运行测试*）。

# 移植安装目标

我们现在可以配置、编译、链接和测试代码，但我们缺少安装目标，我们将在本节中添加它。

这是 Autotools 构建和安装代码的方法：

```cpp
$ ./configure --prefix=/some/install/path
$ make
$ make install
```

这就是 CMake 的方式：

```cpp
$ mkdir -p build
$ cd build
$ cmake -D CMAKE_INSTALL_PREFIX=/some/install/path ..
$ cmake --build .
$ cmake --build . --target install
```

要添加安装目标，我们需在`src/CMakeLists.txt`中添加以下代码片段：

```cpp
install(
  TARGETS
    vim
  RUNTIME DESTINATION
    ${CMAKE_INSTALL_BINDIR}
  )
```

在本例中，我们只安装了可执行文件。Vim 项目在安装二进制文件的同时安装了大量文件（符号链接和文档文件）。为了使本节易于理解，我们没有在本例迁移中安装所有其他文件。对于你自己的项目，你应该验证安装步骤的结果是否与遗留构建框架的安装目标相匹配。

# 进一步的步骤

成功移植到 CMake 后，下一步应该是进一步限定目标和变量的范围：考虑将选项、目标和变量移动到它们被使用和修改的位置附近。避免全局变量，因为它们会强制 CMake 命令的顺序，而这个顺序可能不明显，会导致脆弱的 CMake 代码。一种强制分离变量范围的方法是将大型项目划分为 CMake 项目，这些项目使用超级构建模式（参见第八章，*超级构建模式*）。考虑将大型`CMakeLists.txt`文件拆分为较小的模块。

接下来的步骤可能是在其他平台和操作系统上测试配置和编译，以便使 CMake 代码更加通用和防弹，并使其更具可移植性。

最后，在将项目迁移到新的构建框架时，开发社区也需要适应它。通过培训、文档和代码审查帮助你的同事。在将代码移植到 CMake 时，最难的部分可能是改变人的习惯。

# 转换项目到 CMake 时的总结和常见陷阱

让我们总结一下本章我们取得了哪些成就以及我们学到了什么。

# 代码变更总结

在本章中，我们讨论了如何将项目移植到 CMake。我们以 Vim 项目为例，并添加了以下文件：

```cpp
.
├── CMakeLists.txt
└── src
    ├── autogenerate.cmake
    ├── CMakeLists.txt
    ├── config.h.cmake.in
    ├── libvterm
    │   └── CMakeLists.txt
    ├── pathdef.c.in
    └── testdir
        ├── CMakeLists.txt
        └── test.cmake
```

可以在线浏览变更：[`github.com/dev-cafe/vim/compare/b476cb7...cmake-support`](https://github.com/dev-cafe/vim/compare/b476cb7...cmake-support)。

这是一个不完整的 CMake 移植概念证明，我们省略了许多选项和调整以简化，并试图专注于最突出的特性和步骤。

# 常见陷阱

我们希望通过指出转向 CMake 时的一些常见陷阱来结束这次讨论。

+   **全局变量是代码异味**：这在任何编程语言中都是如此，CMake 也不例外。跨越 CMake 文件的变量，特别是从叶子到父级`CMakeLists.txt`文件“向上”传递的变量，表明代码存在问题。通常有更好的方式来传递依赖。理想情况下，依赖应该通过目标来导入。不要将一系列库组合成一个变量并在文件之间传递该变量，而是将库一个接一个地链接到它们定义的位置附近。不要将源文件组合成变量，而是使用`target_sources`添加源文件。在链接库时，如果可用，使用导入的目标而不是变量。

+   **最小化顺序影响**：CMake 不是一种声明式语言，但我们也不应该用命令式范式来处理它。强制严格顺序的 CMake 源码往往比较脆弱。这也与变量的讨论有关（见前一段）。某些语句和模块的顺序是必要的，但为了得到稳健的 CMake 框架，我们应该避免不必要的顺序强制。使用`target_sources`、`target_compile_definitions`、`target_include_directories`和`target_link_libraries`。避免全局范围的语句，如`add_definitions`、`include_directories`和`link_libraries`。避免全局定义编译标志。如果可能，为每个目标定义编译标志。

+   **不要将生成的文件放置在构建目录之外**：强烈建议永远不要将生成的文件放置在构建目录之外。这样做的原因是，生成的文件通常依赖于所选的选项、编译器或构建类型，而将文件写入源代码树中，我们放弃了维护多个具有相同源代码的构建的可能性，并且使构建步骤的可重复性变得复杂。

+   **优先使用函数而非宏**：它们具有不同的作用域，函数作用域是有限的。所有变量修改都需要明确标记，这也向读者表明了变量重定义。当你必须使用宏时使用，但如果你能使用函数，则优先使用函数。

+   **避免 shell 命令**：它们可能不兼容其他平台（如 Windows）。优先使用 CMake 的等效命令。如果没有可用的 CMake 等效命令，考虑调用 Python 脚本。

+   **在 Fortran 项目中，注意后缀大小写**：需要预处理的 Fortran 源文件应具有大写的`.F90`后缀。不需要预处理的源文件应具有小写的`.f90`后缀。

+   **避免显式路径**：无论是在定义目标时还是在引用文件时都是如此。使用`CMAKE_CURRENT_LIST_DIR`来引用当前路径。这样做的好处是，当你移动或重命名目录时，它仍然有效。

+   **模块包含不应是函数调用**：将 CMake 代码模块化是一个好的策略，但包含模块理想情况下不应执行 CMake 代码。相反，应将 CMake 代码封装到函数和宏中，并在包含模块后显式调用这些函数和宏。这可以防止无意中多次包含模块时产生的不良后果，并使执行 CMake 代码模块的动作对读者更加明确。
