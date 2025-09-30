

# 第十三章：管理第三方库的现代方法

在现代软件开发中，对第三方库的依赖几乎是不可避免的。从基础组件，如用于安全通信的 OpenSSL 和用于广泛 C++ 库的 Boost，甚至构成 C++ 编程基石的标准库，外部库对于构建功能强大和高效的程序至关重要。这种依赖性突显了理解如何在 C++ 生态系统中管理第三方库的重要性。

由于这些库的复杂性和多样性，开发人员掌握第三方库管理的基本知识至关重要。这种知识不仅有助于将这些库无缝集成到项目中，而且还会影响部署策略。库的编译方法，无论是静态还是动态，都会直接影响部署的文件数量和应用程序的整体影响范围。

与一些受益于标准化库生态系统的其他编程语言不同，C++ 由于缺乏这样的统一系统而面临独特的挑战。本章深入探讨了 C++ 中第三方库管理的现有解决方案，例如 vcpkg、Conan 以及其他工具。通过检查这些工具，我们旨在提供见解，了解哪种解决方案可能最适合您的项目需求，考虑因素包括平台兼容性、易用性和库目录的范围。

随着我们浏览这些解决方案，我们的目标是为您提供知识，以便您在 C++ 项目中集成和管理第三方库时做出明智的决定，从而提高您的开发工作流程和软件质量。

# 链接和共享 V 线程概述：线程静态库

在 C 和 C++ 开发背景下，第三方实体是开发人员将其集成到项目中的外部库或框架。这些实体旨在提高功能或利用现有解决方案。这些第三方组件在范围上可能差异很大，从最小实用库到提供广泛功能的全面框架。

将第三方库集成到项目中的过程涉及使用概述这些库接口的头文件。这些头文件包含库提供的类、函数和变量的声明，使编译器能够理解成功编译所需的签名和结构。在 C++ 源文件中包含头文件实际上是将头文件的内容连接到包含点，从而允许访问库的接口，而无需在源文件中嵌入实际的实现。

这些库的实现通过编译后的目标代码提供，通常以静态库或共享库的形式分发。静态库是目标文件的归档，由链接器直接合并到最终的可执行文件中，由于库代码的嵌入，导致可执行文件大小更大。另一方面，共享库，在 Windows 上称为**动态链接库（DLLs**），或在类 Unix 系统上称为**共享对象（SOs**），不会嵌入到可执行文件中。相反，包括对这些库的引用，操作系统在运行时将它们加载到内存中。这种机制允许多个应用程序利用相同的库代码，从而节省内存。

共享库旨在促进多个应用程序之间共享常见库，如 libc 或 C++标准库。这种做法对于频繁使用的库特别有利。这种设计从理论上也允许用户在不升级整个应用程序的情况下更新共享库。然而，在实践中，这并不总是无缝的，可能会引入兼容性问题，使得应用程序提供其依赖项作为共享库的优势减少。此外，选择共享库而不是静态库可以减少链接器时间，因为链接器不需要将库代码嵌入到可执行文件中，这可以加快构建过程。

链接器在这个过程中扮演着至关重要的角色，它将各种目标文件和库合并成一个单一的可执行文件或库，并在过程中解决符号引用，以确保最终的二进制文件完整且可执行。

在静态链接和动态链接之间的选择对应用程序的性能、大小和部署策略有显著影响。静态链接通过创建自包含的可执行文件简化了部署，但代价是文件大小更大，以及需要重新编译以更新库。动态链接，通过在应用程序之间共享库代码以减少内存使用并简化库更新，在部署中引入了复杂性，以确保满足所有依赖项。

考虑到与链接外部共享对象相关的复杂性以及 C++中模板代码的广泛应用，许多库开发者已经开始倾向于提供“仅头文件”库。仅头文件库是一个完全包含在头文件中的库，没有单独的实现文件或预编译的二进制文件。这意味着所有代码，包括函数和类定义，都包含在头文件中。

这种方法显著简化了集成过程。当开发者从一个仅包含头文件的库中包含头文件时，他们不仅仅是包含接口声明，还包括整个实现。因此，不需要对库的实现进行单独的编译或链接；当包含头文件时，编译器将库的代码直接包含并编译到开发者的源代码中。这种直接包含可能导致编译器进行更有效的内联和优化，从而可能由于消除了函数调用开销而生成更快的可执行代码。

然而，值得注意的是，虽然仅包含头文件的库提供了便利和易于集成的优势，但它们也有一些缺点。由于整个库被包含并编译到包含它的每个源文件中，这可能导致编译时间增加，尤其是在大型库或包含库在多个文件中的项目中。此外，任何对头文件的更改都需要重新编译包含它的所有源文件，这可能会进一步增加开发时间。

尽管存在缺点，但由于其分发和使用简单，C++中的仅包含头文件的方案对许多开发者和用户来说极具吸引力。此外，它有助于避免链接问题，并为模板密集型库提供好处。这种模式在那些大量使用模板的库中尤为普遍，例如提供元编程功能的库，因为模板必须在编译时完整地提供给编译器，这使得仅包含头文件的模型成为一种自然的选择。

从本质上讲，C++项目中第三方依赖项的管理涉及对头文件、静态库和共享库的深入了解，以及链接过程的复杂性。开发者必须仔细考虑在应用需求和部署环境中静态链接和动态链接之间的权衡，平衡性能、大小和维护便利性等因素。

# C++中管理第三方库

管理第三方库是 C++开发的关键方面。虽然 C++没有标准化的包管理器，但已经采用了各种方法和工具来简化此过程，每种方法都有自己的实践和受支持的平台。

## 使用操作系统包管理器安装库

许多开发者依赖于操作系统的包管理器来安装第三方库。在 Ubuntu 和其他基于 Debian 的系统上，通常使用`apt`：

```cpp
sudo apt install libboost-all-dev
```

对于基于 Red Hat 的系统，`yum`或其继任者`dnf`是首选选项：

```cpp
sudo yum install boost-devel
```

在 macOS 上，Homebrew 是管理包的流行选择：

```cpp
brew install boost
```

Windows 用户通常转向 Chocolatey 或`vcpkg`（后者也作为 Windows 之外的通用 C++库管理器使用）：

```cpp
choco install boost
```

这些操作系统包管理器对于常见库来说很方便，但可能并不总是提供最新的版本或开发所需的具体配置。

## 通过子模块使用 Git 作为第三方管理器

Git 子模块允许开发者直接在其仓库中包含和管理第三方库的源代码。此方法有利于确保所有团队成员和构建系统使用库的确切版本。添加子模块并将其与 CMake 集成的典型工作流程可能如下所示：

```cpp
git submodule add https://github.com/google/googletest.git external/googletest
git submodule update --init
```

在`CMakeLists.txt`中，您将包括子模块：

```cpp
add_subdirectory(external/googletest)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
```

此方法将您的项目与特定版本的库紧密耦合，并通过 Git 促进更新跟踪。

## 使用 CMake FetchContent 下载库

CMake 的`FetchContent`模块通过在配置时下载依赖项，而不需要在您的仓库中直接包含它们，提供了一种比子模块更灵活的替代方案：

```cpp
include(FetchContent)
FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.7.3
)
FetchContent_MakeAvailable(json)
```

此方法与 Git 子模块不同，因为它不需要库的源代码存在于您的仓库中或手动更新它。`FetchContent`动态检索指定的版本，使其更容易管理和更新依赖项。

# Conan – 高级依赖管理

Conan 是 C 和 C++的强大包管理器，简化了集成第三方库和在各种平台和配置中管理依赖项的过程。它因其能够处理库的多个版本、复杂的依赖图和不同的构建配置而脱颖而出，是现代 C++开发的必备工具。

## Conan 配置和功能

Conan 的配置存储在`conanfile.txt`或`conanfile.py`中，开发者在此指定所需的库、版本、设置和选项。此文件作为项目依赖项的清单，允许对项目中使用的库进行精确控制。

**关键特性**：

+   **多平台支持**：Conan 旨在在 Windows、Linux、macOS 和 FreeBSD 上运行，在不同操作系统上提供一致的经验

+   **构建配置管理**：开发者可以指定设置，如编译器版本、架构和构建类型（调试、发布），以确保项目的兼容性和最佳构建

+   **版本管理**：Conan 可以管理同一库的多个版本，允许项目根据需要依赖特定版本

+   **依赖解析**：它自动解析和下载传递依赖项，确保所有必需的库在构建过程中可用

## 图书馆位置和康南中心

Conan 包的主要仓库是**康南中心**，这是一个包含大量开源 C 和 C++库的集合。康南中心是查找和下载包的首选地点，但开发者也可以为他们的项目指定自定义或私有仓库。

除了 Conan Center 之外，公司和开发团队可以托管自己的 Conan 服务器或使用 Artifactory 等服务来管理私有或专有包，从而在组织内部实现依赖项管理的集中化方法。

## 配置静态或动态链接

Conan 允许开发者指定是否为库使用静态或动态链接。这通常通过 `conanfile.txt` 或 `conanfile.py` 中的选项来完成。以下是一个示例：

```cpp
[options]
Poco:shared=True  # Use dynamic linking for Poco
Or in conanfile.py:
class MyProject(ConanFile):
    requires = “poco/1.10.1”
    default_options = {“poco:shared”: True}
```

这些设置指示 Conan 下载并使用指定库的动态版本。同样，将选项设置为 `False` 将优先考虑静态库。需要注意的是，并非所有包都支持这两种链接选项，这取决于它们是如何为 Conan 打包的。

## 通过自定义包扩展 Conan

Conan 的一个优势是其可扩展性。如果所需的库在 Conan Center 中不可用或不符合特定需求，开发者可以创建并贡献他们自己的包。Conan 提供了一个基于 Python 的开发工具包来创建包，其中包括定义构建过程、依赖项和包元数据的工具。

为了创建一个 Conan 包，开发者定义 `conanfile.py` 文件，该文件描述了如何获取、构建和打包库。该文件包括 `source()`、`build()` 和 `package()` 等方法，这些方法在包创建过程中由 Conan 执行。

一旦开发了一个包，它可以通过提交以供包含或通过私有仓库进行分发来通过 Conan Center 进行共享，以保持对分发和使用的控制。

Conan 的灵活性、对多个平台和配置的支持以及其全面的包仓库使其成为 C++ 开发者的无价之宝。通过利用 Conan，团队可以简化他们的依赖管理流程，确保在不同环境中构建的一致性和可重复性。能够配置静态或动态链接，以及可以扩展仓库以包含自定义包的选项，突显了 Conan 对不同项目需求的适应性。无论是与广泛使用的开源库还是专用专有代码一起工作，Conan 都提供了一个强大而有效的框架来高效有效地管理 C++ 依赖项。

Conan 是一个专门的 C++ 包管理器，在管理库的不同版本及其依赖项方面表现出色。它独立于操作系统的包管理器，并提供高水平的控制和灵活性。典型的 Conan 工作流程涉及创建 `conanfile.txt` 或 `conanfile.py` 来声明依赖项。

# CMake 集成

CMake 因其强大的脚本能力和跨平台支持而在 C++ 项目中得到广泛应用。将 Conan 与 CMake 集成可以显著简化依赖项管理的过程。以下是如何实现这种集成的步骤：

+   在项目的 `CMakeLists.txt` 中由 Conan 生成的 `conanbuildinfo.cmake` 文件：

    ```cpp
    include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
    ```

    ```cpp
    conan_basic_setup(TARGETS)
    ```

    此脚本设置了必要的 `include` 路径和库路径，并定义了由 Conan 管理的依赖项，使它们可用于您的 CMake 项目。

+   `conan_basic_setup()` 中的 `TARGETS` 选项为您的 Conan 依赖项生成 CMake 目标，允许您使用 CMake 中的 `target_link_libraries()` 函数来链接它们：

    ```cpp
    target_link_libraries(my_project_target CONAN_PKG::poco)
    ```

    此方法提供了一种清晰明确的方式来链接您的项目目标到由 Conan 管理的库。

## 其他构建系统集成

Conan 的灵活性也扩展到其他构建系统，使其能够适应各种项目需求：

+   可包含在 Makefile 中的 `include` 路径、库路径和标志：

    ```cpp
    include conanbuildinfo.mak
    ```

+   可导入到 Visual Studio 项目的 `.props` 文件，提供与 MSBuild 生态系统的无缝集成。

+   **Bazel, Meson 和其他工具**: 尽管对某些构建系统（如 Bazel 或 Meson）的直接支持可能需要自定义集成脚本或工具，但 Conan 社区通常贡献生成器和工具来弥合这些差距，使 Conan 能够扩展到几乎任何构建系统。

## 自定义集成

对于没有直接支持或对项目有独特要求的构建系统，Conan 提供了自定义生成文件或编写自定义生成器的功能。这允许开发者根据其特定的构建过程定制集成，使 Conan 成为依赖项管理的高度适应性的工具。

## 结论

Conan 与 CMake 和其他构建系统的集成凸显了其在 C++ 项目包管理器方面的多功能性。通过提供将依赖项纳入各种构建环境的简单机制，Conan 不仅简化了依赖项管理，还增强了不同平台和配置下的构建可重复性和一致性。无论您是在使用广泛使用的构建系统（如 CMake）还是更专业的设置，Conan 的灵活集成选项都能确保您能够保持高效和流畅的开发工作流程。

# vcpkg

由微软开发的 vcpkg 是一个跨平台的 C++ 包管理器，简化了获取和构建 C++ 开源库的过程。它旨在与 CMake 和其他构建系统无缝协作，提供一种简单一致的方式来管理 C++ 库依赖项。

## 与 Conan 的关键区别

虽然 vcpkg 和 Conan 都旨在简化 C++ 项目的依赖项管理，但它们在方法和生态系统方面存在显著差异：

+   **起源和支持**: vcpkg 由微软创建并维护，确保了与 Visual Studio 和 MSBuild 系统的紧密集成，尽管它在不同的平台和开发环境中仍然完全功能性和有用。

+   **包源**：vcpkg 专注于从源编译，确保库使用与消费项目相同的编译器和设置进行构建。这种方法与 Conan 形成对比，Conan 可以管理预编译的二进制文件，允许更快地集成，但可能导致二进制不兼容问题。

+   **集成**：vcpkg 与 CMake 和 Visual Studio 原生集成，提供项目级集成的清单文件。这对于已经使用这些工具的项目尤其有吸引力，可以提供更无缝的集成体验。

+   **生态系统和库**：这两个包管理器都拥有大量可用的库，但它们的生态系统可能因每个项目的社区和支持而略有不同。

## 操作系统支持

vcpkg 旨在跨平台，支持以下平台：

+   Windows

+   Linux

+   macOS

这广泛的兼容性使其成为在多样化开发环境中工作的开发者的多功能选择。

## 使用 vcpkg 配置项目的示例

为了说明在项目中使用 vcpkg，让我们通过一个简单的示例来展示如何将库（例如，现代 C++的 JSON 库`nlohmann-json`）集成到 C++项目中，使用 CMake 进行操作。

克隆 vcpkg 仓库并运行引导脚本：

```cpp
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.sh  # Use bootstrap-vcpkg.bat on Windows
Install nlohmann-json using vcpkg:
./vcpkg install nlohmann-json
```

`vcpkg`将下载并编译库，使其可用于项目。

要在 CMake 项目中使用 vcpkg，您可以在配置项目时将`CMAKE_TOOLCHAIN_FILE`变量设置为`vcpkg.cmake`工具链文件的路径：

```cpp
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=[vcpkg root]/scripts/buildsystems/vcpkg.cmake
Replace [vcpkg root] with the path to your vcpkg installation.
In your CMakeLists.txt, find and link against the nlohmann-json package:
cmake_minimum_required(VERSION 3.0)
project(MyVcpkgProject)
find_package(nlohmann_json CONFIG REQUIRED)
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE nlohmann_json::nlohmann_json)
In your main.cpp, you can now use the nlohmann-json library:
#include <nlohmann/json.hpp>
int main() {
    nlohmann::json j;
    j[“message”] = “Hello, world!”;
    std::cout << j << std::endl;
    return 0;
}
```

`vcpkg`强调基于源的分发，并与 CMake 和 Visual Studio 集成，为希望有效管理库依赖的 C++开发者提供了一个强大的解决方案。它的简单性，加上微软的支持，使其成为优先考虑与构建环境一致性以及与现有微软工具无缝集成的项目的诱人选择。虽然它与 Conan 在简化依赖管理方面有共同目标，但 vcpkg 和 Conan 之间的选择可能取决于具体的项目需求、首选的工作流程和开发生态系统。

## 利用 Docker 进行 C++构建

在 C++中，一个显著的不足是其缺乏管理依赖项的内建机制。因此，引入第三方元素是通过一系列异构的方法实现的：利用 Linux 发行版提供的包管理器（例如，`apt-get`），通过`make install`直接安装，将第三方库作为 Git 子模块包含并随后在项目的源树中编译，或者采用 Conan 或 Vcpkg 等包管理解决方案。

不幸的是，每种方法都有自己的缺点：

+   在开发机器上直接安装依赖通常会影响环境的清洁性，使其与 CI/CD 管道或生产环境不同——这种差异随着第三方组件的每次更新而变得更加明显。

+   确保所有开发者使用的编译器、调试器和其他工具版本的一致性通常是一项艰巨的任务。这种缺乏标准化可能导致一种情况，即构建在个别开发者的机器上成功执行，但在 CI/CD 环境中失败。

+   将第三方库作为 Git 子模块集成并在项目的源目录中编译的做法提出了挑战，尤其是在处理大型库（如 Boost、Protobuf、Thrift 等）时。这种方法可能导致构建过程显著减速，以至于开发者可能会犹豫清除构建目录或在不同分支之间切换。

+   包管理解决方案如 Conan 可能并不总是提供特定依赖项所需版本，包含该版本需要编写额外的 Python 代码，在我看来，这是不必要的负担。

### 一个单独的隔离且可重复的构建环境

解决上述挑战的最佳方案是制定一个 Docker 镜像，其中包含所有必需的依赖项和工具，如编译器和调试器，以方便在从该镜像派生的容器中编译项目。

这个特定的镜像作为构建环境的基石，被开发者在其各自的工作站以及 CI/CD 服务器上统一使用，有效地消除了“在我的机器上运行正常但在 CI 中失败”的常见差异。

由于容器内构建过程的封装特性，它对任何外部变量、工具或配置都保持免疫，这些变量、工具或配置是特定于个别开发者的本地设置，因此使得构建环境**隔离**。

在理想情况下，Docker 镜像会仔细标记有意义的版本标识符，使用户能够通过从注册表中检索适当的镜像来无缝地在不同环境之间切换。此外，如果镜像不再在注册表中可用，值得注意的是，Docker 镜像是由 Dockerfile 构建的，这些 Dockerfile 通常维护在 Git 仓库中。这确保了，如果需要，始终有可能从之前的 Dockerfile 版本重新构建镜像。这种 Docker 化构建框架的特性使其具有**可重复性**。

### 创建构建镜像

我们将开始开发一个简单的应用程序，并在容器内编译它。该应用程序的本质是利用 `boost::filesystem` 显示其大小。选择 Boost 进行此演示是有意为之，旨在展示 Docker 与“重量级”第三方库的集成：

```cpp
#include <boost/filesystem/operations.hpp>
#include <iostream>
int main(int argc, char *argv[]) {
    std::cout << “The path to the binary is: “
              << boost::filesystem::absolute(argv[0])
              << “, the size is:” << boost::filesystem::file_size(argv[0]) << ‘\n’;
    return 0;
}
```

CMake 文件相当简单：

```cpp
cmake_minimum_required(VERSION 3.10.2)
project(a.out)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Remove for compiler-specific features
set(CMAKE_CXX_EXTENSIONS OFF)
string(APPEND CMAKE_CXX_FLAGS “ -Wall”)
string(APPEND CMAKE_CXX_FLAGS “ -Wbuiltin-macro-redefined”)
string(APPEND CMAKE_CXX_FLAGS “ -pedantic”)
string(APPEND CMAKE_CXX_FLAGS “ -Werror”)
# clangd completion
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include_directories(${CMAKE_SOURCE_DIR})
file(GLOB SOURCES “${CMAKE_SOURCE_DIR}/*.cpp”)
add_executable(${PROJECT_NAME} ${SOURCES})
set(Boost_USE_STATIC_LIBS        ON) # only find static libs
set(Boost_USE_MULTITHREADED      ON)
set(Boost_USE_STATIC_RUNTIME    OFF) # do not look for boost libraries linked against static C++ std lib
find_package(Boost REQUIRED COMPONENTS filesystem)
target_link_libraries(${PROJECT_NAME}
    Boost::filesystem
)
```

注意

在这个例子中，Boost 是静态链接的，因为如果目标机器没有预装正确的 Boost 版本，则必须使用它；此建议适用于 Docker 镜像中预装的所有依赖项。

用于此任务的 Dockerfile 非常简单：

```cpp
FROM ubuntu:18.04
LABEL Description=”Build environment”
ENV HOME /root
SHELL [“/bin/bash”, “-c”]
RUN apt-get update && apt-get -y --no-install-recommends install \
    build-essential \
    clang \
    cmake \
    gdb \
    wget
# Let us add some heavy dependency
RUN cd ${HOME} && \
    wget --no-check-certificate --quiet \
        https://boostorg.jfrog.io/artifactory/main/release/1.77.0/source/boost_1_77_0.tar.gz && \
        tar xzf ./boost_1_77_0.tar.gz && \
        cd ./boost_1_77_0 && \
        ./bootstrap.sh && \
        ./b2 install && \
        cd .. && \
        rm -rf ./boost_1_77_0
```

为了确保其名称独特且不与现有的 Dockerfile 冲突，同时清楚地传达其目的，我将其命名为 `DockerfileBuildEnv`：

```cpp
$ docker build -t example/example_build:0.1 -f DockerfileBuildEnv .
Here is supposed to be a long output of boost build
```

*注意，版本号不是“最新”的，而是有一个有意义的名称（例如，0.1）。

一旦镜像成功构建，我们就可以继续进行项目的构建过程。第一步是启动一个基于我们构建的镜像的 Docker 容器，然后在此容器中执行 Bash shell：

```cpp
$ cd project
$ docker run -it --rm --name=example \
 --mount type=bind,source=${PWD},target=/src \
 example/example_build:0.1 \
 bash
```

在这个上下文中，特别重要的参数是 `--mount type=bind,source=$` **{PWD},target=/src**。此指令指示 Docker 将当前目录（包含源代码）绑定挂载到容器内的 `/src` 目录。这种方法避免了将源文件复制到容器中的需要。此外，正如随后将演示的那样，它还允许直接在主机文件系统上存储输出二进制文件，从而消除了重复复制的需要。为了理解剩余的标志和选项，建议查阅官方 Docker 文档。

在容器内，我们将继续编译项目：

```cpp
root@3abec58c9774:/# cd src
root@3abec58c9774:/src# mkdir build && cd build
root@3abec58c9774:/src/build# cmake ..
-- The C compiler identification is GNU 7.5.0
-- The CXX compiler identification is GNU 7.5.0
-- Check for working C compiler: /usr/bin/cc
-- Check for working C compiler: /usr/bin/cc -- works
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Detecting C compile features
-- Detecting C compile features - done
-- Check for working CXX compiler: /usr/bin/c++
-- Check for working CXX compiler: /usr/bin/c++ -- works
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Boost  found.
-- Found Boost components:
   filesystem
-- Configuring done
-- Generating done
-- Build files have been written to: /src/build
root@3abec58c9774:/src/build# make
Scanning dependencies of target a.out
[ 50%] Building CXX object CMakeFiles/a.out.dir/main.cpp.o
[100%] Linking CXX executable a.out
[100%] Built target a.out
```

Et voilà，项目成功构建了！

生成的二进制文件在容器和主机上都能成功运行，因为 Boost 是静态链接的：

```cpp
$ build/a.out
The size of “/home/dima/dockerized_cpp_build_example/build/a.out” is 177320
```

### 使环境可使用

在这个阶段，面对众多 Docker 命令可能会感到不知所措，并 wonder 如何期望记住它们所有。重要的是要强调，开发者不需要为了项目构建目的而记住这些命令的每一个细节。为了简化这个过程，我建议将 Docker 命令封装在一个大多数开发者都熟悉的工具中 – `make`。

为了方便起见，我建立了一个 GitHub 仓库 ([`github.com/f-squirrel/dockerized_cpp`](https://github.com/f-squirrel/dockerized_cpp))，其中包含一个通用的 Makefile。这个 Makefile 设计得易于适应，通常可以用于几乎任何使用 CMake 的项目，而无需进行修改。用户可以选择直接从该仓库下载它，或者将其作为 Git 子模块集成到他们的项目中，以确保访问最新的更新。我提倡后者方法，并将提供更多详细信息。

Makefile 已配置为支持基本命令。用户可以通过在终端中执行 `make help` 来显示可用的命令选项：

```cpp
$ make help
gen_cmake                      Generate cmake files, used internally
build                          Build source. In order to build a specific target run: make TARGET=<target name>.
test                           Run all tests
clean                          Clean build directory
login                          Login to the container. Note: if the container is already running, login into the existing one
build-docker-deps-image        Build the deps image.
```

要将 Makefile 集成到我们的示例项目中，我们首先将其添加为 `build_tools` 目录内的 Git 子模块：

```cpp
git submodule add  https://github.com/f-squirrel/dockerized_cpp.git build_tools/
```

下一步是在仓库的根目录下创建另一个 Makefile，并包含我们刚刚检出的 Makefile：

```cpp
include build_tools/Makefile
```

在项目编译之前，明智的做法是调整某些默认设置以更好地满足项目的特定需求。这可以通过在包含 `build_tools/Makefile` 之前在顶级 Makefile 中声明变量来实现。这种预防性声明允许自定义各种参数，确保构建环境和过程针对项目的需求进行了最佳配置：

```cpp
PROJECT_NAME=example
DOCKER_DEPS_VERSION=0.1
include build_tools/Makefile
By defining the project name, we automatically set the build image name as example/example_build.
```

Make 现在已准备好构建镜像：

```cpp
$ make build-docker-deps-image
docker build  -t example/example_build:latest \
 -f ./DockerfileBuildEnv .
Sending build context to Docker daemon  1.049MB
Step 1/6 : FROM ubuntu:18.04
< long output of docker build >
Build finished. Docker image name: “example/example_build:latest”.
Before you push it to Docker Hub, please tag it(DOCKER_DEPS_VERSION + 1).
If you want the image to be the default, please update the following variables:
/home/dima/dockerized_cpp_build_example/Makefile: DOCKER_DEPS_VERSION
```

默认情况下，Makefile 将最新标签分配给 Docker 镜像。为了更好的版本控制和与我们的项目当前阶段保持一致，建议使用特定版本标记镜像。在此上下文中，我们将镜像标记为 `0.1`。

最后，让我们构建项目：

```cpp
$ make
docker run -it --init --rm --memory-swap=-1 --ulimit core=-1 --name=”example_build” --workdir=/example --mount type=bind,source=/home/dima/dockerized_cpp_build_example,target=/example  example/example_build:0.1 \
 bash -c \
 “mkdir -p /example/build && \
 cd build && \
 CC=clang CXX=clang++ \
 cmake  ..”
-- The C compiler identification is Clang 6.0.0
-- The CXX compiler identification is Clang 6.0.0
-- Check for working C compiler: /usr/bin/clang
-- Check for working C compiler: /usr/bin/clang -- works
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Detecting C compile features
-- Detecting C compile features - done
-- Check for working CXX compiler: /usr/bin/clang++
-- Check for working CXX compiler: /usr/bin/clang++ -- works
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Boost  found.
-- Found Boost components:
   filesystem
-- Configuring done
-- Generating done
-- Build files have been written to: /example/build
CMake finished.
docker run -it --init --rm --memory-swap=-1 --ulimit core=-1 --name=”example_build” --workdir=/example --mount type=bind,source=/home/dima/dockerized_cpp_build_example,target=/example  example/example_build:latest \
 bash -c \
 “cd build && \
 make -j $(nproc) “
Scanning dependencies of target a.out
[ 50%] Building CXX object CMakeFiles/a.out.dir/main.cpp.o
[100%] Linking CXX executable a.out
[100%] Built target a.out
Build finished. The binaries are in /home/dima/dockerized_cpp_build_example/build
```

在检查主机上的构建目录后，你会注意到输出二进制文件已经无缝地放置在那里，便于访问和管理。

Makefile 及其默认值的一个示例项目可以在 GitHub 上找到。这提供了一个实际演示，说明如何将 Makefile 集成到项目中，为寻求在 C++ 项目中实现 Dockerized 构建环境的开发者提供了一个即插即用的解决方案：

+   Makefile 仓库：[`github.com/f-squirrel/dockerized_cpp`](https://github.com/f-squirrel/dockerized_cpp)

+   示例项目：[`github.com/f-squirrel/dockerized_cpp`](https://github.com/f-squirrel/dockerized_cpp)

### Dockerized 构建中的用户管理增强

基于 Docker 的构建系统的初始迭代是在 root 用户的权限下执行操作的。虽然这种设置通常不会立即引起问题——开发者有修改文件权限使用`chmod`的选项——但从安全角度来看，通常不建议以 root 用户身份运行 Docker 容器。更重要的是，如果任何构建目标修改了源代码，例如代码格式化或通过`make`命令应用`clang-tidy`修正，这种方法可能会导致问题。此类修改可能导致源文件归 root 用户所有，从而限制从宿主直接编辑这些文件的能力。

为了解决这个担忧，对基于 Docker 的构建系统的源代码进行了修改，使容器能够通过指定当前用户 ID 和组 ID 以宿主用户身份执行。这种调整现在是标准配置，以提高安全性和可用性。如果需要将容器回滚到以 root 用户身份运行，可以使用以下命令：

```cpp
make DOCKER_USER_ROOT=ON
```

重要的是要认识到，Docker 镜像并不能完全复制宿主用户的环境——没有对应的家目录，用户名和组也不会在容器内复制。这意味着如果构建过程依赖于访问家目录，这种修改后的方法可能不适用。

# 摘要

在本章中，我们探讨了管理 C++项目中第三方依赖项的各种策略和工具，这是影响开发过程效率和可靠性的关键方面。我们深入研究了传统方法，例如利用操作系统包管理器和通过 Git 子模块直接合并依赖项，每种方法都有其独特的优点和局限性。

然后，我们转向了更专业的 C++包管理器，重点介绍了 Conan 和 vcpkg。Conan 凭借其强大的生态系统、通过 Conan Center 提供的广泛库支持以及灵活的配置选项，为管理复杂依赖项、无缝集成到多个构建系统和支持静态和动态链接提供了一个全面的解决方案。它处理多个版本库的能力以及开发者扩展存储库以自定义包的简便性，使其成为现代 C++开发的宝贵工具。

由微软开发的 vcpkg 采用了一种略有不同的方法，侧重于基于源的分发，并确保库使用与消费项目相同的编译器和设置来构建。它与 CMake 和 Visual Studio 的紧密集成，加上微软的支持，确保了流畅的使用体验，尤其是在微软生态系统内的项目。强调从源编译可以解决潜在的二进制不兼容性问题，使 vcpkg 成为管理依赖项的可靠选择。

最后，我们讨论了采用 Docker 化构建作为创建一致、可重复的构建环境的高级策略，这在 Linux 系统中尤其有益。这种方法虽然更复杂，但在隔离性、可扩展性和开发、测试和部署阶段的一致性方面提供了显著的优势。

在本章中，我们的目标是为您提供必要的知识和工具，以便在 C++ 项目的依赖管理领域中导航。通过理解每种方法和工具的优势和局限性，开发者可以做出针对项目特定需求的有信息量的决策，从而实现更高效和可靠的软件开发过程。
