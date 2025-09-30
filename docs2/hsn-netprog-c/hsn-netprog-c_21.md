# 第十七章：在 Linux 上设置您的 C 编译器

**Linux**是 C 编程的一个优秀选择。它在本书涵盖的三个操作系统中最容易设置，并且对 C 编程的支持最好。

使用 Linux 也允许您走道德的高尚之路，并因支持免费软件而感到良好。

在描述 Linux 的设置时有一个问题，那就是有众多不同的 Linux 发行版，它们有不同的软件。在本附录中，我们将提供在`apt`包管理器使用的系统上设置所需的命令，例如**Debian Linux**和**Ubuntu Linux**。如果您使用的是不同的 Linux 发行版，您需要找到与您的系统相关的命令。请参考您发行版的文档以获取帮助。

在深入之前，花点时间确保您的包列表是最新的。这可以通过以下命令完成：

```cpp
sudo apt-get update
```

随着`apt`就绪，设置变得简单。让我们开始吧。

# 安装 GCC

第一步是安装 C 编译器`gcc`。

假设您的系统使用`apt`作为其包管理器，尝试以下命令安装`gcc`并为 C 编程准备您的系统：

```cpp
sudo apt-get install build-essential
```

一旦`install`命令完成，您应该能够运行以下命令来查找已安装的`gcc`版本：

```cpp
gcc --version
```

# 安装 Git

您需要安装 Git 版本控制软件来下载这本书的代码。

假设您的系统使用`apt`包管理器，您可以使用以下命令安装 Git：

```cpp
sudo apt-get install git
```

使用以下命令检查 Git 是否已成功安装：

```cpp
git --version
```

# 安装 OpenSSL

**OpenSSL**可能有些棘手。您可以使用以下命令尝试您发行版的包管理器：

```cpp
sudo apt-get install openssl libssl-dev
```

问题在于您的发行版可能有一个旧的 OpenSSL 版本。如果是这样，您应该直接从[`www.openssl.org/source/`](https://www.openssl.org/source/)获取 OpenSSL 库。当然，在可以使用之前，您需要构建 OpenSSL。构建 OpenSSL 并不容易，但`INSTALL`文件中提供了构建说明。请注意，其构建系统要求您已安装**Perl**。

# 安装 libssh

您可以尝试使用以下命令通过您的包管理器安装`libssh`：

```cpp
sudo apt-get install libssh-dev
```

问题在于这本书中的代码与较旧的`libssh`版本不兼容。因此，我建议您自己构建`libssh`。

您可以从[`www.libssh.org/`](https://www.libssh.org/)获取最新的`libssh`库。如果您擅长安装 C 库，请随意尝试。否则，请继续阅读逐步说明。

在开始之前，请确保您已成功安装了 OpenSSL 库。这些库是`libssh`库所必需的。

我们还需要安装 CMake 来构建`libssh`。您可以从[`cmake.org/`](https://cmake.org/)获取 CMake。您也可以使用以下命令从您的发行版的打包工具中获取它：

```cpp
sudo apt-get install cmake
```

最后，`libssh` 也需要 `zlib` 库。您可以使用以下命令安装 `zlib` 库：

```cpp
sudo apt-get install zlib1g-dev
```

一旦安装了 CMake、`zlib` 库和 OpenSSL 库，请从 [`www.libssh.org/`](https://www.libssh.org/) 网站找到您想要的 `libssh` 版本。撰写本文时，0.8.7 是最新版本。您可以使用以下命令下载并解压 `libssh` 源代码：

```cpp
 wget https://www.libssh.org/files/0.8/libssh-0.8.7.tar.xz
 tar xvf libssh-0.8.7.tar.xz
 cd libssh-0.8.7
```

我建议您查看 `libssh` 包含的安装说明。您可以使用 `less` 命令查看它们。按 *Q* 键退出 `less`：

```cpp
less INSTALL
```

一旦您熟悉了构建说明，您可以使用以下命令尝试构建 `libssh`：

```cpp
mkdir build
cd build
cmake ..
make
```

最后一步是使用以下命令安装库：

```cpp
sudo make install
```
