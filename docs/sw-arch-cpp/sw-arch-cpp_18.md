# 第十四章：容器

从开发到生产的过渡一直是一个痛苦的过程。它涉及大量文档、交接、安装和配置。由于每种编程语言产生的软件行为略有不同，异构应用程序的部署总是困难的。

其中一些问题已经通过容器得到缓解。使用容器，安装和配置大多是标准化的。处理分发的方式有几种，但这个问题也有一些标准可遵循。这使得容器成为那些希望增加开发和运维之间合作的组织的绝佳选择。

本章将涵盖以下主题：

+   构建容器

+   测试和集成容器

+   理解容器编排

# 技术要求

本章列出的示例需要以下内容：

+   Docker 20.10

+   manifest-tool ([`github.com/estesp/manifest-tool`](https://github.com/estesp/manifest-tool))

+   Buildah 1.16

+   Ansible 2.10

+   ansible-bender

+   CMake 3.15

本章中的代码已放在 GitHub 上，网址为[`github.com/PacktPublishing/Software-Architecture-with-Cpp/tree/master/Chapter14`](https://github.com/PacktPublishing/Software-Architecture-with-Cpp/tree/master/Chapter14)。

# 重新介绍容器

最近容器引起了很多关注。有人可能认为它们是一种以前不可用的全新技术。然而，事实并非如此。在 Docker 和 Kubernetes 崛起之前，这两者目前在行业中占主导地位，已经有了诸如 LXC 之类的解决方案，它们提供了许多类似的功能。

我们可以追溯到自 1979 年 UNIX 系统中可用的 chroot 机制，将一个执行环境与另一个分离的起源。类似的概念也在 FreeBSD jails 和 Solaris Zones 中使用过。

容器的主要任务是将一个执行环境与另一个隔离开。这个隔离的环境可以有自己的配置、不同的应用程序，甚至不同的用户帐户，与主机环境不同。

尽管容器与主机隔离，它们通常共享相同的操作系统内核。这是与虚拟化环境的主要区别。虚拟机有专用的虚拟资源，这意味着它们在硬件级别上是分离的。容器在进程级别上是分离的，这意味着运行它们的开销更小。

容器的一个强大优势是能够打包和运行另一个已经针对运行您的应用程序进行了优化和配置的操作系统。没有容器，构建和部署过程通常包括几个步骤：

1.  应用已构建。

1.  提供示例配置文件。

1.  准备安装脚本和相关文档。

1.  应用程序已打包为目标操作系统（如 Debian 或 Red Hat）。

1.  软件包部署到目标平台。

1.  安装脚本为应用程序运行准备了基础。

1.  配置必须进行调整以适应现有系统。

当您切换到容器时，就不再需要强大的安装脚本。应用程序只会针对一个众所周知的操作系统进行目标设置——容器中存在的操作系统。配置也是一样：应用程序预先配置为目标操作系统并与其一起分发，而不是准备许多可配置选项。部署过程只包括解压容器镜像并在其中运行应用程序进程。

虽然容器和微服务经常被认为是同一件事，但它们并不是。此外，容器可能意味着应用容器或操作系统容器，只有应用容器与微服务配合得很好。接下来的章节将告诉您原因。我们将描述您可能遇到的不同容器类型，向您展示它们与微服务的关系，并解释何时最好使用它们（以及何时避免使用它们）。

## 探索容器类型

到目前为止描述的容器中，操作系统容器与由 Docker、Kubernetes 和 LXD 领导的当前容器趋势有根本的不同。应用容器专注于在容器内运行单个进程-即应用程序，而不是专注于重新创建具有诸如 syslog 和 cron 等服务的整个操作系统。

专有解决方案替换了所有通常的操作系统级服务。这些解决方案提供了一种统一的方式来管理容器内的应用程序。例如，不使用 syslog 来处理日志，而是将 PID 1 的进程的标准输出视为应用程序日志。不使用`init.d`或 systemd 等机制，而是由运行时应用程序处理应用容器的生命周期。

由于 Docker 目前是应用容器的主要解决方案，我们将在本书中大多数情况下使用它作为示例。为了使画面完整，我们将提出可行的替代方案，因为它们可能更适合您的需求。由于项目和规范是开源的，这些替代方案与 Docker 兼容，并且可以用作替代品。

在本章的后面，我们将解释如何使用 Docker 来构建、部署、运行和管理应用容器。

## 微服务的兴起

Docker 的成功与微服务的采用增长同时出现并不奇怪，因为微服务和应用容器自然地结合在一起。

没有应用容器，没有一种简单而统一的方式来打包、部署和维护微服务。尽管一些公司开发了一些解决这些问题的解决方案，但没有一种解决方案足够流行，可以成为行业标准。

没有微服务，应用容器的功能相当有限。软件架构专注于构建专门为给定的服务集合明确配置的整个系统。用另一个服务替换一个服务需要改变架构。

应用容器提供了一种标准的分发微服务的方式。每个微服务都带有其自己的嵌入式配置，因此诸如自动扩展或自愈等操作不再需要了解底层应用程序。

您仍然可以在没有应用容器的情况下使用微服务，也可以在应用容器中托管微服务。例如，尽管 PostgreSQL 数据库和 Nginx Web 服务器都不是设计为微服务，但它们通常在应用容器中使用。

## 选择何时使用容器

容器方法有几个好处。操作系统容器和应用容器在其优势所在的一些不同用例中也有所不同。

### 容器的好处

与隔离环境的另一种流行方式虚拟机相比，容器在运行时需要更少的开销。与虚拟机不同，不需要运行一个单独的操作系统内核版本，并使用硬件或软件虚拟化技术。应用容器也不运行通常在虚拟机中找到的其他操作系统服务，如 syslog、cron 或 init。此外，应用容器提供更小的镜像，因为它们通常不必携带整个操作系统副本。在极端情况下，应用容器可以由单个静态链接的二进制文件组成。

此时，你可能会想，如果里面只有一个单一的二进制文件，为什么还要费心使用容器呢？拥有统一和标准化的构建和运行容器的方式有一个特定的好处。由于容器必须遵循特定的约定，因此比起常规的二进制文件，对它们进行编排更容易，后者可能对日志记录、配置、打开端口等有不同的期望。

另一件事是，容器提供了内置的隔离手段。每个容器都有自己的进程命名空间和用户帐户命名空间，等等。这意味着一个容器中的进程（或进程）对主机上的进程或其他容器中的进程没有概念。沙盒化甚至可以进一步进行，因为你可以为你的容器分配内存和 CPU 配额，使用相同的标准用户界面（无论是 Docker、Kubernetes 还是其他什么）。

标准化的运行时也意味着更高的可移植性。一旦容器构建完成，通常可以在不同的操作系统上运行，而无需修改。这也意味着在运行的东西与开发中运行的东西非常接近或相同。问题的再现更加轻松，调试也更加轻松。

### 容器的缺点

由于现在有很大的压力要将工作负载迁移到容器中，作为架构师，你需要了解与这种迁移相关的所有风险。利益无处不在，你可能已经理解了它们。

容器采用的主要障碍是，并非所有应用程序都能轻松迁移到容器中。特别是那些以微服务为设计目标的应用程序容器。如果你的应用程序不是基于微服务架构的，将其放入容器中可能会带来更多问题。

如果你的应用程序已经很好地扩展，使用基于 TCP/IP 的 IPC，并且大部分是无状态的，那么转移到容器应该不会有挑战。否则，这些方面中的每一个都将带来挑战，并促使重新思考现有的设计。

与容器相关的另一个问题是持久存储。理想情况下，容器不应该有自己的持久存储。这样可以利用快速启动、轻松扩展和灵活的调度。问题在于提供业务价值的应用程序不能没有持久存储���

这个缺点通常可以通过使大多数容器无状态，并依赖于一个外部的非容器化组件来存储数据和状态来减轻。这样的外部组件可以是传统的自托管数据库，也可以是来自云提供商的托管数据库。无论选择哪个方向，都需要重新考虑架构并相应地进行修改。

由于应用程序容器遵循特定的约定，应用程序必须修改以遵循这些约定。对于一些应用程序来说，这将是一个低成本的任务。对于其他一些应用程序，比如使用内存 IPC 的多进程组件，这将是复杂的。

经常被忽略的一点是，只要容器内的应用程序是本地 Linux 应用程序，应用程序容器就能很好地工作。虽然支持 Windows 容器，但它们既不方便也不像它们的 Linux 对应物那样受支持。它们还需要运行作为主机的经过许可的 Windows 机器。

如果你从头开始构建一个新的应用程序，并且可以基于这项技术设计，那么很容易享受应用程序容器的好处。将现有的应用程序移植到应用程序容器中，特别是如果它很复杂，将需要更多的工作，可能还需要对整个架构进行改造。在这种情况下，我们建议您特别仔细地考虑所有的利弊。做出错误的决定可能会损害产品的交付时间、可用性和预算。

# 构建容器

应用程序容器是本节的重点。虽然操作系统容器大多遵循系统编程原则，但应用程序容器带来了新的挑战和模式。它们还提供了专门的构建工具来处理这些挑战。我们将考虑的主要工具是 Docker，因为它是当前构建和运行应用程序容器的事实标准。我们还将介绍一些构建应用程序容器的替代方法。

除非另有说明，从现在开始，当我们使用“容器”这个词时，它指的是“应用程序容器”。

在这一部分，我们将专注于使用 Docker 构建和部署容器的不同方法。

## 解释容器镜像

在我们描述容器镜像及如何构建它们之前，了解容器和容器镜像之间的区别至关重要。这两个术语经常会引起混淆，尤其是在非正式的对话中。

容器和容器镜像之间的区别与运行中的进程和可执行文件之间的区别相同。

**容器镜像是静态的**：它们是特定文件系统的快照和相关的元数据。元数据描述了在运行时设置了哪些环境变量，或者在创建容器时运行哪个程序，等等。

**容器是动态的**：它们运行在容器镜像内的一个进程。我们可以从容器镜像创建容器，也可以通过对运行中的容器进行快照来创建容器镜像。事实上，容器镜像构建过程包括创建多个容器，执行其中的命令，并在命令完成后对它们进行快照。

为了区分容器镜像引入的数据和运行时生成的数据，Docker 使用联合挂载文件系统来创建不同的文件系统层。这些层也存在于容器镜像中。通常，容器镜像的每个构建步骤对应于结果容器镜像中的一个新层。

## 使用 Dockerfiles 构建应用程序

使用 Docker 构建应用程序容器镜像的最常见方法是使用 Dockerfile。Dockerfile 是一种描述生成结果镜像所需操作的命令式语言。一些操作会创建新的文件系统层，而其他操作则会操作元数据。

我们不会详细介绍和具体涉及 Dockerfiles。相反，我们将展示不同的方法来将 C++应用程序容器化。为此，我们需要介绍一些与 Dockerfiles 相关的语法和概念。

这是一个非常简单的 Dockerfile 的示例：

```cpp
FROM ubuntu:bionic

RUN apt-get update && apt-get -y install build-essentials gcc

CMD /usr/bin/gcc
```

通常，我们可以将 Dockerfile 分为三个部分：

+   导入基本镜像（`FROM`指令）

+   在容器内执行操作，将导致容器镜像（`RUN`指令）

+   运行时使用的元数据（`CMD`命令）

后两部分可能会交错进行，每个部分可能包含一个或多个指令。也可以省略任何后续部分，因为只有基本镜像是必需的。这并不意味着你不能从空文件系统开始。有一个名为`scratch`的特殊基本镜像就是为了这个目的。在否则空的文件系统中添加一个单独的静态链接二进制文件可能看起来像下面这样：

```cpp
FROM scratch

COPY customer /bin/customer

CMD /bin/customer
```

在第一个 Dockerfile 中，我们采取的步骤如下：

1.  导入基本的 Ubuntu Bionic 镜像。

1.  在容器内运行命令。命令的结果将在目标镜像内创建一个新的文件系统层。这意味着使用`apt-get`安装的软件包将在所有基于此镜像的容器中可用。

1.  设置运行时元数据。在基于此镜像创建容器时，我们希望将`GCC`作为默认进程运行。

要从 Dockerfile 构建镜像，您将使用`docker build`命令。它需要一个必需的参数，即包含构建上下文的目录，这意味着 Dockerfile 本身和您想要复制到容器内的其他文件。要从当前目录构建 Dockerfile，请使用`docker build`。

这将构建一个匿名镜像，这并不是很有用��大多数情况下，您希望使用命名的镜像。在命名容器镜像时有一个惯例要遵循，我们将在下一节中介绍。

## 命名和分发镜像

Docker 中的每个容器镜像都有一个独特的名称，由三个元素组成：注册表的名称，镜像的名称，一个标签。容器注册表是保存容器镜像的对象仓库。Docker 的默认容器注册表是`docker.io`。当从这个注册表中拉取镜像时，我们可以省略注册表的名称。

我们之前的例子中，`ubuntu:bionic`的完整名称是`docker.io/ubuntu:bionic`。在这个例子中，`ubuntu`是镜像的名称，而`bionic`是代表镜像特定版本的标签。

在基于容器的应用程序构建时，您将有兴趣存储所有的注册表镜像。可以搭建自己的私有注册表并在那里保存镜像，或者使用托管解决方案。流行的托管解决方案包括以下内容：

+   Docker Hub

+   quay.io

+   GitHub

+   云提供商（如 AWS、GCP 或 Azure）

Docker Hub 仍然是最受欢迎的，尽管一些公共镜像正在迁移到 quay.io。两者都是通用的，允许存储公共和私有镜像。如果您已经在使用特定平台并希望将镜像保持接近 CI 流水线或部署目标，GitHub 或云提供商对您来说可能更具吸引力。如果您希望减少使用的个别服务数量，这也是有帮助的。

如果以上解决方案都不适合您，那么搭建自己的本地注册表也非常简单，只需要运行一个容器。

要构建一个命名的镜像，您需要向`docker build`命令传递`-t`参数。例如，要构建一个名为`dominicanfair/merchant:v2.0.3`的镜像，您将使用`docker build -t dominicanfair/merchant:v2.0.3 .`。

## 已编译的应用程序和容器

对于解释性语言（如 Python 或 JavaScript）的应用程序构建容器镜像，方法基本上是相同的：

1.  安装依赖项。

1.  将源文件复制到容器镜像中。

1.  复制必要的配置。

1.  设置运行时命令。

然而，对于已编译的应用程序，还有一个额外的步骤是首先编译应用程序。有几种可能的方法来实现这一步骤，每种方法都有其优缺点。

最明显的方法是首先安装所有的依赖项，复制源文件，然后编译应用程序作为容器构建步骤之一。主要的好处是我们可以准确控制工具链的内容和配置，因此有一种便携的方式来构建应用程序。然而，缺点是太大而无法忽视：生成的容器镜像包含了许多不必要的文件。毕竟，在运行时我们既不需要源代码也不需要工具链。由于叠加文件系统的工作方式，无法在引入到先前层中的文件之后删除这些文件。而且，如果攻击者设法侵入容器，容器中的源代码可能会构成安全风险。

它可以看起来像这样：

```cpp
FROM ubuntu:bionic

RUN apt-get update && apt-get -y install build-essentials gcc cmake

ADD . /usr/src

WORKDIR /usr/src

RUN mkdir build && \
    cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    cmake --build . && \
    cmake --install .

CMD /usr/local/bin/customer
```

另一种明显的方法，也是我们之前讨论过的方法，是在主机上构建应用程序，然后只将生成的二进制文件复制到容器映像中。当已经建立了一个构建过程时，这需要对当前构建过程进行较少的更改。主要的缺点是您必须在构建机器上与容器中使用相同的库集。例如，如果您的主机操作系统是 Ubuntu 20.04，那么您的容器也必须基于 Ubuntu 20.04。否则，您会面临不兼容性的风险。使用这种方法，还需要独立配置工具链而不是容器。

就像这样：

```cpp
FROM scratch

COPY customer /bin/customer

CMD /bin/customer
```

一种稍微复杂的方法是采用多阶段构建。使用多阶段构建，一个阶段可能专门用于设置工具链和编译项目，而另一个阶段则将生成的二进制文件复制到目标容器映像中。这比以前的解决方案有几个好处。首先，Dockerfile 现在控制工具链和运行时环境，因此构建的每一步都有详细记录。其次，可以使用带有工具链的映像来确保开发和持续集成/持续部署（CI/CD）流水线之间的兼容性。这种方式还使得更容易分发工具链本身的升级和修复。主要的缺点是容器化的工具链可能不像本机工具链那样方便使用。此外，构建工具并不特别适合应用容器，后者要求每个容器只运行一个进程。这可能导致一些进程崩溃或被强制停止时出现意外行为。

前面示例的多阶段版本如下所示：

```cpp
FROM ubuntu:bionic AS builder

RUN apt-get update && apt-get -y install build-essentials gcc cmake

ADD . /usr/src

WORKDIR /usr/src

RUN mkdir build && \
    cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    cmake --build .

FROM ubuntu:bionic

COPY --from=builder /usr/src/build/bin/customer /bin/customer

CMD /bin/customer
```

从第一个 `FROM` 命令开始的第一个阶段设置了构建器，添加了源代码并构建了二进制文件。然后，从第二个 `FROM` 命令开始的第二阶段，复制了上一阶段的结果二进制文件，而没有复制工具链或源代码。

## 通过清单定位多个架构

使用 Docker 的应用容器通常在 x86_64（也称为 AMD64）机器上使用。如果您只针对这个平台，那就没什么好担心的。但是，如果您正在开发物联网、嵌入式或边缘应用程序，您可能对多架构映像感兴趣。

由于 Docker 可用于许多不同的 CPU 架构，有多种方法可以处理多平台上的映像管理。

处理为不同目标构建的映像的一种方法是使用映像标签来描述特定平台。例如，我们可以使用 `merchant:v2.0.3-aarch64` 而不是 `merchant:v2.0.3`。尽管这种方法可能看起来最容易实现，但实际上有点问题。

不仅需要更改构建过程以在标记过程中包含架构。在拉取映像以运行它们时，还必须手动在所有地方添加预期的后缀。如果使用编排器，将无法以直接的方式在��同平台之间共享清单，因为标签将是特定于平台的。

一种更好的方法，不需要修改部署步骤，是使用 `manifest-tool`（https://github.com/estesp/manifest-tool）。首先，构建过程看起来与之前建议的类似。映像在所有支持的架构上分别构建，并带有标签中的平台后缀推送到注册表。在所有映像都推送后，`manifest-tool` 合并映像以提供单个多架构映像。这样，每个支持的平台都能使用完全相同的标签。

这里提供了 `manifest-tool` 的示例配置：

```cpp
image: hosacpp/merchant:v2.0.3
manifests:
  - image: hosacpp/merchant:v2.0.3-amd64
    platform:
      architecture: amd64
      os: linux
  - image: hosacpp/merchant:v2.0.3-arm32
    platform:
      architecture: arm
      os: linux
  - image: hosacpp/merchant:v2.0.3-arm64
    platform:
      architecture: arm64
      os: linux
```

在这里，我们有三个支持的平台，每个平台都有其相应的后缀（`hosacpp/merchant:v2.0.3-amd64`，`hosacpp/merchant:v2.0.3-arm32`和`hosacpp/merchant:v2.0.3-arm64`）。`Manifest-tool`将为每个平台构建的镜像合并，并生成一个`hosacpp/merchant:v2.0.3`镜像，我们可以在任何地方使用。

另一种可能性是使用 Docker 内置的名为 Buildx 的功能。使用 Buildx，你可以附加多个构建器实例，每个实例针对所需的架构。有趣的是，你不需要本机机器来运行构建；你还可以在多阶段构建中使用 QEMU 模拟或交叉编译。尽管它比之前的方法更强大，但 Buildx 也相当复杂。在撰写本文时，它需要 Docker 实验模式和 Linux 内核 4.8 或更高版本。你需要设置和管理构建器，并且并非所有功能都以直观的方式运行。它可能会在不久的将来改进并变得更加稳定。

准备构建环境并构建多平台镜像的示例代码可能如下所示：

```cpp
# create two build contexts running on different machines
docker context create \
    --docker host=ssh://docker-user@host1.domifair.org \
    --description="Remote engine amd64" \
    node-amd64
docker context create \
    --docker host=ssh://docker-user@host2.domifair.org \
    --description="Remote engine arm64" \
    node-arm64

# use the contexts
docker buildx create --use --name mybuild node-amd64
docker buildx create --append --name mybuild node-arm64

# build an image
docker buildx build --platform linux/amd64,linux/arm64 .
```

正如你所看到的，如果你习惯于常规的`docker build`命令，这可能会有点令人困惑。

## 构建应用程序容器的替代方法

使用 Docker 构建容器镜像需要 Docker 守护程序运行。Docker 守护程序需要 root 权限，在某些设置中可能会带来安全问题。即使进行构建的 Docker 客户端可能由非特权用户运行，但在构建环境中安装 Docker 守护程序并非总是可行。

### Buildah

Buildah 是一个替代工具，可以配置为在没有 root 访问权限的情况下运行。Buildah 可以使用常规的 Dockerfile，我们之前讨论过。它还提供了自己的命令行界面，你可以在 shell 脚本或其他更直观的自动化中使用。将之前的 Dockerfile 重写为使用 buildah 接口的 shell 脚本之一将如下所示：

```cpp
#!/bin/sh

ctr=$(buildah from ubuntu:bionic)

buildah run $ctr -- /bin/sh -c 'apt-get update && apt-get install -y build-essential gcc'

buildah config --cmd '/usr/bin/gcc' "$ctr"

buildah commit "$ctr" hosacpp-gcc

buildah rm "$ctr"
```

Buildah 的一个有趣特性是它允许你将容器镜像文件系统挂载到主机文件系统中。这样，你可以使用主机的命令与镜像的内容进行交互。如果你有一些不想（或者由于许可限制而无法）放入容器中的软件，使用 Buildah 时仍然可以在容器外部调用它。

### Ansible-bender

Ansible-bender 使用 Ansible playbooks 和 Buildah 来构建容器镜像。所有配置，包括基本镜像和元数据，都作为 playbook 中的变量传递。以下是我们之前的示例转换为 Ansible 语法的示例：

```cpp
---
- name: Container image with ansible-bender
  hosts: all
  vars:
    ansible_bender:
      base_image: python:3-buster

      target_image:
        name: hosacpp-gcc
        cmd: /usr/bin/gcc
  tasks:
  - name: Install Apt packages
    apt:
      pkg:
        - build-essential
        - gcc
```

正如你所看到的，`ansible_bender`变量负责所有与容器特定配置相关的内容。下面呈现的任务在基于`base_image`的容器内执行。

需要注意的一点是，Ansible 需要基本镜像中存在 Python 解释器。这就是为什么我们不得不将在之前的示例中使用的`ubuntu:bionic`更改为`python:3-buster`。`ubuntu:bionic`是一个没有预安装 Python 解释器的 Ubuntu 镜像。

### 其他

还有其他构建容器镜像的方法。你可以使用 Nix 创建文件系统镜像，然后使用 Dockerfile 的`COPY`指令将其放入镜像中，例如。更进一步，你可以通过任何其他方式准备文件系统镜像，然后使用`docker import`将其导入为基本容器镜像。

选择符合你特定需求的解决方案。请记住，使用`docker build`使用 Dockerfile 进行构建是最流行的方法，因此它是最有文档支持的。使用 Buildah 更加灵活，可以更好地将创建容器镜像融入到构建过程中。最后，如果你已经在 Ansible 中投入了大量精力，并且想要重用已有的模块，`ansible-bender`可能是一个不错的解决方案。

## 将容器与 CMake 集成

在这一部分，我们将演示如何通过使用 CMake 来创建 Docker 镜像。

### 使用 CMake 配置 Dockerfile

首先，我们需要一个 Dockerfile。让我们使用另一个 CMake 输入文件来实现这一点：

```cpp
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Dockerfile.in
                ${PROJECT_BINARY_DIR}/Dockerfile @ONLY)
```

请注意，我们使用`PROJECT_BINARY_DIR`来避免覆盖源树中其他项目创建的 Dockerfile，如果我们的项目是更大项目的一部分。

我们的`Dockerfile.in`文件将如下所示：

```cpp
FROM ubuntu:latest
ADD Customer-@PROJECT_VERSION@-Linux.deb .
RUN apt-get update && \
    apt-get -y --no-install-recommends install ./Customer-@PROJECT_VERSION@-Linux.deb && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -r /var/lib/apt/lists/* Customer-@PROJECT_VERSION@-Linux.deb
ENTRYPOINT ["/usr/bin/customer"]
EXPOSE 8080
```

首先，我们指定我们将使用最新的 Ubuntu 镜像，在其中安装我们的 DEB 包及其依赖项，然后进行整理。在安装软件包的同时更新软件包管理器缓存是很重要的，以避免由于 Docker 层的工作方式而导致的旧缓存问题。清理也作为相同的`RUN`命令的一部分进行（在同一层），以使层大小更小。安装软件包后，我们让我们的镜像在启动时运行`customer`微服务。最后，我们告诉 Docker 暴露它将监听的端口。

现在，回到我们的`CMakeLists.txt`文件。

### 将容器与 CMake 集成

对于基于 CMake 的项目，可以包含一个负责构建容器的构建步骤。为此，我们需要告诉 CMake 找到 Docker 可执行文件，并在找不到时退出。我们可以使用以下方法来实现：

```cpp
find_program(Docker_EXECUTABLE docker)
 if(NOT Docker_EXECUTABLE)
   message(FATAL_ERROR "Docker not found")
 endif()
```

让我们重新访问第七章中的一个示例，*构建和打包*。在那里，我们为客户应用程序构建了一个二进制文件和一个 Conan 软件包。现在，我们希望将这个应用程序打包为一个 Debian 存档，并构建一个预安装软件包的 Debian 容器镜像，用于客户应用程序。

为了创建我们的 DEB 软件包，我们需要一个辅助目标。让我们使用 CMake 的`add_custom_target`功能来实现这一点：

```cpp
add_custom_target(
   customer-deb
   COMMENT "Creating Customer DEB package"
   COMMAND ${CMAKE_CPACK_COMMAND} -G DEB
   WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
   VERBATIM)
 add_dependencies(customer-deb libcustomer)
```

我们的目标调用 CPack 来创建我们感兴趣的一个软件包，并省略其余的软件包。我们希望软件包在与 Dockerfile 相同的目录中创建，以方便起见。推荐使用`VERBATIM`关键字，因为使用它，CMake 将转义有问题的字符。如果未指定，您的脚本的行为可能会因不同平台而异。

`add_dependencies`调用将确保在 CMake 构建`customer-deb`目标之前，`libcustomer`已经构建。现在我们有了辅助目标，让我们在创建容器镜像时使用它：

```cpp
add_custom_target(
   docker
   COMMENT "Preparing Docker image"
   COMMAND ${Docker_EXECUTABLE} build ${PROJECT_BINARY_DIR}
           -t dominicanfair/customer:${PROJECT_VERSION} -t dominicanfair/customer:latest
   VERBATIM)
 add_dependencies(docker customer-deb)
```

如您所见，我们调用了我们之前在包含我们的 Dockerfile 和 DEB 软件包的目录中找到的 Docker 可执行文件，以创建一个镜像。我们还告诉 Docker 将我们的镜像标记为最新版本和我们项目的版本。最后，我们确保在调用我们的 Docker 目标时将构建 DEB 软件包。

如果您选择的生成器是`make`，那么构建镜像就像`make docker`一样简单。如果您更喜欢完整的 CMake 命令（例如，为了创建与生成器无关的脚本），那么调用是`cmake --build . --target docker`。

# 测试和集成容器

容器非常适合 CI/CD 流水线。由于它们大多数情况下除了容器运行时本身不需要其他依赖项，因此它们可以很容易地进行测试。工作机器不必被配置以满足测试需求，因此添加更多节点更容易。而且，它们���是通用的，因此它们可以充当构建者、测试运行者，甚至是部署执行者，而无需任何先前的配置。

在**CI**/**CD**中使用容器的另一个巨大好处是它们彼此隔离。这意味着在同一台机器上运行的多个副本不应该相互干扰。这是真的，除非测试需要一些来自主机操作系统的资源，例如端口转发或卷挂载。因此最好设计测试，使这些资源不是必需的（或者至少它们不会发生冲突）。端口随机化是一种有用的技术，可以避免冲突，例如。

## 容器内的运行时库

容器的选择可能会影响工具链的选择，因此也会影响应用程序可用的 C++语言特性。由于容器通常基于 Linux，可用的系统编译器通常是带有 glibc 标准库的 GNU GCC。然而，一些流行的用于容器的 Linux 发行版，如 Alpine Linux，基于不同的标准库 musl。

如果你的目标是这样的发行版，确保你将要使用的代码，无论是内部开发的还是来自第三方提供者，都与 musl 兼容。musl 和 Alpine Linux 的主要优势是它们可以生成更小的容器镜像。例如，为 Debian Buster 构建的 Python 镜像约为 330MB，精简版的 Debian 版本约为 40MB，而 Alpine 版本仅约为 16MB。更小的镜像意味着更少的带宽浪费（用于上传和下载）和更快的更新。

Alpine 可能也会引入一些不需要的特性，比如更长的构建时间、隐晦的错误或性能降低。如果你想使用它来减小大小，务必进行适当的测试，确保应用程序没有问题。

为了进一步减小镜像的大小，你可以考虑放弃底层操作系统。这里所说的操作系统是指通常存在于容器中的所有用户空间工具，如 shell、包管理器和共享库。毕竟，如果你的应用是唯一要运行的东西，其他一切都是不必要的。

Go 或 Rust 应用程序通常提供一个自包含的静态构建，可以形成一个容器镜像。虽然在 C++中可能不那么直接，但也值得考虑。

减小镜像大小也有一些缺点。首先，如果你决定使用 Alpine Linux，请记住它不像 Ubuntu、Debian 或 CentOS 那样受欢迎。尽管它经常是容器开发者的首选平台，但对于其他用途来说非常不寻常。

这意味着可能会出现新的兼容性问题，主要源自它不是基于事实上的标准 glibc 实现。如果你依赖第三方组件，提供者可能不会为这个平台提供支持。

如果你决定采用容器镜像中的单个静态链接二进制文件路线，也有一些挑战需要考虑。首先，你不建议静态链接 glibc，因为它内部使用 dlopen 来处理**Name Service Switch**（NSS）和 iconv。如果你的软件依赖于 DNS 解析或字符集转换，你仍然需要提供 glibc 和相关库的副本。

另一个需要考虑的问题是，通常会使用 shell 和包管理器来调试行为异常的容器。当你的某个容器表现出奇怪的行为时，你可以在容器内启动另一个进程，并通过使用诸如`ps`、`ls`或`cat`等标准 UNIX 工具来弄清楚容器内部发生了什么。要在容器内运行这样的应用程序，它必须首先存在于容器镜像中。一些解决方法允许操作员在运行的容器内注入调试二进制文件，但目前没有一个得到很好的支持。

## 替代容器运行时

Docker 是构建和运行容器的最流行方式，但由于容器标准是开放的，也有其他可供选择的运行时。用于替代 Docker 并提供类似用户体验的主要工具是 Podman。与前一节中描述的 Buildah 一起，它们是旨在完全取代 Docker 的工具。

它们的另一个好处是*不需要在主机上运行额外的守护程序，就像 Docker 一样*。它们两者也都支持（尽管尚不成熟）无根操作，这使它们更适合安全关键操作。Podman 接受您期望 Docker CLI 执行的所有命令，因此您可以简单地将其用作别名。

另一种旨在提供更好安全性的容器方法是**Kata Containers**倡议。Kata Containers 使用轻量级虚拟机来利用硬件虚拟化，以在容器和主机操作系统之间提供额外的隔离级别。

Cri-O 和 containerd 也是 Kubernetes 使用的流行运行时。

# 理解容器编排

一些容器的好处只有在使用容器编排器来管理它们时才会显现出来。编排器会跟踪将运行您的工作负载的所有节点，并监视这些节点上分布的容器的健康和状态。

例如，高可用性等更高级的功能需要正确设置编排器，通常意味着至少要为控制平面专门分配三台机器，另外还需要为工作节点分配三台机器。节点的自动缩放，以及容器的自动缩放，还需要编排器具有能够控制底层基础设施的驱动程序（例如，通过使用云提供商的 API）。

在这里，我们将介绍一些最受欢迎的编排器，您可以选择其中一个作为系统的基础。您将在下一章[Kubernetes](https://cdp.packtpub.com/hands_on_software_architecture_with_c__/wp-admin/post.php?post=41&action=edit)中找到更多关于 Kubernetes 的实用信息，*云原生设计*。在这里，我们给您一个可能的选择概述。

所提供的编排器操作类似的对象（服务、容器、批处理作业），尽管每个对象的行为可能不同。可用的功能和操作原则在它们之间也有所不同。它们的共同之处在于，通常您会编写一个配置文件，以声明方式描述所需的资源，然后使用专用的 CLI 工具应用此配置。为了说明工具之间的差异，我们提供了一个示例配置，指定了之前介绍的一个 Web 应用程序（商家服务）和一个流行的 Web 服务器 Nginx 作为代理。

## 自托管解决方案

无论您是在本地运行应用程序，还是在私有云或公共云中运行，您可能希望对所选择的编排器有严格的控制。以下是这个领域中的一些自托管解决方案。请记住，它们中的大多数也可以作为托管服务提供。但是，选择自托管可以帮助您防止供应商锁定，这可能对您的组织是可取的。

### Kubernetes

Kubernetes 可能是我们在这里提到的所有编排器中最为人所知的。它很普遍，这意味着如果您决定实施它，将会有很多文档和社区支持。

尽管 Kubernetes 使用与 Docker 相同的应用程序容器格式，但基本上这就是所有相似之处的结束。不可能使用标准的 Docker 工具直接与 Kubernetes 集群和资源进行交互。在使用 Kubernetes 时，需要学习一套新的工具和概念。

与 Docker 不同，容器是您将操作的主要对象，而在 Kubernetes 中，运行时的最小单元称为 Pod。Pod 可能由一个或多个共享挂载点和网络资源的容器组成。Pod 本身很少引起兴趣，因为 Kubernetes 还具有更高级的概念，如复制控制器、部署控制器或守护进程集。它们的作用是跟踪 Pod 并确保节点上运行所需数量的副本。

Kubernetes 中的网络模型也与 Docker 非常不同。在 Docker 中，您可以将容器的端口转发，使其可以从不同的机器访问。在 Kubernetes 中，如果要访问一个 pod，通常会创建一个 Service 资源，它可以作为负载均衡器来处理指向服务后端的流量。服务可以用于 pod 之间的通信，也可以暴露给互联网。在内部，Kubernetes 资源使用 DNS 名称执行服务发现。

Kubernetes 是声明性的，最终一致的。这意味着您不必直接创建和分配资源，只需提供所需最终状态的描述，Kubernetes 将完成将集群带到所需状态所需的工作。资源通常使用 YAML 描述。

由于 Kubernetes 具有高度的可扩展性，因此在**Cloud Native Computing Foundation**（**CNCF**）下开发了许多相关项目，将 Kubernetes 转变为一个与提供商无关的云开发平台。我们将在下一章第十五章中更详细地介绍 Kubernetes，*云原生设计*。

以下是使用 YAML（`merchant.yaml`）在 Kubernetes 中的资源定义方式：

```cpp
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    app: dominican-front
  name: dominican-front
spec:
  selector:
    matchLabels:
      app: dominican-front
  template:
    metadata:
      labels:
        app: dominican-front
    spec:
      containers:
        - name: webserver
          imagePullPolicy: Always
          image: nginx
          ports:
            - name: http
              containerPort: 80
              protocol: TCP
      restartPolicy: Always
---
apiVersion: v1
kind: Service
metadata:
  labels:
    app: dominican-front
  name: dominican-front
spec:
  ports:
    - port: 80
      protocol: TCP
      targetPort: 80
  selector:
    app: dominican-front
  type: ClusterIP
---
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    app: dominican-merchant
  name: merchant
spec:
  selector:
    matchLabels:
      app: dominican-merchant
  replicas: 3
  template:
    metadata:
      labels:
        app: dominican-merchant
    spec:
      containers:
        - name: merchant
          imagePullPolicy: Always
          image: hosacpp/merchant:v2.0.3
          ports:
            - name: http
              containerPort: 8000
              protocol: TCP
      restartPolicy: Always
---
apiVersion: v1
kind: Service
metadata:
  labels:
    app: dominican-merchant
  name: merchant
spec:
  ports:
    - port: 80
      protocol: TCP
      targetPort: 8000
  selector:
    app: dominican-merchant
    type: ClusterIP
```

要应用此配置并编排容器，请使用`kubectl apply -f merchant.yaml`。

### Docker Swarm

Docker 引擎，也需要构建和运行 Docker 容器，预装有自己的编排器。这个编排器是 Docker Swarm，其主要特点是通过使用 Docker API 与现有的 Docker 工具高度兼容。

Docker Swarm 使用服务的概念来管理健康检查和自动扩展。它原生支持服务的滚动升级。服务能够发布它们的端口，然后由 Swarm 的负载均衡器提供服务。它支持将配置存储为对象以进行运行时自定义，并内置了基本的秘密管理。

Docker Swarm 比 Kubernetes 简单得多，可扩展性较差。如果您不想了解 Kubernetes 的所有细节，这可能是一个优势。然而，主要的缺点是缺乏流行度，这意味着更难找到有关 Docker Swarm 的相关材料。

使用 Docker Swarm 的好处之一是您不必学习新的命令。如果您已经习惯了 Docker 和 Docker Compose，Swarm 可以使用相同的资源。它允许特定选项扩展 Docker 以处理部署。

使用 Swarm 编排的两个服务看起来像这样（`docker-compose.yml`）：

```cpp
version: "3.8"
services:
  web:
    image: nginx
    ports:
      - "80:80"
    depends_on:
      - merchant
  merchant:
    image: hosacpp/merchant:v2.0.3
    deploy:
      replicas: 3
    ports:
      - "8000"
```

应用配置时，您可以运行`docker stack deploy --compose-file docker-compose.yml dominican`。

### Nomad

Nomad 与前两种解决方案不同，因为它不仅专注于容器。它是一个通用的编排器，支持 Docker、Podman、Qemu 虚拟机、隔离的 fork/exec 和其他几种任务驱动程序。如果您想获得容器编排的一些优势而不将应用迁移到容器中，那么了解 Nomad 是值得的。

它相对容易设置，并且与其他 HashiCorp 产品（如 Consul 用于服务发现和 Vault 用于秘密管理）很好地集成。与 Docker 或 Kubernetes 一样，Nomad 客户端可以在本地运行，并连接到负责管理集群的服务器。

Nomad 有三种作业类型可用：

+   **服务**：一个不应该在没有手动干预的情况下退出的长期任务（例如，Web 服务器或数据库）。

+   **批处理**：一个较短寿命的任务，可以在几分钟内完成。如果批处理作业返回指示错误的退出代码，则根据配置重新启动或重新调度。

+   **系统**：必须在集群中的每个节点上运行的任务（例如，日志代理）。

与其他编排器相比，Nomad 在安装和维护方面相对容易。在任务驱动程序或设备插件（用于访问专用硬件，如 GPU 或 FPGA）方面也是可扩展的。与 Kubernetes 相比，Nomad 在社区支持和第三方集成方面欠缺。Nomad 不需要您重新设计应用程序的架构以获得提供的好处，而这在 Kubernetes 中经常发生。

要使用 Nomad 配置这两个服务，我们需要两个配置文件。第一个是`nginx.nomad`：

```cpp
job "web" {
  datacenters = ["dc1"]
  type = "service"
  group "nginx" {
    task "nginx" {
      driver = "docker"
      config {
        image = "nginx"
        port_map {
          http = 80
        }
      }
      resources {
        network {
          port "http" {
              static = 80
          }
        }
      }
      service {
        name = "nginx"
        tags = [ "dominican-front", "web", "nginx" ]
        port = "http"
        check {
          type = "tcp"
          interval = "10s"
          timeout = "2s"
        }
      }
    }
  }
}
```

第二个描述了商户应用程序，因此被称为`merchant.nomad`：

```cpp
job "merchant" {
  datacenters = ["dc1"]
  type = "service"
  group "merchant" {
    count = 3
    task "merchant" {
      driver = "docker"
      config {
        image = "hosacpp/merchant:v2.0.3"
        port_map {
          http = 8000
        }
      }
      resources {
        network {
          port "http" {
              static = 8000
          }
        }
      }
      service {
        name = "merchant"
        tags = [ "dominican-front", "merchant" ]
        port = "http"
        check {
          type = "tcp"
          interval = "10s"
          timeout = "2s"
        }
      }
    }
  }
}
```

要应用配置，您需要运行`nomad job run merchant.nomad && nomad job run nginx.nomad`。

### OpenShift

OpenShift 是红帽的基于 Kubernetes 构建的商业容器平台。它包括许多在 Kubernetes 集群的日常运营中有用的附加组件。您将获得一个容器注册表，一个类似 Jenkins 的构建工具，用于监控的 Prometheus，用于服务网格的 Istio 和用于跟踪的 Jaeger。它与 Kubernetes 不完全兼容，因此不应将其视为可直接替换的产品。

它是建立在现有的红帽技术之上，如 CoreOS 和红帽企业 Linux。您可以在本地使用它，在红帽云中使用它，在受支持的公共云提供商之一（包括 AWS、GCP、IBM 和 Microsoft Azure）中使用它，或者作为混合云使用。

还有一个名为 OKD 的开源社区支持项目，它是红帽 OpenShift 的基础。如果您不需要商业支持和 OpenShift 的其他好处，仍然可以在 Kubernetes 工作流程中使用 OKD。

## 托管服务

如前所述，一些前述的编排器也可以作为托管服务提供。例如，Kubernetes 可以作为多个公共云提供商的托管解决方案。本节将向您展示一些不基于上述任何解决方案的容器编排的不同方法。

### AWS ECS

在 Kubernetes 发布其 1.0 版本之前，亚马逊网络服务提出了自己的容器编排技术，称为弹性容器服务（ECS）。ECS 提供了一个编排器，可以在需要时监视、扩展和重新启动您的服务。

要在 ECS 中运行容器，您需要提供工作负载将运行的 EC2 实例。您不需要为编排器的使用付费，但您需要为通常使用的所有 AWS 服务付费（例如底层的 EC2 实例或 RDS 数据库）。

ECS 的一个重要优势是其与 AWS 生态系统的出色集成。如果您已经熟悉 AWS 服务并投资于该平台，您将更容易理解和管理 ECS。

如果您不需要许多 Kubernetes 高级功能和其扩展功能，ECS 可能是更好的选择，因为它更直接，更容易学习。

### AWS Fargate

AWS 还提供了另一个托管的编排器 Fargate。与 ECS 不同，它不需要您为底层的 EC2 实例进行配置和付费。您需要关注的唯一组件是容器、与其连接的网络接口和 IAM 权限。

与其他解决方案相比，Fargate 需要的维护量最少，也是最容易学习的。由于现有的 AWS 产品在这一领域已经提供了自动扩展和负载平衡功能。

这里的主要缺点是与 ECS 相比，您为托管服务支付的高额费用。直接比较是不可能的，因为 ECS 需要支付 EC2 实例的费用，而 Fargate 需要独立支付内存和 CPU 使用费用。对集群缺乏直接控制可能会导致一旦服务开始自动扩展就会产生高昂的成本。

### Azure Service Fabric

所有先前解决方案的问题在于它们大多针对首先是 Linux 中心的 Docker 容器。另一方面，Azure Service Fabric 是由微软支持的首先是 Windows 的产品。它可以在不修改的情况下运行传统的 Windows 应用程序，这可能有助于您迁移应用程序，如果它依赖于这些服务。

与 Kubernetes 一样，Azure Service Fabric 本身并不是一个容器编排器，而是一个平台，您可以在其上构建应用程序。其中一个构建块恰好是容器，因此它作为编排器运行良好。

随着 Azure Kubernetes Service 的最新推出，这是 Azure 云中的托管 Kubernetes 平台，使用 Service Fabric 的需求减少了。

# 总结

当您是现代软件的架构师时，您必须考虑现代技术。考虑它们并不意味着盲目地追随潮流；它意味着能够客观地评估特定建议是否在您的情况下有意义。

在前几章中介绍的微服务和本章介绍的容器都值得考虑和理解。它们是否值得实施？这在很大程度上取决于您正在设计的产品类型。如果您已经读到这里，那么您已经准备好自己做出决定了。

下一章专门讨论云原生设计。这是一个非常有趣但也复杂的主题，涉及面向服务的架构、CI/CD、微服务、容器和云服务。事实证明，C++的出色性能是一些云原生构建块的受欢迎特性。

# 问题

1.  应用程序容器与操作系统容器有何不同？

1.  UNIX 系统中一些早期的沙盒环境示例是什么？

1.  为什么容器非常适合微服务？

1.  容器和虚拟机之间的主要区别是什么？

1.  应用程序容器何时不是一个好选择？

1.  有哪些构建多平台容器映像的工具？

1.  除了 Docker，还有哪些其他容器运行时？

1.  一些流行的编排器是什么？

# 进一步阅读

+   *学习 Docker-第二版*：[`www.packtpub.com/product/learning-docker-second-edition/9781786462923`](https://www.packtpub.com/product/learning-docker-second-edition/9781786462923)

+   *学习 OpenShift*：[`www.packtpub.com/product/learn-openshift/9781788992329`](https://www.packtpub.com/product/learn-openshift/9781788992329)

+   *面向开发人员的 Docker*：[`www.packtpub.com/product/docker-for-developers/9781789536058`](https://www.packtpub.com/product/docker-for-developers/9781789536058)
