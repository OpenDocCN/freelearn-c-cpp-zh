# 什么是 WebAssembly？

**WebAssembly**（**Wasm**）是 Web 平台的一个重要里程碑。它使开发者能够在无需插件或浏览器锁定的情况下在网络上运行编译后的代码，这带来了许多新的机会。关于 WebAssembly 是什么，存在一些混淆，对其持久性的怀疑也存在。

在本章中，我们将讨论 WebAssembly 如何产生，官方定义下的 WebAssembly 是什么，以及它所包含的技术。还将涵盖潜在用途案例、支持的语言和限制，以及如何找到更多信息。

我们本章的目标是理解以下内容：

+   为 WebAssembly 开辟道路的技术

+   WebAssembly 是什么以及它的一些潜在用途案例

+   哪些编程语言可以与 WebAssembly 一起使用

+   WebAssembly 的当前限制

+   WebAssembly 如何与 Emscripten 和 asm.js 相关

# WebAssembly 的道路

至少可以说，Web 开发有着有趣的历史。已经尝试过几次（失败的）扩展平台以支持不同的语言。像插件这样的笨拙解决方案未能经受时间的考验，将用户限制在单个浏览器中是灾难的预兆。

WebAssembly 是作为一个优雅的解决方案来应对自浏览器首次能够执行代码以来就存在的问题：*如果你想要为 Web 开发，你必须使用 JavaScript*。幸运的是，使用 JavaScript 并没有像 2000 年初那样有负面含义，但它作为编程语言仍然存在某些限制。在本节中，我们将讨论导致 WebAssembly 的技术，以便更好地理解为什么需要这项新技术。

# JavaScript 的演变

JavaScript 是由布兰登·艾奇在 1995 年仅用 10 天创造的。最初被程序员视为一种 *玩具* 语言，主要用于在网页上使按钮闪烁或显示横幅。在过去的十年里，JavaScript 从一种玩具语言演变成为一个具有深远能力和庞大追随者的平台。

2008 年浏览器市场的激烈竞争导致了 **即时**（**JIT**）编译器的添加，这使得 JavaScript 的执行速度提高了 10 倍。Node.js 在 2009 年推出，代表了 Web 开发的一个范式转变。瑞安·达尔结合了谷歌的 V8 JavaScript 引擎、事件循环和低级 I/O API，构建了一个允许在服务器和客户端使用 JavaScript 的平台。Node.js 导致了 `npm` 的出现，这是一个包管理器，允许在 Node.js 生态系统中开发库。截至写作时，有超过 600,000 个包可用，每天都有数百个新包被添加：

![](img/d473abb9-dda2-4db0-acfb-0a63607c8190.png)

从 Modulecounts 获取的 npm 包数量增长情况，自 2012 年以来

不仅 Node.js 生态系统在增长；JavaScript 本身也在积极开发中。负责制定 JavaScript 标准并监督新语言特性添加的 ECMA **技术委员会 39**（**TC39**），通过社区驱动的提案流程，每年都会发布 JavaScript 的更新。凭借其丰富的库和工具、语言的持续改进以及庞大的程序员社区，JavaScript 已经成为不可忽视的力量。

但这种语言确实存在一些不足：

+   直到最近，JavaScript 只包含 64 位浮点数。这可能导致处理非常大或非常小的数字时出现问题。`BigInt`，一种新的数值原语，可以缓解一些这些问题，目前正在被添加到 ECMAScript 规范中，但可能需要一些时间才能在浏览器中得到全面支持。

+   JavaScript 是弱类型语言，这增加了其灵活性，但可能导致混淆和错误。它实际上给你提供了足够的绳子来自挂。

+   尽管浏览器供应商做出了最大努力，JavaScript 的性能仍然不如编译型语言。

+   如果开发者想要创建一个网络应用程序，他们需要学习 JavaScript——无论他们是否喜欢。

为了避免编写超过几行 JavaScript，一些开发者构建了 **转换器**（transpilers）来将其他语言转换为 JavaScript。转换器（或源到源编译器）是一种编译器，它将一种编程语言的源代码转换为另一种编程语言的等效源代码。TypeScript，这是一种流行的前端 JavaScript 开发工具，可以将 TypeScript 转换为目标浏览器或 Node.js 的有效 JavaScript。选择任何编程语言，都有很大可能性有人为它创建了 JavaScript 转换器。例如，如果你喜欢编写 Python，你大约有 15 种不同的工具可以使用来生成 JavaScript。然而，最终，它仍然是 JavaScript，所以你仍然会受到语言特性的影响。

随着网络逐渐成为构建和分发应用程序的有效平台，越来越多的复杂和资源密集型应用程序被创建。为了满足这些应用程序的需求，浏览器供应商开始开发新技术，以将其集成到他们的软件中，而不会干扰网络开发的正常进程。Chrome 和 Firefox 的创造者 Google 和 Mozilla 分别采取了不同的路径来实现这一目标，最终导致了 WebAssembly 的诞生。

# Google 和 Native Client

Google 开发了 **原生客户端**（**NaCl**），旨在在网页浏览器中安全地运行原生代码。可执行代码将在 **沙盒**中运行，并提供了原生代码执行的性能优势。

在软件开发背景下，沙盒是一个防止可执行代码与系统其他部分交互的环境。它的目的是防止恶意代码的传播并对软件的功能施加限制。

NaCl 与特定架构绑定，而 **可移植本地客户端** (**PNaCl**) 是一个架构无关的 NaCl 版本，旨在在任何平台上运行。该技术由两个元素组成：

+   能够将 C/C++ 代码转换为 NaCl 模块的工具链

+   运行时组件，这些是嵌入浏览器中允许执行 NaCl 模块的组件：

![](img/ee7ab5c5-f671-4caa-8073-2c3ef941c399.png)

本地客户端工具链及其输出

NaCl 的架构特定可执行文件 (`nexe`) 仅限于从 Google 的 Chrome Web Store 安装的程序和扩展，但 PNaCl 可执行文件 (`pexe`) 可以在网络上自由分发并嵌入到 Web 应用程序中。通过 Pepper，一个用于创建 NaCl 模块的开源 API 以及其相应的插件 API (PPAPI)，实现了可移植性。Pepper 允许 NaCl 模块与宿主浏览器之间的通信，并以安全且可移植的方式访问系统级功能。应用程序可以通过包含清单文件和编译模块 (`pexe`) 以及相应的 HTML、CSS 和 JavaScript 来轻松分发：

![](img/0a230248-b946-4f66-b811-ff2530fc48d1.png)

Pepper 在本地客户端应用程序中的作用

NaCl 为克服网络性能限制提供了有希望的机会，但它也有一些缺点。尽管 Chrome 内置了对 PNaCl 可执行文件和 Pepper 的支持，但其他主要浏览器并没有。该技术的批评者对应用程序的黑盒性质以及潜在的安全风险和复杂性提出了问题。

Mozilla 将其努力集中在通过 `asm.js` 提高 JavaScript 的性能上。由于 Pepper 的 API 规范不完整和有限的文档，他们不会在 Firefox 中添加对 Pepper 的支持。最终，NaCl 于 2017 年 5 月被弃用，转而使用 WebAssembly。

# Mozilla 和 asm.js

Mozilla 于 2013 年推出了 `asm.js`，并为开发者提供了一种将他们的 C 和 C++ 源代码转换为 JavaScript 的方法。`asm.js` 的官方规范将其定义为 JavaScript 的一个严格子集，可以作为编译器的低级、高效目标语言使用。它仍然是有效的 JavaScript，但语言特性仅限于那些适合 **编译时优化** (**AOT**) 的特性。AOT 是浏览器 JavaScript 引擎使用的一种技术，通过将代码编译成原生机器代码来提高代码的执行效率。`asm.js` 通过实现 100% 的类型一致性和手动内存管理来实现这些性能提升。

使用像 Emscripten 这样的工具，可以将 C/C++ 代码转换为 `asm.js` 并通过与正常 JavaScript 相同的方式轻松分发。访问 `asm.js` 模块中的函数需要 **链接**，这涉及到调用其函数以获取包含模块导出的对象。

`asm.js` 非常灵活，然而，与模块的某些交互可能会导致性能损失。例如，如果 `asm.js` 模块被赋予了访问一个失败动态或静态验证的自定义 JavaScript 函数的权限，代码就无法利用 AOT 并回退到解释器：

![图片](img/3b40bcf8-0a50-4ed5-806e-3f3a5f64679a.png)

asm.js AOT 编译工作流程

`asm.js` 不仅仅是一个垫脚石。它是 WebAssembly 的 **最小可行产品** (**MVP**) 的基础。官方 WebAssembly 网站在标题为 *WebAssembly 高级目标* 的部分中明确提到了 `asm.js`。

所以为什么要在可以使用 `asm.js` 的情况下创建 WebAssembly 呢？除了可能出现的性能损失外，一个 `asm.js` 模块是一个必须在任何编译发生之前通过网络传输的文本文件。而 WebAssembly 模块是二进制格式，由于其尺寸更小，这使得传输效率更高。

WebAssembly 模块使用基于承诺的实例化方法，这利用了现代 JavaScript 并消除了任何 *是否已加载* *了？* 代码的需求。

# WebAssembly 诞生

**万维网联盟** (**W3C**)，一个旨在开发 Web 标准的国际社区，于 2015 年 4 月成立了 WebAssembly 工作组，以标准化 WebAssembly 并监督规范和提案流程。从那时起，已经发布了 *核心规范* 以及相应的 *JavaScript API* 和 *Web API*。浏览器对 WebAssembly 的初始支持基于 `asm.js` 的功能集。WebAssembly 的二进制格式和相应的 `.wasm` 文件结合了 `asm.js` 输出与 PNaCl 的分布式可执行概念。

那么 WebAssembly 将如何成功而 NaCl 失败呢？根据 Dr. Axel Rauschmayer 的说法，有三个原因，详细内容请见 [`2ality.com/2015/06/web-assembly.html#what-is-different-this-time`](http://2ality.com/2015/06/web-assembly.html#what-is-different-this.time)：

"首先，这是一个协作努力，没有哪家公司是独自行动的。目前，以下项目参与了其中：Firefox、Chromium、Edge 和 WebKit。

第二，与 Web 平台和 JavaScript 的互操作性非常好。从 JavaScript 中使用 WebAssembly 代码将像导入一个模块一样简单。

第三，这并不是要取代 JavaScript 引擎，更多的是要为它们添加一个新特性。这大大减少了实现 WebAssembly 的工作量，并有助于获得 Web 开发社区的认可。

- Dr. Axel Rauschmayer

# 那么 WebAssembly 究竟是什么，我可以在哪里使用它？

WebAssembly 在官方网站上有简洁且描述性的定义，但它只是拼图的一部分。还有其他几个组件属于 WebAssembly 的范畴。了解每个组件所扮演的角色将使你对这项技术有一个更全面的了解。在本节中，我们将详细分解 WebAssembly 的定义并描述潜在的应用场景。

# 官方定义

官方 WebAssembly 网站 ([`webassembly.org`](https://webassembly.org)) 提供了以下定义：

Wasm 是一种基于栈的虚拟机的二进制指令格式。Wasm 被设计为编译高级语言（如 C/C++/Rust）的可移植目标，使得客户端和服务器应用程序能够在网页上部署。

让我们将这个定义分解成几个部分以增加一些解释。

# 二进制指令格式

WebAssembly 实际上包含几个元素——一个二进制格式和一个文本格式，这些都在 *核心规范* 中进行了文档化，相应的 API（JavaScript 和网页），以及一个编译目标。二进制和文本格式都映射到一个以 **抽象语法** 形式的公共结构。为了更好地理解抽象语法，它可以在 **抽象语法树**（**AST**）的上下文中进行解释。AST 是编程语言源代码结构的树形表示。例如，ESLint 这样的工具使用 JavaScript 的 AST 来查找代码风格错误。以下示例包含一个函数及其对应的 JavaScript（来自 [`astexplorer.net`](https://astexplorer.net)）AST。

以下是一个简单的 JavaScript 函数：

```cpp
function doStuff(thingToDo) {
  console.log(thingToDo);
}
```

对应的 AST 如下所示：

```cpp
{
  "type": "Program",
  "start": 0,
  "end": 57,
  "body": [
    {
      "type": "FunctionDeclaration",
      "start": 9,
      "end": 16,
      "id": {
        "type": "Identifier",
        "start": 17,
        "end": 26,
        "name": "doStuff"
      },
      "generator": false,
      "expression": false,
      "params": [
        {
          "type": "Identifier",
          "start": 28,
          "end": 57,
          "name": "thingToDo"
        }
      ],
      "body": {
        "type": "BlockStatement",
        "start": 32,
        "end": 55,
        "body": [
          {
            "type": "ExpressionStatement",
            "start": 32,
            "end": 55,
            "expression": {
              "type": "CallExpression",
              "start": 32,
              "end": 54,
              "callee": {
                "type": "MemberExpression",
                "start": 32,
                "end": 43,
                "object": {
                  "type": "Identifier",
                  "start": 32,
                  "end": 39,
                  "name": "console"
                },
                "property": {
                  "type": "Identifier",
                  "start": 40,
                  "end": 43,
                  "name": "log"
                },
                "computed": false
              },
              "arguments": [
                {
                  "type": "Identifier",
                  "start": 44,
                  "end": 53,
                  "name": "thingToDo"
                }
              ]
            }
          }
        ]
      }
    }
  ],
  "sourceType": "module"
}
```

AST 可能比较冗长，但它出色地描述了程序组件。以 AST 的形式表示源代码使得验证和编译变得简单高效。WebAssembly 的文本格式代码被序列化为 AST 并编译成二进制格式（作为 `.wasm` 文件），然后由网页获取、加载和使用。当模块加载时，浏览器的 JavaScript 引擎使用一个 **解码栈** 将 `.wasm` 文件解码为 AST，执行类型检查，并对其进行解释以执行函数。WebAssembly 最初是一种用于 AST 的二进制指令格式。由于验证返回 `void` 的 Wasm 表达式的性能影响，二进制指令格式被更新以针对 **栈机器**。

栈机器由两个元素组成：栈和指令。栈是一种具有两种操作的数据结构：*push* 和 *pop*。项目被推入栈中，随后以 **后进先出**（**LIFO**）的顺序从栈中弹出。栈还包括一个 **指针**，它指向栈顶的项目。指令代表对栈中的项目执行的操作。例如，一个 `ADD` 指令可能会从栈中弹出顶部两个项目（值 `100` 和 `10`），并将一个包含总和的单个项目推回栈中（值 `110`）：

![图片](img/1e2d75b9-2021-4486-8049-726f6134410d.png)

简单的栈机器

WebAssembly 的栈机器以相同的方式运行。一个程序计数器（指针）维护代码中的执行位置，一个虚拟控制栈跟踪 `blocks` 和 `if` 构造在进入（推入）和退出（弹出）时的状态。指令在没有引用抽象语法树（AST）的情况下执行。因此，定义中的 **二进制指令格式** 部分指的是指令的二元表示，这些指令的格式可以被浏览器中的解码栈读取。

# 可移植的编译目标

WebAssembly 从一开始就被设计成具有可移植性。在这个上下文中，可移植性意味着 WebAssembly 的二进制格式可以在各种操作系统和指令集架构上高效执行，无论是在网络上还是在网络之外。WebAssembly 的规范定义了在执行环境中的可移植性。WebAssembly 被设计成在满足某些特定特性的环境中高效运行，其中大部分与内存相关。WebAssembly 的可移植性也可以归因于核心技术的周围没有特定的 API。相反，它定义了一个 `import` 机制，其中可用的导入集由宿主环境定义。

简而言之，这意味着 WebAssembly 并不绑定到特定的环境，例如网络或桌面。WebAssembly 工作组定义了一个 *Web API*，但这与 *核心规范* 是分开的。*Web API* 旨在满足 WebAssembly，而不是相反。

定义中的 **编译** 方面表明，WebAssembly 将很容易从用高级语言编写的源代码编译为其二进制格式。MVP 专注于两种语言，C 和 C++，但鉴于其与 C++ 的相似性，也可以使用 Rust。编译将通过使用 Clang/LLVM 后端来实现，尽管在这本书中我们将使用 Emscripten 生成我们的 Wasm 模块。计划最终添加对其他语言和编译器（如 GCC）的支持，但 MVP 专注于 LLVM。

# 核心规范

官方定义对整体技术提供了一些高级见解，但为了完整性，值得深入挖掘。WebAssembly 的**核心规范**是如果您想非常细致地了解 WebAssembly，则应参考的官方文件。如果您对了解执行环境的运行时结构特性感兴趣，请查看第四部分：*执行*。我们在这里不会涵盖这一点，但了解**核心规范**的位置将有助于建立 WebAssembly 的完整定义。

# 语言概念

**核心规范**指出 WebAssembly 编码了一种低级、类似汇编的编程语言。该规范定义了这种语言的结构、执行和验证，以及二进制和文本格式的细节。该语言本身围绕以下概念构建：

+   **值**，或者说 WebAssembly 提供的值类型

+   在堆栈机器中执行的**指令**

+   在错误条件下产生的**陷阱**和终止执行

+   **函数**，代码组织到其中，每个函数都接受一系列值作为参数，并返回一系列值作为结果

+   **表**，这是特定元素类型（如函数引用）的值数组，可以被执行程序选择

+   **线性内存**，这是一个可以用来存储和加载值的原始字节数组

+   **模块**，包含函数、表和线性内存的 WebAssembly 二进制文件（`.wasm`文件）

+   **嵌入器**，WebAssembly 可以在宿主环境（如网页浏览器）中执行的方式

函数、表、内存和模块与**JavaScript API**有直接关联，并且需要了解。这些概念描述了语言本身的底层结构以及如何编写或编码 WebAssembly。关于使用方面，理解 WebAssembly 的相应语义阶段提供了该技术的完整定义：

![图片](img/52382cbb-fa93-4206-adde-c38848bf1429.png)

语言概念及其关系

# 语义阶段

**核心规范**描述了编码的模块（`.wasm`文件）在宿主环境（如网页浏览器）中利用时经历的各个阶段。该规范的这一方面代表了如何处理和执行输出：

+   **解码**：将二进制格式转换为模块

+   **验证**：解码的模块经过验证检查（如类型检查），以确保模块结构良好且安全

+   **执行，第一部分：实例化**：通过初始化**全局变量**、**内存**和**表**来实例化模块实例，这是模块的动态表示，并调用模块的`start()`函数

+   **执行，第二部分：调用**：从模块实例中调用导出的函数：

以下图表提供了语义阶段的视觉表示：

![](img/51755ca6-f4c7-43b3-93d3-81575523ae30.png)

模块使用的语义阶段

# JavaScript 和 Web API

WebAssembly 工作组还发布了与 JavaScript 和网页交互的 API 规范，这使得它们有资格被纳入 WebAssembly 技术空间。*JavaScript API*的范围限定在 JavaScript 语言本身，而不特定于某个环境（例如，网络浏览器或 Node.js）。它定义了与 WebAssembly 交互以及管理编译和实例化过程的类、方法和对象。*Web API*是*JavaScript API*的扩展，它定义了特定于网络浏览器的功能。*Web API*规范目前仅定义了两个方法，`compileStreaming`和`instantiateStreaming`，这些是简化浏览器中 Wasm 模块使用的便利方法。这些内容将在第二章，*WebAssembly 元素 - Wat, Wasm 和 JavaScript API*中更详细地介绍。

# 那它是否会取代 JavaScript？

WebAssembly 的最终目标不是取代 JavaScript，而是与之互补。JavaScript 丰富的生态系统和灵活性仍然使其成为网络的最佳语言。WebAssembly 的 JavaScript API 使得两种技术之间的互操作性相对简单。那么，你能否仅使用 WebAssembly 构建一个 Web 应用？WebAssembly 的一个明确目标是可移植性，复制 JavaScript 的所有功能可能会阻碍这一目标。然而，官方网站包括一个目标，即与现有 Web 平台良好执行和集成，所以只有时间才能揭晓。可能不实用将整个代码库用编译成 WebAssembly 的语言编写，但将一些应用程序逻辑移动到 Wasm 模块中可能在性能和加载时间方面有益。

# 我在哪里可以使用它？

WebAssembly 的官方网站列出了大量的潜在用例。我不会在这里全部介绍，但有几个用例代表了 Web 平台能力的重大提升：

+   图像/视频编辑

+   游戏

+   音乐应用（流媒体，缓存）

+   图像识别

+   实时视频增强

+   虚拟现实和增强现实

虽然一些用例在技术上可以使用 JavaScript、HTML 和 CSS 实现，但使用 WebAssembly 可以提供显著的性能提升。提供二进制文件（而不是单个 JavaScript 文件）可以大大减少包的大小，在页面加载时实例化 Wasm 模块可以加快代码执行速度。

WebAssembly 不仅限于浏览器。在浏览器之外，你可以用它来在移动设备上构建混合原生应用或执行不受信任代码的服务器端计算。使用 Wasm 模块为手机应用提供的服务在功耗和性能方面可能非常有益。

WebAssembly 还提供了关于其使用方式的灵活性。你可以用 WebAssembly 编写整个代码库，尽管在当前形式或 Web 应用程序的环境中这可能并不实用。鉴于 WebAssembly 强大的 JavaScript API，你可以用 JavaScript/HTML 编写 UI，并使用 Wasm 模块来实现不直接访问 DOM 的功能。一旦支持更多语言，对象可以轻松地在 Wasm 模块和 JavaScript 代码之间传递，这将极大地简化集成并增加开发者的采用率。

# 支持哪些语言？

WebAssembly 的 MVP 的高级目标是为 `asm.js` 提供大致相同的功能。这两种技术非常相似。C、C++ 和 Rust 是支持手动内存分配的非常流行的语言，这使得它们成为初始实现的理想候选。在本节中，我们将简要概述每种编程语言。

# C 和 C++

C 和 C++ 是存在了超过 30 年的低级编程语言。C 是过程式编程语言，本身不支持类和继承等面向对象编程概念，但它速度快、可移植且广泛使用。

C++ 是为了填补 C 的不足而构建的，它通过添加如运算符重载和改进的类型检查等特性来增强 C 的功能。这两种语言一直位居最受欢迎的编程语言前十名，这使得它们非常适合 MVP：

![](img/049542c7-ebe1-4b35-b4a1-de0ec95a532c.png)

TIOBE 编程语言长期趋势排名前十

C 和 C++ 的支持也集成到了 Emscripten 中，因此除了简化编译过程外，它还允许你利用 WebAssembly 的全部功能。使用 LLVM 也可以将 C/C++ 代码编译成 `.wasm` 文件。LLVM 是一组模块化和可重用的编译器和工具链技术。简而言之，它是一个简化从源代码到机器代码配置过程的框架。如果你自己开发了一种编程语言并希望构建编译器，LLVM 提供了简化该过程的工具。我将在第十章“高级工具和即将推出的功能”中介绍如何使用 LLVM 将 C/C++ 编译成 `.wasm` 文件。

以下代码片段展示了如何使用 C++ 将 `Hello World!` 打印到控制台：

```cpp
#include <iostream>

int main() {
    std::cout << "Hello, World!\n";
    return 0;
}
```

# Rust

C 和 C++ 被设计为 WebAssembly 的主要使用语言，但 Rust 是一个完全合适的替代品。Rust 是一种与 C++ 语法相似的系统编程语言，它以内存安全为设计理念，同时仍保留了 C 的性能优势。Rust 编译器的当前夜间构建可以从 Rust 源代码生成 `.wasm` 文件，因此如果你喜欢 Rust 并且熟悉 C++，你应该能够使用 Rust 来完成本书中的大多数示例。

以下代码片段演示了如何使用 Rust 将`Hello World!`打印到控制台：

```cpp
fn main() {
    println!("Hello World!");
}
```

# 其他语言

存在一些工具可以启用 WebAssembly 与其他一些流行编程语言的结合使用，尽管它们大多是实验性的：

+   通过 Blazor 的 C#

+   通过 WebIDL 的 Haxe

+   通过 TeaVM 或 Bytecoder 的 Java

+   通过 TeaVM 的 Kotlin

+   通过 AssemblyScript 的 TypeScript

技术上也可以将一种语言转换为 C，然后将其编译为 Wasm 模块，但编译的成功取决于转换器的输出。很可能会需要对代码进行重大修改才能使其工作。

# 限制是什么？

诚然，WebAssembly 并非没有局限性。新功能正在积极开发中，该技术也在不断进化，但 MVP 功能仅代表了 WebAssembly 能力的一部分。在本节中，我们将介绍一些这些局限性以及它们如何影响开发过程。

# 没有垃圾回收

WebAssembly 支持扁平线性内存，这本身并不是一个限制，但需要了解如何显式地分配内存以执行代码。C 和 C++是 MVP 的合理选择，因为内存管理是语言本身的一部分。一些更受欢迎的高级语言，如 Java 最初未包括在内，是因为存在一种称为**垃圾回收**（**GC**）的东西。

GC 是一种自动内存管理形式，其中不再被程序使用的对象占用的内存会自动回收。GC 类似于汽车上的自动变速器。它已被熟练的工程师高度优化，以尽可能高效地运行，但限制了驾驶员的控制范围。手动分配内存就像驾驶一辆手动变速的汽车。它提供了对速度和扭矩的更大控制，但误用或缺乏经验可能导致汽车严重损坏。C 和 C++出色的性能和速度部分归因于内存的手动分配。

GC 语言允许你编程时无需担心内存可用性或分配。JavaScript 是 GC 语言的一个例子。浏览器引擎使用一种称为标记-清除算法的东西来收集不可达的对象并释放相应的内存。WebAssembly 目前正在努力支持 GC 语言，但很难说具体何时会完成。

# 没有直接 DOM 访问

WebAssembly 无法直接访问 DOM，因此任何 DOM 操作都需要通过 JavaScript 或使用 Emscripten 等工具间接完成。目前有计划添加直接引用 DOM 和其他 Web API 对象的能力，但这仍处于提案阶段。DOM 操作很可能会与 GC 语言相结合，因为它将允许 WebAssembly 和 JavaScript 代码之间无缝传递对象。

# 在旧浏览器中没有支持

较旧的浏览器没有全局的 `WebAssembly` 对象可供实例化和加载 Wasm 模块。如果找不到该对象，有实验性的 polyfills 会使用 `asm.js`，但 WebAssembly 工作组目前没有计划创建一个。由于 `asm.js` 和 WebAssembly 密切相关，如果 `WebAssembly` 对象不可用，仅提供 `asm.js` 文件仍将提供性能提升，同时兼顾向后兼容性。您可以在 [`caniuse.com/#feat=wasm`](https://caniuse.com/#feat=wasm) 查看当前支持 WebAssembly 的浏览器。

# 它与 Emscripten 有何关联？

Emscripten 是一种源到源编译器，可以从 C 和 C++ 源代码生成 `asm.js`。我们将将其用作构建工具以生成 Wasm 模块。在本节中，我们将快速回顾 Emscripten 与 WebAssembly 的关系。

# Emscripten 的作用

Emscripten 是一个 LLVM 到 JavaScript 的编译器，这意味着它将 Clang（用于 C 和 C++）等编译器的 LLVM 位码输出转换为 JavaScript。它不是一个特定的技术，而是一系列协同工作以构建、编译和运行 `asm.js` 的技术组合。为了生成 Wasm 模块，我们将使用 **Emscripten SDK**（**EMSDK**）管理器：

![图片](img/4d92b3fc-da38-44bc-97b4-e1be3b1fc6a3.png)

使用 EMSDK 生成 Wasm 模块

# EMSDK 和 Binaryen

在 第四章 *安装所需依赖项* 中，我们将安装 EMSDK 并使用它来管理编译 C 和 C++ 到 Wasm 模块所需的依赖项。Emscripten 使用 Binaryen 的 `asm2wasm` 工具将 Emscripten 生成的 `asm.js` 输出编译为 `.wasm` 文件。Binaryen 是一个编译器和工具链基础设施库，包括将各种格式编译到 WebAssembly 模块以及反向转换的工具。了解 Binaryen 的内部工作原理不是使用 WebAssembly 所必需的，但了解底层技术和它们如何协同工作是很重要的。通过将某些标志传递给 Emscripten 的编译命令（`emcc`），我们可以将生成的 `asm.js` 代码管道传输到 Binaryen，以输出我们的 `.wasm` 文件。

# 摘要

在本章中，我们讨论了 WebAssembly 的历史，以及导致其创建的技术。提供了 WebAssembly 定义的详细概述，以便更好地理解涉及到的底层技术。

*核心规范*、*JavaScript API* 和 *Web API* 被视为 WebAssembly 的重要元素，并展示了这项技术将如何发展。我们还回顾了潜在的使用案例、目前支持的语言以及使非支持语言得以使用的工具。

WebAssembly 的限制包括没有 GC、无法直接与 DOM 通信以及不支持旧版浏览器。这些限制被讨论是为了传达这项技术的创新性并揭示其一些不足之处。最后，我们讨论了 Emscripten 在开发过程中的作用以及它在 WebAssembly 开发工作流程中的位置。

在第二章，《WebAssembly 元素 - Wat, Wasm 和 JavaScript API》，我们将更深入地探讨构成 WebAssembly 的元素：WebAssembly 文本格式（**Wat**）、二进制格式（Wasm）、JavaScript 和 Web API。

# 问题

1.  哪两种技术影响了 WebAssembly 的创建？

1.  什么是栈式机器，它与 WebAssembly 有何关联？

1.  WebAssembly 如何补充 JavaScript？

1.  哪三种编程语言可以编译成 Wasm 模块？

1.  LLVM 在 WebAssembly 中扮演什么角色？

1.  WebAssembly 有哪些潜在的应用场景？

1.  DOM 访问和垃圾回收（GC）有何关联？

1.  Emscripten 使用什么工具生成 Wasm 模块？

# 进一步阅读

+   官方 WebAssembly 网站：[`webassembly.org`](https://webassembly.org)

+   原生客户端技术概述：[`developer.chrome.com/native-client/overview`](https://developer.chrome.com/native-client/overview)

+   LLVM 编译器基础设施项目：[`llvm.org`](https://llvm.org)

+   关于 Emscripten：[`kripken.github.io/emscripten-site/docs/introducing_emscripten/about_emscripten.html`](http://kripken.github.io/emscripten-site/docs/introducing_emscripten/about_emscripten.html)

+   asm.js 规范：[`asmjs.org/spec/latest`](http://asmjs.org/spec/latest)
