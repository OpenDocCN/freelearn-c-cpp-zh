# 第十三章：评估

本节包含所有章节的问题答案。

# 第六章，扩展预处理器

1.  大多数情况下，标记是从提供的源代码中收集的，但在某些情况下，标记可能会在 `Preprocessor` 内部动态生成。例如，内置宏 `__LINE__` 被展开为当前行号，而宏 `__DATE__` 被展开为当前的日历日期。Clang 如何将这些生成的文本内容放入 `SourceManager` 的源代码缓冲区？Clang 如何将这些 `SourceLocation` 分配给这些标记？

    +   开发者可以利用 `clang::ScratchBuffer` 类来插入动态的 `Token` 实例。

1.  当我们讨论实现自定义的 `PragmaHandler` 时，我们使用 `Preprocessor::Lex` 来获取紧随 `pragma` 名称之后的标记，直到遇到 `eod` 标记类型。我们能否在 `eod` 标记之后继续进行词法分析？如果你可以消费 `#pragma` 指令之后任意跟随的标记，你将做些什么有趣的事情？

    +   是的，我们可以在 `eod` 标记之后继续进行词法分析。它只是消费了 `#pragma` 行之后的内 容。这样，你可以创建一个自定义的 `#pragma`，允许你写入 *任意* 内容（在其下方）——例如，编写 Clang 不支持的编程语言。以下是一个示例：

        ```cpp
        #pragma that allows you to define a JavaScript function below it.
        ```

1.  在 *开发自定义预处理器插件和回调* 部分的 `macro guard` 项目中，警告消息的格式为 `[WARNING] In <source location>: ….`。显然，这不是我们从 `Clang` 看到的典型编译器警告，其格式看起来像 `<source location>: warning: …`：

    ```cpp
    ./simple_warn.c:2:7: warning: unused variable 'y'…
      int y = x + 1;
          ^
    1 warning generated.
    ```

    `warning` 字符串甚至在支持的终端中着色。我们如何打印这样的警告消息？Clang 中是否有用于此目的的基础设施？

    +   开发者可以使用 Clang 中的诊断框架来打印此类消息。在 *第七章* 的 *打印诊断消息* 部分，即 *处理 AST*，我们将向您展示该框架的一些用法。

# 第八章，与编译器标志和工具链一起工作

1.  覆盖汇编和链接阶段是很常见的，因为不同的平台通常支持不同的汇编器和链接器。但是，是否可以覆盖 *编译* 阶段（即 Clang）？如果可以，我们该如何做？人们可能出于什么原因这样做？

    +   你可以覆盖 `ToolChain::SelectTool` 方法并提供一个替代的 `Tool` 实例（它代表编译阶段），根据参数提供。以下是一个示例：

        ```cpp
        Tool*
        MyCompiler – which is a class derived from Tool, if we are trying to compile the code for a certain hardware architecture.Providing an alternative compiler instance is useful when your target platform (for example, the `CUSTOM_HARDWARE` in the preceding snippet) or input file is not supported by Clang, but you still want to use the *same* `clang` command-line interface for all the build jobs. For example, suppose you are trying to cross-compile the same projects to *multiple* different architectures, but some of them are not supported by Clang yet. Therefore, you can create a custom Clang toolchain and redirect the compilation job to an external compiler (for example, `gcc`) when the `clang` command-line tool is asked to build the project for those architectures.
        ```

1.  当我们处理 `tools::zipline::Linker::ConstructJob` 时，我们简单地使用 `llvm_unreachable` 来通过 `-fuse-ld` 标志退出编译过程，如果用户提供了不支持的压缩器名称。我们可以用 Clang 的 `Driver` 类提供的快捷方式来替换它，访问诊断框架。在一个 `Tool` 的派生类中，您可以使用 `getToolChain().getDriver()` 来获取一个 `Driver` 实例，然后使用 `Driver::Diag` 方法打印出诊断信息。

1.  就像我们可以使用 `-Xclang` 将标志直接传递给前端一样，我们也可以通过驱动器标志（如 `-Wa` 用于汇编器，`-Wl` 用于链接器）将汇编器特定或链接器特定的标志直接传递给汇编器或链接器。我们如何在 Zipline 的自定义汇编器和链接器阶段消耗这些标志？

    +   在 `ConstructJob` 方法内部，您可以读取 `options::OPT_Wa_COMMA` 和 `options::OPT_Wl_COMMA` 的值，分别检索汇编器和链接器特定的命令行标志。以下是一个示例：

        ```cpp
        void
        MyAssembler::ConstructJob(Compilation &C,
                                  const JobAction &JA,
                                  const InputInfo &Output,
                                  const InputInfoList &Inputs,
                                  const ArgList &Args,
                                  const char *LinkingOutput)                           const {
          if (Arg *A = Args.getLastArg(options::OPT_Wl_COMMA)) {
            // `A` contains linker-specific flags
            …
          }
          …
        }
        ```

# 第九章，使用 PassManager 和 AnalysisManager

1.  在 *Writing a LLVM Pass for the new PassManager* 部分的 StrictOpt 示例中，我们如何编写一个不继承 `PassInfoMixin` 类的 Pass？

    +   `PassInfoMixin` 类仅为您定义了一个实用函数 `name`，该函数返回此 Pass 的名称。因此，您可以轻松地自己创建一个。以下是一个示例：

        ```cpp
        struct MyPass {
          static StringRef name() { return "MyPass"; }
          PreservedAnalyses run(Function&, FunctionAnalysisManager&);
        };
        ```

1.  我们如何为新的 PassManager 开发自定义的仪器？我们如何在不修改 LLVM 源树的情况下完成它？（提示：使用本章中我们学习过的 Pass 插件。）

    +   Pass 仪器是一段在 LLVM Pass 之前和/或之后运行的代码。这篇博客文章展示了通过 Pass 插件开发自定义 Pass 仪器的一个示例：`medium.com/@mshockwave/writing-pass-instrument-for-llvm-newpm-f17c57d3369f`。
