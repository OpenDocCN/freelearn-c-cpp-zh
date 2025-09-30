

# 第三章：将源文件转换为抽象语法树

正如我们在上一章所学，编译器通常分为两个部分——前端和后端。在本章中，我们将实现编程语言的前端——即主要处理源语言的部分。我们将了解现实世界编译器使用的技巧，并将它们应用到我们的编程语言中。

我们的旅程将从定义我们的编程语言的语法开始，并以一个**抽象语法树**（**AST**）结束，它将成为代码生成的基石。你可以使用这种方法为任何你想要实现编译器的编程语言应用。

在本章中，你将学习以下内容：

+   定义一个真正的编程语言，你将学习关于`tinylang`语言的知识，它是真实编程语言的一个子集，并且你将为它实现编译器前端

+   组织编译器项目的目录结构

+   了解如何处理编译器的多个输入文件

+   处理用户消息并以愉快的方式通知他们问题的技巧

+   使用模块化组件构建词法分析器

+   从语法规则中构建递归下降解析器以执行语法分析

+   通过创建 AST 并分析其特征来执行语义分析

通过本章你将获得的技能，你将能够为任何编程语言构建编译器前端。

# 定义一个真正的编程语言

真实编程带来的挑战比上一章的简单 calc 语言要多。为了查看细节，我们将在这章和下一章中使用*Modula-2*的一个小子集。Modula-2 设计良好，并可选择支持`tinylang`。

让我们从`tinylang`程序的一个例子开始。以下函数使用*欧几里得算法*计算最大公约数：

```cpp

MODULE Gcd;
PROCEDURE GCD(a, b: INTEGER) : INTEGER;
VAR t: INTEGER;
BEGIN
  IF b = 0 THEN
    RETURN a;
  END;
  WHILE b # 0 DO
    t := a MOD b;
    a := b;
    b := t;
  END;
  RETURN a;
END GCD;
END Gcd.
```

现在我们对语言中的程序外观有了感觉，让我们快速浏览一下本章中使用的`tinylang`子集的语法。在接下来的几节中，我们将使用这个语法从中推导出词法分析和解析器：

```cpp

compilationUnit
  : "MODULE" identifier ";" ( import )* block identifier "." ;
Import : ( "FROM" identifier )? "IMPORT" identList ";" ;
Block
  : ( declaration )* ( "BEGIN" statementSequence )? "END" ;
```

Modula-2 的编译单元以`MODULE`关键字开始，后跟模块的名称。模块的内容可以有一个导入模块的列表、声明和一个包含初始化时运行的语句的块：

```cpp

declaration
  : "CONST" ( constantDeclaration ";" )*
  | "VAR" ( variableDeclaration ";" )*
  | procedureDeclaration ";" ;
```

声明引入常量、变量和过程。常量的声明以`CONST`关键字为前缀。同样，变量声明以`VAR`关键字开始。常量的声明非常简单：

```cpp

constantDeclaration : identifier "=" expression ;
```

标识符是常量的名称。值是从一个表达式中派生的，该表达式必须在编译时可计算。变量的声明稍微复杂一些：

```cpp

variableDeclaration : identList ":" qualident ;
qualident : identifier ( "." identifier )* ;
identList : identifier ( "," identifier)* ;
```

为了能够一次声明多个变量，使用了一个标识符列表。类型名称可能来自另一个模块，在这种情况下，它以模块名称为前缀。这被称为 **有资格的标识符**。过程需要最详细的描述：

```cpp

procedureDeclaration
  : "PROCEDURE" identifier ( formalParameters )? ";"
    block identifier ;
formalParameters
  : "(" ( formalParameterList )? ")" ( ":" qualident )? ;
formalParameterList
  : formalParameter (";" formalParameter )* ;
formalParameter : ( "VAR" )? identList ":" qualident ;
```

上一段代码展示了如何声明常量、变量和过程。过程可以有参数和返回类型。正常参数按值传递，而 `VAR` 参数按引用传递。`block` 规则中缺少的另一部分是 `statementSequence`，它是一系列单个语句：

```cpp

statementSequence
  : statement ( ";" statement )* ;
```

如果一个语句后面跟着另一个语句，则该语句由分号分隔。再次强调，仅支持 *Modula-2* 语句的一个子集：

```cpp

statement
  : qualident ( ":=" expression | ( "(" ( expList )? ")" )? )
  | ifStatement | whileStatement | "RETURN" ( expression )? ;
```

该规则的第一部分描述了一个赋值或过程调用。一个有资格的标识符后跟 `:=` 是一个赋值。如果它后面跟着 `(`，则它是一个过程调用。其他语句是常见的控制语句：

```cpp

ifStatement
  : "IF" expression "THEN" statementSequence
    ( "ELSE" statementSequence )? "END" ;
```

`IF` 语句也有简化的语法，因为它只能有一个 `ELSE` 块。有了这个语句，我们可以有条件地保护一个语句：

```cpp

whileStatement
  : "WHILE" expression "DO" statementSequence "END" ;
```

`WHILE` 语句描述了一个由条件保护的循环。与 `IF` 语句一起，这使我们能够在 `tinylang` 中编写简单的算法。最后，缺少的是表达式的定义：

```cpp

expList
  : expression ( "," expression )* ;
expression
  : simpleExpression ( relation simpleExpression )? ;
relation
  : "=" | "#" | "<" | "<=" | ">" | ">=" ;
simpleExpression
  : ( "+" | "-" )? term ( addOperator term )* ;
addOperator
  : "+" | "-" | "OR" ;
term
  : factor ( mulOperator factor )* ;
mulOperator
  : "*" | "/" | "DIV" | "MOD" | "AND" ;
factor
  : integer_literal | "(" expression ")" | "NOT" factor
  | qualident ( "(" ( expList )? ")" )? ;
```

表达式语法与上一章中的 calc 非常相似。仅支持 `INTEGER` 和 `BOOLEAN` 数据类型。

此外，还使用了 `identifier` 和 `integer_literal` 标记。一个 `H`。

这些规则已经很多了，而我们只覆盖了 Modula-2 的一部分！尽管如此，在这个子集中仍然可以编写小型应用程序。让我们来实现一个 `tinylang` 编译器！

# 创建项目布局

`tinylang` 的项目布局遵循我们在 *第一章* 中概述的方法，即 *安装 LLVM*。每个组件的源代码位于 `lib` 目录的子目录中，头文件位于 `include/tinylang` 目录的子目录中。子目录以组件命名。在 *第一章* 中，*安装 LLVM*，我们只创建了 `Basic` 组件。

从上一章，我们知道我们需要实现一个词法分析器、一个解析器、一个抽象语法树（AST）和一个语义分析器。每个都是其自身的组件，分别称为 `Lexer`、`Parser`、`AST` 和 `Sema`。本章将使用的目录结构如下所示：

![图 3.1 – `tinylang` 项目的目录结构](img/B19561_03_1.jpg)

图 3.1 – `tinylang` 项目的目录结构

组件有明确定义的依赖关系。`Lexer` 只依赖于 `Basic`。`Parser` 依赖于 `Basic`、`Lexer`、`AST` 和 `Sema`。`Sema` 只依赖于 `Basic` 和 `AST`。明确的依赖关系有助于我们重用组件。

让我们更详细地看看实现过程！

# 管理编译器的输入文件

一个真正的编译器必须处理许多文件。通常，开发者使用主编译单元的名称调用编译器。这个编译单元可以引用其他文件——例如，通过 C 中的`#include`指令或 Python 或 Modula-2 中的`import`语句。一个导入的模块可以导入其他模块，依此类推。所有这些文件都必须加载到内存中，并经过编译器的分析阶段。在开发过程中，开发者可能会犯语法或语义错误。当检测到错误时，应该打印出包括源行和标记的错误消息。这个基本组件并不简单。

幸运的是，LLVM 提供了一个解决方案：`llvm::SourceMgr`类。通过调用`AddNewSourceBuffer()`方法，可以向`SourceMgr`添加一个新的源文件。或者，可以通过调用`AddIncludeFile()`方法来加载一个文件。这两种方法都返回一个 ID 来标识缓冲区。你可以使用这个 ID 来检索关联文件的内存缓冲区的指针。为了在文件中定义一个位置，你可以使用`llvm::SMLoc`类。这个类封装了一个指向缓冲区的指针。各种`PrintMessage()`方法允许你向用户发出错误和其他信息性消息。

# 处理用户消息

只缺少消息的集中定义。在一个大型软件（如编译器）中，你不想将消息字符串散布在各个地方。如果有更改消息或将其翻译成其他语言的需求，那么最好将它们放在一个中心位置！

一种简单的方法是，每条消息都有一个 ID（一个`enum`成员），一个严重程度级别，如`Error`或`Warning`，以及包含消息的字符串。在你的代码中，你只引用消息 ID。严重程度级别和消息字符串仅在打印消息时使用。这三个项目（ID、安全级别和消息）必须一致管理。LLVM 库使用预处理器来解决这个问题。数据存储在一个以`.def`后缀结尾的文件中，并包含在一个宏名称中。该文件通常被包含多次，宏有不同的定义。定义在`include/tinylang/Basic/Diagnostic.def`文件路径中，如下所示：

```cpp

#ifndef DIAG
#define DIAG(ID, Level, Msg)
#endif
DIAG(err_sym_declared, Error, "symbol {0} already declared")
#undef DIAG
```

第一个宏参数`ID`是枚举标签，第二个参数`Level`是严重程度，第三个参数`Msg`是消息文本。有了这个定义，我们可以定义一个`DiagnosticsEngine`类来发出错误消息。接口在`include/tinylang/Basic/Diagnostic.h`文件中：

```cpp

#ifndef TINYLANG_BASIC_DIAGNOSTIC_H
#define TINYLANG_BASIC_DIAGNOSTIC_H
#include "tinylang/Basic/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <utility>
namespace tinylang {
```

在包含必要的头文件后，可以使用`Diagnostic.def`来定义枚举。为了不污染全局命名空间，使用了一个名为`diag`的嵌套命名空间：

```cpp

namespace diag {
enum {
#define DIAG(ID, Level, Msg) ID,
#include "tinylang/Basic/Diagnostic.def"
};
} // namespace diag
```

`DiagnosticsEngine` 类使用 `SourceMgr` 实例通过 `report()` 方法发出消息。消息可以有参数。为了实现这一功能，使用了 LLVM 提供的变长格式支持。消息文本和严重程度级别通过 `static` 方法检索。作为额外的好处，发出的错误消息数量也被计数：

```cpp

class DiagnosticsEngine {
  static const char *getDiagnosticText(unsigned DiagID);
  static SourceMgr::DiagKind
  getDiagnosticKind(unsigned DiagID);
```

消息字符串由 `getDiagnosticText()` 返回，而级别由 `getDiagnosticKind()` 返回。这两种方法稍后在 `.cpp` 文件中实现：

```cpp

  SourceMgr &SrcMgr;
  unsigned NumErrors;
public:
  DiagnosticsEngine(SourceMgr &SrcMgr)
      : SrcMgr(SrcMgr), NumErrors(0) {}
  unsigned nunErrors() { return NumErrors; }
```

由于消息可以有可变数量的参数，C++ 中的解决方案是使用变长模板。当然，这也被 LLVM 提供的 `formatv()` 函数所使用。要获取格式化的消息，我们只需要转发模板参数：

```cpp

  template <typename... Args>
  void report(SMLoc Loc, unsigned DiagID,
              Args &&... Arguments) {
    std::string Msg =
        llvm::formatv(getDiagnosticText(DiagID),
                      std::forward<Args>(Arguments)...)
            .str();
    SourceMgr::DiagKind Kind = getDiagnosticKind(DiagID);
    SrcMgr.PrintMessage(Loc, Kind, Msg);
    NumErrors += (Kind == SourceMgr::DK_Error);
  }
};
} // namespace tinylang
#endif
```

这样，我们就实现了大多数类。只有 `getDiagnosticText()` 和 `getDiagnosticKind()` 还缺失。它们在 `lib/Basic/Diagnostic.cpp` 文件中定义，并也使用了 `Diagnostic.def` 文件：

```cpp

#include "tinylang/Basic/Diagnostic.h"
using namespace tinylang;
namespace {
const char *DiagnosticText[] = {
#define DIAG(ID, Level, Msg) Msg,
#include "tinylang/Basic/Diagnostic.def"
};
```

如同头文件中定义的那样，`DIAG` 宏被用来检索所需的部分。在这里，我们定义了一个数组来存储文本消息。因此，`DIAG` 宏只返回 `Msg` 部分。对于级别，我们采用相同的方法：

```cpp

SourceMgr::DiagKind DiagnosticKind[] = {
#define DIAG(ID, Level, Msg) SourceMgr::DK_##Level,
include "tinylang/Basic/Diagnostic.def"
};
} // namespace
```

毫不奇怪，这两个函数只是简单地索引数组以返回所需的数据：

```cpp

const char *
DiagnosticsEngine::getDiagnosticText(unsigned DiagID) {
  return DiagnosticText[DiagID];
}
SourceMgr::DiagKind
DiagnosticsEngine::getDiagnosticKind(unsigned DiagID) {
  return DiagnosticKind[DiagID];
}
```

`SourceMgr` 类和 `DiagnosticsEngine` 类的组合为其他组件提供了一个良好的基础。我们首先将在词法分析器中使用它们！

# 结构化词法分析器

正如我们从上一章所知，我们需要一个 `Token` 类和一个 `Lexer` 类。此外，还需要一个 `TokenKind` 枚举来为每个令牌类分配一个唯一的数字。将所有内容放在一个头文件和实现文件中并不易于扩展，因此让我们移动这些项。`TokenKind` 可以被普遍使用，并放置在 `Basic` 组件中。`Token` 和 `Lexer` 类属于 `Lexer` 组件，但被放置在不同的头文件和实现文件中。

有三种不同的令牌类别：`CONST` 关键字、`;` 分隔符和 `ident` 令牌，分别代表源代码中的标识符。每个令牌需要一个枚举成员名称。关键字和标点符号有自然的显示名称，可以用于消息。

就像在许多编程语言中一样，关键字是标识符的一个子集。为了将令牌分类为关键字，我们需要一个关键字过滤器，该过滤器检查找到的标识符是否确实是关键字。这与 C 或 C++ 中的行为相同，其中关键字也是标识符的一个子集。编程语言会不断发展，并且可能会引入新的关键字。例如，原始的 K&R C 语言没有使用 `enum` 关键字定义枚举。因此，应该有一个标志来指示关键字的语言级别。

我们收集了几个信息片段，所有这些信息都属于 `TokenKind` 枚举的一个成员：枚举成员的标签、运算符的拼写和关键字标志。对于诊断消息，我们集中存储信息在一个名为 `include/tinylang/Basic/TokenKinds.def` 的 `.def` 文件中，其外观如下。需要注意的是，关键字前缀为 `kw_`：

```cpp

#ifndef TOK
#define TOK(ID)
#endif
#ifndef PUNCTUATOR
#define PUNCTUATOR(ID, SP) TOK(ID)
#endif
#ifndef KEYWORD
#define KEYWORD(ID, FLAG) TOK(kw_ ## ID)
#endif
TOK(unknown)
TOK(eof)
TOK(identifier)
TOK(integer_literal)
PUNCTUATOR(plus,                "+")
PUNCTUATOR(minus,               "-")
// …
KEYWORD(BEGIN                       , KEYALL)
KEYWORD(CONST                       , KEYALL)
// …
#undef KEYWORD
#undef PUNCTUATOR
#undef TOK
```

通过这些集中定义，在 `include/tinylang/Basic/TokenKinds.h` 文件中创建 `TokenKind` 枚举变得容易。再次，枚举被放入其自己的命名空间 `tok` 中：

```cpp

#ifndef TINYLANG_BASIC_TOKENKINDS_H
#define TINYLANG_BASIC_TOKENKINDS_H
namespace tinylang {
  namespace tok {
    enum TokenKind : unsigned short {
#define TOK(ID) ID,
#include "TokenKinds.def"
      NUM_TOKENS
    };
```

填充数组的模式现在应该已经熟悉了。`TOK` 宏被定义为仅返回 `ID`。作为一个有用的补充，我们还定义了 `NUM_TOKENS` 作为枚举的最后一个成员，它表示定义的标记数量：

```cpp

    const char *getTokenName(TokenKind Kind);
    const char *getPunctuatorSpelling(TokenKind Kind);
    const char *getKeywordSpelling(TokenKind Kind);
  }
}
#endif
```

实现文件 `lib/Basic/TokenKinds.cpp` 也使用 `.def` 文件来检索名称：

```cpp

#include "tinylang/Basic/TokenKinds.h"
#include "llvm/Support/ErrorHandling.h"
using namespace tinylang;
static const char * const TokNames[] = {
#define TOK(ID) #ID,
#define KEYWORD(ID, FLAG) #ID,
#include "tinylang/Basic/TokenKinds.def"
  nullptr
};
```

标记的文本名称是从其枚举标签 `ID` 派生的。有两个特殊情况：

+   首先，我们需要定义 `TOK` 和 `KEYWORD` 宏，因为 `KEYWORD` 的默认定义没有使用 `TOK` 宏

+   第二，在数组末尾添加了一个 `nullptr` 值，以考虑添加的 `NUM_TOKENS` 枚举成员：

    ```cpp

    const char *tok::getTokenName(TokenKind Kind) {
      return TokNames[Kind];
    }
    ```

在 `getPunctuatorSpelling()` 和 `getKeywordSpelling()` 函数中，我们采取了稍微不同的方法。这些函数只为枚举的子集返回有意义的值。这可以通过 `switch` 语句实现，默认返回 `nullptr` 值：

```cpp

const char *tok::getPunctuatorSpelling(TokenKind Kind) {
  switch (Kind) {
#define PUNCTUATOR(ID, SP) case ID: return SP;
#include "tinylang/Basic/TokenKinds.def"
    default: break;
  }
  return nullptr;
}
const char *tok::getKeywordSpelling(TokenKind Kind) {
  switch (Kind) {
#define KEYWORD(ID, FLAG) case kw_ ## ID: return #ID;
#include "tinylang/Basic/TokenKinds.def"
    default: break;
  }
  return nullptr;
}
```

提示

注意宏是如何定义的，以从文件中检索必要的信息。

在上一章中，`Token` 类与 `Lexer` 类在同一个头文件中声明。为了使其更灵活，我们将 `Token` 类放入其自己的头文件 `include/Lexer/Token.h` 中。像以前一样，`Token` 存储标记开始的指针、其长度和标记类型，如之前定义的：

```cpp

class Token {
  friend class Lexer;
  const char *Ptr;
  size_t Length;
  tok::TokenKind Kind;
public:
  tok::TokenKind getKind() const { return Kind; }
  size_t getLength() const { return Length; }
```

表示消息中源位置的 `SMLoc` 实例是从标记的指针创建的：

```cpp

  SMLoc getLocation() const {
    return SMLoc::getFromPointer(Ptr);
  }
```

`getIdentifier()` 和 `getLiteralData()` 方法允许访问标识符和字面数据的标记文本。对于任何其他标记类型，没有必要访问文本，因为这由标记类型隐含：

```cpp

  StringRef getIdentifier() {
    assert(is(tok::identifier) &&
           "Cannot get identfier of non-identifier");
    return StringRef(Ptr, Length);
  }
  StringRef getLiteralData() {
    assert(isOneOf(tok::integer_literal,
                   tok::string_literal) &&
           "Cannot get literal data of non-literal");
    return StringRef(Ptr, Length);
  }
};
```

我们在 `include/Lexer/Lexer.h` 头文件中声明了 `Lexer` 类，并将实现放在 `lib/Lexer/lexer.cpp` 文件中。结构与上一章的 calc 语言相同。在这里，我们需要仔细查看两个细节：

+   首先，一些运算符具有相同的词缀 – 例如，`<` 和 `<=`。当我们查看当前字符时，如果它是 `<`，那么我们必须在决定我们找到了哪个标记之前检查下一个字符。记住，输入需要以空字节结束。因此，如果当前字符有效，则下一个字符总是可以使用的：

    ```cpp

        case '<':
          if (*(CurPtr + 1) == '=')
            formTokenWithChars(token, CurPtr + 2,
                               tok::lessequal);
          else
            formTokenWithChars(token, CurPtr + 1, tok::less);
          break;
    ```

+   另一个细节是现在有更多的关键字。我们该如何处理这个问题呢？一个简单快捷的解决方案是将关键字填充到一个散列表中，这些关键字都存储在`TokenKinds.def`文件中。这可以在`Lexer`类的实例化过程中完成。采用这种方法，也可以支持语言的不同级别，因为关键字可以通过附加的标志进行过滤。在这里，这种灵活性还不是必需的。在头文件中，关键字过滤器定义如下，使用`llvm::StringMap`实例作为散列表：

    ```cpp

    class KeywordFilter {
      llvm::StringMap<tok::TokenKind> HashTable;
      void addKeyword(StringRef Keyword,
                      tok::TokenKind TokenCode);
    public:
      void addKeywords();
    ```

    `getKeyword()`方法返回给定字符串的标记类型，或者如果字符串不表示关键字则返回默认值：

    ```cpp
      tok::TokenKind getKeyword(
          StringRef Name,
          tok::TokenKind DefaultTokenCode = tok::unknown) {
        auto Result = HashTable.find(Name);
        if (Result != HashTable.end())
          return Result->second;
        return DefaultTokenCode;
      }
    };
    ```

    在实现文件中，填充关键字表：

    ```cpp
    void KeywordFilter::addKeyword(StringRef Keyword,
                                   tok::TokenKind TokenCode) {
      HashTable.insert(std::make_pair(Keyword, TokenCode));
    }
    void KeywordFilter::addKeywords() {
    #define KEYWORD(NAME, FLAGS)                               \
      addKeyword(StringRef(#NAME), tok::kw_##NAME);
    #include "tinylang/Basic/TokenKinds.def"
    }
    ```

通过你刚刚学到的技术，编写一个高效的词法分析器类并不困难。由于编译速度很重要，许多编译器使用手写的词法分析器，其中一个是 clang。

# 构建递归下降解析器。

如前一章所示，解析器是从语法派生出来的。让我们回顾一下所有的*构建规则*。对于语法的每个规则，你创建一个以规则左侧的非终结符命名的方法来解析规则的右侧。遵循右侧的定义，你做以下操作：

+   对于每个非终结符，调用相应的对应方法。

+   每个标记都被消耗。

+   对于可选或重复的组，通过查看前瞻标记（下一个未消耗的标记）来决定继续的位置。

让我们将这些构建规则应用到以下语法规则中：

```cpp

ifStatement
  : "IF" expression "THEN" statementSequence
    ( "ELSE" statementSequence )? "END" ;
```

我们可以轻松地将它转换为以下 C++方法：

```cpp

void Parser::parseIfStatement() {
  consume(tok::kw_IF);
  parseExpression();
  consume(tok::kw_THEN);
  parseStatementSequence();
  if (Tok.is(tok::kw_ELSE)) {
    advance();
    parseStatementSequence();
  }
  consume(tok::kw_END);
}
```

可以将`tinylang`的整个语法以这种方式转换为 C++。一般来说，你必须小心避免一些陷阱，因为你在互联网上找到的大多数语法都不适合这种构建。

语法和解析器。

有两种不同的解析器类型：自顶向下的解析器和自底向上的解析器。它们的名称来源于解析过程中处理规则顺序。解析器的输入是由词法分析器生成的标记序列。

自顶向下的解析器会扩展规则中的最左边的符号，直到匹配到一个标记。如果所有标记都被消耗并且所有符号都被扩展，解析就成功了。这正是 tinylang 解析器的工作方式。

自底向上的解析器做的是相反的事情：它查看标记序列，并尝试用语法的符号替换标记。例如，如果下一个标记是`IF`、`3`、`+`和`4`，那么自底向上的解析器将`3 + 4`标记替换为`expression`符号，从而得到`IF` `expression`序列。当看到属于`IF`语句的所有标记时，这个标记和符号序列就被替换为`ifStatement`符号。

解析成功是指所有标记都被消耗，并且剩下的唯一符号是起始符号。虽然自顶向下解析器可以很容易地手工构建，但对于自底向上解析器来说并非如此。

通过首先扩展哪些符号来描述这两种类型的解析器是另一种方法。两者都是从左到右读取输入，但自顶向下解析器首先扩展最左边的符号，而自底向上解析器首先扩展最右边的符号。因此，自顶向下解析器也被称为 LL 解析器，而自底向上解析器被称为 LR 解析器。

语法必须具有某些属性，以便从中导出 LL 或 LR 解析器。这些语法相应地命名：你需要一个 LL 语法来构建一个 LL 解析器。

你可以在关于编译器构造的大学教科书中找到更多详细信息，例如 Wilhelm, Seidl, 和 Hack 的 *Compiler Design. Syntactic and Semantic Analysis*，Springer 2013，以及 Grune 和 Jacobs 的 *Parsing Techniques, A practical guide*，Springer 2008。

一个需要关注的问题是左递归规则。如果一个规则的右侧以左侧相同的终结符开始，则该规则被称为**左递归**。一个典型的例子可以在表达式语法中找到：

```cpp

expression : expression "+" term ;
```

如果语法本身没有明确，那么将其翻译成 C++ 就会使这种无限递归的结果变得明显：

```cpp

Void Parser::parseExpression() {
  parseExpression();
  consume(tok::plus);
  parseTerm();
}
```

左递归也可以间接发生并涉及更多规则，这要难于发现得多。这就是为什么存在一个算法可以检测并消除左递归。

注意

左递归规则仅是 LL 解析器的问题，例如 `tinylang` 的递归下降解析器。原因是这些解析器首先扩展最左边的符号。相比之下，如果你使用解析器生成器生成一个 LR 解析器，它首先扩展最右边的符号，那么你应该避免右递归规则。

在每一步中，解析器通过仅使用前瞻标记来决定如何继续。如果这个决定不能确定性地做出，则语法存在冲突。为了说明这一点，看看 C# 中的 `using` 语句。像 C++ 一样，`using` 语句可以用来在命名空间中使一个符号可见，例如在 `using Math;` 中。也可以使用 `using M = Math;` 定义导入符号的别名。在语法中，这可以表示如下：

```cpp

usingStmt : "using" (ident "=")? ident ";"
```

这里有一个问题：在解析器消耗了 `using` 关键字之后，前瞻标记是 `ident`。然而，这些信息不足以让我们决定是否必须跳过或解析可选组。如果可选组可以开始的标记集与可选组后面的标记集重叠，这种情况总会出现。

让我们用备选方案而不是可选组重写规则：

```cpp

usingStmt : "using" ( ident "=" ident | ident ) ";" ;
```

现在，存在一个不同的冲突：两个备选方案以相同的标记开始。仅从前瞻标记来看，解析器无法决定哪个备选方案是正确的。

这些冲突非常常见。因此，了解如何处理它们是很好的。一种方法是将语法重写，使得冲突消失。在先前的例子中，两种选择都以相同的令牌开始。这可以被提取出来，得到以下规则：

```cpp

usingStmt : "using" ident ("=" ident)? ";" ;
```

这种表述没有冲突，但应该注意的是，它表达性较差。在另外两种表述中，很明显哪个`ident`是别名，哪个`ident`是命名空间名。在无冲突的规则中，最左边的`ident`改变其角色。首先，它是命名空间名，但如果后面跟着一个等号，那么它就变成了别名。

第二种方法是添加一个谓词来区分这两种情况。这个谓词通常被称为`Token &peek(int n)`，它返回当前前瞻令牌之后的第*n*个令牌。在这里，等号的存在可以用作决策中的附加谓词：

```cpp

if (Tok.is(tok::ident) && Lex.peek(0).is(tok::equal)) {
  advance();
  consume(tok::equal);
}
consume(tok::ident);
```

第三种方法是使用回溯。为此，你需要保存当前状态。然后，你必须尝试解析冲突组。如果这没有成功，那么你需要回到保存的状态并尝试其他路径。在这里，你正在寻找可以应用的正确规则，这不如其他方法高效。因此，你应该只将这种方法作为最后的手段。

现在，让我们结合错误恢复。在上一章中，我介绍了所谓的*恐慌模式*作为错误恢复的技术。基本思想是跳过令牌，直到找到一个适合继续解析的令牌。例如，在`tinylang`中，一个语句后面跟着一个分号（`;`）。

如果`IF`语句中存在语法问题，那么你会跳过所有令牌，直到找到一个分号。然后，你继续执行下一个语句。而不是使用针对令牌集的特定定义，使用系统性的方法会更好。

对于每个非终结符，你计算可以跟随非终结符的令牌集合（称为`;`、`ELSE`和`END`令牌可以跟随。因此，你必须在`parseStatement()`的错误恢复部分使用这个集合。这种方法假设语法错误可以在本地处理。通常情况下，这是不可能的。因为解析器会跳过令牌，所以可能会跳过很多令牌，直到达到输入的末尾。在这个点上，局部恢复是不可能的。

为了防止出现无意义的错误信息，调用方法需要被告知错误恢复尚未完成。这可以通过`bool`来实现。如果它返回`true`，这意味着错误恢复尚未完成，而`false`表示解析（包括可能的错误恢复）已成功。

有许多方法可以扩展这个错误恢复方案。使用活动调用者的`FOLLOW`集合是一种流行的方法。作为一个简单的例子，假设`parseStatement()`被`parseStatementSequence()`调用，而`parseStatementSequence()`本身又被`parseBlock()`和`parseModule()`调用。

在这里，每个相应的非终结符都有一个`FOLLOW`集合。如果解析器在`parseStatement()`中检测到语法错误，则跳过标记直到标记至少属于活动调用者的一个`FOLLOW`集合。如果标记在语句的`FOLLOW`集合中，则错误被局部恢复，并返回一个`false`值给调用者。否则，返回一个`true`值，意味着错误恢复必须继续。为此扩展的一个可能的实现策略是将`std::bitset`或`std::tuple`传递给被调用者，以表示当前`FOLLOW`集合的并集。

最后一个问题仍然悬而未决：我们如何调用错误恢复？在前一章中，使用了`goto`跳转到错误恢复块。这虽然可行，但不是一个令人满意的解决方案。根据我们之前讨论的内容，我们可以通过一个单独的方法跳过标记。Clang 有一个名为`skipUntil()`的方法用于此目的；我们也为`tinylang`使用了这个方法。

因为下一步是向解析器添加语义动作，所以如果需要的话，有一个中央位置来放置清理代码也会很方便。嵌套函数对于这个目的来说非常理想。C++没有嵌套函数。相反，Lambda 函数可以起到类似的作用。当我们最初查看`parseIfStatement()`方法时，添加了完整的错误恢复代码后，它看起来如下所示：

```cpp

bool Parser::parseIfStatement() {
  auto _errorhandler = [this] {
    return skipUntil(tok::semi, tok::kw_ELSE, tok::kw_END);
  };
  if (consume(tok::kw_IF))
    return _errorhandler();
  if (parseExpression(E))
    return _errorhandler();
  if (consume(tok::kw_THEN))
    return _errorhandler();
  if (parseStatementSequence(IfStmts))
    return _errorhandler();
  if (Tok.is(tok::kw_ELSE)) {
    advance();
    if (parseStatementSequence(ElseStmts))
      return _errorhandler();
  }
  if (expect(tok::kw_END))
    return _errorhandler();
  return false;
}
```

解析器和词法分析器生成器

手动构建解析器和词法分析器可能是一项繁琐的任务，尤其是当你试图发明一种新的编程语言并且经常更改语法时。幸运的是，一些工具可以自动化这项任务。

经典的 Linux 工具是**flex**（[`github.com/westes/flex`](https://github.com/westes/flex)）和**bison**（[`www.gnu.org/software/bison/`](https://www.gnu.org/software/bison/)）。flex 从一组正则表达式生成词法分析器，而 bison 从语法描述生成**LALR(1)**解析器。这两个工具都生成 C/C+源代码，并且可以一起使用。

另一个流行的工具是**AntLR**（https://www.antlr.org/）。AntLR 可以从语法描述中生成一个词法分析器、一个解析器和 AST。生成的解析器属于**LL(*)**类别，这意味着它是一个自顶向下的解析器，使用可变数量的前瞻来解决冲突。这个工具是用 Java 编写的，但可以生成许多流行语言的源代码，包括 C/C++。

所有这些工具都需要一些库支持。如果你正在寻找一个可以生成自包含的词法分析器和解析器的工具，那么**Coco/R**（[`ssw.jku.at/Research/Projects/Coco/`](https://ssw.jku.at/Research/Projects/Coco/))可能就是你要找的工具。Coco/R 可以从**LL(1)**语法描述生成一个词法分析器和递归下降解析器，类似于本书中使用的那个。生成的文件基于一个模板文件，如果需要可以更改。这个工具是用 C#编写的，但可以移植到 C++、Java 和其他语言。

有许多其他工具可供选择，它们在支持的特性和输出语言方面差异很大。当然，在选择工具时，也需要考虑权衡。例如，bison 这样的 LALR(1)解析器生成器可以消费广泛的语法，你可以在互联网上找到的免费语法通常都是 LALR(1)语法。

作为缺点，这些生成器生成的状态机需要在运行时进行解释，这可能会比递归下降解析器慢。错误处理也更复杂。bison 有处理语法错误的基本支持，但正确使用需要深入理解解析器的工作原理。相比之下，AntLR 消耗的语法类略小，但可以自动生成错误处理，还可以生成 AST。因此，重写语法以便与 AntLR 一起使用可能会加快后续的开发速度。

# 执行语义分析

我们在上一节中构建的解析器只检查输入的语法。下一步是添加执行语义分析的能力。在上一章的 calc 示例中，解析器构建了一个 AST。在单独的阶段，语义分析器处理这个树。这种方法始终可以使用。在本节中，我们将使用一种稍微不同的方法，并将解析器和语义分析器更紧密地交织在一起。

语义分析器需要做什么？让我们先看看：

+   对于每一次声明，必须检查变量、对象等的名称，以确保它们没有在其他地方声明过。

+   对于表达式或语句中名称的每一次出现，都必须检查该名称是否已声明，以及所需的使用是否符合声明。

+   对于每个表达式，必须计算其结果类型。还必须计算表达式是否为常量，如果是，它具有哪个值。

+   对于赋值和参数传递，我们必须检查类型是否兼容。此外，我们还必须检查`IF`和`WHILE`语句中的条件是否为`BOOLEAN`类型。

对于这样一个小子集的编程语言来说，已经有很多需要检查的内容了！

## 处理名称的作用域

首先看看名称的作用域。名称的作用域是名称可见的范围。像 C 语言一样，`tinylang`使用声明先于使用模型。例如，`B`和`X`变量在模块级别声明为`INTEGER`类型：

```cpp

VAR B, X: INTEGER;
```

在声明之前，变量是未知的，不能使用。只有在声明之后才能这样做。在过程内部，可以声明更多变量：

```cpp

PROCEDURE Proc;
VAR B: BOOLEAN;
BEGIN
  (* Statements *)
END Proc;
```

在程序内部，在注释所在的位置，对`B`的使用指的是局部变量`B`，而对`X`的使用指的是全局变量`X`。局部变量`B`的作用域是`Proc`。如果当前作用域中找不到名称，则搜索将继续在封装作用域中进行。因此，可以在程序内部使用`X`变量。在`tinylang`中，只有模块和程序会打开新的作用域。其他语言结构，如结构体和类，通常也会打开作用域。预定义实体，如`INTEGER`类型和`TRUE`字面量，是在全局作用域中声明的，包围着模块的作用域。

在`tinylang`中，只有名称是关键的。因此，作用域可以作为一个从名称到其声明的映射来实现。只有当新名称不存在时，才能插入新名称。对于查找，还必须知道封装或父作用域。接口（在`include/tinylang/Sema/Scope.h`文件中）如下所示：

```cpp

#ifndef TINYLANG_SEMA_SCOPE_H
#define TINYLANG_SEMA_SCOPE_H
#include "tinylang/Basic/LLVM.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
namespace tinylang {
class Decl;
class Scope {
  Scope *Parent;
  StringMap<Decl *> Symbols;
public:
  Scope(Scope *Parent = nullptr) : Parent(Parent) {}
  bool insert(Decl *Declaration);
  Decl *lookup(StringRef Name);
  Scope *getParent() { return Parent; }
};
} // namespace tinylang
#endif
```

`lib/Sema/Scope.cpp`文件中的实现如下：

```cpp

#include "tinylang/Sema/Scope.h"
#include "tinylang/AST/AST.h"
using namespace tinylang;
bool Scope::insert(Decl *Declaration) {
  return Symbols
      .insert(std::pair<StringRef, Decl *>(
          Declaration->getName(), Declaration))
      .second;
}
```

请注意，`StringMap::insert()`方法不会覆盖现有条目。结果`std::pair`的`second`成员指示是否更新了表。此信息返回给调用者。

为了实现符号声明的搜索，`lookup()`方法在当前作用域中搜索，如果没有找到，则搜索通过`parent`成员链接的作用域：

```cpp

Decl *Scope::lookup(StringRef Name) {
  Scope *S = this;
  while (S) {
    StringMap<Decl *>::const_iterator I =
        S->Symbols.find(Name);
    if (I != S->Symbols.end())
      return I->second;
    S = S->getParent();
  }
  return nullptr;
}
```

然后按照以下方式处理变量声明：

+   当前作用域是模块作用域。

+   查找`INTEGER`类型声明。如果没有找到声明或它不是一个类型声明，则这是一个错误。

+   实例化一个新的 AST 节点，名为`VariableDeclaration`，其中重要的属性是名称`B`和类型。

+   将名称`B`插入到当前作用域中，映射到声明实例。如果该名称已经在作用域中，则这是一个错误。在这种情况下，当前作用域的内容不会改变。

+   对于`X`变量也执行同样的操作。

这里执行了两个任务。与 calc 示例一样，构建了 AST 节点。同时，计算了节点的属性，如类型。为什么这是可能的？

语义分析器可以回退到两组不同的属性集。作用域是从调用者继承的。类型声明可以通过评估类型声明的名称来计算（或合成）。语言被设计成这样的方式，这两组属性足以计算 AST 节点的所有属性。

一个重要的方面是*声明先于使用*模型。如果一个语言允许在声明之前使用名称，例如 C++中的类成员，那么就无法一次性计算 AST 节点的所有属性。在这种情况下，必须使用仅部分计算属性或仅使用普通信息（如 calc 示例）来构建 AST 节点。

然后，AST 必须被访问一次或多次以确定缺失的信息。在`tinylang`（和 Modula-2）的情况下，可能不需要 AST 构造——AST 是通过`parseXXX()`方法的调用层次间接表示的。从 AST 生成代码更为常见，所以我们在这里也构建了一个 AST。

在我们将这些部分组合在一起之前，我们需要了解 LLVM 使用**运行时类型信息**（**RTTI**）的风格。

## 使用 LLVM 风格的 RTTI 对 AST 进行操作

自然地，AST 节点是类层次结构的一部分。声明总是有一个名称。其他属性取决于正在声明的对象。如果一个变量被声明，则需要一个类型。常量声明需要一个类型、一个值等等。当然，在运行时，你需要找出你正在处理哪种类型的声明。可以使用`dynamic_cast<>` C++运算符来完成这个任务。问题是，所需的 RTTI 仅在 C++类附加了虚拟表时才可用——也就是说，它使用了虚拟函数。另一个缺点是 C++ RTTI 很庞大。为了避免这些缺点，LLVM 开发者引入了一种自制的 RTTI 风格，该风格被用于整个 LLVM 库中。

我们层次结构的（抽象）基类是`Decl`。为了实现 LLVM 风格的 RTTI，必须添加一个包含每个子类标签的公共枚举。还需要这个类型的私有成员和一个公共获取器。私有成员通常称为`Kind`。在我们的情况下，它看起来如下：

```cpp

class Decl {
public:
  enum DeclKind { DK_Module, DK_Const, DK_Type,
                  DK_Var, DK_Param, DK_Proc };
private:
  const DeclKind Kind;
public:
  DeclKind getKind() const { return Kind; }
};
```

每个子类现在需要一个特殊的功能成员，称为`classof`。这个函数的目的是确定给定的实例是否为请求的类型。对于`VariableDeclaration`，它的实现如下：

```cpp

static bool classof(const Decl *D) {
  return D->getKind() == DK_Var;
}
```

现在，你可以使用特殊的模板，`llvm::isa<>`，来检查一个对象是否为请求的类型，以及`llvm::dyn_cast<>`来动态转换对象。更多模板存在，但这两个是最常用的。对于其他模板，请参阅[`llvm.org/docs/ProgrammersManual.html#the-isa-cast-and-dyn-cast-templates`](https://llvm.org/docs/ProgrammersManual.html#the-isa-cast-and-dyn-cast-templates)，以及更多关于 LLVM 风格的详细信息，包括更高级的使用，请参阅[`llvm.org/docs/HowToSetUpLLVMStyleRTTI.html`](https://llvm.org/docs/HowToSetUpLLVMStyleRTTI.html)。

## 创建语义分析器

带着这些知识，我们现在可以实施所有部分。首先，我们必须在`include/llvm/tinylang/AST/AST.h`文件中创建 AST 节点变量的定义。除了支持 LLVM 风格的 RTTI 之外，基类存储了声明的名称、名称的位置以及指向封装声明的指针。后者在嵌套过程的代码生成期间是必需的。`Decl`基类声明如下：

```cpp

class Decl {
public:
  enum DeclKind { DK_Module, DK_Const, DK_Type,
                  DK_Var, DK_Param, DK_Proc };
private:
  const DeclKind Kind;
protected:
  Decl *EnclosingDecL;
  SMLoc Loc;
  StringRef Name;
public:
  Decl(DeclKind Kind, Decl *EnclosingDecL, SMLoc Loc,
       StringRef Name)
      : Kind(Kind), EnclosingDecL(EnclosingDecL), Loc(Loc),
        Name(Name) {}
  DeclKind getKind() const { return Kind; }
  SMLoc getLocation() { return Loc; }
  StringRef getName() { return Name; }
  Decl *getEnclosingDecl() { return EnclosingDecL; }
};
```

变量的声明仅添加一个指向类型声明的指针：

```cpp

class TypeDeclaration;
class VariableDeclaration : public Decl {
  TypeDeclaration *Ty;
public:
  VariableDeclaration(Decl *EnclosingDecL, SMLoc Loc,
                      StringRef Name, TypeDeclaration *Ty)
      : Decl(DK_Var, EnclosingDecL, Loc, Name), Ty(Ty) {}
  TypeDeclaration *getType() { return Ty; }
  static bool classof(const Decl *D) {
    return D->getKind() == DK_Var;
  }
};
```

解析器中的方法需要扩展语义动作和收集信息的变量：

```cpp

bool Parser::parseVariableDeclaration(DeclList &Decls) {
  auto _errorhandler = [this] {
    while (!Tok.is(tok::semi)) {
      advance();
      if (Tok.is(tok::eof)) return true;
    }
    return false;
  };
  Decl *D = nullptr; IdentList Ids;
  if (parseIdentList(Ids)) return _errorhandler();
  if (consume(tok::colon)) return _errorhandler();
  if (parseQualident(D)) return _errorhandler();
  Actions.actOnVariableDeclaration(Decls, Ids, D);
  return false;
}
```

`DeclList` 是一个声明列表，`std::vector<Decl*>`，而 `IdentList` 是一个位置和标识符列表，`std::vector<std::pair<SMLoc, StringRef>>`。

`parseQualident()` 方法返回一个声明，在这种情况下，预期是一个类型声明。

解析器类知道语义分析器类的实例，`Sema`，它存储在 `Actions` 成员中。对 `actOnVariableDeclaration()` 的调用运行语义分析器和 AST 构建过程。实现位于 `lib/Sema/Sema.cpp` 文件中：

```cpp

void Sema::actOnVariableDeclaration(DeclList &Decls,
                                    IdentList &Ids,
                                    Decl *D) {
  if (TypeDeclaration *Ty = dyn_cast<TypeDeclaration>(D)) {
    for (auto &[Loc, Name] : Ids) {
      auto *Decl = new VariableDeclaration(CurrentDecl, Loc,
                                           Name, Ty);
      if (CurrentScope->insert(Decl))
        Decls.push_back(Decl);
      else
        Diags.report(Loc, diag::err_symbold_declared, Name);
    }
  } else if (!Ids.empty()) {
    SMLoc Loc = Ids.front().first;
    Diags.report(Loc, diag::err_vardecl_requires_type);
  }
}
```

使用 `llvm::dyn_cast<TypeDeclaration>` 检查类型声明。如果不是类型声明，则打印错误消息。否则，对于 `Ids` 列表中的每个名称，实例化 `VariableDeclaration` 并将其添加到声明列表中。如果由于名称已声明而无法将变量添加到当前作用域，则也会打印错误消息。

大多数其他实体以相同的方式构建——语义分析复杂性是唯一的不同之处。对于模块和过程，需要做更多的工作，因为它们打开了一个新的作用域。打开新的作用域很简单：只需实例化一个新的 `Scope` 对象。一旦模块或过程被解析，就必须删除该作用域。

这必须可靠地完成，因为我们不希望在语法错误的情况下将名称添加到错误的作用域中。这是 C++ 中 **资源获取即初始化（RAII**） 习语的经典用法。另一个复杂之处在于，一个过程可以递归地调用自身。因此，在可以使用之前，必须将过程的名称添加到当前作用域中。语义分析器有两种方法来进入和退出作用域。作用域与一个声明相关联：

```cpp

void Sema::enterScope(Decl *D) {
  CurrentScope = new Scope(CurrentScope);
  CurrentDecl = D;
}
void Sema::leaveScope() {
  Scope *Parent = CurrentScope->getParent();
  delete CurrentScope;
  CurrentScope = Parent;
  CurrentDecl = CurrentDecl->getEnclosingDecl();
}
```

使用一个简单的辅助类来实现资源获取即初始化（RAII）习语：

```cpp

class EnterDeclScope {
  Sema &Semantics;
public:
  EnterDeclScope(Sema &Semantics, Decl *D)
      : Semantics(Semantics) {
    Semantics.enterScope(D);
  }
  ~EnterDeclScope() { Semantics.leaveScope(); }
};
```

在解析模块或过程时，与语义分析器发生两次交互。第一次是在名称解析之后。在这里，构建了一个（几乎为空的）抽象语法树（AST）节点，并建立了一个新的作用域：

```cpp

bool Parser::parseProcedureDeclaration(/* … */) {
  /* … */
  if (consume(tok::kw_PROCEDURE)) return _errorhandler();
  if (expect(tok::identifier)) return _errorhandler();
  ProcedureDeclaration *D =
      Actions.actOnProcedureDeclaration(
          Tok.getLocation(), Tok.getIdentifier());
  EnterDeclScope S(Actions, D);
  /* … */
}
```

语义分析器检查当前作用域中的名称，并返回 AST 节点：

```cpp

ProcedureDeclaration *
Sema::actOnProcedureDeclaration(SMLoc Loc, StringRef Name) {
  ProcedureDeclaration *P =
      new ProcedureDeclaration(CurrentDecl, Loc, Name);
  if (!CurrentScope->insert(P))
    Diags.report(Loc, diag::err_symbold_declared, Name);
  return P;
}
```

在解析所有声明和过程体之后，实际的工作才完成。您只需检查过程声明末尾的名称是否等于过程的名称，以及用于返回类型的声明是否是类型声明：

```cpp

void Sema::actOnProcedureDeclaration(
    ProcedureDeclaration *ProcDecl, SMLoc Loc,
    StringRef Name, FormalParamList &Params, Decl *RetType,
    DeclList &Decls, StmtList &Stmts) {
  if (Name != ProcDecl->getName()) {
    Diags.report(Loc, diag::err_proc_identifier_not_equal);
    Diags.report(ProcDecl->getLocation(),
                 diag::note_proc_identifier_declaration);
  }
  ProcDecl->setDecls(Decls);
  ProcDecl->setStmts(Stmts);
  auto *RetTypeDecl =
      dyn_cast_or_null<TypeDeclaration>(RetType);
  if (!RetTypeDecl && RetType)
    Diags.report(Loc, diag::err_returntype_must_be_type,
                 Name);
  else
    ProcDecl->setRetType(RetTypeDecl);
}
```

一些声明固有的存在，不能由开发者定义。这包括 `BOOLEAN` 和 `INTEGER` 类型以及 `TRUE` 和 `FALSE` 文本。这些声明存在于全局作用域中，必须通过程序添加。Modula-2 还预定义了一些过程，如 `INC` 或 `DEC`，可以添加到全局作用域中。考虑到我们的类，初始化全局作用域很简单：

```cpp

void Sema::initialize() {
  CurrentScope = new Scope();
  CurrentDecl = nullptr;
  IntegerType =
      new TypeDeclaration(CurrentDecl, SMLoc(), "INTEGER");
  BooleanType =
      new TypeDeclaration(CurrentDecl, SMLoc(), "BOOLEAN");
  TrueLiteral = new BooleanLiteral(true, BooleanType);
  FalseLiteral = new BooleanLiteral(false, BooleanType);
  TrueConst = new ConstantDeclaration(CurrentDecl, SMLoc(),
                                      "TRUE", TrueLiteral);
  FalseConst = new ConstantDeclaration(
      CurrentDecl, SMLoc(), "FALSE", FalseLiteral);
  CurrentScope->insert(IntegerType);
  CurrentScope->insert(BooleanType);
  CurrentScope->insert(TrueConst);
  CurrentScope->insert(FalseConst);
}
```

使用此方案，可以对`tinylang`的所有必需计算进行操作。例如，让我们看看如何计算一个表达式是否得到一个常量值：

+   我们必须确保字面量或常量声明的引用是一个常量

+   如果表达式的两边都是常量，那么应用运算符也会得到一个常量

这些规则在创建表达式 AST 节点时嵌入到语义分析器中。同样，类型和常量值也可以计算。

应该注意的是，并非所有类型的计算都可以用这种方式进行。例如，为了检测未初始化变量的使用，可以使用一种称为*符号解释*的方法。在其一般形式中，该方法需要通过 AST 的特殊遍历顺序，这在构建时是不可能的。好消息是，所提出的方法创建了一个完全装饰的 AST，它已准备好用于代码生成。这个 AST 可以用于进一步分析，前提是昂贵的分析可以根据需要打开或关闭。

为了玩转前端，你还需要更新驱动程序。由于缺少代码生成，正确的`tinylang`程序不会产生输出。尽管如此，它可以用来探索错误恢复并引发语义错误：

```cpp

#include "tinylang/Basic/Diagnostic.h"
#include "tinylang/Basic/Version.h"
#include "tinylang/Parser/Parser.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"
using namespace tinylang;
int main(int argc_, const char **argv_) {
  llvm::InitLLVM X(argc_, argv_);
  llvm::SmallVector<const char *, 256> argv(argv_ + 1,
                                            argv_ + argc_);
  llvm::outs() << "Tinylang "
               << tinylang::getTinylangVersion() << "\n";
  for (const char *F : argv) {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
        FileOrErr = llvm::MemoryBuffer::getFile(F);
    if (std::error_code BufferError =
            FileOrErr.getError()) {
      llvm::errs() << "Error reading " << F << ": "
                   << BufferError.message() << "\n";
      continue;
    }
    llvm::SourceMgr SrcMgr;
    DiagnosticsEngine Diags(SrcMgr);
    SrcMgr.AddNewSourceBuffer(std::move(*FileOrErr),
                              llvm::SMLoc());
    auto TheLexer = Lexer(SrcMgr, Diags);
    auto TheSema = Sema(Diags);
    auto TheParser = Parser(TheLexer, TheSema);
    TheParser.parse();
  }
}
```

恭喜！你已经完成了`tinylang`的前端实现！你可以使用*定义真实编程语言*部分提供的示例程序`Gcd.mod`来运行前端：

```cpp

$ tinylang Gcd.mod
```

当然，这是一个有效的程序，看起来好像没有发生任何事情。务必修改文件并引发一些错误消息。我们将在下一章中继续添加代码生成，继续这项有趣的旅程。

# 摘要

在本章中，你学习了现实世界编译器在前端使用的技巧。从项目布局开始，你创建了用于词法分析器、解析器和语义分析器的单独库。为了向用户输出消息，你扩展了一个现有的 LLVM 类，允许消息集中存储。词法分析器现在被分割成几个接口。

然后，你学习了如何从语法描述中构建递归下降解析器，了解了要避免的陷阱，并学习了如何使用生成器来完成这项工作。你构建的语义分析器在解析器和 AST 构建过程中执行了语言所需的所有语义检查。

你的编码努力的结果是一个完全装饰的 AST。你将在下一章中使用它来生成 IR 代码，最终生成目标代码。
