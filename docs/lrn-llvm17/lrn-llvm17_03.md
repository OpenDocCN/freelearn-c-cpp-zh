

# 第二章：编译器的结构

编译技术是计算机科学中一个研究得很好的领域。高级任务是将源语言翻译成机器代码。通常，这个任务被分为三个部分，**前端**、**中间端**和**后端**。前端主要处理源语言，而中间端执行转换以改进代码，后端负责生成机器代码。由于 LLVM 核心库提供了中间端和后端，因此在本章中我们将重点关注前端。

在本章中，你将涵盖以下部分和主题：

+   *编译器的构建块*，其中你将了解在编译器中通常可以找到的组件

+   *算术表达式语言*，将介绍一个示例语言并展示如何使用语法来定义语言

+   *词法分析*，讨论如何为语言实现一个词法分析器

+   *语法分析*，涵盖了从语法构建解析器的构造

+   *语义分析*，其中你将了解如何实现语义检查

+   *使用 LLVM 后端进行代码生成*，讨论如何与 LLVM 后端接口并将所有前面的阶段粘合在一起以创建一个完整的编译器

# 编译器的构建块

自从计算机变得可用以来，已经开发了数千种编程语言。结果证明，所有编译器都必须解决相同的问题，并且编译器的实现最好是根据这些任务来结构化。在高级别上，有三个组件。前端将源代码转换为**中间表示**（**IR**）。然后中间端对 IR 进行转换，目的是提高性能或减少代码的大小。最后，后端从 IR 生成机器代码。LLVM 核心库提供了由非常复杂的转换和所有流行平台的后端组成的中间端。此外，LLVM 核心库还定义了一个中间表示，用作中间端和后端的输入。这种设计的好处是，你只需要关注你想要实现的编程语言的前端。

前端输入是源代码，通常是文本文件。为了理解它，前端首先识别语言的单词，如数字和标识符，通常称为标记。这一步由**词法分析器**执行。接下来，分析由标记形成的句法结构。所谓的**解析器**执行这一步，结果是**抽象语法树**（**AST**）。最后，前端需要检查编程语言的规则是否被遵守，这是通过**语义分析器**完成的。如果没有检测到错误，那么 AST 将转换为 IR 并传递给中间端。

在接下来的章节中，我们将构建一个表达式语言的编译器，它将输入转换为 LLVM IR。然后，代表后端的 LLVM `llc` 静态编译器可以用来将 IR 编译成目标代码。一切始于定义语言。请记住，本章中所有文件的 C++ 实现都将包含在一个名为 `src/` 的目录中。

# 算术表达式语言

算术表达式是每种编程语言的一部分。以下是一个名为 **calc** 的算术表达式计算语言的示例。calc 表达式被编译成一个应用程序，该应用程序评估以下表达式：

```cpp

with a, b: a * (4 + b)
```

表达式中所使用的变量必须用关键字 `with` 声明。这个程序被编译成一个应用程序，该应用程序会询问用户 `a` 和 `b` 变量的值，并打印结果。

示例总是受欢迎的，但作为一个编译器编写者，你需要比这更详尽的规范来进行实现和测试。编程语言语法的载体是语法。

## 编程语言语法的形式化规范

语言元素，例如，关键字、标识符、字符串、数字和运算符，被称为**标记**。在这个意义上，程序是一系列标记的序列，而语法指定了哪些序列是有效的。

通常，语法是用**扩展的巴科斯-诺尔范式**（**EBNF**）编写的。语法规则有一个左侧和一个右侧。左侧只是一个称为**非终结符**的单个符号。规则的右侧由非终结符、标记和用于选择和重复的元符号组成。让我们看看 calc 语言的语法：

```cpp

calc : ("with" ident ("," ident)* ":")? expr ;
expr : term (( "+" | "-" ) term)* ;
term : factor (( "*" | "/") factor)* ;
factor : ident | number | "(" expr ")" ;
ident : ([a-zAZ])+ ;
number : ([0-9])+ ;
```

在第一行中，`calc` 是一个非终结符。除非另有说明，否则语法中的第一个非终结符是起始符号。冒号 (`:`) 是规则左右两侧的分隔符。在这里，`"with"`、`,` 和 `":"` 是代表这个字符串的标记。括号用于分组。一个分组可以是可选的或可重复的。在闭括号后面的问号 (`?`) 表示一个可选分组。星号 `*` 表示零个或多个重复，而加号 `+` 表示一个或多个重复。`Ident` 和 `expr` 是非终结符。对于它们中的每一个，都存在另一个规则。分号 (`;`) 标记规则的结束。在第二行中，竖线 `|` 表示选择。最后，在最后两行中，方括号 `[ ]` 表示字符类。有效的字符写在方括号内。例如，字符类 `[a-zA-Z]` 匹配大写或小写字母，而 `([a-zA-Z])+"` 匹配这些字母的一个或多个。这对应于正则表达式。

## 语法如何帮助编译器编写者？

这样的语法可能看起来像是一个理论玩具，但对编译器编写者来说是有价值的。首先，定义所有标记，这是创建词法分析器所需的。语法的规则可以翻译成解析器。当然，如果关于解析器是否正确工作有疑问，那么语法就作为一个好的规范。

然而，语法并没有定义编程语言的各个方面。句法的意义——即语义——也必须定义。为此也开发了形式化方法，但它们通常以纯文本形式指定，因为它们通常在语言最初引入时制定。

带着这些知识，接下来的两节将展示词法分析如何将输入转换为标记序列，以及语法是如何在 C++ 中编码以进行句法分析的。

# 词法分析

如前节示例所示，一种编程语言由许多元素组成，如关键字、标识符、数字、运算符等。词法分析器的任务是从文本输入中创建一个标记序列。calc 语言由以下标记组成：`with`、`:`、`+`、`-`、`*`、`/`、`(`、`)`，以及正则表达式 `([a-zA-Z])+"`（一个标识符）和 `([0-9])+"`（一个数字）。我们为每个标记分配一个唯一的数字，以便更容易地处理标记。

## 手写词法分析器

词法分析器的实现通常被称为 `Lexer`。让我们创建一个名为 `Lexer.h` 的头文件，并开始定义 `Token`。它以通常的头文件保护符和包含所需头文件开始：

```cpp

#ifndef LEXER_H
#define LEXER_H
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
```

`llvm::MemoryBuffer` 类提供了对包含文件内容的内存块的只读访问。在请求时，会在缓冲区末尾添加一个尾随零字符（`'\x00'`）。我们使用这个特性来读取缓冲区，而无需在每次访问时检查缓冲区的长度。`llvm::StringRef` 类封装了一个指向 C 字符串及其长度的指针。因为长度被存储，字符串不需要以零字符（`'\x00'`）结尾，就像正常的 C 字符串一样。这允许 `StringRef` 实例指向由 `MemoryBuffer` 管理的内存。

在此基础上，我们开始实现 `Lexer` 类：

1.  首先，`Token` 类包含了之前提到的唯一标记数字枚举的定义：

    ```cpp

    class Lexer;
    class Token {
      friend class Lexer;
    public:
      enum TokenKind : unsigned short {
        eoi, unknown, ident, number, comma, colon, plus,
        minus, star, slash, l_paren, r_paren, KW_with
      };
    ```

    除了为每个标记定义一个成员外，我们还添加了两个额外的值：`eoi` 和 `unknown`。`eoi` 代表 *输入结束*，当处理完输入的所有字符时返回。`unknown` 用于词法层面的错误事件，例如，`#` 不是语言的标记，因此会被映射到 `unknown`。

1.  除了枚举之外，该类还有一个 `Text` 成员，它指向标记文本的开始。它使用之前提到的 `StringRef` 类：

    ```cpp

    private:
      TokenKind Kind;
      llvm::StringRef Text;
    public:
      TokenKind getKind() const { return Kind; }
      llvm::StringRef getText() const { return Text; }
    ```

    这对于语义处理很有用，例如，对于一个标识符，知道其名称是有用的。

1.  `is()` 和 `isOneOf()` 方法用于测试标记是否属于某种类型。`isOneOf()` 方法使用变长模板，允许有可变数量的参数：

    ```cpp

      bool is(TokenKind K) const { return Kind == K; }
      bool isOneOf(TokenKind K1, TokenKind K2) const {
        return is(K1) || is(K2);
      }
      template <typename... Ts>
      bool isOneOf(TokenKind K1, TokenKind K2, Ts... Ks) const {
        return is(K1) || isOneOf(K2, Ks...);
      }
    };
    ```

1.  `Lexer` 类本身也有一个类似的简单接口，并在头文件中紧接着：

    ```cpp

    class Lexer {
      const char *BufferStart;
      const char *BufferPtr;
    public:
      Lexer(const llvm::StringRef &Buffer) {
        BufferStart = Buffer.begin();
        BufferPtr = BufferStart;
      }
      void next(Token &token);
    private:
      void formToken(Token &Result, const char *TokEnd,
                     Token::TokenKind Kind);
    };
    #endif
    ```

    除了构造函数外，公共接口只有 `next()` 方法，该方法返回下一个标记。该方法的行为像一个迭代器，总是前进到下一个可用的标记。该类唯一的成员是指向输入开始和下一个未处理字符的指针。假设缓冲区以终止的 `0` 结尾（就像 C 字符串一样）。

1.  让我们在 `Lexer.cpp` 文件中实现 `Lexer` 类。它开始于一些辅助函数来分类字符：

    ```cpp

    #include "Lexer.h"
    namespace charinfo {
    LLVM_READNONE inline bool isWhitespace(char c) {
      return c == ' ' || c == '\t' || c == '\f' ||
             c == '\v' ||
    c == '\r' || c == '\n';
    }
    LLVM_READNONE inline bool isDigit(char c) {
      return c >= '0' && c <= '9';
    }
    LLVM_READNONE inline bool isLetter(char c) {
      return (c >= 'a' && c <= 'z') ||
             (c >= 'A' && c <= 'Z');
    }
    }
    ```

    这些函数用于使条件更易读。

注意

我们没有使用 `<cctype>` 标准库头文件提供的函数有两个原因。首先，这些函数的行为取决于环境中定义的区域设置。例如，如果区域设置为德语区域，那么德语的重音符号可以被分类为字母。在编译器中这通常是不希望的。其次，由于这些函数的参数类型为 `int`，需要从 `char` 类型进行转换。这个转换的结果取决于 `char` 是否被视为有符号或无符号类型，这会导致可移植性问题。

1.  从上一节中的语法，我们知道该语言的所有标记。但是语法并没有定义应该忽略的字符。例如，空格或换行符仅添加空白，通常会被忽略。`next()` 方法开始时就会忽略这些字符：

    ```cpp

    void Lexer::next(Token &token) {
      while (*BufferPtr &&
             charinfo::isWhitespace(*BufferPtr)) {
        ++BufferPtr;
      }
    ```

1.  接下来，确保还有字符需要处理：

    ```cpp

      if (!*BufferPtr) {
        token.Kind = Token::eoi;
        return;
      }
    ```

    至少有一个字符需要处理。

1.  我们首先检查字符是否为小写或大写。在这种情况下，标记要么是一个标识符，要么是 `with` 关键字，因为标识符的正则表达式也会匹配到关键字。这里最常见的解决方案是收集正则表达式匹配到的字符并检查该字符串是否恰好是关键字：

    ```cpp

      if (charinfo::isLetter(*BufferPtr)) {
        const char *end = BufferPtr + 1;
        while (charinfo::isLetter(*end))
          ++end;
        llvm::StringRef Name(BufferPtr, end - BufferPtr);
        Token::TokenKind kind =
            Name == "with" ? Token::KW_with : Token::ident;
        formToken(token, end, kind);
        return;
      }
    ```

    `formToken()` 私有方法用于填充标记。

1.  接下来，我们检查一个数字。这段代码与前面的代码非常相似：

    ```cpp

      else if (charinfo::isDigit(*BufferPtr)) {
        const char *end = BufferPtr + 1;
        while (charinfo::isDigit(*end))
          ++end;
        formToken(token, end, Token::number);
        return;
      }
    ```

    现在只剩下由固定字符串定义的标记。

1.  这可以通过 `switch` 实现得很容易。由于所有这些标记只有一个字符，因此使用了 `CASE` 预处理器宏来减少输入：

    ```cpp

      else {
        switch (*BufferPtr) {
    #define CASE(ch, tok) \
    case ch: formToken(token, BufferPtr + 1, tok); break
    CASE('+', Token::plus);
    CASE('-', Token::minus);
    CASE('*', Token::star);
    CASE('/', Token::slash);
    CASE('(', Token::Token::l_paren);
    CASE(')', Token::Token::r_paren);
    CASE(':', Token::Token::colon);
    CASE(',', Token::Token::comma);
    #undef CASE
    ```

1.  最后，我们需要检查意外的字符：

    ```cpp

        default:
          formToken(token, BufferPtr + 1, Token::unknown);
        }
        return;
      }
    }
    ```

    只缺少 `formToken()` 私有辅助方法。

1.  它填充了 `Token` 实例的成员并更新了指向下一个未处理字符的指针：

    ```cpp

    void Lexer::formToken(Token &Tok, const char *TokEnd,
                          Token::TokenKind Kind) {
      Tok.Kind = Kind;
      Tok.Text = llvm::StringRef(BufferPtr,
                                 TokEnd - BufferPtr);
      BufferPtr = TokEnd;
    }
    ```

在下一节中，我们将探讨如何构建用于句法分析的解析器。

# 句法分析

语法分析由解析器执行，我们将实现它。这是基于前几节中的语法和词法分析器。解析过程的结果是一个称为抽象语法树（**AST**）的动态数据结构。AST 是输入的一个非常紧凑的表示，非常适合语义分析。

首先，我们将实现解析器，然后我们将查看在 AST 中的解析过程。

## 手写解析器

解析器的接口定义在头文件 `Parser.h` 中。它以一些 `include` 声明开始：

```cpp

#ifndef PARSER_H
#define PARSER_H
#include "AST.h"
#include "Lexer.h"
#include "llvm/Support/raw_ostream.h"
```

`AST.h` 头文件声明了 AST 的接口，稍后展示。LLVM 的编码指南禁止使用 `<iostream>` 库，因此包含等效 LLVM 功能的头文件。这是发出错误消息所需的：

1.  `Parser` 类首先声明了一些私有成员：

    ```cpp

    class Parser {
      Lexer &Lex;
      Token Tok;
      bool HasError;
    ```

    `Lex` 和 `Tok` 是前几节中类的实例。`Tok` 存储下一个标记（前瞻），`Lex` 用于从输入中检索下一个标记。`HasError` 标志指示是否检测到错误。

1.  几个方法处理标记：

    ```cpp

      void error() {
        llvm::errs() << "Unexpected: " << Tok.getText()
                     << "\n";
        HasError = true;
      }
      void advance() { Lex.next(Tok); }
      bool expect(Token::TokenKind Kind) {
        if (Tok.getKind() != Kind) {
          error();
          return true;
        }
        return false;
      }
      bool consume(Token::TokenKind Kind) {
        if (expect(Kind))
          return true;
        advance();
        return false;
      }
    ```

    `advance()` 从词法分析器中检索下一个标记。`expect()` 测试前瞻是否有预期的类型，如果没有，则发出错误消息。最后，`consume()` 如果前瞻有预期的类型，则检索下一个标记。如果发出错误消息，则将 `HasError` 标志设置为 true。

1.  对于语法中的每个非终结符，声明了一个解析规则的方法：

    ```cpp

      AST *parseCalc();
      Expr *parseExpr();
      Expr *parseTerm();
      Expr *parseFactor();
    ```

注意：

对于 `ident` 和 `number` 没有方法。这些规则只返回标记，并被相应的标记替换。

1.  公共接口如下。构造函数初始化所有成员并从词法分析器中检索第一个标记：

    ```cpp

    public:
      Parser(Lexer &Lex) : Lex(Lex), HasError(false) {
        advance();
      }
    ```

1.  需要一个函数来获取错误标志的值：

    ```cpp

      bool hasError() { return HasError; }
    ```

1.  最后，`parse()` 方法是解析的主要入口点：

    ```cpp

      AST *parse();
    };
    #endif
    ```

### 解析器实现

让我们深入了解解析器的实现！

1.  我们在 `Parser.cpp` 文件中的实现以 `parse()` 方法开始：

    ```cpp

    #include "Parser.h"
    AST *Parser::parse() {
      AST *Res = parseCalc();
      expect(Token::eoi);
      return Res;
    }
    ```

    `parse()` 方法的要点是整个输入已经被消耗。你还记得第一部分的解析示例添加了一个特殊符号来表示输入的结束吗？我们在这里检查它。

1.  `parseCalc()` 方法实现了相应的规则。值得仔细看看这个方法，因为其他解析方法遵循相同的模式。让我们回忆一下第一部分的规则：

    ```cpp

    calc : ("with" ident ("," ident)* ":")? expr ;
    ```

1.  方法以声明一些局部变量开始：

    ```cpp

    AST *Parser::parseCalc() {
      Expr *E;
      llvm::SmallVector<llvm::StringRef, 8> Vars;
    ```

1.  首先要做的决定是是否必须解析可选组。组以 `with` 标记开始，所以我们比较标记与此值：

    ```cpp

      if (Tok.is(Token::KW_with)) {
        advance();
    ```

1.  接下来，我们期望一个标识符：

    ```cpp

        if (expect(Token::ident))
          goto _error;
        Vars.push_back(Tok.getText());
        advance();
    ```

    如果有一个标识符，则将其保存到 `Vars` 向量中。否则，它是语法错误，将单独处理。

1.  在语法中接下来是一个重复组，它解析更多的标识符，用逗号分隔：

    ```cpp

        while (Tok.is(Token::comma)) {
          advance();
          if (expect(Token::ident))
            goto _error;
          Vars.push_back(Tok.getText());
          advance();
        }
    ```

    到现在为止，这应该不会令人惊讶。重复组以标记（`,`）开始。对标记的测试成为`while`循环的条件，实现零次或多次重复。循环内的标识符处理方式与之前相同。

1.  最后，可选组需要在末尾有一个冒号：

    ```cpp

        if (consume(Token::colon))
          goto _error;
      }
    ```

1.  最后，必须解析`expr`的规则：

    ```cpp

      E = parseExpr();
    ```

1.  通过这个调用，规则的解析成功完成。现在收集到的信息被用来创建这个规则的 AST 节点：

    ```cpp

      if (Vars.empty()) return E;
      else return new WithDecl(Vars, E);
    ```

现在只缺少错误处理。检测语法错误很容易，但从中恢复却出人意料地复杂。在这里，使用了一种称为**恐慌模式**的简单方法。

在恐慌模式下，从标记流中删除标记，直到找到一个解析器可以用来继续其工作的标记。大多数编程语言都有表示结束的符号，例如，在 C++中，有`;`（语句结束）或`}`（块结束）。这样的标记是寻找的好候选。

另一方面，错误可能是因为我们正在寻找的符号缺失。在这种情况下，解析器在继续之前可能已经删除了大量的标记。这并不像听起来那么糟糕。如今，编译器速度快更重要。一旦发生错误，开发者会查看第一条错误信息，修复它，然后重新启动编译器。这与使用穿孔卡片的情况截然不同，当时尽可能多地获取错误信息很重要，因为下一次编译器的运行可能只有第二天。

### 错误处理

而不是使用一些任意的标记来查找，这里使用另一组标记。对于每个非终结符，都有一个可以跟随该非终结符的标记集合：

1.  在`calc`的情况下，只有输入的结束符跟随这个非终结符。实现很简单：

    ```cpp

    _error:
      while (!Tok.is(Token::eoi))
        advance();
      return nullptr;
    }
    ```

1.  其他解析方法的结构类似。`parseExpr()`是`expr`规则的翻译：

    ```cpp

    Expr *Parser ::parseExpr() {
      Expr *Left = parseTerm() ;
      while (Tok.isOneOf(Token::plus, Token::minus)) {
        BinaryOp::Operator Op =
           Tok.is(Token::plus) ? BinaryOp::Plus :
                                 BinaryOp::Minus;
        advance();
        Expr *Right = parseTerm();
        Left = new BinaryOp(Op, Left, Right);
      }
      return Left;
    }
    ```

    规则内的重复组被翻译为一个`while`循环。注意`isOneOf()`方法的使用如何简化了对多个标记的检查。

1.  `term`规则的编码看起来相同：

    ```cpp

    Expr *Parser::parseTerm() {
      Expr *Left = parseFactor();
      while (Tok.isOneOf(Token::star, Token::slash)) {
        BinaryOp::Operator Op =
            Tok.is(Token::star) ? BinaryOp::Mul :
                                  BinaryOp::Div;
        advance();
        Expr *Right = parseFactor();
        Left = new BinaryOp(Op, Left, Right);
      }
      return Left;
    }
    ```

    这种方法与`parseExpr()`非常相似，你可能想将它们合并为一个。在语法中，可以有一个规则处理乘法和加法运算符。使用两个规则的优势在于，这样运算符的优先级与数学评估顺序很好地匹配。如果你将两个规则合并，那么你需要在其他地方确定评估顺序。

1.  最后，你需要实现`factor`的规则：

    ```cpp

    Expr *Parser::parseFactor() {
      Expr *Res = nullptr;
      switch (Tok.getKind()) {
      case Token::number:
        Res = new Factor(Factor::Number, Tok.getText());
        advance(); break;
    ```

    与使用一系列`if`和`else if`语句相比，这里使用`switch`语句似乎更合适，因为每个备选方案都只从单个标记开始。一般来说，你应该考虑你喜欢的翻译模式。如果你以后需要更改解析方法，那么如果每个方法没有不同的语法规则实现方式，这将是一个优点。

1.  如果你使用`switch`语句，那么错误处理发生在`default`情况下：

    ```cpp

      case Token::ident:
        Res = new Factor(Factor::Ident, Tok.getText());
        advance(); break;
      case Token::l_paren:
        advance();
        Res = parseExpr();
        if (!consume(Token::r_paren)) break;
      default:
        if (!Res) error();
    ```

    我们在这里保护发出错误信息，因为存在跌落。

1.  如果括号表达式中有语法错误，那么已经发出了错误信息。保护防止第二个错误信息：

    ```cpp

        while (!Tok.isOneOf(Token::r_paren, Token::star,
                            Token::plus, Token::minus,
                            Token::slash, Token::eoi))
          advance();
      }
      return Res;
    }
    ```

这很简单，不是吗？一旦你记住了使用的模式，根据语法规则编写解析器几乎是一项枯燥的工作。这种解析器被称为**递归下降解析器**。

并非所有语法都可以构建递归下降解析器

语法必须满足某些条件才能适合构建递归下降解析器。这类语法被称为 LL(1)。实际上，你可以在互联网上找到的大多数语法都不属于这类语法。大多数关于编译器构造理论的书籍都解释了这一点。关于这个主题的经典书籍是所谓的*龙书*，Aho、Lam、Sethi 和 Ullman 合著的《编译器：原理、技术和工具》。

## 抽象语法树

解析过程的结果是 AST。AST 是输入程序的另一种紧凑表示。它捕获了关键信息。许多编程语言都有作为分隔符但不含进一步意义的符号。例如，在 C++中，分号`;`表示单个语句的结束。当然，这个信息对于解析器来说很重要。一旦我们将语句转换为内存表示，分号就不再重要了，可以省略。

如果你查看示例表达式语言的第一条规则，那么很明显，`with`关键字、逗号（`,`）和冒号（`:`）对于程序的意义并不重要。重要的是声明的变量列表，这些变量可以用在表达式中。结果是，只需要几个类来记录信息：`Factor`保存数字或标识符，`BinaryOp`保存算术运算符和表达式的左右两侧，`WithDecl`存储声明的变量列表和表达式。`AST`和`Expr`仅用于创建一个公共类层次结构。

除了解析输入的信息外，使用`AST.h`头文件进行树遍历：

1.  它从访问者接口开始：

    ```cpp

    #ifndef AST_H
    #define AST_H
    #include "llvm/ADT/SmallVector.h"
    #include "llvm/ADT/StringRef.h"
    class AST;
    class Expr;
    class Factor;
    class BinaryOp;
    class WithDecl;
    class ASTVisitor {
    public:
      virtual void visit(AST &){};
      virtual void visit(Expr &){};
      virtual void visit(Factor &) = 0;
      virtual void visit(BinaryOp &) = 0;
      virtual void visit(WithDecl &) = 0;
    };
    ```

    访问者模式需要知道要访问的每个类。因为每个类也引用了访问者，所以我们将在文件顶部声明所有类。请注意，`AST`和`Expr`的`visit()`方法有一个默认实现，它什么都不做。

1.  `AST`类是层次结构的根：

    ```cpp

    class AST {
    public:
      virtual ~AST() {}
      virtual void accept(ASTVisitor &V) = 0;
    };
    ```

1.  类似地，`Expr`是`AST`相关表达式类的根：

    ```cpp

    class Expr : public AST {
    public:
      Expr() {}
    };
    ```

1.  `Factor`类存储一个数字或变量的名称：

    ```cpp

    class Factor : public Expr {
    public:
      enum ValueKind { Ident, Number };
    private:
      ValueKind Kind;
      llvm::StringRef Val;
    public:
      Factor(ValueKind Kind, llvm::StringRef Val)
          : Kind(Kind), Val(Val) {}
      ValueKind getKind() { return Kind; }
      llvm::StringRef getVal() { return Val; }
      virtual void accept(ASTVisitor &V) override {
        V.visit(*this);
      }
    };
    ```

    在这个例子中，数字和变量被几乎同等对待，因此我们决定只创建一个 AST 节点类来表示它们。`Kind`成员告诉我们实例代表的是哪一个情况。在更复杂的语言中，你通常希望有不同的 AST 类，例如为数字创建一个`NumberLiteral`类，为变量引用创建一个`VariableAccess`类。

1.  `BinaryOp`类包含评估表达式所需的数据：

    ```cpp

    class BinaryOp : public Expr {
    public:
      enum Operator { Plus, Minus, Mul, Div };
    private:
      Expr *Left;
      Expr *Right;
      Operator Op;
    public:
      BinaryOp(Operator Op, Expr *L, Expr *R)
          : Op(Op), Left(L), Right(R) {}
      Expr *getLeft() { return Left; }
      Expr *getRight() { return Right; }
      Operator getOperator() { return Op; }
      virtual void accept(ASTVisitor &V) override {
        V.visit(*this);
      }
    };
    ```

    与解析器不同，`BinaryOp`类在乘法和加法运算符之间没有区别。运算符的优先级在树结构中隐式可用。

1.  最后，`WithDecl`类存储声明的变量和表达式：

    ```cpp

    class WithDecl : public AST {
      using VarVector =
                       llvm::SmallVector<llvm::StringRef, 8>;
      VarVector Vars;
      Expr *E;
    public:
      WithDecl(llvm::SmallVector<llvm::StringRef, 8> Vars,
               Expr *E)
          : Vars(Vars), E(E) {}
      VarVector::const_iterator begin()
                                    { return Vars.begin(); }
      VarVector::const_iterator end() { return Vars.end(); }
      Expr *getExpr() { return E; }
      virtual void accept(ASTVisitor &V) override {
        V.visit(*this);
      }
    };
    #endif
    ```

AST 在解析过程中构建。语义分析检查树是否遵循语言的意义（例如，使用的变量必须声明），并且可能增强树。之后，树被用于代码生成。

# 语义分析

语义分析器遍历 AST 并检查语言的各个语义规则，例如，变量在使用前必须声明，或者变量在表达式中的类型必须兼容。如果语义分析器发现可以改进的情况，它还可以打印出警告。对于示例表达式语言，语义分析器必须检查每个使用的变量是否已声明，因为这正是语言的要求。一个可能的扩展（在这里没有实现）是在声明了但未使用的变量上打印警告。

语义分析器在`Sema`类中实现，由`semantic()`方法执行。以下是完整的`Sema.h`头文件：

```cpp

#ifndef SEMA_H
#define SEMA_H
#include "AST.h"
#include "Lexer.h"
class Sema {
public:
  bool semantic(AST *Tree);
};
#endif
```

实现在`Sema.cpp`文件中。有趣的部分是语义分析，它使用访问者实现。基本思想是每个声明的变量名都存储在一个集合中。在创建集合的过程中，可以检查每个名称的唯一性，稍后可以检查给定的名称是否在集合中：

```cpp

#include "Sema.h"
#include "llvm/ADT/StringSet.h"
namespace {
class DeclCheck : public ASTVisitor {
  llvm::StringSet<> Scope;
  bool HasError;
  enum ErrorType { Twice, Not };
  void error(ErrorType ET, llvm::StringRef V) {
    llvm::errs() << "Variable " << V << " "
                 << (ET == Twice ? "already" : "not")
                 << " declared\n";
    HasError = true;
  }
public:
  DeclCheck() : HasError(false) {}
  bool hasError() { return HasError; }
```

在`Parser`类中，一个标志被用来指示发生了错误。名称存储在一个名为`Scope`的集合中。在一个持有变量名的`Factor`节点上，会检查变量名是否在集合中：

```cpp

  virtual void visit(Factor &Node) override {
    if (Node.getKind() == Factor::Ident) {
      if (Scope.find(Node.getVal()) == Scope.end())
        error(Not, Node.getVal());
    }
  };
```

对于`BinaryOp`节点，除了检查两边是否存在并且被访问之外，没有其他要检查的内容：

```cpp

  virtual void visit(BinaryOp &Node) override {
    if (Node.getLeft())
      Node.getLeft()->accept(*this);
    else
      HasError = true;
    if (Node.getRight())
      Node.getRight()->accept(*this);
    else
      HasError = true;
  };
```

在`WithDecl`节点上，集合被填充，并开始遍历表达式：

```cpp

  virtual void visit(WithDecl &Node) override {
    for (auto I = Node.begin(), E = Node.end(); I != E;
         ++I) {
      if (!Scope.insert(*I).second)
        error(Twice, *I);
    }
    if (Node.getExpr())
      Node.getExpr()->accept(*this);
    else
      HasError = true;
  };
};
}
```

`semantic()`方法仅启动树遍历并返回错误标志：

```cpp

bool Sema::semantic(AST *Tree) {
  if (!Tree)
    return false;
  DeclCheck Check;
  Tree->accept(Check);
  return Check.hasError();
}
```

如果需要，这里可以做得更多。还可以在声明了但未使用的变量上打印警告。我们将其留给你作为练习来实现。如果语义分析没有错误完成，那么我们可以从 AST 生成 LLVM IR。这将在下一节中完成。

# 使用 LLVM 后端生成代码

后端的任务是从模块的 LLVM IR 生成优化的机器代码。IR 是后端接口，可以使用 C++ 接口或文本形式创建。同样，IR 是从 AST 生成的。

## LLVM IR 的文本表示

在尝试生成 LLVM IR 之前，应该清楚我们想要生成什么。对于我们的示例表达式语言，高级计划如下：

1.  询问用户每个变量的值。

1.  计算表达式的值。

1.  打印结果。

要让用户提供变量的值并打印结果，使用了两个库函数：`calc_read()` 和 `calc_write()`。对于 `with a: 3*a` 表达式，生成的 IR 如下：

1.  库函数必须像在 C 中一样声明。语法也类似于 C。函数名前的类型是返回类型。括号内的类型名是参数类型。声明可以出现在文件的任何位置：

    ```cpp

    declare i32 @calc_read(ptr)
    declare void @calc_write(i32)
    ```

1.  `calc_read()` 函数接受变量名作为参数。以下构造定义了一个常量，包含 `a` 和用作 C 中字符串终止符的空字节：

    ```cpp

    @a.str = private constant [2 x i8] c"a\00"
    ```

1.  它跟随 `main()` 函数。省略了参数名称，因为它们没有被使用。就像在 C 中一样，函数体被括号包围：

    ```cpp

    define i32 @main(i32, ptr) {
    ```

1.  每个基本块都必须有一个标签。因为这个是函数的第一个基本块，我们将其命名为 `entry`：

    ```cpp

    entry:
    ```

1.  调用 `calc_read()` 函数读取 `a` 变量的值。嵌套的 `getelemenptr` 指令执行索引计算以计算字符串常量第一个元素的指针。函数结果被赋值给未命名的 `%2` 变量。

    ```cpp

      %2 = call i32 @calc_read(ptr @a.str)
    ```

1.  接下来，变量被乘以 `3`：

    ```cpp

      %3 = mul nsw i32 3, %2
    ```

1.  通过调用 `calc_write()` 函数将结果打印到控制台：

    ```cpp

      call void @calc_write(i32 %3)
    ```

1.  最后，`main()` 函数返回 `0` 以指示执行成功：

    ```cpp

      ret i32 0
    }
    ```

LLVM IR 中的每个值都有类型，其中 `i32` 表示 32 位整数类型，`ptr` 表示指针。

注意

LLVM 的早期版本使用有类型的指针。例如，在 LLVM 中，字节的指针表示为 i8*。自 LLVM 16 以来，使用 `ptr`。

由于现在已经很清楚 IR 的样子，让我们从 AST 生成它。

## 从 AST 生成 IR

在 `CodeGen.h` 头文件中提供的接口非常小：

```cpp

#ifndef CODEGEN_H
#define CODEGEN_H
#include "AST.h"
class CodeGen
{
public:
 void compile(AST *Tree);
};
#endif
```

由于 AST 包含信息，基本思想是使用访问者遍历 AST。`CodeGen.cpp` 文件实现如下：

1.  所需的包含在文件顶部：

    ```cpp

    #include "CodeGen.h"
    #include "llvm/ADT/StringMap.h"
    #include "llvm/IR/IRBuilder.h"
    #include "llvm/IR/LLVMContext.h"
    #include "llvm/Support/raw_ostream.h"
    ```

1.  使用 LLVM 库的命名空间进行名称查找：

    ```cpp

    using namespace llvm;
    ```

1.  首先，在访问者中声明一些私有成员。每个编译单元在 LLVM 中由`Module`类表示，访问者有一个指向模块的指针，称为`M`。为了方便 IR 生成，使用`Builder`（类型为`IRBuilder<>)`。LLVM 有一个类层次结构来表示 IR 中的类型。您可以从 LLVM 上下文中查找基本类型，如`i32`的实例。

    这些基本类型使用非常频繁。为了避免重复查找，我们缓存所需类型实例：`VoidTy`、`Int32Ty`、`PtrTy`和`Int32Zero`。`V`成员是当前计算值，它通过树遍历更新。最后，`nameMap`将变量名映射到`calc_read()`函数返回的值：

    ```cpp

    namespace {
    class ToIRVisitor : public ASTVisitor {
      Module *M;
      IRBuilder<> Builder;
      Type *VoidTy;
      Type *Int32Ty;
      PointerType *PtrTy;
      Constant *Int32Zero;
      Value *V;
      StringMap<Value *> nameMap;
    ```

1.  构造函数初始化所有成员：

    ```cpp

    public:
      ToIRVisitor(Module *M) : M(M), Builder(M->getContext())
      {
        VoidTy = Type::getVoidTy(M->getContext());
        Int32Ty = Type::getInt32Ty(M->getContext());
        PtrTy = PointerType::getUnqual(M->getContext());
        Int32Zero = ConstantInt::get(Int32Ty, 0, true);
      }
    ```

1.  对于每个函数，必须创建一个`FunctionType`实例。在 C++术语中，这被称为函数原型。函数本身是通过`Function`实例定义的。`run()`方法首先在 LLVM IR 中定义`main()`函数：

    ```cpp

      void run(AST *Tree) {
        FunctionType *MainFty = FunctionType::get(
            Int32Ty, {Int32Ty, PtrTy}, false);
        Function *MainFn = Function::Create(
            MainFty, GlobalValue::ExternalLinkage,
            "main", M);
    ```

1.  然后我们创建带有`entry`标签的`BB`基本块，并将其附加到 IR 构建器：

    ```cpp

        BasicBlock *BB = BasicBlock::Create(M->getContext(),
                                            "entry", MainFn);
        Builder.SetInsertPoint(BB);
    ```

1.  准备工作完成后，可以开始树遍历：

    ```cpp

        Tree->accept(*this);
    ```

1.  树遍历完成后，通过调用`calc_write()`函数打印计算值。同样，必须创建一个函数原型（`FunctionType`的实例）。唯一的参数是当前值`V`：

    ```cpp

        FunctionType *CalcWriteFnTy =
            FunctionType::get(VoidTy, {Int32Ty}, false);
        Function *CalcWriteFn = Function::Create(
            CalcWriteFnTy, GlobalValue::ExternalLinkage,
            "calc_write", M);
        Builder.CreateCall(CalcWriteFnTy, CalcWriteFn, {V});
    ```

1.  生成过程通过从`main()`函数返回`0`结束：

    ```cpp

        Builder.CreateRet(Int32Zero);
      }
    ```

1.  `WithDecl`节点包含声明的变量名。首先，我们为`calc_read()`函数创建一个函数原型：

    ```cpp

      virtual void visit(WithDecl &Node) override {
        FunctionType *ReadFty =
            FunctionType::get(Int32Ty, {PtrTy}, false);
        Function *ReadFn = Function::Create(
            ReadFty, GlobalValue::ExternalLinkage,
            "calc_read", M);
    ```

1.  方法遍历变量名：

    ```cpp

        for (auto I = Node.begin(), E = Node.end(); I != E;
             ++I) {
    ```

1.  对于每个变量，创建一个包含变量名的字符串：

    ```cpp

          StringRef Var = *I;
          Constant *StrText = ConstantDataArray::getString(
              M->getContext(), Var);
          GlobalVariable *Str = new GlobalVariable(
              *M, StrText->getType(),
              /*isConstant=*/true,
              GlobalValue::PrivateLinkage,
              StrText, Twine(Var).concat(".str"));
    ```

1.  然后创建调用`calc_read()`函数的 IR 代码。上一步创建的字符串作为参数传递：

    ```cpp

          CallInst *Call =
              Builder.CreateCall(ReadFty, ReadFn, {Str});
    ```

1.  返回值存储在`mapNames`映射中，以供以后使用：

    ```cpp

          nameMap[Var] = Call;
        }
    ```

1.  树遍历继续进行到表达式：

    ```cpp

        Node.getExpr()->accept(*this);
      };
    ```

1.  `Factor`节点可以是变量名或数字。对于变量名，值在`mapNames`映射中查找。对于数字，值转换为整数并转换为常量值：

    ```cpp

      virtual void visit(Factor &Node) override {
        if (Node.getKind() == Factor::Ident) {
          V = nameMap[Node.getVal()];
        } else {
          int intval;
          Node.getVal().getAsInteger(10, intval);
          V = ConstantInt::get(Int32Ty, intval, true);
        }
      };
    ```

1.  最后，对于`BinaryOp`节点，必须使用正确的计算操作：

    ```cpp

      virtual void visit(BinaryOp &Node) override {
        Node.getLeft()->accept(*this);
        Value *Left = V;
        Node.getRight()->accept(*this);
        Value *Right = V;
        switch (Node.getOperator()) {
        case BinaryOp::Plus:
          V = Builder.CreateNSWAdd(Left, Right); break;
        case BinaryOp::Minus:
          V = Builder.CreateNSWSub(Left, Right); break;
        case BinaryOp::Mul:
          V = Builder.CreateNSWMul(Left, Right); break;
        case BinaryOp::Div:
          V = Builder.CreateSDiv(Left, Right); break;
        }
      };
    };
    }
    ```

1.  这样，访问者类就完成了。`compile()`方法创建全局上下文和模块，运行树遍历，并将生成的 IR 输出到控制台：

    ```cpp

    void CodeGen::compile(AST *Tree) {
      LLVMContext Ctx;
      Module *M = new Module("calc.expr", Ctx);
      ToIRVisitor ToIR(M);
      ToIR.run(Tree);
      M->print(outs(), nullptr);
    }
    ```

现在我们已经实现了编译器的前端，从读取源代码到生成 IR。当然，所有这些组件必须协同工作以处理用户输入，这是编译器驱动程序的任务。我们还需要实现运行时所需的函数。这两者都是下一节的主题。

## 缺少的部分——驱动程序和运行时库

前几节的所有阶段都通过 `Calc.cpp` 驱动程序粘合在一起，我们按照以下方式实现：声明一个输入表达式的参数，初始化 LLVM，并调用前几节的所有阶段：

1.  首先，我们包含所需的头文件：

    ```cpp

    #include "CodeGen.h"
    #include "Parser.h"
    #include "Sema.h"
    #include "llvm/Support/CommandLine.h"
    #include "llvm/Support/InitLLVM.h"
    #include "llvm/Support/raw_ostream.h"
    ```

1.  LLVM 自带一套用于声明命令行选项的系统。你只需要为每个需要的选项声明一个静态变量。这样做时，选项会通过全局命令行解析器进行注册。这种方法的优点是每个组件可以在需要时添加命令行选项。我们声明了一个用于输入表达式的选项：

    ```cpp

    static llvm::cl::opt<std::string>
        Input(llvm::cl::Positional,
              llvm::cl::desc("<input expression>"),
              llvm::cl::init(""));
    ```

1.  在 `main()` 函数内部，首先初始化 LLVM 库。你需要调用 `ParseCommandLineOptions()` 函数来处理命令行上的选项。这也处理了打印帮助信息。如果发生错误，此方法将退出应用程序：

    ```cpp

    int main(int argc, const char **argv) {
      llvm::InitLLVM X(argc, argv);
      llvm::cl::ParseCommandLineOptions(
          argc, argv, "calc - the expression compiler\n");
    ```

1.  接下来，我们调用词法分析和语法分析器。在语法分析之后，我们检查是否发生了任何错误。如果是这种情况，则通过返回代码指示失败退出编译器：

    ```cpp

      Lexer Lex(Input);
      Parser Parser(Lex);
      AST *Tree = Parser.parse();
      if (!Tree || Parser.hasError()) {
        llvm::errs() << "Syntax errors occured\n";
        return 1;
      }
    ```

1.  如果存在语义错误，我们也会这样做：

    ```cpp

      Sema Semantic;
      if (Semantic.semantic(Tree)) {
        llvm::errs() << "Semantic errors occured\n";
        return 1;
      }
    ```

1.  在驱动程序的最后一个步骤中，调用代码生成器：

    ```cpp

      CodeGen CodeGenerator;
      CodeGenerator.compile(Tree);
      return 0;
    }
    ```

现在我们已经成功为用户输入创建了一些 IR 代码。我们将目标代码生成委托给 LLVM 的 `llc` 静态编译器，这样我们就完成了编译器的实现。我们将所有组件链接在一起以创建 `calc` 应用程序。

运行时库由一个文件 `rtcalc.c` 组成。它包含了 `calc_read()` 和 `calc_write()` 函数的实现，这些函数是用 C 编写的：

```cpp

#include <stdio.h>
#include <stdlib.h>
void calc_write(int v)
{
  printf("The result is: %d\n", v);
}
```

`calc_write()` 只将结果值写入终端：

```cpp

int calc_read(char *s)
{
  char buf[64];
  int val;
  printf("Enter a value for %s: ", s);
  fgets(buf, sizeof(buf), stdin);
  if (EOF == sscanf(buf, "%d", &val))
  {
    printf("Value %s is invalid\n", buf);
    exit(1);
  }
  return val;
}
```

`calc_read()` 从终端读取一个整数。没有任何东西阻止用户输入字母或其他字符，因此我们必须仔细检查输入。如果输入不是数字，我们退出应用程序。更复杂的方法是让用户意识到问题，并再次请求输入一个数字。

下一步是构建并尝试我们的编译器 `calc`，这是一个从表达式创建 IR 的应用程序。

### 构建并测试 calc 应用程序

为了构建 `calc`，我们首先需要在原始 `src` 目录之外创建一个新的 `CMakeLists.txt` 文件，该文件包含所有源文件实现：

1.  首先，我们将所需的最低 CMake 版本设置为 LLVM 所需的版本，并将项目命名为 `calc`：

    ```cpp

    cmake_minimum_required (VERSION 3.20.0)
    project ("calc")
    ```

1.  接下来，需要加载 LLVM 包，并将 LLVM 提供的 CMake 模块目录添加到搜索路径：

    ```cpp

    find_package(LLVM REQUIRED CONFIG)
    message("Found LLVM ${LLVM_PACKAGE_VERSION}, build type ${LLVM_BUILD_TYPE}")
    list(APPEND CMAKE_MODULE_PATH ${LLVM_DIR})
    ```

1.  我们还需要添加来自 LLVM 的定义和包含路径。使用的 LLVM 组件通过函数调用映射到库名称：

    ```cpp

    separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
    add_definitions(${LLVM_DEFINITIONS_LIST})
    include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
    llvm_map_components_to_libnames(llvm_libs Core)
    ```

1.  最后，我们指出需要将 `src` 子目录包含在我们的构建中，因为这个目录包含了本章内完成的全部 C++ 实现：

    ```cpp

    add_subdirectory ("src")
    ```

在 `src` 子目录内还需要有一个新的 `CMakeLists.txt` 文件。这个位于 `src` 目录中的 CMake 描述如下。我们简单地定义了可执行文件的名字，称为 `calc`，然后列出要编译的源文件和要链接的库：

```cpp

add_executable (calc
  Calc.cpp CodeGen.cpp Lexer.cpp Parser.cpp Sema.cpp)
target_link_libraries(calc PRIVATE ${llvm_libs})
```

最后，我们可以开始构建 `calc` 应用程序。在 `src` 目录之外，我们创建一个新的构建目录并切换到该目录。之后，我们可以按照以下方式运行 CMake 和构建调用：

```cpp

$ cmake -GNinja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_DIR=<path to llvm installation configuration> ../
$ ninja
```

现在我们应该有一个新构建的、功能齐全的 `calc` 应用程序，它可以生成 LLVM IR 代码。这可以进一步与 `llc` 一起使用，`llc` 是 LLVM 静态后端编译器，用于将 IR 代码编译成目标文件。

然后，你可以使用你喜欢的 C 编译器来链接到小的运行时库。在 Unix 的 X86 上，你可以输入以下内容：

```cpp

$ calc "with a: a*3" | llc –filetype=obj \
  -relocation-model=pic  –o=expr.o
$ clang –o expr expr.o rtcalc.c
$ expr
Enter a value for a: 4
The result is: 12
```

在其他 Unix 平台，如 AArch64 或 PowerPC 上，你必须移除 `-relocation-model=pic` 选项。

在 Windows 上，你需要使用 `cl` 编译器，如下所示：

```cpp

$ calc "with a: a*3" | llc –filetype=obj –o=expr.obj
$ cl expr.obj rtcalc.c
$ expr
Enter a value for a: 4
The result is: 12
```

你现在已经创建出了你的第一个基于 LLVM 的编译器！请花些时间尝试各种表达式。特别是要检查乘法运算符是否在加法运算符之前被评估，以及使用括号是否会改变评估顺序，正如我们从一个基本的计算器所期望的那样。

# 摘要

在本章中，你了解了编译器的典型组件。使用算术表达式语言介绍了编程语言的语法。你学习了如何开发这种语言前端典型的组件：词法分析器、解析器、语义分析器和代码生成器。代码生成器仅生成 LLVM IR，并使用 LLVM 的 `llc` 静态编译器从它创建目标文件。你现在已经开发出了你的第一个基于 LLVM 的编译器！

在下一章中，你将深化这些知识，构建编程语言的前端。
