# 第三章：编译器的结构

编译器技术是计算机科学中一个深入研究的领域。它的高级任务是将源语言翻译成机器码。通常，这个任务分为两部分：前端和后端。前端主要处理源语言，而后端负责生成机器码。

在本章中，我们将涵盖以下主题：

+   编译器的构建模块，您将了解到编译器中通常找到的组件。

+   算术表达式语言，将为您介绍一个示例语言。您将学习语法如何用于定义语言。

+   词法分析，将讨论如何为语言实现词法分析器。

+   语法分析，涵盖如何从语法构建解析器。

+   语义分析，您将学习如何实现语义检查。

+   使用 LLVM 后端进行代码生成，将讨论如何与 LLVM 后端进行接口，以及如何将所有阶段连接在一起创建完整的编译器。

# 技术要求

本章的代码文件可在以下链接找到：[`github.com/PacktPublishing/Learn-LLVM-12/tree/master/Chapter03/calc`](https://github.com/PacktPublishing/Learn-LLVM-12/tree/master/Chapter03/calc)

您可以在以下链接找到代码的操作视频：[`bit.ly/3nllhED`](https://bit.ly/3nllhED)

# 编译器的构建模块

自从上个世纪中期计算机问世以来，很快就显而易见，比汇编语言更抽象的语言对编程是有用的。早在 1957 年，Fortran 就是第一种可用的高级编程语言。从那时起，成千上万种编程语言被开发出来。事实证明，所有编译器都必须解决相同的任务，并且编译器的实现最好根据这些任务进行结构化。

在最高级别上，编译器由两部分组成：前端和后端。前端负责特定于语言的任务。它读取源文件并计算其语义分析表示，通常是带注释的**抽象语法树**（**AST**）。后端从前端的结果创建优化的机器码。前端和后端之间有区分的动机是可重用性。假设前端和后端之间的接口定义良好。在这里，您可以将 C 和 Modula-2 前端连接到相同的后端。或者，如果您有一个用于 X86 的后端和一个用于 Sparc 的后端，那么您可以将 C++前端连接到两者。

前端和后端有特定的结构。前端通常执行以下任务：

1.  词法分析器读取源文件并生成标记流。

1.  解析器从标记流创建 AST。

1.  语义分析器向 AST 添加语义信息。

1.  代码生成器从 AST 生成**中间表示**（**IR**）。

中间表示是后端的接口。后端执行以下任务：

1.  后端对 IR 进行与目标无关的优化。

1.  然后，它为 IR 代码选择指令。

1.  然后，它对指令执行与目标相关的优化。

1.  最后，它会发出汇编代码或目标文件。

当然，这些说明仅在概念层面上。实现方式各不相同。LLVM 核心库定义了一个中间表示作为后端的标准接口。其他工具可以使用带注释的 AST。C 预处理器是一种独立的语言。它可以作为一个独立的应用程序实现，输出预处理的 C 源代码，或者作为词法分析器和解析器之间的附加组件。在某些情况下，AST 不必显式构造。如果要实现的语言不太复杂，那么将解析器和语义分析器结合起来，然后在解析过程中生成代码是一种常见的方法。即使程序设计语言的特定实现没有明确命名这些组件，也要记住这些任务仍然必须完成。

在接下来的章节中，我们将为一个表达式语言构建一个编译器，该编译器可以从输入中生成 LLVM IR。LLVM 静态编译器`llc`代表后端，然后可以用于将 IR 编译成目标代码。一切都始于定义语言。

# 算术表达式语言

算术表达式是每种编程语言的一部分。这里有一个名为**calc**的算术表达式计算语言的示例。calc 表达式被编译成一个应用程序，用于计算以下表达式：

```cpp
with a, b: a * (4 + b)
```

表达式中使用的变量必须使用`with`关键字声明。这个程序被编译成一个应用程序，该应用程序要求用户输入`a`和`b`变量的值，并打印结果。

示例总是受欢迎的，但作为编译器编写者，你需要比这更彻底的规范来进行实现和测试。编程语言的语法的载体是其语法。

## 用于指定编程语言语法的形式化方法

语言的元素，如关键字、标识符、字符串、数字和运算符，被称为**标记**。从这个意义上说，程序是一系列标记的序列，语法规定了哪些序列是有效的。

通常，语法是用**扩展的巴科斯-瑙尔范式（EBNF）**编写的。语法的一个规则是它有左侧和右侧。左侧只是一个称为**非终结符**的单个符号。规则的右侧由非终结符、标记和用于替代和重复的元符号组成。让我们来看看 calc 语言的语法：

```cpp
calc : ("with" ident ("," ident)* ":")? expr ;
expr : term (( "+" | "-" ) term)* ;
term : factor (( "*" | "/") factor)* ;
factor : ident | number | "(" expr ")" ;
ident : ([a-zAZ])+ ;
number : ([0-9])+ ;
```

在第一行中，`calc`是一个非终结符。如果没有另外说明，那么语法的第一个非终结符是起始符号。冒号`:`是规则左侧和右侧的分隔符。`"with"`、`,`和`":"`是代表这个字符串的标记。括号用于分组。一个组可以是可选的或重复的。括号后面的问号`?`表示一个可选组。星号`*`表示零次或多次重复，加号`+`表示一次或多次重复。`ident`和`expr`是非终结符。对于每一个，都存在另一个规则。分号`;`标记了规则的结束。第二行中的竖线`|`表示一个替代。最后，最后两行中的方括号`[]`表示一个字符类。有效的字符写在方括号内。例如，`[a-zA-Z]`字符类匹配大写或小写字母，`([a-zA-Z])+`匹配一个或多个这些字母。这对应于一个正则表达式。

## 语法如何帮助编译器编写者

这样的语法可能看起来像一个理论上的玩具，但对于编译器编写者来说是有价值的。首先，定义了所有的标记，这是创建词法分析器所需的。语法的规则可以被转换成解析器。当然，如果对解析器是否正确工作有疑问，那么语法就是一个很好的规范。

然而，语法并没有定义编程语言的所有方面。语法的含义 - 语义 - 也必须被定义。为此目的开发了形式化方法，但通常是以纯文本的方式指定的，类似于语言首次引入时的情况。

掌握了这些知识，接下来的两节将向您展示词法分析如何将输入转换为标记序列，以及如何在 C++中对语法进行编码以进行语法分析。

# 词法分析

正如我们在上一节的示例中看到的，编程语言由许多元素组成，如关键字、标识符、数字、运算符等。词法分析的任务是接受文本输入并从中创建一个标记序列。calc 语言由`with`、`:`、`+`、`-`、`*`、`/`、`(`和`)`标记以及`([a-zA-Z])+`（标识符）和`([0-9])+`（数字）正则表达式组成。我们为每个标记分配一个唯一的数字，以便更容易地处理它们。

## 手写词法分析器

词法分析器的实现通常称为`Lexer`。让我们创建一个名为`Lexer.h`的头文件，并开始定义`Token`。它以通常的头文件保护和所需的头文件开始：

```cpp
#ifndef LEXER_H
#define LEXER_H
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
```

`llvm::MemoryBuffer`类提供对填充有文件内容的内存块的只读访问。在请求时，会在缓冲区的末尾添加一个尾随的零字符（`'\x00'`）。我们使用这个特性来在不检查每次访问时缓冲区的长度的情况下读取缓冲区。`llvm::StringRef`类封装了指向 C 字符串及其长度的指针。由于长度被存储，字符串不需要像普通的 C 字符串那样以零字符（`'\x00'`）结尾。这允许`StringRef`的实例指向由`MemoryBuffer`管理的内存。让我们更详细地看一下这个：

1.  首先，`Token`类包含了先前提到的唯一标记数字的枚举定义：

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

除了为每个标记定义一个成员之外，我们还添加了两个额外的值：`eoi`和`unknown`。`eoi`代表结束输入，`unknown`用于在词法级别出现错误的情况下；例如，`#`不是语言的标记，因此会被映射为`unknown`。

1.  除了枚举之外，该类还有一个成员`Text`，它指向标记文本的开头。它使用了之前提到的`StringRef`类：

```cpp
private:
  TokenKind Kind;
  llvm::StringRef Text;
public:
  TokenKind getKind() const { return Kind; }
  llvm::StringRef getText() const { return Text; }
```

这对于语义处理很有用，因为知道标识符的名称是很有用的。

1.  `is()`和`isOneOf()`方法用于测试标记是否属于某种类型。`isOneOf()`方法使用可变模板，允许可变数量的参数：

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

1.  `Lexer`类本身具有类似的简单接口，并在头文件中紧随其后：

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

除了构造函数之外，公共接口只包含`next()`方法，它返回下一个标记。该方法的行为类似于迭代器，总是前进到下一个可用的标记。该类的唯一成员是指向输入开头和下一个未处理字符的指针。假定缓冲区以终止`0`（类似于 C 字符串）结束。

1.  让我们在`Lexer.cpp`文件中实现`Lexer`类。它以一些辅助函数开始，以帮助对字符进行分类：

```cpp
#include "Lexer.h"
namespace charinfo {
LLVM_READNONE inline bool isWhitespace(char c) {
  return c == ' ' || c == '\t' || c == '\f' ||         c == '\v' ||
         c == '\r' || c == '\n';
}
LLVM_READNONE inline bool isDigit(char c) {
  return c >= '0' && c <= '9';
}
LLVM_READNONE inline bool isLetter(char c) {
  return (c >= 'a' && c <= 'z') ||         (c >= 'A' && c <= 'Z');
}
}
```

这些函数用于使条件更易读。

注意

我们不使用`<cctype>`标准库头文件提供的函数有两个原因。首先，这些函数根据环境中定义的区域设置而改变行为。例如，如果区域设置是德语区域设置，则德语变音符可以被分类为字母。这通常不是编译器所希望的。其次，由于这些函数的参数类型为`int`，我们必须从`char`类型转换。这种转换的结果取决于`char`是作为有符号类型还是无符号类型处理，这会导致可移植性问题。

1.  根据上一节中的语法，我们知道语言的所有标记。但是语法并没有定义应该忽略的字符。例如，空格或换行符只会添加空白并经常被忽略。`next()`方法首先忽略这些字符：

```cpp
void Lexer::next(Token &token) {
  while (*BufferPtr &&         charinfo::isWhitespace(*BufferPtr)) {
    ++BufferPtr;
  }
```

1.  接下来，确保仍有字符需要处理：

```cpp
  if (!*BufferPtr) {
    token.Kind = Token::eoi;
    return;
  }
```

至少有一个字符需要处理。

1.  因此，我们首先检查字符是小写还是大写。在这种情况下，标记要么是标识符，要么是`with`关键字，因为标识符的正则表达式也匹配关键字。常见的解决方案是收集正则表达式匹配的字符，并检查字符串是否恰好是关键字：

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

私有的`formToken()`方法用于填充标记。

1.  接下来，我们检查是否为数字。以下代码与先前显示的代码非常相似：

```cpp
  else if (charinfo::isDigit(*BufferPtr)) {
    const char *end = BufferPtr + 1;
    while (charinfo::isDigit(*end))
      ++end;
    formToken(token, end, Token::number);
    return;
  }
```

1.  现在，只剩下由固定字符串定义的标记。这很容易用`switch`来实现。由于所有这些标记只有一个字符，所以使用`CASE`预处理宏来减少输入：

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

1.  最后，我们需要检查是否有意外的字符：

```cpp
    default:
      formToken(token, BufferPtr + 1, Token::unknown);
    }
    return;
  }
}
```

只有私有的辅助方法`formToken()`还缺失。

1.  这个私有的辅助方法填充了`Token`实例的成员并更新了指向下一个未处理字符的指针：

```cpp
void Lexer::formToken(Token &Tok, const char *TokEnd,
                      Token::TokenKind Kind) {
  Tok.Kind = Kind;
  Tok.Text = llvm::StringRef(BufferPtr, TokEnd -                              BufferPtr);
  BufferPtr = TokEnd;
}
```

在下一节中，我们将看一下如何构建用于语法分析的解析器。

# 语法分析

语法分析由我们将在下一步实现的解析器完成。它的基础是前几节的语法和词法分析器。解析过程的结果是一种称为**抽象语法树**（**AST**）的动态数据结构。AST 是输入的非常简洁的表示形式，并且非常适合语义分析。首先，我们将实现解析器。之后，我们将看一下 AST。

## 手写解析器

解析器的接口在`Parser.h`头文件中定义。它以一些`include`语句开始：

```cpp
#ifndef PARSER_H
#define PARSER_H
#include "AST.h"
#include "Lexer.h"
#include "llvm/Support/raw_ostream.h"
```

`AST.h`头文件声明了 AST 的接口，并将在稍后显示。LLVM 的编码指南禁止使用`<iostream>`库，因此必须包含等效的 LLVM 功能的头文件。需要发出错误消息。让我们更详细地看一下这个：

1.  首先，`Parser`类声明了一些私有成员：

```cpp
class Parser {
  Lexer &Lex;
  Token Tok;
  bool HasError;
```

`Lex`和`Tok`是前一节中的类的实例。`Tok`存储下一个标记（向前看），而`Lex`用于从输入中检索下一个标记。`HasError`标志指示是否检测到错误。

1.  有几种方法处理标记：

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

`advance()`从词法分析器中检索下一个标记。`expect()`测试向前看是否是预期的类型，如果不是则发出错误消息。最后，`consume()`如果向前看是预期的类型，则检索下一个标记。如果发出错误消息，则将`HasError`标志设置为 true。

1.  对于语法中的每个非终结符，声明了一个解析规则的方法：

```cpp
  AST *parseCalc();
  Expr *parseExpr();
  Expr *parseTerm();
  Expr *parseFactor();
```

注意

`ident`和`number`没有方法。这些规则只返回标记，并由相应的标记替换。

1.  以下是公共接口。构造函数初始化所有成员并从词法分析器中检索第一个标记：

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

1.  最后，`parse()`方法是解析的主要入口点：

```cpp
  AST *parse();
};
#endif
```

在下一节中，我们将学习如何实现解析器。

### 解析器实现

让我们深入了解解析器的实现：

1.  解析器的实现可以在`Parser.cpp`文件中找到，并以`parse()`方法开始：

```cpp
#include "Parser.h"
AST *Parser::parse() {
  AST *Res = parseCalc();
  expect(Token::eoi);
  return Res;
}
```

`parse()`方法的主要目的是整个输入已被消耗。您还记得第一节中解析示例添加了一个特殊符号来表示输入的结束吗？我们将在这里检查这一点。

1.  `parseCalc()`方法实现了相应的规则。让我们回顾一下第一节的规则：

```cpp
calc : ("with" ident ("," ident)* ":")? expr ;
```

1.  该方法开始声明一些局部变量：

```cpp
AST *Parser::parseCalc() {
  Expr *E;
  llvm::SmallVector<llvm::StringRef, 8> Vars;
```

1.  首先要做出的决定是是否必须解析可选组。该组以`with`标记开始，因此我们将标记与此值进行比较：

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

如果有一个标识符，那么我们将其保存在`Vars`向量中。否则，这是一个语法错误，需要单独处理。

1.  语法中现在跟随一个重复组，它解析更多的标识符，用逗号分隔：

```cpp
    while (Tok.is(Token::comma)) {
      advance();
      if (expect(Token::ident))
        goto _error;
      Vars.push_back(Tok.getText());
      advance();
    }
```

这一点现在对你来说应该不足为奇了。重复组以`the`标记开始。标记的测试成为`while`循环的条件，实现零次或多次重复。循环内的标识符被视为之前处理的方式。

1.  最后，可选组需要在末尾加上冒号：

```cpp
    if (consume(Token::colon))
      goto _error;
  }
```

1.  现在，必须解析`expr`规则：

```cpp
  E = parseExpr();
```

1.  通过这个调用，规则已经成功解析。我们收集的信息现在用于创建这个规则的 AST 节点：

```cpp
  if (Vars.empty()) return E;
  else return new WithDecl(Vars, E);
```

现在，只有错误处理代码还缺失。检测语法错误很容易，但从中恢复却令人惊讶地复杂。在这里，必须使用一种称为**恐慌模式**的简单方法。

在恐慌模式中，从标记流中删除标记，直到找到解析器可以继续工作的标记为止。大多数编程语言都有表示结束的符号；例如，在 C++中，我们可以使用`;`（语句的结束）或`}`（块的结束）。这些标记是寻找的好候选者。

另一方面，错误可能是我们正在寻找的符号丢失了。在这种情况下，可能会在解析器继续之前删除很多标记。这并不像听起来那么糟糕。今天，编译器的速度更重要。在出现错误时，开发人员查看第一个错误消息，修复它，然后重新启动编译器。这与使用穿孔卡完全不同，那时尽可能多地获得错误消息非常重要，因为下一次运行编译器只能在第二天进行。

### 错误处理

不是使用一些任意的标记，而是使用另一组标记。对于每个非终端，都有一组可以在规则中跟随这个非终端的标记。让我们来看一下：

1.  在`calc`的情况下，只有输入的结尾跟随这个非终端。它的实现是微不足道的：

```cpp
_error:
  while (!Tok.is(Token::eoi))
    advance();
  return nullptr;
}
```

1.  其他解析方法的构造方式类似。`parseExpr()`是对`expr`规则的翻译：

```cpp
Expr *Parser::parseExpr() {
  Expr *Left = parseTerm();
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

规则内的重复组被翻译成了`while`循环。请注意`isOneOf()`方法的使用简化了对多个标记的检查。

1.  `term`规则的编码看起来是一样的：

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

这个方法与`parseExpr()`非常相似，你可能会想将它们合并成一个。在语法中，可以有一个处理乘法和加法运算符的规则。使用两个规则而不是一个的优势在于运算符的优先级与数学计算顺序很匹配。如果合并这两个规则，那么你需要在其他地方找出评估顺序。

1.  最后，你需要实现`factor`规则：

```cpp
Expr *Parser::parseFactor() {
  Expr *Res = nullptr;
  switch (Tok.getKind()) {
  case Token::number:
    Res = new Factor(Factor::Number, Tok.getText());
    advance(); break;
```

与使用一系列`if`和`else if`语句不同，这里似乎更适合使用`switch`语句，因为每个备选方案都以一个标记开始。一般来说，你应该考虑使用哪种翻译模式。如果以后需要更改解析方法，那么如果不是每个方法都有不同的实现语法规则的方式，那就是一个优势。

1.  如果使用`switch`语句，那么错误处理发生在`default`情况下：

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

我们在这里防止发出错误消息，因为会出现错误。

1.  如果括号表达式中有语法错误，那么会发出错误消息。保护措施防止发出第二个错误消息：

```cpp
    while (!Tok.isOneOf(Token::r_paren, Token::star,
                        Token::plus, Token::minus,
                        Token::slash, Token::eoi))
      advance();
  }
  return Res;
}
```

这很容易，不是吗？一旦你记住了使用的模式，根据语法规则编写解析器几乎是乏味的。这种类型的解析器称为**递归下降解析器**。

递归下降解析器无法从所有语法构造出来

语法必须满足一定条件才能适合构造递归下降解析器。这类语法称为 LL(1)。事实上，大多数你可以在互联网上找到的语法都不属于这类语法。大多数关于编译器构造理论的书都解释了这个原因。这个主题的经典书籍是所谓的“龙书”，即 Aho、Lam、Sethi 和 Ullman 的*编译器原理、技术和工具*。

## 抽象语法树

解析过程的结果是一个`;`，表示单个语句的结束。当然，这对解析器很重要。一旦我们将语句转换为内存表示，分号就不再重要，可以被丢弃。

如果你看一下例子表达式语言的第一个规则，那么很明显`with`关键字，逗号`,`和冒号`:`对程序的含义并不重要。重要的是声明的变量列表，这些变量可以在表达式中使用。结果是只需要几个类来记录信息：`Factor`保存数字或标识符，`BinaryOp`保存算术运算符和表达式的左右两侧，`WithDecl`保存声明的变量列表和表达式。`AST`和`Expr`仅用于创建一个公共类层次结构。

除了从解析输入中获得的信息外，还要在使用`AST.h`头文件时进行树遍历。让我们来看一下：

1.  它以访问者接口开始：

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

访问者模式需要知道它必须访问的每个类。因为每个类也引用了访问者，我们在文件顶部声明所有类。请注意，`AST`和`Expr`的`visit()`方法具有默认实现，什么也不做。

1.  `AST`类是层次结构的根：

```cpp
class AST {
public:
  virtual ~AST() {}
  virtual void accept(ASTVisitor &V) = 0;
};
```

1.  同样，`Expr`是与表达式相关的`AST`类的根：

```cpp
class Expr : public AST {
public:
  Expr() {}
};
```

1.  `Factor`类存储数字或变量的名称：

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

在这个例子中，数字和变量几乎被处理得一样，因此我们决定只创建一个 AST 节点类来表示它们。`Kind`成员告诉我们实例代表这两种情况中的哪一种。在更复杂的语言中，通常希望有不同的 AST 类，比如`NumberLiteral`类用于数字，`VariableAccess`类用于引用变量。

1.  `BinaryOp`类保存了评估表达式所需的数据：

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

与解析器相比，`BinaryOp`类在乘法和加法运算符之间没有区别。运算符的优先级隐含在树结构中。

1.  最后，`WithDecl`存储了声明的变量和表达式：

```cpp
class WithDecl : public AST {
  using VarVector =                   llvm::SmallVector<llvm::StringRef, 8>;
  VarVector Vars;
  Expr *E;
public:
  WithDecl(llvm::SmallVector<llvm::StringRef, 8> Vars,
           Expr *E)
      : Vars(Vars), E(E) {}
  VarVector::const_iterator begin()                                 { return Vars.begin(); }
  VarVector::const_iterator end() { return Vars.end(); }
  Expr *getExpr() { return E; }
  virtual void accept(ASTVisitor &V) override {
    V.visit(*this);
  }
};
#endif
```

AST 在解析过程中构建。语义分析检查树是否符合语言的含义（例如，使用的变量是否已声明），并可能增强树。之后，树被用于代码生成。

# 语义分析

语义分析器遍历 AST 并检查语言的各种语义规则；例如，变量必须在使用前声明，或者表达式中的变量类型必须兼容。如果语义分析器发现可以改进的情况，还可以打印警告。对于示例表达语言，语义分析器必须检查每个使用的变量是否已声明，因为语言要求如此。可能的扩展（这里不会实现）是在未使用的情况下打印警告消息。

语义分析器实现在 `Sema` 类中，语义分析由 `semantic()` 方法执行。以下是完整的 `Sema.h` 头文件：

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

实现在 `Sema.cpp` 文件中。有趣的部分是语义分析，它使用访问者来实现。基本思想是每个声明的变量名都存储在一个集合中。在创建集合时，我们可以检查每个名称是否唯一，然后稍后检查名称是否在集合中：

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

与 `Parser` 类一样，使用标志来指示是否发生错误。名称存储在名为 `Scope` 的集合中。在包含变量名的 `Factor` 节点中，我们检查变量名是否在集合中：

```cpp
  virtual void visit(Factor &Node) override {
    if (Node.getKind() == Factor::Ident) {
      if (Scope.find(Node.getVal()) == Scope.end())
        error(Not, Node.getVal());
    }
  };
```

对于 `BinaryOp` 节点，我们只需要检查两侧是否存在并已被访问：

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

在 `WithDecl` 节点中，集合被填充，并开始对表达式的遍历：

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

`semantic()` 方法只是开始树遍历并返回错误标志：

```cpp
bool Sema::semantic(AST *Tree) {
  if (!Tree)
    return false;
  DeclCheck Check;
  Tree->accept(Check);
  return Check.hasError();
}
```

如果需要，这里可以做更多的工作。还可以打印警告消息，如果声明的变量未被使用。我们留给您来实现。如果语义分析没有错误完成，那么我们可以从 AST 生成 LLVM IR。我们将在下一节中进行这个操作。

# 使用 LLVM 后端生成代码

后端的任务是从模块的 **IR** 创建优化的机器代码。IR 是后端的接口，可以使用 C++ 接口或文本形式创建。同样，IR 是从 AST 生成的。

## LLVM IR 的文本表示

在尝试生成 LLVM IR 之前，我们需要了解我们想要生成什么。对于示例表达语言，高级计划如下：

1.  询问用户每个变量的值。

1.  计算表达式的值。

1.  打印结果。

要求用户为变量提供一个值并打印结果，使用了两个库函数 `calc_read()` 和 `calc_write()`。对于 `with a: 3*a` 表达式，生成的 IR 如下：

1.  库函数必须像 C 语言一样声明。语法也类似于 C 语言。函数名前的类型是返回类型。括号中的类型是参数类型。声明可以出现在文件的任何位置：

```cpp
declare i32 @calc_read(i8*)
declare void @calc_write(i32)
```

1.  `calc_read()` 函数以变量名作为参数。以下结构定义了一个常量，保存了 `a` 和在 C 语言中用作字符串终结符的空字节：

```cpp
@a.str = private constant [2 x i8] c"a\00"
```

1.  它跟在 `main()` 函数后面。参数的名称被省略，因为它们没有被使用。与 C 语言一样，函数的主体用大括号括起来：

```cpp
define i32 @main(i32, i8**) {
```

1.  每个基本块必须有一个标签。因为这是函数的第一个基本块，我们将其命名为 `entry`：

```cpp
entry:
```

1.  调用 `calc_read()` 函数来读取 `a` 变量的值。嵌套的 `getelemenptr` 指令执行索引计算以计算字符串常量的第一个元素的指针。函数的结果被赋值给未命名的 `%2` 变量：

```cpp
  %2 = call i32 @calc_read(i8* getelementptr inbounds
                 ([2 x i8], [2 x i8]* @a.str, i32 0, i32 0))
```

1.  接下来，变量乘以 `3`：

```cpp
  %3 = mul nsw i32 3, %2
```

1.  结果通过调用 `calc_write()` 函数打印到控制台：

```cpp
  call void @calc_write(i32 %3)
```

1.  最后，`main()` 函数返回 `0` 表示执行成功：

```cpp
  ret i32 0
}
```

LLVM IR 中的每个值都是有类型的，`i32`表示 32 位整数类型，`i8*`表示指向字节的指针。IR 代码非常可读（也许除了`getelementptr`操作之外，在*第五章**，IR 生成基础*中将详细解释）。现在清楚了 IR 的样子，让我们从 AST 生成它。

## 从 AST 生成 IR。

在`CodeGen.h`头文件中提供的接口非常小：

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

因为 AST 包含了语义分析阶段的信息，基本思想是使用访问者遍历 AST。`CodeGen.cpp`文件的实现如下：

1.  所需的包含在文件顶部：

```cpp
#include "CodeGen.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/raw_ostream.h"
```

1.  LLVM 库的命名空间用于名称查找：

```cpp
using namespace llvm;
```

1.  首先，在访问者中声明了一些私有成员。LLVM 中，每个编译单元都由`Module`类表示，访问者有一个指向模块调用`M`的指针。为了方便生成 IR，使用了`Builder`（`IRBuilder<>`类型）。LLVM 有一个类层次结构来表示 IR 中的类型。您可以在 LLVM 上下文中查找基本类型的实例，比如`i32`。这些基本类型经常被使用。为了避免重复查找，我们缓存所需的类型实例，可以是`VoidTy`、`Int32Ty`、`Int8PtrTy`、`Int8PtrPtrTy`或`Int32Zero`。`V`是当前计算的值，通过树遍历更新。最后，`nameMap`将变量名映射到`calc_read()`函数返回的值：

```cpp
namespace {
class ToIRVisitor : public ASTVisitor {
  Module *M;
  IRBuilder<> Builder;
  Type *VoidTy;
  Type *Int32Ty;
  Type *Int8PtrTy;
  Type *Int8PtrPtrTy;
  Constant *Int32Zero;
  Value *V;
  StringMap<Value *> nameMap;
```

1.  构造函数初始化了所有成员：

```cpp
public:
  ToIRVisitor(Module *M) : M(M), Builder(M->getContext()) 
  {
    VoidTy = Type::getVoidTy(M->getContext());
    Int32Ty = Type::getInt32Ty(M->getContext());
    Int8PtrTy = Type::getInt8PtrTy(M->getContext());
    Int8PtrPtrTy = Int8PtrTy->getPointerTo();
    Int32Zero = ConstantInt::get(Int32Ty, 0, true);
  }
```

1.  对于每个函数，必须创建一个`FunctionType`实例。在 C++术语中，这是一个函数原型。函数本身是用`Function`实例定义的。首先，`run()`方法在 LLVM IR 中定义了`main()`函数：

```cpp
  void run(AST *Tree) {
    FunctionType *MainFty = FunctionType::get(
        Int32Ty, {Int32Ty, Int8PtrPtrTy}, false);
    Function *MainFn = Function::Create(
        MainFty, GlobalValue::ExternalLinkage,
        "main", M);
```

1.  然后，使用`entry`标签创建`BB`基本块，并将其附加到 IR 构建器：

```cpp
    BasicBlock *BB = BasicBlock::Create(M->getContext(),
                                        "entry", MainFn);
    Builder.SetInsertPoint(BB);
```

1.  准备工作完成后，树遍历可以开始：

```cpp
    Tree->accept(*this);
```

1.  树遍历后，通过调用`calc_write()`函数打印计算出的值。再次，必须创建函数原型（`FunctionType`的实例）。唯一的参数是当前值`V`：

```cpp
    FunctionType *CalcWriteFnTy =
        FunctionType::get(VoidTy, {Int32Ty}, false);
    Function *CalcWriteFn = Function::Create(
        CalcWriteFnTy, GlobalValue::ExternalLinkage,
        "calc_write", M);
    Builder.CreateCall(CalcWriteFnTy, CalcWriteFn, {V});
```

1.  生成完成后，从`main()`函数返回`0`：

```cpp
    Builder.CreateRet(Int32Zero);
  }
```

1.  `WithDecl`节点保存了声明变量的名称。首先，必须为`calc_read()`函数创建函数原型：

```cpp
  virtual void visit(WithDecl &Node) override {
    FunctionType *ReadFty =
        FunctionType::get(Int32Ty, {Int8PtrTy}, false);
    Function *ReadFn = Function::Create(
        ReadFty, GlobalValue::ExternalLinkage, 
        "calc_read", M);
```

1.  该方法循环遍历变量名：

```cpp
    for (auto I = Node.begin(), E = Node.end(); I != E;
         ++I) {
```

1.  为每个变量创建一个带有变量名的字符串：

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

1.  然后，创建调用`calc_read()`函数的 IR 代码。将在上一步中创建的字符串作为参数传递：

```cpp
      Value *Ptr = Builder.CreateInBoundsGEP(
          Str, {Int32Zero, Int32Zero}, "ptr");
      CallInst *Call =
          Builder.CreateCall(ReadFty, ReadFn, {Ptr});
```

1.  返回的值存储在`mapNames`映射中以供以后使用：

```cpp
      nameMap[Var] = Call;
    }
```

1.  树遍历继续进行，表达式如下：

```cpp
    Node.getExpr()->accept(*this);
  };
```

1.  `Factor`节点可以是变量名或数字。对于变量名，在`mapNames`映射中查找值。对于数字，将值转换为整数并转换为常量值：

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

1.  这样，访问者类就完成了。`compile()`方法创建全局上下文和模块，运行树遍历，并将生成的 IR 转储到控制台：

```cpp
void CodeGen::compile(AST *Tree) {
  LLVMContext Ctx;
  Module *M = new Module("calc.expr", Ctx);
  ToIRVisitor ToIR(M);
  ToIR.run(Tree);
  M->print(outs(), nullptr);
}
```

通过这样，我们已经实现了编译器的前端，从读取源代码到生成 IR。当然，所有这些组件必须在用户输入上一起工作，这是编译器驱动程序的任务。我们还需要实现运行时所需的函数。我们将在下一节中涵盖这两个方面。

## 缺失的部分 - 驱动程序和运行时库

前几节的所有阶段都由`Calc.cpp`驱动程序连接在一起，我们将在这里实现。此时，声明了输入表达式的参数，初始化了 LLVM，并调用了前几节的所有阶段。让我们来看一下：

1.  首先，必须包含所需的头文件：

```cpp
#include "CodeGen.h"
#include "Parser.h"
#include "Sema.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"
```

1.  LLVM 有自己的命令行选项声明系统。您只需要为每个需要的选项声明一个静态变量。这样做，选项就会在全局命令行解析器中注册。这种方法的优势在于每个组件都可以在需要时添加命令行选项。我们必须为输入表达式声明一个选项：

```cpp
static llvm::cl::opt<std::string>
    Input(llvm::cl::Positional,
          llvm::cl::desc("<input expression>"),
          llvm::cl::init(""));
```

1.  在`main()`函数内，初始化了 LLVM 库。您需要调用`ParseCommandLineOptions`来处理命令行上的选项。这也处理打印帮助信息。在出现错误的情况下，此方法会退出应用程序：

```cpp
int main(int argc, const char **argv) {
  llvm::InitLLVM X(argc, argv);
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "calc - the expression compiler\n");
```

1.  接下来，我们调用词法分析器和语法分析器。在语法分析之后，我们检查是否发生了错误。如果是这种情况，那么我们以一个返回代码退出编译器，表示失败：

```cpp
  Lexer Lex(Input);
  Parser Parser(Lex);
  AST *Tree = Parser.parse();
  if (!Tree || Parser.hasError()) {
    llvm::errs() << "Syntax errors occured\n";
    return 1;
  }
```

1.  如果有语义错误，我们也会这样做。

```cpp
  Sema Semantic;
  if (Semantic.semantic(Tree)) {
    llvm::errs() << "Semantic errors occured\n";
    return 1;
  }
```

1.  最后，在驱动程序中，调用了代码生成器：

```cpp
  CodeGen CodeGenerator;
  CodeGenerator.compile(Tree);
  return 0;
}
```

有了这个，我们已经成功地为用户输入创建了 IR 代码。我们将对象代码生成委托给 LLVM 静态编译器`llc`，因此这完成了我们的编译器的实现。我们必须将所有组件链接在一起，以创建`calc`应用程序。

运行时库由一个名为`rtcalc.c`的单个文件组成。它包含了用 C 编写的`calc_read()`和`calc_write()`函数的实现：

```cpp
#include <stdio.h>
#include <stdlib.h>
void calc_write(int v)
{
  printf("The result is: %d\n", v);
}
```

`calc_write()`只是将结果值写入终端：

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

`calc_read()`从终端读取一个整数。没有任何限制阻止用户输入字母或其他字符，因此我们必须仔细检查输入。如果输入不是数字，我们就退出应用程序。一个更复杂的方法是让用户意识到问题，并再次要求输入一个数字。

现在，我们可以尝试我们的编译器。`calc`应用程序从表达式创建 IR。LLVM 静态编译器`llc`将 IR 编译为一个目标文件。然后，您可以使用您喜欢的 C 编译器链接到小型运行时库。在 Unix 上，您可以输入以下内容：

```cpp
$ calc "with a: a*3" | llc –filetype=obj –o=expr.o
$ clang –o expr expr.o rtcalc.c
$ expr
Enter a value for a: 4
The result is: 12
```

在 Windows 上，您很可能会使用`cl`编译器：

```cpp
$ calc "with a: a*3" | llc –filetype=obj –o=expr.obj
$ cl expr.obj rtcalc.c
$ expr
Enter a value for a: 4
The result is: 12
```

有了这个，您已经创建了您的第一个基于 LLVM 的编译器！请花一些时间玩弄各种表达式。还要检查乘法运算符在加法运算符之前进行评估，并且使用括号会改变评估顺序，这是我们从基本计算器中期望的。

# 总结

在本章中，您了解了编译器的典型组件。一个算术表达式语言被用来向您介绍编程语言的语法。然后，您学会了如何为这种语言开发典型的前端组件：词法分析器、语法分析器、语义分析器和代码生成器。代码生成器只产生了 LLVM IR，LLVM 静态编译器`llc`用它来创建目标文件。最后，您开发了您的第一个基于 LLVM 的编译器！

在下一章中，您将加深这些知识，以构建一个编程语言的前端。
