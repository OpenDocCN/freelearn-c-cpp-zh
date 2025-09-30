# 高级领域特定语言

在上一章中，我们开发了一个**领域特定语言**（**DSL**）。在本章中，我们将以几种方式改进语言：

+   我们将添加**选择**和**迭代**。更具体地说，我们将添加`if`和`while`指令。在上一章的语言中，动作以直接的方式执行。在本章中，可以在不同的选择之间进行选择，并遍历代码的一部分。

+   我们将添加**变量**。在上一章中，我们可以将值赋给一个名称一次。然而，在本章中，值被赋给可以在程序执行过程中重新赋值的名称。

+   我们将添加**函数**，带有参数和返回值。在上一章中，一个程序由一系列指令组成。在本章中，它是一系列函数。类似于 C++，必须有`main`函数作为执行开始的地方。

+   最后，我们将在从源代码到查看器的过程中添加另一个模块。在上一章中，解析器生成了一系列由查看器显示的动作。在本章中，解析器生成的是一系列**指令**，这些指令随后由**评估器**评估为动作。

+   由于本章的语言支持选择、迭代、变量和函数调用，它开始看起来像一种传统的编程语言。

本章我们将涵盖的主题包括：

+   正如上一章一样，我们将通过查看一个示例来非正式地研究我们的领域特定语言（DSL）的源代码。然而，在这个示例中，我们将使用变量和函数调用，我们还将使用`if`和`while`指令。

+   我们将正式定义我们的语言，使用**语法**。这个语法是上一章语法的扩展。我们将添加函数定义、调用和返回的指令，以及选择（`if`）和迭代（`while`）的指令。

+   当我们定义了语法后，我们将编写**扫描器**。本章的扫描器几乎与上一章的扫描器相同。唯一的区别是我们将添加一些关键字。

+   当我们编写了扫描器后，我们将编写**解析器**。解析器是上一章解析器的扩展，我们添加了用于函数、选择和迭代的函数。然而，上一章的解析器生成了一系列**动作**，这些动作由**查看器**读取并执行。然而，在本章中，解析器生成的是一系列由**评估器**读取的指令。

+   在本章中，下一步是评估者而不是查看者。评估者接收解析器生成的指令序列，并生成一系列动作，这些动作由查看者读取和执行。评估者与将值分配给名称的映射一起工作。有一个**值映射栈**确保每个被调用的函数都得到它自己的新值映射。还有一个**值栈**，用于在评估表达式时存储临时值。最后，还有一个**调用栈**，用于存储函数调用的返回地址。

+   最后，查看者与上一章中的工作方式相同。它遍历评估者生成的动作列表，并在 Qt 小部件中显示图形对象。

# 改进源语言 – 一个例子

让我们看看一个新的例子，其中我们定义并调用一个名为`triangle`的函数，该函数使用不同大小和不同笔触绘制三角形。请注意，函数不必按任何特定顺序出现。

我们首先将`left`和`length`变量设置为`50`。它们持有第一个三角形最左边的*x*坐标和其底边长度。我们还设置了`index`变量为零；其值将在`while`循环中使用：

```cpp
function main() { 
  left = 50; 
  length = 50; 
  index = 0; 
```

我们会一直迭代，直到`index`小于四。注意，在本章中，我们向`Value`类添加了布尔值。当`index`持有偶数值时，我们将画笔样式设置为实线，而当它持有奇数值时，我们将画笔样式设置为虚线。注意，我们已经通过关系表达式和取模运算符（`%`）扩展了语言：

```cpp
  while (index < 4) { 
    if ((index % 2) == 0) { 
      SetPenStyle(SolidLine); 
    } 
    else { 
      SetPenStyle(DashLine); 
    } 
```

我们设置三角形的左上角，并调用`drawTriangle`函数来执行三角形的实际绘制：

```cpp
            topLeft = point(left, 25); 
            call drawTriangle(topLeft, length); 
```

在调用`triangle`之后，我们增加下一个三角形的底边长度和最左边的角：

```cpp
    length = length + 25; 
    left = left + length; 
    index = index + 1; 
  } 
} 
```

在`drawTriangle`函数中，我们调用`getTopRight`和`getBottomMiddle`函数来获取三角形的右上角和底中点。最后，我们通过调用`drawLine`来绘制三角形的三条线：

```cpp
function drawTriangle(topLeft, length) { 
  topRight = call getTopRight(topLeft, length); 
  bottomMiddle = call getBottomMiddle(topLeft, length); 
  drawLine(topLeft, topRight); 
  drawLine(topRight, bottomMiddle); 
  drawLine(bottomMiddle, topLeft); 
} 
```

`getTopRight`函数提取左上角的*x*和*y*坐标，并返回一个*x*坐标增加了三角形底边长度的点：

```cpp
function getTopRight(topLeft, length) { 
  return point(xCoordinate(topLeft) + length, 
               yCoordinate(topLeft)); 
} 
```

`getBottomMiddle`函数也提取左上角的*x*和*y*坐标。然后它计算中间底部的*x*和*y*坐标，并返回`point`：

```cpp
function getBottomMiddle(topLeft, length) { 
  left = xCoordinate(topLeft); 
  top = yCoordinate(topLeft); 
  middle = left + length / 2; 
  bottom = top + length; 
  return point(middle, bottom); 
} 
```

代码执行的输出显示在下述屏幕截图：

![图片](img/2f238186-dcca-40cf-8b90-2a24bbf9da4d.png)

# 改进语法

在本章中，我们将改进我们语言的语法。首先，一个程序由一系列函数组成，而不是指令。技术上，一个程序可以包含零个函数。然而，语义错误会报告`main`函数缺失：

```cpp
program -> functionDefinitionList 
functionDefinitionList -> functionDefinition*
```

函数的定义由关键字 `function`、括号内的名称列表和括号内的指令列表组成。`nameList` 由零个或多个名称组成，名称之间用逗号分隔：

```cpp
functionDefinition -> function name(nameList) { instructionList } 
```

当涉及到指令时，我们添加了函数调用的功能。我们可以直接作为指令调用函数，例如前面的例子中的 `call` `drawTriangle`，或者作为表达式的一部分（`callgetTopRight` 和 `call` `getBottomMiddle`）。

我们还添加了 `while` 指令和带有或不带有 `else` 部分的 `if` 指令。最后，还有块指令：由括号包围的指令列表：

```cpp
instruction -> callExpression ; 
             | while (expression) instruction 
             | if (expression) instruction 
             | if (expression) instruction else instruction 
             | { instructionList } 
             | ... 

callInstruction -> callExpression ; 
```

当涉及到表达式时，唯一的区别是我们增加了函数调用。`expressionList` 是由逗号分隔的零个或多个表达式的列表：

```cpp
primaryExpression -> call name(expressionList) 
                   |
```

# 标记和扫描器

与上一章类似，该语言的最终目标代码是动作，尽管它们是由评估器而不是解析器生成的。`Action` 类与上一章的类相同。同样，`Value` 和 `ViewerWidget` 类，以及颜色和错误处理也是如此。然而，`Token` 和 `Scanner` 类已被扩展。`TokenId` 枚举已被扩展以包含更多的标记标识符。

**Token.h:**

```cpp
class Token { 
  // ... 
  enum TokenId {BlockId, CallId, ElseId, FunctionId, GotoId, 
                IfId, IfNotGotoId, ReturnId, WhileId, // ... 
               }; 
  // ... 
}; 
```

同样，`Scanner` 中的 `init` 已通过关键字扩展。

**Scanner.cpp:**

```cpp
void Scanner::init() { 
  ADD_TO_KEYWORD_MAP(CallId) 
  ADD_TO_KEYWORD_MAP(ElseId) 
  ADD_TO_KEYWORD_MAP(FunctionId) 
  ADD_TO_KEYWORD_MAP(IfId) 
  ADD_TO_KEYWORD_MAP(ReturnId) 
  ADD_TO_KEYWORD_MAP(WhileId) 
// ... 
} 
```

# 解析器

解析器已通过对应于新语法规则的方法进行扩展。此外，本章的解析器不生成动作；相反，它生成 **指令**。这是因为，尽管上一章的源代码包含从开始到结束执行的指令，但本章的源代码包含选择、迭代和可以改变指令流程的函数调用。因此，引入一个中间层是有意义的——解析器生成指令，这些指令被评估为动作。

由于本章的语言支持函数，我们需要 `Function` 类来存储函数。它存储了形式参数的名称和函数的起始地址。

**Function.h:**

```cpp
#ifndef FUNCTION_H 
#define FUNCTION_H 

#include <QtWidgets> 

#include "Value.h" 
#include "Action.h" 

class Function { 
  public: 
    Function() {} 
    Function(const QList<QString>& nameList, int address); 
    const QList<QString>& nameList() const {return m_nameList;} 
    int address() {return m_address;} 

    Function(const Function& function); 
    Function operator=(const Function& function); 

  private: 
    QList<QString> m_nameList; 
    int m_address; 
}; 

#endif // FUNCTION_H 
```

`Function.cpp` 文件包含 `Function` 类方法的定义。

**Function.cpp:**

```cpp
#include "Function.h" 

Function::Function(const QList<QString>& nameList, int address) 
 :m_nameList(nameList), 
  m_address(address) { 
  // Empty. 
} 

Function::Function(const Function& function) 
 :m_nameList(function.m_nameList), 
  m_address(function.m_address) { 
  // Empty. 
} 

Function Function::operator=(const Function& function) { 
  m_nameList = function.m_nameList; 
  m_address = function.m_address; 
  return *this; 
} 
```

由于本章中的解析器生成的是一系列指令而不是动作，因此我们还需要 `Directive` 类来保存指令。在大多数情况下，一个 `Directive` 对象只保存 `TokenId` 枚举的身份标识。然而，在函数调用的例子中，我们需要存储函数名称和实际参数的数量。在函数定义的情况下，我们存储对 `Function` 对象的引用。在由值名称组成的表达式中，我们需要存储名称或值。最后，还有几种跳转指令，在这种情况下，我们需要存储地址。

**Directive.h:**

```cpp
#ifndef DIRECTIVE_H 
#define DIRECTIVE_H 

#include <QtWidgets> 

#include "Token.h" 
#include "Value.h" 
#include "Function.h" 

class Directive { 
  public: 
    Directive(TokenId tokenId); 
    Directive(TokenId tokenId, int address); 
    Directive(TokenId tokenId, const QString& name); 
    Directive(TokenId tokenId, const QString& name, 
              int parameters); 
    Directive(TokenId tokenId, const Value& value); 
    Directive(TokenId tokenId, const Function& function); 

    Directive(const Directive& directive); 
    Directive operator=(const Directive& directive); 

    TokenId directiveId() {return m_directiveId;} 
    const QString& name() {return m_name;} 
    const Value& value() {return m_value;} 
    const Function& function() {return m_function;} 

    int parameters() const {return m_parameters;} 
    int address() const {return m_address;} 
    void setAddress(int address) {m_address = address;} 

  private:          
    TokenId m_directiveId; 
    QString m_name; 
    int m_parameters, m_address; 
    Value m_value; 
    Function m_function; 
}; 

#endif // DIRECTIVE_H 
```

`Directive.cpp`文件包含`Directive`类方法的定义。

**Directive.cpp:**

```cpp
#include "Directive.h"
```

在大多数情况下，我们只创建一个带有指令身份的`Directive`类对象：

```cpp
Directive::Directive(TokenId directiveId) 
 :m_directiveId(directiveId) { 
  // Empty. 
} 
```

跳转指令需要跳转地址：

```cpp
Directive::Directive(TokenId directiveId, int address) 
 :m_directiveId(directiveId), 
  m_address(address) { 
  // Empty. 
} 
```

当给变量赋值时，我们需要变量的名称。然而，我们不需要值，因为它将被存储在栈上。此外，当表达式由一个名称组成时，我们需要存储该名称：

```cpp
Directive::Directive(TokenId directiveId, const QString& name) 
 :m_directiveId(directiveId), 
  m_name(name) { 
  // Empty. 
} 
```

函数调用指令需要函数名称和实际参数的数量：

```cpp
Directive::Directive(TokenId directiveId, const QString& name, 
                     int parameters) 
 :m_directiveId(directiveId), 
  m_name(name), 
  m_parameters(parameters) { 
  // Empty. 
} 
```

当一个表达式仅由一个值组成时，我们只需将值存储在指令中：

```cpp
Directive::Directive(TokenId directiveId, const Value& value) 
 :m_directiveId(directiveId), 
  m_value(value) { 
  // Empty. 
} 
```

最后，在函数定义中，我们存储一个`Function`类的对象：

```cpp
Directive::Directive(TokenId directiveId, 
                     const Function& function) 
 :m_directiveId(directiveId), 
  m_function(function) { 
  // Empty. 
} 
```

`Parser`类已经通过新语法规则的方法扩展：函数定义和`if`、`while`、`call`和`return`指令。

**Parser.h:**

```cpp
// ... 

class Parser { 
  private: 
    void functionDefinitionList(); 
    void functionDefinition(); 

```

`nameList`方法收集函数的形式参数，而`expressionList`收集函数调用的实际参数：

```cpp
            QList<QString> nameList(); 
            int expressionList(); 
```

`callExpression`方法也被添加到`Parser`类中，因为函数可以作为**指令**或表达式的一部分显式调用：

```cpp
    void callExpression(); 
    // ... 
}; 
```

`Parser.cpp`文件包含`Parser`类方法的定义。

本章解析器的`start`方法为`functionDefinitionList`。只要没有达到文件末尾，它就会调用`functionDefinition`。

**Parser.cpp:**

```cpp
void Parser::functionDefinitionList() { 
  while (m_lookAHead.id() != EndOfFileId) { 
    functionDefinition(); 
  } 
} 
```

`functionDefinition`方法解析函数定义。我们首先匹配`function`关键字并存储函数的名称：

```cpp
void Parser::functionDefinition() { 
  match(FunctionId); 
  QString name = m_lookAHead.name(); 
  match(NameId); 
```

函数名称后面跟着括号内的参数名称列表。我们将名称列表存储在`nList`字段中。我们不能将字段命名为`nameList`，因为该名称已经被方法占用：

```cpp
  match(LeftParenthesisId); 
  QList<QString> nList = nameList(); 
  match(RightParenthesisId); 
```

我们将指令列表的当前大小存储为函数的起始地址，创建一个带有名称列表和起始地址的`Function`对象，并将一个带有函数的`Directive`对象添加到指令列表中：

```cpp
   int startAddress = (int) m_directiveList.size(); 
   Function function(nList, startAddress); 
   m_directiveList.push_back(Directive(FunctionId, function)); 
```

名称列表后面跟着一个由括号括起来的指令列表：

```cpp
  match(LeftBracketId); 
  instructionList(); 
  match(RightBracketId); 
```

为了确保函数确实将控制权返回给调用函数，我们添加了一个带有`return`标记身份的`Directive`对象：

```cpp
  m_directiveList.push_back(Directive(ReturnId)); 
```

当函数被定义后，我们检查没有其他函数具有相同的名称：

```cpp
  check(!m_functionMap.contains(name), 
        "function "" + name + "" already defined"); 
```

如果函数名为`"main"`，它是程序的开始函数，并且它不能有参数：

```cpp
  check(!((name == "main") && (nList.size() > 0)), 
        "function "main" cannot have parameters"); 
```

最后，我们将函数添加到`functionMap`中：

```cpp
  m_functionMap[name] = function; 
}
```

`nameList`方法解析括号内逗号分隔的名称列表：

```cpp
QList<QString> Parser::nameList() { 
  QList <QString> nameList; 
```

我们会继续，直到遇到右括号：

```cpp
  while (m_lookAHead.id() != RightParenthesisId) { 
    QString name = m_lookAHead.name(); 
    nameList.push_back(name); 
    match(NameId); 
```

在匹配名称后，我们检查下一个标记是否为右括号。如果是，则表示名称列表的末尾，并中断迭代：

```cpp
    if (m_lookAHead.id() == RightParenthesisId) { 
      break; 
    } 
```

如果下一个标记不是右括号，我们则假设它是一个逗号，匹配它，并继续使用下一个表达式迭代：

```cpp
    match(CommaId); 
  } 
```

最后，在我们返回名称列表之前，我们需要检查名称列表中没有任何名称重复。我们遍历名称列表并将名称添加到一个集合中：

```cpp
  QSet<QString> nameSet; 
  for (const QString& name : nameList) { 
    if (nameSet.contains(name)) { 
      semanticError("parameter "" + name + "" defined twice"); 
    } 

    nameSet.insert(name); 
  } 

  return nameList; 
} 

```

在本章中，`instructionList` 方法看起来略有不同，因为它被放置在指令块内部。我们迭代，直到遇到右括号为止：

```cpp
void Parser::instructionList() { 
  while (m_lookAHead.id() != RightBracketId) { 
    instruction(); 
  } 
}
```

由于函数可以作为指令显式调用，或者作为表达式的一部分，我们只需调用 `callExpression` 并在调用指令的情况下匹配分号：

```cpp
void Parser::instruction() { 
  switch (m_lookAHead.id()) { 
    case CallId: 
      callExpression(); 
      match(SemicolonId); 
      break; 
```

在返回指令中，我们匹配 `return` 关键字并检查它是否后面跟着分号。如果没有跟着分号，我们解析一个表达式，然后假设下一个标记是一个分号。注意，我们不会存储表达式的结果。评估器将在处理过程中稍后将其值放置在栈上：

```cpp
    case ReturnId: 
      match(ReturnId); 

      if (m_lookAHead.id() != SemicolonId) { 
        expression(); 
      } 

      m_directiveList.push_back(Directive(ReturnId)); 
      match(SemicolonId); 
      break; 
```

在 `if` 关键字的情况下，我们匹配它并解析括号内的表达式：

```cpp
    case IfId: { 
        match(IfId); 
        match(LeftParenthesisId); 
        expression(); 
        match(RightParenthesisId); 
```

如果表达式评估为假值，我们将跳过 `if` 表达式之后的指令。因此，我们添加一个 `IfNotGoto` 指令，目的是跳过 `if` 关键字之后的指令：

```cpp
        int ifNotIndex = (int) m_directiveList.size(); 
        m_directiveList.push_back(Directive(IfNotGotoId, 0)); 
        instruction();
```

如果指令后面跟着 `else` 关键字，我们匹配它并添加一个 `Goto` 指令，目的是在 `if` 指令表达式的真值情况下跳过 `else` 部分：

```cpp
        if (m_lookAHead.id() == ElseId) { 
          match(ElseId); 
          int elseIndex = (int) m_directiveList.size(); 
          m_directiveList.push_back(Directive(GotoId, 0)); 
```

然后，我们设置前一个 `IfNotTrue` 指令的跳转地址。如果表达式不为真，程序将跳转到这个点：

```cpp
          m_directiveList[ifNotIndex]. 
            setAddress((int) m_directiveList.size()); 
          instruction(); 
```

另一方面，如果 `if` 指令的表达式为真，程序将跳过 `else` 部分跳转到这个点：

```cpp
          m_directiveList[elseIndex]. 
            setAddress((int) m_directiveList.size()); 
        } 
```

如果 `if` 指令后面没有跟 `else` 关键字，如果表达式不为真，它将跳转到程序的这个点：

```cpp
        else { 
          m_directiveList[ifNotIndex]. 
            setAddress((int) m_directiveList.size()); 
        } 
      } 
      break; 
```

在 `while` 关键字的情况下，我们匹配它并将指令列表的当前索引存储起来，以便程序在每次迭代后跳回到这个点：

```cpp
    case WhileId: { 
        match(WhileId); 
        int whileIndex = (int) m_directiveList.size(); 
```

然后，我们解析表达式及其括号：

```cpp
        match(LeftParenthesisId); 
        expression(); 
        match(RightParenthesisId);
```

如果表达式不为真，我们添加一个 `IfNotGoto` 指令，以便程序跳出迭代：

```cpp
        int ifNotIndex = (int) m_directiveList.size(); 
        m_directiveList.push_back(Directive(IfNotGotoId, 0)); 
        instruction(); 
```

在 `while` 表达式之后的指令后添加一个 `Goto` 指令，这样程序就可以在每次迭代结束时跳回到表达式：

```cpp
        m_directiveList.push_back(Directive(GotoId, whileIndex)); 
```

最后，我们将 `IfNotTrue` 指令的地址设置在 `while` 指令的开始处，这样如果表达式不为真，程序就可以跳转到这个程序点：

```cpp
        m_directiveList[ifNotIndex]. 
          setAddress((int) m_directiveList.size()); 
      } 
      break; 
```

在左括号的情况下，我们有一个由括号包围的指令序列。我们解析这对括号并调用 `instructionList`：

```cpp
    case LeftBracketId: 
      match(LeftBracketId); 
      instructionList(); 
      match(RightBracketId); 
      break; 
```

最后，在名称的情况下，我们有一个赋值操作。我们匹配 `name` 关键字和赋值运算符（`=`），解析表达式，并匹配分号。然后我们向指令列表添加一个包含要赋予值的名称的 `Assign` 对象。请注意，我们不会存储表达式的值，因为它将被评估器推入值栈：

```cpp
    case NameId: { 
        QString name = m_lookAHead.name(); 
        match(NameId); 
        match(AssignId); 
        expression(); 
        match(SemicolonId); 
        m_directiveList.push_back(Directive(AssignId, name)); 
      } 
      break; 

      // ... 
  } 
} 
```

`callExpression` 方法匹配 `call` 关键字，存储函数的名称，解析参数表达式，并将包含调用的 `Directive` 对象添加到指令列表中。请注意，在此点我们不会检查函数是否存在或计算参数的数量，因为函数可能尚未定义。所有类型检查都由评估器在后续过程中处理：

```cpp
void Parser::callExpression() { 
  match(CallId); 
  QString name = m_lookAHead.name(); 
  match(NameId); 
  match(LeftParenthesisId); 
  int size = expressionList(); 
  match(RightParenthesisId); 
  m_directiveList.push_back(Directive(CallId, name, size)); 
} 
```

`expressionList` 方法解析表达式列表。与前面的名称列表情况不同，我们不返回列表本身，只返回其大小。表达式生成自己的指令，其值在后续过程中由评估器存储在栈上：

```cpp
int Parser::expressionList() { 
  int size = 0; 
```

我们会一直迭代，直到遇到右括号：

```cpp
  while (m_lookAHead.id() != RightParenthesisId) { 
    expression(); 
    ++size; 
```

解析表达式后，我们检查下一个标记是否是右括号。如果是，表达式列表就完成了，我们中断迭代：

```cpp
    if (m_lookAHead.id() == RightParenthesisId) { 
      break; 
    } 
```

如果下一个标记不是右括号，我们假设它是一个逗号，匹配它，并继续迭代：

```cpp
    match(CommaId); 
  }
```

最后，经过迭代后，我们返回表达式的数量：

```cpp
  return size; 
} 
```

# 评估器

**评估器**评估一系列指令并生成一个列表，该列表稍后由查看器读取和执行。评估从第一行的指令开始，该指令是跳转到 `main` 函数的起始地址。评估在遇到没有返回地址的 `return` 指令时停止。在这种情况下，我们已经到达 `main` 的末尾，执行应该完成。

评估器针对值栈进行操作。每次评估一个值时，它就会被推入栈中，每次需要值来评估表达式时，它们就会从栈中弹出。

**Evaluator.h:** 

```cpp
#ifndef EVALUATOR_H 
#define EVALUATOR_H 

#include <QtWidgets> 

#include "Error.h" 
#include "Directive.h" 
#include "Action.h" 
#include "Function.h" 
```

`Evaluator` 类的构造函数使用函数映射评估指令列表：

```cpp
class Evaluator { 
  public: 
    Evaluator(const QList<Directive>& directiveList, 
              QList<Action>& actionList, 
              QMap<QString,Function> functionMap);
```

`checkType` 和 `evaluate` 方法与上一章相同。它们已从 `Parser` 移至 `Evaluator`。`checkType` 方法检查与标记关联的表达式是否具有正确的类型，而 `evaluate` 方法评估表达式：

```cpp
  private: 
    void checkType(TokenId tokenId, const Value& value); 
    void checkType(TokenId tokenId, const Value& leftValue, 
                   const Value& rightValue); 

    Value evaluate(TokenId tokenId, const Value& value); 
    Value evaluate(TokenId tokenId, const Value& leftValue, 
                   const Value& rightValue); 
```

当评估一个表达式时，其值会被推入 `m_valueStack`。当一个变量被赋予一个值时，其名称和值会被存储在 `m_valueMap` 中。请注意，在本章中，一个值可以被赋予一个变量多次。当一个函数调用另一个函数时，调用函数的值映射会被推入 `m_valueMapStack` 以给被调用函数提供一个全新的值映射，并且返回地址会被推入 `m_returnAddressStack`：

```cpp
    QStack<Value> m_valueStack; 
    QMap<QString,Value> m_valueMap; 
    QStack<QMap<QString,Value>> m_valueMapStack; 
    QStack<int> m_returnAddressStack; 
}; 

#endif // EVALUATOR_H 
```

`Evaluator.cpp`文件包含`Evaluator`类的方法定义：

**Evaluator.cpp:** 

```cpp
#include <CAssert> 
using namespace std; 

#include "Error.h" 
#include "Evaluator.h" 
```

`Evaluator`类的构造函数可以被视为评估器的核心。

构造函数中的`directiveIndex`字段是指令列表中当前`Directive`对象的索引。通常，它会在每次迭代中增加。然而，由于`if`或`while`指令以及函数调用和返回，它也可以被赋予不同的值：

```cpp
Evaluator::Evaluator(const QList<Directive>& directiveList, 
                     QList<Action>& actionList, 
                     QMap<QString,Function> functionMap) { 
  int directiveIndex = 0; 

  while (true) { 
    Directive directive = directiveList[directiveIndex]; 
    TokenId directiveId = directive.directiveId(); 
```

当调用函数时，我们首先在函数映射中查找函数名，如果没有找到则报告语义错误。然后我们检查实际参数的数量是否等于形式参数的数量（`Function`对象中名称列表的大小）：

```cpp
    switch (directiveId) { 
      case CallId: { 
          QString name = directive.name(); 
          check(functionMap.contains(name), 
                "missing function: "" + name + """); 
          Function function = functionMap[name]; 
          check(directive.parameters() == 
                function.nameList().size(), 
                "invalid number of parameters"); 
```

当我们调用函数时，我们在返回地址栈上推送下一个指令的索引，以便被调用的函数可以返回到正确的地址。我们在值映射栈上推送调用函数的值映射，以便在调用后检索它。然后我们清除值映射，以便它对被调用的函数是新鲜的。最后，我们将指令索引设置为被调用函数的起始地址，这会将控制权转移到被调用函数的开始处。请注意，我们对实际参数表达式没有做任何事情。它们已经被评估，并且它们的值被推送到值栈上：

```cpp
          m_returnAddressStack.push(directiveIndex + 1); 
          m_valueMapStack.push(m_valueMap); 
          m_valueMap.clear(); 
          directiveIndex = function.address(); 
        } 
        break;
```

函数开始时，我们为每个参数弹出值栈，并将每个参数名与其值在值映射中关联。记住，在调用函数之前已经评估了参数表达式，并且它们的值被推送到值栈上。还要记住，第一个参数首先被推送到栈上，并且位于其他参数的下方，这就是为什么我们以相反的顺序分配参数。最后，记住调用函数时值映射被推送到值映射栈上，值栈在函数调用期间被清除，因此在函数开始时当前值映射为空：

```cpp
      case FunctionId: { 
          const Function& function = directive.function(); 
          const QList<QString>& nameList = function.nameList(); 

          for (int listIndex = ((int) nameList.size() - 1); 
               listIndex >= 0; --listIndex) { 
            const QString& name = nameList[listIndex]; 
            m_valueMap[name] = m_valueStack.pop(); 
          } 
        } 
        ++directiveIndex; 
        break; 
```

当从函数返回时，我们首先检查返回地址栈是否为空。如果不为空，我们执行正常的函数返回。通过弹出值映射栈，我们恢复调用函数的值映射。我们还通过弹出返回地址栈将指令索引设置为函数调用后的地址：

```cpp
      case ReturnId: 
        if (!m_returnAddressStack.empty()) { 
          m_valueMap = m_valueMapStack.pop(); 
          directiveIndex = m_returnAddressStack.pop(); 
        } 
```

然而，如果返回地址栈为空，我们有一个特殊情况——我们已经到达了`main`函数的末尾。在这种情况下，我们不应返回到调用函数（没有调用函数）。相反，我们应通过调用返回来完成评估器的执行。记住，我们处于`Evaluator`类的构造函数中，并且从构造函数返回：

```cpp
        else { 
          return; 
        } 
        break;
```

`IfNotGoto` 指令是在解析 `if` 或 `while` 指令时由解析器添加的。我们弹出值栈；如果值为假，我们通过调用指令的 `address` 方法来设置指令索引以执行跳转。记住，在本章中，我们已经向 `Value` 类添加了布尔值：

```cpp
      case IfNotGotoId: { 
          Value value = m_valueStack.pop(); 

          if (!value.booleanValue()) { 
            directiveIndex = directive.address(); 
          } 
```

如果值为真，我们不执行跳转；我们只是简单地增加指令索引：

```cpp
          else { 
            ++directiveIndex; 
          } 
        } 
        break; 
```

`Goto` 指令执行无条件跳转；我们只需设置新的指令索引。由于 `IfNotGoto` 和 `Goto` 指令是由解析器生成的，我们不需要执行任何类型检查：

```cpp
      case GotoId: 
        directiveIndex = directive.address(); 
        break; 
```

设置指令的工作方式与上一章中的解析器相对应。在早期指令的评估过程中，表达式的值已经推入值栈。我们从值栈中弹出值并检查它是否包含正确的类型。然后我们将带有值的操作添加到操作列表中并增加指令索引：

```cpp
      case SetPenColorId: 
      case SetPenStyleId: 
      case SetBrushColorId: 
      case SetBrushStyleId: 
      case SetFontId: 
      case SetHorizontalAlignmentId: 
      case SetVerticalAlignmentId: { 
          Value value = m_valueStack.pop(); 
          checkType(directiveId, value); 
          actionList.push_back(Action(directiveId, value)); 
          ++directiveIndex; 
        } 
        break;
```

此外，绘图指令与上一章中的解析器类似。它们的第一个和第二个值以相反的顺序弹出，因为第一个值首先被推入，因此位于栈中的第二个值下方。然后我们检查值是否具有正确的类型，将操作添加到操作列表中，并增加指令索引：

```cpp
      case DrawLineId: 
      case DrawRectangleId: 
      case DrawEllipseId: 
      case DrawTextId: { 
          Value secondValue = m_valueStack.pop(); 
          Value firstValue = m_valueStack.pop(); 
          checkType(directiveId, firstValue, secondValue); 
          actionList.push_back(Action(directiveId, firstValue, 
                                      secondValue)); 
          ++directiveIndex; 
        } 
        break; 
```

赋值指令将名称与值映射中的值关联起来。请注意，如果名称已经与一个值关联，则之前的值将被覆盖。另外请注意，值映射是当前函数的局部变量，潜在的调用函数有自己的值映射推入值映射栈：

```cpp
      case AssignId: { 
          Value value = m_valueStack.pop(); 
          m_valueMap[directive.name()] = value; 
          ++directiveIndex; 
        } 
        break; 
```

在包含一个值的表达式中，其值从栈中弹出，检查其类型，并计算表达式的结果值并将其推入值栈。最后，增加指令索引：

```cpp
      case XCoordinateId: 
      case YCoordinateId: { 
          Value value = m_valueStack.pop(); 
          checkType(directiveId, value); 
          Value resultValue = evaluate(directiveId, value); 
          m_valueStack.push(resultValue); 
          ++directiveIndex; 
        } 
        break;
```

在包含两个值的表达式中，其第一个和第二个值从栈中弹出（顺序相反），检查它们的类型，并计算表达式的结果值并将其推入值栈。最后，增加指令索引：

```cpp
      case AddId: 
      case SubtractId: 
      case MultiplyId: 
      case DivideId: 
      case PointId: { 
          Value rightValue = m_valueStack.pop(); 
          Value leftValue = m_valueStack.pop(); 
          checkType(directiveId, leftValue, rightValue); 
          Value resultValue = 
            evaluate(directiveId, leftValue, rightValue); 
          m_valueStack.push(resultValue); 
          ++directiveIndex; 
        } 
        break; 
```

在颜色表达式中，红色、绿色和蓝色组件值从值栈中弹出（顺序相反），检查它们的类型，并将结果颜色推入值栈。最后，增加指令索引：

```cpp
      case ColorId: { 
          Value blueValue = m_valueStack.pop(); 
          Value greenValue = m_valueStack.pop(); 
          Value redValue = m_valueStack.pop(); 
          checkColorType(redValue, greenValue, blueValue); 
          QColor color(redValue.numericalValue(), 
                       greenValue.numericalValue(), 
                       blueValue.numericalValue()); 
          m_valueStack.push(Value(color)); 
          ++directiveIndex; 
        } 
        break; 
```

在字体表达式中，名称和大小值从值栈中弹出（顺序相反）并检查它们的类型。然后将结果字体推入值栈并增加指令索引：

```cpp
      case FontId: { 
          Value sizeValue = m_valueStack.pop(); 
          Value nameValue = m_valueStack.pop(); 
          checkFontType(nameValue, sizeValue, 
                        boldValue, italicValue); 
          QFont font(nameValue.stringValue(), 
                     sizeValue.numericalValue()); 
          m_valueStack.push(Value(font)); 
          ++directiveIndex; 
        } 
        break; 
```

在名称的情况下，我们查找其值并将其推入值栈，并增加指令索引。如果没有与名称关联的值，则报告语义错误：

```cpp
      case NameId: { 
          QString name = directive.name(); 
          check(m_valueMap.contains(name), 
                "unknown name: "" + name +"""); 
          m_valueStack.push(m_valueMap[name]); 
          ++directiveIndex; 
        } 
        break; 
```

最后，当我们有一个值时，我们只需将其推入值栈并增加指令索引：

```cpp
      case ValueId: 
        m_valueStack.push(directive.value()); 
        ++directiveIndex; 
        break; 
    } 
  } 
} 
```

# 主函数

最后，`main` 函数几乎与上一个函数相同。

**Main.cpp:**

```cpp
#include <QApplication> 
#include <QMessageBox> 
#include <IOStream> 
using namespace std; 

#include "Action.h" 
#include "Error.h" 
#include "Scanner.h" 
#include "Parser.h" 
#include "Evaluator.h" 
#include "ViewerWidget.h" 

int main(int argc, char *argv[]) { 
  Scanner::init(); 
  QApplication application(argc, argv); 

  try { 
    QString path = "C:\Input.dsl"; 

    QFile file(path); 
    if (!file.open(QIODevice::ReadOnly)) { 
      error("Cannot open file "" + path + "" for reading."); 
    } 

    QString buffer(file.readAll()); 
    Scanner scanner(buffer); 
```

唯一的不同之处在于，解析器生成一系列指令而不是动作，以及一个函数映射，这些被发送到评估器，评估器生成最终的动作列表，该列表被读取和执行以显示图形对象：

```cpp
    QList<Directive> directiveList; 
    QMap<QString,Function> functionMap; 
    Parser(scanner, directiveList, functionMap); 

    QList<Action> actionList; 
    Evaluator evaluator(directiveList, actionList, functionMap); 

    ViewerWidget mainWidget(actionList); 
    mainWidget.show(); 
    return application.exec(); 
  } 
  catch (exception e) { 
    QMessageBox messageBox(QMessageBox::Information, 
                           QString("Error"), QString(e.what())); 
    messageBox.exec(); 
  } 
}
```

# 摘要

在本章中，我们改进了我们在上一章开始工作的领域特定语言（DSL）。我们添加了选择、迭代、变量和函数调用。我们还添加了评估器，它接收解析器生成的指令，并生成由查看器读取和执行的动作。当指令正在执行时，表达式的值存储在栈上，分配给名称的值存储在映射中，函数调用的返回地址存储在栈上。

这就是最后一章了，希望你喜欢这本书！
