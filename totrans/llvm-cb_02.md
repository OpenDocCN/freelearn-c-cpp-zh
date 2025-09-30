# 第二章。编写前端步骤

在本章中，我们将涵盖以下配方：

+   定义一个 TOY 语言

+   实现一个词法分析器

+   定义抽象语法树

+   实现一个解析器

+   解析简单表达式

+   解析二进制表达式

+   调用解析驱动程序

+   在我们的 TOY 语言上运行词法分析和解析器

+   为每个 AST 类定义 IR 代码生成方法

+   为表达式生成 IR 代码

+   为函数生成 IR 代码

+   添加 IR 优化支持

# 简介

在本章中，你将了解如何编写一种语言的前端。通过使用自定义的 TOY 语言，你将获得如何编写词法分析和解析器以及如何从前端生成的**抽象语法树**（**AST**）生成 IR 代码的配方。

# 定义一个 TOY 语言

在实现词法分析和解析器之前，需要首先确定语言的语法和语法规则。在本章中，使用 TOY 语言来演示如何实现词法分析和解析器。本配方的目的是展示如何浏览一种语言。为此，要使用的 TOY 语言简单但有意义。

一种语言通常有一些变量、一些函数调用、一些常量等等。为了保持简单，我们考虑的 TOY 语言只有 32 位整型常量 A，一个不需要声明其类型（如 Python，与 C/C++/Java 不同，后者需要类型声明）的变量。

## 如何做到这一点...

语法可以定义为如下（生产规则定义如下，非终结符在**左侧**（**LHS**）上，终结符和非终结符的组合在**右侧**（**RHS**）上；当遇到 LHS 时，它将产生生产规则中定义的适当 RHS）：

1.  一个数值表达式将给出一个常数：

    ```cpp
    numeric_expr := number
    ```

1.  括号表达式将在一个开括号和一个闭括号之间有一个表达式：

    ```cpp
    paran_expr := '(' expression ')'
    ```

1.  标识符表达式将产生一个标识符或一个函数调用：

    ```cpp
    identifier_expr
    := identifier
    := identifier '('expr_list ')'
    ```

1.  如果标识符 `_expr` 是一个函数调用，它将没有参数或由逗号分隔的参数列表：

    ```cpp
    expr_list
    := (empty)
    := expression (',' expression)*
    ```

1.  将有一些原始表达式，语法的起点，它可能产生一个标识符表达式、一个数值表达式或一个括号表达式：

    ```cpp
    primary := identifier_expr
    :=numeric_expr
    :=paran_expr
    ```

1.  一个表达式可以导致一个二进制表达式：

    ```cpp
    expression := primary binoprhs
    ```

1.  右侧的二进制运算可以产生二进制运算符和表达式的组合：

    ```cpp
    binoprhs := ( binoperator primary )*
    binoperators := '+'/'-'/'*'/'/'
    ```

1.  函数声明可以有如下语法：

    ```cpp
    func_decl := identifier '(' identifier_list ')'
    identifier_list := (empty)
                            := (identifier)*
    ```

1.  函数定义通过一个`def`关键字后跟一个函数声明和一个定义其体的表达式来区分：

    ```cpp
    function_defn := 'def' func_decl expression
    ```

1.  最后，将有一个顶层表达式，它将产生一个表达式：

    ```cpp
    toplevel_expr := expression  
    ```

基于先前定义的语法的 TOY 语言的一个示例可以写成如下：

```cpp
def foo (x , y)
x +y * 16
```

由于我们已经定义了语法，下一步是为其编写词法分析和解析器。

# 实现一个词法分析器

Lexer 是程序编译的第一个阶段的组成部分。Lexer 将程序中的输入流进行标记化。然后 parser 消费这些标记以构建 AST。标记化的语言通常是上下文无关语言。标记是一组一个或多个字符的字符串，这些字符作为一个整体是有意义的。从字符输入流中形成标记的过程称为标记化。某些分隔符用于识别单词组作为标记。存在一些 lexer 工具来自动进行词法分析，例如 **LEX**。在以下过程中演示的 TOY lexer 是一个使用 C++ 编写的 hand-written lexer。

## 准备中

我们必须对配方中定义的 TOY 语言有一个基本理解。创建一个名为 `toy.cpp` 的文件，如下所示：

```cpp
$ vim toy.cpp
```

接下来的所有代码将包含 lexer、parser 和代码生成逻辑。

## 如何做到这一点...

在实现 lexer 时，定义标记类型以对输入字符串流进行分类（类似于自动机的状态）。这可以通过使用 **枚举** （**enum**） 类型来完成：

1.  按如下方式打开 `toy.cpp` 文件：

    ```cpp
    $ vim toy.cpp
    ```

1.  在 `toy.cpp` 文件中按如下方式编写 `enum`：

    ```cpp
    enum Token_Type {
    EOF_TOKEN = 0,
    NUMERIC_TOKEN,
    IDENTIFIER_TOKEN,
    PARAN_TOKEN,
    DEF_TOKEN
    };
    ```

    以下是为前一个示例定义的术语列表：

    +   `EOF_TOKEN`: 它表示文件结束

    +   `NUMERIC_TOKEN:` 当前标记是数值类型

    +   `IDENTIFIER_TOKEN:` 当前标记是标识符

    +   `PARAN_TOKEN:` 当前标记是括号

    +   `DEF_TOKEN`: 当前标记 `def` 表示随后的内容是一个函数定义

1.  要存储数值，可以在 `toy.cpp` 文件中定义一个静态变量，如下所示：

    ```cpp
    static int Numeric_Val;
    ```

1.  要存储 `Identifier` 字符串名称，可以在 `toy.cpp` 文件中定义一个静态变量，如下所示：

    ```cpp
        static std::string Identifier_string;
    ```

1.  现在可以在 `toy.cpp` 文件中使用库函数如 `isspace()`、`isalpha()` 和 `fgetc()` 定义 lexer 函数，如下所示：

    ```cpp
    static int get_token() {
      static int LastChar = ' ';

      while(isspace(LastChar))
      LastChar = fgetc(file);

      if(isalpha(LastChar)) {
        Identifier_string = LastChar;
        while(isalnum((LastChar = fgetc(file))))
        Identifier_string += LastChar;

        if(Identifier_string == "def")
        return DEF_TOKEN;
        return IDENTIFIER_TOKEN;
      }

      if(isdigit(LastChar)) {
        std::string NumStr;
        do {
          NumStr += LastChar;
          LastChar = fgetc(file);
        } while(isdigit(LastChar));

        Numeric_Val = strtod(NumStr.c_str(), 0);
        return NUMERIC_TOKEN;
      }

      if(LastChar == '#') {
        do LastChar = fgetc(file);
        while(LastChar != EOF && LastChar != '\n'
        && LastChar != '\r');

        if(LastChar != EOF) return get_token();
      }

      if(LastChar == EOF) return EOF_TOKEN;

      int ThisChar = LastChar;
      LastChar = fgetc(file);
      return ThisChar;
    }
    ```

## 它是如何工作的...

之前定义的示例 TOY 语言如下：

```cpp
def foo (x , y)
x + y * 16
```

lexer 将获取前面的程序作为输入。它将遇到 `def` 关键字并确定随后的内容是一个定义标记，因此返回枚举值 `DEF_TOKEN`。之后，它将遇到函数定义及其参数。然后，有一个涉及两个二元运算符、两个变量和一个数值常数的表达式。以下食谱演示了这些是如何存储在数据结构中的。

## 参见

+   在 [`clang.llvm.org/doxygen/Lexer_8cpp_source.html`](http://clang.llvm.org/doxygen/Lexer_8cpp_source.html) 查看为 C++ 语言编写的更复杂和详细的 hand-written lexer。

# 定义抽象语法树

AST 是编程语言源代码的抽象语法结构的树表示。编程结构（如表达式、流程控制语句等）的 AST 被分组为操作符和操作数。AST 表示编程结构之间的关系，而不是它们由语法生成的途径。AST 忽略了诸如标点符号和分隔符等不重要的编程元素。AST 通常包含每个元素的附加属性，这些属性在后续编译阶段很有用。源代码的位置是这样一个属性，当根据语法确定源代码的正确性时，如果遇到错误，可以使用它来抛出错误行号（位置、行号、列号等，以及其他相关属性存储在 Clang 前端 C++的`SourceManager`类对象中）。

在语义分析期间，AST 被大量使用，编译器检查程序元素和语言的正确使用。编译器在语义分析期间还基于 AST 生成符号表。对树的完整遍历允许验证程序的正确性。验证正确性后，AST 作为代码生成的基。

## 准备工作

到现在为止，我们必须已经运行了词法分析器以获取用于生成 AST 的标记。我们打算解析的语言包括表达式、函数定义和函数声明。我们再次有各种类型的表达式——变量、二元运算符、数值表达式等。

## 如何实现...

要定义 AST 结构，请按照以下步骤进行：

1.  按如下方式打开`toy.cpp`文件：

    ```cpp
    $ vi toy.cpp
    ```

    在词法分析代码下方定义 AST。

1.  解析表达式的`base`类可以定义为以下内容：

    ```cpp
    class BaseAST {
      public :
      virtual ~BaseAST();
    };
    ```

    然后，为要解析的每种表达式类型定义几个派生类。

1.  变量表达式的 AST 类可以定义为以下内容：

    ```cpp
    class VariableAST  : public BaseAST{
      std::string Var_Name;  
    // string object to store name of
    // the variable.
      public:
      VariableAST (std::string &name) : Var_Name(name) {}  // ..// parameterized constructor of variable AST class to be initialized with the string passed to the constructor.
    };
    ```

1.  语言有一些数值表达式。此类数值表达式的`AST`类可以定义为以下内容：

    ```cpp
    class NumericAST : public BaseAST {
      int numeric_val;
      public :
      NumericAST (intval) :numeric_val(val)  {}
    };
    ```

1.  对于涉及二元运算的表达式，`AST`类可以定义为以下内容：

    ```cpp
    Class BinaryAST : public BaseAST {
      std::string Bin_Operator;  // string object to store
      // binary operator
      BaseAST  *LHS, *RHS;  // Objects used to store LHS and   
    // RHS of a binary Expression. The LHS and RHS binary   
    // operation can be of any type, hence a BaseAST object 
    // is used to store them.
      public:
      BinaryAST (std::string op, BaseAST *lhs, BaseAST *rhs ) :
      Bin_Operator(op), LHS(lhs), RHS(rhs) {}  // Constructor
      //to initialize binary operator, lhs and rhs of the binary
      //expression.
    };
    ```

1.  函数声明的`AST`类可以定义为以下内容：

    ```cpp
    class FunctionDeclAST {
      std::string Func_Name;
      std::vector<std::string> Arguments;
      public:
      FunctionDeclAST(const std::string &name, const       std::vector<std::string> &args) :
      Func_Name(name), Arguments(args) {};
    };
    ```

1.  函数定义的`AST`类可以定义为以下内容：

    ```cpp
    class FunctionDefnAST {
      FunctionDeclAST *Func_Decl;
      BaseAST* Body;
      public:
      FunctionDefnAST(FunctionDeclAST *proto, BaseAST *body) :
      Func_Decl(proto), Body(body) {}
    };
    ```

1.  函数调用的`AST`类可以定义为以下内容：

    ```cpp
    class FunctionCallAST : public BaseAST {
      std::string Function_Callee;
      std::vector<BaseAST*> Function_Arguments;
      public:
      FunctionCallAST(const std::string &callee, std::vector<BaseAST*> &args) :
      Function_Callee(callee), Function_Arguments(args) {}
    };
    ```

AST 的基本骨架现在已准备好使用。

## 它是如何工作的…

AST 充当存储由词法分析器提供的各种信息的数据结构。这些信息在解析器逻辑中生成，并根据正在解析的标记类型填充 AST。

## 参见

+   生成 AST 后，我们将实现解析器，然后我们将看到同时调用词法分析和解析器的示例。有关 Clang 中 C++ 的更详细 AST 结构，请参阅：[`clang.llvm.org/docs/IntroductionToTheClangAST.html`](http://clang.llvm.org/docs/IntroductionToTheClangAST.html)。

# 实现解析器

解析器根据语言的语法规则对代码进行语法分析。解析阶段确定输入代码是否可以根据定义的语法形成标记串。在此阶段构建一个解析树。解析器定义函数将语言组织成称为 AST 的数据结构。本食谱中定义的解析器使用递归下降解析技术，这是一种自顶向下的解析器，并使用相互递归的函数来构建 AST。

## 准备工作

我们必须拥有自定义的语言，即在本例中的 TOY 语言，以及由词法分析器生成的标记流。

## 如何操作...

在我们的 TOY 解析器中定义一些基本值持有者，如下所示：

1.  按照以下步骤打开 `toy.cpp` 文件：

    ```cpp
    $ vi toy.cpp
    ```

1.  定义一个全局静态变量以保存来自词法分析器的当前标记，如下所示：

    ```cpp
    static int Current_token;
    ```

1.  按照以下方式定义一个从输入流中获取下一个标记的函数：

    ```cpp
    static void next_token() {
      Current_token =  get_token();
    }
    ```

1.  下一步是定义使用上一节中定义的 AST 数据结构进行表达式解析的函数。

1.  定义一个通用函数，根据词法分析器确定的标记类型调用特定的解析函数，如下所示：

    ```cpp
    static BaseAST* Base_Parser() {
      switch (Current_token) {
        default: return 0;
        case IDENTIFIER_TOKEN : return identifier_parser();
        case NUMERIC_TOKEN : return numeric_parser();
        case '(' : return paran_parser();
      }
    }
    ```

## 它是如何工作的...

输入流被标记化并传递给解析器。`Current_token` 保存要处理的标记。在此阶段已知标记的类型，并调用相应的解析函数以初始化 AST。

## 参见

+   在接下来的几个食谱中，你将学习如何解析不同的表达式。有关 Clang 中实现的 C++ 语言的更详细解析，请参阅：[`clang.llvm.org/doxygen/classclang_1_1Parser.html`](http://clang.llvm.org/doxygen/classclang_1_1Parser.html)。

# 解析简单表达式

在本食谱中，你将学习如何解析一个简单的表达式。一个简单的表达式可能包括数值、标识符、函数调用、函数声明和函数定义。对于每种类型的表达式，都需要定义单独的解析逻辑。

## 准备工作

我们必须拥有自定义的语言——在本例中即为 TOY 语言，以及由词法分析器生成的标记流。我们已经在上面定义了抽象语法树（AST）。进一步地，我们将解析表达式并调用每种类型表达式的 AST 构造函数。

## 如何操作...

要解析简单表达式，按照以下代码流程进行：

1.  按照以下步骤打开 `toy.cpp` 文件：

    ```cpp
    $ vi toy.cpp
    ```

    我们已经在 `toy.cpp` 文件中实现了词法分析器逻辑。接下来的代码需要附加在 `toy.cpp` 文件中的词法分析器代码之后。

1.  按照以下方式定义用于数值表达式的 `parser` 函数：

    ```cpp
    static BaseAST *numeric_parser() {
      BaseAST *Result = new NumericAST(Numeric_Val);
      next_token();
      return Result;
    }
    ```

1.  定义标识符表达式的`parser`函数。请注意，标识符可以是变量引用或函数调用。它们通过检查下一个标记是否为`(`来区分。这如下实现：

    ```cpp
    static BaseAST* identifier_parser() {
      std::string IdName = Identifier_string;

      next_token();

      if(Current_token != '(')
      return new VariableAST(IdName);

      next_token();

      std::vector<BaseAST*> Args;
      if(Current_token != ')') {
        while(1) {
          BaseAST* Arg = expression_parser();
          if(!Arg) return 0;
          Args.push_back(Arg);

          if(Current_token == ')') break;

          if(Current_token != ',')
          return 0;
          next_token();
        }
      }
      next_token();

      return new FunctionCallAST(IdName, Args);
    }
    ```

1.  定义函数声明`parser`函数如下：

    ```cpp
    static FunctionDeclAST *func_decl_parser() {
      if(Current_token != IDENTIFIER_TOKEN)
      return 0;

      std::string FnName = Identifier_string;
      next_token();

      if(Current_token != '(')
      return 0;

      std::vector<std::string> Function_Argument_Names;
      while(next_token() == IDENTIFIER_TOKEN)
      Function_Argument_Names.push_back(Identifier_string);
      if(Current_token != ')')
      return 0;

      next_token();

      return new FunctionDeclAST(FnName, Function_Argument_Names);
    }
    ```

1.  定义函数定义的`parser`函数如下：

    ```cpp
    static FunctionDefnAST *func_defn_parser() {
      next_token();
      FunctionDeclAST *Decl = func_decl_parser();
      if(Decl == 0) return 0;

      if(BaseAST* Body = expression_parser())
      return new FunctionDefnAST(Decl, Body);
      return 0;
    }
    ```

    注意，在前面代码中使用的名为`expression_parser`的函数用于解析表达式。该函数可以定义为以下内容：

    ```cpp
    static BaseAST* expression_parser() {
      BaseAST *LHS = Base_Parser();
      if(!LHS) return 0;
      return binary_op_parser(0, LHS);
    }
    ```

## 它是如何工作的…

如果遇到数字标记，将调用数字表达式的构造函数，并返回由解析器返回的 AST 对象，用数字数据填充 AST 的数值部分。

类似地，对于标识符表达式，解析的数据将是变量或函数调用。对于函数声明和定义，将解析函数名和函数参数，并调用相应的 AST 类构造函数。

# 解析二进制表达式

在这个菜谱中，你将学习如何解析二进制表达式。

## 准备工作

我们必须有一个自定义定义的语言，即在本例中的玩具语言，以及由词法分析器生成的标记流。二进制表达式解析器需要二进制运算符的优先级来确定左右顺序。可以使用 STL map 来定义二进制运算符的优先级。

## 如何做到这一点…

要解析二进制表达式，请按照以下代码流程进行：

1.  按如下方式打开`toy.cpp`文件：

    ```cpp
    $ vi toy.cpp
    ```

1.  在`toy.cpp`文件中声明一个用于运算符优先级的`map`，以在全局范围内存储优先级，如下所示：

    ```cpp
    static std::map<char, int>Operator_Precedence;
    ```

    用于演示的 TOY 语言有 4 个运算符，运算符的优先级定义为`-` < `+` < `/` < `*`。

1.  可以在全局范围内定义一个初始化优先级的函数，即存储`map`中的优先级值，如下所示：

    ```cpp
    static void init_precedence() {
      Operator_Precedence['-'] = 1;
      Operator_Precedence['+'] = 2;
      Operator_Precedence['/'] = 3;
      Operator_Precedence['*'] = 4;
    }
    ```

1.  可以定义一个辅助函数来返回二进制运算符的优先级，如下所示：

    ```cpp
    static int getBinOpPrecedence() {
      if(!isascii(Current_token))
    return -1;

      int TokPrec = Operator_Precedence[Current_token];
      if(TokPrec <= 0) return -1;
      return TokPrec;
    }
    ```

1.  现在，可以定义如下所示的`binary`运算符解析器：

    ```cpp
    static BaseAST* binary_op_parser(int Old_Prec, BaseAST *LHS) {
      while(1) {
        int Operator_Prec = getBinOpPrecedence();

        if(Operator_Prec < Old_Prec)
        return LHS;

        int BinOp = Current_token;
        next_token();

        BaseAST* RHS = Base_Parser();
        if(!RHS) return 0;

        int Next_Prec = getBinOpPrecedence();
        if(Operator_Prec < Next_Prec) {
          RHS = binary_op_parser(Operator_Prec+1, RHS);
          if(RHS == 0) return 0;
        }

        LHS = new BinaryAST(std::to_string(BinOp), LHS, RHS);
      }
    }
    ```

    在这里，通过检查当前运算符的优先级与旧运算符的优先级，并根据二进制运算符的左右两边（LHS 和 RHS）来决定结果。请注意，由于右边的表达式可以是一个表达式而不是单个标识符，因此二进制运算符解析器是递归调用的。

1.  可以定义一个用于括号的`parser`函数，如下所示：

    ```cpp
    static BaseAST* paran_parser() {
      next_token();
      BaseAST* V = expression_parser();
      if (!V) return 0;

      if(Current_token != ')')
        return 0;
      return V;
    }
    ```

1.  一些作为这些`parser`函数包装器的顶级函数可以定义为以下内容：

    ```cpp
    static void HandleDefn() {
      if (FunctionDefnAST *F = func_defn_parser()) {
        if(Function* LF = F->Codegen()) {
      }
      }
      else {
        next_token();
      }
    }

    static void HandleTopExpression() {
      if(FunctionDefnAST *F = top_level_parser()) {
        if(Function *LF = F->Codegen()) {
      }
      }
      else {
        next_token();
      }
    }
    ```

## 参见

+   本章中所有剩余的菜谱都与用户对象相关。有关表达式的详细解析和 C++解析，请参阅：[`clang.llvm.org/doxygen/classclang_1_1Parser.html`](http://clang.llvm.org/doxygen/classclang_1_1Parser.html)。

# 调用解析驱动程序

在这个菜谱中，你将学习如何在我们的 TOY 解析器的`main`函数中调用解析函数。

## 如何做到这一点…

要调用驱动程序以开始解析，请定义如以下所示的驱动函数：

1.  打开`toy.cpp`文件：

    ```cpp
    $ vi toy.cpp
    ```

1.  从主函数中调用的`Driver`函数，现在可以定义解析器如下：

    ```cpp
    static void Driver() {
      while(1) {
        switch(Current_token) {
        case EOF_TOKEN : return;
        case ';' : next_token(); break;
        case DEF_TOKEN : HandleDefn(); break;
        default : HandleTopExpression(); break;
      }
      }
    }
    ```

1.  可以定义如下`main()`函数来运行整个程序：

    ```cpp
    int main(int argc, char* argv[]) {
      LLVMContext &Context = getGlobalContext();
      init_precedence();
      file = fopen(argv[1], "r");
      if(file == 0) {
        printf("Could not open file\n");
      }
      next_token();
      Module_Ob = new Module("my compiler", Context);
      Driver();
      Module_Ob->dump();
          return 0;
    }
    ```

## 它是如何工作的…

主函数负责调用词法分析和解析器，以便它们可以作用于输入到编译器前端的代码片段。从主函数中，调用驱动函数以启动解析过程。

## 参见

+   有关 Clang 中 c++解析的主函数和驱动函数的工作原理的详细信息，请参阅[`llvm.org/viewvc/llvm-project/cfe/trunk/tools/driver/cc1_main.cpp`](http://llvm.org/viewvc/llvm-project/cfe/trunk/tools/driver/cc1_main.cpp)

# 在我们的玩具语言上运行词法分析和解析器

现在我们已经为我们的 TOY 语言语法定义了完整的词法分析和解析器，是时候在示例 TOY 语言上运行它了。

## 准备工作

要做到这一点，您应该了解 TOY 语言语法和本章的所有先前的配方。

## 如何做…

如以下所示，在 TOY 语言上运行并测试词法分析和解析器：

1.  第一步是将`toy.cpp`程序编译成可执行文件：

    ```cpp
    $ clang++ toy.cpp  -O3 -o toy
    ```

1.  `toy`可执行文件是我们的 TOY 编译器前端。要解析的`toy`语言在名为`example`的文件中：

    ```cpp
    $ cat example
    def foo(x , y)
    x + y * 16
    ```

1.  此文件作为参数传递给`toy`编译器进行处理：

    ```cpp
    $ ./toy example
    ```

## 它是如何工作的…

TOY 编译器将以读取模式打开`example`文件。然后，它将对单词流进行标记化。它将遇到`def`关键字并返回`DEF_TOKEN`。然后，将调用`HandleDefn()`函数，该函数将存储函数名和参数。然后，它将递归检查令牌类型，然后调用特定的令牌处理函数将它们存储到相应的 AST 中。

## 参见

+   上述词法分析和解析器除了处理一些简单的语法错误外，不处理语法错误。要实现错误处理，请参阅[`llvm.org/docs/tutorial/LangImpl2.html#parser-basics`](http://llvm.org/docs/tutorial/LangImpl2.html#parser-basics)。

# 为每个 AST 类定义 IR 代码生成方法

现在，由于 AST 已经准备好，其数据结构中包含所有必要的信息，下一个阶段是生成 LLVM IR。在此代码生成中使用了 LLVM API。LLVM IR 有一个预定义的格式，该格式由 LLVM 内置 API 生成。

## 准备工作

您必须已从 TOY 语言的任何输入代码中创建 AST。

## 如何做…

为了生成 LLVM IR，在 AST 类中定义了一个虚拟的`CodeGen`函数（这些 AST 类在 AST 部分中定义较早；这些函数是这些类的附加功能）如下：

1.  按如下方式打开`toy.cpp`文件：

    ```cpp
    $ vi toy.cpp
    ```

1.  在之前定义的`BaseAST`类中，按如下方式添加`Codegen()`函数：

    ```cpp
    class BaseAST {
      …
      …
      virtual Value* Codegen() = 0;
    };
    class NumericAST : public BaseAST {
      …
      …
      virtual Value* Codegen();
    };
    class VariableAST : public BaseAST {
      …
      …
      virtual Value* Codegen();
    };
    ```

    这个虚拟的`Codegen()`函数包含在我们定义的每个 AST 类中。

    此函数返回一个 LLVM Value 对象，它代表 LLVM 中的**静态单赋值**（**SSA**）值。定义了一些额外的静态变量，这些变量将在 Codegen 过程中使用。

1.  按如下方式在全局作用域中声明以下静态变量：

    ```cpp
    static Module *Module_Ob;
    static IRBuilder<> Builder(getGlobalContext());
    static std::map<std::string, Value*>Named_Values;
    ```

## 它是如何工作的…

`Module_Ob`模块包含代码中的所有函数和变量。

`Builder`对象帮助生成 LLVM IR，并跟踪程序中的当前位置以插入 LLVM 指令。`Builder`对象有创建新指令的函数。

`Named_Values`映射表跟踪当前作用域中定义的所有值，就像符号表一样。对于我们的语言，这个映射表将包含函数参数。

# 为表达式生成 IR 代码

在这个示例中，你将看到如何使用编译器前端生成表达式的 IR 代码。

## 如何做…

要为我们的 TOY 语言实现 LLVM IR 代码生成，请按照以下代码流程进行：

1.  按如下方式打开`toy.cpp`文件：

    ```cpp
    $ vi toy.cpp
    ```

1.  生成数值代码的功能可以定义为如下：

    ```cpp
    Value *NumericAST::Codegen() {
      return ConstantInt::get(Type::getInt32Ty(getGlobalContext()), numeric_val);
    }
    ```

    在 LLVM IR 中，整数常量由`ConstantInt`类表示，其数值由`APInt`类持有。

1.  生成变量表达式代码的函数可以定义为如下：

    ```cpp
    Value *VariableAST::Codegen() {
      Value *V = Named_Values[Var_Name];
      return V ? V : 0;
    }
    ```

1.  二元表达式的`Codegen()`函数可以定义为如下：

    ```cpp
    Value *BinaryAST::Codegen() {
      Value *L = LHS->Codegen();
      Value *R = RHS->Codegen();
      if(L == 0 || R == 0) return 0;

      switch(atoi(Bin_Operator.c_str())) {
        case '+' : return Builder.CreateAdd(L, R, "addtmp");
        case '-' : return Builder.CreateSub(L, R, "subtmp");
        case '*' : return Builder.CreateMul(L, R, "multmp");
        case '/' : return Builder.CreateUDiv(L, R, "divtmp");
        default : return 0;
      }
    }
    ```

    如果上面的代码生成了多个`addtmp`变量，LLVM 将自动为每个变量提供一个递增的唯一数字后缀。

## 参见

+   下一个示例将展示如何为函数生成 IR 代码；我们将学习代码生成是如何实际工作的。

# 为函数生成 IR 代码

在这个示例中，你将学习如何为函数生成 IR 代码。

## 如何做…

执行以下步骤：

1.  函数调用的`Codegen()`函数可以定义为如下：

    ```cpp
    Value *FunctionCallAST::Codegen() {
      Function *CalleeF =
      Module_Ob->getFunction(Function_Callee);
      std::vector<Value*>ArgsV;
      for(unsigned i = 0, e = Function_Arguments.size();
      i != e; ++i) {
        ArgsV.push_back(Function_Arguments[i]->Codegen());
        if(ArgsV.back() == 0) return 0;
      }
      return Builder.CreateCall(CalleeF, ArgsV, "calltmp");
    }
    ```

    一旦我们有了要调用的函数，我们将递归地调用`Codegen()`函数，为每个要传入的参数调用，并创建一个 LLVM 调用指令。

1.  现在，函数调用的`Codegen()`函数已经定义，是时候定义声明和函数定义的`Codegen()`函数了。

    函数声明的`Codegen()`函数可以定义为如下：

    ```cpp
    Function *FunctionDeclAST::Codegen() {
      std::vector<Type*>Integers(Arguments.size(), Type::getInt32Ty(getGlobalContext()));
      FunctionType *FT = FunctionType::get(Type::getInt32Ty(getGlobalContext()), Integers, false);
      Function *F = Function::Create(FT,  Function::ExternalLinkage, Func_Name, Module_Ob);

      if(F->getName() != Func_Name) {
        F->eraseFromParent();
        F = Module_Ob->getFunction(Func_Name);

        if(!F->empty()) return 0;

        if(F->arg_size() != Arguments.size()) return 0;

      }

      unsigned Idx = 0;
      for(Function::arg_iterator Arg_It = F->arg_begin(); Idx != Arguments.size(); ++Arg_It, ++Idx) {
        Arg_It->setName(Arguments[Idx]);
        Named_Values[Arguments[Idx]] = Arg_It;
      }

      return F;
    }
    ```

    函数定义的`Codegen()`函数可以定义为如下：

    ```cpp
    Function *FunctionDefnAST::Codegen() {
      Named_Values.clear();

      Function *TheFunction = Func_Decl->Codegen();
      if(TheFunction == 0) return 0;

      BasicBlock *BB = BasicBlock::Create(getGlobalContext(),"entry", TheFunction);
      Builder.SetInsertPoint(BB);

      if(Value *RetVal = Body->Codegen()) {
        Builder.CreateRet(RetVal);
        verifyFunction(*TheFunction);
        return TheFunction;
      }

      TheFunction->eraseFromParent();
      return 0;
    }
    ```

1.  就这样！LLVM IR 现在准备好了。这些`Codegen()`函数可以在解析顶层表达式的包装器中调用，如下所示：

    ```cpp
    static void HandleDefn() {
      if (FunctionDefnAST *F = func_defn_parser()) {
        if(Function* LF = F->Codegen()) {
        }
      }
      else {
        next_token();
      }
    }
    static void HandleTopExpression() {
      if(FunctionDefnAST *F = top_level_parser()) {
        if(Function *LF = F->Codegen()) {
        }
      }
      else {
        next_token();
      }
    }
    ```

    因此，在解析成功后，将调用相应的`Codegen()`函数来生成 LLVM IR。调用`dump()`函数以打印生成的 IR。

## 它是如何工作的…

`Codegen()`函数使用 LLVM 内置函数调用来生成 IR。为此需要包含的头文件有`llvm/IR/Verifier.h`、`llvm/IR/DerivedTypes.h`、`llvm/IR/IRBuilder.h`和`llvm/IR/LLVMContext.h`、`llvm/IR/Module.h`。

1.  在编译时，此代码需要与 LLVM 库链接。为此，可以使用`llvm-config`工具，如下所示：

    ```cpp
    llvm-config  --cxxflags  --ldflags  --system-libs  --libs core.
    ```

1.  为了这个目的，使用以下附加标志重新编译 `toy` 程序：

    ```cpp
    $ clang++ -O3 toy.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core` -o toy

    ```

1.  当 `toy` 编译器现在运行 `example` 代码时，它将生成如下所示的 LLVM IR：

    ```cpp
    $ ./toy example

    define i32 @foo (i32 %x, i32 %y) {
      entry:
      %multmp = muli32 %y, 16
      %addtmp = add i32 %x, %multmp
      reti32 %addtmp
    }
    ```

    另一个 `example2` 文件有一个函数 `call.$ cat example2`：

    ```cpp
    foo(5, 6);
    ```

    它的 LLVM IR 将如下所示：

    ```cpp
    $ ./toy example2
    define i32 @1 () {
      entry:
      %calltmp = call i32@foo(i32 5, i32 6)
      reti32 %calltmp
    }
    ```

## 参见

+   关于 Clang 中 C++ 的 `Codegen()` 函数的详细信息，请参阅 [`llvm.org/viewvc/llvm-project/cfe/trunk/lib/CodeGen/`](http://llvm.org/viewvc/llvm-project/cfe/trunk/lib/CodeGen/)

# 添加 IR 优化支持

LLVM 提供了各种优化过程。LLVM 允许编译器实现决定使用哪些优化、它们的顺序等等。在本教程中，你将学习如何添加 IR 优化支持。

## 如何做到这一点...

执行以下步骤：

1.  要开始添加 IR 优化支持，首先需要定义一个用于函数管理器的静态变量，如下所示：

    ```cpp
    static FunctionPassManager *Global_FP;
    ```

1.  然后，需要为之前使用的 `Module` 对象定义一个函数过程管理器。这可以在 `main()` 函数中如下完成：

    ```cpp
    FunctionPassManager My_FP(TheModule);
    ```

1.  现在可以在 `main()` 函数中添加各种优化过程的管道，如下所示：

    ```cpp
    My_FP.add(createBasicAliasAnalysisPass());
    My_FP.add(createInstructionCombiningPass());
    My_FP.add(createReassociatePass());
    My_FP.add(createGVNPass());
    My_FP.doInitialization();
    ```

1.  现在将静态全局函数 Pass Manager 分配给此管道，如下所示：

    ```cpp
    Global_FP = &My_FP;
    Driver();
    ```

    这个 PassManager 有一个运行方法，我们可以在函数定义的 `Codegen()` 返回之前对生成的函数 IR 运行它。如下所示进行演示：

    ```cpp
    Function* FunctionDefnAST::Codegen() {
      Named_Values.clear();
      Function *TheFunction = Func_Decl->Codegen();
      if (!TheFunction) return 0;
      BasicBlock *BB = BasicBlock::Create(getGlobalContext(), "entry", TheFunction);
      Builder.SetInsertPoint(BB);
      if (Value* Return_Value = Body->Codegen()) {
        Builder.CreateRet(Return_Value);
        verifyFunction(*TheFunction);
        Global_FP->run(*TheFunction);
        returnTheFunction;
      }
      TheFunction->eraseFromParent();
      return 0;
    }
    ```

这更有益，因为它在原地优化函数，提高了为函数体生成的代码。

## 参见

+   如何添加我们自己的优化过程及其运行方法将在后续章节中演示
