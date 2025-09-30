# 第三章. 扩展前端和添加 JIT 支持

在本章中，我们将涵盖以下食谱：

+   处理决策范式 - `if/then/else`结构

+   生成循环代码

+   处理用户定义的运算符 - 二元运算符

+   处理用户定义的运算符 - 一元运算符

+   添加 JIT 支持

# 简介

在上一章中，定义了语言前端组件的基本。这包括为不同类型的表达式定义标记，编写一个词法分析器来标记输入流，为各种表达式的抽象语法树勾勒出框架，编写解析器，并为语言生成代码。还解释了如何将各种优化连接到前端。

当一种语言具有控制流和循环来决定程序流程时，它就更加强大和表达。JIT 支持探索了即时编译代码的可能性。在本章中，将讨论这些更复杂的编程范式的实现。本章处理了使编程语言更有意义和强大的增强。本章中的食谱展示了如何为给定的语言包含这些增强。

# 处理决策范式 - `if/then/else`结构

在任何编程语言中，根据某些条件执行语句给语言带来了非常强大的优势。`if`/`then`/`else`结构提供了根据某些条件改变程序控制流的能力。条件存在于`if`结构中。如果条件为真，则执行`then`结构之后的表达式。如果它是`false`，则执行`else`结构之后的表达式。这个食谱演示了解析和生成`if`/`then`/`else`结构代码的基本基础设施。

## 准备工作

对于`if`/`then`/`else`的 TOY 语言可以定义为：

```cpp
if x < 2 then
x + y
else
x - y
```

为了检查条件，需要一个比较运算符。一个简单的`<`（小于）运算符将满足这个目的。为了处理`<`，需要在`init_precedence()`函数中定义优先级，如下所示：

```cpp
static void init_precedence() {
  Operator_Precedence['<'] = 0;
  …
  …
}
```

此外，还需要包含二进制表达式的`codegen()`函数以处理`<`：

```cpp
Value* BinaryAST::Codegen() {
…
…
…
case '<' :
L = Builder.CreateICmpULT(L, R, "cmptmp");
return Builder.CreateZExt(L, Type::getInt32Ty(getGlobalContext()),
                                "booltmp");…
…
}
```

现在，LLVM IR 将生成一个比较指令和一个布尔指令作为比较的结果，这将用于确定程序控制流的方向。现在是时候处理`if`/`then`/`else`范式了。

## 如何操作...

执行以下步骤：

1.  `toy.cpp`文件中的词法分析器需要扩展以处理`if`/`then`/`else`结构。这可以通过在`enum`标记中添加一个标记来完成：

    ```cpp
    enum Token_Type{
    …
    …
    IF_TOKEN,
    THEN_TOKEN,
    ELSE_TOKEN
    }
    ```

1.  下一步是在`get_token()`函数中添加这些标记的条目，其中我们匹配字符串并返回适当的标记：

    ```cpp
    static int get_token() {
    …
    …
    …
    if (Identifier_string == "def")  return DEF_TOKEN;
    if(Identifier_string == "if") return IF_TOKEN;
    if(Identifier_string == "then") return THEN_TOKEN;
    if(Identifier_string == "else") return ELSE_TOKEN;
    …
    …
    }
    ```

1.  然后在`toy.cpp`文件中定义一个 AST 节点：

    ```cpp
    class ExprIfAST : public BaseAST {
      BaseAST *Cond, *Then, *Else;

    public:
      ExprIfAST(BaseAST *cond, BaseAST *then, BaseAST * else_st)
          : Cond(cond), Then(then), Else(else_st) {}
      Value *Codegen() override;
    };
    ```

1.  下一步是定义`if`/`then`/`else`结构的解析逻辑：

    ```cpp
    static BaseAST *If_parser() {
      next_token();

      BaseAST *Cond = expression_parser();
      if (!Cond)
        return 0;

      if (Current_token != THEN_TOKEN)
        return 0;
      next_token();

      BaseAST *Then = expression_parser();
      if (Then == 0)
        return 0;

      if (Current_token != ELSE_TOKEN)
        return 0;

      next_token();

      BaseAST *Else = expression_parser();
      if (!Else)
        return 0;

      return new ExprIfAST(Cond, Then, Else);
    }
    ```

    解析器的逻辑很简单：首先，搜索`if`标记，并解析其后的表达式作为条件。之后，识别`then`标记并解析真条件表达式。然后搜索`else`标记并解析假条件表达式。

1.  接下来我们将之前定义的函数与`Base_Parser()`连接起来：

    ```cpp
    static BaseAST* Base_Parser() {
    switch(Current_token) {
    …
    …
    …
    case IF_TOKEN : return If_parser();
    …
    }
    ```

1.  现在已经通过解析器将`if`/`then`/`else`的 AST 填充了表达式，是时候生成条件范式的 LLVM IR 了。让我们定义`Codegen()`函数：

    ```cpp
    Value *ExprIfAST::Codegen() {
      Value *Condtn = Cond->Codegen();
      if (Condtn == 0)
        return 0;

      Condtn = Builder.CreateICmpNE(
          Condtn, Builder.getInt32(0), "ifcond");

      Function *TheFunc = Builder.GetInsertBlock()->getParent();

      BasicBlock *ThenBB =
          BasicBlock::Create(getGlobalContext(), "then", TheFunc);
      BasicBlock *ElseBB = BasicBlock::Create(getGlobalContext(), "else");
      BasicBlock *MergeBB = BasicBlock::Create(getGlobalContext(), "ifcont");

      Builder.CreateCondBr(Condtn, ThenBB, ElseBB);

      Builder.SetInsertPoint(ThenBB);

      Value *ThenVal = Then->Codegen();
      if (ThenVal == 0)
        return 0;

      Builder.CreateBr(MergeBB);
      ThenBB = Builder.GetInsertBlock();

      TheFunc->getBasicBlockList().push_back(ElseBB);
      Builder.SetInsertPoint(ElseBB);

      Value *ElseVal = Else->Codegen();
      if (ElseVal == 0)
        return 0;

      Builder.CreateBr(MergeBB);
      ElseBB = Builder.GetInsertBlock();

      TheFunc->getBasicBlockList().push_back(MergeBB);
      Builder.SetInsertPoint(MergeBB);
      PHINode *Phi = Builder.CreatePHI(Type::getInt32Ty(getGlobalContext()), 2, "iftmp");

      Phi->addIncoming(ThenVal, ThenBB);
      Phi->addIncoming(ElseVal, ElseBB);
      return Phi;
    }
    ```

现在我们已经准备好了代码，让我们在一个包含`if`/`then`/`else`结构的示例程序上编译并运行它。

## 如何工作…

执行以下步骤：

1.  编译`toy.cpp`文件：

    ```cpp
    $ g++ -g toy.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core ` -O3 -o toy

    ```

1.  打开一个示例文件：

    ```cpp
    $ vi example

    ```

1.  在示例文件中编写以下`if`/`then`/`else`代码：

    ```cpp
    def fib(x)
      if x < 3 then
        1
      Else
        fib(x-1)+fib(x-2);
    ```

1.  使用 TOY 编译器编译示例文件：

    ```cpp
    $ ./toy example

    ```

为`if`/`then`/`else`代码生成的 LLVM IR 看起来如下：

```cpp
; ModuleID = 'my compiler'
target datalayout = "e-m:e-p:32:32-f64:32:64-f80:32-n8:16:32-S128"

define i32 @fib(i32 %x) {
entry:
  %cmptmp = icmp ult i32 %x, 3
  br i1 %cmptmp, label %ifcont, label %else

else:                                             ; preds = %entry
  %subtmp = add i32 %x, -1
  %calltmp = call i32 @fib(i32 %subtmp)
  %subtmp1 = add i32 %x, -2
  %calltmp2 = call i32 @fib(i32 %subtmp1)
  %addtmp = add i32 %calltmp2, %calltmp
  br label %ifcont

ifcont:                                           ; preds = %entry, %else
  %iftmp = phi i32 [ %addtmp, %else ], [ 1, %entry ]
  ret i32 %iftmp
}
```

下面是输出结果的样子：

![如何工作…](img/image00254.jpeg)

解析器识别`if`/`then`/`else`结构和在真和假条件下要执行的语句，并将它们存储在 AST 中。然后代码生成器将 AST 转换为 LLVM IR，其中生成条件语句。为真和假条件都生成了 IR。根据条件变量的状态，在运行时执行适当的语句。

## 参见

+   有关 Clang 如何处理 C++中的`if else`语句的详细示例，请参阅[`clang.llvm.org/doxygen/classclang_1_1IfStmt.html`](http://clang.llvm.org/doxygen/classclang_1_1IfStmt.html)。

# 生成循环的代码

循环使语言足够强大，可以在有限的代码行数内执行相同的操作多次。几乎每种语言都有循环。这个示例展示了在 TOY 语言中如何处理循环。

## 准备工作

循环通常有一个初始化归纳变量的起始点，一个指示归纳变量增加或减少的步长，以及一个终止循环的结束条件。在我们的 TOY 语言中，循环可以定义为以下内容：

```cpp
for i = 1, i < n, 1 in
     x + y;
```

起始表达式是`i = 1`的初始化。循环的结束条件是`i<n`。代码的第一行表示`i`增加`1`。

只要结束条件为真，循环就会迭代，并且在每次迭代后，归纳变量`i`会增加 1。一个有趣的现象叫做**PHI**节点，它将决定归纳变量`i`将取哪个值。记住，我们的 IR 是**单赋值**（**SSA**）形式。在控制流图中，对于给定的变量，其值可能来自两个不同的块。为了在 LLVM IR 中表示 SSA，定义了`phi`指令。以下是一个`phi`的例子：

```cpp
%i = phi i32 [ 1, %entry ], [ %nextvar, %loop ]
```

上述 IR 表明`i`的值可以来自两个基本块：`%entry`和`%loop`。`%entry`块的值将是`1`，而`%nextvar`变量将来自`%loop`。我们将在实现我们的玩具编译器的循环后看到详细信息。

## 如何做到这一点...

和任何其他表达式一样，循环也是通过在词法分析器中包含状态、定义用于存储循环值的 AST 数据结构以及定义解析器和`Codegen()`函数来生成 LLVM IR 来处理的：

1.  第一步是在`toy.cpp`文件中的词法分析器中定义标记：

    ```cpp
    enum Token_Type {
      …
      …
      FOR_TOKEN,
      IN_TOKEN
      …
      …
    };
    ```

1.  然后我们在词法分析器中包含逻辑：

    ```cpp
    static int get_token() {
      …
      …
    if (Identifier_string == "else")
          return ELSE_TOKEN;
        if (Identifier_string == "for")
          return FOR_TOKEN;
        if (Identifier_string == "in")
          return IN_TOKEN;
      …
      …
    }
    ```

1.  下一步是定义`for`循环的 AST：

    ```cpp
    class ExprForAST  : public BaseAST {
      std::string Var_Name;
      BaseAST *Start, *End, *Step, *Body;

    public:
      ExprForAST (const std::string &varname, BaseAST *start, BaseAST *end,
                 BaseAST *step, BaseAST *body)
          : Var_Name(varname), Start(start), End(end), Step(step), Body(body) {}
      Value *Codegen() override;
    };
    ```

1.  然后我们定义循环的解析逻辑：

    ```cpp
    static BaseAST *For_parser() {
      next_token();

      if (Current_token != IDENTIFIER_TOKEN)
        return 0;

      std::string IdName = Identifier_string;
      next_token();

      if (Current_token != '=')
        return 0;
      next_token();

      BaseAST *Start = expression_parser();
      if (Start == 0)
        return 0;
      if (Current_token != ',')
        return 0;
      next_token();

      BaseAST *End = expression_parser();
      if (End == 0)
        return 0;

      BaseAST *Step = 0;
      if (Current_token == ',') {
        next_token();
        Step = expression_parser();
        if (Step == 0)
          return 0;
      }

      if (Current_token != IN_TOKEN)
        return 0;
      next_token();

      BaseAST *Body = expression_parser();
      if (Body == 0)
        return 0;

      return new ExprForAST (IdName, Start, End, Step, Body);
    }
    ```

1.  接下来我们定义`Codegen()`函数以生成 LLVM IR：

    ```cpp
    Value *ExprForAST::Codegen() {

      Value *StartVal = Start->Codegen();
      if (StartVal == 0)
        return 0;

      Function *TheFunction = Builder.GetInsertBlock()->getParent();
      BasicBlock *PreheaderBB = Builder.GetInsertBlock();
      BasicBlock *LoopBB =
          BasicBlock::Create(getGlobalContext(), "loop", TheFunction);

      Builder.CreateBr(LoopBB);

      Builder.SetInsertPoint(LoopBB);

      PHINode *Variable = Builder.CreatePHI(Type::getInt32Ty(getGlobalContext()), 2, Var_Name.c_str());
      Variable->addIncoming(StartVal, PreheaderBB);

      Value *OldVal = Named_Values[Var_Name];
      Named_Values[Var_Name] = Variable;

      if (Body->Codegen() == 0)
        return 0;

      Value *StepVal;
      if (Step) {
        StepVal = Step->Codegen();
        if (StepVal == 0)
          return 0;
      } else {
        StepVal = ConstantInt::get(Type::getInt32Ty(getGlobalContext()), 1);
      }

      Value *NextVar = Builder.CreateAdd(Variable, StepVal, "nextvar");

      Value *EndCond = End->Codegen();
      if (EndCond == 0)
        return EndCond;

      EndCond = Builder.CreateICmpNE(
          EndCond, ConstantInt::get(Type::getInt32Ty(getGlobalContext()), 0), "loopcond");

      BasicBlock *LoopEndBB = Builder.GetInsertBlock();
      BasicBlock *AfterBB =
          BasicBlock::Create(getGlobalContext(), "afterloop", TheFunction);

      Builder.CreateCondBr(EndCond, LoopBB, AfterBB);

      Builder.SetInsertPoint(AfterBB);

      Variable->addIncoming(NextVar, LoopEndBB);

      if (OldVal)
        Named_Values[Var_Name] = OldVal;
      else
        Named_Values.erase(Var_Name);

      return Constant::getNullValue(Type::getInt32Ty(getGlobalContext()));
    }
    ```

## 它是如何工作的...

执行以下步骤：

1.  编译`toy.cpp`文件：

    ```cpp
    $ g++ -g toy.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core ` -O3 -o toy

    ```

1.  打开一个示例文件：

    ```cpp
    $ vi example

    ```

1.  在示例文件中为`for`循环编写以下代码：

    ```cpp
    def printstar(n x)
      for i = 1, i < n, 1.0 in
        x + 1
    ```

1.  使用 TOY 编译器编译示例文件：

    ```cpp
    $ ./toy example

    ```

1.  以下`for`循环代码的 LLVM IR 如下：

    ```cpp
    ; ModuleID = 'my compiler'
    target datalayout = "e-m:e-p:32:32-f64:32:64-f80:32-n8:16:32-S128"

    define i32 @printstar(i32 %n, i32 %x) {
    entry:
      br label %loop

    loop:                                             ; preds = %loop, %entry
      %i = phi i32 [ 1, %entry ], [ %nextvar, %loop ]
      %nextvar = add i32 %i, 1
      %cmptmp = icmp ult i32 %i, %n
      br i1 %cmptmp, label %loop, label %afterloop

    afterloop:                                        ; preds = %loop
      ret i32 0
    }
    ```

你刚才看到的解析器识别了循环、归纳变量的初始化、终止条件、归纳变量的步长值以及循环体。然后它将 LLVM IR 中每个块转换为前面看到的。

如前所述，一个`phi`指令从两个基本块`%entry`和`%loop`中为变量`i`获取两个值。在前面的例子中，`%entry`块表示循环开始时分配给归纳变量的值（这是`1`）。`i`的下一个更新值来自`%loop`块，它完成了循环的一次迭代。

## 参见

+   要详细了解 Clang 中如何处理 C++中的循环，请访问[`llvm.org/viewvc/llvm-project/cfe/trunk/lib/Parse/ParseExprCXX.cpp`](http://llvm.org/viewvc/llvm-project/cfe/trunk/lib/Parse/ParseExprCXX.cpp)

# 处理用户定义的操作符 - 二元操作符

用户定义的操作符类似于 C++中的操作符重载概念，其中默认的操作符定义被修改以在多种对象上操作。通常，操作符是一元或二元操作符。使用现有基础设施实现二元操作符重载更容易。一元操作符需要一些额外的代码来处理。首先，将定义二元操作符重载，然后研究一元操作符重载。

## 准备工作

第一部分是定义一个用于重载的二进制操作符。逻辑或操作符（`|`）是一个很好的起点。在我们的 TOY 语言中，`|`操作符可以如下使用：

```cpp
def binary | (LHS RHS)
if LHS then
1
else if RHS then
1
else
0;
```

如前所述，如果 LHS 或 RHS 的任何值不等于 0，则返回`1`。如果 LHS 和 RHS 都为 null，则返回`0`。

## 如何做到这一点...

执行以下步骤：

1.  第一步，像往常一样，是追加二元操作符的`enum`状态，并在遇到`binary`关键字时返回枚举状态：

    ```cpp
     enum Token_Type {
    …
    …
    BINARY_TOKEN
    }
    static int get_token() {
    …
    …
    if (Identifier_string == "in") return IN_TOKEN;
    if (Identifier_string == "binary") return BINARY_TOKEN;
    …
    …
    }
    ```

1.  下一步是为同一运算符添加 AST。请注意，不需要定义新的 AST。它可以由函数声明 AST 处理。我们只需要通过添加一个标志来修改它，以表示它是否为二进制运算符。如果是，则确定其优先级：

    ```cpp
    class FunctionDeclAST {
      std::string Func_Name;
      std::vector<std::string> Arguments;
      bool isOperator;
      unsigned Precedence;
    public:
      FunctionDeclAST(const std::string &name, const std::vector<std::string> &args,
                   bool isoperator = false, unsigned prec = 0)
          : Func_Name(name), Arguments(args), isOperator(isoperator), Precedence(prec) {}

      bool isUnaryOp() const { return isOperator && Arguments.size() == 1; }
      bool isBinaryOp() const { return isOperator && Arguments.size() == 2; }

      char getOperatorName() const {
        assert(isUnaryOp() || isBinaryOp());
        return Func_Name[Func_Name.size() - 1];
      }

      unsigned getBinaryPrecedence() const { return Precedence; }

      Function *Codegen();
    };
    ```

1.  一旦修改后的 AST 准备好了，下一步是修改函数声明的解析器：

    ```cpp
    static FunctionDeclAST *func_decl_parser() {
      std::string FnName;

      unsigned Kind = 0;
      unsigned BinaryPrecedence = 30;

      switch (Current_token) {
      default:
        return 0;
      case IDENTIFIER_TOKEN:
        FnName = Identifier_string;
        Kind = 0;
        next_token();
        break;
      case UNARY_TOKEN:
        next_token();
        if (!isascii(Current_token))
          return 0;
        FnName = "unary";
        FnName += (char)Current_token;
        Kind = 1;
        next_token();
        break;
      case BINARY_TOKEN:
        next_token();
        if (!isascii(Current_token))
          return 0;
        FnName = "binary";
        FnName += (char)Current_token;
        Kind = 2;
        next_token();

        if (Current_token == NUMERIC_TOKEN) {
          if (Numeric_Val < 1 || Numeric_Val > 100)
            return 0;
          BinaryPrecedence = (unsigned)Numeric_Val;
          next_token();
        }
        break;
      }

      if (Current_token != '(')
        return 0;

      std::vector<std::string> Function_Argument_Names;
      while (next_token() == IDENTIFIER_TOKEN)
        Function_Argument_Names.push_back(Identifier_string);
      if (Current_token != ')')
        return 0;

      next_token();

      if (Kind && Function_Argument_Names.size() != Kind)
        return 0;

      return new FunctionDeclAST(FnName, Function_Argument_Names, Kind != 0, BinaryPrecedence);
    }
    ```

1.  然后我们修改二进制 AST 的`Codegen()`函数：

    ```cpp
    Value* BinaryAST::Codegen() {
     Value* L = LHS->Codegen();
    Value* R = RHS->Codegen();
    switch(Bin_Operator) {
    case '+' : return Builder.CreateAdd(L, R, "addtmp");
    case '-' : return Builder.CreateSub(L, R, "subtmp");
    case '*': return Builder.CreateMul(L, R, "multmp");
    case '/': return Builder.CreateUDiv(L, R, "divtmp");
    case '<' :
    L = Builder.CreateICmpULT(L, R, "cmptmp");
    return Builder.CreateUIToFP(L, Type::getIntTy(getGlobalContext()), "booltmp");
    default :
    break;
    }
    Function *F = TheModule->getFunction(std::string("binary")+Op);
      Value *Ops[2] = { L, R };
      return Builder.CreateCall(F, Ops, "binop");
    }
    ```

1.  下一步我们修改函数定义；它可以定义为：

    ```cpp
    Function* FunctionDefnAST::Codegen() {
    Named_Values.clear();
    Function *TheFunction = Func_Decl->Codegen();
    if (!TheFunction) return 0;
    if (Func_Decl->isBinaryOp())
        Operator_Precedence [Func_Decl->getOperatorName()] = Func_Decl->getBinaryPrecedence();
    BasicBlock *BB = BasicBlock::Create(getGlobalContext(), "entry", TheFunction);
    Builder.SetInsertPoint(BB);
    if (Value* Return_Value = Body->Codegen()) {
        Builder.CreateRet(Return_Value);
    …
    …
    ```

## 它是如何工作的...

执行以下步骤：

1.  编译`toy.cpp`文件：

    ```cpp
    $ g++ -g toy.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core ` -O3 -o toy

    ```

1.  打开一个示例文件：

    ```cpp
    $ vi example

    ```

1.  在示例文件中编写以下二进制运算符重载代码：

    ```cpp
    def binary| 5 (LHS RHS)
      if LHS then
        1
      else if RHS then
        1
      else
        0;
    ```

1.  使用 TOY 编译器编译示例文件：

    ```cpp
    $ ./toy example

    output :

    ; ModuleID = 'my compiler'
    target datalayout = "e-m:e-p:32:32-f64:32:64-f80:32-n8:16:32-S128"

    define i32 @"binary|"(i32 %LHS, i32 %RHS) {
    entry:
     %ifcond = icmp eq i32 %LHS, 0
     %ifcond1 = icmp eq i32 %RHS, 0
     %. = select i1 %ifcond1, i32 0, i32 1
     %iftmp5 = select i1 %ifcond, i32 %., i32 1
     ret i32 %iftmp5
    }

    ```

我们刚刚定义的二进制运算符将被解析。其定义也将被解析。每当遇到`|`二进制运算符时，LHS 和 RHS 将被初始化，并执行定义体，根据定义给出适当的结果。在上面的示例中，如果 LHS 或 RHS 中任意一个不为零，则结果为`1`。如果 LHS 和 RHS 都为零，则结果为`0`。

## 参见

+   有关处理其他二进制运算符的详细示例，请参阅[`llvm.org/docs/tutorial/LangImpl6.html`](http://llvm.org/docs/tutorial/LangImpl6.html)

# 处理用户定义的运算符 - 一元运算符

我们在先前的配方中看到了如何处理二进制运算符。一种语言也可能有一些一元运算符，它们对一个操作数进行操作。在这个配方中，我们将看到如何处理一元运算符。

## 准备工作

第一步是在 TOY 语言中定义一元运算符。一个简单的 NOT 一元运算符（`!`）可以作为一个很好的例子；让我们看看一个定义：

```cpp
def unary!(v)
  if v then
    0
  else
    1;
```

如果值`v`等于`1`，则返回`0`。如果值是`0`，则输出`1`。

## 如何做到这一点...

执行以下步骤：

1.  第一步是在`toy.cpp`文件中定义一元运算符的`enum`标记：

    ```cpp
    enum Token_Type {
    …
    …
    BINARY_TOKEN,
    UNARY_TOKEN
    }
    ```

1.  然后我们识别一元字符串并返回一个一元标记：

    ```cpp
    static int get_token() {
    …
    …
    if (Identifier_string == "in") return IN_TOKEN;
    if (Identifier_string == "binary") return BINARY_TOKEN;
    if (Identifier_string == "unary") return UNARY_TOKEN;

    …
    …
    }
    ```

1.  接下来，我们为一元运算符定义 AST：

    ```cpp
    class ExprUnaryAST : public BaseAST {
      char Opcode;
      BaseAST *Operand;
    public:
      ExprUnaryAST(char opcode, BaseAST *operand)
        : Opcode(opcode), Operand(operand) {}
      virtual Value *Codegen();
    };
    ```

1.  AST 现在准备好了。让我们为一元运算符定义一个解析器：

    ```cpp
    static BaseAST *unary_parser() {

      if (!isascii(Current_token) || Current_token == '(' || Current_token == ',')
        return Base_Parser();

        int Op = Current_token;

      next_token();

      if (ExprAST *Operand = unary_parser())
        return new ExprUnaryAST(Opc, Operand);

    return 0;
    }
    ```

1.  下一步是从二进制运算符解析器中调用`unary_parser()`函数：

    ```cpp
    static BaseAST *binary_op_parser(int Old_Prec, BaseAST *LHS) {

      while (1) {
        int Operator_Prec = getBinOpPrecedence();

        if (Operator_Prec < Old_Prec)
          return LHS;

        int BinOp = Current_token;
        next_token();

        BaseAST *RHS = unary_parser();
        if (!RHS)
          return 0;

        int Next_Prec = getBinOpPrecedence();
        if (Operator_Prec < Next_Prec) {
          RHS = binary_op_parser(Operator_Prec + 1, RHS);
          if (RHS == 0)
            return 0;
        }

        LHS = new BinaryAST(std::to_string(BinOp), LHS, RHS);
      }
    }
    ```

1.  现在让我们从表达式解析器中调用`unary_parser()`函数：

    ```cpp
    static BaseAST *expression_parser() {
      BaseAST *LHS = unary_parser();
      if (!LHS)
        return 0;

      return binary_op_parser(0, LHS);
    }
    ```

1.  然后我们修改函数声明解析器：

    ```cpp
    static FunctionDeclAST* func_decl_parser() {
    std::string Function_Name = Identifier_string;
    unsigned Kind = 0;
    unsigned BinaryPrecedence = 30;
    switch (Current_token) {
      default:
        return 0;
      case IDENTIFIER_TOKEN:
        Function_Name = Identifier_string;
        Kind = 0;
        next_token();
        break;
      case UNARY_TOKEN:
      next_token();
    if (!isascii(Current_token))
          return0;
        Function_Name = "unary";
        Function_Name += (char)Current_token;
        Kind = 1;
        next_token();
        break;
      case BINARY_TOKEN:
        next_token();
        if (!isascii(Current_token))
          return 0;
        Function_Name = "binary";
        Function_Name += (char)Current_token;
        Kind = 2;
       next_token();
       if (Current_token == NUMERIC_TOKEN) {
          if (Numeric_Val < 1 || Numeric_Val > 100)
            return 0;
          BinaryPrecedence = (unsigned)Numeric_Val;
          next_token();
        }
        break;
      }
    if (Current_token ! = '(') {
    printf("error in function declaration");
    return 0;
    }
    std::vector<std::string> Function_Argument_Names;
    while(next_token() == IDENTIFIER_TOKEN) Function_Argument_Names.push_back(Identifier_string);
    if(Current_token != ')')  {                      printf("Expected ')' ");                      return 0;
    }
    next_token();
    if (Kind && Function_Argument_Names.size() != Kind)
        return 0;
    return new FunctionDeclAST(Function_Name, Function_Arguments_Names, Kind !=0, BinaryPrecedence);
    }
    ```

1.  最后一步是为一元运算符定义`Codegen()`函数：

    ```cpp
    Value *ExprUnaryAST::Codegen() {

      Value *OperandV = Operand->Codegen();

      if (OperandV == 0) return 0;

      Function *F = TheModule->getFunction(std::string("unary")+Opcode);

      if (F == 0)
        return 0;

      return Builder.CreateCall(F, OperandV, "unop");
    }
    ```

## 它是如何工作的...

执行以下步骤：

1.  编译`toy.cpp`文件：

    ```cpp
    $ g++ -g toy.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core ` -O3 -o toy

    ```

1.  打开一个示例文件：

    ```cpp
    $ vi example

    ```

1.  在示例文件中编写以下一元运算符重载代码：

    ```cpp
    def unary!(v)
      if v then
        0
      else
        1;
    ```

1.  使用 TOY 编译器编译示例文件：

    ```cpp
    $ ./toy example

    ```

    输出应该如下所示：

    ```cpp
    ; ModuleID = 'my compiler'
    target datalayout = "e-m:e-p:32:32-f64:32:64-f80:32-n8:16:32-S128"

    define i32 @"unary!"(i32 %v) {
    entry:
     %ifcond = icmp eq i32 %v, 0
     %. = select i1 %ifcond, i32 1, i32 0
     ret i32 %.
    }

    ```

用户定义的一元运算符将被解析，并为它生成 IR。在你刚才看到的例子中，如果一元操作数不为零，则结果为`0`。如果操作数为零，则结果为`1`。

## 参见

+   要了解一元运算符的更详细实现，请访问 [`llvm.org/docs/tutorial/LangImpl6.html`](http://llvm.org/docs/tutorial/LangImpl6.html)

# 添加 JIT 支持

可以将各种工具应用于 LLVM IR。例如，正如在第一章中所示，*LLVM 设计和使用*，IR 可以被转换为位码或汇编。可以在 IR 上运行一个名为 opt 的优化工具。IR 作为通用平台——所有这些工具的抽象层。

可以添加 JIT 支持。它立即评估输入的最高级表达式。例如，`1 + 2;`，一旦输入，就会评估代码并打印出值 `3`。

## 如何做到这一点...

执行以下步骤：

1.  在 `toy.cpp` 文件中为执行引擎定义一个静态全局变量：

    ```cpp
    static ExecutionEngine *TheExecutionEngine;
    ```

1.  在 `toy.cpp` 文件的 `main()` 函数中，编写 JIT 代码：

    ```cpp
    int main() {
    …
    …
    init_precedence();
    TheExecutionEngine = EngineBuilder(TheModule).create();
    …
    …
    }
    ```

1.  修改 `toy.cpp` 文件中的顶层表达式解析器：

    ```cpp
    static void HandleTopExpression() {

    if (FunctionDefAST *F = expression_parser())
       if (Function *LF = F->Codegen()) {
            LF -> dump();
           void *FPtr = TheExecutionEngine->getPointerToFunction(LF);
          int (*Int)() = (int (*)())(intptr_t)FPtr;

        printf("Evaluated to %d\n", Int());
    }
       else
    next_token();
    }
    ```

## 它是如何工作的...

执行以下步骤：

1.  编译 `toy.cpp` 程序：

    ```cpp
    $ g++ -g toy.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core mcjit native` -O3 -o toy

    ```

1.  打开一个示例文件：

    ```cpp
    $ vi example

    ```

1.  在示例文件中编写以下 TOY 代码：

    ```cpp
    …
    4+5;
    ```

1.  最后，在示例文件上运行 TOY 编译器：

    ```cpp
    $ ./toy example
    The output will be
    define i32 @0() {
    entry:
     ret i32 9
    }

    ```

LLVM JIT 编译器匹配原生平台 ABI，将结果指针转换为该类型的函数指针，并直接调用它。JIT 编译的代码与与应用程序静态链接的原生机器代码之间没有区别。
