# 8

# TableGen 语言

LLVM 后端的大部分内容是用 TableGen 语言编写的，这是一种用于生成 C++源代码片段的特殊语言，以避免为每个后端实现相似代码并缩短源代码量。因此，了解 TableGen 是很重要的。

在本章中，你将学习以下内容：

+   在*理解 TableGen 语言*中，你将了解 TableGen 背后的主要思想

+   在*实验 TableGen 语言*中，你将定义自己的 TableGen 类和记录，并学习 TableGen 语言的语法

+   在*从 TableGen 文件生成 C++代码*中，你将开发自己的 TableGen 后端

+   TableGen 的缺点

到本章结束时，你将能够使用现有的 TableGen 类来定义你自己的记录。你还将获得如何从头创建 TableGen 类和记录的知识，以及如何开发一个 TableGen 后端以生成源代码。

# 技术要求

你可以在 GitHub 上找到本章使用的源代码：[`github.com/PacktPublishing/Learn-LLVM-17/tree/main/Chapter08`](https://github.com/PacktPublishing/Learn-LLVM-17/tree/main/Chapter08)。

# 理解 TableGen 语言

LLVM 自带一种名为**TableGen**的**领域特定语言**（**DSL**）。它被用于生成适用于广泛用例的 C++代码，从而减少了开发者需要编写的代码量。TableGen 语言不是一个完整的编程语言。它仅用于定义记录，这是一个指代名称和值集合的术语。为了理解为什么这种受限的语言是有用的，让我们考察两个例子。

定义 CPU 的一个机器指令通常需要以下典型数据：

+   指令的助记符

+   位模式

+   操作数数量和类型

+   可能的限制或副作用

很容易看出这些数据可以表示为一个记录。例如，一个名为`asmstring`的字段可以保存助记符的值；比如说，`"add"`。还有一个名为`opcode`的字段可以保存指令的二进制表示。这些字段共同描述了一个额外的指令。每个 LLVM 后端都以这种方式描述指令集。

记录是一个如此通用的概念，以至于你可以用它们描述各种各样的数据。另一个例子是命令行选项的定义。一个命令行选项：

+   有一个名称

+   可能有一个可选参数

+   有帮助文本

+   可能属于一组选项

再次，这些数据可以很容易地被视为一个记录。Clang 使用这种方法为 Clang 驱动器的命令行选项。

TableGen 语言

在 LLVM 中，TableGen 语言被用于各种任务。后端的大部分内容是用 TableGen 语言编写的；例如，寄存器文件的定义，所有带有助记符和二进制编码的指令，调用约定，指令选择的模式，以及指令调度的调度模型。LLVM 的其他用途包括内建函数的定义，属性的定义，以及命令行选项的定义。

你可以在[`llvm.org/docs/TableGen/ProgRef.html`](https://llvm.org/docs/TableGen/ProgRef.html)找到《程序员参考》，在[`llvm.org/docs/TableGen/BackGuide.html`](https://llvm.org/docs/TableGen/BackGuide.html)找到《后端开发者指南》。

为了实现这种灵活性，TableGen 语言的解析和语义是在一个库中实现的。要从记录生成 C++代码，你需要创建一个工具，该工具接受解析后的记录并从中生成 C++代码。在 LLVM 中，这个工具被称为`llvm-tblgen`，在 Clang 中被称为`clang-tblgen`。这些工具包含项目所需的代码生成器。但它们也可以用来学习更多关于 TableGen 语言的知识，这就是我们在下一节将要做的。

# 尝试使用 TableGen 语言

初学者往往觉得 TableGen 语言令人不知所措。但一旦你开始尝试使用这种语言，它就会变得容易得多。

## 定义记录和类

让我们定义一个简单的指令记录：

```cpp

def ADD {
  string Mnemonic = "add";
  int Opcode = 0xA0;
}
```

`def` 关键字表示定义一个记录。其后跟随记录的名称。记录体被大括号包围，体由字段定义组成，类似于 C++中的结构体。

你可以使用`llvm-tblgen`工具查看生成的记录。将前面的源代码保存为`inst.td`文件，然后运行以下命令：

```cpp

$ llvm-tblgen --print-records inst.td
------------- Classes -----------------
------------- Defs -----------------
def ADD {
  string Mnemonic = "add";
  int Opcode = 160;
}
```

这还不是特别令人兴奋；它只表明定义的记录被正确解析。

使用单个记录定义指令并不太方便。现代 CPU 有数百条指令，这么多记录，很容易在字段名称中引入打字错误。如果你决定重命名一个字段或添加一个新字段，那么需要更改的记录数量就成为一个挑战。因此，需要一个蓝图。在 C++中，类有类似的作用，在 TableGen 中，它也被称为`Inst`类和基于该类的两个记录：

```cpp

class Inst<string mnemonic, int opcode> {
  string Mnemonic = mnemonic;
  int Opcode = opcode;
}
def ADD : Inst<"add", 0xA0>;
def SUB : Inst<"sub", 0xB0>;
```

类的语法与记录类似。`class`关键字表示定义了一个类，其后跟随类的名称。类可以有一个参数列表。在这里，`Inst`类有两个参数，`mnemonic`和`opcode`，它们用于初始化记录的字段。这些字段的值在类实例化时给出。`ADD`和`SUB`记录展示了类的两个实例。再次使用`llvm-tblgen`来查看记录：

```cpp

$ llvm-tblgen --print-records inst.td
------------- Classes -----------------
class Inst<string Inst:mnemonic = ?, int Inst:opcode = ?> {
  string Mnemonic = Inst:mnemonic;
  int Opcode = Inst:opcode;
}
------------- Defs -----------------
def ADD {       // Inst
  string Mnemonic = "add";
  int Opcode = 160;
}
def SUB {       // Inst
  string Mnemonic = "sub";
  int Opcode = 176;
}
```

现在，你有一个类定义和两个记录。用于定义记录的类的名称显示为注释。请注意，类的参数默认值为 `?`，表示 `int` 未初始化。

调试技巧

要获取记录的更详细输出，可以使用 `–-print-detailed-records` 选项。输出包括记录和类定义的行号，以及记录字段初始化的位置。如果你试图追踪为什么记录字段被赋予某个特定的值，它们可能非常有帮助。

通常，`ADD` 和 `SUB` 指令有很多共同之处，但也有区别：加法是交换律操作，而减法不是。我们也要在记录中捕捉这一事实。一个小挑战是 TableGen 只支持有限的数据类型。你已经在示例中使用了 `string` 和 `int`。其他可用的数据类型有 `bit`、`bits<n>`、`list<type>` 和 `dag`。`bit` 类型表示单个位；即 `0` 或 `1`。如果你需要一个固定数量的位，那么你将使用 `bits<n>` 类型。例如，`bits<5>` 是一个 5 位宽的整型。要基于其他类型定义列表，你将使用 `list<type>` 类型。例如，`list<int>` 是一个整数列表，而 `list<Inst>` 是从示例中 `Inst` 类的记录列表。`dag` 类型表示 **有向无环图**（DAG）节点。这种类型对于定义模式和操作非常有用，并且在 LLVM 后端中被广泛使用。

为了表示一个标志，一个单独的位就足够了，所以你可以使用一个来标记指令为可交换的。大多数指令都不是可交换的，所以你可以利用默认值：

```cpp

class Inst<string mnemonic, int opcode, bit commutable = 0> {
  string Mnemonic = mnemonic;
  int Opcode = opcode;
  bit Commutable = commutable;
}
def ADD : Inst<"add", 0xA0, 1>;
def SUB : Inst<"sub", 0xB0>;
```

你应该运行 `llvm-tblgen` 来验证记录是否按预期定义。

类不需要有参数。也可以稍后分配值。例如，你可以定义所有指令都不是可交换的：

```cpp

class Inst<string mnemonic, int opcode> {
  string Mnemonic = mnemonic;
  int Opcode = opcode;
  bit Commutable = 0;
}
def SUB : Inst<"sub", 0xB0>;
```

使用 `let` 语句，你可以覆盖该值：

```cpp

let Commutable = 1 in
  def ADD : Inst<"add", 0xA0>;
```

或者，你可以打开记录体来覆盖该值：

```cpp

def ADD : Inst<"add", 0xA0> {
  let Commutable = 1;
}
```

再次提醒，请使用 `llvm-tblgen` 验证在两种情况下 `Commutable` 标志是否设置为 `1`。

类和记录可以从多个类继承，并且总是可以添加新字段或覆盖现有字段的值。你可以使用继承来引入一个新的 `CommutableInst` 类：

```cpp

class Inst<string mnemonic, int opcode> {
  string Mnemonic = mnemonic;
  int Opcode = opcode;
  bit Commutable = 0;
}
class CommutableInst<string mnemonic, int opcode>
  : Inst<mnemonic, opcode> {
  let Commutable = 1;
}
def SUB : Inst<"sub", 0xB0>;
def ADD : CommutableInst<"add", 0xA0>;
```

结果记录始终相同，但语言允许你以不同的方式定义记录。请注意，在后一个示例中，`Commutable` 标志可能是多余的：代码生成器可以查询记录所基于的类，如果该列表包含 `CommutableInst` 类，则它可以内部设置该标志。

## 使用多类一次创建多个记录

另一个经常使用的语句是 `multiclass`。多类允许你一次定义多个记录。让我们扩展示例来展示这为什么有用。

`add`指令的定义非常简单。在现实中，CPU 通常有几个`add`指令。一个常见的变体是，一个指令有两个寄存器操作数，而另一个指令有一个寄存器操作数和一个立即数操作数，这是一个小的数字。假设对于具有立即数操作的指令，指令集的设计者决定用`i`作为后缀来标记它们。因此，我们最终得到`add`和`addi`指令。进一步假设操作码相差`1`。许多算术和逻辑指令遵循此方案；因此，您希望定义尽可能紧凑。

第一个挑战是您需要操作值。您可以使用有限数量的运算符来修改一个值。例如，要生成`1`和字段 opcode 值的和，您可以这样写：

```cpp

!add(opcode, 1)
```

这样的表达式最好用作类的参数。测试字段值并根据找到的值进行更改通常是不可能的，因为这需要动态语句，而这些语句是不可用的。始终记住，所有计算都是在记录构建时完成的！

以类似的方式，字符串可以连接：

```cpp

!strconcat(mnemonic,"i")
```

因为所有运算符都以感叹号（`!`）开头，它们也被称为**感叹号运算符**。您可以在*程序员参考*中找到完整的感叹号运算符列表：[`llvm.org/docs/TableGen/ProgRef.html#appendix-a-bang-operators`](https://llvm.org/docs/TableGen/ProgRef.html#appendix-a-bang-operators)。

现在，您可以定义一个多类。`Inst`类再次作为基类：

```cpp

class Inst<string mnemonic, int opcode> {
  string Mnemonic = mnemonic;
  int Opcode = opcode;
}
```

多类的定义稍微复杂一些，所以让我们分步骤来做：

1.  多类定义使用的语法与类类似。新的多类名为`InstWithImm`，有两个参数，`mnemonic`和`opcode`：

    ```cpp

    multiclass InstWithImm<string mnemonic, int opcode> {
    ```

1.  首先，您需要使用两个寄存器操作数定义一个指令。就像在正常的记录定义中一样，您使用`def`关键字来定义记录，并使用`Inst`类来创建记录内容。您还需要定义一个空名称。我们稍后会解释为什么这是必要的：

    ```cpp

      def "": Inst<mnemonic, opcode>;
    ```

1.  接下来，您使用立即数操作数定义一个指令。您使用感叹号运算符从多类的参数中推导出助记符和操作码的值。记录命名为`I`：

    ```cpp

      def I: Inst<!strconcat(mnemonic,"i"), !add(opcode, 1)>;
    ```

1.  那就是全部了；类体可以像这样关闭：

    ```cpp

    }
    ```

要实例化记录，您必须使用`defm`关键字：

```cpp

defm ADD : InstWithImm<"add", 0xA0>;
```

这些语句的结果如下：

1.  `Inst<"add", 0xA0>`记录被实例化。记录的名称是`defm`关键字后面的名称和多层语句中`def`后面的名称的连接，结果为名称`ADD`。

1.  `Inst<"addi", 0xA1>`记录被实例化，并按照相同的方案，被赋予名称`ADDI`。

让我们用`llvm-tblgen`验证这个说法：

```cpp

$ llvm-tblgen –print-records inst.td
------------- Classes -----------------
class Inst<string Inst:mnemonic = ?, int Inst:opcode = ?> {
  string Mnemonic = Inst:mnemonic;
  int Opcode = Inst:opcode;
}
------------- Defs -----------------
def ADD {       // Inst
  string Mnemonic = "add";
  int Opcode = 160;
}
def ADDI {      // Inst
  string Mnemonic = "addi";
  int Opcode = 161;
}
```

使用多类，一次生成多个记录非常容易。这个特性被非常频繁地使用！

记录不需要有名称。匿名记录完全可以接受。要定义一个匿名记录，只需省略名称即可。由多类生成的记录名称由两个名称组成，创建一个命名记录时必须提供这两个名称。如果在`defm`之后省略名称，则只会创建匿名记录。同样，如果多类内部的`def`后面没有跟名称，也会创建一个匿名记录。这就是为什么在多类示例中的第一个定义使用了空名称`""`：没有它，记录将是匿名的。

## 模拟函数调用

在某些情况下，使用类似于前例中的多类可能会导致重复。假设 CPU 还支持内存操作数，方式与立即操作数类似。你可以通过向多类中添加一个新的记录定义来支持这一点：

```cpp

multiclass InstWithOps<string mnemonic, int opcode> {
  def "": Inst<mnemonic, opcode>;
  def "I": Inst<!strconcat(mnemonic,"i"), !add(opcode, 1)>;
  def "M": Inst<!strconcat(mnemonic,"m"), !add(opcode, 2)>;
}
```

这完全没问题。但现在，想象一下你不需要定义 3 个记录，而是需要定义 16 个记录，并且需要多次这样做。这种情况可能出现的典型场景是当 CPU 支持许多向量类型，并且向量指令根据使用的类型略有不同。

请注意，所有带有`def`语句的三行具有相同的结构。变化仅在于名称和助记符的后缀，以及将 delta 值添加到操作码中。在 C 语言中，你可以将数据放入一个数组中，并实现一个基于索引值返回数据的函数。然后，你可以创建一个循环来遍历数据，而不是手动重复语句。

令人惊讶的是，你可以在 TableGen 语言中做类似的事情！以下是转换示例的方式：

1.  为了存储数据，你定义一个包含所有必需字段的类。这个类被称为`InstDesc`，因为它描述了指令的一些属性：

    ```cpp

    class InstDesc<string name, string suffix, int delta> {
      string Name = name;
      string Suffix = suffix;
      int Delta = delta;
    }
    ```

1.  现在，你可以为每种操作数类型定义记录。请注意，它精确地捕捉到观察到的数据中的差异：

    ```cpp

    def RegOp : InstDesc<"", "", 0>;
    def ImmOp : InstDesc<"I", """, 1>;
    def MemOp : InstDesc"""","""", 2>;
    ```

1.  假设你有一个枚举数字`0`、`1`和`2`的循环，并且你想根据索引选择之前定义的其中一个记录。你该如何做？解决方案是创建一个`getDesc`类，它接受索引作为参数。它有一个单一的字段`ret`，你可以将其解释为返回值。为了将正确的值分配给此字段，使用了`!cond`运算符：

    ```cpp

    class getDesc<int n> {
      InstDesc ret = !cond(!eq(n, 0) : RegOp,
                           !eq(n, 1) : ImmOp,
                           !eq(n, 2) : MemOp);
    }
    ```

    此运算符的工作方式与 C 语言中的`switch`/`case`语句类似。

1.  现在，你准备好定义多类。TableGen 语言有一个`loop`语句，它还允许我们定义变量。但请记住，没有动态执行！因此，循环范围是静态定义的，你可以给变量赋值，但之后不能改变这个值。然而，这足以检索数据。请注意，使用`getDesc`类的方式类似于函数调用。但没有函数调用！相反，创建了一个匿名记录，值是从该记录中取出的。最后，过去操作符（`#`）执行字符串连接，类似于之前使用的`!strconcat`操作符：

    ```cpp

    multiclass InstWithOps<string mnemonic, int opcode> {
      foreach I = 0-2 in {
        defvar Name = getDesc<I>.ret.Name;
        defvar Suffix = getDesc<I>.ret.Suffix;
        defvar Delta = getDesc<I>.ret.Delta;
        def Name: Inst<mnemonic # Suffix,
                       !add(opcode, Delta)>;
      }
    }
    ```

1.  现在，你使用多类定义记录，就像之前一样：

    ```cpp

    defm ADD : InstWithOps<"add", 0xA0>;
    ```

请运行`llvm-tblgen`并检查记录。除了各种`ADD`记录外，你还会看到一些由`getDesc`类使用生成的匿名记录。

这种技术被用于几个 LLVM 后端的指令定义中。凭借你获得的知识，你应该没有问题理解这些文件。

`foreach`语句使用`0-2`语法来表示范围的界限。这被称为`0...3`，如果数字是负数时很有用。最后，你不仅限于数值范围；你还可以遍历元素列表，这允许你使用字符串或先前定义的记录。例如，你可能喜欢使用`foreach`语句，但认为使用`getDesc`类太复杂。在这种情况下，遍历`InstDesc`记录是解决方案：

```cpp

multiclass InstWithOps<string mnemonic, int opcode> {
  foreach I = [RegOp, ImmOp, MemOp] in {
    defvar Name = I.Name;
    defvar Suffix = I.Suffix;
    defvar Delta = I.Delta;
    def Name: Inst<mnemonic # Suffix, !add(opcode, Delta)>;
  }
}
```

到目前为止，你只使用 TableGen 语言定义了记录，使用了最常用的语句。在下一节中，你将学习如何从 TableGen 语言中定义的记录生成 C++源代码。

# 从 TableGen 文件生成 C++代码

在上一节中，你使用 TableGen 语言定义了记录。为了使用这些记录，你需要编写自己的 TableGen 后端，该后端可以生成 C++源代码或使用记录作为输入执行其他操作。

在*第三章*，“将源文件转换为抽象语法树”，`Lexer`类的实现使用数据库文件来定义标记和关键字。各种查询函数都利用了那个数据库文件。除此之外，数据库文件还用于实现关键字过滤器。关键字过滤器是一个哈希表，使用`llvm::StringMap`类实现。每当找到一个标识符时，都会调用关键字过滤器来检查该标识符是否实际上是一个关键字。如果你仔细查看*第六章*“高级 IR 生成”中使用的`ppprofiler`传递的实现，你会发现这个函数被调用得相当频繁。因此，尝试不同的实现来使该功能尽可能快可能是有用的。

然而，这并不像看起来那么简单。例如，你可以尝试用二分搜索替换哈希表中的查找。这要求数据库文件中的关键字是有序的。目前这似乎是正确的，但在开发过程中，可能会在不被发现的情况下在错误的位置添加一个新的关键字。确保关键字顺序正确的方法是添加一些在运行时检查顺序的代码。

你可以通过改变内存布局来加速标准的二分搜索。例如，你不必对关键字进行排序，可以使用 Eytzinger 布局，该布局按广度优先顺序枚举搜索树。这种布局增加了数据的缓存局部性，因此加快了搜索速度。就个人而言，在数据库文件中手动以广度优先顺序维护关键字是不可能的。

另一种流行的搜索方法是生成最小完美哈希函数。如果你将一个新的键插入到像 `llvm::StringMap` 这样的动态哈希表中，那么这个键可能会映射到一个已经占用的槽位。这被称为 `gperf` GNU 工具。

总结来说，有一些动力能够从关键字生成查找函数。因此，让我们将数据库文件移动到 TableGen！

## 在 TableGen 语言中定义数据

`TokenKinds.def` 数据库文件定义了三个不同的宏。`TOK` 宏用于没有固定拼写的标记，例如用于整型字面量。`PUNCTUATOR` 宏用于所有类型的标点符号，并包含一个首选拼写。最后，`KEYWORD` 宏定义了一个由字面量和标志组成的关键字，该标志用于指示这个字面量在哪个语言级别上是关键字。例如，`thread_local` 关键字被添加到 C++11 中。

在 TableGen 语言中表达这一点的办法是创建一个 `Token` 类来保存所有数据。然后你可以添加该类的子类以使使用更加方便。你还需要一个 `Flag` 类来定义与关键字一起定义的标志。最后，你需要一个类来定义关键字过滤器。这些类定义了基本的数据结构，并且可以在其他项目中潜在地重用。因此，你为它创建了一个 `Keyword.td` 文件。以下是步骤：

1.  标志被建模为一个名称和一个相关联的值。这使得从这个数据生成枚举变得容易：

    ```cpp

    class Flag<string name, int val> {
        string Name = name;
        int Val = val;
    }
    ```

1.  `Token` 类用作基类。它只携带一个名称。请注意，这个类没有参数：

    ```cpp

    class Token {
        string Name;
    }
    ```

1.  `Tok` 类与数据库文件中相应的 `TOK` 宏具有相同的功能。它表示一个没有固定拼写的标记。它从基类 `Token` 继承，并仅添加了名称的初始化：

    ```cpp

    class Tok<string name> : Token {
        let Name = name;
    }
    ```

1.  同样地，`Punctuator` 类类似于 `PUNCTUATOR` 宏。它为标记的拼写添加了一个字段：

    ```cpp

    class Punctuator<string name, string spelling> : Token {
        let Name = name;
        string Spelling = spelling;
    }
    ```

1.  最后，`Keyword` 类需要一个标志列表：

    ```cpp

    class Keyword<string name, list<Flag> flags> : Token {
        let Name = name;
        list<Flag> Flags = flags;
    }
    ```

1.  在这些定义到位后，你现在可以定义一个名为`TokenFilter`的关键字过滤器类。它接受一个标记列表作为参数：

    ```cpp

    class TokenFilter<list<Token> tokens> {
        string FunctionName;
        list<Token> Tokens = tokens;
    }
    ```

使用这些类定义，你当然能够从`TokenKinds.def`数据库文件中捕获所有数据。TinyLang 语言不利用标志，因为只有这个语言版本。现实世界的语言，如 C 和 C++，已经经历了几次修订，并且通常需要标志。因此，我们以 C 和 C++的关键字为例。让我们创建一个`KeywordC.td`文件，如下所示：

1.  首先，你包含之前创建的类定义：

    ```cpp

    Include "Keyword.td"
    ```

1.  接下来，你定义标志。标志的值是标志的二进制值。注意`!or`运算符是如何用来为`KEYALL`标志创建值的：

    ```cpp

    def KEYC99  : Flag<"KEYC99", 0x1>;
    def KEYCXX  : Flag<"KEYCXX", 0x2>;
    def KEYCXX11: Flag<"KEYCXX11", 0x4>;
    def KEYGNU  : Flag<"KEYGNU", 0x8>;
    def KEYALL  : Flag<"KEYALL",
                       !or(KEYC99.Val, KEYCXX.Val,
                           KEYCXX11.Val , KEYGNU.Val)>;
    ```

1.  有些标记没有固定的拼写——例如，一个注释：

    ```cpp

    def : Tok<"comment">;
    ```

1.  运算符使用`Punctuator`类定义，就像这个例子一样：

    ```cpp

    def : Punctuator<"plus", "+">;
    def : Punctuator<"minus", "-">;
    ```

1.  关键字需要使用不同的标志：

    ```cpp

    def kw_auto: Keyword<"auto", [KEYALL]>;
    def kw_inline: Keyword<"inline", [KEYC99,KEYCXX,KEYGNU]>;
    def kw_restrict: Keyword<"restrict", [KEYC99]>;
    ```

1.  最后，这是关键字过滤器的定义：

    ```cpp

    def : TokenFilter<[kw_auto, kw_inline, kw_restrict]>;
    ```

当然，这个文件并没有包含 C 和 C++中所有的标记。然而，它展示了定义的 TableGen 类所有可能的用法。

基于这些 TableGen 文件，你将在下一节实现 TableGen 后端。

## 实现 TableGen 后端

由于解析和记录的创建是通过 LLVM 库完成的，你只需要关注后端实现，这主要是由基于记录信息生成 C++源代码片段组成的。首先，你需要明确要生成什么源代码，然后才能将其放入后端。

### 绘制要生成的源代码草图

TableGen 工具的输出是一个包含 C++片段的单个文件。这些片段由宏保护。目标是替换`TokenKinds.def`数据库文件。根据 TableGen 文件中的信息，你可以生成以下内容：

1.  用于定义标志的枚举成员。开发者可以自由命名类型；然而，它应该基于`unsigned`类型。如果生成的文件命名为`TokenKinds.inc`，那么预期的用途如下：

    ```cpp

    enum Flags : unsigned {
    #define GET_TOKEN_FLAGS
    #include "TokenKinds.inc"
    }
    ```

1.  `TokenKind`枚举，以及`getTokenName()`、`getPunctuatorSpelling()`和`getKeywordSpelling()`函数的原型和定义。这段代码替换了`TokenKinds.def`数据库文件，大多数`TokenKinds.h`包含文件和`TokenKinds.cpp`源文件。

1.  一个新的`lookupKeyword()`函数，它可以用来替代当前使用`llvm::StringMap`类型的实现。这是你想要优化的函数。

了解你想要生成的内容后，你现在可以转向实现后端。

### 创建一个新的 TableGen 工具

为您的新工具创建一个简单的结构，可以有一个驱动程序来评估命令行选项，并在不同的文件中调用生成函数和实际的生成函数。让我们将驱动程序文件命名为 `TableGen.cpp`，将包含生成器的文件命名为 `TokenEmitter.cpp`。您还需要一个 `TableGenBackends.h` 头文件。让我们从在 `TokenEmitter.cpp` 文件中生成 C++ 代码开始实现：

1.  如同往常，文件以包含所需的头文件开始。其中最重要的是 `llvm/TableGen/Record.h`，它定义了一个 `Record` 类，用于存储由解析 `.td` 文件生成的记录：

    ```cpp

    #include "TableGenBackends.h"
    #include "llvm/Support/Format.h"
    #include "llvm/TableGen/Record.h"
    #include "llvm/TableGen/TableGenBackend.h"
    #include <algorithm>
    ```

1.  为了简化编码，导入了 `llvm` 命名空间：

    ```cpp

    using namespace llvm;
    ```

1.  `TokenAndKeywordFilterEmitter` 类负责生成 C++ 源代码。`emitFlagsFragment()`、`emitTokenKind()` 和 `emitKeywordFilter()` 方法发出源代码，正如上一节中所述的 *绘制要生成的源代码*。唯一的公共方法是 `run()`，它调用所有代码发出方法。记录存储在 `RecordKeeper` 实例中，该实例作为参数传递给构造函数。该类位于匿名命名空间内：

    ```cpp

    namespace {
    class TokenAndKeywordFilterEmitter {
      RecordKeeper &Records;
    public:
      explicit TokenAndKeywordFilterEmitter(RecordKeeper &R)
          : Records(R) {}
      void run(raw_ostream &OS);
    private:
      void emitFlagsFragment(raw_ostream &OS);
      void emitTokenKind(raw_ostream &OS);
      void emitKeywordFilter(raw_ostream &OS);
    };
    } // End anonymous namespace
    ```

1.  `run()` 方法调用所有发出方法。它还记录了每个阶段的长度。您指定 `--time-phases` 选项，然后所有代码生成完成后会显示计时：

    ```cpp

    void TokenAndKeywordFilterEmitter::run(raw_ostream &OS) {
      // Emit Flag fragments.
      Records.startTimer("Emit flags");
      emitFlagsFragment(OS);
      // Emit token kind enum and functions.
      Records.startTimer("Emit token kind");
      emitTokenKind(OS);
      // Emit keyword filter code.
      Records.startTimer("Emit keyword filter");
      emitKeywordFilter(OS);
      Records.stopTimer();
    }
    ```

1.  `emitFlagsFragment()` 方法展示了函数发出 C++ 源代码的典型结构。生成的代码由 `GET_TOKEN_FLAGS` 宏保护。要发出 C++ 源代码片段，您需要遍历 TableGen 文件中从 `Flag` 类派生的所有记录。拥有这样的记录后，查询记录的名称和值就变得很容易。请注意，名称 `Flag`、`Name` 和 `Val` 必须与 TableGen 文件中的完全一致。如果您在 TableGen 文件中将 `Val` 重命名为 `Value`，那么您也需要更改此函数中的字符串。所有生成的源代码都写入提供的流 `OS` 中：

    ```cpp

    void TokenAndKeywordFilterEmitter::emitFlagsFragment(
        raw_ostream &OS) {
      OS << "#ifdef GET_TOKEN_FLAGS\n";
      OS << "#undef GET_TOKEN_FLAGS\n";
      for (Record *CC :
           Records.getAllDerivedDefinitions("Flag")) {
        StringRef Name = CC->getValueAsString("Name");
        int64_t Val = CC->getValueAsInt("Val");
        OS << Name << " = " << format_hex(Val, 2) << ",\n";
      }
      OS << "#endif\n";
    }
    ```

1.  `emitTokenKind()` 方法发出标记分类函数的声明和定义。让我们先看看如何发出声明。整体结构与上一个方法相同——只是发出的 C++ 源代码更多。生成的源代码片段由 `GET_TOKEN_KIND_DECLARATION` 宏保护。请注意，此方法试图生成格式良好的 C++ 代码，使用换行和缩进来模拟人类开发者。如果发出的源代码不正确，并且您需要检查它以找到错误，这将非常有帮助。这样的错误也很容易犯：毕竟，您正在编写一个发出 C++ 源代码的 C++ 函数。

    首先，发出 `TokenKind` 枚举。关键字的名称应该以 `kw_` 字符串为前缀。循环遍历 `Token` 类的所有记录，您可以查询记录是否也是 `Keyword` 类的子类，这使您能够发出前缀：

    ```cpp

      OS << "#ifdef GET_TOKEN_KIND_DECLARATION\n"
         << "#undef GET_TOKEN_KIND_DECLARATION\n"
         << "namespace tok {\n"
         << "  enum TokenKind : unsigned short {\n";
      for (Record *CC :
           Records.getAllDerivedDefinitions("Token")) {
        StringRef Name = CC->getValueAsString("Name");
        OS << "    ";
        if (CC->isSubClassOf("Keyword"))
          OS << "kw_";
        OS << Name << ",\n";
      }
      OS << „    NUM_TOKENS\n"
         << „  };\n";
    ```

1.  接下来，发出函数声明。这只是一个常量字符串，所以没有发生什么激动人心的事情。这完成了声明的发出：

    ```cpp

      OS << "  const char *getTokenName(TokenKind Kind) "
            "LLVM_READNONE;\n"
         << "  const char *getPunctuatorSpelling(TokenKind "
            "Kind) LLVM_READNONE;\n"
         << "  const char *getKeywordSpelling(TokenKind "
            "Kind) "
            "LLVM_READNONE;\n"
         << "}\n"
         << "#endif\n";
    ```

1.  现在，让我们转向发出定义。同样，生成的代码由一个名为 `GET_TOKEN_KIND_DEFINITION` 的宏保护。首先，令牌名称被发出到 `TokNames` 数组中，`getTokenName()` 函数使用该数组来检索名称。请注意，当在字符串内部使用时，引号符号必须转义为 `\"`：

    ```cpp

      OS << "#ifdef GET_TOKEN_KIND_DEFINITION\n";
      OS << "#undef GET_TOKEN_KIND_DEFINITION\n";
      OS << "static const char * const TokNames[] = {\n";
      for (Record *CC :
           Records.getAllDerivedDefinitions("Token")) {
        OS << "  \"" << CC->getValueAsString("Name")
           << "\",\n";
      }
      OS << "};\n\n";
      OS << "const char *tok::getTokenName(TokenKind Kind) "
            "{\n"
         << "  if (Kind <= tok::NUM_TOKENS)\n"
         << "    return TokNames[Kind];\n"
         << "  llvm_unreachable(\"unknown TokenKind\");\n"
         << "  return nullptr;\n"
         << "};\n\n";
    ```

1.  接下来，发出 `getPunctuatorSpelling()` 函数。与其他部分相比，唯一的显著区别是循环遍历从 `Punctuator` 类派生的所有记录。此外，生成一个 `switch` 语句而不是数组：

    ```cpp

      OS << "const char "
            "*tok::getPunctuatorSpelling(TokenKind "
            "Kind) {\n"
         << "  switch (Kind) {\n";
      for (Record *CC :
           Records.getAllDerivedDefinitions("Punctuator")) {
        OS << "    " << CC->getValueAsString("Name")
           << ": return \""
           << CC->getValueAsString("Spelling") << "\";\n";
      }
      OS << "    default: break;\n"
         << "  }\n"
         << "  return nullptr;\n"
         << "};\n\n";
    ```

1.  最后，发出 `getKeywordSpelling()` 函数。编码与发出 `getPunctuatorSpelling()` 类似。这次，循环遍历 `Keyword` 类的所有记录，并且名称再次以 `kw_` 前缀：

    ```cpp

      OS << "const char *tok::getKeywordSpelling(TokenKind "
            "Kind) {\n"
         << "  switch (Kind) {\n";
      for (Record *CC :
           Records.getAllDerivedDefinitions("Keyword")) {
        OS << "    kw_" << CC->getValueAsString("Name")
           << ": return \"" << CC->getValueAsString("Name")
           << "\";\n";
      }
      OS << "    default: break;\n"
         << "  }\n"
         << "  return nullptr;\n"
         << «};\n\n»;
      OS << «#endif\n»;
    }
    ```

1.  `emitKeywordFilter()` 方法比之前的方法更复杂，因为发出过滤器需要从记录中收集一些数据。生成的源代码使用 `std::lower_bound()` 函数，从而实现二分搜索。

    现在，让我们简化一下。在 TableGen 文件中可以定义多个 `TokenFilter` 类的记录。为了演示目的，只需发出最多一个令牌过滤器方法：

    ```cpp

      std::vector<Record *> AllTokenFilter =
          Records.getAllDerivedDefinitionsIfDefined(
              "TokenFilter");
      if (AllTokenFilter.empty())
        return;
    ```

1.  用于过滤的关键字位于名为 `Tokens` 的列表中。为了访问该列表，您首先需要查找记录中的 `Tokens` 字段。这返回一个指向 `RecordVal` 类实例的指针，您可以通过调用方法 `getValue()` 从该实例中检索 `Initializer` 实例。`Tokens` 字段定义为列表，因此您将初始化器实例转换为 `ListInit`。如果失败，则退出函数：

    ```cpp

      ListInit *TokenFilter = dyn_cast_or_null<ListInit>(
          AllTokenFilter[0]
              ->getValue("Tokens")
              ->getValue());
      if (!TokenFilter)
        return;
    ```

1.  现在，您已经准备好构建一个过滤器表。对于存储在 `TokenFilter` 列表中的每个关键字，您需要 `Flag` 字段的名称和值。该字段再次定义为列表，因此您需要遍历这些元素来计算最终值。结果名称/标志值对存储在 `Table` 向量中：

    ```cpp

      using KeyFlag = std::pair<StringRef, uint64_t>;
      std::vector<KeyFlag> Table;
      for (size_t I = 0, E = TokenFilter->size(); I < E;
           ++I) {
       Record *CC = TokenFilter->getElementAsRecord(I);
       StringRef Name = CC->getValueAsString("Name");
       uint64_t Val = 0;
       ListInit *Flags = nullptr;
       if (RecordVal *F = CC->getValue("Flags"))
          Flags = dyn_cast_or_null<ListInit>(F->getValue());
       if (Flags) {
          for (size_t I = 0, E = Flags->size(); I < E; ++I) {
            Val |=
                Flags->getElementAsRecord(I)->getValueAsInt(
                    "Val");
          }
       }
       Table.emplace_back(Name, Val);
      }
    ```

1.  为了能够执行二分搜索，该表需要排序。比较函数由一个 lambda 函数提供：

    ```cpp

      llvm::sort(Table.begin(), Table.end(),
                 [](const KeyFlag A, const KeyFlag B) {
                   return A.first < B.first;
                 });
    ```

1.  现在，您可以发出 C++ 源代码。首先，您需要发出包含关键字名称和相关标志值的排序表：

    ```cpp

      OS << "#ifdef GET_KEYWORD_FILTER\n"
         << "#undef GET_KEYWORD_FILTER\n";
      OS << "bool lookupKeyword(llvm::StringRef Keyword, "
            "unsigned &Value) {\n";
      OS << "  struct Entry {\n"
         << "    unsigned Value;\n"
         << "    llvm::StringRef Keyword;\n"
         << "  };\n"
         << "static const Entry Table[" << Table.size()
         << "] = {\n";
      for (const auto &[Keyword, Value] : Table) {
       OS << "    { " << Value << ", llvm::StringRef(\""
          << Keyword << "\", " << Keyword.size()
          << ") },\n";
      }
      OS << "  };\n\n";
    ```

1.  接下来，你使用`std::lower_bound()`标准 C++函数在排序表中查找关键字。如果关键字在表中，则`Value`参数接收与关键字关联的标志值，函数返回`true`。否则，函数简单地返回`false`：

    ```cpp

      OS << "  const Entry *E = "
            "std::lower_bound(&Table[0], "
            "&Table["
         << Table.size()
         << "], Keyword, [](const Entry &A, const "
            "StringRef "
            "&B) {\n";
      OS << "    return A.Keyword < B;\n";
      OS << "  });\n";
      OS << "  if (E != &Table[" << Table.size()
         << "]) {\n";
      OS << "    Value = E->Value;\n";
      OS << "    return true;\n";
      OS << "  }\n";
      OS << "  return false;\n";
      OS << "}\n";
      OS << "#endif\n";
    }
    ```

1.  现在唯一缺少的部分是调用此实现的方法，为此你定义了一个全局函数`EmitTokensAndKeywordFilter()`。在`llvm/TableGen/TableGenBackend.h`头文件中声明的`emitSourceFileHeader()`函数在生成的文件顶部输出一个注释：

    ```cpp

    void EmitTokensAndKeywordFilter(RecordKeeper &RK,
                                    raw_ostream &OS) {
      emitSourceFileHeader("Token Kind and Keyword Filter "
                           "Implementation Fragment",
                           OS);
      TokenAndKeywordFilterEmitter(RK).run(OS);
    }
    ```

有了这些，你就在`TokenEmitter.cpp`文件中完成了源发射器的实现。总体来说，代码并不复杂。

`TableGenBackends.h`头文件只包含`EmitTokensAndKeywordFilter()`函数的声明。为了避免包含其他文件，你使用前向声明为`raw_ostream`和`RecordKeeper`类：

```cpp

#ifndef TABLEGENBACKENDS_H
#define TABLEGENBACKENDS_H
namespace llvm {
class raw_ostream;
class RecordKeeper;
} // namespace llvm
void EmitTokensAndKeywordFilter(llvm::RecordKeeper &RK,
                                llvm::raw_ostream &OS);
#endif
```

缺失的部分是驱动程序的实现。其任务是解析 TableGen 文件并根据命令行选项输出记录。实现位于`TableGen.cpp`文件中：

1.  如同往常，实现从包含所需的头文件开始。最重要的是`llvm/TableGen/Main.h`，因为这个头文件声明了 TableGen 的前端：

    ```cpp

    #include "TableGenBackends.h"
    #include "llvm/Support/CommandLine.h"
    #include "llvm/Support/PrettyStackTrace.h"
    #include "llvm/Support/Signals.h"
    #include "llvm/TableGen/Main.h"
    #include "llvm/TableGen/Record.h"
    ```

1.  为了简化编码，导入了`llvm`命名空间：

    ```cpp

    using namespace llvm;
    ```

1.  用户可以选择一个操作。`ActionType`枚举包含所有可能的操作：

    ```cpp

    enum ActionType {
      PrintRecords,
      DumpJSON,
      GenTokens,
    };
    ```

1.  使用一个名为`Action`的单个命令行选项对象。用户需要指定`--gen-tokens`选项来输出你实现的令牌过滤器。其他两个选项`--print-records`和`--dump-json`是用于输出读取记录的标准选项。注意，该对象位于匿名命名空间中：

    ```cpp

    namespace {
    cl::opt<ActionType> Action(
        cl::desc("Action to perform:"),
        cl::values(
            clEnumValN(
                PrintRecords, "print-records",
                "Print all records to stdout (default)"),
            clEnumValN(DumpJSON, "dump-json",
                       "Dump all records as "
                       "machine-readable JSON"),
            clEnumValN(GenTokens, "gen-tokens",
                       "Generate token kinds and keyword "
                       "filter")));
    ```

1.  `Main()`函数根据`Action`的值执行请求的操作。最重要的是，如果命令行中指定了`--gen-tokens`，则会调用你的`EmitTokensAndKeywordFilter()`函数。函数结束后，匿名命名空间关闭：

    ```cpp

    bool Main(raw_ostream &OS, RecordKeeper &Records) {
      switch (Action) {
      case PrintRecords:
        OS << Records; // No argument, dump all contents
        break;
      case DumpJSON:
        EmitJSON(Records, OS);
        break;
      case GenTokens:
        EmitTokensAndKeywordFilter(Records, OS);
        break;
      }
      return false;
    }
    } // namespace
    ```

1.  最后，你定义了一个`main()`函数。在设置堆栈跟踪处理程序和解析命令行选项后，调用`TableGenMain()`函数来解析 TableGen 文件并创建记录。如果没有任何错误，该函数还会调用你的`Main()`函数：

    ```cpp

    int main(int argc, char **argv) {
      sys::PrintStackTraceOnErrorSignal(argv[0]);
      PrettyStackTraceProgram X(argc, argv);
      cl::ParseCommandLineOptions(argc, argv);
      llvm_shutdown_obj Y;
      return TableGenMain(argv[0], &Main);
    }
    ```

你自己的 TableGen 工具现在已经实现。编译后，你可以使用`KeywordC.td`样本输入文件运行它，如下所示：

```cpp

$ tinylang-tblgen --gen-tokens –o TokenFilter.inc KeywordC.td
```

生成的 C++源代码被写入`TokenFilter.inc`文件。

令牌过滤器的性能

使用简单的二分搜索进行关键字过滤器搜索并不比基于`llvm::StringMap`类型的实现有更好的性能。要超越当前实现的性能，你需要生成一个完美的哈希函数。

来自捷克共和国的 Havas 和 Majewski 的经典算法可以轻松实现，并且提供了非常好的性能。它描述在*生成最小完美哈希函数的最优算法*，*信息处理信件*，*第 43 卷*，*第 5 期*，*1992 年*。见 https://www.sciencedirect.com/science/article/abs/pii/002001909290220P。

最先进的算法是 Pibiri 和 Trani 的 PTHash，在*PTHash：重新审视 FCH 最小完美哈希*，*SIGIR’21*中描述。见[`arxiv.org/pdf/2104.10402.pdf`](https://arxiv.org/pdf/2104.10402.pdf)。

这两种算法都是生成一个比`llvm::StringMap`实际更快的标记过滤器的好候选。

# TableGen 的缺点

这里有一些 TableGen 的缺点：

+   TableGen 语言建立在简单概念之上。因此，它不具备其他 DSLs 相同的计算能力。显然，一些程序员希望用一种不同、更强大的语言来替换 TableGen，这个话题在 LLVM 讨论论坛上时不时会出现。

+   有可能实现自己的后端，TableGen 语言非常灵活。然而，这也意味着给定定义的语义隐藏在后台中。因此，你可以创建其他开发者基本上无法理解的 TableGen 文件。

+   最后，如果你尝试解决一个非平凡的任务，后端实现可能会非常复杂。如果 TableGen 语言更加强大，预期这种努力会降低，这是合理的。

即使不是所有开发者都对 TableGen 的功能感到满意，这个工具在 LLVM 中仍然被广泛使用，对于开发者来说，理解它是很重要的。

# 摘要

在本章中，你首先学习了 TableGen 背后的主要思想。然后，你在 TableGen 语言中定义了你的第一个类和记录，并获得了 TableGen 语法的知识。最后，你基于定义的 TableGen 类开发了一个生成 C++源代码片段的 TableGen 后端。

在下一章中，我们将探讨 LLVM 的另一个独特特性：一步生成和执行代码，也称为**即时编译**（**JIT**）。
