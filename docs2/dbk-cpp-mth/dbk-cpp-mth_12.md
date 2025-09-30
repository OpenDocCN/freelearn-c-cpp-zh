

# Rust 将取代 C++

*如果 4 件事情* *同时发生*

Rust 在过去几年中作为系统编程和 C++的竞争者崛起。这有很好的理由：Rust 是一种现代语言，提供了一套良好的工具集、简单的语法以及有助于推理代码的创新。因此，Rust 是否会取代 C++的问题，许多希望了解自己职业未来投资方向的程序员都在思考。接下来，我们将探讨 Rust 的有趣之处以及它需要发生什么才能取代 C++。

在本章中，我们将涵盖以下主要主题：

+   为什么有竞争？

+   Rust 的核心特性

+   Rust 的优势

+   C++的优势在哪里

+   C++还需要什么

# 技术要求

本章的代码可在 GitHub 仓库（[`github.com/PacktPublishing/Debunking-C-Myths`](https://github.com/PacktPublishing/Debunking-C-Myths)）中的**ch12**文件夹找到。要运行代码，您需要 Rust，并按照他们网站上的说明操作：

[`www.rust-lang.org/tools/install`](https://www.rust-lang.org/tools/install)

# 为什么有竞争？

作为大约 2001 年在巴黎工作的初级 C++程序员，我最大的挑战是让我的代码完成所需的功能。该项目是一个工业印刷机的知识库，允许操作员识别打印错误的原因。当时，此类桌面应用的主要选项是在 Windows 下使用 C++，通过 Visual C++、**微软基础类库**（**MFC**）和 Windows API 开发，后者是微软推广的模型-视图-控制器（Model-View-Controller）的一个较弱的分支。这个项目让我面临了极大的挑战：我不仅要与 C++的内存管理作斗争，还要处理 MFC 和 Windows API 的怪癖。那时的支持主要来自官方文档、[`codeproject.com`](https://codeproject.com)网站以及一位很少有空闲的资深同事。基本上，我必须作为一个独立开发者，在没有太多支持的情况下处理复杂的技术。欢迎来到 21 世纪初的软件开发！请别误会，我并不是在抱怨：正是因为其挑战，这段经历对我来说非常有帮助，并且具有教育意义。

在那个阶段，我唯一关注的是我正在使用的科技。我听说过 PHP，之前也用 Java 开发过小程序和 Web 应用，但 C++、MFC 和 Windows API 占据了大部分精力。通勤大约需要 90 分钟，足够一年内在公共交通工具上读完整本《指环王》。

在我的职业生涯中，第二个重要的项目完全不同：仍然是 C++，但在一个 NoSQL 数据库引擎被命名之前，采用了一种非常结构化和指导性的方法来构建。当时，我学会了如何编写测试，因为我们没有为 C++编写自己的测试引擎。通过编写设计文档并与同事审查它们，我学到了很多关于软件设计的东西。我学习了代码审查。通过深入研究包括 Scott Meyers 的《Effective C++》和《More Effective C++》以及 Andrei Alexandrescu 的《Modern C++ Design》等经典书籍，我深入了解了 C++的方方面面。因此，我甚至更深入地研究了同一技术。

然后，C#出现了，我决定转换技术。在做过一些 Java，对 C++有深入的了解，并以一种结构化的方式学习 C#之后，我意识到两件事：转换技术做得越多，就越容易，每种技术都有其自身的优缺点。在 C#中构建桌面应用程序要容易得多，因为我们不必过多关注内存管理和其潜在问题。编程更有趣，更重要的是，我们开发速度更快。我们用这两个好处交换了两个缺点：更少的控制和编程方法上的不够严谨。

在我的职业生涯后期，我开始思考市场上可用的众多编程语言。据我估计，我们可能需要大约 5-7 种编程语言，纯粹出于技术原因：一种用于网页开发，一种用于系统编程，一种用于脚本编写，其余的用于各种细分市场，如人工智能、工作流程、解方程等。假设我错了，我们可能需要 20 种。然而，现实是，今天我们可以使用数百种编程语言，包括主流、细分和古怪的语言，如 Brainfuck 或 Whitespace。我们可以在 TIOBE 编程社区指数中看到许多这样的语言，该指数监控编程语言的流行度。为什么会有这么多？

我的最佳猜测是，这不仅仅是一个技术需求的问题，而是一个文化问题。当然，技术方面很重要。面向对象和后来的函数式编程特性被引入到所有主流语言中。安全性、并行性和并发性、编程的易用性、社区和生态系统都是编程语言的重要方面。然而，决定创建一种新的编程语言来自人们，他们在设计语言时所做的决定来自他们的个人偏好。文学和哲学的趋势遵循相同的模式：主流和逆流或反动。在文学中，浪漫主义是对古典主义的反应，现实主义是对浪漫主义的反应。在编程语言中也有类似的情况：Java 是对 C++的反应，Ruby on Rails 是对 Java 的反应。在文学中，主流部分由社会变革决定，而在技术中，主流既由景观中的运动决定，也由年轻一代程序员的偏好决定，这些偏好以非常高的速度增加。技术景观变化的例子是互联网的兴起，这有利于 Java 作为对 C++的回应，用于 Web 应用程序。有趣的是，计算从服务器到客户端的移动现在似乎有利于 Web Assembly 应用程序的出现，这些应用程序目前需要用 C++或 Rust 进行低级编程。至于新一代程序员，Ruby on Rails 在很大程度上是对感知到的旧式 Java 语言的反应。Rails 提供了 Java 没有的表达自由，以及随着进步而感到的满足感。这种感觉几乎没有技术基础，但技术方面对人们来说并不全是，甚至对软件开发者来说也是如此。

你现在应该能看出这是怎么回事：Rust 是对 C++的反应。它是对 C++当前技术烦恼的反应，也是对 C++做事方式的反应。因此，让我们看看 Rust 带来了什么。

# Rust 的核心特性

我们可以用来理解 Rust 核心特性的第一个地方是官方网站，[`www.rust-lang.org/`](https://www.rust-lang.org/)。该网站非常出色地强调了 Rust 最重要的特性：

+   快速且内存高效

+   无运行时

+   无垃圾回收器

+   与其他语言集成

+   通过丰富的类型系统和所有权模型实现内存安全和线程安全

+   优秀的文档

+   友好的编译器，带有有用的错误信息

+   集成软件包管理和构建工具

+   自动格式化

+   智能多编辑器支持，包括自动完成和类型检查

仅从这一描述中，我们就可以看到一些与 C++的相似之处，以及 C++当前状态的改进。这些相似之处在于控制级别：原生编译、没有垃圾回收器、速度和内存效率是 C++也吹嘘的品质。而不同之处则指向我们在本书中详细讨论过的事情：标准包管理器、标准工具和友好的编译器。最后这个品质对于收到大量错误信息的任何 C++程序员来说都是美妙的；我记得在 2000 年代，我在 Visual C++中遇到了一个错误，错误信息的大致内容是“错误信息太长，我们无法显示它们”。而今天的 C++更加友好，但在使用模板时找出哪里出了问题仍然是一件痛苦的事情。

然而，让我们超越网站首页上所写的内容。接下来，我们将看看一些我认为与 C++相比非常有用和有趣的功能。

## 项目模板和包管理

作为一名热衷于使用命令行和 neovim 代码编辑器的用户，我喜欢那些允许我从命令行直接创建项目的技术。Rust 附带了一个**cargo**工具，允许创建项目、构建、运行、打包和发布。要创建一个新的项目，只需调用**cargo new project-name**。你可以用**cargo run**运行它，用**cargo check**检查它是否有编译错误，用**cargo build**编译它，用–你猜对了！–**cargo package**打包它，并用（敲锣打鼓）...**cargo publish**发布它。

我们当然可以用**cargo**创建库和可执行文件。不仅如此，我们还可以使用位于[`cargo-generate.github.io/cargo-generate/`](https://cargo-generate.github.io/cargo-generate/)的 cargo generate 工具，从项目模板开始。

我知道这对大多数 C++开发者来说可能看起来不多，因为你们很少创建新的项目。这是我教 C++程序员单元测试或测试驱动开发时的一个惊喜：我们必须共同努力设置一个与生产项目相对应的测试项目以及相应的引用，这是我理所当然的事情。当我说这不仅在项目开始时很有用，而且在小型实验、个人或练习代码库中，以及减少编译时间方面也非常有用时，请相信我。如果你发现项目编译得太慢，C++为你提供的一个简单方法是创建一个新的编译单元，由你正在修改的少数文件组成，并将其余部分作为二进制文件引用。在我使用 SSD 硬盘大大加快编译速度之前，我广泛地使用了这种技术。

新项目已经足够了。让我们来写一些代码。让我们修改一些变量...或者也许不。

## 不可变性

Rust 默认使用不可变性。文档中的说法是 *“一旦值绑定到名称，就不能更改该值。”* 让我们看看一个简单的例子，我将一个字符串值赋给一个变量，显示它，然后尝试修改它：

```cpp
fn main() {
    let the_message = "Hello, world!";
    println!("{the_message}");
    the_message = "A new hello!";
    println!("{the_message}");
}
```

尝试编译这个程序会导致 **无法将不可变变量 `the_message` 赋值两次** 的编译错误。幸运的是，错误信息中包含了 **关于此错误的更多信息，请尝试 `rustc –explain E0384`** 的提示。错误信息的解释包含了一个错误示例，以及如何使变量可变的非常有帮助的提示：

**"默认情况下，Rust 中的变量是不可变的。要修复此错误，在声明变量时在 let 关键字后添加关键字 must。"**

以下是一个代码示例，当进行适配时，可以使程序编译：

```cpp
    let mut the_message = "Hello, world!";
    println!("{the_message}");
    the_message = "A new hello!";
    println!("{the_message}");
```

如您所见，可变变量必须指定为 **mut**，因此默认是不可变的。正如我们在前面的章节中看到的，这有助于解决许多问题，例如并行性和并发性、自动化测试以及代码的简洁性。

## 复合类型的简单语法

Rust 从像 Python 或 Ruby 这样的语言中借鉴了数组和解构的语法。下面是它的样子：

```cpp
let months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
println!("{:?}", months);
let (one, two) = (1, 1+1);
println!("{one} and {two}");
```

这可能看起来不多，但它有助于简化代码。

值得注意的是，C++ 在 C++ 11 中引入了类似的语法，并在后续版本中通过列表初始化和花括号进行了改进：

```cpp
std::vector<string> months = {"January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"};
```

我很希望看到这方面的进一步改进，但 C++ 的语法已经相当复杂，所以我不期望它会有所改变。

## 可选的返回关键字

Rust 中的函数允许返回函数中的最后一个值。下一个示例使用这个结构来增加一个数字：

```cpp
fn main() {
    let two = increment(1);
    println!("{two}");
}
fn increment(x:i32) -> i32{
    x+1
}
```

我通常避免在前面这样的函数中使用它，但避免使用 **return** 关键字可以简化闭包，正如我们接下来将要看到的。

## 闭包

让我们增加向量的所有元素：

```cpp
fn increment_all() -> Vec<i32>{
    let values : Vec<i32> = vec![1, 2, 3];
    return values.iter().map(|x| x+1).collect();
}
```

对于函数式编程结构，就像 C++ 中的 **ranges** 库一样，我们需要获取一个迭代器，调用 map 函数——在 C++ 中相当于转换算法——使用闭包，然后调用 **collect** 来获取结果。闭包有一个非常简单的语法，这是由可选的返回语句实现的。

## 标准库中的单元测试

单元测试是软件开发中非常重要的实践，令人惊讶的是，只有少数语言在标准库中提供了对它的支持。Rust 默认提供，而且使用起来相当简单。让我们添加一个单元测试来验证我们的 **increment_all** 函数是否按预期工作：

```cpp
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn it_works() {
        assert_eq!(vec![2, 3, 4], increment_all());
    }
}
```

作为一个加分项，我喜欢在同一个编译单元（在 Rust 中称为 **crate**）中编写单元测试，就像生产代码一样。如果你把单元测试看作是一项义务，这可能看起来不多，但我不经常使用单元测试来实验和设计，所以我非常喜欢这个功能。

## 特性

Rust（或 Go）与其他主流语言的一个重大区别是，Rust 不支持继承，而是更倾向于组合。为了在不使用继承的情况下实现多态行为，Rust 提供了特质。

Rust 特质在面向对象语言中类似于接口，因为它们定义了一组需要为从它们派生的每个对象实现的方法。然而，Rust 特质有一个特定的特性：你可以将特质添加到你并不拥有的类型上。这类似于 C#中的扩展方法，尽管并不完全相同。

Rust 文档通过使用两个结构体来举例说明特质，一个代表推文，另一个代表新闻文章，并将**Summary**特质添加到两者中，目的是创建相应消息的摘要。正如以下示例所示，特质的实现与结构体的实现和特质的定义是分开的，这使得它非常灵活。

让我们先看看这两个结构体。首先，**NewsArticle**包含了一些字段：

```cpp
pub struct NewsArticle {
    pub headline: String,
    pub location: String,
    pub author: String,
    pub content: String,
}
```

然后，**Tweet**结构体包含它自己的字段：

```cpp
pub struct Tweet {
    pub username: String,
    pub content: String,
    pub reply: bool,
    pub retweet: bool,
}
```

独立地，我们定义了一个带有单个方法`summarize`返回字符串的**Summary**特质：

```cpp
pub trait Summary {
    fn summarize(&self) -> String;
}
```

现在让我们为**Tweet**结构体实现**Summary**特质。这是通过指定这个特质的实现适用于结构体来完成的，如下所示：

```cpp
impl Summary for Tweet {
    fn summarize(&self) -> String {
        format!("{}: {}", self.username, self.content)
    }
}
```

测试工作得非常完美：

```cpp
    #[test]
    fn summarize_tweet() {
        let tweet = Tweet {
            username: String::from("me"),
            content: String::from("a message"),
            reply: false,
            retweet: false,
        };
        assert_eq!("me: a message", tweet.summarize());
    }
```

最后，让我们为新闻文章实现这个特质：

```cpp
impl Summary for NewsArticle {
    fn summarize(&self) -> String {
        format!("{}, by {} ({})", self.headline, self.author, self.location)
    }
}
#[test]
    fn summarize_news_article() {
        let news_article = NewsArticle {
            headline: String::from("Big News"),
            location: String::from("Undisclosed"),
            author: String::from("Me"),
            content: String::from("Big News here, must follow"),
        };
        assert_eq!("Big News, by Me (Undisclosed)", news_article.summarize());
    }
```

Rust 中的特质具有更多功能。我们可以实现默认行为，指定参数的类型需要是单个或多个特质类型，在多个类型上泛型实现特质，等等。实际上，Rust 特质是 OO 接口、C#扩展方法和 C++概念的组合。然而，这超出了本章的范围。值得记住的是，Rust 对继承的处理与 C++非常不同。

## 所有权模型

Rust 的一个有趣特性，也许是最受宣传的特性，就是所有权模型。这是 Rust 对 C++中内存安全问题的一种反应，但与 Java 或 C#中的垃圾收集器不同，设计者通过更明确的内存所有权来解决这个问题。我们将看看 Rust 书籍中的一段引文（[https://doc.rust-lang.org/book/ch04-01-what-is-ownership.html]）：）

“*内存通过一套规则进行管理，这些规则由编译器检查。如果违反了任何规则，程序将无法编译。在程序运行期间，所有权的任何特性都不会减慢你的程序速度。*”

Rust 中有三个所有权规则：

+   每个 Rust 中的值都有一个*所有者*

+   一次只能有一个所有者

+   当所有者超出作用域时，值将被丢弃

让我们首先看看一个与 C++中相同工作的示例。如果我们有一个在栈上分配的变量，比如一个整数，那么复制变量将以非常熟悉的方式工作：

```cpp
    #[test]
    fn copy_on_stack() {
        let stack_value = 1;
        let copied_stack_value = stack_value;
        assert_eq!(1, stack_value);
        assert_eq!(1, copied_stack_value);
    }
```

两个变量具有相同的值，正如预期的那样。然而，如果我们尝试用堆上分配的变量执行相同的代码，我们会得到一个错误：

```cpp
    #[test]
    fn copy_on_heap() {
        let heap_value = String::from("A string");
        let copied_heap_value = heap_value;
        assert_eq!(String::from("A string"), heap_value);
        assert_eq!(String::from("A string"), copied_heap_value);
    }
```

当运行这个程序时，我们得到**错误[E0382]: borrow of moved value: `heap_value`**错误。发生了什么？

好吧，当我们将**heap_value**的值赋给**copied_heap_value**时，**heap_value**变量就失效了。这和 C++中的移动语义行为相同，只是程序员不需要做任何额外的工作。在幕后，这是通过使用两个特质：**Copy**和**Drop**来实现的。如果一个类型实现了**Copy**特质，那么它就像第一个示例中那样工作，而如果一个类型实现了**Drop**特质，那么它就像第二个示例中那样工作。没有类型可以同时实现这两个特质。

为了使上述示例工作，我们需要克隆值而不是使用默认的移动机制：

```cpp
    #[test]
    fn clone_on_heap() {
        let heap_value = String::from("A string");
        let copied_heap_value = heap_value.clone();
        assert_eq!(String::from("A string"), heap_value);
        assert_eq!(String::from("A string"), copied_heap_value);
    }
```

这个示例工作得很好，所以值被克隆了。然而，这表明这是一个新的堆分配，而不是对相同值的引用。

移动语义对于函数调用也是相同的。让我们初始化一个值并将其传递给一个返回它未更改的函数，看看会发生什么：

```cpp
    fn call_me(value: String) -> String {
        return value;
    }
    #[test]
    fn move_semantics_method_call() {
        let heap_value = String::from("A string");
        let result = call_me(heap_value);
        assert_eq!(String::from("A string"), heap_value);
        assert_eq!(String::from("A string"), result);
    }
```

当尝试编译这段代码时，我们得到与之前相同的错误：**错误[E0382]: borrow of moved value: `heap_value`**。值是在堆上创建的，移动到**call_me**函数中，因此从当前作用域中删除。我们可以通过指定被调用的函数应该只借用所有权而不是接管它来使这段代码工作。这是通过使用引用和解除引用运算符来实现的，这与 C++中的相同：

```cpp
    fn i_borrow(value: &String) -> &String {
        return value;
    }
    #[test]
    fn borrow_method_call() {
        let heap_value = String::from("A string");
        let result = i_borrow(&heap_value);
        assert_eq!(String::from("A string"), heap_value);
        assert_eq!(String::from("A string"), *result);
    }
```

C++引用和 Rust 引用之间的重要区别是，Rust 引用默认是不可变的。

当然，关于 Rust 中的所有权模型还有很多东西要学习，但我相信这已经足够让你了解它是如何工作的，以及它是如何旨在防止内存安全问题的。

# Rust 的优势

总结来说，Rust 相对于 C++有一些优势。作为一个较新的语言，它有从其前辈学习并使用最佳模式的优势。我发现将不可变性与所有权模型结合起来，对于默认工作良好的代码来说非常好。由于它不是典型的内存管理风格，所以学习起来可能需要一点时间，但一旦你了解了如何使用它，它就允许你编写几乎无挑战性的代码。

标准库中的单元测试支持、包管理器和多编辑器支持应该是任何现代编程语言的一部分。当涉及到闭包和复合类型时，语法更优雅。

在这个阶段，我们可能会想：C++有机会吗？为什么，在哪里？

# C++的优势在哪里

C++是一种非常强大、先进的编程语言，正在不断改进。语言进步很快。很难与 C++生态系统相提并论：其社区、可用的库和框架数量惊人，以及教你如何以各种方式使用 C++解决任何可能问题的文章、博客和书籍。尽管有这些好处，与 C++相比，Rust 是一种较新的语言，这在考虑系统编程技术选择时应该让你三思。然而，Rust 已被用于 Linux 和 Android 的子系统，这证明了它是一个值得尊敬的竞争对手。

C++标准化委员会一直致力于简化语法和减轻程序员在处理各种代码结构时的心理负担。部分努力源于竞争，许多在 C++17 及以后版本中引入的特性是对 Rust 设计选择的回应。虽然我不期望 C++的语法会像 Rust 那样简单，但这里提到的其他因素也必须对选择产生同样甚至更大的影响。

# C++仍需改进之处

在本书中，我们看到了 C++的一些挑战。一个标准的包管理器将非常有帮助，即使社区效仿 Java 和 C#，选择一个开源的既定标准。一个标准的单元测试库将非常有益，即使现有的代码可能需要很长时间才能迁移，如果它真的迁移了的话。

Unicode 和 utf-8 的支持仍需改进。标准的多线程支持才刚刚开始。安全配置文件将非常有助于最小化内存安全问题。

从这份清单中可以看出，C++有很多需要改进的地方。好消息是标准化委员会正在努力解决这些问题。不那么好的消息是，定义这些改进需要时间，适应编译器需要更多时间，适应现有代码则需要更多时间。希望通用人工智能能够足够强大，以加快这些改进的速度，同时保持代码的完整性。

# 摘要

在本章中，我们看到了 Rust 是一个非常有趣的编程语言，其设计者知道如何利用前辈们积累的知识，并在正确的地方进行创新。结果是语法简洁，处理内存的方式更自然，无需使用垃圾回收器，整体开发体验现代化。我们在这章中探讨了这一点。

然而，C++很难与之竞争。世界上关于 C++的库、框架、博客、文章、代码示例、书籍和经验数量庞大，短时间内无法匹敌。Rust 在 Web Assembly 应用和各种工具中找到了自己的 niche，但它还远未取代 C++。

然而，我们还得记住，语言的选择并不一定基于技术原因，文化因素也同样重要。新一代的程序员可能比 C++更喜欢 Rust，而且随着国家安全局和白宫将焦点放在内存安全语言上，Rust 可能在新的项目中获得优势。

结论是什么？预测未来很难，但我们可以想象 Rust 如何接管。在我看来，这需要四个因素：越来越多的程序员选择 Rust，它受到法规的要求，C++在内存安全方面的进化速度不够快，以及生成式 AI 在将 C++转换为 Rust 方面变得足够好。

因此，存在机会，但我认为可以安全地说，至少在未来十年内，C++还将继续存在。
