# 11

# 处理异常

本章将开始我们的探索之旅，旨在扩展你的 C++编程知识库，使其超越面向对象的概念，目标是使你能够编写更健壮和可扩展的代码。我们将从探索 C++中的异常处理开始这一努力。在我们的代码中添加语言规定的错误处理方法将使我们能够编写更少错误和更可靠的程序。通过使用语言内建的正式异常处理机制，我们可以实现错误的统一处理，这导致代码更容易维护。

在本章中，我们将涵盖以下主要主题：

+   理解异常处理基础知识 – `try`，`throw`和`catch`

+   探索异常处理机制 – 尝试可能引发异常的代码，抛出（抛出），捕获，并使用多种变体处理异常

+   利用标准异常对象或创建自定义异常类来利用异常层次结构

到本章结束时，你将了解如何在 C++中利用异常处理。你将看到如何识别错误以引发异常，通过抛出异常将程序控制权转移到指定区域，然后通过捕获异常来处理错误，并希望修复当前的问题。

你还将学习如何利用 C++标准库中的标准异常，以及创建自定义异常对象。可以设计一个异常类层次结构，以添加强大的错误检测和处理能力。

让我们通过探索 C++内建的异常处理机制来扩展我们对 C++的理解。

# 技术要求

完整程序示例的在线代码可以在以下 GitHub URL 找到：[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/tree/main/Chapter11`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/tree/main/Chapter11)。每个完整的程序示例都可以在 GitHub 的相应章节标题（子目录）下找到，对应章节的文件名，后面跟着一个连字符，然后是当前章节的示例编号。例如，本章节的第一个完整程序可以在上述 GitHub 目录下的`Chapter11`子目录中找到，文件名为`Chp11-Ex1.cpp`。

本章节的 CiA 视频可以在以下链接查看：[`bit.ly/3QZi638`](https://bit.ly/3QZi638)。

# 理解异常处理

应用程序中可能会发生错误条件，这会阻止程序正确继续。这些错误条件可能包括超出应用程序限制的数据值，必要的输入文件或数据库变得不可用，堆内存耗尽，或任何其他可想象的问题。C++异常提供了一种统一、语言支持的程序异常处理方式。

在引入语言支持的异常处理机制之前，每个程序员都会以自己的方式处理错误，有时甚至不处理。未处理的程序错误和异常意味着在应用程序的某个地方，将发生意外的结果，并且应用程序通常会异常终止。这些潜在的结果当然是不希望的！

C++的**异常处理**提供了一种语言支持的机制来检测和纠正程序异常，以便应用程序可以继续运行，而不是突然终止。

让我们看看机制，从语言支持的`try`、`throw`和`catch`关键字开始，这些构成了 C++中的异常处理。 

## 利用 try、throw 和 catch 进行异常处理

**异常处理**检测程序异常，由程序员或类库定义，并将控制权传递到应用程序的另一部分，在那里可以处理特定的问题。只有在最后不得已的情况下，才需要退出应用程序。

让我们从查看支持异常处理的关键字开始。关键字如下：

+   `try`：允许程序员尝试可能引发异常的代码部分。

+   `throw`：一旦发现错误，`throw`会引发异常。这将导致跳转到关联的 try 块下面的 catch 块；`throw`将允许将参数返回到关联的 catch 块。抛出的参数可以是任何标准或用户定义的类型。

+   `catch`：指定一个代码块，用于寻找已抛出的异常，并尝试纠正情况。同一作用域中的每个 catch 块将处理不同类型的异常。

当使用异常处理时，回顾回溯的概念是有用的。当一系列函数被调用时，我们会在栈上建立适用于每个后续函数调用的状态信息（参数、局部变量和返回值空间），以及每个函数的返回地址。当抛出异常时，我们可能需要将栈回溯到函数调用序列（或 try 块）开始的点，同时重置栈指针。这个过程被称为**回溯**，允许程序返回到代码中的早期序列。回溯不仅适用于函数调用，还适用于嵌套块，包括嵌套的 try 块。

这里有一个简单的例子，用于说明基本的异常处理语法和用法。尽管为了节省空间没有显示代码的部分，但完整的例子可以在我们的 GitHub 上找到，如下所示：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter11/Chp11-Ex1.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter11/Chp11-Ex1.cpp)

```cpp
// Assume Student class is as seen before, but with one
// additional virtual mbr function. Assume usual headers.
void Student::Validate() // defined as virtual in class def
{                        // so derived classes may override
    // check constructed Student; see if standards are met
    // if not, throw an exception
    throw string("Does not meet prerequisites");
}
int main()
{
    Student s1("Sara", "Lin", 'B', "Dr.", 3.9,
               "C++", "23PSU");
    try      // Let's 'try' this block of code -- 
    {        // Validate() may raise an exception
        s1.Validate(); // does s1 meet admission standards?
    }
    catch (const string &err) 
    {
        cout << err << endl;
        // try to fix problem here…
        exit(1); // only if you can't fix error, 
    }            // exit as gracefully as possible
    cout << "Moving onward with remainder of code.";
    cout << endl;
    return 0;
}
```

在前面的代码片段中，我们可以看到 `try`、`throw` 和 `catch` 关键字在起作用。首先，让我们注意到 `Student::Validate()` 成员函数。想象一下，在这个虚函数中，我们验证一个 `Student` 是否符合入学标准。如果是这样，函数将正常结束。如果不是，将抛出一个异常。在这个例子中，抛出了一个简单的 `string`，封装了消息 `"Does not meet prerequisites"`。

在我们的 `main()` 函数中，我们首先实例化一个 `Student`，即 `s1`。然后，我们将对 `s1.Validate()` 的调用嵌套在一个 try 块中。我们实际上是在说我们想要 *尝试* 这段代码。如果 `Student::Validate()` 如预期那样工作，没有错误，我们的程序将完成 try 块，跳过 try 块下面的捕获块，并继续执行任何捕获块下面的代码。

然而，如果 `Student::Validate()` 抛出异常，我们将跳过 try 块中剩余的任何代码，并寻找一个随后定义的匹配的捕获块中与 `const string &` 类型匹配的异常。在这里，在匹配的捕获块中，我们的目标是尽可能纠正错误。如果我们成功，我们的程序将继续执行捕获器下面的代码。如果我们不成功，我们的任务是优雅地结束程序。

让我们看看上述程序的输出：

```cpp
Student does not meet prerequisites 
```

接下来，让我们用以下逻辑总结异常处理的总体流程：

+   当程序完成 try 块而没有遇到任何抛出的异常时，代码序列将继续执行 catch 块后面的语句。多个具有不同参数类型的 catch 块可以跟在 try 块后面。

+   当抛出异常时，程序必须回溯并返回到包含原始函数调用的 try 块。程序可能需要回溯多个函数。当回溯发生时，堆栈上遇到的对象将被弹出，因此将被销毁。

+   一旦程序（抛出异常）回溯到执行 try 块的函数，程序将继续执行与抛出的异常类型匹配的签名匹配的 catch 块（跟随 try 块）。

+   类型转换（除了通过公共继承相关联的对象的上转型）不会执行以匹配潜在的捕获块。然而，可以捕获任何类型的异常的省略号（`…`）捕获块可以用作最通用的捕获块。

+   如果不存在匹配的捕获块，程序将调用 C++ 标准库中的 `terminate()`。请注意，`terminate()` 将调用 `abort()`；然而，程序员可以通过 `set_terminate()` 函数注册另一个函数来代替 `terminate()`。

现在，让我们看看如何使用 `set_terminate()` 注册一个函数。尽管我们在这里只展示了代码的关键部分，但完整的程序可以在我们的 GitHub 上找到：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter11/Chp11-Ex2.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter11/Chp11-Ex2.cpp)

```cpp
void AppSpecificTerminate()
{   // first, do what's necessary to end program gracefully
    cout << "Uncaught exception. Program terminating";
    cout << endl;
    exit(1);
}
int main()
{   
    set_terminate(AppSpecificTerminate);  // register fn.
    return 0;
}
```

在之前的代码片段中，我们定义了自己的`AppSpecificTerminate()`函数。这是我们希望`terminate()`函数调用的函数，而不是其默认行为调用`abort()`。也许我们会使用`AppSpecificTerminate()`来更优雅地结束我们的应用程序，保存关键数据结构或数据库值。当然，我们也会自己`exit()`（或`abort()`）。

在`main()`中，我们只是调用`set_terminate(AppSpecificTerminate)`来将我们的终止函数注册到`set_terminate()`。现在，当`abort()`会被调用时，我们的函数将被调用。

有趣的是，`set_terminate()`返回一个指向之前安装的`terminate_handler`的函数指针（在其第一次调用时将是一个指向`abort()`的指针）。如果我们选择保存这个值，我们可以使用它来恢复之前注册的终止处理器。请注意，我们没有选择在这个例子中保存这个函数指针。

下面是使用上述代码未捕获异常的输出示例：

```cpp
Uncaught exception. Program terminating
```

请记住，`terminate()`、`abort()`和`set_terminate()`等函数来自标准库。尽管我们可以使用作用域解析运算符在它们的名字前加上库名，例如`std::terminate()`，但这不是必要的。

注意

异常处理并不是要取代简单的程序员错误检查；异常处理有更大的开销。异常处理应该保留用于以统一的方式和在共同的位置处理更严重的程序错误。

现在我们已经看到了异常处理的基本机制，让我们看看稍微复杂一些的异常处理示例。

## 探索具有典型变化的异常处理机制

异常处理可以比之前展示的基本机制更复杂和灵活。让我们看看异常处理基本原理的各种组合和变化，因为每种可能适用于不同的编程场景。

### 将异常传递给外部处理器

捕获的异常可以被传递给外部处理器进行处理。或者，异常可以被部分处理，然后抛出到外部作用域进行进一步处理。

让我们基于之前的例子来演示这个原则。完整的程序可以在以下 GitHub 目录中查看：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter11/Chp11-Ex3.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter11/Chp11-Ex3.cpp)

```cpp
// Assume Student class is as seen before, but with
// two additional member fns. Assume usual header files.
void Student::Validate() // defined as virtual in class def
{                        // so derived classes may override
    // check constructed student; see if standards are met
    // if not, throw an exception
    throw string("Does not meet prerequisites");
}
bool Student::TakePrerequisites()  
{
    // Assume this function can correct the issue at hand
    // if not, it returns false
    return false;
}
int main()
{
    Student s1("Alex", "Ren", 'Z', "Dr.", 3.9, 
               "C++", "89CU");
    try    // illustrates a nested try block 
    {   
        // Assume another important task occurred in this
        // scope, which may have also raised an exception
        try
        {   
            s1.Validate();  // may raise an exception
        }
        catch (const string &err)
        {
            cout << err << endl;
            // try to correct (or partially handle) error.
            // If you cannot, pass exception to outer scope
            if (!s1.TakePrerequisites())
                throw;    // re-throw the exception
        }
    }
    catch (const string &err) // outer scope catcher 
    {                         // (e.g. handler)
        cout << err << endl;
        // try to fix problem here…
        exit(1); // only if you can't fix, exit gracefully
    } 
    cout << "Moving onward with remainder of code. ";
    cout << endl;
    return 0;
}
```

在上述代码中，让我们假设我们已经包含了我们常用的头文件，并定义了`Student`类的常用类定义。现在我们将通过添加`Student::Validate()`方法（虚拟的，以便它可以被重写）和`Student::TakePrerequisites()`方法（非虚拟的，后代应该直接使用它）来增强`Student`类。

注意到我们的`Student::Validate()`方法抛出了一个异常，它只是一个包含指示当前问题的消息的字符串字面量。我们可以想象`Student::TakePrerequisites()`方法的完整实现验证`Student`是否满足适当的先决条件，并相应地返回`true`或`false`布尔值。

在我们的`main()`函数中，我们现在注意到一组嵌套的`try`块。这里的目的是说明一个可能调用方法（例如`s1.Validate()`）的内部`try`块，该方法可能会抛出异常。请注意，与内部`try`块相同级别的处理程序捕获了这个异常。理想情况下，异常应该在与它起源的`try`块相同的级别上得到处理，所以让我们假设在这个作用域中的捕获器试图这样做。例如，我们的最内层捕获块可能试图纠正错误，并通过调用`s1.TakePrerequisites()`来测试是否已进行了纠正。

但也许这个捕获器只能部分处理异常。也许存在这样的知识，即外层处理程序知道如何进行剩余的修正。在这种情况下，将这个异常重新抛出到外层（嵌套）级别是可以接受的。我们最内层的捕获块中的简单`throw;`语句正是这样做的。请注意，外层确实有一个捕获器。如果抛出的异常在类型上匹配，那么现在外层级别将有机会进一步处理异常，并希望纠正问题，以便应用程序可以继续运行。只有当外层捕获块无法纠正错误时，应用程序才应该退出。在我们的例子中，每个捕获器都会打印出表示错误信息的字符串；因此，这个消息在输出中出现了两次。

让我们看看上述程序的输出：

```cpp
Student does not meet prerequisites
Student does not meet prerequisites
```

现在我们已经看到了如何使用嵌套的`try`和`catch`块，让我们继续前进，看看如何将各种抛出类型和多种捕获块结合起来使用。

### 添加一系列处理器

有时，从内部作用域可能会抛出各种异常，这就需要为各种数据类型创建处理器。异常处理器（即捕获块）可以接收任何数据类型的异常。我们可以通过使用基类类型的捕获块来最小化我们引入的捕获器的数量；我们知道派生类对象（通过公共继承相关）总是可以被向上转换为它们的基类类型。我们还可以在捕获块中使用省略号（`…`）来允许我们捕获之前未指定的任何内容。

让我们基于最初的示例来构建一个示例，以展示各种处理器的实际应用。虽然程序示例被简化了，但完整的程序示例可以在我们的 GitHub 上找到，如下所示：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter11/Chp11-Ex4.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter11/Chp11-Ex4.cpp)

```cpp
// Assume Student class is as seen before, but with one
// additional virtual member function, Graduate(). Assume 
// a simple Course class exists. All headers are as usual.
void Student::Graduate()
{   // Assume the below if statements are fully implemented 
    if (gpa < 2.0) // if gpa doesn't meet requirements
        throw gpa;
    // if Student is short credits, throw number missing
        throw numCreditsMissing;  // assume this is an int
    // or if Student is missing a Course, construct, then
    // throw the missing Course as a referenceable object
    // Assume appropriate Course constructor exists
        throw Course("Intro. To Programming", 1234); 
    // or if another issue, throw a diagnostic message
        throw string("Does not meet requirements"); 
}
int main()
{
    Student s1("Ling", "Mau", 'I', "Ms.", 3.1, 
               "C++", "55UD");
    try  
    {  
        s1.Graduate();
    }
    catch (float err)
    {
        cout << "Too low gpa: " << err << endl;
        exit(1); // only if you can't fix, exit gracefully
    } 
    catch (int err)
    {
        cout << "Missing " << err << " credits" << endl;
        exit(2);
    }
    catch (const Course &err)
    {
        cout << "Need to take: " << err.GetTitle() << endl;
        cout << "Course #: " << err.GetCourseNum() << endl; 
        // Ideally, correct the error, and continue program 
        exit(3); // Otherwise, exit, gracefully if possible
    }             
    catch (const string &err)
    {
        cout << err << endl;
        exit(4); 
    }
    catch (...)
    {
        cout << "Exiting" << endl;
        exit(5);
    }
    cout << "Moving onward with remainder of code.";
    cout << endl;
    return 0;
}
```

在上述代码段中，我们首先检查 `Student::Graduate()` 成员函数。在这里，我们可以想象这个方法会运行许多毕业要求，因此可能会抛出各种不同类型的异常。例如，如果 `Student` 实例的 `gpa` 太低，会抛出一个浮点数作为异常，表示学生的 `gpa` 很差。如果 `Student` 的学分太少，会抛出一个整数，表示学生还需要获得多少学分才能获得学位。

`Student::Graduate()` 可能引发的最有趣的潜在错误是，如果学生的毕业要求中缺少一个必需的 `Course`。在这种情况下，`Student::Graduate()` 会实例化一个新的 `Course` 对象，通过构造函数填充 `Course` 名称和编号。这个匿名对象随后会从 `Student::Graduate()` 中抛出，就像在这个方法中可以交替抛出的匿名 `string` 对象一样。然后处理器可以通过引用捕获 `Course`（或 `string`）对象。

在 `main()` 函数中，我们仅仅将 `Student::Graduate()` 的调用封装在一个 try 块中，因为这个语句可能会抛出异常。在 try 块之后跟随一系列的捕获器 – 每个捕获器对应可能抛出的对象类型。在这个序列中的最后一个捕获块使用了省略号（`…`），表示这个捕获器将处理 `Student::Graduate()` 抛出的任何其他类型的异常，这些异常没有被其他捕获器捕获。

实际参与捕获的捕获块是使用 `const Course &err` 捕获 `Course` 的那个。由于有 `const` 关键字，我们在处理程序中不能修改 `Course`，因此我们只能对此对象应用 `const` 成员函数。

注意，尽管每个早期的捕获器只是打印出错误然后退出，理想情况下，捕获器会尝试纠正错误，这样应用程序就不需要终止，允许 catch 块下面的代码继续执行。

让我们看看上述程序的输出：

```cpp
Need to take: Intro. to Programming
Course #: 1234
```

现在我们已经看到了各种抛出类型和捕获块，让我们继续了解我们应该在单个 try 块中一起组合哪些内容。

### 在 try 块中将相关项分组

重要的是要记住，当`try`块中的一行代码遇到异常时，`try`块剩余的部分将被忽略。相反，程序将继续执行匹配的 catcher（如果不存在合适的 catcher，则调用`terminate()`）。然后，如果错误被修复，catcher 之后的代码开始执行。请注意，我们永远不会返回以完成初始`try`块的剩余部分。这种行为的意义是，你应该只将属于`try`块中的相关元素组合在一起。也就是说，如果一个项引发了异常，那么完成该分组中的其他项就不再重要了。

请记住，catcher 的目标是在可能的情况下纠正错误。这意味着程序可以在适当的 catch 块之后继续执行。你可能会问：现在跳过关联的`try`块中的某个项是否可以接受？如果答案是否定的，那么请重写你的代码。例如，你可能会想要在`try`-`catch`分组周围添加一个循环，这样如果 catcher 纠正了错误，整个尝试将从最初的`try`块重新开始。

或者，创建更小的连续`try`-`catch`分组。也就是说，仅在它自己的`try`块中（后面跟着相应的 catcher）尝试一个重要的任务。然后，在它自己的`try`块中尝试下一个任务，并带有其关联的 catcher，依此类推。

接下来，让我们看看如何在函数原型中包含可能抛出的异常类型。

### 检查函数原型中的异常规范

我们可以通过扩展函数的签名来指定 C++函数可能抛出的异常类型，包括可能抛出的对象类型。然而，由于一个函数可能抛出多种类型的异常（或者根本不抛出），检查实际抛出的类型必须在运行时完成。因此，这些增强的指定符在函数原型中也被称为`noexcept`指定符，我们将在稍后看到。动态异常的使用也存在于现有的代码库和库中，所以让我们简要地考察其用法。

让我们通过一个示例来看看在函数的扩展签名中使用异常类型：

```cpp
void Student::Graduate() throw(float, int, 
                               Course &, string)
{
   // this method might throw any type included in 
   // its extended signature
}
void Student::Enroll() throw()
{
   // this method might throw any type of exception
}
```

在上述代码片段中，我们可以看到`Student`类的两个成员函数。`Student::Graduate()`函数在其参数列表之后包含了`throw`关键字，并在其扩展签名中包含了可能从这个函数抛出的对象类型。请注意，`Student::Enroll()`方法在其扩展签名中仅在`throw()`之后有一个空列表。这意味着`Student::Enroll()`可能会抛出任何类型的异常。

在这两种情况下，通过在签名中添加带有可选数据类型的`throw()`关键字，我们为用户提供了宣布可能抛出的对象类型的方式。然后我们要求程序员在适当的位置包含对这种方法的方法调用，并在其后添加相应的 catcher。

我们将看到，尽管扩展签名的想法看起来非常有帮助，但在实践中却存在不利的因素。因此，动态异常规范已被**弃用**。由于你可能会在现有的代码中看到这些规范的使用，包括标准库原型（例如异常处理），因此编译器仍然支持这个弃用的特性，你需要了解它们的用法。

尽管动态异常（如之前所述的扩展函数签名）已被弃用，但为了达到类似的目的，语言中已添加了一个指定符，即`noexcept`关键字。

这个指定符可以按照以下方式添加到扩展签名之后：

```cpp
void Student::Graduate() noexcept   // will not throw() 
{          // same as  noexcept(true) in extended signature
}          // same as deprecated throw() in ext. signature
void Student::Enroll() noexcept(false)  // may throw()
{                                       // an exception
}                                     
```

尽管如此，让我们通过查看当我们的应用程序抛出不属于函数扩展签名的异常时会发生什么来调查与动态异常相关的不利问题。

### 处理意外的动态异常类型

如果抛出的异常类型与扩展函数原型中指定的类型不同，C++标准库中的`unexpected()`将被调用。你可以使用`unexpected()`注册你自己的函数，就像我们在本章前面注册`set_terminate()`时做的那样。

你可以让你的`AppSpecificUnexpected()`函数重新抛出原始函数应该抛出的异常类型；然而，如果这种情况没有发生，`terminate()`将被调用。此外，如果不存在可能的匹配 catcher 来处理从原始函数正确抛出（或由`AppSpecificUnexpected()`重新抛出）的内容，那么`terminate()`将被调用。

让我们看看如何使用`set_unexpected()`与我们的函数结合使用：

```cpp
void AppSpecificUnexpected()
{
    cout << "An unexpected type was thrown" << endl;
    // optionally re-throw the correct type, or
    // terminate() will be called.
}
int main()
{
   set_unexpected(AppSpecificUnexpected)
}
```

如前所述的代码片段所示，使用`set_unexpected()`将我们自己的函数注册起来非常简单。

从历史上看，在函数的扩展签名中使用异常规范的一个激励因素是为了提供文档效果。也就是说，你可以通过检查函数的签名来看到函数可能会抛出的异常。然后你可以计划在函数调用周围放置 try 块，并提供适当的 catcher 来处理任何潜在的情况。

然而，关于动态异常，值得注意的是，编译器不会检查函数体中实际抛出的异常类型是否与函数扩展签名中指定的类型匹配。确保它们同步是程序员的职责。因此，这个弃用的特性可能会引起错误，并且总体上不如其原始意图有用。

尽管初衷良好，但动态异常目前除了在大量的库代码（如标准 C++库）中之外，并未被使用。由于你不可避免地会使用这些库，因此了解这些过时的特性很重要。

重要提示

动态异常指定（即在方法扩展签名中指定异常类型的能力）在 C++中已被*弃用*。这是因为编译器无法验证其使用，这必须延迟到运行时。尽管它们的使用仍然得到支持（许多库有这样的指定），但现在已被弃用。

既然我们已经看到了各种异常处理检测、抛出、捕获以及（希望）纠正方案，让我们来看看如何创建一个异常类层次结构，以增强我们的错误处理能力。

# 利用异常层次结构

创建一个封装与程序错误相关的详细信息的类似乎是一项有用的任务。实际上，C++标准库已经创建了一个这样的通用类，即`exception`，为构建整个有用的异常类层次结构提供基础。

让我们来看看`exception`类及其标准库派生类，然后探讨如何通过我们自己的类扩展`exception`。

## 使用标准异常对象

`<exception>`头文件。`exception`类包含以下签名的虚拟函数：`virtual const char *what() const noexcept`和`virtual const char *what() const throw()`。这些签名表明派生类应该重新定义`what()`以返回一个描述当前错误的`const char *`。`what()`后面的`const`关键字表明这些是`const`成员函数；它们不会改变派生类的任何成员。第一个原型中`noexcept`的使用表明`what()`是非抛出的。第二个原型扩展签名中的`throw()`表明此函数可能抛出任何类型。第二个签名中`throw()`的使用是一个过时的做法，不应在新代码中使用。

`std::exception`类是多种预定义的 C++异常类的基类，包括`bad_alloc`、`bad_cast`、`bad_exception`、`bad_function_call`、`bad_typeid`、`bad_weak_ptr`、`logic_error`、`runtime_error`以及嵌套类`ios_base::failure`。其中许多派生类本身也有派生，为预定义的异常层次结构添加了额外的标准异常。

如果一个函数抛出上述任何异常，这些异常可以通过捕获基类类型`exception`或捕获单个派生类类型来捕获。根据您的处理程序将采取的行动，您可以决定是否希望将其作为其泛化基类类型或其特定类型捕获。

正如标准库已经基于`exception`类建立了一个类层次结构一样，您也可以这样做。接下来，让我们看看我们如何做到这一点！

## 创建自定义异常类

作为程序员，你可能会决定建立自己的专用异常类型是有益的。每种类型都可以将有关应用程序中发生什么错误的有用信息打包到对象中。此外，你还可以将有关如何纠正当前错误的线索打包到（将被抛出的）对象中。只需从标准库 `exception` 类派生你的类即可。

让我们通过检查下一个示例的关键部分来看看这有多容易做到，该示例作为一个完整的程序可以在我们的 GitHub 上找到：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter11/Chp11-Ex5.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter11/Chp11-Ex5.cpp)

```cpp
#include <iostream>
#include <exception>
// See online code for many using std:: inclusions
class StudentException: public exception
{
private:
    int errCode = 0;  // in-class init, will be over-
    // written with bonified value after successful 
    // alternate constructor completion
    string details;
public:
    StudentException(const string &det, int num):
                     errCode(num), details(det) { } 
    // Base class destructor (exception class) is virtual. 
    // Override at this level if there's work to do. 
    // We can omit the default destructor prototype.
    // ~StudentException() override = default;
    const char *what() const noexcept override
    {   // overridden function from exception class
        return "Student Exception";
    } 
    int GetCode() const { return errCode; }
    const string &GetDetails() const { return details; }
};
// Assume Student class is as we've seen before, but with
// one additional virtual member function, Graduate() 
void Student::Graduate() // fn. may throw StudentException
{
   // if something goes wrong, construct a 
   // StudentException, packing it with relevant data, 
   // and then throw it as a referenceable object
   throw StudentException("Missing Credits", 4);
}
int main()
{
    Student s1("Alexandra", "Doone", 'G', "Miss", 3.95, 
               "C++", "231GWU");
    try
    {
        s1.Graduate();
    }
    catch (const StudentException &e)  // catch exc. by ref
    { 
        cout << e.what() << endl;
        cout << e.GetCode() << " " << e.GetDetails();
        cout << endl;
        // Grab useful info from e and try to fix problem
        // so that the program can continue.
        exit(1);  // only exit if we can't fix the problem!
    }
    return 0;
}
```

让我们花几分钟时间检查之前的代码段。首先，请注意我们定义了自己的异常类，`StudentException`。它是从 C++ 标准库 `exception` 类派生的。

`StudentException` 类包含数据成员来存储错误代码以及使用数据成员 `errCode` 和 `details` 分别描述错误条件的字母数字细节。我们有两个简单的访问函数，`StudentException::GetCode()` 和 `StudentException::GetDetails()`，以便轻松检索这些值。由于这些方法不修改对象，它们是 `const` 成员函数。

我们注意到，`StudentException` 构造函数初始化了两个数据成员——一个通过成员初始化列表，一个在构造函数体中。我们还重写了 `StudentException` 类中的 `virtual const char *what() const noexcept` 方法（由 `exception` 类引入），以返回字符串 `"Student Exception"`。

接下来，让我们检查我们的 `Student::Graduate()` 方法。此方法可能会抛出 `StudentException` 异常。如果必须抛出异常，我们将实例化一个异常，使用诊断数据构造它，然后从该函数中 `throw` `StudentException` 异常。请注意，在此方法中抛出的对象没有局部标识符——因为没有必要，因为任何这样的局部变量名在抛出后很快就会从栈上弹出。

在我们的 `main()` 函数中，我们将对 `s1.Graduate()` 的调用包裹在一个 try 块中，后面跟着一个 catch 块，该块接受一个对 `StudentException` 的引用（`&`），我们将其视为 `const`。在这里，我们首先调用我们重写的 `what()` 方法，然后从异常 `e` 中打印出诊断细节。理想情况下，我们会使用这些信息来尝试纠正当前错误，只有在真正必要时才退出应用程序。

让我们看看上述程序的输出：

```cpp
Student Exception
4 Missing Credits
```

虽然从标准 `exception` 类派生一个类来创建自定义异常类是最常见的方式，但你可能也希望利用不同的技术，即嵌入异常类技术。

### 创建嵌套异常类

作为一种替代实现，异常处理可以通过在特定外部类的公共访问区域添加嵌套类定义来嵌入到类中。内部类将代表定制的异常类。

可以创建嵌套的用户定义类型对象并将其抛出，以便捕获器可以捕获这些类型。这些嵌套类被构建在外部类的公共访问区域，使得它们对于派生类的使用和特殊化很容易访问。一般来说，嵌入到外部类中的异常类必须是公共的，这样抛出的嵌套类型实例才能在外部类的范围之外（即主要的外部实例存在的范围）被捕获和处理。

让我们通过检查代码的关键部分来查看这个异常类的替代实现，完整的程序可以在我们的 GitHub 上找到：

[`github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter11/Chp11-Ex6.cpp`](https://github.com/PacktPublishing/Deciphering-Object-Oriented-Programming-with-CPP/blob/main/Chapter11/Chp11-Ex6.cpp)

```cpp
// Assume Student class is as before, but with the addition 
// of a nested exception class. All headers are as usual.
class Student: public Person
{
private:  // assume usual data members
public:   // usual constructors, destructor, and methods
    virtual void Graduate();
    class StudentException   // nested exception class
    {
    private:
        int number = 0;  // will be over-written after 
        // successful alternate constructor 
        // note: there is no default constructor
    public:
        StudentException(int num): number(num) { }
        // Remember, it is unnecessary to proto. default ~
        // ~StudentException() = default;
        int GetNum() const { return number; }
    };
};
void Student::Graduate()
{   // assume we determine an error and wish to throw
    // the nested exception type
    throw StudentException(5);
}
int main()
{
    Student s1("Ling", "Mau", 'I', "Ms.", 3.1, 
               "C++", "55UD");
    try
    {
        s1.Graduate();
    }
    // following is one of many catch blocks in online code
    catch (const Student::StudentException &err)
    {
        cout << "Error: " << err.GetNum() << endl;
        // If you correct error, continue the program
        exit(5);  // Otherwise, exit application 
    }
    cout << "Moving onward with remainder of code.";
    cout << endl;
    return 0;
}
```

在之前的代码片段中，我们将 `Student` 类扩展，包括一个名为 `StudentException` 的私有嵌套类。尽管显示的类过于简化，但嵌套类理想情况下应该定义一种方法来记录所讨论的错误，以及收集任何有用的诊断信息。

在我们的 `main()` 函数中，我们实例化了一个 `Student` 对象，即 `s1`。然后在 try 块中调用 `s1.Graduate();`。我们的 `Student::Graduate()` 方法可能检查 `Student` 是否满足毕业要求，如果没有，则抛出嵌套类类型的异常，即 `Student::StudentException`（根据需要实例化）。

注意，我们的相应 catch 块使用作用域解析来指定 `err`（引用的对象，即 `const Student::StudentException &err`）的内部类类型。尽管我们理想情况下希望在处理程序中纠正程序错误，如果我们不能这样做，我们只需打印一条消息并调用 `exit()`。

让我们看看上述程序的输出：

```cpp
Error: 5
```

理解如何创建我们自己的异常类（无论是作为嵌套类还是从 `std::exception` 派生）是有用的。我们可能还希望创建一个特定应用级别的异常层次结构。让我们继续前进，看看如何做到这一点。

## 创建用户定义异常类型的层次结构

应用程序可能希望定义一系列支持异常处理的类，以引发特定的错误，并希望提供一种收集错误诊断信息的方法，以便在代码的适当部分处理错误。

你可能希望创建一个从 C++ 标准库 `exception` 派生的子层次结构，以包含你自己的异常类。务必使用公有继承。当使用这些类时，你将实例化一个你想要的异常类型的对象（填充有有价值的诊断信息），然后抛出该对象。

此外，如果你创建了一个异常类型的层次结构，你的捕获器可以捕获特定的派生类类型或更一般的基类类型。选择权在你，取决于你将如何计划处理异常。然而，请记住，如果你同时有一个基类和派生类类型的捕获器，请将派生类类型放在前面——否则，你的抛出对象将首先匹配到基类类型的捕获器，而不会意识到有一个更合适的派生类匹配可用。

我们现在已经看到了 C++ 标准库异常类的层次结构，以及如何创建和使用你自己的异常类。现在，在我们继续前进到下一章之前，让我们简要回顾一下本章学到的异常特性。

# 摘要

在本章中，我们开始扩展我们的 C++ 编程库，不仅包括面向对象语言特性，还包括将使我们能够编写更健壮程序的特性。用户代码不可避免地具有错误倾向；使用语言支持的异常处理可以帮助我们实现更少错误和更可靠的代码。

我们已经看到了如何使用 `try`、`throw` 和 `catch` 核心异常处理特性。我们已经看到了这些关键字的各种用法——向外部处理程序抛出异常，使用各种类型的处理程序，例如，在单个 try 块内选择性地将程序元素分组在一起。我们已经看到了如何使用 `set_terminate()` 和 `set_unexpected()` 注册我们自己的函数。我们已经看到了如何利用现有的 C++ 标准库 `exception` 层次结构。我们还探讨了定义我们自己的异常类以扩展这个层次结构。

通过探索异常处理机制，我们已经增加了我们的 C++ 技能的关键特性。我们现在准备向前推进到 *第十二章*，*友元和运算符重载*，这样我们就可以继续使用有用的语言特性来扩展我们的 C++ 编程库，使我们成为更好的程序员。让我们继续前进！

# 问题

1.  将异常处理添加到你的上一章 *第十章*，*实现关联、聚合和组合* 的 `Student` / `University` 练习中，如下所示：

    1.  如果一个 `Student` 尝试注册超过 `MAX` 定义的数量允许的课程，则抛出 `TooFullSchedule` 异常。这个类可能从标准库 `exception` 类派生。

    1.  如果一个`Student`试图报名一个已经满员的`Course`，`Course::AddStudent(Student *)`方法应该抛出一个`CourseFull`异常。这个类可以继承自标准库的`exception`类。

    1.  在`Student` / `University`应用程序中还有许多其他区域可以利用异常处理。决定哪些区域应该使用简单的错误检查，哪些值得使用异常处理。
