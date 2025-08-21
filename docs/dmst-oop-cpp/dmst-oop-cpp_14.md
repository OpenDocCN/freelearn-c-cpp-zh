# 第十一章：处理异常

本章将开始我们的探索，扩展你的 C++编程技能，超越面向对象编程的概念，目标是让你能够编写更健壮、更可扩展的代码。我们将通过探索 C++中的异常处理来开始这个努力。在我们的代码中添加语言规定的方法来处理错误，将使我们能够实现更少的错误和更可靠的程序。通过使用语言内置的正式异常处理机制，我们可以实现对错误的统一处理，从而实现更易于维护的代码。

在本章中，我们将涵盖以下主要主题：

+   理解异常处理的基础知识——`try`、`throw`和`catch`

+   探索异常处理机制——尝试可能引发异常的代码，引发（抛出）、捕获和处理异常，使用多种变体

+   利用标准异常对象或创建自定义异常类的异常层次结构

通过本章结束时，你将了解如何在 C++中利用异常处理。你将看到如何识别错误以引发异常，通过抛出异常将程序控制转移到指定区域，然后通过捕获异常来处理错误，并希望修复手头的问题。

你还将学习如何利用 C++标准库中的标准异常，以及如何创建自定义异常对象。可以设计一组异常类的层次结构，以增加健壮的错误检测和处理能力。

通过探索内置的语言异常处理机制，扩展我们对 C++的理解。

# 技术要求

完整程序示例的在线代码可以在以下 GitHub URL 找到：[`github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter11`](https://github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter11)。每个完整程序示例都可以在 GitHub 存储库中找到，位于相应章节标题（子目录）下的文件中，文件名与所在章节编号相对应，后跟该章节中的示例编号。例如，本章的第一个完整程序可以在名为`Chp11-Ex1.cpp`的文件中的子目录`Chapter11`中找到，位于上述 GitHub 目录下。

本章的 CiA 视频可在以下链接观看：[`bit.ly/3r8LHd5`](https://bit.ly/3r8LHd5)。

# 理解异常处理

应用程序中可能会出现错误条件，这些错误条件可能会阻止程序正确地继续运行。这些错误条件可能包括超出应用程序限制的数据值、必要的输入文件或数据库不可用、堆内存耗尽，或者任何其他可能的问题。C++异常提供了一种统一的、语言支持的方式来处理程序异常。

在引入语言支持的异常处理机制之前，每个程序员都会以自己的方式处理错误，有时甚至根本不处理。程序错误和未处理的异常意味着在应用程序的其他地方，将会发生意外的结果，应用程序往往会异常终止。这些潜在的结果肯定是不可取的！

C++异常处理提供了一种语言支持的机制，用于检测和纠正程序异常，使应用程序能够继续运行，而不是突然结束。

让我们从语言支持的关键字`try`、`throw`和`catch`开始，来看一下这些机制，它们构成了 C++中的异常处理。

## 利用 try、throw 和 catch 进行异常处理

**异常处理**检测到程序异常，由程序员或类库定义，并将控制传递到应用程序的另一个部分，该部分可能处理特定的问题。只有作为最后的手段，才需要退出应用程序。

让我们首先看一下支持异常处理的关键字。这些关键字是：

+   `try`：允许程序员*尝试*可能引发异常的代码部分。

+   `throw`：一旦发现错误，`throw`会引发异常。这将导致跳转到与关联 try 块下面的 catch 块。Throw 将允许将参数返回到关联的 catch 块。抛出的参数可以是任何标准或用户定义的类型。

+   `catch`：指定一个代码块，旨在寻找已抛出的异常，以尝试纠正情况。同一作用域中的每个 catch 块将处理不同类型的异常。

在使用异常处理时，回溯的概念是有用的。当调用一系列函数时，我们在堆栈上建立起与每个连续函数调用相关的状态信息（参数、局部变量和返回值空间），以及每个函数的返回地址。当抛出异常时，我们可能需要解开堆栈，直到这个函数调用序列（或 try 块）开始的原点，同时重置堆栈指针。这个过程被称为**回溯**，它允许程序返回到代码中的较早序列。回溯不仅适用于函数调用，还适用于包括嵌套 try 块在内的嵌套块。

这里有一个简单的例子，用来说明基本的异常处理语法和用法。尽管代码的部分没有显示出来以节省空间，但完整的示例可以在我们的 GitHub 上找到，如下所示：

[`github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter11/Chp11-Ex1.cpp`](https://github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter11/Chp11-Ex1.cpp)

```cpp
// Assume Student class is as we've seen before, but with one
// additional virtual member function. Assume usual headers.
void Student::Validate()  // defined as virtual in class def
{                         // so derived classes may override
    // check constructed Student to see if standards are met
    // if not, throw an exception
    throw "Does not meet prerequisites";
}
int main()
{
    Student s1("Sara", "Lin", 'B', "Dr.", 3.9,"C++", "23PSU");
    try    // Let's 'try' this block of code -- 
    {      // Validate() may raise an exception
        s1.Validate();  // does s1 meet admission standards?
    }
    catch (const char *err)
    {
        cout << err << endl;
        // try to fix problem here…
        exit(1); // only if you can't fix, exit gracefully
    } 
    cout << "Moving onward with remainder of code." << endl;
    return 0;
}
```

在上面的代码片段中，我们可以看到关键字`try`、`throw`和`catch`的作用。首先，让我们注意`Student::Validate()`成员函数。想象一下，在这个虚方法中，我们验证一个`Student`是否符合入学标准。如果是，函数会正常结束。如果不是，就会抛出异常。在这个例子中，抛出一个简单的`const char *`，其中包含消息"`Does not meet prerequisites`"。

在我们的`main()`函数中，我们首先实例化一个`Student`，即`s1`。然后，我们将对`s1.Validate()`的调用嵌套在一个 try 块中。我们实际上是在说，我们想*尝试*这个代码块。如果`Student::Validate()`按预期工作，没有错误，我们的程序将完成 try 块，跳过 try 块下面的 catch 块，并继续执行 catch 块下面的代码。

然而，如果`Student::Validate()`抛出异常，我们将跳过 try 块中的任何剩余代码，并在随后定义的 catch 块中寻找与`const char *`类型匹配的异常。在匹配的 catch 块中，我们的目标是尽可能地纠正错误。如果成功，我们的程序将继续执行 catch 块下面的代码。如果不成功，我们的工作就是优雅地结束程序。

让我们看一下上述程序的输出：

```cpp
Student does not meet prerequisites 
```

接下来，让我们总结一下异常处理的整体流程，具体如下：

+   当程序完成 try 块而没有遇到任何抛出的异常时，代码序列将继续执行 catch 块后面的语句。多个 catch 块（带有不同的参数类型）可以跟在 try 块后面。

+   当抛出异常时，程序必须回溯并返回到包含原始函数调用的 try 块。程序可能需要回溯多个函数。当回溯发生时，遇到的对象将从堆栈中弹出，因此被销毁。

+   一旦程序（引发异常）回溯到执行 try 块的函数，程序将继续执行与抛出的异常类型匹配的 catch 块（在 try 块之后）。

+   类型转换（除了通过公共继承相关的向上转型对象）不会被执行以匹配潜在的 catch 块。然而，带有省略号（`…`）的 catch 块可以作为最一般类型的 catch 块使用，并且可以捕获任何类型的异常。

+   如果不存在匹配的`catch`块，程序将调用 C++标准库中的`terminate()`。请注意，`terminate()`将调用`abort()`，但程序员可以通过`set_terminate()`函数注册另一个函数供`terminate()`调用。

现在，让我们看看如何使用`set_terminate()`注册一个函数。虽然我们这里只展示了代码的关键部分，完整的程序可以在我们的 GitHub 上找到：

[`github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter11/Chp11-Ex2.cpp`](https://github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter11/Chp11-Ex2.cpp)

```cpp
void AppSpecificTerminate()
{   // first, do what is necessary to end program gracefully
    cout << "Uncaught exception. Program terminating" << endl;
    exit(1);
}
int main()
{   
    set_terminate(AppSpecificTerminate);  // register fn.
    return 0;
}
```

在前面的代码片段中，我们定义了自己的`AppSpecificTerminate()`函数。这是我们希望`terminate()`函数调用的函数，而不是调用`abort()`的默认行为。也许我们使用`AppSpecificTerminate()`来更优雅地结束我们的应用程序，保存关键数据结构或数据库值。当然，我们也会自己`exit()`（或`abort()`）。

在`main()`中，我们只需调用`set_terminate(AppSpecificTerminate)`来注册我们的`terminate`函数到`set_terminate()`。现在，当否则会调用`abort()`时，我们的函数将被调用。

有趣的是，`set_terminate()`返回一个指向先前安装的`terminate_handler`的函数指针（在第一次调用时将是指向`abort()`的指针）。如果我们选择保存这个值，我们可以使用它来恢复先前注册的终止处理程序。请注意，在这个示例中，我们选择不保存这个函数指针。

以下是使用上述代码未捕获异常的输出：

```cpp
Uncaught exception. Program terminating
```

请记住，诸如`terminate()`、`abort()`和`set_terminate()`之类的函数来自标准库。虽然我们可以使用作用域解析运算符在它们的名称前加上库名称，比如`std::terminate()`，但这并非必需。

注意

异常处理并不意味着取代简单的程序员错误检查；异常处理的开销更大。异常处理应该保留用于以统一方式和在一个公共位置处理更严重的程序错误。

现在我们已经了解了异常处理的基本机制，让我们来看一些稍微复杂的异常处理示例。

## 探索异常处理机制及典型变化

异常处理可以比之前所示的基本机制更加复杂和灵活。让我们来看看异常处理基础的各种组合和变化，因为每种情况可能适用于不同的编程情况。

### 将异常传递给外部处理程序

捕获的异常可以传递给外部处理程序进行处理。或者，异常可以部分处理，然后抛出到外部范围进行进一步处理。

让我们在之前的示例基础上演示这个原则。完整的程序可以在以下 GitHub 位置看到：

[`github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter11/Chp11-Ex3.cpp`](https://github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter11/Chp11-Ex3.cpp)

```cpp
// Assume Student class is as we've seen it before, but with
// two additional member functions. Assume usual header files.
void Student::Validate()  // defined as virtual in class def
{                         // so derived classes may override
    // check constructed student to see if standards are met
    // if not, throw an exception
    throw "Does not meet prerequisites";
}
bool Student::TakePrerequisites()  
{
    // Assume this function can correct the issue at hand
    // if not, it returns false
    return false;
}
int main()
{
    Student s1("Alex", "Ren", 'Z', "Dr.", 3.9, "C++", "89CU");
    try    // illustrates a nested try block 
    {   
        // Assume another important task occurred in this
        // scope, which may have also raised an exception
        try
        {   
            s1.Validate();  // may raise an exception
        }
        catch (const char *err)
        {
            cout << err << endl;
            // try to correct (or partially handle) error.
            // If you cannot, pass exception to outer scope
            if (!s1.TakePrerequisites())
                throw;    // re-throw the exception
        }
    }
    catch (const char *err) // outer scope catcher (handler)
    {
        cout << err << endl;
        // try to fix problem here…
        exit(1); // only if you can't fix, exit gracefully
    } 
    cout << "Moving onward with remainder of code. " << endl;
    return 0;
}
```

在上述代码中，假设我们已经包含了我们通常的头文件，并且已经定义了`Student`的通常类定义。现在我们将通过添加`Student::Validate()`方法（虚拟的，以便可以被覆盖）和`Student::TakePrerequisites()`方法（非虚拟的，后代应该按原样使用）来增强`Student`类。

请注意，我们的`Student::Validate()`方法抛出一个异常，这只是一个包含指示问题的消息的字符串字面量。我们可以想象`Student::TakePrerequisites()`方法的完整实现验证了`Student`是否满足适当的先决条件，并根据情况返回`true`或`false`的布尔值。

在我们的`main()`函数中，我们现在注意到一组嵌套的 try 块。这里的目的是说明一个内部 try 块可能调用一个方法，比如`s1.Validate()`，这可能会引发异常。注意到与内部 try 块相同级别的处理程序捕获了这个异常。理想情况下，异常应该在与其来源的 try 块相等的级别上处理，所以让我们假设这个范围内的捕获器试图这样做。例如，我们最内层的 catch 块可能试图纠正错误，并通过调用`s1.TakePrerequisites()`来测试是否已经进行了纠正。

但也许这个捕获器只能部分处理异常。也许有一个外层处理程序知道如何进行剩余的修正。在这种情况下，将这个异常重新抛出到外层（嵌套）级别是可以接受的。我们在最内层的 catch 块中的简单的`throw;`语句就是这样做的。注意外层有一个捕获器。如果抛出的异常与外层的类型匹配，现在外层就有机会进一步处理异常，并希望纠正问题，以便应用程序可以继续。只有当这个外部 catch 块无法纠正错误时，应用程序才应该退出。在我们的例子中，每个捕获器都打印表示错误消息的字符串；因此这条消息在输出中出现了两次。

让我们看看上述程序的输出：

```cpp
Student does not meet prerequisites
Student does not meet prerequisites
```

现在我们已经看到了如何使用嵌套的 try 和 catch 块，让我们继续看看如何一起使用各种抛出类型和各种 catch 块。

### 添加各种处理程序

有时，内部范围可能会引发各种异常，从而需要为各种数据类型制定处理程序。异常处理程序（即 catch 块）可以接收任何数据类型的异常。我们可以通过使用基类类型的 catch 块来最小化引入的捕获器数量；我们知道派生类对象（通过公共继承相关）总是可以向上转换为它们的基类类型。我们还可以在 catch 块中使用省略号（`…`）来允许我们捕获以前未指定的任何东西。

让我们在我们的初始示例上建立，以说明各种处理程序的操作。虽然缩写，但我们完整的程序示例可以在我们的 GitHub 上找到：

[`github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter11/Chp11-Ex4.cpp`](https://github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter11/Chp11-Ex4.cpp)

```cpp
// Assume Student class is as we've seen before, but with one
// additional virtual member function, Graduate(). Assume 
// a simple Course class exists. All headers are as usual.
void Student::Graduate()
{   // Assume the below if statements are fully implemented 
    if (gpa < 2.0) // if gpa doesn't meet requirements
        throw gpa;
    // if Student is short credits, throw how many are missing
        throw numCreditsMissing;  // assume this is an int
    // or if Student is missing a Course, construct and
    // then throw the missing Course as a referenceable object
    // Assume appropriate Course constructor exists
        throw *(new Course("Intro. To Programming", 1234)); 
    // or if another issue, throw a diagnostic message
        throw ("Does not meet requirements"); 
}
int main()
{
    Student s1("Ling", "Mau", 'I', "Ms.", 3.1, "C++", "55UD");
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
        cout << "Needs to take: " << err.GetTitle() << endl;
        cout << "Course #: " << err.GetCourseNum() << endl;
        // If you correct the error, and continue the program, 
        // be sure to deallocate heap mem using: delete &err;
        exit(3);  // Otherwise, heap memory for err will be 
    }             // reclaimed upon exit()
    catch (const char *err)
    {
        cout << err << endl;
        exit(4); 
    }
    catch (...)
    {
        cout << "Exiting" << endl;
        exit(5);
    }
    cout << "Moving onward with remainder of code." << endl;
    return 0;
}
```

在上述代码段中，我们首先检查了`Student::Graduate()`成员函数。在这里，我们可以想象这个方法通过许多毕业要求，并且因此可能引发各种不同类型的异常。例如，如果`Student`实例的`gpa`太低，就会抛出一个浮点数作为异常，指示学生的`gpa`太低。如果`Student`的学分太少，就会抛出一个整数，指示学生还需要多少学分才能获得学位。

也许`Student::Graduate()`可能引发的最有趣的潜在错误是，如果学生的毕业要求中缺少了一个必需的`Course`。在这种情况下，`Student::Graduate()`将分配一个新的`Course`对象，通过构造函数填充`Course`的名称和编号。接下来，`Course`的指针被解引用，并且对象被引用抛出。处理程序随后可以通过引用捕获这个对象。

在`main()`函数中，我们只是在 try 块中包装了对`Student::Graduate()`的调用，因为这个语句可能会引发异常。接着 try 块后面是一系列的 catch 块 - 每种可能被抛出的对象类型对应一个`catch`语句。在这个序列中的最后一个 catch 块使用省略号(`…`)，表示这个 catch 块将处理`Student::Graduate()`抛出的任何其他类型的异常，这些异常没有被其他 catch 块捕获到。

实际上被激活的 catch 块是使用`const Course &err`捕获`Course`的那个。有了`const`限定符，我们不能在处理程序中修改`Course`，所以我们只能对这个对象应用`const`成员函数。

请注意，尽管上面显示的每个 catch 块只是简单地打印出错误然后退出，但理想情况下，catch 块应该尝试纠正错误，这样应用程序就不需要终止，允许在 catch 块下面的代码继续执行。

让我们看看上述程序的输出：

```cpp
Needs to take: Intro. to Programming
Course #: 1234
```

现在我们已经看到了各种抛出的类型和各种 catch 块，让我们继续向前了解在单个 try 块中应该将什么内容分组在一起。

### 在 try 块中分组相关的项目

重要的是要记住，当 try 块中的一行代码遇到异常时，try 块的其余部分将被忽略。相反，程序将继续执行匹配的 catch 块（或者如果没有合适的 catch 块存在，则调用`terminate()`）。然后，如果错误被修复，catch 块之后的代码将开始执行。请注意，我们永远不会返回来完成初始 try 块的其余部分。这种行为的含义是，你应该只在 try 块中将一起的元素分组在一起。也就是说，如果一个项目引发异常，完成该分组中的其他项目就不再重要了。

请记住，catch 块的目标是尽可能纠正错误。这意味着在适用的 catch 块之后，程序可能会继续向前。你可能会问：现在跳过了与 try 块相关的项目是否可以接受？如果答案是否定的，那么请重写你的代码。例如，你可能想在`try`-`catch`分组周围添加一个循环，这样如果 catch 块纠正了错误，整个企业就会重新开始，从初始的 try 块开始重试。

或者，将较小的、连续的`try`-`catch`分组。也就是说，*try*只在自己的 try 块中尝试一个重要的任务（后面跟着适用的 catch 块）。然后在自己的 try 块中尝试下一个任务，后面跟着适用的 catch 块，依此类推。

接下来，让我们看一种在函数原型中包含它可能抛出的异常类型的方法。

### 检查函数原型中的异常规范

我们可以通过扩展函数的签名来可选地指定 C++函数可能抛出的异常类型，包括可能被抛出的对象类型。然而，因为一个函数可能抛出多种类型的异常（或者根本不抛出异常），所以必须在运行时检查实际抛出的类型。因此，函数原型中的这些增强规范也被称为**动态异常规范**。

让我们看一个在函数的扩展签名中使用异常类型的例子：

```cpp
void Student::Graduate() throw(float, int, Course &, char *)
{
   // this method might throw any of the above mentioned types
}
void Student::Enroll() throw()
{
   // this method might throw any type of exception
}
```

在上述代码片段中，我们看到了`Student`的两个成员函数。`Student::Graduate()`在其参数列表后包含`throw`关键字，然后作为该方法的扩展签名的一部分，包含了可能从该函数中抛出的对象类型。请注意，`Student::Enroll()`方法在其扩展签名中仅在`throw()`后面有一个空列表。这意味着`Student::Enroll()`可能抛出任何类型的异常。

在这两种情况下，通过在签名中添加`throw()`关键字和可选的数据类型，我们提供了一种向该函数的用户宣布可能被抛出的对象类型的方法。然后我们要求程序员在 try 块中包含对该方法的任何调用，然后跟上适当的 catcher。

我们将看到，尽管扩展签名的想法似乎非常有帮助，但在实践中存在不利问题。因此，动态异常规范已被*弃用*。因为您可能仍然会在现有代码中看到这些规范的使用，包括标准库原型（如异常），编译器仍然支持这个已弃用的特性，您需要了解它们的用法。

尽管动态异常（如前所述的扩展函数签名）已被弃用，但语言中已添加了具有类似目的的指定符号`noexcept`关键字。此指定符号可以在扩展签名之后添加如下：

```cpp
void Student::Graduate() noexcept   // will not throw() 
{            // same as  noexcept(true) in extended signature
}            // same as deprecated throw() in ext. signature
void Student::Enroll() noexcept(false)  // may throw()
{                                       // an exception
}                                     
```

尽管如此，让我们调查一下为什么与动态异常相关的不利问题存在，看看当我们的应用程序抛出不属于函数扩展签名的异常时会发生什么。

### 处理意外类型的动态异常

如果在扩展函数原型中指定的类型之外抛出了异常，C++标准库中的`unexpected()`将被调用。您可以像我们在本章前面注册`set_terminate()`时那样，注册自己的函数到`unexpected()`。

您可以允许您的`AppSpecificUnexpected()`函数重新抛出应该由原始函数抛出的异常类型，但是如果没有发生这种情况，将会调用`terminate()`。此外，如果没有可能匹配的 catcher 存在来处理从原始函数正确抛出的内容（或者由您的`AppSpecificUnexpected()`重新抛出），那么将调用`terminate()`。

让我们看看如何使用我们自己的函数`set_unexpected()`：

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

注册我们自己的函数到`set_unexpected()`非常简单，就像前面章节中所示的代码片段一样。

历史上，在函数的扩展签名中使用异常规范的一个激励原因是提供文档效果。也就是说，您可以通过检查其签名来看到函数可能抛出的异常，然后计划在 try 块中封装该函数调用，并提供适当的 catcher 来处理任何潜在情况。

然而，关于动态异常，值得注意的是编译器不会检查函数体中实际抛出的异常类型是否与函数扩展签名中指定的类型匹配。这取决于程序员来确保它们同步。因此，这个已弃用的特性可能容易出错，总体上比其原始意图更少用。

尽管初衷良好，动态异常目前未被使用，除了在大量的库代码中，比如 C++标准库。由于您将不可避免地使用这些库，了解这些过时的特性非常重要。

注意

在 C++中，动态异常规范（即在方法的扩展签名中指定异常类型的能力）已经被*弃用*。这是因为编译器无法验证它们的使用，必须延迟到运行时。尽管它们仍然受支持（许多库具有这种规范），但现在已经被弃用。

现在我们已经看到了一系列异常处理检测、引发、捕获和（希望）纠正方案，让我们看看如何创建一系列异常类的层次结构，以增强我们的错误处理能力。

# 利用异常层次结构

创建一个类来封装与程序错误相关的细节似乎是一个有用的努力。事实上，C++标准库已经创建了一个这样的通用类，`exception`，为构建整个有用的异常类层次结构提供了基础。

让我们看看带有其标准库后代的`exception`类，然后看看我们如何用自己的类扩展`exception`。

## 使用标准异常对象

`<exception>`头文件。`exception`类包括一个带有以下签名的虚函数：`virtual const char *what() const throw()`。这个签名表明派生类应该重新定义`what()`，返回一个描述手头错误的`const char *`。`what()`后面的`const`关键字表示这是一个`const`成员函数；它不会改变派生类的任何成员。扩展签名中的`throw()`表示这个函数可能抛出任何类型。在签名中使用`throw()`是一个已弃用的陈词滥调。

`std::exception`类是各种预定义的 C++异常类的基类，包括`bad_alloc`、`bad_cast`、`bad_exception`、`bad_function_call`、`bad_typeid`、`bad_weak_ptr`、`logic_error`、`runtime_error`和嵌套类`ios_base::failure`。这些派生类中的许多都有自己的后代，为预定义的异常层次结构添加了额外的标准异常。

如果函数抛出了上述任何异常，这些异常可以通过捕获基类类型`exception`或捕获单个派生类类型来捕获。根据处理程序将采取的行动，您可以决定是否希望将这样的异常作为其广义基类类型或特定类型捕获。

就像标准库基于`exception`类建立了一系列类的层次结构一样，你也可以。接下来让我们看看我们可能如何做到这一点！

## 创建自定义异常类

作为程序员，您可能会认为建立自己的专门异常类型是有利的。每种类型可以将有用的信息打包到一个对象中，详细说明应用程序出了什么问题。此外，您可能还可以将线索打包到（将被抛出的）对象中，以指导如何纠正手头的错误。只需从标准`exception`类派生您的类。

让我们通过检查我们下一个示例的关键部分来看看这是如何轻松实现的，完整的程序可以在我们的 GitHub 上找到：

[`github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter11/Chp11-Ex5.cpp`](https://github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter11/Chp11-Ex5.cpp)

```cpp
#include <iostream>
#include <exception>
using namespace std;
class StudentException: public exception
{
private:
    int errCode;  
    char *details;
public:           
    StudentException(const char *det, int num): errCode(num)
    {
        details = new char[strlen(det) + 1];
        strcpy(details, det);
    }   
    virtual ~StudentException() { delete details; }
    virtual const char *what() const throw()
    {   // overridden function from exception class
        return "Student Exception";
    } 
    int GetCode() const { return errCode; }
    const char *GetDetails() const { return details; }
};
// Assume Student class is as we've seen before, but with one
// additional virtual member function Graduate() 
void Student::Graduate()  // fn. may throw (StudentException)
{
   // if something goes wrong, instantiate a StudentException,
   // pack it with relevant data during construction, and then
   // throw the dereferenced pointer as a referenceable object
   throw *(new StudentException("Missing Credits", 4));
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
        cout << e.GetCode() << " " << e.GetDetails() << endl;
        // Grab useful info from e and try to fix the problem
        // so that the program can continue.
        // If we fix the problem, deallocate heap memory for
        // thrown exception (take addr. of a ref): delete &e; 
        // Otherwise, memory will be reclaimed upon exit()
        exit(1);  // only exit if necessary!
    }
    return 0;
}
```

让我们花几分钟来检查前面的代码段。首先，注意我们定义了自己的异常类，`StudentException`。它是从 C++标准库`exception`类派生的类。

`StudentException`类包含数据成员来保存错误代码以及使用数据成员`errCode`和`details`描述错误条件的字母数字细节。我们有两个简单的访问函数，`StudentException::GetCode()`和`StudentException::GetDetails()`，可以轻松地检索这些值。由于这些方法不修改对象，它们是`const`成员函数。

我们注意到`StudentException`构造函数通过成员初始化列表初始化了两个数据成员，一个在构造函数的主体中初始化。我们还重写了`exception`类引入的`virtual const char *what() const throw()`方法。请注意，`exception::what()`方法在其扩展签名中使用了不推荐的`throw()`规范，这也是你必须在你的重写方法中做的事情。

接下来，让我们检查一下我们的`Student::Graduate()`方法。这个方法可能会抛出一个`StudentException`。如果必须抛出异常，我们使用`new()`分配一个异常，用诊断数据构造它，然后从这个函数中`throw`解引用指针（这样我们抛出的是一个可引用的对象，而不是一个对象的指针）。请注意，在这个方法中抛出的对象没有本地标识符 - 没有必要，因为任何这样的本地变量名很快就会在`throw`发生后从堆栈中弹出。

在我们的`main()`函数中，我们将对`s1.Graduate()`的调用包装在一个 try 块中，后面是一个接受`StudentException`的引用（`&`）的 catch 块，我们将其视为`const`。在这里，我们首先调用我们重写的`what()`方法，然后从异常`e`中打印出诊断细节。理想情况下，我们将使用这些信息来尝试纠正手头的错误，只有在真正必要时才退出应用程序。

让我们看一下上述程序的输出：

```cpp
Student Exception
4 Missing Credits
```

尽管创建自定义异常类的最常见方式是从标准的`exception`类派生一个类，但也可以利用不同的技术，即嵌套异常类。

### 创建嵌套异常类

作为另一种实现，异常处理可以通过在特定外部类的公共访问区域添加嵌套类定义来嵌入到一个类中。内部类将代表自定义异常类。

嵌套的、用户定义的类型的对象可以被创建并抛出给预期这种类型的 catcher。这些嵌套类内置在外部类的公共访问区域，使它们很容易为派生类的使用和特化而使用。一般来说，内置到外部类中的异常类必须是公共的，以便可以在外部类的范围之外（即在主要的外部实例存在的范围内）捕获和处理抛出的嵌套类型的实例。

让我们通过检查代码的关键部分来看一下异常类的另一种实现，完整的程序可以在我们的 GitHub 上找到：

[`github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter11/Chp11-Ex6.cpp`](https://github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/blob/master/Chapter11/Chp11-Ex6.cpp)

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
        int number;
    public:
        StudentException(int num): number(num) { }
        ~StudentException() { }
        int GetNum() const { return number; }
    };
};
void Student::Graduate()
{   // assume we determine an error and wish to throw
    // the nested exception type
    throw *(new StudentException(5));
}
int main()
{
    Student s1("Ling", "Mau", 'I', "Ms.", 3.1, "C++", "55UD");
    try
    {
        s1.Graduate();
    }
    catch (const Student::StudentException &err)
    {
        cout << "Error: " << err.GetNum() << endl;
        // If you correct err and continue with program, be
        // sure to delete heap mem for err: delete &err;
        exit(1);  // Otherwise, heap memory for err will be 
    }             // reclaimed upon exit()
    cout << "Moving onward with remainder of code." << endl;
    return 0;
}
```

在前面的代码片段中，我们扩展了`Student`类，包括一个名为`StudentException`的私有嵌套类。尽管所示的类过于简化，但嵌套类理想上应该定义一种方法来记录相关错误以及收集任何有用的诊断信息。

在我们的`main()`函数中，我们实例化了一个`Student`，名为`s1`。然后在 try 块中调用`s1.Graduate()`。我们的`Student::Graduate()`方法可能会检查`Student`是否符合毕业要求，如果不符合，则抛出一个嵌套类类型`Student::StudentException`的异常（根据需要实例化）。

请注意，我们相应的`catch`块利用作用域解析来指定`err`的内部类类型（即`const Student::StudentException &err`）。虽然我们理想情况下希望在处理程序内部纠正程序错误，但如果我们无法这样做，我们只需打印一条消息并`exit()`。

让我们看看上述程序的输出：

```cpp
Error: 5
```

了解如何创建我们自己的异常类（作为嵌套类或派生自`std::exception`）是有用的。我们可能还希望创建一个特定于应用程序的异常的层次结构。让我们继续看看如何做到这一点。

## 创建用户定义异常类型的层次结构

一个应用程序可能希望定义一系列支持异常处理的类，以引发特定错误，并希望提供一种收集错误诊断信息的方法，以便在代码的适当部分处理错误。

您可能希望创建一个从标准库`exception`派生的子层次结构，属于您自己的异常类。确保使用公共继承。在使用这些类时，您将实例化所需异常类型的对象（填充有有价值的诊断信息），然后抛出该对象。请记住，您希望新分配的对象存在于堆上，以便在函数返回时不会从堆栈中弹出（因此使用`new`进行分配）。在抛出之前简单地对这个对象进行解引用，以便它可以被捕获为对该对象的引用，这是标准做法。

此外，如果您创建异常类型的层次结构，您的 catcher 可以捕获特定的派生类类型或更一般的基类类型。选择权在您手中，取决于您计划如何处理异常。但请记住，如果您对基类和派生类类型都有 catcher，请将派生类类型放在前面 - 否则，您抛出的对象将首先匹配到基类类型的 catcher，而不会意识到更合适的派生类匹配是可用的。

我们现在已经看到了 C++标准库异常类的层次结构，以及如何创建和利用自己的异常类。让我们在继续前进到下一章之前，简要回顾一下本章中我们学到的异常特性。

# 总结

在本章中，我们已经开始将我们的 C++编程技能扩展到 OOP 语言特性之外，以包括能够编写更健壮程序的特性。用户代码不可避免地具有错误倾向；使用语言支持的异常处理可以帮助我们实现更少错误和更可靠的代码。

我们已经看到如何使用`try`、`throw`和`catch`来利用核心异常处理特性。我们已经看到了这些关键字的各种用法 - 将异常抛出到外部处理程序，使用各种类型的处理程序，以及在单个 try 块内有选择地将程序元素分组在一起，例如。我们已经看到如何使用`set_terminate()`和`set_unexpected()`注册我们自己的函数。我们已经看到了如何利用现有的 C++标准库`exception`层次结构。我们还探讨了定义我们自己的异常类以扩展此层次结构。

通过探索异常处理机制，我们已经为我们的 C++技能增加了关键特性。现在我们准备继续前进到*第十二章*，*友元和运算符重载*，以便我们可以继续扩展我们的 C++编程技能，使用有用的语言特性，使我们成为更好的程序员。让我们继续前进！

# 问题

1.  将异常处理添加到您之前的`Student` / `University`练习中*第十章*，*实现关联、聚合和组合*：

a. 如果一个`学生`尝试注册超过每个`学生`允许的`最大`定义课程数量，抛出`TooFullSchedule`异常。这个类可以从标准库`exception`类派生。

b. 如果一个`学生`尝试注册一个已经满员的`课程`，让`Course::AddStudent(Student *)`方法抛出一个`CourseFull`异常。这个类可以从标准库`exception`类派生。

c. `学生`/`大学`申请中还有许多其他领域可以利用异常处理。决定哪些领域应该采用简单的错误检查，哪些值得异常处理。
