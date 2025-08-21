# 第二十一章：评估

每章的编程解决方案可以在我们的 GitHub 存储库的以下 URL 找到：[`github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/tree/master`](https://github.com/PacktPublishing/Demystified-Object-Oriented-Programming-with-CPP/tree/master)。每个完整的程序解决方案可以在 GitHub 的适当章节标题下（子目录，如`Chapter01`）的`Assessments`子目录中找到，文件名对应于章节编号，后跟着该章节中的解决方案编号的破折号。例如，第一章问题 3 的解决方案可以在 GitHub 目录中的`Chapter01/Assessments`子目录中的名为`Chp1-Q3.cpp`的文件中找到。

非编程问题的书面答复可以在以下部分找到。如果一个练习有编程部分和后续问题，后续问题的答案可以在下一部分和 GitHub 上编程解决方案的顶部评论中找到（因为可能需要查看解决方案才能完全理解问题的答案）。

# 第一章 - 理解基本的 C++假设

1.  在不希望光标移到下一行进行输出的情况下，使用`flush`可能比`endl`更有用，用于清除与`cout`相关的缓冲区的内容。请记住，`endl`操作符仅仅是一个换行字符加上一个缓冲区刷新。

1.  选择变量的前置增量还是后置增量，比如`++i`（与`i++`相比），将影响与复合表达式一起使用时的代码。一个典型的例子是`result = array[i++];`与`result = array[++i];`。使用后置增量（`i++`），`array[i]`的内容将被赋给`result`，然后`i`被增加。使用前置增量，`i`首先被增加，然后`result`将具有`array[i]`的值（即，使用`i`的新值作为索引）。

1.  请参阅 GitHub 存储库中的`Chapter01/Assessments/Chp1-Q3.cpp`。

# 第二章 - 添加语言必需品

1.  函数的签名是函数的名称加上其类型和参数数量（没有返回类型）。这与名称修饰有关，因为签名帮助编译器为每个函数提供一个唯一的内部名称。例如，`void Print(int, float);`可能有一个名称修饰为`Print_int_float();`。这通过为每个函数提供一个唯一的名称来促进重载函数，因此当调用被执行时，可以根据内部函数名称明确调用哪个函数。

1.  在 GitHub 存储库中的`Chapter02/Assessments/Chp2-Q2.cpp`。

# 第三章 - 间接寻址：指针

1.  在 GitHub 存储库中的`Chapter03/Assessments/Chp3-Q1.cpp`。

`Print(Student)`比`Print(const Student *)`效率低，因为这个函数的初始版本在堆栈上传递整个对象，而重载版本只在堆栈上传递一个指针。

1.  假设我们有一个指向`Student`类型对象的现有指针，比如：

`Student *s0 = new Student;`（这个`Student`还没有用数据初始化）

`const Student *s1;`（不需要初始化）

`Student *const s2 = s0;`（需要初始化）

`const Student *const s3 = s0;`（也需要初始化）

1.  将类型为`const Student *`的参数传递给`Print()`将允许将`Student`的指针传递给`Print()`以提高速度，但指向的对象不能被取消引用和修改。然而，将`Student * const`作为`Print()`的参数传递是没有意义的，因为指针的副本将被传递给`Print()`。将该副本标记为`const`（意味着不允许更改指针的指向）将是没有意义的，因为不允许更改指针的*副本*对原始指针本身没有影响。原始指针从未面临在函数内部更改其地址的风险。

1.  有许多编程情况可能使用动态分配的 3-D 数组。例如，如果一个图像存储在 2-D 数组中，一组图像可能存储在 3-D 数组中。动态分配的 3-D 数组允许从文件系统中读取任意数量的图像并在内部存储。当然，在进行 3-D 数组分配之前，你需要知道要读取多少图像。例如，一个 3-D 数组可能包含 30 张图像，其中 30 是第三维，用于收集图像集。为了概念化一个 4-D 数组，也许你想要组织前述 3-D 数组的集合。

例如，也许你有一个包含 31 张图片的一月份的图片集。这组一月份的图片是一个 3-D 数组（2-D 用于图像，第三维用于包含一月份的 31 张图片的集合）。你可能希望对每个月都做同样的事情。我们可以创建一个第四维来将一年的数据收集到一个集合中，而不是为每个月的图像集创建单独的 3-D 数组变量。第四维将为一年的 12 个月中的每个月都有一个元素。那么 5-D 数组呢？你可以通过将第五维作为收集各年数据的方式来扩展这个图像的想法，比如收集一个世纪的图像（第五维）。现在我们有了按世纪组织的图像，然后按年份组织，然后按月份组织，最后按图像组织（图像需要前两个维度）。

# 第四章 - 间接寻址：引用

1.  在 GitHub 存储库中的`Chapter04/Assessments/Chp4-Q1.cpp`。

`ReadData(Student *)`接受一个指向`Student`的指针和引用变量不仅需要调用接受`Student`引用的`ReadData(Student &)`版本。例如，指针变量可以使用`*`取消引用，然后调用接受引用的版本。同样，引用变量可以使用`&`取其地址，然后调用接受指针的版本（尽管这种情况较少见）。你只需要确保传递的数据类型与函数期望的匹配。

# 第五章 - 详细探讨类

1.  在 GitHub 存储库中的`Chapter05/Assessments/Chp5-Q1.cpp`。

# 第六章 - 使用单继承实现层次结构

1.  在 GitHub 存储库中的`Chapter06/Assessments/Chp6-Q1.cpp`。

1.  在 GitHub 存储库中的`Chapter06/Assessments/Chp6-Q2.cpp`。

# 第七章 - 通过多态性利用动态绑定

1.  在 GitHub 存储库中的`Chapter07/Assessments/Chp7-Q1.cpp`。

# 第八章 - 掌握抽象类

1.  在 GitHub 存储库中的`Chapter08/Assessments/Chp8-Q1.cpp`。

`Shape`类可能被视为接口类，也可能不是。如果你的实现是一个不包含数据成员，只包含抽象方法（纯虚函数）的抽象类，那么你的`Shape`实现被认为是一个接口类。然而，如果你的`Shape`类在派生类中的重写`Area()`方法计算出`area`后将其存储为数据成员，那么它只是一个抽象基类。

# 第九章 - 探索多重继承

1.  请参阅 GitHub 存储库中的`Chapter09/Assessments/Chp9-Q1.cpp`。

`LifeForm`子对象。

`LifeForm`构造函数和析构函数各被调用一次。

如果`Centaur`构造函数的成员初始化列表中删除了`LifeForm(1000)`的替代构造函数的规范，则将调用`LifeForm`。

1.  请在 GitHub 存储库中查看`Chapter09/Assessments/Chp9-Q2.cpp`。

`LifeForm`子对象。

`LifeForm`构造函数和析构函数各被调用两次。

# 第十章-实现关联、聚合和组合

1.  请在 GitHub 存储库中查看`Chapter10/Assessments/Chp10-Q1.cpp`。

(后续问题)一旦您重载了一个接受`University &`作为参数的构造函数，可以通过首先取消引用构造函数调用中的`University`指针来调用这个版本（使其成为可引用的对象）。

1.  在 GitHub 存储库中的`Chapter10/Assessments/Chp10-Q2.cpp`。

1.  在 GitHub 存储库中的`Chapter10/Assessments/Chp10-Q3.cpp`。

# 第十一章-处理异常

1.  在 GitHub 存储库中的`Chapter11/Assessments/Chp11-Q1.cpp`。

# 第十二章-友元和运算符重载

1.  请在 GitHub 存储库中查看`Chapter12/Assessments/Chp12-Q1.cpp`。

1.  请在 GitHub 存储库中查看`Chapter12/Assessments/Chp12-Q2.cpp`。

1.  请在 GitHub 存储库中查看`Chapter12/Assessments/Chp12-Q3.cpp`。

# 第十三章-使用模板

1.  在 GitHub 存储库中的`Chapter13/Assessments/Chp13-Q1.cpp`。

1.  请在 GitHub 存储库中查看`Chapter13/Assessments/Chp13-Q2.cpp`。

# 第十四章-理解 STL 基础

1.  在 GitHub 存储库中的`Chapter14/Assessments/Chp14-Q1.cpp`。

1.  请在 GitHub 存储库中查看`Chapter14/Assessments/Chp14-Q2.cpp`。

1.  请在 GitHub 存储库中查看`Chapter14/Assessments/Chp14-Q3.cpp`。

1.  请在 GitHub 存储库中查看`Chapter14/Assessments/Chp14-Q4.cpp`。

# 第十五章-测试类和组件

1.  **a**：如果每个类都包括（用户指定的）默认构造函数、复制构造函数、重载的赋值运算符和虚析构函数，则您的类遵循正统的规范类形式。如果它们还包括移动复制构造函数和重载的移动赋值运算符，则您的类还遵循扩展的规范类形式。

**b**：如果您的类遵循规范类形式，并确保类的所有实例都具有完全构造的手段，则您的类将被视为健壮的。测试类可以确保健壮性。

1.  在 GitHub 存储库中的`Chapter15/Assessments/Chp15-Q2.cpp`。

1.  请在 GitHub 存储库中查看`Chapter15/Assessments/Chp15-Q3.cpp`。

# 第十六章-使用观察者模式

1.  在 GitHub 存储库中的`Chapter16/Assessments/Chp16-Q1.cpp`。

1.  其他很容易包含观察者模式的例子包括任何需要顾客接收所需产品缺货通知的应用程序。例如，许多人可能希望接种 Covid-19 疫苗，并希望在疫苗分发站的等候名单上。在这里，`VaccineDistributionSite`（感兴趣的主题）可以从`Subject`继承，并包含一个`Person`对象列表，其中`Person`继承自`Observer`。`Person`对象将包含一个指向`VaccineDistributionSite`的指针。一旦在给定的`VaccineDistributionSite`上存在足够的疫苗供应（即，分发事件已发生），就可以调用`Notify()`来更新`Observer`实例（等候名单上的人）。每个`Observer`将收到一个`Update()`，这将是允许该人安排约会的手段。如果`Update()`返回成功并且该人已经安排了约会，`Observer`可以通过`Subject`从等候名单中释放自己。

# 第十七章-应用工厂模式

1.  在 GitHub 存储库中的`Chapter17/Assessments/Chp17-Q1.cpp`。

1.  其他可能很容易融入工厂方法模式的例子包括许多类型的应用程序，其中根据提供的特定值实例化各种派生类可能是必要的。例如，工资单应用程序可能需要各种类型的`Employee`实例，如`Manager`、`Engineer`、`Vice-President`等。工厂方法可以根据雇佣`Employee`时提供的信息来实例化各种类型的`Employee`。工厂方法模式是一种可以应用于许多类型的应用程序的模式。

# 第十八章 - 应用适配器模式

1.  在 GitHub 存储库中的`Chapter18/Assessments/Chp18-Q1.cpp`。

1.  其他可能很容易融入适配器模式的例子包括许多重用现有、经过充分测试的非面向对象代码以提供面向对象接口（即适配器的包装类型）的例子。其他例子包括创建一个适配器，将以前使用的类转换为当前需要的类（再次使用先前创建和经过充分测试的组件的想法）。一个例子是将以前用于表示汽油发动机汽车的`Car`类改编为模拟`ElectricCar`的类。

# 第十九章 - 使用单例模式

1.  `Chapter19/Assessments/Chp19-Q1.cpp`

1.  我们不能将`Singleton`中的`static instance()`方法标记为虚拟的，并在`President`中重写它，因为静态方法永远不可能是虚拟的。它们是静态绑定的，也永远不会接收到`this`指针。此外，签名可能需要不同（没有人喜欢无意的函数隐藏情况）。

1.  其他例子可能很容易地融入单例模式，包括创建一个公司的单例`CEO`，或者一个国家的单例`TreasuryDepartment`，或者一个国家的单例`Queen`。这些单例实例都提供了建立注册表以跟踪多个单例对象的机会。也就是说，许多国家可能只有一个`Queen`。在这种情况下，注册表不仅允许每种对象类型有一个单例，而且还允许每个其他限定符（如*国家*）有一个单例。这是一个罕见的例子，其中同一类型的单例对象可能会出现多个（但始终是受控数量的对象）。

# 第二十章 - 使用 pImpl 模式去除实现细节

1.  请参阅 GitHub 存储库中的`Chapter20/Assessments/Chp20-Q1.cpp`。

1.  请参阅 GitHub 存储库中的`Chapter20/Assessments/Chp20-Q2.cpp`。

（后续问题）在本章中，从`Person`类中简单地继承`Student`，这个类采用了 pImpl 模式，不会出现后勤上的困难。此外，修改`Student`类以使用 pImpl 模式并利用独特指针更具挑战性。各种方法可能会遇到各种困难，包括处理内联函数、向下转型、避免显式调用底层实现，或需要反向指针来帮助调用虚拟函数。有关详细信息，请参阅在线解决方案。

1.  其他可能很容易融入 pImpl 模式以实现相对独立的实现的例子包括创建通用的 GUI 组件，比如`Window`、`Scrollbar`、`Textbox`等，用于各种平台（派生类）。实现细节可以很容易地隐藏起来。其他例子包括希望隐藏在头文件中可能看到的实现细节的专有商业类。
