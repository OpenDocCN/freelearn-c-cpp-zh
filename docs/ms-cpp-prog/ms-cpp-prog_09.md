# 第九章：调试技术

在本章中，我们将涵盖以下主题：

+   有效的调试

+   调试策略

+   调试工具

+   使用 GDB 调试应用程序

+   使用 Valgrind 调试内存泄漏

+   日志记录

# 有效的调试

调试是一门艺术而不是一门科学，它本身是一个非常庞大的主题。强大的调试技能是一个优秀开发人员的优势。所有专业的开发人员都有一些共同的特点，其中强大的问题解决和调试技能是最重要的。修复错误的第一步是复现问题。高效地捕获复现错误所涉及的步骤至关重要。有经验的 QA 工程师会知道捕获详细的复现步骤的重要性，因为如果无法复现问题，开发人员将很难修复问题。

在我看来，无法复现的错误无法修复。人们可以猜测和打草稿，但如果一开始就无法复现问题，就无法确定问题是否真正被修复。

以下详细信息将帮助开发人员更快地复现和调试问题：

+   详细的复现问题的步骤

+   错误的屏幕截图

+   优先级和严重程度

+   复现问题的输入和场景

+   预期和实际输出

+   错误日志

+   应用程序日志和跟踪

+   在应用程序崩溃时转储文件

+   环境详细信息

+   操作系统详细信息

+   软件版本

一些常用的调试技术如下：

+   使用`cout`/`cerr`打印语句非常方便

+   核心转储、迷你转储和完整转储有助于远程分析错误

+   使用调试工具逐步执行代码，检查变量、参数、中间值等

+   测试框架有助于在第一时间防止问题的发生

+   性能分析工具可以帮助找到性能问题

+   检测内存泄漏、资源泄漏、死锁等工具

`log4cpp`开源 C++库是一个优雅且有用的日志实用程序，它可以添加支持调试的调试消息，在发布模式或生产环境中可以禁用。

# 调试策略

调试策略有助于快速复现、调试、检测和高效修复问题。以下列表解释了一些高级调试策略：

+   使用缺陷跟踪系统，如 JIRA、Bugzilla、TFS、YouTrack、Teamwork 等

+   应用程序崩溃或冻结必须包括核心转储、迷你转储或完整转储

+   应用程序跟踪日志在所有情况下都是一个很好的帮助

+   启用多级错误日志

+   在调试和发布模式下捕获应用程序跟踪日志

# 调试工具

调试工具通过逐步执行、断点、变量检查等帮助缩小问题范围。尽管逐步调试问题可能是一项耗时的任务，但这绝对是确定问题的一种方法，我可以说这几乎总是有效的。

以下是 C++的调试工具列表：

+   **GDB**：这是一个开源的 CLI 调试器

+   **Valgrind**：这是一个用于内存泄漏、死锁、竞争检测等的开源 CLI 工具

+   **Affinic debugger**：这是一个用于 GDB 的商业 GUI 工具

+   **GNU DDD**：这是一个用于 GDB、DBX、JDB、XDB 等的开源图形调试器

+   **GNU Emacs GDB 模式**：这是一个带有最小图形调试器支持的开源工具

+   **KDevelop**：这是一个带有图形调试器支持的开源工具

+   **Nemiver**：这是一个在 GNOME 桌面环境中运行良好的开源工具

+   **SlickEdit**：适用于调试多线程和多处理器代码

在 C++中，有很多开源和商业许可的调试工具。然而，在本书中，我们将探索 GDB 和 Valgrind 这两个开源命令行界面工具。

# 使用 GDB 调试应用程序

经典的老式 C++开发人员使用打印语句来调试代码。然而，使用打印跟踪消息进行调试是一项耗时的任务，因为您需要在多个地方编写打印语句，重新编译并执行应用程序。

老式的调试方法需要许多这样的迭代，通常每次迭代都需要添加更多的打印语句以缩小问题范围。一旦问题解决了，我们需要清理代码并删除打印语句，因为太多的打印语句会减慢应用程序的性能。此外，调试打印消息会分散注意力，对于在生产环境中使用您产品的最终客户来说是无关紧要的。

C++调试`assert()`宏语句与`<cassert>`头文件一起使用于调试。C++ `assert()`宏在发布模式下可以被禁用，只有在调试模式下才启用。

调试工具可以帮助您摆脱这些繁琐的工作。GDB 调试器是一个开源的 CLI 工具，在 Unix/Linux 世界中是 C++的调试器。对于 Windows 平台，Visual Studio 是最受欢迎的一站式 IDE，具有内置的调试功能。

让我们举一个简单的例子：

```cpp
#include <iostream>
#include <vector>
#include <iterator>
#include <algorithm>
using namespace std; //Use this judiciously - this is applicable throughout the book

class MyInteger {
      private:
           int number;

      public:
           MyInteger( int value ) {
                this->number = value;
           }

           MyInteger(const MyInteger & rhsObject ) {
                this->number = rhsObject.number;
           }

           MyInteger& operator = (const MyInteger & rhsObject ) {

                if ( this != &rhsObject )
                     this->number = rhsObject.number;

                return *this;
           }

           bool operator < (const MyInteger &rhsObject) {
                return this->number > rhsObject.number;
           }

           bool operator > (const MyInteger &rhsObject) {
                return this->number > rhsObject.number;
           }

           friend ostream & operator << ( ostream &output, const MyInteger &object );
};

ostream & operator << (ostream &o, const MyInteger& object) {
    o << object.number;
}

int main ( ) {

    vector<MyInteger> v = { 10, 100, 40, 20, 80, 70, 50, 30, 60, 90 };

    cout << "\nVectors entries before sorting are ..." << endl;
    copy ( v.begin(), v.end() , ostream_iterator<MyInteger>( cout, "\t" ) );
    cout << endl;

    sort ( v.begin(), v.end() );

    cout << "\nVectors entries after sorting are ..." << endl;
    copy ( v.begin(), v.end() , ostream_iterator<MyInteger>( cout, "\t" ) );
    cout << endl;

    return 0;
}
```

程序的输出如下：

```cpp
Vectors entries before sorting are ...
10 100 40 20 80 70 50 30 60 90

Vectors entries after sorting are ...
100 90 80 70 60 50 40 30 20 10
```

然而，我们期望的输出如下：

```cpp
Vectors entries before sorting are ...
10 100 40 20 80 70 50 30 60 90

Vectors entries after sorting are ...
10 20 30 40 50 60 70 80 90 100
```

错误是显而易见的；让我们轻松地学习 GDB。让我们首先以调试模式编译程序，也就是启用调试元数据和符号表，如下所示：

```cpp
g++ main.cpp -std=c++17 -g
```

# GDB 命令快速参考

以下 GDB 快速提示表将帮助您找到调试应用程序的 GDB 命令：

| **命令** | **简短命令** | **描述** |
| --- | --- | --- |
| `gdb yourappln.exe` | `-` | 在 GDB 中打开应用程序进行调试 |
| `break main` | `b main` | 将断点设置为`main`函数 |
| `run` | `r` | 执行程序直到达到逐步执行的断点 |
| `next` | `n` | 逐步执行程序 |
| `step` | `s` | 步入函数以逐步执行函数 |
| `continue` | `c` | 继续执行程序直到下一个断点；如果没有设置断点，它将正常执行应用程序 |
| `backtrace` | `bt` | 打印整个调用堆栈 |
| `quit` | `q`或`Ctrl + d` | 退出 GDB |
| `-help` | `-h` | 显示可用选项并简要显示其用法 |

有了上述基本的 GDB 快速参考，让我们开始调试我们有问题的应用程序以检测错误。让我们首先使用以下命令启动 GDB：

```cpp
gdb ./a.out
```

然后，让我们在`main()`处添加一个断点以进行逐步执行：

```cpp
jegan@ubuntu:~/MasteringC++Programming/Debugging/Ex1$ g++ main.cpp -g
jegan@ubuntu:~/MasteringC++Programming/Debugging/Ex1$ ls
a.out main.cpp
jegan@ubuntu:~/MasteringC++Programming/Debugging/Ex1$ gdb ./a.out

GNU gdb (Ubuntu 7.12.50.20170314-0ubuntu1.1) 7.12.50.20170314-git
Copyright (C) 2017 Free Software Foundation, Inc.
License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law. Type "show copying"
and "show warranty" for details.
This GDB was configured as "x86_64-linux-gnu".
Type "show configuration" for configuration details.
For bug reporting instructions, please see:
<http://www.gnu.org/software/gdb/bugs/>.
Find the GDB manual and other documentation resources online at:
<http://www.gnu.org/software/gdb/documentation/>.
For help, type "help".
Type "apropos word" to search for commands related to "word"...
Reading symbols from ./a.out...done.
(gdb) b main
Breakpoint 1 at 0xba4: file main.cpp, line 46.
(gdb) l
32 
33 bool operator > (const MyInteger &rhsObject) {
34 return this->number < rhsObject.number;
35 }
36 
37 friend ostream& operator << ( ostream &output, const MyInteger &object );
38 
39 };
40 
41 ostream& operator << (ostream &o, const MyInteger& object) {
(gdb)
```

使用`gdb`启动我们的应用程序后，`b main`命令将在`main()`函数的第一行添加一个断点。现在让我们尝试执行应用程序：

```cpp
(gdb) run
Starting program: /home/jegan/MasteringC++Programming/Debugging/Ex1/a.out 

Breakpoint 1, main () at main.cpp:46
46 int main ( ) {
(gdb) 
```

正如您可能已经观察到的，程序执行在我们的`main()`函数的行号`46`处暂停，因为我们在`main()`函数中添加了一个断点。

此时，让我们逐步执行应用程序，如下所示：

```cpp
(gdb) run
Starting program: /home/jegan/MasteringC++Programming/Debugging/Ex1/a.out 

Breakpoint 1, main () at main.cpp:46
46 int main ( ) {
(gdb) next
48   vector<MyInteger> v = { 10, 100, 40, 20, 80, 70, 50, 30, 60, 90 };
(gdb) next
50   cout << "\nVectors entries before sorting are ..." << endl;
(gdb) n
Vectors entries before sorting are ...51   copy ( v.begin(), v.end() , ostream_iterator<MyInteger>( cout, "\t" ) );
(gdb) n
52   cout << endl;
(gdb) n
10 100 40 20 80 70 50 30 60 90 
54   sort ( v.begin(), v.end() );
(gdb) 
```

现在，让我们在行号`29`和`33`处再添加两个断点，如下所示：

```cpp
Breakpoint 1 at 0xba4: file main.cpp, line 46.Breakpoint 1 at 0xba4: file main.cpp, line 46.(gdb) run
Starting program: /home/jegan/Downloads/MasteringC++Programming/Debugging/Ex1/a.out 
Breakpoint 1, main () at main.cpp:46
46 int main ( ) {
(gdb) l
41 ostream& operator << (ostream &o, const MyInteger& object) {
42    o << object.number;
43 }
44 
45 
46 
int main ( ) {
47 
48   vector<MyInteger> v = { 10, 100, 40, 20, 80, 70, 50, 30, 60, 90 };
49    
50   cout << "\nVectors entries before sorting are ..." << endl;
(gdb) n
48   vector<MyInteger> v = { 10, 100, 40, 20, 80, 70, 50, 30, 60, 90 };
(gdb) n
50   cout << "\nVectors entries before sorting are ..." << endl;
(gdb) n
Vectors entries before sorting are ...
51   copy ( v.begin(), v.end() , ostream_iterator<MyInteger>( cout, "\t" ) );
(gdb) break 29
Breakpoint 2 at 0x555555554f88: file main.cpp, line 29.
(gdb) break 33
Breakpoint 3 at 0x555555554b80: file main.cpp, line 33.
(gdb) 
```

从中，您将了解到断点可以通过函数名或行号添加。现在让程序继续执行，直到达到我们设置的断点之一：

```cpp
(gdb) break 29
Breakpoint 2 at 0x555555554f88: file main.cpp, line 29.
(gdb) break 33
Breakpoint 3 at 0x555555554b80: file main.cpp, line 33.
(gdb) continue Continuing.
Breakpoint 2, MyInteger::operator< (this=0x55555576bc24, rhsObject=...) at main.cpp:30 30 return this->number > rhsObject.number; (gdb) 
```

正如你所看到的，程序执行在行号`29`处暂停，因为每当`sort`函数需要决定是否交换两个项目以按升序排序向量条目时，它就会被调用。

让我们探索如何检查或打印变量`this->number`和`rhsObject.number`：

```cpp
(gdb) break 29
Breakpoint 2 at 0x400ec6: file main.cpp, line 29.
(gdb) break 33
Breakpoint 3 at 0x400af6: file main.cpp, line 33.
(gdb) continue
Continuing.
Breakpoint 2, MyInteger::operator< (this=0x617c24, rhsObject=...) at main.cpp:30
30 return this->number > rhsObject.number;
(gdb) print this->number $1 = 100 (gdb) print rhsObject.number $2 = 10 (gdb) 
```

您是否注意到`<`和`>`操作符的实现方式？该操作符检查*小于*操作，而实际的实现检查*大于*操作，并且`>`操作符重载方法中也观察到了类似的 bug。请检查以下代码：

```cpp
bool operator < ( const MyInteger &rhsObject ) {
        return this->number > rhsObject.number;
}

bool operator > ( const MyInteger &rhsObject ) {
        return this->number < rhsObject.number;
}
```

虽然`sort()`函数应该按升序对`vector`条目进行排序，但输出显示它是按降序对它们进行排序的，前面的代码是问题的根源。因此，让我们修复问题，如下所示：

```cpp
bool operator < ( const MyInteger &rhsObject ) {
        return this->number < rhsObject.number;
}

bool operator > ( const MyInteger &rhsObject ) {
        return this->number > rhsObject.number;
}
```

有了这些更改，让我们编译并运行程序：

```cpp
g++ main.cpp -std=c++17 -g

./a.out
```

这是您将获得的输出：

```cpp
Vectors entries before sorting are ...
10   100   40   20   80   70   50   30   60   90

Vectors entries after sorting are ...
10   20   30   40   50   60   70   80   90   100
```

很好，我们修复了 bug！毋庸置疑，您已经认识到了 GDB 调试工具的用处。虽然我们只是浅尝辄止了 GDB 工具的功能，但它提供了许多强大的调试功能。然而，在本章中，涵盖 GDB 工具支持的每一个功能是不切实际的；因此，我强烈建议您查阅 GDB 文档以进行进一步学习[`sourceware.org/gdb/documentation/`](https://sourceware.org/gdb/documentation/)。

# 使用 Valgrind 调试内存泄漏

Valgrind 是 Unix 和 Linux 平台的一组开源 C/C++调试和性能分析工具。Valgrind 支持的工具集如下：

+   **Cachegrind**：这是缓存分析器

+   **Callgrind**：这与缓存分析器类似，但支持调用者-被调用者序列

+   **Helgrind**：这有助于检测线程同步问题

+   **DRD**：这是线程错误检测器

+   **Massif**：这是堆分析器

+   **Lackey**：这提供了关于应用程序的基本性能统计和测量

+   **exp-sgcheck**：这检测堆栈越界；通常用于查找 Memcheck 无法找到的问题

+   **exp-bbv**：这对计算机架构研发工作很有用

+   **exp-dhat**：这是另一个堆分析器

+   **Memcheck**：这有助于检测内存泄漏和与内存问题相关的崩溃

在本章中，我们将只探讨 Memcheck，因为展示每个 Valgrind 工具不在本书的范围内。

# Memcheck 工具

Valgrind 使用的默认工具是 Memcheck。Memcheck 工具可以检测出相当详尽的问题列表，其中一些如下所示：

+   访问数组、堆栈或堆越界的边界外

+   未初始化内存的使用

+   访问已释放的内存

+   内存泄漏

+   `new`和`free`或`malloc`和`delete`的不匹配使用

让我们在接下来的小节中看一些这样的问题。

# 检测数组边界外的内存访问

以下示例演示了对数组边界外的内存访问：

```cpp
#include <iostream>
using namespace std;

int main ( ) {
    int a[10];

    a[10] = 100;
    cout << a[10] << endl;

    return 0;
}
```

以下输出显示了 Valgrind 调试会话，准确指出了数组边界外的内存访问：

```cpp
g++ arrayboundsoverrun.cpp -g -std=c++17 

jegan@ubuntu  ~/MasteringC++/Debugging  valgrind --track-origins=yes --read-var-info=yes ./a.out
==28576== Memcheck, a memory error detector
==28576== Copyright (C) 2002-2015, and GNU GPL'd, by Julian Seward et al.
==28576== Using Valgrind-3.11.0 and LibVEX; rerun with -h for copyright info
==28576== Command: ./a.out
==28576== 
100
*** stack smashing detected ***: ./a.out terminated
==28576== 
==28576== Process terminating with default action of signal 6 (SIGABRT)
==28576== at 0x51F1428: raise (raise.c:54)
==28576== by 0x51F3029: abort (abort.c:89)
==28576== by 0x52337E9: __libc_message (libc_fatal.c:175)
==28576== by 0x52D511B: __fortify_fail (fortify_fail.c:37)
==28576== by 0x52D50BF: __stack_chk_fail (stack_chk_fail.c:28)
==28576== by 0x4008D8: main (arrayboundsoverrun.cpp:11)
==28576== 
==28576== HEAP SUMMARY:
==28576== in use at exit: 72,704 bytes in 1 blocks
==28576== total heap usage: 2 allocs, 1 frees, 73,728 bytes allocated
==28576== 
==28576== LEAK SUMMARY:
==28576== definitely lost: 0 bytes in 0 blocks
==28576== indirectly lost: 0 bytes in 0 blocks
==28576== possibly lost: 0 bytes in 0 blocks
==28576== still reachable: 72,704 bytes in 1 blocks
==28576== suppressed: 0 bytes in 0 blocks
==28576== Rerun with --leak-check=full to see details of leaked memory
==28576== 
==28576== For counts of detected and suppressed errors, rerun with: -v
==28576== ERROR SUMMARY: 0 errors from 0 contexts (suppressed: 0 from 0)
[1] 28576 abort (core dumped) valgrind --track-origins=yes --read-var-info=yes ./a.out
```

正如您所注意到的，应用程序由于非法内存访问而崩溃并生成了核心转储。在前面的输出中，Valgrind 工具准确指出了导致崩溃的行。

# 检测对已释放内存位置的内存访问

以下示例代码演示了对已释放内存位置的内存访问：

```cpp
#include <iostream>
using namespace std;

int main( ) {

    int *ptr = new int();

    *ptr = 100;

    cout << "\nValue stored at pointer location is " << *ptr << endl;

    delete ptr;

    *ptr = 200;
    return 0;
}
```

让我们编译前面的程序并学习 Valgrind 如何报告试图访问已释放内存位置的非法内存访问：

```cpp
==118316== Memcheck, a memory error detector
==118316== Copyright (C) 2002-2015, and GNU GPL'd, by Julian Seward et al.
==118316== Using Valgrind-3.11.0 and LibVEX; rerun with -h for copyright info
==118316== Command: ./a.out
==118316== 

Value stored at pointer location is 100
==118316== Invalid write of size 4
==118316== at 0x400989: main (illegalaccess_to_released_memory.cpp:14)
==118316== Address 0x5ab6c80 is 0 bytes inside a block of size 4 free'd
==118316== at 0x4C2F24B: operator delete(void*) (in /usr/lib/valgrind/vgpreload_memcheck-amd64-linux.so)
==118316== by 0x400984: main (illegalaccess_to_released_memory.cpp:12)
==118316== Block was alloc'd at
==118316== at 0x4C2E0EF: operator new(unsigned long) (in /usr/lib/valgrind/vgpreload_memcheck-amd64-linux.so)
==118316== by 0x400938: main (illegalaccess_to_released_memory.cpp:6)
==118316== 
==118316== 
==118316== HEAP SUMMARY:
==118316== in use at exit: 72,704 bytes in 1 blocks
==118316== total heap usage: 3 allocs, 2 frees, 73,732 bytes allocated
==118316== 
==118316== LEAK SUMMARY:
==118316== definitely lost: 0 bytes in 0 blocks
==118316== indirectly lost: 0 bytes in 0 blocks
==118316== possibly lost: 0 bytes in 0 blocks
==118316== still reachable: 72,704 bytes in 1 blocks
==118316== suppressed: 0 bytes in 0 blocks
==118316== Rerun with --leak-check=full to see details of leaked memory
==118316== 
==118316== For counts of detected and suppressed errors, rerun with: -v
==118316== ERROR SUMMARY: 1 errors from 1 contexts (suppressed: 0 from 0)
```

Valgrind 准确指出了尝试访问在第`12`行释放的内存位置的行号。

# 检测未初始化内存访问

以下示例代码演示了未初始化内存访问的使用以及如何使用 Memcheck 检测相同的问题：

```cpp
#include <iostream>
using namespace std;

class MyClass {
    private:
       int x;
    public:
      MyClass( );
  void print( );
}; 

MyClass::MyClass() {
    cout << "\nMyClass constructor ..." << endl;
}

void MyClass::print( ) {
     cout << "\nValue of x is " << x << endl;
}

int main ( ) {

    MyClass obj;
    obj.print();
    return 0;

}
```

现在让我们编译并使用 Memcheck 检测未初始化内存访问问题：

```cpp
g++ main.cpp -g

valgrind ./a.out --track-origins=yes

==51504== Memcheck, a memory error detector
==51504== Copyright (C) 2002-2015, and GNU GPL'd, by Julian Seward et al.
==51504== Using Valgrind-3.11.0 and LibVEX; rerun with -h for copyright info
==51504== Command: ./a.out --track-origins=yes
==51504== 

MyClass constructor ...

==51504== Conditional jump or move depends on uninitialised value(s)
==51504== at 0x4F3CCAE: std::ostreambuf_iterator<char, std::char_traits<char> > std::num_put<char, std::ostreambuf_iterator<char, std::char_traits<char> > >::_M_insert_int<long>(std::ostreambuf_iterator<char, std::char_traits<char> >, std::ios_base&, char, long) const (in /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.21)
==51504== by 0x4F3CEDC: std::num_put<char, std::ostreambuf_iterator<char, std::char_traits<char> > >::do_put(std::ostreambuf_iterator<char, std::char_traits<char> >, std::ios_base&, char, long) const (in /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.21)
==51504== by 0x4F493F9: std::ostream& std::ostream::_M_insert<long>(long) (in /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.21)
==51504== by 0x40095D: MyClass::print() (uninitialized.cpp:19)
==51504== by 0x4009A1: main (uninitialized.cpp:26)
==51504== 
==51504== Use of uninitialised value of size 8
==51504== at 0x4F3BB13: ??? (in /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.21)
==51504== by 0x4F3CCD9: std::ostreambuf_iterator<char, std::char_traits<char> > std::num_put<char, std::ostreambuf_iterator<char, std::char_traits<char> > >::_M_insert_int<long>(std::ostreambuf_iterator<char, std::char_traits<char> >, std::ios_base&, char, long) const (in /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.21)
==51504== by 0x4F3CEDC: std::num_put<char, std::ostreambuf_iterator<char, std::char_traits<char> > >::do_put(std::ostreambuf_iterator<char, std::char_traits<char> >, std::ios_base&, char, long) const (in /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.21)
==51504== by 0x4F493F9: std::ostream& std::ostream::_M_insert<long>(long) (in /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.21)
==51504== by 0x40095D: MyClass::print() (uninitialized.cpp:19)
==51504== by 0x4009A1: main (uninitialized.cpp:26)
==51504== 
==51504== Conditional jump or move depends on uninitialised value(s)
==51504== at 0x4F3BB1F: ??? (in /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.21)
==51504== by 0x4F3CCD9: std::ostreambuf_iterator<char, std::char_traits<char> > std::num_put<char, std::ostreambuf_iterator<char, std::char_traits<char> > >::_M_insert_int<long>(std::ostreambuf_iterator<char, std::char_traits<char> >, std::ios_base&, char, long) const (in /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.21)
==51504== by 0x4F3CEDC: std::num_put<char, std::ostreambuf_iterator<char, std::char_traits<char> > >::do_put(std::ostreambuf_iterator<char, std::char_traits<char> >, std::ios_base&, char, long) const (in /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.21)
==51504== by 0x4F493F9: std::ostream& std::ostream::_M_insert<long>(long) (in /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.21)
==51504== by 0x40095D: MyClass::print() (uninitialized.cpp:19)
==51504== by 0x4009A1: main (uninitialized.cpp:26)
==51504== 
==51504== Conditional jump or move depends on uninitialised value(s)
==51504== at 0x4F3CD0C: std::ostreambuf_iterator<char, std::char_traits<char> > std::num_put<char, std::ostreambuf_iterator<char, std::char_traits<char> > >::_M_insert_int<long>(std::ostreambuf_iterator<char, std::char_traits<char> >, std::ios_base&, char, long) const (in /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.21)
==51504== by 0x4F3CEDC: std::num_put<char, std::ostreambuf_iterator<char, std::char_traits<char> > >::do_put(std::ostreambuf_iterator<char, std::char_traits<char> >, std::ios_base&, char, long) const (in /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.21)
==51504== by 0x4F493F9: std::ostream& std::ostream::_M_insert<long>(long) (in /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.21)
==51504== by 0x40095D: MyClass::print() (uninitialized.cpp:19)
==51504== by 0x4009A1: main (uninitialized.cpp:26)
==51504== 
Value of x is -16778960
==51504== 
==51504== HEAP SUMMARY:
==51504== in use at exit: 72,704 bytes in 1 blocks
==51504== total heap usage: 2 allocs, 1 frees, 73,728 bytes allocated
==51504== 
==51504== LEAK SUMMARY:
==51504== definitely lost: 0 bytes in 0 blocks
==51504== indirectly lost: 0 bytes in 0 blocks
==51504== possibly lost: 0 bytes in 0 blocks
==51504== still reachable: 72,704 bytes in 1 blocks
==51504== suppressed: 0 bytes in 0 blocks
==51504== Rerun with --leak-check=full to see details of leaked memory
==51504== 
==51504== For counts of detected and suppressed errors, rerun with: -v
==51504== Use --track-origins=yes to see where uninitialised values come from
==51504== ERROR SUMMARY: 18 errors from 4 contexts (suppressed: 0 from 0)

```

在前面的输出中，加粗显示的行清楚地指出了访问未初始化变量的确切行号（`14`）：

```cpp
==51504== by 0x40095D: MyClass::print() (uninitialized.cpp:19)
==51504== by 0x4009A1: main (uninitialized.cpp:26)

 18 void MyClass::print() {
 19 cout << "\nValue of x is " << x << endl;
 20 } 
```

上面的代码片段是供你参考的；然而，Valgrind 不会显示代码细节。底线是 Valgrind 精确指出了访问未初始化变量的行，这通常很难用其他方法检测到。

# 检测内存泄漏

让我们来看一个有一些内存泄漏的简单程序，并探索 Valgrind 工具如何在 Memcheck 的帮助下帮助我们检测内存泄漏。由于 Memcheck 是 Valgrind 默认使用的工具，因此在发出 Valgrind 命令时不需要显式调用 Memcheck 工具：

```cpp
valgrind application_debugged.exe --tool=memcheck
```

以下代码实现了一个单链表：

```cpp
#include <iostream>
using namespace std;

struct Node {
  int data;
  Node *next;
};

class List {
private:
  Node *pNewNode;
  Node *pHead;
  Node *pTail;
  int __size;
  void createNewNode( int );
public:
  List();
  ~List();
  int size();
  void append ( int data );
  void print( );
};
```

正如你可能已经观察到的，前面的类声明有`append()`一个新节点的方法，`print()`列表的方法，以及一个`size()`方法，返回列表中节点的数量。

让我们探索实现`append()`方法、`print()`方法、构造函数和析构函数的`list.cpp`源文件：

```cpp
#include "list.h"

List::List( ) {
  pNewNode = NULL;
  pHead = NULL;
  pTail = NULL;
  __size = 0;
}

List::~List() {}

void List::createNewNode( int data ) {
  pNewNode = new Node();
  pNewNode->next = NULL;
  pNewNode->data = data;
}

void List::append( int data ) {
  createNewNode( data );
  if ( pHead == NULL ) {
    pHead = pNewNode;
    pTail = pNewNode;
    __size = 1;
  }
  else {
    Node *pCurrentNode = pHead;
    while ( pCurrentNode != NULL ) {
      if ( pCurrentNode->next == NULL ) break;
      pCurrentNode = pCurrentNode->next;
    }

    pCurrentNode->next = pNewNode;
    ++__size;
  }
}

void List::print( ) {
  cout << "\nList entries are ..." << endl;
  Node *pCurrentNode = pHead;
  while ( pCurrentNode != NULL ) {
    cout << pCurrentNode->data << "\t";
    pCurrentNode = pCurrentNode->next;
  }
  cout << endl;
}
```

以下代码演示了`main()`函数：

```cpp
#include "list.h"

int main ( ) {
  List l;

  for (int count = 0; count < 5; ++count )
    l.append ( (count+1) * 10 );
  l.print();

  return 0;
}
```

让我们编译程序并尝试在前面的程序中检测内存泄漏：

```cpp
g++ main.cpp list.cpp -std=c++17 -g

valgrind ./a.out --leak-check=full 

==99789== Memcheck, a memory error detector
==99789== Copyright (C) 2002-2015, and GNU GPL'd, by Julian Seward et al.
==99789== Using Valgrind-3.11.0 and LibVEX; rerun with -h for copyright info
==99789== Command: ./a.out --leak-check=full
==99789== 

List constructor invoked ...

List entries are ...
10 20 30 40 50 
==99789== 
==99789== HEAP SUMMARY:
==99789== in use at exit: 72,784 bytes in 6 blocks
==99789== total heap usage: 7 allocs, 1 frees, 73,808 bytes allocated
==99789== 
==99789== LEAK SUMMARY:
==99789== definitely lost: 16 bytes in 1 blocks
==99789== indirectly lost: 64 bytes in 4 blocks
==99789== possibly lost: 0 bytes in 0 blocks
==99789== still reachable: 72,704 bytes in 1 blocks
==99789== suppressed: 0 bytes in 0 blocks
==99789== Rerun with --leak-check=full to see details of leaked memory
==99789== 
==99789== For counts of detected and suppressed errors, rerun with: -v
==99789== ERROR SUMMARY: 0 errors from 0 contexts (suppressed: 0 from 0)

```

从前面的输出可以看出，我们的应用泄漏了 80 字节。虽然`definitely lost`和`indirectly lost`表示我们的应用泄漏的内存，但`still reachable`并不一定表示我们的应用，它可能是由第三方库或 C++运行时库泄漏的。可能它们并不是真正的内存泄漏，因为 C++运行时库可能使用内存池。

# 修复内存泄漏

让我们尝试通过在`List::~List()`析构函数中添加以下代码来修复内存泄漏问题：

```cpp
List::~List( ) {

        cout << "\nList destructor invoked ..." << endl;
        Node *pTemp = NULL;

        while ( pHead != NULL ) {

                pTemp = pHead;
                pHead = pHead->next;

                delete pTemp;
        }

        pNewNode = pHead = pTail = pTemp = NULL;
        __size = 0;

}
```

从下面的输出中，你会发现内存泄漏已经被修复：

```cpp
g++ main.cpp list.cpp -std=c++17 -g

valgrind ./a.out --leak-check=full

==44813== Memcheck, a memory error detector
==44813== Copyright (C) 2002-2015, and GNU GPL'd, by Julian Seward et al.
==44813== Using Valgrind-3.11.0 and LibVEX; rerun with -h for copyright info
==44813== Command: ./a.out --leak-check=full
==44813== 

List constructor invoked ...

List entries are ...
10 20 30 40 50 
Memory utilised by the list is 80

List destructor invoked ...
==44813== 
==44813== HEAP SUMMARY:
==44813== in use at exit: 72,704 bytes in 1 blocks
==44813== total heap usage: 7 allocs, 6 frees, 73,808 bytes allocated
==44813== 
==44813== LEAK SUMMARY:
==44813== definitely lost: 0 bytes in 0 blocks
==44813== indirectly lost: 0 bytes in 0 blocks
==44813== possibly lost: 0 bytes in 0 blocks
==44813== still reachable: 72,704 bytes in 1 blocks
==44813== suppressed: 0 bytes in 0 blocks
==44813== Rerun with --leak-check=full to see details of leaked memory
==44813== 
==44813== For counts of detected and suppressed errors, rerun with: -v
==44813== ERROR SUMMARY: 0 errors from 0 contexts (suppressed: 0 from 0)

```

如果你仍然对前面输出中报告的`still reachable`问题不满意，让我们尝试在`simple.cpp`中尝试以下代码，以了解这是否在我们的控制之内：

```cpp
#include <iostream>
using namespace std;

int main ( ) {

    return 0;

} 
```

执行以下命令：

```cpp
g++ simple.cpp -std=c++17 -g

valgrind ./a.out --leak-check=full

==62474== Memcheck, a memory error detector
==62474== Copyright (C) 2002-2015, and GNU GPL'd, by Julian Seward et al.
==62474== Using Valgrind-3.11.0 and LibVEX; rerun with -h for copyright info
==62474== Command: ./a.out --leak-check=full
==62474== 
==62474== 
==62474== HEAP SUMMARY:
==62474== in use at exit: 72,704 bytes in 1 blocks
==62474== total heap usage: 1 allocs, 0 frees, 72,704 bytes allocated
==62474== 
==62474== LEAK SUMMARY:
==62474== definitely lost: 0 bytes in 0 blocks
==62474== indirectly lost: 0 bytes in 0 blocks
==62474== possibly lost: 0 bytes in 0 blocks
==62474== still reachable: 72,704 bytes in 1 blocks
==62474== suppressed: 0 bytes in 0 blocks
==62474== Rerun with --leak-check=full to see details of leaked memory
==62474== 
==62474== For counts of detected and suppressed errors, rerun with: -v
==62474== ERROR SUMMARY: 0 errors from 0 contexts (suppressed: 0 from 0)

```

正如你所看到的，`main()`函数除了返回`0`之外什么也没做，Valgrind 报告说这个程序也有相同的部分：`still reachable": 72, 704 bytes in 1 blocks`。因此，在`Valgrind`泄漏摘要中真正重要的是是否有泄漏报告在以下任何或所有部分：`definitely lost`，`indirectly lost`和`possibly lost`。

# new 和 free 或 malloc 和 delete 的不匹配使用

这种问题很少见，但不能排除它们发生的可能性。可能会出现这样的情况，当一个基于 C 的遗留工具被移植到 C++时，一些内存分配被错误地分配，但使用`delete`关键字或反之亦然被释放。

以下示例演示了使用 Valgrind 检测问题：

```cpp
#include <stdlib.h>

int main ( ) {

        int *ptr = new int();

        free (ptr); // The correct approach is delete ptr

        char *c = (char*)malloc ( sizeof(char) );

        delete c; // The correct approach is free ( c )

        return 0;
}
```

以下输出演示了一个 Valgrind 会话，检测到了`free`和`delete`的不匹配使用：

```cpp
g++ mismatchingnewandfree.cpp -g

valgrind ./a.out 
==76087== Memcheck, a memory error detector
==76087== Copyright (C) 2002-2015, and GNU GPL'd, by Julian Seward et al.
==76087== Using Valgrind-3.11.0 and LibVEX; rerun with -h for copyright info
==76087== Command: ./a.out
==76087== 
==76087== Mismatched free() / delete / delete []
==76087== at 0x4C2EDEB: free (in /usr/lib/valgrind/vgpreload_memcheck-amd64-linux.so)
==76087== by 0x4006FD: main (mismatchingnewandfree.cpp:7)
==76087== Address 0x5ab6c80 is 0 bytes inside a block of size 4 alloc'd
==76087== at 0x4C2E0EF: operator new(unsigned long) (in /usr/lib/valgrind/vgpreload_memcheck-amd64-linux.so)
==76087== by 0x4006E7: main (mismatchingnewandfree.cpp:5)
==76087== 
==76087== Mismatched free() / delete / delete []
==76087== at 0x4C2F24B: operator delete(void*) (in /usr/lib/valgrind/vgpreload_memcheck-amd64-linux.so)
==76087== by 0x400717: main (mismatchingnewandfree.cpp:11)
==76087== Address 0x5ab6cd0 is 0 bytes inside a block of size 1 alloc'd
==76087== at 0x4C2DB8F: malloc (in /usr/lib/valgrind/vgpreload_memcheck-amd64-linux.so)
==76087== by 0x400707: main (mismatchingnewandfree.cpp:9)
==76087== 
==76087== 
==76087== HEAP SUMMARY:
==76087== in use at exit: 72,704 bytes in 1 blocks
==76087== total heap usage: 3 allocs, 2 frees, 72,709 bytes allocated
==76087== 
==76087== LEAK SUMMARY:
==76087== definitely lost: 0 bytes in 0 blocks
==76087== indirectly lost: 0 bytes in 0 blocks
==76087== possibly lost: 0 bytes in 0 blocks
==76087== still reachable: 72,704 bytes in 1 blocks
==76087== suppressed: 0 bytes in 0 blocks
==76087== Rerun with --leak-check=full to see details of leaked memory
==76087== 
==76087== For counts of detected and suppressed errors, rerun with: -v
==76087== ERROR SUMMARY: 2 errors from 2 contexts (suppressed: 0 from 0)
```

# 总结

在本章中，你学习了各种 C++调试工具以及 Valgrind 工具的应用，比如检测未初始化的变量访问和检测内存泄漏。你还学习了 GDB 工具和检测由于非法内存访问已释放内存位置而引起的问题。

在下一章中，你将学习代码异味和清洁代码实践。
