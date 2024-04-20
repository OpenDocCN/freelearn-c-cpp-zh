# C++ Rx 编程的设计模式和成语

我们已经在使用 C++的响应式编程模型方面取得了相当大的进展。到目前为止，我们已经了解了 RxCpp 库及其编程模型、RxCpp 库的关键元素、响应式 GUI 编程以及编写自定义操作符的主题。现在，为了将问题提升到下一个级别，我们将涵盖一些设计模式和成语，这些模式和成语有助于我们进行高级软件开发任务。

在本章中，我们将涵盖以下主题：

+   模式和模式运动的介绍

+   GOF 设计模式和响应式编程

+   一些响应式编程模式和成语

# 面向对象编程和设计模式运动

在 90 年代初，面向对象编程（OOP）达到了临界点，当时 C++编程语言开始在 C 编程语言是主要编程语言的领域中取得进展。1992 年微软 C++编译器的出现，随后是微软基础类（MFC）库，使 C++编程成为了微软 Windows 下的主流。在 POSIX 世界中，C++ GUI 工具包如 WxWidgets 和 Qt，标志着面向对象编程的到来。面向对象编程运动的早期先驱者在各种杂志上写文章，如《Dr. Dobb's Journal》、《C++ Report》、《Microsoft Systems Journal》等，以传播他们的想法。

詹姆斯·科普利恩出版了一本有影响力的书，名为《高级 C++风格和成语》，讨论了与 C++编程语言的使用相关的低级模式（成语）。尽管它并不被广泛引用，但这本书的作者们认为它是一本记录面向对象编程最佳实践和技术的重要书籍。

埃里希·伽玛开始在他的博士论文中编写模式目录，从克里斯托弗·亚历山大的《城镇和建筑的模式》一书中获得灵感。在论文的过程中，有类似想法的人，即拉尔夫·约翰逊、约翰·弗利西德斯和理查德·赫尔姆，与埃里希·伽玛一起创建了一个包含 23 种设计模式的目录，现在被称为**四人帮**（**GOF**）设计模式。Addison Wesley 在 1994 年出版了基于他们工作的书籍《设计模式：可重用面向对象软件的元素》。这很快成为程序员的重要参考，并推动了面向模式的软件开发。GOF 目录主要集中在软件设计上，很快模式目录开始出现在建筑、企业应用集成、企业应用架构等领域。

1996 年，西门子的一群工程师出版了《面向模式的软件架构（POSA）》一书，主要关注系统建设的架构方面。整个 POSA 模式目录被记录在由约翰·威利和儿子出版的五本书中。在这两项倡议之后，出现了一大波活动。其他值得注意的模式目录如下

+   《企业应用架构模式》，作者马丁·福勒等。

+   《企业集成模式》，作者格雷戈尔·霍普和鲍比·沃尔夫。

+   《核心 J2EE 模式》，作者迪帕克·阿卢等。

+   《领域驱动设计》，作者埃里克·埃文斯。

+   《企业模式和 MDA》，作者吉姆·阿洛和伊拉·纽斯塔特。

尽管这些书在自己的领域内具有重要意义，但它们偏向于当时蓬勃发展的企业软件开发领域。对于 C++开发人员，GOF 目录和 POSA 目录是最重要的。

# 关键模式目录

模式是软件设计中常见问题的命名解决方案。模式通常被编入某种存储库。其中一些被出版成书。最受欢迎和广泛使用的模式目录是 GOF。

# GOF 目录

Gang of Four（GOF）以目录的创建者命名，开始了模式运动。创建者们主要关注面向对象软件的设计和架构。克里斯托弗·亚历山大的想法从建筑架构中借鉴并应用到软件工程中。很快，人们开始在应用架构、并发、安全等领域进行模式倡议。Gang Of Four 将目录分为结构、创建和行为模式。原始书籍使用 C++和 Smalltalk 来解释这些概念。这些模式已经被移植并在今天存在的大多数面向对象的编程语言中得到利用。下表列出了 GOF 目录中的模式。

| **序号** | **模式类型** | **模式** |
| --- | --- | --- |
| 1 | 创建模式 | 抽象工厂，生成器，工厂方法，原型，单例 |
| 2 | 结构模式 | 适配器，桥接，组合，装饰器，外观，享元，代理 |
| 3 | 行为模式 | 责任链，命令，解释器，迭代器，中介者，备忘录，观察者，状态，策略，模板方法，访问者 |

我们认为对 GOF 模式的深入理解对于任何程序员都是必要的。这些模式无论在应用领域如何，都随处可见。GOF 模式帮助我们以一种与语言无关的方式来交流和推理软件系统。它们在 C++、.NET 和 Java 世界中得到广泛实现。Qt 框架广泛利用了 GOF 存储库中的模式，为 C++编程语言提供了直观的编程模型，主要用于编写 GUI 应用程序。

# POSA 目录

*软件架构模式*（五卷）是一本有影响力的书系，涵盖了开发关键任务系统的大部分适用模式。该目录适合编写大型软件的关键子系统的人，特别是数据库引擎、分布式系统、中间件系统等。该目录的另一个优点是非常适合 C++程序员。

该目录共有五卷，值得独立研究。如果我们想要编写像 Web 服务器、协议服务器、数据库服务器等工业强度的中间件软件，这个目录非常方便。以下表格包含了一系列模式类型和相关模式

| **序号** | **模式类型** | **模式** |
| --- | --- | --- |
| 1 | 架构 | 层，管道和过滤器，黑板，经纪人，MVC，表示-抽象-控制，微内核，反射 |
| 2 | 设计 | 整体-部分，主从，代理，命令处理器，视图处理器，转发-接收器，客户端-调度器-服务器，发布者-订阅者 |
| 3 | 服务访问和配置模式 | 包装器外观，组件配置器，拦截器，扩展接口 |
| 4 | 事件处理模式 | 反应器，主动器，异步完成令牌，接收器-连接器 |
| 5 | 同步模式 | 作用域锁定，策略化锁定，线程安全接口，双重检查锁定优化 |
| 6 | 并发模式 | 主动对象，监视器对象，半同步/半异步，领导者/跟随者，线程特定存储 |
| 7 | 资源获取模式 | 查找，延迟获取，急切获取，部分获取 |
| 8 | 资源生命周期 | 缓存，池化，协调器，资源生命周期管理器 |
| 9 | 资源释放模式 | 租赁，驱逐者 |
| 10 | 分布式计算的模式语言 | 不是引入新模式，而是在分布式编程的上下文中对来自不同目录的模式进行整合 |
| 11 | 关于模式和模式语言 | 这最后一卷提供了有关模式、模式语言和使用的一些元信息 |

需要研究 POSA 目录，以深入了解部署在全球范围内的大型系统的架构基础。我们认为，尽管其重要性，这个目录并没有得到应有的关注。

# 设计模式重温

GOF 模式和响应式编程确实有比表面上显而易见的更深层次的联系。GOF 模式主要关注编写基于面向对象的软件。响应式编程是函数式编程、流编程和并发编程的结合。我们已经了解到，响应式编程纠正了经典的 GOF 观察者模式的一些缺陷（在第五章的第一节“可观察对象简介”中，我们涵盖了这个问题）。

编写面向对象的软件基本上是关于建模层次结构，从模式世界来看，组合模式是建模部分/整体层次结构的方法。无论何处有一个组合（用于建模结构），都会有一系列访问者模式的实现（用于建模行为）。访问者模式的主要目的是处理组合。换句话说，组合-访问者二元组是编写面向对象系统的规范模型。

访问者的实现应该对组合的结构具有一定的认识。使用访问者模式进行行为处理变得困难，因为给定组合的访问者数量不断增加。此外，向处理层添加转换和过滤进一步复杂化了问题。

引入迭代器模式，用于导航序列或项目列表。使用对象/函数式编程构造，我们可以非常容易地过滤和转换序列。微软的 LINQ 和 Java（8 及以上）中使用 lambda 处理集合类的例子都是迭代器模式的好例子。

那么，我们如何将层次数据转换为线性结构呢？大多数层次结构可以被展平为一个列表以进行进一步处理。最近，人们已经开始做以下事情：

+   使用组合模式对它们的层次进行建模。

+   使用专门用于此目的的访问者将层次结构展平为序列。

+   使用迭代器模式导航这些序列。

+   在执行操作之前，对序列应用一系列转换和过滤。

上述方法被称为“拉”式编程方法。消费者或客户端从事件或数据源中拉取数据进行处理。这种方案存在以下问题：

+   数据被不必要地拉入客户端。

+   转换和过滤应用在事件接收器（客户端）端。

+   事件接收器可以阻塞服务器。

+   这种风格不适合异步处理，其中数据随时间变化。

解决这个问题的一个好方法是逆向注视，即数据从服务器异步地作为流推送，事件接收器将对流做出反应。这种系统的另一个优点是在事件源端放置转换和过滤。这导致了一个场景，即只有绝对必要的数据需要在接收端进行处理。

方案如下：

+   数据被视为称为可观察对象的流。

+   我们可以对它们应用一系列操作符，或者更高级的操作符。

+   操作符总是接收一个可观察对象并返回另一个可观察对象。

+   我们可以订阅一个可观察对象以获取通知。

+   观察者有标准机制来处理它们。

在本节中，我们学习了面向对象编程模式和响应式编程是如何密切相关的。合理地混合这两种范式可以产生高质量、可维护的代码。我们还讨论了如何将面向对象编程设计模式（组合/访问者）转换（扁平化结构）以利用迭代器模式。我们讨论了如何通过轻微的改进（在事件源端使用一种忘记即可的习语）来改进迭代方案，从而得到可观察对象。在下一节中，我们将通过编写代码来演示整个技术。

# 从设计模式到响应式编程

尽管设计模式运动与面向对象编程相一致，而响应式编程则更倾向于函数式编程，但它们之间存在着密切的相似之处。在前一章（第五章，*可观察对象简介*）中，我们学到了以下内容：

+   面向对象编程模型适用于对系统的结构方面进行建模。

+   函数式编程模型适用于对系统的行为方面进行建模。

为了说明面向对象编程和响应式编程之间的联系，我们将编写一个程序，用于遍历目录以枚举给定文件夹中的文件和子文件夹。

我们将创建一个包含以下内容的组合结构：

+   一个继承自抽象类`EntryNode`的`FileNode`，用于模拟文件信息

+   一个继承自抽象类`EntryNode`的`DirectoryNode`，用于模拟文件夹信息

在定义了上述的组合后，我们将为以下内容定义访问者：

+   打印文件名和文件夹名

+   将组合层次结构转换为文件名列表

话不多说，让我们来看看这段代码：

```cpp
//---------- DirReact.cpp 
#include <rxcpp/rx.hpp> 
#include <memory> 
#include <map> 
#include <algorithm> 
#include <string> 
#include <vector> 
#include <windows.h> // This is omitted in POSIX version 
#include <functional> 
#include <thread> 
#include <future> 
using namespace std; 
//////////////////////////////////// 
//-------------- Forward Declarations 
//-------------- Model Folder/File 
class FileNode; 
class DirectoryNode; 
//////////////////////////////// 
//------------- The Visitor Interface 
class IFileFolderVisitor; 
```

上述的前向声明是为了在编译程序时消除编译器发出的错误和警告。`FileNode`存储文件名和文件大小作为实例变量。`DirectoryNode`存储文件夹名和`FileNode`列表，以表示目录中的文件和文件夹。`FileNode`/`DirectoryNode`层次结构由`IFileFolderVisitor`接口处理。现在，让我们为这些数据类型进行声明。

```cpp
///////////////////////////////// 
//------ a Type to store FileInformation 
struct FileInformation{ 
   string name; 
   long size; 
   FileInformation( string pname,long psize ) 
   { name = pname;size = psize; } 
}; 
////////////////////////////// 
//-------------- Base class for File/Folder data structure 
class EntryNode{ 
    protected: 
      string  name; 
      int isdir; 
      long size; 
    public: 
      virtual bool Isdir() = 0; 
      virtual long getSize() = 0; 
      virtual void Accept(IFileFolderVisitor& ivis)=0; 
      virtual ~EntryNode() {} 
};
```

当我们创建一个组合时，我们需要创建一个作为层次结构所有成员的基类的节点类。在我们的情况下，`EntryNode`类就是这样做的。我们在基类中存储文件或文件夹的名称、大小等。除了应该由派生类实现的三个虚拟函数之外，我们还有一个虚拟析构函数。虚拟析构函数的存在确保了适当地应用析构函数，以避免资源泄漏。现在，让我们看看下面给出的访问者基类声明。

```cpp
//-------------The Visitor Interface 
class IFileFolderVisitor{ 
   public: 
    virtual void Visit(FileNode& fn )=0; 
    virtual void Visit(DirectoryNode& dn )=0; 
}; 
```

每当我们使用组合模式风格的实现来定义层次结构时，我们会定义一个访问者接口来处理组合中的节点。对于组合中的每个节点，在访问者接口中都会有一个相应的`visit`方法。组合中类层次结构的每个节点都将有一个`accept`方法，在遍历组合时，访问者接口会将调用分派到相应节点的`accept`方法。`accept`方法将调用正确的访问者中的`visit`方法。这个过程被称为**双重分派**：

```cpp
// The Node which represents Files 
class FileNode : public EntryNode { 
   public:  
   FileNode(string pname, long psize) {  isdir = 0; name = pname; size = psize;} 
   ~FileNode() {cout << "....Destructor FileNode ...." << name << endl; } 
   virtual bool  Isdir() { return isdir == 1; } 
   string getname() { return name; }
   virtual long getSize() {return size; } 
   //------------- accept method 
   //------------- dispatches call to correct node in
   //------------- the Composite
   virtual void Accept( IFileFolderVisitor& ivis ){ivis.Visit(*this);} 
}; 
```

`FileNode`类只存储文件的名称和大小。该类还实现了基类（`EntryNode`）中声明的所有虚拟方法。`accept`方法将调用重定向到正确的访问者级别方法，如下所示：

```cpp
// Node which represents Directory 
class DirectoryNode : public EntryNode { 
  list<unique_ptr<EntryNode>> files;   
public: 
  DirectoryNode(string pname)  
  { files.clear(); isdir = 1; name = pname;} 
  ~DirectoryNode() {files.clear();} 
  list<unique_ptr<EntryNode>>& GetAllFiles() {return files;} 
  bool AddFile( string pname , long size) { 
       files.push_back(unique_ptr<EntryNode> (new FileNode(pname,size))); 
       return true; 
  } 
  bool AddDirectory( DirectoryNode *dn ) { 
        files.push_back(unique_ptr<EntryNode>(dn)); 
        return true; 
  } 
  bool Isdir() { return isdir == 1; } 
  string  getname() { return name; } 
  void   setname(string pname) { name = pname; } 
  long getSize() {return size; } 
  //
  //--------------------- accept method
  void Accept( IFileFolderVisitor& ivis ){ivis.Visit(*this); } 
}; 
```

`DirectoryNode` 类模拟了一个带有文件和子文件夹列表的文件夹。我们使用智能指针来存储条目。和往常一样，我们也实现了与 `EntryNode` 类相关的所有虚拟函数。`AddFile` 和 `AddDirectory` 方法用于填充列表。在使用特定于操作系统的函数遍历目录时，我们使用前面两种方法填充了 `DirectoryNode` 对象的内容。让我们看一下目录遍历辅助函数的原型。我们省略了源代码的完整列表（可在网上找到）。

```cpp
//------Directory Helper Has to be written for Each OS 
class DirHelper { 
 public: 
    static  DirectoryNode  *SearchDirectory(
            const std::string& refcstrRootDirectory){ 
           //--------------- Do some OS specific stuff to retrieve 
           //--------------- File/Folder hierarchy from the root folder 
           return DirNode; 
}}; 
```

`DirHelper` 逻辑在 Windows 和 GNU Linux/macOS X 之间有所不同。我们省略了书中实现的源代码。相关网站包含了前述类的完整源代码。基本上，该代码递归遍历目录以填充数据结构。现在，我们将转移到上面创建的 Composite 的遍历主题。以下代码展示了如何使用实现了 IFileFolderVisitor 接口的 Visitor 类来遍历 Composite。

```cpp
///////////////////////////////////// 
//----- A Visitor Interface that prints 
//----- The contents of a Folder 
class PrintFolderVisitor : public IFileFolderVisitor 
{ 
  public: 
    void Visit(FileNode& fn ) {cout << fn.getname() << endl; } 
    void Visit(DirectoryNode& dn ) { 
      cout << "In a directory " << dn.getname() << endl; 
      list<unique_ptr<EntryNode>>& ls = dn.GetAllFiles(); 
      for ( auto& itr : ls ) { itr.get()->Accept(*this);} 
    } 
}; 
```

`PrintFolderVisitor` 类是一个 Visitor 实现，用于在控制台上显示文件和文件夹信息。该类演示了如何为 Composite 实现一个基本的访问者。在我们的情况下，Composite 只有两个节点，编写访问者实现非常容易。

在某些情况下，层次结构中节点类型的数量很多，编写访问者实现并不容易。为访问者编写过滤器和转换可能很困难，逻辑是临时的。让我们编写一个程序来打印文件夹的内容。代码如下：

```cpp
//--------------- has used raw pointers 
//--------------- in a production implementation, use smart pointer
void TestVisitor( string directory ){ 
  // Search files including subdirectories 
  DirectoryNode *dirs = DirHelper::SearchDirectory(directory); 
  if ( dirs == 0 ) {return;} 
  PrintFolderVisitor *fs = new PrintFolderVisitor (); 
  dirs->Accept(*fs); delete fs; delete dirs; 
} 
```

上述函数递归遍历目录并创建一个 Composite（`DirectoryNode *`）。我们使用 `PrintFolderVisitor` 来打印文件夹的内容，如下所示：

```cpp
int main(int argc, char *argv[]) {  TestVisitor("D:\\Java"); }
```

# 将层次结构展平以便遍历

访问者实现必须对 Composite 的结构有一定的了解。在某些 Composite 实现中，需要实现大量的访问者。此外，在访问者接口的情况下，对节点应用转换和过滤有些困难。GOF 模式目录中有一个迭代器模式，可用于遍历一系列项。问题是：如何使用迭代器模式将层次结构线性化以进行处理？大多数层次结构可以通过编写用于此目的的访问者实现来展平为列表、序列或流。让我们为所述任务编写一个展平访问者。

看一下以下代码：

```cpp
// Flatten the File/Folders into a linear list 
class FlattenVisitor : public IFileFolderVisitor{ 
    list <FileInformation> files; 
    string CurrDir; 
 public: 
    FlattenVisitor() { CurrDir = "";} 
    ~FlattenVisitor() { files.clear();} 
    list<FileInformation> GetAllFiles() { return files; } 
    void Visit(FileNode& fn ) { 
       files.push_back( FileInformation{ 
                  CurrDir +"\" + fn.getname(),fn.getSize())); 
    } 
    void Visit(DirectoryNode& dn ) { 
        CurrDir = dn.getname(); 
        files.push_back( FileInformation( CurrDir, 0 )); 
        list<unique_ptr<EntryNode>>& ls = dn.GetAllFiles(); 
        for ( auto& itr : ls ) { itr.get()->Accept(*this);} 
    } 
}; 
```

`FlattenVisitor` 类在 STL 列表中收集文件和文件夹。对于每个目录，我们遍历文件列表并使用熟悉的双重分发调用 `accept` 方法。让我们编写一个函数，返回一个 `FileInformation` 列表供我们遍历。代码如下：

```cpp
list<FileInformation> GetAllFiles(string dirname ){ 
   list<FileInformation> ret_val; 
   // Search files including subdirectories 
   DirectoryNode *dirs = DirHelper::SearchDirectory(dirname); 
   if ( dirs == 0 ) {return ret_val;} 
   //--  We have used Raw pointers here...
   //--- In Modern C++, one can use smart pointer here
   //  unique_ptr<FlattenVisitor> fs(new FlattenVisitor());
   //  We can avoid delete fs
   FlattenVisitor *fs = new FlattenVisitor(); 
   dirs->Accept(*fs); 
   ret_val = fs->GetAllFiles(); 
   //--------- use of Raw pointer 
   delete fs; delete dirs; 
   return ret_val; 
} 
int main(int argc, char *argv[]) { 
  list<FileInformation> rs = GetAllFiles("D:\JAVA"); 
  for( auto& as : rs ) 
    cout << as.name << endl; 
} 
```

`FlattenVisitor` 类遍历 `DirectoryNode` 层次结构，并将完全展开的路径名收集到 STL 列表容器中。一旦我们将层次结构展平为列表，就可以对其进行迭代。

我们已经学会了如何将层次结构建模为 Composite，并最终将其展平为适合使用迭代器模式进行导航的形式。在下一节中，我们将学习如何将迭代器转换为可观察对象。我们将使用 RxCpp 来实现可观察对象，通过使用一种推送值从事件源到事件接收端的“发射并忘记”模型。

# 从迭代器到可观察对象

迭代器模式是从 STL 容器、生成器和流中拉取数据的标准机制。它们非常适合在空间中聚合的数据。基本上，这意味着我们预先知道应该检索多少数据，或者数据已经被捕获。有些情况下，数据是异步到达的，消费者不知道有多少数据或数据何时到达。在这种情况下，迭代器需要等待，或者我们需要采用超时策略来处理这种情况。在这种情况下，基于推送的方法似乎是更好的选择。使用 Rx 的 Subject 构造，我们可以使用 fire and forget 策略。让我们编写一个类，发出目录的内容，如下所示：

```cpp
////////////////////////////// 
// A Toy implementation of Active  
// Object Pattern... Will be explained as a separate pattern
template <class T> 
struct ActiveObject { 
    rxcpp::subjects::subject<T> subj; 
    // fire-and-forget 
    void FireNForget(T & item){subj.get_subscriber().on_next(item);} 
    rxcpp::observable<T> GetObservable()  
    { return subj.get_observable(); } 
    ActiveObject(){}  
    ~ActiveObject() {} 
}; 
/////////////////////// 
// The class uses a FireNForget mechanism to  
// push data to the Data/Event sink 
// 
class DirectoryEmitter { 
      string rootdir; 
      //-------------- Active Object ( a Pattern in it's own right ) 
      ActiveObject<FileInformation> act; // more on this below  
  public: 
      DirectoryEmitter(string s )   { 
         rootdir = s; 
         //----- Subscribe  
         act.GetObservable().subscribe([] ( FileInformation item ) { 
            cout << item.name << ":" << item.size << endl; 
         }); 
      } 
      bool Trigger() { 
           std::packaged_task<int()> task([&]() {  EmitDirEntry(); return 1; }); 
           std::future<int> result = task.get_future(); 
           task(); 
           //------------ Comment the below lineto return immediately 
           double dresult = result.get(); 
           return true; 
      } 
      //----- Iterate over the list of files  
      //----- uses ActiveObject Pattern to do FirenForget 
      bool EmitDirEntry() { 
           list<FileInformation> rs = GetAllFiles(rootdir); 
           for( auto& a : rs ) { act.FireNForget(a); } 
           return false; 
      } 
}; 
int main(int argc, char *argv[]) { 
  DirectoryEmitter emitter("D:\\JAVA"); 
  emitter.Trigger(); return 0; 
} 
```

`DirectoryEmitter`类使用现代 C++的`packaged_task`构造以 fire and forget 的方式进行异步调用。在前面的列表中，我们正在等待结果（使用`std::future<T>`）。我们可以在上面的代码列表中注释一行（参见列表中的内联注释），以立即返回。

# Cell 模式

我们已经学到，响应式编程是关于处理随时间变化的值。响应式编程模型以 Observable 的概念为中心。Observable 有两种变体，如下所示：

+   单元：单元是一个实体（变量或内存位置），其值随时间定期更新。在某些情境中，它们也被称为属性或行为。

+   流：流代表一系列事件。它们通常与动作相关的数据。当人们想到 Observable 时，他们脑海中有 Observable 的流变体。

我们将实现一个 Cell 编程模式的玩具版本。我们只专注于实现基本功能。该代码需要整理以供生产使用。

以下的实现可以进行优化，如果我们正在实现一个名为 Cell controller 的控制器类。然后，Cell controller 类（包含所有单元的单个 Rx Subject）可以从所有单元（到一个中央位置）接收通知，并通过评估表达式来更新依赖关系。在这里，我们已经为每个单元附加了 Subject。这个实现展示了 Cell 模式是一个可行的依赖计算机制：

```cpp
//------------------ CellPattern.cpp 
#include <rxcpp/rx.hpp> 
#include <memory> 
#include <map> 
#include <algorithm> 
using namespace std; 
class Cell 
{ 
  private: 
    std::string name; 
    std::map<std::string,Cell *> parents; 
    rxcpp::subjects::behavior<double> *behsubject;   
  public: 
    string get_name() { return name;} 
    void SetValue(double v )  
    { behsubject->get_subscriber().on_next(v);} 
    double GetValue()  
    { return behsubject->get_value(); } 
    rxcpp::observable<double> GetObservable()  
    { return behsubject->get_observable(); } 
    Cell(std::string pname) { 
       name = pname; 
       behsubject = new rxcpp::subjects::behavior<double>(0); 
    } 
    ~Cell() {delete behsubject; parents.clear();} 
    bool GetCellNames( string& a , string& b ) 
    { 
         if ( parents.size() !=2 ) { return false; } 
         int i = 0; 
         for(auto p  : parents ) { 
            ( i == 0 )? a = p.first : b = p.first; 
            i++;      
         } 
         return true; 
    } 
    ///////////////////////////// 
    // We will just add two parent cells... 
    // in real life, we need to implement an  
    // expression evaluator 
    bool Recalculate() { 
        string as , bs ; 
        if (!GetCellNames(as,bs) ) { return false; } 
        auto a = parents[as]; 
        auto b = parents[bs]; 
        SetValue( a->GetValue() + b->GetValue() ); 
        return true; 
    } 
    bool Attach( Cell& s ) { 
       if ( parents.size() >= 2 ) { return false; } 
       parents.insert(pair<std::string,Cell *>(s.get_name(),&s)); 
       s.GetObservable().subscribe( [=] (double a ) { Recalculate() ;}); 
       return true; 
    } 
    bool Detach( Cell& s ) { //--- Not Implemented  
    } }; 
```

Cell 类假设每个单元有两个父依赖关系（为了简化实现），每当父级的值发生变化时，单元的值将被重新计算。我们只实现了加法运算符（为了保持列表的简洁）。`recalculate`方法实现了上面显示的逻辑：让我们编写一个主程序把所有东西放在一起。

```cpp
int main(int argc, char *argv[]) {     
    Cell a("a");  
    Cell b("b"); 
    Cell c("c"); 
    Cell d("d"); 
    Cell e("e"); 
    //-------- attach a to c 
    //-------- attach b to c 
    //-------- c is a + b  
    c.Attach(a); 
    c.Attach(b); 
    //---------- attach c to e 
    //---------- attach d to e 
    //---------- e is c + d or e is a + b + d; 
    e.Attach(c); 
    e.Attach(d); 
    a.SetValue(100);  // should print 100 
    cout << "Value is " << c.GetValue() << endl; 
    b.SetValue(200);  // should print 300 
    cout << "Value is " << c.GetValue() << endl; 
    b.SetValue(300);  // should print 400 
    cout << "Value is " << c.GetValue() << endl; 
    d.SetValue(-400); // should be Zero 
    cout << "Value is " << e.GetValue() << endl; 
} 
```

主程序演示了如何使用 Cell 模式将更改传播到依赖项中。通过更改 cless 中的值，我们强制重新计算依赖单元中的值。

# Active 对象模式

Active 对象是一个将方法调用和方法执行分离的类，非常适合于 fire and forget 的异步调用。附加到类的调度程序处理执行请求。该模式由六个元素组成，如下所示：

+   代理，为客户端提供具有公开可访问方法的接口

+   定义 Active 对象上的方法请求的接口

+   来自客户端的待处理请求列表

+   决定下一个要执行的请求的调度程序

+   Active 对象方法的实现

+   客户端接收结果的回调或变量

我们将剖析 Active 对象模式的实现。这个程序是为了阐明而编写的；在生产中使用，我们需要使用更复杂的方法。尝试生产质量的实现会使代码变得相当长。让我们看一下代码：

```cpp
#include <rxcpp/rx.hpp> 
#include <memory> 
#include <map> 
#include <algorithm> 
#include <string> 
#include <vector> 
#include <windows.h> 
#include <functional> 
#include <thread> 
#include <future> 
using namespace std; 
//------- Active Object Pattern Implementation 
template <class T> 
class ActiveObject { 
    //----------- Dispatcher Object 
    rxcpp::subjects::subject<T> subj; 
    protected: 
    ActiveObject(){ 
       subj.get_observable().subscribe([=] (T s ) 
       { Execute(s); }); 
    }  
    virtual void Execute(T s) {} 
    public: 
    // fire-and-forget 
    void FireNForget(T item){ subj.get_subscriber().on_next(item);} 
    rxcpp::observable<T> GetObservable() { return subj.get_observable(); } 
    virtual ~ActiveObject() {} 
}; 
```

前面的实现声明了一个`subject<T>`类的实例，作为通知机制。`FireNForget`方法通过调用`get_subscriber`方法将值放入 subject 中。该方法立即返回，订阅方法将检索该值并调用`Execute`方法。该类应该被具体实现所重写。让我们来看一下代码：

```cpp
class ConcreteObject : public ActiveObject<double> { 
    public: 
     ConcreteObject() {} 
     virtual void Execute(double a ) 
     { cout << "Hello World....." << a << endl;} 
}; 
int main(int argc, char *argv[]) { 
  ConcreteObject temp; 
  for( int i=0; i<=10; ++i ) 
      temp.FireNForget(i*i); 
  return 0; 
}
```

前面的代码片段调用了`FireNForget`方法，传入了一个双精度值。在控制台上，我们可以看到该值被显示出来。重写的`Execute`方法会自动被调用。

# 资源借贷模式

借贷模式，正如其名字所示，将资源借给一个函数。在下面给出的示例中，文件句柄被借给了类的消费者。它执行以下步骤：

1.  它创建一个可以使用的资源（文件句柄）

1.  它将资源（文件句柄）借给将使用它的函数（lambda）

1.  这个函数由调用者传递并由资源持有者执行

1.  资源（文件句柄）由资源持有者关闭或销毁

以下代码实现了资源管理的资源借贷模式。该模式有助于在编写代码时避免资源泄漏：

```cpp
//----------- ResourceLoan.cpp 
#include <rxcpp/rx.hpp> 
using namespace std; 
////////////////////////// 
// implementation of Resource Loan  Pattern. The Implementation opens a file 
// and does not pass the file handle to user  defined Lambda. The Ownership remains with 
// the class  
class ResourceLoan { 
   FILE *file;  // This is the resource which is being loaned
   string filename; 
  public: 
     ResourceLoan(string pfile) { 
        filename = pfile; 
        //---------- Create the resource
        file = fopen(filename.c_str(),"rb"); 
     }   
     //////////////////////////// 
     // Read upto 1024 bytes to a buffer  
     // return the buffer contents and number of bytes 
     int ReadBuffer( std::function<int(char pbuffer[],int val )> func ) 
     { 
          if (file == nullptr ) { return -1; } 
          char buffer[1024]; 
          int result = fread (buffer,1,1024,file); 
          return func(buffer,result); 
     }  
     //---------- close the resource 
     ~ResourceLoan() { fclose(file);} 
}; 
//////////////////////////////// 
// A Sample Program to invoke the preceding 
// class 
// 
int main(int argc, char *argv[]) { 
  ResourceLoan res("a.bin"); 
  int nread ; 
  //------------- The conents of the buffer 
  //------------- and size of buffer is stored in val 
  auto rlambda =  [] (char buffer[] , int val ) { 
       cout <<  "Size " << val << endl; 
       return val; 
  }; 
  //------- The File Handle is not available to the  
  //------- User defined Lambda It has been loaned to the 
  //-------- consumer of the class
  while ((nread = res.ReadBuffer(rlambda)) > 0) {} 
  //---- When the ResourceLoan object goes out of scope 
  //---- File Handle is closed 
  return 0; 
} 
```

资源借贷模式适用于避免资源泄漏。资源的持有者从不直接将资源的句柄或指针交给其消费者。主程序演示了我们如何消费该实现。ResourceLoan 类从不允许其消费者直接访问文件句柄。

# 事件总线模式

事件总线充当事件源和事件接收器之间的中介。事件源或生产者向总线发出事件，订阅事件的类（消费者）将收到通知。该模式可以是中介者设计模式的一个实例。在事件总线实现中，我们有以下原型

+   **生产者**：产生事件的类

+   **消费者**：消费事件的类

+   **控制器**：充当生产者和消费者的类

在接下来的实现中，我们省略了控制器的实现。以下代码实现了事件总线的一个玩具版本：

```cpp
//----------- EventBus.cpp 
#include <rxcpp/rx.hpp> 
#include <memory> 
#include <map> 
#include <algorithm> 
using namespace std; 
//---------- Event Information 
struct EVENT_INFO{ 
   int id; 
   int err_code; 
   string description; 
   EVENT_INFO() { id = err_code = 0 ; description ="default";} 
   EVENT_INFO(int pid,int perr_code,string pdescription ) 
   { id = pid; err_code = perr_code; description = pdescription; } 
   void Print() { 
      cout << "id & Error Code" << id << ":" << err_code << ":"; 
      cout << description << endl; 
   } 
}; 
```

`EVENT_INFO`结构模拟了一个事件，它包含以下内容：

+   `Id`：事件 ID

+   `err_code`：错误代码

+   `description`：事件描述

其余的代码相当明显；在这里是：

```cpp
//----------- The following method 
//----------- will be invoked by  
//----------- Consumers 
template <class T> 
void DoSomeThingWithEvent( T ev ) 
{ev.Print();} 

//---------- Forward Declarations  
template <class T> 
class EventBus; 
//------------- Event Producer 
//------------- Just Inserts event to a Bus 
template <class T> 
class Producer { 
  string name; 
 public: 
   Producer(string pname ) { name = pname;} 
   bool Fire(T ev,EventBus<T> *bev ) { 
         bev->FireEvent(ev); 
         return false; 
   } 
}; 
```

生产者类的实现相当简单。骨架实现相当琐碎。`Fire`方法以兼容的`EventBus<T>`作为参数，并调用`EventBus<T>`类的`FireEvent`方法。生产实现需要一些花里胡哨的东西。让我们来看一下消费者类的代码。

```cpp
//------------ Event Consumer 
//------------ Subscribes to a Subject 
//------------ to Retrieve Events 
template <class T> 
class Consumer { 
  string name; 
  //--------- The subscription member helps us to 
  //--------- Unsubscribe to an Observable  
  rxcpp::composite_subscription subscription; 
public: 
  Consumer(string pname) { name = pname;} 
  //--------- Connect a Consumer to a Event Bus 
  bool Connect( EventBus<T> *bus ) { 
      //------ If already subscribed, Unsubscribe! 
      if ( subscription.is_subscribed() ) 
             subscription.unsubscribe(); 
      //------- Create a new Subscription 
      //------- We will call DoSomeThingWithEvent method 
      //------- from Lambda function 
      subscription = rxcpp::composite_subscription(); 
      auto subscriber = rxcpp::make_subscriber<T>( 
        subscription,={ 
            DoSomeThingWithEvent<T>(value); 
        },[](){ printf("OnCompletedn");}); 
      //----------- Subscribe! 
      bus->GetObservable().subscribe(subscriber); 
      return true; 
  } 
  //-------- DTOR ....Unsubscribe 
  ~Consumer() { Disconnect(); } 
  bool Disconnect() { 
       if (subscription.is_subscribed() ) 
        subscription.unsubscribe(); 
  } 
}; 
```

`Consumer<T>`的功能非常明显。`Connect`方法负责订阅`EventBus<T>`类中 Subject 的 Observable 端。每当有新的连接请求时，现有的订阅将被取消订阅，如下所示：

```cpp
//--- The implementation of the EventBus class 
//--- We have not taken care of Concurrency issues 
//--- as our purpose is to demonstrate the pattern 
template <class T> 
class EventBus 
{ 
  private: 
    std::string name; 
    //----- Reference to the Subject... 
    //----- Consumers get notification by  
    //----- Subscribing to the Observable side of the subject 
    rxcpp::subjects::behavior<T> *replaysubject;  
  public: 
    EventBus<T>() {replaysubject = new rxcpp::subjects::behavior<T>(T());} 
    ~EventBus() {delete replaysubject;} 
    //------ Add a Consumer to the Bus... 
    bool AddConsumer( Consumer<T>& b ) {b.Connect(this);} 
    //------ Fire the Event... 
    bool FireEvent ( T& event ) { 
       replaysubject->get_subscriber().on_next(event); 
       return true; 
    } 
    string get_name() { return name;} 
    rxcpp::observable<T> GetObservable()  
    { return replaysubject->get_observable(); } 
}; 
```

`EventBus<T>`类充当生产者和消费者之间的导管。我们在底层使用`replaysubject`来通知消费者。现在，我们已经完成了生产者和消费者类的编写，让我们看看如何利用上面编写的代码。

```cpp
///////////////////// 
//The EntryPoint 
// 
// 
int main(int argc, char *argv[]) { 
    //---- Create an instance of the EventBus 
    EventBus<EVENT_INFO> program_bus; 
    //---- Create a Producer and Two Consumers 
    //---- Add Consumers to the EventBus 
    Producer<EVENT_INFO> producer_one("first"); 
    Consumer<EVENT_INFO> consumer_one("one"); 
    Consumer<EVENT_INFO> consumer_two("two"); 
    program_bus.AddConsumer(consumer_one); 
    program_bus.AddConsumer(consumer_two); 
    //---- Fire an Event... 
    EVENT_INFO ev; 
    ev.id = 100; 
    ev.err_code = 0; 
    ev.description = "Hello World.."; 
    producer_one.Fire(ev,&program_bus); 
    //---- fire another by creating a second  
    //---- Producer 
    ev.id = 100; 
    ev.err_code = 10; 
    ev.description = "Error Happened.."; 
    Producer<EVENT_INFO> producer_two("second"); 
    producer_two.Fire(ev,&program_bus); 
} 
```

在主函数中，我们正在执行以下任务：

1.  创建`EventBus<T>`的实例

1.  创建生产者的实例

1.  创建消费者的实例

1.  向总线分发事件

我们只涵盖了适用于编写响应式程序的设计模式的子集。主要，我们的重点是从 GOF 设计模式过渡到响应式编程世界。事实上，本书的作者认为响应式编程模型是经典 GOF 设计模式的增强实现。这种增强是由于现代编程语言中添加的函数式编程构造。事实上，对象/函数式编程是编写现代 C++代码的良好方法。本章在很大程度上是基于这个想法。

# 总结

在本章中，我们深入探讨了与 C++编程和响应式编程相关的设计模式/习惯用法的美妙世界。从 GOF 设计模式开始，我们转向了响应式编程模式，逐渐过渡从面向对象编程到响应式编程是本章的亮点。之后，我们涵盖了诸如 Cell、Active object、Resource loan 和 Event bus 等响应式编程模式。从 GOF 模式过渡到响应式编程有助于你以更广泛的视角看待响应式编程。在下一章中，我们将学习使用 C++进行微服务开发。
