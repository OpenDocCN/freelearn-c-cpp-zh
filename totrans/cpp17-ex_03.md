# 构建图书馆管理系统

在本章中，我们研究了一个图书馆管理系统。我们继续开发 C++类，就像前几章一样。然而，在本章中，我们开发了一个更贴近现实世界的系统。本章的图书馆系统可以被真实的图书馆使用。

图书馆由书籍和客户的集合组成。书籍跟踪哪些客户借阅或预订了它们。客户跟踪他们借阅和预订了哪些书籍。

主要思想是图书馆包含一组书籍和一组客户。每本书都标记为已借出或未借出。如果已借出，则存储借阅该书的客户的身份号码。此外，一本书也可以被一个或多个客户预订。因此，每本书还包含一个已预订该书的客户身份号码列表。它必须是一个列表而不是集合，因为书籍应按照客户预订的顺序借出。

每个客户持有两个集合，包含他们借阅和预订的书籍的身份号码。在这两种情况下，我们使用集合而不是列表，因为它们借阅或预订书籍的顺序并不重要。

在本章中，我们将涵盖以下主题：

+   处理书籍和客户类，这些类构成了一个小型数据库，以整数作为键。

+   使用标准输入和输出流处理，其中我们写入有关书籍和客户的信息，并提示用户输入。

+   使用文件处理和流。书籍和客户使用标准 C++文件流写入和读取。

+   最后，我们使用 C++标准库中的泛型类`set`和`list`。

# `Book`类

我们有三个类：`Book`、`Customer`和`Library`：

+   `Book`类跟踪一本书。每本书都有一个作者、一个标题和一个唯一的身份号码。

+   `Customer`类跟踪一个客户。每个客户都有一个姓名、一个地址和一个唯一的身份号码。

+   `Library`类跟踪图书馆操作，例如添加和删除书籍和客户、借阅、归还和预订书籍，以及列出书籍和客户。

+   `main`函数简单地创建一个`Library`类的对象。

此外，每本书还记录了它是否被借阅的信息。如果被借阅，则存储借阅该书的客户的身份号码。每本书还包含一个预订列表。同样，每个客户还包含他们当前借阅和预订的书籍集合。

`Book`类有两个构造函数。第一个构造函数是一个默认构造函数，用于从文件中读取书籍。第二个构造函数用于向图书馆添加新书。它接受书籍的作者名和标题作为参数。

**Book.h**

```cpp
class Book { 
  public: 
    Book(void); 
    Book(const string& author, const string& title); 
```

`author`和`title`方法简单地返回书籍的作者和标题：

```cpp
    const string& author(void) const { return m_author; } 
    const string& title(void) const { return m_title; } 
```

图书馆的书籍可以读取和写入文件：

```cpp
    void read(ifstream& inStream); 
    void write(ofstream& outStream) const; 
```

一本书可以被借出、预订或归还。预订也可以被删除。请注意，当书籍被借出或预订时，我们需要提供顾客的身份证号。然而，在归还书籍时，这不必要，因为 `Book` 类跟踪当前借出书籍的顾客：

```cpp
    void borrowBook(int customerId); 
    int reserveBook(int customerId); 
    void unreserveBookation(int customerId); 
    void returnBook(); 
```

当书籍被借出时，顾客的身份证号被存储，并由 `bookId` 返回：

```cpp
    int bookId(void) const { return m_bookId; } 
```

`borrowed` 方法返回 true，如果此时书籍已被借出。在这种情况下，`customerId` 返回借出书籍的顾客的身份证号：

```cpp
    bool borrowed(void) const { return m_borrowed; } 
    int customerId(void) const { return m_customerId; } 
```

一本书可以被一组顾客预订，并且 `reservationList` 返回该列表：

```cpp
    list<int>& reservationList(void) { return m_reservationList; } 
```

`MaxBookId` 字段是静态的，这意味着它是类中所有对象的公共属性：

```cpp
    static int MaxBookId; 
```

输出流操作符写入书籍的信息：

```cpp
    friend ostream& operator<<(ostream& outStream, 
                               const Book& book); 
```

当书籍被借出时，`m_borrowed` 字段为 true。书籍和潜在借阅者的身份信息存储在 `m_bookId` 和 `m_customerId` 中：

```cpp
    private: 
      bool m_borrowed = false; 
      int m_bookId, m_customerId; 
```

作者的名字和书的标题存储在 `m_author` 和 `m_title` 中：

```cpp
      string m_author, m_title; 
```

多个顾客可以预订一本书。当他们这样做时，他们的身份信息被存储在 `m_reservationList` 中。它是一个列表而不是集合，因为预订是按顺序存储的。当一本书被归还时，下一个按预订顺序的顾客借阅这本书：

```cpp
      list<int> m_reservationList; 
      }; 
```

在本章中，我们使用 C++ 标准库中的泛型 `set`、`map` 和 `list` 类。它们的规范存储在 `Set`、`Map` 和 `List` 头文件中。`set` 和 `list` 类包含与上一章中我们的集合和列表类类似的集合和列表。一个映射是一个结构，其中每个值都通过一个唯一的键来标识，以便提供快速访问。

**Book.cpp**

```cpp
    #include <Set> 
    #include <Map> 
    #include <List> 
    #include <String> 
    #include <FStream> 
    using namespace std; 

    #include "Book.h" 
    #include "Customer.h" 
    #include "Library.h" 
```

由于 `MaxBookId` 是静态的，我们使用双冒号 (`::`) 符号初始化它。每个静态字段都需要在类定义之外初始化：

```cpp
    int Book::MaxBookId = 0; 
```

默认构造函数不执行任何操作。它在从文件读取时使用。尽管如此，我们仍然必须有一个默认构造函数来创建 `Book` 类的对象：

```cpp
    Book::Book(void) { 
      // Empty. 
    } 
```

当创建新书时，它被分配一个唯一的身份号码。这个身份号码存储在 `MaxBookId` 中，每次创建新的 `Book` 对象时都会增加：

```cpp
    Book::Book(const string& author, const string& title) 
     :m_bookId(++MaxBookId), 
      m_author(author), 
      m_title(title) { 
      // Empty. 
    } 
```

# 写入书籍

以类似的方式将书籍写入流。但是，我们使用 `write` 而不是 `read`。它们的工作方式类似：

```cpp
    void Book::write(ofstream& outStream) const { 
      outStream.write((char*) &m_bookId, sizeof m_bookId); 
```

当读取字符串时，我们使用 `getline` 而不是流操作符，因为流操作符只读取一个单词，而 `getline` 读取多个单词。然而，在写入流时，我们可以使用流操作符。名字和标题是由一个或多个单词组成无关紧要：

```cpp
    outStream << m_author << endl; 
    outStream << m_title << endl; 

    outStream.write((char*) &m_borrowed, sizeof m_borrowed); 
    outStream.write((char*) &m_customerId, sizeof m_customerId); 
```

与这里的读取情况类似，我们首先在列表中写入预订的数量。然后写入预订身份本身：

```cpp
    { int reservationListSize = m_reservationList.size(); 
      outStream.write((char*) &reservationListSize, 
                    sizeof reservationListSize); 

      for (int customerId : m_reservationList) { 
        outStream.write((char*) &customerId, sizeof customerId); 
      } 
    } 
    } 
```

# 读取书籍

当从文件中读取任何类型的值（除了字符串）时，我们使用 `read` 方法，该方法读取固定数量的字节。`sizeof` 操作符给我们 `m_bookId` 字段的大小（以字节为单位）。`sizeof` 操作符也可以用来查找类型的大小。例如，`sizeof (int)` 给出 `int` 类型值的字节大小。类型必须用括号括起来：

```cpp
    void Book::read(ifstream& inStream) { 
      inStream.read((char*) &m_bookId, sizeof m_bookId); 
```

当从文件中读取字符串值时，我们使用 C++ 标准函数 `getline` 来读取作者的名字和书籍的标题。如果名字由多个单词组成，使用输入流操作符将不起作用。如果作者或标题由多个单词组成，则只会读取第一个单词。其余的单词将不会被读取：

```cpp
    getline(inStream, m_author);
    getline(inStream, m_title);
```

注意，我们甚至使用 `read` 方法来读取 `m_borrowed` 字段的值，尽管它持有的是 `bool` 类型而不是 `int` 类型：

```cpp
    inStream.read((char*) &m_borrowed, sizeof m_borrowed);
    inStream.read((char*) &m_customerId, sizeof m_customerId);
```

在读取预订列表时，我们首先读取列表中的预订数量。然后读取预订的身份证号码：

```cpp
    { int reservationListSize;
      inStream.read((char*) &reservationListSize,
                  sizeof reservationListSize);
      for (int count = 0; count < reservationListSize; ++count) {
        int customerId;
        inStream.read((char*) &customerId, sizeof customerId);
        m_reservationList.push_back(customerId);
      }
    }
  } 
```

# 借阅和预订书籍

当书籍被借出时，`m_borrowed` 变为 `true`，并且 `m_customerId` 被设置为借出书籍的顾客的身份证号：

```cpp
void Book::borrowBook(int customerId) { 
  m_borrowed = true; 
  m_customerId = customerId; 
} 
```

当书籍被预订时，情况略有不同。虽然一本书只能被一位顾客借阅，但它可以被多位顾客预订。顾客的身份证号被添加到 `m_reservationList` 中。列表的大小被返回给调用者，以便他们知道自己在预订列表中的位置：

```cpp
    int Book::reserveBook(int customerId) { 
      m_reservationList.push_back(customerId); 
      return m_reservationList.size(); 
    } 
```

当书籍归还时，我们只需将 `m_borrowed` 设置为 `false`。我们不需要将 `m_customerId` 设置为任何特定的值。只要书籍没有被借出，这与它无关：

```cpp
    void Book::returnBook() { 
      m_borrowed = false; 
    } 
```

顾客可以自己从预订列表中移除。在这种情况下，我们在 `m_reservationList` 上调用 `remove`：

```cpp
    void Book::unreserveBookation(int customerId) { 
      m_reservationList.remove(customerId); 
    } 
```

# 显示书籍

输出流操作符写入书籍的标题和作者。如果书籍被借出，则写入顾客的姓名，如果预订列表已满，则写入预订顾客的姓名：

```cpp
    ostream& operator<<(ostream& outStream, const Book& book) { 
      outStream << """ << book.m_title << "" by " << book.m_author; 
```

当访问静态字段时，我们使用双冒号表示法（`::`），例如 `Library` 中的 `s_customerMap`：

```cpp
  if (book.m_borrowed) { 
    outStream << endl << "  Borrowed by: " 
              << Library::s_customerMap[book.m_customerId].name() 
              << "."; 
  } 

  if (!book.m_reservationList.empty()) { 
    outStream << endl << "  Reserved by: "; 

    bool first = true; 
    for (int customerId : book.m_reservationList) { 
      outStream << (first ? "" : ",") 
                << Library::s_customerMap[customerId].name(); 
      first = false; 
    } 

    outStream << "."; 
  } 

  return outStream; 
} 
```

# `Customer` 类

`Customer` 类跟踪顾客信息。它持有顾客当前借阅和预订的书籍集合。

**Customer.h**

```cpp
class Customer { 
  public: 
    Customer(void); 
    Customer(const string& name, const string& address); 

    void read(ifstream& inStream); 
    void write(ofstream& outStream) const; 

    void borrowBook(int bookId); 
    void reserveBook(int bookId); 
    void returnBook(int bookId); 
    void unreserveBook(int bookId); 
```

`hasBorrowed` 方法返回 `true`，如果顾客此时至少借阅了一本书。在下一节的 `Library` 类中，无法移除当前借阅书籍的顾客：

```cpp
    bool hasBorrowed(void) const { return !m_loanSet.empty(); } 

    const string& name(void) const {return m_name;} 
    const string& address(void) const {return m_address;} 
    int id(void) const {return m_customerId;} 
```

与之前使用的 `Book` 类类似，我们使用静态字段 `MaxCustomerId` 来计数顾客的身份证号。我们还使用输出流操作符来写入有关顾客的信息：

```cpp
    static int MaxCustomerId; 
    friend ostream& operator<<(ostream& outStream, 
                               const Customer& customer); 
```

每个客户都有一个姓名、地址和唯一的身份号码。集合`m_loanSet`和`m_reservationSet`保存了客户当前借阅和预订的书籍的身份号码。请注意，我们使用集合而不是列表，因为借阅和预订的书籍的顺序并不重要：

```cpp
    private: 
      int m_customerId; 
      string m_name, m_address; 
      set<int> m_loanSet, m_reservationSet; 
  }; 
```

**Customer.cpp**

```cpp
    #include <Set> 
    #include <Map> 
    #include <List> 
    #include <String> 
    #include <FStream> 
    using namespace std; 

    #include "Book.h" 
    #include "Customer.h" 
    #include "Library.h" 
```

由于`MaxCustomerId`是一个静态字段，它需要在类外部定义：

```cpp
    int Customer::MaxCustomerId; 
```

默认构造函数用于仅从文件中加载对象。因此，不需要初始化字段：

```cpp
    Customer::Customer(void) { 
      // Empty. 
    } 
```

第二个构造函数用于创建新的书籍对象。我们使用`MaxCustomerId`字段来初始化客户身份号码；我们还初始化他们的`name`和`address`：

```cpp
    Customer::Customer(const string& name, const string& address) 
     :m_customerId(++MaxCustomerId), 
      m_name(name), 
      m_address(address) { 
      // Empty. 
    } 
```

# 从文件中读取客户信息

`read`方法从文件流中读取客户信息：

```cpp
    void Customer::read(ifstream& inStream) { 
     inStream.read((char*) &m_customerId, sizeof m_customerId); 
```

与`Book`类的`read`方法相同，我们必须使用`getline`函数而不是输入流运算符，因为输入流运算符只会读取一个单词：

```cpp
    getline(inStream, m_name); 
    getline(inStream, m_address); 

    { int loanSetSize; 
      inStream.read((char*) &loanSetSize, sizeof loanSetSize); 

      for (int count = 0; count < loanSetSize; ++count) { 
        int bookId; 
        inStream.read((char*) &bookId, sizeof bookId); 
        m_loanSet.insert(bookId); 
      } 
    } 

    { int reservationListSize; 
      inStream.read((char*) &reservationListSize, 
                  sizeof reservationListSize); 

      for (int count = 0; count < reservationListSize; ++count) { 
        int bookId; 
        inStream.read((char*) &bookId, sizeof bookId); 
        m_loanSet.insert(bookId); 
      } 
    } 
  } 
```

# 将客户信息写入文件

`write`方法以与之前`Book`类中相同的方式将客户信息写入流中：

```cpp
    void Customer::write(ofstream& outStream) const { 
      outStream.write((char*) &m_customerId, sizeof m_customerId); 
      outStream << m_name << endl; 
      outStream << m_address << endl; 
```

当写入集合时，我们首先写入集合的大小，然后是集合的各个值：

```cpp
    { int loanSetSize = m_loanSet.size(); 
      outStream.write((char*) &loanSetSize, sizeof loanSetSize); 

      for (int bookId : m_loanSet) { 
        outStream.write((char*) &bookId, sizeof bookId); 
      } 
    } 

    { int reservationListSize = m_reservationSet.size(); 
      outStream.write((char*) &reservationListSize, 
                    sizeof reservationListSize); 

      for (int bookId : m_reservationSet) { 
        outStream.write((char*) &bookId, sizeof bookId); 
      } 
    } 
  } 
```

# 借阅和预订书籍

当客户借阅书籍时，它被插入到客户的借阅集中：

```cpp
    void Customer::borrowBook(int bookId) { 
      m_loanSet.insert(bookId); 
    } 
```

同样，当客户预订书籍时，它被插入到客户的预订集中：

```cpp
    void Customer::reserveBook(int bookId) { 
      m_reservationSet.insert(bookId); 
    } 
```

当客户归还或取消预订书籍时，它将从借阅集或预订集中删除：

```cpp
    void Customer::returnBook(int bookId) { 
      m_loanSet.erase(bookId); 
    } 

    void Customer::unreserveBook(int bookId) { 
      m_reservationSet.erase(bookId); 
    } 
```

# 显示客户

输出流运算符写入客户的姓名和地址。如果客户借阅或预订了书籍，它们也会被写入：

```cpp
    ostream& operator<<(ostream& outStream, const Customer& customer){ 
      outStream << customer.m_customerId << ". " << customer.m_name 
                << ", " << customer.m_address << "."; 

      if (!customer.m_loanSet.empty()) { 
        outStream << endl << "  Borrowed books: "; 

        bool first = true; 
        for (int bookId : customer.m_loanSet) { 
          outStream << (first ? "" : ",") 
                    << Library::s_bookMap[bookId].author(); 
          first = false; 
        } 
     } 

      if (!customer.m_reservationSet.empty()) { 
        outStream << endl << "  Reserved books: "; 

        bool first = true; 
        for (int bookId : customer.m_reservationSet) { 
          outStream << (first ? "" : ",") 
                    << Library::s_bookMap[bookId].title(); 
          first = false; 
        } 
      } 

      return outStream; 
    }   
```

# `Library`类

最后，`Library`类处理图书馆本身。它执行一系列关于借阅和归还书籍的任务。

**Library.h**

```cpp
class Library { 
  public: 
    Library(); 

  private: 
    static string s_binaryPath; 
```

`lookupBook`方法通过作者和标题查找书籍。如果找到书籍，则返回 true。如果找到，其信息（`Book`类的对象）将被复制到由`bookPtr`指向的对象中：

```cpp
    bool lookupBook(const string& author, const string& title, 
                    Book* bookPtr = nullptr); 
```

同样，`lookupCustomer`通过姓名和地址查找客户。如果找到客户，则返回 true，并将信息复制到由`customerPtr`指向的对象中：

```cpp
    bool lookupCustomer(const string& name, const string& address, 
                        Customer* customerPtr = nullptr); 
```

本章的应用围绕以下方法展开。它们执行图书馆系统的任务。每个方法都会提示用户输入，然后执行一项任务，例如借阅或归还书籍。

以下方法各自执行一项任务，包括查找书籍或客户的信息、添加或删除书籍、列出书籍、从图书馆添加和删除书籍以及借阅、预订和归还书籍：

```cpp
    void addBook(void); 
    void deleteBook(void); 
    void listBooks(void); 
    void addCustomer(void); 
    void deleteCustomer(void); 
    void listCustomers(void); 
    void borrowBook(void); 
    void reserveBook(void); 
    void returnBook(void); 
```

`load`和`save`方法在执行开始和结束时被调用：

```cpp
    void load();  
    void save(); 
```

有两个映射分别存储图书馆的书籍和客户。如前所述，映射是一种结构，其中每个值都通过一个唯一键来标识，以便提供快速访问。书籍和客户的唯一身份号码是键：

```cpp
  public: 
    static map<int,Book> s_bookMap; 
    static map<int,Customer> s_customerMap; 
}; 
```

**Library.cpp**

```cpp
#include <Set> 
#include <Map> 
#include <List> 
#include <String> 
#include <FStream> 
#include <IOStream> 
#include <Algorithm> 
using namespace std; 

#include "Book.h" 
#include "Customer.h" 
#include "Library.h" 

map<int,Book> Library::s_bookMap; 
map<int,Customer> Library::s_customerMap; 
```

在两次执行之间，图书馆信息存储在硬盘上的`Library.bin`文件中。注意，我们在`string`中使用两个反斜杠来表示一个反斜杠。第一个反斜杠表示该字符是一个特殊字符，第二个反斜杠表示它是一个反斜杠：

```cpp
string Library::s_binaryPath("Library.bin"); 
```

构造函数加载图书馆，显示菜单，并迭代直到用户退出。在执行完成之前，保存图书馆：

```cpp
Library::Library(void) { 
```

在显示菜单之前，从文件中加载图书馆信息（书籍、客户、贷款和预订）：

```cpp
  load(); 
```

当`quit`为真时，while 语句会继续执行。它保持为假，直到用户从菜单中选择退出选项：

```cpp
  bool quit = false; 
  while (!quit) { 
    cout << "1\. Add Book" << endl 
         << "2\. Delete Book" << endl 
         << "3\. List Books" << endl 
         << "4\. Add Customer" << endl 
         << "5\. Delete Customer" << endl 
         << "6\. List Customers" << endl 
         << "7\. Borrow Book" << endl 
         << "8\. Reserve Book" << endl 
         << "9\. Return Book" << endl 
         << "0\. Quit" << endl 
         << ": "; 
```

用户从控制台输入流（`cin`）中输入一个整数值，该值存储在`choice`变量中：

```cpp
    int choice; 
    cin >> choice; 
```

我们使用`switch`语句来执行请求的任务：

```cpp
    switch (choice) { 
      case 1: 
        addBook(); 
        break; 

      case 2: 
        deleteBook(); 
        break; 

      case 3: 
        listBooks(); 
        break; 

      case 4: 
        addCustomer(); 
        break; 

      case 5: 
        deleteCustomer(); 
        break; 

      case 6: 
        listCustomers(); 
        break; 

      case 7: 
        borrowBook(); 
        break; 

      case 8: 
        reserveBook(); 
        break; 

      case 9: 
        returnBook(); 
        break; 

      case 0: 
        quit = true; 
        break; 
    } 

    cout << endl; 
  } 
```

在程序完成之前，保存图书馆信息：

```cpp
      save(); 
    } 
```

# 查找书籍和客户

`lookupBook`方法遍历书籍映射。如果存在具有作者和标题的书籍，则返回 true。如果书籍存在，其信息被复制到由`bookPtr`参数指向的对象中，并且只要指针不为空，就返回 true。如果书籍不存在，则返回 false，并且不会将信息复制到对象中：

```cpp
    bool Library::lookupBook(const string& author, 
        const string& title, Book* bookPtr /* = nullptr*/) { 
      for (const pair<int,Book>& entry : s_bookMap) { 
        const Book& book = entry.second; 
```

注意，`bookPtr`可能为`nullptr`。在这种情况下，只返回 true，并且不会将信息写入`bookPtr`所指向的对象：

```cpp
    if ((book.author() == author) && (book.title() == title)) { 
      if (bookPtr != nullptr) { 
        *bookPtr = book; 
      } 

      return true; 
    } 
  } 

  return false; 
} 
```

同样，`lookupCustomer`遍历客户映射，如果存在具有相同名称的客户，则返回 true，并将客户信息复制到`Customer`对象中：

```cpp
    bool Library::lookupCustomer(const string& name, 
       const string& address, Customer* customerPtr /*=nullptr*/){ 
      for (const pair<int,Customer>& entry : s_customerMap) { 
        const Customer& customer = entry.second; 
```

此外，在这种情况下，`customerPtr`可能为`nullptr`。在这种情况下，只返回 true。当添加新客户时，我们希望知道是否已经存在具有相同名称和地址的客户：

```cpp
    if ((customer.name() == name) && 
        (customer.address() == address)) { 
      if (customerPtr != nullptr) { 
        *customerPtr = customer; 
      } 

      return true; 
    } 
  } 

  return false; 
} 
```

# 添加书籍

`addBook`方法提示用户输入新书的名称和标题：

```cpp
    void Library::addBook(void) { 
      string author; 
      cout << "Author: "; 
      cin >> author; 

      string title; 
      cout << "Title: "; 
      cin >> title; 
```

如果存在具有相同`author`和`title`的书籍，将显示错误消息：

```cpp
  if (lookupBook(author, title)) { 
    cout << endl << "The book "" <<  title << "" by " 
         << author << " already exists." << endl; 
    return; 
  } 
```

如果书籍尚未存在，我们创建一个新的`Book`对象并将其添加到书籍映射中：

```cpp
  Book book(author, title); 
  s_bookMap[book.bookId()] = book; 
  cout << endl << "Added: " << book << endl; 
} 
```

# 删除书籍

`deleteBook`方法提示用户输入书籍的作者和标题，如果存在则删除：

```cpp
    void Library::deleteBook() { 
      string author; 
      cout << "Author: "; 
      cin >> author; 

      string title; 
      cout << "Title: "; 
      cin >> title; 
```

如果书籍不存在，将显示错误消息：

```cpp
  Book book; 
  if (!lookupBook(author, title, &book)) { 
    cout << endl << "There is no book "" << title << "" by " 
         << "author " << author << "." << endl; 
    return; 
  } 
```

当删除书籍时，我们遍历所有客户，并对每个客户返回并取消预订书籍。我们为每本书都这样做，以防书籍已被客户借阅或预订。在下一章中，我们将使用指针，这允许我们更有效地返回和取消预订书籍。

注意，当我们遍历映射并获取每个 `Customer` 对象时，在修改了其字段值之后，我们需要将其放回映射中：

```cpp
    for (pair<int,Customer> entry : s_customerMap) { 
      Customer& customer = entry.second; 
      customer.returnBook(book.bookId()); 
      customer.unreserveBook(book.bookId()); 
      s_customerMap[customer.id()] = customer; 
    } 
```

最后，当我们确认书籍存在，并且已经归还和取消预订后，我们将它从书籍映射中删除：

```cpp
    s_bookMap.erase(book.bookId()); 
    cout << endl << "Deleted." << endl; 
  } 
```

# 列出书籍

`listBook` 方法相当简单。首先，我们检查书籍映射是否为空。如果为空，我们写入 `"No books."`。如果书籍映射不为空，我们遍历它，并且对于每本书，我们将其信息写入控制台输出流 (`cout`)：

```cpp
    void Library::listBooks(void) { 
      if (s_bookMap.empty()) { 
        cout << "No books." << endl; 
        return; 
      } 

      for (const pair<int,Book>& entry : s_bookMap) { 
        const Book& book = entry.second; 
        cout << book << endl; 
      } 
    } 
```

# 添加客户

`addCustomer` 方法提示用户输入新客户的 `name` 和 `address`：

```cpp
    void Library::addCustomer(void) { 
      string name; 
      cout << "Name: "; 
      cin >> name; 

      string address; 
      cout << "Address: "; 
      cin >> address; 
```

如果存在具有相同 `name` 和 `address` 的客户，将显示错误信息：

```cpp
    if (lookupCustomer(name, address)) { 
      cout << endl << "A customer with name " << name 
           << " and address " << address << " already exists." 
           << endl; 
      return; 
    } 
```

最后，我们创建一个新的 `Customer` 对象并将其添加到客户映射中：

```cpp
    Customer customer(name, address); 
    s_customerMap[customer.id()] = customer; 
    cout << endl << "Added." << endl; 
  } 
```

# 删除客户

`deleteCustomer` 方法如果客户存在，则删除客户：

```cpp
    void Library::deleteCustomer(void) { 
      string name; 
      cout << "Name: "; 
      cin >> name; 

      string address; 
      cout << "Address: "; 
      cin >> address; 

      Customer customer; 
      if (!lookupCustomer(name, address, &customer)) { 
        cout << endl << "There is no customer with name " << name 
             << " and address " << address << "." << endl; 
        return; 
      } 
```

如果客户至少借阅了一本书，在客户被删除之前必须归还：

```cpp
  if (customer.hasBorrowed()) { 
    cout << "Customer " << name << " has borrowed at least " 
         << "one book and cannot be deleted." << endl; 
    return; 
  } 
```

然而，如果客户已经预订了书籍，我们在删除客户之前先取消预订：

```cpp
  for (pair<int,Book> entry : s_bookMap) { 
    Book& book = entry.second; 
    book.unreserveBookation(customer.id()); 
    s_bookMap[book.bookId()] = book; 
  } 

  cout << endl << "Deleted." << endl; 
  s_customerMap.erase(customer.id()); 
} 
```

# 列出客户

`listCustomer` 方法的工作方式与 `listBooks` 类似。如果没有客户，我们写入 `"No Customers."`。如果有客户，我们将它们写入控制台输出流 (`cout`)：

```cpp
    void Library::listCustomers(void) { 
      if (s_customerMap.empty()) { 
        cout << "No customers." << endl; 
        return; 
      } 

      for (const pair<int,Customer>& entry : s_customerMap) { 
        const Customer& customer = entry.second; 
        cout << customer << endl; 
      } 
    } 
```

# 借阅书籍

`borrowBook` 方法提示用户输入书籍的 `author` 和 `title`：

```cpp
    void Library::borrowBook(void) { 
      string author; 
      cout << "Author: "; 
      cin >> author; 

      string title; 
      cout << "Title: "; 
      cin >> title; 
```

如果不存在具有 `author` 和 `title` 的书籍，将显示错误信息：

```cpp
    Book book; 
    if (!lookupBook(author, title, &book)) { 
      cout << endl << "There is no book "" << title << "" by " 
           << "author " << author << "." << endl; 
      return; 
    } 
```

此外，如果 `book` 已经被借出，将显示错误信息：

```cpp
    if (book.borrowed()) { 
      cout << endl << "The book "" << title << "" by " << author 
           << " has already been borrowed." << endl; 
      return; 
    } 
```

然后我们提示用户输入客户的 `name` 和 `address`：

```cpp
  string name; 
  cout << "Customer name: "; 
  cin >> name; 

  string address; 
  cout << "Adddress: "; 
  cin >> address; 
```

如果没有具有 `name` 和 `address` 的 `customer`，将显示错误信息：

```cpp
    Customer customer; 
    if (!lookupCustomer(name, address, &customer)) { 
      cout << endl << "There is no customer with name " << name 
           << " and address " << address << "." << endl; 
      return; 
    } 
```

然而，如果书籍存在且尚未被借出，并且客户存在，我们将书籍添加到客户的借阅集合中，并标记书籍为客户借阅：

```cpp
    book.borrowBook(customer.id()); 
    customer.borrowBook(book.bookId()); 
```

注意，在修改了 `Book` 和 `Customer` 对象之后，我们必须将它们放回它们的映射中。在下一章中，我们将使用更直接的方式来处理指针：

```cpp
    s_bookMap[book.bookId()] = book; 
    s_customerMap[customer.id()] = customer; 
    cout << endl << "Borrowed." << endl; 
  } 
```

# 预订书籍

`reserveBook` 方法的工作方式与 `borrowBook` 相同。它提示用户输入书籍的 `author` 和 `title`：

```cpp
    void Library::reserveBook(void) { 
      string author; 
      cout << "Author: "; 
      cin >> author; 

      string title; 
      cout << "Title: "; 
      cin >> title; 
```

与 `borrowBook` 的情况类似，我们检查具有 `author` 和 `title` 的书籍是否存在：

```cpp
  Book book; 
  if (!lookupBook(author, title, &book)) { 
    cout << endl << "There is no book "" << title << "" by " 
         << "author " << author << "." << endl; 
    return; 
  } 
```

然而，与 `borrowBook` 相比的一个区别是，书籍必须已被借出才能被预订。如果没有被借出，就没有预订的必要。在这种情况下，用户应该借阅该书籍：

```cpp
  if (!book.borrowed()) { 
    cout << endl << "The book with author " << author 
         << " and title "" << title << "" has not been " 
         << "borrowed. Please borrow the book instead." << endl; 
    return; 
  } 
```

如果书籍存在且未被借出，我们提示用户输入客户的 `name` 和 `address`：

```cpp
  string name; 
  cout << "Customer name: "; 
  cin >> name; 

  string address; 
  cout << "Address: "; 
  cin >> address; 
```

如果客户不存在，将显示错误信息：

```cpp
  Customer customer; 
  if (!lookupCustomer(name, address, &customer)) { 
    cout << endl << "No customer with name " << name 
         << " and address " << address << " exists." << endl; 
    return; 
  } 
```

此外，如果客户已经借阅了书籍，我们将显示错误信息：

```cpp
    if (book.customerId() == customer.id()) { 
      cout << endl << "The book has already been borrowed by " 
           << name << "." << endl; 
      return; 
    } 
```

如果书籍存在且已被借出，但不是由该客户借出，我们将客户添加到书籍的预订列表中，并将书籍添加到客户的预订集中：

```cpp
    customer.reserveBook(book.bookId()); 
    int position = book.reserveBook(customer.id()); 
```

此外，在这种情况下，我们必须将 `Book` 和 `Customer` 对象放回它们的映射中：

```cpp
    s_bookMap[book.bookId()] = book; 
    s_customerMap[customer.id()] = customer; 
```

最后，我们写入客户在预订列表中的位置：

```cpp
    cout << endl << position << "nd reserve." << endl; 
  } 
```

# 返回书籍

`returnBook` 方法提示用户输入书籍的作者和标题：

```cpp
    void Library::returnBook(void) { 
      string author; 
      cout << "Author: "; 
      cin >> author; 

      string title; 
      cout << "Title: "; 
      cin >> title; 
```

如果书籍不存在，将显示错误信息：

```cpp
    Book book; 
    if (!lookupBook(author, title, &book)) { 
      cout << endl << "No book "" << title 
           << "" by " << author << " exists." << endl; 
      return; 
    } 
```

如果书籍未被借出，将显示错误信息：

```cpp
    if (!book.borrowed()) { 
      cout << endl << "The book "" << title 
           << "" by " << author 
           << "" has not been borrowed." << endl; 
      return; 
    } 
```

与之前描述的方法不同，在这种情况下，我们不询问客户。相反，我们返回书籍，并在每位客户的预订列表中查找该书籍：

```cpp
    book.returnBook(); 
    cout << endl << "Returned." << endl; 

    Customer customer = s_customerMap[book.customerId()]; 
    customer.returnBook(book.bookId()); 
    s_customerMap[customer.id()] = customer; 
```

如果书籍已被预订，我们查找预订列表中的第一位客户，将其从预订列表中删除，并允许他们借阅书籍：

```cpp
    list<int>& reservationList = book.reservationList(); 

    if (!reservationList.empty()) { 
      int newCustomerId = reservationList.front(); 
      reservationList.erase(reservationList.begin()); 
      book.borrowBook(newCustomerId); 

      Customer newCustomer = s_customerMap[newCustomerId]; 
      newCustomer.borrowBook(book.bookId()); 

      s_customerMap[newCustomerId] = newCustomer; 
      cout << endl << "Borrowed by " << newCustomer.name() << endl; 
    } 

    s_bookMap[book.bookId()] = book; 
  } 
```

# 将图书馆信息保存到文件中

当保存图书馆信息时，我们首先打开文件：

```cpp
void Library::save() { 
  ofstream outStream(s_binaryPath); 
```

如果文件正确打开，首先我们写入书籍的数量，然后通过在 `Book` 对象上调用 `write` 方法来写入每本书的信息：

```cpp
  if (outStream) { 
    int numberOfBooks = s_bookMap.size(); 
    outStream.write((char*) &numberOfBooks, sizeof numberOfBooks); 

    for (const pair<int,Book>& entry : s_bookMap) { 
      const Book& book = entry.second; 
      book.write(outStream); 
    } 
```

同样，我们通过调用 `write` 方法写入客户数量，然后写入每位客户的信息：

```cpp
    int numberOfCustomers = s_customerMap.size(); 
    outStream.write((char*) &numberOfCustomers, 
                    sizeof numberOfCustomers); 

    for (const pair<int,Customer>& entry : s_customerMap) { 
      const Customer& customer = entry.second; 
      customer.write(outStream); 
    } 
  } 
} 
```

# 从文件中加载图书馆信息

当从文件中加载图书馆信息时，我们使用与 `read` 相同的方法。我们首先打开文件：

```cpp
void Library::load() { 
  ifstream inStream(s_binaryPath); 
```

我们读取书籍数量，然后通过调用 `read` 方法读取每本书的信息：

```cpp
  if (inStream) { 
    int numberOfBooks; 
    inStream.read((char*) &numberOfBooks, sizeof numberOfBooks); 
```

对于每一本书，我们创建一个新的 `Book` 对象，通过调用 `read` 方法读取其信息，并将其添加到书籍映射中。我们还通过将自身和书籍的身份证号的最大值赋给静态字段 `MaxBookId` 来计算新的 `MaxBookId` 值：

```cpp
    for (int count = 0; count < numberOfBooks; ++count) { 
      Book book; 
      book.read(inStream); 
      s_bookMap[book.bookId()] = book; 
      Book::MaxBookId = max(Book::MaxBookId, book.bookId()); 
    } 
```

同样，我们读取客户数量，然后通过调用 `read` 方法读取每位客户的信息：

```cpp
    int numberOfCustomers; 
    inStream.read((char*) &numberOfCustomers, 
                  sizeof numberOfCustomers); 
```

对于每一位客户，我们创建一个 `Customer` 对象，从文件中读取其信息，将其添加到客户映射中，并为静态字段 `MaxCustomerId` 计算一个新的值：

```cpp
    for (int count = 0; count < numberOfCustomers; ++count) { 
      Customer customer; 
      customer.read(inStream); 
      s_customerMap[customer.id()] = customer; 
      Customer::MaxCustomerId = 
        max(Customer::MaxCustomerId, customer.id()); 
    } 
  } 
} 
```

# 主函数

最后，我们写入 `main` 函数，该函数执行图书馆。这相当简单；唯一要做的就是实例化 `Library` 类的对象。然后构造函数显示主菜单：

**Main.cpp**

```cpp
    #include <Set> 
    #include <Map> 
    #include <List> 
    #include <String> 
    #include <FStream> 
    #include <IOStream> 
    using namespace std; 

    #include "Book.h" 
    #include "Customer.h" 
    #include "Library.h" 

    void main(void) { 
      Library(); 
    } 
```

# 概述

在本章中，我们构建了一个由类 `Book`、`Customer` 和 `Library` 组成的图书馆管理系统。

`Book` 类包含关于一本书的信息。每个 `Book` 对象都有一个唯一的身份号码。它还跟踪借阅者（如果这本书被借出）和预订列表。同样，`Customer` 类包含关于客户的信息。与书类似，每个客户也持有唯一的身份号码。每个 `Customer` 对象还持有借阅和预订的书籍集合。最后，`Library` 类提供了一系列服务，例如添加和删除书籍和客户，借阅、归还和预订书籍，以及显示书籍和客户列表。

在本章中，每本书和每个客户都有一个唯一的身份号码。在下一章中，我们将再次探讨图书馆系统。然而，我们将省略身份号码，而是使用指针来工作。
