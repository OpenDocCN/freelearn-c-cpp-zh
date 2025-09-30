# 基于指针的图书馆管理系统

在本章中，我们将继续研究一个图书馆管理系统。类似于第三章，*构建图书馆管理系统*，我们有三个类—`Book`、`Customer`和`Library`。然而，有一个很大的不同：我们不使用身份号码。相反，我们使用指针；每个`Book`对象都包含一个指向借阅该书的客户（`Customer`类的对象）的指针，以及一个指向已预订该书的客户的指针列表。同样，每个客户都持有他们借阅和预订的书籍（`Book`类的对象）的指针集合。

然而，这种方法引发了一个问题；我们无法直接在文件中存储指针的值。相反，当我们保存文件时，我们需要将指针转换为书籍和客户列表中的索引，而当我们加载文件时，我们需要将索引转换回指针。这个过程被称为**棉花糖化**。

在本章中，我们将更深入地探讨以下主题：

+   正如第三章，*构建图书馆管理系统*中一样，我们将使用构成小型数据库的书籍和客户类。然而，在本章中，我们将直接使用指针而不是整数数字。

+   由于我们使用指针而不是整数数字，文件处理变得更加复杂。我们需要执行一个称为棉花糖化的过程。

+   最后，我们将使用通用的标准 C++类`set`和`list`。然而，在本章中，它们持有指向书籍和客户对象的指针，而不是对象本身。

# 书籍类

与上一章的系统类似，我们有三个类：`Book`、`Customer`和`Library`。`Book`类跟踪书籍，其中每本书都有一个作者和标题。`Customer`类跟踪客户，其中每个客户都有一个姓名和地址。`Library`类跟踪图书馆操作，如借阅、归还和预订。最后，`main`函数简单地创建了一个`Library`类的对象。

`Book`类与第三章，*构建图书馆管理系统*中的`Book`类相似。唯一的真正区别是没有身份号码，只有指针。

**Book.h:**

```cpp
    class Customer; 

    class Book { 
      public: 
      Book(); 
      Book(const string& author, const string& title); 

      const string& author() const { return m_author; } 
      const string& title() const { return m_title; } 

      void read(ifstream& inStream); 
      void write(ofstream& outStream) const; 

      int reserveBook(Customer* customerPtr); 
      void removeReservation(Customer* customerPtr); 
      void returnBook(); 
```

我们没有返回书籍身份号码的方法，因为本章中的书籍不使用身份号码。

`borrowedPtr`方法返回借阅书籍的客户的地址，或者如果此刻没有书籍被借出，则返回`nullptr`。它有两种版本，其中第一种版本返回对`Customer`对象指针的引用。这样，我们可以分配指针的新值给客户。第二种版本是常量版本，这意味着我们可以在常量对象上调用它：

```cpp
    Customer*& borrowerPtr() { return m_borrowerPtr; } 
    const Customer* borrowerPtr() const { return m_borrowerPtr; } 
```

注意，在本章中我们没有`borrowed`方法。我们不需要它，因为如果此时没有借阅书籍，`borrowerPtr`将返回`nullptr`。

在本章中，`reservationPtrList`返回客户指针的列表而不是整数值。它有两种版本，其中第一种返回列表的引用。这样，我们可以向列表中添加和移除指针。第二种版本是常量，返回一个常量列表，这意味着它可以在常量`Book`对象上调用，并返回一个不可更改的列表：

```cpp
    list<Customer*>& reservationPtrList() 
                     { return m_reservationPtrList; } 
    const list<Customer*> reservationPtrList() const 
                          { return m_reservationPtrList; } 
```

输出流操作符的工作方式与第三章，*构建图书馆管理系统*中的方式相同：

```cpp
    friend ostream& operator<<(ostream& outStream, 
          const Book& book); 
```

`m_author`和`m_title`字段是字符串，类似于第三章，*构建图书馆管理系统*：

```cpp
    private: 
      string m_author, m_title; 
```

然而，我们省略了`m_bookId`字段，因为在本章中我们不使用身份号码。我们还用`m_borrowerPtr`替换了`m_borrowedId`和`m_customerId`字段，因为从开始就没有借阅书籍，所以它被初始化为`nullptr`：

```cpp
    Customer* m_borrowerPtr = nullptr; 
```

`m_reservationPtrList`字段包含指向已预约书籍的客户的指针列表，而不是第三章，*构建图书馆管理系统*中的整数身份号码列表：

```cpp
    list<Customer*> m_reservationPtrList; 
      }; 
```

**Book.cpp:** 

```cpp
    #include <Set> 
    #include <Map> 
    #include <String> 
    #include <FStream> 
    #include <Algorithm> 
    using namespace std; 

    #include "Book.h" 
    #include "Customer.h" 
    #include "Library.h" 
```

默认构造函数与第三章，*构建图书馆管理系统*的构造函数类似：

```cpp
    Book::Book() { 
      // Empty. 
    } 
```

第二个构造函数与第三章，*构建图书馆管理系统*的构造函数类似。但是没有`m_bookId`字段需要初始化：

```cpp
    Book::Book(const string& author, const string& title) 
    :m_author(author), 
    m_title(title) { 
      // Empty. 
    } 
```

# 阅读和写入书籍

在本章中，`read`和`write`方法已被简化。它们只读取和写入书籍的作者和标题。潜在的借阅和预约列表由`Library`类的`save`和`write`方法读取和写入：

```cpp
    void Book::read(ifstream& inStream) { 
      getline(inStream, m_author); 
      getline(inStream, m_title); 
    } 

    void Book::write(ofstream& outStream) const { 
      outStream << m_author << endl; 
      outStream << m_title << endl; 
    }
```

# 借阅和预约书籍

当客户预约书籍时，将`Customer`对象的指针添加到书籍的预约指针列表中。返回列表的大小，以便客户知道他们在预约列表中的位置：

```cpp
    int Book::reserveBook(Customer* borrowerPtr) { 
      m_reservationPtrList.push_back(borrowerPtr); 
      return m_reservationPtrList.size(); 
    } 
```

当客户归还书籍时，我们只需将`m_borrowerPtr`设置为`nullptr`，这表示书籍不再被借阅：

```cpp
    void Book::returnBook() { 
      m_borrowerPtr = nullptr; 
    } 
```

`removeReservation`方法简单地从预约列表中移除客户指针：

```cpp
    void Book::removeReservation(Customer* customerPtr) { 
      m_reservationPtrList.remove(customerPtr); 
    } 
```

# 显示书籍

输出流操作符写入标题和作者，以及借阅书籍的客户（如果有），以及已预约书籍的客户（如果有）：

```cpp
    ostream& operator<<(ostream& outStream, const Book& book) { 
     outStream << """ << book.m_title << "" by " << book.m_author; 
```

如果书籍被借阅，我们将借阅者写入流中：

```cpp
    if (book.m_borrowerPtr != nullptr) { 
        outStream << endl << "  Borrowed by: " 
              << book.m_borrowerPtr->name() << "."; 
    } 
```

如果书籍的预约列表不为空，我们遍历它，并为每个预约写入客户：

```cpp
    if (!book.m_reservationPtrList.empty()) { 
      outStream << endl << "  Reserved by: "; 

      bool first = true; 
      for (Customer* customerPtr : book.m_reservationPtrList) { 
        outStream << (first ? "" : ",") << customerPtr->name(); 
        first = false; 
      } 

      outStream << "."; 
    } 

    return outStream; 
```

# 客户类

本章的`Customer`类与第三章，《构建图书馆管理系统》中的`Customer`类相似。再次，在这种情况下，区别在于我们使用指针而不是整数标识号。

**Customer.h:**

```cpp
class Customer { 
  public: 
    Customer(); 
    Customer(const string& name, const string& address); 

    const string& name() const { return m_name; } 
    const string& address() const { return m_address; } 

    void read(ifstream& inStream); 
    void write(ofstream& outStream) const; 
```

`borrowBook`、`returnBook`、`reserveBook`和`unreserveBook`方法接受一个指向`Book`对象的指针作为参数：

```cpp
    void borrowBook(Book* bookPtr); 
    void returnBook(Book* bookPtr); 
    void reserveBook(Book* bookPtr); 
    void unreserveBook(Book* bookPtr); 
```

`loadPtrSet`和`reservationPtrSet`方法返回`Book`指针的集合，而不是整数标识号的集合：

```cpp
    set<Book*>& loanPtrSet() { return m_loanPtrSet; } 
    const set<Book*> loanPtrSet() const { return m_loanPtrSet; } 

    set<Book*>& reservationPtrSet(){ return m_reservationPtrSet; } 
    const set<Book*> reservationPtrSet() const 
                     { return m_reservationPtrSet; } 
```

输出流操作符与第三章，《构建图书馆管理系统》相比没有变化：

```cpp
    friend ostream& operator<<(ostream& outStream, 
                               const Customer& customer); 
```

`m_name`和`m_address`字段存储客户的名称和地址，正如在第三章，《构建图书馆管理系统》中一样：

```cpp
  private: 
    string m_name, m_address; 
```

`m_loanPtrSet`和`m_reservationPtrSet`字段持有指向`Book`对象的指针，而不是整数标识号：

```cpp
    set<Book*> m_loanPtrSet, m_reservationPtrSet; 
      }; 
```

**Customer.cpp:**

```cpp
    #include <Set> 
    #include <Map> 
    #include <String> 
    #include <FStream> 
    using namespace std; 

    #include "Book.h" 
    #include "Customer.h" 
    #include "Library.h" 
```

构造函数与第三章，《构建图书馆管理系统》中的构造函数相似。第一个构造函数不执行任何操作，并在从文件加载客户列表时被调用：

```cpp
    Customer::Customer() { 
     // Empty. 
    } 
```

第二个构造函数初始化客户的名称和地址。然而，与第三章，《构建图书馆管理系统》中的构造函数相比，没有初始化`m_customerId`字段：

```cpp
    Customer::Customer(const string& name, const string& address) 
    :m_name(name), 
    m_address(address) { 
       // Empty. 
    } 
```

# 读取和写入客户信息

与前面的`Book`案例类似，`read`和`write`方法已被简化。它们只读取和写入名称和地址。借阅和预约集合在`Library`类中读取和写入，如下所示：

```cpp
    void Customer::read(ifstream& inStream) { 
      getline(inStream, m_name); 
      getline(inStream, m_address); 
    } 

    void Customer::write(ofstream& outStream) const { 
     outStream << m_name << endl; 
     outStream << m_address << endl; 
    } 
```

# 借阅和预约书籍

`borrowBook`方法将书籍指针添加到借阅集合中，并从预约集合中移除，以防它已被预约：

```cpp
    void Customer::borrowBook(Book* bookPtr) { 
      m_loanPtrSet.insert(bookPtr); 
      m_reservationPtrSet.erase(bookPtr); 
    } 
```

`reserveBook`方法简单地将书籍指针添加到预约列表中，而`returnBook`和`unreserveBook`方法从借阅和预约集合中移除书籍指针：

```cpp
    void Customer::reserveBook(Book* bookPtr) { 
      m_reservationPtrSet.insert(bookPtr); 
    } 

    void Customer::returnBook(Book* bookPtr) { 
      m_loanPtrSet.erase(bookPtr); 
    } 

    void Customer::unreserveBook(Book* bookPtr) { 
      m_reservationPtrSet.erase(bookPtr); 
    } 
```

# 显示客户信息

输出流操作符与第三章，《构建图书馆管理系统》中的操作方式相同。它写入客户的名称和地址，以及借阅和预约的书籍集合（如果有）：

```cpp
    ostream& operator<<(ostream& outStream, const Customer& customer){ 
      outStream << customer.m_name << ", " 
      << customer.m_address << "."; 
```

如果客户的借阅列表不为空，我们遍历它，并对每个借阅项写入书籍：

```cpp
    if (!customer.m_loanPtrSet.empty()) { 
      outStream << endl << "  Borrowed books: "; 

      bool first = true; 
      for (const Book* bookPtr : customer.m_loanPtrSet) { 
        outStream << (first ? "" : ", ") << bookPtr->author(); 
        first = false; 
      } 
    } 
```

同样，如果客户的预约列表不为空，我们遍历它，并对每个预约项写入书籍：

```cpp
    if (!customer.m_reservationPtrSet.empty()) { 
      outStream << endl << "  Reserved books: "; 

      bool first = true; 
      for (Book* bookPtr : customer.m_reservationPtrSet) { 
        outStream << (first ? "" : ", ") << bookPtr->author(); 
        first = false; 
      } 
    } 

    return outStream;
```

# 图书馆类

`Library`类与第三章，《构建图书馆管理系统》中的对应类非常相似。然而，我们在保存和加载图书馆信息到文件时添加了查找`方法`，以在指针和列表索引之间进行转换：

**Library.h:**

```cpp
class Library { 
  public: 
    Library(); 
```

析构函数释放应用程序中所有动态分配的内存：

```cpp
    ~Library(); 

    private: 
      static string s_binaryPath; 
```

`lookupBook` 和 `lookupCustomer` 方法返回指向 `Book` 和 `Customer` 对象的指针。如果书籍或客户不存在，则返回 `nullptr`：

```cpp
    Book* lookupBook(const string& author, const string& title); 
    Customer* lookupCustomer(const string& name, 
                             const string& address); 

    void addBook(); 
    void deleteBook(); 
    void listBooks(); 
    void addCustomer(); 
    void deleteCustomer(); 
    void listCustomers(); 
    void borrowBook(); 
    void reserveBook(); 
    void returnBook(); 
```

`lookupBookIndex` 和 `lookupCustomerIndex` 方法接受一个指针，在指向的对象之后的书籍和客户列表中进行搜索，并返回其在列表中的索引：

```cpp
    int lookupBookIndex(const Book* bookPtr); 
    int lookupCustomerIndex(const Customer* customerPtr);
```

`lookupBookPtr` 和 `lookupCustomerPtr` 方法接受一个索引，并返回书籍和客户列表中该位置的指针：

```cpp
    Book* lookupBookPtr(int bookIndex); 
    Customer* lookupCustomerPtr(int customerIndex); 
```

`save` 和 `write` 方法从文件中保存和加载图书馆信息。然而，它们比 第三章，*构建图书馆管理系统* 中的对应方法更复杂：

```cpp
    void save(); 
    void load(); 
```

`m_bookPtrList` 和 `m_customerPtrList` 字段持有指向 `Book` 和 `Customer` 对象的指针，而不是对象本身，正如 [第三章](https://cdp.packtpub.com/c___by_example/wp-admin/post.php?post=47&action=edit#post_43)，*构建图书馆管理系统* 所述：

```cpp
      list<Book*> m_bookPtrList; 
      list<Customer*> m_customerPtrList; 
      }; 
```

**Library.cpp:**

```cpp
   #include <Set> 
   #include <Map> 
   #include <List> 
   #include <String> 
   #include <FStream> 
   #include <IOStream> 
   #include <CAssert> 
   using namespace std; 

   #include "Book.h" 
   #include "Customer.h" 
   #include "Library.h" 

   string Library::s_binaryPath("C:\Users\Stefan\Library.binary"); 
```

构造函数与 [第三章](https://cdp.packtpub.com/c___by_example/wp-admin/post.php?post=47&action=edit#post_43)，*构建图书馆管理系统* 的构造函数相同：

```cpp
    Library::Library() { 
      load(); 

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

        int choice; 
        cin >> choice; 
        cout << endl; 

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

    save(); 
    } 
```

# 查找书籍和客户

本章的 `lookupBook` 方法通过作者和标题搜索具有 `Book` 对象，其方式类似于 [第三章](https://cdp.packtpub.com/c___by_example/wp-admin/post.php?post=47&action=edit#post_43)，*构建图书馆管理系统*。然而，如果找到与作者和标题匹配的 `Book` 对象，它不会将信息复制到指定的对象中。相反，它只是返回该对象的指针。如果没有找到 `Book` 对象，则返回 `nullptr`：

```cpp
    Book* Library::lookupBook(const string& author, 
                          const string& title) { 
    for (Book* bookPtr : m_bookPtrList) { 
      if ((bookPtr->author() == author) && 
         (bookPtr->title() == title)) { 
        return bookPtr; 
      } 
    } 

    return nullptr; 
   } 
```

同样，`lookupCustomer` 尝试查找与名称和地址匹配的 `Customer` 对象。如果找到该对象，则返回其指针。如果没有找到，则返回 `nullptr`：

```cpp
    Customer* Library::lookupCustomer(const string& name, 
         const string& address) { 
    for (Customer* customerPtr : m_customerPtrList) { 
      if ((customerPtr->name() == name) && 
         (customerPtr->address() == address)) { 
         return customerPtr; 
      } 
    } 
    return nullptr; 
```

# 添加书籍

`addBook` 方法提示用户输入作者和标题：

```cpp
    void Library::addBook() { 
      string author; 
      cout << "Author: "; 
      cin >> author; 

     string title; 
     cout << "Title: "; 
     cin >> title; 
```

在检查书籍是否已存在时，我们调用 `lookupBook`。如果书籍存在，则返回 `Book` 对象的指针。如果书籍不存在，则返回 `nullptr`。因此，我们测试返回值是否不等于 `nullptr`。如果不等于 `nullptr`，则表示书籍已存在，并显示错误消息：

```cpp
      if (lookupBook(author, title) != nullptr) { 
        cout << endl << "The book "" << title << "" by " 
          << author << " already exists." << endl; 
        return; 
      } 
```

在添加书籍时，我们使用 `new` 运算符动态创建一个新的 `Book` 对象。我们使用标准的 C++ `assert` 宏来检查书籍指针是否不为空。如果为空，则执行将因错误消息而终止：

```cpp
      Book* bookPtr = new Book(author, title); 
      assert(bookPtr != nullptr); 
      m_bookPtrList.push_back(bookPtr); 
      cout << endl << "Added." << endl; 
    } 
```

# 删除书籍

`deleteBook` 方法通过提示用户关于书籍的作者和标题来从图书馆中删除书籍。如果书籍存在，我们返回、取消保留并删除它：

```cpp
     void Library::deleteBook() { 
       string author; 
       cout << "Author: "; 
       cin >> author; 

       string title; 
       cout << "Title: "; 
       cin >> title; 
```

我们通过调用 `lookupBook` 获取 `Book` 对象的指针：

```cpp
     Book* bookPtr = lookupBook(author, title); 
```

如果指针是`nullptr`，则书籍不存在，并显示错误消息：

```cpp
       if (bookPtr == nullptr) { 
         cout << endl << "The book "" << title << "" by " 
           << author << " does not exist." << endl; 
         return; 
       } 
```

我们通过查找借阅者来检查书籍是否已被借阅：

```cpp
        Customer* borrowerPtr = bookPtr->borrowerPtr(); 
```

如果`borrowerPtr`返回的指针不是`nullptr`，我们通过调用借阅者的`returnBook`方法来归还书籍。这样，书籍就不再被注册为客户借阅的书籍：

```cpp
  if (borrowerPtr != nullptr) { 
    borrowerPtr->returnBook(bookPtr); 
  } 
```

此外，我们需要检查书籍是否已被其他客户预留。我们通过获取书籍的预留列表来实现，并对列表中的每个客户，取消书籍的预留：

```cpp
    list<Customer*> reservationPtrList = 
      bookPtr->reservationPtrList(); 
```

注意，我们并不检查书籍是否实际上已被客户预留，我们只是取消预留。另外，注意我们不需要将任何对象放回列表中，因为我们处理的是对象的指针，而不是对象的副本：

```cpp
      for (Customer* reserverPtr : reservationPtrList) { 
        reserverPtr->unreserveBook(bookPtr); 
      }
```

当移除书籍时，我们从书籍指针列表中移除书籍指针，然后释放`Book`对象。看起来我们首先显示消息然后删除书籍指针似乎很奇怪。然而，顺序必须如此。删除对象后，我们无法再对其进行任何操作。我们不能先删除对象再写入它，这会导致内存错误：

```cpp
      m_bookPtrList.remove(bookPtr); 
        n cout << endl << "Deleted:" << bookPtr << endl; 
        delete bookPtr; 
      } 
```

# 列出书籍

当列出书籍时，我们首先检查列表是否为空。如果为空，我们简单地写入`"No books."`：

```cpp
    void Library::listBooks() { 
      if (m_bookPtrList.empty()) { 
       cout << "No books." << endl; 
       return; 
      } 
    }
```

然而，如果列表不为空，我们遍历书籍指针列表，并对每个书籍指针，解引用指针并写入信息：

```cpp
      for (const Book* bookPtr : m_bookPtrList) { 
        cout << (*bookPtr) << endl; 
        } 
    } 
```

# 添加客户

`addCustomer`方法提示用户输入客户的名称和地址：

```cpp
    void Library::addCustomer() { 
      string name; 
       cout << "Name: "; 
       cin >> name; 

       string address; 
       cout << "Address: "; 
       cin >> address;
```

如果已存在具有相同名称和地址的客户，将显示错误消息：

```cpp
      if (lookupCustomer(name, address) != nullptr) { 
        cout << endl << "A customer with name " << name 
         << " and address " << address << " already exists." 
         << endl; 
       return; 
      } 
```

当添加客户时，我们动态创建一个新的`Customer`对象，并将其添加到客户对象指针列表中：

```cpp
      Customer* customerPtr = new Customer(name, address); 
        assert(customerPtr != nullptr); 
        m_customerPtrList.push_back(customerPtr); 
        cout << endl << "Added." << endl; 
      } 
```

# 删除客户

当删除客户时，我们查找他们，如果不存在则显示错误消息：

```cpp
    void Library::deleteCustomer() { 
      string name; 
      cout << "Customer name: "; 
      cin >> name; 

      string address; 
      cout << "Address: "; 
      cin >> address; 

      Customer* customerPtr = lookupCustomer(name, address); 
```

如果给定名称和地址的客户不存在，则显示错误消息。考虑以下代码：

```cpp
    if (customerPtr == nullptr) { 
      cout << endl << "Customer " << name 
         << " does not exists." << endl; 
      return; 
    }
```

如果客户至少借阅了一本书，则不能删除，并显示错误消息，如下所示：

```cpp
     if (!customerPtr->loanPtrSet().empty()) { 
      cout << "The customer " << customerPtr->name() 
         << " has borrowed books and cannot be deleted." << endl; 
      return; 
     } 
```

然而，如果客户没有借阅任何书籍，客户首先从图书馆中每本书的预留列表中移除，如下面的代码所示：

```cpp
     for (Book* bookPtr : m_bookPtrList) { 
       bookPtr->removeReservation(customerPtr); 
     } 
```

然后客户从客户列表中移除，并通过`delete`运算符释放`Customer`对象。再次注意，我们首先必须写入客户信息，然后删除其对象。反过来是不行的，因为我们无法检查已删除的对象。这会导致内存错误：

```cpp
      m_customerPtrList.remove(customerPtr); 
      cout << endl << "Deleted." << (*customerPtr) << endl; 
      delete customerPtr; 
    } 
```

# 列出客户

当列出客户时，我们遍历客户列表，并对每个客户，解引用`Customer`对象指针并写入对象信息：

```cpp
    void Library::listCustomers() { 
      if (m_customerPtrList.empty()) { 
        cout << "No customers." << endl; 
        return; 
      } 

      for (const Customer* customerPtr: m_customerPtrList) { 
        cout << (*customerPtr) << endl; 
      } 
    }
```

# 借阅书籍

当借阅书籍时，我们首先提示用户输入作者和标题，如下面的代码片段所示：

```cpp
    void Library::borrowBook() { 
      string author; 
      cout << "Author: "; 
      cin >> author; 

      string title; 
      cout << "Title: "; 
      cin >> title; 
```

我们查找书籍，如果书籍不存在，将显示错误信息，如下面的代码所示：

```cpp
     Book* bookPtr = lookupBook(author, title); 

     if (bookPtr == nullptr) { 
       cout << endl << "There is no book "" << title 
         << "" by " << author << "." << endl; 
       return; 
     } 
```

如果这本书已经被其他顾客借走，则不能再被借阅：

```cpp
    if (bookPtr->borrowerPtr() != nullptr) { 
      cout << endl << "The book "" << title << "" by " << author  
         << " has already been borrowed." << endl; 
      return; 
    } 
```

我们提示用户输入顾客的姓名和地址：

```cpp
     string name; 
     cout << "Customer name: "; 
     cin >> name; 

     string address; 
     cout << "Address: "; 
     cin >> address; 

     Customer* customerPtr = lookupCustomer(name, address);
```

如果没有找到具有给定姓名和地址的顾客，将显示错误信息：

```cpp
     if (customerPtr == nullptr) { 
      cout << endl << "No customer with name " << name 
         << " and address " << address << " exists."  << endl; 
      return; 
     } 
```

最后，我们将书籍添加到顾客的借阅集合中，并将顾客标记为书籍的借阅者：

```cpp
     bookPtr->borrowerPtr() = customerPtr; 
     customerPtr->borrowBook(bookPtr); 
     cout << endl << "Borrowed." << endl; 
   } 
```

# 预订书籍

预订过程与之前的借阅过程类似。我们提示用户输入书籍的作者和标题，以及顾客的姓名和地址，如下所示：

```cpp
    void Library::reserveBook() { 
      string author; 
      cout << "Author: "; 
      cin >> author; 

      string title; 
      cout << "Title: "; 
      cin >> title; 

    Book* bookPtr = lookupBook(author, title); 
```

如果书籍不存在，将显示错误信息：

```cpp
    if (bookPtr == nullptr) { 
      cout << endl << "There is no book "" << title 
         << "" by " << author << "." << endl; 
      return; 
    } 
```

如果书籍尚未被借阅，则无法预订。相反，我们鼓励用户借阅这本书：

```cpp
     if (bookPtr->borrowerPtr() == nullptr) { 
       cout << endl << "The book "" << title << "" by " 
         << author << " has not been not borrowed. " 
         << "Please borrow the book instead of reserving it." 
         << endl; 
      return; 
    } 
```

我们提示用户输入顾客的姓名和地址：

```cpp
     string name; 
     cout << "Customer name: "; 
     cin >> name; 

     string address; 
     cout << "Address: "; 
     cin >> address; 

     Customer* customerPtr = lookupCustomer(name, address); 
```

如果顾客不存在，将显示错误信息：

```cpp
     if (customerPtr == nullptr) { 
      cout << endl << "There is no customer with name " << name 
         << " and address " << address << "." << endl; 
      return; 
     } 
```

如果顾客已经借阅了这本书，他们也不能预订这本书：

```cpp
     if (bookPtr->borrowerPtr() == customerPtr) { 
      cout << endl << "The book has already been borrowed by " 
         << name << "." << endl; 
      return; 
     } 
```

最后，我们将顾客添加到书籍的预订列表中，并将书籍添加到顾客的预订集合中。请注意，对于书籍有一个预订顾客列表，而对于顾客有一个已预订书籍集合。这样做的原因是当一本书被归还时，预订列表中的第一个顾客会借阅这本书。对于顾客的预订集合没有这样的限制：

```cpp
     int position = bookPtr->reserveBook(customerPtr); 
     customerPtr->reserveBook(bookPtr); 
```

我们通知顾客其在预订列表中的位置：

```cpp
     cout << endl << position << "nd reserve." << endl; 
     }
```

# 归还书籍

当归还一本书时，我们提示用户输入其作者和标题。然而，我们不询问借阅这本书的顾客。这些信息已经存储在`Book`对象中：

```cpp
    void Library::returnBook() { 
      string author; 
      cout << "Author: "; 
      cin >> author; 

      string title; 
      cout << "Title: "; 
      cin >> title; 

      Book* bookPtr = lookupBook(author, title); 
```

如果给定作者和标题的书籍不存在，将显示错误信息：

```cpp
     if (bookPtr == nullptr) { 
      cout << endl << "There is no book "" << title << "" by " 
         << author << "." << endl; 
      return; 
     } 

     Customer* customerPtr = bookPtr->borrowerPtr(); 
```

如果给定姓名和地址的顾客不存在，将显示错误信息：

```cpp
     if (customerPtr == nullptr) { 
      cout << endl << "The book "" << title << "" by " 
         << author << " has not been borrowed." << endl; 
      return; 
     } 

     bookPtr->returnBook(); 
     customerPtr->returnBook(bookPtr); 
     cout << endl << "Returned." << endl; 
```

当我们归还了书籍后，我们需要找出是否有任何顾客已经预订了它：

```cpp
     list<Customer*>& reservationPtrList = 
       bookPtr->reservationPtrList();
```

如果书籍的预订列表中至少有一个顾客，我们获取该顾客，将其从书籍的预订列表中移除，将顾客标记为书籍的借阅者，并将书籍添加到顾客的借阅集合中：

```cpp
     if (!reservationPtrList.empty()) { 
       Customer* newCustomerPtr = reservationPtrList.front(); 
       reservationPtrList.erase(reservationPtrList.begin()); 

       bookPtr->borrowBook(newCustomerPtr); 
       newCustomerPtr->borrowBook(bookPtr); 
       cout << endl << "Borrowed by " 
          << newCustomerPtr->name() << endl; 
       } 
     } 
```

# 查找书籍和顾客

当从文件保存和加载图书馆信息时，我们需要在`Book`和`Customer`对象的指针与书籍和顾客列表中的索引之间进行转换。`lookupIndex`方法接受一个指向`Book`对象的指针，并返回它在书籍列表中的索引：

```cpp
    int Library::lookupBookIndex(const Book* bookPtr) { 
      int index = 0; 

      for (Book* testPtr : m_bookPtrList) { 
        if (bookPtr == testPtr) { 
        return index; 
      } 

      ++index; 
    } 
```

如果我们达到这个点，执行将通过`assert`宏显示错误信息而终止。然而，我们不应该达到这个点，因为`Book`指针应该在书籍指针列表中：

```cpp
     assert(false); 
      return -1; 
     }
```

`lookupBookPtr` 方法执行相反的任务。它根据`bookIndex`在书籍指针列表中的位置找到`Book`对象指针。如果索引超出列表范围，`assert`宏会通过错误信息终止执行。然而，这种情况不应该发生，因为所有索引都应该在范围内：

```cpp
    Book* Library::lookupBookPtr(int bookIndex) { 
      assert((bookIndex >= 0) && 
       (bookIndex < ((int) m_bookPtrList.size()))); 

      auto iterator = m_bookPtrList.begin(); 
      for (int count = 0; count < bookIndex; ++count) { 
        ++iterator; 
      } 

      return *iterator; 
    }  
```

`lookupCustomerIndex`方法以与前面`lookupBookIndex`方法相同的方式给出`Customer`指针在客户指针列表中的索引：

```cpp
    int Library::lookupCustomerIndex(const Customer* customerPtr) { 
      int index = 0; 

      for (Customer* testPtr : m_customerPtrList) { 
        if (customerPtr == testPtr) { 
        return index; 
      } 

      ++index; 
     } 

     assert(false); 
     return -1; 
    } 
```

`lookupCustomerPtr`方法以与前面`lookupBookPtr`方法相同的方式在客户指针列表中查找`Customer`指针的索引：

```cpp
    Customer* Library::lookupCustomerPtr(int customerIndex) { 
      assert((customerIndex >= 0) && 
       (customerIndex < ((int) m_customerPtrList.size()))); 

      auto iterator = m_customerPtrList.begin(); 
      for (int count = 0; count < customerIndex; ++count) { 
        ++iterator; 
      }  

      return *iterator; 
    } 
```

# Marshmallowing

本章中`Library`类的`save`和`load`方法比第三章中对应的*构建图书馆管理系统*要复杂一些。原因是我们不能直接保存指针，因为指针持有可能在执行之间改变的内存地址。相反，我们需要将它们的索引保存到文件中。将指针转换为索引和索引转换为指针的过程称为**Marshmallowing**。当保存图书馆时，我们将保存过程分为几个步骤：

+   保存书籍列表：在这个阶段，我们只保存作者和标题。

+   保存客户列表：在这个阶段，我们只保存姓名和地址。

+   对于每本书：保存借阅者（如果书籍被借出）和（可能为空的）预订列表。我们保存客户列表索引，而不是客户的指针。

+   对于每个客户，我们保存借阅和预订集合。我们保存书籍列表索引，而不是书籍的指针。

# 将图书馆信息保存到文件中

`Save`方法打开文件，如果成功打开，则读取图书馆的书籍和客户：

```cpp
    void Library::save() { 
      ofstream outStream(s_binaryPath); 
```

# 编写书籍对象

我们保存书籍对象。我们通过为每个`Book`对象调用`write`来只保存书籍的作者和标题。在这个阶段，我们不保存潜在的借阅者和预订列表。

我们首先将列表中的书籍数量写入文件：

```cpp
    if (outStream) { 
      { int bookPtrListSize = m_bookPtrList.size(); 
         outStream.write((char*) &bookPtrListSize, 
           sizeof bookPtrListSize); 
```

然后我们通过在每个`Book`对象指针上调用`write`来将每本书的信息写入文件：

```cpp
      for (const Book* bookPtr : m_bookPtrList) { 
        bookPtr->write(outStream); 
    } 
   } 
```

# 编写客户对象

我们保存客户对象。类似于前面的书籍案例，我们通过为每个`Customer`对象调用`write`来只保存客户的名字和地址。在这个阶段，我们不保存借阅和预订的书籍集合。

同样地，就像前面的书籍案例一样，我们首先将列表中的客户数量写入文件：

```cpp
    { int customerPtrListSize = m_customerPtrList.size(); 
      outStream.write((char*) &customerPtrListSize, 
                      sizeof customerPtrListSize); 
```

然后我们通过在每个`Customer`对象指针上调用`write`方法来将每个客户的信息写入文件：

```cpp
      for (const Customer* customerPtr : m_customerPtrList) { 
        customerPtr->write(outStream); 
      } 
    } 
```

# 编写借阅者索引

对于每个`Book`对象，如果书籍被借出，我们查找并保存`Customer`的索引，而不是对象的指针：

```cpp
    for (const Book* bookPtr : m_bookPtrList) { 
      { const Customer* borrowerPtr = bookPtr->borrowerPtr(); 
```

对于每本书，我们首先检查它是否已被借出。如果已被借出，我们将值`true`写入文件，以表示它已被借出：

```cpp
        if (borrowerPtr != nullptr) { 
          bool borrowed = true; 
          outStream.write((char*) &borrowed, sizeof borrowed); 
```

然后在客户指针列表中查找借阅了这本书的客户索引，并将索引写入文件：

```cpp
          int loanIndex = lookupCustomerIndex(borrowerPtr); 
          outStream.write((char*) &loanIndex, sizeof loanIndex); 
        } 
```

如果书籍没有被借出，我们只需将值`false`写入文件，以表示书籍没有被借出：

```cpp
        else { 
          bool borrowed = false; 
          outStream.write((char*) &borrowed, sizeof borrowed); 
        } 
      } 
```

# 写入预约索引

由于一本书可以被多个客户预约，我们遍历预约列表并保存每个客户在预约列表中的索引：

```cpp
      { const list<Customer*>& reservationPtrList = 
          bookPtr->reservationPtrList(); 
```

对于每一本书，我们首先将书的预约数量写入文件：

```cpp
        int reserveSetSize = reservationPtrList.size(); 
        outStream.write((char*) &reserveSetSize, 
                        sizeof reserveSetSize); 
```

然后我们遍历预约列表，对于每个预约，我们在文件中查找并写入预约了这本书的每个客户的索引：

```cpp
        for (const Customer* customerPtr : reservationPtrList) { 
          int customerIndex = lookupCustomerIndex(customerPtr); 
          outStream.write((char*) &customerIndex, 
                          sizeof customerIndex); 
        } 
      } 
    }
```

# 写入借阅书籍索引

对于每个客户，我们保存他们借阅的书籍索引。首先保存借阅列表的大小，然后是书籍索引：

```cpp
    for (const Customer* customerPtr : m_customerPtrList) { 
      { const set<Book*>& loanPtrSet = 
          customerPtr->loanPtrSet(); 
```

对于每个客户，我们首先写入其借阅数量到文件：

```cpp
        int loanPtrSetSize = loanPtrSet.size(); 
        outStream.write((char*) &loanPtrSetSize, 
                        sizeof loanPtrSetSize); 
```

然后我们遍历借阅集合，对于每个借阅，我们在文件中查找并写入每本书的索引：

```cpp
        for (const Book* customerPtr : loanPtrSet) { 
          int customerIndex = lookupBookIndex(customerPtr); 
          outStream.write((char*) &customerIndex, 
                          sizeof customerIndex); 
        } 
      } 
```

# 写入预约书籍索引

同样地，对于每个客户，我们保存他们预约的书籍索引。首先保存预约列表的大小，然后是预约的书籍索引：

```cpp
      { const set<Book*>& reservedPtrSet = 
          customerPtr->reservationPtrSet(); 
```

对于每个客户，我们首先写入其预约书籍的数量到文件：

```cpp
        int reservationPtrSetSize = reservationPtrSet.size(); 
        outStream.write((char*) &reservationPtrSetSize, 
                        sizeof reservationPtrSetSize); 
```

然后我们遍历预约集合，对于每个预约，我们在文件中查找并写入每本书的索引：

```cpp
        for (const Book* reservedPtr : reservationPtrSet) { 
          int customerIndex = lookupBookIndex(reservedPtr); 
          outStream.write((char*) &customerIndex, 
                          sizeof customerIndex); 
        } 
      } 
    } 
  } 
} 
```

# 从文件中加载图书馆信息

当加载文件时，我们按照保存文件时的相同方式操作：

```cpp
    void Library::load() { 
      ifstream inStream(s_binaryPath); 
```

# 读取书籍对象

我们读取书籍列表的大小，然后读取书籍本身。记住，到目前为止，我们只读取了书籍的作者和标题：

```cpp
    if (inStream) { 
      { int bookPtrListSize; 
```

我们首先读取书籍数量：

```cpp
        inStream.read((char*) &bookPtrListSize, 
                    sizeof bookPtrListSize); 
```

然后读取书籍本身。对于每本书，我们动态分配一个`Book`对象，通过调用指针上的`read`方法读取其信息，并将指针添加到书籍指针列表中：

```cpp
      for (int count = 0; count < bookPtrListSize; ++count) { 
        Book *bookPtr = new Book(); 
        assert(bookPtr != nullptr); 
        bookPtr->read(inStream); 
        m_bookPtrList.push_back(bookPtr); 
      } 
    }
```

# 读取客户对象

同样地，我们读取客户列表的大小，然后读取客户本身。到目前为止，我们只读取了客户的姓名和地址：

```cpp
    { int customerPtrListSize; 
```

我们首先读取客户数量：

```cpp
      inStream.read((char*) &customerPtrListSize, 
                    sizeof customerPtrListSize); 
```

然后我们读取客户本身。对于每个客户，我们动态分配一个`Customer`对象，通过调用指针上的`read`方法读取其信息，并将指针添加到书籍指针列表中：

```cpp
      for (int count = 0; count < customerPtrListSize; ++count) { 
        Customer *customerPtr = new Customer(); 
        assert(customerPtr != nullptr); 
        customerPtr->read(inStream); 
        m_customerPtrList.push_back(customerPtr); 
      } 
    } 
```

# 读取借阅者索引

对于每本书，我们读取借阅了它的客户（如果有）以及预约了这本书的客户列表：

```cpp
    for (Book* bookPtr : m_bookPtrList) { 
      { bool borrowed; 
        inStream.read((char*) &borrowed, sizeof borrowed); 
```

如果`borrowed`是`true`，则表示书籍已被借出。在这种情况下，我们读取客户索引。然后我们查找`Customer`对象的指针，将其添加到书籍的预约列表中：

```cpp
        if (borrowed) { 
          int loanIndex; 
          inStream.read((char*) &loanIndex, sizeof loanIndex); 
          bookPtr->borrowerPtr() = lookupCustomerPtr(loanIndex); 
        }
```

如果 `borrowed` 是 `false`，则表示书籍尚未被借出。在这种情况下，我们将借出书籍的客户指针设置为 `nullptr`：

```cpp
        else { 
          bookPtr->borrowerPtr() = nullptr; 
        } 
      } 
```

# 读取预订索引

对于每一本书，我们也会读取预订列表。首先，我们读取列表的大小，然后是客户索引本身：

```cpp
      { list<Customer*>& reservationPtrList = 
          bookPtr->reservationPtrList(); 
        int reservationPtrListSize; 
```

我们首先读取书籍的预订数量：

```cpp
        inStream.read((char*) &reservationPtrListSize, 
                      sizeof reservationPtrListSize); 
```

对于每一笔预订，我们读取客户的索引并调用 `lookupCustomerPtr` 来获取 `Customer` 对象的指针，然后将其添加到书籍的预订指针列表中：

```cpp
        for (int count = 0; count < reservationPtrListSize; 
             ++count) { 
          int customerIndex; 
          inStream.read((char*) &customerIndex, 
                        sizeof customerIndex); 
          Customer* customerPtr = 
            lookupCustomerPtr(customerIndex); 
          reservationPtrList.push_back(customerPtr); 
        } 
      } 
    } 
```

# 读取借阅书籍索引

对于每一位客户，我们读取借阅的书籍集合：

```cpp
    for (Customer* customerPtr : m_customerPtrList) { 
      { set<Book*>& loanPtrSet = customerPtr->loanPtrSet(); 
        int loanPtrSetSize = loanPtrSet.size();
```

我们首先读取借阅列表的大小：

```cpp
        inStream.read((char*) &loanPtrSetSize, 
                      sizeof loanPtrSetSize); 
```

对于每一笔借阅，我们读取书籍的索引并调用 `lookupBookPtr` 来获取 `Book` 对象的指针，然后将其添加到借阅指针列表中：

```cpp
        for (int count = 0; count < loanPtrSetSize; ++count) { 
          int bookIndex; 
          inStream.read((char*) &bookIndex, sizeof bookIndex); 
          Book* bookPtr = lookupBookPtr(bookIndex); 
          loanPtrSet.insert(bookPtr); 
        } 
      } 
```

# 读取预订书籍索引

同样地，对于每一位客户，我们读取预订的书籍集合：

```cpp
      { set<Book*>& reservationPtrSet = 
          customerPtr->reservationPtrSet(); 
```

我们首先读取预订列表的大小：

```cpp
        int reservationPtrSetSize = reservationPtrSet.size(); 
        inStream.read((char*) &reservationPtrSetSize, 
                      sizeof reservationPtrSetSize); 
```

对于每一笔预订，我们读取书籍的索引并调用 `lookupBookPtr` 来获取 `Book` 对象的指针，然后将其添加到预订指针列表中：

```cpp
        for (int count = 0; count < reservationPtrSetSize; 
             ++count) { 
          int bookIndex; 
          inStream.read((char*) &bookIndex, sizeof bookIndex); 
          Book* bookPtr = lookupBookPtr(bookIndex); 
          reservationPtrSet.insert(bookPtr); 
        } 
      } 
    } 
  } 
}
```

# 释放内存

由于我们已经将动态分配的 `Book` 和 `Customer` 对象添加到列表中，我们需要在执行结束时释放它们。析构函数遍历书籍和客户指针列表，并释放所有书籍和客户指针：

```cpp
    Library::~Library() { 
      for (const Book* bookPtr : m_bookPtrList) { 
        delete bookPtr; 
      } 

      for (const Customer* customerPtr : m_customerPtrList) { 
        delete customerPtr; 
      } 
    } 
```

# 主函数

与 第三章，*构建图书馆管理系统* 类似，`main` 函数只是创建一个 `Library` 对象：

**Main.cpp**

```cpp
    #include <Set> 
    #include <Map> 
    #include <String> 
    #include <FStream> 
    #include <IOStream> 
    using namespace std; 

    #include "Book.h" 
    #include "Customer.h" 
    #include "Library.h" 

    void main() { 
      Library(); 
    }
```

# 摘要

在本章中，我们构建了一个类似于 第三章，*构建图书馆管理系统* 的图书馆管理系统。然而，我们省略了所有整数身份号码，并用指针替换了它们。这使我们能够更直接地存储借阅和预订，但也使得我们保存和加载到文件中变得更加困难。

在 第五章，*Qt 图形应用程序* 中，我们将探讨图形应用程序。
