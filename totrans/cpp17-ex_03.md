# Building a Library Management System

In this chapter, we study a system for the management of a library. We continue to develop C++ classes, as in the previous chapters. However, in this chapter, we develop a more real-world system. The library system of this chapter can be used by a real library.

The library is made up of sets of books and customers. The books keep track of which customers have borrowed or reserved them. The customers keep track of which books they have borrowed and reserved.

The main idea is that the library holds a set of books and a set of customers. Each book is marked as borrowed or unborrowed. If it is borrowed, the identity number of the customer that borrowed the book is stored. Moreover, a book can also be reserved by one or several customers. Therefore, each book also holds a list of identity numbers for the customers that have reserved the book. It must be a list rather than a set, since the book shall be loaned to the customers in the order that they reserved the book.

Each customer holds two sets with the identity numbers of the book they have borrowed and reserved. In both cases, we use sets rather than lists since the order they have borrowed or reserved the books does not matter.

In this chapter, we will cover the following topics:

*   Working with classes for books and customers that constitute a small database with integer numbers as keys.
*   Working with standard input and output streams, where we write information about the books and customers, and prompt the user for input.
*   Working with file handling and streams. The books and customers are written and read with standard C++ file streams.
*   Finally, we work with the generic classes `set` and `list` from the C++ standard library.

# The Book class

We have three classes: `Book`, `Customer`, and `Library`:

*   The `Book` class keeps track of a book. Each book has an author and a title, and a unique identity number.
*   The `Customer` class keeps track of a customer. Each customer has a name and an address, and a unique identity number.
*   The `Library` class keeps track of the library operations, such as adding and removing books and customers, borrowing, returning, and reserving books, as well as listing books and customers.
*   The `main` function simply creates an object of the `Library` class.

Moreover, each book holds information on whether it is borrowed at the moment. If it is borrowed, the identity number of the customer who has borrowed the book is also stored. Each book also holds a list of reservations. In the same way, each customer holds sets of books currently borrowed and reserved.

The `Book` class holds two constructors. The first constructor is a default constructor and is used when reading books from a file. The second constructor is used when adding a new book to the library. It takes the name of the author and the title of the book as parameters.

**Book.h**

```cpp
class Book { 
  public: 
    Book(void); 
    Book(const string& author, const string& title); 
```

The `author` and `title` methods simply return the author and title of the book:

```cpp
    const string& author(void) const { return m_author; } 
    const string& title(void) const { return m_title; } 
```

The books of the library can be read from and written to a file:

```cpp
    void read(ifstream& inStream); 
    void write(ofstream& outStream) const; 
```

A book can be borrowed, reserved, or returned. A reservation can also be removed. Note that when a book is borrowed or reserved, we need to provide the identity number of the customer. However, that is not necessary when returning a book, since the `Book` class keeps track of the customer that has currently borrowed the book:

```cpp
    void borrowBook(int customerId); 
    int reserveBook(int customerId); 
    void unreserveBookation(int customerId); 
    void returnBook(); 
```

When the book is borrowed, the customer's identity number is stored, which is returned by `bookId`:

```cpp
    int bookId(void) const { return m_bookId; } 
```

The `borrowed` method returns true if the book is borrowed at the moment. In that case, `customerId` returns the identity number of the customer who has borrowed the book:

```cpp
    bool borrowed(void) const { return m_borrowed; } 
    int customerId(void) const { return m_customerId; } 
```

A book can be reserved by a list of customers, and `reservationList` returns that list:

```cpp
    list<int>& reservationList(void) { return m_reservationList; } 
```

The `MaxBookId` field is static, which means that it is common to all objects of the class:

```cpp
    static int MaxBookId; 
```

The output stream operator writes the information of the book:

```cpp
    friend ostream& operator<<(ostream& outStream, 
                               const Book& book); 
```

The `m_borrowed` field is true when the book is borrowed. The identity of the book and potential borrower are stored in `m_bookId` and `m_customerId`:

```cpp
    private: 
      bool m_borrowed = false; 
      int m_bookId, m_customerId; 
```

The name of the author and the title of the book are stored in `m_author` and `m_title`:

```cpp
      string m_author, m_title; 
```

More than one customer can reserve a book. When they do, their identities are stored in `m_reservationList`. It is a list rather than a set because the reservations are stored in order. When a book is returned, the next customer, in reservation order, borrows the book:

```cpp
      list<int> m_reservationList; 
      }; 
```

In this chapter, we use the generic `set`, `map`, and `list` classes from the C++ standard library. Their specifications are stored in the `Set`, `Map`, and `List` header files. The `set` and `list` classes hold a set and a list similar to our set and list classes in the previous chapter. A map is a structure where each value is identified by a unique key in order to provide fast access.

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

Since `MaxBookId` is static, we initialize it with the double colon (`::`) notation. Every static field needs to be initialized outside the class definition:

```cpp
    int Book::MaxBookId = 0; 
```

The default constructor does nothing. It is used when reading from a file. Nevertheless, we still must have a default constructor to create objects of the `Book` class:

```cpp
    Book::Book(void) { 
      // Empty. 
    } 
```

When a new book is created, it is assigned a unique identity number. The identity number is stored in `MaxBookId`, which is increased for each new `Book` object:

```cpp
    Book::Book(const string& author, const string& title) 
     :m_bookId(++MaxBookId), 
      m_author(author), 
      m_title(title) { 
      // Empty. 
    } 
```

# Writing the book

A book is written to a stream in a similar manner. However, instead of `read` we use `write`. They work in a similar manner:

```cpp
    void Book::write(ofstream& outStream) const { 
      outStream.write((char*) &m_bookId, sizeof m_bookId); 
```

When reading a string we use `getline` instead of the stream operator, since the stream operator reads one word only, while `getline` reads several words. When writing to a stream, however, we can use the stream operator. It does not matter whether the name and title are made up of one or several words:

```cpp
    outStream << m_author << endl; 
    outStream << m_title << endl; 

    outStream.write((char*) &m_borrowed, sizeof m_borrowed); 
    outStream.write((char*) &m_customerId, sizeof m_customerId); 
```

Similar to the reading case here, we first write the number of reservations in the list. Then we write the reservation identities themselves:

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

# Reading the book

When reading any kind of value (except strings) from a file, we use the `read` method, which reads a fixed number of bytes. The `sizeof` operator gives us the size, in bytes, of the `m_bookId` field. The `sizeof` operator can also be used to find the size of a type. For instance, `sizeof (int)` gives us the size in bytes of a value of the type `int`. The type must be enclosed in parentheses:

```cpp
    void Book::read(ifstream& inStream) { 
      inStream.read((char*) &m_bookId, sizeof m_bookId); 
```

When reading string values from a file, we use the C++ standard function `getline` to read the name of the author and the title of the book. It would not work to use the input stream operator if the name is made up of more than one word. If the author or title is made up of more than one word, only the first word would be read. The remaining words would not be read:

```cpp
    getline(inStream, m_author);
    getline(inStream, m_title);
```

Note that we use the `read` method to read the value of the `m_borrowed` field, too, even though it holds the `bool` type rather than `int`:

```cpp
    inStream.read((char*) &m_borrowed, sizeof m_borrowed);
    inStream.read((char*) &m_customerId, sizeof m_customerId);
```

When reading the reservation list, we first read the number of reservations in the list. Then we read the reservation identity numbers themselves:

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

# Borrowing and reserving the book

When the book is borrowed, `m_borrowed` becomes `true` and `m_customerId` is set to the identity number of the customer that borrowed the book:

```cpp
void Book::borrowBook(int customerId) { 
  m_borrowed = true; 
  m_customerId = customerId; 
} 
```

It is a little bit different when the book is reserved. While a book can be borrowed by one customer only, it can be reserved by more than one customer. The identity number of the customer is added to `m_reservationList`. The size of the list is returned for the caller to know their position in the reservation list:

```cpp
    int Book::reserveBook(int customerId) { 
      m_reservationList.push_back(customerId); 
      return m_reservationList.size(); 
    } 
```

When the book is returned, we just set `m_borrowed` to false. We do not need to set `m_customerId` to anything specific. It is not relevant as long as the book is not borrowed:

```cpp
    void Book::returnBook() { 
      m_borrowed = false; 
    } 
```

A customer can remove themselves from the reservation list. In that case, we call `remove` on `m_reservationList`:

```cpp
    void Book::unreserveBookation(int customerId) { 
      m_reservationList.remove(customerId); 
    } 
```

# Displaying the book

The output stream operator writes the title and author of the book. If the book is borrowed, the customer's name is written, and if the reservation list is full, the reservation customers' names are written:

```cpp
    ostream& operator<<(ostream& outStream, const Book& book) { 
      outStream << """ << book.m_title << "" by " << book.m_author; 
```

We use the double-colon notation (`::`) when accessing a static field, such as `s_customerMap` in `Library`:

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

# The Customer class

The `Customer` class keeps track of a customer. It holds sets of the books the customer currently has borrowed and reserved.

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

The `hasBorrowed` method returns true if the customer has at least one book borrowed at the moment. In the `Library` class in the next section, it is not possible to remove a customer who currently has borrowed books:

```cpp
    bool hasBorrowed(void) const { return !m_loanSet.empty(); } 

    const string& name(void) const {return m_name;} 
    const string& address(void) const {return m_address;} 
    int id(void) const {return m_customerId;} 
```

In the same way, as in the `Book` class, which was used previously, we use the static field `MaxCustomerId` to count the identity number of the customers. We also use the output stream operator to write information about the customer:

```cpp
    static int MaxCustomerId; 
    friend ostream& operator<<(ostream& outStream, 
                               const Customer& customer); 
```

Each customer has a name, address, and unique identity number. The sets `m_loanSet` and `m_reservationSet` hold the identity numbers of the books currently borrowed and reserved by the customer. Note that we use sets instead of lists, since the order of the books borrowed and reserved does not matter:

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

Since `MaxCustomerId` is a static field, it needs to be defined outside the class:

```cpp
    int Customer::MaxCustomerId; 
```

The default constructor is used when loading objects from a file only. Therefore, there is no need to initialize the fields:

```cpp
    Customer::Customer(void) { 
      // Empty. 
    } 
```

The second constructor is used when creating new book objects. We use the `MaxCustomerId` field to initialize the identity number of the customer; we also initialize their `name` and `address`:

```cpp
    Customer::Customer(const string& name, const string& address) 
     :m_customerId(++MaxCustomerId), 
      m_name(name), 
      m_address(address) { 
      // Empty. 
    } 
```

# Reading the customer from a file

The `read` method reads the information on a customer from the file stream:

```cpp
    void Customer::read(ifstream& inStream) { 
     inStream.read((char*) &m_customerId, sizeof m_customerId); 
```

In the same way, as in the `read` method of the `Book` class, we have to use the `getline` function instead of the input stream operator, since the input stream operator would read one word only:

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

# Writing the customer to a file

The `write` method writes information on the customer to the stream in the same way as in the `Book` class previously:

```cpp
    void Customer::write(ofstream& outStream) const { 
      outStream.write((char*) &m_customerId, sizeof m_customerId); 
      outStream << m_name << endl; 
      outStream << m_address << endl; 
```

When writing a set, we first write the size of the set, and then the individual values of the set:

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

# Borrowing and reserving a book

When a customer borrows a book, it is inserted into the loan set of the customer:

```cpp
    void Customer::borrowBook(int bookId) { 
      m_loanSet.insert(bookId); 
    } 
```

In the same way, when a customer reserves a book, it is inserted into the reservation set of the customer:

```cpp
    void Customer::reserveBook(int bookId) { 
      m_reservationSet.insert(bookId); 
    } 
```

When a customer returns or unreserves a book, it is removed from the loan set or reservation set:

```cpp
    void Customer::returnBook(int bookId) { 
      m_loanSet.erase(bookId); 
    } 

    void Customer::unreserveBook(int bookId) { 
      m_reservationSet.erase(bookId); 
    } 
```

# Displaying the customer

The output stream operator writes the name and address of the customer. If the customer has borrowed or reserved books, they are written too:

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

# The Library class

Finally, the `Library` class handles the library itself. It performs a set of tasks regarding borrowing and returning books.

**Library.h**

```cpp
class Library { 
  public: 
    Library(); 

  private: 
    static string s_binaryPath; 
```

The `lookupBook` method looks up a book by the author and title. It returns true if the book is found. If it is found, its information (an object of the `Book` class) is copied into the object pointed at by `bookPtr`:

```cpp
    bool lookupBook(const string& author, const string& title, 
                    Book* bookPtr = nullptr); 
```

In the same way, `lookupCustomer` looks up a customer by the name and address. If the customer is found, true is returned, and the information is copied into the object pointed at by `customerPtr`:

```cpp
    bool lookupCustomer(const string& name, const string& address, 
                        Customer* customerPtr = nullptr); 
```

The application of this chapter revolves around the following methods. They perform the tasks of the library system. Each of the methods will prompt the user for input and then perform a task, such as borrowing or returning a book.

The following methods perform one task each, which are looking up the information about a book or a customer, adding or deleting a book, listing the books, adding and deleting books from the library, and borrowing, reserving, and returning books:

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

The `load` and `save` methods are called at the beginning and the end of the execution:

```cpp
    void load();  
    void save(); 
```

There are two maps holding the books and the customers of the library. As mentioned previously, a map is a structure where each value is identified by a unique key in order to provide fast access. The unique identity numbers of the books and customers are the keys:

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

Between executions, the library information is stored in the `Library.bin` file on the hard drive. Note that we use two backslashes to represent one backslash in the `string`. The first backslash indicates that the character is a special character, and the second backslash states that it is a backslash:

```cpp
string Library::s_binaryPath("Library.bin"); 
```

The constructor ­loads the library, presents a menu, and iterates until the user quits. Before the execution is finished, the library is saved:

```cpp
Library::Library(void) { 
```

Before the menu is presented, the library information (books, customers, loans, and reservations) is loaded from the file:

```cpp
  load(); 
```

The while statement continues as long as `quit` is true. It remains false until the user chooses the Quit option from the menu:

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

The user inputs an integer value from the console input stream (`cin`), which is stored in `choice`:

```cpp
    int choice; 
    cin >> choice; 
```

We use a `switch` statement to perform the requested task:

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

Before the program is finished, the library information is saved:

```cpp
      save(); 
    } 
```

# Looking up books and customers

The `lookupBook` method iterates through the book map. It returns true if a book with the author and title exists. If the book exists, its information is copied to the object pointed at by the `bookPtr` parameter and true is returned, as long as the pointer is not null. If the book does not exist, false is returned, and no information is copied into the object:

```cpp
    bool Library::lookupBook(const string& author, 
        const string& title, Book* bookPtr /* = nullptr*/) { 
      for (const pair<int,Book>& entry : s_bookMap) { 
        const Book& book = entry.second; 
```

Note that `bookPtr` may be `nullptr`. In that case, only true is returned, and no information is written to the object pointed at by `bookPtr`:

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

In the same way, `lookupCustomer` iterates through the customer map and returns true, as well as copies the customer information to a `Customer` object if a customer with the name exists:

```cpp
    bool Library::lookupCustomer(const string& name, 
       const string& address, Customer* customerPtr /*=nullptr*/){ 
      for (const pair<int,Customer>& entry : s_customerMap) { 
        const Customer& customer = entry.second; 
```

Also, in this case, `customerPtr` may be `nullptr`. In that case, only true is returned. When adding a new customer, we would like to know if there already is a customer with the same name and address:

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

# Adding a book

The `addBook` method prompts the user for the name and title of the new book:

```cpp
    void Library::addBook(void) { 
      string author; 
      cout << "Author: "; 
      cin >> author; 

      string title; 
      cout << "Title: "; 
      cin >> title; 
```

If a book with the `author` and `title` already exists, an error message is displayed:

```cpp
  if (lookupBook(author, title)) { 
    cout << endl << "The book "" <<  title << "" by " 
         << author << " already exists." << endl; 
    return; 
  } 
```

If the book does not already exist, we create a new `Book` object that we add to the book map:

```cpp
  Book book(author, title); 
  s_bookMap[book.bookId()] = book; 
  cout << endl << "Added: " << book << endl; 
} 
```

# Deleting a book

The `deleteBook` method prompts the user for the author and title of the book, and deletes it if it exists:

```cpp
    void Library::deleteBook() { 
      string author; 
      cout << "Author: "; 
      cin >> author; 

      string title; 
      cout << "Title: "; 
      cin >> title; 
```

If the book does not exist, an error message is displayed:

```cpp
  Book book; 
  if (!lookupBook(author, title, &book)) { 
    cout << endl << "There is no book "" << title << "" by " 
         << "author " << author << "." << endl; 
    return; 
  } 
```

When a book is being deleted, we iterate through all customers and, for each customer, return, and unreserve the book. We do that for every book just in case the book has been borrowed or reserved by customers. In the next chapter, we will work with pointers, which allow us to return and unreserve books in a more effective manner.

Note that when we iterate through a map and obtain each `Customer` object, we need to put it back in the map after we have modified the values of its fields:

```cpp
    for (pair<int,Customer> entry : s_customerMap) { 
      Customer& customer = entry.second; 
      customer.returnBook(book.bookId()); 
      customer.unreserveBook(book.bookId()); 
      s_customerMap[customer.id()] = customer; 
    } 
```

Finally, when we have made sure the book exists, and when we have returned and unreserved it, we remove it from the book map:

```cpp
    s_bookMap.erase(book.bookId()); 
    cout << endl << "Deleted." << endl; 
  } 
```

# Listing the books

The `listBook` method is quite simple. First, we check if the book map is empty. If it is empty, we write `"No books."` If the book map is not empty, we iterate through it, and for each book, we write its information to the console output stream (`cout`):

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

# Adding a customer

The `addCustomer` method prompts the user for the `name` and `address` of the new customer:

```cpp
    void Library::addCustomer(void) { 
      string name; 
      cout << "Name: "; 
      cin >> name; 

      string address; 
      cout << "Address: "; 
      cin >> address; 
```

If a customer with the same `name` and `address` already exists, an error message is displayed:

```cpp
    if (lookupCustomer(name, address)) { 
      cout << endl << "A customer with name " << name 
           << " and address " << address << " already exists." 
           << endl; 
      return; 
    } 
```

Finally, we create a new `Customer` object that we add to the customer map:

```cpp
    Customer customer(name, address); 
    s_customerMap[customer.id()] = customer; 
    cout << endl << "Added." << endl; 
  } 
```

# Deleting a customer

The `deleteCustomer` method deletes the customer if they exist:

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

If the customer has borrowed at least one book, it must be returned before the customer can be removed:

```cpp
  if (customer.hasBorrowed()) { 
    cout << "Customer " << name << " has borrowed at least " 
         << "one book and cannot be deleted." << endl; 
    return; 
  } 
```

However, if the customer has reserved books, we just unreserve them before removing the customer:

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

# Listing the customers

The `listCustomer` method works in a way similar to `listBooks`. If there are no customers, we write `"No Customers."` If there are customers, we write them to the console output stream (`cout`):

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

# Borrowing a book

The `borrowBook` method prompts the user for the `author` and `title` of the book:

```cpp
    void Library::borrowBook(void) { 
      string author; 
      cout << "Author: "; 
      cin >> author; 

      string title; 
      cout << "Title: "; 
      cin >> title; 
```

If a book with the `author` and `title` does not exist, an error message is displayed:

```cpp
    Book book; 
    if (!lookupBook(author, title, &book)) { 
      cout << endl << "There is no book "" << title << "" by " 
           << "author " << author << "." << endl; 
      return; 
    } 
```

Also, if the `book` is already borrowed, an error message is displayed:

```cpp
    if (book.borrowed()) { 
      cout << endl << "The book "" << title << "" by " << author 
           << " has already been borrowed." << endl; 
      return; 
    } 
```

Then we prompt the user for the customer's `name` and `address`:

```cpp
  string name; 
  cout << "Customer name: "; 
  cin >> name; 

  string address; 
  cout << "Adddress: "; 
  cin >> address; 
```

If there is no `customer` with the `name` and `address`, an error message is displayed:

```cpp
    Customer customer; 
    if (!lookupCustomer(name, address, &customer)) { 
      cout << endl << "There is no customer with name " << name 
           << " and address " << address << "." << endl; 
      return; 
    } 
```

However, if the book exists and is not already borrowed, and the customer exists, we add the book to the loan set of the customer and mark the book as to be borrowed by the customer:

```cpp
    book.borrowBook(customer.id()); 
    customer.borrowBook(book.bookId()); 
```

Note that we have to put the `Book` and `Customer` objects back into their maps after we have altered them. In the next chapter, we will work with a more direct approach to pointers:

```cpp
    s_bookMap[book.bookId()] = book; 
    s_customerMap[customer.id()] = customer; 
    cout << endl << "Borrowed." << endl; 
  } 
```

# Reserving a book

The `reserveBook` method works in the same way as `borrowBook`. It prompts the user for the `author` and `title` of the book:

```cpp
    void Library::reserveBook(void) { 
      string author; 
      cout << "Author: "; 
      cin >> author; 

      string title; 
      cout << "Title: "; 
      cin >> title; 
```

Similar to the `borrowBook` case, we check that the book with the `author` and `title` exists:

```cpp
  Book book; 
  if (!lookupBook(author, title, &book)) { 
    cout << endl << "There is no book "" << title << "" by " 
         << "author " << author << "." << endl; 
    return; 
  } 
```

However, one difference compared to `borrowBook` is that the book must have been borrowed in order to be reserved. If it has not been borrowed, there is no point reserving it. In that case, the user should borrow the book instead:

```cpp
  if (!book.borrowed()) { 
    cout << endl << "The book with author " << author 
         << " and title "" << title << "" has not been " 
         << "borrowed. Please borrow the book instead." << endl; 
    return; 
  } 
```

If the book exists and has not been borrowed, we prompt the user for the `name` and `address` of the customer:

```cpp
  string name; 
  cout << "Customer name: "; 
  cin >> name; 

  string address; 
  cout << "Address: "; 
  cin >> address; 
```

If the customer does not exist, an error message is displayed:

```cpp
  Customer customer; 
  if (!lookupCustomer(name, address, &customer)) { 
    cout << endl << "No customer with name " << name 
         << " and address " << address << " exists." << endl; 
    return; 
  } 
```

Moreover, if a book has already been borrowed by the customer, we display an error message:

```cpp
    if (book.customerId() == customer.id()) { 
      cout << endl << "The book has already been borrowed by " 
           << name << "." << endl; 
      return; 
    } 
```

If the book exists and has been borrowed, but not by the customer, we add the customer to the reservation list for the book and the book to the reservation set of the customer:

```cpp
    customer.reserveBook(book.bookId()); 
    int position = book.reserveBook(customer.id()); 
```

Also, in this case, we have to put the `Book` and `Customer` objects back into their maps:

```cpp
    s_bookMap[book.bookId()] = book; 
    s_customerMap[customer.id()] = customer; 
```

Finally, we write the position of the customer in the reservation list:

```cpp
    cout << endl << position << "nd reserve." << endl; 
  } 
```

# Returning a Book

The `returnBook` method prompts the user for the author and title of the book:

```cpp
    void Library::returnBook(void) { 
      string author; 
      cout << "Author: "; 
      cin >> author; 

      string title; 
      cout << "Title: "; 
      cin >> title; 
```

If the book does not exist, an error message is displayed:

```cpp
    Book book; 
    if (!lookupBook(author, title, &book)) { 
      cout << endl << "No book "" << title 
           << "" by " << author << " exists." << endl; 
      return; 
    } 
```

If the book has not been borrowed, an error message is displayed:

```cpp
    if (!book.borrowed()) { 
      cout << endl << "The book "" << title 
           << "" by " << author 
           << "" has not been borrowed." << endl; 
      return; 
    } 
```

Unlike the methods described previously, in this case, we do not ask for the customer. Instead, we return the book and look up the book in the reservation list of each customer:

```cpp
    book.returnBook(); 
    cout << endl << "Returned." << endl; 

    Customer customer = s_customerMap[book.customerId()]; 
    customer.returnBook(book.bookId()); 
    s_customerMap[customer.id()] = customer; 
```

If the book has been reserved, we look up the first customer in the reservation list, remove them from the reservation list, and let them borrow the book:

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

# Saving the library information to a file

When saving the library information, we first open the file:

```cpp
void Library::save() { 
  ofstream outStream(s_binaryPath); 
```

If the file was correctly opened, first we write the number of books, and then we write the information for each book by calling `write` on the `Book` objects:

```cpp
  if (outStream) { 
    int numberOfBooks = s_bookMap.size(); 
    outStream.write((char*) &numberOfBooks, sizeof numberOfBooks); 

    for (const pair<int,Book>& entry : s_bookMap) { 
      const Book& book = entry.second; 
      book.write(outStream); 
    } 
```

In the same way, we write the number of customers, and then the information of each customer, by calling `write`:

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

# Loading the library information from a file

When loading the library information from a file, we use the same method we would for `read`. We start by opening the file:

```cpp
void Library::load() { 
  ifstream inStream(s_binaryPath); 
```

We read the number of books and then the information of each book:

```cpp
  if (inStream) { 
    int numberOfBooks; 
    inStream.read((char*) &numberOfBooks, sizeof numberOfBooks); 
```

For each book, we create a new `Book` object, read its information by calling `read`, and add it to the book map. We also calculate the new value of the `MaxBookId` static field by assigning it the maximum value of itself and the identity number of the book:

```cpp
    for (int count = 0; count < numberOfBooks; ++count) { 
      Book book; 
      book.read(inStream); 
      s_bookMap[book.bookId()] = book; 
      Book::MaxBookId = max(Book::MaxBookId, book.bookId()); 
    } 
```

In the same way, we read the number of customers and then the information of each customer by calling `read`:

```cpp
    int numberOfCustomers; 
    inStream.read((char*) &numberOfCustomers, 
                  sizeof numberOfCustomers); 
```

For each customer, we create a `Customer` object, read its information from the file, add it to the customer map, and calculate a new value for the `MaxCustomerId` static field:

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

# The main function

Finally, we write the `main` function, which executes the library. It is quite easy; the only thing to do is to instantiate an object of the `Library` class. Then the constructor displays the main menu:

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

# Summary

In this chapter, we built a library management system made up of the classes `Book`, `Customer`, and `Library`.

The `Book` class holds information about a book. Each `Book` object holds a unique identity number. It also keeps track of the borrower (if the book is borrowed) and a list of reservations. In the same way, the `Customer` class holds information about a customer. Similar to the book, each customer holds a unique identity number. Each `Customer` object also holds a set of borrowed and reserved books. Finally, the `Library` class provides a set of services, such as adding and removing books and customers, borrowing, returning, and reserving books, as well as displaying lists of books and customers.

In this chapter, each book and customer have a unique identity number. In the next chapter, we will look into to the library system again. However, we will omit the identity numbers and work with pointers instead.