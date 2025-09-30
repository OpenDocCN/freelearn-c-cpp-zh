# Library Management System with Pointers

In this chapter, we will continue to study a system for the management of a library. Similar to [Chapter 3](6814bf19-e75b-4083-8447-892dd8416f49.xhtml), *Building a Library Management System*, we have three classes—`Book`, `Customer`, and `Library`. However, there is one large difference: we do not work with identity numbers. Instead, we work with pointers; each `Book` object holds a pointer to the customer (an object of the `Customer` class) that has borrowed the book as well as a list of pointers to the customers that have reserved the book. In the same way, each customer holds sets of pointers for the books (objects of the `Book` class) they have borrowed and reserved.

However, this approach gives rise to a problem; we cannot store the values of pointers directly in the file. Instead, when we save the file we need to convert from pointers to indexes in the book and customer lists, and when we load the file we need to transform the indexes back to pointers. This process is called **marshmallowing**.

In this chapter, we are going to dive deeper into the following topics:

*   Just as in [Chapter 3](6814bf19-e75b-4083-8447-892dd8416f49.xhtml), *Building a Library Management System*, we will work with classes for books and customers that constitute a small database. However, in this chapter, we will work directly with pointers instead of integer numbers.
*   As we work with pointers instead of integer numbers, the file handling becomes more complicated. We need to perform a process called marshmallowing.
*   Finally, we will work with the generic standard C++ classes, `set` and `list`. However, in this chapter they hold pointers to book and customer objects instead of objects.

# The Book class

Similar to the system of the previous chapter, we have three classes: `Book`, `Customer`, and `Library`. The `Book` class keeps track of a book, where each book has an author and a title. The `Customer` class keeps track of a customer, where each customer has a name and an address. The `Library` class keeps track of the library operations, such as borrowing, returning, and reserving. Finally, the `main` function simply creates an object of the `Library` class.

The `Book` class is similar to the `Book` class of [Chapter 3](6814bf19-e75b-4083-8447-892dd8416f49.xhtml), *Building a Library Management System*. The only real difference is that there are no identity numbers, only pointers.

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

We do not have a method returning the identity number of the book, since the books in this chapter do not use identity numbers.

The `borrowedPtr` method returns the address of the customer who has borrowed the book, or `nullptr` if the book is not borrowed at the moment. It comes in two versions, where the first version returns a reference to a pointer to a `Customer` object. In that way, we can assign a new value of the pointer to the customer. The second version is constant, which means that we can call it on constant objects:

```cpp
    Customer*& borrowerPtr() { return m_borrowerPtr; } 
    const Customer* borrowerPtr() const { return m_borrowerPtr; } 
```

Note that we do not have a `borrowed` method in this chapter. We do not need it since `borrowerPtr` returns `nullptr` if the book is not borrowed at the moment.

In this chapter, `reservationPtrList` returns a list of customer pointers instead of integer values. It comes in two versions, where the first version returns a reference to the list. In that way, we can add and remove pointers from the list. The second version is constant and returns a constant list, which means it can be called on constant `Book` objects and returns a list that cannot be changed:

```cpp
    list<Customer*>& reservationPtrList() 
                     { return m_reservationPtrList; } 
    const list<Customer*> reservationPtrList() const 
                          { return m_reservationPtrList; } 
```

The output stream operator works in the same way as in [Chapter 3](6814bf19-e75b-4083-8447-892dd8416f49.xhtml), *Building a Library Management System*:

```cpp
    friend ostream& operator<<(ostream& outStream, 
          const Book& book); 
```

The `m_author` and `m_title` fields are strings similar to [Chapter 3](6814bf19-e75b-4083-8447-892dd8416f49.xhtml), *Building a Library Management System*:

```cpp
    private: 
      string m_author, m_title; 
```

However, we have omitted the `m_bookId` field, since we do not use identity numbers in this chapter. We have also replaced the `m_borrowedId` and `m_customerId` fields with `m_borrowerPtr`, which is initialized to `nullptr` since the book is not borrowed from the beginning:

```cpp
    Customer* m_borrowerPtr = nullptr; 
```

The `m_reservationPtrList` field holds a list of pointers to the customers that have reserved the book, rather than a list of integer identity numbers of [Chapter 3](6814bf19-e75b-4083-8447-892dd8416f49.xhtml), *Building a Library Manageme**nt System*:

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

The default constructor is similar to the constructor of [Chapter 3](6814bf19-e75b-4083-8447-892dd8416f49.xhtml), *Building a Library Management System*:

```cpp
    Book::Book() { 
      // Empty. 
    } 
```

The second constructor is also similar to the constructor of [Chapter 3](6814bf19-e75b-4083-8447-892dd8416f49.xhtml), *Building a Library Management System*. However, there is no `m_bookId` field to initialize:

```cpp
    Book::Book(const string& author, const string& title) 
    :m_author(author), 
    m_title(title) { 
      // Empty. 
    } 
```

# Reading and writing the book

The `read` and `write` methods have been shortened in this chapter. They only read and write the author and title of the book. The potential loan and reservation lists are read and written by the `save` and `write` methods of the `Library` class:

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

# Borrowing and reserving the book

When a customer reserves a book, the pointer to the `Customer` object is added to the reservation pointer list of the book. The size of the list is returned for the customer to be notified of their position in the reservation list:

```cpp
    int Book::reserveBook(Customer* borrowerPtr) { 
      m_reservationPtrList.push_back(borrowerPtr); 
      return m_reservationPtrList.size(); 
    } 
```

When a customer returns a book, we simply set `m_borrowerPtr` to `nullptr`, which indicates that the book is no longer borrowed:

```cpp
    void Book::returnBook() { 
      m_borrowerPtr = nullptr; 
    } 
```

The `removeReservation` method simply removes the customer pointer from the reservation list:

```cpp
    void Book::removeReservation(Customer* customerPtr) { 
      m_reservationPtrList.remove(customerPtr); 
    } 
```

# Displaying the book

The output stream operator writes the title and author, the customer that has borrowed the book (if any), and the customers that have reserved the book (if any):

```cpp
    ostream& operator<<(ostream& outStream, const Book& book) { 
     outStream << """ << book.m_title << "" by " << book.m_author; 
```

If the book is borrowed, we write the borrower to the stream:

```cpp
    if (book.m_borrowerPtr != nullptr) { 
        outStream << endl << "  Borrowed by: " 
              << book.m_borrowerPtr->name() << "."; 
    } 
```

If the reservation list of the book is not empty, we iterate through it, and for each reservation, we write the customer:

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

# The Customer class

The `Customer` class of this chapter is similar to the `Customer` class of [Chapter 3](6814bf19-e75b-4083-8447-892dd8416f49.xhtml), *Building a Library Management System*. Again, in this case, the difference is that we work with pointers instead of integer identity numbers.

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

The `borrowBook`, `returnBook`, `reserveBook`, and `unreserveBook` take a pointer to a `Book` object as the parameter:

```cpp
    void borrowBook(Book* bookPtr); 
    void returnBook(Book* bookPtr); 
    void reserveBook(Book* bookPtr); 
    void unreserveBook(Book* bookPtr); 
```

The `loadPtrSet` and `reservationPtrSet` methods return sets of `Book` pointers, rather than sets of integer identity numbers:

```cpp
    set<Book*>& loanPtrSet() { return m_loanPtrSet; } 
    const set<Book*> loanPtrSet() const { return m_loanPtrSet; } 

    set<Book*>& reservationPtrSet(){ return m_reservationPtrSet; } 
    const set<Book*> reservationPtrSet() const 
                     { return m_reservationPtrSet; } 
```

The output stream operator is unchanged, compared to [Chapter 3](6814bf19-e75b-4083-8447-892dd8416f49.xhtml), *Building a Library Managemen**t System*:

```cpp
    friend ostream& operator<<(ostream& outStream, 
                               const Customer& customer); 
```

The `m_name` and `m_address` fields store the name and address of the customer, just as in [Chapter 3](6814bf19-e75b-4083-8447-892dd8416f49.xhtml), *Building a Library Mana**gement System*:

```cpp
  private: 
    string m_name, m_address; 
```

The `m_loanPtrSet` and `m_reservationPtrSet` fields hold pointers to `Book` objects, rather than integer identity numbers:

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

The constructors are similar to the constructors of [Chapter 3](6814bf19-e75b-4083-8447-892dd8416f49.xhtml), *Building a Library Management System*. The first constructor does nothing and is called when the customer list is loaded from a file:

```cpp
    Customer::Customer() { 
     // Empty. 
    } 
```

The second constructor initializes the name and address of the customer. However, compared to the constructor of [Chapter 3](6814bf19-e75b-4083-8447-892dd8416f49.xhtml), *Building a Library Management System*, there is no `m_customerId` field to initialize:

```cpp
    Customer::Customer(const string& name, const string& address) 
    :m_name(name), 
    m_address(address) { 
       // Empty. 
    } 
```

# Reading and writing the customer

Similar to the preceding `Book` case, the `read` and `write` methods have been shortened. They only read and write the name and address. The loan and reservation sets are read and written in the `Library` class, shown as follows:

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

# Borrowing and reserving a book

The `borrowBook` method adds the book pointer to the loan set and removes it from the reservation set in case it was reserved:

```cpp
    void Customer::borrowBook(Book* bookPtr) { 
      m_loanPtrSet.insert(bookPtr); 
      m_reservationPtrSet.erase(bookPtr); 
    } 
```

The `reserveBook` method simply adds the book pointer to the reservation list, and `returnBook` and `unreserveBook` remove the book pointer from the loan and reservation sets:

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

# Displaying the customer

The output stream operator works in the same way as in [Chapter 3](6814bf19-e75b-4083-8447-892dd8416f49.xhtml), *Building a Library Management System*. It writes the name and address of the customer, as well as the sets of borrowed and reserved books (if any):

```cpp
    ostream& operator<<(ostream& outStream, const Customer& customer){ 
      outStream << customer.m_name << ", " 
      << customer.m_address << "."; 
```

If the loan list of the customer is not empty, we iterate through it, and for each loan, we write the book:

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

In the same way, if the reservation list of the customer is not empty, we iterate through it, and for each reservation, we write the book:

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

# The Library class

The `Library` class is quite similar to its counterpart in  [Chapter 3](6814bf19-e75b-4083-8447-892dd8416f49.xhtml), *Building a Library Management System*. However, we have added lookup `methods` to transform between pointers and list indexes when saving and loading the library information to a file:

**Library.h:**

```cpp
class Library { 
  public: 
    Library(); 
```

The destructor deallocates all the dynamically allocated memory of this application:

```cpp
    ~Library(); 

    private: 
      static string s_binaryPath; 
```

The `lookupBook` and `lookupCustomer` methods return pointers to `Book` and `Customer` objects. If the book or customer does not exist, `nullptr` is returned:

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

The `lookupBookIndex` and `lookupCustomerIndex` methods take a pointer, search the book and customer lists after the object pointed at, and return its index in the lists:

```cpp
    int lookupBookIndex(const Book* bookPtr); 
    int lookupCustomerIndex(const Customer* customerPtr);
```

The `lookupBookPtr` and `lookupCustomerPtr` methods take an index and return a pointer to the object at the position in the book and customer lists:

```cpp
    Book* lookupBookPtr(int bookIndex); 
    Customer* lookupCustomerPtr(int customerIndex); 
```

The `save` and `write` methods save and load the library information from a file. However, they are more complicated than their counterparts in [Chapter 3](6814bf19-e75b-4083-8447-892dd8416f49.xhtml), *Building a Library Management System*:

```cpp
    void save(); 
    void load(); 
```

The `m_bookPtrList` and `m_customerPtrList` fields hold pointers to `Book` and `Customer` objects, rather than the objects themselves, as in  [Chapter 3](https://cdp.packtpub.com/c___by_example/wp-admin/post.php?post=47&action=edit#post_43), *Building a Library Management System*:

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

The constructor is identical to the constructor of  [Chapter 3](https://cdp.packtpub.com/c___by_example/wp-admin/post.php?post=47&action=edit#post_43), *Building a Library Management System*:

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

# Looking up books and customers

The `lookupBook` method of this chapter searches for the `Book` object with the author and title, in a way similar to [Chapter 3](https://cdp.packtpub.com/c___by_example/wp-admin/post.php?post=47&action=edit#post_43), *Building a Library Management System*. However, if it finds a `Book` object that matches the author and title, it does not copy the information to a given object. Instead, it simply returns a pointer to the object. If it does not find the `Book` object, `nullptr` is returned:

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

In the same way, `lookupCustomer` tries to find a `Customer` object that matches the name and address. If it finds the object, its pointer is returned. If it does not find it, `nullptr` is returned:

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

# Adding a book

The `addBook` method prompts the user for the author and the title:

```cpp
    void Library::addBook() { 
      string author; 
      cout << "Author: "; 
      cin >> author; 

     string title; 
     cout << "Title: "; 
     cin >> title; 
```

When checking if the book already exists, we call `lookupBook`. If the book exists, a pointer to the `Book` object is returned. If the book does not exist, `nullptr` is returned. Therefore, we test whether the return value does not equal `nullptr`. If it does not equal `nullptr`, the book already exists and an error message is displayed:

```cpp
      if (lookupBook(author, title) != nullptr) { 
        cout << endl << "The book "" << title << "" by " 
          << author << " already exists." << endl; 
        return; 
      } 
```

When adding the book, we dynamically create a new `Book` object with the `new` operator. We use the standard C++ `assert` macro to check that the book pointer is not null. If it is null, the execution will be aborted with an error message:

```cpp
      Book* bookPtr = new Book(author, title); 
      assert(bookPtr != nullptr); 
      m_bookPtrList.push_back(bookPtr); 
      cout << endl << "Added." << endl; 
    } 
```

# Deleting a book

The `deleteBook` method deletes a book from the library by prompting the user about the author and title of the book. If the book exists, we return, unreserve, and delete it:

```cpp
     void Library::deleteBook() { 
       string author; 
       cout << "Author: "; 
       cin >> author; 

       string title; 
       cout << "Title: "; 
       cin >> title; 
```

We obtain a pointer to the `Book` object by calling `lookupBook`:

```cpp
     Book* bookPtr = lookupBook(author, title); 
```

If the pointer is `nullptr`, the book does not exist and an error message is displayed:

```cpp
       if (bookPtr == nullptr) { 
         cout << endl << "The book "" << title << "" by " 
           << author << " does not exist." << endl; 
         return; 
       } 
```

We check whether the book has been borrowed by looking up the borrower:

```cpp
        Customer* borrowerPtr = bookPtr->borrowerPtr(); 
```

If the pointer returned by `borrowerPtr` is not `nullptr`, we return the book by calling `returnBook` of the borrower. In that way, the book is no longer registered as borrowed by the customer:

```cpp
  if (borrowerPtr != nullptr) { 
    borrowerPtr->returnBook(bookPtr); 
  } 
```

Moreover, we need to check whether the book has been reserved by any other customer. We do so by obtaining the reservation list of the book and, for every customer in the list, we unreserve the book:

```cpp
    list<Customer*> reservationPtrList = 
      bookPtr->reservationPtrList(); 
```

Note that we do not check whether the book has actually been reserved by the customer, we simply unreserve the book. Also note that we do not need to put back any object to the list, since we work with pointers to objects and do not copy objects:

```cpp
      for (Customer* reserverPtr : reservationPtrList) { 
        reserverPtr->unreserveBook(bookPtr); 
      }
```

When removing the book, we remove the book pointer from the book pointer list, and then deallocate the `Book` object. It may seem strange that we first display the message and then delete the book pointer. However, it has to be in that order. After we have deleted the object, we can do nothing with it. We cannot delete the object and then write it, it would cause memory errors:

```cpp
      m_bookPtrList.remove(bookPtr); 
        n cout << endl << "Deleted:" << bookPtr << endl; 
        delete bookPtr; 
      } 
```

# Listing the books

When listing the books, we first check whether the list is empty. If it is empty, we simply write `"No books."`:

```cpp
    void Library::listBooks() { 
      if (m_bookPtrList.empty()) { 
       cout << "No books." << endl; 
       return; 
      } 
    }
```

However, if the list is not empty, we iterate through the book pointer list and, for each book pointer, dereference the pointer and write the information:

```cpp
      for (const Book* bookPtr : m_bookPtrList) { 
        cout << (*bookPtr) << endl; 
        } 
    } 
```

# Adding a customer

The `addCustomer` method prompts the user for the name and address of the customer:

```cpp
    void Library::addCustomer() { 
      string name; 
       cout << "Name: "; 
       cin >> name; 

       string address; 
       cout << "Address: "; 
       cin >> address;
```

If a customer with the name and address already exists, an error message is displayed:

```cpp
      if (lookupCustomer(name, address) != nullptr) { 
        cout << endl << "A customer with name " << name 
         << " and address " << address << " already exists." 
         << endl; 
       return; 
      } 
```

When adding the customer, we dynamically create a new `Customer` object that we add to the customer object pointer list:

```cpp
      Customer* customerPtr = new Customer(name, address); 
        assert(customerPtr != nullptr); 
        m_customerPtrList.push_back(customerPtr); 
        cout << endl << "Added." << endl; 
      } 
```

# Deleting a customer

When deleting a customer, we look them up and display an error message if they do not exist:

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

If the customer with the given name and address does not exist, an error message is displayed. Consider the following code:

```cpp
    if (customerPtr == nullptr) { 
      cout << endl << "Customer " << name 
         << " does not exists." << endl; 
      return; 
    }
```

If the customer has borrowed at least one book, they cannot be deleted, and an error message is displayed, which is shown as follows:

```cpp
     if (!customerPtr->loanPtrSet().empty()) { 
      cout << "The customer " << customerPtr->name() 
         << " has borrowed books and cannot be deleted." << endl; 
      return; 
     } 
```

However, if the customer has not borrowed any books, the customer is first removed from the reservation list of every book in the library, shown in the following code:

```cpp
     for (Book* bookPtr : m_bookPtrList) { 
       bookPtr->removeReservation(customerPtr); 
     } 
```

Then the customer is removed from the customer list, and the `Customer` object is deallocated by the `delete` operator. Again, note that we first must write the customer information, and then delete its object. The other way around would not have worked since we cannot inspect a deleted object. That would have caused memory errors:

```cpp
      m_customerPtrList.remove(customerPtr); 
      cout << endl << "Deleted." << (*customerPtr) << endl; 
      delete customerPtr; 
    } 
```

# Listing the customers

When listing the customer, we go through the customer list and, for each customer, dereference the `Customer` object pointer and write the information of the object:

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

# Borrowing a book

When borrowing a book, we start by prompting the user for the author and title, which is shown in the following code snippet:

```cpp
    void Library::borrowBook() { 
      string author; 
      cout << "Author: "; 
      cin >> author; 

      string title; 
      cout << "Title: "; 
      cin >> title; 
```

We look up the book and if the book does not exist, an error message is displayed, which is shown in the following code:

```cpp
     Book* bookPtr = lookupBook(author, title); 

     if (bookPtr == nullptr) { 
       cout << endl << "There is no book "" << title 
         << "" by " << author << "." << endl; 
       return; 
     } 
```

If the book has already been borrowed by another customer, it cannot be borrowed again:

```cpp
    if (bookPtr->borrowerPtr() != nullptr) { 
      cout << endl << "The book "" << title << "" by " << author  
         << " has already been borrowed." << endl; 
      return; 
    } 
```

We prompt the user for the name and address of the customer:

```cpp
     string name; 
     cout << "Customer name: "; 
     cin >> name; 

     string address; 
     cout << "Address: "; 
     cin >> address; 

     Customer* customerPtr = lookupCustomer(name, address);
```

If there is no customer with the given name and address, an error message is displayed:

```cpp
     if (customerPtr == nullptr) { 
      cout << endl << "No customer with name " << name 
         << " and address " << address << " exists."  << endl; 
      return; 
     } 
```

Finally, we add the book to the customer's loan set and we mark the customer as the borrower of the book:

```cpp
     bookPtr->borrowerPtr() = customerPtr; 
     customerPtr->borrowBook(bookPtr); 
     cout << endl << "Borrowed." << endl; 
   } 
```

# Reserving a book

The reservation process is similar to the preceding borrowing process. We prompt the user for the author and title of the book, as well as the name and address of the customer, which is shown as follows:

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

If the book does not exist, an error message is displayed:

```cpp
    if (bookPtr == nullptr) { 
      cout << endl << "There is no book "" << title 
         << "" by " << author << "." << endl; 
      return; 
    } 
```

If the book has not been borrowed, it is not possible to reserve it. Instead, we encourage the user to borrow the book:

```cpp
     if (bookPtr->borrowerPtr() == nullptr) { 
       cout << endl << "The book "" << title << "" by " 
         << author << " has not been not borrowed. " 
         << "Please borrow the book instead of reserving it." 
         << endl; 
      return; 
    } 
```

We prompt the user for the name and address of the customer:

```cpp
     string name; 
     cout << "Customer name: "; 
     cin >> name; 

     string address; 
     cout << "Address: "; 
     cin >> address; 

     Customer* customerPtr = lookupCustomer(name, address); 
```

If the customer does not exist, an error message is displayed:

```cpp
     if (customerPtr == nullptr) { 
      cout << endl << "There is no customer with name " << name 
         << " and address " << address << "." << endl; 
      return; 
     } 
```

If the customer has already borrowed the book, they cannot also reserve the book:

```cpp
     if (bookPtr->borrowerPtr() == customerPtr) { 
      cout << endl << "The book has already been borrowed by " 
         << name << "." << endl; 
      return; 
     } 
```

Finally, we add the customer to the reservation list of the book and we add the book to the reservation set of the customer. Note that there is a list of reservation customers for the book, while there is a set of reserved books for the customer. The reason for this is that when a book is returned, the first customer in the reservation list borrows the book. There are no such restrictions when it comes to a set of reservations for a customer:

```cpp
     int position = bookPtr->reserveBook(customerPtr); 
     customerPtr->reserveBook(bookPtr); 
```

We notify the customer of its position on the reservation list:

```cpp
     cout << endl << position << "nd reserve." << endl; 
     }
```

# Returning a book

When returning a book, we prompt the user for its author and title. However, we do not ask for the customer who has borrowed the book. That information is already stored in the `Book` object:

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

If the book with the given author and title does not exist, an error message is displayed:

```cpp
     if (bookPtr == nullptr) { 
      cout << endl << "There is no book "" << title << "" by " 
         << author << "." << endl; 
      return; 
     } 

     Customer* customerPtr = bookPtr->borrowerPtr(); 
```

If the customer with the given name and address does not exist, an error message is displayed:

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

When we have returned the book, we need to find out whether any customer has reserved it:

```cpp
     list<Customer*>& reservationPtrList = 
       bookPtr->reservationPtrList();
```

If there is at least one customer in the reservation list of the book, we obtain that customer, remove them from the reservation list of the book, mark the customer as the borrower of the book, and add the book to the loan set of the customer:

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

# Looking up books and customers

When saving and loading the library information from a file, we need to transform between pointers to `Book` and `Customer` objects and indexes in the book and customer lists. The `lookupIndex` method takes a pointer to a `Book` object and returns its index in the book list:

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

If we reach this point, the execution is aborted with an error message by the `assert` macro. However, we should not reach this point, since the `Book` pointer should be in the book pointer list:

```cpp
     assert(false); 
      return -1; 
     }
```

The `lookupBookPtr` method performs the opposite task. It finds the `Book` object pointer at the position given by `bookIndex` in the book pointer list. The `assert` macro aborts the execution with an error message if the index is outside the scope of the list. However, that should not happen since all indexes shall be within the scope:

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

The `lookupCustomerIndex` method gives the index of the `Customer` pointer in the customer pointer list, in the same way as shown in the preceding `lookupBookIndex` method:

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

The `lookupCustomerPtr` method looks up the index of the `Customer` pointer in the customer pointer list in the same way as shown in the preceding `lookupBookPtr` method:

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

The `save` and `load` methods of the `Library` class of this chapter are a bit more complicated than their counterparts in [Chapter 3](6814bf19-e75b-4083-8447-892dd8416f49.xhtml), *Building a Library Management System*. The reason for this is that we cannot save pointers directly, since a pointer holds a memory address that can be changed between executions. Instead, we need to save their indexes to the file. The process of transforming pointers to indexes and indexes to pointers is called **marshmallowing**. When saving the library, we divide the saving process into several steps:

*   Saving the book list:  At this point, we save the author and title only.
*   Saving the customer list:  At this point, we save the name and address only.
*   For each book: Save the borrower (if the book is borrowed) and the (possibly empty) reservation list. We save the customer list indexes, rather than the pointers to the customers.
*   For each customer, we save the loan and reservation sets. We save the book list indexes, rather than the pointers to the books.

# Saving the library information to a file

The `Save` method opens the file and, if it was successfully opened, reads the books and customers of the library:

```cpp
    void Library::save() { 
      ofstream outStream(s_binaryPath); 
```

# Writing the book objects

We save the book objects. We only save the author and title of the books by calling `write` for each `Book` object. We do not save the potential borrower and reservation list at this point.

We start by writing the number of books in the list to the file:

```cpp
    if (outStream) { 
      { int bookPtrListSize = m_bookPtrList.size(); 
         outStream.write((char*) &bookPtrListSize, 
           sizeof bookPtrListSize); 
```

Then we write the information of each book to the file by calling `write` on each `Book` object pointer:

```cpp
      for (const Book* bookPtr : m_bookPtrList) { 
        bookPtr->write(outStream); 
    } 
   } 
```

# Writing the customer objects

We save the customer objects. Similar to the preceding book case, we only save the name and address of the customers by calling `write` for each `Customer` object. We do not save sets of borrowed and reserved books at this point.

In the same way, as in the preceding book case, we start by writing the number of customers on the list to the file:

```cpp
    { int customerPtrListSize = m_customerPtrList.size(); 
      outStream.write((char*) &customerPtrListSize, 
                      sizeof customerPtrListSize); 
```

Then we write the information of each customer to the file by calling the `write` method on each `Customer` object pointer:

```cpp
      for (const Customer* customerPtr : m_customerPtrList) { 
        customerPtr->write(outStream); 
      } 
    } 
```

# Writing the borrower index

For each `Book` object, if the book is borrowed we look up and save the index of the `Customer`, rather than the pointer to the object:

```cpp
    for (const Book* bookPtr : m_bookPtrList) { 
      { const Customer* borrowerPtr = bookPtr->borrowerPtr(); 
```

For each book, we start by checking if it has been borrowed. If it has been borrowed, we write the value `true` to the file, to indicate that it is borrowed:

```cpp
        if (borrowerPtr != nullptr) { 
          bool borrowed = true; 
          outStream.write((char*) &borrowed, sizeof borrowed); 
```

Then we look up the index of the customer that has borrowed the book in the customer pointer list and write the index to the file:

```cpp
          int loanIndex = lookupCustomerIndex(borrowerPtr); 
          outStream.write((char*) &loanIndex, sizeof loanIndex); 
        } 
```

If the book is not borrowed, we just write the value `false` to the file, to indicate that the book has not been borrowed:

```cpp
        else { 
          bool borrowed = false; 
          outStream.write((char*) &borrowed, sizeof borrowed); 
        } 
      } 
```

# Writing the reservation indexes

As a book can be reserved for more than one customer, we iterate through the list of reservations and save the index of each customer in the reservation list:

```cpp
      { const list<Customer*>& reservationPtrList = 
          bookPtr->reservationPtrList(); 
```

For each book, we start by writing the number of reservations of the book to the file:

```cpp
        int reserveSetSize = reservationPtrList.size(); 
        outStream.write((char*) &reserveSetSize, 
                        sizeof reserveSetSize); 
```

Then we iterate through the reservation list and, for each reservation, we look up and write the index of each customer that reserved the book:

```cpp
        for (const Customer* customerPtr : reservationPtrList) { 
          int customerIndex = lookupCustomerIndex(customerPtr); 
          outStream.write((char*) &customerIndex, 
                          sizeof customerIndex); 
        } 
      } 
    }
```

# Writing the loan book indexes

For each customer, we save the indexes of the books they have borrowed. First, we save the size of the loan list and then the book indexes:

```cpp
    for (const Customer* customerPtr : m_customerPtrList) { 
      { const set<Book*>& loanPtrSet = 
          customerPtr->loanPtrSet(); 
```

For each customer, we start by writing the number of loans to the file:

```cpp
        int loanPtrSetSize = loanPtrSet.size(); 
        outStream.write((char*) &loanPtrSetSize, 
                        sizeof loanPtrSetSize); 
```

Then we iterate through the loan set and, for each loan, we look up and write the index of each book to the file:

```cpp
        for (const Book* customerPtr : loanPtrSet) { 
          int customerIndex = lookupBookIndex(customerPtr); 
          outStream.write((char*) &customerIndex, 
                          sizeof customerIndex); 
        } 
      } 
```

# Writing the reservation book indexes

In the same way, for each customer, we save the indexes of the books they have reserved. First, we save the size of the reservation list and then the indexes of the books they reserved:

```cpp
      { const set<Book*>& reservedPtrSet = 
          customerPtr->reservationPtrSet(); 
```

For each customer, we start by writing the number of reserved books to the file:

```cpp
        int reservationPtrSetSize = reservationPtrSet.size(); 
        outStream.write((char*) &reservationPtrSetSize, 
                        sizeof reservationPtrSetSize); 
```

Then we iterate through the reservation set and, for each reservation, we look up and write the index of each book to the file:

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

# Loading the library information from a file

When loading the file, we proceed in the same manner as when we saved the file:

```cpp
    void Library::load() { 
      ifstream inStream(s_binaryPath); 
```

# Reading the book objects

We read the size of the book list, and then the books themselves. Remember that we have so far read the author and title of the books only:

```cpp
    if (inStream) { 
      { int bookPtrListSize; 
```

We start by reading the number of books:

```cpp
        inStream.read((char*) &bookPtrListSize, 
                    sizeof bookPtrListSize); 
```

Then we read the books themselves. For each book, we dynamically allocate a `Book` object, read its information by calling `read` on the pointer, and add the pointer to the book pointer list:

```cpp
      for (int count = 0; count < bookPtrListSize; ++count) { 
        Book *bookPtr = new Book(); 
        assert(bookPtr != nullptr); 
        bookPtr->read(inStream); 
        m_bookPtrList.push_back(bookPtr); 
      } 
    }
```

# Reading the customer objects

In the same way, we read the size of the customer list and then the customers themselves. Up until this point, we read the name and address of the customers only:

```cpp
    { int customerPtrListSize; 
```

We start by reading the number of customers:

```cpp
      inStream.read((char*) &customerPtrListSize, 
                    sizeof customerPtrListSize); 
```

Then we read the customers themselves. For each customer, we dynamically allocate a `Customer` object, read its information by calling `read` on the pointer, and add the pointer to the book pointer list:

```cpp
      for (int count = 0; count < customerPtrListSize; ++count) { 
        Customer *customerPtr = new Customer(); 
        assert(customerPtr != nullptr); 
        customerPtr->read(inStream); 
        m_customerPtrList.push_back(customerPtr); 
      } 
    } 
```

# Reading the borrower index

For each book, we read the customers that have borrowed it (if any) and the list of customers that have reserved the book:

```cpp
    for (Book* bookPtr : m_bookPtrList) { 
      { bool borrowed; 
        inStream.read((char*) &borrowed, sizeof borrowed); 
```

If `borrowed` is `true`, the book has been borrowed. In that case, we read the index of the customer. We then look up the pointer of the `Customer` object, which we add to the reservation list of the book:

```cpp
        if (borrowed) { 
          int loanIndex; 
          inStream.read((char*) &loanIndex, sizeof loanIndex); 
          bookPtr->borrowerPtr() = lookupCustomerPtr(loanIndex); 
        }
```

If `borrowed` is `false`, the book has not been borrowed. In that case, we set the pointer to the customer that has borrowed the book to `nullptr`:

```cpp
        else { 
          bookPtr->borrowerPtr() = nullptr; 
        } 
      } 
```

# Reading the reservation indexes

For each book, we also read the reservation list. First, we read the size of the list and then the customer indexes themselves:

```cpp
      { list<Customer*>& reservationPtrList = 
          bookPtr->reservationPtrList(); 
        int reservationPtrListSize; 
```

We start by reading the number of reservations of the book:

```cpp
        inStream.read((char*) &reservationPtrListSize, 
                      sizeof reservationPtrListSize); 
```

For each reservation, we read the index of the customer and call `lookupCustomerPtr` to obtain the pointer to the `Customer` object, which we add to the reservation pointer list of the book:

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

# Reading the loan book indexes

For each customer, we read the set of borrowed books:

```cpp
    for (Customer* customerPtr : m_customerPtrList) { 
      { set<Book*>& loanPtrSet = customerPtr->loanPtrSet(); 
        int loanPtrSetSize = loanPtrSet.size();
```

We start by reading the size of the loan list:

```cpp
        inStream.read((char*) &loanPtrSetSize, 
                      sizeof loanPtrSetSize); 
```

For each loan, we read the index of the book and call `lookupBookPtr` to obtain the pointer to the `Book` object, which we add to the loan pointer list:

```cpp
        for (int count = 0; count < loanPtrSetSize; ++count) { 
          int bookIndex; 
          inStream.read((char*) &bookIndex, sizeof bookIndex); 
          Book* bookPtr = lookupBookPtr(bookIndex); 
          loanPtrSet.insert(bookPtr); 
        } 
      } 
```

# Reading the reservation book indexes

In the same way, for each customer, we read the set of reserved books:

```cpp
      { set<Book*>& reservationPtrSet = 
          customerPtr->reservationPtrSet(); 
```

We start by reading the size of the reservation list:

```cpp
        int reservationPtrSetSize = reservationPtrSet.size(); 
        inStream.read((char*) &reservationPtrSetSize, 
                      sizeof reservationPtrSetSize); 
```

For each reservation, we read the index of the book and call `lookupBookPtr` to obtain the pointer to the `Book` object, which we add to the reservation pointer list:

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

# Deallocating memory

Since we have added dynamically allocated `Book` and `Customer` objects to the lists, we need to deallocate them at the end of the execution. The destructor iterates through the book and customer pointer lists and deallocates all the book and customer pointers:

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

# The main function

Similar to [Chapter 3](6814bf19-e75b-4083-8447-892dd8416f49.xhtml), *Building a Library Management System*, the `main` function simply creates a `Library` object:

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

# Summary

In this chapter, we built a library management system similar to the system of [Chapter 3](6814bf19-e75b-4083-8447-892dd8416f49.xhtml), *Building a Library Management System*. However, we omitted all integer identity numbers and replaced them with pointers. This gives us the advantage that we can store loans and reservations more directly, but it also makes it harder for us to save and load them into a file.

In [Chapter 5](411aae8c-9215-4315-8a2e-882bf028834c.xhtml), *Qt Graphical Applications*, we will look at graphical applications.