# *Chapter 5*

# Standard Library Containers and Algorithms

## Chapter Objectives

By the end of this chapter, you will be able to:

*   Explain what iterators are
*   Demonstrate the use of sequential containers, container adaptors, and associative containers
*   Understand and use unconventional containers
*   Explain cases of iterator invalidation
*   Discover algorithms implemented in the standard library
*   Use user-defined operations on algorithms with lambda expressions

## Introduction

The core of C++ is its **Standard Template Library** (**STL**), which represents a set of important data structures and algorithms that facilitates the programmer's task and improves code efficiency.

The components of the STL are parametric so that they can be reused and combined in different ways. The STL is mainly made up of container classes, iterators, and algorithms.

Containers are used to store collections of elements of a certain type. Usually, the type of the container is a template parameter, which allows the same container class to support arbitrary elements. There are several container classes, each of them with different characteristics and features.

Iterators are used to traverse the elements of a container. Iterators offer the programmer a simple and common interface to access containers of a different type.

Iterators are similar to raw pointers, which can also iterate through elements using the increment and the decrement operators, or can access a specific element using the de-reference (`*`) operator.

Algorithms are used to perform standard operations on the elements stored in the containers. They use iterators to traverse the collections, since their interface is common to all the containers, so that the algorithm can be agnostic about the container it's operating on.

Algorithms treat functions as parameters that are provided by the programmer in order to be more flexible in the operation that's being performed. It is common to see an algorithm applied to a container of objects of a user-defined type. This algorithm, to execute correctly, needs to know how to treat the object in detail. For this reason, the programmer provides a function to the algorithm to specify the operations to be executed on the objects.

## Sequence Containers

**Sequence containers,** sometimes referred to as **sequential containers**, are a particular class of containers where the order in which their elements are stored is decided by the *programmer* rather than by the values of the elements. Every element has a certain position that is independent of its value.

The STL contains five sequence container classes:

![Figure 5.1: Table representing the sequence container class and their description](img/C11557_05_01.jpg)

###### Figure 5.1: Table presenting the sequence container classes and their descriptions

### Array

The array container is a fixed-size data structure of contiguous elements. It recalls the static array that we saw in *Chapter 1*, *Getting Started*:

![Figure 5.2: Array elements are stored in contiguous memory](img/C11557_05_02.jpg)

###### Figure 5.2: Array elements are stored in contiguous memory

An array's size needs to be specified at compile time. Once defined, the size of the array *cannot be changed*.

When an array is created, the `size` elements it contains are initialized next to each other in memory. While elements cannot be added or removed, their values can be modified.

Arrays can be randomly accessed using the access operator with the corresponding element's index. To access an element at a given position, we can use the operator `[]` or the `at()` member function. The former does not perform any range checks, while the latter throws an exception if the index is out of range. Moreover, the first and the last element can be accessed using the `front()` and `back()` member functions.

These operations are fast: since the elements are contiguous, we can compute the position in memory of an element given its position in the array, and access that directly.

The size of the array can be obtained using the `size()` member function. Whether the container is empty can be checked using the `empty()` function, which returns *true* if `size()` is zero.

The array class is defined in the `<array>` header file, which has to be included before usage.

### Vector

The vector container is a data structure of contiguous elements whose size can be dynamically modified: it does not require to specify its size at creation time:

![Figure 5.3: Vector elements are contiguous, and their size can grow dynamically](img/C11557_05_03.jpg)

###### Figure 5.3: Vector elements are contiguous, and their size can grow dynamically

The `vector` class is defined in the `<vector>` header file.

A vector stores the elements it contains in a single section of memory. Usually, the section of memory has enough space for more elements than the number of elements stored in the vector. When a new element is added to the vector, if there is enough space in the section of memory, the element is added after the last element in the vector. If there isn't enough space, the vector gets a new, bigger section of memory and copies all the existing elements into the new section of memory, then it deletes the old section of memory. To us, it will seem like the size of the section of memory has increased:

![Figure 5.4: Memory allocation of vectors](img/C11557_05_04.jpg)

###### Figure 5.4: Memory allocation of vectors

When the vector is created, it is empty.

Most of the interface is similar to the array's, but with a few differences.

Elements can be appended using the `push_back()` function or inserted at a generic position using the `insert()` function. The last element can be removed using `pop_back()` or at a generic position using the `erase()` function.

Appending or deleting the last element is fast, while inserting or removing other elements of the vector is considered slow, as it requires moving all the elements to make space for the new element or to keep all the elements contiguous:

![Figure 5.5: Elements being moved during insertions or deletions inside a vector](img/C11557_05_05.jpg)

###### Figure 5.5: Elements being moved during insertions or deletions inside a vector

Vectors, just like arrays, allow efficient access of elements at random positions. A vector's size is also retrieved with the `size()` member function, but this should not be confused with `capacity()`. The former is the actual number of elements in the vector, and the latter returns the maximum number of elements that can be inserted in the current section of memory.

For example, in the preceding diagram, initially, the array had a size of 4 and a capacity of 8\. So, even when an element had to be moved to the right, the vector's capacity did not change, as we never had to get a new, bigger section of memory to store the elements.

The operation of getting a new section of memory is called reallocation. Since reallocation is considered an expensive operation, it is possible to *reserve* enough memory for a given number of elements by enlarging a vector's capacity using the `reserve()` member function. The vector's capacity can also be reduced to fit the number of elements using the `shrink_to_fit()` function in order to release memory that is not needed anymore.

#### Note

Vector is the most commonly used container for a sequence of elements and is often the best one performance-wise.

Let's look at the following example to understand how `vector::front()` and `vector::back()` work in C++:

```cpp
#include <iostream>
#include <vector>
// Import the vector library
int main()
{
  std::vector<int> myvector;
  myvector.push_back(100);
  // Both front and back of vector contains a value 100
  myvector.push_back(10);
  // Now, the back of the vector holds 10 as a value, the front holds 100
  myvector.front() -= myvector.back();
  // We subtracted front value with back
std::cout << "Front of the vector: " << myvector.front() << std::endl;
std::cout << "Back of the vector: " << myvector.back() << std::endl;
}
Output:
Front of the vector: 90
Back of the vector: 10
```

### Deque

The *deque* container (pronounced *deck)* is short for "double-ended queue." Like *vector*, it allows for fast, direct access of deque elements and fast insertion and deletion at the back. Unlike *vector*, it also allows for fast insertion and deletion at the front of the deque:

![Figure 5.6: Deque elements can be added and removed at the start and the end](img/C11557_05_06.jpg)

###### Figure 5.6: Deque elements can be added and removed at the start and the end

The `deque` class is defined in the `<deque>` header file.

*Deque* generally requires more memory than *vector*, and *vector* is more performant for accessing the elements and `push_back`, so unless it is required to insert at the front, *vector* is usually preferred.

### List

The list container is a data structure of nonadjacent elements that can be dynamically grown:

![Figure 5.7: List elements are stored in different sections of memory, and have connecting links ](img/C11557_05_07.jpg)

###### Figure 5.7: List elements are stored in different sections of memory, and have connecting links

The `list` class is defined in the `<list>` header file.

Each element in the list has its memory segment and a link to its predecessor and its successor. The structure containing the element, which is the link to its predecessor and to its successor, is called a **node**.

When an element is inserted in a list, the predecessor node needs to be updated so that its successor link points to the new element. Similarly, the successor node needs to be updated so that its predecessor link points to the new element:

![Figure 5.8: C is to be inserted between A and B. A's successor and B's predecessor link must be updated to point to C (orange). C's link to the predecessor and successor are updated to points A and B (green)](img/C11557_05_08.jpg)

###### Figure 5.8: C is to be inserted between A and B. A's successor and B's predecessor link must be updated to point to C (orange). C's link to the predecessor and successor are updated to points A and B (green)

When an element is removed from the list, we need to update the successor link of the predecessor node to point to the successor of the removed node. Similarly, the predecessor link of the successor node needs to be updated to point to the predecessor of the removed node.

In the preceding diagram, if we were to remove **C**, we would have to update **A**'s successor to point to **C**'s successor (**B**), and **B**'s predecessor to point to **C**'s predecessor (**A**).

Unlike vectors, lists do not provide random access. Elements are accessed by linearly following the chain of elements: starting from the first, we can follow the successor link to find the next node, or from the last node we can follow the predecessor link to find the previous node, until we reach the element we are interested into.

The advantage of `list` is that insertion and removal are fast at any position, if we already know the node at which we want to insert or remove. The disadvantage of this is that getting to a specific node is slow.

The interface is similar to a vector, except that lists don't provide `operator[]`.

### Forward-List

The `forward_list` container is similar to the list container, with the difference that its nodes only have the link to the successor. For this reason, it is not possible to iterate over a `forward_list` in backward order:

![Figure 5.9: Forward-list elements are like List, but only have one-way connecting links](img/C11557_05_09.jpg)

###### Figure 5.9: Forward-list elements are like List elements, but only have one-way connecting links

As usual, the `forward_list` class is defined in the `<forward_list>` header file.

The `forward_list` class does not even provide `push_back()` or `size()`. Inserting an element is done using `insert_after()`, which is a variation of the `insert()` function, where the new element is inserted after the provided position. The same idea applies to element removal, which is done through `erase_after()`, which removes the element after the provided position.

### Providing Initial Values to Sequence Containers

All the sequence containers we have looked at are empty when they are first created.

When we want to create a container containing some elements, it can be repetitive to call the `push_back()` or `insert()` functions repeatedly for each element.

Fortunately, all the containers can be initialized with a sequence of elements when they are created.

The sequence must be provided in curly brackets, and the elements need to be comma-separated. This is called an initializer list:

```cpp
#include <vector>
int main()
{
    // initialize the vector with 3 numbers
    std::vector<int> numbers = {1, 2, 3};
}
```

This works for any of the containers we have seen in this chapter.

### Activity 19: Storing User Accounts

We want to store the account balance, stored as an `int` instance, for 10 users. The account balance starts with 0\. We then want to increase the balance of the first and last user by 100.

These steps will help you complete the activity:

1.  Include the header for the `array` class.
2.  Declare an integer array of ten elements.
3.  Initialize the array using the `for` loop. The `size()` operator to evaluate the size of the array and the `operator[]` to access every position of the array.
4.  Update the value for the first and last user.

    #### Note

    The solution for this activity can be found on page 304.

Now let’s do the same using a vector:

1.  Include the **vector** header.
2.  Declare a vector of integer type and reserve memory to store 100 users with resize it to be able to contain 10 users.
3.  Use a for loop to initialize the vector.

With this activity, we learned how we can store an arbitrary number of accounts.

## Associative Containers

`operator<`, although the user can supply a `Functor` (function object) as a parameter to specify how the elements should be compared. The `<functional>` header contains many such objects that can be used to sort the associative containers, like `std::less` or `std::less`.

![Figure 5.10: Table representing associative containers and its description](img/C11557_05_10.jpg)

###### Figure 5.10: Table presenting associative containers and their descriptions

Typically, associative containers are implemented as variations of binary trees, providing fast element lookup by exploiting the logarithmic complexity of the underlying structure.

### Set and Multiset

A **Set** is a container that contains a unique group of sorted elements. A **Multiset** is similar to *Set*, but it allows duplicate elements:

![Figure 5.11: Set and Multiset store a sorted group of elements](img/C11557_05_11.jpg)

###### Figure 5.11: Set and Multiset store a sorted group of elements

Set and multiset have `size()` and `empty()` function members to check how many elements are contained and whether any elements are contained.

Insertion and removal is done through the `insert()` and `erase()` functions. Because the order of the elements is determined by the *comparator*, they do not take a position argument like they do for sequential containers. Both insertion and removal are fast.

Since sets are optimized for element lookup, they provide special search functions. The `find()` function returns the position of the first element equal to the provided value, or the position past the end of the set when the element is not found. When we look for an element with `find`, we should always compare it with the result of calling `end()` on the container to check whether the element was found.

Let's examine the following code:

```cpp
#include <iostream>
#include <set>
int main() {
    std::set<int> numbers;
    numbers.insert(10);
    if (numbers.find(10) != numbers.end()) {
        std::cout << "10 is in numbers" << std::endl;
    }
}
```

Finally, `count()` returns the number of elements equal to the value provided.

The `set` and `multiset` classes are defined in the `<set>` header file.

Example of a set with a custom comparator:

```cpp
#include <iostream>
#include <set>
#include <functional>
int main() {
    std::set<int> ascending = {5,3,4,2,1};
    std::cout << "Ascending numbers:";
    for(int number : ascending) {
        std::cout << " " << number;
    }
    std::cout << std::endl;

    std::set<int, std::greater<int>> descending = {5,3,4,2,1};
    std::cout << "Descending numbers:";
    for(int number : descending) {
        std::cout << " " << number;
    }
    std::cout << std::endl;
}
```

Output:

Ascending numbers: 1 2 3 4 5

Descending numbers: 5 4 3 2 1

### Map and Multimap

**Map** and **multimap** are containers that manage **key/value** pairs as elements. The elements are sorted automatically according to the provided comparator and applied to the *key*: the *value* does not influence the order of the elements:

![Figure 5.12: Map and multimap store a sorted group of keys, which is associated to a value](img/C11557_05_12.jpg)

###### Figure 5.12: Map and multimap store a sorted group of keys, which is associated to a value

Map allows you to associate a single value to a key, while multimap allows you to associate multiple values to the same key.

The `map` and `multimap` classes are defined in the `<map>` header file.

To insert values into a map, we can call `insert()`, providing a `true` if the element was inserted, or `false` if an element with the same key already exists.

Once values are inserted into the map, there are several ways to look up a key/value pair in a map.

Similar to set, map provides a `find()` function, which looks for a key in the map and returns the position of the key/value pair if it exists, or the same result of calling `end()`.

From the position, we can access the key with `position->first` and the value with `position->second`:

```cpp
#include <iostream>
#include <string>
#include <map>
int main()
{
    std::map<int, std::string> map;
    map.insert(std::make_pair(1, "some text"));
    auto position = map.find(1);
    if (position != map.end() ) {
        std::cout << "Found! The key is " << position->first << ", the value is " << position->second << std::endl;
    }
}
```

An alternative to accessing a value from a key is to use `at()`, which takes a key and returns the associated value.

If there is no associated value, `at()` will throw an exception.

A last alternative to get the value associated with a key is to use `operator[]`.

The `operator[]` returns the value associated with a key, and if the key is not present, it inserts a new key/value pair with the provided key, and a default value for the value. Because `operator[]` could modify the map by inserting into it, it cannot be used on a *const* map:

```cpp
#include <iostream>
#include <map>
int main()
{
    std::map<int, int> map;
    std::cout << "We ask for a key which does not exists: it is default inserted: " << map[10] << std::endl;
    map.at(10) += 100;
    std::cout << "Now the value is present: " << map.find(10)->second << std::endl;
}
```

### Activity 20: Retrieving a User's Balance from their Given Username

We'd like to be able to quickly retrieve the balance of a user given their username.

To quickly retrieve the balance from the username, we store the balance inside a map, using the name of the user as a key.

The name of the user is of type `std::string`, while the balance is an `int`. Add the balance for the users `Alice`, `Bob`, and `Charlie` with a balance of 50 each. Then, check whether the user `Donald` has a balance.

Finally, print the account balance of `Alice`:

1.  Include the header file for the `map` class and the header for `string`:

    ```cpp
    #include <string>
    #include <map>
    #include <string>
    ```

2.  Create a map with the key being `std::string` and the value being `int`.
3.  Insert the balances of the users inside the map by using `insert` and `std::make_pair`. The first argument is the `key`, while the second one is the `value`:

    ```cpp
    balances.insert(std::make_pair("Alice",50));
    ```

4.  Use the `find` function, providing the name of the user to find the position of the account in the map. Compare it with `end()` to check whether a position was found.
5.  Now, look for the account of Alice. We know Alice has an account, so there is no need to check whether we found a valid position. We can print the value of the account using `->second`:

    ```cpp
    auto alicePosition = balances.find("Alice");
    std::cout << "Alice balance is: " << alicePosition->second << std::endl;
    ```

    #### Note

    The solution for this activity can be found on page 305.

## Unordered Containers

**Unordered associative containers** differ from associative containers in that the elements have no defined order. Visually, unordered containers are often imagined as bags of elements. Because the elements are not sorted, unordered containers do not accept a comparator object to provide an order to the elements. On the other hand, all the unordered containers depend on a hash function.

he user can provide a `Functor` (function object) as a parameter to specify how the keys should be hashed:

![Figure 5.13: Table representing unordered container and its description](img/C11557_05_13.jpg)

###### Figure 5.13: Table presenting unordered containers and their descriptions

Typically, unordered containers are implemented as **hash tables**. The position in the array is determined using the hash function, which given a value returns the position at which it should be stored. Ideally, most of the elements will be mapped into different positions, but the hash function can potentially return the same position for different elements. This is called a *collision*. This problem is solved by using linked lists to chain elements that map into the same position, so that multiple elements can be stored in the same position. Because there might be multiple elements at the same position, the position is often called **bucket**.

Implementing unordered containers using a hash table allows us to find an element with a specific value in constant time complexity, which translates to an even faster lookup when compared to associative containers:

![Figure 5.14: When an element is added to the set, its hash is computed to decide in which bucket the element should be added. The elements inside a bucket are stored in a list.](img/C11557_05_14.jpg)

###### Figure 5.14: When an element is added to the set, its hash is computed to decide in which bucket the element should be added. The elements inside a bucket are stored as nodes of a list.

When a key/value pair is added to the map, the hash of the key is computed to decide in which bucket the key/value pair should be added:

![Figure 5.15: Representation of storing the bucket elements in a list.](img/C11557_05_15.jpg)

###### Figure 5.15: Representation of computing the bucket of an element from the key, and storing the key/value pair as nodes in a list.

Unordered associative containers and ordered associative containers provide the same functionalities, and the explanations in the previous section apply to the unordered associative containers as well. Unordered associative containers can be used to get better performances when the order of the elements is not important.

## Container Adaptors

Additional container classes that are provided by the STL library are container adaptors. Container adaptors provide constrained access policies on top of the containers we have looked at in this chapter.

Container adaptors have a template parameter that the user can provide to specify the type of container to wrap:

![](img/Image18089.jpg)

###### Figure 5.16: Table presenting container adaptors and their descriptions

### Stack

The stack container implements the LIFO access policy, where the elements are virtually stacked one on the top of the other so that the last inserted element is always on top. Elements can only be read or removed from the top, so the last inserted element is the first that gets removed. A stack is implemented using a sequence container class internally, which is used to store all the elements and emulate the stack behavior.

The access pattern of the stack data structure happens mainly through three core member functions: `push()`, `top()`, and `pop()`. The `push()` function is used to insert an element into the stack, `top()` used to access the element on top of the stack, and `pop()` is used to remove the top element.

The `stack` class is defined in the `<stack>` header file.

### Queue

The `queue` class implements the FIFO access policy, where the elements are enqueued one after the other, so that elements inserted before are ahead of elements inserted after. Elements are inserted at the end of the queue and removed at the start.

The interface of the queue data structure is composed of the `push()`, `front()`, `back()`, and `pop()` member functions.

The `push()` function is used to insert an element into the `queue()`; `front()` and `back()` return the next and last elements of the queue, respectively; the `pop()` is used to remove the next element from the queue.

The `queue` class is defined in the `<queue>` header file.

### Priority Queue

Finally, the priority queue is a queue where the elements are accessed according to their priority, in descending order (highest priority first).

The interface is similar to the normal queue, where `push()` inserts a new element and `top()` and `pop()` access and remove the next element. The difference is in the way the next element is determined. Rather than being the first inserted element, it is the element that has the highest priority.

By default, the priority of the elements is computed by comparing the elements with the `operator<`, so that an element that is less than another comes after it. A user-defined sorting criterion can be provided to specify how to sort the elements by priority in regard to their priority in the queue.

The priority queue class is also defined in the `<queue>` header file.

### Activity 21: Processing User Registration in Order

When a user registers to our website, we need to process the registration form at the end of the day.

We want to process the registration in reverse order of registration:

1.  Assume that the class for the registration form is already provided:

    ```cpp
    struct RegistrationForm {
        std::string userName;
    };
    ```

2.  Create a `stack` to store the users.
3.  We want to store the user registration form when the user registers, as well as process the registration at the end of the day. The function for processing the form is provided:

    ```cpp
    void processRegistration(RegistrationForm form) {
        std::cout << "Processing form for user: " << form.userName << std::endl;
    }
    ```

4.  Additionally, there are already two functions that are called when a user registers.
5.  Fill the code inside the following two functions to store the user form and process it:

    ```cpp
    void storeRegistrationForm(std::stack<RegistrationForm>& stack, RegistrationForm form) {
    }
    void endOfDayRegistrationProcessing(std::stack<RegistrationForm>& stack) {
    }
    ```

We'll see that the registration forms are processed in reverse order as the users are registered.

#### Note

The solution for this activity can be found at page 306.

## Unconventional Containers

Up until now, we've seen containers that are used to store groups of elements of the same type.

The C++ standard defines some other types that can contain types but offer a different set of functionalities from the containers we saw previously.

These types are as follows:

1.  String
2.  Pair and tuple
3.  Optional
4.  Variant

### Strings

A string is a data structure that's used to manipulate mutable sequences of contiguous characters. The C++ string classes are STL containers: they behave similarly to *vectors*, but provide additional functionalities that ease the programmer to perform common operations of sequences of characters easily.

There exist several string implementations in the standard library that are useful for different lengths of character sets, such as `string`, `wstring`, `u16string`, and `u32string`. All of them are a specialization of the `basic_string` base class and they all have the same interface.

The most commonly used type is `std::string`.

All types and functions for strings are deﬁned in the `<string>` header file.

A string can be converted into a *null-terminating string*, which is an array of characters that terminate with the special null character (represented with '`\0`') via the use of the `data()` or `c_str()` functions. Null-terminating strings, also called *C-strings*, are the way to represent sequences of character in the C language and they are often used when the program needs to interoperate with a C library; they are represented with the `const char *` type and are the type of the *literal strings* in our programs.

### Exercise 12: Demonstrating Working Mechanism of the `c_str()` Function

Let's examine the following code to understand how the `c_str()` function works:

1.  First include the required header files as illustrated:

    ```cpp
    #include <iostream>
    #include <string>
    ```

2.  Now, in the `main` function add a constant char variable named `charString` with capacity as `8` characters:

    ```cpp
    int main()
    {
      // Construct a C-string being explicit about the null terminator
      const char charString[8] = {'C', '+', '+', ' ', '1', '0', '1', '\0'};
      // Construct a C-string from a literal string. The compiler automatically adds the \0 at the end
      const char * literalString = "C++ Fundamentals";
      // Strings can be constructed from literal strings.
      std::string strString = literalString;
    ```

3.  Use the `c_str()` function and assign the value of `strString` to `charString2`:

    ```cpp
      const char *charString2 = strString.c_str();
    ```

4.  Print the `charString` and `charString2` using the print function:

    ```cpp
      std::cout << charString << std::endl;
      std::cout << charString2 << std::endl;
    }
    ```

    The output is as follows:

    ```cpp
    Output:
    C++ 101
    C++ Fundamentals
    ```

As for vectors, strings have `size()`, `empty()`, and `capacity()` member functions, but there is an additional function called `length()`, which is just an alias for `size()`.

Strings can be accessed in a character-by-character fashion using `operator[]` or the `at()`, `front()`, and `back()` member functions:

```cpp
std::string chapter = "We are learning about strings";
std::cout << "Length: " << chapter.length() << ", the second character is " << chapter[1] << std::endl;
```

The usual comparison operators are provided for strings, thus simplifying the way two string objects can be compared.

Since strings are like vectors, we can add and remove characters from them.

Strings can be made empty by assigning an empty string, by calling the `clear()`, or `erase()` functions.

Let's look at the following code to understand the usage of the `clear()` and `erase()` functions:

```cpp
#include <iostream>
#include <string>
int main()
{
  std::string str = "C++ Fundamentals.";
  std::cout << str << std::endl;
  str.erase(5,10);
  std::cout << "Erased: " << str << std::endl;
  str.clear();
  std::cout << "Cleared: " << str << std::endl;
}
Output:
C++ Fundamentals.
Erased: C++ Fs.
Cleared: 
```

C++ also provides many convenience functions to convert a string into numeric values or vice versa. For example, the `stoi()` and `stod()` functions (which stand for *string-to-int* and *string-to-double*) are used to convert `string` to `int` and `double`, respectively. Instead, to convert a value into a string, it is possible to use the overloaded function `to_string()`.

Let's demystify these functions using the following code:

```cpp
#include <iostream>
#include <string>
using namespace std;
int main()
{
  std::string str = "55";
  std::int strInt = std::stoi(str);
  double strDou = std::stod(str);
  std::string valToString = std::to_string(strInt);

  std::cout << str << std::endl;
  std::cout << strInt << std::endl;
  std::cout << strDou << std::endl;
  std::cout << valToString << std::endl;
}
Output:
55
55
55
55
```

### Pairs and Tuples

The **pair** and **tuple** classes are similar to some extent, in the way they can store a collection of heterogeneous elements.

The **pair** class can store the values of two types, while the **tuple** class extended this concept to any length.

Pair is defined in the `<utility>` header, while tuple is in the `<tuple>` header.

The pair constructor takes two types as template parameters, used to specify the types for the first and second values. Those elements are accessed directly using the `first` and `second` data. Equivalently, these members can be accessed with the `get<0>()` and `get<1>()` functions.

The `make_pair()` convenience function is used to create a value pair without explicitly specifying the types:

```cpp
std::pair<std::string, int> nameAndAge = std::make_pair("John", 32);
std::cout << "Name: " << nameAndAge.first << ", age: " << nameAndAge.second << std::endl;
```

The second line is equivalent to the following one:

```cpp
std::cout << "Name: " << std::get<0>(nameAndAge) << ", age: " << std::get<1>(nameAndAge) << std::endl;
```

Pairs are used by unordered map, unordered multimap, map, and multimap containers to manage their key/value elements.

Tuples are similar to pairs. The constructor allows you to provide a variable number of template arguments. Elements are accessed with the `get<N>()` function only, which returns the nth element inside the tuple, and there is a convenience function to create them similar to that for pair, named `make_tuple()`.

Additionally, tuples have another convenience function that's used to extract values from them. The `tie()` function allows for the creation of a tuple of references, which is useful in assigning selected elements from a tuple to specific variables.

Let's understand how to use the `make_tuple()` and `get()` functions to retrieve data from a tuple:

```cpp
#include <iostream>
#include <tuple>
#include <string>
int main()
{
  std::tuple<std::string, int, float> james = std::make_tuple("James", 7, 1.90f);
  std::cout << "Name: " << std::get<0>(james) << ". Agent number: " << std::get<1>(james) << ". Height: " << std::get<2>(james) << std::endl;
}
Output:
Name: James. Agent number: 7\. Height: 1.9
```

## std::optional

`optional<T>` is a that's used to contain a value that might be present or not.

The class takes a template parameter, `T`, which represents the type that the `std::optional` template class might contain. Value type means that the instance of the class contains the value. Copying `optional` will create a new copy of the contained data.

At any point in the execution of the program, `optional<T>` either contains nothing, when it's empty, or contains a value of type `T`.

Optional is defined in the `<optional>` header.

Let's imagine our application is using a class named `User` for managing registered users. We would like to have a function that gets us the information of a user from their email: `User getUserByEmail(Email email);`.

But what happens when a user is not registered? That is, when we can determine that our system does not have the associated `User` instance?

Some would suggest throwing an exception. In C++, exceptions are used for *exceptional* situations, ones that should almost never happen. A user not being registered on our website is a perfectly normal situation.

In these situations, we can use the `optional` template class to represent the fact that we might not have the data:

```cpp
std::optional<User> tryGetUserByEmail(Email email);
```

The `optional` template provides two easy methods to work with:

*   `has_value()`: This returns `true` if `optional` is currently holding a value, and `false` if the variant is empty.
*   `value()`: This function returns the value currently held by `optional`, or throws an exception if it's not present.
*   Additionally, `optional` can be used as a condition in an `if` statement: it will evaluate to `true` if it contains a value, or `false` otherwise.

Let's look at the following example to understand how the `has_value()` and `value()` functions work:

```cpp
#include <iostream>
#include <optional>
int main()
{
  // We might not know the hour. But if we know it, it's an integer
  std::optional<int> currentHour;
  if (not currentHour.has_value()) {
    std::cout << "We don't know the time" << std::endl;   
  }
  currentHour = 18;
  if (currentHour) {
    std::cout << "Current hour is: " << currentHour.value() << std::endl;
  }
}
Output:
We don't know the time
Current hour is: 18
```

The `optional` template comes with additional convenience features. We can assign the `std::nullopt` value to `optional` to make it explicit when we want it empty, and we can use the `make_optional` value to create an optional from a value. Additionally, we can use the dereference operator, `*`, to access the value of `optional` without throwing an exception if the value is not present. In such cases, we will access invalid data, so we need to be sure that `optional` contains a value when we use `*`:

```cpp
std::optional<std::string> maybeUser = std::nullopt;
if (not maybeUser) {
  std::cout << "The user is not present" << std::endl;
}
maybeUser = std::make_optional<std::string>("email@example.com");
if (maybeUser) {
  std::cout << "The user is: " << *maybeUser  << std::endl;
}
```

Another handy method is `value_or(defaultValue)`. This function takes a default value and returns the value contained by `optional` if it currently holds a value, otherwise it returns the default value. Let's explore the following example:

```cpp
#include <iostream>
#include <optional>
int main()
{
  std::optional<int> x;
  std::cout << x.value_or(10) << std::endl;
  //Will return value of x as 10
  x = 15;
  std::cout << x.value_or(10)<< std::endl;
  //Will return value of x as 15
}
Output:
10
15
```

In addition to return values, `optional` is useful when accepting it as an argument to represent arguments that can be present or not.

Let's recall our `User` class that's composed of an email address, a phone number, and a physical address. Sometimes, users don't have a phone number and don't want to provide a physical address, so the only required field we have in `User` is the email address:

```cpp
User::User(Email email, std::optional<PhoneNumber> phoneNumber = std::nullopt, std::optional<Address> address = std::nullopt){
...
}
```

This constructor allows us to pass in all the information we have on the user. If, instead of using `optional`, we used multiple overloads, we would have had four overloads:

1.  Only email
2.  Email and phone number
3.  Email and address
4.  Email with phone number and address

You can see that the number of overloads grows quickly when there are more arguments that we might not want to pass.

## std::variant

`variant` is a value type that's used to represent a *choice of types*. The class takes a list of types, and the variant will be able to contain one value of any of those types.

It is often referred to as **tagged union**, because similar to a union, it can store multiple types, with only one present at a time. It also keeps track of which type is currently stored.

During the execution of a program, `variant` will contain exactly one of the possible types at a time.

Like `optional`, `variant` is a value type: when we create a copy of `variant`, the element that is currently stored is copied into the new `variant`.

To interact with `std::variant`, the C++ standard library gives us two main functions:

*   `holds_alternative<Type>(variant)`: It returns `true` if the variant is currently holding the provided type, if not then `false`.
*   `get(variant)`: There are two versions: `get<Type>(variant)` and `get<Index>(variant)`.

`get<Type>(variant)` gets the value of the type that's currently stored inside the variant. Before calling this function, the caller needs to be sure that `holds_alternative<Type>(variant)` returns `true`.

`get<Index>(variant)` gets the value of the index type that's currently stored inside `variant`. Like before, the caller needs to be sure that `variant` is holding the correct type.

For example, with `std::variant<string, float> variant`, calling `get<0>(variant)` will give us the `string` value, but we need to be sure that `variant` is currently storing a string at the moment. Usually, it is preferable to access the elements with `get<Type>()` so that we are explicit on the type that we expect and that if the order of the types in the variant changes, we will still get the same result:

### Exercise 13: Using Variant in the Program

Let's perform the following steps to understand how to use variant in the program:

1.  Include the required header files:

    ```cpp
    #include <iostream>
    #include <variant>
    ```

2.  In the main function, add the variant with the value type as string and integer:

    ```cpp
    int main()
    {
      std::variant<std::string, int> variant = 42;
    ```

3.  Now using the two print statements call the variant in different ways:

    ```cpp
      std::cout << get<1>(variant) << std::endl;
      std::cout << get<int>(variant) << std::endl;
    ```

The output is as follows:

```cpp
Output:
42
42
```

An alternative way to get the content of `variant` is to use `std::visit(visitor, variant)`, which takes `variant` and a callable object. The callable objects need to support an overload of `operator()`, taking a type for each of the possible types stored inside `variant`. Then, `visit` will make sure to call the function that accepts the current type that's stored inside `variant`:

### Exercise 14: Visitor Variant

Let's perform the following steps to understand how to use std::visit(visitor, variant) in the program:

1.  Add the following header files at the start of the program:

    ```cpp
    #include <iostream>
    #include <string>
    #include <variant>
    ```

2.  Now, add the struct Visitor as illustrated:

    ```cpp
    struct Visitor {
        void operator()(const std::string& value){
            std::cout << "a string: " << value << std::endl;
        }
        void operator()(const int& value){
            std::cout << "an int: " << value << std::endl;
        }
    };
    ```

3.  Now, in the main function, call the struct Visitor and pass values as illustrated:

    ```cpp
    int main()
    {
        std::variant<std::string, int> variant = 42;
        Visitor visitor;
        std::cout << "The variant contains ";
        std::visit(visitor, variant);
        variant = std::string("Hello world");
        std::cout << "The variant contains ";
        std::visit(visitor, variant);
    }
    ```

The output is as follows:

```cpp
The variant contains an int: 42
The variant contains a string: Hello world
```

`variant` is incredibly valuable when we want to represent a set of values of different types. Typical examples are as follows:

*   A function returning different types depending on the current state of the program
*   A class that represents several states

Let's imagine our `std::optional<User> tryGetUserByEmail()` function, which we described earlier.

Thanks to `optional`, we could now write the function in a clear way, showing that sometimes we would not retrieve the user. It is likely that if the user is not registered, we might ask them whether they want to register.

Let's imagine we have `struct UserRegistrationForm`, which contains the information that's needed to let the user register.

Our function can now return `std::variant<User, UserRegistrationForm> tryGetUserByEmail()`. When the user is registered, we return `User`, but if the user is not registered, we can return the registration form.

Additionally, what should we do when there is an error? With `variant`, we could have `struct GetUserError` storing all the information we have so that our application will be able to recover from the error and add it to the return type: `std::variant<User`, `UserRegistrationForm`, `GetUserError>`, or `tryGetUserByEmail()`.

Now we can have the complete picture of what is going to happen when we call `getUserByEmail()` by just looking at the function signature, and the compiler will help us make sure that we handle all the cases.

Alternatively, `variant` can also be used to represent the various states in which a class can be. Each state contains the data that's required for that state, and the class only manages the transitions from one state to another.

### Activity 22: Airport System Management

Let's write a program to create airport system management:

1.  We want to represent the state of an airplane in an airport system. The airplane can be in three states: `at_gate`, `taxi`, or `flying`. The three states store different information.
2.  With `at_gate`, the airplane stores the gate number at which it is. With `taxi`, we store which lane the airplane is assigned and how many passengers are on board. With `flying`, we store the speed:

    ```cpp
    struct AtGate {
        int gate;
    };
    struct Taxi {
        int lane;
        int numPassengers;
    };
    struct Flying {
        float speed;
    };
    ```

3.  The airplane should have three methods:
    *   `startTaxi()`: This method takes the lane the airplane should go on and the number of passengers on board. The airplane can start taxi only if it is at the gate.
    *   `takeOff()`: This method takes the speed at which the airplane should fly. The airplane can start flying only if it is in the taxi state.
    *   `currentStatus()`: This method prints the current status of the airplane.

        #### Note

        The solution for this activity can be found on page 306.

## Iterators

In this chapter, we've mentioned multiple times that elements have a position in a container: for example, we said that we can insert an element in a specific position in a list.

Iterators are the way in which the position of an element in a container is represented.

They provide a consistent way to operate on elements of the container, abstracting the details of the container to which the elements belong.

An iterator always belongs to a range. The iterator representing the start of the range, can be accessed by the `begin()` function, while the iterator representing the end of the range, non-inclusive, can be obtained with the `end()` function. The range where the first element is included, but where the last one is excluded, is referred to as half-open.

The interface that the iterator must offer is composed of four functions:

1.  The `*` operator provides access to the element at the position currently referenced by the iterator.
2.  The `++` operator is used to move forward to the next element.
3.  Then, the `==` operator is used to compare two iterators to check whether they are pointing to the same position.

    Note that two iterators can only be compared if they are part of the same range: they must represent the position of elements of the same container.

4.  Finally, the `=` operator is used to assign an iterator.

Every container class in C++ must specify the type of iterator that it provides to access its elements as a member type alias named `iterator`. For example, for a vector of integer, the type would be `std::vector<int>::iterator`.

Let's see how we could use iterators to iterate over all the elements of a container (a vector, in this case):

```cpp
#include <iostream>
#include <vector>
int main()
{
    std::vector<int> numbers = {1, 2, 3};
    for(std::vector<int>::iterator it = numbers.begin(); it != numbers.end(); ++it) {
        std::cout << "The number is: " << *it << std::endl;    
    }
}
```

This looks complex for such an operation, and we saw in *Chapter 1, Getting Started* how we can use *range-based for:*

```cpp
for(int number: numbers) {
    std::cout << "The number is: " << number << std::endl;    
}
```

R*ange-based for* works thanks to iterators: the compiler rewrites our *range-based for* to look like the one we wrote with iterators. This allows the *range-based for* to work with any type that provides `begin()` and `end()` functions and returns iterators.

The way operators provided by the iterators are implemented depends on the container on which the iterator operates.

Iterator can be grouped into four categories. Each category builds on the previous category, thus offering additional functionality:

![](img/C11557_05_17.jpg)

###### Figure 5.17: Table presenting iterators and their descriptions

The following diagram gives more detail about C++ iterators:

![Figure 5.18: Representation of iterators in C++](img/C11557_05_18.jpg)

###### Figure 5.18: Representation of iterators hierarchy in C++

Let's understand each of these iterators in more detail:

*   `==` and `!=` operators to check whether the iterator is equal to the `end()` value.

    Typically, input iterators are used to access elements from a stream of elements, where the whole sequence is not stored in memory, but we are obtaining one element at a time.

*   **Forward iterators** are very similar to input iterators but provide additional guarantees.

    The same iterator can be dereferenced several times to access the element it points to.

    Additionally, when we increment or dereference a forward iterator, the other copies are not invalidated: if we make a copy of a forward iterator, we can advance the first one, and the second can still be used to access the previous element.

    Two iterators that refer to the same element are guaranteed to be equal.

*   `operator--` (position decrement) member function.
*   `operator[]` member function to access elements at generic indexes and the binary `operator+` and `operator-` to step forward and backward of any quantity.

### Exercise 15: Exploring Iterator

Perform the following steps to explore the four categories discussed in the previous section and writing to the element it points to, it is also an Output Iterator:

1.  Add the following header files at the start of the program:

    ```cpp
    #include <iostream>
    #include <vector>
    ```

2.  In the main function declare the vector named number:

    ```cpp
    int main()
    {
        std::vector<int> numbers = {1, 2, 3, 4, 5};
        auto it = numbers.begin();
    ```

3.  Perform the various arithmetic operations as illustrated:

    ```cpp
        std::cout << *it << std::endl; // dereference: points to 1
        it++; // increment: now it points to 2
        std::cout << *it << std::endl;
        // random access: access the 2th element after the current one
        std::cout << it[2] << std::endl;
        --it; // decrement: now it points to 1 again
        std::cout << *it << std::endl;
        it += 4; // advance the iterator by 4 positions: now it points to 5
        std::cout << *it << std::endl;
        it++; // advance past the last element;
        std::cout << "'it' is after the past element: " << (it == numbers.end()) << std::endl;
    }
    ```

The output is as follows:

```cpp
1
2
4
1
5
'it' is after the past element: 1
```

Many of the iterators we will talk about are defined in the `<iterator>` header.

### Reverse Iterators

Sometimes, we need to iterate though a collection of elements in reverse order.

C++ provides an iterator that allows us to do this: the *reverse iterator*.

A *reverse iterator* wraps a *bidirectional iterator* and swaps the operation increment with the operation of decrement, and vice versa.

Because of this, when we are iterating a reverse iterator in the forward direction, we are visiting the elements in a range in backward order.

We can reverse the range of a container by calling the following methods on a container:

![Figure 5.19: Table representing iterator functions and its description](img/C11557_05_19.jpg)

###### Figure 5.19: Table presenting iterator functions and their descriptions

Code that works on normal iterators, it will also work with reverse iterators.

For example, we can see how similar the code is to iterate in reverse order.

### Exercise 16: Exploring Functions of Reverse Iterator

Let's perform the following steps to understand how functions in reverse iterator works:

1.  Add the following header files at the start of the program:

    ```cpp
    #include <iostream>
    #include <vector>
    ```

2.  In the main function, add the vector named numbers as illustrated:

    ```cpp
    int main()
    {
        std::vector<int> numbers = {1, 2, 3, 4, 5};
    ```

3.  Now iterate through the number vector as illustrated:

    ```cpp
        for(auto rit = numbers.rbegin(); rit != numbers.rend(); ++rit) {
            std::cout << "The number is: " << *rit << std::endl;    
        }
    }
    ```

The output is as follows:

```cpp
The number is: 5
The number is: 4
The number is: 3
The number is: 2
The number is: 1
```

### Insert Iterators

**Insert iterators**, also called **inserters**, are used to insert new values into a container rather than overwrite them.

There exist three types of inserters, which differ on the position in the container at which they insert the elements.

The following table summarizes the different categories:

![Figure 5.20: Table representing iterator functions and its description](img/C11557_05_20.jpg)

###### Figure 5.20: Table presenting iterator functions and their descriptions

Some algorithms, which we are going to see later in this chapter, require an iterator for storing data. Insert iterators are usually used with such algorithms.

### Stream Iterators

`Stream iterators` allow us to use streams as a source to read elements from or as a destination to write elements to:

![Figure 5.21: Table representing iterator functions and its description ](img/C11557_05_21.jpg)

###### Figure 5.21: Table presenting iterator functions and their descriptions

Because we don't have a container in this case, we cannot call the `end()` method to get the `end` iterator. A default constructed stream iterator counts as the end of any stream range.

Let's look at a program that reads space-separated integers from the standard input.

### Exercise 17: Stream Iterator

Let's perform the following steps to understand how functions in reverse stream works:

1.  Add the required header files as illustrated:

    ```cpp
    #include <iostream>
    #include <iterator>
    ```

2.  Now, in the main function, add the istream iterator as illustrated:

    ```cpp
    int main()
    {
        std::istream_iterator<int> it = std::istream_iterator<int>(std::cin);
        std::istream_iterator<int> end;
        for(; it != end; ++it) {
            std::cout << "The number is: " << *it << std::endl;
        }
    }
    ```

The output is as follows (input: 10):

```cpp
The number is: 10
```

### Iterator Invalidation

As we said, iterators represent the position of elements in a container.

This means that they are tightly tied with the container, and changes to the container might move the elements: this means that iterators pointing to such an element can no longer be used – they are **invalidated**.

It is extremely important to always check the invalidation contract when using iterators with containers, as it is not specified what happens when using an invalidated iterator. More commonly, invalid data is accessed or the program crashes, leading to bugs that are hard to find.

If we keep in mind how the containers are implemented, as we saw earlier in this chapter, we can more easily remember when an iterator is invalidated.

For example, we said that when we insert an element in a vector, we might have to get more memory to store the element, in which case all the previous elements are moved to the newly obtained memory. This means that all the iterators pointing to the elements are now pointing to the old location of the elements: they are invalidated.

On the other hand, we saw that when we insert an element into the list, we only have to update the predecessor and successor nodes, but the elements are not moved. This means that the iterators to the elements remain valid:

```cpp
#include <iostream>
#include <vector>
#include <list>
int main()
{
    std::vector<int> vector = {1};
    auto first_in_vec = vector.begin();
    std::cout << "Before vector insert: " << *first_in_vec << std::endl;
    vector.push_back(2);
    // first_number is invalidated! We can no longer use it!
    std::list<int> list = {1};
    auto first_in_list = list.begin();
    list.push_back(2);
    // first_in_list is not invalidated, we can use it.
    std::cout << "After list insert: " << *first_in_list << std::endl;
}
Output:
Before vector insert: 1
After list insert: 1
```

When there is a need to store iterators to elements, iterator invalidation is an important consideration to make when deciding which container to use.

### Exercise 18: Printing All of the Customers' Balances

We want to print the balances for all of the customers of our application. The balances are already stored inside a vector as integers.

We want to use iterators to traverse the vector of balances. Follow these steps to do so:

1.  Initially, we include the header file for the `vector` class, and we declare a vector of 10 elements of type `int`:

    ```cpp
    #include <vector>
    std::vector<int> balances = {10, 34, 64, 97, 56, 43, 50, 89, 32, 5};
    ```

2.  The `for` loop has been modified to iterate using the vector's iterator, starting from the position returned by `begin()` until it reaches the one returned by `end()`:

    ```cpp
    for (auto pos = numbers.begin(); pos != numbers.end(); ++pos)
    {
        // to be filled
    }
    ```

3.  The element of the array is accessed using the dereference operator (`*`) on the iterator:

    ```cpp
    for (auto pos = numbers.begin(); pos != numbers.end(); ++pos)
    {
        std::cout << "Balance: " << *pos << std::endl;
    }
    ```

## Algorithms Provided by the C++ Standard Template Library

Algorithms are a way to operate on containers in an abstract way.

The C++ standard library provides a wide range of algorithms for all the common operations that can be performed on ranges of elements.

Because algorithms accept iterators, they can operate on any container, even user-defined containers, as long as they provide iterators.

This allows us to have a large number of algorithms that work with a large number of containers, without the need for the algorithm to know how the container is implemented.

The following are some of the most important and common algorithms that are provided by the STL.

#### Note

Algorithms operate on ranges, so they normally take a pair of iterators: *first* and *last*.

As we said earlier in this chapter, the *last* iterator denotes the element past the end of the range – it is not part of the range.

This means that when we want to operate on a full container, we can pass `begin()` and `end()` as arguments to the algorithm, but if we want to operate on a shorter sequence, we must be sure that our *last* iterator is past the last item we want to include in the range.

### Lambda

Most of the algorithms accept a unary or binary predicate: a `Functor` (function object), which accepts either one or two parameters. These predicates allow the user to specify some of the actions that the algorithm requires. What the actions are vary from algorithm to algorithm.

As we saw at the end of *Chapter 3, Classes*, to write a function object, we have to create a class and overload the `operator()`.

This can be very verbose, especially when the functor should perform a simple operation.

To overcome this with C++, the user has to write a **lambda expression**, also called just a *lambda*.

A *lambda expression* creates a special function object, with a type known only by the compiler, that behaves like a function but can access the variables in the scope in which it is created.

It is defined with a syntax very similar to the one of functions:

```cpp
[captured variables] (arguments) { body }
```

This creates a new object that, when called with the arguments specified in the lambda expression, executes the body of the function.

*Arguments* is the list of arguments the function accepts, and *body* is the sequence of statements to execute when the function is invoked. They have the same meaning that they have for functions, and the same rules we saw in *Chapter 2, Functions,* apply.

For example, let's create a lambda that takes two integers and returns their sum:

```cpp
#include <iostream>
int main()
{
    auto sum_numbers = [] (int a, int b) { return a + b; };
    std::cout << sum_numbers(10, 20) << std::endl;
}
Output:
30
```

By default, the body of the lambda can only reference the variables that are defined in the argument list and inside the body, like for functions.

Additionally, *lambdas* can **capture** a variable in the local scope, and use it in their body.

*Captured variables* entail a list of variable names that can be referenced in the body of the lambda.

When a variable is captured, it is stored inside the created function object, and it can be referenced in the body.

By default, the variables are *captured by value*, so they are copied inside the function object:

```cpp
#include <iostream>
int main()
{
    int addend = 1;
    auto sum_numbers = [addend](int b) { return addend + b; };
    addend = 2;
    std::cout << sum_numbers(3) << std::endl;
}
Output:
4
```

When we created the lambda, we captured `addend` by value: it was copied into the `sum_numbers` object. Even if we modified the value of `addend`, we did not change the copy stored inside `sum_numbers`, so when `sum_numbers` is executed, it sums 1 to `b`.

In some situations, we want to be able to modify the value of a variable in the scope in which the *lambda* is created, or we want to access the actual value, not the value that the variable had when the lambda was created.

In that case, we can capture by reference by prepending `&` to the variable name.

#### Note

When we capture by reference, we need to make sure that the variable that's been captured by reference is still valid when the lambda is invoked, otherwise the body of the function accesses an invalid object, resulting in bugs.Prefer to capture by value when it is possible.

Let's look at an example:

```cpp
#include <iostream>
int main()
{
    int multiplier = 1;
    auto multiply_numbers = [&multiplier](int b) { return multiplier * b; };
    multiplier = 2;
    std::cout << multiply_numbers(3) << std::endl;
}
Output:
6
```

Here, we capture the `multiplier` variable by reference: only a reference to it was stored into `multiply_numbers`.

When we invoke `multiply_numbers`, the body accesses the current value of `multiplier`, and since `multiplier` was changed to 2, that is the value that's used by the *lambda*.

A lambda can capture multiple variables, and each one can be either captured by value or by reference, independently one from the other.

### Read-Only Algorithms

Read-only algorithms are algorithms that inspect the elements stored inside a container but do not modify the order of the elements of the container.

The following are the most common operations that inspect the elements of a range:

![Figure 5.22: Table representing the operations that inspect elements of a range](img/C11557_05_22.jpg)

###### Figure 5.22: Table presenting the operations that inspect elements of a range

Let's see how we can use these functions:

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
int main()
{
    std::vector<int> vector = {1, 2, 3, 4};
    bool allLessThen10 = std::all_of(vector.begin(), vector.end(), [](int value) { return value < 10; });
    std::cout << "All are less than 10: " << allLessThen10 << std::endl;
    bool someAreEven = std::any_of(vector.begin(), vector.end(), [](int value) { return value % 2 == 0; });
    std::cout << "Some are even: " << someAreEven << std::endl;
    bool noneIsNegative = std::none_of(vector.begin(), vector.end(), [](int value) { return value < 0; });
    std::cout << "None is negative: " << noneIsNegative << std::endl;

    std::cout << "Odd numbers: " << std::count_if(vector.begin(), vector.end(), [](int value) { return value % 2 == 1; }) << std::endl;

    auto position = std::find(vector.begin(), vector.end(), 6);
    std::cout << "6 was found: " << (position != vector.end()) << std::endl;
}
Output:
All are less than 10: 1
Some are even: 1
None is negative: 1
Odd numbers: 2
6 was found: 0
```

### Modifying Algorithms

Modifying algorithms are algorithms that modify the collections they iterate on:

![Figure 5.23: Table representing the modifying algorithms](img/C11557_05_23.jpg)

###### Figure 5.23: Table presenting the modifying algorithms

Let's see these algorithms in action:

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
int main()
{
    std::vector<std::string> vector = {"Hello", "C++", "Morning", "Learning"};
    std::vector<std::string> longWords;

    std::copy_if(vector.begin(), vector.end(), std::back_inserter(longWords), [](const std::string& s) { return s.length() > 3; });
    std::cout << "Number of longWords: " << longWords.size() << std::endl;

    std::vector<int> lengths;
    std::transform(longWords.begin(), longWords.end(), std::back_inserter(lengths), [](const std::string& s) { return s.length(); });

    std::cout << "Lengths: ";
    std::for_each(lengths.begin(), lengths.end(), [](int length) { std::cout << length << " "; });
    std::cout << std::endl;

    auto newLast = std::remove_if(lengths.begin(), lengths.end(), [](int length) { return length < 7; });
    std::cout << "No element removed yet: " << lengths.size() << std::endl;

    // erase all the elements between the two iterators
    lengths.erase(newLast, lengths.end());
    std::cout << "Elements are removed now. Content: ";
    std::for_each(lengths.begin(), lengths.end(), [](int length) { std::cout << length << " "; });
    std::cout << std::endl;
}
Output:
Number of longWords: 3
Lengths: 5 7 8 
No element removed yet: 3
Elements are removed now. Content: 7 8
```

### Mutating Algorithms

Mutating algorithms are algorithms that change the order of elements:

![Figure 5.24: Table representing the mutating algorithms](img/C11557_05_24.jpg)

###### Figure 5.24: Table presenting mutating algorithms

Let's see how we can use them:

```cpp
#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include <iterator>
int main()
{
    std::vector<int> vector = {1, 2, 3, 4, 5, 6};

    std::random_device randomDevice;
    std::mt19937 randomNumberGenerator(randomDevice());
    std::shuffle(vector.begin(), vector.end(), randomNumberGenerator);
    std::cout << "Values: ";
    std::for_each(vector.begin(), vector.end(), [](int value) { std::cout << value << " "; });
    std::cout << std::endl;
}
Output:
Values: 5 2 6 4 3 1
```

### Sorting Algorithms

This class of algorithms rearranges the order of elements within a container in a specific order:

![Figure 5.25: Table representing the sorting algorithms](img/C11557_05_25.jpg)

###### Figure 5.25: Table presenting sorting algorithms

Here is how to sort a vector:

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
int main()
{
    std::vector<int> vector = {5, 2, 6, 4, 3, 1};

    std::sort(vector.begin(), vector.end());
    std::cout << "Values: ";
    std::for_each(vector.begin(), vector.end(), [](int value) { std::cout << value << " "; });
    std::cout << std::endl;
}
Output:
Values: 1 2 3 4 5 6
```

### Binary Search Algorithms

The following table explains the use of `binary_search`:

![](img/C11557_05_26.jpg)

###### Figure 5.26: Table presenting the use of binary_search

Here's how you can utilize the binary search algorithm:

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
int main()
{
    std::vector<int> vector = {1, 2, 3, 4, 5, 6};

    bool found = std::binary_search(vector.begin(), vector.end(), 2);
    std::cout << "Found: " << found << std::endl;
}
Output:
Found: 1
```

### Numeric Algorithms

This class of algorithms combines numeric elements using a linear operation in different ways:

![Figure 5.27: Table representing the numeric algorithm](img/C11557_05_27.jpg)

###### Figure 5.27: Table presenting the numeric algorithm

Let's see how we can use `accumulate` in the following program:

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
int main()
{
    std::vector<int> costs = {1, 2, 3};

    int budget = 10;
    int margin = std::accumulate(costs.begin(), costs.end(), budget, [](int a, int b) { return a - b; });
    std::cout << "Margin: " << margin << std::endl;
}
Output:
Margin: 4
```

### Exercise 19: Customer Analytics

We have the information of many customers of our application and we want to compute analytics data on that.

Given a map that has a username as a key and a user account as a value, we would like to print the balances of the new users in descending order.

A user is considered new if they registered no more than 15 days ago. The struct representing the user's account is provided and is as follows:

```cpp
struct UserAccount {
    int balance;
    int daysSinceRegistered;
};
```

Write the `void computeAnalytics(std::map<std::string, UserAccount>& accounts)` function, which prints the desired balances.

1.  Make sure to include all the required headers for the solution:

    ```cpp
    #include <iostream>
    #include <vector>
    #include <iterator>
    #include <map>
    #include <algorithm>
    ```

2.  First, we need to extract `UserAccount` from the map. Remember that the element the map stores is `pair` containing a key and value. Since we need to transform the type into `UserAccount`, we can use `std::transform`, by passing a `lambda` that only returns the user account from the `pair`. To insert this into `vector`, we can use `std::back_inserter`. Make sure to use a `const` reference when accepting `pair` in the lambda that's passed to transform:

    ```cpp
    void computeAnalytics(std::map<std::string, UserAccount>& accounts) {
        // Balance of accounts newer than 15 days, in descending order
        std::vector<UserAccount> newAccounts;
        std::transform(accounts.begin(), accounts.end(), std::back_inserter(newAccounts),
                     [](const std::pair<std::string, UserAccount>& user) { return user.second; });
        }   
    ```

3.  After we have extracted the accounts in `vector`, we can use `remove_if` to remove all accounts that are older than 15 days:

    ```cpp
        auto newEnd = std::remove_if(newAccounts.begin(), newAccounts.end(), [](const UserAccount& account) { return account.daysSinceRegistered > 15; } );
        newAccounts.erase(newEnd, newAccounts.end());
    ```

4.  After removing the old accounts, we need to sort the balances in descending order. By default, `std::sort` uses an ascending order, so we need to provide a `lambda` to change the order:

    ```cpp
        std::sort(newAccounts.begin(), newAccounts.end(), [](const UserAccount& lhs, const UserAccount& rhs) { return lhs.balance > rhs.balance; } );
    Now that the data is sorted, we can print it:
        for(const UserAccount& account : newAccounts) {
            std::cout << account.balance << std::endl;
        }   
    }
    ```

5.  We can now invoke our function with the following test data:

    ```cpp
    int main()
    {
        std::map<std::string, UserAccount> users = {
            {"Alice", UserAccount{500, 15}},
            {"Bob", UserAccount{1000, 50}},
            {"Charlie", UserAccount{600, 17}},
            {"Donald", UserAccount{1500, 4}}
        };
        computeAnalytics(users);
    }
    ```

## Summary

In this chapter, we introduced sequential containers – containers whose elements can be accessed in sequence. We looked at the `array`, `vector`, `deque`, `list`, and `forward_list` sequential containers.

We saw what functionality they offer and how we can operate on them, and we saw how they are implemented and how storage works for vector and list.

We followed this up with associative containers, containers that allow the fast lookup of their elements, always kept in order. `Set`, `multiset`, `map`, and `multimap` are part of this category.

We looked at the operations they support and how map and multimap are used to associate a value to a key. We also saw their unordered version, which does not keep elements in order but provides higher performance. `Unordered_set` and `unordered_map` are in this category.

Finally, we looked at unconventional containers. `String` is used to manipulate sequences of characters, `pair` and `tuple` are used to hold various elements of different types, `optional` is used to add optionality to a type, and `variant` is used to store a value that could be of several types.

We then explored iterators and learned how they are used to abstract the concept of containers and provide a common set of functionalities.

We looked at the various types of iterators, and we learned what iterator invalidation is and why it is important to be aware of it.

We finally moved on to algorithms in the C++ standard, after explaining that `lambda` is a convenient way of defining a function that can also access variables in the scope in which it is created.

We divided the most common algorithms into various categories, and we looked at the most important algorithms in those categories, including `find`, `remove`, and `sort`.

In the next chapter, you will learn how to use the advanced features of C++ to create dynamic programs.