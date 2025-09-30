# *Chapter 4*

# Generic Programming and Templates

## Lesson Objectives

By the end of this chapter, you will be able to:

*   Understand how templates work and when to use them
*   Identify and implement templated functions
*   Implement template classes
*   Write code that works for multiple types

In this chapter, you will learn how to use templates effectively in your program.

## Introduction

When programming, it is common to face problems that are recurring for different types of objects, such as storing a list of objects, or searching elements in a list, or finding the maximum between two elements.

Let's say that in our program we want to be able to find the maximum between two elements, either integers or doubles. With the features we have learned so far, we could write the following code:

```cpp
int max(int a, int b) {
  if ( a > b) return a;
  else return b;
}
double max(double a, double b) {
  if ( a> b) return a;
  else return b;
}
```

In the previous code, the two functions are identical except for the *types* of the parameters and the *return type*. Ideally, we would like to write these kind of operations only once and reuse them in the entire program.

Moreover, our `max()` function can only be called with types for which an overload exists: `int` and `double` in this case. If we wanted it to work with any numerical type, we would need to write an **overload** for each of the numerical types: we would need to know in advance about all the types that will be used to call it, especially when the function is part of a library that is intended to be used by other developers, as it becomes impossible for us to know the types that will be used when calling the function.

We can see that there is nothing specific to integers being required to find the maximum elements; if the elements implement `operator<`, then it is possible to find the greater of the two numbers, and the algorithm does not change. In these situations, C++ offers an effective tool—**templates**.

## Templates

Templates are a way to define functions or classes that can work for many different types, while still writing them only once.

They do so by having special kinds of parameters—**type parameters**.

When writing the template code, we can use this type parameter as if it were a real type, such as `int` or `string`.

When the templated function is called or the template class is instantiated, the type parameter is substituted with the real type that's used by the calling code.

Now let's look at an example of a template in C++ code:

```cpp
template<typename T>
T max(T a, T b) {
  if(a>b) {
    return a;
  } else {
    return b;
  }
}
```

A template always starts with the `template` keyword, followed by the list of template parameters enclosed in *angle* brackets.

A template parameter list is a list of comma-separated parameters. In this case, we only have one—`typename T`.

The `typename` keyword tells the template that we are writing a templated function that uses a generic type, which we are going to name `T`.

#### Note

You can also use the `class` keyword in place of `typename`, since there is no difference between them.

Then, the definition of the function follows. In the function definition, we can use the name `T` when we want to refer to the generic type.

To call the template, we specify the name of the template, followed by the list of types we want to use as *type arguments*, enclosed in angle brackets:

```cpp
max<int>(10, 15);
```

This calls the templated function `max`, specifying `int` as the type parameter. We say that we instantiated the templated function `max` with type `int`, and then called that instance.

We do not always need to specify the type parameters of a template; the compiler can deduce them from the calling code. A later section will describe this feature.

Because of how powerful templates are, the big part of the C++ standard library is based on templates, as we will see in *Chapter 5*, *Standard Library Containers and Algorithms*.

Now we'll explore in depth what happens when we compile the code that contains templates.

### Compiling the Template Code

Similar to functions and classes, a template needs to be *declared* before being used.

When the compiler first encounters a template definition in the program, it parses it and performs only *part* of the checks it usually does on the rest of the code.

This happens because the compiler does not know which type is going to be used with the template when it parses it, since the types are parameters themselves. This prevents the compiler from performing checks that involve the parameter types, or anything that depends on them.

Because of this, you get notified of some errors in the template only when you instantiate it.

Once we define a template, we can instantiate it in our code.

When a template is instantiated, the compiler looks at the definition of the template and uses it to generate a new instance of the code, where all the references to the type parameters are replaced by the types that are provided when instantiating it.

For example: when we call `max<int>(1,2)`, the compiler looks at the template definition we specified earlier and generates code as if we wrote the following:

```cpp
int max(int a, int b) {
  if(a>b) {
    return a;
  } else {
    return b;
  }
}
```

#### Note

Since the compiler generates the code from the template definition, it means that the full definitions need to be visible to the calling code, not only the declaration, as was the case for functions and classes.

The template can still be forward declared, but the compiler must also see the definition. Because of this, when writing templates that should be accessed by several files, both the definition and the declaration of the templates must be in the **header** file.

This restriction does not apply if the template is used only in one file.

### Exercise 11: Finding the Bank Account of the User with the Highest Balance

Write a template function that accepts details of two bank accounts (of the same type) and returns the balance of the bank account with the highest balance.

For this exercise, perform the following steps:

1.  Let's create two structs named `EUBankAccount` and `UKBankAccount` to represent the **European Union** bank account and the **United Kingdom** bank account with the required basic information, as shown in the following code:

    ```cpp
    #include <string>
    struct EUBankAccount {
       std::string IBAN;
       int amount;
    };
    struct UKBankAccount {
       std::string sortNumber;
       std::string accountNumber;
       int amount;
    };
    ```

2.  The template function will have to compare the amount of the bank accounts. We want to work with different bank account types, so we need to use a template:

    ```cpp
    template<typename BankAccount>
    int getMaxAmount(const BankAccount& acc1, const BankAccount& acc2) {
        // All bank accounts have an 'amount' field, so we can access it safely
        if (acc1.amount > acc2.amount) {
            return acc1.amount;
        } else {
            return acc2.amount;
        }
    }
    ```

3.  Now, in the `main` function, call both the structs and the template function, as shown here:

    ```cpp
    int main() {
        EUBankAccount euAccount1{"IBAN1", 1000};
        EUBankAccount euAccount2{"IBAN2", 2000};
        std::cout << "The greater amount between EU accounts is " << getMaxAmount(euAccount1, euAccount2) << std::endl;
        UKBankAccount ukAccount1{"SORT1", "ACCOUNT_NUM1", 2500};
        UKBankAccount ukAccount2{"SORT2", "ACCOUNT_NUM2", 1500};
        std::cout << "The greater amount between UK accounts is " << getMaxAmount(ukAccount1, ukAccount2) << std::endl;
    }
    ```

    The output is as follows:

    ```cpp
    The greater amount between EU accounts is 2000
    The greater amount between UK accounts is 2500
    ```

### Using Template Type Parameters

As we saw earlier, the compiler uses the template as a guide to generate a template instance with some concrete type when the template is used.

This means that we can use the type as a *concrete* type, including applying type modifiers to it.

We saw earlier, a type can be modified by making it constant with the `const` modifier, and we can also take a reference to an object of a specific type by using the *reference* modifier:

```cpp
template<typename T>
T createFrom(const T& other) {
    return T(other);
}
```

Here, we can see a `template` function that creates a new object from a different instance of an object.

Since the function does not modify the original type, the function would like to accept it as a `const` reference.

Since we are declaring the type `T` in the template, in the function definition we can use the modifiers on the type to accept the parameter in the way we deem more appropriate.

Notice that we used the type two times: once with some modifiers and once with no modifiers.

This gives a lot of flexibility when using templates and writing functions, as we can liberally modify the type to suit our needs.

Similarly, we have a lot of freedom in where we can use the template arguments.

Let's see two templates with a multiple template type argument:

```cpp
template<typename A, typename B>
A transform(const B& b) {
    return A(b);
}
template<typename A, typename B>
A createFrom() {
  B factory;
  return factory.getA();
}
```

We can see that we can use the template argument in the function parameter, in the return type, or instantiate it directly in the function body.

Also, the order in which the template arguments are declared does not impact where and how the template parameters can be used.

### Requirements of Template Parameter Types

In the code snippet at the beginning of this chapter, we wrote some templates that accept any kind of type. In reality, our code does not work for any kind of type; for example: `max()` requires the types to support the `<` operation.

We can see that there were some requirements on the type.

Let's try to understand what having a requirement on a type means when using templates in C++ code. We will do so by using the following template code:

```cpp
template<typename Container, typename User>
void populateAccountCollection (Container& container, const User& user) {
  container.push_back(user.getAccount());
}
```

We can then write the following function as main and compile the program:

```cpp
int main() {
  // do nothing
}
```

When we compile this program, the compilation ends successfully without any error.

Let's say we change the `main` function to be the following:

```cpp
int main() {
  std::string accounts;
  int user;
  populateAccountCollection(accounts, user);
}
```

#### Note

We did not specify the type to the template. We will see later in this chapter when the compiler can automatically deduce the types from the call.

The compiler will give us an error when we compile it:

```cpp
error: request for member 'getAccount' in 'user', which is of non-class type 'const int'
```

Note how the error appeared when we used the template function, and that it was not detected before.

The error is telling us that we tried to call the `getAccount` method on an integer, which does not have such a method.

Why didn't the compiler tell us this when we were writing the template?

The reason for this is that the compiler does not know what type `User` will be; therefore, it cannot tell whether the `getAccount` method will exist or not.

When we tried to use the template, we tried to generate the code with two specific types, and the compiler checked that these two types were suitable for the template; they were not, and the compiler gave us an error.

The types we used were not satisfying the requirements of the template types.

Unfortunately, there is no easy way in the current C++ standard, even the most recent C++17, to specify the requirements of templates in the code—for that, we need good documentation.

The template has two type arguments, so we can look at the requirements for each type:

*   `User` object must have a `getAccount` method
*   `Container` object must have a `push_back` method

The compiler finds the first problem when we call the `getAccount()` function and it notifies us.

To solve this issue, let's declare a suitable class, as shown here:

```cpp
struct Account {
  // Some fields
};
class User {
public:
  Account getAccount() const{ 
    return Account();
 }
};
```

Now, let's call the template with the help of the following code:

```cpp
int main() {
  std::string accounts;
  User user;
  populateAccountCollection(accounts, user);
}
```

We still get an error:

```cpp
error: no matching function for call to 'std::__cxx11::basic_string<char>::push_back(Account)'
```

This time, the error message is less clear, but the compiler is telling us that there is no method called `push_back` that accepts an account in `basic_string<char>` (`std::string` is an alias for it). The reason for this is that `std::string` has a method called `push_back`, but it only accepts characters. Since we are calling it with an `Account`, it fails.

We need to be more precise in the requirements for our template:

*   `getAccount` method that returns an object
*   `push_back` method that accepts objects of the type returned by `getAccount` on the user

    #### Note

    The `std::vector` type in the C++ standard library allows to store sequences of elements of an arbitrary type. `push_back` is a method that's used for adding a new element at the end of the vector. We will see more about vectors in *Chapter 5*, *Standard Library Containers and Algorithms*.

We now change the calling code to consider all the requirements:

```cpp
#include <vector>
int main(){
   std::vector<Account> accounts;
   User user;
   populateAccountCollection(accounts, user);
}
```

This time, the code compiles correctly!

This shows us how the compiler checks most of the errors, but only when we instantiate the template.

It is also very important to clearly document the requirements of the template so that the user does not have to read complicated error messages to understand which requirement is not respected.

#### Note

To make it easy to use our templates with many types, we should try to set the least requirements we can on the types.

## Defining Function and Class Templates

In the previous section, we saw the advantages of templates in writing abstractions. In this section, we are going to explore how we can effectively use templates in our code to create **templated functions** and **templated classes**.

### Function Template

In the previous section, we learned how function templates are written.

In this section, we will learn about the two features that were introduced by C++11 that make it easier to write template functions. These two functions are trailing return types and `decltype`.

Let's start with the `decltype`. The `decltype` is a keyword that accepts an expression and returns the type of that expression. Let's examine the following code:

```cpp
int x;
decltype(x) y;
```

In the previous code, `y` is declared as an integer, because we are using the type of the expression `x`, which is `int`.

Any expression can be used inside `decltype`, even complex ones, for example:

```cpp
User user;
decltype(user.getAccount()) account;
```

Let's look at the second feature—**trailing return types**.

We saw that a function definition starts with the return type, followed by the name of the function and then the parameters. For example:

```cpp
int max(int a, int b);
```

Starting from C++11, it is possible to use a trailing return type: specifying the return type at the end of the function signature. The syntax to declare a function with a trailing return type is to use the keyword `auto`, followed by the name of the function and the parameters, and then by an *arrow* and the *return type*.

The following is an example of a trailing return type:

```cpp
auto max(int a, int b) -> int;
```

This is not beneficial when writing regular functions, but it becomes useful when writing templates and when combined with `decltype`.

The reason for this is that `decltype` has access to the variables defined in the parameters of the function, and the return type can be computed from them:

```cpp
template<typename User>
auto getAccount(User user) -> decltype(user.getAccount());
```

This is an example of a `forward declaration` of a function template.

#### Note

When the user wants to provide a definition, it needs to provide the same template declaration, followed by the body of the function.

Without the trailing return type, we would have to know what the type returned by `user.getAccount()` is to use it as the return type of the `getAccount()` function. The return type of `user.getAccount()` can be different depending on the type of the template parameter `User`, which in turn means that the return type of the `getAccount` function could change depending on the `User` type. With the trailing return type, we don't need to know what type is returned by `user.getAccount()`, as it is determined automatically. Even better, when different types are used in our function or a user changes the return type of the `getAccount` method in one of the types that's used to instantiate the template, our code will handle it automatically.

More recently, C++14 introduced the ability to simply specify `auto` in the function declaration, without the need for the trailing return type:

```cpp
auto max(int a, int b)
```

The return type is automatically deduced by the compiler, and to do so, the compiler needs to see the definition of the function—we cannot forward declare functions that return `auto`.

Additionally, `auto` always returns a value—it never returns a reference: this is something to be aware of when using it, as we could unintentionally create copies of the returned value.

One last useful feature of function templates is how to reference them without calling them.

Up until now, we have only seen how to call the function templates, but C++ allows us to pass functions as parameters as well. For example: when sorting a container, a custom comparison function can be provided.

We know that a template is just a blueprint for a function, and the real function is going to be created only when the template is instantiated. C++ allows us to instantiate the template function even without calling it. We can do this by specifying the name of the template function, followed by the template parameters, without adding the parameters for the call.

Let's understand the following example:

```cpp
template<typename T>
void sort(std::array<T, 5> array, bool (*function)(const T&, const T&));
```

The `sort` is a function that takes an array of five elements and a pointer to the function to compare two elements:

```cpp
template<typename T>
bool less(const T& a, const T& b) {
  return a < b;
}
```

To call `sort` with an instance of the `less` template for integers, we would write the following code:

```cpp
int main() {
  std::array<int, 5> array = {4,3,5,1,2};
  sort(array, &less<int>);
}
```

Here, we take a pointer to the instance of `less` for integers. This is particularly useful when using the Standard Template Library, which we will see in *Chapter 5*, *Standard Library Containers and Algorithms*.

### Class Templates

In the previous section, we learned how to write template functions. The syntax for class templates is equivalent to the one for functions: first, there is the template declaration, followed by the declaration of the class:

```cpp
template<typename T>
class MyArray {
  // As usual
};
```

And equivalently to functions, to instantiate a class template, we use the angle brackets containing a list of types:

```cpp
MyArray<int> array;
```

Like functions, class template code gets generated when the template is instantiated, and the same restrictions apply: the definition needs to be available to the compiler and some of the error-checking is executed when the template is instantiated.

As we saw in *Lesson 3*, *Classes*, while writing the body of a class, the name of the class is sometimes used with a special meaning. For example, the name of the constructor functions must match the name of the class.

In the same way, when writing a class template, the name of the class can be used directly, and it will refer to the specific template instance being created:

```cpp
template<typename T>
class MyArray {
  // There is no need to use MyArray<T> to refer to the class, MyArray automatically refers to the current template instantiation
  MyArray();
  // Define the constructor for the current template T
  MyArray<T>();
  // This is not a valid constructor.
};
```

This makes writing template classes a similar experience to writing regular classes, with the added benefit of being able to use the template parameters to make the class work with generic types.

Like regular classes, template classes can have fields and methods. The field can depend on the type declared by the template. Let's review the following code example:

```cpp
template<typename T>
class MyArray {
  T[] internal_array;
};
```

Also when writing methods, the class can use the type parameter of the class:

```cpp
template<typename T>
class MyArray {
  void push_back(const T& element);
};
```

Classes can also have templated methods. Templated methods are similar to template functions, but they can access the class instance data.

Let's review the following example:

```cpp
template<typename T>
class MyArray {
  template<typename Comparator>
  void sort (const Comparator & element);
};
```

The `sort` method will accept any type and will compile if the type satisfies all the requirements that the method imposes on the type.

To call the method, the syntax follows the one for calling functions:

```cpp
MyArray<int> array;
MyComparator comparator;
array.sort<MyComparator>(comparator);
```

#### Note

The method template can be part of a non-template class.

In these situations, the compiler can sometimes deduce the type of the parameter, where the user does not have to specify it.

If a method is only declared in the class, as we did in the example with `sort`, the user can later implement it by specifying the template types of both the class and the method:

```cpp
template<typename T> // template of the class
template<typename Comparator> // template of the method
MyArray<T>::sort(const Comparator& element) {
  // implementation
}
```

The name of the types does not have to match, but it is a good practice to be consistent with the names.

Similar to methods, the class can also have templated overloaded operators. The approach is identical to writing the operator overloads for regular classes, with the difference that the declaration of a template must precede the overload declaration like we saw for method templates.

Finally, something to be aware of is how static methods and static fields interact with the class template.

We need to remember that the template is a guide on the code that will be generated for the specific types. This means that when a template class declares a static member, the member is shared only between the instantiations of the template with the same template parameters:

```cpp
template<typename T>
class MyArray {
  const Static int element_size = sizeof(T);
};
MyArray<int> int_array1;
MyArray<int> int_array2;
MyArray<std::string> string_array;
```

`int_array1` and `int_array2` will share the same static variable, `element_size`, since they are both of the same type: `MyArray<int>`. On the other hand, `string_array` has a different one, because its class type is `MyArray<std::string>`. `MyArray<int>` and `MyArray<std::string>`, even if generated from the same class template, are two different classes, and thus do not share static fields.

### Dependent Types

It's fairly common, especially for code that interacts with templates, to define some public aliases to types.

A typical example would be the `value_type` `type alias` for containers, which specifies the type contained:

```cpp
template<typename T>
class MyArray {
public:
  using value_type = T;
};
```

Why is this being done?

The reason for this is that if we are accepting a generic array as a template parameter, we might want to find out the contained type.

If we were accepting a specific type, this problem would not arise. Since we know the type of vector, we could write the following code:

```cpp
void createOneAndAppend(std::vector<int>& container) {
  int new_element{}; // We know the vector contains int
  container.push_back(new_element);
}
```

But how can we do this when we accept any container that provides the `push_back` method?

```cpp
template<typename Container>
void createOneAndAppend(Container& container) {
  // what type should new_element be?
  container.push_back(new_element);
}
```

We can access the `type alias` declared inside the container, which specifies which kind of values it contains, and we use it to instantiate a new value:

```cpp
template<typename Container>
void createOneAndAppend(Container& container) {
  Container::value_type new_element;
  container.push_back(new_element);
}
```

This code, unfortunately, does not compile.

The reason for this is that `value_type` is a **dependent type**. A dependent type is a type that is derived from one of the template parameters.

When the compiler compiles this code, it notices that we are accessing the `value_type` identifier in the `Container` class.

That could either be a static field or a `type alias`. The compiler cannot know when it parses the template, since it does not know what the `Container` type will be and whether it has a `type alias` or a static variable. Therefore, it assumes we are accessing a static value. If this is the case, the syntax we are using is not valid, since we still have `new_element{}` after access to the field.

To solve this issue, we can tell the compiler that we are accessing a type in the class, and we do so by prepending the `typename` keyword to the type we are accessing:

```cpp
template<typename Container>
void createOneAndAppend(Container& container) {
  typename Container::value_type new_element{};
  container.push_back(new_element);
}
```

### Activity 13: Reading Objects from a Connection

The user is creating an online game which require to send and receive its current state over an internet connection. The application has several types of connections (TCP, UDP, socket) each of them has a `readNext()` method which returns an `std::array` of 100 chars containing the data inside the connection, and a `writeNext()` method which takes an `std::array` of 100 characters which puts data into the connection.

Let's follow these steps to create our online application:

1.  The objects that the application wants to send and receive over the connection have a `serialize()` static method which takes an instance of the object and return an `std::array` of 100 characters representing the object.

    ```cpp
    class UserAccount {
    public:
        static std::array<char, 100> serialize(const UserAccount& account) {
            std::cout << "the user account has been serialized" << std::endl;
            return std::array<char, 100>();
        }
        static UserAccount deserialize(const std::array<char, 100>& blob) {
            std::cout << "the user account has been deserialized" << std::endl;
            return UserAccount();
        }
    };
    class TcpConnection {
    public:
        std::array<char, 100> readNext() {
            std::cout << "the data has been read" << std::endl;
            return std::array<char, 100>{};
        }
        void writeNext(const std::array<char, 100>& blob) {
            std::cout << "the data has been written" << std::endl;
        }
    };
    ```

2.  The `deserialize()` static method takes an `std::array` of 100 characters representing the object, and creates an object from it.
3.  The connection objects are already provided. Create the header `connection.h` with the following declarations:

    ```cpp
    template<typename Object, typename Connection>
    Object readObjectFromConnection(Connection& con) {
      std::array<char, 100> data = con.readNext();
      return Object::deserialize(data);
    }
    ```

4.  Write a function template called `readObjectFromConnection` that takes a connection as the only parameter and the type of the object to read from the connection as a template type parameter. The function returns an instance of the object constructed after deserializing the data in the connection.
5.  Then, call the function with an instance of the `TcpConnection` class, extracting an object of type `UserAccount`:

    ```cpp
    TcpConnection connection;
    UserAccount userAccount = readObjectFromConnection<UserAccount>(connection);
    ```

The aim is to be able to send the information on the account of a user to the other users connected to the same online game, so that they can see the user information like their username and the level of their character.

#### Note

The solution for this activity can be found on page 295.

### Activity 14: Creating a User Account to Support Multiple Currencies

Write a program that supports and stores multiple currencies. Follow these steps:

1.  We want to create an `Account` class that stores the account balance in different currencies.
2.  A `Currency` is a class that represents a certain value in a specific currency. It has a public field called `value` and a template function called `to()` that takes the argument as a `Currency` type and returns an instance of that currency with the value set to the appropriate conversion of the current value of the class:

    ```cpp
    struct Currency {
        static const int conversionRate = CurrencyConversion;
        int d_value;
        Currency(int value): d_value(value) {}
    };
    template<typename OtherCurrency, typename SourceCurrency>
    OtherCurrency to(const SourceCurrency& source) {
        float baseValue = source.d_value / float(source.conversionRate);
        int otherCurrencyValue = int(baseValue * OtherCurrency::conversionRate);
        return OtherCurrency(otherCurrencyValue);
    }
    using USD = Currency<100>;
    using EUR = Currency<87>;
    using GBP = Currency<78>;
    template<typename Currency>
    class UserAccount {
    public:
      Currency balance;
    };
    ```

3.  Our aim is to write an `Account` class that stores the current balance in any currency provided by the `template` parameter.
4.  The user account must provide a method called `addToBalance` that accepts any kind of currency, and after converting it to the correct currency that's used for the account, it should sum the value to the balance:

    ```cpp
    template<typename OtherCurrency>
      void addToBalance(OtherCurrency& other) {
        balance.value += to<Currency>(other).value;
      }
    ```

5.  The user now understands how to write class templates, how to instantiate them, and how to call their templates.

    #### Note

    The solution for this activity can be found on page 296.

## Non-Type Template Parameters

We learned how templates allow you to provide the types as parameters and how we can make use of this to write generic code.

Templates in C++ have an additional feature—**non-type template parameters**.

A non-type template parameter is a template parameter that is not a type—it is a value.

We made use of such non-type template parameters many times when using `std::array<int, 10>;`.

Here, the second parameter is a non-type template parameter, which represents the size of the array.

The declaration of a non-type template parameter is in the parameter list of the template, but instead of starting with a `typename` keyword such as the type parameters, it starts with the type of the value, followed by the identifier.

There are strict restrictions on the types that are supported as non-type template parameters: they must be of integral type.

Let's examine the following example of the declaration of a non-type template parameter:

```cpp
template<typename T, unsigned int size>
Array {
  // Implementation
};
```

For example: here, we declared a class template that takes a type parameter and a non-type parameter.

We already saw that functions can take parameters directly and classes can accept parameters in the constructor. Additionally, the type of regular parameters is not restricted to be an integral type.

What is the difference between template and non-template parameters? Why would we use a non-type template parameter instead of a regular parameter?

The main difference is when the parameter is known to the program. Like all the template parameters and unlike the non-template parameters, the value must be known at compile time.

This is useful when we want to use the parameters in expressions that need to be evaluated at compile time, as we do when declaring the size of an array.

The other advantage is that the compiler has access to the value when compiling the code, so it can perform some computations during compilation, reducing the amount of instruction to execute at runtime, thus making the program faster.

Additionally, knowing some values at compile time allows our program to perform additional checks so that we can identify problems when we compile the program instead of when the program is executed.

### Activity 15: Writing a Matrix Class for Mathematical Operations in a Game

In a game, it is common to represent the orientation of a character in a special kind of matrix: a *quaternion*. We would like to write a `Matrix` class that will be the base of the mathematical operations inside our game.

Our `Matrix` class should be a template that accepts a type, a number of rows, and a number of columns.

We should store the elements of the matrix inside an `std::array`, stored inside the class.

The class should have a method called `get()` that takes a row and a column, and returns a reference to the element in that position.

If the row or column is outside of the matrix, we should call `std::abort()`.

Let's follow these steps:

1.  The `Matrix` class takes three template parameters—one type and the two dimensions of the `Matrix` class. The dimensions are of type `int`.

    ```cpp
    template<typename T, int R, int C>
    class Matrix {
        // We store row_1, row_2, ..., row_C
        std::array<T, R*C> data;
        public:
            Matrix() : data({}) {}
    };
    ```

2.  Now, create a `std::array` with a size of the number of rows times the number of columns so that we have enough space for all the elements of the matrix.
3.  Add a constructor to initialize the array:
4.  We add a `get()` method to the class to return a reference to the element `T`. The method needs to take the row and column we want to access.
5.  If the index are outside of the bounds of the matrix, we call `std::abort()`. In the array, we store all the elements of the first row, then all the elements of the second row, and so on. So, when we want to access the elements of the nth row, we need to skip all the elements of the previous rows, which are going to be the number of elements per row (so the number of columns) times the previous rows:

    ```cpp
    T& get(int row, int col) {
      if (row >= R || col >= C) {
        std::abort();
      }
      return data[row*C + col];
    }
    ```

    The output is as follows:

    ```cpp
    Initial matrix:
    1 2 
    3 4 
    5 6 
    ```

    #### Note

    The solution for this activity can be found on page 298.

**Bonus step:**

In games, multiplying a matrix by a vector is a common operation.

Add a method to the class that takes a `std::array` containing elements of the same type of the matrix, and returns a `std::array` containing the result of the multiplication. See the definition of a matrix-vector product at [https://mathinsight.org/matrix_vector_multiplication](https://mathinsight.org/matrix_vector_multiplication).

**Bonus step**:

We add a new method, `multiply`, which takes a `std::array` of type `T` with the length of `C` by const reference, since we are not modifying it.

The function returns an array of the same type, but a length of `R`?

We follow the definition of the matrix-vector multiplication to compute the result:

```cpp
std::array<T, R> multiply(const std::array<T, C>& vector){
    std::array<T, R> result = {};
    for(int r = 0; r < R; r++) {
      for(int c = 0; c < C; c++) {
        result[r] += get(r, c) * vector[c];
      }
    }  
    return result;
  }
```

## Making Templates Easier to Use

We always said that we need to provide the template arguments to the parameters of a template function or class. Now, in this section, we are going to see two features that C++ offers to make it easier to use templates.

These features are default template arguments and template argument deduction.

### Default Template Arguments

Like function arguments, template arguments can also have default values, both for type and non-type template parameters.

The syntax for default template arguments is to add after the template identifier the equal, followed by the value:

```cpp
template<typename MyType = int>
void foo();
```

When a template provides a default value for a parameter, the user does not have to specify the parameter when instantiating the template. The default parameter must come after the parameters that do not have a default value.

Additionally, you can reference the previous template parameters when defining the default type for a subsequent template parameter.

Let's see some examples of both errors and valid declarations:

```cpp
template<typename T = void, typename A>
void foo();
```

*   `T`, which has a default type, comes before the template parameter `A`, which does not have a default parameter:

    ```cpp
    template<typename T = A, typename A = void>
    void foo();
    ```

*   `T` references the template parameter `A`, which comes after `T`:

    ```cpp
    template<typename T, typename A = T >
    void foo();
    ```

*   `A` has a default value, and no other template parameter without default value comes after it. It also references `T`, which is declared before the template parameter `A`.

The reason to use the default arguments is to provide a sensible option for the template, but still allowing the user to provide their own type or value when needed.

Let's see an example of type arguments:

```cpp
template<typename T>
struct Less {
  bool operator()(const T& a, const T& b) {
    return a < b;
  }
};
template<typename T, typename Comparator= Less<T>>
class SortedArray;
```

The hypothetical type `SortedArray` is an array that keeps its elements always sorted. It accepts the type of the elements it should hold and a comparator. To make it easy to use for the user, it sets the comparator to use the `less` operator by default.

The following code shows how a user can implement it:

```cpp
SortedArray<int> sortedArray1;
SortedArrat<int, Greater<int>> sortedArray2;
```

We can also see an example of a default non-type template parameter:

```cpp
template<size_t Size = 512>
struct MemoryBuffer;
```

The hypothetical type `MemoryBuffer` is an object that reserves an amount of memory on the stack; the program will then allocate objects into that memory. By default, it uses 512 bytes of memory, but the user can specify a different size:

```cpp
MemoryBuffer<> buffer1;
MemoryBuffer<1024> buffer2;
```

Note the empty angle brackets in the `buffer1` declaration. They are needed to signal to the compiler that we are making use of a template. This requirement has been removed in C++17, and we can write `MemoryBuffer buffer1;`.

### Template Argument Deduction

All the template parameters need to be known to instantiate a template, but not all of them need to be explicitly provided by the caller.

**Template argument deduction** refers to the ability of the compiler to automatically understand some of the types that are used to instantiate the template, without the user having to explicitly type them.

We are going to see them for functions as that is supported by most of the versions of C++. C++17 introduced **deduction guides**, which allow the compiler to perform template argument deduction for class templates from the constructor, but we are not going to see them.

The detailed rules for template argument deduction are very complex, and so we are going to see them by example so that we can understand them.

In general, the compiler tries to find the type for which the provided argument and the parameter match the closest.

The code we are going to analyze is as follows:

```cpp
template<typename T>
void foo(T parameter);
```

The calling code is as follows:

```cpp
foo(argument);
```

### Parameter and Argument Types

We are going to see how, based on different pairs of parameters and arguments, the type is deduced:

![Figure 4.1: Different parameter and argument types](img/C11557_04_01.jpg)

###### Figure 4.1: Different parameter and argument types

The error happens because we cannot bind a temporary value, like 1, to a non-`const` reference.

As we can see, the compiler tries to deduce a type so that when it is substituted in the parameter, it matches the argument as best as possible.

The compiler cannot always find such a type; in those situations, it gives an error and it's up to the user to provide the type.

The compiler cannot deduce a type for any of the following reasons:

The type is not used in the parameters. For example: the compiler cannot deduce a type if it is only used in the return type, or only used inside the body of the function.

The type in the parameter is a derived type. For example: `template<typename T> void foo(T::value_type a)`. The compiler cannot find the type `T` given the parameter that's used to call the function.

Knowing these rules, we can derive a best practice for the order of the template parameters when writing templates: the types that we expect the user to provide need to come before the types that are deduced.

The reason for this is that a user can only provide the template arguments in the same order they have been declared.

Let's consider the following template:

```cpp
template<typename A, typename B, typename C>
C foo(A, B);
```

When calling `foo(1, 2.23)`, the compiler can deduce `A` and `B`, but cannot deduce `C`. Since we need all the types, and the user has to provide them in order, the user has to provide all of the types: `foo<int, double, and float>(1, 2.23);`.

Let's say we put the types that cannot be deduced before the types that can be deduced, as in the following example:

```cpp
template< typename C, typename A, typename B>
C foo(A, B);
```

We could call the function with `foo<float>(1, 2.23)`. We would then provide the type to use for `C` and the compiler would automatically deduce `A` and `B`.

In a similar way, we need to reason about default template arguments.

Since they need to come last, we need to make sure to put the types that the user is more likely to want to modify first, since that will force them to provide all the template arguments up to that parameter.

### Activity 16: Making the Matrix Class Easier to Use

The `Matrix` class we created in *Activity 15: Writing a Matrix Class for Mathematical Operations in a Game,* requires that we provide three template parameters.

Now, in this activity, we want to make the class easier to use by requiring that the user is required to only pass two parameters: the number of rows and the number of columns in the `Matrix` class. The class should also take a third argument: the type contained in the `Matrix` class. If not provided, it should default to `int`.

In the previous activity, we added to the matrix a `multiply` operation. We now want to let the user customize the function by specifying how the multiplication between the types should be executed. By default, we want to use the `*` operator. For that, a `class` template named `std::multiplies` from the `<functional>` header exists. It works like the `Less` class we saw previously in this chapter:

1.  We start by importing `<functional>` so that we have access to `std::multiplies`.
2.  We then change the order of the template parameters in the class template so that the size parameters come first. We also add a new template parameter, `Multiply`, which is the type we use for computing the multiplication between the elements in the vector by default, and we store an instance of it in the class.
3.  We now need to make sure that the `multiply` method uses the `Multiply` type provided by the user to perform the multiplication.
4.  To do so, we need to make sure we call `multiplier(operand1, operand2)` instead of `operand1 * operand2` so that we use the instance we stored inside the class:

    ```cpp
    std::array<T, R> multiply(const std::array<T, C>& vector) {
        std::array<T, R> result = {};
        for(int r = 0; r < R; r++) {
            for(int c = 0; c < C; c++) {
                result[r] += multiplier(get(r, c), vector[c]);
            }
        }
        return result;
    }
    ```

5.  Add an example of how we can use the class:

    ```cpp
    // Create a matrix of int, with the 'plus' operation by default
    Matrix<3, 2, int, std::plus<int>> matrixAdd;
    matrixAdd.setRow(0, {1,2});
    matrixAdd.setRow(1, {3,4});
    matrixAdd.setRow(2, {5,6});
    std::array<int, 2> vector = {8, 9};
    // This will call std::plus when doing the multiplication
    std::array<int, 3> result = matrixAdd.multiply(vector);
    ```

    The output is as follows:

    ```cpp
    Initial matrix:
    1 2 
    3 4 
    5 6 
    Result of multiplication (with plus instead of multiply): [20, 24, 28]
    ```

    #### Note

    The solution for this activity can be found on page 300.

## Being Generic in Templates

So far, we have learned how the compiler can make our templated functions easier to use by automatically deducing the types used. The template code decides whether to accept a parameter as a value or a reference, and the compiler finds the type for us. But what do we do if we want to be agnostic regarding whether an argument is a value or a reference, and we want to work with it regardless?

An example would be `std::invoke` in C++17\. `std::invoke` is a function that takes a function as the first argument, followed by a list of arguments, and calls the function with the arguments. For example:

```cpp
void do_action(int, float, double);
double d = 1.5;
std::invoke(do_action, 1, 1.2f, d);
```

Similar examples would apply if you wanted to log before calling a function, or you wanted to execute the function in a different thread, such as `std::async` does.

Let's demystify the difference by using the following code:

```cpp
struct PrintOnCopyOrMove {
  PrintOnCopyOrMove(std::string name) : _name(name) {}
  PrintOnCopyOrMove(const PrintOnCopyOrMove& other) : _name(other._name) { std::cout << "Copy: " << _name << std::endl; }
  PrintOnCopyOrMove(PrintOnCopyOrMove&& other) : _name(other._name) { std::cout << "Move: " << _name << std::endl; }

  std::string _name;
};
void use_printoncopyormove_obj(PrintOnCopyOrMove obj) {}
```

#### Note

`use_printoncopyormove_obj` always accepts the parameter by value.

Let's say we execute the following code:

```cpp
PrintOnCopyOrMove local{"l-value"};
std::invoke(use_printoncopyormove_obj, local);
std::invoke(use_printoncopyormove_obj, PrintOnCopyOrMove("r-value"));
```

The code would print the following:

```cpp
Copy: l-value
Move: r-value
```

How can we write a function such as `std::invoke` that works regardless of the kind of reference (colloquially referred to as "ref-ness", similarly to how "const-ness" is used to talk about whether a type is const qualified) of the parameters?

The answer to that is **forwarding references**.

Forwarding references look like r-value references, but they only apply where the type is deduced by the compiler:

```cpp
void do_action(PrintOnCopyOrMove&&)
// not deduced: r-value reference
template<typename T>
void do_action(T&&) // deduced by the compiler: forwarding reference
```

#### Note

If you see a type identifier declared in the template, the type is deduced, and the type has &&, then it is a forwarding reference.

Let's see how the deduction works for forwarding references:

![Figure 4.2: Forward reference function. ](img/C11557_04_02.jpg)

###### Figure 4.2: Forward reference function.

#### Note

Let's say the type is not deduced, but, it is provided explicitly, for example:

`int x = 0;`

`do_action<int>(x);`

Here, `T` will be `int`, since it was explicitly stated.

The advantage, as we saw before, is that we work with any kind of reference, and when the calling code knows it can move the object, then we can make use of the additional performance provided by the move constructor, but when a reference is preferred, then the code can use it as well.

Additionally, some types do not support copying, and we can make our template work with those types as well.

When we write the body of the template function, the parameter is used as an `l-value` reference, and we can write code ignoring whether `T` is an `l-value` reference or an `r-value` one:

```cpp
template<typename T>
void do_action(T&& obj) { /* forwarding reference, but we can access obj as if it was a normal l-value reference */
  obj.some_method();
  some_function(obj);
}
```

In *Chapter 3*, *Classes*, we learned that `std::move` can make our code more efficient when we need to use an object that we are not going to access after the call happens.

But we saw that we should never move objects we receive as an `l-value` reference parameter, since the code that called us might still use the object after we return.

When we are writing templates using a forwarding reference, we are in front of a dilemma: our type might be a value or a reference, so how do we decide whether we can use `std::move`?

Does it mean we cannot make use of the benefit that `std::move` brings us?

The answer, of course, is *no*:

```cpp
template<typename T>
void do_action(T&& obj) {
  do_something_with_obj(???); 
// We are not using obj after this call.
}
```

Should we use move or not in this case?

The answer is *yes*: we should move if `T` is a value, and, no, we should not move if `T` is a reference.

C++ provides us with a tool to do exactly this: `std::forward`.

`std::forward` is a function template that always takes an explicit template parameter and a function parameter: `std::forward<T>(obj)`.

`Forward` looks at the type of `T`, and if it's an `l-value` reference, then it simply returns a reference to the `obj`, but if it's not, then it is equivalent to calling `std::move` on the object.

Let's see it in action:

```cpp
template<typename T>
void do_action(T&& obj) {
  use_printoncopyormove_obj(std::forward<T>(obj)); 
}
```

Now, we call it by using the following code:

```cpp
PrintOnCopyOrMove local{"l-value"};
do_action(local);
do_action(PrintOnCopyOrMove("r-value"));
do_action(std::move(local));
// We can move because we do not use local anymore
```

When executed, the code will print the following output:

```cpp
Copy: l-val
Move: r-val
Move: l-val
```

We successfully managed to write code that is independent on whether the type is passed as reference or value, removing a possible requirement on the template type parameter.

#### Note

A template can have many type parameters. Forwarding references can apply to any of the type parameters independently.

This is important because the caller of the templated code might know whether it is better to pass values or pass references, and our code should work regardless of whether there is a requirement to ask for a specific ref-ness.

We also saw how we can still maintain the advantages of moving, which is required for some types that do not support copying. This can make our code run much faster, even for types that support copying, without complicating our code: when we have forwarding references we use `std::forward` where we would have used `std::move`.

### Activity 17: Ensuring Users are Logged in When Performing Actions on the Account

We want to allow the users of our e-commerce website to perform arbitrary actions (for the scope of this activity, they will be adding and removing items) on their shopping carts.

Before performing any action, we want to make sure that the user is logged in. Now, let's follow these instructions:

1.  Ensure that there is a `UserIdentifier` type for identifying the user, a `Cart` type that represents the shopping cart of the user, and a `CartItem` type that represents any item in the cart:

    ```cpp
    struct UserIdentifier {
        int userId = 0;
    };
    struct Cart {
        std::vector<Item> items;
    };
    ```

2.  Ensure that there is also a function with the signature `bool isLoggedIn(const UserIdentifier& user)` and a function to retrieve the cart for an user, `Cart getUserCart(const UserIdentifier& user)`:

    ```cpp
    bool isLoggedIn(const UserIdentifier& user) {
        return user.userId % 2 == 0;
    }
    Cart getUserCart(const UserIdentifier& user) {
        return Cart();
    }
    ```

3.  In most of our code, we only have access to the `UserIdentifier` for a user, and we want to make sure that we always check whether the user is logged in before doing any action on the cart.
4.  To solve this problem, we decide to write a function template called `execute_on_user_cart`, which takes the user identifier, an action, and a single parameter. The function will check if the user is logged in and if so, retrieve their cart, then perform the action of passing the cart and the single parameter:

    ```cpp
    template<typename Action, typename Parameter>
    void execute_on_user_cart(UserIdentifier user, Action action, Parameter&& parameter) {
        if(isLoggedIn(user)) {
            Cart cart = getUserCart(user);
            action(cart, std::forward<Parameter>(parameter));
        } else {
            std::cout << "The user is not logged in" << std::endl;
        }
    }
    ```

5.  One of the actions we want to perform is `void remove_item(Cart, CartItem)`. A second action we want to perform is `void add_items(Cart, std::vector<CartItem>)`:

    ```cpp
    void removeItem(Cart& cart, Item cartItem) {
        auto location = std::find(cart.items.begin(), cart.items.end(), cartItem);
        if (location != cart.items.end()) {
            cart.items.erase(location);
        }
        std::cout << "Item removed" << std::endl;
    }
    void addItems(Cart& cart, std::vector<Item> items) {
        cart.items.insert(cart.items.end(), items.begin(), items.end());
        std::cout << "Items added" << std::endl;
    }
    ```

    #### Note

    A parameter of a function template can be used to accept functions as parameters.

    The aim is to create a function that performs the necessary checks on whether the user is logged in so that throughout our program we can use it to perform safely any actions that are required by our business on the user cart, without the risk of forgetting to check the logged status of the user.

6.  We can also move the types that are not forwarding references:

    ```cpp
    template<typename Action, typename Parameter>
    void execute_on_user_cart(UserIdentifier user, Action action, Parameter&& parameter) {
        if(isLoggedIn(user)) {
            Cart cart = getUserCart(user);
            action(std::move(cart), std::forward<Parameter>(parameter));
        }
    }
    ```

7.  Examples of how the `execute_on_user_cart` function can be used with the actions we described earlier in the activity is as follows:

    ```cpp
    UserIdentifier user{/* initialize */};
    execute_on_user_cart(user, remove_item, CartItem{});
    std::vector<CartItem> items = {{"Item1"}, {"Item2"}, {"Item3"}}; // might be very long
    execute_on_user_cart(user, add_items, std::move(items));
    ```

8.  The developers in our software can write the functions they need to execute on the cart, and call `execute_on_user_cart` to safely execute them.

    #### Note

    The solution for this activity can be found on page 302.

## Variadic Templates

We just saw how we can write a template that accepts parameters independently from their ref-ness.

But the two functions we talked about from the standard library, `std::invoke` and `std::async`, have an additional property: they can accept any number of arguments.

In a similar way, `std::tuple`, a type similar to a `std::array` but that can contain values of different types, can contain an arbitrary number of types.

How is it possible for a template to accept an arbitrary number of arguments of different types?

In the past, a solution to this problem was to provide a great number of overloads for the same function, or multiple implementations of the class or struct, one for each number of the parameters.

This is clearly code that is not easy to maintain, as it forces us to write the same code multiple times. Another drawback is that there is a limit to the number of template parameters, so if your code requires more parameters than what is provided, you do not have a way to use the function.

C++11 introduced a nice solution for this problem: **parameter pack**.

A parameter pack is a template parameter that can accept zero or more template arguments.

A parameter pack is declared by appending `…` to the type of the template parameter.

Parameter packs are a functionality that works with any template: both functions and classes:

```cpp
template<typename… Types>
void do_action();
template<typename… Types>
struct MyStruct;
```

A template that has a parameter pack is called a **variadic template**, since it is a template that accepts a varying number of parameters.

When instantiating a variadic template, any number of arguments can be provided to the parameter pack by separating them with a comma:

```cpp
do_action<int, std:string, float>();
do_action<>();
MyStruct<> myStruct0;
MyStruct<float, int> myStruct2;
```

`Types` will contain the list of arguments that are provided when instantiating the template.

A parameter pack by itself is a list of types and the code cannot interact with it directly.

The variadic template can use the parameter pack by expanding it, which happens by appending `…` to a pattern.

When a pattern is expanded, it is repeated as many times as there are types in its parameter pack, separating it with a comma. Of course, to be expanded, a pattern must contain at least a parameter pack. If multiple parameters are present in the pattern, or the same parameter is present several times, they are all expanded at the same time.

The simplest pattern is the name of the parameter pack: `Types…`.

For example: to let a function accept multiple arguments, it would expand the parameter pack in the function arguments:

```cpp
template<typename… MyTypes>
void do_action(MyTypes… my_types);
do_action();
do_action(1, 2, 4.5, 3.5f);
```

When we call the function, the compiler automatically deduces the types of the parameter pack. In the last call, `MyTypes` will contain `int`, `double`, and `float`, and the signature of the generated function would be `void do_action(int __p0, int __p1, double __p2, float __p3)`.

#### Note

A parameter pack in the list of template parameters can only be followed by template parameters that have a default value, or those that are deduced by the compiler.

Most commonly, the parameter pack is the last in the list of template parameters.

The function parameter `my_types` is called a **function parameter pack** and needs to be expanded as well so that it can access the single parameters.

For example: let's write a variadic struct:

```cpp
template<typename… Ts>
struct Variadic {
  Variadic(Ts… arguments);
};
```

Let's write a function that creates the struct:

```cpp
template<typename… Ts>
Variadic<Ts…> make_variadic(Ts… args) {
  return Variadic<Ts…>(args…);
}
```

Here, we have a variadic function that takes a parameter pack and expands it when calling the constructor of another variadic struct.

The function `parameter packs`, which is the function variadic parameter, can be expanded only in some locations—the most common is as parameters when calling a function.

The template `parameter packs`, which is a type variadic parameter, can be expanded in template argument lists: the list of arguments between `<>` when instantiating a template.

As we mentioned previously, the pattern for the expansion might be more complex than just the name of the argument.

For example: we can access type aliases declared in the type or we can call a function on the parameter:

```cpp
template<typename… Containers>
std::tuple<typename Containers::value_type…> get_front(Containers… containers) {
  return std::tuple<typename Containers::value_type…>(containers.front()…);
}
```

We call it like so:

```cpp
std::vector<int> int_vector = {1};
std::vector<double> double_vector = {2.0};
std::vector<float> float_vector = {3.0f};
get_front(int_vector, double_vector, float_vector) // Returns a tuple<int, double, float> containing {1, 2.0, 3.0}
```

Alternatively, we can pass the parameter as an argument to a function:

```cpp
template<typename… Ts>
void modify_and_call (Ts… args) {
  do_things(modify (args)…));
}
```

This will call the `modify` function for each argument and pass the result to `do_things`.

In this section, we saw how the variadic parameter functionality of C++ lets us write functions and classes that work with any number and type of parameters.

While it is not a common everyday task to write variadic templates, almost every programmer uses a variadic template in their day-to-day coding, since it makes it so much easier to write powerful abstractions, and the standard library makes vast use of it.

Additionally, in the right situation, variadic templates can allow us to write expressive code that works in the multitude of situations we need.

### Activity 18: Safely Performing Operations on the User Cart with an Arbitrary Number of Parameters

In the previous activity, we saw a function, `execute_on_user_cart`, which allows us to execute arbitrary functions that take an object of type `Cart` and a single parameter.

In this activity, we want to expand on the supported types of actions we can perform on the shopping cart of the user by allowing any function that takes an object of type `Cart` and an arbitrary number of arguments:

1.  Expand the previous activity to accept any number of the parameter with any kind of ref-ness and pass it to the action provided.
2.  Write variadic templates and learn how to expand them:

    ```cpp
    template<typename Action, typename... Parameters>
    void execute_on_user_cart(UserIdentifier user, Action action, Parameters&&... parameters) {
        if(isLoggedIn(user)) {
            Cart cart = getUserCart(user);
            action(std::move(cart), std::forward<Parameters>(parameters)...);
        }
    }
    ```

    #### Note

    The solution for this activity can be found on page 303.

## Writing Easy-to-Read Templates

Up until now, we have seen many features that we can use to write powerful templates that allow us to create high-level abstractions over the specific problems we face.

But, as usual, code is more often read than written, and we should optimize for readability: the code should express the intentions of the code more than what operation is achieved.

Template code can sometimes make that hard to do, but there are a few patterns that can help.

### Type Alias

Type `name = type`.

After the declaration, everywhere *Name* is used is going to be equivalent to having used *Type*.

This is very powerful for three reasons:

*   It can give a shorter and more meaningful name to complex types
*   It can declare a nested type to simplify access to it
*   It allows you to avoid having to specify the `typename` keyword in front of a dependent type

Let's see examples for these two points.

Imagine we have a type, `UserAccount`, which contains several fields on the user, such as user ID, user balance, user email, and more.

We want to organize the user accounts into a high scoreboard based on their account balances to visualize which users are most actively using our service.

To do so. we can use a data structure that requires a few parameters: the type to store, a way for ordering the types, a way to compare the types, and possibly others.

The type could be as follows:

```cpp
template<typename T, typename Comparison = Less<T>, typename Equality = Equal<T>>
class SortedContainer;
```

To be easy to use, the template correctly provided some default values for `Comparison` and `Equality`, which use the `<` and `==` operators, but our `UserAccount` type does not implement the `<` operator, as there is no clear ordering, and the `==` operator does not do what we want, as we are only interested in comparing balances. To solve this, we implemented two structures to provide the functionality we need:

```cpp
SortedContainer<UserAccount, UserAccountBalanceCompare, UserAccountBalanceEqual> highScoreBoard;
```

The creation of a high scoreboard is both verbose.

Using a type alias, we could write the following:

```cpp
using HighScoreBoard = SortedContainer<UserAccount, UserAccountBalanceCompare, UserAccountBalanceEqual>;
```

Following this, we could create instances of `HighScoreBoard` directly, with little typing and clearly specify the intent:

```cpp
HighScoreBoard highScoreBoard;
```

We now also have a single place to update if we want to change the way in which we want to sort the accounts. For example: if we also wanted to consider how long the user has been registered in the service, we could change the comparator the comparator. Every user of the type alias will be updated, without the risk of forgetting to update one location.

Additionally, we clearly have a location where we can put the documentation on the decision made for using the type we picked.

#### Note

When using type aliases, give a name that represents what the type is for, not how it works. `UserAccountSortedContainerByBalance` is a not a good name because it tells us how the type works instead of what its intention is.

The second case is extremely useful for allowing code to introspect the class, that is, looking into some of the details of the class:

```cpp
template<typename T>
class SortedContainer {
public:
  T& front() const;
};
template<typename T>
class ReversedContainer {
public:
  T& front() const;
}
```

We have several containers, which mostly support the same operations. We would like to write a template function that takes any container and returns the first element, `front`:

```cpp
template<typename Container>
??? get_front(const Container& container);
```

How can we find out what type is returned?

A common pattern is to add a type alias inside the class, like so:

```cpp
template<typename T>
class SortedContainer {
  using value_type = T; // type alias
  T& front() const;
};
```

Now the function can access the type of the contained element:

```cpp
template<typename Container>
typename Container::value_type& get_front(const Container& container);
```

#### Note

Remember that `value_type` depends on the `Container` type, so it is a dependent type. When we use dependent types, we must use the `typename` keyword in `front`.

This way, our code can work with any type that declares the nested type `value_type`.

The third use case, that is, to avoid having to type the `typename` keyword repeatedly, is common when interacting with code that follows the previous pattern.

For example: we can have a class that accepts a type:

```cpp
template<typename Container>
class ContainerWrapper {
  using value_type = typename Container::value_type;
}
```

In the rest of the class, we can use `value_type` directly, without having to type `typename` anymore. This allows us to avoid a lot of repetitions.

The three techniques can also be combined. For example: you can have the following:

```cpp
template<typename T>
class MyObjectWrapper {
  using special_type = MyObject<typename T::value_type>;
};
```

### Template Type Alias

The ability to create type aliases, as described in the previous part of this chapter, is already very useful for improving the readability of our code.

C++ gives us the ability to define generic type aliases so that they can simply be reused by the users of our code.

A template alias is a template that generates aliases.

Like all the templates we saw in this chapter, they start with a template declaration and follow with the alias declaration, which can depend on the type that's declared in the template:

```cpp
template<typename Container>
using ValueType = typename Container::value_type;
```

A `ValueType` is a template alias that can be instantiated with the usual template syntax: `ValueType<SortedContainer> myValue;`.

This allows the code to just use the alias `ValueType` whenever they want to access the `value_type` type inside any container.

Template aliases can combine all the features of templates: they can accept multiple parameters, accept non-type parameters, and even use parameter packs.

## Summary

In this chapter, the students were introduced to templates in C++. We saw that templates exist to create high-level abstractions that work independently from the types of the objects at zero overhead at runtime. We explained the concept of type requirements: the requirements a type must satisfy to work correctly with the templates. We then showed the students how to write function templates and class templates, mentioning dependent types as well, to give the students the tools to understand a class of errors that happen when writing template code.

We then showed how templates can work with non-type parameters, and how templates can be made easier to use by providing default template arguments, thanks to template argument deduction.

We then showed the students how to write more generic templates, thanks to the forwarding reference, `std::forward`, and the template parameter pack.

Finally, we concluded with some tools to make templates easier to read and more maintainable.

In the next chapter, we will cover standard library containers and algorithms.