# *Appendix*

## About

This section is included to assist the students to perform the activities in the book. It includes detailed steps that are to be performed by the students to achieve the objectives of the activities.

## Lesson 1: Getting Started

### Activity 1: Find the Factors of 7 between 1 and 100 Using a while Loop

1.  Import all the required header files before the `main` function:

    ```cpp
    #include <iostream>
    ```

2.  Inside the `main` function, create a variable `i` of type `unsigned`, and initialize its value as `1`:

    ```cpp
    unsigned i = 1;
    ```

3.  Now, use the `while` loop adding the logic where the value of `i` should be less than `100`:

    ```cpp
    while ( i < 100){ }
    ```

4.  In the scope of the `while` loop, use the if statement with the following logic:

    ```cpp
    if (i%7 == 0) {
       std::cout << i << std::endl;
    }
    ```

5.  Increase the value of the `i` variable to iterate through the `while` loop to validate the condition:

    ```cpp
    i++;
    ```

    The output of the program is as follows:

    ```cpp
    7
    14
    21
    28
    ...
    98
    ```

### Activity 2: Define a Bi-Dimensional Array and Initialize Its Elements

1.  After creating a C++ file, include the following header file at the start of the program:

    ```cpp
    #include <iostream>
    ```

2.  Now, in the `main` function, create a bi-directional array named `foo` of type integer, with three rows and three columns, as shown here:

    ```cpp
    int main()
    {
      int foo[3][3];
    ```

3.  Now, we will use the concept of a nested `for` loop to iterate through each index entry of the `foo` array:

    ```cpp
    for (int x= 0; x < 3; x++){
      for (int y = 0; y < 3; y++){
      }
    }
    ```

4.  In the second `for` loop, add the following statement:

    ```cpp
    foo[x][y] = x + y;
    ```

5.  Finally, iterate over the array again to print its values:

    ```cpp
    for (int x = 0; x < 3; x++){
       for (int y = 0; y < 3; y++){
          std::cout << “foo[“ << x << “][“ << y << “]: “ << foo[x][y] << std::endl;
       }
    }
    ```

    The output is as follows:

    ```cpp
    foo[0][0]: 0
    foo[0][1]: 1
    foo[0][2]: 2
    foo[1][0]: 1
    foo[1][1]: 2
    foo[1][2]: 3
    foo[2][0]: 2
    foo[2][1]: 3
    foo[2][2]: 4
    ```

## Lesson 2: Functions

### Activity 3: Calculating if a Person is Eligible to Vote or Not

1.  Include the header file in the program to print the output as shown here:

    ```cpp
    #include <iostream>
    ```

2.  Now, create a function named `byreference_age_in_5_years` and the `if` loop with the following condition to print the message:

    ```cpp
    void byreference_age_in_5_years(int& age) {
      if (age >= 18) {
        std::cout << “Congratulations! You are eligible to vote for your nation.” << std::endl;
        return;
    ```

3.  Add the `else` block to provide another condition if the age of the user is less than 18 years:

    ```cpp
      } else{
        int reqAge = 18;
        int yearsToGo = reqAge-age;
        std::cout << “No worries, just “<< yearsToGo << “ more years to go.” << std::endl;
      }
    }
    ```

4.  In the `main` function, create a variable of type integer and pass it as a reference in the `byreference_age_in_5_years` function as shown:

    ```cpp
    int main() {
        int age;
        std::cout << “Please enter your age:”;
        std::cin >> age;
        byreference_age_in_5_years(age);
    }
    ```

### Activity 4: Apply the Understanding of Passing by Reference or Value in Functions

1.  After adding all the required header files, create the first function of type integer as shown here:

    ```cpp
    int sum(int a, int b)
    {
      return a + b
    }
    ```

    Take by value, return by value, since the types are small in memory and there is no reason to use references.

2.  The second function should be written as follows:

    ```cpp
    int& getMaxOf(std::array<int, 10>& array1, std::array<int, 10>& array2, int index) {
      if (array1[index] >= array2[index]) {
        return array1[index];
      } else {
        return array2[index];
      }
    }
    ```

### Activity 5: Organizing Functions in Namespaces

1.  Include the required header file and namespace to print the required output:

    ```cpp
    #include <iostream>
    using namespace std;
    ```

2.  Now, create a namespace named `LamborghiniCar` with the following `output` function:

    ```cpp
    namespace LamborghiniCar
    {
      int output(){
        std::cout << “Congratulations! You deserve the Lamborghini.” << std::endl;
        return NULL;
      }
    }
    ```

3.  Create another namespace named `PorscheCar` and add an `output` function as shown:

    ```cpp
    namespace PorscheCar
    {
      int output(){
        std::cout << “Congratulations! You deserve the Porsche.” << std::endl; 
        return NULL;
      }
    }
    ```

In the main function, create a variable named `magicNumber` of type integer to accept the input from the user:

```cpp
int main()
{
  int magicNumber;
  std::cout << “Select a magic number (1 or 2) to win your dream car: “;
  std::cin >> magicNumber;
```

1.  Add the following conditional `if`…`else`-`if`…`else` statement to complete the program:

    ```cpp
      if (magicNumber == 1){
        std::cout << LamborghiniCar::output() << std::endl;
      } else if(magicNumber == 2){
        std::cout << PorscheCar::output() << std::endl;
      }else{
        std::cout << “Please type the correct magic number.” << std::endl;
      }
    }
    ```

### Activity 6: Writing a Math Library for use in a 3D Game

1.  Add the required header files at the start of the program (`mathlib.h` file is provided):

    ```cpp
    #include <mathlib.h>
    #include <array>
    #include <iostream>
    ```

2.  Create a global `const` variable of type `float` as shown here:

    ```cpp
    const float ENEMY_VIEW_RADIUS_METERS = 5;
    ```

3.  In the `main` function, create two arrays of type `float` and assign the following values:

    ```cpp
    int main() {
        std::array<float, 3> enemy1_location = {2, 2 ,0};
        std::array<float, 3> enemy2_location = {2, 4 ,0};
    ```

4.  Now, create a variable named `enemy_distance` of type `float` and use the distance function to assign the value after calculating it:

    ```cpp
        float enemy_distance = johnny::mathlib::distance(enemy1_location, enemy2_location);
        float distance_from_center = johnny::mathlib::distance(enemy1_location);
    ```

5.  Using the `circumference` function of `mathlib.h`, calculate and assign the enemy visual radius to `view_circumference_for_enemy` of type `float`:

    ```cpp
        using johnny::mathlib::circumference;
        float view_circumference_for_enemy = circumference(ENEMY_VIEW_RADIUS_METERS);
    ```

6.  Create a variable named `total_distance` of type `float` and assign the distance difference between the two enemies as shown in the following code:

    ```cpp
        float total_distance = johnny::mathlib::total_walking_distance({
            enemy1_location,
            {2, 3, 0}, // y += 1
            {2, 3, 3}, // z += 3
            {5, 3, 3}, // x += 3
            {8, 3, 3}, // x += 3
            {8, 3, 2}, // z -= 1
            {2, 3, 2}, // x -= 6
            {2, 3, 1}, // z -= 1
            {2, 3, 0}, // z -= 1
            enemy2_location
        });
    ```

7.  Print the output using the following print statement:

    ```cpp
        std::cout << “The two enemies are “ << enemy_distance << “m apart and can see for a circumference of “
                  << view_circumference_for_enemy << “m. To go to from one to the other they need to walk “
                  << total_distance << “m.”;
    }
    ```

## Lesson 3: Classes

### Activity 7: Information Hiding Through Getters and Setters

1.  Define a class named `Coordinates` with its members under a `private` access specifier:

    ```cpp
    class Coordinates {
      private:
        float latitude;
        float longitude;
    };
    ```

2.  Add the four operations as specified above and make them publicly accessible by preceding their declaration with the `public` access specifier. The setters (`set_latitude` and `set_longitude`) should take an `int` as a parameter and return `void`, while the getters do not take any parameter and return a `float`:

    ```cpp
    class Coordinates {
      private:
        float latitude;
        float longitude;
      public:
        void set_latitude(float value){}
        void set_longitude(float value){}
        float get_latitude(){}
        float get_longitude(){}
    };
    ```

3.  The four methods should now be implemented. The setters assign the given value to the corresponding members they are supposed to set; the getters return the values that are stored.

    ```cpp
    class Coordinates {
      private:
        float latitude;
        float longitude;
      public:
        void set_latitude(float value){ latitude = value; }
        void set_longitude(float value){ longitude = value; }
        float get_latitude(){ return latitude; }
        float get_longitude(){ return longitude; }
    };
    ```

    An example is as follows:

    ```cpp
    #include <iostream>
    int main() {
      Coordinates washington_dc;
      std::cout << “Object named washington_dc of type Coordinates created.” << std::endl;

      washington_dc.set_latitude(38.8951);
      washington_dc.set_longitude(-77.0364);
      std::cout << “Object’s latitude and longitude set.” << std::endl;

      std::cout << “Washington DC has a latitude of “ 
      << washington_dc.get_latitude() 
      << “ and longitude of “ << washington_dc.get_longitude() << std::endl;
    }
    ```

### Activity 8: Representing Positions in a 2D Map

1.  The first step is to create a class named `Coordinates` containing the coordinates as data members. These are two floating-point values, `_latitude` and `_longitude`, which identify the coordinates on a geographic coordinate system. Additionally, these data members are initialized with a `private` access specifier:

    ```cpp
    class Coordinates {
      private:
        float _latitude;
        float _longitude;
    };
    ```

2.  Then, the class is extended with a `public` constructor which takes two arguments used to initialize the data members of the class:

    ```cpp
    class Coordinates {
      public:
        Coordinates(float latitude, float longitude) 
        : _latitude(latitude), _longitude(longitude) {}
      private:
        int _latitude;
        int _longitude;
    };
    ```

3.  We can also add getters as seen previously to access the class members. An example is as follows:

    ```cpp
    #include <iostream>
    int main() {
      Coordinates washington_dc(38.8951, -77.0364);
      std::cout << “Object named washington_dc of type Coordinates created.” 
      << std::endl;

      std::cout << “Washington DC has a latitude of “ 
      << washington_dc.get_latitude() 
      << “ and longitude of “ << washington_dc.get_longitude() 
      << std::endl;
    }
    ```

### Activity 9: Storing Multiple Coordinates of Different Positions in the Map

1.  Using the RAII programming idiom, write a class that manages memory allocation and deletion of an array of `int`. The class has an array of integers as member data, which will be used to store the values.

    The constructor takes the size of the array as a parameter.

    The constructor also takes care of allocating memory, which is used to store the coordinates.

2.  Finally, define a destructor and make sure to free the previously allocated array in its implementation.
3.  We can add print statements to visualize what is happening:

    ```cpp
    class managed_array {
      public:
        explicit managed_array(size_t size) {
          array = new int[size];
          std::cout << “Array of size “ << size << “ created.” << std::endl;
        }
      ~managed_array() {
        delete[] array;
        std::cout << “Array deleted.” << std::endl;
      }
      private:
        int *array;
    };
    ```

4.  We can use our `managed_array` class as follows:

    ```cpp
    int main() {
        managed_array m(10);
    }
    ```

    The output will be as follows:

    ```cpp
    Array of size 10 created.
    Array deleted.
    ```

### Activity 10: The AppleTree Class, which Creates an Apple Instance

1.  First, we need to create a class with a `private` constructor. In this way, the object cannot be constructed, because the constructor is not publicly accessible:

    ```cpp
    class Apple
    {
      private:
        Apple() {}
        // do nothing
    };
    ```

2.  The `AppleTree` class is defined and contains a method called `createFruit` that is in charge of creating an `Apple` and returning it:

    ```cpp
    #include <iostream>
    class AppleTree
    {
      public:
        Apple createFruit(){
          Apple apple;
          std::cout << “apple created!” << std::endl;
          return apple;
        }
    };
    ```

3.  If we compile this code, we will get an error. At this point, the `Apple` constructor is `private`, so the `AppleTree` class cannot access it. We need to declare the `AppleTree` class as a `friend` of `Apple` to allow `AppleTree` to access the `private` methods of `Apple`:

    ```cpp
    class Apple
    {
      friend class AppleTree;
      private:
        Apple() {}
        // do nothing
    }
    ```

4.  The `Apple` object can now be constructed using the following code:

    ```cpp
    int main() {
      AppleTree tree;
      Apple apple = tree.createFruit();
    }
    ```

    This prints the following:

    ```cpp
    apple created!
    ```

### Activity 11: Ordering Point Objects

1.  We need to add an overload for the `<` operator to the `Point` class that we have previously defined. This takes another object of type `Point` as an argument and returns a Boolean indicating whether the object is less than the one provided as the parameter, using the previous definition for how to compare the two points:

    ```cpp
    class Point
    {
      public:
        bool operator< (const Point &other){
          return x < other.x || (x == other.x && y < other.y);
        }
      int x;
      int y;
    };
    ```

2.  At this point, we are able to compare the two `Point` objects:

    ```cpp
    #include <iostream>
    int main() {
      Point p_1, p_2;
      p_1.x = 1;
      p_1.y = 2;
      p_2.x = 2; 
      p_2.y = 1;
      std::cout << std::boolalpha << (p_1 < p_2) << std::endl;
    }
    ```

3.  Since in our example `p_1.x` is initialized to `1` and `p_2.x` to `2`, the result of the comparison will be `true`, which indicates that `p_1` comes earlier than `p_2` in the order.

### Activity 12: Implementing Functors

1.  Define a class constituted by a `private` data member of type `int` and add a constructor to initialize it:

    ```cpp
    class AddX {
      public:
        AddX(int x) : x(x) {}
      private:
        int x;
    };
    ```

2.  Extend it with the call operator `operator()` which takes an `int` as a parameter and returns an `int`. The implementation in the function body should return the addition of the previously defined `x` value and the parameter of the function named `y`:

    ```cpp
    class AddX {
      public:
        AddX(int x) : x(x) {}
        int operator() (int y) { return x + y; }
      private:
        int x;
    };
    ```

3.  Instantiate an object of the class just defined and invoke the call operator:

    ```cpp
    int main() {
      AddX add_five(5);
      std::cout << add_five(4) << std::endl;
    }
    ```

    The output will be as follows:

    ```cpp
    9
    ```

## Lesson 04: Generic Programming and Templates

### Activity 13: Read Objects from a Connection

1.  We start by including the headers of the files that provided the connection and the user account object:

    ```cpp
    #include <iostream>
    #include <connection.h>
    #include <useraccount.h>
    ```

2.  We can then start to write the `writeObjectToConnection` function. Declare a template which takes two `typename` parameters: an `Object` and a `Connection`. Call the `static` method `serialize()` on the object to get the `std::array` representing the object, then call `writeNext()` on the connection to write the data to it:

    ```cpp
    template<typename Object, typename Connection>
    void writeObjectToConnection(Connection& con, const Object& obj) {
        std::array<char, 100> data = Object::serialize(obj);
        con.writeNext(data);
    }
    ```

3.  We can then write `readObjectFromConnection`. Declare a template taking the same two parameters as before: an `Object` and a `Connection`. Inside, we call the connection `readNext()` to get the data stored inside the connection, then we call the `static` method on the object type `deserialize()` to get an instance of the object and return it:

    ```cpp
    template<typename Object, typename Connection>
    Object readObjectFromConnection(Connection& con) {
        std::array<char, 100> data = con.readNext();
        return Object::deserialize(data);
    }
    ```

4.  Finally, in the `main` function, we can call the functions we created to serialize objects. Both with `TcpConnection`:

    ```cpp
    std::cout << “serialize first user account” << std::endl;
    UserAccount firstAccount;
    TcpConnection tcpConnection;
    writeObjectToConnection(tcpConnection, firstAccount);
    UserAccount transmittedFirstAccount = readObjectFromConnection<UserAccount>(tcpConnection);
    ```

5.  And with `UdpConnection`:

    ```cpp
    std::cout << “serialize second user account” << std::endl;
    UserAccount secondAccount;
    UdpConnection udpConnection;
    writeObjectToConnection(udpConnection, secondAccount);
    UserAccount transmittedSecondAccount = readObjectFromConnection<UserAccount>(udpConnection);
    ```

    The output of the program is as follows:

    ```cpp
    serialize first user account
    the user account has been serialized
    the data has been written
    the data has been read
    the user account has been deserialized
    serialize second user account
    the user account has been serialized
    the data has been written
    the data has been read
    the user account has been deserialized
    ```

### Activity 14: UserAccount to Support Multiple Currencies

1.  We start by including the file defining the currencies:

    ```cpp
    #include <currency.h>
    #include <iostream>
    ```

2.  We then declare the template class `Account`. It should take a template parameter: `Currency`. We store the current balance of the account inside a data member of type `Currency`. We also provide a method in order to extract the current value of the balance:

    ```cpp
    template<typename Currency>
    class Account {
      public:
        Account(Currency amount) : balance(amount) {}
        Currency getBalance() const {
            return balance;
        }
      private:
        Currency balance;
    };
    ```

3.  Next, we create the method `addToBalance`. It should be a template with one type parameter, the other currency. The method takes a value of `OtherCurrency` and converts it to the value of the currency of the current account with the `to()` function, specifying to which currency the value should be converted to. It then adds it to the balance:

    ```cpp
    template<typename OtherCurrency>
    void addToBalance(OtherCurrency amount) {
        balance.d_value += to<Currency>(amount).d_value;
    }
    ```

4.  Finally, we can try to call our class in the `main` function with some data:

    ```cpp
    Account<GBP> gbpAccount(GBP(1000));
    // Add different currencies
    std::cout << “Balance: “ << gbpAccount.getBalance().d_value << “ (GBP)” << std::endl;
    gbpAccount.addToBalance(EUR(100));
    std::cout << “+100 (EUR)” << std::endl;
    std::cout << “Balance: “ << gbpAccount.getBalance().d_value << “ (GBP)” << std::endl;
    ```

    The output of the program is:

    ```cpp
    Balance: 1000 (GBP)
    +100 (EUR)
    Balance: 1089 (GBP)
    ```

### Activity 15: Write a Matrix Class for Mathematical Operations in a Game

1.  We start by defining a `Matrix` class which takes three template parameters: one type and the two dimensions of the `Matrix` class. The dimensions are of type `int`. Internally, we create a `std::array` with the size of the number of rows times the number of columns, in order to have enough space for all elements of the matrix. We add a constructor to initialize the array to *empty*, and a constructor to provide a list of values:

    ```cpp
    #include <array>
    template<typename T, int R, int C>
    class Matrix {
      // We store row_1, row_2, ..., row_C
      std::array<T, R*C> data;
      public:
        Matrix() : data({}) {}
        Matrix(std::array<T, R*C> initialValues) : data(initialValues) {}
    };
    ```

2.  We add a method `get()` to the class to return a reference to the element `T`. The method needs to take the row and column we want to access.
3.  We make sure that the requested indexes are inside the bounds of the matrix, otherwise we call `std::abort()`. In the array, we first store all the elements of the first row, then all the elements of the second row, and so on. When we want to access the elements of the *nth* row, we need to skip all the elements of the previous rows, which are going to be the number of elements per row (so the number of columns) times the previous rows, resulting in the following method:

    ```cpp
    T& get(int row, int col) {
      if (row >= R || col >= C) {
        std::abort();
      }
      return data[row*C + col];
    }
    ```

4.  For convenience, we define a function to print the class as well. We print all the elements in the columns separated by spaces, with one column per line:

    ```cpp
    template<typename T, size_t R, size_t C>
    std::ostream& operator<<(std::ostream& os, Matrix<T, R, C> matrix) {
        os << ‘\n’;
        for(int r=0; r < R; r++) {
            for(int c=0; c < C; c++) {
                os << matrix.get(r, c) << ‘ ‘;
            }
            os << “\n”;
        }
        return os;
    }
    ```

5.  In the `main` function, we can now use the functions we have defined:

    ```cpp
    Matrix<int, 3, 2> matrix({
      1, 2,
      3, 4,
      5, 6
    });
    std::cout << “Initial matrix:” << matrix << std::endl;
    matrix.get(1, 1) = 7;
    std::cout << “Modified matrix:” << matrix << std::endl;
    ```

    The output is as follows:

    ```cpp
    Initial matrix:
    1 2 
    3 4 
    5 6 
    Modified matrix:
    1 2 
    3 7 
    5 6
    ```

### **Solution bonus step**:

1.  We can add a new method, `multiply`, which takes a `std::array` of type `T` with the length of `C` by `const` reference, since we are not modifying it.

    The function returns an array of the same type, but length `R`.

2.  We follow the definition of matrix-vector multiplication to compute the result:

    ```cpp
    std::array<T, R> multiply(const std::array<T, C>& vector){
        std::array<T, R> result = {};
        for(size_t r = 0; r < R; r++) {
          for(size_t c = 0; c < C; c++) {
            result[r] += get(r, c) * vector[c];
          }
        }
        return result;
    }
    ```

3.  We can now extend our `main` function to call the `multiply` function:

    ```cpp
    std::array<int, 2> vector = {8, 9};
    std::array<int, 3> result = matrix.multiply(vector);
    std::cout << “Result of multiplication: [“ << result[0] << “, “
      << result[1] << “, “ << result[2] << “]” << std::endl;
    ```

    The output is as follows:

    ```cpp
    Result of multiplication: [26, 87, 94]
    ```

### Activity 16: Make the Matrix Class Easier to Use

1.  We start by importing `<functional>` in order to have access to `std::multiplies`:

    ```cpp
    #include <functional>
    ```

2.  We then change the order of the template parameters in the class `template`, so that the size parameters come first. We also add a new template parameter, `Multiply`, which is the type we will use for computing the multiplication between the elements in the `vector` by default, and we store an instance of it in the class:

    ```cpp
    template<int R, int C, typename T = int, typename Multiply=std::multiplies<T> >
    class Matrix {
      std::array<T, R*C> data;
      Multiply multiplier;
      public:
        Matrix() : data({}), multiplier() {}
        Matrix(std::array<T, R*C> initialValues) : data(initialValues), multiplier() {}
    };
    ```

    The `get()` function remains the same as the previous activity.

3.  We now need to make sure that the `Multiply` method uses the `Multiply` type provided by the user to perform the multiplication.
4.  To do so, we need to make sure to call `multiplier(operand1, operand2)` instead of `operand1 * operand2`, so that we use the instance we stored inside the class:

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

5.  We can now add an example of how we can use the class:

    ```cpp
    // Create a matrix of int, with the ‘plus’ operation by default
    Matrix<3, 2, int, std::plus<int>> matrixAdd({
        1, 2,
        3, 4,
        5, 6
    });
    std::array<int, 2> vector = {8, 9};
    // This will call std::plus when doing the multiplication
    std::array<int, 3> result = matrixAdd.multiply(vector);
    std::cout << “Result of multiplication(with +): [“ << result[0] << “, “
              << result[1] << “, “ << result[2] << “]” << std::endl;
    ```

    The output is as follows:

    ```cpp
    Result of multiplication(with +): [20, 24, 28]
    ```

### Activity 17: Ensure Users are Logged in When Performing Actions on the Account

1.  We first declare a template function which takes two type parameters: an `Action` and a `Parameter` type.
2.  The function should take the user identification, the action and the parameter. The parameter should be accepted as a forwarding reference. As a first step, it should check if the user is logged in, by calling the `isLoggenIn()` function. If the user is logged in, it should call the `getUserCart()` function, then call the action passing the cart and forwarding the parameter:

    ```cpp
    template<typename Action, typename Parameter>
    void execute_on_user_cart(UserIdentifier user, Action action, Parameter&& parameter) {
        if(isLoggedIn(user)) {
            Cart cart = getUserCart(user);
            action(cart, std::forward<Parameter>(parameter));
        } else {
            std::cout << “The user is not logged in” << std::endl;
        }
    }
    ```

3.  We can test how `execute_on_user_cart` works by calling it in the `main` function:

    ```cpp
    Item toothbrush{1023};
    Item toothpaste{1024};
    UserIdentifier loggedInUser{0};
    std::cout << “Adding items if the user is logged in” << std::endl;
    execute_on_user_cart(loggedInUser, addItems, std::vector<Item>({toothbrush, toothpaste}));
    UserIdentifier loggedOutUser{1};
    std::cout << “Removing item if the user is logged in” << std::endl;
    execute_on_user_cart(loggedOutUser, removeItem, toothbrush);
    ```

    The output is as follows:

    ```cpp
    Adding items if the user is logged in
    Items added
    Removing item if the user is logged in
    The user is not logged in
    ```

### Activity 18: Safely Perform Operations on User Cart with an Arbitrary Number of Parameters

1.  We need to expand the previous activity to accept any number of parameters with any kind of ref-ness and pass it to the action provided. To do so, we need to create a `variadic` template.
2.  Declare a `template` function that takes an action and a `variadic` number of parameters as template parameters. The function parameters should be the user action, the action to perform, and the expanded template parameter `pack`, making sure that the parameters are accepted as forwarding references.
3.  Inside the function, we perform the same checks as before, but now we expand the parameters when we forward them to the action:

    ```cpp
    template<typename Action, typename... Parameters>
    void execute_on_user_cart(UserIdentifier user, Action action, Parameters&&... parameters) {
        if(isLoggedIn(user)) {
            Cart cart = getUserCart(user);
            action(cart, std::forward<Parameters>(parameters)...);
        } else {
            std::cout << “The user is not logged in” << std::endl;
        }
    }
    ```

4.  Let’s test the new function in our `main` function:

    ```cpp
    Item toothbrush{1023};
    Item apples{1024};
    UserIdentifier loggedInUser{0};
    std::cout << “Replace items if the user is logged in” << std::endl;
    execute_on_user_cart(loggedInUser, replaceItem, toothbrush, apples);
    UserIdentifier loggedOutUser{1};
    std::cout << “Replace item if the user is logged in” << std::endl;
    execute_on_user_cart(loggedOutUser, removeItem, toothbrush);
    ```

    The output is as follows:

    ```cpp
    Replace items if the user is logged in
    Replacing item
    Item removed
    Items added
    Replace item if the user is logged in
    The user is not logged in
    ```

## Lesson 5: Standard Library Containers and Algorithms

### Activity 19: Storing User Accounts

1.  First, we include the header files for the `array` class and input/output operations with the required namespace:

    ```cpp
    #include <array>
    ```

2.  An array of ten elements of type `int` is declared:

    ```cpp
    array<int,10> balances;
    ```

3.  Initially, the values of the elements are undefined since it is an array of the fundamental data type `int`. The array is initialized using a `for` loop, where each element is initialized with its index. The operator `size()` is used to evaluate the size of the array and the subscript operator `[ ]` is used to access every position of the array:

    ```cpp
    for (int i=0; i < balances.size(); ++i) 
    {
      balances[i] = 0;
    }
    ```

4.  We now want to update the value for the first and last user. We can use `front()` and `back()` to access the accounts of these users:

    ```cpp
    balances.front() += 100;
    balances.back() += 100;
    ```

    We would like to store the account balance of an arbitrary number of users. We then want to add 100 users to the account list, with a balance of 500.

5.  We can use `vector` to store an arbitrary number of users. It is defined in the `<vector>` header:

    ```cpp
    #include <vector>
    ```

6.  Then, we declare a vector of type `int`. Optionally, we reserve enough memory to store the 100 users’ account by calling `reserve(100)` to avoid memory reallocation:

    ```cpp
    std::vector<int> balances;
    balances.reserve(100);
    ```

7.  Finally, we modify the `for` loop to add the balance for the users at the end of the accounts vector:

    ```cpp
    for (int i=0; i<100; ++i) 
    {
      balances.push_back(500);
    }
    ```

### Activity 20: Retrieving a User’s Balance from their Given Username

1.  Include the header file for the `map` class and the header for `string`:

    ```cpp
    #include <map>
    #include <string>
    ```

2.  Create a map with the key being `std::string` and the value `int`:

    ```cpp
    std::map<std::string, int> balances;
    ```

3.  Insert the balances of the users inside `map` by using `insert` and `std::make_pair`. The first argument is the key, the second one is the value:

    ```cpp
    balances.insert(std::make_pair(“Alice”,50));
    balances.insert(std::make_pair(“Bob”, 50));
    balances.insert(std::make_pair(“Charlie”, 50));
    ```

4.  Use the `find` function providing the name of the user to find the position of the account in the map. Compare it with `end()` to check whether a position was found:

    ```cpp
    auto donaldAccountPos = balances.find(“Donald”);
    bool hasAccount = (donaldAccountPos !=  balances.end());
    std::cout << “Donald has an account: “ << hasAccount << std::endl;
    ```

5.  Now, look for the account of `Alice`. We know `Alice` has an account, so there is no need to check whether we found a valid position. We can print the value of the account using `->second`:

    ```cpp
    auto alicePosition = balances.find(“Alice”);
    std::cout << “Alice balance is: “ << alicePosition->second << std::endl;
    ```

### Activity 21: Processing User Registration in Order

1.  First, we include the header file for the `stack` class:

    ```cpp
    #include <stack>
    ```

2.  Create a `stack` providing the type to `store`:

    ```cpp
    std::stack<RegistrationForm> registrationForms;
    ```

3.  We start by storing the form inside the `stack` when the user registers. In the body of the `storeRegistrationForm` function, push the element into the queue:

    ```cpp
    stack.push(form);
    std::cout << “Pushed form for user “ << form.userName << std::endl;
    ```

4.  Now, inside `endOfDayRegistrationProcessing`, we get all the elements inside the `stack` and then process them. Use the `top()` method to access the top element in the `stack` and `pop()` to remove the top element. We stop getting and removing the first element when no element is left:

    ```cpp
    while(not stack.empty()) {
      processRegistration(stack.top());
      stack.pop();
    }
    ```

5.  Finally, we call our functions with some test data:

    ```cpp
    int main(){
      std::stack<RegistrationForm> registrationForms;
      storeRegistrationForm(registrationForms, RegistrationForm{“Alice”});
      storeRegistrationForm(registrationForms, RegistrationForm{“Bob”});
      storeRegistrationForm(registrationForms, RegistrationForm{“Charlie”});
      endOfDayRegistrationProcessing(registrationForms);
    }
    ```

### Activity 22: Airport System Management

1.  We start by creating the class for `Airplane`. Make sure to first include the header for `variant`:

    ```cpp
    #include <variant>
    ```

2.  Then, create the class with a constructor that sets the current state of the airplane to `AtGate`:

    ```cpp
    class Airplane {
      std::variant<AtGate, Taxi, Flying> state;
      public:
        Airplane(int gate) : state(AtGate{gate}) {
          std::cout << “At gate “ << gate << std::endl;
        }
    };
    ```

3.  Now, implement the `startTaxi()` method. First, check the current state of the airplane with `std::holds_alternative<>()`, and if the airplane is not in the correct state, write an error message and return.
4.  If the airplane is in the correct state, change the state to taxi by assigning it to the `variant`:

    ```cpp
    void startTaxi(int lane, int numPassengers) {
        if (not std::holds_alternative<AtGate>(state)) {
            std::cout << “Not at gate: the plane cannot start taxi to lane “ << lane << std::endl;
            return;
        }
        std::cout << “Taxing to lane “ << lane << std::endl;
        state = Taxi{lane, numPassengers};   
    }
    ```

5.  We repeat the same process for the `takeOff()` method:

    ```cpp
    void takeOff(float speed) {
        if (not std::holds_alternative<Taxi>(state)) {
            std::cout << “Not at lane: the plane cannot take off with speed “ << speed << std::endl;
            return;
        }
        std::cout << “Taking off at speed “ << speed << std::endl;
        state = Flying{speed}; 
    }
    ```

6.  We can now start looking at the `currentStatus()` method. Since we want to perform an operation for each of the states in the `variant`, we can use a visitor.
7.  Outside the `Airplane` class, create a class that has a method `operator()` for each of the types in the airplane state. Inside the method, print the information of the state. Remember to make the methods public:

    ```cpp
    class AirplaneStateVisitor {
      public:
        void operator()(const AtGate& atGate) {
           std::cout << “AtGate: “ << atGate.gate << std::endl;
        }
      void operator()(const Taxi& taxi) {
        std::cout << “Taxi: lane “ << taxi.lane << “ with “ << taxi.numPassengers << “ passengers” << std::endl;
      }
      void operator()(const Flying& flying) {
        std::cout << “Flaying: speed “ << flying.speed << std::endl;
      }
    };
    ```

8.  Now, create the `currentStatus()` method and call the visitor on the state using `std::visit`:

    ```cpp
    void currentStatus() {
        AirplaneStateVisitor visitor;
        std::visit(visitor, state);
    }
    ```

9.  We can now try to call the functions of `Airplane` from the `main` function:

    ```cpp
    int main()
    {
        Airplane airplane(52);
        airplane.currentStatus();
        airplane.startTaxi(12, 250);
        airplane.currentStatus();
        airplane.startTaxi(13, 250);
        airplane.currentStatus();
        airplane.takeOff(800);
        airplane.currentStatus();
        airplane.takeOff(900);
    }
    ```

## Lesson 6: Object-Oriented Programming

### Activity 23: Creating Game Characters

1.  Create a `Character` class that has a `public` method `moveTo` that prints `Moved to position`:

    ```cpp
    class Character {
      public:
        void moveTo(Position newPosition) {
          position = newPosition;
          std::cout << “Moved to position “ << newPosition.positionIdentifier << std::endl;
        }
      private:
        Position position;
    };
    ```

2.  Create a `struct` named `Position`:

    ```cpp
    struct Position {
      // Fields to describe the position go here
      std::string positionIdentifier;
    };
    ```

3.  Create two classes `Hero` and `Enemy` that are derived from the class `Character`:

    ```cpp
    // Hero inherits publicly from Character: it has
    // all the public member of the Character class.
    class Hero : public Character {
    };
    // Enemy inherits publicly from Character, like Hero
    class Enemy : public Character {
    };
    ```

4.  Create a class `Spell` with the constructor that prints the name of the person casting the spell:

    ```cpp
    class Spell {
    public:
        Spell(std::string name) : d_name(name) {}
        std::string name() const {
            return d_name;
        }
    private:
        std::string d_name;
    };
    ```

5.  The class `Hero` should have a public method to cast a spell. Use the value from the `Spell` class:

    ```cpp
    public:
        void cast(Spell spell) {
            // Cast the spell
            std::cout << “Casting spell “ << spell.name() << std::endl;
        }
    ```

6.  The class `Enemy` should have a public method to swing a sword which prints `Swinging sword`:

    ```cpp
    public:
        void swingSword() {
            // Swing the sword
            std::cout << “Swinging sword” << std::endl;
        }
    ```

7.  Implement the `main` method that calls these methods in various classes:

    ```cpp
    int main()
    {
        Position position{“Enemy castle”};
        Hero hero;
        Enemy enemy;
        // We call moveTo on Hero, which calls the method inherited
        // from the Character class
        hero.moveTo(position);
        enemy.moveTo(position);
        // We can still use the Hero and Enemy methods
        hero.cast(Spell(“fireball”));
        enemy.swingSword();
    }
    ```

### Activity 24: Calculating Employee Salaries

1.  We can create a class `Employee` with two virtual methods, `getBaseSalary` and `getBonus`, since we want to change those methods based on the type of employee:

    ```cpp
    class Employee {
      public:
        virtual int getBaseSalary() const { return 100; }
        virtual int getBonus(const Deparment& dep) const {
          if (dep.hasReachedTarget()) {
          }
          return 0;
        }
    ```

2.  We also define a method, `getTotalComp`, which does not need to be virtual, but will call the two virtual methods:

    ```cpp
        int getTotalComp(const Deparment& dep) {

        }
    };
    ```

3.  We then derive a `Manager` class from it, overriding the method for computing the bonus. We might also want to override `getBaseSalary` if we want to give a different base salary to managers:

    ```cpp
    class Manager : public Employee {
      public:
        virtual int getBaseSalary() const override { return 150; }
        virtual int getBonus(const Deparment& dep) const override {
          if (dep.hasReachedTarget()) {
            int additionalDeparmentEarnings = dep.effectiveEarning() - dep.espectedEarning();
            return 0.2 * getBaseSalary() + 0.01 * additionalDeparmentEarnings;
          }
          return 0;
        }
    };
    ```

4.  Create a class `Department` as shown:

    ```cpp
    class Department {
      public:
        bool hasReachedTarget() const {return true;}
        int espectedEarning() const {return 1000;}
        int effectiveEarning() const {return 1100;}
    };
    ```

5.  Now, in the `main` function, call the `Department`, `Employee`, and `Manager` classes as shown:

    ```cpp
    int main()
    {
      Department dep;
      Employee employee;
      Manager manager;
      std::cout << “Employee: “ << employee.getTotalComp(dep) << “. Manager: “ << manager.getTotalComp(dep) << std::endl;
    }
    ```

### Activity 25: Retrieving User Information

1.  We have to write the code that can be independent of where the data is coming from. So, we create an interface `UserProfileStorage` for retrieving the `CustomerProfile` from a `UserId`:

    ```cpp
    struct UserProfile {};
    struct UserId {};
    class UserProfileStorage {
      public:
        virtual UserProfile getUserProfile(const UserId& id) const = 0;

        virtual ~UserProfileStorage() = default;
      protected:
        UserProfileStorage() = default;
        UserProfileStorage(const UserProfileStorage&) = default;
        UserProfileStorage& operator=(const UserProfileStorage&) = default;
    };
    ```

2.  Now, write the `UserProfileCache` class that inherits from `UserProfileStorage`:

    ```cpp
    class UserProfileCache : public UserProfileStorage {
      public:
        UserProfile getUserProfile(const UserId& id) const override { 
        std::cout << “Getting the user profile from the cache” << std::endl;
        return UserProfile(); }
    };
    void exampleOfUsage(const UserProfileStorage& storage) {
        UserId user;
        std::cout << “About to retrieve the user profile from the storage” <<std::endl;
        UserProfile userProfile = storage.getUserProfile(user);
    }
    ```

3.  In the `main` function, call the `UserProfileCache` class and `exampleOfUsage` function as shown:

    ```cpp
    int main()
    {
      UserProfileCache cache;
      exampleOfUsage (cache);
    }
    ```

### Activity 26: Creating a Factory for UserProfileStorage

1.  Write the following code that needs the `UserProfileStorage` class, as shown. To allow that, we provide a factory class, which has a method `create` that provides an instance of `UserProfileStorage`. Write this class making sure that the user does not have to manage the memory for the interface manually:

    ```cpp
    #include <iostream>
    #include <memory>
    #include <userprofile_activity18.h>
    class UserProfileStorageFactory {
    public:
        std::unique_ptr<UserProfileStorage> create() const {
            return std::make_unique<UserProfileCache>();
        }
    }; 
    ```

2.  We want the `UserProfileStorageFactory` class to return a `unique_ptr` so that it manages the lifetime of the interface:

    ```cpp
    void getUserProfile(const UserProfileStorageFactory& storageFactory) {
      std::unique_ptr<UserProfileStorage> storage = storageFactory.create();
      UserId user;
      storage->getUserProfile(user);
      // The storage is automatically destroyed
    }
    ```

3.  Now, in the `main` function, call the `UserProfileStorageFactory` class as shown:

    ```cpp
    int main()
    {
      UserProfileStorageFactory factory;
      getUserProfile(factory);
    ```

### Activity 27: Using a Database Connection for Multiple Operations

1.  First, create a `DatabaseConnection` class that can be used in parallel. We want to reuse it as much as possible, and we know we can use `std::async` to start a new parallel task:

    ```cpp
    #include <future>
    struct DatabaseConnection {};
    ```

2.  Assuming there are two functions `updateOrderList(DatabaseConnection&)` and `scheduleOrderProcessing(DatabaseConnection&)`, write a function that creates a `DatabaseConnection` and gives it to the two parallel tasks. (Note that we don’t know which task finishes first):

    ```cpp
    void updateOrderList(DatabaseConnection&) {}
    void scheduleOrderProcessing(DatabaseConnection&) {}
    ```

3.  You must understand when and how to create a `shared_ptr`. You can also use the following code to write the `shared_ptr` correctly.

    ```cpp
    /* We need to get a copy of the shared_ptr so it stays alive until this function finishes */
    void updateWithConnection(std::shared_ptr<DatabaseConnection> connection) {
        updateOrderList(*connection);
    }
    ```

    There are several users of the connection, and we do not know which one is the owner, since the connection needs to stay alive as long as anyone is using it.

4.  To model this, we use a `shared_ptr`. Remember that we need a copy of the `shared_ptr` to exist in order for the connection to remain valid:

    ```cpp
    /* We need to get a copy of the shared_ptr so it stays alive until this function finishes. */
    void scheduleWithConnection(std::shared_ptr<DatabaseConnection> connection) {
        scheduleOrderProcessing(*connection);
    }
    ```

5.  Create the `main` function as follows:

    ```cpp
    int main()
    {
        std::shared_ptr<DatabaseConnection> connection = std::make_shared<DatabaseConnection>();
        std::async(std::launch::async, updateWithConnection, connection);
        std::async(std::launch::async, scheduleWithConnection, connection);
    }
    ```