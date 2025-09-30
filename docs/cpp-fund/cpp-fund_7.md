# 第八章：*附录*

## 关于

本节包含帮助学生执行书中活动的概念。它包括学生为实现活动目标必须执行的详细步骤。

## 第 1 课：入门

### 活动一：使用 `while` 循环在 1 到 100 之间找到 7 的因子

1.  在 `main` 函数之前导入所有必需的头文件：

    ```cpp
    #include <iostream>
    ```

1.  在 `main` 函数内部，创建一个类型为 `unsigned` 的变量 `i`，并将其值初始化为 `1`：

    ```cpp
    unsigned i = 1;
    ```

1.  现在，使用 `while` 循环添加逻辑，其中 `i` 的值应小于 `100`：

    ```cpp
    while ( i < 100){ }
    ```

1.  在 `while` 循环的作用域内，使用以下逻辑的 `if` 语句：

    ```cpp
    if (i%7 == 0) {
       std::cout << i << std::endl;
    }
    ```

1.  将 `i` 变量的值增加以迭代 `while` 循环以验证条件：

    ```cpp
    i++;
    ```

    程序的输出如下：

    ```cpp
    7
    14
    21
    28
    ...
    98
    ```

### 活动二：定义一个二维数组并初始化其元素

1.  在创建 C++ 文件后，在程序开始处包含以下头文件：

    ```cpp
    #include <iostream>
    ```

1.  现在，在 `main` 函数中，创建一个名为 `foo` 的双向数组，类型为整数，具有三行三列，如下所示：

    ```cpp
    int main()
    {
      int foo[3][3];
    ```

1.  现在，我们将使用嵌套 `for` 循环的概念来迭代 `foo` 数组的每个索引条目：

    ```cpp
    for (int x= 0; x < 3; x++){
      for (int y = 0; y < 3; y++){
      }
    }
    ```

1.  在第二个 `for` 循环中，添加以下语句：

    ```cpp
    foo[x][y] = x + y;
    ```

1.  最后，再次迭代数组以打印其值：

    ```cpp
    for (int x = 0; x < 3; x++){
       for (int y = 0; y < 3; y++){
          std::cout << “foo[“ << x << “][“ << y << “]: “ << foo[x][y] << std::endl;
       }
    }
    ```

    输出如下：

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

## 第 2 课：函数

### 活动三：计算一个人是否有资格投票或不

1.  在程序中包含头文件以打印如下所示的输出：

    ```cpp
    #include <iostream>
    ```

1.  现在，创建一个名为 `byreference_age_in_5_years` 的函数，并使用以下条件编写 `if` 循环以打印消息：

    ```cpp
    void byreference_age_in_5_years(int& age) {
      if (age >= 18) {
        std::cout << “Congratulations! You are eligible to vote for your nation.” << std::endl;
        return;
    ```

1.  添加 `else` 块以提供另一个条件，如果用户的年龄小于 18 岁：

    ```cpp
      } else{
        int reqAge = 18;
        int yearsToGo = reqAge-age;
        std::cout << “No worries, just “<< yearsToGo << “ more years to go.” << std::endl;
      }
    }
    ```

1.  在 `main` 函数中，创建一个类型为整数的变量，并将其作为引用传递给 `byreference_age_in_5_years` 函数，如下所示：

    ```cpp
    int main() {
        int age;
        std::cout << “Please enter your age:”;
        std::cin >> age;
        byreference_age_in_5_years(age);
    }
    ```

### 活动四：在函数中应用通过引用或值传递的理解

1.  在添加所有必需的头文件后，创建第一个类型为整数的函数，如下所示：

    ```cpp
    int sum(int a, int b)
    {
      return a + b
    }
    ```

    采用值传递，返回值传递，因为类型在内存中较小，没有使用引用的理由。

1.  第二个函数应编写如下：

    ```cpp
    int& getMaxOf(std::array<int, 10>& array1, std::array<int, 10>& array2, int index) {
      if (array1[index] >= array2[index]) {
        return array1[index];
      } else {
        return array2[index];
      }
    }
    ```

### 活动五：在命名空间中组织函数

1.  包含所需的头文件和命名空间以打印所需的输出：

    ```cpp
    #include <iostream>
    using namespace std;
    ```

1.  现在，创建一个名为 `LamborghiniCar` 的命名空间，并使用以下 `output` 函数：

    ```cpp
    namespace LamborghiniCar
    {
      int output(){
        std::cout << “Congratulations! You deserve the Lamborghini.” << std::endl;
        return NULL;
      }
    }
    ```

1.  创建另一个名为 `PorscheCar` 的命名空间，并添加一个 `output` 函数，如下所示：

    ```cpp
    namespace PorscheCar
    {
      int output(){
        std::cout << “Congratulations! You deserve the Porsche.” << std::endl; 
        return NULL;
      }
    }
    ```

在 `main` 函数中，创建一个名为 `magicNumber` 的类型为整数的变量以接受用户的输入：

```cpp
int main()
{
  int magicNumber;
  std::cout << “Select a magic number (1 or 2) to win your dream car: “;
  std::cin >> magicNumber;
```

1.  添加以下条件 `if`…`else`-`if`…`else` 语句以完成程序：

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

### 活动六：编写用于 3D 游戏的数学库

1.  在程序开始处添加所需的头文件（提供 `mathlib.h` 文件）：

    ```cpp
    #include <mathlib.h>
    #include <array>
    #include <iostream>
    ```

1.  创建一个全局 `const` 变量，类型为 `float`，如下所示：

    ```cpp
    const float ENEMY_VIEW_RADIUS_METERS = 5;
    ```

1.  在 `main` 函数中，创建两个类型为 `float` 的数组，并分配以下值：

    ```cpp
    int main() {
        std::array<float, 3> enemy1_location = {2, 2 ,0};
        std::array<float, 3> enemy2_location = {2, 4 ,0};
    ```

1.  现在，创建一个名为 `enemy_distance` 的 `float` 类型的变量，并使用距离函数在计算后赋值：

    ```cpp
        float enemy_distance = johnny::mathlib::distance(enemy1_location, enemy2_location);
        float distance_from_center = johnny::mathlib::distance(enemy1_location);
    ```

1.  使用 `mathlib.h` 中的 `circumference` 函数，计算并分配敌人可视半径到 `view_circumference_for_enemy` 的 `float` 类型：

    ```cpp
        using johnny::mathlib::circumference;
        float view_circumference_for_enemy = circumference(ENEMY_VIEW_RADIUS_METERS);
    ```

1.  创建一个名为 `total_distance` 的 `float` 类型的变量，并将两个敌人之间的距离差赋值，如下代码所示：

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

1.  使用以下打印语句打印输出：

    ```cpp
        std::cout << “The two enemies are “ << enemy_distance << “m apart and can see for a circumference of “
                  << view_circumference_for_enemy << “m. To go to from one to the other they need to walk “
                  << total_distance << “m.”;
    }
    ```

## 第 3 课：类

### 活动 7：通过获取器和设置器实现信息隐藏

1.  定义一个名为 `Coordinates` 的类，其成员在 `private` 访问修饰符下：

    ```cpp
    class Coordinates {
      private:
        float latitude;
        float longitude;
    };
    ```

1.  添加上述指定的四个操作，并通过在它们的声明前加上 `public` 访问修饰符使它们公开可访问。设置器（`set_latitude` 和 `set_longitude`）应接受一个 `int` 参数并返回 `void`，而获取器不接收任何参数并返回一个 `float`：

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

1.  现在应该实现四个方法。设置器将给定的值赋给它们应该设置的相应成员；获取器返回存储的值。

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

    以下是一个示例：

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

### 活动 8：在 2D 地图上表示位置

1.  第一步是创建一个名为 `Coordinates` 的类，其中包含坐标作为数据成员。这些是两个浮点值，`_latitude` 和 `_longitude`，它们标识地理坐标系统上的坐标。此外，这些数据成员使用 `private` 访问修饰符初始化：

    ```cpp
    class Coordinates {
      private:
        float _latitude;
        float _longitude;
    };
    ```

1.  然后，通过一个接受两个参数的 `public` 构造函数扩展该类，这两个参数用于初始化类的数据成员：

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

1.  我们还可以添加之前看到的获取器来访问类成员。以下是一个示例：

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

### 活动 9：在地图中存储不同位置的多组坐标

1.  使用 RAII 编程习惯，编写一个管理数组内存分配和删除的类。该类有一个整数数组作为成员数据，将用于存储值。

    构造函数接受数组的大小作为参数。

    构造函数还负责分配内存，用于存储坐标。

1.  最后，定义一个析构函数，并确保在其实现中释放之前分配的数组。

1.  我们可以添加打印语句来可视化正在发生的事情：

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

1.  我们可以使用我们的 `managed_array` 类如下：

    ```cpp
    int main() {
        managed_array m(10);
    }
    ```

    输出结果如下：

    ```cpp
    Array of size 10 created.
    Array deleted.
    ```

### 活动 10：创建苹果实例的 AppleTree 类

1.  首先，我们需要创建一个具有 `private` 构造函数的类。这样，对象就不能被构造，因为构造函数不是公开可访问的：

    ```cpp
    class Apple
    {
      private:
        Apple() {}
        // do nothing
    };
    ```

1.  `AppleTree` 类被定义，并包含一个名为 `createFruit` 的方法，该方法负责创建一个 `Apple` 并返回它：

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

1.  如果我们编译此代码，我们将得到一个错误。在此点，`Apple` 构造函数是 `private` 的，因此 `AppleTree` 类无法访问它。我们需要将 `AppleTree` 类声明为 `Apple` 的 `friend`，以便允许 `AppleTree` 访问 `Apple` 的 `private` 方法：

    ```cpp
    class Apple
    {
      friend class AppleTree;
      private:
        Apple() {}
        // do nothing
    }
    ```

1.  现在可以使用以下代码构造 `Apple` 对象：

    ```cpp
    int main() {
      AppleTree tree;
      Apple apple = tree.createFruit();
    }
    ```

    这将打印以下内容：

    ```cpp
    apple created!
    ```

### 活动 11：对点对象进行排序

1.  我们需要为之前定义的 `Point` 类添加一个 `<` 操作符的重载。这个重载接受另一个类型为 `Point` 的对象作为参数，并返回一个布尔值，指示该对象是否小于作为参数提供的对象，使用之前定义的比较两个点的方法：

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

1.  到目前为止，我们能够比较两个 `Point` 对象：

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

1.  由于在我们的示例中 `p_1.x` 被初始化为 `1`，而 `p_2.x` 被初始化为 `2`，比较的结果将是 `true`，这表明在顺序中 `p_1` 比 `p_2` 更早。

### 活动 12：实现仿函数

1.  定义一个由类型为 `int` 的 `private` 数据成员构成的类，并添加一个构造函数来初始化它：

    ```cpp
    class AddX {
      public:
        AddX(int x) : x(x) {}
      private:
        int x;
    };
    ```

1.  通过调用操作符 `operator()` 扩展它，它接受一个 `int` 作为参数并返回一个 `int`。在函数体内的实现应该返回先前定义的 `x` 值与函数参数 `y` 的和：

    ```cpp
    class AddX {
      public:
        AddX(int x) : x(x) {}
        int operator() (int y) { return x + y; }
      private:
        int x;
    };
    ```

1.  实例化一个刚刚定义的类的对象并调用调用操作符：

    ```cpp
    int main() {
      AddX add_five(5);
      std::cout << add_five(4) << std::endl;
    }
    ```

    输出将如下所示：

    ```cpp
    9
    ```

## 第 04 课：泛型编程和模板

### 活动 13：从连接中读取对象

1.  我们首先包含提供连接和用户账户对象的文件头：

    ```cpp
    #include <iostream>
    #include <connection.h>
    #include <useraccount.h>
    ```

1.  然后，我们可以开始编写 `writeObjectToConnection` 函数。声明一个模板，它接受两个 `typename` 参数：一个 `Object` 和一个 `Connection`。在对象上调用 `static` 方法 `serialize()` 以获取表示对象的 `std::array`，然后调用连接上的 `writeNext()` 将数据写入它：

    ```cpp
    template<typename Object, typename Connection>
    void writeObjectToConnection(Connection& con, const Object& obj) {
        std::array<char, 100> data = Object::serialize(obj);
        con.writeNext(data);
    }
    ```

1.  然后，我们可以编写 `readObjectFromConnection`。声明一个模板，它接受与之前相同的两个参数：一个 `Object` 和一个 `Connection`。在内部，我们调用连接的 `readNext()` 来获取存储在连接中的数据，然后我们调用对象类型的 `static` 方法 `deserialize()` 来获取对象的实例并返回它：

    ```cpp
    template<typename Object, typename Connection>
    Object readObjectFromConnection(Connection& con) {
        std::array<char, 100> data = con.readNext();
        return Object::deserialize(data);
    }
    ```

1.  最后，在 `main` 函数中，我们可以调用我们创建的序列化对象函数。无论是使用 `TcpConnection`：

    ```cpp
    std::cout << “serialize first user account” << std::endl;
    UserAccount firstAccount;
    TcpConnection tcpConnection;
    writeObjectToConnection(tcpConnection, firstAccount);
    UserAccount transmittedFirstAccount = readObjectFromConnection<UserAccount>(tcpConnection);
    ```

1.  还是使用 `UdpConnection`：

    ```cpp
    std::cout << “serialize second user account” << std::endl;
    UserAccount secondAccount;
    UdpConnection udpConnection;
    writeObjectToConnection(udpConnection, secondAccount);
    UserAccount transmittedSecondAccount = readObjectFromConnection<UserAccount>(udpConnection);
    ```

    程序的输出如下：

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

### 活动 14：支持多种货币的用户账户

1.  我们首先包含定义货币的文件：

    ```cpp
    #include <currency.h>
    #include <iostream>
    ```

1.  我们随后声明了一个模板类 `Account`。它应该接受一个模板参数：`Currency`。我们将账户的当前余额存储在类型为 `Currency` 的数据成员中。我们还提供了一个方法来提取当前余额的值：

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

1.  接下来，我们创建一个名为 `addToBalance` 的方法。它应该是一个带有单个类型参数的模板，即其他货币。该方法接受一个 `OtherCurrency` 类型的值，并使用 `to()` 函数将其转换为当前账户货币的值，指定要将值转换为哪种货币。然后将其添加到余额中：

    ```cpp
    template<typename OtherCurrency>
    void addToBalance(OtherCurrency amount) {
        balance.d_value += to<Currency>(amount).d_value;
    }
    ```

1.  最后，我们可以在 `main` 函数中使用一些数据来尝试调用我们的类：

    ```cpp
    Account<GBP> gbpAccount(GBP(1000));
    // Add different currencies
    std::cout << “Balance: “ << gbpAccount.getBalance().d_value << “ (GBP)” << std::endl;
    gbpAccount.addToBalance(EUR(100));
    std::cout << “+100 (EUR)” << std::endl;
    std::cout << “Balance: “ << gbpAccount.getBalance().d_value << “ (GBP)” << std::endl;
    ```

    程序的输出如下：

    ```cpp
    Balance: 1000 (GBP)
    +100 (EUR)
    Balance: 1089 (GBP)
    ```

### 活动 15：为游戏中的数学运算编写一个矩阵类

1.  我们首先定义一个 `Matrix` 类，它接受三个模板参数：一个类型和 `Matrix` 类的两个维度。维度是 `int` 类型。内部，我们创建一个大小为行数乘以列数的 `std::array`，以便为矩阵的所有元素提供足够的空间。我们添加了一个构造函数来初始化数组为 *空*，以及一个构造函数来提供值列表：

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

1.  我们在类中添加了一个 `get()` 方法来返回对元素 `T` 的引用。该方法需要接受我们想要访问的行和列。

1.  我们确保请求的索引在矩阵的范围内，否则我们调用 `std::abort()`。在数组中，我们首先存储第一行的所有元素，然后存储第二行的所有元素，依此类推。当我们想要访问第 *n* 行的元素时，我们需要跳过之前行的所有元素，这些元素是每行的元素数量（即列数）乘以之前的行数，结果如下所示的方法：

    ```cpp
    T& get(int row, int col) {
      if (row >= R || col >= C) {
        std::abort();
      }
      return data[row*C + col];
    }
    ```

1.  为了方便起见，我们定义了一个打印类的函数。我们按列分隔所有元素，每列一行打印：

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

1.  在 `main` 函数中，我们现在可以使用我们定义的函数：

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

    输出如下：

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

### **解决方案奖励步骤**：

1.  我们可以添加一个新的方法 `multiply`，它接受一个长度为 `C` 的 `std::array` 类型的 `T`，以 `const` 引用方式，因为我们没有修改它。

    该函数返回一个类型相同但长度为 `R` 的数组。

1.  我们遵循矩阵-向量乘法的定义来计算结果：

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

1.  现在，我们可以扩展 `main` 函数来调用 `multiply` 函数：

    ```cpp
    std::array<int, 2> vector = {8, 9};
    std::array<int, 3> result = matrix.multiply(vector);
    std::cout << “Result of multiplication: [“ << result[0] << “, “
      << result[1] << “, “ << result[2] << “]” << std::endl;
    ```

    输出如下：

    ```cpp
    Result of multiplication: [26, 87, 94]
    ```

### 活动 16：使矩阵类更容易使用

1.  我们首先导入 `<functional>` 以便访问 `std::multiplies`：

    ```cpp
    #include <functional>
    ```

1.  然后，我们将类 `template` 中的模板参数顺序改变，使得大小参数排在前面。我们还添加了一个新的模板参数 `Multiply`，这是我们默认用于在 `vector` 元素之间进行乘法运算的类型，并将其实例存储在类中：

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

    `get()` 函数与上一个活动保持相同。

1.  现在，我们需要确保 `Multiply` 方法使用用户提供的 `Multiply` 类型来执行乘法。

1.  要做到这一点，我们需要确保调用`multiplier(operand1, operand2)`而不是`operand1 * operand2`，这样我们就能使用类内部存储的实例：

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

1.  现在我们可以添加一个示例，说明我们如何使用这个类：

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

    输出如下：

    ```cpp
    Result of multiplication(with +): [20, 24, 28]
    ```

### 活动十七：确保在执行账户操作时用户已登录

1.  我们首先声明一个模板函数，它接受两个类型参数：一个`Action`类型和一个`Parameter`类型。

1.  函数应该接受用户标识、操作和参数。参数应该作为转发引用接受。作为第一步，它应该通过调用`isLoggenIn()`函数检查用户是否已登录。如果用户已登录，它应该调用`getUserCart()`函数，然后调用操作，传递购物车和转发参数：

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

1.  我们可以通过在`main`函数中调用它来测试`execute_on_user_cart`的工作方式：

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

    输出如下：

    ```cpp
    Adding items if the user is logged in
    Items added
    Removing item if the user is logged in
    The user is not logged in
    ```

### 活动十八：使用任意数量的参数安全地执行用户购物车操作

1.  我们需要扩展先前的活动以接受任何类型的引用和任意数量的参数，并将其传递给提供的操作。为此，我们需要创建一个`可变参数`模板。

1.  声明一个模板函数，它接受一个操作和一个`可变参数`数量的模板参数。函数参数应该是用户操作、要执行的操作以及扩展的模板参数`pack`，确保参数作为转发引用被接受。

1.  在函数内部，我们执行与之前相同的检查，但现在我们在将参数转发到操作时扩展它们：

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

1.  让我们在`main`函数中测试这个新函数：

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

    输出如下：

    ```cpp
    Replace items if the user is logged in
    Replacing item
    Item removed
    Items added
    Replace item if the user is logged in
    The user is not logged in
    ```

## 课五：标准库容器和算法

### 活动十九：存储用户账户

1.  首先，我们包含`array`类和输入/输出操作的必要头文件以及所需的命名空间：

    ```cpp
    #include <array>
    ```

1.  声明了一个包含十个`int`类型元素的数组：

    ```cpp
    array<int,10> balances;
    ```

1.  初始时，元素的值是未定义的，因为它是一个基本数据类型`int`的数组。数组使用`for`循环初始化，其中每个元素使用其索引初始化。使用`size()`运算符来评估数组的大小，使用下标运算符`[ ]`来访问数组的每个位置：

    ```cpp
    for (int i=0; i < balances.size(); ++i) 
    {
      balances[i] = 0;
    }
    ```

1.  现在，我们想要更新第一个和最后一个用户的值。我们可以使用`front()`和`back()`来访问这些用户的账户：

    ```cpp
    balances.front() += 100;
    balances.back() += 100;
    ```

    我们希望存储任意数量用户的账户余额。然后我们想要向账户列表中添加 100 个用户，每个用户的余额为 500。

1.  我们可以使用`vector`存储任意数量的用户。它在`<vector>`头文件中定义：

    ```cpp
    #include <vector>
    ```

1.  然后，我们声明了一个`int`类型的`vector`。可选地，我们可以通过调用`reserve(100)`预留足够的内存来存储 100 个用户的账户，以避免内存重新分配：

    ```cpp
    std::vector<int> balances;
    balances.reserve(100);
    ```

1.  最后，我们修改`for`循环，在账户向量末尾添加用户的余额：

    ```cpp
    for (int i=0; i<100; ++i) 
    {
      balances.push_back(500);
    }
    ```

### 活动 20：根据给定的用户名检索用户的余额

1.  包含 `map` 类的头文件和 `string` 的头文件：

    ```cpp
    #include <map>
    #include <string>
    ```

1.  创建一个键为 `std::string`，值为 `int` 的映射：

    ```cpp
    std::map<std::string, int> balances;
    ```

1.  使用 `insert` 和 `std::make_pair` 将用户的余额插入到 `map` 中。第一个参数是键，第二个参数是值：

    ```cpp
    balances.insert(std::make_pair(“Alice”,50));
    balances.insert(std::make_pair(“Bob”, 50));
    balances.insert(std::make_pair(“Charlie”, 50));
    ```

1.  使用 `find` 函数提供用户名以找到账户在映射中的位置。将其与 `end()` 进行比较以检查是否找到了位置：

    ```cpp
    auto donaldAccountPos = balances.find(“Donald”);
    bool hasAccount = (donaldAccountPos !=  balances.end());
    std::cout << “Donald has an account: “ << hasAccount << std::endl;
    ```

1.  现在，寻找 `Alice` 的账户。我们知道 `Alice` 有账户，所以没有必要检查我们是否找到了有效的位置。我们可以使用 `->second` 打印账户的值：

    ```cpp
    auto alicePosition = balances.find(“Alice”);
    std::cout << “Alice balance is: “ << alicePosition->second << std::endl;
    ```

### 活动 21：按顺序处理用户注册

1.  首先，包括 `stack` 类的头文件：

    ```cpp
    #include <stack>
    ```

1.  创建一个提供 `store` 类型的 `stack`：

    ```cpp
    std::stack<RegistrationForm> registrationForms;
    ```

1.  我们在用户注册时开始将表单存储在 `stack` 中。在 `storeRegistrationForm` 函数的主体中，将元素推入队列：

    ```cpp
    stack.push(form);
    std::cout << “Pushed form for user “ << form.userName << std::endl;
    ```

1.  现在，在 `endOfDayRegistrationProcessing` 内部，我们获取 `stack` 中的所有元素，然后处理它们。使用 `top()` 方法访问 `stack` 中的顶部元素，并使用 `pop()` 移除顶部元素。当我们没有元素时停止获取和移除第一个元素：

    ```cpp
    while(not stack.empty()) {
      processRegistration(stack.top());
      stack.pop();
    }
    ```

1.  最后，我们使用一些测试数据调用我们的函数：

    ```cpp
    int main(){
      std::stack<RegistrationForm> registrationForms;
      storeRegistrationForm(registrationForms, RegistrationForm{“Alice”});
      storeRegistrationForm(registrationForms, RegistrationForm{“Bob”});
      storeRegistrationForm(registrationForms, RegistrationForm{“Charlie”});
      endOfDayRegistrationProcessing(registrationForms);
    }
    ```

### 活动 22：机场系统管理

1.  我们首先创建 `Airplane` 类。确保首先包含 `variant` 的头文件：

    ```cpp
    #include <variant>
    ```

1.  然后，创建一个具有构造函数的类，该构造函数将飞机的当前状态设置为 `AtGate`：

    ```cpp
    class Airplane {
      std::variant<AtGate, Taxi, Flying> state;
      public:
        Airplane(int gate) : state(AtGate{gate}) {
          std::cout << “At gate “ << gate << std::endl;
        }
    };
    ```

1.  现在，实现 `startTaxi()` 方法。首先，使用 `std::holds_alternative<>()` 检查飞机的当前状态，如果飞机不在正确的状态，则写入错误信息并返回。

1.  如果飞机处于正确的状态，则将状态更改为 taxi，通过将其分配给 `variant`：

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

1.  我们对 `takeOff()` 方法重复相同的过程：

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

1.  我们现在可以开始查看 `currentStatus()` 方法。由于我们想要对 `variant` 中的每个状态执行操作，我们可以使用访问者。

1.  在 `Airplane` 类外部，创建一个具有为飞机状态中的每个类型提供 `operator()` 方法的类。在方法内部，打印状态信息。请记住使这些方法为公共：

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

1.  现在，创建 `currentStatus()` 方法并使用 `std::visit` 在状态上调用访问者：

    ```cpp
    void currentStatus() {
        AirplaneStateVisitor visitor;
        std::visit(visitor, state);
    }
    ```

1.  我们现在可以尝试从 `main` 函数中调用 `Airplane` 的函数：

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

## 第 6 课：面向对象编程

### 活动 23：创建游戏角色

1.  创建一个具有 `public` 方法 `moveTo` 的 `Character` 类，该方法打印 `Moved to position`：

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

1.  创建一个名为 `Position` 的 `struct`：

    ```cpp
    struct Position {
      // Fields to describe the position go here
      std::string positionIdentifier;
    };
    ```

1.  创建两个从 `Character` 类派生的类 `Hero` 和 `Enemy`：

    ```cpp
    // Hero inherits publicly from Character: it has
    // all the public member of the Character class.
    class Hero : public Character {
    };
    // Enemy inherits publicly from Character, like Hero
    class Enemy : public Character {
    };
    ```

1.  创建一个具有打印施展法术的人名的构造函数的 `Spell` 类：

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

1.  类 `Hero` 应该有一个公共方法来施展法术。使用 `Spell` 类的值：

    ```cpp
    public:
        void cast(Spell spell) {
            // Cast the spell
            std::cout << “Casting spell “ << spell.name() << std::endl;
        }
    ```

1.  类 `Enemy` 应该有一个公共方法来挥舞剑，打印 `Swinging sword`：

    ```cpp
    public:
        void swingSword() {
            // Swing the sword
            std::cout << “Swinging sword” << std::endl;
        }
    ```

1.  实现调用这些方法的各种类的 `main` 方法：

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

### 活动 24：计算员工工资

1.  我们可以创建一个具有两个虚拟方法 `getBaseSalary` 和 `getBonus` 的 `Employee` 类，因为我们希望根据员工类型更改这些方法：

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

1.  我们还定义了一个方法 `getTotalComp`，它不需要是虚拟的，但会调用两个虚拟方法：

    ```cpp
        int getTotalComp(const Deparment& dep) {

        }
    };
    ```

1.  然后，从它派生出 `Manager` 类，重写计算奖金的方法。我们可能还希望重写 `getBaseSalary`，如果我们想给经理提供不同的基本工资：

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

1.  创建一个 `Department` 类，如下所示：

    ```cpp
    class Department {
      public:
        bool hasReachedTarget() const {return true;}
        int espectedEarning() const {return 1000;}
        int effectiveEarning() const {return 1100;}
    };
    ```

1.  现在，在 `main` 函数中，按照如下所示调用 `Department`、`Employee` 和 `Manager` 类：

    ```cpp
    int main()
    {
      Department dep;
      Employee employee;
      Manager manager;
      std::cout << “Employee: “ << employee.getTotalComp(dep) << “. Manager: “ << manager.getTotalComp(dep) << std::endl;
    }
    ```

### 活动 25：检索用户信息

1.  我们必须编写可以独立于数据来源的代码。因此，我们创建了一个接口 `UserProfileStorage`，用于从 `UserId` 检索 `CustomerProfile`：

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

1.  现在，编写继承自 `UserProfileStorage` 的 `UserProfileCache` 类：

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

1.  在 `main` 函数中，按照如下所示调用 `UserProfileCache` 类和 `exampleOfUsage` 函数：

    ```cpp
    int main()
    {
      UserProfileCache cache;
      exampleOfUsage (cache);
    }
    ```

### 活动 26：创建 UserProfileStorage 工厂

1.  编写以下需要 `UserProfileStorage` 类的代码，如下所示。为了实现这一点，我们提供了一个工厂类，它有一个 `create` 方法，提供 `UserProfileStorage` 的实例。编写这个类时，确保用户不需要手动管理接口的内存：

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

1.  我们希望 `UserProfileStorageFactory` 类返回一个 `unique_ptr`，以便它管理接口的生存期：

    ```cpp
    void getUserProfile(const UserProfileStorageFactory& storageFactory) {
      std::unique_ptr<UserProfileStorage> storage = storageFactory.create();
      UserId user;
      storage->getUserProfile(user);
      // The storage is automatically destroyed
    }
    ```

1.  现在，在 `main` 函数中，按照如下所示调用 `UserProfileStorageFactory` 类：

    ```cpp
    int main()
    {
      UserProfileStorageFactory factory;
      getUserProfile(factory);
    ```

### 活动 27：使用数据库连接进行多项操作

1.  首先，创建一个可以在并行中使用 `DatabaseConnection` 类。我们希望尽可能多地重用它，我们知道我们可以使用 `std::async` 来启动一个新的并行任务：

    ```cpp
    #include <future>
    struct DatabaseConnection {};
    ```

1.  假设有两个函数 `updateOrderList(DatabaseConnection&)` 和 `scheduleOrderProcessing(DatabaseConnection&)`，编写一个函数来创建 `DatabaseConnection` 并将其传递给两个并行任务。（注意，我们不知道哪个任务先完成）：

    ```cpp
    void updateOrderList(DatabaseConnection&) {}
    void scheduleOrderProcessing(DatabaseConnection&) {}
    ```

1.  你必须理解何时以及如何创建 `shared_ptr`。你也可以使用以下代码来正确编写 `shared_ptr`。

    ```cpp
    /* We need to get a copy of the shared_ptr so it stays alive until this function finishes */
    void updateWithConnection(std::shared_ptr<DatabaseConnection> connection) {
        updateOrderList(*connection);
    }
    ```

    有多个用户使用这个连接，我们不知道哪个是所有者，因为只要有人使用它，连接就需要保持活跃。

1.  为了模拟这种情况，我们使用 `shared_ptr`。记住，为了使连接保持有效，我们需要一个 `shared_ptr` 的副本：

    ```cpp
    /* We need to get a copy of the shared_ptr so it stays alive until this function finishes. */
    void scheduleWithConnection(std::shared_ptr<DatabaseConnection> connection) {
        scheduleOrderProcessing(*connection);
    }
    ```

1.  创建 `main` 函数如下：

    ```cpp
    int main()
    {
        std::shared_ptr<DatabaseConnection> connection = std::make_shared<DatabaseConnection>();
        std::async(std::launch::async, updateWithConnection, connection);
        std::async(std::launch::async, scheduleWithConnection, connection);
    }
    ```
