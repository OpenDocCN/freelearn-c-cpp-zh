# 7

# C++ 中的类、对象和面向对象编程（OOP）

在本章中，我们深入探讨了 C++ 中类、对象和面向对象编程（OOP）的复杂领域。针对高级 C++ 实践者，我们的重点是提高你对类设计、方法实现、继承和模板使用的理解，避免对这些概念进行入门级解释。我们的目标是提高你使用高级面向对象技术构建健壮和高效软件架构的能力。

讨论从检查定义类时必要的复杂考虑开始，引导你通过决策过程来确定类封装的最佳候选者。这包括区分简单数据结构（如结构体）可能更合适的情况，从而优化性能和可读性。

此外，我们探讨了类内方法的设计——突出各种类型的方法，如访问器、修改器和工厂方法，并建立促进代码清晰性和可维护性的约定。特别关注高级方法设计实践，包括 const 正确性和可见性范围，这对于确保和优化对类数据的访问至关重要。

继承，作为面向对象（OOP）的基石，不仅被审查其优点，还被审查其缺点。为了提供一个平衡的视角，我们提出了如组合和接口隔离等替代方案，这些方案可能在某些情况下更好地服务于你的设计目标。这种细微的讨论旨在为你提供必要的洞察力，以便根据项目的具体需求和约束选择最佳的继承策略或其替代方案。

将讨论扩展到泛型编程，我们深入探讨了复杂的模板使用，包括模板元编程等高级技术。本节旨在展示如何利用模板创建高度可重用和高效的代码。此外，我们还将简要介绍使用面向对象原则设计 API，强调精心设计的接口可以显著提高软件组件的可使用性和持久性。

每个主题都配有来自现实应用中的实际例子和案例研究，展示了这些高级技术在现代软件开发中的应用。到本章结束时，你应该对如何利用 C++ 中的面向对象（OOP）特性来构建优雅、高效和可扩展的软件架构有更深入的理解。

# 类的好候选者

在面向对象（OOP）中识别类的好候选者涉及寻找自然封装数据和行为的实体。

## 内聚性

一个类应该代表一组紧密相关的功能。这意味着类中的所有方法和数据都直接与其提供的特定功能相关。例如，一个`Timer`类是一个很好的候选者，因为它封装了与计时相关的属性和方法（开始、停止、重置时间），保持了高度的聚合性。

## 封装

具有应从外部干扰或误用中屏蔽的属性和行为实体可以封装在类中。

一个`BankAccount`类封装了余额（属性）以及如`deposit`、`withdraw`和`transfer`之类的行为，确保余额操作仅通过受控和安全操作进行。

## 可重用性

类应该设计成可以在程序的不同部分或甚至在不同程序中重用。

一个管理数据库连接的`DatabaseConnection`类可以在需要数据库交互的多个应用程序中重用，处理连接、断开连接和错误管理。

## 抽象

一个类应该通过隐藏复杂的逻辑来提供一个简化的接口，代表更高层次的抽象。例如，标准库中有如`std::vector`之类的类，它们抽象了动态数组的复杂性，为数组操作提供了一个简单的接口。

## 实际实体

类通常代表与正在建模的系统相关的现实世界中的对象。

在航班预订系统中，如`Flight`、`Passenger`和`Ticket`之类的类是很好的候选者，因为它们直接代表具有清晰属性和行为的现实世界对象。

## 管理复杂性

类应该通过将大问题分解成更小、更易于管理的部分来帮助管理复杂性。

这里有一个例子——在图形编辑软件中，一个`GraphicObject`类可能作为更具体图形对象（如`Circle`、`Rectangle`和`Polygon`）的基类，系统地组织图形属性和功能。

## 通过封装最小化类的职责

封装是面向对象编程中的一个基本概念，它涉及将数据（属性）和操作数据的方法（函数）捆绑成一个单一单元或类。它不仅隐藏了对象的内部状态，还模块化了其行为，使软件更容易管理和扩展。然而，一个类应该封装多少功能和数据可以显著影响应用程序的可维护性和可扩展性。

### 类中的过度封装——一个常见的陷阱

在实践中，在单个类中封装过多的功能和数据是一个常见的错误，可能导致多个问题。这通常会导致一个**神对象**——一个控制应用程序中太多不同部分的类，自己承担了太多工作。这样的类通常难以理解，难以维护，且测试起来有问题。

让我们看看一个封装不良的`Car`类的例子。

考虑以下`Car`类的示例，它试图管理汽车的基本属性以及其内部系统的详细方面，如发动机、变速箱和娱乐系统：

```cpp
#include <iostream>
#include <string>
class Car {
private:
    std::string _model;
    double _speed;
    double _fuel_level;
    int _gear;
    bool _entertainment_system_on;
public:
    Car(const std::string& model) : _model(model), _speed(0), _fuel_level(50), _gear(1), _entertainment_system_on(false) {}
    void accelerate() {
        if (_fuel_level > 0) {
            _speed += 10;
            _fuel_level -= 5;
            std::cout << "Accelerating. Current speed: " << _speed << " km/h, Fuel level: " << _fuel_level << " liters" << std::endl;
        } else {
            std::cout << "Not enough fuel." << std::endl;
        }
    }
    void change_gear(int new_gear) {
        _gear = new_gear;
        std::cout << "Gear changed to: " << _gear << std::endl;
    }
    void toggle_entertainment_system() {
        _entertainment_system_on = !_entertainment_system_on;
        std::cout << "Entertainment System is now " << (_entertainment_system_on ? "on" : "off") << std::endl;
    }
    void refuel(double amount) {
        _fuel_level += amount;
        std::cout << "Refueling. Current fuel level: " << _fuel_level << " liters" << std::endl;
    }
};
```

这个`Car`类有问题，因为它试图管理汽车功能太多的方面，这些方面最好由专门的组件来处理。

### 使用组合进行适当的封装

一个更好的方法是使用组合将责任委托给其他类，每个类处理系统功能的一个特定部分。这不仅遵循单一职责原则，而且使系统更加模块化，更容易维护。

下面是一个使用组合设计良好的`Car`类的示例：

```cpp
#include <iostream>
#include <string>
class Engine {
private:
    double _fuel_level;
public:
    Engine() : _fuel_level(50) {}
    void consume_fuel(double amount) {
        _fuel_level -= amount;
        std::cout << "Consuming fuel. Current fuel level: " << _fuel_level << " liters" << std::endl;
    }
    void refuel(double amount) {
        _fuel_level += amount;
        std::cout << "Engine refueled. Current fuel level: " << _fuel_level << " liters" << std::endl;
    }
    double get_fuel_level() const {
        return _fuel_level;
    }
};
class Transmission {
private:
    int _gear;
public:
    Transmission() : _gear(1) {}
    void change_gear(int new_gear) {
        _gear = new_gear;
        std::cout << "Transmission: Gear changed to " << _gear << std::endl;
    }
};
class EntertainmentSystem {
private:
    bool _is_on;
public:
    EntertainmentSystem() : _is_on(false) {}
    void toggle() {
        _is_on = !_is_on;
        std::cout << "Entertainment System is now " << (_is_on ? "on" : "off") << std::endl;
    }
};
class Car {
private:
    std::string _model;
    double _speed;
    Engine _engine;
    Transmission _transmission;
    EntertainmentSystem _entertainment_system;
public:
    Car(const std::string& model) : _model(model), _speed(0) {}
    void accelerate() {
        if (_engine.get_fuel_level() > 0) {
            _speed += 10;
            _engine.consume_f
uel(5);
            std::cout << "Car accelerating. Current speed: " << _speed << " km/h" << std::endl;
        } else {
            std::cout << "Not enough fuel to accelerate." << std::endl;
        }
    }
    void change_gear(int gear) {
        _transmission.change_gear(gear);
    }
    void toggle_entertainment_system() {
        _entertainment_system.toggle();
    }
    void refuel(double amount) {
        _engine.refuel(amount);
    }
};
```

在这个改进的设计中，`Car`类充当其组件之间的协调者，而不是直接管理每个细节。每个子系统——发动机、变速箱和娱乐系统——处理自己的状态和行为，导致一个更容易维护、测试和扩展的设计。这个例子展示了适当的封装和组合如何显著提高面向对象软件的结构和质量。

## C++中结构和类的使用

在 C++中，结构和类都用于定义用户定义的类型，可以包含数据和函数。它们之间的主要区别在于它们的默认访问级别：类的成员默认是私有的，而结构体的成员默认是公开的。这种区别微妙地影响了它们在 C++编程中的典型用途。

### 结构体——理想的被动数据结构

在 C++中，结构体特别适合创建被动数据结构，其主要目的是存储数据而不需要封装太多行为。由于它们的默认公开性质，结构体通常用于当你想要允许直接访问数据成员时，这可以简化代码并减少操作数据所需额外函数的需求。

以下列表概述了你应该使用结构体的实例：

+   **数据对象**：结构体非常适合创建**纯数据**（**POD**）结构。这些是主要持有数据且功能很少或没有的方法简单对象。例如，结构体通常用于表示空间中的坐标、RGB 颜色值或设置配置，在这些情况下，直接访问数据字段比通过获取器和设置器更方便：

    ```cpp
    struct Color {
    ```

    ```cpp
        int red = 0;
    ```

    ```cpp
        int green = 0;
    ```

    ```cpp
        int blue = 0;
    ```

    ```cpp
    };
    ```

    ```cpp
    struct Point {
    ```

    ```cpp
        double x = 0.0;
    ```

    ```cpp
        double y = 0.0;
    ```

    ```cpp
        double z = 0.0;
    ```

    ```cpp
    };
    ```

    ```cpp
    Fortunately, C++ 11 and C++ 20 provide aggregate initialization and designated initializers, making it easier to initialize structs with default values.
    ```

    ```cpp
    // C++ 11
    ```

    ```cpp
       auto point = Point {1.1, 2.2, 3.3};
    ```

    ```cpp
    // C++ 20
    ```

    ```cpp
       auto point2 = Point {.x = 1.1, .y = 2.2, .z = 3.3};
    ```

    如果你的项目不支持 C++ 20，你可以利用 C99 指定的初始化器来实现类似的效果：

    ```cpp
       auto point3 = Point {.x = 1.1, .y = 2.2, .z = 3.3};
    ```

+   **互操作性**：结构体在接口 C 或数据对齐和布局至关重要的系统中很有用。它们确保在底层操作（如硬件接口或网络通信）中的兼容性和性能。

+   **轻量级容器**：当你需要轻量级容器来组合几个变量时，结构体比类提供更透明和更不繁琐的方法。它们对于封装不是主要关注的小聚合来说很理想。

### 类 – 封装复杂性

类是 C++面向对象编程的支柱，用于将数据和行为封装成一个单一实体。默认的私有访问修饰符鼓励隐藏内部状态和实现细节，促进遵循封装和抽象原则的更严格设计。

以下列表解释了何时应该使用类：

+   **复杂系统**：对于涉及复杂数据处理、状态管理和接口控制的组件，类是首选选择。它们提供了数据保护和接口抽象的机制，这对于维护软件系统的完整性和稳定性至关重要：

    ```cpp
    class Car {
    ```

    ```cpp
    private:
    ```

    ```cpp
        int speed;
    ```

    ```cpp
        double fuel_level;
    ```

    ```cpp
    public:
    ```

    ```cpp
        void accelerate();
    ```

    ```cpp
        void brake();
    ```

    ```cpp
        void refuel(double amount);
    ```

    ```cpp
    };
    ```

+   **行为封装**：当功能（方法）与数据一样重要时，类是理想的。将行为与数据封装到类中可以允许更易于维护和更无错误的代码，因为对数据的操作是紧密控制和明确定义的。

+   **继承和多态**：类支持继承和多态，能够创建可以动态扩展和修改的复杂对象层次结构。这在许多软件设计模式和高级系统架构中是必不可少的。

在 C++中选择结构体和类应根据预期用途进行指导：结构体用于简单、透明的数据容器，其中直接数据访问是可以接受或必要的，而类用于需要封装、行为和接口控制的更复杂系统。理解和利用每个的优点可以导致更干净、更高效和可扩展的代码。

## 类中的常见方法类型 – 获取器和设置器

在面向对象编程（OOP）中，尤其是在像 Java 这样的语言中，**获取器**和**设置器**是标准方法，它们作为访问和修改类私有数据成员的主要接口。这些方法提供了对对象属性的受控访问，遵循封装原则，这是有效面向对象设计的基石。

### 获取器和设置器的目的和约定

获取器（也称为访问器）是用于检索私有字段值的函数。它们不会修改数据。设置器（也称为修改器）是允许根据接收到的输入修改私有字段的函数。这些方法通过在设置数据时可能强制执行约束或条件，使对象的内部状态保持一致和有效。

这里是获取器和设置器的约定：

+   `x`的 getter 命名为`get_x()`，setter 命名为`set_x(value)`。这种命名约定在 Java 中几乎是通用的，并且在支持基于类的 OOP 的其他编程语言中也普遍采用。

+   **返回类型和参数**：属性的 getter 返回与属性本身相同的类型，并且不接受任何参数，而 setter 返回 void，并接受与设置的属性相同类型的参数。

下面是一个 C++中的例子：

```cpp
class Person {
private:
    std::string _name;
    int _age;
public:
    // Getter for the name property
    std::string get_name() const { return _name; }
    // Setter for the name property
    void set_name(const std::string& name) { _name = name; }
    // Getter for the age property
    int get_age() const { return _age; }
    // Setter for the age property
    void set_age(int age) {
        if (age >= 0) { // validate the age
            _age = age;
        }
    }
};
```

### 有用性和建议

**受控访问和验证**：getter 和 setter 封装了类的字段，提供了受控访问和验证逻辑。这有助于维护数据的完整性，确保不会设置无效或不适当的值。

**灵活性**：通过使用 getter 和 setter，开发者可以在不改变类的外部接口的情况下更改数据存储和检索的底层实现。这在维护向后兼容性或需要为优化更改数据表示时特别有用。

**一致性**：这些方法可以强制执行需要在对象生命周期内持续维护的规则。例如，确保字段永远不会持有 null 值或遵循特定的格式。

### 何时使用 getter 和 setter，何时不使用

常规做法是在存在封装、业务逻辑或继承复杂性的类中使用 getter 和 setter。例如，对于具有相对复杂逻辑的`Car`和`Engine`类，getter 和 setter 对于维护数据的完整性和确保系统正确运行是必不可少的。另一方面，对于像`Point`或`Color`这样的简单数据结构，其主要目的是存储数据而不涉及太多行为，使用具有公共数据成员的结构体可能更合适。请注意，如果结构体是库或 API 的一部分，为了未来的可扩展性，提供 getter 和 setter 可能是有益的。

这种细微的方法允许开发者平衡控制与简单性，为软件组件的具体需求选择最合适的工具。

# C++中的继承

继承和组合是 C++中两个基本面向对象编程概念，它们使得创建复杂且可重用的软件设计成为可能。它们促进了代码重用，并有助于模拟现实世界的关系，尽管它们的工作方式不同。

继承允许一个类（称为派生类或子类）从另一个类（称为基类或超类）继承属性和行为。这使得派生类可以重用基类中的代码，同时扩展或覆盖其功能。例如，考虑一个`BaseSocket`类及其派生类`TcpSocket`和`UdpSocket`。派生类继承了`BaseSocket`的基本功能，并添加了它们特定的实现：

```cpp
class BaseSocket {
public:
    virtual ssize_t send(const std::vector<uint8_t>& data) = 0;
    virtual ~BaseSocket() = default;
};
class TcpSocket : public BaseSocket {
public:
    ssize_t send(const std::vector<uint8_t>& data) override {
        // Implement TCP-specific send logic here
    }
};
class UdpSocket : public BaseSocket {
public:
    ssize_t send(const std::vector<uint8_t>& data) override {
        // Implement UDP-specific send logic here
    }
};
```

在这个例子中，`TcpSocket` 和 `UdpSocket` 类继承自 `BaseSocket`，展示了继承如何促进代码重用并建立“是一种”关系。继承还支持多态，允许派生类的对象被当作基类的实例来处理，从而实现动态方法绑定。

另一方面，组合涉及通过包含其他类的对象来创建类。而不是从基类继承，一个类由一个或多个其他类的对象组成，这些对象用于实现所需的功能。这代表了一种“有”的关系。例如，考虑一个可以拥有 `BaseSocket` 的 `CommunicationChannel` 类。`CommunicationChannel` 类使用 `BaseSocket` 对象来实现其通信功能，展示了组合：

```cpp
class CommunicationChannel {
public:
    CommunicationChannel(std::unique_ptr<BaseSocket> sock) : _socket(sock) {}
    bool transmit(const std::vector<uint8_t>& data) {
        size_t total_sent = 0;
        size_t data_size = data.size();
        while (total_sent < data_size) {
            ssize_t bytesSent = _socket->send({data.begin() + total_sent, data.end()});
            if (bytesSent < 0) {
                std::cerr << "Error sending data." << std::endl;
                return false;
            }
            total_sent += bytesSent;
        }
        std::cout << "Communication channel transmitted " << total_sent << " bytes." << std::endl;
        return true;
    }
private:
    std::unique_ptr<BaseSocket> _socket;
};
int main() {
    TcpSocket tcp;
    CommunicationChannel channel(std::make_unique<TcpSocket>());
    std::vector<uint8_t> data = {1, 2, 3, 4, 5};
    if (channel.transmit(data)) {
        std::cout << "Data transmitted successfully." << std::endl;
    } else {
        std::cerr << "Data transmission failed." << std::endl;
    }
    return 0;
}
```

在这个例子中，`CommunicationChannel` 类包含一个 `BaseSocket` 对象，并使用它来实现其功能。`transmit` 方法将数据分块发送，直到所有数据发送完毕，并检查错误（当返回值小于 `0` 时）。这展示了组合如何提供灵活性，允许对象在运行时动态组装。它还通过包含对象并仅暴露必要的接口来促进更好的封装，从而避免类之间的紧密耦合，使代码更模块化且易于维护。

总结来说，继承和组合都是 C++ 中创建可重用和维护性代码的重要工具。继承适用于具有明确层次关系且需要多态的场景，而组合则是从更简单的组件组装复杂行为时的理想选择，提供了灵活性和更好的封装。理解何时使用每种方法对于有效的面向对象设计至关重要。

## C++ 中继承的演变

最初，继承被视为一种强大的工具，可以减少代码重复并增强代码的表达性。它允许创建一个派生类，该类从基类继承属性和行为。然而，随着 C++ 在复杂系统中的应用增长，继承作为一刀切解决方案的局限性变得明显。

## 二进制级别的继承实现

有趣的是，在二进制级别，C++ 中的继承实现与组合类似。本质上，派生类在其结构中包含基类的一个实例。这可以通过一个简化的 ASCII 图表来可视化：

```cpp
+-------------------+
|   Derived Class   |
|-------------------|
|  Base Class Part  | <- Base class subobject
|-------------------|
| Derived Class Data| <- Additional data members of the derived class
+-------------------+
```

在这种布局中，派生类对象中的基类部分包含属于基类的所有数据成员，并在内存中直接跟在其后的是派生类的附加数据成员。请注意，内存中数据成员的实际顺序可能受到对齐要求、编译器优化等因素的影响。

## 继承的优缺点

这里是继承的优点：

+   `MediaContent` 类将作为所有类型媒体内容的基类。它将封装常见的属性和行为，例如 `title`（标题）、`duration`（时长）和基本的播放控制（`play`（播放）、`pause`（暂停）、`stop`（停止））：

    ```cpp
    #include <iostream>
    ```

    ```cpp
    #include <string>
    ```

    ```cpp
    // Base class for all media content
    ```

    ```cpp
    class MediaContent {
    ```

    ```cpp
    protected:
    ```

    ```cpp
        std::string _title;
    ```

    ```cpp
        int _duration; // Duration in seconds
    ```

    ```cpp
    public:
    ```

    ```cpp
        MediaContent(const std::string& title, int duration)
    ```

    ```cpp
            : _title(title), _duration(duration) {}
    ```

    ```cpp
        auto title() const { return _title; }
    ```

    ```cpp
        auto duration() const { return duration; }
    ```

    ```cpp
        virtual void play() = 0; // Start playing the content
    ```

    ```cpp
        virtual void pause() = 0;
    ```

    ```cpp
        virtual void stop() = 0;
    ```

    ```cpp
        virtual ~MediaContent() = default;
    ```

    ```cpp
    };
    ```

    `Audio` 类扩展了 `MediaContent`，添加了与音频文件相关的特定属性，例如比特率：

    ```cpp
    class Audio : public MediaContent {
    ```

    ```cpp
    private:
    ```

    ```cpp
        int _bitrate; // Bitrate in kbps
    ```

    ```cpp
    public:
    ```

    ```cpp
        Audio(const std::string& title, int duration, int bitrate)
    ```

    ```cpp
            : MediaContent(title, duration), _bitrate(bitrate) {}
    ```

    ```cpp
        auto bitrate() const { return _bitrate; }
    ```

    ```cpp
        void play() override {
    ```

    ```cpp
            std::cout << "Playing audio: " << title << ", Duration: " << duration
    ```

    ```cpp
                      << "s, Bitrate: " << bitrate << "kbps" << std::endl;
    ```

    ```cpp
        }
    ```

    ```cpp
        void pause() override {
    ```

    ```cpp
            std::cout << "Audio paused: " << title << std::endl;
    ```

    ```cpp
        }
    ```

    ```cpp
        void stop() override {
    ```

    ```cpp
            std::cout << "Audio stopped: " << title << std::endl;
    ```

    ```cpp
        }
    ```

    ```cpp
    };
    ```

    同样，`Video` 类扩展了 `MediaContent` 并引入了额外的属性，例如 `resolution`（分辨率）：

    ```cpp
    class Video : public MediaContent {
    ```

    ```cpp
    private:
    ```

    ```cpp
        std::string _resolution; // Resolution as width x height
    ```

    ```cpp
    public:
    ```

    ```cpp
        Video(const std::string& title, int duration, const std::string& resolution)
    ```

    ```cpp
            : MediaContent(title, duration), _resolution(resolution) {}
    ```

    ```cpp
        auto resolution() const { return _resolution; }
    ```

    ```cpp
        void play() override {
    ```

    ```cpp
            std::cout << "Playing video: " << title << ", Duration: " << duration
    ```

    ```cpp
                      << "s, Resolution: " << resolution << std::endl;
    ```

    ```cpp
        }
    ```

    ```cpp
        void pause() override {
    ```

    ```cpp
            std::cout << "Video paused: " << title << std::endl;
    ```

    ```cpp
        }
    ```

    ```cpp
        void stop() override {
    ```

    ```cpp
            std::cout << "Video stopped: " << title << std::endl;
    ```

    ```cpp
        }
    ```

    ```cpp
    };
    ```

    下面是如何在简单的媒体播放器系统中使用这些类：

    ```cpp
    int main() {
    ```

    ```cpp
        Audio my_song("Song Example", 300, 320);
    ```

    ```cpp
        Video my_movie("Movie Example", 7200, "1920x1080");
    ```

    ```cpp
        my_song.play();
    ```

    ```cpp
        my_song.pause();
    ```

    ```cpp
        my_song.stop();
    ```

    ```cpp
        my_movie.play();
    ```

    ```cpp
        my_movie.pause();
    ```

    ```cpp
        my_movie.stop();
    ```

    ```cpp
        return 0;
    ```

    ```cpp
    }
    ```

    在这个例子中，`Audio` 和 `Video` 都继承自 `MediaContent`。这使我们能够重用 `title` 和 `duration` 属性，并需要实现针对每种媒体类型的播放控制（`play`、`pause`、`stop`）。这个层次结构展示了继承如何促进代码重用和系统可扩展性，同时在一个统一的框架中为不同类型的媒体内容启用特定的行为。每个类只添加其类型独有的内容，遵循基类提供通用功能，派生类为特定需求扩展或修改该功能的原理。

+   **多态性**：通过继承，C++ 支持多态性，允许使用基类引用来引用派生类对象。这实现了动态方法绑定和对多个派生类型的灵活接口。我们的媒体内容层次结构可以用于实现一个可以统一处理不同类型媒体内容的媒体播放器：

    ```cpp
    class MediaPlayer {
    ```

    ```cpp
    private:
    ```

    ```cpp
        std::vector<std::unique_ptr<MediaContent>> _playlist;
    ```

    ```cpp
    public:
    ```

    ```cpp
        void add_media(std::unique_ptr<MediaContent> media) {
    ```

    ```cpp
            _playlist.push_back(std::move(media));
    ```

    ```cpp
        }
    ```

    ```cpp
        void play_all() {
    ```

    ```cpp
            for (auto& media : _playlist) {
    ```

    ```cpp
                media->play();
    ```

    ```cpp
                // Additional controls can be implemented
    ```

    ```cpp
            }
    ```

    ```cpp
        }
    ```

    ```cpp
    };
    ```

    ```cpp
    int main() {
    ```

    ```cpp
        MediaPlayer player;
    ```

    ```cpp
        player.add(std::make_unique<Audio>("Jazz in Paris", 192, 320));
    ```

    ```cpp
        player.add(std::make_unique<Video>("Tour of Paris", 1200, "1280x720"));
    ```

    ```cpp
        player.play_all();
    ```

    ```cpp
        return 0;
    ```

    ```cpp
    }
    ```

    `add` 方法接受任何从 `MediaContent` 派生的媒体内容类型，通过使用基类指针来引用派生类对象，展示了多态性。这是通过将媒体项存储在 `std::vector` 的 `std::unique_ptr<MediaContent>` 中实现的。`play_all` 方法遍历存储的媒体，并对每个项目调用播放方法。尽管实际的媒体类型不同（音频或视频），媒体播放器将它们都视为 `MediaContent`。正确的播放方法（来自 `Audio` 或 `Video`）在运行时被调用，这是动态多态性（也称为动态分派）的一个例子。

+   **分层结构**：它提供了一种自然的方式，以分层的方式组织相关类，从而模拟现实世界的关系。

这里是继承的缺点：

+   **紧密耦合**：继承在基类和派生类之间创建了一种紧密耦合。基类中的更改可能会无意中影响派生类，导致代码脆弱，当修改基类时可能会崩溃。以下示例通过继承在软件系统中说明了紧密耦合的问题。我们将使用一个涉及在线商店的场景，该商店使用类层次结构管理不同类型的折扣。

## 基类 – 折扣

`Discount` 类为所有类型的折扣提供了基本的结构和功能。它根据百分比减少来计算折扣；

```cpp
#include <iostream>
class Discount {
protected:
    double _discount_percent;  // Percent of discount
public:
    Discount(double percent) : _discount_percent(percent) {}
    virtual double apply_discount(double amount) {
        return amount * (1 - _discount_percent / 100);
    }
};
```

## 派生类 – 季节性折扣

`SeasonalDiscount`类扩展了`Discount`，并根据季节因素修改折扣计算，例如在假日季节增加折扣：

```cpp
class SeasonalDiscount : public Discount {
public:
    SeasonalDiscount(double percent) : Discount(percent) {}
    double apply_discount(double amount) override {
        // Let's assume the discount increases by an additional 5% during holidays
        double additional = 0.05;  // 5% extra during holidays
        return amount * (1 - (_discount_percent / 100 + additional));
    }
};
```

## 派生类 – ClearanceDiscount

`ClearanceDiscount`类也扩展了`Discount`，用于处理折扣可能显著更高的清仓商品：

```cpp
class ClearanceDiscount : public Discount {
public:
    ClearanceDiscount(double percent) : Discount(percent) {}
    double apply_discount(double amount) override {
        // Clearance items get an extra 10% off beyond the configured discount
        double additional = 0.10;  // 10% extra for clearance items
        return amount * (1 - (_discount_percent / 100 + additional));
    }
};
```

演示和紧耦合问题：

```cpp
int main() {
    Discount regular(20); // 20% regular discount
    SeasonalDiscount holiday(20); // 20% holiday discount, plus extra
    ClearanceDiscount clearance(20); // 20% clearance discount, plus extra
    std::cout << "Regular Price $100 after discount: $" << regular.apply_discount(100) << std::endl;
    std::cout << "Holiday Price $100 after discount: $" << holiday.apply_discount(100) << std::endl;
    std::cout << "Clearance Price $100 after discount: $" << clearance.apply_discount(100) << std::endl;
    return 0;
}
```

## 紧耦合问题

以下是一个紧耦合问题的列表：

+   `apply_discount`)。任何对基类方法签名或`apply_discount`内部逻辑的更改都可能需要修改所有派生类。

+   `_discount_percent`。如果基类中的公式发生变化（例如，包含最小或最大限制），所有子类可能需要进行大量修改以符合新的逻辑。

+   **不灵活性**：这种耦合使得在不影响其他类型的情况下修改一种折扣类型的行为变得困难。这种设计在可能需要独立演变折扣计算策略的地方缺乏灵活性。

## 解决方案 – 使用策略模式解耦

减少这种耦合的一种方法是通过使用**策略模式**，它涉及定义一组算法（折扣策略），封装每个算法，并使它们可互换。这允许折扣算法独立于使用它们的客户端而变化：

```cpp
class DiscountStrategy {
public:
    virtual double calculate(double amount) = 0;
    virtual ~DiscountStrategy() {}
};
class RegularDiscountStrategy : public DiscountStrategy {
public:
    double calculate(double amount) override {
        return amount * 0.80; // 20% discount
    }
};
class HolidayDiscountStrategy : public DiscountStrategy {
public:
    double calculate(double amount) override {
        return amount * 0.75; // 25% discount
    }
};
class ClearanceDiscountStrategy : public DiscountStrategy {
public:
    double calculate(double amount) override {
        return amount * 0.70; // 30% discount
    }
};
// Use these strategies in a Discount context class
class Discount {
private:
    std::unique_ptr<DiscountStrategy> _strategy;
public:
    Discount(std::unique_ptr<DiscountStrategy> strat) : _strategy(std::move(strat)) {}
    double apply_discount(double amount) {
        return _strategy->calculate(amount);
    }
};
```

这种方法通过将折扣计算与使用它的客户端（`Discount`）解耦，允许每个折扣策略独立演变而不影响其他策略。减少耦合的其他几种方法包括：

+   `HybridFlyingElectricCar`类继承自`ElectricCar`和`FlyingCar`，每个这些类进一步继承自它们各自的层次结构，导致高度纠缠的类结构。这种复杂性使得系统难以调试、扩展或可靠地使用，同时也增加了在各种场景下测试和维护一致行为所面临的挑战。

    为了管理由广泛使用继承引入的复杂性，可以推荐几种策略。优先考虑组合而非继承通常提供更大的灵活性，允许系统由定义良好、松散耦合的组件组成，而不是依赖于僵化的继承结构。保持继承链短且可管理——通常不超过两到三级——有助于保持系统清晰性和可维护性。在 Java 和 C#等语言中使用接口提供了一种实现多态行为的方法，而不需要与继承相关的开销。当多继承不可避免时，确保清晰的文档并考虑使用类似接口的结构或混入（mixins）至关重要，这有助于最小化复杂性并增强系统健壮性。

+   **Liskov 替换原则 (LSP)**：我们在本书中较早提到了这个原则；LSP 声明，超类对象应该可以替换为其子类对象，而不会改变程序的可取属性（正确性、执行的任务等）。继承有时可能导致违反此原则，特别是当子类偏离基类预期的行为时。以下各节包括与 LSP 违反相关的典型问题，通过简单的示例进行说明。

### 派生类中的意外行为

当派生类以改变预期行为的方式覆盖基类的方法时，这些对象被互换使用时可能会导致意外结果：

```cpp
class Bird {
public:
    virtual void fly() {
        std::cout << "This bird flies" << std::endl;
    }
};
class Ostrich : public Bird {
public:
    void fly() override {
        throw std::logic_error("Ostriches can't fly!");
    }
};
void make_bird_fly(Bird& b) {
    b.fly();  // Expecting all birds to fly
}
```

在这里，将 `Bird` 对象替换为 `Ostrich` 对象在 `make_bird_fly` 函数中会导致运行时错误，因为鸵鸟不能飞，违反了 LSP。`Bird` 类的用户期望任何子类都能飞行，而 `Ostrich` 打破了这一期望。

### 方法先决条件问题

如果派生类对方法施加的先决条件比基类施加的更严格，它可能会限制子类的可用性并违反 LSP：

```cpp
class Payment {
public:
    virtual void pay(int amount) {
        if (amount <= 0) {
            throw std::invalid_argument("Amount must be positive");
        }
        std::cout << "Paying " << amount << std::endl;
    }
};
class CreditPayment : public Payment {
public:
    void pay(int amount) override {
        if (amount < 100) {  // Stricter precondition than the base class
            throw std::invalid_argument("Minimum amount for credit payment is 100");
        }
        std::cout << "Paying " << amount << " with credit" << std::endl;
    }
};
```

在这里，`CreditPayment` 类不能替代 `Payment` 类，否则可能会因为金额低于 100 而抛出错误，尽管这样的金额对于基类来说是完全有效的。

### LSP 违反的解决方案

+   **以 LSP 为设计理念**：在设计你的类层次结构时，确保任何子类都可以替代父类而不改变程序的可取属性

+   **使用组合而非继承**：如果子类完全遵守基类契约没有意义，请使用组合而非继承

+   **明确定义行为契约**：记录并强制执行基类的预期行为，并确保所有派生类严格遵循这些契约，不引入更严格的先决条件或改变后置条件

通过密切关注这些原则和潜在陷阱，开发者可以创建更稳健和可维护的面向对象设计。

虽然 C++ 中的继承仍然是一个有价值的特性，但理解何时以及如何有效地使用它至关重要。继承在二进制层面上类似于组合的实现细节突显了它本质上是在对象内存布局中结构和访问数据。从业者必须仔细考虑是否继承或组合（或两者的组合）将最好地服务于他们的设计目标，特别是在系统灵活性、可维护性和对 OOP 原则（如 LSP）的稳健应用方面。与软件开发中的许多特性一样，关键在于为正确的工作使用正确的工具。

# 模板和泛型编程

模板和泛型编程是 C++的关键特性，它们使得创建灵活且可重用的组件成为可能。虽然本章提供了这些强大工具的概述，但重要的是要注意，模板的主题，尤其是模板元编程，内容丰富到足以填满整本书。对于那些寻求深入探索的人，推荐阅读关于 C++模板和元编程的专门资源。

## 模板有什么好处？

模板在需要在不同类型的数据上执行相似操作的场景中特别有用。它们允许你编写一段可以与任何类型一起工作的代码。以下小节概述了一些常见的用例和示例。

## 泛型算法

算法可以在不重写针对每种类型的代码的情况下作用于不同的类型。例如，标准库中的`std::sort`函数可以排序任何类型的元素，只要元素可以进行比较：

```cpp
#include <algorithm>
#include <vector>
#include <iostream>
template <typename T>
void print(const std::vector<T>& vec) {
    for (const T& elem : vec) {
        std::cout << elem << " ";
    }
    std::cout << std::endl;
}
int main() {
    std::vector<int> int_vec = {3, 1, 4, 1, 5};
    std::sort(int_vec.begin(), int_vec.end());
    print(int_vec); // Outputs: 1 1 3 4 5
    std::vector<std::string> string_vec = {"banana", "apple", "cherry"};
    std::sort(string_vec.begin(), string_vec.end());
    print(string_vec); // Outputs: apple banana cherry
    return 0;
}
```

## 容器类

模板在标准库中大量使用，例如`std::vector`、`std::list`和`std::map`，这些容器可以存储任何类型的元素：

```cpp
#include <vector>
#include <iostream>
int main() {
    std::vector<int> int_vec = {1, 2, 3};
    std::vector<std::string> string_vec = {"hello", "world"};
    for (int val : int_vec) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    for (const std::string& str : string_vec) {
        std::cout << str << " ";
    }
    std::cout << std::endl;
    return 0;
}
```

如果不使用模板，开发者在使用集合时的选择将限于为每种类型的集合创建单独的类（例如，`IntVector`、`StringVector`等），或者要求使用一个公共基类，这会需要类型转换并失去类型安全性，例如：

```cpp
class BaseObject {};
class Vector {
public:
    void push_back(BaseObject* obj);
};
```

另一种选择是存储一些`void`指针，并在检索时将它们转换为所需的类型，但这种方法更容易出错。

标准库使用模板为智能指针如`std::unique_ptr`和`std::shared_ptr`，它们管理动态分配对象的生存期：

```cpp
#include <memory>
#include <iostream>
int main() {
    std::unique_ptr<int> ptr = std::make_unique<int>(42);
    std::cout << "Value: " << *ptr << std::endl; // Outputs: Value: 42
    std::shared_ptr<int> shared_ptr = std::make_shared<int>(100);
    std::cout << "Shared Value: " << *shared_ptr << std::endl; // Outputs: Shared Value: 100
    return 0;
}
```

模板通过允许编译器在模板实例化期间检查类型来确保类型安全性，从而减少运行时错误：

```cpp
template <typename T>
T add(T a, T b) {
    return a + b;
}
int main() {
    std::cout << add<int>(5, 3) << std::endl;      // Outputs: 8
    std::cout << add<double>(2.5, 3.5) << std::endl; // Outputs: 6.0
    return 0;
}
```

## 模板的工作原理

C++中的模板不是实际的代码，而是作为代码生成的蓝图。当模板用特定类型实例化时，编译器会生成一个具体的模板实例，其中模板参数被指定的类型所替换。

### 函数模板

函数模板定义了一个函数的模式，该函数可以作用于不同的数据类型：

```cpp
template <typename T>
T add(T a, T b) {
    return a + b;
}
int main() {
    std::cout << add<int>(5, 3) << std::endl;      // Outputs: 8
    std::cout << add<double>(2.5, 3.5) << std::endl; // Outputs: 6.0
    return 0;
}
```

模板实例化后实际生成的函数可能如下所示（取决于编译器）：

```cpp
int addInt(int a, int b) {
    return a + b;
}
double addDouble(double a, double b) {
    return a + b;
}
```

### 类模板

类模板定义了一个可以作用于不同数据类型的类的模式：

```cpp
template <typename T>
class Box {
private:
    T content;
public:
    void set_content(const T& value) {
        content = value;
    }
    T get_content() const {
        return content;
    }
};
int main() {
    Box<int> intBox;
    intBox.set_content(123);
    std::cout << intBox.get_content() << std::endl; // Outputs: 123
    Box<std::string> stringBox;
    stringBox.set_content("Hello Templates!");
    std::cout << stringBox.get_content() << std::endl; // Outputs: Hello Templates!
    return 0;
}
```

模板实例化后实际生成的类可能如下所示（取决于编译器）：

```cpp
class BoxInt { /*Box<int>*/ };
class BoxString { /*Box<int>*/ };
```

# 模板的实例化方式

当模板与特定类型一起使用时，编译器会创建一个新实例的模板，其中指定的类型替换了模板参数。这个过程被称为**模板实例化**，可以隐式或显式地发生：

+   **隐式实例化**：这发生在编译器遇到使用特定类型的模板时：

    ```cpp
    int main() {
    ```

    ```cpp
        std::cout << add(5, 3) << std::endl; // The compiler infers the type as int
    ```

    ```cpp
        return 0;
    ```

    ```cpp
    }
    ```

+   **显式实例化**：程序员明确指定类型：

    ```cpp
    int main() {
    ```

    ```cpp
        std::cout << add<int>(5, 3) << std::endl; // Explicitly specifies the type as int
    ```

    ```cpp
        return 0;
    ```

    ```cpp
    }
    ```

# C++ 中模板使用的真实世界示例

在金融软件领域，以灵活、类型安全和高效的方式处理各种类型的资产和货币至关重要。C++ 模板提供了一种强大的机制，通过允许开发者编写通用和可重用的代码，这些代码可以与任何数据类型一起操作。

想象一下开发一个必须处理多种货币（如 USD 和 EUR）以及管理各种资产（如股票或债券）的金融系统。通过使用模板，我们可以定义操作这些类型的通用类，而无需为每种特定货币或资产类型重复代码。这种方法不仅减少了冗余，还增强了系统的可扩展性和可维护性。

在以下章节中，我们将详细探讨使用 C++ 模板实现的金融系统示例。这个示例将向您展示如何定义和操作不同货币的价格，如何创建和管理资产，以及如何确保操作保持类型安全和高效。通过这个示例，我们旨在说明在现实世界的 C++ 应用中使用模板的实际好处，以及它们如何导致代码更加清晰、易于维护和更健壮。

## 定义货币

在设计金融系统时，处理多种货币的方式必须防止错误并确保类型安全性。让我们首先定义需求并探讨各种设计选项。

这里是需求：

+   **类型安全性**：确保不同货币不会意外混合

+   **可扩展性**：轻松添加新货币而无需大量代码重复

+   **灵活性**：以类型安全的方式支持对价格进行加法和减法等操作

这里是设计选项：

+   `int` 或 `double`。然而，这种方法有显著的缺点。它允许意外混合不同的货币，导致计算错误：

    ```cpp
    double usd = 100.0;
    ```

    ```cpp
    double eur = 90.0;
    ```

    ```cpp
    double total = usd + eur; // Incorrectly adds USD and EUR
    ```

    这种方法容易出错且缺乏类型安全性。请注意，由于浮点运算中的精度问题，通常不建议使用 `double` 来表示货币值。

+   `Currency` 类并从中继承特定的货币。虽然这种方法引入了一些结构，但它仍然允许混合不同的货币，并且需要大量努力来实现每种新货币：

    ```cpp
    class Currency {
    ```

    ```cpp
    public:
    ```

    ```cpp
        virtual std::string name() const = 0;
    ```

    ```cpp
        virtual ~Currency() = default;
    ```

    ```cpp
    };
    ```

    ```cpp
    class USD : public Currency {
    ```

    ```cpp
    public:
    ```

    ```cpp
        std::string name() const override { return "USD"; }
    ```

    ```cpp
    };
    ```

    ```cpp
    class Euro : public Currency {
    ```

    ```cpp
    public:
    ```

    ```cpp
        std::string name() const override { return "EUR"; }
    ```

    ```cpp
    };
    ```

    ```cpp
    // USD and Euro can still be mixed inadvertently
    ```

+   `struct`，并且操作是通过模板实现的：

    ```cpp
    struct Usd {
    ```

    ```cpp
        static const std::string &name() {
    ```

    ```cpp
            static std::string name = "USD";
    ```

    ```cpp
            return name;
    ```

    ```cpp
        }
    ```

    ```cpp
    };
    ```

    ```cpp
    struct Euro {
    ```

    ```cpp
        static const std::string &name() {
    ```

    ```cpp
            static std::string name = "EUR";
    ```

    ```cpp
            return name;
    ```

    ```cpp
        }
    ```

    ```cpp
    };
    ```

    ```cpp
    template <typename Currency>
    ```

    ```cpp
    class Price {
    ```

    ```cpp
    public:
    ```

    ```cpp
        Price(int64_t amount) : _amount(amount) {}
    ```

    ```cpp
        int64_t count() const { return _amount; }
    ```

    ```cpp
    private:
    ```

    ```cpp
        int64_t _amount;
    ```

    ```cpp
    };
    ```

    ```cpp
    template <typename Currency>
    ```

    ```cpp
    std::ostream &operator<<(std::ostream &os, const Price<Currency> &price) {
    ```

    ```cpp
        os << price.count() << " " << Currency::name();
    ```

    ```cpp
        return os;
    ```

    ```cpp
    }
    ```

    ```cpp
    template <typename Currency>
    ```

    ```cpp
    Price<Currency> operator+(const Price<Currency> &lhs, const Price<Currency> &rhs) {
    ```

    ```cpp
        return Price<Currency>(lhs.count() + rhs.count());
    ```

    ```cpp
    }
    ```

    ```cpp
    template <typename Currency>
    ```

    ```cpp
    Price<Currency> operator-(const Price<Currency> &lhs, const Price<Currency> &rhs) {
    ```

    ```cpp
        return Price<Currency>(lhs.count() - rhs.count());
    ```

    ```cpp
    }
    ```

    ```cpp
    // User can define other arithmetic operations as needed
    ```

    基于模板的这种方法确保不同货币的价格不能混合：

    ```cpp
    int main() {
    ```

    ```cpp
        Price<Usd> usd(100);
    ```

    ```cpp
        Price<Euro> euro(90);
    ```

    ```cpp
        // The following line would cause a compile-time error
    ```

    ```cpp
        // source>:113:27: error: no match for 'operator+' (operand types are 'Price<Usd>' and 'Price<Euro>')
    ```

    ```cpp
        // Price<Usd> total= usd + euro;
    ```

    ```cpp
        Price<Usd> total = usd+ Price<Usd>(50); // Correct usage
    ```

    ```cpp
        std::cout << total<< std::endl; // Outputs: 150 USD
    ```

    ```cpp
        return 0;
    ```

    ```cpp
    }
    ```

## 定义资产

接下来，我们定义可以以不同货币计价的资产。使用模板，我们可以确保每个资产都与正确的货币相关联：

```cpp
template <typename TickerT>
class Asset;
struct Apple {
    static const std::string &name() {
        static std::string name = "AAPL";
        return name;
    }
    static const std::string &exchange() {
        static std::string exchange = "NASDAQ";
        return exchange;
    }
    using Asset = class Asset<Apple>;
    using Currency = Usd;
};
struct Mercedes {
    static const std::string &name() {
        static std::string name = "MGB";
        return name;
    }
    static const std::string &exchange() {
        static std::string exchange = "FRA";
        return exchange;
    }
    using Asset = class Asset<Mercedes>;
    using Currency = Euro;
};
template <typename TickerT>
class Asset {
public:
    using Ticker   = TickerT;
    using Currency = typename Ticker::Currency;
    Asset(int64_t amount, Price<Currency> price)
        : _amount(amount), _price(price) {}
    auto amount() const { return _amount; }
    auto price() const { return _price; }
private:
    int64_t _amount;
    Price<Currency> _price;
};
template <typename TickerT>
std::ostream &operator<<(std::ostream &os, const Asset<TickerT> &asset) {
    os << TickerT::name() << ", amount: " << asset.amount()
       << ", price: " << asset.price();
    return os;
}
```

## 使用金融系统

最后，我们演示如何使用定义的模板来管理资产和价格：

```cpp
int main() {
    Price<Usd> usd_price(100);
    usd_price = usd_price + Price<Usd>(1);
    std::cout << usd_price << std::endl; // Outputs: 101 USD
    Asset<Apple> apple{10, Price<Usd>(100)};
    Asset<Mercedes> mercedes{5, Price<Euro>(100)};
    std::cout << apple << std::endl; // Outputs: AAPL, amount: 10, price: 100 USD
    std::cout << mercedes << std::endl; // Outputs: MGB, amount: 5, price: 100 EUR
    return 0;
}
```

## 使用模板在系统设计中的缺点

虽然 C++中的模板提供了一种强大且灵活的方式来创建类型安全的通用组件，但这种方法有几个缺点。这些缺点在处理多种货币和资产的金融系统背景下尤其相关。在决定在设计中使用模板时，了解这些潜在的缺点是至关重要的。

### 代码膨胀

模板可能导致代码膨胀，这是由于生成多个模板实例化而导致的二进制文件大小增加。编译器为每个唯一的类型实例化生成模板代码的单独版本。在一个支持各种货币和资产的金融系统中，这可能导致编译的二进制文件大小显著增加。

例如，如果我们为 `Price` 和 `Asset` 实例化了不同的类型，如 `Usd`、`Euro`、`Apple` 和 `Mercedes`，编译器将为每个组合生成单独的代码：

```cpp
Price<Usd> usdPrice(100);
Price<Euro> euroPrice(90);
Asset<Apple> appleAsset(10, Price<Usd>(100));
Asset<Mercedes> mercedesAsset(5, Price<Euro>(100));
```

每个实例化都会产生额外的代码，从而增加整体二进制文件的大小。随着支持的货币和资产数量的增加，代码膨胀的影响变得更加明显。二进制文件大小会影响应用程序的性能、内存使用和加载时间，尤其是在资源受限的环境中，这主要是由于缓存效率较低。

### 编译时间增加

模板可以显著增加项目的编译时间。每次模板与新类型的实例化都会导致编译器生成新的代码。在一个支持数百种货币和来自不同国家和证券交易所的资产的金融系统中，编译器必须实例化所有需要的组合，从而导致构建时间更长。

例如，假设我们的系统支持以下内容：

+   50 种不同的货币

+   来自各种证券交易所的 10000 种不同的资产类型

然后，编译器将为每个 `Price` 和 `Asset` 的组合生成代码，导致大量的模板实例化。这可能会显著减慢编译过程，影响开发工作流程，并降低反馈循环的效率。

### 与其他代码的交互不太明显

模板代码可能很复杂，在与其他代码库的交互方面不太明显。对模板不太熟悉的开发者可能会发现理解和维护模板密集型代码具有挑战性。语法可能很冗长，编译器错误信息可能难以理解，这使得调试和故障排除变得更加复杂。

例如，模板参数中的简单错误可能导致令人困惑的错误信息：

```cpp
template <typename T>
class Price {
    // Implementation
};
Price<int> price(100); // Intended to be Price<Usd> but mistakenly used int
```

在这种情况下，开发者必须理解模板和编译器生成的特定错误信息，以解决问题。这可能成为经验不足的开发者的障碍。

C++ 20 提供了概念来改进模板错误消息和约束，这可以帮助使模板代码更易于阅读和理解。我们可以创建一个名为 `BaseCurrency` 的基类，并从它派生所有货币类。这样，我们可以确保所有货币类都有一个共同的接口，并且可以互换使用：

```cpp
struct BaseCurrency {
};
struct Usd : public BaseCurrency {
    static const std::string &name() {
        static std::string name = "USD";
        return name;
    }
};
// Define a concept for currency classes
template<class T, class U>
concept Derived = std::is_base_of<U, T>::value;
// Make sure that template parameter is derived from BaseCurrency
template <Derived<BaseCurrency> CurrencyT>
class Price {
public:
    Price(int64_t amount) : _amount(amount) {}
    int64_t count() const { return _amount; }
private:
    int64_t _amount;
};
```

在这些更改之后，尝试实例化 `Price<int>` 将导致编译时错误，从而清楚地表明类型必须从 `BaseCurrency` 派生：

```cpp
In function 'int main()':
error: template constraint failure for 'template<class CurrencyT>  requires  Derived<CurrencyT, Currency> class Price'
 auto p = Price<int>(100);
                   ^
note: constraints not satisfied
In substitution of 'template<class CurrencyT>  requires  Derived<CurrencyT, Currency> class Price [with CurrencyT = int]':
```

C++ 20 之前的版本也提供了一种方法，通过使用 `std::enable_if` 和 `std::is_base_of` 的组合来强制模板参数的约束，从而防止意外的模板实例化：

```cpp
template <typename CurrencyT,
          typename Unused=typename std::enable_if<std::is_base_of<BaseCurrency,CurrencyT>::value>::type>
class Price {
public:
    Price(int64_t amount) : _amount(amount) {}
    int64_t count() const { return _amount; }
private:
    int64_t _amount;
};
```

现在尝试初始化 `Price<int>` 将导致编译时错误，表明类型必须从 `BaseCurrency` 派生，然而，错误信息将有点晦涩难懂：

```cpp
error: no type named 'type' in 'struct std::enable_if<false, void>'
auto p = Price<int>(100);
      |                       ^
error: template argument 2 is invalid
```

### 工具支持有限和调试

调试模板代码可能具有挑战性，因为工具支持有限。许多调试器处理模板实例化不佳，使得难以逐步执行模板代码并检查模板参数和实例化。这可能会阻碍调试过程，并使识别和修复问题变得更加困难。

例如，在调试器中检查模板化的 `Price<Usd>` 对象的状态可能无法提供对底层类型和值的清晰洞察，尤其是如果调试器不完全支持模板参数检查。

大多数自动完成和 IDE 工具与模板配合得不是很好，因为它们无法假设模板参数的类型。这可能会使导航和理解模板密集型代码库变得更加困难。

### 模板的高级特性可能难以使用

C++ 中的模板提供了编写通用和可重用代码的机制。然而，在某些情况下，需要针对特定类型自定义默认模板行为。这就是模板特化的用武之地。模板特化允许你为特定类型定义特殊行为，确保模板对该类型的行为正确。

#### 为什么使用模板特化？

当通用模板实现对于特定类型不正确或不高效，或者特定类型需要完全不同的实现时，会使用模板特化。这可能是由于各种原因，例如性能优化、对某些数据类型的特殊处理，或符合特定要求。

例如，考虑一个场景，你有一个通用的 `Printer` 模板类，它可以打印任何类型的对象。然而，对于 `std::string`，你可能希望在打印时在字符串周围添加引号。

#### 基本模板特化示例

下面是一个模板特化工作方式的示例：

```cpp
#include <iostream>
#include <string>
// General template
template <typename T>
class Printer {
public:
    void print(const T& value) {
        std::cout << value << std::endl;
    }
};
// Template specialization for std::string
template <>
class Printer<std::string> {
public:
    void print(const std::string& value) {
        std::cout << "\"" << value << "\"" << std::endl;
    }
};
int main() {
    Printer<int> int_printer;
    int_printer.print(123); // Outputs: 123
    Printer<std::string> string_printer;
    string_printer.print("Hello, World!"); // Outputs: "Hello, World!" with quotes
    return 0;
}
```

在这个示例中，通用的`Printer`模板类可以打印任何类型。然而，对于`std::string`，特化版本在打印字符串时会添加引号。

#### 包含特化头文件

在使用模板特化时，包含包含特化定义的头文件至关重要。如果没有包含特化头文件，编译器将实例化模板的默认版本，从而导致行为不正确。

例如，考虑以下文件：

`printer.h`（通用模板定义）：

```cpp
#ifndef PRINTER_H
#define PRINTER_H
#include <iostream>
template <typename T>
class Printer {
public:
    void print(const T& value) {
        std::cout << value << std::endl;
    }
};
#endif // PRINTER_H
```

`printer_string.h`（针对`std::string`的模板特化）：

```cpp
#ifndef PRINTER_STRING_H
#define PRINTER_STRING_H
#include "printer.h"
#include <string>
template <>
class Printer<std::string> {
public:
    void print(const std::string& value) {
        std::cout << "\"" << value << "\"" << std::endl;
    }
};
#endif // PRINTER_STRING_H
```

`main.cpp`（使用模板和特化）：

```cpp
#include "printer.h"
// #include "printer_string.h" // Uncomment this line to use the specialization
int main() {
    Printer<int> int_printer;
    int_printer.print(123); // Outputs: 123
    Printer<std::string> string_printer;
    string_printer.print("Hello, World!"); // Outputs: Hello, World! without quotes if the header is not included
    return 0;
}
```

在这个配置中，如果`main.cpp`中没有包含`printer_string.h`头文件，编译器将使用默认的`Printer`模板为`std::string`，从而导致行为不正确（打印字符串时不加引号）。

模板是 C++编程语言的重要组成部分，提供了创建通用、可重用和类型安全的代码的强大功能。在各种场景中都是必不可少的，例如开发通用算法、容器类、智能指针和其他需要与多种数据类型无缝工作的实用工具。模板使开发者能够编写灵活且高效的代码，确保相同的功能可以应用于不同的类型而无需重复。

然而，模板的强大功能并非没有代价。模板的使用可能导致编译时间增加和代码膨胀，尤其是在支持广泛类型和组合的系统中。语法和产生的错误信息可能很复杂，难以理解，这对经验不足的开发者来说是一个挑战。此外，由于工具支持有限和模板实例化的复杂性质，调试模板密集型代码可能很繁琐。

此外，模板可能会引入与代码库的其他部分不太明显的交互，如果不妥善管理，可能会引起问题。开发者还必须意识到需要谨慎包含特化头文件的高级特性，如模板特化，以避免不正确的行为。

考虑到这些注意事项，开发者在将模板纳入其项目之前必须仔细思考。虽然它们提供了显著的好处，但潜在的缺点需要深思熟虑的方法，以确保优势超过复杂性。正确理解和审慎使用模板可以导致更健壮、可维护和高效的 C++应用程序。

# 概述

在本章中，我们探讨了高级 C++编程的复杂性，重点关注类设计、继承和模板。我们首先介绍了有效类设计的原则，强调封装最小必要功能和数据以实现更好的模块化和可维护性的重要性。通过实际示例，我们突出了良好的和不良的设计实践。接着转向继承，我们探讨了它的好处，如代码重用、层次结构化和多态性，同时也指出了其缺点，包括紧密耦合、复杂的层次结构和可能违反 LSP（里氏替换原则）的风险。我们提供了何时使用继承以及何时考虑替代方案如组合的建议。在模板部分，我们深入探讨了它们在启用泛型编程中的作用，允许灵活且可重用的组件与任何数据类型一起工作。我们讨论了模板的优势，如代码重用性、类型安全和性能优化，但也指出了它们的缺点，包括编译时间增加、代码膨胀以及理解和调试模板密集型代码的复杂性。在这些讨论中，我们强调了在利用这些强大功能时进行仔细考虑和理解的需要，以确保构建健壮且可维护的 C++应用程序。在下一章中，我们将把重点转向 API 设计，探讨在 C++中创建清晰、高效和用户友好界面的最佳实践。
