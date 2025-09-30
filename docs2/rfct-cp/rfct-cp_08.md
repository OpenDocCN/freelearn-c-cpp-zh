

# 在 C++中设计和开发 API

在软件开发的世界里，**应用程序编程接口**（API）的设计至关重要。好的 API 是软件库的骨架，促进不同软件组件之间的交互，使开发者能够高效有效地利用功能。设计良好的 API 直观、易用且可维护，在软件项目的成功和持久性中扮演着关键角色。在本章中，我们将深入探讨为在 C++中开发的库设计可维护 API 的原则和实践。我们将探讨 API 设计的要点，包括清晰性、一致性和可扩展性，并提供具体示例来说明最佳实践。通过理解和应用这些原则，您将能够创建不仅满足用户当前需求，而且随着时间的推移保持稳健和适应性强的 API，确保您的库既强大又用户友好。

# 简约 API 设计原则

简约 API 旨在提供执行特定任务所需的必要功能，避免不必要的特性和复杂性。主要目标是提供一个干净、高效且用户友好的界面，便于轻松集成和使用。简约 API 的关键优势包括以下内容：

+   **易用性**：用户可以快速理解和利用 API，无需进行广泛的学习或查阅文档，从而促进更快的开发周期

+   **可维护性**：简化的 API 更容易维护，允许进行简单的更新和错误修复，而不会引入新的复杂性

+   **性能**：由于减少了开销和更高效的执行路径，更轻量级的 API 往往具有更好的性能

+   **可靠性**：由于组件和交互较少，错误和意外问题的可能性最小化，从而使得软件更加可靠和稳定

简洁和清晰是设计简约 API 的基本原则。这些原则确保 API 保持可访问性和用户友好性，从而提升整体开发体验。简洁和清晰的关键方面包括以下内容：

+   **直观界面**：设计简单明了的界面有助于开发者快速掌握可用功能，使其更容易集成并有效使用 API

+   **降低认知负荷**：通过最小化理解和使用 API 所需的脑力劳动，开发者犯错误的可能性降低，从而提高开发过程的效率

+   **直观设计**：遵循简洁和清晰的 API 与常见的使用模式和开发者期望紧密一致，使其更加直观且易于采用

过度设计和不必要的复杂性会严重削弱 API 的有效性。为了避免这些陷阱，请考虑以下策略：

+   **关注核心功能**：专注于提供解决主要用例的基本功能。避免添加与 API 核心目的不直接相关的额外功能。

+   **迭代设计**：从**最小可行产品（MVP）**开始，并根据用户反馈和实际需求逐步添加功能，而不是基于推测性需求。 

+   **清晰的文档**：提供全面而简洁的文档，重点关注核心功能和常见用例。这有助于防止混淆和误用。

+   **一致的命名约定**：为函数、类和参数使用一致且描述性的名称，以增强清晰性和可预测性。

+   **最小依赖性**：减少外部依赖项的数量以简化集成过程并最小化潜在的兼容性问题。

# 实现极简主义的技术

功能分解是将复杂功能分解成更小、更易于管理的单元的过程。这项技术对于创建极简 API 至关重要，因为它促进了简单性和模块化。通过分解函数，你确保 API 的每个部分都有一个清晰、明确的目的，这增强了可维护性和可用性。

功能分解的关键方面包括以下内容：

+   **模块化设计**：设计 API，使每个模块或函数处理整体功能的一个特定方面。这种**关注点分离（SoC）**确保 API 的每个部分都有一个清晰、明确的目的。

+   **单一职责原则（SRP）**：每个函数或类应该只有一个，并且只有一个，改变的理由。这一原则有助于保持 API 简单并专注于目标。

+   **可重用组件**：通过将函数分解成更小的单元，可以创建可重用组件，这些组件可以以不同的方式组合来实现各种任务，从而增强 API 的灵活性和可重用性。

接口分离旨在保持接口精简并专注于特定任务，避免设计试图覆盖过多用例的单一接口。这一原则确保客户端只需了解与他们相关的方 法，使 API 更容易使用和理解。

接口分离的关键方面包括以下内容：

+   **特定接口**：而不是一个大型、通用接口，设计多个较小、特定的接口。每个接口应针对功能的一个特定方面。

+   **以用户为中心的设计**：考虑 API 的最终用户的需要。设计直观的接口，只提供他们完成任务所需的方法，避免不必要的复杂性。

+   **减少客户端影响**：较小的、专注的接口在需要更改时对客户端的影响最小。使用特定接口的客户端不太可能受到无关功能更改的影响。

让我们考虑一个例子，其中复杂的 API 类负责各种功能，如加载、处理和保存数据：

```cpp
class ComplexAPI {
public:
    void initialize();
    void load_data_from_file(const std::string& filePath);
    void load_data_from_database(const std::string& connection_string);
    void process_data(int mode);
    void save_data_to_file(const std::string& filePath);
    void save_data_to_database(const std::string& connection_string);
    void cleanup();
};
```

主要问题是该类承担了过多的责任，混合了不同的数据源和目的地，导致复杂性和缺乏专注。让我们从将加载和处理功能提取到单独的类开始：

```cpp
class FileDataLoader {
public:
    explicit FileDataLoader(const std::string& filePath) : filePath(filePath) {}
    void load() {
        // Code to load data from a file
    }
private:
    std::string filePath;
};
class DatabaseDataLoader {
public:
    explicit DatabaseDataLoader(const std::string& connection_string) : _connection_string(connection_string) {}
    void load() {
        // Code to load data from a database
    }
private:
    std::string _connection_string;
};
class DataProcessor {
public:
    void process(int mode) {
        // Code to process data based on the mode
    }
};
```

下一步是将保存功能提取到单独的类中：

```cpp
class DataSaver {
public:
    virtual void save() = 0;
    virtual ~DataSaver() = default;
};
class FileDataSaver : public DataSaver {
public:
    explicit FileDataSaver(const std::string& filePath) : filePath(filePath) {}
    void save() override {
        // Code to save data to a file
    }
private:
    std::string filePath;
};
class DatabaseDataSaver : public DataSaver {
public:
    explicit DatabaseDataSaver(const std::string& connection_string) : _connection_string(connection_string) {}
    void save() override {
        // Code to save data to a database
    }
private:
    std::string _connection_string;
};
```

最小化 API 所需的依赖数量对于实现简约至关重要。更少的依赖导致 API 更稳定、可靠和易于维护。依赖关系可能会复杂化集成，增加兼容性问题风险，并使 API 更难理解。

减少依赖的关键策略包括以下内容：

+   **核心功能重点**：专注于在 API 内部实现核心功能，除非绝对必要，否则避免依赖外部库或组件。

+   **选择性使用库**：当需要外部库时，选择那些稳定、维护良好且广泛使用的库。确保它们与 API 的需求紧密一致。

+   **解耦设计**：尽可能设计 API 使其能够独立于外部组件运行。使用**依赖注入**（**DI**）或其他设计模式将实现与特定依赖解耦。

+   **版本管理**：仔细管理和指定任何依赖的版本，以避免兼容性问题。确保依赖的更新不会破坏 API 或引入不稳定性。

# 简约 API 设计的现实世界示例

为了巩固我们对这些概念的理解，我们将检查几个 C++中 API 设计的现实世界示例。这些示例将突出常见挑战和有效解决方案，展示如何在实际场景中应用良好的 API 设计原则。通过这些示例，我们旨在提供清晰、可操作的见解，您可以将它们应用于自己的项目，确保您的 API 不仅功能齐全，而且优雅且易于维护。让我们深入了解现实世界 API 设计的复杂性，并看看这些原则如何在实践中发挥作用：

+   **现代 C++的 JSON（nlohmann/json）**：这个库是简约 API 设计的优秀示例。它提供了直观且直接的方法来解析、序列化和操作 C++中的 JSON 数据，并具有以下优点：

    +   **简洁性**：清晰简洁的界面，易于使用。

    +   **功能分解**：每个函数处理与 JSON 处理相关的特定任务。

    +   **最小依赖**：设计为与 C++标准库一起工作，避免不必要的外部依赖：

        ```cpp
        #include <nlohmann/json.hpp>
        ```

        ```cpp
        nlohmann::json j = {
        ```

        ```cpp
            {"pi", 3.141},
        ```

        ```cpp
            {"happy", true},
        ```

        ```cpp
            {"name", "Niels"},
        ```

        ```cpp
            {"nothing", nullptr},
        ```

        ```cpp
            {"answer", {
        ```

        ```cpp
                {"everything", 42}
        ```

        ```cpp
            }},
        ```

        ```cpp
            {"list", {1, 0, 2}},
        ```

        ```cpp
            {"object", {
        ```

        ```cpp
                {"currency", "USD"},
        ```

        ```cpp
                {"value", 42.99}
        ```

        ```cpp
            }}
        ```

        ```cpp
        };
        ```

+   **SQLite C++接口（SQLiteCpp）**：这个库为使用 C++与 SQLite 数据库交互提供了一个简约的接口。它有以下优点：

    +   **简单性**：提供直观且清晰的数据库操作 API。

    +   **接口隔离**：为不同的数据库操作（如查询和事务）创建不同的类。

    +   **最小依赖性**：构建用于 SQLite 和 C++ 标准库：

        ```cpp
        #include <SQLiteCpp/SQLiteCpp.h>
        ```

        ```cpp
        SQLite::Database db("test.db", SQLite::OPEN_READWRITE|SQLite::OPEN_CREATE);
        ```

        ```cpp
        db.exec("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)");
        ```

        ```cpp
        SQLite::Statement query(db, "INSERT INTO test (value) VALUES (?)");
        ```

        ```cpp
        query.bind(1, "Sample value");
        ```

        ```cpp
        query.exec();
        ```

# 常见陷阱及其避免方法

当 API 设计包含不必要的功能或复杂性时，会发生过度复杂化，使其难以使用和维护。以下是减轻这种情况的方法：

+   **避免策略**：关注最终用户所需的核心功能。定期审查 API 设计，以消除任何不必要的功能。

功能蔓延发生在不断向 API 添加额外功能时，导致复杂性增加和可用性降低。以下是您可以避免这种情况的方法：

+   **避免策略**：实施严格的特性优先级排序流程。确保新特性与 API 的核心目的相一致，并且对于目标用户来说是必要的。

# 在 C++ 中开发共享库的重要注意事项

在 C++ 中开发共享库需要仔细考虑以确保兼容性、稳定性和可用性。最初，共享库旨在促进代码重用、模块化和高效内存使用，允许多个程序同时使用相同的库代码。这种方法预计可以减少冗余、节省系统资源，并能够仅替换应用程序的部分。虽然这种方法对于广泛使用的库（如 `libc`、`libstdc++`、OpenSSL 等）效果良好，但它对于应用程序来说效率较低。与应用程序一起提供的共享库很少能够完美地替换为较新版本。通常，需要替换整个安装套件，包括应用程序及其所有依赖项。

现在，共享库通常用于实现不同编程语言之间的互操作性。例如，C++ 库可能被用于用 Java 或 Python 编写的应用程序中。这种跨语言功能扩展了库的可用性和范围，但同时也引入了某些复杂性和注意事项，开发者必须考虑。

## 单个项目内的共享库

如果共享库设计为在单个项目中使用，并且由使用相同编译器编译的可执行文件加载，那么具有 C++ 接口的共享对象（或 DLL）通常是可接受的。然而，这种方法存在一些注意事项，例如单例的使用，这可能导致多线程问题和意外的初始化顺序。当使用单例时，在多线程环境中管理它们的初始化和销毁可能具有挑战性，可能导致潜在的竞争条件和不可预测的行为。此外，确保全局状态初始化和销毁的正确顺序很复杂，这可能导致微妙且难以诊断的错误。

## 用于更广泛分发的共享库

如果预期共享库将被更广泛地分发，开发者无法预测最终用户使用的编译器，或者如果库可能被用于其他编程语言，那么 C++ 共享库并不是一个理想的选择。这主要是因为 C++ 的 `libc` 或操作系统系统调用，它们也在 C 中。解决这个问题的常见方法是在 C++ 代码周围开发一个 C 封装器，并附带 C 接口。

## 示例 - `MessageSender` 类

下面的示例展示了这种方法，其中我们创建了一个 C++ 的 `MessageSender` 类，并为它提供了一个 C 封装器。该类有一个构造函数，用于使用指定的接收者初始化 `MessageSender` 实例，并且有两个重载的 `send` 方法，允许以 `std::vector<uint8_t>` 实例或指定长度的原始指针的形式发送消息。实现将消息打印到控制台以展示功能。

下面是 C++ 库的实现：

```cpp
// MessageSender.hpp
#pragma once
#include <string>
#include <vector>
class MessageSender {
public:
    MessageSender(const std::string& receiver);
    void send(const std::vector<uint8_t>& message) const;
    void send(const uint8_t* message, size_t length) const;
};
// MessageSender.cpp
#include "MessageSender.h"
#include <iostream>
MessageSender::MessageSender(const std::string& receiver) {
    std::cout << "MessageSender created for receiver: " << receiver << std::endl;
}
void MessageSender::send(const std::vector<uint8_t>& message) const {
    std::cout << "Sending message of size: " << message.size() << std::endl;
}
void MessageSender::send(const uint8_t* message, size_t length) const {
    std::cout << "Sending message of length: " << length << std::endl;
}
```

下面是 C 封装器的实现：

```cpp
// MessageSender.h (C Wrapper Header)
#ifdef __cplusplus
extern "C" {
#endif
typedef void* MessageSenderHandle;
MessageSenderHandle create_message_sender(const char* receiver);
void destroy_message_sender(MessageSenderHandle handle);
void send_message(MessageSenderHandle handle, const uint8_t* message, size_t length);
#ifdef __cplusplus
}
#endif
// MessageSenderC.cpp (C Wrapper Implementation)
#include "MessageSenderC.h"
#include "MessageSender.hpp"
MessageSenderHandle create_message_sender(const char* receiver) {
    return new(std::nothrow) MessageSender(receiver);
}
void destroy_message_sender(MessageSenderHandle handle) {
    MessageSender* instance = reinterpret_cast<MessageSender*>(handle);
    assert(instance);
    delete instance;
}
void send_message(MessageSenderHandle handle, const uint8_t* message, size_t length) {
    MessageSender* instance = reinterpret_cast<MessageSender*>(handle);
    assert(instance);
    instance->send(message, length);
}
```

在这个示例中，C++ 的 `MessageSender` 类定义在 `MessageSender.hpp` 和 `MessageSender.cpp` 文件中。该类有一个构造函数，用于使用指定的接收者初始化 `MessageSender` 实例，并且有两个重载的 `send` 方法，允许以 `std::vector<uint8_t>` 实例或指定长度的原始指针的形式发送消息。实现将消息打印到控制台以展示功能。

为了使这个 C++ 类可以从其他编程语言或不同的编译器中使用，我们创建了一个 C 封装器。C 封装器定义在 `MessageSender.h` 和 `MessageSenderC.cpp` 文件中。头文件使用 `extern "C"` 块来确保 C++ 函数可以从 C 中调用，防止名称修饰。C 封装器使用不透明的句柄 `void*`（定义为 `MessageSenderHandle`），在 C 中表示 `MessageSender` 实例，抽象了实际的 C++ 类。

`create_message_sender` 函数分配并初始化一个 `MessageSender` 实例，并返回其句柄。请注意，它使用 `new(std::nothrow)` 以避免在内存分配失败时抛出异常。即使 C 或其他不支持异常的编程语言也可以无问题地使用此函数。

`destroy_message_sender` 函数释放 `MessageSender` 实例，以确保正确清理。`send_message` 函数使用句柄调用 `MessageSender` 实例上的相应 `send` 方法，从而简化消息发送过程。

通过在同一个二进制文件内处理内存分配和释放，这种方法避免了与最终用户使用不同内存分配器相关的问题，这些问题可能导致内存损坏或泄漏。C 包装器提供了一个稳定且一致的接口，可以在不同的编译器和语言中使用，确保更高的兼容性和稳定性。这种方法解决了开发共享库的复杂性，并确保它们的广泛可用性和可靠性。

如果预计 C++库会抛出异常，那么在 C 包装器函数中正确处理这些异常是很重要的，以防止异常传播到调用者。例如，我们可以有以下异常类型：

```cpp
class ConnectionError : public std::runtime_error {
public:
    ConnectionError(const std::string& message) : std::runtime_error(message) {}
};
class SendError : public std::runtime_error {
public:
    SendError(const std::string& message) : std::runtime_error(message) {}
};
```

然后，C 包装器函数可以捕获这些异常，并向调用者返回适当的错误代码或消息：

```cpp
// MessageSender.h (C Wrapper Header)
typedef enum {
    OK,
    CONNECTION_ERROR,
    SEND_ERROR,
} MessageSenderStatus;
// MessageSenderC.cpp (C Wrapper Implementation)
MessageSenderStatus send_message(MessageSenderHandle handle, const uint8_t* message, size_t length) {
    try {
        MessageSender* instance = reinterpret_cast<MessageSender*>(handle);
        instance->send(message, length);
        return OK;
    } catch (const ConnectionError&) {
        return CONNECTION_ERROR;
    } catch (const SendError&) {
        return SEND_ERROR;
    } catch (...) {
       std::abort();
    }
}
```

注意，在遇到未知异常时，我们使用`std::abort`，因为将未知异常传播到语言边界是不安全的。

这个例子说明了如何创建一个 C 包装器来确保在开发共享库时的兼容性和稳定性。遵循这些指南，开发者可以创建健壮、可维护且广泛兼容的共享库，确保它们在各种平台和编程环境中的可用性。

# 摘要

在本章中，我们探讨了设计和开发 C++共享库的关键方面。共享库最初是为了促进代码重用、模块化和高效内存使用而设计的，允许多个程序同时利用相同的库代码。这种方法减少了冗余并节省了系统资源。

我们深入探讨了在不同上下文中开发共享库的细微差别。当共享库打算在单个项目中使用并与相同的编译器编译时，具有 C++接口的共享对象（或 DLL）可能是合适的，尽管需要小心处理单例和全局状态，以避免多线程问题和不可预测的初始化顺序。

然而，对于更广泛的分发，如果最终用户的编译器或编程语言可能不同，由于 C++ ABI 在不同编译器和版本之间的不稳定性，直接使用 C++共享库就不太可取。为了克服这一点，我们讨论了在 C++代码周围创建 C 包装器，利用稳定的 C ABI 以实现更广泛的兼容性和跨语言功能。

我们提供了一个使用`MessageSender`类的综合示例，说明了如何创建 C++库及其相应的 C 包装器。示例强调了通过确保在同一个二进制文件内进行分配和释放以及通过在 C 接口中以枚举状态表示来优雅地处理异常来安全地管理内存。

通过遵循这些指南，开发者可以创建健壮、可维护且广泛兼容的共享库，确保它们在各种平台和编程环境中的可用性。本章为开发者提供了解决常见问题并在共享库开发中实施最佳实践所需的知识，从而培养出有效且可靠的软件解决方案。

在下一章中，我们将把我们的重点转向代码格式化，探讨创建清晰、一致和可读代码的最佳实践，这对于协作和长期维护至关重要。
