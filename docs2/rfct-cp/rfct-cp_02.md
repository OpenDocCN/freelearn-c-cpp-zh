

# 主要软件开发原则

在本章中，我们将探讨用于创建结构良好且易于维护的代码的主要软件设计原则。其中最重要的原则之一是 SOLID 原则，它代表单一职责原则、开闭原则、里氏替换原则、接口隔离原则和依赖倒置原则。这些原则旨在帮助开发者创建易于理解、测试和修改的代码。我们还将讨论抽象层次的重要性，这是将复杂系统分解成更小、更易于管理的部分的做法。此外，我们还将探讨副作用和可变性概念及其如何影响软件的整体质量。通过理解和应用这些原则，开发者可以创建更健壮、更可靠和可扩展的软件。

# SOLID

SOLID 原则是一组原则，最初由 Robert C. Martin 在他的 2000 年出版的《敏捷软件开发：原则、模式和实践》一书中提出。Robert C. Martin，也被称为 Uncle Bob，是一位软件工程师、作家和演讲家。他被认为是在软件开发行业中最具影响力的人物之一，以其对 SOLID 原则的工作和对面向对象编程领域的贡献而闻名。Martin 作为一名软件工程师已有 40 多年的经验，参与过从小型系统到大型企业系统的各种项目。他也是一位知名的演讲家，在世界各地的许多会议和活动中发表了关于软件开发的演讲。他是敏捷方法的倡导者，对敏捷宣言的发展产生了重要影响。SOLID 原则的开发是为了帮助开发者通过促进良好的设计实践来创建更易于维护和可扩展的代码。这些原则基于 Martin 作为软件工程师的经验以及他观察到许多软件项目因设计不佳而难以理解、更改和维护的观察。

SOLID 原则旨在作为面向对象软件设计的指南，并基于软件应易于理解、随时间变化和扩展的想法。这些原则旨在与其他软件开发实践结合使用，例如测试驱动开发和持续集成。遵循 SOLID 原则，开发者可以创建更健壮、更少出现错误且易于长期维护的代码。

## 单一职责原则

**单一职责原则**（**SRP**）是面向对象软件设计的五个 SOLID 原则之一。它指出，一个类应该只有一个改变的理由，这意味着一个类应该只有一个职责。这个原则旨在促进易于理解、更改和测试的代码。

SRP 背后的理念是一个类应该有一个单一、明确的目的。这使得理解类的行为更加容易，并减少了类变更产生意外后果的可能性。当一个类只有一个职责时，它也更不容易出现错误，并且为其编写自动化测试也更加容易。

应用 SRP 可以通过使系统更加模块化和易于理解来提高软件系统的设计。通过遵循这个原则，开发者可以创建小型、专注且易于推理的类。这使得随着时间的推移维护和改进软件变得更加容易。

让我们看看一个支持通过网络发送多种消息类型的消息系统。该系统有一个`Message`类，它接收发送者和接收者 ID 以及要发送的原始数据。此外，它还支持将消息保存到磁盘并通过`send`方法发送自身：

```cpp
class Message {
public:
  Message(SenderId sender_id, ReceiverId receiver_id,
          const RawData& data)
    : sender_id_{sender_id},
      receiver_id_{receiver_id}, raw_data_{data} {}
  SenderId sender_id() const { return sender_id_; }
  ReceiverId receiver_id() const { return receiver_id_; }
  void save(const std::string& file_path) const {
    // serializes a message to raw bytes and saves
    // to file system
  }
  std::string serialize() const {
    // serializes to JSON
    return {"JSON"};
  }
  void send() const {
    auto sender = Communication::get_instance();
    sender.send(sender_id_, receiver_id_, serialize());
  }
private:
  SenderId sender_id_;
  ReceiverId receiver_id_;
  RawData raw_data_;
};
```

`Message`类负责多个关注点，例如从/到文件系统保存消息、序列化数据、发送消息以及持有发送者和接收者 ID 和原始数据。将这些职责分离到不同的类或模块中会更好。

`Message`类只负责存储数据和将其序列化为 JSON 格式：

```cpp
class Message {
public:
  Message(SenderId sender_id, ReceiverId receiver_id,
          const RawData& data)
    : sender_id_{sender_id},
      receiver_id_{receiver_id}, raw_data_{data} {}
  SenderId sender_id() const { return sender_id_; }
  ReceiverId receiver_id() const { return receiver_id_; }
  std::string serialize() const {
    // serializes to JSON
    return {"JSON"};
  }
private:
  SenderId sender_id_;
  ReceiverId receiver_id_;
  RawData raw_data_;
};
```

`save`方法可以被提取到一个单独的`MessageSaver`类中，拥有单一职责：

```cpp
class MessageSaver {
public:
  MessageSaver(const std::string& target_directory);
  void save(const Message& message) const;
};
```

而`send`方法是在一个专门的`MessageSender`类中实现的。这三个类都有单一且明确的责任，并且任何对其中任何一个类的进一步更改都不会影响其他类。这种方法允许在代码库中隔离更改。在需要长时间编译的复杂系统中，这一点变得至关重要。

总结来说，SRP（单一职责原则）指出，一个类应该只有一个改变的理由，这意味着一个类应该只有一个职责。这个原则旨在促进易于理解、更改和测试的代码，并有助于创建更模块化、可维护和可扩展的代码库。遵循这个原则，开发者可以创建小型、专注且易于推理的类。

### SRP 的其他应用

单一职责原则（SRP）不仅适用于类，也适用于更大的组件，如应用程序。在架构层面，SRP 通常实现为微服务架构。微服务的理念是将软件系统构建为一系列小型、独立的服务的集合，这些服务通过网络相互通信，而不是将其构建为一个单体应用程序。每个微服务*负责特定的业务能力，并且可以独立于其他服务进行开发、部署和扩展*。这允许有更大的灵活性、可扩展性和易于维护，因为对一个服务的更改不会影响整个系统。微服务还使开发过程更加敏捷，因为团队可以并行工作在不同的服务上，同时也允许对安全、监控和测试采取更细粒度的方法，因为每个服务都可以单独处理。

## 开放-封闭原则

开放-封闭原则指出，模块或类应该是可扩展的，但应该是封闭的以进行修改。换句话说，应该能够在不修改现有代码的情况下向模块或类添加新功能。这个原则有助于促进软件的可维护性和灵活性。C++中这个原则的一个例子是使用继承和多态。可以编写一个基类，使其能够被派生类扩展，从而在不修改基类的情况下添加新功能。另一个例子是使用接口或抽象类来定义一组相关类的合同，允许添加符合合同的新类，而无需修改现有代码。

开放-封闭原则可以用来改进我们的消息发送组件。当前版本只支持一种消息类型。如果我们想添加更多数据，我们需要修改`Message`类：添加字段，保留一个额外的消息类型变量，而且不用说基于这个变量的序列化。为了避免对现有代码的修改，让我们将`Message`类重写为纯虚拟的，提供`serialize`方法：

```cpp
class Message {
public:
  Message(SenderId sender_id, ReceiverId receiver_id)
    : sender_id_{sender_id}, receiver_id_{receiver_id} {}
  SenderId sender_id() const { return sender_id_; }
  ReceiverId receiver_id() const { return receiver_id_; }
  virtual std::string serialize() const = 0;
private:
  SenderId sender_id_;
  ReceiverId receiver_id_;
};
```

现在，让我们假设我们需要添加另外两种消息类型：一种支持启动延迟的“启动”消息（通常用于调试目的）和一种支持停止延迟的“停止”消息（可用于调度）；它们可以按以下方式实现：

```cpp
class StartMessage : public Message {
public:
  StartMessage(SenderId sender_id, ReceiverId receiver_id,
               std::chrono::milliseconds start_delay)
    : Message{sender_id, receiver_id},
      start_delay_{start_delay} {}
  std::string serialize() const override {
    return {"naive serialization to JSON"};
  }
private:
  const std::chrono::milliseconds start_delay_;
};
class StopMessage : public Message {
public:
  StopMessage(SenderId sender_id, ReceiverId receiver_id,
              std::chrono::milliseconds stop_delay)
    : Message{sender_id, receiver_id},
      stop_delay_{stop_delay} {}
  std::string serialize() const override {
    return {"naive serialization to JSON"};
  }
private:
  const std::chrono::milliseconds stop_delay_;
};
```

注意，这些实现都不需要修改其他类，每个实现都提供了自己的`serialize`方法版本。`MessageSender`和`MessageSaver`类不需要额外的调整来支持消息的新类层次结构。然而，我们也将对它们进行修改。主要原因是为了使它们可扩展而不需要修改。例如，消息不仅可以保存到文件系统，还可以保存到远程存储。在这种情况下，`MessageSaver`变为纯虚拟的：

```cpp
class MessageSaver {
public:
  virtual void save(const Message& message) const = 0;
};
```

负责保存到文件系统的实现是从 `MessageSaver` 派生出来的类：

```cpp
class FilesystemMessageSaver : public MessageSaver {
public:
  FilesystemMessageSaver(const std::string&
    target_directory);
  void save(const Message& message) const override;
};
```

并且远程存储保存器是层次结构中的另一个类：

```cpp
class RemoteMessageSaver : public MessageSaver {
public:
    RemoteMessageSaver(const std::string&
      remote_storage_address);
    void save(const Message& message) const override;
};
```

## Liskov 替换原则

**Liskov 替换原则**（**LSP**）是面向对象编程中的一个基本原则，它指出，超类对象应该能够被子类对象替换，而不会影响程序的正确性。这个原则也被称为 Liskov 原则，以 Barbara Liskov 的名字命名，她是第一个提出这个原则的人。LSP 基于继承和多态的概念，其中子类可以继承其父类的属性和方法，并且可以与它互换使用。

为了遵循 LSP，子类必须与它们的父类“行为兼容”。这意味着它们应该有相同的方法签名并遵循相同的契约，例如输入和输出类型和范围。此外，子类中方法的行为不应违反父类中建立的任何契约。

让我们考虑一个新的 `Message` 类型，`InternalMessage`，它不支持 `serialize` 方法。有人可能会倾向于以下方式实现它：

```cpp
class InternalMessage : public Message {
public:
    InternalMessage(SenderId sender_id, ReceiverId
      receiver_id)
        : Message{sender_id, receiver_id} {}
    std::string serialize() const override {
        throw std::runtime_error{"InternalMessage can't be
          serialized!"};
    }
};
```

在前面的代码中，`InternalMessage` 是 `Message` 的一个子类型，但不能被序列化，而是抛出异常。这种设计存在几个问题：

+   `InternalMessage` 是 `Message` 的一个子类型，那么我们应该能够在期望 `Message` 的任何地方使用 `InternalMessage`，而不会影响程序的正确性。通过在 `serialize` 方法中抛出异常，我们打破了这一原则。

+   `serialize` 必须处理异常，这在处理其他 `Message` 类型时可能并不必要。这引入了额外的复杂性，并在调用者代码中引入了错误的可能性。

+   **程序崩溃**：如果异常没有得到适当的处理，可能会导致程序崩溃，这当然不是期望的结果。

我们可以返回一个空字符串而不是抛出异常，但这仍然违反了 LSP，因为 `serialize` 方法预期返回一个序列化的消息，而不是一个空字符串。这也引入了歧义，因为不清楚空字符串是没有任何数据的消息成功序列化的结果，还是 `InternalMessage` 序列化失败的结果。

一个更好的方法是分离 `Message` 和 `SerializableMessage` 的关注点，其中只有 `SerializableMessage` 有 `serialize` 方法：

```cpp
class Message {
public:
    virtual ~Message() = default;
    // other common message behaviors
};
class SerializableMessage : public Message {
public:
    virtual std::string serialize() const = 0;
};
class StartMessage : public SerializableMessage {
    // ...
};
class StopMessage : public SerializableMessage {
    // ...
};
class InternalMessage : public Message {
    // InternalMessage doesn't have serialize method now.
};
```

在这个修正的设计中，基类 `Message` 不包括 `serialize` 方法，并引入了一个新的 `SerializableMessage` 类，其中包含这个方法。这样，只有可以序列化的消息才会从 `SerializableMessage` 继承，并且我们遵循了 LSP。

遵循 LSP（里氏替换原则）可以使代码更加灵活和易于维护，因为它允许使用多态，并允许用子类对象替换类对象，而不会影响程序的整体行为。这样，程序可以利用子类提供的新功能，同时保持与超类相同的行为。

接口隔离原则

**接口隔离原则**（ISP）是面向对象编程中的一个原则，它指出一个类应该只实现它使用的接口。换句话说，它建议接口应该是细粒度和客户端特定的，而不是一个单一、庞大且包罗万象的接口。ISP 基于这样一个观点：拥有许多小型接口，每个接口定义一组特定的方法，比拥有一个定义了许多方法的单一大型接口要好。

ISP（接口隔离原则）的一个关键好处是它促进了更模块化和灵活的设计，因为它允许创建针对客户端特定需求的接口。这样，它减少了客户端需要实现的不必要方法数量，同时也减少了客户端依赖于它不需要的方法的风险。

当从 MessagePack 或 JSON 文件创建我们的示例消息时，可以观察到 ISP（接口隔离原则）的一个例子。遵循最佳实践，我们会创建一个提供两个方法`from_message_pack`和`from_json`的接口。

当前的实现需要实现这两个方法，但如果一个特定的类不需要支持这两种选项怎么办？接口越小越好。`MessageParser`接口将被拆分为两个独立的接口，每个接口都需要实现 JSON 或 MessagePack 之一：

```cpp
class JsonMessageParser {
public:
  virtual std::unique_ptr<Message>
  parse(const std::vector<uint8_t>& message_pack)
    const = 0;
};
class MessagePackMessageParser {
public:
  virtual std::unique_ptr<Message>
  parse(const std::vector<uint8_t>& message_pack)
    const = 0;
};
```

这种设计允许从`JsonMessageParser`和`MessagePackMessageParser`派生的对象理解如何分别从 JSON 和 MessagePack 构建自己，同时保持每个函数的独立性和功能性。系统保持灵活性，因为新的更小的对象仍然可以组合起来以实现所需的功能。

遵循 ISP 可以使代码更易于维护且更少出错，因为它减少了客户端需要实现的不必要方法数量，同时也减少了客户端依赖于它不需要的方法的风险。

## 依赖倒置原则

依赖倒置原则基于这样一个观点：依赖抽象比依赖具体实现要好，因为它提供了更大的灵活性和可维护性。它允许将高级模块与低级模块解耦，使它们更加独立，并减少对低级模块变化的敏感性。这样，它使得在不影响高级模块的情况下轻松更改低级实现，反之亦然。

如果我们尝试通过另一个类使用所有组件，可以说明 DIP（依赖倒置原则）在我们的消息系统中是如何体现的。让我们假设有一个负责消息路由的类。为了构建这样一个类，我们将使用`MessageSender`作为通信模块，`Message`基于的类，以及`MessageSaver`：

```cpp
class MessageRouter {
public:
  MessageRouter(ReceiverId id)
    : id_{id} {}
  void route(const Message& message) const {
    if (message.receiver_id() == id_) {
      handler_.handle(message);
    } else {
      try {
        sender_.send(message);
      } catch (const CommunicationError& e) {
        saver_.save(message);
      }
    }
  }
private:
  const ReceiverId id_;
  const MessageHandler handler_;
  const MessageSender sender_;
  const MessageSaver saver_;
};
```

新的类只提供了一个`route`方法，该方法在新的消息可用时被调用一次。如果消息的发送者 ID 与路由器相同，路由器将处理消息到`MessageHandler`类。否则，路由器将消息转发到相应的接收者。如果消息传递失败且通信层抛出异常，路由器将通过`MessageSaver`保存消息。这些消息将在其他时间传递。

唯一的问题是，如果任何依赖项需要更改，路由器的代码必须相应更新。例如，如果应用程序需要支持多种类型的发送者（TCP 和 UDP）、消息保存器（文件系统与远程）或消息处理逻辑发生变化。为了使`MessageRouter`对这种变化无感知，我们可以使用 DIP 原则重写它：

```cpp
class BaseMessageHandler {
public:
    virtual ~BaseMessageHandler() {}
    virtual void handle(const Message& message) const = 0;
};
class BaseMessageSender {
public:
    virtual ~BaseMessageSender() {}
    virtual void send(const Message& message) const = 0;
};
class BaseMessageSaver {
public:
    virtual ~BaseMessageSaver() {}
    virtual void save(const Message& message) const = 0;
};
class MessageRouter {
public:
    MessageRouter(ReceiverId id,
                  const BaseMessageHandler& handler,
                  const BaseMessageSender& sender,
                  const BaseMessageSaver& saver)
        : id_{id}, handler_{handler}, sender_{sender},
          saver_{saver} {}
    void route(const Message& message) const {
        if (message.receiver_id() == id_) {
            handler_.handle(message);
        } else {
            try {
                sender_.send(message);
            } catch (const CommunicationError& e) {
                saver_.save(message);
            }
        }
    }
private:
    ReceiverId id_;
    const BaseMessageHandler& handler_;
    const BaseMessageSender& sender_;
    const BaseMessageSaver& saver_;
};
int main() {
  auto id      = ReceiverId{42};
  auto handler = MessageHandler{};
  auto sender = MessageSender{
    Communication::get_instance()};
  auto saver =
    FilesystemMessageSaver{"/tmp/undelivered_messages"};
  auto router = MessageRouter{id, sender, saver};
}
```

在这个代码的修订版本中，`MessageRouter`现在与消息处理、发送和保存逻辑的具体实现解耦。相反，它依赖于由`BaseMessageHandler`、`BaseMessageSender`和`BaseMessageSaver`表示的抽象。这样，任何从这些基类派生的类都可以与`MessageRouter`一起使用，这使得代码更加灵活，并便于未来扩展。路由器不关心消息处理、发送或保存的具体细节——它只需要知道这些操作可以执行。

遵循 DIP（依赖倒置原则）可以使代码更易于维护且更不易出错。它将高级模块与低级模块解耦，使它们更加独立，并减少对低级模块变化的敏感性。它还提供了更大的灵活性，使得在不影响高级模块的情况下轻松更改低级实现，反之亦然。本书后面将介绍依赖倒置如何帮助我们开发单元测试时模拟系统的一部分。

# KISS 原则

KISS 原则，即“保持简单，傻瓜”，是一种强调保持事物简单直接的设计哲学。这一原则在编程领域尤为重要，因为复杂的代码可能导致错误、困惑和缓慢的开发速度。

下面是一些如何在 C++中应用 KISS 原则的例子：

+   使用`for`循环代替复杂的算法往往同样有效，而且更容易理解。

+   **保持函数简洁**：C++中的函数应该小巧、专注且易于理解。复杂的函数很快就会变得难以维护和调试，因此尽量保持函数尽可能简单和简洁。一个很好的经验法则是使函数的代码行数不超过 30-50 行。

+   **使用清晰简洁的变量名**：在 C++中，变量名在使代码可读和理解方面起着至关重要的作用。避免使用缩写，而应选择清晰简洁的名称，准确描述变量的用途。

+   **避免深层嵌套**：嵌套循环和条件语句会使代码难以阅读和遵循。尽量保持嵌套级别尽可能浅，并考虑将复杂的函数分解成更小、更简单的函数。

+   **编写简单、易读的代码**：首先，目标是编写易于理解和遵循的代码。这意味着使用清晰简洁的语言，并避免复杂的表达式和结构。简单且易于遵循的代码更有可能易于维护且无错误。

+   **避免复杂的继承层次结构**：复杂的继承层次结构会使代码更难以理解、调试和维护。继承结构越复杂，跟踪类之间的关系以及确定更改如何影响其余代码就越困难。

总结来说，KISS 原则是一种简单直接的设计理念，可以帮助开发者编写清晰、简洁且易于维护的代码。通过保持简单，开发者可以避免错误和混淆，并加快开发速度。

## KISS 原则和 SOLID 原则都是软件开发中的重要设计理念，但它们有时可能会相互矛盾。

SOLID 原则和 KISS 原则都是软件开发中的重要设计理念，但它们有时可能会相互矛盾。

SOLID 原则是一套指导软件开发设计的五个原则，旨在使软件更具可维护性、可扩展性和灵活性。它们侧重于创建一个干净、模块化的架构，遵循良好的面向对象设计实践。

另一方面，KISS 原则的核心是保持简单。它提倡简单直接的方法，避免复杂的算法和结构，这些可能会使代码难以理解和维护。

虽然 SOLID 原则和 KISS 原则都旨在提高软件质量，但它们有时可能会产生冲突。例如，遵循 SOLID 原则可能会导致代码更加复杂且难以理解，以实现更大的模块化和可维护性。同样，KISS 原则可能会导致代码不够灵活和可扩展，以保持其简单和直接。

在实践中，开发者通常需要在 SOLID 原则和 KISS 原则之间取得平衡。一方面，他们希望编写可维护、可扩展和灵活的代码。另一方面，他们希望编写简单且易于理解的代码。找到这种平衡需要仔细考虑权衡，并理解何时采用每种方法最为合适。

当我必须在 SOLID 方法和 KISS 方法之间做出选择时，我会想起我的老板 Amir Taya 说过的话：“当你建造法拉利时，你需要从一辆踏板车开始。”这句话是 KISS 的一个夸张例子：如果你不知道如何构建一个功能，就创建一个最简单的可工作版本（KISS），然后迭代，并在需要时使用 SOLID 原则扩展解决方案。

# 副作用和不可变性

副作用和不可变性是编程中的两个重要概念，对代码的质量和可维护性有重大影响。

副作用是指由于执行特定函数或代码片段而导致程序状态发生变化。副作用可以是显式的，例如将数据写入文件或更新变量，也可以是隐式的，例如修改全局状态或在代码的其他部分引起意外的行为。

另一方面，不可变性是指变量或数据结构在创建后不能被修改的特性。在函数式编程中，通过使数据结构和变量成为常量并避免副作用来实现不可变性。

避免副作用和使用不可变变量的重要性在于，它们使代码更容易理解、调试和维护。当代码副作用较少时，更容易推理出它做什么以及它不做什么。这使得找到和修复错误以及修改代码更容易，而不会影响系统的其他部分。

相比之下，具有许多副作用的代码更难以理解，因为程序的状态可能会以意想不到的方式发生变化。这使得调试和维护更加困难，并可能导致错误和意外行为。

函数式编程语言长期以来一直强调使用不可变性和避免副作用，但现在使用 C++编写具有这些特性的代码也是可能的。实现它的最简单方法是遵循**C++核心指南中的常量和不可变性**。

## Con.1 – 默认情况下，使对象不可变

您可以将内置数据类型或用户定义数据类型的实例声明为常量，从而产生相同的效果。尝试修改它将导致编译器错误：

```cpp
struct Data {
  int val{42};
};
int main() {
  const Data data;
  data.val = 43; // assignment of member 'Data::val' in
                 // read-only object
  const int val{42};
  val = 43; // assignment of read-only variable 'val'
}
```

同样适用于循环：

```cpp
for (const int i : array) {
  std::cout << i << std::endl; // just reading: const
}
for (int i : array) {
  std::cout << i << std::endl; // just reading: non-const
}
```

这种方法可以防止难以察觉的值的变化。

可能唯一的例外是按值传递的函数参数：

```cpp
void foo(const int value);
```

这样的参数很少作为`const`传递，也很少被修改。为了避免混淆，建议在这种情况下不要强制执行此规则。

## Con.2 – 默认情况下，使成员函数为 const

成员函数（方法）应当被标记为`const`，除非它改变了对象的可观察状态。这样做的原因是为了给出更精确的设计意图声明，更好的可读性，更好的可维护性，编译器能捕获更多错误，以及理论上更多的优化机会：

```cpp
class Book {
public:
  std::string name() { return name_; }
private:
  std::string name_;
};
void print(const Book& book) {
  cout << book.name()
       << endl; // ERROR: 'this' argument to member
                // function
                // 'name' has type 'const Book', but
                // function is not marked
                // const clang(member_function_call_bad_cvr)
}
```

存在两种类型的 const 属性：**物理**和**逻辑**：

`const`且不能被更改。

`const`但可以被更改。

逻辑常量属性可以通过`mutable`关键字实现。通常情况下，这是一个很少用的用例。我能想到的唯一好例子是将数据存储在内部缓存中或使用互斥锁：

```cpp
class DataReader {
public:
  Data read() const {
    auto lock = std::lock_guard<std::mutex>(mutex);
    // read data
    return Data{};
  }
private:
  mutable std::mutex mutex;
};
```

在这个例子中，我们需要更改`mutex`变量来锁定它，但这不会影响对象的逻辑常量属性。

请注意，存在一些遗留代码/库提供了声明`T*`的函数，尽管它们没有对`T`进行任何更改。这给试图将所有逻辑上常量方法标记为`const`的个人带来了问题。为了强制 const 属性，你可以执行以下操作：

+   更新库/代码以使其符合 const-correct，这是首选解决方案。

+   提供一个包装函数来去除 const 属性。

示例

```cpp
void read_data(int* data); // Legacy code: read_data does
                           // not modify `*data`
void read_data(const int* data) {
  read_data(const_cast<int*>(data));
}
```

注意，这个解决方案是一个补丁，只能在无法修改`read_data`的声明时使用。

## Con.3 – 默认情况下，传递指针和引用到 const

这个很简单；当被调用的函数不修改状态时，推理程序更容易。

让我们看看以下两个函数：

```cpp
void foo(char* p);
void bar(const char* p);
```

`foo`函数是否修改了`p`指针指向的数据？仅通过查看声明我们无法回答，所以我们默认假设它修改了。然而，`bar`函数明确指出`p`的内容将不会被更改。

## Con.4 – 使用 const 定义在构造后值不改变的对象

这条规则与第一条非常相似，强制对象在未来的预期中不被更改的 const 属性。这对于像`Config`这样的类非常有帮助，这些类在应用程序开始时创建，并在其生命周期内不发生变化：

```cpp
class Config {
public:
  std::string hostname() const;
  uint16_t port() const;
};
int main(int argc, char* argv[]) {
  const Config config = parse_args(argc, argv);
  run(config);
}
```

## Con.5 – 对于可以在编译时计算出的值使用 constexpr

如果值在编译时计算，将变量声明为`constexpr`比声明为`const`更可取。它提供了更好的性能、更好的编译时检查、保证的编译时评估以及没有竞争条件发生的可能性。

## 常量属性和数据竞争

当多个线程同时访问一个共享变量，并且至少有一个尝试修改它时，就会发生数据竞争。有一些同步原语，如互斥锁、临界区、自旋锁和信号量，可以防止数据竞争。这些原语的问题在于它们要么执行昂贵的系统调用，要么过度使用 CPU，这使代码效率降低。然而，如果没有线程修改变量，就没有数据竞争的地方。我们了解到 `constexpr` 是线程安全的（不需要同步），因为它是在编译时定义的。那么 `const` 呢？在以下条件下它可以线程安全。

变量自创建以来一直是 `const` 的。如果一个线程直接或间接（通过指针或引用）对变量有非 `const` 访问权限，所有读取者都需要使用互斥锁。以下代码片段展示了从多个线程对常量和非常量访问的示例：

```cpp
void a() {
  auto value = int{42};
  auto t = std::thread([&]() { std::cout << value; });
  t.join();
}
void b() {
  auto value = int{42};
  auto t = std::thread([&value = std::as_const(value)]() {
    std::cout << value;
  });
  t.join();
}
void c() {
  const auto value = int{42};
  auto t = std::thread([&]() {
      auto v = const_cast<int&>(value);
      std::cout << v;
  });
  t.join();
}
void d() {
  const auto value = int{42};
  auto t = std::thread([&]() { std::cout << value; });
  t.join();
}
```

在 `a` 函数中，`value` 变量由主线程和 `t` 都以非 `const` 的方式拥有，这使得代码可能不是线程安全的（如果开发者在主线程中稍后决定更改 `value`）。在 `b` 中，主线程对 `value` 有“写入”访问权限，而 `t` 通过一个 `const` 引用接收它，但仍然不是线程安全的。`c` 函数是糟糕代码的一个例子：`value` 在主线程中被创建为一个常量，并通过 `const` 引用传递，但随后常量性被取消，这使得这个函数不是线程安全的。只有 `d` 函数是线程安全的，因为主线程和 `t` 都不能修改这个变量。

变量的数据类型及其所有子类型要么是物理常量，要么它们的逻辑常量实现是线程安全的。例如，在以下示例中，`Point` 结构体是物理常量，因为它的 `x` 和 `y` 字段成员是原始整数，并且两个线程都只有对它的 `const` 访问权限：

```cpp
struct Point {
  int x;
  int y;
};
void foo() {
  const auto point = Point{.x = 10, .y = 10};
  auto t           = std::thread([&]() { std::cout <<
    point.x; });
  t.join();
}
```

我们之前看到的 `DataReader` 类在逻辑上是常量的，因为它有一个可变的变量 `mutex`，但这个实现也是线程安全的（由于锁的存在）：

```cpp
class DataReader {
public:
  Data read() const {
    auto lock = std::lock_guard<std::mutex>(mutex);
    // read data
    return Data{};
  }
private:
  mutable std::mutex mutex;
};
```

然而，让我们看看以下情况。`RequestProcessor` 类处理一些重量级请求并将结果缓存到一个内部变量中：

```cpp
class RequestProcessor {
public:
  Result process(uint64_t request_id,
                 Request request) const {
    if (auto it = cache_.find(request_id); it !=
      cache_.cend()) {
      return it->second;
    }
    // process request
    // create result
    auto result = Result{};
    cache_[request_id] = result;
    return result;
  }
private:
  mutable std::unordered_map<uint64_t, Result> cache_;
};
void process_request() {
  auto requests = std::vector<std::tuple<uint64_t,
    Request>>{};
  const auto processor = RequestProcessor{};
  for (const auto& request : requests) {
    auto t = std::thread([&]() {
      processor.process(std::get<0>(request),
                        std::get<1>(request));
    });
    t.detach();
  }
}
```

这个类在逻辑上是安全的，但 `cache_` 变量以非线程安全的方式更改，这使得即使在声明为 `const` 的情况下，这个类也不是线程安全的。

注意，当与 STL 容器一起工作时，必须记住，尽管当前实现倾向于线程安全（在物理和逻辑上），但标准提供了非常具体的线程安全保证。

容器中的所有函数都可以被不同容器上的各种线程同时调用。从广义上讲，除非通过函数参数可访问，否则 C++ 标准库中的函数不会读取其他线程可访问的对象，这包括 `this` 指针。

所有`const`成员函数都是线程安全的，这意味着它们可以被多个线程在同一个容器上同时调用。此外，`begin()`、`end()`、`rbegin()`、`rend()`、`front()`、`back()`、`data()`、`find()`、`lower_bound()`、`upper_bound()`、`equal_range()`、`at()`和`operator[]`（在关联容器中除外）成员函数在线程安全性方面也表现为`const`。换句话说，它们也可以被多个线程在同一个容器上调用。广泛地说，C++ 标准库函数不会修改对象，除非这些对象可以通过函数的非`const`参数直接或间接地访问，这包括`this`指针。

同一容器中的不同元素可以由不同的线程同时修改，但`std::vector<bool>`元素除外。例如，一个`std::vector`的`std::future`对象可以一次从多个线程接收值。

迭代器上的操作，如递增迭代器，读取底层容器但不修改它。这些操作可以与其他迭代器的操作、`const`成员函数或对元素的读取同时进行。然而，会使任何迭代器失效的操作会修改容器，并且不能与任何现有迭代器的操作同时进行，即使是那些未被失效的迭代器。

同一容器的元素可以与那些不访问这些元素的成员函数同时修改。广泛地说，C++ 标准库函数不会通过其参数间接读取对象（包括容器的其他元素），除非其规范要求。

最后，只要用户可见的结果不受影响，容器上的操作（以及算法或其他 C++ 标准库函数）可以在内部并行化。例如，`std::transform`可以并行化，但`std::for_each`不能，因为它指定了按顺序访问序列中的每个元素。

将对象的单个可变引用作为 Rust 编程语言的一个支柱的想法。此规则旨在防止数据竞争，当多个线程同时访问相同的可变数据时，就会发生数据竞争，导致不可预测的行为和潜在的崩溃。通过一次只允许对对象的一个可变引用，Rust 确保了对同一数据的并发访问得到适当的同步，并避免了数据竞争。

此外，此规则有助于防止可变别名，当存在对同一数据的多个可变引用同时存在时，就会发生可变别名。可变别名可能导致微妙的错误，并使代码难以推理，尤其是在大型和复杂的代码库中。通过只允许对对象的一个可变引用，Rust 避免了可变别名，并有助于确保代码的正确性和易于理解。

然而，值得注意的是，Rust 也允许对对象有多个不可变引用，这在需要并发访问但不需要修改的场景中可能很有用。通过允许多个不可变引用，Rust 可以在保持安全性和正确性的同时提供更好的性能和并发性。

# 摘要

在本章中，我们介绍了 SOLID 原则、KISS 原则、const 属性和不可变性。让我们看看你学到了什么！

+   SOLID 原则：SOLID 是一组五个原则，帮助我们创建易于维护、可扩展和灵活的代码。通过理解这些原则，你将走向设计出易于工作的代码的梦想之路！

+   KISS 原则：KISS 原则的核心是保持简单。通过遵循这一原则，你可以避免过度复杂化你的代码，使其更容易维护和调试。

+   Const 属性：Const 属性是 C++中的一个特性，它使对象变为只读。通过将对象声明为`const`，你可以确保它们的值不会意外改变，从而使你的代码更加稳定和可预测。

+   不可变性：不可变性确保对象在创建后不能被改变。通过使对象不可变，你可以避免隐蔽的 bug，并使你的代码更具可预测性。

在掌握这些设计原则之后，你将走向编写既健壮又可靠的代码的道路。祝您编码愉快！

在下一章中，我们将尝试理解导致糟糕代码的原因。
