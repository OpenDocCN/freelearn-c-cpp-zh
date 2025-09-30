

# 第四章：确定重构的理想候选者 - 模式和反模式

重构是软件开发中的一个关键技术，它涉及对现有代码进行更改以改进其结构、可读性和可维护性，而不改变其行为。它对于几个原因至关重要。

它有助于消除技术债务并提高代码库的整体质量。开发者可以通过删除冗余或重复的代码、简化复杂的代码和改进代码可读性来实现这一点，从而产生更易于维护和健壮的软件。

重构促进了未来的开发。通过重构代码以使其更模块化，开发者可以更有效地重用现有代码，节省未来开发的时间和精力。这使得代码更具灵活性和适应性，更容易添加新功能、修复错误和优化性能。

结构良好且易于维护的代码使得多个开发者能够更有效地在项目上进行协作。重构有助于标准化代码实践，减少复杂性，并改进文档，使开发者更容易理解和贡献代码库。

最终，重构可以降低长期软件开发相关的成本。通过提高代码质量和可维护性，重构可以帮助减少修复错误、更新和其他维护任务所需的时间和精力。

在本章中，我们将专注于在 C++项目中识别重构的良好候选者。然而，在大型和复杂的系统中，确定适合重构的正确代码段可能具有挑战性。因此，了解使代码段成为重构理想候选者的因素至关重要。在本章中，我们将探讨这些因素，并提供在 C++中识别重构良好候选者的指南。我们还将讨论可以用来提高 C++代码质量的常见重构技术和工具。

# 哪种代码值得重写？

确定是否值得重写的代码取决于几个因素，包括代码的可维护性、可读性、性能、可扩展性和遵循最佳实践的程度。让我们看看代码可能值得重写的一些情况。

**有问题的代码**通常是代码需要重写的迹象。这些是设计或实现不佳的迹象，例如方法过长、类过大、代码重复或命名约定不佳。解决这些代码问题可以改善代码库的整体质量，并使其长期维护更容易。

表现出低内聚或高耦合的代码可能值得重写。低内聚意味着模块或类内的元素之间没有紧密关联，并且模块或类有太多的职责。高耦合指的是模块或类之间的高度依赖性，使得代码更难以维护和修改。重构此类代码可以导致更模块化和易于理解的架构。

在前面的章节中，我们讨论了 SOLID 原则的重要性；违反这些原则的代码也值得重写。

另一个重写代码的原因是如果它依赖于过时的技术、库或编程实践。随着时间的推移，此类代码可能越来越难以维护，并且可能无法利用更新、更有效的方法或工具。将代码更新为使用当前技术和实践可以提高其性能、安全性和可维护性。

最后，如果代码存在性能或可扩展性问题，可能值得重写。这可能涉及优化算法、数据结构或资源管理，以确保代码运行得更高效，并能处理更大的工作负载。

# 代码恶臭及其基本特征

**有问题的代码**，也称为**代码恶臭**，指的是代码库中表明潜在设计或实现问题的症状。这些症状并不一定是错误，但它们是潜在问题的指示，这些问题可能会使代码更难以理解、维护和修改。代码恶臭往往是由于糟糕的编码实践或随着时间的推移技术债务的积累造成的。尽管代码恶臭可能不会直接影响程序的功能，但它们可以显著影响代码的整体质量，导致错误风险增加和开发者生产率下降。

解决代码恶臭的一个方面是识别并应用适当的设计模式。设计模式是解决软件设计中常见问题的可重用解决方案。它们为解决特定问题提供了一个经过验证的框架，使开发者能够建立在其他开发者的集体智慧和经验之上。通过应用这些模式，可以将恶臭代码重构为更结构化、模块化和易于维护的形式。让我们看看一些例子。

策略模式允许我们定义一组算法，将每个算法封装在单独的类中，并在运行时使它们可互换。策略模式对于重构具有多个分支或执行类似任务但实现略有不同的条件的代码非常有用。

让我们考虑一个使用不同存储策略保存数据的应用程序的例子，例如保存到磁盘或远程存储服务：

```cpp
#include <iostream>
#include <fstream>
#include <string>
#include <assert>
enum class StorageType {
    Disk,
    Remote
};
class DataSaver {
public:
    DataSaver(StorageType storage_type) : storage_type_(storage_type) {}
    void save_data(const std::string& data) const {
        switch (storage_type_) {
            case StorageType::Disk:
                save_to_disk(data);
                break;
            case StorageType::Remote:
                save_to_remote(data);
                break;
            default:
                assert(false && “Unknown storage type.”);
        }
    }
    void set_storage_type(StorageType storage_type) {
        storage_type_ = storage_type;
    }
private:
    void save_to_disk(const std::string& data) const {
        // saving to disk
    }
    void save_to_remote(const std::string& data) const {
        // saving data to a remote storage service.
    }
    StorageType storage_type_;
};
int main() {
    DataSaver disk_data_saver(StorageType::Disk);
    disk_data_saver.save_data(“Save this data to disk.”);
    DataSaver remote_data_saver(StorageType::Remote);
    remote_data_saver.save_data(“Save this data to remote storage.”);
    // Switch the storage type at runtime.
    disk_data_saver.set_storage_type(StorageType::Remote);
    disk_data_saver.save_data(“Save this data to remote storage after switching storage type.”);
    return 0;
}
```

在本课程中，`save_data`方法在每次调用时都会检查存储类型，并使用`switch-case`块来决定使用哪种保存方法。这种方法可行，但有一些缺点：

+   `DataSaver`类负责处理所有不同的存储类型，这使得维护和扩展更困难。

+   添加新的存储类型需要修改`DataSaver`类和`StorageType`枚举，增加了引入错误或破坏现有功能的风险。例如，如果由于某种原因提供了错误的枚举类型，代码将终止。

+   与将行为封装在单独类中的策略模式相比，代码的模块化和灵活性较低。

通过实现策略模式，我们可以解决这些缺点，并为`DataSaver`类创建一个更易于维护、灵活和可扩展的设计。首先，定义一个名为`SaveStrategy`的接口，它代表保存行为：

```cpp
class SaveStrategy {
public:
    virtual ~SaveStrategy() {}
    virtual void save_data(const std::string& data) const = 0;
};
```

接下来，为每种存储类型实现具体的`SaveStrategy`类：

```cpp
class DiskSaveStrategy : public SaveStrategy {
public:
    void save_data(const std::string& data) const override {
        // ...
    }
};
class RemoteSaveStrategy : public SaveStrategy {
public:
    void save_data(const std::string& data) const override {
        // ...
    }
};
```

现在，创建一个使用策略模式将保存行为委托给适当的`SaveStrategy`实现的`DataSaver`类：

```cpp
class DataSaver {
public:
    DataSaver(std::unique_ptr<SaveStrategy> save_strategy)
        : save_strategy_(std::move(save_strategy)) {}
    void save_data(const std::string& data) const {
        save_strategy_->save_data(data);
    }
    void set_save_strategy(std::unique_ptr<SaveStrategy> save_strategy) {
        save_strategy_ = std::move(save_strategy);
    }
private:
    std::unique_ptr<SaveStrategy> save_strategy_;
};
```

最后，这里是一个如何使用`DataSaver`类和不同的保存策略的示例：

```cpp
int main() {
    DataSaver disk_data_saver(std::make_unique<DiskSaveStrategy>());
    disk_data_saver.save_data(“Save this data to disk.”);
    DataSaver remote_data_saver(std::make_unique<RemoteSaveStrategy>());
    remote_data_saver.save_data(“Save this data to remote storage.”);
    // Switch the saving strategy at runtime.
    disk_data_saver.set_save_strategy(std::make_unique<RemoteSaveStrategy>());
    disk_data_saver.save_data(“Save this data to remote storage after switching strategy.”);
    return 0;
}
```

在这个例子中，`DataSaver`类使用策略模式将它的保存行为委托给不同的`SaveStrategy`实现，这使得它能够轻松地在保存到磁盘和保存到远程存储之间切换。这种设计使代码更加模块化、可维护和灵活，允许以最小的现有代码更改添加新的存储策略。此外，新版本的代码不需要在错误的保存策略类型上终止或抛出异常。

假设我们有两个格式的文件解析实现，CSV 和 JSON：

```cpp
class CsvParser {
public:
    void parse_file(const std::string& file_path) {
        std::ifstream file(file_path);
        if (!file) {
            std::cerr << “Error opening file: “ << file_path << std::endl;
            return;
        }
        std::string line;
        while (std::getline(file, line)) {
            process_line(line);
        }
        file.close();
        post_process();
    }
private:
    void process_line(const std::string& line) {
        // Implement the CSV-specific parsing logic.
        std::cout << “Processing CSV line: “ << line << std::endl;
    }
    void post_process() {
        std::cout << “CSV parsing completed.” << std::endl;
    }
};
class JsonParser {
public:
    void parse_file(const std::string& file_path) {
        std::ifstream file(file_path);
        if (!file) {
            std::cerr << “Error opening file: “ << file_path << std::endl;
            return;
        }
        std::string line;
        while (std::getline(file, line)) {
            process_line(line);
        }
        file.close();
        post_process();
    }
private:
    void process_line(const std::string& line) {
        // Implement the JSON-specific parsing logic.
        std::cout << “Processing JSON line: “ << line << std::endl;
    }
    void post_process() {
        std::cout << “JSON parsing completed.” << std::endl;
    }
};
```

在这个例子中，`CsvParser`和`JsonParser`类有`parse_file`方法的独立实现，其中包含打开、读取和关闭文件的重复代码。特定格式的解析逻辑在`process_line`和`post_process`方法中实现。

虽然这种设计可行，但它有一些缺点：共享的解析步骤在两个类中都重复，这使得维护和更新代码更困难，并且添加对新文件格式的支持需要创建具有类似代码结构的新的类，这可能导致更多的代码重复。

通过实现模板方法模式，你可以解决这些缺点，并为文件解析器创建一个更易于维护、可扩展和可重用的设计。`FileParser`基类处理常见的解析步骤，而派生类实现特定格式的解析逻辑。

如前例所示，让我们从创建一个抽象基类开始。`FileParser`代表通用的文件解析过程：

```cpp
class FileParser {
public:
    void parse_file(const std::string& file_path) {
        std::ifstream file(file_path);
        if (!file) {
            std::cerr << “Error opening file: “ << file_path << std::endl;
            return;
        }
        std::string line;
        while (std::getline(file, line)) {
            process_line(line);
        }
        file.close();
        post_process();
    }
protected:
    virtual void process_line(const std::string& line) = 0;
    virtual void post_process() = 0;
};
```

`FileParser`类有一个`parse_file`方法，它处理打开文件、逐行读取其内容以及关闭文件的常见步骤。特定格式的解析逻辑通过纯虚函数`process_line`和`post_process`方法实现，这些方法将由派生类覆盖。

现在，为不同的文件格式创建派生类：

```cpp
class CsvParser : public FileParser {
protected:
    void process_line(const std::string& line) override {
        // Implement the CSV-specific parsing logic.
        std::cout << “Processing CSV line: “ << line << std::endl;
    }
    void post_process() override {
        std::cout << “CSV parsing completed.” << std::endl;
    }
};
class JsonParser : public FileParser {
protected:
    void process_line(const std::string& line) override {
        // Implement the JSON-specific parsing logic.
        std::cout << “Processing JSON line: “ << line << std::endl;
    }
    void post_process() override {
        std::cout << “JSON parsing completed.” << std::endl;
    }
};
```

在这个例子中，`CsvParser`和`JsonParser`类从`FileParser`继承，并在`process_line`和`post_process`方法中实现特定格式的解析逻辑。

下面是一个如何使用文件解析器的示例：

```cpp
int main() {
    CsvParser csv_parser;
    csv_parser.parse_file(“data.csv”);
    JsonParser json_parser;
    json_parser.parse_file(“data.json”);
    return 0;
}
```

通过实现模板方法模式，`FileParser`类提供了一个处理文件解析常见步骤的可重用模板，同时允许派生类实现特定格式的解析逻辑。这种设计使得在不修改基类`FileParser`的情况下添加对新文件格式的支持变得容易，从而使得代码库更加易于维护和扩展。重要的是要注意，通常实现这种设计模式的复杂部分是识别类之间的共同逻辑。通常，实现需要某种形式的共同逻辑的统一。

另一个值得关注的模式是观察者模式。前一章提到了其技术实现细节（原始、共享或弱指针实现）。然而，在这一章中，我想从设计角度介绍其用法。

观察者模式定义了对象之间的一对多依赖关系，允许在主题状态发生变化时通知多个观察者。当重构涉及事件处理或多个依赖组件更新的代码时，这种模式可能是有益的。

考虑一个汽车系统，其中`Engine`类包含汽车当前的速度和每分钟转速（RPM）。有几个元素需要了解这些值，例如`Dashboard`和`Controller`。仪表盘显示来自发动机的最新更新，而`Controller`根据速度和转速调整汽车的行为。实现这一点的直接方法是让`Engine`类直接在每个显示元素上调用`update`方法：

```cpp
class Dashboard {
public:
    void update(int speed, int rpm) {
        // display the current speed
    }
};
class Controller {
public:
    void update(int speed, int rpm) {
        // Adjust car’s behavior based on the speed and RPM.
    }
};
class Engine {
public:
    void set_dashboard(Dashboard* dashboard) {
        dashboard_ = dashboard;
    }
    void set_controller(Controller* controller) {
        controller_ = controller;
    }
    void set_measurements(int speed, int rpm) {
        speed_ = speed;
        rpm_ = rpm;
        measurements_changed();
    }
private:
    void measurements_changed() {
        dashboard_->update(_speed, rpm_);
        controller_->update(_speed, rpm_);
    }
    int speed_;
    int rpm_;
    Dashboard* dashboard_;
    Controller* controller_;
};
int main() {
    Engine engine;
    engine.set_measurements(80, 3000);
    return 0;
}
```

这段代码有几个问题：

+   `Engine`类与`Dashboard`和`Controller`紧密耦合，这使得添加或删除可能对汽车速度和转速感兴趣的其它组件变得困难。

+   `Engine`类直接负责更新显示元素，这使代码变得复杂，并降低了其灵活性。

我们可以使用观察者模式重构代码，将`Engine`与显示元素解耦。`Engine`类将成为主题，而`Dashboard`和`Controller`将成为观察者：

```cpp
class Observer {
public:
    virtual ~Observer() {}
    virtual void update(int speed, int rpm) = 0;
};
class Dashboard : public Observer {
public:
    void update(int speed, int rpm) override {
        // display the current speed
    }
};
class Controller : public Observer {
public:
    void update(int speed, int rpm) override {
        // Adjust car’s behavior based on the speed and RPM.
    }
};
class Engine {
public:
    void register_observer(Observer* observer) {
        observers_.push_back(observer);
    }
    void remove_observer(Observer* observer) {
        observers_.erase(std::remove(_observers.begin(), observers_.end(), observer), observers_.end());
    }
    void set_measurements(int speed, int rpm) {
        speed_ = speed;
        rpm_ = rpm;
        notify_observers();
    }
private:
    void notify_observers() {
        for (auto observer : observers_) {
            observer->update(_speed, _rpm);
        }
    }
    std::vector<Observer*> observers_;
    int speed_;
    int rpm_;
};
```

以下代码片段展示了新类层次结构的用法：

```cpp
int main() {
    Engine engine;
    Dashboard dashboard;
    Controller controller;
    // Register observers
    engine.register_observer(&dashboard);
    engine.register_observer(&controller);
    // Update measurements
    engine.set_measurements(80, 3000);
    // Remove an observer
    engine.remove_observer(&dashboard);
    // Update measurements again
    engine.set_measurements(100, 3500);
    return 0;
}
```

在这个例子中，`Dashboard`和`Controller`被注册为`Engine`主题的观察者。当发动机的速度和转速发生变化时，会调用`set_measurements`，触发`notify_observers`，进而调用每个注册观察者的`update`方法。这使得`Dashboard`和`Controller`能够接收到更新的速度和转速值。

然后，`仪表盘`被取消注册为观察者。当引擎的速度和 RPM 再次更新时，只有`控制器`会接收到更新的值。

在这种设置下，添加或删除观察者就像在`Engine`上调用`register_observer`或`remove_observer`一样简单，并且无需在添加新的观察者类型时修改`Engine`类。现在，`Engine`类与特定的观察者类解耦，使系统更加灵活且易于维护。

另一个伟大的模式是状态机。它不是一个经典模式，但可能是最强大的一种。状态机，也称为**有限状态机**（**FSMs**），是计算数学模型。它们用于表示和控制硬件和软件设计中的执行流程。状态机具有有限数量的状态，在任何给定时间，它都处于这些状态之一。它根据外部输入或预定义条件从一个状态转换到另一个状态。

在硬件领域，状态机经常用于数字系统的设计，作为从小型微控制器到大型**中央处理器**（**CPUs**）的控制逻辑。它们控制操作序列，确保动作按正确的顺序发生，并且系统能够适当地响应不同的输入或条件。

在软件中，状态机同样有用，尤其是在程序流程受一系列状态及其之间转换影响的系统中。应用范围从嵌入式系统中的简单按钮消抖到复杂的游戏角色行为或通信协议管理。

状态机非常适合那些系统有一个明确的状态集合，并且这些状态通过特定的事件或条件循环切换的情况。它们在系统的行为不仅取决于当前输入，还取决于系统历史的情况下特别有用。状态机通过当前状态的形式封装了这一历史，使其明确且易于管理。

使用状态机可以带来许多好处。它们可以简化复杂的条件逻辑，使其更容易理解、调试和维护。它们还使得在不干扰现有代码的情况下添加新状态或转换变得容易，增强了模块化和灵活性。此外，它们使系统的行为明确且可预测，降低了意外行为的风险。

让我们考虑一个分布式计算系统的真实场景，其中一项工作被提交以进行处理。这项工作会经历各种状态，如 `Submitted`、`Queued`、`Running`、`Completed` 和 `Failed`。我们将使用 `Boost.Statechart` 库来模拟这个过程。`Boost.Statechart` 是一个 C++ 库，它提供了一个构建状态机的框架。它是 Boost 库集合的一部分。这个库简化了分层状态机的开发，允许你使用复杂的状态和转换来模拟复杂系统。它的目标是使处理复杂状态逻辑时编写结构良好、模块化和可维护的代码变得更加容易。`Boost.Statechart` 提供了编译时和运行时检查，以帮助确保状态机行为的正确性。

首先，我们包含必要的头文件并设置一些命名空间：

```cpp
#include <boost/statechart/state_machine.hpp>
#include <boost/statechart/simple_state.hpp>
#include <boost/statechart/transition.hpp>
#include <iostream>
namespace sc = boost::statechart;
```

接下来，我们定义我们的事件：`JobSubmitted`、`JobQueued`、`JobRunning`、`JobCompleted` 和 `JobFailed`：

```cpp
struct EventJobSubmitted : sc::event< EventJobSubmitted > {};
struct EventJobQueued : sc::event< EventJobQueued > {};
struct EventJobRunning : sc::event< EventJobRunning > {};
struct EventJobCompleted : sc::event< EventJobCompleted > {};
struct EventJobFailed : sc::event< EventJobFailed > {};
```

然后，我们定义我们的状态，每个状态都是一个继承自 `sc::simple_state` 的类。我们将有五个状态：`Submitted`、`Queued`、`Running`、`Completed` 和 `Failed`：

```cpp
struct Submitted;
struct Queued;
struct Running;
struct Completed;
struct Failed;
struct Submitted : sc::simple_state< Submitted, Job > {
    typedef sc::transition< EventJobQueued, Queued > reactions;
    Submitted() { std::cout << “Job Submitted\n”; }
};
struct Queued : sc::simple_state< Queued, Job > {
    typedef sc::transition< EventJobRunning, Running > reactions;
    Queued() { std::cout << “Job Queued\n”; }
};
struct Running : sc::simple_state< Running, Job > {
    typedef boost::mpl::list<
        sc::transition< EventJobCompleted, Completed >,
        sc::transition< EventJobFailed, Failed >
    > reactions;
    Running() { std::cout << “Job Running\n”; }
};
struct Completed : sc::simple_state< Completed, Job > {
    Completed() { std::cout << “Job Completed\n”; }
};
struct Failed : sc::simple_state< Failed, Job > {
    Failed() { std::cout << “Job Failed\n”; }
};
```

最后，我们定义我们的状态机，`Job`，它从 `Submitted` 状态开始。

```cpp
struct Job : sc::state_machine< Job, Submitted > {};
```

在一个 `main` 函数中，我们可以创建我们的 `Job` 状态机实例并处理一些事件：

```cpp
int main() {
    Job my_job;
    my_job.initiate();
    my_job.process_event(EventJobQueued());
    my_job.process_event(EventJobRunning());
    my_job.process_event(EventJobCompleted());
    return 0;
}
```

这将输出以下内容：

```cpp
Job Submitted
Job Queued
Job Running
Job Completed
```

这个简单的例子展示了如何使用状态机来模拟具有多个状态和转换的过程。我们使用事件来触发状态之间的转换。另一种方法是使用状态反应，其中状态可以根据其拥有的条件或数据来决定何时进行转换。

这可以通过在 `Boost.Statechart` 中使用自定义反应来实现。自定义反应是一个在处理事件时被调用的成员函数。它可以决定要做什么：忽略事件、消费事件而不进行转换，或者转换到新状态。

让我们修改 `Job` 状态机，使其能够根据工作的完成状态来决定何时从 `Running` 转换到 `Completed` 或 `Failed`。

首先，我们将定义一个新的事件，`EventJobUpdate`，它将携带工作的完成状态：

```cpp
struct EventJobUpdate : sc::event< EventJobUpdate > {
    EventJobUpdate(bool is_complete) : is_complete(is_complete) {}
    bool is_complete;
};
```

然后，在 `Running` 状态中，我们将为这个事件定义一个自定义反应：

```cpp
struct Running : sc::simple_state< Running, Job > {
    typedef sc::custom_reaction< EventJobUpdate > reactions;
    sc::result react(const EventJobUpdate& event) {
        if (event.is_complete) {
            return transit<Completed>();
        } else {
            return transit<Failed>();
        }
    }
    Running() { std::cout << “Job Running\n”; }
};
```

现在，`Running` 状态将根据 `EventJobUpdate` 事件的 `is_complete` 字段来决定何时转换到 `Completed` 或 `Failed`。

在 `main` 函数中，我们现在可以处理 `EventJobUpdate` 事件：

```cpp
int main() {
    Job my_job;
    my_job.initiate();
    my_job.process_event(EventJobQueued());
    my_job.process_event(EventJobRunning());
    my_job.process_event(EventJobUpdate(true)); // The job is complete.
    return 0;
}
```

这将输出以下内容：

```cpp
Job Submitted
Job Queued
Job Running
Job Completed
```

如果我们用 `false` 处理 `EventJobUpdate`：

```cpp
my_job.process_event(EventJobUpdate(false)); // The job is not complete.
```

它将输出以下内容：

```cpp
Job Submitted
Job Queued
Job Running
Job Failed
```

这展示了状态如何根据其拥有的条件或数据来决定何时进行转换。

作为状态机实现的逻辑可以通过添加新的状态和它们之间的转换规则来轻松扩展。然而，在某个时刻，状态机可能包含太多的状态（比如说，超过七个）。这通常是一个代码有问题的症状。这意味着状态机被太多的状态所压垮，这些状态实现了多个状态机。例如，我们的分布式系统本身可以作为一个状态机来实现。系统可以有自己的状态，例如`Idle`（空闲）、`ProcessingJobs`（处理作业）和`SystemFailure`（系统故障）。`ProcessingJobs`状态将进一步包含`Job`状态机作为子状态机。`System`状态机可以通过处理事件与`Job`子状态机通信。当`System`转换到`ProcessingJobs`状态时，它可以处理一个`EventJobSubmitted`事件来启动`Job`子状态机。当`Job`转换到`Completed`或`Failed`状态时，它可以处理一个`EventJobFinished`事件来通知`System`。

首先，我们定义了`EventJobFinished`事件：

```cpp
struct EventJobFinished : sc::event< EventJobFinished > {};
```

然后，在`Job`状态机的`Completed`和`Failed`状态中，我们处理`EventJobFinished`事件：

```cpp
struct Completed : sc::simple_state< Completed, Job > {
    Completed() {
        std::cout << “Job Completed\n”;
        context< Job >().outermost_context().process_event(EventJobFinished());
    }
};
struct Failed : sc::simple_state< Failed, Job > {
    Failed() {
        std::cout << “Job Failed\n”;
        context< Job >().outermost_context().process_event(EventJobFinished());
    }
};
```

在`System`状态机的`ProcessingJobs`状态中，我们为`EventJobFinished`事件定义了一个自定义反应：

```cpp
struct ProcessingJobs : sc::state< ProcessingJobs, System, Job > {
    typedef sc::custom_reaction< EventJobFinished > reactions;
    sc::result react(const EventJobFinished&) {
        std::cout << “Job Finished\n”;
        return transit<Idle>();
    }
    ProcessingJobs(my_context ctx) : my_base(ctx) {
        std::cout << “System Processing Jobs\n”;
        context< System >().process_event(EventJobSubmitted());
    }
};
```

在`main`函数中，我们可以创建我们的`System`状态机的一个实例并启动它：

```cpp
int main() {
    System my_system;
    my_system.initiate();
    return 0;
}
```

这将输出以下内容：

```cpp
System Idle
System Processing Jobs
Job Submitted
Job Queued
Job Running
Job Completed
Job Finished
System Idle
```

这展示了`System`状态机如何与`Job`子状态机交互。当`System`转换到`ProcessingJobs`状态时，它会启动`Job`，而当`Job`完成时，它会通知`System`。这允许`System`管理`Job`的生命周期并对它的状态变化做出反应。

这可以使你的状态机更加灵活和动态。

通常，状态机是管理复杂行为的一种强大工具，既稳健又易于理解。尽管它们很有用，但状态机并不总是代码结构的首选，可能是因为它们被认为很复杂或缺乏熟悉度。然而，当处理一个以复杂条件逻辑为特征的系统时，考虑使用状态机可能是一个明智的选择。这是一个强大的工具，可以为你的软件设计带来清晰性和稳健性，使其成为 C++或其他任何语言重构工具包的一个基本组成部分。

# 反模式

与设计模式不同，反模式是长期来看可能产生反效果或有害的常见解决方案。识别和避免反模式对于解决代码问题至关重要，因为应用它们可能会加剧现有问题并引入新的问题。一些反模式的例子包括 Singleton（单例）、God Object（上帝对象）、Copy-Paste Programming（复制粘贴编程）、Premature Optimization（过早优化）和 Spaghetti Code（意大利面代码）。

单例模式众所周知违反了依赖倒置和开闭原则。它创建了一个全局实例，这可能导致类之间的隐藏依赖，使得代码难以理解和维护。它违反了依赖倒置原则，因为它鼓励高层模块依赖于低层模块而不是依赖于抽象。此外，单例模式通常使得在扩展类或进行测试时难以用不同的实现替换单例实例。这违反了开闭原则，因为它要求修改代码以改变或扩展行为。在以下代码示例中，我们有一个单例类 `Database`，它被 `OrderManager` 类使用：

```cpp
class Database {
public:
    static Database& get_instance() {
        static Database instance;
        return instance;
    }
    template<typename T>
    std::optional<T> get(const Id& id) const;
    template<typename T>
    void save(const T& data);
private:
    Database() {} // Private constructor
    Database(const Database&) = delete; // Delete copy constructor
    Database& operator=(const Database&) = delete; // Delete copy assignment operator
};
class OrderManager {
public:
  void addOrder(const Order& order) {
    auto db = Database::get_instance();
    // check order validity
    // notify other components about the new order, etc
    db.save(order);
  }
};
```

将数据库连接表示为单例的想法是非常合理的：应用程序允许每个应用程序实例只有一个数据库连接，数据库在代码的各个地方都被使用。单例的使用隐藏了 `OrderManager` 依赖于 `Database` 的这一事实，这使得代码不那么直观和可预测。单例的使用几乎使得通过单元测试测试 `OrderManager` 的业务逻辑变得不可能，除非运行一个真实的数据库实例。

可以通过在 `main` 函数的开始处创建一个 `Database` 实例并将其传递给所有需要数据库连接的类来解决这个问题：

```cpp
class OrderManager {
  public:
  OrderManager(Database& db);
  // the rest of the code is the same
};
int main() {
  auto db = Database{};
  auto order_manager = OrderManager{db};
}
```

注意，尽管 `Database` 已不再是单例（即其构造函数是公开的），但它仍然不能被复制。技术上，这允许开发者创建新的实例，但这并不是期望的行为。根据我的经验，可以通过团队内的知识共享和代码审查来轻松避免这种情况。那些认为这还不够的开发者可以保持 `Database` 不变，但确保 `get_instance` 只被调用一次，并且从那时起通过引用传递：

```cpp
int main() {
  auto db = Database::get_instance();
  auto order_manager = OrderManager{db};
}
```

如果一个代码问题涉及一个具有太多责任的类，应用上帝对象反模式是不合适的，因为这只会使类更加复杂和难以维护。一般来说，上帝类是对单一责任原则的过度违反。例如，让我们看看以下类，`EcommerceSystem`：

```cpp
class ECommerceSystem {
public:
    // Product management
    void add_product(int id, const std::string& name, uint64_t price) {
        products_[id] = {name, price};
    }
    void remove_product(int id) {
        products_.erase(id);
    }
    void update_product(int id, const std::string& name, uint64_t price) {
        products_[id] = {name, price};
    }
    void list_products() {
        // print the list of products
    }
    // Cart management
    void add_to_cart(int product_id, int quantity) {
        cart_[product_id] += quantity;
    }
    void remove_from_cart(int product_id) {
        cart_.erase(product_id);
    }
    void update_cart(int product_id, int quantity) {
        cart_[product_id] = quantity;
    }
    uint64_t calculate_cart_total() {
        uint64_t total = 0;
        for (const auto& item : cart_) {
            total += products_[item.first].second * item.second;
        }
        return total;
    }
    // Order management
    void place_order() {
        // Process payment, update inventory, send confirmation email, etc.
        // ...
        cart_.clear();
    }
    // Persistence
    void save_to_file(const std::string& file_name) {
        // serializing the state to a file
    }
    void load_from_file(const std::string& file_name) {
        // loading a file and parsing it
    }
private:
    std::map<int, std::pair<std::string, uint64_t>> products_;
    std::map<int, int> cart_;
};
```

在这个例子中，`ECommerceSystem` 类承担了多个责任，如产品管理、购物车管理、订单管理和持久化（从文件中保存和加载数据）。这个类难以维护、理解和修改。

一个更好的方法是将 `ECommerceSystem` 分解成更小、更专注的类，每个类处理特定的责任：

+   `ProductManager` 类管理产品

+   `CartManager` 类管理购物车

+   `OrderManager` 类管理订单和相关任务（例如，处理支付和发送确认电子邮件）

+   `PersistenceManager`类负责从文件中保存和加载数据

这些类可以按以下方式实现：

```cpp
class ProductManager {
public:
    void add_product(int id, const std::string& name, uint64_t price) {
        products_[id] = {name, price};
    }
    void remove_product(int id) {
        products_.erase(id);
    }
    void update_product(int id, const std::string& name, uint64_t price) {
        products_[id] = {name, price};
    }
    std::pair<std::string, uint64_t> get_product(int id) {
        return products_[id];
    }
    void list_products() {
        // print the list of products
    }
private:
    std::map<int, std::pair<std::string, uint64_t>> products_;
};
class CartManager {
public:
    void add_to_cart(int product_id, int quantity) {
        cart_[product_id] += quantity;
    }
    void remove_from_cart(int product_id) {
        cart_.erase(product_id);
    }
    void update_cart(int product_id, int quantity) {
        cart_[product_id] = quantity;
    }
    std::map<int, int> get_cart_contents() {
        return cart_;
    }
    void clear_cart() {
        cart_.clear();
    }
private:
    std::map<int, int> cart_;
};
class OrderManager {
public:
    OrderManager(ProductManager& product_manager, CartManager& cart_manager)
        : product_manager_(product_manager), cart_manager_(cart_manager) {}
    uint64_t calculate_cart_total() {
        // calculate cart’s total the same as before
    }
    void place_order() {
        // Process payment, update inventory, send confirmation email, etc.
        // ...
        cart_manager_.clear_cart();
    }
private:
    ProductManager& product_manager_;
    CartManager& cart_manager_;
};
class PersistenceManager {
public:
    PersistenceManager(ProductManager& product_manager)
        : product_manager_(product_manager) {}
    void save_to_file(const std::string& file_name) {
      // saving
    }
    void load_from_file(const std::string& file_name) {
      // loading
    }
private:
    ProductManager& product_manager_;
};
```

最终，拥有新类并提供其功能代理方法的`ECommerce`类：

```cpp
// include the new classes
class ECommerce {
public:
    void add_product(int id, const std::string& name, uint64_t price) {
        product_manager_.add_product(id, name, price);
    }
    void remove_product(int id) {
        product_manager_.remove_product(id);
    }
    void update_product(int id, const std::string& name, uint64_t price) {
        product_manager_.update_product(id, name, price);
    }
    void list_products() {
        product_manager_.list_products();
    }
    void add_to_cart(int product_id, int quantity) {
        cart_manager_.add_to_cart(product_id, quantity);
    }
    void remove_from_cart(int product_id) {
        cart_manager_.remove_from_cart(product_id);
    }
    void update_cart(int product_id, int quantity) {
        cart_manager_.update_cart(product_id, quantity);
    }
    uint64_t calculate_cart_total() {
        return order_manager_.calculate_cart_total();
    }
    void place_order() {
        order_manager_.place_order();
    }
    void save_to_file(const std::string& filename) {
        persistence_manager_.save_to_file(filename);
    }
    void load_from_file(const std::string& filename) {
        persistence_manager_.load_from_file(filename);
    }
private:
    ProductManager product_manager_;
    CartManager cart_manager_;
    OrderManager order_manager_{product_manager_, cart_manager_};
    PersistenceManager persistence_manager_{product_manager_};
};
int main() {
    ECommerce e_commerce;
    e_commerce.add_product(1, “Laptop”, 999.99);
    e_commerce.add_product(2, “Smartphone”, 699.99);
    e_commerce.add_product(3, “Headphones”, 99.99);
    e_commerce.list_products();
    e_commerce.add_to_cart(1, 1); // Add 1 Laptop to the cart
    e_commerce.add_to_cart(3, 2); // Add 2 Headphones to the cart
    uint64_t cart_total = e_commerce.calculate_cart_total();
    std::cout << “Cart Total: $” << cart_total << std::endl;
    e_commerce.place_order();
    std::cout << “Order placed successfully!” << std::endl;
    e_commerce.save_to_file(“products.txt”);
    e_commerce.remove_product(1);
    e_commerce.remove_product(2);
    e_commerce.remove_product(3);
    std::cout << “Loading products from file...” << std::endl;
    e_commerce.load_from_file(“products.txt”);
    e_commerce.list_products();
    return 0;
}
```

通过将责任分配给多个较小的类，代码变得更加模块化，更容易维护，更适合实际应用。对其中一个子类内部业务逻辑的微小更改不需要更新`ECommerce`类。在 C++中，由于臭名昭著的编译时间问题，这可能更为重要。单独测试这些类或完全替换其中一个类的实现（例如，将数据保存到远程存储而不是磁盘）更容易。

## 魔法数字的陷阱——关于数据分块的一个案例研究

让我们考虑以下 C++函数`send`，该函数旨在将数据块发送到某个目的地。以下是函数的外观：

```cpp
#include <cstddef>
#include <algorithm>
// Actually sending the data
void do_send(const std::uint8_t* data, size_t size);
void send(const std::uint8_t* data, size_t size) {
    for (std::size_t position = 0; position < size;) {
        std::size_t length = std::min(size_t{256}, size - position);  // 256 is a magic number
        do_send(data + position, position + length);
        position += length;
    }
}
```

### 代码做了什么？

`send`函数接受一个指向`std::uint8_t`数组的指针（`data`）及其大小（`size`）。然后它继续将此数据以块的形式发送到`do_send`函数，该函数负责实际发送过程。每个块的最大大小为 256 字节，如`send`函数中定义的那样。

### 为什么魔法数字有问题？

数字 256 直接嵌入到代码中，没有解释它代表什么。这是一个经典的**魔法数字**例子。任何阅读此代码的人都会猜测为什么选择 256。是硬件限制？协议约束？性能调整参数？

### `constexpr`解决方案

提高此代码清晰度的一种方法是将魔法数字替换为命名的`constexpr`变量。例如，代码可以重写如下：

```cpp
#include <cstddef>
#include <algorithm>
constexpr std::size_t MAX_DATA_TO_SEND = 256;  // Named constant replaces magic number
// Actually sending the data
void do_send(const std::uint8_t* data, size_t size);
void send(const std::uint8_t* data, size_t size) {
    for (std::size_t position = 0; position < size;) {
        std::size_t length = std::min(MAX_DATA_TO_SEND, size - position);  // Use the named constant
        do_send(data + position, position + length);
        position += length;
    }
}
```

### 使用`constexpr`的优点

将魔法数字替换为`MAX_DATA_TO_SEND`使得理解这个限制的原因更加容易。此外，如果你有另一个函数，比如`read`，它也需要以 256 字节的块读取数据，使用`constexpr`变量可以确保一致性。如果块的大小需要更改，你只需在一个地方更新它，从而降低错误和不一致的风险。

当处理糟糕的代码时，理解这些问题的根本原因并应用正确的模式或避免反模式以有效地重构代码是至关重要的。例如，如果一个代码问题涉及重复的代码，应避免复制粘贴编程，而是应用模板方法或策略模式等模式以促进代码重用并减少重复。同样，如果一个代码问题涉及紧密耦合的模块或类，应应用适配器或依赖倒置原则等模式以减少耦合并提高模块化。

重要的是要记住，重构代码异味应该是一个迭代和逐步的过程。开发者应持续审查和评估他们的代码库中的异味，进行小而专注的更改，这些更改会逐渐提高代码的质量和可维护性。这种方法可以更好地进行风险管理，因为它最小化了在重构过程中引入新错误或问题的可能性。实现这一点的最佳方式是单元测试。它们有助于验证重构后的代码仍然满足其原始要求，并在修改其内部结构或组织后仍然按预期运行。在开始重构过程之前，拥有强大的测试集可以让开发者有信心他们的更改不会对应用程序的行为产生负面影响。这使他们能够专注于改进代码的设计、可读性和可维护性，而无需担心无意中破坏功能。我们将在*第十三章*中探讨单元测试。

总之，“代码异味”这个术语描述的是代码库中表明潜在设计或实现问题的症状。解决代码异味涉及识别和应用适当的设计模式，以及避免可能损害代码质量的反模式。通过理解代码异味的根本原因，并有效地使用模式和反模式，开发者可以将代码库重构得更加易于维护、可读，并能适应未来的变化。持续评估和逐步重构是防止代码异味并确保高质量、高效代码库能够适应不断变化的需求和需求的关键。

# 旧代码

重构旧版 C++代码是一项重大的任务，它有可能为老化的代码库注入新的活力。通常，旧代码是用 C++98 或 C++03 等旧的 C++方言编写的，这些方言没有利用 C++11、C++14、C++17 和 C++20 引入的新语言特性和标准库改进。

现代化中的一个常见领域是内存管理。传统的 C++代码通常使用原始指针来管理动态内存，这可能导致内存泄漏和空指针解引用等潜在问题。此类代码可以被重构为使用智能指针，例如`std::unique_ptr`和`std::shared_ptr`，这些智能指针会自动管理它们所指向的对象的生命周期，从而降低内存泄漏的风险。

另一个现代化机会在于采用 C++11 中引入的基于范围的`for`循环。可以使用更简洁、更直观的基于范围的循环来替换旧的显式迭代器或索引变量的循环。这不仅使代码更容易阅读，还减少了出现偏移量错误和迭代器无效化错误的可能性。

如前几章所述，遗留的 C++代码库通常大量使用原始数组和 C 风格字符串。此类代码可以重构为使用`std::array`、`std::vector`和`std::string`，这些更安全、更灵活，并提供有用的成员函数。

最后，现代 C++通过引入 C++11 中的`std::thread`、`std::async`和`std::future`以及随后的标准中的进一步增强，在提高并发支持方面取得了重大进展。使用特定平台线程或较旧并发库的遗留代码可以从重构以使用这些现代、可移植的并发工具中受益。

让我们从使用`pthread`创建新线程的遗留代码示例开始。此线程将执行一个简单的计算：

```cpp
#include <pthread.h>
#include <iostream>
void* calculate(void* arg) {
    int* result = new int(0);
    for (int i = 0; i < 10000; ++i)
        *result += i;
    pthread_exit(result);
}
int main() {
    pthread_t thread;
    if (pthread_create(&thread, nullptr, calculate, nullptr)) {
        std::cerr << “Error creating thread\n”;
        return 1;
    }
    int* result = nullptr;
    if (pthread_join(thread, (void**)&result)) {
        std::cerr << “Error joining thread\n”;
        return 2;
    }
    std::cout << “Result: “ << *result << ‘\n’;
    delete result;
    return 0;
}
```

现在，我们可以使用 C++11 中的`std::async`重构此代码：

```cpp
#include <future>
#include <iostream>
int calculate() {
    int result = 0;
    for (int i = 0; i < 10000; ++i)
        result += i;
    return result;
}
int main() {
    std::future<int> future = std::async(std::launch::async, calculate);
    try {
        int result = future.get();
        std::cout << “Result: “ << result << ‘\n’;
    } catch (const std::exception& e) {
        std::cerr << “Error: “ << e.what() << ‘\n’;
        return 1;
    }
    return 0;
}
```

在重构版本中，我们使用`std::async`启动一个新任务，并使用`std::future::get`获取结果。计算函数直接返回一个`int`类型的结果，这比在`pthread`版本中分配内存要简单和安全得多。有几个需要注意的事项。对`std::future::get`的调用会阻塞执行，直到异步操作完成。此外，示例使用`std::launch::async`，这确保任务在单独的线程中启动。C++11 标准允许实现决定默认策略：单独的线程或延迟执行。在撰写本文时，Microsoft Visual C++、GCC 和 Clang 默认在单独的线程中运行任务。唯一的区别是，虽然 GCC 和 Clang 为每个任务创建一个新的线程，但 Microsoft Visual C++会重用内部线程池中的线程。错误处理也更简单，因为计算函数抛出的任何异常都将被`std::future::get`捕获。

通常，遗留代码使用围绕`pthread`和其他平台特定 API 的对象封装。用标准 C++实现替换它们可以减少开发者需要支持的代码量，并使代码更具可移植性。然而，多线程是一个复杂的话题，所以如果现有代码有一些丰富的线程相关逻辑，确保它保持完整是很重要的。

现代 C++提供的内置算法可以提高遗留代码的可读性和可维护性。通常，开发者需要检查数组是否包含某个特定值。C++11 之前的语言允许这样做：

```cpp
#include <vector>
#include <iostream>
int main() {
    std::vector<int> numbers = {1, 2, 3, 4, 5, 6};
    bool has_even = false;
    for (size_t i = 0; i < numbers.size(); ++i) {
        if (numbers[i] % 2 == 0) {
            has_even = true;
            break;
        }
    }
    if (has_even)
        std::cout << “The vector contains an even number.\n”;
    else
        std::cout << “The vector does not contain any even numbers.\n”;
    return 0;
}
```

使用 C++11，我们可以使用`std::any_of`，这是一种新算法，用于检查范围中的任何元素是否满足谓词。这允许我们编写更简洁、更具表现力的代码：

```cpp
#include <vector>
#include <algorithm>
#include <iostream>
int main() {
    std::vector<int> numbers = {1, 2, 3, 4, 5, 6};
    bool has_even = std::any_of(numbers.begin(), numbers.end(),
                                [](int n) { return n % 2 == 0; });
    if (has_even)
        std::cout << “The vector contains an even number.\n”;
    else
        std::cout << “The vector does not contain any even numbers.\n”;
    return 0;
}
```

在这个重构版本中，我们使用 lambda 函数作为`std::any_of`的谓词。这使得代码更简洁，意图更清晰。算法如```cpp std::all_of`` ```和`std::none_of`允许清晰地表达类似的检查

记住，重构应该逐步进行，每次更改都要彻底测试，以确保不会引入新的错误或回归。这可能是一个耗时的过程，但就提高代码质量、可维护性和性能而言，其好处可能是巨大的。

# 摘要

在本章中，我们探讨了可以在重构遗留 C++代码中发挥关键作用的一些关键设计模式，包括策略模式、模板方法模式和观察者模式。当谨慎应用时，这些模式可以显著改善代码的结构，使其更加灵活、可维护和适应变化。

尽管我们已经提供了实用的、现实世界的例子来展示这些模式的使用，但这绝对不是一种详尽无遗的处理方式。设计模式是一个庞大而深奥的主题，还有许多更多的模式和变体需要探索。为了更全面地理解设计模式，我强烈推荐您深入研究 Erich Gamma、Richard Helm、Ralph Johnson 和 John Vlissides 所著的奠基性作品《设计模式：可复用面向对象软件元素》，这本书通常被称为*四人帮*书。

此外，为了跟上最新的发展和新兴的最佳实践，可以考虑像 Fedor G. Pikus 所著的《动手学设计模式：使用现代设计模式解决常见的 C++问题并构建健壮的应用程序》和 Anthony Williams 所著的《C++并发实战》这样的资源。这些作品将为您提供更广阔的视角和更深入的理解，了解设计模式在构建高质量 C++软件中扮演的强大角色。

记住，重构和应用设计模式的目标不仅仅是编写出能工作的代码，而是编写出干净、易于理解、易于修改和长期易于维护的代码。

在接下来的章节中，我们将更深入地探讨 C++的世界，特别是关注命名约定、它们在编写干净和可维护的代码中的重要性以及社区建立的最佳实践。
