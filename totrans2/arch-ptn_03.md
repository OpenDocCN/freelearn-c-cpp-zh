# Object-Oriented Software Engineering Patterns

Object-oriented (OO) concepts in software engineering are not new, and let's start this chapter with a brief introduction before we dive into OO design patterns. While you are reading this chapter, look around you; whatever you see is an object: the book, bookshelves, reading lamp, table, chair, and so on. Everything around you can be imagined as an object, and all of them share two primary characteristics, as follows:

*   State
*   Behavior

A reading lamp has *off and on* as states, and *turn on and turn off* as behaviors. Objects may also have many states and many behaviors, sometimes even other objects as well.

**Object-oriented design** (**OOD**) intends to provide modularity, abstraction (information hiding), code reuse, and pluggable (plug and play) and easy code debug.

Grady Booch defined OOD in his book titled *Object Oriented Analysis and Design with Application* as follows:

"OOD is a method of design encompassing the process of object-oriented decomposition and a notation for depicting both logical and physical as well as static and dynamic models of the system under design."

This chapter covers the following elements of OOD:

*   Essential and non-essential elements of OOD
*   Primary characteristics of OOD
*   Core principles of OOD
*   Most common design patterns of OOD
*   Cross-reference of OO design patterns

# Key elements of OOD

There are four key elements of OOD. They are as follows:

*   **Abstraction**: Hiding the complexity and low-level implementation details of internals.
    For instance, you see electrical switch buttons that can toggle on and off, but how it is achieving on and off is not shown to outside world, and in fact, it is not necessary for the common users.
*   **Encapsulation**: Bundling of the data with the methods that operate on that data, preventing accidental or unauthorized access to the data.
    For example, switching off function should turn only the targeted element off, say a reading lamp, and it should not affect any other electrical functions that are part of the same electrical system.
*   **Modularization**: The process of decomposing and making it as modules to reduce the complexity of the overall program/function.
    For example, switch off and on is a common functionality of an electrical system. Switching a reading lamp on and off may be a separate module and decoupled from other complex functions such as switching off washing machine and AC.
*   **Hierarchy**: It is ordering of abstraction and hierarchy of an interrelated system with other subsystems. Those subsystems might own other subsystems as well, so hierarchy helps reach the smallest possible level of components in a given system.

# Additional elements of OOD

There are three additional elements of OOD. They are as follows:

*   **Typing**: Characterization of a set of items. A Class (in object-oriented programming) is a distinct type. It has two subtypes. They are as follows:
    *   Strong Typing
    *   Weak Typing
*   **Concurrency**: Operating system allows Performing multiple tasks or process simultaneously.
*   **Persistence**: Class or object occupies space and exists for a particular time.

# Design principles

This chapter and the following sections cover object-oriented design principles, its characteristics, and the design patterns in detail. Each pattern section covers its need, design considerations, and best practices so that readers get the idea of patterns and its applications.

Let's start with a core principle usually referred to as an acronym "SOLID," in detail.

# Single responsibility principle (SRP) – SOLID

In object-oriented programming style, the *single responsibility* enforces that each class should represent one and only one responsibility and so if it needs to undergo changes, that should be for only one reason, that is, a class should have one and only one reason to change.

When we design a class or refactor a class and if it needs more than one reason to change, split the functionality into as many parts as there are classes and so, each class represents only one responsibility.

Responsibility in this context is any changes to the function/business rules that causes the class to change; any changes to the connected database schema, user interfaces, report format, or any other system should not force that class also to change:

![](img/a2e91c5f-a337-4475-af95-9423ab1932cc.png)

The preceding class diagram depicts a **Person** class having two responsibilities: one responsibility is to greet the user with their last name or surname, and another responsibility is to validate the email. If we need to apply SRP on the **Person** class, we can separate it into two; **Person** has a method greet, and **Email** has email validation.

The SRP applies not only at the class level, but also on methods, packages, and the modules.

# Open and close principle – SOLID

The open and close principle of OO programming suggests that the OO software entities such as classes, methods or functions, and modules, should be open for extensions, but closed for any modifications.

Imagine a class that you never need to change, and any new functionality gets added only by adding new methods or subclasses, or by reusing the existing code, and so we can prevent any new defects to the existing code or functionality.

![](img/07114d55-b27f-400d-8902-18a99699cc7d.png)

The preceding class diagram shows the application of the open and close principle on the **Account** class and its subclasses. The account can be any type, savings or current. A **SavingsAccount** may categorize as **GeneralAccount**, **KidsAccount**, and so on, so we can enforce that Account and other subclasses are available for Enhancements but closed for modifications.

The open and close principle brings benefits of no changes to the code, no introduction of any new defects but perhaps a disadvantage that the existing defects never get addressed as well.

# Liskov substitution principle (LSP) – SOLID

This principle states that any of the child classes should not break the parent class's type definitions or, in other words, derived classes should be substitutable for their base classes.

Let's first understand the violation of substitution principle, and then we see how we can resolve the same by taking our earlier example of account classes as LSP is a prime enabler of OCP:

![](img/f201284f-16ac-4eb5-9417-efae901abc89.png)

Let's assume that withdrawal from kids account is not allowed, unlike general account. As you see in the preceding class diagram, a *withdraw* method in the kids account class is a breach of LSP, so by introducing other withdrawable and non-withdrawable classes inherited from **SavingsAccount** class to handle non-withdrawable behavior, we can get rid of the breach and the subclass does not change the base class behavior:

![](img/fe87bbb2-9dd8-4bcd-813b-be4e1e66cb87.png)

So, the behavior of **SavingsAccount** is preserved while inheriting it for **KidsAccount**. The preceding code snippet proves the same.

# Interface segregation principle (ISP) – SOLID

Imagine that you are implementing an interface of a class pets, and the compiler complains about the non-inclusion of bark method in your **Cat** class; strange, isn't it?

ISP suggests a*ny interface of a class should not force the clients to include any unrequired methods by that client*; in our example, **Cat** does not need to implement bark method, and it is exclusive to **Dog** class:

![](img/1e387677-582b-4bbc-9178-5c645a43eba0.png)

The preceding diagram depicts ISP violation and how to get rid of the same by splitting the **<<IPets>>** interface to represent the **Cat** and **Dog** interface explicitly.

# Dependency inversion principle (DIP) – SOLID

The DIP enforces two points, as listed:

*   Any higher-level modules should not depend on lower-level modules, and both should depend on abstract modules
*   Abstraction of modules should not depend on its implementation or details, but the implementation should depend on abstraction

Please refer to the earlier *Interface segregation principle (ISP) – SOLID* section, and the example classes (Figure 3.5) Pets classes and its abstract classes. **Dog** and **Cat** depend on abstractions (interface), and any changes to any of the underlying implementation do not impact any other implementations.

# Other common design principles

Other common principles are as follows; however, detailing of each principle is not in the scope of this chapter, and we request you to refer to other materials if you need to read more information about those principles:

*   Encapsulate
*   Always encapsulate the code that you think may change sooner or later
*   Composition over inheritance
*   In some cases, you may need the class behavior to change during runtime, and those cases favor composition over inheritance
*   Program for interface (not for the implementation)
*   Bring flexibility to the code and can work with any new implementation
*   **General responsibility assignment software patterns** (**GRASP**)
*   Guides in assigning responsibilities to collaborate objects
*   **Don't repeat yourself** (**DRY**)
*   Avoid duplicate codes by proper abstraction of the common codes into one place
*   **Single layer abstraction principle** (**SLAP**)
*   Every line in a method should be on the same level of abstraction

# OO design patterns

Object-oriented design patterns solve many common software design problems, as follows, that architects come across every day:

*   Finding appropriate objects
*   Determining object granularity
*   Specifying object interfaces
*   Specifying object implementations
*   Programming to an interface, not an implementation
*   Putting the reuse mechanism to work

We will touch upon some of the common problems and how design patterns solve the mentioned glitches in this section and cover OO design patterns in detail.

We can categorize the patterns into three types: creational, structural, and behavioral. Refer to the table at the end of this chapter, which depicts the patterns and its categories as a simple reference before we move ahead with the details.

# Creational design patterns

The creational patterns intend to advocate a better way of creating objects or classes, and its primary focuses are as follows:

*   Abstracting the class instantiation process
*   Defining ways to create, compose, and represent objects and hide the implementation details from the involving system
*   Emphasizing avoiding hard code of a fixed set of behaviors and defining a smaller set of core behaviors instead, which can compose into any number of (complex) sets

Creational design patterns have two basic characteristics: one is that they encapsulate knowledge about which concrete class the system use, and the second is that they hide how the instances of these classes are created and put together.

The class creational pattern uses inheritance for instantiation, whereas object creations delegates it to another object.

The following section deals with each pattern, its general structure, and sample implementation diagram in most of the cases.

# Factory method (virtual constructor)

This pattern suggests to let the subclasses instantiate the needed classes. The factory method defines an interface, but the instantiation is done by subclasses:

![](img/fd0f457d-4c77-4a7d-9ac6-91e5b13a0492.png)

The preceding structure depicts a factory method, and an application uses a factory to create subtypes with an interface.

The benefits of using this are as listed:

*   **Loose coupling**: Separates application from the classes and subclasses
*   **Customization hooks**: The factory method gives subclasses a hook for providing an extended version of an object

The impact of using this is that it creates parallel class hierarchies (mirroring each other's structures), so we need to structure in the best possible ways using intelligent children pattern or Defer identification of state variables pattern.

# Abstract factory (kit)

Abstract factory pattern is intended to provide an interface if we want to create families of related or dependent objects, but without explicitly specifying their concrete classes:

![](img/29f30562-ce33-4a4f-9fe7-b923af2335ee.png)

The preceding class diagram depicts the **AbstractFactory** class structure and a real-time implementation of an abstract factory pattern for an application that combines a different set of (heterogeneous) products from two different groups (**<<Bank>>** and **<<Loan>>**).

The benefits of this are the following:

*   Isolating concrete classes
*   Making exchanging product families easy
*   Promoting consistency among products

Impact is such as; supporting new kinds of the product is difficult.

# Builder

The builder is intended to separate the construction of a complex object from its representation so that the same construction process can create different representations. In other words, use this pattern to simplify the construction of complex object with simple objects in a step-by-step manner:

![](img/43004272-349a-4b1f-9534-d32c97b09ab8.png)

The class diagram depicts a typical builder pattern structure and a sample implementation classes for the **Builder** pattern. The **Builder** (**TextConverter**) is an abstract Interface that creates parts of a product page. The **Concrete Builder** (**AsciiConv**, **TexConv**) constructs and assembles parts by interface implementation, the **Director** (**Reader**) constructs an object with the builder interface, and the **Products** (**AsciiTxt**, **Text**) are under construction complex objects.

The benefits are as listed:

*   Allows changing the internal representation and defines new kind of builder
*   Isolates code for construction and representation
*   Provides finer control over the construction process

Impacts are as listed:

*   Leads to creating a separate concrete builder for each type of product
*   Leads to mutable **Builder** classes

# Prototype

Prototype pattern suggests copying or cloning the existing object and customizing it if needed rather than creating a new object. Choose this pattern when a system should be independent of its products creation, compose, and representation:

![](img/3a70ee05-e5c0-4624-8ff7-eacbc650719a.png)

We can create a copy of **PublicProfile** (limited information) or **FullProfile** at runtime. Those two classes share a few combination of states, so it is good that we design as a prototype.

Let's take a look at its benefits:

*   Adding and removing products at runtime
*   Specifying new objects by varying values and structures
*   Reduced subclasses
*   Dynamic class configuration to an application

The impact is, each subclass must implement clone operation, and it is not possible to clone circular reference classes.

# Singleton

This pattern suggests that you create one and only one instance and provide a global point of access to the created object:

![](img/c118a358-bda8-4199-8e9f-355eab6c9257.png)

The DB connection in the preceding diagram is intended to be a singleton and provides a getter for its only object.

Here are its benefits:

*   Controlled access to a sole instance
*   Reduced namespace
*   Flexibility to refinement of operations and representations
*   More flexible than class operations

Impacts are as follows:

*   Carry states for the whole lifetime of the application, creating additional overhead for unit tests
*   Some level of violation of single responsibility principle
*   By using singleton as a global instance, it hides the dependencies of the application; rather, it should get exposed through interfaces

# Structural design patterns

The structural patterns provide guidelines to compose classes and objects to form a larger structure in accordance with the OO design principles.

The structural class pattern uses inheritance to compose interfaces or implementations, and structural object patterns advocate ways to compose objects and realize the new functionality.

Some focus areas of Structural design pattern are as follows:

*   Providing a uniform abstraction of different interfaces (Adapter)
*   Changing the composition at runtime and providing flexibility of object composition; otherwise, it is impossible with static class composition
*   Ensuring efficiency and consistency by sharing objects
*   Adding object responsibility dynamically

The following section describes each structural pattern with standard structure and sample implementation structure as a diagram as well.

# Adapter class (wrapper)

Convert one interface of a class into another interface that the client wanted. In other words, the adapter makes heterogeneous classes work together:

![](img/2e5a4a22-ab05-4375-a8e4-82815b93cc4c.png)

The preceding class diagram depicts an adapter called **OnlineLinkedAccounts** that adopts a savings account's details and a target interface called **credit card details**, and combine the results to show both account numbers.

# Adapter (object)

An adapter object relies on object composition, and when we need to use several of the existing subclasses, we can use object adapter to adapt the interface of the parent class:

![](img/8ab6bdc9-06c6-4546-bb31-8b76d6df9311.png)

The preceding diagram depicts the formal structure of an **Adapter**.

These are the benefits:

*   Saves time during development and testing by emulating a similar behavior of different parts of the application
*   Provides easy extensions for new features with similar behaviors
*   Allows a single adapter works with many adaptees (adapter object)

Impacts are as follows:

*   Leads to needlessly duplicated codes between classes (less usage of inherited classes' functionalities)
*   May lead to nested adaptions to reach for intended types that are in longer chains
*   Make it more difficult to override adaptee behavior (adapter object)

# Bridge (handle/body)

Bridge pattern intent is to decouple the abstraction from its implementation, so abstraction and implementation are independent (not bound at compile time, so no impact to the client):

![](img/aeca4612-f975-405e-8cb3-be0abefe969d.png)

The benefits are as mentioned:

*   Decoupling interfaces from the implementation
*   Configuring the implementation of an abstraction at runtime
*   Elimination of compile-time dependency
*   Improved extensibility
*   Hiding implementation details from the client

The impact is, introducing some level of complexity.

# Composite

**Composite** objects let clients treat individual objects and composition of objects uniformly. **Composite** represents the hierarchies of objects as tree structures.

![](img/5d9d53e9-be35-4e61-aa8d-f110afc41b83.png)

The preceding diagram depicts the standard structure of the **Composite** pattern and an implementation of a part-whole hierarchy (employee part of agent, **Accountant**, and teller), and to the **Client,** all objects are **Composite** and structured uniformly.

These are the benefits:

*   It simplifies the client code by hiding the complex communications (leaf or composite component)
*   It is easier to add new components, and client does not need a change when new components get added

The impact is such that it makes the design overly general and open as there are no restrictions to add any new components to composite classes. 

# Decorator

The decorator pattern attaches additional responsibilities to an object dynamically. It provides an alternative way (by composition) to subclass and to extend the functionality of an object at runtime.

This pattern creates a decorator class by wrapping the original class to provide additional functionalities without impact to the signature of methods.

![](img/59103344-aea2-49f7-96eb-2485e7f9787f.png)

Observe the preceding diagram as it depicts invoice functionalities extended by composition dynamically (runtime).

Let's list the benefits:

*   It reduces time for upgrades
*   It simplifies enhancing the functionalities from the targeted classes and incorporates behavior into objects (changes class responsibilities, not the interface)

Impacts are as follows:

*   It tends to introduce more look-alike objects
*   It leads to debugging difficulties as it is adding functionality at runtime

# Façade

Façade suggests providing a high-level interface that unifies set of interfaces of subsystems, so it simplifies the subsystem usage.

![](img/ba70359e-9d2d-48ba-aa1a-05d6d55280c4.png)

A sample implementation of a service façade as in the preceding diagram, the session subsystem are unified with session façade (local and remote).

Let's look at the benefits:

*   It promotes loose coupling (between clients and subsystems)
*   It hides complexities of the subsystem from the clients

The impact is such that it may lead to façade to check whether the subsystem structure changes.

# Flyweight

**Flyweight** suggests using the shared support of a vast number of fine-grained objects. We can use the **Flyweight** pattern to reduce the number of objects created (by sharing) and thereby reduce the memory footprint and improve the performance.

![](img/c62ab08b-407c-4a43-8180-ec6d4cb7c37a.png)

The preceding diagram depicts the general structure of the **Flyweight** pattern and a sample implementation. Consider a massive object that is shared across printer and a screen; **Flyweight** is a good option and can be cached as well (say for printing multiple copies).

Here are the benefits:

*   It leads to good performance due to reduction in the total number of instances (by shared objects)
*   It makes implementation for objects cache easy

The impact is such that it may introduce runtime costs associated with transferring, finding, or computing foreign (extrinsic) state.

# Proxy

The proxy pattern suggests providing a placeholder (surrogate) for another object to control and get access to it. It is the best fit for lazy loading of objects (defer the creation and initialization until we need to use it).

![](img/5a8b6703-53df-43f2-9199-0702c4359698.png)

The preceding diagram shows a sample implementation of a proxy pattern for a payment class, and the payment can be either by check or by pay order. However, the actual access would be to **DebitAccount** object, so **PayOrderProxy** and **CheckProxy** are both surrogates for Debit Account.

The following are the benefits:

*   It introduces the right level of indirections when accessing an object (abstraction of an object that resides in a different space)
*   Creating objects on demand
*   Copy-on-write (may reduce the copying of heavy objects if not modified)

The impact is such that it can make some implementations less efficient due to indirections.

# Behavioral patterns

Behavioral patterns provide guidelines on assigning responsibilities between objects. It does help with ways to implement algorithms and with communication between classes and objects.

Behavioral pattern focuses on the following characteristics:

*   Communication between objects and classes
*   Characterizing the complex control flow; flow of control in software programming (otherwise, it is hard to follow at runtime)
*   Enforcing object composition rather than inheritance
*   Loose coupling between the peer objects, and at the same time, they know each other (by indirections)
*   Encapsulating the behavior in an object and delegating request to it

There are various design patterns available to enforce the above said behavioral focusses and characteristics. We will see details of those behavioral patterns in this section . We also provided a sample implementation structure as a diagram for some of the patterns.

# Chain of responsibility

This pattern suggests avoiding coupling the client object (sender of requests) with the receiver object by enabling objects (more than one) to handle the request.

![](img/23bb8393-93ad-48be-9fc8-cf469334f9b6.png)

The preceding diagram depicts the typical structure of the chain of responsibility; the handler is the interface to define the requests and optionally express the successors along with concrete handlers that can handle the requests and forwards the same if needed.

Here's a list of the benefits:

*   Reduced coupling (objects do not know which other objects handle the requests)
*   Additional flexibilities in responsibilities assignments (of objects)

The impact is, no handshakes between the request handlers, so no guarantee of handling the request by other objects, and it may fall off from the chain unnoticed.

# Command (action/transaction)

This pattern suggests encapsulation of requests as an object, parameterizing clients with different requests; it can placed over message queues, can be logged, and supports undo operations.

![](img/61a4b414-aef0-480e-9e04-58fffeb712da.png)

The preceding diagram depicts the structure of a command pattern and a sample implementation for a stockbroker application classes. **<<StockOrder>>** interface is a **Command**, and **Stock** concrete class creates requests. **Buy** and **Sell** are concrete classes implementing the **<<StockOrder>>**. The **StockBroker** is an invoker, and its objects execute specific commands depending on the type that it receives.

Here are the benefits:

*   Encapsulation of object facilitates the changing of requests partially (by changing a single command) and no impacts to the rest of the flow
*   Separates the invoking object from the actual action performing object
*   Easy to add new commands without any impact to the existing classes

The impact is, the number of classes and objects increases over time or depends on the number of commands (concrete command implementations).

# Interpreter

This pattern suggests defining grammar along with an interpreter that uses representations so that the system can interpret any given sentences of a language.

![](img/69393cc5-e685-4e6f-92fd-3063eb5493fe.png)

Abstract expression or regular expression declares interpret operation, terminal expressions or literal expressions implements symbols in the grammar, and non-terminal expressions (alternate, sequence, repetition) has nonterminal symbols in the grammar.

Let's look at the benefits:

*   It is easy to change and extend the grammar
*   Implementing the grammar is easy as well
*   Helps introduce new ways to interpret expressions
*   Impacts
*   Introduces maintenance overhead for complex grammars

# Iterator (cursor)

This pattern suggests providing an approach to sequentially access the elements of an aggregate object and, at the same time, hide the underlying implementations.

![](img/df228b63-eada-4f06-a3da-c48f78505152.png)

The preceding diagram depicts the structure of the iteration pattern in which the iterator interface defines traversing methods, and the concrete iterator implements the interface. Aggregate defines an interface for creating an iterator object, while a Concrete aggregate implements the aggregate interface to create an object.

Here are the benefits:

*   It supports variations in the aggregate traversals
*   Iterators simplify the aggregate interfaces
*   It may have null iterators and helps handle boundary conditions better
*   Impacts
*   It may introduce additional maintenance cost (dynamic allocation of polymorphic iterators)
*   It may have privileged access and thus introduces complexities to define new traversal methods in iterators

# Mediator

The **Mediator** pattern advocates defining ways of interactions between encapsulated objects without depending on each other by explicit reference.

![](img/228791ae-4bd0-4d9d-83e8-96d04fffda93.png)

The preceding diagram is a typical structure of the **Mediator** pattern, where **Mediator** or dialog director defines an interface to communicate with other colleague objects; concrete mediator implements cooperative behavior by coordinating colleague objects.

Let's look at the benefits:

*   Limits subclassing (by localizing behavior and restricting the distribution of behaviors to several other objects)
*   Enforcing decoupling between colleagues objects
*   Simplifying object protocols (replaces many-to-many interactions to one-to-one)
*   Providing clear clarification on how objects should interact

Impacts is centralized control, leading to more complex and monolithic systems.

# Memento

This pattern suggests capturing and externalizing an object's internal state without violating encapsulation principles; so, we can restore the captured object.

![](img/f3d3a1bf-46d4-4717-967a-741d7a23b699.png)

The preceding diagram depicts the structure of the memento pattern and a sample implementation for a calculator application. The **Caretaker** interface helps restore the previous operation that's handled in the **<<Calculator>>** concrete class.

These are the benefits:

*   It preserves encapsulation boundaries by exposing information limited to the originator
*   It simplifies the originator

Impacts are as follows:

*   Memento implementation might be expensive, as it needs to copy large amounts of data to store into the memento
*   It may be difficult to implement (through some programming languages) and ensure that only the originator is accessing the memento's state
*   It might incur hidden storage and maintenance costs at the caretaker implementations

# Observer (dependents/publish/subscribe)

The **Observer** pattern suggests that when one object changes the state, it notifies its dependents and updates automatically. When implementation is in need of one-to-many dependencies, you would want to use this pattern.

![](img/5c2b367d-7e5a-4e01-98fe-505d89ac8d33.png)

The preceding diagram depicts **Observer** pattern structure and a sample implementation of the same for a publications app; whenever an event occurs, subscribers need to be informed. The subscribers have a different mode of publishing (SMS, print, and emailing) and may need to support new modes as well in the future, so the best fit is **Observer**, as we just saw.

Let's go through its benefits:

*   Enables easy broadcast of communication
*   Supports loose coupling between objects as it's capable of sending data to other objects without any change in the subject
*   Abstract coupling between subject and observer (changes in the observer do not impact subject)
*   Can add or remove Observers any time

Impacts are as follows:

*   Accidental or unintended updates impact the system heavily as it cascades to the observer down the layers
*   May lead to performance issues
*   Independent notifications may result in inconsistent state or behavior (no handshakes)

# State (objects for states)

These allow an object to alter its behavior when its internal state changes, and it appears as the class changes.

Use state pattern when an object's behavior depends on its state and change at runtime depends on that state.

![](img/76cdf2c1-cc78-4e72-9c61-6f55be131808.png)

The diagram depicts both structure of State pattern and a sample implementation; Context class carries states, and **Off** and **On** classes implement State interface so that context can use the action on each concrete class's off/on.

Listed are the benefits:

*   Suggest localizes state-specific behavior and partitions behavior for different states (new states and transitions can be added easily by subclass definitions)
*   Makes state transitions explicit
*   State objects are shareable

The impact is, it may make adding a new concrete element difficult.

# Strategy (policy)

Strategy pattern, also known as policy, defines a family or set of algorithms, encapsulates each one, and make them interchangeable. Strategy lets the algorithm vary independently of the clients that use it. When a group of classes differs only on their behavior, it is better to isolate the algorithms in separate classes and provide the ability to choose different algorithms at runtime.

![](img/a5fbbdbe-a91b-4aa1-afc5-c375524ae362.png)

The preceding diagram shows the strategy structure, and implementation of sorting algorithms (as a family) and depends on the input depends on the volume for sort, then the client can use the intended algorithm from the Concrete strategy sorting classes.

The benefits are as listed:

*   Enables open and closed principle
*   Enables large-scale reusability
*   Eliminates conditional statements (leads to clean code, well-defined responsibilities, easy to test, and so on)

Impacts are as follows:

*   Clients need to be aware of different strategies and how they differ
*   Communication overhead between strategy and context
*   Increased number of objects

# The template method

This suggests providing a skeleton of an algorithm in operation, and deferring a few steps to subclasses. The template method lets subclasses redefine a few specific actions of a defined algorithm without changing the algorithm structure.

![](img/4809418f-9d85-4716-a069-c460fb93524e.png)

The following are the benefits:

*   Fundamental technique for code reuse
*   Allows partial implementation of business process while delegating implementation-specific portion to implementation objects (flexible in creating prototypes)
*   Helps implement the Hollywood principle (inverted control structure, *Don't call us, we will call you*)

Impacts are as follows:

*   Sequence of flow might lead to confusion
*   High maintenance cost and impacts are high on any changes to the code

# Visitor

The visitor pattern represents an operation performed on the objects. It lets us define a new operation without changing the class elements on which it operates. In simple words, we use the visitor class to alter the execution of an algorithm as and when the visitor varies.

>![](img/7b0e13b8-3355-4577-a5c3-58d815229791.png)

Here are the benefits:

*   Adding new operations over an object structure is straightforward and easy (by adding a new visitor)
*   Visitor separates unrelated operations and gathers related operations

Impacts are as follows:

*   The visitor class hierarchy can be difficult to maintain when a new concrete element class gets added
*   Implementation often forces to provide public operation that accesses an element's internal state (leads to compromising its encapsulation)

# Concurrency patterns

In software paradigm, the ability to perform multiple tasks at the same time (concurrency) by a software application is a critical factor; most software applications have some or other sort of concurrency. Keeping this in mind, let's briefly touch upon on a few concurrency patterns here, as other chapters in this book cover many (concurrency) related patterns in detail.

# Concurrency design pattern

In many situations the automated system may have to handle many different events simultaneously called concurrency. OOP provides an adequate means (abstraction, reusability, sharing of distributed persistent data, parallel executions and so on) of dealing with concurrency. This section will cover few concurrency patterns in brief.

# Producer-consumer

The producer-consumer pattern decouples the produce consume data processes. The process may handle data at different rates. Producer and consumer pattern's parallel loops are broken down into two categories as those that produce data and those that consume the produced data.

Data queues are used to communicate data between loops in the producer/consumer design pattern. These queues are offered data buffering between the producer and consumer loops.

# Active object

The active object pattern enforces decoupling of method execution from the method invocation and so enhances the concurrency and simplifies synchronized access to the objects that reside in their (own) threads of control.

We use this pattern where an application handles multiple client requests simultaneously to improve its quality of service.

# Monitor object

This pattern suggests synchronization on concurrent method execution to ensure that only one method runs within an object at a time. Monitors also allow an object's methods to execute scheduled sequences cooperatively.

We use this pattern (implement synchronization) when multiple threads are invoking methods of an object that modify its internal state. Contrary to active objects, monitor object belongs to the groups of passive objects; monitors are not having its (own) thread of control.

# Concurrency architectural pattern

**Half-sync**/**Half-async**: In concurrent systems, decoupling of synchronous and asynchronous service processing brings programming simplicity without reducing the performance. Half-sync/Half-async introduces two intercommunicating layers, one for synchronous and another for asynchronous service processing, with a queuing layer in-between.

This pattern enables the synchronous and asynchronous processing services to communicate with each other and helps those processes to decompose into layers.

**Leader**/**Followers**: If we need an efficient concurrency model where multiple threads need to take turns sharing a set of event sources that detect, de-multiplex, dispatch, and process event-sources' service requests, then the best choice is to implement the Leaders/Followers pattern in our system.

The aim of this pattern is to provide an elegant solution to process multiple events concurrently, such as in multithreaded server applications.

# Summary

The design patterns have evolved since 1992, and even today, it is inevitable in solving many software design problems in a proven technique and practices called design patterns. It is not difficult to see any specific pattern as a solution or technique that can be analyzed, implemented, and reused, but it is difficult to characterize the problem it solves and the context in which it is the best fit. It is critical to know the purpose of the patterns, as it helps understand the existing design of any given system.

With this chapter, we touched upon the key elements of OOD, abstraction, encapsulation, modularization, and hierarchy along with a few additional items such as typing, concurrency, and persistence.

Also, we discussed the design principles, hoping that the readers get a *SOLID* understanding of what OO principles offer to OO software designers. We believe that the SOLID principles are the fundamental training material for anyone who wants to step into software design and development even in today's world.

We touched upon three broad categories of OO design patterns: creational, structural, and behavioral. We also discussed the benefits and impacts of each pattern so that the readers will be able to easily characterize the problems it solves and the context it best suits as a software solution.

We also added a section, hoping readers to get a fair amount of introduction about concurrency (design and architectural) patterns as well.

# References

The following table refers to cross-reference of OO software design patterns:

![](img/96ce5f1e-3d60-4dea-8db9-524ee9ace6a7.png)

Reference books are as follows*:*

*   Design Patterns: Elements of Reusable Object-Oriented Software by Erich Gamma, Richard Helm, Ralph Johnson and John Vlissides
*   *Object-Oriented Analysis and Design with Applications (2nd Edition)* by Grady Booch

Other references for this chapter:

*   [http://www.oodesign.com/](http://www.oodesign.com/)
*   [https://www.tutorialspoint.com/design_pattern](https://www.tutorialspoint.com/design_pattern)
*   [https://sourcemaking.com/design_patterns/](https://sourcemaking.com/design_patterns/)
*   [http://www.blackwasp.co.uk/GofPatterns.aspx](http://www.blackwasp.co.uk/GofPatterns.aspx)
*   [http://www.mif.vu.lt/~plukas/resources/DPBook/](http://www.mif.vu.lt/~plukas/resources/DPBook/)
*   [www.dzone.com](http://www.dzone.com)
*   [http://www.javaworld.com](http://www.javaworld.com)
*   [https://sudo.ch/unizh/concurrencypatterns/ConcurrencyPatterns.pdf](https://sudo.ch/unizh/concurrencypatterns/ConcurrencyPatterns.pdf)
*   [http://www.cs.wustl.edu/~schmidt/POSA/POSA2/conc-patterns.html](http://www.cs.wustl.edu/~schmidt/POSA/POSA2/conc-patterns.html)
*   [https://en.wikipedia.org/wiki/Concurrency_pattern](https://en.wikipedia.org/wiki/Concurrency_pattern)