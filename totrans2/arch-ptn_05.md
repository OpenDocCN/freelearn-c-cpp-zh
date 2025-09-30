# Domain-Driven Design (DDD) Principles and Patterns

Most of the commercial software application is created with a set of complex business requirements to solve the specific business problems or needs. However, expecting all the software developers/architects to be experts on business domains and expecting them to know entire business functions is also impractical. On the other side, how do we create software that brings value and get consumers with automated business needs to use the software? Software applications cannot just be a showpiece of technical excellence, but in most cases, they also have to have a real ease of automated business excellence. The domain-driven design and models are the answers to our questions.

This section will cover most of DDD aspects and patterns that can help successful implementations of DDD-based software.

![](img/b2d2b6ab-d5ef-40d4-ad36-9b48065c3518.png)

The preceding diagram is an attempt to visualize a domain-driven software model driven by collaborated effort from domain and technology experts.

DDD concepts, principles, and patterns bring technology and business excellence together to any sophisticated software applications that can be created and managed. DDD was coined by Evan and most of the content of this chapter is the influence of his book *Domain-Driven Design - Tackling Complexity in the Heart of Software* and also from the book *Pattern-Principles-And Practices* by Scott Millett and Nick Tune.

This section intends to cover a few essential aspects of DDD and also discuss a few common domain-driven design patterns in detail.

# Principles, characteristics, and practices of DDD

Before we delve into various design patterns, let us touch upon the fundamental principles of DDD, then few primary characteristics constituents, and also few best practices that help teams to adopt and follow DDD.

# Principles

The fundamental principles of DDD are described in the following sections.

# Focusing on the core domain

This principle suggests product development teams to focus more on the core domain, that is, the parts that are most important to a business and which need more attention than others. So, we need to identify the core domain by distilling and decomposing a big problem domain into subdomains. For instance, when designing a retail banking software, we should focus on the credit and debit accounting instead of the manufacturing and distribution of credit cards and debit cards as they support functions and they can be outsourced as well.

# Collaborate and learn

As we mentioned in the introduction section, software experts may not know the domain, and the domain analysts may not know the technology and software implementations. So, collaboration and learning from each other is inevitable for DDD aspects, without which the software design or development won't happen at all. For instance, to develop a back office software application for an investment bank, the risk management experts and Software experts need to work together to learn the systems, applicability, usability, banking customer's intentions, and so on.

In recent days, traditional banks are collaborating with financial technology startups aka fintech, as they see significant benefits of data analytics, AI, and machine learning into core banking systems as they would be able to take accurate decisions, innovate faster along with solving banking industry's everyday problems.

# Model the domain

As we now understand the collaboration and learn principle from the previous section, the collaboration, deep learning, and get insights of the core domain along with fundamental functions, this is inevitable. The output expected out of model the domain principle is a domain model, which is well-organized and structured knowledge of the problem in the core domain space along with fundamental concepts, vocabulary, issues, and relationships among the involved entities. You can seek contributions from different stakeholders such as analysts, domain experts, business partners, tech savvy users, and core developers and build these domain models, so everyone in a team understands the functional concepts and definitions and also how the current problem is tackled and solved.

# Evolve

Another critical aspect of the domain model is evolution. Domain models need to evolve over time through iterations and feedback. The design team starts with one significant problem and traverses through different scopes of the core domain along with generated models with incremental changes iteratively. It is critical as models need to adjust to feedback from domain experts while delivering domain models dealing with complexity.

# Talk in ubiquitous language

Collaborating, learning, and defining a model brings a lot of initial communication barriers between software specialists and domain experts. So, evolving domain models by practicing the same type of communications (discussions, writings, and diagrams) within a context is paramount for successful implementations, and this sort of conversation is called ubiquitous language. It is structured around the domain model and extensively used by all the team members within a bounded context. It should be the medium or mode to connect all the activities of the team during the development of software.

The design team can establish deep understanding, and connect domain jargons and software entities with ubiquitous language to keep discovering and evolving their domain models.

# Characteristics

The following characteristics are the primary constituents and may serve as a glossary of items that we will discuss in this chapter. You will see that many of these are factored into the patterns that we present in this section:

*   **Domain model**: Organized and structured knowledge related to the specific problem.
*   **Bounded context**: A system fulfils the fundamental goals of the real-world complex business problems, provides a clear and shared understanding of what can be consistent, and what can be independent.
*   **Entities**: These are mutable objects, which can change their attributes without changing their identity (for example, the employee's ID doesn't change even when their email ID, address, and name changes).
*   **Value objects**: These are immutable objects (unlike entities), distinguishable only by the state of their properties. The equality of value objects is not based on their identity. (Two location objects can be the same by their long and latitude values.)
*   **Encapsulation**: Fields of an object are exposed only for private access, in other words, detected only through accessor methods (setters and getters).
*   **Aggregate**: This is a collection of entities (for example, a computer is an aggregate of entities such as software and hardware). The aggregate may not work without those objects.
*   **Aggregate root**: This is an entry point to aggregates, and only known reference to any outside object. This helps to create the precise boundary around aggregates.

# Best practices

We have listed a few best practices for a team that intends to dwell in DDD for their software product development:

*   Gather requirements and capture required behaviors
*   Focus on what stakeholders want, when, and why
*   Distill the problem space
*   Be a problem solver first, technologist comes second
*   Manage the complexity with abstraction and create subdomains
*   Understand the reality of the landscape with context maps and bounded contexts
*   Model a solution, tackle ambiguity, and carve out an area of safety
*   Make implicit concepts explicit

# DDD patterns

In this section, we will browse through a set of patterns to build enterprise applications from the domain models. Applying these design patterns together with OO concepts to a system helps meet the business needs.

This section covers significant aspects of DDD design patterns grouped as strategic design patterns and tactical design patterns of DDD.

# Strategic patterns

The primary aim of this group is to bring understanding and consensus between the business and the software development teams with more emphasis on business interests and goals. The strategic patterns help software development team members focus on what is more important and critical to the business by identifying a core domain. The core domain is a particular area of the company or even a specific slice that is critical.

Few primary constituents of strategic patters are ubiquitous language, domain, subdomain, core domain, bounded context, and a context map. We will see how one can integrate the disparate systems via the strategic design patterns such as bounded context, messaging, and REST discussed in this chapter with those constituents.

# Ubiquitous language

A model acts as a universal language to manage communication between software developers and domain experts. The following table shows the example of ubiquitous languages and their equivalent pseudo code:

| **Ubiquitous language** | **Equivalent pseudo code** | **Comments** |
| We administer vaccines | `AdministerVaccines {}` | Not a core domain—need some more specific details |
| We administer flu shots to patients | `patientNeedAFluShot()` | Better, may be missing some domain concepts |
| Nurse administers flu vaccines to patient in standard doses | `Nurse->administer vaccine(patient, Vaccine.getStandardDose())` | Much better, and may be good to start with |

# Domain, subdomain, and core domain

Domain refers to a problem space that software teams try to create a solution for, and represents how the actual business works. The vaccines example from the table can be seen as domain, with the end-to-end flow, managing vaccinations, preventive medicines, dosages, side effects, and so on. Core domain is the core business that the organization does not want to outsource. So, the core domain here in this context is vaccination, and other functions like patients management, cost of vaccines, vaccination camps, and so on are subdomains and are outside of the core domain. Core domains interact with subdomains.

# Bounded contexts

Bounded contexts are distinctive conceptual lines that define the boundaries and separate the contexts from other parts of the system. A bounded context represents refined business capabilities, and it is the focus of DDD. It deals with large distributed models and teams by dividing them into different bounded contexts and being explicit about their interrelationships.

![](img/4ca6eb76-6a72-46f1-83ef-1d3175cd3572.png)

Before we go deeper into patterns, let's refresh the idea about bounded contexts. The preceding diagram depicts account in both contexts; though the account doesn't differ, the contexts do. The following sections deal with patterns that help integrate the bounded contexts for any DDD solution.

# Integrating bounded contexts

The Bounded contexts help in identifying the  relationships between subsystems and so one can choose the communication methods between those subsystems. Selecting appropriate communication and establishing relationships with the established communication is the responsibility of the designers, which helps them too to ensure there is no impact on project delivery timelines and efficiency. An example of integration and establishing communication reflecting explicit models could be integrating a payment system with an e-commerce sales system. Choosing the communication method is critical, and we will see more of integrating bounded contexts in the following sections.

# Autonomous bounded context

To ensure atomicity, design loosely coupled systems with fewer dependencies; solutions can also be developed in isolation.

# The shared-nothing architecture

While guaranteeing bounded contexts to be self-reliant, retaining the integrity of the bounded context is also critical. The shared-nothing pattern suggests that each bounded context has its own data stores, codebases, and developers, as shown in the following diagram:

![](img/f8f15882-8a22-4a16-b5b1-7921ad3f16fd.png)

As each bounded context is physically isolated, it can evolve independently for internal reasons, resulting in uncompromised domain model with super-efficient and faster delivery of business values.

# Single responsibility codes

It's a best practice to partition the software systems according to their business capabilities, that is, by isolating separate business capabilities into different bounded contexts. For example, the shipping code of the business is not affected by a new shipping provider that got added to Sales.

# Multiple bounded contexts (within a solution)

Depending on the code (language), deployments, and infrastructure, there are situations where different bounded context resides in the same code repository or a solution with combined contexts to depict one big picture of full business use cases.

![](img/40bf1981-ce39-40b0-b26b-c53c86c066fe.png)

To maintain the different contexts within a solution, this pattern suggests keeping namespaces distinct or recommends projects to keep bounded contexts separate.

# Adoption of SOA principles

Build highly scalable systems using DDD with SOA concepts and patterns. Build bounded context as SOA services to solve the technical and social challenges (integrating teams and developing at a high velocity) of bounded context integration. Please refer to [Chapter 7](45460494-ac40-47e3-9d76-731dd2a48e12.xhtml), *Service-Oriented Architecture (SOA)*, for more details on SOA's principles and practices.

# Integrating with legacy systems

Legacy systems is always a case in the real world, and they come with exciting challenges while we try to incorporate the latest industry improvements into them. In DDD, this problem is more interesting to address as there are many handy patterns available that help limit the impact of the legacy on the other parts of the system, manage complexity, and save designers from having to reduce explicitness (against DDD philosophy) of their new code to integrate into legacy modules or components.

We will touch upon bubble context, autonomous bubble context, and expose legacy systems as services in this section.

# The bubble context

If a team wants to start applying the DDD to the legacy systems but it is not yet familiar with DDD practices, then the bubble context pattern can be considered. As the bounded context in the legacy may be an isolated codebase, the bubble context pattern provides clarity, and directions that to the team to create domain models and evolve as well. The bubble context reflects the best of the DDD philosophy of iteration, and it progresses by having full control over the domain model.

It is considered as the best fit to facilitate frequent iterations and get insights even when legacy code is involved.

![](img/51d152a0-ea88-4b0d-a18e-1d7c174f9e7c.png)

When you need to integrate with legacy code but do not want to create any dependency or tight coupling with a legacy system, as bubble context does, this pattern suggests using an anonymous bubble called **autonomous bubble context**. Bubble context gets all its data from the legacy system, whereas the autonomous bubble context has its own data store and is able to run in isolation of the legacy code or other bounded contexts.

![](img/eb068434-670e-4196-8d98-7ed13058de43.png)

The preceding diagram depicts the autonomous bubble context, and you may notice that the bubble context has dependencies with legacy context. However, the autonomous bubble context has its own storage, and so it can run in isolation.

# The anti-corruption layer

An isolating layer talks to the other systems through its existing interface with little or no modifications (to the other systems) and provides clients with the functionality of their own domain. This layer is responsible for translating communication between the two models in both directions, as needed.

# Expose as a service

It may be a good idea to expose legacy system as a service, especially when the legacy context needs to be consumed by multiple new contexts. This pattern is also known as the **open host pattern**.

![](img/f38594dc-1293-4a6c-9a0a-7f1feb3385bd.png)

Each new context still has to translate the response from the legacy to its internals; however, with simplified open host APIs, one can mitigate the translation complexity.

With this pattern, there is a need for some modifications to the legacy context (unlike the bubble context); also, standardization of consumable API SLAs may be challenging as it has multiple consumers.

We can clearly justify that a lot of legacy systems in the real world would like to adopt DDD; however, with the lack of right patterns and given the cost and impacts, there are genuine reasons and hesitations to move toward DDD. Recognizing and harnessing these models should ease the situations and encourage organizations to adopt DDD for their legacy systems and progress toward faster delivery.

# Distributed bounded context integration strategies

Distribution is inevitable in the modern world for various reasons, and primarily for system abilities such as availability, scalability, reliability, and fault tolerance. This section briefly touches upon a few integration strategies for the distributed bounded context, such as Database integration, Flat file integration, Messaging, and REST. We will cover how those patterns help integeratting distributed bounded contexts. Also, we will see (briefly) how reactive solutions help in integration strategies.

# Database integration

The database integration pattern is one of the conventional approaches of using a single data source that lets an application write to a specific database location and lets another application read from it. The access by another application can be made as polling with some frequency. This pattern might come in handy for prototypes or even for **most viable product** (**MVP**) delivery.

![](img/151448d8-35a0-4240-b087-b5c825f79970.png)

The preceding diagram depicts an example of database integration, where the sales team inserts the records and the billing context polls to the same data source. If it finds the sales record, it processes and updates the same row.

While this pattern has advantages of loose coupling, it also has a few drawbacks, such as single point of failure, and needs a mechanism for efficient fault handling and so on. DB down scenario is a SPOF example, and to mitigate, one may need to go with a clustered DB, buy more hardware to scale, or consider the cloud infrastructure, and so on.

# Flat file integration

The flat file integration pattern is similar to database integration; however, instead of having a database to integrate two components, it suggests using flat files. The updates, inserts, and polling are needed just as we would in another pattern, but this is a little more flexible. However, this comes with some disadvantages like managing the file formats, concurrency, and locks, among other things, would need more involvement and effort, leading to scalability and reliability issues.

![](img/8bce9704-1c8d-4154-a20c-5a9ee448daf4.png)

This diagram is the sample implementation for flat file integration and involves polling, update, and delete.

# Event-driven architecture and messaging

Messaging and event-driven architecture pattern bring the best out of modeling communication between bounded contexts with distributed systems. This section under DDD intends to ensure you understand the significance of EDA and messaging patterns within the context of DDD. And also to emphasize the benefit of implementing asynchronous messaging and EDA patterns for communication between the contexts. The benefits include increased reliability even on failures of subsystems. We have covered most of the EDA and messaging patterns well and in-depth in [Chapter 8](dd57ac86-dadf-486b-9ecd-068e1f8ffc59.xhtml), *Event-Driven Architecture Patterns,* and [Chapter 9](45854889-267b-45bb-b951-a54c22f5d850.xhtml), *Microservices Architecture Patterns*, and we encourage you to refer to those chapters and get insights about event-driven and messaging patterns.

# Tactical patterns

Tactical patterns help manage complexities and provide clarity in behaviors of domain models. The primary focus of these patterns is to protect models from corruption by offering protection layers.

In this section, we will touch upon the few of the common patterns that help in creating object-oriented domain models.

At the end of this section, we will also briefly cover the emerging patterns of event sourcing and domain events.

# Patterns to model the domain

This section will discuss few tactical patterns, and explain how they represent the policies and logic within the problem domain. They express elements of models in the code, the relationship between the objects and model rules, and bind the analysis details to the code implementation.

We will discuss the following patterns in details:

*   Entities
*   Value Objects
*   Domain Services
*   Modules
*   Aggregates
*   Factories
*   Repositories

The following diagram depicts various tactical patterns and their logical flow:

![](img/79db211f-8530-4ad0-b25d-4283d0b24f4b.png)

# Entities

As stated in the introduction section, an entity is a mutable object. It can change its attributes without changing its identity. For example, a product is an entity, which is unique and won't change its ID (distinctiveness) once it is set.

However, its price, description, and so on, can be changed as many as times it needs to.

![](img/96f748f2-ac41-4feb-b6d5-d5376530207c.png)

The preceding diagram depicts an entity along with an example. An employee ID is unique and never changes. However, there is a contact detail that can be modified by accessor methods.

Entities have the following properties:

*   They are defined by their identity
*   The identity remains the same throughout its lifetime
*   They are responsible for equality checks

# Value objects

Unlike entities, value objects are immutable and used as descriptors for model elements. They are known to the system only by their characteristics, and they don't need to have unique identifiers. They are always associated with other objects (for example, Delivery Address in the sales order can be a value object) and it is consistently associated with sales order context; otherwise, it doesn't have any meaning.

![](img/5e0227f1-046c-4230-a4d3-43310452eaa1.png)

The preceding diagram depicts the basic concept of a value object along with an example, and the following diagram is a sample class representation of entity and value object:

![](img/5110e924-7c70-440e-8836-b5d28e7666e2.png)

The following list describes the characteristics of value objects:

*   They describe the properties and characteristics within a problem domain
*   They do not have an identity
*   They are immutable, that is, the content of the object cannot be changed; instead, properties modeled as value objects must be replaced

# Domain services

In ubiquitous language, there are situations where actions cannot be attributed to any specific entity or value object, and those operations can be termed as **domain service** (not an application service).

Domain services encapsulate the domain logic and concepts that may not be modeled as entities or value objects, and they are responsible for orchestrating business logic using entities and value objects. The following are a few characteristics/features of domain services:

*   Domain services neither have identity nor state
*   Any operation performed by the domain services does not belong to any of the existing entities
*   Any domain operation in the domain service carry specific objects of the domain model

The following class diagram depicts a sample money transfer operation from one account to another. As we won't be knowing in which object we can store the transfer operation, we choose domain service for this operation:

![](img/8f5e3c27-0b10-4fb4-9a90-dd8884af7fa9.png)

# Modules

Modules are used to decompose the domain model. Naming the modules is part of the ubiquitous language, and they represent a distinct part of domain models and enable clarity when in isolation. Modules help developers to quickly read and understand the domain model in code before deep diving into class development. Note that decomposing domain models is different from subdomains' decomposition of the domain and bounded context.

![](img/579f737d-efe5-433f-89e5-fbf1c7c566c9.png)

The preceding diagram depicts a sample module name and a sample template to follow.

# Aggregates

In DDD, the concept of an aggregate is a boundary that helps in decomposing larger modules into smaller clusters of domain objects, and so the technical complexities can be managed as a high level of abstraction. Aggregates help in doing the following:

*   Reducing and constraining relationships between domain objects
*   Grouping objects of the same business use cases and viewing them as a unified model

Every aggregate has a specific root and border, and within that particular border, all the possible invariants should be satisfied. Domain invariants are statements or rules that always need to be adhered to and help preserve consistency (also known as **atomic transactional coherence**).

![](img/93238711-632e-48b8-a34c-44633a53bfa4.png)

The preceding diagram represents an aggregator sample implementation and brief information about each class and its characteristics associated with aggregates context as follows:

*   **CreditReport**: This includes user information and links, and saves and stores external linkage by **Customer ID** (identifier).
*   **CustomerID**: This an independent aggregate that preserves user information
*   **CreditScore**: This holds credit rating estimation rule and act as invariants. This invariant gets modified/impacted based on credit modifications history.
*   **CreditHistoryEntry**: This helps achieve transactional coherence when it's modified.
*   **Inquiry**: This can handle specific credit score requests from third-party organizations.

# Factories

Factories are a pattern to separate the use (of the object) from the construction (of the object). Aggregates, entities, and value objects create some level of complexity within a domain model, especially with larger domain models. Factories help to express the (creation and use of) complex objects in a better manner.

![](img/0764ab63-8a60-4c70-b081-b79bde8e523f.png)

The preceding diagram might help grasp a quick detail about factory creation from the DDD perspective. The following are some characteristics of factories that we would want to be refreshed with:

*   Separating use from construction
*   Encapsulating internals (and avoid exposing the internals of aggregate)
*   Hiding decisions on the creation type-domain layer factories to abstract the type of class to be created
*   Decluttering complex domain models

![](img/b2dd5863-bb9b-4278-8514-721072e32706.png)

The preceding class diagram intends to give a sample view of the factory implementation for car models to be created; the creation complexity is abstract to the domain.

# Repositories

Repositories are patterns to manage aggregate persistence and retrieval while ensuring a clear separation between the data model and the domain model. Repositories are mediators that act as a collection of facades for storage and persistence.

![](img/5ea8b761-2ccd-4cc5-b6a1-611e7cdab2c2.png)

The preceding diagram depicts a sample structure of a repository model. It shows the client operation of save and update (persistence) with aggregates, through repository, while there is a separate access to the repository (**Deals with Aggregate** in the above diagram); a clear separation between the domain and data model. 

Repositories differ from traditional data access strategies in the following three ways:

*   They restrict access to domain objects by allowing the retrieval and persistence of aggregate roots while ensuring that all the changes and invariants are handled by aggregates
*   They hide the underlying technology used for persistence and retrieval of aggregates from the facades
*   They define a boundary between the domain model and the data model

We have the following two types of repositories:

*   Repositories as collections
*   Repositories as permanent data storage

![](img/3371eeb4-6e3f-4210-8c97-a3ac8f7e1442.png)

The preceding class diagram depicts a sample structure of a repository class and its underlying layer. The repository is within the infrastructure layer and extends the domain layer interface (restrict access).

# Emerging patterns

In this section, we will cover the following two emerging patterns:

*   **Domain events**: They enforce consistency between multiple aggregates of the same domain
*   **Event sourcing**: This is a way of persisting the application's state and finding the current state by traversing through the history of those saved states

# Domain events

The domain event pattern is a preferred a way to trigger side effects across multiple aggregates within the same domain. A domain event is an event that occurs in a particular domain, which the other parts of the same domain (subdomain) should also be aware of and may need to react to it as well.

![](img/fdc631d0-c222-41c1-82c3-b5b02d1d8742.png)

A domain event pattern helps to do the following:

*   Express the side effects of an event in a domain explicitly
*   Maintain consistency of the side effects (either all the operations related to the business task are performed or none of them are)
*   Enable a better separation of concerns among classes within the same domain

# Event sourcing

Event sourcing provides simplification of various events, and it is a way of persisting an application's state and finding the current state by traversing through the history of those saved states. An example could be a seat reservation system that scans the completed bookings and finds out how many more seats are available when a new booking request arrives.

The seat allocation depends on various events (booking, cancellations, modifications, and so on), and it can be handled differently with the event sourcing pattern. This is of immense help in some domains where the audit trail is a critical requirement, (accounting, financial transactions, flight reservations, and so on), and also the pattern helps achieve better performance as events are immutable and support append-only operations.

The following requirements may hint where we need to use event sourcing as a pattern:

*   A simple standalone object to access complex relational storage module
*   Audit Trails (these are a critical requirement)
*   Integration with other subsystems
*   Production troubleshooting (by storing the events and replaying)

We need to be aware of a few general concerns as follows about event sourcing so that we can have trade-offs and mitigation plans:

*   **Versioning**: As event sourcing systems are append-only models, they face unique versioning challenges. Imagine we need to read an event that was created/written years ago into the event-sourcing system. So versioning is necessary to change the definition of a specific event type or aggregate at some point in the future, and one needs to have clear and definite plans and strategies for managing multiple versions for event-source models.
*   **Querying**: This is a little expensive as it gets deeper. It depends on the level and period of the states to be retrieved.
*   **Timeout**: This is the time taken to load the domain object state by querying the event store for all the events related to the state of the aggregate.

# Other patterns

Before concluding this chapter, take a look at the following list of patterns that are important as a part of DDD, however, not covered in this chapter. You are encouraged to review our references section to get an insight into the following topics:

*   Layered architecture
*   Service layers
*   Application services
*   Refactoring toward deeper insight
*   Supple design
*   Making behavior visible (intention revealing interfaces)
*   Side-effect-free functions
*   **Representational state transfer** (**REST**)

# Summary

Sometimes, software design experts get into confusion when to and when not to use domain models. The following points might help you get an insight into DDD for efficient decision making and decide to implement DDD or not:

*   Business cases and requirements are particular, specific to domains, and not related to technology implementations
*   As an independent team, they wanted to go to DDD when:
    *   The team has never done earlier that sort of business cases
    *   The team need help from domain experts
    *   The business cases are more complex
    *   The team need to start from ground zero, and there are no previous models exists
*   When the given design problem is important to your business
*   Skilled, motivated, and passionate team to execute
*   Have greater access to domain experts who are aligned with product vision
*   Willing to follow iterative methodology
*   Nontrivial problem domain that is critical to business
*   Great understanding of vision
*   Business goals, values, success and failure factors, and how is it going to be different from earlier implementations

To summarize, this chapter has short introductions to core principles, characteristics, and best practices for a team to get a head start and adopt DDD. Then, we introduced strategic patterns such as ubiquitous language, domain, subdomain, core domain, and bounded context in detail. We also covered the most essential aspects of DDD, such as autonomous bounded context, shared nothing architecture, single responsibility codes, multiple bounded contexts, and a bit of thought process about SOA principles concerning DDD aspects as part of integrating bounded contexts. We also saw the bubble context, autonomous bubble context, and expose as a service as part of the significant real-world problem of integrating with legacy systems. We introduced you to database integration, flat file integration, and event-driven messaging as part of distributed bounded context integration strategies.

As part of tactical patterns, this chapter covered entity, value objects, domain services, modules, aggregates, factories, and repositories and also discussed two emerging patterns: domain events and event sourcing.

# References and further reading materials

For more information, you can refer to the following books:

*   *Domain-Driven DESIGN - Tackling Complexity in the Heart of Software* - Eric Evans (Pearson)
*   *Patterns, Principles, and Practices of Domain-Driven Design* - Scott Millet with Nick Tune (Wrox)

You can refer to the following online resources, too:

*   DDD quickly: [https://www.infoq.com/minibooks/domain-driven-design-quickly](https://www.infoq.com/minibooks/domain-driven-design-quickly)
*   Framework and tools: [https://isis.apache.org/documentation.html](https://isis.apache.org/documentation.html)
*   Three guiding principles: [https://techbeacon.com/get-your-feet-wet-domain-driven-design-3-guiding-principles](https://techbeacon.com/get-your-feet-wet-domain-driven-design-3-guiding-principles)
*   Getting started with DDD: [https://dzone.com/storage/assets/1216461-dzone-rc-domain-driven-design.pdf](https://dzone.com/storage/assets/1216461-dzone-rc-domain-driven-design.pdf)
*   Model evaluation and management: [https://arxiv.org/ftp/arxiv/papers/1409/1409.2361.pdf](https://arxiv.org/ftp/arxiv/papers/1409/1409.2361.pdf)
*   [https://www.infoq.com/articles/ddd-in-practice](https://www.infoq.com/articles/ddd-in-practice) (characteristics of DDD)
*   [https://www.codeproject.com/Articles/1158628/Domain-Driven-Design-What-You-Need-to-Know-About-S](https://www.codeproject.com/Articles/1158628/Domain-Driven-Design-What-You-Need-to-Know-About-S)
*   [https://www.codeproject.com/Articles/1164363/Domain-Driven-Design-Tactical-Design-Patterns-Part](https://www.codeproject.com/Articles/1164363/Domain-Driven-Design-Tactical-Design-Patterns-Part)
*   [https://www.slideshare.net/SpringCentral/ddd-rest-domain-driven-apis-for-the-web](https://www.slideshare.net/SpringCentral/ddd-rest-domain-driven-apis-for-the-web)
*   [https://www.infoq.com/presentations/ddd-rest](https://www.infoq.com/presentations/ddd-rest)
*   [https://ordina-jworks.github.io/conference/2016/07/10/SpringIO16-DDD-Rest.html](https://ordina-jworks.github.io/conference/2016/07/10/SpringIO16-DDD-Rest.html)
*   [https://www.slideshare.net/canpekdemir/domain-driven-design-71055163](https://www.slideshare.net/canpekdemir/domain-driven-design-71055163)
*   [https://msdn.microsoft.com/magazine/dn342868.aspx](https://msdn.microsoft.com/magazine/dn342868.aspx)
*   [http://mkuthan.github.io/blog/2013/11/04/ddd-architecture-summary/](http://mkuthan.github.io/blog/2013/11/04/ddd-architecture-summary/)