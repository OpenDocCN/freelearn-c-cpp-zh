# Software-Defined Clouds - the Architecture and Design Patterns

The cloud paradigm is on the fast track. There are a number of game-changing advancements in the cloud space, and hence the adoption rate of the cloud concept is consistently on the rise. Legacy applications are being accordingly modified and migrated to cloud environments (private, public, and hybrid). There is a bevy of enabling tools for cloud migration, integration, orchestration, brokerage, deployment, delivery, and management propping up the strategically relevant cloud journey. There are integrated processes, best practices, key guidelines, evaluation metrics, highly synchronized platforms, and so on to make the cloud idea penetrative, participative, and pervasive. Furthermore, there is a growing family of architectural and design patterns for producing optimized cloud environments and applications. This chapter is specially prepared for throwing sufficient light on the patterns emerging and evolving in the cloud landscape. How those patterns are being used in order to simplify and streamline the cloud adoption will be articulated in this chapter.

# Reflecting the cloud journey

With the evolutionary and revolutionary traits of cloud computing, there is a major awareness on the charter of data center optimization and transformation. The acts of simplification and standardization for achieving IT industrialization are garnering a lot of attention these days. The various IT resources, such as memory, disk storage, processing power, and I/O consumption are critically and cognitively monitored, measured, and managed towards their utmost utilization. The pooling and sharing of IT solutions and services are being given paramount importance towards the strategic IT optimization. Also, having a dynamic pool of computing, storage, and network resources enable IT service providers, as well as enterprise IT teams to meet any kinds of spikes and emergencies in resource needs for their customers and users.

The mesmerizing cloud paradigm has, therefore, become the mainstream concept in IT today. And its primary and ancillary technologies are simply flourishing due to the overwhelming acceptance and adoption of cloud theory. The cloudification movement has blossomed these days and most of the IT infrastructures and platforms, along with business applications, are being methodically remedied to be cloud-ready in order to reap all the originally envisaged benefits of the cloud idea. The new buzzword of **Cloud Enablement** has caught up fast and there are collaborative and concerted initiatives to unearth techniques, best practices, patterns, metrics, products and other enablers to understand the cloud fitment and to modernize IT assets and software applications to be cloud-oriented for the ensuing era of knowledge.

Even with all the unprecedented advancements in the cloud landscape, there are a plenty of futuristic and fulsome opportunities and possibilities for IT professors and professionals to take the cloud idea to the next level in its long journey. Therefore, the concept of **software-defined cloud environments** (**SDCEs**) is gaining a lot of accreditation these days. Product vendors, cloud service providers, system integrators, and other principal stakeholders are keen to have such advanced and acclaimed environments for their clients, customers, and consumers. The right and relevant technologies for the realization and sustenance of software-defined cloud environments are fast maturing and stabilizing, and hence the days of SDCEs are not too far away.

In conclusion, the various technological evolutions and revolutions are remarkably enhancing the quality of human lives across the world. Carefully choosing and smartly leveraging the fully matured and stabilized technological solutions and services towards the much-anticipated and acclaimed digital transformation is necessary for a safe, smarter, and sustainable planet.

# Traditional application architecture versus cloud application architecture

As articulated previously, we are heading towards SDCEs that comprise **software-defined compute** (**SDC**), **software-defined storage** (**SDS**), and **software-defined networking** (**SDN**). The virtualization and containerization enable software-defined clouds towards workload-aware and elastic infrastructures. The maneuverability or programmability, consumability, accessibility, sustainability, and simplicity of software-defined clouds are greater compared to the inflexible infrastructures. There are new patterns (architecture and design) being introduced for cloud infrastructures and applications. The emergence of the cloud idea has brought in telling impacts on the application architectures. In this section, we will discuss how cloud application architectures differ from the legacy application architectures.

# The traditional application architecture

Most of the traditional applications were built using the matured three-tier application architecture patterns (presentation tier, middle tier, and data tier). Each tier runs on a dedicated server and is statically configured with the hostnames and IP addresses of the servers of the other tiers it depends on. These applications have very little knowledge of the infrastructure they run on. If the infrastructure changes or fails, these applications also fail. Therefore, these applications are mandated to be hosted on highly reliable and resilient networks and servers. When the load (user and/or data) gets increased, these applications could not automatically scale up or scale out. Scaling is instead done manually through the purchase and installation of additional server machines. This is a time-consuming process, aggravating the complexity. Load balancers are being put up in front of web and application servers in order to bring in the much-needed auto-scaling. However, with the conventional application architecture, the real scalability could not be achieved.

# The cloud architecture

As articulated previously, the concept of virtualization has brought in a programmable infrastructure. That is, the scalability of applications is being achieved through the inherent elasticity of infrastructural components. The resource utilization with the conscious adoption of virtualized infrastructures has gone up significantly. The virtualization idea has penetrated into every infrastructural module these days creating waves of innovations, disruption, transformations, and optimizations for IT environments. That is not only server virtualization, but also network virtualization, storage virtualization, service virtualization, database virtualization, and so on, are being systematically realized in order to bring the originally envisaged virtual, open, flexible, and adaptive IT infrastructures that are intrinsically ready to anticipate and act upon business changes and challenges. The smart usage of cloud technologies, tools, and tips are resulting in business-aware IT infrastructures. The tool ecosystem is steadily growing in order to automate tasks, such as resource provisioning, software deployment, infrastructure monitoring, measurement and management, orchestration, security, governance, and so on. 

The cloud management layer also provides user interfaces for developers and architects to programmatically design and build the infrastructure they need to run their applications. The cloud APIs provided by the cloud management layer also allows applications to take control of the infrastructure they run on. The cloud applications can dynamically scale up or scale down, deploying or removing application components on the infrastructure. The game-changing concept of virtualization and containerization has made it possible to have programmable infrastructures. That is, hardware modules are being expressed as services to be found, used, and even composed. The hardware programming is becoming real these days with the cloud movement. Such a scenario is enabling the days of flexible and maneuverable infrastructures that guarantee workload-awareness, productivity, and high utilization.

# The cloud application architecture

As indicated previously, the traditional applications need to be accordingly modernized in order to reap the cloud benefits. The scalability and other requirements of modern applications need to be inscribed within the application. There are certain programming languages and architectural patterns in order to attach **non-functional requirements** (**NFRs**) into applications. The traditional applications typically use a single database to store all the application information. This database provides the information stored in it to various application clients (users as well as other application components) on a need basis. However, with the data explosion, the conventional databases could be scaled up.

The scale-out (horizontal scalability) of SQL databases is beset with a lot of challenges. However, due to the massiveness of cloud infrastructures, cloud databases have to be designed and developed using new database types, such as NoSQL, NewSQL, in-memory, and in-database databases for data storage and analytics. The object storage is very popular in the cloud era. Every cloud service provider is betting and banking on cloud storage to meet the fast-rising storage needs. Apart from databases, there are enterprise-grade data warehouses and data lakes. Data storage options are on the rise. Cache storage is one such which is garnering a lot of support. Furthermore, there are distributed filesystems, such as HDFS for big data storage and analytics. There are database abstractions on filesystems in order to provide several possibilities for developers, database administers, and businesses. Besides, there are backup and archival options for data in order to ensure data and **disaster recovery** (**DR**). The following diagram vividly illustrates where and how cloud application architecture deviates and differs from traditional applications. With multi-channel, device, media, and modal clients, cloud applications are being methodically advanced:

![](img/7e360909-28c4-4e51-8a80-60d31b30a95c.jpg)

Cloud applications are distinctively different. Increasingly, applications are service-oriented with the faster maturity and stability of service-oriented applications. In the recent past, there have been further refinements and optimizations in order to tackle newer requirements. Polyglot programming is picking up. That is, there are several programming and script languages to bring forth cloud applications as a dynamic collection of microservices. In addition, there is a myriad of database management systems. Each database type is appropriate for certain application needs. Thus, the flexibility of linking multiple technologies, tools, and techniques for bringing forth cloud applications is being facilitated by the most popular **microservices architecture** (**MSA**). With the widespread adoption of containers (Docker) as the highly optimized application holder and runtime environment, there is a sharp convergence of multiple technologies in order to enable agile and accelerated software engineering, deployment, delivery, and management. Multi-container and multi-host cloud applications spearheaded and shepherded by the containerization movement is the talk of the town.

In short, with the cloud embarkation, the widely deliberated **quality of service** (**QoS**) and **quality of experience** (**QoE**) factors and facets of next-generation applications are being accurately accomplished. The cloud infrastructures are being astutely tweaked in order to tackle brewing challenges at the infrastructure level. Another prominent design requirement is to design cloud applications to handle the latency issue. That is, fault tolerance is one such important factor for cloud applications, platforms, and infrastructures. The cascading effect of failures and bugs needs to be arrested in the budding stage itself. As clouds are being built on commodity servers, the failure rate is quite high. Besides, there can be network congestion/outages, resource conflicts, request contentions, and IOPS challenges for storage systems. Furthermore, there can be hardware and software failures. Cloud environments are becoming hugely complicated, and hence viable complexity mitigation and moderation techniques need to be in place. Systems have to come out gracefully from any kind of constricting and cascading issues and limitations. A popular design pattern to address latency and failure is the request/response queue, where requests and responses are stored in queues. Also, cloud and application interfaces have to be highly intuitive, informative, and instructive. The user experience has to be maintained even if cloud resources and assets are not responsive.

# Cloud integration patterns

There are a number of noteworthy advancements happening in the field of cloud computing. Patterns are of much use for the complicated and growing subject of cloud computing. Patterns are being unearthed with the aim of simplifying the deeper understanding and adoption of the cloud paradigm. A number of prospective areas, such as **Infrastructure as a Service** (**IaaS**), **Platform as a Service** (**PaaS**), **Software as a Service** (**SaaS**), **Business Process as a Service** (**BPaaS**), and so on, in the cloud space are being revisited to bring forth fresh and competent patterns. There are special patterns being readied for cloud application development. There are cloud integration platforms (**Integration Platform as a Service** (**IPaaS**)) for enabling a kind of seamless and spontaneous integration among different and distributed cloud applications and data sources. Therefore, integration-specific patterns are being formed and articulated.

These days, the cloud concept has matured and stabilized beautifully in order to give hundreds of novel services for business houses and individuals. On the data services side, we have a **Database as a Service** (**DBaaS**), **Data Warehouse as a Service** (**DWaaS**), **Data Lake as a Service** (**DLaaS**), and so on. Newer databases have emerged in order to tackle a different set of requirements. We all hear, read, and even experience NoSQL and NewSQL databases. Then there are in-memory databases and **in-memory data grids** (**IMDGs**). Data analytics happens within the database itself, and hence we read about in-database analytics. Similarly, the cloud environments are being prescribed as the best-in-class for various other application domains. All kinds of operational, transactional, and analytical applications are being hosted, managed, and delivered through cloud infrastructures and platforms. Then there are new-generation web, mobile, gaming, wearable, embedded, enterprise, IoT, and blockchain applications getting developed, deployed, and delivered through cloud infrastructures and instances. Everything is being expressed and exposed as a service, and undoubtedly, clouds are the elegant, enabling, and execution environments. In the recent past, we have heard more about cloud orchestration, configuration, deployment, migration, governance, and brokerage services. **DevOps** is another buzzword in the cloud landscape. With such a legion of cloud services, there is a need expressed widely by many and a collective call to create beneficial cloud-centric patterns for fulfilling various IT and business capabilities.

# Tier/Layer-based decomposition

The complex functionality of this application is divided into multiple discrete, easily manageable, and loosely-coupled components. Each component is ordained to do one task well. This partition or componentization of application functionality results in a logical decomposition of the original application. These logically separated components run in multiple tiers of a server cluster, and this kind of segmentation is done at the infrastructure level.

# Process-based decomposition

The next is process-based decomposition. Enterprise-grade and complicated applications are typically process-centric. Herein, we can bring in process-based decomposition. Each process internally comprises many tasks that need to be performed in a certain sequence. Each task is done separately and aggregated in the desired order to get the application functionality. There are plenty of automated tools for enabling such decomposition, automation, and finally, orchestration.

# Pipes-and-filters-based decomposition

The third decomposition type is pipes-and-filters-based decomposition, that focuses on the data-centric processing of an application. Each filter provides a certain function that is performed on input data and produces output data after processing. Multiple filters are interconnected with pipes, that is, through messaging.

These layering and decomposition patterns aptly decompose the application into logical layers, enabling independent deployment and horizontal scalability. The layering of application and cloud infrastructures is being touted as the most vital need for developing, deploying, and delivering next-generation distributed applications.

# Service messaging pattern

**Messages as the unifying mechanism**: Messages are the most unifying factor among disparate and distributed cloud services. The goals of cloud service integration get accomplished through message passing. The following section lists and details the various service message patterns. Service messages can be authenticated, routed, enriched, filtered, secured, and composed in order to fulfill the expectations of federated clouds. Cloud intermediation and remediation can be performed through smart messaging. Path-breaking and hitherto unknown services can be built and deployed through the innovative usage of service messages.

How do different distributed and decentralized cloud services find, bind, access, and collaborate with one another in a loosely coupled as well as decoupled manner?

| **Problem** | Services can be run on one virtual machine or in different virtual machines within a cloud environment. Services can even be run on geographically distributed clouds. There are public, private, and hybrid clouds and there are a few communication protocols. The conventional protocols induce a possibility of tight coupling between services. These, in turn, impose certain restrictions on service reusability, testability, and modifiability. |
| **Solution** | Going forward, loose coupling and decoupling are the viable and valuable solution approaches. As even loose coupling has some constraints, decoupling among services is being touted as the most promising solution, and messaging is the way forward for establishing decoupled communication, which in turn eliminates the drawbacks of traditional communication methods |
| **Impacts** | Messaging technology brings a few QoS concerns, such as reliable delivery, security, performance, and transactions. |

**Problem**: Different applications usually use different languages, data formats, and technology platforms. When one application (component) needs to exchange information with another one, the format of the target application has to be respected. Sending messages directly to the target application results in a tight coupling of sender and receiver since format changes directly affect both implementations. Also, direct sending tightly couples the applications regarding the addresses by which they can be reached.

Cloud applications and services communicate using a variety of protocols. **Remote procedure call** (**RPC**), **remote method invocation** (**RMI**), **Windows Communication Framework** (**WCF**), and service protocols (SOAP and REST over HTTP) are some of the leading mechanisms for cloud resources to connect and collaborate purposefully. However, all these lead to a kind of tight coupling, which in turn becomes a hitch or hurdle for services to seamlessly and spontaneously cooperate to achieve bigger and better things. The urgent requirements are therefore loose coupling and decoupling. How can cloud application services communicate remotely through messages while being loosely coupled regarding their location and message format? Another brewing requirement is to enable complete decoupling among services.

**Solution**: The context is that distributed applications or their service components exchange information using messaging. Messaging comes as a viable alternative communication scheme that does not rely on persistent connections. Instead, messages are being transmitted as independent units of communication routed through the underlying infrastructure. That is, simply connect applications through an intermediary; the message-oriented middleware hides the complexity of addressing and availability of communication partners as well as supports transformation of different message formats.

Communication partners can now communicate through messages without the need to know the message format used by the communication partner or the address by which it can be reached. The message-oriented middleware provides message channels (also referred to as **queues**). Messages can be written to these queues or read from them. Additionally, the message-oriented middleware contains components that route messages between channels to intended receivers, as well as handle message format transformation.

The messaging framework must have the following capabilities:

*   Guaranteeing the delivery of each message or guaranteeing a notification of failed deliveries
*   Securing message contents beyond the transport
*   Managing state and context data across a service activity
*   Transmitting messages efficiently as part of real-time interactions
*   Coordinating cross-service transactions

Without these types of extensions in place, the availability, reliability, and reusability of services will impose limitations that can undermine the strategic goals associated with cloud-hosted services.

# Messaging metadata pattern

| **Problem** | Services generally work in a stateless fashion. That is, they do not store any state data in order to facilitate the next course of action. As the message is the intermediary in order to empower different and distributed services to interact together towards accomplishing business transactions and operations, they need to have or carry all the state data (metadata). |
| **Solution** | The content encapsulated within the message envelope, therefore, has to be supplemented with activity-specific metadata that can be interpreted and processed separately at runtime. |
| **Impacts** | The interpretation and processing of messaging metadata adds to runtime performance overhead and increases service activity design complexity. |

**Problem**: In the traditional method, the state and context data about the current service interaction are placed in the memory. However, in a service environment, services are being designed, developed, and deployed as stateless resources to be highly reusable. Therefore, the messages that are being transmitted among services are being mandated to carry the right and relevant data to initiate the correct actions sequentially to accomplish the business process tasks.

**Solution**: As messages carry the state data, business rules, and even processing instructions, services can be designed in a very generic manner. The service complexity will come down, the reusability level will go up, modifiability will be easier, enrichment will be quicker, and so on.

Though the overall memory consumption is reduced by avoiding a persistent binary connection, the performance demands are increased by the requirement for services to interpret and process metadata at runtime. Agnostic services especially can impose more runtime cycles, as they may need to be outfitted with highly generic routines capable of interpreting and processing different types of messaging headers so as to participate effectively in multiple composition activities. Due to the prevalence and range of technology standards that intrinsically support and are based on messaging metadata, a wide variety of sophisticated message exchanges can be designed. This can lead to overly creative and complex message paths that may be difficult to govern and evolve.

# Service agent pattern

How can event capturing and processing logic be separated and governed independently?

| **Problem** | Service composition (orchestration and choreography) can become large and inefficient, especially when required to invoke granular capabilities across multiple services. |
| **Solution** | Event-driven service composition is emerging as an important factor for crafting composite services. Event-driven logic can be easily deferred to event-driven programs that don't require explicit invocation, thereby reducing the size and performance strain of service composition. |
| **Impacts** | The complexity of composition logic increases when it is distributed across services, and event-driven agents and reliance on service agents can further tie inventory architecture to proprietary vendor technology. |

**Problem**: Decomposition and composition are the highly successful methods for simplifying and streamlining software engineering. In a service environment, applications are built by assembling a variety of services. In software engineering, the application to be realized is to start with a series of business processes (simple and compound) and each process, in turn, gets implemented by leveraging a number of services (process elements). That is, applications are decomposed into a collection of interoperable and interactive services and services are smartly composed to form next-generation applications.

Service composition logic consists of a series of service invocations, and each invocation enlists a service to carry out a segment of the overall parent business process logic. Larger business processes can be enormously complex, especially when having to incorporate numerous *what if* conditions through compensation and exception handling subprocesses. Therefore, service composition can grow correspondingly large. Furthermore, each service invocation comes with a performance hit resulting from having to explicitly invoke and communicate with the service itself. The performance of larger compositions can suffer from the collective overhead of having to invoke multiple services to automate a single task.

**Solution**: *Separation of concerns* has been an interesting technique in software engineering. Service logic that is triggered by a predictable event can be isolated into a separate program specially designed for automatic invocation upon the occurrence of the event. This reduces the amount of composition logic that needs to reside within services and further decreases the number of service invocations required for a given composition.

Event-driven agents provide yet another layer of abstraction to which multiple service compositions can form dependencies. Although the perceived size of the composition may be reduced, the actual complexity of the composition itself does not decrease. Composition logic is simply more decentralized as it now also encompasses service agents that automatically perform portions of the overall task.

# Intermediate routing pattern

How can dynamic runtime factors affect the path of a message?:

| **Problem** | The larger and more complex a service composition is, the more difficult it is to anticipate and design for all possible runtime scenarios in advance, especially with asynchronous and messaging-based communication. |
| **Solution** | Message paths can be dynamically determined through the use of intermediary routing logic. |
| **Impacts** | Dynamically determining a message path adds layers of processing logic and correspondingly can increase performance overhead. Also, the use of multiple routing logic can result in overly complex service activities. |

**Problem**: A service composition can be viewed as a chain of point-to-point data exchanges between composition participants. Collectively, these exchanges end up automating a parent business process. The message routing logic (the decision logic that determines how messages are passed from one service to another) can be embedded within the logic of each service in a composition. This allows for the successful execution of predetermined message paths. However, there may be unforeseen factors that are not accounted for in the embedded routing logic, which can lead to unanticipated system failures. For example:

*   The destination service a message is being transmitted to is temporarily (or even permanently) unavailable
*   The embedded routing logic contains a *catch-all* condition to handle exceptions, but the resulting message destination is still incorrect
*   The originally planned message path cannot be carried out, resulting in a rejection of the message from the service's previous consumer

Alternatively, there may simply be functional requirements that are dynamic in nature and for which services cannot be designed in advance.

**Solution**: Generic and multi-purpose routing logic can be abstracted so that it exists as a separate part of the architecture in support of multiple services and service compositions. Most commonly, this is achieved through the use of event-driven service agents that transparently intercept messages and dynamically determine their paths.

This pattern is usually applied as a specialized implementation of a service agent. Routing-centric agents required to perform dynamic routing are often provided by messaging middleware and are fundamental components of **enterprise service bus** (**ESB**) products. These types of out-of-the-box agents can be configured to carry out a range of routing functions. However, the creation of custom routing agents is also possible and not uncommon, especially in environments that need to support complex service compositions with special requirements.

# State messaging pattern

How can services remain stateless while contributing to stateful interactions?:

| **Problem** | When services are required to maintain state information in memory between message exchanges with consumers, their scalability can be compromised, and they can become a performance bottleneck on the surrounding infrastructure. |
| **Solution** | Instead of retaining the state data in memory, its storage is temporarily delegated to messages. |
| **Impacts** | This pattern may not be suitable for all forms of state data and should the message be lost, any state information they carried may be lost as well. |

**Problem**: Services are sometimes required to be involved in runtime activities that span multiple message exchanges. In these cases, a service may need to retain state information until the overarching task is completed. This is especially common with services that act as composition controllers. By default, services are often designed to keep this state data in memory so that it is easily accessible and essentially remains alive for as long as the service instance is active. However, this design approach can lead to serious scalability problems and further runs contrary to the service statelessness design principle.

**Solution**: Instead of the service maintaining state data in memory, it moves the data to the message. During a conversational interaction, the service retrieves the latest state data from the next input message.

There are two common approaches for applying this pattern, both of which affect how the service consumer relates to the state data. The consumer retains a copy of the latest state data in memory and only the service benefits from delegating the state data to the message. This approach is suitable for when this pattern is implemented using WS-Addressing, due to the one-way conversational nature of **endpoint references** (**EPRs**).

Both the consumer and the service use messages to temporarily offload state data. This two-way interaction with state data may be appropriate when both consumer and service are actual services within a larger composition. This technique can be achieved using custom message headers.

When following the two-way model with custom headers, messages that are lost due to runtime failure or exception conditions will further lose the state data, thereby placing the overarching task in jeopardy. It is also important to consider the security implications of state data placed on the messaging layer. For services that handle sensitive or private data, the corresponding state information should either be suitably encrypted and/or digitally signed, and it is not uncommon for the consumer to not gain access to protected state data.

Furthermore, because this pattern requires that state data be stored within messages that are passed back and forth with every request and response, it is important to consider the size of this information and the implications on bandwidth and runtime latency. As with other patterns that require new infrastructure extensions, establishing inventory-wide support for state messaging will introduce cost and effort due to the necessary infrastructure upgrades.

# Service callback pattern

How can a service sync up asynchronously with its consumers?:

| **Problem** | When a service needs to respond to a consumer request for the issuance of multiple messages or when service message processing requires a large amount of time, it is often not possible to communicate synchronously. |
| **Solution** | A service can require that consumers communicate with it asynchronously and provide a callback address to which the service can send response messages. |
| **Impacts** | Asynchronous communication can introduce reliability concerns and can further require that surrounding infrastructure be upgraded to fully support the necessary callback correlation. |

**Problem**: When service logic requires that a consumer request is responded to with multiple messages, a standard request-response messaging exchange is not appropriate. Similarly, when a given consumer request requires that the service perform prolonged processing before being able to respond, synchronous communication is not possible without jeopardizing scalability and reliability of the service and its surrounding architecture.

**Solution**: Services are designed in such a manner that consumers provide them with a callback address at which they can be contacted by the service at some point after the service receives the initial consumer request message. Consumers are furthermore asked to supply correlation details that allow the service to send an identifier within future messages so that consumers can associate them with the original task.

# Service instance routing

How can consumers contact and interact with service instances without the need for proprietary processing logic?:

| **Problem** | When required to repeatedly access a specific stateful service instance, consumers must rely on the custom logic that more tightly couples them to the service. |
| **Solution** | The service provides an instance identifier along with its destination information in a standardized format that shields the consumer from having to resort to custom logic. |
| **Impacts** | This pattern can introduce the need for significant infrastructure upgrades, and when misused can further lead to overly stateful messaging activities that can violate the service statelessness principle. |

**Problem**: There are cases where a consumer sends multiple messages to a service and the messages need to be processed within the same runtime context. Such services are intentionally designed to remain stateful so that they can carry out conversational or session-centric message exchanges. However, service contracts generally do not provide a standardized means of representing or targeting instances of services. Therefore, consumer and service designers need to resort to passing proprietary instance identifiers as part of the regular message data, which results in the need for proprietary instance processing logic.

**Solution**: The underlying infrastructure is extended to support the processing of message metadata that enables a service instance identifier to be placed into a reference to the overall destination of the service. This reference (also referred to as an **endpoint reference**) is managed by the messaging infrastructure so that messages issued by the consumer are automatically routed to the destination represented by the reference. As a result, the processing of instance IDs does not negatively affect consumer-to-service coupling because consumers are not required to contain proprietary service instance processing logic. Because the instance IDs are part of a reference that is managed by the infrastructure, they are opaque to consumers. This means that consumers do not need to be aware of whether they are sending messages to a service or one of its instances because this is the responsibility of the routing logic within the messaging infrastructure.

# Asynchronous queuing pattern

How can a service and its consumers accommodate isolated failures and avoid unnecessarily locking resources?:

| **Problem** | When a service capability requires that consumers interact with it synchronously, it can inhibit performance and compromise reliability. |
| **Solution** | A service can exchange messages with its consumers through an intermediary buffer, allowing services and consumers to process messages independently by remaining temporally decoupled. |
| **Impacts** | There may be no acknowledgment of successful message delivery, and atomic transactions may not be possible. |

**Problem**: Synchronous communication requires an immediate response to each request, and therefore forces two-way data exchange for every service interaction. When services need to carry out synchronous communication, both service and service consumer must be available and ready to complete the data exchange. This can introduce reliability issues when either the service cannot guarantee its availability to receive the request message or the service consumer cannot guarantee its availability to receive the response to its request. Because of its sequential nature, synchronous message exchanges can further impose processing overhead, as the service consumer needs to wait until it receives a response from its original request before proceeding to its next action. Prolonged responses can introduce latency by temporarily locking both consumer and service.

Another problem forced synchronous communication can cause is an overload of services required to facilitate a great deal of concurrent access. Because services are expected to process requests as soon as they are received, usage thresholds can be more easily reached, thereby exposing the service to multi-consumer latency or overall failure.

**Solution**: A queue is introduced as an intermediary buffer that receives request messages and then forwards them on behalf of the service consumers. If the target service is unavailable, the queue acts as temporary storage and retains the message. It then periodically attempts retransmission. Similarly, if there is a response, it can be issued through the same queue that will forward it back to the service consumer when the consumer is available. While either service or consumer is processing message contents, the other can deactivate itself (or move on to other processing) in order to minimize memory consumption.

# Reliable messaging pattern

How do we enable and ensure services to interact reliably in an unreliable environment?:

| **Problem** | Messages need to reach the right services and should not be tampered within their path. That is, unreliable communication protocols and service environments are said to be the main barriers for service reliability. |
| **Solution** | An intermediate reliability mechanism has to be in place in order to guarantee that messages reach the right services and their integrity and confidentiality are being maintained appropriately. Also, this middleware has to guarantee message delivery. |
| **Impacts** | Using a reliability framework adds processing overhead that can affect service activity performance. It also increases composition design complexity and may not be compatible with atomic service transactions. |

**Problem**: When services are designed to activate and act through messages, there is a natural tendency for the loss of quality of service due to the stateless nature of underlying messaging protocols, such as HTTP. The binary communication protocols maintain a persistent connection until the data transmission between a sender and receiver is completed. However, with message exchanges, the runtime platform may not be able to provide feedback to the sender as to whether or not the message was successfully delivered to the target service endpoint. With more services and more network links, the complexity of service composition grows accordingly.

If the middleware infrastructure being employed is not able to guarantee reliable message delivery, then risks erupt. How can messages be exchanged while guaranteeing that messages are not lost in the case of system or communication failures? Reliability agents further manage the confirmation of successful and failed message deliveries through positive (ACK) and negative (NACK) acknowledgment notifications. Messages may be transmitted and acknowledged individually, or they may be bundled into message sequences that are acknowledged in groups (and may also have sequence-related delivery rules).

When messages are exchanged in distributed systems, errors can occur during the transmission of messages over communication links or during the processing of messages in system components. Under these conditions, it should be guaranteed that no messages are lost and that messages can be eventually recovered after a system failure.

**Solution: **The underlying infrastructure is fitted with a reliability framework that tracks and temporarily persists message transmissions and issues positive and negative acknowledgments to communicate successful and failed transmissions to message senders. Message exchange during communication partners is performed under transactional context, guaranteeing ACID behavior. In the cloud, there are several messaging systems that can be accessed as a service, such as Amazon SQS or the queue service part of Windows Azure Storage.

As articulated previously, cloud integration patterns are very vital for cloud-based distributed application development, deployment, and delivery. There are specialized adapters, connectors, drivers, and other plugins to simplify and streamline cloud integration requirements. The integration patterns are crucial for the success of the cloud paradigm.

# Cloud design patterns

This section will discuss various cloud application design patterns that are highly useful for building reliable, scalable, and secure applications in the cloud. Readers can find deeper and decisive details of the patterns on the Microsoft website: [https://docs.microsoft.com/en-us/azure/architecture/patterns/](https://docs.microsoft.com/en-us/azure/architecture/patterns/).

# Cache-aside pattern

The gist of this pattern is to load data on demand into a cache from a data store. This pattern can improve performance and also helps to maintain consistency between data held in the cache and the data in the data store. Applications use a cache to optimize repeated access to information held in a data store. However, it is usually impractical to expect that cached data will always be completely consistent with the data in the data store.

There are many commercial caching systems providing read-through and write-through/write-behind operations. In these systems, an application retrieves data by referencing the cache. If the data is not available in the cache, it is transparently retrieved from the distant data store and added to the cache. Any modifications to data held in the cache are automatically written back to the data store as well. For caches that do not provide this functionality, it is the responsibility of the applications that use the cache to maintain the data in the cache. An application can emulate the functionality of read-through caching by implementing the cache-aside strategy. This strategy effectively loads data into the cache on demand.

Cloud application performance is often questioned by many. Hence, there is a bevy of performance enhancement techniques and tips being unearthed and promoted. This pattern is one such breakthrough solution technique in order to supply all the right and relevant information for application designers to substantially increase cloud performance.

The usage scenarios include:

*   A cache doesn't provide native read-through and write-through operations
*   Resource demand is unpredictable

# Circuit breaker pattern

We all know that distributed computing is the way forward for new-generation businesses. Connectivity to remote services and resources is a core requirement in distributed computing environments. Remote connectivity has the habit of failure. That is, an application is trying to get connected with a remote service or data source and is not able to get access due to some transient fault, such as slow network connection, timeouts, the resources being overloaded, temporarily unavailable, and so on. These faults typically correct themselves after a short period of time, and a robust cloud application should be prepared to overcome these by using a well-drawn strategy, such as that described by the retry pattern.

However, there may also be situations where faults occur out of unexpected events that are quite tough to anticipate. Furthermore, those faults may take a longer time to get rectified. These faults can range in severity, from a partial loss of connectivity to the complete failure of a service. In these situations, it may be pointless for an application to continually retry performing an operation that is unlikely to succeed, and instead, the application should quickly accept that the operation has failed and handle the failure accordingly.

The circuit breaker pattern can prevent an application repeatedly trying to execute an operation that is likely to fail, allowing it to continue without waiting for the fault to be rectified or wasting CPU cycles while it determines that the fault is long lasting. The circuit breaker pattern also enables an application to detect whether the fault has been resolved. If the problem appears to have been rectified, the application can attempt to invoke the operation.

The circuit breaker pattern is different from the retry pattern. The retry pattern enables an application to retry an operation with the expectation that it will succeed, but the circuit breaker pattern prevents an application from performing an operation that is likely to fail. An application may combine these two patterns by using the retry pattern to invoke an operation through a circuit breaker. However, the retry logic should be sensitive to any exceptions returned by the circuit breaker and abandon retry attempts if the circuit breaker indicates that a fault is not transient.

A circuit breaker acts as a proxy for operations that may fail. The proxy should monitor the number of recent failures that have occurred, and then use this information to decide whether to allow the operation to proceed, or simply return an exception immediately.

# Compensating transaction pattern

We all know that any business and financial transaction has to strictly fulfill the ACID properties. Steadily, transactional applications are being deployed in cloud environments. Now, in the big data era, distributed computing is becoming the mainstream computing model. NoSQL databases are very prominent and dominant these days in order to do justice to big data. Increasingly, there is an assortment of segregated yet connected data sources as well as stores to perform high-performance data access, processing, and retrieval. In this case, strong transactional consistency is not being maintained. Rather, the application should go for eventual consistency. While these steps are being performed, the overall view of the system state may be inconsistent, but when the operation has completed and all of the steps have been executed, the system should become consistent again.

A significant challenge in the eventual consistency model is how to handle a step that has failed irrecoverably. In this case, it may be necessary to undo all of the work completed by the previous steps in the operation. However, the data cannot simply be rolled back because other concurrent instances of the application may have since changed it. Even in cases where the data has not been changed by a concurrent instance, undoing a step might not simply be a matter of restoring the original state. It may be necessary to apply various business-specific rules.

Compensation has been the typical response when a transaction fails. This pattern is mainly to undo the work performed by a series of steps, which together define an eventually consistent operation if one or more of the steps fail. A compensating transaction might not be able to simply replace the current state with the state the system was in at the start of the operation because this approach could overwrite changes made by other concurrent instances of an application. Rather, it must be an intelligent process that takes into account any work done by concurrent instances. This process will usually be application-specific, driven by the nature of the work performed by the original operation.

A common approach to implementing an eventually consistent operation that requires compensation is to use a workflow. As the original operation proceeds, the system records information about each step and how the work performed by that step can be undone. If the operation fails at any point, the workflow rewinds back through the steps it has completed and performs the work that reverses each step.

# Competing consumers pattern

With the surging popularity of web-scale applications, there can be a large number of requests from different parts of the world for those applications. The user and data loads are generally unpredictable. The task/operation complexity is also unpredictable. Because of heavy loads, cloud applications find it difficult to process every request and deliver the reply within the stipulated timeline. One option is to add new server instances. There are some practical difficulties in clustered and load-balanced environments too. However, these consumers must be coordinated to ensure that each message is only delivered to a single consumer. The workload also needs to be load balanced across consumers to prevent an instance from becoming a bottleneck.

An overwhelming solution approach here is to use a messaging system (message queue or broker) in between any requesting applications/users and the processing applications. A **message-oriented middleware** (**MOM**) is a way forward for meeting a large number of concurrent consumers. This middleware approach supports asynchronous communication and processing, thereby the massive number of requests can be answered quickly.

A message queue/broker/bus is used to establish the communication channel between the application and the instances of the consumer service. The application posts requests in the form of messages to the queue and the consumer service instances receive messages from the queue and process them. This approach enables the same pool of consumer service instances to handle messages from any instance of the application. The following figure illustrates this architecture:

![](img/378c8a83-880e-45e9-95c9-7642e2a044bb.png)

This pattern enables multiple concurrent consumers to process messages received on the same messaging channel. This pattern enables a system to process multiple messages concurrently to optimize throughput, to improve scalability and availability, and to balance the workload.

# Compute resource consolidation pattern

There are several architectural patterns, such as MVC, **service-oriented architecture** (**SOA**), **event-driven architecture** (**EDA**), **resource-oriented architecture** (**ROA**), **microservices architecture** (**MSA**), and so on, recommending the application partitioning for various benefits. However, there are occasions wherein consolidating multiple tasks or operations into a single computational unit brings forth a number of advantages.

A common approach is to look for tasks that have a similar profile concerning their scalability, lifetime, and processing requirements. Grouping these items together allows them to scale as a unit. The elasticity provided by many cloud environments enables additional instances of a computational unit to be started and stopped according to the workload. This pattern can increase compute resource utilization, and reduce the costs and management overhead associated with performing compute processing in cloud-hosted applications.

# Command and query responsibility segregation (CQRS) pattern

In traditional database management systems, both commands (updates to the data) and queries (requests for data) are executed against the same set of entities in a single data repository. These entities may be a subset of the rows in one or more tables in an RDBMS. Typically, in these systems, all **create**, **read**, **update**, and **delete** (**CRUD**) operations are applied to the same representation of the entity. Traditional CRUD designs work well when there is only limited business logic applied to the data operations. There are a few serious issues being associated with the CRUD approach, as follows:

*   There may be a mismatch between the read and write representations of the data, such as additional columns or properties that must be updated correctly even though they are not required as a part of an operation
*   It risks encountering data contention in a collaborative domain (where multiple actors operate in parallel on the same set of data) when records are locked in the data store, or update conflicts caused by concurrent updates when optimistic locking is used

This pattern segregates operations that read data from operations that update data by using separate interfaces. This pattern can maximize performance, scalability, and security, support evolution of the system over time through higher flexibility, and prevent update commands from causing merge conflicts at the domain level.

**The event sourcing pattern and the CQRS pattern**: CQRS-based systems use separate read and write data models. Each model is tailored to relevant tasks and often located in physically separate stores. When used with event sourcing, the store of events is the write model, and this is the authoritative source of information. The read model of a CQRS-based system provides materialized views of the data as highly denormalized views. These views are tailored to the interfaces and display requirements of the application and this helps to maximize both display and query performance.

Using the stream of events as the write store, rather than the actual data at a point in time, avoids update conflicts on a single aggregate and maximizes performance and scalability. The events can be used to asynchronously generate materialized views of the data that are used to populate the read store.

# Event sourcing pattern

Most applications work with data, and the overwhelming approach is for the application to maintain the current state of the data by updating it as users work with the data. For example, in the CRUD model, a data process reads data from the store, makes some modifications to it, and updates the current state of the data with the new values. The problem with this approach is that performing update operations directly against a data store may degrade performance and responsiveness. The scalability aspect may also be affected. In a collaborative environment, there are many concurrent users, and hence there is a high possibility for data update conflicts because the update operations take place on a single item of data.

The events are persisted in an event store that acts as the source of truth about the current state of the data. The event store typically publishes these events so that consumers can be notified and can handle them if needed. Consumers could, for example, initiate tasks that apply the operations in the events to other systems. The point to be noted here is that application code that generates the events is decoupled from the systems that subscribe to the events.

The solution is to use an append-only store to record the full series of events that describe actions taken on data in a domain, rather than storing just the current state so that the store can be used to materialize the domain objects. This pattern can simplify tasks in complex domains by avoiding the requirement to synchronize the data model and the business domain. This pattern improves performance, scalability, and responsiveness. Furthermore, it provides consistency for transactional data and maintains full audit trails and history that may enable compensating actions.

# External configuration store pattern

The majority of application runtime environments include configuration information that is held in files deployed with the application, located within the application folders. In some cases, it is possible to edit these files to change the behavior of the application after it has been deployed. However, in many cases, changes to the configuration require the application to be redeployed, resulting in unacceptable downtime and additional administrative overhead.

Local configuration files also limit the configuration to a single application, whereas, in some scenarios, it would be useful to share configuration settings across multiple applications. Managing changes to local configurations across multiple running instances of the application is another challenge. The approach is to store the configuration information in external storage. This moves configuration information out of the application deployment package to a centralized location. This pattern can provide opportunities for easier management and control of configuration data, and for sharing configuration data across applications and application instances**.**

# Federated identity pattern

There are many applications hosted by different cloud service providers. Predominantly, there are email and social networking applications. These applications are being subscribed to and used by many people from different parts of the world. Typically, users need to memorize and use different credentials for accessing each of these customer-centric, collaborative, and cloud applications. Managing multiple credentials is a tough assignment. The solution is to implement an authentication mechanism that can use the proven concept of federated identity. This is accomplished by separating the aspect of user authentication from the application logic code and delegating the authentication requirement to a trusted and third-party identity service provider. The trusted identity providers can authenticate users on behalf of application service providers. The identity service providers have, for example, a Microsoft, Google, Yahoo!, or Facebook account. This identity pattern can simplify development, minimize the requirement for user administration, and improve the user experience of the application.

# Gatekeeper pattern

Cloud applications need to be protected from malicious users. Also, some cloud applications are provided by multiple cloud service providers. Users are therefore in a position to choose one service provider according to his or her terms. Cloud broker is a new software product enabling users to zero down the appropriate cloud service providers. Thus, a kind of gatekeeper software solution is needed to act as a broker between application clients and application services, validate and sanitize requests, and pass requests and data between them. This can provide an additional layer of security, and limit the attack surface of the system.

This pattern minimizes the risk of clients gaining access to sensitive information and services. This gateway solution contributes as a façade or a dedicated task that interacts with clients and then hands off the request perhaps through a decoupled interface to the hosts or tasks that'll handle the request.

# Application health monitoring pattern

Cloud applications ought to fulfill the various service and operational expectations that are formally contracted through an SLA agreement. Hence, it is pertinent to have a competent health monitoring system to precisely and minutely monitor the functioning and health of cloud applications, database systems, middleware solutions, and so on. The health check is, therefore, an important factor in ensuring the agreed quality parameters. The monitoring is not an easy thing to do. Cloud environments are hugely complicated due to the massive scale, such as increasingly software-defined, federated, and shared. The way forward here is to put a health monitoring system in place in order to send requests to an endpoint on the application so as to capture the right and relevant data to act upon with clarity and confidence.

# Leader election pattern

Cloud applications are extremely complicated yet sophisticated. Multiple instances of cloud applications can run in a cloud environment. Similarly, different components of an application can run on clouds. The tasks might be working together in parallel to perform the individual parts of a complex calculation.

The task instances might run separately for much of the time, but it might also be necessary to coordinate the actions of each instance to ensure that they don't create any sort of conflict, cause contention for shared resources, or accidentally interfere with the work that other task instances are performing. So, there is a need for adept coordinator software; each action has to be coordinated.

A single task instance should be elected to act as the leader, and this instance should coordinate the actions of the other subordinate task instances. If all of the task instances are running the same code, then one instance can act as the leader. However, the election of the leader has to be done smartly. There has to be a robust mechanism in place for leader selection. This selection method has to cope with the events, such as network outages or process failures. In many solutions, the subordinate task instances monitor the leader through some type of heartbeat method, or by polling. If the designated leader terminates unexpectedly, or a network failure makes the leader unavailable to the subordinate task instances, it is necessary for them to elect a new leader.

# Materialized views pattern

When storing data, developers and database administrators are more concerned about how the data is stored. They are least bothered about how the data will be read. The chosen data storage format is usually closely related to the format of the data, requirements for managing data size and data integrity, and the kind of store in use. For example, when using a NoSQL document store, the data is often represented as a series of aggregates, each containing all of the information for that entity. However, this can have a negative effect on queries. When a query only needs a subset of the data from some entities, such as a summary of orders for several customers without all of the order details, it must extract all of the data for the relevant entities in order to obtain the required details.

To support efficient querying, a common solution is to generate, in advance, a view that materializes the data in a format suited to the required results set. The materialized view pattern describes generating prepopulated views of data in environments where the source data isn't in a suitable format for querying, where generating a suitable query is difficult, or where query performance is poor due to the nature of the data or the data store.

These materialized views, which only contain data required by a query, allow applications to quickly obtain the information they need. In addition to joining tables or combining data entities, materialized views can include the current values of calculated columns or data items, the results of combining values or executing transformations on the data items, and values specified as part of the query. A materialized view can even be optimized for just a single query. This pattern can help support efficient querying and data extraction, and improve application performance.

# Pipes and filters pattern

An application is mandated to perform a variety of tasks of varying complexity on the information that it receives, processes, and presents. Traditionally, a monolithic application is produced to perform this duty. However, the monolithic architecture and approach are bound to fail in due course due to various reasons (modifiability, replaceability, reusability, substitutability, simplicity, accessibility, sustainability, scalability, and so on). Therefore, the proven and potential technique of *divide and conquer* has become a preferred approach in the field of software engineering. **Aspect-oriented programming** (**AOP**) is a popular method. There are other decomposition approaches.

Furthermore, some of the tasks that the monolithic modules perform are functionally very similar, but the modules have been designed separately. Some tasks might be compute intensive and could benefit from running on powerful hardware, while others might not require such expensive resources. Also, additional processing might be required in the future, or the order in which the tasks are performed by the processing could change.

Considering all these limitations, the recommended approach is to break down the processing required for each stream of tasks into a set of separate components (filters), and each component (filter) is assigned to perform a single task. By standardizing the format of the data that each component receives and sends, these filters can be combined together into a pipeline. This helps to avoid duplicating code and makes it easy to remove, replace, or integrate additional components if the processing requirements change. This unique pattern can substantially improve performance, scalability, and reusability by allowing task elements that perform the processing to be deployed and scaled independently.

# Priority queue pattern

Applications can delegate specific tasks to other services to perform them, such as some background processing or the integration with third-party or external applications or services. Employing middleware solutions to perform those intermediary jobs has been a widely followed activity. The message queue is a prominent one in enterprise and cloud environments to realize tasks, such as intermediation, enrichment, filtering and funneling, and so on. Here, the order of the requests is not important. That is, giving a kind of priority for a particular task is being insisted in certain scenarios. These requests should be processed earlier than lower priority requests that were sent previously by the application.

A queue is usually a **first-in, first-out** (**FIFO**) structure, and consumers typically receive messages in the same order that they were posted to the queue. However, some message queues support priority messaging. The application posting a message can assign a priority and the messages in the queue are automatically reordered so that those with a higher priority will be received before those with a lower priority. This pattern is useful in applications that offer different service-level guarantees to individual clients.

# Queue-based load leveling pattern

Cloud applications may sometimes be subjected to heavy loads (user and data). Applications are being designed and developed accordingly and are made to run on clustered environments in order to meet sudden or seasonal spikes. When applications are under heavy loads, the application performance may go down. Especially, some crucial tasks that are the part of the application may come under heavy bombardment.

The viable approach is to refactor the application and introduce a queue between the task and the service. The idea here is that the task and the service run asynchronously. The task posts a message containing the data required by the service to a queue. The queue acts as a buffer, storing the message until it's retrieved by the service. The service retrieves the messages from the queue and processes them. Requests from a number of tasks, which can be generated at a highly variable rate, can be passed to the service through the same message queue. This pattern can help to minimize the impact of peaks in demand on availability and responsiveness for both the task and the service.

# Retry pattern

Applications are spread across multiple clouds, across continents, countries, counties, and cities. Not only public clouds are being leveraged as the application deployment, delivery, and management platform, but also mission-critical, high-performance, and secure applications and data stores are being deployed and delivered through private clouds. Some enterprises continue with traditional IT environments. Applications literally have to connect, access and use nearby, as well as remotely held, applications or databases often as a part of successfully fulfilling any brewing business process requirements. But applications connecting and collaborating with other applications in the vicinity or in off-premise environments are not that straightforward.

There can be transient faults in the way of accessing other applications. The network connectivity, the failure of the requested applications due to overload, the temporary unavailability of the application, and so on, are being touted as the challenges for applications talking to one another over any network.

Applications ought to be designed in such a way that they try again to connect and proceed with their task-fulfillment. If the application request fails, the application can wait and make another attempt. If necessary, this process can be repeated with increasing delays between retry attempts, until some maximum number of requests has been attempted. The delay can be increased incrementally or exponentially, depending on the type of failure and the probability that it'll be corrected during this time.

# Runtime reconfiguration pattern

Traditionally, a static configuration has been the way for any application. If there is a need to make changes in the configuration, then the application has to be shut down and restarted after incorporating the configuration changes. Now in the web world, the downtime is not liked. Therefore, there is a need for a workable technique to achieve runtime configuration. That is, while the application is still running and delivering its service, the required configuration has to be brought in. The application has to immediately consider the changes and act on that. Similarly, the application has to convey the configuration changes to all its components.

The success of this pattern squarely depends on the features available in the application runtime environment. Typically, the application code will respond to one or more events that are raised by the hosting infrastructure when it detects a change to the application configuration. This is usually the result of uploading a new configuration file, or in response to changes in the configuration through the administration portal or by accessing an API.

The source code that handles the configuration change events can examine the changes and apply them to the components of the application. These components have to detect and react to the changes. The components should use the new values so that the intended application behavior can be achieved. This helps to maintain availability and minimize downtime.

# Scheduler agent supervisor pattern

Enterprise-class applications are slated to do many tasks in sequence or in parallel. Each task is performed by a microservice architecture that can comfortably run inside Docker containers. Some tasks may have to connect and collaborate with remote application services or third-party services. As stated previously, the remote connectivity is beset with a number of challenges because there are other components contributing to the remote connectivity and access. Now, complex applications are being simplified through process flows comprising control as well as data flows. That means an application has to orchestrate all the steps/services in order to ensure its capability for consumers. In the distributed computing arena, all services have to play their unique role and deliver value to their application. If anyone fails to transact, then the retry pattern can be leveraged. If that also fails to take off, then the entire operation has to be canceled.

The solution is to use the scheduler agent supervisor pattern that skillfully orchestrates all the right and relevant steps to finish the expected job. This orchestration software solution manages all the participating and contributing steps in a resilient and rewarding fashion in distributed work environments. The scheduler, which is the principal component of the scheduler agent supervisor, arranges for the steps that make up the task to be executed and orchestrates their operation. These steps can be combined into a pipeline or workflow. The scheduler is responsible for ensuring that the steps in this workflow are performed in the right order. The self-recovery of services is being termed as one of the paramount properties of new-generation software services.

# Sharding pattern

In the big data world, data stores need to store a humongous amount of data. Due to the extraordinary growth of data collection, storage, processing, and analysis, there arise several operational and management challenges including storage space. Furthermore, interactive querying and data retrieval are also difficult.

Data is becoming big data that in turn promises big insights. Batch and real-time processing of big data are also mandated by business houses. The new normal is poly-structured data. Thus, massive amounts of multi-structured data structurally and operationally challenge the traditional SQL database management systems. That is, in the new world order, NoSQL and NewSQL databases are very popular. The prime reason for this new trend is the faster maturity and stability of sharding, which is unambiguously partitioning big databases into smaller and manageable databases. These segregated databases are being run in different and distributed commoditized server machines. The sharding intrinsically supports horizontal scalability (scale out), whereas the SQL databases support the scale up (vertical scalability). The runtime incorporation of schema changes is also being supported by NoSQL databases.

This pattern has the following benefits:

*   The database system can scale out by adding further shards running on additional storage nodes
*   A system can use off-the-shelf hardware rather than specialized and expensive computers for each storage node
*   You can reduce contention and improve performance by balancing the workload across shards
*   In the cloud, shards can be located physically close to the users that'll access the data

# Throttling pattern

The load on a cloud application typically varies over time based on the number of active users or the types of activities they are performing. There can be more users during business hours. During festivities, more users will come to electronic commerce and e-business applications. There might also be sudden and unanticipated bursts in activity. If the processing requirements of the system exceed the capacity of the resources that are available, it will suffer from poor performance and can even fail. If the system has to meet an agreed level of service, such kinds of failures could be unacceptable.

There are several strategies and workarounds for tackling this important challenge. A viable solution is to use resources only up to a limit and then throttle them when the assigned limit is reached. An alternative strategy to auto-scaling is to allow applications to use resources only up to a limit, and then throttle them when this limit is reached.

The system should monitor how it's using resources so that, when usage exceeds the threshold, it can throttle requests from one or more users. This will enable the system to continue functioning and meet any **service level agreements** (**SLAs**) that are in place.

# Workload distribution pattern

IT resources and business workloads are sometimes subjected to heavy usage. When the number of users goes up sharply, the problems, such as performance degradation, reduced availability, reliability, and so on, can arise and choke the system. There are a few interesting solutions being recommended for overcoming these issues. Horizontal scalability and the leverage of load balancers in front of web, application, and database servers are being widely and wisely implemented in order to fulfill the agreed SLAs between the providers and the users. Workload instances need to be distributed to tackle heavy user loads.

# Cloud workload scheduler pattern

The cloud workload scheduler automates, monitors, and controls the workload throughout the cloud infrastructure. This automation usually manages hundreds of thousands of workloads per day from a single point of control. The cloud scheduler could also be an orchestration engine automatically scheduling workloads. The scheduler must be provided, the security level required by the workload.

There are fresh design patterns for accelerating cloud application design. While the cloud idea is progressing fast and is seeing a surging popularity, there can be additional design patterns. We will come across exclusive and elegant patterns for cloud brokerage services and orchestration capabilities. There will be focuses on unearthing competent solutions and patterns for deeper and decisive automation of cloud activities. **Self-service** is another buzzword being given extreme importance so that clouds become business-friendly and business-aware. Serverless computing is another pragmatic and popular topic of deeper study and research in the cloud arena. Docker-enabled containerization is the mainstream topic of deliberations and discourses, and in the near future, we will hear more about containerized cloud infrastructures, platforms, and application workloads. Highly beneficial design patterns will emerge and empower next-generation cloud applications.

# Cloud reliability and resilience patterns

The reliability or dependability of cloud applications has to be ensured through technologically sound solutions. Cloud infrastructures too have to be accordingly empowered to be reliable. The second aspect is cloud resilience. As business workloads and IT platforms are being increasingly modernized and moved to cloud environments, the need for cloud resilience has gone up drastically. Viable mechanisms are being worked by cloud professionals in order to boost the confidence of people on the cloud paradigm. Having competent patterns for those recurring requirements and common problems is one sure way for tackling the QoS and QoE factors. This section is dedicated to illustrating prominent cloud reliability and resilience patterns.

# Resource pooling pattern

For scalability purposes, IT resources have to be pooled in order to provide additional resources on a need basis. The auto-scaling capability can be realized when the appropriate resources are pooled. The challenge here is that of manually establishing and maintaining the level of required synchronicity across a collection of shared resources. Any kind of variance or disparity among shared IT resources potentially can lead to inconsistency and sometimes result in risky operations. The solution is to get identical IT resources and pool them to be leveraged when necessary. The key resources include **bare metal** (**BM**) servers, **virtual machines** (**VMs**), and containers. Furthermore, the fine-grained resources include memory, storage, processing cores, I/O, and so on. There are several monitoring, measurement, and management tools in place for resource provisioning, replication, and utilization.

# Resource reservation pattern

Capacity planning is an important factor in realizing highly optimized IT infrastructures and resources for meeting the various tricky demands of applications. If not properly done, then there is a possibility of getting into the issue of resource constraints. When more cloud consumers try to access a shared IT resource, which does not have the capacity to fulfill the consumers' processing needs, then this condition of resource constraint creeps in. The result may be performance degradation or even the request may not be fulfilled at all. Depending on how IT resources are designed for shared usage and depending on their available levels of capacity, concurrent access can lead to a runtime exception condition called **resource constraint**.

A resource constraint is a condition that occurs when two or more cloud consumers have been allocated to share an IT resource that does not have the capacity to accommodate the entire processing requirements of the cloud consumers. As a result, one or more of the consumers will encounter a sort of degraded performance or be rejected altogether.

The solution is primarily dynamic capacity planning and to have an IT resource reservation system in order to protect cloud service consumers. This reservation system guarantees a minimum amount of IT resources for each cloud consumer.

# Hypervisor clustering pattern

Any kind of IT infrastructures and resources can go down at any point in time. It is good practice to expect failure of IT systems in order to design IT systems in a better-informed fashion. Now hypervisors, alternatively touted as **virtual machine monitors** (**VMMs**), represent an additional abstraction in order to emulate underlying hardware. The issue is that hypervisors too are liable failure. When hypervisors fail, then all the virtual machines on them are bound to fail. Thus, it becomes critical for the high-availability of hypervisors.

A high-availability hypervisor cluster is created to establish a group of hypervisors that span physical servers. As a result, if a given physical server or hypervisor becomes unavailable, hosted virtual servers can be moved to another physical server or hypervisor.

# Redundant storage pattern

Cloud storage is gaining a lot of attention these days because of the enhanced flexibility and extreme affordability. There are block storage, object storage, file storage, and so on. Storage devices are also subject to failure and disruption due to a variety of causes including network connectivity issues, storage controller failures, general hardware failure, and security breaches. When a cloud storage system gets compromised, the result will be unprecedented. A secondary cloud storage device is incorporated into a system that synchronizes its data with the data in the primary cloud storage device. When the primary device fails, a storage service gateway diverts requests to the secondary device automatically to fulfill the **business continuity** (**BC**) requirements.

This pattern fundamentally relies on the resource replication mechanism to keep the primary cloud storage device synchronized with secondary cloud storage devices. Cloud service providers may put secondary storage appliances in a geographically different location for ensuring data and disaster recovery.

# Dynamic failure detection and recovery pattern

Cloud environments comprise a large number of IT infrastructures in a consolidated and centralized fashion in order to fulfill the variable IT needs of worldwide consumers. Cloud environments ensure self-service capability. The major portions of the IT infrastructures are virtualized, shared, and commoditized servers, storage appliances, and networking solutions. The failure rate is quite high and hence failure detection proactively is turning out to be a key requirement for successfully running cloud environments.

A resilient watchdog system has to be established to monitor, measure, and respond to a wider range of predefined failure scenarios. This system is further able to notify and escalate certain failure conditions that it cannot automatically solve itself.

# Redundant physical connection for virtual servers pattern

A virtual server is connected to an external network through a virtual switch uplink port. If the uplink fails (due to cable disconnection, port failure, or any other accidents and incidents), the virtual server becomes isolated and disconnects from the external network. One or more redundant uplink connections are established and positioned in standby mode. A redundant uplink connection is available to take over as the active uplink connection whenever the primary uplink connection becomes unavailable or experiences failure conditions.

Cloud environments promise to have some unique capabilities, such as infrastructure elasticity and application scalability. These enhance cloud availability. There are techniques and patterns being experimented in order to guarantee cloud reliability/dependability. Furthermore, resiliency is being given the sufficient thrust by cloud professors so that the goals of reliability and resiliency out of cloud assets can be met quite easily and quickly.

# Cloud security patterns

As widely accepted and articulated, the issue of cloud security has been the principal barrier for individuals, institutions, and innovators towards readily and confidently leveraging the cloud environments; especially the public clouds for hosting and delivering their enterprise-grade, business-critical, and high-performance applications and databases (customer, corporate, and confidential). In this section, we will discuss some of the prominent cloud security patterns in order to empower cloud security architects, consultants, and evangelists with all the right details.

# Cryptographic key management system pattern

Cryptography is the unique approach to ensuring data security. Encryption and decryption are the two major components of this mathematical theory. Keys are generated and stored securely and are used fluently for securing data while in transit, rest, and being used by software applications. The worry here is how to safely and securely keep the keys generated. If the keys are somehow lost, then the encrypted data cannot be decrypted. Therefore, the industry recommends having a **cryptographic key management system** (**CKMS**), which consists of policies, procedures, components, and devices that are used to protect, manage, and distribute cryptographic keys and certain specific information, called **metadata**. A CKMS includes all devices or subsystems that can access an unencrypted key or its metadata. Encrypted keys and their cryptographically protected metadata can be handled by computers and transmitted through communication systems and stored in media that are not considered to be part of a CKMS.

# Virtual private network (VPN) pattern

In the connected world, the internet is the cheap, open, flexible and public communication infrastructure. The **virtual private network** (**VPN**) is a network that uses a public telecommunication infrastructure, such as the internet, to provide consumers with secure connections to their organization's network. The VPN ensures privacy through security procedures and tunneling protocols, including the **layer two tunneling protocol** (**L2TP**). Data is encrypted at the sending end for transmission and decrypted at the receiving end, as shown in the following figure:

![](img/b9bd0d01-9d19-490d-b51a-6003e6ca249e.png)

The figure shows two firewalls establishing a VPN between two clouds. They first exchange each other's certificates and use asymmetric encryption to securely exchange keying material to establish efficient symmetric key encryption. IPsec is a framework of open standards for private communications over public networks. It is a network layer security control that is used to create the VPN.

# Cloud authentication gateway pattern

Cloud consumers are compelled to support multiple authentications, communication, and session protocols in order to access and use various cloud services. An authentication service authenticates cloud consumers to access cloud services. The authentication service uses the diverse protocols required by cloud service providers for authenticating cloud consumers.

An **authentication gateway service** (**AGS**) can be established as a reverse proxy frontend between the cloud consumer and the cloud resource. This AGS intercepts and terminates the consumer's encrypted network connection and authenticates the cloud consumer. Furthermore, it authenticates itself and the consumer to the cloud provider and then proxies all communication between the two.

# In-transit cloud data encryption pattern

Data security is an important component for the continued growth of the cloud concept. With data analytics gaining widespread significance, the need for secure data capture, transmission, exchange, persistence, and usage has grown greatly. Data transmission networks, data management systems, data analytics platforms, data storage appliances, filesystems, and so on, are the prominent ingredients for the next-generation knowledge era. Encryption is the primary mechanism for securing data interchanged between data sources and servers.

# Cloud storage device masking pattern

As illustrated previously, data security is essential for boosting the confidence of cloud consumers on cloud-based enterprise applications and databases. Authorized data access is the foremost thing for ensuring utmost data security. Data stored in a shared cloud environment can be vulnerable to many security risks, threats, vulnerabilities, and holes. An LUN masking mechanism can enforce defined policies at the physical storage array in order to prevent unauthorized cloud consumers from accessing a specific cloud storage device in a shared cloud environment.

# Cloud storage data at rest encryption pattern

Data stored in a cloud environment requires security against access to the physical hard disks forming the cloud storage device. The solution is to leverage any encryption mechanism supported by the physical storage arrays to automatically encrypt data stored on the disks and decrypt data leaving the disks.

# Endpoint threat detection and response pattern

Endpoint security refers to the protection of an organization's network when accessed through remote devices, such as smartphones, tablets, and laptops. **Endpoint threat detection and response** (**ETDR**) focuses on the endpoint as opposed to the network. It is recommended to leverage integrated security tools in order to understand the security holes of edge devices in order to strengthen the cloud networks and servers.

# Threat intelligence processing pattern

The act of analytics is becoming pervasive these days. Operational, behavioral, security, log, and performance data of IT environments are consciously collected and subjected to a variety of investigations. Deeper and decisive analytics on security-related data emits a lot of useful information for security analysts, architects, and advisors. The extracted insights come in handy in proactively putting appropriate security mechanisms in place in order to ward off any kind of security attacks and exploitations.

A threat intelligence system can be put in place to receive and process external intelligence feeds as well as to gain intelligence gained from analyzing attacks internally. The details received and collected can be fed into security-enablement systems, such as security information and **event management systems** (**SIEMs**), **network forensics monitors** (**NFM**), **endpoint threat detection and response systems** (**ETDRs**), **intrusion detection and protections systems** (**IDPSs**), and so on. Also, those sensitive details can be shared across cloud security operational teams to enable them to ponder and proceed with the best course of action.

# Cloud denial of service (DoS) protection pattern

Cloud DoS attacks are multifaceted and prevent consumers of cloud services from accessing their cloud resources. A cloud DoS protection service has to be incorporated into the security architecture to shield the cloud provider from DoS attacks. A network DoS protection service updates the **domain name service** (**DNS**) to route all cloud provider traffic through the protection service, which filters attack traffic and routes only legitimate traffic to the cloud provider. Alternately, the cloud provider can route traffic to a DoS protection service when experiencing an attack, or create its own DoS protection service. Considering the insistence for unbreakable and impenetrable cloud security solutions, fresh cloud security patterns are being unearthed by security experts and researchers. In the future, there will be a few more security-related patterns.

# Summary

Patterns have been the principal enabling tools for strategic and simplified design and engineering of all kinds of business and social systems. The IT domain too has embraced the proven and potential concept of patterns in order to overcome the inherent limitations of IT systems and services engineering. This chapter is specially prepared for describing the various architectural and design patterns being unearthed and articulated by various cloud computing professionals. The readers can find the right amount of detail for each of the patterns. With the cloud paradigm on the fast track, there is a need for detailing various and recently articulated cloud patterns and their correct details. This chapter comes in handy for interested IT people in understanding cloud-related patterns.

# Bibliography

The cloud patterns registry: [http://cloudpatterns.org/](http://cloudpatterns.org/)

Cloud design patterns by Microsoft: [https://docs.microsoft.com/en-us/azure/architecture/patterns/](https://docs.microsoft.com/en-us/azure/architecture/patterns/)

Cloud computing patterns: [http://www.cloudcomputingpatterns.org/](http://www.cloudcomputingpatterns.org/)

Cloud design patterns by Amazon Web Services: [http://en.clouddesignpattern.org/index.php/Main_Page](http://en.clouddesignpattern.org/index.php/Main_Page)

Cloud architecture patterns: [http://shop.oreilly.com/product/0636920023777.do](http://shop.oreilly.com/product/0636920023777.do)