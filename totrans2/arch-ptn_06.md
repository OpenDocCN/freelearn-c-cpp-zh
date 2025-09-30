# Enterprise Architecture Platforms and Tools

This chapter contains two main sections.The objective of the chapter is to provide an overview of the two prominent enterprise patterns which are used in the industry nowadays. Some of the prominent Enterprise Architecture platforms and tools are also covered in this chapter. The first section focuses on the two popular enterprise architecture framework that are used nowadays:

*   **The open group architecture framework** (**TOGAF**)
*   Zachman framework

In the second section, we will focus on the prominent **enterprise architecture** (**EA**) platforms and tools that are used by organizations. We will cover the following popular platforms:

*   Enterprise architect
*   Dragon1
*   ABACUS

# Overview of enterprise architecture frameworks

An **enterprise architecture framework** (**EAF**) helps to map all the software-related development processes within an enterprise to fulfill the goals and objectives of the enterprise. EAF also provides a framework for the organizations to analyze and understand their weaknesses and inconsistencies. There are many popular and well established EAF frameworks that exist in the industry today. Some of them were developed for specific areas, whereas others have a broader scope. Some of the EAF frameworks that exist in the market are the following:

*   **Department of defense architecture framework** (**DoDAF**)
*   **Federal enterprise architecture framework** (**FEAF**)
*   **Treasury enterprise architecture framework** (**TEAF**)
*   **The open group architecture framework** (**TOGAF**)
*   Zachman framework for enterprise architecture

Though there are four or five prominent EAFs in the industry, the most popular and widely used ones are TOGAF and Zachman framework for enterprise architecture. Hence in this chapter, our discussions will be focused only on these two frameworks.

# Getting started with TOGAF

TOGAF is an extremely popular architecture framework that is used to design an enterprise architecture. It offers all the toolsets and techniques that are used in the design, production, and maintenance of enterprise architecture. It is developed based on a process model that uses industry best practices and a set of reusable architecture assets.

As per TOGAF, **architecture** is defined as *the fundamental organization of a system, embodied in its components, their relationships to each other and the environment, and the principles governing its design and evolution* (you can refer to [http://pubs.opengroup.org/architecture/togaf9-doc/arch/](http://pubs.opengroup.org/architecture/togaf9-doc/arch/) for more information). In short, the architecture of a system provides an elaborate plan for its implementation. The architecture also highlights the various components that are present in the system and the interrelationships that exist among them.

TOGAF is designed to support four architecture domains of enterprise architecture. These four domains are highlighted in the following diagram:

![](img/b61f6d3f-7f7b-442a-916c-74700ec3039f.png)

Each of the architecture domains listed in the previous diagram plays a vital role in defining the architecture of an enterprise. The roles are listed as follows:

*   The **business architecture** provides a blueprint of the overall business activities, such as the business strategy, organization, core business processes, and so on.
*   The **data architecture** provides a blueprint of the organization's data assets, be it logical or physical. It also specifies the various data management resources of the organization.
*   The **application architecture** provides a blueprint for the various applications that must be deployed in the organization, along with their interactions and dependencies on the various business processes that are present in the organization.
*   The **technology architecture** provides a high-level description of the hardware and software assets that are required to support the various data, application, and business services that are needed in the organization. Technology architecture focuses mainly on the infrastructure components, processing standards, and so on.

# Architecture development method (ADM)

The core of the TOGAF architecture is the ADM. It provides a set of tested and repeatable processes for developing enterprise architectures.

The following are the key activities that are captured in the ADM:

*   Establish a framework for architecture
*   Develop content for architecture
*   Provide guidance for realization of architectures

All the ADM activities follow an iterative cycle, which includes both architecture definition and realization. This helps the organizations to transform their architectures in a step-wise manner that is aligned with their business goals and opportunities. The phases of an ADM are shown in the following diagram:

![](img/6e931f0f-a949-4d90-ab89-4e203264b2b8.png)

Activities that happen in each phase within the ADM are explained as follows:

*   **Preliminary phase**: The main initiation activities that are required for architecture capability building are done. Some examples of activities that are done in this phase are customization of TOGAF, the definition of principles to be used for architecture design, and so on.
*   **Phase A - Architecture vision**: In this initial phase, the key actors are involved in the definition of scope. Some other activities are stakeholder identification, getting all necessary approvals for architecture design and development, and so on.
*   **Phase B - Business architecture**: In this phase, an agreement is made to develop a business architecture that is aligned with the architecture vision of the organization.
*   **Phase C - Information systems architecture**: In this phase, an agreement is made to develop an information system architecture that is aligned with the architecture vision of the organization.
*   **Phase D - Technology architecture**: It mainly deals with developing a technology blueprint that is aligned with the architecture vision of the organization.
*   **Phase E - Opportunities and solutions**: It deals with doing the initial implementation planning and identification of different formats of architecture delivery.
*   **Phase F - Migration planning**: It mainly deals with the steps involved in moving from a baseline to final target architectures. The various steps involved in migration are generally captured in an implementation and migration plan.
*   **Phase G - Implementation governance**: It provides an overview of the implementation.
*   **Phase H - Architecture change management**: It deals with carving out a change management plan to handle changes that come up in the architecture.
*   **Requirements management**: It deals with managing the architecture requirements that evolve throughout the various phases of ADM.

# Deliverables, artifacts, and building blocks

Throughout the execution of an ADM, several types of outputs are produced. Some of them are process flows, project plans, compliance assessments, and so on. TOGAF provides an architecture content framework that offers a structural model for the architectural content. This structural model allows several types of work products to be defined, structured, and presented in a consistent manner.

The architecture content framework basically uses three types of categories to denote the specific type of architectural work product under consideration. They are the following:

![](img/29375aa8-ec35-4135-aee2-5e3689c36405.png)

A **deliverable** is a type of work product that is reviewed and agreed upon formally by the stakeholders. Deliverables are typical outputs of projects and they are in the form of documents. These deliverables are either archived or transferred to an architecture repository as a model standard at the time of completion of the project.

An **artifact** is a type of work product that describes some specific aspect of an architecture.

Some important categories of artifacts are as follows:

*   Catalogs (list things)
*   Matrices (show relationship between various things)
*   Diagrams (depict pictures of things)

Some common examples are a requirements catalog, use-case diagram, interaction diagram, and so on.

A **building block** denotes a fundamental component of IT or architectural capability that can potentially be combined with other building blocks to develop and deliver architectures.

Building blocks can be specified at various levels of detail based on the stage at which architecture development of the system has reached. For example, at very initial stages, the building block could be just a name that may later get involved into a complete specification of the component and its design.

There are two types of building blocks, they are:

*   **Architecture building blocks** (**ABBs**): They describe the capability that is expected from the architecture. This capability then describes the specification that will be used for making the building blocks of the solution. For example, customer service could be an example of a capability that is needed within an enterprise, which may have several solutions blocks such as applications, processes, data, and so on.
*   **Solution building blocks** (**SBBs**): They denote the various components that will be used in the implementation of the required capability.

The relationships between deliverables, artifacts, and building blocks are depicted in the following diagram:

![](img/c6720c3c-3c6c-4f4d-9b8c-d45c1fea0497.png)

All the artifacts pertaining to architecture are interrelated in some way or the other. A specific  architecture definition document may refer several other complementary artifacts. The artifacts could belong to various building blocks which are part of the architecture under consideration. The following example pertains to the target call handling process. The various references to other building blocks are depicted in the following diagram:

![](img/0cd95f5f-3bb6-4a89-8380-b1895ed4ae8f.png)

# Enterprise continuum

TOGAF includes a concept called **enterprise continuum**. This concept explains how certain generic solutions can be customized and used as per specific requirements of an organization. Enterprise continuum provides a view of the architecture repository that provides ways and techniques for classifying architecture and other related artifacts as they transform from generic architecture to specific architectures that are suitable for specific needs of the organization. Enterprise continuum has two complementary concepts associated with it, they are:

*   Architecture continuum
*   Solutions continuum

The architecture of enterprise continuum is depicted in the following diagram:

![](img/8783121b-eba4-407e-9bcb-3a5611540e76.png)

# Architecture repository

Another important concept of TOGAF is architecture repository. This can be used to store diverse types of architectural outputs, each at varying levels of abstraction; these outputs are created by ADM. This concept of TOGAF helps to provide cooperation and collaboration between practitioners and architects who are working at various levels in an organization.

Both enterprise continuum and architecture repository allow architects to use all architectural resources and assets that are available in an organization-specific architecture.

In general, TOGAF ADM can be considered typically as a process lifecycle that operates at various levels in an organization. ADM operates under a holistic governance framework and produces outputs that are placed in an architecture repository. Enterprise continuum provides a very good context for understanding the various architectural models, the building blocks, and the relationship of the building blocks to each other. The structure of TOGAF architecture repository is given in the following diagram:

![](img/4a0a0c20-1838-46ad-aab2-b6fd6f875a49.png)

The following list shows major components of an architecture repository and the functionalities provided by those components:

*   **Architecture metamodel**: This component describes the architecture framework, which is tailor-made as per the needs of the organization.
*   **Architecture capability**: This component describes parameters, processes, and so on that support governance of the architecture repository.
*   **Architecture landscape**: This is the representation of architectural assets that are deployed within an organization at any point in time. There is always a possibility that the landscape exists at various levels of abstraction, which are aligned to different sets of architecture objectives.
*   **Standards information base**: This component describes the standards to which new architectures must comply. Standards in this context may include industry standards, standards from products and services that are deployed in the organization, and so on.
*   **Reference library**: This component provides guidelines, templates, and so on that can be used as a reference to create new architectures for the organization.
*   **Governance log**: This component maintains a log of governance activity that happens across the enterprise.

# Advantages of using TOGAF

The following are the main benefits of using TOGAF for EA design:

*   The TOGAF framework provides a good understanding of the techniques to be used to integrate architecture development with strategies that are aligned with the objectives of the organization
*   The framework provides well-defined guidelines on the steps to integrate architecture governance with IT governance
*   The framework provides many checklists on how to support IT governance within the organization
*   The framework provides a lot of reusable artifacts that can be used to create diverse types of architecture for organizations based on the varying requirements
*   The framework provides a lot of options to reduce IT operating costs and helps in the design of portable applications

# Limitations of TOGAF

There are certain limitations too, which are listed as follows:

*   The framework plays the role of a design authority in an enterprise and offers very few features for the architects to maintain enterprise-wide descriptions, standards, principles, and so on
*   The framework provides very limited guidance to solution architects
*   The framework assumes that enterprises will have their own processes that will be integrated with TOGAF
*   It is just a framework, not a modeling language or any other component that could be treated as a replacement for architect skills
*   It is not very consistent with the metamodels it supports

In the next topic, we will examine the details of the Zachman framework, which has also gained a lot of popularity and traction in the enterprise architecture domain.

# Zachman framework for enterprise architecture

*The* Zachman framework was published by John Zachman for EA in 1987\. Zachman was motivated by increased levels of complexity involved in the design of information systems, which forced him to think of a logical construct for designing the architecture of enterprises, which in turn led to the development of the Zachman framework for enterprise architecture. The framework does not focus much on providing any form of guidance on sequence, process, or implementation. The core focus is to ensure that all views are well established, ensuring a complete system regardless of the order in which they were established. The Zachman framework does not have any explicit compliance rules as it does not belong to the category of a standard written by a professional organization.

The Zachman framework was initially developed for IBM but now has been standardized for use across enterprises. The main motivation behind the Zachman framework is to derive a simple logical structure for enterprises by classifying and organizing the various components of an enterprise in a manner that enables easy management of enterprises and facilitates easy development of enterprise systems such as manual systems and automated systems. The simplest form of the Zachman framework has the following depictions:

*   Perspectives depicted in the design process, that is owner, designer, and builder.
*   Product abstractions, such as what (material it is made of) and how (a process by which it works).
*   Where (geometry by which components are related to one another), who (operating instructions) is doing what kind of work, when (timing of when things happen), why (engineering aspects due to which things happen). In some of the older versions of the framework, there were some additional perspectives present such as planner, sub-contractor, and so on.

The various perspectives that are typically used in the Zachman framework, as well as their roles in the enterprise architecture landscape, are as follows:

*   **Planner**: A planner positions the product in the context of its environment and specifies the scope
*   **Owner**: An owner will be interested in the business benefits of the product, how it will be used in the organization, and the added value it will offer to the organization
*   **Designer**: A designer will carve out the specifications of the product to ensure that it meets the expectations of the owner. All aspects of product design are taken care of by the designer
*   **Builder**: A builder manages the process of assembling various components of the product
*   **Sub-contractor**: A sub-contractor incorporates out-of-context components that are specified by the builder

Please note that perspectives with respect to the Zachman framework keep changing as per the enterprise landscape.

The simplest depiction of the framework has the following components:

![](img/39b7329a-d24b-4416-904a-6a0d7cbb81e8.png)

# Advantages

The following are the main advantages of the Zachman framework:

*   It provides a framework for improving several types of professional communication within the organization
*   It provides details about the reasons and risks of not using any architectural perspective
*   It provides options to explore and compare a wide variety of tools and/or methodologies
*   It has options that will suggest the development of new and improved approaches for producing various architectural representations

# Restrictions

The Zachman framework has the following limitations:

*   It can lead to a process-heavy/documentation-heavy approach as there is a lot of data that needs to be filled out in the cells that are used to capture the data pertaining to the framework
*   It does not provide a step-by-step process for designing a new architecture
*   It does not provide any guidelines for assessing the suitability of a future architecture for an organization
*   It does not provide a framework for implementing governance in an architecture

# Guidelines for choosing EAF

Given an option to choose the architecture that is best suited for your enterprise, what are the parameters you will use to make a decision? The following table helps you to choose one based on some common parameters that are prominent in the industry landscape (in a five-point scale):

![](img/f6ee1fcc-845a-4de0-bb4d-1e10450c42ef.png)

Some key terms used in the table are as follows:

*   **Process completeness**: This criterion helps to find the level of step-by-step guidance provided by the framework for architecture implementation
*   **Business focus**: This criterion helps us to find the technology choice flexibility which in turn will help in alignment with business objectives
*   **Partitioning guidance**: This criterion helps to judge the flexibility offered by the framework for partitioning of the enterprise to manage complexity effectively
*   **Time to value**: This criterion provides guidance on the time taken by a solution built using this framework to deliver business value to the organization

In the next section, we will examine the prominent platforms and tools that are available for deployment/design of enterprise architecture.

# Enterprise architecture platforms and tools

The following are some of the main parameters to be considered by enterprise architects while choosing an enterprise architecture platform:

*   **Innovation**: Enterprise architects will need a lot of features that will enable them to think and work in an innovative manner. At the same time, they should have access to all tools and features that are available in any EA environment.
*   **Visualization**: Most of the EA tools also perform the function of business support tools. In such a scenario, it becomes necessary that the tool offers a lot of rich visualization and animation features, which are expected as a part of normal business support activities.
*   **Mapping and modeling**: One of the most important feature requirements of EA tools is modeling. The tools should be able to provide diverse types of modeling such as contextual modeling, logical modeling, and physical modeling. The need for advanced modeling capabilities becomes more prominent in the context of present-day digital businesses with a lot of customer centricity.
*   **Analysis and design**: One of the key requirements of any EA tool is to support analysis and design. Now, because of the changes in the enterprise landscape, it becomes necessary for the tools to support advanced features such as business intelligence.
*   **Speeding time to value**: EA tools should be able to provide features that enable easy integration with a lot of third-party tools and interfaces. These capabilities will help them to deliver business value quickly.
*   **Business architecture design**: EA tools should offer features that help in accommodating rapidly changing features as a part of architecture design. They should also provide features to develop new types of business models quickly.

In the next section, we will examine some popular platforms and tools that are available for the design and development of enterprise architectures.

# Enterprise Architect from Sparx Systems

It is a comprehensive enterprise architecture platform that offers the following core capabilities pertaining to enterprise architecture design:

*   Analysis
*   Modeling
*   Design
*   Construction
*   Testing and management

This tool offers integrated support and full traceability between all tasks, phases, components, domain, and lifecycle management of enterprise architecture. Enterprise Architect combines the rich UML toolset with a high performance and interactive user interface to provide an advanced toolset for the enterprise architects.

The intuitive user interface of the Enterprise Architect tool is depicted in the following screenshot:

![](img/4fcfe7dc-45cd-4c69-a577-901bea944a79.png)

The following are the main industries supported by the Enterprise Architect tool:

*   Aerospace
*   Banking
*   Web development
*   Engineering
*   Finance
*   Medicine
*   Military
*   Research and academia
*   Transport
*   Retail
*   Utilities

Enterprise Architect is also a tool that is widely used by standard organizations across the world to organize their knowledge, models, and messages. This tool has been continuously updated as per the changes in the UML standards.

Enterprise Architect is a proven, scalable, effective, and affordable full life cycle development platform for:

*   Strategic modeling
*   Requirements gathering
*   Business analysis
*   Software design
*   Software profiling
*   Software testing and debugging
*   Modeling and simulation of business processes
*   Systems and software engineering
*   enterprise architecture design
*   Database design and engineering
*   Design and development of XML schemas
*   Project management activities for various stages of enterprise architecture design and development
*   Testing and maintenance of applications
*   Reporting

Enterprise Architect is optimized for the following activities, which are involved as a part of enterprise architecture design:

*   Creating, capturing, and working with a rich and diverse set of architecture requirements from multiple stakeholders
*   Modeling, designing, and architecting a wide range of software systems as per the requirements of the organization
*   Business analysis, modeling the business process, and strategic modeling as per the needs of the organization
*   Modeling of systems, modeling of system architecture, and component design
*   Comparing, designing, constructing, modeling, and updating database systems
*   Defining and generating schema based on XSD and other standards
*   Creating architecture designs based on domain-specific modeling languages such as UML
*   Simulating and visualizing a wide range of systems, tasks, and associated processes
*   Designing, developing, executing, debugging, and testing codes written in a wide variety of software languages
*   Simulating behavioral processes and designing state machines and their various interactions
*   Building executable codes for types of state machines based on the architectural design and providing an environment that supports simulation of these executables
*   Collaborating and sharing information
*   Providing the capabilities for quality control and testing of complex architectural systems
*   Project managing tasks that are associated with enterprise architecture design and development
*   Team-based collaboration features using the concept of cloud-based repositories that are optimized for access over diverse types of LAN and WAN networks

# Dragon1

Dragon1 is a very popular enterprise architecture platform that offers the following features, which are mandatory for the design and development of enterprise architecture:

*   Technology roadmaps that can be used to derive architectures aligned to business objectives of the organization
*   Heat maps that show the pain points for architecture design in an organization
*   Capability maps that provide a blueprint of the capabilities that exist in an organization
*   Glossary of terms used in architecture design and development
*   Architecture description document
*   Decision matrix sketches of total concepts
*   Drawing on architecture principles
*   enterprise architecture blueprints
*   Application landscapes
*   Models atlas

Dragon1 offers features that can be used by individuals at various levels of an organization. The main groups that will benefit from using Dragon1 in an organization are:

*   **Analysts**: Helps them to do impact analyses
*   **Architects**: Helps diverse types of architects such as business, enterprise, information, IT, and so on, for creating architecture designs as per their domain
*   **Designers**: Helps them to create both functional and technical designs
*   **Managers** (IT and business): Provides toolsets that help them to monitor and manage operations visually through features such as dashboards
*   **Program and project managers**: Monitoring and managing changes in their schedules on a real-time basis visually
*   **CxOs** (CIO, CEO, CFO, and so on): Dashboard view of various organization domains in an easy to understand manner

Some of the core capabilities of Dragon1 are the following:

*   **Publishing and reporting**: Any type of document/information can be uploaded, stored, and published using Dragon1
*   **Data management**: Provides support for storage, updating, and deletion of any type of data
*   **Requirement management**: Provides exhaustive features for requirements gathering from diverse types of stakeholders
*   **Process, application, and metamodelling**: Offers capabilities and features to build models for any type of entity class, such as process, application, and so on
*   **System design**: Provides a rich set of features that allows systems to be designed at conceptual, logical, or physical level
*   **User management**: Provides role-based features that can be used by employees at various levels in an organization
*   **Architecture visualization**: Offers extensive graphical features that help in the creation of rich visualizations
*   **Dashboards and scenarios**: Offers rich features that enable the creation of dashboard features and scenario analysis

To learn more, you can visit [https://www.dragon1.com/products/enterprise-architecture-tool-comparison](https://www.dragon1.com/products/enterprise-architecture-tool-comparison).

# ABACUS from avolution software

Avolution's ABACUS suite is one of the best EA modeling tools as per Gartner's magical quadrant. It comes with a large library of architecture frameworks and patterns for most of the common platforms. It also provides support for data imported from a broad range of modeling solutions and third-party sources.

ABACUS comes in two variants; they are as follows:

*   Standard
*   Professional

ABACUS's standard suite of products offers only architecture modeling functionality, whereas ABACUS's professional suite provides architecture modeling and scenario analysis capability. ABACUS provides enterprise modeling capability based on components, constraints, and connection framework. It also has features for assigning properties for connections and components that can be accessed through a tabular view in the user interface.

# Architecture of ABACUS

ABACUS consists of metamodels and multiple architectures along with a view for each architecture. The solution provides a large set of libraries that are derived from industry standard frameworks such as TOGAF and several others. ABACUS has an XML-based file format that acts as an objects database. Any new file format that is added to ABACUS is stored as another object in an objects database. This feature is extremely helpful for the creation of new architecture models or adding enhancements to the existing metamodels, because these aspects can be done with the help of right-clicks and do not need any changes to the internal database as required by several other EA tools in the market.

The ABACUS approach that is used to define metamodels basically uses three key units, they are:

*   Component
*   Connection
*   Constraints

These key units conform to the IEEE1471 standard. ABACUS ships with different libraries that contain these key units. The list of libraries that are present in ABACUS includes more than 50 prominent architectural patterns. The flexibility of Avolution provides support for a larger number of architectural frameworks when compared to other EA platforms that are available in the market. ABACUS has features that allow users to create new libraries or merge existing libraries to create a new one within a matter of minutes.

Apart from the tools that were discussed in this section, Gartner's magic quadrant, shown in the following image, provides the list of enterprise architecture platforms that are prominent in the enterprise architecture landscape. This could be used by any Enterprise Architect as a basis for decision making. For more on this, visit [https://www.gartner.com/doc/reprints?id=1-2Q45NIB&ct=151020&st=sb](https://www.gartner.com/doc/reprints?id=1-2Q45NIB&ct=151020&st=sb).

# Summary

In the first section of the chapter, we discussed the Zachman framework and the TOGAF framework, which are prominent nowadays. The components of these frameworks and the advantages and disadvantages of each of them were examined. Finally, we provided a set of metrics that could be used to evaluate and choose the best EA framework for an organization based on various parameters. Next, we examined the various popular EA platforms and tools that are used for the design and development of enterprise architecture. We concluded the chapter by providing a list of EA tools that are supported by Gartner's magical quadrant.

# References

[https://www.sparxsystems.com.au/products/ea/](https://www.sparxsystems.com.au/products/ea/)

[https://www.avolutionsoftware.com/downloads/ABACUS%20TA001910ITM.pdf](https://www.avolutionsoftware.com/downloads/ABACUS%20TA001910ITM.pdf)