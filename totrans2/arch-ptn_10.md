# Patterns for Containerized and Reliable Applications

The Docker-enabled containerization paradigm is on the right track to becoming an impactful and insightful technology with a number of crucial advancements being brought in by a growing array of third-party products and tool vendors. Especially, the future belongs to containerized cloud environments with the ready availability of proven container development, deployment, networking, and composition technologies and tools. The Docker-enabled containers in association with orchestration, governance, monitoring, measurement, and management platforms such as Kubernetes, Mesos, and so on, are to contribute immensely to setting up and sustaining next-generation containerized cloud environments that are very famous for delivering enterprise-class, microservices-based, event-driven, service-oriented, cloud-hosted, knowledge-filled, insights-attached, AI-enabled, people-centric, carrier-grade, production-ready, and infrastructure-aware applications. Besides containers, the concepts of microservices and microservices-centric applications acquire special significance. The basic requirement for building reliable applications lies with the faster realization of resilient microservices, which are being positioned as the standard and optimized building-block and deployment unit for the next-generation applications. This chapter focuses on the following topics:

*   The containerization patterns
*   Resilient microservices patterns
*   Reliable applications patterns

# Introduction

Undeniably, Docker is the most popular and powerful technology these days in the **information technology** (**IT**) sector. There are two principal trends in the Docker landscape. Firstly, the open-source Docker platform is being continuously equipped with more right and relevant features and functionalities in order to make it the most exemplary IT platform, not only for software developers, but also for on-premises as well as off-premises IT operational teams. The second trend is the unprecedented adoption of the Docker-inspired containerization technology by various IT service and solution providers across the globe in order to bring forth a growing array of premium offerings to their venerable consumers and clients. The enhanced simplicity in developing fresh software applications, the automated and accelerated deployment of Docker containers, and the extreme maneuverability of Docker containers are being widely touted as the key differentiators for its unprecedented success.

We would like to shed more light on Docker and show why it is being touted as the next best thing for the impending digital, idea, API, knowledge and insightful economy.

# The key drivers for containerization

The first and foremost driver for Docker-enabled containerization is to competently and completely overcome the widely expressed limitations of the virtualization paradigm. Actually, we have been working on the proven virtualization techniques and tools for quite a long time now in order to realize the much-demanded software portability. That is, with the goal of decimating the inhibiting dependency between software and hardware, there have been several initiatives that incidentally include the matured and stabilized virtualization paradigm. Virtualization is a kind of beneficial abstraction, which is accomplished through the incorporation of an additional layer of indirection between hardware resources and software components. Through this freshly introduced abstraction layer (hypervisor or **virtual machine monitor** (**VMM**)), any kind of software application can run on any underlying hardware without any hitch or hurdle. In short, the software portability is being achieved through this middleware layer. However, the much-published portability target is not fully met even by the virtualization technique. The hypervisor software and different data encapsulation formats from different vendors come in the way of ensuring the much-needed application portability. Furthermore, the distribution, version, edition, and patching differences of operating systems and application workloads hinder the smooth portability of workloads across systems and locations.

Similarly, there are various other drawbacks being attached with the virtualization paradigm. In data centers and server farms, the virtualization technique is typically used for creating multiple VMs out of physical machines and each VM has its own **operating system** (**OS**). Through this solid and sound isolation enacted through automated tools and controlled resource-sharing, multiple and heterogeneous applications are being accommodated in a physical machine. That is, the hardware-assisted virtualization enables disparate applications to be run simultaneously on a single physical server. With the virtualization paradigm, various kinds of IT infrastructures (server machines, storage appliances, and networking solutions) become open, programmable, remotely monitorable, manageable, and maintainable. However, because of the verbosity and bloatedness (every VM carries its own operating system), VM provisioning typically takes a few minutes. This is a big setback for real-time and on-demand scalability.

The other widely expressed drawback that is being closely associated with virtualization is that the performance of virtualized systems also goes down due to the excessive usage of precious and expensive IT resources (processing, memory, storage, network bandwidth, and so on). The execution time of virtual machines is on the higher side because of multiple layers ranging from a guest OS, a hypervisor, and the underlying hardware.

Finally, the compute virtualization has flourished, whereas the other closely associated network and storage virtualization concepts are just taking off.  Precisely speaking, building distributed applications and fulfilling varying business expectations mandate for the faster and flexible provisioning, high availability, reliability, scalability, and maneuverability of all the participating IT resources. Compute, storage, and networking components need to work together in accomplishing the varying IT and business needs. With more virtualized elements and entities in an IT environment, the operational complexity is bound to grow rapidly.

Move over to the world of containerization; all the preceding barriers get resolved in a single stroke. That is, the evolving concept of application containerization coolly and confidently contributes to the unprecedented success of the software portability goal. A container generally contains an application/service/process. Along with the primary application, all of its relevant libraries, binaries, files, and other dependencies are stuffed and squeezed together to be packaged and presented as a comprehensive yet compact container. The application containers can be readily shipped, run, and managed in any local as well as remote environments. Containers are exceptionally lightweight, highly portable, rapidly deployable, extensible, horizontally scalable, and so on. Furthermore, many industry leaders have come together to form a kind of consortium to embark on a decisive and deft journey towards the systematic production, packaging, and delivery of industry-strength and standardized containers.

This conscious and collective move makes Docker deeply penetrative and pervasive. The open-source community is simultaneously spearheading the containerization conundrum through an assortment of concerted activities for simplifying and streamlining the containerization concept. The containerization life cycle steps are being automated through a variety of third-party tools.

The Docker ecosystem also grows fast in order to bring in as much automation as possible in the IT landscape. Container clustering and orchestration are gaining a lot of attention, thereby geographically distributed containers and their clusters can be readily linked up to produce bigger and better process-aware and composite containers. The new concept of containerization assists with distributed computing. Containers enable the formation of federated cloud environments in order to accomplish specialized business targets. Cloud service providers and enterprise IT environments are all set to embrace this unique compartmentalization technology in order to escalate the resource utilization and to take the much-insisted infrastructure optimization to the next level. On the performance side, there are sufficient tests showcasing Docker containers achieving the bare metal server performance. In short, the IT agility through the DevOps aspect is being guaranteed through the smart leverage of the Docker-enabled containerization and this, in turn, leads to business agility, adaptivity, and affordability.

# Design patterns for Docker containers

The Docker-enabled containerization is fast emerging and evolving. With the complexity of the container lifecycle management escalating, the need for enabling patterns is being felt. The concerned professionals and pundits are working in unison to formulate and firm up various container-specific patterns. In the days ahead, we will come across many more patterns. Whatever is widely articulated and accepted is concisely presented in this section and in the forthcoming sections.

With the unprecedented proliferation of the Docker-enabled containers in cloud environments (public, private, and fog/edge), Docker enthusiasts, evangelists, and experts consciously bring forth a bevy of enabling patterns. The readers can find them in this section. Let us start with container building patterns. Building Docker images and containers is constrained with a number of challenges and concerns. The Docker patterns need to reach a level of stability.

# Container building patterns

This section describes a few common ways to build Docker images. As per Alex Collins ([https://alexecollins.com/developing-with-docker-building-patterns/](https://alexecollins.com/developing-with-docker-building-patterns/)), there are several choices: `scratch + binary`, `language stack`, and `distribution+ package manager`. The `scratch + binary - scratch` is the most basic base image and it does not contain any files or programs at all. We must build a *standalone binary application* to use this. Here is an example. Firstly, we will build a standalone binary application using Docker. The steps are as follows:

1.  Create an empty directory and then create a `main.go` application:

[PRE0]

2.  Compile the application:

[PRE1]

3.  Create a Dockerfile for the application:

[PRE2]

4.  Finally, build and run the image:

[PRE3]

This outputs `Hello World` in the terminal.

This is suitable for applications that can be packaged as standalone binaries. As there is no language runtime, larger applications are bound to consume more disk space.

Docker provides a number of pre-built base images for the runtime for common languages. Here is an example as follows:

1.  Create a new empty directory and detail the `Main.java` application:

[PRE4]

2.  Now, compile this application using the **Java Development Kit** (**JDK**):

[PRE5]

3.  Create the following Dockerfile with the **Java Runtime Environment** (**JRE**):

[PRE6]

4.  Finally, build and run this Docker image:

[PRE7]

It is faster to deploy this application once the base image is downloaded, and if the same base image is used for many other applications, then the additional layer needed is very small.

To build an image that is not on a supported language stack, it is necessary to roll your own image starting with a distribution, and then it is all about using a package manager to add the mandated dependencies. Linux always contains a package manager.

This comment installs the JRE:

[PRE8]

Now, build and run this base image:

[PRE9]

The advantage is that we can build an application and it is possible to put multiple applications into a single image (using `systemd`).

# Docker image building patterns

As we all know, Docker containers are a fantastic way to optimally and organically encapsulate complex build processes. Typically, any software package requires a host of dependencies. As indicated in [Chapter 9](https://cdp.packtpub.com/architectural_patterns/wp-admin/post.php?post=271&action=edit), *Microservices Architecture Patterns*, every microservice is being developed and delivered as a Docker image. Each microservice has its own code repository (GitHub) and its own CI build job.  Microservices can be coded using any programming language. Let us focus on the Java language here. If a service is built and run using a compiled language (Java, Go, and so on), then the build environment can be separated from the runtime environment. A Java service's `Dockerfile.build` is from the `openjdk-7-jdk` directory and its Dockerfile is from the `openjdk-7-jre` directory which is substantially smaller than JDK.

For the Java programming language, it requires additional tooling and processes before its microservices become executable. However, the JDK are not required when a compiled program is running. Another reason is that the JDK is a bigger package when compared with the **Java Runtime Environment** (**JRE**). Furthermore, it seems farsighted to develop and reuse a repeatable process and a uniform environment for deploying microservices. It is therefore paramount to package the Java tools and packages into containers. This setup allows the building of Java-based microservices on any machine, including a CI server, without any specific environmental requirements such as JDK version, profiling and testing tools, OS, Maven, environment variables, and so on.

Resultantly, for every service, there are two Dockerfiles: one for service runtime and the second is packed with the required tools to build the service. First, it is all about crafting the `Dockerfile.build` file, which can speed up the Maven build. Now, it is straightforward to compile and run the microservice on any machine (local or remote). This segregated approach goes a long way in simplifying the **continuous integration** (**CI**) process.

The recipe is as follows:

1.  **Build file**: Have one Dockerfile with all the tools and packages required to build any service. Name it `Dockerfile.build`.
2.  **Run file**: Have another Dockerfile with all the packages required to run the service. Keep both files along with the service code.
3.  Build a new builder image, create a container from it, and extract build artifacts using volumes or the `docker cp` command.
4.  Build the service image.

Thus, segregating the building process from the runtime process stands well for the intended success of the containerization paradigm. One is to perform a build and another is to ship the results of the first build without the penalty of the build-chain and tooling in the first image. Terra Nullius has posted the relevant details at [http://blog.terranillius.com/post/docker_builder_pattern/](http://blog.terranillius.com/post/docker_builder_pattern/). The builder pattern describes the setup that developers have to follow for building a container. It generally involves two Docker images:

*   A *build* image with all the build tools installed, capable of creating production-ready application files
*   A *service* image capable of running the application

The basic idea behind the builder pattern is simple: create additional Docker images with the required tools (compilers, linkers, and testing tools), and use these images to produce lean, secure, and production-ready Docker images.

# Multi-stage image building pattern

The latest Docker release facilitates the creation of a single Dockerfile that can build multiple helper images with compilers, tools, and tests, and use files from images to produce the *final* Docker image, as vividly illustrated in the following section.

The Docker platform can build Docker images by reading the instructions from a Dockerfile. A Dockerfile is a text file that contains a list of all the commands needed to build a new Docker image. The syntax and core principle of a Dockerfile is pretty simple and straightforward as follows:

[PRE10]

That is, every Dockerfile creates a Docker image. This principle works just fine for basic use cases, but for creating advanced, secure, and lean Docker images, a single Dockerfile is just not enough.

Multi-stage builds are a new feature incorporated in the latest Docker version, and this is interesting for anyone who has struggled to optimize Dockerfiles while keeping them easy to read and maintain. One of the biggest challenges when building Docker images is keeping the image size down. Each instruction in the Dockerfile adds a layer to the image. The software engineer has to clean up any artifacts that are not needed before moving on to the next layer. To write a really efficient Dockerfile, he traditionally needs to employ the shell tricks and other logic to keep the layers as lean and light as possible and to ensure that each layer has the artifacts it needs from the previous layer and nothing else.

It is always common to have one Dockerfile for development and a slimmed-down version of the Dockerfile for production. Maintaining two Dockerfiles is not ideal. With multi-stage builds, he can use multiple `FROM` statements in his Dockerfile. Each `FROM` instruction can use a different base, and each of them begins a new stage of the build. He can selectively copy artifacts from one stage to another, leaving behind everything he doesn't want in the final image. The end result is the same tiny production image as before, with a significant reduction in complexity.

# The pattern for file sharing between containers 

Docker is a popular containerization tool used to package and provide software applications with a filesystem that contains everything they need to run. Docker containers are ephemeral in the sense that they can run for as long as it takes for the command issued in the container to complete. There are occasions wherein applications need access to data, to share data to, or do data persistence after a container is deleted. Typically, Docker images are not suitable for databases; user-generated content for a website and log files that applications have to access to do the required processing. The much-needed persistent access to data is provided with Docker volumes. At some point, the production-ready application files need to be copied from the build container to the host machine. There are two ways of accomplishing that:

*   Using `docker cp`
*   Using `bind-mount volumes`

Matthias Noback ([https://matthiasnoback.nl/2017/04/docker-build-patterns/](https://matthiasnoback.nl/2017/04/docker-build-patterns/)) has supplied the description for both along with an easy-to-understand example.

# Using bind-mount volumes

It is not good to have the compilation step as a part of the build process of the container. The overwhelming expectation is that Docker images need to be highly reusable. If the source code is modified, then it is necessary to rebuild the build image, but it is desired to *run the same build image* again.

Therefore, the compilation step has to be moved to the ENTRYPOINT ([https://docs.docker.com/engine/reference/builder/#entrypoint](https://docs.docker.com/engine/reference/builder/#entrypoint)) or CMD instruction. The source/files shouldn't be part of the build context and instead, mounted as a bind-mount volume inside the running build container.

The advantages are many here. Every time one runs the build container, it will compile the files in the `/project/source/` and produce a new executable in the `/project/target/`. Since `/project` is a bind-mount volume, the executable file is automatically available on the host machine in `target/`. There is no need to explicitly copy it from the container. Once the application files are on the host machine, it will be easy to copy them to the service image, since that can be done using the regular `COPY` instruction.

# Pipes and filters pattern

An application is required to perform a variety of tasks of varying complexity on the information that it receives. A monolithic module could do this, but there are several inflexibilities. Suppose an application receives and processes data from two sources. The data from each source is processed by a separate module that performs a series of tasks to transform this data, before passing the result to the business logic of the application. The processing tasks performed by each module or the deployment requirements for each task could change. Some tasks might be compute-intensive and could benefit from running on powerful hardware, while others might not require such expensive resources. Also, additional processing might be required in the future, or the order in which the tasks are performed by the processing could change.

The viable solution is to break down the processing required for each data stream into a set of separate components (or filters), each performing a single task. By standardizing the format of the data that each component receives and sends, these filters can be combined together into a pipeline. This helps to avoid duplicating code and makes it easy to remove, replace, or integrate additional components if the processing requirements change.

The time it takes to process a single request depends on the speed of the slowest filter in the pipeline. One or more filters could be a bottleneck, especially if a large number of requests appear in a stream from a particular data source. A key advantage of the pipeline structure is that it provides opportunities for running parallel instances of slow filters, enabling the system to spread the load and improve throughput.

The filters that make up a pipeline can run on different machines, enabling them to be scaled independently and take advantage of the elasticity that many cloud environments provide. A filter that is computationally intensive can run on high-performance hardware, while other less demanding filters can be hosted on less expensive commodity hardware.

If the input and output of a filter are structured as a stream, it is possible to perform the processing for each filter in parallel. The first filter in the pipeline can start its work and output its results, which are passed directly on to the next filter in the sequence before the first filter has completed its work. If a filter fails or the machine it's running on is no longer available, the pipeline can reschedule the work that the filter was performing and direct this work to another instance of the component.

By using the proven pipes and filters pattern in conjunction with the compensating transaction pattern, there is an alternative approach to implement the complex distributed transactions. A distributed transaction can be broken down into separate and compensable tasks, each of which can be implemented by using a filter that also implements the compensating transaction pattern. The filters in a pipeline can be implemented as separate hosted tasks running close to the data that they maintain, thus there emerge newer possibilities.

For the container world, the preceding pattern is beneficial. That is, for taking the generated files out of a container, streaming the file to `stdout` leveraging the preceding *pipes and filters* pattern is being made out as an interesting workaround. This streaming has many advantages too:

*   The data doesn't have to end up in a file anymore as it can stay in memory. This offers faster access to the data.
*   Using `stdout` allows sending the output directly to some other process using the pipe operator (`|`). Other processes may modify the output, then do the same thing, or store the final result in a file.
*   The exact location of files becomes irrelevant. There is no coupling through the filesystem if we only use `stdin` and `stdout`. The build container would not have to put its files in `/target`, and the build script would not have to look in `/target`, they just pass along data.

# Containerized applications - Autopilot pattern

Deploying containerized applications and connecting them together is a definite challenge because typically, cloud-native applications are made up of hundreds of microservices. Microservice architectures provide organizations with a tool to manage the burgeoning complexity of the development process, and application containers provide a new means to manage the dependencies to accelerate the deployment of those microservices. But deploying and connecting those services together is still a challenge.

Operationalizing microservices-based applications brings forth several challenges. Developers have to embed several things inside for simplified deployment and delivery. Autopilot applications are a powerful design pattern for solving these problems. The autopilot pattern automates in the code the repetitive and boring operational tasks of an application, including start-up, shutdown, scaling, and recovery from anticipated failure conditions for reliability, ease of use, and improved productivity. By embedding the distinct responsibility and the operational tasks into the application, the workload on operational team members is bound to come down.

The autopilot pattern is for both developers and operators. It is for operators that want to bring sanity to their lives and for developers who want to make their applications easy to use. It is primarily for microservices applications and multi-tiered applications. Most importantly, it is designed to live and grow with our applications at all stages of development and operations.

The autopilot pattern automates the life cycle of each component of the application. There can be multiple components in any application. Web and application server, DB server, in-memory cache, reverse proxy, and so on, are the most prominent components for any application. Each of these components can be containerized and each container contributing for the application has its own life cycle. Most autopilot pattern implementations embrace single-purpose or single-service containers. The autopilot pattern does require developers and operators to think about how the application is operated at critical points in the life cycle of each component. The author of this unique pattern has provided some valid questions at [http://autopilotpattern.io/](http://autopilotpattern.io/), and those questions come in handy while designing the autopilot pattern.

There are some applications emerging with at least some of this logic built in. Traefik is a proxy server with automatic discovery of its backends using Consul or other service catalogs. Traefik does not self-register in those service catalogs so that it can be used by other applications. ContainerPilot, a helper written in Golang that lives inside the container, can help with this.

ContainerPilot provides microservices architectures with application orchestration, dependency management, health checks, error handling, lifecycle management, and linear and non-linear scaling of stateful services. Furthermore, it provides a private init system designed to live inside the container. It acts as a process supervisor, reaps zombies, runs health checks, registers the app in the service catalog, watches the service catalog for changes, and runs your user-specified code at events in the life cycle of the container to make it all work correctly. ContainerPilot uses Consul to coordinate global state among the application containers.

Using a small configuration file, ContainerPilot can trigger events inside the container to automate operations on these events, including preStart (formerly onStart), health, onChange, preStop, and postStop.

Here is a sample scenario (readers can find the details at [http://autopilotpattern.io/example](http://autopilotpattern.io/example)). The author of this example has started with two services, sales, and customers. Nginx acts as a reverse proxy. Requests for `/customers/ go` to customer's, and `/sales/` to sales. The sales service needs to get some data from the customer's service to serve its requests, and vice versa. There are a few crucial problems here. The configuration is static. This prevents adding new instances and makes it harder to work with a failed instance. Configuring this stack via configuration management tools means packaging new dependencies with this application, but configuring statically means redeploying most of the containers every time a new instance gets added. There is a need for a mechanism to have the applications self-assemble and self-manage the everyday tasks, and hence there is a surging popularity for the autopilot pattern.

Engaging autopilot! The author of this autopilot design pattern has come out with the appropriate Dockerfile for the customer's service. It's a small Node.js application that listens on port `4000`. He uses Consul for service discovery and each service will send TTL heartbeats to Consul. All nodes know about all other nodes, and hence there is no need to use an external proxy or load balancer for communicating between the nodes. The diagram vividly illustrates everything.

However, there is a need to make each of the services aware of Consul. For that, the author uses ContainerPilot. The source code and other implementation details are given at [https://github.com/autopilotpattern/workshop](https://github.com/autopilotpattern/workshop).

A re-usable Nginx base image got implemented according to the autopilot pattern for automatic discovery and configuration. The goal is to create a Nginx image that can be reused across environments without having to rebuild the entire image. The configuration of Nginx is entirely through ContainerPilot jobs and watch handlers, which update the Nginx configuration on disk through consul-template. The relevant details are supplied at [https://github.com/autopilotpattern/nginx](https://github.com/autopilotpattern/nginx). Similarly, there are autopilot implementations for other popular applications such as WordPress. Bringing a bevy of automation into various microservices-based software applications is gaining a lot of momentum.

As indicated previously, a number of manual tasks are getting automated at different layers and levels, especially some of the crucial automation requirements are increasingly implemented at the application level. With the faster maturity and stability of the Docker platform, Docker containers are spreading their wings fast and wide. With the widespread availability of container management software solutions, microservices-based applications are gaining a lot of market and mind shares. Furthermore, there are a few service mesh frameworks, and hence the days of resilient microservices and reliable applications are not too far away. A growing bunch of automation capabilities is being attached to these applications, and this advancement enables the applications to exhibit adaptive behavior. Now, the autopilot pattern plays a key role in adding additional automation features and facilities.    

# Containers - persistent storage patterns

Typically, the container space originates with application containers that are not for permanently storing data. That is, when a container collapses, the data stored or buffered in the container gets lost. However, the aspect of data persistence is insisted for several reasons, including the realization of stateful applications, and hence fresh mechanisms are being worked out in order to safely and securely persist data in containers. Therefore, for persisting data, additional container types, such as data or volume containers were introduced. 

# The context for persistent storages

There is a concern widely expressed that containers are great for stateless applications, but are not so good for stateful applications that persist data. Thus, persistent storage patterns are acquiring special significance in the container world. A brief description about stateless and stateful applications is given as follows paragraph.

A random number generator is stateless because we get a different value from it every time we run it. We could easily Dockerize it and if the instance fails, we can have it running in another host instantaneously to continue the service without any break and lag. The instance's behavior remains the same in the new host as well. However, that is not the case with our bank accounts. If the bank account application has to be re-provisioned on a new server, it has to have the original data from the first server instance.

Here is a stateful data categorization. Typically, configuration data, including keys and other secrets, is often written to disk in various files. This data is easy to recover when provisioning instances. User-generated content includes text, video, or bank transactions. There are dynamic configuration details. The suitable example is of those services *A* and *B* connecting with one another. Connecting an application/service to its backend database system is another prominent example. Typically, applications/services treat this discovery and connectivity as configuration data along with other configuration details. In order for an application to be scalable and resilient, it is necessary to update this configuration information while the application is running. That is, as we add or remove instances of a service, we have to update all the other service instances that connect to the service. Otherwise, the intended performance increment would not happen. Other pertinent configuration details can include performance-tuning parameters. These configuration details can be stocked in the application repository in order to facilitate the application/service versioning and easier tracking. The other option for configuration information is leveraging the dynamic storage so they can be changed without re-building and re-deploying the application. It is also possible to do automatic replication of repository contents to the configuration store using a tool such as `git2consul`. The best practice is to keep configuration data and templates in a consistent distributed key/value data store.

# The persistent storage options

Containers are meant to be ephemeral, and so scale pretty well for stateless applications. Stateful containers, however, need to be treated differently. For stateful applications, a persistent storage mechanism has to be there for the container idea to be right and relevant. Containers can be developed and dismantled without the data persistence. The data resides within the container. If there is any change, then the data gets lost. For some situations, this data loss is not a big issue. For certain scenarios, the data loss is not accepted; the data persistence feature has to be there. The solution approach prescribed by Docker is given in the following section.

It is possible to store data within the writable layer of a container, but there are a few downsides:

*   The data won't persist when that container is no longer running, and it can be difficult to get the data out of the container if another process needs it.
*   A container's writable layer is tightly coupled to the host machine where the container is running. Moving the data somewhere else is a difficult affair.
*   Writing into a container's writable layer requires a storage driver to manage the filesystem. The storage driver provides a union filesystem, using the Linux kernel. This extra abstraction reduces performance as compared to using *data volumes*, which write directly to the host filesystem.

Docker offers three different ways to mount data into a container from the Docker host: *volumes*, *bind mounts*, or *tmpfs** mounts*. Volumes are almost always the right choice. Volumes are the preferred mechanism for persisting data generated by and used by Docker containers. While bind mounts are dependent on the directory structure of the host machine, volumes are completely managed by Docker. Volumes have several advantages over bind mounts:

*   Volumes are easier to back up or migrate than bind mounts
*   Volumes are easy to manage by using Docker CLI commands or the Docker API
*   Volumes work on both Linux and Windows containers
*   Volumes can be more safely shared among multiple containers
*   Volume drivers allow storing volumes on remote hosts or cloud providers, to encrypt the contents of volumes, or to add other functionality
*   A new volume's contents can be pre-populated by a container

Volumes are often a better choice than persisting data in a container's writable layer, because using a volume does not increase the size of containers using it, and the volume's contents exist outside the life cycle of a given container. 

If a container generates non-persistent state data, then consider using a tmpfs mount to avoid storing the data anywhere permanently, and to increase the container's performance by avoiding writing into the container's writable layer.

All the three options are discussed as follows:

*   **Volumes** are stored in a part of the host filesystem that is *managed by Docker* (`/var/lib/docker/volumes/` on Linux). Non-Docker processes cannot modify this part of the filesystem.
*   **Bind mounts** may be stored *anywhere* on the host system. They may even be important system files or directories. Non-Docker processes on the Docker host or a Docker container can modify them at any time.
*   **The tmpfs ****mounts** are stored in the host system's memory only and are never written to the host system's filesystem.

Let's discuss more about them.

# Volumes

We can create a volume explicitly using the `docker volume create` command, or Docker can create a volume during container or service creation. When we create a volume, it is stored in a directory on the Docker host. When we mount the volume into a container, this is the directory that is mounted on the container. This is similar to the way that bind mounts work, except that volumes are managed by Docker and are isolated from the core functionality of the host machine.

A given volume can be mounted into multiple containers simultaneously. When no running container is using a volume, the volume is still available to Docker and is not removed automatically. You can remove unused volumes using `docker volume prune`. Volumes also support the use of *volume drivers*, which allow the storing of data on remote hosts, cloud providers, and so on.

# Bind mounts

When we use a bind mount, a file or directory on the *host machine* is mounted on a container. The file or directory is referenced by its full path on the host machine. The file or directory does not need to exist on the Docker host already and it can be created on demand. Bind mounts are very performant, but they rely on the host machine's filesystem having a specific directory structure available. It is not possible to use Docker CLI commands to directly manage bind mounts.

# The tmpfs mounts

A tmpfs mount is not persisted on disk either on the Docker host or within a container. It can be used by a container during the lifetime of the container, to store non-persistent state or sensitive information. For instance, internally, swarm services use tmpfs mounts to mount secrets into a service's containers.

# Docker compose configuration pattern

We are increasingly hearing, reading, and even experiencing multi-container applications. That is, composite applications are being achieved through multi-container composition. The composition technique acquires special significance because of two key trends. Firstly, the powerful concept of microservices is gradually changing the IT industry. That is, large monolithic services are slowly giving way to swarms of small and autonomous microservices. Different and distributed microservices are being found, checked and chained together to create and run business-class, production-ready, process-aware, mission-critical, enterprise-grade, composite applications. The second is that the Docker-enabled containerization changes not only the architecture of services but also the structure of environments used to create them. Now, software gets methodically containerized, stocked, and distributed and developers gain the full freedom to choose the preferred applications. Resultantly, even complex environments such as **continuous integration** (**CI**) servers with database backend systems and analytical infrastructure can be instantiated within seconds. In short, software development, deployment, and delivery become easier and faster.

Docker Compose is a tool for defining and running complex applications with Docker. With Compose, it is possible to define a multi-container application in a single file, and then spin the application up in a single command that does everything that needs to be done to get it running. Using Compose is basically a three-step process:

1.  Define your application's environment with a Dockerfile so it can be reproduced anywhere
2.  Define the services that make up the application in `docker-compose.yml` so they can be run together in an isolated environment
3.  Lastly, run `docker-compose up` and compose will start and run the entire application

We can pass in environment variables via Docker Compose in order to realize a container image once and reuse it on any environment (development, staging, and production). With this approach, it is possible to develop compose-centric containers that require a piece of configuration management for handling pre-start events based on the values of the environment variables. The author of this pattern has detailed the source code at [https://github.com/jay-johnson/docker-schema-prototyping-with-mysql](https://github.com/jay-johnson/docker-schema-prototyping-with-mysql).

The author has built this project for rapid-prototyping a database schema using a MySQL Docker container that deploys its own ORM schema file and populates the initial records on startup. By setting a couple of environment variables, it is possible to provide our own Docker container with a usable MySQL instance, browser-ready phpMyAdmin server, and our database, including the tables, initialized exactly how we want. Interested readers are requested to visit the preceding page to get finer details on this unique pattern.

# Docker container anti-patterns

We have discussed most of the available container-specific patterns in the previous section. Many exponents and evangelists of Docker-enabled containerization have brought in a few anti-patterns based on their vast experience in developing, deploying, and delivering containerized services and applications. This section is exclusively allocated for conveying the anti-patterns discovered and disseminated by Docker practitioners.

Container creation and deployment are becoming easier and faster with the ready availability of both open-source and commercial-grade tools. DevOps team members ought to learn some of the techniques and tips in order to avoid mistakes when migrating to Docker.

# Installing an OS inside a Docker container 

There is rarely a good reason to host an entire OS inside a container using Docker. There are platforms for generating and running system containers. The Docker platform is specially crafted and fine-tuned for producing application containers. That is, applications and their runtime dependencies are being stuffed together, packaged, and transmitted to their destinations. 

# Go for optimized Docker images

When building container images, we should include only the services that are absolutely essential for the application the container will host. Anything extra wastes resources and widens the potential attack vector that could ultimately lead to security problems. For example, it is not good to run an SSH server inside the container because we can use the Docker *exec *call to interact with the containerized application. The related suggestions here are to create a new directory and include the Dockerfile and other relevant files in that directory. Also consider using `.dockerignore` to remove any logs, source code, and so on before creating the image. Furthermore, make it a habit to remove any downloaded artifacts after they are unzipped.

It is not correct to use different images or even different tags in development, testing, staging, and production environments. The image that is the *source of truth* should be created once and pushed to a repository. That image should be used for different environments going forward. Any system integration testing should be done on the image that will be pushed into production.

The containers produced by the Docker image should be as ephemeral as possible. By *ephemeral*, it is meant that it can be stopped and destroyed and a new one can be built and put in place with an absolute minimum of setup and configuration.

The best practice is to not keep critical data inside containers. There are two prime reasons for this. When containers collapse inadvertently or deliberately, the data inside them gets lost immediately. The second reason is that the security situation of containers is not as good as virtual machines, and hence storing confidential, critical, customer, and corporate information, inside containers is not a way forward. For persisting data, there are mechanisms to be used. The popular ELK stack could be used to store and process logs. If managed volumes are used during the early testing process, then it is recommended to remove them using the `-v` switch with the `docker rm` command.

Also, do not store any security credentials in the Dockerfile. They are in clear text and this makes them completely vulnerable. Do not forget to use `-e` to specify passwords as runtime environment variables. Alternatively, `--env-file` can be used to read environment variables from a file. Also, go for CMD or ENTRYPOINT to specify a script, and this script will pull the credentials from a third party and then configure the application.

# Storing container images only inside a container registry

A container registry is designed solely for the purpose of hosting container images. It is not good to use the registry as a general-purpose repository for hosting other types of data.

# Hosting only one service inside a container

In the microservices world, applications are being partitioned into a dynamic collection of interactive, single-purpose, autonomous, API-driven, easily manageable, and composable services*.* Containers emerge as the best-in-class runtime environment for microservices. Thus, it is logical to have one service inside a container. Thus, for running an application, multiple containers need to be leveraged for running many services. For example, one container would install and use MySQL, WordPress, possibly even phpMyAdmin, nginx, and an SSH daemon. Also, multiple instances of a service can be hosted in different containers. The redundancy capability being achieved through containers goes a long way in ensuring the business continuity through fault-tolerance, high availability, horizontal scalability, independent deployment, and so on. Now, with the emergence of powerful container orchestration platforms, distributed and multiple containers can be linked up to come out with composite applications. An advantage of containerization is the ability to quickly re-build images in the case of a security issue, for example, and roll out a whole new set of containers quickly. And because containers are single-concern, there is no need to redeploy the cloud infrastructure every time. Similarly, multiple Docker images can be built from a base image. Furthermore, containers can be also converted to new images.

We can use the CMD and ENTRYPOINT commands while formulating a Dockerfile. Often, CMD will use a script that will perform some configurations of the image and then start the container. It is better to avoid starting multiple processes using that script. This will make managing containers, collecting logs, and updating each individual process hard. That is, we need to follow the *separation of concerns* pattern when creating Docker images. Breaking up an application into multiple containers and managing them separately is the way forward.

# Latest doesn't mean best

It is incredibly tempting when writing a Dockerfile to grab the latest version of every dependency. The *golden rule *though is to create containers with known and stable versions of the system and dependencies that we know our software will work on.

# Docker containers with SSH

A related and equally unfortunate practice is to bake an SSH daemon into an image. Having an SSH daemon inside a container may lead to undocumented, untraceable changes to the container infrastructure, but Docker containers are being touted as the immutable infrastructure.

There are a few use cases for SSHing into a container:

*   Update the OS, services, or dependencies
*   Git pull or update any application in some other fashion
*   Check logs
*   Backup some files
*   Restart a service

Instead of using SSH, it is recommended to use the following mechanisms:

*   Make the change in the container Dockerfile, rebuild the image, and deploy the container.
*   Use an environment variable or configuration file accessible via volume sharing to make the change and possibly restart the container.
*   As indicated before, use `docker exec`. The `docker exec` command starts a new command in a running container, and hence has to be the last resort.

# IP addresses of a container

Each container get assigned with an IP address. In a containerized environment, multiple containers have to interact with one another in order to achieve business goals. Also, containers are terminated often and fresh containers are being created. Thus, relying upon IP addresses of containers for initiating container communication is beset with real challenges. The preferred approach is to create services. This will provide a logical name that can be referred to independent of the growing and shrinking number of containers. And it also provides a basic load balancing.

Also, do not use `-p` to publish all the exposed ports. This facilitates in running multiple containers and publishing their exposed ports. But this comes with a price. That is, all the ports will be published, resulting in a security risk. Instead, use `-p` to publish specific ports.

# Root user

This is a security-mitigation tip. Don't run containers as a root user. The host and the container share the same kernel. If the container is compromised, a root user can do more damage to the underlying hosts. Instead, create a group and a user in it. Use the user instruction to switch to that user. Each user creates a new layer in the image. Also, avoid switching the user back and forth to reduce the number of layers.

# Dependency between containers

Often, applications rely upon containers to be started in a certain order. For example, a database container must be up before an application can connect to it. The application should be resilient to such changes as the containers may be terminated or started at any time. In this case, have the application container wait for the database connection to succeed before proceeding further. Do not use *wait-for* scripts in a Dockerfile for the containers to start up in a specific order.

In conclusion, containers are the new and powerful unit of development, deployment, and execution. Business applications, IT platforms, databases, and middleware are formally containerized and stocked in publically available and accessible image repositories so that software developers can pick up and leverage them for their software-building requirements. The system portability is a key advantage. The easier and faster maneuverability, testability, and composability of container images are being touted as the most promising and potential advantages of containerization. The inevitability of distributed computing is greatly simplified by the concept of containerization. Multiple containers across clusters can be easily linked to realizing smart and sophisticated applications.

# Patterns for highly reliable applications

The IT systems are indispensable for business automation. The widely articulated challenge for our IT and business applications is to showcase high reliability. Systems ought to be responsive, resilient, elastic, and secure in order to intrinsically demonstrate the required dependability. Systems are increasingly multimodal and multimedia. Systems have to capture, understand, and exhibit the appropriate behavior. Also, systems have to respond all the time under any circumstance. Also, with the dawn of big data, distributed computing is all set to the mainstream compute model. In this section, we will discuss the prominent patterns for constructing reliable systems for professional as well as personal requirements. The promising approaches include:

*   Reactive and cognitive programming techniques
*   Resilient microservices
*   Containerized cloud environments

In a distributed system, failures are bound to happen because of multiple moving parts and the sickening dependencies between the participating systems' modules. Hardware can fail, the application may go down, and the network can have transient failures. Rarely, an entire service or region may experience a disruption. Clouds are emerging as the one-stop IT solution.

**Resiliency** is the ability of a system to withstand and tolerate faults in order to function continuously. Even if it fails, it has the wherewithal to bounce back to the original state. Precisely speaking, it is all about not avoiding failures but how quickly it can recover from the failures to serve without any breakdown and slowdown. Also, a fault in a component of a system should not cascade into other components in order to bring down the whole system. There are resiliency strategies, patterns, best practices, approaches, and techniques. 

**High availability **(**HA**) is the ability of the application to continue running in a healthy state, without significant downtime. That is, the application continues to be responsive to users' requests.

**Disaster recovery **(**DR**) is the ability to recover from rare but major incidents: non-transient, wide-scale failures, such as service disruption that affects an entire region. Disaster recovery includes data backup and archiving, and may include manual intervention, such as restoring a database from backup.

Resiliency must be designed into the system, and here is a general model to follow:

1.  **Define** the application availability requirements based on business needs.
2.  **Design** the application architecture for resiliency. Start with an architecture that follows proven practices and architectural decisions, and then identify the possible failure points in that architecture. Take care of the dependencies. Also, choose the best-in-class architectural patterns and styles that intrinsically support resiliency.
3.  **Develop** the application using the appropriate design patterns and incorporate strategies to detect and recover from failures.
4.  **Build and test** the implementation by simulating faults and triggering forced failovers and debug the identified issues to the fullest.
5.  **Decide** the infrastructure capacity accordingly and **provision** them.
6.  **Deploy** the application into production using a reliable and repeatable process.
7.  **Monitor** the application to detect failures. The monitoring activity helps to gauge the health of the application. The health check comes in handy in providing instantaneous responses.  
8.  **Respond** if there are incidents that require any kind of manual interventions.

# Resiliency implementation strategies

As the resiliency requirement is insisted, IT departments of various business enterprises are exploring various ways and means in order to build and release resilient application services. At different levels (infrastructure, platform, database, middleware, network, and application), the virtue of resiliency is being mandated so that the whole system and environment become resilient.

In this section, we will dig deeper and describe how the elusive target of resiliency is being endeavored and enunciated to see the reality. There are a few noteworthy failures. The key ones are include as follows:

*   **Retry transient failures**: Transient failures can occur due to many causes, deficiencies, and disturbances. Often, a temporary failure can be resolved simply by retrying the request. However, each retry adds to the total latency. Also, too many failed requests can cause a bottleneck as pending requests accumulate in the queue. These blocked requests might hold critical system resources such as memory, threads, database connections, and so on. A sellable workaround here is to increase the delay between each retry attempt and limit the total number of failed requests.
*   **Load balance across instances**: This is a common thing happening in IT environments. A **load balancer** (**LB**) instance in front of an application facilitates adding more application instances in order to improve resiliency.
*   **Replicating data**: It has been a standard approach for handling non-transient failures in a database and filesystem. The data storage technologies innately provide built-in replication. However, to fulfill the high-availability requirement, replicas are being made and put up in geographically distributed locations. So, if one region goes down, the other region can take care of the business continuity. However, this significantly increases the latency when replicating the data across the regions. Typically, considering the long distance between regions, the data replication happens in an asynchronous fashion. In this case, we can not expect real-time and strong consistency. Instead, we need to settle for eventual consistency. Corporates have to tolerate for potential data loss if a replica fails.
*   **Degrade gracefully**: If a service fails and there is no failover path, the application may be able to degrade gracefully while still providing an acceptable user experience.
*   **Throttle high-volume users**: Sometimes, a small number of users create excessive load. This can have a bad impact on other users. The application might throttle the high-volume users for a certain period of time. Throttling does not imply the users are acting maliciously. The throttling starts if the number of requests exceeds the threshold.

# The testing approaches for resiliency

Testers have to test how the end-to-end workload performs under failure conditions that only occur intermittently, there are two types as follows:

*   **Fault injection testing**: This is one way of testing the resiliency of the system during failures, either by triggering actual failures or by simulating them.
*   **Load testing**: There are open source as well as commercial-grade load generation tools, and through those tools load testing of the application is being insisted. Load testing is crucial for identifying failures that only happen under loads such as the backend database being overwhelmed or service throttling. Test for peak load, using production data or synthetic data that is as close to production data as possible. The goal is to see how the application behaves under real-world conditions.

# The resilient deployment approaches

Software deployment is an important facet for establishing and sustaining resiliency. After applications are deployed in production-grade servers, the software updates also can be a source for errors. Any incomplete and bad update results in system breakdown. There are a few proven deployment and update methods in order to avoid any kind of downtime. The proper checks have to be in place before deployment and subsequent updates. Deployment typically includes provisioning of various server, network, and storage resources, deploying the curated and refined application code, and applying the required and right configuration settings. An update may involve all three or a subset of the three tasks. It is therefore recommended to have a tool-assisted, automated, and idempotent process in place. There are two major concepts related to resilient deployment:

*   **Infrastructure as code** is the practice of using code to provision and configure infrastructure. Infrastructure as code may use a declarative approach or an imperative approach, or a combination of both.

*   **Immutable infrastructure** complies with the principle that the infrastructure should not be disturbed or modified after it has gone to production.

# The deployment patterns

**Blue-green deployment** is a technique where an update is deployed into a production environment separate from the live application. After the deployment gets validated, then switch the traffic routing to the updated version.

In the case of **canary releases**, instead of switching all traffic to the updated version, we can roll out the update to a small percentage of users, by routing a portion of the traffic to the new deployment. If there is a problem, back off and revert to the old deployment. Otherwise, route more of the traffic to the new version, until it gets 100% of the traffic.

Whatever approach is preferred, it is mandatory to make sure that we can roll back to the last-known good deployment, in case the new version is not functioning as per the expectation. Also, if errors occur, the application logs must indicate which version caused the error.

# Monitoring and diagnostics

Continuous and tools-assisted monitoring of applications is crucial for achieving resiliency. If something drags, lags, or fails, the operational team has to be informed immediately along with all the right and relevant details to consider and proceed with a correct course of action. As we all agree, monitoring a large-scale distributed system poses a greater challenge. With the overwhelming acceptance of the *divide and conquer* technique, the number of moving parts of any enterprise-scale application has grown steadily and sharply. Today, as a part of the compartmentalization, we have virtualization and containerization concepts widely accepted and adopted. The number of VMs in any IT environment is growing. Furthermore, due to the lightweight nature, the number of containers being leveraged to run any mission-critical application has escalated rapidly and remarkably. In short, monitoring bare metal servers, VMs, and containers precisely is definitely a challenge for operational teams. Also, every kind of software and hardware generates a lot of log files resulting in massive operational data. It has become common to subject all sorts of operational data to extract actionable insights. Not only are the IT systems distributed, but they are also extremely dynamic. The monitoring, measuring, and management complexities of tomorrow's data centers and server farms are consistently on the climb. 

Monitoring is not the same as failure detection. For example, our application might detect a transient error and retry, resulting in no downtime. But it should also log the retry operation so that we can monitor the error rate, in order to get an overall picture of application health.

The resiliency strategy is essential to ensure the service resiliency of IT systems and business applications. As enterprises increasingly embrace the cloud model, the cloud service providers are focusing on enhancing the resiliency capability of their cloud servers, storage, and networks. Application developers are also learning the tricks and techniques fast in order to bring forth resilient applications. With the combination of resilient infrastructures, platforms, and applications, the days of the resilient IT, which is mandatory towards agile, dynamic, productive, and adaptive businesses, is not too far away.  

# Resiliency realization patterns

Patterns are always a popular and peerless mechanism for unearthing and articulating competent solutions for a variety of recurring problems. We will look at a host of promising and proven design patterns for accomplishing the most important goal of resiliency.

# Circuit breaker pattern

The circuit breaker pattern can prevent an application from repeatedly trying an operation that is likely to fail. The circuit breaker wraps the calls to a service. It can handle faults that might take a variable amount of time to recover from when connecting to a remote service or resource. This can improve the stability and resiliency of an application.

**The problem description**—Remote connectivity is common in a distributed application. Due to a host of transient faults such as slow network speed, timeouts, the service unavailability, or the huge load on the service, calls to remote application services can fail. These faults, being transient, typically correct themselves after a short period of time. The retry pattern strategy suggests that a robust cloud application can handle these transient faults easily in order to meet up the service requests.

However, there can also be situations wherein the faults are due to bigger issues. The severity levels vary from temporary connectivity loss to the complete failure of the service due to various reasons and causes. Here, it is illogical to continuously retry to establish the broken connectivity. Instead, the application has to understand and accept the situation to handle the failure in a graceful manner.

Suppose the requested service is very busy, then there is a possibility for the whole system to break down.

Generally, an operation that invokes a service is configured to implement a timeout and to reply with a failure message if the service fails to respond within the indicated time period. However, this strategy could cause many concurrent requests to the same operation to be blocked until the timeout period expires. These blocked requests might hold critical system resources such as memory, threads, database connections, and so on. Finally, the resources could become exhausted, causing failure of other associated and even unrelated system components. The idea is to facilitate the operation to fail immediately and only to attempt to invoke the service again if it is likely to succeed. The point here is to set up a timeout intelligently because a shorter timeout might help to resolve this problem but the shorter timeout may cause the operation to fail most of the time.

**The solution** **approach**—The solution is the proven circuit breaker pattern, which can prevent an application from repeatedly trying to execute an operation that's likely to fail. This allows it to continue without waiting for the fault to be fixed or wasting CPU cycles while it determines that the fault is long lasting. The circuit breaker pattern also enables an application to detect whether the fault has been resolved. If the problem appears to have been fixed, the application can try to invoke the operation.

The retry pattern enables an application to retry an operation in the expectation that it will succeed. On the other hand, the circuit breaker pattern prevents an application from performing an operation that is likely to fail. An application can combine these two patterns by using the retry pattern to invoke an operation through a circuit breaker. However, the retry logic should be highly sensitive to any exceptions returned by the circuit breaker and abandon retry attempts if the circuit breaker indicates that a fault is not transient. Also, a circuit breaker acts as a proxy for operations that might fail. The proxy should monitor the number of recent failures that have occurred, and use this information to decide whether to allow the operation to proceed, or simply return an exception immediately. The proxy can be implemented as a state machine with the following states:

*   **Closed**: This is the original state of the circuit breaker. Therefore, the circuit breaker sends requests to the service and a counter continuously tracks the number of recent failures. If the failure count goes above the threshold level within a given time period, then the circuit breaker switches to the *open* state.
*   **Open**: In this state, the circuit breaker opens up and immediately fails all requests without calling the service. The application instead has to make use of a mitigation path such as reading data from a replica database or simply returning an error to the user. When the circuit breaker switches to the open state, it starts a timer. When the timer expires, the circuit breaker switches to the half-open state.
*   **Half-open**: In this state, the circuit breaker lets a limited number of requests go through to the service. If they succeed, the service is assumed to be recovered and the circuit breaker switches back to the original closed state. Otherwise, it reverts to the open state. The half-open state prevents a recovering service from suddenly being inundated with a series of service requests.

The circuit breaker pattern ensures the system's stability while the system slowly yet steadily recovers from a failure and minimizes the impact on the system's performance. It can help to maintain the response time of the system by quickly rejecting a request for an operation that is likely to fail rather than waiting for the operation to time out. If the circuit breaker raises an event each time, it changes the state. This information can be used to monitor the health of the part of the system protected by the circuit breaker or to alert an administrator when a circuit breaker trips to the open state.

The pattern is highly customizable and can be adapted according to the type of the possible failure. For example, it is possible to use an increasing timeout timer to a circuit breaker. We can place the circuit breaker in the open state for a few seconds initially and if the failure hasn't yet been resolved, then increase the timeout to a few minutes, and so on. In some cases, rather than the open state returning a failure and raising an exception, it could be useful to return a default value that is meaningful to the application.

In summary, this pattern is used to prevent an application from trying to invoke a remote service or access a shared resource if this operation is highly likely to fail. This pattern is not:

*   For handling access to local private resources in an application, such as an in-memory data structure
*   As a substitute for handling exceptions in the business logic of your applications

The circuit breaker pattern is becoming very common with microservices, emerging as the most optimized way of partitioning massive applications and presenting applications as an organized collection of microservices.

# Bulkhead pattern

**The problem description**—cloud applications typically comprise multiple and inter-linked services. A service can run on different and distributed services as service instances. There can be multiple requests from multiple consumers for each of those service instances. When the consumer sends a request to a service that is misconfigured or not responding, the resources used by the client's request may not be freed in a timely manner.

As requests to the service continue incessantly, those resources may soon be exhausted. The resources occupied include the database connection. The ultimate result is that any request to other services of the cloud application gets impacted. Eventually, the cloud application may not be available to the consumer. This is the case with other consumers too. In short, a large number of requests originating from one client may exhaust available resources in the service. This is the cascading effect and this pattern comes in handy in surmounting this issue.

**The solution approach**—the solution is to smartly partition service instances into different groups, based on consumer load and availability requirements. This design helps to isolate failures, and allows sustaining service functionality for some consumers, even during a failure. A consumer can also partition resources, to ensure that resources used to call one service don't affect the resources used to call another service. For example, choosing different connection pools for different services is a workable option. Thus, the collapse of one connection pool does not stop other connections.

The benefits of this pattern include the following:

*   This isolates service consumers and services from cascading failures. This isolation firmly prevents an entire solution from going down.
*   The instance-level isolation helps to retain the other instances of the services. Thus, the service availability is guaranteed and similarly, other services of the application continue to deliver their assigned functionality.
*   This helps to identify the demands of consuming applications and accordingly allows deploying services that offer a different **Quality of Service** (**QoS**). That is, a high-priority consumer pool can be configured to use high-priority services.

In summary, any sort of failures in one subsystem can sometimes cascade to other components resulting in the system breakdown. To avoid this, we need to partition a system into a few isolated groups, so that any failure in one partition does not percolate to others. Containerization in conjunction with polyglot microservices is an overwhelming option for having partitioned and problem-free systems.

# Compensating transaction pattern

This is a transaction that undoes the effects of another completed transaction. In a distributed system, it can be very difficult to achieve strong transactional consistency. Compensating transactions are a way to achieve consistency by using a series of smaller and individual transactions that can be undone at each step.

**The problem description**—a typical business operation consists of a series of separate steps. While these steps are being performed, the overall view of the system state might be inconsistent, but when the operation has completed and all of the steps have been executed, the system should become consistent again. A challenge in the eventual consistency model is how to handle a step that has failed. In this case, it might be necessary to undo all of the work completed by the previous steps in the operation. However, the data can't simply be rolled back because other concurrent instances of the application might have changed it. Even in cases where the data hasn't been changed by a concurrent instance, undoing a step might not simply be a matter of restoring the original state. This mandates the application of various business-specific rules. If an operation that implements eventual consistency spans several heterogeneous data stores, undoing the steps in the operation will require visiting each data store in turn. The work performed in every data store must be undone reliably to prevent the system from remaining inconsistent.

In a **service-oriented architecture** (**SOA**) environment, an operation could invoke an action in a service and cause a change in the state held by that service. To undo the operation, this state change must also be undone. This can involve invoking the service again and perform another action that reverses the effects of the first.

**The solution approach**—the solution is to implement a compensating transaction. The steps in a compensating transaction must undo the effects of the steps in the original operation. A compensating transaction might not be able to simply replace the current state with the state the system was in at the start of the operation because this approach could overwrite changes made by other concurrent instances of an application. Instead, it must be an intelligent process that takes into account any work done by concurrent instances. This process will usually be application specific, driven by the nature of the work performed by the original operation.

A common approach is to use a workflow to implement an eventually consistent operation that requires compensation. As the original operation proceeds, the system records information about each step and how the work performed by that step can be undone. If the operation fails at any point, the workflow rewinds back through the steps it has completed and performs the work that reverses each step.

It is recommended to use this pattern only for operations that must be undone if they fail. If possible, design solutions to avoid the complexity of requiring compensating transactions.

# Health endpoint monitoring pattern

**The problem description**—applications and their services need to be continuously monitored to gain a firm grip on their availability and performance levels and patterns. Monitoring services running in off-premises, on-demand, and online environments are quite difficult compared to any on-premises services. There are many factors that affect cloud-hosted applications, such as network latency, the performance and availability of the underlying compute and storage systems, and the network bandwidth between them. The service can fail entirely or partially due to any of these factors. Therefore, we must verify at regular intervals that the service is performing correctly.

**The solution approach**—we need to do health monitoring by sending requests to an endpoint on the application. The application should perform the necessary checks and return an indication of its status. A health-monitoring check typically combines two factors:

*   The assigned checks performed by the application or service in response to the request to the health verification endpoint
*   The analysis of the results by the health-monitoring tool that performs the health verification check.

There are several parameters and conditions being checked by a health-monitoring tool in order to completely and concisely understand the state of the application.

It is also useful to run these checks from different on-premises or hosted locations to measure and compare response times. As customers are geographically distributed, the checks have to be initiated and implemented from those locations that are close to customers.

Another point is to expose at least one endpoint for the core services that the application uses and another for lower priority services. This allows different levels of importance to be assigned to each monitoring result. Also, it is good to consider exposing more endpoints such as one for each core service for additional monitoring granularity. Increasingly, health-verification checks are being done on the database, storage, and other critical services. The uptime and response time decide the quality of applications.

This pattern is extremely useful for checking the health condition of websites, web and mobile applications, and cloud-hosted applications.

# Leader election pattern

**The problem description**—a typical cloud application has many tasks working in a coordinated manner. These tasks could all be instances running the same code and requiring access to the same resources, or they might be working together in parallel to perform the individual parts of a complex calculation. The task instances might run separately for much of the time, but it might also be necessary to coordinate the actions of each instance to ensure that they don't conflict, cause contention for shared resources, or accidentally interfere with the work that other task instances are performing.

For example, cloud systems guarantee scalability through scale-up or scale-out. In the case of scale-out (horizontal scaling), there can be multiple instances of the same task/service. Each instance serves different users. If these instances write to a shared resource, then it is necessary to coordinate their actions to prevent each instance from overwriting the changes made by the others. Similarly, if the tasks are performing individual elements of a complex calculation in parallel, the results need to be duly aggregated to give the final answer. The task instances are all peers, so there isn't a natural leader that can act as the coordinator or aggregator.

**The solution approach**—a single task instance should be elected to act as the leader, and this instance should coordinate the actions of the other subordinate task instances. If all of the task instances are running the same code, they are each capable of acting as the leader. Therefore, the election process must be managed carefully to prevent two or more instances taking over the leader role at the same time. The system must provide a robust mechanism for selecting the leader. This method has to cope with events such as network outages or process failures. In many solutions, the subordinate task instances monitor the leader through some type of heartbeat method or by polling. If the designated leader terminates unexpectedly, or a network failure makes the leader unavailable to the subordinate task instances, it's necessary for them to elect a new leader. This is like choosing a cluster head in a sensor mesh.

This pattern performs best when the tasks in a distributed application, such as a cloud-hosted solution, need careful coordination and there is no natural leader. It is prudent to avoid making the leader a bottleneck in the system. The purpose of the leader is to coordinate the work of the subordinate tasks, and it doesn't necessarily have to participate in this work itself—although it should be able to do so if the task isn't elected as the leader.

# Queue-based load leveling pattern

Applications may experience sudden spikes in traffic, which can bombard backend systems. If a backend service cannot respond to requests quickly enough, it may cause requests to queue (back up), or it can cause the service to throttle the application. To avoid this, we can use a queue as a buffer. When there is a new work item, instead of calling the backend service immediately, the application queues a work item to run asynchronously. The queue acts as a buffer that smooths out peaks in the load.

**The problem description**—for arriving at competent and composite applications that are business-centric and process-aware, cloud applications ought to interact with one another. The services can be locally available or accessible remotely. Various enthusiastic software developers bring modern applications and provide them for worldwide subscribers for a small fee, or sometimes for free. Similarly, there are **independent software vendors** (**ISVs**) contracting with hosted service providers to run their software to be found and bound. That is, various cloud services have to connect and collaborate with many others in order to be right and relevant to their consumers. In this intertwined environment, if a service is subjected to intermittent heavy loads, it can potentially cause performance or reliability issues. The predictability of the number of service users at a particular time is also a tough affair. Thus, static capacity planning is out of the discussion. *Dynamism* is the new buzzword in the IT landscape.

As indicated previously, an application can be segmented into multiple services. Each service can be run in different containers as separate instances. That is, multiple instances of a service can be run in an IT environment. In the service world, everything is API-enabled in order to be found and leveraged by other services. A service can be used by many tasks concurrently. A service could be part of the same application as the tasks that use it or it could be provided by a third-party service provider. For example, the service can be a resource service, such as a cache or a storage service.

A service might experience peaks in demand that cause it to overload and be unable to respond to requests in a timely manner. Flooding a service with a large number of concurrent requests can also result in the service failing if it's unable to handle the contention these requests cause.

**The solution approach**—it is suggested to refactor the solution and introduce a queue between the task and the service. The task and the service run asynchronously. The task posts a message containing the data required by the service to a queue. The queue acts as a buffer, storing the message until it is retrieved by the service. The service retrieves the messages from the queue and processes them. Requests from a number of tasks, which can be generated at a highly variable rate, can be passed to the service through the same message queue.

This pattern provides the following benefits:

*   It can help to maximize availability of applications because delays arising in services will not have an immediate and direct impact on the application, which can continue to post messages to the queue even when the service is not available or is not currently processing messages
*   It can help to maximize scalability because both the number of queues and the number of services can be varied to meet demand
*   It can help to control costs because the number of service instances deployed only has to be adequate to meet the average load rather than the peak load

# Retry pattern

**Problem description**—we have discussed a bit about this pattern previously. Applications are distributed in the sense that the application components are being expressed and exposed as a service and delivered from different IT environments (private, public, and edge clouds). Typically, the IT spans across embedded, enterprise, and cloud domains. With the fast-growing device ecosystem, the connectivity has grown to various devices at the ground level. That is the reason that we very often hear, read, and even experience **cyber-physical system** (**CPS**). Also, the enterprise-scale applications (both legacy and modern) are accordingly modernized and moved to cloud environments to reap the distinct benefits of the cloud idea. However, certain applications, due to some specific reasons, are being kept in enterprise servers/private clouds. With embedded and networked devices joining in the mainstream computing, edge/fog devices are being enabled to form kind of ad hoc clouds to facilitate real-time data capture, storage, processing, and decision-making. The point to be noted here is that application services ought to connect to other services in the vicinity and remotely hold services over different networks. Faults can occur, stampeding the application calls. As articulated previously, there are temporary faults impacting the service connectivity, interaction, and execution. However, these faults are typically self-correcting and if the action that triggered a fault is repeated after a suitable delay, the connectivity and accessibility may go through.

**The solution approach**—in cloud environments, transient faults are common and an application should be designed to handle them elegantly and transparently. This minimizes the effects faults can have on the business tasks the application is duly performing. If an application detects a failure when it tries to send a request to a remote service, it can handle the failure using the following strategies:

*   **Cancellation**: If the fault indicates that the failure is not temporary (that is, persists for more time), or is likely to be unsuccessful if repeated, the application should cancel the operation and report an exception.
*   **Retry**: If the specific fault reported is unusual or rare, it might have been caused by some unusual circumstances such as a network packet getting corrupted while it was being transmitted. In this case, the application can try again as the subsequent request may attain the required success.
*   **Retry after delay**: If the fault is caused by one of the more commonplace connectivity or busy failures, then the application has to wait for some time and try again.

The application should wrap all attempts to access a remote service in code that implements a retry policy matching one of the strategies listed previously. Requests sent to different services can be subjected to different policies. Some vendors provide libraries that implement retry policies, where the application can specify the maximum number of retries, the time between retry attempts, and other parameters. An application should log the details of faults and failing operations. This information is useful to operators. If a service is frequently unavailable or busy, it's often because the service has exhausted its resources. We can reduce the frequency of these faults by scaling out the service. For example, if a database service is continually overloaded, it might be beneficial to partition the database and spread the load across multiple servers.

In conclusion, having understood the strategic significance that the resiliency, robustness, and reliability of next-generation IT systems are to fulfil the various business and people needs with all the QoS and **Quality of Experience** (**QoE**) traits and tenets enshrined and etched, IT industry professionals, academic professors, and researchers are investing their talents, treasures, and time to unearth scores of easy-to-understand and useful techniques, tips, and tricks to simplify and streamline software and infrastructure engineering tasks. I ask the readers to visit [https://docs.microsoft.com/en-us/azure/architecture/patterns/category/resiliency](https://docs.microsoft.com/en-us/azure/architecture/patterns/category/resiliency) for further reading.

# Summary

Both legacy and modern applications are remedied to be a collection of interactive microservices. Microservices can be hosted and run inside containers. There can be multiple instances for each microservice. Each container can run a microservice instance. Thus, in a typical IT environment, there can be hundreds of physical machines (also called **bare metal servers**). Each physical machine, in turn, is capable of running hundreds of containers. Thus, there will be tens of thousands of containers. The management and operational complexities are therefore bound to escalate. This pattern comes handy in successfully running microservice-hosted containers. There are technologies, such as Istio and Linkerd, for ensuring the resiliency of microservices. This resiliency ultimately ensures the application's reliability. Together with software-defined cloud infrastructures, reliable applications ensure the reliability of cloud environments for hosting and delivering next-generation business workloads. 

The forthcoming chapters will discuss the various software-defined cloud application design and deployment patterns.