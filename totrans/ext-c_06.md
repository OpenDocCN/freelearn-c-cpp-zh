# Chapter 06

# OOP and Encapsulation

There are many great books and articles on the subject of object-oriented programming or OOP. But I don't think that many of them address the same topic using a non-OOP language such as C! How is that even possible? Are we even able to write object-oriented programs with a programming language that has no support for it? To be precise, is it possible to write an object-oriented program using C?

The short answer to the above question is yes, but before explaining how, we need to explain why. We need to break the question down and see what OOP really means. Why is it possible to write an object-oriented program using a language that has no claim for object-orientation support? This seems like a paradox, but it's not, and our effort in this chapter is to explain why that's possible and how it should be done.

Another question that may puzzle you is that what's the point of having such discussions and knowing about OOP when you are going to use C as your primary programming language? Almost all existing mature C code bases such as open source kernels, implementation of services like HTTPD, Postfix, nfsd, ftpd, and many other C libraries such as OpenSSL and OpenCV, are all written in an object-oriented fashion. This doesn't mean that C is object-oriented; instead, the approach these projects have taken to organize their internal structure comes from an object-oriented mindset.

I highly recommend reading this chapter together with the next three chapters and getting to know more about OOP because firstly, it will enable you to think and design like the engineers who have designed the libraries mentioned before, and secondly, it would be highly beneficial when reading the sources of such libraries.

C does not support object-oriented concepts such as classes, inheritance, and virtual functions in its syntax. However, it does support object-oriented concepts – in an indirect way. In fact, nearly all the computer languages through history have supported OOP intrinsically – way before the days of Smalltalk, C++, and Java. That's because there must be a way in every general-purpose programming language to extend its data types and it is the first step towards OOP.

C cannot and *should not* support object-oriented features in its syntax; not because of its age, but because of very good reasons that we're going to talk about in this chapter. Simply put, you can still write an object-oriented program using C, but it takes a bit of extra effort to get around the complexity.

There are a few books and articles regarding OOP in C, and they usually try to create a *type system* for writing classes, implementing inheritance, polymorphism, and more, using C. These books look at adding OOP support as a set of functions, macros, and a preprocessor, all of which can be used together to write object-oriented programs with C. This won't be the approach we take in this chapter. We are not going to create a new C++ out of C; instead, we want to speculate how C has the potential to be used for OOP.

It is usually said that OOP is another programming paradigm together with procedural and functional paradigms. But OOP is more than that. OOP is more like a way of thinking about and analyzing a problem. It is an attitude towards the universe and the hierarchy of objects within it. It is part of our ancient, intrinsic, and inherited method for comprehending and analyzing the physical and abstract entities around us. It is so fundamental to our understanding of nature.

We've always thought about every problem from an object-oriented point of view. OOP is just about applying the same point of view that humans have always adopted, but this time using a programming language to solve a computational problem. All this explains why OOP is the most common programming paradigm used for writing software.

This chapter, together with the following three chapters, are going to show that any concept within OOP can be implemented in C – even though it might be complex to do. We know we can have OOP with C because some people have already done it, especially when they created C++ on top of C, and since they have built many complex and successful programs in C in an object-oriented fashion.

What these chapters are *not* going to suggest is a certain library or set of macros that you should use for declaring a class or establishing an inheritance relation or working with other OOP concepts. In addition, we won't impose any methodology or discipline such as specific naming conventions. We will simply use raw C to implement OOP concepts.

The reason why we're dedicating **four whole chapters** to OOP with C is because of the heavy theory behind object orientation and the various examples that are necessary to be explored in order to demonstrate all of it. Most of the essential theory behind OOP is going to be explained in this chapter, while the more practical topics will be dealt with in the following chapters. With all that said, we need to discuss the theory because the OOP concepts are usually new to most skilled C programmers, even those with many years of experience.

The upcoming four chapters together cover almost anything that you might come across in OOP. In this chapter, we are going to discuss the following:

*   First of all, we'll give definitions for the most fundamental terms used in OOP literature. We'll define classes, objects, attributes, behaviors, methods, domains, and more. These terms will be used heavily throughout these four chapters. They are also vital to your understanding of other OOP-related resources because they are a staple part of the accepted language of OOP.
*   The first part of this chapter is not wholly about terminology; we'll also heavily discuss the roots of object orientation and the philosophy behind it, exploring the nature of object-oriented thinking.
*   The second section of this chapter is dedicated to C and why it is not, and cannot, be object-oriented. This is an important question that should be asked and properly answered. This topic will be further discussed in *Chapter 10*, *Unix – History and Architecture*, where we will be exploring Unix and its close relationship to C.
*   The third section of this chapter talks about *encapsulation*, which is one of the most fundamental concepts of OOP. Put simply, it's what allows you to create objects and use them. The fact that you can put variables and methods inside an object comes directly from encapsulation. This is discussed thoroughly in the third section, and several examples are given.
*   The chapter then moves on to *information-hiding*, which is something of a side effect (though a very important one) of having encapsulation. Without information-hiding, we wouldn't be able to isolate and decouple software modules, and we'd effectively be unable to provide implementation-independent APIs to clients. This is the last thing we discuss as part of this chapter.

As mentioned, the whole topic will cover four chapters, with the following chapters picking up from the *composition* relationship. From there, the upcoming chapters will cover *aggregation*, *inheritance*, *polymorphism*, *abstraction*.

In this chapter, though, we'll start with the theory behind OOP and look at how we can extract an object model from our thought process regarding a software component.

# Object-oriented thinking

As we said in the chapter introduction, object-oriented thinking is the way in which we break down and analyze what surrounds us. When you're looking at a vase on a table, you're able to understand that the vase and the table are separate objects without any heavy analysis.

Unconsciously, you are aware that there is a border between them that separates them. You know that you could change the color of the vase, and the color of the table would remain unchanged.

These observations show us that we view our environment from an object-oriented perspective. In other words, we are just creating a reflection of the surrounding object-oriented reality in our minds. We also see this a lot in computer games, 3D modeling software, and engineering software, all of which can entail many objects interacting with each other.

OOP is about bringing object-oriented thinking to software design and development. Object-oriented thinking is our default way of processing our surroundings, and that's why OOP has become the most commonly used paradigm for writing software.

Of course, there are problems that would be hard to solve if you go with the object-oriented approach, and they would have been analyzed and resolved easier if you chose another paradigm, but these problems can be considered relatively rare.

In the following sections, we are going to find out more about the translation of object-oriented thinking into writing object-oriented code.

## Mental concepts

You'd be hard-pressed to find a program that completely lacks at least some traces of object-oriented thinking, even if it had been written using C or some other non-OOP language. If a human writes a program, it will be naturally object-oriented. This will be evident even just in the variable names. Look at the following example. It declares the variables required to keep the information of 10 students:

```cpp
  char*  student_first_names[10];
  char*  student_surnames[10];
   int   student_ages[10];
double   student_marks[10];
```

Code Box 6-1: Four arrays related by having the student_ prefix, according to a naming convention, supposed to keep the information of 10 students

The declarations found in *Code Box 6-1* show how we use variable names to group some variables under the same concept, which in this case is the *student*. We have to do this; else we would get confused by ad hoc names that don't make any sense to our object-oriented minds. Suppose that we had something such as this instead:

```cpp
  char*  aaa[10];
  char*  bbb[10];
   int   ccc[10];
double   ddd[10];
```

Code Box 6-2: Four arrays with ad hoc names supposed to keep the information of 10 students!

Using such variable names as seen in *Code Box 6-2*, however much experience in programming you have, you must admit that you'd have a lot of trouble dealing with this when writing an algorithm. Variable naming is – and has always been – important, because the names remind us of the concepts in our mind and the relationships between data and those concepts. By using this kind of ad hoc naming, we lose those concepts and their relationships in the code. This may not pose an issue for the computer, but it complicates the analysis and troubleshooting for us programmers and increases the likelihood of us making mistakes.

Let's clarify more about what we mean by a concept in our current context. A concept is a mental or abstract image that exists in the mind as a thought or an idea. A *concept* could be formed by the perception of a real-world entity or could simply be entirely imaginary and abstract. When you look at a tree or when you think about a car, their corresponding images come to mind as two different concepts.

Note that sometimes we use the term concept in a different context, such as in "object-oriented concepts," which obviously doesn't use the word concept in the same way as the definition we just gave. The word concept, used in relation to technology-related topics, simply refers to the principles to understand regarding a topic. For now, we'll use this technology-related definition.

Concepts are important to object-oriented thinking because if you cannot form and maintain an understanding of objects in your mind, you cannot extract details about what they represent and relate to, and you cannot understand their interrelations.

So, object-oriented thinking is about thinking in terms of concepts and their relationships. It follows, then, that if you want to write a proper object-oriented program, you need to have a proper understanding of all the relevant objects, their corresponding concepts, and also their relationships, in your mind.

An object-oriented map formed in your mind, which consists of many concepts and their mutual interrelations, cannot be easily communicated to others, for instance when approaching a task as a team. More than that, such mental concepts are volatile and elusive, and they can get forgotten very easily. This also puts an extra emphasis on the fact that you will need models and other tools for representation, in order to translate your mind map into communicable ideas.

## Mind maps and object models

In this section, we look at an example to understand further what we've been discussing so far. Suppose that we have a written description of a scene. The purpose of describing something is to communicate the related specific concepts to the audience. Think of it this way: the one who is describing has a map in their mind that lays out various concepts and how they all link together; their aim is to communicate that mind map to the audience. You might say that this is more or less the goal of all artistic expression; it is actually what's happening when you look at a painting, listen to a piece of music, or read a novel.

Now we are going to look at a written description. It describes a classroom. Relax your mind and try to imagine what you are reading about. Everything you see in your mind is a concept communicated by the following description:

*Our classroom is an old room with two big windows. When you enter the room, you can see the windows on the opposite wall. There are a number of brown wooden chairs in the middle of the room. There are five students sitting on the chairs, and two of them are boys. There is a green, wooden blackboard on the wall to your right, and the teacher is talking to the students. He is an old man wearing a blue shirt.*

Now, let's see what concepts have formed in our minds. Before we do that though, bear in mind that your imagination can run away without you noticing. So, let's do our best to limit ourselves to the boundaries of the description. For example, I could imagine more and say that the girls are blonde. But that is not mentioned in the description, so we won't take that into account. In the next paragraph, I explain what has been shaped in my mind, and before continuing, you should also try to do that for yourself.

In my mind, there are five concepts (or mental images, or objects), one for each student in the class. There are also another five concepts for the chairs. There is another concept for the wood and another one for the glass. And I know that every chair is made from wood. This is a relationship, between the concept of wood and the concepts of the chairs. In addition, I know that every student is sitting on a chair. As such, there are five relationships – between chairs and students. We could continue to identify more concepts and relate them. In no time, we'd have a huge and complex graph describing the relationships of hundreds of concepts.

Now, pause for a moment and see how differently you were extracting the concepts and their relationships. That's a lesson that everyone can do this in a different way. This procedure also happens when you want to solve a particular problem. You need to create a mind map before attacking the problem. This is the phase that we call the *understanding phase*.

You solve a problem using an approach that is based on the concepts of the problem and the relationships you find between them. You explain your solution in terms of those concepts, and if someone wants to understand your solution, they should understand the concepts and their relationships first.

You'd be surprised if I told you this is what exactly happens when you try to solve a problem using a computer, but that is exactly the case. You break the problem into objects (same as the concepts in a mental context) and the relationships between them, and then you try to write a program, based on those objects, that eventually resolves the problem.

The program that you write simulates the concepts and their relations as you have them in your mind. The computer runs the solution, and you can verify whether it works. You are still the person who solves the problem, but now a computer is your colleague, since it can execute your solution, which is now described as a series of machine-level instructions translated from your mind map, much faster and more accurately.

An object-oriented program simulates concepts in terms of objects, and while we create a mind map for a problem in our minds, the program creates an object model in its memory. In other words, the terms *concept*, *mind*, and *mind map* are equivalent to *object*, *memory*, and *object model* respectively, if we are going to compare a human with an object-oriented program. This is the most important correlation we offer in this section, which relates the way we think to an object-oriented program.

But why are we using computers to simulate our mind maps? Because computers are good when it comes to speed and precision. This is a very classic answer to such questions, but it is still a relevant answer to our question. Creating and maintaining a big mind map and the corresponding object model is a complex task and is one that computers can do very well. As another advantage, the object model created by a program can be stored on a disk and used later.

A mind map can be forgotten or altered by emotions, but computers are emotionless, and object models are far more robust than human thoughts. That's why we should write object-oriented programs: to be able to transfer the concepts of our minds to effective programs and software.

**Note**:

So far, nothing has been invented that can download and store a mind map from someone's mind – but perhaps in the future!

## Objects are not in code

If you look at the memory of a running object-oriented program, you'll find it full of objects, all of which are interrelated. That's the same for humans. If you consider a human as a machine, you could say that they are always up and running until they die. Now, that's an important analogy. Objects can only exist in a running program, just as concepts can only exist in a living mind. That means you have objects only when you have a running program.

This may look like a paradox because when you are writing a program (an object-oriented one), the program doesn't yet exist and so cannot be running! So, how can we write object-oriented code when there is no running program and no objects?

**Note**:

When you are writing object-oriented code, no object exists. The objects are created once you build the code into an executable program and run it.

OOP is not actually about creating objects. It is about creating a set of instructions that will lead to a fully dynamic object model when the program is run. So, the object-oriented code should be able to create, modify, relate, and even delete objects, once compiled and run.

As such, writing object-oriented code is a tricky task. You need to imagine the objects and their relations before they exist. This is exactly the reason why OOP can be complex and why we need a programming language that supports object-orientation. The art of imagining something which is not yet created and describing or engineering its various details is usually called *design*. That's why this process is usually called **object-oriented design** (**OOD**) in object-oriented programming.

In object-oriented code, we only plan to create objects. OOP leads to a set of instructions for when and how an object should be created. Of course, it is not only about creation. All the operations regarding an object can be detailed using a programming language. An OOP language is a language that has a set of instructions (and grammar rules) that allow you to write and plan different object-related operations.

So far, we've seen that there is a clear correspondence between concepts in the human mind and objects in a program's memory. So, there should be a correspondence between the operations that can be performed on concepts and objects.

Every object has a dedicated life cycle. This is also true for concepts in the mind. At some point, an idea comes to mind and creates a mental image as a concept, and at some other point, it fades away. The same is true for objects. An object is constructed at one point and is destructed at another time.

As a final note, some mental concepts are very firm and constant (as opposed to volatile and transient concepts which come and go). It seems that these concepts are independent of any mind and have been in existence even when there were no minds to comprehend them. They are mostly mathematical concepts. The number 2 is an example. We have only one number 2 in the whole universe! That's amazing. It means that you and I have the very same concept in our minds of the number 2; if we tried to change it, it would no longer be the number 2\. This is exactly where we leave the object-oriented realm, and we step into another realm, full of immutable objects, that is described under the title of the *functional programming* paradigm.

## Object attributes

Each concept in any mind has some attributes associated with it. If you remember, in our classroom description, we had a chair, named *chair1*, that was brown. In other words, every chair object has an attribute called color and it was brown for the *chair1* object. We know that there were four other chairs in the classroom, and they had their color attributes which could have different values. In our description, all of them were brown, but it could be that in another description, one or two of them were yellow.

An object can have more than one attribute or a set of attributes. We call the values assigned to these attributes, collectively, the *state* of an object. The state can be thought of simply as a list of values, each one belonging to a certain attribute, attached to an object. An object can be modified during its lifetime. Such an object is said to be *mutable*. This simply means that the state can be changed during its lifetime. Objects can also be *stateless*, which means that they don't carry any state (or any attributes).

An object can be *immutable* as well, exactly like the concept (or object) corresponding to the number 2, which cannot be altered — being immutable means that the state is determined upon construction and cannot be modified after that.

**Note**:

A stateless object can be thought of as an immutable object because its state cannot be changed throughout its lifetime. In fact, it has no state to be changed.

As a final note, immutable objects are especially important. The fact that their state cannot be altered is an advantage, especially when they are shared in a multithreaded environment.

## Domain

Every program written to solve a particular problem, even an exceedingly small one, has a well-defined domain. Domain is another big term that is used widely in the literature of software engineering. The domain defines the boundaries in which software exhibits its functionality. It also defines the requirements that software should address.

A domain uses a specific and predetermined terminology (glossary) to deliver its mission and have engineers stay within its boundaries. Everyone participating in a software project should be aware of the domain in which their project is defined.

As an example, banking software is usually built for a very well-defined domain. It has a set of well-known terms as its glossary which includes account, credit, balance, transfer, loan, interest, and so on.

The definition of a domain is made clear by the terms found in its glossary; you wouldn't find the terms patient, medicine, and dosage in the banking domain, for instance.

If a programming language doesn't provide facilities for working with the concepts specific to a given domain (such as the concepts of patients and medicines in the healthcare domain), it would be difficult to write the software for that domain using that programming language – not impossible, but certainly complex. Moreover, the bigger the software is, the harder it becomes to develop and maintain.

## Relations among objects

Objects can be inter-related; they can refer to each other to denote relationships. For example, as part of our classroom description, the object *student4* (the fourth student) might be related to the object *chair3* (the third chair) in regard to a relationship named *sitting on*. In other words, *student4* sits on *chair3*. This way, all objects within a system refer to each other and form a network of objects called an object model. As we've said before, an object model is the correspondent of the mind map that we form in our minds.

When two objects are related, a change in the state of one might affect the state of the other. Let's explain this by giving an example. Suppose that we have two unrelated objects, `p1` and `p2`, representing pixels.

Object `p1` has a set of attributes as follows: `{x: 53, y: 345, red: 120, green: 45, blue: 178}`. Object `p2` has the attributes `{x: 53, y: 346, red: 79, green: 162, blue: 23}`.

**Note:**

The notation we used is almost but not quite the same as **JavaScript Object Notation** or **JSON**. In this notation, the attributes of an individual object are embraced within two curly braces, and the attributes are separated by commas. Each attribute has a corresponding value separated from the attribute by a colon.

Now, in order to make them related, they need to have an extra attribute to denote the relationship between themselves. The state of object `p1` would change to `{x: 53, y: 345, red: 120, green: 45, blue: 178, adjacent_down_pixel: p2}`, and that of `p2` would change to `{x: 53, y: 346, red: 79, green: 162, blue: 23, adjacent_up_pixel: p1}`.

The `adjacent_down_pixel` and `adjacent_up_pixel` attributes denote the fact that these pixel objects are adjacent; their `y` attributes differ only by 1 unit. Using such extra attributes, the objects realize that they are in a relationship with other objects. For instance, `p1` knows that its `adjacent_down_pixel` is `p2`, and `p2` knows that its `adjacent_up_pixel` is `p1`.

So, as we can see, if a relationship is formed between two objects, the states of those objects (or the lists of the values corresponding to their attributes) are changed. So, the relationship among objects is created by adding new attributes to them and because of that, the relationship becomes part of the objects' states. This, of course, has ramifications for the mutability or immutability of these objects.

Note that the subset of the attributes which define the state and immutability of an object can be changed from a domain to another, and it doesn't necessarily encompass all the attributes. In one domain, we might use only non-referring attributes (`x`, `y`, `red`, `green`, and `blue`, in the preceding example) as the state and in another domain, we might combine them all together with referring attributes (`adjacent_up_pixel` and `adjacent_down_pixel` in the preceding example).

## Object-oriented operations

An OOP language allows us to plan the object construction, object destruction, and altering the states of an object in a soon-to-be-running program. So, let's start by looking at the object construction.

**Note:**

The term *construction* has been chosen carefully. We could use creation or building, but these terms are not accepted as part of the standard terminology in OOP literature. Creation refers to the memory allocation for an object, while construction means the initialization of its attributes.

There are two ways to plan the construction of an object:

*   The first approach involves either constructing an empty object – one *without* any attributes in its state – or, more commonly, an object with a set of minimum attributes.
*   More attributes will be determined and added as the code is being run. Using this method, the same object can have different attributes in two different executions of the same program, in accordance with the changes found in the surrounding environment.
*   Each object is treated as a separate entity, and any two objects, even if they seem to belong to the same group (or class), by having a list of common attributes, may get different attributes in their states as the program continues.
*   As an example, the already mentioned pixel objects `p1` and `p2` are both pixels (or they both belong to the same class named `pixel`) because they have the same attributes – `x`, `y`, `red`, `green`, and `blue`. After forming a relationship, they would have different states because they then have new and different attributes: `p1` has the `adjacent_down_pixel` attribute, and `p2` has the `adjacent_up_pixel` attribute.
*   This approach is used in programming languages such as JavaScript, Ruby, Python, Perl, and PHP. Most of them are *interpreted programming languages*, and the attributes are kept as a *map* (or a *hash*) in their internal data structures that can be easily changed at runtime. This technique is usually called **prototype-based OOP**.
*   The second approach involves constructing an object whose attributes are predetermined and won't change in the middle of execution. No more attributes are allowed to be added to such an object at runtime, and the object will retain its structure. Only the values of the attributes are allowed to change, and that's possible only when the object is mutable.
*   To apply this approach, a programmer should create a predesigned *object template* or *class* that keeps track of all the attributes that need to be present in the object at runtime. Then, this template should be compiled and fed into the object-oriented language at runtime.
*   In many programming languages, this object template is called a class. Programming languages such as Java, C++, and Python use this term to denote their object templates. This technique is usually known as **class-based OOP**. Note that Python supports both prototype-based and class-based OOP.

**Note:**

A class only determines the list of attributes present in an object but not the actual values assigned to them at runtime.

Note that an object and an *instance* are the same thing, and they can be used interchangeably. However, in some texts, there might be some slight differences between them. There is also another term, *reference*, which is worth mentioning and explaining. The term object or instance is used to refer to the actual place allocated in the memory for the values of that object, while a reference is like a pointer that refers to that object. So, we can have many references referring to the same object. Generally speaking, an object usually has no name, but a reference does have a name.

**Note:**

In C, we have pointers as the corresponding syntax for references. We also have both Stack objects and Heap objects. A Heap object does not have a name and we use pointers to refer to it. In contrast, a Stack object is actually a variable and hence has a name.

While it is possible to use both approaches, C and especially C++ are officially designed in a way to support the class-based approach. Therefore, when a programmer wants to create an object in C or C++, they need to have a class first. We will talk more about the class and its role in OOP in future sections.

The following discussion might seem a bit unrelated, but, in fact, it isn't. There are two schools of thought regarding how humans grow through life, and they match quite accurately the object construction approaches that we've talked about. One of these philosophies says that the human is empty at birth and has no essence (or state).

By living and experiencing different good and bad events in life, their essence starts to grow and evolves into something that has an independent and mature character. *Existentialism* is a philosophical tradition that promotes this idea.

Its famous precept is "Existence precedes essence". This simply means that the human first comes to existence and then gains their essence through life experience. This idea is awfully close to our prototype-based approach to object construction, in which the object is constructed empty and then evolves at runtime.

The other philosophy is older and is promoted mostly by religions. In this, the human is created based on an image (or an essence), and this image has been determined before the human comes to exist. This is most similar to the way in which we plan to construct an object based on a template or class. As the object creators, we prepare a class, and then a program starts to create objects according to that class.

**Note:**

There has been a great correspondence between the approaches that people in novels or stories, including both literature and history sources, take to overcome a certain difficulty and the algorithms we have designed in computer science to solve similar problems. I deeply believe that the way humans live and the reality they experience are in great harmony with what we understand about algorithms and data structures as part of computer science. The preceding discussion was a great example of such harmony between OOP and Philosophy.

Like object construction, object destruction happens at runtime; we have only the power to plan it in code. All resources allocated by an object throughout its lifetime should be released when it is destroyed. When an object is being destructed, all other related objects should be changed so that they no longer refer to the destroyed object. An object shouldn't have an attribute that refers to a non-existent object, otherwise we lose the *referential integrity* in our object model. It can lead to runtime errors such as memory corruption or segmentation fault, as well as logical errors such as miscalculations.

Modifying an object (or altering the state of an object) can happen in two different ways. It could simply be either a change in the value of an existing attribute or it could be the addition or removal of an attribute to/from the set of attributes in that object. The latter can only happen if we have chosen the prototype-based approach to object construction. Remember that altering the state of an object that is immutable is forbidden and usually, it is not permitted by an object-oriented language.

## Objects have behaviors

Every object, together with its attributes, has a certain list of functionalities that it can perform. For instance, a car object is able to speed up, slow down, turn, and so on. In OOP, these functionalities are always in accordance with the domain requirements. For example, in a banking object model, a client can order a new account but cannot eat. Of course, the client is a person and can eat, but as long as eating functionality is not related to the banking domain, we don't consider it as a necessary functionality for a client object.

Every functionality is able to change the state of an object by altering the values of its attributes. As a simple example, a car object can accelerate. Acceleration is a functionality of the car object, and by accelerating, the speed of the car, which is one of its attributes, changes.

In summary, an object is simply a group of attributes and functionalities. In the later sections, we'll talk more about how to put these things together in an object.

So far, we have explained the fundamental terminology needed to study and understand OOP. The next step is to explain the fundamental concept of encapsulation. But, as a break, let's read about why C cannot be an OOP language.

# C is not object-oriented, but why?

C is not object-oriented, but not because of its age. If age was a reason, we could have found a way to make it object-oriented by now. But, as you will see in *Chapter 12*, *The Most Recent C*, the latest standard of the C programming language, C18, doesn't try to make C an object-oriented language.

On the other hand, we have C++, which is the result of all efforts to have an OOP language based on C. If the fate of C was for it to be replaced by an object-oriented language, then there wouldn't be any demand for C today, mainly because of C++ – but the current demand for C engineers shows that this is not the case.

A human thinks in an object-oriented way, but a CPU executes machine-level instructions which are procedural. A CPU just executes a set of instructions one by one, and from time to time, it has to jump, fetch, and execute other instructions from a different address in memory; quite similar to function calls in a program written using a procedural programming language like C.

C cannot be object-oriented because it is located on the barrier between object orientation and procedural programming. Object orientation is the human understanding of a problem and procedural execution is what a CPU can do. Therefore, we need something to be in this position and make this barrier. Otherwise high-level programs, which are usually written in an object-oriented way, cannot be translated directly into procedural instructions to be fed into the CPU.

If you look at high-level programming languages like Java, JavaScript, Python, Ruby, and so on, they have a component or layer within their architecture which bridges between their environment and the actual C library found inside the operating system (the Standard C Library in Unix-like systems and Win32 API in Windows systems). For instance, **Java Virtual Machine** (**JVM**) does that in a Java platform. While not all these environments are necessarily object-oriented (for example JavaScript or Python can be both procedural and object-oriented), they need this layer to translate their high-level logic to low-level procedural instructions.

# Encapsulation

In the previous sections, we saw that each object has a set of attributes and a set of functionalities attached to it. Here, we are going to talk about putting those attributes and functionalities into an entity called an object. We do this through a process called *encapsulation*.

Encapsulation simply means putting related things together into a *capsule* that represents an object. It happens first in your mind, and then it should be transferred to the code. The moment that you feel an object needs to have some attributes and functionalities, you are doing encapsulation in your mind; that encapsulation then needs to be transferred to the code level.

It is crucial to be able to encapsulate things in a programming language, otherwise keeping related variables together becomes an untenable struggle (we mentioned using naming conventions to accomplish this).

An object is made from a set of attributes and a set of functionalities. Both of these should be encapsulated into the object capsule. Let's first talk about *attribute encapsulation*.

## Attribute encapsulation

As we saw before, we can always use variable names to do encapsulation and tie different variables together and group them under the same object. Following is an example:

```cpp
int pixel_p1_x     = 56;
int pixel_p1_y     = 34;
int pixel_p1_red   = 123;
int pixel_p1_green = 37;
int pixel_p1_blue  = 127;
int pixel_p2_x     = 212;
int pixel_p2_y     = 994;
int pixel_p2_red   = 127;
int pixel_p2_green = 127;
int pixel_p2_blue  = 0;
```

Code Box 6-3: Some variables representing two pixels grouped by their names

This example clearly shows how variable names are used to group variables under `p1` and `p2`, which somehow are *implicit* objects. By implicit, we mean that the programmer is the only one who is aware of the existence of such objects; the programming language doesn't know anything about them.

The programming language only sees 10 variables that seem to be independent of each other. This would be a very low level of encapsulation, to such an extent that it would not be officially considered as encapsulation. Encapsulation by variable names exists in all programming languages (because you can name variables), even in an assembly language.

What we need are approaches offering *explicit* encapsulation. By explicit, we mean that both the programmer and the programming language are aware of the encapsulation and the capsules (or objects) that exist. Programming languages that do not offer explicit *attribute encapsulation* are very hard to use.

Fortunately, C does offer explicit encapsulation, and that's one of the reasons behind why we are able to write so many intrinsically object-oriented programs with it more or less easily. On the other hand, as we see shortly in the next section, C doesn't offer explicit behavior encapsulation, and we have to come up with an implicit discipline to support this.

Note that having an explicit feature such as encapsulation in a programming language is always desired. Here, we only spoke about encapsulation, but this can be extended to many other object-oriented features, such as inheritance and polymorphism. Such explicit features allow a programming language to catch relevant errors at compile time instead of runtime.

Resolving errors at runtime is a nightmare, and so we should always try to catch errors at compile time. This is the main advantage of having an object-oriented language, which is completely aware of the object-oriented way of our thinking. An object-oriented language can find and report errors and violations in our design at compile time and keep us from having to resolve many severe bugs at runtime. Indeed, this is the reason why we are seeing more complex programming languages every day – to make everything explicit to the language.

Unfortunately, not all object-oriented features are explicit in C. That's basically why it is hard to write an object-oriented program with C. But there are more explicit features in C++ and, indeed, that's why it is called an object-oriented programming language.

In C, structures offer encapsulation. Let's change the code inside *Code Box 6-3*, and rewrite it using structures:

```cpp
typedef struct {
  int x, y;
  int red, green, blue;
} pixel_t;
pixel_t p1, p2;
p1.x = 56;
p1.y = 34;
p1.red = 123;
p1.green = 37;
p1.blue = 127;
p2.x = 212;
p2.y = 994;
p2.red = 127;
p2.green = 127;
p2.blue = 0;
```

Code Box 6-4: The pixel_t structure and declaring two pixel_t variables

There are some important things to note regarding *Code Box 6-4*:

*   The attribute encapsulation happens when we put the `x`, `y`, `red`, `green`, and `blue` attributes into a new type, `pixel_t`.
*   Encapsulation always creates a new type; attribute encapsulation does this particularly in C. This is very important to note. In fact, this is the way that we make encapsulation explicit. Please note the `_t` suffix at the end of the `pixel_t`. It is very common in C to add the `_t` suffix to the end of the name of new types, but it is not mandatory. We use this convention throughout this book.
*   `p1` and `p2` will be our explicit objects when this code is executed. Both of them are of the `pixel_t` type, and they have only the attributes dictated by the structure. In C, and especially C++, types dictate the attributes to their objects.
*   The new type, `pixel_t`, is only the attributes of a class (or the object template). The word "class," remember, refers to a template of objects containing both attributes and functionalities. Since a C structure only keeps attributes, it cannot be a counterpart for a class. Unfortunately, we have no counterpart concept for a class in C; attributes and functionalities exist separately, and we implicitly relate them to each other in the code. Every class is implicit to C and it refers to a single structure together with a list of C functions. You'll see more of this in the upcoming examples, as part of this chapter and the future chapters.
*   As you see, we are constructing objects based on a template (here, the structure of `pixel_t`), and the template has the predetermined attributes that an object should have at birth. Like we said before, the structure only stores attributes and not the functionalities.
*   Object construction is very similar to the declaration of a new variable. The type comes first, then the variable name (here the object name) after that. While declaring an object, two things happen almost at the same time: first the memory is allocated for the object (creation), and then, the attributes are initialized (construction) using the default values. In the preceding example, since all attributes are integers, the default integer value in C is going to be used which is 0.
*   In C and many other programming languages, we use a dot (`.`) to access an attribute inside an object, or an arrow (`->`) while accessing the attributes of a structure indirectly through its address stored in a pointer. The statement `p1.x` (or `p1->x` if `p1` is a pointer) should be read as *the x attribute in the p1 object*.

As you know by now, attributes are certainly not the only things that can be encapsulated into objects. Now it is time to see how functionalities are encapsulated.

## Behavior encapsulation

An object is simply a capsule of attributes and methods. The method is another standard term that we usually use to denote a piece of logic or functionality being kept in an object. It can be considered as a C function that has a name, a list of arguments, and a return type. Attributes convey *values* and methods convey *behaviors*. Therefore, an object has a list of values and can perform certain behaviors in a system.

In class-based object-oriented languages such as C++, it is very easy to group a number of attributes and methods together in a class. In prototype-based languages such as JavaScript, we usually start with an empty object (*ex nihilo*, or "from nothing") or clone from an existing object. To have behaviors in the object, we need to add methods. Look at the following example, which helps you gain an insight into how prototype-based programming languages work. It is written in JavaScript:

```cpp
// Construct an empty object
var clientObj = {};
// Set the attributes
clientObj.name = "John";
clientObj.surname = "Doe";
// Add a method for ordering a bank account
clientObj.orderBankAccount = function () {
  ...
}
...
// Call the method
clientObj.orderBankAccount();
```

Code Box 6-5: Constructing a client object in JavaScript

As you see in this example, on the 2nd line, we create an empty object. In the following two lines, we add two new attributes, `name` and `surname`, to our object. And on the following line, we add a new method, `orderBankAccount`, which points to a function definition. This line is an assignment actually. On the right-hand side is an *anonymous function*, which does not have a name and is assigned to the `orderBankAccount` attribute of the object, on the left-hand side. In other words, we store a function into the `orderBankAccount` attribute. On the last line, the object's method `orderBankAccount` is called. This example is a great demonstration of prototype-based programming languages, which only rely on having an empty object at first and nothing more.

The preceding example would be different in a class-based programming language. In these languages, we start by writing a class because without having a class, we can't have any object. The following code box contains the previous example but written in C++:

```cpp
class Client {
public:
  void orderBankAccount() {
    ...
  }
  std::string name;
  std::string surname:
};
...
Client clientObj;
clientObj.name = "John";
clientObj.surname = "Doe";
...
clientObj.orderBankAccount ();
```

Code Box 6-6: Constructing the client object in C++

As you see, we started by declaring a new class, `Client`. On the 1st line, we declared a class, which immediately became a new C++ type. It resembles a capsule and is surrounded by braces. After declaring the class, we constructed the object `clientObj` from the `Client` type.

On the following lines, we set the attributes, and finally, we called the `orderBankAccount` method on the `clientObj` object.

**Note:**

In C++, methods are usually called *member functions* and attributes are called *data members*.

If you look at the techniques employed by open source and well-known C projects in order to encapsulate some items, you notice that there is a common theme among them. In the rest of this section, we are going to propose a behavior encapsulation technique which is based on the similar techniques observed in such projects.

Since we'll be referring back to this technique often, I'm going to give it a name. We call this technique **implicit encapsulation**. It's implicit because it doesn't offer an explicit behavior encapsulation that C knows about. Based on what we've got so far in the ANSI C standard, it is not possible to let C know about classes. So, all techniques that try to address object orientation in C have to be implicit.

The implicit encapsulation technique suggests the following:

*   Using C structures to keep the attributes of an object (explicit attribute encapsulation). These structures are called **attribute structures**.
*   For behavior encapsulation, C functions are used. These functions are called **behavior functions**. As you might know, we cannot have functions in structures in C. So, these functions have to exist outside the attribute structure (implicit behavior encapsulation).
*   Behavior functions must accept a structure pointer as one of their arguments (usually the first argument or the last one). This pointer points to the attribute structure of the object. That's because the behavior functions might need to read or modify the object's attributes, which is very common.
*   Behavior functions should have proper names to indicate that they are related to the same class of objects. That's why sticking to a consistent naming convention is very important when using this technique. This is one of the two naming conventions that we try to stick to in these chapters in order to have a clear encapsulation. The other one is using `_t` suffix in the names of the attribute structures. However, of course, we don't force them and you can use your own custom naming conventions.
*   The declaration statements corresponding to the behavior functions are usually put in the same header file that is used for keeping the declaration of the attribute structure. This header is called the **declaration header**.
*   The definitions of the behavior functions are usually put in one or various separate source files which include the declaration header.

Note that with implicit encapsulation, classes do exist, but they are implicit and known only to the programmer. The following example, *example 6.1*, shows how to use this technique in a real C program. It is about a car object that accelerates until it runs out of fuel and stops.

The following header file, as part of *example 6.1*, contains the declaration of the new type, `car_t`, which is the attribute structure of the `Car` class. The header also contains the declarations required for the behavior functions of the `Car` class. We use the phrase "the `Car` class" to refer to the implicit class that is missing from the C code and it encompasses collectively the attribute structure and the behavior functions:

```cpp
#ifndef EXTREME_C_EXAMPLES_CHAPTER_6_1_H
#define EXTREME_C_EXAMPLES_CHAPTER_6_1_H
// This structure keeps all the attributes
// related to a car object
typedef struct {
  char name[32];
  double speed;
  double fuel;
} car_t;
// These function declarations are 
// the behaviors of a car object
void car_construct(car_t*, const char*);
void car_destruct(car_t*);
void car_accelerate(car_t*);
void car_brake(car_t*);
void car_refuel(car_t*, double);
#endif
```

Code Box 6-7 [ExtremeC_examples_chapter6_1.h]: The declarations of the attribute structure and the behavior functions of the Car class

As you see, the attribute structure `car_t` has three fields – `name`, `speed`, and `fuel` – which are the attributes of the car object. Note that `car_t` is now a new type in C, and we can now declare variables of this type. The behavior functions are also usually declared in the same header file, as you can see in the preceding code box. They start with the `car_` prefix to put emphasis on the fact that all of them belong to the same class.

Something very important regarding the implicit encapsulation technique: each object has its own unique attribute structure variable, but all objects share the same behavior functions. In other words, we have to create a dedicated variable from the attribute structure type for each object, but we only write behavior functions once and we call them for different objects.

Note that the `car_t` attribute structure is not a class itself. It only contains the attributes of the `Car` class. The declarations all together make the implicit `Car` class. You'll see more examples of this as we go on.

There are many famous open source projects that use the preceding technique to write semi-object-oriented code. One example is `libcurl`. If you have a look at its source code, you will see a lot of structures and functions starting with `curl_`. You can find the list of such functions here: [https://curl.haxx.se/libcurl/c/allfuncs.html](https://curl.haxx.se/libcurl/c/allfuncs.html).

The following source file contains the definitions of the behavior functions as part of example 6.1:

```cpp
#include <string.h>
#include "ExtremeC_examples_chapter6_1.h"
// Definitions of the above functions
void car_construct(car_t* car, const char* name) {
  strcpy(car->name, name);
  car->speed = 0.0;
  car->fuel = 0.0;
}
void car_destruct(car_t* car) {
  // Nothing to do here!
}
void car_accelerate(car_t* car) {
  car->speed += 0.05;
  car->fuel -= 1.0;
  if (car->fuel < 0.0) {
    car->fuel = 0.0;
  }
}
void car_brake(car_t* car) {
  car->speed -= 0.07;
  if (car->speed < 0.0) {
    car->speed = 0.0;
  }
  car->fuel -= 2.0;
  if (car->fuel < 0.0) {
    car->fuel = 0.0;
  }
}
void car_refuel(car_t* car, double amount) {
  car->fuel = amount;
}
```

Code Box 6-8 [ExtremeC_examples_chapter6_1.c]: The definitions of the behavior functions as part of the Car class

The `Car`'s behavior functions are defined in *Code Box 6-8*. As you can see, all the functions accept a `car_t` pointer as their first argument. This allows the function to read and modify the attributes of an object. If a function is not receiving a pointer to an attribute structure, then it can be considered as an ordinary C function that does not represent an object's behavior.

Note that the declarations of behavior functions are usually found next to the declarations of their corresponding attribute structure. That's because the programmer is the sole person in charge of maintaining the correspondence of the attribute structure and the behavior functions, and the maintenance should be easy enough. That's why keeping these two sets close together, usually in the same header file, helps in maintaining the overall structure of the class, and eases the pain for future efforts.

In the following code box, you'll find the source file that contains the `main` function and performs the main logic. All the behavior functions will be used here:

```cpp
#include <stdio.h>
#include "ExtremeC_examples_chapter6_1.h"
// Main function
int main(int argc, char** argv) {
  // Create the object variable
  car_t car;
  // Construct the object
  car_construct(&car, "Renault");
  // Main algorithm
  car_refuel(&car, 100.0);
  printf("Car is refueled, the correct fuel level is %f\n",
    car.fuel);
  while (car.fuel > 0) {
    printf("Car fuel level: %f\n", car.fuel);
    if (car.speed < 80) {
      car_accelerate(&car);
      printf("Car has been accelerated to the speed: %f\n", 
  car.speed);
    } else {
      car_brake(&car);
      printf("Car has been slowed down to the speed: %f\n",
  car.speed);
    }
  }
  printf("Car ran out of the fuel! Slowing down ...\n");
  while (car.speed > 0) {
    car_brake(&car);
    printf("Car has been slowed down to the speed: %f\n", 
      car.speed);
  }
  // Destruct the object
  car_destruct(&car);
  return 0;
} 
```

Code Box 6-9 [ExtremeC_examples_chapter6_1_main.c]: The main function of example 6.1

As the first instruction in the `main` function, we've declared the `car` variable from the `car_t` type. The variable `car` is our first `car` object. On this line, we have allocated the memory for the object's attributes. On the following line, we constructed the object. Now on this line, we have initialized the attributes. You can initialize an object only when there is memory allocated for its attributes. In the code, the constructor accepts a second argument as the car's name. You may have noticed that we are passing the address of the `car` object to all `car_*` behavior functions.

Following that in the `while` loop, the `main` function reads the `fuel` attribute and checks whether its value is greater than zero. The fact that the `main` function, which is not a behavior function, is able to access (read and write) the `car`'s attributes is an important thing. The `fuel` and `speed` attributes, for instance, are examples of *public* attributes, which functions (external code) other than the behavior functions can access. We will come back to this point in the next section.

Before leaving the `main` function and ending the program, we've destructed the `car` object. This simply means that resources allocated by the object have been released at this phase. Regarding the `car` object in this example, there is nothing to be done for its destruction, but it is not always the case and destruction might have steps to be followed. We will see more of this in the upcoming examples. The destruction phase is mandatory and prevents memory leaks in the case of Heap allocations.

It would be good to see how we could write the preceding example in C++. This would help you to get an insight into how an OOP language understands classes and objects and how it reduces the overhead of writing proper object-oriented code.

The following code box, as part of *example 6.2*, shows the header file containing the `Car` class in C++:

```cpp
#ifndef EXTREME_C_EXAMPLES_CHAPTER_6_2_H
#define EXTREME_C_EXAMPLES_CHAPTER_6_2_H
class Car {
public:
  // Constructor
  Car(const char*);
  // Destructor
  ~Car();
  void Accelerate();
  void Brake();
  void Refuel(double);
  // Data Members (Attributes in C)
  char name[32];
  double speed;
  double fuel;
};
#endif
```

Code Box 6-10 [ExtremeC_examples_chapter6_2.h]: The declaration of the Car class in C++

The main feature of the preceding code is the fact that C++ knows about classes. Therefore, the preceding code demonstrates an explicit encapsulation; both attribute and behavior encapsulations. More than that, C++ supports more object-oriented concepts such as constructors and destructors.

In the C++ code, all the declarations, both attributes and behaviors, are encapsulated in the class definition. This is the explicit encapsulation. Look at the two first functions that we have declared as the constructor and the destructor of the class. C doesn't know about the constructors and destructors; but C++ has a specific notation for them. For instance, the destructor starts with ~ and it has the same name as the class does.

In addition, as you can see, the behavior functions are missing the first pointer argument. That's because they all have access to the attributes inside the class. The next code box shows the content of the source file that contains the definition of the declared behavior functions:

```cpp
#include <string.h>
#include "ExtremeC_examples_chapter6_2.h"
Car::Car(const char* name) {
  strcpy(this->name, name);
  this->speed = 0.0;
  this->fuel = 0.0;
}
Car::~Car() {
  // Nothing to do
}
void Car::Accelerate() {
  this->speed += 0.05;
  this->fuel -= 1.0;
  if (this->fuel < 0.0) {
    this->fuel = 0.0;
  }
}
void Car::Brake() {
  this->speed -= 0.07;
  if (this->speed < 0.0) {
    this->speed = 0.0;
  }
  this->fuel -= 2.0;
  if (this->fuel < 0.0) {
    this->fuel = 0.0;
  }
}
void Car::Refuel(double amount) {
  this->fuel = amount;
}
```

Code Box 6-11 [ExtremeC_examples_chapter6_2.cpp]: The definition of the Car class in C++

If you look carefully, you'll see that the `car` pointer in the C code has been replaced by a `this` pointer, which is a keyword in C++. The keyword `this` simply means the current object. I'm not going to explain it any further here, but it is a smart workaround to eliminate the pointer argument in C and make behavior functions simpler.

And finally, the following code box contains the `main` function that uses the preceding class:

```cpp
// File name: ExtremeC_examples_chapter6_2_main.cpp
// Description: Main function
#include <iostream>
#include "ExtremeC_examples_chapter6_2.h"
// Main function
int main(int argc, char** argv) {
  // Create the object variable and call the constructor
  Car car("Renault");
  // Main algorithm
  car.Refuel(100.0);
  std::cout << "Car is refueled, the correct fuel level is "
    << car.fuel << std::endl;
  while (car.fuel > 0) {
    std::cout << "Car fuel level: " << car.fuel << std::endl;
    if (car.speed < 80) {
      car.Accelerate();
      std::cout << "Car has been accelerated to the speed: "
        << car.speed << std::endl;
    } else {
      car.Brake();
      std::cout << "Car has been slowed down to the speed: "
        << car.speed << std::endl;
    }
  }
  std::cout << "Car ran out of the fuel! Slowing down ..."
    << std::endl;
  while (car.speed > 0) {
    car.Brake();
    std::cout << "Car has been slowed down to the speed: "
      << car.speed << std::endl;
  }
  std::cout << "Car is stopped!" << std::endl;
  // When leaving the function, the object 'car' gets
  // destructed automatically.
  return 0;
}
```

Code Box 6-12 [ExtremeC_examples_chapter6_2_main.cpp]: The main function of example 6.2

The `main` function written for C++ is very similar to the one we wrote for C, except that it allocates the memory for a class variable instead of a structure variable.

In C, we can't put attributes and behavior functions in a bundle that is known to C. Instead, we have to use files to group them. But in C++, we have a syntax for this bundle, which is the *class definition*. It allows us to put data members (or attributes) and member functions (or behavior functions) in the same place.

Since C++ knows about the encapsulation, it is redundant to pass the pointer argument to the behavior functions, and as you can see, in C++, we don't have any first pointer arguments in member function declarations like those we see in the C version of the `Car` class.

So, what happened? We wrote an object-oriented program in both C, which is a procedural programming language, and in C++, which is an object-oriented one. The biggest change was using `car.Accelerate()` instead of `car_accelerate(&car)`, or using `car.Refuel(1000.0)` instead of `car_refuel(&car, 1000.0)`.

In other words, if we are doing a call such as `func(obj, a, b, c, ...)` in a procedural programming language, we can do it as `obj.func(a, b, c, ...)` in an object-oriented language. They are equivalent but coming from different programming paradigms. Like we said before, there are numerous examples of C projects that use this technique.

**Note:**

In *Chapter 9*, *Abstraction and OOP in C++*, you will see that C++ uses exactly the same preceding technique in order to translate high-level C++ function calls to low-level C function calls.

As a final note, there is an important difference between C and C++ regarding object destruction. In C++, the destructor function is invoked automatically whenever an object is allocated on top of the Stack and it is going out of scope, like any other Stack variable. This is a great achievement in C++ memory management, because in C, you may easily forget to call the destructor function and eventually experience a memory leak.

Now it is time to talk about other aspects of encapsulation. In the next section, we will talk about a consequence of encapsulation: information-hiding.

## Information hiding

So far, we've explained how encapsulation bundles attributes (which represent values) and functionalities (which represent behaviors) together to form objects. But it doesn't end there.

Encapsulation has another important purpose or consequence, which is *information-hiding*. Information-hiding is the act of protecting (or hiding) some attributes and behaviors that should not be visible to the outer world. By the outer world, we mean all parts of the code that do not belong to the behaviors of an object. By this definition, no other code, or simply no other C function, can access a private attribute or a private behavior of an object if that attribute or behavior is not part of the public interface of the class.

Note that the behaviors of two objects from the same type, such as `car1` and `car2` from the `Car` class, can access the attributes of any object from the same type. That's because of the fact that we write behavior functions once for all objects in a class.

In *example 6.1*, we saw that the `main` function was easily accessing the `speed` and `fuel` attributes in the `car_t` attribute structure. This means that all attributes in the `car_t` type were public. Having a public attribute or behavior can be a bad thing because it might have some long-lasting and dangerous.

As a consequence, the implementation details could leak out. Suppose that you are going to use a car object. Usually, it is only important to you that it has a behavior that accelerates the car; and you are not curious about how it is done. There may be even more internal attributes in the object that contribute to the acceleration process, but there is no valid reason that they should be visible to the consumer logic.

For instance, the amount of the electrical current being delivered to the engine starter could be an attribute, but it should be just private to the object itself. This also holds for certain behaviors that are internal to the object. For example, injecting the fuel into the combustion chamber is an internal behavior that should not be visible and accessible to you, otherwise, you could interfere with that and interrupt the normal process of the engine.

From another point of view, the implementation details (how the car works) vary from one car manufacturer to another but being able to accelerate a car is a behavior that is provided by all car manufacturers. We usually say that being able to accelerate a car is part of the *public API* or the *public interface* of the `Car` class.

Generally, the code using an object becomes dependent on the public attributes and behaviors of that object. This is a serious concern. Leaking out an internal attribute by declaring it public at first and then making it private can effectively break the build of the dependent code. It is expected that other parts of the code that are using that attribute as a public thing won't get compiled after the change.

This would mean you've broken the backward compatibility. That's why we choose a conservative approach and make every single attribute private by default until we find sound reasoning for making it public.

To put it simply, exposing private code from a class effectively means that rather than being dependent on a light public interface, we have been dependent on a thick implementation. These consequences are serious and have the potential to cause a lot of rework in a project. So, it is important to keep attributes and behaviors as private as they can be.

The following code box, as part of *example 6.3*, will demonstrate how we can have private attributes and behaviors in C. The example is about a `List` class that is supposed to store some integer values:

```cpp
#ifndef EXTREME_C_EXAMPLES_CHAPTER_6_3_H
#define EXTREME_C_EXAMPLES_CHAPTER_6_3_H
#include <unistd.h>
// The attribute structure with no disclosed attribute
struct list_t;
// Allocation function
struct list_t* list_malloc();
// Constructor and destructor functions
void list_init(struct list_t*);
void list_destroy(struct list_t*);
// Public behavior functions
int list_add(struct list_t*, int);
int list_get(struct list_t*, int, int*);
void list_clear(struct list_t*);
size_t list_size(struct list_t*);
void list_print(struct list_t*);
#endif
```

Code Box 6-13 [ExtremeC_examples_chapter6_3.h]: The public interface of the List class

What you see in the preceding code box is the way that we make the attributes private. If another source file, such as the one that contains the `main` function, includes the preceding header, it'll have no access to the attributes inside the `list_t` type. The reason is simple. The `list_t` is just a declaration without a definition, and with just a structure declaration, you cannot access the fields of the structure. You cannot even declare a variable out of it. This way, we guarantee the information-hiding. This is actually a great achievement.

Once again, before creating and publishing a header file, it is mandatory to double-check whether we need to expose something as public or not. By exposing a public behavior or a public attribute, you'll create dependencies whose breaking would cost you time, development effort, and eventually money.

The following code box demonstrates the actual definition of the `list_t` attribute structure. Note that it is defined inside a source file and not a header file:

```cpp
#include <stdio.h>
#include <stdlib.h>
#define MAX_SIZE 10
// Define the alias type bool_t
typedef int bool_t;
// Define the type list_t
typedef struct {
 size_t size;
 int* items;
} list_t;
// A private behavior which checks if the list is full
bool_t __list_is_full(list_t* list) {
  return (list->size == MAX_SIZE);
}
// Another private behavior which checks the index
bool_t __check_index(list_t* list, const int index) {
  return (index >= 0 && index <= list->size);
}
// Allocates memory for a list object
list_t* list_malloc() {
  return (list_t*)malloc(sizeof(list_t));
}
// Constructor of a list object
void list_init(list_t* list) {
  list->size = 0;
  // Allocates from the heap memory
  list->items = (int*)malloc(MAX_SIZE * sizeof(int));
}
// Destructor of a list object
void list_destroy(list_t* list) {
  // Deallocates the allocated memory
  free(list->items);
}
int list_add(list_t* list, const int item) {
  // The usage of the private behavior
  if (__list_is_full(list)) {
    return -1;
  }
  list->items[list->size++] = item;
  return 0;
}
int list_get(list_t* list, const int index, int* result) {
  if (__check_index(list, index)) {
    *result = list->items[index];
    return 0;
  }
  return -1;
}
void list_clear(list_t* list) {
  list->size = 0;
}
size_t list_size(list_t* list) {
  return list->size;
}
void list_print(list_t* list) {
  printf("[");
  for (size_t i = 0; i < list->size; i++) {
    printf("%d ", list->items[i]);
  }
  printf("]\n");
}
```

Code Box 6-14 [ExtremeC_examples_chapter6_3.c]: The definition of the List class

All the definitions that you see in the preceding code box are private. The external logic that is going to use a `list_t` object does not know anything about the preceding implementations, and the header file is the only piece of code that the external code will be dependent on.

Note that the preceding file has not even included the header file! As long as the definitions and function signatures match the declarations in the header file, that's all that's needed. However, it is recommended to do so because it guarantees the compatibility between the declarations and their corresponding definitions. As you've seen in *Chapter 2*, *Compilation and Linking*, the source files are compiled separately and finally linked together.

In fact, the linker brings private definitions to the public declarations and makes a working program out of them.

**Note:**

We can use a different notation for private behavior functions. We use the prefix `__` in their names. As an example, the `__check_index` function is a private function. Note that a private function does not have any corresponding declaration in the header file.

The following code box contains *example 6.3*'s `main` function that creates two list objects, populates the first one, and uses the second list to store the reverse of the first list. Finally, it prints them out:

```cpp
#include <stdlib.h>
#include "ExtremeC_examples_chapter6_3.h"
int reverse(struct list_t* source, struct list_t* dest) {
  list_clear(dest);
  for (size_t i = list_size(source) - 1; i >= 0; i--) {
    int item;
    if(list_get(source, i, &item)) {
      return -1;
    }
    list_add(dest, item);
  }
  return 0;
}
int main(int argc, char** argv) {
  struct list_t* list1 = list_malloc();
  struct list_t* list2 = list_malloc();
  // Construction
  list_init(list1);
  list_init(list2);
  list_add(list1, 4);
  list_add(list1, 6);
  list_add(list1, 1);
  list_add(list1, 5);
  list_add(list2, 9);
  reverse(list1, list2);
  list_print(list1);
  list_print(list2);
  // Destruction
  list_destroy(list1);
  list_destroy(list2);
  free(list1);
  free(list2);
  return 0;
}
```

Code Box 6-15 [ExtremeC_examples_chapter6_3_main.c]: The main function of example 6.3

As you can see in the preceding code box, we wrote the `main` and `reverse` functions only based on the things declared in the header file. In other words, these functions are using only the public API (or public interface) of the `List` class; the declarations of the attribute structure `list_t` and its behavior functions. This example is a nice demonstration of how to break the dependencies and hide the implementation details from other parts of the code.

**Note:**

Using the public API, you can write a program that compiles, but it cannot turn into a real working program unless you provide the corresponding object files of the private part and link them together.

There are some points related to the preceding code that we explore in more detail here. We needed to have a `list_malloc` function in order to allocate memory for a `list_t` object. Then, we can use the function `free` to release the allocated memory when we're done with the object.

You cannot use `malloc` directly in the preceding example. That's because if you are going to use `malloc` inside the `main` function, you have to pass `sizeof(list_t)` as the required number of bytes that should be allocated. However, you cannot use `sizeof` for an incomplete type.

The `list_t` type included from the header file is an *incomplete type* because it is just a declaration that doesn't give any information regarding its internal fields, and we don't know its size while compiling it. The real size will be determined only at link time when we know the implementation details. As a solution, we had to have the `list_malloc` function defined and have `malloc` used in a place where `sizeof(list_t)` is determined.

In order to build *example 6.3*, we need to compile the sources first. The following commands produce the necessary object files before the linking phase:

```cpp
$ gcc -c ExtremeC_examples_chapter6_3.c -o private.o
$ gcc -c ExtremeC_examples_chapter6_3_main.c -o main.o
```

Shell Box 6-1: Compiling example 6.3

As you see, we have compiled the private part into `private.o` and the main part into `main.o`. Remember that we don't compile header files. The public declarations in the header are included as part of the `main.o` object file.

Now we need to link the preceding object files together, otherwise `main.o` alone cannot turn into an executable program. If you try to create an executable file using only `main.o`, you will see the following errors:

```cpp
$ gcc main.o -o ex6_3.out
main.o: In function 'reverse':
ExtremeC_examples_chapter6_3_main.c:(.text+0x27): undefined reference to 'list_clear'
...
main.o: In function 'main':
ExtremeC_examples_chapter6_3_main.c:(.text+0xa5): undefined reference to 'list_malloc'
...                                                                                                                               collect2: error: ld returned 1 exit status
$
```

Shell Box 6-2: Trying to link example 6.3 by just providing main.o

You see that the linker cannot find the definitions of the functions declared in the header file. The proper way to link the example is as follows:

```cpp
$ gcc main.o private.o -o ex6_3.out
$ ./ex6_3.out
[4 6 1 5 ]
[5 1 6 4 ]
$
```

Shell Box 6-3: Linking and running example 6.3

What happens if you change the implementation behind the `List` class?

Say, instead of using an array, you use a linked list. It seems that we don't need to generate the `main.o` again, because it is nicely independent of the implementation details of the list it uses. So, we need only to compile and generate a new object file for the new implementation; for example, `private2.o`. Then, we just need to relink the object files and get the new executable:

```cpp
$ gcc main.o private2.o -o ex6_3.out
$ ./ex6_3.out
[4 6 1 5 ]
[5 1 6 4 ]
$
```

Shell Box 6-4: Linking and running example 6.3 with a different implementation of the List class

As you see, from the user's point of view, nothing has changed, but the underlying implementation has been replaced. That is a great achievement and this approach is being used heavily in C projects.

What if we wanted to not repeat the linking phase in case of a new list implementation? In that case, we could use a shared library (or `.so` file) to contain the private object file. Then, we could load it dynamically at runtime, removing the need to relink the executable again. We have discussed shared libraries as part of *Chapter 3*, *Object Files*.

Here, we bring the current chapter to an end and we will continue our discussion in the following chapter. The next two chapters will be about the possible relationships which can exist between two classes.

# Summary

In this chapter, the following topics have been discussed:

*   We gave a thorough explanation of object-orientation philosophy and how you can extract an object model from your mind map.
*   We also introduced the concept of the domain and how it should be used to filter the mind map to just keep relevant concepts and ideas.
*   We also introduced the attributes and behaviors of a single object and how they should be extracted from either the mind map or the requirements given in the description of a domain.
*   We explained why C cannot be an OOP language and explored its role in the translation of OOP programs into low-level assembly instructions that eventually will be run on a CPU.
*   Encapsulation, as the first principle in OOP, was discussed. We use encapsulation to create capsules (or objects) that contain a set of attributes (placeholders for values) and a set of behaviors (placeholders for logic).
*   Information-hiding was also discussed, including how it can lead to interfaces (or APIs) that can be used without having to become dependent on the underlying implementation.
*   While discussing information-hiding, we demonstrated how to make attributes or methods private in C code.

The next chapter will be the opening to the discussion regarding possible relations between classes. We start *Chapter 7*, *Composition, and Aggregation*, with talking about composition relationship and then, we continue with inheritance and polymorphism as part of *Chapter 8*, *Inheritance and Polymorphism*.