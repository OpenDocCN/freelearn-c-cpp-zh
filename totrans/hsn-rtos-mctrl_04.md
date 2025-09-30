# Task Signaling and Communication Mechanisms

In the previous chapter, the task was introduced. Toward the end, we looked at examples of preemptive scheduling for multiple tasks in the system and the fact that a task will run whenever it isn't waiting on something (in the blocked state) and can do something useful. In this chapter, the core mechanisms for task signaling and inter-task communication will be briefly introduced. These primitives are fundamental to event-driven parallel programming, which is the foundation of a well implemented RTOS-based application.

Rather than dive right into the FreeRTOS API, each primitive will be presented along with a few graphical examples and some suggestions on when each of the mechanisms can be used. Don't worry: in later chapters, we'll get into the nitty-gritty of working with the API. For now, let's concentrate on the fundamentals.

In this chapter, we'll be introducing the following topics:

*   RTOS queues
*   RTOS semaphores
*   RTOS mutexes

# Technical requirements

There are no software or hardware requirements for this chapter.

# RTOS queues

Queues are quite simple in concept, but they are also extremely powerful and flexible, especially if you've traditionally programmed on bare metal with C. At its heart, a queue is simply a circular buffer. However, this buffer contains some very special properties, such as native multi-thread safety, the flexibility for each queue to hold any type of data, and waking up other tasks that are waiting on an item to appear in the queue. By default, data is stored in queues using **First In First Out** (**FIFO**) ordering – the first item to be put into the queue is the first item to be removed from the queue.

We'll start by taking a look at some simple behavior of queues when they are in different states and used in different ways (sending versus receiving) and then move on to how queues can be used to pass information between tasks.

# Simple queue send

The first queue example is simply adding (also referred to as *sending)* an item to a queue that has empty space:

![](img/851688d9-f8de-47c0-8d9c-8754ef87f4ae.png)

When an item is added to a queue with available space, the addition happens immediately. Because space was available in the queue, the task *sending* the item to the queue continues running, unless there is another higher priority task waiting on an item to appear in the queue.

Although interaction with queues typically happens from within tasks, this isn't *always* the case. There are some special cases where queues can also be accessed from within ISRs (but that behavior has different rules). For the examples in this chapter, we'll assume that tasks are sending and receiving items from the queues.

# Simple queue receive

In the following diagram, a task is shown *receiving* an item from a queue: 

![](img/7d18370c-5991-4c5f-a0f8-1324f6460aee.png)

When a task is ready to receive an item from a queue, by default, it will get the oldest item. In this example, since there is at least one item in the queue, the *receive *is processed immediately and the task continues to run.

# Full queue send

When a queue is full, no information is thrown away. Instead, the task attempting to *send* the item to the queue will wait for up to a predetermined amount of time for available space in the queue:

![](img/deda0050-1e45-4fb6-bdfc-34fd51212d02.png)

When a queue is full, the task that is attempting to send an item to the queue will wait until a space in the queue becomes available, but only up to the timeout value specified.

In this example, if a task was attempting to send to a full queue and it has a timeout of 10 ms – it would only wait 10 ms for space to become available in the queue. After the timeout expires, the call will return and notify the calling code that the send has failed. What to do regarding this failure is at the discretion of the programmer setting up the calling code and will vary depending on the use case. Extremely large timeout values can be used for truly non-critical functions. Just be aware that this will cause the sending task to effectively wait forever for a slot in the queue to become available (this is obviously no longer real time)!

Your code will typically be structured so that attempts to send to a queue won't timeout. It is up to you, as the programmer, to determine what an acceptable amount of time is on a case-by-case basis. It is also your responsibility to determine the severity and corrective actions if a timeout does occur. Potential corrective actions could range from nothing (think of a dropped frame in a video call) to an emergency shutdown.

# Empty queue receive

Another case where accessing a queue can cause a task to block is attempting to *receive* from an empty queue:

![](img/2ebe3c14-9c4d-4212-948e-2a329269f695.png)

Similar to a *Send* waiting on space to become available, a task *receiving* from a queue also has the potential to be delayed. In the case of an empty queue, the task that is attempting to receive from the queue will be blocked until an item appears in the queue. If no item is available before the timeout expires, the calling code will be notified of the failure. Again, the exact course of action to take varies. 

Sometimes, infinite waits are used. You'll often encounter very long wait periods for queues that are receiving input from external interfaces, such as serial ports, which may not be sending data constantly. There is no issue at all if a human user on the other end of a serial port hasn't sent data for an extended period of time.

On the other hand, receive timeouts can also be used to ensure that you have a minimum acceptable amount of data to process. Let's use a sensor that is meant to provide a new reading at 10 Hz (10 readings per second). If you were implementing an algorithm that relies on *fresh* readings from this sensor, a timeout of slightly greater than 100 ms could be used to trigger an error. This timeout would guarantee that the algorithm is always acting on *fresh* sensor readings. In this case, hitting a timeout could be used to trigger some type of corrective action or notification that the sensor wasn't performing as expected.

# Queues for inter-task communication

Now that the simple behaviors of queues have been covered, we'll take a look at how they can be used to move data between tasks. A very common use case for queues is to have one task populate the queue while another is reading from the same queue. This is generally straightforward but may have some nuances, depending on how the system is set up:

![](img/e055b8d1-e83c-47c0-8c7e-dce58908e629.png)

In the preceding example, `Task 1` and `Task 2` are both interacting with the same Queue. `Task 1` will *send* an item to the Queue. As long as `Task 2` has a higher priority than `Task 1`, it will immediately *receive* the item.

Let's consider another instance that often occurs in practice when multiple tasks are interacting with queues. Since a preemptive scheduler always runs the task with the highest priority, if that task always has data to write to the queue, the queue will fill before another task is given a chance to read from the queue. Here is an example of how this may play out:

![](img/27e18540-cd90-40d7-b29f-04d827ec5b28.png)

The following numbers correspond to indexes along the time axis:

1.  `Task2` attempts to receive an item from the empty queue. No items are available, so `Task2` blocks.
2.  `Task1` adds items to queue. Since it is the highest priority task in the system, `Task1` adds items to the queue until it doesn't have any more items to add, or until the queue is full.
3.  The queue is filled, so `Task1` is blocked.
4.  `Task2` is given context by the scheduler since it is now the highest priority task that may run.
5.  As soon as an item is removed from the Queue, `Task1` is given context again (this is the highest priority task in the system and it is now able to run because it was blocking while waiting for space to become available in the queue). After adding a single item, the queue is full and `Task1` is blocked.
6.  `Task2` is given context and receives an item from the queue:

A real-world example of the preceding situation is covered in [Chapter 9](495bdcc0-2a86-4b22-9628-4c347e67e49e.xhtml),* Intertask Communication*,in the section *Passing data through queues*.`Chapter_9/src/*mainQueueCom**positePassByValue.c*`illustrates the exact setup and a thorough empirical execution analysis is performed using `SystemView`.

![](img/50f984da-cb2d-43a4-82bc-a20941886c49.png)

Another extremely common use case for queues is to have a single queue accept input from many different sources. This is especially useful for something like a debug serial port or a log file. Many different tasks can be writing to the queue, with a single task responsible for receiving data from the queue and pushing it out to the shared resource. 

While queues are generally used for passing data between tasks, semaphores can be used for signaling and synchronizing tasks. Let's learn more about this next.

# RTOS semaphores

Semaphores are another very straightforward, but powerful, construct. The word *semaphore* has a Greek origin – the approximate English translation is *sign-bearer*, which is a wonderfully intuitive way to think about them. Semaphores are used to indicate that something has happened; they signal events. Some example use cases of semaphores include the following:

*   An ISR is finished servicing a peripheral. It may *give* a semaphore to provide tasks with a signal indicating that data is ready for further processing.
*   A task has reached a juncture where it needs to wait for other tasks in the system to *catch up* before moving on. In this case, a semaphore could be used to synchronize the tasks.
*   Restricting the number of simultaneous users of a restricted resource.

One of the convenient aspects of using an RTOS is the pre-existence of semaphores. They are included in every implementation of an RTOS because of how basic (and crucial) their functionality is. There are two different types of semaphores to cover – counting semaphores and binary semaphores.

# Counting semaphores

Counting semaphores are most often used to manage a shared resource that has limitations on the number of simultaneous users. Upon creation, they can be configured to hold a maximum value, called a *ceiling. *The example normally given for counting semaphores is readers in a database ... Well, we're talking about an MCU-based embedded system here, so let's keep our examples relevant. If you're interested in databases, you're probably better off with a general-purpose OS! For our example, let's say you're implementing a socket-based communication driver and your system only has enough memory for a limited number of simultaneous socket connections.

In the following diagram, we have a shared network resource that can accommodate two simultaneous socket connections. However, there are three tasks that need access. A counting semaphore is used to limit the number of simultaneous socket connections. Each time a task is finished with the shared resource (that is, its socket closes), it must give back its semaphore so another task can gain access to the network. If a task happens to *give* a semaphore that is already at its maximum count, the count will remain unchanged:

![](img/4fc26255-cb6f-4221-975f-cdceff8b8b3b.png)

The preceding diagram plays out the example of a shared resource only capable of servicing two simultaneous tasks (although three tasks in the system need to use the resource). If a task is going to use a socket, which is protected by the counting semaphore, it must first *take* a semaphore from the pool. If no semaphore is available, then the task must wait until a semaphore becomes available:

1.  Initially, a semaphore is created with a maximum (ceiling) of 2 and an initial count of 0.
2.  When `TaskA` and `TaskB` attempt to *take* a `semphr`, they're immediately successful. At this point, they can each open up a socket and communicate over the network. 
3.  `TaskC` was a bit later, so it will need to wait until the count of `semphr` is less than 2, which is when a network socket will be free to use.
4.  After `TaskB` is finished communicating over its socket, it returns the semaphore. 
5.  Now that a semaphore is available, `TaskC` completes its *take* and is allowed network access.
6.  Shortly after `TaskC` gets access, `TaskB` has another message to send, so it attempts to take a semaphore, but needs to wait for one to become available, so it is put to sleep.
7.  While `TaskC` is communicating over the network, `TaskA` finishes and returns its semaphore. 
8.  `TaskB` is woken up and its take completes, which enables it to start communicating over the network. 
9.  After `TaskB` is given its semaphore, `TaskC` completes its transaction and gives back the semaphore it had.

*Waiting* for semaphores is where an RTOS differs from most other semaphore implementations – a task can *timeout* during a semaphore wait. If a task fails to acquire the semaphore in time, it must not access the shared resource. Instead, it must take an alternative course of action. This alternative could be any number of actions that can range from a failure so severe that it triggers an emergency shutdown sequence, to something so benign that it is merely mentioned in a log file or pushed to a debug serial port for analysis later on. As a programmer, it is up to you to determine what the appropriate course of action is, which can sometimes prompt some difficult discussions with other disciplines.

# Binary semaphores

Binary semaphores are really just counting semaphores with a maximum count of 1\. They are most often used for synchronization. When a task needs to synchronize on an event, it will attempt to *take* a semaphore, blocking until the semaphore becomes available or until the specified timeout has elapsed. Another asynchronous part of the system (either a task or an ISR) will *give* a semaphore. Binary semaphores can be *given* more than once; there is no need for that piece of code to *return* them. In the following example, `TaskA` only *gives* a semaphore, while `TaskB` only *takes* a semaphore:

![](img/02afc9ad-fd70-4699-be36-885fc8545359.png)

`TaskB` is set up to wait for a signal (semaphore) before proceeding with its duties: 

1.  Initially, `TaskB` attempts to *take* the semaphore, but it wasn't available, so `TaskB` went to sleep. 
2.  Sometime later, `TaskA` *gives* the signal.
3.  `TaskB` is woken up (by the scheduler; this happens in the background) and now has the semaphore. It will go about the duties required of it until it is finished. Notice, however, that `TaskB` doesn't need to give back the binary semaphore. Instead, it simply waits for it again. 
4.  `TaskB` is blocked again because the semaphore isn't available (just like the first time), so it goes to sleep until a semaphore is available.
5.  The cycle repeats.

If `TaskB` were to "give back" the binary semaphore, it would immediately run again, without receiving the go-ahead from `TaskA`. The result would just be a loop running full speed, rather than being signaled on a condition signaled from `TaskA`.

Next, we'll discuss a special type of semaphore with some additional properties that make it especially well suited for protecting items that can be accessed from different tasks – the mutex.

# RTOS mutexes

The term **mutex** is shorthand for **mutual** **exclusion.** In the context of shared resources and tasks, mutual exclusion means that, if a task is using a shared resource, then that task is the *only* t*ask* that is permitted to use the resource – all others will need to wait.

If all this sounds a lot like a binary semaphore, that's because it is. However, it has an additional feature that we'll cover soon. First, let's take a look at the problem with using a binary semaphore to provide mutual exclusion.

# Priority inversion

Let's look at a common problem that occurs when attempting to use a binary semaphore to provide mutual exclusion functionality.

Consider three tasks, A, B, and C, where A has the highest priority, B the middle priority, and C the lowest priority. Tasks A and C rely on a semaphore to give access to a resource that is shared between them. Since Task A is the highest priority task in the system, it should always be running before other tasks. However, since Task A and Task C both rely on a resource shared between them (guarded by the binary semaphore), there is an unexpected dependency here:

![](img/b12a6d93-2088-4e62-b827-d7898385ba24.png)

Let's walk step by step through the example to see how this scenario plays out:

1.  `Task C` (the lowest priority task in the system) acquires a binary semaphore and starts to do some work.
2.  Before `Task C` completes its work, `Task A` (the highest priority task) interrupts and attempts to acquire the same semaphore, but is forced to wait because `Task C` has already acquired the semaphore.
3.  `Task B` preempts `Task C` as well, because `Task B` has a higher priority than `Task C`. `Task B` performs whatever work it has and then goes to sleep.
4.  `Task C` runs through the remainder of its work with the shared resource, at which point it gives the semaphore back.
5.  `Task A` is *finally* able to run.

`Task A` was able to run, eventually, but not until TWO lower priority tasks had run. `Task C` finishing its work with the shared resource was unavoidable (unless a design change could have been made to prevent it from accessing the same shared resource as `Task A`). However, `Task B` also had an opportunity to run to completion, even though `Task A` was waiting around and had a higher priority! This is priority inversion – a higher priority task in the system is waiting to run, but it is forced to wait while a lower priority task is running instead – the priorities of the two tasks are effectively *inverted* in this case.

# Mutexes minimize priority inversion

Earlier, we had said that, in FreeRTOS, mutexes were binary semaphores with one important additional feature. That important feature is priority inheritance – mutexes have the ability to temporarily change the priority of a task to avoid causing major delays in the system. This plays out when the scheduler finds that a high priority task is attempting to acquire a mutex already held by a lower priority task. In this specific case, the scheduler will temporarily increase the priority of the lower task until it releases the mutex. At this point, the priority of the lower task will be set back to what it was prior to the priority inheritance. Let's take a look at the exact same example from the preceding diagram implemented using a mutex (instead of a binary semaphore):

![](img/5543d397-b1a5-4986-8c80-0eda35913a16.png)

Let's walk step by step through the example to see how this scenario plays out:

1.  `Task A` is still waiting for `Task C` to return the mutex.
2.  The priority of `Task C` is brought up to be the same as that of the higher priority `Task A`. `Task C` runs to completion since it holds the mutex and is a high priority task.
3.  `Task C` returns the mutex and its priority is demoted to whatever it was before it was holding a mutex that was delaying the high priority task.
4.  `Task A` takes the mutex and completes its work.
5.  `Task B` is allowed to run.

Depending on how long `Task C` is taking with the shared resource and how time sensitive `Task A` is, this could either be a major source of concern or no big deal. Timing analysis can be performed to ensure that `Task A` is still meeting deadlines, but tracking all possible causes of priority inversion and other high priority asynchronous events could prove to be challenging. At a minimum, the user should make use of the built-in timeouts provided for taking mutexes and perform a suitable alternative action if a mutex has failed to be taken in a timely manner. More details on exactly how to accomplish this can be found in [Chapter 9](495bdcc0-2a86-4b22-9628-4c347e67e49e.xhtml), *Intertask Communication*.

Mutexes and semaphores are fairly standard mechanisms for signaling between tasks. They are very standard between different RTOSes and provide excellent flexibility.

# Summary

This chapter introduced queues, semaphores, and mutexes. A few common use cases of each of these core building blocks for RTOS applications were also discussed at a high level and some of the more subtle behaviors of each were highlighted. The diagrams presented in this chapter should serve as a reference point to return to as we move on to more complex real-world examples in later chapters.

We've now covered some of the core RTOS concepts. In the next chapter, we'll turn our attention to another very important step of developing a solid real-time system. This step affects how efficiently firmware can be run and has a major impact on system performance – MCU selection.

# Questions 

As we conclude, here is a list of questions for you to test your knowledge regarding this chapter's material. You will find the answers in the *Assessments* section of the Appendix:

1.  Which RTOS primitive is most commonly used for sending and receiving data between tasks?
2.  Is it possible for a queue to interact with more than two tasks?
3.  Which RTOS primitive is commonly used for signaling and synchronizing tasks?
4.  What is an example of when a counting semaphore could be used?
5.  Name one major difference between binary semaphores and mutexes.
6.  When protecting a resource shared between tasks, should a binary semaphore or a mutex be used?
7.  What is priority inversion and why is it dangerous for a real-time system?