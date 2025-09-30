# Implementing Concurrency

Multitasking is a key feature in almost all operating systems; it increases the efficiency of the CPU and utilizes resources in a better manner. Threads are the best way to implement multitasking. A process can contain more than one thread to implement multitasking. 

In this chapter, we will cover the following recipes involving threads:

*   Performing a task with a single thread
*   Performing multiple tasks with multiple threads
*   Using `mutex` to share data between two threads
*   Understanding how a deadlock is created
*   Avoiding a deadlock 

The terms process and thread can be confusing, so first, we'll make sure that you understand them.

# What are processes and threads?

Whenever we run a program, the moment that it is loaded from the hard disk (or any other storage) into the memory, it becomes a process**.** A **process** is executed by a processor, and for its execution, it requires a **program counter (PC)** to keep track of the next instruction to be executed, the CPU registers, the signals, and so on.

A **thread** refers to a set of instructions within a program that can be executed independently. A thread has its own PC and set of registers, among other things. In that way, a process is comprised of several threads. Two or more threads can share their code, data, and other resources, but special care must be taken when sharing resources among threads, as it might lead to ambiguity and deadlock. An operating system also manages a thread pool. 

A **thread pool** contains a collection of threads that are waiting for tasks to be allocated to them for concurrent execution. Using threads from the thread pool instead of instantiating new threads helps to avoid the delay that is caused by creating and destroying new threads; hence, it increases the overall performance of the application.

Basically, threads enhance the efficiency of an application through parallelism, that is, by running two or more independent sets of code simultaneously. This is called **multithreading**.

Multithreading is not supported by C, so to implement it, POSIX threads (`Pthreads`) are used. GCC allows for the implementation of a `pthread`.

While using a `pthread`, a variable of the type `pthread_t` is defined to store the thread identifier. A **thread identifier** is a unique integer, that is ,assigned to a thread in the system.

You must be wondering which function is used for creating a thread. The `pthread_create` function is invoked to create a thread. The following four arguments are passed to the `pthread_create` function:

*   A pointer to the thread identifier, which is set by this function
*   The attributes of the thread; usually, `NULL` is provided for this argument to use the default attributes
*   The name of the function to execute for the creation of the thread
*   The arguments to be passed to the thread, set to `NULL` if no arguments need to be passed to the thread

When two or more threads operate on the same data, that is, when they share the same resources, certain check measures must be applied so that only one thread is allowed to manipulate the shared resource at a time; other threads' access must be blocked. One of the methods that helps to avoid ambiguity when a resource is shared among threads is mutual exclusion.

# Mutual exclusion

To avoid ambiguity when two or more threads access the same resource, **mutual exclusion** implements serializing access to the shared resources. When one thread is using a resource, no other thread is allowed to access the same resource. All of the other threads are blocked from accessing the same resource until the resource is free again.

A `mutex` is basically a lock that is associated with the shared resource. To read or modify the shared resource, a thread must first acquire the lock for that resource. Once a thread acquires a lock (or `mutex`) for that resource, it can go ahead with processing that resource. All of the other threads that wish to work on it will be compelled to wait until the resource is unlocked. When the thread finishes its processing on the shared resource, it unlocks the `mutex`, enabling the other waiting threads to acquire a `mutex` for that resource. Aside from `mutex`, a semaphore is also used in process synchronization.

A **semaphore** is a concept that is used to avoid two or more processes from accessing a common resource in a concurrent system. It is basically a variable that is manipulated to only allow one process to have access to a common resource and implement process synchronization. A semaphore uses the signaling mechanism, that is, it invokes `wait` and `signal` functions, respectively, to inform that the common resource has been acquired or released. A `mutex`, on the other hand, uses the locking mechanism—the process has to acquire the lock on the `mutex` object before working on the common resource.

Although `mutex` helps to manage shared resources among threads, there is a problem. An application of `mutex` in the wrong order may lead to a deadlock. A deadlock occurs in a situation when a thread that has `lock X` tries to acquire `lock Y` to complete its processing, while another thread that has `lock Y` tries to acquire `lock X` to finish its execution. In such a situation, a deadlock will occur, as both of the threads will keep waiting indefinitely for the other thread to release its lock. As no thread will be able to finish its execution, no thread will be able to free up its locks, either. One solution to avoid a deadlock is to let threads acquire locks in a specific order.

The following functions are used to create and manage threads:

*   `pthread_join`: This function makes the thread wait for the completion of all its spawned threads. If it is not used, the thread will exit as soon as it completes its task, ignoring the states of its spawned threads. In other words, `pthread_join` blocks the calling thread until the thread specified in this function terminates.
*   `pthread_mutex_init`: This function initializes the `mutex` object with the specified attributes. If `NULL` is used for the attributes, the default `mutex` attributes are used for initializing the `mutex` object. When the `mutex` is initialized, it is in an unlocked state.
*   `pthread_mutex_lock`: This function locks the specified `mutex` object. If the `mutex` is already locked by some other thread, the calling thread will get suspended, that is, it will be asked to wait until the `mutex` gets unlocked. This function returns the `mutex` object in a locked state. The thread that locks the `mutex` becomes its owner and remains the owner until it unlocks the `mutex`.
*   `pthread_mutex_unlock`: This function releases the specified `mutex` object. The thread that has invoked the `pthread_mutex_lock` function and is waiting for the `mutex` to get unlocked will become unblocked and acquire the `mutex` object, that is, the waiting thread will be able to access and lock the `mutex` object. If there are no threads waiting for the `mutex`, the `mutex` will remain in the unlocked state without any owner thread.
*   `pthread_mutex_destroy`: This function destroys a `mutex` object and frees up the resources allocated to it. The `mutex` must be in an unlocked state before invoking this method.

Depending on the operating system, a lock may be a **spinlock**. If any thread tries to acquire a lock but the lock is not free, a spinlock will make the thread wait in a loop until the lock becomes free. Such locks keep the thread busy while it's waiting for the lock to free up. They are efficient, as they avoid the consumption of time and resources in process rescheduling or context switching.

That is enough theory. Now, let's start with some practical examples!

# Performing a task with a single thread

In this recipe, we will be creating a thread to perform a task. In this task, we will display the sequence numbers from `1` to `5`. The focus of this recipe is to learn how a thread is created and how the main thread is asked to wait until the thread finishes its task.

# How to do it…

1.  Define a variable of the type `pthread_t` to store the thread identifier:

```cpp
pthread_t tid;
```

2.  Create a thread and pass the identifier that was created in the preceding step to the `pthread_create` function. The thread is created with the default attributes. Also, specify a function that needs to be executed to create the thread:

```cpp
pthread_create(&tid, NULL, runThread, NULL);
```

3.  In the function, you will be displaying a text message to indicate that the thread has been created and is running:

```cpp
printf("Running Thread \n");
```

4.  Invoke a `for` loop to display the sequence of numbers from `1` to `5` through the running thread:

```cpp
for(i=1;i<=5;i++) printf("%d\n",i);
```

5.  Invoke the `pthread_join` method in the main function to make the `main` method wait until the thread completes its task:

```cpp
pthread_join(tid, NULL);
```

The `createthread.c` program for creating a thread and making it perform a task is as follows:

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

void *runThread(void *arg)
{
    int i;
    printf("Running Thread \n");
    for(i=1;i<=5;i++) printf("%d\n",i);
    return NULL;
}

int main()
{
    pthread_t tid;
    printf("In main function\n");
    pthread_create(&tid, NULL, runThread, NULL);
    pthread_join(tid, NULL);
    printf("Thread over\n");
    return 0;
}
```

Now, let's go behind the scenes.

# How it works...

We will define a variable called `tid` of the type `pthread_t` to store the thread identifier. A **thread identifier** is a unique integer, that is, assigned to a thread in the system. Before creating a thread, the message `In main function` is displayed on the screen. We will create a thread and pass the identifier `tid` to the `pthread_create` function. The thread is created with the default attributes, and the `runThread` function is set to execute to create the thread.

In the `runThread` function, we will display the text message `Running Thread` to indicate that the thread was created and is running. We will invoke a `for` loop to display the sequence of numbers from `1` to `5` through the running thread. By invoking the `pthread_join` method, we will make the `main` method wait until the thread completes its task. It is essential to invoke the `pthread_join` here; otherwise, the `main` method will exit without waiting for the completion of the thread.

Let's use GCC to compile the `createthread.c` program, as follows:

```cpp
D:\CBook>gcc createthread.c -o createthread
```

If you get no errors or warnings, that means the `createthread.c` program  has been compiled into an executable file, `createthread.exe`. Let's run this executable file:

![](img/c7707dfb-2a8a-47ac-b7c5-bf97b96c5e84.png)

Voila! We've successfully completed a task with a single thread. Now, let's move on to the next recipe!

# Performing multiple tasks with multiple threads

In this recipe, you will learn how to multitask by executing two threads in parallel. Both of the threads will do their tasks independently. As the two threads will not be sharing a resource, there will not be a situation of race condition or ambiguity. The CPU will execute any thread randomly at a time, but finally, both of the threads will finish the assigned task. The task that the two threads will perform is displaying the sequence of numbers from `1` to `5`.

# How to do it…

1.  Define two variables of the type `pthread_t` to store two thread identifiers:

```cpp
pthread_t tid1, tid2;
```

2.  Invoke the `pthread_create` function twice to create two threads, and assign the identifiers that we created in the previous step. The two threads are created with the default attributes. Specify two respective functions that need to be executed for the two threads:

```cpp
pthread_create(&tid1,NULL,runThread1,NULL);
pthread_create(&tid2,NULL,runThread2,NULL);
```

3.  In the function of the first thread, display a text message to indicate that the first thread was created and is running:

```cpp
printf("Running Thread 1\n");
```

4.  To indicate the execution of the first thread, execute a `for` loop in the first function to display the sequence of numbers from `1` to `5`. To distinguish from the second thread, the sequence of numbers that were generated by the first thread are prefixed by `Thread 1`:

```cpp
for(i=1;i<=5;i++)
    printf("Thread 1 - %d\n",i);
```

5.  Similarly, in the second thread, display a text message to inform that the second thread has also been created and is running:

```cpp
  printf("Running Thread 2\n");
```

6.  Again, in the second function, execute a `for` loop to display the sequence of numbers from `1` to `5`. To differentiate these numbers from the ones generated by `thread1`, this sequence of numbers will be preceded by the text `Thread 2`:

```cpp
for(i=1;i<=5;i++)
    printf("Thread 2 - %d\n",i);
```

7.  Invoke the `pthread_join` twice, and pass the thread identifiers we created in step 1 to it. `pthread_join` will make the two threads, and the `main` method will wait until both of the threads have completed their tasks:

```cpp
pthread_join(tid1,NULL);
pthread_join(tid2,NULL);
```

8.  When both of the threads are finished, a text message will be displayed to confirm this:

```cpp
printf("Both threads are over\n");
```

The `twothreads.c` program for creating two threads and making them work on independent resources is as follows:

```cpp
#include<pthread.h>
#include<stdio.h>

void *runThread1(void *arg){
    int i;
    printf("Running Thread 1\n");
    for(i=1;i<=5;i++)
        printf("Thread 1 - %d\n",i);
}

void *runThread2(void *arg){
    int i;
    printf("Running Thread 2\n");
    for(i=1;i<=5;i++)
        printf("Thread 2 - %d\n",i);
}

int main(){
    pthread_t tid1, tid2;
    pthread_create(&tid1,NULL,runThread1,NULL);
    pthread_create(&tid2,NULL,runThread2,NULL);
    pthread_join(tid1,NULL);
    pthread_join(tid2,NULL);
    printf("Both threads are over\n");
    return 0;
}
```

Now, let's go behind the scenes.

# How it works...

We will define two variables of the type `pthread_t`, by the names `tid1` and `tid2`, to store two thread identifiers. These thread identifiers uniquely represent the threads in the system. We will invoke the `pthread_create` function twice to create two threads and assign their identifiers to the two variables `tid1` and `tid2`, whose addresses are passed to the `pthread_create` function.

The two threads are created with the default attributes. We will execute the function `runThread1` to create the first thread, and then the `runThread2` function to create the second thread.

In the `runThread1` function, we will display the message `Running Thread 1` to indicate that the first thread was created and is running. In addition, we will invoke a `for` loop to display the sequence of numbers from `1` to `5` through the running thread. The sequence of numbers that are generated by the first thread will be prefixed by `Thread 1`.

Similarly, in the `runThread2` function, we will display the message `Running Thread 2` to inform that the second thread was also created and is running. Again, we will invoke a `for` loop to display the sequence of numbers from `1` to `5`. To differentiate these numbers from the ones generated by `thread1`, these numbers are preceded by the text `Thread 2`.

We will then invoke the `pthread_join` method twice and pass our two thread identifiers, `tid1` and `tid2`, to it. The `pthread_join` is invoked to make the two threads, and the `main` method waits until both of the threads have completed their respective tasks. When both of the threads are over, that is, when the functions `runThread1` and `runThread2` are over, a message saying that `Both threads are over` will be displayed in the `main` function.

Let's use GCC to compile the `twothreads.c` program, as follows:

```cpp
D:\CBook>gcc twothreads.c -o twothreads
```

If you get no errors or warnings, that means the `twothreads.c` program  has been compiled into an executable file, `twothreads.exe`. Let's run this executable file:

![](img/ff390ba6-5215-4415-851b-69a3a23e003a.png)

You may not get exactly the same output, as it depends on the CPU, but it is certain that both threads will exit simultaneously.

Voila! We've successfully completed multiple tasks with multiple threads. Now, let's move on to the next recipe!

# Using mutex to share data between two threads

Running two or more threads independently, where each accesses its own resources, is quite convenient. However, sometimes, we want the threads to share and process the same resource simultaneously so that we can finish a task faster. Sharing a common resource may lead to problems, as one thread might read the data before the other thread writes the updated data, leading to an ambiguous situation. To avoid such a situation, `mutex` is used. In this recipe, you will learn how to share common resources between two threads.

# How to do it…

1.  Define two variables of the `pthread_t` type  to store two thread identifiers. Also, define a `mutex` object:

```cpp
pthread_t tid1,tid2;
pthread_mutex_t lock;
```

2.  Invoke the `pthread_mutex_init` method to initialize the `mutex` object with the default `mutex` attributes:

```cpp
pthread_mutex_init(&lock, NULL)
```

3.  Invoke the `pthread_create` function twice to create two threads, and assign the identifiers that we created in step 1\. Execute a function for creating the two threads:

```cpp
pthread_create(&tid1, NULL, &runThread, NULL);
pthread_create(&tid2, NULL, &runThread, NULL);
```

4.  In the function, the `pthread_mutex_lock` method is invoked and the `mutex` object is passed to it to lock it:

```cpp
pthread_mutex_lock(&lock);
```

5.  Invoke the `pthread_self` method and assign the ID of the calling thread to a variable of the `pthread_t` type. Invoke the `pthread_equal` method and compare it with the variable to find out which thread is currently executing. If the first thread is being executed, display the message `First thread is running` on the screen:

```cpp
pthread_t id = pthread_self();
if(pthread_equal(id,tid1))                                
    printf("First thread is running\n");
```

6.  To indicate that the thread is executing a common resource, display the text message `Processing the common resource` on the screen:

```cpp
printf("Processing the common resource\n");
```

7.  Invoke the `sleep` method to make the first thread sleep for `5` seconds:

```cpp
sleep(5);
```

8.  After a duration of `5` seconds, display the message `First thread is over` on the screen:

```cpp
printf("First thread is over\n\n");
```

9.  The `pthread_mutex_unlock` function will be invoked, and the `mutex` object that we created in the first step will be passed to it to unlock it:

```cpp
pthread_mutex_unlock(&lock);  
```

10.  The `thread` function will be invoked by the second thread. Lock the `mutex` object again:

```cpp
pthread_mutex_lock(&lock);

```

11.  To indicate that the second thread is running at the moment, display the message `Second thread is running` on the screen:

```cpp
printf("Second thread is running\n");
```

12.  Again, to indicate that the common resource is being accessed by the thread, display the message `Processing the common resource` on the screen:

```cpp
printf("Processing the common resource\n");
```

13.  Introduce a delay of `5` seconds. Then, display the message `second thread is over` on the screen:

```cpp
sleep(5);
printf("Second thread is over\n\n"); 
```

14.  Unlock the `mutex` object:

```cpp
pthread_mutex_unlock(&lock);  
```

15.  Invoke the `pthread_join` method twice and pass the thread identifiers to it:

```cpp
pthread_join(tid1, NULL);
pthread_join(tid2, NULL);
```

16.  Invoke the `pthread_mutex_destroy` method to destroy the `mutex` object:

```cpp
pthread_mutex_destroy(&lock);
```

The `twothreadsmutex.c` program for creating two threads that share common resources is as follows:

```cpp
#include<stdio.h>
#include<pthread.h>
#include<unistd.h>
pthread_t tid1,tid2;
pthread_mutex_t lock;

void* runThread(void *arg)
{
    pthread_mutex_lock(&lock);
    pthread_t id = pthread_self();
    if(pthread_equal(id,tid1))
        printf("First thread is running\n");
    else
        printf("Second thread is running\n");
    printf("Processing the common resource\n");
    sleep(5);
    if(pthread_equal(id,tid1))
        printf("First thread is over\n\n");
    else
        printf("Second thread is over\n\n"); 
    pthread_mutex_unlock(&lock);  
    return NULL;
}

int main(void)
{ 
    if (pthread_mutex_init(&lock, NULL) != 0)
        printf("\n mutex init has failed\n");
    pthread_create(&tid1, NULL, &runThread, NULL);
    pthread_create(&tid2, NULL, &runThread, NULL);
    pthread_join(tid1, NULL);
    pthread_join(tid2, NULL);
    pthread_mutex_destroy(&lock);
    return 0;
}
```

Now, let's go behind the scenes.

# How it works...

We will first define a `mutex` object by the name `lock`. Recall that a `mutex` is basically a lock associated with a shared resource. To read or modify the shared resource, a thread needs to first acquire the lock for that resource. We will define two variables of the `pthread_t` type , with the names `tid1` and `tid2`, to store two thread identifiers.

We will invoke the `pthread_mutex_init` method that initializes the `lock` object with the default `mutex` attributes. When it's initialized, the `lock` object is in an unlocked state. We then invoke the `pthread_create` function twice to create two threads and assign their identifiers to the two variables `tid1` and `tid2`, whose addresses are passed to the `pthread_create` function. The two threads are created with the default attributes.

Next, we will execute the `runThread` function to create the two threads. In the `runThread` function, we will invoke the `pthread_mutex_lock` method and pass the `mutex` object `lock` to it to lock it. Now, the rest of the threads (if any) will be asked to wait until the `mutex` object `lock` is unlocked. We will invoke the `pthread_self` method and assign the ID of the calling thread to the variable `id` of the `pthread_t` type. We will then invoke the `pthread_equal` method to ensure that if the calling thread is the one with the identifier assigned to the `tid1` variable, then the message `First thread is running` will display on the screen.

Next, the message `Processing the common resource` is displayed on the screen. We will invoke the `sleep` method to make the first thread sleep for `5` seconds. After a duration of `5` seconds, the message `First thread is over` will be displayed on the screen to indicate that the first thread is over. We will then invoke `pthread_mutex_unlock` and pass the `mutex` object `lock` to it to unlock it. Unlocking the `mutex` object is an indication to the other threads that the common resource can be used by other threads, too.

The `runThread` method will be invoked by the second thread, with the identifier `tid2`. Again, the `mutex` object `lock` is locked, and the `id` of the calling thread, that is, the second thread, is assigned to the variable `id`. The message `Second thread is running` is displayed on the screen, followed by the message `Processing the common resource`.

We will introduce a delay of `5` seconds to indicate that the second thread is processing the common resource. Then, the message `second thread is over` will be displayed on the screen. The `mutex` object `lock` is now unlocked. We will invoke the `pthread_join` method twice and pass the `tid1` and `tid2` thread identifiers to it. `pthread_join` is invoked to make the two threads and the `main` method wait until both of the threads have completed their tasks.

When both of the threads are over, we will invoke the `pthread_mutex_destroy` method to destroy the `mutex` object `lock` and free up the resources allocated to it.

Let's use GCC to compile the `twothreadsmutex.c` program, as follows:

```cpp
D:\CBook>gcc twothreadsmutex.c -o twothreadsmutex
```

If you get no errors or warnings, that means the `twothreadsmutex.c` program has been compiled into an executable file, `twothreadsmutex.exe`. Let's run this executable file:

![](img/cc35a7de-e737-4389-a59c-7f71884a88f5.png)

Voila! We've successfully used `mutex` to share data between two threads. Now, let's move on to the next recipe!

# Understanding how a deadlock is created

Locking a resource helps in non-ambiguous results, but locking can also lead to a deadlock. A **deadlock** is a situation wherein a thread has acquired the lock for one resource and wants to acquire the lock for a second resource. However, at the same time, another thread has acquired the lock for the second resource, but wants the lock for the first resource. Because the first thread will keep waiting for the second resource lock to be free and the second thread will keep waiting for the first resource lock to be free, the threads will not be able to proceed further, and the application will hang (as the following diagram illustrates):

![](img/7ddc7847-dee9-4cfc-b4b9-932e1deb7809.png)

In this recipe, we will use a stack. A stack requires two operations—`push` and `pop`. To make only one thread execute a `push` or `pop` operation at a time, we will use two `mutex` objects—`pop_mutex` and `push_mutex`. The thread needs to acquire locks on both of the objects to operate on the stack. To create a situation of deadlock, we will make a thread acquire one lock and ask it to acquire another lock, which was already acquired by another thread.

# How to do it…

1.  Define a macro of the value `10`, and define an array of an equal size:

```cpp
#define max 10
int stack[max];
```

2.  Define two `mutex` objects; one will be used while popping from the stack (`pop_mutex`), and the other will be used while pushing a value to the stack (`push_mutex`):

```cpp
pthread_mutex_t pop_mutex;
pthread_mutex_t push_mutex;
```

3.  To use the `stack`, initialize the value of `top` to `-1`:

```cpp
int top=-1;
```

4.  Define two variables of the type `pthread_t` to store two thread identifiers:

```cpp
pthread_t tid1,tid2;
```

5.  Invoke the `pthread_create` function to create the first thread; the thread will be created with the default attributes. Execute the `push` function to create this thread:

```cpp
pthread_create(&tid1,NULL,&push,NULL);
```

6.  Invoke the `pthread_create` function again to create the second thread; this thread will also be created with the default attributes. Execute the `pop` function to create this thread:

```cpp
pthread_create(&tid2,NULL,&pop,NULL);
```

7.  In the `push` function, invoke the `pthread_mutex_lock` method and pass the `mutex` object for the `push` operation (`push_mutex`) to lock it:

```cpp
pthread_mutex_lock(&push_mutex);
```

8.  Then, the `mutex` object for the `pop` operation (`pop_mutex`) will be locked by the first thread:

```cpp
pthread_mutex_lock(&pop_mutex);
```

9.  The user is asked to enter the value to be pushed to the `stack`:

```cpp
printf("Enter the value to push: ");
scanf("%d",&n);
```

10.  The value of `top` is incremented to `0`. The value that was entered in the previous step is pushed to the location `stack[0]`:

```cpp
top++;
stack[top]=n;
```

11.  Invoke `pthread_mutex_unlock` and unlock the `mutex` objects meant for the `pop` (`pop_mutex`) and `push` operations (`push_mutex`):

```cpp
pthread_mutex_unlock(&pop_mutex);                                                       pthread_mutex_unlock(&push_mutex);  
```

12.  At the bottom of the `push` function, display a text message indicating that the value is pushed to the stack:

```cpp
printf("Value is pushed to stack \n");
```

13.  In the `pop` function, invoke the `pthread_mutex_lock` function to lock the `mutex` object `pop_mutex`. It will lead to a deadlock:

```cpp
pthread_mutex_lock(&pop_mutex);
```

14.  Again, try to lock the `push_mutex` object, too (although it is not possible, as it is always acquired by the first thread):

```cpp
sleep(5);
pthread_mutex_lock(&push_mutex);
```

15.  The value in the stack, that is, pointed to by the `top` pointer is popped:

```cpp
k=stack[top];
```

16.  Thereafter, the value of `top` is decremented by `1` to make it `-1` again. The value, that, is, popped from the stack is displayed on the screen:

```cpp
top--;
printf("Value popped is %d \n",k);
```

17.  Then, unlock the `mutex` object `push_mutex` and the `pop_mutex` object:

```cpp
pthread_mutex_unlock(&push_mutex);     
pthread_mutex_unlock(&pop_mutex);
```

18.  In the `main` function, invoke the `pthread_join` method and pass the thread identifiers that were created in step 1 to it:

```cpp
pthread_join(tid1,NULL);
pthread_join(tid2,NULL);
```

The `deadlockstate.c` program for creating two threads and understanding how a deadlock occurs while acquiring locks is as follows:

```cpp
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>

#define max 10
pthread_mutex_t pop_mutex;
pthread_mutex_t push_mutex;
int stack[max];
int top=-1;

void * push(void *arg) {
    int n;
    pthread_mutex_lock(&push_mutex);
    pthread_mutex_lock(&pop_mutex);
    printf("Enter the value to push: ");
    scanf("%d",&n);
    top++;
    stack[top]=n;
    pthread_mutex_unlock(&pop_mutex);
    pthread_mutex_unlock(&push_mutex);
    printf("Value is pushed to stack \n");
}
void * pop(void *arg) {
    int k;
    pthread_mutex_lock(&pop_mutex);
    pthread_mutex_lock(&push_mutex);
    k=stack[top];
    top--;
    printf("Value popped is %d \n",k);
    pthread_mutex_unlock(&push_mutex);
    pthread_mutex_unlock(&pop_mutex);
}

int main() {
    pthread_t tid1,tid2;
    pthread_create(&tid1,NULL,&push,NULL);
    pthread_create(&tid2,NULL,&pop,NULL);
    printf("Both threads are created\n");
    pthread_join(tid1,NULL);
    pthread_join(tid2,NULL);
    return 0;
}
```

Now, let's go behind the scenes.

# How it works...

We will first define a macro called `max` of the value `10`, along with an array stack of the size `max`. Then, we will define two `mutex` objects with the names `pop_mutex` and `push_mutex`. To use the `stack`, we will initialize the value of `top` to `-1`. We will also define two variables of the type `pthread_t`, with the names `tid1` and `tid2`, to store two thread identifiers.

We will invoke the `pthread_create` function to create the first thread, and we will assign the identifier returned by the function to the variable `tid1`. The thread will be created with the default attributes, and we will execute the `push` function to create this thread.

We will invoke the `pthread_create` function again to create the second thread, and we will assign the identifier returned by the function to the variable `tid2`. This thread is also created with the default attributes, and we will execute the `pop` function to create this thread. On the screen, we will display the message `Both threads are created`.

In the `push` function, we will invoke the `pthread_mutex_lock` method and pass the `mutex` object `push_mutex` to it to lock it. Now, if any other thread asks for the `push_mutex` object, it will need to wait until the object is unlocked.

Then, the `mutex` object `pop_mutex` will be locked by the first thread. We will be asked to enter the value to be pushed to the stack. The entered value will be assigned to the variable `n`. The value of `top` will be incremented to `0`.  The value that we enter will be pushed to the location `stack[0]`. 

Next, we will invoke the `pthread_mutex_unlock` and pass the `mutex` object `pop_mutex` to it to unlock it. Also, the `mutex` object `push_mutex` will be unlocked. At the bottom of the `push` function, we will display the message `Value is pushed to stack`.

In the `pop` function, the `mutex` object `pop_mutex` will be locked, and then it will try to lock the `push_mutex` object that is already locked by first thread. The value in the stack, that is, pointed at by the pointer `top` will be popped. Because the value of `top` is `0`, the value at the `stack[0]` location will be picked up and assigned to the variable `k`. Thereafter, the value of `top` will decrement by `1` to make it `-1` again. The value, that is, popped from the stack will be displayed on the screen. Then, the `mutex` object `push_mutex` will be unlocked, followed by unlocking the `pop_mutex` object.

In the `main` function, we will invoke the `pthread_join` method twice and pass the `tid1` and `tid2` thread identifiers to it. The reason that we invoke the `pthread_join` method is to make the two threads and the `main` method wait until both of the threads have completed their tasks.

In this program, a deadlock has occurred because in the `push` function, the first thread locked the `push_mutex` object and tried to get the lock of the `pop_mutex` object, which was already locked by the second thread in the `pop` function. In the `pop` function, the thread locked the `mutex` object `pop_mutex` and tried to lock the `push_mutex` object, which was already locked by the first thread. So, neither of the threads will be able to finish, and they will keep waiting indefinitely for the other thread to release its `mutex` object.

Let's use GCC to compile the `deadlockstate.c` program, as follows:

```cpp
D:\CBook>gcc deadlockstate.c -o deadlockstate
```

If you get no errors or warnings, that means the `deadlockstate.c` program is compiled into an executable file, `deadlockstate.exe`. Let's run this executable file:

![](img/79c47507-1c97-4278-9737-a3a8613babab.png)

You've now seen how a deadlock can occur. Now, let's move on to the next recipe!

# Avoiding a deadlock

A deadlock can be avoided if the threads are allowed to acquire the locks in a sequence. Let's suppose that a thread acquires the lock for a resource and wants to acquire the lock for a second resource. Any other thread that tries to acquire the first lock will be asked to wait, as it was already acquired by the first thread. Therefore, the second thread will not be able to acquire the lock for the second resource either, since it can only acquire locks in a sequence. However, our first thread will be allowed to acquire the lock to the second resource without waiting.

Applying a sequence to the locking of resources is the same as allowing only one thread to acquire resources at a time. The other threads will only be able to acquire the resources after the previous thread is over. This way, we will not have a deadlock on our hands.

# How to do it…

1.  Define an array of `10` elements:

```cpp
#define max 10
int stack[max];
```

2.  Define two `mutex` objects—one to indicate the `pop` operation of the stack (`pop_mutex`), and another to represent the `push` operation of the stack (`push_mutex`):

```cpp
pthread_mutex_t pop_mutex;
pthread_mutex_t push_mutex;
```

3.  To use the `stack`, the value of `top` is initialized to `-1`:

```cpp
int top=-1;
```

4.  Define two variables of the type `pthread_t`, to store two thread identifiers:

```cpp
pthread_t tid1,tid2;
```

5.  Invoke the `pthread_create` function to create the first thread. The thread is created with the default attributes, and the `push` function is executed to create the thread:

```cpp
pthread_create(&tid1,NULL,&push,NULL);
```

6.  Invoke the `pthread_create` function again to create the second thread. The thread is created with the default attributes, and the `pop` function is executed to create this thread:

```cpp
pthread_create(&tid2,NULL,&pop,NULL);
```

7.  To indicate that the two threads were created, display the message `Both threads are created`:

```cpp
printf("Both threads are created\n");
```

8.  In the `push` function, invoke the `pthread_mutex_lock` method and pass the `mutex` object `push_mutex`, related to the `push` operation, to it, in order to lock it:

```cpp
pthread_mutex_lock(&push_mutex);
```

9.  After a sleep of `2` seconds, the `mutex` object, that is, meant to invoke the `pop` operation `pop_mutex` will be locked by the first thread:

```cpp
sleep(2);
pthread_mutex_lock(&pop_mutex);
```

10.  Enter the value to be pushed to the stack:

```cpp
printf("Enter the value to push: ");
scanf("%d",&n);
```

11.  The value of `top` is incremented to `0`. To `stack[0]` location, the value, that is, entered by the user is pushed:

```cpp
top++;
stack[top]=n;
```

12.  Invoke `pthread_mutex_unlock` and pass the `mutex` object `pop_mutex` to it to unlock it. Also, the `mutex` object `push_mutex` will be unlocked:

```cpp
pthread_mutex_unlock(&pop_mutex);                                                   pthread_mutex_unlock(&push_mutex);
```

13.  At the bottom of the `push` function, display the message `Value is pushed to stack`:

```cpp
printf("Value is pushed to stack \n");
```

14.  In the `pop` function, the `pthread_mutex_lock` function is invoked to lock the `mutex` object `push_mutex`:

```cpp
pthread_mutex_lock(&push_mutex);
```

15.  After a sleep (or delay) of `5` seconds, the `pop` function will try to lock the `pop_mutex` object, too. However, the `pthread_mutex_lock` function will not be invoked, as the thread is kept waiting for the `push_mutex` object to be unlocked:

```cpp
sleep(5);
pthread_mutex_lock(&pop_mutex);
```

16.  The value in the stack pointed to by the pointer `top` is popped. Because the value of `top` is `0`, the value at the location `stack[0]` is picked up:

```cpp
k=stack[top];
```

17.  Thereafter, the value of `top` will be decremented by `1` to make it `-1` again. The value, that is, popped from the stack will be displayed on the screen:

```cpp
top--;
printf("Value popped is %d \n",k);
```

18.  Then, the `mutex` object `pop_mutex` will be unlocked, followed by the `push_mutex` object:

```cpp
pthread_mutex_unlock(&pop_mutex);
pthread_mutex_unlock(&push_mutex);
```

19.  In the `main` function, invoke the `pthread_join` method twice and pass the thread identifiers that were created in step 1 to it:

```cpp
pthread_join(tid1,NULL);
pthread_join(tid2,NULL);
```

The `avoiddeadlockst.c` program for creating two threads and understanding how a deadlock can be avoided while acquiring locks is as follows:

```cpp
#include <stdio.h>
#include <pthread.h>
#include<unistd.h>
#include <stdlib.h>

#define max 10
pthread_mutex_t pop_mutex;
pthread_mutex_t push_mutex;
int stack[max];
int top=-1;

void * push(void *arg) {
    int n;
    pthread_mutex_lock(&push_mutex);
    sleep(2);
    pthread_mutex_lock(&pop_mutex);
    printf("Enter the value to push: ");
    scanf("%d",&n);
    top++;
    stack[top]=n;
    pthread_mutex_unlock(&pop_mutex);
    pthread_mutex_unlock(&push_mutex);
    printf("Value is pushed to stack \n");
}

void * pop(void *arg) {
    int k;
    pthread_mutex_lock(&push_mutex);
    sleep(5);
    pthread_mutex_lock(&pop_mutex);
    k=stack[top];
    top--;
    printf("Value popped from stack is %d \n",k);
    pthread_mutex_unlock(&pop_mutex);
    pthread_mutex_unlock(&push_mutex);
}

int main() {
    pthread_t tid1,tid2;
    pthread_create(&tid1,NULL,&push,NULL);
    pthread_create(&tid2,NULL,&pop,NULL);
    printf("Both threads are created\n");
    pthread_join(tid1,NULL);
    pthread_join(tid2,NULL);
    return 0;
}
```

Now, let's go behind the scenes.

# How it works...

We will start by defining a macro called `max` of the value `10`. Then, we will define an array `stack` of the size `max`. We will define two `mutex` objects with the names `pop_mutex` and `push_mutex`.

To use the stack, the value of `top` will be initialized to `-1`. We will define two variables of the type `pthread_t`, with the names `tid1` and `tid2`, to store two thread identifiers.

We will invoke the `pthread_create` function to create the first thread and assign the identifier returned by the function to the variable `tid1`. The thread will be created with the default attributes, and the `push` function will be executed to create this thread.

We will invoke the `pthread_create` function a second time to create the second thread, and we'll assign the identifier returned by the function to the variable `tid2`. The thread will be created with the default attributes and the  `pop` function will be executed to create this thread. On the screen, we will display the message `Both threads are created`.

In the `push` function, the `pthread_mutex_lock` method is invoked, and the `mutex` object `push_mutex` is passed to it to lock it. Now, if any other thread asks for the `pop_mutex` object, it will need to wait until the object is unlocked. After a sleep of `2` seconds, the `mutex` object `pop_mutex` is locked by the first thread.

We will be prompted to enter the value to be pushed to the stack. The entered value will be assigned to the variable `n`. The value of `top` will increment to `0`. The value that we enter will be pushed to the location `stack[0]`. Now, the `pthread_mutex_unlock` will be invoked, and the `mutex` object `pop_mutex` will be passed to it to unlock it. Also, the `mutex` object `push_mutex` will be unlocked. At the bottom of the `push` function, the message `Value is pushed to stack` will be displayed.

In the `pop` function, it will try to lock the `mutex` object `push_mutex`, but because it is already locked by the first thread, this thread will be asked to wait. After a sleep or delay of `5` seconds, it will also try to lock the `pop_mutex` object. The value in the stack, that is, pointed at by the pointer `top` will be popped. Because the value of top is `0`, the value at `stack[0]` is picked up and assigned to the variable `k`. Thereafter, the value of `top` will decrement by `1` to make it `-1` again. The value, that is, popped from the stack will be displayed on the screen. Then, the `mutex` object `pop_mutex` will be unlocked, followed by the `push_mutex` object.

In the `main` function, the `pthread_join` method is invoked twice, and the `tid1` and `tid2` thread identifiers are passed to it. The `pthread_join` is invoked to make the two threads and the `main` method wait until both of the threads have completed their tasks.

Here, we avoided a deadlock because the locking and unlocking of the `mutex` objects was done in a sequence. In the `push` function, the first thread locked the `push_mutex` object and tried to get a lock on the `pop_mutex` object. The `pop_mutex` was kept free because the second thread in the `pop` function first tried to lock the `push_mutex` object, followed by the `pop_mutex` object. Since the first thread had already locked the `push_mutex` object, the second thread was asked to wait. Consequently, both of the `mutex` objects, `push_mutex` and `pop_mutex`, were in an unlocked state, and the first thread was able to easily lock both of the `mutex` objects and use the common resource. After finishing its task, the first thread will unlock both of the `mutex` objects, enabling the second thread to lock both of the `mutex` objects and access the common resource thread.

Let's use GCC to compile the `avoiddeadlockst.c` program, as follows:

```cpp
D:\CBook>gcc avoiddeadlockst.c -o avoiddeadlockst
```

If you get no errors or warnings, that means the `avoiddeadlockst.c` program has been compiled into an executable file, `avoiddeadlockst.exe`. Let's run this executable file:

![](img/b2c30d55-c16f-4f32-adad-e6d9ba978ff7.png)

Voila! We've successfully avoided a deadlock.