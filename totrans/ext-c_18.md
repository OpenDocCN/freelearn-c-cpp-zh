# Chapter 18

# Process Synchronization

This chapter continues our discussion in the previous chapter, *Process Execution*, and our main focus will be on process synchronization. Control mechanisms in multi-process programs are different from the control techniques we met in multi-threaded programs. It is not just the memory which differs; there are other factors that you cannot find in a multi-threaded program, and they exist in a multi-process environment.

Despite threads that are bound to a process, processes can live freely on any machine, with any operating system, located anywhere within a network as big as the internet. As you might imagine, things become complicated. It will not be easy to synchronize a number of processes in such a distributed system.

This chapter is dedicated to process synchronization happening in just one machine. In other words, it mainly talks about single-host synchronization and the techniques around it. We discuss briefly the process synchronization in distributed systems, but we won't go into extensive detail.

This chapter covers the following topics:

*   Firstly, we describe multi-process software where all processes are being run on the same machine. We introduce the techniques that are available in single-host environments. We use the knowledge from the previous chapter in order to give some examples that demonstrate these techniques.
*   In our first attempt to synchronize a number of processes, we use named POSIX semaphores. We explain how they should be used and then we give an example that resolves a race condition issue we encountered in the previous chapters.
*   After that, we talk about named POSIX mutexes and we show how we can use shared memory regions to have named mutexes up and working. As an example, we solve the same race condition resolved by semaphores, this time using named mutexes.
*   As the last technique to synchronize a number of processes, we discuss named POSIX condition variables. Like named mutexes, they need to be put in a shared memory region to become accessible to a number of processes. We give a thorough example regarding this technique which shows how named POSIX condition variables can be used to synchronize a multi-process system.
*   As our final discussion in this chapter, we briefly talk about the multi-process systems which have their own processes distributed around a network. We discuss their features and the problematic differences that they have in comparison to a single-host multi-process system.

Let us start the chapter with talking a bit more about single-host concurrency control and what techniques are available as part of it.

# Single-host concurrency control

It is pretty common to be in situations where there are a number of processes running on a single machine that, at the same time, need to have simultaneous access to a shared resource. Since all of the processes are running within the same operating system, they have access to all the facilities which their operating system provides.

In this section, we show how to use some of these facilities to create a control mechanism that synchronizes the processes. Shared memory plays a key role in most of these control mechanisms; therefore, we heavily rely on what we explained about shared memory in the previous chapter.

The following is a list of POSIX-provided control mechanisms that can be employed while all processes are running on the same POSIX-compliant machine:

*   **Named POSIX semaphores**: The same POSIX semaphores that we explained in *Chapter 16*, *Thread Synchronization*, but with one difference: they have a name now and can be used globally throughout the system. In other words, they are not *anonymous* or *private* semaphores anymore.
*   **Named mutexes**: Again, the same POSIX mutexes with the same properties which were explained in *Chapter 16*, *Thread Synchronization*, but now named and can be used throughout the system. These mutexes should be placed inside a shared memory in order to be available to multiple processes.
*   **Named condition variables**: The same POSIX condition variables which we explained in *Chapter 16*, *Thread Synchronization*, but like mutexes, they should be placed inside a shared memory object in order to be available to a number of processes.

In the upcoming sections, we discuss all the above techniques and give examples to demonstrate how they work. In the following section, we are going to discuss named POSIX semaphores.

# Named POSIX semaphores

As you saw in *Chapter 16*, *Thread Synchronization*, semaphores are the main tool to synchronize a number of concurrent tasks. We saw them in multi-threaded programs and saw how they help to overcome the concurrency issues.

In this section, we are going to show how they can be used among some processes. *Example 18.1* shows how to use a POSIX semaphore to solve the data races we encountered in *examples 17.6* and *17.7* given in the previous chapter, *Process Execution*. The example is remarkably similar to *example 17.6*, and it again uses a shared memory region for storing the shared counter variable. But it uses named semaphores to synchronize the access to the shared counter.

The following code boxes show the way that we use a named semaphore to synchronize two processes while accessing a shared variable. The following code box shows the global declarations of *example 18.1*:

```cpp
#include <stdio.h>
...
#include <semaphore.h>  // For using semaphores
#define SHARED_MEM_SIZE 4
// Shared file descriptor used to refer to the
// shared memory object
int shared_fd = -1;
// The pointer to the shared counter
int32_t* counter = NULL;
// The pointer to the shared semaphore
sem_t* semaphore = NULL;
```

Code Box 18-1 [ExtremeC_examples_chapter18_1.c]: The global declarations of example 18.1

In *Code Box 18-1*, we have declared a global counter and a global pointer to a semaphore object which will be set later. This pointer will be used by both parent and child processes to have synchronized access to the shared counter, addressed by the counter pointer.

The following code shows the function definitions supposed to do the actual process synchronization. Some of the definitions are the same as we had in *example 17.6* and those lines are removed from the following code box:

```cpp
void init_control_mechanism() {
  semaphore = sem_open("/sem0", O_CREAT | O_EXCL, 0600, 1);
  if (semaphore == SEM_FAILED) {
    fprintf(stderr, "ERROR: Opening the semaphore failed: %s\n",
        strerror(errno));
    exit(1);
  }
}
void shutdown_control_mechanism() {
  if (sem_close(semaphore) < 0) {
    fprintf(stderr, "ERROR: Closing the semaphore failed: %s\n",
        strerror(errno));
    exit(1);
  }
  if (sem_unlink("/sem0") < 0) {
    fprintf(stderr, "ERROR: Unlinking failed: %s\n",
        strerror(errno));
    exit(1);
  }
}
void init_shared_resource() {
  ... as in the example 17.6 ...
}
void shutdown_shared_resource() {
  ... as in the example 17.6 ...
}
```

Code Box 18-2 [ExtremeC_examples_chapter18_1.c]: The definition of synchronization functions

We have added two new functions compared to *example 17.6*: `init_control_mechanism` and `shutdown_control_mechanism`. We also made some changes to the `inc_counter` function (shown in *Code Box 18-3*) to use the semaphore and form a critical section inside.

Inside the `init_control_mechanism` and `shutdown_control_mechanism` functions, we are using a similar API to the shared memory API to open, close, and unlink a named semaphore.

The functions `sem_open`, `sem_close`, and `sem_unlink` can be seen as similar to `shm_open`, `shm_close`, and `shm_unlink`. There is one difference and that is that the function `sem_open` returns a semaphore pointer instead of a file descriptor.

Note that the API used for working with the semaphore in this example is the same as we have seen before, so the rest of the code can remain unchanged as with *example 17.6*. In this example, the semaphore is initialized with value `1`, which makes it a mutex. The following code box shows the critical section and how the semaphore is used to synchronize the read and write operations performed on the shared counter:

```cpp
void inc_counter() {
  usleep(1);
  sem_wait(semaphore); // Return value should be checked.
  int32_t temp = *counter;
  usleep(1);
  temp++;
  usleep(1);
  *counter = temp;
  sem_post(semaphore); // Return value should be checked.
  usleep(1);
}
```

Code Box 18-3 [ExtremeC_examples_chapter18_1.c]: The critical section where the shared counter is being incremented

Comparing to *example 17.6*, in the function `inc_counter`, the functions `sem_wait` and `sem_post` are used to enter and exit the critical sections, respectively.

In the following code box, you can see the function `main`. It is almost the same as *example 17.6* and we only see some changes in the initial and final parts, and that is in accordance with the addition of two new functions seen in *Code Box 18-2*:

```cpp
int main(int argc, char** argv) {
  // Parent process needs to initialize the shared resource
  init_shared_resource();
  // Parent process needs to initialize the control mechanism
  init_control_mechanism();
  ... as in the example 17.6 ...
  // Only parent process needs to shut down the shared resource
  // and the employed control mechanism
  if (pid) {
    shutdown_shared_resource();
    shutdown_control_mechanism();
  }
  return 0;
}
```

Code Box 18-4 [ExtremeC_examples_chapter18_1.c]: The main function of example 18.1

In the following shell box, you can see the output for two successive runs of *example 18.1*:

```cpp
$ gcc ExtremeC_examples_chapter18_1.c -lrt -lpthread -o ex18_1.out
$ ./ex18_1.out
Shared memory is created with fd: 3
The memory region is truncated.
The child process sees the counter as 1.
The parent process sees the counter as 2.
The child process finished with status 0.
$ ./ex18_1.out
Shared memory is created with fd: 3
The memory region is truncated.
The parent process sees the counter as 1.
The child process sees the counter as 2.
The child process finished with status 0.
$
```

Shell Box 18-1: Building in Linux and two successive runs of example 18.1

Note that we need to link the above code with the `pthread` library because we are using POSIX semaphores. We need also to link it with the `rt` library in Linux in order to use the shared memories.

The preceding output is clear. Sometimes the child process gets the CPU first and increments the counter, and sometimes the parent process does so. There is no time when both enter the critical section, and therefore they satisfy the data integrity of the shared counter.

Note that it is not required to use the fork API in order to use named semaphores. Completely separated processes, which are not parent and child, can still open and use the same semaphores if they are run on the same machine and inside the same operating system. In *example 18.3*, we show how this is possible.

As the final note in this section, you should know that we have two types of named semaphores in Unix-like operating systems. One is *System V Semaphores*, and the other is *POSIX semaphores*. In this section, we explained the POSIX semaphores because they have a better reputation for their nice API and performance. The following link is a Stack Overflow question which nicely explains the differences between System V semaphores and POSI[X semaphores: https://stackoverflow.com/questions/368322/differences-between-system-v-and-po](https://stackoverflow.com/questions/368322/differences-between-system-v-and-posix-semaphores)six-semaphores.

**Note:**

Microsoft Windows is not POSIX-compliant in terms of using semaphores, and it has its own API to create and manage semaphores.

In the next section, we discuss named mutexes. In short, named mutexes are ordinary mutex objects that are put into a shared memory region.

# Named mutexes

POSIX mutexes work simply in multi-threaded programs; we demonstrated this in *Chapter 16*, *Thread Synchronization*. This would not be the case with regard to multiple process environments, however. To have a mutex work among a number of processes, it would need to be defined within a place that is accessible to all of them.

The best choice for a shared place such as this is a shared memory region. Therefore, to have a mutex that works in a multi-process environment, it should be distributed in a shared memory region.

## The first example

The following example, *example 18.2*, is a clone of *example 18.1*, but it solves the potential race condition using named mutexes instead of named semaphores. It also shows how to make a shared memory region and use it to store a shared mutex.

Since each shared memory object has a global name, a mutex stored in a shared memory region can be considered *named* and can be accessed by other processes throughout the system.

The following code box shows the declarations required for *example 18.2*. It shows what is needed for having a shared mutex:

```cpp
#include <stdio.h>
...
#include <pthread.h> // For using pthread_mutex_* functions
#define SHARED_MEM_SIZE 4
// Shared file descriptor used to refer to shared memory object
int shared_fd = -1;
// Shared file descriptor used to refer to the mutex's shared
// memory object
int mutex_shm_fd = -1;
// The pointer to the shared counter
int32_t* counter = NULL;
// The pointer to shared mutex
pthread_mutex_t* mutex = NULL;
```

Code Box 18-5 [ExtremeC_examples_chapter18_2.c]: The global declarations of example 18.2

As you can see, we have declared:

*   A global file descriptor for pointing to a shared memory region that is meant to store the shared counter variable
*   A global file descriptor for the shared memory region storing the shared mutex
*   A pointer to the shared counter
*   A pointer to the shared mutex

These variables will be populated accordingly by the upcoming logic.

The following code boxes show all the functions we had in *example 18.1*, but as you see, the definitions are updated to work with a named mutex instead of a named semaphore:

```cpp
void init_control_mechanism() {
  // Open the mutex shared memory
  mutex_shm_fd = shm_open("/mutex0", O_CREAT | O_RDWR, 0600);
  if (mutex_shm_fd < 0) {
    fprintf(stderr, "ERROR: Failed to create shared memory: %s\n"
        , strerror(errno));
    exit(1);
  }
  // Allocate and truncate the mutex's shared memory region
  if (ftruncate(mutex_shm_fd, sizeof(pthread_mutex_t)) < 0) {
    fprintf(stderr, "ERROR: Truncation of mutex failed: %s\n",
        strerror(errno));
    exit(1);
  }
  // Map the mutex's shared memory
  void* map = mmap(0, sizeof(pthread_mutex_t),
          PROT_READ | PROT_WRITE, MAP_SHARED, mutex_shm_fd, 0);
  if (map == MAP_FAILED) {
    fprintf(stderr, "ERROR: Mapping failed: %s\n",
            strerror(errno));
    exit(1);
  }
  mutex = (pthread_mutex_t*)map;
  // Initialize the mutex object
  int ret = -1;
  pthread_mutexattr_t attr;
  if ((ret = pthread_mutexattr_init(&attr))) {
    fprintf(stderr, "ERROR: Failed to init mutex attrs: %s\n",
        strerror(ret));
    exit(1);
  }
  if ((ret = pthread_mutexattr_setpshared(&attr,
                  PTHREAD_PROCESS_SHARED))) {
    fprintf(stderr, "ERROR: Failed to set the mutex attr: %s\n",
        strerror(ret));
    exit(1);
  }
  if ((ret = pthread_mutex_init(mutex, &attr))) {
    fprintf(stderr, "ERROR: Initializing the mutex failed: %s\n",
        strerror(ret));
    exit(1);
  }
  if ((ret = pthread_mutexattr_destroy(&attr))) {
    fprintf(stderr, "ERROR: Failed to destroy mutex attrs : %s\n"
        , strerror(ret));
    exit(1);
  }
}
```

Code Box 18-6 [ExtremeC_examples_chapter18_2.c]: The function init_control_mechanism in example 18.2

As part of the function `init_control_mechanism`, we have created a new shared memory object named `/mutex0`. The size of the shared memory region is initialized to `sizeof(pthread_mutex_t)` which shows our intention to share a POSIX mutex object there.

Following that, we get a pointer to the shared memory region. Now we have a mutex which is allocated from the shared memory, but it still needs to be initialized. The next step is therefore to initialize the mutex object using the function `pthread_mutex_init`, with attributes that indicate that the mutex object should be shared and accessible by other processes. This is especially important; otherwise, the mutex does not work in a multi-process environment, even though it is placed inside a shared memory region. As you have seen in the preceding code box and as part of the function `init_control_mechanism`, we have set the attribute `PTHREAD_PROCESS_SHARED` to mark the mutex as shared. Let's look at the next function:

```cpp
void shutdown_control_mechanism() {
  int ret = -1;
  if ((ret = pthread_mutex_destroy(mutex))) {
    fprintf(stderr, "ERROR: Failed to destroy mutex: %s\n",
        strerror(ret));
    exit(1);
  }
  if (munmap(mutex, sizeof(pthread_mutex_t)) < 0) {
    fprintf(stderr, "ERROR: Unmapping the mutex failed: %s\n",
        strerror(errno));
    exit(1);
  }
  if (close(mutex_shm_fd) < 0) {
    fprintf(stderr, "ERROR: Closing the mutex failed: %s\n",
        strerror(errno));
    exit(1);
  }
  if (shm_unlink("/mutex0") < 0) {
    fprintf(stderr, "ERROR: Unlinking the mutex failed: %s\n",
        strerror(errno));
    exit(1);
  }
}
```

Code Box 18-7 [ExtremeC_examples_chapter18_2.c]: The function destroy_control_mechanism in example 18.2

In the function `destroy_control_mechanism` we destroy the mutex object, and after that we close and unlink its underlying shared memory region. This is the same way that we destroy an ordinary shared memory object. Let's continue with other codes in the example:

```cpp
void init_shared_resource() {
  ... as in the example 18.1 ...
}
void shutdown_shared_resource() {
  ... as in the example 18.1 ...
}
```

Code Box 18-8 [ExtremeC_examples_chapter18_2.c]: These functions are the same as we have seen in example 18.1

As you see, the preceding functions are not changed at all and they are the same as we had in *example 18.1*. Let's look at the critical section inside the function `inc_counter` which now uses a named mutex instead of a named semaphore.

```cpp
void inc_counter() {
  usleep(1);
  pthread_mutex_lock(mutex); // Should check the return value.
  int32_t temp = *counter;
  usleep(1);
  temp++;
  usleep(1);
  *counter = temp;
  pthread_mutex_unlock(mutex); // Should check the return value.
  usleep(1);
}
int main(int argc, char** argv) {
  ... as in the example 18.1 ...
}
```

Code Box 18-9 [ExtremeC_examples_chapter18_2.c]: The critical section now uses a named mutex to protect the shared counter

Generally, as you see in the preceding code boxes, a few places are different from *example 18.1*, and we have had to change only three functions greatly. For instance, the function `main` has not changed at all, and it is the same as in *example 18.1*. This is simply because we have used a different control mechanism in comparison to *example 18.1*, and the remaining logic is the same.

As the final note about *Code Box 18-9*, in the function `inc_counter`, we have used the mutex object exactly as we did in a multi-threaded program. The API is the same, and it is designed in a way that mutexes can be used both in multi-threaded and multi-process environments using the same API. This is a great feature of POSIX mutexes because it enables us to use the same written code in both multi-threaded and multi-process environments when consuming these objects – while of course, the initialization and destruction can be different.

The output of the preceding code is very similar to what we observed for *example 18.1*. While the shared counter is protected by a mutex in this example, it was being protected by a semaphore in the previous example. The semaphore used in the previous example was actually a binary semaphore, and as we have explained in *Chapter 16*, *Thread Synchronization*, a binary semaphore can mimic a mutex. Therefore, not much is new in *example 18.2*, apart from replacing the binary semaphore with a mutex.

## The second example

The named shared memories and mutexes can be used throughout the system by any process. It is not mandatory to have a forked process to be able to use these objects. The following example, *example 18.3*, tries to show how we can use a shared mutex and a shared memory to simultaneously terminate a number of processes that are all running at the same time. We expect to have all processes terminated after pressing the key combination `Ctrl` + `C` in only one of them.

Note that the code is going to be provided in multiple steps. The comments related to each step are provided right after it. Let's present the first step.

### Step 1 – Global declarations

In this example, we write a single source file that can be compiled and executed multiple times to create multiple processes. The processes use some shared memory regions to synchronize their execution. One of the processes is elected to be the owner of the shared memory regions and manages their creation and destruction. Other processes just use the created shared memories.

The first step is going to declare some global objects that we need throughout the code. We will initialize them later on in the code. Note that the global variables defined in the following code box, such as `mutex`, are not actually shared between the processes. They have these variables in their own memory space but each of the processes maps their own global variable to the objects or variables located in the various shared memory regions:

```cpp
#include <stdio.h>
...
#include <pthread.h> // For using pthread_mutex_* functions
typedef uint16_t bool_t;
#define TRUE 1
#define FALSE 0
#define MUTEX_SHM_NAME "/mutex0"
#define SHM_NAME "/shm0"
// Shared file descriptor used to refer to the shared memory
// object containing the cancel flag
int cancel_flag_shm_fd = -1;
// A flag which indicates whether the current process owns the
// shared memory object
bool_t cancel_flag_shm_owner = FALSE;
// Shared file descriptor used to refer to the mutex's shared
// memory object
int mutex_shm_fd = -1;
// The shared mutex
pthread_mutex_t* mutex = NULL;
// A flag which indicates whether the current process owns the
// shared memory object
bool_t mutex_owner = FALSE;
// The pointer to the cancel flag stored in the shared memory
bool_t* cancel_flag = NULL;
```

Code Box 18-10 [ExtremeC_examples_chapter18_3.c]: The global declaration in example 18.3

In the preceding code, we can see the global declarations used in the code. We are going to use a shared flag to let the processes know about the cancellation signal. Note that, in this example, we are going to take the busy-wait approach to wait for the cancellation flag to become `true`.

We have a dedicated shared memory object for the cancellation flag and another shared memory object for the mutex protecting the flag as we did in *example 18.2*. Note that we could construct a single structure and define both the cancellation flag and the mutex object as its fields, and then use a single shared memory region to store them. But we have chosen to use separate shared memory regions to fulfill our purpose.

In this example, one important note about the shared memory objects is that the cleanup should be performed by the process which has created and initialized them in the first place. Since all processes are using the same code, somehow, we need to know which process has created a certain shared memory object and make that process the owner of that object. Then, while cleaning up the objects, only the owner process can proceed and do the actual cleanup. Therefore, we had to declare two Boolean variables for this purpose: `mutex_owner` and `cancel_flag_shm_owner`.

### Step 2 – Cancellation flag's shared memory

The following code box shows the initialization of the shared memory region dedicated to the cancellation flag:

```cpp
void init_shared_resource() {
  // Open the shared memory object
  cancel_flag_shm_fd = shm_open(SHM_NAME, O_RDWR, 0600);
  if (cancel_flag_shm_fd >= 0) {
    cancel_flag_shm_owner = FALSE;
    fprintf(stdout, "The shared memory object is opened.\n");
  } else if (errno == ENOENT) {
    fprintf(stderr,
            "WARN: The shared memory object doesn't exist.\n");
    fprintf(stdout, "Creating the shared memory object ...\n");
    cancel_flag_shm_fd = shm_open(SHM_NAME,
            O_CREAT | O_EXCL | O_RDWR, 0600);
    if (cancel_flag_shm_fd >= 0) {
      cancel_flag_shm_owner = TRUE;
      fprintf(stdout, "The shared memory object is created.\n");
    } else {
      fprintf(stderr,
          "ERROR: Failed to create shared memory: %s\n",
          strerror(errno));
      exit(1);
    }
  } else {
      fprintf(stderr,
          "ERROR: Failed to create shared memory: %s\n",
          strerror(errno));
    exit(1);
  }
  if (cancel_flag_shm_owner) {
    // Allocate and truncate the shared memory region
    if (ftruncate(cancel_flag_shm_fd, sizeof(bool_t)) < 0) {
      fprintf(stderr, "ERROR: Truncation failed: %s\n",
              strerror(errno));
      exit(1);
    }
    fprintf(stdout, "The memory region is truncated.\n");
  }
  // Map the shared memory and initialize the cancel flag
  void* map = mmap(0, sizeof(bool_t), PROT_WRITE, MAP_SHARED,
      cancel_flag_shm_fd, 0);
  if (map == MAP_FAILED) {
    fprintf(stderr, "ERROR: Mapping failed: %s\n",
            strerror(errno));
    exit(1);
  }
  cancel_flag = (bool_t*)map;
  if (cancel_flag_shm_owner) {
    *cancel_flag = FALSE;
  }
}
```

Code Box 18-11 [ExtremeC_examples_chapter18_3.c]: Initialization of the cancellation flag's shared memory

The approach we took is different from what we did in *example 18.2*. That's because whenever a new process is run, it should check whether the shared memory object has already been created by another process. Note that we are not using the fork API to create new processes as part of this example and the user can use their shell and start a new process at will.

For this reason, a new process first tries to open the shared memory region by only providing the flag `O_RDWR`. If it succeeds, then it's a sign that the current process is not the owner of that region, and it proceeds with mapping the shared memory region. If it fails, it means that the shared memory region does not exist, and it is an indication that the current process should create the region and becomes its owner. So, it proceeds and tries to open the region with different flags; `O_CREAT` and `O_EXCL`. These flags create a shared memory object if it does not exist.

If the creation succeeds, the current process is the owner, and it continues by truncating and mapping the shared memory region.

There is a small chance that between the two successive calls of the `shm_open` function in the previous scenario, another process creates the same shared memory region, and therefore the second `shm_open` fails. The flag `O_EXCL` prevents the current process from creating an object which already exists, and then it quits by showing a proper error message. If this happens, which should be very rare, we can always try to run the process again and it won't face the same issue in the second run.

The following code is the reverse operation for destructing the cancellation flag and its shared memory region:

```cpp
void shutdown_shared_resource() {
  if (munmap(cancel_flag, sizeof(bool_t)) < 0) {
    fprintf(stderr, "ERROR: Unmapping failed: %s\n",
            strerror(errno));
    exit(1);
  }
  if (close(cancel_flag_shm_fd) < 0) {
    fprintf(stderr,
        "ERROR: Closing the shared memory fd filed: %s\n",
        strerror(errno));
    exit(1);
  }
  if (cancel_flag_shm_owner) {
    sleep(1);
    if (shm_unlink(SHM_NAME) < 0) {
      fprintf(stderr,
          "ERROR: Unlinking the shared memory failed: %s\n",
          strerror(errno));
      exit(1);
    }
  }
}
```

Code Box 18-12 [ExtremeC_examples_chapter18_3.c]: Closing the resources allocated for the cancellation flag's shared memory

As you can see in *Code Box 18-12*, the written logic is very similar to what we've seen so far, as part of previous examples, about releasing a shared memory object. But there is a difference here and it is the fact that only the owner process can unlink the shared memory object. Note that the owner process waits for 1 second before unlinking the shared memory object, in order to let other processes finalize their resources. This wait is not usually necessary due to the fact that, in most POSIX-compliant systems, the shared memory object remains in place until all depending processes quit.

### Step 3 – Named mutex's shared memory

The following code box shows how to initialize the shared mutex and its associated shared memory object:

```cpp
void init_control_mechanism() {
  // Open the mutex shared memory
  mutex_shm_fd = shm_open(MUTEX_SHM_NAME, O_RDWR, 0600);
  if (mutex_shm_fd >= 0) {
    // The mutex's shared object exists and I'm now the owner.
    mutex_owner = FALSE;
    fprintf(stdout,
            "The mutex's shared memory object is opened.\n");
  } else if (errno == ENOENT) {
    fprintf(stderr,
            "WARN: Mutex's shared memory doesn't exist.\n");
    fprintf(stdout,
            "Creating the mutex's shared memory object ...\n");
    mutex_shm_fd = shm_open(MUTEX_SHM_NAME,
            O_CREAT | O_EXCL | O_RDWR, 0600);
    if (mutex_shm_fd >= 0) {
      mutex_owner = TRUE;
      fprintf(stdout,
              "The mutex's shared memory object is created.\n");
    } else {
      fprintf(stderr,
          "ERROR: Failed to create mutex's shared memory: %s\n",
          strerror(errno));
      exit(1);
    }
  } else {
    fprintf(stderr,
        "ERROR: Failed to create mutex's shared memory: %s\n",
        strerror(errno));
    exit(1);
  }
  if (mutex_owner) {
    // Allocate and truncate the mutex's shared memory region
  }
  if (mutex_owner) {
    // Allocate and truncate the mutex's shared memory region
    if (ftruncate(mutex_shm_fd, sizeof(pthread_mutex_t)) < 0) {
      fprintf(stderr,
          "ERROR: Truncation of the mutex failed: %s\n",
          strerror(errno));
      exit(1);
    }
  }
  // Map the mutex's shared memory
  void* map = mmap(0, sizeof(pthread_mutex_t),
          PROT_READ | PROT_WRITE, MAP_SHARED, mutex_shm_fd, 0);
  if (map == MAP_FAILED) {
    fprintf(stderr, "ERROR: Mapping failed: %s\n",
            strerror(errno));
    exit(1);
  }
  mutex = (pthread_mutex_t*)map;
  if (mutex_owner) {
    int ret = -1;
    pthread_mutexattr_t attr;
    if ((ret = pthread_mutexattr_init(&attr))) {
      fprintf(stderr,
          "ERROR: Initializing mutex attributes failed: %s\n",
          strerror(ret));
      exit(1);
    }
    if ((ret = pthread_mutexattr_setpshared(&attr,
                    PTHREAD_PROCESS_SHARED))) {
      fprintf(stderr,
          "ERROR: Setting the mutex attribute failed: %s\n",
          strerror(ret));
      exit(1);
    }
    if ((ret = pthread_mutex_init(mutex, &attr))) {
      fprintf(stderr,
          "ERROR: Initializing the mutex failed: %s\n",
          strerror(ret));
      exit(1);
    }
    if ((ret = pthread_mutexattr_destroy(&attr))) {
      fprintf(stderr,
          "ERROR: Destruction of mutex attributes failed: %s\n",
          strerror(ret));
      exit(1);
    }
  }
}
```

Code Box 18-13 [ExtremeC_examples_chapter18_3.c]: Initializing the shared mutex and its underlying shared memory region

Similarly to what we did while trying to create the shared memory region associated with the cancellation flag, we have done the same thing to create and initialize the shared memory region beneath the shared mutex. Note that, just like in *example 18.2*, the mutex has been marked as `PTHREAD_PROCESS_SHARED`, which allows it to be used by multiple processes.

The following code box shows how to finalize the shared mutex:

```cpp
void shutdown_control_mechanism() {
  sleep(1);
  if (mutex_owner) {
    int ret = -1;
    if ((ret = pthread_mutex_destroy(mutex))) {
      fprintf(stderr,
          "WARN: Destruction of the mutex failed: %s\n",
          strerror(ret));
    }
  }
  if (munmap(mutex, sizeof(pthread_mutex_t)) < 0) {
    fprintf(stderr, "ERROR: Unmapping the mutex failed: %s\n",
        strerror(errno));
    exit(1);
  }
  if (close(mutex_shm_fd) < 0) {
    fprintf(stderr, "ERROR: Closing the mutex failed: %s\n",
        strerror(errno));
    exit(1);
  }
  if (mutex_owner) {
    if (shm_unlink(MUTEX_SHM_NAME) < 0) {
      fprintf(stderr, "ERROR: Unlinking the mutex failed: %s\n",
          strerror(errno));
      exit(1);
    }
  }
}
```

Code Box 18-14 [ExtremeC_examples_chapter18_3.c]: Closing the shared mutex and its associated shared memory region

Again, the owner process can only unlink the shared memory object of the shared mutex.

### Step 4 – Setting the cancellation flag

The following code box shows the functions which allow the processes to read or set the cancellation flag:

```cpp
bool_t is_canceled() {
  pthread_mutex_lock(mutex); // Should check the return value
  bool_t temp = *cancel_flag;
  pthread_mutex_unlock(mutex); // Should check the return value
  return temp;
}
void cancel() {
  pthread_mutex_lock(mutex); // Should check the return value
  *cancel_flag = TRUE;
  pthread_mutex_unlock(mutex); // Should check the return value
}
```

Code Box 18-15 [ExtremeC_examples_chapter18_3.c]: The synchronized functions that read and set the cancellation flag protected by the shared mutex

The preceding two functions allow us to have synchronized access to the shared cancellation flag. The function `is_canceled` is used to check the value of the flag, and the function `cancel` is used to set the flag. As you see, both are protected by the same shared mutex.

### Step 5 – The main function

And finally, the following code box shows the `main` function and a *signal handler* which we explain shortly:

```cpp
void sigint_handler(int signo) {
  fprintf(stdout, "\nHandling INT signal: %d ...\n", signo);
  cancel();
}
int main(int argc, char** argv) {
  signal(SIGINT, sigint_handler);
  // Parent process needs to initialize the shared resource
  init_shared_resource();
  // Parent process needs to initialize the control mechanism
  init_control_mechanism();
  while(!is_canceled()) {
    fprintf(stdout, "Working ...\n");
    sleep(1);
  }
  fprintf(stdout, "Cancel signal is received.\n");
  shutdown_shared_resource();
  shutdown_control_mechanism();
  return 0;
}
```

Code Box 18-16 [ExtremeC_examples_chapter18_3.c]: The function main and the signal handler function as part of example 18.3

As you see, the logic inside the `main` function is clear and straightforward. It initializes the shared flag and mutex and then goes into a busy-wait until the cancellation flag becomes `true`. Finally, it shuts down all shared resources and terminates.

One thing which is new here is the usage of the `signal` function which assigns a signal handler to a specific set of *signals*. Signals are one of the facilities provided by all POSIX-compliant operating systems and using it, the processes within the system can send signals to each other. The *terminal* is just one normal process that the user interacts with and it can be used to send signals to other processes. Pressing `Ctrl` + `C` is one convenient way to send `SIGINT` to the foreground process running in a terminal.

`SIGINT` is the *interrupt signal* which can be received by a process. In the preceding code, we assign the function `sigint_handler` to be the handler of the `SIGINT` signal. In other words, whenever the signal `SIGINT` is received by the process, the function `sigint_handler` will be called. If the signal `SIGINT` is not handled, the default routine is to terminate the process, but this can be overridden using signal handlers like above.

There are many ways to send a `SIGINT` signal to a process, but one of the easiest is to press the `Ctrl` + `C` keys on the keyboard. The process will immediately receive the `SIGINT` signal. As you see, within the signal handler, we set the shared cancellation flag to `true`, and after this point, all the processes start to exit their busy-wait loops.

Following is a demonstration of how the preceding code compiles and works. Let's build the preceding code and run the first process:

```cpp
$ gcc ExtremeC_examples_chapter18_3.c -lpthread -lrt -o ex18_3.out
$ ./ex18_3.out
WARN: The shared memory object doesn't exist.
Creating a shared memory object ...
The shared memory object is created.
The memory region is truncated.
WARN: Mutex's shared memory object doesn't exist.
Creating the mutex's shared memory object ...
The mutex's shared memory object is created.
Working ...
Working ...
Working ...
```

Shell Box 18-2: Compilation of example 18.3 and running the first process

As you see, the preceding process is the first to be run, and therefore, it is the owner of the mutex and cancellation flag. The following is the run of the second process:

```cpp
$ ./ex18_3.out
The shared memory object is opened.
The mutex's shared memory object is opened.
Working ...
Working ...
Working ...
```

Shell Box 18-3: Running the second process

As you see, the second process only opens the shared memory objects, and it is not the owner. The following output is showing when `Ctrl` + `C` has been pressed on the first process:

```cpp
...
Working ...
Working ...
^C
Handling INT signal: 2 ...
Cancel signal is received.
$
```

Shell Box 18-4: The output of the first process when Ctrl + C has been pressed

As you see, the first process prints that it is handling a signal with the number `2` which is the standard signal number of the `SIGINT`. It sets the cancellation flag, and it exits immediately. And following it, the second process exits. The following is the output of the second process:

```cpp
...
Working ...
Working ...
Working ...
Cancel signal is received.
$
```

Shell Box 18-5: The output of the second process when it sees that the cancellation flag is set

Also, you can send `SIGINT` to the second process and the result will be the same; both processes will get the signal and will quit. Also, you can create more than two processes and all of them will synchronously quit using the same shared memory and mutex.

In the next section, we demonstrate how to use condition variables. Like named mutexes, if you place a condition variable inside a shared memory region, it can be accessed and used by multiple processes using the shared memory's name.

# Named condition variables

As we explained before, similar to named POSIX mutexes, we need to allocate a POSIX condition variable from a shared memory region in order to use it in a multi-processing system. The following example, *example 18.4*, shows how to do so in order to make a number of processes count in a specific order. As you know from *Chapter 16*, *Thread Synchronization*, every condition variable should be used together with a companion mutex object which protects it. Therefore, we will have three shared memory regions in *example 18.4*; one for the shared counter, one for the shared *named condition variable*, and one for the shared *named mutex* protecting the shared condition variable.

Note that instead of having three different shared memories, we could also use a single shared memory. This is possible by defining a structure that encompasses all the required objects. In this example, we are not going to take this approach and we will define a separate shared memory region for each object.

*Example 18.4* is about a number of processes which should count in an ascending order. Each process is given a number, starting from 1 and up to the number of processes, and the given number indicates the process's rank within the other processes. The process must wait for the other processes with smaller numbers (ranks) to count first and then it can count its turn and exit. Of course, the process assigned the number 1 counts first, even if it is the latest spawned process.

Since we are going to have three different shared memory regions, each of which requiring its own steps to get initialized and finalized, we would have a lot of code duplication if we wanted to take the same approach as we have so far in the previous examples. For reducing the amount of code that we write, and factoring out the duplications into some functions, and having a better-organized code, we are going to make it object-oriented according to the topics and procedures discussed in *Chapter 6*, *OOP and Encapsulation*, *Chapter 7*, *Composition and Aggregation*, and *Chapter 8*, *Inheritance and Polymorphism*. We are going to write *example 18.4* in an object-oriented manner and use inheritance to reduce the amount of duplicated code.

We will define a parent class for all classes which need to be built upon a shared memory region. Therefore, while having the parent shared memory class, there will be one child class defined for the shared counter, one child class for the shared named mutex, and another child class for the shared named condition variable. Each class will have its own pair of header and source files, and all of them will be used finally in the main function of the example.

The following sections go through the mentioned classes one by one. First of all, let's being with the parent class: shared memory.

## Step 1 – Class of shared memory

The following code box shows the declarations of the shared memory class:

```cpp
struct shared_mem_t;
typedef int32_t bool_t; 
struct shared_mem_t* shared_mem_new();
void shared_mem_delete(struct shared_mem_t* obj);
void shared_mem_ctor(struct shared_mem_t* obj,
                     const char* name,
                     size_t size);
void shared_mem_dtor(struct shared_mem_t* obj);
char* shared_mem_getptr(struct shared_mem_t* obj);
bool_t shared_mem_isowner(struct shared_mem_t* obj);
void shared_mem_setowner(struct shared_mem_t* obj, bool_t is_owner);
```

Code Box 18-17 [ExtremeC_examples_chapter18_4_shared_mem.h]: The public interface of the shared memory class

The preceding code contains the declarations (public API) needed to use a shared memory object. The functions `shared_mem_getptr`, `shared_mem_isowner`, and `shared_mem_setowner` are the behaviors of this class.

If this syntax is not familiar to you, please have a read of *Chapter 6*, *OOP and Encapsulation*, *Chapter 7*, *Composition and Aggregation*, and *Chapter 8*, *Inheritance and Polymorphism*.

The following code box shows the definitions of the functions declared as part of the public interface of the class, as seen in *Code Box 18-17*:

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#define TRUE 1
#define FALSE 0
typedef int32_t bool_t;
bool_t owner_process_set = FALSE;
bool_t owner_process = FALSE;
typedef struct {
  char* name;
  int shm_fd;
  void* map_ptr;
  char* ptr;
  size_t size;
} shared_mem_t;
shared_mem_t* shared_mem_new() {
  return (shared_mem_t*)malloc(sizeof(shared_mem_t));
}
void shared_mem_delete(shared_mem_t* obj) {
  free(obj->name);
  free(obj);
}
void shared_mem_ctor(shared_mem_t* obj, const char* name,
        size_t size) {
  obj->size = size;
  obj->name = (char*)malloc(strlen(name) + 1);
  strcpy(obj->name, name);
  obj->shm_fd = shm_open(obj->name, O_RDWR, 0600);
  if (obj->shm_fd >= 0) {
    if (!owner_process_set) {
      owner_process = FALSE;
      owner_process_set = TRUE;
    }
    printf("The shared memory %s is opened.\n", obj->name);
  } else if (errno == ENOENT) {
    printf("WARN: The shared memory %s does not exist.\n",
            obj->name);
    obj->shm_fd = shm_open(obj->name,
            O_CREAT | O_RDWR, 0600);
    if (obj->shm_fd >= 0) {
      if (!owner_process_set) {
        owner_process = TRUE;
        owner_process_set = TRUE;
      }
      printf("The shared memory %s is created and opened.\n",
              obj->name);
      if (ftruncate(obj->shm_fd, obj->size) < 0) {
        fprintf(stderr, "ERROR(%s): Truncation failed: %s\n",
            obj->name, strerror(errno));
        exit(1);
      }
    } else {
      fprintf(stderr,
          "ERROR(%s): Failed to create shared memory: %s\n",
          obj->name, strerror(errno));
      exit(1);
    }
  } else {
      fprintf(stderr,
          "ERROR(%s): Failed to create shared memory: %s\n",
          obj->name, strerror(errno));
    exit(1);
  }
  obj->map_ptr = mmap(0, obj->size, PROT_READ | PROT_WRITE,
      MAP_SHARED, obj->shm_fd, 0);
  if (obj->map_ptr == MAP_FAILED) {
    fprintf(stderr, "ERROR(%s): Mapping failed: %s\n",
        name, strerror(errno));
    exit(1);
  }
  obj->ptr = (char*)obj->map_ptr;
}
void shared_mem_dtor(shared_mem_t* obj) {
  if (munmap(obj->map_ptr, obj->size) < 0) {
    fprintf(stderr, "ERROR(%s): Unmapping failed: %s\n",
        obj->name, strerror(errno));
    exit(1);
  }
  printf("The shared memory %s is unmapped.\n", obj->name);
  if (close(obj->shm_fd) < 0) {
    fprintf(stderr,
        "ERROR(%s): Closing the shared memory fd failed: %s\n",
        obj->name, strerror(errno));
    exit(1);
  }
  printf("The shared memory %s is closed.\n", obj->name);
  if (owner_process) {
    if (shm_unlink(obj->name) < 0) {
      fprintf(stderr,
          "ERROR(%s): Unlinking the shared memory failed: %s\n",
          obj->name, strerror(errno));
      exit(1);
    }
    printf("The shared memory %s is deleted.\n", obj->name);
  }
}
char* shared_mem_getptr(shared_mem_t* obj) {
  return obj->ptr;
}
bool_t shared_mem_isowner(shared_mem_t* obj) {
  return owner_process;
}
void shared_mem_setowner(shared_mem_t* obj, bool_t is_owner) {
    owner_process = is_owner;
}
```

Code Box 18-18 [ExtremeC_examples_chapter18_4_shared_mem.c]: The definitions of all functions found in the shared memory class

As you see, we have just copied the code we wrote for shared memories as part of the previous examples. The structure `shared_mem_t` encapsulates all we need to address a POSIX shared memory object. Note the global Boolean variable `process_owner`. It indicates whether the current process is the owner of all shared memory regions. It is set only once.

## Step 2 – Class of shared 32-bit integer counter

The following code box contains the declaration of the shared counter class which is a 32-bit integer counter. This class inherits from the shared memory class. As you might have noticed, we are using the second approach we described as part of *Chapter 8*, *Inheritance and Polymorphism*, to implement the inheritance relationship:

```cpp
struct shared_int32_t;
struct shared_int32_t* shared_int32_new();
void shared_int32_delete(struct shared_int32_t* obj);
void shared_int32_ctor(struct shared_int32_t* obj,
                       const char* name);
void shared_int32_dtor(struct shared_int32_t* obj);
void shared_int32_setvalue(struct shared_int32_t* obj,
                           int32_t value);
void shared_int32_setvalue_ifowner(struct shared_int32_t* obj,
                                   int32_t value);
int32_t shared_int32_getvalue(struct shared_int32_t* obj);
```

Code Box 18-19 [ExtremeC_examples_chapter18_4_shared_int32.h]: The public interface of the shared counter class

And the following code box shows the implementations of the preceding declared functions:

```cpp
#include "ExtremeC_examples_chapter18_4_shared_mem.h"
typedef struct {
  struct shared_mem_t* shm;
  int32_t* ptr;
} shared_int32_t;
shared_int32_t* shared_int32_new(const char* name) {
  shared_int32_t* obj =
      (shared_int32_t*)malloc(sizeof(shared_int32_t));
  obj->shm = shared_mem_new();
  return obj;
}
void shared_int32_delete(shared_int32_t* obj) {
  shared_mem_delete(obj->shm);
  free(obj);
}
void shared_int32_ctor(shared_int32_t* obj, const char* name) {
  shared_mem_ctor(obj->shm, name, sizeof(int32_t));
  obj->ptr = (int32_t*)shared_mem_getptr(obj->shm);
}
void shared_int32_dtor(shared_int32_t* obj) {
  shared_mem_dtor(obj->shm);
}
void shared_int32_setvalue(shared_int32_t* obj, int32_t value) {
  *(obj->ptr) = value;
}
void shared_int32_setvalue_ifowner(shared_int32_t* obj,
                                   int32_t value) {
  if (shared_mem_isowner(obj->shm)) {
    *(obj->ptr) = value;
  }
}
int32_t shared_int32_getvalue(shared_int32_t* obj) {
  return *(obj->ptr);
}
```

Code Box 18-20 [ExtremeC_examples_chapter18_4_shared_int32.c]: The definitions of all functions found in the shared counter class

As you can see, we have written a lot less code thanks to inheritance. All the necessary code for managing the associated shared memory object has been brought in by the field `shm` in the structure `shared_int32_t`.

## Step 3 – Class of shared mutex

The following code box contains the declaration of the shared mutex class:

```cpp
#include <pthread.h>
struct shared_mutex_t;
struct shared_mutex_t* shared_mutex_new();
void shared_mutex_delete(struct shared_mutex_t* obj);
void shared_mutex_ctor(struct shared_mutex_t* obj,
                       const char* name);
void shared_mutex_dtor(struct shared_mutex_t* obj);
pthread_mutex_t* shared_mutex_getptr(struct shared_mutex_t* obj);
void shared_mutex_lock(struct shared_mutex_t* obj);
void shared_mutex_unlock(struct shared_mutex_t* obj);
#if !defined(__APPLE__)
void shared_mutex_make_consistent(struct shared_mutex_t* obj);
#endif
```

Code Box 18-21 [ExtremeC_examples_chapter18_4_shared_mutex.h]: The public interface of the shared mutex class

As you see, the above class has three exposed behaviors as expected; `shared_mutex_lock`, `shared_mutex_unlock`, and `shared_mutex_make_consistent`. But there is one exception, which is that the behavior `shared_mutex_make_consistent` is only available in POSIX systems which are not macOS (Apple) based. That's because *robust mutexes* are not supported by Apple systems. We will discuss what a robust mutex is in the upcoming paragraphs. Note that we have used the macro `__APPLE__` to detect whether we are compiling on an Apple system or not.

The following code box shows the implementation of the preceding class:

```cpp
#include "ExtremeC_examples_chapter18_4_shared_mem.h"
typedef struct {
  struct shared_mem_t* shm;
  pthread_mutex_t* ptr;
} shared_mutex_t;
shared_mutex_t* shared_mutex_new() {
  shared_mutex_t* obj =
      (shared_mutex_t*)malloc(sizeof(shared_mutex_t));
  obj->shm = shared_mem_new();
  return obj;
}
void shared_mutex_delete(shared_mutex_t* obj) {
  shared_mem_delete(obj->shm);
  free(obj);
}
void shared_mutex_ctor(shared_mutex_t* obj, const char* name) {
  shared_mem_ctor(obj->shm, name, sizeof(pthread_mutex_t));
  obj->ptr = (pthread_mutex_t*)shared_mem_getptr(obj->shm);
  if (shared_mem_isowner(obj->shm)) {
    pthread_mutexattr_t mutex_attr;
    int ret = -1;
    if ((ret = pthread_mutexattr_init(&mutex_attr))) {
      fprintf(stderr,
          "ERROR(%s): Initializing mutex attrs failed: %s\n",
          name, strerror(ret));
      exit(1);
    }
#if !defined(__APPLE__)
    if ((ret = pthread_mutexattr_setrobust(&mutex_attr,
                    PTHREAD_MUTEX_ROBUST))) {
      fprintf(stderr,
          "ERROR(%s): Setting the mutex as robust failed: %s\n",
          name, strerror(ret));
      exit(1);
    }
#endif
    if ((ret = pthread_mutexattr_setpshared(&mutex_attr,
                    PTHREAD_PROCESS_SHARED))) {
      fprintf(stderr,
          "ERROR(%s): Failed to set as process-shared: %s\n",
          name, strerror(ret));
      exit(1);
    }
    if ((ret = pthread_mutex_init(obj->ptr, &mutex_attr))) {
      fprintf(stderr,
          "ERROR(%s): Initializing the mutex failed: %s\n",
          name, strerror(ret));
      exit(1);
    }
    if ((ret = pthread_mutexattr_destroy(&mutex_attr))) {
      fprintf(stderr,
          "ERROR(%s): Destruction of mutex attrs failed: %s\n",
          name, strerror(ret));
      exit(1);
    }
  }
}
void shared_mutex_dtor(shared_mutex_t* obj) {
  if (shared_mem_isowner(obj->shm)) {
    int ret = -1;
    if ((ret = pthread_mutex_destroy(obj->ptr))) {
      fprintf(stderr,
          "WARN: Destruction of the mutex failed: %s\n",
          strerror(ret));
    }
  }
  shared_mem_dtor(obj->shm);
}
pthread_mutex_t* shared_mutex_getptr(shared_mutex_t* obj) {
  return obj->ptr;
}
#if !defined(__APPLE__)
void shared_mutex_make_consistent(shared_mutex_t* obj) {
  int ret = -1;
  if ((ret = pthread_mutex_consistent(obj->ptr))) {
    fprintf(stderr,
        "ERROR: Making the mutex consistent failed: %s\n",
        strerror(ret));
    exit(1);
  }
}
#endif
void shared_mutex_lock(shared_mutex_t* obj) {
  int ret = -1;
  if ((ret = pthread_mutex_lock(obj->ptr))) {
#if !defined(__APPLE__)
    if (ret == EOWNERDEAD) {
        fprintf(stderr,
                "WARN: The owner of the mutex is dead ...\n");
        shared_mutex_make_consistent(obj);
        fprintf(stdout, "INFO: I'm the new owner!\n");
        shared_mem_setowner(obj->shm, TRUE);
        return;
    }
#endif
    fprintf(stderr, "ERROR: Locking the mutex failed: %s\n",
        strerror(ret));
    exit(1);
  }
}
void shared_mutex_unlock(shared_mutex_t* obj) {
  int ret = -1;
  if ((ret = pthread_mutex_unlock(obj->ptr))) {
    fprintf(stderr, "ERROR: Unlocking the mutex failed: %s\n",
        strerror(ret));
    exit(1);
  }
}
```

Code Box 18-22 [ExtremeC_examples_chapter18_4_shared_mutex.c]: The definitions of all functions found in the shared named mutex class

In the preceding code, we do only the POSIX mutex initialization, finalization, and exposing some of the trivial behaviors such as locking and unlocking. Everything else regarding the shared memory object is being handled in the shared memory class. That's a benefit of using inheritance.

Note that in the constructor function `shared_mutex_ctor`, we set the mutex as a *shared process* mutex to be accessible to all processes. This is absolutely necessary to multi-process software. Note that in systems which are not Apple-based, we go further and configure the mutex as a *robust mutex*.

For an ordinary mutex that is locked by a process, if the process should suddenly die then the mutex goes into a non-consistent state. For a robust mutex, if this happens, the mutex can be put back in a consistent state. The next process, which is usually waiting for the mutex, can lock the mutex only by making it consistent. You can see how it can be done in the function `shared_mutex_lock`. Note that this functionality is not present in Apple systems.

## Step 4 – Class of shared condition variable

The following code box shows the declaration of the shared condition variable class:

```cpp
struct shared_cond_t;
struct shared_mutex_t;
struct shared_cond_t* shared_cond_new();
void shared_cond_delete(struct shared_cond_t* obj);
void shared_cond_ctor(struct shared_cond_t* obj,
                      const char* name);
void shared_cond_dtor(struct shared_cond_t* obj);
void shared_cond_wait(struct shared_cond_t* obj,
                      struct shared_mutex_t* mutex);
void shared_cond_timedwait(struct shared_cond_t* obj,
                           struct shared_mutex_t* mutex,
                           long int time_nanosec);
void shared_cond_broadcast(struct shared_cond_t* obj);
```

Code Box 18-23 [ExtremeC_examples_chapter18_4_shared_cond.h]: The public interface of the shared condition variable class

Three behaviors are exposed; `shared_cond_wait`, `shared_cond_timedwait`, and `shared_cond_broadcast`. If you remember from *Chapter 16*, *Thread Synchronization*, the behavior `shared_cond_wait` waits for a signal on a condition variable.

Above, we have added a new version of waiting behavior; `shared_cond_timedwait`. It waits for the signal for a specified amount of time and then it gets timed out if the condition variable doesn't receive a signal. On the other hand, the `shared_cond_wait` never exists until it receives some sort of signal. We will use the timed version of waiting in *example 18.4*. Note that both waiting behavior functions receive a pointer to the companion shared mutex just like what we saw in multi-threaded environments.

The following code box contains the actual implementation of the shared condition variable class:

```cpp
#include "ExtremeC_examples_chapter18_4_shared_mem.h"
#include "ExtremeC_examples_chapter18_4_shared_mutex.h"
typedef struct {
  struct shared_mem_t* shm;
  pthread_cond_t* ptr;
} shared_cond_t;
shared_cond_t* shared_cond_new() {
  shared_cond_t* obj =
      (shared_cond_t*)malloc(sizeof(shared_cond_t));
  obj->shm = shared_mem_new();
  return obj;
}
void shared_cond_delete(shared_cond_t* obj) {
  shared_mem_delete(obj->shm);
  free(obj);
}
void shared_cond_ctor(shared_cond_t* obj, const char* name) {
  shared_mem_ctor(obj->shm, name, sizeof(pthread_cond_t));
  obj->ptr = (pthread_cond_t*)shared_mem_getptr(obj->shm);
  if (shared_mem_isowner(obj->shm)) {
    pthread_condattr_t cond_attr;
    int ret = -1;
    if ((ret = pthread_condattr_init(&cond_attr))) {
      fprintf(stderr,
          "ERROR(%s): Initializing cv attrs failed: %s\n",
          name, strerror(ret));
      exit(1);
    }
    if ((ret = pthread_condattr_setpshared(&cond_attr,
                    PTHREAD_PROCESS_SHARED))) {
      fprintf(stderr,
          "ERROR(%s): Setting as process shared failed: %s\n",
          name, strerror(ret));
      exit(1);
    }
    if ((ret = pthread_cond_init(obj->ptr, &cond_attr))) {
      fprintf(stderr,
          "ERROR(%s): Initializing the cv failed: %s\n",
          name, strerror(ret));
      exit(1);
    }
    if ((ret = pthread_condattr_destroy(&cond_attr))) {
      fprintf(stderr,
          "ERROR(%s): Destruction of cond attrs failed: %s\n",
          name, strerror(ret));
      exit(1);
    }
  }
}
void shared_cond_dtor(shared_cond_t* obj) {
  if (shared_mem_isowner(obj->shm)) {
    int ret = -1;
    if ((ret = pthread_cond_destroy(obj->ptr))) {
      fprintf(stderr, "WARN: Destruction of the cv failed: %s\n",
          strerror(ret));
    }
  }
  shared_mem_dtor(obj->shm);
}
void shared_cond_wait(shared_cond_t* obj,
                      struct shared_mutex_t* mutex) {
  int ret = -1;
  if ((ret = pthread_cond_wait(obj->ptr,
                  shared_mutex_getptr(mutex)))) {
    fprintf(stderr, "ERROR: Waiting on the cv failed: %s\n",
            strerror(ret));
    exit(1);
  }
}
void shared_cond_timedwait(shared_cond_t* obj,
                           struct shared_mutex_t* mutex,
                           long int time_nanosec) {
  int ret = -1;
  struct timespec ts;
  ts.tv_sec = ts.tv_nsec = 0;
  if ((ret = clock_gettime(CLOCK_REALTIME, &ts))) {
    fprintf(stderr,
            "ERROR: Failed at reading current time: %s\n",
            strerror(errno));
    exit(1);
  }
  ts.tv_sec += (int)(time_nanosec / (1000L * 1000 * 1000));
  ts.tv_nsec += time_nanosec % (1000L * 1000 * 1000);
  if ((ret = pthread_cond_timedwait(obj->ptr,
                  shared_mutex_getptr(mutex), &ts))) {
#if !defined(__APPLE__)
    if (ret == EOWNERDEAD) {
      fprintf(stderr,
              "WARN: The owner of the cv's mutex is dead ...\n");
      shared_mutex_make_consistent(mutex);
      fprintf(stdout, "INFO: I'm the new owner!\n");
      shared_mem_setowner(obj->shm, TRUE);
      return;
    } else if (ret == ETIMEDOUT) {
#else
    if (ret == ETIMEDOUT) {
#endif
      return;
    }
    fprintf(stderr, "ERROR: Waiting on the cv failed: %s\n",
            strerror(ret));
    exit(1);
  }
}
void shared_cond_broadcast(shared_cond_t* obj) {
  int ret = -1;
  if ((ret = pthread_cond_broadcast(obj->ptr))) {
    fprintf(stderr, "ERROR: Broadcasting on the cv failed: %s\n",
        strerror(ret));
    exit(1);
  }
}
```

Code Box 18-24 [ExtremeC_examples_chapter18_4_shared_cond.c]: The definitions of all functions found in the shared condition variable class

In our shared condition variable class, we have only exposed the *broadcasting* behavior. We could also expose the *signaling* behavior. As you might remember from *Chapter 16*, *Thread Synchronization*, signaling a condition variable wakes up only one of the many waiting processes, without the ability to specify or predict which one. Broadcasting in contrast will wake all the waiting processes. In *example 18.4* we'll only use broadcasting, and that's why we have only exposed that function.

Note that since every condition variable has a companion mutex, the shared mutex class should be able to use an instance of the shared mutex class, and that's why we have declared `shared_mutex_t` as a forward declaration.

## Step 5 – The main logic

The following code box contains the main logic implemented for our example:

```cpp
#include "ExtremeC_examples_chapter18_4_shared_int32.h"
#include "ExtremeC_examples_chapter18_4_shared_mutex.h"
#include "ExtremeC_examples_chapter18_4_shared_cond.h"
int int_received = 0;
struct shared_cond_t* cond = NULL;
struct shared_mutex_t* mutex = NULL;
void sigint_handler(int signo) {
  fprintf(stdout, "\nHandling INT signal: %d ...\n", signo);
  int_received = 1;
}
int main(int argc, char** argv) {
  signal(SIGINT, sigint_handler);
  if (argc < 2) {
    fprintf(stderr,
            "ERROR: You have to provide the process number.\n");
    exit(1);
  }
  int my_number = atol(argv[1]);
  printf("My number is %d!\n", my_number);
  struct shared_int32_t* counter = shared_int32_new();
  shared_int32_ctor(counter, "/counter0");
  shared_int32_setvalue_ifowner(counter, 1);
  mutex = shared_mutex_new();
  shared_mutex_ctor(mutex, "/mutex0");
  cond = shared_cond_new();
  shared_cond_ctor(cond, "/cond0");
  shared_mutex_lock(mutex);
  while (shared_int32_getvalue(counter) < my_number) {
    if (int_received) {
        break;
    }
    printf("Waiting for the signal, just for 5 seconds ...\n");
    shared_cond_timedwait(cond, mutex, 5L * 1000 * 1000 * 1000);
    if (int_received) {
        break;
    }
    printf("Checking condition ...\n");
  }
  if (int_received) {
    printf("Exiting ...\n");
    shared_mutex_unlock(mutex);
    goto destroy;
  }
  shared_int32_setvalue(counter, my_number + 1);
  printf("My turn! %d ...\n", my_number);
  shared_mutex_unlock(mutex);
  sleep(1);
  // NOTE: The broadcasting can come after unlocking the mutex.
  shared_cond_broadcast(cond);
destroy:
  shared_cond_dtor(cond);
  shared_cond_delete(cond);
  shared_mutex_dtor(mutex);
  shared_mutex_delete(mutex);
  shared_int32_dtor(counter);
  shared_int32_delete(counter);
  return 0;
}
```

Code Box 18-25 [ExtremeC_examples_chapter18_4_main.c]: The main function of example 18.4

As you can see, the program accepts an argument indicating its number. As soon as the process finds out about its number, it starts to initialize the shared counter, the shared mutex, and the shared condition variable. It then enters a critical section being protected by the shared mutex.

Inside a loop, it waits for the counter to become equal to its number. Since it waits for 5 seconds, there could be a timeout and we may leave the `shared_cond_timedwait` function after 5 seconds. This basically means that the condition variable has not been notified during that 5 seconds. The process then checks the condition again and it goes to sleep for another 5 seconds. This continues until the process gets the turn.

When this happens, the process prints its number, increments the shared counter, and by broadcasting a signal on the shared condition variable object, it notifies the rest of the waiting processes about the modification which it has made to the shared counter. Only then does it prepare to quit.

In the meantime, if the user presses `Ctrl` + `C`, the signal handler defined as part of the main logic sets the local flag `int_received` and as soon as the process leaves the function `shared_mutex_timedwait` when it is inside the main loop, it notices the interrupt signal and exits the loop.

The following shell box shows how to compile *example 18.4*. We are going to compile it in Linux:

```cpp
$ gcc -c ExtremeC_examples_chapter18_4_shared_mem.c -o shared_mem.o
$ gcc -c ExtremeC_examples_chapter18_4_shared_int32.c -o shared_int32.o
$ gcc -c ExtremeC_examples_chapter18_4_shared_mutex.c -o shared_mutex.o
$ gcc -c ExtremeC_examples_chapter18_4_shared_cond.c -o shared_cond.o
$ gcc -c ExtremeC_examples_chapter18_4_main.c -o main.o
$ gcc shared_mem.o shared_int32.o shared_mutex.o shared_cond.o \  main.o -lpthread -lrt -o ex18_4.out
$
```

Shell Box 18-6: Compiling the sources of example 18.4 and producing the final executable file

Now that we have got the final executable file `ex18_4.out`, we can run three processes and see how they count in sequence, no matter how you assign them the numbers and in what order they are run. Let's run the first process. We assign to this process the the number 3, by passing the number as an option to the executable file:

```cpp
$ ./ex18_4.out 3
My number is 3!
WARN: The shared memory /counter0 does not exist.
The shared memory /counter0 is created and opened.
WARN: The shared memory /mutex0 does not exist.
The shared memory /mutex0 is created and opened.
WARN: The shared memory /cond0 does not exist.
The shared memory /cond0 is created and opened.
Waiting for the signal, just for 5 seconds ...
Checking condition ...
Waiting for the signal, just for 5 seconds ...
Checking condition ...
Waiting for the signal, just for 5 seconds ...
```

Shell Box 18-7: Running the first process which takes the number 3

As you see in the preceding output, the first process creates all the required shared objects and becomes the owner of the shared resources. Now, let's run the second process in a separate Terminal. It takes the number 2:

```cpp
$ ./ex18_4.out 2
My number is 2!
The shared memory /counter0 is opened.
The shared memory /mutex0 is opened.
The shared memory /cond0 is opened.
Waiting for the signal, just for 5 seconds ...
Checking condition ...
Waiting for the signal, just for 5 seconds ...
```

Shell Box 18-8: Running the second process which takes the number 2

And finally, the last process takes the number 1\. Since this process has been assigned the number 1, it prints its number immediately, increments the shared counter, and notifies the rest of the processes about it:

```cpp
$ ./ex18_4.out 1
My number is 1!
The shared memory /counter0 is opened.
The shared memory /mutex0 is opened.
The shared memory /cond0 is opened.
My turn! 1 ...
The shared memory /cond0 is unmapped.
The shared memory /cond0 is closed.
The shared memory /mutex0 is unmapped.
The shared memory /mutex0 is closed.
The shared memory /counter0 is unmapped.
The shared memory /counter0 is closed.
$
```

Shell Box 18-9: Running the third process which takes the number 1\. This process will exit immediately since it has the number 1.

Now, if you go back to the second process, it prints out its number, increments the shared counter, and notifies the third process about that:

```cpp
...
Waiting for the signal, just for 5 seconds ...
Checking condition ...
My turn! 2 ...
The shared memory /cond0 is unmapped.
The shared memory /cond0 is closed.
The shared memory /mutex0 is unmapped.
The shared memory /mutex0 is closed.
The shared memory /counter0 is unmapped.
The shared memory /counter0 is closed.
$
```

Shell Box 18-10: The second process prints its number and exits

Finally, going back to the first process, it gets notified by the second process, then it prints out its number and exits.

```cpp
...
Waiting for the signal, just for 5 seconds ...
Checking condition ...
My turn! 3 ...
The shared memory /cond0 is unmapped.
The shared memory /cond0 is closed.
The shared memory /cond0 is deleted.
The shared memory /mutex0 is unmapped.
The shared memory /mutex0 is closed.
The shared memory /mutex0 is deleted.
The shared memory /counter0 is unmapped.
The shared memory /counter0 is closed.
The shared memory /counter0 is deleted.
$
```

Shell Box 18-11: The first process prints its number and exits. It also deletes all shared memory entries.

Since the first process is the owner of all shared memories, it should delete them upon exiting. Releasing the allocated resources in a multi-processing environment can be quite tricky and complex because a simple mistake is enough to cause all the processes to crash. Further synchronization is required when a shared resource is going to be removed from the system.

Suppose that, in the preceding example, we'd run the first process with the number 2 and the second process with the number 3\. Therefore, the first process should print its number before the second process. When the first process exits since it's the creator of all shared resources, it deletes the shared objects and the second process crashes as soon as it wants to access them.

This is just a simple example of how finalization can be tricky and problematic in multi-process systems. In order to mitigate the risk of such crashes, one needs to introduce further synchronization among processes.

During the previous sections, we covered the mechanisms which can be employed to synchronize a number of processes while all of them are running on the same host. In the following section, we are going to briefly talk about distributed concurrency control mechanisms and their features.

# Distributed concurrency control

So far in this chapter we have assumed that all processes exist within the same operating system, and hence the same machine. In other words, we were constantly talking about a single-host software system.

But real software systems usually go beyond that. Conversely to the single-host software system, we have distributed software systems. These systems have processes distributed throughout a network, and they function through communicating over the network.

Regarding a distributed system of processes, we can see more challenges in some aspects that are not present in a centralized or single-host system in that degree. Next, we discuss some of them briefly:

*   **In a distributed software system, you are probably experiencing parallelism instead of concurrency**. Since each process runs on a separate machine, and each process has its own specific processor, we will be observing parallelism instead of concurrency. Concurrency is usually limited to the borders of a single machine. Note that interleavings still exist and we might experience the same non-determinism as we saw in concurrent systems.
*   **Not all processes within a distributed software system are written using a single programming language**. It is pretty common to see various programming languages being used in a distributed software system. It is also common to see the same diversity in the processes of a single-host software system. Despite our implicit assumption about the processes within a system, which is that all of them have been written using C, we can have processes written using any other language. Different languages provide different ways of having concurrency and control mechanisms. Therefore, for example, in some languages, you may not be able to use a named mutex very easily. Diverse technologies and programming languages used in a software system, single-host or distributed, force us to use concurrency control mechanisms that are abstract enough to be available in all of them. This might limit us to using a specific synchronization technique which is available in a certain technology or a programming language.
*   **In a distributed system, you always have a network as the communication channel between two processes not residing on the same machine**. This is converse to our implicit assumption about the single-host system where all processes are running within the same operating system and using the available messaging infrastructures to communicate with each other.
*   **Having a network in the middle means that you have latency**. There is a slight latency in single-host systems as well, but it is determined and manageable. It is also much lower than the latency you might experience in a network. Latency simply means that a process may not receive a message immediately because of many reasons having roots in the networking infrastructure. Nothing should be considered immediate in these systems.
*   **Having a network in the middle also results in security issues**. When you have all the processes in one system, and all of them are communicating within the same boundary using mechanisms with extremely low latency, the security issues are greatly different. One has to firstly access the system itself in order to attack the system, but in a distributed system, all message passing is being done through the network. You might get an *eavesdropper* in the middle to sniff or, even worse, alter the messages. Regarding our discussion about synchronization in distributed systems, this is also applicable to messages meant to synchronize the processes within a distributed system.
*   **Other than latency and security issues, you might have delivery issues that happen far less frequently in single-host multi-process systems**. Messages should be delivered to be processed. When a process sends a message to another process within the system, somehow the sender process should make sure that its message is received by the other end. *Delivery guarantee* mechanisms are possible, but they're costly and, in some scenarios, it is just not possible to use them at all. In those situations, a special kind of messaging problem is seen, which is usually modeled by the famous *Two Generals Problem*.

The preceding differences and possible issues are enough to force us to invent new ways of synchronization among processes and various components of giant distributed systems. Generally, there are two ways to make a distributed system transactional and synchronized:

*   **Centralized process synchronization**: These techniques need a central process (or node) that manages the processes. All the other processes within the system should be in constant communication with this central node, and they need its approval in order to enter their critical sections.
*   **Distributed (or peer-to-peer) process synchronization**: Having a process synchronization infrastructure that does not have a central node is not an easy task. This is actually an active field of research, and there are some ad hoc algorithms.

In this section, we tried to shine a little light over the complexity of concurrency control in a distributed multi-process system. Further discussions about distributed concurrency control would be out of the scope of this book.

# Summary

In this chapter, we completed our discussion regarding multi-processing environments. As part of this chapter, we discussed the following:

*   What a named semaphore is and how it can be created and used by multiple processes.
*   What a named mutex is and how it should be used using a shared memory region.
*   We gave an example which was about termination orchestration in which a number of processes were waiting for a sign to get terminated and the signal was received and handled by one of the processes and propagated to others. We implemented this example using shared mutexes.
*   What a named condition variable is and how it can become shared and named using a shared memory region.
*   We demonstrated another example of counting processes. As part of this example, we used inheritance to reduce the amount of code duplication for mutex and condition variable objects having an associated shared memory region.
*   We briefly explored the differences and challenges found in a distributed system.
*   We briefly discussed the methods which can be employed to bring concurrency control into distributed software.

In the upcoming chapter, we start our discussions regarding **Inter-Process Communication** (**IPC**) techniques. Our discussions will span two chapters and we will cover many topics such as computer networks, transport protocols, socket programming, and many more useful topics.