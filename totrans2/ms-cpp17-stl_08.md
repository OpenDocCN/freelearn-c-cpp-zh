# Allocators

We've seen in the preceding chapters that C++ has a love-hate relationship with dynamic memory allocation.

On one hand, dynamic memory allocation from the heap is a "code smell"; chasing pointers can hurt a program's performance, the heap can be exhausted unexpectedly (leading to exceptions of type `std::bad_alloc`), and manual memory management is so subtly difficult that C++11 introduced several different "smart pointer" types to manage the complexity (see [Chapter 6](part0093.html#2OM4A0-2fdac365b8984feebddfbb9250eaf20d), *Smart Pointers*). Successive versions of C++ after 2011 have also added a great number of non-allocating algebraic data types, such as `tuple`, `optional`, and `variant` (see [Chapter 5](part0074.html#26I9K0-2fdac365b8984feebddfbb9250eaf20d), *Vocabulary Types*) that can express ownership or containment without ever touching the heap.

On the other hand, the new smart pointer types do effectively manage the complexity of memory management; in modern C++ you can safely allocate and deallocate memory without ever using raw `new` or `delete` and without fear of memory leaks. And heap allocation is used "under the hood" of many of the new C++ features (`any`, `function`, `promise`) just as it continues to be used by many of the old ones (`stable_partition`, `vector`).

So there's a conflict here: How can we use these great new features (and the old ones) that depend on heap allocation, if we are simultaneously being told that good C++ code avoids heap allocation?

In most cases, you should err on the side of *using the features that C++ provides*. If you want a resizeable vector of elements, you *should* be using the default `std::vector`, unless you have measured an actual performance problem with using it in your case. But there also exists a class of programmers--working in very constrained environments such as flight software--who have to avoid touching the heap for a very simple reason: "the heap" does not exist on their platforms! In these embedded environments, the entire footprint of the program must be laid out at compile time. Some such programs simply avoid any algorithm that resembles heap allocation--you can never encounter unexpected resource exhaustion if you never dynamically allocate resources of any kind! Other such programs do use algorithms resembling heap allocation, but require that the "heap" be represented explicitly in their program (say, by a very large array of `char` and functions for "reserving" and "returning" consecutive chunks of that array).

It would be extremely unfortunate if programs of this last kind were unable to use the features that C++ provides, such as `std::vector` and `std::any`. So, ever since the original standard in 1998, the standard library has provided a feature known as *allocator-awareness*. When a type or an algorithm is *allocator-aware*, it provides a way for the programmer to specify exactly how the type or algorithm ought to reserve and return dynamic memory. This "how" is reified into an object known as an *allocator*.

In this chapter we'll learn:

*   The definitions of "allocator" and "memory resource"
*   How to create your own memory resource that allocates out of a static buffer
*   How to make your own containers "allocator-aware"
*   The standard memory-resource types from namespace `std::pmr`, and their surprising pitfalls
*   That many of the strange features of the C++11 allocator model are intended purely to support `scoped_allocator_adaptor`
*   What makes a type a "fancy pointer" type, and where such types might be useful

# An allocator is a handle to a memory resource

In reading this chapter, you'll have to keep in mind the difference between two fundamental concepts, which I am going to call *memory resource* and *allocator*. A *memory resource* (a name inspired by the standard's own terminology--you might find it more natural to call it "a heap") is a long-lived object that can dole out chunks of memory on request (usually by carving them out of a big block of memory that is owned by the memory resource itself). Memory resources have classically object-oriented semantics (see [Chapter 1](part0021.html#K0RQ0-2fdac365b8984feebddfbb9250eaf20d), *Classical Polymorphism and Generic Programming*): you create a memory resource once and never move or copy it, and equality for memory resources is generally defined by *object identity*. On the other hand, an *allocator* is a short-lived handle *pointing* to a memory resource. Allocators have pointer semantics: you can copy them, move them around, and generally mess with them as much as you want, and equality for allocators is generally defined by whether they point to the same memory resource. Instead of saying an allocator "points to" a particular memory resource, we might also say that the allocator is "backed by" that memory resource; the terms are interchangeable.

When I talk about "memory resources" and "allocators" in this chapter, I will be talking about the preceding concepts. The standard library also has a couple of types named `memory_resource` and `allocator`; whenever I'm talking about those types I'll be careful to use `typewriter text`. It shouldn't be too confusing. The situation is similar to [Chapter 2](part0026.html#OPEK0-2fdac365b8984feebddfbb9250eaf20d), *Iterators and Ranges*, where we talked about "iterators" and also about `std::iterator`. Of course that was easier because I only mentioned `std::iterator` in order to tell you never to use it; it has no place in well-written C++ code. In this chapter we'll learn that `std::pmr::memory_resource` *does* have a place in certain C++ programs!

Even though I described an allocator as a handle "pointing to" a memory resource, you should notice that sometimes the memory resource in question is a global singleton--one example of such a singleton is the global heap, whose accessors are the global `operator new` and `operator delete`. Just as a lambda which "captures" a global variable doesn't actually capture anything, an allocator backed by the global heap doesn't actually need any state. In fact, `std::allocator<T>` is just such a stateless allocator type--but we're getting ahead of ourselves here!

# Refresher - Interfaces versus concepts

Recall from [Chapter 1](part0021.html#K0RQ0-2fdac365b8984feebddfbb9250eaf20d), *Classical Polymorphism and Generic Programming*, that C++ offers two mostly incompatible ways of dealing with polymorphism. Static, compile-time polymorphism is called *generic programming*; it relies on expressing the polymorphic interface as a *concept* with many possible *models*, and the code that interacts with the interface is expressed in terms of *templates*. Dynamic, runtime polymorphism is called *classical polymorphism*; it relies on expressing the polymorphic interface as a *base class* with many possible *derived classes*, and the code that interacts with the interface is expressed in terms of calls to *virtual methods*.

In this chapter we'll have our first (and last) really close encounter with generic programming. It is impossible to make sense of C++ allocators unless you can hold in your mind two ideas at once: on one hand the *concept* `Allocator`, which defines an interface, and on the other hand some particular *model*, such as `std::allocator`, that implements behavior conforming to the `Allocator` concept.

To complicate matters further, the `Allocator` concept is really a templated family of concepts! It would be more accurate to talk about the family of concepts `Allocator<T>`; for example, `Allocator<int>` would be the concept defining "an allocator that allocates `int` objects," and `Allocator<char>` would be "an allocator that allocates `char` objects," and so on. And, for example, the concrete class `std::allocator<int>` is a model of the concept `Allocator<int>`, but it is *not* a model of `Allocator<char>`.

Every allocator of `T` (every `Allocator<T>`) is required to provide a member function named `allocate`, such that `a.allocate(n)` returns a pointer to enough memory for an array of `n` objects of type `T`. (That pointer will come from the memory resource that backs the allocator instance.) It is not specified whether the `allocate` member function ought to be static or non-static, nor whether it ought to take exactly one parameter (`n`) or perhaps some additional parameters with default values. So both of the following class types would be acceptable models of `Allocator<int>` in that respect:

[PRE0]

The class designated `int_allocator_2017` is obviously a *simpler* way to model `Allocator<int>`, but `int_allocator_2014` is just as correct a model, because in both cases the expression `a.allocate(n)` will be accepted by the compiler; and that's all we ask for, when we're talking about *generic programming*.

In contrast, when we do classical polymorphism, we specify a fixed signature for each method of the base class, and derived classes are not allowed to deviate from that signature:

[PRE1]

The derived class `classical_derived` is not allowed to add any extra parameters onto the signature of the `allocate` method; it's not allowed to change the return type; it's not allowed to make the method `static`. The interface is more "locked down" with classical polymorphism than it is with generic programming.

Because a "locked-down" classical interface is naturally easier to describe than a wide-open conceptual one, we'll start our tour of the allocator library with C++17's brand-new, classically polymorphic `memory_resource`.

# Defining a heap with memory_resource

Recall that on resource-constrained platforms, we might not be permitted to use "the heap" (for example via `new` and `delete`), because the platform's runtime might not support dynamic memory allocation. But we can make our own little heap--not "the heap," just "a heap"--and simulate the effect of dynamic memory allocation by writing a couple of functions `allocate` and `deallocate` that reserve chunks of a big statically allocated array of `char`, something like this:

[PRE2]

To keep the code as simple as possible, I made `deallocate` a no-op. This little heap allows the caller to allocate up to 10,000 bytes of memory, and then starts throwing `bad_alloc` from then on.

With a little more investment in the code, we can allow the caller to allocate and deallocate an infinite number of times, as long as the total outstanding amount of allocated memory doesn't exceed 10,000 bytes and as long as the caller always follows a "last-allocated-first-deallocated" protocol:

[PRE3]

The salient point here is that our heap has some *state* (in this case, `big_buffer` and `index`), and a couple of functions that manipulate this state. We've seen two different possible implementations of `deallocate` already--and there are other possibilities, with additional shared state, that wouldn't be so "leaky"--yet the interface, the signatures of `allocate` and `deallocate` themselves, has remained constant. This suggests that we could wrap up our state and accessor functions into a C++ object; and the wide variety of implementation possibilities plus the constancy of our function signatures suggests that we could use some classical polymorphism.

The C++17 allocator model does exactly that. The standard library provides the definition of a classically polymorphic base class, `std::pmr::memory_resource`, and then we implement our own little heap as a derived class. (In practice we might use one of the derived classes provided by the standard library, but let's finish up our little example before talking about those.) The base class `std::pmr::memory_resource` is defined in the standard header `<memory_resource>`:

[PRE4]

Notice the curious layer of indirection between the `public` interface of the class and the `virtual` implementation. Usually when we're doing classical polymorphism, we have just one set of methods that are both `public` and `virtual`; but in this case, we have a `public` non-virtual interface that calls down into the private virtual methods. This splitting of the interface from the implementation has a few obscure benefits--for example, it prevents any child class from invoking `this->SomeBaseClass::allocate()` using the "directly invoke a virtual method non-virtually" syntax--but honestly, its main benefit to us is that when we define a derived class, we don't have to use the `public` keyword at all. Because we are specifying only the *implementation*, not the interface, all the code we write can be `private`. Here's our trivial little leaky heap:

[PRE5]

Notice that the standard library's `std::pmr::memory_resource::allocate` takes not only a size in bytes, but also an alignment. We need to make sure that whatever pointer we return from `do_allocate` is suitably aligned; for example, if our caller is planning to store `int` in the memory we give him, he might ask for four-byte alignment.

The last thing to notice about our derived class `example_resource` is that it represents the actual resources controlled by our "heap"; that is, it actually contains, owns, and manages the `big_buffer` out of which it's allocating memory. For any given `big_buffer`, there will be exactly one `example_resource` object in our program that manipulates that buffer. Just as we said earlier: objects of type `example_resource` are "memory resources," and thus they are *not* intended to be copied or moved around; they are classically object-oriented, not value-semantic.

The standard library provides several species of memory resource, all derived from `std::pmr::memory_resource`. Let's look at a few of them.

# Using the standard memory resources

Memory resources in the standard library come in two flavors. Some of them are actual class types, of which you can create instances; and some of them are "anonymous" class types accessed only via singleton functions. Generally you can predict which is which by thinking about whether two objects of the type could ever possibly be "different," or whether the type is basically a singleton anyway.

The simplest memory resource in the `<memory_resource>` header is the "anonymous" singleton accessed via `std::pmr::null_memory_resource()`. The definition of this function is something like this:

[PRE6]

Notice that the function returns a pointer to the singleton instance. Generally, `std::pmr::memory_resource` objects will be manipulated via pointers, because the `memory_resource` objects themselves cannot move around.

`null_memory_resource` seems fairly useless; all it does is throw an exception when you try to allocate from it. However, it can be useful when you start using the more complicated memory resources which we'll see in a moment.

The next most complicated memory resource is the singleton accessed via `std::pmr::new_delete_resource()`; it uses `::operator new` and `::operator delete` to allocate and deallocate memory.

Now we move on to talking about the named class types. These are resources where it makes sense to have multiple resources of identical type in a single program. For example, there's `class std::pmr::monotonic_buffer_resource`. This memory resource is fundamentally the same as our `example_resource` from earlier, except for two differences: Instead of holding its big buffer as member data (`std::array`-style), it just holds a pointer to a big buffer allocated from somewhere else (`std::vector`-style). And when its first big buffer runs out, rather than immediately starting to throw `bad_alloc`, it will attempt to allocate a *second* big buffer, and allocate chunks out of that buffer until *it's* all gone; at which point it will allocate a third big buffer... and so on, until eventually it cannot even allocate any more big buffers. As with our `example_resource`, none of the deallocated memory is ever freed until the resource object itself is destroyed. There is one useful escape valve: If you call the method `a.release()`, the `monotonic_buffer_resource` will release all of the buffers it's currently holding, sort of like calling `clear()` on a vector.

When you construct a resource of type `std::pmr::monotonic_buffer_resource`, you need to tell it two things: Where is its first big buffer located? and, when that buffer is exhausted, who it should ask for another buffer? The first of these questions is answered by providing a pair of arguments `void*, size_t` that describes the first big buffer (optionally `nullptr`); and the second question is answered by providing a `std::pmr::memory_resource*` that points to this resource's "upstream" resource. One sensible thing to pass in for the "upstream" resource would be `std::pmr::new_delete_resource()`, so as to allocate new buffers using `::operator new`. Or, another sensible thing to pass in would be `std::pmr::null_memory_resource()`, so as to put a hard cap on the memory usage of this particular resource. Here's an example of the latter:

[PRE7]

If you forget what upstream resource a particular `monotonic_buffer_resource` is using, you can always find out by calling `a.upstream_resource()`; that method returns a pointer to the upstream resource that was provided to the constructor.

# Allocating from a pool resource

The final kind of memory resource provided by the C++17 standard library is what's called a "pool resource." A pool resource doesn't just manage one big buffer, such as `example_resource`; or even a monotonically increasing chain of buffers, such as `monotonic_buffer_resource`. Instead it manages a whole lot of "blocks" of various sizes. All the blocks of a given size are stored together in a "pool," so that we can talk about "the pool of blocks of size 4," "the pool of blocks of size 16," and so on. When a request comes in for an allocation of size *k*, the pool resource will look in the pool of blocks of size *k*, pull one out and return it. If the pool for size *k* is empty, then the pool resource will attempt to allocate some more blocks from its upstream resource. Also, if a request comes in for an allocation so large that we don't even have a pool for blocks of that size, then the pool resource is allowed to pass the request directly on to its upstream resource.

Pool resources come in two flavors: *synchronized* and *unsynchronized*, which is to say, thread-safe and thread-unsafe. If you're going to be accessing a pool from two different threads concurrently, then you should use `std::pmr::synchronized_pool_resource`, and if you're definitely never going to do that, and you want raw speed, then you should use `std::pmr::unsynchronized_pool_resource`. (By the way, `std::pmr::monotonic_buffer_resource` is always thread-unsafe; and `new_delete_resource()` is effectively thread-safe, since all it does is call `new` and `delete`.)

When you construct a resource of type `std::pmr::synchronized_pool_resource`, you need to tell it three things: Which block sizes it should keep in its pools; how many blocks it should glom together into a "chunk" when it goes to get more blocks from the upstream resource; and who is its upstream resource. Unfortunately, the standard interface leaves much to be desired here--so much so that frankly I recommend that if these parameters truly matter to you, you should be implementing your own derived `memory_resource` and not touching the standard library's version at all. The syntax for expressing these options is also fairly wonky:

[PRE8]

Notice that there is no way to specify exactly which block sizes you want; that's left up to the vendor's implementation of `synchronized_pool_resource`. If you're lucky, it will choose decent block sizes that match your use-case; but personally I wouldn't rely on that assumption. Notice also that there's no way to use different upstream resources for the different block sizes, nor a different upstream resource for the "fallback" resource that's used when the caller requests an unusually sized allocation.

In short, I would steer clear of the built-in `pool_resource` derived classes for the foreseeable future. But the fundamental idea of deriving your own classes from `memory_resource` is solid. If you're concerned about memory allocation and managing your own little heaps, I'd recommend adopting `memory_resource` into your codebase.

Now, so far we've only been talking about various allocation strategies, as "personified" by the different `memory_resource` derived classes. We still need to see how to hook `memory_resource` into the algorithms and containers of the Standard Template Library. And to do that, we'll have to transition from the classically polymorphic world of `memory_resource` back into the value-semantic world of the C++03 STL.

# The 500 hats of the standard allocator

The standard allocator model must have seemed amazing in 2011\. We're about to see how, with just one C++ type, we can accomplish all of the following feats:

*   Specify a memory resource to be used for allocating memory.
*   Annotate each allocated pointer with some metadata that will be carried along
    for its whole lifetime, all the way to deallocation time.
*   Associate a container object with a particular memory resource, and make sure
    that association is "sticky"--this container object will always use the given
    heap for its allocations.
*   Associate a container *value* with a particular memory resource, meaning
    that the container can be efficiently moved around using value semantics without
    forgetting how to deallocate its contents.
*   Choose between the two mutually exclusive behaviors above.
*   Specify a strategy for allocating memory at all levels of a multi-level
    container, such as a vector of vectors.
*   Redefine what it means to "construct" the contents of a container, so that
    for example, `vector<int>::resize` could be defined to default-initialize new elements instead of zero-initializing them.

This is just an *insane* number of hats for any one class type to wear--a massive violation of the Single Responsibility Principle. Nevertheless, this is what the standard allocator model does; so let's try to explain all these features.

Remember that a "standard allocator" is just any class type that satisfies the concept `Allocator<T>` for some type `T`. The standard library provides three standard allocator types: `std::allocator<T>`, `std::pmr::polymorphic_allocator<T>`, and `std::scoped_allocator_adaptor<A...>`.

Let's start by looking at `std::allocator<T>`:

[PRE9]

`std::allocator<T>` has the member functions `allocate` and `deallocate` that are required by the `Allocator<T>` concept. Remember that we are in the world of concept-based generic programming now! The classically polymorphic `memory_resource` *also* had member functions named `allocate` and `deallocate`, but they always returned `void*`, not `T*`. (Also, `memory_resource::allocate()` took two arguments--`bytes` and `align`--whereas `allocator<T>::allocate()` takes only one argument. The first reason for this is that `allocator<T>` predated the mainstream understanding that alignment was a big deal; remember that the `sizeof` operator was inherited from C in the 1980s but the `alignof` operator only showed up in C++11\. The second reason is that in the context of `std::allocator<T>`, we know that the type of the objects being allocated is `T`, and thus the requested alignment must necessarily be `alignof(T)`. `std::allocator<T>` doesn't use that information, because it predates `alignof`; but in principle it could, and that's why the `Allocator<T>` concept requires only the signature `a.allocate(n)` instead of `a.allocate(n, align)`.)

The constructor marked `NOTE 1` is important; every allocator needs a templated constructor modeled after this one. The constructors following the line marked `NOTE 2` are unimportant; the only reason we wrote them explicitly in the code is because if we had not written them, they would have been implicitly deleted due to the presence of a user-defined constructor (namely, the `NOTE 1` constructor).

The idea of any standard allocator is that we can plug it in as the very last template type parameter of any standard container ([Chapter 4](part0052.html#1HIT80-2fdac365b8984feebddfbb9250eaf20d), *The Container Zoo*) and the container will then use that allocator instead of its usual mechanisms anytime it needs to allocate memory for any reason. Let's see an example:

[PRE10]

Here our class `helloworld<int>` models `Allocator<int>`; but we've omitted the templated constructor. This is fine if we're dealing only with `vector`, because `vector` will allocate only arrays of its element type. However, watch what happens if we change the test case to use `list` instead:

[PRE11]

Under libc++, this code spews several dozen lines of error messages, which boil down to the essential complaint "no known conversion from `helloworld<int>` to `helloworld<std::__1::__list_node<int, void *>>`." Recall from the diagram in [Chapter 4](part0052.html#1HIT80-2fdac365b8984feebddfbb9250eaf20d), *The Container Zoo*, that `std::list<T>` stores its elements in nodes that are larger than the size of `T` itself. So `std::list<T>` isn't going to be trying to allocate any `T` objects; it wants to allocate objects of type `__list_node`. To allocate memory for `__list_node` objects, it needs an allocator that models the concept `Allocator<__list_node>`, not `Allocator<int>`.

Internally, the constructor of `std::list<int>` takes our `helloworld<int>` and attempts to "rebind" it to allocate `__list_node` objects instead of `int` objects. This is accomplished via a *traits class--*a C++ idiom that we first encountered in [Chapter 2](part0026.html#OPEK0-2fdac365b8984feebddfbb9250eaf20d), *Iterators and Ranges*:

[PRE12]

The standard class template `std::allocator_traits<A>` wraps up a lot of information about the allocator type `A` into one place, so it's easy to get at. For example, `std::allocator_traits<A>::value_type` is an alias for the type `T` whose memory is allocated by `A`; and `std::allocator_traits<A>::pointer` is an alias for the corresponding pointer type (generally `T*`).

The nested alias template `std::allocator_traits<A>::rebind_alloc<U>` is a way of "converting" an allocator from one type `T` to another type `U`. This type trait uses metaprogramming to crack open the type `A` and see: first, whether `A` has a nested template alias `A::rebind<U>::other` (this is rare), and second, whether type `A` can be expressed in the form `Foo<Bar,Baz...>` (where `Baz...` is some list of types which might be an empty list). If `A` can be expressed that way, then `std::allocator_traits<A>::rebind_alloc<U>` will be a synonym for `Foo<U,Baz...>`. Philosophically, this is completely arbitrary; but in practice it works for every allocator type you'll ever see. In particular, it works for `helloworld<int>`--which explains why we didn't have to muck around with providing a nested alias `rebind<U>::other` in our `helloworld` class. By providing a sensible default behavior, the `std::allocator_traits` template has saved us some boilerplate. This is the reason `std::allocator_traits` exists.

You might wonder why `std::allocator_traits<Foo<Bar,Baz...>>::value_type` doesn't default to `Bar`. Frankly, I don't know either. It seems like a no-brainer; but the standard library doesn't do it. Therefore, every allocator type you write (remember now we're talking about classes modeling `Allocator<T>`, and *not* about classes derived from `memory_resource`) must provide a nested typedef `value_type` that is an alias for `T`.

However, once you've defined the nested typedef for `value_type`, you can rely on `std::allocator_traits` to infer the correct definitions for its nested typedef `pointer` (that is, `T*`), and `const_pointer` (that is, `const T*`), and `void_pointer` (that is, `void*`), and so on. If you were following the previous discussion of `rebind_alloc`, you might guess that "converting" a pointer type like `T*` to `void*` is just as difficult or easy as "converting" an allocator type `Foo<T>` to `Foo<void>`; and you'd be correct! The values of these pointer-related type aliases are all computed via a *second* standard traits class, `std::pointer_traits<P>`:

[PRE13]

This traits class becomes very important when we talk about the next responsibility of `Allocator<T>`, which was "annotate each allocated pointer with some metadata that will be carried along for its whole lifetime."

# Carrying metadata with fancy pointers

Consider the following high-level design for a memory resource, which should remind you very much of `std::pmr::monotonic_buffer_resource`:

*   Keep a list of chunks of memory we've gotten from the system. For each chunk, also store an `index` of how many bytes we've allocated from the beginning of the chunk; and store a count `freed` of how many bytes we've deallocated from this specific chunk.
*   When someone calls `allocate(n)`, increment any one of our chunks' `index` by the appropriate number of bytes if possible, or get a new chunk from the upstream resource if absolutely necessary.
*   When someone calls `deallocate(p, n)`, figure out which of our chunks `p` came from and increment its `freed += n`. If `freed == index`, then the entire chunk is empty, so set `freed = index = 0`.

It's pretty straightforward to turn the foregoing description into code. The only problematic item is: in `deallocate(p, n)`, how do we figure out which of our chunks `p` came from?

This would be easy if we simply recorded the identity of the chunk in the "pointer" itself:

[PRE14]

Then in our `deallocate(p, n)` function, all we'd have to do is to look at `p.chunk()`. But to make this work, we'd need to change the signature of the `allocate(n)` and `deallocate(p, n)` functions so that `deallocate` took a `ChunkyPtr<T>` instead of `T*`, and `allocate` returned `ChunkyPtr<T>` instead of `T*`.

Fortunately, the C++ standard library gives us a way to do this! All we need to do is define our own type that models `Allocator<T>` and give it a member typedef `pointer` that evaluates to `ChunkyPtr<T>`:

[PRE15]

The traits classes `std::allocator_traits` and `std::pointer_traits` will take care of inferring the other typedefs--such as `void_pointer`, which through the magic of `pointer_traits::rebind` will end up as an alias for `ChunkyPtr<void>`.

I've left out the implementations of the `allocate` and `deallocate` functions here because they would depend on the interface of `ChunkyMemoryResource`. We might implement `ChunkyMemoryResource` something like this:

[PRE16]

Now we can use our `ChunkyMemoryResource` to allocate memory for standard allocator-aware containers like this:

[PRE17]

Now, I've chosen this example to make it look very simple and straightforward; and I've left out a lot of the details of the `ChunkyPtr<T>` type itself. If you try copying this code yourself, you'll find that you need to provide `ChunkyPtr` with a lot of overloaded operators such as `==`, `!=`, `<`, `++`, `--`, and `-`; and you'll also need to provide a specialization for `ChunkyPtr<void>` that omits the overloaded `operator*`. Most of the details are the same as what we covered in [Chapter 2](part0026.html#OPEK0-2fdac365b8984feebddfbb9250eaf20d), *Iterators and Ranges*, when we implemented our own iterator type. In fact, every "fancy pointer" type is required to be usable as a *random-access iterator*--which means that you must provide the five nested typedefs listed at the end of [Chapter 2](part0026.html#OPEK0-2fdac365b8984feebddfbb9250eaf20d), *Iterators and Ranges*: `iterator_category`, `difference_type`, `value_type`, `pointer`, and `reference`.

Finally, if you want to use certain containers such as `std::list` and `std::map`, you'll need to implement a static member function with the surprising name `pointer_to(r)`:

[PRE18]

This is because--as you may recall from [Chapter 4](part0052.html#1HIT80-2fdac365b8984feebddfbb9250eaf20d), *The Container Zoo*--a few containers such as `std::list` store their data in nodes whose `prev` and `next` pointers need to be able to point *either* to an allocated node *or* to a node which is contained within the member data of the `std::list` object itself. There are two obvious ways to accomplish this: Either every `next` pointer must be stored in a sort of tagged union of a fancy pointer and a raw pointer (perhaps a `std::variant` as described in [Chapter 5](part0074.html#26I9K0-2fdac365b8984feebddfbb9250eaf20d), *Vocabulary Types*), or else we must find a way of encoding a raw pointer *as* a fancy pointer. The standard library chose the latter approach. So, whenever you write a fancy pointer type, not only must it do all the things required of it by the allocator, and not only must it satisfy the requirements of a random-access iterator, but it must *also* have a way of representing any arbitrary pointer in the program's address space--at least if you want to use your allocator with node-based containers such as `std::list`.

Even after jumping through all these hoops, you'll find that (as of press time) neither libc++ nor libstdc++ can handle fancy pointers in any container more complicated than `std::vector`. They support just enough to work with a single fancy pointer type--`boost::interprocess::offset_ptr<T>`, which carries no metadata. And the standard continues to evolve; `std::pmr::memory_resource` was newly introduced in C++17, and as of this writing it is still not implemented by libc++ nor libstdc++.

You may also have noticed the lack of any standard base class for memory resources that use fancy pointers. Fortunately, this is easy to write yourself:

[PRE19]

The standard library provides no allocators that use fancy pointers; every library-provided allocator type uses raw pointers.

# Sticking a container to a single memory resource

The next hat worn by the standard allocator model--the next feature controlled by `std::allocator_traits`--is the ability to associate specific container objects with specific heaps. We used three bullet points to describe this feature earlier:

*   Associate a container object with a particular memory resource, and make sure
    that association is "sticky"--this container object will always use the given
    heap for its allocations.
*   Associate a container *value* with a particular memory resource, meaning
    that the container can be efficiently moved around using value semantics without
    forgetting how to deallocate its contents.
*   Choose between the two mutually exclusive behaviors just mentioned.

Let's look at an example, using `std::pmr::monotonic_buffer_resource` for our resource but using a hand-written class type for our allocator type. (Just to reassure you that you haven't missed anything: Indeed, we *still* haven't covered any standard-library-provided allocator types--except for `std::allocator<T>`, the trivial stateless allocator that is a handle to the global heap managed by `new` and `delete`.)

[PRE20]

Here our `Widget` is a classically object-oriented class type; we expect it to live at a specific memory address for its entire lifetime. Then, to reduce heap fragmentation or to improve cache locality, we've placed a large buffer inside each `Widget` object and made the `Widget` use that buffer as the backing store for its data members `v` and `lst`.

Now look at the `Widget::swap_elems(a, b)` function. It swaps the `v` data members of `Widget a` and `Widget b`. You might recall from [Chapter 4](part0052.html#1HIT80-2fdac365b8984feebddfbb9250eaf20d), *The Container Zoo*, that a `std::vector` is little more than a pointer to a dynamically allocated array, and so *usually* the library can swap two instances of `std::vector` by simply swapping their underlying pointers, without moving any of the underlying data--making vector swap an O(1) operation instead of an O(*n*) operation.

Furthermore, `vector` is smart enough to know that if it swaps pointers, it also needs to swap allocators--so that the information about how to deallocate travels along with the pointer that will eventually be in need of deallocation.

But in this case, if the library just swapped the pointers and allocators, it would be disastrous! We'd have a vector `a.v` whose underlying array was now "owned" by `b.mr`, and vice versa. If we destroyed `Widget b`, then the next time we accessed the elements of `a.v` we'd be accessing freed memory. And furthermore, even if we never accessed `a.v` again, our program would likely crash when the destructor of `a.v` attempted to call the `deallocate` method of the long-dead `b.mr`!

Fortunately, the standard library saves us from this fate. One of the responsibilities of an allocator-aware container is to appropriately *propagate* its allocator on copy-assignment, move-assignment, and swap. For historical reasons this is handled by a whole mess of typedefs in the `allocator_traits` class template, but in order to *use* allocator propagation correctly, you only have to know a couple of things:

*   Whether the allocator propagates itself, or whether it sticks firmly to a specific container, is a property of the *allocator type*. If you want one allocator to "stick" while another propagates, you *must* make them different types.
*   When an allocator is "sticky," it sticks to a particular (classical, object-oriented)
    container object. Operations that with a non-sticky allocator type would be O(1) pointer-swaps may become O(*n*), because "adopting" elements from some other allocator's memory space into our own requires allocating room for them in our own memory space.
*   Stickiness has a clear use-case (as we have just shown with `Widget`), and
    the effects of non-stickiness can be disastrous (again, see `Widget`). Therefore, `std::allocator_traits` assumes by default that an allocator type is sticky, unless it can tell that the allocator type is *empty* and thus is quite definitely *stateless*. The default for *empty* allocator types is effectively non-stickiness.
*   As a programmer, you basically always want the default: stateless allocators might as well propagate, and stateful allocators *probably* don't have much use outside of `Widget`-like scenarios where stickiness is required.

# Using the standard allocator types

Let's talk about the allocator types provided by the standard library.

`std::allocator<T>` is the default allocator type; it is the default value of the template type parameter to every standard container. So for example when you write `std::vector<T>` in your code, that's secretly the exact same type as `std::vector<T, std::allocator<T>>`. As we've mentioned before in this chapter, `std::allocator<T>` is a stateless empty type; it is a "handle" to the global heap managed by `new` and `delete`. Because `std::allocator` is a stateless type, `allocator_traits` assumes (correctly) that it should be non-sticky. This means that operations such as `std::vector<T>::swap` and `std::vector<T>::operator=` are guaranteed to be very efficient pointer-swaps--because any object of type `std::vector<T, std::allocator<T>>` always knows how to deallocate memory that was originally allocated by any other `std::vector<T, std::allocator<T>>`.

`std::pmr::polymorphic_allocator<T>` is a new addition in C++17\. It is a stateful, non-empty type; its one data member is a pointer to a `std::pmr::memory_resource`. (In fact, it is almost identical to `WidgetAlloc` in our sample code from earlier in this chapter!) Two different instances of `std::pmr::polymorphic_allocator<T>` are not necessarily interchangeable, because their pointers might point to completely different `memory_resource`s; this means that an object of type `std::vector<T, std::pmr::polymorphic_allocator<T>>` does *not* necessarily know how to deallocate memory that was originally allocated by some other `std::vector<T, std::pmr::polymorphic_allocator<T>>`. That, in turn, means that `std::pmr::polymorphic_allocator<T>` is a "sticky" allocator type; and *that* means that operations such as `std::vector<T, std::pmr::polymorphic_allocator<T>>::operator=` can end up doing lots of copying.

By the way, it's quite tedious to write out the name of the type `std::vector<T, std::pmr::polymorphic_allocator<T>>` over and over. Fortunately, the standard library implementors came to the same realization, and so the standard library provides type aliases in the `std::pmr` namespace:

[PRE21]

# Setting the default memory resource

The biggest difference between the standard `polymorphic_allocator` and our example `WidgetAlloc` is that `polymorphic_allocator` is default-constructible. Default-constructibility is arguably an attractive feature of an allocator; it means that we can write the second of these two lines instead of the first:

[PRE22]

On the other hand, when you look at that second line, you might wonder, "Where is the underlying array actually being allocated?" After all, the main point of specifying an allocator is that we want to know where our bytes are coming from! That's why the *normal* way to construct a standard `polymorphic_allocator` is to pass in a pointer to a `memory_resource`--in fact, this idiom is expected to be *so* common that the conversion from `std::pmr::memory_resource*` to `std::pmr::polymorphic_allocator` is an implicit conversion. But `polymorphic_allocator` does have a default, zero-argument constructor as well. When you default-construct a `polymorphic_allocator`, you get a handle to the "default memory resource," which by default is `new_delete_resource()`. However, you can change this! The default memory resource pointer is stored in a global atomic (thread-safe) variable which can be manipulated with the library functions `std::pmr::get_default_resource()` (which returns the pointer) and `std::pmr::set_default_resource()` (which assigns a new value to the pointer and returns the previous value).

If you want to avoid heap allocation via `new` and `delete` altogether, it might make sense to call `std::pmr::set_default_resource(std::pmr::null_memory_resource())` at the start of your program. Of course you can't stop any other part of your program from going rogue and calling `set_default_resource` itself; and because the same global variable is shared by every thread in your program, you might run into some very strange behavior if you *try* to modify the default resource during the program's execution. There is no way to say "set the default resource only for my current thread," for example. Furthermore, calling `get_default_resource()` (such as from the default constructor of `polymorphic_allocator`) performs an atomic access, which will tend to be marginally slower than if the atomic access could have been avoided. Therefore, your best course of action is to avoid the default constructor of `polymorphic_allocator`; always be explicit as to which memory resource you're trying to use. For absolute foolproofness, you might consider simply using the above `WidgetAlloc` instead of `polymorphic_allocator`; having *no* default constructor, `WidgetAlloc` flatly cannot be misused.

# Making a container allocator-aware

Having covered memory resources (heaps) and allocators (handles to heaps), let's turn now to the third leg of the tripod: container classes. Inside each allocator-aware container, at least four things have to happen:

*   The container instance must store an allocator instance as member data. (Therefore the container must take the type of the allocator as a template parameter; otherwise it can't know how much space to reserve for that member variable.)
*   The container must provide constructors taking an allocator argument.
*   The container must actually use its allocator to allocate and deallocate memory; every use of `new` or `delete` must be banished.
*   The container's move constructor, move assignment operator, and `swap` function must all propagate the allocator according to its `allocator_traits`.

Here is a very simple allocator-aware container--a container of just one single object, allocated on the heap. This is something like an allocator-aware version of `std::unique_ptr<T>` from [Chapter 6](part0093.html#2OM4A0-2fdac365b8984feebddfbb9250eaf20d), *Smart Pointers*:

[PRE23]

Notice that where `unique_ptr` uses `T*`, our present code uses `allocator_traits<A>::pointer`; and where `make_unique` uses `new` and `delete`, our present code uses the one-two punch of `allocator_traits<A>::allocate`/`construct` and `allocator_traits<A>::destroy`/`deallocate`. We've already discussed the purpose of `allocate` and `deallocate`--they deal with getting memory from the appropriate memory resource. But those chunks of memory are just raw bytes; to turn a chunk of memory into a usable object we have to construct an instance of `T` at that address. We could use "placement `new`" syntax for this purpose; but we'll see in the next section why it's important to use `construct` and `destroy` instead.

Finally, before we proceed, notice that the destructor of `uniqueish` checks to see whether an allocation exists before trying to deallocate it. This is important because it gives us a value of `uniqueish` representing the "empty object"--a value that can be constructed without allocating any memory, and that is a suitably "moved-from" representation for our type.

Now let's implement the move operations for our type. We'd like to ensure that after you move out of a `uniqueish<T>` object, the moved-from object is "empty." Furthermore, if the left-hand object and the right-hand object share the same allocator, or if the allocator type is "not sticky," then we'd like to avoid calling the move constructor of `T` at all--we'd like to transfer ownership of the allocated pointer from the right-hand-side object to the left-hand object:

[PRE24]

The move *constructor* is just about as simple as it ever was. The only minor difference is that we have to remember to construct our `m_allocator` as a copy of the right-hand object's allocator.

We could use `std::move` to move the allocator instead of copying it, but I didn't think it was worth it for this example. Remember that an allocator is just a thin "handle" pointing to the actual memory resource, and that a lot of allocator types, such as `std::allocator<T>`, are actually empty. Copying an allocator type should always be relatively cheap. Still, using `std::move` here wouldn't have hurt.

The move *assignment operator*, on the other hand, is very complicated! The first thing we need to do is check whether our allocator type is "sticky" or not. Non-stickiness is denoted by having a true value for `propagate_on_container_move_assignment::value`, which we abbreviate to "`pocma`." (Actually, the standard says that `propagate_on_container_move_assignment` ought to be *exactly* the type `std::true_type`; and GNU's libstdc++ will hold you firmly to that requirement. So watch out when defining your own allocator types.) If the allocator type is non-sticky, then our most efficient course of action for move-assignment is to destroy our current value (if any)--making sure to use our old `m_allocator`--and then adopt the right-hand object's pointer along with its allocator. Because we adopt the allocator along with the pointer, we can be sure that we'll know how to deallocate the pointer down the road.

On the other hand, if our allocator type *is* "sticky," then we cannot adopt the allocator of the right-hand object. If our current ("stuck") allocator instance happens to be equal to the right-hand object's allocator instance, then we can adopt the right-hand object's pointer anyway; we already know how to deallocate pointers allocated by this particular allocator instance.

Finally, if we cannot adopt the right-hand object's allocator instance, and our current allocator instance isn't equal to the right-hand object's, then we cannot adopt the right-hand object's pointer--because at some point down the road we're going to have to free that pointer, and the only way to free that pointer is to use the right-hand object's allocator instance, and we're not allowed to adopt the right-hand object's allocator instance because our own instance is "stuck." In this case, we actually have to allocate a completely new pointer using our own allocator instance, and then copy over the data from `rhs.value()` to our own value by invoking the move constructor of `T`. This final case is the only one where we actually call the move constructor of `T`!

Copy assignment follows similar logic for the propagation of the right-hand allocator instance, except that it looks at the trait `propagate_on_container_copy_assignment`, or "`pocca`."

Swap is particularly interesting because its final case (when the allocator type is "sticky" and the allocator instances are unequal) requires extra allocations:

[PRE25]

On each of the two lines marked "might throw," we're calling the move assignment operator, which in this case might call `emplace`, which will ask the allocator for memory. If the underlying memory resource has been exhausted, then `Traits::allocate(m_allocator, 1)` might well throw an exception--and then we'd be in trouble, for two reasons. First, we've already started moving state around and deallocating old memory, and we might find it impossible to "unwind" back to a reasonable state. Second, and more importantly, `swap` is one of those functions that is so primitive and so fundamental that the standard library makes no provision for its failing--for example, the `std::swap` algorithm ([Chapter 3](part0036.html#12AK80-2fdac365b8984feebddfbb9250eaf20d), *The Iterator-Pair Algorithms*) is declared as `noexcept`, which means it *must* succeed; it is not allowed to throw an exception.

Thus, if allocation fails during our `noexcept` swap function, we'll see a `bad_alloc` exception percolate up through the call stack until it reaches our `noexcept` swap function declaration; at which point the C++ runtime will stop unwinding and call `std::terminate`, which (unless the programmer has altered its behavior via `std::set_terminate`) will cause our program to crash and burn.

The C++17 Standard goes several steps further than this in its specification of what *ought* to happen during the swapping of standard container types. First, instead of saying that allocation failure during `swap` will result in a call to `std::terminate`, the Standard simply says that allocation failure during `swap` will result in *undefined behavior*. Second, the Standard does not limit that undefined behavior to allocation failure! According to the C++17 Standard, merely *calling* `swap` on any standard library container instances whose allocators do not compare equally will result in undefined behavior, whether an allocation failure would have been encountered or not!

In fact, libc++ exploits this optimization opportunity to generate code for all standard container `swap` functions that looks roughly like this:

[PRE26]

Notice that if you use this code (as libc++ does) to `swap` two containers with unequal allocators, you'll wind up with a mismatch between pointers and their allocators, and then your program will probably crash--or worse--the next time you try to deallocate one of those pointers using the mismatched allocator. It is supremely important that you remember this pitfall when dealing with the C++17 "convenience" types such as `std::pmr::vector`!

[PRE27]

If your code design allows containers backed by different memory resources to be swapped with each other, then you must avoid `std::swap` and instead use this safe idiom:

[PRE28]

When I say "avoid `std::swap`," I mean "avoid any of the permutative algorithms in the STL," including such algorithms as `std::reverse` and `std::sort`. This would be quite an undertaking and I do not advise attempting it!

If your code design allows containers backed by different memory resources to be swapped with each other, then really, you *might* want to reconsider your design. If you can fix it so that you only ever swap containers that share the same memory resource, or if you can avoid stateful and/or sticky allocators entirely, then you will never need to think about this particular pitfall.

# Propagating downwards with scoped_allocator_adaptor

In the preceding section, we introduced `std::allocator_traits<A>::construct(a, ptr, args...)` and described it as a preferable alternative to the placement-`new` syntax `::new ((void*)ptr) T(args...)`. Now we'll see why the author of a particular allocator might want to give it different semantics.

One perhaps obvious way to change the semantics of `construct` for our own allocator type would be to make it trivially default-initialize primitive types instead of zero-initializing them. That code would look like this:

[PRE29]

Now you can use `std::vector<int, my_allocator<int>>` as a "vector-like" type satisfying all the usual invariants of `std::vector<int>`, except that when you implicitly create new elements via `v.resize(n)` or `v.emplace_back()`, the new elements are created uninitialized, just like stack variables, instead of being zero-initialized.

In a sense, what we've designed here is an "adaptor" that fits over the top of `std::allocator<T>` and modifies its behavior in an interesting way. It would be even better if we could modify or "adapt" any arbitrary allocator in the same way; to do that, we'd just change our `template<class T>` to `template<class A>` and inherit from `A` where the old code inherited from `std::allocator<T>`. Of course our new adaptor's template parameter list no longer starts with `T`, so we'd have to implement `rebind` ourselves; this path quickly gets into deep metaprogramming, so I won't digress to show it.

However, there's another useful way we could fiddle with the `construct` method for our own allocator type. Consider the following code sample, which creates a vector of vectors of `int`:

[PRE30]

Suppose we wanted to "stick" this container to a memory resource of our own devising, such as our favorite `WidgetAlloc`. We'd have to write something repetitive like this:

[PRE31]

Notice the repetition of the allocator object's initializer `&mr` at both levels. The need to repeat `&mr` makes it difficult to use our vector `vv` in generic contexts; for example, we can't easily pass it to a function template to populate it with data, because every time the callee would want to `emplace_back` a new vector-of-`int`, it would need to know the address `&mr` that is only known to the caller. What we'd like to do is wrap up and reify the notion that "every time you construct an element of the vector-of-vectors, you need to tack `&mr` onto the end of the argument list." And the standard library has us covered!

Since C++11, the standard library has provided (in the header named `<scoped_allocator>`) a class template called `scoped_allocator_adaptor<A>`. Just like our default-initializing "adaptor," `scoped_allocator_adaptor<A>` inherits from `A`, thus picking up all of `A`'s behaviors; and then it overrides the `construct` method to do something different. Namely, it attempts to figure out whether the `T` object it's currently constructing "uses an allocator," and if so, it will pass itself down as an extra argument to the constructor of `T`.

To decide whether type `T` "uses an allocator," `scoped_allocator_adaptor<A>::construct` defers to the type trait `std::uses_allocator_v<T,A>`, which (unless you've specialized it, which you probably shouldn't) will be true if and only if `A` is implicitly convertible to `T::allocator_type`. If `T` doesn't have an `allocator_type`, then the library will assume that `T` doesn't care about allocators, except in the special cases of `pair` and `tuple` (which all have special overloads of their constructors intended specifically to propagate allocators downward to their members) and in the special case of `promise` (which can allocate its shared state with an allocator even though it provides no way of referring to that allocator object afterward; we say that `promise`'s allocator support is "type-erased" even more thoroughly than the examples of type erasure we saw in [Chapter 5](part0074.html#26I9K0-2fdac365b8984feebddfbb9250eaf20d), *Vocabulary Types*).

For historical reasons, the constructors of allocator-aware types can follow either of two different patterns, and `scoped_allocator_adaptor` is smart enough to know them both. Older and simpler types (that is, everything except `tuple` and `promise`) tend to have constructors of the form `T(args..., A)` where the allocator `A` comes at the end. For `tuple` and `promise`, the standard library has introduced a new pattern: `T(std::allocator_arg, A, args...)` where the allocator `A` comes at the beginning but is preceded by the special tag value `std::allocator_arg`, whose sole purpose is to indicate that the next argument in the argument list represents an allocator, similarly to how the sole purpose of the tag `std::nullopt` is to indicate that an `optional` has no value (see [Chapter 5](part0074.html#26I9K0-2fdac365b8984feebddfbb9250eaf20d), *Vocabulary Types*). Just as the standard forbids creating the type `std::optional<std::nullopt_t>`, you will also find yourself in a world of trouble if you attempt to create `std::tuple<std::allocator_arg_t>`.

Using `scoped_allocator_adaptor`, we can rewrite our cumbersome example from earlier in a slightly less cumbersome way:

[PRE32]

Notice that the allocator type has gotten *more* cumbersome, but the important thing is that the `&mr` argument to `emplace_back` has disappeared; we can now use `vv` in contexts that expect to be able to push back elements in a natural way, without having to remember to add `&mr` all over the place. In our case, because we're using our `WidgetAlloc`, which is not default-constructible, the symptom of a forgotten `&mr` is a spew of compile-time errors. But you may recall from preceding sections in this chapter that `std::pmr::polymorphic_allocator<T>` will happily allow you to default-construct it, with potentially disastrous results; so if you are planning to use `polymorphic_allocator`, it might also be wise to look into `scoped_allocator_adaptor` just in order to limit the number of places in which you might forget to specify your allocation strategy.

# Propagating different allocators

In my introduction of `scoped_allocator_adaptor<A>`, I left out one more complication. The template parameter list isn't limited to just one allocator type argument! You can actually create a scoped-allocator type with multiple allocator type arguments, like this:

[PRE33]

Having set up these typedefs, we proceed to set up three distinct memory resources and construct an instance of `scoped_allocator_adaptor` capable of remembering all three of the memory resources (because it contains three distinct instances of `WidgetAlloc`, one per "level"):

[PRE34]

Finally, we can construct an instance of `OuterVector`, passing in our `scoped_allocator_adaptor` argument; and that's all! The overridden `construct` method hidden deep within our carefully crafted allocator type takes care of passing the argument `&bm` or `&bi` to any constructor that needs one of them:

[PRE35]

As you can see, a deeply nested `scoped_allocator_adaptor` is not for the faint of heart; and they're really only usable at all if you make a lot of "helper" typedefs along the way, as we did in this example.

One last note about `std::scoped_allocator_adaptor<A...>`: if the nesting of containers goes deeper than the number of allocator types in the template parameter list, then `scoped_allocator_adaptor` will act as if the last allocator type in its parameter list repeats forever. For example:

[PRE36]

We actually relied on this behavior in our very first `scoped_allocator_adaptor` example, the one involving `vv`, even though I didn't mention it at the time. Now that you know about it, you might want to go back and study that example to see where the "repeat forever" behavior is being used, and how you'd change that code if you wanted to use a different memory resource for the inner array of `int` than for the outer array of `InnerVector`.

# Summary

Allocators are a fundamentally arcane topic in C++, mainly for historical reasons. Several different interfaces, with different obscure use-cases, are piled one on top of the other; all of them involve intense metaprogramming; and vendor support for many of these features, even relatively old C++11 features such as fancy pointers, is still lacking.

C++17 offers the standard library type `std::pmr::memory_resource` to clarify the existing distinction between *memory resources* (a.k.a. *heaps*) and `allocators` (a.k.a. *handles* to heaps). Memory resources provide `allocate` and `deallocate` methods; allocators provide those methods as well as `construct` and `destroy`.

If you implement your own allocator type `A`, it must be a template; its first template parameter should be the type `T` that it expects to `allocate`. Your allocator type `A` must also have a templated constructor to support "rebinding" from `A<U>` to `A<T>`. Just like any other kind of pointer, an allocator type must support the `==` and `!=` operators.

A heap's `deallocate` method is allowed to require additional metadata attached to the incoming pointer. C++ handles this via *fancy pointers*. C++17's `std::pmr::memory_resource` does not support fancy pointers, but it's easy to implement your own.

Fancy pointer types must satisfy all the requirements of random access iterators, and must be nullable, and must be convertible to plain raw pointers. If you want to use your fancy pointer type with node-based containers such as `std::list`, you must give it a static `pointer_to` member function.

C++17 distinguishes between "sticky" and "non-sticky" allocator types. Stateless allocator types such as `std::allocator<T>` are non-sticky; stateful allocator types such as `std::pmr::polymorphic_allocator<T>` are sticky by default. Making your own allocator type of a non-default stickiness requires setting all three of the member typedefs familiarly known as "POCCA," "POCMA," and "POCS." Sticky allocator types such as `std::pmr::polymorphic_allocator<T>` are useful primarily--perhaps only--in classical object-oriented situations, where a container object is pinned to a particular memory address. Value-oriented programming (with lots of moves and swaps) calls for stateless allocator types, or else for everyone in the program to use the same heap and a single sticky but *effectively stateless* allocator type.

`scoped_allocator_adaptor<A...>` can help simplify the usage of deeply nested containers that use custom allocators or memory resources. Just about any deeply nested container using a non-default allocator type requires a lot of helper typedefs to remain even remotely readable.

Swapping two containers with unequal sticky allocators: in theory this invokes undefined behavior, and in practice it corrupts memory and segfaults. Don't do it!