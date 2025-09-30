# Chapter 9. Containers

In this chapter we will cover:

*   Comparing strings in an ultra-fast manner
*   Using an unordered set and map
*   Making a map, where value is also a key
*   Using multi-index containers
*   Getting the benefits of single-linked list and memory pool
*   Using flat associative containers

# Introduction

This chapter is devoted to the Boost containers and the things directly connected with them. This chapter provides information about the Boost classes that can be used in every day programming, and that will make your code much faster, and the development of new applications easier.

Containers differ not only by functionality, but also by the efficiency (complexity) of some of their members. The knowledge about complexities is essential for writing fast applications. This chapter doesn't just introduce some new containers to you; it gives you tips on when and when not to use a specific type of container or its methods.

So, let's begin!

# Comparing strings in an ultra-fast manner

It is a common task to manipulate strings. Here we'll see how the operation of string comparison can be done quickly using some simple tricks. This recipe is a trampoline for the next one, where the techniques described here will be used to achieve constant time-complexity searches.

So, we need to make a class that is capable of quickly comparing strings for equality. We'll make a template function to measure the speed of comparison:

[PRE0]

## Getting ready

This recipe requires only the basic knowledge of STL and C++.

## How to do it...

We'll make `std::string` a public field in our own class, and add all the comparison code to our class, without writing helper methods to work with stored `std::string`, as shown in the following steps:

1.  To do so, we'll need the following header:

    [PRE1]

2.  Now we can create our fast comparison class:

    [PRE2]

3.  Do not forget to define the equality comparison operators:

    [PRE3]

4.  And, that's it! Now we can run our tests and see the result using the following code:

    [PRE4]

## How it works...

The comparison of strings is slow because we are required to compare all the characters of the string one-by-one, if the strings are of equal length. Instead of doing that, we replace the comparison of strings with the comparison of integers. This is done via the hash function—the function that makes some short-fixed length representation of the string. Let us talk about the hash values on apples. Imagine that you have two apples with labels, as shown in the following diagram, and you wish to check that the apples are of the same cultivar. The simplest way to compare those apples is to compare them by labels. Otherwise you'll lose a lot of time comparing the apples based on the color, size, form, and other parameters. A hash is something like a label that reflects the value of the object.

![How it works...](img/4880OS_09_01.jpg)

So, let's move step-by-step.

In step 1, we include the header file that contains the definitions of the hash functions. In step 2, we declare our new string class that contains `str_`, which is the original value of the string and `comparison_`, which is the computed hash value. Note the construction:

[PRE5]

Here, `boost::hash<std::string>` is a structure, a functional object just like `std::negate<>`. That is why we need the first parenthesis—we construct that functional object. The second parenthesis with `s` inside is a call to `std::size_t operator()(const std::string& s)`, which will compute the hash value.

Now take a look at step 3 where we define `operator==`. Look at the following code:

[PRE6]

And, take additional care about the second part of the expression. The hashing operation loses information, which means that there is a possibility that more than one string produces exactly the same hash value. It means that if the hashes mismatch, there is a 100 percent guarantee that the strings will not match, otherwise we are required to compare the strings using the traditional methods.

Well, it's time to compare numbers. If we measure the execution time using the default comparison method, it will give us 819 milliseconds; however, our hashing comparison works almost two times faster and finishes in 475 milliseconds.

## There's more...

C++11 has the hash functional object, you may find it in the `<functional>` header in the `std::` namespace. You will know that the default Boost implementation of hash does not allocate additional memory and also does not have virtual functions. Hashing in Boost and STL is fast and reliable.

You can also specialize hashing for your own types. In Boost, it is done via specializing the `hash_value` function in the namespace of a custom type:

[PRE7]

This is different from STL specialization of `std::hash`, where you are required to make a template specialization of the `hash<>` structure in the `std::` namespace.

Hashing in Boost is defined for all the basic type arrays (such as `int`, `float`, `double`, and `char`), and for all the STL containers including `std::array`, `std::tuple`, and `std::type_index`. Some libraries also provide hash specializations, for example, `Boost.Variant` can hash any `boost::variant` class.

## See also

*   Read the *Using an unordered set and map* recipe for more information about the hash functions' usage.
*   The official documentation of `Boost.Functional/Hash` will tell you how to combine multiple hashes and provides more examples. Read about it at [http://www.boost.org/doc/libs/1_53_0/doc/html/hash.html](http://www.boost.org/doc/libs/1_53_0/doc/html/ha).

# Using an unordered set and map

In the previous recipe, we saw how string comparison can be optimized using hashing. After reading it, the following question may arise, "Can we make a container that will cache hashed values to use faster comparison?".

The answer is yes, and we can do much more. We can achieve almost constant time complexities for search, insertion, and removal of elements.

## Getting ready

Basic knowledge of C++ and STL containers is required. Reading the previous recipe will also help.

## How to do it...

This will be the simplest of all recipes:

1.  All you need to do is just include the `<boost/unordered_map.hpp>` header, if we wish to use maps or the `<boost/unordered_set.hpp>` header, if we wish to use sets.
2.  Now you are free to use `boost::unordered_map`, instead of `std::map` and `boost::unordered_set` instead of `std::set`:

    [PRE8]

## How it works...

Unordered containers store values and remember the hash of each value. Now if you wish to find a value in them, they will compute the hash of that value and search for that hash in the container. After the hash is found, the containers check for equality between the found value and the searched value. Then, the iterator to the value, or to the end of the container is returned.

Because the container can search for a constant width integral hash value, it may use some optimizations and algorithms suitable only for integers. Those algorithms guarantee constant search complexity O(1), when traditional `std::set` and `std::map` provide worse complexity O(log(N)), where N is the number of elements in the container. This leads us to a situation where the more elements in traditional `std::set` or `std::map` , the slower it works. However, the performance of unordered containers does not depend on the element count.

Such good performance never comes free of cost. In unordered containers, values are unordered (you are not surprised, are you?). It means that if we'll be outputting elements of containers from `begin()` to `end()`, as follows:

[PRE9]

We'll get the following output for `std::set` and `boost::unordered_set`:

[PRE10]

So, how much does the performance differ? Have a look at the following output:

[PRE11]

The performance was measured using the following code:

[PRE12]

Note that the code contains a lot of string constructions, so it is not 100 percent correct to measure the speedup using this test. It is here to show that unordered containers are usually faster than ordered ones.

Sometimes a task might arise where we need to use a user-defined type in unordered containers:

[PRE13]

To do that, we need to write a comparison operator for that type:

[PRE14]

Now, specialize the hashing function for that type. If the type consists of multiple fields, we usually just need to combine the hashes of all the fields that participate in equal comparison:

[PRE15]

### Note

It is highly recommended to combine hashes using the `boost::hash_combine` function.

## There's more...

Multiversions of containers are also available: `boost::unordered_multiset` is defined in the `<boost/unordered_set.hpp>` header, and `boost::unordered_multimap` is defined in the `<boost/unordered_map.hpp>` header. Just like in the case of STL, multiversions of containers are capable of storing multiple equal key values.

All the unordered containers allow you to specify your own hashing functor, instead of the default `boost::hash`. They also allow you to specialize your own equal comparison functor, instead of the default `std::equal_to`.

C++11 has all the unordered containers from Boost. You may find them in the headers: `<unordered_set>` and `<unordered_map>`, in the `std::` namespace, instead of `boost::`. The Boost and the STL versions have the same performance, and must work in the same way. However, Boost's unordered containers are available even on C++03 compilers, and make use of the rvalue reference emulation of `Boost.Move`, so you can use those containers for the move-only classes in C++03.

C++11 has no `hash_combine` function, so you will need to write your own:

[PRE16]

Or just use `boost::hash_combine`.

## See also

*   The recipe *Using the C++11 move emulation* in [Chapter 1](ch01.html "Chapter 1. Starting to Write Your Application"), *Starting to Write Your Application*, for more details on rvalue reference emulation of `Boost.Move`
*   More information about the unordered containers is available on the official site at [http://www.boost.org/doc/libs/1_53_0/doc/html/unordered.html](http://www.boost.org/doc/libs/1_53_0/doc/html/unordered.html)
*   More information about combining hashes and computing hashes for ranges is available at [http://www.boost.org/doc/libs/1_53_0/do](http://www.boost.org/doc/libs/1_53_0/do)[c/html/hash.html](http://c/html/hash.html)

# Making a map, where value is also a key

Several times in a year, we need something that can store and index a pair of values. Moreover, we need to get the first part of the pair using the second, and get the second part using the first. Confused? Let me show you an example. We are creating a vocabulary class, wherein when the users put values into it, the class must return identifiers and when the users put identifiers into it, the class must return values.

To be more practical, users will be entering login names into our vocabulary, and wish to get the unique identifier of a person. They will also wish to get all the persons' names using identifiers.

Let's see how it can be implemented using Boost.

## Getting ready

Basic knowledge of STL and templates are required for this recipe.

## How to do it...

This recipe is about the abilities of the `Boost.Bimap` library. Let's see how it can be used to implement this task:

1.  We'll need the following includes:

    [PRE17]

2.  Now we are ready to make our vocabulary structure:

    [PRE18]

3.  It can be filled using the following syntax:

    [PRE19]

4.  We can work with the left part of bimap just like with a map:

    [PRE20]

5.  The right part of bimap is almost the same as the left:

    [PRE21]

6.  We also need to ensure that there is such a person in the vocabulary:

    [PRE22]

7.  That's it. Now, if we put all the code (except includes) inside `int main()`, we'll get the following output:

    [PRE23]

## How it works...

In step 2, we define the `bimap` type:

[PRE24]

The first template parameter tells that the first key must have type `std::string`, and should work as `std::set`. The second template parameter tells that the second key must have type `std::size_t`. Multiple first keys can have a single second key value, just like in `std::multimap`.

We can specify the underlying behavior of `bimap` using classes from the `boost::bimaps::` namespace. We can use hash map as an underlying type for the first key:

[PRE25]

When we do not specify the behavior of the key, and just specify its type, `Boost.Bimap` uses `boost::bimaps::set_of` as a default behavior. Just like in our example, we can try to express the following code using STL:

[PRE26]

Using STL it would look like a combination of the following two variables:

[PRE27]

As we can see from the preceding comments, a call to `name_id.left` (in step 4) will return a reference to something with an interface close to `std::map<std::string, std::size_t>`. A call to `name_id.right` from step 5 will return something with an interface close to `std::multimap<std::size_t, std::string>`.

In step 6, we work with a whole `bimap`, searching for a pair of keys, and making sure that they are in the container.

## There's more...

Unfortunately, C++11 has nothing close to `Boost.Bimap`. Here we have some other bad news: `Boost.Bimap` does not support rvalue references, and on some compilers, insane numbers of warnings will be shown. Refer to your compiler's documentation to get the information about suppressing specific warnings.

The good news is that `Boost.Bimap` usually uses less memory than two STL containers, and makes searches as fast as STL containers. It has no virtual function calls inside, but does use dynamic allocations.

## See also

*   The next recipe, *Using multi-index containers*, will give you more information about multi-indexing, and about the Boost library that can be used instead of `Boost.Bimap`
*   Read the official documentation for more examples and information about `bimap` at [http://www.boost.org/doc/libs/1_53_0/libs/bimap/doc/html/index.html](http://www.boost.org/doc/libs/1_53_0/libs/bimap/doc/html/index.html)

# Using multi-index containers

In the previous recipe, we made some kind of vocabulary, which is good when we need to work with pairs. But, what if we need much more advanced indexing? Let's make a program that indexes persons:

[PRE28]

We will need a lot of indexes; for example, by name, ID, height, and weight.

## Getting ready

Basic knowledge of STL containers and unordered maps is required.

## How to do it...

All the indexes can be constructed and managed by a single `Boost.Multiindex` container.

1.  To do so, we will need a lot of includes:

    [PRE29]

2.  The hardest part is to construct the multi-index type:

    [PRE30]

3.  Now we may insert values into our multi-index:

    [PRE31]

4.  Let's construct a function for printing the index content:

    [PRE32]

5.  Print all the indexes as follows:

    [PRE33]

6.  Some code from the previous recipe can also be used:

    [PRE34]

7.  Now if we run our example, it will output the content of the indexes:

    [PRE35]

## How it works...

The hardest part here is the construction of a multi-index type using `boost::multi_index::multi_index_container`. The first template parameter is a class that we are going to index. In our case, it is `person`. The second parameter is a type `boost::multi_index::indexed_by`, all the indexes must be described as a template parameter of that class.

Now, let's take a look at the first index description:

[PRE36]

The usage of the `boost::multi_index::ordered_unique` class means that the index must work like `std::set`, and have all of its members. The `boost::multi_index::identity<person>` class means that the index will use the `operator <` of a `person` class for orderings.

The next table shows the relation between the `Boost.MultiIndex` types and the STL containers:

| The Boost.MultiIndex types | STL containers |
| --- | --- |
| `boost::multi_index::ordered_unique` | `std::set` |
| `boost::multi_index::ordered_non_unique` | `std::multiset` |
| `boost::multi_index::hashed_unique` | `std::unordered_set` |
| `boost::multi_index::hashed_non_unique` | `std::unordered_mutiset` |
| `boost::multi_index::sequenced` | `std::list` |

Let's take a look at the second index:

[PRE37]

The `boost::multi_index::hashed_non_unique` type means that the index will work like `std::set`, and `boost::multi_index::member<person, std::size_t, &person::id_>` means that the index will apply the hash function only to a single member field of the person structure, to `person::id_`.

The remaining indexes won't be a trouble now, so let's take a look at the usage of indexes in the print function instead. Getting the type of iterator for a specific index is done using the following code:

[PRE38]

This looks slightly overcomplicated because `Indexes` is a template parameter. The example would be simpler, if we could write this code in the scope of `indexes_t`:

[PRE39]

The `nth_index` member metafunction takes a zero-based number of index to use. In our example, index 1 is the index of IDs, index 2 is the index of heights and so on.

Now, let's take a look at how to use `const_iterator_t`:

[PRE40]

This can also be simplified for `indexes_t` being in scope:

[PRE41]

The function `get<indexNo>()` returns index. We can use that index almost like an STL container.

## There's more...

C++11 has no multi-index library. The `Boost.MultiIndex` library is a fast library that uses no virtual functions. The official documentation of `Boost.MultiIndex` contains performance and memory usage measures, showing that this library in most cases uses less memory than STL-based handwritten code. Unfortunately, `boost::multi_index::multi_index_container` does not support C++11 features, and also has no rvalue references emulation using `Boost.Move`.

## See also

*   The official documentation of `Boost.MultiIndex` contains tutorials, performance measures, examples, and other `Boost.Multiindex` libraries' description of useful features. Read about it at [http://www.boost.org/doc/libs/1_53_0/libs/multi_index/doc/index.html](http://www.boost.org/doc/libs/1_53_0/libs/multi_index/doc/index.html).

# Getting the benefits of single-linked list and memory pool

Nowadays, we usually use `std::vector` when we need nonassociative and nonordered containers. This is recommended by *Andrei Alexandrescu* and *Herb Sutter* in the book *C++ Coding Standards*, and even those users who did not read the book usually use `std::vector`. Why? Well, `std::list` is slower, and uses much more resources than `std::vector`. The `std::deque` container is very close to `std::vector` , but stores values noncontinuously.

Everything is good until we do not need a container; however, if we need a container, erasing and inserting elements does not invalidate iterators. Then we are forced to choose the slower `std::list`.

But wait, there is a good solution in Boost for such cases!

## Getting ready

Good knowledge of STL containers is required to understand the introductory part. After that, only basic knowledge of C++ and STL containers is required.

## How to do it...

In this recipe, we'll be using two Boost libraries at the same time: `Boost.Pool` and single-linked list from `Boost.Container`.

1.  We'll need the following headers:

    [PRE42]

2.  Now we need to describe the type of our list. This can be done as shown in the following code:

    [PRE43]

3.  We can work with our single-linked list like with `std::list`. Take a look at the function that is used to measure the speed of both the list types:

    [PRE44]

4.  Features specific for each type of list are moved to `list_specific` functions:

    [PRE45]

## How it works...

When we are using `std::list`, we may notice a slowdown because each node of the list needs a separate allocation. It means that usually when we insert 10 elements into `std::list`, the container calls new 10 times.

That is why we used boost`::fast_pool_allocator<int>` from `Boost.Pool`. This allocator tries to allocate bigger blocks of memory, so that at a later stage, multiple nodes can be constructed without any calls to allocate new ones.

The `Boost.Pool` library has a drawback—it uses memory for internal needs. Usually, an additional `sizeof` pointer is used per element. To workaround that issue, we are using a single linked list from `Boost.Containers`.

The `boost::container::slist` class is more compact, but its iterators can iterate only forward. Step 3 will be trivial for those readers who are aware of STL containers, so we move to step 4 to see some `boost::container::slist` specific features. Since the single-linked list iterator could iterate only forward, traditional algorithms of insertion and deletion will take linear time O(N). That's because when we are erasing or inserting, the previous element must be modified to point at new elements of the list. To workaround that issue, the single-linked list has the methods `erase_after` and `insert_after` that work for constant time O(1). These methods insert or erase elements right after the current position of the iterator.

### Note

However, erasing and inserting values at the beginning of single-linked lists makes no big difference.

Take a careful look at the following code:

[PRE46]

It is required because `boost::fast_pool_allocator` does not free memory, so we must do it by hand. The *Doing something at scope exit* recipe from [Chapter 3](ch03.html "Chapter 3. Managing Resources"), *Managing Resources*, will be a help in freeing `Boost.Pool`.

Let's take a look at the execution results to see the difference:

[PRE47]

As we can see, `slist_t` uses half the memory, and is twice as fast compared to the `std::list` class.

## There's more...

C++11 has `std::forward_list`, which is very close to `boost::containers::slist`. It also has the `*_after` methods, but has no `size()` method. They have the same performance and neither of them have virtual functions, so these containers are fast and reliable. However, the Boost version is also usable on C++03 compilers, and even has support for rvalue references emulation via `Boost.Move`.

Pools are not part of C++11\. Use the version from Boost; it is fast and does not use virtual functions.

### Note

Guessing why `boost::fast_pool_allocator` does not free the memory by itself? That's because C++03 has no stateful allocators, so the containers are not copying and storing allocators. That makes it impossible to implement a `boost::fast_pool_allocator` function that deallocates memory by itself.

## See also

*   The official documentation of `Boost.Pool` contains more examples and classes to work with memory pools. Read about it at [http://www.boost.org/doc/libs/1_53_0/libs/pool/doc/html/index.html](http://www.boost.org/doc/libs/1_53_0/libs/pool/doc/html/index.html).
*   The *Using flat associative containers* recipe will introduce you to some more classes from `Boost.Container`. You can also read the official documentation of `Boost.Container` to study that library by yourself, or get full reference documentation of its classes at [http://www.boost.org/doc/libs/1_53_0/doc/html/container.html](http://www.boost.org/doc/libs/1_53_0/doc/html/container.html).
*   Read about why stateful allocators may be required at [http://www.boost.org/doc/libs/1_53_0/doc/html/interprocess/allocators_containers.html#interprocess.allocators_containers.allocator_introduction](http://www.boost.org/doc/libs/1_53_0/doc/html/interprocess/allocators_containers.html#interprocess.allocators_containers.allocator_introduction).
*   *Vector vs List*, and other interesting topics from *Bjarne Stroustrup*, the inventor of the C++ programming language, can be found at [http://channel9.msdn.com/Events/GoingNative/GoingNative-2012/Keynote-Bjarne-Stroustrup](http://channel9.msdn.com/Events/GoingNative/GoingNative-2012/Keynote-Bjarne-Stroustrup)[-Cpp11-Style](http://-Cpp11-Style).

# Using flat associative containers

After reading the previous recipe, some of the readers may start using fast pool allocators everywhere; especially, for `std::set` and `std::map`. Well, I'm not going to stop you from doing that, but let's at least take a look at an alternative: flat associative containers. These containers are implemented on top of the traditional vector container and store the values ordered.

## Getting ready

Basic knowledge of STL associative containers is required.

## How to do it...

The flat containers are part of the `Boost.Container` library. We already saw how to use some of its containers in the previous recipes. In this recipe we'll be using a `flat_set` associative container:

1.  We'll need to include only a single header file:

    [PRE48]

2.  After that, we are free to construct the flat container:

    [PRE49]

3.  Reserving space for elements:

    [PRE50]

4.  Filling the container:

    [PRE51]

5.  Now we can work with it just like with `std::set`:

    [PRE52]

## How it works...

Steps 1 and 2 are trivial, but step 3 requires attention. It is one of the most important steps while working with flat associative containers and `std::vector`.

The `boost::container::flat_set` class stores its values ordered in vector, which means that any insertion or deletion of elements takes linear time O(N), just like in case of `std::vector`. This is a necessary evil. But for that, we gain almost three times less memory usage per element, more processor cache friendly storage, and random access iterators. Take a look at step 5, `5.1`, where we were getting the distance between two iterators returned by calls to the `lower_bound` member functions. Getting distance with a flat set takes constant time O(1), while the same operation on iterators of `std::set` takes linear time O(N). In the case of `5.1`, getting the distance using `std::set` would be 400 times slower than getting the distance for flat set containers.

Back to step 3\. Without reserving memory, insertion of elements can become at times slower and less memory efficient. The `std::vector` class allocates the required chunk of memory and the in-place construct elements on that chunk. When we insert some element without reserving the memory, there is a chance that there is no free space remaining on the preallocated chunk of memory, so `std::vector` will allocate twice the chunk of memory that was allocated previously. After that, `std::vector` will copy or move elements from the first chunk to the second, delete elements of the first chunk, and deallocate the first chunk. Only after that, insertion will occur. Such copying and deallocation may occur multiple times during insertions, dramatically reducing the speed.

### Note

If you know the count of elements that `std::vector` or any flat container must store, reserve the space for those elements before insertion. There are no exceptions from that rule!

Step 4 is trivial, we are inserting elements here. Note that we are inserting ordered elements. This is not required, but recommended to speedup insertion. Inserting elements at the end of `std::vector` is much more cheaper than in the middle or at the beginning.

In step 5, `5.2` and `5.3` do not differ much, except of their execution speed. Rules for erasing elements are pretty much the same as for inserting them, so see the preceding paragraph for explanations.

### Note

Maybe I'm telling you trivial things about containers, but I have seen some very popular products that use features of C++11, have an insane amount of optimizations and lame usage of STL containers, especially `std::vector`.

In step 5, `5.4` shows you that the `std::lower_bound` function will work faster with `boost::container::flat_set` than with `std::set`, because of random access iterators.

In step 5, `5.5` also shows you the benefit of random access iterators. Note that we did not use the `std::find` function here. This is because that function takes linear time O(N), while the member `find` functions take logarithmic time O(log(N)).

## There's more...

When should we use flat containers, and when should we use usual ones? Well, it's up to you, but here is a list of differences from the official documentation of `Boost.Container` that will help you to decide:

*   Faster lookup than standard associative containers
*   Much faster iteration than standard associative containers
*   Less memory consumption for small objects (and for large objects if `shrink_to_fit` is used)
*   Improved cache performance (data is stored in contiguous memory)
*   Nonstable iterators (iterators are invalidated when inserting and erasing elements)
*   Non-copyable and non-movable value types can't be stored
*   Weaker exception safety than standard associative containers (copy/move constructors can throw an exception when shifting values in erasures and insertions)
*   Slower insertion and erasure than standard associative containers (specially for non-movable types)

C++11 unfortunately has no flat containers. Flat containers from Boost are fast, have a lot of optimizations, and do not use virtual functions. Classes from `Boost.Containers` have support of rvalue reference emulation via `Boost.Move` so you are free to use them even on C++03 compilers.

## See also

*   Refer to the *Getting the benefits of single-linked list and memory pool* recipe for more information about `Boost.Container`.
*   The recipe *Using the C++11 move emulation* in [Chapter 1](ch01.html "Chapter 1. Starting to Write Your Application"), *Starting to Write Your Application*, will give you the basics of emulation rvalue references on C++03 compatible compilers.
*   The official documentation of `Boost.Container` contains a lot of useful information about `Boost.Container` and full reference of each class. Read about it at [http://www.boost.org/doc/libs/1_53_0/doc/html/container.html](http://www.boost.org/doc/libs/1_53_0/doc/html/container.html).