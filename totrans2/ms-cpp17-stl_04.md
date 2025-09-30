# The Container Zoo

In the previous two chapters, we introduced the ideas of *iterators* and *ranges* ([Chapter 2](part0026.html#OPEK0-2fdac365b8984feebddfbb9250eaf20d), *Iterators and Ranges*) and the vast library of standard *generic algorithms* that operate on ranges of data elements defined by pairs of those iterators ([Chapter 3](part0036.html#12AK80-2fdac365b8984feebddfbb9250eaf20d), *The Iterator-Pair Algorithms*). In this chapter, we'll look at where those data elements themselves are allocated and stored. That is, now that we know all about how to iterate, the question gains urgency: what is it that we are iterating *over?*

In the Standard Template Library, the answer to that question is generally: We are iterating over some sub-range of the elements contained in a *container*. A container is simply a C++ class (or class template) which, by its nature, *contains* (or *owns*) a homogeneous range of data elements, and exposes that range for iteration by generic algorithms.

Topics we will cover in this chapter are:

*   The notion of one object *owning* another (this being the essential difference between a *container* and a *range*)
*   The sequence containers (`array`, `vector`, `list`, and `forward_list`)
*   The pitfalls of iterator invalidation and reference invalidation
*   The container adaptors (`stack`, `queue`, and `priority_queue`)
*   The associative containers (`set`, `map`, and friends)
*   When it is appropriate to provide a *comparator*, *hash function*, *equality comparator*, or *allocator* as additional template type parameters

# The notion of ownership

When we say that object `A` *owns* object `B`, what we mean is that object `A` manages the lifetime of object `B`--that `A` controls the construction, copying, moving, and destruction of object `B`. The user of object `A` can (and should) "forget about" managing `B` (for example, via explicit calls to `delete B`, `fclose(B)`, and so on).

The simplest way for an object `A` to "own" an object `B` is for `B` to be a member variable of `A`. For example:

[PRE0]

Another way is for `A` to hold a pointer to `B`, with the appropriate code in `~A()` (and, if necessary, in the copy and move operations of `A`) to clean up the resources associated with that pointer:

[PRE1]

The notion of *ownership* is tightly bound up with the C++-specific catchphrase **Resource Allocation Is Initialization**, which you will often see abbreviated as **RAII**. (That cumbersome abbreviation should properly have been more like "Resource Freeing Is Destruction", but that acronym was taken.)

The goal of the standard *container classes* is to provide access to a particular bunch of data objects `B`, while making sure that the *ownership* of those objects is always clear--namely, a container always has ownership of its data elements. (Contrariwise, an *iterator*, or a pair of iterators defining a *range*, never owns its data elements; we saw in [Chapter 3](part0036.html#12AK80-2fdac365b8984feebddfbb9250eaf20d), *The Iterator-Pair Algorithms*, that the standard iterator-based algorithms such as `std::remove_if` never actually deallocate any elements, but instead simply permute the values of the elements in various ways.)

In the remainder of this chapter, we'll explore the various standard container classes.

# The simplest container: std::array<T, N>

The simplest standard container class is `std::array<T, N>`, which behaves just like a built-in ("C-style") array. The first template parameter to `std::array` indicates the type of the array's elements, and the second template parameter indicates the number of elements in the array. This is one of the very few places in the standard library where a template parameter is an integer value instead of the name of a type.

![](img/00006.jpeg)

Normal C-style arrays, being part of the core language (and a part that dates back to the 1970s, at that!), do not provide any built-in operations that would take linear time to run. C-style arrays let you index into them with `operator[]`, and compare their addresses, since those operations can be done in constant time; but if you want to assign the entire contents of one C-style array to another, or compare the contents of two arrays, you'll find that you can't do it straightforwardly. You'll have to use some of the standard algorithms we discussed in [Chapter 3](part0036.html#12AK80-2fdac365b8984feebddfbb9250eaf20d), *The Iterator-Pair Algorithms*, such as `std::copy` or `std::equal` (the function template `std::swap`, being an "algorithm" already, *does* work for C-style arrays. It would be a shame if it didn't work.):

[PRE2]

`std::array` behaves just like a C-style array, but with more syntactic sugar. It offers `.begin()` and `.end()` member functions; and it overloads the operators `=`, `==`, and `<` to do the natural things. All of these operations still take time linear in the size of the array, because they have to walk through the array copying (or swapping or comparing) each individual element one at a time.

One gripe about `std::array`, which you'll see recurring for a few of these standard container classes, is that when you construct a `std::array` with an initializer list inside a set of curly braces, you actually need to write *two* sets of curly braces. That's one set for the "outer object" of type `std::array<T, N>`, and another set for the "inner data member" of type `T[N]`. This is a bit annoying at first, but the double-brace syntax will quickly become second nature once you have used it a few times:

[PRE3]

One other benefit of `std::array` is that you can return one from a function, which you can't do with C-style arrays:

[PRE4]

Because `std::array` has a copy constructor and a copy assignment operator, you can also store them in containers: for example, `std::vector<std::array<int, 3>>` is fine whereas `std::vector<int[3]>` wouldn't work.

However, if you find yourself returning arrays from functions or storing arrays in containers very often, you should consider whether "array" is really the right abstraction for your purposes. Would it be more appropriate to wrap that array up into some kind of class type?

In the case of our `cross_product` example, it turns out to be an extremely good idea to encapsulate our "array of three integers" in a class type. Not only does this allow us to name the members (`x`, `y`, and `z`), but we can also initialize objects of the `Vec3` class type more easily (no second pair of curly braces!) and perhaps most importantly for our future sanity, we can avoid defining the comparison operators such as `operator<` which don't actually make sense for our mathematical domain. Using `std::array`, we have to deal with the fact that the array `{1, 2, 3}` compares "less than" the array `{1, 3, -9}`--but when we define our own `class Vec3`, we can simply omit any mention of `operator<` and thus ensure that nobody will ever accidentally misuse it in a mathematical context:

[PRE5]

`std::array` holds its elements inside itself. Therefore, `sizeof (std::array<int, 100>)` is equal to `sizeof (int[100])`, which is equal to `100 * sizeof (int)`. Don't make the mistake of trying to place a gigantic array on the stack as a local variable!

[PRE6]

Working with "gigantic arrays" is a job for the next container on our list: `std::vector`.

# The workhorse: std::vector<T>

`std::vector` represents a contiguous array of data elements, but allocated on the heap instead of on the stack. This improves on `std::array` in two ways: First, it allows us to create a really gigantic array without blowing our stack. Second, it allows us to resize the underlying array dynamically--unlike `std::array<int, 3>` where the size of the array is an immutable part of the type, a `std::vector<int>` has no intrinsic size. A vector's `.size()` method actually yields useful information about the current state of the vector.

A `std::vector` has one other salient attribute: its *capacity*. The capacity of a vector is always at least as large as its size, and represents the number of elements that the vector currently *could* hold, before it would need to reallocate its underlying array:

![](img/00007.jpeg)

Other than its resizeability, `vector` behaves similarly to `array`. Like arrays, vectors are copyable (copying all their data elements, in linear time) and comparable (`std::vector<T>::operator<` will report the lexicographical order of the operands by delegating to `T::operator<`).

Generally speaking, `std::vector` is the most commonly used container in the entire standard library. Any time you need to store a "lot" of elements (or "I'm not sure how many elements I have"), your first thought should always be to use a `vector`. Why? Because `vector` gives you all the flexibility of a resizeable container, with all the simplicity and efficiency of a contiguous array.

Contiguous arrays are the most efficient data structures (on typical hardware) because they provide good *locality*, also known as `cache-friendliness`. When you're traversing a vector in order from its `.begin()` to its `.end()`, you're also traversing *memory* in order, which means that the computer's hardware can predict with very high accuracy the next piece of memory you're going to look at. Compare this to a linked list, in which traversing from `.begin()` to `.end()` might well involve following pointers all over the address space, and accessing memory locations in no sensible order. With a linked list, pretty much every address you hit will be unrelated to the previous one, and so none of them will be in the CPU's cache. With a vector (or array), the opposite is true: every address you hit will be related to the previous one by a simple linear relationship, and the CPU will be able to have the values all ready and waiting for you by the time you need them.

Even if your data is "more structured" than a simple list of values, you can often get away with using a `vector` to store it. We'll see near the end of this chapter how you can use `vector` to simulate a stack or a priority queue.

# Resizing a std::vector

`std::vector` has a whole family of member functions concerned with adding and deleting elements. These member functions aren't present in `std::array` because `std::array` isn't resizable; but they *are* present in most of the other containers we're going to be talking about in this chapter. So it's a good idea to get familiar with them now.

Let's start with the two primitive operations specific to `vector` itself: `.resize()` and `.reserve()`.

`vec.reserve(c)` updates the capacity of the vector--it "reserves" space for as many as `c` elements (total) in the underlying array. If `c <= vec.capacity()` then nothing happens; but if `c > vec.capacity()` then the vector will have to reallocate its underlying array. Reallocation follows an algorithm equivalent to the following:

[PRE7]

If you've been reading this book in order, you might recognize that the crucial for-loop in this `.reserve()` function closely resembles the implementation of `std::uninitialized_copy(a,b,c)` from [Chapter 3](part0036.html#12AK80-2fdac365b8984feebddfbb9250eaf20d), *The Iterator-Pair Algorithms*. Indeed, if you were implementing `.reserve()` on a container that was not allocator-aware (see [Chapter 8](part0129.html#3R0OI0-2fdac365b8984feebddfbb9250eaf20d), *Allocators*), you might reuse that standard algorithm:

[PRE8]

`vec.resize(s)` changes the size of the vector--it chops elements off the end of the vector (calling their destructors in the process), or adds additional elements to the vector (default-constructing them), until the size of the vector is equal to `s`. If `s > vec.capacity()`, then the vector will have to reallocate its underlying array, just as in the `.reserve()` case.

You may have noticed that when a vector reallocates its underlying array, the elements change addresses: the address of `vec[0]` before the reallocation is different from the address of `vec[0]` after the reallocation. Any pointers that pointed to the vector's old elements become "dangling pointers." And since `std::vector::iterator` is essentially just a pointer as well, any *iterators* that pointed to the vector's old elements become invalid as well. This phenomenon is called *iterator invalidation*, and it is a major source of bugs in C++ code. Watch out when you're dealing with iterators and resizing vectors at the same time!

Here are some classic cases of iterator invalidation:

[PRE9]

And here's another case, familiar from many other programming languages as well, in which erasing elements from a container while iterating over it produces subtle bugs:

[PRE10]

# Inserting and erasing in a std::vector

`vec.push_back(t)` adds an item to the end of the vector. There is no corresponding `.push_front()` member function, because as you can see from the diagram at the start of this section, there's no efficient way to push anything onto the *front* of a vector.

`vec.emplace_back(args...)` is a perfect-forwarding variadic function template that acts just like `.push_back(t)`, except that, instead of placing a copy of `t` at the end of the vector, it places a `T` object constructed as if by `T(args...)`.

Both `push_back` and `emplace_back` have what is called "amortized constant time" performance. To see what this means, consider what would happen to a naive vector if you call `v.emplace_back()` a hundred times in a row. With each call, the vector needs to get just a little bit bigger; so it reallocates its underlying array and moves all `v.size()` elements from the old array to the new one. Soon you'd be spending more time copying old data from place to place than you're spending actually "pushing back" new data! Fortunately, `std::vector` is smart enough to avoid this trap. Whenever an operation such as `v.emplace_back()` causes reallocation, the vector won't make room for just `capacity() + 1` elements in the new array; it will make room for `k * capacity()` elements (where `k` is 2 for libc++ and libstdc++, and approximately 1.5 for Visual Studio). So, although reallocation gets more and more expensive as the vector grows, you do fewer and fewer reallocations per `push_back`--and so the cost of a single `push_back` is constant, *on average*. This trick is known as *geometric resizing*.

`vec.insert(it, t)` adds an item into the middle of the vector, at the position indicated by the iterator `it`. If `it == vec.end()`, then this is equivalent to `push_back`; if `it == vec.begin()`, then this is a poor man's version of `push_front`. Notice that, if you insert anywhere but the end of the vector, all the elements after the insertion point in the underlying array will get shifted over to make room; this can be expensive.

There are several different overloads of `.insert()`. Generally speaking, none of these will be useful to you, but you might want to be aware of them in order to interpret the cryptic error messages (or cryptic runtime bugs) that will show up if you accidentally provide the wrong arguments to `.insert()` and overload resolution ends up picking one of these instead of the one you expected:

[PRE11]

`vec.emplace(it, args...)` is to `insert` as `emplace_back` is to `push_back`: it's a perfect-forwarding version of the C++03 function. Prefer `emplace` and `emplace_back` over `insert` and `push_back`, when possible.

`vec.erase(it)` erases a single item from the middle of a vector, at the position indicated by the iterator `it`. There's also a two-iterator version, `vec.erase(it, it)`, which erases a contiguous range of items. Notice that this two-iterator version is the same one we used in the *erase-remove idiom* in the previous chapter.

To delete just the last element from the vector, you could use either `vec.erase(vec.end()-1)` or `vec.erase(vec.end()-1, vec.end())`; but since this is a common operation, the standard library provides a synonym in the form of `vec.pop_back()`. You can implement a dynamically growable *stack* using nothing more than the `push_back()` and `pop_back()` methods of `std::vector`.

# Pitfalls with vector<bool>

The `std::vector` template has one special case: `std::vector<bool>`. Since the `bool` datatype has only two possible values, the values of eight bools can be packed into a single byte. `std::vector<bool>` uses this optimization, which means that it uses eight times less heap-allocated memory than you might naturally expect.

![](img/00008.jpeg)

The downside of this packing is that the return type of `vector<bool>::operator[]` cannot be `bool&`, because the vector doesn't store actual `bool` objects anywhere. Therefore, `operator[]` returns a customized class type, `std::vector<bool>::reference`, which is convertible to `bool` but which is not, itself, a `bool` (types like this are often called "proxy types" or "proxy references").

The result type of `operator[] const` is "officially" `bool`, but in practice, some libraries (notably libc++) return a proxy type for `operator[] const`. This means that code using `vector<bool>` is not only subtle but sometimes non-portable as well; I advise avoiding `vector<bool>` if you can:

[PRE12]

# Pitfalls with non-noexcept move constructors

Recall the implementation of `vector::resize()` from section *Resizing a std::vector*. When the vector resizes, it reallocates its underlying array and moves its elements into the new array--unless the element type is not "nothrow move-constructible," in which case it *copies* its elements! What this means is that resizing a vector of your own class type will be unnecessarily "pessimized" unless you go out of your way to specify that your move constructor is `noexcept`.

Consider the following class definitions:

[PRE13]

We can test the behavior of these classes in isolation using a test harness such as the following. Running `test()` will print "copy Bad--move Good--copy Bad--move Good." What an appropriate mantra!

[PRE14]

This is a subtle and arcane point, but it can have a major effect on the efficiency of your C++ code in practice. A good rule of thumb is: Whenever you declare your own move constructor or swap function, make sure you declare it `noexcept`.

# The speedy hybrid: std::deque<T>

Like `std::vector`, `std::deque` presents the interface of a contiguous array--it is random-access, and its elements are stored in contiguous blocks for cache-friendliness. But unlike `vector`, its elements are only "chunkwise" contiguous. A single deque is made up of an arbitrary number of "chunks," each containing a fixed number of elements. To insert more elements on either end of the container is cheap; to insert elements in the middle is still expensive. In memory it looks something like this:

![](img/00009.jpeg)

`std::deque<T>` exposes all the same member functions as `std::vector<T>`, including an overloaded `operator[]`. In addition to vector's `push_back` and `pop_back` methods, `deque` exposes an efficient `push_front` and `pop_front`.

Notice that, when you repeatedly `push_back` into a vector, you eventually trigger a reallocation of the underlying array and invalidate all your iterators and all your pointers and references to elements within the container. With `deque`, iterator invalidation still happens, but individual elements never change their addresses unless you insert or erase elements in the middle of the deque (in which case one end of the deque or the other will have to shift outward to make room, or shift inward to fill the gap):

[PRE15]

Another advantage of `std::deque<T>` is that there is no specialization for `std::deque<bool>`; the container presents a uniform public interface no matter what `T` is.

The disadvantage of `std::deque<T>` is that its iterators are significantly more expensive to increment and dereference, since they have to navigate the array of pointers depicted in the following diagram. This is a significant enough disadvantage that it makes sense to stick with `vector`, unless you happen to need quick insertion and deletion at both ends of the container.

# A particular set of skills: std::list<T>

The container `std::list<T>` represents a linked list in memory. Schematically, it looks like this:

![](img/00010.jpeg)

Notice that each node in the list contains pointers to its "next" and "previous" nodes, so this is a doubly linked list. The benefit of a doubly linked list is that its iterators can move both forwards and backwards through the list--that is, `std::list<T>::iterator` is a *bidirectional iterator* (but it is not *random-access*; getting to the *n*th element of the list still requires O(*n*) time).

`std::list` supports many of the same operations as `std::vector`, except for those operations that require random access (such as `operator[]`). It can afford to add member functions for pushing and popping from the front of the list, since pushing and popping from a `list` doesn't require expensive move operations.

In general, `std::list` is much less performant than a contiguous data structure such as `std::vector` or `std::deque`, because following pointers to "randomly" allocated addresses is much harder on the cache than following pointers into a contiguous block of memory. Therefore, you should treat `std::list` as a generally *undesirable* container; you should only pull it out of your toolbox when you absolutely need one of the things it does *better* than `vector`.

# What are the special skills of std::list?

First, there's no *iterator invalidation* for lists! `lst.push_back(v)` and `lst.push_front(v)` always operate in constant time, and don't ever need to "resize" or "move" any data.

Second, many mutating operations that would be expensive on `vector` or require out-of-line storage ("scratch space") become cheap for linked lists. Here are some examples:

`lst.splice(it, otherlst)` "splices" the entirety of `otherlst` into `lst`, as if by repeated calls to `lst.insert(it++, other_elt)`; except that the "inserted" nodes are actually stolen from the right-hand `otherlst`. The entire splicing operation can be done with just a couple of pointer swaps. After this operation, `otherlst.size() == 0`.

`lst.merge(otherlst)` similarly empties out `otherlst` into `lst` using only pointer swaps, but has the effect of "merging sorted lists." For example:

[PRE16]

As always with STL operations that involve comparison, there is a version taking a comparator: `lst.merge(otherlst, less)`.

Another operation that can be done only with pointer swaps is reversing the list in place: `lst.reverse()` switches all the "next" and "previous" links so that the head of the list is now the tail, and vice versa.

Notice that all of these operations *mutate the list in place*, and generally return `void`.

Another kind of operation that is cheap on linked lists (but not on contiguous containers) is removal of elements. Recall from [Chapter 3](part0036.html#12AK80-2fdac365b8984feebddfbb9250eaf20d), *The Iterator-Pair Algorithms*, that the STL provides algorithms such as `std::remove_if` and `std::unique` for use with contiguous containers; these algorithms shuffle the "removed" elements to the end of the container so that they can be picked off in a single `erase()`. With `std::list`, shuffling elements is more expensive than simply erasing them in-place. So, `std::list` provides the following member functions, with names that are unfortunately similar to the non-erasing STL algorithms:

*   `lst.remove(v)` removes *and erases* all elements equal to `v`.
*   `lst.remove_if(p)` removes *and erases* all elements `e` which satisfy the unary predicate `p(e)`.
*   `lst.unique()` removes *and erases* all but the first element of each "run" of consecutive equal elements. As always with STL operations that involve comparison, there is a version taking a comparator: `lst.unique(eq)` removes and erases `e2` whenever `p(e1, e2)`.
*   `lst.sort()` sorts the list in-place. This is particularly helpful because the permutative algorithm `std::sort(ctr.begin(), ctr.end())` does not work on the non-random-access `std::list::iterator`.

It's strange that `lst.sort()` can only sort the entire container, instead of taking a sub-range the way `std::sort` does. But if you want to sort just a sub-range of `lst`, you can do it with--say it with me--just a couple of pointer swaps!

[PRE17]

# Roughing it with std::forward_list<T>

The standard container `std::forward_list<T>` is a linked list like `std::list`, but with fewer amenities--no way to get its size, no way to iterate backward. In memory it looks similar to `std::list<T>`, but with smaller nodes:

![](img/00011.jpeg)

Nevertheless, `std::forward_list` retains almost all of the "special skills" of `std::list`. The only operations that it can't do are `splice` (because that involves inserting "before" the given iterator) and `push_back` (because that involves finding the end of the list in constant time).

`forward_list` replaces these missing member functions with `_after` versions:

*   `flst.erase_after(it)` to erase the element *after* the given position
*   `flst.insert_after(it, v)` to insert a new element *after* the given position
*   `flst.splice_after(it, otherflst)` to insert the elements of `otherflst` *after* the given position

As with `std::list`, you should avoid using `forward_list` at all unless you are in need of its particular set of skills.

# Abstracting with std::stack<T> and std::queue<T>

We've now seen three different standard containers with the member functions `push_back()` and `pop_back()` (and, although we didn't mention it, `back()` to retrieve a reference to the last element of the container). These are the operations we'd need if we wanted to implement a stack data structure.

The standard library provides a convenient way to abstract the idea of a stack, with the container known as (what else?) `std::stack`. Unlike the containers we've seen so far, though, `std::stack` takes an extra template parameter.

`std::stack<T, Ctr>` represents a stack of elements of type `T`, where the underlying storage is managed by an instance of the container type `Ctr`. For example, `stack<T, vector<T>>` uses a vector to manage its elements; `stack<T, list<T>>` uses a list; and so on. The default value for the template parameter `Ctr` is actually `std::deque<T>`; you may recall that `deque` takes up more memory than `vector` but has the benefit of never needing to reallocate its underlying array or move elements post-insertion.

To interact with a `std::stack<T, Ctr>`, you must restrict yourself to only the operations `push` (corresponding to `push_back` on the underlying container), `pop` (corresponding to `pop_back`), `top` (corresponding to `back`), and a few other accessors such as `size` and `empty`:

[PRE18]

One bizarre feature of `std::stack` is that it supports the comparison operators `==`, `!=`, `<`, `<=`, `>`, and `>=`; and that these operators work by comparing the underlying containers (using whatever semantics the underlying container type has defined). Since the underlying container type generally compares via lexicographical order, the result is that comparing two stacks compares them "lexicographically bottom up."

[PRE19]

This is fine if you're using only `==` and `!=`, or if you're relying on `operator<` to produce a consistent ordering for `std::set` or `std::map`; but it's certainly surprising the first time you see it!

The standard library also provides an abstraction for "queue." `std::queue<T, Ctr>` exposes the methods `push_back` and `pop_front` (corresponding to `push_back` and `pop_front` on the underlying container), as well as a few other accessors such as `front`, `back`, `size`, and `empty`.

Knowing that the container must support these primitive operations as efficiently as possible, you should be able to guess the *default* value of `Ctr`. Yes, it's `std::deque<T>`, the low-overhead double-ended queue.

Notice that, if you were implementing a queue from scratch using `std::deque<T>`, you could choose whether to push on the front of the deque and pop from the back, or to push on the back of the deque and pop from the front. The standard `std::queue<T, std::deque<T>>` chooses specifically to push on the back and pop from the front, which is easy to remember if you think about a "queue" in the real world. When you're queueing up at a ticket counter or a lunch line, you join the queue at the back and are served when you get to the front--never vice versa! It is a useful art to choose technical terms (such as `queue`, `front`, and `back`) whose technical meanings are an accurate mirror of their real-world counterparts.

# The useful adaptor: std::priority_queue<T>

In [Chapter 3](part0036.html#12AK80-2fdac365b8984feebddfbb9250eaf20d), *The Iterator-Pair Algorithms*, we introduced the family of "heap" algorithms: `make_heap`, `push_heap`, and `pop_heap`. You can use these algorithms to give a range of elements the max-heap property. If you maintain the max-heap property on your data as an invariant, you get a data structure commonly known as a *priority queue*. In data-structure textbooks, a priority queue is often depicted as a kind of *binary tree*, but as we saw in [Chapter 3](part0036.html#12AK80-2fdac365b8984feebddfbb9250eaf20d), *The Iterator-Pair Algorithms*, there's nothing about the max-heap property that requires an explicitly pointer-based tree structure.

The standard container `std::priority_queue<T, Ctr, Cmp>` represents a priority queue, represented internally as an instance of `Ctr` where the elements of the `Ctr` are invariably in max-heap order (as determined by an instance of the comparator type `Cmp`).

The default value of `Ctr` in this case is `std::vector<T>`. Remember that `vector` is the most efficient container; the only reason `std::stack` and `std::queue` chose `deque` as their default is that they didn't want to move elements after they'd been inserted. But with a priority queue, the elements are moving all the time, moving up and down in the max-heap as other elements are inserted or erased. So there's no particular benefit to using `deque` as the underlying container; therefore, the standard library followed the same rule I've been repeating like a drumbeat--use `std::vector` unless you have a specific reason to need something else!

The default value of `Cmp` is the standard library type `std::less<T>`, which represents `operator<`. In other words, the `std::priority_queue` container uses the same comparator by default as the `std::push_heap` and `std::pop_heap` algorithms from [Chapter 3](part0036.html#12AK80-2fdac365b8984feebddfbb9250eaf20d), *The Iterator-Pair Algorithms*.

The member functions exposed by `std::priority_queue<T, Ctr>` are `push`, `pop`, and `top`. Conceptually, the item at the front of the underlying container is at the "top" of the heap. One thing to remember is that in a max-heap, the item at the "top" of the heap is the *greatest* item--think of the items as playing King of the Hill, so that the biggest one wins and ends up on the top of the heap.

*   `pq.push(v)` inserts a new item into the priority queue, as if by `std::push_heap()` on the underlying container
*   `pq.top()` returns a reference to the element currently on top of the priority queue, as if by calling `ctr.front()` on the underlying container
*   `pq.pop()` pops off the maximum element and updates the heap, as if by `std::pop_heap()` on the underlying container

To get a *min-heap* instead of a max-heap, simply reverse the sense of the comparator you provide to the `priority_queue` template:

[PRE20]

# The trees: std::set<T> and std::map<K, V>

The class template `std::set<T>` provides the interface of a "unique set" for any `T` that implements `operator<`. As always with STL operations that involve comparison, there is a version taking a comparator: `std::set<T, Cmp>` provides "unique set" functionality using `Cmp(a,b)` instead of `(a < b)` to sort the data elements.

A `std::set` is conceptually a binary search tree, analogous to Java's `TreeSet`. In all popular implementations it's specifically a *red-black tree*, which is a particular kind of self-balancing binary search tree: even if you are constantly inserting and removing items from the tree, it will never get *too* unbalanced, which means that `insert` and `find` will always run in O(log *n*) time on average. Notice the number of pointers involved in its memory layout:

![](img/00012.jpeg)

Since, by definition, a binary search tree's elements are stored in their sort order (least to greatest), it would not be meaningful for `std::set` to provide member functions `push_front` or `push_back`. Instead, to add an element `v` to the set, you use `s.insert(v)`; and to delete an element, you use `s.erase(v)` or `s.erase(it)`:

[PRE21]

The return value of `s.insert(v)` is interesting. When we `insert` into a vector, there are only two possible outcomes: either the value is successfully added to the vector (and we get back an iterator to the newly inserted element), or else the insertion fails and an exception is thrown. When we `insert` into a set, there is a third possible outcome: maybe the insertion doesn't happen because there is already a copy of `v` in the set! That's not a "failure" worthy of exceptional control flow, but it's still something that the caller might want to know about. So `s.insert(v)` always returns a `pair` of return values: `ret.first` is the usual iterator to the copy of `v` now in the data structure (no matter whether it was just now inserted), and `ret.second` is `true` if the pointed-to `v` was just inserted and `false` if the pointed-to `v` was already in the set to begin with:

[PRE22]

The square-bracketed variable definitions in the preceding snippet are using C++17 *structured bindings*.

As the example just prior to this one shows, the elements of a `set` are stored in order--not just conceptually but visibly, in that `*s.begin()` is going to be the least element in the set and `*std::prev(s.end())` is going to be the greatest element. Iterating over the set using a standard algorithm or a ranged `for` loop will give you the set's elements in ascending order (remember, what "ascending" means is dictated by your choice of comparator--the `Cmp` parameter to the class template `set`).

The tree-based structure of a `set` implies that some standard algorithms such as `std::find` and `std::lower_bound` ([Chapter 3](part0036.html#12AK80-2fdac365b8984feebddfbb9250eaf20d), *The Iterator-Pair Algorithms*) will still work, but only inefficiently--the algorithm's iterators will spend a lot of time climbing up and down in the foothills of the tree, whereas if we had access to the tree structure itself, we could descend directly from the root of the tree and find a given element's position very quickly. Therefore, `std::set` provides member functions that can be used as replacements for the inefficient algorithms:

*   For `std::find(s.begin(), s.end(), v)`, use `s.find(v)`
*   For `std::lower_bound(s.begin(), s.end(), v)`, use `s.lower_bound(v)`
*   For `std::upper_bound(s.begin(), s.end(), v)`, use `s.upper_bound(v)`
*   For `std::count(s.begin(), s.end(), v)`, use `s.count(v)`
*   For `std::equal_range(s.begin(), s.end(), v)`, use `s.equal_range(v)`

Notice that `s.count(v)` will only ever return 0 or 1, because the set's elements are deduplicated. This makes `s.count(v)` a handy synonym for the set-membership operation--what Python would call `v in s` or what Java would call `s.contains(v)`.

`std::map<K, V>` is just like `std::set<K>`, except that each key `K` is allowed to have a value `V` associated with it; this makes a data structure analogous to Java's `TreeMap` or Python's `dict`. As always, there's `std::map<K, V, Cmp>` if you need a sorting order on your keys that's different from the natural `K::operator<`. Although you won't often think of `std::map` as "just a thin wrapper around a `std::set` of pairs," that's exactly how it looks in memory:

![](img/00013.jpeg)

`std::map` supports indexing with `operator[]`, but with a surprising twist. When you index into a size-zero vector with `vec[42]`, you get undefined behavior. When you index into a size-zero *map* with `m[42]`, the map helpfully inserts the key-value pair `{42, {}}` into itself and returns a reference to the second element of that pair!

This quirky behavior is actually helpful for writing code that's easy on the eyes:

[PRE23]

But it can lead to confusion if you don't pay attention:

[PRE24]

You'll notice that there is no `operator[] const` for maps, because `operator[]` always reserves the potential to insert a new key-value pair into `*this`. If you have a const map--or just a map that you really don't want to insert into right now--then the appropriate way to query it non-mutatively is with `m.find(k)`. Another reason to avoid `operator[]` is if your map's value type `V` is not default-constructible, in which case `operator[]` simply won't compile. In that case (real talk: in *any* case) you should use `m.insert(kv)` or `m.emplace(k, v)` to insert the new key-value pair exactly as you want it, instead of default-constructing a value just to assign over it again. Here's an example:

[PRE25]

Received wisdom in the post–C++11 world is that `std::map` and `std::set`, being based on trees of pointers, are so cache-unfriendly that you should avoid them by default and prefer to use `std::unordered_map` and `std::unordered_set` instead.

# A note about transparent comparators

In the last code example, I wrote `m.find("hello")`. Notice that `"hello"` is a value of type `const char[6]`, whereas `decltype(m)::key_type` is `std::string`, and (since we didn't specify anything special) `decltype(m)::key_compare` is `std::less<std::string>`. This means that when we call `m.find("hello")`, we're calling a function whose first parameter is of type `std::string`--and so we're implicitly constructing `std::string("hello")` to pass as the argument to `find`. In general, the argument to `m.find` is going to get implicitly converted to `decltype(m)::key_type`, which may be an expensive conversion.

If our `operator<` behaves properly, we can avoid this overhead by changing the comparator of `m` to some class with a *heterogeneous* `operator()` which also defines the member typedef `is_transparent`, like this:

[PRE26]

The "magic" here is all happening inside the library's implementation of `std::map`; the `find` member function specifically checks for the member `is_transparent` and changes its behavior accordingly. The member functions `count`, `lower_bound`, `upper_bound`, and `equal_range` all change their behavior as well. But oddly, the member function `erase` does not! This is probably because it would be too difficult for overload resolution to distinguish an intended `m.erase(v)` from an intended `m.erase(it)`. Anyway, if you want heterogeneous comparison during deletion as well, you can get it in two steps:

[PRE27]

# Oddballs: std::multiset<T> and std::multimap<K, V>

In STL-speak, a "set" is an ordered, deduplicated collection of elements. So naturally, a "multiset" is an ordered, non-deduplicated collection of elements! Its memory layout is exactly the same as the layout of `std::set`; only its invariants are different. Notice in the following diagram that `std::multiset` allows two elements with value `42`:

![](img/00014.jpeg)

`std::multiset<T, Cmp>` behaves just like `std::set<T, Cmp>`, except that it can store duplicate elements. The same goes for `std::multimap<K, V, Cmp>`:

[PRE28]

In a multiset or multimap, `mm.find(v)` returns an iterator to *some* element (or key-value pair) matching `v`--not necessarily the first one in iteration order. `mm.erase(v)` erases all the elements (or key-value pairs) with keys equal to `v`. And `mm[v]` doesn't exist. For example:

[PRE29]

# Moving elements without moving them

Recall that, with `std::list`, we were able to splice lists together, move elements from one list to another, and so on, by using the "particular set of skills" of `std::list`. As of C++17, the tree-based containers have acquired similar skills!

The syntax for merging two sets or maps (or multisets or multimaps) is deceptively similar to the syntax for merging sorted `std::list`:

[PRE30]

However, notice what happens when there are duplicates! The duplicated elements are *not* transferred; they're left behind in the right-hand-side map! This is the exact opposite of what you'd expect if you're coming from a language such as Python, where `d.update(otherd)` inserts all the mappings from the right-hand dict into the left-hand dict, overwriting anything that was there already.

The C++ equivalent of `d.update(otherd)` is `m.insert(otherm.begin(), otherm.end()`. The only case in which it makes sense to use `m.merge(otherm)` is if you know that you don't want to overwrite duplicates, *and* you're okay with trashing the old value of `otherm` (for example, if it's a temporary that's going out of scope soon).

Another way to transfer elements between tree-based containers is to use the member functions `extract` and `insert` to transfer individual elements:

[PRE31]

The type of the object returned by `extract` is something called a "node handle"--essentially a pointer into the guts of the data structure. You can use the accessor methods `nh.key()` and `nh.mapped()` to manipulate the pieces of the entry in a `std::map` (or `nh.value()` for the single piece of data in an element of a `std::set`). Thus you can extract, manipulate, and reinsert a key without ever copying or moving its actual data! In the following code sample, the "manipulation" consists of a call to `std::transform`:

[PRE32]

As you can see, the interface to this functionality isn't as tidy as `lst.splice(it, otherlst)`; the subtlety of the interface is one reason it took until C++17 to get this functionality into the standard library. There is one clever bit to notice, though: Suppose you `extract` a node from a set and then throw an exception before you've managed to `insert` it into the destination set. What happens to the orphaned node--does it leak? It turns out that the designers of the library thought of this possibility; if a node handle's destructor is called before the node handle has been inserted into its new home, the destructor will correctly clean up the memory associated with the node. Therefore, `extract` by itself (without `insert`) will behave just like `erase`!

# The hashes: std::unordered_set<T> and std::unordered_map<K, V>

The `std::unordered_set` class template represents a chained hash table--that is, a fixed-size array of "buckets," each bucket containing a singly linked list of data elements. As new data elements are added to the container, each element is placed in the linked list associated with the "hash" of the element's value. This is almost exactly the same as Java's `HashSet`. In memory it looks like this:

![](img/00015.jpeg)

The literature on hash tables is extensive, and `std::unordered_set` does not represent even remotely the state of the art; but because it eliminates a certain amount of pointer-chasing, it tends to perform better than the tree-based `std::set`.

To eliminate the rest of the pointers, you'd have to replace the linked lists with a technique called "open addressing," which is far out of scope for this book; but it's worth looking up if `std::unordered_set` proves too slow for your use-case.

`std::unordered_set` was designed to be a drop-in replacement for `std::set`, so it provides the same interface that we've already seen: `insert` and `erase`, plus iteration with `begin` and `end`. However, unlike `std::set`, the elements of a `std::unordered_set` are not stored in sorted order (it's *unordered*, you see?) and it provides only forward iterators, as opposed to the bidirectional iterators provided by `std::set`. (Check the preceding illustration--there are "next" pointers but no "previous" pointers, so iterating backwards in a `std::unordered_set` is impossible.)

`std::unordered_map<K, V>` is to `std::unordered_set<T>` as `std::map<K, V>` is to `std::set<T>`. That is, it looks exactly the same in memory, except that it stores key-value pairs instead of just keys:

![](img/00016.jpeg)

Like `set` and `map`, which take an optional comparator parameter, `unordered_set` and `unordered_map` take some optional parameters as well. The two optional parameters are `Hash` (which defaults to `std::hash<K>`) and `KeyEqual` (which defaults to `std::equal_to<K>`, which is to say, `operator==`). Passing in a different hash function or a different key-comparison function causes the hash table to use those functions instead of the defaults. This might be useful if you're interfacing with some old-school C++ class type that doesn't implement value semantics or `operator==`:

[PRE33]

# Load factor and bucket lists

Like Java's `HashSet`, `std::unordered_set` exposes all kinds of administrative details about its buckets. You probably will never need to interact with these administrative functions!

*   `s.bucket_count()` returns the current number of buckets in the array.
*   `s.bucket(v)` returns the index *i* of the bucket in which you'd find the
    element `v`, if it existed in this `unordered_set`.
*   `s.bucket_size(i)` returns the number of elements in the *i*th bucket. Observe that invariably `s.count(v) <= s.bucket_size(s.bucket(v))`.
*   `s.load_factor()` returns `s.size() / s.bucket_count()` as a `float` value.
*   `s.rehash(n)` increases (or decreases) the size of the bucket array to exactly `n`.

You might have noticed that `load_factor` seems out of place so far; what's so important about `s.size() / s.bucket_count()` that it gets its own member function? Well, this is the mechanism by which `unordered_set` scales itself as its number of elements grows. Each `unordered_set` object `s` has a value `s.max_load_factor()` indicating exactly how large `s.load_factor()` is allowed to get. If an insertion would push `s.load_factor()` over the top, then `s` will reallocate its array of buckets and rehash its elements in order to keep `s.load_factor()` smaller than `s.max_load_factor()`.

`s.max_load_factor()` is `1.0` by default. You can set it to a different value `k` by using the one-parameter overload: `s.max_load_factor(k)`. However, that's basically never necessary or a good idea.

One administrative operation that *does* make sense is `s.reserve(k)`. Like `vec.reserve(k)` for vectors, this `reserve` member function means "I'm planning to do insertions that bring the size of this container up into the vicinity of `k`. Please pre-allocate enough space for those `k` elements right now." In the case of `vector`, that meant allocating an array of `k` elements. In the case of `unordered_set`, it means allocating a bucket array of `k / max_load_factor()` pointers, so that even if `k` elements are inserted (with the expected number of collisions), the load factor will still only be `max_load_factor()`.

# Where does the memory come from?

Throughout this whole chapter, I've actually been lying to you! Each of the containers described in this chapter--except for `std::array`--takes one *more* optional template type parameter. This parameter is called the *allocator*, and it indicates where the memory comes from for operations such as "reallocating the underlying array" or "allocating a new node on the linked list." `std::array` doesn't need an allocator because it holds all of its memory inside itself; but every other container type needs to know where to get its allocations from.

The default value for this template parameter is the standard library type `std::allocator<T>`, which is certainly good enough for most users. We'll talk more about allocators in [Chapter 8](part0129.html#3R0OI0-2fdac365b8984feebddfbb9250eaf20d), *Allocators*.

# Summary

In this chapter we've learned the following: A *container* manages the *ownership* of a collection of elements. STL containers are always class templates parameterized on the element type, and sometimes on other relevant parameters as well. Every container except `std::array<T, N>` can be parameterized by an *allocator* type to specify the manner in which it allocates and deallocates memory. Containers that use comparison can be parameterized by a *comparator* type. Consider using transparent comparator types such as `std::less<>` instead of homogeneous comparators.

When using `std::vector`, watch out for reallocation and address invalidation. When using most container types, watch out for iterator invalidation.

The standard library's philosophy is to support no operation that is naturally inefficient (such as `vector::push_front`); and to support any operation that is naturally efficient (such as `list::splice`). If you can think of an efficient implementation for a particular operation, odds are that the STL has already implemented it under some name; you just have to figure out how it's spelled.

When in doubt, use `std::vector`. Use other container types only when you need their particular set of skills. Specifically, avoid the pointer-based containers (`set`, `map`, `list`) unless you need their special skills (maintaining sorted order; extracting, merging, and splicing).

Online references such as [cppreference.com](http://cppreference.com) are your best resource for figuring these things out.