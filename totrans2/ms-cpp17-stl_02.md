# Iterators and Ranges

In the previous chapter, we implemented several generic algorithms that operated on containers, but in an inefficient manner. In this chapter, you'll learn:

*   How and why C++ generalizes the idea of pointers to create the *iterator* concept
*   The importance of *ranges* in C++, and the standard way to express a half-open range as a pair of iterators
*   How to write your own rock-solid, const-correct iterator types
*   How to write generic algorithms that operate on iterator pairs
*   The standard iterator hierarchy and its algorithmic importance

# The problem with integer indices

In the previous chapter, we implemented several generic algorithms that operated on containers. Consider one of those algorithms again:

[PRE0]

This algorithm is defined in terms of the lower-level operations `.size()` and `.at()`. This works reasonably well for a container type such as `array_of_ints` or `std::vector`, but it doesn't work nearly so well for, say, a linked list such as the previous chapter's `list_of_ints`:

[PRE1]

The implementation of `list_of_ints::at()` is O(*n*) in the length of the list--the longer our list gets, the slower `at()` gets. And particularly, when our `count_if` function loops over each element of the list, it's calling that `at()` function *n* times, which makes the runtime of our generic algorithm O(*n*²)--for a simple counting operation that ought to be O(*n*)!

It turns out that integer indexing with `.at()` isn't a very good foundation on which to build algorithmic castles. We ought to pick a primitive operation that's closer to how computers actually manipulate data.

# On beyond pointers

In the absence of any abstraction, how does one normally identify an element of an array, an element of a linked list, or an element of a tree? The most straightforward way would be to use a *pointer* to the element's address in memory. Here are some examples of pointers to elements of various data structures:

![](img/00005.jpeg)

To iterate over an *array*, all we need is that pointer; we can handle all the elements in the array by starting with a pointer to the first element and simply incrementing that pointer until it reaches the last element. In C:

[PRE2]

But in order to efficiently iterate over a *linked list*, we need more than just a raw pointer; incrementing a pointer of type `node*` is highly unlikely to produce a pointer to the next node in the list! In that case, we need something that acts like a pointer--in particular, we should be able to dereference it to retrieve or modify the pointed-to element--but has special, container-specific behavior associated with the abstract concept of incrementing.

In C++, given that we have operator overloading built into the language, when I say "associate special behavior with the concept of incrementing", you should be thinking "let's overload the `++` operator." And indeed, that's what we'll do:

[PRE3]

Notice that we also overload the unary `*` operator (for dereferencing) and the `==` and `!=` operators; our `count_if` template requires all of these operations be valid for the loop control variable `it`. (Well, okay, technically our `count_if` doesn't require the `==` operation; but if you're going to overload one of the comparison operators, you should overload the other as well.)

# Const iterators

There's just one more complication to consider, before we abandon this list iterator example. Notice that I quietly changed our `count_if` function template so that it takes `Container&` instead of `const Container&`! That's because the `begin()` and `end()` member functions we provided are non-const member functions; and that's because they return iterators whose `operator*` returns non-const references to the elements of the list. We'd like to make our list type (and its iterators) completely const-correct--that is, we'd like you to be able to define and use variables of type `const list_of_ints`, but prevent you from modifying the elements of a `const` list.

The standard library generally deals with this issue by giving each standard container two different kinds of iterator: `bag::iterator` and `bag::const_iterator`. The non-const member function `bag::begin()` returns an `iterator` and the `bag::begin() const` member function returns a `const_iterator`. The underscore is all-important! Notice that `bag::begin() const` does not return a mere `const iterator`; if the returned object were `const`, we wouldn't be allowed to `++` it. (Which, in turn, would make it darn difficult to iterate over a `const bag`!) No, `bag::begin() const` returns something more subtle: a non-const `const_iterator` object whose `operator*` simply happens to yield a *const* reference to its element.

Maybe an example would help. Let's go ahead and implement `const_iterator` for our `list_of_ints` container.

Since most of the code for the `const_iterator` type is going to be exactly the same as the code for the `iterator` type, our first instinct might be to cut and paste. But this is C++! When I say "most of this code is going to be exactly the same as this other code," you should be thinking "let's make the common parts into a template." And indeed, that's what we'll do:

[PRE4]

The preceding code implements fully const-correct iterator types for our `list_of_ints`.

# A pair of iterators defines a range

Now that we understand the fundamental concept of an iterator, let's put it to some practical use. We've already seen that if you have a pair of iterators as returned from `begin()` and `end()`, you can use a for-loop to iterate over all the elements of the underlying container. But more powerfully, you can use some pair of iterators to iterate over any sub-range of the container's elements! Let's say you only wanted to view the first half of a vector:

[PRE5]

Notice that in the first and second test cases in `main()` we pass in a pair of iterators derived from `v.begin()`; that is, two values of type `std::vector::iterator`. In the third test case, we pass in two values of type `int*`. Since `int*` satisfies all the requirements of an iterator type in this case--namely: it is incrementable, comparable, and dereferenceable--our code works fine even with pointers! This example demonstrates the flexibility of the iterator-pair model. (However, in general you should avoid messing around with raw pointers, if you're using a container such as `std::vector` that offers a proper `iterator` type. Use iterators derived from `begin()` and `end()` instead.)

We can say that a pair of iterators implicitly defines a *range* of data elements. And for a surprisingly large family of algorithms, that's good enough! We don't need to have access to the *container* in order to perform certain searches or transformations; we only need access to the particular *range* of elements being searched or transformed. Going further down this line of thought will eventually lead us to the concept of a *non-owning view* (which is to a data sequence as a C++ reference is to a single variable), but views and ranges are still more modern concepts, and we ought to finish up with the 1998-vintage STL before we talk about those things.

In the previous code sample, we saw the first example of a real STL-style generic algorithm. Admittedly, `double_each_element` is not a terribly generic algorithm in the sense of implementing a behavior that we might want to reuse in other programs; but this version of the function is now perfectly generic in the sense of operating only on pairs of `Iterators`, where `Iterator` can be any type in the world that implements incrementability, comparability, and dereferenceability. (We'll see a version of this algorithm that is more generic in that first sense in this book's next chapter, when we talk about `std::transform`.)

# Iterator categories

Let's revisit the `count` and `count_if` functions that we introduced in
[Chapter 1](part0021.html#K0RQ0-2fdac365b8984feebddfbb9250eaf20d), *Classical Polymorphism and Generic Programming*. Compare the function template definition in this next example to the similar code from that chapter; you'll see that it's identical except for the substitution of a pair of `Iterators` (that is, an implicitly defined *range*) for the `Container&` parameter--and except that I've changed the name of the first function from `count` to `distance`. That's because you can find this function, almost exactly as described here, in the Standard Template Library under the name `std::distance` and you can find the second function under the name `std::count_if`:

[PRE6]

But let's consider the line marked `DUBIOUS` in that example. Here we're computing the distance between two `Iterators` by repeatedly incrementing the one until it reaches the other. How performant is this approach? For certain kinds of iterators--for example, `list_of_ints::iterator`--we're not going to be able to do better than this. But for `vector::iterator` or `int*`, which iterate over contiguous data, it's a little silly of us to be using a loop and an O(n) algorithm when we could accomplish the same thing in O(1) time by simple pointer subtraction. That is, we'd like the standard library version of `std::distance` to include a template specialization something like this:

[PRE7]

But we don't want the specialization to exist only for `int*` and `std::vector::iterator`. We want the standard library's `std::distance` to be efficient for all the iterator types that support this particular operation. That is, we're starting to develop an intuition that there are (at least) two different kinds of iterators: there are those that are incrementable, comparable, and dereferenceable; and then there are those that are incrementable, comparable, dereferenceable, *and also subtractable!* It turns out that for any iterator type where the operation `i = p - q` makes sense, its inverse operation `q = p + i` also makes sense. Iterators that support subtraction and addition are called *random-access iterators*.

So, the standard library's `std::distance` ought to be efficient for both random-access iterators and other kinds of iterators. To make it easier to supply the partial specializations for these templates, the standard library introduced the idea of a hierarchy of iterator kinds. Iterators such as `int*`, which support addition and subtraction, are known as random-access iterators. We'll say that they satisfy the concept `RandomAccessIterator`.

Iterators slightly less powerful than random-access iterators might not support addition or subtraction of arbitrary distances, but they at least support incrementing and decrementing with `++p` and `--p`. Iterators of this nature are called `BidirectionalIterator`. All `RandomAccessIterator` are `BidirectionalIterator`, but not necessarily vice versa. In some sense, we can imagine `RandomAccessIterator` to be a sub-class or sub-concept relative to `BidirectionalIterator`; and we can say that `BidirectionalIterator` is a *weaker concept*, imposing fewer requirements, than `RandomAccessIterator`.

An even weaker concept is the kind of iterators that don't even support decrementing. For example, our `list_of_ints::iterator` type doesn't support decrementing, because our linked list has no previous pointers; once you've got an iterator pointing at a given element of the list, you can only move forward to later elements, never backward to previous ones. Iterators that support `++p` but not `--p` are called `ForwardIterator`. `ForwardIterator` is a weaker concept than `BidirectionalIterator`.

# Input and output iterators

We can imagine even weaker concepts than `ForwardIterator`! For example, one useful thing you can do with a `ForwardIterator` is to make a copy of it, save the copy, and use it to iterate twice over the same data. Manipulating the iterator (or copies of it) doesn't affect the underlying range of data at all. But we could invent an iterator like the one in the following snippet, where there is no underlying data at all, and it's not even meaningful to make a copy of the iterator:

[PRE8]

(In fact, the standard library contains some iterator types very similar to this one; we'll discuss one such type, `std::istream_iterator`, in [Chapter 9](part0144.html#49AH00-2fdac365b8984feebddfbb9250eaf20d), *Iostreams*.) Such iterators, which are not meaningfully copyable, and do not point to data elements in any meaningful sense, are called `InputIterator` types.

The mirror-image case is also possible. Consider the following invented iterator type:

[PRE9]

(Again, the standard library contains some iterator types very similar to this one; we'll discuss `std::back_insert_iterator` in [Chapter 3](part0036.html#12AK80-2fdac365b8984feebddfbb9250eaf20d), *The Iterator-Pair Algorithms*, and `std::ostream_iterator` in [Chapter 9](part0144.html#49AH00-2fdac365b8984feebddfbb9250eaf20d), *Iostreams.*) Such iterators, which are not meaningfully copyable, and are writeable-into but not readable-out-of, are called `OutputIterator` types.

Every iterator type in C++ falls into at least one of the following five categories:

*   `InputIterator`
*   `OutputIterator`
*   `ForwardIterator`
*   `BidirectionalIterator`, and/or
*   `RandomAccessIterator`

Notice that while it's easy to figure out at compile time whether a particular iterator type conforms to the `BidirectionalIterator` or `RandomAccessIterator` requirements, it's impossible to figure out (purely from the syntactic operations it supports) whether we're dealing with an `InputIterator`, an `OutputIterator`, or a `ForwardIterator`. In our examples just a moment ago, consider: `getc_iterator`, `putc_iterator`, and `list_of_ints::iterator` support exactly the same syntactic operations--dereferencing with `*it`, incrementing with `++it`, and comparison with `it != it`. These three classes differ only at the semantic level. So how can the standard library distinguish between them?

It turns out that the standard library needs a bit of help from the implementor of each new iterator type. The standard library's algorithms will work only with iterator classes which define a *member typedef* named `iterator_category`. That is:

[PRE10]

Then any standard (or heck, non-standard) algorithm that wants to customize its behavior based on the iterator categories of its template type parameters can do that customization simply by inspecting those types' `iterator_category`.

The iterator categories described in the preceding paragraph, correspond to the following five standard tag types defined in the `<iterator>` header:

[PRE11]

Notice that `random_access_iterator_tag` actually derives (in the classical-OO, polymorphic-class-hierarchy sense) from `bidirectional_iterator_tag`, and so on: the *conceptual hierarchy* of iterator kinds is reflected in the *class hierarchy* of `iterator_category` tag classes. This turns out to be useful in template metaprogramming when you're doing tag dispatch; but all you need to know about it for the purposes of using the standard library is that if you ever want to pass an `iterator_category` to a function, a tag of type `random_access_iterator_tag` will be a match for a function expecting an argument of type `bidirectional_iterator_tag`:

[PRE12]

At this point I expect you're wondering: "But what about `int*`? How can we provide a member typedef to something that isn't a class type at all, but rather a primitive scalar type? Scalar types can't have member typedefs." Well, as with most problems in software engineering, this problem can be solved by adding a layer of indirection. Rather than referring directly to `T::iterator_category`, the standard algorithms are careful always to refer to `std::iterator_traits<T>::iterator_category`. The class template `std::iterator_traits<T>` is appropriately specialized for the case where `T` is a pointer type.

Furthermore, `std::iterator_traits<T>` proved to be a convenient place to hang other member typedefs. It provides the following five member typedefs, if and only if `T` itself provides all five of them (or if `T` is a pointer type): `iterator_category`, `difference_type`, `value_type`, `pointer`, and `reference`.

# Putting it all together

Putting together everything we've learned in this chapter, we can now write code like the following example. In this example, we're implementing our own `list_of_ints` with our own iterator class (including a const-correct `const_iterator` version); and we're enabling it to work with the standard library by providing the five all-important member typedefs.

[PRE13]

Then, to show that we understand how the standard library implements generic algorithms, we'll implement the function templates `distance` and `count_if` exactly as the C++17 standard library would implement them.

Notice the use of C++17's new `if constexpr` syntax in `distance`. We won't talk about C++17 core language features very much in this book, but suffice it to say, you can use `if constexpr` to eliminate a lot of awkward boilerplate compared to what you'd have had to write in C++14.

[PRE14]

In the next chapter we'll stop implementing so many of our own function templates from scratch, and start marching through the function templates provided by the Standard Template Library. But before we leave this deep discussion of iterators, there's one more thing I'd like to talk about.

# The deprecated std::iterator

You might be wondering: "Every iterator class I implement needs to provide the same five member typedefs. That's a lot of boilerplate--a lot of typing that I'd like to factor out, if I could." Is there no way to eliminate all that boilerplate?

Well, in C++98, and up until C++17, the standard library included a helper class template to do exactly that. Its name was `std::iterator`, and it took five template type parameters that corresponded to the five member typedefs required by `std::iterator_traits`. Three of these parameters had "sensible defaults," meaning that the simplest use-case was pretty well covered:

[PRE15]

Unfortunately for `std::iterator`, real life wasn't that simple; and `std::iterator` was deprecated in C++17 for several reasons that we're about to discuss.

As we saw in the section *Const iterators*, const-correctness requires us to provide a const iterator type along with every "non-const iterator" type. So what we really end up with, following that example, is code like this:

[PRE16]

The preceding code isn't any easier to read or write than the version that didn't use `std::iterator`; and furthermore, using `std::iterator` in the intended fashion complicates our code with *public inheritance*, which is to say, something that looks an awful lot like the classical object-oriented class hierarchy. A beginner might well be tempted to use that class hierarchy in writing functions like this one:

[PRE17]

This looks superficially similar to our examples of "polymorphic programming" from [Chapter 1](part0021.html#K0RQ0-2fdac365b8984feebddfbb9250eaf20d), *Classical Polymorphism and Generic Programming*, a function that implements different behaviors by taking parameters of type reference-to-base-class. But in the case of `std::iterator` this similarity is purely accidental and misleading; inheriting from `std::iterator` does *not* give us a polymorphic class hierarchy, and referring to that "base class" from our own functions is never the correct thing to do!

So, the C++17 standard deprecates `std::iterator` with an eye toward removing it completely in 2020 or some later standard. You shouldn't use `std::iterator` in code you write.

However, if you use Boost in your codebase, you might want to check out the Boost equivalent of `std::iterator`, which is spelled `boost::iterator_facade`. Unlike `std::iterator`, the `boost::iterator_facade` base class provides default functionality for pesky member functions such as `operator++(int)` and `operator!=` that would otherwise be tedious boilerplate. To use `iterator_facade`, simply inherit from it and define a few primitive member functions such as `dereference`, `increment`, and `equal`. (Since our list iterator is a `ForwardIterator`, that's all we need. For a `BidirectionalIterator` you would also need to provide a `decrement` member function, and so on.)

Since these primitive member functions are `private`, we grant Boost access to them via the declaration `friend class boost::iterator_core_access;`:

[PRE18]

Notice that the first template type argument to `boost::iterator_facade` is always the class whose definition you're writing: this is the Curiously Recurring Template Pattern, which we'll see again in [Chapter 6](part0093.html#2OM4A0-2fdac365b8984feebddfbb9250eaf20d), *Smart Pointers*.

This list-iterator code using `boost::iterator_facade` is significantly shorter than the same code in the previous section; the savings comes mainly from not having to repeat the relational operators. Because our list iterator is a `ForwardIterator`, we only had two relational operators; but if it were a `RandomAccessIterator`, then `iterator_facade` would generate default implementations of operators `-`, `<`, `>`, `<=`, and `>=` all based on the single primitive member function `distance_to`.

# Summary

In this chapter, we've learned that traversal is one of the most fundamental things you can do with a data structure. However, raw pointers alone are insufficient for traversing complicated structures: applying `++` to a raw pointer often doesn't "go on to the next item" in the intended way.

The C++ Standard Template Library provides the concept of *iterator* as a generalization of raw pointers. Two iterators define a *range* of data. That range might be only part of the contents of a container; or it might be unbacked by any memory at all, as we saw with `getc_iterator` and `putc_iterator`. Some of the properties of an iterator type are encoded in its iterator category--input, output, forward, bidirectional, or random-access--for the benefit of function templates that can use faster algorithms on certain categories of iterators.

If you're defining your own container type, you'll need to define your own iterator types as well--both const and non-const versions. Templates are a handy way to do that. When implementing your own iterator types, avoid the deprecated `std::iterator`, but consider `boost::iterator_facade`.