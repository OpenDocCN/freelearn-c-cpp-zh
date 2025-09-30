# Vocabulary Types

It has been increasingly recognized over the past decade that one of the important roles of a standard language or standard library is to provide *vocabulary types*. A "vocabulary" type is a type that purports to provide a single *lingua franca*, a common language, for dealing with its domain.

Notice that even before C++ existed, the C programming language had already made a decent shot at the vocabulary of some areas, providing standard types or type aliases for integer math (`int`), floating-point math (`double`), timepoints expressed in the Unix epoch (`time_t`), and byte counts (`size_t`).

In this chapter we'll learn:

*   The history of vocabulary types in C++, from `std::string` to `std::any`
*   The definitions of *algebraic data type*, *product type*, and *sum type*
*   How to manipulate tuples and visit variants
*   The role of `std::optional<T>` as "maybe a `T`" or "not yet a `T`"
*   `std::any` as the algebraic-data-type equivalent of "infinity"
*   How to implement type erasure, how it's used in `std::any` and `std::function`, and its intrinsic limitations
*   Some pitfalls with `std::function`, and third-party libraries that fix them

# The story of std::string

Consider the domain of character strings; for example, the phrase `hello world`. In C, the *lingua franca* for dealing with strings was `char *`:

[PRE0]

This was all right for a while, but dealing with raw `char *`s had some problems for the users of the language and the creators of third-party libraries and routines. For one thing, the C language was so old that `const` had not been invented at the outset, which meant that certain old routines would expect their strings as `char *` and certain newer ones expect `const char *`. For another thing, `char *` didn't carry a *length* with it; so some functions expected both a pointer and a length, and some functions expected only the pointer and simply couldn't deal with embedded bytes of value `'\0'`.

The most vital piece missing from the `char *` puzzle was *lifetime management* and *ownership* (as discussed at the start of [Chapter 4](part0052.html#1HIT80-2fdac365b8984feebddfbb9250eaf20d), *The Container Zoo*). When a C function wants to receive a string from its caller, it takes `char *` and generally leaves it up to the caller to manage the ownership of the characters involved. But what if it wants to *return* a string? Then it has to return `char *` and hope that the caller remembers to free it (`strdup`, `asprintf`), or take in a buffer from the caller and hope it's big enough for the output (`sprintf`, `snprintf`, `strcat`). The difficulty of managing the ownership of strings in C (and in pre-standard C++) was so great that there was a proliferation of "string libraries" to deal with the problem: Qt's `QString`, glib's `GString`, and so on.

Into this chaos stepped C++ in 1998 with a miracle: a *standard* string class! The new `std::string` encapsulated the bytes of a string *and* its length, in a natural way; it could deal correctly with embedded null bytes; it supported formerly complicated operations such as `hello + world` by quietly allocating exactly as much memory as it needed; and because of RAII, it would never leak memory or incite confusion about who owned the underlying bytes. Best of all, it had an implicit conversion from `char *`:

[PRE1]

Now C++ functions dealing with strings (such as `greet()` in the preceding code) could take `std::string` parameters and return `std::string` results. Even better, because the string type was *standardized*, within a few years you could be reasonably confident that when you picked up some third-party library to integrate it into your codebase, any of its functions that took strings (filenames, error messages, what-have-you) would be using `std::string`. Everybody could communicate more efficiently and effectively by sharing the *lingua franca* of `std::string`.

# Tagging reference types with reference_wrapper

Another vocabulary type introduced in C++03 was `std::reference_wrapper<T>`. It has a simple implementation:

[PRE2]

`std::reference_wrapper` has a slightly different purpose from vocabulary types such as `std::string` and `int`; it's meant specifically as a way to the "tag" values that we'd like to behave as references in contexts where passing native C++ references doesn't work the way we'd like:

[PRE3]

The constructor of `std::thread` is written with specific special cases to handle `reference_wrapper` parameters by "decaying" them into native references. The same special cases apply to the standard library functions `make_pair`, `make_tuple`, `bind`, `invoke`, and everything based on `invoke` (such as `std::apply`, `std::function::operator()`, and `std::async`).

# C++11 and algebraic types

As C++11 took shape, there was growing recognition that another area ripe for vocabularization was that of the so-called *algebraic data types*. Algebraic types arise naturally in the functional-programming paradigm. The essential idea is to think about the domain of a type--that is, the set of all possible values of that type. To keep things simple, you might want to think about C++ `enum` types, because it's easy to talk about the number of different values that an object of `enum` type might assume at one time or another:

[PRE4]

Given the types `Color` and `Size`, can you create a data type whose instances might assume any of 2 × 3 = 6 values? Yes; this type represents "one of each" of `Color` and `Size`, and is called a *product type*, because its set of possible values is the *Cartesian product* of its elements' sets of possible values.

How about a data type whose instances might assume any of 2 + 3 = 5 different values? Also yes; this type represents "either a `Color` or a `Size` but never both at once," and is called a *sum type*. (Confusingly, mathematicians do not use the term *Cartesian sum* for this concept.)

In a functional-programming language such as Haskell, these two exercises would be spelled like this:

[PRE5]

In C++, they're spelled like this:

[PRE6]

The class template `std::pair<A, B>` represents an ordered pair of elements: one of type `A`, followed by one of type `B`. It's very similar to a plain old `struct` with two elements, except that you don't have to write the struct definition yourself:

[PRE7]

Notice that there are only cosmetic differences between `std::pair<A, A>` and `std::array<A, 2>`. We might say that `pair` is a *heterogeneous* version of `array` (except that `pair` is restricted to holding only two elements).

# Working with std::tuple

C++11 introduced a full-fledged heterogeneous array; it's called `std::tuple<Ts...>`. A tuple of only two element types--for example, `tuple<int, double>`--is no different from `pair<int, double>`. But tuples can hold more than just a pair of elements; though the magic of C++11 variadic templates they can hold triples, quadruples, quintuples,... hence the generic name `tuple`. For example, `tuple<int, int, char, std::string>` is analogous to a `struct` whose members are an `int`, another `int`, a `char`, and finally a `std::string`.

Because the first element of a tuple has a different type from the second element, we can't use the "normal" `operator[](size_t)` to access the elements by indices that might vary at runtime. Instead, we must tell the compiler *at compile time* which element of the tuple we're planning to access, so that the compiler can figure out what type to give the expression. The C++ way to provide information at compile time is to force it into the type system via template parameters, and so that's what we do. When we want to access the first element of a tuple `t`, we call `std::get<0>(t)`. To access the second element, we call `std::get<1>(t)`, and so on.

This becomes the pattern for dealing with `std::tuple`--where the homogeneous container types tend to have *member functions* for accessing and manipulating them, the heterogeneous algebraic types tend to have *free function templates* for accessing and manipulating them.

However, generally speaking, you won't do a lot of *manipulating* of tuples. Their primary use-case, outside of template metaprogramming, is as an economical way to temporarily bind a number of values together in a context that requires a single value. For example, you might remember `std::tie` from the example in section "The simplest container" in [Chapter 4](part0052.html#1HIT80-2fdac365b8984feebddfbb9250eaf20d), *The Container Zoo*. It's a cheap way of binding together an arbitrary number of values into a single unit that can be compared lexicographically with `operator<`. The "sense" of the lexicographical comparison depends on the order in which you bind the values together:

[PRE8]

The reason that `std::tie` is so cheap is that it actually creates a tuple of *references* to its arguments' memory locations, rather than copying its arguments' values. This leads to a second common use for `std::tie`: simulating the "multiple assignment" found in languages such as Python:

[PRE9]

Notice that the phrase "at once" in the preceding comment doesn't have any bearing on concurrency (see [Chapter 7](part0108.html#36VSO0-2fdac365b8984feebddfbb9250eaf20d), *Concurrency*) or the order in which the side effects are performed; I just mean that both values can be assigned in a single assignment statement, instead of taking two or more lines.

As the preceding example illustrates, `std::make_tuple(a, b, c...)` can be used to create a tuple of *values*; that is, `make_tuple` does construct copies of its arguments' values, rather than merely taking their addresses.

Lastly, in C++17 we are allowed to use constructor template parameter deduction to write simply `std::tuple(a, b, c...)`; but it's probably best to avoid this feature unless you know specifically that you want its behaviour. The only thing that template parameter deduction will do differently from `std::make_tuple` is that it will preserve `std::reference_wrapper` arguments rather than decaying them to native C++ references:

[PRE10]

# Manipulating tuple values

Most of these functions and templates are useful only in the context of template metaprogramming; you're unlikely to use them on a daily basis:

*   `std::get<I>(t)`: Retrieves a reference to the `I`th element of `t`.
*   `std::tuple_size_v<decltype(t)>`: Tells the *size* of the given tuple. Because this is a compile-time constant property of the tuple's type, this is expressed as a variable template parameterized on that type. If you'd rather use more natural-looking syntax, you can write a helper function in either of the following ways:

[PRE11]

*   `std::tuple_element_t<I, decltype(t)>`: Tells the *type* of the `I`th element of the given tuple type. Again, the standard library exposes this information in a more awkward way than the core language does. Generally, to find the type of the `I`th element of a tuple, you'd just write `decltype(std::get<I>(t))`.
*   `std::tuple_cat(t1, t2, t3...)`: Concatenates all the given tuples together, end to end.
*   `std::forward_as_tuple(a, b, c...)`: Creates a tuple of references, just like `std::tie`; but whereas `std::tie` demands lvalue references, `std::forward_as_tuple` will accept any kind of references as input, and perfectly forward them into the tuple so that they can later be extracted by `std::get<I>(t)...`:

[PRE12]

# A note about named classes

As we saw in [Chapter 4](part0052.html#1HIT80-2fdac365b8984feebddfbb9250eaf20d), *The Container Zoo*, when we compared `std::array<double, 3>` to `struct Vec3`, using an STL class template can shorten your development time and eliminate sources of error by reusing well-tested STL components; but it can also make your code less readable or give your types *too much* functionality. In our example from [Chapter 4](part0052.html#1HIT80-2fdac365b8984feebddfbb9250eaf20d), *The Container Zoo,* `std::array<double, 3>` turned out to be a poor choice for `Vec3` because it exposed an unwanted `operator<`.

Using any of the algebraic types (`tuple`, `pair`, `optional`, or `variant`) directly in your interfaces and APIs is probably a mistake. You'll find that your code is easier to read, understand, and maintain if you write named classes for your own "domain-specific vocabulary" types, even if--*especially* if--they end up being thin wrappers around the algebraic types.

# Expressing alternatives with std::variant

Whereas `std::tuple<A,B,C>` is a *product type*, `std::variant<A,B,C>` is a *sum type*. A variant is allowed to hold either an `A`, a `B`, or a `C`--but never more (or less) than one of those at a time. Another name for this concept is *discriminated union*, because a variant behaves a lot like a native C++ `union`; but unlike a native `union`, a variant is always able to tell you which of its elements, `A`, `B`, or `C`, is "active" at any given time. The official name for these elements is "alternatives," since only one can be active at once:

[PRE13]

As with `tuple`, you can get a specific element of the `variant` using `std::get<I>(v)`. If your variant object's alternatives are all distinct (which should be the most common case, unless you're doing deep metaprogramming), you can use `std::get<T>(v)` with types as well as with indices--for an example, look at the preceding code sample, where `std::get<0>(v1)` and `std::get<int>(v1)` work interchangeably because the zeroth alternative in the variant `v1` is of type `int`. Unlike `tuple`, however, `std::get` on a variant is allowed to fail! If you call `std::get<double>(v1)` while `v1` currently holds a value of type `int`, then you'll get an exception of type `std::bad_variant_access`. `std::get_if` is the "non-throwing" version of `std::get`. As shown in the preceding example, `get_if` returns a *pointer* to the specified alternative if it's the active one, and otherwise returns a null pointer. Therefore the following code snippets are all equivalent:

[PRE14]

# Visiting variants

In the preceding example, we showed how when we had a variable `std::variant<int, double> v`, calling `std::get<double>(v)` would give us the current value *if* the variant currently held a `double`, but would throw an exception if the variant held an `int`. This might have struck you as odd--since `int` is convertible to `double`, why couldn't it just have given us the converted value?

We can get that behaviour if we want it, but not from `std::get`. We have to re-express our desire this way: "I have a variant. If it currently holds a `double`, call it `d`, then I want to get `double(d)`. If it holds an `int i`, then I want to get `double(i)`." That is, we have a list of behaviors in mind, and we want to invoke exactly one of those behaviors on whichever alternative is currently held by our variant `v`. The standard library expresses this algorithm by the perhaps obscure name `std::visit`:

[PRE15]

Generally speaking, when we `visit` a variant, all of the behaviors that we have in mind are fundamentally similar. Because we're writing in C++, with its overloading of functions and operators, we can generally express our similar behaviors using exactly identical syntax. If we can express them with identical syntax, we can wrap them up into a template function or--the most common case--a C++14 generic lambda, like this:

[PRE16]

Notice the use of C++17 `if constexpr` to take care of the one case that's fundamentally unlike the others. It's somewhat a matter of taste whether you prefer to use explicit switching on `decltype` like this, or to make a helper class such as the previous code sample's `Visitor` and rely on overload resolution to pick out the correct overload of `operator()` for each possible alternative.

There is also a variadic version of `std::visit` taking two, three, or even more `variant` objects, of the same or different types. This version of `std::visit` can be used to implement a kind of "multiple dispatch," as shown in the following code. However, you almost certainly will never need this version of `std::visit` unless you're doing really intense metaprogramming:

[PRE17]

# What about make_variant? and a note on value semantics

Since you can create a tuple object with `std::make_tuple`, or a pair with `make_pair`, you might reasonably ask, "Where is `make_variant`?" It turns out that there is none. The primary reason for its absence is that whereas `tuple` and `pair` are product types, `variant` is a sum type. To create a tuple, you always have to provide all *n* of its elements' values, and so the element types can always be inferred. With `variant`, you only have to provide one of its values--of type let's say `A`--but the compiler can't create a `variant<A,B,C>` object without knowing the identities of types `B` and `C` as well. So there'd be no point in providing a function `my::make_variant<A,B,C>(a)`, given that the actual class constructor can be spelled more concisely than that: `std::variant<A,B,C>(a)`.

We have already alluded to the secondary reason for the existence of `make_pair` and `make_tuple`: They automatically decay the special vocabulary type `std::reference_wrapper<T>` into `T&`, so that `std::make_pair(std::ref(a), std::cref(b))` creates an object of type `std::pair<A&, const B&>`. Objects of "pair-of-reference" or "tuple-of-reference" type behave very strangely: you can compare and copy them with the usual semantics, but when you assign to an object of this type, rather than "rebinding" the reference elements (so that they refer to the objects on the right-hand side), the assignment operator actually "assigns through," changing the values of the referred-to objects. As we saw in the code sample in section "Working with `std::tuple`", this deliberate oddity allows us to use `std::tie` as a sort of "multiple assignment" statement.

So another reason that we might expect or desire to see a `make_variant` function in the standard library would be for its reference-decaying ability. However, this is a moot point for one simple reason--the standard forbids making variants whose elements are reference types! We will see later in this chapter that `std::optional` and `std::any` are likewise forbidden from holding reference types. (However, `std::variant<std::reference_wrapper<T>, ...>` is perfectly legitimate.) This prohibition comes because the designers of the library have not come to a consensus as to what a variant of references should mean. Or, for that matter, what a *tuple* of references should mean! only reason we have tuples of references in the language today is because `std::tie` seemed like such a good idea in 2011\. In 2017, nobody is particularly eager to compound the confusion by introducing variants, optionals, or "anys" of references.

We have established that a `std::variant<A,B,C>` always holds exactly one value of type `A`, `B`, or `C`--no more and no less. Well, that's not technically correct. *Under very unusual circumstances,* it is possible to construct a variant with no value whatsoever. The only way to make this happen is to construct the variant with a value of type `A`, and then assign it a value of type `B` in such a way that the `A` is successfully destroyed but the constructor `B` throws an exception and the `B` is never actually emplaced. When this happens, the variant object enters a state known as "valueless by exception":

[PRE18]

This will never happen to you, unless you are writing code where your constructors or conversion operators throw exceptions. Furthermore, by using `operator=` instead of `emplace`, you can avoid valueless variants in every case except when you have a move constructor that throws:

[PRE19]

Recall from the discussion of `std::vector` in [Chapter 4](part0052.html#1HIT80-2fdac365b8984feebddfbb9250eaf20d), *The Container Zoo*, that your types' move constructors should always be marked `noexcept`; so, if you follow that advice religiously, you'll be able to avoid dealing with `valueless_by_exception` at all.

Anyway, when a variant *is* in this state, its `index()` method returns `size_t(-1)` (a constant also known as `std::variant_npos`) and any attempt to `std::visit` it will throw an exception of type `std::bad_variant_access`.

# Delaying initialization with std::optional

You might already be thinking that one potential use for `std::variant` would be to represent the notion of "Maybe I have an object, and maybe I don't." For example, we could represent the "maybe I don't" state using the standard tag type `std::monostate`:

[PRE20]

You'll be pleased to know that this is *not* the best way to accomplish that goal! The standard library provides the *vocabulary type* `std::optional<T>` specifically to deal with the notion of "maybe I have an object, maybe I don't."

[PRE21]

In the logic of algebraic data types, `std::optional<T>` is a sum type: it has exactly as many possible values as `T` does, plus one. This one additional value is called the "null," "empty," or "disengaged" state, and is represented in source code by the special constant `std::nullopt`.

Do not confuse `std::nullopt` with the similarly named `std::nullptr`! They have nothing in common except that they're both vaguely null-ish.

Unlike `std::tuple` and `std::variant` with their mess of free (non-member) functions, the `std::optional<T>` class is full of convenient member functions. `o.has_value()` is true if the optional object `o` currently holds a value of type `T`. The "has-value" state is commonly known as the "engaged" state; an optional object containing a value is "engaged" and an optional object in the empty state is "disengaged."

The comparison operators `==`, `!=`, `<`, `<=`, `>`, and `>=` are all overloaded for `optional<T>` if they are valid for `T`. To compare two optionals, or to compare an optional to a value of type `T`, all you need to remember is that an optional in the disengaged state compares "less than" any real value of `T`.

`bool(o)` is a synonym for `o.has_value()`, and `!o` is a synonym for `!o.has_value()`. Personally, I recommend that you always use `has_value`, since there's no difference in runtime cost; the only difference is in the readability of your code. If you do use the abbreviated conversion-to-`bool` form, be aware that for a `std::optional<bool>`, `o == false` and `!o` mean very different things!

`o.value()` returns a reference to the value contained by `o`. If `o` is currently disengaged, then `o.value()` throws an exception of type `std::bad_optional_access`.

`*o` (using the overloaded unary `operator*`) returns a reference to the value contained by `o`, without checking for engagement. If `o` is currently disengaged and you call `*o`, that's undefined behavior, just as if you called `*p` on a null pointer. You can remember this behavior by noticing that the C++ standard library likes to use punctuation for its most efficient, least sanity-checked operations. For example, `std::vector::operator[]` does less bounds-checking than `std::vector::at()`. Therefore, by the same logic, `std::optional::operator*` does less bounds-checking than `std::optional::value()`.

`o.value_or(x)` returns a copy of the value contained by `o`, or, if `o` is disengaged, it returns a copy of `x` converted to type `T`. We can use `value_or` to rewrite the preceding code sample into a one-liner of utter simplicity and readability:

[PRE22]

The preceding examples have shown how to use `std::optional<T>` as a way to handle "maybe a `T`" in flight (as a function return type, or as a parameter type). Another common and useful way to use `std::optional<T>` is as a way to handle "not yet a `T`" at rest, as a class data member. For example, suppose we have some type `L` which is not default-constructible, such as the closure type produced by a lambda expression:

[PRE23]

Then a class with a member of that type would also fail to be default-constructible:

[PRE24]

But, by giving our class a member of type `std::optional<L>`, we allow it to be used in contexts that require default-constructibility:

[PRE25]

It would be very difficult to implement this behavior without `std::optional`. You could do it with placement-new syntax, or using a `union`, but essentially you'd have to reimplement at least half of `optional` yourself. Much better to use `std::optional`!

And notice that if for some reason we wanted to get undefined behavior instead of the possibility of throwing from `call()`, we could just replace `fn_.value()` with `*fn_`.

`std::optional` is truly one of the biggest wins among the new features of C++17, and you'll benefit immensely by getting familiar with it.

From `optional`, which could be described as a sort of limited one-type `variant`, we now approach the other extreme: the algebraic-data-type equivalent of *infinity*.

# Revisiting variant

The `variant` data type is good at representing simple alternatives, but as of C++17, it is not particularly suitable for representing *recursive* data types such as JSON lists. That is, the following C++17 code will fail to compile:

[PRE26]

There are several possible workarounds. The most robust and correct is to continue using the C++11 Boost library `boost::variant`, which specifically supports recursive variant types via the marker type `boost::recursive_variant_`:

[PRE27]

You could also get around the problem by introducing a new class type called `JSONValue`, which either **HAS-A** or **IS-A** `std::variant` of the recursive type.

Notice that in the following example I chose HAS-A rather than IS-A; inheriting from non-polymorphic standard library types is almost always a really bad idea.

Since forward references to class types are acceptable to C++, this will compile:

[PRE28]

The final possibility is to switch to an algebraic type from the standard library that is even more powerful than `variant`.

# Infinite alternatives with std::any

To paraphrase Henry Ford, an object of type `std::variant<A, B, C>` can hold a value
of any type--as long as it's `A`, `B`, or `C`. But suppose we wanted to hold a value of *truly* any type? Perhaps our program will load plugins at runtime that might contain new types impossible to predict. We can't specify those types in a `variant`. Or perhaps we are in the "recursive data type" situation detailed in the preceding section.

For these situations, the C++17 standard library provides an algebraic-data-type version of "infinity": the type `std::any`. This is a sort of a container (see [Chapter 4](part0052.html#1HIT80-2fdac365b8984feebddfbb9250eaf20d), *The Container Zoo*) for a single object of any type at all. The container may be empty, or it may contain an object. You can perform the following fundamental operations on an `any` object:

*   Ask if it currently holds an object
*   Put a new object into it (destroying the old object, whatever it was)
*   Ask the type of the held object
*   Retrieve the held object, by correctly naming its type

In code the first three of these operations look like this:

[PRE29]

The fourth operation is a little more fiddly. It is spelled `std::any_cast`, and, like `std::get` for variants, it comes in two flavors: a `std::get`-like flavor that throws `std::bad_any_cast` on failure, and a `std::get_if`-like flavor that returns a null pointer on failure:

[PRE30]

Observe that in either case, you must name the type that you want to retrieve from the `any` object. If you get the type wrong, then you'll get an exception or a null pointer. There is no way to say "Give me the held object, no matter what type it is," since then what would be the type of that expression?

Recall that when we faced a similar problem with `std::variant` in the preceding section, we solved it by using `std::visit` to visit some generic code onto the held alternative. Unfortunately, there is no equivalent `std::visit` for `any`. The reason is simple and insurmountable: separate compilation. Suppose in one source file, `a.cc`, I have:

[PRE31]

And in another source file, `b.cc`, (perhaps compiled into a different plugin, `.dll`, or shared object file) I have:

[PRE32]

How should the compiler know, when compiling `b.cc`, that it needs to output a template instantiation for `size(Widget<int>&)` as opposed to, let's say, `size(Widget<double>&)`? When someone changes `a.cc` to return `make_any(Widget<char>&)`, how should the compiler know that it needs to recompile `b.cc` with a fresh instantiation of `size(Widget<char>&)` and that the instantiation of `size(Widget<int>&)` is no longer needed--unless of course we're anticipating being linked against a `c.cc` that *does* require that instantiation! Basically, there's no way for the compiler to figure out what kind of code-generation might possibly be needed by visitation, on a container that can by definition contain *any* type and trigger *any* code-generation.

Therefore, in order to extract any function of the contained value of an `any`, you must know up front what the type of that contained value might be. (And if you guess wrong--go fish!)

# std::any versus polymorphic class types

`std::any` occupies a position in between the compile-time polymorphism of `std::variant<A, B, C>` and the runtime polymorphism of polymorphic inheritance hierarchies and `dynamic_cast`. You might wonder whether `std::any` interacts with the machinery of `dynamic_cast` at all. The answer is "no, it does not"--nor is there any standard way to get that behavior. `std::any` is one hundred percent statically type-safe: there is no way to break into it and get a "pointer to the data" (for example, a `void *`) without knowing the exact static type of that data:

[PRE33]

# Type erasure in a nutshell

Let's look briefly at how `std::any` might be implemented by the standard library. The core idea is called "type erasure," and the way we achieve it is to identify the salient or relevant operations that we want to support for *all* types `T`, and then "erase" every other idiosyncratic operation that might be supported by any specific type `T`.

For `std::any`, the salient operations are as follows:

*   Constructing a copy of the contained object
*   Constructing a copy of the contained object "by move"
*   Getting `typeid` of the contained object

Construction and destruction are also required, but those two operations are concerned with the lifetime management of the contained object itself, not "what you can do with it," so at least in this case we don't need to consider them.

So we invent a polymorphic class type (call it `AnyBase`) which supports only those three operations as overrideable `virtual` methods, and then we create a brand-new derived class (call it `AnyImpl<T>`) each time the programmer actually stores an object of a specific type `T` into `any`:

[PRE34]

With these helper classes, the code to implement `std::any` becomes fairly trivial, especially when we use a smart pointer (see [Chapter 6](part0093.html#2OM4A0-2fdac365b8984feebddfbb9250eaf20d), *Smart Pointers*) to manage the lifetime of our `AnyImpl<T>` object:

[PRE35]

The preceding code sample omits the implementation of move-assignment. It can be done in the same way as copy-assignment, or it can be done by simply swapping the pointers. The standard library actually prefers to swap pointers when possible, because that is guaranteed to be `noexcept`; the only reason that you might see `std::any` *not* swapping pointers is if it uses a "small object optimization" to avoid heap allocation altogether for very small, nothrow-move-constructible types `T`. As of this writing, libstdc++ (the library used by GCC) will use small object optimization and avoid heap allocation for types up to 8 bytes in size; libc++ (the library used by Clang) will use small object optimization for types up to 24 bytes in size.

Unlike the standard containers discussed in [Chapter 4](part0052.html#1HIT80-2fdac365b8984feebddfbb9250eaf20d), *The Container Zoo*, `std::any` does *not* take an allocator parameter and does *not* allow you to customize or configure the source of its heap memory. If you use C++ on a real-time or memory-constrained system where heap allocation is not allowed, then you should not use `std::any`. Consider an alternative such as Tiemo Jung's `tj::inplace_any<Size, Alignment>`. If all else fails, you have now seen how to roll your own!

# std::any and copyability

Notice that our definition of `AnyImpl<T>::copy_to` required `T` to be copy-constructible. This is true of the standard `std::any` as well; there is simply no way to store a move-only type into a `std::any` object. The way to work around this is with a sort of a "shim" wrapper, whose purpose is to make its move-only object conform to the syntactic requirement of copy-constructibility while eschewing any actual copying:

[PRE36]

Notice the use of `std::optional<T>` in the preceding code sample; this guards our fake copy constructor against the possibility that `T` might not be default-constructible.

# Again with the type erasure: std::function

We observed that for `std::any`, the salient operations were as follows:

*   Constructing a copy of the contained object
*   Constructing a copy of the contained object "by move"
*   Getting the `typeid` of the contained object

Suppose we were to add one to this set of salient operations? Let's say our set is:

*   Constructing a copy of the contained object
*   Constructing a copy of the contained object "by move"
*   Getting the `typeid` of the contained object
*   Calling the contained object with a particular fixed sequence of argument types `A...`, and converting the result to some particular fixed type `R`

The type-erasure of this set of operations corresponds to the standard library type `std::function<R(A...)>`!

[PRE37]

Copying `std::function` always makes a copy of the contained object, if the contained object has state. Of course if the contained object is a function pointer, you won't observe any difference; but you can see the copying happen if you try it with an object of user-defined class type, or with a stateful lambda:

[PRE38]

Just as with `std::any`, `std::function<R(A...)` allows you to retrieve the `typeid` of the contained object, or to retrieve a pointer to the object itself as long as you statically know (or can guess) its type:

*   `f.target_type()` is the equivalent of `a.type()`
*   `f.target<T>()` is the equivalent of `std::any_cast<T*>(&a)`

[PRE39]

That said, I have never seen a use-case for these methods in real life. Generally, if you have to ask what the contained type of a `std::function` is, you've already done something wrong.

The most important use-case for `std::function` is as a vocabulary type for passing "behaviors" across module boundaries, where using a template would be impossible--for example, when you need to pass a callback to a function in an external library, or when you're writing a library that needs to receive a callback from its caller:

[PRE40]

We started this chapter talking about `std::string`, the standard vocabulary type for passing strings between functions; now, as the end of the chapter draws near, we're talking about `std::function`, the standard vocabulary type for passing *functions* between functions!

# std::function, copyability, and allocation

Just like `std::any`, `std::function` requires that whatever object you store in it must be copy-constructible. This can present a problem if you are using a lot of lambdas that capture `std::future<T>`, `std::unique_ptr<T>`, or other move-only types: such lambda types will be move-only themselves. One way to fix that was demonstrated in the *std::any and copyability* section in this chapter: we could introduce a shim that is syntactically copyable but throws an exception if you try to copy it.

When working with `std::function` and lambda captures, it might often be preferable to capture your move-only lambda captures by `shared_ptr`. We'll cover `shared_ptr` in the next chapter:

[PRE41]

Like `std::any`, `std::function` does *not* take an allocator parameter and does *not* allow you to customize or configure the source of its heap memory. If you use C++ on a real-time or memory-constrained system where heap allocation is not allowed, then you should not use `std::function`. Consider an alternative such as Carl Cook's `sg14::inplace_function<R(A...), Size, Alignment>`.

# Summary

Vocabulary types like `std::string` and `std::function` allow us to share a *lingua franca* for dealing with common programming concepts. In C++17, we have a rich set of vocabulary types for dealing with the *algebraic data types*: `std::pair` and `std::tuple` (product types), `std::optional` and `std::variant` (sum types), and `std::any` (the ultimate in sum types--it can store almost anything). However, don't get carried away and start using `std::tuple` and `std::variant` return types from every function! Named class types are still the most effective way to keep your code readable.

Use `std::optional` to signal the possible lack of a value, or to signal the "not-yet-ness" of a data member.

Use `std::get_if<T>(&v)` to query the type of a `variant`; use `std::any_cast<T>(&a)` to query the type of an `any`. Remember that the type you provide must be an exact match; if it's not, you'll get `nullptr`.

Be aware that `make_tuple` and `make_pair` do more than construct `tuple` and `pair` objects; they also decay `reference_wrapper` objects into native references. Use `std::tie` and `std::forward_as_tuple` to create tuples of references. `std::tie` is particularly useful for multiple assignment and for writing comparison operators. `std::forward_as_tuple` is useful for metaprogramming.

Be aware that `std::variant` always has the possibility of being in a "valueless by exception" state; but know that you don't have to worry about that case unless you write classes with throwing move-constructors. Separately: don't write classes with throwing move-constructors!

Be aware that the *type-erased* types `std::any` and `std::function` implicitly use the heap. Third-party libraries provide non-standard `inplace_` versions of these types. Be aware that `std::any` and `std::function` require copyability of their contained types. Use "capture by `shared_ptr`" to deal with this case if it arises.