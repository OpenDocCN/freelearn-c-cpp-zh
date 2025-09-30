# Chapter 2. Converting Data

In this chapter we will cover:

*   Converting strings to numbers
*   Converting numbers to strings
*   Converting numbers to numbers
*   Converting user-defined types to/from strings
*   Casting polymorphic objects
*   Parsing simple input
*   Parsing input

# Introduction

Now that we know some of the basic Boost types, it is time to get to know some data-converting functions. In this chapter we'll see how to convert strings, numbers, and user-defined types to each other, how to safely cast polymorphic types, and how to write small and large parsers right inside the C++ source files.

# Converting strings to numbers

Converting strings to numbers in C++ makes a lot of people depressed because of its inefficiency and user unfriendliness. Let's see how string `100` can be converted to `int`:

[PRE0]

C methods are not much better:

[PRE1]

## Getting ready

Only basic knowledge of C++ and STL is required for this recipe.

## How to do it...

There is a library in Boost which will help you cope with the depressing difficulty of string to number conversions. It is called `Boost.LexicalCast` and consists of a `boost::bad_lexical_cast` exception class and a few `boost::lexical_cast` functions:

[PRE2]

It can even be used for non-zero-terminated strings:

[PRE3]

## How it works...

The `boost::lexical_cast` function accepts string as input and converts it to the type specified in triangular brackets. The `boost::lexical_cast` function will even check bounds for you:

[PRE4]

And also check for the correct syntax of input:

[PRE5]

## There's more...

Lexical cast just like all of the `std::stringstreams` classes uses `std::locale` and can convert localized numbers, but also has an impressive set of optimizations for C locale and for locales without number groupings:

[PRE6]

And that isn't all! You can even simply create template functions for conversions to numbers. Let's make a function that converts a container of some `string` values to a vector of `long int` values:

[PRE7]

## See also

*   Refer to the *Converting numbers to strings* recipe for information about `boost::lexical_cast` performance.
*   The official documentation for `Boost.LexicalCast` contains some examples, performance measures, and answers to frequently asked questions. It is available at the following location:

    [http://www.boost.org/doc/libs/1_53_0/doc/html/boost_lexical_cast.html](http://www.boost.org/doc/libs/1_53_0/doc/html/boost_lexical_cast.html)

# Converting numbers to strings

In this recipe we will continue discussing lexical conversions, but now we will be converting numbers to strings using `Boost.LexicalCast`. And as usual, `boost::lexical_cast` will provide a very simple way to convert the data.

## Getting ready

Only basic knowledge of C++ and STL is required for this recipe.

## How to do it...

1.  Let's convert integer `100` to `std::string` using `boost::lexical_cast`:

    [PRE8]

2.  Compare this to the traditional C++ conversion method:

    [PRE9]

    And against the C conversion method:

    [PRE10]

## How it works...

The `boost::lexical_cast` function may also accept numbers as input and convert them to the string type specified in triangular brackets. Pretty close to what we did in the previous recipe.

## There's more...

A careful reader will note that in the case of `lexical_cast` we have an additional call to string copy the constructor and that such a call will be a hit on the performance. It is true, but only for old or bad compilers. Modern compilers implement a **named return value optimization** (**NRVO**), which will eliminate the unnecessary call to copy the constructor and destructor. Even if the C++11-compatible compilers don't detect NRVO, they will use a move copy constructor of `std::string`, which is fast and efficient. The *Performance* section of the `Boost.LexicalCast` documentation shows the conversion speed on different compilers for different types, and in most cases `lexical_cast` is faster than the `std::stringstream` and `printf` functions.

If `boost::array` or `std::array` is passed to `boost::lexical_cast` as the output parameter type, less dynamic memory allocations will occur (or there will be no memory allocations at all; it depends on the `std::locale` implementation).

## See also

*   Boost's official documentation contains tables that compare the `lexical_cast` performance against other conversion approaches. And in most cases it wins. [http://www.boost.org/doc/libs/1_53_0/doc/html/boost_lexical_cast.html](http://www.boost.org/doc/libs/1_53_0/doc/html/boost_lexical_cast.html). It also has some more examples and a frequently asked questions section.
*   The *Converting strings to numbers* recipe.
*   The *Converting user-defined types to/from strings* recipe.

# Converting numbers to numbers

You might remember situations where you wrote something like the following code:

[PRE11]

Usually, programmers just ignore such warnings by implicitly casting to unsigned short datatype, as demonstrated in the following code snippet:

[PRE12]

But this may make it extremely hard to detect errors. Such errors may exist in code for years before they get caught:

[PRE13]

## Getting ready

Only basic knowledge of C++ is required for this recipe.

## How to do it...

1.  The library `Boost.NumericConversion` provides a solution for such cases. And it is easy to modify the existing code to use safe casts, just replace `static_cast` with `boost::numeric_cast`. It will throw an exception when the source value cannot be stored in the target. Let's take a look at the following example:

    [PRE14]

2.  Now if we run `test_function()` it will output the following:

    [PRE15]

3.  We can even detect specific overflow types:

    [PRE16]

    The `test_function1()`function will output the following:

    [PRE17]

## How it works...

It checks if the value of the input parameter fits into the new type without losing data and throws an exception if something is lost during conversion.

The `Boost.NumericConversion` library has a very fast implementation; it can do a lot of work at compile time. For example, when converting to types of a wider range, the source will just call the `static_cast` method.

## There's more...

The `boost::numeric_cast` function is implemented via `boost::numeric::converter`, which can be tuned to use different overflow, range checking, and rounding policies. But usually, `numeric_cast` is just what you need.

Here is a small example that demonstrates how to make our own `mythrow_overflow_handler` overflow handler for `boost::numeric::cast`:

[PRE18]

And this will output the following:

[PRE19]

## See also

*   Boost's official documentation contains detailed descriptions of all of the template parameters of the numeric converter; it is available at the following link:

    [http://www.boost.org/doc/libs/1_53_0/libs/numeric/conversion/doc/html/index.html](http://www.boost.org/doc/libs/1_53_0/libs/numeric/conversion/doc/html/index.html)

# Converting user-defined types to/from strings

There is a feature in `Boost.LexicalCast` that allows users to use their own types in `lexical_cast`. This feature just requires the user to write the correct `std::ostream` and `std::istream` operators for their types.

## How to do it...

1.  All you need is to provide an `operator<< and operator>> stream` operators. If your class is already streamable, nothing needs to be done:

    [PRE20]

2.  Now we may use `boost::lexical_cast` for conversions to and from the `negative_number` class. Here's an example:

    [PRE21]

## How it works...

The `boost::lexical_cast` function can detect and use stream operators for converting user-defined types.

The `Boost.LexicalCast` library has many optimizations for basic types and they will be triggered when a user-defined type is being cast to basic type or when a basic type is being cast to a user-defined type.

## There's more...

The `boost::lexical_cast` function may also convert to wide character strings, but the correct `basic_istream` and `basic_ostream` operator overloads are required for that:

[PRE22]

The `Boost.LexicalCast` library is not a part of C++11, but there is a proposal to add it to C++ standard. A lot of Boost libraries use it and I hope that it will make your life easier as well.

## See also

*   The `Boost.LexicalCast` documentation contains some examples, performance measures, and answers to frequently asked questions; it is available at [http://www.boost.org/doc/libs/1_53_0/doc/html/boost_lexical_cast.html](http://www.boost.org/doc/libs/1_53_0/doc/html/boost_lexical_cast.html)
*   The *Converting strings to numbers* recipe
*   The *Converting numbers to strings* recipe

# Casting polymorphic objects

Imagine that some programmer designed an awful interface as follows (this is a good example of how interfaces should not be written):

[PRE23]

And our task is to make a function that eats bananas, and throws exceptions if something instead of banana came along (eating pidgins gross!). If we dereference a value returned by the `try_produce_banana()` function, we are getting in danger of dereferencing a null pointer.

## Getting ready

Basic knowledge of C++ is required for this recipe.

## How to do it...

So we need to write the following code:

[PRE24]

Ugly, isn't it? `Boost.Conversion` provides a slightly better solution:

[PRE25]

## How it works...

The `boost::polymorphic_cast` function just wraps around code from the first example, and that is all. It checks input for null and then tries to do a dynamic cast. Any error during those operations will throw a `std::bad_cast` exception.

## There's more...

The `Boost.Conversion` library also has a `polymorphic_downcast` function, which should be used only for downcasts that will always succeed. In debug mode (when `NDEBUG` is not defined) it will check for the correct downcast using `dynamic_cast`. When `NDEBUG` is defined, the `polymorphic_downcast` function will just do a `static_cast` operation. It is a good function to use in performance-critical sections, while still leaving the ability to detect errors in debug compilations.

## See also

*   Initially, the `polymorphic_cast` idea was proposed in the book *The C++ Programming Language*, *Bjarne Stroustrup*. Refer to this book for more information and some good ideas on different topics.
*   The official documentation may also be helpful; it is available at [http://www.boost.org/doc/libs/1_53_0/libs/conversion/cast.htm](http://www.boost.org/doc/libs/1_53_0/libs/conversion/cast.htm).

# Parsing simple input

It is a common task to parse a small text. And such situations are always a dilemma: shall we use some third-party professional tools for parsing such as Bison or ANTLR, or shall we try to write it by hand using only C++ and STL? The third-party tools are good for handling the parsing of complex texts and it is easy to write parsers using them, but they require additional tools for creating C++ or C code from their grammar, and add more dependencies to your project. Handwritten parsers are usually hard to maintain, but they require nothing except C++ compiler.

![Parsing simple input](img/4880OS_02_02.jpg)

Let's start with a very simple task to parse a date in ISO format as follows:

[PRE26]

The following are the examples of possible input:

[PRE27]

Let's take a look at the parser's grammar from the following link [http://www.ietf.org/rfc/rfc3339.txt](http://www.ietf.org/rfc/rfc3339.txt):

[PRE28]

## Getting ready

Make sure that you are familiar with the placeholders concept or read the *Reordering the parameters of function* and *Binding a value as a function parameter* recipes in [Chapter 1](ch01.html "Chapter 1. Starting to Write Your Application"), *Starting to Write Your Application*. Basic knowledge of parsing tools would be good.

## How to do it...

Let me introduce you to a `Boost.Spirit` library. It allows writing parsers (and lexers and generators) directly in C++ code format, which are immediately executable (that is, do not require additional tools for C++ code generation). The grammar of `Boost`.`Spirit` is very close to **Extended Backus-Naur Form** (**EBNF**), which is used for expressing grammar by many standards and understood by other popular parsers. The grammar at the beginning of this chapter is in EBNF.

1.  We need to include the following headers:

    [PRE29]

2.  Now it's time to make a `date` structure to hold the parsed data:

    [PRE30]

3.  Now let's look at the parser (a step-by-step description of how it works can be found in the next section):

    [PRE31]

4.  Now we may use this parser wherever we want:

    [PRE32]

## How it works...

This is a very simple implementation; it does not check the digit count for numbers. Parsing occurs in the `boost::spirit::qi::parse` function. Let's simplify it a little bit, removing the actions on successful parsing:

[PRE33]

The `first` argument points to the beginning of the data to parse; it must be a modifiable (non-constant) variable because the `parse` function will use it to show the end of the parsed sequence. The `end` argument points to the element beyond the last one. `first` and `end` shall be iterators.

The third argument to the function is a parsing rule. And it does exactly what is written in the EBNF rule:

[PRE34]

We just replaced white spaces with the `>>` operator.

The `parse` function returns true on success. If we want to make sure that the whole string was successfully parsed, we need to check for the parser's return value and equality of the input iterators.

Now we need to deal with the actions on successful parse and this recipe will be over. Semantic actions in `Boost.Spirit` are written inside `[]` and they can be written using function pointers, function objects, `boost::bind`, `std::bind` (or the other `bind()` implementations), or C++11 lambda functions.

So, you could also write a rule for `YYYY` using C++11 lambda:

[PRE35]

Now, let's take a look at the month's semantic action closer:

[PRE36]

For those who have read the book from the beginning, this would remind you about `boost::bind` and placeholders. `ref(res.month)` means pass `res.month` as a modifiable reference and `_1` means the first input parameter, which would be a number (the result of `ushort_ parsing`).

## There's more...

Now let's modify our parser, so it can take care of the digits count. For that purpose, we will take the `unit_parser` template class and just set up the correct parameters:

[PRE37]

Don't worry if those examples seem complicated. The first time I was also frightened by `Boost.Spirit`, but now it really simplifies my life. You are extremely brave, if this code does not scare you.

If you want to avoid code bloat, try to write parsers in source files and not in headers. Also take care of iterator types passed to the `boost::spirit::parse` function, the fewer different types of iterators you use, the smaller binary you'll get. Writing parsers in source files has one more advantage: it does not slow down the project compilation (as you may notice, the `Spirit` parsers are slow to compile, so it is better to compile them once in the source file, than define them in the header files and use this file all around the project).

If you are now thinking that parsing dates was simpler to implement by hand using STL... you are right! But only for now. Take a look at the next recipe; it will give you more examples on `Boost.Spirit` usage and extend this example for a situation when writing the parser by hand is harder than using `Boost.Spirit`.

The `Boost.Spirit` library is not a part of C++11 and as far as I know, it is not proposed for inclusion in the closest upcoming C++ standard.

## See also

*   The *Reordering the parameters of function* recipe in [Chapter 1](ch01.html "Chapter 1. Starting to Write Your Application"), *Starting to Write Your Application*.
*   The *Binding a value as a function parameter* recipe.
*   `Boost.Spirit` is a huge header-only library. A separate book may be written about it, so feel free to use its documentation [http://www.boost.org/doc/libs/1_53_0/libs/spirit/doc/html/index.html](http://www.boost.org/doc/libs/1_53_0/libs/spirit/doc/html/index.html). You may also find information on how to write lexers and generators directly in C++11 code using Boost.

# Parsing input

In the previous recipe we were writing a simple parser for dates. Imagine that some time has passed and the task has changed. Now we need to write a date-time parser that will support multiple input formats plus zone offsets. So now our parser should understand the following inputs:

[PRE38]

## Getting ready

We'll be using the `Spirit` library, which was described in the *Parsing simple input* recipe. Read it before getting hands on with this recipe.

## How to do it...

1.  Let's start with writing a date-time structure that will hold a parsed result:

    [PRE39]

2.  Now let's write a function for setting the zone offset:

    [PRE40]

3.  Writing a parser can be split into writing a few simple parsers, so we start with writing a zone-offset parser.

    [PRE41]

4.  Let's finish our example by writing the remaining parsers:

    [PRE42]

## How it works...

A very interesting method here is `boost::spirit::qi::rule<const char*, void()>`. It erases the type and allows you to write parsers in source files and export them to headers. For example:

[PRE43]

But remember that this class implies an optimization barrier for compilers, so do not use it when it is not required.

## There's more...

We can make our example slightly faster by removing the `rule<>` objects that do type erasure. For our example in C++11, we can just replace them with the `auto` keyword.

The `Boost.Spirit` library generates very fast parsers; there are some performance measures at the official site. There are also some recommendations for working with the `Boost.Spirit` library; one of them is to generate a parser only once, and then just re-use it (in our example this is not shown).

The rule that parses specific zone offset in `timezone_parser` uses the `boost::phoenix::bind` call, which is not mandatory. However, without it we'll be dealing with `boost::fusion::vector<char, unsigned short, unsigned short>`, which is not as user friendly as `bind(&set_zone_offset, ref(ret), _1, _2, _3)`.

When parsing large files, consider reading the *The fastest way to read files* recipe in [Chapter 11](ch11.html "Chapter 11. Working with the System"), *Working with the System*, because incorrect work with files may slow down your program much more than parsing.

Compiling the code that uses the library `Boost.Spirit` (or `Boost.Fusion`) may take a lot of time, because of a huge number of template instantiations. When experimenting with the `Boost.Spirit` library use modern compilers, they provide better compilation times.

## See also

*   The `Boost.Spirit` library is worth writing a separate book on. It's impossible to describe all of its features in a few recipes, so referring to the documentation will help you to get more information about it. It is available at [http://www.boost.org/doc/libs/1_53_0/libs/spirit/doc/html/index.html](http://www.boost.org/doc/libs/1_53_0/libs/spirit/doc/html/index.html). There you'll find many more examples, ready parsers, and information on how to write lexers and generators directly in C++11 code using Boost.