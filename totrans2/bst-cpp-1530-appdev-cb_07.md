# Chapter 7. Manipulating Strings

In this chapter we will cover:

*   Changing cases and case-insensitive comparison
*   Matching strings using regular expressions
*   Searching and replacing strings using regular expressions
*   Formatting strings using safe printf-like functions
*   Replacing and erasing strings
*   Representing a string with two iterators
*   Using a reference to string type

# Introduction

This whole chapter is devoted to different aspects of changing, searching, and representing strings. We'll see how some common string-related tasks can be easily done using the Boost libraries. This chapter is easy enough; it addresses very common string manipulation tasks. So, let's begin!

# Changing cases and case-insensitive comparison

This is a pretty common task. We have two non-Unicode or ANSI character strings:

[PRE0]

We need to compare them in a case-insensitive manner. There are a lot of methods to do that; let's take a look at Boost's.

## Getting ready

Basic knowledge of `std::string` is all we need here.

## How to do it...

Here are some different ways to do case-insensitive comparisons:

1.  The most trivial one is:

    [PRE1]

2.  Using the Boost predicate and STL method:

    [PRE2]

3.  Making a lowercase copy of both the strings:

    [PRE3]

4.  Making an uppercase copy of the original strings:

    [PRE4]

5.  Converting the original strings to lowercase:

    [PRE5]

## How it works...

The second method is not an obvious one. In the second method, we compare the length of the strings; if they have the same length, we compare the strings character by character using an instance of the `boost::is_iequal` predicate. The `boost::is_iequal` predicate compares two characters in a case-insensitive way.

### Note

The `Boost.StringAlgorithm` library uses `i` in the name of the method or class, if this method is case-insensitive. For example, `boost::is_iequal`, `boost::iequals`, `boost::is_iless`, and others.

## There's more...

Each function and the functional object of the `Boost.StringAlgorithm` library that work with cases accept `std::locale`. By default (and in our examples), methods and classes use a default constructed `std::locale`. If we work a lot with strings, it may be a good optimization to construct a `std::locale` variable once and pass it to all the methods. Another good optimization would be to use the 'C' locale (if your application logic permits that) via `std::locale::classic()`:

[PRE6]

### Note

Nothing forbids you to use both optimizations.

Unfortunately, C++11 has no string functions from `Boost.StringAlgorithm`. All the algorithms are fast and reliable, so do not be afraid to use them in your code.

## See also

*   Official documentation on the Boost String Algorithms library can be found at [http://www.boost.org/doc/libs/1_53_0/doc/html/string_algo.html](http://www.boost.org/doc/libs/1_53_0/doc/html/string_algo.html)
*   See the *C++ Coding Standards* book by Andrei Alexandrescu and Herb Sutter for an example on how to make a case-insensitive string with a few lines of code

# Matching strings using regular expressions

Let's do something useful! It's common that the user's input must be checked using some regular expression-specific pattern that provides a flexible means of match. The problem is that there are a lot of regex syntaxes; expressions written using one syntax are not handled well by the other syntax. Another problem is that long regexes are not easy to write.

So in this recipe, we'll write a program that may use different types of regular expression syntaxes and checks that the input strings match the specified regexes.

## Getting ready

This recipe requires basic knowledge of STL. Knowledge of regular expression syntax can be helpful, but it is not really required.

Linking examples against the `libboost_regex` library is required.

## How to do it...

This regex matcher consists of a few lines of code in the `main()` function; however, I use it a lot. It'll help you some day.

1.  To implement it, we'll need the following headers:

    [PRE7]

2.  At the start of the program, we need to output the available regex syntaxes:

    [PRE8]

3.  Now correctly set up flags, according to the chosen syntax:

    [PRE9]

4.  Now we'll be requesting regex patterns in a loop:

    [PRE10]

5.  Getting a string to match in a loop:

    [PRE11]

6.  Applying regex to it and outputting the result:

    [PRE12]

7.  Finishing our example by restoring `std::cin` and requesting new regex patterns:

    [PRE13]

    Now if we run the preceding example, we'll get the following output:

    [PRE14]

## How it works...

All this is done by the `boost::regex` class. It constructs an object that is capable of regex parsing and compilation. The `flags` variable adds additional configuration options.

If the regular expression is incorrect, it throws an exception; if the `boost::regex::no_except` flag was passed, it reports an error returning as non-zero in the `status()` call (just like in our example):

[PRE15]

This will result in:

[PRE16]

Regular expression matching is done by a call to the `boost::regex_match` function. It returns `true` in case of a successful match. Additional flags may be passed to `regex_match`, but we avoided their usage for brevity of the example.

## There's more...

C++11 contains almost all the `Boost.Regex` classes and flags. They can be found in the `<regex>` header of the `std::` namespace (instead of `boost::`). Official documentation provides information about the differences between C++11 and `Boost.Regex`. It also contains some performance measures that tell `Boost.Regex` is fast.

## See also

*   The *Searching and replacing strings using regular expressions* recipe will give you more information about `Boost.Regex` usage
*   You may also consider official documentation to get more information about flags, performance measures, regular expression syntaxes, and C++11 conformance at [http://www.boost.org/doc/libs/1_53_0/libs/regex/doc/html/index.html](http://www.boost.org/doc/libs/1_53_0/libs/regex/doc/html/index.html)

# Searching and replacing strings using regular expressions

My wife enjoyed the *Matching strings using regular expressions* recipe very much and told me that I'll get no food until I improve it to be able to replace parts of the input string according to a regex match. Each matched subexpression (part of the regex in parenthesis) must get a unique number starting from 1; this number will be used to create a new string.

This is how an updated program will work like:

[PRE17]

## Getting ready

We'll be using the code from the *Matching strings using regular expressions* recipe. You should read it before getting your hands on this one.

Linking the example against the `libboost_regex` library is required.

## How to do it...

This recipe is based on the code from the previous one. Let's see what must be changed.

1.  No additional headers will be included; however, we'll need an additional string to store the replace pattern:

    [PRE18]

2.  We'll replace `boost::regex_match` with `boost::regex_find` and output matched results:

    [PRE19]

3.  After that, we need to get the replace pattern and apply it:

    [PRE20]

That's it! Everyone's happy and I'm fed.

## How it works...

The `boost::regex_search` function doesn't only return a true or a false (such as the `boost::regex_match` function does) value, but also stores matched parts. We output matched parts using the following construction:

[PRE21]

Note that we outputted the results by skipping the first result (`results.begin() + 1`); that is because `results.begin()` contains the whole regex match.

The `boost::regex_replace` function does all the replacing and returns the modified string.

## There's more...

There are different variants of the `regex_*` function; some of them receive bidirectional iterators instead of strings and some provide output to the iterator.

`boost::smatch` is a `typedef` for `boost::match_results<std::string::const_iterator>`; so if you are using some other bidirectional iterators instead of `std::string::const_iterator`, you will need to use the type of your bidirectional iterators as a template parameter for `match_results`.

`match_results` has a format function, so we can tune our example with it. Instead of:

[PRE22]

We may use the following:

[PRE23]

By the way, `replace_string` may have different formats:

[PRE24]

All the classes and functions from this recipe exist in C++11, in the `std::` namespace of the `<regex>` header.

## See also

*   The official documentation on `Boost.Regex` will give you more examples and information about performance, C++11 standard compatibility, and regular expression syntax at [http://www.boost.org/doc/libs/1_53_0/libs/regex/doc/html/index.html](http://www.boost.org/doc/libs/1_53_0/libs/regex/doc/html/index.html). The *Matching strings using regular expressions* recipe will tell you the basics of `Boost.Regex`.

# Formatting strings using safe printf-like functions

The `printf` family of functions is a threat to security. It is a very bad design to allow users to put their own strings as a type and format the specifiers. So what do we do when user-defined format is required? How shall we implement the `std::string to_string(const std::string& format_specifier) const;` member function of the following class?

[PRE25]

## Getting ready

Basic knowledge of STL is more than enough for this recipe.

## How to do it...

We wish to allow users to specify their own output format for a string.

1.  To do that in a safe manner, we'll need the following header:

    [PRE26]

2.  Now we will add some comments for the user:

    [PRE27]

3.  Now it is time to make all of them work:

    [PRE28]

    That's all. Take a look at this code:

    [PRE29]

    Imagine that `class_instance` has a member `i` equal to `100`, an `s` member equal to `"Reader"`, and a member `c` equal to `'!'`. Then, the program will output the following:

    [PRE30]

## How it works...

The `boost::format` class accepts the string that specifies the resulting string. Arguments are passed to `boost::format` using `operator%`. Values `%1%`, `%2%`, `%3%`, `%4%`, and so on, in the format specifying string, will be replaced by arguments passed to `boost::format`.

We disable the exceptions for cases when a format string contains fewer arguments than passed to `boost::format`:

[PRE31]

This is done to allow some formats like this:

[PRE32]

## There's more...

And what will happen in case of an incorrect format?

[PRE33]

Well, in that case, no assertion will be triggered and the following lines will be outputted to the console:

[PRE34]

C++11 has no `std::format`. The `Boost.Format` library is not a very fast library; try not to use it much in performance critical sections.

## See also

*   The official documentation contains more information about the performance of the `Boost.Format` library. More examples and documentation on extended printf-like format is available at [http://www.boost.org/doc/libs/1_53_0/libs/format/](http://www.boost.org/doc/libs/1_53_0/libs/format/)

# Replacing and erasing strings

Situations where we need to erase something in a string, replace a part of the string, or erase the first or last occurrence of some substring are very common. STL allows us to do most of this, but it usually involves writing too much code.

We saw the `Boost.StringAlgorithm` library in action in the *Changing cases and case-insensitive comparison* recipe. Let's see how it can be used to simplify our lives when we need to modify some strings:

[PRE35]

## Getting ready

Basic knowledge of C++ is required for this example.

## How to do it...

This recipe shows how different string-erasing and replacing methods from the `Boost.StringAlgorithm` library work.

Erasing requires the `#include <boost/algorithm/string/erase.hpp>` header:

[PRE36]

This code will output the following:

[PRE37]

Replacing requires the `<boost/algorithm/string/replace.hpp>` header:

[PRE38]

This code will output the following:

[PRE39]

## How it works...

All the examples are self-documenting. The only one that is not obvious is the `replace_head_copy` function. It accepts a number of bytes to replace as a second parameter and a replace string as the third parameter. So, in the preceding example, `Hello` gets replaced with `Whaaaaaaa!`.

## There's more...

There are also methods that modify strings in-place. They don't just end on `_copy` and return `void`. All the case insensitive methods (the ones that start with `i`) accept `std::locale` as the last parameter, and use a default constructed locale as a default parameter.

C++11 does not have `Boost.StringAlgorithm` methods and classes.

## See also

*   The official documentation contains a lot of examples and a full reference on all the methods at [http://www.boost.org/doc/libs/1_53_0/doc/html/string_algo.html](http://www.boost.org/doc/libs/1_53_0/doc/html/string_algo.html)
*   See the *Changing cases and case-insensitive comparison* recipe from this chapter for more information about the `Boost.StringAlgorithm` library.

# Representing a string with two iterators

There are situations when we need to split some strings into substrings and do something with those substrings. For example, count whitespaces in the string and, of course, we want to use Boost and be as efficient as possible.

## Getting ready

You'll need some basic knowledge of STL algorithms for this recipe.

## How to do it...

We won't be counting whitespaces; instead we'll split the string into sentences. You'll see that it is very easy with Boost.

1.  First of all, include the right headers:

    [PRE40]

2.  Now let's define our test string:

    [PRE41]

3.  Now we make a `typedef` for our splitting iterator:

    [PRE42]

4.  Construct that iterator:

    [PRE43]

5.  Now we can iterate between matches:

    [PRE44]

6.  Count the number of characters:

    [PRE45]

7.  And count the whitespaces:

    [PRE46]

    That's it. Now if we run this example, it will output:

    [PRE47]

## How it works...

The main idea of this recipe is that we do not need to construct `std::string` from substrings. We even do not need to tokenize the whole string at once. All we need to do is find the first substring and return it as a pair of iterators to the beginning and to the end of substring. If we need more substrings, find the next substring and return a pair of iterators for that substring.

![How it works...](img/4880OS_07_02.jpg)

Now let's take a closer look at `boost::split_iterator`. We constructed one using the `boost::make_split_iterator` function that takes `range` as the first argument and a binary finder predicate (or binary predicate) as the second. When `split_iterator` is dereferenced, it returns the first substring as `boost::iterator_range<const char*>`, which just holds a pair of iterators and has a few methods to work with them. When we increment `split_iterator`, it will try to find the next substring, and if there is no substring found, `split_iterator::eof()` will return `true`.

## There's more...

The `boost::iterator_range` class is widely used across all the Boost libraries. You may find it useful for your own code and libraries in situations where a pair of iterators must be returned or where a function should accept/work with a pair of iterators.

The `boost::split_iterator<>` and `boost::iterator_range<>` classes accept a forward iterator type as a template parameter. Because we were working with a character array in the preceding example, we provided `const char*` as iterators. If we were working with `std::wstring`, we would need to use the `boost::split_iterator<std::wstring::const_iterator>` and `boost::iterator_range<std::wstring::const_iterator>` types.

C++11 has neither `iterator_range` nor `split_iterator`.

As the `boost::iterator_range` class has no virtual functions and no dynamic memory allocations, it is fast and efficient. However, its output stream operator `<<` has no specific optimizations for character arrays, so streaming it is slow.

The `boost::split_iterator` class has a `boost::function` class in it, so constructing it may be slow; however, iterating adds only a tiny overhead that you won't notice even in performance critical sections.

## See also

*   The next recipe will tell you about a nice replacement for `boost::iterator_range<const char*>`.
*   The official documentation for `Boost.StringAlgorithm` will provide you with more detailed information about classes and a whole bunch of examples at [http://www.boost.org/doc/libs/1_53_0/doc/html/string_algo.html](http://www.boost.org/doc/libs/1_53_0/doc/html/string_algo.html).
*   More information about `boost::iterator_range` can be found here: [http://www.boost.org/doc/libs/1_53_0/libs/range/doc/html/range/reference/utilities.html](http://www.boost.org/doc/libs/1_53_0/libs/range/doc/html/range/reference/utilities.html). It is a part of the `Boost.Range` library that is not described in this book, but you may wish to study it by yourself.

# Using a reference to string type

This recipe is the most important recipe in this chapter! Let's take a look at a very common case, where we write a function that accepts a string and returns the part of the string between character values passed in the `starts` and `ends` arguments:

[PRE48]

Do you like this implementation? In my opinion, it looks awful; consider the following call to it:

[PRE49]

In this example, a temporary `std::string` variable will be constructed from `"Getting expression (between brackets)"`. The character array is long enough, so there is a big chance that dynamic memory allocation will be called inside the `std::string` constructor and the character array will be copied into it. Then, somewhere inside the `between_str` function, new `std::string` will be constructed, which may also lead to another dynamic memory allocation and result in copying.

So, this simple function may, and in most cases will:

*   Call dynamic memory allocation (twice)
*   Copy string (twice)
*   Deallocate memory (twice)

Can we do better?

## Getting ready

This recipe requires basic knowledge of STL and C++.

## How to do it...

We do not really need a `std::string` class here, we only need some pointer to the character array and the array's size. Boost has the `std::string_ref` class.

1.  To use the `boost::string_ref` class, include the following header:

    [PRE50]

2.  Change the method's signature:

    [PRE51]

3.  Change `std::string` to `boost::string_ref:` everywhere inside the function body:

    [PRE52]

4.  The `boost::string_ref` constructor accepts size as a second parameter, so we need to slightly change the code:

    [PRE53]

    That's it! Now we may call `between("Getting expression (between brackets)", '(', ')')` and it will work without any dynamic memory allocation and characters copying. And we can still use it for `std::string`:

    [PRE54]

## How it works...

As already mentioned, `boost::string_ref` contains only a pointer to the character array and size of data. It has a lot of constructors and may be initialized in different ways:

[PRE55]

The `boost::string_ref` class has all the methods required by the container class, so it is usable with STL algorithms and Boost algorithms:

[PRE56]

### Note

The `boost::string_ref` class does not really own string, so all its methods return constant iterators. Because of that, we cannot use it in methods that modify data, such as `boost::to_lower(r)`.

While working with `boost::string_ref`, we should take additional care about data that it refers to; it must exist and be valid for the whole lifetime of `boost::string_ref`.

## There's more...

The `boost::string_ref` class is not a part of C++11, but it is proposed for inclusion in the next standard.

The `string_ref` classes are fast and efficient; use them wherever it is possible.

The `boost::string_ref` class is actually a typedef in the `boost::` namespace:

[PRE57]

You may also find useful the following typedefs for wide characters in the `boost::` namespace:

[PRE58]

## See also

*   The official `string_ref` proposal for inclusion in C++ standard can be found at [http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2012/n3442.html](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2012/n3442.html)
*   Boost documentation for `string_ref` could be found at [http://www.boost.org/doc/libs/1_53_0/libs/utility/doc/html/string_ref.html](http://www.boost.org/doc/libs/1_53_0/libs/utility/doc/html/string_ref.html)