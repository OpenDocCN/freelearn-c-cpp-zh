# Regular Expressions

In the previous chapter, we learned all about formatted input and output in C++. We saw that there are good solutions for formatted output--as long as you make sure you're in the `C` locale--but that despite the many approaches to input parsing, even the simple task of parsing an `int` out of a string can be quite difficult. (Recall that of the two most foolproof methods, `std::stoi(x)` requires converting `x` to a heap-allocated `std::string`, and the verbose `std::from_chars(x.begin(), x.end(), &value, 10)` is lagging the rest of C++17 in vendor adoption.) The fiddliest part of parsing numbers is figuring out what to do with the part of the input that *isn't* numeric!

Parsing gets easier if you can split it into two subtasks: First, figure out exactly how many bytes of the input correspond to one "input item" (this is called *lexing*); and second, parse the value of that item, with some error recovery in the case that the item's value is out of range or otherwise nonsensical. If we apply this approach to integer input, *lexing* corresponds to finding the longest initial sequence of digits in the input, and *parsing* corresponds to computing the numeric value of that sequence in base 10.

*Regular expressions* (or *regexes*) is a tool provided by many programming languages that solve the lexing problem, not just for sequences of digits but for arbitrarily complicated input formats. Regular expressions have been part of the C++ standard library since 2011, in the `<regex>` header. In this chapter we'll show you how to use regexes to simplify some common parsing tasks.

Bear in mind that regexes are likely to be overkill for *most* parsing tasks that you'll face in your daily work. They can be slow and bloated, and unavoidably require heap allocation (that is, the regex data types are not *allocator-aware* as described in [Chapter 8](part0129.html#3R0OI0-2fdac365b8984feebddfbb9250eaf20d), *Allocators*). Where regexes really shines is for complicated tasks where hand-written parsing code would be just as slow anyway; and for extremely simple tasks where the readability and robustness of regular expressions outweigh their performance costs. In short, regex support has taken C++ one step closer to the everyday usability of scripting languages such as Python and Perl.

In this chapter we'll learn:

*   "Modified ECMAScript", the dialect used by C++ regexes
*   How to match, search, and even replace substrings using regexes
*   Further dangers of dangling iterators
*   Regex features to avoid

# What are regular expressions?

A *regular expression* is a way of writing down the rules for recognizing a string of bytes or characters as belonging (or not belonging) to a certain "language." In this context, a "language" can be anything from "the set of all digit-sequences" to "the set of all sequences of valid C++ tokens." Essentially, a "language" is just a rule for dividing the world of all strings into two sets--the set of strings matching the rules of the language, and the set of strings that *don't* match.

Some kinds of languages follow simple enough rules that they can be recognized by a *finite state machine*, a computer program with no memory at all--just a program counter and a pointer that scans over the input in a single pass. The language of "digit-sequences" is certainly in the category of languages that can be recognized by a finite state machine. We call these languages *regular languages*.

There also exist non-regular languages. One very common non-regular language is "valid arithmetic expressions," or, to boil it down to its essence, "properly matched parentheses." Any program that can distinguish the properly matched string `(((())))` from the improperly matched strings `(((()))` and `(((()))))` must essentially be able to "count"--to distinguish the case of *four* parentheses from the cases of *three* or *five*. Counting in this way cannot be done without a modifiable variable or a push-down stack; so parenthesis-matching is *not* a regular language.

It turns out that, given any regular language, there is a nice straightforward way to write a representation of the finite state machine that recognizes it, which of course is also a representation of the rules of the language itself. We call this representation a *regular expression*, or *regex*. The standard notation for regexes was developed in the 1950s, and was really set in stone by the late 1970s in Unix programs such as `grep` and `sed`--programs which are still very much worth learning today, but which are of course outside the scope of this book.

The C++ standard library offers several different "flavors" of regex syntax, but the default flavor (and the one you should always use) was borrowed wholesale from the standard for ECMAScript--the language better known as JavaScript--with only minor modifications in the vicinity of square-bracket constructs. I've included a primer on ECMAScript regex syntax near the end of this chapter; but if you've ever used `grep`, you'll be able to follow the rest of this chapter easily without consulting that section.

# A note on backslash-escaping

In this chapter, we'll be referring frequently to strings and regular expressions that contain literal backslashes. As you know, to write a string containing a literal backslash in C++, you have to *escape* the backslash with another backslash: thus `"\n"` represents a newline character but `"\\n"` represents the two-character string of "backslash" followed by "n". This kind of thing is usually easy to keep track of, but in this chapter we're going to have to take special pains. Regexes are implemented purely as a library feature; so when you write `std::regex("\n")` the regex library will see a "regex" containing only a single whitespace character, and if you write `std::regex("\\n")` the library will see a two-character string starting with a backslash, which *the library will interpret* as a two-character escape sequence meaning "newline." If you want to communicate the idea of a *literal* backslash-n to the regex library, you'll have to get the regex library to see the three-character string `\\\\n`, which means writing the five-character string `"\\\\n"` in your C++ source code.

You might have noticed in the preceding paragraph the solution I'm going to be using in this chapter. When I talk about a *C++ string literal* or string value, I will put it in double quotes, like this: `"cat"`, `"a\\.b"`. When I talk about a *regular expression* as you would type it in an email or a text editor, or hand it to the library for evaluation, I will express it without quotes: `cat`, `a\.b`. Just remember that when you see an unquoted string, that's a literal sequence of characters, and if you want to put it into a C++ string literal, you'll need to double up all the backslashes, thus: `a\.b` goes into your source code as `std::regex("a\\.b")`.

I hear some of you asking: What about *raw string literals*? Raw string literals are a C++11 feature that allows you to write the character sequence `a\.b` by "escaping" the entire string with an `R` and some parentheses, like this--`R"(a\.b)"`--instead of escaping each backslash in the string. If your string contains parentheses itself, then you can get fancier by writing any arbitrary string before the first parenthesis and after the last, like this: `R"fancy(a\.b)fancy"`. A raw string literal like this one is allowed to contain any characters--backslashes, quotation marks, even newlines--as long as it doesn't contain the consecutive sequence `)fancy"` (and if you think there's a chance it might contain that sequence, then you just pick a new arbitrary string, such as `)supercalifragilisticexpialidocious"`).

The syntax of C++ raw string literals, with its leading `R`, is reminiscent of the raw string literal syntax in Python (with its leading `r`). In Python, `r"a\.b"` similarly represents the literal string `a\.b`; and it is both common and idiomatic to represent regular expressions in code by strings such as `r"abc"` even if they don't contain any special characters. But notice the all-important difference between `r"a\.b"` and `R"(a\.b)"`--the C++ version has an extra set of parentheses! And parentheses are *significant special characters* in the regex grammar. The C++ string literals `"(cat)"` and `R"(cat)"` are as different as night and day--the former represents the five-character regex `(cat)`, and the latter represents the three-character regex `cat`. If you trip up and write `R"(cat)"` when you meant `"(cat)"` (or equivalently, `R"((cat))"`), your program will have a very subtle bug. Even more sadistically, `R"a*(b*)a*"` is a valid regex with a surprising meaning! Therefore, I recommend that you use raw string literals for regexes with great caution; generally it is safer and clearer to double *all* your backslashes than to worry about doubling only the *outermost* of your parentheses.

Where raw string literals *are* useful is for what other languages call "heredocs":

[PRE0]

That is, raw string literals are the only kind of string literal in C++ that can encode newline characters without any kind of escaping. This is useful for printing long messages to the user, or maybe for things such as HTTP headers; but raw strings' behavior with parentheses makes them mildly dangerous for use with regular expressions--I will not be using them in this book.

# Reifying regular expressions into std::regex objects

To use regular expressions in C++, you can't use a string such as `"c[a-z]*t"` directly. Instead, you have to use that string to construct a *regular expression object* of type `std::regex`, and then pass the `regex` object as one of the arguments to a *matching function* such as `std::regex_match`, `std::regex_search`, or `std::regex_replace`. Each object of type `std::regex` encodes a complete finite state machine for the given expression, and constructing this finite state machine requires a lot of computation and memory allocation; so if we are going to match a lot of input text against the same regex, it is convenient that the library gives us a way to pay for that expensive construction just once. On the other hand, this means that the `std::regex` objects are relatively slow to construct and expensive to copy; constructing a regex inside a tight inner loop is a good way to kill your program's performance:

[PRE1]

Keep in mind that this `regex` object has value semantics; when we "match" an input string against a regex, we aren't mutating the `regex` object itself. A regex has no memory of what it's been matched against. Therefore, when we want to pull information out of a regex-matching operation--such as "did the command say to move left or right? what was the number we saw?"--we'll have to introduce a new entity that we can mutate.

A `regex` object offers the following methods:

`std::regex(str, flags)` constructs a new `std::regex` object by translating (or "compiling") the given `str` into a finite state machine. Options affecting the compilation process itself can be specified via the bitmask argument `flags`:

*   `std::regex::icase`: Treat all alphabetic characters as case-insensitive
*   `std::regex::nosubs`: Treat all parenthesized groups as non-capturing
*   `std::regex::multiline`: Make the non-consuming assertion `^` (and `$`) match immediately after (and before) a `"\n"` character in the input, rather than only at the beginning (and end) of the input

There are several other options that you could bitwise-OR into flags; but the others either change the "flavor" of regex syntax away from ECMAScript towards less well-documented and less well-tested flavors (`basic`, `extended`, `awk`, `grep`, `egrep`), introduce locale dependencies (`collate`), or simply don't do anything at all (`optimize`). Therefore, you should avoid all of them in production code.

Notice that even though the process of turning a string into a `regex` object is often called "compiling the regex," it is still a dynamic process that happens at runtime when the `regex` constructor is called, not during the compilation of your C++ program. If you make a syntax error in your regular expression, it will be caught not at compile time, but at runtime--the `regex` constructor will throw an exception of type `std::regex_error`, which is a subclass of `std::runtime_error`. Properly robust code should also be prepared for the `regex` constructor to throw `std::bad_alloc`; recall that `std::regex` is not allocator-aware.

`rx.mark_count()` returns the number of parenthesized capturing groups in the regex. The name of this method comes from the phrase "marked subexpression," an older synonym for "capturing group."

`rx.flags()` returns the bit-mask that was passed to the constructor originally.

# Matching and searching

To ask whether a given input string `haystack` conforms to a given regex `rneedle`, you can use `std::regex_match(haystack, rneedle)`. The regex always comes last, which is reminiscent of JavaScript's syntax `haystack.match(rneedle)` and Perl's `haystack =~ rneedle` even as it's opposed to Python's `re.match(rneedle, haystack)`. The `regex_match` function returns `true` if the regex matches the entire input string, and `false` otherwise:

[PRE2]

The `regex_search` function returns `true` if the regex matches any portion of the input string. Essentially, it just puts `.*` on both sides of the regex you provided and then runs the `regex_match` algorithm; but implementations can generally perform a `regex_search` faster than they could recompile a whole new regex.

To match within just part of a character buffer (such as you might do when pulling data in bulk over a network connection or from a file), you can pass an iterator pair to `regex_match` or `regex_search`, very similarly to what we saw in [Chapter 3](part0036.html#12AK80-2fdac365b8984feebddfbb9250eaf20d), *The Iterator-Pair Algorithms*. In the following example, bytes outside the range `[p, end)` are never considered, and the "string" `p` doesn't need to be null-terminated:

[PRE3]

This interface is similar to what we saw with `std::from_chars` in [Chapter 9](part0144.html#49AH00-2fdac365b8984feebddfbb9250eaf20d), *Iostreams*.

# Pulling submatches out of a match

To use regexes for the *lexing* stage of input, you'll need a way to pull out the input substrings that matched each capturing group. The way you do this in C++ is by creating a *match object* of type `std::smatch`. No, that's not a typo! The name of the match-object type really is `smatch`, which stands for `std::string` match; there is also `cmatch` for `const char *` match. The difference between `smatch` or `cmatch` is the *type of iterator* they store internally: `smatch` stores `string::const_iterator`, while `cmatch` stores `const char *`.

Having constructed an empty `std::smatch` object, you'll pass it by reference as the middle parameter to `regex_match` or `regex_search`. Those functions will "fill in" the `smatch` object with information about the substrings that matched, *if* the regex match actually succeeded. If the match failed, then the `smatch` object will become (or remain) empty.

Here's an example of using `std::smatch` to pull out the substrings matching the direction and the integer distance from our "robot command":

[PRE4]

Notice that we use a `static` regex object to avoid constructing ("compiling") a new regex object every time the function is entered. Here's the same code using `const char *` and `std::cmatch` just for comparison:

[PRE5]

In both cases, something interesting happens on the line with the `return`. Having successfully matched the input string against our regex, we can query the match object `m` to find out which pieces of the input string correspond to the individual capturing groups in our regex. The first capturing group (`(left|right)` in our example) corresponds to `m[1]`, the second group (`([0-9]+)` in our example) corresponds to `m[2]`, and so on. If you try to refer to a group that doesn't exist in the regex, such as `m[3]` in our example, you'll get an empty string; accessing a match object will never throw an exception.

The group `m[0]` is a special case: it refers to the entire matched sequence. If the match was filled in by `std::regex_match`, this will always be the entire input string; if the match was filled in by `std::regex_search`, then this will be just the part of the string that matched the regex.

There are also two named groups: `m.prefix()` and `m.suffix()`. These refer to the sequences that were *not* part of the match--before the matched substring and after it, respectively. It is an invariant that if the match succeeded at all, then `m.prefix() + m[0] + m.suffix()` represents the entire input string.

All of these "group" objects are represented not by `std::string` objects--that would be too expensive--but by lightweight objects of type `std::sub_match<It>` (where `It` is either `std::string::const_iterator` or `const char *` as noted previously). Every `sub_match` object is implicitly convertible to `std::string`, and otherwise behaves a lot like a `std::string_view`: you can compare submatches against string literals, ask them for their lengths, and even output them to a C++ stream with `operator<<`, without ever converting them to `std::string`. The downside of this lightweight efficiency is the same downside we get every time we deal with iterators pointing into a container we may not own: we run the risk of *dangling iterators*:

[PRE6]

[PRE7]

Fortunately, the standard library foresaw this lurking horror and evaded it by providing a special-case overload `regex_match(std::string&&, std::smatch&, const std::regex&)`, which is *explicitly deleted* (using the same `=delete` syntax you'd use to delete an unwanted special member function). This ensures that the preceding innocent-looking code will fail to compile, rather than being a source of iterator-invalidation bugs. Still, iterator invalidation bugs can happen, as in the previous example; to prevent them, you should treat `smatch` objects as extremely temporary, kind of like a `[&]` lambda that captures the world by reference. Once a `smatch` object has been filled in, don't touch anything else in the environment until you've extracted the parts of the `smatch` that you care about!

To summarize, a `smatch` or `cmatch` object offers the following methods:

*   `m.ready()`: True if `m` has been filled in at all, in the time since its construction.
*   `m.empty()`: True if `m` represents a failed match (that is, if it was most recently filled in by a failed `regex_match` or `regex_search`); false if `m` represents a successful match.
*   `m.prefix()`, `m[0]`, `m.suffix()`: `sub_match` objects representing the unmatched prefix, matched, and unmatched suffix parts of the input string. (If `m` represents a failed match, then none of these are meaningful.)
*   `m[k]`: A `sub_match` object representing the part of the input string matched by the *k*th capturing group. `m.str(k)` is a convenient shorthand for `m[k].str()`.
*   `m.size()`: Zero if `m` represents a failed match; otherwise, one more than the number of capturing groups in the regex whose successful match is represented by `m`. Notice that `m.size()` always agrees with `operator[]`; the range of meaningful submatch objects is always `m[0]` through `m[m.size()-1]`.
*   `m.begin()`, `m.end()`: Iterators enabling ranged for-loop syntax over a match object.

And a `sub_match` object offers the following methods:

*   `sm.first`: The iterator to the beginning of the matched input substring.
*   `sm.second`: The iterator to the end of the matched input substring.
*   `sm.matched`: True if `sm` was involved in the successful match; false if `sm` was part of an optional branch that got bypassed. For example, if the regex was `(a)|(b)` and the input was `"a"`, we would have `m[1].matched && !m[2].matched`; whereas if the input were `"b"`, we would have `m[2].matched && !m[1].matched`.
*   `sm.str()`: The matched input substring, pulled out and converted to `std::string`.
*   `sm.length()`: The length of the matched input substring (`second - first`). Equivalent to `sm.str().length()`, but much faster.
*   `sm == "foo"`: Comparison against `std::string`, `const char *`, or a single `char`. Equivalent to `sm.str() == "foo"`, but much faster. Unfortunately, the C++17 standard library does not provide any overload of `operator==` taking `std::string_view`.

Although you will likely never have a use for this in real code, it is possible to create a match or submatch object storing iterators into containers other than `std::string` or buffers of `char`. For example, here's our same function, but matching our regex against a `std::list<char>`--silly, but it works!

[PRE8]

# Converting submatches to data values

Just to close the loop on parsing, here's an example of how we could parse string and integer values out of our submatches to actually move our robot:

[PRE9]

Any unrecognized or invalid string input is diagnosed either by our custom`"Failed to lex"` exception or by the `std::out_of_range` exception thrown by `std::stoi()`. If we were to add a check for integer overflow before modifying `pos`, we'd have a rock-solid input parser.

If we wanted to handle negative integers and case-insensitive directions, the following modifications would do the trick:

[PRE10]

# Iterating over multiple matches

Consider the regex `(?!\d)\w+`, which matches a single C++ identifier. We already know how to use `std::regex_match` to tell whether an input string *is* a C++ identifier, and how to use `std::regex_search` to find the *first* C++ identifier in a given input line. But what if we want to find *all* the C++ identifiers in a given input line?

The fundamental idea here is to call `std::regex_search` in a loop. This gets complicated, though, because of the non-consuming "lookbehind" anchors such as `^` and `\b`. To implement a loop over `std::regex_search` correctly from scratch, we'd have to preserve the state of these anchors. `std::regex_search` (and `std::regex_match` for that matter) supports this use-case by providing flags of its own--flags which determine the *starting state* of the finite state machine for this particular matching operation. For our purposes, the only important flag is `std::regex::match_prev_avail`, which tells the library that the iterator `begin`, representing the start of the input, is not actually at the "beginning" of the input (that is, it might not match `^`) and that if you want to know the previous character of the input for purposes of `\b`, it is safe to inspect `begin[-1]`:

[PRE11]

In the preceding example, when `!be_correct`, each `regex_search` invocation is treated independently, so there is no difference between searching for `\bb.` from the first letter of the word `"by"` or from the third letter of the word `"baby"`. But when we pass `match_prev_avail` to the later invocations of `regex_search`, it takes a step back--literally--to see whether the letter before `"by"` was a "word" letter or not. Since the preceding `"a"` is a word letter, the second `regex_search` correctly refuses to treat `"by"` as a match.

Using `regex_search` in a loop like this is easy... unless the given regex might match an empty string! If the regex ever returns a successful match `m` where `m[0].length() == 0`, then we'll have an infinite loop. So the inner loop of our `get_all_matches()` should really look more like this:

[PRE12]

The standard library provides a "convenience" type called `std::regex_iterator` that will encapsulate the preceding code snippets' logic; using `regex_iterator` might conceivably save you some subtle bugs related to zero-length matches. Sadly, it won't save you any typing, and it slightly increases the chances of dangling-iterator pitfalls. `regex_iterator` is templated on its underlying iterator type in the same way as `match_results`, so if you're matching `std::string` input you want `std::sregex_iterator` and if you're matching on `const char *` input you want `std::cregex_iterator`. Here's the preceding example, recoded in terms of `sregex_iterator`:

[PRE13]

Consider how this awkward for-loop might benefit from a helper class, along the lines
of `streamer<T>` from the example near the end of [Chapter 9](part0144.html#49AH00-2fdac365b8984feebddfbb9250eaf20d), *Iostreams*.

You can also iterate over the submatches within each match, either manually or using a "convenience" library type. Manually, it would look something like this:

[PRE14]

Recall that `regex_iterator` is just a wrapper around `regex_search`, so `m.prefix()` in this case is guaranteed to hold an entire non-matching portion, all the way back to the end of the previous match. By alternately pushing back non-matching prefixes and matches, and finishing with a special case for the non-matching suffix, we split the input string into a vector of "words" alternating with "word separators." It's easy to modify this code to save only the "words" or only the "separators" if that's all you need; or even to save `m[1]` instead of `m[0]`, and so forth.

The library type `std::sregex_token_iterator` encapsulates all of this logic very directly, although its constructor interface is fairly confusing if you aren't already familiar with the preceding manual code. `sregex_token_iterator`'s constructor takes an input iterator-pair, a regex, and then a *vector of submatch indices*, where the index `-1` is a special case meaning "prefixes (and also, suffix)."

[PRE15]

If we change the array `{-1, 0}` to just `{0}`, then our resulting vector will contain
only the pieces of the input string matching `rx`. If we change it to `{1, 2, 3}`, our
loop will see only those submatches (`m[1]`, `m[2]`, and `m[3]`) in each match `m` of `rx`. Recall that because of the `|` operator, submatches can be bypassed, leaving `m[k].matched` false. `regex_token_iterator` does not skip those matches. For example:

[PRE16]

The most attractive use of `regex_token_iterator` might be to split a string into "words" at whitespace boundaries. Unfortunately it is not significantly easier to use--or to debug--than old-school approaches such as `istream_iterator<string>` (see [Chapter 9](part0144.html#49AH00-2fdac365b8984feebddfbb9250eaf20d), *Iostreams*) or `strtok_r`.

# Using regular expressions for string replacement

If you're coming from Perl, or if you often use the command-line utility `sed`, you probably think of regexes primarily as a way to *modify* strings--for example, "remove all substrings matching this regex," or "replace all instances of this word with another word." The C++ standard library does provide a sort of replace-by-regex functionality, under the name `std::regex_replace`. It's based on the JavaScript `String.prototype.replace` method, which means that it comes with its own idiosyncratic formatting mini-language.

`std::regex_replace(str, rx, "replacement")` returns a `std::string` constructed by searching through `str` for every substring matching the regex `rx` and replacing each such substring with the literal string `"replacement"`. For example:

[PRE17]

However, if `"replacement"` contains any `'$'` characters, special things will happen!

*   `"$&"` is replaced with the entire matching substring, `m[0]`. Both libstdc++ and libc++ support `"$0"` as a non-standard synonym for `"$&"`.
*   `"$1"` is replaced with the first submatch, `m[1]`; `"$2"` is replaced with `m[2]`; and so on, all the way up to `"$99"`. There is no way to refer to the 100th submatch. `"$100"` represents "`m[10]` followed by a literal `'0'`." To express "`m[1]` followed by a literal `'0'`," write `"$010"`.
*   ``"$`"`` (that's a backtick) is replaced with `m.prefix()`.
*   `"$'"` (that's a single-quote) is replaced with `m.suffix()`.
*   `"$$"` is replaced with a literal dollar sign.

Notice that ``"$`"`` and `"$'"` are far from symmetrical, because `m.prefix()` always refers to the part of the string between the end of the last match and the start of the current one, but `m.suffix()` always refers to the part of the string between the end of the current match and the *end of the string*! You'll never use either ``"$`"`` or `"$'"` in real code.

Here's an example of using `regex_replace` to remove all the instances of `std::` from a piece of code, or to change them all to `my::`:

[PRE18]

JavaScript's `String.prototype.replace` allows you to pass in an arbitrary function instead of a dollar-sign-studded format string. C++'s `regex_replace` doesn't support arbitrary functions yet, but it's easy to write your own version that does:

[PRE19]

With this improved `regex_replace` in hand, you can perform complicated operations such as "convert every identifier from `snake_case` to `CamelCase`" with ease.

This concludes our whirlwind tour of the facilities provided in C++'s `<regex>` header. The remainder of this chapter consists of a detailed introduction to the ECMAScript dialect of regex notation. I hope it will be useful to readers who haven't worked with regexes before, and that it will serve as a refresher and reference for those who have.

# A primer on the ECMAScript regex grammar

The rules for reading and writing regexes in the ECMAScript dialect are simple. A regex is just a string of characters (such as `a[bc].d*e`), and you read it from left to right. Most characters represent only themselves, so that `cat` is a valid regex and matches only the literal string `"cat"`. The only characters that don't represent themselves--and thus the only way to build regexes that represent languages more interesting than `"cat"`--are the following punctuation characters:

[PRE20]

`\`--if you're using a regex to describe a set of strings involving punctuation characters, you can use a backslash to escape those special characters. For example, `\$42\.00` is a regex for the singleton language whose only member is the string `"$42.00"`. Perhaps confusingly, backslash is *also* used to turn some normal characters into special characters! `n` is a regex for the letter "n", but `\n` is a regex for the newline character. `d` is a regex for the letter "d", but `\d` is a regex equivalent to `[0-9]`.

The complete list of backslash characters recognized by C++'s regex grammar is:

*   `\1`, `\2`, ... `\10`, ... for backreferences (to be avoided)
*   `\b` for a word boundary and `\B` for `(?!\b)`
*   `\d` for `[[:digit:]]` and `\D` for `[^[:digit:]]`
*   `\s` for `[[:space:]]` and `\S` for `[^[:space:]]`
*   `\w` for `[0-9A-Za-z_]` and `\W` for `[^0-9A-Za-z_]`
*   `\cX` for various "control characters" (to be avoided)
*   `\xXX` for hexadecimal, with the usual meaning
*   `\u00XX` for Unicode, with the usual meaning
*   `\0`, `\f`, `\n`, `\r`, `\t`, `\v` with their usual meanings

`.`--This special character represents "exactly one character," with almost no other requirements. For example, `a.c` is a valid regex and matches inputs such as `"aac"`, `"a!c"`, and `"a\0c"`. However, `.` will *never* match a newline or carriage-return character; and because C++ regexes work at the byte level, not the Unicode level, `.` will match any single byte (other than `'\\n'` and `'\\r'`) but will never match a sequence of multiple bytes even if they happen to make up a valid UTF-8 codepoint.

`[]`--A group of characters enclosed in square brackets represents "exactly one of this set," so that `c[aou]t` is a valid regex and matches the strings `"cat"`, `"cot"`, and `"cut"`. You can use square-bracket syntax to "escape" most characters; for example, `[$][.][*][+][?][(][)][[][{][}][|]` is a regex for the singleton language whose only member is the string `"$.*+?()[{}|"`. However, you cannot use brackets to escape `]`, `\`, or `^`.

`[^]`--A group of characters enclosed in square brackets with a leading `^` represents "exactly one, *not* of this set," so that `c[^aou]t` will match `"cbt"` or `"c^t"` but not `"cat"`. The ECMAScript dialect does not treat the trivial cases `[]` or `[^]` specially; `[]` means "exactly one character from the empty set" (which is to say, it never matches anything), and `[^]` means "exactly one character *not* from the empty set" (which is to say, it matches any single character--just like `.` but better, because it *will* match newline and carriage-return characters).

The `[]` syntax treats a couple more characters specially: If `-` appears inside square brackets anywhere except as the first or last character, it denotes a "range" with its left and right neighbors. So `ro[s-v]e` is a regex for the language whose members are the four strings `"rose"`, `"rote"`, `"roue"`, and `"rove"`. A few commonly useful ranges--the same ranges exposed via the `<ctype.h>` header--are built in using the syntax `[:foo:]` inside square brackets: `[[:digit:]]` is the same as `[0-9]`, `[[:upper:][:lower:]]` is the same as `[[:alpha:]]` is the same as `[A-Za-z]`, and so on.

There are also built-in syntaxes that look like `[[.x.]]` and `[[=x=]]`; they deal with locale-dependent comparisons and you will never have to use them. Merely be aware that if you ever need to include the character `[` inside a square-bracketed character class, it will be in your best interest to backslash-escape it: both `foo[=([;]` and `foo[(\[=;]` match the strings `"foo="`, `"foo("`, `"foo["`, and `"foo;"`, but `foo[([=;]` is an invalid regex and will throw an exception at runtime when you try to construct a `std::regex` object from it.

`+`--An expression or single character followed immediately by `+` matches the previous expression or character any positive number of times. For example, the regex `ba+` matches the strings `"ba"`, `"baa"`, `"baaa"`, and so on.

`*`--An expression or single character followed immediately by `*` matches the previous expression or character any number of times--even no times at all! So the regex `ba*` matches the strings `"ba"`, `"baa"`, and `"baaa"`, and also matches `"b"` alone.

`?`--An expression or single character followed immediately by `?` matches the previous expression or character exactly zero or one times. For example, `coo?t` is a regex matching only `"cot"` and `"coot"`.

`{n}`--An expression or single character followed immediately by a curly-braced integer matches the previous expression or character exactly the number of times indicated. For example, `b(an){2}a` is a regex matching `"banana"`; `b(an){3}a` is a regex matching `"bananana"`.

`{m,n}`--When the curly-braced construct has the form of two integers *m* and *n* separated by a comma, the construct matches the previous expression or character anywhere from *m* to *n* times (inclusive). So `b(an){2,3}a` is a regex matching only the strings `"banana"` and `"bananana"`.

`{m,}`--Leaving *n* blank effectively makes it infinite; so `x{42,}` means "match `x` 42 or more times," and is equivalent to `x{42}x*`. The ECMAScript dialect does not allow leaving *m* blank.

`|`--Two regular expressions can be "glued together" with `|` to express the idea of *either-or*. For example, `cat|dog` is a regex matching only the strings `"cat"` and `"dog"`; and `(tor|shark)nado` matches either `"tornado"` or `"sharknado"`. The `|` operator has very low precedence in regexes, just as it does in C++ expressions.

`()`--Parentheses work just as in mathematics, to enclose a sub-expression that you want to bind tightly together and treat as a unit. For example, `ba*` means "the character `b`, and then zero or more instances of `a`; but `(ba)*` means "zero or more instances of `ba`." So the former matches `"b"`, `"ba"`, `"baa"`, and so on; but the version with parentheses matches `""`, `"ba"`, `"baba"`, and so on.

Parentheses also have a second purpose--they are used not just for *grouping* but also for *capturing* parts of a match for further processing. Each opening `(` in the regex generates another submatch in the resulting `std::smatch` object.

If you want to group some subexpression tightly together without generating a submatch, you can use a *non-capturing* group with the syntax `(?:foo)`:

[PRE21]

Non-capturing might be useful in some obscure context; but generally, it will be clearer to the reader if you just use regular capturing `()` and ignore the submatches you don't care about, as opposed to scattering `(?:)` around your codebase in an attempt to squelch all unused submatches. Unused submatches are very cheap, performance-wise.

# Non-consuming constructs

`(?=foo)` matches the pattern `foo` against the input, and then "rewinds" so that none of the input is actually consumed. This is called "lookahead." So for example `c(?=a)(?=a)(?=a)at` matches `"cat"`; and `(?=.*[A-Za-z])(?=.*[0-9]).*` matches any string containing at least one alphabetic character and at least one digit.

`(?!foo)` is a "negative lookahead"; it looks ahead to match `foo` against the input, but then *rejects* the match if `foo` would have accepted, and *accepts* the match if `foo` would have rejected. So, for example, `(?!\d)\w+` matches any C++ identifier or keyword--that is, any sequence of alphanumeric characters that does *not* start with a digit. Notice that the first character must not match `\d` but is not consumed by the `(?!\d)` construct; it must still be accepted by `\w`. The similar-looking regex `[^0-9]\w+` would "erroneously" accept strings such as `"#xyzzy"` which are not valid identifiers.

Both `(?=)` and `(?!)` are not only non-consuming but also *non-capturing*, just like `(?:)`. But it is perfectly fine to write `(?=(foo))` to capture all or part of the "looked-ahead" portion.

`^` and `$`--A caret `^` on its own, outside any square brackets, matches only at the beginning of the string to be matched; and `$` matches only at the end. This is useful to "anchor" the regex to the beginning or end of the input string, in the context of `std::regex_search`. In `std::regex::multiline` regexes, `^` and `$` act as "lookbehind" and "lookahead" assertions respectively:

[PRE22]

Putting it all together, we might write the regex `foo[a-z_]+(\d|$)` to match "the letters `foo`, followed by one or more other letters and/or underscore; followed by either a digit or the end of the line."

If you need a deeper dive into regex syntax, consult [cppreference.com](https://cppreference.com). And if that's not enough--the best thing about C++'s copying the ECMAScript flavor of regexes is that any tutorial on JavaScript regexes will also be applicable to C++! You can even test out regular expressions in your browser's console. The only difference between C++ regexes and JavaScript regexes is that C++ supports the double-square-bracket syntax for character classes such as `[[:digit:]]`, `[[.x.]]`, and `[[=x=]]`, whereas JavaScript doesn't. JavaScript treats those regexes as equivalent to `[\[:digt]\]`, `[\[.x]\]`, and `[\[=x]\]` respectively.

# Obscure ECMAScript features and pitfalls

Earlier in this chapter I mentioned a few features of `std::regex` that you would be better off to avoid, such as `std::regex::collate`, `std::regex::optimize`, and flags that change the dialect away from ECMAScript. The ECMAScript regex grammar itself contains a few obscure and avoid worthy features as well.

A backslash followed by one or more digits (other than `\0`) creates a *backreference*. The backreference `\1` matches "the same sequence of characters that was matched by my first capturing group"; so for example the regex `(cat|dog)\1` will match the strings `"catcat"` and `"dogdog"` but not `"catdog"`, and `(a*)(b*)c\2\1` will match `"aabbbcbbbaa"` but not `"aabbbcbbba"`. Backreferences can have subtly weird semantics, especially when combined with non-consuming constructs such as `(?=foo)`, and I recommend avoiding them when possible.

If you're having trouble with backreferences, the first thing to check is your backslash-escaping. Remember that `std::regex("\1")` is a regex matching ASCII control character number 1\. What you meant to type was `std::regex("\\1")`.

Using backreferences takes you out of the world of *regular languages* and into the wider world of *context-sensitive languages*, which means that the library must trade in its extremely efficient finite-state-machine-based matching algorithm for more powerful but expensive and slow "backtracking" algorithms. This seems like another good reason to avoid backreferences unless they're absolutely necessary.

However, as of 2017, most vendors do not actually switch algorithms based on the *presence* of backreferences in a regex; they'll use the slower backtracking algorithm based on the *mere possibility* of backreferences in the ECMAScript regex dialect. And then, because no vendor wants to implement a whole second algorithm just for the backreference-less dialects `std::regex::awk` and `std::regex::extended`, they end up using the backtracking algorithm even for those dialects! Similarly, most vendors will implement `regex_match(s, rx)` in terms of `regex_match(s, m, rx)` and then throw out the expensively computed `m`, rather than using a potentially faster algorithm for `regex_match(s, rx)`. Optimizations like this might come to a library near you sometime in the next 10 years, but I wouldn't hold your breath waiting for them.

Another obscure quirk is that the `*`, `+`, and `?` quantifiers are all *greedy* by default, meaning that, for example, `(a*)` will prefer to match as many `a` characters as it can. You can turn a greedy quantifier *non-greedy* by suffixing an extra `?`; so for example `(a*?)` matches the *smallest* number of `a` characters it can. This makes no difference at all unless you're using capturing groups. Here's an example:

[PRE23]

In the first case, `.*` greedily matches `abc`, leaving only `d` to be matched by the capturing group. In the second case, `.*?` non-greedily matches only `a`, leaving `bcd` for the capturing group. (In fact, `.*?` would have preferred to match the empty string; but it couldn't do that without the overall match being rejected.)

Notice that the syntax for non-greediness doesn't follow the "normal" rules of operator composition. From what we know of C++'s operator syntax, we'd expect that `a+*` would mean `(a+)*` (which it does) and `a+?` would mean `(a+)?` (which it doesn't). So, if you see consecutive punctuation characters in a regular expression, watch out--it may mean something different from what your intuition tells you!

# Summary

Regular expressions (regexes) are a good way to *lex* out the pieces of an input string before parsing them. The default regex dialect in C++ is the same as in JavaScript. Use this to your advantage.

Prefer to avoid raw string literals in situations where an extra pair of parentheses could be confusing. When possible, limit the number of escaped backslashes in your regexes by using square brackets to escape special characters instead.

`std::regex rx` is basically immutable and represents a finite state machine. `std::smatch m` is mutable and holds information about a particular match within the haystack string. Submatch `m[0]` represents the whole matched substring; `m[k]` represents the *k*th capturing group.

`std::regex_match(s, m, rx)` matches the needle against the *entire* haystack string; `std::regex_search(s, m, rx)` looks for the needle *in* the haystack. Remember that the haystack goes first and the needle goes last, just like in JavaScript and Perl.

`std::regex_iterator`, `std::regex_token_iterator`, and `std::regex_replace` are relatively inconvenient "convenience" functions built on top of `regex_search`. Get comfortable with `regex_search` before worrying about these wrappers.

Beware of dangling-iterator bugs! Never modify or destroy a `regex` that is still referenced by `regex_iterator`; and never modify or destroy a `string` that is still referenced by `smatch`.