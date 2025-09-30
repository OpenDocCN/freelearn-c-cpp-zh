# Filesystem

One of the biggest new features of C++17 is its `<filesystem>` library. This library, like many other major features of modern C++, originated in the Boost project. In 2015, it went into a standard technical specification to gather feedback, and finally, was merged into the C++17 standard with some changes based on that feedback.

In this chapter, you'll learn the following:

*   How `<filesystem>` returns dynamically typed errors without throwing exceptions, and how you can too
*   The format of a *path,* and the fundamentally incompatible positions of POSIX and Windows on the subject
*   How to stat files and walk directories using portable C++17
*   How to create, copy, rename, and remove files and directories
*   How to fetch the free space of a filesystem

# A note about namespaces

The standard C++17 filesystem facilities are all provided in a single header, `<filesystem>`, and everything in that header is placed in its own namespace: `namespace std::filesystem`. This follows the precedent set by C++11's `<chrono>` header with its `namespace std::chrono`. (This book omits a full treatment of `<chrono>`. Its interactions with `std::thread` and `std::timed_mutex` are covered briefly in [Chapter 7](part0108.html#36VSO0-2fdac365b8984feebddfbb9250eaf20d), *Concurrency*.)

This namespacing strategy means that when you use the `<filesystem>` facilities, you'll be using identifiers such as `std::filesystem::directory_iterator` and `std::filesystem::temp_directory_path()`. These fully qualified names are quite unwieldy! But pulling the entire namespace into your current context with a `using` declaration is probably an overkill, especially, if you have to do it at file scope. We've all been taught over the past decade never to write `using namespace std`, and that advice won't change, no matter how deeply the standard library nests its namespaces. Consider the following code:

[PRE0]

A better solution for everyday purposes is to define a *namespace alias* at file scope (in a `.cc` file) or namespace scope (in a `.h` file). A namespace alias allows you to refer to an existing namespace by a new name, as seen in the following example:

[PRE1]

In the remainder of this chapter, I will be using the namespace alias `fs` to refer to `namespace std::filesystem`. When I say `fs::path`, I mean `std::filesystem::path`. When I say `fs::remove`, I mean `std::filesystem::remove`.

Defining a namespace alias `fs` somewhere global has another pragmatic benefit as well. At press time, of all the major library vendors, only Microsoft Visual Studio claims to have implemented the C++17 `<filesystem>` header. However, the facilities of `<filesystem>` are very similar to those provided by libstdc++ and libc++ in `<experimental/filesystem>`, and by Boost in `<boost/filesystem.hpp>`. So, if you consistently refer to these facilities by a custom namespace alias, such as `fs`, you'll be able to switch from one vendor's implementation to another just by changing the target of that alias--a one-line change, as opposed to a massive and error-prone search-and-replace operation on your entire codebase. This can be seen in the following example:

[PRE2]

# A very long note on error-reporting

C++ has a love-hate relationship with error-reporting. By "error-reporting" in this context, I mean "what to do, when you can't do what you were asked". The classical, typical, and still the best-practice way to report this kind of "disappointment" in C++ is to throw an exception. We have seen in the previous chapters that, sometimes, throwing an exception is the *only* sensible thing to do, because there is no way to return to your caller. For example, if your task was to construct an object, and construction fails, you cannot return; when a constructor fails, the only same course of action is to throw. However, we have *also* seen (in [Chapter 9](part0144.html#49AH00-2fdac365b8984feebddfbb9250eaf20d)*, Iostreams*) that C++'s own `<iostream>` library does not take this sane course of action! If the construction of a `std::fstream` object fails (because the named file cannot be opened), you will get an exception; you'll get a fully constructed `fstream` object where `f.fail() && !f.is_open()`.

The reason we gave in [Chapter 9](https://cdp.packtpub.com/mastering_c___stl/wp-admin/post.php?post=64&action=edit#post_58)*, Iostreams*, for the "bad" behavior of `fstream` was the *relatively high likelihood* that the named file will not be openable. Throwing an exception every time a file can't be opened is uncomfortably close to using exceptions for control flow, which we have been taught--properly--to avoid. So, rather than force the programmer to write `try` and `catch` blocks everywhere, the library returns as if the operation had succeeded, but allows the user to check (with a normal `if`, not a `catch`) whether the operation really did succeed or not.

That is, we can avoid writing this cumbersome code:

[PRE3]

Instead, we can simply write this:

[PRE4]

The iostreams approach works pretty well when the result of the operation is described by a heavyweight object (such as an `fstream`) which has a natural *failed* state, or where such a *failed* state can be added during the design stage. However, it has some downsides as well, and it flatly cannot be used if there is no heavyweight type involved. We saw this scenario at the end of [Chapter 9](part0144.html#49AH00-2fdac365b8984feebddfbb9250eaf20d), *Iostreams*, when we looked at ways of parsing integers from strings. If we don't expect failure, or don't mind the performance hit of "using exceptions for control flow," then we use `std::stoi`:

[PRE5]

If we need portability to C++03, we use `strtol`, which reports errors via the thread-local global variable `errno`, as seen in this code:

[PRE6]

And in bleeding-edge C++17 style, we use `std::from_chars`, which returns a lightweight struct containing the end-of-string pointer and a value of the strong enum type `std::errc` indicating success or failure, as follows:

[PRE7]

The `<filesystem>` library needs approximately the same capacity for error-reporting as `std::from_chars`. Pretty much any operation you can perform on your filesystem might fail due to the actions of other processes running on the system; so, throwing an exception on every failure (á là `std::stoi`) seems uncomfortably close to using exceptions for control flow. But threading an "error result" like `ec` through your entire codebase can also be tedious and (no pun intended) error-prone. So, the standard library decided to have its cake and eat it too by providing *two interfaces* to almost every function in the `<filesystem>` header!

For example, the following are the two `<filesystem>` functions for determining the size of a file on disk:

[PRE8]

Both the preceding functions take an `fs::path` (which we'll discuss more further in the chapter), and return a `uintmax_t` telling the size of the named file in bytes. But what if the file doesn't exist, or it exists, but the current user-account doesn't have permission to query its size? Then, the first overload will simply *throw an exception* of type `fs::filesystem_error`, indicating what went wrong. But the second overload will never throw (in fact, it's marked `noexcept`). Instead, it takes an out-parameter of type `std::error_code`, which the library will fill in with an indication of what went wrong (or clear, if nothing went wrong at all).

Comparing the signatures of `fs::file_size` and `std::from_chars`, you might notice that `from_chars` deals in `std::errc`, and `file_size` deals in `std::error_code`. These two types, while related, are not the same! To understand the difference--and the entire design of the non-throwing `<filesystem>` API--we'll have to take a quick detour into another part of the C++11 standard library.

# Using <system_error>

The difference between the error-reporting mechanisms of `std::from_chars` and `fs::file_size` is a difference in their intrinsic complexity. `from_chars` can fail in exactly two ways-- either the given string had no initial string of digits at all, else there were so *many* digits that it would cause an overflow to read them all. In the former case, a classic (but inefficient and, generally, dangerous) way to report the error would be to set `errno` to `EINVAL` (and return some useless value such as `0`). In the latter case, a classic approach would be to set `errno` to `ERANGE` (and return some useless value). This is more or less (but rather less than more) the approach taken by `strtol`.

The salient point is that with `from_chars`, there are exactly two things that can possibly *ever* go wrong, and they are completely describable by the single set of error codes provided by POSIX `<errno.h>`. So, in order to bring the 1980's `strtol` into the twenty-first century, all we need to fix is to make it return its error code directly to the caller rather than indirectly, via the thread-local `errno`. And so, that's all the standard library did. The classic POSIX `<errno.h>` values are still provided as macros via `<cerrno>`, but as of C++11, they're also provided via a strongly typed enumeration in `<system_error>`, as shown in the following code:

[PRE9]

`std::from_chars` reports errors by returning a struct (`struct from_chars_result`) containing a member variable of type `enum std::errc`, which will be either `0` for *no error*, or one of the two possible error-indicating values.

Now, what about `fs::file_size`? The set of possible errors encountered by `file_size` is much much larger--in fact, when you think of the number of operating systems in existence, and the number of different filesystems supported by each, and the fact that some filesystems (such as NFS) are distributed over *networks* of various types, the set of possible errors seems an awful lot like an *open set*. It might be possible to boil them all down onto the seventy-eight standard `sys::errc` enumerators (one for each POSIX `errno` value except `EDQUOT`, `EMULTIHOP`, and `ESTALE`), but that would lose a lot of information. Heck, at least one of the missing POSIX enumerators (`ESTALE`) is a legitimate failure mode of `fs::file_size`! And, of course, your underlying filesystem might want to report its own filesystem-specific errors; for example, while there is a standard POSIX error code for *name too long*, there is no POSIX error code for *name contains disallowed character* (for reasons we'll see in the next major section of this chapter). A filesystem might want to report exactly that error without worrying that `fs::file_size` was going to squash it down onto some fixed enumeration type.

The essential issue here is that the errors reported by `fs::file_size` might not all come from the same *domain*, and therefore, they cannot be represented by a single fixed-in-stone *type* (for example, `std::errc`). C++ exception-handling solves this problem elegantly; it is fine and natural for different levels of the program to throw different types of exceptions. If the lowest level of a program throws `myfs::DisallowedCharacterInName`, the topmost level can catch it--either by name, by base class, or by `...`. If we follow the general rule that everything thrown in a program should derive from `std::exception`, then any `catch` block will be able to use `e.what()` so that at least the user gets some vaguely human-readable indication of the problem, no matter what the problem was.

The standard library *reifies* the idea of multiple error domains into the base class `std::error_category`, as seen in the following code:

[PRE10]

`error_category` behaves a lot like `memory_resource` from [Chapter 8](part0129.html#3R0OI0-2fdac365b8984feebddfbb9250eaf20d), *Allocators*; it defines a classically polymorphic interface, and certain kinds of libraries are expected to subclass it. With `memory_resource`, we saw that some subclasses are global singletons, and some aren't. With `error_category`, *each* subclass *must* be a global singleton, or it's not going to work.

To make memory resources useful, the library gives us *containers* (see [Chapter 4](part0052.html#1HIT80-2fdac365b8984feebddfbb9250eaf20d), *The Container Zoo*). At the most basic level, a container is a pointer representing some allocated memory, plus a handle to the *memory resource* that knows how to deallocate that pointer. (Recall that a handle to a memory resource is called an *allocator*.)

To make the `error_category` subclasses useful, the library gives us `std::error_code`. At the most basic level (which is the *only* level, in this case), an `error_code` is an `int` representing an error enumerator plus a handle to the `error_category` that knows how to interpret that enumerator. It looks like this:

[PRE11]

So, to create a finicky filesystem library subsystem, we could write the following:

[PRE12]

This preceding code defines a new error domain, the `FinickyFS::Error` domain, reified as `FinickyFS::ErrorCategory::instance()`. It allows us to create objects of type `std::error_code` via expressions such as `make_error_code(FinickyFS::Error::forbidden_word)`.

Notice that **argument-dependent lookup** (**ADL**) will find the correct overload of `make_error_code` without any help from us. `make_error_code` is a customization point in exactly the same way as `swap`: just define a function with that name in your enum's namespace, and it will work without any additional effort.

[PRE13]

We now have a way to pass `FinickyFS::Error` codes losslessly through the system--by wrapping them inside trivially copyable `std::error_code` objects, and getting the original error back out at the topmost level. When I put it that way, it sounds almost like magic--like exception handling without exceptions! But as we've just seen, it's very simple to implement.

# Error codes and error conditions

Notice that `FinickyFS::Error` is not implicitly convertible to `std::error_code`; in the last example, we used the syntax `make_error_code(FinickyFS::Error::forbidden_word)` to construct our initial `error_code` object. We can make `FinickyFS::Error` more convenient for the programmer if we tell `<system_error>` to enable implicit conversions from `FinickyFS::Error` to `std::error_code`, as follows:

[PRE14]

Be careful when reopening namespace `std`--remember that you must be outside any other namespace when you do it! Otherwise, you'll be creating a nested namespace such as namespace `FinickyFS::std`. In this particular case, if you get it wrong, the compiler will helpfully error out when you try to specialize the non-existent `FinickyFS::std::is_error_code_enum`. As long as you only ever reopen namespace `std` in order to specialize templates (and as long as you don't mess up the template-specialization syntax), you won't have to worry too much about anything *quietly* failing.

Once you've specialized `std::is_error_code_enum` for your enum type, the library takes care of the rest, as seen in this code:

[PRE15]

The implicit conversion seen in the previous code enables convenient syntax such as direct comparisons via `==`, but because each `std::error_code` object carries its domain along with it, comparisons are strongly typed. Value-equality for the `error_code` objects depends not only on their *integer* *value*, but also the *address* of their associated error-category singletons.

[PRE16]

Specializing `is_error_code_enum<X>` is helpful if you're often going to be assigning `X` to variables of type `std::error_code`, or returning it from functions that return `std::error_code`. In other words, it's useful if your type `X` really does represent *the source of an error*--the throwing side of the equation, so to speak. But what about the catching side? Suppose you notice that you've written this function, and several more like it:

[PRE17]

The preceding function defines a *unary* *predicate* over the entire universe of error codes; it returns `true` for any error code associated with the concept of malformed names as far as our `FinickyFS` library is concerned. We can just drop this function straight into our library as `FinickyFS::is_malformed_name()`--and, in fact, that's the approach I personally recommend--but the standard library also provides another possible approach. You can define not an `error_code`, but an `error_condition`, as follows:

[PRE18]

Once you've done this, you can get the effect of calling `FinickyFS::is_malformed_name(ec)` by writing the comparison `(ec == FinickyFS::Condition::malformed_name)`, like this:

[PRE19]

However, because we did not provide a function `make_error_code(FinickyFS::Condition)`, there will be no easy way to construct a `std::error_code`} object holding one of these conditions. This is appropriate; condition enums are for testing against on the catching side, not for converting to `error_code` on the throwing side.

The standard library provides two code enum types (`std::future_errc` and `std::io_errc`), and one condition enum type (`std::errc`). That's right--the POSIX error enum `std::errc` actually enumerates *conditions*, not *codes*! This means that if you're trying to stuff POSIX error codes into a `std::error_code` object, you're doing it wrong; they are *conditions*, which means they're for *testing* *against* on the catching side, not for throwing. Sadly, the standard library gets this wrong in at least two ways. First, as we've seen, `std::from_chars` does throw a value of type `std::errc` (which is doubly inconvenient; it would be more consistent to throw a `std::error_code`). Second, the function `std::make_error_code(std::errc)` exists, cluttering up the semantic space, when really only `std::make_error_condition(std::errc)` should (and does) exist.

# Throwing errors with std::system_error

So far, we've considered `std::error_code`, a nifty non-throwing alternative to C++ exception-handling. But sometimes, you need to mix non-throwing and throwing libraries at different levels of the system. The standard library has your back--for one-half of the problem, anyway. `std::system_error` is a concrete exception type derived from `std::runtime_error`, which has just enough storage for a single `error_code`. So, if you are writing a library API which is throw-based, not `error_code`-based, and you receive an `error_code` indicating failure from a lower level of the system, it is perfectly appropriate to wrap that `error_code` in a `system_error` object, and `throw` it upward.

[PRE20]

In the opposite case--where you've written your library API to be non-throwing, but you make calls into lower levels that might throw--the standard library provides, basically, no help. But you can write an `error_code` unwrapper fairly easily yourself:

[PRE21]

[PRE22]

This concludes our digression into the confusing world of `<system_error>`. We now return you to your regularly scheduled `<filesystem>`, already in progress.

# Filesystems and paths

In [Chapter 9](part0144.html#49AH00-2fdac365b8984feebddfbb9250eaf20d), *Iostreams*, we discussed the POSIX concept of file descriptors. A file descriptor
represents a source or sink of data which can be targeted by `read` and/or `write`; often, but not always, it corresponds to a file on disk. (Recall that file descriptor number `1` refers to `stdout`, which is usually connected to the human user's screen. File descriptors can also refer to network sockets, devices such as `/dev/random`, and so on.)

Furthermore, POSIX file descriptors, `<stdio.h>`, and `<iostream>` are all concerned, specifically, with the *contents* of a file on disk (or wherever)--the sequence of bytes that makes up the *contents* of the file. A file in the *file**system* sense has many more salient attributes that are not exposed by the file-reading-and-writing APIs. We cannot use the APIs of [Chapter 9](part0144.html#49AH00-2fdac365b8984feebddfbb9250eaf20d), *Iostreams*, to determine the ownership of a file, or its last-modified date; nor can we determine the number of files in a given directory. The purpose of `<filesystem>` is to allow our C++ programs to interact with these *filesystem* attributes in a portable, cross-platform way.

Let's begin again. What is a filesystem? A filesystem is an abstract mapping from *paths* to *files*, by means of *directory* *entries*. Perhaps a diagram will help, if you take it with a large grain of salt:

![](img/00025.jpeg)

At the top of the preceding diagram, we have the somewhat abstract world of "names." We have a mapping from those names (such as `speech.txt`) onto concrete structures that POSIX calls *inodes*. The term "inode" is not used by the C++ standard--it uses the generic term "file"--but I will try to use the term inode when I want to be precise. Each inode contains a full set of attributes describing a single file on disk: its owner, its date of last modification, its *type*, and so on. Most importantly, the inode also tells exactly how big the file is, and gives a pointer to its actual contents (similarly to how a `std::vector` or `std::list` holds a pointer to *its* contents). The exact representation of inodes and blocks on disk depends on what kind of filesystem you're running; names of some common filesystems include ext4 (common on Linux), HFS+ (on OS X), and NTFS (on Windows).

Notice that a few of the blocks in that diagram hold data that is just a tabular mapping of *names* to *inode numbers*. This brings us full circle! A *directory* is just an inode with a certain *type*, whose contents are a tabular mapping of names to inode numbers. Each filesystem has one special well-known inode called its *root directory*.

Suppose that the inode labeled "`2`" in our diagram is the *root directory*. Then we can unambiguously identify the file containing "Now is the time..." by a path of names that leads from the root directory down to that file. For example, `/My Documents/speech.txt` is such a path: starting from the root directory, `My Documents` maps to inode 42, which is a directory where `speech.txt` maps to inode 17, which is a normal file whose contents on disk are "Now is the time...". We use slashes to compose these individual names into a single path, and we put a single slash on the front to indicate that we're starting from the root directory. (In Windows, each partition or drive has a separate root directory. So, instead of writing just `/My Documents/speech.txt`, we might write `c:/My Documents/speech.txt` to indicate that we're starting from drive C's root directory.)

Alternatively, "`/alices-speech.txt`" is a path leading straight from the root directory to inode 17\. We say that these two paths ("`/My Documents/speech.txt`" and "`/alices-speech.txt`") are both *hard*-*links* for the same underlying inode, which is to say, the same underlying *file*. Some filesystems (such as the FAT filesystem used by many USB sticks) do not support having multiple hard links to the same file. When multiple hard links *are* supported, the filesystem must count the number of references to each inode so that it knows when it's safe to delete and free up an inode--in a procedure exactly analogous to the `shared_ptr` reference-counting we saw in [Chapter 6](part0093.html#2OM4A0-2fdac365b8984feebddfbb9250eaf20d), *Smart Pointers*.

When we ask a library function such as `open` or `fopen` to "open a file," this is the process it's going through deep down in the innards of the filesystem. It takes the filename you gave it and treats it as a *path*--splits it up at the slashes, and descends into the directory structure of the filesystem until it finally reaches the inode of the file you asked for (or until it hits a dead end). Notice that once we have reached the inode, there is no longer any sense in asking "What is the name of this file?", as it has at least as many names as there are hard-links to it.

# Representing paths in C++

Throughout [Chapter 9](part0144.html#49AH00-2fdac365b8984feebddfbb9250eaf20d), *Iostreams*, every function that expected a "filename" (that is, a path) as a parameter was happy to take that path as a simple const char `*`. But in the `<filesystem>` library, we're going to complicate that picture, all because of Windows.

All POSIX filesystems store names (like `speech.txt`) as simple raw byte strings. The only rules in POSIX are that your names can't contain `'\0'`, and your names can't contain `'/'` (because that's the character we're going to split on). On POSIX, `"\xC1.h"` is a perfectly valid filename, despite the fact that it is *not* valid UTF-8 and *not* valid ASCII, and the way it'll display on your screen when you `ls .` is completely dependent on your current locale and codepage. After all, it's just a string of three bytes, none of which are `'/'`.

On the other hand, Window's native file APIs, such as `CreateFileW`, always store names as UTF-16\. This means that, by definition, paths in Windows are always valid Unicode strings. This is a major philosophical difference between POSIX and NTFS! Let me say it again, slowly: In POSIX, file names are *strings of bytes*. In Windows, file names are *strings of Unicode characters*.

If you follow the general principle from [Chapter 9](part0144.html#49AH00-2fdac365b8984feebddfbb9250eaf20d), *Iostreams* that everything in the world should be encoded with UTF-8, then the difference between POSIX and Windows will be manageable--maybe, even negligible. But if you are ever required to debug problems with strangely named files on one or the other system, keep in mind: In POSIX, filenames are strings of bytes. In Windows, filenames are strings of characters.

Since Windows APIs expect UTF-16 strings (`std::u16string`) and POSIX APIs expect byte strings (`std::string`), neither representation is exactly appropriate for a cross-platform library. So, `<filesystem>` invents a new type: `fs::path`. (Recall that we're using our namespace alias throughout this chapter. That's `std::filesystem::path` in reality.) `fs::path` looks something like this:

[PRE23]

Notice that `fs::path::value_type` is `wchar_t` in Windows, even though C++11's UTF-16 character type `char16_t` would be more appropriate. This is just an artifact of the library's historical roots in Boost, which dates back to before C++11\. In this chapter, whenever we talk about `wchar_t`, you can assume we're talking about UTF-16, and vice versa.

To write portable code, pay attention to the return type of any function you use to convert an `fs::path` to a string. For example, notice that the return type of `path.c_str()` is not the const char `*`--it's const value_type `*`!

[PRE24]

The preceding example, case `c`, is guaranteed to compile, but its behavior differs on the two platforms: in POSIX platforms, it'll give you the raw byte-string you want, and in Windows, it'll expensively convert `path.native()` from UTF-16 to UTF-8 (which is exactly what you asked for--but your program might be faster if you found a way to avoid asking).

`fs::path` has a templated constructor that can construct a `path` from just about any argument. The argument can be a sequence of any character type (`char`, `wchar_t`, `char16_t`, or `char32_t`), and that sequence can be expressed as a pointer to a null-terminated string, an *iterator* to a null-terminated string, a `basic_string`, a `basic_string_view`, or an iterator-pair. As usual, I mention this huge variety of overloads not because you'll want to use any of them beyond the basics, but so that you'll know how to avoid them.

The standard also provides the free function `fs::u8path("path")`, which is just a synonym for `fs::path("path")`, but might serve as a reminder that the string you're passing in is supposed to be UTF-8-encoded. I recommend ignoring `u8path`.

This all might sound scarier than it is. Bear in mind that if you stick to ASCII filenames, you won't need to worry about encoding issues; and if you remember to avoid the "native" accessor methods, `path.native()` and `path.c_str()`, and avoid the implicit conversion to `fs::path::string_type`, then you won't have to worry too much about portability.

# Operations on paths

[PRE25]

[PRE26]

So, for example, given the path `p = "c:/foo/hello.txt"`, we have `p.root_name() == "c:"`, `p.root_directory() == "/"`, `p.relative_path() == "foo/hello.txt"`, `p.stem() == "hello"`, and `p.extension() == ".txt"`. At least, that's what we'd have in Windows! Notice that in Windows, an absolute path requires both a root name and a root directory (neither "`c:foo/hello.txt`" nor "`/foo/hello.txt`" is an absolute path), whereas, in POSIX, where root names don't exist, an absolute path requires only a root directory ("`/foo/hello.txt`" is an absolute path, and "`c:foo/hello.txt`" is a relative path that starts with the funny-looking directory name "`c:foo`").

[PRE27]

[PRE28]

Paths also support concatenation with and without slashes under the confusing member-function names `path.concat("foo")` (without slash) and `path.append("foo")`(with slash). Beware that this is exactly backwards from what you'd expect! Therefore, I strongly advise never to use the named member functions; always use the operators (perhaps including your custom-defined `operator+` as described in the preceding code).

The last potentially confusing thing about `fs::path` is that it provides `begin` and `end` methods, just like `std::string`. But unlike `std::string`, the unit of iteration is not the single character--the unit of iteration is the *name*! This is seen in the following example:

[PRE29]

You'll never have a reason to iterate over an absolute `fs::path` in real code. Iterating over `p.relative_path().parent_path()`--where every iterated element is guaranteed to be a directory name--might have some value in unusual circumstances.

# Statting files with directory_entry

Beware! `directory_entry` is the most bleeding-edge part of the C++17 `<filesystem>` library. What I am about to describe is neither implemented by Boost, nor by `<experimental/filesystem>`.

Retrieving a file's metadata from its inode is done by querying an object of type `fs::directory_entry`. If you're familiar with the POSIX approach to retrieving metadata, imagine that a `fs::directory_entry` contains a member of `type fs::path` and a member of type `std::optional<struct stat>`. Calling `entry.refresh()` is, basically, the same thing as calling the POSIX function `stat()`; and calling any `accessor` method, such as `entry.file_size()`, will implicitly call `stat()` if and only if the optional member is still disengaged. Merely constructing an instance of `fs::directory_entry` won't query the filesystem; the library waits until you ask a specific question before it acts. Asking a specific question, such as `entry.file_size()`, may cause the library to query the filesystem, or (if the optional member is already engaged) it might just use the cached value from the last time it queried.

[PRE30]

An older way to accomplish the same goal is to use `fs::status("path")` or `fs::symlink_status("path")` to retrieve an instance of the class `fs::file_status`, and then to pull information out of the `file_status` object via cumbersome operations such as `status.type() == fs::file_type::directory`. I recommend you not try to use `fs::file_status`; prefer to use `entry.is_directory()` and so on. For the masochistic, you can still retrieve a `fs::file_status` instance directly from a `directory_entry`: `entry.status()` is the equivalent of `fs::status(entry.path())`, and `entry.symlink_status()` is the equivalent of `fs::symlink_status(entry.path())`, which, in turn, is a slightly faster equivalent of
`fs::status(entry.is_symlink() ? fs::read_symlink(entry.path()) : entry.path())`.

Incidentally, the free function `fs::equivalent(p, q)` can tell you if two paths are both hard-linked to the same inode; and `entry.hard_link_count()` can tell you the total number of hard-links to this particular inode. (The only way to determine the *names* of those hard-links is to walk the entire filesystem; and even then, your current user account might not have the permission to stat those paths.)

# Walking directories with directory_iterator

A `fs::directory_iterator` is just what it says on the tin. An object of this type lets you walk the contents of a single directory, entry by entry:

[PRE31]

Incidentally, notice the use of `entry.path().string()` in the preceding code. This is required, because `operator<<` acts extremely bizarrely on path objects--it always outputs as if you'd written `std::quoted(path.string())`. If you want the path itself, with no extra quotes, you always have to convert to `std::string` before outputting. (Similarly, `std::cin >> path` won't work to get a path from the user, but that's less obnoxious, since you should never use `operator>>` anyway. See [Chapters 9](part0144.html#49AH00-2fdac365b8984feebddfbb9250eaf20d), *Iostreams*, and [Chapter 10](part0161.html#4PHAI0-2fdac365b8984feebddfbb9250eaf20d), *Regular Expressions*, for more information on lexing and parsing input from the user.)

# Recursive directory walking

To recurse down a whole directory tree, in the style of Python's `os.walk()`, you can use this recursive function modeled on the previous code snippet:

[PRE32]

Or, you can simply use a `fs::recursive_directory_iterator`:

[PRE33]

The constructor of `fs::recursive_directory_iterator` can take an extra argument of type `fs::directory_options`, which modifies the exact nature of the recursion. For example, you can pass `fs::directory_options::follow_directory_symlink` to follow symlinks, although this is a good way to wind up in an infinite loop if a malicious user creates a symlink pointing back to its own parent directory.

# Modifying the filesystem

Most of the `<filesystem>` header's facilities are concerned with examining the filesystem, not modifying it. But there are several gems hidden in the rubble. Many of these functions seem designed to make the effects of the classic POSIX command-line utilities available in portable C++:

*   `fs::copy_file(old_path, new_path)` : Copy the file at `old_path` to a new file (that is, a new inode) at `new_path`, as if by `cp -n`. Error if `new_path` already exists.
*   `fs::copy_file(old_path, new_path, fs::copy_options::overwrite_existing)`: Copy `old_path` to `new_path`. Overwrite `new_path` if possible. Error if `new_path` exists and is not a regular file, or if it's the same as `old_path`.
*   `fs::copy_file(old_path, new_path, fs::copy_options::update_existing)`: Copy `old_path` to `new_path`. Overwrite `new_path` if and only if it's older than the file at `old_path`.
*   `fs::copy(old_path, new_path, fs::copy_options::recursive | fs::copy_options::copy_symlinks)`: Copy an entire directory from `old_path` to `new_path` as if by `cp -R`.
*   `fs::create_directory(new_path)`: Create a directory as if by `mkdir`.
*   `fs::create_directories(new_path)`: Create a directory as if by mkdir `-p`.
*   `fs::create_directory(new_path, old_path)` (notice the reversal of the arguments!): Create a directory, but copy its attributes from those of the directory at `old_path`.
*   `fs::create_symlink(old_path, new_path)`: Create a symlink from `new_path` to `old_path`.
*   `fs::remove(path)`: Remove a file or an empty directory as if by `rm`.
*   `fs::remove_all(path)`: Remove a file or directory as if by `rm -r`.
*   `fs::rename(old_path, new_path)`: Rename a file or directory as if by `mv`.
*   `fs::resize_file(path, new_size)`: Extend (with zeroes) or truncate a regular file.

# Reporting disk usage

Speaking of classic command-line utilities, one final thing we might want to do with a filesystem is ask how full it is. This is the domain of the command-line utility `df -h` or the POSIX library function `statvfs`. In C++17, we can do it with `fs::space("path")`, which returns (by value) a struct of type `fs::space_info`:

[PRE34]

Each of these fields is measured in bytes, and we should have `available <= free <= capacity`. The distinction between `available` and `free` has to do with user limits: On some filesystems, a portion of the free space might be reserved for the root user, and on others, there might be per-user-account disk quotas.

# Summary

Use namespace aliases to save typing, and to allow dropping in alternative implementations
of a library namespace, such as Boost.

`std::error_code` provides a very neat way to pass integer error codes up the stack without exception handling; consider using it if you work in a domain where exception handling is frowned upon. (In which case, that is likely *all* you will be able to take away from this particular chapter! The `<filesystem>` library provides both throwing and non-throwing APIs; however, both APIs use the heap-allocating (and, potentially, throwing `fs::path` as a vocabulary type. The only reason to use the non-throwing API is if it eliminates a case of "using exceptions for control flow.)

`std::error_condition` provides only syntactic sugar for "catching" error codes; avoid it like the plague.

A `path` consists of a `root_name`, a `root_directory`, and a `relative_path`; the last of these is made up of *names* separated by slashes. To POSIX, a *name* is a string of raw bytes; to Windows, a *name* is a string of Unicode characters. The `fs::path` type attempts to use the appropriate kind of string for each platform. To avoid portability problems, beware of `path.c_str()` and implicit conversions to `fs::path::string_type`.

Directories store mappings from *names* to *inodes* (which the C++ standard just calls "files"). In C++, you can loop over an `fs::directory_iterator` to retrieve `fs::directory_entry` objects; methods on the `fs::directory_entry` allow you to query the corresponding inode. Restatting an inode is as simple as calling `entry.refresh()`.

`<filesystem>` provides a whole zoo of free functions for creating, copying, renaming, removing, and resizing files and directories, and one last function to get the total capacity of the filesystem.

Much of what was discussed in this chapter (the `<filesystem>` parts, at least) is bleeding-edge C++17 that, as of press time, has not been implemented by any compiler vendor. Use such new features with caution.