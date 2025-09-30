# Chapter 12

# The Most Recent C

Change cannot be prevented, and C is no exception. The C programming language is standardized by an ISO standard, and it is constantly under revision by a group of people who are trying to make it better and bring new features to it. This doesn't mean that the language will necessarily get easier, however; we might see novel and complex features emerge in the language as new content is added.

In this chapter, we are going to have a brief look at C11's features. You might know that C11 has replaced the old C99 standard, and it has been superseded by the C18 standard. In other words, C18 is the latest version of the C standard, and just before that we had C11.

It's interesting to know that C18 doesn't offer any new features; it just contains fixes for the issues found in C11\. Therefore, talking about C11 is basically the same as talking about C18, and it will lead us to the most recent C standard. As you can see, we are observing constant improvement in the C language… contrary to the belief that it is a long-dead language!

This chapter will give a brief overview of the following topics:

*   How to detect the C version and how to write a piece of C code which is compatible with various C versions
*   New features for writing optimized and secure code, such as *no-return* functions and *bounds-checking* functions
*   New data types and memory alignment techniques
*   Type-generic functions
*   Unicode support in C11, which was missing from the language in the older standards
*   Anonymous structures and unions
*   Standard support for multithreading and synchronization techniques in C11

Let's begin the chapter by talking about C11 and its new features.

# C11

Gathering a new standard for a technology that has been in use for more than 30 years is not an easy task. Millions (if not billions!) of lines of C code exist, and if you are about to introduce new features, this must be done while keeping previous code or features intact. New features shouldn't create new problems for the existing programs, and they should be bug-free. While this view seems to be idealistic, it is something that we should be committed to.

The following PDF document resides on the *Open Standards* website and contains the worries and thoughts that people in the C community had in mind before starting to shape C11: http://www.open-std.org/JTC1/SC22/wg14/www/docs/n1250.pdf. It would be useful to give it a read because it will introduce you to the experience of authoring a new standard for a programming language that several thousand pieces of software have been built upon.

Finally, with these things in mind, we consider the release of C11\. When C11 came out, it was not in its ideal form and was in fact suffering from some serious defects. You can see the list of these defects he[re: http://www.open-std.org/jtc1/sc22/wg14/www/docs/n2244](http://www.open-std.org/jtc1/sc22/wg14/www/docs/n2244.htm).htm.

Seven years after the launch of C11, C18 was introduced, which came about to fix the defects found in C11\. Note that C18 is also *informally* referred to as C17, and both C17 and C18 refer to the same C standard. If you open the previous link you will see the defects and their current statuses. If the status of a defect is "C17," it means that the defect is solved as part of C18\. This shows how hard and delicate process it is to assemble a standard that has as many users as C does.

In the following sections, we'll talk about the new features of C11\. Before going through them however, we need a way to be sure that we are really writing C11 code, and that we are using a compatible compiler. The following section will address this requirement.

# Finding a supported version of C standard

At the time of writing, it has been almost 8 years since C11 came out. Therefore, it would be expected that many compilers should support the standard, and this is indeed the case. Open source compilers such as `gcc` and `clang` both support C11 perfectly, and they can switch back to C99 or even older versions of C if needed. In this section, we show how to use specific macros to detect the C version and, depending on the version, how to use the supported features.

The first thing that is necessary when using a compiler that supports different versions of the C standard is being able to identify which version of the C standard is currently in use. Every C standard defines a special macro that can be used to find out what version is being used. So far, we have used `gcc` in Linux and `clang` in macOS systems. As of version 4.7, `gcc` offers C11 as one of its supported standards.

Let's look at the following example and see how already-defined macros can be used to detect the current version of the C standard at runtime:

```cpp
#include <stdio.h>
int main(int argc, char** argv) {
#if __STDC_VERSION__ >=  201710L
  printf("Hello World from C18!\n");
#elif __STDC_VERSION__ >= 201112L
  printf("Hello World from C11!\n");
#elif __STDC_VERSION__ >= 199901L
  printf("Hello World from C99!\n");
#else
  printf("Hello World from C89/C90!\n");
#endif
  return 0;
}
```

Code Box 12-1 [ExtremeC_examples_chapter12_1.c]: Detecting the version of the C standard

As you can see, the preceding code can distinguish between various versions of the C standard. In order to see how various C versions can lead to various printings, we have to compile the preceding source code multiple times with various versions of C standard that are supported by the compiler.

To ask the compiler to use a specific version of the C standard, we have to pass the -`std=CXX` option to the C compiler. Look at the following commands and the produced output:

```cpp
$ gcc ExtremeC_examples_chapter12_1.c -o ex12_1.out
$ ./ex12_1.out
Hello World from C11!
$ gcc ExtremeC_examples_chapter12_1.c -o ex12_1.out -std=c11
$ ./ex12_1.out
Hello World from C11!
$ gcc ExtremeC_examples_chapter12_1.c -o ex12_1.out -std=c99
$ ./ex12_1.out
Hello World from C99!
$ gcc ExtremeC_examples_chapter12_1.c -o ex12_1.out -std=c90
$ ./ex12_1.out
Hello World from C89/C90!
$ gcc ExtremeC_examples_chapter12_1.c -o ex12_1.out -std=c89
$ ./ex12_1.out
Hello World from C89/C90!
$
```

Shell Box 12-1: Compiling example 12.1 with various versions of C standard

As you can see, the default C standard version in newer compilers is C11\. With older versions, you have to specify the version using the `-std` option, if you want to enable C11\. Note the comments made at the beginning of the file. I have used `/* ... */` comments (multiline comments) instead of `//` comments (one-line comments). That's because one-line comments were not supported in standards before C99\. Therefore, we had to use multiline comments in order to have the preceding code compiled with all C versions.

# Removal of the gets function

In C11, the famous `gets` function is removed. The `gets` function was subject to *buffer overflow* attacks, and in older versions it was decided to make the function *deprecated*. Later, as part of the C11 standard, it was removed. Therefore, older source code that uses the `gets` function won't be compiled using a C11 compiler.

The `fgets` function can be used instead of `gets`. The following is an excerpt from the `gets` manual page (man page) in macOS:

> SECURITY CONSIDERATIONS

> The gets() function cannot be used securely. Because of its lack of bounds checking, and the inability for the calling program to reliably determine the length of the next incoming line, the use of this function enables malicious users to arbitrarily change a running program's functionality through a buffer overflow attack. It is strongly suggested that the fgets() function be used in all cases. (See the FSA.)

# Changes to fopen function

The `fopen` function is usually used for opening a file and returning a file descriptor to that file. The concept of a file is very general in Unix, and by using the term *file*, we don't necessarily mean a file located on the filesystem. The `fopen` function has the following signatures:

```cpp
FILE* fopen(const char *pathname, const char *mode);
FILE* fdopen(int fd, const char *mode);
FILE* freopen(const char *pathname, const char *mode, FILE *stream);
```

Code Box 12-2: Various signatures of the family of fopen functions

As you can see, all of the preceding signatures accept a `mode` input. This input parameter is a string that determines how the file should be opened. The following description in *Shell Box 12-2* is obtained from the FreeBSD manual for the `fopen` function and explains how `mode` should be used:

```cpp
$ man 3 fopen
...
The argument mode points to a string beginning with one of the following letters:
     "r"     Open for reading.  The stream is positioned at the beginning
             of the file.  Fail if the file does not exist.
     "w"     Open for writing.  The stream is positioned at the beginning
             of the file.  Create the file if it does not exist.
     "a"     Open for writing.  The stream is positioned at the end of
             the file. Subsequent writes to the file will always end up
             at the then current end of file, irrespective of 
             any intervening fseek(3) or similar. Create the file 
             if it does not exist.
     An optional "+" following "r", "w", or "a" opens the file
     for both reading and writing.  An optional "x" following "w" or
     "w+" causes the fopen() call to fail if the file already exists.
     An optional "e" following the above causes the fopen() call to set
     the FD_CLOEXEC flag on the underlying file descriptor.
     The mode string can also include the letter "b" after either 
     the "+" or the first letter.
...
$
```

Shell Box 12-2: An excerpt from the fopen's manual page in FreeBSD

The mode `x`, explained in the preceding extract from the `fopen` manual page, has been introduced as part of C11\. To open a file in order to write to it, the mode `w` or `w+` should be supplied to `fopen`. The problem is that, if the file already exists, the `w` or `w+` mode will truncate (empty) the file.

Therefore, if the programmer wants to append to a file and keep its current content, they have to use a different mode, `a`. Hence, they have to check for the file's existence, using a filesystem API such as `stat`, before calling `fopen`, and then choose the proper mode based on the result. Now however, with the new mode `x`, the programmer first tries with the mode `wx` or `w+x`, and if the file already exists the `fopen` will fail. Then the programmer can continue with the `a` mode.

Thus, less boilerplate code needs to be written to open a file without using the filesystem API to check for the file's existence. From now on, `fopen` is enough to open a file in every desired mode.

Another change in C11 was the introduction of the `fopen_s` API. This function serves as a secure `fopen`. According to the documentation for `fopen_s` found at https://en.cppreference.com/w/c/io/fopen, performs extra checking on the provided buffers and their boundaries in order to detect any flaw in them.

# Bounds-checking functions

One of the serious problems with C programs operating on strings and byte arrays is the ability to go easily beyond the boundary defined for a buffer or a byte array.

As a reminder, a buffer is a region of memory that is used as the place holder for a byte array or a string variable. Going beyond the boundary of a buffer causes a *buffer overflow* and based on that a malicious entity can organize an attack (usually called a *buffer overflow attack*). This type of attack either results in a **denial of service** (**DOS**) or in *exploitation* of the victim C program.

Most such attacks usually start in a function operating on character or byte arrays. String manipulation functions found in `string.h`, such as `strcpy` and `strcat`, are among the *vulnerable* functions that lack a boundary checking mechanism to prevent buffer overflow attacks.

However, as part of C11, a new set of functions has been introduced. *Bounds-checking* functions borrow the same name from the string manipulation functions but with an `_s` at the end. The suffix `_s` distinguishes them as a *secure* or *safe* flavor of those functions that conduct more runtime checks in order to shut down the vulnerabilities. Functions such as `strcpy_s` and `strcat_s` have been introduced as part of bounds-checking functions in C11.

These functions accept some extra arguments for the input buffers that restrict them from performing dangerous operations. As an example, the `strcpy_s` function has the following signature:

```cpp
errno_t strcpy_s(char *restrict dest, rsize_t destsz, const char *restrict src);
```

Code Box 12-3: Signature of the strcpy_s function

As you can see, the second argument is the length of the `dest` buffer. Using that, the function performs some runtime checks, such as ensuring that the `src` string is shorter than or at the same size of the `dest` buffer in order to prevent writing to unallocated memory.

# No-return functions

A function call can end either by using the `return` keyword or by reaching the end of the function's block. There are also situations in which a function call never ends, and this is usually done intentionally. Look at the following code example contained in *Code Box 12-4*:

```cpp
void main_loop() {
  while (1) {
    ...
  }
}

int main(int argc, char** argv) {
  ...
  main_loop();
  return 0;
}
```

Code Box 12-4: Example of a function that never returns

As you can see, the function `main_loop` performs the main task of the program, and if we return from the function, the program could be considered as finished. In these exceptional cases, the compiler can perform some more optimizations, but somehow, it needs to know that the function `main_loop` never returns.

In C11, you have the ability to mark a function as a *no-return* function. The `_Noreturn` keyword from the `stdnoreturn.h` header file can be used to specify that a function never exits. So, the code in *Code Box 12-4* can be changed for C11 to look like this:

```cpp
_Noreturn void main_loop() {
  while (true) {
    ...
  }
}
```

Code Box 12-5: Using the _Noreturn keyword to mark main_loop as a never-ending function

There are other functions, such as `exit`, `quick_exit` (added recently as part of C11 for quick termination of the program), and `abort`, that are considered to be no-return functions. In addition, knowing about no-return functions allows the compiler to recognize function calls that unintentionally won't return and produce proper warnings because they could be a sign of a logical bug. Note that if a function marked as `_Noreturn` returns, then this would be an *undefined behavior* and it is highly discouraged.

# Type generic macros

In C11, a new keyword has been introduced: `_Generic`. It can be used to write macros that are type-aware at compile time. In other words, you can write macros that can change their value based on the type of their arguments. This is usually called *generic selection*. Look at the following code example in *Code Box 12-6*:

```cpp
#include <stdio.h>
#define abs(x) _Generic((x), \
                        int: absi, \
                        double: absd)(x)
int absi(int a) {
  return a > 0 ? a : -a;
}
double absd(double a) {
  return a > 0 ? a : -a;
}
int main(int argc, char** argv) {
  printf("abs(-2): %d\n", abs(-2));
  printf("abs(2.5): %f\n", abs(2.5));;
  return 0;
}
```

Code Box 12-6: Example of a generic macro

As you can see in the macro definition, we have used different expressions based on the type of the argument `x`. We use `absi` if it is an integer value, and `absd` if it is a double value. This feature is not new to C11, and you can find it in older C compilers, but it wasn't part of the C standard. As of C11, it is now standard, and you can use this syntax to write type-aware macros.

# Unicode

One of the greatest features that has been added to the C11 standard is support for Unicode through UTF-8, UTF-16, and UTF-32 encodings. C was missing this feature for a long time, and C programmers had to use third-party libraries such as **IBM International Components for Unicode** (**ICU**) to fulfill their needs.

Before C11, we only had `char` and `unsigned char` types, which were 8-bit variables used to store ASCII and Extended ASCII characters. By creating arrays of these ASCII characters, we could create ASCII strings.

**Note**:

ASCII standard has 128 characters which can be stored in 7 bits. Extended ASCII is an extension to ASCII which adds another 128 characters to make them together 256 characters. Then, an 8-bit or one-byte variable is enough to store all of them. In the upcoming text, we will only use the term ASCII, and by that we refer to both ASCII standard and Extended ASCII.

Note that support for ASCII characters and strings is fundamental, and it will never be removed from C. Thus, we can be confident that we will always have ASCII support in C. From C11, they have added support for new characters, and therefore new strings that use a different number of bytes, not just one byte, for each character.

To explain this further, in ASCII, we have one byte for each character. Therefore, the bytes and characters can be used interchangeably, but this is *not* true in general. Different encodings define new ways to store a wider range of characters in multiple bytes.

In ASCII, altogether we have 256 characters. Therefore, a single one-byte (8-bit) character is enough to store all of them. If we are going to have more than 256 characters, however, we must use more than one byte to store their numerical values after 255\. Characters that need more than one byte to store their values are usually called *wide characters*. By this definition, ASCII characters are not considered as wide characters.

The Unicode standard introduced various methods of using more than one byte to encode all characters in ASCII, Extended ASCII, and wide characters. These methods are called *encodings*. Through Unicode, there are three well-known encodings: UTF-8, UTF-16, and UTF-32\. UTF-8 uses the first byte for storing the first half of the ASCII characters, and the next bytes, usually up to 4 bytes, for the other half of ASCII characters together with all other wide characters. Hence, UTF-8 is considered as a variable-sized encoding. It uses certain bits in the first byte of the character to denote the number of actual bytes that should be read to retrieve the character fully. UTF-8 is considered a superset of ASCII because for ASCII characters (not Extended ASCII characters) the representation is the same.

Like UTF-8, UTF-16 uses one or two *words* (each word has 16 bits within) for storing all characters; hence it is also a variable-sized encoding. UTF-32 uses exactly 4 bytes for storing the values of all characters; therefore, it is a fixed-sized encoding. UTF-8, and after that, UTF-16, are suitable for the applications in which a smaller number of bytes should be used for more frequent characters.

UTF-32 uses a fixed number of bytes even for ASCII characters. So, it consumes more memory space to store strings using this encoding compared to others; but it requires less computation power when using UTF-32 characters. UTF-8 and UTF-16 can be considered as compressed encodings, but they need more computation to return the actual value of a character.

**Note**:

More information about UTF-8, UTF-16, and UTF-32 strings and how to decode them can be found on Wik[ipedia or other sources like:](https://unicodebook.readthedocs.io/unicode_encodings.html)

[https://unicodebook.readth](https://unicodebook.readthedocs.io/unicode_encodings.html)e[docs.io/unicode_encodings.html](https://javarevisited.blogspot.com/2015/02/difference-between-utf-8-utf-16-and-utf.html)

[https://javarevisited.blogspot.com/2015/02/difference-be](https://javarevisited.blogspot.com/2015/02/difference-between-utf-8-utf-16-and-utf.html)t[ween-utf-8-utf-16-and-utf.html](https://unicodebook.readthedocs.io/unicode_encodings.html).

In C11 we have support for all the above Unicode encodings. Look at the following example, *example 12.3*. It defines various ASCII, UTF-8, UTF-16, and UTF-32 strings, and counts the number of actual bytes used to store them and the number of characters observed within them. We present the code in multiple steps in order to give additional comments on the code. The following code box demonstrates the inclusions and declarations required:

```cpp
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#ifdef __APPLE__
#include <stdint.h>
typedef uint16_t char16_t;
typedef uint32_t char32_t;
#else
#include <uchar.h> // Needed for char16_t and char32_t
#endif
```

Code Box 12-7 [ExtremeC_examples_chapter12_3.c]: Inclusions and declarations required for example 12.3 to get built

The preceding lines are the `include` statements for *example 12.3*. As you can see, in macOS we do not have the `uchar.h` header and we have to define new types for the `char16_t` and `char32_t` types. The whole functionality of Unicode strings is supported, however. On Linux, we don't have any issues with Unicode support in C11.

The next part of the code demonstrates the functions used for counting the number of bytes and characters in various kinds of Unicode strings. Note that no utility function is offered by C11 to operate on Unicode strings, therefore we have to write a new `strlen` for them. In fact, our versions of `strlen` functions do more just than returning the number of characters; they return the number of consumed bytes as well. The implementation details won't be described, but it is strongly recommended to give them a read:

```cpp
typedef struct {
  long num_chars;
  long num_bytes;
} unicode_len_t;
unicode_len_t strlen_ascii(char* str) {
  unicode_len_t res;
  res.num_chars = 0;
  res.num_bytes = 0;
  if (!str) {
    return res;
  }
  res.num_chars = strlen(str) + 1;
  res.num_bytes = strlen(str) + 1;
  return res;
}
unicode_len_t strlen_u8(char* str) {
  unicode_len_t res;
  res.num_chars = 0;
  res.num_bytes = 0;
  if (!str) {
    return res;
  }
  // Last null character
  res.num_chars = 1;
  res.num_bytes = 1;
  while (*str) {
    if ((*str | 0x7f) == 0x7f) { // 0x7f = 0b01111111
      res.num_chars++;
      res.num_bytes++;
      str++;
    } else if ((*str & 0xc0) == 0xc0) { // 0xc0 = 0b11000000
      res.num_chars++;
      res.num_bytes += 2;
      str += 2;
    } else if ((*str & 0xe0) == 0xe0) { // 0xe0 = 0b11100000
      res.num_chars++;
      res.num_bytes += 3;
      str += 3;
    } else if ((*str & 0xf0) == 0xf0) { // 0xf0 = 0b11110000
      res.num_chars++;
      res.num_bytes += 4;
      str += 4;
    } else {
      fprintf(stderr, "UTF-8 string is not valid!\n");
      exit(1);
    }
  }
  return res;
}
unicode_len_t strlen_u16(char16_t* str) {
  unicode_len_t res;
  res.num_chars = 0;
  res.num_bytes = 0;
  if (!str) {
    return res;
  }
  // Last null character
  res.num_chars = 1;
  res.num_bytes = 2;
  while (*str) {
    if (*str < 0xdc00 || *str > 0xdfff) {
      res.num_chars++;
      res.num_bytes += 2;
      str++;
    } else {
      res.num_chars++;
      res.num_bytes += 4;
      str += 2;
    }
  }
  return res;
}
unicode_len_t strlen_u32(char32_t* str) {
  unicode_len_t res;
  res.num_chars = 0;
  res.num_bytes = 0;
  if (!str) {
    return res;
  }
  // Last null character
  res.num_chars = 1;
  res.num_bytes = 4;
  while (*str) {
      res.num_chars++;
      res.num_bytes += 4;
      str++;
  }
  return res;
}
```

Code Box 12-8 [ExtremeC_examples_chapter12_3.c]: The definitions of the functions used in example 12.3

The last part is the `main` function. It declares some different strings in English, Persian, and some alien language to evaluate the preceding functions:

```cpp
int main(int argc, char** argv) {
  char ascii_string[32] = "Hello World!";
  char utf8_string[32] = u8"Hello World!";
  char utf8_string_2[32] = u8"درود دنیا!";
  char16_t utf16_string[32] = u"Hello World!";
  char16_t utf16_string_2[32] = u"درود دنیا!";
  char16_t utf16_string_3[32] = u"হহহ!";
  char32_t utf32_string[32] = U"Hello World!";
  char32_t utf32_string_2[32] = U"درود دنیا!";
  char32_t utf32_string_3[32] = U"হহহ!";
  unicode_len_t len = strlen_ascii(ascii_string);
  printf("Length of ASCII string:\t\t\t %ld chars, %ld bytes\n\n",
      len.num_chars, len.num_bytes);
  len = strlen_u8(utf8_string);
  printf("Length of UTF-8 English string:\t\t %ld chars, %ld bytes\n",
      len.num_chars, len.num_bytes);
  len = strlen_u16(utf16_string);
  printf("Length of UTF-16 english string:\t %ld chars, %ld bytes\n",
      len.num_chars, len.num_bytes);
  len = strlen_u32(utf32_string);
  printf("Length of UTF-32 english string:\t %ld chars, %ld bytes\n\n",
      len.num_chars, len.num_bytes);
  len = strlen_u8(utf8_string_2);
  printf("Length of UTF-8 Persian string:\t\t %ld chars, %ld bytes\n",
      len.num_chars, len.num_bytes);
  len = strlen_u16(utf16_string_2);
  printf("Length of UTF-16 persian string:\t %ld chars, %ld bytes\n",
      len.num_chars, len.num_bytes);
  len = strlen_u32(utf32_string_2);
  printf("Length of UTF-32 persian string:\t %ld chars, %ld bytes\n\n",
      len.num_chars, len.num_bytes);
  len = strlen_u16(utf16_string_3);
  printf("Length of UTF-16 alien string:\t\t %ld chars, %ld bytes\n",
      len.num_chars, len.num_bytes);
  len = strlen_u32(utf32_string_3);
  printf("Length of UTF-32 alien string:\t\t %ld chars, %ld bytes\n",
      len.num_chars, len.num_bytes);
  return 0;
}
```

Code Box 12-9 [ExtremeC_examples_chapter12_3.c]: The main function of example 12.3

Now, we must compile the preceding example. Note that the example can only be compiled using a C11 compiler. You can try using older compilers and take a look at the resulting errors. The following commands compile and run the preceding program:

```cpp
$ gcc ExtremeC_examples_chapter12_3.c -std=c11 -o ex12_3.out
$ ./ex12_3.out
Length of ASCII string:            13 chars, 13 bytes
Length of UTF-8 english string:      13 chars, 13 bytes
Length of UTF-16 english string:     13 chars, 26 bytes
Length of UTF-32 english string:     13 chars, 52 bytes
Length of UTF-8 persian string:      11 chars, 19 bytes
Length of UTF-16 persian string:     11 chars, 22 bytes
Length of UTF-32 persian string:     11 chars, 44 bytes
Length of UTF-16 alien string:       5 chars, 14 bytes
Length of UTF-32 alien string:       5 chars, 20 bytes
$
```

Shell Box 12-3: Compiling and running example 12.3

As you can see, the same string with the same number of characters uses a different number of bytes to encode and store the same value. UTF-8 uses the least number of bytes, especially when a large number of characters in a text are ASCII characters, simply because most of the characters will use only one byte.

As we go through the characters that are more distinct from the Latin characters, such as characters in Asian languages, UTF-16 has a better balance between the number of characters and the number of used bytes, because most of the characters will use up to two bytes.

UTF-32 is rarely used, but it can be used in systems where having a fixed-length *code print* for characters is useful; for example, if the system suffers from low computational power or is benefiting from some parallel processing pipelines. Therefore, UTF-32 characters can be used as keys in mappings from the characters to any kind of data. In other words, they can be used to build up some indexes to look up data very quickly.

# Anonymous structures and anonymous unions

Anonymous structures and anonymous unions are type definitions without names, and they are usually used in other types as a nested type. It is easier to explain them with an example. Here, you can see a type that has both an anonymous structure and an anonymous union in one place, displayed in *Code Box 12-10*:

```cpp
typedef struct {
  union {
    struct {
      int x;
      int y;
    };
    int data[2];
  };
} point_t;
```

Code Box 12-10: Example of an anonymous structure together with an anonymous union

The preceding type uses the same memory for the anonymous structure and the byte array field `data`. The following code box shows how it can be used in a real example:

```cpp
#include <stdio.h>
typedef struct {
  union {
    struct {
      int x;
      int y;
    };
    int data[2];
  };
} point_t;
int main(int argc, char** argv) {
  point_t p;
  p.x = 10;
  p.data[1] = -5;
  printf("Point (%d, %d) using an anonymous structure inside an anonymous union.\n", p.x, p.y);
  printf("Point (%d, %d) using byte array inside an anonymous union.\n",
      p.data[0], p.data[1]);
  return 0;
}
```

Code box 12-11 [ExtremeC_examples_chapter12_4.c]: The main function using an anonymous structure together with an anonymous union

In this example we are creating an anonymous union that has an anonymous structure within. Therefore, the same memory region is used to store an instance of the anonymous structure and the two-element integer array. Next, you can see the output of the preceding program:

```cpp
$ gcc ExtremeC_examples_chapter12_4.c -std=c11 -o ex12_4.out
$ ./ex12_4.out
Point (10, -5) using anonymous structure.
Point (10, -5) using anonymous byte array.
$
```

Shell Box 12-4: Compiling and running example 12.4

As you can see, any changes to the two-element integer array can be seen in the structure variable, and vice versa.

# Multithreading

Multithreading support has been available in C for a long time via POSIX threading functions, or the `pthreads` library. We have covered multithreading thoroughly in *Chapter 15*, *Thread Execution*, and *Chapter 16*, *Thread Synchronization*.

The POSIX threading library, as the name implies, is only available in POSIX-compliant systems such as Linux and other Unix-like systems. Therefore, if you are on a non-POSIX compliant operating system such as Microsoft Windows, you have to use the library provided by the operating system. As part of C11, a standard threading library is provided that can be used on all systems that are using standard C, regardless of whether it's POSIX-compliant or not. This is the biggest change we see in the C11 standard.

Unfortunately, C11 threading is not implemented for Linux and macOS. Therefore, we cannot provide working examples at the time of writing.

# A bit about C18

As we've mentioned in the earlier sections, the C18 standard contains all the fixes that were made in C11, and no new feature has been introduced as part of it. As said before, the following link takes you to a page on which you can see the issues created and being tracked for C11 and the discussions around them: http://www.open-std.org/jtc1/sc22/wg14/www/docs/n2244.htm.

# Summary

In this chapter, we went through C11, C18, and the most recent C standards, and we explored C11's various new features. Unicode support, anonymous structures and unions, and the new standard threading library (despite the fact that it is not available in recent compilers and platforms to date) are among the most important features that have been introduced in modern C. We will look forward to seeing new versions of the C standard in the future.

In the next chapter, we begin to talk about concurrency and the theory behind concurrent systems. This will begin a long journey through six chapters in which we'll cover multithreading and multi-processing in order to fulfil our purpose to be able to write concurrent systems.