# Appendix A. Rational and Complex Numbers

This Appendix defines the `Rational` and `Complex` classes from the *Converters* section in the previous chapter.

# Rational numbers

A **rational** **number** can be expressed as a fraction of two integers, called the **numerator** and **denominator**.

**Rational.h**

[PRE0]

The default constructor initializes the numerator and denominator to 0 and 1, respectively. The second constructor takes a string and throws a `NotaRationalNumber` exception if the string does not hold a valid rational number. The copy constructor and the assignment operator take another rational number. The `String` conversion operator returns the rational number as a string:

[PRE1]

A rational number is always normalized when it has been created by the constructor or any of the arithmetic operators: the numerator and the denominator are divided by their **Greatest Common Divisor** (**GCD**):

[PRE2]

**Rational.cpp**

[PRE3]

The default constructor initializes the numerator and the denominator, and throws an exception if the denominator is zero. This constructor and the next constructor that takes a string are actually the only places where the denominator can be zero. The following constructors and arithmetic operators always produce a rational number with non-zero denominators:

[PRE4]

Text can hold a rational number in two formats: as an integer followed by a slash (**/**) and another integer, or as a single integer. We start by initializing the numerator and the denominator to 0 and 1:

[PRE5]

First, we try two integers and a slash; we read the numerator, slash, and denominator. Before the slash we set the `skipws` flag, which causes the stream to skip any potential white spaces before the slash. If we have reached the end of the line, the denominator is not 0, the character read into the `slash` variable really is a slash, the text holds a rational number, and we have read the numerator and denominator, then we are done and we return:

[PRE6]

If using two integers and a slash does not work, we try the case of a single integer. We create a new stream and read the numerator. If we have reached the end of the stream after that, the string holds a valid integer. We let the numerator hold its initialized value, which was 1, and return.

[PRE7]

If two integers and a slash as well as a single integer both failed, we have to draw the conclusion that the string does not hold a valid rational number, and we throw a `NotaRationalNumber` exception:

[PRE8]

The copy constructor simply copies the numerator and denominator of the rational number:

[PRE9]

The assignment operator also copies the numerator and denominator of the rational number and returns its own `Rational` object (`*this`):

[PRE10]

The `String` conversion operator creates an `OStringStream` object and looks into the denominator. If it is 1, the rational number can be expressed as a single integer; otherwise, it needs to be expressed as a fraction of the numerator and denominator. Finally, the stream is converted into a string that is returned:

[PRE11]

As rational numbers are always normalized, we can conclude that two rational numbers are equal if they have the same numerator and denominator:

[PRE12]

When deciding whether a rational number is smaller than another rational number, in order not to involve floating values, we multiply both sides by the denominator and compare the products:

![Rational numbers](img/B05475_Appendix_01.jpg)

[PRE13]

When adding two rational numbers, we multiply the numerator by the opposite denominator in each term:

![Rational numbers](img/B05475_Appendix_02.jpg)

[PRE14]

When subtracting two rational numbers, we also multiply the numerator by the opposite denominator in each term:

![Rational numbers](img/B05475_Appendix_03.jpg)

[PRE15]

When multiplying two rational numbers, we simply multiply the numerators and denominators:

![Rational numbers](img/B05475_Appendix_04.jpg)

[PRE16]

When dividing two rational numbers, we invert the second operand and then multiply the numerators and denominators:

![Rational numbers](img/B05475_Appendix_05.jpg)

[PRE17]

When normalizing the rational number, we first look into the numerator. If it is 0, we set the denominator to 1 regardless of its previous value and return:

[PRE18]

However, if the numerator is not 0, we look into the denominator. If it is less than 0, we switch the sign of both the numerator and denominator so that the denominator is always greater than 0:

[PRE19]

Then we calculate the Greatest Common Divisor by calling `GCD`, and then we divide both the numerator and denominator by the Greatest Common Divisor:

[PRE20]

The `GCD` method calls itself recursively by comparing the numbers and subtracting the smaller number from the larger number. When they are equal, we return the number. The GCD algorithm is regarded as the world's oldest non-trivial algorithm.

[PRE21]

# Complex numbers

A **complex** **number** *z* = *x* + *yi* is the sum of a real number *x* and a real number *y* multiplied by the **imaginary unit** *i*, *i* ² = -1 ⇒ *i* = ±√(-1) , which is the solution of the equation *x* ² + 1 = 0.

**Complex.h**

[PRE22]

The constructors, assignment operators, and the `String` conversion operator are similar to their counterparts in `Rational`:

[PRE23]

When comparing two complex number, their absolute values (refer to `Abs`) are compared.

[PRE24]

The arithmetic operators apply to complex numbers and double values:

[PRE25]

The absolute value of a complex number (and its value converted to a `double`) is the Pythagoras theorem of the real and imaginary part, that is, the square root of the sum of the squares of the parts:

[PRE26]

**Complex.cpp**

[PRE27]

When interpreting a text holding a rational number, we read the text from a stream, and we need some auxiliary functions to start with. The `ReadWhiteSpaces` method reads (and disposes of) all white spaces at the beginning of the stream:

[PRE28]

The `Peek` method reads the white spaces and returns the zero character (\0) if it has reached the end of the stream. If not, we look into what comes next in the stream by calling `peek`, and return its resulting value. Note that `peek` does not consume the character from the stream; it just checks out the next character:

[PRE29]

The `ReadI` method verifies whether the next character in the stream is **i** or **I**. If it is, it reads the character from the stream and returns `true`:

[PRE30]

The `ReadSign` method verifies that the next character in the stream is a plus or minus sign. If it is, it reads the character from the stream, sets the sign parameter to **+** or **-**, and returns `true`:

[PRE31]

The `ReadValue` method verifies that the next two characters in the stream are a plus or a minus sign followed by a digit or a dot, or whether the first character is a digit or a dot. If the latter is the case, it reads the `value` parameter from the beginning of the stream and returns `true`:

[PRE32]

The `EndOfLine` method simply returns `true` if the next character in the stream is the zero character (\0), in which case we have reached the end of the string:

[PRE33]

Now we are ready to interpret a string as a rational number. We have the following ten cases, where *x* and *y* are real values, *i* is the imaginary unit, and ± is plus or minus. All ten cases represent valid complex numbers:

1.  *x* ± *yi*
2.  *x* ± ±*i*
3.  *x* ± *i*
4.  *yi* ± *x*
5.  ±*i* ± *x*
6.  *i* ± *x*
7.  *yi*
8.  ±*i*
9.  *i*
10.  *x*

The `ReadStream` method creates an input stream from the text and tries to interpret it as one of the preceding ten cases. The idea is that we read the stream and try one part of the potential complex number at a time:

[PRE34]

If the stream is made up of a value, a sign, another value, and i or I, we set *x* and *y* in accordance with case 1 (*x* ± *yi*) and return `true`. The *y* field is negative if the sign is minus. However, the second value may also be negative, in which case *y*is positive:

[PRE35]

If the sign is not followed by a value, but by another sign and i or I, case 2 (*x* ± ±*i*) applies and we return `true`. In this case, we actually have to adjust the value of *y* twice in accordance with both signs:

[PRE36]

If the sign is not followed by a value or another sign, but by i or I, case 3 (*x* ± *i*) applies and we return `true`:

[PRE37]

If the value is not followed by a sign but by i or I, another sign, and another value, case 4 (*yi* ± *x*) applies and we return `true`:

[PRE38]

If the value is followed by i or I and nothing else, case 7 (*yi*) applies and we return `true`:

[PRE39]

If the value is followed by nothing else, case 10 (*x*) applies and we return `true`:

[PRE40]

If the stream does not start with a value, but with a sign followed by i or I, another sign and another value, case 5 (±*i* ± *x*) applies and we return `true`:

[PRE41]

If the stream starts with a sign followed by i or I and nothing else, case 8 (±*i*) applies and we return `true`:

[PRE42]

If the stream does not start with a value or a sign, but with i or I followed by a sign and a value, case 6 (*i* ± *x*) applies and we return `true`:

[PRE43]

If the stream is made up by i or I and nothing else, case 9 (*i*) applies and we return `true`:

[PRE44]

Finally, if none of the above cases apply, the text does not hold a complex number and we return `false`:

[PRE45]

The constructor that takes a text simply calls `ReadStream` and throws a `NotaComplexNumber` exception if `ReadStream` returns `false`. However, if `ReadStream` returns `true`, *x* and *y* are set to the appropriate values:

[PRE46]

In the `String` conversion operator, we look into several different cases:

1.  *x* + *i*
2.  *x* - *i*
3.  *x* ± *i*
4.  *x*
5.  +*i*
6.  -*i*
7.  *yi*
8.  0

If the real part *x* is not 0, we write its value on the stream and look into the first four cases with regard to the imaginary part, *y*. If *y* is plus or minus 1, we simply write `+i` or `-i`. If it is not plus or minus 1, and not 0, we write its value with the `showpos` flag, which forces the plus sign to be present in the case of a positive value. Finally, if *y* is 0, we do not write it at all:

[PRE47]

If *x* is zero, we omit it and write the value of *y* in the same manner as we did earlier. However, if *y* is zero, we write 0; otherwise, nothing will be written if both *x* and *y* are 0\. Moreover, we omit the `showpos` flag, since it is not necessary to write the plus sign in the case of a positive value:

[PRE48]

Two complex numbers are equal if their real and imaginary parts are equal:

[PRE49]

When deciding whether a complex number is smaller than another complex number, we chose to compare their absolute values, which is given by the `Abs` method:

[PRE50]

The addition operators all call the following final operator, which works for all four arithmetic operators:

[PRE51]

When adding two complex numbers, we add the real and imaginary parts separately:

[PRE52]

When subtracting two complex numbers, we subtract the real and imaginary parts separately:

[PRE53]

The product of two complex numbers can be established by some algebra:

(*x* [1] + *y* [1]*i*)(*x* [2] + *y[2]i*) = *x* [1]*x* [2] + *x* [1]*y* [2]*i* + *y* [1]*ix* [2] + *y* [1]*y* [2]*i* ² = *x* [1]*x* [2] + *x* [1]*y* [2]*i* + *y* [1]*ix* [2] *+ y* [1]*y* [2] (-1) = *x* [1]*x* [2] + *x* [1]*y* [2]*i* + *x* [2]*y* [1]*i* - *y* [1]*y* [2] = (*x* [1]*x* [2] - *y* [1]*y* [2]) + (*x* [1]*y* [2] + *x* [2]*y* [1])*i*

[PRE54]

The quotient between two complex numbers can also be established by some algebra. The **conjugate** of a complex number *x* [2] + *y* [2]*i* is *x* [2] - *y* [2]*i*, which we can use in the conjugate rule:

(*x* [2]+ *y* [2]*i*)(*x* [2] - *y* [2]*i*) = *x* [2]² - *x* [2]*y* [2]*i* + *x* [2]*y* [2]*i* - *y* [2]² (-1) = *x* [2]² - *x* [2]*y* [2]*i* + *x* [2]*y* [2]*i* + *y* [2]² = *x* [2]² + *y* [2]²

We can use the conjugate rule when dividing two complex numbers by multiplying the conjugate by both the numerator and the denominator:

![Complex numbers](img/B05475_Appendix_08.jpg)

[PRE55]

# Summary

By reading this book you have learned how to develop applications in Windows with Small Windows, a C++ object-oriented class library for graphical applications in Windows. I hope you have enjoyed the book!