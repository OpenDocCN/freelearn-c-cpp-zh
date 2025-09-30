# Exploring Functions

Whenever you need to create a large application, it is a wise decision to divide it into manageable chunks, called **functions**. Functions are small modules that represent tasks that can be executed independently. The code written inside a function can be invoked several times, which helps to avoid repetitive statements.

Functions help in the teamwork, debugging, and scaling of any application. Whenever you want to add more features to an application, simply add a few functions to it. When calling functions, the caller function may pass certain arguments, called **actual arguments**; these are then assigned to the parameters of the function. The parameters are also known as formal parameters.

The following recipes will help you understand how functions can be used to make complex applications easier and more manageable. Normally, a function can return only a single value. But in this chapter, we will learn a technique to return more than one value from a function. We will also learn how to apply recursion in functions.

In this chapter, we will cover the following recipes on strings:

*   Determining whether a number is an Armstrong number
*   Returning the maximum and minimum values of an array
*   Finding GCD using recursion
*   Converting a binary number into a hexadecimal number
*   Determining whether a number is a palindrome

As I will be using a stack structure in the recipes in this chapter, let's have a quick introduction to stack.

# What is a stack?

A Stack is a structure that can be implemented with arrays as well as linked lists. It is a sort of a bucket where the value you enter will be added at the bottom. The next item that you add to a stack will be kept just above the item that was added previously. The procedure of adding a value to the stack is called a `push` operation and the procedure of getting a value out of the stack is called a `pop` operation. The location where the value can be added or taken out of the stack is pointed at by a pointer called **top**. The value of the **top** pointer is **-1** when the stack is empty:

![](img/34321603-bd0c-4774-bf42-a057d40dd21e.png)

Figure 3.1

When the `push` operation is executed, the value of **top** is incremented by **1**, so that it can point to the location in the stack where the value can be pushed:

![](img/d24cf70f-618e-4bba-a2a9-53c433dbf139.png)

Figure 3.2

Now, the next value that will be pushed will be kept above value 1\. More precisely, the value of the **top** pointer will be incremented by **1**, making its value 1, and the next value will be pushed to the **stack[1]** location, as follows:

![](img/32a351fe-0e7f-4289-86ca-a77cddc573b3.png)

Figure 3.3

So, you can see that the stack is a **Last In First Out** (**LIFO**) structure; that is, the last value that was pushed sits at the top.

Now, when we execute a `pop` operation, the value at the top, that is, value **2**, will be popped out first, followed by the popping out of value **1**. Basically, in the `pop` operation, the value pointed at by the **top** pointer is taken out, and then the value of **top** is decremented by 1 so that it can point at the next value to be popped out.

Now, that we've understood stacks, let's begin with the first recipe.

# Finding whether a number is an Armstrong number

An Armstrong number is a three-digit integer that is the sum of the cubes of its digits. This simply means that if *xyz = x³+y³+z³*, it is an Armstrong number. For example, 153 is an Armstrong number because *1³+5³+3³ = 153*.

Similarly, a number that comprises four digits is an Armstrong number if the sum of its digits raised to the power of four results in that number. For example, *pqrs = p⁴+q ⁴+r ⁴+s⁴.*

# How to do it…

1.  Enter a number to assign to the `n` variable:

```cpp
printf("Enter a number ");
scanf("%d",&n);
```

2.  Invoke the `findarmstrong` function. The value assigned to `n` will get passed to this function:

```cpp
findarmstrong(n)
```

3.  In the function, the passed argument, n, is assigned to the `numb` parameter. Execute a `while` loop to separate out all the digits in the `numb` parameter:

```cpp
while(numb >0)
```

4.  In the `while` loop, apply the mod 10 (`%10`) operator on the number assigned to the `numb` variable. The mod operator divides a number and returns the remainder:

```cpp
remainder=numb%10;
```

5.  Push the remainder to the stack:

```cpp
push(remainder);
```

6.  Remove the last digit of the number in the `numb` variable by dividing the `numb` variable by `10`:

```cpp
numb=numb/10;
```

7.  Repeat steps 4 to 6 until the number in the `numb` variable becomes 0\. In addition, create a `count` counter to count the number of digits in the number. Initialize the counter to `0` and it will get incremented during the `while` loop:

```cpp
count++;
```

8.  Pop all the digits from the stack and raise it to the given power. To pop all the digits from the stack, execute a `while` loop that will execute until `top` is greater than or equal to `0`, that is, until the stack is empty:

```cpp
while(top >=0)
```

9.  Inside the `while` loop, pop off a digit from the stack and raise it to the power of `count`, which is the count of the number of digits in the selected number. Then, add all the digits to the `value`:

```cpp
j=pop();
value=value+pow(j,count);
```

10.  Compare the number in the `value` variable with the number in the `numb` variable, and code it to return the value of `1` if both the compared numbers match:

```cpp
if(value==numb)return 1;
```

If the numbers in the `numb` and `value` variables are the same, returning the Boolean value of `1`, that means the number is an Armstrong number. 

Here is the `armstrong.c` program for finding out whether the specified number is an Armstrong number:

```cpp
/* Finding whether the entered number is an Armstrong number */
# include <stdio.h>
# include <math.h>

#define max 10

int top=-1;
int stack[max];
void push(int);
int pop();
int findarmstrong(int );
void main()
{
   int n;
   printf("Enter a number ");
   scanf("%d",&n);
   if (findarmstrong(n))
      printf("%d is an armstrong number",n);
   else printf("%d is not an armstrong number", n);
}
int findarmstrong(int numb)
{
   int j, remainder, temp,count,value;
   temp=numb;
   count=0;
   while(numb >0)
   {
      remainder=numb%10;
      push(remainder);
      count++;
      numb=numb/10;
   }
   numb=temp;
   value=0;
   while(top >=0)
   {
      j=pop();
      value=value+pow(j,count);
   }
   if(value==numb)return 1;
   else return 0;
}
void push(int m)
{
   top++;
   stack[top]=m;
}
int pop()
{
   int j;
   if(top==-1)return(top);
   else
   {
      j=stack[top];
      top--;
      return(j);
   }
}
```

Now, let's go behind the scenes.

# How it works...

First, we will apply the mod **10** operator to separate our digits. Assuming the number entered by us is **153**, you can see that **153** is divided by **10** and the remaining **3** is pushed to the stack:

![](img/a36dff84-2a1a-40e4-9f6d-626f46b204c4.png)

Figure 3.4

The value in the stack is pushed at the index location indicated by **top**. Initially, the value of **top** is -1. It is so because before the `push` operation, the value of **top** is incremented by 1, and the array is zero-based, that is, the first element in the array is placed at the 0 index location. Consequently, the value of **top** has to be initialized to -1\. As mentioned, the value of **top** is incremented by 1 before pushing, that is, the value of **top** will become **0**, and the remainder of **3** is pushed to **stack[0]**.

In the stack, the value of `top` is incremented by 1 to indicate the location in the stack where the value will be pushed.

We will again apply the mod **10** operator to the **15** quotient. The remainder that we will get is **5**, which will be pushed to the stack. Again, before pushing to the stack, the value of **top**, which was 0, is incremented to 1\. At **stack[1]**, the remainder is pushed:

![](img/576b0f6b-b356-4f46-af8a-6b3b92fb5aec.png)

Figure 3.5

To the **1** quotient, we will again apply the mod **10** operator. But because 1 is not divisible by **10**, **1** itself will be considered as the remainder and will be pushed to the stack. The value of **top** will again be incremented by 1 and **1** will be pushed to **stack[2]**:

![](img/cdc5263b-c2ba-469e-a96c-2ba772e27adc.png)

Figure 3.6

Once all the digits are separated and placed in the stack, we will pop them out one by one. Then, we will raise each digit to the power equal to the count of the digits. Because the number **153** consists of three digits, each digit is raised to the power of **3**.

While popping values out of the stack, the value indicated by the **top** pointer is popped out. The value of **top** is **2**, hence the value at **stack[2]**, that is, **1**, is popped out and raised to the power of **3**, as follows:

![](img/9be5fb2b-60c5-499f-b37e-b3efae996ce6.png)

Figure 3.7

After the `pop` operation, the value of **top** will be decremented to 1 to indicate the next location to be popped out. Next, the value at **stack[1]** will be popped out and raised to the power of **3**. We will then add this value to our previous popped-out one:

![](img/8a08cff2-e263-4875-bf5a-d251d7881b1a.png)

Figure 3.8

After the popping-out operation, the value of **top** is decremented by 1, now making its value **0**. So, the value at **stack[0]** is popped out and raised to the power of **3**. The result is added to our earlier computation:

![](img/3bf83d80-13ee-478b-bfe3-44f7a8db459f.png)

Figure 3.9

The result after computing **1³ + 5³ + 3³** is **153**, which is the same as the original number. This proves that **153** is an Armstrong number.

Let's use GCC to compile the `armstrong.c` program, as follows:

```cpp
D:\CBook>gcc armstrong.c -o armstrong 
```

Let's check whether `127` is an Armstrong number:

```cpp
D:\CBook>./armstrong
Enter a number 127
127 is not an armstrong number
```

Let's check whether `153` is an Armstrong number:

```cpp
D:\CBook>./armstrong
Enter a number 153
153 is an armstrong number
```

Let's check whether `1634` is an Armstrong number:

```cpp
D:\CBook>./armstrong
Enter a number 1634
1634 is an armstrong number
```

Voilà! We've successfully made a function to find whether a specified number is an Armstrong number or not.

Now, let's move on to the next recipe!

# Returning maximum and minimum values in an array

C functions cannot return more than one value. But what if you want a function to return more than one value? The solution is to store the values to be returned in an array and make the function return the array instead.

In this recipe, we will make a function return two values, the maximum and minimum values, and store them in another array. Thereafter, the array containing the maximum and minimum values will be returned from the function.

# How to do it…

1.  The size of the array whose maximum and minimum values have to be found out is not fixed, hence we will define a macro called `max` of size `100`:

```cpp
#define max 100
```

2.  We will define an `arr` array of the max size, that is, `100` elements:

```cpp
int arr[max];
```

3.  You will be prompted to specify the number of elements in the array; the length you enter will be assigned to the `n` variable:

```cpp
printf("How many values? ");
scanf("%d",&n);
```

4.  Execute a `for` loop for `n` number of times to accept `n` values for the `arr` array:

```cpp
for(i=0;i<n;i++)                                
    scanf("%d",&arr[i]);
```

5.  Invoke the `maxmin` function to pass the `arr` array and its length, `n`, to it. The array that will be returned by the `maxmin` function will be assigned to the integer pointer, `*p`:

```cpp
p=maxmin(arr,n);
```

6.  When you look at the function definition, `int *maxmin(int ar[], int v){ }`, the `arr` and `n` arguments passed to the `maxmin` function are assigned to the `ar` and `v` parameters, respectively. In the `maxmin` function, define an `mm` array of two elements:

```cpp
static int mm[2];
```

7.  To compare it with the rest of the elements, the first element of `ar` array is stored at `mm[0]` and `mm[1]`. A loop is executed from the `1` value till the end of the length of the array and within the loop, the following two formulas are applied:

*   We will use `mm[0]` to store the minimum value of the `arr` array. The value in `mm[0]` is compared with the rest of the elements. If the value in `mm[0]` is greater than any of the array elements, we will assign the smaller element to `mm[0]`:

```cpp
if(mm[0] > ar[i])
    mm[0]=ar[i];
```

*   We will use `mm[1]` to store the maximum value of the `arr` array. If the value at `mm[1]` is found to be smaller than any of the rest of the array element, we will assign the larger array element to `mm[1]`:

```cpp
if(mm[1]< ar[i])
    mm[1]= ar[i];
```

8.  After the execution of the `for` loop, the `mm` array will have the minimum and maximum values of the `arr` array at `mm[0]` and `mm[1]`, respectively. We will return this `mm` array to the `main` function where the `*p` pointer is set to point at the returned array, `mm`:

```cpp
return mm;
```

9.  The `*p` pointer will first point to the memory address of the first index location, that is, `mm[0]`. Then, the content of that memory address, that is, the minimum value of the array, is displayed. After that, the value of the `*p` pointer is incremented by 1 to make it point to the memory address of the next element in the array, that is, the `mm[1]` location:

```cpp
printf("Minimum value is %d\n",*p++);
```

10.  The `mm[1]` index location contains the maximum value of the array. Finally, the maximum value pointed to by the `*p` pointer is displayed on the screen:

```cpp
printf("Maximum value is %d\n",*p);
```

The `returnarray.c` program explains how an array can be returned from a function. Basically, the program returns the minimum and maximum values of an array:

```cpp
/* Find out the maximum and minimum values using a function returning an array */
# include <stdio.h>
#define max 100
int *maxmin(int ar[], int v);
void main()
{
    int  arr[max];
    int n,i, *p;
    printf("How many values? ");
    scanf("%d",&n);
    printf("Enter %d values\n", n);
    for(i=0;i<n;i++)
        scanf("%d",&arr[i]);
    p=maxmin(arr,n);
    printf("Minimum value is %d\n",*p++);
    printf("Maximum value is %d\n",*p);
}
int *maxmin(int ar[], int v)
{
    int i;
    static int mm[2];
    mm[0]=ar[0];
    mm[1]=ar[0];
    for (i=1;i<v;i++)
    {
        if(mm[0] > ar[i])
            mm[0]=ar[i];
        if(mm[1]< ar[i])
            mm[1]= ar[i];
    }
    return mm;
}
```

Now, let's go behind the scenes.

# How it works...

We will use two arrays in this recipe. The first array will contain the values from which the maximum and minimum values have to be found. The second array will be used to store the minimum and maximum values of the first array.

Let's call the first array **arr** and define it to contain five elements with the following values:

![](img/3a4a75ea-5c7d-4df6-aea0-22b4450bdc1a.png)

Figure 3.10

Let's call our second array **mm**. The first location, **mm[0]**, of the **mm** array will be used for storing the minimum value and the second location, **mm[1]**, for storing the maximum value of the **arr** array. To enable comparison of the elements of the **mm** array with the elements of the **arr** array, copy the first element of the **arr** array at **arr[0]** to both **mm[0]** and **mm[1]**:

![](img/b2a64a98-5764-4b94-b97f-1d44d6889b35.png)

Figure 3.11

Now, we will compare the rest of the elements of the **arr** array with **mm[0]** and **mm[1]**. To keep the minimum value at **mm[0]**, any element smaller than the value at **mm[0]** will be assigned to **mm[0]**. Values larger than **mm[0]** are simply ignored. For example, the value at **arr[1]** is smaller than that at **mm[0]**, that is, 8 < 30\. So, the smaller value will be assigned to **mm[0]**:

![](img/01805e53-5780-46be-8208-23cff08a12e9.png)

Figure 3.12

We will apply reverse logic to the element at **mm[1]**. Because we want the maximum value of the **arr** array at **mm[1]**, any element found larger than the value at **mm[1]** will be assigned to **mm[1]**. All smaller values will be simply ignored. 

We will continue this process with the next element in the **arr** array, which is **arr[2].** Because 77 > 8, it will be ignored when compared with **mm[0]**. But 77 > 30, so it will be assigned to **mm[1]**:

![](img/90d201b4-b466-44a1-bf6d-247e6ffc3a73.png)

Figure 3.13

We will repeat this procedure with the rest of the elements of the **arr** array. Once all the elements of the **arr** array are compared with both the elements of the **mm** array, we will have the minimum and maximum values at **mm[0]** and **mm[1]**, respectively:

![](img/33b2b70c-8899-4f6a-a0e5-0fcbc98cedc2.png)

Figure 3.14

Let's use GCC to compile the `returnarray.c` program, as follows:

```cpp
D:\CBook>gcc returnarray.c -o returnarray
```

Here is the output of the program:

```cpp
D:\CBook>./returnarray
How many values? 5
Enter 5 values
30
8
77
15
9
Minimum value is 8
Maximum value is 77
```

Voilà! We've successfully returned the maximum and minimum values in an array.

Now, let's move on to the next recipe!

# Finding the greatest common divisor using recursion

In this recipe, we will use recursive functions to find the **greatest common divisor** (**GCD)**, also known as the highest common factor) of two or more integers. The GCD is the largest positive integer that divides each of the integers. For example, the GCD of 8 and 12 is 4, and the GCD of 9 and 18 is 9.

# How to do it…

The `int gcd(int x, int y)` recursive function finds the GCD of two integers, x and y, using the following three rules:

*   If y=0, the GCD of `x` and `y` is `x`.
*   If x mod y is 0, the GCD of x and y is y.
*   Otherwise, the GCD of x and y is `gcd(y, (x mod y))`.

Follow the given steps to find the GCD of two integers recursively:

1.  You will be prompted to enter two integers. Assign the integers entered to two variables, `u` and `v`:

```cpp
printf("Enter two numbers: ");
scanf("%d %d",&x,&y);
```

2.  Invoke the `gcd` function and pass the `x` and `y` values to it. The `x` and `y` values will be assigned to the `a` and `b` parameters, respectively. Assign the GCD value returned by the `gcd` function to the `g` variable:

```cpp
g=gcd(x,y);
```

3.  In the `gcd` function, `a % b` is executed. The `%` (mod) operator divides the number and returns the remainder:

```cpp
m=a%b;
```

4.  If the remainder is non-zero, call the `gcd` function again, but this time the arguments will be `gcd(b,a % b)`, that is, `gcd(b,m)`, where `m` stands for the mod operation:

```cpp
gcd(b,m);
```

5.  If this again results in a non-zero remainder, that is, if `b % m` is non-zero, repeat the `gcd` function with the new values obtained from the previous execution:

```cpp
gcd(b,m);
```

6.  If the result of `b % m` is zero, `b` is the GCD of the supplied arguments and is returned back to the `main` function:

```cpp
return(b);
```

7.  The result, `b`, that is returned back to the `main` function is assigned to the `g` variable, which is then displayed on the screen:

```cpp
printf("Greatest Common Divisor of %d and %d is %d",x,y,g);
```

The `gcd.c` program explains how the greatest common divisor of two integers is computed through the recursive function:

```cpp
#include <stdio.h>
int gcd(int p, int q);
void main()
{
    int x,y,g;
    printf("Enter two numbers: ");
    scanf("%d %d",&x,&y);
    g=gcd(x,y);
    printf("Greatest Common Divisor of %d and %d is %d",x,y,g);
}
int gcd(int a, int b)
{
    int m;
    m=a%b;
    if(m==0)
        return(b);
    else
        gcd(b,m);
}
```

Now, let's go behind the scenes.

# How it works...

Let's assume we want to find the GCD of two integers, **18** and **24**. To do so, we will invoke the `gcd(x,y)` function, which in this case is `gcd(18,24)`. Because **24**, that is, y, is not zero, Rule 1 is not applicable here. Next, we will use Rule 2 to check whether `18%24` (`x % y`) is equal to **0**. Because **18** cannot be divided by **24**, **18** will be the remainder:

![](img/7b79b3f0-b216-4b53-b651-307af39db939.png)

Figure 3.15

Since the parameters of Rule 2 were also not met, we will use Rule 3\. We will invoke the `gcd` function with the `gcd(b,m)` argument, which is `gcd(24,18%24)`. Now, m stands for the mod operation. At this stage, we will again apply Rule 2 and collect the remainder:

![](img/ed70fd9d-5a89-4a58-8cf0-9ba032fc5057.png)

Figure 3.16

Because the result of `24%18` is a non-zero value, we will invoke the `gcd` function again with the `gcd(b, m)` argument, which is now `gcd(18, 24%18)`, since we were left with **18** and **6** from the previous execution. We will again apply Rule 2 to this execution. When **18** is divided by **6**, the remainder is **0**:

![](img/1c063c94-df3c-496d-a531-63ec718a1bec.png)

Figure 3.17

At this stage, we have finally fulfilled the requirements of one of the rules, Rule 2\. If you recall, Rule 2 says that if x mod y is **0**, the GCD is y. Because the result of **18** mod **6** is **0**, the GCD of **18** and **24** is **6**.

Let's use GCC to compile the `gcd.c` program, as follows:

```cpp
D:\CBook>gcc gcd.c -o gcd
```

Here is the output of the program:

```cpp
D:\CBook>./gcd
Enter two numbers: 18 24
Greatest Common Divisor of 18 and 24 is 6
D:\CBook>./gcd
Enter two numbers: 9 27
Greatest Common Divisor of 9 and 27 is 9
```

Voilà! We've successfully found the GCD using recursion.

Now, let's move on to the next recipe!

# Converting a binary number into a hexadecimal number

In this recipe, we will learn how to convert a binary number into a hexadecimal number. A binary number comprises two bits, 0 and 1\. To convert a binary number into a hexadecimal number, we first need to convert the binary number into a decimal number and then convert the resulting decimal number to hexadecimal.

# How to do it…

1.  Enter a binary number and assign it to the `b` variable:

```cpp
printf("Enter a number in binary number ");
scanf("%d",&b);
```

2.  Invoke the `intodecimal` function to convert the binary number into a decimal number, and pass the `b` variable to it as an argument. Assign the decimal number returned by the `intodecimal` function to the `d` variable:

```cpp
d=intodecimal(b);
```

3.  On looking at the `intodecimal` definition, `int intodecimal(int bin) { }`, we can see that the `b` argument is assigned to the `bin` parameter of the `intodecimal` function.
4.  Separate all the binary digits and multiply them by 2 raised to the power of their position in the binary number. Sum the results to get the decimal equivalent. To separate each binary digit, we need to execute a `while` loop until the binary number is greater than `0`:

```cpp
while(bin >0)
```

5.  Within the `while` loop, apply the mod 10 operator on the binary number and push the remainder to the stack:

```cpp
remainder=bin%10;
push(remainder);
```

6.  Execute another `while` loop to get the decimal number of all the binary digits from the stack. The `while` loop will execute until the stack becomes empty (that is, until the value of `top` is greater than or equal to `0`):

```cpp
while(top >=0)
```

7.  In the `while` loop, pop off all the binary digits from the stack and multiply each one by `2` raised to the power of `top`. Sum the results to get the decimal equivalent of the entered binary number:

```cpp
j=pop();
deci=deci+j*pow(2,exp);
```

8.  Invoke the `intohexa` function and pass the binary number and the decimal number to it to get the hexadecimal number:

```cpp
void intohexa(int bin, int deci)
```

9.  Apply the mod `16` operator in the `intohexa` function on the decimal number to get its hexadecimal. Push the remainder that you get to the stack. Apply mod `16` to the quotient again and repeat the process until the quotient becomes smaller than `16`:

```cpp
remainder=deci%16;
push(remainder);
```

10.  Pop off the remainders that are pushed to the stack to display the hexadecimal number:

```cpp
j=pop();
```

If the remainder that is popped off from the stack is less than 10, it is displayed as such. Otherwise, it is converted to its equivalent letter, as mentioned in the following table, and the resulting letter is displayed:

| **Decimal** | **Hexadecimal** |
| 10 | A |
| 11 | B |
| 12 | C |
| 13 | D |
| 14 | E |
| 15 | F |

```cpp
if(j<10)printf("%d",j);
else printf("%c",prnhexa(j));
```

The `binarytohexa.c` program explains how a binary number can be converted into a hexadecimal number:

```cpp
//Converting binary to hex
# include <stdio.h>
#include  <math.h>
#define max 10
int top=-1;
int stack[max];
void push();
int pop();
char prnhexa(int);
int intodecimal(int);
void intohexa(int, int);
void main()
{
    int b,d;
    printf("Enter a number in binary number ");
    scanf("%d",&b);
    d=intodecimal(b);
    printf("The decimal of binary number %d is %d\n", b, d);
    intohexa(b,d);
}
int intodecimal(int bin)
{
    int deci, remainder,exp,j;
    while(bin >0)
    {
        remainder=bin%10;
        push(remainder);
        bin=bin/10;
    }
    deci=0;
    exp=top;
    while(top >=0)
    {
        j=pop();
        deci=deci+j*pow(2,exp);
        exp--;
    }
    return (deci);
}
void intohexa(int bin, int deci)
{
    int remainder,j;
    while(deci >0)
    {
        remainder=deci%16;
        push(remainder);
        deci=deci/16;
    }
    printf("The hexa decimal format of binary number %d is ",bin);
    while(top >=0)
    {
        j=pop();
        if(j<10)printf("%d",j);
        else printf("%c",prnhexa(j));
    }
}
void push(int m)
{
    top++;
    stack[top]=m;
}
int pop()
{
    int j;
    if(top==-1)return(top);
    j=stack[top];
    top--;
    return(j);
}
char prnhexa(int v)
{
    switch(v)
    {
        case 10: return ('A');
                 break;
        case 11: return ('B');
                 break;
        case 12: return ('C');
                 break;
        case 13: return ('D');
                 break;
        case 14: return ('E');
                 break;
        case 15: return ('F');
                 break;
    }
}
```

Now, let's go behind the scenes.

# How it works...

The first step is to convert the binary number into a decimal number. To do so, we will separate all the binary digits and multiply each by **2** raised to the power of their position in the binary number. We will then apply the mod **10** operator in order to separate the binary number into individual digits. Every time mod **10** is applied to the binary number, its last digit is separated and then pushed to the stack.

Let's assume that the binary number that we need to convert into a hexadecimal format is **110001**. We will apply the mod **10** operator to this binary number. The mod operator divides the number and returns the remainder. On application of the mod **10** operator, the last binary digit—in other words the rightmost digit will be returned as the remainder (as is the case with all divisions by **10**). 

The operation is pushed in the stack at the location indicated by the **top** pointer. The value of **top** is initially -1\. Before pushing to the stack, the value of **top** is incremented by 1\. So, the value of **top** increments to 0 and the binary digit that appeared as the remainder (in this case, 1) is pushed to **stack[0]** (see *Figure 3.18*), and**11000** is returned as the quotient:

![](img/41c50e5b-0332-4a43-a71c-9ea179a153b2.png)

Figure 3.18

We will again apply the mod **10** operator to the quotient to separate the last digit of the present binary number. This time, **0** will be returned as the remainder and **1100** as the quotient on the application of the mod **10** operator. The remainder is again pushed to the stack. As mentioned before, the value of **top** is incremented before applying the `push` operation. As the value of **top** was **0**, it is incremented to **1** and our new remainder, **0**, is pushed to **stack[1]**:

![](img/3b430279-777e-44bb-8776-3919ec581aad.png)

Figure 3.19

We will repeat this procedure until all the digits of the binary number are separated and pushed to the stack, as follows:

![](img/2f060848-a1c9-4c71-93bd-9777de3ab8fe.png)

Figure 3.20

Once that's done, the next step is to pop the digits out one by one and multiply every digit by **2** raised to the power of **top**. For example, **2** raised to the power of top means **2** will be raised to the value of the index location from where the binary digit was popped off. The value from the stack is popped out from the location indicated by **top**.

The value of **top** is currently **5**, hence the element at **stack[5]** will be popped out and multiplied by **2** raised to the power **5**, as follows:

![](img/121a2a0e-4474-4fde-87ea-005350876818.png)

Figure 3.21

After popping a value from the stack, the value of **top** is decremented by 1 to point at the next element to be popped out. The procedure is repeated until every digit is popped out and multiplied by **2** raised to the power of its top location value. *Figure 3.19* shows how all the binary digits are popped from the stack and multiplied by **2** raised to the power of **top**:

![](img/ec5d96f3-ae58-493d-a0ba-9954d54cdb2c.png)

Figure 3.22

The resulting number we get is the decimal equivalent of the binary number that was entered by the user.

Now, to convert a decimal number into a hexadecimal format, we will divide it by 16\. We need to keep dividing the number until the quotient becomes smaller than `16`. The remainders of the division are displayed in LIFO order. If the remainder is below 10, it is displayed as is; otherwise, its equivalent letter is displayed. You can use the preceding table to find the equivalent letter if you get a remainder between 10 and 15.

In the following figure, you can see that the decimal number **49** is divided by **16**. The remainders are displayed in LIFO order to display the hexadecimal, hence 31 is the hexadecimal of the binary number **110001**. You don’t need to apply the preceding table as both the remainders are less than 10:

![](img/e08ed714-487a-4d7d-ae17-f20202ade8e9.png)

Figure 3.23

Let's use GCC to compile the `binaryintohexa.c` program, as follows:

```cpp
D:\CBook>gcc binaryintohexa.c -o binaryintohexa
```

Here is one output of the program:

```cpp
D:\CBook>./binaryintohexa
Enter a number in binary number 110001
The decimal of binary number 110001 is 49
The hexa decimal format of binary number 110001 is 31
```

Here is another output of the program:

```cpp
D:\CBook>./binaryintohexa
Enter a number in binary number 11100
The decimal of binary number 11100 is 28
The hexa decimal format of binary number 11100 is 1C
```

Voilà! We've successfully converted a binary number into a hexadecimal number.

Now, let's move on to the next recipe!

# Finding whether a number is a palindrome 

A palindrome number is one that appears the same when read forward and backward. For example, 123 is not a palindrome but 737 is. To find out whether a number is a palindrome, we need to split it into separate digits and convert the unit of the original number to hundred and the hundred to unit.

For example, a `pqr` number will be called a **palindrome** **number** if `pqr=rqp`. And `pqr` will be equal to `rqp` only if the following is true:

*p x 100 + q x 10 + r = r x 100 + q x 10 + p*

In other words, we will have to multiply the digit in the unit place by 10² to convert it into the hundreds and convert the digit in the hundreds place to unit by multiplying it by 1\. If the result matches the original number, it is a palindrome.

# How to do it…

1.  Enter a number to assign to the `n` variable:

```cpp
printf("Enter a number ");
scanf("%d",&n);
```

2.  Invoke the `findpalindrome` function and pass the number in the `n` variable to it as an argument:

```cpp
findpalindrome(n)
```

3.  The `n` argument is assigned to the `numb` parameter in the `findpalindrome` function. We need to separate each digit of the number; to do so, we will execute a `while` loop for the time the value in the `numb` variable is greater than `0`:

```cpp
while(numb >0)
```

4.  Within the `while` loop, we will apply mod 10 on the number. On application of the mod `10` operator, we will get the remainder, which is basically the last digit of the number:

```cpp
remainder=numb%10;
```

5.  Push that remainder to the stack:

```cpp
push(remainder);
```

6.  Because the last digit of the number is separated, we need to remove the last digit from the existing number. That is done by dividing the number by 10 and truncating the fraction. The `while` loop will terminate when the number is individually divided into separate digits and all the digits are pushed to the stack:

```cpp
numb=numb/10;
```

7.  The number at the top of the stack will be the hundred and the one at the bottom of the stack will be the unit of the original number. Recall that we need to convert the hundred of the original number to the unit and vice versa. Pop all the digits out from the stack one by one and multiply each of them by `10` raised to a power. The power will be 0 for the first digit that is popped off. The power will be incremented with every value that is popped off. After being multiplied by `10` raised to the respective power, the digits are added into a separate variable, called `value`:

```cpp
j=pop();
value=value+j*pow(10,count);
count++;
```

8.  If the numbers in the `numb` and `value` variables match, that means the number is a palindrome. If the number is a palindrome, the `findpalindrome` function will return a value of `1`, otherwise it will return a value of `0`:

```cpp
if(numb==value) return (1);
else return (0);
```

The `findpalindrome.c` program determines whether the entered number is a palindrome number:

```cpp
//Find out whether the entered number is a palindrome or not
# include <stdio.h>
#include <math.h>
#define max 10
int top=-1;
int stack[max];
void push();
int pop();
int findpalindrome(int);
void main()
{
    int n;
    printf("Enter a number ");
    scanf("%d",&n);   
    if(findpalindrome(n))
        printf("%d is a palindrome number",n);
    else
        printf("%d is not a palindrome number", n);
}
int findpalindrome(int numb)
{
    int j, value, remainder, temp,count;
    temp=numb;
    while(numb >0)
    {
        remainder=numb%10;
        push(remainder);
        numb=numb/10;
    }
    numb=temp;
    count=0;
    value=0;
    while(top >=0)
    {
        j=pop();
        value=value+j*pow(10,count);
        count++;
    }
    if(numb==value) return (1);
    else return (0);
}
void push(int m)
{
    top++;
    stack[top]=m;
}
int pop()
{
    int j;
    if(top==-1)return(top);
    else
    {
        j=stack[top];
        top--;
        return(j);
   }
}
```

Now, let's go behind the scenes.

# How it works...

Let's assume that the number we entered is **737**. Now, we want to know whether **737** is a palindrome. We will start by applying the mod **10** operator on **737**. On application, we will receive the remainder, **7**, and the quotient, **73**. The remainder, **7**, will be pushed to the stack. Before pushing to the stack, however, the value of the **top** pointer is incremented by 1\. The value of **top** is -1 initially; it is incremented to **0** and the remainder of **7** is pushed to **stack[0]** (see *Figure 3.21* ).

The mod **10** operator returns the last digit of the number as the remainder. The quotient that we get on the application of the mod **10** operator is the original number with its last digit removed. That is, the quotient that we will get on the application of mod **10** operator on **737** is **73**:

![](img/bc80e6f2-d3d2-4bf0-8e3c-92e0928abec1.png)

Figure 3.24

To the quotient, **73**, we will apply the mod **10** operator again. The remainder will be the last digit, which is **3**, and the quotient will be **7**. The value of **top** is incremented by 1, making its value 1, and the remainder is pushed to **stack[1]**. To the quotient, **7**, we will again apply the mod **10** operator. Because **7** cannot be divided by **10**, **7** itself is returned and is pushed to the stack. Again, before the `push` operation, the value of **top** is incremented by 1, making its value **2**. The value of **7** will be pushed to **stack[2]**:

![](img/b73eda63-524a-41cb-8894-2edcefc99280.png)

Figure 3.25

After separating the number into individual digits, we need to pop each digit from the stack one by one and multiply each one by **10** raised to a power. The power will be **0** for the topmost digit on the stack and will increment by 1 after every `pop` operation. The digit that will be popped from the stack will be the one indicated to by the top pointer. The value of **top** is **2**, so the digit on **stack[2]** is popped out and is multiplied by **10** raised to power of **0**:

![](img/6bebbc2a-81ad-44d2-8aa6-00aa8fc65571.png)

Figure 3.26

After every `pop` operation, the value of **top** is decremented by 1 and the value of the power is incremented by 1\. The next digit that will be popped out from the stack is the one on **stack[1]**. That is, **3** will be popped out and multiplied by **10** raised to the power of **1**. Thereafter, the value of **top** will be decremented by 1, that is, the value of **top** will become **0**, and the value of the power will be incremented by 1, that is, the value of the power that was **1** will be incremented to **2**. The digit on **stack[0] **will be popped out and multiplied by **10** raised to the power of **2**:

![](img/55a2916d-a8ca-41c2-9a81-8c16ab812ee1.png)

Figure 3.27

All the digits that are multiplied by **10** raised to the respective power are then summed. Because the result of the computation matches the original number, **737** is a palindrome.

Let's use GCC to compile the `findpalindrome.c` program, as shown in the following statement:

```cpp
D:\CBook>gcc findpalindrome.c -o findpalindrome
```

Let's check whether `123` is a palindrome number:

```cpp
D:\CBook>./findpalindrome
Enter a number 123
123 is not a palindrome number
```

Let's check whether `737` is a palindrome number:

```cpp
 D:\CBook>./findpalindrome
Enter a number 737
737 is a palindrome number
```

Voilà! We've successfully determined whether a number was a palindrome.